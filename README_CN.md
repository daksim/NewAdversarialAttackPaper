# Latest Adversarial Attack Papers
**update at 2022-08-29 06:31:21**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Semantic Preserving Adversarial Attack Generation with Autoencoder and Genetic Algorithm**

基于自动编码和遗传算法的语义保持敌意攻击生成 cs.LG

8 pages conference paper, accepted for publication in IEEE GLOBECOM  2022

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12230v1)

**Authors**: Xinyi Wang, Simon Yusuf Enoch, Dong Seong Kim

**Abstracts**: Widely used deep learning models are found to have poor robustness. Little noises can fool state-of-the-art models into making incorrect predictions. While there is a great deal of high-performance attack generation methods, most of them directly add perturbations to original data and measure them using L_p norms; this can break the major structure of data, thus, creating invalid attacks. In this paper, we propose a black-box attack, which, instead of modifying original data, modifies latent features of data extracted by an autoencoder; then, we measure noises in semantic space to protect the semantics of data. We trained autoencoders on MNIST and CIFAR-10 datasets and found optimal adversarial perturbations using a genetic algorithm. Our approach achieved a 100% attack success rate on the first 100 data of MNIST and CIFAR-10 datasets with less perturbation than FGSM.

摘要: 广泛使用的深度学习模型具有较差的稳健性。微小的噪音可以愚弄最先进的模型做出错误的预测。虽然有很多高性能的攻击生成方法，但它们大多直接对原始数据添加扰动，并使用L_p范数进行度量，这会破坏数据的主要结构，从而产生无效攻击。本文提出了一种黑盒攻击，它不修改原始数据，而是修改由自动编码器提取的数据的潜在特征，然后在语义空间中测量噪声来保护数据的语义。我们在MNIST和CIFAR-10数据集上训练自动编码器，并使用遗传算法找到最优的对抗性扰动。我们的方法在MNIST和CIFAR-10数据集的前100个数据上取得了100%的攻击成功率，并且比FGSM具有更小的扰动。



## **2. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12216v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **3. Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study**

非结构化网络威胁情报自动测绘的实验研究 cs.CR

2022 IEEE 33rd International Symposium on Software Reliability  Engineering (ISSRE)

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12144v1)

**Authors**: Vittorio Orbinato, Mariarosaria Barbaraci, Roberto Natella, Domenico Cotroneo

**Abstracts**: Proactive approaches to security, such as adversary emulation, leverage information about threat actors and their techniques (Cyber Threat Intelligence, CTI). However, most CTI still comes in unstructured forms (i.e., natural language), such as incident reports and leaked documents. To support proactive security efforts, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.

摘要: 主动的安全方法，如对手模拟，利用有关威胁参与者及其技术的信息(网络威胁情报，CTI)。然而，大多数CTI仍然是非结构化的形式(即自然语言)，例如事件报告和泄露的文件。为了支持主动安全工作，我们提出了一项使用机器学习(ML)将非结构化CTI自动分类为攻击技术的实验研究。我们为CTI分析提供了两个新的数据集，并评估了几个ML模型，包括传统的和基于深度学习的模型。我们提供了几个经验教训，关于ML如何在这项任务中执行，哪些分类器在哪些条件下表现最好，哪些是分类错误的主要原因，以及CTI分析未来的挑战。



## **4. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

ECG-ATK-GAN：使用条件生成对抗网络对ECG的对抗攻击的稳健性 eess.SP

Accepted to MICCAI2022 Applications of Medical AI (AMAI) Workshop

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2110.09983v3)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce the first novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and new blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare them with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.

摘要: 从心电中自动检测心律失常需要一个健壮和可信的系统，在电子干扰下保持高精度。许多机器学习方法在区分心律失常和心电信号方面已经达到了人类的水平。然而，这些体系结构容易受到敌意攻击，这些攻击可能会降低模型的准确性，从而导致心电信号的误分类。对抗性攻击是注入到原始数据中的小的精心设计的扰动，它显示了信号的不分布转移，以错误地分类正确的类别。因此，出现了对虚假住院和滥用这些扰动的保险欺诈的安全担忧。为了缓解这一问题，我们引入了第一个新的条件生成对抗网络(GAN)，它对对手攻击的心电信号具有鲁棒性，并保持了较高的准确率。我们的体系结构集成了一个新的类别加权目标函数来识别对抗性扰动，以及新的块来识别和组合学习过程中信号的非分布变化，以准确地分类各种心律失常类型。此外，我们在六种不同的白盒和黑盒攻击上测试了我们的体系结构，并将它们与最近提出的其他心律失常分类模型在两个公开可用的心电心律失常数据集上进行了比较。实验证明，该模型对心律失常的分类具有较强的鲁棒性，分类准确率较高。



## **5. A Perturbation Resistant Transformation and Classification System for Deep Neural Networks**

一种抗扰动的深度神经网络变换与分类系统 cs.CV

12 pages, 4 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.11839v1)

**Authors**: Nathaniel Dean, Dilip Sarkar

**Abstracts**: Deep convolutional neural networks accurately classify a diverse range of natural images, but may be easily deceived when designed, imperceptible perturbations are embedded in the images. In this paper, we design a multi-pronged training, input transformation, and image ensemble system that is attack agnostic and not easily estimated. Our system incorporates two novel features. The first is a transformation layer that computes feature level polynomial kernels from class-level training data samples and iteratively updates input image copies at inference time based on their feature kernel differences to create an ensemble of transformed inputs. The second is a classification system that incorporates the prediction of the undefended network with a hard vote on the ensemble of filtered images. Our evaluations on the CIFAR10 dataset show our system improves the robustness of an undefended network against a variety of bounded and unbounded white-box attacks under different distance metrics, while sacrificing little accuracy on clean images. Against adaptive full-knowledge attackers creating end-to-end attacks, our system successfully augments the existing robustness of adversarially trained networks, for which our methods are most effectively applied.

摘要: 深层卷积神经网络可以准确地对多种自然图像进行分类，但在设计时很容易被欺骗，图像中嵌入了难以察觉的扰动。在本文中，我们设计了一个多管齐下的训练、输入变换和图像集成系统，该系统与攻击无关，不易估计。我们的系统有两个新颖的特点。第一个是变换层，它从类级训练数据样本计算特征级多项式核，并在推理时基于它们的特征核差异迭代地更新输入图像副本，以创建变换输入的集成。第二种是一种分类系统，它结合了对无防御网络的预测和对过滤图像集合的硬投票。我们在CIFAR10数据集上的评估表明，我们的系统提高了无防御网络在不同距离度量下对各种有界和无界白盒攻击的健壮性，而对干净图像的准确性几乎没有牺牲。针对自适应全知识攻击者制造的端到端攻击，我们的系统成功地增强了对手训练网络的现有健壮性，我们的方法在这些网络中得到了最有效的应用。



## **6. A New Kind of Adversarial Example**

一种新的对抗性例证 cs.CV

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.02430v2)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.

摘要: 几乎所有的对抗性攻击都是为了给图像添加一个难以察觉的扰动，以愚弄模型。在这里，我们考虑的是相反的情况，即可以愚弄人类但不能愚弄模型的对抗性例子。一个足够大和可感知的扰动被添加到图像中，使得模型保持其原始决定，而如果被迫做出决定(或者选择根本不决定)，人类很可能会犯错误。现有的有针对性的攻击可以重新制定，以合成这种对抗性的例子。我们提出的名为NKE的攻击在本质上类似于愚弄图像，但由于它使用了梯度下降而不是进化算法，因此效率更高。它还为敌方脆弱性问题提供了一个新的统一视角。在MNIST和CIFAR-10数据集上的实验结果表明，我们的攻击在欺骗深度神经网络方面是相当有效的。代码可在https://github.com/aliborji/NKE.上找到



## **7. Attacking Neural Binary Function Detection**

攻击神经二进制函数检测 cs.CR

18 pages

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11667v1)

**Authors**: Joshua Bundt, Michael Davinroy, Ioannis Agadakos, Alina Oprea, William Robertson

**Abstracts**: Binary analyses based on deep neural networks (DNNs), or neural binary analyses (NBAs), have become a hotly researched topic in recent years. DNNs have been wildly successful at pushing the performance and accuracy envelopes in the natural language and image processing domains. Thus, DNNs are highly promising for solving binary analysis problems that are typically hard due to a lack of complete information resulting from the lossy compilation process. Despite this promise, it is unclear that the prevailing strategy of repurposing embeddings and model architectures originally developed for other problem domains is sound given the adversarial contexts under which binary analysis often operates.   In this paper, we empirically demonstrate that the current state of the art in neural function boundary detection is vulnerable to both inadvertent and deliberate adversarial attacks. We proceed from the insight that current generation NBAs are built upon embeddings and model architectures intended to solve syntactic problems. We devise a simple, reproducible, and scalable black-box methodology for exploring the space of inadvertent attacks - instruction sequences that could be emitted by common compiler toolchains and configurations - that exploits this syntactic design focus. We then show that these inadvertent misclassifications can be exploited by an attacker, serving as the basis for a highly effective black-box adversarial example generation process. We evaluate this methodology against two state-of-the-art neural function boundary detectors: XDA and DeepDi. We conclude with an analysis of the evaluation data and recommendations for how future research might avoid succumbing to similar attacks.

摘要: 基于深度神经网络(DNN)或神经二进制分析(NBAs)的二进制分析是近年来研究的热点。在自然语言和图像处理领域，DNN在提高性能和准确率方面取得了巨大的成功。因此，DNN在解决二进制分析问题方面非常有希望，这些问题通常很难解决，因为有损编译过程导致缺乏完整的信息。尽管有这样的承诺，但考虑到二元分析经常在敌对的环境下运行，目前尚不清楚重新调整最初为其他问题领域开发的嵌入和模型体系结构的用途的流行策略是否合理。在这篇文章中，我们经验地证明，神经功能边界检测的当前技术水平容易受到无意和故意的敌意攻击。我们的出发点是，当前一代的NBA是建立在旨在解决语法问题的嵌入和模型体系结构之上的。我们设计了一种简单、可重复和可扩展的黑盒方法，用于探索意外攻击的空间-可能由常见编译器工具链和配置发出的指令序列-利用了这一语法设计重点。然后，我们展示了这些无意的错误分类可以被攻击者利用，作为高效的黑盒对抗性示例生成过程的基础。我们用两种最先进的神经功能边界检测器：XDA和DeepDi对该方法进行了评估。最后，我们对评估数据进行了分析，并就未来的研究如何避免屈服于类似的攻击提出了建议。



## **8. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2103.09151v3)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As the research in deep neural networks advances, deep convolutional networks become feasible for automated driving tasks. There is an emerging trend of employing end-to-end models in the automation of driving tasks. However, previous research unveils that deep neural networks are vulnerable to adversarial attacks in classification tasks. While for regression tasks such as autonomous driving, the effect of these attacks remains rarely explored. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving model takes an image as input and outputs the steering angle. Our attacks can manipulate the behavior of the autonomous driving system only by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. This research aims to raise concerns over applications of end-to-end models in safety-critical systems.

摘要: 随着深度神经网络研究的深入，深度卷积网络在自动驾驶任务中变得可行。在驾驶任务的自动化中使用端到端模型是一种新兴的趋势。然而，以往的研究表明，深度神经网络在分类任务中容易受到敌意攻击。而对于自动驾驶等回归任务，这些攻击的影响仍然很少被研究。在本研究中，我们设计了两种针对端到端自动驾驶系统的白盒针对性攻击。驾驶模型以图像为输入，输出转向角。我们的攻击只能通过干扰输入图像来操纵自动驾驶系统的行为。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。这项研究旨在引起人们对端到端模型在安全关键系统中的应用的关注。



## **9. Unrestricted Black-box Adversarial Attack Using GAN with Limited Queries**

基于有限查询GAN的无限制黑盒对抗性攻击 cs.CV

Accepted to the ECCV 2022 Workshop on Adversarial Robustness in the  Real World

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11613v1)

**Authors**: Dongbin Na, Sangwoo Ji, Jong Kim

**Abstracts**: Adversarial examples are inputs intentionally generated for fooling a deep neural network. Recent studies have proposed unrestricted adversarial attacks that are not norm-constrained. However, the previous unrestricted attack methods still have limitations to fool real-world applications in a black-box setting. In this paper, we present a novel method for generating unrestricted adversarial examples using GAN where an attacker can only access the top-1 final decision of a classification model. Our method, Latent-HSJA, efficiently leverages the advantages of a decision-based attack in the latent space and successfully manipulates the latent vectors for fooling the classification model.   With extensive experiments, we demonstrate that our proposed method is efficient in evaluating the robustness of classification models with limited queries in a black-box setting. First, we demonstrate that our targeted attack method is query-efficient to produce unrestricted adversarial examples for a facial identity recognition model that contains 307 identities. Then, we demonstrate that the proposed method can also successfully attack a real-world celebrity recognition service.

摘要: 对抗性例子是为愚弄深度神经网络而故意生成的输入。最近的研究提出了不受规范约束的无限制对抗性攻击。然而，以前的不受限制的攻击方法仍然有局限性，无法在黑盒设置中愚弄现实世界的应用程序。在本文中，我们提出了一种利用GAN生成无限制敌意实例的新方法，其中攻击者只能访问分类模型的TOP-1最终决策。该方法有效地利用了基于决策的攻击在潜在空间中的优势，并成功地操纵了潜在向量来欺骗分类模型。通过大量的实验，我们证明了我们提出的方法在黑盒环境下评估有限查询的分类模型的稳健性是有效的。首先，我们证明了我们的定向攻击方法是查询高效的，可以为包含307个身份的面部身份识别模型生成不受限制的对抗性示例。然后，我们证明了所提出的方法也可以成功地攻击真实世界的名人识别服务。



## **10. Robustness of the Tangle 2.0 Consensus**

Tangle 2.0共识的健壮性 cs.DC

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.08254v2)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.

摘要: 本文研究了Tange2.0一致性协议在拜占庭环境下的性能。我们使用了一个基于代理的仿真模型，该模型结合了Tangel2.0共识协议的主要特征。实验结果表明，Tangel2.0协议对诱饵切换攻击具有较强的鲁棒性，达到了敌手33%投票权重的理论上限。我们进一步证明了Tange2.0中的普通硬币机制对于抵抗强大的对手是必要的。实验结果表明，该协议在典型场景下可以达到1s左右的确认时间，且无冲突事务的确认时间不受冲突的影响。



## **11. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

LPF-Defense：基于频率分析的三维对抗性防御 cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2202.11287v2)

**Authors**: Hanieh Naderi, Kimia Noorbakhsh, Arian Etemadi, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.

摘要: 虽然三维点云分类最近在不同的应用场景中得到了广泛的部署，但它仍然非常容易受到对抗性攻击。这增加了在面对对手攻击时对3D模型进行稳健训练的重要性。基于对现有对抗性攻击性能的分析，在输入数据的中高频成分中发现了更多的对抗性扰动。因此，通过抑制训练阶段的高频内容，提高了模型对敌意样本的稳健性。实验表明，该防御方法降低了对PointNet、PointNet++、和DGCNN模型的6次攻击的成功率。特别是，与现有方法相比，Drop100攻击的分类准确率平均提高了3.8%，Drop200攻击的分类准确率平均提高了4.26%。与其他可用的方法相比，该方法还提高了原始数据集上的模型精度。



## **12. Trace and Detect Adversarial Attacks on CNNs using Feature Response Maps**

使用特征响应映射跟踪和检测对CNN的敌意攻击 cs.CV

13 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11436v1)

**Authors**: Mohammadreza Amirian, Friedhelm Schwenker, Thilo Stadelmann

**Abstracts**: The existence of adversarial attacks on convolutional neural networks (CNN) questions the fitness of such models for serious applications. The attacks manipulate an input image such that misclassification is evoked while still looking normal to a human observer -- they are thus not easily detectable. In a different context, backpropagated activations of CNN hidden layers -- "feature responses" to a given input -- have been helpful to visualize for a human "debugger" what the CNN "looks at" while computing its output. In this work, we propose a novel detection method for adversarial examples to prevent attacks. We do so by tracking adversarial perturbations in feature responses, allowing for automatic detection using average local spatial entropy. The method does not alter the original network architecture and is fully human-interpretable. Experiments confirm the validity of our approach for state-of-the-art attacks on large-scale models trained on ImageNet.

摘要: 对卷积神经网络(CNN)的敌意攻击的存在质疑了这种模型在严肃应用中的适用性。这些攻击操作输入图像，从而在人类观察者看来仍然正常的情况下引发错误分类--因此它们不容易被检测到。在另一种情况下，对CNN隐藏层的反向传播激活--对给定输入的“特征响应”--有助于人类“调试者”在计算其输出时可视化CNN“所看到的”。在这项工作中，我们提出了一种新的对抗性实例检测方法来防止攻击。我们通过跟踪特征响应中的对抗性扰动来实现这一点，允许使用平均局部空间熵进行自动检测。该方法不改变原有的网络体系结构，完全是人类可理解的。实验证实了该方法对基于ImageNet训练的大规模模型进行最先进攻击的有效性。



## **13. Towards an Awareness of Time Series Anomaly Detection Models' Adversarial Vulnerability**

认识时间序列异常检测模型的对抗性漏洞 cs.LG

Part of Proceedings of the 31st ACM International Conference on  Information and Knowledge Management (CIKM '22)

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11264v1)

**Authors**: Shahroz Tariq, Binh M. Le, Simon S. Woo

**Abstracts**: Time series anomaly detection is extensively studied in statistics, economics, and computer science. Over the years, numerous methods have been proposed for time series anomaly detection using deep learning-based methods. Many of these methods demonstrate state-of-the-art performance on benchmark datasets, giving the false impression that these systems are robust and deployable in many practical and industrial real-world scenarios. In this paper, we demonstrate that the performance of state-of-the-art anomaly detection methods is degraded substantially by adding only small adversarial perturbations to the sensor data. We use different scoring metrics such as prediction errors, anomaly, and classification scores over several public and private datasets ranging from aerospace applications, server machines, to cyber-physical systems in power plants. Under well-known adversarial attacks from Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) methods, we demonstrate that state-of-the-art deep neural networks (DNNs) and graph neural networks (GNNs) methods, which claim to be robust against anomalies and have been possibly integrated in real-life systems, have their performance drop to as low as 0%. To the best of our understanding, we demonstrate, for the first time, the vulnerabilities of anomaly detection systems against adversarial attacks. The overarching goal of this research is to raise awareness towards the adversarial vulnerabilities of time series anomaly detectors.

摘要: 时间序列异常检测在统计学、经济学和计算机科学中都有广泛的研究。多年来，已经提出了许多基于深度学习的时间序列异常检测方法。其中许多方法在基准数据集上展示了最先进的性能，给人一种错误的印象，即这些系统在许多实际和工业真实世界的场景中都是健壮的和可部署的。在这篇文章中，我们证明了最新的异常检测方法的性能在很大程度上由于只向传感器数据添加小的对抗性扰动而降低。我们在从航空航天应用程序、服务器机器到发电厂的网络物理系统等多个公共和私有数据集上使用不同的评分指标，例如预测误差、异常和分类分数。在已知的快速梯度符号方法(FGSM)和投影梯度下降(PGD)方法的敌意攻击下，我们证明了最新的深度神经网络(DNNS)和图神经网络(GNNS)方法的性能下降到了0%，这两种方法声称对异常具有健壮性，并可能集成到现实系统中。据我们所知，我们第一次展示了异常检测系统对对手攻击的脆弱性。这项研究的首要目标是提高人们对时间序列异常检测器的对抗性脆弱性的认识。



## **14. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

ObfuNAS：一种基于神经结构搜索的DNN混淆方法 cs.CR

9 pages

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.08569v2)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.

摘要: 恶意体系结构提取已经成为深度神经网络(DNN)安全的一个重要问题。作为防御，体系结构混淆被提出将受害者DNN重新映射到不同的体系结构。尽管如此，我们观察到，只要提取一个混淆的DNN体系结构，攻击者仍然可以高性能(例如，准确性)重新训练替代模型，使得混淆技术无效。为了缓解这一未被充分挖掘的漏洞，我们提出了ObfuNAS，它将DNN体系结构的混淆转化为神经体系结构搜索(NAS)问题。ObfuNAS结合了函数保留混淆策略，确保了混淆后的DNN架构只能达到比受害者更低的准确率。我们使用NAS-BENCH-101和NAS-BENCH-301等开源架构数据集验证了ObfuNAS的性能。实验结果表明，ObfuNAS能够在给定的FLOPS约束下成功地找到受害者模型的最优掩码，使得仅需0.14倍FLOPS开销的攻击者的推理准确率降低2.6%。代码可从以下网址获得：https://github.com/Tongzhou0101/ObfuNAS.



## **15. Auditing Membership Leakages of Multi-Exit Networks**

审计多出口网络的成员泄漏 cs.CR

Accepted by CCS 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.11180v1)

**Authors**: Zheng Li, Yiyong Liu, Xinlei He, Ning Yu, Michael Backes, Yang Zhang

**Abstracts**: Relying on the fact that not all inputs require the same amount of computation to yield a confident prediction, multi-exit networks are gaining attention as a prominent approach for pushing the limits of efficient deployment. Multi-exit networks endow a backbone model with early exits, allowing to obtain predictions at intermediate layers of the model and thus save computation time and/or energy. However, current various designs of multi-exit networks are only considered to achieve the best trade-off between resource usage efficiency and prediction accuracy, the privacy risks stemming from them have never been explored. This prompts the need for a comprehensive investigation of privacy risks in multi-exit networks.   In this paper, we perform the first privacy analysis of multi-exit networks through the lens of membership leakages. In particular, we first leverage the existing attack methodologies to quantify the multi-exit networks' vulnerability to membership leakages. Our experimental results show that multi-exit networks are less vulnerable to membership leakages and the exit (number and depth) attached to the backbone model is highly correlated with the attack performance. Furthermore, we propose a hybrid attack that exploits the exit information to improve the performance of existing attacks. We evaluate membership leakage threat caused by our hybrid attack under three different adversarial setups, ultimately arriving at a model-free and data-free adversary. These results clearly demonstrate that our hybrid attacks are very broadly applicable, thereby the corresponding risks are much more severe than shown by existing membership inference attacks. We further present a defense mechanism called TimeGuard specifically for multi-exit networks and show that TimeGuard mitigates the newly proposed attacks perfectly.

摘要: 由于并非所有的输入都需要相同的计算量才能得出可信的预测，多出口网络作为一种突破有效部署极限的重要方法正受到人们的关注。多出口网络为主干模型提供了早期出口，允许在模型的中间层获得预测，从而节省计算时间和/或能量。然而，目前各种设计的多出口网络只考虑在资源使用效率和预测精度之间实现最佳折衷，并没有探讨它们所带来的隐私风险。这促使需要对多出口网络中的隐私风险进行全面调查。本文首次从成员泄漏的角度对多出口网络进行了隐私分析。特别是，我们首先利用现有的攻击方法来量化多出口网络对成员泄漏的脆弱性。我们的实验结果表明，多出口网络不太容易受到成员泄漏的影响，并且连接到主干模型的出口(数量和深度)与攻击性能高度相关。此外，我们还提出了一种利用出口信息的混合攻击来提高现有攻击的性能。我们评估了我们的混合攻击在三种不同的对手设置下所造成的成员泄漏威胁，最终得出了一个无模型、无数据的对手。这些结果清楚地表明，我们的混合攻击具有非常广泛的适用性，因此相应的风险比现有的成员推理攻击要严重得多。我们进一步提出了一种专门针对多出口网络的防御机制TimeGuard，并证明了TimeGuard可以很好地缓解新提出的攻击。



## **16. Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**

说话人自动确认对抗模型中的对抗性说话人提取 cs.SD

Accepted by ISCA SPSC 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2203.17031v5)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect ASV systems from spoof attacks and prevent resulting personal information leakage in Automatic Speaker Verification (ASV) system. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems, confining the model size under a limitation. To better trade off the CM model sizes and performance, we proposed an adversarial speaker distillation method, which is an improved version of knowledge distillation method combined with generalized end-to-end (GE2E) pre-training and adversarial fine-tuning. In the evaluation phase of the ASVspoof 2021 Logical Access task, our proposed adversarial speaker distillation ResNetSE (ASD-ResNetSE) model reaches 0.2695 min t-DCF and 3.54\% EER. ASD-ResNetSE only used 22.5\% of parameters and 19.4\% of multiply and accumulate operands of ResNetSE model.

摘要: 在自动说话人确认(ASV)系统中，为了保护ASV系统免受欺骗攻击，并防止由此导致的个人信息泄露，提出了对策(CM)模型。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备具有更有限的计算资源和存储空间，从而将模型大小限制在一定范围内。为了更好地权衡CM模型的规模和性能，我们提出了一种对抗性说话人蒸馏方法，它是一种改进的知识蒸馏方法，结合了广义端到端(GE2E)预训练和对抗性微调。在ASVspoof 2021逻辑访问任务的评估阶段，我们提出的对抗性说话人蒸馏ResNetSE(ASD-ResNetSE)模型达到了0.2695分钟的t-DCF和3.54EER。ASD-ResNetSE只使用了ResNetSE模型的22.5个参数和19.4个乘法和累加操作数。



## **17. Privacy Enhancement for Cloud-Based Few-Shot Learning**

增强基于云的极少机会学习的隐私 cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.07864v2)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.

摘要: 对于精确的模型，少镜头学习需要较少的数据，在许多应用领域都表现出了健壮性和通用性。然而，在不受信任的环境中部署极少的模型可能会引起隐私问题，例如，可能会破坏用户提供的数据的隐私的攻击或对手。通过建立一种新的隐私保护嵌入空间来保护数据隐私并保持模型的准确性，研究了在不可信环境(如云)下的少机会学习的隐私增强问题。研究了模糊、像素化、高斯噪声、差分隐私像素化等图像隐私保护方法对少镜头图像分类的影响，提出了一种通过联合损失学习隐私保护表示的方法。实证结果表明，对于隐私增强型少镜头学习，隐私性能与性能之间的权衡是如何协商的。



## **18. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

跨域实时目标检测网络研究综述 cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10895v1)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstracts**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.

摘要: 基于深度神经网络的目标检测器正在不断发展，并在许多应用中使用，每个应用都有其自己的一组要求。安全关键型应用程序需要高准确性和可靠性，而低延迟任务则需要资源和能效高的网络。实时检测器在高影响的现实世界应用中是必不可少的，不断被提出，但它们过分强调精度和速度的提高，而忽略了其他功能，如通用性、健壮性、资源和能源效率。现有网络没有参考基准，也没有设计新网络的标准评估指南，这导致比较不明确和不一致。因此，我们在广泛的数据集上对多个实时检测器(锚点、关键点和变压器)进行了全面的研究，并报告了一组广泛的指标的结果。我们还研究了图像大小、锚点维度、置信度阈值和架构层等变量对整体性能的影响。我们分析了检测网络对分布偏移、自然破坏和敌意攻击的稳健性。此外，我们还提供了校准分析，以衡量预测的可靠性。最后，为了突出现实世界的影响，我们进行了两个独特的案例研究，分别是自动驾驶和医疗保健应用。为了进一步衡量网络在关键实时应用中的能力，我们报告了在边缘设备上部署检测网络后的性能。我们广泛的实证研究可以为工业界在现有网络上做出明智的选择提供指导。我们还希望激励研究界在网络的设计和评估方面朝着一个新的方向前进，专注于更大和更全面的概览，以产生深远的影响。



## **19. Transferability Ranking of Adversarial Examples**

对抗性例证的可转移性排名 cs.LG

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10878v1)

**Authors**: Mosh Levy, Yuval Elovici, Yisroel Mirsky

**Abstracts**: Adversarial examples can be used to maliciously and covertly change a model's prediction. It is known that an adversarial example designed for one model can transfer to other models as well. This poses a major threat because it means that attackers can target systems in a blackbox manner.   In the domain of transferability, researchers have proposed ways to make attacks more transferable and to make models more robust to transferred examples. However, to the best of our knowledge, there are no works which propose a means for ranking the transferability of an adversarial example in the perspective of a blackbox attacker. This is an important task because an attacker is likely to use only a select set of examples, and therefore will want to select the samples which are most likely to transfer.   In this paper we suggest a method for ranking the transferability of adversarial examples without access to the victim's model. To accomplish this, we define and estimate the expected transferability of a sample given limited information about the victim. We also explore practical scenarios: where the adversary can select the best sample to attack and where the adversary must use a specific sample but can choose different perturbations. Through our experiments, we found that our ranking method can increase an attacker's success rate by up to 80% compared to the baseline (random selection without ranking).

摘要: 敌意的例子可能被用来恶意和秘密地改变模型的预测。众所周知，为一种模型设计的对抗性例子也可以转移到其他模型上。这构成了一个重大威胁，因为这意味着攻击者可以以黑盒方式将系统作为目标。在可转移性领域，研究人员已经提出了一些方法来使攻击更具可转移性，并使模型对已转移的示例更健壮。然而，就我们所知，目前还没有文献提出从黑盒攻击者的角度对对抗性例子的可转移性进行排序的方法。这是一项重要的任务，因为攻击者可能只使用一组选定的示例，因此希望选择最有可能传输的样本。在这篇文章中，我们提出了一种在不使用受害者模型的情况下对对抗性例子的可转移性进行排序的方法。为了实现这一点，我们定义并估计了给定关于受害者的有限信息的样本的预期可转移性。我们还探索了实际场景：其中对手可以选择最佳样本进行攻击，以及对手必须使用特定样本但可以选择不同的扰动。通过实验，我们发现我们的排序方法可以使攻击者的成功率比基线(随机选择而不排序)提高高达80%。



## **20. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

具有最优码率的抗平均攻击和对抗噪声的完全可追溯性多媒体指纹编码 cs.IT

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2108.09015v4)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.

摘要: 在本文中，我们考虑了完全可追踪性多媒体指纹码，它能抵抗平均攻击和对抗噪声。最近的研究表明，对于任意线性攻击的情况，不存在这样的码。然而，对于平均攻击的情况，Egorova等人于2020年构造了抵抗恒定对抗性噪声的指数基数完全可追溯性多媒体指纹码。我们继续这项工作，并提供了这些码率的一个改进的下界。



## **21. Evaluating Machine Unlearning via Epistemic Uncertainty**

基于认知不确定性的机器遗忘评估 cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10836v1)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.

摘要: 最近，人们对机器遗忘的兴趣与日俱增，主要是因为法律要求，如一般数据保护法规(GDPR)和加州消费者隐私法。因此，人们提出了多种方法来消除特定目标数据点对训练模型的影响。然而，在评估遗忘的成功时，目前的方法要么使用对抗性攻击，要么将结果与最优解决方案进行比较，后者通常包括从头开始的再培训。我们认为，这两种方式在实践中都是不够的。在这项工作中，我们提出了一种基于认知不确定性的机器遗忘算法评价指标。据我们所知，这是对机器遗忘的一般评估指标的第一次定义。



## **22. UKP-SQuARE v2 Explainability and Adversarial Attacks for Trustworthy QA**

UKP-Square v2可解析性和可信QA的对抗性攻击 cs.CL

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.09316v2)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstracts**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.

摘要: 问答(QA)系统越来越多地部署在支持现实世界决策的应用程序中。然而，最先进的模型依赖于深度神经网络，这很难被人类解释。本质上可解释的模型或事后可解释的方法可以帮助用户理解模型如何达到其预测，如果成功，则增加他们对系统的信任。此外，研究人员可以利用这些洞察力来开发更准确、更少偏见的新方法。在本文中，我们引入了Square的新版本Square v2，以提供基于显著图和基于图的解释等方法的模型比较的可解释性基础设施。虽然显著图有助于检查每个输入标记对于模型预测的重要性，但来自外部知识图的基于图形的解释使用户能够验证模型预测背后的推理。此外，我们还提供了多个对抗性攻击来比较QA模型的健壮性。通过这些可解释性方法和对抗性攻击，我们的目标是简化可信QA模型的研究。Square在https://square.ukp-lab.de.上可用



## **23. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2009.04131v7)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **24. Adversarial Vulnerability of Temporal Feature Networks for Object Detection**

用于目标检测的时态特征网络的对抗脆弱性 cs.CV

Accepted for publication at ECCV 2022 SAIAD workshop

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10773v1)

**Authors**: Svetlana Pavlitskaya, Nikolai Polley, Michael Weber, J. Marius Zöllner

**Abstracts**: Taking into account information across the temporal domain helps to improve environment perception in autonomous driving. However, it has not been studied so far whether temporally fused neural networks are vulnerable to deliberately generated perturbations, i.e. adversarial attacks, or whether temporal history is an inherent defense against them. In this work, we study whether temporal feature networks for object detection are vulnerable to universal adversarial attacks. We evaluate attacks of two types: imperceptible noise for the whole image and locally-bound adversarial patch. In both cases, perturbations are generated in a white-box manner using PGD. Our experiments confirm, that attacking even a portion of a temporal input suffices to fool the network. We visually assess generated perturbations to gain insights into the functioning of attacks. To enhance the robustness, we apply adversarial training using 5-PGD. Our experiments on KITTI and nuScenes datasets demonstrate, that a model robustified via K-PGD is able to withstand the studied attacks while keeping the mAP-based performance comparable to that of an unattacked model.

摘要: 考虑跨时间域的信息有助于改善自动驾驶中的环境感知。然而，到目前为止，时间融合的神经网络是否容易受到故意产生的扰动，即对抗性攻击，或者时间历史是否是对它们的固有防御，还没有被研究过。在这项工作中，我们研究了用于目标检测的时态特征网络是否容易受到普遍的敌意攻击。我们评估了两种类型的攻击：整个图像的不可感知噪声攻击和局部绑定的敌意补丁攻击。在这两种情况下，使用PGD以白盒方式产生扰动。我们的实验证实，攻击即使是时间输入的一部分，也足以愚弄网络。我们直观地评估产生的扰动，以深入了解攻击的功能。为了增强算法的稳健性，我们使用了5-PGD进行对抗性训练。我们在Kitti和nuScenes数据集上的实验表明，通过K-PGD健壮的模型能够抵抗所研究的攻击，同时保持基于地图的性能与未受攻击的模型相当。



## **25. MALICE: Manipulation Attacks on Learned Image ComprEssion**

恶意：对学习图像压缩的操纵攻击 cs.CV

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.13253v2)

**Authors**: Kang Liu, Di Wu, Yiru Wang, Dan Feng, Benjamin Tan, Siddharth Garg

**Abstracts**: Deep learning techniques have shown promising results in image compression, with competitive bitrate and image reconstruction quality from compressed latent. However, while image compression has progressed towards a higher peak signal-to-noise ratio (PSNR) and fewer bits per pixel (bpp), their robustness to adversarial images has never received deliberation. In this work, we, for the first time, investigate the robustness of image compression systems where imperceptible perturbation of input images can precipitate a significant increase in the bitrate of their compressed latent. To characterize the robustness of state-of-the-art learned image compression, we mount white-box and black-box attacks. Our white-box attack employs fast gradient sign method on the entropy estimation of the bitstream as its bitrate approximation. We propose DCT-Net simulating JPEG compression with architectural simplicity and lightweight training as the substitute in the black-box attack and enable fast adversarial transferability. Our results on six image compression models, each with six different bitrate qualities (thirty-six models in total), show that they are surprisingly fragile, where the white-box attack achieves up to 56.326x and black-box 1.947x bpp change. To improve robustness, we propose a novel compression architecture factorAtn which incorporates attention modules and a basic factorized entropy model, resulting in a promising trade-off between the rate-distortion performance and robustness to adversarial attacks that surpasses existing learned image compressors.

摘要: 深度学习技术在图像压缩方面取得了很好的效果，压缩后的潜伏期具有较高的比特率和图像重建质量。然而，虽然图像压缩已经朝着更高的峰值信噪比(PSNR)和更少的每像素位(BPP)发展，但它们对敌意图像的稳健性从未得到深思熟虑。在这项工作中，我们首次研究了图像压缩系统的稳健性，在这种情况下，输入图像的不可察觉的扰动可以导致其压缩潜伏期的比特率显著增加。为了表征最先进的学习图像压缩的稳健性，我们安装了白盒和黑盒攻击。我们的白盒攻击使用快速梯度符号方法对码流进行熵估计作为其码率近似。我们提出了一种结构简单、训练轻量级的模拟JPEG压缩的DCT-Net作为黑盒攻击的替代方案，并实现了快速的对抗性转移。我们在6种不同码率质量(总共36种)的6种图像压缩模型上的结果表明，它们令人惊讶地脆弱，其中白盒攻击达到56.326倍，黑盒攻击达到1.947倍的BPP变化。为了提高鲁棒性，我们提出了一种新的压缩体系结构factorAtn，它结合了注意力模块和基本的因子化熵模型，在率失真性能和对敌意攻击的稳健性之间取得了很好的权衡，其性能超过了现有的学习图像压缩器。



## **26. RAB: Provable Robustness Against Backdoor Attacks**

RAB：针对后门攻击的可证明的健壮性 cs.LG

IEEE Symposium on Security and Privacy 2023

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2003.08904v7)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks (DNNs) are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, the provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We prove the robustness bound for machine learning models trained with RAB and prove that our robustness bound is tight. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm that eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning (ML) models such as DNNs, support vector machines, and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed light on further robust learning strategies against general training time attacks.

摘要: 最近的研究表明，深度神经网络(DNN)容易受到敌意攻击，包括逃避和后门(中毒)攻击。在防御方面，已经在提高对逃避攻击的经验性和可证明的稳健性方面做出了密集的努力；然而，对后门攻击的可证明的稳健性在很大程度上仍然有待探索。在本文中，我们重点证明机器学习模型对一般威胁模型，特别是后门攻击的稳健性。我们首先通过随机平滑技术提供了一个统一的框架，并展示了如何将其实例化来证明对规避和后门攻击的健壮性。然后，我们提出了第一个稳健训练过程RAB，以平滑训练的模型并证明其对后门攻击的稳健性。我们证明了用RAB训练的机器学习模型的稳健界，并证明了我们的稳健界是紧的。此外，我们从理论上证明了对于K近邻分类器等简单模型，可以有效地训练稳健的平滑模型，并提出了一种精确的平滑训练算法，该算法消除了对此类模型从噪声分布中进行采样的需要。在经验上，我们在MNIST、CIFAR-10和ImageNette数据集上对不同的机器学习(ML)模型(如DNN、支持向量机和K-NN模型)进行了全面的实验，并提供了第一个经过验证的针对后门攻击的健壮性基准。此外，我们在垃圾邮件数据库表格数据集上对K-NN模型进行了评估，以展示所提出的精确算法的优势。理论分析和对不同ML模型和数据集的综合评价都为进一步研究针对一般训练时间攻击的稳健学习策略提供了理论依据。



## **27. Hierarchical Perceptual Noise Injection for Social Media Fingerprint Privacy Protection**

用于社交媒体指纹隐私保护的分层感知噪声注入 cs.CV

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10688v1)

**Authors**: Simin Li, Huangxinxin Xu, Jiakai Wang, Aishan Liu, Fazhi He, Xianglong Liu, Dacheng Tao

**Abstracts**: Billions of people are sharing their daily life images on social media every day. However, their biometric information (e.g., fingerprint) could be easily stolen from these images. The threat of fingerprint leakage from social media raises a strong desire for anonymizing shared images while maintaining image qualities, since fingerprints act as a lifelong individual biometric password. To guard the fingerprint leakage, adversarial attack emerges as a solution by adding imperceptible perturbations on images. However, existing works are either weak in black-box transferability or appear unnatural. Motivated by visual perception hierarchy (i.e., high-level perception exploits model-shared semantics that transfer well across models while low-level perception extracts primitive stimulus and will cause high visual sensitivities given suspicious stimulus), we propose FingerSafe, a hierarchical perceptual protective noise injection framework to address the mentioned problems. For black-box transferability, we inject protective noises on fingerprint orientation field to perturb the model-shared high-level semantics (i.e., fingerprint ridges). Considering visual naturalness, we suppress the low-level local contrast stimulus by regularizing the response of Lateral Geniculate Nucleus. Our FingerSafe is the first to provide feasible fingerprint protection in both digital (up to 94.12%) and realistic scenarios (Twitter and Facebook, up to 68.75%). Our code can be found at https://github.com/nlsde-safety-team/FingerSafe.

摘要: 每天都有数十亿人在社交媒体上分享他们的日常生活照片。然而，他们的生物特征信息(例如指纹)很容易从这些图像中被窃取。来自社交媒体的指纹泄露威胁引发了人们对匿名共享图像同时保持图像质量的强烈渴望，因为指纹就像是一个终身的个人生物识别密码。为了防止指纹泄漏，对抗性攻击作为一种解决方案，通过在图像上添加不可感知的扰动来实现。然而，现有的作品要么黑盒可转移性差，要么显得不自然。受视觉感知层次的启发(即高层感知利用模型共享的语义在模型之间很好地传递，而低层感知提取原始刺激并在可疑刺激下导致高视觉敏感度)，我们提出了FingerSafe，一个层次化感知保护性噪声注入框架来解决上述问题。为了保证黑盒的可转移性，我们在指纹方向场上注入保护性噪声，以扰乱模型共享的高层语义(即指纹脊线)。考虑到视觉的自然性，我们通过规则化外侧膝状体反应来抑制低水平的局部对比度刺激。我们的FingerSafe率先在数字(高达94.12%)和现实场景(Twitter和Facebook，高达68.75%)下提供可行的指纹保护。我们的代码可以在https://github.com/nlsde-safety-team/FingerSafe.上找到



## **28. Efficient Detection and Filtering Systems for Distributed Training**

用于分布式训练的高效检测和过滤系统 cs.LG

18 pages, 14 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.08085v3)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: A plethora of modern machine learning tasks requires the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities that only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate a 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.

摘要: 过多的现代机器学习任务需要利用大规模分布式集群作为培训管道的关键组成部分。然而，工作者节点的异常拜占庭行为会破坏训练，影响推理的质量。此类行为可归因于无意的系统故障或精心策划的攻击；因此，某些节点可能会向协调训练的参数服务器(PS)返回任意结果。最近的工作考虑了广泛的攻击模型，并探索了稳健的聚集和/或计算冗余来纠正扭曲的梯度。在这项工作中，我们考虑了从强到强的攻击模型：$q$全知的对手，完全了解防御协议，可以从一个迭代到另一个迭代变化；$q$随机选择的对手，合谋能力有限，一次只有几个迭代改变。我们的算法依赖于冗余的任务分配以及对敌对行为的检测。对于强攻击，我们展示了与以前最先进的技术相比，扭曲梯度的比例降低了16%-99%。我们在CIFAR-10数据集上的TOP-1分类精度结果显示，在最复杂的攻击下，与最先进的方法相比，准确率(在强和弱情况下平均)高出25%。



## **29. Optimal Bootstrapping of PoW Blockchains**

PoW区块链的最优自举 cs.CR

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10618v1)

**Authors**: Ranvir Rana, Dimitris Karakostas, Sreeram Kannan, Aggelos Kiayias, Pramod Viswanath

**Abstracts**: Proof of Work (PoW) blockchains are susceptible to adversarial majority mining attacks in the early stages due to incipient participation and corresponding low net hash power. Bootstrapping ensures safety and liveness during the transient stage by protecting against a majority mining attack, allowing a PoW chain to grow the participation base and corresponding mining hash power. Liveness is especially important since a loss of liveness will lead to loss of honest mining rewards, decreasing honest participation, hence creating an undesired spiral; indeed existing bootstrapping mechanisms offer especially weak liveness guarantees.   In this paper, we propose Advocate, a new bootstrapping methodology, which achieves two main results: (a) optimal liveness and low latency under a super-majority adversary for the Nakamoto longest chain protocol and (b) immediate black-box generalization to a variety of parallel-chain based scaling architectures, including OHIE and Prism. We demonstrate via a full-stack implementation the robustness of Advocate under a 90% adversarial majority.

摘要: 工作证明(PoW)区块链在早期阶段容易受到对抗性多数挖掘攻击，原因是早期参与和相应的低网络散列能力。Bootstrapping通过防止大多数挖掘攻击，允许POW链扩大参与基础和相应的挖掘散列能力，确保了临时阶段的安全性和活跃性。活力的丧失尤其重要，因为活力的丧失将导致诚实挖掘奖励的丧失，减少诚实的参与，从而造成不受欢迎的螺旋；事实上，现有的自举机制提供的活力保障尤其薄弱。在本文中，我们提出了一种新的自举方法Advocate，它获得了两个主要结果：(A)对于Nakamoto最长链协议，在超级多数对手的情况下，获得了最优的活跃度和低延迟；(B)立即将黑盒推广到了各种基于并行链的伸缩架构，包括Ohie和Prism。我们通过全栈实现演示了Advocate在90%的对手多数情况下的健壮性。



## **30. Different Spectral Representations in Optimized Artificial Neural Networks and Brains**

优化人工神经网络和大脑中的不同谱表示 cs.LG

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10576v1)

**Authors**: Richard C. Gerum, Cassidy Pirlot, Alona Fyshe, Joel Zylberberg

**Abstracts**: Recent studies suggest that artificial neural networks (ANNs) that match the spectral properties of the mammalian visual cortex -- namely, the $\sim 1/n$ eigenspectrum of the covariance matrix of neural activities -- achieve higher object recognition performance and robustness to adversarial attacks than those that do not. To our knowledge, however, no previous work systematically explored how modifying the ANN's spectral properties affects performance. To fill this gap, we performed a systematic search over spectral regularizers, forcing the ANN's eigenspectrum to follow $1/n^\alpha$ power laws with different exponents $\alpha$. We found that larger powers (around 2--3) lead to better validation accuracy and more robustness to adversarial attacks on dense networks. This surprising finding applied to both shallow and deep networks and it overturns the notion that the brain-like spectrum (corresponding to $\alpha \sim 1$) always optimizes ANN performance and/or robustness. For convolutional networks, the best $\alpha$ values depend on the task complexity and evaluation metric: lower $\alpha$ values optimized validation accuracy and robustness to adversarial attack for networks performing a simple object recognition task (categorizing MNIST images of handwritten digits); for a more complex task (categorizing CIFAR-10 natural images), we found that lower $\alpha$ values optimized validation accuracy whereas higher $\alpha$ values optimized adversarial robustness. These results have two main implications. First, they cast doubt on the notion that brain-like spectral properties ($\alpha \sim 1$) \emph{always} optimize ANN performance. Second, they demonstrate the potential for fine-tuned spectral regularizers to optimize a chosen design metric, i.e., accuracy and/or robustness.

摘要: 最近的研究表明，与哺乳动物视觉皮质的光谱特性匹配的人工神经网络(ANN)--即神经活动协方差矩阵的1/n特征谱--比不匹配的人工神经网络具有更高的目标识别性能和对对手攻击的鲁棒性。然而，据我们所知，以前还没有系统地研究改变ANN的光谱属性对性能的影响。为了填补这一空白，我们对频谱正则化进行了系统的搜索，迫使ANN的本征谱遵循具有不同指数$\α$的$1/n^\α$幂定律。我们发现，更大的功率(大约2-3)会导致更好的验证精度和对密集网络上的对手攻击更强的稳健性。这一令人惊讶的发现既适用于浅层网络，也适用于深层网络，它推翻了类脑频谱(对应于$\α\sim 1$)总是优化ANN性能和/或健壮性的概念。对于卷积网络，最佳的$\α$值取决于任务的复杂性和评估指标：对于执行简单的目标识别任务(对手写数字的MNIST图像进行分类)的网络，较低的$\α$值优化了验证准确性和对对手攻击的稳健性；对于较复杂的任务(对CIFAR-10自然图像进行分类)，我们发现较低的$\α$值优化了验证精度，而较高的$\α$值优化了对手攻击的鲁棒性。这些结果有两个主要影响。首先，他们对类似大脑的光谱特性($\α\sim 1$)\emph{Always}优化ANN性能的概念提出了质疑。其次，它们展示了微调频谱正则化的潜力，以优化所选择的设计度量，即准确性和/或稳健性。



## **31. On the Decision Boundaries of Neural Networks: A Tropical Geometry Perspective**

关于神经网络的决策边界：热带几何观点 cs.LG

First two authors contributed equally to this work

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2002.08838v3)

**Authors**: Motasem Alfarra, Adel Bibi, Hasan Hammoud, Mohamed Gaafar, Bernard Ghanem

**Abstracts**: This work tackles the problem of characterizing and understanding the decision boundaries of neural networks with piecewise linear non-linearity activations. We use tropical geometry, a new development in the area of algebraic geometry, to characterize the decision boundaries of a simple network of the form (Affine, ReLU, Affine). Our main finding is that the decision boundaries are a subset of a tropical hypersurface, which is intimately related to a polytope formed by the convex hull of two zonotopes. The generators of these zonotopes are functions of the network parameters. This geometric characterization provides new perspectives to three tasks. (i) We propose a new tropical perspective to the lottery ticket hypothesis, where we view the effect of different initializations on the tropical geometric representation of a network's decision boundaries. (ii) Moreover, we propose new tropical based optimization reformulations that directly influence the decision boundaries of the network for the task of network pruning. (iii) At last, we discuss the reformulation of the generation of adversarial attacks in a tropical sense. We demonstrate that one can construct adversaries in a new tropical setting by perturbing a specific set of decision boundaries by perturbing a set of parameters in the network.

摘要: 这项工作解决了描述和理解具有分段线性非线性激励的神经网络的决策边界的问题。我们利用热带几何这一代数几何领域的新发展来刻画形式为(Affine，Relu，Affine)的简单网络的决策边界。我们的主要发现是，决策边界是热带超曲面的一个子集，它与由两个极地凸壳形成的多面体密切相关。这些地带区的生成器是网络参数的函数。这种几何特征为三项任务提供了新的视角。(I)我们对彩票假设提出了一个新的热带观点，其中我们考虑了不同的初始化对网络决策边界的热带几何表示的影响。(Ii)此外，我们还提出了直接影响网络剪枝任务的网络决策边界的新的热带优化重构公式。(Iii)最后，我们讨论了热带意义上的对抗性攻击生成的重新表述。我们证明了一个人可以通过扰动网络中的一组参数来扰动一组特定的决策边界，从而在新的热带环境中构建对手。



## **32. Toward Better Target Representation for Source-Free and Black-Box Domain Adaptation**

无源域和黑箱域自适应的更好的目标表示 cs.CV

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10531v1)

**Authors**: Qucheng Peng, Zhengming Ding, Lingjuan Lyu, Lichao Sun, Chen Chen

**Abstracts**: Domain adaptation aims at aligning the labeled source domain and the unlabeled target domain, and most existing approaches assume the source data is accessible. Unfortunately, this paradigm raises concerns in data privacy and security. Recent studies try to dispel these concerns by the Source-Free setting, which adapts the source-trained model towards target domain without exposing the source data. However, the Source-Free paradigm is still at risk of data leakage due to adversarial attacks to the source model. Hence, the Black-Box setting is proposed, where only the outputs of source model can be utilized. In this paper, we address both the Source-Free adaptation and the Black-Box adaptation, proposing a novel method named better target representation from Frequency Mixup and Mutual Learning (FMML). Specifically, we introduce a new data augmentation technique as Frequency MixUp, which highlights task-relevant objects in the interpolations, thus enhancing class-consistency and linear behavior for target models. Moreover, we introduce a network regularization method called Mutual Learning to the domain adaptation problem. It transfers knowledge inside the target model via self-knowledge distillation and thus alleviates overfitting on the source domain by learning multi-scale target representations. Extensive experiments show that our method achieves state-of-the-art performance on several benchmark datasets under both settings.

摘要: 域自适应的目的是将已标记的源域和未标记的目标域对齐，现有的大多数方法都假设源数据是可访问的。不幸的是，这种模式引发了对数据隐私和安全的担忧。最近的研究试图通过无源设置来消除这些担忧，该设置使源训练模型适应目标领域，而不暴露源数据。然而，由于对源模型的敌意攻击，自由源代码范例仍然面临数据泄露的风险。因此，提出了黑箱设置，其中只能利用源模型的输出。本文针对无源自适应和黑盒自适应两种情况，提出了一种新的基于混频互学习的目标更好表示方法(FMML)。具体地说，我们引入了一种新的数据增强技术--频率混合，它在内插中突出了与任务相关的对象，从而增强了目标模型的类一致性和线性行为。此外，我们还将一种称为相互学习的网络正则化方法引入到领域自适应问题中。它通过自知识蒸馏在目标模型内部传递知识，从而通过学习多尺度目标表示来缓解源域上的过度匹配。大量的实验表明，我们的方法在两种设置下都能在多个基准数据集上获得最好的性能。



## **33. BARReL: Bottleneck Attention for Adversarial Robustness in Vision-Based Reinforcement Learning**

Barrel：基于视觉的强化学习中对抗性鲁棒性的瓶颈关注 cs.LG

5 pages, 2 figures, 3 tables

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10481v1)

**Authors**: Eugene Bykovets, Yannick Metz, Mennatallah El-Assady, Daniel A. Keim, Joachim M. Buhmann

**Abstracts**: Robustness to adversarial perturbations has been explored in many areas of computer vision. This robustness is particularly relevant in vision-based reinforcement learning, as the actions of autonomous agents might be safety-critic or impactful in the real world. We investigate the susceptibility of vision-based reinforcement learning agents to gradient-based adversarial attacks and evaluate a potential defense. We observe that Bottleneck Attention Modules (BAM) included in CNN architectures can act as potential tools to increase robustness against adversarial attacks. We show how learned attention maps can be used to recover activations of a convolutional layer by restricting the spatial activations to salient regions. Across a number of RL environments, BAM-enhanced architectures show increased robustness during inference. Finally, we discuss potential future research directions.

摘要: 对对抗性扰动的稳健性已经在计算机视觉的许多领域进行了探索。这种稳健性在基于视觉的强化学习中特别相关，因为自主代理的行为在现实世界中可能是安全关键的或有影响的。我们研究了基于视觉的强化学习代理对基于梯度的对抗性攻击的敏感性，并评估了一种潜在的防御。我们观察到CNN体系结构中包含的瓶颈注意模块(BAM)可以作为潜在的工具来增强对对手攻击的健壮性。我们展示了如何通过将空间激活限制在显著区域来使用学习的注意图来恢复卷积层的激活。在许多RL环境中，BAM增强型体系结构在推理过程中表现出更强的健壮性。最后，我们讨论了未来可能的研究方向。



## **34. Membership-Doctor: Comprehensive Assessment of Membership Inference Against Machine Learning Models**

成员资格-博士：针对机器学习模型的成员资格推理的综合评估 cs.CR

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10445v1)

**Authors**: Xinlei He, Zheng Li, Weilin Xu, Cory Cornelius, Yang Zhang

**Abstracts**: Machine learning models are prone to memorizing sensitive data, making them vulnerable to membership inference attacks in which an adversary aims to infer whether an input sample was used to train the model. Over the past few years, researchers have produced many membership inference attacks and defenses. However, these attacks and defenses employ a variety of strategies and are conducted in different models and datasets. The lack of comprehensive benchmark, however, means we do not understand the strengths and weaknesses of existing attacks and defenses.   We fill this gap by presenting a large-scale measurement of different membership inference attacks and defenses. We systematize membership inference through the study of nine attacks and six defenses and measure the performance of different attacks and defenses in the holistic evaluation. We then quantify the impact of the threat model on the results of these attacks. We find that some assumptions of the threat model, such as same-architecture and same-distribution between shadow and target models, are unnecessary. We are also the first to execute attacks on the real-world data collected from the Internet, instead of laboratory datasets. We further investigate what determines the performance of membership inference attacks and reveal that the commonly believed overfitting level is not sufficient for the success of the attacks. Instead, the Jensen-Shannon distance of entropy/cross-entropy between member and non-member samples correlates with attack performance much better. This gives us a new way to accurately predict membership inference risks without running the attack. Finally, we find that data augmentation degrades the performance of existing attacks to a larger extent, and we propose an adaptive attack using augmentation to train shadow and attack models that improve attack performance.

摘要: 机器学习模型容易记住敏感数据，这使得它们很容易受到成员资格推理攻击，在这种攻击中，对手的目标是推断输入样本是否用于训练模型。在过去的几年里，研究人员已经产生了许多成员推理攻击和防御。然而，这些攻击和防御使用了各种策略，并在不同的模型和数据集中进行。然而，缺乏全面的基准意味着我们不了解现有攻击和防御的优势和劣势。我们通过提供不同成员推理攻击和防御的大规模测量来填补这一空白。我们通过对9个攻击和6个防御的研究，使隶属度推理系统化，并在整体评估中衡量不同攻击和防御的性能。然后，我们量化威胁模型对这些攻击结果的影响。我们发现，威胁模型中的一些假设是不必要的，例如阴影模型和目标模型之间的相同体系结构和相同分布。我们也是第一个对从互联网收集的真实世界数据而不是实验室数据集进行攻击的国家。我们进一步研究了是什么决定了成员关系推理攻击的性能，并揭示了人们普遍认为的过度匹配水平不足以使攻击成功。相反，成员样本和非成员样本之间的熵/交叉熵的Jensen-Shannon距离与攻击性能的相关性要好得多。这为我们在不运行攻击的情况下准确预测成员关系推断风险提供了一种新的方法。最后，我们发现数据扩充在很大程度上降低了现有攻击的性能，并提出了一种利用扩充来训练影子和攻击模型的自适应攻击，从而提高了攻击性能。



## **35. On Deep Learning in Password Guessing, a Survey**

密码猜测中的深度学习研究综述 cs.CR

8 pages, 4 figures, 3 tables. arXiv admin note: substantial text  overlap with arXiv:2208.06943

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10413v1)

**Authors**: Fangyi Yu

**Abstracts**: The security of passwords is dependent on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to be representative of the actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper compares various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. The involved model categories are Recurrent Neural Networks, Generative Adversarial Networks, Autoencoder, and Attention mechanisms. Additionally, we proposed a promising research experimental design on using variations of IWGAN on password guessing under non-targeted offline attacks. Using these advanced strategies, we can enhance password security and create more accurate and efficient Password Strength Meters.

摘要: 密码的安全性取决于对攻击者使用的策略的透彻理解。不幸的是，现实世界中的对手使用的是实用的猜测策略，如字典攻击，这在密码安全研究中很难模拟。必须仔细配置和修改字典攻击，才能代表实际威胁。然而，这种方法需要难以复制的特定领域的知识和专业技能。本文比较了各种基于深度学习的密码猜测方法，这些方法不需要领域知识，也不需要假设用户的密码结构和组合。涉及的模型类别包括递归神经网络、生成性对抗网络、自动编码器和注意机制。此外，我们还提出了一种在非目标离线攻击下使用IWGAN变体进行密码猜测的研究实验设计。使用这些高级策略，我们可以增强密码安全性，并创建更准确和高效的密码强度计。



## **36. Fight Fire With Fire: Reversing Skin Adversarial Examples by Multiscale Diffusive and Denoising Aggregation Mechanism**

以火攻火：用多尺度扩散和去噪聚集机制逆转皮肤对抗性实例 cs.CV

11 pages

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2208.10373v1)

**Authors**: Yongwei Wang, Yuan Li, Zhiqi Shen

**Abstracts**: Reliable skin cancer diagnosis models play an essential role in early screening and medical intervention. Prevailing computer-aided skin cancer classification systems employ deep learning approaches. However, recent studies reveal their extreme vulnerability to adversarial attacks -- often imperceptible perturbations to significantly reduce performances of skin cancer diagnosis models. To mitigate these threats, this work presents a simple, effective and resource-efficient defense framework by reverse engineering adversarial perturbations in skin cancer images. Specifically, a multiscale image pyramid is first established to better preserve discriminative structures in medical imaging domain. To neutralize adversarial effects, skin images at different scales are then progressively diffused by injecting isotropic Gaussian noises to move the adversarial examples to the clean image manifold. Crucially, to further reverse adversarial noises and suppress redundant injected noises, a novel multiscale denoising mechanism is carefully designed that aggregates image information from neighboring scales. We evaluated the defensive effectiveness of our method on ISIC 2019, a largest skin cancer multiclass classification dataset. Experimental results demonstrate that the proposed method can successfully reverse adversarial perturbations from different attacks and significantly outperform some state-of-the-art methods in defending skin cancer diagnosis models.

摘要: 可靠的皮肤癌诊断模型在早期筛查和医疗干预中起着至关重要的作用。流行的计算机辅助皮肤癌分类系统采用深度学习方法。然而，最近的研究揭示了它们对对抗性攻击的极端脆弱性--通常是难以察觉的干扰，从而显著降低皮肤癌诊断模型的性能。为了减轻这些威胁，这项工作提出了一种简单、有效和资源高效的防御框架，通过反向工程皮肤癌图像中的对抗性扰动。具体地说，首先建立了多尺度图像金字塔，以更好地保留医学成像领域的区分结构。为了中和对抗性效果，然后通过注入各向同性高斯噪声来逐步扩散不同尺度上的皮肤图像，以将对抗性示例移动到干净的图像流形上。为了进一步逆转对抗性噪声和抑制多余的注入噪声，仔细设计了一种新的多尺度去噪机制，该机制聚集了相邻尺度上的图像信息。我们在最大的皮肤癌多类分类数据集ISIC 2019上评估了我们方法的防御效果。实验结果表明，该方法能够成功地逆转来自不同攻击的对抗性扰动，并且在防御皮肤癌诊断模型方面明显优于一些最新的方法。



## **37. Adversarial Classification under Gaussian Mechanism: Calibrating the Attack to Sensitivity**

高斯机制下的对抗性分类：将攻击校准为敏感度 cs.IT

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2201.09751v4)

**Authors**: Ayse Unsal, Melek Onen

**Abstracts**: This work studies anomaly detection under differential privacy (DP) with Gaussian perturbation using both statistical and information-theoretic tools. In our setting, the adversary aims to modify the content of a statistical dataset by inserting additional data without being detected by using the DP guarantee to her own benefit. To this end, we characterize information-theoretic and statistical thresholds for the first and second-order statistics of the adversary's attack, which balances the privacy budget and the impact of the attack in order to remain undetected. Additionally, we introduce a new privacy metric based on Chernoff information for classifying adversaries under differential privacy as a stronger alternative to $(\epsilon, \delta)-$ and Kullback-Leibler DP for the Gaussian mechanism. Analytical results are supported by numerical evaluations.

摘要: 本文利用统计学和信息论的方法研究了高斯扰动下的异常检测问题。在我们的设置中，敌手的目标是通过插入附加数据来修改统计数据集的内容，而不会被使用DP保证检测到，从而使自己受益。为此，我们刻画了攻击者攻击的一阶和二阶统计量的信息论和统计阈值，该阈值平衡了隐私预算和攻击的影响，以便保持未被检测到。此外，我们引入了一种新的基于Chernoff信息的隐私度量，用于区分隐私下的攻击者，作为对$(\epsilon，\Delta)-$和用于高斯机制的Kullback-Leibler DP的更强选择。分析结果得到数值评估的支持。



## **38. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

自主车辆轨迹预测的对抗稳健性研究 cs.CV

13 pages, 13 figures, accepted by CVPR 2022

**SubmitDate**: 2022-08-22    [paper-pdf](http://arxiv.org/pdf/2201.05057v3)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing. The implementation is open source at https://github.com/zqzqz/AdvTrajectoryPrediction.

摘要: 轨迹预测是自动驾驶车辆进行安全规划和导航的重要组成部分。然而，很少有研究分析弹道预测的对抗稳健性，也没有研究最坏情况的预测是否仍能导致安全规划。为了弥补这一差距，我们研究了轨迹预测模型的对抗稳健性，提出了一种新的对抗性攻击，通过扰动正常的车辆轨迹来最大化预测误差。我们在三个模型和三个数据集上的实验表明，对抗性预测使预测误差增加了150%以上。我们的案例研究表明，如果对手沿着敌对轨迹驾驶车辆接近目标AV，AV可能会做出不准确的预测，甚至做出不安全的驾驶决策。我们还通过数据增强和轨迹平滑来探索可能的缓解技术。该实现在https://github.com/zqzqz/AdvTrajectoryPrediction.上是开源的



## **39. Inferring Sensitive Attributes from Model Explanations**

从模型解释中推断敏感属性 cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09967v1)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.

摘要: 模型解释为模型构建者提供了对经过训练的机器学习模型的黑箱行为的透明性。它们表明了不同的输入属性对其相应模型预测的影响。解释对输入的依赖引发了对敏感用户数据的隐私问题。然而，目前的文献对模型解释的隐私风险的讨论有限。我们专注于属性推理攻击的特定隐私风险，其中对手根据输入的模型解释推断输入的敏感属性(例如，种族和性别)。我们针对两个威胁模型中的模型解释设计了第一个属性推理攻击，在这两个模型中，建模者或者(A)在训练数据和输入中包括敏感属性，或者(B)通过在训练数据和输入中不包括敏感属性来审查敏感属性。我们在四个基准数据集和四个最先进的算法上评估了我们提出的攻击。我们表明，攻击者可以从两种威胁模型中的解释中准确地推断出敏感属性的值。此外，即使只利用与敏感属性对应的解释，攻击也是成功的。这些都表明，我们的攻击针对解释是有效的，并对数据隐私构成了实际威胁。在将模型预测(先前攻击所利用的攻击面)与解释相结合时，我们注意到攻击成功率并没有提高。此外，与仅利用模型预测相比，利用模型解释的攻击成功更好。这些都表明，模型解释是对手可以利用的强大攻击面。



## **40. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

增量对抗性(IMA)训练提高神经网络对抗性鲁棒性 cs.CV

12 pages

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2005.09147v9)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial noises. By adding adversarial noises to training samples, adversarial training can improve the model's robustness against adversarial noises. However, adversarial training samples with excessive noises can harm standard accuracy, which may be unacceptable for many medical image analysis applications. This issue has been termed the trade-off between standard accuracy and adversarial robustness. In this paper, we hypothesize that this issue may be alleviated if the adversarial samples for training are placed right on the decision boundaries. Based on this hypothesis, we design an adaptive adversarial training method, named IMA. For each individual training sample, IMA makes a sample-wise estimation of the upper bound of the adversarial perturbation. In the training process, each of the sample-wise adversarial perturbations is gradually increased to match the margin. Once an equilibrium state is reached, the adversarial perturbations will stop increasing. IMA is evaluated on publicly available datasets under two popular adversarial attacks, PGD and IFGSM. The results show that: (1) IMA significantly improves adversarial robustness of DNN classifiers, which achieves state-of-the-art performance; (2) IMA has a minimal reduction in clean accuracy among all competing defense methods; (3) IMA can be applied to pretrained models to reduce time cost; (4) IMA can be applied to the state-of-the-art medical image segmentation networks, with outstanding performance. We hope our work may help to lift the trade-off between adversarial robustness and clean accuracy and facilitate the development of robust applications in the medical field. The source code will be released when this paper is published.

摘要: 深度神经网络(DNN)很容易受到对抗性噪声的影响。通过在训练样本中加入对抗性噪声，对抗性训练可以提高模型对对抗性噪声的鲁棒性。然而，含有过多噪声的对抗性训练样本可能会损害标准精度，这对于许多医学图像分析应用来说可能是不可接受的。这个问题被称为标准准确性和对抗性稳健性之间的权衡。在本文中，我们假设，如果用于训练的对手样本正确地放置在决策边界上，这个问题可能会得到缓解。基于这一假设，我们设计了一种自适应对抗性训练方法IMA。对于每个单独的训练样本，IMA逐个估计对抗性扰动的上界。在训练过程中，每个样本对抗性扰动都会逐渐增加，以匹配差值。一旦达到均衡状态，对抗性扰动将停止增加。IMA是在两种流行的对抗性攻击PGD和IFGSM下基于公开可用的数据集进行评估的。结果表明：(1)IMA显著提高了DNN分类器的对抗健壮性，达到了最好的性能；(2)IMA在所有竞争防御方法中干净准确率的降幅最小；(3)IMA可以应用于预先训练的模型，以减少时间开销；(4)IMA可以应用于最先进的医学图像分割网络，性能优异。我们希望我们的工作可以帮助消除对抗性健壮性和干净准确性之间的权衡，并促进医疗领域健壮性应用的发展。源代码将在这篇论文发表时发布。



## **41. MockingBERT: A Method for Retroactively Adding Resilience to NLP Models**

MockingBERT：一种向NLP模型追溯添加弹性的方法 cs.CL

8 pages (excl. bibiography and appendix), 2 figures The code  necessary for reproduction is available at  https://github.com/akash13singh/resilient_nlp To be published in Proceedings  of the 29th International Conference on Computational Linguistics (COLING  2022)

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09915v1)

**Authors**: Jan Jezabek, Akash Singh

**Abstracts**: Protecting NLP models against misspellings whether accidental or adversarial has been the object of research interest for the past few years. Existing remediations have typically either compromised accuracy or required full model re-training with each new class of attacks. We propose a novel method of retroactively adding resilience to misspellings to transformer-based NLP models. This robustness can be achieved without the need for re-training of the original NLP model and with only a minimal loss of language understanding performance on inputs without misspellings. Additionally we propose a new efficient approximate method of generating adversarial misspellings, which significantly reduces the cost needed to evaluate a model's resilience to adversarial attacks.

摘要: 在过去的几年里，保护NLP模型免受拼写错误的影响，无论是偶然的还是对抗性的，一直是人们感兴趣的研究对象。现有的补救措施通常要么影响准确性，要么需要对每一类新的攻击进行完整的模型重新训练。我们提出了一种新的方法，在基于变压器的NLP模型中回溯添加对拼写错误的恢复能力。这种稳健性可以在不需要重新训练原始NLP模型的情况下实现，并且在输入没有拼写错误的情况下只会对语言理解性能造成最小的损失。此外，我们还提出了一种新的生成对抗性拼写错误的有效近似方法，该方法大大降低了评估模型对对抗性攻击的恢复能力所需的成本。



## **42. On The Robustness of Channel Allocation in Joint Radar And Communication Systems: An Auction Approach**

雷达与通信联合系统中信道分配的稳健性：拍卖方法 cs.GT

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09821v1)

**Authors**: Ismail Lotfi, Hongyang Du, Dusit Niyato, Sumei Sun, Dong In Kim

**Abstracts**: Joint radar and communication (JRC) is a promising technique for spectrum re-utilization, which enables radar sensing and data transmission to operate on the same frequencies and the same devices. However, due to the multi-objective property of JRC systems, channel allocation to JRC nodes should be carefully designed to maximize system performance. Additionally, because of the broadcast nature of wireless signals, a watchful adversary, i.e., a warden, can detect ongoing transmissions and attack the system. Thus, we develop a covert JRC system that minimizes the detection probability by wardens, in which friendly jammers are deployed to improve the covertness of the JRC nodes during radar sensing and data transmission operations. Furthermore, we propose a robust multi-item auction design for channel allocation for such a JRC system that considers the uncertainty in bids. The proposed auction mechanism achieves the properties of truthfulness, individual rationality, budget feasibility, and computational efficiency. The simulations clearly show the benefits of our design to support covert JRC systems and to provide incentive to the JRC nodes in obtaining spectrum, in which the auction-based channel allocation mechanism is robust against perturbations in the bids, which is highly effective for JRC nodes working in uncertain environments.

摘要: 联合雷达与通信(JRC)是一种很有前途的频谱再利用技术，它使雷达感知和数据传输能够在相同的频率和相同的设备上运行。然而，由于JRC系统的多目标特性，必须仔细设计JRC节点的信道分配以最大化系统性能。此外，由于无线信号的广播性质，警惕的对手，即典狱长，可以检测到正在进行的传输并攻击系统。因此，我们开发了一种隐蔽的JRC系统，该系统可以最小化管理员的发现概率，在该系统中部署友好的干扰器来提高JRC节点在雷达侦听和数据传输操作中的隐蔽性。此外，我们还提出了一种稳健的多物品拍卖设计，用于考虑出价不确定性的JRC系统的信道分配。该拍卖机制具有真实性、个体合理性、预算可行性和计算效率等特点。仿真结果表明，本文设计的JRC系统支持隐蔽JRC系统，并激励JRC节点获得频谱，其中基于拍卖的信道分配机制对投标中的扰动具有较强的鲁棒性，这对于工作在不确定环境中的JRC节点是非常有效的。



## **43. PointDP: Diffusion-driven Purification against Adversarial Attacks on 3D Point Cloud Recognition**

PointDP：扩散驱动的三维点云识别对抗攻击净化 cs.CV

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09801v1)

**Authors**: Jiachen Sun, Weili Nie, Zhiding Yu, Z. Morley Mao, Chaowei Xiao

**Abstracts**: 3D Point cloud is becoming a critical data representation in many real-world applications like autonomous driving, robotics, and medical imaging. Although the success of deep learning further accelerates the adoption of 3D point clouds in the physical world, deep learning is notorious for its vulnerability to adversarial attacks. In this work, we first identify that the state-of-the-art empirical defense, adversarial training, has a major limitation in applying to 3D point cloud models due to gradient obfuscation. We further propose PointDP, a purification strategy that leverages diffusion models to defend against 3D adversarial attacks. We extensively evaluate PointDP on six representative 3D point cloud architectures, and leverage 10+ strong and adaptive attacks to demonstrate its lower-bound robustness. Our evaluation shows that PointDP achieves significantly better robustness than state-of-the-art purification methods under strong attacks. Results of certified defenses on randomized smoothing combined with PointDP will be included in the near future.

摘要: 3D点云正在成为自动驾驶、机器人和医学成像等许多现实世界应用中的关键数据表示。尽管深度学习的成功进一步加速了3D点云在物理世界中的采用，但深度学习因其易受对手攻击而臭名昭著。在这项工作中，我们首先确定了最先进的经验防御，对抗性训练，由于梯度混淆，在应用于3D点云模型时有一个主要限制。我们进一步提出了PointDP，这是一种利用扩散模型来防御3D对手攻击的净化策略。我们在六种典型的三维点云架构上对PointDP进行了广泛的评估，并利用10+强大的自适应攻击来展示其下界健壮性。我们的评估表明，PointDP在强攻击下比最先进的净化方法具有更好的健壮性。随机平滑与PointDP相结合的认证防御结果将在不久的将来公布。



## **44. Robust Node Classification on Graphs: Jointly from Bayesian Label Transition and Topology-based Label Propagation**

图上稳健的节点分类：贝叶斯标签转移和基于拓扑的标签传播 cs.LG

The paper is accepted for CIKM 2022

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09779v1)

**Authors**: Jun Zhuang, Mohammad Al Hasan

**Abstracts**: Node classification using Graph Neural Networks (GNNs) has been widely applied in various real-world scenarios. However, in recent years, compelling evidence emerges that the performance of GNN-based node classification may deteriorate substantially by topological perturbation, such as random connections or adversarial attacks. Various solutions, such as topological denoising methods and mechanism design methods, have been proposed to develop robust GNN-based node classifiers but none of these works can fully address the problems related to topological perturbations. Recently, the Bayesian label transition model is proposed to tackle this issue but its slow convergence may lead to inferior performance. In this work, we propose a new label inference model, namely LInDT, which integrates both Bayesian label transition and topology-based label propagation for improving the robustness of GNNs against topological perturbations. LInDT is superior to existing label transition methods as it improves the label prediction of uncertain nodes by utilizing neighborhood-based label propagation leading to better convergence of label inference. Besides, LIndT adopts asymmetric Dirichlet distribution as a prior, which also helps it to improve label inference. Extensive experiments on five graph datasets demonstrate the superiority of LInDT for GNN-based node classification under three scenarios of topological perturbations.

摘要: 基于图神经网络(GNN)的节点分类在各种现实场景中有着广泛的应用。然而，近年来，令人信服的证据表明，基于GNN的节点分类的性能可能会因拓扑扰动而显著恶化，例如随机连接或对抗性攻击。人们已经提出了各种解决方案，如拓扑去噪方法和机制设计方法，以开发基于GNN的健壮节点分类器，但这些工作都不能完全解决与拓扑扰动相关的问题。最近，贝叶斯标签转移模型被提出来解决这一问题，但其收敛速度慢可能导致性能下降。在这项工作中，我们提出了一种新的标签推理模型Lindt，该模型集成了贝叶斯标签转移和基于拓扑的标签传播，以提高GNN对拓扑扰动的健壮性。LINDT通过利用基于邻域的标签传播改进了对不确定节点的标签预测，从而使标签推理具有更好的收敛性能，因此优于现有的标签转换方法。此外，Lindt采用了非对称Dirichlet分布作为先验，这也有助于改进标签推理。在五个图数据集上的大量实验证明了LINDT在三种拓扑扰动场景下对基于GNN的节点分类的优越性。



## **45. GAIROSCOPE: Injecting Data from Air-Gapped Computers to Nearby Gyroscopes**

GIROSCOPE：将气隙计算机的数据注入附近的陀螺仪 cs.CR

**SubmitDate**: 2022-08-21    [paper-pdf](http://arxiv.org/pdf/2208.09764v1)

**Authors**: Mordechai Guri

**Abstracts**: It is known that malware can leak data from isolated, air-gapped computers to nearby smartphones using ultrasonic waves. However, this covert channel requires access to the smartphone's microphone, which is highly protected in Android OS and iOS, and might be non-accessible, disabled, or blocked.   In this paper we present `GAIROSCOPE,' an ultrasonic covert channel that doesn't require a microphone on the receiving side. Our malware generates ultrasonic tones in the resonance frequencies of the MEMS gyroscope. These inaudible frequencies produce tiny mechanical oscillations within the smartphone's gyroscope, which can be demodulated into binary information. Notably, the gyroscope in smartphones is considered to be a 'safe' sensor that can be used legitimately from mobile apps and javascript. We introduce the adversarial attack model and present related work. We provide the relevant technical background and show the design and implementation of GAIROSCOPE. We present the evaluation results and discuss a set of countermeasures to this threat. Our experiments show that attackers can exfiltrate sensitive information from air-gapped computers to smartphones located a few meters away via Speakers-to-Gyroscope covert channel.

摘要: 众所周知，恶意软件可以使用超声波将数据从隔离的、有空气间隙的计算机泄露到附近的智能手机。然而，这个隐蔽频道需要访问智能手机的麦克风，而麦克风在Android OS和iOS中受到高度保护，可能无法访问、禁用或阻止。在本文中，我们介绍了一种不需要在接收侧安装麦克风的超声波隐蔽通道--“GAIROSCOPE”。我们的恶意软件在MEMS陀螺仪的共振频率上产生超声波音调。这些听不见的频率会在智能手机的陀螺仪中产生微小的机械振荡，可以解调成二进制信息。值得注意的是，智能手机中的陀螺仪被认为是一种安全的传感器，可以在移动应用程序和Java脚本中合法使用。我们介绍了对抗性攻击模型，并介绍了相关的工作。我们提供了相关的技术背景，并展示了GAIROSCOPE的设计和实现。我们给出了评估结果，并讨论了一套应对这一威胁的对策。我们的实验表明，攻击者可以通过扬声器到陀螺仪的秘密通道，将敏感信息从有空气间隔的计算机泄露到几米外的智能手机。



## **46. GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification**

GAT：用于对抗性实例检测和稳健分类的生成性对抗性训练 cs.LG

ICLR 2020, code is available at  https://github.com/xuwangyin/GAT-Generative-Adversarial-Training

**SubmitDate**: 2022-08-20    [paper-pdf](http://arxiv.org/pdf/1905.11475v3)

**Authors**: Xuwang Yin, Soheil Kolouri, Gustavo K. Rohde

**Abstracts**: The vulnerabilities of deep neural networks against adversarial examples have become a significant concern for deploying these models in sensitive domains. Devising a definitive defense against such attacks is proven to be challenging, and the methods relying on detecting adversarial samples are only valid when the attacker is oblivious to the detection mechanism. In this paper we propose a principled adversarial example detection method that can withstand norm-constrained white-box attacks. Inspired by one-versus-the-rest classification, in a K class classification problem, we train K binary classifiers where the i-th binary classifier is used to distinguish between clean data of class i and adversarially perturbed samples of other classes. At test time, we first use a trained classifier to get the predicted label (say k) of the input, and then use the k-th binary classifier to determine whether the input is a clean sample (of class k) or an adversarially perturbed example (of other classes). We further devise a generative approach to detecting/classifying adversarial examples by interpreting each binary classifier as an unnormalized density model of the class-conditional data. We provide comprehensive evaluation of the above adversarial example detection/classification methods, and demonstrate their competitive performances and compelling properties.

摘要: 深层神经网络对敌意例子的脆弱性已经成为在敏感领域部署这些模型的一个重要问题。事实证明，针对此类攻击设计明确的防御是具有挑战性的，依赖于检测对手样本的方法只有在攻击者忘记检测机制时才有效。本文提出了一种能够抵抗范数约束白盒攻击的原则性对抗性实例检测方法。受一对一分类的启发，在一个K类分类问题中，我们训练了K个二进制分类器，其中第i个二进制分类器用于区分第I类的干净数据和其他类的相反扰动样本。在测试时，我们首先使用训练好的分类器来获得输入的预测标签(比如k)，然后使用第k个二进制分类器来确定输入是(k类的)干净样本还是(其他类的)相反的扰动样本。我们进一步设计了一种生成性方法，通过将每个二进制分类器解释为类条件数据的非归一化密度模型来检测/分类敌意示例。我们对上述恶意范例检测/分类方法进行了综合评价，并展示了它们的竞争性能和令人信服的性质。



## **47. Analyzing Adversarial Robustness of Vision Transformers against Spatial and Spectral Attacks**

视觉变形器对抗空间和频谱攻击的稳健性分析 cs.CV

11 pages, 13 figures

**SubmitDate**: 2022-08-20    [paper-pdf](http://arxiv.org/pdf/2208.09602v1)

**Authors**: Gihyun Kim, Jong-Seok Lee

**Abstracts**: Vision Transformers have emerged as a powerful architecture that can outperform convolutional neural networks (CNNs) in image classification tasks. Several attempts have been made to understand robustness of Transformers against adversarial attacks, but existing studies draw inconsistent results, i.e., some conclude that Transformers are more robust than CNNs, while some others find that they have similar degrees of robustness. In this paper, we address two issues unexplored in the existing studies examining adversarial robustness of Transformers. First, we argue that the image quality should be simultaneously considered in evaluating adversarial robustness. We find that the superiority of one architecture to another in terms of robustness can change depending on the attack strength expressed by the quality of the attacked images. Second, by noting that Transformers and CNNs rely on different types of information in images, we formulate an attack framework, called Fourier attack, as a tool for implementing flexible attacks, where an image can be attacked in the spectral domain as well as in the spatial domain. This attack perturbs the magnitude and phase information of particular frequency components selectively. Through extensive experiments, we find that Transformers tend to rely more on phase information and low frequency information than CNNs, and thus sometimes they are even more vulnerable under frequency-selective attacks. It is our hope that this work provides new perspectives in understanding the properties and adversarial robustness of Transformers.

摘要: 视觉转换器已经成为一种强大的体系结构，它在图像分类任务中的表现优于卷积神经网络(CNN)。人们曾试图了解变形金刚对敌方攻击的健壮性，但现有的研究得出了不一致的结果，即一些人得出结论，变形金刚比CNN更健壮，而另一些人发现它们具有类似程度的健壮性。在这篇文章中，我们解决了现有研究中未探索的两个问题，以检验变形金刚的对抗性稳健性。首先，我们认为在评估对抗稳健性时应同时考虑图像质量。我们发现，一种体系结构相对于另一种体系结构在稳健性方面的优势可以根据受攻击图像质量所表示的攻击强度而变化。其次，通过注意到Transformers和CNN依赖于图像中不同类型的信息，我们制定了一个称为傅立叶攻击的攻击框架，作为实现灵活攻击的工具，其中图像可以在谱域和空间域中被攻击。这种攻击选择性地干扰特定频率分量的幅度和相位信息。通过大量的实验，我们发现变压器往往比CNN更依赖于相位信息和低频信息，因此有时它们在频率选择性攻击下更容易受到攻击。我们希望这项工作为理解变形金刚的特性和对手的健壮性提供了新的视角。



## **48. Gender Bias and Universal Substitution Adversarial Attacks on Grammatical Error Correction Systems for Automated Assessment**

性别偏见和普遍替代对自动评估语法纠错系统的敌意攻击 cs.CL

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09466v1)

**Authors**: Vyas Raina, Mark Gales

**Abstracts**: Grammatical Error Correction (GEC) systems perform a sequence-to-sequence task, where an input word sequence containing grammatical errors, is corrected for these errors by the GEC system to output a grammatically correct word sequence. With the advent of deep learning methods, automated GEC systems have become increasingly popular. For example, GEC systems are often used on speech transcriptions of English learners as a form of assessment and feedback - these powerful GEC systems can be used to automatically measure an aspect of a candidate's fluency. The count of \textit{edits} from a candidate's input sentence (or essay) to a GEC system's grammatically corrected output sentence is indicative of a candidate's language ability, where fewer edits suggest better fluency. The count of edits can thus be viewed as a \textit{fluency score} with zero implying perfect fluency. However, although deep learning based GEC systems are extremely powerful and accurate, they are susceptible to adversarial attacks: an adversary can introduce a small, specific change at the input of a system that causes a large, undesired change at the output. When considering the application of GEC systems to automated language assessment, the aim of an adversary could be to cheat by making a small change to a grammatically incorrect input sentence that conceals the errors from a GEC system, such that no edits are found and the candidate is unjustly awarded a perfect fluency score. This work examines a simple universal substitution adversarial attack that non-native speakers of English could realistically employ to deceive GEC systems used for assessment.

摘要: 语法纠错(GEC)系统执行序列到序列任务，其中包含语法错误的输入单词序列由GEC系统针对这些错误进行校正，以输出语法正确的单词序列。随着深度学习方法的出现，自动化GEC系统变得越来越流行。例如，GEC系统经常用于英语学习者的语音转录，作为一种评估和反馈的形式--这些强大的GEC系统可以用来自动测量考生流利性的一个方面。从候选人的输入句子(或文章)到GEC系统语法更正后的输出句子的编辑次数表明了候选人的语言能力，编辑次数越少，表明语言的流利性越好。因此，可以将编辑次数视为文本{流畅度分数}，0表示完全流畅。然而，尽管基于深度学习的GEC系统非常强大和准确，但它们容易受到对手的攻击：对手可以在系统的输入端引入小的、特定的更改，从而在输出端造成巨大的、不希望看到的更改。当考虑将GEC系统应用于自动语言评估时，对手的目的可能是通过对语法错误的输入句子进行微小的改变来欺骗，该输入句子对GEC系统隐藏了错误，从而没有发现编辑并且不公平地给予候选人满分流利性分数。这项工作考察了一种简单的通用替代对抗性攻击，非英语母语者可以现实地使用该攻击来欺骗用于评估的GEC系统。



## **49. Curbing Task Interference using Representation Similarity-Guided Multi-Task Feature Sharing**

基于表示相似度的多任务特征共享抑制任务干扰 cs.CV

Published at 1st Conference on Lifelong Learning Agents (CoLLAs 2022)

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09427v1)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Multi-task learning of dense prediction tasks, by sharing both the encoder and decoder, as opposed to sharing only the encoder, provides an attractive front to increase both accuracy and computational efficiency. When the tasks are similar, sharing the decoder serves as an additional inductive bias providing more room for tasks to share complementary information among themselves. However, increased sharing exposes more parameters to task interference which likely hinders both generalization and robustness. Effective ways to curb this interference while exploiting the inductive bias of sharing the decoder remains an open challenge. To address this challenge, we propose Progressive Decoder Fusion (PDF) to progressively combine task decoders based on inter-task representation similarity. We show that this procedure leads to a multi-task network with better generalization to in-distribution and out-of-distribution data and improved robustness to adversarial attacks. Additionally, we observe that the predictions of different tasks of this multi-task network are more consistent with each other.

摘要: 密集预测任务的多任务学习通过共享编码器和解码器，而不是仅共享编码器，为提高精度和计算效率提供了一个有吸引力的前沿。当任务相似时，共享解码器充当额外的感应偏向，为任务之间共享互补信息提供更多空间。然而，增加共享会使更多的参数暴露在任务干扰中，这可能会阻碍泛化和健壮性。在利用共用解码器的感应偏差的同时抑制这种干扰的有效方法仍然是一个悬而未决的挑战。为了应对这一挑战，我们提出了渐进译码融合算法(PDF)，它基于任务间表示的相似性逐步合并任务译码。实验结果表明，该过程可以得到一个对分布内和分布外数据具有更好的泛化能力的多任务网络，并提高了对对手攻击的稳健性。此外，我们观察到这个多任务网络对不同任务的预测彼此更一致。



## **50. A Pragmatic Methodology for Blind Hardware Trojan Insertion in Finalized Layouts**

一种实用的在最终版图中插入硬件木马的方法 cs.CR

9 pages, 6 figures, 3 tables, to be published in ICCAD 2022

**SubmitDate**: 2022-08-19    [paper-pdf](http://arxiv.org/pdf/2208.09235v1)

**Authors**: Alexander Hepp, Tiago Perez, Samuel Pagliarini, Georg Sigl

**Abstracts**: A potential vulnerability for integrated circuits (ICs) is the insertion of hardware trojans (HTs) during manufacturing. Understanding the practicability of such an attack can lead to appropriate measures for mitigating it. In this paper, we demonstrate a pragmatic framework for analyzing HT susceptibility of finalized layouts. Our framework is representative of a fabrication-time attack, where the adversary is assumed to have access only to a layout representation of the circuit. The framework inserts trojans into tapeout-ready layouts utilizing an Engineering Change Order (ECO) flow. The attacked security nodes are blindly searched utilizing reverse-engineering techniques. For our experimental investigation, we utilized three crypto-cores (AES-128, SHA-256, and RSA) and a microcontroller (RISC-V) as targets. We explored 96 combinations of triggers, payloads and targets for our framework. Our findings demonstrate that even in high-density designs, the covert insertion of sophisticated trojans is possible. All this while maintaining the original target logic, with minimal impact on power and performance. Furthermore, from our exploration, we conclude that it is too naive to only utilize placement resources as a metric for HT vulnerability. This work highlights that the HT insertion success is a complex function of the placement, routing resources, the position of the attacked nodes, and further design-specific characteristics. As a result, our framework goes beyond just an attack, we present the most advanced analysis tool to assess the vulnerability of HT insertion into finalized layouts.

摘要: 集成电路(IC)的一个潜在漏洞是在制造过程中插入硬件特洛伊木马(HTS)。了解这种攻击的实用性可以导致采取适当的措施来减轻它。在这篇文章中，我们展示了一个实用的框架来分析最终版面的HT易感性。我们的框架代表了制造时间攻击，其中假设对手只能访问电路的布局表示。该框架利用工程变更单(ECO)流将特洛伊木马程序插入到支持Tapeout的布局中。利用逆向工程技术对被攻击的安全节点进行盲目搜索。在我们的实验研究中，我们使用了三个密码核(AES-128、SHA-256和RSA)和一个微控制器(RISC-V)作为目标。我们为我们的框架探索了触发器、有效负载和目标的96种组合。我们的发现表明，即使在高密度设计中，隐藏复杂的特洛伊木马也是可能的。所有这些都保持了原始的目标逻辑，对功率和性能的影响最小。此外，从我们的探索中，我们得出结论，只利用放置资源作为HT脆弱性的度量太幼稚了。这项工作突出了HT插入的成功是布局、布线资源、受攻击节点的位置以及进一步的设计特定特征的复杂函数。因此，我们的框架不仅仅是一次攻击，我们提供了最先进的分析工具来评估在最终布局中插入HT的脆弱性。



