# Latest Adversarial Attack Papers
**update at 2022-05-14 06:31:24**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Anomaly Detection of Adversarial Examples using Class-conditional Generative Adversarial Networks**

基于类条件生成对抗性网络的对抗性实例异常检测 cs.LG

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2105.10101v2)

**Authors**: Hang Wang, David J. Miller, George Kesidis

**Abstracts**: Deep Neural Networks (DNNs) have been shown vulnerable to Test-Time Evasion attacks (TTEs, or adversarial examples), which, by making small changes to the input, alter the DNN's decision. We propose an unsupervised attack detector on DNN classifiers based on class-conditional Generative Adversarial Networks (GANs). We model the distribution of clean data conditioned on the predicted class label by an Auxiliary Classifier GAN (AC-GAN). Given a test sample and its predicted class, three detection statistics are calculated based on the AC-GAN Generator and Discriminator. Experiments on image classification datasets under various TTE attacks show that our method outperforms previous detection methods. We also investigate the effectiveness of anomaly detection using different DNN layers (input features or internal-layer features) and demonstrate, as one might expect, that anomalies are harder to detect using features closer to the DNN's output layer.

摘要: 深度神经网络(DNN)已经被证明容易受到测试时间逃避攻击(TTE，或对抗性例子)，这些攻击通过对输入进行微小的改变来改变DNN的决策。提出了一种基于类别条件生成对抗网络(GANS)的DNN分类器的无监督攻击检测器。我们以辅助分类器GaN(AC-GaN)预测的类别标签为条件，对清洁数据的分布进行建模。在给定测试样本及其预测类别的情况下，基于AC-GaN生成器和鉴别器计算了三个检测统计量。在不同TTE攻击下的图像分类数据集上的实验表明，该方法的性能优于以往的检测方法。我们还研究了使用不同的DNN层(输入特征或内部层特征)进行异常检测的有效性，并证明了，正如人们所预期的那样，使用离DNN输出层更近的特征更难检测到异常。



## **2. Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks**

抗规避攻击的稳健学习决策表的样本复杂性界 cs.LG

To appear in the proceedings of International Joint Conference on  Artificial Intelligence (2022)

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06127v1)

**Authors**: Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell

**Abstracts**: A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed $k$ the class of $k$-decision lists has polynomial sample complexity against a $\log(n)$-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient $\log(n)$-robust learning algorithm under the uniform distribution.

摘要: 对抗性机器学习中的一个基本问题是量化在存在逃避攻击的情况下需要多少训练数据。在本文中，我们在PAC学习的框架内解决这个问题，重点是决策列表的类。鉴于分布假设在对抗性环境中是必不可少的，我们在满足Lipschitz条件的输入数据上使用概率分布：邻近的点具有类似的概率。我们的关键结果表明，对手的预算(即，它可以在每一次输入上扰动的比特数)是决定稳健学习的样本复杂性的基本量。我们的第一个主要结果是一个样本复杂性下界：单调合取类(本质上是布尔超立方体上最简单的非平凡假设类)和任何超类在对手的预算中至少具有指数级的样本复杂性。我们的第二个主要结果是相应的上界：对于每一个固定的$k$，对于$\log(N)$有界的对手，这类$k$-决策列表具有多项式样本复杂性。这进一步揭示了在均匀分布下，有效的PAC学习算法是否总是可以用作有效的$\log(N)$稳健学习算法的问题。



## **3. From IP to transport and beyond: cross-layer attacks against applications**

从IP到传输乃至更远：针对应用程序的跨层攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06085v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: We perform the first analysis of methodologies for launching DNS cache poisoning: manipulation at the IP layer, hijack of the inter-domain routing and probing open ports via side channels. We evaluate these methodologies against DNS resolvers in the Internet and compare them with respect to effectiveness, applicability and stealth. Our study shows that DNS cache poisoning is a practical and pervasive threat.   We then demonstrate cross-layer attacks that leverage DNS cache poisoning for attacking popular systems, ranging from security mechanisms, such as RPKI, to applications, such as VoIP. In addition to more traditional adversarial goals, most notably impersonation and Denial of Service, we show for the first time that DNS cache poisoning can even enable adversaries to bypass cryptographic defences: we demonstrate how DNS cache poisoning can facilitate BGP prefix hijacking of networks protected with RPKI even when all the other networks apply route origin validation to filter invalid BGP announcements. Our study shows that DNS plays a much more central role in the Internet security than previously assumed.   We recommend mitigations for securing the applications and for preventing cache poisoning.

摘要: 我们对发起DNS缓存中毒的方法进行了第一次分析：在IP层操纵、劫持域间路由和通过侧通道探测开放端口。我们针对互联网中的域名解析程序对这些方法进行评估，并在有效性、适用性和隐蔽性方面对它们进行比较。我们的研究表明，DNS缓存中毒是一种实际且普遍存在的威胁。然后，我们演示了利用DNS缓存毒化来攻击流行系统的跨层攻击，攻击范围从安全机制(如RPKI)到应用程序(如VoIP)。除了更传统的敌意目标之外，最显著的是模拟和拒绝服务，我们首次展示了DNS缓存中毒甚至可以使攻击者绕过加密防御：我们演示了DNS缓存中毒如何促进对受RPKI保护的网络的BGP前缀劫持，即使所有其他网络都应用路由来源验证来过滤无效的BGP通告。我们的研究表明，在互联网安全中，域名系统扮演的角色比之前设想的要重要得多。我们建议采取缓解措施来保护应用程序和防止缓存中毒。



## **4. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2204.05276v2)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **5. Stalloris: RPKI Downgrade Attack**

Stalloris：RPKI降级攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06064v1)

**Authors**: Tomas Hlavacek, Philipp Jeitner, Donika Mirdita, Haya Shulman, Michael Waidner

**Abstracts**: We demonstrate the first downgrade attacks against RPKI. The key design property in RPKI that allows our attacks is the tradeoff between connectivity and security: when networks cannot retrieve RPKI information from publication points, they make routing decisions in BGP without validating RPKI. We exploit this tradeoff to develop attacks that prevent the retrieval of the RPKI objects from the public repositories, thereby disabling RPKI validation and exposing the RPKI-protected networks to prefix hijack attacks.   We demonstrate experimentally that at least 47% of the public repositories are vulnerable against a specific version of our attacks, a rate-limiting off-path downgrade attack. We also show that all the current RPKI relying party implementations are vulnerable to attacks by a malicious publication point. This translates to 20.4% of the IPv4 address space.   We provide recommendations for preventing our downgrade attacks. However, resolving the fundamental problem is not straightforward: if the relying parties prefer security over connectivity and insist on RPKI validation when ROAs cannot be retrieved, the victim AS may become disconnected from many more networks than just the one that the adversary wishes to hijack. Our work shows that the publication points are a critical infrastructure for Internet connectivity and security. Our main recommendation is therefore that the publication points should be hosted on robust platforms guaranteeing a high degree of connectivity.

摘要: 我们演示了针对RPKI的第一次降级攻击。RPKI中允许我们攻击的关键设计属性是连接性和安全性之间的权衡：当网络无法从发布点检索RPKI信息时，它们在BGP中做出路由决定，而不验证RPKI。我们利用这一权衡来开发攻击，以阻止从公共存储库中检索RPKI对象，从而禁用RPKI验证并使受RPKI保护的网络暴露于前缀劫持攻击。我们通过实验证明，至少47%的公共存储库容易受到我们的特定版本的攻击，这是一种限速的非路径降级攻击。我们还表明，所有当前的RPKI依赖方实现都容易受到恶意发布点的攻击。这相当于IPv4地址空间的20.4%。我们提供了防止降级攻击的建议。然而，解决根本问题并不简单：如果依赖方更看重安全而不是连接，并在无法检索ROA时坚持RPKI验证，受害者AS可能会断开与更多网络的连接，而不仅仅是对手希望劫持的网络。我们的工作表明，发布点是互联网连接和安全的关键基础设施。因此，我们的主要建议是，发布点应设在保证高度连通性的强大平台上。



## **6. Infrared Invisible Clothing:Hiding from Infrared Detectors at Multiple Angles in Real World**

红外隐身衣：在现实世界中从多个角度躲避红外探测器 cs.CV

Accepted by CVPR 2022, ORAL

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.05909v1)

**Authors**: Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, Xiaolin Hu

**Abstracts**: Thermal infrared imaging is widely used in body temperature measurement, security monitoring, and so on, but its safety research attracted attention only in recent years. We proposed the infrared adversarial clothing, which could fool infrared pedestrian detectors at different angles. We simulated the process from cloth to clothing in the digital world and then designed the adversarial "QR code" pattern. The core of our method is to design a basic pattern that can be expanded periodically, and make the pattern after random cropping and deformation still have an adversarial effect, then we can process the flat cloth with an adversarial pattern into any 3D clothes. The results showed that the optimized "QR code" pattern lowered the Average Precision (AP) of YOLOv3 by 87.7%, while the random "QR code" pattern and blank pattern lowered the AP of YOLOv3 by 57.9% and 30.1%, respectively, in the digital world. We then manufactured an adversarial shirt with a new material: aerogel. Physical-world experiments showed that the adversarial "QR code" pattern clothing lowered the AP of YOLOv3 by 64.6%, while the random "QR code" pattern clothing and fully heat-insulated clothing lowered the AP of YOLOv3 by 28.3% and 22.8%, respectively. We used the model ensemble technique to improve the attack transferability to unseen models.

摘要: 热红外成像广泛应用于体温测量、安防监测等领域，但其安全性研究直到最近几年才引起人们的重视。我们提出了红外防御服，可以从不同角度欺骗红外行人探测器。我们模拟了数字世界中从布料到衣物的过程，然后设计了对抗性的“二维码”图案。该方法的核心是设计一种可周期性扩展的基本图案，使任意裁剪和变形后的图案仍然具有对抗效果，从而可以将带有对抗图案的平面布加工成任何3D服装。结果表明，在数字世界中，优化的二维码模式使YOLOv3的平均准确率下降了87.7%，而随机的二维码模式和空白模式分别使YOLOv3的平均准确率下降了57.9%和30.1%。然后，我们用一种新材料制作了一件对抗性衬衫：气凝胶。实物实验表明，对抗性的“二维码”图案服装使YOLOv3的AP降低了64.6%，而随机的“二维码”图案服装和完全隔热的服装分别使YOLOv3的AP降低了28.3%和22.8%。我们使用模型集成技术来提高攻击到不可见模型的可转移性。



## **7. Using Frequency Attention to Make Adversarial Patch Powerful Against Person Detector**

利用频率注意使对抗性补丁成为强大的抗人检测器 cs.CV

10pages, 4 figures

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.04638v2)

**Authors**: Xiaochun Lei, Chang Lu, Zetao Jiang, Zhaoting Gong, Xiang Cai, Linjun Lu

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, object detectors may be attacked by applying a particular adversarial patch to the image. However, because the patch shrinks during preprocessing, most existing approaches that employ adversarial patches to attack object detectors would diminish the attack success rate on small and medium targets. This paper proposes a Frequency Module(FRAN), a frequency-domain attention module for guiding patch generation. This is the first study to introduce frequency domain attention to optimize the attack capabilities of adversarial patches. Our method increases the attack success rates of small and medium targets by 4.18% and 3.89%, respectively, over the state-of-the-art attack method for fooling the human detector while assaulting YOLOv3 without reducing the attack success rate of big targets.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。具体地，可以通过将特定的敌意补丁应用于图像来攻击对象检测器。然而，由于补丁在预处理过程中会缩小，现有的大多数使用对抗性补丁攻击目标检测器的方法都会降低对中小目标的攻击成功率。提出了一种用于指导补丁生成的频域注意模块FRAN。这是首次引入频域注意力来优化敌方补丁攻击能力的研究。该方法在不降低大目标攻击成功率的前提下，将小目标和中型目标的攻击成功率分别提高了4.18%和3.89%。



## **8. The Hijackers Guide To The Galaxy: Off-Path Taking Over Internet Resources**

《银河劫机者指南：越轨接管互联网资源》 cs.CR

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.05473v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: Internet resources form the basic fabric of the digital society. They provide the fundamental platform for digital services and assets, e.g., for critical infrastructures, financial services, government. Whoever controls that fabric effectively controls the digital society.   In this work we demonstrate that the current practices of Internet resources management, of IP addresses, domains, certificates and virtual platforms are insecure. Over long periods of time adversaries can maintain control over Internet resources which they do not own and perform stealthy manipulations, leading to devastating attacks. We show that network adversaries can take over and manipulate at least 68% of the assigned IPv4 address space as well as 31% of the top Alexa domains. We demonstrate such attacks by hijacking the accounts associated with the digital resources.   For hijacking the accounts we launch off-path DNS cache poisoning attacks, to redirect the password recovery link to the adversarial hosts. We then demonstrate that the adversaries can manipulate the resources associated with these accounts. We find all the tested providers vulnerable to our attacks.   We recommend mitigations for blocking the attacks that we present in this work. Nevertheless, the countermeasures cannot solve the fundamental problem - the management of the Internet resources should be revised to ensure that applying transactions cannot be done so easily and stealthily as is currently possible.

摘要: 互联网资源构成了数字社会的基本结构。它们为数字服务和资产提供基础平台，例如关键基础设施、金融服务、政府。无论谁控制了这种结构，谁就有效地控制了数字社会。在这项工作中，我们证明了当前互联网资源管理的做法，即IP地址、域、证书和虚拟平台是不安全的。在很长一段时间内，对手可以保持对他们不拥有的互联网资源的控制，并执行秘密操作，导致毁灭性的攻击。我们发现，网络攻击者可以接管和操纵至少68%的分配的IPv4地址空间以及31%的顶级Alexa域。我们通过劫持与数字资源相关的帐户来演示此类攻击。对于劫持帐户，我们发起非路径的DNS缓存中毒攻击，将密码恢复链接重定向到恶意主机。然后，我们将演示对手可以操纵与这些帐户关联的资源。我们发现所有经过测试的供应商都容易受到我们的攻击。我们建议采取缓解措施来阻止我们在此工作中提出的攻击。然而，这些对策不能解决根本问题--对互联网资源的管理应加以修订，以确保申请交易不能像目前那样容易和秘密地进行。



## **9. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2204.08189v2)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **10. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

利用计算机视觉技术开发隐形敌方补丁来伪装军事资产 cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2202.08892v2)

**Authors**: Chris Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.

摘要: 卷积神经网络(CNN)在目标检测方面取得了快速的进展和很高的成功率。然而，最近的证据突显了它们在对抗性攻击中的脆弱性。这些攻击是经过计算的图像扰动或对抗性补丁，导致目标错误分类或检测抑制。传统的伪装方法用于伪装飞机和其他大型机动资产，使其免受情报、监视和侦察技术以及第五代导弹的自主探测，是不切实际的。在这篇文章中，我们提出了一种独特的方法，可以从计算机视觉启用的技术中产生能够伪装大型军事资产的隐形补丁。我们开发了这些补丁，通过最大化目标检测损失，同时限制补丁的颜色敏感度。这项工作还旨在进一步理解对抗性例子及其对目标检测算法的影响。



## **11. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

一句话抵得上一千美元：敌意攻击推特傻瓜股预测 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.01094v2)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **12. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

对基于投影的可取消生物识别方案的身份验证攻击(长版) cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2110.15163v4)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **13. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

SYNFI：一种开源安全元件的硅前故障分析 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.04775v1)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.

摘要: 故障攻击是一种主动的物理攻击，攻击者可以利用这些攻击来改变嵌入式设备的控制流，从而获得对敏感信息的访问权限或绕过保护机制。由于这些攻击的严重性，制造商将基于硬件的故障防御部署到安全关键系统中，例如安全元件。这些对策的开发是一项具有挑战性的任务，因为电路元件之间的复杂相互作用，以及现代设计自动化工具倾向于优化插入的结构，从而违背了它们的目的。因此，至关重要的是，这些对策在合成后得到严格验证。由于传统的功能验证技术无法评估对策的有效性，开发人员不得不求助于能够在模拟测试台或物理芯片中注入故障的方法。然而，开发测试序列以在模拟中注入故障是一项容易出错的任务，在芯片上执行故障攻击需要专门的设备，并且非常耗时。为此，本文引入了SYNFI，这是一个运行在合成网表上的形式化的预硅故障验证框架。SYNFI可以用来分析故障对电路输入输出关系的一般影响及其故障对策，从而使硬件设计者能够以系统和半自动的方式评估和验证嵌入式对策的有效性。为了证明SYNFI能够处理使用商业和开放工具合成的未经修改的工业级网表，我们分析了第一个开源安全元素OpenTitan。在我们的分析中，我们确定了未受保护的AES块中的关键安全漏洞，开发了有针对性的对策，重新评估了它们的安全性，并将这些对策贡献给了OpenTitan存储库。



## **14. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

基于后向误差分析的联合学习半目标模型中毒攻击 cs.LG

Published in IJCNN 2022

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.11633v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.

摘要: 针对联邦学习(FL)的模型中毒攻击通过破坏边缘模型来侵入整个系统，导致机器学习模型故障。这种被破坏的模型被篡改，以执行对手所希望的行为。特别是，我们考虑了一种半目标的情况，其中源类是预先确定的，而目标类不是。其目的是使全局分类器对源类的数据进行错误分类。虽然已经采用了标签翻转等方法向FL注入有毒参数，但研究表明，它们的性能通常是类敏感的，随着所使用的目标类的不同而变化。通常，当转移到不同的目标类别时，攻击可能会变得不那么有效。为了克服这一挑战，我们提出了攻击距离感知攻击(ADA)，通过在特征空间中找到优化的目标类来增强中毒攻击。此外，我们研究了一种更具挑战性的情况，即对手对客户数据的先验知识有限。为了解决这一问题，ADA基于向后误差分析，从共享的模型参数中推导出潜在特征空间中不同类别之间的成对距离。我们通过在三种不同的图像分类任务中改变攻击频率的因素，对ADA进行了广泛的经验评估。结果，在最具挑战性的情况下，ADA成功地将攻击性能提高了1.8倍，攻击频率为0.01。



## **15. Fingerprinting of DNN with Black-box Design and Verification**

基于黑盒设计和验证的DNN指纹识别 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.10902v3)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Ruoxi Sun, Minhui Xue, Surya Nepal, Seyit Camtepe, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.

摘要: 支持云的机器学习即服务(MLaaS)显示出巨大的潜力，可以改变深度学习模型的开发和部署方式。尽管如此，使用此类服务仍存在潜在风险，因为恶意方可能会对其进行修改以达到不利的结果。因此，模型所有者、服务提供商和最终用户必须验证部署的模型是否未被篡改。这样的验证需要公开的可验证性(即，指纹模式可供各方使用，包括对手)，并需要通过API对部署的模型进行黑盒访问。然而，现有的水印和指纹方法需要白盒知识(如梯度)来设计指纹，并且只支持私密可验证性，即由诚实的一方进行验证。在本文中，我们描述了一种实用的水印技术，该技术能够在指纹设计中提供黑盒知识，并在验证过程中提供黑盒查询。该服务通过公开验证来确保基于云的服务的完整性(即指纹模式可供各方使用，包括对手)。如果对手操纵了一个模型，这将导致决策边界的转变。因此，双黑水印的基本原理是，模型的决策边界可以作为水印的固有指纹。我们的方法通过生成有限数量的包络样本指纹来捕获决策边界，这些样本指纹是围绕模型决策边界的一组自然转换和扩充的输入，以捕获模型的固有指纹。我们针对各种模型完整性攻击和模型压缩攻击对我们的水印方法进行了评估。



## **16. Energy-bounded Learning for Robust Models of Code**

代码健壮模型的能量受限学习 cs.LG

There are some flaws in our experiments, we would like to fix it and  publish a fixed version again in the very near future

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.11226v2)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.

摘要: 在编程中，学习代码表示法有多种应用，包括代码分类、代码搜索、注释生成、错误预测等。已经提出了关于令牌、语法树、依赖图、代码导航路径或其变体的组合的代码的各种表示，然而，现有的普通学习技术在稳健性方面具有主要限制，即，当输入以微妙的方式改变时，模型容易做出不正确的预测。为了增强鲁棒性，现有的方法侧重于识别敌意样本，而不是识别属于给定分布之外的有效样本，我们称之为分布外(OOD)样本。识别这类面向对象的样本是本文研究的新问题。为此，我们建议首先用分布外样本来扩充In=分布数据集，以便当它们一起训练时，将增强模型的稳健性。我们提出使用一个能量受限的学习目标函数来给分布内样本赋予较高的分数，而对分布外样本赋予较低的分数，以便将这种分布外样本纳入源代码模型的训练过程。在OOD检测和敌意样本检测方面，我们的评估结果表明，现有的源代码模型在更准确地识别OOD数据的同时，更能抵抗敌意攻击，具有更强的鲁棒性。此外，所提出的能量受限分数在很大程度上超过了所有现有的OOD检测分数，包括Softmax置信度分数、马氏分数和ODIN。



## **17. Do You Think You Can Hold Me? The Real Challenge of Problem-Space Evasion Attacks**

你觉得你能抱住我吗？问题空间规避攻击的真正挑战 cs.CR

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04293v1)

**Authors**: Harel Berger, Amit Dvir, Chen Hajaj, Rony Ronen

**Abstracts**: Android malware is a spreading disease in the virtual world. Anti-virus and detection systems continuously undergo patches and updates to defend against these threats. Most of the latest approaches in malware detection use Machine Learning (ML). Against the robustifying effort of detection systems, raise the \emph{evasion attacks}, where an adversary changes its targeted samples so that they are misclassified as benign. This paper considers two kinds of evasion attacks: feature-space and problem-space. \emph{Feature-space} attacks consider an adversary who manipulates ML features to evade the correct classification while minimizing or constraining the total manipulations. \textit{Problem-space} attacks refer to evasion attacks that change the actual sample. Specifically, this paper analyzes the gap between these two types in the Android malware domain. The gap between the two types of evasion attacks is examined via the retraining process of classifiers using each one of the evasion attack types. The experiments show that the gap between these two types of retrained classifiers is dramatic and may increase to 96\%. Retrained classifiers of feature-space evasion attacks have been found to be either less effective or completely ineffective against problem-space evasion attacks. Additionally, exploration of different problem-space evasion attacks shows that retraining of one problem-space evasion attack may be effective against other problem-space evasion attacks.

摘要: Android恶意软件是一种在虚拟世界中传播的疾病。反病毒和检测系统不断进行补丁和更新，以防御这些威胁。大多数最新的恶意软件检测方法都使用机器学习(ML)。针对检测系统的粗暴努力，提出\emph{躲避攻击}，其中对手更改其目标样本，以便将其错误分类为良性。本文考虑了两种规避攻击：特征空间和问题空间。EMPH{Feature-space}攻击考虑对手操纵ML特征来逃避正确分类，同时最小化或约束总的操纵次数。\textit{问题空间}攻击是指更改实际样本的规避攻击。具体地说，本文分析了这两种类型在Android恶意软件领域的差距。通过使用每一种逃避攻击类型的分类器的重新训练过程来检查这两种类型的逃避攻击之间的差距。实验表明，这两种重新训练的分类器之间的差距很大，可能会增加到96%。重新训练的特征空间逃避攻击分类器被发现对问题空间逃避攻击不是很有效，就是完全无效。此外，对不同问题空间逃避攻击的研究表明，对一种问题空间逃避攻击进行再训练可能对其他问题空间逃避攻击有效。



## **18. Federated Multi-Armed Bandits Under Byzantine Attacks**

拜占庭式攻击下的联邦多臂土匪 cs.LG

13 pages, 15 figures

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04134v1)

**Authors**: Ilker Demirel, Yigit Yildirim, Cem Tekin

**Abstracts**: Multi-armed bandits (MAB) is a simple reinforcement learning model where the learner controls the trade-off between exploration versus exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is a recently emerging framework where a cohort of learners with heterogeneous local models play a MAB game and communicate their aggregated feedback to a parameter server to learn the global feedback model. Federated learning models are vulnerable to adversarial attacks such as model-update attacks or data poisoning. In this work, we study an FMAB problem in the presence of Byzantine clients who can send false model updates that pose a threat to the learning process. We borrow tools from robust statistics and propose a median-of-means-based estimator: Fed-MoM-UCB, to cope with the Byzantine clients. We show that if the Byzantine clients constitute at most half the cohort, it is possible to incur a cumulative regret on the order of ${\cal O} (\log T)$ with respect to an unavoidable error margin, including the communication cost between the clients and the parameter server. We analyze the interplay between the algorithm parameters, unavoidable error margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.

摘要: 多臂强盗(MAB)是一种简单的强化学习模型，学习者控制探索和剥削之间的权衡，以最大化其累积回报。联邦多臂强盗(FMAB)是一种新出现的框架，在该框架中，具有不同局部模型的一群学习者玩MAB游戏，并将他们聚集的反馈传递给参数服务器以学习全局反馈模型。联合学习模型容易受到敌意攻击，如模型更新攻击或数据中毒。在这项工作中，我们研究了在拜占庭客户端存在的情况下的FMAB问题，这些客户端可能会发送虚假的模型更新，从而对学习过程构成威胁。我们借用稳健统计中的工具，提出了一种基于均值中位数的估计量：FED-MOM-UCB，以应对拜占庭式的客户。我们证明，如果拜占庭客户端至多构成队列的一半，则对于不可避免的误差容限(包括客户端和参数服务器之间的通信成本)，有可能产生大约${\cal O}(\logT)$的累积遗憾。我们分析了算法参数、不可避免的误差率、遗憾、通信开销和ARM的次优差距之间的相互影响。我们通过实验证明了在拜占庭攻击存在的情况下，FED-MOM-UCB相对于基线的有效性。



## **19. ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning**

ResSFL：一种抵抗分裂联邦学习中模型反转攻击的阻力转移框架 cs.LG

Accepted to CVPR 2022

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04007v1)

**Authors**: Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, Chaitali Chakrabarti

**Abstracts**: This work aims to tackle Model Inversion (MI) attack on Split Federated Learning (SFL). SFL is a recent distributed training scheme where multiple clients send intermediate activations (i.e., feature map), instead of raw data, to a central server. While such a scheme helps reduce the computational load at the client end, it opens itself to reconstruction of raw data from intermediate activation by the server. Existing works on protecting SFL only consider inference and do not handle attacks during training. So we propose ResSFL, a Split Federated Learning Framework that is designed to be MI-resistant during training. It is based on deriving a resistant feature extractor via attacker-aware training, and using this extractor to initialize the client-side model prior to standard SFL training. Such a method helps in reducing the computational complexity due to use of strong inversion model in client-side adversarial training as well as vulnerability of attacks launched in early training epochs. On CIFAR-100 dataset, our proposed framework successfully mitigates MI attack on a VGG-11 model with a high reconstruction Mean-Square-Error of 0.050 compared to 0.005 obtained by the baseline system. The framework achieves 67.5% accuracy (only 1% accuracy drop) with very low computation overhead. Code is released at: https://github.com/zlijingtao/ResSFL.

摘要: 该工作旨在解决分裂联邦学习(SFL)上的模型反转(MI)攻击。SFL是最近的分布式训练方案，其中多个客户端将中间激活(即，特征地图)而不是原始数据发送到中央服务器。虽然这样的方案有助于减少客户端的计算负荷，但它本身也允许服务器从中间激活重建原始数据。现有的保护SFL的工作只考虑推理，不处理训练过程中的攻击。因此，我们提出了一种分离的联邦学习框架ResSFL，该框架被设计为在训练过程中抵抗MI。它的基础是通过攻击者感知训练派生出抵抗特征提取器，并在标准的SFL训练之前使用该提取器来初始化客户端模型。这种方法有助于降低在客户端对抗性训练中使用强反转模型所带来的计算复杂性，以及在早期训练期发起的攻击的脆弱性。在CIFAR-100数据集上，我们的框架成功地缓解了对VGG-11模型的MI攻击，重建均方误差为0.050，而基线系统的重建均方误差为0.005。该框架以很低的计算开销获得了67.5%的准确率(仅1%的准确率下降)。代码发布地址：https://github.com/zlijingtao/ResSFL.



## **20. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

三角攻击：一种查询高效的基于决策的对抗性攻击 cs.CV

10 pages

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.06569v2)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.

摘要: 基于决策的攻击将目标模型视为黑匣子，只访问硬预测标签，对现实世界的应用构成了严重威胁。最近已经做出了很大的努力来减少查询的数量；然而，现有的基于决策的攻击仍然需要数千个查询才能生成高质量的对抗性例子。在这项工作中，我们发现一个良性样本、当前和下一个对抗性样本可以自然地在子空间中为任何迭代攻击构造一个三角形。基于正弦定律，提出了一种新的三角形攻击算法(TA)，该算法利用任意三角形中长边总是与较大角相对的几何信息来优化扰动。然而，直接将这些信息应用于输入图像是无效的，因为它不能在高维空间中彻底探索输入样本的邻域。为了解决这个问题，由于这种几何性质的普遍性，TA优化了低频空间中的扰动，以实现有效的降维。对ImageNet数据集的广泛评估表明，与现有的基于决策的攻击相比，TA在1000个查询中实现了更高的攻击成功率，并且在各种扰动预算下需要更少的查询才能达到相同的攻击成功率。在如此高的效率下，我们进一步证明了TA在现实世界的API上的适用性，即腾讯云API。



## **21. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

私人眼睛：视频会议中通过眼镜反射窥视文本屏幕的限度 cs.CR

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2205.03971v1)

**Authors**: Yan Long, Chen Yan, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Personal video conferencing has become the new norm after COVID-19 caused a seismic shift from in-person meetings and phone calls to video conferencing for daily communications and sensitive business. Video leaks participants' on-screen information because eyeglasses and other reflective objects unwittingly expose partial screen contents. Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual information gleamed from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our experimental results and models show it is possible to reconstruct and recognize on-screen text with a height as small as 10 mm with a 720p webcam. We further apply this threat model to web textual content with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution toward 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Our research proposes near-term mitigations, and justifies the importance of following the principle of least privilege for long-term defense against this attack. For privacy-sensitive scenarios, it's further recommended to develop technologies that blur all objects by default, then only unblur what is absolutely necessary to facilitate natural-looking conversations.

摘要: 在新冠肺炎引发了从面对面会议和电话到用于日常交流和敏感事务的视频会议的巨变之后，个人视频会议已成为新的常态。视频会泄露参与者的屏幕信息，因为眼镜和其他反光物体在不知不觉中暴露了部分屏幕内容。通过数学建模和人体实验，这项研究探索了新兴的网络摄像头可能在多大程度上泄露从网络摄像头捕捉到的眼镜反射中闪烁的可识别的文本信息。我们工作的主要目标是测量、计算和预测随着未来网络摄像头技术的发展而产生的可识别性的因素、限制和阈值。我们的工作利用视频帧序列上的多帧超分辨率技术，探索和表征了基于光学攻击的可行威胁模型。我们的实验结果和模型表明，使用720p网络摄像头可以重建和识别高度低至10 mm的屏幕文本。我们进一步将此威胁模型应用于具有不同攻击者能力的Web文本内容，以找出文本变得可识别的阈值。我们对20名参与者的用户研究表明，目前的720p网络摄像头足以让对手在大字体网站上重建文本内容。我们的模型进一步表明，向4K摄像头的演变将使文本泄漏的门槛倾斜到重建流行网站上的大多数标题文本。我们的研究提出了近期缓解措施，并证明了遵循最小特权原则对长期防御这种攻击的重要性。对于隐私敏感的场景，进一步建议开发默认模糊所有对象的技术，然后只对绝对必要的内容进行模糊处理，以促进看起来自然的对话。



## **22. mFI-PSO: A Flexible and Effective Method in Adversarial Image Generation for Deep Neural Networks**

MFI-PSO：一种灵活有效的深度神经网络对抗性图像生成方法 cs.LG

Accepted by 2022 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2006.03243v3)

**Authors**: Hai Shu, Ronghua Shi, Qiran Jia, Hongtu Zhu, Ziqi Chen

**Abstracts**: Deep neural networks (DNNs) have achieved great success in image classification, but can be very vulnerable to adversarial attacks with small perturbations to images. To improve adversarial image generation for DNNs, we develop a novel method, called mFI-PSO, which utilizes a Manifold-based First-order Influence measure for vulnerable image and pixel selection and the Particle Swarm Optimization for various objective functions. Our mFI-PSO can thus effectively design adversarial images with flexible, customized options on the number of perturbed pixels, the misclassification probability, and the targeted incorrect class. Experiments demonstrate the flexibility and effectiveness of our mFI-PSO in adversarial attacks and its appealing advantages over some popular methods.

摘要: 深度神经网络(DNN)在图像分类方面取得了很大的成功，但对图像的扰动很小，很容易受到敌意攻击。为了改进DNN的敌意图像生成，我们提出了一种新的方法，称为MFI-PSO，它利用基于流形的一阶影响度量来选择易受攻击的图像和像素，并使用粒子群优化算法来选择各种目标函数。因此，我们的MFI-PSO可以有效地设计对抗性图像，具有灵活的定制选项，包括扰动像素数、误分类概率和目标错误类别。实验证明了MFI-PSO在对抗性攻击中的灵活性和有效性，以及它相对于一些流行方法的优势。



## **23. IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection**

IDSGAN：针对入侵检测的产生式攻击生成对抗网络 cs.CR

Accepted for publication in the 26th Pacific-Asia Conference on  Knowledge Discovery and Data Mining (PAKDD 2022)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/1809.02077v5)

**Authors**: Zilong Lin, Yong Shi, Zhi Xue

**Abstracts**: As an essential tool in security, the intrusion detection system bears the responsibility of the defense to network attacks performed by malicious traffic. Nowadays, with the help of machine learning algorithms, intrusion detection systems develop rapidly. However, the robustness of this system is questionable when it faces adversarial attacks. For the robustness of detection systems, more potential attack approaches are under research. In this paper, a framework of the generative adversarial networks, called IDSGAN, is proposed to generate the adversarial malicious traffic records aiming to attack intrusion detection systems by deceiving and evading the detection. Given that the internal structure and parameters of the detection system are unknown to attackers, the adversarial attack examples perform the black-box attacks against the detection system. IDSGAN leverages a generator to transform original malicious traffic records into adversarial malicious ones. A discriminator classifies traffic examples and dynamically learns the real-time black-box detection system. More significantly, the restricted modification mechanism is designed for the adversarial generation to preserve original attack functionalities of adversarial traffic records. The effectiveness of the model is indicated by attacking multiple algorithm-based detection models with different attack categories. The robustness is verified by changing the number of the modified features. A comparative experiment with adversarial attack baselines demonstrates the superiority of our model.

摘要: 入侵检测系统作为一种必不可少的安全工具，担负着防御恶意流量进行的网络攻击的重任。如今，在机器学习算法的帮助下，入侵检测系统得到了迅速发展。然而，当该系统面临敌意攻击时，其健壮性是值得怀疑的。对于检测系统的健壮性，更多的潜在攻击方法正在研究中。提出了一种产生式恶意流量记录生成框架IDSGAN，用于生成恶意流量记录，通过欺骗和逃避检测来攻击入侵检测系统。在攻击者未知检测系统内部结构和参数的情况下，对抗性攻击实例对检测系统进行黑盒攻击。IDSGAN利用生成器将原始恶意流量记录转换为对抗性恶意流量记录。鉴别器对流量样本进行分类，动态学习实时黑匣子检测系统。更重要的是，受限修改机制是为对抗性生成而设计的，以保留对抗性流量记录的原始攻击功能。通过对不同攻击类别的多个基于算法的检测模型的攻击，验证了该模型的有效性。通过改变修改后的特征数来验证算法的稳健性。通过与对抗性攻击基线的对比实验，验证了该模型的优越性。



## **24. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于全局对抗性扰动的深度神经网络指纹识别 cs.CR

Accepted to CVPR 2022 (Oral Presentation)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2202.08602v3)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its Universal Adversarial Perturbations (UAPs). UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via contrastive learning that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence > 99.99 within only 20 fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 在本文中，我们提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是，DNN模型的决策边界的轮廓可以唯一地由其通用对抗性扰动(UAP)来表征。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并通过对比学习训练编码者，以指纹为输入，输出相似度分数。大量的研究表明，我们的框架可以在可疑模型的20个指纹中检测到模型IP违规行为，置信度>99.99。它在不同的模型体系结构上具有良好的通用性，并且对被盗模型的后期修改具有健壮性。



## **25. Poisoning Semi-supervised Federated Learning via Unlabeled Data: Attacks and Defenses**

利用未标记数据毒化半监督联合学习：攻击与防御 cs.LG

Updated Version

**SubmitDate**: 2022-05-07    [paper-pdf](http://arxiv.org/pdf/2012.04432v2)

**Authors**: Yi Liu, Xingliang Yuan, Ruihui Zhao, Cong Wang, Dusit Niyato, Yefeng Zheng

**Abstracts**: Semi-supervised Federated Learning (SSFL) has recently drawn much attention due to its practical consideration, i.e., the clients may only have unlabeled data. In practice, these SSFL systems implement semi-supervised training by assigning a "guessed" label to the unlabeled data near the labeled data to convert the unsupervised problem into a fully supervised problem. However, the inherent properties of such semi-supervised training techniques create a new attack surface. In this paper, we discover and reveal a simple yet powerful poisoning attack against SSFL. Our attack utilizes the natural characteristic of semi-supervised learning to cause the model to be poisoned by poisoning unlabeled data. Specifically, the adversary just needs to insert a small number of maliciously crafted unlabeled samples (e.g., only 0.1\% of the dataset) to infect model performance and misclassification. Extensive case studies have shown that our attacks are effective on different datasets and common semi-supervised learning methods. To mitigate the attacks, we propose a defense, i.e., a minimax optimization-based client selection strategy, to enable the server to select the clients who hold the correct label information and high-quality updates. Our defense further employs a quality-based aggregation rule to strengthen the contributions of the selected updates. Evaluations under different attack conditions show that the proposed defense can well alleviate such unlabeled poisoning attacks. Our study unveils the vulnerability of SSFL to unlabeled poisoning attacks and provides the community with potential defense methods.

摘要: 半监督联合学习(SSFL)由于其实用性，即客户端可能只有未标记的数据，近年来引起了人们的广泛关注。在实际应用中，这些SSFL系统通过给标记数据附近的未标记数据分配一个“猜测”标签来实现半监督训练，从而将无监督问题转化为完全监督问题。然而，这种半监督训练技术的固有特性创造了一个新的攻击面。在本文中，我们发现并揭示了一种简单而强大的针对SSFL的中毒攻击。我们的攻击利用了半监督学习的自然特性，通过毒化未标记的数据来使模型中毒。具体地说，攻击者只需插入少量恶意制作的未标记样本(例如，仅占数据集的0.1%)即可影响模型性能和错误分类。大量的案例研究表明，我们的攻击在不同的数据集和常见的半监督学习方法上都是有效的。为了缓解攻击，我们提出了一种防御策略，即基于极小极大优化的客户端选择策略，使服务器能够选择拥有正确标签信息和高质量更新的客户端。我们的辩护进一步采用了基于质量的聚合规则来加强选定更新的贡献。在不同攻击条件下的评估表明，所提出的防御方案能够很好地缓解这种未标记的中毒攻击。我们的研究揭示了SSFL对未标记中毒攻击的脆弱性，并为社区提供了潜在的防御方法。



## **26. Using cyber threat intelligence to support adversary understanding applied to the Russia-Ukraine conflict**

利用网络威胁情报支持对手理解在俄乌冲突中的应用 cs.CR

in Spanish language

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03469v1)

**Authors**: Oscar Sandoval Carlos

**Abstracts**: In military organizations, Cyber Threat Intelligence (CTI) supports cyberspace operations by providing the commander with essential information about the adversary, their capabilities and objectives as they operate through cyberspace. This paper, combines CTI with the MITRE ATT&CK framework in order to establish an adversary profile. In addition, it identifies the characteristics of the attack phase by analyzing the WhisperGate operation that occurred in Ukraine in January 2022, and suggests the minimum essential measures for defense.

摘要: 在军事组织中，网络威胁情报(CTI)通过向指挥官提供有关对手的基本信息、他们在网络空间中行动的能力和目标来支持网络空间行动。本文将CTI与MITRE的ATT&CK框架相结合，以建立一个对手档案。此外，通过对2022年1月发生在乌克兰的WhisperGate行动的分析，确定了攻击阶段的特征，并提出了防御的最低必要措施。



## **27. Subverting Fair Image Search with Generative Adversarial Perturbations**

利用生成性对抗性扰动颠覆公平图像搜索 cs.LG

Accepted as a full paper at the 2022 ACM Conference on Fairness,  Accountability, and Transparency (FAccT 22)

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02414v2)

**Authors**: Avijit Ghosh, Matthew Jagielski, Christo Wilson

**Abstracts**: In this work we explore the intersection fairness and robustness in the context of ranking: when a ranking model has been calibrated to achieve some definition of fairness, is it possible for an external adversary to make the ranking model behave unfairly without having access to the model or training data? To investigate this question, we present a case study in which we develop and then attack a state-of-the-art, fairness-aware image search engine using images that have been maliciously modified using a Generative Adversarial Perturbation (GAP) model. These perturbations attempt to cause the fair re-ranking algorithm to unfairly boost the rank of images containing people from an adversary-selected subpopulation.   We present results from extensive experiments demonstrating that our attacks can successfully confer significant unfair advantage to people from the majority class relative to fairly-ranked baseline search results. We demonstrate that our attacks are robust across a number of variables, that they have close to zero impact on the relevance of search results, and that they succeed under a strict threat model. Our findings highlight the danger of deploying fair machine learning algorithms in-the-wild when (1) the data necessary to achieve fairness may be adversarially manipulated, and (2) the models themselves are not robust against attacks.

摘要: 在这项工作中，我们探讨了排名上下文中的交集公平性和稳健性：当一个排名模型已经被校准以实现某种公平性定义时，外部对手是否可能在没有访问该模型或训练数据的情况下使该排名模型表现不公平？为了研究这个问题，我们提供了一个案例研究，在这个案例中，我们开发了一个最先进的、公平感知的图像搜索引擎，然后使用使用生成性对抗扰动(GAP)模型被恶意修改的图像来攻击该搜索引擎。这些扰动试图导致公平重新排序算法不公平地提升包含来自对手选择的子群体的人的图像的排名。我们给出了大量实验的结果，表明我们的攻击可以成功地向大多数类别的人提供相对于排名公平的基线搜索结果的显著不公平优势。我们证明，我们的攻击在许多变量中都是健壮的，它们对搜索结果的相关性几乎没有影响，并且它们在严格的威胁模型下成功。我们的发现突显了在以下情况下部署公平机器学习算法的危险：(1)实现公平所需的数据可能被恶意操纵，以及(2)模型本身对攻击没有健壮性。



## **28. Leveraging strategic connection migration-powered traffic splitting for privacy**

利用战略性连接迁移支持的流量拆分保护隐私 cs.CR

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03326v1)

**Authors**: Mona Wang, Anunay Kulshrestha, Liang Wang, Prateek Mittal

**Abstracts**: Network-level adversaries have developed increasingly sophisticated techniques to surveil and control users' network traffic. In this paper, we exploit our observation that many encrypted protocol connections are no longer tied to device IP address (e.g., the connection migration feature in QUIC, or IP roaming in WireGuard and Mosh), due to the need for performance in a mobile-first world. We design and implement a novel framework, Connection Migration Powered Splitting (CoMPS), that utilizes these performance features for enhancing user privacy. With CoMPS, we can split traffic mid-session across network paths and heterogeneous network protocols. Such traffic splitting mitigates the ability of a network-level adversary to perform traffic analysis attacks by limiting the amount of traffic they can observe. We use CoMPS to construct a website fingerprinting defense that is resilient against traffic analysis attacks by a powerful adaptive adversary in the open-world setting. We evaluate our system using both simulated splitting data and real-world traffic that is actively split using CoMPS. In our real-world experiments, CoMPS reduces the precision and recall of VarCNN to 29.9% and 36.7% respectively in the open-world setting with 100 monitored classes. CoMPS is not only immediately deployable with any unaltered server that supports connection migration, but also incurs little overhead, decreasing throughput by only 5-20%.

摘要: 网络级的对手已经开发出越来越复杂的技术来监视和控制用户的网络流量。在本文中，我们利用我们观察到的许多加密协议连接不再绑定到设备IP地址(例如，Quic中的连接迁移功能，或WireGuard和Mosh中的IP漫游)的观察结果，这是因为在移动优先的世界中需要性能。我们设计并实现了一种新的框架，连接迁移支持的拆分(COMS)，它利用这些性能特征来增强用户隐私。使用COMPS，我们可以在会话期间跨网络路径和不同的网络协议拆分流量。这种流量拆分通过限制他们可以观察到的流量数量，减轻了网络级对手执行流量分析攻击的能力。我们使用COMPS来构建一个网站指纹防御，该防御系统能够在开放世界的环境下抵御强大的自适应对手的流量分析攻击。我们使用模拟拆分数据和使用COMPS主动拆分的真实流量来评估我们的系统。在我们的真实世界实验中，在100个监控类的开放环境下，COMPS将VarCNN的准确率和召回率分别降低到29.9%和36.7%。Comps不仅可以立即部署到任何支持连接迁移的未经更改的服务器上，而且几乎不会产生开销，吞吐量只会减少5%-20%。



## **29. Adversarial Classification under Gaussian Mechanism: Calibrating the Attack to Sensitivity**

高斯机制下的对抗性分类：将攻击校准为敏感度 cs.IT

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2201.09751v3)

**Authors**: Ayse Unsal, Melek Onen

**Abstracts**: This work studies anomaly detection under differential privacy (DP) with Gaussian perturbation using both statistical and information-theoretic tools. In our setting, the adversary aims to modify the content of a statistical dataset by inserting additional data without being detected by using the DP guarantee to her own benefit. To this end, we characterize information-theoretic and statistical thresholds for the first and second-order statistics of the adversary's attack, which balances the privacy budget and the impact of the attack in order to remain undetected. Additionally, we introduce a new privacy metric based on Chernoff information for classifying adversaries under differential privacy as a stronger alternative to $(\epsilon, \delta)-$ and Kullback-Leibler DP for the Gaussian mechanism. Analytical results are supported by numerical evaluations.

摘要: 本文利用统计学和信息论的方法研究了高斯扰动下的异常检测问题。在我们的设置中，敌手的目标是通过插入附加数据来修改统计数据集的内容，而不会被使用DP保证检测到，从而使自己受益。为此，我们刻画了攻击者攻击的一阶和二阶统计量的信息论和统计阈值，该阈值平衡了隐私预算和攻击的影响，以便保持未被检测到。此外，我们引入了一种新的基于Chernoff信息的隐私度量，用于区分隐私下的攻击者，作为对$(\epsilon，\Delta)-$和用于高斯机制的Kullback-Leibler DP的更强选择。分析结果得到数值评估的支持。



## **30. Learning Optimal Propagation for Graph Neural Networks**

图神经网络的学习最优传播算法 cs.LG

7 pages, 3 figures

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02998v1)

**Authors**: Beidi Zhao, Boxin Du, Zhe Xu, Liangyue Li, Hanghang Tong

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in a variety of real-world applications by relying on the fixed graph data as input. However, the initial input graph might not be optimal in terms of specific downstream tasks, because of information scarcity, noise, adversarial attacks, or discrepancies between the distribution in graph topology, features, and groundtruth labels. In this paper, we propose a bi-level optimization-based approach for learning the optimal graph structure via directly learning the Personalized PageRank propagation matrix as well as the downstream semi-supervised node classification simultaneously. We also explore a low-rank approximation model for further reducing the time complexity. Empirical evaluations show the superior efficacy and robustness of the proposed model over all baseline methods.

摘要: 图形神经网络依靠固定的图形数据作为输入，在各种实际应用中取得了巨大的成功。然而，就特定的下游任务而言，初始输入图可能不是最优的，原因是信息稀缺、噪声、对抗性攻击，或者图拓扑、特征和基本事实标签的分布之间的差异。在本文中，我们提出了一种基于双层优化的方法，通过同时学习个性化PageRank传播矩阵和下游的半监督节点分类来学习最优的图结构。为了进一步降低时间复杂度，我们还探索了一种低阶近似模型。实证评估表明，该模型的有效性和稳健性优于所有的基线方法。



## **31. Privacy-from-Birth: Protecting Sensed Data from Malicious Sensors with VERSA**

从出生起就保护隐私：使用反之亦然保护传感数据免受恶意传感器的侵害 cs.CR

13 pages paper and 4 pages appendix. To be published at 2022 IEEE  Symposium on Security and Privacy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02963v1)

**Authors**: Ivan De Oliveira Nunes, Seoyeon Hwang, Sashidhar Jakkamsetti, Gene Tsudik

**Abstracts**: There are many well-known techniques to secure sensed data in IoT/CPS systems, e.g., by authenticating communication end-points, encrypting data before transmission, and obfuscating traffic patterns. Such techniques protect sensed data from external adversaries while assuming that the sensing device itself is secure. Meanwhile, both the scale and frequency of IoT-focused attacks are growing. This prompts a natural question: how to protect sensed data even if all software on the device is compromised? Ideally, in order to achieve this, sensed data must be protected from its genesis, i.e., from the time when a physical analog quantity is converted into its digital counterpart and becomes accessible to software. We refer to this property as PfB: Privacy-from-Birth.   In this work, we formalize PfB and design Verified Remote Sensing Authorization (VERSA) -- a provably secure and formally verified architecture guaranteeing that only correct execution of expected and explicitly authorized software can access and manipulate sensing interfaces, specifically, General Purpose Input/Output (GPIO), which is the usual boundary between analog and digital worlds on IoT devices. This guarantee is obtained with minimal hardware support and holds even if all device software is compromised. VERSA ensures that malware can neither gain access to sensed data on the GPIO-mapped memory nor obtain any trace thereof. VERSA is formally verified and its open-sourced implementation targets resource-constrained IoT edge devices, commonly used for sensing. Experimental results show that PfB is both achievable and affordable for such devices.

摘要: 有许多众所周知的技术来保护物联网/CPS系统中的感测数据，例如通过认证通信端点、在传输之前加密数据、以及混淆业务模式。这类技术在假设传感设备本身是安全的同时，保护传感数据不受外部攻击者的攻击。与此同时，以物联网为重点的攻击的规模和频率都在增长。这就引出了一个自然的问题：即使设备上的所有软件都被攻破了，如何保护传感数据？理想情况下，为了实现这一点，感测数据必须受到保护，使其不受其产生的影响，即从物理模拟量转换为其数字对应物并变得可由软件访问时起。我们将这一特性称为pfb：从出生起的隐私。在这项工作中，我们将PFB形式化，并设计了经过验证的遥感授权(Versa)--这是一种可证明的安全且经过正式验证的体系结构，确保只有正确执行预期和明确授权的软件才能访问和操作传感接口，特别是通用输入/输出(GPIO)，它是物联网设备上模拟和数字世界之间的通常边界。这种保证只需最少的硬件支持即可获得，即使所有设备软件都被攻破，这种保证也有效。Versa确保恶意软件既不能访问GPIO映射存储器上的感测数据，也不能获得任何痕迹。Versa经过正式验证，其开源实施针对资源受限的物联网边缘设备，通常用于传感。实验结果表明，对于这类器件，PFB是可以实现的，也是可以负担得起的。



## **32. Transferring Adversarial Robustness Through Robust Representation Matching**

通过鲁棒表示匹配传递对抗性鲁棒性 cs.LG

To appear at USENIX Security '22. Updated version with artifact  evaluation badges and appendix

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2202.09994v2)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.

摘要: 随着机器学习的广泛使用，人们对其安全性和可靠性的担忧也变得普遍。因此，许多公司已经开发出防御措施，以加强神经网络对敌意例子的抵挡，这些例子是潜移默化的，输入被可靠地错误分类。对抗性训练，即在训练期间生成和使用对抗性例子，是为数不多的能够可靠地抵御针对神经网络的此类攻击的已知防御措施之一。然而，对抗性训练带来了巨大的训练开销，并且与模型复杂性和输入维度的可比性很差。在本文中，我们提出了一种低成本的方法--稳健表示匹配(RRM)，该方法将对抗性训练的模型的稳健性转移到为同一任务训练的新模型，而不考虑体系结构的差异。受师生学习的启发，我们的方法引入了一种新颖的训练损失，鼓励学生学习教师的健壮表示。与以前的工作相比，RRM在模型性能和对抗性训练时间方面都具有优势。在CIFAR-10上，RRM训练的健壮模型$\sim\比最先进的模型快1.8倍。此外，RRM在高维数据集上仍然有效。在受限的ImageNet上，RRM训练ResNet50型号的速度比标准对手训练快18倍。



## **33. Can collaborative learning be private, robust and scalable?**

协作学习能否做到私密性、健壮性和可扩展性？ cs.LG

Submitted to TPDP 2022

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02652v1)

**Authors**: Dmitrii Usynin, Helena Klause, Daniel Rueckert, Georgios Kaissis

**Abstracts**: We investigate the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples in train- and inference-time attacks. We explore the applications of these techniques as well as their combinations to determine which method performs best, without a significant utility trade-off. Our investigation provides a practical overview of various methods that allow one to achieve a competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation.

摘要: 我们研究了将差分隐私、模型压缩和对抗性训练相结合来提高模型在训练和推理时间攻击中对对抗性样本的稳健性的有效性。我们探索这些技术的应用以及它们的组合，以确定哪种方法执行得最好，而不需要进行重大的实用权衡。我们的研究提供了各种方法的实用概述，这些方法允许人们在不严重性能下降的情况下实现竞争性模型性能、显著减小模型规模和改善经验对抗性稳健性。



## **34. Holistic Approach to Measure Sample-level Adversarial Vulnerability and its Utility in Building Trustworthy Systems**

样本级别敌方脆弱性的整体度量方法及其在构建可信系统中的应用 cs.CV

Accepted in CVPR Workshop 2022 on Human-centered Intelligent  Services: Safe and Trustworthy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02604v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Rohit Lal, Himanshu Patil, Anirban Chakraborty

**Abstracts**: Adversarial attack perturbs an image with an imperceptible noise, leading to incorrect model prediction. Recently, a few works showed inherent bias associated with such attack (robustness bias), where certain subgroups in a dataset (e.g. based on class, gender, etc.) are less robust than others. This bias not only persists even after adversarial training, but often results in severe performance discrepancies across these subgroups. Existing works characterize the subgroup's robustness bias by only checking individual sample's proximity to the decision boundary. In this work, we argue that this measure alone is not sufficient and validate our argument via extensive experimental analysis. It has been observed that adversarial attacks often corrupt the high-frequency components of the input image. We, therefore, propose a holistic approach for quantifying adversarial vulnerability of a sample by combining these different perspectives, i.e., degree of model's reliance on high-frequency features and the (conventional) sample-distance to the decision boundary. We demonstrate that by reliably estimating adversarial vulnerability at the sample level using the proposed holistic metric, it is possible to develop a trustworthy system where humans can be alerted about the incoming samples that are highly likely to be misclassified at test time. This is achieved with better precision when our holistic metric is used over individual measures. To further corroborate the utility of the proposed holistic approach, we perform knowledge distillation in a limited-sample setting. We observe that the student network trained with the subset of samples selected using our combined metric performs better than both the competing baselines, viz., where samples are selected randomly or based on their distances to the decision boundary.

摘要: 对抗性攻击使图像受到难以察觉的噪声干扰，从而导致错误的模型预测。最近，一些研究表明与这种攻击相关的固有偏见(健壮性偏差)，其中数据集中的某些子组(例如，基于类别、性别等)。都不如其他人那么健壮。这种偏见不仅在对抗性训练后仍然存在，而且经常导致这些小组之间的严重表现差异。已有的工作仅通过检查单个样本与决策边界的接近程度来表征子组的稳健性偏差。在这项工作中，我们认为单靠这一措施是不够的，并通过广泛的实验分析验证了我们的论点。已经观察到，对抗性攻击经常破坏输入图像的高频分量。因此，我们通过结合不同的角度，即模型对高频特征的依赖程度和(常规)样本到决策边界的距离，提出了一种量化样本对抗脆弱性的整体方法。我们证明，通过使用所提出的整体度量在样本级别可靠地估计对手漏洞，有可能开发一个可信的系统，在该系统中，可以向人类发出关于在测试时极有可能被错误分类的传入样本的警报。当我们的整体度量用于单个度量时，这是以更高的精度实现的。为了进一步证实所提出的整体方法的实用性，我们在有限样本环境下进行了知识提炼。我们观察到，用使用我们的组合度量选择的样本子集训练的学生网络比两个竞争基线(即，随机选择样本或基于样本到决策边界的距离)的性能都要好。



## **35. Resilience of Bayesian Layer-Wise Explanations under Adversarial Attacks**

贝叶斯层次型解释在对抗性攻击下的弹性 cs.LG

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2102.11010v3)

**Authors**: Ginevra Carbone, Guido Sanguinetti, Luca Bortolussi

**Abstracts**: We consider the problem of the stability of saliency-based explanations of Neural Network predictions under adversarial attacks in a classification task. Saliency interpretations of deterministic Neural Networks are remarkably brittle even when the attacks fail, i.e. for attacks that do not change the classification label. We empirically show that interpretations provided by Bayesian Neural Networks are considerably more stable under adversarial perturbations of the inputs and even under direct attacks to the explanations. By leveraging recent results, we also provide a theoretical explanation of this result in terms of the geometry of the data manifold. Additionally, we discuss the stability of the interpretations of high level representations of the inputs in the internal layers of a Network. Our results demonstrate that Bayesian methods, in addition to being more robust to adversarial attacks, have the potential to provide more stable and interpretable assessments of Neural Network predictions.

摘要: 在一个分类任务中，我们考虑了在对抗性攻击下神经网络预测的基于显著的解释的稳定性问题。即使攻击失败，即对于没有改变分类标签的攻击，确定性神经网络的显著解释也是非常脆弱的。我们的经验表明，贝叶斯神经网络提供的解释在输入的对抗性扰动下甚至在对解释的直接攻击下都更稳定。通过利用最近的结果，我们还从数据流形的几何角度对这一结果进行了理论解释。此外，我们还讨论了网络内部层中输入的高级表示的解释的稳定性。我们的结果表明，贝叶斯方法除了对对手攻击具有更强的鲁棒性外，还有可能为神经网络预测提供更稳定和更可解释的评估。



## **36. Robust Conversational Agents against Imperceptible Toxicity Triggers**

强大的对话代理，可抵御潜伏的毒性触发 cs.CL

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02392v1)

**Authors**: Ninareh Mehrabi, Ahmad Beirami, Fred Morstatter, Aram Galstyan

**Abstracts**: Warning: this paper contains content that maybe offensive or upsetting. Recent research in Natural Language Processing (NLP) has advanced the development of various toxicity detection models with the intention of identifying and mitigating toxic language from existing systems. Despite the abundance of research in this area, less attention has been given to adversarial attacks that force the system to generate toxic language and the defense against them. Existing work to generate such attacks is either based on human-generated attacks which is costly and not scalable or, in case of automatic attacks, the attack vector does not conform to human-like language, which can be detected using a language model loss. In this work, we propose attacks against conversational agents that are imperceptible, i.e., they fit the conversation in terms of coherency, relevancy, and fluency, while they are effective and scalable, i.e., they can automatically trigger the system into generating toxic language. We then propose a defense mechanism against such attacks which not only mitigates the attack but also attempts to maintain the conversational flow. Through automatic and human evaluations, we show that our defense is effective at avoiding toxic language generation even against imperceptible toxicity triggers while the generated language fits the conversation in terms of coherency and relevancy. Lastly, we establish the generalizability of such a defense mechanism on language generation models beyond conversational agents.

摘要: 警告：本文包含可能冒犯或令人反感的内容。自然语言处理(NLP)的最新研究推动了各种毒性检测模型的发展，目的是从现有系统中识别和缓解有毒语言。尽管在这一领域进行了大量的研究，但对迫使系统生成有毒语言的对抗性攻击以及对它们的防御的关注较少。生成此类攻击的现有工作要么基于代价高昂且不可扩展的人为生成的攻击，要么在自动攻击的情况下，攻击向量不符合可使用语言模型丢失来检测的类人类语言。在这项工作中，我们提出了针对不可察觉的会话代理的攻击，即它们在连贯性、关联性和流畅性方面符合会话，而它们是有效和可扩展的，即它们可以自动触发系统生成有毒语言。然后，我们提出了一种针对此类攻击的防御机制，该机制不仅可以缓解攻击，还可以尝试保持会话流。通过自动和人工评估，我们的防御措施有效地避免了有毒语言的生成，即使是针对潜在的有毒触发，而生成的语言在连贯性和关联性方面符合会话。最后，我们建立了这种防御机制在会话主体之外的语言生成模型上的泛化能力。



## **37. Zero Day Threat Detection Using Graph and Flow Based Security Telemetry**

基于图和流的安全遥测零日威胁检测 cs.CR

11 pages, 6 figures, submitting to NeurIPS 2022

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02298v1)

**Authors**: Christopher Redino, Dhruv Nandakumar, Robert Schiller, Kevin Choi, Abdul Rahman, Edward Bowen, Matthew Weeks, Aaron Shaha, Joe Nehila

**Abstracts**: Zero Day Threats (ZDT) are novel methods used by malicious actors to attack and exploit information technology (IT) networks or infrastructure. In the past few years, the number of these threats has been increasing at an alarming rate and have been costing organizations millions of dollars to remediate. The increasing expansion of network attack surfaces and the exponentially growing number of assets on these networks necessitate the need for a robust AI-based Zero Day Threat detection model that can quickly analyze petabyte-scale data for potentially malicious and novel activity. In this paper, the authors introduce a deep learning based approach to Zero Day Threat detection that can generalize, scale, and effectively identify threats in near real-time. The methodology utilizes network flow telemetry augmented with asset-level graph features, which are passed through a dual-autoencoder structure for anomaly and novelty detection respectively. The models have been trained and tested on four large scale datasets that are representative of real-world organizational networks and they produce strong results with high precision and recall values. The models provide a novel methodology to detect complex threats with low false-positive rates that allow security operators to avoid alert fatigue while drastically reducing their mean time to response with near-real-time detection. Furthermore, the authors also provide a novel, labelled, cyber attack dataset generated from adversarial activity that can be used for validation or training of other models. With this paper, the authors' overarching goal is to provide a novel architecture and training methodology for cyber anomaly detectors that can generalize to multiple IT networks with minimal to no retraining while still maintaining strong performance.

摘要: 零日威胁(ZDT)是恶意行为者用来攻击和利用信息技术(IT)网络或基础设施的新方法。在过去的几年里，这些威胁的数量一直在以惊人的速度增长，并花费了组织数百万美元来补救。网络攻击面日益扩大，这些网络上的资产数量呈指数级增长，这就需要一个强大的基于人工智能的零日威胁检测模型，该模型可以快速分析PB级数据，以发现潜在的恶意和新活动。在本文中，作者介绍了一种基于深度学习的零日威胁检测方法，该方法可以近乎实时地概括、扩展和有效地识别威胁。该方法利用网络流量遥测和资产级别图形特征，这些特征通过一个双自动编码器结构分别用于异常和新奇检测。这些模型已经在代表真实世界组织网络的四个大规模数据集上进行了训练和测试，它们产生了具有高精确度和召回值的强大结果。这些模型提供了一种新的方法来检测错误率低的复杂威胁，使安全操作员能够避免警报疲劳，同时通过近实时检测大幅缩短他们的平均响应时间。此外，作者还提供了一个新的、标记的、从敌对活动中生成的网络攻击数据集，可以用于验证或训练其他模型。在这篇论文中，作者的总体目标是为网络异常检测器提供一种新的体系结构和训练方法，该结构和训练方法可以推广到多个IT网络，而不需要重新训练，同时仍然保持强大的性能。



## **38. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

31 pages, 6 figures, small tweak

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.01663v2)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用一个语言生成任务作为测试平台，通过对抗性训练来实现高可靠性。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们简单的“避免受伤”任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。在我们选择的阈值下，使用我们的基准分类器进行过滤可以将分发内数据的不安全完成率从大约2.4%降低到0.003%，这接近我们的测量能力极限。我们发现，对抗性训练显著提高了对我们训练的对抗性攻击的健壮性，而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **39. Rethinking Classifier And Adversarial Attack**

对量词与对抗性攻击的再思考 cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02743v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e. not approaching the lower bound of robustness). To solve this problem, this paper first uses the Decouple Space method to divide the classifier into two parts: non-linear and linear. On this basis, this paper defines the representation vector of original example (and its space, i.e., the representation space) and uses Absolute Classification Boundaries Initialization (ACBI) iterative optimization to obtain a better attack starting point (i.e. attacking from this point can approach the lower bound of robustness faster). Particularly, this paper apply ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.

摘要: 人们提出了各种防御模型来抵抗对抗攻击算法，但现有的对抗稳健性评估方法往往高估了这些模型的对抗稳健性(即没有接近稳健性的下界)。为了解决这一问题，本文首先利用解耦空间方法将分类器分为两部分：非线性部分和线性部分。在此基础上，定义了原始样本的表示向量(及其空间，即表示空间)，并通过绝对分类边界初始化(ACBI)迭代优化，获得了更好的攻击起点(即从这里攻击可以更快地逼近鲁棒性下界)。特别是，本文将ACBI应用于近50个广泛使用的防御模型(包括8个体系结构)。实验结果表明，ACBI在所有情况下都表现出较低的稳健性。



## **40. Based-CE white-box adversarial attack will not work using super-fitting**

基于CE的白盒对抗性攻击将不会使用超级拟合 cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02741v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep Neural Networks (DNN) are widely used in various fields due to their powerful performance, but recent studies have shown that deep learning models are vulnerable to adversarial attacks-by adding a slight perturbation to the input, the model will get wrong results. It is especially dangerous for some systems with high security requirements, so this paper proposes a new defense method by using the model super-fitting status. Model's adversarial robustness (i.e., the accuracry under adversarial attack) has been greatly improved in this status. This paper mathematically proves the effectiveness of super-fitting, and proposes a method to make the model reach this status quickly-minimaze unrelated categories scores (MUCS). Theoretically, super-fitting can resist any existing (even future) Based on CE white-box adversarial attack. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting and other nearly 50 defense models from recent conferences. The experimental results show that super-fitting method in this paper can make the trained model obtain the highest adversarial performance robustness.

摘要: 深度神经网络(DNN)以其强大的性能在各个领域得到了广泛的应用，但最近的研究表明，深度学习模型容易受到对手的攻击--只要在输入中加入一点扰动，模型就会得到错误的结果。对于一些安全性要求较高的系统尤其危险，因此本文提出了一种利用模型的超拟合态进行防御的新方法。在这种情况下，模型的对抗稳健性(即在对抗攻击下的准确性)得到了很大的提高。本文从数学上证明了超拟合的有效性，并提出了一种使模型快速达到这一状态的方法--最小化无关类别得分(MUC)。理论上，超拟合可以抵抗任何现有的(甚至是未来的)基于CE白盒的对抗性攻击。此外，本文使用多种强大的攻击算法对最近几次会议上的超拟合等近50种防御模型的对抗健壮性进行了评估。实验结果表明，本文提出的超拟合方法可以使训练后的模型获得最高的对抗性能稳健性。



## **41. Few-Shot Backdoor Attacks on Visual Object Tracking**

视觉目标跟踪中的几次后门攻击 cs.CV

This work is accepted by the ICLR 2022. The first two authors  contributed equally to this work. In this version, we fix some typos and  errors contained in the last one. 21 pages

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2201.13178v2)

**Authors**: Yiming Li, Haoxiang Zhong, Xingjun Ma, Yong Jiang, Shu-Tao Xia

**Abstracts**: Visual object tracking (VOT) has been widely adopted in mission-critical applications, such as autonomous driving and intelligent surveillance systems. In current practice, third-party resources such as datasets, backbone networks, and training platforms are frequently used to train high-performance VOT models. Whilst these resources bring certain convenience, they also introduce new security threats into VOT models. In this paper, we reveal such a threat where an adversary can easily implant hidden backdoors into VOT models by tempering with the training process. Specifically, we propose a simple yet effective few-shot backdoor attack (FSBA) that optimizes two losses alternately: 1) a \emph{feature loss} defined in the hidden feature space, and 2) the standard \emph{tracking loss}. We show that, once the backdoor is embedded into the target model by our FSBA, it can trick the model to lose track of specific objects even when the \emph{trigger} only appears in one or a few frames. We examine our attack in both digital and physical-world settings and show that it can significantly degrade the performance of state-of-the-art VOT trackers. We also show that our attack is resistant to potential defenses, highlighting the vulnerability of VOT models to potential backdoor attacks.

摘要: 视觉对象跟踪(VOT)已被广泛应用于任务关键型应用，如自动驾驶和智能监控系统。在目前的实践中，经常使用数据集、骨干网、培训平台等第三方资源来培训高性能的VOT模型。这些资源在带来一定便利的同时，也给VOT模型带来了新的安全威胁。在本文中，我们揭示了这样一种威胁，其中对手可以通过调整训练过程来轻松地在VOT模型中植入隐藏的后门。具体地说，我们提出了一种简单而有效的少射击后门攻击(FSBA)，它交替优化了两种损失：1)定义在隐藏特征空间中的a\emph{特征损失}，2)标准\emph{跟踪损失}。我们证明，一旦FSBA将后门嵌入到目标模型中，它就可以欺骗模型，使其失去对特定对象的跟踪，即使\emph{触发器}只出现在一个或几个帧中。我们在数字和物理环境中检查了我们的攻击，并表明它可以显著降低最先进的VoT跟踪器的性能。我们还表明，我们的攻击是抵抗潜在防御的，这突显了VOT模型对潜在后门攻击的脆弱性。



## **42. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

AdaptOver：蜂窝网络中的自适应遮蔽攻击 cs.CR

This version introduces uplink overshadowing

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2106.05039v2)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Röschlin, Srdjan Čapkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver -- a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.

摘要: 在蜂窝网络中，对移动设备和核心网络之间的通信链路的攻击会严重影响隐私和可用性。到目前为止，伪基站已经被要求执行这样的攻击。由于它们需要持续高的输出功率来吸引受害者，因此它们的射程有限，运营商和用户智能手机上的专用应用程序都很容易检测到它们。本文介绍了一种专为LTE和5G-NSA蜂窝网络设计的MITM攻击系统AdaptOver。AdaptOver允许对手在网络和移动设备之间的任一方向上通过空中解码、掩盖(替换)和注入任意消息。使用遮蔽，AdaptOver可以触发UE以纯文本形式传输其永久标识符(IMSI)，从而导致持续($\geq$12h)DoS或隐私泄露。这些攻击可以针对一个小区内的所有用户，也可以根据受害者的电话号码专门针对受害者。我们使用软件定义的无线电和低成本的放大设置来实现AdaptOver。我们使用各种智能手机演示了这些攻击对实时运行的LTE和5G-NSA网络的影响和实用性。我们的实验表明，AdaptOver可以对距离攻击者3.8公里以上的受害者发动攻击。考虑到其实用性和效率，AdaptOver表明，专注于伪基站的现有对策不再足够，标志着蜂窝网络安全机制设计的范式转变。



## **43. Can Rationalization Improve Robustness?**

合理化能提高健壮性吗？ cs.CL

Accepted to NAACL 2022; The code is available at  https://github.com/princeton-nlp/rationale-robustness

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2204.11790v2)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.

摘要: 越来越多的工作研究了神经NLP模型的发展，这种模型可以产生原理--输入的子集可以解释他们的模型预测。在本文中，我们询问这些基本模型除了具有可解释的性质外，是否还可以提供对对手攻击的稳健性。由于这些模型在做出预测(“预测者”)之前需要首先生成理由(“理性器”)，因此它们有可能忽略噪声或相反添加的文本，只需将其从生成的理由中掩盖出来。为此，我们系统地为标记和句子级合理化任务生成了各种类型的AddText攻击，并在五个不同的任务中对最先进的理性模型进行了广泛的经验评估。我们的实验表明，当理性器对位置偏差或攻击文本的词汇选择敏感时，基本模型显示出提高稳健性的前景，而它们在某些场景中却举步维艰。此外，利用人的理性作为监督并不总是能转化为更好的业绩。我们的研究是探索在合理化-然后预测框架中可解释性和稳健性之间的相互作用的第一步。



## **44. Don't sweat the small stuff, classify the rest: Sample Shielding to protect text classifiers against adversarial attacks**

不要为小事操心，对其余的事情进行分类：样本屏蔽保护文本分类器免受对手攻击 cs.CL

9 pages, 8 figures, Accepted to NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01714v1)

**Authors**: Jonathan Rusert, Padmini Srinivasan

**Abstracts**: Deep learning (DL) is being used extensively for text classification. However, researchers have demonstrated the vulnerability of such classifiers to adversarial attacks. Attackers modify the text in a way which misleads the classifier while keeping the original meaning close to intact. State-of-the-art (SOTA) attack algorithms follow the general principle of making minimal changes to the text so as to not jeopardize semantics. Taking advantage of this we propose a novel and intuitive defense strategy called Sample Shielding. It is attacker and classifier agnostic, does not require any reconfiguration of the classifier or external resources and is simple to implement. Essentially, we sample subsets of the input text, classify them and summarize these into a final decision. We shield three popular DL text classifiers with Sample Shielding, test their resilience against four SOTA attackers across three datasets in a realistic threat setting. Even when given the advantage of knowing about our shielding strategy the adversary's attack success rate is <=10% with only one exception and often < 5%. Additionally, Sample Shielding maintains near original accuracy when applied to original texts. Crucially, we show that the `make minimal changes' approach of SOTA attackers leads to critical vulnerabilities that can be defended against with an intuitive sampling strategy.

摘要: 深度学习正被广泛地用于文本分类。然而，研究人员已经证明了这种分类器在对抗攻击时的脆弱性。攻击者以一种误导量词的方式修改文本，同时几乎保持原始含义不变。最新的(SOTA)攻击算法遵循对文本进行最小程度的更改以不危及语义的一般原则。利用这一点，我们提出了一种新颖而直观的防御策略，称为样本屏蔽。它与攻击者和分类器无关，不需要重新配置分类器或外部资源，并且易于实现。基本上，我们对输入文本的子集进行采样，对它们进行分类，并将其总结为最终决策。我们用样本屏蔽了三个流行的DL文本分类器，在真实的威胁设置中测试了它们在三个数据集上对抗四个SOTA攻击者的弹性。即使在知道我们的屏蔽策略的优势下，对手的攻击成功率也是<=10%，只有一个例外，而且通常<5%。此外，当应用于原始文本时，样本屏蔽保持了接近原始的准确性。至关重要的是，我们展示了SOTA攻击者的“最小改动”方法导致了可以通过直观的抽样策略防御的关键漏洞。



## **45. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

受限特征空间中的对抗性攻防统一框架 cs.AI

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2112.01156v2)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

摘要: 为了正确评估在受限特征空间中工作的模型，需要生成可行的对抗性示例。然而，对为计算机视觉设计的攻击实施约束仍然是一项具有挑战性的任务。我们提出了一个统一的框架来生成满足给定领域约束的可行对抗性实例。该框架既可以处理线性约束，也可以处理非线性约束。我们将我们的框架实例化为两种算法：一种是在损失函数中引入约束以最大化的基于梯度的攻击，另一种是以误分类、扰动最小化和约束满足为目标的多目标搜索算法。我们表明，我们的方法在四个不同的领域是有效的，成功率高达100%，在这些领域，最先进的攻击无法生成一个可行的例子。除了对抗性再训练，我们还建议引入工程非凸约束来提高模型对抗性的稳健性。我们证明了这种新的防御与对抗性的再训练一样有效。我们的框架构成了受限对抗攻击研究的起点，并为未来的研究提供了相关的基线和数据集。



## **46. On the uncertainty principle of neural networks**

论神经网络的不确定性原理 cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01493v1)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang

**Abstracts**: Despite the successes in many fields, it is found that neural networks are vulnerability and difficult to be both accurate and robust (robust means that the prediction of the trained network stays unchanged for inputs with non-random perturbations introduced by adversarial attacks). Various empirical and analytic studies have suggested that there is more or less a trade-off between the accuracy and robustness of neural networks. If the trade-off is inherent, applications based on the neural networks are vulnerable with untrustworthy predictions. It is then essential to ask whether the trade-off is an inherent property or not. Here, we show that the accuracy-robustness trade-off is an intrinsic property whose underlying mechanism is deeply related to the uncertainty principle in quantum mechanics. We find that for a neural network to be both accurate and robust, it needs to resolve the features of the two conjugated parts $x$ (the inputs) and $\Delta$ (the derivatives of the normalized loss function $J$ with respect to $x$), respectively. Analogous to the position-momentum conjugation in quantum mechanics, we show that the inputs and their conjugates cannot be resolved by a neural network simultaneously.

摘要: 尽管在许多领域取得了成功，但人们发现神经网络是脆弱的，很难既准确又稳健(稳健是指对于受到对抗性攻击引入的非随机扰动的输入，训练网络的预测保持不变)。各种经验和分析研究表明，神经网络的准确性和稳健性之间或多或少存在权衡。如果这种权衡是与生俱来的，那么基于神经网络的应用程序很容易受到不可信预测的影响。因此，至关重要的是要问一问，这种权衡是否是一种固有属性。在这里，我们证明了精度-稳健性权衡是一种内在的性质，其潜在的机制与量子力学中的测不准原理密切相关。我们发现，为了使神经网络既准确又稳健，它需要分别解析两个共轭部分$x$(输入)和$\Delta$(归一化损失函数$J$关于$x$的导数)的特征。类似于量子力学中的位置-动量共轭，我们证明了输入及其共轭不能由神经网络同时求解。



## **47. Self-Ensemble Adversarial Training for Improved Robustness**

提高健壮性的自我集成对抗性训练 cs.LG

18 pages, 3 figures, ICLR 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2203.09678v2)

**Authors**: Hongjun Wang, Yisen Wang

**Abstracts**: Due to numerous breakthroughs in real-world applications brought by machine intelligence, deep neural networks (DNNs) are widely employed in critical applications. However, predictions of DNNs are easily manipulated with imperceptible adversarial perturbations, which impedes the further deployment of DNNs and may result in profound security and privacy implications. By incorporating adversarial samples into the training data pool, adversarial training is the strongest principled strategy against various adversarial attacks among all sorts of defense methods. Recent works mainly focus on developing new loss functions or regularizers, attempting to find the unique optimal point in the weight space. But none of them taps the potentials of classifiers obtained from standard adversarial training, especially states on the searching trajectory of training. In this work, we are dedicated to the weight states of models through the training process and devise a simple but powerful \emph{Self-Ensemble Adversarial Training} (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise. We also discuss the relationship between the ensemble of predictions from different adversarially trained models and the prediction of weight-ensembled models, as well as provide theoretical and empirical evidence that the proposed self-ensemble method provides a smoother loss landscape and better robustness than both individual models and the ensemble of predictions from different classifiers. We further analyze a subtle but fatal issue in the general settings for the self-ensemble model, which causes the deterioration of the weight-ensembled method in the late phases.

摘要: 由于机器智能在实际应用中取得了许多突破，深度神经网络(DNN)被广泛应用于关键应用中。然而，DNN的预测很容易受到潜移默化的敌意干扰，这阻碍了DNN的进一步部署，并可能导致深刻的安全和隐私影响。通过将对抗性样本纳入训练数据库，对抗性训练是各种防御方法中对抗各种对抗性攻击的最强原则性策略。最近的工作主要集中在开发新的损失函数或正则化函数，试图在权空间中找到唯一的最优点。但它们都没有挖掘从标准对抗性训练中获得的分类器的潜力，特别是在训练的搜索轨迹上。在这项工作中，我们致力于通过训练过程来研究模型的权重状态，并设计了一种简单但强大的自集成对抗性训练(SEAT)方法，通过平均历史模型的权重来产生稳健的分类器。这在很大程度上提高了目标模型对几种众所周知的敌意攻击的稳健性，甚至仅仅利用天真的交叉熵损失来监督。我们还讨论了不同对手训练模型的预测集成与权重集成模型预测之间的关系，并提供了理论和经验证据，表明所提出的自集成方法提供了比单个模型和来自不同分类器的预测集成更平滑的损失情况和更好的稳健性。我们进一步分析了自集成模型一般设置中的一个微妙但致命的问题，该问题导致了权重集成方法在后期的恶化。



## **48. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01287v1)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **49. MIRST-DM: Multi-Instance RST with Drop-Max Layer for Robust Classification of Breast Cancer**

MIRST-DM：用于乳腺癌稳健分类的Drop-Max层多实例RST eess.IV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.01674v1)

**Authors**: Shoukun Sun, Min Xian, Aleksandar Vakanski, Hossny Ghanem

**Abstracts**: Robust self-training (RST) can augment the adversarial robustness of image classification models without significantly sacrificing models' generalizability. However, RST and other state-of-the-art defense approaches failed to preserve the generalizability and reproduce their good adversarial robustness on small medical image sets. In this work, we propose the Multi-instance RST with a drop-max layer, namely MIRST-DM, which involves a sequence of iteratively generated adversarial instances during training to learn smoother decision boundaries on small datasets. The proposed drop-max layer eliminates unstable features and helps learn representations that are robust to image perturbations. The proposed approach was validated using a small breast ultrasound dataset with 1,190 images. The results demonstrate that the proposed approach achieves state-of-the-art adversarial robustness against three prevalent attacks.

摘要: 稳健自训练(RST)可以在不显著牺牲模型泛化能力的情况下，增强图像分类模型的对抗性。然而，RST和其他最先进的防御方法未能保持其泛化能力，并在小的医学图像集上重现其良好的对抗性鲁棒性。在这项工作中，我们提出了具有最大丢弃层的多实例RST，即MIRST-DM，它在训练过程中包含一系列迭代生成的对抗性实例，以在小数据集上学习更平滑的决策边界。提出的Drop-max层消除了不稳定的特征，并帮助学习对图像扰动具有鲁棒性的表示。使用1,190幅图像的小型乳腺超声数据集对所提出的方法进行了验证。实验结果表明，该方法对三种流行的攻击具有最好的抗攻击能力。



## **50. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

分段和完全：利用稳健的补丁检测保护对象检测器免受敌意补丁攻击 cs.CV

CVPR 2022 camera ready

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2112.04532v2)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.

摘要: 目标检测在许多安全关键系统中起着关键作用。对抗性补丁攻击很容易在物理世界中实现，对最先进的对象检测器构成了严重威胁。为目标探测器开发可靠的防御补丁攻击是至关重要的，但研究严重不足。在本文中，我们提出了分段和完全防御(SAC)，这是一个通用的框架，通过检测和删除敌意补丁来防御对象检测器的补丁攻击。我们首先训练一个补丁分割器，该分割器输出提供对抗性补丁像素级定位的补丁掩码。然后，我们提出了一种自对抗训练算法来增强补丁分割器的鲁棒性。此外，我们还设计了一种稳健的形状补全算法，如果斑块分割器的输出与地面真实斑块掩模的汉明距离在一定范围内，该算法就能保证从图像中去除整个斑块。我们在CoCo和xView数据集上的实验表明，SAC在不降低对干净图像的性能的情况下，即使在强自适应攻击下也具有优异的鲁棒性，并且对看不见的补丁形状、攻击预算和看不见的攻击方法具有很好的泛化能力。此外，我们还给出了APRICOT-MASK数据集，它用对抗性斑块的像素级标注来扩充APRICOT数据集。结果表明，SAC能够显著降低物理补丁攻击的定向攻击成功率。我们的代码可以在https://github.com/joellliu/SegmentAndComplete.上找到



