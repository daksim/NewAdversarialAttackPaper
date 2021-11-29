# Latest Adversarial Attack Papers
**update at 2021-11-29 23:56:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. The Geometry of Adversarial Training in Binary Classification**

二元分类中对抗性训练的几何问题 cs.LG

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2111.13613v1)

**Authors**: Leon Bungert, Nicolás García Trillos, Ryan Murray

**Abstracts**: We establish an equivalence between a family of adversarial training problems for non-parametric binary classification and a family of regularized risk minimization problems where the regularizer is a nonlocal perimeter functional. The resulting regularized risk minimization problems admit exact convex relaxations of the type $L^1+$ (nonlocal) $\operatorname{TV}$, a form frequently studied in image analysis and graph-based learning. A rich geometric structure is revealed by this reformulation which in turn allows us to establish a series of properties of optimal solutions of the original problem, including the existence of minimal and maximal solutions (interpreted in a suitable sense), and the existence of regular solutions (also interpreted in a suitable sense). In addition, we highlight how the connection between adversarial training and perimeter minimization problems provides a novel, directly interpretable, statistical motivation for a family of regularized risk minimization problems involving perimeter/total variation. The majority of our theoretical results are independent of the distance used to define adversarial attacks.

摘要: 我们建立了一类非参数二元分类的对抗性训练问题和一类正则化风险最小化问题之间的等价性，其中正则化子是非局部周长泛函。由此产生的正则化风险最小化问题允许类型为$L^1+$(非局部)$\操作符名称{TV}$的精确凸松弛，这是图像分析和基于图的学习中经常研究的一种形式。这种改写揭示了一个丰富的几何结构，进而允许我们建立原问题最优解的一系列性质，包括最小解和最大解(在适当意义下解释)的存在性，以及正则解(也在适当意义上解释)的存在性。此外，我们强调了对抗性训练和周长最小化问题之间的联系如何为一类涉及周长/总变异的正则化风险最小化问题提供了一种新颖的、直接可解释的统计动机。我们的大部分理论结果与用于定义对抗性攻击的距离无关。



## **2. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

基于可解释性的点云神经网络单点攻击 cs.CV

**SubmitDate**: 2021-11-26    [paper-pdf](http://arxiv.org/pdf/2110.04158v2)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.

摘要: 随着神经网络在点云领域的提出，深度学习开始在三维物体识别领域闪耀光芒，而研究人员对利用对抗性攻击来研究点云网络的可靠性也表现出越来越大的兴趣。然而，现有的大多数研究都是为了欺骗人类或防御算法，而少数针对模型本身操作原理的研究在临界点选择方面仍然存在缺陷。在这项工作中，我们提出了两种对抗方法：单点攻击(OPA)和临界遍历攻击(CTA)，它们融合了可解释性技术，旨在探索点云网络的内在工作原理及其对临界点扰动的敏感性。我们的结果表明，只要从输入实例中移动一个点，流行的点云网络就可以被欺骗，成功率接近100美元。此外，我们还展示了不同的点属性分布对点云网络对抗健壮性的影响。最后，我们讨论了我们的方法如何促进点云网络的可解释性研究。据我们所知，这是第一种基于点云的对抗性解释方法。我们的代码可在https://github.com/Explain3D/Exp-One-Point-Atk-PC.获得



## **3. Privacy-Preserving Synthetic Smart Meters Data**

保护隐私的合成智能电表数据 eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2012.04475v2)

**Authors**: Ganesh Del Grosso, Georg Pichler, Pablo Piantanida

**Abstracts**: Power consumption data is very useful as it allows to optimize power grids, detect anomalies and prevent failures, on top of being useful for diverse research purposes. However, the use of power consumption data raises significant privacy concerns, as this data usually belongs to clients of a power company. As a solution, we propose a method to generate synthetic power consumption samples that faithfully imitate the originals, but are detached from the clients and their identities. Our method is based on Generative Adversarial Networks (GANs). Our contribution is twofold. First, we focus on the quality of the generated data, which is not a trivial task as no standard evaluation methods are available. Then, we study the privacy guarantees provided to members of the training set of our neural network. As a minimum requirement for privacy, we demand our neural network to be robust to membership inference attacks, as these provide a gateway for further attacks in addition to presenting a privacy threat on their own. We find that there is a compromise to be made between the privacy and the performance provided by the algorithm.

摘要: 电力消耗数据非常有用，因为它可以优化电网，检测异常和防止故障，此外还可以用于不同的研究目的。然而，用电数据的使用引起了很大的隐私问题，因为这些数据通常属于电力公司的客户。作为解决方案，我们提出了一种生成合成功耗样本的方法，该方法忠实地模仿原始功耗样本，但与客户及其身份分离。我们的方法是基于产生式对抗性网络(GANS)的。我们的贡献是双重的。首先，我们将重点放在生成数据的质量上，这不是一项微不足道的任务，因为没有标准的评估方法可用。然后，我们研究了提供给神经网络训练集成员的隐私保证。作为隐私的最低要求，我们要求我们的神经网络对成员关系推断攻击具有健壮性，因为这些攻击除了本身构成隐私威胁之外，还为进一步的攻击提供了一个网关。我们发现需要在隐私和算法提供的性能之间进行折衷。



## **4. Real-Time Privacy-Preserving Data Release for Smart Meters**

智能电表的实时隐私保护数据发布 eess.SP

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/1906.06427v4)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: Smart Meters (SMs) are able to share the power consumption of users with utility providers almost in real-time. These fine-grained signals carry sensitive information about users, which has raised serious concerns from the privacy viewpoint. In this paper, we focus on real-time privacy threats, i.e., potential attackers that try to infer sensitive information from SMs data in an online fashion. We adopt an information-theoretic privacy measure and show that it effectively limits the performance of any attacker. Then, we propose a general formulation to design a privatization mechanism that can provide a target level of privacy by adding a minimal amount of distortion to the SMs measurements. On the other hand, to cope with different applications, a flexible distortion measure is considered. This formulation leads to a general loss function, which is optimized using a deep learning adversarial framework, where two neural networks -- referred to as the releaser and the adversary -- are trained with opposite goals. An exhaustive empirical study is then performed to validate the performance of the proposed approach and compare it with state-of-the-art methods for the occupancy detection privacy problem. Finally, we also investigate the impact of data mismatch between the releaser and the attacker.

摘要: 智能电表(SMS)能够几乎实时地与公用事业提供商共享用户的电能消耗。这些细粒度的信号携带着用户的敏感信息，从隐私的角度来看，这引起了严重的担忧。在本文中，我们关注的是实时隐私威胁，即试图以在线方式从短信数据中推断敏感信息的潜在攻击者。我们采用了一种信息论的隐私措施，并表明它有效地限制了任何攻击者的表现。然后，我们提出了一个通用的公式来设计一种私有化机制，该机制可以通过对短信测量添加最小的失真来提供目标级别的隐私。另一方面，为了适应不同的应用，考虑了一种灵活的失真度量。这一公式导致了一般的损失函数，该函数使用深度学习对手框架进行优化，其中两个神经网络--称为释放者和对手--被训练成具有相反的目标。在此基础上，进行了详尽的实证研究，验证了该方法的性能，并将其与现有的占有率检测隐私问题的方法进行了比较。最后，我们还调查了发布者和攻击者之间数据不匹配的影响。



## **5. Simple Post-Training Robustness Using Test Time Augmentations and Random Forest**

使用测试时间增加和随机森林的简单训练后鲁棒性 cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2109.08191v2)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Although Deep Neural Networks (DNNs) achieve excellent performance on many real-world tasks, they are highly vulnerable to adversarial attacks. A leading defense against such attacks is adversarial training, a technique in which a DNN is trained to be robust to adversarial attacks by introducing adversarial noise to its input. This procedure is effective but must be done during the training phase. In this work, we propose Augmented Random Forest (ARF), a simple and easy-to-use strategy for robustifying an existing pretrained DNN without modifying its weights. For every image, we generate randomized test time augmentations by applying diverse color, blur, noise, and geometric transforms. Then we use the DNN's logits output to train a simple random forest to predict the real class label. Our method achieves state-of-the-art adversarial robustness on a diversity of white and black box attacks with minimal compromise on the natural images' classification. We test ARF also against numerous adaptive white-box attacks and it shows excellent results when combined with adversarial training. Code is available at https://github.com/giladcohen/ARF.

摘要: 尽管深度神经网络(DNNs)在许多现实世界的任务中取得了优异的性能，但它们非常容易受到对手的攻击。对抗这种攻击的主要防御是对抗性训练，这是一种通过在输入中引入对抗性噪声来训练DNN对对抗性攻击具有健壮性的技术。此程序是有效的，但必须在培训阶段进行。在这项工作中，我们提出了增广随机森林(ARF)，这是一种简单易用的策略，可以在不修改权值的情况下对现有的预先训练的DNN进行鲁棒性处理。对于每个图像，我们通过应用不同的颜色、模糊、噪声和几何变换来生成随机的测试时间增量。然后利用DNN的Logits输出训练一个简单的随机森林来预测真实的类标签。我们的方法在对自然图像分类影响最小的情况下，实现了对各种白盒和黑盒攻击的最好的对抗鲁棒性。我们还测试了ARF对许多自适应白盒攻击的攻击，当与对抗性训练相结合时，它显示出极好的结果。代码可在https://github.com/giladcohen/ARF.上获得



## **6. EAD: an ensemble approach to detect adversarial examples from the hidden features of deep neural networks**

EAD：一种从深层神经网络隐含特征中检测敌意实例的集成方法 cs.CV

Corrected Figure 4

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12631v2)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: One of the key challenges in Deep Learning is the definition of effective strategies for the detection of adversarial examples. To this end, we propose a novel approach named Ensemble Adversarial Detector (EAD) for the identification of adversarial examples, in a standard multiclass classification scenario. EAD combines multiple detectors that exploit distinct properties of the input instances in the internal representation of a pre-trained Deep Neural Network (DNN). Specifically, EAD integrates the state-of-the-art detectors based on Mahalanobis distance and on Local Intrinsic Dimensionality (LID) with a newly introduced method based on One-class Support Vector Machines (OSVMs). Although all constituting methods assume that the greater the distance of a test instance from the set of correctly classified training instances, the higher its probability to be an adversarial example, they differ in the way such distance is computed. In order to exploit the effectiveness of the different methods in capturing distinct properties of data distributions and, accordingly, efficiently tackle the trade-off between generalization and overfitting, EAD employs detector-specific distance scores as features of a logistic regression classifier, after independent hyperparameters optimization. We evaluated the EAD approach on distinct datasets (CIFAR-10, CIFAR-100 and SVHN) and models (ResNet and DenseNet) and with regard to four adversarial attacks (FGSM, BIM, DeepFool and CW), also by comparing with competing approaches. Overall, we show that EAD achieves the best AUROC and AUPR in the large majority of the settings and comparable performance in the others. The improvement over the state-of-the-art, and the possibility to easily extend EAD to include any arbitrary set of detectors, pave the way to a widespread adoption of ensemble approaches in the broad field of adversarial example detection.

摘要: 深度学习的关键挑战之一是定义有效的策略来检测敌意示例。为此，我们提出了一种新的方法，称为集成对抗性检测器(EAD)，用于在标准的多类分类场景中识别对抗性实例。EAD结合了多个检测器，这些检测器利用预先训练的深度神经网络(DNN)的内部表示中的输入实例的不同属性。具体地说，EAD将基于马氏距离和基于局部本征维数(LID)的最新检测器与一种新的基于单类支持向量机(OSVMs)的方法相结合。尽管所有的构成方法都假设测试实例与正确分类的训练实例集合的距离越大，其成为对抗性示例的概率就越高，但是它们在计算该距离的方式上是不同的。为了利用不同方法在捕捉数据分布的不同特性方面的有效性，从而有效地撞击泛化和过拟合之间的权衡，经过独立的超参数优化，EAD使用特定于检测器的距离得分作为Logistic回归分类器的特征。我们在不同的数据集(CIFAR-10、CIFAR-100和SVHN)和模型(ResNet和DenseNet)以及四种对手攻击(FGSM、BIM、DeepFool和CW)上对EAD方法进行了评估，并与其他方法进行了比较。总体而言，我们显示EAD在绝大多数设置中实现了最好的AUROC和AUPR，而在其他设置中实现了相当的性能。对现有技术的改进，以及轻松扩展EAD以包括任意一组检测器的可能性，为在广泛的对抗性示例检测领域广泛采用集成方法铺平了道路。



## **7. Can You Spot the Chameleon? Adversarially Camouflaging Images from Co-Salient Object Detection**

你能认出变色龙吗？协同显著目标检测中的对抗性伪装图像 cs.CV

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2009.09258v3)

**Authors**: Ruijun Gao, Qing Guo, Felix Juefei-Xu, Hongkai Yu, Huazhu Fu, Wei Feng, Yang Liu, Song Wang

**Abstracts**: Co-salient object detection (CoSOD) has recently achieved significant progress and played a key role in retrieval-related tasks. However, it inevitably poses an entirely new safety and security issue, \ie, highly personal and sensitive content can potentially be extracting by powerful CoSOD methods. In this paper, we address this problem from the perspective of adversarial attacks and identify a novel task: adversarial co-saliency attack. Specially, given an image selected from a group of images containing some common and salient objects, we aim to generate an adversarial version that can mislead CoSOD methods to predict incorrect co-salient regions. Note that, compared with general white-box adversarial attacks for classification, this new task faces two additional challenges: (1) low success rate due to the diverse appearance of images in the group; (2) low transferability across CoSOD methods due to the considerable difference between CoSOD pipelines. To address these challenges, we propose the very first black-box joint adversarial exposure and noise attack (Jadena), where we jointly and locally tune the exposure and additive perturbations of the image according to a newly designed high-feature-level contrast-sensitive loss function. Our method, without any information on the state-of-the-art CoSOD methods, leads to significant performance degradation on various co-saliency detection datasets and makes the co-salient objects undetectable. This can have strong practical benefits in properly securing the large number of personal photos currently shared on the internet. Moreover, our method is potential to be utilized as a metric for evaluating the robustness of CoSOD methods.

摘要: 共显著对象检测(CoSOD)近年来取得了重大进展，在检索相关任务中发挥了关键作用。然而，这不可避免地带来了一个全新的安全问题，即可以通过强大的CoSOD方法潜在地提取高度个人化和敏感的内容。在本文中，我们从对抗性攻击的角度来研究这一问题，并提出了一种新的任务：对抗性共显攻击。特别地，给定一幅从一组包含一些常见和显著对象的图像中选择的图像，我们的目标是生成一个可能误导CoSOD方法预测错误共显著区域的对抗性版本。值得注意的是，与一般的白盒对抗性分类攻击相比，这个新任务面临着两个额外的挑战：(1)由于组内图像的多样性，成功率较低；(2)由于CoSOD管道之间的巨大差异，CoSOD方法之间的可移植性较低。为了应对这些挑战，我们提出了第一个黑盒联合对抗性曝光和噪声攻击(JADENA)，其中我们根据新设计的高特征级对比度敏感损失函数来联合和局部地调整图像的曝光和加性扰动。我们的方法没有任何关于最新CoSOD方法的信息，导致在各种共显性检测数据集上的性能显著下降，并且使得共显性对象不可检测。这对妥善保护目前在互联网上共享的大量个人照片有很大的实际好处。此外，我们的方法有可能被用作评估CoSOD方法稳健性的一个度量。



## **8. AdvBokeh: Learning to Adversarially Defocus Blur**

AdvBokeh：学会对抗性散焦模糊 cs.CV

13 pages

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12971v1)

**Authors**: Yihao Huang, Felix Juefei-Xu, Qing Guo, Weikai Miao, Yang Liu, Geguang Pu

**Abstracts**: Bokeh effect is a natural shallow depth-of-field phenomenon that blurs the out-of-focus part in photography. In pursuit of aesthetically pleasing photos, people usually regard the bokeh effect as an indispensable part of the photo. Due to its natural advantage and universality, as well as the fact that many visual recognition tasks can already be negatively affected by the `natural bokeh' phenomenon, in this work, we systematically study the bokeh effect from a new angle, i.e., adversarial bokeh attack (AdvBokeh) that aims to embed calculated deceptive information into the bokeh generation and produce a natural adversarial example without any human-noticeable noise artifacts. To this end, we first propose a Depth-guided Bokeh Synthesis Network (DebsNet) that is able to flexibly synthesis, refocus, and adjust the level of bokeh of the image, with a one-stage training procedure. The DebsNet allows us to tap into the bokeh generation process and attack the depth map that is needed for generating realistic bokeh (i.e., adversarially tuning the depth map) based on subsequent visual tasks. To further improve the realisticity of the adversarial bokeh, we propose depth-guided gradient-based attack to regularize the gradient.We validate the proposed method on a popular adversarial image classification dataset, i.e., NeurIPS-2017 DEV, and show that the proposed method can penetrate four state-of-the-art (SOTA) image classification networks i.e., ResNet50, VGG, DenseNet, and MobileNetV2 with a high success rate as well as high image quality. The adversarial examples obtained by AdvBokeh also exhibit high level of transferability under black-box settings. Moreover, the adversarially generated defocus blur images from the AdvBokeh can actually be capitalized to enhance the performance of SOTA defocus deblurring system, i.e., IFAN.

摘要: 波克效应是一种自然的浅景深现象，它模糊了摄影中的失焦部分。在美观照片的追打中，人们通常将波克效应视为照片中不可或缺的一部分。由于其天然的优势和广泛性，以及许多视觉识别任务已经受到“自然波克”现象的负面影响，本文从一个新的角度系统地研究了波克效应，即对抗性波克攻击(AdvBokeh)，其目的是将计算出的欺骗性信息嵌入到波克生成中，并在没有任何人类可察觉的噪声伪影的情况下产生一个自然的对抗性例子。在这项工作中，我们从一个新的角度对波克效应进行了系统的研究，即对抗性波克攻击(AdvBokeh Attack，AdvBokeh)。为此，我们首先提出了一种深度引导的Bokeh合成网络(DebsNet)，它能够灵活地合成、重新聚焦和调整图像的Bokeh级别，并且只需一步训练过程。DebsNet允许我们利用bokeh生成过程，并基于后续视觉任务攻击生成真实bokeh所需的深度图(即相反地调整深度图)。为了进一步提高对抗性图像分类的真实性，提出了基于深度引导的梯度正则化攻击方法，并在一个流行的对抗性图像分类数据集NeurIPS-2017 DEV上进行了验证，结果表明该方法能够穿透ResNet50、VGG、DenseNet和MobileNetV2四种最新的图像分类网络，具有较高的分类成功率和图像质量。AdvBokeh获得的对抗性例子在黑盒设置下也表现出很高的可转移性。此外，来自AdvBokeh的相反产生的散焦模糊图像实际上可以被资本化以增强SOTA散焦模糊系统(即IFAN)的性能。



## **9. Normal vs. Adversarial: Salience-based Analysis of Adversarial Samples for Relation Extraction**

正常VS对抗性：基于显著性的对抗性样本关系提取分析 cs.CL

IJCKG 2021

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2104.00312v4)

**Authors**: Luoqiu Li, Xiang Chen, Zhen Bi, Xin Xie, Shumin Deng, Ningyu Zhang, Chuanqi Tan, Mosha Chen, Huajun Chen

**Abstracts**: Recent neural-based relation extraction approaches, though achieving promising improvement on benchmark datasets, have reported their vulnerability towards adversarial attacks. Thus far, efforts mostly focused on generating adversarial samples or defending adversarial attacks, but little is known about the difference between normal and adversarial samples. In this work, we take the first step to leverage the salience-based method to analyze those adversarial samples. We observe that salience tokens have a direct correlation with adversarial perturbations. We further find the adversarial perturbations are either those tokens not existing in the training set or superficial cues associated with relation labels. To some extent, our approach unveils the characters against adversarial samples. We release an open-source testbed, "DiagnoseAdv" in https://github.com/zjunlp/DiagnoseAdv.

摘要: 最近的基于神经的关系提取方法，虽然在基准数据集上取得了有希望的改进，但已经报告了它们对对手攻击的脆弱性。到目前为止，努力主要集中在生成对抗性样本或防御对抗性攻击，但对正常样本和对抗性样本之间的区别知之甚少。在这项工作中，我们迈出了第一步，利用基于显著性的方法来分析这些敌意样本。我们观察到显著标记与对抗性扰动有直接关系。我们进一步发现，对抗性扰动要么是那些不存在于训练集中的标记，要么是与关系标签相关联的表面线索。在某种程度上，我们的方法针对对手样本揭开了人物的面纱。我们在https://github.com/zjunlp/DiagnoseAdv.中发布了一个开源的测试平台“DiagnoseAdv



## **10. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

走向实用化部署--深度神经网络的阶段后门攻击 cs.CR

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12965v1)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.

摘要: AI安全社区的一个主要目标是安全可靠地为现实世界的应用程序生成和部署深度学习模型。为此，基于数据中毒的深度神经网络(DNNs)在生产阶段(或训练阶段)的后门攻击以及相应的防御措施近年来得到了广泛的研究。具有讽刺意味的是，部署阶段的后门攻击通常会发生在非专业用户的设备上，因此在现实场景中可以说威胁要大得多，但社区对此的关注要少得多。我们将这种警惕性的不平衡归因于现有部署阶段后门攻击算法的实用性较弱，以及现实世界攻击演示的不足。为了填补这一空白，在这项工作中，我们研究了部署阶段后门攻击对DNNs的现实威胁。我们的研究基于一种常用的部署阶段攻击范例--对抗性权重攻击，在这种攻击中，攻击者有选择地修改模型权重，以将后门嵌入到部署的DNN中。为了更接近实际应用，我们提出了第一种灰盒物理可实现权重的后门注入攻击算法，即子网替换攻击算法(SRA)，该算法只需要受害者模型的体系结构信息，能够支持现实世界中的物理触发器。进行了广泛的实验模拟和系统级的真实世界攻击演示。我们的结果不仅表明了提出的攻击算法的有效性和实用性，而且揭示了一种新型计算机病毒的实际风险，这种病毒可能会广泛传播并偷偷地向用户设备中的DNN模型注入后门。通过我们的研究，我们呼吁更多地关注DNNs在部署阶段的脆弱性。



## **11. Clustering Effect of (Linearized) Adversarial Robust Models**

(线性化)对抗性稳健模型的聚类效应 cs.LG

Accepted by NeurIPS 2021, spotlight

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12922v1)

**Authors**: Yang Bai, Xin Yan, Yong Jiang, Shu-Tao Xia, Yisen Wang

**Abstracts**: Adversarial robustness has received increasing attention along with the study of adversarial examples. So far, existing works show that robust models not only obtain robustness against various adversarial attacks but also boost the performance in some downstream tasks. However, the underlying mechanism of adversarial robustness is still not clear. In this paper, we interpret adversarial robustness from the perspective of linear components, and find that there exist some statistical properties for comprehensively robust models. Specifically, robust models show obvious hierarchical clustering effect on their linearized sub-networks, when removing or replacing all non-linear components (e.g., batch normalization, maximum pooling, or activation layers). Based on these observations, we propose a novel understanding of adversarial robustness and apply it on more tasks including domain adaption and robustness boosting. Experimental evaluations demonstrate the rationality and superiority of our proposed clustering strategy.

摘要: 随着对抗性实例的研究，对抗性鲁棒性受到越来越多的关注。到目前为止，已有的工作表明，鲁棒模型不仅对各种敌意攻击具有鲁棒性，而且在一些下游任务中也提高了性能。然而，对抗健壮性的潜在机制仍不清楚。本文从线性分量的角度来解释对抗稳健性，发现综合稳健性模型具有一定的统计特性。具体地说，当移除或替换所有非线性组件(例如，批归一化、最大池化或激活层)时，鲁棒模型在其线性化的子网络上显示出明显的层次聚类效应。基于这些观察结果，我们提出了一种新的对敌方鲁棒性的理解，并将其应用于更多的任务，包括领域自适应和鲁棒性增强。实验结果证明了本文提出的聚类策略的合理性和优越性。



## **12. Robustness against Adversarial Attacks in Neural Networks using Incremental Dissipativity**

基于增量耗散的神经网络对敌意攻击的鲁棒性 cs.LG

**SubmitDate**: 2021-11-25    [paper-pdf](http://arxiv.org/pdf/2111.12906v1)

**Authors**: Bernardo Aquino, Arash Rahnama, Peter Seiler, Lizhen Lin, Vijay Gupta

**Abstracts**: Adversarial examples can easily degrade the classification performance in neural networks. Empirical methods for promoting robustness to such examples have been proposed, but often lack both analytical insights and formal guarantees. Recently, some robustness certificates have appeared in the literature based on system theoretic notions. This work proposes an incremental dissipativity-based robustness certificate for neural networks in the form of a linear matrix inequality for each layer. We also propose an equivalent spectral norm bound for this certificate which is scalable to neural networks with multiple layers. We demonstrate the improved performance against adversarial attacks on a feed-forward neural network trained on MNIST and an Alexnet trained using CIFAR-10.

摘要: 对抗性示例很容易降低神经网络的分类性能。已经提出了提高此类例子稳健性的经验方法，但往往既缺乏分析洞察力，也缺乏形式上的保证。近年来，一些基于系统论概念的健壮性证书出现在文献中。本文以线性矩阵不等式的形式为每一层提出了一种基于耗散性的增量式神经网络鲁棒性证书。我们还给出了该证书的一个等价谱范数界，它可扩展到多层神经网络。我们在使用MNIST训练的前馈神经网络和使用CIFAR-10训练的Alexnet上展示了改进的抗敌意攻击性能。



## **13. On the Impact of Side Information on Smart Meter Privacy-Preserving Methods**

浅谈边信息对智能电表隐私保护方法的影响 eess.SP

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2006.16062v2)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: Smart meters (SMs) can pose privacy threats for consumers, an issue that has received significant attention in recent years. This paper studies the impact of Side Information (SI) on the performance of distortion-based real-time privacy-preserving algorithms for SMs. In particular, we consider a deep adversarial learning framework, in which the desired releaser (a recurrent neural network) is trained by fighting against an adversary network until convergence. To define the loss functions, two different approaches are considered: the Causal Adversarial Learning (CAL) and the Directed Information (DI)-based learning. The main difference between these approaches is in how the privacy term is measured during the training process. On the one hand, the releaser in the CAL method, by getting supervision from the actual values of the private variables and feedback from the adversary performance, tries to minimize the adversary log-likelihood. On the other hand, the releaser in the DI approach completely relies on the feedback received from the adversary and is optimized to maximize its uncertainty. The performance of these two algorithms is evaluated empirically using real-world SMs data, considering an attacker with access to SI (e.g., the day of the week) that tries to infer the occupancy status from the released SMs data. The results show that, although they perform similarly when the attacker does not exploit the SI, in general, the CAL method is less sensitive to the inclusion of SI. However, in both cases, privacy levels are significantly affected, particularly when multiple sources of SI are included.

摘要: 智能电表(SMS)可能会对消费者的隐私构成威胁，这一问题近年来受到了极大的关注。研究了边信息(SI)对基于失真的短信实时隐私保护算法性能的影响。特别地，我们考虑了一种深度对抗性学习框架，在该框架中，期望的释放者(递归神经网络)通过与对手网络战斗直到收敛来训练。为了定义损失函数，考虑了两种不同的方法：因果对抗学习(CAL)和基于定向信息(DI)的学习。这些方法之间的主要区别在于如何在培训过程中衡量隐私条款。一方面，CAL方法中的发布者从私有变量的实际值中获得监督，并从对手的表现中获得反馈，试图最小化对手的对数似然。另一方面，DI方法中的发布者完全依赖于从对手那里收到的反馈，并被优化以最大化其不确定性。考虑到具有访问SI(例如，星期几)的攻击者试图从发布的SMS数据推断占用状态，使用真实世界SMS数据来经验地评估这两个算法的性能。结果表明，尽管在攻击者没有利用SI时，它们的表现相似，但通常情况下，CAL方法对SI的包含不那么敏感。然而，在这两种情况下，隐私级别都会受到很大影响，特别是在包含多个SI源的情况下。



## **14. Deep Directed Information-Based Learning for Privacy-Preserving Smart Meter Data Release**

基于深度定向信息的隐私保护智能电表数据发布 cs.LG

to appear in IEEESmartGridComm 2019. arXiv admin note: substantial  text overlap with arXiv:1906.06427

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2011.11421v3)

**Authors**: Mohammadhadi Shateri, Francisco Messina, Pablo Piantanida, Fabrice Labeau

**Abstracts**: The explosion of data collection has raised serious privacy concerns in users due to the possibility that sharing data may also reveal sensitive information. The main goal of a privacy-preserving mechanism is to prevent a malicious third party from inferring sensitive information while keeping the shared data useful. In this paper, we study this problem in the context of time series data and smart meters (SMs) power consumption measurements in particular. Although Mutual Information (MI) between private and released variables has been used as a common information-theoretic privacy measure, it fails to capture the causal time dependencies present in the power consumption time series data. To overcome this limitation, we introduce the Directed Information (DI) as a more meaningful measure of privacy in the considered setting and propose a novel loss function. The optimization is then performed using an adversarial framework where two Recurrent Neural Networks (RNNs), referred to as the releaser and the adversary, are trained with opposite goals. Our empirical studies on real-world data sets from SMs measurements in the worst-case scenario where an attacker has access to all the training data set used by the releaser, validate the proposed method and show the existing trade-offs between privacy and utility.

摘要: 数据收集的爆炸式增长引发了用户对隐私的严重担忧，因为共享数据可能还会泄露敏感信息。隐私保护机制的主要目标是防止恶意第三方在保持共享数据有用的同时推断敏感信息。在本文中，我们在时间序列数据和智能电表(SMS)功耗测量的背景下研究这一问题。虽然私有变量和已发布变量之间的互信息(MI)已被用作常见的信息论隐私度量，但它不能捕获电力消耗时间序列数据中存在的因果时间依赖关系。为了克服这一局限性，我们引入了定向信息(DI)作为一种更有意义的隐私度量方法，并提出了一种新的损失函数。然后使用对抗性框架执行优化，其中两个递归神经网络(RNN)，称为释放者和对手，以相反的目标进行训练。我们对来自SMS测量的真实世界数据集进行了实证研究，在最坏的情况下，攻击者可以访问发布者使用的所有训练数据集，验证了所提出的方法，并显示了隐私和效用之间的现有权衡。



## **15. Estimating g-Leakage via Machine Learning**

基于机器学习的g泄漏估计 cs.CR

This is the extended version of the paper which will appear in the  Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications  Security (CCS '20), November 9-13, 2020, Virtual Event, USA

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2005.04399v3)

**Authors**: Marco Romanelli, Konstantinos Chatzikokolakis, Catuscia Palamidessi, Pablo Piantanida

**Abstracts**: This paper considers the problem of estimating the information leakage of a system in the black-box scenario. It is assumed that the system's internals are unknown to the learner, or anyway too complicated to analyze, and the only available information are pairs of input-output data samples, possibly obtained by submitting queries to the system or provided by a third party. Previous research has mainly focused on counting the frequencies to estimate the input-output conditional probabilities (referred to as frequentist approach), however this method is not accurate when the domain of possible outputs is large. To overcome this difficulty, the estimation of the Bayes error of the ideal classifier was recently investigated using Machine Learning (ML) models and it has been shown to be more accurate thanks to the ability of those models to learn the input-output correspondence. However, the Bayes vulnerability is only suitable to describe one-try attacks. A more general and flexible measure of leakage is the g-vulnerability, which encompasses several different types of adversaries, with different goals and capabilities. In this paper, we propose a novel approach to perform black-box estimation of the g-vulnerability using ML. A feature of our approach is that it does not require to estimate the conditional probabilities, and that it is suitable for a large class of ML algorithms. First, we formally show the learnability for all data distributions. Then, we evaluate the performance via various experiments using k-Nearest Neighbors and Neural Networks. Our results outperform the frequentist approach when the observables domain is large.

摘要: 本文考虑了黑盒情况下系统信息泄漏的估计问题。假设学习者不了解系统的内部结构，或者系统太复杂而无法分析，并且唯一可用的信息是成对的输入-输出数据样本，可能是通过向系统提交查询获得的，也可能是由第三方提供的。以往的研究主要集中在统计频率来估计输入输出条件概率(简称频域方法)，但是当可能的输出范围较大时，这种方法并不准确。为了克服这一困难，最近使用机器学习(ML)模型研究了理想分类器的贝叶斯误差估计，由于这些模型具有学习输入输出对应的能力，因此它被证明是更精确的。然而，贝叶斯漏洞仅适用于描述一次性攻击。更通用、更灵活的漏洞度量是g漏洞，它包含几种不同类型的对手，具有不同的目标和能力。本文提出了一种利用ML对g漏洞进行黑盒估计的新方法。该方法的一个特点是不需要估计条件概率，适用于一大类最大似然算法。首先，我们形式化地证明了所有数据分布的可学习性。然后，通过使用k近邻和神经网络的各种实验对该算法的性能进行了评估。当可观域较大时，我们的结果优于频域方法。



## **16. SoK: Plausibly Deniable Storage**

SOK：貌似可否认的存储 cs.CR

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12809v1)

**Authors**: Chen Chen, Xiao Liang, Bogdan Carbunar, Radu Sion

**Abstracts**: Data privacy is critical in instilling trust and empowering the societal pacts of modern technology-driven democracies. Unfortunately, it is under continuous attack by overreaching or outright oppressive governments, including some of the world's oldest democracies. Increasingly-intrusive anti-encryption laws severely limit the ability of standard encryption to protect privacy. New defense mechanisms are needed.   Plausible deniability (PD) is a powerful property, enabling users to hide the existence of sensitive information in a system under direct inspection by adversaries. Popular encrypted storage systems such as TrueCrypt and other research efforts have attempted to also provide plausible deniability. Unfortunately, these efforts have often operated under less well-defined assumptions and adversarial models. Careful analyses often uncover not only high overheads but also outright security compromise. Further, our understanding of adversaries, the underlying storage technologies, as well as the available plausible deniable solutions have evolved dramatically in the past two decades. The main goal of this work is to systematize this knowledge. It aims to:   - identify key PD properties, requirements, and approaches;   - present a direly-needed unified framework for evaluating security and performance;   - explore the challenges arising from the critical interplay between PD and modern system layered stacks;   - propose a new "trace-oriented" PD paradigm, able to decouple security guarantees from the underlying systems and thus ensure a higher level of flexibility and security independent of the technology stack.   This work is meant also as a trusted guide for system and security practitioners around the major challenges in understanding, designing, and implementing plausible deniability into new or existing systems.

摘要: 数据隐私在灌输信任和增强现代技术驱动的民主国家的社会契约方面至关重要。不幸的是，它不断受到过度或彻底压迫的政府的攻击，包括一些世界上最古老的民主国家。越来越具侵入性的反加密法律严重限制了标准加密保护隐私的能力。需要新的防御机制。似是而非否认(PD)是一种强大的属性，使用户能够在攻击者直接检查的系统中隐藏敏感信息的存在。诸如TrueCrypt等流行的加密存储系统和其他研究努力也试图提供似是而非的否认。不幸的是，这些努力往往是在定义不太明确的假设和对抗性模型下进行的。仔细的分析往往不仅揭示了高昂的管理费用，而且还暴露了直接的安全隐患。此外，在过去的二十年里，我们对对手、底层存储技术以及可用的看似合理的可否认解决方案的理解发生了巨大的变化。这项工作的主要目标是使这方面的知识系统化。它的目标是：-确定关键的PD属性、要求和方法；-为评估安全性和性能提供一个急需的统一框架；-探索PD和现代系统分层堆栈之间的关键相互作用所带来的挑战；-提出一种新的“面向跟踪的”PD范例，该范例能够将安全保证与底层系统分离，从而确保独立于技术堆栈的更高水平的灵活性和安全性。这项工作还可以作为系统和安全从业者的可信指南，帮助他们解决在理解、设计和实现新系统或现有系统中的似是而非的可否认性方面面临的主要挑战。



## **17. On the Effect of Pruning on Adversarial Robustness**

论修剪对对手健壮性的影响 cs.CV

Published at International Conference on Computer Vision Workshop  (ICCVW), 2021

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2108.04890v2)

**Authors**: Artur Jordao, Helio Pedrini

**Abstracts**: Pruning is a well-known mechanism for reducing the computational cost of deep convolutional networks. However, studies have shown the potential of pruning as a form of regularization, which reduces overfitting and improves generalization. We demonstrate that this family of strategies provides additional benefits beyond computational performance and generalization. Our analyses reveal that pruning structures (filters and/or layers) from convolutional networks increase not only generalization but also robustness to adversarial images (natural images with content modified). Such achievements are possible since pruning reduces network capacity and provides regularization, which have been proven effective tools against adversarial images. In contrast to promising defense mechanisms that require training with adversarial images and careful regularization, we show that pruning obtains competitive results considering only natural images (e.g., the standard and low-cost training). We confirm these findings on several adversarial attacks and architectures; thus suggesting the potential of pruning as a novel defense mechanism against adversarial images.

摘要: 剪枝是一种众所周知的降低深卷积网络计算成本的机制。然而，研究表明，修剪作为正则化的一种形式具有潜力，它减少了过度拟合，提高了泛化能力。我们证明了这一系列策略在计算性能和泛化之外提供了额外的好处。我们的分析表明，卷积网络的剪枝结构(滤波器和/或层)不仅提高了泛化能力，而且对敌意图像(修改了内容的自然图像)具有鲁棒性。这样的成就是可能的，因为修剪降低了网络容量并提供了正规化，这已经被证明是对抗敌对图像的有效工具。与需要用对抗性图像训练和仔细正则化的有希望的防御机制不同，我们证明了剪枝只考虑自然图像(例如，标准和低成本的训练)就可以获得好胜的结果。我们在几个对抗性攻击和架构上证实了这些发现，从而暗示了剪枝作为对抗对抗性图像的一种新的防御机制的潜力。



## **18. Real-time Adversarial Perturbations against Deep Reinforcement Learning Policies: Attacks and Defenses**

针对深度强化学习策略的实时对抗性扰动：攻击与防御 cs.LG

13 pages, 6 figures

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2106.08746v2)

**Authors**: Buse G. A. Tekgul, Shelly Wang, Samuel Marchal, N. Asokan

**Abstracts**: Recent work has shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial perturbations. Adversaries can mislead policies of DRL agents by perturbing the state of the environment observed by the agents. Existing attacks are feasible in principle but face challenges in practice, for example by being too slow to fool DRL policies in real time. We show that using the Universal Adversarial Perturbation (UAP) method to compute perturbations, independent of the individual inputs to which they are applied to, can fool DRL policies effectively and in real time. We describe three such attack variants. Via an extensive evaluation using three Atari 2600 games, we show that our attacks are effective, as they fully degrade the performance of three different DRL agents (up to 100%, even when the $l_\infty$ bound on the perturbation is as small as 0.01). It is faster compared to the response time (0.6ms on average) of different DRL policies, and considerably faster than prior attacks using adversarial perturbations (1.8ms on average). We also show that our attack technique is efficient, incurring an online computational cost of 0.027ms on average. Using two further tasks involving robotic movement, we confirm that our results generalize to more complex DRL tasks. Furthermore, we demonstrate that the effectiveness of known defenses diminishes against universal perturbations. We propose an effective technique that detects all known adversarial perturbations against DRL policies, including all the universal perturbations presented in this paper.

摘要: 最近的研究表明，深度强化学习(DRL)策略容易受到对抗性扰动的影响。攻击者可以通过扰乱代理观察到的环境状态来误导DRL代理的策略。现有的攻击在原则上是可行的，但在实践中面临挑战，例如，速度太慢，无法实时愚弄DRL策略。我们表明，使用通用对抗扰动(UAP)方法来计算扰动，而与应用扰动的个体输入无关，可以有效且实时地欺骗DRL策略。我们描述了三种这样的攻击变体。通过使用三款Atari 2600游戏进行的广泛评估，我们表明我们的攻击是有效的，因为它们完全降低了三种不同DRL代理的性能(高达100%，即使在扰动上的$l_\infty$约束小到0.01)。它比不同DRL策略的响应时间(平均0.6ms)更快，并且比以前使用对抗性扰动的攻击(平均1.8ms)要快得多。我们还证明了我们的攻击技术是有效的，平均在线计算开销为0.027ms。使用另外两个涉及机器人移动的任务，我们确认我们的结果推广到更复杂的DRL任务。此外，我们还证明了已知防御措施对普遍扰动的有效性会降低。我们提出了一种有效的技术来检测所有已知的针对DRL策略的敌意扰动，包括本文提出的所有通用扰动。



## **19. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.06628v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知散列系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知散列的综合实证分析。具体地说，我们表明当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的改变来操纵散列值，这些改变要么是由基于梯度的方法引起的，要么是简单地通过执行标准图像转换来强制或防止散列冲突。这样的攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **20. REGroup: Rank-aggregating Ensemble of Generative Classifiers for Robust Predictions**

REGROUP：用于稳健预测的生成分类器的等级聚合集成 cs.CV

WACV,2022. Project Page : https://lokender.github.io/REGroup.html

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2006.10679v2)

**Authors**: Lokender Tiwari, Anish Madan, Saket Anand, Subhashis Banerjee

**Abstracts**: Deep Neural Networks (DNNs) are often criticized for being susceptible to adversarial attacks. Most successful defense strategies adopt adversarial training or random input transformations that typically require retraining or fine-tuning the model to achieve reasonable performance. In this work, our investigations of intermediate representations of a pre-trained DNN lead to an interesting discovery pointing to intrinsic robustness to adversarial attacks. We find that we can learn a generative classifier by statistically characterizing the neural response of an intermediate layer to clean training samples. The predictions of multiple such intermediate-layer based classifiers, when aggregated, show unexpected robustness to adversarial attacks. Specifically, we devise an ensemble of these generative classifiers that rank-aggregates their predictions via a Borda count-based consensus. Our proposed approach uses a subset of the clean training data and a pre-trained model, and yet is agnostic to network architectures or the adversarial attack generation method. We show extensive experiments to establish that our defense strategy achieves state-of-the-art performance on the ImageNet validation set.

摘要: 深度神经网络(DNNs)经常因为易受敌意攻击而受到批评。大多数成功的防御策略采用对抗性训练或随机输入转换，这通常需要重新训练或微调模型以获得合理的性能。在这项工作中，我们对预先训练的DNN的中间表示的研究导致了一个有趣的发现，指出了对对手攻击的内在鲁棒性。我们发现，我们可以通过统计表征中间层的神经响应来学习生成式分类器，以清理训练样本。多个这样的基于中间层的分类器的预测，当聚合时，显示出对对手攻击的意外鲁棒性。具体地说，我们设计了这些生成性分类器的集成，通过基于Borda计数的共识对它们的预测进行排名聚合。我们提出的方法使用了干净训练数据的子集和预先训练的模型，但与网络结构或敌意攻击生成方法无关。我们展示了大量的实验来证明我们的防御策略在ImageNet验证集上达到了最先进的性能。



## **21. Thundernna: a white box adversarial attack**

Thundernna：白盒对抗性攻击 cs.LG

10 pages, 5 figures

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12305v1)

**Authors**: Linfeng Ye

**Abstracts**: The existing work shows that the neural network trained by naive gradient-based optimization method is prone to adversarial attacks, adds small malicious on the ordinary input is enough to make the neural network wrong. At the same time, the attack against a neural network is the key to improving its robustness. The training against adversarial examples can make neural networks resist some kinds of adversarial attacks. At the same time, the adversarial attack against a neural network can also reveal some characteristics of the neural network, a complex high-dimensional non-linear function, as discussed in previous work.   In This project, we develop a first-order method to attack the neural network. Compare with other first-order attacks, our method has a much higher success rate. Furthermore, it is much faster than second-order attacks and multi-steps first-order attacks.

摘要: 已有的工作表明，基于朴素梯度优化方法训练的神经网络容易受到敌意攻击，在普通输入上添加少量恶意信息就足以使神经网络出错。同时，针对神经网络的攻击是提高其鲁棒性的关键。针对对抗性实例的训练可以使神经网络抵抗某些类型的对抗性攻击。同时，对神经网络的敌意攻击也可以揭示神经网络这一复杂的高维非线性函数的一些特征，如前文所讨论的那样。在这个项目中，我们开发了一种一阶方法来攻击神经网络。与其他一阶攻击方法相比，我们的方法具有更高的成功率。而且，它比二阶攻击和多步一阶攻击要快得多。



## **22. Subspace Adversarial Training**

子空间对抗训练 cs.LG

**SubmitDate**: 2021-11-24    [paper-pdf](http://arxiv.org/pdf/2111.12229v1)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to $0\%$ during the training. In this paper, we understand this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand the robust overfitting phenomenon in multi-step AT. To control the growth of the gradient during the training, we propose a new AT method, subspace adversarial training (Sub-AT), which constrains the AT in a carefully extracted subspace. It successfully resolves both two kinds of overfitting and hence significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, which further improves the robustness performance. As a result, we achieve the state-of-the-art single-step AT performance: our pure single-step AT can reach over $\mathbf{51}\%$ robust accuracy against strong PGD-50 attack with radius $8/255$ on CIFAR-10, even surpassing the standard multi-step PGD-10 AT with huge computational advantages. The code is released$\footnote{\url{https://github.com/nblt/Sub-AT}}$.

摘要: 单步对抗性训练(AT)因其高效、健壮而受到广泛关注。然而，存在一个严重的灾难性过拟合问题，即在训练过程中，对投影梯度下降(PGD)攻击的鲁棒精度突然下降到0美元。本文从一个新的优化角度来理解这一问题，首次揭示了每个样本快速增长的梯度与过拟合之间的密切联系，这也可以用来理解多步AT中的鲁棒过拟合现象。为了在训练过程中控制梯度的增长，我们提出了一种新的AT方法-子空间对抗训练(Sub-AT)，它将AT约束在一个仔细提取的子空间中。它成功地解决了这两种过拟合问题，从而显着提高了鲁棒性。在子空间中，我们还允许步长更大、半径更大的单步AT，进一步提高了算法的鲁棒性。因此，我们实现了最先进的单步AT性能：我们的纯单步AT在CIFAR-10上对半径为8/255美元的强PGD-50攻击可以达到超过$mathbf{51}$的鲁棒精度，甚至超过了标准的多步PGD-10AT，具有巨大的计算优势。代码是released$\footnote{\url{https://github.com/nblt/Sub-AT}}$.



## **23. Fixed Points in Cyber Space: Rethinking Optimal Evasion Attacks in the Age of AI-NIDS**

网络空间中的固定点：对AI-NIDS时代最优逃避攻击的再思考 cs.CR

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12197v1)

**Authors**: Christian Schroeder de Witt, Yongchao Huang, Philip H. S. Torr, Martin Strohmeier

**Abstracts**: Cyber attacks are increasing in volume, frequency, and complexity. In response, the security community is looking toward fully automating cyber defense systems using machine learning. However, so far the resultant effects on the coevolutionary dynamics of attackers and defenders have not been examined. In this whitepaper, we hypothesise that increased automation on both sides will accelerate the coevolutionary cycle, thus begging the question of whether there are any resultant fixed points, and how they are characterised. Working within the threat model of Locked Shields, Europe's largest cyberdefense exercise, we study blackbox adversarial attacks on network classifiers. Given already existing attack capabilities, we question the utility of optimal evasion attack frameworks based on minimal evasion distances. Instead, we suggest a novel reinforcement learning setting that can be used to efficiently generate arbitrary adversarial perturbations. We then argue that attacker-defender fixed points are themselves general-sum games with complex phase transitions, and introduce a temporally extended multi-agent reinforcement learning framework in which the resultant dynamics can be studied. We hypothesise that one plausible fixed point of AI-NIDS may be a scenario where the defense strategy relies heavily on whitelisted feature flow subspaces. Finally, we demonstrate that a continual learning approach is required to study attacker-defender dynamics in temporally extended general-sum games.

摘要: 网络攻击在数量、频率和复杂性上都在增加。作为回应，安全界正期待使用机器学习实现网络防御系统的完全自动化。然而，到目前为止，对攻击者和防御者的共同进化动态的综合影响还没有被检验。在这份白皮书中，我们假设双方自动化程度的提高将加速共同进化周期，从而回避了是否存在任何结果固定点的问题，以及它们是如何表征的。在欧洲最大的网络防御演习“锁定盾牌”的威胁模型中，我们研究了针对网络分类器的黑盒对抗性攻击。鉴于已有的攻击能力，我们质疑基于最小规避距离的最佳规避攻击框架的实用性。相反，我们提出了一种新的强化学习设置，它可以有效地产生任意的对抗性扰动。然后，我们论证了攻守不动点本身就是具有复杂相变的一般和博弈，并引入了一个时间扩展的多智能体强化学习框架，在该框架中可以研究所产生的动力学。我们假设AI-NIDS的一个看似合理的不动点可能是防御策略严重依赖于白名单上的特征流子空间的场景。最后，我们证明了在时间扩展的一般和博弈中需要一种连续学习的方法来研究攻防双方的动态行为。



## **24. Watermarking Graph Neural Networks based on Backdoor Attacks**

基于后门攻击的水印图神经网络 cs.LG

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.11024v2)

**Authors**: Jing Xu, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise on fine-tuning the model. What is more, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, it is necessary to verify the ownership of the GNN models.   In this paper, we present a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (around $100\%$) for both tasks. In addition, we experimentally show that our watermarking approach is still effective even when considering suspicious models obtained from different architectures than the owner's.

摘要: 图神经网络(GNNs)在各种实际应用中取得了良好的性能。构建一个强大的GNN模型并不是一件轻而易举的任务，因为它需要大量的训练数据、强大的计算资源和微调模型的人力专业知识。更重要的是，随着敌意攻击(如模型窃取攻击)的发展，GNNs对模型认证提出了挑战。为了避免对GNN的版权侵权，有必要对GNN模型的所有权进行验证。在这篇文章中，我们提出了一个用于GNNs的水印框架，既适用于图任务，也适用于节点分类任务。我们设计了两种策略来生成用于图分类的水印数据和一种用于节点分类任务的水印数据，2)通过训练将水印嵌入到宿主模型中以获得带水印的GNN模型，3)在黑盒环境下验证可疑模型的所有权。实验表明，对于这两个任务，我们的框架可以很高的概率(约100美元)验证GNN模型的所有权。此外，我们的实验表明，即使考虑到来自与所有者不同的架构的可疑模型，我们的水印方法仍然是有效的。



## **25. Adversarial machine learning for protecting against online manipulation**

用于防止在线操纵的对抗性机器学习 cs.LG

To appear on IEEE Internet Computing. `Accepted manuscript' version

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2111.12034v1)

**Authors**: Stefano Cresci, Marinella Petrocchi, Angelo Spognardi, Stefano Tognazzi

**Abstracts**: Adversarial examples are inputs to a machine learning system that result in an incorrect output from that system. Attacks launched through this type of input can cause severe consequences: for example, in the field of image recognition, a stop signal can be misclassified as a speed limit indication.However, adversarial examples also represent the fuel for a flurry of research directions in different domains and applications. Here, we give an overview of how they can be profitably exploited as powerful tools to build stronger learning models, capable of better-withstanding attacks, for two crucial tasks: fake news and social bot detection.

摘要: 对抗性示例是机器学习系统的输入，导致该系统的输出不正确。通过这种输入发起的攻击可能会造成严重的后果：例如，在图像识别领域，停车信号可能被错误地归类为限速指示，但敌意的例子也代表了不同领域和应用中一系列研究方向的燃料。在这里，我们概述了如何将它们作为强大的工具有利可图地利用来构建更强大的学习模型，能够更好地抵御攻击，用于两个关键任务：假新闻和社交机器人检测。



## **26. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2110.01818v3)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a new improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的搜索能力、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过四个测试函数进行了验证。仿真结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。最后，将该算法应用于神经网络的对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **27. Relevance Attack on Detectors**

对检测器的相关性攻击 cs.CV

accepted by Pattern Recognition

**SubmitDate**: 2021-11-23    [paper-pdf](http://arxiv.org/pdf/2008.06822v4)

**Authors**: Sizhe Chen, Fan He, Xiaolin Huang, Kun Zhang

**Abstracts**: This paper focuses on high-transferable adversarial attacks on detectors, which are hard to attack in a black-box manner, because of their multiple-output characteristics and the diversity across architectures. To pursue a high attack transferability, one plausible way is to find a common property across detectors, which facilitates the discovery of common weaknesses. We are the first to suggest that the relevance map from interpreters for detectors is such a property. Based on it, we design a Relevance Attack on Detectors (RAD), which achieves a state-of-the-art transferability, exceeding existing results by above 20%. On MS COCO, the detection mAPs for all 8 black-box architectures are more than halved and the segmentation mAPs are also significantly influenced. Given the great transferability of RAD, we generate the first adversarial dataset for object detection and instance segmentation, i.e., Adversarial Objects in COntext (AOCO), which helps to quickly evaluate and improve the robustness of detectors.

摘要: 针对检测器的高可移植性对抗性攻击，由于其多输出特性和跨体系结构的多样性，很难以黑盒方式进行攻击。为了追求较高的攻击可转移性，一种可行的方法是在多个检测器之间找到共同的属性，这有助于发现共同的弱点。我们是第一个提出从解释器到检测器的相关性映射就是这样一个属性的人。在此基础上，设计了一种基于检测器的关联攻击(RAD)，实现了最新的可移植性，比已有结果提高了20%以上。在MS Coco上，所有8个黑盒架构的检测图都减少了一半以上，分割图也受到了显著影响。考虑到RAD具有很强的可移植性，我们生成了第一个用于对象检测和实例分割的对抗性数据集，即上下文中的对抗性对象(AOCO)，这有助于快速评估和提高检测器的鲁棒性。



## **28. A Comparison of State-of-the-Art Techniques for Generating Adversarial Malware Binaries**

生成敌意恶意软件二进制文件的最新技术比较 cs.CR

18 pages, 7 figures; summer project report from NREIP internship at  Naval Research Laboratory

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11487v1)

**Authors**: Prithviraj Dasgupta, Zachariah Osman

**Abstracts**: We consider the problem of generating adversarial malware by a cyber-attacker where the attacker's task is to strategically modify certain bytes within existing binary malware files, so that the modified files are able to evade a malware detector such as machine learning-based malware classifier. We have evaluated three recent adversarial malware generation techniques using binary malware samples drawn from a single, publicly available malware data set and compared their performances for evading a machine-learning based malware classifier called MalConv. Our results show that among the compared techniques, the most effective technique is the one that strategically modifies bytes in a binary's header. We conclude by discussing the lessons learned and future research directions on the topic of adversarial malware generation.

摘要: 我们考虑了网络攻击者生成恶意软件的问题，其中攻击者的任务是策略性地修改现有二进制恶意软件文件中的某些字节，以便修改后的文件能够躲避恶意软件检测器(如基于机器学习的恶意软件分类器)。我们使用来自单个公开可用的恶意软件数据集的二进制恶意软件样本评估了最近的三种敌意恶意软件生成技术，并比较了它们在逃避基于机器学习的恶意软件分类器MalConv方面的性能。我们的结果表明，在比较的技术中，最有效的技术是策略性地修改二进制头中的字节。最后，我们讨论了在恶意软件生成这一主题上的经验教训和未来的研究方向。



## **29. Adversarial Examples on Segmentation Models Can be Easy to Transfer**

细分模型上的对抗性示例可以很容易地转移 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11368v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classification can be misled by adversarial examples with small and quasi-imperceptible perturbations. Furthermore, the adversarial examples created on one classification model can also fool another different model. The transferability of the adversarial examples has recently attracted a growing interest since it makes black-box attacks on classification models feasible. As an extension of classification, semantic segmentation has also received much attention towards its adversarial robustness. However, the transferability of adversarial examples on segmentation models has not been systematically studied. In this work, we intensively study this topic. First, we explore the overfitting phenomenon of adversarial examples on classification and segmentation models. In contrast to the observation made on classification models that the transferability is limited by overfitting to the source model, we find that the adversarial examples on segmentations do not always overfit the source models. Even when no overfitting is presented, the transferability of adversarial examples is limited. We attribute the limitation to the architectural traits of segmentation models, i.e., multi-scale object recognition. Then, we propose a simple and effective method, dubbed dynamic scaling, to overcome the limitation. The high transferability achieved by our method shows that, in contrast to the observations in previous work, adversarial examples on a segmentation model can be easy to transfer to other segmentation models. Our analysis and proposals are supported by extensive experiments.

摘要: 基于深度神经网络的图像分类容易受到具有小扰动和准不可察觉扰动的对抗性样本的误导。此外，在一个分类模型上创建的对抗性示例也可以欺骗另一个不同的模型。对抗性例子的可转移性最近引起了人们越来越大的兴趣，因为它使得对分类模型的黑盒攻击成为可能。语义分割作为分类的一种扩展，也因其对抗性的鲁棒性而备受关注。然而，对抗性例子在分词模型上的可转移性还没有得到系统的研究。在这项工作中，我们对这一主题进行了深入的研究。首先，我们探讨了对抗性例子在分类和分割模型上的过度拟合现象。与在分类模型上观察到的对源模型的过度拟合限制了可转移性的观察相比，我们发现关于分割的对抗性例子并不总是对源模型过度拟合。即使没有出现过拟合，对抗性例子的可转换性也是有限的。我们将其局限性归因于分割模型的结构特性，即多尺度目标识别。然后，我们提出了一种简单而有效的方法，称为动态缩放，以克服这一局限性。我们的方法达到了很高的可移植性，这表明与以前的工作相比，分割模型上的对抗性例子可以很容易地转移到其他分割模型上。我们的分析和建议得到了大量实验的支持。



## **30. Shift Invariance Can Reduce Adversarial Robustness**

移位不变性会降低对手的健壮性 cs.LG

Published as a conference paper at NeurIPS 2021

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2103.02695v3)

**Authors**: Songwei Ge, Vasu Singla, Ronen Basri, David Jacobs

**Abstracts**: Shift invariance is a critical property of CNNs that improves performance on classification. However, we show that invariance to circular shifts can also lead to greater sensitivity to adversarial attacks. We first characterize the margin between classes when a shift-invariant linear classifier is used. We show that the margin can only depend on the DC component of the signals. Then, using results about infinitely wide networks, we show that in some simple cases, fully connected and shift-invariant neural networks produce linear decision boundaries. Using this, we prove that shift invariance in neural networks produces adversarial examples for the simple case of two classes, each consisting of a single image with a black or white dot on a gray background. This is more than a curiosity; we show empirically that with real datasets and realistic architectures, shift invariance reduces adversarial robustness. Finally, we describe initial experiments using synthetic data to probe the source of this connection.

摘要: 移位不变性是CNN提高分类性能的一个重要性质。然而，我们发现对循环移位的不变性也可以导致对敌意攻击更敏感。当使用平移不变的线性分类器时，我们首先表征类之间的边缘。我们表明，裕度只能取决于信号的直流分量。然后，利用关于无限宽网络的结果，我们证明了在一些简单的情况下，完全连通和平移不变的神经网络产生线性决策边界。利用这一点，我们证明了神经网络中的平移不变性对于两类简单的情况产生了对抗性的例子，每一类都由单个图像组成，在灰色背景上有一个黑点或白点。这不仅仅是一种好奇；我们的经验表明，对于真实的数据集和现实的体系结构，移位不变性会降低对手的健壮性。最后，我们描述了使用合成数据来探索这种联系的来源的初步实验。



## **31. NTD: Non-Transferability Enabled Backdoor Detection**

NTD：启用不可转移的后门检测 cs.CR

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11157v1)

**Authors**: Yinshan Li, Hua Ma, Zhi Zhang, Yansong Gao, Alsharif Abuadbba, Anmin Fu, Yifeng Zheng, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: A backdoor deep learning (DL) model behaves normally upon clean inputs but misbehaves upon trigger inputs as the backdoor attacker desires, posing severe consequences to DL model deployments. State-of-the-art defenses are either limited to specific backdoor attacks (source-agnostic attacks) or non-user-friendly in that machine learning (ML) expertise or expensive computing resources are required. This work observes that all existing backdoor attacks have an inevitable intrinsic weakness, non-transferability, that is, a trigger input hijacks a backdoored model but cannot be effective to another model that has not been implanted with the same backdoor. With this key observation, we propose non-transferability enabled backdoor detection (NTD) to identify trigger inputs for a model-under-test (MUT) during run-time.Specifically, NTD allows a potentially backdoored MUT to predict a class for an input. In the meantime, NTD leverages a feature extractor (FE) to extract feature vectors for the input and a group of samples randomly picked from its predicted class, and then compares similarity between the input and the samples in the FE's latent space. If the similarity is low, the input is an adversarial trigger input; otherwise, benign. The FE is a free pre-trained model privately reserved from open platforms. As the FE and MUT are from different sources, the attacker is very unlikely to insert the same backdoor into both of them. Because of non-transferability, a trigger effect that does work on the MUT cannot be transferred to the FE, making NTD effective against different types of backdoor attacks. We evaluate NTD on three popular customized tasks such as face recognition, traffic sign recognition and general animal classification, results of which affirm that NDT has high effectiveness (low false acceptance rate) and usability (low false rejection rate) with low detection latency.

摘要: 后门深度学习(DL)模型在干净的输入上行为正常，但在触发器输入上行为不当，正如后门攻击者所希望的那样，这会给DL模型的部署带来严重后果。最先进的防御要么局限于特定的后门攻击(与来源无关的攻击)，要么不利于用户，因为需要机器学习(ML)专业知识或昂贵的计算资源。这项工作观察到，所有现有的后门攻击都有一个不可避免的内在弱点，即不可转移性，即一个触发器输入劫持了一个后门模型，但不能对另一个没有植入相同后门的模型有效。基于这一关键观察，我们提出了不可转移性启用后门检测(NTD)来识别运行时被测模型(MUT)的触发输入，具体地说，NTD允许潜在的后门MUT预测输入的类。同时，NTD利用特征提取器(FE)来提取输入的特征向量和从其预测类中随机选取的一组样本，然后在FE的潜在空间中比较输入和样本之间的相似度。如果相似度较低，则输入为对抗性触发器输入；否则，为良性输入。FE是一个免费的预先培训的模型，私下保留在开放平台上。由于FE和MUT来自不同的来源，攻击者不太可能将相同的后门插入到两者中。由于不可转移性，在MUT上起作用的触发效果不能转移到FE上，从而使NTD能够有效地对抗不同类型的后门攻击。我们在人脸识别、交通标志识别和一般动物分类这三个流行的定制任务上对NTD进行了评估，结果证实了NDT具有高效率(低错误接受率)和易用性(低错误拒绝率)和低检测延迟的特点。



## **32. Efficient Combinatorial Optimization for Word-level Adversarial Textual Attack**

词级对抗性文本攻击的高效组合优化 cs.CL

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2109.02229v3)

**Authors**: Shengcai Liu, Ning Lu, Cheng Chen, Ke Tang

**Abstracts**: Over the past few years, various word-level textual attack approaches have been proposed to reveal the vulnerability of deep neural networks used in natural language processing. Typically, these approaches involve an important optimization step to determine which substitute to be used for each word in the original input. However, current research on this step is still rather limited, from the perspectives of both problem-understanding and problem-solving. In this paper, we address these issues by uncovering the theoretical properties of the problem and proposing an efficient local search algorithm (LS) to solve it. We establish the first provable approximation guarantee on solving the problem in general cases.Extensive experiments involving 5 NLP tasks, 8 datasets and 26 NLP models show that LS can largely reduce the number of queries usually by an order of magnitude to achieve high attack success rates. Further experiments show that the adversarial examples crafted by LS usually have higher quality, exhibit better transferability, and can bring more robustness improvement to victim models by adversarial training.

摘要: 在过去的几年里，各种词级文本攻击方法被提出，以揭示深度神经网络在自然语言处理中的脆弱性。通常，这些方法涉及一个重要的优化步骤，以确定对原始输入中的每个单词使用哪个替身。然而，目前对这一步骤的研究还相当有限，无论是从问题理解的角度还是从问题解决的角度。在本文中，我们通过揭示问题的理论性质并提出一种有效的局部搜索算法(LS)来解决这些问题。通过对5个NLP任务、8个数据集和26个NLP模型的大量实验表明，LS算法可以极大地减少查询次数，通常可以减少一个数量级的查询次数，从而获得较高的攻击成功率。进一步的实验表明，LS生成的对抗性实例质量较高，具有较好的可移植性，通过对抗性训练可以给受害者模型带来更多的健壮性提升。



## **33. Myope Models -- Are face presentation attack detection models short-sighted?**

近视模型--面部呈现攻击检测模型是近视吗？ cs.CV

Accepted at the 2ND WORKSHOP ON EXPLAINABLE & INTERPRETABLE  ARTIFICIAL INTELLIGENCE FOR BIOMETRICS AT WACV 2022

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11127v1)

**Authors**: Pedro C. Neto, Ana F. Sequeira, Jaime S. Cardoso

**Abstracts**: Presentation attacks are recurrent threats to biometric systems, where impostors attempt to bypass these systems. Humans often use background information as contextual cues for their visual system. Yet, regarding face-based systems, the background is often discarded, since face presentation attack detection (PAD) models are mostly trained with face crops. This work presents a comparative study of face PAD models (including multi-task learning, adversarial training and dynamic frame selection) in two settings: with and without crops. The results show that the performance is consistently better when the background is present in the images. The proposed multi-task methodology beats the state-of-the-art results on the ROSE-Youtu dataset by a large margin with an equal error rate of 0.2%. Furthermore, we analyze the models' predictions with Grad-CAM++ with the aim to investigate to what extent the models focus on background elements that are known to be useful for human inspection. From this analysis we can conclude that the background cues are not relevant across all the attacks. Thus, showing the capability of the model to leverage the background information only when necessary.

摘要: 演示攻击是对生物识别系统的反复威胁，冒名顶替者试图绕过这些系统。人类经常使用背景信息作为视觉系统的上下文线索。然而，对于基于人脸的系统，背景通常被丢弃，因为人脸呈现攻击检测(PAD)模型大多是用人脸作物来训练的。这项工作提出了两种情况下的脸垫模型(包括多任务学习、对抗性训练和动态帧选择)的比较研究：有作物和没有作物两种情况下的人脸模型(包括多任务学习、对抗性训练和动态帧选择)。结果表明，当图像中存在背景时，性能始终较好。所提出的多任务方法在ROSE-YOTU数据集上以0.2%的同等错误率大大超过了最新的结果。此外，我们使用Grad-CAM++分析了模型的预测，目的是调查模型在多大程度上关注已知对人类检查有用的背景元素。根据这一分析，我们可以得出结论，背景线索在所有攻击中并不相关。因此，显示了模型仅在必要时利用背景信息的能力。



## **34. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

基于黑盒随机搜索的对抗性攻击搜索分布元学习 cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references; added acknowledgements

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.01714v3)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

摘要: 近年来，基于随机搜索方案的对抗性攻击在黑盒健壮性评估方面取得了最新的研究成果。然而，正如我们在这项工作中演示的那样，它们在不同查询预算机制中的效率取决于对底层提案分布的手动设计和启发式调优。我们研究了如何根据攻击期间获得的信息在线调整建议分发来解决这个问题。我们考虑Square攻击，这是一种最先进的基于分数的黑盒攻击，并展示了如何通过学习控制器在攻击期间在线调整建议分布的参数来提高其性能。我们在带有白盒访问的CIFAR10模型上使用基于梯度的端到端训练来训练控制器。我们证明，对于具有黑盒访问的大范围不同模型，在不同的查询机制下，将学习控制器插入攻击可持续提高其黑盒健壮性估计高达20%。我们进一步表明，学习的适应原则很好地移植到其他数据分布，如CIFAR100或ImageNet，以及目标攻击设置。



## **35. Evaluating Adversarial Attacks on ImageNet: A Reality Check on Misclassification Classes**

评估ImageNet上的敌意攻击：对误分类类的现实检验 cs.CV

Accepted for publication in 35th Conference on Neural Information  Processing Systems (NeurIPS 2021), Workshop on ImageNet: Past,Present, and  Future

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.11056v1)

**Authors**: Utku Ozbulak, Maura Pintor, Arnout Van Messem, Wesley De Neve

**Abstracts**: Although ImageNet was initially proposed as a dataset for performance benchmarking in the domain of computer vision, it also enabled a variety of other research efforts. Adversarial machine learning is one such research effort, employing deceptive inputs to fool models in making wrong predictions. To evaluate attacks and defenses in the field of adversarial machine learning, ImageNet remains one of the most frequently used datasets. However, a topic that is yet to be investigated is the nature of the classes into which adversarial examples are misclassified. In this paper, we perform a detailed analysis of these misclassification classes, leveraging the ImageNet class hierarchy and measuring the relative positions of the aforementioned type of classes in the unperturbed origins of the adversarial examples. We find that $71\%$ of the adversarial examples that achieve model-to-model adversarial transferability are misclassified into one of the top-5 classes predicted for the underlying source images. We also find that a large subset of untargeted misclassifications are, in fact, misclassifications into semantically similar classes. Based on these findings, we discuss the need to take into account the ImageNet class hierarchy when evaluating untargeted adversarial successes. Furthermore, we advocate for future research efforts to incorporate categorical information.

摘要: 虽然ImageNet最初是作为计算机视觉领域中性能基准测试的数据集提出的，但它也支持了各种其他研究工作。对抗性机器学习就是这样一种研究成果，它使用欺骗性的输入来愚弄模型做出错误的预测。为了评估对抗性机器学习领域中的攻击和防御，ImageNet仍然是最常用的数据集之一。然而，一个尚未调查的话题是对抗性例子被错误分类的类别的性质。在本文中，我们利用ImageNet的类层次结构，并测量上述类型的类在对抗性示例的未受干扰的来源中的相对位置，对这些误分类类进行了详细的分析。我们发现，实现模型到模型的对抗性转移的对抗性例子中，有$71\$被错误地归入了对底层源图像预测的前5类之一。我们还发现，非目标错误分类的很大子集实际上是误分类到语义相似的类中。基于这些发现，我们讨论了在评估非定向对抗性成功时是否需要考虑ImageNet类层次结构。此外，我们主张未来的研究努力纳入分类信息。



## **36. Selection of Source Images Heavily Influences the Effectiveness of Adversarial Attacks**

源图像的选择在很大程度上影响着对抗性攻击的效果 cs.CV

Accepted for publication in the 32nd British Machine Vision  Conference (BMVC)

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2106.07141v3)

**Authors**: Utku Ozbulak, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem

**Abstracts**: Although the adoption rate of deep neural networks (DNNs) has tremendously increased in recent years, a solution for their vulnerability against adversarial examples has not yet been found. As a result, substantial research efforts are dedicated to fix this weakness, with many studies typically using a subset of source images to generate adversarial examples, treating every image in this subset as equal. We demonstrate that, in fact, not every source image is equally suited for this kind of assessment. To do so, we devise a large-scale model-to-model transferability scenario for which we meticulously analyze the properties of adversarial examples, generated from every suitable source image in ImageNet by making use of three of the most frequently deployed attacks. In this transferability scenario, which involves seven distinct DNN models, including the recently proposed vision transformers, we reveal that it is possible to have a difference of up to $12.5\%$ in model-to-model transferability success, $1.01$ in average $L_2$ perturbation, and $0.03$ ($8/225$) in average $L_{\infty}$ perturbation when $1,000$ source images are sampled randomly among all suitable candidates. We then take one of the first steps in evaluating the robustness of images used to create adversarial examples, proposing a number of simple but effective methods to identify unsuitable source images, thus making it possible to mitigate extreme cases in experimentation and support high-quality benchmarking.

摘要: 尽管深度神经网络(DNNs)的采用率近年来有了很大的提高，但对于它们对敌意例子的脆弱性还没有找到解决方案。因此，大量的研究工作致力于修复这一弱点，许多研究通常使用源图像的子集来生成对抗性示例，将该子集中的每一幅图像视为平等。我们证明，事实上，并不是每一幅源图像都同样适合这种评估。为此，我们设计了一个大规模的模型到模型可转移性场景，在该场景中，我们通过使用三种最频繁部署的攻击，仔细分析了从ImageNet中每个合适的源映像生成的敌意示例的属性。在这个包含7个不同的DNN模型(包括最近提出的视觉转换器)的可转移性场景中，我们发现当在所有合适的候选者中随机抽样$1,000$源图像时，模型到模型的可转移性成功率可能存在高达$12.5\$的差异，平均$L2$扰动可能存在$1.01$的差异，平均$L3$($8/225$)的扰动可能存在。然后，我们在评估用于创建对抗性示例的图像的稳健性方面迈出了第一步，提出了一些简单但有效的方法来识别不合适的源图像，从而使得有可能缓解实验中的极端情况，并支持高质量的基准测试。



## **37. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

三维点云分类的潜移式攻防 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10990v1)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.

摘要: 虽然近年来人们在二维图像领域的攻防方面做了很多努力，但很少有方法研究三维模型的脆弱性。现有的3D攻击者一般对点云进行逐点摄动，产生变形的结构或离群点，这很容易被人察觉到。此外，它们的对抗性例子是在白盒环境下产生的，当转移到攻击远程黑盒模型时，白盒模型的成功率往往很低。本文从两个新的具有挑战性的角度对三维点云攻击进行了研究，提出了一种新的不可感知性转移攻击(ITA)：1)不可感知性：我们约束每个点沿其邻域曲面的法向量的扰动方向，从而生成具有相似几何性质的示例，从而增强了不可感知性。2)可转换性：我们建立了一个对抗性转换模型来产生最有害的扭曲，并加强了对抗性例子来抵抗它，提高了它们到未知黑盒模型的可转移性。此外，我们建议通过学习更具区别性的点云表示来训练更健壮的黑盒3D模型来防御此类ITA攻击。广泛的评估表明，我们的ITA攻击比最先进的攻击更具隐蔽性和可移动性，验证了我们防御战略的优越性。



## **38. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2021-11-22    [paper-pdf](http://arxiv.org/pdf/2111.10969v1)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **39. Local Model Poisoning Attacks to Byzantine-Robust Federated Learning**

对拜占庭鲁棒联合学习的局部模型中毒攻击 cs.CR

Appeared in Usenix Security Symposium 2020. Fixed an error in Theorem  1. For demo code, see https://people.duke.edu/~zg70/code/fltrust.zip . For  slides, see https://people.duke.edu/~zg70/code/Secure_Federated_Learning.pdf  . For the talk, see https://www.youtube.com/watch?v=LP4uqW18yA0

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/1911.11815v4)

**Authors**: Minghong Fang, Xiaoyu Cao, Jinyuan Jia, Neil Zhenqiang Gong

**Abstracts**: In federated learning, multiple client devices jointly learn a machine learning model: each client device maintains a local model for its local training dataset, while a master device maintains a global model via aggregating the local models from the client devices. The machine learning community recently proposed several federated learning methods that were claimed to be robust against Byzantine failures (e.g., system failures, adversarial manipulations) of certain client devices. In this work, we perform the first systematic study on local model poisoning attacks to federated learning. We assume an attacker has compromised some client devices, and the attacker manipulates the local model parameters on the compromised client devices during the learning process such that the global model has a large testing error rate. We formulate our attacks as optimization problems and apply our attacks to four recent Byzantine-robust federated learning methods. Our empirical results on four real-world datasets show that our attacks can substantially increase the error rates of the models learnt by the federated learning methods that were claimed to be robust against Byzantine failures of some client devices. We generalize two defenses for data poisoning attacks to defend against our local model poisoning attacks. Our evaluation results show that one defense can effectively defend against our attacks in some cases, but the defenses are not effective enough in other cases, highlighting the need for new defenses against our local model poisoning attacks to federated learning.

摘要: 在联合学习中，多个客户端设备共同学习机器学习模型：每个客户端设备维护其本地训练数据集的本地模型，而主设备通过聚集来自客户端设备的本地模型来维护全局模型。机器学习社区最近提出了几种联合学习方法，这些方法声称对某些客户端设备的拜占庭故障(例如，系统故障、敌意操纵)是健壮的。在这项工作中，我们首次系统地研究了针对联邦学习的局部模型中毒攻击。我们假设攻击者已经侵入了一些客户端设备，并且攻击者在学习过程中操纵了受损客户端设备上的本地模型参数，使得全局模型具有很大的测试错误率。我们将我们的攻击描述为优化问题，并将我们的攻击应用于最近的四种拜占庭鲁棒联邦学习方法。我们在四个真实数据集上的实验结果表明，我们的攻击可以显著提高联邦学习方法学习的模型的错误率，这些方法号称对一些客户端设备的拜占庭故障具有鲁棒性。我们总结了两种针对数据中毒攻击的防御措施，以防御我们的本地模型中毒攻击。我们的评估结果表明，在某些情况下，一种防御可以有效地防御我们的攻击，但在其他情况下，防御效果不够好，这突显了针对联邦学习的本地模型中毒攻击需要新的防御措施。



## **40. Denoised Internal Models: a Brain-Inspired Autoencoder against Adversarial Attacks**

去噪内部模型：一种抗敌意攻击的脑启发自动编码器 cs.CV

16 pages, 3 figures

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10844v1)

**Authors**: Kaiyuan Liu, Xingyu Li, Yi Zhou, Jisong Guan, Yurui Lai, Ge Zhang, Hang Su, Jiachen Wang, Chunxu Guo

**Abstracts**: Despite its great success, deep learning severely suffers from robustness; that is, deep neural networks are very vulnerable to adversarial attacks, even the simplest ones. Inspired by recent advances in brain science, we propose the Denoised Internal Models (DIM), a novel generative autoencoder-based model to tackle this challenge. Simulating the pipeline in the human brain for visual signal processing, DIM adopts a two-stage approach. In the first stage, DIM uses a denoiser to reduce the noise and the dimensions of inputs, reflecting the information pre-processing in the thalamus. Inspired from the sparse coding of memory-related traces in the primary visual cortex, the second stage produces a set of internal models, one for each category. We evaluate DIM over 42 adversarial attacks, showing that DIM effectively defenses against all the attacks and outperforms the SOTA on the overall robustness.

摘要: 尽管深度学习取得了巨大的成功，但它的健壮性严重不足；也就是说，深度神经网络非常容易受到对手的攻击，即使是最简单的攻击。受脑科学最新进展的启发，我们提出了去噪内部模型(DIM)，这是一种新颖的基于生成式自动编码器的模型，以应对撞击的这一挑战。模拟人脑中视觉信号处理的管道，DIM采用了两个阶段的方法。在第一阶段，DIM使用去噪器来降低输入的噪声和维数，反映了丘脑的信息预处理。第二阶段的灵感来自于初级视觉皮层中与记忆相关的痕迹的稀疏编码，第二阶段产生了一组内部模型，每个类别一个。我们对DIM42个对抗性攻击进行了评估，结果表明，DIM有效地防御了所有攻击，并且在整体鲁棒性上优于SOTA。



## **41. Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception**

具有多个接收者的网络欺骗直接消息传递网络建模 cs.CR

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.11932v1)

**Authors**: Kristen Moore, Cody J. Christopher, David Liebowitz, Surya Nepal, Renee Selvey

**Abstracts**: Cyber deception is emerging as a promising approach to defending networks and systems against attackers and data thieves. However, despite being relatively cheap to deploy, the generation of realistic content at scale is very costly, due to the fact that rich, interactive deceptive technologies are largely hand-crafted. With recent improvements in Machine Learning, we now have the opportunity to bring scale and automation to the creation of realistic and enticing simulated content. In this work, we propose a framework to automate the generation of email and instant messaging-style group communications at scale. Such messaging platforms within organisations contain a lot of valuable information inside private communications and document attachments, making them an enticing target for an adversary. We address two key aspects of simulating this type of system: modelling when and with whom participants communicate, and generating topical, multi-party text to populate simulated conversation threads. We present the LogNormMix-Net Temporal Point Process as an approach to the first of these, building upon the intensity-free modeling approach of Shchur et al.~\cite{shchur2019intensity} to create a generative model for unicast and multi-cast communications. We demonstrate the use of fine-tuned, pre-trained language models to generate convincing multi-party conversation threads. A live email server is simulated by uniting our LogNormMix-Net TPP (to generate the communication timestamp, sender and recipients) with the language model, which generates the contents of the multi-party email threads. We evaluate the generated content with respect to a number of realism-based properties, that encourage a model to learn to generate content that will engage the attention of an adversary to achieve a deception outcome.

摘要: 网络欺骗正在成为保护网络和系统免受攻击者和数据窃贼攻击的一种很有前途的方法。然而，尽管部署成本相对较低，但由于丰富的交互式欺骗性技术主要是手工制作的，大规模生成逼真内容的成本非常高。随着机器学习的最新改进，我们现在有机会将规模化和自动化带到创建逼真和诱人的模拟内容的过程中。在这项工作中，我们提出了一个框架来自动生成大规模的电子邮件和即时消息样式的群组通信。组织内的此类消息传递平台在私人通信和文档附件中包含大量有价值的信息，使其成为诱人的对手攻击目标。我们解决了模拟这种类型的系统的两个关键方面：模拟参与者何时以及与谁通信，以及生成主题多方文本以填充模拟的对话线索。我们提出了LogNormMix-net时点过程作为第一种方法，它建立在Shchur等人的无强度建模方法的基础上，为单播和多播通信创建了一个生成性模型。我们演示了如何使用微调的、预先训练的语言模型来生成令人信服的多方对话线索。通过将LogNormMix-Net TPP(生成通信时间戳、发送者和接收者)与语言模型(生成多方电子邮件线程的内容)结合起来，模拟了一个实时电子邮件服务器。我们根据一些基于现实主义的属性来评估生成的内容，这些属性鼓励模型学习生成将吸引对手注意力的内容，以实现欺骗结果。



## **42. Inconspicuous Adversarial Patches for Fooling Image Recognition Systems on Mobile Devices**

用于欺骗移动设备上的图像识别系统的不起眼的对抗性补丁 cs.CV

accpeted by iotj. arXiv admin note: substantial text overlap with  arXiv:2009.09774

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2106.15202v2)

**Authors**: Tao Bai, Jinqi Luo, Jun Zhao

**Abstracts**: Deep learning based image recognition systems have been widely deployed on mobile devices in today's world. In recent studies, however, deep learning models are shown vulnerable to adversarial examples. One variant of adversarial examples, called adversarial patch, draws researchers' attention due to its strong attack abilities. Though adversarial patches achieve high attack success rates, they are easily being detected because of the visual inconsistency between the patches and the original images. Besides, it usually requires a large amount of data for adversarial patch generation in the literature, which is computationally expensive and time-consuming. To tackle these challenges, we propose an approach to generate inconspicuous adversarial patches with one single image. In our approach, we first decide the patch locations basing on the perceptual sensitivity of victim models, then produce adversarial patches in a coarse-to-fine way by utilizing multiple-scale generators and discriminators. The patches are encouraged to be consistent with the background images with adversarial training while preserving strong attack abilities. Our approach shows the strong attack abilities in white-box settings and the excellent transferability in black-box settings through extensive experiments on various models with different architectures and training methods. Compared to other adversarial patches, our adversarial patches hold the most negligible risks to be detected and can evade human observations, which is supported by the illustrations of saliency maps and results of user evaluations. Lastly, we show that our adversarial patches can be applied in the physical world.

摘要: 基于深度学习的图像识别系统在当今世界的移动设备上得到了广泛的应用。然而，在最近的研究中，深度学习模型被证明容易受到对抗性例子的影响。对抗性例子的一种变体，称为对抗性补丁，由于其强大的攻击能力而引起了研究者的注意。尽管敌意补丁的攻击成功率很高，但由于补丁和原始图像之间的视觉不一致，它们很容易被检测出来。此外，文献中的对抗性补丁生成通常需要大量的数据，计算量大，耗时长。针对撞击面临的这些挑战，我们提出了一种利用一张图片生成不明显的敌意补丁的方法。在我们的方法中，我们首先根据受害者模型的感知敏感度来确定补丁的位置，然后利用多尺度生成器和鉴别器从粗到精的方式生成对抗性的补丁。通过对抗性训练，鼓励补丁与背景图像保持一致，同时保持较强的攻击能力。通过在不同架构和训练方法的不同模型上的大量实验，表明该方法在白盒环境下具有较强的攻击能力，在黑盒环境下具有良好的可移植性。与其他对抗性补丁相比，我们的对抗性补丁具有最容易被检测到的风险，并且可以躲避人类的观察，这一点从显著图和用户评估结果的插图中得到了支持。最后，我们证明了我们的对抗性补丁可以应用于物理世界。



## **43. Adversarial Mask: Real-World Adversarial Attack Against Face Recognition Models**

对抗性面具：针对人脸识别模型的真实对抗性攻击 cs.CV

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10759v1)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, requiring, for example, the placement of a sticker on the face, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical adversarial universal perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask effectiveness in real-world experiments by printing the adversarial pattern on a fabric medical face mask, causing the FR system to identify only 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks).

摘要: 在过去的几年里，基于深度学习的面部识别(FR)模型展示了最先进的性能，即使在冠状病毒大流行期间戴防护医用口罩变得司空见惯。考虑到这些模型的出色性能，机器学习研究界对挑战它们的鲁棒性表现出了越来越大的兴趣。最初，研究人员在数字领域进行对抗性攻击，后来将攻击转移到物理领域。然而，在许多情况下，物理域中的攻击是显眼的，例如需要在脸上放置贴纸，因此可能会在现实环境(例如机场)中引起怀疑。在这篇文章中，我们提出了对抗面具，一种针对最先进的FR模型的物理对抗普遍扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可移植性。此外，我们在真实世界的实验中验证了我们的对抗面具的有效性，方法是将对抗图案印刷在织物医用面膜上，导致FR系统只能识别戴着该面具的3.34%的参与者(相比之下，使用其他评估的面具的最低识别率为83.34%)。



## **44. Stochastic Variance Reduced Ensemble Adversarial Attack for Boosting the Adversarial Transferability**

提高对抗性转移的随机方差降低集成对抗性攻击 cs.LG

10 pages, 5 figures, submitted to a conference for review

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2111.10752v1)

**Authors**: Yifeng Xiong, Jiadong Lin, Min Zhang, John E. Hopcroft, Kun He

**Abstracts**: The black-box adversarial attack has attracted impressive attention for its practical use in the field of deep learning security, meanwhile, it is very challenging as there is no access to the network architecture or internal weights of the target model. Based on the hypothesis that if an example remains adversarial for multiple models, then it is more likely to transfer the attack capability to other models, the ensemble-based adversarial attack methods are efficient and widely used for black-box attacks. However, ways of ensemble attack are rather less investigated, and existing ensemble attacks simply fuse the outputs of all the models evenly. In this work, we treat the iterative ensemble attack as a stochastic gradient descent optimization process, in which the variance of the gradients on different models may lead to poor local optima. To this end, we propose a novel attack method called the stochastic variance reduced ensemble (SVRE) attack, which could reduce the gradient variance of the ensemble models and take full advantage of the ensemble attack. Empirical results on the standard ImageNet dataset demonstrate that the proposed method could boost the adversarial transferability and outperforms existing ensemble attacks significantly.

摘要: 黑盒对抗性攻击因其在深度学习安全领域的实际应用而备受关注，同时，由于不能访问目标模型的网络结构或内部权重，因此具有很大的挑战性。基于这样的假设，如果一个示例在多个模型上保持对抗性，则攻击能力更有可能转移到其他模型上，基于集成的对抗性攻击方法是一种有效的、广泛应用于黑盒攻击的方法。然而，集成攻击方式的研究相对较少，现有的集成攻击只是简单地将所有模型的输出均匀地融合。在本文中，我们将迭代集成攻击看作一个随机梯度下降优化过程，其中不同模型上梯度的变化可能导致局部最优解较差。为此，我们提出了一种新的攻击方法，称为随机方差减少集成(SVRE)攻击，它可以降低集成模型的梯度方差，并充分利用集成攻击的优势。在标准ImageNet数据集上的实验结果表明，该方法可以提高攻击的对抗性可转换性，并明显优于现有的集成攻击。



## **45. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

AEVA：基于对抗性极值分析的黑盒后门检测 cs.LG

**SubmitDate**: 2021-11-21    [paper-pdf](http://arxiv.org/pdf/2110.14880v3)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.

摘要: 深度神经网络(DNNs)被证明是易受后门攻击的。通过将后门触发器注入到训练示例中，通常将后门嵌入到目标DNN中，这可能导致目标DNN对与后门触发器附加的输入进行错误分类。现有的后门检测方法通常需要访问原始有毒训练数据、目标DNN的参数或每个给定输入的预测置信度，这在许多真实世界应用中是不切实际的，例如在设备上部署的DNN。我们解决了黑盒硬标签后门检测问题，其中DNN是完全黑盒的，并且只有其最终输出标签是可访问的。我们从优化的角度来研究这个问题，并证明了后门检测的目标是由一个对抗性目标限定的。进一步的理论和实证研究表明，这种对抗性目标导致了一个具有高度偏态分布的解决方案；在一个被后门感染的例子的对抗性地图中经常观察到一个奇点，我们称之为对抗性奇点现象。基于这一观察，我们提出了对抗性极值分析(AEVA)来检测黑盒神经网络中的后门。AEVA是基于对敌方地图的极值分析，通过蒙特卡洛梯度估计计算出来的。通过对多个流行任务和后门攻击的大量实验证明，我们的方法在黑盒硬标签场景下检测后门攻击是有效的。



## **46. Are Vision Transformers Robust to Patch Perturbations?**

视觉变形器对补丁扰动有健壮性吗？ cs.CV

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10659v1)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: The recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-wise input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of vision transformers to patch-wise perturbations. Surprisingly, we find that vision transformers are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we conduct extensive qualitative and quantitative experiments to understand the robustness to patch perturbations. We have revealed that ViT's stronger robustness to natural corrupted patches and higher vulnerability against adversarial patches are both caused by the attention mechanism. Specifically, the attention model can help improve the robustness of vision transformers by effectively ignoring natural corrupted patches. However, when vision transformers are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake.

摘要: 近年来，视觉变换器(VIT)在图像分类中表现出了令人印象深刻的性能，使其成为卷积神经网络(CNN)的一种有前途的替代方案。与CNN不同，VIT将输入图像表示为图像补丁序列。基于补丁的输入图像表示使得以下问题变得有趣：与CNN相比，当单个输入图像补丁受到自然破坏或敌意扰动时，VIT的性能如何？在这项工作中，我们研究了视觉转换器对面片扰动的鲁棒性。令人惊讶的是，我们发现视觉转换器对自然腐烂的补丁比CNN更健壮，而它们更容易受到敌意补丁的攻击。此外，我们还进行了大量的定性和定量实验，以了解该算法对补丁扰动的鲁棒性。我们发现，VIT对自然破坏补丁具有较强的健壮性，对敌意补丁具有较高的脆弱性，这都是由注意机制造成的。具体地说，注意力模型可以通过有效地忽略自然损坏的斑块来帮助提高视觉转换器的鲁棒性。然而，当视觉变形器受到敌人的攻击时，注意力机制很容易被愚弄，将注意力更多地集中在对手扰乱的补丁上，从而导致错误。



## **47. Modeling Design and Control Problems Involving Neural Network Surrogates**

涉及神经网络代理的建模设计与控制问题 math.OC

24 Pages, 11 Figures

**SubmitDate**: 2021-11-20    [paper-pdf](http://arxiv.org/pdf/2111.10489v1)

**Authors**: Dominic Yang, Prasanna Balaprakash, Sven Leyffer

**Abstracts**: We consider nonlinear optimization problems that involve surrogate models represented by neural networks. We demonstrate first how to directly embed neural network evaluation into optimization models, highlight a difficulty with this approach that can prevent convergence, and then characterize stationarity of such models. We then present two alternative formulations of these problems in the specific case of feedforward neural networks with ReLU activation: as a mixed-integer optimization problem and as a mathematical program with complementarity constraints. For the latter formulation we prove that stationarity at a point for this problem corresponds to stationarity of the embedded formulation. Each of these formulations may be solved with state-of-the-art optimization methods, and we show how to obtain good initial feasible solutions for these methods. We compare our formulations on three practical applications arising in the design and control of combustion engines, in the generation of adversarial attacks on classifier networks, and in the determination of optimal flows in an oil well network.

摘要: 我们考虑涉及以神经网络为代表的代理模型的非线性优化问题。我们首先演示了如何将神经网络评估直接嵌入到优化模型中，强调了这种方法可能会阻止收敛的一个困难，然后描述了这类模型的平稳性。然后，在具有RELU激活的前馈神经网络的具体情况下，我们给出了这些问题的两种可供选择的形式：作为混合整数优化问题和作为具有互补约束的数学规划。对于后一种公式，我们证明了该问题在某一点的平稳性对应于嵌入公式的平稳性。这些公式中的每一个都可以用最先进的优化方法求解，我们展示了如何为这些方法获得良好的初始可行解。我们比较了我们的公式在内燃机设计和控制、产生对分类器网络的敌意攻击和确定油井网络中的最优流量的三个实际应用中的应用。



## **48. Zero-Shot Certified Defense against Adversarial Patches with Vision Transformers**

使用Vision Transformers对敌方补丁进行零射击认证防御 cs.CV

12 pages, 5 figures

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10481v1)

**Authors**: Yuheng Huang, Yuanchun Li

**Abstracts**: Adversarial patch attack aims to fool a machine learning model by arbitrarily modifying pixels within a restricted region of an input image. Such attacks are a major threat to models deployed in the physical world, as they can be easily realized by presenting a customized object in the camera view. Defending against such attacks is challenging due to the arbitrariness of patches, and existing provable defenses suffer from poor certified accuracy. In this paper, we propose PatchVeto, a zero-shot certified defense against adversarial patches based on Vision Transformer (ViT) models. Rather than training a robust model to resist adversarial patches which may inevitably sacrifice accuracy, PatchVeto reuses a pretrained ViT model without any additional training, which can achieve high accuracy on clean inputs while detecting adversarial patched inputs by simply manipulating the attention map of ViT. Specifically, each input is tested by voting over multiple inferences with different attention masks, where at least one inference is guaranteed to exclude the adversarial patch. The prediction is certifiably robust if all masked inferences reach consensus, which ensures that any adversarial patch would be detected with no false negative. Extensive experiments have shown that PatchVeto is able to achieve high certified accuracy (e.g. 67.1% on ImageNet for 2%-pixel adversarial patches), significantly outperforming state-of-the-art methods. The clean accuracy is the same as vanilla ViT models (81.8% on ImageNet) since the model parameters are directly reused. Meanwhile, our method can flexibly handle different adversarial patch sizes by simply changing the masking strategy.

摘要: 对抗性补丁攻击旨在通过任意修改输入图像的受限区域内的像素来愚弄机器学习模型。这类攻击是对部署在物理世界中的模型的主要威胁，因为它们可以通过在相机视图中呈现自定义对象来轻松实现。由于补丁程序的任意性，防御此类攻击具有挑战性，而且现有的可证明防御系统存在认证准确性差的问题。在本文中，我们提出了PatchVeto，一种基于视觉转换器(VIT)模型的零命中认证的恶意补丁防御方案。PatchVeto没有训练健壮的模型来抵抗不可避免地会牺牲准确性的对抗性补丁，而是重用了预先训练的VIT模型，无需任何额外的训练，通过简单地操作VIT的注意图，可以在检测干净输入的同时检测到对抗性补丁输入。具体地说，通过对具有不同注意掩码的多个推论进行投票来测试每个输入，其中至少有一个推论被保证排除敌意补丁。如果所有掩蔽的推论都达到共识，则预测是可证明的稳健的，这确保了任何敌意补丁都将被检测到而不会出现假阴性。广泛的实验表明，PatchVeto能够达到很高的认证准确率(例如，ImageNet上2%像素的对抗性补丁的准确率为67.1%)，远远超过最先进的方法。由于直接重用了模型参数，因此其清洁精度与普通VIT模型相同(在ImageNet上为81.8%)。同时，我们的方法只需改变掩蔽策略，就可以灵活地处理不同的敌意补丁大小。



## **49. Rethinking Clustering for Robustness**

重新考虑集群以实现健壮性 cs.LG

Accepted to the 32nd British Machine Vision Conference (BMVC'21)

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2006.07682v3)

**Authors**: Motasem Alfarra, Juan C. Pérez, Adel Bibi, Ali Thabet, Pablo Arbeláez, Bernard Ghanem

**Abstracts**: This paper studies how encouraging semantically-aligned features during deep neural network training can increase network robustness. Recent works observed that Adversarial Training leads to robust models, whose learnt features appear to correlate with human perception. Inspired by this connection from robustness to semantics, we study the complementary connection: from semantics to robustness. To do so, we provide a robustness certificate for distance-based classification models (clustering-based classifiers). Moreover, we show that this certificate is tight, and we leverage it to propose ClusTR (Clustering Training for Robustness), a clustering-based and adversary-free training framework to learn robust models. Interestingly, \textit{ClusTR} outperforms adversarially-trained networks by up to $4\%$ under strong PGD attacks.

摘要: 本文研究了在深度神经网络训练过程中鼓励语义对齐的特征如何提高网络的鲁棒性。最近的工作观察到，对抗性训练导致健壮的模型，其学习的特征似乎与人类的感知相关。受这种从鲁棒性到语义的联系的启发，我们研究了这种互补的联系：从语义到鲁棒性。为此，我们为基于距离的分类模型(基于聚类的分类器)提供了健壮性证书。此外，我们还证明了该证书是严格的，并利用该证书提出了ClusTR(聚类健壮性训练)，这是一个基于聚类的、无对手的训练框架，用于学习健壮模型。有趣的是，在强PGD攻击下，textit{ClusTR}的性能比经过恶意训练的网络高出4美元。



## **50. Meta Adversarial Perturbations**

元对抗扰动 cs.LG

**SubmitDate**: 2021-11-19    [paper-pdf](http://arxiv.org/pdf/2111.10291v1)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.

摘要: 已经提出了大量的攻击方法来生成对抗性实例，其中迭代方法已被证明具有发现强攻击的能力。然而，计算新数据点的对抗性扰动需要从头开始解决耗时的优化问题。要生成更强的攻击，通常需要更新迭代次数更多的数据点。本文证明了元对抗扰动(MAP)的存在性，并提出了一种计算这种扰动的算法。MAP是一种较好的初始化方法，它只通过一步梯度上升更新就会导致自然图像在更新后被高概率地误分类。我们进行了大量的实验，实验结果表明，最新的深度神经网络容易受到元扰动的影响。我们进一步表明，这些扰动不仅是图像不可知的，而且也是模型不可知的，因为单个扰动很好地概括了不可见的数据点和不同的神经网络结构。



