# Latest Adversarial Attack Papers
**update at 2021-12-07 23:56:41**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Context-Aware Transfer Attacks for Object Detection**

面向对象检测的上下文感知传输攻击 cs.CV

accepted to AAAI 2022

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03223v1)

**Authors**: Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to $20$ percentage points improvement in performance compared to the other state-of-the-art methods.

摘要: 近年来，针对图像分类器的黑盒传输攻击得到了广泛的研究。相比之下，针对目标检测器的传输攻击研究进展甚微。对象检测器对图像进行整体观察，并且一个对象(或其缺失)的检测通常取决于场景中的其他对象。这使得这类检测器固有的上下文感知和敌意攻击比那些针对图像分类器的攻击更具挑战性。本文提出了一种生成对象检测器上下文感知攻击的新方法。通过使用对象的共现及其相对位置和大小作为上下文信息，我们可以成功地生成具有针对性的误分类攻击，从而在黑盒对象检测器上获得比现有技术更高的传输成功率。我们使用Pascal VOC和MS Coco数据集的图像在各种对象探测器上测试了我们的方法，与其他最先进的方法相比，性能提高了多达20个百分点。



## **2. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

利用自监督学习提高说话人确认的对抗性 cs.SD

Accepted by TASLP

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2106.00273v3)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.

摘要: 以往的研究表明，自动说话人验证(ASV)很容易受到恶意欺骗攻击，如重放、合成语音以及最近出现的敌意攻击。人们一直致力于保护ASV免受重播和合成语音的攻击，然而，只探索了几种方法来应对对抗性攻击。现有的撞击对抗性攻击方法都需要生成对抗性样本的知识，但是防御者要知道野外攻击者使用的确切攻击算法是不切实际的。这项工作是第一批在不知道具体攻击算法的情况下对ASV进行对抗性防御的工作之一。受自监督学习模型(SSLMs)减少输入表面噪声和从中断样本中重构干净样本等优点的启发，本文将对抗性扰动视为一种噪声，利用SSLMs对ASV进行对抗性防御。具体地说，我们提出从两个角度进行对抗性防御：1)对抗性扰动净化和2)对抗性扰动检测。实验结果表明，我们的检测模块通过检测敌意样本，有效地屏蔽了ASV，准确率在80%左右。此外，由于ASV的对抗防御性能没有统一的评价指标，本文还考虑了基于净化和基于检测的方法，形式化了对抗防御的评价指标。我们真诚地鼓励今后的工作在拟议的评价框架基础上对其方法进行基准。



## **3. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

DNN模型的对抗性范例检测：综述与实验比较 cs.CV

To be published on Artificial Intelligence Review journal (after  minor revision)

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2105.00203v3)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.

摘要: 深度学习(DL)在许多与人类相关的任务中取得了巨大的成功，这使得它被许多基于计算机视觉的应用所采用，如安全监控系统、自动驾驶汽车和医疗保健。此类安全关键型应用程序一旦具备了克服安全关键型挑战的能力，就必须为成功部署画上句号。在这些挑战中，包括防御或/和检测对抗性示例(AEs)。攻击者可以小心翼翼地制造称为扰动的小噪音，通常是难以察觉的，并将其添加到干净的图像中，以生成AE。AE的目的是愚弄DL模型，使其成为DL应用程序的潜在风险。文献中提出了许多测试时间逃避攻击和对策，即防御或检测方法。此外，很少有综述和调查发表，从理论上给出了威胁的分类和对策方法，而对声发射检测方法的关注较少。本文以图像分类任务为研究对象，对神经网络分类器测试时间逃避攻击的检测方法进行了综述。对这些方法进行了详细的讨论，并给出了在四个数据集上的不同场景下八个最先进检测器的实验结果。我们还对这一研究方向提出了潜在的挑战和未来的展望。



## **4. Robust Person Re-identification with Multi-Modal Joint Defence**

基于多模态联合防御的鲁棒人物再识别 cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.09571v2)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.

摘要: 基于度量学习的人物识别(ReID)系统继承了深层神经网络(DNNs)易被恶意度量攻击欺骗的弱点。现有的工作主要依靠对抗性训练进行度量防御，更多的方法还没有得到充分的研究。通过研究攻击对底层特征的影响，提出了有针对性的度量攻击方法和防御方法。在度量攻击方面，我们利用局部颜色偏差来构造输入的类内变异来攻击颜色特征。在度量防御方面，我们提出了一种包括主动防御和被动防御两部分的联合防御方法。主动防御通过从多模态图像构造不同的输入来增强模型对颜色变化的鲁棒性和跨多模态的结构关系的学习，而被动防御通过迂回缩放利用结构特征在变化的像素空间中的不变性来保留结构特征，同时消除一些对抗性噪声。大量实验表明，与现有的对抗性度量防御方法相比，本文提出的联合防御方法不仅可以同时防御多个攻击，而且没有显着降低模型的泛化能力。代码可在https://github.com/finger-monkey/multi-modal_joint_defence.上获得



## **5. ML Attack Models: Adversarial Attacks and Data Poisoning Attacks**

ML攻击模型：对抗性攻击和数据中毒攻击 cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02797v1)

**Authors**: Jing Lin, Long Dang, Mohamed Rahouti, Kaiqi Xiong

**Abstracts**: Many state-of-the-art ML models have outperformed humans in various tasks such as image classification. With such outstanding performance, ML models are widely used today. However, the existence of adversarial attacks and data poisoning attacks really questions the robustness of ML models. For instance, Engstrom et al. demonstrated that state-of-the-art image classifiers could be easily fooled by a small rotation on an arbitrary image. As ML systems are being increasingly integrated into safety and security-sensitive applications, adversarial attacks and data poisoning attacks pose a considerable threat. This chapter focuses on the two broad and important areas of ML security: adversarial attacks and data poisoning attacks.

摘要: 许多最先进的ML模型在图像分类等各种任务中的表现都超过了人类。ML模型以其出色的性能在今天得到了广泛的应用。然而，敌意攻击和数据中毒攻击的存在确实对ML模型的稳健性提出了质疑。例如，Engstrom等人。展示了最先进的图像分类器可以很容易地被任意图像上的小旋转所愚弄。随着ML系统越来越多地集成到安全和安全敏感的应用程序中，对抗性攻击和数据中毒攻击构成了相当大的威胁。本章重点介绍ML安全的两个广泛而重要的领域：对抗性攻击和数据中毒攻击。



## **6. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v4)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的搜索能力、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过四个测试函数进行了验证。仿真结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。最后，将该算法应用于神经网络的对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **7. Staring Down the Digital Fulda Gap Path Dependency as a Cyber Defense Vulnerability**

向下看数字富尔达缺口路径依赖是一个网络防御漏洞 cs.CY

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02773v1)

**Authors**: Jan Kallberg

**Abstracts**: Academia, homeland security, defense, and media have accepted the perception that critical infrastructure in a future cyber war cyber conflict is the main gateway for a massive cyber assault on the U.S. The question is not if the assumption is correct or not, the question is instead of how did we arrive at that assumption. The cyber paradigm considers critical infrastructure the primary attack vector for future cyber conflicts. The national vulnerability embedded in critical infrastructure is given a position in the cyber discourse as close to an unquestionable truth as a natural law.   The American reaction to Sept. 11, and any attack on U.S. soil, hint to an adversary that attacking critical infrastructure to create hardship for the population could work contrary to the intended softening of the will to resist foreign influence. It is more likely that attacks that affect the general population instead strengthen the will to resist and fight, similar to the British reaction to the German bombing campaign Blitzen in 1940. We cannot rule out attacks that affect the general population, but there are not enough adversarial offensive capabilities to attack all 16 critical infrastructure sectors and gain strategic momentum. An adversary has limited cyberattack capabilities and needs to prioritize cyber targets that are aligned with the overall strategy. Logically, an adversary will focus their OCO on operations that has national security implications and support their military operations by denying, degrading, and confusing the U.S. information environment and U.S. cyber assets.

摘要: 学术界、国土安全、国防和媒体已经接受了这样的看法，即未来网络战中的关键基础设施网络冲突是针对美国的大规模网络攻击的主要门户。问题不是假设是否正确，而是我们如何得出这个假设。网络范式认为关键基础设施是未来网络冲突的主要攻击载体。关键基础设施中嵌入的国家脆弱性在网络话语中被赋予了与自然法一样接近毋庸置疑的真理的地位。美国对9·11事件的反应。11，以及对美国领土的任何袭击，都暗示着对手，攻击关键基础设施给人民带来困难，可能与抵制外国影响的意愿软化的意图背道而驰。更有可能的是，影响到普通民众的袭击反而增强了抵抗和战斗的意志，类似于英国对1940年德国轰炸行动Blitzen的反应。我们不能排除影响到普通民众的袭击，但没有足够的对抗性进攻能力来攻击所有16个关键基础设施部门，并获得战略势头。对手的网络攻击能力有限，需要优先考虑与整体战略一致的网络目标。从逻辑上讲，对手将把他们的OCO集中在影响国家安全的行动上，并通过否认、贬低和混淆美国信息环境和美国网络资产来支持他们的军事行动。



## **8. Label-Only Membership Inference Attacks**

仅标签成员关系推理攻击 cs.CR

16 pages, 11 figures, 2 tables Revision 2: 19 pages, 12 figures, 3  tables. Improved text and additional experiments. Final ICML paper

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2007.14321v3)

**Authors**: Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot

**Abstracts**: Membership inference attacks are one of the simplest forms of privacy leakage for machine learning models: given a data point and model, determine whether the point was used to train the model. Existing membership inference attacks exploit models' abnormal confidence when queried on their training data. These attacks do not apply if the adversary only gets access to models' predicted labels, without a confidence measure. In this paper, we introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples. We empirically show that our label-only membership inference attacks perform on par with prior attacks that required access to model confidences. We further demonstrate that label-only attacks break multiple defenses against membership inference attacks that (implicitly or explicitly) rely on a phenomenon we call confidence masking. These defenses modify a model's confidence scores in order to thwart attacks, but leave the model's predicted labels unchanged. Our label-only attacks demonstrate that confidence-masking is not a viable defense strategy against membership inference. Finally, we investigate worst-case label-only attacks, that infer membership for a small number of outlier data points. We show that label-only attacks also match confidence-based attacks in this setting. We find that training models with differential privacy and (strong) L2 regularization are the only known defense strategies that successfully prevents all attacks. This remains true even when the differential privacy budget is too high to offer meaningful provable guarantees.

摘要: 成员关系推理攻击是机器学习模型隐私泄露的最简单形式之一：给定一个数据点和模型，确定该点是否被用来训练该模型。现有的隶属度推理攻击利用模型在查询训练数据时的异常置信度。如果对手只能访问模型的预测标签，而没有置信度度量，则这些攻击不适用。在本文中，我们引入了仅标签成员关系推理攻击。我们的攻击不依赖于置信度分数，而是评估模型的预测标签在扰动下的鲁棒性，以获得细粒度的成员资格信号。这些扰动包括常见的数据扩充或对抗性示例。我们的经验表明，我们的仅标签成员关系推理攻击的性能与之前需要访问模型可信度的攻击相当。我们进一步证明，仅标签攻击打破了对(隐式或显式)依赖于我们称为置信度掩蔽现象的成员关系推断攻击的多个防御。这些防御措施修改模型的置信度分数以阻止攻击，但保持模型的预测标签不变。我们的仅标签攻击表明，置信度掩蔽不是一种可行的针对成员关系推断的防御策略。最后，我们研究了最坏情况下的仅标签攻击，即推断少量离群点的成员资格。我们表明，在此设置下，仅标签攻击也与基于置信度的攻击相匹配。我们发现，具有差异隐私和(强)L2正则化的训练模型是唯一已知的成功阻止所有攻击的防御策略。即使差别隐私预算太高，无法提供有意义的、可证明的保证，这一点仍然成立。



## **9. Learning Swarm Interaction Dynamics from Density Evolution**

从密度演化中学习群体相互作用动力学 eess.SY

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02675v1)

**Authors**: Christos Mavridis, Amoolya Tirumalai, John Baras

**Abstracts**: We consider the problem of understanding the coordinated movements of biological or artificial swarms. In this regard, we propose a learning scheme to estimate the coordination laws of the interacting agents from observations of the swarm's density over time. We describe the dynamics of the swarm based on pairwise interactions according to a Cucker-Smale flocking model, and express the swarm's density evolution as the solution to a system of mean-field hydrodynamic equations. We propose a new family of parametric functions to model the pairwise interactions, which allows for the mean-field macroscopic system of integro-differential equations to be efficiently solved as an augmented system of PDEs. Finally, we incorporate the augmented system in an iterative optimization scheme to learn the dynamics of the interacting agents from observations of the swarm's density evolution over time. The results of this work can offer an alternative approach to study how animal flocks coordinate, create new control schemes for large networked systems, and serve as a central part of defense mechanisms against adversarial drone attacks.

摘要: 我们考虑理解生物或人造蜂群的协调运动的问题。在这方面，我们提出了一种学习方案，通过观察种群密度随时间的变化来估计相互作用Agent的协调规律。我们根据Cucker-Smer群集模型描述了基于成对相互作用的群体动力学，并将群体密度演化表示为平均场流体动力学方程组的解。我们提出了一族新的参数函数族来模拟两两相互作用，使得平均场宏观积分微分方程组可以作为一个增广的偏微分方程组有效地求解。最后，我们将增广系统结合到迭代优化方案中，通过观察种群密度随时间的演变来学习交互Agent的动态。这项工作的结果可以提供另一种方法来研究动物群是如何协调的，为大型网络系统创造新的控制方案，并作为对抗无人机攻击的防御机制的核心部分。



## **10. Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness**

随机本地赢家通吃网络实现深刻的对手鲁棒性 cs.LG

Bayesian Deep Learning Workshop, NeurIPS 2021

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02671v1)

**Authors**: Konstantinos P. Panousis, Sotirios Chatzis, Sergios Theodoridis

**Abstracts**: This work explores the potency of stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA), against powerful (gradient-based) white-box and black-box adversarial attacks; we especially focus on Adversarial Training settings. In our work, we replace the conventional ReLU-based nonlinearities with blocks comprising locally and stochastically competing linear units. The output of each network layer now yields a sparse output, depending on the outcome of winner sampling in each block. We rely on the Variational Bayesian framework for training and inference; we incorporate conventional PGD-based adversarial training arguments to increase the overall adversarial robustness. As we experimentally show, the arising networks yield state-of-the-art robustness against powerful adversarial attacks while retaining very high classification rate in the benign case.

摘要: 这项工作探索了基于随机竞争的激活，即随机局部赢家通吃(LWTA)，对抗强大的(基于梯度的)白盒和黑盒对抗性攻击的有效性；我们特别关注对抗性训练环境。在我们的工作中，我们用由局部和随机竞争的线性单元组成的块来代替传统的基于REU的非线性。现在，每个网络层的输出都会产生稀疏输出，具体取决于每个挡路中获胜者采样的结果。我们依靠变分贝叶斯框架进行训练和推理；我们结合了传统的基于PGD的对抗性训练论据，以增加对抗性的整体健壮性。正如我们的实验所表明的那样，出现的网络对强大的对手攻击产生了最先进的健壮性，同时在良性情况下保持了非常高的分类率。



## **11. Formalizing and Estimating Distribution Inference Risks**

配电推理风险的形式化与估计 cs.LG

Shorter version of work available at arXiv:2106.03699 Update: New  version with more theoretical results and a deeper exploration of results

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v4)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型基于私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即，生成捕获有关分布的统计属性的模型。在Yeom等人的成员关系推理框架的启发下，我们提出了分布推理攻击的形式化定义，该定义足够通用，可以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均节点度或聚类系数。为了了解分布推理风险，我们引入了一个度量，通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，对观察到的泄漏进行量化。我们报告了使用新颖的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。



## **12. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

ADV-4-ADV：通过对抗性领域适应挫败不断变化的对抗性扰动 cs.CV

9 pages

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.00428v2)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.

摘要: 虽然对抗性训练对对抗特定的对抗性干扰是有用的，但事实证明，它们也不能有效地概括出与用于训练的攻击不同的攻击。然而，我们观察到这种低效与领域适应性有内在的联系，这是深度学习中的另一个关键问题，对抗性领域适应似乎是一个有希望的解决方案。因此，我们提出了ADV-4-ADV作为一种新的对抗性训练方法，旨在保持对不可见的对抗性扰动的鲁棒性。从本质上讲，ADV-4-ADV将遭受不同扰动的攻击视为不同的域，并利用敌对域自适应的能力，旨在去除域/攻击特定的特征。这迫使训练后的模型学习健壮的领域不变表示，进而增强其泛化能力。在Fashion-MNIST、SVHN、CIFAR-10和CIFAR-100上的广泛评估表明，由ADV-4-ADV基于简单攻击(例如FGSM)构造的样本训练的模型可以推广到更高级的攻击(例如PGD)，并且性能超过了在这些数据集上的最新建议。



## **13. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2009.04131v3)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **14. Statically Detecting Adversarial Malware through Randomised Chaining**

通过随机链静态检测敌意恶意软件 cs.CR

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2111.14037v2)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.

摘要: 随着恶意软件攻击的快速增长，越来越多的反病毒开发人员考虑将机器学习技术部署到他们的产品中。近年来，研究人员和开发人员发布了各种基于机器学习的恶意软件检测高精度检测器。虽然有许多基于机器学习的恶意软件检测器可用，但它们面临着各种机器学习目标攻击，包括逃避和敌意攻击。该项目探讨了敌意实例如何以及为什么躲避恶意软件检测器，然后提出了一种随机链接的方法来静态防御敌意恶意软件。这项研究对于打击相关的恶意软件网络犯罪至关重要。



## **15. Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing**

逆稳健假设检验的广义似然比检验 stat.ML

Submitted to the IEEE Transactions on Signal Processing

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.02209v1)

**Authors**: Bhagyashree Puranik, Upamanyu Madhow, Ramtin Pedarsani

**Abstracts**: Machine learning models are known to be susceptible to adversarial attacks which can cause misclassification by introducing small but well designed perturbations. In this paper, we consider a classical hypothesis testing problem in order to develop fundamental insight into defending against such adversarial perturbations. We interpret an adversarial perturbation as a nuisance parameter, and propose a defense based on applying the generalized likelihood ratio test (GLRT) to the resulting composite hypothesis testing problem, jointly estimating the class of interest and the adversarial perturbation. While the GLRT approach is applicable to general multi-class hypothesis testing, we first evaluate it for binary hypothesis testing in white Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations, for which a known minimax defense optimizing for the worst-case attack provides a benchmark. We derive the worst-case attack for the GLRT defense, and show that its asymptotic performance (as the dimension of the data increases) approaches that of the minimax defense. For non-asymptotic regimes, we show via simulations that the GLRT defense is competitive with the minimax approach under the worst-case attack, while yielding a better robustness-accuracy tradeoff under weaker attacks. We also illustrate the GLRT approach for a multi-class hypothesis testing problem, for which a minimax strategy is not known, evaluating its performance under both noise-agnostic and noise-aware adversarial settings, by providing a method to find optimal noise-aware attacks, and heuristics to find noise-agnostic attacks that are close to optimal in the high SNR regime.

摘要: 众所周知，机器学习模型容易受到敌意攻击，这种攻击可能会通过引入小但设计良好的扰动而导致误分类。在这篇文章中，我们考虑了一个经典的假设检验问题，以发展对这种敌对扰动的防御的基本见解。我们将敌意扰动解释为干扰参数，并提出了一种基于广义似然比检验(GLRT)的防御方法，将广义似然比检验应用于由此产生的复合假设检验问题，联合估计感兴趣的类别和对抗性扰动。虽然GLRT方法适用于一般的多类假设检验，但我们首先评估了它在高斯白噪声中范数有界的对抗扰动下的二元假设检验，一个已知的针对最坏情况攻击的极小极大防御优化提供了一个基准。我们推导了GLRT防御的最坏情况攻击，并证明了它的渐近性能(随着数据维数的增加)接近极小极大防御的渐近性能。对于非渐近体制，我们通过仿真表明，在最坏情况下，广义似然比防御是基于极小极大方法的好胜防御，而在较弱攻击下获得了较好的稳健性和准确性折衷。我们还举例说明了GLRT方法用于多类假设检验问题，对于未知的极小极大策略，通过提供一种寻找最优噪声感知攻击的方法和寻找在高信噪比条件下接近最优的噪声不可知攻击的启发式方法，来评估其在噪声不可知性和噪声感知对抗环境下的性能。



## **16. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线传感的对策 cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01967v1)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，今天无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过偷听标准通信信号，窃听者可以获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为对抗敌意无线传感的一种新的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行模糊处理。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **17. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

注意方框：$l_1$-针对图像分类器的稀疏对抗性攻击的APGD cs.LG

In ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.01208v2)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

摘要: 我们证明了当同时考虑象域$[0，1]^d$时，所建立的$l_1$投影梯度下降(PGD)攻击是次优的，因为它们没有考虑到有效的威胁模型是$l_1$球和$[0，1]^d$的交集。我们研究了该有效威胁模型的最陡下降步长的期望稀疏性，并证明了在该集合上的精确投影在计算上是可行的，并且产生了更好的性能。此外，我们还提出了一种自适应形式的PGD，即使在很小的迭代预算下也是非常有效的。我们得到的$l_1$-APGD是一个强白盒攻击，表明以前的工作高估了它们的$l_1$-稳健性。利用$l_1$-APGD进行对抗性训练，得到一个具有SOTA$l_1$-鲁棒性的鲁棒分类器。最后，我们将$l_1$-APGD和对$l_1$的Square攻击的改进结合成$l_1$-AutoAttack，这是一个攻击集合，它可靠地评估了$l_1$-ball与$[0，1]^d$相交的威胁模型的对手健壮性。



## **18. Graph Neural Networks Inspired by Classical Iterative Algorithms**

受经典迭代算法启发的图神经网络 cs.LG

accepted as long oral for ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.06064v4)

**Authors**: Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf

**Abstracts**: Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS.

摘要: 尽管图神经网络(GNN)最近取得了成功，但常见的体系结构通常表现出显著的局限性，包括对过度平滑、长范围依赖和伪边的敏感性，例如，由于图的异嗜性或敌意攻击而可能发生的情况。为了在一个简单透明的框架内至少部分解决这些问题，我们考虑了一族新的GNN层，它们被设计成模仿和集成两种经典迭代算法的更新规则，即最近梯度下降和迭代重加权最小二乘(IRLS)。前者定义了一个可扩展的基本GNN体系结构，该体系结构不受过度平滑的影响，同时通过允许任意传播步骤来捕获远程依赖关系。相反，后者产生了一种新的注意机制，该机制显式地锚定在潜在的端到端能量函数上，有助于相对于边缘不确定性的稳定性。当组合在一起时，我们得到了一个极其简单但健壮的模型，我们可以跨不同的场景进行评估，包括标准化的基准测试、对抗性干扰图、具有异质性的图以及涉及长范围依赖的图。在此过程中，我们将其与明确为各自任务设计的Sota GNN方法进行比较，以达到好胜或更高的节点分类精度。我们的代码可在https://github.com/FFTYYY/TWIRLS.获得



## **19. Blackbox Untargeted Adversarial Testing of Automatic Speech Recognition Systems**

自动语音识别系统的黑盒非目标对抗性测试 cs.SD

10 pages, 6 figures and 7 tables

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01821v1)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) systems are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the correctness of ASRS, we propose techniques that automatically generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. Much of the existing work on adversarial ASR testing focuses on targeted attacks, i.e generating audio samples given an output text. Targeted techniques are not portable, customised to the structure of DNNs (whitebox) within a specific ASR. In contrast, our method attacks the signal processing stage of the ASR pipeline that is shared across most ASRs. Additionally, we ensure the generated adversarial audio samples have no human audible difference by manipulating the acoustic signal using a psychoacoustic model that maintains the signal below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and three input audio datasets using the metrics - WER of output text, Similarity to original audio and attack Success Rate on different ASRs. We found our testing techniques were portable across ASRs, with the adversarial audio samples producing high Success Rates, WERs and Similarities to the original audio.

摘要: 自动语音识别(ASR)系统很普遍，特别是在用于语音导航和家用电器的语音控制的应用中。ASR的计算核心是深度神经网络(DNNs)，已经证明它们容易受到对手的干扰；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的正确性，我们提出了自动生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。对抗性ASR测试的大部分现有工作都集中在有针对性的攻击上，即在给定输出文本的情况下生成音频样本。目标技术不是便携的，不能根据特定ASR内的DNN(白盒)结构进行定制。相反，我们的方法攻击大多数ASR共享的ASR流水线的信号处理阶段。此外，我们通过使用将信号保持在人类感知阈值以下的心理声学模型来处理声音信号，以确保生成的敌意音频样本没有人耳可闻的差异。我们使用三个流行的ASR和三个输入音频数据集，使用输出文本的WER、与原始音频的相似度和对不同ASR的攻击成功率来评估我们的技术的可移植性和有效性。我们发现我们的测试技术在ASR之间是可移植的，敌意音频样本产生了很高的成功率，与原始音频有很大的相似之处。



## **20. Attack-Centric Approach for Evaluating Transferability of Adversarial Samples in Machine Learning Models**

机器学习模型中以攻击为中心评估敌方样本可转移性的方法 cs.LG

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01777v1)

**Authors**: Tochukwu Idika, Ismail Akturk

**Abstracts**: Transferability of adversarial samples became a serious concern due to their impact on the reliability of machine learning system deployments, as they find their way into many critical applications. Knowing factors that influence transferability of adversarial samples can assist experts to make informed decisions on how to build robust and reliable machine learning systems. The goal of this study is to provide insights on the mechanisms behind the transferability of adversarial samples through an attack-centric approach. This attack-centric perspective interprets how adversarial samples would transfer by assessing the impact of machine learning attacks (that generated them) on a given input dataset. To achieve this goal, we generated adversarial samples using attacker models and transferred these samples to victim models. We analyzed the behavior of adversarial samples on victim models and outlined four factors that can influence the transferability of adversarial samples. Although these factors are not necessarily exhaustive, they provide useful insights to researchers and practitioners of machine learning systems.

摘要: 敌意样本的可转移性成为一个严重的问题，因为它们会影响机器学习系统部署的可靠性，因为它们会进入许多关键应用程序。了解影响对抗性样本可转移性的因素可以帮助专家就如何建立健壮可靠的机器学习系统做出明智的决定。这项研究的目的是通过以攻击为中心的方法，对敌方样本的可转移性背后的机制提供洞察力。这种以攻击为中心的观点通过评估机器学习攻击(生成它们的)对给定输入数据集的影响，解释了敌意样本将如何传输。为了实现这一目标，我们使用攻击者模型生成对抗性样本，并将这些样本传输到受害者模型。我们分析了对抗性样本在受害者模型上的行为，并概述了影响对抗性样本可转移性的四个因素。虽然这些因素不一定是详尽的，但它们为机器学习系统的研究人员和实践者提供了有用的见解。



## **21. Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach**

针对恶意软件检测器的单发黑盒对抗性攻击：一种因果语言模型方法 cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01724v1)

**Authors**: James Lee Hu, Mohammadreza Ebrahimi, Hsinchun Chen

**Abstracts**: Deep Learning (DL)-based malware detectors are increasingly adopted for early detection of malicious behavior in cybersecurity. However, their sensitivity to adversarial malware variants has raised immense security concerns. Generating such adversarial variants by the defender is crucial to improving the resistance of DL-based malware detectors against them. This necessity has given rise to an emerging stream of machine learning research, Adversarial Malware example Generation (AMG), which aims to generate evasive adversarial malware variants that preserve the malicious functionality of a given malware. Within AMG research, black-box method has gained more attention than white-box methods. However, most black-box AMG methods require numerous interactions with the malware detectors to generate adversarial malware examples. Given that most malware detectors enforce a query limit, this could result in generating non-realistic adversarial examples that are likely to be detected in practice due to lack of stealth. In this study, we show that a novel DL-based causal language model enables single-shot evasion (i.e., with only one query to malware detector) by treating the content of the malware executable as a byte sequence and training a Generative Pre-Trained Transformer (GPT). Our proposed method, MalGPT, significantly outperformed the leading benchmark methods on a real-world malware dataset obtained from VirusTotal, achieving over 24.51\% evasion rate. MalGPT enables cybersecurity researchers to develop advanced defense capabilities by emulating large-scale realistic AMG.

摘要: 基于深度学习(DL)的恶意软件检测器越来越多地被用于网络安全中的恶意行为的早期检测。然而，他们对敌意恶意软件变体的敏感性引发了巨大的安全担忧。防御者生成这种敌意变体对于提高基于DL的恶意软件检测器对它们的抵抗力至关重要。这种必要性已经引起了一种新兴的机器学习研究流，即对抗性恶意软件示例生成(AMG)，其目的是生成保留给定恶意软件的恶意功能的闪避性对抗性恶意软件变体。在AMG研究中，黑盒方法比白盒方法更受关注。然而，大多数黑盒AMG方法需要与恶意软件检测器进行多次交互才能生成敌意恶意软件示例。鉴于大多数恶意软件检测器强制执行查询限制，这可能导致生成由于缺乏隐蔽性而很可能在实践中被检测到的不切实际的对抗性示例。在这项研究中，我们提出了一种新的基于DL的因果语言模型，通过将恶意软件可执行文件的内容视为一个字节序列，并训练一个生成式预训练转换器(GPT)，从而实现单发规避(即只需对恶意软件检测器进行一次查询)。我们提出的MalGPT方法在从VirusTotal获得的真实恶意软件数据集上的性能明显优于领先的基准测试方法，达到了24.51%以上的逃避率。MalGPT使网络安全研究人员能够通过模拟大规模现实AMG来开发高级防御能力。



## **22. Adversarial Attacks against a Satellite-borne Multispectral Cloud Detector**

针对星载多光谱云探测器的敌意攻击 cs.CV

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01723v1)

**Authors**: Andrew Du, Yee Wei Law, Michele Sasdelli, Bo Chen, Ken Clarke, Michael Brown, Tat-Jun Chin

**Abstracts**: Data collected by Earth-observing (EO) satellites are often afflicted by cloud cover. Detecting the presence of clouds -- which is increasingly done using deep learning -- is crucial preprocessing in EO applications. In fact, advanced EO satellites perform deep learning-based cloud detection on board the satellites and downlink only clear-sky data to save precious bandwidth. In this paper, we highlight the vulnerability of deep learning-based cloud detection towards adversarial attacks. By optimising an adversarial pattern and superimposing it into a cloudless scene, we bias the neural network into detecting clouds in the scene. Since the input spectra of cloud detectors include the non-visible bands, we generated our attacks in the multispectral domain. This opens up the potential of multi-objective attacks, specifically, adversarial biasing in the cloud-sensitive bands and visual camouflage in the visible bands. We also investigated mitigation strategies against the adversarial attacks. We hope our work further builds awareness of the potential of adversarial attacks in the EO community.

摘要: 地球观测卫星(EO)收集的数据经常受到云层的影响。检测云的存在--越来越多地使用深度学习来完成--在EO应用程序中是至关重要的预处理。事实上，先进的地球观测卫星在卫星上进行基于深度学习的云层探测，只下行晴空数据，以节省宝贵的带宽。在本文中，我们强调了基于深度学习的云检测对敌意攻击的脆弱性。通过优化对抗性模式并将其叠加到无云场景中，我们将神经网络偏向于检测场景中的云。由于云探测器的输入光谱中包含不可见波段，因此我们在多光谱域中产生了我们的攻击。这打开了多目标攻击的可能性，具体地说，云敏感波段的对抗性偏向和可见波段的视觉伪装。我们还研究了针对对抗性攻击的缓解策略。我们希望我们的工作进一步提高人们对EO社区中潜在的对抗性攻击的认识。



## **23. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

RobustBench/AutoAttack是衡量对手健壮性的合适基准吗？ cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01601v1)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.

摘要: 最近，RobustBench(Croce et al.2020)已经成为图像分类网络对抗性健壮性的广泛认可的基准。在其最常见的子任务中，RobustBench在AutoAttack(Croce和Hein 2020b)下评估和排名了CIFAR10上训练的神经网络的对抗鲁棒性，其中l-inf扰动限制在EPS=8/255。目前性能最好的模型的领先分数约为基准的60%，可以公平地将此基准描述为相当具有挑战性。尽管它在最近的文献中被广泛接受，但我们的目标是促进关于RobustBench作为健壮性的关键指标的适宜性的讨论，这可以推广到实际应用中。我们对此的论证是双重的，并得到了本文提出的过多实验的支持：我们认为：i)AutoAttack与l-inf，EPS=8/255的数据交互是不切实际的，导致即使使用简单的检测算法和人类观察者，也能获得接近完美的敌意样本检测率。我们还表明，在获得类似成功率的情况下，其他攻击方法要难得多。ii)在低分辨率数据集(如CIFAR10)上的结果不能很好地推广到更高分辨率的图像，因为基于梯度的攻击似乎随着分辨率的增加而变得更容易检测到。



## **24. Is Approximation Universally Defensive Against Adversarial Attacks in Deep Neural Networks?**

深度神经网络中的近似是否普遍防御敌意攻击？ cs.LG

Accepted for publication in DATE 2022

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01555v1)

**Authors**: Ayesha Siddique, Khaza Anuarul Hoque

**Abstracts**: Approximate computing is known for its effectiveness in improvising the energy efficiency of deep neural network (DNN) accelerators at the cost of slight accuracy loss. Very recently, the inexact nature of approximate components, such as approximate multipliers have also been reported successful in defending adversarial attacks on DNNs models. Since the approximation errors traverse through the DNN layers as masked or unmasked, this raises a key research question-can approximate computing always offer a defense against adversarial attacks in DNNs, i.e., are they universally defensive? Towards this, we present an extensive adversarial robustness analysis of different approximate DNN accelerators (AxDNNs) using the state-of-the-art approximate multipliers. In particular, we evaluate the impact of ten adversarial attacks on different AxDNNs using the MNIST and CIFAR-10 datasets. Our results demonstrate that adversarial attacks on AxDNNs can cause 53% accuracy loss whereas the same attack may lead to almost no accuracy loss (as low as 0.06%) in the accurate DNN. Thus, approximate computing cannot be referred to as a universal defense strategy against adversarial attacks.

摘要: 近似计算在以轻微精度损失为代价提高深度神经网络(DNN)加速器的能效方面是众所周知的。最近，近似分量的不精确性质，如近似乘子，也被报道成功地防御了对DNNs模型的敌意攻击。由于近似误差以屏蔽或非屏蔽的形式遍历DNN各层，这就提出了一个关键的研究问题--近似计算是否总能为DNN中的敌意攻击提供防御，即它们是否具有普遍的防御性？为此，我们使用最先进的近似乘子对不同近似DNN加速器(AxDNNs)进行了广泛的对抗健壮性分析。特别地，我们使用MNIST和CIFAR-10数据集评估了10种对抗性攻击对不同AxDNNs的影响。实验结果表明，对AxDNNs的敌意攻击可以导致53%的准确率损失，而在精确DNN中，同样的攻击可能几乎不会造成准确率损失(低至0.06%)。因此，近似计算不能被称为对抗对手攻击的通用防御策略。



## **25. FedRAD: Federated Robust Adaptive Distillation**

FedRAD：联合鲁棒自适应精馏 cs.LG

Accepted for 1st NeurIPS Workshop on New Frontiers in Federated  Learning (NFFL 2021), Virtual Meeting

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01405v1)

**Authors**: Stefán Páll Sturluson, Samuel Trew, Luis Muñoz-González, Matei Grama, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary

**Abstracts**: The robustness of federated learning (FL) is vital for the distributed training of an accurate global model that is shared among large number of clients. The collaborative learning framework by typically aggregating model updates is vulnerable to model poisoning attacks from adversarial clients. Since the shared information between the global server and participants are only limited to model parameters, it is challenging to detect bad model updates. Moreover, real-world datasets are usually heterogeneous and not independent and identically distributed (Non-IID) among participants, which makes the design of such robust FL pipeline more difficult. In this work, we propose a novel robust aggregation method, Federated Robust Adaptive Distillation (FedRAD), to detect adversaries and robustly aggregate local models based on properties of the median statistic, and then performing an adapted version of ensemble Knowledge Distillation. We run extensive experiments to evaluate the proposed method against recently published works. The results show that FedRAD outperforms all other aggregators in the presence of adversaries, as well as in heterogeneous data distributions.

摘要: 联邦学习(FL)的健壮性对于分布式训练大量客户端共享的精确全局模型至关重要。典型地聚合模型更新的协作学习框架容易受到来自敌对客户端的模型中毒攻击。由于全局服务器和参与者之间的共享信息仅限于模型参数，因此检测错误的模型更新是具有挑战性的。此外，现实世界的数据集通常是异构的，参与者之间并不是独立且相同分布的(非IID)，这使得设计这样健壮的FL流水线变得更加困难。在这项工作中，我们提出了一种新的健壮聚合方法，联邦健壮自适应蒸馏(FedRAD)，根据中值统计特性检测对手并健壮聚合局部模型，然后执行改进版本的集成知识蒸馏。我们进行了大量的实验，以评估所提出的方法与最近发表的作品。结果表明，FedRAD在存在对手的情况下，以及在异构数据分布的情况下，性能优于所有其他聚合器。



## **26. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

受限特征空间中的对抗性攻防统一框架 cs.AI

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01156v1)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work on constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework supports the use cases reported in the literature and can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective on two datasets from different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

摘要: 生成可行的对抗性示例对于正确评估工作在受限特征空间上的模型是必要的。然而，对专为计算机视觉设计的攻击实施约束仍然是一项具有挑战性的任务。我们提出了一个统一的框架来生成满足给定领域约束的可行对抗性示例。我们的框架支持文献中报告的用例，并且可以处理线性和非线性约束。我们将我们的框架实例化为两种算法：一种是在损失函数中引入约束以最大化的基于梯度的攻击算法，另一种是以误分类、扰动最小化和约束满足为目标的多目标搜索算法。我们在两个来自不同领域的数据集上证明了我们的方法是有效的，成功率高达100%，其中最先进的攻击没有产生一个可行的例子。除了对抗性再训练之外，我们还建议引入工程非凸约束来提高模型对抗性的稳健性。我们证明了这种新的防御和对抗性的再训练一样有效。我们的框架构成了受限对抗攻击研究的起点，并为未来的研究提供了相关的基线和数据集。



## **27. Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems**

基于深度强化学习的动态推荐系统的对抗鲁棒性 cs.LG

arXiv admin note: text overlap with arXiv:2006.07934

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.00973v1)

**Authors**: Siyu Wang, Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Quan Z. Sheng

**Abstracts**: Adversarial attacks, e.g., adversarial perturbations of the input and adversarial samples, pose significant challenges to machine learning and deep learning techniques, including interactive recommendation systems. The latent embedding space of those techniques makes adversarial attacks difficult to detect at an early stage. Recent advance in causality shows that counterfactual can also be considered one of ways to generate the adversarial samples drawn from different distribution as the training samples. We propose to explore adversarial examples and attack agnostic detection on reinforcement learning-based interactive recommendation systems. We first craft different types of adversarial examples by adding perturbations to the input and intervening on the casual factors. Then, we augment recommendation systems by detecting potential attacks with a deep learning-based classifier based on the crafted data. Finally, we study the attack strength and frequency of adversarial examples and evaluate our model on standard datasets with multiple crafting methods. Our extensive experiments show that most adversarial attacks are effective, and both attack strength and attack frequency impact the attack performance. The strategically-timed attack achieves comparative attack performance with only 1/3 to 1/2 attack frequency. Besides, our black-box detector trained with one crafting method has the generalization ability over several other crafting methods.

摘要: 对抗性攻击，例如输入和对抗性样本的对抗性扰动，给机器学习和深度学习技术(包括交互式推荐系统)带来了重大挑战。这些技术的潜在嵌入空间使得敌意攻击很难在早期阶段被发现。最近因果关系的进展表明，反事实也可以被认为是生成来自不同分布的对抗性样本作为训练样本的方法之一。我们提出在基于强化学习的交互式推荐系统上探索敌意示例和攻击不可知检测。我们首先制作不同类型的对抗性例子，通过在输入中添加扰动和对偶然因素进行干预来创建不同类型的对抗性例子。然后，我们利用基于深度学习的分类器基于精心制作的数据来检测潜在的攻击，从而增强推荐系统。最后，我们研究了敌意实例的攻击强度和攻击频率，并在标准数据集上采用多种制作方法对我们的模型进行了评估。我们的大量实验表明，大多数对抗性攻击都是有效的，攻击强度和攻击频率都会影响攻击性能。战略计时攻击仅用1/3到1/2的攻击频率就达到了比较的攻击性能。此外，用一种工艺方法训练的黑盒检测器比其他几种工艺方法具有更好的泛化能力。



## **28. Learning Task-aware Robust Deep Learning Systems**

学习任务感知的鲁棒深度学习系统 cs.LG

9 Pages

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2010.05125v2)

**Authors**: Keji Han, Yun Li, Xianzhong Long, Yao Ge

**Abstracts**: Many works demonstrate that deep learning system is vulnerable to adversarial attack. A deep learning system consists of two parts: the deep learning task and the deep model. Nowadays, most existing works investigate the impact of the deep model on robustness of deep learning systems, ignoring the impact of the learning task. In this paper, we adopt the binary and interval label encoding strategy to redefine the classification task and design corresponding loss to improve robustness of the deep learning system. Our method can be viewed as improving the robustness of deep learning systems from both the learning task and deep model. Experimental results demonstrate that our learning task-aware method is much more robust than traditional classification while retaining the accuracy.

摘要: 大量研究表明，深度学习系统容易受到敌意攻击。深度学习系统由两部分组成：深度学习任务和深度模型。目前，已有的工作大多研究深度模型对深度学习系统鲁棒性的影响，忽略了学习任务的影响。本文采用二进制和区间标签编码策略重新定义分类任务，并设计相应的损失来提高深度学习系统的鲁棒性。我们的方法可以看作是从学习任务和深度模型两个方面提高深度学习系统的鲁棒性。实验结果表明，我们的学习任务感知方法在保持分类准确率的同时，比传统的分类方法具有更强的鲁棒性。



## **29. They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors**

他们看到我在滚动：CMOS图像传感器中滚动快门的固有弱点 cs.CV

15 pages, 15 figures

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2101.10011v2)

**Authors**: Sebastian Köhler, Giulio Lovisotto, Simon Birnbach, Richard Baker, Ivan Martinovic

**Abstracts**: In this paper, we describe how the electronic rolling shutter in CMOS image sensors can be exploited using a bright, modulated light source (e.g., an inexpensive, off-the-shelf laser), to inject fine-grained image disruptions. We demonstrate the attack on seven different CMOS cameras, ranging from cheap IoT to semi-professional surveillance cameras, to highlight the wide applicability of the rolling shutter attack. We model the fundamental factors affecting a rolling shutter attack in an uncontrolled setting. We then perform an exhaustive evaluation of the attack's effect on the task of object detection, investigating the effect of attack parameters. We validate our model against empirical data collected on two separate cameras, showing that by simply using information from the camera's datasheet the adversary can accurately predict the injected distortion size and optimize their attack accordingly. We find that an adversary can hide up to 75% of objects perceived by state-of-the-art detectors by selecting appropriate attack parameters. We also investigate the stealthiness of the attack in comparison to a na\"{i}ve camera blinding attack, showing that common image distortion metrics can not detect the attack presence. Therefore, we present a new, accurate and lightweight enhancement to the backbone network of an object detector to recognize rolling shutter attacks. Overall, our results indicate that rolling shutter attacks can substantially reduce the performance and reliability of vision-based intelligent systems.

摘要: 在这篇文章中，我们描述了如何利用CMOS图像传感器中的电子滚动快门，使用明亮的调制光源(例如，廉价的现成激光器)来注入细粒度的图像干扰。我们演示了对七种不同CMOS摄像头的攻击，从廉价的物联网到半专业的监控摄像头，以突出滚动快门攻击的广泛适用性。我们模拟了在不受控制的环境下影响滚动快门攻击的基本因素。然后，我们对攻击对目标检测任务的影响进行了详尽的评估，考察了攻击参数的影响。我们通过在两个不同的摄像机上收集的经验数据验证了我们的模型，结果表明，通过简单地使用摄像机数据表中的信息，对手可以准确地预测注入失真的大小，并相应地优化他们的攻击。我们发现，通过选择合适的攻击参数，敌手可以隐藏最新检测器感知到的高达75%的对象。与单纯的相机盲攻击相比，我们还研究了该攻击的隐蔽性，发现普通的图像失真度量不能检测到攻击的存在，因此，我们提出了一种新的、准确的、轻量级的对象检测器主干网络增强方法来识别滚动快门攻击，结果表明，滚动快门攻击会大大降低基于视觉的智能系统的性能和可靠性。



## **30. Certified Adversarial Defenses Meet Out-of-Distribution Corruptions: Benchmarking Robustness and Simple Baselines**

认证的对抗性防御遇到分布外的腐败：基准、健壮性和简单的基线 cs.LG

21 pages, 15 figures, and 9 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00659v1)

**Authors**: Jiachen Sun, Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Dan Hendrycks, Jihun Hamm, Z. Morley Mao

**Abstracts**: Certified robustness guarantee gauges a model's robustness to test-time attacks and can assess the model's readiness for deployment in the real world. In this work, we critically examine how the adversarial robustness guarantees from randomized smoothing-based certification methods change when state-of-the-art certifiably robust models encounter out-of-distribution (OOD) data. Our analysis demonstrates a previously unknown vulnerability of these models to low-frequency OOD data such as weather-related corruptions, rendering these models unfit for deployment in the wild. To alleviate this issue, we propose a novel data augmentation scheme, FourierMix, that produces augmentations to improve the spectral coverage of the training data. Furthermore, we propose a new regularizer that encourages consistent predictions on noise perturbations of the augmented data to improve the quality of the smoothed models. We find that FourierMix augmentations help eliminate the spectral bias of certifiably robust models enabling them to achieve significantly better robustness guarantees on a range of OOD benchmarks. Our evaluation also uncovers the inability of current OOD benchmarks at highlighting the spectral biases of the models. To this end, we propose a comprehensive benchmarking suite that contains corruptions from different regions in the spectral domain. Evaluation of models trained with popular augmentation methods on the proposed suite highlights their spectral biases and establishes the superiority of FourierMix trained models at achieving better-certified robustness guarantees under OOD shifts over the entire frequency spectrum.

摘要: 认证的健壮性保证衡量模型对测试时间攻击的健壮性，并可以评估模型在现实世界中部署的准备情况。在这项工作中，我们批判性地研究了当最新的可证明鲁棒性模型遇到分布外(OOD)数据时，基于随机平滑的认证方法所保证的敌意鲁棒性是如何改变的。我们的分析表明，这些模型对低频OOD数据(如与天气相关的损坏)存在以前未知的脆弱性，使得这些模型不适合在野外部署。为了缓解这一问题，我们提出了一种新的数据增强方案FURIERMIX，该方案通过产生增强来提高训练数据的频谱覆盖率。此外，我们还提出了一种新的正则化方法，它鼓励对增强数据的噪声扰动进行一致的预测，以提高平滑模型的质量。我们发现，傅立叶混合增强有助于消除可证明的健壮性模型的频谱偏差，使它们能够在一系列面向对象设计基准上获得显着更好的健壮性保证。我们的评估还揭示了当前OOD基准在突出模型的光谱偏差方面的不足。为此，我们提出了一个全面的基准测试套件，该套件包含来自谱域中不同区域的腐败。在建议的套件上用流行的增强方法训练的模型的评估突出了它们的频谱偏差，并确立了傅里叶混合训练的模型在整个频谱上的OOD漂移下实现更好的认证鲁棒性保证的优势。



## **31. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 16 pages, 11 figures, 13 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2110.06537v3)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and the growth of margin. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to learning. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verify the theoretical results or through the significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that because our idea can solve these three issues, we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不良的示例，而忽略远离决策边界的分类良好的示例。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种普遍的做法阻碍了表征学习、能量优化和边际增长。为了弥补这一不足，我们建议向分类良好的例子发放额外奖金，以恢复他们对学习的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在不同任务(包括图像分类、图形分类和机器翻译)上的显著性能改进来实证支持这一主张。此外，本文还表明，由于我们的思想可以解决这三个问题，所以我们可以处理复杂的场景，如不平衡分类、面向对象的检测以及在对抗性攻击下的应用。代码可在以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **32. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

理解深度强化学习中对观测的敌意攻击 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2106.15860v2)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.

摘要: 深度强化学习模型很容易受到敌意攻击，这种攻击会通过操纵受害者的观察结果来降低受害者的累积预期回报。尽管以前的基于优化的方法在监督学习中产生对抗性噪声是有效的，但是这些方法可能不能获得最低的累积奖励，因为它们通常不探索环境动态。本文通过在函数空间中重新表述强化学习的对抗性攻击问题，为更好地理解现有方法提供了一个框架。我们的重构在目标攻击的函数空间中生成一个最优对手，通过一个通用的两阶段框架击退它们。在第一阶段，我们通过黑客攻击环境来训练欺骗性策略，并发现一组通往最低回报或最坏情况表现的轨迹。接下来，对手通过扰乱观察来误导受害者模仿欺骗性的政策。与现有的方法相比，我们从理论上证明了在适当的噪声水平下，我们的对手更强。大量的实验证明了我们的方法在效率和有效性方面的优越性，在Atari和MuJoCo环境中都实现了最先进的性能。



## **33. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

$\ell_\infty$-健壮性和超越：释放高效的对抗性训练 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00378v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, which has hampered its effectiveness. Recently, Fast Adversarial Training was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection we show how selecting a small subset of training data provides a more principled approach towards reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. Our experimental results indicate that our approach speeds up adversarial training by 2-3 times, while experiencing a small reduction in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练鲁棒模型对抗此类攻击的最有效方法之一。但是，由于它在每次迭代时都需要为整个训练数据构造对抗性样本，因此比神经网络的香草训练慢得多，这就阻碍了它的有效性。最近，人们提出了一种快速对抗性训练方法，可以有效地获得稳健的模型。然而，其成功背后的原因还没有被完全理解，更重要的是，由于它在训练期间使用FGSM，所以它只能训练健壮的模型来应对$\ell_\$有界攻击。在本文中，通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种更有原则的方法来降低鲁棒训练的时间复杂度。与现有方法不同，我们的方法可以适应广泛的训练目标，包括行业、$\ell_p$-PGD和知觉对抗性训练。我们的实验结果表明，我们的方法将对抗性训练的速度提高了2-3倍，同时经历了干净和健壮的准确率的小幅下降。



## **34. Designing a Location Trace Anonymization Contest**

设计一个位置跟踪匿名化竞赛 cs.CR

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2107.10407v2)

**Authors**: Takao Murakami, Hiromi Arai, Koki Hamada, Takuma Hatano, Makoto Iguchi, Hiroaki Kikuchi, Atsushi Kuromasa, Hiroshi Nakagawa, Yuichi Nakamura, Kenshiro Nishiyama, Ryo Nojima, Hidenobu Oguri, Chiemi Watanabe, Akira Yamada, Takayasu Yamaguchi, Yuji Yamaoka

**Abstracts**: For a better understanding of anonymization methods for location traces, we have designed and held a location trace anonymization contest. Our contest deals with a long trace (400 events per user) and fine-grained locations (1024 regions). In our contest, each team anonymizes her original traces, and then the other teams perform privacy attacks against the anonymized traces in a partial-knowledge attacker model where the adversary does not know the original traces. To realize such a contest, we propose a location synthesizer that has diversity and utility; the synthesizer generates different synthetic traces for each team while preserving various statistical features of real traces. We also show that re-identification alone is insufficient as a privacy risk and that trace inference should be added as an additional risk. Specifically, we show an example of anonymization that is perfectly secure against re-identification and is not secure against trace inference. Based on this, our contest evaluates both the re-identification risk and trace inference risk and analyzes their relationship. Through our contest, we show several findings in a situation where both defense and attack compete together. In particular, we show that an anonymization method secure against trace inference is also secure against re-identification under the presence of appropriate pseudonymization.

摘要: 为了更好地了解位置踪迹的匿名化方法，我们设计并举办了位置踪迹匿名化大赛。我们的竞赛涉及长跟踪(每个用户400个事件)和细粒度位置(1024个区域)。在我们的比赛中，每个团队匿名她的原始痕迹，然后其他团队在部分知识攻击者模型中对匿名的痕迹进行隐私攻击，其中对手不知道原始痕迹。为了实现这样的竞赛，我们提出了一种具有多样性和实用性的位置合成器，该合成器在保留真实轨迹的各种统计特征的同时，为每个团队生成不同的合成轨迹。我们还表明，仅重新识别作为隐私风险是不够的，应该添加跟踪推断作为附加风险。具体地说，我们展示了一个匿名化的例子，它对于重新识别是完全安全的，而对于跟踪推理是不安全的。在此基础上，对再识别风险和痕迹推理风险进行了评估，并分析了它们之间的关系。通过我们的比赛，我们展示了在防守和进攻同时竞争的情况下的几个发现。特别地，我们证明了在存在适当的假名的情况下，一个安全的抗踪迹推理的匿名化方法也是安全的。



## **35. Push Stricter to Decide Better: A Class-Conditional Feature Adaptive Framework for Improving Adversarial Robustness**

越严越优：一种提高对手健壮性的类条件特征自适应框架 cs.CV

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00323v1)

**Authors**: Jia-Li Yin, Lehui Xie, Wanqing Zhu, Ximeng Liu, Bo-Hao Chen

**Abstracts**: In response to the threat of adversarial examples, adversarial training provides an attractive option for enhancing the model robustness by training models on online-augmented adversarial examples. However, most of the existing adversarial training methods focus on improving the robust accuracy by strengthening the adversarial examples but neglecting the increasing shift between natural data and adversarial examples, leading to a dramatic decrease in natural accuracy. To maintain the trade-off between natural and robust accuracy, we alleviate the shift from the perspective of feature adaption and propose a Feature Adaptive Adversarial Training (FAAT) optimizing the class-conditional feature adaption across natural data and adversarial examples. Specifically, we propose to incorporate a class-conditional discriminator to encourage the features become (1) class-discriminative and (2) invariant to the change of adversarial attacks. The novel FAAT framework enables the trade-off between natural and robust accuracy by generating features with similar distribution across natural and adversarial data, and achieve higher overall robustness benefited from the class-discriminative feature characteristics. Experiments on various datasets demonstrate that FAAT produces more discriminative features and performs favorably against state-of-the-art methods. Codes are available at https://github.com/VisionFlow/FAAT.

摘要: 为了应对对抗性示例的威胁，对抗性训练通过训练在线扩充的对抗性示例模型，为增强模型的稳健性提供了一种有吸引力的选择。然而，现有的对抗性训练方法大多侧重于通过加强对抗性实例来提高鲁棒准确率，而忽略了自然数据与对抗性实例之间不断增加的偏移，导致自然精确度急剧下降。为了保持自然和鲁棒精度之间的折衷，我们从特征自适应的角度缓解了这一转变，并提出了一种特征自适应对抗训练(FAAT)，优化了跨自然数据和对抗性示例的类条件特征自适应。具体地说，我们建议加入类条件鉴别器，以鼓励特征成为(1)类可分辨的和(2)对敌方攻击变化不变的特征。新的FAAT框架通过在自然数据和对抗性数据上生成分布相似的特征，能够在自然和鲁棒精度之间进行权衡，并得益于类区分特征特性而获得更高的整体鲁棒性。在不同的数据集上的实验表明，FAAT产生了更具区分性的特征，并且与最先进的方法相比表现出了良好的性能。有关代码，请访问https://github.com/VisionFlow/FAAT.。



## **36. Adversarial Attacks Against Deep Generative Models on Data: A Survey**

针对数据深层生成模型的对抗性攻击：综述 cs.CR

To be published in IEEE Transactions on Knowledge and Data  Engineering

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00247v1)

**Authors**: Hui Sun, Tianqing Zhu, Zhiqiu Zhang, Dawei Jin. Ping Xiong, Wanlei Zhou

**Abstracts**: Deep generative models have gained much attention given their ability to generate data for applications as varied as healthcare to financial technology to surveillance, and many more - the most popular models being generative adversarial networks and variational auto-encoders. Yet, as with all machine learning models, ever is the concern over security breaches and privacy leaks and deep generative models are no exception. These models have advanced so rapidly in recent years that work on their security is still in its infancy. In an attempt to audit the current and future threats against these models, and to provide a roadmap for defense preparations in the short term, we prepared this comprehensive and specialized survey on the security and privacy preservation of GANs and VAEs. Our focus is on the inner connection between attacks and model architectures and, more specifically, on five components of deep generative models: the training data, the latent code, the generators/decoders of GANs/ VAEs, the discriminators/encoders of GANs/ VAEs, and the generated data. For each model, component and attack, we review the current research progress and identify the key challenges. The paper concludes with a discussion of possible future attacks and research directions in the field.

摘要: 深度生成模型因其能够为从医疗保健到金融技术再到监控等各种应用程序生成数据而备受关注-最受欢迎的模型是生成性对抗性网络和变化式自动编码器。然而，与所有机器学习模型一样，人们一直担心安全漏洞和隐私泄露，深度生成模型也不例外。近年来，这些模式发展如此之快，其安全方面的工作仍处于初级阶段。为了审计这些模式当前和未来的威胁，并为短期内的防御准备提供路线图，我们准备了这项关于GAN和VAE的安全和隐私保护的全面而专业的调查。我们的重点是攻击和模型体系结构之间的内在联系，更具体地说，是深入生成模型的五个组成部分：训练数据、潜在代码、GANS/VAE的生成器/解码器、GANS/VAE的鉴别器/编码器和生成的数据。对于每个模型、组件和攻击，我们回顾了当前的研究进展，并确定了关键挑战。最后，对未来可能的攻击和该领域的研究方向进行了讨论。



## **37. Model Extraction Attacks on Graph Neural Networks: Taxonomy and Realization**

对图神经网络的模型提取攻击：分类与实现 cs.LG

This paper has been published in the 17th ACM ASIA Conference on  Computer and Communications Security (ACM ASIACCS 2022)

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2010.12751v2)

**Authors**: Bang Wu, Xiangwen Yang, Shirui Pan, Xingliang Yuan

**Abstracts**: Machine learning models are shown to face a severe threat from Model Extraction Attacks, where a well-trained private model owned by a service provider can be stolen by an attacker pretending as a client. Unfortunately, prior works focus on the models trained over the Euclidean space, e.g., images and texts, while how to extract a GNN model that contains a graph structure and node features is yet to be explored. In this paper, for the first time, we comprehensively investigate and develop model extraction attacks against GNN models. We first systematically formalise the threat modelling in the context of GNN model extraction and classify the adversarial threats into seven categories by considering different background knowledge of the attacker, e.g., attributes and/or neighbour connections of the nodes obtained by the attacker. Then we present detailed methods which utilise the accessible knowledge in each threat to implement the attacks. By evaluating over three real-world datasets, our attacks are shown to extract duplicated models effectively, i.e., 84% - 89% of the inputs in the target domain have the same output predictions as the victim model.

摘要: 机器学习模型面临着模型提取攻击的严重威胁，在这种攻击中，服务提供商拥有的训练有素的私有模型可能会被冒充客户端的攻击者窃取。遗憾的是，以前的工作主要集中在欧氏空间上训练的模型，例如图像和文本，而如何提取包含图结构和节点特征的GNN模型还有待探索。本文首次全面研究并开发了针对GNN模型的模型提取攻击。我们首先在GNN模型提取的背景下系统地形式化威胁建模，并通过考虑攻击者的不同背景知识(例如攻击者获取的节点的属性和/或邻居连接)将敌意威胁分类为七类。然后，我们给出了利用每个威胁中可访问的知识来实施攻击的详细方法。通过对三个真实数据集的评估，我们的攻击可以有效地提取重复模型，即目标领域中84%-89%的输入与受害者模型具有相同的输出预测。



## **38. Robust Multiple-Path Orienteering Problem: Securing Against Adversarial Attacks**

鲁棒多路径定向问题：抵抗敌方攻击的安全 cs.RO

submitted to TRO

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2003.13896v3)

**Authors**: Guangyao Shi, Lifeng Zhou, Pratap Tokekar

**Abstracts**: The multiple-path orienteering problem asks for paths for a team of robots that maximize the total reward collected while satisfying budget constraints on the path length. This problem models many multi-robot routing tasks such as exploring unknown environments and information gathering for environmental monitoring. In this paper, we focus on how to make the robot team robust to failures when operating in adversarial environments. We introduce the Robust Multiple-path Orienteering Problem (RMOP) where we seek worst-case guarantees against an adversary that is capable of attacking at most $\alpha$ robots. We consider two versions of this problem: RMOP offline and RMOP online. In the offline version, there is no communication or replanning when robots execute their plans and our main contribution is a general approximation scheme with a bounded approximation guarantee that depends on $\alpha$ and the approximation factor for single robot orienteering. In particular, we show that the algorithm yields a (i) constant-factor approximation when the cost function is modular; (ii) $\log$ factor approximation when the cost function is submodular; and (iii) constant-factor approximation when the cost function is submodular but the robots are allowed to exceed their path budgets by a bounded amount. In the online version, RMOP is modeled as a two-player sequential game and solved adaptively in a receding horizon fashion based on Monte Carlo Tree Search (MCTS). In addition to theoretical analysis, we perform simulation studies for ocean monitoring and tunnel information-gathering applications to demonstrate the efficacy of our approach.

摘要: 多路径定向问题要求一组机器人在满足路径长度的预算约束的同时最大化所收集的总奖励的路径。该问题模拟了许多多机器人路由任务，如探索未知环境和为环境监测收集信息。在本文中，我们重点研究如何使机器人团队在对抗性环境中工作时对故障具有健壮性。我们引入了鲁棒多路径定向问题(RMOP)，其中我们寻求对最多能够攻击$\α$机器人的对手的最坏情况保证。我们考虑此问题的两个版本：RMOP离线和RMOP在线。在离线版本中，机器人在执行其计划时不会进行通信或重新规划，我们的主要贡献是提供了一种一般的近似方案，它具有有界的逼近保证，它依赖于$\α和单个机器人定向的逼近因子。特别地，我们证明了当代价函数是模数时，该算法产生了常数因子近似；当代价函数是子模时，算法产生了$\log$因子逼近；以及(Iii)当代价函数是子模的，但允许机器人超出路径预算有限量时，算法得到了常数因子逼近。在在线版本中，RMOP被建模为一个两人序列博弈，并基于蒙特卡罗树搜索(MCTS)以滚动时域的方式自适应求解。除了理论分析外，我们还对海洋监测和隧道信息收集应用进行了仿真研究，以证明该方法的有效性。



## **39. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.05978v2)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Ghulam Rasool, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **40. Defending Against Adversarial Denial-of-Service Data Poisoning Attacks**

防御敌意的拒绝服务数据中毒攻击 cs.CR

Published at ACSAC DYNAMICS 2020

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2104.06744v3)

**Authors**: Nicolas M. Müller, Simon Roschmann, Konstantin Böttinger

**Abstracts**: Data poisoning is one of the most relevant security threats against machine learning and data-driven technologies. Since many applications rely on untrusted training data, an attacker can easily craft malicious samples and inject them into the training dataset to degrade the performance of machine learning models. As recent work has shown, such Denial-of-Service (DoS) data poisoning attacks are highly effective. To mitigate this threat, we propose a new approach of detecting DoS poisoned instances. In comparison to related work, we deviate from clustering and anomaly detection based approaches, which often suffer from the curse of dimensionality and arbitrary anomaly threshold selection. Rather, our defence is based on extracting information from the training data in such a generalized manner that we can identify poisoned samples based on the information present in the unpoisoned portion of the data. We evaluate our defence against two DoS poisoning attacks and seven datasets, and find that it reliably identifies poisoned instances. In comparison to related work, our defence improves false positive / false negative rates by at least 50%, often more.

摘要: 数据中毒是机器学习和数据驱动技术面临的最相关的安全威胁之一。由于许多应用程序依赖于不可信的训练数据，攻击者可以很容易地手工制作恶意样本并将其注入到训练数据集中，从而降低机器学习模型的性能。最近的研究表明，这种拒绝服务(DoS)数据中毒攻击非常有效。为了缓解这种威胁，我们提出了一种检测DoS中毒实例的新方法。与相关工作相比，我们偏离了基于聚类和异常检测的方法，这些方法经常受到维数灾难和任意选择异常阈值的影响。相反，我们的辩护是基于以一种普遍的方式从训练数据中提取信息，以便我们可以基于数据的未中毒部分中存在的信息来识别中毒样本。我们评估了我们对两个DoS中毒攻击和七个数据集的防御，发现它可以可靠地识别中毒实例。与相关工作相比，我们的辩护将假阳性/假阴性率提高了至少50%，往往更高。



## **41. FROB: Few-shot ROBust Model for Classification and Out-of-Distribution Detection**

FROB：一种用于分类和越界检测的少射鲁棒模型 cs.LG

Paper, 22 pages, Figures, Tables

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15487v1)

**Authors**: Nikolaos Dionelis

**Abstracts**: Nowadays, classification and Out-of-Distribution (OoD) detection in the few-shot setting remain challenging aims due to rarity and the limited samples in the few-shot setting, and because of adversarial attacks. Accomplishing these aims is important for critical systems in safety, security, and defence. In parallel, OoD detection is challenging since deep neural network classifiers set high confidence to OoD samples away from the training data. To address such limitations, we propose the Few-shot ROBust (FROB) model for classification and few-shot OoD detection. We devise FROB for improved robustness and reliable confidence prediction for few-shot OoD detection. We generate the support boundary of the normal class distribution and combine it with few-shot Outlier Exposure (OE). We propose a self-supervised learning few-shot confidence boundary methodology based on generative and discriminative models. The contribution of FROB is the combination of the generated boundary in a self-supervised learning manner and the imposition of low confidence at this learned boundary. FROB implicitly generates strong adversarial samples on the boundary and forces samples from OoD, including our boundary, to be less confident by the classifier. FROB achieves generalization to unseen OoD with applicability to unknown, in the wild, test sets that do not correlate to the training datasets. To improve robustness, FROB redesigns OE to work even for zero-shots. By including our boundary, FROB reduces the threshold linked to the model's few-shot robustness; it maintains the OoD performance approximately independent of the number of few-shots. The few-shot robustness analysis evaluation of FROB on different sets and on One-Class Classification (OCC) data shows that FROB achieves competitive performance and outperforms benchmarks in terms of robustness to the outlier few-shot sample population and variability.

摘要: 目前，少射环境下的分类和失配(OOD)检测仍然是一个极具挑战性的课题，因为少射环境下的稀有性和有限的样本，以及敌方攻击的存在。实现这些目标对于安全、安保和防御方面的关键系统非常重要。同时，由于深度神经网络分类器对远离训练数据的OOD样本设置了很高的置信度，因此OOD检测是具有挑战性的。为了解决这些局限性，我们提出了用于分类和少射OOD检测的少射鲁棒(FROB)模型。我们设计了FROB来提高少射OOD检测的鲁棒性和可靠的置信度预测。我们生成正态类分布的支持边界，并将其与少镜头离群点曝光(OE)相结合。提出了一种基于产生式模型和判别式模型的自监督学习小概率置信边界方法。FROB的贡献是将以自监督学习方式生成的边界与在该学习边界上施加的低置信度相结合。FROB隐式地在边界上生成强对抗性样本，并迫使来自OOD(包括我们的边界)的样本被分类器降低信心。FROB实现了对看不见的OOD的泛化，适用于与训练数据集不相关的未知的野外测试集。为了提高健壮性，FROB重新设计了OE，使其即使在零炮情况下也能工作。通过包括我们的边界，FROB降低了与模型的少镜头稳健性相关的阈值；它保持了OOD性能与少镜头数量大致无关。在不同集合和一类分类数据上的少镜头稳健性分析评估表明，在对离群点、少镜头样本总体和变异性的鲁棒性方面，FROB达到了好胜的性能，并优于基准测试结果。结果表明，FROB达到了好胜的性能，并且在对异常点、少镜头样本总体和变异性的鲁棒性方面优于基准。



## **42. A Face Recognition System's Worst Morph Nightmare, Theoretically**

从理论上讲，人脸识别系统最糟糕的梦魇 cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15416v1)

**Authors**: Una M. Kelly, Raymond Veldhuis, Luuk Spreeuwers

**Abstracts**: It has been shown that Face Recognition Systems (FRSs) are vulnerable to morphing attacks, but most research focusses on landmark-based morphs. A second method for generating morphs uses Generative Adversarial Networks, which results in convincingly real facial images that can be almost as challenging for FRSs as landmark-based attacks. We propose a method to create a third, different type of morph, that has the advantage of being easier to train. We introduce the theoretical concept of \textit{worst-case morphs}, which are those morphs that are most challenging for a fixed FRS. For a set of images and corresponding embeddings in an FRS's latent space, we generate images that approximate these worst-case morphs using a mapping from embedding space back to image space. While the resulting images are not yet as challenging as other morphs, they can provide valuable information in future research on Morphing Attack Detection (MAD) methods and on weaknesses of FRSs. Methods for MAD need to be validated on more varied morph databases. Our proposed method contributes to achieving such variation.

摘要: 已有研究表明，人脸识别系统(FRS)容易受到变形攻击，但大多数研究都集中在基于标志性的变形攻击上。第二种生成变形的方法使用生成性对抗网络，它产生令人信服的真实面部图像，这对FRS来说几乎和基于里程碑的攻击一样具有挑战性。我们提出了一种方法来创建第三种不同类型的变形，其优点是更容易训练。我们引入了最坏情况变形的理论概念，它们是对固定FRS最具挑战性的变形。对于FRS的潜在空间中的一组图像和相应的嵌入，我们使用从嵌入空间返回到图像空间的映射来生成近似这些最坏情况下的变形的图像。虽然生成的图像还不像其他变形图像那样具有挑战性，但它们可以为未来变形攻击检测(MAD)方法的研究和FRS的弱点提供有价值的信息。MAD的方法需要在更多不同的变形数据库上进行验证。我们提出的方法有助于实现这种变化。



## **43. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.10969v2)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **44. COREATTACK: Breaking Up the Core Structure of Graphs**

COREATTACK：打破图的核心结构 cs.SI

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15276v1)

**Authors**: Bo Zhou, Yuqian Lv, Jinhuan Wang, Jian Zhang, Qi Xuan

**Abstracts**: The concept of k-core in complex networks plays a key role in many applications, e.g., understanding the global structure, or identifying central/critical nodes, of a network. A malicious attacker with jamming ability can exploit the vulnerability of the k-core structure to attack the network and invalidate the network analysis methods, e.g., reducing the k-shell values of nodes can deceive graph algorithms, leading to the wrong decisions. In this paper, we investigate the robustness of the k-core structure under adversarial attacks by deleting edges, for the first time. Firstly, we give the general definition of targeted k-core attack, map it to the set cover problem which is NP-hard, and further introduce a series of evaluation metrics to measure the performance of attack methods. Then, we propose $Q$ index theoretically as the probability that the terminal node of an edge does not belong to the innermost core, which is further used to guide the design of our heuristic attack methods, namely COREATTACK and GreedyCOREATTACK. The experiments on a variety of real-world networks demonstrate that our methods behave much better than a series of baselines, in terms of much smaller Edge Change Rate (ECR) and False Attack Rate (FAR), achieving state-of-the-art attack performance. More impressively, for certain real-world networks, only deleting one edge from the k-core may lead to the collapse of the innermost core, even if this core contains dozens of nodes. Such a phenomenon indicates that the k-core structure could be extremely vulnerable under adversarial attacks, and its robustness thus should be carefully addressed to ensure the security of many graph algorithms.

摘要: 在复杂网络中，k核的概念在许多应用中起着关键作用，例如，理解网络的全局结构或识别网络的中心/关键节点。具有干扰能力的恶意攻击者可以利用k-core结构的漏洞攻击网络，使网络分析方法失效，例如降低节点的k-shell值可以欺骗图算法，导致错误的决策。本文首次通过删除边的方法研究了k-core结构在敌意攻击下的健壮性。首先，给出了目标k-核攻击的一般定义，将其映射到NP-hard的集合覆盖问题，并进一步引入了一系列评价指标来衡量攻击方法的性能。然后，从理论上提出了$q$指标作为边的末端节点不属于最内核节点的概率，并进一步用它来指导我们的启发式攻击方法COREATTACK和GreedyCOREATTACK的设计。在各种真实网络上的实验表明，我们的方法在边缘变化率(ECR)和错误攻击率(FAR)方面明显优于一系列基线，达到了最先进的攻击性能。更令人印象深刻的是，对于某些现实世界的网络，只从k核中删除一条边可能会导致最里面的核崩溃，即使这个核包含几十个节点。这种现象表明k-core结构在敌意攻击下极易受到攻击，因此其健壮性是保证许多图算法安全的关键。



## **45. Black-box Adversarial Attacks on Commercial Speech Platforms with Minimal Information**

基于最小信息的商业语音平台黑盒对抗性攻击 cs.CR

A version of this paper appears in the proceedings of the 28th ACM  Conference on Computer and Communications Security (CCS 2021). The notes in  Tables 1 and 4 have been updated

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2110.09714v2)

**Authors**: Baolin Zheng, Peipei Jiang, Qian Wang, Qi Li, Chao Shen, Cong Wang, Yunjie Ge, Qingyang Teng, Shenyi Zhang

**Abstracts**: Adversarial attacks against commercial black-box speech platforms, including cloud speech APIs and voice control devices, have received little attention until recent years. The current "black-box" attacks all heavily rely on the knowledge of prediction/confidence scores to craft effective adversarial examples, which can be intuitively defended by service providers without returning these messages. In this paper, we propose two novel adversarial attacks in more practical and rigorous scenarios. For commercial cloud speech APIs, we propose Occam, a decision-only black-box adversarial attack, where only final decisions are available to the adversary. In Occam, we formulate the decision-only AE generation as a discontinuous large-scale global optimization problem, and solve it by adaptively decomposing this complicated problem into a set of sub-problems and cooperatively optimizing each one. Our Occam is a one-size-fits-all approach, which achieves 100% success rates of attacks with an average SNR of 14.23dB, on a wide range of popular speech and speaker recognition APIs, including Google, Alibaba, Microsoft, Tencent, iFlytek, and Jingdong, outperforming the state-of-the-art black-box attacks. For commercial voice control devices, we propose NI-Occam, the first non-interactive physical adversarial attack, where the adversary does not need to query the oracle and has no access to its internal information and training data. We combine adversarial attacks with model inversion attacks, and thus generate the physically-effective audio AEs with high transferability without any interaction with target devices. Our experimental results show that NI-Occam can successfully fool Apple Siri, Microsoft Cortana, Google Assistant, iFlytek and Amazon Echo with an average SRoA of 52% and SNR of 9.65dB, shedding light on non-interactive physical attacks against voice control devices.

摘要: 针对商业黑盒语音平台的对抗性攻击，包括云语音API和语音控制设备，直到最近几年才受到很少关注。目前的“黑匣子”攻击都严重依赖预测/置信分数的知识来制作有效的对抗性例子，服务提供商无需返回这些消息就可以直观地进行防御。在这篇文章中，我们提出了两个新的对抗性攻击，在更实际和更严格的情况下。对于商用云语音API，我们提出了OCCAM，这是一种仅限决策的黑盒对抗攻击，只有最终决策才能提供给对手。在OCCAM中，我们将只有决策的AE生成问题描述为一个不连续的大规模全局优化问题，并将这个复杂问题自适应地分解成一组子问题并对每个子问题进行协同优化来求解。我们的OCCAM是一种一刀切的方法，在谷歌、阿里巴巴、微软、腾讯、iFLYTEK、京东等各种流行的语音和说话人识别API上，实现了100%的攻击成功率，平均SNR为14.23dB，表现优于最先进的黑匣子攻击。对于商用语音控制设备，我们提出了NI-OCCAM，这是第一种非交互式的物理对抗攻击，对手不需要查询先知，也不能访问它的内部信息和训练数据。我们将对抗性攻击和模型反转攻击相结合，在不与目标设备交互的情况下生成物理上有效、可移植性高的音频AEs。我们的实验结果表明，NI-Occam能够成功欺骗Apple Siri、Microsoft Cortana、Google Assistant、iFLYTEK和Amazon Echo，平均SRoA达到52%，信噪比达到9.65dB，为针对语音控制设备的非交互式物理攻击提供了线索。



## **46. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

通过将不同的副本分发给不同的用户来缓解敌意攻击 cs.CR

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15160v1)

**Authors**: Jiyi Zhang, Wesley Joon-Wie Tann, Ee-Chien Chang

**Abstracts**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is to be distributed to multiple users, among which a malicious user attempts to attack another user. The malicious user probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's model in order to replicate the attack. We point out that by distributing different copies of the model to different users, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We first observed that training a model with different randomness indeed mitigates such replication to certain degree. However, there is no guarantee and retraining is computationally expensive. Next, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to induce different sets of adversarial samples in different copies in a more controllable manner. Experimentation studies show that our approach can significantly mitigate the attacks while retaining high classification accuracy. From this study, we believe that there are many further directions worth exploring.

摘要: 机器学习模型容易受到敌意攻击。在本文中，我们考虑将一个模型分发给多个用户的场景，其中一个恶意用户试图攻击另一个用户。恶意用户探测其模型副本以搜索敌意样本，然后将找到的样本呈现给受害者的模型，以便复制攻击。我们指出，通过将模型的不同副本分发给不同的用户，我们可以减轻攻击，使得在一个副本上发现的敌意样本在另一个副本上不起作用。我们首先观察到，训练具有不同随机性的模型确实在一定程度上减轻了这种复制。然而，这是没有保证的，再培训在计算上是昂贵的。接下来，我们提出了一种灵活的参数重写方法，可以直接修改模型的参数。这种方法不需要额外的训练，并且能够以更可控的方式在不同的副本中诱导不同的对抗性样本集。实验研究表明，该方法在保持较高分类准确率的同时，能显着缓解攻击。从本次研究来看，我们认为还有很多值得进一步探索的方向。



## **47. Adversarial Robustness of Deep Code Comment Generation**

深层代码注释生成的对抗健壮性 cs.SE

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2108.00213v3)

**Authors**: Yu Zhou, Xiaoqing Zhang, Juanjuan Shen, Tingting Han, Taolue Chen, Harald Gall

**Abstracts**: Deep neural networks (DNNs) have shown remarkable performance in a variety of domains such as computer vision, speech recognition, or natural language processing. Recently they also have been applied to various software engineering tasks, typically involving processing source code. DNNs are well-known to be vulnerable to adversarial examples, i.e., fabricated inputs that could lead to various misbehaviors of the DNN model while being perceived as benign by humans. In this paper, we focus on the code comment generation task in software engineering and study the robustness issue of the DNNs when they are applied to this task. We propose ACCENT, an identifier substitution approach to craft adversarial code snippets, which are syntactically correct and semantically close to the original code snippet, but may mislead the DNNs to produce completely irrelevant code comments. In order to improve the robustness, ACCENT also incorporates a novel training method, which can be applied to existing code comment generation models. We conduct comprehensive experiments to evaluate our approach by attacking the mainstream encoder-decoder architectures on two large-scale publicly available datasets. The results show that ACCENT efficiently produces stable attacks with functionality-preserving adversarial examples, and the generated examples have better transferability compared with baselines. We also confirm, via experiments, the effectiveness in improving model robustness with our training method.

摘要: 深度神经网络(DNNs)在计算机视觉、语音识别、自然语言处理等领域表现出显著的性能。最近，它们还被应用于各种软件工程任务，通常涉及处理源代码。众所周知，DNN很容易受到敌意示例的攻击，即在人类认为DNN模型是良性的情况下，可能会导致DNN模型的各种错误行为的捏造输入。本文针对软件工程中的代码注释生成任务，研究了DNN应用于该任务时的健壮性问题。我们提出了一种标识符替换方法Accent来制作敌意代码片段，这些代码片段在语法上是正确的，在语义上也接近于原始代码片段，但可能会误导DNN生成完全不相关的代码注释。为了提高鲁棒性，Accent还引入了一种新的训练方法，该方法可以应用于现有的代码注释生成模型。我们在两个大规模公开可用的数据集上进行了全面的实验，通过攻击主流的编解码器架构来评估我们的方法。实验结果表明，重音算法能有效地产生稳定的攻击，且生成的实例与基线相比具有更好的可移植性。通过实验，我们也证实了我们的训练方法在提高模型鲁棒性方面的有效性。



## **48. Living-Off-The-Land Command Detection Using Active Learning**

基于主动学习的陆上生活指挥检测 cs.CR

14 pages, published in RAID 2021

**SubmitDate**: 2021-11-30    [paper-pdf](http://arxiv.org/pdf/2111.15039v1)

**Authors**: Talha Ongun, Jack W. Stokes, Jonathan Bar Or, Ke Tian, Farid Tajaddodianfar, Joshua Neil, Christian Seifert, Alina Oprea, John C. Platt

**Abstracts**: In recent years, enterprises have been targeted by advanced adversaries who leverage creative ways to infiltrate their systems and move laterally to gain access to critical data. One increasingly common evasive method is to hide the malicious activity behind a benign program by using tools that are already installed on user computers. These programs are usually part of the operating system distribution or another user-installed binary, therefore this type of attack is called "Living-Off-The-Land". Detecting these attacks is challenging, as adversaries may not create malicious files on the victim computers and anti-virus scans fail to detect them. We propose the design of an Active Learning framework called LOLAL for detecting Living-Off-the-Land attacks that iteratively selects a set of uncertain and anomalous samples for labeling by a human analyst. LOLAL is specifically designed to work well when a limited number of labeled samples are available for training machine learning models to detect attacks. We investigate methods to represent command-line text using word-embedding techniques, and design ensemble boosting classifiers to distinguish malicious and benign samples based on the embedding representation. We leverage a large, anonymized dataset collected by an endpoint security product and demonstrate that our ensemble classifiers achieve an average F1 score of 0.96 at classifying different attack classes. We show that our active learning method consistently improves the classifier performance, as more training data is labeled, and converges in less than 30 iterations when starting with a small number of labeled instances.

摘要: 近年来，企业一直是高级对手的目标，他们利用创造性的方式渗透到他们的系统中，并横向移动以获取关键数据。一种越来越常见的规避方法是通过使用已安装在用户计算机上的工具将恶意活动隐藏在良性程序后面。这些程序通常是操作系统发行版或其他用户安装的二进制文件的一部分，因此这种类型的攻击被称为“生活在陆地上”(Living-Off-Land)。检测这些攻击具有挑战性，因为攻击者可能不会在受攻击的计算机上创建恶意文件，并且防病毒扫描无法检测到它们。我们提出了一个称为LOLAL的主动学习框架来检测生活在陆地上的攻击，该框架迭代地选择一组不确定和异常的样本供人类分析员进行标记。LOLAL专门设计用于在有限数量的标签样本可用于训练机器学习模型以检测攻击时很好地工作。我们研究了使用词嵌入技术来表示命令行文本的方法，并设计了基于嵌入表示的集成Boosting分类器来区分恶意样本和良性样本。我们利用终端安全产品收集的大型匿名数据集，展示了我们的集成分类器在分类不同攻击类别时的平均F1得分为0.96。我们表明，随着更多的训练数据被标注，我们的主动学习方法持续地提高了分类器的性能，并且当从少量的标注实例开始时，在不到30次的迭代中就收敛了。



## **49. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

对弗兰克-沃尔夫对抗性训练的认识与提高效率 cs.LG

Accepted to ICML 2021 Adversarial Machine Learning Workshop. Under  review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2012.12368v4)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense against such attacks. Due to the high computation time for generating strong adversarial examples for AT, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training. Although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal $\ell_2$ distortion, while standard networks have lower distortion. Furthermore, it is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps to increase efficiency without compromising robustness. FW-AT-Adapt provides training times on par with single-step fast AT methods and improves closing the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

摘要: 深度神经网络很容易被称为对抗性攻击的小扰动所愚弄。对抗性训练(AT)是一种近似地解决鲁棒优化问题以最小化最坏情况下的损失的技术，被广泛认为是对抗此类攻击的最有效的防御方法。由于AT生成强对抗性样本的计算时间较长，因此提出了一种单步方法来减少训练时间。然而，这些方法存在灾难性的过度拟合问题，在训练过程中对抗性准确率会下降。虽然已经提出了一些改进措施，但它们增加了训练时间，鲁棒性远不及多步AT。我们建立了一个基于FW优化的对抗性训练理论框架(FW-AT)，该框架揭示了损失情况与$ELL_INFTY$FW攻击的$\ELL_2$失真之间的几何关系。分析表明，FW攻击的高失真等价于攻击路径上的小梯度变化。在不同的深度神经网络结构上实验证明，对鲁棒模型的$ellinty$攻击获得了接近最大的$ell2$失真，而标准网络具有较低的失真。此外，实验还表明，灾难性过拟合与FW攻击的低失真有很强的相关性。为了证明我们的理论框架的有效性，我们开发了一种新的对抗性训练算法FW-AT-Adapt，它使用一个简单的失真度量来调整攻击步骤的数量，从而在不影响鲁棒性的情况下提高效率。FW-AT-Adapt提供与单步快速AT方法相当的训练时间，并改善了快速AT方法与多步PGD-AT之间的差距，同时最大限度地降低了白盒和黑盒设置中的对抗精度损失。



## **50. MedRDF: A Robust and Retrain-Less Diagnostic Framework for Medical Pretrained Models Against Adversarial Attack**

MedRDF：一种健壮且无需再训练的医学预训练模型对抗攻击诊断框架 cs.CV

TMI under review

**SubmitDate**: 2021-11-29    [paper-pdf](http://arxiv.org/pdf/2111.14564v1)

**Authors**: Mengting Xu, Tao Zhang, Daoqiang Zhang

**Abstracts**: Deep neural networks are discovered to be non-robust when attacked by imperceptible adversarial examples, which is dangerous for it applied into medical diagnostic system that requires high reliability. However, the defense methods that have good effect in natural images may not be suitable for medical diagnostic tasks. The preprocessing methods (e.g., random resizing, compression) may lead to the loss of the small lesions feature in the medical image. Retraining the network on the augmented data set is also not practical for medical models that have already been deployed online. Accordingly, it is necessary to design an easy-to-deploy and effective defense framework for medical diagnostic tasks. In this paper, we propose a Robust and Retrain-Less Diagnostic Framework for Medical pretrained models against adversarial attack (i.e., MedRDF). It acts on the inference time of the pertained medical model. Specifically, for each test image, MedRDF firstly creates a large number of noisy copies of it, and obtains the output labels of these copies from the pretrained medical diagnostic model. Then, based on the labels of these copies, MedRDF outputs the final robust diagnostic result by majority voting. In addition to the diagnostic result, MedRDF produces the Robust Metric (RM) as the confidence of the result. Therefore, it is convenient and reliable to utilize MedRDF to convert pre-trained non-robust diagnostic models into robust ones. The experimental results on COVID-19 and DermaMNIST datasets verify the effectiveness of our MedRDF in improving the robustness of medical diagnostic models.

摘要: 由于深层神经网络应用于可靠性要求高的医疗诊断系统，在受到潜伏的敌意攻击时表现出非稳健性，这是很危险的。然而，在自然图像中效果较好的防御方法可能不适用于医学诊断任务。预处理方法(例如，随机调整大小、压缩)可能会导致医学图像中的小病变特征丢失。对于已经在线部署的医疗模型来说，在增强的数据集上重新培训网络也是不切实际的。因此，有必要为医疗诊断任务设计一个易于部署和有效的防御框架。在本文中，我们提出了一个健壮且无需再训练的医学预训练模型抗攻击诊断框架(MedRDF)。它作用于相关医学模型的推理时间。具体地说，对于每幅测试图像，MedRDF首先为其创建大量的噪声副本，并从预先训练的医疗诊断模型中获得这些副本的输出标签。然后，基于这些副本的标签，MedRDF通过多数投票输出最终的鲁棒诊断结果。除了诊断结果之外，MedRDF还生成稳健度量(RM)作为结果的置信度。因此，利用MedRDF将预先训练好的非稳健诊断模型转换为稳健的诊断模型是方便可靠的。在冠状病毒和DermaMNIST数据集上的实验结果验证了我们的MedRDF在提高医疗诊断模型鲁棒性方面的有效性。



