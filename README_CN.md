# Latest Adversarial Attack Papers
**update at 2021-12-20 23:31:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Formalizing Generalization and Robustness of Neural Networks to Weight Perturbations**

神经网络对权重摄动的泛化和鲁棒性的形式化 cs.LG

This version has been accepted for poster presentation at NeurIPS  2021

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2103.02200v2)

**Authors**: Yu-Lin Tsai, Chia-Yi Hsu, Chia-Mu Yu, Pin-Yu Chen

**Abstracts**: Studying the sensitivity of weight perturbation in neural networks and its impacts on model performance, including generalization and robustness, is an active research topic due to its implications on a wide range of machine learning tasks such as model compression, generalization gap assessment, and adversarial attacks. In this paper, we provide the first integral study and analysis for feed-forward neural networks in terms of the robustness in pairwise class margin and its generalization behavior under weight perturbation. We further design a new theory-driven loss function for training generalizable and robust neural networks against weight perturbations. Empirical experiments are conducted to validate our theoretical analysis. Our results offer fundamental insights for characterizing the generalization and robustness of neural networks against weight perturbations.

摘要: 研究神经网络中权重扰动的敏感性及其对模型性能(包括泛化和鲁棒性)的影响是一个活跃的研究课题，因为它涉及到广泛的机器学习任务，如模型压缩、泛化差距评估和敌意攻击。本文首次对前馈神经网络的两类边界鲁棒性及其在权值扰动下的泛化行为进行了整体研究和分析。我们进一步设计了一种新的理论驱动的损失函数，用于训练泛化的、抗权重扰动的鲁棒神经网络。通过实证实验验证了我们的理论分析。我们的结果为表征神经网络的泛化和抗权重扰动的鲁棒性提供了基本的见解。



## **2. Reasoning Chain Based Adversarial Attack for Multi-hop Question Answering**

基于推理链的多跳问答对抗性攻击 cs.CL

10 pages including reference, 4 figures

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09658v1)

**Authors**: Jiayu Ding, Siyuan Wang, Qin Chen, Zhongyu Wei

**Abstracts**: Recent years have witnessed impressive advances in challenging multi-hop QA tasks. However, these QA models may fail when faced with some disturbance in the input text and their interpretability for conducting multi-hop reasoning remains uncertain. Previous adversarial attack works usually edit the whole question sentence, which has limited effect on testing the entity-based multi-hop inference ability. In this paper, we propose a multi-hop reasoning chain based adversarial attack method. We formulate the multi-hop reasoning chains starting from the query entity to the answer entity in the constructed graph, which allows us to align the question to each reasoning hop and thus attack any hop. We categorize the questions into different reasoning types and adversarially modify part of the question corresponding to the selected reasoning hop to generate the distracting sentence. We test our adversarial scheme on three QA models on HotpotQA dataset. The results demonstrate significant performance reduction on both answer and supporting facts prediction, verifying the effectiveness of our reasoning chain based attack method for multi-hop reasoning models and the vulnerability of them. Our adversarial re-training further improves the performance and robustness of these models.

摘要: 近年来，在具有挑战性的多跳QA任务方面取得了令人印象深刻的进展。然而，当遇到输入文本中的某些干扰时，这些QA模型可能会失败，并且它们对进行多跳推理的解释力仍然不确定。以往的对抗性攻击工作通常都是对整个问句进行编辑，这对测试基于实体的多跳推理能力的效果有限。本文提出了一种基于多跳推理链的对抗性攻击方法。在构造的图中，我们构造了从查询实体到答案实体的多跳推理链，允许我们将问题对齐到每个推理跳，从而攻击任何一跳。我们将问题分类为不同的推理类型，并对所选择的推理跳对应的部分问题进行对抗性修改，以生成分散注意力的句子。我们在HotpotQA数据集上的三个QA模型上测试了我们的对抗性方案。实验结果表明，基于推理链的多跳推理模型攻击方法在答案和支持事实预测方面都有明显的性能下降，验证了该方法的有效性和脆弱性。我们的对抗性再训练进一步提高了这些模型的性能和鲁棒性。



## **3. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？基于Deep RL的最优高效规避攻击研究 cs.LG

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2106.05087v2)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named ''actor'' and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)Agent在状态观测(在一定约束范围内)的最强/最优对抗扰动下的最坏情况下的性能，对于理解RL Agent的鲁棒性是至关重要的。然而，无论是从我们是否能找到最佳攻击，还是从我们找到最佳攻击的效率来看，找到最佳对手都是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将Agent视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能会变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作效率要高得多。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验鲁棒性。



## **4. Dynamics-aware Adversarial Attack of 3D Sparse Convolution Network**

三维稀疏卷积网络的动态感知敌意攻击 cs.CV

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09428v1)

**Authors**: An Tao, Yueqi Duan, He Wang, Ziyi Wu, Pengliang Ji, Haowen Sun, Jie Zhou, Jiwen Lu

**Abstracts**: In this paper, we investigate the dynamics-aware adversarial attack problem in deep neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed networks, e.g. 3D sparse convolution network, which contains input-dependent execution to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture changes afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we re-formulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on various datasets show that our LGM achieves impressive performance on semantic segmentation and classification. Compared with the dynamic-unaware methods, LGM achieves about 20% lower mIoU averagely on the ScanNet and S3DIS datasets. LGM also outperforms the recent point cloud attacks.

摘要: 本文研究了深层神经网络中动态感知的敌意攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中都是固定的。然而，这一假设并不适用于最近提出的许多网络，例如3D稀疏卷积网络，它包含依赖输入的执行以提高计算效率。这导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种领先梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度来感知网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不感知动态变化的方法更好地“引导”下一步。在不同数据集上的大量实验表明，我们的LGM在语义分割和分类方面取得了令人印象深刻的性能。与动态无感知方法相比，LGM在ScanNet和S3DIS数据集上的MIU值平均降低了20%左右。LGM的性能也优于最近的点云攻击。



## **5. APTSHIELD: A Stable, Efficient and Real-time APT Detection System for Linux Hosts**

APTSHIELD：一种稳定、高效、实时的Linux主机APT检测系统 cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09008v2)

**Authors**: Tiantian Zhu, Jinkai Yu, Tieming Chen, Jiayu Wang, Jie Ying, Ye Tian, Mingqi Lv, Yan Chen, Yuan Fan, Ting Wang

**Abstracts**: Advanced Persistent Threat (APT) attack usually refers to the form of long-term, covert and sustained attack on specific targets, with an adversary using advanced attack techniques to destroy the key facilities of an organization. APT attacks have caused serious security threats and massive financial loss worldwide. Academics and industry thereby have proposed a series of solutions to detect APT attacks, such as dynamic/static code analysis, traffic detection, sandbox technology, endpoint detection and response (EDR), etc. However, existing defenses are failed to accurately and effectively defend against the current APT attacks that exhibit strong persistent, stealthy, diverse and dynamic characteristics due to the weak data source integrity, large data processing overhead and poor real-time performance in the process of real-world scenarios.   To overcome these difficulties, in this paper we propose APTSHIELD, a stable, efficient and real-time APT detection system for Linux hosts. In the aspect of data collection, audit is selected to stably collect kernel data of the operating system so as to carry out a complete portrait of the attack based on comprehensive analysis and comparison of existing logging tools; In the aspect of data processing, redundant semantics skipping and non-viable node pruning are adopted to reduce the amount of data, so as to reduce the overhead of the detection system; In the aspect of attack detection, an APT attack detection framework based on ATT\&CK model is designed to carry out real-time attack response and alarm through the transfer and aggregation of labels. Experimental results on both laboratory and Darpa Engagement show that our system can effectively detect web vulnerability attacks, file-less attacks and remote access trojan attacks, and has a low false positive rate, which adds far more value than the existing frontier work.

摘要: 高级持续威胁(APT)攻击通常是指敌方利用先进的攻击技术，对特定目标进行长期、隐蔽、持续的攻击，破坏组织的关键设施。APT攻击在全球范围内造成了严重的安全威胁和巨大的经济损失。学术界和产业界为此提出了一系列检测APT攻击的解决方案，如动态/静电代码分析、流量检测、沙盒技术、端点检测与响应等。然而，现有的防御措施在现实场景中，由于数据源完整性差、数据处理开销大、实时性差等原因，无法准确有效地防御当前APT攻击，表现出较强的持久性、隐蔽性、多样性和动态性。为了克服这些困难，本文提出了一种稳定、高效、实时的Linux主机APT检测系统APTSHIELD。在数据采集方面，在综合分析比较现有日志记录工具的基础上，选择AUDIT稳定采集操作系统内核数据，对攻击进行完整的刻画；在数据处理方面，采用冗余语义跳过和不可生存节点剪枝，减少了数据量，降低了检测系统的开销；在攻击检测方面，设计了基于ATT\&CK模型的APT攻击检测框架，通过传输进行实时攻击响应和报警实验室和DAPA实验结果表明，该系统能够有效地检测出Web漏洞攻击、无文件攻击和远程访问木马攻击，并且误报率较低，比现有的前沿工作有更大的增值价值。研究结果表明，该系统能够有效地检测出网络漏洞攻击、无文件攻击和远程访问木马攻击，并且具有较低的误报率。



## **6. Deep Bayesian Learning for Car Hacking Detection**

深度贝叶斯学习在汽车黑客检测中的应用 cs.CR

**SubmitDate**: 2021-12-17    [paper-pdf](http://arxiv.org/pdf/2112.09333v1)

**Authors**: Laha Ale, Scott A. King, Ning Zhang

**Abstracts**: With the rise of self-drive cars and connected vehicles, cars are equipped with various devices to assistant the drivers or support self-drive systems. Undoubtedly, cars have become more intelligent as we can deploy more and more devices and software on the cars. Accordingly, the security of assistant and self-drive systems in the cars becomes a life-threatening issue as smart cars can be invaded by malicious attacks that cause traffic accidents. Currently, canonical machine learning and deep learning methods are extensively employed in car hacking detection. However, machine learning and deep learning methods can easily be overconfident and defeated by carefully designed adversarial examples. Moreover, those methods cannot provide explanations for security engineers for further analysis. In this work, we investigated Deep Bayesian Learning models to detect and analyze car hacking behaviors. The Bayesian learning methods can capture the uncertainty of the data and avoid overconfident issues. Moreover, the Bayesian models can provide more information to support the prediction results that can help security engineers further identify the attacks. We have compared our model with deep learning models and the results show the advantages of our proposed model. The code of this work is publicly available

摘要: 随着自动驾驶汽车和联网汽车的兴起，汽车配备了各种设备来辅助司机或支持自动驾驶系统。毫无疑问，汽车已经变得更加智能，因为我们可以在汽车上部署越来越多的设备和软件。因此，汽车中助手和自动驾驶系统的安全成为一个危及生命的问题，因为智能汽车可能会受到恶意攻击，导致交通事故。目前，规范的机器学习和深度学习方法被广泛应用于汽车黑客检测中。然而，机器学习和深度学习方法很容易过于自信，并被精心设计的对抗性例子所击败。此外，这些方法不能为安全工程师提供进一步分析的解释。在这项工作中，我们研究了深度贝叶斯学习模型来检测和分析汽车黑客行为。贝叶斯学习方法可以捕捉数据的不确定性，避免过度自信的问题。此外，贝叶斯模型可以提供更多的信息来支持预测结果，从而帮助安全工程师进一步识别攻击。我们将该模型与深度学习模型进行了比较，结果表明了该模型的优越性。这部作品的代码是公开提供的



## **7. Generation of Wheel Lockup Attacks on Nonlinear Dynamics of Vehicle Traction**

车辆牵引非线性动力学中车轮闭锁攻击的产生 eess.SY

Submitted to American Control Conference 2022 (ACC 2022), 6 pages

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09229v1)

**Authors**: Alireza Mohammadi, Hafiz Malik, Masoud Abbaszadeh

**Abstracts**: There is ample evidence in the automotive cybersecurity literature that the car brake ECUs can be maliciously reprogrammed. Motivated by such threat, this paper investigates the capabilities of an adversary who can directly control the frictional brake actuators and would like to induce wheel lockup conditions leading to catastrophic road injuries. This paper demonstrates that the adversary despite having a limited knowledge of the tire-road interaction characteristics has the capability of driving the states of the vehicle traction dynamics to a vicinity of the lockup manifold in a finite time by means of a properly designed attack policy for the frictional brakes. This attack policy relies on employing a predefined-time controller and a nonlinear disturbance observer acting on the wheel slip error dynamics. Simulations under various road conditions demonstrate the effectiveness of the proposed attack policy.

摘要: 汽车网络安全文献中有大量证据表明，汽车刹车ECU可以被恶意重新编程。在这种威胁的驱使下，本文调查了一个可以直接控制摩擦制动执行器并想要诱导车轮锁定条件导致灾难性道路伤害的对手的能力。通过合理设计摩擦制动器的攻击策略，证明了敌方尽管对轮胎-路面相互作用特性知之甚少，但仍有能力在有限的时间内将车辆牵引动力学状态驱动到闭锁歧管附近。该攻击策略依赖于采用预定义时间控制器和作用于车轮打滑误差动态的非线性扰动观测器。在不同路况下的仿真实验验证了所提出的攻击策略的有效性。



## **8. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

您所需要的只是RAW：使用摄像机图像管道防御敌意攻击 cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09219v1)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.

摘要: 现有的用于计算机视觉任务的神经网络容易受到敌意攻击：向输入图像添加不可察觉的扰动可以欺骗这些方法在没有扰动的情况下对正确预测的图像进行错误预测。各种防御方法已经提出了图像到图像的映射方法，或者在训练过程中包括这些扰动，或者在预处理去噪步骤中去除它们。在这样做时，现有方法通常忽略没有捕获今天数据集中的自然的rgb图像，而实际上是从在捕获中遭受各种降级的原始颜色过滤阵列捕获中恢复的。在这项工作中，我们利用这个原始数据分布作为对抗防御的经验先验。具体地说，我们提出了一种模型不可知的对抗防御方法，该方法将输入的RGB图像映射到拜耳原始空间，然后使用学习的摄像机图像信号处理(ISP)流水线将输入的RGB图像映射回输出RGB，以消除潜在的敌对模式。该方法作为一个现成的预处理模块，与特定模型的对抗性训练方法不同，不需要对抗性图像进行训练。因此，该方法可以推广到看不见的任务，而不需要额外的再培训。在不同视觉任务(如分类、语义分割、目标检测)的大规模数据集(如ImageNet、CoCo)上的实验验证了该方法在跨任务域的性能上显著优于现有方法。



## **9. Direction-Aggregated Attack for Transferable Adversarial Examples**

可转移对抗性实例的方向聚集攻击 cs.LG

ACM JETC JOURNAL Accepted

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2104.09172v2)

**Authors**: Tianjin Huang, Vlado Menkovski, Yulong Pei, YuHao Wang, Mykola Pechenizkiy

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that are crafted by imposing imperceptible changes to the inputs. However, these adversarial examples are most successful in white-box settings where the model and its parameters are available. Finding adversarial examples that are transferable to other models or developed in a black-box setting is significantly more difficult. In this paper, we propose the Direction-Aggregated adversarial attacks that deliver transferable adversarial examples. Our method utilizes aggregated direction during the attack process for avoiding the generated adversarial examples overfitting to the white-box model. Extensive experiments on ImageNet show that our proposed method improves the transferability of adversarial examples significantly and outperforms state-of-the-art attacks, especially against adversarial robust models. The best averaged attack success rates of our proposed method reaches 94.6\% against three adversarial trained models and 94.8\% against five defense methods. It also reveals that current defense approaches do not prevent transferable adversarial attacks.

摘要: 深层神经网络很容易受到敌意例子的攻击，这些例子是通过对输入进行潜移默化的改变而精心设计的。然而，这些对抗性的例子在模型及其参数可用的白盒设置中最为成功。寻找可以转移到其他模型或在黑盒环境中开发的对抗性示例要困难得多。在这篇文章中，我们提出了提供可转移的对抗性例子的方向聚集对抗性攻击。我们的方法在攻击过程中利用聚合方向来避免生成的对抗性示例与白盒模型过度拟合。在ImageNet上的大量实验表明，我们提出的方法显著提高了对抗性实例的可移植性，并且优于最新的攻击，特别是针对对抗性健壮性模型的攻击。该方法对3种对抗性训练模型的最优平均攻击成功率为94.6%，对5种防御方法的最优平均攻击成功率为94.8%。它还揭示了当前的防御方法不能阻止可转移的对抗性攻击。



## **10. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

TAFIM：针对面部图像处理的有针对性的敌意攻击 cs.CV

Paper Video: https://youtu.be/btHCrVMKbzw Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09151v1)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face image manipulation methods, despite having many beneficial applications in computer graphics, can also raise concerns by affecting an individual's privacy or spreading disinformation. In this work, we propose a proactive defense to prevent face manipulation from happening in the first place. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones. In addition, we propose to leverage a differentiable compression approximation, hence making generated perturbations robust to common image compression. We further show that a generated perturbation can simultaneously prevent against multiple manipulation methods.

摘要: 尽管人脸图像处理方法在计算机图形学中有许多有益的应用，但也可能会影响个人隐私或传播虚假信息，从而引起人们的担忧。在这项工作中，我们提出了一种主动防御措施，从一开始就防止面部操纵的发生。为此，我们引入了一种新的数据驱动方法，该方法产生嵌入在原始图像中的特定于图像的扰动。其关键思想是，这些受保护的图像通过使操作模型产生预定义的操作目标(在我们的例子中为均匀着色的输出图像)而不是实际的操作来防止面部操作。与单独优化每个图像的噪声模式的传统对抗性攻击相比，我们的通用模型只需要一次前向传递，因此运行速度快几个数量级，并允许轻松集成到图像处理堆栈中，即使在资源受限的设备(如智能手机)上也是如此。此外，我们建议利用可微压缩近似，从而使生成的扰动对普通图像压缩具有鲁棒性。我们进一步证明了一个产生的扰动可以同时防止多种处理方法。



## **11. Combating Adversaries with Anti-Adversaries**

以反制敌，以反制敌 cs.LG

Accepted to AAAI Conference on Artificial Intelligence (AAAI'22)

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2103.14347v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Ali Thabet, Adel Bibi, Philip H. S. Torr, Bernard Ghanem

**Abstracts**: Deep neural networks are vulnerable to small input perturbations known as adversarial attacks. Inspired by the fact that these adversaries are constructed by iteratively minimizing the confidence of a network for the true class label, we propose the anti-adversary layer, aimed at countering this effect. In particular, our layer generates an input perturbation in the opposite direction of the adversarial one and feeds the classifier a perturbed version of the input. Our approach is training-free and theoretically supported. We verify the effectiveness of our approach by combining our layer with both nominally and robustly trained models and conduct large-scale experiments from black-box to adaptive attacks on CIFAR10, CIFAR100, and ImageNet. Our layer significantly enhances model robustness while coming at no cost on clean accuracy.

摘要: 深度神经网络容易受到被称为对抗性攻击的小输入扰动的影响。受这些敌手是通过迭代最小化网络对真实类标签的置信度来构建的事实的启发，我们提出了反敌手层，旨在对抗这一影响。具体地说，我们的层在对抗性的输入扰动的相反方向上生成输入扰动，并向分类器提供输入的扰动版本。我们的方法是免培训的，理论上是有支持的。我们通过将我们的层与名义上和鲁棒训练的模型相结合来验证我们的方法的有效性，并对CIFAR10、CIFAR100和ImageNet进行了从黑盒到自适应攻击的大规模实验。我们的层极大地增强了模型的健壮性，同时在干净的精确度上没有任何代价。



## **12. Anti-Tamper Radio: System-Level Tamper Detection for Computing Systems**

防篡改无线电：面向计算系统的系统级篡改检测 cs.CR

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.09014v1)

**Authors**: Paul Staat, Johannes Tobisch, Christian Zenger, Christof Paar

**Abstracts**: A whole range of attacks becomes possible when adversaries gain physical access to computing systems that process or contain sensitive data. Examples include side-channel analysis, bus probing, device cloning, or implanting hardware Trojans. Defending against these kinds of attacks is considered a challenging endeavor, requiring anti-tamper solutions to monitor the physical environment of the system. Current solutions range from simple switches, which detect if a case is opened, to meshes of conducting material that provide more fine-grained detection of integrity violations. However, these solutions suffer from an intricate trade-off between physical security on the one side and reliability, cost, and difficulty to manufacture on the other. In this work, we demonstrate that radio wave propagation in an enclosed system of complex geometry is sensitive against adversarial physical manipulation. We present an anti-tamper radio (ATR) solution as a method for tamper detection, which combines high detection sensitivity and reliability with ease-of-use. ATR constantly monitors the wireless signal propagation behavior within the boundaries of a metal case. Tamper attempts such as insertion of foreign objects, will alter the observed radio signal response, subsequently raising an alarm. The ATR principle is applicable in many computing systems that require physical security such as servers, ATMs, and smart meters. As a case study, we use 19" servers and thoroughly investigate capabilities and limits of the ATR. Using a custom-built automated probing station, we simulate probing attacks by inserting needles with high precision into protected environments. Our experimental results show that our ATR implementation can detect 16 mm insertions of needles of diameter as low as 0.1 mm under ideal conditions. In the more realistic environment of a running 19" server, we demonstrate reliable [...]

摘要: 当攻击者获得对处理或包含敏感数据的计算系统的物理访问权限时，整个范围的攻击都成为可能。示例包括侧信道分析、总线探测、设备克隆或植入硬件特洛伊木马。防御这类攻击被认为是一项具有挑战性的工作，需要防篡改解决方案来监控系统的物理环境。目前的解决方案范围很广，从检测案件是否打开的简单开关，到提供更细粒度的完整性违规检测的导电材料网状结构。然而，这些解决方案需要在物理安全性和可靠性、成本以及制造难度之间进行复杂的权衡。在这项工作中，我们证明了无线电波在复杂几何封闭系统中的传播对敌方的物理操纵是敏感的。我们提出了一种防篡改无线电(ATR)解决方案，作为篡改检测的一种方法，它结合了高检测灵敏度和可靠性以及易用性。ATR持续监控金属外壳边界内的无线信号传播行为。诸如插入异物之类的篡改尝试将改变观测到的无线电信号响应，随后发出警报。ATR原则适用于许多需要物理安全的计算系统，如服务器、自动取款机和智能电表。作为案例研究，我们使用了19“服务器，并深入研究了ATR的能力和局限性。使用定制的自动探测站，我们通过在受保护环境中插入高精度的针来模拟探测攻击。我们的实验结果表明，在理想条件下，我们的ATR实现可以检测到直径低至0.1 mm的16 mm插入针。在运行19”服务器的更真实的环境中，我们展示了可靠的[.]



## **13. A Heterogeneous Graph Learning Model for Cyber-Attack Detection**

一种用于网络攻击检测的异构图学习模型 cs.CR

12pages,7figures,40 references

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08986v1)

**Authors**: Mingqi Lv, Chengyu Dong, Tieming Chen, Tiantian Zhu, Qijie Song, Yuan Fan

**Abstracts**: A cyber-attack is a malicious attempt by experienced hackers to breach the target information system. Usually, the cyber-attacks are characterized as hybrid TTPs (Tactics, Techniques, and Procedures) and long-term adversarial behaviors, making the traditional intrusion detection methods ineffective. Most existing cyber-attack detection systems are implemented based on manually designed rules by referring to domain knowledge (e.g., threat models, threat intelligences). However, this process is lack of intelligence and generalization ability. Aiming at this limitation, this paper proposes an intelligent cyber-attack detection method based on provenance data. To effective and efficient detect cyber-attacks from a huge number of system events in the provenance data, we firstly model the provenance data by a heterogeneous graph to capture the rich context information of each system entities (e.g., process, file, socket, etc.), and learns a semantic vector representation for each system entity. Then, we perform online cyber-attack detection by sampling a small and compact local graph from the heterogeneous graph, and classifying the key system entities as malicious or benign. We conducted a series of experiments on two provenance datasets with real cyber-attacks. The experiment results show that the proposed method outperforms other learning based detection models, and has competitive performance against state-of-the-art rule based cyber-attack detection systems.

摘要: 网络攻击是有经验的黑客恶意尝试入侵目标信息系统。通常情况下，网络攻击的特点是战术、技术和过程的混合性和长期的敌意行为，使得传统的入侵检测方法失效。大多数现有的网络攻击检测系统都是基于人工设计的规则，通过参考领域知识(例如，威胁模型、威胁情报)来实现的。然而，这一过程缺乏智能性和泛化能力。针对这一局限性，本文提出了一种基于起源数据的智能网络攻击检测方法。为了有效、高效地从起源数据中的大量系统事件中检测网络攻击，我们首先用异构图对起源数据进行建模，以获取每个系统实体(如进程、文件、套接字等)丰富的上下文信息，并学习每个系统实体的语义向量表示。然后，通过从异构图中抽取一个小而紧凑的局部图进行在线网络攻击检测，并对关键系统实体进行恶意或良性分类。我们在两个具有真实网络攻击的来源数据集上进行了一系列的实验。实验结果表明，该方法的性能优于其他基于学习的检测模型，与目前最新的基于规则的网络攻击检测系统相比，具有好胜的性能。



## **14. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

Accepted at NeurIPS 2021, including the appendix. In the previous  versions (v1 and v2), the experimental results of Table 10 are incorrect and  have been corrected in the current version

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2111.07492v3)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法没有超参数和预训练。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **15. Addressing Adversarial Machine Learning Attacks in Smart Healthcare Perspectives**

从智能医疗的角度解决对抗性机器学习攻击 cs.DC

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08862v1)

**Authors**: Arawinkumaar Selvakkumar, Shantanu Pal, Zahra Jadidi

**Abstracts**: Smart healthcare systems are gaining popularity with the rapid development of intelligent sensors, the Internet of Things (IoT) applications and services, and wireless communications. However, at the same time, several vulnerabilities and adversarial attacks make it challenging for a safe and secure smart healthcare system from a security point of view. Machine learning has been used widely to develop suitable models to predict and mitigate attacks. Still, the attacks could trick the machine learning models and misclassify outputs generated by the model. As a result, it leads to incorrect decisions, for example, false disease detection and wrong treatment plans for patients. In this paper, we address the type of adversarial attacks and their impact on smart healthcare systems. We propose a model to examine how adversarial attacks impact machine learning classifiers. To test the model, we use a medical image dataset. Our model can classify medical images with high accuracy. We then attacked the model with a Fast Gradient Sign Method attack (FGSM) to cause the model to predict the images and misclassify them inaccurately. Using transfer learning, we train a VGG-19 model with the medical dataset and later implement the FGSM to the Convolutional Neural Network (CNN) to examine the significant impact it causes on the performance and accuracy of the machine learning model. Our results demonstrate that the adversarial attack misclassifies the images, causing the model's accuracy rate to drop from 88% to 11%.

摘要: 随着智能传感器、物联网(IoT)应用和服务以及无线通信的快速发展，智能医疗系统越来越受欢迎。然而，与此同时，几个漏洞和对抗性攻击从安全角度对安全、安全的智能医疗系统提出了挑战。机器学习已被广泛用于开发合适的模型来预测和减轻攻击。尽管如此，这些攻击可能会欺骗机器学习模型，并对模型生成的输出进行错误分类。因此，它会导致错误的决定，例如，对患者的错误疾病检测和错误的治疗计划。在本文中，我们讨论了敌意攻击的类型及其对智能医疗系统的影响。我们提出了一个模型来检验敌意攻击如何影响机器学习分类器。为了测试该模型，我们使用了一个医学图像数据集。该模型能够对医学图像进行高精度的分类。然后，我们利用快速梯度符号方法(FGSM)攻击该模型，使该模型预测图像并对其进行错误分类。利用转移学习，我们用医学数据集训练了一个VGG-19模型，然后将FGSM实现到卷积神经网络(CNN)中，以检验它对机器学习模型的性能和精度造成的显著影响。实验结果表明，敌意攻击导致图像分类错误，导致模型的准确率从88%下降到11%。



## **16. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

面向鲁棒神经图像压缩：对抗性攻击与模型优化 cs.CV

**SubmitDate**: 2021-12-16    [paper-pdf](http://arxiv.org/pdf/2112.08691v1)

**Authors**: Tong Chen, Zhan Ma

**Abstracts**: Deep neural network based image compression has been extensively studied. Model robustness is largely overlooked, though it is crucial to service enabling. We perform the adversarial attack by injecting a small amount of noise perturbation to original source images, and then encode these adversarial examples using prevailing learnt image compression models. Experiments report severe distortion in the reconstruction of adversarial examples, revealing the general vulnerability of existing methods, regardless of the settings used in underlying compression model (e.g., network architecture, loss function, quality scale) and optimization strategy used for injecting perturbation (e.g., noise threshold, signal distance measurement). Later, we apply the iterative adversarial finetuning to refine pretrained models. In each iteration, random source images and adversarial examples are mixed to update underlying model. Results show the effectiveness of the proposed finetuning strategy by substantially improving the compression model robustness. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learnt image compression solution. All materials have been made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.

摘要: 基于深度神经网络的图像压缩已经得到了广泛的研究。模型健壮性在很大程度上被忽视了，尽管它对服务启用至关重要。我们通过在原始源图像中注入少量的噪声扰动来执行对抗性攻击，然后使用主流的学习图像压缩模型对这些对抗性示例进行编码。实验报告了对抗性示例的重建中的严重失真，揭示了现有方法的一般脆弱性，而与底层压缩模型(例如，网络架构、损失函数、质量尺度)和用于注入扰动的优化策略(例如，噪声阈值、信号距离测量)中使用的设置无关。然后，我们应用迭代对抗性精调来精炼预先训练的模型。在每一次迭代中，随机源图像和对抗性示例混合在一起来更新底层模型。结果表明，所提出的微调策略有效地提高了压缩模型的鲁棒性。总体而言，我们的方法简单、有效、通用性强，对于开发健壮的学习图像压缩解决方案很有吸引力。所有材料都已在https://njuvision.github.io/RobustNIC上公开访问，以进行可重现的研究。



## **17. Model Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的模型窃取攻击 cs.CR

To Appear in the 43rd IEEE Symposium on Security and Privacy, May  22-26, 2022

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.08331v1)

**Authors**: Yun Shen, Xinlei He, Yufei Han, Yang Zhang

**Abstracts**: Many real-world data come in the form of graphs. Graph neural networks (GNNs), a new family of machine learning (ML) models, have been proposed to fully leverage graph data to build powerful applications. In particular, the inductive GNNs, which can generalize to unseen data, become mainstream in this direction. Machine learning models have shown great potential in various tasks and have been deployed in many real-world scenarios. To train a good model, a large amount of data as well as computational resources are needed, leading to valuable intellectual property. Previous research has shown that ML models are prone to model stealing attacks, which aim to steal the functionality of the target models. However, most of them focus on the models trained with images and texts. On the other hand, little attention has been paid to models trained with graph data, i.e., GNNs. In this paper, we fill the gap by proposing the first model stealing attacks against inductive GNNs. We systematically define the threat model and propose six attacks based on the adversary's background knowledge and the responses of the target models. Our evaluation on six benchmark datasets shows that the proposed model stealing attacks against GNNs achieve promising performance.

摘要: 许多现实世界的数据都是以图表的形式出现的。图神经网络(GNNs)是一类新的机器学习(ML)模型，被提出用来充分利用图数据来构建功能强大的应用程序。特别是，可以推广到不可见数据的感应式GNN成为这一方向的主流。机器学习模型已经在各种任务中显示出巨大的潜力，并已被部署在许多现实场景中。要训练一个好的模型，需要大量的数据和计算资源，从而产生宝贵的知识产权。以往的研究表明，ML模型容易受到模型窃取攻击，目的是窃取目标模型的功能。然而，它们大多集中在用图像和文本训练的模型上。另一方面，很少有人关注用图形数据训练的模型，即GNN。在本文中，我们提出了第一个针对感应性GNNs的窃取攻击模型，填补了这一空白。我们系统地定义了威胁模型，并根据对手的背景知识和目标模型的响应提出了六种攻击。我们在六个基准数据集上的评估表明，所提出的针对GNNs的窃取攻击模型取得了令人满意的性能。



## **18. Meta Adversarial Perturbations**

元对抗扰动 cs.LG

Published in AAAI 2022 Workshop

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2111.10291v2)

**Authors**: Chia-Hung Yuan, Pin-Yu Chen, Chia-Mu Yu

**Abstracts**: A plethora of attack methods have been proposed to generate adversarial examples, among which the iterative methods have been demonstrated the ability to find a strong attack. However, the computation of an adversarial perturbation for a new data point requires solving a time-consuming optimization problem from scratch. To generate a stronger attack, it normally requires updating a data point with more iterations. In this paper, we show the existence of a meta adversarial perturbation (MAP), a better initialization that causes natural images to be misclassified with high probability after being updated through only a one-step gradient ascent update, and propose an algorithm for computing such perturbations. We conduct extensive experiments, and the empirical results demonstrate that state-of-the-art deep neural networks are vulnerable to meta perturbations. We further show that these perturbations are not only image-agnostic, but also model-agnostic, as a single perturbation generalizes well across unseen data points and different neural network architectures.

摘要: 已经提出了大量的攻击方法来生成对抗性实例，其中迭代方法已被证明具有发现强攻击的能力。然而，计算新数据点的对抗性扰动需要从头开始解决耗时的优化问题。要生成更强的攻击，通常需要更新迭代次数更多的数据点。本文证明了元对抗扰动(MAP)的存在性，并提出了一种计算这种扰动的算法。MAP是一种较好的初始化方法，它只通过一步梯度上升更新就会导致自然图像在更新后被高概率地误分类。我们进行了大量的实验，实验结果表明，最新的深度神经网络容易受到元扰动的影响。我们进一步表明，这些扰动不仅是图像不可知的，而且也是模型不可知的，因为单个扰动很好地概括了不可见的数据点和不同的神经网络结构。



## **19. Temporal Shuffling for Defending Deep Action Recognition Models against Adversarial Attacks**

用于防御敌方攻击的深层动作识别模型的时间洗牌 cs.CV

**SubmitDate**: 2021-12-15    [paper-pdf](http://arxiv.org/pdf/2112.07921v1)

**Authors**: Jaehui Hwang, Huan Zhang, Jun-Ho Choi, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, video-based action recognition methods using convolutional neural networks (CNNs) achieve remarkable recognition performance. However, there is still lack of understanding about the generalization mechanism of action recognition models. In this paper, we suggest that action recognition models rely on the motion information less than expected, and thus they are robust to randomization of frame orders. Based on this observation, we develop a novel defense method using temporal shuffling of input videos against adversarial attacks for action recognition models. Another observation enabling our defense method is that adversarial perturbations on videos are sensitive to temporal destruction. To the best of our knowledge, this is the first attempt to design a defense method specific to video-based action recognition models.

摘要: 近年来，基于卷积神经网络(CNNs)的视频动作识别方法取得了显著的识别效果。然而，对动作识别模型的泛化机制还缺乏了解。本文提出动作识别模型对运动信息的依赖程度低于预期，因而对帧阶数的随机化具有较强的鲁棒性。基于这一观察结果，我们提出了一种新的行为识别模型的防御方法，该方法利用输入视频的时间洗牌来抵御敌意攻击。支持我们防御方法的另一个观察结果是，视频上的对抗性扰动对时间破坏很敏感。据我们所知，这是首次尝试设计专门针对基于视频的动作识别模型的防御方法。



## **20. Adversarial Examples for Extreme Multilabel Text Classification**

极端多标签文本分类的对抗性实例 cs.LG

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07512v1)

**Authors**: Mohammadreza Qaraei, Rohit Babbar

**Abstracts**: Extreme Multilabel Text Classification (XMTC) is a text classification problem in which, (i) the output space is extremely large, (ii) each data point may have multiple positive labels, and (iii) the data follows a strongly imbalanced distribution. With applications in recommendation systems and automatic tagging of web-scale documents, the research on XMTC has been focused on improving prediction accuracy and dealing with imbalanced data. However, the robustness of deep learning based XMTC models against adversarial examples has been largely underexplored.   In this paper, we investigate the behaviour of XMTC models under adversarial attacks. To this end, first, we define adversarial attacks in multilabel text classification problems. We categorize attacking multilabel text classifiers as (a) positive-targeted, where the target positive label should fall out of top-k predicted labels, and (b) negative-targeted, where the target negative label should be among the top-k predicted labels. Then, by experiments on APLC-XLNet and AttentionXML, we show that XMTC models are highly vulnerable to positive-targeted attacks but more robust to negative-targeted ones. Furthermore, our experiments show that the success rate of positive-targeted adversarial attacks has an imbalanced distribution. More precisely, tail classes are highly vulnerable to adversarial attacks for which an attacker can generate adversarial samples with high similarity to the actual data-points. To overcome this problem, we explore the effect of rebalanced loss functions in XMTC where not only do they increase accuracy on tail classes, but they also improve the robustness of these classes against adversarial attacks. The code for our experiments is available at https://github.com/xmc-aalto/adv-xmtc

摘要: 极端多标签文本分类(XMTC)是一个文本分类问题，其中(I)输出空间非常大，(Ii)每个数据点可能有多个正标签，(Iii)数据服从强不平衡分布。随着XMTC在推荐系统和Web文档自动标注中的应用，XMTC的研究重点放在提高预测精度和处理不平衡数据上。然而，基于深度学习的XMTC模型对敌意示例的稳健性研究还很少。本文研究了XMTC模型在对抗性攻击下的行为。为此，我们首先定义了多标签文本分类问题中的对抗性攻击。我们将攻击多标签文本分类器分为(A)正向目标，其中目标正向标签应该落在前k个预测标签之外；(B)负向目标，其中目标负向标签应该在前k个预测标签中。然后，通过在APLC-XLNet和AttentionXML上的实验表明，XMTC模型对正目标攻击具有很强的脆弱性，但对负目标攻击具有较强的鲁棒性。此外，我们的实验表明，正向对抗性攻击的成功率分布不均衡。更准确地说，Tail类非常容易受到敌意攻击，对于这种攻击，攻击者可以生成与实际数据点高度相似的对抗性样本。为了克服这个问题，我们探索了XMTC中重新平衡损失函数的效果，在XMTC中，它们不仅提高了尾类的准确性，而且还提高了这些类对对手攻击的鲁棒性。我们实验的代码可以在https://github.com/xmc-aalto/adv-xmtc上找到



## **21. Multi-Leader Congestion Games with an Adversary**

有对手的多队长拥堵对策 cs.GT

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07435v1)

**Authors**: Tobias Harks, Mona Henle, Max Klimm, Jannik Matuschke, Anja Schedel

**Abstracts**: We study a multi-leader single-follower congestion game where multiple users (leaders) choose one resource out of a set of resources and, after observing the realized loads, an adversary (single-follower) attacks the resources with maximum loads, causing additional costs for the leaders. For the resulting strategic game among the leaders, we show that pure Nash equilibria may fail to exist and therefore, we consider approximate equilibria instead. As our first main result, we show that the existence of a $K$-approximate equilibrium can always be guaranteed, where $K \approx 1.1974$ is the unique solution of a cubic polynomial equation. To this end, we give a polynomial time combinatorial algorithm which computes a $K$-approximate equilibrium. The factor $K$ is tight, meaning that there is an instance that does not admit an $\alpha$-approximate equilibrium for any $\alpha<K$. Thus $\alpha=K$ is the smallest possible value of $\alpha$ such that the existence of an $\alpha$-approximate equilibrium can be guaranteed for any instance of the considered game. Secondly, we focus on approximate equilibria of a given fixed instance. We show how to compute efficiently a best approximate equilibrium, that is, with smallest possible $\alpha$ among all $\alpha$-approximate equilibria of the given instance.

摘要: 研究了一个多领导者单跟随者拥塞博弈，其中多个用户(领导者)从一组资源中选择一个资源，在观察到实现的负载后，一个对手(单一跟随者)攻击具有最大负载的资源，从而给领导者带来额外的成本。对于由此产生的领导者之间的战略博弈，我们表明纯纳什均衡可能不存在，因此，我们考虑近似均衡。作为我们的第一个主要结果，我们证明了$K$-近似均衡的存在性总是可以保证的，其中$K\约1.1974$是一个三次多项式方程的唯一解。为此，我们给出了一个计算$K$-近似均衡的多项式时间组合算法。因子$K$是紧的，这意味着对于任何$\α<K$，都存在一个不允许$\α$-近似均衡的实例。因此，$\α=K$是$\α$的最小可能值，使得对于所考虑的博弈的任何实例，都可以保证存在$\α$-近似均衡。其次，我们重点研究了给定固定实例的近似均衡。我们展示了如何有效地计算最佳近似均衡，即在给定实例的所有$\α$-近似均衡中，具有最小可能的$\α$。



## **22. Robustifying automatic speech recognition by extracting slowly varying features**

通过提取缓慢变化的特征来实现自动语音识别的ROBUST化 eess.AS

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07400v1)

**Authors**: Matias Pizarro, Dorothea Kolossa, Asja Fischer

**Abstracts**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.

摘要: 在过去的几年里，通过对抗性的例子表明，深度学习系统在受到攻击时是非常脆弱的。基于神经网络的自动语音识别(ASR)系统也不例外。有针对性和无针对性的攻击可以修改音频输入信号，使人类仍能识别相同的单词，而ASR系统则被引导预测不同的转录。本文提出了一种针对目标攻击的防御机制，即在将输入输入到自动语音识别系统之前，通过慢速特征分析、低通过滤或两者结合的方法，从音频信号中去除快速变化的特征。我们对以这种方式预处理的数据训练的混合ASR模型进行了实证分析。虽然最终得到的模型在良性数据上表现得相当好，但它们对目标攻击的鲁棒性要强得多：我们最终提出的模型在干净数据上的性能与基准模型相似，但健壮性要高出四倍多。



## **23. On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training**

论对抗性训练中硬性对抗性实例对过度适应的影响 cs.LG

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07324v1)

**Authors**: Chen Liu, Zhichao Huang, Mathieu Salzmann, Tong Zhang, Sabine Süsstrunk

**Abstracts**: Adversarial training is a popular method to robustify models against adversarial attacks. However, it exhibits much more severe overfitting than training on clean inputs. In this work, we investigate this phenomenon from the perspective of training instances, i.e., training input-target pairs. Based on a quantitative metric measuring instances' difficulty, we analyze the model's behavior on training instances of different difficulty levels. This lets us show that the decay in generalization performance of adversarial training is a result of the model's attempt to fit hard adversarial instances. We theoretically verify our observations for both linear and general nonlinear models, proving that models trained on hard instances have worse generalization performance than ones trained on easy instances. Furthermore, we prove that the difference in the generalization gap between models trained by instances of different difficulty levels increases with the size of the adversarial budget. Finally, we conduct case studies on methods mitigating adversarial overfitting in several scenarios. Our analysis shows that methods successfully mitigating adversarial overfitting all avoid fitting hard adversarial instances, while ones fitting hard adversarial instances do not achieve true robustness.

摘要: 对抗性训练是一种流行的增强模型抵御对抗性攻击的方法。然而，它表现出比清洁投入培训更严重的过度适应。在本工作中，我们从训练实例(即训练输入-目标对)的角度来研究这一现象。基于一种度量实例难度的量化度量，分析了该模型在不同难度级别的训练实例上的行为。这让我们看到，对抗性训练泛化性能的下降是模型试图拟合硬对抗性实例的结果。我们从理论上验证了我们对线性和一般非线性模型的观察结果，证明了在硬实例上训练的模型比在简单实例上训练的模型具有更差的泛化性能。此外，我们还证明了由不同难度水平的实例训练的模型之间的泛化差距的差异随着对抗预算的大小而增大。最后，我们对几种场景下缓解对抗性过拟合的方法进行了案例研究。我们的分析表明，成功缓解对抗性过拟合的方法都避免了拟合硬对抗性实例，而适合硬对抗性实例的方法并不能达到真正的鲁棒性。



## **24. Improving Calibration through the Relationship with Adversarial Robustness**

通过与对抗性稳健性的关系改进校准 cs.LG

Published at NeurIPS-2021

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2006.16375v2)

**Authors**: Yao Qin, Xuezhi Wang, Alex Beutel, Ed H. Chi

**Abstracts**: Neural networks lack adversarial robustness, i.e., they are vulnerable to adversarial examples that through small perturbations to inputs cause incorrect predictions. Further, trust is undermined when models give miscalibrated predictions, i.e., the predicted probability is not a good indicator of how much we should trust our model. In this paper, we study the connection between adversarial robustness and calibration and find that the inputs for which the model is sensitive to small perturbations (are easily attacked) are more likely to have poorly calibrated predictions. Based on this insight, we examine if calibration can be improved by addressing those adversarially unrobust inputs. To this end, we propose Adversarial Robustness based Adaptive Label Smoothing (AR-AdaLS) that integrates the correlations of adversarial robustness and calibration into training by adaptively softening labels for an example based on how easily it can be attacked by an adversary. We find that our method, taking the adversarial robustness of the in-distribution data into consideration, leads to better calibration over the model even under distributional shifts. In addition, AR-AdaLS can also be applied to an ensemble model to further improve model calibration.

摘要: 神经网络缺乏对抗性，也就是说，它们很容易受到对抗性示例的影响，这些示例通过对输入的微小扰动而导致不正确的预测。此外，当模型给出错误校准的预测时，信任被破坏，即，预测的概率不是我们应该信任我们的模型的良好指示器。在本文中，我们研究了敌方鲁棒性和校准之间的关系，发现模型对小扰动敏感(容易受到攻击)的输入更有可能具有较差的校准预测。基于这一见解，我们检查是否可以通过解决这些相反的不健壮的输入来改善校准。为此，我们提出了基于对手健壮性的自适应标签平滑(AR-ADALS)，它将对手健壮性和校准的相关性整合到训练中，根据对手攻击的容易程度对标签进行自适应软化。我们发现，我们的方法考虑了分布内数据的对抗性鲁棒性，即使在分布偏移的情况下也能对模型进行更好的校准。此外，AR-ADALS还可以应用于系综模型，以进一步改进模型校准。



## **25. Defending Against Multiple and Unforeseen Adversarial Videos**

防御多个不可预见的对抗性视频 cs.LG

Accepted in IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2009.05244v3)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstracts**: Adversarial robustness of deep neural networks has been actively investigated. However, most existing defense approaches are limited to a specific type of adversarial perturbations. Specifically, they often fail to offer resistance to multiple attack types simultaneously, i.e., they lack multi-perturbation robustness. Furthermore, compared to image recognition problems, the adversarial robustness of video recognition models is relatively unexplored. While several studies have proposed how to generate adversarial videos, only a handful of approaches about defense strategies have been published in the literature. In this paper, we propose one of the first defense strategies against multiple types of adversarial videos for video recognition. The proposed method, referred to as MultiBN, performs adversarial training on multiple adversarial video types using multiple independent batch normalization (BN) layers with a learning-based BN selection module. With a multiple BN structure, each BN brach is responsible for learning the distribution of a single perturbation type and thus provides more precise distribution estimations. This mechanism benefits dealing with multiple perturbation types. The BN selection module detects the attack type of an input video and sends it to the corresponding BN branch, making MultiBN fully automatic and allowing end-to-end training. Compared to present adversarial training approaches, the proposed MultiBN exhibits stronger multi-perturbation robustness against different and even unforeseen adversarial video types, ranging from Lp-bounded attacks and physically realizable attacks. This holds true on different datasets and target models. Moreover, we conduct an extensive analysis to study the properties of the multiple BN structure.

摘要: 深度神经网络的对抗性鲁棒性已经得到了积极的研究。然而，大多数现有的防御方法仅限于特定类型的对抗性扰动。具体地说，它们往往不能同时抵抗多种攻击类型，即缺乏多扰动鲁棒性。此外，与图像识别问题相比，视频识别模型的对抗性鲁棒性相对较少。虽然已经有几项研究提出了如何生成对抗性视频，但文献中只发表了几种关于防御策略的方法。在本文中，我们提出了针对多种类型的对抗性视频的视频识别的首批防御策略之一。该方法称为MultiBN，使用多个独立的批归一化(BN)层和基于学习的BN选择模块对多种对抗性视频类型进行对抗性训练。对于多BN结构，每个BN分支负责学习单个扰动类型的分布，从而提供更精确的分布估计。这种机制有利于处理多种扰动类型。BN选择模块检测输入视频的攻击类型，并将其发送到相应的BN分支，使MultiBN完全自动化，并允许端到端的训练。与现有的对抗性训练方法相比，所提出的多重BN对不同甚至不可预见的对抗性视频类型具有更强的多扰动鲁棒性，包括Lp有界攻击和物理可实现攻击。这适用于不同的数据集和目标模型。此外，我们还对多重BN结构的性质进行了广泛的分析。



## **26. MuxLink: Circumventing Learning-Resilient MUX-Locking Using Graph Neural Network-based Link Prediction**

MuxLink：基于图神经网络的链路预测规避学习弹性MUX锁定 cs.CR

Will be published in Proc. Design, Automation and Test in Europe  (DATE) 2022

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07178v1)

**Authors**: Lilas Alrahis, Satwik Patnaik, Muhammad Shafique, Ozgur Sinanoglu

**Abstracts**: Logic locking has received considerable interest as a prominent technique for protecting the design intellectual property from untrusted entities, especially the foundry. Recently, machine learning (ML)-based attacks have questioned the security guarantees of logic locking, and have demonstrated considerable success in deciphering the secret key without relying on an oracle, hence, proving to be very useful for an adversary in the fab. Such ML-based attacks have triggered the development of learning-resilient locking techniques. The most advanced state-of-the-art deceptive MUX-based locking (D-MUX) and the symmetric MUX-based locking techniques have recently demonstrated resilience against existing ML-based attacks. Both defense techniques obfuscate the design by inserting key-controlled MUX logic, ensuring that all the secret inputs to the MUXes are equiprobable.   In this work, we show that these techniques primarily introduce local and limited changes to the circuit without altering the global structure of the design. By leveraging this observation, we propose a novel graph neural network (GNN)-based link prediction attack, MuxLink, that successfully breaks both the D-MUX and symmetric MUX-locking techniques, relying only on the underlying structure of the locked design, i.e., in an oracle-less setting. Our trained GNN model learns the structure of the given circuit and the composition of gates around the non-obfuscated wires, thereby generating meaningful link embeddings that help decipher the secret inputs to the MUXes. The proposed MuxLink achieves key prediction accuracy and precision up to 100% on D-MUX and symmetric MUX-locked ISCAS-85 and ITC-99 benchmarks, fully unlocking the designs. We open-source MuxLink [1].

摘要: 逻辑锁定作为一种重要的保护设计知识产权免受不可信任实体，尤其是铸造企业的技术，已经引起了人们的极大兴趣。最近，基于机器学习(ML)的攻击对逻辑锁定的安全保证提出了质疑，并在不依赖先知的情况下成功地解密了密钥，因此，事实证明，这对制造厂的对手非常有用。这类基于ML的攻击引发了具有学习弹性的锁定技术的发展。最先进的欺骗性基于MUX的锁定(D-MUX)和基于对称MUX的锁定技术最近显示出对现有的基于ML的攻击的弹性。这两种防御技术都通过插入密钥控制的多路复用器逻辑来混淆设计，确保多路复用器的所有秘密输入都是等概率的。在这项工作中，我们表明，这些技术主要是在不改变设计的全局结构的情况下，对电路进行局部和有限的改变。通过利用这一观察结果，我们提出了一种新的基于图神经网络(GNN)的链接预测攻击MuxLink，该攻击仅依赖于锁定设计的底层结构，即在无Oracle的环境下，成功地打破了D-MUX和对称MUX锁定技术。我们训练的GNN模型学习给定电路的结构和非混淆导线周围的门的组成，从而生成有意义的链路嵌入，帮助破译MUX的秘密输入。建议的MuxLink在D-MUX和对称MUX锁定的ISCAS-85和ITC-99基准上实现了关键预测准确性和精度高达100%，完全解锁了设计。我们开源MuxLink[1]。



## **27. CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes**

CMUA-水印：一种抗深度伪装的跨模型通用对抗水印 cs.CV

9 pages, 7 figures, Thirty-Sixth AAAI Conference on Artificial  Intelligence, AAAI22

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2105.10872v2)

**Authors**: Hao Huang, Yongtao Wang, Zhaoyu Chen, Yuze Zhang, Yuheng Li, Zhi Tang, Wei Chu, Jingdong Chen, Weisi Lin, Kai-Kuang Ma

**Abstracts**: Malicious applications of deepfakes (i.e., technologies generating target facial attributes or entire faces from facial images) have posed a huge threat to individuals' reputation and security. To mitigate these threats, recent studies have proposed adversarial watermarks to combat deepfake models, leading them to generate distorted outputs. Despite achieving impressive results, these adversarial watermarks have low image-level and model-level transferability, meaning that they can protect only one facial image from one specific deepfake model. To address these issues, we propose a novel solution that can generate a Cross-Model Universal Adversarial Watermark (CMUA-Watermark), protecting a large number of facial images from multiple deepfake models. Specifically, we begin by proposing a cross-model universal attack pipeline that attacks multiple deepfake models iteratively. Then, we design a two-level perturbation fusion strategy to alleviate the conflict between the adversarial watermarks generated by different facial images and models. Moreover, we address the key problem in cross-model optimization with a heuristic approach to automatically find the suitable attack step sizes for different models, further weakening the model-level conflict. Finally, we introduce a more reasonable and comprehensive evaluation method to fully test the proposed method and compare it with existing ones. Extensive experimental results demonstrate that the proposed CMUA-Watermark can effectively distort the fake facial images generated by multiple deepfake models while achieving a better performance than existing methods.

摘要: 深度假冒(即从面部图像生成目标面部属性或整个面部的技术)的恶意应用已经对个人的声誉和安全构成了巨大的威胁。为了缓解这些威胁，最近的研究提出了对抗水印来对抗深度伪模型，导致它们产生失真的输出。尽管取得了令人印象深刻的结果，但这些对抗性水印的图像级和模型级可转移性较低，这意味着它们只能保护一幅面部图像免受一个特定的深伪模型的攻击。为了解决这些问题，我们提出了一种新的解决方案，它可以生成一个跨模型通用对抗水印(CMUA-Watermark)，保护大量的人脸图像免受多个深伪模型的攻击。具体地说，我们首先提出了一种跨模型的通用攻击流水线，迭代地攻击多个深度伪模型。然后，设计了一种两级扰动融合策略来缓解不同人脸图像和模型产生的对抗性水印之间的冲突。此外，我们采用启发式方法解决了跨模型优化中的关键问题，自动为不同模型找到合适的攻击步长，进一步弱化了模型间的冲突。最后，我们引入了一种更合理、更全面的评价方法，对所提出的方法进行了充分的测试，并与现有的评价方法进行了比较。大量的实验结果表明，提出的CMUA-水印能够有效地对多个深度伪模型生成的假人脸图像进行失真，同时取得了比现有方法更好的性能。



## **28. Real-Time Neural Voice Camouflage**

实时神经语音伪装 cs.SD

14 pages

**SubmitDate**: 2021-12-14    [paper-pdf](http://arxiv.org/pdf/2112.07076v1)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 4.17x more than baselines as measured through word error rate, and 7.27x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.

摘要: 自动语音识别系统为应用创造了令人兴奋的可能性，然而它们也为系统窃听提供了机会。我们提出了一种方法，从这些系统中伪装出人的空中语音，而不会给房间里的人之间的对话带来不便。标准对抗性攻击在实时流情况下无效，因为在执行攻击时信号的特性将发生变化。我们引入预测性攻击，通过预测未来最有效的攻击来实现实时性能。在实时约束条件下，我们的方法对已建立的语音识别系统DeepSpeech的拥塞程度分别是基线的4.17倍(词错误率)和7.27倍(字符错误率)。我们进一步证明了我们的方法在物理距离上的现实环境中是实际有效的。



## **29. On the Privacy Risks of Deploying Recurrent Neural Networks in Machine Learning**

递归神经网络在机器学习中的隐私风险研究 cs.CR

Under Double-Blind Review

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2110.03054v2)

**Authors**: Yunhao Yang, Parham Gohari, Ufuk Topcu

**Abstracts**: We study the privacy implications of deploying recurrent neural networks (RNNs) in machine learning models. We focus on a class of privacy threats, called membership inference attacks (MIAs), which aim to infer whether or not specific data records have been used to train a model. Considering three machine learning applications, namely, machine translation, deep reinforcement learning, and image classification, we provide empirical evidence that RNNs are more vulnerable to MIAs than the alternative feed-forward architectures. We then study differential privacy methods to protect the privacy of the training dataset of RNNs. These methods are known to provide rigorous privacy guarantees irrespective of the adversary's model. We develop an alternative differential privacy mechanism to the so-called DP-FedAvg algorithm, which instead of obfuscating gradients during training, obfuscates the model's output. Unlike the existing work, the mechanism allows for post-training adjustment of the privacy parameters without having to retrain the model. We provide numerical results suggesting that the mechanism provides a strong shield against MIAs while trading off marginal utility.

摘要: 我们研究了在机器学习模型中部署递归神经网络(RNNs)对隐私的影响。我们关注一类隐私威胁，称为成员关系推断攻击(MIA)，其目的是推断特定的数据记录是否已被用于训练模型。考虑到三种机器学习应用，即机器翻译、深度强化学习和图像分类，我们提供了经验证据，表明RNN比其他前馈结构更容易受到MIA的影响。在此基础上，研究了RNN训练数据集隐私保护的差分隐私保护方法。众所周知，无论对手的模型如何，这些方法都能提供严格的隐私保证。我们提出了一种替代DP-FedAvg算法的差分隐私机制，它不是在训练过程中混淆梯度，而是混淆模型的输出。与现有工作不同的是，该机制允许在训练后调整隐私参数，而不必重新训练模型。我们提供的数值结果表明，该机制在牺牲边际效用的同时，提供了对MIA的强大屏蔽。



## **30. Signal Injection Attacks against CCD Image Sensors**

针对CCD图像传感器的信号注入攻击 cs.CR

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2108.08881v2)

**Authors**: Sebastian Köhler, Richard Baker, Ivan Martinovic

**Abstracts**: Since cameras have become a crucial part in many safety-critical systems and applications, such as autonomous vehicles and surveillance, a large body of academic and non-academic work has shown attacks against their main component - the image sensor. However, these attacks are limited to coarse-grained and often suspicious injections because light is used as an attack vector. Furthermore, due to the nature of optical attacks, they require the line-of-sight between the adversary and the target camera.   In this paper, we present a novel post-transducer signal injection attack against CCD image sensors, as they are used in professional, scientific, and even military settings. We show how electromagnetic emanation can be used to manipulate the image information captured by a CCD image sensor with the granularity down to the brightness of individual pixels. We study the feasibility of our attack and then demonstrate its effects in the scenario of automatic barcode scanning. Our results indicate that the injected distortion can disrupt automated vision-based intelligent systems.

摘要: 由于摄像头已经成为许多安全关键系统和应用的关键组成部分，如自动驾驶汽车和监控，大量的学术和非学术研究表明，摄像头的主要组件-图像传感器-受到了攻击。然而，这些攻击仅限于粗粒度且通常可疑的注入，因为光被用作攻击载体。此外，由于光学攻击的性质，它们需要对手和目标摄像机之间的视线。在本文中，我们提出了一种针对CCD图像传感器的新型换能器后信号注入攻击，因为它们被用于专业、科学甚至军事环境中。我们展示了如何使用电磁辐射来处理CCD图像传感器捕获的图像信息，其粒度可以细化到单个像素的亮度。我们研究了该攻击的可行性，并在条码自动扫描场景中演示了该攻击的效果。我们的结果表明，注入的失真会扰乱基于视觉的自动化智能系统。



## **31. Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training**

安全胜过遗憾：通过对抗性训练防止妄想对手 cs.LG

NeurIPS 2021

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2102.04716v4)

**Authors**: Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.

摘要: 妄想攻击的目的是通过对正确标记的训练样本的特征进行轻微扰动来显著降低学习模型的测试精度。通过将这种恶意攻击形式化为在特定的$\infty$-Wasserstein球中寻找最坏情况的训练数据，我们表明最小化扰动数据上的敌意风险等价于优化原始数据上的自然风险上界。这意味着对抗性训练可以作为对抗妄想攻击的原则性防御。因此，通过对抗性训练可以在很大程度上恢复由于妄想攻击而降低的测试精度。为了进一步了解防御的内在机制，我们揭示了对抗性训练可以通过防止学习者在自然环境中过度依赖非鲁棒特征来抵抗妄想干扰。最后，我们通过在流行的基准数据集上的一组实验来补充我们的理论发现，这些实验表明该防御系统可以抵御六种不同的实际攻击。在面对妄想性对手时，无论是理论结果还是经验结果都支持对抗性训练。



## **32. A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks**

数据迟钝和数据感知中毒攻击的分离结果 cs.LG

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2003.12020v3)

**Authors**: Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta

**Abstracts**: Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Some of the stronger poisoning attacks require the full knowledge of the training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.   In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that full-information adversaries (that craft poisoning examples based on the rest of the training data) are provably stronger than the optimal attacker that is oblivious to the training set yet has access to the distribution of the data. Our separation result shows that the two setting of data-aware and data-oblivious are fundamentally different and we cannot hope to always achieve the same attack or defense results in these scenarios.

摘要: 中毒攻击已经成为机器学习算法的重大安全威胁。已经证明，对训练集进行微小更改的对手，例如添加巧尽心思构建的数据点，可能会损害输出模型的性能。一些更强的中毒攻击需要完全了解训练数据。这使得使用不完全了解干净训练集的中毒攻击获得相同的攻击结果的可能性仍然存在。在这项工作中，我们开始了对上述问题的理论研究。具体地说，对于使用套索进行特征选择的情况，我们证明了全信息对手(基于训练数据的睡觉来制作中毒实例)比最优攻击者(即对训练集是迟钝但可以访问数据分布的攻击者)更强大。我们的分离结果表明，数据感知和数据迟钝这两个设置是根本不同的，我们不能指望在这些场景下总是能达到相同的攻防效果。



## **33. Learning Classical Readout Quantum PUFs based on single-qubit gates**

基于单量子位门的经典读出量子PUF的学习 quant-ph

11 pages, 9 figures

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06661v1)

**Authors**: Anna Pappa, Niklas Pirnay, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.

摘要: 已经提出了物理不可克隆功能(PUF)作为识别和认证电子设备的方式。最近，已经有几个想法被提出，目的是在量子设备上实现同样的目标。其中一些结构采用单量子比特门，以便提供量子设备的安全指纹。在这项工作中，我们使用统计查询(SQ)模型形式化了经典读出量子PUF(CR-QPUF)类，并显式地证明了当敌手拥有对CR-QPUF的SQ访问权限时，基于单量子比特旋转门的CR-QPUF是不安全的。我们演示了恶意方如何学习CR-QPUF特征，并通过使用简单的低次多项式回归进行建模攻击来伪造量子设备的签名。所提出的建模攻击在真实的IBMQ量子机上的真实场景中被成功地实现。我们深入讨论了利用量子器件缺陷作为安全指纹的CR-QPUF的前景和存在的问题。



## **34. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

三角攻击：一种查询高效的基于决策的对抗性攻击 cs.CV

10 pages

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06569v1)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.

摘要: 基于决策的攻击将目标模型视为一个黑匣子，只访问硬预测标签，对现实世界的应用构成了严重威胁。最近已经做出了很大的努力来减少查询的数量；然而，现有的基于决策的攻击仍然需要数千个查询才能生成高质量的对抗性示例。在这项工作中，我们发现一个良性样本、当前和下一个对抗性样本可以自然地在子空间中为任何迭代攻击构造一个三角形。基于正弦定律，提出了一种新的三角形攻击算法(TA)，利用任意三角形中长边总是与大角相对的几何信息来优化摄动。然而，直接在输入图像上应用这样的信息是无效的，因为它不能在高维空间中彻底探索输入样本的邻域。为了解决这一问题，TA优化了低频空间中的扰动，以达到有效降维的目的，这是因为这种几何特性具有普遍性。对ImageNet数据集的广泛评估表明，与现有的基于决策的攻击相比，TA在1000个查询中实现了更高的攻击成功率，并且在各种扰动预算下需要更少的查询就可以达到相同的攻击成功率。如此高的效率，进一步证明了TA在真实接口，即腾讯云接口上的适用性。



## **35. Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning**

回到绘图板：对产生式联合学习的毒害攻击的批判性评估 cs.LG

To appear in the IEEE Symposium on Security & Privacy (Oakland), 2022

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2108.10241v2)

**Authors**: Virat Shejwalkar, Amir Houmansadr, Peter Kairouz, Daniel Ramage

**Abstracts**: While recent works have indicated that federated learning (FL) may be vulnerable to poisoning attacks by compromised clients, their real impact on production FL systems is not fully understood. In this work, we aim to develop a comprehensive systemization for poisoning attacks on FL by enumerating all possible threat models, variations of poisoning, and adversary capabilities. We specifically put our focus on untargeted poisoning attacks, as we argue that they are significantly relevant to production FL deployments.   We present a critical analysis of untargeted poisoning attacks under practical, production FL environments by carefully characterizing the set of realistic threat models and adversarial capabilities. Our findings are rather surprising: contrary to the established belief, we show that FL is highly robust in practice even when using simple, low-cost defenses. We go even further and propose novel, state-of-the-art data and model poisoning attacks, and show via an extensive set of experiments across three benchmark datasets how (in)effective poisoning attacks are in the presence of simple defense mechanisms. We aim to correct previous misconceptions and offer concrete guidelines to conduct more accurate (and more realistic) research on this topic.

摘要: 虽然最近的研究表明，联合学习(FL)可能容易受到受危害客户的中毒攻击，但它们对产生式FL系统的真正影响还没有完全了解。在这项工作中，我们的目标是通过列举所有可能的威胁模型、中毒的变体和对手的能力来开发一个全面的系统来毒化对FL的攻击。我们特别将重点放在无针对性的中毒攻击上，因为我们认为它们与生产FL部署密切相关。我们通过仔细描述一组现实的威胁模型和对抗能力，对实际生产FL环境下的非目标中毒攻击进行了批判性分析。我们的发现相当令人惊讶：与既定的信念相反，我们表明，即使使用简单、低成本的防御措施，FL在实践中也是高度健壮的。我们甚至更进一步，提出了新颖的、最先进的数据和模型中毒攻击，并通过对三个基准数据集的广泛实验表明，在存在简单的防御机制的情况下，中毒攻击是如何(无效)有效的。我们的目标是纠正之前的误解，并提供具体的指导方针，以便在这个主题上进行更准确(和更现实)的研究。



## **36. Detecting Audio Adversarial Examples with Logit Noising**

利用Logit噪声检测音频对抗性实例 cs.CR

10 pages, 12 figures, In Proceedings of the 37th Annual Computer  Security Applications Conference (ACSAC) 2021

**SubmitDate**: 2021-12-13    [paper-pdf](http://arxiv.org/pdf/2112.06443v1)

**Authors**: Namgyu Park, Sangwoo Ji, Jong Kim

**Abstracts**: Automatic speech recognition (ASR) systems are vulnerable to audio adversarial examples that attempt to deceive ASR systems by adding perturbations to benign speech signals. Although an adversarial example and the original benign wave are indistinguishable to humans, the former is transcribed as a malicious target sentence by ASR systems. Several methods have been proposed to generate audio adversarial examples and feed them directly into the ASR system (over-line). Furthermore, many researchers have demonstrated the feasibility of robust physical audio adversarial examples(over-air). To defend against the attacks, several studies have been proposed. However, deploying them in a real-world situation is difficult because of accuracy drop or time overhead. In this paper, we propose a novel method to detect audio adversarial examples by adding noise to the logits before feeding them into the decoder of the ASR. We show that carefully selected noise can significantly impact the transcription results of the audio adversarial examples, whereas it has minimal impact on the transcription results of benign audio waves. Based on this characteristic, we detect audio adversarial examples by comparing the transcription altered by logit noising with its original transcription. The proposed method can be easily applied to ASR systems without any structural changes or additional training. The experimental results show that the proposed method is robust to over-line audio adversarial examples as well as over-air audio adversarial examples compared with state-of-the-art detection methods.

摘要: 自动语音识别(ASR)系统容易受到试图通过向良性语音信号添加扰动来欺骗ASR系统的音频敌意示例的攻击。尽管人类无法区分对抗性的例子和原始的良性波，但前者被ASR系统转录为恶意目标句子。已经提出了几种方法来生成音频对抗性示例，并将其直接馈送到ASR系统(在线)。此外，许多研究人员已经证明了健壮的物理音频对抗性例子(空中)的可行性。为了防御这些攻击，已经提出了几项研究。然而，由于准确性下降或时间开销的原因，在实际情况下部署它们是困难的。在本文中，我们提出了一种新的方法，在将音频对抗性示例送入ASR的解码器之前，通过在LOGITS中添加噪声来检测音频对抗性示例。我们表明，精心选择的噪声可以显著影响音频对抗性示例的转录结果，而对良性音波的转录结果的影响很小。基于这一特点，我们通过将Logit噪声改变后的转录与其原始转录进行比较来检测音频恶意示例。该方法可以方便地应用到ASR系统中，而不需要任何结构改变或额外的训练。实验结果表明，与现有的检测方法相比，该方法对在线音频攻击示例和空中音频攻击示例均具有较好的鲁棒性。



## **37. BACKDOORL: Backdoor Attack against Competitive Reinforcement Learning**

背景：对好胜强化学习的后门攻击 cs.CR

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2105.00579v3)

**Authors**: Lun Wang, Zaynah Javed, Xian Wu, Wenbo Guo, Xinyu Xing, Dawn Song

**Abstracts**: Recent research has confirmed the feasibility of backdoor attacks in deep reinforcement learning (RL) systems. However, the existing attacks require the ability to arbitrarily modify an agent's observation, constraining the application scope to simple RL systems such as Atari games. In this paper, we migrate backdoor attacks to more complex RL systems involving multiple agents and explore the possibility of triggering the backdoor without directly manipulating the agent's observation. As a proof of concept, we demonstrate that an adversary agent can trigger the backdoor of the victim agent with its own action in two-player competitive RL systems. We prototype and evaluate BACKDOORL in four competitive environments. The results show that when the backdoor is activated, the winning rate of the victim drops by 17% to 37% compared to when not activated.

摘要: 最近的研究证实了深度强化学习(RL)系统中后门攻击的可行性。然而，现有的攻击需要能够任意修改代理的观察，将应用范围限制在简单的RL系统，如Atari游戏。在本文中，我们将后门攻击迁移到涉及多个Agent的更复杂的RL系统，并探索在不直接操作Agent的观察的情况下触发后门的可能性。作为概念证明，我们证明了在双人好胜RL系统中，敌对代理可以通过自己的操作触发受害者代理的后门。我们在四个好胜环境中对BACKDOORL进行了原型和评估。结果显示，后门开启时，受害者中签率较未开启时下降17%至37%。



## **38. Interpolated Joint Space Adversarial Training for Robust and Generalizable Defenses**

插值联合空间对抗训练的健壮性和泛化防御 cs.CV

Under submission

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2112.06323v1)

**Authors**: Chun Pong Lau, Jiang Liu, Hossein Souri, Wei-An Lin, Soheil Feizi, Rama Chellappa

**Abstracts**: Adversarial training (AT) is considered to be one of the most reliable defenses against adversarial attacks. However, models trained with AT sacrifice standard accuracy and do not generalize well to novel attacks. Recent works show generalization improvement with adversarial samples under novel threat models such as on-manifold threat model or neural perceptual threat model. However, the former requires exact manifold information while the latter requires algorithm relaxation. Motivated by these considerations, we exploit the underlying manifold information with Normalizing Flow, ensuring that exact manifold assumption holds. Moreover, we propose a novel threat model called Joint Space Threat Model (JSTM), which can serve as a special case of the neural perceptual threat model that does not require additional relaxation to craft the corresponding adversarial attacks. Under JSTM, we develop novel adversarial attacks and defenses. The mixup strategy improves the standard accuracy of neural networks but sacrifices robustness when combined with AT. To tackle this issue, we propose the Robust Mixup strategy in which we maximize the adversity of the interpolated images and gain robustness and prevent overfitting. Our experiments show that Interpolated Joint Space Adversarial Training (IJSAT) achieves good performance in standard accuracy, robustness, and generalization in CIFAR-10/100, OM-ImageNet, and CIFAR-10-C datasets. IJSAT is also flexible and can be used as a data augmentation method to improve standard accuracy and combine with many existing AT approaches to improve robustness.

摘要: 对抗性训练(AT)被认为是对抗对抗性攻击最可靠的防御手段之一。然而，使用AT训练的模型牺牲了标准的准确性，并且不能很好地推广到新的攻击。最近的工作表明，在新的威胁模型(如流形威胁模型或神经感知威胁模型)下，敌意样本的泛化性能有所提高。前者需要精确的流形信息，而后者需要算法松弛。基于这些考虑，我们用规格化的流来开发潜在的流形信息，以确保精确的流形假设成立。此外，我们还提出了一种新的威胁模型，称为联合空间威胁模型(JSTM)，它可以作为神经感知威胁模型的特例，不需要额外的松弛来构造相应的敌意攻击。在JSTM下，我们开发了新颖的对抗性攻击和防御。混合策略提高了神经网络的标准精度，但与AT结合时牺牲了鲁棒性。针对撞击的这个问题，我们提出了鲁棒混合策略，该策略最大限度地消除了插值图像的不利影响，获得了稳健性，并防止了过拟合。实验表明，插值联合空间对抗训练(IJSAT)在CIFAR-10/100、OM-ImageNet和CIFAR-10-C数据集上取得了良好的标准准确率、鲁棒性和泛化性能。IJSAT也是灵活的，可以作为一种数据增强方法来提高标准精度，并与许多现有的AT方法相结合来提高鲁棒性。



## **39. A Game-Theoretical Self-Adaptation Framework for Securing Software-Intensive Systems**

软件密集型系统安全的博弈论自适应框架 cs.SE

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2112.07588v1)

**Authors**: Mingyue Zhang, Nianyu Li, Sridhar Adepu, Eunsuk Kang, Zhi Jin

**Abstracts**: The increasing prevalence of security attacks on software-intensive systems calls for new, effective methods for detecting and responding to these attacks. As one promising approach, game theory provides analytical tools for modeling the interaction between the system and the adversarial environment and designing reliable defense. In this paper, we propose an approach for securing software-intensive systems using a rigorous game-theoretical framework. First, a self-adaptation framework is deployed on a component-based software intensive system, which periodically monitors the system for anomalous behaviors. A learning-based method is proposed to detect possible on-going attacks on the system components and predict potential threats to components. Then, an algorithm is designed to automatically build a \emph{Bayesian game} based on the system architecture (of which some components might have been compromised) once an attack is detected, in which the system components are modeled as independent players in the game. Finally, an optimal defensive policy is computed by solving the Bayesian game to achieve the best system utility, which amounts to minimizing the impact of the attack. We conduct two sets of experiments on two general benchmark tasks for security domain. Moreover, we systematically present a case study on a real-world water treatment testbed, i.e. the Secure Water Treatment System. Experiment results show the applicability and the effectiveness of our approach.

摘要: 针对软件密集型系统的安全攻击日益盛行，这就需要新的、有效的方法来检测和响应这些攻击。作为一种很有前途的方法，博弈论为建模系统与对手环境之间的交互作用和设计可靠的防御提供了分析工具。在本文中，我们提出了一种使用严格的博弈论框架来保护软件密集型系统的方法。首先，在基于组件的软件密集型系统上部署自适应框架，定期监视系统的异常行为。提出了一种基于学习的方法来检测对系统组件可能的持续攻击，并预测对组件的潜在威胁。然后，设计了一种算法，一旦检测到攻击，就根据系统架构(其中的一些组件可能已经被攻破)自动构建\emph{贝叶斯游戏}，其中系统组件被建模为游戏中的独立参与者。最后，通过求解贝叶斯博弈计算出最优的防御策略，使系统效用最优，从而使攻击的影响最小化。我们在两个通用的安全领域基准任务上进行了两组实验。此外，我们还系统地介绍了一个真实的水处理试验台--安全水处理系统的案例研究。实验结果表明了该方法的适用性和有效性。



## **40. FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack**

FCA：学习用于多视点物理对抗攻击的3D全覆盖车辆伪装 cs.CV

9 pages, 5 figures

**SubmitDate**: 2021-12-12    [paper-pdf](http://arxiv.org/pdf/2109.07193v3)

**Authors**: Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Xiaoya Zhang, Zhiqiang Gong, Wen Yao, Xiaoqian Chen

**Abstracts**: Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle's surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors. The code of FCA will be available at: https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.

摘要: 物理对抗性攻击在目标检测中受到越来越多的关注。然而，以往的工作大多集中在通过生成一个单独的对抗性面片来隐藏检测器，该面片只覆盖了车辆表面的平面部分，不能在多视角、远距离和部分遮挡的物理场景中攻击检测器。为了弥合数字攻击和物理攻击之间的差距，我们利用全3D车辆表面提出了一种健壮的全覆盖伪装攻击(FCA)来欺骗检测器。具体地说，我们首先尝试在整个车辆表面上渲染非平面伪装纹理。为了模拟真实世界的环境条件，我们引入了一个变换函数来将渲染的伪装车辆转换为照片逼真的场景。最后，我们设计了一个有效的损失函数来优化伪装纹理。实验表明，全覆盖伪装攻击不仅在各种测试用例下的性能优于最新的伪装攻击方法，而且可以推广到不同的环境、车辆和目标探测器。FCA的代码可在以下网址获得：https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.



## **41. A Note on the Post-Quantum Security of (Ring) Signatures**

关于(环)签名的后量子安全性的一个注记 quant-ph

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2112.06078v1)

**Authors**: Rohit Chatterjee, Kai-Min Chung, Xiao Liang, Giulio Malavolta

**Abstracts**: This work revisits the security of classical signatures and ring signatures in a quantum world. For (ordinary) signatures, we focus on the arguably preferable security notion of blind-unforgeability recently proposed by Alagic et al. (Eurocrypt'20). We present two short signature schemes achieving this notion: one is in the quantum random oracle model, assuming quantum hardness of SIS; and the other is in the plain model, assuming quantum hardness of LWE with super-polynomial modulus. Prior to this work, the only known blind-unforgeable schemes are Lamport's one-time signature and the Winternitz one-time signature, and both of them are in the quantum random oracle model.   For ring signatures, the recent work by Chatterjee et al. (Crypto'21) proposes a definition trying to capture adversaries with quantum access to the signer. However, it is unclear if their definition, when restricted to the classical world, is as strong as the standard security notion for ring signatures. They also present a construction that only partially achieves (even) this seeming weak definition, in the sense that the adversary can only conduct superposition attacks over the messages, but not the rings. We propose a new definition that does not suffer from the above issue. Our definition is an analog to the blind-unforgeability in the ring signature setting. Moreover, assuming the quantum hardness of LWE, we construct a compiler converting any blind-unforgeable (ordinary) signatures to a ring signature satisfying our definition.

摘要: 这项工作回顾了经典签名和环签名在量子世界中的安全性。对于(普通的)签名，我们关注最近由Alial等人提出的盲不可伪造性这一有争议的更好的安全概念。(Eurocrypt‘20)。我们提出了两个实现这一概念的短签名方案：一个是在量子随机预言模型中，假设SIS的量子硬度；另一个是在平面模型中，假设LWE的量子硬度为超多项式模数。在此之前，唯一已知的盲不可伪造方案是Lamport的一次性签名和Winteritz一次性签名，并且这两个方案都是量子随机预言模型。对于环签名，Chatterjee等人最近的工作。(Crypto‘21)提出了一种定义，试图通过量子访问签名者来捕获对手。然而，目前还不清楚它们的定义，当被限制在古典世界时，是否像标准的环签名安全概念一样强大。他们还提出了一种结构，该结构仅部分地实现(甚至)这一看似薄弱的定义，在某种意义上，对手只能对消息进行叠加攻击，而不能对环进行叠加攻击。我们提出了一个不受上述问题影响的新定义。我们的定义类似于环签名设置中的盲不可伪造性。此外，在假设LWE的量子硬度的情况下，我们构造了一个编译器，将任何盲不可伪造(普通)签名转换为满足我们定义的环签名。



## **42. MedAttacker: Exploring Black-Box Adversarial Attacks on Risk Prediction Models in Healthcare**

MedAttacker：探索医疗保健领域风险预测模型的黑箱对抗性攻击 cs.LG

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2112.06063v1)

**Authors**: Muchao Ye, Junyu Luo, Guanjie Zheng, Cao Xiao, Ting Wang, Fenglong Ma

**Abstracts**: Deep neural networks (DNNs) have been broadly adopted in health risk prediction to provide healthcare diagnoses and treatments. To evaluate their robustness, existing research conducts adversarial attacks in the white/gray-box setting where model parameters are accessible. However, a more realistic black-box adversarial attack is ignored even though most real-world models are trained with private data and released as black-box services on the cloud. To fill this gap, we propose the first black-box adversarial attack method against health risk prediction models named MedAttacker to investigate their vulnerability. MedAttacker addresses the challenges brought by EHR data via two steps: hierarchical position selection which selects the attacked positions in a reinforcement learning (RL) framework and substitute selection which identifies substitute with a score-based principle. Particularly, by considering the temporal context inside EHRs, it initializes its RL position selection policy by using the contribution score of each visit and the saliency score of each code, which can be well integrated with the deterministic substitute selection process decided by the score changes. In experiments, MedAttacker consistently achieves the highest average success rate and even outperforms a recent white-box EHR adversarial attack technique in certain cases when attacking three advanced health risk prediction models in the black-box setting across multiple real-world datasets. In addition, based on the experiment results we include a discussion on defending EHR adversarial attacks.

摘要: 深度神经网络(DNNs)已被广泛应用于健康风险预测，以提供医疗诊断和治疗。为了评估其鲁棒性，现有研究在模型参数可访问的白盒/灰盒设置下进行对抗性攻击。然而，更现实的黑盒对抗性攻击被忽略了，即使大多数现实世界的模型都是用私人数据训练的，并在云上作为黑盒服务发布。为了填补这一空白，我们提出了第一个针对健康风险预测模型的黑盒对抗性攻击方法MedAttacker来调查它们的脆弱性。MedAttacker通过两个步骤来应对电子病历数据带来的挑战：分层位置选择和替身选择，前者在强化学习(RL)框架中选择受攻击的位置，后者根据分数的原则识别替身。特别地，通过考虑EHR内部的时间上下文，利用每次访问的贡献度分数和每个代码的显著性分数来初始化其RL位置选择策略，该策略可以很好地与由分数变化决定的确定性替身选择过程相结合。在实验中，当在黑盒设置中攻击多个现实世界数据集中的三个高级健康风险预测模型时，MedAttacker始终获得最高的平均成功率，甚至在某些情况下性能优于最近的白盒EHR对抗性攻击技术。此外，基于实验结果，我们还讨论了如何防御EHR对抗性攻击。



## **43. Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting**

利用调整大小的多元输入、多元集成和区域拟合提高对抗性范例的可转移性 cs.CV

Accepted to ECCV2020

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2112.06011v1)

**Authors**: Junhua Zou, Zhisong Pan, Junyang Qiu, Xin Liu, Ting Rui, Wei Li

**Abstracts**: We introduce a three stage pipeline: resized-diverse-inputs (RDIM), diversity-ensemble (DEM) and region fitting, that work together to generate transferable adversarial examples. We first explore the internal relationship between existing attacks, and propose RDIM that is capable of exploiting this relationship. Then we propose DEM, the multi-scale version of RDIM, to generate multi-scale gradients. After the first two steps we transform value fitting into region fitting across iterations. RDIM and region fitting do not require extra running time and these three steps can be well integrated into other attacks. Our best attack fools six black-box defenses with a 93% success rate on average, which is higher than the state-of-the-art gradient-based attacks. Besides, we rethink existing attacks rather than simply stacking new methods on the old ones to get better performance. It is expected that our findings will serve as the beginning of exploring the internal relationship between attack methods. Codes are available at https://github.com/278287847/DEM.

摘要: 我们引入了三个阶段的流水线：调整大小的多样性输入(RDIM)、多样性集成(DEM)和区域拟合，它们共同作用来生成可转移的对抗性示例。我们首先探讨了现有攻击之间的内在关系，并提出了能够利用这种关系的RDIM。在此基础上，提出了RDIM的多尺度版本DEM来生成多尺度梯度。在前两步之后，我们将值拟合转换为跨迭代的区域拟合。RDIM和区域拟合不需要额外的运行时间，这三个步骤可以很好地集成到其他攻击中。我们最好的攻击愚弄了6个黑匣子防御，平均成功率为93%，高于最先进的基于梯度的攻击。此外，我们重新考虑了现有的攻击，而不是简单地将新方法叠加在旧方法上以获得更好的性能。预计我们的发现将作为探索攻击方法之间的内在联系的开始。有关代码，请访问https://github.com/278287847/DEM.。



## **44. Making Adversarial Examples More Transferable and Indistinguishable**

使对抗性例子更具可转移性和不可区分性 cs.CV

Accepted to AAAI2022

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2007.03838v2)

**Authors**: Junhua Zou, Yexin Duan, Boyu Li, Wu Zhang, Yu Pan, Zhisong Pan

**Abstracts**: Fast gradient sign attack series are popular methods that are used to generate adversarial examples. However, most of the approaches based on fast gradient sign attack series cannot balance the indistinguishability and transferability due to the limitations of the basic sign structure. To address this problem, we propose a method, called Adam Iterative Fast Gradient Tanh Method (AI-FGTM), to generate indistinguishable adversarial examples with high transferability. Besides, smaller kernels and dynamic step size are also applied to generate adversarial examples for further increasing the attack success rates. Extensive experiments on an ImageNet-compatible dataset show that our method generates more indistinguishable adversarial examples and achieves higher attack success rates without extra running time and resource. Our best transfer-based attack NI-TI-DI-AITM can fool six classic defense models with an average success rate of 89.3% and three advanced defense models with an average success rate of 82.7%, which are higher than the state-of-the-art gradient-based attacks. Additionally, our method can also reduce nearly 20% mean perturbation. We expect that our method will serve as a new baseline for generating adversarial examples with better transferability and indistinguishability.

摘要: 快速梯度符号攻击序列是生成对抗性实例的常用方法。然而，由于基本符号结构的限制，大多数基于快速梯度符号攻击序列的方法不能平衡不可区分性和可转移性。针对这一问题，我们提出了一种亚当迭代快速梯度Tanh方法(AI-FGTM)来生成不可区分的、可移植性高的对抗性实例。此外，还采用较小的核函数和动态步长生成对抗性实例，进一步提高了攻击成功率。在与ImageNet兼容的数据集上的大量实验表明，该方法在不增加运行时间和资源的情况下，生成了更多难以区分的敌意示例，并获得了更高的攻击成功率。我们最好的基于传输的攻击NI-TI-DI-AITM可以欺骗6个经典的防御模型，平均成功率为89.3%，可以欺骗3个高级的防御模型，平均成功率为82.7%，高于最先进的基于梯度的攻击。此外，我们的方法还可以减少近20%的平均摄动。我们期望我们的方法将作为生成具有更好的可移植性和不可区分性的对抗性示例的新基线。



## **45. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text; typos corrected

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2009.04131v5)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **46. Bad Characters: Imperceptible NLP Attacks**

坏人：潜移默化的NLP攻击 cs.CL

To appear in the 43rd IEEE Symposium on Security and Privacy.  Revisions: NER & sentiment analysis experiments, previous work comparison,  defense evaluation

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2106.09898v2)

**Authors**: Nicholas Boucher, Ilia Shumailov, Ross Anderson, Nicolas Papernot

**Abstracts**: Several years of research have shown that machine-learning systems are vulnerable to adversarial examples, both in theory and in practice. Until now, such attacks have primarily targeted visual models, exploiting the gap between human and machine perception. Although text-based models have also been attacked with adversarial examples, such attacks struggled to preserve semantic meaning and indistinguishability. In this paper, we explore a large class of adversarial examples that can be used to attack text-based models in a black-box setting without making any human-perceptible visual modification to inputs. We use encoding-specific perturbations that are imperceptible to the human eye to manipulate the outputs of a wide range of Natural Language Processing (NLP) systems from neural machine-translation pipelines to web search engines. We find that with a single imperceptible encoding injection -- representing one invisible character, homoglyph, reordering, or deletion -- an attacker can significantly reduce the performance of vulnerable models, and with three injections most models can be functionally broken. Our attacks work against currently-deployed commercial systems, including those produced by Microsoft and Google, in addition to open source models published by Facebook, IBM, and HuggingFace. This novel series of attacks presents a significant threat to many language processing systems: an attacker can affect systems in a targeted manner without any assumptions about the underlying model. We conclude that text-based NLP systems require careful input sanitization, just like conventional applications, and that given such systems are now being deployed rapidly at scale, the urgent attention of architects and operators is required.

摘要: 几年的研究表明，无论是在理论上还是在实践中，机器学习系统都很容易受到对抗性例子的攻击。到目前为止，这类攻击主要针对视觉模型，利用人类和机器感知之间的差距。虽然基于文本的模型也受到了对抗性例子的攻击，但这样的攻击很难保持语义意义和不可区分。在这篇文章中，我们探索了一大类敌意的例子，这些例子可以用来攻击黑盒环境中基于文本的模型，而不需要对输入进行任何人类可感知的视觉修改。我们使用人眼察觉不到的特定于编码的扰动来操作从神经机器翻译管道到网络搜索引擎的广泛的自然语言处理(NLP)系统的输出。我们发现，通过一次不可察觉的编码注入--表示一个不可见的字符、同源字形、重新排序或删除--攻击者可以显著降低易受攻击模型的性能，并且通过三次注入，大多数模型可以在功能上被破坏。我们的攻击针对的是当前部署的商业系统，包括Microsoft和Google生产的系统，以及Facebook、IBM和HuggingFace发布的开源模型。这一系列新颖的攻击对许多语言处理系统构成了重大威胁：攻击者可以有针对性地影响系统，而无需对底层模型进行任何假设。我们的结论是，与传统应用程序一样，基于文本的NLP系统需要仔细的输入清理，并且鉴于此类系统现在正迅速大规模部署，架构师和操作员迫切需要关注。



## **47. Attacking Point Cloud Segmentation with Color-only Perturbation**

基于纯颜色摄动的攻击点云分割 cs.CV

**SubmitDate**: 2021-12-11    [paper-pdf](http://arxiv.org/pdf/2112.05871v1)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng Yufeng Ding, Zhou Li

**Abstracts**: Recent research efforts on 3D point-cloud semantic segmentation have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that semantic segmentation has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point cloud have been studied, we found all of them were targeting single-object recognition, and the perturbation is done on the point coordinates. We argue that the coordinate-based perturbation is unlikely to realize under the physical-world constraints. Hence, we propose a new color-only perturbation method named COLPER, and tailor it to semantic segmentation. By evaluating COLPER on an indoor dataset (S3DIS) and an outdoor dataset (Semantic3D) against three point cloud segmentation models (PointNet++, DeepGCNs, and RandLA-Net), we found color-only perturbation is sufficient to significantly drop the segmentation accuracy and aIoU, under both targeted and non-targeted attack settings.

摘要: 最近的三维点云语义分割研究采用深度卷积神经网络(CNN)和图卷积网络(GCN)，取得了很好的效果。然而，这些复杂模型的稳健性还没有得到系统的分析。鉴于语义分割已经应用于许多安全关键应用(例如，自动驾驶、地质传感)，填补这一知识空白是很重要的，特别是这些模型在敌意样本下是如何受到影响的。在对点云进行对抗性攻击的研究中，我们发现它们都是针对单目标识别的，并且扰动都是在点坐标上进行的。我们认为，在物理世界的约束下，基于坐标的微扰是不可能实现的。为此，我们提出了一种新的纯颜色扰动方法COLPER，并对其进行了语义分割。通过针对三种点云分割模型(PointNet++、DeepGCNs和RandLA-Net)评估室内数据集(S3DIS)和室外数据集(Semanc3D)上的COLPER，我们发现，在目标攻击和非目标攻击设置下，仅颜色扰动就足以显著降低分割精度和AIoU。



## **48. Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks**

保护用户免受中间人攻击的抢占式图像盗用 cs.LG

Accepted and to appear at AAAI 2022

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05634v1)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstracts**: Deep neural networks have become the driving force of modern image recognition systems. However, the vulnerability of neural networks against adversarial attacks poses a serious threat to the people affected by these systems. In this paper, we focus on a real-world threat model where a Man-in-the-Middle adversary maliciously intercepts and perturbs images web users upload online. This type of attack can raise severe ethical concerns on top of simple performance degradation. To prevent this attack, we devise a novel bi-level optimization algorithm that finds points in the vicinity of natural images that are robust to adversarial perturbations. Experiments on CIFAR-10 and ImageNet show our method can effectively robustify natural images within the given modification budget. We also show the proposed method can improve robustness when jointly used with randomized smoothing.

摘要: 深度神经网络已经成为现代图像识别系统的驱动力。然而，神经网络对敌意攻击的脆弱性对受这些系统影响的人们构成了严重威胁。在这篇文章中，我们关注的是一个真实世界的威胁模型，在这个模型中，中间人对手恶意截取和干扰网络用户在线上传的图像。除了简单的性能降级之外，此类攻击还会引发严重的道德问题。为了防止这种攻击，我们设计了一种新的双层优化算法，该算法在自然图像附近寻找对对手扰动具有鲁棒性的点。在CIFAR-10和ImageNet上的实验表明，我们的方法可以在给定的修改预算内有效地增强自然图像的鲁棒性。我们还表明，当与随机平滑联合使用时，所提出的方法可以提高鲁棒性。



## **49. How Private Is Your RL Policy? An Inverse RL Based Analysis Framework**

您的RL政策有多私密？一种基于逆向RL的分析框架 cs.LG

15 pages, 7 figures, 5 tables, version accepted at AAAI 2022

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05495v1)

**Authors**: Kritika Prakash, Fiza Husain, Praveen Paruchuri, Sujit P. Gujar

**Abstracts**: Reinforcement Learning (RL) enables agents to learn how to perform various tasks from scratch. In domains like autonomous driving, recommendation systems, and more, optimal RL policies learned could cause a privacy breach if the policies memorize any part of the private reward. We study the set of existing differentially-private RL policies derived from various RL algorithms such as Value Iteration, Deep Q Networks, and Vanilla Proximal Policy Optimization. We propose a new Privacy-Aware Inverse RL (PRIL) analysis framework, that performs reward reconstruction as an adversarial attack on private policies that the agents may deploy. For this, we introduce the reward reconstruction attack, wherein we seek to reconstruct the original reward from a privacy-preserving policy using an Inverse RL algorithm. An adversary must do poorly at reconstructing the original reward function if the agent uses a tightly private policy. Using this framework, we empirically test the effectiveness of the privacy guarantee offered by the private algorithms on multiple instances of the FrozenLake domain of varying complexities. Based on the analysis performed, we infer a gap between the current standard of privacy offered and the standard of privacy needed to protect reward functions in RL. We do so by quantifying the extent to which each private policy protects the reward function by measuring distances between the original and reconstructed rewards.

摘要: 强化学习(RL)使座席能够学习如何从头开始执行各种任务。在自动驾驶、推荐系统等领域，如果策略记住了私人奖励的任何部分，那么学习到的最优RL策略可能会导致隐私泄露。我们研究了现有的差分私有RL策略集，这些策略来自各种RL算法，如值迭代、深度Q网络和Vanilla近似值策略优化。我们提出了一种新的隐私感知逆向RL(Pril)分析框架，该框架将报酬重构作为对代理可能部署的私有策略的对抗性攻击来执行。为此，我们引入了奖赏重构攻击，利用逆RL算法从隐私保护策略中重构原始奖赏。如果代理人使用严格的私人策略，则对手在重建原始奖励函数方面肯定做得很差。使用该框架，我们在不同复杂度的FrozenLake域的多个实例上对私有算法提供的隐私保证的有效性进行了实证测试。在分析的基础上，我们推断现行的隐私标准与RL中保护奖励功能所需的隐私标准之间存在差距。为此，我们通过测量原始奖励和重建奖励之间的距离来量化每个私人政策保护奖励功能的程度。



## **50. SoK: On the Security & Privacy in Federated Learning**

SOK：论联合学习中的安全与隐私 cs.CR

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05423v1)

**Authors**: Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstracts**: Advances in Machine Learning (ML) and its wide range of applications boosted its popularity. Recent privacy awareness initiatives as the EU General Data Protection Regulation (GDPR) - European Parliament and Council Regulation No 2016/679, subdued ML to privacy and security assessments. Federated Learning (FL) grants a privacy-driven, decentralized training scheme that improves ML models' security. The industry's fast-growing adaptation and security evaluations of FL technology exposed various vulnerabilities. Depending on the FL phase, i.e., training or inference, the adversarial actor capabilities, and the attack type threaten FL's confidentiality, integrity, or availability (CIA). Therefore, the researchers apply the knowledge from distinct domains as countermeasures, like cryptography and statistics.   This work assesses the CIA of FL by reviewing the state-of-the-art (SoTA) for creating a threat model that embraces the attack's surface, adversarial actors, capabilities, and goals. We propose the first unifying taxonomy for attacks and defenses by applying this model. Additionally, we provide critical insights extracted by applying the suggested novel taxonomies to the SoTA, yielding promising future research directions.

摘要: 机器学习(ML)的发展及其广泛的应用推动了它的普及。最近的隐私意识举措，如欧盟一般数据保护条例(GDPR)-欧洲议会和理事会第2016/679号条例，降低了ML对隐私和安全评估的要求。联邦学习(FL)提供了一种隐私驱动的、分散的训练方案，提高了ML模型的安全性。业界对FL技术的快速适应和安全评估暴露了各种漏洞。根据FL阶段，即训练或推理，敌方参与者的能力和攻击类型威胁FL的机密性、完整性或可用性(CIA)。因此，研究人员将来自不同领域的知识作为对策，如密码学和统计学。这项工作评估中央情报局的FL通过审查国家的艺术(SOTA)创建了一个威胁模型，涵盖了攻击的表面，敌对行为者，能力和目标。通过应用该模型，我们提出了第一个统一的攻击和防御分类法。此外，我们提供了通过将建议的新分类法应用于SOTA而提取的批判性见解，产生了有前途的未来研究方向。



