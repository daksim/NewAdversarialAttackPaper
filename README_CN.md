# Latest Adversarial Attack Papers
**update at 2022-12-16 11:43:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks**

交替的目标会产生更强的基于PGD的对抗性攻击 cs.LG

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07992v1) [paper-pdf](http://arxiv.org/pdf/2212.07992v1)

**Authors**: Nikolaos Antoniou, Efthymios Georgiou, Alexandros Potamianos

**Abstract**: Designing powerful adversarial attacks is of paramount importance for the evaluation of $\ell_p$-bounded adversarial defenses. Projected Gradient Descent (PGD) is one of the most effective and conceptually simple algorithms to generate such adversaries. The search space of PGD is dictated by the steepest ascent directions of an objective. Despite the plethora of objective function choices, there is no universally superior option and robustness overestimation may arise from ill-suited objective selection. Driven by this observation, we postulate that the combination of different objectives through a simple loss alternating scheme renders PGD more robust towards design choices. We experimentally verify this assertion on a synthetic-data example and by evaluating our proposed method across 25 different $\ell_{\infty}$-robust models and 3 datasets. The performance improvement is consistent, when compared to the single loss counterparts. In the CIFAR-10 dataset, our strongest adversarial attack outperforms all of the white-box components of AutoAttack (AA) ensemble, as well as the most powerful attacks existing on the literature, achieving state-of-the-art results in the computational budget of our study ($T=100$, no restarts).

摘要: 设计强大的对抗性攻击对于评估$\ellp$受限的对抗性防御是至关重要的。投影梯度下降(PGD)算法是生成此类攻击的最有效且概念简单的算法之一。PGD的搜索空间由目标的最陡上升方向决定。尽管有过多的目标函数选择，但并不存在普遍的最优选择，而且不合适的目标选择可能会导致稳健性高估。在这一观察结果的驱动下，我们假设通过一个简单的损失交替方案将不同的目标组合在一起，使PGD对设计选择更加稳健。我们在一个合成数据示例上实验验证了这一断言，并通过25个不同的$稳健模型和3个数据集评估了我们提出的方法。与单损对应的性能相比，性能改进是一致的。在CIFAR-10数据集中，我们最强的对手攻击超过了AutoAttack(AA)集成的所有白盒组件，以及文献中存在的最强大的攻击，在我们研究的计算预算中获得了最先进的结果($T=100美元，没有重启)。



## **2. A Feedback-optimization-based Model-free Attack Scheme in Networked Control Systems**

一种基于反馈优化的网络控制系统无模型攻击方案 math.OC

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07633v1) [paper-pdf](http://arxiv.org/pdf/2212.07633v1)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: The data-driven attack strategies recently have received much attention when the system model is unknown to the adversary. Depending on the unknown linear system model, the existing data-driven attack strategies have already designed lots of efficient attack schemes. In this paper, we study a completely model-free attack scheme regardless of whether the unknown system model is linear or nonlinear. The objective of the adversary with limited capability is to compromise state variables such that the output value follows a malicious trajectory. Specifically, we first construct a zeroth-order feedback optimization framework and uninterruptedly use probing signals for real-time measurements. Then, we iteratively update the attack signals along the composite direction of the gradient estimates of the objective function evaluations and the projected gradients. These objective function evaluations can be obtained only by real-time measurements. Furthermore, we characterize the optimality of the proposed model-free attack via the optimality gap, which is affected by the dimensions of the attack signal, the iterations of solutions, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the system with internal inherent noise and analyze the effects of noise on the optimality gap. Extensive simulations are conducted to show the effectiveness of the proposed attack scheme.

摘要: 近年来，在敌方未知系统模型的情况下，数据驱动攻击策略受到了广泛的关注。基于未知的线性系统模型，现有的数据驱动攻击策略已经设计了很多有效的攻击方案。本文研究了一种完全无模型的攻击方案，无论未知系统模型是线性的还是非线性的。能力有限的对手的目标是妥协状态变量，使输出值遵循恶意轨迹。具体地说，我们首先构建了零阶反馈优化框架，并不间断地使用探测信号进行实时测量。然后，我们沿着目标函数评估的梯度估计和投影梯度的合成方向迭代地更新攻击信号。这些目标函数评估只能通过实时测量来获得。此外，我们通过最优性间隙来刻画所提出的无模型攻击的最优性，该最优性间隙受攻击信号的维度、解的迭代次数和系统的收敛速度的影响。最后，我们将所提出的攻击方案扩展到具有内部固有噪声的系统，并分析了噪声对最优性间隙的影响。通过大量的仿真实验，验证了该攻击方案的有效性。



## **3. Dissecting Distribution Inference**

剖析分布推理 cs.LG

Accepted at SaTML 2023

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07591v1) [paper-pdf](http://arxiv.org/pdf/2212.07591v1)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference

摘要: 分布推断攻击旨在推断用于训练机器学习模型的数据的统计特性。这些攻击有时威力惊人，但影响分布推断风险的因素并未得到很好的理解，已证明的攻击往往依赖于强大而不切实际的假设，例如完全了解训练环境，即使在假设的黑箱威胁场景中也是如此。为了提高对分布推断风险的理解，我们开发了一种新的黑盒攻击，该攻击在大多数情况下甚至比最著名的白盒攻击性能更好。使用这种新的攻击，我们评估了分布推断风险，同时放松了在黑盒访问下关于对手知识的各种假设，如已知的模型体系结构和仅标签访问。最后，我们评估了以前提出的防御措施的有效性，并引入了新的防御措施。我们发现，尽管基于噪声的防御似乎无效，但简单的重新采样防御可以非常有效。代码可在https://github.com/iamgroot42/dissecting_distribution_inference上找到



## **4. SAIF: Sparse Adversarial and Interpretable Attack Framework**

SAIF：稀疏对抗性可解释攻击框架 cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.07495v1) [paper-pdf](http://arxiv.org/pdf/2212.07495v1)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，将计算的小失真添加到图像可以欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新的攻击技术，称为稀疏对抗性和可解释攻击框架(SAIF)。具体地说，我们设计了在少量像素处包含低幅度扰动的不可察觉攻击，并利用这些稀疏攻击来揭示分类器的脆弱性。我们使用Frank-Wolfe(条件梯度)算法来同时优化有界模和稀疏性的攻击扰动，并且具有$O(1/\Sqrt{T})$收敛。实验结果表明，该算法能够计算高度不可察觉和可解释的敌意实例，并且在ImageNet数据集上的性能优于最新的稀疏攻击方法。



## **5. XRand: Differentially Private Defense against Explanation-Guided Attacks**

XRand：针对解释制导攻击的差异化私人防御 cs.LG

To be published at AAAI 2023

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.04454v3) [paper-pdf](http://arxiv.org/pdf/2212.04454v3)

**Authors**: Truc Nguyen, Phung Lai, NhatHai Phan, My T. Thai

**Abstract**: Recent development in the field of explainable artificial intelligence (XAI) has helped improve trust in Machine-Learning-as-a-Service (MLaaS) systems, in which an explanation is provided together with the model prediction in response to each query. However, XAI also opens a door for adversaries to gain insights into the black-box models in MLaaS, thereby making the models more vulnerable to several attacks. For example, feature-based explanations (e.g., SHAP) could expose the top important features that a black-box model focuses on. Such disclosure has been exploited to craft effective backdoor triggers against malware classifiers. To address this trade-off, we introduce a new concept of achieving local differential privacy (LDP) in the explanations, and from that we establish a defense, called XRand, against such attacks. We show that our mechanism restricts the information that the adversary can learn about the top important features, while maintaining the faithfulness of the explanations.

摘要: 可解释人工智能(XAI)领域的最新发展有助于提高对机器学习即服务(MLaaS)系统的信任，在MLaaS系统中，响应于每个查询，提供解释和模型预测。然而，Xai也为对手打开了一扇门，让他们能够洞察MLaaS中的黑盒模型，从而使这些模型更容易受到几次攻击。例如，基于特征的解释(例如，Shap)可以揭示黑盒模型关注的最重要的特征。这种披露已被利用来手工创建针对恶意软件分类器的有效后门触发器。为了解决这种权衡，我们在解释中引入了实现本地差异隐私(LDP)的新概念，并由此建立了针对此类攻击的防御系统，称为XRand。我们表明，我们的机制限制了攻击者可以了解的关于顶级重要特征的信息，同时保持了解释的可靠性。



## **6. The Devil is in the GAN: Backdoor Attacks and Defenses in Deep Generative Models**

魔鬼在甘地：深层生成模型中的后门攻击和防御 cs.CR

17 pages, 11 figures, 3 tables

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2108.01644v2) [paper-pdf](http://arxiv.org/pdf/2108.01644v2)

**Authors**: Ambrish Rawat, Killian Levacher, Mathieu Sinn

**Abstract**: Deep Generative Models (DGMs) are a popular class of deep learning models which find widespread use because of their ability to synthesize data from complex, high-dimensional manifolds. However, even with their increasing industrial adoption, they haven't been subject to rigorous security and privacy analysis. In this work we examine one such aspect, namely backdoor attacks on DGMs which can significantly limit the applicability of pre-trained models within a model supply chain and at the very least cause massive reputation damage for companies outsourcing DGMs form third parties.   While similar attacks scenarios have been studied in the context of classical prediction models, their manifestation in DGMs hasn't received the same attention. To this end we propose novel training-time attacks which result in corrupted DGMs that synthesize regular data under normal operations and designated target outputs for inputs sampled from a trigger distribution. These attacks are based on an adversarial loss function that combines the dual objectives of attack stealth and fidelity. We systematically analyze these attacks, and show their effectiveness for a variety of approaches like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as well as different data domains including images and audio. Our experiments show that - even for large-scale industry-grade DGMs (like StyleGAN) - our attacks can be mounted with only modest computational effort. We also motivate suitable defenses based on static/dynamic model and output inspections, demonstrate their usefulness, and prescribe a practical and comprehensive defense strategy that paves the way for safe usage of DGMs.

摘要: 深度生成模型(DGM)是一类流行的深度学习模型，由于其能够从复杂的高维流形中合成数据而得到广泛的应用。然而，即使它们越来越多地被工业采用，它们也没有受到严格的安全和隐私分析。在这项工作中，我们研究了一个这样的方面，即对DGM的后门攻击，这种攻击可以显著限制预先训练的模型在模型供应链中的适用性，至少会给从第三方外包DGM的公司造成巨大的声誉损害。虽然类似的攻击场景已经在经典预测模型的背景下进行了研究，但它们在DGM中的表现并没有得到同样的关注。为此，我们提出了新的训练时间攻击，它导致损坏的DGM，这些DGM在正常操作下合成规则数据，并为从触发分布采样的输入指定目标输出。这些攻击基于一种对抗性损失函数，该函数结合了攻击隐形和保真的双重目标。我们系统地分析了这些攻击，并展示了它们对各种方法的有效性，如生成性对抗网络(GANS)和变分自动编码器(VAES)，以及包括图像和音频在内的不同数据域。我们的实验表明，即使对于大规模的工业级DGM(如StyleGAN)，我们的攻击也可以通过适度的计算工作来发起。我们还根据静态/动态模型和输出检查来激发适当的防御，展示其有效性，并制定实用和全面的防御战略，为DGM的安全使用铺平道路。



## **7. Object-fabrication Targeted Attack for Object Detection**

面向目标检测的目标制造定向攻击 cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06431v2) [paper-pdf](http://arxiv.org/pdf/2212.06431v2)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hang Wang, Hongbin Sun, Nanning Zheng

**Abstract**: Recent researches show that the deep learning based object detection is vulnerable to adversarial examples. Generally, the adversarial attack for object detection contains targeted attack and untargeted attack. According to our detailed investigations, the research on the former is relatively fewer than the latter and all the existing methods for the targeted attack follow the same mode, i.e., the object-mislabeling mode that misleads detectors to mislabel the detected object as a specific wrong label. However, this mode has limited attack success rate, universal and generalization performances. In this paper, we propose a new object-fabrication targeted attack mode which can mislead detectors to `fabricate' extra false objects with specific target labels. Furthermore, we design a dual attention based targeted feature space attack method to implement the proposed targeted attack mode. The attack performances of the proposed mode and method are evaluated on MS COCO and BDD100K datasets using FasterRCNN and YOLOv5. Evaluation results demonstrate that, the proposed object-fabrication targeted attack mode and the corresponding targeted feature space attack method show significant improvements in terms of image-specific attack, universal performance and generalization capability, compared with the previous targeted attack for object detection. Code will be made available.

摘要: 最近的研究表明，基于深度学习的目标检测容易受到敌意例子的影响。通常，用于目标检测的对抗性攻击包括定向攻击和非定向攻击。根据我们的详细调查，对前者的研究相对较少，现有的所有定向攻击方法都遵循相同的模式，即对象错误标记模式，即误导检测器将检测到的对象错误标记为特定的错误标记。然而，该模式的攻击成功率、通用性和泛化性能有限。本文提出了一种新的目标制造定向攻击模式，该模式可以误导检测器用特定的目标标签‘制造’额外的虚假目标。此外，我们设计了一种基于双重注意力的目标特征空间攻击方法来实现所提出的目标攻击模式。利用FasterRCNN和YOLOv5对该模型和方法在MS COCO和BDD100K数据集上的攻击性能进行了评估。评估结果表明，与以往的目标检测的目标攻击方法相比，本文提出的目标制造目标攻击模式和相应的目标特征空间攻击方法在图像针对性攻击、通用性能和泛化能力方面都有明显的提高。代码将可用。



## **8. ARCADE: Adversarially Regularized Convolutional Autoencoder for Network Anomaly Detection**

Arcade：一种用于网络异常检测的对数正则卷积自动编码器 cs.LG

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2205.01432v3) [paper-pdf](http://arxiv.org/pdf/2205.01432v3)

**Authors**: Willian T. Lunardi, Martin Andreoni Lopez, Jean-Pierre Giacalone

**Abstract**: As the number of heterogenous IP-connected devices and traffic volume increase, so does the potential for security breaches. The undetected exploitation of these breaches can bring severe cybersecurity and privacy risks. Anomaly-based \acp{IDS} play an essential role in network security. In this paper, we present a practical unsupervised anomaly-based deep learning detection system called ARCADE (Adversarially Regularized Convolutional Autoencoder for unsupervised network anomaly DEtection). With a convolutional \ac{AE}, ARCADE automatically builds a profile of the normal traffic using a subset of raw bytes of a few initial packets of network flows so that potential network anomalies and intrusions can be efficiently detected before they cause more damage to the network. ARCADE is trained exclusively on normal traffic. An adversarial training strategy is proposed to regularize and decrease the \ac{AE}'s capabilities to reconstruct network flows that are out-of-the-normal distribution, thereby improving its anomaly detection capabilities. The proposed approach is more effective than state-of-the-art deep learning approaches for network anomaly detection. Even when examining only two initial packets of a network flow, ARCADE can effectively detect malware infection and network attacks. ARCADE presents 20 times fewer parameters than baselines, achieving significantly faster detection speed and reaction time.

摘要: 随着异类IP连接设备的数量和流量的增加，安全漏洞的可能性也在增加。未被发现的对这些漏洞的利用可能会带来严重的网络安全和隐私风险。基于异常的ACP(入侵检测系统)在网络安全中起着至关重要的作用。本文提出了一种实用的基于无监督异常的深度学习检测系统ARCADE(对抗性正则化卷积自动编码器用于无监督网络异常检测)。通过卷积\ac{AE}，Arcade使用几个网络流初始数据包的原始字节子集自动构建正常流量的配置文件，以便在潜在的网络异常和入侵对网络造成更大损害之前有效地检测到它们。拱廊是专门针对正常交通进行培训的。提出了一种对抗性训练策略，以规范和降低Ac{AE}重构非正态分布网络流的能力，从而提高其异常检测能力。该方法比现有的深度学习网络异常检测方法更有效。即使只检测网络流的两个初始包，ARCADE也可以有效地检测恶意软件感染和网络攻击。ARCADE提供的参数比基线少20倍，检测速度和反应时间显著加快。



## **9. The Quantum Chernoff Divergence in Advantage Distillation for QKD and DIQKD**

QKD和DIQKD优势蒸馏中的量子Chernoff发散 quant-ph

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06975v1) [paper-pdf](http://arxiv.org/pdf/2212.06975v1)

**Authors**: Mikka Stasiuk, Norbert Lutkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible

摘要: 独立于设备的量子密钥分发(DIQKD)旨在通过提供一种在适度的安全假设下提取密钥的方法来减少对量子设备中缺陷的恶意利用。优势蒸馏，一种纠错的双向通信过程，已被证明在提高设备相关和设备无关的量子密钥分发中的噪声容限方面都是有效的。以前，针对一种称为重码协议的优势提取协议，基于涉及协议中某些状态之间的保真度的安全条件，开发了针对IID集体攻击的独立于设备的安全证明。然而，在充分和必要的安全条件之间存在差距，这阻碍了基于保真度的紧噪声容限的计算。我们通过提出另一种证明结构来缩小这一差距，该结构用量子切尔诺夫发散取代了保真度，量子切尔诺夫发散是对称假设检验中出现的一种可区分性衡量标准。在IID集体攻击模型下，根据量子Chernoff散度，我们得到了重码协议安全的充要条件(直到关于后者的一个自然猜想)，从而表明这是该协议的相关关注量。此外，利用这一安全条件，我们在DIQKD的噪声容忍门限上得到了一些改进。我们的结果为量子信息论中的一个基本问题提供了洞察，该问题涉及在什么情况下DIQKD是可能的



## **10. Adversarial Attacks and Defences for Skin Cancer Classification**

针对皮肤癌分类的对抗性攻击和防御 cs.CV

6 pages, 7 figures, 2 tables, 2nd International Conference for  Advancement in Technology (ICONAT 2023), Goa, India

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06822v1) [paper-pdf](http://arxiv.org/pdf/2212.06822v1)

**Authors**: Vinay Jogani, Joy Purohit, Ishaan Shivhare, Samina Attari, Shraddha Surtkar

**Abstract**: There has been a concurrent significant improvement in the medical images used to facilitate diagnosis and the performance of machine learning techniques to perform tasks such as classification, detection, and segmentation in recent years. As a result, a rapid increase in the usage of such systems can be observed in the healthcare industry, for instance in the form of medical image classification systems, where these models have achieved diagnostic parity with human physicians. One such application where this can be observed is in computer vision tasks such as the classification of skin lesions in dermatoscopic images. However, as stakeholders in the healthcare industry, such as insurance companies, continue to invest extensively in machine learning infrastructure, it becomes increasingly important to understand the vulnerabilities in such systems. Due to the highly critical nature of the tasks being carried out by these machine learning models, it is necessary to analyze techniques that could be used to take advantage of these vulnerabilities and methods to defend against them. This paper explores common adversarial attack techniques. The Fast Sign Gradient Method and Projected Descent Gradient are used against a Convolutional Neural Network trained to classify dermatoscopic images of skin lesions. Following that, it also discusses one of the most popular adversarial defense techniques, adversarial training. The performance of the model that has been trained on adversarial examples is then tested against the previously mentioned attacks, and recommendations to improve neural networks robustness are thus provided based on the results of the experiment.

摘要: 近年来，用于促进诊断的医学图像和用于执行分类、检测和分割等任务的机器学习技术的性能有了显著的改进。因此，在医疗保健行业中可以观察到这种系统的使用的快速增加，例如以医学图像分类系统的形式，其中这些模型已经实现了与人类医生的诊断等同。可以观察到这一点的一个这样的应用是在计算机视觉任务中，例如皮肤镜图像中的皮肤病变的分类。然而，随着医疗保健行业的利益相关者(如保险公司)继续在机器学习基础设施上进行广泛投资，了解此类系统中的漏洞变得越来越重要。由于这些机器学习模型执行的任务具有高度关键的性质，因此有必要分析可用于利用这些漏洞的技术和防御它们的方法。本文探讨了常见的对抗性攻击技术。利用快速符号梯度法和投影下降梯度法对训练好的卷积神经网络进行训练，对皮肤镜图像进行分类。其次，还讨论了一种最流行的对抗性防守技术--对抗性训练。然后对经过对抗性样本训练的模型的性能针对上述攻击进行了测试，并根据实验结果提出了改善神经网络稳健性的建议。



## **11. Towards Efficient and Domain-Agnostic Evasion Attack with High-dimensional Categorical Inputs**

面向高维类别输入的高效领域不可知回避攻击 cs.LG

AAAI 2023

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06836v1) [paper-pdf](http://arxiv.org/pdf/2212.06836v1)

**Authors**: Hongyan Bao, Yufei Han, Yujun Zhou, Xin Gao, Xiangliang Zhang

**Abstract**: Our work targets at searching feasible adversarial perturbation to attack a classifier with high-dimensional categorical inputs in a domain-agnostic setting. This is intrinsically an NP-hard knapsack problem where the exploration space becomes explosively larger as the feature dimension increases. Without the help of domain knowledge, solving this problem via heuristic method, such as Branch-and-Bound, suffers from exponential complexity, yet can bring arbitrarily bad attack results. We address the challenge via the lens of multi-armed bandit based combinatorial search. Our proposed method, namely FEAT, treats modifying each categorical feature as pulling an arm in multi-armed bandit programming. Our objective is to achieve highly efficient and effective attack using an Orthogonal Matching Pursuit (OMP)-enhanced Upper Confidence Bound (UCB) exploration strategy. Our theoretical analysis bounding the regret gap of FEAT guarantees its practical attack performance. In empirical analysis, we compare FEAT with other state-of-the-art domain-agnostic attack methods over various real-world categorical data sets of different applications. Substantial experimental observations confirm the expected efficiency and attack effectiveness of FEAT applied in different application scenarios. Our work further hints the applicability of FEAT for assessing the adversarial vulnerability of classification systems with high-dimensional categorical inputs.

摘要: 我们的工作目标是在领域不可知的环境中寻找可行的对抗性扰动来攻击具有高维类别输入的分类器。这本质上是一个NP难背包问题，其中随着特征维度的增加，探索空间变得爆炸性地大。在没有领域知识的帮助下，用启发式方法如分支定界法求解该问题具有指数级的复杂性，但会带来任意恶劣的攻击结果。我们通过基于多臂强盗的组合搜索的镜头来解决这一挑战。我们提出的方法，即FEAT，将修改每个类别特征视为在多臂强盗编程中拉动手臂。我们的目标是使用一种正交匹配追踪(OMP)增强的上置信限(UCB)探索策略来实现高效和有效的攻击。我们的理论分析弥补了壮举的遗憾差距，保证了它的实际进攻性能。在实证分析中，我们将FEAT与其他最先进的领域无关攻击方法在不同应用的各种真实世界分类数据集上进行了比较。大量的实验观察证实了Feat在不同应用场景下的预期效率和攻击效果。我们的工作进一步暗示了FEAT在评估具有高维分类输入的分类系统的对抗性脆弱性方面的适用性。



## **12. The Importance of Image Interpretation: Patterns of Semantic Misclassification in Real-World Adversarial Images**

图像解释的重要性：现实世界中对抗性图像中的语义错误分类模式 cs.CV

International Conference on Multimedia Modeling (MMM) 2023. Resources  are publicly available at  https://github.com/ZhengyuZhao/Targeted-Transfer/tree/main/human_eval

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2206.01467v2) [paper-pdf](http://arxiv.org/pdf/2206.01467v2)

**Authors**: Zhengyu Zhao, Nga Dang, Martha Larson

**Abstract**: Adversarial images are created with the intention of causing an image classifier to produce a misclassification. In this paper, we propose that adversarial images should be evaluated based on semantic mismatch, rather than label mismatch, as used in current work. In other words, we propose that an image of a "mug" would be considered adversarial if classified as "turnip", but not as "cup", as current systems would assume. Our novel idea of taking semantic misclassification into account in the evaluation of adversarial images offers two benefits. First, it is a more realistic conceptualization of what makes an image adversarial, which is important in order to fully understand the implications of adversarial images for security and privacy. Second, it makes it possible to evaluate the transferability of adversarial images to a real-world classifier, without requiring the classifier's label set to have been available during the creation of the images. The paper carries out an evaluation of a transfer attack on a real-world image classifier that is made possible by our semantic misclassification approach. The attack reveals patterns in the semantics of adversarial misclassifications that could not be investigated using conventional label mismatch.

摘要: 创建敌意图像的目的是使图像分类器产生错误分类。在本文中，我们提出了基于语义失配而不是基于标签失配的对抗性图像评价方法。换句话说，我们建议，如果一个“杯子”的图像被归类为“萝卜”，那么它将被认为是对抗性的，而不是像目前的系统所假设的那样被归类为“杯子”。我们提出的在对抗性图像的评估中考虑语义错误分类的新想法提供了两个好处。首先，它是一种更现实的概念，即是什么使图像具有对抗性，这对于充分理解对抗性图像对安全和隐私的影响是很重要的。其次，它使得评估敌意图像到真实世界的分类器的可转移性成为可能，而不需要分类器的标签集在图像创建期间是可用的。本文对基于语义错误分类方法的真实世界图像分类器的传输攻击进行了评估。该攻击揭示了对抗性错误分类的语义模式，这些模式无法使用传统的标签不匹配进行调查。



## **13. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06776v1) [paper-pdf](http://arxiv.org/pdf/2212.06776v1)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **14. Adversarial Detection by Approximation of Ensemble Boundary**

基于集合边界逼近的对抗性检测 cs.LG

6 pages, 3 figures, 5 tables

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2211.10227v3) [paper-pdf](http://arxiv.org/pdf/2211.10227v3)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area. Code for this paper implementing Walsh Coefficient Examples of approximating artificial Boolean functions can be found at https://doi.org/10.24433/CO.3695905.v1

摘要: 提出一种布尔函数的谱逼近方法，用于逼近求解两类模式识别问题的深度神经网络(DNN)集成的决策边界。实验表明，相对较弱的DNN分类器的Walsh组合能够检测到对抗性攻击。通过观察干净图像和敌意图像在沃尔什系数逼近上的差异，可以看出攻击的可转移性可以用于检测。近似决策边界也有助于理解DNN的学习和可转移性。虽然这里的实验使用的是图像，但所提出的建模两类集合决策边界的方法原则上可以应用于任何应用领域。本文实现沃尔什系数的代码可以在https://doi.org/10.24433/CO.3695905.v1上找到逼近人工布尔函数的示例



## **15. Pixel is All You Need: Adversarial Trajectory-Ensemble Active Learning for Salient Object Detection**

像素就是您所需要的：对抗性轨迹-集成主动学习用于显著目标检测 cs.CV

9 pages, 8 figures

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.06493v1) [paper-pdf](http://arxiv.org/pdf/2212.06493v1)

**Authors**: Zhenyu Wu, Lin Wang, Wei Wang, Qing Xia, Chenglizhao Chen, Aimin Hao, Shuo Li

**Abstract**: Although weakly-supervised techniques can reduce the labeling effort, it is unclear whether a saliency model trained with weakly-supervised data (e.g., point annotation) can achieve the equivalent performance of its fully-supervised version. This paper attempts to answer this unexplored question by proving a hypothesis: there is a point-labeled dataset where saliency models trained on it can achieve equivalent performance when trained on the densely annotated dataset. To prove this conjecture, we proposed a novel yet effective adversarial trajectory-ensemble active learning (ATAL). Our contributions are three-fold: 1) Our proposed adversarial attack triggering uncertainty can conquer the overconfidence of existing active learning methods and accurately locate these uncertain pixels. {2)} Our proposed trajectory-ensemble uncertainty estimation method maintains the advantages of the ensemble networks while significantly reducing the computational cost. {3)} Our proposed relationship-aware diversity sampling algorithm can conquer oversampling while boosting performance. Experimental results show that our ATAL can find such a point-labeled dataset, where a saliency model trained on it obtained $97\%$ -- $99\%$ performance of its fully-supervised version with only ten annotated points per image.

摘要: 虽然弱监督技术可以减少标注工作量，但目前尚不清楚用弱监督数据(例如，点标注)训练的显著模型是否能够达到其完全监督版本的同等性能。本文试图通过证明一个假设来回答这个悬而未决的问题：存在一个点标记的数据集，在该数据集上训练的显著模型在密集标注的数据集上训练时可以获得相同的性能。为了证明这一猜想，我们提出了一种新颖而有效的对抗轨迹--集成主动学习(ALAL)。我们的贡献有三个方面：1)提出的基于不确定性的对抗性攻击能够克服现有主动学习方法的过度自信，并准确定位这些不确定的像素。{2)}我们提出的轨迹集成不确定性估计方法保持了集成网络的优点，同时显著降低了计算成本。{3)}我们提出的关系感知多样性采样算法可以在提高性能的同时克服过采样问题。实验结果表明，我们的ALAL算法可以找到这样一个点标记的数据集，在此基础上训练的显著模型在每幅图像上只有10个注释点的情况下获得了其完全监督版本的$97$-$99$的性能。



## **16. Adversarially Robust Video Perception by Seeing Motion**

通过视觉运动实现相对强健的视频感知 cs.CV

**SubmitDate**: 2022-12-13    [abs](http://arxiv.org/abs/2212.07815v1) [paper-pdf](http://arxiv.org/pdf/2212.07815v1)

**Authors**: Lingyu Zhang, Chengzhi Mao, Junfeng Yang, Carl Vondrick

**Abstract**: Despite their excellent performance, state-of-the-art computer vision models often fail when they encounter adversarial examples. Video perception models tend to be more fragile under attacks, because the adversary has more places to manipulate in high-dimensional data. In this paper, we find one reason for video models' vulnerability is that they fail to perceive the correct motion under adversarial perturbations. Inspired by the extensive evidence that motion is a key factor for the human visual system, we propose to correct what the model sees by restoring the perceived motion information. Since motion information is an intrinsic structure of the video data, recovering motion signals can be done at inference time without any human annotation, which allows the model to adapt to unforeseen, worst-case inputs. Visualizations and empirical experiments on UCF-101 and HMDB-51 datasets show that restoring motion information in deep vision models improves adversarial robustness. Even under adaptive attacks where the adversary knows our defense, our algorithm is still effective. Our work provides new insight into robust video perception algorithms by using intrinsic structures from the data. Our webpage is available at https://motion4robust.cs.columbia.edu.

摘要: 尽管表现出色，但最先进的计算机视觉模型在遇到对抗性例子时往往会失败。视频感知模型在攻击下往往更加脆弱，因为对手在高维数据中有更多的地方可以操纵。在本文中，我们发现视频模型易受攻击的原因之一是它们在对抗性扰动下无法感知正确的运动。受运动是人类视觉系统的关键因素这一广泛证据的启发，我们建议通过恢复感知的运动信息来纠正模型所看到的。由于运动信息是视频数据的内在结构，因此可以在没有任何人工注释的情况下在推断时恢复运动信号，这使得模型能够适应不可预见的、最坏情况的输入。在UCF-101和HMDB-51数据集上的可视化和经验实验表明，在深度视觉模型中恢复运动信息提高了对手的鲁棒性。即使在对手知道我们的防御的自适应攻击下，我们的算法仍然有效。我们的工作通过使用数据的内在结构为稳健的视频感知算法提供了新的见解。我们的网页位于https://motion4robust.cs.columbia.edu.



## **17. Securing the Spike: On the Transferability and Security of Spiking Neural Networks to Adversarial Examples**

保护尖峰：尖峰神经网络对对抗性例子的可转移性和安全性 cs.NE

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2209.03358v2) [paper-pdf](http://arxiv.org/pdf/2209.03358v2)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstract**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remain relatively underdeveloped. In this work we focus on advancing the adversarial attack side of SNNs and make three major contributions. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, using the best surrogate gradient technique, we analyze the transferability of adversarial attacks on SNNs and other state-of-the-art architectures like Vision Transformers (ViTs) and Big Transfer Convolutional Neural Networks (CNNs). We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Third, due to the lack of an ubiquitous white-box attack that is effective across both the SNN and CNN/ViT domains, we develop a new white-box attack, the Auto Self-Attention Gradient Attack (Auto SAGA). Our novel attack generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Auto SAGA is as much as $87.9\%$ more effective on SNN/ViT model ensembles than conventional white-box attacks like PGD. Our experiments and analyses are broad and rigorous covering three datasets (CIFAR-10, CIFAR-100 and ImageNet), five different white-box attacks and nineteen different classifier models (seven for each CIFAR dataset and five different models for ImageNet).

摘要: 尖峰神经网络(SNN)因其高能量效率和分类性能的最新进展而备受关注。然而，与传统的深度学习方法不同的是，对SNN对敌意例子的稳健性的分析和研究还相对较不发达。在这项工作中，我们专注于推进SNN的对抗性攻击端，并做出了三个主要贡献。首先，我们证明了针对SNN的成功的白盒对抗攻击高度依赖于潜在的代理梯度技术。其次，利用最优代理梯度技术，分析了对抗性攻击对SNN以及视觉转换器(VITS)和大转移卷积神经网络(CNN)等体系结构的可转移性。我们证明了SNN不会经常被Vision Transformers和某些类型的CNN生成的敌意例子所欺骗。第三，由于缺乏一种在SNN和CNN/VIT域都有效的普遍存在的白盒攻击，我们开发了一种新的白盒攻击，自动自我注意梯度攻击(Auto SAGA)。我们的新攻击生成了能够同时愚弄SNN模型和非SNN模型的敌意示例。Auto Saga在SNN/VIT模式下比传统的白盒攻击(如PGD)更有效，最高可达87.9美元。我们的实验和分析涵盖了三个数据集(CIFAR-10、CIFAR-100和ImageNet)、五个不同的白盒攻击和19个不同的分类器模型(每个CIFAR数据集七个，ImageNet的五个不同模型)。



## **18. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

使用校准的工作证明增加模型提取的成本 cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2201.09243v3) [paper-pdf](http://arxiv.org/pdf/2201.09243v3)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstract**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.

摘要: 在模型提取攻击中，攻击者可以通过反复查询通过公共API暴露的机器学习模型，并根据获得的预测调整自己的模型，从而窃取该模型。为了防止模型窃取，现有的防御措施侧重于检测恶意查询、截断或扭曲输出，因此必然会在健壮性和模型实用程序之间为合法用户带来折衷。相反，我们建议通过要求用户在阅读模型预测之前完成工作证明来阻碍模型提取。这大大增加了(甚至高达100倍)利用查询访问进行模型提取所需的计算工作量，从而阻止了攻击者。由于我们对完成每个查询的工作证明所需的工作量进行了校准，因此这只会给普通用户带来很小的开销(最高可达2倍)。为了实现这一点，我们的校准应用了来自差异隐私的工具来衡量查询所揭示的信息。我们的方法不需要修改受害者模型，并且可以被机器学习从业者应用，以保护他们公开曝光的模型不会轻易被窃取。



## **19. Dictionary Attacks on Speaker Verification**

针对说话人确认的词典攻击 cs.SD

Accepted in IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2204.11304v2) [paper-pdf](http://arxiv.org/pdf/2204.11304v2)

**Authors**: Mirko Marras, Pawel Korus, Anubhav Jain, Nasir Memon

**Abstract**: In this paper, we propose dictionary attacks against speaker verification - a novel attack vector that aims to match a large fraction of speaker population by chance. We introduce a generic formulation of the attack that can be used with various speech representations and threat models. The attacker uses adversarial optimization to maximize raw similarity of speaker embeddings between a seed speech sample and a proxy population. The resulting master voice successfully matches a non-trivial fraction of people in an unknown population. Adversarial waveforms obtained with our approach can match on average 69% of females and 38% of males enrolled in the target system at a strict decision threshold calibrated to yield false alarm rate of 1%. By using the attack with a black-box voice cloning system, we obtain master voices that are effective in the most challenging conditions and transferable between speaker encoders. We also show that, combined with multiple attempts, this attack opens even more to serious issues on the security of these systems.

摘要: 在本文中，我们提出了针对说话人验证的词典攻击，这是一种新的攻击向量，旨在随机匹配大部分说话人群体。我们介绍了一种可用于各种语音表示和威胁模型的攻击的通用公式。攻击者使用对抗性优化来最大化种子语音样本和代理群体之间说话人嵌入的原始相似性。由此产生的主音成功地匹配了未知人群中的一小部分人。使用我们的方法获得的对抗性波形可以在严格的判决阈值下与目标系统中登记的平均69%的女性和38%的男性匹配，该阈值被校准为产生1%的错误警报率。通过使用黑匣子语音克隆系统进行攻击，我们获得了在最具挑战性的条件下有效的主音，并且可以在说话人编码者之间传输。我们还表明，与多次尝试相结合，这种攻击会给这些系统的安全带来更严重的问题。



## **20. Reversing Skin Cancer Adversarial Examples by Multiscale Diffusive and Denoising Aggregation Mechanism**

多尺度扩散去噪聚集机制逆转皮肤癌对抗性实例 cs.CV

11 pages

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2208.10373v2) [paper-pdf](http://arxiv.org/pdf/2208.10373v2)

**Authors**: Yongwei Wang, Yuan Li, Zhiqi Shen

**Abstract**: Reliable skin cancer diagnosis models play an essential role in early screening and medical intervention. Prevailing computer-aided skin cancer classification systems employ deep learning approaches. However, recent studies reveal their extreme vulnerability to adversarial attacks -- often imperceptible perturbations to significantly reduce the performances of skin cancer diagnosis models. To mitigate these threats, this work presents a simple, effective, and resource-efficient defense framework by reverse engineering adversarial perturbations in skin cancer images. Specifically, a multiscale image pyramid is first established to better preserve discriminative structures in the medical imaging domain. To neutralize adversarial effects, skin images at different scales are then progressively diffused by injecting isotropic Gaussian noises to move the adversarial examples to the clean image manifold. Crucially, to further reverse adversarial noises and suppress redundant injected noises, a novel multiscale denoising mechanism is carefully designed that aggregates image information from neighboring scales. We evaluated the defensive effectiveness of our method on ISIC 2019, a largest skin cancer multiclass classification dataset. Experimental results demonstrate that the proposed method can successfully reverse adversarial perturbations from different attacks and significantly outperform some state-of-the-art methods in defending skin cancer diagnosis models.

摘要: 可靠的皮肤癌诊断模型在早期筛查和医疗干预中起着至关重要的作用。流行的计算机辅助皮肤癌分类系统采用深度学习方法。然而，最近的研究揭示了它们对对抗性攻击的极端脆弱性--通常是难以察觉的干扰，从而显著降低皮肤癌诊断模型的性能。为了减轻这些威胁，这项工作提出了一种简单、有效和资源高效的防御框架，通过反向工程皮肤癌图像中的对抗性扰动。具体地说，首先建立了多尺度图像金字塔，以更好地保留医学成像领域的区分结构。为了中和对抗性效果，然后通过注入各向同性高斯噪声来逐步扩散不同尺度上的皮肤图像，以将对抗性示例移动到干净的图像流形上。为了进一步逆转对抗性噪声和抑制多余的注入噪声，仔细设计了一种新的多尺度去噪机制，该机制聚集了相邻尺度上的图像信息。我们在最大的皮肤癌多类分类数据集ISIC 2019上评估了我们方法的防御效果。实验结果表明，该方法能够成功地逆转来自不同攻击的对抗性扰动，并且在防御皮肤癌诊断模型方面明显优于一些最新的方法。



## **21. HOTCOLD Block: Fooling Thermal Infrared Detectors with a Novel Wearable Design**

HOTCOLD模块：用新的可穿戴设计愚弄热红外探测器 cs.CV

Accepted to AAAI 2023

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2212.05709v1) [paper-pdf](http://arxiv.org/pdf/2212.05709v1)

**Authors**: Hui Wei, Zhixiang Wang, Xuemei Jia, Yinqiang Zheng, Hao Tang, Shin'ichi Satoh, Zheng Wang

**Abstract**: Adversarial attacks on thermal infrared imaging expose the risk of related applications. Estimating the security of these systems is essential for safely deploying them in the real world. In many cases, realizing the attacks in the physical space requires elaborate special perturbations. These solutions are often \emph{impractical} and \emph{attention-grabbing}. To address the need for a physically practical and stealthy adversarial attack, we introduce \textsc{HotCold} Block, a novel physical attack for infrared detectors that hide persons utilizing the wearable Warming Paste and Cooling Paste. By attaching these readily available temperature-controlled materials to the body, \textsc{HotCold} Block evades human eyes efficiently. Moreover, unlike existing methods that build adversarial patches with complex texture and structure features, \textsc{HotCold} Block utilizes an SSP-oriented adversarial optimization algorithm that enables attacks with pure color blocks and explores the influence of size, shape, and position on attack performance. Extensive experimental results in both digital and physical environments demonstrate the performance of our proposed \textsc{HotCold} Block. \emph{Code is available: \textcolor{magenta}{https://github.com/weihui1308/HOTCOLDBlock}}.

摘要: 对热红外成像的敌意攻击暴露了相关应用的风险。评估这些系统的安全性对于在现实世界中安全地部署它们至关重要。在许多情况下，在物理空间实现攻击需要精心设计的特殊扰动。这些解决方案往往是不切实际的，也是吸引眼球的。为了满足对物理实用和隐形攻击的需求，我们引入了一种新型的红外探测器物理攻击--阻止使用可穿戴的温贴和冷贴。通过将这些随手可得的温控材料附着在人体上，BLOCK可以有效地躲避人眼。此外，与已有的构建具有复杂纹理和结构特征的对抗性补丁的方法不同，文本块使用面向SSP的对抗性优化算法来实现纯颜色块的攻击，并探索大小、形状和位置对攻击性能的影响。在数字和物理环境中的大量实验结果证明了我们所提出的文本块的性能。\EMPH{代码可用：\textcolor{magenta}{https://github.com/weihui1308/HOTCOLDBlock}}.



## **22. REAP: A Large-Scale Realistic Adversarial Patch Benchmark**

REAP：一个大规模的现实对抗性补丁基准 cs.CV

Code and benchmark can be found at  https://github.com/wagner-group/reap-benchmark

**SubmitDate**: 2022-12-12    [abs](http://arxiv.org/abs/2212.05680v1) [paper-pdf](http://arxiv.org/pdf/2212.05680v1)

**Authors**: Nabeel Hingun, Chawin Sitawarin, Jerry Li, David Wagner

**Abstract**: Machine learning models are known to be susceptible to adversarial perturbation. One famous attack is the adversarial patch, a sticker with a particularly crafted pattern that makes the model incorrectly predict the object it is placed on. This attack presents a critical threat to cyber-physical systems that rely on cameras such as autonomous cars. Despite the significance of the problem, conducting research in this setting has been difficult; evaluating attacks and defenses in the real world is exceptionally costly while synthetic data are unrealistic. In this work, we propose the REAP (REalistic Adversarial Patch) benchmark, a digital benchmark that allows the user to evaluate patch attacks on real images, and under real-world conditions. Built on top of the Mapillary Vistas dataset, our benchmark contains over 14,000 traffic signs. Each sign is augmented with a pair of geometric and lighting transformations, which can be used to apply a digitally generated patch realistically onto the sign. Using our benchmark, we perform the first large-scale assessments of adversarial patch attacks under realistic conditions. Our experiments suggest that adversarial patch attacks may present a smaller threat than previously believed and that the success rate of an attack on simpler digital simulations is not predictive of its actual effectiveness in practice. We release our benchmark publicly at https://github.com/wagner-group/reap-benchmark.

摘要: 众所周知，机器学习模型容易受到对抗性扰动的影响。一种著名的攻击是对抗性补丁，这是一种带有特别精心制作的图案的贴纸，使模型无法正确预测它所放置的对象。这种攻击对自动驾驶汽车等依赖摄像头的网络物理系统构成了严重威胁。尽管这个问题很重要，但在这种情况下进行研究一直很困难；评估现实世界中的攻击和防御成本异常高昂，而合成数据是不切实际的。在这项工作中，我们提出了REAP(现实对抗补丁)基准，这是一个数字基准，允许用户在真实世界的条件下评估对真实图像的补丁攻击。我们的基准建立在Mapillary Vistas数据集的基础上，包含超过14,000个交通标志。每个标志都增加了一对几何和照明变换，可以用来将数字生成的补丁逼真地应用到标志上。使用我们的基准，我们在现实条件下执行了第一次大规模的对抗性补丁攻击评估。我们的实验表明，对抗性补丁攻击可能比之前认为的威胁更小，并且在更简单的数字模拟上的攻击成功率并不能预测其在实践中的实际有效性。我们在https://github.com/wagner-group/reap-benchmark.上公开发布我们的基准



## **23. DISCO: Adversarial Defense with Local Implicit Functions**

DISCO：局部隐含函数的对抗性防御 cs.CV

Accepted to Neurips 2022

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05630v1) [paper-pdf](http://arxiv.org/pdf/2212.05630v1)

**Authors**: Chih-Hui Ho, Nuno Vasconcelos

**Abstract**: The problem of adversarial defenses for image classification, where the goal is to robustify a classifier against adversarial examples, is considered. Inspired by the hypothesis that these examples lie beyond the natural image manifold, a novel aDversarIal defenSe with local impliCit functiOns (DISCO) is proposed to remove adversarial perturbations by localized manifold projections. DISCO consumes an adversarial image and a query pixel location and outputs a clean RGB value at the location. It is implemented with an encoder and a local implicit module, where the former produces per-pixel deep features and the latter uses the features in the neighborhood of query pixel for predicting the clean RGB value. Extensive experiments demonstrate that both DISCO and its cascade version outperform prior defenses, regardless of whether the defense is known to the attacker. DISCO is also shown to be data and parameter efficient and to mount defenses that transfers across datasets, classifiers and attacks.

摘要: 考虑了图像分类的对抗性防御问题，其中目标是针对对抗性实例使分类器具有更强的鲁棒性。受这些例子位于自然图像流形之外的假设启发，提出了一种新的基于局部隐函数的对抗性防御算法(DISCO)，通过局部流形投影消除对抗性扰动。DISCO使用敌意图像和查询像素位置，并在该位置输出干净的RGB值。该算法由一个编码器和一个局部隐式模块实现，前者产生每个像素的深度特征，后者利用查询像素附近的特征来预测干净的RGB值。广泛的实验表明，无论攻击者是否知道防御，迪斯科及其级联版本都优于之前的防御。DISCO还被证明是数据和参数高效的，并安装了跨数据集、分类器和攻击传输的防御系统。



## **24. On Deep Learning in Password Guessing, a Survey**

密码猜测中的深度学习研究综述 cs.CR

8 pages, 4 figures, 3 tables. arXiv admin note: substantial text  overlap with arXiv:2208.06943

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2208.10413v2) [paper-pdf](http://arxiv.org/pdf/2208.10413v2)

**Authors**: Fangyi Yu

**Abstract**: The security of passwords is dependent on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to be representative of the actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper compares various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. The involved model categories are Recurrent Neural Networks, Generative Adversarial Networks, Autoencoder, and Attention mechanisms. Additionally, we proposed a promising research experimental design on using variations of IWGAN on password guessing under non-targeted offline attacks. Using these advanced strategies, we can enhance password security and create more accurate and efficient Password Strength Meters.

摘要: 密码的安全性取决于对攻击者使用的策略的透彻理解。不幸的是，现实世界中的对手使用的是实用的猜测策略，如字典攻击，这在密码安全研究中很难模拟。必须仔细配置和修改字典攻击，才能代表实际威胁。然而，这种方法需要难以复制的特定领域的知识和专业技能。本文比较了各种基于深度学习的密码猜测方法，这些方法不需要领域知识，也不需要假设用户的密码结构和组合。涉及的模型类别包括递归神经网络、生成性对抗网络、自动编码器和注意机制。此外，我们还提出了一种在非目标离线攻击下使用IWGAN变体进行密码猜测的研究实验设计。使用这些高级策略，我们可以增强密码安全性，并创建更准确和高效的密码强度计。



## **25. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

虚假数据注入攻击下的大规模网络安全防御：一种攻击检测调度方法 eess.SY

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05500v1) [paper-pdf](http://arxiv.org/pdf/2212.05500v1)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large scale networks, communication links between nodes are easily injected with false data by adversaries, so this paper proposes a novel security defense strategy to ensure the security of the network from the perspective of attack detection scheduling. Compared with existing attack detection methods, the attack detection scheduling strategy in this paper only needs to detect half of the neighbor node information to ensure the security of the node local state estimation. We first formulate the problem of selecting the sensor to be detected as a combinatorial optimization problem, which is Nondeterminism Polynomial hard (NP-hard). To solve the above problem, we convert the objective function into a submodular function. Then, we propose an attack detection scheduling algorithm based on sequential submodular maximization, which incorporates expert problem to better cope with dynamic attack strategies. The proposed algorithm can run in polynomial time with a theoretical lower bound on the optimization rate. In addition, the proposed algorithm can guarantee the security of the whole network under two kinds of insecurity conditions from the perspective of the augmented estimation error. Finally, a numerical simulation of the industrial continuous stirred tank reactor verifies the effectiveness of the developed approach.

摘要: 在大规模网络中，节点之间的通信链路很容易被对手注入虚假数据，因此从攻击检测调度的角度提出了一种新的安全防御策略来保证网络的安全性。与现有的攻击检测方法相比，本文提出的攻击检测调度策略只需要检测一半的邻居节点信息，保证了节点局部状态估计的安全性。首先将传感器的选择问题描述为一个组合优化问题，这是一个非确定多项式困难(NP-Hard)问题。为了解决上述问题，我们将目标函数转换为子模函数。在此基础上，提出了一种基于顺序子模块最大化的攻击检测调度算法，该算法引入了专家问题，能够更好地应对动态攻击策略。该算法可以在多项式时间内运行，并给出了优化速度的理论下界。另外，从估计误差增大的角度来看，该算法在两种不安全情况下都能保证整个网络的安全性。最后，对工业连续搅拌釜式反应器进行了数值模拟，验证了该方法的有效性。



## **26. General Adversarial Defense Against Black-box Attacks via Pixel Level and Feature Level Distribution Alignments**

基于像素级和特征级分布对齐的黑盒攻击一般对抗防御 cs.CV

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05387v1) [paper-pdf](http://arxiv.org/pdf/2212.05387v1)

**Authors**: Xiaogang Xu, Hengshuang Zhao, Philip Torr, Jiaya Jia

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to the black-box adversarial attack that is highly transferable. This threat comes from the distribution gap between adversarial and clean samples in feature space of the target DNNs. In this paper, we use Deep Generative Networks (DGNs) with a novel training mechanism to eliminate the distribution gap. The trained DGNs align the distribution of adversarial samples with clean ones for the target DNNs by translating pixel values. Different from previous work, we propose a more effective pixel level training constraint to make this achievable, thus enhancing robustness on adversarial samples. Further, a class-aware feature-level constraint is formulated for integrated distribution alignment. Our approach is general and applicable to multiple tasks, including image classification, semantic segmentation, and object detection. We conduct extensive experiments on different datasets. Our strategy demonstrates its unique effectiveness and generality against black-box attacks.

摘要: 深度神经网络(DNN)很容易受到具有高度可转移性的黑盒攻击。这种威胁来自于敌意样本和干净样本在目标DNN特征空间中的分布差距。在本文中，我们使用深度生成网络(DGNs)和一种新颖的训练机制来消除分布差距。经过训练的DGN通过转换像素值将目标DNN的对抗性样本的分布与干净的样本的分布对齐。与以前的工作不同，我们提出了一种更有效的像素级训练约束来实现这一点，从而增强了对对手样本的稳健性。在此基础上，提出了一种类感知特征级约束，用于集成分布对齐。我们的方法是通用的，适用于多种任务，包括图像分类、语义分割和目标检测。我们在不同的数据集上进行了广泛的实验。我们的策略展示了它对黑盒攻击的独特有效性和通用性。



## **27. Mitigating Adversarial Gray-Box Attacks Against Phishing Detectors**

减轻针对网络钓鱼检测器的敌意灰盒攻击 cs.CR

**SubmitDate**: 2022-12-11    [abs](http://arxiv.org/abs/2212.05380v1) [paper-pdf](http://arxiv.org/pdf/2212.05380v1)

**Authors**: Giovanni Apruzzese, V. S. Subrahmanian

**Abstract**: Although machine learning based algorithms have been extensively used for detecting phishing websites, there has been relatively little work on how adversaries may attack such "phishing detectors" (PDs for short). In this paper, we propose a set of Gray-Box attacks on PDs that an adversary may use which vary depending on the knowledge that he has about the PD. We show that these attacks severely degrade the effectiveness of several existing PDs. We then propose the concept of operation chains that iteratively map an original set of features to a new set of features and develop the "Protective Operation Chain" (POC for short) algorithm. POC leverages the combination of random feature selection and feature mappings in order to increase the attacker's uncertainty about the target PD. Using 3 existing publicly available datasets plus a fourth that we have created and will release upon the publication of this paper, we show that POC is more robust to these attacks than past competing work, while preserving predictive performance when no adversarial attacks are present. Moreover, POC is robust to attacks on 13 different classifiers, not just one. These results are shown to be statistically significant at the p < 0.001 level.

摘要: 虽然基于机器学习的算法已被广泛用于检测钓鱼网站，但关于攻击者如何攻击此类“钓鱼检测器”(简称PD)的工作相对较少。在本文中，我们提出了一组针对PD的Gray-Box攻击，对手可能会使用这些攻击，这些攻击取决于他对PD的了解。我们表明，这些攻击严重降低了几个现有PD的有效性。然后，我们提出了操作链的概念，该操作链迭代地将原始特征集映射到新的特征集，并开发了保护操作链(POC)算法。POC利用随机特征选择和特征映射的组合，以增加攻击者对目标PD的不确定性。使用3个现有的公开可用的数据集加上我们已经创建并将在本文发表后发布的第四个数据集，我们表明POC比过去竞争的工作更健壮，同时在没有对手攻击的情况下保持预测性能。此外，PoC对13个不同分类器的攻击具有健壮性，而不是只有一个。这些结果在p<0.001水平上具有统计学意义。



## **28. Targeted Adversarial Attacks on Deep Reinforcement Learning Policies via Model Checking**

基于模型检测的深度强化学习策略定向攻击 cs.LG

ICAART 2023 Paper (Technical Report)

**SubmitDate**: 2022-12-10    [abs](http://arxiv.org/abs/2212.05337v1) [paper-pdf](http://arxiv.org/pdf/2212.05337v1)

**Authors**: Dennis Gross, Thiago D. Simao, Nils Jansen, Guillermo A. Perez

**Abstract**: Deep Reinforcement Learning (RL) agents are susceptible to adversarial noise in their observations that can mislead their policies and decrease their performance. However, an adversary may be interested not only in decreasing the reward, but also in modifying specific temporal logic properties of the policy. This paper presents a metric that measures the exact impact of adversarial attacks against such properties. We use this metric to craft optimal adversarial attacks. Furthermore, we introduce a model checking method that allows us to verify the robustness of RL policies against adversarial attacks. Our empirical analysis confirms (1) the quality of our metric to craft adversarial attacks against temporal logic properties, and (2) that we are able to concisely assess a system's robustness against attacks.

摘要: 深度强化学习(RL)代理在其观察中容易受到对抗性噪声的影响，这可能会误导其策略并降低其性能。然而，对手可能不仅对减少奖励感兴趣，而且还对修改策略的特定时间逻辑属性感兴趣。本文提出了一种度量对抗性攻击对此类属性的确切影响的度量方法。我们使用这一指标来设计最优的对抗性攻击。此外，我们引入了一种模型检测方法，该方法允许我们验证RL策略对对手攻击的健壮性。我们的经验分析证实了(1)我们的度量标准的质量，可以针对时态逻辑特性进行敌意攻击，(2)我们能够简明地评估系统对攻击的健壮性。



## **29. UPTON: Unattributable Authorship Text via Data Poisoning**

Upton：数据中毒导致的不明作者身份文本 cs.CY

**SubmitDate**: 2022-12-10    [abs](http://arxiv.org/abs/2211.09717v2) [paper-pdf](http://arxiv.org/pdf/2211.09717v2)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.

摘要: 在彭博社、《卫报》、《西部日报》的观点专栏等网络媒体上，有抱负的作家出于各种原因发布自己的作品，经常自豪地打开自己的名字。然而，可能会发生这样的作者想要在其他场所匿名或以化名(例如，活动家、告密者)写作的情况。然而，如果攻击者已经基于来自这些平台的作品构建了准确的作者归属(AA)模型，则可以将匿名作品归因于已知的作者。因此，在这项工作中，我们提出了一个问题：是否可以让意见分享平台等开放空间中的文字和文本T无法归因于T，从而使从T训练的AA模型无法很好地归属作者？针对这一问题，我们提出了一种新的解决方案Upton，它利用文本数据毒化方法来干扰AA模型的训练过程。厄普顿使用数据中毒来破坏只有在训练样本中才有的作者特征，并试图使已发布的文本数据在深层神经元网络上无法学习。不同于以往的混淆工作，它使用对抗性攻击来修改测试样本，误导AA模型；后门工作，在测试和训练样本中都使用触发词，只有在触发词出现时才改变模型输出。然后，使用四个作者的数据集(例如，IMDb10，IMDb64，Enron和WJO)，我们提供了经验验证，其中：(1)Upton能够通过精心设计的目标选择方法将测试准确率降低到约30%。(2)Upton中毒能够保留大部分原始语义。CLEAN文本和Upton中毒文本之间的BERTSCORE均大于0.95。这个数字非常接近1.00，这意味着没有语义变化。(3)厄普顿对拼写纠正系统也很感兴趣。



## **30. Understanding and Combating Robust Overfitting via Input Loss Landscape Analysis and Regularization**

通过输入损失景观分析和正则化理解和抗击稳健过拟合 cs.LG

published in journal Pattern Recognition:  https://www.sciencedirect.com/science/article/pii/S0031320322007087?via%3Dihub

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04985v1) [paper-pdf](http://arxiv.org/pdf/2212.04985v1)

**Authors**: Lin Li, Michael Spratling

**Abstract**: Adversarial training is widely used to improve the robustness of deep neural networks to adversarial attack. However, adversarial training is prone to overfitting, and the cause is far from clear. This work sheds light on the mechanisms underlying overfitting through analyzing the loss landscape w.r.t. the input. We find that robust overfitting results from standard training, specifically the minimization of the clean loss, and can be mitigated by regularization of the loss gradients. Moreover, we find that robust overfitting turns severer during adversarial training partially because the gradient regularization effect of adversarial training becomes weaker due to the increase in the loss landscapes curvature. To improve robust generalization, we propose a new regularizer to smooth the loss landscape by penalizing the weighted logits variation along the adversarial direction. Our method significantly mitigates robust overfitting and achieves the highest robustness and efficiency compared to similar previous methods. Code is available at https://github.com/TreeLLi/Combating-RO-AdvLC.

摘要: 对抗性训练被广泛用于提高深度神经网络对对抗性攻击的稳健性。然而，对抗性训练容易出现过度适应的情况，原因还远不清楚。这项工作通过分析W.r.t.的损失情况，揭示了过度拟合的潜在机制。输入。我们发现，稳健的过拟合来自标准训练，特别是净损失的最小化，并且可以通过正则化损失梯度来减轻。此外，我们发现，在对抗性训练中，稳健过适应变得更加严重，部分原因是由于损失景观曲率的增加，对抗性训练的梯度正则化效果变得更弱。为了改进鲁棒性泛化，我们提出了一种新的正则化方法，通过惩罚沿对抗方向的加权Logits变化来平滑损失情况。与以前的同类方法相比，我们的方法显著地减轻了稳健过拟合度，并获得了最高的稳健性和效率。代码可在https://github.com/TreeLLi/Combating-RO-AdvLC.上找到



## **31. Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**

基于随机梯度阈值的快速显著引导混合算法 cs.CV

Accepted Long paper at 2nd Practical-DL Workshop at AAAI 2023

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04875v1) [paper-pdf](http://arxiv.org/pdf/2212.04875v1)

**Authors**: Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang

**Abstract**: Mix-up training approaches have proven to be effective in improving the generalization ability of Deep Neural Networks. Over the years, the research community expands mix-up methods into two directions, with extensive efforts to improve saliency-guided procedures but minimal focus on the arbitrary path, leaving the randomization domain unexplored. In this paper, inspired by the superior qualities of each direction over one another, we introduce a novel method that lies at the junction of the two routes. By combining the best elements of randomness and saliency utilization, our method balances speed, simplicity, and accuracy. We name our method R-Mix following the concept of "Random Mix-up". We demonstrate its effectiveness in generalization, weakly supervised object localization, calibration, and robustness to adversarial attacks. Finally, in order to address the question of whether there exists a better decision protocol, we train a Reinforcement Learning agent that decides the mix-up policies based on the classifier's performance, reducing dependency on human-designed objectives and hyperparameter tuning. Extensive experiments further show that the agent is capable of performing at the cutting-edge level, laying the foundation for a fully automatic mix-up. Our code is released at [https://github.com/minhlong94/Random-Mixup].

摘要: 混合训练方法已被证明是提高深度神经网络泛化能力的有效方法。多年来，研究界将混淆方法扩展到两个方向，广泛努力改进显著引导程序，但对任意路径的关注很少，留下了随机化领域的未探索。在这篇文章中，我们受到每个方向相对于另一个方向的优越性质的启发，提出了一种位于两条路线交界处的新方法。通过结合随机性和显着性利用的最佳元素，我们的方法平衡了速度、简单性和准确性。我们根据“随机混合”的概念将我们的方法命名为R-Mix。我们证明了它在泛化、弱监督目标定位、校准和对对手攻击的稳健性方面的有效性。最后，为了解决是否存在更好的决策协议的问题，我们训练了一个强化学习代理，它根据分类器的性能来决定混合策略，减少了对人为设计目标和超参数调整的依赖。广泛的实验进一步表明，该试剂能够在尖端水平上发挥作用，为全自动混合奠定了基础。我们的代码在[https://github.com/minhlong94/Random-Mixup].]上发布



## **32. Unfooling Perturbation-Based Post Hoc Explainers**

基于非愚弄扰动的帖子随机解说器 cs.AI

Accepted to AAAI-23. 9 pages (not including references and  supplemental)

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2205.14772v2) [paper-pdf](http://arxiv.org/pdf/2205.14772v2)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstract**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.

摘要: 人工智能(AI)的巨大进步吸引了医生、贷款人、法官和其他专业人士的兴趣。尽管这些事关重大的决策者对这项技术持乐观态度，但那些熟悉人工智能系统的人对其决策过程缺乏透明度持谨慎态度。基于扰动的后自组织解释器提供了一种模型不可知的方法来解释这些系统，而只需要查询级别的访问。然而，最近的研究表明，这些解释程序可能会被相反的人愚弄。这一发现对审计师、监管者和其他哨兵产生了不利影响。考虑到这一点，几个自然的问题就产生了--我们如何审计这些黑匣子系统？我们如何确定被审计人是真诚地遵守审计的？在这项工作中，我们严格地形式化了这个问题，并设计了一个防御对基于扰动的解释器的敌意攻击。在新的条件异常检测方法KNN-CAD的辅助下，我们提出了针对这些攻击的检测(CAD-检测)和防御(CAD-防御)算法。我们证明，我们的方法成功地检测到黑盒系统是否恶意地隐藏了其决策过程，并缓解了流行的解释程序LIME和Shap对真实数据的恶意攻击。



## **33. Robust Graph Representation Learning via Predictive Coding**

基于预测编码的稳健图表示学习 cs.LG

27 Pages, 31 Figures

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2212.04656v1) [paper-pdf](http://arxiv.org/pdf/2212.04656v1)

**Authors**: Billy Byiringiro, Tommaso Salvatori, Thomas Lukasiewicz

**Abstract**: Predictive coding is a message-passing framework initially developed to model information processing in the brain, and now also topic of research in machine learning due to some interesting properties. One of such properties is the natural ability of generative models to learn robust representations thanks to their peculiar credit assignment rule, that allows neural activities to converge to a solution before updating the synaptic weights. Graph neural networks are also message-passing models, which have recently shown outstanding results in diverse types of tasks in machine learning, providing interdisciplinary state-of-the-art performance on structured data. However, they are vulnerable to imperceptible adversarial attacks, and unfit for out-of-distribution generalization. In this work, we address this by building models that have the same structure of popular graph neural network architectures, but rely on the message-passing rule of predictive coding. Through an extensive set of experiments, we show that the proposed models are (i) comparable to standard ones in terms of performance in both inductive and transductive tasks, (ii) better calibrated, and (iii) robust against multiple kinds of adversarial attacks.

摘要: 预测编码是一种消息传递框架，最初是为了模拟大脑中的信息处理而开发的，由于一些有趣的特性，现在也是机器学习的研究主题。这些特性之一是生成模型学习稳健表示的自然能力，这要归功于它们独特的信用分配规则，该规则允许神经活动在更新突触权重之前收敛到一个解。图形神经网络也是消息传递模型，它最近在机器学习的不同类型的任务中显示出突出的结果，在结构化数据上提供了跨学科的最先进的性能。然而，它们很容易受到不可察觉的对手攻击，不适合于分布外的泛化。在这项工作中，我们通过建立模型来解决这个问题，这些模型具有与流行的图神经网络结构相同的结构，但依赖于预测编码的消息传递规则。通过大量的实验，我们发现所提出的模型(I)在归纳和引导任务中的性能方面与标准模型相当，(Ii)更好地校准，(Iii)对多种类型的对手攻击具有较强的鲁棒性。



## **34. Effective and Imperceptible Adversarial Textual Attack via Multi-objectivization**

基于多对象化的有效隐蔽对抗性文本攻击 cs.CL

**SubmitDate**: 2022-12-09    [abs](http://arxiv.org/abs/2111.01528v3) [paper-pdf](http://arxiv.org/pdf/2111.01528v3)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstract**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also essential for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written texts, making them easily perceptible. In this work, we advocate leveraging multi-objectivization to address such issue. Specifically, we formulate the problem of crafting AEs as a multi-objective optimization problem, where the imperceptibility of attacks is considered as auxiliary objectives. Then, we propose a simple yet effective evolutionary algorithm, dubbed HydraText, to solve this problem. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves competitive attack success rates and better attack imperceptibility than the recently proposed attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written texts. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target model by adversarial training.

摘要: 对抗性文本攻击领域在过去几年中显著增长，其中通常被认为的目标是制作能够成功愚弄目标模型的对抗性示例(AE)。然而，攻击的不可感知性对于实际攻击者来说也是必不可少的，但以往的研究往往忽略了这一点。因此，精心制作的AEs往往与原始的人类书面文本在结构和语义上有明显的差异，使得它们很容易被察觉。在这项工作中，我们主张利用多对象化来解决这一问题。具体地说，我们将攻击的隐蔽性作为辅助目标，将攻击的隐蔽性问题描述为一个多目标优化问题。然后，我们提出了一个简单而有效的进化算法，称为HydraText，来解决这个问题。据我们所知，HydraText是目前唯一可以有效应用于基于分数和基于决策的攻击设置的方法。44237个实例的详尽实验表明，与最近提出的攻击方法相比，HydraText始终具有与之相当的攻击成功率和更好的攻击不可见性。一项人类评估研究还表明，HydraText制作的AEs与人类书写的文本更难区分。最后，这些训练引擎表现出良好的可移植性，通过对抗性训练可以显著提高目标模型的稳健性。



## **35. Traditional Classification Neural Networks are Good Generators: They are Competitive with DDPMs and GANs**

传统的分类神经网络是很好的生成器：它们与DDPM和GANS具有竞争力 cs.CV

This paper has 29 pages with 22 figures, including rich supplementary  information. Project page is at  \url{https://classifier-as-generator.github.io/}

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2211.14794v2) [paper-pdf](http://arxiv.org/pdf/2211.14794v2)

**Authors**: Guangrun Wang, Philip H. S. Torr

**Abstract**: Classifiers and generators have long been separated. We break down this separation and showcase that conventional neural network classifiers can generate high-quality images of a large number of categories, being comparable to the state-of-the-art generative models (e.g., DDPMs and GANs). We achieve this by computing the partial derivative of the classification loss function with respect to the input to optimize the input to produce an image. Since it is widely known that directly optimizing the inputs is similar to targeted adversarial attacks incapable of generating human-meaningful images, we propose a mask-based stochastic reconstruction module to make the gradients semantic-aware to synthesize plausible images. We further propose a progressive-resolution technique to guarantee fidelity, which produces photorealistic images. Furthermore, we introduce a distance metric loss and a non-trivial distribution loss to ensure classification neural networks can synthesize diverse and high-fidelity images. Using traditional neural network classifiers, we can generate good-quality images of 256$\times$256 resolution on ImageNet. Intriguingly, our method is also applicable to text-to-image generation by regarding image-text foundation models as generalized classifiers.   Proving that classifiers have learned the data distribution and are ready for image generation has far-reaching implications, for classifiers are much easier to train than generative models like DDPMs and GANs. We don't even need to train classification models because tons of public ones are available for download. Also, this holds great potential for the interpretability and robustness of classifiers. Project page is at \url{https://classifier-as-generator.github.io/}.

摘要: 分类器和生成器长期以来一直是分开的。我们打破了这种分离，并展示了传统的神经网络分类器可以生成大量类别的高质量图像，可与最先进的生成模型(例如，DDPM和GAN)相媲美。我们通过计算分类损失函数相对于输入的偏导数来实现这一点，以优化输入以产生图像。由于众所周知，直接优化输入类似于无法生成对人类有意义的图像的定向对抗性攻击，我们提出了一种基于掩模的随机重建模型，使梯度能够感知语义，从而合成可信图像。我们进一步提出了一种渐进分辨率技术来保证保真度，从而产生照片级真实感图像。此外，我们还引入了距离度量损失和非平凡分布损失，以确保分类神经网络能够合成各种高保真图像。使用传统的神经网络分类器，我们可以在ImageNet上生成256美元\x 256美元分辨率的高质量图像。有趣的是，我们的方法也适用于文本到图像的生成，因为我们将图像-文本基础模型视为广义分类器。证明分类器已经学习了数据分布并准备好生成图像具有深远的意义，因为分类器比DDPM和Gans等生成模型更容易训练。我们甚至不需要训练分类模型，因为有大量的公共模型可供下载。此外，这对分类器的可解释性和健壮性具有很大的潜力。项目页面位于\url{https://classifier-as-generator.github.io/}.



## **36. Universal codes in the shared-randomness model for channels with general distortion capabilities**

具有一般失真能力的信道的共享随机性模型中的通用码 cs.IT

Removed the mentioning of online matching, which is not used here

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2007.02330v5) [paper-pdf](http://arxiv.org/pdf/2007.02330v5)

**Authors**: Bruno Bauwens, Marius Zimand

**Abstract**: We put forth new models for universal channel coding. Unlike standard codes which are designed for a specific type of channel, our most general universal code makes communication resilient on every channel, provided the noise level is below the tolerated bound, where the noise level t of a channel is the logarithm of its ambiguity (the maximum number of strings that can be distorted into a given one). The other more restricted universal codes still work for large classes of natural channels. In a universal code, encoding is channel-independent, but the decoding function knows the type of channel. We allow the encoding and the decoding functions to share randomness, which is unavailable to the channel. There are two scenarios for the type of attack that a channel can perform. In the oblivious scenario, codewords belong to an additive group and the channel distorts a codeword by adding a vector from a fixed set. The selection is based on the message and the encoding function, but not on the codeword. In the Hamming scenario, the channel knows the codeword and is fully adversarial. For a universal code, there are two parameters of interest: the rate, which is the ratio between the message length k and the codeword length n, and the number of shared random bits. We show the existence in both scenarios of universal codes with rate 1-t/n - o(1), which is optimal modulo the o(1) term. The number of shared random bits is O(log n) in the oblivious scenario, and O(n) in the Hamming scenario, which, for typical values of the noise level, we show to be optimal, modulo the constant hidden in the O() notation. In both scenarios, the universal encoding is done in time polynomial in n, but the channel-dependent decoding procedures are in general not efficient. For some weaker classes of channels we construct universal codes with polynomial-time encoding and decoding.

摘要: 我们提出了通用信道编码的新模型。与为特定类型的通道设计的标准代码不同，我们最通用的代码使通信在每个通道上都具有弹性，前提是噪声水平低于可容忍的界限，其中通道的噪声水平t是其模糊性的对数(可失真为给定字符串的最大数量)。其他更受限制的通用代码仍然适用于大类自然频道。在通用编码中，编码是独立于信道的，但解码功能知道信道的类型。我们允许编码和解码功能共享信道不可用的随机性。对于通道可以执行的攻击类型，有两种情况。在不经意的情况下，码字属于加法组，并且信道通过添加来自固定集合的矢量来扭曲码字。选择是基于消息和编码功能，而不是基于码字。在汉明方案中，信道知道码字并且是完全对抗性的。对于通用码，有两个感兴趣的参数：速率，其是消息长度k和码字长度n之间的比率，以及共享随机比特的数量。我们证明了在这两种情况下都存在码率为1-t/n-o(1)的通用码，它是模0(1)项的最优模。在不经意的情况下，共享随机比特数为O(Logn)，在汉明情况下，共享随机比特数为O(N)，对于噪声电平的典型值，我们证明这是最优的，对隐藏在O()记号中的常量取模。在这两种情况下，通用编码以n中的时间多项式进行，但依赖于信道的解码过程通常效率不高。对于一些较弱的信道类，我们构造了具有多项式时间编码和译码的通用码。



## **37. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

用于稳健心电分类的解相关网络结构 cs.LG

16 pages, 6 figures

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2207.09031v2) [paper-pdf](http://arxiv.org/pdf/2207.09031v2)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstract**: Artificial intelligence has made great progress in medical data analysis, but the lack of robustness and trustworthiness has kept these methods from being widely deployed. As it is not possible to train networks that are accurate in all situations, models must recognize situations where they cannot operate confidently. Bayesian deep learning methods sample the model parameter space to estimate uncertainty, but these parameters are often subject to the same vulnerabilities, which can be exploited by adversarial attacks. We propose a novel ensemble approach based on feature decorrelation and Fourier partitioning for teaching networks diverse complementary features, reducing the chance of perturbation-based fooling. We test our approach on electrocardiogram classification, demonstrating superior accuracy confidence measurement, on a variety of adversarial attacks. For example, on our ensemble trained with both decorrelation and Fourier partitioning scored a 50.18% inference accuracy and 48.01% uncertainty accuracy (area under the curve) on {\epsilon} = 50 projected gradient descent attacks, while a conventionally trained ensemble scored 21.1% and 30.31% on these metrics respectively. Our approach does not require expensive optimization with adversarial samples and can be scaled to large problems. These methods can easily be applied to other tasks for more robust and trustworthy models.

摘要: 人工智能在医疗数据分析方面取得了很大进展，但缺乏健壮性和可信性，阻碍了这些方法的广泛部署。由于不可能训练出在所有情况下都准确的网络，模型必须认识到它们不能自信地运行的情况。贝叶斯深度学习方法对模型参数空间进行采样以估计不确定性，但这些参数经常受到相同的漏洞的影响，这可能被对抗性攻击所利用。我们提出了一种新的基于特征去相关和傅立叶划分的集成方法，用于训练网络中不同的互补特征，减少了基于扰动的愚弄的机会。我们在心电图分类上测试了我们的方法，在各种对抗性攻击上展示了优越的准确性和置信度测量。例如，在我们用去相关和傅立叶划分训练的集成上，在{\epsilon}=50个投影梯度下降攻击上，推理准确率和不确定性准确率(曲线下面积)分别为50.18%和48.01%，而常规训练的集成在这些指标上的得分分别为21.1%和30.31%。我们的方法不需要使用对抗性样本进行昂贵的优化，并且可以扩展到大型问题。这些方法可以很容易地应用于其他任务，以获得更健壮和可靠的模型。



## **38. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

Garnet：强健可扩展图神经网络的降阶拓扑学习 cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2201.12741v5) [paper-pdf](http://arxiv.org/pdf/2201.12741v5)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.

摘要: 图形神经网络(GNN)已被越来越多地应用于涉及非欧几里德数据学习的各种应用中。然而，最近的研究表明，GNN容易受到图的对抗性攻击。虽然有几种防御方法可以通过消除敌对组件来提高GNN的健壮性，但它们也可能损害有助于GNN训练的底层干净的图形结构。此外，这些防御模型中很少有能够扩展到大型图形的，因为它们的计算复杂性和内存使用量很高。在本文中，我们提出了Garnet，一种可伸缩的谱方法来提高GNN模型的对抗健壮性。Garnet First利用加权谱嵌入来构造基图，该基图不仅能抵抗敌方攻击，而且还包含GNN训练所需的关键(干净)图结构。接下来，Garnet基于概率图模型，通过剪枝额外的非关键边来进一步精化基图。石榴石已经在各种数据集上进行了评估，包括一个包含数百万个节点的大型图表。我们的大量实验结果表明，与现有的GNN(防御)模型相比，Garnet的对抗准确率提高了13.27%，运行时加速比提高了14.7倍。



## **39. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

RoVISQ：通过对基于深度学习的视频压缩进行对抗性攻击来降低视频服务质量 cs.CV

Accepted at NDSS 2023

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2203.10183v3) [paper-pdf](http://arxiv.org/pdf/2203.10183v3)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstract**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion ($\textit{R}$-$\textit{D}$) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to $\sim$ 2.4$\times$ while achieving over 90$\%$ attack success rate on a downstream classifier. Our user study further demonstrates the effect of RoVISQ attacks on users' QoE.

摘要: 视频压缩在视频流和分类系统中起着至关重要的作用，它在给定的带宽预算下最大化最终用户的体验质量(QOE)。本文首次对基于深度学习的视频压缩和下行分类系统的敌意攻击进行了系统的研究。我们的攻击框架RoVISQ通过操纵视频压缩模型的率失真关系($\textit{R}$-$\textit{D}$)来实现以下一个或两个目标：(1)增加网络带宽；(2)降低最终用户的视频质量。我们进一步制定了针对下游视频分类服务的定向和非定向攻击的新目标。最后，我们设计了一种输入不变的扰动，该扰动普遍地扰乱了视频压缩和分类系统的实时。与之前提出的针对视频分类的攻击不同，我们的对抗性扰动最先经受住了压缩。我们经验地展示了RoVISQ攻击对各种防御措施的弹性，即对抗性训练、视频去噪和JPEG压缩。我们在不同视频数据集上的大量实验结果表明，RoVISQ攻击使峰值信噪比下降5.6dB，比特率下降2.4倍，而在下游分类器上的攻击成功率超过90美元。我们的用户研究进一步证明了RoVISQ攻击对用户QOE的影响。



## **40. Contrastive Weighted Learning for Near-Infrared Gaze Estimation**

基于对比加权学习的近红外凝视估计 cs.CV

There were deficiencies in the experiment process for validation

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2211.03073v2) [paper-pdf](http://arxiv.org/pdf/2211.03073v2)

**Authors**: Adam Lee

**Abstract**: Appearance-based gaze estimation has been very successful with the use of deep learning. Many following works improved domain generalization for gaze estimation. However, even though there has been much progress in domain generalization for gaze estimation, most of the recent work have been focused on cross-dataset performance -- accounting for different distributions in illuminations, head pose, and lighting. Although improving gaze estimation in different distributions of RGB images is important, near-infrared image based gaze estimation is also critical for gaze estimation in dark settings. Also there are inherent limitations relying solely on supervised learning for regression tasks. This paper contributes to solving these problems and proposes GazeCWL, a novel framework for gaze estimation with near-infrared images using contrastive learning. This leverages adversarial attack techniques for data augmentation and a novel contrastive loss function specifically for regression tasks that effectively clusters the features of different samples in the latent space. Our model outperforms previous domain generalization models in infrared image based gaze estimation and outperforms the baseline by 45.6\% while improving the state-of-the-art by 8.6\%, we demonstrate the efficacy of our method.

摘要: 随着深度学习的使用，基于外表的凝视估计已经非常成功。随后的许多工作改进了视线估计的领域泛化。然而，尽管在凝视估计的领域泛化方面已经有了很大的进展，但最近的大部分工作都集中在跨数据集性能上--考虑了光照、头部姿势和光照的不同分布。虽然在不同分布的RGB图像中改进凝视估计是重要的，但基于近红外图像的凝视估计对于黑暗环境下的凝视估计也是至关重要的。此外，回归任务仅依靠有监督的学习也存在固有的局限性。为了解决这些问题，本文提出了一种新的基于对比学习的近红外图像凝视估计框架GazeCWL。该算法利用对抗性攻击技术进行数据扩充，并针对回归任务提出了一种新的对比损失函数，该函数能有效地将不同样本的特征在潜在空间中聚类。我们的模型在基于红外图像的视线估计中优于以往的域泛化模型，并且比基线提高了45.6\%，同时提高了8.6%，验证了我们的方法的有效性。



## **41. Targeted Adversarial Attacks against Neural Network Trajectory Predictors**

针对神经网络轨迹预测器的定向对抗性攻击 cs.LG

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2212.04138v1) [paper-pdf](http://arxiv.org/pdf/2212.04138v1)

**Authors**: Kaiyuan Tan, Jun Wang, Yiannis Kantaros

**Abstract**: Trajectory prediction is an integral component of modern autonomous systems as it allows for envisioning future intentions of nearby moving agents. Due to the lack of other agents' dynamics and control policies, deep neural network (DNN) models are often employed for trajectory forecasting tasks. Although there exists an extensive literature on improving the accuracy of these models, there is a very limited number of works studying their robustness against adversarially crafted input trajectories. To bridge this gap, in this paper, we propose a targeted adversarial attack against DNN models for trajectory forecasting tasks. We call the proposed attack TA4TP for Targeted adversarial Attack for Trajectory Prediction. Our approach generates adversarial input trajectories that are capable of fooling DNN models into predicting user-specified target/desired trajectories. Our attack relies on solving a nonlinear constrained optimization problem where the objective function captures the deviation of the predicted trajectory from a target one while the constraints model physical requirements that the adversarial input should satisfy. The latter ensures that the inputs look natural and they are safe to execute (e.g., they are close to nominal inputs and away from obstacles). We demonstrate the effectiveness of TA4TP on two state-of-the-art DNN models and two datasets. To the best of our knowledge, we propose the first targeted adversarial attack against DNN models used for trajectory forecasting.

摘要: 轨迹预测是现代自主系统不可或缺的组成部分，因为它允许预见附近移动代理的未来意图。由于缺乏其他智能体的动力学和控制策略，通常采用深度神经网络(DNN)模型进行轨迹预测。虽然存在大量关于提高这些模型的准确性的文献，但研究它们对相反的输入轨迹的稳健性的著作数量非常有限。为了弥补这一差距，在本文中，我们提出了一种针对DNN模型的针对轨迹预测任务的对抗性攻击。我们将所提出的攻击称为TA4TP，用于轨迹预测的定向对抗性攻击。我们的方法生成敌意输入轨迹，能够愚弄DNN模型预测用户指定的目标/期望轨迹。我们的攻击依赖于求解一个非线性约束优化问题，其中目标函数捕获预测轨迹与目标轨迹的偏差，而约束建模对手输入应满足的物理要求。后者确保投入看起来很自然，并且可以安全地执行(例如，它们接近名义投入，远离障碍物)。我们在两个最先进的DNN模型和两个数据集上验证了TA4TP的有效性。据我们所知，我们提出了第一个针对用于轨迹预测的DNN模型的有针对性的对抗性攻击。



## **42. ADMM based Distributed State Observer Design under Sparse Sensor Attacks**

稀疏传感器攻击下基于ADMM的分布式状态观测器设计 eess.SY

7 pages, 6 figures

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2209.06292v2) [paper-pdf](http://arxiv.org/pdf/2209.06292v2)

**Authors**: Vinaya Mary Prinse, Rachel Kalpana Kalaimani

**Abstract**: This paper considers the design of a distributed state-observer for discrete-time Linear Time-Invariant (LTI) systems in the presence of sensor attacks. We assume there is a network of observer nodes, communicating with each other over an undirected graph, each with partial measurements of the output corrupted by some adversarial attack. We address the case of sparse attacks where the attacker targets a small subset of sensors. An algorithm based on Alternating Direction Method of Multipliers (ADMM) is developed which provides an update law for each observer which ensures convergence of each observer node to the actual state asymptotically.

摘要: 研究了传感器攻击下离散线性定常系统的分布式状态观测器的设计问题。我们假设有一个观察者节点的网络，它们通过一个无向图相互通信，每个节点都有被一些敌意攻击破坏的输出的部分测量。我们解决了稀疏攻击的情况，其中攻击者的目标是一小部分传感器。提出了一种基于乘子交替方向法(ADMM)的算法，该算法为每个观测器提供一个更新律，保证每个观测器节点渐近收敛到实际状态。



## **43. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

基于图神经网络的任务和模型不可知的对抗攻击 cs.LG

To appear as a full paper in AAAI 2023

**SubmitDate**: 2022-12-08    [abs](http://arxiv.org/abs/2112.13267v3) [paper-pdf](http://arxiv.org/pdf/2112.13267v3)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Arnab Bhattacharya, Sayan Ranu

**Abstract**: Adversarial attacks on Graph Neural Networks (GNNs) reveal their security vulnerabilities, limiting their adoption in safety-critical applications. However, existing attack strategies rely on the knowledge of either the GNN model being used or the predictive task being attacked. Is this knowledge necessary? For example, a graph may be used for multiple downstream tasks unknown to a practical attacker. It is thus important to test the vulnerability of GNNs to adversarial perturbations in a model and task agnostic setting. In this work, we study this problem and show that GNNs remain vulnerable even when the downstream task and model are unknown. The proposed algorithm, TANDIS (Targeted Attack via Neighborhood DIStortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, TANDIS designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep Q-learning. Extensive experiments on real datasets and state-of-the-art models show that, on average, TANDIS is up to 50% more effective than state-of-the-art techniques, while being more than 1000 times faster.

摘要: 对图神经网络(GNN)的敌意攻击暴露了它们的安全漏洞，限制了它们在安全关键应用中的采用。然而，现有的攻击策略依赖于正在使用的GNN模型或被攻击的预测任务的知识。这方面的知识有必要吗？例如，图可能用于实际攻击者未知的多个下游任务。因此，重要的是在模型和任务不可知的情况下测试GNN对对抗性扰动的脆弱性。在这项工作中，我们研究了这个问题，并证明了即使在下游任务和模型未知的情况下，GNN仍然是脆弱的。本文提出的基于邻域失真的目标攻击算法TANDIS表明，节点邻域失真对预测性能的影响是有效的。虽然邻域失真是一个NP-Hard问题，但Tandis通过图同构网络和深度Q-学习的新组合设计了一个有效的启发式算法。在真实数据集和最先进的模型上进行的广泛实验表明，平均而言，tandis的效率比最先进的技术高出50%，同时速度快1000倍以上。



## **44. SwitchX: Gmin-Gmax Switching for Energy-Efficient and Robust Implementation of Binary Neural Networks on ReRAM Xbars**

SwitchX：GMIN-GMAX交换，在ReRAM Xbar上实现能效和健壮的二进制神经网络 cs.ET

Accepted to ACM Transactions on Design Automation of Electronic  Systems on 30 Nov 2022

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2011.14498v3) [paper-pdf](http://arxiv.org/pdf/2011.14498v3)

**Authors**: Abhiroop Bhattacharjee, Priyadarshini Panda

**Abstract**: Memristive crossbars can efficiently implement Binarized Neural Networks (BNNs) wherein the weights are stored in high-resistance states (HRS) and low-resistance states (LRS) of the synapses. We propose SwitchX mapping of BNN weights onto ReRAM crossbars such that the impact of crossbar non-idealities, that lead to degradation in computational accuracy, are minimized. Essentially, SwitchX maps the binary weights in such manner that a crossbar instance comprises of more HRS than LRS synapses. We find BNNs mapped onto crossbars with SwitchX to exhibit better robustness against adversarial attacks than the standard crossbar-mapped BNNs, the baseline. Finally, we combine SwitchX with state-aware training (that further increases the feasibility of HRS states during weight mapping) to boost the robustness of a BNN on hardware. We find that this approach yields stronger defense against adversarial attacks than adversarial training, a state-of-the-art software defense. We perform experiments on a VGG16 BNN with benchmark datasets (CIFAR-10, CIFAR-100 & TinyImagenet) and use Fast Gradient Sign Method and Projected Gradient Descent adversarial attacks. We show that SwitchX combined with state-aware training can yield upto ~35% improvements in clean accuracy and ~6-16% in adversarial accuracies against conventional BNNs. Furthermore, an important by-product of SwitchX mapping is increased crossbar power savings, owing to an increased proportion of HRS synapses, that is furthered with state-aware training. We obtain upto ~21-22% savings in crossbar power consumption for state-aware trained BNN mapped via SwitchX on 16x16 & 32x32 crossbars using the CIFAR-10 & CIFAR-100 datasets.

摘要: 记忆交叉开关可以有效地实现二值化神经网络(BNN)，其中权重存储在突触的高阻态(HRS)和低阻态(LRS)中。我们建议将BNN权重的SwitchX映射到ReRAM纵横杆上，以使导致计算精度降低的纵横杆非理想性的影响最小化。从本质上讲，SwitchX映射二进制权重的方式是，Crosbar实例包含的HRS多于LRS突触。我们发现，与标准的交叉开关映射的BNN相比，使用SwitchX映射到Crosbar上的BNN表现出更好的抗敌意攻击的健壮性。最后，我们将SwitchX与状态感知训练相结合(这进一步增加了权重映射过程中HRS状态的可行性)，以提高BNN在硬件上的健壮性。我们发现，这种方法比对抗性训练(一种最先进的软件防御)产生了更强的对抗性攻击防御。我们使用基准数据集(CIFAR-10、CIFAR-100和TinyImagenet)在VGG16 BNN上进行了实验，并使用快速梯度符号方法和投影梯度下降对抗性攻击。我们表明，SwitchX与状态感知训练相结合，可以在清理准确率上提高~35%，在对抗传统BNN的准确率上提高~6-16%。此外，SwitchX映射的一个重要副产品是由于HRS突触比例的增加，以及状态感知训练的进一步推进，增加了交叉开关的能量节省。使用CIFAR-10和CIFAR-100数据集，通过SwitchX映射到16x16和32x32纵横线上的状态感知训练型BNN，我们的纵横线功耗最多可节省约21%-22%。



## **45. Learning Polysemantic Spoof Trace: A Multi-Modal Disentanglement Network for Face Anti-spoofing**

学习多语义欺骗痕迹：一种面向人脸反欺骗的多模式解缠网络 cs.CV

Accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03943v1) [paper-pdf](http://arxiv.org/pdf/2212.03943v1)

**Authors**: Kaicheng Li, Hongyu Yang, Binghui Chen, Pengyu Li, Biao Wang, Di Huang

**Abstract**: Along with the widespread use of face recognition systems, their vulnerability has become highlighted. While existing face anti-spoofing methods can be generalized between attack types, generic solutions are still challenging due to the diversity of spoof characteristics. Recently, the spoof trace disentanglement framework has shown great potential for coping with both seen and unseen spoof scenarios, but the performance is largely restricted by the single-modal input. This paper focuses on this issue and presents a multi-modal disentanglement model which targetedly learns polysemantic spoof traces for more accurate and robust generic attack detection. In particular, based on the adversarial learning mechanism, a two-stream disentangling network is designed to estimate spoof patterns from the RGB and depth inputs, respectively. In this case, it captures complementary spoofing clues inhering in different attacks. Furthermore, a fusion module is exploited, which recalibrates both representations at multiple stages to promote the disentanglement in each individual modality. It then performs cross-modality aggregation to deliver a more comprehensive spoof trace representation for prediction. Extensive evaluations are conducted on multiple benchmarks, demonstrating that learning polysemantic spoof traces favorably contributes to anti-spoofing with more perceptible and interpretable results.

摘要: 随着人脸识别系统的广泛使用，它们的脆弱性也变得突出起来。虽然现有的人脸反欺骗方法可以在不同的攻击类型之间泛化，但由于欺骗特征的多样性，通用的解决方案仍然具有挑战性。近年来，欺骗跟踪解缠框架在处理看得见和看不见的欺骗场景方面显示出了巨大的潜力，但其性能在很大程度上受到单模式输入的限制。针对这一问题，本文提出了一种多模式解缠模型，该模型有针对性地学习多语义欺骗痕迹，以更准确和健壮地检测通用攻击。特别是，基于对抗性学习机制，设计了一个双流解缠网络，分别从RGB和深度输入估计欺骗模式。在这种情况下，它捕获了不同攻击中蕴含的补充欺骗线索。此外，还开发了一个融合模块，该模块在多个阶段重新校准两个表示，以促进每个单独通道的解缠。然后，它执行跨通道聚合以提供更全面的用于预测的欺骗跟踪表示。在多个基准上进行了广泛的评估，结果表明，学习多语义欺骗跟踪有助于反欺骗，获得更可感知和更可解释的结果。



## **46. Multiple Perturbation Attack: Attack Pixelwise Under Different $\ell_p$-norms For Better Adversarial Performance**

多重扰动攻击：在不同的$\ell_p$规范下进行像素攻击以获得更好的对抗性能 cs.CV

18 pages, 8 figures, 7 tables

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03069v2) [paper-pdf](http://arxiv.org/pdf/2212.03069v2)

**Authors**: Ngoc N. Tran, Anh Tuan Bui, Dinh Phung, Trung Le

**Abstract**: Adversarial machine learning has been both a major concern and a hot topic recently, especially with the ubiquitous use of deep neural networks in the current landscape. Adversarial attacks and defenses are usually likened to a cat-and-mouse game in which defenders and attackers evolve over the time. On one hand, the goal is to develop strong and robust deep networks that are resistant to malicious actors. On the other hand, in order to achieve that, we need to devise even stronger adversarial attacks to challenge these defense models. Most of existing attacks employs a single $\ell_p$ distance (commonly, $p\in\{1,2,\infty\}$) to define the concept of closeness and performs steepest gradient ascent w.r.t. this $p$-norm to update all pixels in an adversarial example in the same way. These $\ell_p$ attacks each has its own pros and cons; and there is no single attack that can successfully break through defense models that are robust against multiple $\ell_p$ norms simultaneously. Motivated by these observations, we come up with a natural approach: combining various $\ell_p$ gradient projections on a pixel level to achieve a joint adversarial perturbation. Specifically, we learn how to perturb each pixel to maximize the attack performance, while maintaining the overall visual imperceptibility of adversarial examples. Finally, through various experiments with standardized benchmarks, we show that our method outperforms most current strong attacks across state-of-the-art defense mechanisms, while retaining its ability to remain clean visually.

摘要: 对抗性机器学习近年来一直是一个重要的关注和热点问题，尤其是在当前深度神经网络的普遍使用下。对抗性的攻击和防御通常被比作猫和老鼠的游戏，其中防御者和攻击者随着时间的推移而演变。一方面，目标是开发强大而健壮的深层网络，抵御恶意行为者。另一方面，为了实现这一目标，我们需要设计出更强大的对抗性攻击来挑战这些防御模式。现有的攻击大多使用单个$ell_p$距离(通常是$p in{1，2，inty$)来定义贴近度的概念，并执行最陡的梯度上升w.r.t.此$p$-规范以相同的方式更新对抗性示例中的所有像素。这些$\ell_p$攻击各有优缺点；没有一种攻击可以成功突破同时对多个$ell_p$规范具有健壮性的防御模型。受这些观察结果的启发，我们提出了一种自然的方法：将不同的$\ell_p$梯度投影组合在一个像素级别上，以实现联合对抗性扰动。具体地说，我们学习了如何扰动每个像素以最大化攻击性能，同时保持对抗性例子的整体视觉不可感知性。最后，通过标准化基准的各种实验，我们表明我们的方法在保持视觉清洁的能力的同时，在最先进的防御机制上优于目前大多数的强攻击。



## **47. Universal Backdoor Attacks Detection via Adaptive Adversarial Probe**

基于自适应对抗性探测的通用后门攻击检测 cs.CV

8 pages, 8 figures

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2209.05244v3) [paper-pdf](http://arxiv.org/pdf/2209.05244v3)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstract**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor attacks detection. Most detection methods are designed to verify whether a model is infected with presumed types of backdoor attacks, yet the adversary is likely to generate diverse backdoor attacks in practice that are unforeseen to defenders, which challenge current detection strategies. In this paper, we focus on this more challenging scenario and propose a universal backdoor attacks detection method named Adaptive Adversarial Probe (A2P). Specifically, we posit that the challenge of universal backdoor attacks detection lies in the fact that different backdoor attacks often exhibit diverse characteristics in trigger patterns (i.e., sizes and transparencies). Therefore, our A2P adopts a global-to-local probing framework, which adversarially probes images with adaptive regions/budgets to fit various backdoor triggers of different sizes/transparencies. Regarding the probing region, we propose the attention-guided region generation strategy that generates region proposals with different sizes/locations based on the attention of the target model, since trigger regions often manifest higher model activation. Considering the attack budget, we introduce the box-to-sparsity scheduling that iteratively increases the perturbation budget from box to sparse constraint, so that we could better activate different latent backdoors with different transparencies. Extensive experiments on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins (+12%).

摘要: 大量证据表明，深度神经网络(DNN)很容易受到后门攻击，这推动了后门攻击检测的发展。大多数检测方法旨在验证模型是否感染了假定类型的后门攻击，但对手在实践中可能会产生防御者无法预见的各种后门攻击，这对当前的检测策略构成了挑战。在本文中，我们针对这种更具挑战性的场景，提出了一种通用的后门攻击检测方法--自适应对抗探测(A2P)。具体地说，我们假设通用后门攻击检测的挑战在于不同的后门攻击通常在触发模式(即大小和透明度)上表现出不同的特征。因此，我们的A2P采用了全局到局部探测框架，该框架相反地探测具有自适应区域/预算的图像，以适应不同大小/透明度的各种后门触发。对于探测区域，我们提出了注意力引导的区域生成策略，该策略根据目标模型的关注度生成不同大小/位置的区域建议，因为触发区域通常表现出较高的模型活跃度。考虑到攻击预算，我们引入了盒到稀疏调度，将扰动预算从盒子迭代增加到稀疏约束，以便更好地激活不同透明度的不同潜在后门。在多个数据集(CIFAR-10，GTSRB，Tiny-ImageNet)上的大量实验表明，我们的方法比最先进的基线方法有很大的优势(+12%)。



## **48. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

Leno：具有可学习噪声的对抗性鲁棒显著目标检测网络 cs.CV

8 pages, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2210.15392v2) [paper-pdf](http://arxiv.org/pdf/2210.15392v2)

**Authors**: He Wang, Lin Wan, He Tang

**Abstract**: Pixel-wise prediction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remarkable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust saliency (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected conditional random field (CRF). Different from ROSA that relies on various pre- and post-processings, this paper proposes a light-weight Learnable Noise (LeNo) to defend adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also on clean images, which contributes stronger robustness for SOD. Our code is available at https://github.com/ssecv/LeNo.

摘要: 基于深度神经网络的像素预测已成为显著目标检测的一种有效范例，并取得了显著的效果。然而，很少有SOD模型对人类视觉上不可察觉的对抗性攻击具有健壮性。该算法首先对预分块后的超像素进行置乱处理，然后利用稠密连接的条件随机场(CRF)对粗略显著图进行细化。不同于ROSA依赖于各种前后处理，本文提出了一种轻量级可学习噪声(Leno)来防御对SOD模型的敌意攻击。Leno保持了SOD模型在对抗性图像和干净图像上的准确性，以及推理速度。一般来说，LENO由简单的浅层噪声和噪声估计组成，分别嵌入到任意SOD网络的编码器和译码中。受人类视觉注意机制中心先验的启发，我们用十字形高斯分布对浅层噪声进行初始化，以更好地防御对手的攻击。所提出的噪声估计只需修改解码器的一个通道，而不是为后处理增加额外的网络组件。通过在最新的RGB和RGB-D SOD网络上进行深度监督噪声解耦训练，Leno不仅在对抗性图像上而且在干净图像上都优于以往的工作，这为SOD提供了更强的稳健性。我们的代码可以在https://github.com/ssecv/LeNo.上找到



## **49. Pre-trained Encoders in Self-Supervised Learning Improve Secure and Privacy-preserving Supervised Learning**

自我监督学习中的预训练编码器改进安全和隐私保护的监督学习 cs.CR

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2212.03334v1) [paper-pdf](http://arxiv.org/pdf/2212.03334v1)

**Authors**: Hongbin Liu, Wenjie Qu, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Classifiers in supervised learning have various security and privacy issues, e.g., 1) data poisoning attacks, backdoor attacks, and adversarial examples on the security side as well as 2) inference attacks and the right to be forgotten for the training data on the privacy side. Various secure and privacy-preserving supervised learning algorithms with formal guarantees have been proposed to address these issues. However, they suffer from various limitations such as accuracy loss, small certified security guarantees, and/or inefficiency. Self-supervised learning is an emerging technique to pre-train encoders using unlabeled data. Given a pre-trained encoder as a feature extractor, supervised learning can train a simple yet accurate classifier using a small amount of labeled training data. In this work, we perform the first systematic, principled measurement study to understand whether and when a pre-trained encoder can address the limitations of secure or privacy-preserving supervised learning algorithms. Our key findings are that a pre-trained encoder substantially improves 1) both accuracy under no attacks and certified security guarantees against data poisoning and backdoor attacks of state-of-the-art secure learning algorithms (i.e., bagging and KNN), 2) certified security guarantees of randomized smoothing against adversarial examples without sacrificing its accuracy under no attacks, 3) accuracy of differentially private classifiers, and 4) accuracy and/or efficiency of exact machine unlearning.

摘要: 有监督学习中的分类器存在各种各样的安全和隐私问题，例如：1)安全端的数据中毒攻击、后门攻击和敌意例子；2)隐私端的推理攻击和训练数据的遗忘权。各种形式保证的安全和隐私保护的监督学习算法已经被提出来解决这些问题。然而，它们受到各种限制，例如准确性损失、较小的认证安全保证和/或效率低下。自我监督学习是一种新兴的技术，可以使用未标记的数据来预先训练编码者。给定一个预先训练的编码器作为特征提取者，监督学习可以使用少量的标记训练数据来训练简单而准确的分类器。在这项工作中，我们进行了第一次系统的、原则性的测量研究，以了解预先训练的编码器是否以及何时可以解决安全或隐私保护的监督学习算法的限制。我们的主要发现是，预先训练的编码器显著提高了1)无攻击情况下的准确率，以及针对最先进的安全学习算法(即Bging和KNN)的数据中毒和后门攻击的认证安全保证，2)针对敌意示例的随机平滑的认证安全保证而不牺牲其在无攻击情况下的精度，3)差分私有分类器的准确性，以及4)精确机器遗忘的准确性和/或效率。



## **50. Robust Models are less Over-Confident**

稳健的模型不那么过度自信 cs.CV

accepted at NeurIPS 2022

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.05938v2) [paper-pdf](http://arxiv.org/pdf/2210.05938v2)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation

摘要: 尽管卷积神经网络(CNN)在许多计算机视觉任务的学术基准中取得了成功，但它们在现实世界中的应用仍然面临着根本性的挑战。这些悬而未决的问题之一是固有的健壮性不足，这一点从对抗性攻击的惊人有效性中可见一斑。目前的攻击方法能够通过向输入添加特定但少量的噪声来操纵网络的预测。反过来，对抗性训练(AT)的目的是通过将对抗性样本包括在训练集中来实现对此类攻击的健壮性，并且理想地实现更好的模型泛化能力。然而，对由此产生的超越对抗性稳健性的稳健性模型的深入分析仍然悬而未决。在这篇文章中，我们实证分析了各种对抗训练的模型，这些模型在面对最先进的攻击时获得了很高的稳健精度，我们发现AT有一个有趣的副作用：它导致模型对他们的决策不那么过度自信，即使是在干净的数据上也是如此。此外，我们对稳健模型的分析表明，不仅AT而且模型的构件(如激活函数和池化)对模型的预测置信度有很大的影响。数据与项目网站：https://github.com/GeJulia/robustness_confidences_evaluation



