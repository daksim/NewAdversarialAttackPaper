# Latest Adversarial Attack Papers
**update at 2022-08-17 10:03:11**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Relighting Against Face Recognition**

对抗人脸识别的对抗性重发 cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2108.07920v3)

**Authors**: Ruijun Gao, Qing Guo, Qian Zhang, Felix Juefei-Xu, Hongkai Yu, Wei Feng

**Abstracts**: Deep face recognition (FR) has achieved significantly high accuracy on several challenging datasets and fosters successful real-world applications, even showing high robustness to the illumination variation that is usually regarded as a main threat to the FR system. However, in the real world, illumination variation caused by diverse lighting conditions cannot be fully covered by the limited face dataset. In this paper, we study the threat of lighting against FR from a new angle, i.e., adversarial attack, and identify a new task, i.e., adversarial relighting. Given a face image, adversarial relighting aims to produce a naturally relighted counterpart while fooling the state-of-the-art deep FR methods. To this end, we first propose the physical model-based adversarial relighting attack (ARA) denoted as albedo-quotient-based adversarial relighting attack (AQ-ARA). It generates natural adversarial light under the physical lighting model and guidance of FR systems and synthesizes adversarially relighted face images. Moreover, we propose the auto-predictive adversarial relighting attack (AP-ARA) by training an adversarial relighting network (ARNet) to automatically predict the adversarial light in a one-step manner according to different input faces, allowing efficiency-sensitive applications. More importantly, we propose to transfer the above digital attacks to physical ARA (Phy-ARA) through a precise relighting device, making the estimated adversarial lighting condition reproducible in the real world. We validate our methods on three state-of-the-art deep FR methods, i.e., FaceNet, ArcFace, and CosFace, on two public datasets. The extensive and insightful results demonstrate our work can generate realistic adversarial relighted face images fooling FR easily, revealing the threat of specific light directions and strengths.

摘要: 深度人脸识别(FR)已经在几个具有挑战性的数据集上取得了显著的高精度，并促进了现实世界的成功应用，甚至对通常被视为FR系统主要威胁的光照变化表现出高度的稳健性。然而，在现实世界中，有限的人脸数据集不能完全覆盖由于光照条件的变化而引起的光照变化。本文从对抗性攻击这一新的角度研究了闪电对火箭弹的威胁，并提出了一种新的任务，即对抗性重发。给定脸部图像，对抗性重光旨在产生自然重光的对应物，同时愚弄最先进的深度FR方法。为此，我们首先提出了基于物理模型的对抗性重亮攻击(ARA)，称为基于反照率商的对抗性重亮攻击(AQ-ARA)。它在物理照明模型和FR系统的指导下产生自然的对抗性光，并合成对抗性重光的人脸图像。此外，我们提出了自动预测对抗性重光攻击(AP-ARA)，通过训练对抗性重光网络(ARNet)来根据不同的输入人脸一步自动预测对抗性光，从而允许对效率敏感的应用。更重要的是，我们建议通过精确的重新照明装置将上述数字攻击转移到物理ARA(Phy-ARA)，使估计的对抗性照明条件在现实世界中可重现。在两个公开的数据集上，我们在三种最先进的深度FR方法，即FaceNet，ArcFace和CosFace上对我们的方法进行了验证。广泛而有洞察力的结果表明，我们的工作可以生成现实的对抗性重光照人脸图像，轻松愚弄FR，揭示特定光方向和强度的威胁。



## **2. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2202.07568v3)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了StratDef，这是一个针对恶意软件检测领域定制的基于运动目标防御方法的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御措施来看，只有少数经过对抗性训练的模型提供了比只使用普通模型更好的保护，但仍然优于StratDef。



## **3. A Physical-World Adversarial Attack for 3D Face Recognition**

一种面向3D人脸识别的物理世界对抗性攻击 cs.CV

7 pages, 5 figures, Submit to AAAI 2023

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2205.13412v2)

**Authors**: Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao

**Abstracts**: The 3D face recognition has long been considered secure for its resistance to current physical adversarial attacks, like adversarial patches. However, this paper shows that a 3D face recognition system can be easily attacked, leading to evading and impersonation attacks. We are the first to propose a physically realizable attack for the 3D face recognition system, named structured light imaging attack (SLIA), which exploits the weakness of structured-light-based 3D scanning devices. SLIA utilizes the projector in the structured light imaging system to create adversarial illuminations to contaminate the reconstructed point cloud. Firstly, we propose a 3D transform-invariant loss function (3D-TI) to generate adversarial perturbations that are more robust to head movements. Then we integrate the 3D imaging process into the attack optimization, which minimizes the total pixel shifting of fringe patterns. We realize both dodging and impersonation attacks on a real-world 3D face recognition system. Our methods need fewer modifications on projected patterns compared with Chamfer and Chamfer+kNN-based methods and achieve average attack success rates of 0.47 (impersonation) and 0.89 (dodging). This paper exposes the insecurity of present structured light imaging technology and sheds light on designing secure 3D face recognition authentication systems.

摘要: 长期以来，3D人脸识别一直被认为是安全的，因为它能抵抗当前的物理对抗性攻击，比如对抗性补丁。然而，本文指出，3D人脸识别系统很容易受到攻击，从而导致规避和冒充攻击。我们首次提出了一种针对3D人脸识别系统的物理可实现的攻击，称为结构光成像攻击(SLIA)，它利用了基于结构光的3D扫描设备的弱点。SLIA利用结构光成像系统中的投影仪来产生对抗性照明来污染重建的点云。首先，我们提出了一种3D变换不变损失函数(3D-TI)来产生对抗性扰动，该扰动对头部运动具有更强的鲁棒性。然后，我们将3D成像过程整合到攻击优化中，使条纹图案的总像素漂移最小化。我们在一个真实的3D人脸识别系统上实现了躲避攻击和模仿攻击。与基于Chamfer和Chamfer+KNN的方法相比，我们的方法需要对投影模式进行更少的修改，并且获得了0.47(模仿)和0.89(躲避)的平均攻击成功率。本文揭示了目前结构光成像技术的不足，为设计安全的三维人脸识别认证系统提供了参考。



## **4. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2008.09312v2)

**Authors**: Shiliang Zuo

**Abstracts**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.

摘要: 我们考虑了一个随机多臂强盗问题，其中报酬服从对抗性腐败。我们提出了一种新的攻击策略，它利用UCB原理来拉动一些非最优目标臂$T-o(T)$次，累积代价可扩展到$\Sqrt{\log T}$，其中$T$是轮数。我们还证明了累积攻击代价的第一个下界。我们的下界与上界匹配，最高可达$\log\log T$因子，表明我们的攻击接近最优。



## **5. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2203.04713v3)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks.

摘要: 骨骼运动在人类活动识别(HAR)中得到了广泛的应用。最近，基于骨架的HAR在各种分类器和数据中发现了一个普遍的漏洞，需要缓解。为此，我们提出了第一种基于骨架的HAR黑盒防御方法。我们的方法的特点是对干净数据、对手和分类器进行全面的贝叶斯处理，导致(1)新的基于贝叶斯能量的稳健判别分类器的形成，(2)基于自然运动流形的新的对手采样方案，(3)新的训练后贝叶斯策略用于黑盒防御。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的骨架HAR分类器和数据集上展示了令人惊讶的和普遍的有效性。



## **6. CTI4AI: Threat Intelligence Generation and Sharing after Red Teaming AI Models**

CTI4AI：Red Teaming AI模型后威胁情报的生成和共享 cs.CR

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2208.07476v1)

**Authors**: Chuyen Nguyen, Caleb Morgan, Sudip Mittal

**Abstracts**: As the practicality of Artificial Intelligence (AI) and Machine Learning (ML) based techniques grow, there is an ever increasing threat of adversarial attacks. There is a need to red team this ecosystem to identify system vulnerabilities, potential threats, characterize properties that will enhance system robustness, and encourage the creation of effective defenses. A secondary need is to share this AI security threat intelligence between different stakeholders like, model developers, users, and AI/ML security professionals. In this paper, we create and describe a prototype system CTI4AI, to overcome the need to methodically identify and share AI/ML specific vulnerabilities and threat intelligence.

摘要: 随着人工智能(AI)和机器学习(ML)技术的实用化程度的提高，对手攻击的威胁越来越大。有必要对这一生态系统进行红色团队合作，以识别系统漏洞和潜在威胁，确定将增强系统健壮性的特性，并鼓励创建有效的防御措施。第二个需求是在不同的利益相关者之间共享这种AI安全威胁情报，比如模型开发人员、用户和AI/ML安全专业人员。在本文中，我们创建并描述了一个原型系统CTI4AI，以克服有条不紊地识别和共享AI/ML特定漏洞和威胁情报的需要。



## **7. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07316v1)

**Authors**: Yanran Chen, Steffen Eger

**Abstracts**: Recently proposed BERT-based evaluation metrics perform well on standard evaluation benchmarks but are vulnerable to adversarial attacks, e.g., relating to factuality errors. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when we combine existing metrics with our NLI metrics, we obtain both higher adversarial robustness (+20% to +30%) and higher quality metrics as measured on standard benchmarks (+5% to +25%).

摘要: 最近提出的基于BERT的评估指标在标准评估基准上表现良好，但容易受到敌意攻击，例如与真实性错误有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当我们将现有指标与我们的NLI指标相结合时，我们获得了更高的对抗性健壮性(+20%至+30%)和标准基准测试的更高质量指标(+5%至+25%)。



## **8. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2202.12232v3)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。在本文中，我们给出了当训练算法提供$\epsilon$-DP或$(\epsilon，\Delta)$-DP时，MI对手的正确率(即攻击精度)的一个更严格的界。我们的界提供了一种新的隐私放大方案的设计，其中有效的训练集在训练开始之前从较大的集合中被亚采样，以极大地降低对MI准确率的界。因此，我们的方案允许DP用户在训练他们的模型时采用更宽松的DP保证来限制任何MI对手的成功；这确保了模型的准确性较少地受到隐私保证的影响。最后，我们讨论了我们的MI界在机器遗忘领域的意义。



## **9. Man-in-the-Middle Attack against Object Detection Systems**

针对目标检测系统的中间人攻击 cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07174v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Is deep learning secure for robots? As embedded systems have access to more powerful CPUs and GPUs, deep-learning-enabled object detection systems become pervasive in robotic applications. Meanwhile, prior research unveils that deep learning models are vulnerable to adversarial attacks. Does this put real-world robots at threat? Our research borrows the idea of the Main-in-the-Middle attack from Cryptography to attack an object detection system. Our experimental results prove that we can generate a strong Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. Our findings raise a serious concern over the applications of deep learning models in safety-critical systems such as autonomous driving.

摘要: 深度学习对机器人来说安全吗？随着嵌入式系统能够获得更强大的CPU和GPU，支持深度学习的目标检测系统在机器人应用中变得普遍。与此同时，先前的研究表明，深度学习模型很容易受到对手的攻击。这会对现实世界的机器人构成威胁吗？我们的研究借用了密码学中的中间主攻击的思想来攻击目标检测系统。我们的实验结果证明，我们可以在一分钟内产生一个强的通用对抗扰动(UAP)，然后利用该扰动通过中间人攻击来攻击检测系统。我们的发现引发了人们对深度学习模型在自动驾驶等安全关键系统中的应用的严重担忧。



## **10. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2204.09803v3)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 图卷积网络(GCNS)容易受到微小的敌意扰动，这是一种严重的威胁，在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，称为图通用对抗防御(GARD)。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显着提高了几个已建立的GCN对多个对手攻击的稳健性，并且远远超过了最先进的防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **11. A Multi-objective Memetic Algorithm for Auto Adversarial Attack Optimization Design**

自动对抗性攻击优化设计的多目标Memtic算法 cs.CV

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06984v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstracts**: The phenomenon of adversarial examples has been revealed in variant scenarios. Recent studies show that well-designed adversarial defense strategies can improve the robustness of deep learning models against adversarial examples. However, with the rapid development of defense technologies, it also tends to be more difficult to evaluate the robustness of the defensed model due to the weak performance of existing manually designed adversarial attacks. To address the challenge, given the defensed model, the efficient adversarial attack with less computational burden and lower robust accuracy is needed to be further exploited. Therefore, we propose a multi-objective memetic algorithm for auto adversarial attack optimization design, which realizes the automatical search for the near-optimal adversarial attack towards defensed models. Firstly, the more general mathematical model of auto adversarial attack optimization design is constructed, where the search space includes not only the attacker operations, magnitude, iteration number, and loss functions but also the connection ways of multiple adversarial attacks. In addition, we develop a multi-objective memetic algorithm combining NSGA-II and local search to solve the optimization problem. Finally, to decrease the evaluation cost during the search, we propose a representative data selection strategy based on the sorting of cross entropy loss values of each images output by models. Experiments on CIFAR10, CIFAR100, and ImageNet datasets show the effectiveness of our proposed method.

摘要: 对抗性例子的现象在不同的情景中被揭示出来。最近的研究表明，设计好的对抗性防御策略可以提高深度学习模型对对抗性例子的稳健性。然而，随着防御技术的快速发展，由于现有人工设计的对抗性攻击性能较弱，评估防御模型的稳健性也变得更加困难。为了应对这一挑战，在防御模型的情况下，需要进一步开发计算负担较小、鲁棒性较低的高效对抗性攻击。为此，本文提出了一种自动对抗性攻击优化设计的多目标迷因算法，实现了对防御模型的近优对抗性攻击的自动搜索。首先，建立了更一般的自动对抗攻击优化设计数学模型，该模型的搜索空间不仅包括攻击操作、规模、迭代次数和损失函数，还包括多个对抗攻击的连接方式。此外，我们还提出了一种结合NSGA-II和局部搜索的多目标模因算法来解决该优化问题。最后，为了降低搜索过程中的评价代价，提出了一种基于模型对每幅图像输出交叉熵损失值排序的代表性数据选择策略。在CIFAR10、CIFAR100和ImageNet数据集上的实验表明了该方法的有效性。



## **12. InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee**

隐形Tee：带Te的人跟踪系统的角度不可知隐形 cs.CV

12 pages, 10 figures and the ICANN 2022 accpeted paper

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06962v1)

**Authors**: Yaxian Li, Bingqing Zhang, Guoping Zhao, Mingyu Zhang, Jiajun Liu, Ziwei Wang, Jirong Wen

**Abstracts**: After a survey for person-tracking system-induced privacy concerns, we propose a black-box adversarial attack method on state-of-the-art human detection models called InvisibiliTee. The method learns printable adversarial patterns for T-shirts that cloak wearers in the physical world in front of person-tracking systems. We design an angle-agnostic learning scheme which utilizes segmentation of the fashion dataset and a geometric warping process so the adversarial patterns generated are effective in fooling person detectors from all camera angles and for unseen black-box detection models. Empirical results in both digital and physical environments show that with the InvisibiliTee on, person-tracking systems' ability to detect the wearer drops significantly.

摘要: 在调查了个人跟踪系统引起的隐私问题之后，我们提出了一种针对最新的人类检测模型的黑盒对抗性攻击方法InvisibiliTee。该方法学习了T恤的可打印对抗性图案，这些T恤将穿着者遮盖在现实世界中的人面前。我们设计了一种角度不可知的学习方案，它利用时尚数据集的分割和几何扭曲过程，从而生成的对抗性模式能够有效地愚弄来自所有摄像机角度的人检测器和不可见的黑匣子检测模型。在数字和物理环境中的经验结果表明，随着隐形设备的开启，个人跟踪系统检测佩戴者的能力显著下降。



## **13. ARIEL: Adversarial Graph Contrastive Learning**

Ariel：对抗性图形对比学习 cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06956v1)

**Authors**: Shengyu Feng, Baoyu Jing, Yada Zhu, Hanghang Tong

**Abstracts**: Contrastive learning is an effective unsupervised method in graph representation learning, and the key component of contrastive learning lies in the construction of positive and negative samples. Previous methods usually utilize the proximity of nodes in the graph as the principle. Recently, the data augmentation based contrastive learning method has advanced to show great power in the visual domain, and some works extended this method from images to graphs. However, unlike the data augmentation on images, the data augmentation on graphs is far less intuitive and much harder to provide high-quality contrastive samples, which leaves much space for improvement. In this work, by introducing an adversarial graph view for data augmentation, we propose a simple but effective method, Adversarial Graph Contrastive Learning (ARIEL), to extract informative contrastive samples within reasonable constraints. We develop a new technique called information regularization for stable training and use subgraph sampling for scalability. We generalize our method from node-level contrastive learning to the graph-level by treating each graph instance as a supernode. ARIEL consistently outperforms the current graph contrastive learning methods for both node-level and graph-level classification tasks on real-world datasets. We further demonstrate that ARIEL is more robust in face of adversarial attacks.

摘要: 对比学习是图形表示学习中一种有效的无监督方法，而对比学习的关键在于正负样本的构造。以往的方法通常以图中节点的邻近度为原则。近年来，基于数据增强的对比学习方法在视觉领域显示出强大的生命力，一些工作将该方法从图像扩展到图形。然而，与图像上的数据增强不同，图形上的数据增强的直观性差得多，难以提供高质量的对比样本，这就留下了很大的改进空间。在这项工作中，通过引入一种用于数据增强的对抗性图视图，我们提出了一种简单但有效的方法--对抗性图形对比学习(Ariel)，以在合理的约束下提取信息丰富的对比样本。我们开发了一种称为信息正则化的新技术来稳定训练，并使用子图抽样来实现可伸缩性。通过将每个图实例看作一个超节点，将我们的方法从节点级的对比学习推广到图级。对于实际数据集上的节点级和图级分类任务，Ariel始终优于当前的图对比学习方法。我们进一步证明了Ariel在面对对手攻击时具有更强的鲁棒性。



## **14. GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing**

GNPassGAN：改进的用于拖网离线口令猜测的生成性对抗网络 cs.CR

9 pages, 8 tables, 3 figures

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06943v1)

**Authors**: Fangyi Yu, Miguel Vargas Martin

**Abstracts**: The security of passwords depends on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to represent an actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper reviews various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. It also introduces GNPassGAN, a password guessing tool built on generative adversarial networks for trawling offline attacks. In comparison to the state-of-the-art PassGAN model, GNPassGAN is capable of guessing 88.03\% more passwords and generating 31.69\% fewer duplicates.

摘要: 密码的安全性取决于对攻击者使用的策略的透彻理解。不幸的是，现实世界中的对手使用的是实用的猜测策略，如字典攻击，这在密码安全研究中很难模拟。必须仔细配置和修改字典攻击，以表示实际威胁。然而，这种方法需要难以复制的特定领域的知识和专业技能。本文综述了各种基于深度学习的密码猜测方法，这些方法不需要领域知识或关于用户密码结构和组合的假设。它还介绍了GNPassGAN，这是一个建立在生成性对手网络上的密码猜测工具，用于拖网离线攻击。与最新的PassGAN模型相比，GNPassGAN模型能够多猜测88.03个口令，生成的重复项减少31.69个。



## **15. Gradient Mask: Lateral Inhibition Mechanism Improves Performance in Artificial Neural Networks**

梯度掩模：侧抑制机制改善人工神经网络的性能 cs.CV

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06918v1)

**Authors**: Lei Jiang, Yongqing Liu, Shihai Xiao, Yansong Chua

**Abstracts**: Lateral inhibitory connections have been observed in the cortex of the biological brain, and has been extensively studied in terms of its role in cognitive functions. However, in the vanilla version of backpropagation in deep learning, all gradients (which can be understood to comprise of both signal and noise gradients) flow through the network during weight updates. This may lead to overfitting. In this work, inspired by biological lateral inhibition, we propose Gradient Mask, which effectively filters out noise gradients in the process of backpropagation. This allows the learned feature information to be more intensively stored in the network while filtering out noisy or unimportant features. Furthermore, we demonstrate analytically how lateral inhibition in artificial neural networks improves the quality of propagated gradients. A new criterion for gradient quality is proposed which can be used as a measure during training of various convolutional neural networks (CNNs). Finally, we conduct several different experiments to study how Gradient Mask improves the performance of the network both quantitatively and qualitatively. Quantitatively, accuracy in the original CNN architecture, accuracy after pruning, and accuracy after adversarial attacks have shown improvements. Qualitatively, the CNN trained using Gradient Mask has developed saliency maps that focus primarily on the object of interest, which is useful for data augmentation and network interpretability.

摘要: 在生物大脑的皮质中观察到了侧抑制连接，并就其在认知功能中的作用进行了广泛的研究。然而，在深度学习中的反向传播的香草版本中，所有的梯度(可以理解为包括信号和噪声梯度)在权重更新期间流经网络。这可能会导致过度适应。在这项工作中，受生物侧向抑制的启发，我们提出了梯度掩模，它有效地滤除了反向传播过程中的噪声梯度。这允许学习的特征信息更密集地存储在网络中，同时过滤掉噪声或不重要的特征。此外，我们还解析地演示了人工神经网络中的侧抑制如何提高传播梯度的质量。提出了一种新的梯度质量判据，可作为各种卷积神经网络(CNN)训练过程中的一种衡量标准。最后，我们进行了几个不同的实验，从定量和定性两个方面研究了梯度掩码如何提高网络的性能。在数量上，原始CNN架构中的准确性、修剪后的准确性和对抗性攻击后的准确性都显示出改进。在质量上，使用梯度蒙版训练的CNN已经开发出主要集中在感兴趣对象上的显著地图，这对于数据增强和网络可解释性很有用。



## **16. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

IPv6 SeeYou：利用IPv6中泄漏的标识符进行街道级地理定位 cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06767v1)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.

摘要: 我们提出了IPv6 SeeYou，这是一种隐私攻击，允许远程和非特权对手以街道级别的精度物理定位许多住宅IPv6主机和网络。我们方法的关键涉及：1)从家庭路由器远程发现广域(WAN)硬件MAC地址；2)将这些MAC地址与已知位置的对应WiFi BSSID关联；以及3)通过关联连接到公共倒数第二个提供商路由器的设备来扩展覆盖范围。我们首先通过高速网络探测获得嵌入在IPv6地址中的大量MAC语料库。这些MAC地址有效地沿协议堆栈向上泄露，主要代表住宅路由器的广域网接口，其中许多是也提供WiFi的一体化设备。我们开发了一种技术来统计推断路由器的广域网和跨制造商和设备的WiFi MAC地址之间的映射，并发动大规模数据融合攻击，将广域网MAC与战争驾驶(地理定位)数据库中提供的WiFi BSSID相关联。利用这些相关性，我们在146个国家和地区对价值超过1200万美元的路由器的IPv6前缀进行了地理定位。选定的验证确认地理位置误差的中位数为39米。然后，我们利用技术和部署限制将攻击扩展到更大的一组IPv6住宅路由器，方法是将设备与常见的倒数第二个提供商路由器进行集群和关联。虽然我们负责任地向几家制造商和供应商披露了我们的结果，但已部署的住宅有线电视和DSL路由器的僵化生态系统表明，在可预见的未来，我们的攻击仍将对隐私构成威胁。



## **17. Adversarial Texture for Fooling Person Detectors in the Physical World**

物理世界中愚人探测器的对抗性纹理 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2203.03373v4)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Fuchun Sun, Bo Zhang, Xiaolin Hu

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.

摘要: 如今，配备人工智能系统的摄像头可以捕捉和分析图像，自动检测人。然而，人工智能系统在接收到现实世界中故意设计的模式时可能会出错，即物理对抗性例子。以前的工作已经表明，可以在衣服上打印敌意补丁来躲避基于DNN的个人探测器。然而，当视角(即相机指向对象的角度)改变时，这些对抗性的例子可能会使攻击成功率灾难性地下降。为了进行多角度攻击，我们提出了对抗性纹理(AdvTexture)。AdvTexture可以覆盖任意形状的衣服，这样穿着这种衣服的人就可以从不同的视角隐藏起来，躲避人的探测器。提出了一种基于环形裁剪的可扩展产生式攻击方法(TC-EGA)，用于制作具有重复结构的AdvTexture。我们用AdvTexure打印了几块布，然后在现实世界中制作了T恤、裙子和连衣裙。实验表明，这些衣服可以愚弄物理世界中的人体探测器。



## **18. An Analytic Framework for Robust Training of Artificial Neural Networks**

一种神经网络稳健训练的分析框架 cs.LG

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2205.13502v2)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.

摘要: 学习模型的可靠性是机器学习在各个行业成功部署的关键。创建一个健壮的模型，特别是一个不受对抗性攻击影响的模型，需要对对抗性例子现象有一个全面的了解。然而，由于机器学习中问题的复杂性，这一现象很难描述。因此，许多研究通过提出对抗性例子如何发生的简化模型来研究这一现象，并通过预测该现象的某些方面来验证该模型。虽然这些研究涵盖了对抗性例子的许多不同特征，但它们还没有达成对这一现象的几何和解析建模的整体方法。本文提出了一种学习理论中研究这一现象的形式化框架，并利用复分析和全纯理论为人工神经网络提供了一种稳健的学习规则。在复杂分析的帮助下，我们可以毫不费力地在现象的几何和解析视角之间切换，并通过揭示它与调和函数的联系来提供对该现象的进一步见解。使用我们的模型，我们可以解释对抗性例子的一些最有趣的特征，包括对抗性例子的可转移性，并为缓解这一现象的影响的新方法铺平道路。



## **19. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

图神经网络图分类中的敌意攻击再探 cs.SI

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06651v1)

**Authors**: Beini Xie, Heng Chang, Xin Wang, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstracts**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and diverse downstream real-world applications. Despite their success, existing approaches are either limited to structure attacks or restricted to local information. This calls for a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" problem, we present a general framework CAMA to generate adversarial examples by manipulating graph structure and node features in a hierarchical style. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.

摘要: 图形神经网络(GNN)在图形分类和各种下游实际应用中取得了巨大的成功。尽管它们取得了成功，但现有的方法要么限于结构攻击，要么限于局部信息。这就需要一个更通用的图分类攻击框架，由于利用全局图级信息生成局部节点级对抗性实例的复杂性，该框架面临着巨大的挑战。为了解决这个“从全局到局部”的问题，我们提出了一个通用的CAMA框架，通过以层次化的方式操纵图的结构和节点特征来生成对抗性实例。具体地说，我们利用图类激活映射及其变体来产生与图分类任务相对应的节点级重要性。然后通过算法的启发式设计，在节点级和子图级重要性的帮助下，在不可察觉的扰动预算下执行特征攻击和结构攻击。在六个真实世界基准上对四个最先进的图分类模型进行了攻击实验，验证了该框架的灵活性和有效性。



## **20. Poison Ink: Robust and Invisible Backdoor Attack**

毒墨：强大而隐形的后门攻击 cs.CR

IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2108.02488v3)

**Authors**: Jie Zhang, Dongdong Chen, Qidong Huang, Jing Liao, Weiming Zhang, Huamin Feng, Gang Hua, Nenghai Yu

**Abstracts**: Recent research shows deep neural networks are vulnerable to different types of attacks, such as adversarial attack, data poisoning attack and backdoor attack. Among them, backdoor attack is the most cunning one and can occur in almost every stage of deep learning pipeline. Therefore, backdoor attack has attracted lots of interests from both academia and industry. However, most existing backdoor attack methods are either visible or fragile to some effortless pre-processing such as common data transformations. To address these limitations, we propose a robust and invisible backdoor attack called "Poison Ink". Concretely, we first leverage the image structures as target poisoning areas, and fill them with poison ink (information) to generate the trigger pattern. As the image structure can keep its semantic meaning during the data transformation, such trigger pattern is inherently robust to data transformations. Then we leverage a deep injection network to embed such trigger pattern into the cover image to achieve stealthiness. Compared to existing popular backdoor attack methods, Poison Ink outperforms both in stealthiness and robustness. Through extensive experiments, we demonstrate Poison Ink is not only general to different datasets and network architectures, but also flexible for different attack scenarios. Besides, it also has very strong resistance against many state-of-the-art defense techniques.

摘要: 最近的研究表明，深度神经网络容易受到不同类型的攻击，如对抗性攻击、数据中毒攻击和后门攻击。其中，后门攻击是最狡猾的一种，几乎可以发生在深度学习管道的每个阶段。因此，后门攻击引起了学术界和产业界的广泛关注。然而，大多数现有的后门攻击方法对于一些毫不费力的预处理，如常见的数据转换，要么是可见的，要么是脆弱的。为了解决这些局限性，我们提出了一种强大且不可见的后门攻击，称为“毒墨”。具体地说，我们首先利用图像结构作为目标中毒区域，并在其中填充毒墨(信息)来生成触发图案。由于图像结构在数据转换过程中可以保持其语义，因此这种触发模式对数据转换具有内在的健壮性。然后利用深度注入网络将这种触发模式嵌入到封面图像中，以实现隐身。与现有流行的后门攻击方法相比，毒墨在隐蔽性和健壮性方面都更胜一筹。通过大量的实验，我们证明了毒墨不仅对不同的数据集和网络体系结构具有通用性，而且对不同的攻击场景也具有很强的灵活性。此外，它还对许多最先进的防御技术具有很强的抵抗力。



## **21. MaskBlock: Transferable Adversarial Examples with Bayes Approach**

MaskBlock：贝叶斯方法的可转移对抗性实例 cs.LG

Under Review

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06538v1)

**Authors**: Mingyuan Fan, Cen Chen, Ximeng Liu, Wenzhong Guo

**Abstracts**: The transferability of adversarial examples (AEs) across diverse models is of critical importance for black-box adversarial attacks, where attackers cannot access the information about black-box models. However, crafted AEs always present poor transferability. In this paper, by regarding the transferability of AEs as generalization ability of the model, we reveal that vanilla black-box attacks craft AEs via solving a maximum likelihood estimation (MLE) problem. For MLE, the results probably are model-specific local optimum when available data is small, i.e., limiting the transferability of AEs. By contrast, we re-formulate crafting transferable AEs as the maximizing a posteriori probability estimation problem, which is an effective approach to boost the generalization of results with limited available data. Because Bayes posterior inference is commonly intractable, a simple yet effective method called MaskBlock is developed to approximately estimate. Moreover, we show that the formulated framework is a generalization version for various attack methods. Extensive experiments illustrate MaskBlock can significantly improve the transferability of crafted adversarial examples by up to about 20%.

摘要: 在黑盒对抗性攻击中，攻击者无法获取有关黑盒模型的信息，因此对抗性示例在不同模型之间的可转移性至关重要。然而，精心制作的AE总是表现出较差的可转移性。通过将攻击事件的可转移性作为模型的泛化能力，揭示了香草黑盒攻击通过求解一个极大似然估计问题来欺骗攻击事件。对于最大似然估计，当可用数据很小时，结果可能是特定于模型的局部最优，即限制了AE的可转移性。相反，我们将构造可转移的AEs重新描述为最大后验概率估计问题，这是在可用数据有限的情况下提高结果普适性的有效方法。由于贝叶斯后验推断通常很难处理，因此发展了一种简单而有效的方法-MaskBlock来近似估计。此外，我们还证明了该框架是各种攻击方法的泛化版本。大量的实验表明，MaskBlock可以显著提高特制的对抗性例子的可转移性，最高可提高约20%。



## **22. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

捉迷藏：关于深度学习系统攻击的隐蔽性 cs.CR

To appear in European Symposium on Research in Computer Security  (ESORICS) 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2205.15944v2)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.

摘要: 随着人工智能和机器学习的日益普及，文献中提出了针对深度学习模型的广泛攻击。逃避攻击和投毒攻击都试图利用敌意更改的样本来愚弄受害者模型来错误分类敌意样本。虽然这种攻击声称是或预计是隐蔽的，即人眼看不见，但这种说法很少得到评估。本文首次对深度学习攻击中使用的敌意样本的隐蔽性进行了大规模研究。我们已经在六个流行的基准数据集上实施了20个具有代表性的对抗性ML攻击。我们使用两种互补的方法来评估攻击样本的隐蔽性：(1)采用24个度量来评估图像相似性或质量的数值研究；(2)用户研究3组问卷，从1000多个回复中收集了20,000多个注释。我们的结果表明，现有的大多数攻击都引入了不可忽略的扰动，这些扰动对人眼来说是不隐形的。进一步分析了影响攻击隐蔽性的因素。我们进一步检验了数值分析和用户研究之间的相关性，并证明了一些图像质量度量可以为攻击设计提供有用的指导，而评估的图像质量和攻击的视觉隐蔽性之间仍然存在着显著的差距。



## **23. PRIVEE: A Visual Analytic Workflow for Proactive Privacy Risk Inspection of Open Data**

Privee：开放数据主动隐私风险检测的可视化分析工作流 cs.CR

Accepted for IEEE Symposium on Visualization in Cyber Security, 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06481v1)

**Authors**: Kaustav Bhattacharjee, Akm Islam, Jaideep Vaidya, Aritra Dasgupta

**Abstracts**: Open data sets that contain personal information are susceptible to adversarial attacks even when anonymized. By performing low-cost joins on multiple datasets with shared attributes, malicious users of open data portals might get access to information that violates individuals' privacy. However, open data sets are primarily published using a release-and-forget model, whereby data owners and custodians have little to no cognizance of these privacy risks. We address this critical gap by developing a visual analytic solution that enables data defenders to gain awareness about the disclosure risks in local, joinable data neighborhoods. The solution is derived through a design study with data privacy researchers, where we initially play the role of a red team and engage in an ethical data hacking exercise based on privacy attack scenarios. We use this problem and domain characterization to develop a set of visual analytic interventions as a defense mechanism and realize them in PRIVEE, a visual risk inspection workflow that acts as a proactive monitor for data defenders. PRIVEE uses a combination of risk scores and associated interactive visualizations to let data defenders explore vulnerable joins and interpret risks at multiple levels of data granularity. We demonstrate how PRIVEE can help emulate the attack strategies and diagnose disclosure risks through two case studies with data privacy experts.

摘要: 包含个人信息的开放数据集即使在匿名的情况下也容易受到敌意攻击。通过对具有共享属性的多个数据集执行低成本连接，开放数据门户的恶意用户可能会访问侵犯个人隐私的信息。然而，开放数据集主要是使用一种即发布即忘的模式发布的，在这种模式下，数据所有者和托管人很少或根本没有意识到这些隐私风险。我们通过开发可视化分析解决方案来解决这一关键差距，该解决方案使数据防御者能够意识到本地可合并数据社区的披露风险。解决方案是通过与数据隐私研究人员的设计研究得出的，在设计研究中，我们最初扮演红色团队的角色，并根据隐私攻击场景进行道德的数据黑客练习。我们使用这个问题和领域特征来开发一套视觉分析干预作为防御机制，并在Privee中实现它们，Privee是一个视觉风险检测工作流，充当数据防御者的主动监视器。Privee结合使用风险分值和关联的交互式可视化，让数据防御者能够在多个级别的数据粒度上探索易受攻击的连接并解释风险。通过与数据隐私专家的两个案例研究，我们展示了Privee如何帮助模拟攻击策略和诊断泄露风险。



## **24. UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks**

UniNet：一个统一的场景理解网络，通过对抗性攻击的镜头探索多任务关系 cs.CV

Accepted at DeepMTL workshop, ICCV 2021

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2108.04584v2)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Scene understanding is crucial for autonomous systems which intend to operate in the real world. Single task vision networks extract information only based on some aspects of the scene. In multi-task learning (MTL), on the other hand, these single tasks are jointly learned, thereby providing an opportunity for tasks to share information and obtain a more comprehensive understanding. To this end, we develop UniNet, a unified scene understanding network that accurately and efficiently infers vital vision tasks including object detection, semantic segmentation, instance segmentation, monocular depth estimation, and monocular instance depth prediction. As these tasks look at different semantic and geometric information, they can either complement or conflict with each other. Therefore, understanding inter-task relationships can provide useful cues to enable complementary information sharing. We evaluate the task relationships in UniNet through the lens of adversarial attacks based on the notion that they can exploit learned biases and task interactions in the neural network. Extensive experiments on the Cityscapes dataset, using untargeted and targeted attacks reveal that semantic tasks strongly interact amongst themselves, and the same holds for geometric tasks. Additionally, we show that the relationship between semantic and geometric tasks is asymmetric and their interaction becomes weaker as we move towards higher-level representations.

摘要: 场景理解对于打算在真实世界中运行的自主系统至关重要。单任务视觉网络仅基于场景的某些方面来提取信息。另一方面，在多任务学习(MTL)中，这些单一任务是共同学习的，从而为任务提供了共享信息和获得更全面理解的机会。为此，我们开发了UniNet，这是一个统一的场景理解网络，可以准确高效地推断重要的视觉任务，包括对象检测、语义分割、实例分割、单目深度估计和单目实例深度预测。当这些任务查看不同的语义和几何信息时，它们可以相互补充，也可以相互冲突。因此，了解任务间的关系可以为实现互补信息共享提供有用的线索。我们通过对抗性攻击的视角来评估UniNet中的任务关系，基于这样的概念，即它们可以利用神经网络中的学习偏差和任务交互。在城市景观数据集上使用无目标和目标攻击进行的广泛实验表明，语义任务之间相互作用很强，几何任务也是如此。此外，我们还证明了语义任务和几何任务之间的关系是不对称的，并且随着我们向更高级别的表示前进，它们之间的交互作用变得更弱。



## **25. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

统一梯度以提高深度网络的真实稳健性 stat.ML

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06228v1)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among them score-based query attacks (SQAs) are the most threatening ones because of their practicalities and effectiveness: the attackers only need dozens of queries on model outputs to seriously hurt a victim network. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with attackers. In this paper, we propose a real-world defense, called Unifying Gradients (UniG), to unify gradients of different data so that attackers could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. To enhance UniG's practical significance in real-world applications, we implement it as a Hadamard product module that is computationally-efficient and readily plugged into any model. According to extensive experiments on 5 SQAs and 4 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a CIFAR-10 model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG greatly surpasses all compared baselines in clean accuracy and the modification degree of outputs. The code would be released.

摘要: 深度神经网络(DNN)的广泛应用要求人们越来越关注其在现实世界中的健壮性，即DNN是否能抵抗黑箱对抗攻击，其中基于分数的查询攻击(Score-Based Query Attack，SBA)因其实用性和有效性而成为最具威胁性的攻击：攻击者只需对模型输出进行数十次查询即可严重损害受害者网络。由于用户的服务目的，防御SBA需要稍微但巧妙地改变输出，因为用户与攻击者共享相同的输出信息。在本文中，我们提出了一种称为统一梯度的真实防御方法，将不同数据的梯度统一起来，使得攻击者只能探测对不同样本相似的弱得多的攻击方向。由于这种通用的攻击扰动已被验证为不如特定于输入的扰动那么具侵略性，unig通过向攻击者指示一个扭曲的、信息量较少的攻击方向来保护现实世界的DNN。为了增强unig在现实世界应用程序中的实际意义，我们将其实现为Hadamard产品模块，该模块计算效率高，可以方便地插入到任何型号中。根据在5个SQA和4个防御基线上的广泛实验，unig在不损害CIFAR10和ImageNet上的干净准确性的情况下，显著提高了真实世界的健壮性。例如，在2500-Query Square攻击下，unig保持了77.80%的CIFAR-10模型，而最新的对抗性训练模型在CIFAR10上只有67.34%的准确率。同时，unig在清洁精度和输出修改程度上大大超过了所有比较的基线。代码将会被发布。



## **26. Scale-free Photo-realistic Adversarial Pattern Attack**

无尺度照片真实感对抗性模式攻击 cs.CV

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06222v1)

**Authors**: Xiangbo Gao, Weicheng Xie, Minmin Liu, Cheng Luo, Qinliang Lin, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstracts**: Traditional pixel-wise image attack algorithms suffer from poor robustness to defense algorithms, i.e., the attack strength degrades dramatically when defense algorithms are applied. Although Generative Adversarial Networks (GAN) can partially address this problem by synthesizing a more semantically meaningful texture pattern, the main limitation is that existing generators can only generate images of a specific scale. In this paper, we propose a scale-free generation-based attack algorithm that synthesizes semantically meaningful adversarial patterns globally to images with arbitrary scales. Our generative attack approach consistently outperforms the state-of-the-art methods on a wide range of attack settings, i.e. the proposed approach largely degraded the performance of various image classification, object detection, and instance segmentation algorithms under different advanced defense methods.

摘要: 传统的像素级图像攻击算法对防御算法的健壮性较差，即应用防御算法后攻击强度急剧下降。虽然生成性对抗网络(GAN)可以通过合成更具语义意义的纹理模式来部分解决这个问题，但主要限制是现有生成器只能生成特定规模的图像。在本文中，我们提出了一种基于无尺度生成的攻击算法，该算法将全局具有语义意义的攻击模式合成到任意尺度的图像中。我们的生成性攻击方法在广泛的攻击环境下始终优于最先进的方法，即在不同的高级防御方法下，所提出的方法在很大程度上降低了各种图像分类、目标检测和实例分割算法的性能。



## **27. A Knowledge Distillation-Based Backdoor Attack in Federated Learning**

联邦学习中一种基于知识蒸馏的后门攻击 cs.LG

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06176v1)

**Authors**: Yifan Wang, Wei Fan, Keke Yang, Naji Alhusaini, Jing Li

**Abstracts**: Federated Learning (FL) is a novel framework of decentralized machine learning. Due to the decentralized feature of FL, it is vulnerable to adversarial attacks in the training procedure, e.g. , backdoor attacks. A backdoor attack aims to inject a backdoor into the machine learning model such that the model will make arbitrarily incorrect behavior on the test sample with some specific backdoor trigger. Even though a range of backdoor attack methods of FL has been introduced, there are also methods defending against them. Many of the defending methods utilize the abnormal characteristics of the models with backdoor or the difference between the models with backdoor and the regular models. To bypass these defenses, we need to reduce the difference and the abnormal characteristics. We find a source of such abnormality is that backdoor attack would directly flip the label of data when poisoning the data. However, current studies of the backdoor attack in FL are not mainly focus on reducing the difference between the models with backdoor and the regular models. In this paper, we propose Adversarial Knowledge Distillation(ADVKD), a method combine knowledge distillation with backdoor attack in FL. With knowledge distillation, we can reduce the abnormal characteristics in model result from the label flipping, thus the model can bypass the defenses. Compared to current methods, we show that ADVKD can not only reach a higher attack success rate, but also successfully bypass the defenses when other methods fails. To further explore the performance of ADVKD, we test how the parameters affect the performance of ADVKD under different scenarios. According to the experiment result, we summarize how to adjust the parameter for better performance under different scenarios. We also use several methods to visualize the effect of different attack and explain the effectiveness of ADVKD.

摘要: 联邦学习(FL)是一种新型的去中心化机器学习框架。由于FL的分散性，它在训练过程中很容易受到对抗性攻击，例如后门攻击。后门攻击的目的是向机器学习模型中注入一个后门，以便该模型将使用某个特定的后门触发器在测试样本上做出任意不正确的行为。尽管已经引入了一系列FL的后门攻击方法，但也有一些方法可以防御它们。许多防御方法利用了后门模型的异常特性，或者利用了后门模型与常规模型的区别。为了绕过这些防御，我们需要减少差异和异常特征。我们发现这种异常的一个来源是后门攻击在毒化数据时会直接翻转数据的标签。然而，目前对FL后门攻击的研究主要集中在缩小后门模型与常规模型之间的差异上。本文提出了一种将知识提取与后门攻击相结合的方法--对抗性知识提取(ADVKD)。通过知识提取，可以减少模型中因标签翻转而导致的异常特征，从而使模型能够绕过防御。与现有方法相比，我们证明了ADVKD不仅可以达到更高的攻击成功率，而且在其他方法失败的情况下可以成功地绕过防御。为了进一步探索ADVKD的性能，我们测试了不同场景下参数对ADVKD性能的影响。根据实验结果，总结了在不同场景下如何调整参数以获得更好的性能。我们还使用几种方法来可视化不同攻击的效果，并解释了ADVKD的有效性。



## **28. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

MulVAL扩展及其攻击场景覆盖研究综述 cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.

摘要: 组织使用各种对手模型来评估攻击对其网络的风险和潜在影响。攻击图表示攻击者可以采取的漏洞和行动，以识别和危害组织的资产。攻击图便于以攻击路径的形式对攻击场景进行可视化呈现和算法分析。MulVAL是一个用于构建逻辑攻击图的通用开源框架，已经被研究人员和实践者广泛使用，并被他们用额外的攻击场景进行扩展。本文综述了现有的所有MulVAL扩展，并将所有的MulVAL交互规则映射到MITRE ATT&CK技术，以估计它们的攻击场景覆盖率。这项调查将当前的MulVAL扩展与统一的本体概念保持一致，并强调了存在的差距。它为MulVAL的系统改进和对MITRE ATT&CK捕获的敌对行为的整个场景的全面建模铺平了道路。



## **29. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

注意空间上可转移对抗性攻击的不同生成性对抗性扰动 cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.

摘要: 具有改进的可转移性的对抗性攻击--在已知模型上制作的对抗性例子也能够愚弄未知模型的能力--由于其实用性最近受到了极大的关注。然而，现有的可转移攻击以确定性的方式制造扰动，往往不能充分探索损失曲面，从而陷入较差的局部最优，且可转移性较低。为了解决这一问题，我们提出了注意力多样性攻击(ADA)，它以随机的方式破坏不同的显著特征，以提高可转移性。首先，我们扰乱图像注意力，以扰乱不同模型共享的通用特征。然后，为了有效地避免局部最优，我们以随机的方式破坏了这些特征，并更详尽地探索了可转移扰动的搜索空间。更具体地说，我们使用生成器来产生对抗性扰动，每个扰动都以不同的方式干扰特征，具体取决于输入的潜在代码。广泛的实验评估表明，我们的方法是有效的，超过了最先进的方法的可转移性。有关代码，请访问https://github.com/wkim97/ADA.



## **30. Controlled Quantum Teleportation in the Presence of an Adversary**

对手在场时的受控量子隐形传态 quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.

摘要: 在接收者不可信任的情况下，我们提出了受控量子隐形传态的设备无关分析。我们证明了真正的三方非局部性的概念允许我们在这种情况下证明控制权。通过考虑具有去极化噪声特征的设备上的特定对抗攻击策略，我们发现控制功率是真三方非定域性的单调递增函数。这些结果对构建实用的量子通信网络具有重要意义，也有助于揭示非定域性在多体量子信息处理中的作用。



## **31. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

Pikachu：通过使用Taproot检查点进入比特币PoW来保护PoS区块链免受远程攻击 cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.

摘要: 基于可重用资源的区块链系统，如风险证明(POS)，提供的安全保证比基于工作证明的系统更弱。具体地说，它们容易受到远程攻击，在远程攻击中，对手可以破坏之前的参与者，以便重写链的完整历史。为了防止这种对PoS链的攻击，我们提出了一种协议，将PoS链的状态检查到工作证明区块链，如比特币。因此，我们的检查点协议不依赖于任何中央机构。我们的工作使用Schnorr签名并利用比特币最近的Taproot升级，使我们能够创建恒定大小的检查点交易。我们为协议的安全性进行了论证，并给出了一个在比特币测试网上进行测试的开源实现。



## **32. Reducing Exploitability with Population Based Training**

通过基于人口的培训减少可利用性 cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05083v1)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.

摘要: 自我发挥强化学习在各种零和游戏中实现了最先进的，往往是超人的表现。然而，先前的工作已经发现，对常规对手具有高度能力的政策，可能会在对抗对手的政策上灾难性地失败：一个明确针对受害者的对手。使用对抗性训练的先前防御能够使受害者对特定的对手变得健壮，但受害者仍然容易受到新对手的攻击。我们推测，这一限制是由于训练过程中看到的对手多样性不足所致。我们建议使用基于人口的训练来防御，让受害者与不同的对手对抗。我们在两个低维环境中评估了该防御对新对手的健壮性。我们的防御提高了对抗对手的健壮性，这是通过攻击者训练时间步数来衡量的，以利用受害者。此外，我们还证明了健壮性与对手种群的大小相关。



## **33. Adversarial Machine Learning-Based Anticipation of Threats Against Vehicle-to-Microgrid Services**

基于对抗性机器学习的车辆到微电网服务威胁预测 cs.CR

IEEE Global Communications Conference (Globecom), 2022, 6 pages, 2  Figures, 4 Tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2208.05073v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstracts**: In this paper, we study the expanding attack surface of Adversarial Machine Learning (AML) and the potential attacks against Vehicle-to-Microgrid (V2M) services. We present an anticipatory study of a multi-stage gray-box attack that can achieve a comparable result to a white-box attack. Adversaries aim to deceive the targeted Machine Learning (ML) classifier at the network edge to misclassify the incoming energy requests from microgrids. With an inference attack, an adversary can collect real-time data from the communication between smart microgrids and a 5G gNodeB to train a surrogate (i.e., shadow) model of the targeted classifier at the edge. To anticipate the associated impact of an adversary's capability to collect real-time data instances, we study five different cases, each representing different amounts of real-time data instances collected by an adversary. Out of six ML models trained on the complete dataset, K-Nearest Neighbour (K-NN) is selected as the surrogate model, and through simulations, we demonstrate that the multi-stage gray-box attack is able to mislead the ML classifier and cause an Evasion Increase Rate (EIR) up to 73.2% using 40% less data than what a white-box attack needs to achieve a similar EIR.

摘要: 本文研究了对抗性机器学习(AML)不断扩大的攻击面和针对车辆到微电网(V2M)服务的潜在攻击。我们提出了一种多阶段灰盒攻击的预期研究，它可以获得与白盒攻击相当的结果。攻击者的目标是在网络边缘欺骗目标机器学习(ML)分类器，以对来自微电网的传入能源请求进行错误分类。利用推理攻击，攻击者可以从智能微网和5G gNodeB之间的通信中收集实时数据，以在边缘训练目标分类器的代理(即，影子)模型。为了预测对手收集实时数据实例的能力的相关影响，我们研究了五个不同的案例，每个案例代表了对手收集的不同数量的实时数据实例。在完整数据集上训练的6个ML模型中，选择K-近邻(K-NN)作为代理模型，通过仿真，我们证明了多级灰盒攻击能够误导ML分类器，并导致高达73.2%的逃避增加率(EIR)，而白盒攻击所需的数据比白盒攻击所需的数据少40%。



## **34. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

愚弄你的敌人：用于防御IoMT数据模型反转攻击的近距离梯度分裂学习 cs.CR

10 pages, 5 figures, 2 tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2201.04569v3)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 14.0$\%$, 17.9$\%$, and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.

摘要: 在过去的十年中，人工智能(AI)，特别是深度学习网络，在医疗物联网(IoMT)生态系统中得到了迅速的采用。然而，最近的研究表明，深度学习网络可以被敌意攻击所利用，这些攻击不仅使物联网容易受到数据窃取的攻击，而且还容易受到医疗诊断的篡改。现有的研究认为在原始IoMT数据或模型参数中加入噪声，不仅降低了医学推断的整体性能，而且对梯度法等深度泄漏方法无效。在这项工作中，我们提出了近邻梯度分裂学习(PSGL)方法来防御模型反转攻击。该方法在客户端进行深度神经网络训练时，对IoMT数据进行故意攻击。提出了利用近邻梯度法恢复梯度图，并采用决策层融合策略来提高识别性能。广泛的分析表明，PGSL不仅提供了对模型反转攻击的有效防御机制，而且有助于提高对公开可用的数据集的识别性能。与重建图像和对抗性攻击图像相比，准确率分别提高了14.0、17.9和36.9美元。



## **35. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.04435v1)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **36. Can collaborative learning be private, robust and scalable?**

协作学习能否做到私密性、健壮性和可扩展性？ cs.LG

Accepted at MICCAI DeCaF 2022

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.02652v2)

**Authors**: Dmitrii Usynin, Helena Klause, Johannes C. Paetzold, Daniel Rueckert, Georgios Kaissis

**Abstracts**: In federated learning for medical image analysis, the safety of the learning protocol is paramount. Such settings can often be compromised by adversaries that target either the private data used by the federation or the integrity of the model itself. This requires the medical imaging community to develop mechanisms to train collaborative models that are private and robust against adversarial data. In response to these challenges, we propose a practical open-source framework to study the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples under train- and inference-time attacks. Using our framework, we achieve competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation, critical in medical image analysis.

摘要: 在医学图像分析的联合学习中，学习协议的安全性至关重要。此类设置通常会被针对联盟使用的私有数据或模型本身的完整性的攻击者所攻破。这需要医学成像界开发机制，以训练针对敌对数据的私有和健壮的协作模型。针对这些挑战，我们提出了一个实用的开源框架来研究差异隐私、模型压缩和对抗性训练相结合的有效性，以提高模型在训练和推理时间攻击下对对抗性样本的健壮性。使用我们的框架，我们获得了具有竞争力的模型性能，显著减少了模型的规模，并在不严重性能下降的情况下改善了经验对抗鲁棒性，这在医学图像分析中至关重要。



## **37. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

多智能体强化学习中的稀疏对抗性攻击 cs.AI

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.09362v2)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).

摘要: 协作多智能体强化学习(CMARL)有很多实际应用，但已有的cMARL算法训练的策略在实际应用中不够健壮。针对RL系统的对抗性攻击也有很多方法，这意味着RL系统可能会遭受对抗性攻击，但大多数方法都集中在单个代理RL上。本文提出了一种针对cMARL系统的稀疏对抗攻击。我们使用带正则化的(MA)RL来训练攻击策略。我们的实验表明，当团队中只有一个或几个代理(例如，8个代理中的1个或25个代理中的5个)在几个时间步骤(例如，总共40个时间步骤中的攻击3个)受到攻击时，由当前cMARL算法训练的策略会获得较差的性能。



## **38. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

对抗性像素恢复作为可转移扰动的借口任务 cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2207.08803v2)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR

摘要: 可转移对抗性攻击从预先训练的代理模型和已知标签空间中优化对手，以愚弄未知的黑盒模型。因此，这些攻击受到有效代理模型可用性的限制。在这项工作中，我们放松了这一假设，提出了对抗性像素复原作为一种自我监督的替代方案，在没有标签和数据样本的情况下，从零开始训练一个有效的代理模型。我们的训练方法基于最小-最大方案，该方案减少了通过对抗性目标的过度拟合，从而优化了更具普适性的代理模型。我们提出的攻击是对抗性像素恢复的补充，并且独立于任何特定于任务的目标，因为它可以以自我监督的方式发起。我们成功地展示了我们的视觉变形方法以及卷积神经网络方法在分类、目标检测和视频分割任务中的对抗性可转移性。我们的训练方法将基线无监督训练方法在ImageNet Val上的可转移性提高了16.4%。准备好了。我们的代码和预先培训的代孕模型可在以下网址获得：https://github.com/HashmatShadab/APR



## **39. Adversarial robustness of $β-$VAE through the lens of local geometry**

通过局部几何透镜分析$β-$VAE的对抗健壮性 cs.LG

The 2022 ICML Workshop on New Frontiers in Adversarial Machine  Learning

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.03923v1)

**Authors**: Asif Khan, Amos Storkey

**Abstracts**: Variational autoencoders (VAEs) are susceptible to adversarial attacks. An adversary can find a small perturbation in the input sample to change its latent encoding non-smoothly, thereby compromising the reconstruction. A known reason for such vulnerability is the latent space distortions arising from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in the inputs leads to a significant change in the latent space encodings. This paper demonstrates that the sensitivity around a data point is due to a directional bias of a stochastic pullback metric tensor induced by the encoder network. The pullback metric tensor measures the infinitesimal volume change from input to latent space. Thus, it can be viewed as a lens to analyse the effect of small changes in the input leading to distortions in the latent space. We propose robustness evaluation scores using the eigenspectrum of a pullback metric. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE.

摘要: 可变自动编码器(VAE)容易受到敌意攻击。攻击者可以在输入样本中发现微小的扰动，从而非平稳地改变其潜在编码，从而危及重建。这种脆弱性的一个已知原因是由于近似的潜在后验分布和先验分布之间的失配而引起的潜在空间扭曲。因此，输入的微小变化会导致潜在空间编码的显著变化。本文证明了数据点附近的敏感性是由编码器网络引起的随机拉回度量张量的方向偏差造成的。拉回度量张量测量从输入到潜在空间的无限小体积变化。因此，它可以看作是一个透镜，用来分析输入的微小变化导致潜在空间扭曲的影响。我们使用拉回度量的特征谱来提出稳健性评估分数。此外，我们的经验表明，得分与$\beta-$VAE的稳健性参数$\beta$相关。



## **40. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

对抗性后门防御微调：将后门攻击与对抗性攻击联系起来 cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2202.06312v3)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.

摘要: 众所周知，深度神经网络(DNN)既容易受到后门攻击，也容易受到对手攻击。在文献中，这两类攻击通常被视为不同的问题，分别属于训练时间攻击和推理时间攻击。然而，在本文中，我们发现了它们之间的一个有趣的联系：对于一个植入后门的模型，我们观察到其敌对示例与其触发样本具有相似的行为，即两者都激活了相同的DNN神经元子集。它表明，在模型中植入后门将显著影响模型的对抗性示例。基于这些观察结果，我们设计了一种新的对抗精调(AFT)算法来防御后门攻击。我们的实验表明，对于5种最先进的后门攻击，我们的AFT可以有效地清除后门触发，而在干净的样本上没有明显的性能下降，并且显著优于现有的防御方法。



## **41. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v2)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了一种隐私保护方案，该方案在保留VFL给主动方带来的全部好处的同时，恶化了对手的重构攻击。最后，实验结果验证了所提出的攻击和隐私保护方案的有效性。



## **42. Garbled EDA: Privacy Preserving Electronic Design Automation**

乱码EDA：保护隐私的电子设计自动化 cs.CR

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03822v1)

**Authors**: Mohammad Hashemi, Steffi Roy, Fatemeh Ganji, Domenic Forte

**Abstracts**: The complexity of modern integrated circuits (ICs) necessitates collaboration between multiple distrusting parties, including thirdparty intellectual property (3PIP) vendors, design houses, CAD/EDA tool vendors, and foundries, which jeopardizes confidentiality and integrity of each party's IP. IP protection standards and the existing techniques proposed by researchers are ad hoc and vulnerable to numerous structural, functional, and/or side-channel attacks. Our framework, Garbled EDA, proposes an alternative direction through formulating the problem in a secure multi-party computation setting, where the privacy of IPs, CAD tools, and process design kits (PDKs) is maintained. As a proof-of-concept, Garbled EDA is evaluated in the context of simulation, where multiple IP description formats (Verilog, C, S) are supported. Our results demonstrate a reasonable logical-resource cost and negligible memory overhead. To further reduce the overhead, we present another efficient implementation methodology, feasible when the resource utilization is a bottleneck, but the communication between two parties is not restricted. Interestingly, this implementation is private and secure even in the presence of malicious adversaries attempting to, e.g., gain access to PDKs or in-house IPs of the CAD tool providers.

摘要: 现代集成电路(IC)的复杂性需要多方合作，包括第三方知识产权(3PIP)供应商、设计公司、CAD/EDA工具供应商和铸造厂，这危及每一方知识产权的机密性和完整性。研究人员提出的IP保护标准和现有技术是特别的，容易受到许多结构性、功能性和/或旁路攻击。我们的框架，乱码EDA，通过在保护IP、CAD工具和工艺设计工具包(PDK)隐私的安全多方计算环境中描述问题，提出了另一种方向。作为概念验证，乱码EDA在支持多种IP描述格式(Verilog、C、S)的模拟环境中进行评估。我们的结果证明了合理的逻辑资源开销和可以忽略的内存开销。为了进一步减少开销，我们提出了另一种高效的实现方法，当资源利用率成为瓶颈时，该方法是可行的，但双方之间的通信不受限制。有趣的是，即使在恶意攻击者试图例如访问CAD工具提供商的PDK或内部IP的情况下，该实现也是私有和安全的。



## **43. Federated Adversarial Learning: A Framework with Convergence Analysis**

联合对抗性学习：一个收敛分析框架 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03635v1)

**Authors**: Xiaoxiao Li, Zhao Song, Jiaming Yang

**Abstracts**: Federated learning (FL) is a trending training paradigm to utilize decentralized training data. FL allows clients to update model parameters locally for several epochs, then share them to a global model for aggregation. This training paradigm with multi-local step updating before aggregation exposes unique vulnerabilities to adversarial attacks. Adversarial training is a popular and effective method to improve the robustness of networks against adversaries. In this work, we formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime. Our main result theoretically shows that the minimum loss under our algorithm can converge to $\epsilon$ small with chosen learning rate and communication rounds. It is noteworthy that our analysis is feasible for non-IID clients.

摘要: 联合学习(FL)是一种利用分散训练数据的训练范型。FL允许客户本地更新几个纪元的模型参数，然后将它们共享到全局模型以进行聚合。这种在聚集之前进行多局部步骤更新的训练范例暴露了独特的易受敌意攻击的弱点。对抗性训练是提高网络对抗敌手健壮性的一种流行而有效的方法。在这项工作中，我们提出了一种联邦对抗性学习(FAL)的一般形式，它是从集中式对抗性学习改编而来的。在FL训练的客户端，FAL有一个用于生成对抗性训练的对抗性样本的内环和一个用于更新局部模型参数的外环。在服务器端，FAL聚合本地模型更新并广播聚合的模型。我们设计了一个全局稳健的训练损失，并将FAL训练描述为一个最小-最大优化问题。与传统集中式训练中依赖于梯度方向的收敛分析不同，FAL的收敛分析明显困难，原因有三：1)最小-最大优化的复杂性；2)模型不能在梯度方向上更新；3)客户端在聚集前的多局部更新；3)客户端之间的异构性。我们通过使用适当的梯度近似和耦合技术来解决这些挑战，并给出了在过参数区域的收敛分析。理论上，我们的主要结果表明，在选择学习速率和通信轮数的情况下，我们的算法可以使最小损失收敛到较小的值。值得注意的是，我们的分析对非IID客户是可行的。



## **44. Blackbox Attacks via Surrogate Ensemble Search**

通过代理集成搜索进行黑盒攻击 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03610v1)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for blackbox attacks via surrogate ensemble search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet (including VGG-19, DenseNet-121, and ResNext-50). In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for non-targeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% non-targeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks.

摘要: 黑盒对抗性攻击可分为基于传输的攻击和基于查询的攻击。传输方法不需要来自受害者模型的任何反馈，但与基于查询的方法相比，提供了更低的成功率。查询攻击通常需要大量查询才能成功。为了达到这两种方法的最佳效果，最近的努力试图将它们结合起来，但仍然需要数百次查询才能获得高成功率(特别是针对有针对性的攻击)。在本文中，我们提出了一种新的方法，通过代理集成搜索(基)可以用极少的查询生成高度成功的黑盒攻击。我们首先定义了一种微扰机，它通过最小化一组固定代理模型上的加权损失函数来生成扰动图像。为了针对给定的受害者模型生成攻击，我们使用由扰动机器生成的查询来搜索损失函数中的权重。由于搜索空间的维度很小(与代理模型的数量相同)，因此搜索需要少量的查询。我们的实验结果表明，在使用ImageNet(包括VGG-19、DenseNet-121和ResNext-50)训练的不同图像分类器上，我们提出的方法获得了更好的成功率，查询次数至少减少了30倍。特别是，我们的方法只需每幅图像3个查询(平均)就可以实现90%以上的定向攻击成功率和1-2个查询的非定向攻击成功率99%以上。我们的方法在Google Cloud Vision API上也是有效的，在每张图片2.9个查询的情况下，实现了91%的非定向攻击成功率。我们还证明了我们提出的方法产生的扰动具有很高的可转移性，可以用于硬标签黑盒攻击。



## **45. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

未知聚类个数在线聚类的重访高斯神经元 cs.LG

Reviewed at  https://openreview.net/forum?id=h05RLBNweX&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.00920v2)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. Although these weaknesses are not specifically addressed, a novel local learning rule is presented that performs online clustering with an upper limit on the number of clusters to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.

摘要: 尽管人工神经网络最近取得了成功，但可能需要更多生物学上可信的学习方法来解决反向传播训练模型的弱点，如灾难性遗忘和对抗性攻击。虽然没有具体解决这些缺点，但提出了一种新的局部学习规则，该规则执行在线聚类时，对要发现的簇的数量设置上限，而不是固定的簇计数。不使用正交权重或输出激活约束，而是通过侧向高斯神经元的相互排斥来获得激活稀疏性，以确保多个神经元中心不会占据输入域中的相同位置。在数据样本可以用均值和方差表示的情况下，提出了一种调整高斯神经元宽度的更新方法。这些算法被应用于MNIST和CIFAR-10数据集，以创建捕捉不同大小的像素斑块的输入模式的过滤器。实验结果表明，在大量的训练样本中，学习的参数是稳定的。



## **46. On the Fundamental Limits of Formally (Dis)Proving Robustness in Proof-of-Learning**

学习证明中形式(Dis)证明稳健性的基本极限 cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03567v1)

**Authors**: Congyu Fang, Hengrui Jia, Anvith Thudi, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Varun Chandrasekaran, Nicolas Papernot

**Abstracts**: Proof-of-learning (PoL) proposes a model owner use machine learning training checkpoints to establish a proof of having expended the necessary compute for training. The authors of PoL forego cryptographic approaches and trade rigorous security guarantees for scalability to deep learning by being applicable to stochastic gradient descent and adaptive variants. This lack of formal analysis leaves the possibility that an attacker may be able to spoof a proof for a model they did not train.   We contribute a formal analysis of why the PoL protocol cannot be formally (dis)proven to be robust against spoofing adversaries. To do so, we disentangle the two roles of proof verification in PoL: (a) efficiently determining if a proof is a valid gradient descent trajectory, and (b) establishing precedence by making it more expensive to craft a proof after training completes (i.e., spoofing). We show that efficient verification results in a tradeoff between accepting legitimate proofs and rejecting invalid proofs because deep learning necessarily involves noise. Without a precise analytical model for how this noise affects training, we cannot formally guarantee if a PoL verification algorithm is robust. Then, we demonstrate that establishing precedence robustly also reduces to an open problem in learning theory: spoofing a PoL post hoc training is akin to finding different trajectories with the same endpoint in non-convex learning. Yet, we do not rigorously know if priori knowledge of the final model weights helps discover such trajectories.   We conclude that, until the aforementioned open problems are addressed, relying more heavily on cryptography is likely needed to formulate a new class of PoL protocols with formal robustness guarantees. In particular, this will help with establishing precedence. As a by-product of insights from our analysis, we also demonstrate two novel attacks against PoL.

摘要: 学习证明(POL)建议模型所有者使用机器学习训练检查点来建立已经为训练花费了必要的计算机的证明。POL FOREO密码方法的作者通过适用于随机梯度下降和自适应变体，在可伸缩性到深度学习的严格安全保证之间进行权衡。由于缺乏正式的分析，攻击者有可能伪造他们没有训练过的模型的证据。我们对POL协议为什么不能被形式化(DIS)证明对欺骗对手是健壮的进行了正式的分析。为此，我们将POL中证明验证的两个角色分开：(A)有效地确定证明是否为有效的梯度下降轨迹，以及(B)通过使训练完成后制作证明的成本更高(即，欺骗)来建立优先级。我们证明了有效的验证在接受合法证明和拒绝无效证明之间产生了折衷，因为深度学习必然涉及噪声。如果没有准确的分析模型来分析噪声如何影响训练，我们就不能正式保证POL验证算法是否健壮。然后，我们证明了稳健地建立优先权也归结为学习理论中的一个开放问题：欺骗POL后自组织训练类似于在非凸学习中找到具有相同终点的不同轨迹。然而，我们并不确切地知道最终模型权重的先验知识是否有助于发现这样的轨迹。我们的结论是，在上述公开问题得到解决之前，很可能需要更多地依赖密码学来制定具有形式健壮性保证的新型POL协议。特别是，这将有助于确立优先地位。作为我们分析的副产品，我们还演示了针对POL的两种新攻击。



## **47. Preventing or Mitigating Adversarial Supply Chain Attacks; a legal analysis**

预防或减轻对抗性供应链攻击；法律分析 cs.CY

23 pages

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03466v1)

**Authors**: Kaspar Rosager Ludvigsen, Shishir Nagaraja, Angela Daly

**Abstracts**: The world is currently strongly connected through both the internet at large, but also the very supply chains which provide everything from food to infrastructure and technology. The supply chains are themselves vulnerable to adversarial attacks, both in a digital and physical sense, which can disrupt or at worst destroy them. In this paper, we take a look at two examples of such successful attacks and consider what their consequences may be going forward, and analyse how EU and national law can prevent these attacks or otherwise punish companies which do not try to mitigate them at all possible costs. We find that the current types of national regulation are not technology specific enough, and cannot force or otherwise mandate the correct parties who could play the biggest role in preventing supply chain attacks to do everything in their power to mitigate them. But, current EU law is on the right path, and further vigilance may be what is necessary to consider these large threats, as national law tends to fail at properly regulating companies when it comes to cybersecurity.

摘要: 目前，世界通过互联网和供应链紧密相连，供应链提供从食品到基础设施和技术的一切东西。供应链本身在数字和物理意义上都很容易受到敌意攻击，这些攻击可能会扰乱供应链，甚至在最坏的情况下摧毁它们。在这篇文章中，我们看了两个此类成功攻击的例子，并考虑它们未来可能产生的后果，并分析欧盟和国家法律如何防止这些攻击或以其他方式惩罚那些不试图不惜一切代价减轻攻击的公司。我们发现，目前的国家监管类型不够具体，不能强迫或以其他方式强制正确的各方尽其所能缓解供应链攻击，这些各方可以在防止供应链攻击方面发挥最大作用。但是，当前的欧盟法律走在正确的道路上，进一步的警惕可能是考虑这些重大威胁所必需的，因为在网络安全方面，各国法律往往无法对公司进行适当的监管。



## **48. Searching for the Essence of Adversarial Perturbations**

寻找对抗性扰动的本质 cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.15357v2)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance in various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is shown to fool neural networks' predictions. This would lead to potential risks for real-world applications such as endangering autonomous driving and messing up text identification. To mitigate such risks, an understanding of how adversarial examples operate is critical, which however remains unresolved. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction, in contrast to a widely discussed argument that human-imperceptible information plays the critical role in fooling a network. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, including the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.

摘要: 神经网络在不同的机器学习领域取得了最先进的性能，然而在输入数据中加入恶意扰动(对抗性的例子)被证明愚弄了神经网络的预测。这将给现实世界的应用带来潜在风险，如危及自动驾驶和扰乱文本识别。为了减轻这种风险，了解对抗性案例如何运作是至关重要的，但这一问题仍未得到解决。在这里，我们证明了对抗性扰动包含人类可识别的信息，这是导致神经网络错误预测的关键阴谋者，而不是广泛讨论的人类不可感知的信息在愚弄网络方面发挥关键作用的论点。这一人类可识别信息的概念允许我们解释与对抗性扰动相关的关键特征，包括对抗性例子的存在，不同神经网络之间的可转换性，以及用于对抗性训练的神经网络更高的可解释性。揭示了欺骗神经网络的对抗性扰动中的两个独特性质：掩蔽和生成。当神经网络对输入图像进行分类时，识别出一种特殊的类，即互补类。敌意干扰中包含的人类可识别的信息使研究人员能够深入了解神经网络的工作原理，并可能导致开发检测/防御敌意攻击的技术。



## **49. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

不确定性感知深度模型的成功依赖于数据流形几何 cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.01705v2)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.

摘要: 为了在安全关键环境中做出负责任的决策，机器学习模型必须有效地检测和处理边缘案例数据。虽然现有的工作表明，预测不确定性对这些任务是有用的，但从文献中并不明显地看到，哪些不确定性感知模型最适合给定的数据集。因此，我们在一组边缘情况任务上比较了六种不确定性感知的深度学习模型：对对手攻击的健壮性以及分布外和对抗性检测。我们发现，数据子流形的几何形状是决定各种模型成功与否的重要因素。我们的发现为不确定性感知深度学习模型的研究提供了一个有趣的方向。



## **50. Attacking Adversarial Defences by Smoothing the Loss Landscape**

通过平滑损失图景来攻击对抗性防御 cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.

摘要: 本文研究了一系列防御对手攻击的方法，这些攻击的成功部分归因于创建了一个嘈杂的、不连续的或以其他方式崎岖的损失场景，对手发现很难导航。实现这一效果的一种常见但并不普遍的方法是通过使用随机神经网络。我们证明了这是一种梯度混淆的形式，并提出了一种基于魏尔斯特拉斯变换的对基于梯度的攻击的一般扩展，它平滑了损失函数的表面，并提供了更可靠的梯度估计。我们进一步证明，同样的原理可以加强无梯度的对手。我们证明了我们的损失平滑方法对随机和非随机对抗防御的有效性，这些防御由于这种类型的混淆而表现出稳健性。此外，我们还分析了它如何与变换上的期望相互作用，变换上的期望是目前用于攻击随机防御的一种流行的梯度抽样方法。



