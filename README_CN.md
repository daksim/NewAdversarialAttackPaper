# Latest Adversarial Attack Papers
**update at 2022-03-03 06:31:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Proceedings of the Artificial Intelligence for Cyber Security (AICS) Workshop at AAAI 2022**

2022年AAAI 2022年网络安全人工智能(AICS)研讨会论文集 cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.14010v2)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.

摘要: 研讨会将重点讨论人工智能在网络安全问题上的应用。网络系统产生了大量的数据，有效地利用这些数据超出了人类的能力范围。此外，对手还在继续开发新的攻击。因此，需要人工智能方法来理解和保护网络领域。这些挑战在企业网络中得到了广泛的研究，但在研究和实践中还存在许多差距，在其他领域也出现了一些新的问题。总的来说，人工智能技术在现实世界中仍然没有被广泛采用。原因包括：(1)缺乏对人工智能安全的认证，(2)缺乏对网络领域中实际限制(例如，电力、内存、存储)对人工智能系统的影响的正式研究，(3)已知的漏洞，如逃避、中毒攻击，(4)缺乏对安全分析师的有意义的解释，以及(5)分析师对人工智能解决方案缺乏信任。研究界需要为这些实际问题开发新的解决方案。



## **2. Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks**

超越梯度：在模型反转攻击中利用对抗性先验 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00481v1)

**Authors**: Dmitrii Usynin, Daniel Rueckert, Georgios Kaissis

**Abstracts**: Collaborative machine learning settings like federated learning can be susceptible to adversarial interference and attacks. One class of such attacks is termed model inversion attacks, characterised by the adversary reverse-engineering the model to extract representations and thus disclose the training data. Prior implementations of this attack typically only rely on the captured data (i.e. the shared gradients) and do not exploit the data the adversary themselves control as part of the training consortium. In this work, we propose a novel model inversion framework that builds on the foundations of gradient-based model inversion attacks, but additionally relies on matching the features and the style of the reconstructed image to data that is controlled by an adversary. Our technique outperforms existing gradient-based approaches both qualitatively and quantitatively, while still maintaining the same honest-but-curious threat model, allowing the adversary to obtain enhanced reconstructions while remaining concealed.

摘要: 协作式机器学习环境(如联合学习)很容易受到敌意干扰和攻击。一类这样的攻击被称为模型反转攻击，其特征是对手对模型进行逆向工程以提取表示，从而泄露训练数据。该攻击的先前实现通常仅依赖于捕获的数据(即共享梯度)，并且不利用对手自己控制的数据作为训练联盟的一部分。在这项工作中，我们提出了一种新的模型反演框架，它建立在基于梯度的模型反演攻击的基础上，但另外还依赖于将重建图像的特征和样式与对手控制的数据进行匹配。我们的技术在质量和数量上都优于现有的基于梯度的方法，同时仍然保持相同的诚实但好奇的威胁模型，允许攻击者在保持隐蔽的情况下获得增强的重建。



## **3. RAB: Provable Robustness Against Backdoor Attacks**

RAB：针对后门攻击的可证明的健壮性 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2003.08904v6)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We theoretically prove the robustness bound for machine learning models trained with RAB, and prove that our robustness bound is tight. We derive the robustness conditions for different smoothing distributions including Gaussian and uniform distributions. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm which eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning models such as DNNs and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed lights on further robust learning strategies against general training time attacks.

摘要: 最近的研究表明，深层神经网络容易受到敌意攻击，包括逃避和后门(中毒)攻击。在防御方面，已经进行了密集的努力来提高针对规避攻击的经验性和可证明的健壮性；然而，针对后门攻击的可证明的健壮性在很大程度上仍未得到探索。在本文中，我们重点验证机器学习模型对一般威胁模型，特别是后门攻击的鲁棒性。我们首先通过随机平滑技术提供了一个统一的框架，并展示了如何将其实例化来证明对规避和后门攻击的鲁棒性。然后，我们提出了第一个鲁棒训练过程RAB，以平滑训练的模型并证明其对后门攻击的鲁棒性。从理论上证明了RAB训练的机器学习模型的稳健界，并证明了我们的稳健界是紧的。我们推导了不同平滑分布(包括高斯分布和均匀分布)的稳健性条件。此外，我们从理论上证明了对于K近邻分类器等简单模型，可以有效地训练鲁棒平滑模型，并提出了一种精确的平滑训练算法，该算法消除了对此类模型从噪声分布中采样的需要。经验上，我们在MNIST、CIFAR-10和ImageNette数据集上对不同的机器学习模型(如DNNS和K-NN模型)进行了全面的实验，并提供了第一个经验证的针对后门攻击的健壮性基准。此外，我们在垃圾邮件库表格数据集上对K-NN模型进行了评估，以展示所提出的精确算法的优势。理论分析和对不同ML模型和数据集的综合评价，为进一步研究抗一般训练时间攻击的鲁棒学习策略提供了理论依据。



## **4. Adversarial samples for deep monocular 6D object pose estimation**

用于深部单目6维目标姿态估计的对抗性样本 cs.CV

15 pages. arXiv admin note: text overlap with arXiv:2105.14291 by  other authors

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00302v1)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating object 6D pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping, where robustness of the estimation is crucial. In this work, for the first time, we study adversarial samples that can fool state-of-the-art (SOTA) deep learning based 6D pose estimation models. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack all the three main categories of models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object shapes that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. By shifting the segmentation attention map away from its original position, adversarial samples are crafted. We show that such adversarial samples are not only effective for the direct 6D pose estimation models, but also able to attack the two-stage based models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness of our U6DA with large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.

摘要: 从RGB图像估计物体6D姿态对于许多真实世界的应用非常重要，例如自动驾驶和机器人抓取，其中估计的健壮性至关重要。在这项工作中，我们首次研究了可以欺骗基于SOTA深度学习的6D姿态估计模型的对抗性样本。特别地，我们提出了一种统一的6D位姿估计攻击，即U6DA，它可以成功地攻击所有三种主要的6D位姿估计模型。我们的U6DA的关键思想是愚弄模型来预测对象形状的错误结果，这对于正确的6D姿势估计是必不可少的。具体地说，我们探索了一种基于传输的黑盒攻击来进行6D位姿估计。通过将分割注意图从其原始位置移开，可以制作对抗性样本。结果表明，这种对抗性样本不仅对直接6D姿态估计模型有效，而且能够攻击基于两阶段的模型，而不考虑其稳健的RANSAC模型。通过大规模的公共基准测试，验证了我们的U6DA算法的有效性。我们还介绍了一个新的U6DA-Linemod数据集，用于6D位姿估计任务的鲁棒性研究。我们的代码和数据集将在\url{https://github.com/cuge1995/U6DA}.



## **5. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

基于混合对抗训练的鲁棒堆叠式胶囊自动编码器 cs.CV

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.13755v2)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.

摘要: 胶囊网络(CapsNets)是一种基于特征空间关系对图像进行分类的新型神经网络。通过分析特征的姿态及其相对位置，使仿射变换后的图像具有更强的识别能力。堆叠式胶囊自动编码器(SCAE)是一种先进的CapsNet，首次实现了CapsNet的无监督分类。然而，SCAE的安全漏洞和健壮性很少被研究。本文提出了一种针对SCAE的规避攻击，攻击者可以通过减少SCAE中对象胶囊相对于图像原始类别的贡献来产生敌意扰动。然后将对抗性扰动应用于原始图像，并且扰动图像将被错误分类。此外，针对此类逃避攻击，我们提出了一种称为混合对抗训练(HAT)的防御方法。HAT利用对抗性训练和对抗性蒸馏来实现更好的健壮性和稳定性。实验结果表明，改进后的SCAE模型在规避攻击下可以达到82.14%的分类正确率。源代码可以在https://github.com/FrostbiteXSW/SCAE_Defense.上找到



## **6. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.12154v2)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a simple trigger and targeting only one class to using many sophisticated triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. In this paper, we advocate general defenses that are effective and robust against various Trojan attacks and propose two novel "filtering" defenses with these characteristics called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF). VIF and AIF leverage variational inference and adversarial training respectively to purify all potential Trojan triggers in the input at run time without making any assumption about their numbers and forms. We further extend "filtering" to "filtering-then-contrasting" - a new defense mechanism that helps avoid the drop in classification accuracy on clean data caused by filtering. Extensive experimental results show that our proposed defenses significantly outperform 4 well-known defenses in mitigating 5 different Trojan attacks including the two state-of-the-art which defeat many strong defenses.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年里，特洛伊木马攻击已经从只使用一个简单的触发器，只针对一个类，发展到使用许多复杂的触发器，针对多个类。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马触发器和目标类做出过时的假设，因此很容易被现代木马攻击所规避。在本文中，我们提倡对各种特洛伊木马攻击有效和健壮的通用防御，并提出了两种具有这些特性的新型“过滤”防御方案，称为变量输入过滤(VIF)和对抗性输入过滤(AIF)。VIF和AIF分别利用变分推理和对抗性训练在运行时净化输入中所有潜在的特洛伊木马触发器，而不对其数量和形式做出任何假设。我们将“过滤”进一步扩展为“过滤-然后对比”-一种新的防御机制，它有助于避免过滤导致的干净数据分类准确率的下降。广泛的实验结果表明，我们提出的防御方案在缓解5种不同的特洛伊木马攻击方面明显优于4种众所周知的防御方案，其中包括两种最先进的防御方案，它们击败了许多强大的防御方案。



## **7. Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**

基于有限知识的图嵌入模型的对抗性攻击框架 cs.LG

Journal extension of GF-Attack, accepted by TKDE

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2105.12419v2)

**Authors**: Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Xin Wang, Wenwu Zhu, Junzhou Huang

**Abstracts**: With the success of the graph embedding model in both academic and industry areas, the robustness of graph embedding against adversarial attack inevitably becomes a crucial problem in graph learning. Existing works usually perform the attack in a white-box fashion: they need to access the predictions/labels to construct their adversarial loss. However, the inaccessibility of predictions/labels makes the white-box attack impractical to a real graph learning system. This paper promotes current frameworks in a more general and flexible sense -- we demand to attack various kinds of graph embedding models with black-box driven. We investigate the theoretical connections between graph signal processing and graph embedding models and formulate the graph embedding model as a general graph signal process with a corresponding graph filter. Therefore, we design a generalized adversarial attacker: GF-Attack. Without accessing any labels and model predictions, GF-Attack can perform the attack directly on the graph filter in a black-box fashion. We further prove that GF-Attack can perform an effective attack without knowing the number of layers of graph embedding models. To validate the generalization of GF-Attack, we construct the attacker on four popular graph embedding models. Extensive experiments validate the effectiveness of GF-Attack on several benchmark datasets.

摘要: 随着图嵌入模型在学术界和工业界的成功应用，图嵌入对敌意攻击的鲁棒性不可避免地成为图学习中的一个关键问题。现有的作品通常以白盒方式进行攻击：它们需要访问预测/标签来构建它们的对抗性损失。然而，预测/标签的不可访问性使得白盒攻击对于真实的图学习系统来说是不切实际的。本文在更一般、更灵活的意义上提升了现有的框架--我们要求攻击各种黑盒驱动的图嵌入模型。我们研究了图信号处理和图嵌入模型之间的理论联系，并将图嵌入模型表示为具有相应图过滤的一般图信号过程。因此，我们设计了一种广义对抗性攻击者：GF-攻击。GF-Attack在不访问任何标签和模型预测的情况下，可以黑盒方式直接对图过滤进行攻击。进一步证明了GF-攻击可以在不知道图嵌入模型层数的情况下进行有效的攻击。为了验证GF-攻击的泛化能力，我们在四种流行的图嵌入模型上构造了攻击者。在几个基准数据集上的大量实验验证了GF-攻击的有效性。



## **8. Load-Altering Attacks Against Power Grids under COVID-19 Low-Inertia Conditions**

冠状病毒低惯性条件下电网的变负荷攻击 cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2201.10505v2)

**Authors**: Subhash Lakshminarayana, Juan Ospina, Charalambos Konstantinou

**Abstracts**: The COVID-19 pandemic has impacted our society by forcing shutdowns and shifting the way people interacted worldwide. In relation to the impacts on the electric grid, it created a significant decrease in energy demands across the globe. Recent studies have shown that the low demand conditions caused by COVID-19 lockdowns combined with large renewable generation have resulted in extremely low-inertia grid conditions. In this work, we examine how an attacker could exploit these {scenarios} to cause unsafe grid operating conditions by executing load-altering attacks (LAAs) targeted at compromising hundreds of thousands of IoT-connected high-wattage loads in low-inertia power systems. Our study focuses on analyzing the impact of the COVID-19 mitigation measures on U.S. regional transmission operators (RTOs), formulating a plausible and realistic least-effort LAA targeted at transmission systems with low-inertia conditions, and evaluating the probability of these large-scale LAAs. Theoretical and simulation results are presented based on the WSCC 9-bus {and IEEE 118-bus} test systems. Results demonstrate how adversaries could provoke major frequency disturbances by targeting vulnerable load buses in low-inertia systems and offer insights into how the temporal fluctuations of renewable energy sources, considering generation scheduling, impact the grid's vulnerability to LAAs.

摘要: 冠状病毒大流行已经通过迫使政府关门和改变世界各地人们互动的方式影响了我们的社会。在对电网的影响方面，它造成了全球能源需求的显著下降。最近的研究表明，冠状病毒关闭造成的低需求条件，加上大量的可再生能源发电，导致了极低的惯性电网条件。在这项工作中，我们研究了攻击者如何利用这些{场景}通过执行负载改变攻击(LAA)来造成不安全的电网运行条件，LAA的目标是危害低惯性电力系统中数十万个物联网连接的高瓦数负载。我们的研究重点是分析冠状病毒缓解措施对美国区域传输运营商(RTO)的影响，针对低惯性条件下的传输系统制定合理而现实的最小工作量LAA，并评估这些大规模LAA的可能性。给出了基于WSCC9母线{和IEEE118母线}测试系统的理论和仿真结果。结果表明，对手如何通过瞄准低惯性系统中脆弱的负载母线来引发重大频率扰动，并为考虑发电调度的可再生能源的时间波动如何影响电网的LAAS脆弱性提供了深入的见解。



## **9. MaMaDroid2.0 -- The Holes of Control Flow Graphs**

MaMaDroid2.0--控制流图的漏洞 cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13922v1)

**Authors**: Harel Berger, Chen Hajaj, Enrico Mariconti, Amit Dvir

**Abstracts**: Android malware is a continuously expanding threat to billions of mobile users around the globe. Detection systems are updated constantly to address these threats. However, a backlash takes the form of evasion attacks, in which an adversary changes malicious samples such that those samples will be misclassified as benign. This paper fully inspects a well-known Android malware detection system, MaMaDroid, which analyzes the control flow graph of the application. Changes to the portion of benign samples in the train set and models are considered to see their effect on the classifier. The changes in the ratio between benign and malicious samples have a clear effect on each one of the models, resulting in a decrease of more than 40% in their detection rate. Moreover, adopted ML models are implemented as well, including 5-NN, Decision Tree, and Adaboost. Exploration of the six models reveals a typical behavior in different cases, of tree-based models and distance-based models. Moreover, three novel attacks that manipulate the CFG and their detection rates are described for each one of the targeted models. The attacks decrease the detection rate of most of the models to 0%, with regards to different ratios of benign to malicious apps. As a result, a new version of MaMaDroid is engineered. This model fuses the CFG of the app and static analysis of features of the app. This improved model is proved to be robust against evasion attacks targeting both CFG-based models and static analysis models, achieving a detection rate of more than 90% against each one of the attacks.

摘要: Android恶意软件正在对全球数十亿移动用户构成持续扩大的威胁。探测系统会不断更新，以应对这些威胁。然而，反弹采取逃避攻击的形式，在这种攻击中，敌手更改恶意样本，使这些样本被错误地归类为良性样本。本文全面考察了著名的Android恶意软件检测系统MaMaDroid，该系统分析了应用程序的控制流图。考虑对训练集和模型中良性样本部分的改变，以查看它们对分类器的影响。良性样本和恶意样本比例的变化对每个模型都有明显的影响，导致它们的检测率下降了40%以上。此外，还实现了所采用的ML模型，包括5-NN、决策树和Adboost。对这六种模型的研究揭示了基于树的模型和基于距离的模型在不同情况下的典型行为。此外，对于每个目标模型，描述了操纵CFG的三种新攻击及其检测率。这些攻击将大多数模型的检测率降低到0%，涉及到不同比例的良性应用程序和恶意应用程序。因此，一个新版本的MaMaDroid被设计出来。这种模式融合了APP的CFG和静电对APP功能的分析。实验证明，该改进模型对基于cfg模型和静电分析模型的规避攻击均具有较强的鲁棒性，对每种攻击的检测率均在90%以上。



## **10. Formally verified asymptotic consensus in robust networks**

鲁棒网络中渐近一致性的形式化验证 cs.PL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13833v1)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.

摘要: 分布式体系结构被用来提高各种系统的性能和可靠性。分布式体系结构的一项重要功能是在其所有节点之间达成共识的能力。为了实现这一点，已经针对不同的场景提出了几种共识算法，其中许多算法都带有不经过机械检查的正确性证明。不幸的是，众所周知，这些证明错综复杂，容易出错。本文对一种广泛应用于分布式控制领域的一致性算法--由Le Blanc等人提出的加权平均子序列简化(W-MSR)算法进行了形式化和机械检验。该算法提供了一种在分布式控制场景中获得渐近共识的方法，在存在可能不会基于名义共识协议更新其状态并且可能与其邻居共享不准确信息的敌对代理(攻击者)存在的情况下，该算法提供了一种获得渐近共识的方法。利用CoQ证明助手，我们形式化了在假设的攻击者模型下实现弹性渐近共识所需的充要条件。我们利用现有的图论、有限集和Mathcomp库序列的CoQ形式化进行开发。据我们所知，这是渐近一致算法的第一个机械证明。在形式化过程中，我们澄清了论文证明中的几个不精确之处，包括主要定理中关于量词的不精确。



## **11. Robust Textual Embedding against Word-level Adversarial Attacks**

抵抗词级敌意攻击的鲁棒文本嵌入 cs.CL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13817v1)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull the words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows the great potential of improving the textual robustness through robust word embedding.

摘要: 我们将自然语言处理模型的脆弱性归因于相似的输入在嵌入空间被转换为不相似的表示，从而导致输出不一致，并提出了一种新的鲁棒训练方法，称为快速三重度量学习(Fast Triplet Metric Learning，FTML)。具体地说，我们认为原始样本应该与对手样本具有相似的表示，并将其表示与其他样本区分开来，以获得更好的鲁棒性。为此，我们将三元组度量学习引入标准训练中，将单词拉近其正样本(即同义词)，并在嵌入空间中推开其负样本(即非同义词)。大量实验表明，该算法在保持好胜原始样本分类准确率的同时，能显着提高模型对各种高级敌意攻击的鲁棒性。此外，我们的方法是高效的，因为它只需要调整嵌入，并且对标准训练的开销很小。我们的工作显示了通过鲁棒的词嵌入来提高文本鲁棒性的巨大潜力。



## **12. On the Robustness of CountSketch to Adaptive Inputs**

关于CountSketch对自适应输入的鲁棒性 cs.DS

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13736v1)

**Authors**: Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer

**Abstracts**: CountSketch is a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We study the robustness of the sketch in adaptive settings where input vectors may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior work.

摘要: CountSketch是一种流行的降维技术，它使用随机化的线性测量将向量映射到较低的维度。草图支持恢复向量的$\ell_2$重打击数(具有$v[i]^2\geq\frac{1}{k}\|\boldSymbol{v}\^2_2$的条目)。我们研究了草图在自适应环境下的鲁棒性，其中输入向量可能依赖于先前输入的输出。自适应设置出现在具有反馈或敌意攻击的过程中。我们证明了经典的估计器是不稳健的，并且可以用草图大小的数量级的一些查询来攻击。我们提出了一个稳健的估计器(对于略微修改的草图)，它允许草图大小的二次查询数，这比以前的工作提高了$\sqrt{k}$(对于$k$重命中者是$\sqrt{k}$)。



## **13. An Empirical Study on the Intrinsic Privacy of SGD**

关于SGD内在隐私性的实证研究 cs.LG

21 pages, 11 figures, 8 tables

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/1912.02919v4)

**Authors**: Stephanie L. Hyland, Shruti Tople

**Abstracts**: Introducing noise in the training of machine learning systems is a powerful way to protect individual privacy via differential privacy guarantees, but comes at a cost to utility. This work looks at whether the inherent randomness of stochastic gradient descent (SGD) could contribute to privacy, effectively reducing the amount of \emph{additional} noise required to achieve a given privacy guarantee. We conduct a large-scale empirical study to examine this question. Training a grid of over 120,000 models across four datasets (tabular and images) on convex and non-convex objectives, we demonstrate that the random seed has a larger impact on model weights than any individual training example. We test the distribution over weights induced by the seed, finding that the simple convex case can be modelled with a multivariate Gaussian posterior, while neural networks exhibit multi-modal and non-Gaussian weight distributions. By casting convex SGD as a Gaussian mechanism, we then estimate an `intrinsic' data-dependent $\epsilon_i(\mathcal{D})$, finding values as low as 6.3, dropping to 1.9 using empirical estimates. We use a membership inference attack to estimate $\epsilon$ for non-convex SGD and demonstrate that hiding the random seed from the adversary results in a statistically significant reduction in attack performance, corresponding to a reduction in the effective $\epsilon$. These results provide empirical evidence that SGD exhibits appreciable variability relative to its dataset sensitivity, and this `intrinsic noise' has the potential to be leveraged to improve the utility of privacy-preserving machine learning.

摘要: 在机器学习系统的训练中引入噪声是通过不同的隐私保证来保护个人隐私的一种强有力的方式，但这是以实用为代价的。这项工作着眼于随机梯度下降(SGD)固有的随机性是否有助于隐私，从而有效地减少实现给定隐私保证所需的{附加}噪声量。为了检验这一问题，我们进行了大规模的实证研究。通过对四个数据集(表格和图像)上超过12万个模型的网格进行凸和非凸目标的训练，我们证明了随机种子对模型权重的影响比任何单个训练示例都要大。我们对种子引起的权值分布进行了测试，发现简单的凸情况可以用多元高斯后验分布来建模，而神经网络则表现出多峰和非高斯权重分布。通过把凸SGD看作一个高斯机制，然后我们估计一个“固有的”依赖于数据的$\epsilon_i(\mathcal{D})$，得到低至6.3的值，用经验估计降到1.9。我们使用成员关系推理攻击来估计非凸SGD的$\epsilon$，并证明了对对手隐藏随机种子会导致攻击性能在统计上显著降低，相应地，有效的$\epsilon$也会降低。这些结果提供了经验证据，表明SGD相对于其数据集敏感度表现出明显的变异性，并且这种“固有噪声”有可能被用来提高隐私保护机器学习的效用。



## **14. Enhance transferability of adversarial examples with model architecture**

利用模型架构增强对抗性实例的可移植性 cs.LG

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13625v1)

**Authors**: Mingyuan Fan, Wenzhong Guo, Shengxing Yu, Zuobin Ying, Ximeng Liu

**Abstracts**: Transferability of adversarial examples is of critical importance to launch black-box adversarial attacks, where attackers are only allowed to access the output of the target model. However, under such a challenging but practical setting, the crafted adversarial examples are always prone to overfitting to the proxy model employed, presenting poor transferability. In this paper, we suggest alleviating the overfitting issue from a novel perspective, i.e., designing a fitted model architecture. Specifically, delving the bottom of the cause of poor transferability, we arguably decompose and reconstruct the existing model architecture into an effective model architecture, namely multi-track model architecture (MMA). The adversarial examples crafted on the MMA can maximumly relieve the effect of model-specified features to it and toward the vulnerable directions adopted by diverse architectures. Extensive experimental evaluation demonstrates that the transferability of adversarial examples based on the MMA significantly surpass other state-of-the-art model architectures by up to 40% with comparable overhead.

摘要: 对抗性示例的可转移性对于发起黑盒对抗性攻击至关重要，在黑盒对抗性攻击中，攻击者只能访问目标模型的输出。然而，在这样一个具有挑战性但实用的背景下，精心制作的对抗性例子往往容易与所采用的代理模型过度拟合，表现出较差的可移植性。在本文中，我们建议从一个新的角度来缓解过度拟合问题，即设计一个合适的模型体系结构。具体地说，通过深入挖掘可移植性差的底层原因，我们可以将现有的模型体系结构分解和重构为一种有效的模型体系结构，即多轨道模型体系结构(MMA)。在MMA上制作的对抗性示例可以最大限度地缓解模型指定功能对MMA的影响，以及对不同体系结构采用的易受攻击方向的影响。广泛的实验评估表明，基于MMA的对抗性示例的可转移性在开销相当的情况下大大超过了其他最先进的模型体系结构，最高可达40%。



## **15. GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems**

石墨：为机器学习攻击计算机视觉系统生成自动物理示例 cs.CR

IEEE European Symposium on Security and Privacy 2022 (EuroS&P 2022)

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2002.07088v6)

**Authors**: Ryan Feng, Neal Mangaokar, Jiefeng Chen, Earlence Fernandes, Somesh Jha, Atul Prakash

**Abstracts**: This paper investigates an adversary's ease of attack in generating adversarial examples for real-world scenarios. We address three key requirements for practical attacks for the real-world: 1) automatically constraining the size and shape of the attack so it can be applied with stickers, 2) transform-robustness, i.e., robustness of a attack to environmental physical variations such as viewpoint and lighting changes, and 3) supporting attacks in not only white-box, but also black-box hard-label scenarios, so that the adversary can attack proprietary models. In this work, we propose GRAPHITE, an efficient and general framework for generating attacks that satisfy the above three key requirements. GRAPHITE takes advantage of transform-robustness, a metric based on expectation over transforms (EoT), to automatically generate small masks and optimize with gradient-free optimization. GRAPHITE is also flexible as it can easily trade-off transform-robustness, perturbation size, and query count in black-box settings. On a GTSRB model in a hard-label black-box setting, we are able to find attacks on all possible 1,806 victim-target class pairs with averages of 77.8% transform-robustness, perturbation size of 16.63% of the victim images, and 126K queries per pair. For digital-only attacks where achieving transform-robustness is not a requirement, GRAPHITE is able to find successful small-patch attacks with an average of only 566 queries for 92.2% of victim-target pairs. GRAPHITE is also able to find successful attacks using perturbations that modify small areas of the input image against PatchGuard, a recently proposed defense against patch-based attacks.

摘要: 本文调查了一个对手在为真实世界场景生成敌意示例时的攻击易用性。我们解决了现实世界中实际攻击的三个关键要求：1)自动约束攻击的大小和形状，使其可以与贴纸一起应用；2)变换鲁棒性，即攻击对环境物理变化(如视点和光照变化)的鲁棒性；3)不仅支持白盒攻击，而且支持黑盒硬标签场景下的攻击，以便攻击者可以攻击专有模型。在这项工作中，我们提出了石墨，这是一个高效和通用的框架，用于生成满足上述三个关键要求的攻击。石墨利用变换稳健性(一种基于期望重于变换(EoT)的度量)自动生成小掩码，并通过无梯度优化进行优化。石墨还很灵活，因为它可以很容易地权衡黑盒设置中的转换健壮性、扰动大小和查询计数。在硬标签黑盒环境下的GTSRB模型上，我们能够发现对所有可能的1806个受害者-目标类对的攻击，平均变换鲁棒性为77.8%，扰动大小为16.63%的受害者图像，每对126K查询。对于不要求实现变换鲁棒性的纯数字攻击，Graphic能够发现成功的小补丁攻击，92.2%的受害者-目标对平均只有566个查询。石墨还能够通过针对PatchGuard(最近提出的一种针对基于补丁的攻击的一种防御措施)修改输入图像的小区域来发现成功的攻击。



## **16. Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten**

基于马尔可夫链蒙特卡罗的机器遗忘：遗忘需要遗忘的东西 cs.LG

Proceedings of the 2022 ACM Asia Conference on Computer and  Communications Security (ASIA CCS '22), May 30-June 3, 2022, Nagasaki, Japan

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13585v1)

**Authors**: Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, Bryan Kian Hsiang Low

**Abstracts**: As the use of machine learning (ML) models is becoming increasingly popular in many real-world applications, there are practical challenges that need to be addressed for model maintenance. One such challenge is to 'undo' the effect of a specific subset of dataset used for training a model. This specific subset may contain malicious or adversarial data injected by an attacker, which affects the model performance. Another reason may be the need for a service provider to remove data pertaining to a specific user to respect the user's privacy. In both cases, the problem is to 'unlearn' a specific subset of the training data from a trained model without incurring the costly procedure of retraining the whole model from scratch. Towards this goal, this paper presents a Markov chain Monte Carlo-based machine unlearning (MCU) algorithm. MCU helps to effectively and efficiently unlearn a trained model from subsets of training dataset. Furthermore, we show that with MCU, we are able to explain the effect of a subset of a training dataset on the model prediction. Thus, MCU is useful for examining subsets of data to identify the adversarial data to be removed. Similarly, MCU can be used to erase the lineage of a user's personal data from trained ML models, thus upholding a user's "right to be forgotten". We empirically evaluate the performance of our proposed MCU algorithm on real-world phishing and diabetes datasets. Results show that MCU can achieve a desirable performance by efficiently removing the effect of a subset of training dataset and outperform an existing algorithm that utilizes the remaining dataset.

摘要: 随着机器学习(ML)模型的使用在许多现实世界的应用程序中变得越来越流行，需要解决模型维护方面的实际挑战。一个这样的挑战是“取消”用于训练模型的数据集的特定子集的效果。此特定子集可能包含攻击者注入的恶意或敌意数据，这会影响模型性能。另一个原因可能是服务提供商需要移除属于特定用户的数据以尊重该用户的隐私。在这两种情况下，问题都是从训练的模型中“忘却”训练数据的特定子集，而不会招致从零开始重新训练整个模型的昂贵过程。针对这一目标，提出了一种基于马尔可夫链蒙特卡罗的机器遗忘(MCU)算法。MCU帮助有效且高效地从训练数据集子集中去除训练模型。此外，我们还表明，使用MCU，我们能够解释训练数据集的子集对模型预测的影响。因此，MCU对于检查数据子集以识别要移除的敌意数据是有用的。同样，MCU可以用来从经过训练的ML模型中删除用户个人数据的谱系，从而维护用户的“被遗忘权”。我们在真实的网络钓鱼和糖尿病数据集上对我们提出的MCU算法的性能进行了实证评估。结果表明，MCU通过有效地去除训练数据集子集的影响，取得了理想的性能，并优于现有的利用剩余数据集的算法。



## **17. A Unified Wasserstein Distributional Robustness Framework for Adversarial Training**

一种用于对抗性训练的统一Wasserstein分布健壮性框架 cs.LG

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2202.13437v1)

**Authors**: Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung

**Abstracts**: It is well-known that deep neural networks (DNNs) are susceptible to adversarial attacks, exposing a severe fragility of deep learning systems. As the result, adversarial training (AT) method, by incorporating adversarial examples during training, represents a natural and effective approach to strengthen the robustness of a DNN-based classifier. However, most AT-based methods, notably PGD-AT and TRADES, typically seek a pointwise adversary that generates the worst-case adversarial example by independently perturbing each data sample, as a way to "probe" the vulnerability of the classifier. Arguably, there are unexplored benefits in considering such adversarial effects from an entire distribution. To this end, this paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework. This connection leads to an intuitive relaxation and generalization of existing AT methods and facilitates the development of a new family of distributional robustness AT-based algorithms. Extensive experiments show that our distributional robustness AT algorithms robustify further their standard AT counterparts in various settings.

摘要: 众所周知，深度神经网络(DNNs)容易受到敌意攻击，暴露出深度学习系统的严重脆弱性。因此，对抗性训练(AT)方法通过在训练过程中加入对抗性实例，是增强基于DNN的分类器鲁棒性的一种自然而有效的方法。然而，大多数基于AT的方法，特别是PGD-AT和TRADS，通常通过独立扰动每个数据样本来寻找一个点状对手，该对手通过独立扰动每个数据样本来生成最坏情况下的对手示例，作为“探测”分类器漏洞的一种方式。可以说，从整个发行版考虑这样的对抗性影响有未开发的好处。为此，本文提出了一个将Wasserstein分布稳健性与当前最先进的AT方法相结合的统一框架。我们引入了一个新的Wasserstein成本函数和一系列新的风险函数，证明了标准AT方法是我们框架中相应方法的特例。这种联系导致了现有AT方法的直观松弛和泛化，并促进了一类新的基于分布健壮性的AT算法的开发。大量的实验表明，我们的分布式健壮性AT算法在不同的设置下进一步增强了标准AT算法的健壮性。



## **18. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

Accepted at NeurIPS 2021. The missing square term in Eqn.(13), as  well as many other mistakes of the previous version, have been fixed in the  current version

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2111.07492v5)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法是免费的前期培训。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **19. CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks**

CC-Cert：一种验证神经网络一般鲁棒性的概率方法 cs.LG

In Proceedings of AAAI-22, the Thirty-Sixth AAAI Conference on  Artificial Intelligence

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2109.10696v2)

**Authors**: Mikhail Pautov, Nurislam Tursynbek, Marina Munkhoeva, Nikita Muravev, Aleksandr Petiushko, Ivan Oseledets

**Abstracts**: In safety-critical machine learning applications, it is crucial to defend models against adversarial attacks -- small modifications of the input that change the predictions. Besides rigorously studied $\ell_p$-bounded additive perturbations, recently proposed semantic perturbations (e.g. rotation, translation) raise a serious concern on deploying ML systems in real-world. Therefore, it is important to provide provable guarantees for deep learning models against semantically meaningful input transformations. In this paper, we propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings. We estimate the probability of a model to fail if the attack is sampled from a certain distribution. Our theoretical findings are supported by experimental results on different datasets.

摘要: 在安全关键型机器学习应用程序中，保护模型免受敌意攻击--对输入的微小修改会改变预测--是至关重要的。除了严格研究$\ellp$-有界的加性扰动之外，最近提出的语义扰动(如旋转、平移)也引起了人们对在现实世界中部署ML系统的严重关注。因此，针对语义上有意义的输入转换为深度学习模型提供可证明的保证是很重要的。本文提出了一种新的基于Chernoff-Cramer界的通用概率认证方法，该方法适用于一般攻击环境。我们估计了如果攻击是从特定分布中抽样的，模型失败的概率。在不同数据集上的实验结果支持了我们的理论发现。



## **20. Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Learning**

着火的社交机器人：基于多智能体分层强化学习的社交机器人对抗行为建模 cs.SI

Accepted to The ACM Web Conference 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2110.10655v2)

**Authors**: Thai Le, Long Tran-Thanh, Dongwon Lee

**Abstracts**: Socialbots are software-driven user accounts on social platforms, acting autonomously (mimicking human behavior), with the aims to influence the opinions of other users or spread targeted misinformation for particular goals. As socialbots undermine the ecosystem of social platforms, they are often considered harmful. As such, there have been several computational efforts to auto-detect the socialbots. However, to our best knowledge, the adversarial nature of these socialbots has not yet been studied. This begs a question "can adversaries, controlling socialbots, exploit AI techniques to their advantage?" To this question, we successfully demonstrate that indeed it is possible for adversaries to exploit computational learning mechanism such as reinforcement learning (RL) to maximize the influence of socialbots while avoiding being detected. We first formulate the adversarial socialbot learning as a cooperative game between two functional hierarchical RL agents. While one agent curates a sequence of activities that can avoid the detection, the other agent aims to maximize network influence by selectively connecting with right users. Our proposed policy networks train with a vast amount of synthetic graphs and generalize better than baselines on unseen real-life graphs both in terms of maximizing network influence (up to +18%) and sustainable stealthiness (up to +40% undetectability) under a strong bot detector (with 90% detection accuracy). During inference, the complexity of our approach scales linearly, independent of a network's structure and the virality of news. This makes our approach a practical adversarial attack when deployed in a real-life setting.

摘要: 社交机器人是社交平台上由软件驱动的用户账户，自主行动(模仿人类行为)，目的是影响其他用户的意见或为特定目标传播有针对性的错误信息。由于社交机器人破坏了社交平台的生态系统，它们通常被认为是有害的。因此，已经有几种计算努力来自动检测社交机器人。然而，据我们所知，这些社交机器人的对抗性还没有被研究过。这就引出了一个问题：“控制社交机器人的对手能不能利用人工智能技术对他们有利？”对于这个问题，我们成功地证明了攻击者确实有可能利用计算学习机制，如强化学习(RL)来最大化社交机器人的影响力，同时避免被发现。我们首先将对抗性的社会机器人学习描述为两个功能层次化的RL Agent之间的合作博弈。当一个代理策划一系列可以避免检测的活动时，另一个代理的目标是通过有选择地与正确的用户连接来最大化网络影响力。我们提出的策略网络使用大量的合成图形进行训练，并在强大的BOT检测器(具有90%的检测准确率)下，在最大化网络影响(高达+18%)和可持续隐蔽性(高达+40%不可检测性)方面，对不可见的真实图形进行了更好的概括。在推理过程中，我们方法的复杂性呈线性增长，与网络结构和新闻的病毒度无关。这使得我们的方法在实际环境中部署时成为一种实际的对抗性攻击。



## **21. Natural Attack for Pre-trained Models of Code**

针对预先训练的代码模型的自然攻击 cs.SE

To appear in the Technical Track of ICSE 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2201.08698v2)

**Authors**: Zhou Yang, Jieke Shi, Junda He, David Lo

**Abstracts**: Pre-trained models of code have achieved success in many important software engineering tasks. However, these powerful models are vulnerable to adversarial attacks that slightly perturb model inputs to make a victim model produce wrong outputs. Current works mainly attack models of code with examples that preserve operational program semantics but ignore a fundamental requirement for adversarial example generation: perturbations should be natural to human judges, which we refer to as naturalness requirement.   In this paper, we propose ALERT (nAturaLnEss AwaRe ATtack), a black-box attack that adversarially transforms inputs to make victim models produce wrong outputs. Different from prior works, this paper considers the natural semantic of generated examples at the same time as preserving the operational semantic of original inputs. Our user study demonstrates that human developers consistently consider that adversarial examples generated by ALERT are more natural than those generated by the state-of-the-art work by Zhang et al. that ignores the naturalness requirement. On attacking CodeBERT, our approach can achieve attack success rates of 53.62%, 27.79%, and 35.78% across three downstream tasks: vulnerability prediction, clone detection and code authorship attribution. On GraphCodeBERT, our approach can achieve average success rates of 76.95%, 7.96% and 61.47% on the three tasks. The above outperforms the baseline by 14.07% and 18.56% on the two pre-trained models on average. Finally, we investigated the value of the generated adversarial examples to harden victim models through an adversarial fine-tuning procedure and demonstrated the accuracy of CodeBERT and GraphCodeBERT against ALERT-generated adversarial examples increased by 87.59% and 92.32%, respectively.

摘要: 预先训练的代码模型在许多重要的软件工程任务中取得了成功。然而，这些强大的模型容易受到对抗性攻击，这些攻击略微扰动模型输入，使受害者模型产生错误的输出。目前的工作主要是用保持操作程序语义的示例攻击代码模型，而忽略了生成对抗性示例的一个基本要求：扰动对于人类判断来说应该是自然的，我们称之为自然性要求。在本文中，我们提出了ALERT(自然度感知攻击)，这是一种黑盒攻击，它对输入进行恶意转换，使受害者模型产生错误的输出。与以往的工作不同，本文在保留原始输入操作语义的同时，考虑了生成示例的自然语义。我们的用户研究表明，人类开发人员一致认为，ALERT生成的对抗性示例比由Zhang等人的最新工作生成的示例更自然。这忽略了自然度的要求。在攻击CodeBERT上，我们的方法可以在漏洞预测、克隆检测和代码作者归属三个下游任务上获得53.62%、27.79%和35.78%的攻击成功率。在GraphCodeBERT上，我们的方法在三个任务上的平均成功率分别为76.95%、7.96%和61.47%。在两个预先训练的模型上，上述两个模型的平均性能分别比基线高14.07%和18.56%。最后，我们通过对抗性微调过程考察了生成的对抗性实例对硬化受害者模型的价值，并证明了CodeBERT和GraphCodeBERT对警报生成的对抗性实例的准确率分别提高了87.59%和92.32%。结果表明，CodeBERT和GraphCodeBERT对警报生成的对抗性实例的准确率分别提高了87.59%和92.32%。



## **22. Projective Ranking-based GNN Evasion Attacks**

基于投影排名的GNN逃避攻击 cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12993v1)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstracts**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.

摘要: 图神经网络(GNNs)为与图相关的任务提供了很有前途的学习方法。然而，GNN面临着遭到敌意攻击的风险。强调了当前规避攻击方法的两个主要局限性：(1)当前的GradArgmax忽略了扰动的“长期”益处。在某些情况下，它面临着零梯度和无效的效益估计。(2)在基于强化学习的攻击方法中，当攻击预算发生变化时，学习到的攻击策略可能无法迁移。为此，我们首先定义了扰动空间，并提出了评价框架和投影排序方法。我们的目标是学习一种强大的攻击策略，然后尽可能少地对其进行调整，以便在动态预算设置下生成对抗性样本。在我们的方法中，我们基于互信息，对每个扰动的攻击收益进行排序和评估，以确定有效的攻击策略。通过投射策略，当攻击预算发生变化时，我们的方法极大地降低了学习新攻击策略的成本。在与GradArgmax和RL-S2V的对比评估中，结果表明该方法具有较高的攻击性能和有效的可移植性。该方法的可视化还揭示了敌意样本生成过程中的各种攻击模式。



## **23. Attacks and Faults Injection in Self-Driving Agents on the Carla Simulator -- Experience Report**

自动驾驶智能体在CALA模拟器上的攻击和故障注入--经验报告 cs.AI

submitted version; appeared at: International Conference on Computer  Safety, Reliability, and Security. Springer, Cham, 2021

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12991v1)

**Authors**: Niccolò Piazzesi, Massimo Hong, Andrea Ceccarelli

**Abstracts**: Machine Learning applications are acknowledged at the foundation of autonomous driving, because they are the enabling technology for most driving tasks. However, the inclusion of trained agents in automotive systems exposes the vehicle to novel attacks and faults, that can result in safety threats to the driv-ing tasks. In this paper we report our experimental campaign on the injection of adversarial attacks and software faults in a self-driving agent running in a driving simulator. We show that adversarial attacks and faults injected in the trained agent can lead to erroneous decisions and severely jeopardize safety. The paper shows a feasible and easily-reproducible approach based on open source simula-tor and tools, and the results clearly motivate the need of both protective measures and extensive testing campaigns.

摘要: 机器学习应用在自动驾驶的基础上得到认可，因为它们是大多数驾驶任务的使能技术。然而，在汽车系统中加入训练有素的代理会使车辆暴露在新的攻击和故障下，这可能会对驾驶任务造成安全威胁。在本文中，我们报告了我们在驾驶模拟器中运行的自动驾驶代理中注入对抗性攻击和软件故障的实验活动。我们表明，对抗性攻击和错误注入训练有素的代理可能导致错误的决策，并严重危及安全。本文展示了一种基于开源模拟器和工具的可行且易于重现的方法，其结果清楚地激励了保护措施和广泛的测试活动的需要。



## **24. Does Label Differential Privacy Prevent Label Inference Attacks?**

标签差分隐私能防止标签推理攻击吗？ cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12968v1)

**Authors**: Ruihan Wu, Jin Peng Zhou, Kilian Q. Weinberger, Chuan Guo

**Abstracts**: Label differential privacy (LDP) is a popular framework for training private ML models on datasets with public features and sensitive private labels. Despite its rigorous privacy guarantee, it has been observed that in practice LDP does not preclude label inference attacks (LIAs): Models trained with LDP can be evaluated on the public training features to recover, with high accuracy, the very private labels that it was designed to protect. In this work, we argue that this phenomenon is not paradoxical and that LDP merely limits the advantage of an LIA adversary compared to predicting training labels using the Bayes classifier. At LDP $\epsilon=0$ this advantage is zero, hence the optimal attack is to predict according to the Bayes classifier and is independent of the training labels. Finally, we empirically demonstrate that our result closely captures the behavior of simulated attacks on both synthetic and real world datasets.

摘要: 标签差异隐私(Label Differential Privacy，LDP)是一种流行的框架，用于在具有公共特征和敏感私有标签的数据集上训练私有ML模型。尽管LDP提供了严格的隐私保障，但已经观察到，在实践中，LDP并不排除标签推断攻击(LIA)：使用LDP训练的模型可以在公共训练特征上进行评估，以高精度地恢复其设计来保护的非常私有的标签。在这项工作中，我们认为这种现象并不矛盾，与使用贝叶斯分类器预测训练标签相比，LDP只是限制了LIA对手的优势。当LDP$\εsilon=0$时，这一优势为零，因此最优攻击是根据贝叶斯分类器进行预测，并且与训练标签无关。最后，我们通过实验证明，我们的结果很好地捕捉了模拟攻击在合成数据集和真实世界数据集上的行为。



## **25. Robust and Accurate Authorship Attribution via Program Normalization**

基于程序归一化的稳健准确的作者归属 cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.00772v3)

**Authors**: Yizhen Wang, Mohannad Alhanahnah, Ke Wang, Mihai Christodorescu, Somesh Jha

**Abstracts**: Source code attribution approaches have achieved remarkable accuracy thanks to the rapid advances in deep learning. However, recent studies shed light on their vulnerability to adversarial attacks. In particular, they can be easily deceived by adversaries who attempt to either create a forgery of another author or to mask the original author. To address these emerging issues, we formulate this security challenge into a general threat model, the $\textit{relational adversary}$, that allows an arbitrary number of the semantics-preserving transformations to be applied to an input in any problem space. Our theoretical investigation shows the conditions for robustness and the trade-off between robustness and accuracy in depth. Motivated by these insights, we present a novel learning framework, $\textit{normalize-and-predict}$ ($\textit{N&P}$), that in theory guarantees the robustness of any authorship-attribution approach. We conduct an extensive evaluation of $\textit{N&P}$ in defending two of the latest authorship-attribution approaches against state-of-the-art attack methods. Our evaluation demonstrates that $\textit{N&P}$ improves the accuracy on adversarial inputs by as much as 70% over the vanilla models. More importantly, $\textit{N&P}$ also increases robust accuracy to 45% higher than adversarial training while running over 40 times faster.

摘要: 由于深度学习的快速发展，源代码归属方法已经取得了显着的准确性。然而，最近的研究揭示了它们在对抗性攻击中的脆弱性。特别是，他们很容易被试图伪造另一位作者或掩盖原作者的对手欺骗。为了解决这些新出现的问题，我们将此安全挑战表述为一个通用威胁模型，即$\textit{关系对手}$，该模型允许将任意数量的语义保留转换应用于任何问题空间中的输入。我们的理论研究深入地给出了稳健性的条件以及稳健性和准确性之间的权衡。在这些观点的启发下，我们提出了一个新的学习框架$\textit{Normize-and-Predicate}$($\textit{N&P}$)，该框架在理论上保证了任何作者归因方法的健壮性。我们对$\textit{N&P}$在防御两种最新的作者归属方法方面进行了广泛的评估，以抵御最先进的攻击方法。我们的评估表明，与普通模型相比，$\textit{N&P}$将对手输入的准确率提高了70%。更重要的是，$\textit{N&P}$还将健壮的准确率提高到比对抗性训练高出45%，同时运行速度快40倍以上。



## **26. ARIA: Adversarially Robust Image Attribution for Content Provenance**

ARIA：内容来源的逆向稳健图像归因 cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12860v1)

**Authors**: Maksym Andriushchenko, Xiaoyang Rebecca Li, Geoffrey Oxholm, Thomas Gittings, Tu Bui, Nicolas Flammarion, John Collomosse

**Abstracts**: Image attribution -- matching an image back to a trusted source -- is an emerging tool in the fight against online misinformation. Deep visual fingerprinting models have recently been explored for this purpose. However, they are not robust to tiny input perturbations known as adversarial examples. First we illustrate how to generate valid adversarial images that can easily cause incorrect image attribution. Then we describe an approach to prevent imperceptible adversarial attacks on deep visual fingerprinting models, via robust contrastive learning. The proposed training procedure leverages training on $\ell_\infty$-bounded adversarial examples, it is conceptually simple and incurs only a small computational overhead. The resulting models are substantially more robust, are accurate even on unperturbed images, and perform well even over a database with millions of images. In particular, we achieve 91.6% standard and 85.1% adversarial recall under $\ell_\infty$-bounded perturbations on manipulated images compared to 80.1% and 0.0% from prior work. We also show that robustness generalizes to other types of imperceptible perturbations unseen during training. Finally, we show how to train an adversarially robust image comparator model for detecting editorial changes in matched images.

摘要: 图像归属--将图像与可信来源相匹配--是打击在线错误信息的一种新兴工具。最近已经为此目的探索了深度视觉指纹模型。然而，它们对被称为对抗性示例的微小输入扰动并不健壮。首先，我们说明了如何生成有效的敌意图像，这些图像很容易导致错误的图像属性。然后，我们描述了一种通过稳健的对比学习来防止对深度视觉指纹模型的不可察觉的敌意攻击的方法。所提出的训练过程利用了对$\ELL_\INFTY$-有界的对抗性示例的训练，概念上很简单，并且只产生很小的计算开销。由此产生的模型更加健壮，即使在不受干扰的图像上也是准确的，即使在拥有数百万图像的数据库上也表现得很好。特别地，在$\ell_\ininfty$-有界扰动下，我们获得了91.6%的标准召回率和85.1%的敌意召回率，而以前的工作分别为80.1%和0.0%。我们还表明，鲁棒性推广到其他类型的在训练过程中看不到的不可察觉的扰动。最后，我们展示了如何训练一个对抗性的鲁棒图像比较器模型来检测匹配图像中的编辑变化。



## **27. Short Paper: Device- and Locality-Specific Fingerprinting of Shared NISQ Quantum Computers**

短文：共享NISQ量子计算机的特定于设备和位置的指纹识别 cs.CR

5 pages, 8 figures, HASP 2021 author version

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12731v1)

**Authors**: Allen Mi, Shuwen Deng, Jakub Szefer

**Abstracts**: Fingerprinting of quantum computer devices is a new threat that poses a challenge to shared, cloud-based quantum computers. Fingerprinting can allow adversaries to map quantum computer infrastructures, uniquely identify cloud-based devices which otherwise have no public identifiers, and it can assist other adversarial attacks. This work shows idle tomography-based fingerprinting method based on crosstalk-induced errors in NISQ quantum computers. The device- and locality-specific fingerprinting results show prediction accuracy values of $99.1\%$ and $95.3\%$, respectively.

摘要: 量子计算机设备的指纹识别是一个新的威胁，对共享的、基于云的量子计算机构成了挑战。指纹识别可以让攻击者映射量子计算机基础设施，唯一识别没有公共标识符的基于云的设备，还可以协助其他敌意攻击。这项工作展示了NISQ量子计算机中基于串扰引起的错误的基于空闲层析成像的指纹识别方法。设备特定指纹和位置特定指纹的预测准确值分别为99.1美元和95.3美元。



## **28. Detection as Regression: Certified Object Detection by Median Smoothing**

回归检测：基于中值平滑的认证目标检测 cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.03730v4)

**Authors**: Ping-yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, Tom Goldstein

**Abstracts**: Despite the vulnerability of object detectors to adversarial attacks, very few defenses are known to date. While adversarial training can improve the empirical robustness of image classifiers, a direct extension to object detection is very expensive. This work is motivated by recent progress on certified classification by randomized smoothing. We start by presenting a reduction from object detection to a regression problem. Then, to enable certified regression, where standard mean smoothing fails, we propose median smoothing, which is of independent interest. We obtain the first model-agnostic, training-free, and certified defense for object detection against $\ell_2$-bounded attacks. The code for all experiments in the paper is available at http://github.com/Ping-C/CertifiedObjectDetection .

摘要: 尽管物体探测器对敌方攻击很脆弱，但到目前为止，人们所知的防御措施很少。虽然对抗性训练可以提高图像分类器的经验鲁棒性，但直接扩展到目标检测是非常昂贵的。这项工作是由随机平滑认证分类的最新进展推动的。我们首先介绍从目标检测到回归问题的简化。然后，为了实现认证回归，在标准均值平滑失败的情况下，我们提出了中值平滑，这是独立感兴趣的。我们获得了第一个模型不可知的、无需训练的、经过认证的针对$\ELL_2$有界攻击的目标检测防御。论文中所有实验的代码都可以在http://github.com/Ping-C/CertifiedObjectDetection上找到。



## **29. On the Effectiveness of Dataset Watermarking in Adversarial Settings**

数据集水印在对抗性环境中的有效性研究 cs.CR

7 pages, 2 figures. Will appear in the proceedings of CODASPY-IWSPA  2022

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12506v1)

**Authors**: Buse Gul Atli Tekgul, N. Asokan

**Abstracts**: In a data-driven world, datasets constitute a significant economic value. Dataset owners who spend time and money to collect and curate the data are incentivized to ensure that their datasets are not used in ways that they did not authorize. When such misuse occurs, dataset owners need technical mechanisms for demonstrating their ownership of the dataset in question. Dataset watermarking provides one approach for ownership demonstration which can, in turn, deter unauthorized use. In this paper, we investigate a recently proposed data provenance method, radioactive data, to assess if it can be used to demonstrate ownership of (image) datasets used to train machine learning (ML) models. The original paper reported that radioactive data is effective in white-box settings. We show that while this is true for large datasets with many classes, it is not as effective for datasets where the number of classes is low $(\leq 30)$ or the number of samples per class is low $(\leq 500)$. We also show that, counter-intuitively, the black-box verification technique is effective for all datasets used in this paper, even when white-box verification is not. Given this observation, we show that the confidence in white-box verification can be improved by using watermarked samples directly during the verification process. We also highlight the need to assess the robustness of radioactive data if it were to be used for ownership demonstration since it is an adversarial setting unlike provenance identification.   Compared to dataset watermarking, ML model watermarking has been explored more extensively in recent literature. However, most of the model watermarking techniques can be defeated via model extraction. We show that radioactive data can effectively survive model extraction attacks, which raises the possibility that it can be used for ML model ownership verification robust against model extraction.

摘要: 在一个数据驱动的世界里，数据集构成了重要的经济价值。花费时间和金钱收集和管理数据的数据集所有者受到激励，以确保他们的数据集不会以未经授权的方式使用。当这种误用发生时，数据集所有者需要技术机制来证明他们对相关数据集的所有权。数据集水印为所有权证明提供了一种方法，进而可以阻止未经授权的使用。在这篇文章中，我们调查了最近提出的一种数据来源方法，放射性数据，以评估它是否可以用来证明用于训练机器学习(ML)模型的(图像)数据集的所有权。最初的论文报道说，放射性数据在白盒设置中是有效的。我们表明，虽然这对于具有多个类的大型数据集是正确的，但是对于类数量低$(\leq 30)$或每个类的样本数低$(\leq 500)$的数据集就不那么有效了。我们还表明，与直觉相反，黑盒验证技术对本文使用的所有数据集都是有效的，即使白盒验证不是有效的。基于这一观察结果，我们证明了在验证过程中直接使用带水印的样本可以提高白盒验证的置信度。我们还强调，如果放射性数据要用于所有权证明，则需要评估其稳健性，因为这是一种与来源鉴定不同的对抗性环境。与数据集水印相比，ML模型水印在最近的文献中得到了更广泛的研究。然而，大多数模型水印技术都可以通过模型提取来破解。结果表明，放射性数据能够有效抵御模型提取攻击，提高了其用于ML模型所有权验证对模型提取的鲁棒性的可能性。



## **30. Understanding Adversarial Robustness from Feature Maps of Convolutional Layers**

从卷积层特征图理解敌方鲁棒性 cs.CV

10pages

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12435v1)

**Authors**: Cong Xu, Min Yang

**Abstracts**: The adversarial robustness of a neural network mainly relies on two factors, one is the feature representation capacity of the network, and the other is its resistance ability to perturbations. In this paper, we study the anti-perturbation ability of the network from the feature maps of convolutional layers. Our theoretical analysis discovers that larger convolutional features before average pooling can contribute to better resistance to perturbations, but the conclusion is not true for max pooling. Based on the theoretical findings, we present two feasible ways to improve the robustness of existing neural networks. The proposed approaches are very simple and only require upsampling the inputs or modifying the stride configuration of convolution operators. We test our approaches on several benchmark neural network architectures, including AlexNet, VGG16, RestNet18 and PreActResNet18, and achieve non-trivial improvements on both natural accuracy and robustness under various attacks. Our study brings new insights into the design of robust neural networks. The code is available at \url{https://github.com/MTandHJ/rcm}.

摘要: 神经网络的对抗鲁棒性主要取决于两个因素，一是网络的特征表示能力，二是网络的抗扰动能力。本文从卷积层的特征映射出发，研究了该网络的抗扰动能力。我们的理论分析发现，在平均汇集之前，较大的卷积特征有助于更好地抵抗扰动，但对于最大汇集，这一结论并不成立。在理论研究的基础上，我们提出了两种可行的方法来提高现有神经网络的鲁棒性。所提出的方法非常简单，只需要对输入进行上采样或修改卷积算子的步长配置。我们在AlexNet，VGG16，RestNet18和PreActResNet18等几种基准神经网络体系结构上测试了我们的方法，并在各种攻击下在自然准确性和鲁棒性方面都取得了不小的改善。我们的研究为鲁棒神经网络的设计带来了新的见解。代码位于\url{https://github.com/MTandHJ/rcm}.



## **31. AEVA: Black-box Backdoor Detection Using Adversarial Extreme Value Analysis**

AEVA：基于对抗性极值分析的黑盒后门检测 cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2110.14880v4)

**Authors**: Junfeng Guo, Ang Li, Cong Liu

**Abstracts**: Deep neural networks (DNNs) are proved to be vulnerable against backdoor attacks. A backdoor is often embedded in the target DNNs through injecting a backdoor trigger into training examples, which can cause the target DNNs misclassify an input attached with the backdoor trigger. Existing backdoor detection methods often require the access to the original poisoned training data, the parameters of the target DNNs, or the predictive confidence for each given input, which are impractical in many real-world applications, e.g., on-device deployed DNNs. We address the black-box hard-label backdoor detection problem where the DNN is fully black-box and only its final output label is accessible. We approach this problem from the optimization perspective and show that the objective of backdoor detection is bounded by an adversarial objective. Further theoretical and empirical studies reveal that this adversarial objective leads to a solution with highly skewed distribution; a singularity is often observed in the adversarial map of a backdoor-infected example, which we call the adversarial singularity phenomenon. Based on this observation, we propose the adversarial extreme value analysis(AEVA) to detect backdoors in black-box neural networks. AEVA is based on an extreme value analysis of the adversarial map, computed from the monte-carlo gradient estimation. Evidenced by extensive experiments across multiple popular tasks and backdoor attacks, our approach is shown effective in detecting backdoor attacks under the black-box hard-label scenarios.

摘要: 深度神经网络(DNNs)被证明是易受后门攻击的。通过将后门触发器注入到训练示例中，通常将后门嵌入到目标DNN中，这可能导致目标DNN对与后门触发器附加的输入进行错误分类。现有的后门检测方法通常需要访问原始有毒训练数据、目标DNN的参数或每个给定输入的预测置信度，这在许多真实世界应用中是不切实际的，例如在设备上部署的DNN。我们解决了黑盒硬标签后门检测问题，其中DNN是完全黑盒的，并且只有其最终输出标签是可访问的。我们从优化的角度来研究这个问题，并证明了后门检测的目标是由一个对抗性目标限定的。进一步的理论和实证研究表明，这种对抗性目标导致了一个具有高度偏态分布的解决方案；在一个被后门感染的例子的对抗性地图中经常观察到一个奇点，我们称之为对抗性奇点现象。基于这一观察，我们提出了对抗性极值分析(AEVA)来检测黑盒神经网络中的后门。AEVA是基于对敌方地图的极值分析，通过蒙特卡洛梯度估计计算出来的。通过对多个流行任务和后门攻击的大量实验证明，我们的方法在黑盒硬标签场景下检测后门攻击是有效的。



## **32. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.12232v1)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the accuracy of any MI adversary when a training algorithm provides $\epsilon$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables $\epsilon$-DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。在本文中，当训练算法提供$\epsilon$-dp时，我们对任何MI对手的准确性提供了一个更严格的界。我们的界给出了一种新的隐私放大方案的设计，在训练开始之前，从较大的训练集中对有效的训练集进行亚采样，以大大降低MI准确率的界。因此，我们的方案允许$\epsilon$-DP用户在训练他们的模型以限制任何MI对手的成功时采用更松散的DP保证；这确保了模型的准确性较少地受到隐私保证的影响。最后，我们讨论了我们的MI界对机器遗忘领域的启示。



## **33. Dynamic Defense Against Byzantine Poisoning Attacks in Federated Learning**

联邦学习中对拜占庭中毒攻击的动态防御 cs.LG

10 pages

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2007.15030v2)

**Authors**: Nuria Rodríguez-Barroso, Eugenio Martínez-Cámara, M. Victoria Luzón, Francisco Herrera

**Abstracts**: Federated learning, as a distributed learning that conducts the training on the local devices without accessing to the training data, is vulnerable to Byzatine poisoning adversarial attacks. We argue that the federated learning model has to avoid those kind of adversarial attacks through filtering out the adversarial clients by means of the federated aggregation operator. We propose a dynamic federated aggregation operator that dynamically discards those adversarial clients and allows to prevent the corruption of the global learning model. We assess it as a defense against adversarial attacks deploying a deep learning classification model in a federated learning setting on the Fed-EMNIST Digits, Fashion MNIST and CIFAR-10 image datasets. The results show that the dynamic selection of the clients to aggregate enhances the performance of the global learning model and discards the adversarial and poor (with low quality models) clients.

摘要: 联合学习作为一种在本地设备上进行训练而不需要访问训练数据的分布式学习，容易受到拜占庭中毒的对手攻击。我们认为，联合学习模型必须通过联合聚集算子过滤掉敌意客户端来避免这种对抗性攻击。我们提出了一种动态联合聚集算子，该算子动态地丢弃这些敌意客户端，并允许防止全局学习模型的破坏。我们将其评估为在FED-EMNIST Digits、Fashion MNIST和CIFAR-10图像数据集上的联合学习环境中部署深度学习分类模型的防御对手攻击。结果表明，动态选择要聚合的客户端，提高了全局学习模型的性能，丢弃了对抗性差(低质量模型)的客户端。



## **34. HODA: Hardness-Oriented Detection of Model Extraction Attacks**

Hoda：面向硬性的模型提取攻击检测 cs.LG

15 pages, 12 figures, 7 tables, 2 Alg

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2106.11424v2)

**Authors**: Amir Mahdi Sadeghzadeh, Amir Mohammad Sobhanian, Faezeh Dehghan, Rasool Jalili

**Abstracts**: Model Extraction attacks exploit the target model's prediction API to create a surrogate model in order to steal or reconnoiter the functionality of the target model in the black-box setting. Several recent studies have shown that a data-limited adversary who has no or limited access to the samples from the target model's training data distribution can use synthesis or semantically similar samples to conduct model extraction attacks. In this paper, we define the hardness degree of a sample using the concept of learning difficulty. The hardness degree of a sample depends on the epoch number that the predicted label of that sample converges. We investigate the hardness degree of samples and demonstrate that the hardness degree histogram of a data-limited adversary's sample sequences is distinguishable from the hardness degree histogram of benign users' samples sequences. We propose Hardness-Oriented Detection Approach (HODA) to detect the sample sequences of model extraction attacks. The results demonstrate that HODA can detect the sample sequences of model extraction attacks with a high success rate by only monitoring 100 samples of them, and it outperforms all previous model extraction detection methods.

摘要: 模型提取攻击利用目标模型的预测API来创建代理模型，以便窃取或侦察黑盒设置中的目标模型的功能。最近的一些研究表明，数据受限的对手如果无法或有限地访问目标模型的训练数据分布中的样本，就可以使用合成或语义相似的样本来进行模型提取攻击。本文利用学习难度的概念定义了样本的硬度。样本的硬度取决于该样本的预测标号收敛的历元数。研究了样本的硬度，证明了数据受限对手的样本序列的硬度直方图与良性用户的样本序列的硬度直方图是可区分的。提出了面向硬度的检测方法(HodA)来检测模型提取攻击的样本序列。实验结果表明，Hoda算法仅需监测100个样本，即可检测出模型提取攻击的样本序列，检测成功率较高，且性能优于以往的所有模型提取检测方法。



## **35. Feature Importance-aware Transferable Adversarial Attacks**

特征重要性感知的可转移对抗性攻击 cs.CV

Accepted to ICCV 2021

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2107.14185v3)

**Authors**: Zhibo Wang, Hengchang Guo, Zhifei Zhang, Wenxin Liu, Zhan Qin, Kui Ren

**Abstracts**: Transferability of adversarial examples is of central importance for attacking an unknown model, which facilitates adversarial attacks in more practical scenarios, e.g., black-box attacks. Existing transferable attacks tend to craft adversarial examples by indiscriminately distorting features to degrade prediction accuracy in a source model without aware of intrinsic features of objects in the images. We argue that such brute-force degradation would introduce model-specific local optimum into adversarial examples, thus limiting the transferability. By contrast, we propose the Feature Importance-aware Attack (FIA), which disrupts important object-aware features that dominate model decisions consistently. More specifically, we obtain feature importance by introducing the aggregate gradient, which averages the gradients with respect to feature maps of the source model, computed on a batch of random transforms of the original clean image. The gradients will be highly correlated to objects of interest, and such correlation presents invariance across different models. Besides, the random transforms will preserve intrinsic features of objects and suppress model-specific information. Finally, the feature importance guides to search for adversarial examples towards disrupting critical features, achieving stronger transferability. Extensive experimental evaluation demonstrates the effectiveness and superior performance of the proposed FIA, i.e., improving the success rate by 9.5% against normally trained models and 12.8% against defense models as compared to the state-of-the-art transferable attacks. Code is available at: https://github.com/hcguoO0/FIA

摘要: 对抗性示例的可转移性对于攻击未知模型至关重要，这有助于在更实际的场景中进行对抗性攻击，例如黑盒攻击。现有的可转移攻击倾向于通过不分青红皂白地扭曲特征来制作敌意示例，以降低源模型中的预测精度，而不知道图像中对象的固有特征。我们认为，这种暴力降级会将特定于模型的局部最优引入到对抗性例子中，从而限制了可移植性。相反，我们提出了特征重要性感知攻击(FIA)，它破坏了一致主导模型决策的重要对象感知特征。更具体地说，我们通过引入聚合梯度来获得特征重要性，聚合梯度是在原始清洁图像的一批随机变换上计算的关于源模型的特征映射的平均梯度。梯度将与感兴趣的对象高度相关，并且这种相关性在不同模型之间呈现不变性。此外，随机变换将保留对象的固有特征并抑制特定于模型的信息。最后，特征重要度引导搜索对抗性实例，以破坏关键特征，实现更强的可移植性。广泛的实验评估表明了该算法的有效性和优越的性能，即与最先进的可转移攻击相比，相对于正常训练的模型，成功率提高了9.5%，对防御模型的成功率提高了12.8%。代码可在以下网址获得：https://github.com/hcguoO0/FIA



## **36. Robust Probabilistic Time Series Forecasting**

稳健概率时间序列预测 cs.LG

AISTATS 2022 camera ready version

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.11910v1)

**Authors**: TaeHo Yoon, Youngsuk Park, Ernest K. Ryu, Yuyang Wang

**Abstracts**: Probabilistic time series forecasting has played critical role in decision-making processes due to its capability to quantify uncertainties. Deep forecasting models, however, could be prone to input perturbations, and the notion of such perturbations, together with that of robustness, has not even been completely established in the regime of probabilistic forecasting. In this work, we propose a framework for robust probabilistic time series forecasting. First, we generalize the concept of adversarial input perturbations, based on which we formulate the concept of robustness in terms of bounded Wasserstein deviation. Then we extend the randomized smoothing technique to attain robust probabilistic forecasters with theoretical robustness certificates against certain classes of adversarial perturbations. Lastly, extensive experiments demonstrate that our methods are empirically effective in enhancing the forecast quality under additive adversarial attacks and forecast consistency under supplement of noisy observations.

摘要: 概率时间序列预测因其能够量化不确定性而在决策过程中发挥着至关重要的作用。然而，深度预测模型可能容易受到输入扰动，并且这种扰动的概念以及稳健性的概念在概率预测体系中甚至还没有完全建立起来。在这项工作中，我们提出了一个稳健概率时间序列预测的框架。首先，我们推广了对抗性输入扰动的概念，并在此基础上提出了基于有界Wasserstein偏差的鲁棒性概念。然后，我们对随机平滑技术进行扩展，以获得鲁棒的概率预报器，该预报器对某些类型的对抗性扰动具有理论上的稳健性证书。最后，大量的实验表明，我们的方法在提高加性敌方攻击下的预测质量和补充噪声观测下的预测一致性方面是经验性有效的。



## **37. Improving Robustness of Convolutional Neural Networks Using Element-Wise Activation Scaling**

用单元激活尺度提高卷积神经网络的鲁棒性 cs.CV

**SubmitDate**: 2022-02-24    [paper-pdf](http://arxiv.org/pdf/2202.11898v1)

**Authors**: Zhi-Yuan Zhang, Di Liu

**Abstracts**: Recent works reveal that re-calibrating the intermediate activation of adversarial examples can improve the adversarial robustness of a CNN model. The state of the arts [Baiet al., 2021] and [Yanet al., 2021] explores this feature at the channel level, i.e. the activation of a channel is uniformly scaled by a factor. In this paper, we investigate the intermediate activation manipulation at a more fine-grained level. Instead of uniformly scaling the activation, we individually adjust each element within an activation and thus propose Element-Wise Activation Scaling, dubbed EWAS, to improve CNNs' adversarial robustness. Experimental results on ResNet-18 and WideResNet with CIFAR10 and SVHN show that EWAS significantly improves the robustness accuracy. Especially for ResNet18 on CIFAR10, EWAS increases the adversarial accuracy by 37.65% to 82.35% against C&W attack. EWAS is simple yet very effective in terms of improving robustness. The codes are anonymously available at https://anonymous.4open.science/r/EWAS-DD64.

摘要: 最近的工作表明，重新校准对抗性示例的中间激活可以提高CNN模型的对抗性健壮性。当前技术水平[Baiet al.，2021]和[Yanet al.，2021]探讨了通道级别的这一特征，即通道的激活由一个因子统一缩放。在本文中，我们在更细粒度的水平上研究了中间活化操作。我们不是统一地调整激活，而是单独调整激活中的每个元素，从而提出了基于元素的激活缩放，称为EWAS，以提高CNNs的对抗健壮性。用CIFAR10和SVHN在ResNet-18和WideResNet上的实验结果表明，EWAS显著提高了鲁棒性准确率。特别是对于CIFAR10上的ResNet18，EWAS对抗C&W攻击的准确率提高了37.65%~82.35%。EWAS简单但在提高健壮性方面非常有效。这些代码可以在https://anonymous.4open.science/r/EWAS-DD64.上匿名获得



## **38. FastZIP: Faster and More Secure Zero-Interaction Pairing**

FastZip：更快、更安全的零交互配对 cs.CR

ACM MobiSys '21; Fixed ambiguity in flow diagram (Figure 2). Code and  data are available at: https://github.com/seemoo-lab/fastzip

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2106.04907v3)

**Authors**: Mikhail Fomichev, Julia Hesse, Lars Almon, Timm Lippert, Jun Han, Matthias Hollick

**Abstracts**: With the advent of the Internet of Things (IoT), establishing a secure channel between smart devices becomes crucial. Recent research proposes zero-interaction pairing (ZIP), which enables pairing without user assistance by utilizing devices' physical context (e.g., ambient audio) to obtain a shared secret key. The state-of-the-art ZIP schemes suffer from three limitations: (1) prolonged pairing time (i.e., minutes or hours), (2) vulnerability to brute-force offline attacks on a shared key, and (3) susceptibility to attacks caused by predictable context (e.g., replay attack) because they rely on limited entropy of physical context to protect a shared key. We address these limitations, proposing FastZIP, a novel ZIP scheme that significantly reduces pairing time while preventing offline and predictable context attacks. In particular, we adapt a recently introduced Fuzzy Password-Authenticated Key Exchange (fPAKE) protocol and utilize sensor fusion, maximizing their advantages. We instantiate FastZIP for intra-car device pairing to demonstrate its feasibility and show how the design of FastZIP can be adapted to other ZIP use cases. We implement FastZIP and evaluate it by driving four cars for a total of 800 km. We achieve up to three times shorter pairing time compared to the state-of-the-art ZIP schemes while assuring robust security with adversarial error rates below 0.5%.

摘要: 随着物联网(IoT)的到来，在智能设备之间建立安全通道变得至关重要。最近的研究提出了零交互配对(ZIP)，它通过利用设备的物理上下文(例如，环境音频)来获得共享密钥，从而在没有用户帮助的情况下实现配对。现有的ZIP方案有三个局限性：(1)配对时间延长(即几分钟或几小时)，(2)容易受到对共享密钥的暴力离线攻击，以及(3)容易受到由可预测上下文引起的攻击(例如，重放攻击)，因为它们依赖于有限的物理上下文熵来保护共享密钥。我们解决了这些限制，提出了FastZip，这是一种新颖的ZIP方案，可以显著减少配对时间，同时防止离线和可预测的上下文攻击。特别是，我们采用了最近推出的模糊口令认证密钥交换(FPAKE)协议，并利用传感器融合，最大限度地发挥了它们的优势。我们将FastZip实例化用于车内设备配对，以演示其可行性，并展示FastZip的设计如何适用于其他ZIP用例。我们实施了FastZip，并通过驾驶4辆车总共行驶800公里对其进行评估。与最先进的ZIP方案相比，我们实现了高达3倍的配对时间，同时确保了健壮的安全性，对手错误率低于0.5%。



## **39. Distributed and Mobile Message Level Relaying/Replaying of GNSS Signals**

GNSS信号的分布式和移动消息级别中继/重放 cs.CR

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2202.11341v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: With the introduction of Navigation Message Authentication (NMA), future Global Navigation Satellite Systems (GNSSs) prevent spoofing by simulation, i.e., the generation of forged satellite signals based on public information. However, authentication does not prevent record-and-replay attacks, commonly termed as meaconing. These attacks are less powerful in terms of adversarial control over the victim receiver location and time, but by acting at the signal level, they are not thwarted by NMA. This makes replaying/relaying attacks a significant threat for GNSS. While there are numerous investigations on meaconing, the majority does not rely on actual implementation and experimental evaluation in real-world settings. In this work, we contribute to the improvement of the experimental understanding of meaconing attacks. We design and implement a system capable of real-time, distributed, and mobile meaconing, built with off-the-shelf hardware. We extend from basic distributed attacks, with signals from different locations relayed over the Internet and replayed within range of the victim receiver(s): this has high bandwidth requirements and thus depends on the quality of service of the available network to work. To overcome this limitation, we propose to replay on message level, including the authentication part of the payload. The resultant reduced bandwidth enables the attacker to operate in mobile scenarios, as well as to replay signals from multiple GNSS constellations and/or bands simultaneously. Additionally, the attacker can delay individually selected satellite signals to potentially influence the victim position and time solution in a more fine-grained manner. Our versatile test-bench, enabling different types of replaying/relaying attacks, facilitates testing realistic scenarios towards new and improved replaying/relaying-focused countermeasures in GNSS receivers.

摘要: 随着导航信息认证(NMA)的引入，未来的全球导航卫星系统(GNSS)将通过仿真来防止欺骗，即基于公开信息产生伪造的卫星信号。但是，身份验证并不能防止记录和重放攻击，通常称为手段攻击。这些攻击在对受害者接收器位置和时间的敌意控制方面不那么强大，但通过在信号级别采取行动，它们不会被NMA挫败。这使得重放/中继攻击成为GNSS的重大威胁。虽然关于测量的研究很多，但大多数研究并不依赖于在现实世界中的实际实施和实验评估。在这项工作中，我们为提高对手段攻击的实验理解做出了贡献。我们设计并实现了一个具有实时、分布式和移动测量功能的系统，该系统采用现成的硬件构建。我们从基本的分布式攻击扩展，通过Internet中继来自不同位置的信号，并在受害者接收器的范围内重放：这具有很高的带宽要求，因此取决于可用网络的服务质量才能工作。为了克服这一限制，我们建议在消息级别重放，包括有效负载的身份验证部分。由此减少的带宽使攻击者能够在移动场景中操作，以及同时重放来自多个GNSS星座和/或频带的信号。此外，攻击者可以延迟单独选择的卫星信号，以更细粒度的方式潜在地影响受害者的位置和时间解决方案。我们的多功能测试台支持不同类型的重放/中继攻击，便于测试GNSS接收器中新的和改进的重放/中继重点对策的现实场景。



## **40. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

LPF-Defense：基于频率分析的三维对抗性防御 cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-02-23    [paper-pdf](http://arxiv.org/pdf/2202.11287v1)

**Authors**: Hanieh Naderi, Arian Etemadi, Kimia Noorbakhsh, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.

摘要: 虽然三维点云分类近年来在不同的应用场景中得到了广泛的应用，但它仍然非常容易受到敌意攻击。这增加了3D模型在面对敌方攻击时稳健训练的重要性。基于对现有对抗性攻击性能的分析，在输入数据的中高频成分中发现了更多的对抗性扰动。因此，通过抑制训练阶段的高频内容，提高了模型对敌意样本的鲁棒性。实验表明，该防御方法降低了对PointNet、PointNet++、DGCNN模型的6次攻击成功率。特别是，与现有方法相比，Drop100攻击的分类准确率平均提高了3.8%，Drop200攻击的分类准确率平均提高了4.26%。与其他可用的方法相比，该方法还提高了原始数据集上的模型精度。



## **41. Sound Adversarial Audio-Visual Navigation**

声音对抗性视听导航 cs.SD

This work aims to do an adversarial sound intervention for robust  audio-visual navigation

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10910v1)

**Authors**: Yinfeng Yu, Wenbing Huang, Fuchun Sun, Changan Chen, Yikai Wang, Xiaohong Liu

**Abstracts**: Audio-visual navigation task requires an agent to find a sound source in a realistic, unmapped 3D environment by utilizing egocentric audio-visual observations. Existing audio-visual navigation works assume a clean environment that solely contains the target sound, which, however, would not be suitable in most real-world applications due to the unexpected sound noise or intentional interference. In this work, we design an acoustically complex environment in which, besides the target sound, there exists a sound attacker playing a zero-sum game with the agent. More specifically, the attacker can move and change the volume and category of the sound to make the agent suffer from finding the sounding object while the agent tries to dodge the attack and navigate to the goal under the intervention. Under certain constraints to the attacker, we can improve the robustness of the agent towards unexpected sound attacks in audio-visual navigation. For better convergence, we develop a joint training mechanism by employing the property of a centralized critic with decentralized actors. Experiments on two real-world 3D scan datasets, Replica, and Matterport3D, verify the effectiveness and the robustness of the agent trained under our designed environment when transferred to the clean environment or the one containing sound attackers with random policy. Project: \url{https://yyf17.github.io/SAAVN}.

摘要: 视听导航任务要求代理通过利用以自我为中心的视听观察，在真实的、未映射的3D环境中找到声源。现有的视听导航作品假设一个干净的环境，只包含目标声音，然而，由于意外的声音噪声或故意的干扰，这在大多数现实世界的应用中是不合适的。在本工作中，我们设计了一个复杂的声学环境，在这个环境中，除了目标声音外，还有一个声音攻击者与Agent进行零和游戏。更具体地说，攻击者可以移动和改变声音的音量和类别，使代理在试图躲避攻击并导航到干预下的目标时，难以找到探测对象。在对攻击者有一定约束的情况下，可以提高Agent对视听导航中意外声音攻击的鲁棒性。为了更好地收敛，我们利用集中批评家和分散参与者的性质开发了一种联合训练机制。在两个真实的三维扫描数据集Replica和Matterport3D上进行了实验，验证了在我们设计的环境下训练的代理在迁移到干净的环境和包含随机策略的声音攻击者的环境下的有效性和健壮性。项目：\url{https://yyf17.github.io/SAAVN}.



## **42. DEMO: Relay/Replay Attacks on GNSS signals**

演示：对GNSS信号的中继/重放攻击 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10897v1)

**Authors**: M. Lenhart, M. Spanghero, P. Papadimitratos

**Abstracts**: Global Navigation Satellite Systems (GNSS) are ubiquitously relied upon for positioning and timing. Detection and prevention of attacks against GNSS have been researched over the last decades, but many of these attacks and countermeasures were evaluated based on simulation. This work contributes to the experimental investigation of GNSS vulnerabilities, implementing a relay/replay attack with off-the-shelf hardware. Operating at the signal level, this attack type is not hindered by cryptographically protected transmissions, such as Galileo's Open Signals Navigation Message Authentication (OS-NMA). The attack we investigate involves two colluding adversaries, relaying signals over large distances, to effectively spoof a GNSS receiver. We demonstrate the attack using off-the-shelf hardware, we investigate the requirements for such successful colluding attacks, and how they can be enhanced, e.g., allowing for finer adversarial control over the victim receiver.

摘要: 全球导航卫星系统(GNSS)的定位和授时无处不在地依赖于全球导航卫星系统(GNSS)。近几十年来，对GNSS攻击的检测和预防一直在研究之中，但许多攻击和对策都是基于仿真进行评估的。这项工作有助于对GNSS漏洞的实验研究，使用现成的硬件实现中继/重放攻击。在信号级运行，这种攻击类型不会受到密码保护传输的阻碍，例如伽利略的开放信号导航消息验证(OS-NMA)。我们调查的攻击涉及两个串通的对手，他们远距离中继信号，以有效地欺骗GNSS接收器。我们使用现成的硬件演示了攻击，我们调查了这种成功的合谋攻击的要求，以及如何增强这些要求，例如，允许对受害者接收器进行更精细的敌意控制。



## **43. Protecting GNSS-based Services using Time Offset Validation**

使用时间偏移验证保护基于GNSS的服务 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10891v1)

**Authors**: K. Zhang, M. Spanghero, P. Papadimitratos

**Abstracts**: Global navigation satellite systems (GNSS) provide pervasive accurate positioning and timing services for a large gamut of applications, from Time based One-Time Passwords (TOPT), to power grid and cellular systems. However, there can be security concerns for the applications due to the vulnerability of GNSS. It is important to observe that GNSS receivers are components of platforms, in principle having rich connectivity to different network infrastructures. Of particular interest is the access to a variety of timing sources, as those can be used to validate GNSS-provided location and time. Therefore, we consider off-the-shelf platforms and how to detect if the GNSS receiver is attacked or not, by cross-checking the GNSS time and time from other available sources. First, we survey different technologies to analyze their availability, accuracy, and trustworthiness for time synchronization. Then, we propose a validation approach for absolute and relative time. Moreover, we design a framework and experimental setup for the evaluation of the results. Attacks can be detected based on WiFi supplied time when the adversary shifts the GNSS provided time, more than 23.942us; with Network Time Protocol (NTP) supplied time when the adversary-induced shift is more than 2.046ms. Consequently, the proposal significantly limits the capability of an adversary to manipulate the victim GNSS receiver.

摘要: 全球导航卫星系统(GNSS)为从基于时间的一次性密码(TOPT)到电网和蜂窝系统的大量应用提供无处不在的精确定位和授时服务。然而，由于GNSS的脆弱性，应用程序可能存在安全问题。重要的是要注意到，全球导航卫星系统接收器是平台的组成部分，原则上与不同的网络基础设施有丰富的连接。特别令人感兴趣的是对各种计时源的访问，因为这些时间源可用于验证全球导航卫星系统提供的位置和时间。因此，我们考虑了现成的平台，以及如何通过交叉检查GNSS时间和其他可用来源的时间来检测GNSS接收机是否受到攻击。首先，我们综述了不同的时间同步技术，以分析它们在时间同步方面的可用性、准确性和可信性。然后，我们提出了一种绝对时间和相对时间的验证方法。此外，我们还设计了评价结果的框架和实验装置。当对手移动GNSS提供的时间大于23.942us时，可以基于WiFi提供的时间检测攻击；当对手引起的移动超过2.046ms时，基于网络时间协议(NTP)提供的时间可以检测到攻击。因此，该提案极大地限制了对手操纵受害者GNSS接收器的能力。



## **44. Adversarial Defense by Latent Style Transformations**

潜在风格转换的对抗性防御 cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2006.09701v2)

**Authors**: Shuo Wang, Surya Nepal, Alsharif Abuadbba, Carsten Rudolph, Marthie Grobler

**Abstracts**: Machine learning models have demonstrated vulnerability to adversarial attacks, more specifically misclassification of adversarial examples.   In this paper, we investigate an attack-agnostic defense against adversarial attacks on high-resolution images by detecting suspicious inputs.   The intuition behind our approach is that the essential characteristics of a normal image are generally consistent with non-essential style transformations, e.g., slightly changing the facial expression of human portraits.   In contrast, adversarial examples are generally sensitive to such transformations.   In our approach to detect adversarial instances, we propose an in\underline{V}ertible \underline{A}utoencoder based on the \underline{S}tyleGAN2 generator via \underline{A}dversarial training (VASA) to inverse images to disentangled latent codes that reveal hierarchical styles.   We then build a set of edited copies with non-essential style transformations by performing latent shifting and reconstruction, based on the correspondences between latent codes and style transformations.   The classification-based consistency of these edited copies is used to distinguish adversarial instances.

摘要: 机器学习模型已经显示出对敌意攻击的脆弱性，更具体地说，是对对抗性例子的错误分类。在这篇文章中，我们研究了一种通过检测可疑输入来抵抗高分辨率图像上的敌意攻击的攻击不可知性防御方法。我们的方法背后的直觉是，正常图像的基本特征通常与非必要的样式转换一致，例如，稍微改变人物肖像的面部表情。相反，对抗性的例子通常对这样的转换很敏感。在检测敌意实例的方法中，我们提出了一种基于下划线{S}tyleGAN2生成器的下划线{V}易错下划线{A}utoender，它通过下划线{A}变异训练(VASA)将图像逆变成显示分层样式的解缠潜代码。然后根据潜在代码和样式转换之间的对应关系，通过潜移位和重构，构建一组具有非本质样式转换的编辑副本。这些编辑副本的基于分类的一致性被用来区分对抗性实例。



## **45. Surrogate Representation Learning with Isometric Mapping for Gray-box Graph Adversarial Attacks**

基于等距映射的灰盒图对抗攻击代理表示学习 cs.AI

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2110.10482v3)

**Authors**: Zihan Liu, Yun Luo, Zelin Zang, Stan Z. Li

**Abstracts**: Gray-box graph attacks aim at disrupting the performance of the victim model by using inconspicuous attacks with limited knowledge of the victim model. The parameters of the victim model and the labels of the test nodes are invisible to the attacker. To obtain the gradient on the node attributes or graph structure, the attacker constructs an imaginary surrogate model trained under supervision. However, there is a lack of discussion on the training of surrogate models and the robustness of provided gradient information. The general node classification model loses the topology of the nodes on the graph, which is, in fact, an exploitable prior for the attacker. This paper investigates the effect of representation learning of surrogate models on the transferability of gray-box graph adversarial attacks. To reserve the topology in the surrogate embedding, we propose Surrogate Representation Learning with Isometric Mapping (SRLIM). By using Isometric mapping method, our proposed SRLIM can constrain the topological structure of nodes from the input layer to the embedding space, that is, to maintain the similarity of nodes in the propagation process. Experiments prove the effectiveness of our approach through the improvement in the performance of the adversarial attacks generated by the gradient-based attacker in untargeted poisoning gray-box setups.

摘要: 灰盒图攻击的目的是在有限的受害者模型知识下，利用不明显的攻击破坏受害者模型的性能。受害者模型的参数和测试节点的标签对攻击者是不可见的。为了获得节点属性或图结构上的梯度，攻击者构建了一个在监督下训练的虚拟代理模型。然而，对于代理模型的训练和提供的梯度信息的稳健性，目前还缺乏讨论。一般节点分类模型会丢失图上节点的拓扑，这实际上是攻击者可以利用的先验信息。研究了代理模型的表示学习对灰盒图对抗攻击可转移性的影响。为了保留代理嵌入中的拓扑结构，我们提出了基于等距映射的代理表示学习算法(SRLIM)。通过使用等距映射方法，我们提出的SRLIM可以将节点的拓扑结构从输入层约束到嵌入空间，即在传播过程中保持节点的相似性。实验证明，在无目标中毒灰盒设置下，基于梯度的攻击者生成的对抗性攻击的性能得到了提高，证明了该方法的有效性。



## **46. Universal adversarial perturbation for remote sensing images**

遥感图像的普遍对抗性摄动 cs.CV

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10693v1)

**Authors**: Zhaoxia Yin, Qingyu Wang, Jin Tang, Bin Luo

**Abstracts**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been greatly improved compared with traditional technology. However, even state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). To verify that UAP makes the RSI classification model error classification, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism. Firstly, the former can learn the distribution of perturbations better, then the latter is used to find the main regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbations making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the RSI misclassify, and the attack success rate (ASR) of our proposed method on the RSI data set is as high as 97.35%.

摘要: 近年来，随着深度学习技术在遥感图像领域的应用，遥感图像的分类精度与传统技术相比有了很大的提高。然而，即使是最先进的目标识别卷积神经网络也会被普遍的对抗性摄动(UAP)所欺骗。为了验证UAP对RSI分类模型进行错误分类，提出了一种编解码器网络与注意力机制相结合的新方法。前者能更好地学习扰动的分布，后者用于寻找RSI分类模型关注的主要区域。最后，生成的区域被用来微调扰动，使得模型在较少扰动的情况下被误分类。实验结果表明，UAP可以使RSI发生误分类，本文提出的方法在RSI数据集上的攻击成功率高达97.35%。



## **47. Seeing is Living? Rethinking the Security of Facial Liveness Verification in the Deepfake Era**

看就是活？深伪时代下人脸活体验证安全性的再思考 cs.CR

Accepted as a full paper at USENIX Security '22

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10673v1)

**Authors**: Changjiang Li, Li Wang, Shouling Ji, Xuhong Zhang, Zhaohan Xi, Shanqing Guo, Ting Wang

**Abstracts**: Facial Liveness Verification (FLV) is widely used for identity authentication in many security-sensitive domains and offered as Platform-as-a-Service (PaaS) by leading cloud vendors. Yet, with the rapid advances in synthetic media techniques (e.g., deepfake), the security of FLV is facing unprecedented challenges, about which little is known thus far.   To bridge this gap, in this paper, we conduct the first systematic study on the security of FLV in real-world settings. Specifically, we present LiveBugger, a new deepfake-powered attack framework that enables customizable, automated security evaluation of FLV. Leveraging LiveBugger, we perform a comprehensive empirical assessment of representative FLV platforms, leading to a set of interesting findings. For instance, most FLV APIs do not use anti-deepfake detection; even for those with such defenses, their effectiveness is concerning (e.g., it may detect high-quality synthesized videos but fail to detect low-quality ones). We then conduct an in-depth analysis of the factors impacting the attack performance of LiveBugger: a) the bias (e.g., gender or race) in FLV can be exploited to select victims; b) adversarial training makes deepfake more effective to bypass FLV; c) the input quality has a varying influence on different deepfake techniques to bypass FLV. Based on these findings, we propose a customized, two-stage approach that can boost the attack success rate by up to 70%. Further, we run proof-of-concept attacks on several representative applications of FLV (i.e., the clients of FLV APIs) to illustrate the practical implications: due to the vulnerability of the APIs, many downstream applications are vulnerable to deepfake. Finally, we discuss potential countermeasures to improve the security of FLV. Our findings have been confirmed by the corresponding vendors.

摘要: 面部活体验证(FLV)广泛用于许多安全敏感领域的身份验证，并由领先的云供应商以平台即服务(PaaS)的形式提供。然而，随着合成媒体技术(如深度假冒)的快速发展，FLV的安全正面临着前所未有的挑战，目前对此知之甚少。为了弥补这一差距，本文首次对FLV在现实环境下的安全性进行了系统的研究。具体地说，我们介绍了LiveBugger，这是一个新的深度假冒支持的攻击框架，可以对FLV进行可定制的、自动化的安全评估。利用LiveBugger，我们对有代表性的FLV平台进行了全面的实证评估，得出了一系列有趣的发现。例如，大多数FLV API不使用防深伪检测，即使是有这种防御的API，其有效性也是令人担忧的(例如，它可能会检测到高质量的合成视频，但无法检测到低质量的合成视频)。然后，我们深入分析了影响LiveBugger攻击性能的因素：a)FLV中的偏见(如性别或种族)可以被用来选择受害者；b)对抗性训练使深伪更有效地绕过FLV；c)输入质量对不同的绕过FLV的深伪技术有不同的影响。基于这些发现，我们提出了一种定制的两阶段方法，可以将攻击成功率提高高达70%。此外，我们对FLV的几个有代表性的应用程序(即FLV API的客户端)进行了概念验证攻击，以说明其实际意义：由于API的脆弱性，许多下游应用程序都容易受到深度假冒的攻击。最后，我们讨论了提高FLV安全性的潜在对策。我们的发现已经得到了相应供应商的证实。



## **48. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于普遍对抗性扰动的深度神经网络全局指纹识别 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.08602v2)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 本文提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是DNN模型的决策边界的轮廓可以由它的\textit(通用对抗性扰动(UAP))来唯一地刻画。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并以指纹为输入，通过对比学习训练编码器，输出相似度得分。大量的研究表明，我们的框架可以在可疑模型的$2 0$指纹范围内以>99.99$的置信度检测到模型IP泄露。它具有良好的跨不同模型体系结构的通用性，并且对窃取模型的后期修改具有健壮性。



## **49. Robust Stochastic Linear Contextual Bandits Under Adversarial Attacks**

对抗性攻击下的鲁棒随机线性上下文带 stat.ML

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2106.02978v2)

**Authors**: Qin Ding, Cho-Jui Hsieh, James Sharpnack

**Abstracts**: Stochastic linear contextual bandit algorithms have substantial applications in practice, such as recommender systems, online advertising, clinical trials, etc. Recent works show that optimal bandit algorithms are vulnerable to adversarial attacks and can fail completely in the presence of attacks. Existing robust bandit algorithms only work for the non-contextual setting under the attack of rewards and cannot improve the robustness in the general and popular contextual bandit environment. In addition, none of the existing methods can defend against attacked context. In this work, we provide the first robust bandit algorithm for stochastic linear contextual bandit setting under a fully adaptive and omniscient attack with sub-linear regret. Our algorithm not only works under the attack of rewards, but also under attacked context. Moreover, it does not need any information about the attack budget or the particular form of the attack. We provide theoretical guarantees for our proposed algorithm and show by experiments that our proposed algorithm improves the robustness against various kinds of popular attacks.

摘要: 随机线性上下文盗贼算法在推荐系统、在线广告、临床试验等领域有着广泛的应用。最近的研究表明，最优盗贼算法很容易受到敌意攻击，并且在存在攻击的情况下可能完全失效。现有的鲁棒盗版算法只适用于奖励攻击下的非上下文环境，不能提高在一般流行的上下文盗版环境下的鲁棒性。此外，现有的方法都不能抵御上下文攻击。在这项工作中，我们针对随机线性上下文盗贼设置，在具有子线性遗憾的完全自适应和全知攻击下，提供了第一个鲁棒盗贼算法。我们的算法不仅可以在奖励攻击下工作，而且可以在受攻击的环境下工作。此外，它不需要关于攻击预算或特定攻击形式的任何信息。我们为我们提出的算法提供了理论上的保证，实验表明，我们提出的算法提高了对各种流行攻击的鲁棒性。



## **50. Behaviour-Diverse Automatic Penetration Testing: A Curiosity-Driven Multi-Objective Deep Reinforcement Learning Approach**

行为多样化的自动渗透测试：一种好奇心驱动的多目标深度强化学习方法 cs.LG

6 pages,4 Figures

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.10630v1)

**Authors**: Yizhou Yang, Xin Liu

**Abstracts**: Penetration Testing plays a critical role in evaluating the security of a target network by emulating real active adversaries. Deep Reinforcement Learning (RL) is seen as a promising solution to automating the process of penetration tests by reducing human effort and improving reliability. Existing RL solutions focus on finding a specific attack path to impact the target hosts. However, in reality, a diverse range of attack variations are needed to provide comprehensive assessments of the target network's security level. Hence, the attack agents must consider multiple objectives when penetrating the network. Nevertheless, this challenge is not adequately addressed in the existing literature. To this end, we formulate the automatic penetration testing in the Multi-Objective Reinforcement Learning (MORL) framework and propose a Chebyshev decomposition critic to find diverse adversary strategies that balance different objectives in the penetration test. Additionally, the number of available actions increases with the agent consistently probing the target network, making the training process intractable in many practical situations. Thus, we introduce a coverage-based masking mechanism that reduces attention on previously selected actions to help the agent adapt to future exploration. Experimental evaluation on a range of scenarios demonstrates the superiority of our proposed approach when compared to adapted algorithms in terms of multi-objective learning and performance efficiency.

摘要: 渗透测试通过模拟真实的主动对手，在评估目标网络的安全性方面起着至关重要的作用。深度强化学习(RL)被认为是一种很有前途的解决方案，可以通过减少人工工作量和提高可靠性来实现渗透测试过程的自动化。现有的RL解决方案侧重于寻找特定的攻击路径来影响目标主机。然而，在现实中，需要一系列不同的攻击变体来提供对目标网络安全级别的全面评估。因此，攻击代理在渗透网络时必须考虑多个目标。然而，在现有的文献中，这一挑战没有得到充分的解决。为此，我们制定了多目标强化学习(MORL)框架中的自动渗透测试，并提出了切比雪夫分解批评者来寻找在渗透测试中平衡不同目标的不同对手策略。此外，随着代理持续探测目标网络，可用操作的数量增加，使得培训过程在许多实际情况下变得棘手。因此，我们引入了一种基于覆盖的掩蔽机制，该机制减少了对先前选择的操作的关注，以帮助代理适应未来的探索。在一系列场景上的实验评估表明，与自适应算法相比，我们提出的方法在多目标学习和性能效率方面具有优越性。



