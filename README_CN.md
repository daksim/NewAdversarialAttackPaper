# Latest Adversarial Attack Papers
**update at 2023-05-03 17:36:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

神经机器翻译系统中情感感知的敌意攻击 cs.CL

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01437v1) [paper-pdf](http://arxiv.org/pdf/2305.01437v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly.

摘要: 随着深度学习方法的出现，神经机器翻译(NMT)系统变得越来越强大。然而，基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入的不可察觉的变化可能会导致系统输出的不希望看到的变化。到目前为止，很少有人研究针对序列到序列系统的对抗性攻击，例如NMT模型。NMT之前的工作已经检查了攻击，目的是在输出序列中引入目标短语。本文从输出感知的角度研究了NMT系统的敌意攻击问题。因此，攻击的目的是改变对输出序列的感知，而不改变对输入序列的感知。例如，对手可能会扭曲翻译后的评论的情绪，使其具有夸大的积极情绪。在实践中，进行广泛的人类感知实验是具有挑战性的，因此将代理深度学习分类器应用于NMT输出来测量感知变化。实验表明，NMT系统输出序列的情感感知可以发生显著变化。



## **2. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01361v1) [paper-pdf](http://arxiv.org/pdf/2305.01361v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial perturbations. Our extensive experimental results verify the effectiveness of our proposed method, which significantly enhances the transferability of adversarial samples against various baseline models and defense strategies.The source code of this study is available at \href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}.

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解的Top-1奇异值关联特征来计算输出Logit来进行攻击，然后将输出Logit与原始Logit相结合来优化对手的扰动。我们的大量实验结果验证了我们所提出的方法的有效性，它显著地提高了针对不同基线模型和防御策略的对抗性样本的可转移性。这项研究的源代码可以在\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}.上找到



## **3. Improving adversarial robustness by putting more regularizations on less robust samples**

通过对健壮性较差的样本进行更多的正则化来提高对手的稳健性 stat.ML

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2206.03353v3) [paper-pdf](http://arxiv.org/pdf/2206.03353v3)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。与现有的正则化算法相比，该算法的一个新特点是对易受敌意攻击的数据进行了更多的正则化。理论上，我们的算法可以理解为最小化正则化经验风险的算法，该正则化经验风险是由新导出的稳健风险上界引起的。数值实验表明，我们提出的算法同时提高了泛化(例题准确率)和稳健性(对抗性攻击准确率)，达到了最好的性能。



## **4. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2203.16000v3) [paper-pdf](http://arxiv.org/pdf/2203.16000v3)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **5. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

暴露Face反欺骗模型的细粒度攻击漏洞 cs.CV

Accepted by IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR) Workshop, 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2205.14851v3) [paper-pdf](http://arxiv.org/pdf/2205.14851v3)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.

摘要: 人脸反欺骗的目的是区分伪造的人脸图像(如打印的照片)和活的人脸图像。然而，对抗性的例子极大地挑战了它的可信度，在那里添加一些扰动噪声很容易改变预测。以往的工作采用对抗性攻击的方法来评估人脸的反欺骗性能，没有任何细粒度的分析来确定哪个模型、架构或辅助特征容易受到对手的攻击。为了解决这个问题，我们提出了一种新的框架来暴露人脸反欺骗模型的细粒度攻击漏洞，该框架由多任务模块和语义特征增强(SFA)模块组成。多任务模块可以获得不同的语义特征用于进一步的评估，但仅攻击这些语义特征并不能反映与歧视相关的脆弱性。然后，我们设计了SFA模块来引入数据分布，以获得更多与区分相关的梯度方向，以生成对抗性示例。综合实验表明，SFA模块的攻击成功率平均提高了近40美元。我们在不同的注释、几何地图和骨干网络(例如RESNET网络)上进行了这种细粒度的对抗性分析。这些细粒度的对抗性实例可用于选择健壮的主干网络和辅助特征。它们还可以用于对抗性训练，从而进一步提高人脸反欺骗模型的准确性和稳健性。



## **6. Stratified Adversarial Robustness with Rejection**

具有拒绝的分层对抗健壮性 cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01139v1) [paper-pdf](http://arxiv.org/pdf/2305.01139v1)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

摘要: 最近，对抗性地训练具有拒绝选项的分类器(也称为选择性分类器)以增强对抗性健壮性是一种新的兴趣。虽然拒绝在许多应用中可能会导致成本，但现有研究通常将零成本与拒绝扰动输入联系在一起，这可能导致拒绝许多可以正确分类的轻微扰动输入。在这项工作中，我们研究了分层拒绝环境下的具有拒绝的对抗性鲁棒分类，其中拒绝代价由拒绝损失函数来建模，拒绝损失函数在扰动幅度上单调地不增加。我们从理论上分析了分层拒绝的设置，并提出了一种新的防御方法--基于一致预测拒绝的对抗训练(CPR)--来构建一个健壮的选择性分类器。在图像数据集上的实验表明，该方法在强自适应攻击下的性能明显优于已有方法。例如，在CIFAR-10上，CPR在看得见和看不见的攻击下都将总的稳健损失(针对不同的拒绝损失)减少了至少7.3%。



## **7. Randomized Reversible Gate-Based Obfuscation for Secured Compilation of Quantum Circuit**

基于随机可逆门的量子电路安全编译混淆算法 quant-ph

11 pages, 12 figures, conference

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01133v1) [paper-pdf](http://arxiv.org/pdf/2305.01133v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: The success of quantum circuits in providing reliable outcomes for a given problem depends on the gate count and depth in near-term noisy quantum computers. Quantum circuit compilers that decompose high-level gates to native gates of the hardware and optimize the circuit play a key role in quantum computing. However, the quality and time complexity of the optimization process can vary significantly especially for practically relevant large-scale quantum circuits. As a result, third-party (often less-trusted/untrusted) compilers have emerged, claiming to provide better and faster optimization of complex quantum circuits than so-called trusted compilers. However, untrusted compilers can pose severe security risks, such as the theft of sensitive intellectual property (IP) embedded within the quantum circuit. We propose an obfuscation technique for quantum circuits using randomized reversible gates to protect them from such attacks during compilation. The idea is to insert a small random circuit into the original circuit and send it to the untrusted compiler. Since the circuit function is corrupted, the adversary may get incorrect IP. However, the user may also get incorrect output post-compilation. To circumvent this issue, we concatenate the inverse of the random circuit in the compiled circuit to recover the original functionality. We demonstrate the practicality of our method by conducting exhaustive experiments on a set of benchmark circuits and measuring the quality of obfuscation by calculating the Total Variation Distance (TVD) metric. Our method achieves TVD of up to 1.92 and performs at least 2X better than a previously reported obfuscation method. We also propose a novel adversarial reverse engineering (RE) approach and show that the proposed obfuscation is resilient against RE attacks. The proposed technique introduces minimal degradation in fidelity (~1% to ~3% on average).

摘要: 量子电路在为给定问题提供可靠结果方面的成功取决于近期嘈杂的量子计算机中的门数量和深度。量子电路编译器将高层门分解为硬件的本机门，并对电路进行优化，在量子计算中发挥着关键作用。然而，优化过程的质量和时间复杂性可能会有很大的变化，特别是对于实际相关的大规模量子电路。因此，第三方(通常不太可信/不可信)编译器应运而生，声称比所谓的可信编译器提供更好、更快的复杂量子电路优化。然而，不可信的编译器可能会带来严重的安全风险，例如嵌入量子电路中的敏感知识产权(IP)被窃取。我们提出了一种量子电路的混淆技术，该技术使用随机化的可逆门来保护它们在编译过程中免受此类攻击。其想法是在原始电路中插入一个小的随机电路，并将其发送给不可信的编译器。由于电路功能被破坏，对手可能获得错误的IP。但是，用户也可能在编译后得到不正确的输出。为了避免这个问题，我们将编译电路中的随机电路的逆连接起来，以恢复原来的功能。我们在一组基准电路上进行了详尽的实验，并通过计算总变化距离(TVD)度量来衡量混淆质量，从而证明了该方法的实用性。我们的方法获得了高达1.92的TVD，并且比先前报道的混淆方法的性能至少提高了2倍。我们还提出了一种新的对抗性逆向工程(RE)方法，并证明了该方法对逆向工程攻击具有较强的抵抗力。提出的技术在保真度方面引入了最小的降级(平均~1%到~3%)。



## **8. Evaluating Adversarial Robustness on Document Image Classification**

文档图像分类中的对抗健壮性评价 cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2304.12486v2) [paper-pdf](http://arxiv.org/pdf/2304.12486v2)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.

摘要: 近年来，对抗性攻击和防御对计算机视觉系统产生了越来越大的兴趣，但截至目前，大多数调查仅限于图像。然而，许多人工智能模型实际上处理的是纪实数据，这与现实世界的图像有很大不同。因此，在这项工作中，我们试图将对抗性攻击的理念应用于文献和自然数据，并保护模型免受此类攻击。我们的工作集中在基于非目标梯度、基于转移和基于分数的攻击上，并评估了对抗性训练、JPEG输入压缩和灰度输入变换对ResNet50和EfficientNetB0模型架构的健壮性的影响。据我们所知，社区还没有进行过这样的工作，以研究这些攻击对文档图像分类任务的影响。



## **9. Physical Adversarial Attacks for Surveillance: A Survey**

用于监视的物理对抗性攻击：综述 cs.CV

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.01074v1) [paper-pdf](http://arxiv.org/pdf/2305.01074v1)

**Authors**: Kien Nguyen, Tharindu Fernando, Clinton Fookes, Sridha Sridharan

**Abstract**: Modern automated surveillance techniques are heavily reliant on deep learning methods. Despite the superior performance, these learning systems are inherently vulnerable to adversarial attacks - maliciously crafted inputs that are designed to mislead, or trick, models into making incorrect predictions. An adversary can physically change their appearance by wearing adversarial t-shirts, glasses, or hats or by specific behavior, to potentially avoid various forms of detection, tracking and recognition of surveillance systems; and obtain unauthorized access to secure properties and assets. This poses a severe threat to the security and safety of modern surveillance systems. This paper reviews recent attempts and findings in learning and designing physical adversarial attacks for surveillance applications. In particular, we propose a framework to analyze physical adversarial attacks and provide a comprehensive survey of physical adversarial attacks on four key surveillance tasks: detection, identification, tracking, and action recognition under this framework. Furthermore, we review and analyze strategies to defend against the physical adversarial attacks and the methods for evaluating the strengths of the defense. The insights in this paper present an important step in building resilience within surveillance systems to physical adversarial attacks.

摘要: 现代自动监控技术严重依赖深度学习方法。尽管性能优越，但这些学习系统天生就容易受到敌意攻击--恶意设计的输入旨在误导或欺骗模型做出错误的预测。敌手可以通过穿着敌意的t恤、眼镜或帽子或通过特定的行为来改变自己的外表，以潜在地避免监视系统的各种形式的检测、跟踪和识别；并获得对安全财产和资产的未经授权的访问。这对现代监控系统的安全保障构成了严重威胁。本文回顾了最近在学习和设计用于监视应用的物理对抗性攻击方面的尝试和发现。特别是，我们提出了一个分析物理对抗攻击的框架，并在该框架下对物理对抗攻击的四个关键监视任务：检测、识别、跟踪和动作识别进行了全面的调查。此外，我们还回顾和分析了防御物理对抗性攻击的策略和评估防御强度的方法。本文的见解代表了在监视系统中建立对物理对手攻击的复原力的重要一步。



## **10. IoTFlowGenerator: Crafting Synthetic IoT Device Traffic Flows for Cyber Deception**

IoTFlowGenerator：为网络欺骗精心制作合成物联网设备流量 cs.CR

FLAIRS-36

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00925v1) [paper-pdf](http://arxiv.org/pdf/2305.00925v1)

**Authors**: Joseph Bao, Murat Kantarcioglu, Yevgeniy Vorobeychik, Charles Kamhoua

**Abstract**: Over the years, honeypots emerged as an important security tool to understand attacker intent and deceive attackers to spend time and resources. Recently, honeypots are being deployed for Internet of things (IoT) devices to lure attackers, and learn their behavior. However, most of the existing IoT honeypots, even the high interaction ones, are easily detected by an attacker who can observe honeypot traffic due to lack of real network traffic originating from the honeypot. This implies that, to build better honeypots and enhance cyber deception capabilities, IoT honeypots need to generate realistic network traffic flows. To achieve this goal, we propose a novel deep learning based approach for generating traffic flows that mimic real network traffic due to user and IoT device interactions. A key technical challenge that our approach overcomes is scarcity of device-specific IoT traffic data to effectively train a generator. We address this challenge by leveraging a core generative adversarial learning algorithm for sequences along with domain specific knowledge common to IoT devices. Through an extensive experimental evaluation with 18 IoT devices, we demonstrate that the proposed synthetic IoT traffic generation tool significantly outperforms state of the art sequence and packet generators in remaining indistinguishable from real traffic even to an adaptive attacker.

摘要: 多年来，蜜罐成为一种重要的安全工具，用于了解攻击者的意图并欺骗攻击者花费时间和资源。最近，物联网(IoT)设备正在部署蜜罐，以引诱攻击者，并了解他们的行为。然而，由于缺乏源自蜜罐的真实网络流量，大多数现有的物联网蜜罐，即使是高交互的蜜罐，也很容易被攻击者检测到，攻击者可以观察到蜜罐流量。这意味着，为了构建更好的蜜罐并增强网络欺骗能力，物联网蜜罐需要生成真实的网络流量。为了实现这一目标，我们提出了一种新的基于深度学习的方法来生成模拟真实网络流量的用户和物联网设备交互流量。我们的方法克服的一个关键技术挑战是缺乏特定于设备的物联网流量数据来有效培训发电机。我们通过利用针对序列的核心生成性对抗性学习算法以及物联网设备常见的特定领域知识来应对这一挑战。通过对18个物联网设备的广泛实验评估，我们证明了所提出的合成物联网流量生成工具的性能显著优于最新的序列和数据包生成器，即使对于自适应攻击者也是如此。



## **11. Attack-SAM: Towards Evaluating Adversarial Robustness of Segment Anything Model**

攻击-SAM：评估分段Anything模型的对抗健壮性 cs.CV

The first work to evaluate the adversarial robustness of Segment  Anything Model (ongoing)

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00866v1) [paper-pdf](http://arxiv.org/pdf/2305.00866v1)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, previous task-specific models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be easily fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. Specifically, we find that SAM is vulnerable to white-box attacks while maintaining robustness to some extent in the black-box setting. This is an ongoing project and more results and findings will be updated soon through https://github.com/chenshuang-zhang/attack-sam.

摘要: 分段任意模型(SAM)最近受到了极大的关注，因为它在各种下游任务上以零-短的方式表现出令人印象深刻的性能。计算机视觉(CV)领域可能会跟随自然语言处理(NLP)领域，走上一条从特定于任务的视觉模型到基础模型的道路。然而，以前的特定于任务的模型被广泛认为容易受到对抗性例子的影响，这些例子愚弄了模型，使其在不知不觉中做出了错误的预测。在将深度模型应用于安全敏感应用程序时，此类易受敌意攻击的漏洞会引起严重关注。因此，了解VISION基础模型SAM是否也容易被对手攻击愚弄是至关重要的。据我们所知，我们的工作是第一次对如何用对抗性例子攻击SAM进行全面调查。具体地说，我们发现SAM在黑盒环境下很容易受到白盒攻击，同时在一定程度上保持了健壮性。这是一个正在进行的项目，更多的结果和发现将很快通过https://github.com/chenshuang-zhang/attack-sam.更新



## **12. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

ICASSP 2023

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2210.06284v4) [paper-pdf](http://arxiv.org/pdf/2210.06284v4)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **13. Robustness of Graph Neural Networks at Scale**

图神经网络的尺度稳健性 cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2110.14038v4) [paper-pdf](http://arxiv.org/pdf/2110.14038v4)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.

摘要: 图神经网络(GNN)因其普及性和应用的多样性而变得越来越重要。然而，现有的关于它们对对手攻击的脆弱性的研究依赖于相对较小的图表。我们解决了这一差距，并研究了如何大规模攻击和防御GNN。我们提出了两种稀疏性感知的一阶优化攻击，它们在对节点数目为二次的多个参数进行优化的情况下仍能保持有效的表示。我们证明了常见的代理损失并不适用于针对GNN的全球攻击。我们的替代品可以使攻击强度加倍。此外，为了提高GNN的可靠性，我们设计了一个稳健的聚集函数--软中值，从而在所有尺度上都能得到有效的防御。与以前的工作相比，我们在大于100倍的图上使用标准GNN来评估我们的攻击和防御。我们甚至通过将我们的技术扩展到可扩展的GNN来进一步扩展一个数量级。



## **14. Assessing Vulnerabilities of Adversarial Learning Algorithm through Poisoning Attacks**

利用中毒攻击评估对抗性学习算法的脆弱性 cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00399v1) [paper-pdf](http://arxiv.org/pdf/2305.00399v1)

**Authors**: Jingfeng Zhang, Bo Song, Bo Han, Lei Liu, Gang Niu, Masashi Sugiyama

**Abstract**: Adversarial training (AT) is a robust learning algorithm that can defend against adversarial attacks in the inference phase and mitigate the side effects of corrupted data in the training phase. As such, it has become an indispensable component of many artificial intelligence (AI) systems. However, in high-stake AI applications, it is crucial to understand AT's vulnerabilities to ensure reliable deployment. In this paper, we investigate AT's susceptibility to poisoning attacks, a type of malicious attack that manipulates training data to compromise the performance of the trained model. Previous work has focused on poisoning attacks against standard training, but little research has been done on their effectiveness against AT. To fill this gap, we design and test effective poisoning attacks against AT. Specifically, we investigate and design clean-label poisoning attacks, allowing attackers to imperceptibly modify a small fraction of training data to control the algorithm's behavior on a specific target data point. Additionally, we propose the clean-label untargeted attack, enabling attackers can attach tiny stickers on training data to degrade the algorithm's performance on all test data, where the stickers could serve as a signal against unauthorized data collection. Our experiments demonstrate that AT can still be poisoned, highlighting the need for caution when using vanilla AT algorithms in security-related applications. The code is at https://github.com/zjfheart/Poison-adv-training.git.

摘要: 对抗性训练(AT)是一种稳健的学习算法，它能在推理阶段抵抗敌意攻击，在训练阶段减轻数据损坏的副作用。因此，它已经成为许多人工智能(AI)系统不可或缺的组成部分。然而，在高风险的人工智能应用中，了解AT的漏洞以确保可靠的部署至关重要。在本文中，我们调查了AT对中毒攻击的敏感性，这是一种操纵训练数据以损害训练模型性能的恶意攻击。以前的工作主要集中在针对标准训练的投毒攻击，但关于它们对抗AT的有效性的研究很少。为了填补这一空白，我们设计并测试了针对AT的有效中毒攻击。具体地说，我们调查和设计干净标签中毒攻击，允许攻击者在不知不觉中修改一小部分训练数据，以控制算法在特定目标数据点上的行为。此外，我们提出了干净标签无目标攻击，使攻击者能够在训练数据上贴上微小的贴纸，以降低算法在所有测试数据上的性能，其中贴纸可以作为反对未经授权的数据收集的信号。我们的实验证明AT仍然可能中毒，这突出了在与安全相关的应用程序中使用普通AT算法时需要谨慎的必要性。代码在https://github.com/zjfheart/Poison-adv-training.git.



## **15. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

对抗性不变正则化增强对抗性对比学习 cs.LG

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00374v1) [paper-pdf](http://arxiv.org/pdf/2305.00374v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL), without requiring labels, incorporates adversarial data with standard contrastive learning (SCL) and outputs a robust representation which is generalizable and resistant to adversarial attacks and common corruptions. The style-independence property of representations has been validated to be beneficial in improving robustness transferability. Standard invariant regularization (SIR) has been proposed to make the learned representations via SCL to be independent of the style factors. However, how to equip robust representations learned via ACL with the style-independence property is still unclear so far. To this end, we leverage the technique of causal reasoning to propose an adversarial invariant regularization (AIR) that enforces robust representations learned via ACL to be style-independent. Then, we enhance ACL using invariant regularization (IR), which is a weighted sum of SIR and AIR. Theoretically, we show that AIR implicitly encourages the prediction of adversarial data and consistency between adversarial and natural data to be independent of data augmentations. We also theoretically demonstrate that the style-independence property of robust representation learned via ACL still holds in downstream tasks, providing generalization guarantees. Empirically, our comprehensive experimental results corroborate that IR can significantly improve the performance of ACL and its variants on various datasets.

摘要: 对抗性对比学习(ACL)不需要标签，将对抗性数据与标准对比学习(SCL)相结合，输出具有泛化能力和抵抗对抗性攻击和常见腐败的稳健表示。事实证明，表示的风格无关性对于提高健壮性和可转移性是有益的。标准不变量正则化(SIR)被提出，以使通过SCL学习的表示独立于风格因素。然而，到目前为止，如何用样式无关的属性来装备通过ACL学习的健壮表示仍然不清楚。为此，我们利用因果推理技术提出了一种对抗不变正则化(AIR)，它强制通过ACL学习的稳健表示独立于样式。然后，我们使用不变正则化(IR)来增强ACL，IR是SIR和AIR的加权和。理论上，我们证明了AIR隐含地鼓励对抗性数据的预测以及对抗性数据和自然数据之间的一致性独立于数据扩充。我们还从理论上证明了通过ACL学习的健壮表示的风格无关性在下游任务中仍然成立，提供了泛化保证。实验结果表明，IR能够显著提高ACL及其变体在不同数据集上的性能。



## **16. MetaShard: A Novel Sharding Blockchain Platform for Metaverse Applications**

MetaShard：一种适用于Metverse应用的新型分片区块链平台 cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00367v1) [paper-pdf](http://arxiv.org/pdf/2305.00367v1)

**Authors**: Cong T. Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Yong Xiao, Dusit Niyato, Eryk Dutkiewicz

**Abstract**: Due to its security, transparency, and flexibility in verifying virtual assets, blockchain has been identified as one of the key technologies for Metaverse. Unfortunately, blockchain-based Metaverse faces serious challenges such as massive resource demands, scalability, and security concerns. To address these issues, this paper proposes a novel sharding-based blockchain framework, namely MetaShard, for Metaverse applications. Particularly, we first develop an effective consensus mechanism, namely Proof-of-Engagement, that can incentivize MUs' data and computing resource contribution. Moreover, to improve the scalability of MetaShard, we propose an innovative sharding management scheme to maximize the network's throughput while protecting the shards from 51% attacks. Since the optimization problem is NP-complete, we develop a hybrid approach that decomposes the problem (using the binary search method) into sub-problems that can be solved effectively by the Lagrangian method. As a result, the proposed approach can obtain solutions in polynomial time, thereby enabling flexible shard reconfiguration and reducing the risk of corruption from the adversary. Extensive numerical experiments show that, compared to the state-of-the-art commercial solvers, our proposed approach can achieve up to 66.6% higher throughput in less than 1/30 running time. Moreover, the proposed approach can achieve global optimal solutions in most experiments.

摘要: 区块链因其在验证虚拟资产方面的安全性、透明度和灵活性，已被确定为Metverse的关键技术之一。不幸的是，基于区块链的Metverse面临着巨大的资源需求、可扩展性和安全问题等严峻挑战。针对这些问题，本文提出了一种新的基于分片的区块链框架MetaShard。特别是，我们首先开发了一个有效的共识机制，即参与度证明，可以激励MU的数据和计算资源贡献。此外，为了提高MetaShard的可扩展性，我们提出了一种创新的分片管理方案，在最大化网络吞吐量的同时保护分片免受51%的攻击。由于优化问题是NP完全的，我们提出了一种混合方法，将问题分解成可以用拉格朗日方法有效求解的子问题。因此，所提出的方法可以在多项式时间内获得解，从而实现灵活的分片重新配置，并降低来自对手的破坏风险。大量的数值实验表明，与最先进的商业求解器相比，我们提出的方法可以在不到1/30的运行时间内获得高达66.6%的吞吐量。此外，该方法在大多数实验中都能获得全局最优解。



## **17. FedGrad: Mitigating Backdoor Attacks in Federated Learning Through Local Ultimate Gradients Inspection**

FedGrad：通过局部最终梯度检测缓解联合学习中的后门攻击 cs.CV

Accepted for presentation at the International Joint Conference on  Neural Networks (IJCNN 2023)

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2305.00328v1) [paper-pdf](http://arxiv.org/pdf/2305.00328v1)

**Authors**: Thuy Dung Nguyen, Anh Duy Nguyen, Kok-Seng Wong, Huy Hieu Pham, Thanh Hung Nguyen, Phi Le Nguyen, Truong Thao Nguyen

**Abstract**: Federated learning (FL) enables multiple clients to train a model without compromising sensitive data. The decentralized nature of FL makes it susceptible to adversarial attacks, especially backdoor insertion during training. Recently, the edge-case backdoor attack employing the tail of the data distribution has been proposed as a powerful one, raising questions about the shortfall in current defenses' robustness guarantees. Specifically, most existing defenses cannot eliminate edge-case backdoor attacks or suffer from a trade-off between backdoor-defending effectiveness and overall performance on the primary task. To tackle this challenge, we propose FedGrad, a novel backdoor-resistant defense for FL that is resistant to cutting-edge backdoor attacks, including the edge-case attack, and performs effectively under heterogeneous client data and a large number of compromised clients. FedGrad is designed as a two-layer filtering mechanism that thoroughly analyzes the ultimate layer's gradient to identify suspicious local updates and remove them from the aggregation process. We evaluate FedGrad under different attack scenarios and show that it significantly outperforms state-of-the-art defense mechanisms. Notably, FedGrad can almost 100% correctly detect the malicious participants, thus providing a significant reduction in the backdoor effect (e.g., backdoor accuracy is less than 8%) while not reducing the main accuracy on the primary task.

摘要: 联合学习(FL)使多个客户能够在不损害敏感数据的情况下训练模型。FL的分散性使其容易受到对手的攻击，特别是训练期间的后门插入。最近，利用数据分布尾部的边缘情况后门攻击被提出为一种强有力的攻击，这引发了人们对当前防御系统健壮性保证不足的质疑。具体地说，大多数现有的防御系统无法消除边缘情况下的后门攻击，或者在后门防御效率和主要任务的整体性能之间进行权衡。为了应对这一挑战，我们提出了FedGrad，这是一种针对FL的新型后门防御机制，它能够抵抗包括Edge-Case攻击在内的尖端后门攻击，并在异类客户端数据和大量受攻击的客户端下有效执行。FedGrad被设计为一种两层过滤机制，它彻底分析最终层的梯度，以识别可疑的本地更新并将它们从聚合过程中删除。我们在不同的攻击场景下对FedGrad进行了评估，结果表明它的性能明显优于最先进的防御机制。值得注意的是，FedGrad几乎可以100%正确地检测恶意参与者，从而显著降低后门效应(例如，后门准确率低于8%)，同时不降低主要任务的主要准确率。



## **18. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

组合对抗性机器学习的博弈论混合专家 cs.LG

17pages, 10 figures

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2211.14669v2) [paper-pdf](http://arxiv.org/pdf/2211.14669v2)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically customized to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). We first conduct a transferability analysis, to demonstrate the adversarial examples generated by customized attacks on one defense, are not often misclassified by another defense.   This finding leads to two important questions. First, how can the low transferability between defenses be utilized in a game theoretic framework to improve the robustness? Second, how can an adversary within this framework develop effective multi-model attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses. Our framework is called Game theoretic Mixed Experts (GaME). It is designed to find the Mixed-Nash strategy for both a detector based and standard defender, when facing an attacker employing compositional adversarial attacks. We further propose three new attack algorithms, specifically designed to target defenses with randomized transformations, multi-model voting schemes, and adversarial detector architectures. These attacks serve to both strengthen defenses generated by the GaME framework and verify their robustness against unforeseen attacks. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.

摘要: 对抗性机器学习的最新进展表明，被认为是健壮的防御实际上容易受到针对其弱点而专门定制的对抗性攻击。这些防御包括随机变换弹幕(BART)、友好对手训练(FAT)、垃圾就是宝藏(TIT)以及由视觉变形金刚(VITS)、大转移模型和尖峰神经网络(SNN)组成的集成模型。我们首先进行可转移性分析，以证明定制攻击对一个防御系统生成的对抗性示例不会经常被另一个防御系统错误分类。这一发现引出了两个重要问题。首先，如何在博弈论框架中利用防守之间的低可转换性来提高健壮性？第二，在这个框架内的对手如何开发有效的多模式攻击？在这篇文章中，我们为集成对抗性攻击和防御提供了一个博弈论框架。我们的框架称为博弈论混合专家(GAME)。它的设计是为了在面对使用成分对抗攻击的攻击者时，为基于检测器的和标准的防守者找到混合纳什策略。我们进一步提出了三种新的攻击算法，分别针对随机变换、多模型投票方案和对抗性检测器体系结构的目标防御而设计。这些攻击既加强了游戏框架产生的防御，又验证了它们对不可预见的攻击的健壮性。总体而言，我们的框架和分析通过对组合攻击和防御公式产生新的见解，促进了对抗性机器学习领域的发展。



## **19. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

提高多重攻击下的高光谱对抗健壮性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2210.16346v3) [paper-pdf](http://arxiv.org/pdf/2210.16346v3)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.

摘要: 对高光谱图像进行分类的语义分割模型容易受到敌意例子的影响。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法的性能会下降。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。



## **20. The Power of Typed Affine Decision Structures: A Case Study**

类型化仿射决策结构的威力：案例研究 cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14888v1) [paper-pdf](http://arxiv.org/pdf/2304.14888v1)

**Authors**: Gerrit Nolte, Maximilian Schlüter, Alnis Murtovi, Bernhard Steffen

**Abstract**: TADS are a novel, concise white-box representation of neural networks. In this paper, we apply TADS to the problem of neural network verification, using them to generate either proofs or concise error characterizations for desirable neural network properties. In a case study, we consider the robustness of neural networks to adversarial attacks, i.e., small changes to an input that drastically change a neural networks perception, and show that TADS can be used to provide precise diagnostics on how and where robustness errors a occur. We achieve these results by introducing Precondition Projection, a technique that yields a TADS describing network behavior precisely on a given subset of its input space, and combining it with PCA, a traditional, well-understood dimensionality reduction technique. We show that PCA is easily compatible with TADS. All analyses can be implemented in a straightforward fashion using the rich algebraic properties of TADS, demonstrating the utility of the TADS framework for neural network explainability and verification. While TADS do not yet scale as efficiently as state-of-the-art neural network verifiers, we show that, using PCA-based simplifications, they can still scale to mediumsized problems and yield concise explanations for potential errors that can be used for other purposes such as debugging a network or generating new training samples.

摘要: TADS是神经网络的一种新颖、简洁的白盒表示。在本文中，我们将TADS应用于神经网络验证问题，使用它们来生成期望的神经网络性质的证明或简洁的误差特征。在一个案例研究中，我们考虑了神经网络对对抗性攻击的稳健性，即输入的微小变化极大地改变了神经网络的感知，并表明TADS可以用来提供关于健壮性错误如何以及在哪里发生的精确诊断。我们通过引入预条件投影来获得这些结果，这是一种在输入空间的给定子集上产生精确描述网络行为的TADS的技术，并将其与传统的、众所周知的降维技术PCA相结合。实验结果表明，主元分析算法与TADS算法具有很好的兼容性。所有的分析都可以使用TADS丰富的代数特性以一种简单的方式实现，展示了TADS框架在神经网络可解释性和验证方面的实用性。虽然TADS还没有像最先进的神经网络验证器那样有效地进行扩展，但我们表明，使用基于PCA的简化，它们仍然可以扩展到中等规模的问题，并为潜在错误提供简明的解释，这些解释可以用于其他目的，如调试网络或生成新的训练样本。



## **21. Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models**

针对黑盒神经网络排序模型的主题对抗性攻击 cs.IR

Accepted by SIGIR 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14867v1) [paper-pdf](http://arxiv.org/pdf/2304.14867v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have attracted considerable attention in information retrieval. Unfortunately, NRMs may inherit the adversarial vulnerabilities of general neural networks, which might be leveraged by black-hat search engine optimization practitioners. Recently, adversarial attacks against NRMs have been explored in the paired attack setting, generating an adversarial perturbation to a target document for a specific query. In this paper, we focus on a more general type of perturbation and introduce the topic-oriented adversarial ranking attack task against NRMs, which aims to find an imperceptible perturbation that can promote a target document in ranking for a group of queries with the same topic. We define both static and dynamic settings for the task and focus on decision-based black-box attacks. We propose a novel framework to improve topic-oriented attack performance based on a surrogate ranking model. The attack problem is formalized as a Markov decision process (MDP) and addressed using reinforcement learning. Specifically, a topic-oriented reward function guides the policy to find a successful adversarial example that can be promoted in rankings to as many queries as possible in a group. Experimental results demonstrate that the proposed framework can significantly outperform existing attack strategies, and we conclude by re-iterating that there exist potential risks for applying NRMs in the real world.

摘要: 神经排序模型(NRM)在信息检索领域引起了广泛的关注。不幸的是，NRM可能会继承一般神经网络的对抗性漏洞，这可能会被黑帽搜索引擎优化从业者利用。最近，针对NRM的对抗性攻击已经在配对攻击设置中被探索，为特定查询生成对目标文档的对抗性扰动。在本文中，我们着眼于一种更一般的扰动类型，并引入了针对NRMS的面向主题的对抗性排名攻击任务，其目的是找到一种可以促进目标文档对同一主题的一组查询进行排名的不可察觉的扰动。我们定义了任务的静态和动态设置，并专注于基于决策的黑盒攻击。提出了一种新的基于代理排名模型的面向主题攻击性能改进框架。攻击问题被形式化化为马尔可夫决策过程(MDP)，并使用强化学习来解决。具体地说，面向主题的奖励功能引导策略找到一个成功的对抗性例子，该例子可以在排名中提升到一个组中尽可能多的查询。实验结果表明，该框架的性能明显优于已有的攻击策略，并通过反复验证得出结论：在现实世界中应用NRM存在潜在的风险。



## **22. False Claims against Model Ownership Resolution**

针对所有权解决方案范本的虚假索赔 cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.06607v2) [paper-pdf](http://arxiv.org/pdf/2304.06607v2)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的经验评估，我们证明了我们的虚假声明攻击在所有具有现实配置的著名MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **23. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

通过量子噪声验证量子分类器对敌意例子的鲁棒性 quant-ph

Accepted to IEEE ICASSP 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2211.00887v2) [paper-pdf](http://arxiv.org/pdf/2211.00887v2)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been found to be vulnerable to adversarial attacks, in which quantum classifiers are deceived by imperceptible noises, leading to misclassification. In this paper, we propose the first theoretical study demonstrating that adding quantum random rotation noise can improve robustness in quantum classifiers against adversarial attacks. We link the definition of differential privacy and show that the quantum classifier trained with the natural presence of additive noise is differentially private. Finally, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples, supported by experimental results simulated with noises from IBM's 7-qubits device.

摘要: 最近，量子分类器被发现容易受到敌意攻击，其中量子分类器被不可感知的噪声欺骗，导致错误分类。在本文中，我们提出了第一个理论研究，证明了加入量子随机旋转噪声可以提高量子分类器对敌意攻击的稳健性。我们将差分隐私的定义联系起来，证明了在自然存在加性噪声的情况下训练的量子分类器是差分隐私的。最后，我们得到了一个证明的稳健性界限，使量子分类器能够防御敌对的例子，支持用来自IBM的7量子比特设备的噪声模拟的实验结果。



## **24. Fusion is Not Enough: Single-Modal Attacks to Compromise Fusion Models in Autonomous Driving**

融合是不够的：自动驾驶中破坏融合模型的单模式攻击 cs.CV

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14614v1) [paper-pdf](http://arxiv.org/pdf/2304.14614v1)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely adopted for perception in autonomous vehicles (AVs), particularly for the task of 3D object detection with camera and LiDAR sensors. The rationale behind fusion is to capitalize on the strengths of each modality while mitigating their limitations. The exceptional and leading performance of fusion models has been demonstrated by advanced deep neural network (DNN)-based fusion techniques. Fusion models are also perceived as more robust to attacks compared to single-modal ones due to the redundant information in multiple modalities. In this work, we challenge this perspective with single-modal attacks that targets the camera modality, which is considered less significant in fusion but more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion models with adversarial patches. Our approach employs a two-stage optimization-based strategy that first comprehensively assesses vulnerable image areas under adversarial attacks, and then applies customized attack strategies to different fusion models, generating deployable patches. Evaluations with five state-of-the-art camera-LiDAR fusion models on a real-world dataset show that our attacks successfully compromise all models. Our approach can either reduce the mean average precision (mAP) of detection performance from 0.824 to 0.353 or degrade the detection score of the target object from 0.727 to 0.151 on average, demonstrating the effectiveness and practicality of our proposed attack framework.

摘要: 多传感器融合(MSF)被广泛地应用于自主车辆的感知，特别是在具有摄像机和激光雷达传感器的三维目标检测任务中。融合背后的基本原理是利用每种方式的优势，同时减轻它们的局限性。先进的基于深度神经网络(DNN)的融合技术证明了融合模型的卓越和领先的性能。由于多模式中的冗余信息，融合模型也被认为比单模式模型更具稳健性。在这项工作中，我们通过以相机通道为目标的单模式攻击来挑战这一观点，相机通道在融合中被认为不那么重要，但对于攻击者来说更负担得起。我们认为融合模型的最薄弱环节取决于它们最脆弱的通道，并提出了一种针对带有对抗性补丁的高级Camera-LiDAR融合模型的攻击框架。该方法采用两阶段优化策略，首先综合评估图像在敌方攻击下的易受攻击区域，然后将定制的攻击策略应用于不同的融合模型，生成可部署的补丁。在真实数据集上对五种最先进的相机-LiDAR融合模型进行的评估表明，我们的攻击成功地折衷了所有模型。我们的方法可以将检测性能的平均精度(MAP)从0.824降低到0.353，或者将目标对象的检测得分从平均0.727降低到0.151，从而证明了我们所提出的攻击框架的有效性和实用性。



## **25. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

基于在线深度强化学习的高效奖赏中毒攻击 cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2205.14842v2) [paper-pdf](http://arxiv.org/pdf/2205.14842v2)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstract**: We study reward poisoning attacks on online deep reinforcement learning (DRL), where the attacker is oblivious to the learning algorithm used by the agent and the dynamics of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general, black-box reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct two new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. We provide a theoretical analysis of the efficiency of our attack and perform an extensive empirical evaluation. Our results show that our attacks efficiently poison agents learning in several popular classical control and MuJoCo environments with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc.

摘要: 研究了在线深度强化学习(DRL)中的奖赏中毒攻击，攻击者对智能体使用的学习算法和环境的动态特性视而不见。我们通过设计一个称为对抗性MDP攻击的通用黑盒奖励中毒框架来展示最新的DRL算法的内在脆弱性。我们实例化了我们的框架，构造了两个新的攻击，这两个攻击只破坏了总训练时间步骤的一小部分奖励，并使代理学习一个低性能的策略。我们对我们的攻击效率进行了理论分析，并进行了广泛的经验评估。我们的结果表明，我们的攻击有效地毒化了在几个流行的经典控制和MuJoCo环境中学习的代理，并使用了各种先进的DRL算法，如DQN，PPO，SAC等。



## **26. Adversary Aware Continual Learning**

对手意识到的持续学习 cs.LG

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14483v1) [paper-pdf](http://arxiv.org/pdf/2304.14483v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstract**: Class incremental learning approaches are useful as they help the model to learn new information (classes) sequentially, while also retaining the previously acquired information (classes). However, it has been shown that such approaches are extremely vulnerable to the adversarial backdoor attacks, where an intelligent adversary can introduce small amount of misinformation to the model in the form of imperceptible backdoor pattern during training to cause deliberate forgetting of a specific task or class at test time. In this work, we propose a novel defensive framework to counter such an insidious attack where, we use the attacker's primary strength-hiding the backdoor pattern by making it imperceptible to humans-against it, and propose to learn a perceptible (stronger) pattern (also during the training) that can overpower the attacker's imperceptible (weaker) pattern. We demonstrate the effectiveness of the proposed defensive mechanism through various commonly used Replay-based (both generative and exact replay-based) class incremental learning algorithms using continual learning benchmark variants of CIFAR-10, CIFAR-100, and MNIST datasets. Most noteworthy, our proposed defensive framework does not assume that the attacker's target task and target class is known to the defender. The defender is also unaware of the shape, size, and location of the attacker's pattern. We show that our proposed defensive framework considerably improves the performance of class incremental learning algorithms with no knowledge of the attacker's target task, attacker's target class, and attacker's imperceptible pattern. We term our defensive framework as Adversary Aware Continual Learning (AACL).

摘要: 类增量学习方法是有用的，因为它们帮助模型顺序地学习新的信息(类)，同时还保留了以前获得的信息(类)。然而，已有研究表明，这种方法极易受到对抗性后门攻击，在这种攻击中，聪明的对手可以在训练期间以不可察觉的后门模式的形式向模型引入少量错误信息，从而导致在测试时故意忘记特定的任务或类。在这项工作中，我们提出了一个新的防御框架来对抗这样的潜伏攻击，其中我们使用攻击者的主要优势-通过使其对人类不可察觉来隐藏后门模式-来对抗它，并建议学习一种可感知(更强)的模式(也是在训练期间)，该模式可以压倒攻击者不可察觉(更弱)的模式。我们使用CIFAR-10、CIFAR-100和MNIST数据集的连续学习基准变量，通过各种常用的基于重放(包括生成和精确重放)的类增量学习算法，验证了所提出的防御机制的有效性。最值得注意的是，我们提出的防御框架并不假设攻击者的目标任务和目标类对防御者是已知的。防御者也不知道攻击者图案的形状、大小和位置。我们的结果表明，在不知道攻击者的目标任务、攻击者的目标类和攻击者的不可察觉的模式的情况下，我们提出的防御框架显著提高了类增量学习算法的性能。我们将我们的防御框架称为对手感知持续学习(AACL)。



## **27. Attacking Fake News Detectors via Manipulating News Social Engagement**

通过操纵新闻社会参与打击假新闻检测器 cs.SI

ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2302.07363v3) [paper-pdf](http://arxiv.org/pdf/2302.07363v3)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.

摘要: 社交媒体是新闻消费的主要来源之一，尤其是在年轻一代中。随着新闻消费在各种社交媒体平台上的日益流行，错误信息激增，其中包括虚假信息或毫无根据的说法。随着各种基于文本和社会语境的假新闻检测器被提出来检测社交媒体上的错误信息，最近的研究开始关注假新闻检测器的脆弱性。本文提出了第一个针对基于图神经网络(GNN)的假新闻检测器的对抗性攻击框架，以探讨其健壮性。具体地说，我们利用多智能体强化学习(MAIL)框架来模拟社交媒体上欺诈者的对抗行为。研究表明，在现实世界中，欺诈者相互协调，分享不同的新闻，以躲避假新闻检测器的检测。因此，我们将我们的Marl框架建模为一个包含BOT、半机械人和群工代理的马尔可夫博弈，这些代理都有自己独特的成本、预算和影响。然后，我们使用深度Q-学习来搜索最大化回报的最优策略。在两个真实假新闻传播数据集上的大量实验结果表明，我们提出的框架可以有效地破坏基于GNN的假新闻检测器的性能。希望本文能为今后的假新闻检测研究提供一些启示。



## **28. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

点对点分散机器学习的安全性研究 cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning")

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.08443v2) [paper-pdf](http://arxiv.org/pdf/2205.08443v2)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.

摘要: 在这项工作中，我们进行了第一次，深入的，隐私分析的分散学习--一个合作的机器学习框架，旨在解决联合学习的主要限制。我们针对被动的和主动的分散攻击引入了一套新的攻击。我们证明，与去中心化学习提出者所声称的相反，去中心化学习并不比联邦学习提供任何安全优势。相反，它增加了攻击面，使系统中的任何用户都可以执行诸如梯度反转等隐私攻击，甚至获得对诚实用户的本地模型的完全控制。我们还表明，考虑到保护技术的最新水平，去中心化学习的隐私保护配置需要完全连接的网络，失去了与联邦设置相比的任何实际优势，因此完全违背了去中心化方法的目标。



## **29. Robust Resilient Signal Reconstruction under Adversarial Attacks**

对抗性攻击下的稳健恢复信号重构 math.OC

7 pages

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/1807.08004v2) [paper-pdf](http://arxiv.org/pdf/1807.08004v2)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Lalit Mestha, Hema Achanta

**Abstract**: We consider the problem of signal reconstruction for a system under sparse signal corruption by a malicious agent. The reconstruction problem follows the standard error coding problem that has been studied extensively in the literature. We include a new challenge of robust estimation of the attack support. The problem is then cast as a constrained optimization problem merging promising techniques in the area of deep learning and estimation theory. A pruning algorithm is developed to reduce the ``false positive" uncertainty of data-driven attack localization results, thereby improving the probability of correct signal reconstruction. Sufficient conditions for the correct reconstruction and the associated reconstruction error bounds are obtained for both exact and inexact attack support estimation. Moreover, a simulation of a water distribution system is presented to validate the proposed techniques.

摘要: 我们考虑了在稀疏信号被恶意代理破坏的情况下系统的信号重构问题。重建问题遵循在文献中已被广泛研究的标准误差编码问题。我们包括了一个新的挑战，即稳健地估计攻击支持。然后将该问题归结为一个约束优化问题，融合了深度学习和估计理论领域中有前途的技术。为了减少数据驱动攻击定位结果的“假阳性”不确定性，从而提高正确重构信号的概率，提出了一种剪枝算法.对于准确和不精确的攻击支持度估计，得到了正确重构的充分条件和相应的重构误差界.此外，通过对供水系统的仿真，验证了所提方法的有效性.



## **30. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

QEVSEC：通过动态无线电能传输实现电动汽车快速安全充电 cs.CR

6 pages, conference

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.10292v2) [paper-pdf](http://arxiv.org/pdf/2205.10292v2)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.

摘要: 动态无线电能传输(DWPT)可用于电动汽车(EV)行驶时的按需充电。然而，DWPT带来了许多安全和隐私方面的问题。最近，研究人员证明了DWPT系统容易受到敌意攻击。在电动汽车充电场景中，攻击者可以阻止授权客户充电，通过向受害用户收费来获得免费费用，并跟踪目标车辆。依赖于集中式解决方案的最先进的身份验证方案要么容易受到各种攻击，要么具有很高的计算复杂性，不适合动态场景。本文提出了一种新颖、安全、高效的电动汽车动态充电认证协议--快速电动汽车安全充电协议。我们对QEVSEC的想法源于我们在最先进的协议中发现的多个漏洞，该协议允许跟踪用户活动，并且容易受到重播攻击。基于这些观察，提出的协议解决了这些问题，并通过在很短的消息交换中仅使用原始密码操作来实现较低的计算复杂度。QEVSEC在每次迭代中提供了可扩展性和更低的成本，从而降低了对电网所需电力的影响。



## **31. Boosting Big Brother: Attacking Search Engines with Encodings**

助推老大哥：用编码攻击搜索引擎 cs.CR

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14031v1) [paper-pdf](http://arxiv.org/pdf/2304.14031v1)

**Authors**: Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross Anderson, Mauro Conti

**Abstract**: Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.

摘要: 搜索引擎容易受到针对索引和搜索的攻击，这些攻击是通过文本编码操作进行的。通过使用不常见的编码表示在不知不觉中扰乱文本，攻击者可以控制特定搜索查询的搜索引擎结果。我们展示了对两个主要的商业搜索引擎--Google和Bing--和一个开源搜索引擎--Elasticearch的攻击是成功的。我们进一步证明了该攻击对LLM聊天搜索是成功的，包括Bing的GPT-4聊天机器人和Google的Bard聊天机器人。我们还提出了针对文本摘要和抄袭检测模型的攻击的一个变体，这两个ML任务与搜索密切相关。我们提供了一套针对这些技术的防御措施，并警告说，攻击者可以利用这些攻击对毫无戒心的用户发起虚假信息运动，从而刺激搜索引擎维护人员修补已部署的系统的需求。



## **32. You Can't Always Check What You Wanted: Selective Checking and Trusted Execution to Prevent False Actuations in Cyber-Physical Systems**

你不能总是检查你想要的：选择性检查和可信的执行，以防止网络物理系统中的错误驱动 cs.CR

Extended version of SCATE published in ISORC'23

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13956v1) [paper-pdf](http://arxiv.org/pdf/2304.13956v1)

**Authors**: Monowar Hasan, Sibin Mohan

**Abstract**: Cyber-physical systems (CPS) are vulnerable to attacks targeting outgoing actuation commands that modify their physical behaviors. The limited resources in such systems, coupled with their stringent timing constraints, often prevents the checking of every outgoing command. We present a "selective checking" mechanism that uses game-theoretic modeling to identify the right subset of commands to be checked in order to deter an adversary. This mechanism is coupled with a "delay-aware" trusted execution environment (TEE) to ensure that only verified actuation commands are ever sent to the physical system, thus maintaining their safety and integrity. The selective checking and trusted execution (SCATE) framework is implemented on an off-the-shelf ARM platform running standard embedded Linux. We demonstrate the effectiveness of SCATE using four realistic cyber-physical systems (a ground rover, a flight controller, a robotic arm and an automated syringe pump) and study design trade-offs. Not only does SCATE provide a high level of security and high performance, it also suffers from significantly lower overheads (30.48%-47.32% less) in the process. In fact, SCATE can work with more systems without negatively affecting the safety of the system. Considering that most CPS do not have any such checking mechanisms, and SCATE is guaranteed to meet all the timing requirements (i.e., ensure the safety/integrity of the system), our methods can significantly improve the security (and, hence, safety) of the system.

摘要: 网络物理系统(CP)很容易受到攻击，攻击目标是修改其物理行为的传出启动命令。这种系统中的有限资源，再加上它们严格的定时限制，通常会阻止检查每个传出命令。我们提出了一种“选择性检查”机制，它使用博弈论建模来识别要检查的正确命令子集，以威慑对手。该机制与“延迟感知”的可信执行环境(TEE)相结合，以确保只有经过验证的启动命令才会被发送到物理系统，从而维护它们的安全性和完整性。选择性检查和可信执行(SCATE)框架在运行标准嵌入式Linux的现成ARM平台上实现。我们使用四个现实的数字物理系统(一个地面漫游车、一个飞行控制器、一个机械臂和一个自动注射泵)来演示SCATE的有效性，并研究设计的权衡。SCATE不仅提供高级别的安全性和高性能，而且在此过程中的管理费用显著降低(减少30.48%-47.32%)。事实上，SCATE可以与更多的系统一起工作，而不会对系统的安全性造成负面影响。考虑到大多数CP没有任何这样的检查机制，并且SCATE保证满足所有的定时要求(即，确保系统的安全性/完整性)，我们的方法可以显著提高系统的安全性(因此，安全性)。



## **33. Network Cascade Vulnerability using Constrained Bayesian Optimization**

基于约束贝叶斯优化的网络级联漏洞 cs.SI

11 pages, 3 figures

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14420v1) [paper-pdf](http://arxiv.org/pdf/2304.14420v1)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Extensive experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is resource constrained, it is still possible to find settings that produce cascades comparable in severity to instances where there are no constraints.

摘要: 电网脆弱性的衡量标准通常是根据对手对网络造成的破坏程度来评估的。然而，这类攻击的连锁影响往往被忽视，尽管连锁是大规模停电的主要原因之一。本文探讨了输电线路保护设置的修改作为对抗性攻击的候选对象，只要网络平衡状态保持不变，这种攻击就可以保持不可检测。这形成了贝叶斯优化过程中的黑盒函数的基础，其中的目标是找到使由于级联而导致的网络降级最大化的保护设置。广泛的实验表明，与传统观点相反，最大限度地错误配置所有网络线路的保护设置并不会导致最大程度的级联。更令人惊讶的是，即使错误配置的程度是受资源限制的，仍有可能找到与没有限制的情况在严重性上相当的级联设置。



## **34. Detection of Adversarial Physical Attacks in Time-Series Image Data**

时序图像数据中敌意物理攻击的检测 cs.CV

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.13919v1) [paper-pdf](http://arxiv.org/pdf/2304.13919v1)

**Authors**: Ramneet Kaur, Yiannis Kantaros, Wenwen Si, James Weimer, Insup Lee

**Abstract**: Deep neural networks (DNN) have become a common sensing modality in autonomous systems as they allow for semantically perceiving the ambient environment given input images. Nevertheless, DNN models have proven to be vulnerable to adversarial digital and physical attacks. To mitigate this issue, several detection frameworks have been proposed to detect whether a single input image has been manipulated by adversarial digital noise or not. In our prior work, we proposed a real-time detector, called VisionGuard (VG), for adversarial physical attacks against single input images to DNN models. Building upon that work, we propose VisionGuard* (VG), which couples VG with majority-vote methods, to detect adversarial physical attacks in time-series image data, e.g., videos. This is motivated by autonomous systems applications where images are collected over time using onboard sensors for decision-making purposes. We emphasize that majority-vote mechanisms are quite common in autonomous system applications (among many other applications), as e.g., in autonomous driving stacks for object detection. In this paper, we investigate, both theoretically and experimentally, how this widely used mechanism can be leveraged to enhance the performance of adversarial detectors. We have evaluated VG* on videos of both clean and physically attacked traffic signs generated by a state-of-the-art robust physical attack. We provide extensive comparative experiments against detectors that have been designed originally for out-of-distribution data and digitally attacked images.

摘要: 深度神经网络(DNN)已经成为自主系统中一种常见的感知方式，因为它们允许在给定输入图像的情况下从语义上感知周围环境。然而，DNN模型已被证明容易受到敌意的数字和物理攻击。为了缓解这一问题，已经提出了几个检测框架来检测单个输入图像是否被敌对的数字噪声操纵。在我们之前的工作中，我们提出了一个称为VisionGuard(VG)的实时检测器，用于对DNN模型中的单输入图像进行敌意物理攻击。在这项工作的基础上，我们提出了VisionGuard*(VG)，它将VG与多数投票方法相结合，用于检测时间序列图像数据(如视频)中的敌意物理攻击。这是由自主系统应用程序推动的，这些应用程序使用机载传感器随着时间的推移收集图像，以用于决策目的。我们强调，多数投票机制在自主系统应用(在许多其他应用中)中相当常见，例如在用于对象检测的自动驾驶堆栈中。在本文中，我们从理论和实验两方面研究了如何利用这一广泛使用的机制来提高对抗性检测器的性能。我们已经评估了VG*在由最先进的强大的物理攻击生成的干净和物理攻击的交通标志的视频上。我们针对最初为散布数据和受到数字攻击的图像设计的检测器进行了广泛的比较实验。



## **35. Learning Robust Deep Equilibrium Models**

学习稳健的深度均衡模型 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.12707v2) [paper-pdf](http://arxiv.org/pdf/2304.12707v2)

**Authors**: Haoyu Chu, Shikui Wei, Ting Liu, Yao Zhao

**Abstract**: Deep equilibrium (DEQ) models have emerged as a promising class of implicit layer models in deep learning, which abandon traditional depth by solving for the fixed points of a single nonlinear layer. Despite their success, the stability of the fixed points for these models remains poorly understood. Recently, Lyapunov theory has been applied to Neural ODEs, another type of implicit layer model, to confer adversarial robustness. By considering DEQ models as nonlinear dynamic systems, we propose a robust DEQ model named LyaDEQ with guaranteed provable stability via Lyapunov theory. The crux of our method is ensuring the fixed points of the DEQ models are Lyapunov stable, which enables the LyaDEQ models to resist minor initial perturbations. To avoid poor adversarial defense due to Lyapunov-stable fixed points being located near each other, we add an orthogonal fully connected layer after the Lyapunov stability module to separate different fixed points. We evaluate LyaDEQ models on several widely used datasets under well-known adversarial attacks, and experimental results demonstrate significant improvement in robustness. Furthermore, we show that the LyaDEQ model can be combined with other defense methods, such as adversarial training, to achieve even better adversarial robustness.

摘要: 深度平衡(DEQ)模型是深度学习中一类很有前途的隐层模型，它通过求解单个非线性层的不动点来抛弃传统的深度模型。尽管它们取得了成功，但这些模型的固定点的稳定性仍然知之甚少。最近，Lyapunov理论被应用于另一种类型的隐含层模型--神经常微分方程组，以赋予对手健壮性。将DEQ模型视为非线性动态系统，利用Lyapunov理论，提出了一种具有可证明稳定性的鲁棒DEQ模型LyaDEQ。我们方法的关键是确保DEQ模型的不动点是Lyapunov稳定的，这使得LyaDEQ模型能够抵抗微小的初始扰动。为了避免Lyapunov稳定不动点位置较近造成的对抗性差，我们在Lyapunov稳定模后增加了一个正交全连通层来分离不同的不动点。我们在几个广泛使用的数据集上对LyaDEQ模型进行了评估，实验结果表明，LyaDEQ模型在稳健性方面有了显著的提高。此外，我们还证明了LyaDEQ模型可以与其他防御方法相结合，例如对抗训练，以获得更好的对抗健壮性。



## **36. One-vs-the-Rest Loss to Focus on Important Samples in Adversarial Training**

在对抗性训练中专注于重要样本的一对一损失 cs.LG

ICML2023, 26 pages, 19 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2207.10283v3) [paper-pdf](http://arxiv.org/pdf/2207.10283v3)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Kentaro Ohno, Yasutoshi Ida

**Abstract**: This paper proposes a new loss function for adversarial training. Since adversarial training has difficulties, e.g., necessity of high model capacity, focusing on important data points by weighting cross-entropy loss has attracted much attention. However, they are vulnerable to sophisticated attacks, e.g., Auto-Attack. This paper experimentally reveals that the cause of their vulnerability is their small margins between logits for the true label and the other labels. Since neural networks classify the data points based on the logits, logit margins should be large enough to avoid flipping the largest logit by the attacks. Importance-aware methods do not increase logit margins of important samples but decrease those of less-important samples compared with cross-entropy loss. To increase logit margins of important samples, we propose switching one-vs-the-rest loss (SOVR), which switches from cross-entropy to one-vs-the-rest loss for important samples that have small logit margins. We prove that one-vs-the-rest loss increases logit margins two times larger than the weighted cross-entropy loss for a simple problem. We experimentally confirm that SOVR increases logit margins of important samples unlike existing methods and achieves better robustness against Auto-Attack than importance-aware methods.

摘要: 本文提出了一种新的对抗性训练损失函数。由于对抗性训练的难度很大，例如需要较高的模型容量，因此通过加权交叉熵损失来关注重要数据点的方法引起了人们的广泛关注。然而，它们很容易受到复杂的攻击，例如自动攻击。本文通过实验揭示了它们易受攻击的原因是真实标签的对数与其他标签的对数之间的差值很小。由于神经网络根据Logit对数据点进行分类，因此Logit边际应该足够大，以避免因攻击而翻转最大的Logit。与交叉熵损失相比，重要性感知方法不会增加重要样本的Logit裕度，但会减少不太重要的样本的Logit裕度。为了提高重要样本的Logit裕度，我们提出了切换一对一损失(SOVR)，对于Logit裕度较小的重要样本，它从交叉熵切换为一对休息损失。我们证明，对于一个简单的问题，一对一的损失比加权的交叉熵损失增加了两倍的Logit边际。实验证实，与现有方法相比，SOVR提高了重要样本的Logit裕度，并且比重要性感知方法获得了更好的对自动攻击的健壮性。



## **37. Improving Adversarial Transferability by Intermediate-level Perturbation Decay**

利用中层扰动衰减提高对手的可转换性 cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13410v1) [paper-pdf](http://arxiv.org/pdf/2304.13410v1)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **38. Blockchain-based Access Control for Secure Smart Industry Management Systems**

基于区块链的安全智能工业管理系统访问控制 cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13379v1) [paper-pdf](http://arxiv.org/pdf/2304.13379v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Mohammad Saidur Rahman, Abdelaziz Bouras

**Abstract**: Smart manufacturing systems involve a large number of interconnected devices resulting in massive data generation. Cloud computing technology has recently gained increasing attention in smart manufacturing systems for facilitating cost-effective service provisioning and massive data management. In a cloud-based manufacturing system, ensuring authorized access to the data is crucial. A cloud platform is operated under a single authority. Hence, a cloud platform is prone to a single point of failure and vulnerable to adversaries. An internal or external adversary can easily modify users' access to allow unauthorized users to access the data. This paper proposes a role-based access control to prevent modification attacks by leveraging blockchain and smart contracts in a cloud-based smart manufacturing system. The role-based access control is developed to determine users' roles and rights in smart contracts. The smart contracts are then deployed to the private blockchain network. We evaluate our solution by utilizing Ethereum private blockchain network to deploy the smart contract. The experimental results demonstrate the feasibility and evaluation of the proposed framework's performance.

摘要: 智能制造系统涉及大量互联设备，产生了海量数据。云计算技术最近在智能制造系统中获得了越来越多的关注，以促进经济高效的服务提供和海量数据管理。在基于云的制造系统中，确保授权访问数据至关重要。云平台是在单一授权下运行的。因此，云平台容易出现单点故障，容易受到对手的攻击。内部或外部对手可以很容易地修改用户的访问权限，以允许未经授权的用户访问数据。在基于云的智能制造系统中，通过利用区块链和智能契约，提出了一种基于角色的访问控制来防止修改攻击。基于角色的访问控制是为了确定用户在智能合同中的角色和权限。然后，智能合同被部署到私有区块链网络。我们通过利用以太私有区块链网络部署智能合同来评估我们的解决方案。实验结果证明了该框架的可行性和性能评估。



## **39. Blockchain-based Federated Learning with SMPC Model Verification Against Poisoning Attack for Healthcare Systems**

基于区块链的联合学习与SMPC模型验证的医疗系统抗中毒攻击 cs.CR

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13360v1) [paper-pdf](http://arxiv.org/pdf/2304.13360v1)

**Authors**: Aditya Pribadi Kalapaaking, Ibrahim Khalil, Xun Yi

**Abstract**: Due to the rising awareness of privacy and security in machine learning applications, federated learning (FL) has received widespread attention and applied to several areas, e.g., intelligence healthcare systems, IoT-based industries, and smart cities. FL enables clients to train a global model collaboratively without accessing their local training data. However, the current FL schemes are vulnerable to adversarial attacks. Its architecture makes detecting and defending against malicious model updates difficult. In addition, most recent studies to detect FL from malicious updates while maintaining the model's privacy have not been sufficiently explored. This paper proposed blockchain-based federated learning with SMPC model verification against poisoning attacks for healthcare systems. First, we check the machine learning model from the FL participants through an encrypted inference process and remove the compromised model. Once the participants' local models have been verified, the models are sent to the blockchain node to be securely aggregated. We conducted several experiments with different medical datasets to evaluate our proposed framework.

摘要: 由于机器学习应用中隐私和安全意识的提高，联合学习(FL)受到了广泛的关注，并应用于智能医疗系统、基于物联网的行业和智能城市等领域。FL使客户能够协作地训练全局模型，而无需访问其本地训练数据。然而，当前的FL方案容易受到对手的攻击。其体系结构使得检测和防御恶意模型更新变得困难。此外，最近关于在保护模型隐私的同时从恶意更新中检测FL的研究还没有得到充分的探索。针对医疗系统的中毒攻击，提出了基于区块链的联合学习和SMPC模型验证。首先，我们通过加密的推理过程检查FL参与者的机器学习模型，并删除被攻破的模型。一旦参与者的本地模型经过验证，模型就会被发送到区块链节点进行安全聚合。我们使用不同的医学数据集进行了几个实验来评估我们提出的框架。



## **40. On the Risks of Stealing the Decoding Algorithms of Language Models**

论窃取语言模型译码算法的风险 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2303.04729v3) [paper-pdf](http://arxiv.org/pdf/2303.04729v3)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.

摘要: 从现代语言模型(LM)生成文本的一个关键组件是解码算法的选择和调整。这些算法确定如何从LM生成的内部概率分布生成文本。选择解码算法和调整其超参数的过程需要大量的时间、人工和计算，还需要广泛的人工评估。因此，这种译码算法的恒等式和超参数被认为对它们的所有者非常有价值。在这项工作中，我们首次证明，具有典型API访问权限的攻击者可以以非常低的金钱成本窃取其解码算法的类型和超参数。我们的攻击对文本生成API中使用的流行LMS有效，包括GPT-2和GPT-3。我们证明了只需几美元即可窃取此类信息的可行性，例如，对于GPT-3的四个版本，仅需$0.8$、$1$、$4$和$40$。



## **41. SHIELD: Thwarting Code Authorship Attribution**

盾牌：挫败代码作者归属 cs.CR

12 pages, 13 figures

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13255v1) [paper-pdf](http://arxiv.org/pdf/2304.13255v1)

**Authors**: Mohammed Abuhamad, Changhun Jung, David Mohaisen, DaeHun Nyang

**Abstract**: Authorship attribution has become increasingly accurate, posing a serious privacy risk for programmers who wish to remain anonymous. In this paper, we introduce SHIELD to examine the robustness of different code authorship attribution approaches against adversarial code examples. We define four attacks on attribution techniques, which include targeted and non-targeted attacks, and realize them using adversarial code perturbation. We experiment with a dataset of 200 programmers from the Google Code Jam competition to validate our methods targeting six state-of-the-art authorship attribution methods that adopt a variety of techniques for extracting authorship traits from source-code, including RNN, CNN, and code stylometry. Our experiments demonstrate the vulnerability of current authorship attribution methods against adversarial attacks. For the non-targeted attack, our experiments demonstrate the vulnerability of current authorship attribution methods against the attack with an attack success rate exceeds 98.5\% accompanied by a degradation of the identification confidence that exceeds 13\%. For the targeted attacks, we show the possibility of impersonating a programmer using targeted-adversarial perturbations with a success rate ranging from 66\% to 88\% for different authorship attribution techniques under several adversarial scenarios.

摘要: 作者身份的归属变得越来越准确，这对希望保持匿名的程序员构成了严重的隐私风险。在本文中，我们引入Shield来检验不同代码作者归属方法对敌意代码示例的稳健性。我们定义了四种基于归因技术的攻击，包括定向攻击和非定向攻击，并使用对抗性代码扰动来实现它们。我们使用来自Google Code Jam竞赛的200名程序员的数据集来验证我们的方法，目标是六种最先进的作者归属方法，这些方法采用了各种技术从源代码中提取作者特征，包括RNN、CNN和代码样式法。我们的实验证明了现有的作者归属方法在抵抗敌意攻击时的脆弱性。对于非目标攻击，我们的实验证明了现有作者归属方法对攻击的脆弱性，攻击成功率超过98.5%，同时身份识别置信度下降超过13%。对于有针对性的攻击，我们证明了使用有针对性的对抗性扰动来模拟程序员的可能性，在几种对抗性场景下，对于不同的作者归属技术，成功率从66\%到88\%不等。



## **42. Generating Adversarial Examples with Task Oriented Multi-Objective Optimization**

面向任务的多目标优化生成对抗性实例 cs.LG

**SubmitDate**: 2023-04-26    [abs](http://arxiv.org/abs/2304.13229v1) [paper-pdf](http://arxiv.org/pdf/2304.13229v1)

**Authors**: Anh Bui, Trung Le, He Zhao, Quan Tran, Paul Montague, Dinh Phung

**Abstract**: Deep learning models, even the-state-of-the-art ones, are highly vulnerable to adversarial examples. Adversarial training is one of the most efficient methods to improve the model's robustness. The key factor for the success of adversarial training is the capability to generate qualified and divergent adversarial examples which satisfy some objectives/goals (e.g., finding adversarial examples that maximize the model losses for simultaneously attacking multiple models). Therefore, multi-objective optimization (MOO) is a natural tool for adversarial example generation to achieve multiple objectives/goals simultaneously. However, we observe that a naive application of MOO tends to maximize all objectives/goals equally, without caring if an objective/goal has been achieved yet. This leads to useless effort to further improve the goal-achieved tasks, while putting less focus on the goal-unachieved tasks. In this paper, we propose \emph{Task Oriented MOO} to address this issue, in the context where we can explicitly define the goal achievement for a task. Our principle is to only maintain the goal-achieved tasks, while letting the optimizer spend more effort on improving the goal-unachieved tasks. We conduct comprehensive experiments for our Task Oriented MOO on various adversarial example generation schemes. The experimental results firmly demonstrate the merit of our proposed approach. Our code is available at \url{https://github.com/tuananhbui89/TAMOO}.

摘要: 深度学习模型，即使是最先进的模型，也非常容易受到对抗性例子的影响。对抗性训练是提高模型稳健性的最有效方法之一。对抗性训练成功的关键因素是生成满足某些目标/目标的合格的和不同的对抗性范例的能力(例如，找到同时攻击多个模型的最大化模型损失的对抗性范例)。因此，多目标优化(MOO)是敌方实例生成同时实现多个目标/目标的一种自然工具。然而，我们观察到，天真地应用MoO往往会平等地最大化所有目标/目标，而不关心某个目标/目标是否已经实现。这导致进一步改进目标已完成任务的努力是徒劳的，而对目标未完成任务的关注较少。在本文中，我们提出了在明确定义任务的目标实现的情况下，解决这一问题的方法。我们的原则是只维护已实现目标的任务，而让优化器将更多精力花在改进未实现目标的任务上。我们对我们的面向任务的MOO在不同的对抗性实例生成方案上进行了全面的实验。实验结果有力地证明了该方法的优点。我们的代码可在\url{https://github.com/tuananhbui89/TAMOO}.



## **43. Uncovering the Representation of Spiking Neural Networks Trained with Surrogate Gradient**

揭示用代理梯度训练的尖峰神经网络的表示 cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.13098v1) [paper-pdf](http://arxiv.org/pdf/2304.13098v1)

**Authors**: Yuhang Li, Youngeun Kim, Hyoungseob Park, Priyadarshini Panda

**Abstract**: Spiking Neural Networks (SNNs) are recognized as the candidate for the next-generation neural networks due to their bio-plausibility and energy efficiency. Recently, researchers have demonstrated that SNNs are able to achieve nearly state-of-the-art performance in image recognition tasks using surrogate gradient training. However, some essential questions exist pertaining to SNNs that are little studied: Do SNNs trained with surrogate gradient learn different representations from traditional Artificial Neural Networks (ANNs)? Does the time dimension in SNNs provide unique representation power? In this paper, we aim to answer these questions by conducting a representation similarity analysis between SNNs and ANNs using Centered Kernel Alignment (CKA). We start by analyzing the spatial dimension of the networks, including both the width and the depth. Furthermore, our analysis of residual connections shows that SNNs learn a periodic pattern, which rectifies the representations in SNNs to be ANN-like. We additionally investigate the effect of the time dimension on SNN representation, finding that deeper layers encourage more dynamics along the time dimension. We also investigate the impact of input data such as event-stream data and adversarial attacks. Our work uncovers a host of new findings of representations in SNNs. We hope this work will inspire future research to fully comprehend the representation power of SNNs. Code is released at https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.

摘要: 尖峰神经网络(SNN)因其生物合理性和能量效率而被认为是下一代神经网络的候选网络。最近，研究人员已经证明，在使用代理梯度训练的图像识别任务中，SNN能够获得几乎最先进的性能。然而，与SNN相关的一些基本问题很少被研究：用代理梯度训练的SNN是否学习不同于传统人工神经网络(ANN)的表示？SNN中的时间维度是否提供了独特的表征能力？在本文中，我们旨在通过使用中心核对齐(CKA)对SNN和ANN之间的表示相似性进行分析来回答这些问题。我们首先分析网络的空间维度，包括宽度和深度。此外，我们对剩余连接的分析表明，SNN学习一个周期性的模式，这将SNN中的表示纠正为类似ANN的表示。此外，我们还研究了时间维度对SNN表示的影响，发现更深的层促进了沿时间维度的更多动力学。我们还研究了输入数据的影响，如事件流数据和对抗性攻击。我们的工作揭示了SNN中表征的一系列新发现。我们希望这项工作将启发未来的研究，以充分理解SNN的表征能力。代码在https://github.com/Intelligent-Computing-Lab-Yale/SNNCKA.上发布



## **44. Improving Robustness Against Adversarial Attacks with Deeply Quantized Neural Networks**

利用深度量化神经网络提高对抗攻击的稳健性 cs.LG

Accepted at IJCNN 2023. 8 pages, 5 figures

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.12829v1) [paper-pdf](http://arxiv.org/pdf/2304.12829v1)

**Authors**: Ferheen Ayaz, Idris Zakariyya, José Cano, Sye Loong Keoh, Jeremy Singer, Danilo Pau, Mounia Kharbouche-Harrari

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, particularly Deep Neural Networks (DNNs), is essential to enable their deployment into resource-constrained tiny devices. However, a disadvantage of DNN models is their vulnerability to adversarial attacks, as they can be fooled by adding slight perturbations to the inputs. Therefore, the challenge is how to create accurate, robust, and tiny DNN models deployable on resource-constrained embedded devices. This paper reports the results of devising a tiny DNN model, robust to adversarial black and white box attacks, trained with an automatic quantizationaware training framework, i.e. QKeras, with deep quantization loss accounted in the learning loop, thereby making the designed DNNs more accurate for deployment on tiny devices. We investigated how QKeras and an adversarial robustness technique, Jacobian Regularization (JR), can provide a co-optimization strategy by exploiting the DNN topology and the per layer JR approach to produce robust yet tiny deeply quantized DNN models. As a result, a new DNN model implementing this cooptimization strategy was conceived, developed and tested on three datasets containing both images and audio inputs, as well as compared its performance with existing benchmarks against various white-box and black-box attacks. Experimental results demonstrated that on average our proposed DNN model resulted in 8.3% and 79.5% higher accuracy than MLCommons/Tiny benchmarks in the presence of white-box and black-box attacks on the CIFAR-10 image dataset and a subset of the Google Speech Commands audio dataset respectively. It was also 6.5% more accurate for black-box attacks on the SVHN image dataset.

摘要: 减少机器学习(ML)模型的内存占用，特别是深度神经网络(DNN)，对于使其能够部署到资源受限的微型设备是至关重要的。然而，DNN模型的一个缺点是它们容易受到对抗性攻击，因为它们可以通过在输入中添加轻微的扰动来愚弄它们。因此，面临的挑战是如何创建可在资源受限的嵌入式设备上部署的准确、健壮和微小的DNN模型。设计了一种对黑白盒攻击具有较强鲁棒性的微型DNN模型，该模型使用一个自动量化感知训练框架QKera进行训练，并在学习循环中考虑了深度量化损失，从而使所设计的DNN更适合于部署在微型设备上。我们研究了QKera和一种对抗性健壮性技术雅可比正则化(JR)如何通过利用DNN拓扑和逐层JR方法来提供联合优化策略来产生健壮但微小的深度量化的DNN模型。因此，在三个同时包含图像和音频输入的数据集上构思、开发和测试了一个新的DNN模型，并将其与现有的针对各种白盒和黑盒攻击的基准测试进行了比较。实验结果表明，在CIFAR-10图像数据集和Google语音命令音频数据子集上分别存在白盒和黑盒攻击的情况下，我们提出的DNN模型的准确率比MLCommons/Tiny基准分别高8.3%和79.5%。对于SVHN图像数据集的黑盒攻击，它的准确率也提高了6.5%。



## **45. RobCaps: Evaluating the Robustness of Capsule Networks against Affine Transformations and Adversarial Attacks**

RobCaps：评估胶囊网络对仿射变换和对手攻击的稳健性 cs.LG

To appear at the 2023 International Joint Conference on Neural  Networks (IJCNN), Queensland, Australia, June 2023

**SubmitDate**: 2023-04-25    [abs](http://arxiv.org/abs/2304.03973v2) [paper-pdf](http://arxiv.org/pdf/2304.03973v2)

**Authors**: Alberto Marchisio, Antonio De Marco, Alessio Colucci, Maurizio Martina, Muhammad Shafique

**Abstract**: Capsule Networks (CapsNets) are able to hierarchically preserve the pose relationships between multiple objects for image classification tasks. Other than achieving high accuracy, another relevant factor in deploying CapsNets in safety-critical applications is the robustness against input transformations and malicious adversarial attacks.   In this paper, we systematically analyze and evaluate different factors affecting the robustness of CapsNets, compared to traditional Convolutional Neural Networks (CNNs). Towards a comprehensive comparison, we test two CapsNet models and two CNN models on the MNIST, GTSRB, and CIFAR10 datasets, as well as on the affine-transformed versions of such datasets. With a thorough analysis, we show which properties of these architectures better contribute to increasing the robustness and their limitations. Overall, CapsNets achieve better robustness against adversarial examples and affine transformations, compared to a traditional CNN with a similar number of parameters. Similar conclusions have been derived for deeper versions of CapsNets and CNNs. Moreover, our results unleash a key finding that the dynamic routing does not contribute much to improving the CapsNets' robustness. Indeed, the main generalization contribution is due to the hierarchical feature learning through capsules.

摘要: 胶囊网络(CapsNets)能够在图像分类任务中分层地保持多个对象之间的姿势关系。在安全关键型应用程序中部署CapsNet的另一个相关因素是对输入转换和恶意对手攻击的稳健性。本文系统地分析和评价了影响CapsNets健壮性的各种因素，并与传统卷积神经网络(CNN)进行了比较。为了进行全面的比较，我们在MNIST、GTSRB和CIFAR10数据集以及这些数据集的仿射变换版本上测试了两个CapsNet模型和两个CNN模型。通过深入的分析，我们展示了这些体系结构的哪些属性更有助于提高健壮性及其局限性。总体而言，与具有类似参数的传统CNN相比，CapsNets在对抗对手示例和仿射变换方面实现了更好的健壮性。对于CapsNet和CNN的更深版本，也得出了类似的结论。此外，我们的结果揭示了一个关键发现，即动态路由对提高CapsNet的健壮性没有太大帮助。事实上，主要的泛化贡献是由于通过胶囊进行的分层特征学习。



## **46. StratDef: Strategic Defense Against Adversarial Attacks in ML-based Malware Detection**

StratDef：基于ML的恶意软件检测中对抗攻击的战略防御 cs.LG

**SubmitDate**: 2023-04-24    [abs](http://arxiv.org/abs/2202.07568v6) [paper-pdf](http://arxiv.org/pdf/2202.07568v6)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The ML-based malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system based on a moving target defense approach. We overcome challenges related to the systematic construction, selection, and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker while minimizing critical aspects in the adversarial ML domain, like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, of the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。基于ML的恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了一种基于移动目标防御方法的战略防御系统StratDef。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，在现有的防御系统中，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **47. On Adversarial Robustness of Point Cloud Semantic Segmentation**

点云语义分割的对抗性研究 cs.CV

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2112.05871v4) [paper-pdf](http://arxiv.org/pdf/2112.05871v4)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting neural networks. However, the robustness of these complex models have not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications like autonomous driving, it is important to fill this knowledge gap, especially, how these models are affected under adversarial samples. As such, we present a comparative study of PCSS robustness. First, we formally define the attacker's objective under performance degradation and object hiding. Then, we develop new attack by whether to bound the norm. We evaluate different attack options on two datasets and three PCSS models. We found all the models are vulnerable and attacking point color is more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models.

摘要: 近年来，基于神经网络的三维点云语义分割(PCSS)的研究取得了显著的效果。然而，这些复杂模型的稳健性还没有得到系统的分析。鉴于PCSS已经被应用于许多安全关键应用，如自动驾驶，填补这一知识空白是很重要的，特别是这些模型在对抗性样本下是如何受到影响的。因此，我们提出了PCSS稳健性的比较研究。首先，在性能下降和对象隐藏的情况下，形式化地定义了攻击者的目标。然后，我们根据是否绑定规范来开发新的攻击。我们在两个数据集和三个PCSS模型上评估了不同的攻击方案。我们发现所有的模型都是易受攻击的，攻击点颜色更有效。通过这项研究，我们呼吁研究界关注开发新的方法来强化PCSS模型。



## **48. Evading DeepFake Detectors via Adversarial Statistical Consistency**

利用对抗性统计一致性规避DeepFake检测器 cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2304.11670v1) [paper-pdf](http://arxiv.org/pdf/2304.11670v1)

**Authors**: Yang Hou, Qing Guo, Yihao Huang, Xiaofei Xie, Lei Ma, Jianjun Zhao

**Abstract**: In recent years, as various realistic face forgery techniques known as DeepFake improves by leaps and bounds,more and more DeepFake detection techniques have been proposed. These methods typically rely on detecting statistical differences between natural (i.e., real) and DeepFakegenerated images in both spatial and frequency domains. In this work, we propose to explicitly minimize the statistical differences to evade state-of-the-art DeepFake detectors. To this end, we propose a statistical consistency attack (StatAttack) against DeepFake detectors, which contains two main parts. First, we select several statistical-sensitive natural degradations (i.e., exposure, blur, and noise) and add them to the fake images in an adversarial way. Second, we find that the statistical differences between natural and DeepFake images are positively associated with the distribution shifting between the two kinds of images, and we propose to use a distribution-aware loss to guide the optimization of different degradations. As a result, the feature distributions of generated adversarial examples is close to the natural images.Furthermore, we extend the StatAttack to a more powerful version, MStatAttack, where we extend the single-layer degradation to multi-layer degradations sequentially and use the loss to tune the combination weights jointly. Comprehensive experimental results on four spatial-based detectors and two frequency-based detectors with four datasets demonstrate the effectiveness of our proposed attack method in both white-box and black-box settings.

摘要: 近年来，随着各种真实感人脸伪造技术DeepFake的突飞猛进，越来越多的DeepFake检测技术被提出。这些方法通常依赖于在空间域和频域中检测自然(即，真实)和深度错误生成的图像之间的统计差异。在这项工作中，我们建议显式地最小化统计差异，以避开最新的DeepFake检测器。为此，我们提出了一种针对DeepFake检测器的统计一致性攻击(StatAttack)，主要包括两个部分。首先，我们选择了几种统计敏感的自然退化(即曝光、模糊和噪声)，并将它们以对抗性的方式添加到虚假图像中。其次，我们发现自然图像和DeepFake图像之间的统计差异与两种图像之间的分布漂移呈正相关，并提出使用分布感知损失来指导不同降质的优化。将StatAttack扩展到一个更强大的版本MStatAttack，将单层退化扩展到多层退化，并利用损失联合调整组合权值。在四个基于空间的检测器和两个基于频率的检测器上对四个数据集的综合实验结果表明，该攻击方法在白盒和黑盒环境下都是有效的。



## **49. Partial-Information, Longitudinal Cyber Attacks on LiDAR in Autonomous Vehicles**

自主车载激光雷达的部分信息、纵向网络攻击 cs.CR

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2303.03470v2) [paper-pdf](http://arxiv.org/pdf/2303.03470v2)

**Authors**: R. Spencer Hallyburton, Qingzhao Zhang, Z. Morley Mao, Miroslav Pajic

**Abstract**: What happens to an autonomous vehicle (AV) if its data are adversarially compromised? Prior security studies have addressed this question through mostly unrealistic threat models, with limited practical relevance, such as white-box adversarial learning or nanometer-scale laser aiming and spoofing. With growing evidence that cyber threats pose real, imminent danger to AVs and cyber-physical systems (CPS) in general, we present and evaluate a novel AV threat model: a cyber-level attacker capable of disrupting sensor data but lacking any situational awareness. We demonstrate that even though the attacker has minimal knowledge and only access to raw data from a single sensor (i.e., LiDAR), she can design several attacks that critically compromise perception and tracking in multi-sensor AVs. To mitigate vulnerabilities and advance secure architectures in AVs, we introduce two improvements for security-aware fusion: a probabilistic data-asymmetry monitor and a scalable track-to-track fusion of 3D LiDAR and monocular detections (T2T-3DLM); we demonstrate that the approaches significantly reduce attack effectiveness. To support objective safety and security evaluations in AVs, we release our security evaluation platform, AVsec, which is built on security-relevant metrics to benchmark AVs on gold-standard longitudinal AV datasets and AV simulators.

摘要: 如果自动驾驶汽车(AV)的数据被相反地泄露，会发生什么？以前的安全研究大多是通过不现实的威胁模型来解决这个问题，实际意义有限，例如白盒对抗性学习或纳米级激光瞄准和欺骗。随着越来越多的证据表明网络威胁对反病毒和网络物理系统(CP)构成真实、紧迫的威胁，我们提出并评估了一种新的反病毒威胁模型：能够破坏传感器数据但缺乏任何态势感知的网络级攻击者。我们证明，即使攻击者只有很少的知识并且只能访问来自单个传感器(即LiDAR)的原始数据，她也可以设计几种严重危害多传感器AVs感知和跟踪的攻击。为了缓解AVS中的漏洞和推进安全体系结构，我们引入了两个安全感知融合的改进：概率数据不对称监测器和可扩展的3D LiDAR和单目检测的航迹到航迹融合(T2T-3DLM)；我们证明这两种方法显著降低了攻击效率。为了支持AVS的客观安全和安保评估，我们发布了我们的安全评估平台AVSEC，该平台构建在与安全相关的指标基础上，以黄金标准的纵向AV数据集和AV模拟器为基准。



## **50. Disco Intelligent Reflecting Surfaces: Active Channel Aging for Fully-Passive Jamming Attacks**

DISCO智能反射面：用于完全无源干扰攻击的主动通道老化 eess.SP

**SubmitDate**: 2023-04-23    [abs](http://arxiv.org/abs/2302.00415v2) [paper-pdf](http://arxiv.org/pdf/2302.00415v2)

**Authors**: Huan Huang, Ying Zhang, Hongliang Zhang, Yi Cai, A. Lee Swindlehurst, Zhu Han

**Abstract**: Due to the open communications environment in wireless channels, wireless networks are vulnerable to jamming attacks. However, existing approaches for jamming rely on knowledge of the legitimate users' (LUs') channels, extra jamming power, or both. To raise concerns about the potential threats posed by illegitimate intelligent reflecting surfaces (IRSs), we propose an alternative method to launch jamming attacks on LUs without either LU channel state information (CSI) or jamming power. The proposed approach employs an adversarial IRS with random phase shifts, referred to as a "disco" IRS (DIRS), that acts like a "disco ball" to actively age the LUs' channels. Such active channel aging (ACA) interference can be used to launch jamming attacks on multi-user multiple-input single-output (MU-MISO) systems. The proposed DIRS-based fully-passive jammer (FPJ) can jam LUs with no additional jamming power or knowledge of the LU CSI, and it can not be mitigated by classical anti-jamming approaches. A theoretical analysis of the proposed DIRS-based FPJ that provides an evaluation of the DIRS-based jamming attacks is derived. Based on this detailed theoretical analysis, some unique properties of the proposed DIRS-based FPJ can be obtained. Furthermore, a design example of the proposed DIRS-based FPJ based on one-bit quantization of the IRS phases is demonstrated to be sufficient for implementing the jamming attack. In addition, numerical results are provided to show the effectiveness of the derived theoretical analysis and the jamming impact of the proposed DIRS-based FPJ.

摘要: 由于无线信道的开放通信环境，无线网络很容易受到干扰攻击。然而，现有的干扰方法依赖于对合法用户(LU)的信道、额外干扰功率或两者的了解。为了引起人们对非法智能反射面(IRS)潜在威胁的关注，我们提出了一种在没有LU信道状态信息(CSI)或干扰功率的情况下对LU发起干扰攻击的替代方法。所提出的方法采用具有随机相移的对抗性IRS，被称为“迪斯科”IRS(DIRS)，其作用类似于“迪斯科球”来主动老化LU的频道。这种主动信道老化(ACA)干扰可用于对多用户多输入单输出(MU-MISO)系统发起干扰攻击。所提出的基于DIRS的全无源干扰机在不增加干扰功率和不知道逻辑单元CSI的情况下，可以对逻辑单元进行干扰，而且经典的干扰方法不能对其进行抑制。对提出的基于DIRS的干扰攻击进行了理论分析，为基于DIRS的干扰攻击提供了评估。在详细的理论分析的基础上，可以得到基于DIRS的FPJ的一些独特的性质。最后，给出了一个基于IRS相位1比特量化的基于DIRS的FPJ的设计实例，证明了该设计对于实现干扰攻击是足够的。数值结果表明了理论分析的有效性和所提出的基于DIRS的干扰效果。



