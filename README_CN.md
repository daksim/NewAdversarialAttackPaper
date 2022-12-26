# Latest Adversarial Attack Papers
**update at 2022-12-26 13:09:42**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Generate synthetic samples from tabular data**

从表格数据生成合成样本 cs.LG

**SubmitDate**: 2022-12-23    [abs](http://arxiv.org/abs/2209.06113v2) [paper-pdf](http://arxiv.org/pdf/2209.06113v2)

**Authors**: David Banh, Alan Huang

**Abstract**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.

摘要: 从数据集生成新样本可以减少额外昂贵的操作、增加侵入性程序，并缓解隐私问题。当隐私受到关注时，这些在统计上稳健的新样本可以用作临时和中间替代。这种方法可以实现更好的数据共享实践，而不会出现与识别问题或作为对抗性攻击缺陷的偏见有关的问题。



## **2. Learned Systems Security**

学习的系统安全 cs.CR

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.10318v2) [paper-pdf](http://arxiv.org/pdf/2212.10318v2)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned systems under active development. To empirically validate the many vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against prominent learned-system instances. We show that the use of ML caused leakage of past queries in a database, enabled a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enabled index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is a universal threat against learned systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.

摘要: 学习系统在内部使用机器学习(ML)来提高性能。我们可以预计，这样的系统容易受到一些对抗性的ML攻击。通常，学习的组件在相互不信任的用户或进程之间共享，这与缓存等微体系结构资源非常相似，这可能会导致高度逼真的攻击者模型。然而，与对其他基于ML的系统的攻击相比，攻击者面临着一定程度的间接性，因为他们不能直接与学习的模型交互。此外，同一系统的学习版本和非学习版本的攻击面之间的差异通常是微妙的。这些因素混淆了合并ML带来的事实上的风险。我们分析了学习系统中潜在增加的攻击面的根本原因，并开发了一个框架来识别源于ML的使用的漏洞。我们将我们的框架应用于一系列正在积极开发的学习系统。为了经验性地验证我们的框架中出现的许多漏洞，我们选择其中3个漏洞，并针对突出的学习系统实例实施和评估利用漏洞。我们证明了ML的使用导致了数据库中过去查询的泄漏，启用了导致索引结构中指数级内存爆炸并在几秒钟内崩溃的中毒攻击，并使索引用户能够通过对他们自己的键的定时查询来窥探彼此的键分布。我们发现对抗性ML是对学习系统的普遍威胁，指出在我们对学习系统安全的理解中打开了研究缺口，并通过讨论缓解来结束，同时注意到数据泄漏是在其学习组件由多方共享的系统中固有的。



## **3. GAN-based Domain Inference Attack**

基于GAN的领域推理攻击 cs.LG

accepted by AAAI23

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11810v1) [paper-pdf](http://arxiv.org/pdf/2212.11810v1)

**Authors**: Yuechun Gu, Keke Chen

**Abstract**: Model-based attacks can infer training data information from deep neural network models. These attacks heavily depend on the attacker's knowledge of the application domain, e.g., using it to determine the auxiliary data for model-inversion attacks. However, attackers may not know what the model is used for in practice. We propose a generative adversarial network (GAN) based method to explore likely or similar domains of a target model -- the model domain inference (MDI) attack. For a given target (classification) model, we assume that the attacker knows nothing but the input and output formats and can use the model to derive the prediction for any input in the desired form. Our basic idea is to use the target model to affect a GAN training process for a candidate domain's dataset that is easy to obtain. We find that the target model may distract the training procedure less if the domain is more similar to the target domain. We then measure the distraction level with the distance between GAN-generated datasets, which can be used to rank candidate domains for the target model. Our experiments show that the auxiliary dataset from an MDI top-ranked domain can effectively boost the result of model-inversion attacks.

摘要: 基于模型的攻击可以从深度神经网络模型中推断出训练数据信息。这些攻击在很大程度上依赖于攻击者对应用程序域的了解，例如，使用它来确定模型反转攻击的辅助数据。然而，攻击者可能不知道该模型在实践中的用途。提出了一种基于产生式对抗网络(GAN)的目标模型相似或相似域挖掘方法--模型域推理(MDI)攻击。对于给定的目标(分类)模型，我们假设攻击者只知道输入和输出格式，并且可以使用该模型以所需的形式推导出对任何输入的预测。我们的基本思想是使用目标模型来影响候选域数据集的GAN训练过程，该数据集很容易获得。我们发现，如果领域与目标领域更相似，目标模型可能会减少训练过程的分心。然后，我们用GaN生成的数据集之间的距离来衡量干扰程度，这可以用来对目标模型的候选域进行排名。我们的实验表明，来自MDI顶级域的辅助数据集可以有效地提高模型反转攻击的结果。



## **4. Adversarial Machine Learning and Defense Game for NextG Signal Classification with Deep Learning**

基于深度学习的NextG信号分类对抗性机器学习与防御博弈 cs.NI

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11778v1) [paper-pdf](http://arxiv.org/pdf/2212.11778v1)

**Authors**: Yalin E. Sagduyu

**Abstract**: This paper presents a game-theoretic framework to study the interactions of attack and defense for deep learning-based NextG signal classification. NextG systems such as the one envisioned for a massive number of IoT devices can employ deep neural networks (DNNs) for various tasks such as user equipment identification, physical layer authentication, and detection of incumbent users (such as in the Citizens Broadband Radio Service (CBRS) band). By training another DNN as the surrogate model, an adversary can launch an inference (exploratory) attack to learn the behavior of the victim model, predict successful operation modes (e.g., channel access), and jam them. A defense mechanism can increase the adversary's uncertainty by introducing controlled errors in the victim model's decisions (i.e., poisoning the adversary's training data). This defense is effective against an attack but reduces the performance when there is no attack. The interactions between the defender and the adversary are formulated as a non-cooperative game, where the defender selects the probability of defending or the defense level itself (i.e., the ratio of falsified decisions) and the adversary selects the probability of attacking. The defender's objective is to maximize its reward (e.g., throughput or transmission success ratio), whereas the adversary's objective is to minimize this reward and its attack cost. The Nash equilibrium strategies are determined as operation modes such that no player can unilaterally improve its utility given the other's strategy is fixed. A fictitious play is formulated for each player to play the game repeatedly in response to the empirical frequency of the opponent's actions. The performance in Nash equilibrium is compared to the fixed attack and defense cases, and the resilience of NextG signal classification against attacks is quantified.

摘要: 针对基于深度学习的NextG信号分类问题，提出了一种研究攻防交互作用的博弈论框架。下一代系统(例如为大量物联网设备设想的系统)可以使用深度神经网络(DNN)来执行各种任务，例如用户设备识别、物理层身份验证和现有用户的检测(例如在公民宽带无线电服务(CBRS)频段中)。通过将另一个DNN训练为代理模型，敌手可以发起推理(探索性)攻击，以学习受害者模型的行为，预测成功的操作模式(例如，通道访问)，并对其进行干扰。防御机制可以通过在受害者模型的决策中引入受控错误(即，毒化对手的训练数据)来增加对手的不确定性。这种防御对攻击是有效的，但在没有攻击时会降低性能。防御者和对手之间的相互作用被描述为一个非合作博弈，其中防御者选择防御概率或防御水平本身(即决策被篡改的比率)，对手选择攻击概率。防御者的目标是最大化其回报(例如，吞吐量或传输成功率)，而对手的目标是最小化该回报及其攻击成本。纳什均衡策略被确定为操作模式，使得在对方策略固定的情况下，任何参与者都不能单方面提高其效用。根据对手行动的经验频率，为每个玩家制定了一个虚拟游戏，让每个玩家重复玩游戏。将Nash均衡下的性能与固定攻防情况下的性能进行了比较，量化了NextG信号分类对攻击的恢复能力。



## **5. Aliasing is a Driver of Adversarial Attacks**

混叠是敌意攻击的驱动因素 cs.CV

14 pages, 9 figures, 4 tables

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11760v1) [paper-pdf](http://arxiv.org/pdf/2212.11760v1)

**Authors**: Adrián Rodríguez-Muñoz, Antonio Torralba

**Abstract**: Aliasing is a highly important concept in signal processing, as careful consideration of resolution changes is essential in ensuring transmission and processing quality of audio, image, and video. Despite this, up until recently aliasing has received very little consideration in Deep Learning, with all common architectures carelessly sub-sampling without considering aliasing effects. In this work, we investigate the hypothesis that the existence of adversarial perturbations is due in part to aliasing in neural networks. Our ultimate goal is to increase robustness against adversarial attacks using explainable, non-trained, structural changes only, derived from aliasing first principles. Our contributions are the following. First, we establish a sufficient condition for no aliasing for general image transformations. Next, we study sources of aliasing in common neural network layers, and derive simple modifications from first principles to eliminate or reduce it. Lastly, our experimental results show a solid link between anti-aliasing and adversarial attacks. Simply reducing aliasing already results in more robust classifiers, and combining anti-aliasing with robust training out-performs solo robust training on $L_2$ attacks with none or minimal losses in performance on $L_{\infty}$ attacks.

摘要: 混叠在信号处理中是一个非常重要的概念，因为仔细考虑分辨率变化对于确保音频、图像和视频的传输和处理质量至关重要。尽管如此，直到最近，混叠在深度学习中还很少得到考虑，所有常见的体系结构都不小心进行子采样，而不考虑混叠效果。在这项工作中，我们研究了这样的假设，即对抗性扰动的存在部分是由于神经网络中的混叠。我们的最终目标是使用源自混叠优先原则的可解释的、未经训练的、仅结构变化来增强对抗对手攻击的稳健性。我们的贡献如下。首先，我们建立了一般图像变换无混叠的一个充分条件。接下来，我们研究了常见神经网络层中混叠的来源，并根据基本原理进行了简单的修改以消除或减少混叠。最后，我们的实验结果显示了反走样和对抗性攻击之间的紧密联系。简单地减少混叠已经产生了更健壮的分类器，并将抗混叠与健壮训练相结合，在$L_2$攻击上的性能优于单独的健壮训练，而在$L_{\infty}$攻击上的性能损失为零或最小。



## **6. A Theoretical Study of The Effects of Adversarial Attacks on Sparse Regression**

对抗性攻击对稀疏回归影响的理论研究 cs.LG

first version

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2212.11209v2) [paper-pdf](http://arxiv.org/pdf/2212.11209v2)

**Authors**: Deepak Maurya, Jean Honorio

**Abstract**: This paper analyzes $\ell_1$ regularized linear regression under the challenging scenario of having only adversarially corrupted data for training. We use the primal-dual witness paradigm to provide provable performance guarantees for the support of the estimated regression parameter vector to match the actual parameter. Our theoretical analysis shows the counter-intuitive result that an adversary can influence sample complexity by corrupting the irrelevant features, i.e., those corresponding to zero coefficients of the regression parameter vector, which, consequently, do not affect the dependent variable. As any adversarially robust algorithm has its limitations, our theoretical analysis identifies the regimes under which the learning algorithm and adversary can dominate over each other. It helps us to analyze these fundamental limits and address critical scientific questions of which parameters (like mutual incoherence, the maximum and minimum eigenvalue of the covariance matrix, and the budget of adversarial perturbation) play a role in the high or low probability of success of the LASSO algorithm. Also, the derived sample complexity is logarithmic with respect to the size of the regression parameter vector, and our theoretical claims are validated by empirical analysis on synthetic and real-world datasets.

摘要: 本文分析了训练数据只有反向破坏的挑战情形下的正则化线性回归问题。我们使用原始-对偶见证范式来提供可证明的性能保证，以支持估计的回归参数向量与实际参数匹配。我们的理论分析显示了与直觉相反的结果，即对手可以通过破坏不相关的特征来影响样本的复杂性，即对应于回归参数向量的零系数的特征，从而不影响因变量。由于任何对抗健壮算法都有其局限性，我们的理论分析确定了学习算法和对手可以相互支配的制度。它帮助我们分析这些基本极限，并解决关键的科学问题，其中参数(如相互不一致、协方差矩阵的最大和最小特征值以及对抗性扰动的预算)对LASSO算法的成功概率的高低起作用。此外，导出的样本复杂度是关于回归参数向量大小的对数，并且通过对合成和真实世界数据集的实证分析验证了我们的理论主张。



## **7. Suppress with a Patch: Revisiting Universal Adversarial Patch Attacks against Object Detection**

用补丁压制：再论针对目标检测的通用对抗性补丁攻击 cs.CV

Accepted for publication at ICECCME 2022

**SubmitDate**: 2022-12-22    [abs](http://arxiv.org/abs/2209.13353v2) [paper-pdf](http://arxiv.org/pdf/2209.13353v2)

**Authors**: Svetlana Pavlitskaya, Jonas Hendl, Sebastian Kleim, Leopold Müller, Fabian Wylczoch, J. Marius Zöllner

**Abstract**: Adversarial patch-based attacks aim to fool a neural network with an intentionally generated noise, which is concentrated in a particular region of an input image. In this work, we perform an in-depth analysis of different patch generation parameters, including initialization, patch size, and especially positioning a patch in an image during training. We focus on the object vanishing attack and run experiments with YOLOv3 as a model under attack in a white-box setting and use images from the COCO dataset. Our experiments have shown, that inserting a patch inside a window of increasing size during training leads to a significant increase in attack strength compared to a fixed position. The best results were obtained when a patch was positioned randomly during training, while patch position additionally varied within a batch.

摘要: 基于补丁的对抗性攻击旨在通过故意生成的噪声来愚弄神经网络，这些噪声集中在输入图像的特定区域。在这项工作中，我们对不同的块生成参数进行了深入的分析，包括初始化、块大小，特别是在训练过程中在图像中定位块。我们重点研究了目标消失攻击，并以YOLOv3为模型在白盒环境下进行了实验，并使用了COCO数据集中的图像。我们的实验表明，与固定位置相比，在训练期间在不断增大的窗口内插入补丁可以显著增加攻击强度。当补丁在训练过程中随机定位时，当补丁的位置在批次内另外变化时，效果最好。



## **8. Vulnerabilities of Deep Learning-Driven Semantic Communications to Backdoor (Trojan) Attacks**

深度学习驱动的语义通信对后门(木马)攻击的脆弱性 cs.CR

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.11205v1) [paper-pdf](http://arxiv.org/pdf/2212.11205v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Sennur Ulukus, Aylin Yener

**Abstract**: This paper highlights vulnerabilities of deep learning-driven semantic communications to backdoor (Trojan) attacks. Semantic communications aims to convey a desired meaning while transferring information from a transmitter to its receiver. An encoder-decoder pair that is represented by two deep neural networks (DNNs) as part of an autoencoder is trained to reconstruct signals such as images at the receiver by transmitting latent features of small size over a limited number of channel uses. In the meantime, another DNN of a semantic task classifier at the receiver is jointly trained with the autoencoder to check the meaning conveyed to the receiver. The complex decision space of the DNNs makes semantic communications susceptible to adversarial manipulations. In a backdoor (Trojan) attack, the adversary adds triggers to a small portion of training samples and changes the label to a target label. When the transfer of images is considered, the triggers can be added to the images or equivalently to the corresponding transmitted or received signals. In test time, the adversary activates these triggers by providing poisoned samples as input to the encoder (or decoder) of semantic communications. The backdoor attack can effectively change the semantic information transferred for the poisoned input samples to a target meaning. As the performance of semantic communications improves with the signal-to-noise ratio and the number of channel uses, the success of the backdoor attack increases as well. Also, increasing the Trojan ratio in training data makes the attack more successful. In the meantime, the effect of this attack on the unpoisoned input samples remains limited. Overall, this paper shows that the backdoor attack poses a serious threat to semantic communications and presents novel design guidelines to preserve the meaning of transferred information in the presence of backdoor attacks.

摘要: 强调了深度学习驱动的语义通信对后门(特洛伊木马)攻击的脆弱性。语义通信的目的是在将信息从发送者传递到接收者的同时传达所需的意义。作为自动编码器的一部分，由两个深度神经网络(DNN)表示的编码器-解码器对被训练以通过在有限数目的信道使用上传输小尺寸的潜在特征来重建接收器处的信号，例如图像。同时，接收方的语义任务分类器的另一个DNN与自动编码器联合训练，以检查传达给接收方的意义。DNN的复杂决策空间使得语义通信容易受到敌意操纵。在后门(特洛伊木马)攻击中，对手将触发器添加到一小部分训练样本中，并将标签更改为目标标签。当考虑图像的传输时，可以将触发添加到图像，或者等效地添加到对应的发送或接收的信号。在测试时间，对手通过将有毒样本作为输入提供给语义通信的编码器(或解码器)来激活这些触发器。后门攻击可以有效地将有毒输入样本的语义信息转换为目标含义。随着语义通信的性能随信噪比和信道使用次数的增加而提高，后门攻击的成功率也会增加。此外，增加训练数据中的特洛伊木马比率会使攻击更成功。与此同时，这种攻击对未中毒的输入样本的影响仍然有限。总之，本文表明后门攻击对语义通信构成了严重威胁，并提出了新的设计指南，以在存在后门攻击的情况下保留传输的信息的含义。



## **9. Revisiting Residual Networks for Adversarial Robustness: An Architectural Perspective**

从体系结构的角度重新审视残差网络的对抗稳健性 cs.CV

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.11005v1) [paper-pdf](http://arxiv.org/pdf/2212.11005v1)

**Authors**: Shihua Huang, Zhichao Lu, Kalyanmoy Deb, Vishnu Naresh Boddeti

**Abstract**: Efforts to improve the adversarial robustness of convolutional neural networks have primarily focused on developing more effective adversarial training methods. In contrast, little attention was devoted to analyzing the role of architectural elements (such as topology, depth, and width) on adversarial robustness. This paper seeks to bridge this gap and present a holistic study on the impact of architectural design on adversarial robustness. We focus on residual networks and consider architecture design at the block level, i.e., topology, kernel size, activation, and normalization, as well as at the network scaling level, i.e., depth and width of each block in the network. In both cases, we first derive insights through systematic ablative experiments. Then we design a robust residual block, dubbed RobustResBlock, and a compound scaling rule, dubbed RobustScaling, to distribute depth and width at the desired FLOP count. Finally, we combine RobustResBlock and RobustScaling and present a portfolio of adversarially robust residual networks, RobustResNets, spanning a broad spectrum of model capacities. Experimental validation across multiple datasets and adversarial attacks demonstrate that RobustResNets consistently outperform both the standard WRNs and other existing robust architectures, achieving state-of-the-art AutoAttack robust accuracy of 61.1% without additional data and 63.7% with 500K external data while being $2\times$ more compact in terms of parameters. Code is available at \url{ https://github.com/zhichao-lu/robust-residual-network}

摘要: 提高卷积神经网络的对抗稳健性的努力主要集中在开发更有效的对抗训练方法上。相反，很少有人关注分析体系结构元素(如拓扑、深度和宽度)对对手健壮性的作用。本文试图弥合这一差距，并对建筑设计对对手健壮性的影响进行整体研究。我们关注剩余网络，并考虑块级别的体系结构设计，即拓扑、内核大小、激活和归一化，以及网络扩展级别，即网络中每个块的深度和宽度。在这两种情况下，我们首先通过系统的消融实验获得洞察力。然后，我们设计了一个健壮的残差块(称为RobustResBlock)和一个复合缩放规则(称为RobustScaling)，以在所需的触发器计数上分布深度和宽度。最后，我们将RobustResBlock和RobustScaling结合起来，提出了一个具有相反健壮性的残差网络组合RobustResNets，它跨越了广泛的模型容量。在多个数据集和敌意攻击上的实验验证表明，RobustResNets的性能一直优于标准WRNS和其他现有的健壮体系结构，在没有额外数据的情况下获得了61.1%的健壮准确率，在没有额外数据的情况下获得了63.7%的健壮性，同时在参数方面更紧凑了2倍。代码位于\url{https://github.com/zhichao-lu/robust-residual-network}



## **10. SoK: Let The Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning**

索克：让隐私游戏开始吧！机器学习中数据推理隐私的统一处理 cs.LG

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.10986v1) [paper-pdf](http://arxiv.org/pdf/2212.10986v1)

**Authors**: Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin

**Abstract**: Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning.

摘要: 在生产中部署机器学习模型可能会允许攻击者推断有关训练数据的敏感信息。有大量的文献分析了不同类型的推理风险，从成员关系推理到重构攻击。受游戏(即概率实验)在密码学中研究安全属性的成功启发，一些作者使用类似的基于游戏的风格描述了机器学习中的隐私推理风险。然而，对手的能力和目标往往在不同的演示文稿中以微妙的不同方式表达，这使得很难联系和撰写结果。在本文中，我们提出了一个基于博弈的框架来系统化机器学习中隐私推理风险的知识体系。



## **11. Hidden Poison: Machine Unlearning Enables Camouflaged Poisoning Attacks**

隐藏毒药：机器遗忘使伪装的毒药攻击成为可能 cs.LG

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.10717v1) [paper-pdf](http://arxiv.org/pdf/2212.10717v1)

**Authors**: Jimmy Z. Di, Jack Douglas, Jayadev Acharya, Gautam Kamath, Ayush Sekhari

**Abstract**: We introduce camouflaged data poisoning attacks, a new attack vector that arises in the context of machine unlearning and other settings when model retraining may be induced. An adversary first adds a few carefully crafted points to the training dataset such that the impact on the model's predictions is minimal. The adversary subsequently triggers a request to remove a subset of the introduced points at which point the attack is unleashed and the model's predictions are negatively affected. In particular, we consider clean-label targeted attacks (in which the goal is to cause the model to misclassify a specific test point) on datasets including CIFAR-10, Imagenette, and Imagewoof. This attack is realized by constructing camouflage datapoints that mask the effect of a poisoned dataset.

摘要: 我们引入了伪装数据中毒攻击，这是在机器遗忘和其他可能诱导模型重新训练的背景下出现的一种新的攻击矢量。对手首先向训练数据集添加几个精心设计的点，以便对模型预测的影响最小。敌手随后触发一个请求，要求删除引入的点的子集，在这一点上，攻击被释放，模型的预测受到负面影响。特别是，我们考虑了针对包括CIFAR-10、Imagenette和Imagewoof在内的数据集的干净标签定向攻击(目标是导致模型对特定测试点进行错误分类)。这种攻击是通过构造伪装数据点来实现的，这些数据点掩盖了有毒数据集的影响。



## **12. The Quantum Chernoff Divergence in Advantage Distillation for QKD and DIQKD**

QKD和DIQKD优势蒸馏中的量子Chernoff发散 quant-ph

Minor typo fixes

**SubmitDate**: 2022-12-21    [abs](http://arxiv.org/abs/2212.06975v2) [paper-pdf](http://arxiv.org/pdf/2212.06975v2)

**Authors**: Mikka Stasiuk, Norbert Lütkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible.

摘要: 独立于设备的量子密钥分发(DIQKD)旨在通过提供一种在适度的安全假设下提取密钥的方法来减少对量子设备中缺陷的恶意利用。优势蒸馏，一种纠错的双向通信过程，已被证明在提高设备相关和设备无关的量子密钥分发中的噪声容限方面都是有效的。以前，针对一种称为重码协议的优势提取协议，基于涉及协议中某些状态之间的保真度的安全条件，开发了针对IID集体攻击的独立于设备的安全证明。然而，在充分和必要的安全条件之间存在差距，这阻碍了基于保真度的紧噪声容限的计算。我们通过提出另一种证明结构来缩小这一差距，该结构用量子切尔诺夫发散取代了保真度，量子切尔诺夫发散是对称假设检验中出现的一种可区分性衡量标准。在IID集体攻击模型下，根据量子Chernoff散度，我们得到了重码协议安全的充要条件(直到关于后者的一个自然猜想)，从而表明这是该协议的相关关注量。此外，利用这一安全条件，我们在DIQKD的噪声容忍门限上得到了一些改进。我们的结果为量子信息论中的一个基本问题提供了洞察力，这个问题涉及到DIQKD可能的情况。



## **13. Is Semantic Communications Secure? A Tale of Multi-Domain Adversarial Attacks**

语义通信安全吗？多域对抗性攻击的故事 cs.CR

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10438v1) [paper-pdf](http://arxiv.org/pdf/2212.10438v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Sennur Ulukus, Aylin Yener

**Abstract**: Semantic communications seeks to transfer information from a source while conveying a desired meaning to its destination. We model the transmitter-receiver functionalities as an autoencoder followed by a task classifier that evaluates the meaning of the information conveyed to the receiver. The autoencoder consists of an encoder at the transmitter to jointly model source coding, channel coding, and modulation, and a decoder at the receiver to jointly model demodulation, channel decoding and source decoding. By augmenting the reconstruction loss with a semantic loss, the two deep neural networks (DNNs) of this encoder-decoder pair are interactively trained with the DNN of the semantic task classifier. This approach effectively captures the latent feature space and reliably transfers compressed feature vectors with a small number of channel uses while keeping the semantic loss low. We identify the multi-domain security vulnerabilities of using the DNNs for semantic communications. Based on adversarial machine learning, we introduce test-time (targeted and non-targeted) adversarial attacks on the DNNs by manipulating their inputs at different stages of semantic communications. As a computer vision attack, small perturbations are injected to the images at the input of the transmitter's encoder. As a wireless attack, small perturbations signals are transmitted to interfere with the input of the receiver's decoder. By launching these stealth attacks individually or more effectively in a combined form as a multi-domain attack, we show that it is possible to change the semantics of the transferred information even when the reconstruction loss remains low. These multi-domain adversarial attacks pose as a serious threat to the semantics of information transfer (with larger impact than conventional jamming) and raise the need of defense methods for the safe adoption of semantic communications.

摘要: 语义通信寻求从源传递信息，同时向其目的地传达所需的含义。我们将发送器-接收器功能建模为一个自动编码器，后面跟着一个任务分类器，该任务分类器评估传递给接收器的信息的含义。该自动编码器由发射机的编码器和接收机的解码器组成，前者用于联合模拟信源编码、信道编码和调制，后者用于联合模拟解调、信道译码和信源译码。通过用语义损失来增加重构损失，该编解码对中的两个深层神经网络(DNN)与语义任务分类器的DNN交互训练。该方法有效地捕捉了潜在的特征空间，在保持较低的语义损失的同时，以较少的信道使用量可靠地传输压缩的特征向量。我们发现了使用DNN进行语义通信的多域安全漏洞。基于对抗性机器学习，通过在语义通信的不同阶段对DNN的输入进行操纵，我们引入了测试时间(目标和非目标)对抗性攻击。作为一种计算机视觉攻击，微小的扰动被注入到发射机编码器输入的图像中。作为一种无线攻击，传输微小的扰动信号来干扰接收器解码器的输入。通过单独或以多域攻击的组合形式更有效地发起这些隐形攻击，我们证明了即使在重建损失保持较低的情况下，也可以改变所传输信息的语义。这些多领域的对抗性攻击对信息传输的语义构成了严重的威胁(比传统的干扰影响更大)，并提出了安全采用语义通信的防御方法的需求。



## **14. In and Out-of-Domain Text Adversarial Robustness via Label Smoothing**

通过标签平滑实现域内和域外文本对抗健壮性 cs.CL

Preprint. Under Submission

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10258v1) [paper-pdf](http://arxiv.org/pdf/2212.10258v1)

**Authors**: Yahan Yang, Soham Dan, Dan Roth, Insup Lee

**Abstract**: Recently it has been shown that state-of-the-art NLP models are vulnerable to adversarial attacks, where the predictions of a model can be drastically altered by slight modifications to the input (such as synonym substitutions). While several defense techniques have been proposed, and adapted, to the discrete nature of text adversarial attacks, the benefits of general-purpose regularization methods such as label smoothing for language models, have not been studied. In this paper, we study the adversarial robustness provided by various label smoothing strategies in foundational models for diverse NLP tasks in both in-domain and out-of-domain settings. Our experiments show that label smoothing significantly improves adversarial robustness in pre-trained models like BERT, against various popular attacks. We also analyze the relationship between prediction confidence and robustness, showing that label smoothing reduces over-confident errors on adversarial examples.

摘要: 最近有研究表明，最新的自然语言处理模型容易受到敌意攻击，模型的预测可以通过对输入的轻微修改(如同义词替换)来显著改变。虽然已经提出了几种防御技术，并对其进行了调整，以适应文本对抗性攻击的离散性质，但还没有研究通用正则化方法的好处，例如语言模型的标签平滑。在本文中，我们研究了不同的标签平滑策略在基本模型中对不同的NLP任务在域内和域外环境下提供的对抗健壮性。我们的实验表明，在像BERT这样的预先训练的模型中，标签平滑显著提高了对抗各种流行攻击的健壮性。我们还分析了预测置信度和稳健性之间的关系，表明标签平滑减少了对抗性例子中的过度自信错误。



## **15. A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks**

3D目标检测器抗敌意攻击能力的综合研究与比较 cs.CV

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10230v1) [paper-pdf](http://arxiv.org/pdf/2212.10230v1)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Deep learning-based 3D object detectors have made significant progress in recent years and have been deployed in a wide range of applications. It is crucial to understand the robustness of detectors against adversarial attacks when employing detectors in security-critical applications. In this paper, we make the first attempt to conduct a thorough evaluation and analysis of the robustness of 3D detectors under adversarial attacks. Specifically, we first extend three kinds of adversarial attacks to the 3D object detection task to benchmark the robustness of state-of-the-art 3D object detectors against attacks on KITTI and Waymo datasets, subsequently followed by the analysis of the relationship between robustness and properties of detectors. Then, we explore the transferability of cross-model, cross-task, and cross-data attacks. We finally conduct comprehensive experiments of defense for 3D detectors, demonstrating that simple transformations like flipping are of little help in improving robustness when the strategy of transformation imposed on input point cloud data is exposed to attackers. Our findings will facilitate investigations in understanding and defending the adversarial attacks against 3D object detectors to advance this field.

摘要: 基于深度学习的三维物体检测器近年来取得了长足的进步，并得到了广泛的应用。在安全关键应用中使用检测器时，了解检测器对敌意攻击的稳健性至关重要。本文首次尝试对3D检测器在敌方攻击下的健壮性进行了深入的评估和分析。具体地说，我们首先将三种对抗性攻击扩展到3D对象检测任务中，以测试最新的3D对象检测器对Kitti和Waymo数据集的攻击的健壮性，随后分析了检测器的健壮性与性能之间的关系。然后，我们探讨了跨模型、跨任务和跨数据攻击的可转移性。最后，我们对3D探测器进行了全面的防御实验，证明了当输入点云数据的变换策略暴露给攻击者时，像翻转这样的简单变换对提高稳健性几乎没有帮助。我们的研究结果将有助于了解和防御针对3D对象探测器的对抗性攻击，从而推动这一领域的发展。



## **16. Multi-head Uncertainty Inference for Adversarial Attack Detection**

用于敌意攻击检测的多头不确定性推理 cs.LG

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10006v1) [paper-pdf](http://arxiv.org/pdf/2212.10006v1)

**Authors**: Yuqi Yang, Songyun Yang, Jiyang Xie. Zhongwei Si, Kai Guo, Ke Zhang, Kongming Liang

**Abstract**: Deep neural networks (DNNs) are sensitive and susceptible to tiny perturbation by adversarial attacks which causes erroneous predictions. Various methods, including adversarial defense and uncertainty inference (UI), have been developed in recent years to overcome the adversarial attacks. In this paper, we propose a multi-head uncertainty inference (MH-UI) framework for detecting adversarial attack examples. We adopt a multi-head architecture with multiple prediction heads (i.e., classifiers) to obtain predictions from different depths in the DNNs and introduce shallow information for the UI. Using independent heads at different depths, the normalized predictions are assumed to follow the same Dirichlet distribution, and we estimate distribution parameter of it by moment matching. Cognitive uncertainty brought by the adversarial attacks will be reflected and amplified on the distribution. Experimental results show that the proposed MH-UI framework can outperform all the referred UI methods in the adversarial attack detection task with different settings.

摘要: 深度神经网络(DNN)对敌意攻击的微小扰动非常敏感，容易造成错误的预测。为了克服对抗性攻击，近年来发展了各种方法，包括对抗性防御和不确定性推理(UI)。本文提出了一种多头不确定性推理(MH-UI)框架，用于检测对抗性攻击实例。我们采用具有多个预测头(即分类器)的多头结构来获取DNN中不同深度的预测，并为用户界面引入浅层信息。利用不同深度的独立头部，假设归一化预报服从相同的Dirichlet分布，并通过矩匹配估计其分布参数。对抗性攻击带来的认知不确定性将在分布上得到反映和放大。实验结果表明，在不同设置下，MH-UI框架在对抗性攻击检测任务中的性能优于已有的所有UI方法。



## **17. Defending Against Poisoning Attacks in Open-Domain Question Answering**

开放领域答疑中的防中毒攻击 cs.CL

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.10002v1) [paper-pdf](http://arxiv.org/pdf/2212.10002v1)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the input contexts can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we introduce a new method that uses query augmentation to search for a diverse set of retrieved passages that could answer the original question. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call Confidence from Answer Redundancy, e.g. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks and provide gains of 5-20% exact match across varying levels of data poisoning.

摘要: 最近在开放领域问答(ODQA)方面的研究表明，输入上下文的对抗性毒化会导致产生式系统的准确率大幅下降。然而，几乎没有工作提出了防御这些攻击的方法。为此，我们引入了一种新的方法，使用查询增强来搜索可以回答原始问题的不同检索段落集。我们通过设计一种新的置信度方法将这些新的段落集成到模型中，将预测的答案与其在检索到的上下文中的外观进行比较(例如CAR，我们称之为答案冗余的置信度)。这些方法结合在一起，提供了一种简单但有效的方法来防御中毒攻击，并在不同级别的数据中毒中提供5%-20%的精确匹配。



## **18. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

对训练数据进行重复数据删除可降低语言模型中的隐私风险 cs.CR

ICML 2022 Camera Ready Version

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2202.06539v3) [paper-pdf](http://arxiv.org/pdf/2202.06539v3)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstract**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.

摘要: 过去的工作表明，大型语言模型容易受到隐私攻击，即攻击者从训练的模型生成序列，并检测从训练集中记忆的序列。在这项工作中，我们表明，这些攻击的成功在很大程度上是由于常用的Web抓取训练集的复制。我们首先证明了语言模型重新生成训练序列的速度与训练集中序列的计数呈超线性关系。例如，在训练数据中出现10次的序列平均产生的频率是只出现一次的序列的~1000倍。接下来，我们证明了现有的检测记忆序列的方法在非重复训练序列上具有近乎随机的准确率。最后，我们发现，在应用方法对训练数据进行去重之后，语言模型对这些类型的隐私攻击的安全性要高得多。综上所述，我们的结果促使人们更加关注隐私敏感应用程序中的重复数据删除，并重新评估现有隐私攻击的实用性。



## **19. Towards Robustness of Text-to-SQL Models Against Natural and Realistic Adversarial Table Perturbation**

Text-to-SQL模型对自然和现实对抗性表格扰动的稳健性研究 cs.CL

Accepted by ACL 2022 (Oral)

**SubmitDate**: 2022-12-20    [abs](http://arxiv.org/abs/2212.09994v1) [paper-pdf](http://arxiv.org/pdf/2212.09994v1)

**Authors**: Xinyu Pi, Bing Wang, Yan Gao, Jiaqi Guo, Zhoujun Li, Jian-Guang Lou

**Abstract**: The robustness of Text-to-SQL parsers against adversarial perturbations plays a crucial role in delivering highly reliable applications. Previous studies along this line primarily focused on perturbations in the natural language question side, neglecting the variability of tables. Motivated by this, we propose the Adversarial Table Perturbation (ATP) as a new attacking paradigm to measure the robustness of Text-to-SQL models. Following this proposition, we curate ADVETA, the first robustness evaluation benchmark featuring natural and realistic ATPs. All tested state-of-the-art models experience dramatic performance drops on ADVETA, revealing models' vulnerability in real-world practices. To defend against ATP, we build a systematic adversarial training example generation framework tailored for better contextualization of tabular data. Experiments show that our approach not only brings the best robustness improvement against table-side perturbations but also substantially empowers models against NL-side perturbations. We release our benchmark and code at: https://github.com/microsoft/ContextualSP.

摘要: 文本到SQL解析器对敌意干扰的健壮性在提供高度可靠的应用程序方面起着至关重要的作用。以前的研究主要集中在自然语言问题方面的扰动，而忽略了表格的可变性。基于此，我们提出了对抗性表格扰动(ATP)作为一种新的攻击范式来衡量Text-to-SQL模型的健壮性。根据这一命题，我们策划了ADVETA，这是第一个以自然和现实的ATP为特征的健壮性评估基准。所有经过测试的最先进模型在ADVETA上的性能都出现了戏剧性的下降，揭示了模型在现实世界实践中的脆弱性。为了防御ATP，我们构建了一个系统的对抗性训练实例生成框架，该框架为表格数据更好地上下文提供了量身定制的框架。实验表明，我们的方法不仅对表格端的扰动带来了最好的稳健性改进，而且大大增强了模型对NL端的扰动的能力。我们在以下网址发布基准测试和代码：https://github.com/microsoft/ContextualSP.



## **20. Task-Oriented Communications for NextG: End-to-End Deep Learning and AI Security Aspects**

面向NextG的面向任务的通信：端到端深度学习和AI安全方面 cs.NI

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09668v1) [paper-pdf](http://arxiv.org/pdf/2212.09668v1)

**Authors**: Yalin E. Sagduyu, Sennur Ulukus, Aylin Yener

**Abstract**: Communications systems to date are primarily designed with the goal of reliable (error-free) transfer of digital sequences (bits). Next generation (NextG) communication systems are beginning to explore shifting this design paradigm of reliably decoding bits to reliably executing a given task. Task-oriented communications system design is likely to find impactful applications, for example, considering the relative importance of messages. In this paper, a wireless signal classification is considered as the task to be performed in the NextG Radio Access Network (RAN) for signal intelligence and spectrum awareness applications such as user equipment (UE) identification and authentication, and incumbent signal detection for spectrum co-existence. For that purpose, edge devices collect wireless signals and communicate with the NextG base station (gNodeB) that needs to know the signal class. Edge devices may not have sufficient processing power and may not be trusted to perform the signal classification task, whereas the transfer of the captured signals from the edge devices to the gNodeB may not be efficient or even feasible subject to stringent delay, rate, and energy restrictions. We present a task-oriented communications approach, where all the transmitter, receiver and classifier functionalities are jointly trained as two deep neural networks (DNNs), one for the edge device and another for the gNodeB. We show that this approach achieves better accuracy with smaller DNNs compared to the baselines that treat communications and signal classification as two separate tasks. Finally, we discuss how adversarial machine learning poses a major security threat for the use of DNNs for task-oriented communications. We demonstrate the major performance loss under backdoor (Trojan) attacks and adversarial (evasion) attacks that target the training and test processes of task-oriented communications.

摘要: 迄今为止，通信系统的主要设计目标是可靠(无差错)地传输数字序列(比特)。下一代(NextG)通信系统正开始探索将这种可靠解码比特的设计范例转变为可靠地执行给定任务。例如，考虑到消息的相对重要性，面向任务的通信系统设计可能会找到有影响力的应用程序。本文将无线信号分类作为下一代无线接入网(RAN)的任务，用于信号智能和频谱感知应用，如用户设备(UE)识别和鉴权，以及用于频谱共存的现有信号检测。为此，边缘设备收集无线信号并与需要知道信号类别的NextG基站(GNodeB)通信。边缘设备可能没有足够的处理能力，并且可能不被信任来执行信号分类任务，而将捕获的信号从边缘设备传输到gNodeB可能不是有效的，甚至可能不可行，受到严格的延迟、速率和能量限制。我们提出了一种面向任务的通信方法，其中所有的发送器、接收器和分类器功能被联合训练为两个深度神经网络(DNN)，一个用于边缘设备，另一个用于gNodeB。我们表明，与将通信和信号分类视为两个独立任务的基线相比，该方法在较小的DNN上获得了更好的准确性。最后，我们讨论了对抗性机器学习如何对使用DNN进行面向任务的通信构成主要的安全威胁。我们展示了在针对面向任务的通信的训练和测试过程的后门(特洛伊木马)攻击和对手(规避)攻击下的主要性能损失。



## **21. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

针对Windows PE恶意软件检测的对抗性攻击：现状综述 cs.CR

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2112.12310v3) [paper-pdf](http://arxiv.org/pdf/2112.12310v3)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Yaguan Qian, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstract**: Malware has been one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against ever-increasing and ever-evolving malware, tremendous efforts have been made to propose a variety of malware detection that attempt to effectively and efficiently detect malware so as to mitigate possible damages as early as possible. Recent studies have shown that, on the one hand, existing ML and DL techniques enable superior solutions in detecting newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of Windows PE malware. Then, we conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of Windows PE malware detection. Finally, we conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities. In addition, a curated resource list of adversarial attacks and defenses for Windows PE malware detection is also available at https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.

摘要: 恶意软件一直是计算机面临的最具破坏性的威胁之一，这些威胁跨越多个操作系统和各种文件格式。为了防御不断增长和不断演变的恶意软件，人们做出了巨大的努力，提出了各种恶意软件检测方法，试图有效和高效地检测恶意软件，以便尽早减轻可能的损害。最近的研究表明，一方面，现有的ML和DL技术能够在检测新出现的和以前未见过的恶意软件方面提供更好的解决方案。然而，另一方面，ML和DL模型天生就容易受到对抗性例子形式的对抗性攻击。本文以Windows操作系统家族中具有可移植可执行文件(PE)文件格式的恶意软件，即Windows PE恶意软件为典型案例，研究这种对抗性环境下的对抗性攻击方法。具体地说，我们首先概述了基于ML/DL的Windows PE恶意软件检测的一般学习框架，然后重点介绍了在Windows PE恶意软件环境中执行对抗性攻击的三个独特挑战。然后，我们对针对PE恶意软件检测的对抗性攻击进行了全面系统的回顾，并对相应的防御措施进行了分类，以增加Windows PE恶意软件检测的健壮性。最后，我们首先介绍了Windows PE恶意软件检测中除了对抗性攻击之外的其他相关攻击，并对未来的研究方向和机会进行了展望。此外，https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.上还提供了针对Windows PE恶意软件检测的对抗性攻击和防御的精选资源列表



## **22. Adversarial Sticker: A Stealthy Attack Method in the Physical World**

对抗性贴纸：物理世界中的一种隐形攻击方法 cs.CV

accepted by TPAMI 2022

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2104.06728v2) [paper-pdf](http://arxiv.org/pdf/2104.06728v2)

**Authors**: Xingxing Wei, Ying Guo, Jie Yu

**Abstract**: To assess the vulnerability of deep learning in the physical world, recent works introduce adversarial patches and apply them on different tasks. In this paper, we propose another kind of adversarial patch: the Meaningful Adversarial Sticker, a physically feasible and stealthy attack method by using real stickers existing in our life. Unlike the previous adversarial patches by designing perturbations, our method manipulates the sticker's pasting position and rotation angle on the objects to perform physical attacks. Because the position and rotation angle are less affected by the printing loss and color distortion, adversarial stickers can keep good attacking performance in the physical world. Besides, to make adversarial stickers more practical in real scenes, we conduct attacks in the black-box setting with the limited information rather than the white-box setting with all the details of threat models. To effectively solve for the sticker's parameters, we design the Region based Heuristic Differential Evolution Algorithm, which utilizes the new-found regional aggregation of effective solutions and the adaptive adjustment strategy of the evaluation criteria. Our method is comprehensively verified in the face recognition and then extended to the image retrieval and traffic sign recognition. Extensive experiments show the proposed method is effective and efficient in complex physical conditions and has a good generalization for different tasks.

摘要: 为了评估物理世界中深度学习的脆弱性，最近的工作引入了对抗性补丁，并将它们应用于不同的任务。在本文中，我们提出了另一种对抗性补丁：有意义的对抗性贴纸，这是一种利用生活中存在的真实贴纸进行物理上可行的隐身攻击的方法。与以往设计扰动的对抗性补丁不同，该方法通过操控贴纸在物体上的粘贴位置和旋转角度来执行物理攻击。由于位置和旋转角度受印刷损失和颜色失真的影响较小，因此对抗性贴纸可以在物理世界中保持良好的攻击性能。此外，为了使对抗性贴纸在真实场景中更具实用性，我们在信息有限的黑盒环境中进行攻击，而不是在包含威胁模型所有细节的白盒环境中进行攻击。为了有效地求解贴纸参数，设计了基于区域的启发式差异进化算法，该算法利用新发现的有效解的区域聚集和评价标准的自适应调整策略。我们的方法在人脸识别中得到了全面的验证，并扩展到图像检索和交通标志识别中。大量实验表明，该方法在复杂的物理条件下是有效的，对不同的任务具有较好的通用性。



## **23. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

地球科学和遥感的人工智能安全：挑战和未来趋势 cs.CV

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09360v1) [paper-pdf](http://arxiv.org/pdf/2212.09360v1)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising, and inpainting, to high-level vision tasks like scene classification, object detection, and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty, and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.

摘要: 人工智能(AI)的最新进展极大地加强了地学和遥感(RS)领域的研究。人工智能算法，特别是基于深度学习的人工智能算法在遥感数据分析中得到了广泛的应用。人工智能的成功应用几乎涵盖了地球观测(EO)任务的方方面面，从超分辨率、去噪和修复等低层视觉任务，到场景分类、目标检测和语义分割等高级视觉任务。虽然人工智能技术使研究人员能够更准确地观察和了解地球，但考虑到许多地学和遥感任务是高度安全关键的，人工智能模型的脆弱性和不确定性值得进一步关注。本文回顾了人工智能安全在地学和遥感领域的发展现状，包括以下五个重要方面：对抗性攻击、后门攻击、联邦学习、不确定性和可解释性。此外，还讨论了潜在的机会和趋势，为未来的研究提供了见解。据作者所知，本文是第一次对地学和遥感社区中与人工智能安全相关的研究进行系统回顾。文中还列出了可用的代码和数据集，以推动这一充满活力的研究领域向前发展。



## **24. Review of security techniques for memristor computing systems**

忆阻器计算系统的安全技术综述 cs.CR

15 pages, 5 figures

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09347v1) [paper-pdf](http://arxiv.org/pdf/2212.09347v1)

**Authors**: Minhui Zou, Nan Du, Shahar Kvatinsky

**Abstract**: Neural network (NN) algorithms have become the dominant tool in visual object recognition, natural language processing, and robotics. To enhance the computational efficiency of these algorithms, in comparison to the traditional von Neuman computing architectures, researchers have been focusing on memristor computing systems. A major drawback when using memristor computing systems today is that, in the artificial intelligence (AI) era, well-trained NN models are intellectual property and, when loaded in the memristor computing systems, face theft threats, especially when running in edge devices. An adversary may steal the well-trained NN models through advanced attacks such as learning attacks and side-channel analysis. In this paper, we review different security techniques for protecting memristor computing systems. Two threat models are described based on their assumptions regarding the adversary's capabilities: a black-box (BB) model and a white-box (WB) model. We categorize the existing security techniques into five classes in the context of these threat models: thwarting learning attacks (BB), thwarting side-channel attacks (BB), NN model encryption (WB), NN weight transformation (WB), and fingerprint embedding (WB). We also present a cross-comparison of the limitations of the security techniques. This paper could serve as an aid when designing secure memristor computing systems.

摘要: 神经网络(NN)算法已经成为视觉对象识别、自然语言处理和机器人学中的主要工具。为了提高这些算法的计算效率，与传统的冯·诺伊曼计算体系结构相比，研究人员一直将重点放在记忆阻器计算系统上。今天使用忆阻器计算系统的一个主要缺点是，在人工智能(AI)时代，训练有素的NN模型是知识产权，当加载到忆阻器计算系统中时，面临盗窃威胁，特别是在边缘设备中运行时。对手可能会通过学习攻击和旁路分析等高级攻击来窃取训练有素的NN模型。在本文中，我们回顾了用于保护忆阻器计算系统的不同安全技术。基于对对手能力的假设，描述了两种威胁模型：黑盒(BB)模型和白盒(WB)模型。根据这些威胁模型，我们将现有的安全技术分为五类：挫败学习攻击(BB)、挫败侧通道攻击(BB)、神经网络模型加密(WB)、神经网络权重变换(WB)和指纹嵌入(WB)。我们还对安全技术的局限性进行了交叉比较。本文对设计安全的忆阻器计算系统具有一定的参考价值。



## **25. TextGrad: Advancing Robustness Evaluation in NLP by Gradient-Driven Optimization**

TextGrad：基于梯度驱动优化的自然语言处理稳健性评价 cs.CL

18 pages, 2 figures

**SubmitDate**: 2022-12-19    [abs](http://arxiv.org/abs/2212.09254v1) [paper-pdf](http://arxiv.org/pdf/2212.09254v1)

**Authors**: Bairu Hou, Jinghan Jia, Yihua Zhang, Guanhua Zhang, Yang Zhang, Sijia Liu, Shiyu Chang

**Abstract**: Robustness evaluation against adversarial examples has become increasingly important to unveil the trustworthiness of the prevailing deep models in natural language processing (NLP). However, in contrast to the computer vision domain where the first-order projected gradient descent (PGD) is used as the benchmark approach to generate adversarial examples for robustness evaluation, there lacks a principled first-order gradient-based robustness evaluation framework in NLP. The emerging optimization challenges lie in 1) the discrete nature of textual inputs together with the strong coupling between the perturbation location and the actual content, and 2) the additional constraint that the perturbed text should be fluent and achieve a low perplexity under a language model. These challenges make the development of PGD-like NLP attacks difficult. To bridge the gap, we propose TextGrad, a new attack generator using gradient-driven optimization, supporting high-accuracy and high-quality assessment of adversarial robustness in NLP. Specifically, we address the aforementioned challenges in a unified optimization framework. And we develop an effective convex relaxation method to co-optimize the continuously-relaxed site selection and perturbation variables and leverage an effective sampling method to establish an accurate mapping from the continuous optimization variables to the discrete textual perturbations. Moreover, as a first-order attack generation method, TextGrad can be baked into adversarial training to further improve the robustness of NLP models. Extensive experiments are provided to demonstrate the effectiveness of TextGrad not only in attack generation for robustness evaluation but also in adversarial defense.

摘要: 针对敌意例子的稳健性评估对于揭示自然语言处理(NLP)中流行的深层模型的可信度变得越来越重要。然而，与计算机视觉领域使用一阶投影梯度下降(PGD)作为基准方法生成对抗性实例进行健壮性评估相比，NLP缺乏一个原则性的基于一阶梯度的健壮性评估框架。新出现的优化挑战在于1)文本输入的离散性质以及扰动位置和实际内容之间的强耦合，以及2)额外的约束，即在语言模型下，被扰动的文本应该流畅并实现低困惑。这些挑战使得类似PGD的NLP攻击的开发变得困难。为了弥补这一差距，我们提出了一种新的攻击生成器TextGrad，该生成器采用梯度驱动优化，支持高精度和高质量的NLP攻击健壮性评估。具体地说，我们在一个统一的优化框架中解决上述挑战。我们提出了一种有效的凸松弛方法来共同优化连续松弛的选址和扰动变量，并利用有效的抽样方法建立了从连续优化变量到离散文本扰动的精确映射。此外，TextGrad作为一种一阶攻击生成方法，可以被引入到对抗性训练中，进一步提高NLP模型的健壮性。大量的实验表明，TextGrad不仅在攻击生成、健壮性评估方面有效，而且在对抗防御方面也是有效的。



## **26. Minimizing Maximum Model Discrepancy for Transferable Black-box Targeted Attacks**

最小化可转移黑盒定向攻击的最大模型偏差 cs.CV

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2212.09035v1) [paper-pdf](http://arxiv.org/pdf/2212.09035v1)

**Authors**: Anqi Zhao, Tong Chu, Yahao Liu, Wen Li, Jingjing Li, Lixin Duan

**Abstract**: In this work, we study the black-box targeted attack problem from the model discrepancy perspective. On the theoretical side, we present a generalization error bound for black-box targeted attacks, which gives a rigorous theoretical analysis for guaranteeing the success of the attack. We reveal that the attack error on a target model mainly depends on empirical attack error on the substitute model and the maximum model discrepancy among substitute models. On the algorithmic side, we derive a new algorithm for black-box targeted attacks based on our theoretical analysis, in which we additionally minimize the maximum model discrepancy(M3D) of the substitute models when training the generator to generate adversarial examples. In this way, our model is capable of crafting highly transferable adversarial examples that are robust to the model variation, thus improving the success rate for attacking the black-box model. We conduct extensive experiments on the ImageNet dataset with different classification models, and our proposed approach outperforms existing state-of-the-art methods by a significant margin. Our codes will be released.

摘要: 在本文中，我们从模型差异的角度研究了黑盒定向攻击问题。在理论方面，给出了黑盒目标攻击的泛化误差界，为保证攻击的成功提供了严密的理论分析。我们发现，对目标模型的攻击误差主要取决于对替代模型的经验攻击误差和替代模型之间的最大模型偏差。在算法方面，我们在理论分析的基础上提出了一种新的黑盒定向攻击算法，该算法在训练生成器生成对抗性实例时，使替换模型的最大模型偏差(M3D)最小化。通过这种方式，我们的模型能够制作高度可移植的对抗性实例，并且对模型变化具有健壮性，从而提高了攻击黑盒模型的成功率。我们在具有不同分类模型的ImageNet数据集上进行了大量的实验，我们的方法比现有的最先进的方法有很大的优势。我们的代码会被公布的。



## **27. A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness**

以语音为中心的可信机器学习：隐私、安全和公平 cs.SD

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2212.09006v1) [paper-pdf](http://arxiv.org/pdf/2212.09006v1)

**Authors**: Tiantian Feng, Rajat Hebbar, Nicholas Mehlman, Xuan Shi, Aditya Kommineni, and Shrikanth Narayanan

**Abstract**: Speech-centric machine learning systems have revolutionized many leading domains ranging from transportation and healthcare to education and defense, profoundly changing how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we point out several promising future research directions to inspire the researchers who wish to explore further in this area.

摘要: 以语音为中心的机器学习系统已经彻底改变了从交通和医疗到教育和国防等许多领先领域，深刻地改变了人们的生活、工作和相互互动的方式。然而，最近的研究表明，许多以语音为中心的ML系统可能需要被认为更值得信赖，以便更广泛地部署。具体地说，在ML研究领域中发现了对隐私泄露、区分性能和对对手攻击的脆弱性的担忧。为了应对上述挑战和风险，已经做出了大量努力，以确保这些ML系统是值得信任的，特别是私密、安全和公平。在本文中，我们首次对以语音为中心的可信ML话题进行了全面的调查，这些话题涉及隐私、安全和公平。除了作为研究界的总结报告外，我们还指出了几个有前途的未来研究方向，以激励希望在这一领域进一步探索的研究人员。



## **28. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-12-18    [abs](http://arxiv.org/abs/2202.12232v4) [paper-pdf](http://arxiv.org/pdf/2202.12232v4)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstract**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy.   In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $(\varepsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme: an effective training set is sub-sampled from a larger set prior to the beginning of training. We find this greatly reduces the bound on MI positive accuracy. As a result, our scheme allows the use of looser DP guarantees to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. While this clearly benefits entities working with far more data than they need to train on, it can also improve the accuracy-privacy trade-off on benchmarks studied in the academic literature. Consequently, we also find that subsampling decreases the effectiveness of a state-of-the-art MI attack (LiRA) much more effectively than training with stronger DP guarantees on MNIST and CIFAR10. We conclude by discussing implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。本文给出了当训练算法提供$(varepsilon，Delta)$-DP时，任意MI对手的正精度(即攻击精度)的一个更严格的界。我们的界给出了一种新的隐私放大方案的设计：在训练开始之前，从一个更大的集合中分采样一个有效的训练集。我们发现这极大地降低了MI正精度的界限。因此，我们的方案允许使用更宽松的DP保证来限制任何MI对手的成功；这确保了模型的准确性不会受到隐私保证的影响。虽然这显然有利于使用远远超过培训所需的数据的实体，但它也可以提高学术文献中研究的基准的精确度和隐私权衡。因此，我们还发现，在MNIST和CIFAR10上，次采样比具有更强DP保证的训练更有效地降低了最先进的MI攻击(LIRA)的有效性。最后，我们讨论了我们的MI界限在机器遗忘领域的含义。



## **29. Scalable Adversarial Attack Algorithms on Influence Maximization**

基于影响力最大化的可扩展敌意攻击算法 cs.SI

11 pages, 2 figures

**SubmitDate**: 2022-12-17    [abs](http://arxiv.org/abs/2209.00892v2) [paper-pdf](http://arxiv.org/pdf/2209.00892v2)

**Authors**: Lichao Sun, Xiaobin Rui, Wei Chen

**Abstract**: In this paper, we study the adversarial attacks on influence maximization under dynamic influence propagation models in social networks. In particular, given a known seed set S, the problem is to minimize the influence spread from S by deleting a limited number of nodes and edges. This problem reflects many application scenarios, such as blocking virus (e.g. COVID-19) propagation in social networks by quarantine and vaccination, blocking rumor spread by freezing fake accounts, or attacking competitor's influence by incentivizing some users to ignore the information from the competitor. In this paper, under the linear threshold model, we adapt the reverse influence sampling approach and provide efficient algorithms of sampling valid reverse reachable paths to solve the problem. We present three different design choices on reverse sampling, which all guarantee $1/2 - \varepsilon$ approximation (for any small $\varepsilon >0$) and an efficient running time.

摘要: 本文研究了社会网络中动态影响传播模型下影响最大化的对抗性攻击。特别地，给定一个已知的种子集S，问题是通过删除有限数量的节点和边来最小化从S传播的影响。这个问题反映了很多应用场景，比如通过隔离和接种疫苗来阻止病毒(例如新冠肺炎)在社交网络中的传播，通过冻结虚假账号来阻止谣言传播，或者通过激励一些用户忽略竞争对手的信息来攻击竞争对手的影响力。本文在线性门限模型下，采用反向影响抽样方法，给出了有效反向可达路径抽样的有效算法。我们给出了三种不同的反向抽样设计选择，它们都保证了$1/2-\varepsilon$近似(对于任何较小的$\varepsilon>0$)和有效的运行时间。



## **30. Expeditious Saliency-guided Mix-up through Random Gradient Thresholding**

基于随机梯度阈值的快速显著引导混合算法 cs.CV

Accepted Long paper at 2nd Practical-DL Workshop at AAAI 2023. V2 fix  typo

**SubmitDate**: 2022-12-17    [abs](http://arxiv.org/abs/2212.04875v2) [paper-pdf](http://arxiv.org/pdf/2212.04875v2)

**Authors**: Minh-Long Luu, Zeyi Huang, Eric P. Xing, Yong Jae Lee, Haohan Wang

**Abstract**: Mix-up training approaches have proven to be effective in improving the generalization ability of Deep Neural Networks. Over the years, the research community expands mix-up methods into two directions, with extensive efforts to improve saliency-guided procedures but minimal focus on the arbitrary path, leaving the randomization domain unexplored. In this paper, inspired by the superior qualities of each direction over one another, we introduce a novel method that lies at the junction of the two routes. By combining the best elements of randomness and saliency utilization, our method balances speed, simplicity, and accuracy. We name our method R-Mix following the concept of "Random Mix-up". We demonstrate its effectiveness in generalization, weakly supervised object localization, calibration, and robustness to adversarial attacks. Finally, in order to address the question of whether there exists a better decision protocol, we train a Reinforcement Learning agent that decides the mix-up policies based on the classifier's performance, reducing dependency on human-designed objectives and hyperparameter tuning. Extensive experiments further show that the agent is capable of performing at the cutting-edge level, laying the foundation for a fully automatic mix-up. Our code is released at [https://github.com/minhlong94/Random-Mixup].

摘要: 混合训练方法已被证明是提高深度神经网络泛化能力的有效方法。多年来，研究界将混淆方法扩展到两个方向，广泛努力改进显著引导程序，但对任意路径的关注很少，留下了随机化领域的未探索。在这篇文章中，我们受到每个方向相对于另一个方向的优越性质的启发，提出了一种位于两条路线交界处的新方法。通过结合随机性和显着性利用的最佳元素，我们的方法平衡了速度、简单性和准确性。我们根据“随机混合”的概念将我们的方法命名为R-Mix。我们证明了它在泛化、弱监督目标定位、校准和对对手攻击的稳健性方面的有效性。最后，为了解决是否存在更好的决策协议的问题，我们训练了一个强化学习代理，它根据分类器的性能来决定混合策略，减少了对人为设计目标和超参数调整的依赖。广泛的实验进一步表明，该试剂能够在尖端水平上发挥作用，为全自动混合奠定了基础。我们的代码在[https://github.com/minhlong94/Random-Mixup].]上发布



## **31. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

15 pages, 14 figures

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2202.03195v4) [paper-pdf](http://arxiv.org/pdf/2202.03195v4)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA在两种防御下的稳健性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **32. LDL: A Defense for Label-Based Membership Inference Attacks**

低密度脂蛋白：一种基于标签的成员推理攻击防御方案 cs.LG

to appear in ACM ASIA Conference on Computer and Communications  Security (ACM ASIACCS 2023)

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.01688v2) [paper-pdf](http://arxiv.org/pdf/2212.01688v2)

**Authors**: Arezoo Rajabi, Dinuka Sahabandu, Luyao Niu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The data used to train deep neural network (DNN) models in applications such as healthcare and finance typically contain sensitive information. A DNN model may suffer from overfitting. Overfitted models have been shown to be susceptible to query-based attacks such as membership inference attacks (MIAs). MIAs aim to determine whether a sample belongs to the dataset used to train a classifier (members) or not (nonmembers). Recently, a new class of label based MIAs (LAB MIAs) was proposed, where an adversary was only required to have knowledge of predicted labels of samples. Developing a defense against an adversary carrying out a LAB MIA on DNN models that cannot be retrained remains an open problem. We present LDL, a light weight defense against LAB MIAs. LDL works by constructing a high-dimensional sphere around queried samples such that the model decision is unchanged for (noisy) variants of the sample within the sphere. This sphere of label-invariance creates ambiguity and prevents a querying adversary from correctly determining whether a sample is a member or a nonmember. We analytically characterize the success rate of an adversary carrying out a LAB MIA when LDL is deployed, and show that the formulation is consistent with experimental observations. We evaluate LDL on seven datasets -- CIFAR-10, CIFAR-100, GTSRB, Face, Purchase, Location, and Texas -- with varying sizes of training data. All of these datasets have been used by SOTA LAB MIAs. Our experiments demonstrate that LDL reduces the success rate of an adversary carrying out a LAB MIA in each case. We empirically compare LDL with defenses against LAB MIAs that require retraining of DNN models, and show that LDL performs favorably despite not needing to retrain the DNNs.

摘要: 用于在医疗保健和金融等应用中训练深度神经网络(DNN)模型的数据通常包含敏感信息。DNN模型可能会出现过度拟合的问题。过适应的模型已经被证明容易受到基于查询的攻击，例如成员推理攻击(MIA)。MIA旨在确定样本是否属于用于训练分类器的数据集(成员)或不属于(非成员)。最近，一类新的基于标签的MIA(Label Based MIA)被提出，其中对手只需要知道样本的预测标签。针对在不能再训练的DNN模型上进行实验室MIA的对手开发防御仍然是一个悬而未决的问题。我们提出了低密度脂蛋白，一种针对实验室MIA的轻量级防御措施。低密度脂蛋白的工作原理是围绕查询的样本构建一个高维球体，使得球体内样本的(噪声)变体的模型决策保持不变。这种标签不变性的范围造成了歧义，并阻止查询对手正确地确定样本是成员还是非成员。我们分析了当部署低密度脂蛋白时对手进行实验室MIA的成功率，并表明该公式与实验观察一致。我们在七个数据集--CIFAR-10、CIFAR-100、GTSRB、Face、Purchase、Location和Texas--上评估LDL，并使用不同大小的训练数据。所有这些数据集都已被SOTA实验室MIA使用。我们的实验表明，在每种情况下，低密度脂蛋白都会降低对手进行实验室MIA的成功率。我们经验性地比较了低密度脂蛋白与对需要重新训练DNN模型的实验室MIA的防御，并表明尽管不需要重新训练DNN，但低密度脂蛋白的表现良好。



## **33. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

对抗性组间链路注入降低了图神经网络的公平性 cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2209.05957v2) [paper-pdf](http://arxiv.org/pdf/2209.05957v2)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstract**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.

摘要: 我们提出了针对图神经网络(GNN)的对抗性攻击的存在和有效性的证据，这些攻击旨在降低公平性。这些攻击可能使基于GNN的节点分类中的特定节点子组处于不利地位，其中底层网络的节点具有敏感属性，如种族或性别。我们进行了定性和实验分析，解释了敌意链接注入如何损害GNN预测的公平性。例如，攻击者可以通过在属于相反子组和相反类标签的节点之间注入敌对链接来损害基于GNN的节点分类的公平性。我们在经验数据集上的实验表明，对抗性公平攻击能够以较低的扰动率(攻击是有效的)显著降低GNN预测的公平性(攻击是有效的)，并且不会显著降低准确率(攻击是欺骗性的)。这项工作证明了GNN模型对敌意公平攻击的脆弱性。我们希望我们的发现提高我们社区对这个问题的认识，并为未来开发更稳健地抵御此类攻击的GNN模型奠定基础。



## **34. Conditional Generative Adversarial Network for keystroke presentation attack**

用于击键呈现攻击的条件生成性对抗网络 cs.CR

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08445v1) [paper-pdf](http://arxiv.org/pdf/2212.08445v1)

**Authors**: Idoia Eizaguirre-Peral, Lander Segurola-Gil, Francesco Zola

**Abstract**: Cybersecurity is a crucial step in data protection to ensure user security and personal data privacy. In this sense, many companies have started to control and restrict access to their data using authentication systems. However, these traditional authentication methods, are not enough for ensuring data protection, and for this reason, behavioral biometrics have gained importance. Despite their promising results and the wide range of applications, biometric systems have shown to be vulnerable to malicious attacks, such as Presentation Attacks. For this reason, in this work, we propose to study a new approach aiming to deploy a presentation attack towards a keystroke authentication system. Our idea is to use Conditional Generative Adversarial Networks (cGAN) for generating synthetic keystroke data that can be used for impersonating an authorized user. These synthetic data are generated following two different real use cases, one in which the order of the typed words is known (ordered dynamic) and the other in which this order is unknown (no-ordered dynamic). Finally, both keystroke dynamics (ordered and no-ordered) are validated using an external keystroke authentication system. Results indicate that the cGAN can effectively generate keystroke dynamics patterns that can be used for deceiving keystroke authentication systems.

摘要: 网络安全是数据保护的关键一步，以确保用户安全和个人数据隐私。从这个意义上说，许多公司已经开始使用身份验证系统来控制和限制对其数据的访问。然而，这些传统的身份验证方法不足以确保数据保护，因此，行为生物识别变得越来越重要。尽管生物识别系统有很好的结果和广泛的应用，但已经显示出它很容易受到恶意攻击，例如演示攻击。因此，在这项工作中，我们建议研究一种新的方法，旨在向击键认证系统部署呈现攻击。我们的想法是使用条件生成对抗网络(CGAN)来生成可用于模拟授权用户的合成击键数据。这些合成数据是在两个不同的实际用例之后生成的，一个用例中键入的单词的顺序是已知的(有序动态)，另一个用例中该顺序是未知的(无序动态)。最后，使用外部击键认证系统来验证击键动力学(有序和无序)。结果表明，cGAN能够有效地生成可用于欺骗击键认证系统的击键动力学模式。



## **35. CgAT: Center-Guided Adversarial Training for Deep Hashing-Based Retrieval**

CgAT：中心引导的基于深度散列的对抗性训练 cs.CV

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2204.10779v4) [paper-pdf](http://arxiv.org/pdf/2204.10779v4)

**Authors**: Xunguang Wang, Yinqun Lin, Xiaomeng Li

**Abstract**: Deep hashing has been extensively utilized in massive image retrieval because of its efficiency and effectiveness. However, deep hashing models are vulnerable to adversarial examples, making it essential to develop adversarial defense methods for image retrieval. Existing solutions achieved limited defense performance because of using weak adversarial samples for training and lacking discriminative optimization objectives to learn robust features. In this paper, we present a min-max based Center-guided Adversarial Training, namely CgAT, to improve the robustness of deep hashing networks through worst adversarial examples. Specifically, we first formulate the center code as a semantically-discriminative representative of the input image content, which preserves the semantic similarity with positive samples and dissimilarity with negative examples. We prove that a mathematical formula can calculate the center code immediately. After obtaining the center codes in each optimization iteration of the deep hashing network, they are adopted to guide the adversarial training process. On the one hand, CgAT generates the worst adversarial examples as augmented data by maximizing the Hamming distance between the hash codes of the adversarial examples and the center codes. On the other hand, CgAT learns to mitigate the effects of adversarial samples by minimizing the Hamming distance to the center codes. Extensive experiments on the benchmark datasets demonstrate the effectiveness of our adversarial training algorithm in defending against adversarial attacks for deep hashing-based retrieval. Compared with the current state-of-the-art defense method, we significantly improve the defense performance by an average of 18.61%, 12.35%, and 11.56% on FLICKR-25K, NUS-WIDE, and MS-COCO, respectively.

摘要: 深度哈希法以其高效、高效的特点在海量图像检索中得到了广泛应用。然而，深度哈希模型很容易受到敌意例子的攻击，因此有必要开发针对图像检索的对抗性防御方法。现有的解决方案由于使用弱对抗性样本进行训练，并且缺乏区分优化目标来学习稳健特征，使得防御性能受到限制。本文提出了一种基于最小-最大值的中心引导敌意训练算法，即CgAT，通过最坏的敌意例子来提高深度哈希网络的健壮性。具体地说，我们首先将中心代码定义为输入图像内容的语义区分代表，它保留了与正例的语义相似性和与反例的不相似性。我们证明了一个数学公式可以立即计算中心代码。在获得深度散列网络每次优化迭代的中心代码后，将其用于指导对抗性训练过程。一方面，CgAT通过最大化对抗性示例的哈希码与中心码之间的汉明距离来生成最差的对抗性示例作为扩充数据。另一方面，CgAT通过最小化到中心码的汉明距离来学习减轻对抗性样本的影响。在基准数据集上的大量实验表明，我们的对抗性训练算法在防御基于深度散列的检索的对抗性攻击方面是有效的。与当前最先进的防御方法相比，我们在Flickr-25K、NUS-Wide和MS-COCO上的防御性能分别平均提高了18.61%、12.35%和11.56%。



## **36. Jujutsu: A Two-stage Defense against Adversarial Patch Attacks on Deep Neural Networks**

Jujutsu：抵抗深度神经网络对抗性补丁攻击的两阶段防御 cs.CR

To appear in AsiaCCS'23

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2108.05075v4) [paper-pdf](http://arxiv.org/pdf/2108.05075v4)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstract**: Adversarial patch attacks create adversarial examples by injecting arbitrary distortions within a bounded region of the input to fool deep neural networks (DNNs). These attacks are robust (i.e., physically-realizable) and universally malicious, and hence represent a severe security threat to real-world DNN-based systems.   We propose Jujutsu, a two-stage technique to detect and mitigate robust and universal adversarial patch attacks. We first observe that adversarial patches are crafted as localized features that yield large influence on the prediction output, and continue to dominate the prediction on any input. Jujutsu leverages this observation for accurate attack detection with low false positives. Patch attacks corrupt only a localized region of the input, while the majority of the input remains unperturbed. Therefore, Jujutsu leverages generative adversarial networks (GAN) to perform localized attack recovery by synthesizing the semantic contents of the input that are corrupted by the attacks, and reconstructs a ``clean'' input for correct prediction.   We evaluate Jujutsu on four diverse datasets spanning 8 different DNN models, and find that it achieves superior performance and significantly outperforms four existing defenses. We further evaluate Jujutsu against physical-world attacks, as well as adaptive attacks.

摘要: 对抗性补丁攻击通过在输入的有界区域内注入任意扭曲来愚弄深度神经网络(DNN)来创建对抗性示例。这些攻击是健壮的(即，物理上可实现的)且普遍是恶意的，因此对现实世界中基于DNN的系统构成了严重的安全威胁。我们提出了Jujutsu，这是一种两阶段技术，用于检测和缓解健壮的、通用的敌意补丁攻击。我们首先观察到，对抗性补丁被精心设计为对预测输出产生巨大影响的局部特征，并继续主导对任何输入的预测。Jujutsu利用这一观察结果进行准确的攻击检测，误报较低。补丁攻击只破坏输入的局部区域，而大多数输入保持不受干扰。因此，Jujutsu利用生成性对抗性网络(GAN)通过合成被攻击破坏的输入的语义内容来执行局部化攻击恢复，并重建用于正确预测的“干净”输入。我们在横跨8个不同DNN模型的四个不同的数据集上对Jujutsu进行了评估，发现它取得了优越的性能，并且显著超过了现有的四种防御方法。我们进一步评估Jujutsu对物理世界的攻击，以及适应性攻击。



## **37. Adversarial Example Defense via Perturbation Grading Strategy**

基于扰动分级策略的对抗性范例防御 cs.CV

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08341v1) [paper-pdf](http://arxiv.org/pdf/2212.08341v1)

**Authors**: Shaowei Zhu, Wanli Lyu, Bin Li, Zhaoxia Yin, Bin Luo

**Abstract**: Deep Neural Networks have been widely used in many fields. However, studies have shown that DNNs are easily attacked by adversarial examples, which have tiny perturbations and greatly mislead the correct judgment of DNNs. Furthermore, even if malicious attackers cannot obtain all the underlying model parameters, they can use adversarial examples to attack various DNN-based task systems. Researchers have proposed various defense methods to protect DNNs, such as reducing the aggressiveness of adversarial examples by preprocessing or improving the robustness of the model by adding modules. However, some defense methods are only effective for small-scale examples or small perturbations but have limited defense effects for adversarial examples with large perturbations. This paper assigns different defense strategies to adversarial perturbations of different strengths by grading the perturbations on the input examples. Experimental results show that the proposed method effectively improves defense performance. In addition, the proposed method does not modify any task model, which can be used as a preprocessing module, which significantly reduces the deployment cost in practical applications.

摘要: 深度神经网络已经在许多领域得到了广泛的应用。然而，研究表明，DNN容易受到敌意例子的攻击，这些例子具有微小的扰动，极大地误导了对DNN的正确判断。此外，即使恶意攻击者无法获得所有底层模型参数，他们也可以使用对抗性示例来攻击各种基于DNN的任务系统。研究人员已经提出了各种防御方法来保护DNN，例如通过预处理来降低对抗性例子的攻击性，或者通过增加模块来提高模型的健壮性。然而，一些防御方法只对小范围的例子或小扰动有效，而对大扰动的对抗性例子的防御效果有限。通过对输入样本上的扰动进行分级，为不同强度的对抗性扰动分配不同的防御策略。实验结果表明，该方法有效地提高了防御性能。此外，该方法不需要修改任何任务模型，可以作为一个预处理模块，大大降低了实际应用中的部署成本。



## **38. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

对攻击者的对抗性攻击：减轻基于黑盒分数的查询攻击的后处理 cs.LG

accepted by NeurIPS 2022

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2205.12134v3) [paper-pdf](http://arxiv.org/pdf/2205.12134v3)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstract**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure 80.59% accuracy under Square attack (2500 queries), while the best prior defense (i.e., adversarial training) only attains 67.44%. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets, bounds, norms, losses, and strategies. Moreover, AAA calibrates better without hurting the accuracy. Our code is available at https://github.com/Sizhe-Chen/AAA.

摘要: 基于分数的查询攻击(SQA)通过在数十个查询中精心设计敌意扰动，仅使用模型的输出分数，对深度神经网络构成实际威胁。尽管如此，我们注意到，如果产出的损失趋势受到轻微干扰，质量保证人员很容易受到误导，从而变得不那么有效。根据这一思想，我们提出了一种新的防御方法，即对攻击者的对抗性攻击(AAA)，通过略微修改输出日志来迷惑SQA对错误的攻击方向。这样，(1)无论模型在最坏情况下的稳健性如何，都可以防止SQA；(2)原始模型预测几乎不会改变，即不会降低干净的精度；(3)置信度得分的校准可以同时得到改善。通过大量的实验验证了上述优点。例如，通过在CIFAR-10上设置$\ell_\inty=8/255$，我们提出的AAA在Square攻击(2500个查询)下帮助WideResNet-28确保80.59%的准确率，而最好的先前防御(即对抗性训练)仅达到67.44%。由于AAA攻击SQA的一般贪婪策略，因此在6个SQA下的8个CIFAR-10/ImageNet模型上，使用不同的攻击目标、边界、规范、损失和策略，可以一致地观察到AAA相对于8个防御的优势。此外，AAA在不影响精度的情况下校准得更好。我们的代码可以在https://github.com/Sizhe-Chen/AAA.上找到



## **39. A Survey on Biometrics Authentication**

生物特征认证研究综述 cs.CR

6 pages, 9 figures, 9 references

**SubmitDate**: 2022-12-16    [abs](http://arxiv.org/abs/2212.08224v1) [paper-pdf](http://arxiv.org/pdf/2212.08224v1)

**Authors**: Fangshi Zhou, Tianming Zhao

**Abstract**: Nowadays, traditional authentication methods are vulnerable to face attacks that are often based on inherent security issues. Professional attackers leverage adversarial offenses on the security holes. Biometrics has intrinsic advantages to overcome the traditional authentication methods on security, success rates, efficiency, and accessibility. Biometrics has wide prospects to implement various applications in fields. Whether in authentication security or clinical medicine, biometrics is one of the mainstream studies. In this paper, we surveyed and reviewed some related studies of biometrics, which are outstanding and significant in driving the development and popularization of biometrics. Although they still have some inherent disadvantages to restrict popularization, these obstacles could not conceal the promising future of biometrics. Multi-factors continuous biometrics authentication has become the mainstream trend of development. We reflect the findings as well as the challenges of the studies in the survey paper.

摘要: 如今，传统的身份验证方法很容易受到人脸攻击，这些攻击往往基于固有的安全问题。专业攻击者在安全漏洞上利用对抗性进攻。生物特征识别技术在安全性、成功率、效率和可获得性等方面具有克服传统认证方法的固有优势。生物特征识别技术在各个领域有着广阔的应用前景。无论是在身份认证安全领域，还是在临床医学领域，生物特征识别都是主流研究方向之一。本文对生物特征识别的相关研究进行了综述，这些研究对推动生物特征识别技术的发展和普及具有重要意义。虽然它们仍然存在一些限制普及的固有缺陷，但这些障碍并不能掩盖生物识别的光明前景。多因素连续生物特征认证已成为主流的发展趋势。我们在调查文件中反映了调查结果以及研究的挑战。



## **40. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

用投影梯度下降法量化对抗性训练中模型梯度的优先方向 stat.ML

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2009.04709v4) [paper-pdf](http://arxiv.org/pdf/2009.04709v4)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.

摘要: 对抗性训练，特别是投影梯度下降(PGD)，已被证明是提高对抗攻击的稳健性的一种成功方法。经过对抗性训练后，模型相对于其输入的梯度具有优先的方向。然而，对齐的方向在数学上没有很好的确定，因此很难进行定量评估。我们提出了一种新的方向定义，即向量指向决策空间中最接近的不准确类的支持度的最近点的方向。为了在对抗性训练后评估与这一方向的一致性，我们应用了一种度量，该度量使用生成性对抗性网络来产生改变图像中存在的类别所需的最小残差。我们表明，根据我们的定义，PGD训练的模型具有比基线更高的比对，我们的指标比竞争指标公式提供了更高的比对值，并且强制执行这种比对提高了模型的稳健性。



## **41. On Evaluating Adversarial Robustness of Chest X-ray Classification: Pitfalls and Best Practices**

评价胸部X线片分类的对抗性：陷阱和最佳实践 eess.IV

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.08130v1) [paper-pdf](http://arxiv.org/pdf/2212.08130v1)

**Authors**: Salah Ghamizi, Maxime Cordy, Michail Papadakis, Yves Le Traon

**Abstract**: Vulnerability to adversarial attacks is a well-known weakness of Deep Neural Networks. While most of the studies focus on natural images with standardized benchmarks like ImageNet and CIFAR, little research has considered real world applications, in particular in the medical domain. Our research shows that, contrary to previous claims, robustness of chest x-ray classification is much harder to evaluate and leads to very different assessments based on the dataset, the architecture and robustness metric. We argue that previous studies did not take into account the peculiarity of medical diagnosis, like the co-occurrence of diseases, the disagreement of labellers (domain experts), the threat model of the attacks and the risk implications for each successful attack.   In this paper, we discuss the methodological foundations, review the pitfalls and best practices, and suggest new methodological considerations for evaluating the robustness of chest xray classification models. Our evaluation on 3 datasets, 7 models, and 18 diseases is the largest evaluation of robustness of chest x-ray classification models.

摘要: 对敌意攻击的脆弱性是深度神经网络的一个众所周知的弱点。虽然大多数研究都集中在具有ImageNet和CIFAR等标准化基准的自然图像上，但很少有研究考虑现实世界的应用，特别是在医学领域。我们的研究表明，与之前的说法相反，胸部X光分类的稳健性更难评估，并导致基于数据集、体系结构和稳健性度量的评估非常不同。我们认为，以前的研究没有考虑到医学诊断的特殊性，如疾病的共生、标签者(领域专家)的分歧、攻击的威胁模型以及每次成功攻击的风险含义。在这篇文章中，我们讨论了方法学基础，回顾了陷阱和最佳实践，并提出了新的方法学考虑来评估胸部X光分类模型的稳健性。我们对3个数据集、7个模型和18种疾病的评估是对胸部X光分类模型稳健性的最大评估。



## **42. Alternating Objectives Generates Stronger PGD-Based Adversarial Attacks**

交替的目标会产生更强的基于PGD的对抗性攻击 cs.LG

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07992v1) [paper-pdf](http://arxiv.org/pdf/2212.07992v1)

**Authors**: Nikolaos Antoniou, Efthymios Georgiou, Alexandros Potamianos

**Abstract**: Designing powerful adversarial attacks is of paramount importance for the evaluation of $\ell_p$-bounded adversarial defenses. Projected Gradient Descent (PGD) is one of the most effective and conceptually simple algorithms to generate such adversaries. The search space of PGD is dictated by the steepest ascent directions of an objective. Despite the plethora of objective function choices, there is no universally superior option and robustness overestimation may arise from ill-suited objective selection. Driven by this observation, we postulate that the combination of different objectives through a simple loss alternating scheme renders PGD more robust towards design choices. We experimentally verify this assertion on a synthetic-data example and by evaluating our proposed method across 25 different $\ell_{\infty}$-robust models and 3 datasets. The performance improvement is consistent, when compared to the single loss counterparts. In the CIFAR-10 dataset, our strongest adversarial attack outperforms all of the white-box components of AutoAttack (AA) ensemble, as well as the most powerful attacks existing on the literature, achieving state-of-the-art results in the computational budget of our study ($T=100$, no restarts).

摘要: 设计强大的对抗性攻击对于评估$\ellp$受限的对抗性防御是至关重要的。投影梯度下降(PGD)算法是生成此类攻击的最有效且概念简单的算法之一。PGD的搜索空间由目标的最陡上升方向决定。尽管有过多的目标函数选择，但并不存在普遍的最优选择，而且不合适的目标选择可能会导致稳健性高估。在这一观察结果的驱动下，我们假设通过一个简单的损失交替方案将不同的目标组合在一起，使PGD对设计选择更加稳健。我们在一个合成数据示例上实验验证了这一断言，并通过25个不同的$稳健模型和3个数据集评估了我们提出的方法。与单损对应的性能相比，性能改进是一致的。在CIFAR-10数据集中，我们最强的对手攻击超过了AutoAttack(AA)集成的所有白盒组件，以及文献中存在的最强大的攻击，在我们研究的计算预算中获得了最先进的结果($T=100美元，没有重启)。



## **43. Holistic risk assessment of inference attacks in machine learning**

机器学习中推理攻击的整体风险评估 cs.CR

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.10628v1) [paper-pdf](http://arxiv.org/pdf/2212.10628v1)

**Authors**: Yang Yang

**Abstract**: As machine learning expanding application, there are more and more unignorable privacy and safety issues. Especially inference attacks against Machine Learning models allow adversaries to infer sensitive information about the target model, such as training data, model parameters, etc. Inference attacks can lead to serious consequences, including violating individuals privacy, compromising the intellectual property of the owner of the machine learning model. As far as concerned, researchers have studied and analyzed in depth several types of inference attacks, albeit in isolation, but there is still a lack of a holistic rick assessment of inference attacks against machine learning models, such as their application in different scenarios, the common factors affecting the performance of these attacks and the relationship among the attacks. As a result, this paper performs a holistic risk assessment of different inference attacks against Machine Learning models. This paper focuses on three kinds of representative attacks: membership inference attack, attribute inference attack and model stealing attack. And a threat model taxonomy is established. A total of 12 target models using three model architectures, including AlexNet, ResNet18 and Simple CNN, are trained on four datasets, namely CelebA, UTKFace, STL10 and FMNIST.

摘要: 随着机器学习应用范围的扩大，越来越多的隐私和安全问题不容忽视。特别是针对机器学习模型的推理攻击，允许攻击者推断目标模型的敏感信息，如训练数据、模型参数等。推理攻击会导致严重的后果，包括侵犯个人隐私，损害机器学习模型所有者的知识产权。就目前而言，研究人员已经对几种类型的推理攻击进行了深入的研究和分析，尽管是孤立的，但仍然缺乏针对机器学习模型的推理攻击的整体Rick评估，例如它们在不同场景中的应用，影响这些攻击性能的共同因素以及攻击之间的关系。因此，本文针对机器学习模型对不同的推理攻击进行了全面的风险评估。本文重点研究了三种具有代表性的攻击：成员关系推理攻击、属性推理攻击和模型窃取攻击。并建立了威胁模型分类法。在CelebA、UTKFace、STL10和FMNIST四个数据集上，使用AlexNet、ResNet18和Simple CNN等三种模型架构对12个目标模型进行了训练。



## **44. A Feedback-optimization-based Model-free Attack Scheme in Networked Control Systems**

一种基于反馈优化的网络控制系统无模型攻击方案 math.OC

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07633v1) [paper-pdf](http://arxiv.org/pdf/2212.07633v1)

**Authors**: Xiaoyu Luo, Chongrong Fang, Jianping He, Chengcheng Zhao, Dario Paccagnan

**Abstract**: The data-driven attack strategies recently have received much attention when the system model is unknown to the adversary. Depending on the unknown linear system model, the existing data-driven attack strategies have already designed lots of efficient attack schemes. In this paper, we study a completely model-free attack scheme regardless of whether the unknown system model is linear or nonlinear. The objective of the adversary with limited capability is to compromise state variables such that the output value follows a malicious trajectory. Specifically, we first construct a zeroth-order feedback optimization framework and uninterruptedly use probing signals for real-time measurements. Then, we iteratively update the attack signals along the composite direction of the gradient estimates of the objective function evaluations and the projected gradients. These objective function evaluations can be obtained only by real-time measurements. Furthermore, we characterize the optimality of the proposed model-free attack via the optimality gap, which is affected by the dimensions of the attack signal, the iterations of solutions, and the convergence rate of the system. Finally, we extend the proposed attack scheme to the system with internal inherent noise and analyze the effects of noise on the optimality gap. Extensive simulations are conducted to show the effectiveness of the proposed attack scheme.

摘要: 近年来，在敌方未知系统模型的情况下，数据驱动攻击策略受到了广泛的关注。基于未知的线性系统模型，现有的数据驱动攻击策略已经设计了很多有效的攻击方案。本文研究了一种完全无模型的攻击方案，无论未知系统模型是线性的还是非线性的。能力有限的对手的目标是妥协状态变量，使输出值遵循恶意轨迹。具体地说，我们首先构建了零阶反馈优化框架，并不间断地使用探测信号进行实时测量。然后，我们沿着目标函数评估的梯度估计和投影梯度的合成方向迭代地更新攻击信号。这些目标函数评估只能通过实时测量来获得。此外，我们通过最优性间隙来刻画所提出的无模型攻击的最优性，该最优性间隙受攻击信号的维度、解的迭代次数和系统的收敛速度的影响。最后，我们将所提出的攻击方案扩展到具有内部固有噪声的系统，并分析了噪声对最优性间隙的影响。通过大量的仿真实验，验证了该攻击方案的有效性。



## **45. Dissecting Distribution Inference**

剖析分布推理 cs.LG

Accepted at SaTML 2023

**SubmitDate**: 2022-12-15    [abs](http://arxiv.org/abs/2212.07591v1) [paper-pdf](http://arxiv.org/pdf/2212.07591v1)

**Authors**: Anshuman Suri, Yifu Lu, Yanjin Chen, David Evans

**Abstract**: A distribution inference attack aims to infer statistical properties of data used to train machine learning models. These attacks are sometimes surprisingly potent, but the factors that impact distribution inference risk are not well understood and demonstrated attacks often rely on strong and unrealistic assumptions such as full knowledge of training environments even in supposedly black-box threat scenarios. To improve understanding of distribution inference risks, we develop a new black-box attack that even outperforms the best known white-box attack in most settings. Using this new attack, we evaluate distribution inference risk while relaxing a variety of assumptions about the adversary's knowledge under black-box access, like known model architectures and label-only access. Finally, we evaluate the effectiveness of previously proposed defenses and introduce new defenses. We find that although noise-based defenses appear to be ineffective, a simple re-sampling defense can be highly effective. Code is available at https://github.com/iamgroot42/dissecting_distribution_inference

摘要: 分布推断攻击旨在推断用于训练机器学习模型的数据的统计特性。这些攻击有时威力惊人，但影响分布推断风险的因素并未得到很好的理解，已证明的攻击往往依赖于强大而不切实际的假设，例如完全了解训练环境，即使在假设的黑箱威胁场景中也是如此。为了提高对分布推断风险的理解，我们开发了一种新的黑盒攻击，该攻击在大多数情况下甚至比最著名的白盒攻击性能更好。使用这种新的攻击，我们评估了分布推断风险，同时放松了在黑盒访问下关于对手知识的各种假设，如已知的模型体系结构和仅标签访问。最后，我们评估了以前提出的防御措施的有效性，并引入了新的防御措施。我们发现，尽管基于噪声的防御似乎无效，但简单的重新采样防御可以非常有效。代码可在https://github.com/iamgroot42/dissecting_distribution_inference上找到



## **46. SAIF: Sparse Adversarial and Interpretable Attack Framework**

SAIF：稀疏对抗性可解释攻击框架 cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.07495v1) [paper-pdf](http://arxiv.org/pdf/2212.07495v1)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，将计算的小失真添加到图像可以欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新的攻击技术，称为稀疏对抗性和可解释攻击框架(SAIF)。具体地说，我们设计了在少量像素处包含低幅度扰动的不可察觉攻击，并利用这些稀疏攻击来揭示分类器的脆弱性。我们使用Frank-Wolfe(条件梯度)算法来同时优化有界模和稀疏性的攻击扰动，并且具有$O(1/\Sqrt{T})$收敛。实验结果表明，该算法能够计算高度不可察觉和可解释的敌意实例，并且在ImageNet数据集上的性能优于最新的稀疏攻击方法。



## **47. XRand: Differentially Private Defense against Explanation-Guided Attacks**

XRand：针对解释制导攻击的差异化私人防御 cs.LG

To be published at AAAI 2023

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.04454v3) [paper-pdf](http://arxiv.org/pdf/2212.04454v3)

**Authors**: Truc Nguyen, Phung Lai, NhatHai Phan, My T. Thai

**Abstract**: Recent development in the field of explainable artificial intelligence (XAI) has helped improve trust in Machine-Learning-as-a-Service (MLaaS) systems, in which an explanation is provided together with the model prediction in response to each query. However, XAI also opens a door for adversaries to gain insights into the black-box models in MLaaS, thereby making the models more vulnerable to several attacks. For example, feature-based explanations (e.g., SHAP) could expose the top important features that a black-box model focuses on. Such disclosure has been exploited to craft effective backdoor triggers against malware classifiers. To address this trade-off, we introduce a new concept of achieving local differential privacy (LDP) in the explanations, and from that we establish a defense, called XRand, against such attacks. We show that our mechanism restricts the information that the adversary can learn about the top important features, while maintaining the faithfulness of the explanations.

摘要: 可解释人工智能(XAI)领域的最新发展有助于提高对机器学习即服务(MLaaS)系统的信任，在MLaaS系统中，响应于每个查询，提供解释和模型预测。然而，Xai也为对手打开了一扇门，让他们能够洞察MLaaS中的黑盒模型，从而使这些模型更容易受到几次攻击。例如，基于特征的解释(例如，Shap)可以揭示黑盒模型关注的最重要的特征。这种披露已被利用来手工创建针对恶意软件分类器的有效后门触发器。为了解决这种权衡，我们在解释中引入了实现本地差异隐私(LDP)的新概念，并由此建立了针对此类攻击的防御系统，称为XRand。我们表明，我们的机制限制了攻击者可以了解的关于顶级重要特征的信息，同时保持了解释的可靠性。



## **48. The Devil is in the GAN: Backdoor Attacks and Defenses in Deep Generative Models**

魔鬼在甘地：深层生成模型中的后门攻击和防御 cs.CR

17 pages, 11 figures, 3 tables

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2108.01644v2) [paper-pdf](http://arxiv.org/pdf/2108.01644v2)

**Authors**: Ambrish Rawat, Killian Levacher, Mathieu Sinn

**Abstract**: Deep Generative Models (DGMs) are a popular class of deep learning models which find widespread use because of their ability to synthesize data from complex, high-dimensional manifolds. However, even with their increasing industrial adoption, they haven't been subject to rigorous security and privacy analysis. In this work we examine one such aspect, namely backdoor attacks on DGMs which can significantly limit the applicability of pre-trained models within a model supply chain and at the very least cause massive reputation damage for companies outsourcing DGMs form third parties.   While similar attacks scenarios have been studied in the context of classical prediction models, their manifestation in DGMs hasn't received the same attention. To this end we propose novel training-time attacks which result in corrupted DGMs that synthesize regular data under normal operations and designated target outputs for inputs sampled from a trigger distribution. These attacks are based on an adversarial loss function that combines the dual objectives of attack stealth and fidelity. We systematically analyze these attacks, and show their effectiveness for a variety of approaches like Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs), as well as different data domains including images and audio. Our experiments show that - even for large-scale industry-grade DGMs (like StyleGAN) - our attacks can be mounted with only modest computational effort. We also motivate suitable defenses based on static/dynamic model and output inspections, demonstrate their usefulness, and prescribe a practical and comprehensive defense strategy that paves the way for safe usage of DGMs.

摘要: 深度生成模型(DGM)是一类流行的深度学习模型，由于其能够从复杂的高维流形中合成数据而得到广泛的应用。然而，即使它们越来越多地被工业采用，它们也没有受到严格的安全和隐私分析。在这项工作中，我们研究了一个这样的方面，即对DGM的后门攻击，这种攻击可以显著限制预先训练的模型在模型供应链中的适用性，至少会给从第三方外包DGM的公司造成巨大的声誉损害。虽然类似的攻击场景已经在经典预测模型的背景下进行了研究，但它们在DGM中的表现并没有得到同样的关注。为此，我们提出了新的训练时间攻击，它导致损坏的DGM，这些DGM在正常操作下合成规则数据，并为从触发分布采样的输入指定目标输出。这些攻击基于一种对抗性损失函数，该函数结合了攻击隐形和保真的双重目标。我们系统地分析了这些攻击，并展示了它们对各种方法的有效性，如生成性对抗网络(GANS)和变分自动编码器(VAES)，以及包括图像和音频在内的不同数据域。我们的实验表明，即使对于大规模的工业级DGM(如StyleGAN)，我们的攻击也可以通过适度的计算工作来发起。我们还根据静态/动态模型和输出检查来激发适当的防御，展示其有效性，并制定实用和全面的防御战略，为DGM的安全使用铺平道路。



## **49. Object-fabrication Targeted Attack for Object Detection**

面向目标检测的目标制造定向攻击 cs.CV

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2212.06431v2) [paper-pdf](http://arxiv.org/pdf/2212.06431v2)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hang Wang, Hongbin Sun, Nanning Zheng

**Abstract**: Recent researches show that the deep learning based object detection is vulnerable to adversarial examples. Generally, the adversarial attack for object detection contains targeted attack and untargeted attack. According to our detailed investigations, the research on the former is relatively fewer than the latter and all the existing methods for the targeted attack follow the same mode, i.e., the object-mislabeling mode that misleads detectors to mislabel the detected object as a specific wrong label. However, this mode has limited attack success rate, universal and generalization performances. In this paper, we propose a new object-fabrication targeted attack mode which can mislead detectors to `fabricate' extra false objects with specific target labels. Furthermore, we design a dual attention based targeted feature space attack method to implement the proposed targeted attack mode. The attack performances of the proposed mode and method are evaluated on MS COCO and BDD100K datasets using FasterRCNN and YOLOv5. Evaluation results demonstrate that, the proposed object-fabrication targeted attack mode and the corresponding targeted feature space attack method show significant improvements in terms of image-specific attack, universal performance and generalization capability, compared with the previous targeted attack for object detection. Code will be made available.

摘要: 最近的研究表明，基于深度学习的目标检测容易受到敌意例子的影响。通常，用于目标检测的对抗性攻击包括定向攻击和非定向攻击。根据我们的详细调查，对前者的研究相对较少，现有的所有定向攻击方法都遵循相同的模式，即对象错误标记模式，即误导检测器将检测到的对象错误标记为特定的错误标记。然而，该模式的攻击成功率、通用性和泛化性能有限。本文提出了一种新的目标制造定向攻击模式，该模式可以误导检测器用特定的目标标签‘制造’额外的虚假目标。此外，我们设计了一种基于双重注意力的目标特征空间攻击方法来实现所提出的目标攻击模式。利用FasterRCNN和YOLOv5对该模型和方法在MS COCO和BDD100K数据集上的攻击性能进行了评估。评估结果表明，与以往的目标检测的目标攻击方法相比，本文提出的目标制造目标攻击模式和相应的目标特征空间攻击方法在图像针对性攻击、通用性能和泛化能力方面都有明显的提高。代码将可用。



## **50. ARCADE: Adversarially Regularized Convolutional Autoencoder for Network Anomaly Detection**

Arcade：一种用于网络异常检测的对数正则卷积自动编码器 cs.LG

**SubmitDate**: 2022-12-14    [abs](http://arxiv.org/abs/2205.01432v3) [paper-pdf](http://arxiv.org/pdf/2205.01432v3)

**Authors**: Willian T. Lunardi, Martin Andreoni Lopez, Jean-Pierre Giacalone

**Abstract**: As the number of heterogenous IP-connected devices and traffic volume increase, so does the potential for security breaches. The undetected exploitation of these breaches can bring severe cybersecurity and privacy risks. Anomaly-based \acp{IDS} play an essential role in network security. In this paper, we present a practical unsupervised anomaly-based deep learning detection system called ARCADE (Adversarially Regularized Convolutional Autoencoder for unsupervised network anomaly DEtection). With a convolutional \ac{AE}, ARCADE automatically builds a profile of the normal traffic using a subset of raw bytes of a few initial packets of network flows so that potential network anomalies and intrusions can be efficiently detected before they cause more damage to the network. ARCADE is trained exclusively on normal traffic. An adversarial training strategy is proposed to regularize and decrease the \ac{AE}'s capabilities to reconstruct network flows that are out-of-the-normal distribution, thereby improving its anomaly detection capabilities. The proposed approach is more effective than state-of-the-art deep learning approaches for network anomaly detection. Even when examining only two initial packets of a network flow, ARCADE can effectively detect malware infection and network attacks. ARCADE presents 20 times fewer parameters than baselines, achieving significantly faster detection speed and reaction time.

摘要: 随着异类IP连接设备的数量和流量的增加，安全漏洞的可能性也在增加。未被发现的对这些漏洞的利用可能会带来严重的网络安全和隐私风险。基于异常的ACP(入侵检测系统)在网络安全中起着至关重要的作用。本文提出了一种实用的基于无监督异常的深度学习检测系统ARCADE(对抗性正则化卷积自动编码器用于无监督网络异常检测)。通过卷积\ac{AE}，Arcade使用几个网络流初始数据包的原始字节子集自动构建正常流量的配置文件，以便在潜在的网络异常和入侵对网络造成更大损害之前有效地检测到它们。拱廊是专门针对正常交通进行培训的。提出了一种对抗性训练策略，以规范和降低Ac{AE}重构非正态分布网络流的能力，从而提高其异常检测能力。该方法比现有的深度学习网络异常检测方法更有效。即使只检测网络流的两个初始包，ARCADE也可以有效地检测恶意软件感染和网络攻击。ARCADE提供的参数比基线少20倍，检测速度和反应时间显著加快。



