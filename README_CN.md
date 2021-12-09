# Latest Adversarial Attack Papers
**update at 2021-12-09 23:56:43**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On anti-stochastic properties of unlabeled graphs**

关于无标号图的反随机性 cs.DM

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04395v1)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstracts**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.

摘要: 我们研究了均匀分布随机图在遭受敌手攻击时的脆弱性，该敌手的目标是改变分布的全局，但只能对图进行局部改变。如果一个随机图$G$满足$A$的概率很小，但在很高的概率下，存在一个将$G$转换成满足$A$的图的小扰动，我们称图性质$A$是反随机的。而对于有标号的图，这样的性质很容易从二元覆盖码中获得，而无标号图的反随机性的存在就不那么明显了。如果一个允许的扰动是增加或删除一条边，我们表现出一个反随机性质，它由一个概率为$(2+o(1))/n^2$的n阶随机无标号图所满足，它是尽可能小的。我们还用图的度序列来表示另一个反随机性质。该属性的概率为$(2+o(1))/(n\ln)$，最优为因子2。



## **2. SNEAK: Synonymous Sentences-Aware Adversarial Attack on Natural Language Video Localization**

Screak：自然语言视频本地化的同义句感知对抗性攻击 cs.CV

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04154v1)

**Authors**: Wenbo Gou, Wen Shi, Jian Lou, Lijie Huang, Pan Zhou, Ruixuan Li

**Abstracts**: Natural language video localization (NLVL) is an important task in the vision-language understanding area, which calls for an in-depth understanding of not only computer vision and natural language side alone, but more importantly the interplay between both sides. Adversarial vulnerability has been well-recognized as a critical security issue of deep neural network models, which requires prudent investigation. Despite its extensive yet separated studies in video and language tasks, current understanding of the adversarial robustness in vision-language joint tasks like NLVL is less developed. This paper therefore aims to comprehensively investigate the adversarial robustness of NLVL models by examining three facets of vulnerabilities from both attack and defense aspects. To achieve the attack goal, we propose a new adversarial attack paradigm called synonymous sentences-aware adversarial attack on NLVL (SNEAK), which captures the cross-modality interplay between the vision and language sides.

摘要: 自然语言视频定位(NLVL)是视觉-语言理解领域的一项重要任务，不仅需要深入理解计算机视觉和自然语言两个方面，更重要的是要深入理解两者之间的相互作用。对抗性漏洞已被公认为深度神经网络模型中的一个关键安全问题，需要进行仔细的研究。尽管它对视频和语言任务进行了广泛而独立的研究，但目前对NLVL等视觉-语言联合任务中的对抗性健壮性的理解还不够深入。因此，本文旨在通过从攻击和防御两个方面检查漏洞的三个方面来全面研究NLVL模型的对抗健壮性。为了达到攻击目标，我们提出了一种新的对抗性攻击范式，称为NLVL同义句感知对抗性攻击(SINVAK)，它捕捉了视觉和语言双方之间的跨通道交互作用。



## **3. Adversarial Prefetch: New Cross-Core Cache Side Channel Attacks**

对抗性预取：新的跨核心缓存侧通道攻击 cs.CR

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2110.12340v2)

**Authors**: Yanan Guo, Andrew Zigerelli, Youtao Zhang, Jun Yang

**Abstracts**: Modern x86 processors have many prefetch instructions that can be used by programmers to boost performance. However, these instructions may also cause security problems. In particular, we found that on Intel processors, there are two security flaws in the implementation of PREFETCHW, an instruction for accelerating future writes. First, this instruction can execute on data with read-only permission. Second, the execution time of this instruction leaks the current coherence state of the target data.   Based on these two design issues, we build two cross-core private cache attacks that work with both inclusive and non-inclusive LLCs, named Prefetch+Reload and Prefetch+Prefetch. We demonstrate the significance of our attacks in different scenarios. First, in the covert channel case, Prefetch+Reload and Prefetch+Prefetch achieve 782 KB/s and 822 KB/s channel capacities, when using only one shared cache line between the sender and receiver, the largest-to-date single-line capacities for CPU cache covert channels. Further, in the side channel case, our attacks can monitor the access pattern of the victim on the same processor, with almost zero error rate. We show that they can be used to leak private information of real-world applications such as cryptographic keys. Finally, our attacks can be used in transient execution attacks in order to leak more secrets within the transient window than prior work. From the experimental results, our attacks allow leaking about 2 times as many secret bytes, compared to Flush+Reload, which is widely used in transient execution attacks.

摘要: 现代x86处理器有许多预取指令，程序员可以使用这些指令来提高性能。但是，这些说明也可能导致安全问题。特别是，我们发现在Intel处理器上，PREFETCHW的实现存在两个安全缺陷，PREFETCHW是一条用于加速未来写入的指令。首先，此指令可以在具有只读权限的数据上执行。其次，此指令的执行时间会泄漏目标数据的当前一致性状态。基于这两个设计问题，我们构建了两种同时适用于包含性和非包含性LLC的跨核私有缓存攻击，分别称为预取+重新加载和预取+预取。我们在不同的情况下展示了我们的攻击的重要性。首先，在隐蔽通道的情况下，当在发送器和接收器之间仅使用一个共享高速缓存线时，预取+重新加载和预取+预取分别达到782KB/s和822KB/s的通道容量，这是迄今为止CPU高速缓存隐蔽通道的最大单线容量。此外，在旁信道情况下，我们的攻击可以监视受害者在同一处理器上的访问模式，误码率几乎为零。我们证明了它们可以被用来泄露真实世界应用程序的私有信息，例如密钥。最后，我们的攻击可以用于瞬态执行攻击，以便在瞬态窗口内泄露比以往工作更多的秘密。从实验结果看，与瞬时执行攻击中广泛使用的刷新+重新加载相比，我们的攻击允许泄漏大约2倍的秘密字节。



## **4. Two Coupled Rejection Metrics Can Tell Adversarial Examples Apart**

两个耦合的拒绝度量可以区分敌意的例子 cs.LG

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2105.14785v3)

**Authors**: Tianyu Pang, Huishuai Zhang, Di He, Yinpeng Dong, Hang Su, Wei Chen, Jun Zhu, Tie-Yan Liu

**Abstracts**: Correctly classifying adversarial examples is an essential but challenging requirement for safely deploying machine learning models. As reported in RobustBench, even the state-of-the-art adversarially trained models struggle to exceed 67% robust test accuracy on CIFAR-10, which is far from practical. A complementary way towards robustness is to introduce a rejection option, allowing the model to not return predictions on uncertain inputs, where confidence is a commonly used certainty proxy. Along with this routine, we find that confidence and a rectified confidence (R-Con) can form two coupled rejection metrics, which could provably distinguish wrongly classified inputs from correctly classified ones. This intriguing property sheds light on using coupling strategies to better detect and reject adversarial examples. We evaluate our rectified rejection (RR) module on CIFAR-10, CIFAR-10-C, and CIFAR-100 under several attacks including adaptive ones, and demonstrate that the RR module is compatible with different adversarial training frameworks on improving robustness, with little extra computation. The code is available at https://github.com/P2333/Rectified-Rejection.

摘要: 正确分类敌意示例是安全部署机器学习模型的基本要求，但也是具有挑战性的要求。正如RobustBench报道的那样，即使是经过对抗性训练的最先进的模型也难以在CIFAR-10上超过67%的稳健测试准确率，这是远远不现实的。稳健性的一种补充方式是引入拒绝选项，允许模型不返回对不确定输入的预测，其中置信度是常用的确定性代理。伴随着这个例程，我们发现置信度和校正置信度(R-CON)可以形成两个耦合的拒绝度量，它们可以很好地区分错误分类的输入和正确分类的输入。这一耐人寻味的性质有助于使用耦合策略更好地检测和拒绝敌意示例。我们在CIFAR-10、CIFAR-10-C和CIFAR-100上测试了我们的纠偏拒绝(RR)模块在包括自适应攻击在内的几种攻击下的性能，并证明了RR模块在提高鲁棒性方面与不同的对手训练框架兼容，并且几乎不需要额外的计算量。代码可在https://github.com/P2333/Rectified-Rejection.上获得



## **5. Local Convolutions Cause an Implicit Bias towards High Frequency Adversarial Examples**

局部卷积导致对高频对抗性例子的隐性偏向 stat.ML

20 pages, 11 figures, 12 Tables

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2006.11440v4)

**Authors**: Josue Ortega Caro, Yilong Ju, Ryan Pyle, Sourav Dey, Wieland Brendel, Fabio Anselmi, Ankit Patel

**Abstracts**: Adversarial Attacks are still a significant challenge for neural networks. Recent work has shown that adversarial perturbations typically contain high-frequency features, but the root cause of this phenomenon remains unknown. Inspired by theoretical work on linear full-width convolutional models, we hypothesize that the local (i.e. bounded-width) convolutional operations commonly used in current neural networks are implicitly biased to learn high frequency features, and that this is one of the root causes of high frequency adversarial examples. To test this hypothesis, we analyzed the impact of different choices of linear and nonlinear architectures on the implicit bias of the learned features and the adversarial perturbations, in both spatial and frequency domains. We find that the high-frequency adversarial perturbations are critically dependent on the convolution operation because the spatially-limited nature of local convolutions induces an implicit bias towards high frequency features. The explanation for the latter involves the Fourier Uncertainty Principle: a spatially-limited (local in the space domain) filter cannot also be frequency-limited (local in the frequency domain). Furthermore, using larger convolution kernel sizes or avoiding convolutions (e.g. by using Vision Transformers architecture) significantly reduces this high frequency bias, but not the overall susceptibility to attacks. Looking forward, our work strongly suggests that understanding and controlling the implicit bias of architectures will be essential for achieving adversarial robustness.

摘要: 对抗性攻击仍然是神经网络面临的重大挑战。最近的研究表明，对抗性扰动通常包含高频特征，但这种现象的根本原因尚不清楚。受线性全宽度卷积模型理论工作的启发，我们假设当前神经网络中常用的局部(即有界宽度)卷积运算隐含地偏向于学习高频特征，这是造成高频对抗性例子的根本原因之一。为了验证这一假设，我们在空间域和频域分析了线性和非线性结构的不同选择对学习特征的内隐偏差和对抗性扰动的影响。我们发现，高频对抗性扰动严重依赖于卷积运算，因为局部卷积的空间有限性质导致了对高频特征的隐式偏差。对后者的解释涉及到傅立叶测不准原理：空间受限(空间域中的局部)过滤不能也是频率受限的(频域中的局部)。此外，使用更大的卷积核大小或避免卷积(例如，通过使用Vision Transformers架构)可以显著降低这种高频偏差，但不会显著降低对攻击的总体易感性。展望未来，我们的工作强烈表明，理解和控制体系结构的隐含偏差将是实现对抗性健壮性的关键。



## **6. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2009.04131v4)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **7. Saliency Diversified Deep Ensemble for Robustness to Adversaries**

显著多样化的深度集成，增强了对对手的健壮性 cs.CV

Accepted to AAAI Workshop on Adversarial Machine Learning and Beyond  2022

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03615v1)

**Authors**: Alex Bogun, Dimche Kostadinov, Damian Borth

**Abstracts**: Deep learning models have shown incredible performance on numerous image recognition, classification, and reconstruction tasks. Although very appealing and valuable due to their predictive capabilities, one common threat remains challenging to resolve. A specifically trained attacker can introduce malicious input perturbations to fool the network, thus causing potentially harmful mispredictions. Moreover, these attacks can succeed when the adversary has full access to the target model (white-box) and even when such access is limited (black-box setting). The ensemble of models can protect against such attacks but might be brittle under shared vulnerabilities in its members (attack transferability). To that end, this work proposes a novel diversity-promoting learning approach for the deep ensembles. The idea is to promote saliency map diversity (SMD) on ensemble members to prevent the attacker from targeting all ensemble members at once by introducing an additional term in our learning objective. During training, this helps us minimize the alignment between model saliencies to reduce shared member vulnerabilities and, thus, increase ensemble robustness to adversaries. We empirically show a reduced transferability between ensemble members and improved performance compared to the state-of-the-art ensemble defense against medium and high strength white-box attacks. In addition, we demonstrate that our approach combined with existing methods outperforms state-of-the-art ensemble algorithms for defense under white-box and black-box attacks.

摘要: 深度学习模型在众多的图像识别、分类和重建任务中表现出了令人难以置信的性能。尽管它们的预测能力非常有吸引力和价值，但一个共同的威胁仍然难以解决。经过特殊训练的攻击者可以引入恶意输入扰动来欺骗网络，从而导致潜在的有害误判。此外，当对手拥有对目标模型的完全访问权限(白盒)，甚至当此类访问受限(黑盒设置)时，这些攻击也可能成功。模型集合可以防止此类攻击，但在其成员的共享漏洞(攻击可转移性)下可能会变得脆弱。为此，本工作提出了一种新的促进深度集成的多样性学习方法。我们的想法是通过在我们的学习目标中引入一个额外的术语来促进集合成员的显著地图多样性(SMD)，以防止攻击者一次针对所有集合成员。在训练过程中，这有助于我们最大限度地减少模型显著性之间的对齐，以减少共享的成员漏洞，从而提高对对手的整体健壮性。我们的经验表明，与针对中高强度白盒攻击的最先进的组合防御相比，组合成员之间的可传递性降低了，性能得到了提高。此外，我们还证明了我们的方法与现有方法相结合，在白盒和黑盒攻击下的防御性能优于最先进的集成算法。



## **8. Membership Inference Attacks From First Principles**

基于第一性原理的隶属度推理攻击 cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03570v1)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

摘要: 成员关系推断攻击允许对手查询经过训练的机器学习模型，以预测特定示例是否包含在该模型的训练数据集中。目前使用平均案例“准确性”度量来评估这些攻击，这些度量无法表征攻击是否可以自信地识别训练集的任何成员。我们认为，应该通过计算低(例如<0.1%)假阳性率下的真阳性率来评估攻击，并且发现大多数以前的攻击在这样评估时表现很差。为了解决这个问题，我们开发了一种似然比攻击(LIRA)，它仔细地结合了文献中的多种想法。我们的攻击以较低的误报率提高了10倍的威力，并且严格控制了先前对现有指标的攻击。



## **9. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

基于决策的基于补丁对抗性去除的视觉变形金刚黑盒攻击 cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03492v1)

**Authors**: Yucheng Shi, Yahong Han

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Deep Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the existing decision-based attacks for CNNs ignore the difference in noise sensitivity between different regions of the image, which affects the efficiency of noise compression. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we propose a new decision-based black-box attack against ViTs termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on ImageNet-21k, ILSVRC-2012, and Tiny-Imagenet datasets demonstrate that PAR achieves a much lower magnitude of perturbation on average with the same number of queries.

摘要: 与深度卷积神经网络(CNNs)相比，视觉变换器(VITS)表现出令人印象深刻的性能和更强的对抗鲁棒性。一方面，VITS对单个斑块之间全局交互的关注降低了图像的局部噪声敏感度。另一方面，现有的基于决策的CNN攻击忽略了图像不同区域之间噪声敏感性的差异，影响了噪声压缩的效率。因此，当目标模型只能查询时，验证VITS的黑箱对抗鲁棒性仍然是一个具有挑战性的问题。本文提出了一种新的基于决策的针对VITS的黑盒攻击，称为补丁对抗性删除(PAR)。PAR通过从粗到细的搜索过程将图像分成多个块，并分别压缩每个块上的噪声。PAR记录每个面片的噪声大小和噪声敏感度，并选择查询值最高的面片进行噪声压缩。此外，PAR可以作为其他基于判决的攻击的噪声初始化方法，在不引入额外计算的情况下提高VITS和CNN的噪声压缩效率。在ImageNet-21k、ILSVRC-2012和Tiny-Imagenet数据集上的大量实验表明，在相同的查询数量下，PAR的平均扰动幅度要小得多。



## **10. BDFA: A Blind Data Adversarial Bit-flip Attack on Deep Neural Networks**

BDFA：一种基于深度神经网络的盲数据对抗性比特翻转攻击 cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03477v1)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Zhenman Fang, Lesley Shannon

**Abstracts**: Adversarial bit-flip attack (BFA) on Neural Network weights can result in catastrophic accuracy degradation by flipping a very small number of bits. A major drawback of prior bit flip attack techniques is their reliance on test data. This is frequently not possible for applications that contain sensitive or proprietary data. In this paper, we propose Blind Data Adversarial Bit-flip Attack (BDFA), a novel technique to enable BFA without any access to the training or testing data. This is achieved by optimizing for a synthetic dataset, which is engineered to match the statistics of batch normalization across different layers of the network and the targeted label. Experimental results show that BDFA could decrease the accuracy of ResNet50 significantly from 75.96\% to 13.94\% with only 4 bits flips.

摘要: 对神经网络权重的对抗性比特翻转攻击(BFA)可以通过翻转非常少量的比特来导致灾难性的精度降低。现有比特翻转攻击技术的主要缺点是它们对测试数据的依赖。对于包含敏感或专有数据的应用程序而言，这通常是不可能的。在本文中，我们提出了盲数据对抗比特翻转攻击(BDFA)，这是一种新的技术，可以在不访问任何训练或测试数据的情况下实现BFA。这是通过对合成数据集进行优化来实现的，该合成数据集被设计为匹配跨网络的不同层和目标标签的批归一化的统计数据。实验结果表明，只需4位翻转，BDFA就能将ResNet50的精度从75.96降到13.94。



## **11. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

GasHis-Transformer：一种用于胃组织病理图像分类的多尺度视觉变换方法 cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2104.14528v5)

**Authors**: Haoyuan Chen, Chen Li, Xiaoyan Li, Ge Wang, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Yudong Yao, Yueyang Teng, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.

摘要: 现有的胃癌诊断深度学习方法普遍采用卷积神经网络。近年来，视觉变压器因其高性能和高效率而备受关注，但其应用大多集中在计算机视觉领域。本文提出了一种用于胃组织病理图像分类(GHIC)的多尺度视觉转换器模型(简称GasHis-Transformer)，该模型能够自动将胃显微图像分类为异常和正常病例。GasHis-Transformer模型由两个关键模块组成：全局信息模块和局部信息模块，有效地提取组织病理学特征。在我们的实验中，一个公共的苏木精伊红(H&E)染色的胃组织病理学数据集以1：1：2的比例分为训练集、验证集和测试集，训练集、验证集和测试集的比例为1：1：2。应用GasHis-Transformer模型估计胃组织病理学数据集的准确率、召回率、F1得分和准确率分别为98.0%、100.0%、96.0%和98.0%。此外，还对GasHis-Transformer的稳健性进行了关键研究，添加了10种不同的噪声，包括4种对抗性攻击和6种常规图像噪声。另外，利用620幅异常图像对GasHis-Transformer的胃肠道肿瘤识别性能进行了有临床意义的测试，准确率达到96.8%。最后，在淋巴瘤图像数据集和乳腺癌数据集上对H&E和免疫组织化学染色图像的泛化能力进行了比较研究，得到了可比的F1得分(85.6%和82.8%)和准确率(83.9%和89.4%)。总之，GasHisTransformer表现出很高的分类性能，并在GHIC任务中显示出巨大的潜力。



## **12. Introducing the DOME Activation Functions**

介绍穹顶激活功能 cs.LG

16 pages, 9 figures

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2109.14798v2)

**Authors**: Mohamed E. Hussein, Wael AbdAlmageed

**Abstracts**: In this paper, we introduce a novel non-linear activation function that spontaneously induces class-compactness and regularization in the embedding space of neural networks. The function is dubbed DOME for Difference Of Mirrored Exponential terms. The basic form of the function can replace the sigmoid or the hyperbolic tangent functions as an output activation function for binary classification problems. The function can also be extended to the case of multi-class classification, and used as an alternative to the standard softmax function. It can also be further generalized to take more flexible shapes suitable for intermediate layers of a network. We empirically demonstrate the properties of the function. We also show that models using the function exhibit extra robustness against adversarial attacks.

摘要: 本文介绍了一种新的非线性激活函数，它在神经网络的嵌入空间中自发地诱导类紧性和正则化。由于镜像指数项的差异，该函数被称为穹顶。该函数的基本形式可以代替Sigmoid或双曲正切函数作为二进制分类问题的输出激活函数。该功能还可以扩展到多类分类的情况，并用作标准Softmax功能的替代。它还可以被进一步推广，以采取适合于网络中间层的更灵活的形状。我们实证地证明了该函数的性质。我们还表明，使用该函数的模型在抵抗敌意攻击时表现出额外的鲁棒性。



## **13. Adversarial Attacks in Cooperative AI**

协作式人工智能中的对抗性攻击 cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.14833v2)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent work in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making incorrect decisions. In addition, some past research in cooperative AI has relied on new notions of representations, like public beliefs, to accelerate the learning of optimally cooperative behavior. Hence, cooperative AI might introduce new weaknesses not investigated in previous machine learning research. In this paper, our contributions include: (1) arguing that three algorithms inspired by human-like social intelligence introduce new vulnerabilities, unique to cooperative AI, that adversaries can exploit, and (2) an experiment showing that simple, adversarial perturbations on the agents' beliefs can negatively impact performance. This evidence points to the possibility that formal representations of social behavior are vulnerable to adversarial attacks.

摘要: 多智能体环境中的单智能体强化学习算法不能很好地促进协作。如果智能Agent要交互并共同工作来解决复杂问题，就需要针对不合作行为的方法，以便于多个Agent的训练。这是合作AI的目标。然而，最近在对抗性机器学习方面的工作表明，模型(例如，图像分类器)很容易被欺骗，从而做出不正确的决定。此外，过去对合作人工智能的一些研究依赖于新的表征概念，如公众信仰，以加速最佳合作行为的学习。因此，合作人工智能可能会引入以前的机器学习研究中没有研究的新弱点。在本文中，我们的贡献包括：(1)论证了三种受类人类社会智能启发的算法引入了新的漏洞，这些漏洞是合作人工智能所特有的，攻击者可以利用这些漏洞；(2)实验表明，对Agent信念的简单对抗性扰动可能会对性能产生负面影响。这一证据表明，社交行为的正式表述很容易受到敌意攻击。



## **14. Shape Defense Against Adversarial Attacks**

塑造对敌方攻击的防御 cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2008.13336v3)

**Authors**: Ali Borji

**Abstracts**: Humans rely heavily on shape information to recognize objects. Conversely, convolutional neural networks (CNNs) are biased more towards texture. This is perhaps the main reason why CNNs are vulnerable to adversarial examples. Here, we explore how shape bias can be incorporated into CNNs to improve their robustness. Two algorithms are proposed, based on the observation that edges are invariant to moderate imperceptible perturbations. In the first one, a classifier is adversarially trained on images with the edge map as an additional channel. At inference time, the edge map is recomputed and concatenated to the image. In the second algorithm, a conditional GAN is trained to translate the edge maps, from clean and/or perturbed images, into clean images. Inference is done over the generated image corresponding to the input's edge map. Extensive experiments over 10 datasets demonstrate the effectiveness of the proposed algorithms against FGSM and $\ell_\infty$ PGD-40 attacks. Further, we show that a) edge information can also benefit other adversarial training methods, and b) CNNs trained on edge-augmented inputs are more robust against natural image corruptions such as motion blur, impulse noise and JPEG compression, than CNNs trained solely on RGB images. From a broader perspective, our study suggests that CNNs do not adequately account for image structures that are crucial for robustness. Code is available at:~\url{https://github.com/aliborji/Shapedefense.git}.

摘要: 人类在很大程度上依赖于形状信息来识别物体。相反，卷积神经网络(CNN)更偏向于纹理。这可能是CNN容易受到敌意例子攻击的主要原因。在这里，我们将探索如何将形状偏差融入到CNN中，以提高其鲁棒性。基于边缘不变到适度的不可察觉扰动这一观察结果，提出了两种算法。在第一种方法中，分类器以边缘图作为附加通道对图像进行对抗性训练。在推断时，重新计算边缘映射并将其连接到图像。在第二种算法中，训练条件GAN以将边缘图从干净和/或扰动的图像转换成干净的图像。对与输入的边缘映射相对应的生成图像进行推断。在10个数据集上的大量实验证明了所提算法对FGSM和$\ELL_\INFTY$PGD-40攻击的有效性。此外，我们还表明：a)边缘信息也可以用于其他对抗性训练方法；b)与仅训练在RGB图像上的CNN相比，基于边缘增强输入训练的CNN对运动模糊、脉冲噪声和JPEG压缩等自然图像的破坏具有更强的鲁棒性。从更广泛的角度来看，我们的研究表明，CNN没有充分考虑对鲁棒性至关重要的图像结构。代码可用at：~\url{https://github.com/aliborji/Shapedefense.git}.



## **15. Adversarial Machine Learning In Network Intrusion Detection Domain: A Systematic Review**

网络入侵检测领域的对抗性机器学习研究综述 cs.CR

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03315v1)

**Authors**: Huda Ali Alatwi, Charles Morisset

**Abstracts**: Due to their massive success in various domains, deep learning techniques are increasingly used to design network intrusion detection solutions that detect and mitigate unknown and known attacks with high accuracy detection rates and minimal feature engineering. However, it has been found that deep learning models are vulnerable to data instances that can mislead the model to make incorrect classification decisions so-called (adversarial examples). Such vulnerability allows attackers to target NIDSs by adding small crafty perturbations to the malicious traffic to evade detection and disrupt the system's critical functionalities. The problem of deep adversarial learning has been extensively studied in the computer vision domain; however, it is still an area of open research in network security applications. Therefore, this survey explores the researches that employ different aspects of adversarial machine learning in the area of network intrusion detection in order to provide directions for potential solutions. First, the surveyed studies are categorized based on their contribution to generating adversarial examples, evaluating the robustness of ML-based NIDs towards adversarial examples, and defending these models against such attacks. Second, we highlight the characteristics identified in the surveyed research. Furthermore, we discuss the applicability of the existing generic adversarial attacks for the NIDS domain, the feasibility of launching the proposed attacks in real-world scenarios, and the limitations of the existing mitigation solutions.

摘要: 由于深度学习技术在各个领域取得了巨大的成功，越来越多的人将深度学习技术用于设计网络入侵检测解决方案，这些解决方案能够以较高的准确率和最小的特征工程来检测和缓解未知和已知的攻击。然而，人们发现深度学习模型容易受到数据实例的影响，这些数据实例可能会误导模型做出不正确的分类决策，即所谓的(对抗性示例)。这样的漏洞使得攻击者能够通过向恶意通信量添加小而狡猾的干扰来瞄准NIDS，以逃避检测并破坏系统的关键功能。深度对抗性学习问题在计算机视觉领域得到了广泛的研究，但在网络安全应用中仍是一个开放的研究领域。因此，本综述探讨了在网络入侵检测领域应用对抗性机器学习的不同方面的研究，以期为潜在的解决方案提供方向。首先，调查的研究根据它们在生成对抗性实例、评估基于ML的NID对对抗性实例的健壮性以及保护这些模型免受此类攻击方面的贡献进行分类。其次，我们突出了调查研究中确定的特点。此外，我们还讨论了现有针对NIDS域的通用对抗性攻击的适用性、在现实场景中发起攻击的可行性以及现有缓解方案的局限性。



## **16. Context-Aware Transfer Attacks for Object Detection**

面向对象检测的上下文感知传输攻击 cs.CV

accepted to AAAI 2022

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03223v1)

**Authors**: Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to $20$ percentage points improvement in performance compared to the other state-of-the-art methods.

摘要: 近年来，针对图像分类器的黑盒传输攻击得到了广泛的研究。相比之下，针对目标检测器的传输攻击研究进展甚微。对象检测器对图像进行整体观察，并且一个对象(或其缺失)的检测通常取决于场景中的其他对象。这使得这类检测器固有的上下文感知和敌意攻击比那些针对图像分类器的攻击更具挑战性。本文提出了一种生成对象检测器上下文感知攻击的新方法。通过使用对象的共现及其相对位置和大小作为上下文信息，我们可以成功地生成具有针对性的误分类攻击，从而在黑盒对象检测器上获得比现有技术更高的传输成功率。我们使用Pascal VOC和MS Coco数据集的图像在各种对象探测器上测试了我们的方法，与其他最先进的方法相比，性能提高了多达20个百分点。



## **17. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

利用自监督学习提高说话人确认的对抗性 cs.SD

Accepted by TASLP

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2106.00273v3)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.

摘要: 以往的研究表明，自动说话人验证(ASV)很容易受到恶意欺骗攻击，如重放、合成语音以及最近出现的敌意攻击。人们一直致力于保护ASV免受重播和合成语音的攻击，然而，只探索了几种方法来应对对抗性攻击。现有的撞击对抗性攻击方法都需要生成对抗性样本的知识，但是防御者要知道野外攻击者使用的确切攻击算法是不切实际的。这项工作是第一批在不知道具体攻击算法的情况下对ASV进行对抗性防御的工作之一。受自监督学习模型(SSLMs)减少输入表面噪声和从中断样本中重构干净样本等优点的启发，本文将对抗性扰动视为一种噪声，利用SSLMs对ASV进行对抗性防御。具体地说，我们提出从两个角度进行对抗性防御：1)对抗性扰动净化和2)对抗性扰动检测。实验结果表明，我们的检测模块通过检测敌意样本，有效地屏蔽了ASV，准确率在80%左右。此外，由于ASV的对抗防御性能没有统一的评价指标，本文还考虑了基于净化和基于检测的方法，形式化了对抗防御的评价指标。我们真诚地鼓励今后的工作在拟议的评价框架基础上对其方法进行基准。



## **18. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

DNN模型的对抗性范例检测：综述与实验比较 cs.CV

To be published on Artificial Intelligence Review journal (after  minor revision)

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2105.00203v3)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.

摘要: 深度学习(DL)在许多与人类相关的任务中取得了巨大的成功，这使得它被许多基于计算机视觉的应用所采用，如安全监控系统、自动驾驶汽车和医疗保健。此类安全关键型应用程序一旦具备了克服安全关键型挑战的能力，就必须为成功部署画上句号。在这些挑战中，包括防御或/和检测对抗性示例(AEs)。攻击者可以小心翼翼地制造称为扰动的小噪音，通常是难以察觉的，并将其添加到干净的图像中，以生成AE。AE的目的是愚弄DL模型，使其成为DL应用程序的潜在风险。文献中提出了许多测试时间逃避攻击和对策，即防御或检测方法。此外，很少有综述和调查发表，从理论上给出了威胁的分类和对策方法，而对声发射检测方法的关注较少。本文以图像分类任务为研究对象，对神经网络分类器测试时间逃避攻击的检测方法进行了综述。对这些方法进行了详细的讨论，并给出了在四个数据集上的不同场景下八个最先进检测器的实验结果。我们还对这一研究方向提出了潜在的挑战和未来的展望。



## **19. Robust Person Re-identification with Multi-Modal Joint Defence**

基于多模态联合防御的鲁棒人物再识别 cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.09571v2)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.

摘要: 基于度量学习的人物识别(ReID)系统继承了深层神经网络(DNNs)易被恶意度量攻击欺骗的弱点。现有的工作主要依靠对抗性训练进行度量防御，更多的方法还没有得到充分的研究。通过研究攻击对底层特征的影响，提出了有针对性的度量攻击方法和防御方法。在度量攻击方面，我们利用局部颜色偏差来构造输入的类内变异来攻击颜色特征。在度量防御方面，我们提出了一种包括主动防御和被动防御两部分的联合防御方法。主动防御通过从多模态图像构造不同的输入来增强模型对颜色变化的鲁棒性和跨多模态的结构关系的学习，而被动防御通过迂回缩放利用结构特征在变化的像素空间中的不变性来保留结构特征，同时消除一些对抗性噪声。大量实验表明，与现有的对抗性度量防御方法相比，本文提出的联合防御方法不仅可以同时防御多个攻击，而且没有显着降低模型的泛化能力。代码可在https://github.com/finger-monkey/multi-modal_joint_defence.上获得



## **20. ML Attack Models: Adversarial Attacks and Data Poisoning Attacks**

ML攻击模型：对抗性攻击和数据中毒攻击 cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02797v1)

**Authors**: Jing Lin, Long Dang, Mohamed Rahouti, Kaiqi Xiong

**Abstracts**: Many state-of-the-art ML models have outperformed humans in various tasks such as image classification. With such outstanding performance, ML models are widely used today. However, the existence of adversarial attacks and data poisoning attacks really questions the robustness of ML models. For instance, Engstrom et al. demonstrated that state-of-the-art image classifiers could be easily fooled by a small rotation on an arbitrary image. As ML systems are being increasingly integrated into safety and security-sensitive applications, adversarial attacks and data poisoning attacks pose a considerable threat. This chapter focuses on the two broad and important areas of ML security: adversarial attacks and data poisoning attacks.

摘要: 许多最先进的ML模型在图像分类等各种任务中的表现都超过了人类。ML模型以其出色的性能在今天得到了广泛的应用。然而，敌意攻击和数据中毒攻击的存在确实对ML模型的稳健性提出了质疑。例如，Engstrom等人。展示了最先进的图像分类器可以很容易地被任意图像上的小旋转所愚弄。随着ML系统越来越多地集成到安全和安全敏感的应用程序中，对抗性攻击和数据中毒攻击构成了相当大的威胁。本章重点介绍ML安全的两个广泛而重要的领域：对抗性攻击和数据中毒攻击。



## **21. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v4)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的搜索能力、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过四个测试函数进行了验证。仿真结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。最后，将该算法应用于神经网络的对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **22. Staring Down the Digital Fulda Gap Path Dependency as a Cyber Defense Vulnerability**

向下看数字富尔达缺口路径依赖是一个网络防御漏洞 cs.CY

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02773v1)

**Authors**: Jan Kallberg

**Abstracts**: Academia, homeland security, defense, and media have accepted the perception that critical infrastructure in a future cyber war cyber conflict is the main gateway for a massive cyber assault on the U.S. The question is not if the assumption is correct or not, the question is instead of how did we arrive at that assumption. The cyber paradigm considers critical infrastructure the primary attack vector for future cyber conflicts. The national vulnerability embedded in critical infrastructure is given a position in the cyber discourse as close to an unquestionable truth as a natural law.   The American reaction to Sept. 11, and any attack on U.S. soil, hint to an adversary that attacking critical infrastructure to create hardship for the population could work contrary to the intended softening of the will to resist foreign influence. It is more likely that attacks that affect the general population instead strengthen the will to resist and fight, similar to the British reaction to the German bombing campaign Blitzen in 1940. We cannot rule out attacks that affect the general population, but there are not enough adversarial offensive capabilities to attack all 16 critical infrastructure sectors and gain strategic momentum. An adversary has limited cyberattack capabilities and needs to prioritize cyber targets that are aligned with the overall strategy. Logically, an adversary will focus their OCO on operations that has national security implications and support their military operations by denying, degrading, and confusing the U.S. information environment and U.S. cyber assets.

摘要: 学术界、国土安全、国防和媒体已经接受了这样的看法，即未来网络战中的关键基础设施网络冲突是针对美国的大规模网络攻击的主要门户。问题不是假设是否正确，而是我们如何得出这个假设。网络范式认为关键基础设施是未来网络冲突的主要攻击载体。关键基础设施中嵌入的国家脆弱性在网络话语中被赋予了与自然法一样接近毋庸置疑的真理的地位。美国对9·11事件的反应。11，以及对美国领土的任何袭击，都暗示着对手，攻击关键基础设施给人民带来困难，可能与抵制外国影响的意愿软化的意图背道而驰。更有可能的是，影响到普通民众的袭击反而增强了抵抗和战斗的意志，类似于英国对1940年德国轰炸行动Blitzen的反应。我们不能排除影响到普通民众的袭击，但没有足够的对抗性进攻能力来攻击所有16个关键基础设施部门，并获得战略势头。对手的网络攻击能力有限，需要优先考虑与整体战略一致的网络目标。从逻辑上讲，对手将把他们的OCO集中在影响国家安全的行动上，并通过否认、贬低和混淆美国信息环境和美国网络资产来支持他们的军事行动。



## **23. Label-Only Membership Inference Attacks**

仅标签成员关系推理攻击 cs.CR

16 pages, 11 figures, 2 tables Revision 2: 19 pages, 12 figures, 3  tables. Improved text and additional experiments. Final ICML paper

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2007.14321v3)

**Authors**: Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot

**Abstracts**: Membership inference attacks are one of the simplest forms of privacy leakage for machine learning models: given a data point and model, determine whether the point was used to train the model. Existing membership inference attacks exploit models' abnormal confidence when queried on their training data. These attacks do not apply if the adversary only gets access to models' predicted labels, without a confidence measure. In this paper, we introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples. We empirically show that our label-only membership inference attacks perform on par with prior attacks that required access to model confidences. We further demonstrate that label-only attacks break multiple defenses against membership inference attacks that (implicitly or explicitly) rely on a phenomenon we call confidence masking. These defenses modify a model's confidence scores in order to thwart attacks, but leave the model's predicted labels unchanged. Our label-only attacks demonstrate that confidence-masking is not a viable defense strategy against membership inference. Finally, we investigate worst-case label-only attacks, that infer membership for a small number of outlier data points. We show that label-only attacks also match confidence-based attacks in this setting. We find that training models with differential privacy and (strong) L2 regularization are the only known defense strategies that successfully prevents all attacks. This remains true even when the differential privacy budget is too high to offer meaningful provable guarantees.

摘要: 成员关系推理攻击是机器学习模型隐私泄露的最简单形式之一：给定一个数据点和模型，确定该点是否被用来训练该模型。现有的隶属度推理攻击利用模型在查询训练数据时的异常置信度。如果对手只能访问模型的预测标签，而没有置信度度量，则这些攻击不适用。在本文中，我们引入了仅标签成员关系推理攻击。我们的攻击不依赖于置信度分数，而是评估模型的预测标签在扰动下的鲁棒性，以获得细粒度的成员资格信号。这些扰动包括常见的数据扩充或对抗性示例。我们的经验表明，我们的仅标签成员关系推理攻击的性能与之前需要访问模型可信度的攻击相当。我们进一步证明，仅标签攻击打破了对(隐式或显式)依赖于我们称为置信度掩蔽现象的成员关系推断攻击的多个防御。这些防御措施修改模型的置信度分数以阻止攻击，但保持模型的预测标签不变。我们的仅标签攻击表明，置信度掩蔽不是一种可行的针对成员关系推断的防御策略。最后，我们研究了最坏情况下的仅标签攻击，即推断少量离群点的成员资格。我们表明，在此设置下，仅标签攻击也与基于置信度的攻击相匹配。我们发现，具有差异隐私和(强)L2正则化的训练模型是唯一已知的成功阻止所有攻击的防御策略。即使差别隐私预算太高，无法提供有意义的、可证明的保证，这一点仍然成立。



## **24. Learning Swarm Interaction Dynamics from Density Evolution**

从密度演化中学习群体相互作用动力学 eess.SY

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02675v1)

**Authors**: Christos Mavridis, Amoolya Tirumalai, John Baras

**Abstracts**: We consider the problem of understanding the coordinated movements of biological or artificial swarms. In this regard, we propose a learning scheme to estimate the coordination laws of the interacting agents from observations of the swarm's density over time. We describe the dynamics of the swarm based on pairwise interactions according to a Cucker-Smale flocking model, and express the swarm's density evolution as the solution to a system of mean-field hydrodynamic equations. We propose a new family of parametric functions to model the pairwise interactions, which allows for the mean-field macroscopic system of integro-differential equations to be efficiently solved as an augmented system of PDEs. Finally, we incorporate the augmented system in an iterative optimization scheme to learn the dynamics of the interacting agents from observations of the swarm's density evolution over time. The results of this work can offer an alternative approach to study how animal flocks coordinate, create new control schemes for large networked systems, and serve as a central part of defense mechanisms against adversarial drone attacks.

摘要: 我们考虑理解生物或人造蜂群的协调运动的问题。在这方面，我们提出了一种学习方案，通过观察种群密度随时间的变化来估计相互作用Agent的协调规律。我们根据Cucker-Smer群集模型描述了基于成对相互作用的群体动力学，并将群体密度演化表示为平均场流体动力学方程组的解。我们提出了一族新的参数函数族来模拟两两相互作用，使得平均场宏观积分微分方程组可以作为一个增广的偏微分方程组有效地求解。最后，我们将增广系统结合到迭代优化方案中，通过观察种群密度随时间的演变来学习交互Agent的动态。这项工作的结果可以提供另一种方法来研究动物群是如何协调的，为大型网络系统创造新的控制方案，并作为对抗无人机攻击的防御机制的核心部分。



## **25. Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness**

随机本地赢家通吃网络实现深刻的对手鲁棒性 cs.LG

Bayesian Deep Learning Workshop, NeurIPS 2021

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02671v1)

**Authors**: Konstantinos P. Panousis, Sotirios Chatzis, Sergios Theodoridis

**Abstracts**: This work explores the potency of stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA), against powerful (gradient-based) white-box and black-box adversarial attacks; we especially focus on Adversarial Training settings. In our work, we replace the conventional ReLU-based nonlinearities with blocks comprising locally and stochastically competing linear units. The output of each network layer now yields a sparse output, depending on the outcome of winner sampling in each block. We rely on the Variational Bayesian framework for training and inference; we incorporate conventional PGD-based adversarial training arguments to increase the overall adversarial robustness. As we experimentally show, the arising networks yield state-of-the-art robustness against powerful adversarial attacks while retaining very high classification rate in the benign case.

摘要: 这项工作探索了基于随机竞争的激活，即随机局部赢家通吃(LWTA)，对抗强大的(基于梯度的)白盒和黑盒对抗性攻击的有效性；我们特别关注对抗性训练环境。在我们的工作中，我们用由局部和随机竞争的线性单元组成的块来代替传统的基于REU的非线性。现在，每个网络层的输出都会产生稀疏输出，具体取决于每个挡路中获胜者采样的结果。我们依靠变分贝叶斯框架进行训练和推理；我们结合了传统的基于PGD的对抗性训练论据，以增加对抗性的整体健壮性。正如我们的实验所表明的那样，出现的网络对强大的对手攻击产生了最先进的健壮性，同时在良性情况下保持了非常高的分类率。



## **26. Formalizing and Estimating Distribution Inference Risks**

配电推理风险的形式化与估计 cs.LG

Shorter version of work available at arXiv:2106.03699 Update: New  version with more theoretical results and a deeper exploration of results

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v4)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型基于私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即，生成捕获有关分布的统计属性的模型。在Yeom等人的成员关系推理框架的启发下，我们提出了分布推理攻击的形式化定义，该定义足够通用，可以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均节点度或聚类系数。为了了解分布推理风险，我们引入了一个度量，通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，对观察到的泄漏进行量化。我们报告了使用新颖的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。



## **27. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

ADV-4-ADV：通过对抗性领域适应挫败不断变化的对抗性扰动 cs.CV

9 pages

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.00428v2)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.

摘要: 虽然对抗性训练对对抗特定的对抗性干扰是有用的，但事实证明，它们也不能有效地概括出与用于训练的攻击不同的攻击。然而，我们观察到这种低效与领域适应性有内在的联系，这是深度学习中的另一个关键问题，对抗性领域适应似乎是一个有希望的解决方案。因此，我们提出了ADV-4-ADV作为一种新的对抗性训练方法，旨在保持对不可见的对抗性扰动的鲁棒性。从本质上讲，ADV-4-ADV将遭受不同扰动的攻击视为不同的域，并利用敌对域自适应的能力，旨在去除域/攻击特定的特征。这迫使训练后的模型学习健壮的领域不变表示，进而增强其泛化能力。在Fashion-MNIST、SVHN、CIFAR-10和CIFAR-100上的广泛评估表明，由ADV-4-ADV基于简单攻击(例如FGSM)构造的样本训练的模型可以推广到更高级的攻击(例如PGD)，并且性能超过了在这些数据集上的最新建议。



## **28. Statically Detecting Adversarial Malware through Randomised Chaining**

通过随机链静态检测敌意恶意软件 cs.CR

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2111.14037v2)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.

摘要: 随着恶意软件攻击的快速增长，越来越多的反病毒开发人员考虑将机器学习技术部署到他们的产品中。近年来，研究人员和开发人员发布了各种基于机器学习的恶意软件检测高精度检测器。虽然有许多基于机器学习的恶意软件检测器可用，但它们面临着各种机器学习目标攻击，包括逃避和敌意攻击。该项目探讨了敌意实例如何以及为什么躲避恶意软件检测器，然后提出了一种随机链接的方法来静态防御敌意恶意软件。这项研究对于打击相关的恶意软件网络犯罪至关重要。



## **29. Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing**

逆稳健假设检验的广义似然比检验 stat.ML

Submitted to the IEEE Transactions on Signal Processing

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.02209v1)

**Authors**: Bhagyashree Puranik, Upamanyu Madhow, Ramtin Pedarsani

**Abstracts**: Machine learning models are known to be susceptible to adversarial attacks which can cause misclassification by introducing small but well designed perturbations. In this paper, we consider a classical hypothesis testing problem in order to develop fundamental insight into defending against such adversarial perturbations. We interpret an adversarial perturbation as a nuisance parameter, and propose a defense based on applying the generalized likelihood ratio test (GLRT) to the resulting composite hypothesis testing problem, jointly estimating the class of interest and the adversarial perturbation. While the GLRT approach is applicable to general multi-class hypothesis testing, we first evaluate it for binary hypothesis testing in white Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations, for which a known minimax defense optimizing for the worst-case attack provides a benchmark. We derive the worst-case attack for the GLRT defense, and show that its asymptotic performance (as the dimension of the data increases) approaches that of the minimax defense. For non-asymptotic regimes, we show via simulations that the GLRT defense is competitive with the minimax approach under the worst-case attack, while yielding a better robustness-accuracy tradeoff under weaker attacks. We also illustrate the GLRT approach for a multi-class hypothesis testing problem, for which a minimax strategy is not known, evaluating its performance under both noise-agnostic and noise-aware adversarial settings, by providing a method to find optimal noise-aware attacks, and heuristics to find noise-agnostic attacks that are close to optimal in the high SNR regime.

摘要: 众所周知，机器学习模型容易受到敌意攻击，这种攻击可能会通过引入小但设计良好的扰动而导致误分类。在这篇文章中，我们考虑了一个经典的假设检验问题，以发展对这种敌对扰动的防御的基本见解。我们将敌意扰动解释为干扰参数，并提出了一种基于广义似然比检验(GLRT)的防御方法，将广义似然比检验应用于由此产生的复合假设检验问题，联合估计感兴趣的类别和对抗性扰动。虽然GLRT方法适用于一般的多类假设检验，但我们首先评估了它在高斯白噪声中范数有界的对抗扰动下的二元假设检验，一个已知的针对最坏情况攻击的极小极大防御优化提供了一个基准。我们推导了GLRT防御的最坏情况攻击，并证明了它的渐近性能(随着数据维数的增加)接近极小极大防御的渐近性能。对于非渐近体制，我们通过仿真表明，在最坏情况下，广义似然比防御是基于极小极大方法的好胜防御，而在较弱攻击下获得了较好的稳健性和准确性折衷。我们还举例说明了GLRT方法用于多类假设检验问题，对于未知的极小极大策略，通过提供一种寻找最优噪声感知攻击的方法和寻找在高信噪比条件下接近最优的噪声不可知攻击的启发式方法，来评估其在噪声不可知性和噪声感知对抗环境下的性能。



## **30. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线传感的对策 cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01967v1)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，今天无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过偷听标准通信信号，窃听者可以获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为对抗敌意无线传感的一种新的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行模糊处理。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **31. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

注意方框：$l_1$-针对图像分类器的稀疏对抗性攻击的APGD cs.LG

In ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.01208v2)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

摘要: 我们证明了当同时考虑象域$[0，1]^d$时，所建立的$l_1$投影梯度下降(PGD)攻击是次优的，因为它们没有考虑到有效的威胁模型是$l_1$球和$[0，1]^d$的交集。我们研究了该有效威胁模型的最陡下降步长的期望稀疏性，并证明了在该集合上的精确投影在计算上是可行的，并且产生了更好的性能。此外，我们还提出了一种自适应形式的PGD，即使在很小的迭代预算下也是非常有效的。我们得到的$l_1$-APGD是一个强白盒攻击，表明以前的工作高估了它们的$l_1$-稳健性。利用$l_1$-APGD进行对抗性训练，得到一个具有SOTA$l_1$-鲁棒性的鲁棒分类器。最后，我们将$l_1$-APGD和对$l_1$的Square攻击的改进结合成$l_1$-AutoAttack，这是一个攻击集合，它可靠地评估了$l_1$-ball与$[0，1]^d$相交的威胁模型的对手健壮性。



## **32. Graph Neural Networks Inspired by Classical Iterative Algorithms**

受经典迭代算法启发的图神经网络 cs.LG

accepted as long oral for ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.06064v4)

**Authors**: Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf

**Abstracts**: Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS.

摘要: 尽管图神经网络(GNN)最近取得了成功，但常见的体系结构通常表现出显著的局限性，包括对过度平滑、长范围依赖和伪边的敏感性，例如，由于图的异嗜性或敌意攻击而可能发生的情况。为了在一个简单透明的框架内至少部分解决这些问题，我们考虑了一族新的GNN层，它们被设计成模仿和集成两种经典迭代算法的更新规则，即最近梯度下降和迭代重加权最小二乘(IRLS)。前者定义了一个可扩展的基本GNN体系结构，该体系结构不受过度平滑的影响，同时通过允许任意传播步骤来捕获远程依赖关系。相反，后者产生了一种新的注意机制，该机制显式地锚定在潜在的端到端能量函数上，有助于相对于边缘不确定性的稳定性。当组合在一起时，我们得到了一个极其简单但健壮的模型，我们可以跨不同的场景进行评估，包括标准化的基准测试、对抗性干扰图、具有异质性的图以及涉及长范围依赖的图。在此过程中，我们将其与明确为各自任务设计的Sota GNN方法进行比较，以达到好胜或更高的节点分类精度。我们的代码可在https://github.com/FFTYYY/TWIRLS.获得



## **33. Blackbox Untargeted Adversarial Testing of Automatic Speech Recognition Systems**

自动语音识别系统的黑盒非目标对抗性测试 cs.SD

10 pages, 6 figures and 7 tables

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01821v1)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) systems are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the correctness of ASRS, we propose techniques that automatically generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. Much of the existing work on adversarial ASR testing focuses on targeted attacks, i.e generating audio samples given an output text. Targeted techniques are not portable, customised to the structure of DNNs (whitebox) within a specific ASR. In contrast, our method attacks the signal processing stage of the ASR pipeline that is shared across most ASRs. Additionally, we ensure the generated adversarial audio samples have no human audible difference by manipulating the acoustic signal using a psychoacoustic model that maintains the signal below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and three input audio datasets using the metrics - WER of output text, Similarity to original audio and attack Success Rate on different ASRs. We found our testing techniques were portable across ASRs, with the adversarial audio samples producing high Success Rates, WERs and Similarities to the original audio.

摘要: 自动语音识别(ASR)系统很普遍，特别是在用于语音导航和家用电器的语音控制的应用中。ASR的计算核心是深度神经网络(DNNs)，已经证明它们容易受到对手的干扰；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的正确性，我们提出了自动生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。对抗性ASR测试的大部分现有工作都集中在有针对性的攻击上，即在给定输出文本的情况下生成音频样本。目标技术不是便携的，不能根据特定ASR内的DNN(白盒)结构进行定制。相反，我们的方法攻击大多数ASR共享的ASR流水线的信号处理阶段。此外，我们通过使用将信号保持在人类感知阈值以下的心理声学模型来处理声音信号，以确保生成的敌意音频样本没有人耳可闻的差异。我们使用三个流行的ASR和三个输入音频数据集，使用输出文本的WER、与原始音频的相似度和对不同ASR的攻击成功率来评估我们的技术的可移植性和有效性。我们发现我们的测试技术在ASR之间是可移植的，敌意音频样本产生了很高的成功率，与原始音频有很大的相似之处。



## **34. Attack-Centric Approach for Evaluating Transferability of Adversarial Samples in Machine Learning Models**

机器学习模型中以攻击为中心评估敌方样本可转移性的方法 cs.LG

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01777v1)

**Authors**: Tochukwu Idika, Ismail Akturk

**Abstracts**: Transferability of adversarial samples became a serious concern due to their impact on the reliability of machine learning system deployments, as they find their way into many critical applications. Knowing factors that influence transferability of adversarial samples can assist experts to make informed decisions on how to build robust and reliable machine learning systems. The goal of this study is to provide insights on the mechanisms behind the transferability of adversarial samples through an attack-centric approach. This attack-centric perspective interprets how adversarial samples would transfer by assessing the impact of machine learning attacks (that generated them) on a given input dataset. To achieve this goal, we generated adversarial samples using attacker models and transferred these samples to victim models. We analyzed the behavior of adversarial samples on victim models and outlined four factors that can influence the transferability of adversarial samples. Although these factors are not necessarily exhaustive, they provide useful insights to researchers and practitioners of machine learning systems.

摘要: 敌意样本的可转移性成为一个严重的问题，因为它们会影响机器学习系统部署的可靠性，因为它们会进入许多关键应用程序。了解影响对抗性样本可转移性的因素可以帮助专家就如何建立健壮可靠的机器学习系统做出明智的决定。这项研究的目的是通过以攻击为中心的方法，对敌方样本的可转移性背后的机制提供洞察力。这种以攻击为中心的观点通过评估机器学习攻击(生成它们的)对给定输入数据集的影响，解释了敌意样本将如何传输。为了实现这一目标，我们使用攻击者模型生成对抗性样本，并将这些样本传输到受害者模型。我们分析了对抗性样本在受害者模型上的行为，并概述了影响对抗性样本可转移性的四个因素。虽然这些因素不一定是详尽的，但它们为机器学习系统的研究人员和实践者提供了有用的见解。



## **35. Single-Shot Black-Box Adversarial Attacks Against Malware Detectors: A Causal Language Model Approach**

针对恶意软件检测器的单发黑盒对抗性攻击：一种因果语言模型方法 cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01724v1)

**Authors**: James Lee Hu, Mohammadreza Ebrahimi, Hsinchun Chen

**Abstracts**: Deep Learning (DL)-based malware detectors are increasingly adopted for early detection of malicious behavior in cybersecurity. However, their sensitivity to adversarial malware variants has raised immense security concerns. Generating such adversarial variants by the defender is crucial to improving the resistance of DL-based malware detectors against them. This necessity has given rise to an emerging stream of machine learning research, Adversarial Malware example Generation (AMG), which aims to generate evasive adversarial malware variants that preserve the malicious functionality of a given malware. Within AMG research, black-box method has gained more attention than white-box methods. However, most black-box AMG methods require numerous interactions with the malware detectors to generate adversarial malware examples. Given that most malware detectors enforce a query limit, this could result in generating non-realistic adversarial examples that are likely to be detected in practice due to lack of stealth. In this study, we show that a novel DL-based causal language model enables single-shot evasion (i.e., with only one query to malware detector) by treating the content of the malware executable as a byte sequence and training a Generative Pre-Trained Transformer (GPT). Our proposed method, MalGPT, significantly outperformed the leading benchmark methods on a real-world malware dataset obtained from VirusTotal, achieving over 24.51\% evasion rate. MalGPT enables cybersecurity researchers to develop advanced defense capabilities by emulating large-scale realistic AMG.

摘要: 基于深度学习(DL)的恶意软件检测器越来越多地被用于网络安全中的恶意行为的早期检测。然而，他们对敌意恶意软件变体的敏感性引发了巨大的安全担忧。防御者生成这种敌意变体对于提高基于DL的恶意软件检测器对它们的抵抗力至关重要。这种必要性已经引起了一种新兴的机器学习研究流，即对抗性恶意软件示例生成(AMG)，其目的是生成保留给定恶意软件的恶意功能的闪避性对抗性恶意软件变体。在AMG研究中，黑盒方法比白盒方法更受关注。然而，大多数黑盒AMG方法需要与恶意软件检测器进行多次交互才能生成敌意恶意软件示例。鉴于大多数恶意软件检测器强制执行查询限制，这可能导致生成由于缺乏隐蔽性而很可能在实践中被检测到的不切实际的对抗性示例。在这项研究中，我们提出了一种新的基于DL的因果语言模型，通过将恶意软件可执行文件的内容视为一个字节序列，并训练一个生成式预训练转换器(GPT)，从而实现单发规避(即只需对恶意软件检测器进行一次查询)。我们提出的MalGPT方法在从VirusTotal获得的真实恶意软件数据集上的性能明显优于领先的基准测试方法，达到了24.51%以上的逃避率。MalGPT使网络安全研究人员能够通过模拟大规模现实AMG来开发高级防御能力。



## **36. Adversarial Attacks against a Satellite-borne Multispectral Cloud Detector**

针对星载多光谱云探测器的敌意攻击 cs.CV

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01723v1)

**Authors**: Andrew Du, Yee Wei Law, Michele Sasdelli, Bo Chen, Ken Clarke, Michael Brown, Tat-Jun Chin

**Abstracts**: Data collected by Earth-observing (EO) satellites are often afflicted by cloud cover. Detecting the presence of clouds -- which is increasingly done using deep learning -- is crucial preprocessing in EO applications. In fact, advanced EO satellites perform deep learning-based cloud detection on board the satellites and downlink only clear-sky data to save precious bandwidth. In this paper, we highlight the vulnerability of deep learning-based cloud detection towards adversarial attacks. By optimising an adversarial pattern and superimposing it into a cloudless scene, we bias the neural network into detecting clouds in the scene. Since the input spectra of cloud detectors include the non-visible bands, we generated our attacks in the multispectral domain. This opens up the potential of multi-objective attacks, specifically, adversarial biasing in the cloud-sensitive bands and visual camouflage in the visible bands. We also investigated mitigation strategies against the adversarial attacks. We hope our work further builds awareness of the potential of adversarial attacks in the EO community.

摘要: 地球观测卫星(EO)收集的数据经常受到云层的影响。检测云的存在--越来越多地使用深度学习来完成--在EO应用程序中是至关重要的预处理。事实上，先进的地球观测卫星在卫星上进行基于深度学习的云层探测，只下行晴空数据，以节省宝贵的带宽。在本文中，我们强调了基于深度学习的云检测对敌意攻击的脆弱性。通过优化对抗性模式并将其叠加到无云场景中，我们将神经网络偏向于检测场景中的云。由于云探测器的输入光谱中包含不可见波段，因此我们在多光谱域中产生了我们的攻击。这打开了多目标攻击的可能性，具体地说，云敏感波段的对抗性偏向和可见波段的视觉伪装。我们还研究了针对对抗性攻击的缓解策略。我们希望我们的工作进一步提高人们对EO社区中潜在的对抗性攻击的认识。



## **37. Is RobustBench/AutoAttack a suitable Benchmark for Adversarial Robustness?**

RobustBench/AutoAttack是衡量对手健壮性的合适基准吗？ cs.CV

AAAI-22 AdvML Workshop ShortPaper

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01601v1)

**Authors**: Peter Lorenz, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, RobustBench (Croce et al. 2020) has become a widely recognized benchmark for the adversarial robustness of image classification networks. In its most commonly reported sub-task, RobustBench evaluates and ranks the adversarial robustness of trained neural networks on CIFAR10 under AutoAttack (Croce and Hein 2020b) with l-inf perturbations limited to eps = 8/255. With leading scores of the currently best performing models of around 60% of the baseline, it is fair to characterize this benchmark to be quite challenging. Despite its general acceptance in recent literature, we aim to foster discussion about the suitability of RobustBench as a key indicator for robustness which could be generalized to practical applications. Our line of argumentation against this is two-fold and supported by excessive experiments presented in this paper: We argue that I) the alternation of data by AutoAttack with l-inf, eps = 8/255 is unrealistically strong, resulting in close to perfect detection rates of adversarial samples even by simple detection algorithms and human observers. We also show that other attack methods are much harder to detect while achieving similar success rates. II) That results on low-resolution data sets like CIFAR10 do not generalize well to higher resolution images as gradient-based attacks appear to become even more detectable with increasing resolutions.

摘要: 最近，RobustBench(Croce et al.2020)已经成为图像分类网络对抗性健壮性的广泛认可的基准。在其最常见的子任务中，RobustBench在AutoAttack(Croce和Hein 2020b)下评估和排名了CIFAR10上训练的神经网络的对抗鲁棒性，其中l-inf扰动限制在EPS=8/255。目前性能最好的模型的领先分数约为基准的60%，可以公平地将此基准描述为相当具有挑战性。尽管它在最近的文献中被广泛接受，但我们的目标是促进关于RobustBench作为健壮性的关键指标的适宜性的讨论，这可以推广到实际应用中。我们对此的论证是双重的，并得到了本文提出的过多实验的支持：我们认为：i)AutoAttack与l-inf，EPS=8/255的数据交互是不切实际的，导致即使使用简单的检测算法和人类观察者，也能获得接近完美的敌意样本检测率。我们还表明，在获得类似成功率的情况下，其他攻击方法要难得多。ii)在低分辨率数据集(如CIFAR10)上的结果不能很好地推广到更高分辨率的图像，因为基于梯度的攻击似乎随着分辨率的增加而变得更容易检测到。



## **38. Is Approximation Universally Defensive Against Adversarial Attacks in Deep Neural Networks?**

深度神经网络中的近似是否普遍防御敌意攻击？ cs.LG

Accepted for publication in DATE 2022

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01555v1)

**Authors**: Ayesha Siddique, Khaza Anuarul Hoque

**Abstracts**: Approximate computing is known for its effectiveness in improvising the energy efficiency of deep neural network (DNN) accelerators at the cost of slight accuracy loss. Very recently, the inexact nature of approximate components, such as approximate multipliers have also been reported successful in defending adversarial attacks on DNNs models. Since the approximation errors traverse through the DNN layers as masked or unmasked, this raises a key research question-can approximate computing always offer a defense against adversarial attacks in DNNs, i.e., are they universally defensive? Towards this, we present an extensive adversarial robustness analysis of different approximate DNN accelerators (AxDNNs) using the state-of-the-art approximate multipliers. In particular, we evaluate the impact of ten adversarial attacks on different AxDNNs using the MNIST and CIFAR-10 datasets. Our results demonstrate that adversarial attacks on AxDNNs can cause 53% accuracy loss whereas the same attack may lead to almost no accuracy loss (as low as 0.06%) in the accurate DNN. Thus, approximate computing cannot be referred to as a universal defense strategy against adversarial attacks.

摘要: 近似计算在以轻微精度损失为代价提高深度神经网络(DNN)加速器的能效方面是众所周知的。最近，近似分量的不精确性质，如近似乘子，也被报道成功地防御了对DNNs模型的敌意攻击。由于近似误差以屏蔽或非屏蔽的形式遍历DNN各层，这就提出了一个关键的研究问题--近似计算是否总能为DNN中的敌意攻击提供防御，即它们是否具有普遍的防御性？为此，我们使用最先进的近似乘子对不同近似DNN加速器(AxDNNs)进行了广泛的对抗健壮性分析。特别地，我们使用MNIST和CIFAR-10数据集评估了10种对抗性攻击对不同AxDNNs的影响。实验结果表明，对AxDNNs的敌意攻击可以导致53%的准确率损失，而在精确DNN中，同样的攻击可能几乎不会造成准确率损失(低至0.06%)。因此，近似计算不能被称为对抗对手攻击的通用防御策略。



## **39. FedRAD: Federated Robust Adaptive Distillation**

FedRAD：联合鲁棒自适应精馏 cs.LG

Accepted for 1st NeurIPS Workshop on New Frontiers in Federated  Learning (NFFL 2021), Virtual Meeting

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01405v1)

**Authors**: Stefán Páll Sturluson, Samuel Trew, Luis Muñoz-González, Matei Grama, Jonathan Passerat-Palmbach, Daniel Rueckert, Amir Alansary

**Abstracts**: The robustness of federated learning (FL) is vital for the distributed training of an accurate global model that is shared among large number of clients. The collaborative learning framework by typically aggregating model updates is vulnerable to model poisoning attacks from adversarial clients. Since the shared information between the global server and participants are only limited to model parameters, it is challenging to detect bad model updates. Moreover, real-world datasets are usually heterogeneous and not independent and identically distributed (Non-IID) among participants, which makes the design of such robust FL pipeline more difficult. In this work, we propose a novel robust aggregation method, Federated Robust Adaptive Distillation (FedRAD), to detect adversaries and robustly aggregate local models based on properties of the median statistic, and then performing an adapted version of ensemble Knowledge Distillation. We run extensive experiments to evaluate the proposed method against recently published works. The results show that FedRAD outperforms all other aggregators in the presence of adversaries, as well as in heterogeneous data distributions.

摘要: 联邦学习(FL)的健壮性对于分布式训练大量客户端共享的精确全局模型至关重要。典型地聚合模型更新的协作学习框架容易受到来自敌对客户端的模型中毒攻击。由于全局服务器和参与者之间的共享信息仅限于模型参数，因此检测错误的模型更新是具有挑战性的。此外，现实世界的数据集通常是异构的，参与者之间并不是独立且相同分布的(非IID)，这使得设计这样健壮的FL流水线变得更加困难。在这项工作中，我们提出了一种新的健壮聚合方法，联邦健壮自适应蒸馏(FedRAD)，根据中值统计特性检测对手并健壮聚合局部模型，然后执行改进版本的集成知识蒸馏。我们进行了大量的实验，以评估所提出的方法与最近发表的作品。结果表明，FedRAD在存在对手的情况下，以及在异构数据分布的情况下，性能优于所有其他聚合器。



## **40. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

受限特征空间中的对抗性攻防统一框架 cs.AI

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.01156v1)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work on constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework supports the use cases reported in the literature and can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective on two datasets from different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

摘要: 生成可行的对抗性示例对于正确评估工作在受限特征空间上的模型是必要的。然而，对专为计算机视觉设计的攻击实施约束仍然是一项具有挑战性的任务。我们提出了一个统一的框架来生成满足给定领域约束的可行对抗性示例。我们的框架支持文献中报告的用例，并且可以处理线性和非线性约束。我们将我们的框架实例化为两种算法：一种是在损失函数中引入约束以最大化的基于梯度的攻击算法，另一种是以误分类、扰动最小化和约束满足为目标的多目标搜索算法。我们在两个来自不同领域的数据集上证明了我们的方法是有效的，成功率高达100%，其中最先进的攻击没有产生一个可行的例子。除了对抗性再训练之外，我们还建议引入工程非凸约束来提高模型对抗性的稳健性。我们证明了这种新的防御和对抗性的再训练一样有效。我们的框架构成了受限对抗攻击研究的起点，并为未来的研究提供了相关的基线和数据集。



## **41. Adversarial Robustness of Deep Reinforcement Learning based Dynamic Recommender Systems**

基于深度强化学习的动态推荐系统的对抗鲁棒性 cs.LG

arXiv admin note: text overlap with arXiv:2006.07934

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2112.00973v1)

**Authors**: Siyu Wang, Yuanjiang Cao, Xiaocong Chen, Lina Yao, Xianzhi Wang, Quan Z. Sheng

**Abstracts**: Adversarial attacks, e.g., adversarial perturbations of the input and adversarial samples, pose significant challenges to machine learning and deep learning techniques, including interactive recommendation systems. The latent embedding space of those techniques makes adversarial attacks difficult to detect at an early stage. Recent advance in causality shows that counterfactual can also be considered one of ways to generate the adversarial samples drawn from different distribution as the training samples. We propose to explore adversarial examples and attack agnostic detection on reinforcement learning-based interactive recommendation systems. We first craft different types of adversarial examples by adding perturbations to the input and intervening on the casual factors. Then, we augment recommendation systems by detecting potential attacks with a deep learning-based classifier based on the crafted data. Finally, we study the attack strength and frequency of adversarial examples and evaluate our model on standard datasets with multiple crafting methods. Our extensive experiments show that most adversarial attacks are effective, and both attack strength and attack frequency impact the attack performance. The strategically-timed attack achieves comparative attack performance with only 1/3 to 1/2 attack frequency. Besides, our black-box detector trained with one crafting method has the generalization ability over several other crafting methods.

摘要: 对抗性攻击，例如输入和对抗性样本的对抗性扰动，给机器学习和深度学习技术(包括交互式推荐系统)带来了重大挑战。这些技术的潜在嵌入空间使得敌意攻击很难在早期阶段被发现。最近因果关系的进展表明，反事实也可以被认为是生成来自不同分布的对抗性样本作为训练样本的方法之一。我们提出在基于强化学习的交互式推荐系统上探索敌意示例和攻击不可知检测。我们首先制作不同类型的对抗性例子，通过在输入中添加扰动和对偶然因素进行干预来创建不同类型的对抗性例子。然后，我们利用基于深度学习的分类器基于精心制作的数据来检测潜在的攻击，从而增强推荐系统。最后，我们研究了敌意实例的攻击强度和攻击频率，并在标准数据集上采用多种制作方法对我们的模型进行了评估。我们的大量实验表明，大多数对抗性攻击都是有效的，攻击强度和攻击频率都会影响攻击性能。战略计时攻击仅用1/3到1/2的攻击频率就达到了比较的攻击性能。此外，用一种工艺方法训练的黑盒检测器比其他几种工艺方法具有更好的泛化能力。



## **42. Learning Task-aware Robust Deep Learning Systems**

学习任务感知的鲁棒深度学习系统 cs.LG

9 Pages

**SubmitDate**: 2021-12-02    [paper-pdf](http://arxiv.org/pdf/2010.05125v2)

**Authors**: Keji Han, Yun Li, Xianzhong Long, Yao Ge

**Abstracts**: Many works demonstrate that deep learning system is vulnerable to adversarial attack. A deep learning system consists of two parts: the deep learning task and the deep model. Nowadays, most existing works investigate the impact of the deep model on robustness of deep learning systems, ignoring the impact of the learning task. In this paper, we adopt the binary and interval label encoding strategy to redefine the classification task and design corresponding loss to improve robustness of the deep learning system. Our method can be viewed as improving the robustness of deep learning systems from both the learning task and deep model. Experimental results demonstrate that our learning task-aware method is much more robust than traditional classification while retaining the accuracy.

摘要: 大量研究表明，深度学习系统容易受到敌意攻击。深度学习系统由两部分组成：深度学习任务和深度模型。目前，已有的工作大多研究深度模型对深度学习系统鲁棒性的影响，忽略了学习任务的影响。本文采用二进制和区间标签编码策略重新定义分类任务，并设计相应的损失来提高深度学习系统的鲁棒性。我们的方法可以看作是从学习任务和深度模型两个方面提高深度学习系统的鲁棒性。实验结果表明，我们的学习任务感知方法在保持分类准确率的同时，比传统的分类方法具有更强的鲁棒性。



## **43. They See Me Rollin': Inherent Vulnerability of the Rolling Shutter in CMOS Image Sensors**

他们看到我在滚动：CMOS图像传感器中滚动快门的固有弱点 cs.CV

15 pages, 15 figures

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2101.10011v2)

**Authors**: Sebastian Köhler, Giulio Lovisotto, Simon Birnbach, Richard Baker, Ivan Martinovic

**Abstracts**: In this paper, we describe how the electronic rolling shutter in CMOS image sensors can be exploited using a bright, modulated light source (e.g., an inexpensive, off-the-shelf laser), to inject fine-grained image disruptions. We demonstrate the attack on seven different CMOS cameras, ranging from cheap IoT to semi-professional surveillance cameras, to highlight the wide applicability of the rolling shutter attack. We model the fundamental factors affecting a rolling shutter attack in an uncontrolled setting. We then perform an exhaustive evaluation of the attack's effect on the task of object detection, investigating the effect of attack parameters. We validate our model against empirical data collected on two separate cameras, showing that by simply using information from the camera's datasheet the adversary can accurately predict the injected distortion size and optimize their attack accordingly. We find that an adversary can hide up to 75% of objects perceived by state-of-the-art detectors by selecting appropriate attack parameters. We also investigate the stealthiness of the attack in comparison to a na\"{i}ve camera blinding attack, showing that common image distortion metrics can not detect the attack presence. Therefore, we present a new, accurate and lightweight enhancement to the backbone network of an object detector to recognize rolling shutter attacks. Overall, our results indicate that rolling shutter attacks can substantially reduce the performance and reliability of vision-based intelligent systems.

摘要: 在这篇文章中，我们描述了如何利用CMOS图像传感器中的电子滚动快门，使用明亮的调制光源(例如，廉价的现成激光器)来注入细粒度的图像干扰。我们演示了对七种不同CMOS摄像头的攻击，从廉价的物联网到半专业的监控摄像头，以突出滚动快门攻击的广泛适用性。我们模拟了在不受控制的环境下影响滚动快门攻击的基本因素。然后，我们对攻击对目标检测任务的影响进行了详尽的评估，考察了攻击参数的影响。我们通过在两个不同的摄像机上收集的经验数据验证了我们的模型，结果表明，通过简单地使用摄像机数据表中的信息，对手可以准确地预测注入失真的大小，并相应地优化他们的攻击。我们发现，通过选择合适的攻击参数，敌手可以隐藏最新检测器感知到的高达75%的对象。与单纯的相机盲攻击相比，我们还研究了该攻击的隐蔽性，发现普通的图像失真度量不能检测到攻击的存在，因此，我们提出了一种新的、准确的、轻量级的对象检测器主干网络增强方法来识别滚动快门攻击，结果表明，滚动快门攻击会大大降低基于视觉的智能系统的性能和可靠性。



## **44. Certified Adversarial Defenses Meet Out-of-Distribution Corruptions: Benchmarking Robustness and Simple Baselines**

认证的对抗性防御遇到分布外的腐败：基准、健壮性和简单的基线 cs.LG

21 pages, 15 figures, and 9 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00659v1)

**Authors**: Jiachen Sun, Akshay Mehra, Bhavya Kailkhura, Pin-Yu Chen, Dan Hendrycks, Jihun Hamm, Z. Morley Mao

**Abstracts**: Certified robustness guarantee gauges a model's robustness to test-time attacks and can assess the model's readiness for deployment in the real world. In this work, we critically examine how the adversarial robustness guarantees from randomized smoothing-based certification methods change when state-of-the-art certifiably robust models encounter out-of-distribution (OOD) data. Our analysis demonstrates a previously unknown vulnerability of these models to low-frequency OOD data such as weather-related corruptions, rendering these models unfit for deployment in the wild. To alleviate this issue, we propose a novel data augmentation scheme, FourierMix, that produces augmentations to improve the spectral coverage of the training data. Furthermore, we propose a new regularizer that encourages consistent predictions on noise perturbations of the augmented data to improve the quality of the smoothed models. We find that FourierMix augmentations help eliminate the spectral bias of certifiably robust models enabling them to achieve significantly better robustness guarantees on a range of OOD benchmarks. Our evaluation also uncovers the inability of current OOD benchmarks at highlighting the spectral biases of the models. To this end, we propose a comprehensive benchmarking suite that contains corruptions from different regions in the spectral domain. Evaluation of models trained with popular augmentation methods on the proposed suite highlights their spectral biases and establishes the superiority of FourierMix trained models at achieving better-certified robustness guarantees under OOD shifts over the entire frequency spectrum.

摘要: 认证的健壮性保证衡量模型对测试时间攻击的健壮性，并可以评估模型在现实世界中部署的准备情况。在这项工作中，我们批判性地研究了当最新的可证明鲁棒性模型遇到分布外(OOD)数据时，基于随机平滑的认证方法所保证的敌意鲁棒性是如何改变的。我们的分析表明，这些模型对低频OOD数据(如与天气相关的损坏)存在以前未知的脆弱性，使得这些模型不适合在野外部署。为了缓解这一问题，我们提出了一种新的数据增强方案FURIERMIX，该方案通过产生增强来提高训练数据的频谱覆盖率。此外，我们还提出了一种新的正则化方法，它鼓励对增强数据的噪声扰动进行一致的预测，以提高平滑模型的质量。我们发现，傅立叶混合增强有助于消除可证明的健壮性模型的频谱偏差，使它们能够在一系列面向对象设计基准上获得显着更好的健壮性保证。我们的评估还揭示了当前OOD基准在突出模型的光谱偏差方面的不足。为此，我们提出了一个全面的基准测试套件，该套件包含来自谱域中不同区域的腐败。在建议的套件上用流行的增强方法训练的模型的评估突出了它们的频谱偏差，并确立了傅里叶混合训练的模型在整个频谱上的OOD漂移下实现更好的认证鲁棒性保证的优势。



## **45. Well-classified Examples are Underestimated in Classification with Deep Neural Networks**

深度神经网络在分类中低估分类好的样本 cs.LG

Accepted by AAAI 2022; 16 pages, 11 figures, 13 tables

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2110.06537v3)

**Authors**: Guangxiang Zhao, Wenkai Yang, Xuancheng Ren, Lei Li, Xu Sun

**Abstracts**: The conventional wisdom behind learning deep classification models is to focus on bad-classified examples and ignore well-classified examples that are far from the decision boundary. For instance, when training with cross-entropy loss, examples with higher likelihoods (i.e., well-classified examples) contribute smaller gradients in back-propagation. However, we theoretically show that this common practice hinders representation learning, energy optimization, and the growth of margin. To counteract this deficiency, we propose to reward well-classified examples with additive bonuses to revive their contribution to learning. This counterexample theoretically addresses these three issues. We empirically support this claim by directly verify the theoretical results or through the significant performance improvement with our counterexample on diverse tasks, including image classification, graph classification, and machine translation. Furthermore, this paper shows that because our idea can solve these three issues, we can deal with complex scenarios, such as imbalanced classification, OOD detection, and applications under adversarial attacks. Code is available at: https://github.com/lancopku/well-classified-examples-are-underestimated.

摘要: 学习深度分类模型背后的传统智慧是专注于分类不良的示例，而忽略远离决策边界的分类良好的示例。例如，当使用交叉熵损失进行训练时，具有较高似然的示例(即，分类良好的示例)在反向传播中贡献较小的梯度。然而，我们从理论上表明，这种普遍的做法阻碍了表征学习、能量优化和边际增长。为了弥补这一不足，我们建议向分类良好的例子发放额外奖金，以恢复他们对学习的贡献。这个反例从理论上解决了这三个问题。我们通过直接验证理论结果或通过我们的反例在不同任务(包括图像分类、图形分类和机器翻译)上的显著性能改进来实证支持这一主张。此外，本文还表明，由于我们的思想可以解决这三个问题，所以我们可以处理复杂的场景，如不平衡分类、面向对象的检测以及在对抗性攻击下的应用。代码可在以下网址获得：https://github.com/lancopku/well-classified-examples-are-underestimated.



## **46. Understanding Adversarial Attacks on Observations in Deep Reinforcement Learning**

理解深度强化学习中对观测的敌意攻击 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2106.15860v2)

**Authors**: You Qiaoben, Chengyang Ying, Xinning Zhou, Hang Su, Jun Zhu, Bo Zhang

**Abstracts**: Deep reinforcement learning models are vulnerable to adversarial attacks that can decrease a victim's cumulative expected reward by manipulating the victim's observations. Despite the efficiency of previous optimization-based methods for generating adversarial noise in supervised learning, such methods might not be able to achieve the lowest cumulative reward since they do not explore the environmental dynamics in general. In this paper, we provide a framework to better understand the existing methods by reformulating the problem of adversarial attacks on reinforcement learning in the function space. Our reformulation generates an optimal adversary in the function space of the targeted attacks, repelling them via a generic two-stage framework. In the first stage, we train a deceptive policy by hacking the environment, and discover a set of trajectories routing to the lowest reward or the worst-case performance. Next, the adversary misleads the victim to imitate the deceptive policy by perturbing the observations. Compared to existing approaches, we theoretically show that our adversary is stronger under an appropriate noise level. Extensive experiments demonstrate our method's superiority in terms of efficiency and effectiveness, achieving the state-of-the-art performance in both Atari and MuJoCo environments.

摘要: 深度强化学习模型很容易受到敌意攻击，这种攻击会通过操纵受害者的观察结果来降低受害者的累积预期回报。尽管以前的基于优化的方法在监督学习中产生对抗性噪声是有效的，但是这些方法可能不能获得最低的累积奖励，因为它们通常不探索环境动态。本文通过在函数空间中重新表述强化学习的对抗性攻击问题，为更好地理解现有方法提供了一个框架。我们的重构在目标攻击的函数空间中生成一个最优对手，通过一个通用的两阶段框架击退它们。在第一阶段，我们通过黑客攻击环境来训练欺骗性策略，并发现一组通往最低回报或最坏情况表现的轨迹。接下来，对手通过扰乱观察来误导受害者模仿欺骗性的政策。与现有的方法相比，我们从理论上证明了在适当的噪声水平下，我们的对手更强。大量的实验证明了我们的方法在效率和有效性方面的优越性，在Atari和MuJoCo环境中都实现了最先进的性能。



## **47. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

$\ell_\infty$-健壮性和超越：释放高效的对抗性训练 cs.LG

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00378v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, which has hampered its effectiveness. Recently, Fast Adversarial Training was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection we show how selecting a small subset of training data provides a more principled approach towards reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. Our experimental results indicate that our approach speeds up adversarial training by 2-3 times, while experiencing a small reduction in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练鲁棒模型对抗此类攻击的最有效方法之一。但是，由于它在每次迭代时都需要为整个训练数据构造对抗性样本，因此比神经网络的香草训练慢得多，这就阻碍了它的有效性。最近，人们提出了一种快速对抗性训练方法，可以有效地获得稳健的模型。然而，其成功背后的原因还没有被完全理解，更重要的是，由于它在训练期间使用FGSM，所以它只能训练健壮的模型来应对$\ell_\$有界攻击。在本文中，通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种更有原则的方法来降低鲁棒训练的时间复杂度。与现有方法不同，我们的方法可以适应广泛的训练目标，包括行业、$\ell_p$-PGD和知觉对抗性训练。我们的实验结果表明，我们的方法将对抗性训练的速度提高了2-3倍，同时经历了干净和健壮的准确率的小幅下降。



## **48. Designing a Location Trace Anonymization Contest**

设计一个位置跟踪匿名化竞赛 cs.CR

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2107.10407v2)

**Authors**: Takao Murakami, Hiromi Arai, Koki Hamada, Takuma Hatano, Makoto Iguchi, Hiroaki Kikuchi, Atsushi Kuromasa, Hiroshi Nakagawa, Yuichi Nakamura, Kenshiro Nishiyama, Ryo Nojima, Hidenobu Oguri, Chiemi Watanabe, Akira Yamada, Takayasu Yamaguchi, Yuji Yamaoka

**Abstracts**: For a better understanding of anonymization methods for location traces, we have designed and held a location trace anonymization contest. Our contest deals with a long trace (400 events per user) and fine-grained locations (1024 regions). In our contest, each team anonymizes her original traces, and then the other teams perform privacy attacks against the anonymized traces in a partial-knowledge attacker model where the adversary does not know the original traces. To realize such a contest, we propose a location synthesizer that has diversity and utility; the synthesizer generates different synthetic traces for each team while preserving various statistical features of real traces. We also show that re-identification alone is insufficient as a privacy risk and that trace inference should be added as an additional risk. Specifically, we show an example of anonymization that is perfectly secure against re-identification and is not secure against trace inference. Based on this, our contest evaluates both the re-identification risk and trace inference risk and analyzes their relationship. Through our contest, we show several findings in a situation where both defense and attack compete together. In particular, we show that an anonymization method secure against trace inference is also secure against re-identification under the presence of appropriate pseudonymization.

摘要: 为了更好地了解位置踪迹的匿名化方法，我们设计并举办了位置踪迹匿名化大赛。我们的竞赛涉及长跟踪(每个用户400个事件)和细粒度位置(1024个区域)。在我们的比赛中，每个团队匿名她的原始痕迹，然后其他团队在部分知识攻击者模型中对匿名的痕迹进行隐私攻击，其中对手不知道原始痕迹。为了实现这样的竞赛，我们提出了一种具有多样性和实用性的位置合成器，该合成器在保留真实轨迹的各种统计特征的同时，为每个团队生成不同的合成轨迹。我们还表明，仅重新识别作为隐私风险是不够的，应该添加跟踪推断作为附加风险。具体地说，我们展示了一个匿名化的例子，它对于重新识别是完全安全的，而对于跟踪推理是不安全的。在此基础上，对再识别风险和痕迹推理风险进行了评估，并分析了它们之间的关系。通过我们的比赛，我们展示了在防守和进攻同时竞争的情况下的几个发现。特别地，我们证明了在存在适当的假名的情况下，一个安全的抗踪迹推理的匿名化方法也是安全的。



## **49. Push Stricter to Decide Better: A Class-Conditional Feature Adaptive Framework for Improving Adversarial Robustness**

越严越优：一种提高对手健壮性的类条件特征自适应框架 cs.CV

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00323v1)

**Authors**: Jia-Li Yin, Lehui Xie, Wanqing Zhu, Ximeng Liu, Bo-Hao Chen

**Abstracts**: In response to the threat of adversarial examples, adversarial training provides an attractive option for enhancing the model robustness by training models on online-augmented adversarial examples. However, most of the existing adversarial training methods focus on improving the robust accuracy by strengthening the adversarial examples but neglecting the increasing shift between natural data and adversarial examples, leading to a dramatic decrease in natural accuracy. To maintain the trade-off between natural and robust accuracy, we alleviate the shift from the perspective of feature adaption and propose a Feature Adaptive Adversarial Training (FAAT) optimizing the class-conditional feature adaption across natural data and adversarial examples. Specifically, we propose to incorporate a class-conditional discriminator to encourage the features become (1) class-discriminative and (2) invariant to the change of adversarial attacks. The novel FAAT framework enables the trade-off between natural and robust accuracy by generating features with similar distribution across natural and adversarial data, and achieve higher overall robustness benefited from the class-discriminative feature characteristics. Experiments on various datasets demonstrate that FAAT produces more discriminative features and performs favorably against state-of-the-art methods. Codes are available at https://github.com/VisionFlow/FAAT.

摘要: 为了应对对抗性示例的威胁，对抗性训练通过训练在线扩充的对抗性示例模型，为增强模型的稳健性提供了一种有吸引力的选择。然而，现有的对抗性训练方法大多侧重于通过加强对抗性实例来提高鲁棒准确率，而忽略了自然数据与对抗性实例之间不断增加的偏移，导致自然精确度急剧下降。为了保持自然和鲁棒精度之间的折衷，我们从特征自适应的角度缓解了这一转变，并提出了一种特征自适应对抗训练(FAAT)，优化了跨自然数据和对抗性示例的类条件特征自适应。具体地说，我们建议加入类条件鉴别器，以鼓励特征成为(1)类可分辨的和(2)对敌方攻击变化不变的特征。新的FAAT框架通过在自然数据和对抗性数据上生成分布相似的特征，能够在自然和鲁棒精度之间进行权衡，并得益于类区分特征特性而获得更高的整体鲁棒性。在不同的数据集上的实验表明，FAAT产生了更具区分性的特征，并且与最先进的方法相比表现出了良好的性能。有关代码，请访问https://github.com/VisionFlow/FAAT.。



## **50. Adversarial Attacks Against Deep Generative Models on Data: A Survey**

针对数据深层生成模型的对抗性攻击：综述 cs.CR

To be published in IEEE Transactions on Knowledge and Data  Engineering

**SubmitDate**: 2021-12-01    [paper-pdf](http://arxiv.org/pdf/2112.00247v1)

**Authors**: Hui Sun, Tianqing Zhu, Zhiqiu Zhang, Dawei Jin. Ping Xiong, Wanlei Zhou

**Abstracts**: Deep generative models have gained much attention given their ability to generate data for applications as varied as healthcare to financial technology to surveillance, and many more - the most popular models being generative adversarial networks and variational auto-encoders. Yet, as with all machine learning models, ever is the concern over security breaches and privacy leaks and deep generative models are no exception. These models have advanced so rapidly in recent years that work on their security is still in its infancy. In an attempt to audit the current and future threats against these models, and to provide a roadmap for defense preparations in the short term, we prepared this comprehensive and specialized survey on the security and privacy preservation of GANs and VAEs. Our focus is on the inner connection between attacks and model architectures and, more specifically, on five components of deep generative models: the training data, the latent code, the generators/decoders of GANs/ VAEs, the discriminators/encoders of GANs/ VAEs, and the generated data. For each model, component and attack, we review the current research progress and identify the key challenges. The paper concludes with a discussion of possible future attacks and research directions in the field.

摘要: 深度生成模型因其能够为从医疗保健到金融技术再到监控等各种应用程序生成数据而备受关注-最受欢迎的模型是生成性对抗性网络和变化式自动编码器。然而，与所有机器学习模型一样，人们一直担心安全漏洞和隐私泄露，深度生成模型也不例外。近年来，这些模式发展如此之快，其安全方面的工作仍处于初级阶段。为了审计这些模式当前和未来的威胁，并为短期内的防御准备提供路线图，我们准备了这项关于GAN和VAE的安全和隐私保护的全面而专业的调查。我们的重点是攻击和模型体系结构之间的内在联系，更具体地说，是深入生成模型的五个组成部分：训练数据、潜在代码、GANS/VAE的生成器/解码器、GANS/VAE的鉴别器/编码器和生成的数据。对于每个模型、组件和攻击，我们回顾了当前的研究进展，并确定了关键挑战。最后，对未来可能的攻击和该领域的研究方向进行了讨论。



