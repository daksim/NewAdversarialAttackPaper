# Latest Adversarial Attack Papers
**update at 2025-01-29 20:07:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Scanning Trojaned Models Using Out-of-Distribution Samples**

使用分布外样本扫描特洛伊模型 cs.LG

Accepted at the Thirty-Eighth Annual Conference on Neural Information  Processing Systems (NeurIPS) 2024. The code repository is available at:  https://github.com/rohban-lab/TRODO

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.17151v1) [paper-pdf](http://arxiv.org/pdf/2501.17151v1)

**Authors**: Hossein Mirzaei, Ali Ansari, Bahar Dibaei Nia, Mojtaba Nafez, Moein Madadi, Sepehr Rezaee, Zeinab Sadat Taghavi, Arad Maleki, Kian Shamsaie, Mahdi Hajialilue, Jafar Habibi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Scanning for trojan (backdoor) in deep neural networks is crucial due to their significant real-world applications. There has been an increasing focus on developing effective general trojan scanning methods across various trojan attacks. Despite advancements, there remains a shortage of methods that perform effectively without preconceived assumptions about the backdoor attack method. Additionally, we have observed that current methods struggle to identify classifiers trojaned using adversarial training. Motivated by these challenges, our study introduces a novel scanning method named TRODO (TROjan scanning by Detection of adversarial shifts in Out-of-distribution samples). TRODO leverages the concept of "blind spots"--regions where trojaned classifiers erroneously identify out-of-distribution (OOD) samples as in-distribution (ID). We scan for these blind spots by adversarially shifting OOD samples towards in-distribution. The increased likelihood of perturbed OOD samples being classified as ID serves as a signature for trojan detection. TRODO is both trojan and label mapping agnostic, effective even against adversarially trained trojaned classifiers. It is applicable even in scenarios where training data is absent, demonstrating high accuracy and adaptability across various scenarios and datasets, highlighting its potential as a robust trojan scanning strategy.

摘要: 扫描深层神经网络中的特洛伊木马(后门)至关重要，因为它们在现实世界中有着重要的应用。人们越来越关注开发有效的通用特洛伊木马扫描方法来应对各种特洛伊木马攻击。尽管取得了进步，但仍然缺乏有效执行而没有关于后门攻击方法的先入为主的假设的方法。此外，我们还观察到，目前的方法难以识别使用对抗性训练的特洛伊木马分类器。受这些挑战的启发，我们的研究提出了一种新的扫描方法TRODO(特洛伊木马扫描通过检测分布外样本中的敌意偏移)。TRODO利用“盲点”的概念--在这些区域，安装了特洛伊木马的分类器错误地将分布外(OOD)样本识别为分布内(ID)。我们通过相反地将OOD样本转向分布内来扫描这些盲点。受干扰的OOD样本被归类为ID的可能性增加，这是特洛伊木马检测的特征。TRODO是特洛伊木马和标签映射不可知的，即使对抗恶意训练的特洛伊木马分类器也是有效的。它即使在没有训练数据的情况下也适用，在各种情况和数据集上显示出高准确性和适应性，突出了其作为一种强大的特洛伊木马扫描策略的潜力。



## **2. Hybrid Deep Learning Model for Multiple Cache Side Channel Attacks Detection: A Comparative Analysis**

用于多缓存侧通道攻击检测的混合深度学习模型：比较分析 cs.CR

8 pages, 4 figures. Accepted in IEEE's 2nd International Conference  on Computational Intelligence and Network Systems

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.17123v1) [paper-pdf](http://arxiv.org/pdf/2501.17123v1)

**Authors**: Tejal Joshi, Aarya Kawalay, Anvi Jamkhande, Amit Joshi

**Abstract**: Cache side channel attacks are a sophisticated and persistent threat that exploit vulnerabilities in modern processors to extract sensitive information. These attacks leverage weaknesses in shared computational resources, particularly the last level cache, to infer patterns in data access and execution flows, often bypassing traditional security defenses. Such attacks are especially dangerous as they can be executed remotely without requiring physical access to the victim's device. This study focuses on a specific class of these threats: fingerprinting attacks, where an adversary monitors and analyzes the behavior of co-located processes via cache side channels. This can potentially reveal confidential information, such as encryption keys or user activity patterns. A comprehensive threat model illustrates how attackers sharing computational resources with target systems exploit these side channels to compromise sensitive data. To mitigate such risks, a hybrid deep learning model is proposed for detecting cache side channel attacks. Its performance is compared with five widely used deep learning models: Multi-Layer Perceptron, Convolutional Neural Network, Simple Recurrent Neural Network, Long Short-Term Memory, and Gated Recurrent Unit. The experimental results demonstrate that the hybrid model achieves a detection rate of up to 99.96%. These findings highlight the limitations of existing models, the need for enhanced defensive mechanisms, and directions for future research to secure sensitive data against evolving side channel threats.

摘要: 高速缓存侧通道攻击是一种利用现代处理器中的漏洞来提取敏感信息的复杂而持久的威胁。这些攻击利用共享计算资源中的弱点，特别是末级缓存，来推断数据访问和执行流的模式，通常绕过传统的安全防御。这种攻击特别危险，因为它们可以远程执行，而不需要物理访问受害者的设备。这项研究集中在这些威胁的一种特定类别：指纹攻击，在这种攻击中，对手通过缓存端通道监视和分析协同定位进程的行为。这可能会泄露机密信息，如加密密钥或用户活动模式。一个全面的威胁模型说明了与目标系统共享计算资源的攻击者如何利用这些旁路来危害敏感数据。为了降低这种风险，提出了一种用于检测缓存侧通道攻击的混合深度学习模型。与五种广泛使用的深度学习模型：多层感知器、卷积神经网络、简单递归神经网络、长短期记忆和门控递归单元的性能进行了比较。实验结果表明，该混合模型的检测率高达99.96%。这些发现突出了现有模型的局限性，需要增强的防御机制，以及未来研究的方向，以保护敏感数据免受不断演变的旁路威胁。



## **3. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2412.08099v3) [paper-pdf](http://arxiv.org/pdf/2412.08099v3)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in the field of time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like TimeGPT and LLM-Time with GPT-3.5, GPT-4, LLaMa, and Mistral, show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications.

摘要: 大型语言模型最近在时间序列预测领域显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验，包括使用GPT-3.5、GPT-4、LLAMA和Mistral的TimeGPT和LLM-Time模型，表明对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。



## **4. Random-Set Neural Networks (RS-NN)**

随机集神经网络（RS-NN） cs.LG

Published as a conference paper at the Thirteenth International  Conference on Learning Representations (ICLR 2025)

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2307.05772v4) [paper-pdf](http://arxiv.org/pdf/2307.05772v4)

**Authors**: Shireen Kudukkil Manchingal, Muhammad Mubashar, Kaizheng Wang, Keivan Shariatmadar, Fabio Cuzzolin

**Abstract**: Machine learning is increasingly deployed in safety-critical domains where erroneous predictions may lead to potentially catastrophic consequences, highlighting the need for learning systems to be aware of how confident they are in their own predictions: in other words, 'to know when they do not know'. In this paper, we propose a novel Random-Set Neural Network (RS-NN) approach to classification which predicts belief functions (rather than classical probability vectors) over the class list using the mathematics of random sets, i.e., distributions over the collection of sets of classes. RS-NN encodes the 'epistemic' uncertainty induced by training sets that are insufficiently representative or limited in size via the size of the convex set of probability vectors associated with a predicted belief function. Our approach outperforms state-of-the-art Bayesian and Ensemble methods in terms of accuracy, uncertainty estimation and out-of-distribution (OoD) detection on multiple benchmarks (CIFAR-10 vs SVHN/Intel-Image, MNIST vs FMNIST/KMNIST, ImageNet vs ImageNet-O). RS-NN also scales up effectively to large-scale architectures (e.g. WideResNet-28-10, VGG16, Inception V3, EfficientNetB2 and ViT-Base-16), exhibits remarkable robustness to adversarial attacks and can provide statistical guarantees in a conformal learning setting.

摘要: 机器学习越来越多地部署在安全关键领域，在这些领域，错误的预测可能会导致潜在的灾难性后果，这突显了学习系统需要意识到自己对自己的预测有多自信：换句话说，“知道什么时候他们不知道”。在本文中，我们提出了一种新的随机集神经网络(RS-NN)分类方法，它使用随机集的数学，即在类集合集合上的分布来预测类列表上的信任函数(而不是经典的概率向量)。RS-NN通过与预测的信任函数相关联的概率向量凸集的大小来编码不具有足够代表性或大小受限的训练集所引起的“认知”不确定性。在多个基准测试(CIFAR-10与SVHN/Intel-Image、MNIST与FMNIST/KMNIST、ImageNet与ImageNet-O)上，我们的方法在准确性、不确定性估计和OOD检测方面优于最先进的贝叶斯和集成方法。RS-NN还可以有效地扩展到大规模体系结构(如WideResNet-28-10、VGG16、初始V3、EfficientNetB2和Vit-Base-16)，对对手攻击表现出显著的鲁棒性，并可以在共形学习环境下提供统计保证。



## **5. Adversarial Masked Autoencoder Purifier with Defense Transferability**

具有防御可移植性的对抗性掩蔽自动编码器净化器 cs.CV

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16904v1) [paper-pdf](http://arxiv.org/pdf/2501.16904v1)

**Authors**: Yuan-Chih Chen, Chun-Shien Lu

**Abstract**: The study of adversarial defense still struggles to combat with advanced adversarial attacks. In contrast to most prior studies that rely on the diffusion model for test-time defense to remarkably increase the inference time, we propose Masked AutoEncoder Purifier (MAEP), which integrates Masked AutoEncoder (MAE) into an adversarial purifier framework for test-time purification. While MAEP achieves promising adversarial robustness, it particularly features model defense transferability and attack generalization without relying on using additional data that is different from the training dataset. To our knowledge, MAEP is the first study of adversarial purifier based on MAE. Extensive experimental results demonstrate that our method can not only maintain clear accuracy with only a slight drop but also exhibit a close gap between the clean and robust accuracy. Notably, MAEP trained on CIFAR10 achieves state-of-the-art performance even when tested directly on ImageNet, outperforming existing diffusion-based models trained specifically on ImageNet.

摘要: 对抗性防御的研究仍然难以与先进的对抗性攻击相抗衡。与以往大多数研究依赖扩散模型进行测试时间防御以显著增加推理时间不同，我们提出了掩蔽自动编码器净化器(MAEP)，它将掩蔽自动编码器(MAE)集成到对抗性净化器框架中来进行测试时间净化。虽然MAEP实现了有希望的对抗健壮性，但它特别具有模型防御的可转移性和攻击泛化，而不依赖于使用与训练数据集不同的额外数据。据我们所知，MAEP是第一个基于MAE的对抗性净化器的研究。大量的实验结果表明，该方法不仅能够在较小的下降范围内保持较高的精度，而且保持了较高的精度和较好的鲁棒性。值得注意的是，在CIFAR10上训练的MAEP即使在ImageNet上直接测试时也能达到最先进的性能，表现优于专门在ImageNet上训练的现有基于扩散的模型。



## **6. Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks**

文档截图检索器容易受到像素中毒攻击 cs.IR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16902v1) [paper-pdf](http://arxiv.org/pdf/2501.16902v1)

**Authors**: Shengyao Zhuang, Ekaterina Khramtsova, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon

**Abstract**: Recent advancements in dense retrieval have introduced vision-language model (VLM)-based retrievers, such as DSE and ColPali, which leverage document screenshots embedded as vectors to enable effective search and offer a simplified pipeline over traditional text-only methods. In this study, we propose three pixel poisoning attack methods designed to compromise VLM-based retrievers and evaluate their effectiveness under various attack settings and parameter configurations. Our empirical results demonstrate that injecting even a single adversarial screenshot into the retrieval corpus can significantly disrupt search results, poisoning the top-10 retrieved documents for 41.9% of queries in the case of DSE and 26.4% for ColPali. These vulnerability rates notably exceed those observed with equivalent attacks on text-only retrievers. Moreover, when targeting a small set of known queries, the attack success rate raises, achieving complete success in certain cases. By exposing the vulnerabilities inherent in vision-language models, this work highlights the potential risks associated with their deployment.

摘要: 密集检索的最新进展引入了基于视觉语言模型(VLM)的检索器，如DSE和ColPali，它们利用嵌入作为载体的文档屏幕截图来实现有效的搜索，并提供了比传统的纯文本方法更简单的管道。在这项研究中，我们提出了三种像素中毒攻击方法，旨在危害基于VLM的检索者，并在不同的攻击设置和参数配置下评估它们的有效性。我们的实验结果表明，即使在检索语料库中插入一个敌意截图也会显著扰乱搜索结果，在DSE和ColPali的情况下，41.9%的查询和26.4%的查询毒化了检索到的前10个文档。这些脆弱性明显超过了对纯文本检索器的同等攻击。此外，当以一小部分已知查询为目标时，攻击成功率会提高，在某些情况下会实现完全成功。通过暴露视觉语言模型中固有的漏洞，这项工作突出了与其部署相关的潜在风险。



## **7. "My Whereabouts, my Location, it's Directly Linked to my Physical Security": An Exploratory Qualitative Study of Location-Dependent Security and Privacy Perceptions among Activist Tech Users**

“我的地点、位置与我的身体安全直接相关”：活动主义技术用户中位置相关的安全和隐私感知的探索性定性研究 cs.HC

10 pages, incl. interview guide and codebook

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16885v1) [paper-pdf](http://arxiv.org/pdf/2501.16885v1)

**Authors**: Christian Eichenmüller, Lisa Kuhn, Zinaida Benenson

**Abstract**: Digital-safety research with at-risk users is particularly urgent. At-risk users are more likely to be digitally attacked or targeted by surveillance and could be disproportionately harmed by attacks that facilitate physical assaults. One group of such at-risk users are activists and politically active individuals. For them, as for other at-risk users, the rise of smart environments harbors new risks. Since digitization and datafication are no longer limited to a series of personal devices that can be switched on and off, but increasingly and continuously surround users, granular geolocation poses new safety challenges. Drawing on eight exploratory qualitative interviews of an ongoing research project, this contribution highlights what activists with powerful adversaries think about evermore data traces, including location data, and how they intend to deal with emerging risks. Responses of activists include attempts to control one's immediate technological surroundings and to more carefully manage device-related location data. For some activists, threat modeling has also shaped provider choices based on geopolitical considerations. Since many activists have not enough digital-safety knowledge for effective protection, feelings of insecurity and paranoia are widespread. Channeling the concerns and fears of our interlocutors, we call for more research on how activists can protect themselves against evermore fine-grained location data tracking.

摘要: 针对高危用户的数字安全研究尤为迫切。高危用户更有可能受到数字攻击或成为监控的目标，并可能受到为物理攻击提供便利的攻击的不成比例的伤害。这类风险用户中有一群是积极分子和政治活跃的个人。对他们来说，就像对其他高危用户一样，智能环境的崛起蕴藏着新的风险。由于数字化和数据化不再局限于一系列可以开关的个人设备，而是越来越多地、持续地围绕着用户，颗粒状地理定位提出了新的安全挑战。根据对一个正在进行的研究项目的8次探索性定性采访，这篇文章强调了拥有强大对手的活动人士对包括位置数据在内的越来越多的数据痕迹的看法，以及他们打算如何应对新出现的风险。活动人士的反应包括试图控制自己眼前的技术环境，以及更仔细地管理与设备相关的位置数据。对于一些活动人士来说，威胁建模也塑造了基于地缘政治考虑的提供商选择。由于许多活动人士没有足够的数字安全知识来进行有效的保护，不安全感和偏执情绪普遍存在。为了消除对话者的担忧和恐惧，我们呼吁对活动人士如何保护自己免受更多细粒度位置数据跟踪的影响进行更多研究。



## **8. CantorNet: A Sandbox for Testing Geometrical and Topological Complexity Measures**

CantorNet：测试几何和布局复杂性测量的沙盒 cs.NE

Accepted at the NeurIPS Workshop on Symmetry and Geometry in Neural  Representations, 2024

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2411.19713v3) [paper-pdf](http://arxiv.org/pdf/2411.19713v3)

**Authors**: Michal Lewandowski, Hamid Eghbalzadeh, Bernhard A. Moser

**Abstract**: Many natural phenomena are characterized by self-similarity, for example the symmetry of human faces, or a repetitive motif of a song. Studying of such symmetries will allow us to gain deeper insights into the underlying mechanisms of complex systems. Recognizing the importance of understanding these patterns, we propose a geometrically inspired framework to study such phenomena in artificial neural networks. To this end, we introduce \emph{CantorNet}, inspired by the triadic construction of the Cantor set, which was introduced by Georg Cantor in the $19^\text{th}$ century. In mathematics, the Cantor set is a set of points lying on a single line that is self-similar and has a counter intuitive property of being an uncountably infinite null set. Similarly, we introduce CantorNet as a sandbox for studying self-similarity by means of novel topological and geometrical complexity measures. CantorNet constitutes a family of ReLU neural networks that spans the whole spectrum of possible Kolmogorov complexities, including the two opposite descriptions (linear and exponential as measured by the description length). CantorNet's decision boundaries can be arbitrarily ragged, yet are analytically known. Besides serving as a testing ground for complexity measures, our work may serve to illustrate potential pitfalls in geometry-ignorant data augmentation techniques and adversarial attacks.

摘要: 许多自然现象都具有自相似性，例如人脸的对称性，或者一首歌的重复主题。对这种对称性的研究将使我们能够更深入地了解复杂系统的潜在机制。认识到理解这些模式的重要性，我们提出了一个受几何启发的框架来研究人工神经网络中的此类现象。为此，我们引入了Cantor集的三元结构，它是由Georg Cantor在$19世纪引入的。在数学中，康托集是位于一条直线上的一组点，它是自相似的，并且具有不可计数的无限零集的反直觉性质。同样，我们引入了CATORNet作为沙盒，通过新的拓扑和几何复杂性度量来研究自相似性。CatorNet构成了一族RELU神经网络，它跨越了可能的Kolmogorov复杂性的整个频谱，包括两种相反的描述(通过描述长度衡量的线性和指数)。广电网络的决策界限可以是任意模糊的，但从分析上讲是已知的。除了作为复杂性度量的试验场，我们的工作还可以用来说明几何学中的潜在陷阱--无知的数据增强技术和对抗性攻击。



## **9. Bones of Contention: Exploring Query-Efficient Attacks Against Skeleton Recognition Systems**

争夺之骨：探索针对骨架识别系统的查询高效攻击 cs.CR

13 pages, 13 figures

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16843v1) [paper-pdf](http://arxiv.org/pdf/2501.16843v1)

**Authors**: Yuxin Cao, Kai Ye, Derui Wang, Minhui Xue, Hao Ge, Chenxiong Qian, Jin Song Dong

**Abstract**: Skeleton action recognition models have secured more attention than video-based ones in various applications due to privacy preservation and lower storage requirements. Skeleton data are typically transmitted to cloud servers for action recognition, with results returned to clients via Apps/APIs. However, the vulnerability of skeletal models against adversarial perturbations gradually reveals the unreliability of these systems. Existing black-box attacks all operate in a decision-based manner, resulting in numerous queries that hinder efficiency and feasibility in real-world applications. Moreover, all attacks off the shelf focus on only restricted perturbations, while ignoring model weaknesses when encountered with non-semantic perturbations. In this paper, we propose two query-effIcient Skeletal Adversarial AttaCks, ISAAC-K and ISAAC-N. As a black-box attack, ISAAC-K utilizes Grad-CAM in a surrogate model to extract key joints where minor sparse perturbations are then added to fool the classifier. To guarantee natural adversarial motions, we introduce constraints of both bone length and temporal consistency. ISAAC-K finds stronger adversarial examples on $\ell_\infty$ norm, which can encompass those on other norms. Exhaustive experiments substantiate that ISAAC-K can uplift the attack efficiency of the perturbations under 10 skeletal models. Additionally, as a byproduct, ISAAC-N fools the classifier by replacing skeletons unrelated to the action. We surprisingly find that skeletal models are vulnerable to large perturbations where the part-wise non-semantic joints are just replaced, leading to a query-free no-box attack without any prior knowledge. Based on that, four adaptive defenses are eventually proposed to improve the robustness of skeleton recognition models.

摘要: 骨骼动作识别模型由于隐私保护和较低的存储要求，在各种应用中获得了比基于视频的模型更多的关注。骨架数据通常被传输到云服务器进行动作识别，结果通过App/API返回给客户端。然而，骨架模型对对抗性扰动的脆弱性逐渐暴露出这些系统的不可靠性。现有的黑盒攻击都是以基于决策的方式运行的，导致了大量的查询，阻碍了现实世界应用程序的效率和可行性。此外，所有现成的攻击都只关注有限的扰动，而在遇到非语义扰动时忽略了模型的弱点。在本文中，我们提出了两种查询高效的骨架对抗攻击：Isaac-K和Isaac-N。作为一种黑盒攻击，Isaac-K利用代理模型中的Grad-CAM来提取关键节点，然后添加微小的稀疏扰动来愚弄分类器。为了保证自然的对抗性运动，我们引入了骨骼长度和时间一致性的约束。Isaac-K在$\ell_\inty$Norm上找到了更强的对抗性例子，它可以包含那些在其他范数上的例子。详尽的实验证明，在10种骨架模型下，Isaac-K可以提高摄动的攻击效率。此外，作为副产品，Isaac-N通过替换与动作无关的骨架来愚弄分类器。我们惊讶地发现，骨架模型容易受到大扰动的影响，其中部分非语义关节被替换，导致在没有任何先验知识的情况下进行无查询的无框攻击。在此基础上，提出了四种自适应防御机制来提高骨架识别模型的鲁棒性。



## **10. HateBench: Benchmarking Hate Speech Detectors on LLM-Generated Content and Hate Campaigns**

HateBench：对LLM生成的内容和仇恨活动的仇恨言语检测器进行基准测试 cs.CR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16750v1) [paper-pdf](http://arxiv.org/pdf/2501.16750v1)

**Authors**: Xinyue Shen, Yixin Wu, Yiting Qu, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: Large Language Models (LLMs) have raised increasing concerns about their misuse in generating hate speech. Among all the efforts to address this issue, hate speech detectors play a crucial role. However, the effectiveness of different detectors against LLM-generated hate speech remains largely unknown. In this paper, we propose HateBench, a framework for benchmarking hate speech detectors on LLM-generated hate speech. We first construct a hate speech dataset of 7,838 samples generated by six widely-used LLMs covering 34 identity groups, with meticulous annotations by three labelers. We then assess the effectiveness of eight representative hate speech detectors on the LLM-generated dataset. Our results show that while detectors are generally effective in identifying LLM-generated hate speech, their performance degrades with newer versions of LLMs. We also reveal the potential of LLM-driven hate campaigns, a new threat that LLMs bring to the field of hate speech detection. By leveraging advanced techniques like adversarial attacks and model stealing attacks, the adversary can intentionally evade the detector and automate hate campaigns online. The most potent adversarial attack achieves an attack success rate of 0.966, and its attack efficiency can be further improved by $13-21\times$ through model stealing attacks with acceptable attack performance. We hope our study can serve as a call to action for the research community and platform moderators to fortify defenses against these emerging threats.

摘要: 大型语言模型(LLM)在生成仇恨言论时被滥用，这引起了越来越多的关注。在解决这一问题的所有努力中，仇恨言论探测器发挥着至关重要的作用。然而，不同的检测器对LLM产生的仇恨言论的有效性在很大程度上仍不清楚。在这篇文章中，我们提出了一个框架，用于对LLM生成的仇恨言论进行仇恨言语检测器的基准测试。我们首先构建了一个由6个广泛使用的LLMS生成的7838个样本的仇恨语音数据集，覆盖了34个身份组，并由3个标记者进行了细致的标注。然后，我们在LLM生成的数据集上评估了八个具有代表性的仇恨语音检测器的有效性。我们的结果表明，虽然检测器在识别LLM生成的仇恨言论方面通常是有效的，但随着LLMS的更新，它们的性能会下降。我们还揭示了LLM驱动的仇恨运动的潜力，LLM给仇恨言语检测领域带来了新的威胁。通过利用对抗性攻击和模型窃取攻击等先进技术，敌手可以故意避开检测器，并自动在线进行仇恨运动。最强的对抗性攻击达到了0.966的攻击成功率，在攻击性能可接受的情况下，通过模型窃取攻击可以进一步提高攻击效率13-21倍。我们希望我们的研究能够成为研究界和平台主持人的行动号召，以加强对这些新出现的威胁的防御。



## **11. Blockchain Address Poisoning**

区块链地址中毒 cs.CR

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16681v1) [paper-pdf](http://arxiv.org/pdf/2501.16681v1)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on Ethereum and BSC. We identify 13 times the number of attack attempts reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address-generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.

摘要: 在许多区块链中，例如Etherum、Binance Smart Chain(BSC)，用于钱包地址的主要表示是难以记忆的40位十六进制字符串。因此，用户经常从他们最近的交易历史中选择地址，这使得区块链地址中毒成为可能。敌手首先生成与受害者之前交互过的地址相似的地址，然后与受害者交战，对他们的交易历史进行“毒化”。这样做的目的是让受害者错误地向相似的地址发送令牌，而不是预期的收件人。与当代研究相比，本文提供了四个值得注意的贡献。首先，我们开发了一个检测系统，并在以太和BSC上进行了两年多的测量。我们确认的攻击企图数量是之前报告的13倍--总计2.7亿次针对1700万受害者的连锁式攻击。6633起事件已造成至少8380万美元的损失，这使区块链地址中毒成为目前观察到的最大的加密货币钓鱼计划之一。其次，我们使用改进的聚类技术分析了几个大型攻击实体，并对攻击者的盈利能力和竞争能力进行了建模。第三，我们揭示了攻击策略--目标人群、成功条件(地址相似性、时机)和跨链攻击。第四，我们从数学上定义和模拟了各种基于软件和硬件的实施中的相似地址生成过程，并识别了一个似乎使用GPU的大规模攻击者群体。我们还讨论了防御对策。



## **12. Data-Free Model-Related Attacks: Unleashing the Potential of Generative AI**

无数据模型相关攻击：释放生成人工智能的潜力 cs.CR

Accepted at USENIX Security 2025

**SubmitDate**: 2025-01-28    [abs](http://arxiv.org/abs/2501.16671v1) [paper-pdf](http://arxiv.org/pdf/2501.16671v1)

**Authors**: Dayong Ye, Tianqing Zhu, Shang Wang, Bo Liu, Leo Yu Zhang, Wanlei Zhou, Yang Zhang

**Abstract**: Generative AI technology has become increasingly integrated into our daily lives, offering powerful capabilities to enhance productivity. However, these same capabilities can be exploited by adversaries for malicious purposes. While existing research on adversarial applications of generative AI predominantly focuses on cyberattacks, less attention has been given to attacks targeting deep learning models. In this paper, we introduce the use of generative AI for facilitating model-related attacks, including model extraction, membership inference, and model inversion. Our study reveals that adversaries can launch a variety of model-related attacks against both image and text models in a data-free and black-box manner, achieving comparable performance to baseline methods that have access to the target models' training data and parameters in a white-box manner. This research serves as an important early warning to the community about the potential risks associated with generative AI-powered attacks on deep learning models.

摘要: 产生式人工智能技术已经越来越多地融入我们的日常生活，为提高生产力提供了强大的能力。但是，这些相同的功能可能会被恶意攻击者利用。虽然现有的关于生成性人工智能的对抗性应用的研究主要集中在网络攻击上，但对针对深度学习模型的攻击的关注较少。在本文中，我们介绍了产生式人工智能用于促进与模型相关的攻击，包括模型提取、隶属度推理和模型反转。我们的研究表明，攻击者可以以无数据和黑盒的方式对图像和文本模型发起各种与模型相关的攻击，获得与以白盒方式访问目标模型的训练数据和参数的基线方法相当的性能。这项研究是对社区的一个重要的早期警告，即与生成性人工智能支持的对深度学习模型的攻击相关的潜在风险。



## **13. Self-interpreting Adversarial Images**

自我解释对抗图像 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2407.08970v3) [paper-pdf](http://arxiv.org/pdf/2407.08970v3)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer their outputs to express an adversary-chosen style, sentiment, or point of view. Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible--yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions. We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.

摘要: 我们引入了一种针对视觉语言模型的新型间接、跨模式注入攻击，该攻击能够创建自解释图像。这些图像包含隐藏的“元指令”，控制模型如何回答用户关于图像的问题，并引导他们的输出来表达对手选择的风格、情绪或观点。自我解释图像充当软提示，限制模型以满足对手的(元)目标，同时仍根据图像的可视内容生成答案。元指令因此是一种更强的即时注入形式。对抗性的图像看起来很自然，模型的答案是连贯的和似是而非的--但它们也遵循对手选择的解释，例如政治宣传，甚至是无法通过明确的文本说明实现的目标。我们评估了各种模型、解释和用户提示的自我解释图像的有效性。我们描述了这些攻击如何通过创建带有垃圾邮件、错误信息或旋转的自我解释内容来造成危害。最后，我们讨论防御措施。



## **14. Targeting Alignment: Extracting Safety Classifiers of Aligned LLMs**

目标对齐：提取对齐的LLM的安全分类器 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16534v1) [paper-pdf](http://arxiv.org/pdf/2501.16534v1)

**Authors**: Jean-Charles Noirot Ferrand, Yohan Beugin, Eric Pauley, Ryan Sheatsley, Patrick McDaniel

**Abstract**: Alignment in large language models (LLMs) is used to enforce guidelines such as safety. Yet, alignment fails in the face of jailbreak attacks that modify inputs to induce unsafe outputs. In this paper, we present and evaluate a method to assess the robustness of LLM alignment. We observe that alignment embeds a safety classifier in the target model that is responsible for deciding between refusal and compliance. We seek to extract an approximation of this classifier, called a surrogate classifier, from the LLM. We develop an algorithm for identifying candidate classifiers from subsets of the LLM model. We evaluate the degree to which the candidate classifiers approximate the model's embedded classifier in benign (F1 score) and adversarial (using surrogates in a white-box attack) settings. Our evaluation shows that the best candidates achieve accurate agreement (an F1 score above 80%) using as little as 20% of the model architecture. Further, we find attacks mounted on the surrogate models can be transferred with high accuracy. For example, a surrogate using only 50% of the Llama 2 model achieved an attack success rate (ASR) of 70%, a substantial improvement over attacking the LLM directly, where we only observed a 22% ASR. These results show that extracting surrogate classifiers is a viable (and highly effective) means for modeling (and therein addressing) the vulnerability of aligned models to jailbreaking attacks.

摘要: 大型语言模型(LLM)中的对齐用于执行安全等准则。然而，面对修改输入以产生不安全输出的越狱攻击，对齐失败。在本文中，我们提出并评价了一种评估LLM配准稳健性的方法。我们观察到，对齐在目标模型中嵌入了一个安全分类器，该分类器负责在拒绝和遵从之间做出决定。我们试图从LLM中提取该分类器的近似值，称为代理分类器。我们提出了一种从LLM模型的子集中识别候选分类器的算法。我们评估了在良性(F1分数)和对抗性(在白盒攻击中使用代理)环境下，候选分类器与模型嵌入分类器的近似程度。我们的评估显示，最好的候选者仅使用模型体系结构的20%就可以实现准确的一致性(F1得分超过80%)。此外，我们发现安装在代理模型上的攻击可以高精度地转移。例如，一个只使用50%的Llama 2模型的代理程序实现了70%的攻击成功率(ASR)，与我们只观察到22%的ASR的直接攻击LLM相比，这是一个实质性的改进。这些结果表明，提取代理分类器是一种可行的(并且非常有效的)方法，用于建模(并在其中解决)对齐模型对越狱攻击的脆弱性。



## **15. Smoothed Embeddings for Robust Language Models**

稳健语言模型的平滑嵌入 cs.LG

Presented in the Safe Generative AI Workshop at NeurIPS 2024

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16497v1) [paper-pdf](http://arxiv.org/pdf/2501.16497v1)

**Authors**: Ryo Hase, Md Rafi Ur Rashid, Ashley Lewis, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang

**Abstract**: Improving the safety and reliability of large language models (LLMs) is a crucial aspect of realizing trustworthy AI systems. Although alignment methods aim to suppress harmful content generation, LLMs are often still vulnerable to jailbreaking attacks that employ adversarial inputs that subvert alignment and induce harmful outputs. We propose the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense, which adds random noise to the embedding vectors and performs aggregation during the generation of each output token, with the aim of better preserving semantic information. Our experiments demonstrate that our approach achieves superior robustness versus utility tradeoffs compared to the baseline defenses.

摘要: 提高大型语言模型（LLM）的安全性和可靠性是实现值得信赖的人工智能系统的一个重要方面。尽管对齐方法的目的是抑制有害内容的生成，但LLM通常仍然容易受到越狱攻击，这些攻击采用颠覆对齐并引发有害输出的对抗性输入。我们提出了随机嵌入平滑和令牌聚合（RESTA）防御，它向嵌入载体添加随机噪音，并在每个输出令牌的生成过程中执行聚合，目的是更好地保存语义信息。我们的实验表明，与基线防御相比，我们的方法实现了更好的鲁棒性与效用权衡。



## **16. Towards Robust Stability Prediction in Smart Grids: GAN-based Approach under Data Constraints and Adversarial Challenges**

智能电网中的稳健稳定性预测：数据约束和对抗挑战下的基于GAN的方法 cs.CR

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.16490v1) [paper-pdf](http://arxiv.org/pdf/2501.16490v1)

**Authors**: Emad Efatinasab, Alessandro Brighente, Denis Donadel, Mauro Conti, Mirco Rampazzo

**Abstract**: Smart grids are critical for addressing the growing energy demand due to global population growth and urbanization. They enhance efficiency, reliability, and sustainability by integrating renewable energy. Ensuring their availability and safety requires advanced operational control and safety measures. Researchers employ AI and machine learning to assess grid stability, but challenges like the lack of datasets and cybersecurity threats, including adversarial attacks, persist. In particular, data scarcity is a key issue: obtaining grid instability instances is tough due to the need for significant expertise, resources, and time. However, they are essential to test novel research advancements and security mitigations. In this paper, we introduce a novel framework to detect instability in smart grids by employing only stable data. It relies on a Generative Adversarial Network (GAN) where the generator is trained to create instability data that are used along with stable data to train the discriminator. Moreover, we include a new adversarial training layer to improve robustness against adversarial attacks. Our solution, tested on a dataset composed of real-world stable and unstable samples, achieve accuracy up to 97.5\% in predicting grid stability and up to 98.9\% in detecting adversarial attacks. Moreover, we implemented our model in a single-board computer demonstrating efficient real-time decision-making with an average response time of less than 7ms. Our solution improves prediction accuracy and resilience while addressing data scarcity in smart grid management.

摘要: 智能电网对于解决全球人口增长和城市化带来的日益增长的能源需求至关重要。它们通过整合可再生能源来提高效率、可靠性和可持续性。确保它们的可用性和安全性需要先进的操作控制和安全措施。研究人员使用人工智能和机器学习来评估网格稳定性，但缺乏数据集和包括对抗性攻击在内的网络安全威胁等挑战依然存在。特别是，数据稀缺是一个关键问题：由于需要大量的专业知识、资源和时间，获取网格不稳定实例非常困难。然而，它们对于测试新的研究进展和安全缓解是必不可少的。在本文中，我们介绍了一种仅使用稳定数据来检测智能电网不稳定性的新框架。它依赖于生成性对抗网络(GAN)，其中生成器被训练来创建不稳定数据，这些数据与稳定数据一起用于训练鉴别器。此外，我们还包括一个新的对抗性训练层，以提高对对抗性攻击的健壮性。我们的解决方案在由真实世界稳定和不稳定样本组成的数据集上进行了测试，预测网格稳定性的准确率高达97.5%，检测对手攻击的准确率高达98.9%。此外，我们在单板计算机上实现了我们的模型，展示了高效的实时决策，平均响应时间不到7ms。我们的解决方案提高了预测的准确性和弹性，同时解决了智能电网管理中的数据稀缺问题。



## **17. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

LLM攻击者：使用大型语言模型增强自动驾驶的闭环对抗场景生成 cs.LG

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15850v1) [paper-pdf](http://arxiv.org/pdf/2501.15850v1)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.

摘要: 确保和提高自动驾驶系统(ADS)的安全性对于高度自动化车辆的部署至关重要，特别是在安全关键事件中。为了解决这种稀缺性问题，开发了对抗性场景生成方法，在该方法中，交通参与者的行为被操纵以引发安全关键事件。然而，现有的方法仍然面临着两个局限性。首先，对抗性参与者的识别直接影响到生成的有效性。然而，现实世界场景的复杂性，参与者众多，行为多样，使得识别具有挑战性。其次，生成的安全关键场景持续提高广告性能的潜力仍未得到充分开发。为了解决这些问题，我们提出了LLM-攻击者：一个利用大型语言模型(LLMS)的闭环对抗性场景生成框架。具体地说，多个LLM代理被设计和协调以识别最佳攻击者。然后，攻击者的轨迹被优化以生成对抗性场景。这些场景根据广告的表现进行迭代细化，形成一个反馈循环来改进广告。实验结果表明，LLM-攻击者能够产生比其他方法更危险的场景，并且用它训练的ADS的冲突率是正常场景训练的一半。这表明了LLM-攻击者测试和增强ADS安全性和健壮性的能力。提供视频演示，网址为：https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.



## **18. Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis**

通过多模式执行路径分析实现高精度勒索软件检测的智能代码嵌入框架 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15836v1) [paper-pdf](http://arxiv.org/pdf/2501.15836v1)

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats.

摘要: 现代威胁形势继续发展，日益复杂，对传统检测方法提出了挑战，需要能够应对复杂对抗战术的创新解决方案。提出了一种新的框架，通过多模式执行路径分析识别勒索软件活动，结合高维嵌入和动态启发式派生机制来捕获不同攻击变量的行为模式。该方法表现出高度的适应性，有效地缓解了勒索软件家族经常利用的混淆策略和多态特征来逃避检测。全面的实验评估显示，与基准技术相比，尤其是在加密速度可变和执行流模糊的情况下，精确度、召回率和准确率指标都有了显著的进步。该框架实现了可扩展和计算效率高的性能，确保了从资源受限环境到高性能基础设施等一系列系统配置的强大适用性。值得注意的发现包括降低了假阳性率和增加了检测延迟，即使对于使用复杂加密机制的勒索软件系列也是如此。模块化设计允许无缝集成其他医疗设备，实现了针对新出现的威胁载体的可扩展性和面向未来的保护。量化分析进一步强调了该系统的能效，强调了其在具有严格业务限制的环境中部署的实用性。这些结果突显了整合先进的计算技术和动态适应性以保护数字生态系统免受日益复杂的威胁的重要性。



## **19. A Privacy Model for Classical & Learned Bloom Filters**

经典和习得的布鲁姆过滤器的隐私模型 cs.CR

**SubmitDate**: 2025-01-27    [abs](http://arxiv.org/abs/2501.15751v1) [paper-pdf](http://arxiv.org/pdf/2501.15751v1)

**Authors**: Hayder Tirmazi

**Abstract**: The Classical Bloom Filter (CBF) is a class of Probabilistic Data Structures (PDS) for handling Approximate Query Membership (AMQ). The Learned Bloom Filter (LBF) is a recently proposed class of PDS that combines the Classical Bloom Filter with a Learning Model while preserving the Bloom Filter's one-sided error guarantees. Bloom Filters have been used in settings where inputs are sensitive and need to be private in the presence of an adversary with access to the Bloom Filter through an API or in the presence of an adversary who has access to the internal state of the Bloom Filter. Prior work has investigated the privacy of the Classical Bloom Filter providing attacks and defenses under various privacy definitions. In this work, we formulate a stronger differential privacy-based model for the Bloom Filter. We propose constructions of the Classical and Learned Bloom Filter that satisfy $(\epsilon, 0)$-differential privacy. This is also the first work that analyses and addresses the privacy of the Learned Bloom Filter under any rigorous model, which is an open problem.

摘要: 经典Bloom Filter(CBF)是一类用于处理近似查询成员身份的概率数据结构(PDS)。学习Bloom Filter(LBF)是最近提出的一类PDS，它将经典Bloom Filter与学习模型相结合，同时保持Bloom Filter的单边误差保证。Bloom Filter已用于输入敏感且需要在通过API访问Bloom Filter的敌手在场的情况下，或在能够访问Bloom Filter内部状态的敌手在场的情况下进行保密的设置。以前的工作已经研究了经典Bloom Filter的隐私，在不同的隐私定义下提供攻击和防御。在这项工作中，我们为Bloom Filter建立了一个更强的基于差分隐私的模型。我们提出了满足$(\epsilon，0)$-差分隐私的经典Bloom过滤器和学习Bloom过滤器的构造。这也是第一个在任何严格模型下分析和解决学习的Bloom Filter隐私的工作，这是一个开放的问题。



## **20. Adversarially Robust Out-of-Distribution Detection Using Lyapunov-Stabilized Embeddings**

使用Lyapunov稳定嵌入的对抗鲁棒性分布外检测 cs.LG

Accepted at the International Conference on Learning Representations  (ICLR) 2025. Code and pre-trained models are available at  https://github.com/AdaptiveMotorControlLab/AROS

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2410.10744v2) [paper-pdf](http://arxiv.org/pdf/2410.10744v2)

**Authors**: Hossein Mirzaei, Mackenzie W. Mathis

**Abstract**: Despite significant advancements in out-of-distribution (OOD) detection, existing methods still struggle to maintain robustness against adversarial attacks, compromising their reliability in critical real-world applications. Previous studies have attempted to address this challenge by exposing detectors to auxiliary OOD datasets alongside adversarial training. However, the increased data complexity inherent in adversarial training, and the myriad of ways that OOD samples can arise during testing, often prevent these approaches from establishing robust decision boundaries. To address these limitations, we propose AROS, a novel approach leveraging neural ordinary differential equations (NODEs) with Lyapunov stability theorem in order to obtain robust embeddings for OOD detection. By incorporating a tailored loss function, we apply Lyapunov stability theory to ensure that both in-distribution (ID) and OOD data converge to stable equilibrium points within the dynamical system. This approach encourages any perturbed input to return to its stable equilibrium, thereby enhancing the model's robustness against adversarial perturbations. To not use additional data, we generate fake OOD embeddings by sampling from low-likelihood regions of the ID data feature space, approximating the boundaries where OOD data are likely to reside. To then further enhance robustness, we propose the use of an orthogonal binary layer following the stable feature space, which maximizes the separation between the equilibrium points of ID and OOD samples. We validate our method through extensive experiments across several benchmarks, demonstrating superior performance, particularly under adversarial attacks. Notably, our approach improves robust detection performance from 37.8% to 80.1% on CIFAR-10 vs. CIFAR-100 and from 29.0% to 67.0% on CIFAR-100 vs. CIFAR-10.

摘要: 尽管在分发外(OOD)检测方面有了很大的进步，但现有的方法仍然难以保持对对手攻击的健壮性，从而影响了它们在关键现实应用中的可靠性。以前的研究试图通过将探测器暴露于辅助OOD数据集以及对抗性训练来解决这一挑战。然而，对抗性训练中固有的增加的数据复杂性，以及OOD样本在测试过程中可能出现的各种方式，往往阻碍这些方法建立稳健的决策边界。为了克服这些局限性，我们提出了一种新的方法AROS，它利用Lyapunov稳定性定理来利用神经常微分方程组(节点)来获得用于OOD检测的稳健嵌入。通过引入定制的损失函数，我们应用Lyapunov稳定性理论来确保内分布(ID)和OOD数据都收敛到动力系统中的稳定平衡点。这种方法鼓励任何扰动的输入返回到其稳定的平衡，从而增强模型对对抗性扰动的稳健性。为了不使用额外的数据，我们通过从ID数据特征空间的低似然区域采样，逼近OOD数据可能驻留的边界来生成虚假的OOD嵌入。为了进一步增强稳健性，我们提出了在稳定特征空间之后使用一个正交二值层，最大化了ID和OOD样本的平衡点之间的分离。我们通过在几个基准上的大量实验来验证我们的方法，展示了优越的性能，特别是在对抗性攻击下。值得注意的是，我们的方法将健壮性检测性能从CIFAR-10上的37.8%提高到CIFAR-100上的80.1%，以及CIFAR-100上的29.0%到CIFAR-10上的67.0%。



## **21. FIT-Print: Towards False-claim-resistant Model Ownership Verification via Targeted Fingerprint**

FIT-Print：通过目标指纹实现抗虚假声明的模型所有权验证 cs.CR

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15509v1) [paper-pdf](http://arxiv.org/pdf/2501.15509v1)

**Authors**: Shuo Shao, Haozhe Zhu, Hongwei Yao, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Model fingerprinting is a widely adopted approach to safeguard the intellectual property rights of open-source models by preventing their unauthorized reuse. It is promising and convenient since it does not necessitate modifying the protected model. In this paper, we revisit existing fingerprinting methods and reveal that they are vulnerable to false claim attacks where adversaries falsely assert ownership of any third-party model. We demonstrate that this vulnerability mostly stems from their untargeted nature, where they generally compare the outputs of given samples on different models instead of the similarities to specific references. Motivated by these findings, we propose a targeted fingerprinting paradigm (i.e., FIT-Print) to counteract false claim attacks. Specifically, FIT-Print transforms the fingerprint into a targeted signature via optimization. Building on the principles of FIT-Print, we develop bit-wise and list-wise black-box model fingerprinting methods, i.e., FIT-ModelDiff and FIT-LIME, which exploit the distance between model outputs and the feature attribution of specific samples as the fingerprint, respectively. Extensive experiments on benchmark models and datasets verify the effectiveness, conferrability, and resistance to false claim attacks of our FIT-Print.

摘要: 模型指纹识别是一种广泛采用的方法，通过防止未经授权的重复使用来保护开放源码模型的知识产权。由于它不需要修改受保护的模型，因此它是有希望的和方便的。在这篇文章中，我们重新审视了现有的指纹识别方法，并揭示了它们容易受到虚假声明攻击，即对手错误地断言任何第三方模型的所有权。我们证明，该漏洞主要源于它们的无针对性，即它们通常比较不同模型上给定样本的输出，而不是与特定参考的相似性。受这些发现的启发，我们提出了一种有针对性的指纹识别范式(即Fit-print)来对抗虚假声明攻击。具体地说，Fit-Print通过优化将指纹转换为目标签名。基于Fit-print的原理，我们提出了逐位和逐列的黑盒模型指纹识别方法，即Fit-ModelDiff和Fit-LIME，它们分别利用模型输出之间的距离和特定样本的特征属性作为指纹。在基准模型和数据集上的广泛实验验证了我们的Fit-print的有效性、可授权性和对虚假声明攻击的抵抗力。



## **22. Mitigating Spurious Negative Pairs for Robust Industrial Anomaly Detection**

缓解伪负对以实现稳健的工业异常检测 cs.CV

Accepted at the 13th International Conference on Learning  Representations (ICLR) 2025

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15434v1) [paper-pdf](http://arxiv.org/pdf/2501.15434v1)

**Authors**: Hossein Mirzaei, Mojtaba Nafez, Jafar Habibi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Despite significant progress in Anomaly Detection (AD), the robustness of existing detection methods against adversarial attacks remains a challenge, compromising their reliability in critical real-world applications such as autonomous driving. This issue primarily arises from the AD setup, which assumes that training data is limited to a group of unlabeled normal samples, making the detectors vulnerable to adversarial anomaly samples during testing. Additionally, implementing adversarial training as a safeguard encounters difficulties, such as formulating an effective objective function without access to labels. An ideal objective function for adversarial training in AD should promote strong perturbations both within and between the normal and anomaly groups to maximize margin between normal and anomaly distribution. To address these issues, we first propose crafting a pseudo-anomaly group derived from normal group samples. Then, we demonstrate that adversarial training with contrastive loss could serve as an ideal objective function, as it creates both inter- and intra-group perturbations. However, we notice that spurious negative pairs compromise the conventional contrastive loss to achieve robust AD. Spurious negative pairs are those that should be closely mapped but are erroneously separated. These pairs introduce noise and misguide the direction of inter-group adversarial perturbations. To overcome the effect of spurious negative pairs, we define opposite pairs and adversarially pull them apart to strengthen inter-group perturbations. Experimental results demonstrate our superior performance in both clean and adversarial scenarios, with a 26.1% improvement in robust detection across various challenging benchmark datasets. The implementation of our work is available at: https://github.com/rohban-lab/COBRA.

摘要: 尽管异常检测(AD)取得了重大进展，但现有检测方法对对手攻击的稳健性仍然是一个挑战，损害了它们在诸如自动驾驶等关键现实世界应用中的可靠性。这个问题主要来自AD设置，该设置假设训练数据限于一组未标记的正常样本，使得检测器在测试过程中容易受到敌意异常样本的影响。此外，实施对抗性训练作为一种保障遇到了困难，例如在没有标签的情况下制定有效的目标函数。AD对抗性训练的理想目标函数应促进正常组和异常组内部和组之间的强烈扰动，以最大化正常和异常分布之间的差值。为了解决这些问题，我们首先提出了一种从正常组样本中提取的伪异常组。然后，我们证明了对抗性训练和对比损失可以作为一个理想的目标函数，因为它造成了组内和组内的扰动。然而，我们注意到，虚假负对折衷了传统的对比损失，以实现稳健的AD。虚假负对是那些应该被紧密映射但被错误地分开的负对。这些对引入了噪声并误导了组间对抗性扰动的方向。为了克服虚假负对的影响，我们定义了相反的对，并相反地将它们分开以加强组间扰动。实验结果表明，该算法在干净场景和对抗性场景中都具有较好的性能，在各种具有挑战性的基准数据集上，稳健检测的性能提高了26.1%。我们工作的实施可在以下网站上查看：https://github.com/rohban-lab/COBRA.



## **23. MAPPING: Debiasing Graph Neural Networks for Fair Node Classification with Limited Sensitive Information Leakage**

MAPPING：去偏置图神经网络以有限的敏感信息泄露进行公平节点分类 cs.LG

Accepted by WWW Journal. Code is available at  https://github.com/yings0930/MAPPING

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2401.12824v2) [paper-pdf](http://arxiv.org/pdf/2401.12824v2)

**Authors**: Ying Song, Balaji Palanisamy

**Abstract**: Despite remarkable success in diverse web-based applications, Graph Neural Networks(GNNs) inherit and further exacerbate historical discrimination and social stereotypes, which critically hinder their deployments in high-stake domains such as online clinical diagnosis, financial crediting, etc. However, current fairness research that primarily craft on i.i.d data, cannot be trivially replicated to non-i.i.d. graph structures with topological dependence among samples. Existing fair graph learning typically favors pairwise constraints to achieve fairness but fails to cast off dimensional limitations and generalize them into multiple sensitive attributes; besides, most studies focus on in-processing techniques to enforce and calibrate fairness, constructing a model-agnostic debiasing GNN framework at the pre-processing stage to prevent downstream misuses and improve training reliability is still largely under-explored. Furthermore, previous work on GNNs tend to enhance either fairness or privacy individually but few probe into their interplays. In this paper, we propose a novel model-agnostic debiasing framework named MAPPING (\underline{M}asking \underline{A}nd \underline{P}runing and Message-\underline{P}assing train\underline{ING}) for fair node classification, in which we adopt the distance covariance($dCov$)-based fairness constraints to simultaneously reduce feature and topology biases in arbitrary dimensions, and combine them with adversarial debiasing to confine the risks of attribute inference attacks. Experiments on real-world datasets with different GNN variants demonstrate the effectiveness and flexibility of MAPPING. Our results show that MAPPING can achieve better trade-offs between utility and fairness, and mitigate privacy risks of sensitive information leakage.

摘要: 尽管图形神经网络(GNN)在各种基于网络的应用中取得了显著的成功，但它继承并进一步加剧了历史歧视和社会刻板印象，这严重阻碍了它们在高风险领域的部署，如在线临床诊断、金融信贷等。然而，目前主要基于身份识别数据的公平研究不能简单地复制到非身份识别领域。样本间具有拓扑依赖关系的图结构。现有的公平图学习一般倾向于两两约束来实现公平性，但未能摆脱维度限制并将其概括为多个敏感属性；此外，大多数研究侧重于内处理技术来加强和校准公平性，在前处理阶段构建模型不可知的去偏向GNN框架以防止下游误用，提高训练可靠性，还在很大程度上探索不足。此外，以前关于GNN的工作往往会单独提高公平性或隐私性，但很少有人探讨它们之间的相互作用。针对公平节点分类问题，提出了一种新的模型不可知去偏框架：映射(下划线{M}询问{A}和下划线{P}运行，消息下划线{P}通过训练\下划线{ING})，其中我们采用基于距离协方差($dCov$)的公平性约束来同时减少任意维度上的特征和拓扑偏差，并将其与对抗性去偏向相结合来控制属性推理攻击的风险。在具有不同GNN变量的真实数据集上的实验证明了该映射的有效性和灵活性。我们的结果表明，映射可以在效用和公平性之间实现更好的权衡，并降低敏感信息泄露的隐私风险。



## **24. Hiding in Plain Sight: An IoT Traffic Camouflage Framework for Enhanced Privacy**

隐藏在众目睽睽之下：增强隐私的物联网交通伪装框架 cs.CR

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15395v1) [paper-pdf](http://arxiv.org/pdf/2501.15395v1)

**Authors**: Daniel Adu Worae, Spyridon Mastorakis

**Abstract**: The rapid growth of Internet of Things (IoT) devices has introduced significant challenges to privacy, particularly as network traffic analysis techniques evolve. While encryption protects data content, traffic attributes such as packet size and timing can reveal sensitive information about users and devices. Existing single-technique obfuscation methods, such as packet padding, often fall short in dynamic environments like smart homes due to their predictability, making them vulnerable to machine learning-based attacks. This paper introduces a multi-technique obfuscation framework designed to enhance privacy by disrupting traffic analysis. The framework leverages six techniques-Padding, Padding with XORing, Padding with Shifting, Constant Size Padding, Fragmentation, and Delay Randomization-to obscure traffic patterns effectively. Evaluations on three public datasets demonstrate significant reductions in classifier performance metrics, including accuracy, precision, recall, and F1 score. We assess the framework's robustness against adversarial tactics by retraining and fine-tuning neural network classifiers on obfuscated traffic. The results reveal a notable degradation in classifier performance, underscoring the framework's resilience against adaptive attacks. Furthermore, we evaluate communication and system performance, showing that higher obfuscation levels enhance privacy but may increase latency and communication overhead.

摘要: 物联网(IoT)设备的快速增长给隐私带来了重大挑战，特别是随着网络流量分析技术的发展。虽然加密保护数据内容，但数据包大小和计时等流量属性可能会泄露有关用户和设备的敏感信息。现有的单一技术混淆方法，如分组填充，由于其可预测性，在智能家居等动态环境中往往达不到要求，使它们容易受到基于机器学习的攻击。本文介绍了一种多技术混淆框架，旨在通过扰乱流量分析来增强隐私。该框架利用六种技术--填充、异或填充、移位填充、固定大小填充、分段和延迟随机化--来有效地模糊流量模式。对三个公共数据集的评估表明，分类器性能指标显著降低，包括准确率、精确度、召回率和F1分数。我们通过对混淆流量进行再训练和微调神经网络分类器来评估该框架对敌对策略的稳健性。结果表明，分类器性能显著下降，突显了该框架对自适应攻击的弹性。此外，我们评估了通信和系统性能，表明较高的混淆级别增强了隐私，但可能会增加延迟和通信开销。



## **25. AI-Driven Secure Data Sharing: A Trustworthy and Privacy-Preserving Approach**

人工智能驱动的安全数据共享：值得信赖和保护隐私的方法 cs.CR

6 pages, 4 figures

**SubmitDate**: 2025-01-26    [abs](http://arxiv.org/abs/2501.15363v1) [paper-pdf](http://arxiv.org/pdf/2501.15363v1)

**Authors**: Al Amin, Kamrul Hasan, Sharif Ullah, Liang Hong

**Abstract**: In the era of data-driven decision-making, ensuring the privacy and security of shared data is paramount across various domains. Applying existing deep neural networks (DNNs) to encrypted data is critical and often compromises performance, security, and computational overhead. To address these limitations, this research introduces a secure framework consisting of a learnable encryption method based on the block-pixel operation to encrypt the data and subsequently integrate it with the Vision Transformer (ViT). The proposed framework ensures data privacy and security by creating unique scrambling patterns per key, providing robust performance against adversarial attacks without compromising computational efficiency and data integrity. The framework was tested on sensitive medical datasets to validate its efficacy, proving its ability to handle highly confidential information securely. The suggested framework was validated with a 94\% success rate after extensive testing on real-world datasets, such as MRI brain tumors and histological scans of lung and colon cancers. Additionally, the framework was tested under diverse adversarial attempts against secure data sharing with optimum performance and demonstrated its effectiveness in various threat scenarios. These comprehensive analyses underscore its robustness, making it a trustworthy solution for secure data sharing in critical applications.

摘要: 在数据驱动决策的时代，确保共享数据的隐私和安全在各个领域都是至关重要的。将现有的深度神经网络(DNN)应用于加密数据是至关重要的，并且通常会损害性能、安全性和计算开销。为了解决这些局限性，本研究提出了一种安全框架，该框架包括一种基于块像素操作的可学习加密方法来加密数据，并随后将其与视觉转换器(VIT)集成。该框架通过为每个密钥创建唯一的加扰模式来确保数据隐私和安全，从而在不影响计算效率和数据完整性的情况下提供对敌意攻击的稳健性能。该框架在敏感的医学数据集上进行了测试，以验证其有效性，证明了其安全处理高度机密信息的能力。在对真实世界的数据集进行广泛的测试后，所建议的框架被验证为94%的成功率，例如MRI脑瘤以及肺癌和结肠癌的组织扫描。此外，该框架在针对安全数据共享的各种敌意尝试下进行了测试，并以最佳性能展示了其在各种威胁场景中的有效性。这些全面的分析强调了它的稳健性，使其成为关键应用程序中安全数据共享的可靠解决方案。



## **26. A theoretical basis for MEV**

MEV的理论基础 cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2302.02154v4) [paper-pdf](http://arxiv.org/pdf/2302.02154v4)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Maximal Extractable Value (MEV) refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream DeFi protocols are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the increasing real-world impact of these attacks, their theoretical foundations remain insufficiently established. We propose a formal theory of MEV, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against MEV attacks.

摘要: 最大可提取价值（MEV）指的是对公共区块链的广泛一类经济攻击，其中有权重新排序、删除或插入区块交易的对手可以从智能合约中“提取”价值。实证研究表明，主流DeFi协议成为这些攻击的大规模目标，对其用户和区块链网络产生了不利影响。尽管这些攻击对现实世界的影响越来越大，但它们的理论基础仍然不够建立。我们基于区块链和智能合约的一般抽象模型，提出了MEV的形式化理论。我们的理论是针对MEV攻击的安全性证明的基础。



## **27. Killing it with Zero-Shot: Adversarially Robust Novelty Detection**

用零射击杀死它：对抗鲁棒的新奇检测 cs.LG

Accepted to the Proceedings of the IEEE International Conference on  Acoustics, Speech, and Signal Processing (ICASSP) 2024

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15271v1) [paper-pdf](http://arxiv.org/pdf/2501.15271v1)

**Authors**: Hossein Mirzaei, Mohammad Jafari, Hamid Reza Dehbashi, Zeinab Sadat Taghavi, Mohammad Sabokrou, Mohammad Hossein Rohban

**Abstract**: Novelty Detection (ND) plays a crucial role in machine learning by identifying new or unseen data during model inference. This capability is especially important for the safe and reliable operation of automated systems. Despite advances in this field, existing techniques often fail to maintain their performance when subject to adversarial attacks. Our research addresses this gap by marrying the merits of nearest-neighbor algorithms with robust features obtained from models pretrained on ImageNet. We focus on enhancing the robustness and performance of ND algorithms. Experimental results demonstrate that our approach significantly outperforms current state-of-the-art methods across various benchmarks, particularly under adversarial conditions. By incorporating robust pretrained features into the k-NN algorithm, we establish a new standard for performance and robustness in the field of robust ND. This work opens up new avenues for research aimed at fortifying machine learning systems against adversarial vulnerabilities. Our implementation is publicly available at https://github.com/rohban-lab/ZARND.

摘要: 新颖性检测(ND)在机器学习中起着至关重要的作用，它在模型推理过程中识别新的或未见的数据。这种能力对于自动化系统的安全可靠运行尤为重要。尽管在这一领域取得了进展，但现有技术在受到对手攻击时往往无法保持其性能。我们的研究通过将最近邻算法的优点与从在ImageNet上预先训练的模型获得的稳健特征相结合来解决这一差距。我们致力于提高ND算法的稳健性和性能。实验结果表明，我们的方法在各种基准测试中的性能明显优于目前最先进的方法，特别是在对抗性条件下。通过将稳健的预训练特征引入到k-NN算法中，我们在稳健ND领域建立了一个新的性能和稳健性标准。这项工作为旨在加强机器学习系统抵御对手漏洞的研究开辟了新的途径。我们的实现可在https://github.com/rohban-lab/ZARND.上公开获得



## **28. Mirage in the Eyes: Hallucination Attack on Multi-modal Large Language Models with Only Attention Sink**

眼中的幻象：对只有注意力下沉的多模式大型语言模型的幻觉攻击 cs.LG

USENIX Security 2025

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15269v1) [paper-pdf](http://arxiv.org/pdf/2501.15269v1)

**Authors**: Yining Wang, Mi Zhang, Junjie Sun, Chenyue Wang, Min Yang, Hui Xue, Jialing Tao, Ranjie Duan, Jiexi Liu

**Abstract**: Fusing visual understanding into language generation, Multi-modal Large Language Models (MLLMs) are revolutionizing visual-language applications. Yet, these models are often plagued by the hallucination problem, which involves generating inaccurate objects, attributes, and relationships that do not match the visual content. In this work, we delve into the internal attention mechanisms of MLLMs to reveal the underlying causes of hallucination, exposing the inherent vulnerabilities in the instruction-tuning process.   We propose a novel hallucination attack against MLLMs that exploits attention sink behaviors to trigger hallucinated content with minimal image-text relevance, posing a significant threat to critical downstream applications. Distinguished from previous adversarial methods that rely on fixed patterns, our approach generates dynamic, effective, and highly transferable visual adversarial inputs, without sacrificing the quality of model responses. Comprehensive experiments on 6 prominent MLLMs demonstrate the efficacy of our attack in compromising black-box MLLMs even with extensive mitigating mechanisms, as well as the promising results against cutting-edge commercial APIs, such as GPT-4o and Gemini 1.5. Our code is available at https://huggingface.co/RachelHGF/Mirage-in-the-Eyes.

摘要: 多模式大型语言模型将视觉理解融合到语言生成中，正在给视觉语言应用带来革命性的变化。然而，这些模型经常受到幻觉问题的困扰，幻觉问题涉及生成与视觉内容不匹配的不准确的对象、属性和关系。在这项工作中，我们深入研究了MLMS的内部注意机制，以揭示产生幻觉的潜在原因，揭示了教学调整过程中的内在弱点。我们提出了一种新的针对MLLMS的幻觉攻击，该攻击利用注意力吸收行为来触发具有最小图文相关性的幻觉内容，从而对关键的下游应用构成重大威胁。与以前依赖固定模式的对抗性方法不同，我们的方法产生动态、有效和高度可转移的视觉对抗性输入，而不牺牲模型响应的质量。在6个重要的MLLMS上的综合实验证明了我们的攻击在危害黑盒MLLMS方面的有效性，即使具有广泛的缓解机制，以及对尖端商业API，如GPT-40和Gemini 1.5的有希望的结果。我们的代码可以在https://huggingface.co/RachelHGF/Mirage-in-the-Eyes.上找到



## **29. Pre-trained Model Guided Mixture Knowledge Distillation for Adversarial Federated Learning**

预训练模型引导的对抗性联邦学习混合知识提炼 cs.CV

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15257v1) [paper-pdf](http://arxiv.org/pdf/2501.15257v1)

**Authors**: Yu Qiao, Huy Q. Le, Apurba Adhikary, Choong Seon Hong

**Abstract**: This paper aims to improve the robustness of a small global model while maintaining clean accuracy under adversarial attacks and non-IID challenges in federated learning. By leveraging the concise knowledge embedded in the class probabilities from a pre-trained model for both clean and adversarial image classification, we propose a Pre-trained Model-guided Adversarial Federated Learning (PM-AFL) training paradigm. This paradigm integrates vanilla mixture and adversarial mixture knowledge distillation to effectively balance accuracy and robustness while promoting local models to learn from diverse data. Specifically, for clean accuracy, we adopt a dual distillation strategy where the class probabilities of randomly paired images and their blended versions are aligned between the teacher model and the local models. For adversarial robustness, we use a similar distillation approach but replace clean samples on the local side with adversarial examples. Moreover, considering the bias between local and global models, we also incorporate a consistency regularization term to ensure that local adversarial predictions stay aligned with their corresponding global clean ones. These strategies collectively enable local models to absorb diverse knowledge from the teacher model while maintaining close alignment with the global model, thereby mitigating overfitting to local optima and enhancing the generalization of the global model. Experiments demonstrate that the PM-AFL-based paradigm outperforms other methods that integrate defense strategies by a notable margin.

摘要: 本文旨在提高联合学习中小全局模型的健壮性，同时在对抗攻击和非IID挑战下保持干净的准确性。通过利用预先训练的模型中嵌入到类别概率中的简明知识，我们提出了一种预先训练的模型制导的对抗性联合学习(PM-AFL)训练范式。该模型结合了香草混合和对抗性混合的知识提取方法，有效地平衡了精确度和稳健性，同时促进了局部模型从不同数据中学习。具体地说，为了保持清晰的准确性，我们采用了双精馏策略，其中随机配对图像及其混合版本的类概率在教师模型和局部模型之间对齐。为了对抗的稳健性，我们使用类似的蒸馏方法，但用对抗的例子取代本地的干净样本。此外，考虑到局部模型和全局模型之间的偏差，我们还加入了一致性正则化项，以确保局部对抗性预测与相应的全局清洁预测保持一致。这些策略共同使局部模型能够从教师模型中吸收不同的知识，同时保持与全局模型的密切一致，从而缓解了对局部最优的过度拟合，并增强了全局模型的泛化。实验表明，基于PM-AFL的方法比其他集成防御策略的方法有显著的优势。



## **30. Comprehensive Evaluation of Cloaking Backdoor Attacks on Object Detector in Real-World**

现实世界中目标检测器隐蔽后门攻击的综合评估 cs.CR

arXiv admin note: text overlap with arXiv:2201.08619

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.15101v1) [paper-pdf](http://arxiv.org/pdf/2501.15101v1)

**Authors**: Hua Ma, Alsharif Abuadbba, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: The exploration of backdoor vulnerabilities in object detectors, particularly in real-world scenarios, remains limited. A significant challenge lies in the absence of a natural physical backdoor dataset, and constructing such a dataset is both time- and labor-intensive. In this work, we address this gap by creating a large-scale dataset comprising approximately 11,800 images/frames with annotations featuring natural objects (e.g., T-shirts and hats) as triggers to incur cloaking adversarial effects in diverse real-world scenarios. This dataset is tailored for the study of physical backdoors in object detectors. Leveraging this dataset, we conduct a comprehensive evaluation of an insidious cloaking backdoor effect against object detectors, wherein the bounding box around a person vanishes when the individual is near a natural object (e.g., a commonly available T-shirt) in front of the detector. Our evaluations encompass three prevalent attack surfaces: data outsourcing, model outsourcing, and the use of pretrained models. The cloaking effect is successfully implanted in object detectors across all three attack surfaces. We extensively evaluate four popular object detection algorithms (anchor-based Yolo-V3, Yolo-V4, Faster R-CNN, and anchor-free CenterNet) using 19 videos (totaling approximately 11,800 frames) in real-world scenarios. Our results demonstrate that the backdoor attack exhibits remarkable robustness against various factors, including movement, distance, angle, non-rigid deformation, and lighting. In data and model outsourcing scenarios, the attack success rate (ASR) in most videos reaches 100% or near it, while the clean data accuracy of the backdoored model remains indistinguishable from that of the clean model, making it impossible to detect backdoor behavior through a validation set.

摘要: 对对象检测器中的后门漏洞的探索，特别是在现实世界的场景中，仍然有限。一个重大的挑战在于缺乏一个自然的物理后门数据集，而构建这样的数据集既耗时又耗力。在这项工作中，我们通过创建一个包含大约11,800张图像/帧的大规模数据集来解决这一问题，该数据集带有以自然对象(例如T恤和帽子)为特征的注释，以此作为触发器在不同的现实世界场景中引起隐蔽的对抗效果。该数据集是为研究物体探测器中的物理后门而定制的。利用该数据集，我们对针对对象检测器的隐蔽后门效应进行了全面评估，其中当个人接近检测器前面的自然对象(例如，常见的T恤)时，该人周围的边界框消失。我们的评估包括三个常见的攻击面：数据外包、模型外包和使用预先训练的模型。伪装效果成功地植入了所有三个攻击面的目标探测器中。在实际场景中，我们使用19个视频(总计约11800帧)对四种流行的目标检测算法(基于锚点的Yolo-V3、Yolo-V4、更快的R-CNN和无锚点的Centernet)进行了广泛的评估。我们的结果表明，后门攻击对移动、距离、角度、非刚性变形和光照等各种因素表现出显著的鲁棒性。在数据和模型外包场景中，大多数视频的攻击成功率(ASR)都达到100%或接近这一水平，而背靠背模型的干净数据准确率与干净模型仍然没有区别，无法通过验证集检测到后门行为。



## **31. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的主动对抗防御 cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2407.15524v5) [paper-pdf](http://arxiv.org/pdf/2407.15524v5)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对对手攻击的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，现有的反对手方法通常在攻击后抵消对手干扰，而我们设计了一种主动战略，通过预先保护媒体来抢占先机，有效地在第三方攻击发生之前消除潜在的对手影响。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。据我们所知，我们还设计了第一个有效的白盒自适应恢复攻击，并证明了除非主干模型、算法和设置完全受损，否则我们的防御策略添加的保护是不可逆转的。这项工作为主动防御对抗性攻击提供了新的方向。



## **32. Poisoning Prevention in Federated Learning and Differential Privacy via Stateful Proofs of Execution**

通过执行状态证明在联邦学习和差异隐私中预防中毒 cs.CR

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2404.06721v4) [paper-pdf](http://arxiv.org/pdf/2404.06721v4)

**Authors**: Norrathep Rattanavipanon, Ivan De Oliveira Nunes

**Abstract**: The rise in IoT-driven distributed data analytics, coupled with increasing privacy concerns, has led to a demand for effective privacy-preserving and federated data collection/model training mechanisms. In response, approaches such as Federated Learning (FL) and Local Differential Privacy (LDP) have been proposed and attracted much attention over the past few years. However, they still share the common limitation of being vulnerable to poisoning attacks wherein adversaries compromising edge devices feed forged (a.k.a. poisoned) data to aggregation back-ends, undermining the integrity of FL/LDP results.   In this work, we propose a system-level approach to remedy this issue based on a novel security notion of Proofs of Stateful Execution (PoSX) for IoT/embedded devices' software. To realize the PoSX concept, we design SLAPP: a System-Level Approach for Poisoning Prevention. SLAPP leverages commodity security features of embedded devices - in particular ARM TrustZoneM security extensions - to verifiably bind raw sensed data to their correct usage as part of FL/LDP edge device routines. As a consequence, it offers robust security guarantees against poisoning. Our evaluation, based on real-world prototypes featuring multiple cryptographic primitives and data collection schemes, showcases SLAPP's security and low overhead.

摘要: 物联网驱动的分布式数据分析的兴起，加上对隐私的日益担忧，导致了对有效的隐私保护和联合数据收集/模型培训机制的需求。在过去的几年里，联邦学习(FL)和局部差异隐私(LDP)等方法被提出并引起了人们的广泛关注。然而，它们仍然有一个共同的局限性，即容易受到中毒攻击，在这些攻击中，危害边缘设备的对手提供伪造的(也称为。有毒)数据到聚合后端，破坏FL/LDP结果的完整性。在这项工作中，我们提出了一种基于物联网/嵌入式设备软件状态执行证明(PoSX)的新的安全概念来解决这一问题。为了实现PoSX的概念，我们设计了SLAPP：一种系统级的中毒预防方法。SLAPP利用嵌入式设备的商用安全功能--尤其是ARM TrustZoneM安全扩展--作为FL/LDP边缘设备例程的一部分，以可验证的方式将原始感测数据与其正确使用绑定在一起。因此，它为防止中毒提供了强有力的安全保障。我们的评估基于具有多个加密原语和数据收集方案的真实世界原型，展示了SLAPP的安全性和低开销。



## **33. VideoPure: Diffusion-based Adversarial Purification for Video Recognition**

VideoPure：基于扩散的视频识别对抗净化 cs.CV

**SubmitDate**: 2025-01-25    [abs](http://arxiv.org/abs/2501.14999v1) [paper-pdf](http://arxiv.org/pdf/2501.14999v1)

**Authors**: Kaixun Jiang, Zhaoyu Chen, Jiyuan Fu, Lingyi Hong, Jinglun Li, Wenqiang Zhang

**Abstract**: Recent work indicates that video recognition models are vulnerable to adversarial examples, posing a serious security risk to downstream applications. However, current research has primarily focused on adversarial attacks, with limited work exploring defense mechanisms. Furthermore, due to the spatial-temporal complexity of videos, existing video defense methods face issues of high cost, overfitting, and limited defense performance. Recently, diffusion-based adversarial purification methods have achieved robust defense performance in the image domain. However, due to the additional temporal dimension in videos, directly applying these diffusion-based adversarial purification methods to the video domain suffers performance and efficiency degradation. To achieve an efficient and effective video adversarial defense method, we propose the first diffusion-based video purification framework to improve video recognition models' adversarial robustness: VideoPure. Given an adversarial example, we first employ temporal DDIM inversion to transform the input distribution into a temporally consistent and trajectory-defined distribution, covering adversarial noise while preserving more video structure. Then, during DDIM denoising, we leverage intermediate results at each denoising step and conduct guided spatial-temporal optimization, removing adversarial noise while maintaining temporal consistency. Finally, we input the list of optimized intermediate results into the video recognition model for multi-step voting to obtain the predicted class. We investigate the defense performance of our method against black-box, gray-box, and adaptive attacks on benchmark datasets and models. Compared with other adversarial purification methods, our method overall demonstrates better defense performance against different attacks. Our code is available at https://github.com/deep-kaixun/VideoPure.

摘要: 最近的研究表明，视频识别模型容易受到敌意例子的攻击，给下游应用带来严重的安全风险。然而，目前的研究主要集中在对抗性攻击上，探索防御机制的工作有限。此外，由于视频的时空复杂性，现有的视频防御方法面临着成本高、过度匹配和防御性能有限的问题。近年来，基于扩散的对抗性净化方法在图像域取得了较好的防御性能。然而，由于视频中存在额外的时间维度，直接将这些基于扩散的对抗性净化方法应用到视频域会导致性能和效率的下降。为了实现高效有效的视频对抗防御方法，我们提出了第一个基于扩散的视频净化框架来提高视频识别模型的对抗健壮性：VideoPure。给出一个对抗性的例子，我们首先使用时间ddim反转将输入分布变换成一个时间上一致的轨迹定义的分布，在覆盖对抗性噪声的同时保留更多的视频结构。然后，在DDIM去噪过程中，我们利用每个去噪步骤的中间结果并进行引导的时空优化，在保持时间一致性的同时去除对抗性噪声。最后，将优化后的中间结果列表输入到视频识别模型中进行多步投票，得到预测类。我们研究了该方法对基准数据集和模型的黑盒、灰盒和自适应攻击的防御性能。与其他对抗性净化方法相比，我们的方法总体上对不同的攻击具有更好的防御性能。我们的代码可以在https://github.com/deep-kaixun/VideoPure.上找到



## **34. Untelegraphable Encryption and its Applications**

不可电报加密及其应用 quant-ph

56 pages

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2410.24189v2) [paper-pdf](http://arxiv.org/pdf/2410.24189v2)

**Authors**: Jeffrey Champion, Fuyuki Kitagawa, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We initiate the study of untelegraphable encryption (UTE), founded on the no-telegraphing principle, which allows an encryptor to encrypt a message such that a binary string representation of the ciphertext cannot be decrypted by a user with the secret key, a task that is classically impossible. This is a natural relaxation of unclonable encryption (UE), inspired by the recent work of Nehoran and Zhandry (ITCS 2024), who showed a computational separation between the no-cloning and no-telegraphing principles. In this work, we define and construct UTE information-theoretically in the plain model. Building off this, we give several applications of UTE and study the interplay of UTE with UE and well-studied tasks in quantum state learning, yielding the following contributions:   - A construction of collusion-resistant UTE from plain secret-key encryption, which we then show denies the existence of hyper-efficient shadow tomography (HEST). By building a relaxation of collusion-resistant UTE, we show the impossibility of HEST assuming only pseudorandom state generators (which may not imply one-way functions). This almost unconditionally answers an open inquiry of Aaronson (STOC 2018).   - A construction of UTE from a one-shot message authentication code in the classical oracle model, such that there is an explicit attack that breaks UE security for an unbounded polynomial number of decryptors.   - A construction of everlasting secure collusion-resistant UTE, where the decryptor adversary can run in unbounded time, in the quantum random oracle model (QROM), and formal evidence that a construction in the plain model is a challenging task. We leverage this construction to show that HEST with unbounded post-processing time is impossible in the QROM.   - Constructions of secret sharing resilient to joint and unbounded classical leakage and untelegraphable functional encryption.

摘要: 我们启动了基于无电报原理的不可远程传送加密(UTE)的研究，该原理允许加密者对消息进行加密，使得密文的二进制字符串表示不能被用户用秘密密钥解密，这是传统上不可能完成的任务。这是不可克隆加密(UE)的自然放松，灵感来自Nehoran和Zhandry最近的工作(ITCS 2024)，他们展示了无克隆和无电报原则之间的计算分离。在这项工作中，我们定义并构造了UTE信息--理论上是在平面模型中。在此基础上，我们给出了UTE的几个应用，并研究了UE与UE之间的相互作用以及量子态学习中的研究任务，取得了以下贡献：-利用明文密钥加密构造了抗合谋的UTE，然后我们证明了该构造否认了超高效阴影层析(HEST)的存在。通过构造抗合谋UTE的松弛，我们证明了仅假设伪随机状态生成器(这可能不包含单向函数)的Hest是不可能的。这几乎无条件地回答了对Aaronson的公开调查(STOC 2018)。-从经典Oracle模型中的一次性消息认证码构造UE，从而存在破坏无限多项式解密器的UE安全的显式攻击。-在量子随机预言模型(QROM)中构造永久安全的抗合谋UTE，其中解密者对手可以在无限的时间内运行，并且形式证据表明在普通模型中的构造是一项具有挑战性的任务。我们利用这种构造来表明，在QROM中，HEST的后处理时间是不受限制的。-构造具有抗联合无界经典泄漏和不可远程抓取的函数加密的秘密共享结构。



## **35. PANTS: Practical Adversarial Network Traffic Samples against ML-powered Networking Classifiers**

PANDS：针对ML支持的网络分类器的实用对抗网络流量示例 cs.CR

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2409.04691v2) [paper-pdf](http://arxiv.org/pdf/2409.04691v2)

**Authors**: Minhao Jin, Maria Apostolaki

**Abstract**: Multiple network management tasks, from resource allocation to intrusion detection, rely on some form of ML-based network traffic classification (MNC). Despite their potential, MNCs are vulnerable to adversarial inputs, which can lead to outages, poor decision-making, and security violations, among other issues. The goal of this paper is to help network operators assess and enhance the robustness of their MNC against adversarial inputs. The most critical step for this is generating inputs that can fool the MNC while being realizable under various threat models. Compared to other ML models, finding adversarial inputs against MNCs is more challenging due to the existence of non-differentiable components e.g., traffic engineering and the need to constrain inputs to preserve semantics and ensure reliability. These factors prevent the direct use of well-established gradient-based methods developed in adversarial ML (AML). To address these challenges, we introduce PANTS, a practical white-box framework that uniquely integrates AML techniques with Satisfiability Modulo Theories (SMT) solvers to generate adversarial inputs for MNCs. We also embed PANTS into an iterative adversarial training process that enhances the robustness of MNCs against adversarial inputs. PANTS is 70% and 2x more likely in median to find adversarial inputs against target MNCs compared to state-of-the-art baselines, namely Amoeba and BAP. PANTS improves the robustness of the target MNCs by 52.7% (even against attackers outside of what is considered during robustification) without sacrificing their accuracy.

摘要: 从资源分配到入侵检测的多个网络管理任务依赖于某种形式的基于ML的网络流量分类(MNC)。尽管跨国公司有潜力，但它们很容易受到敌意输入的影响，这可能会导致停电、糟糕的决策和违反安全规定等问题。本文的目的是帮助网络运营商评估和提高他们的MNC对敌意输入的健壮性。要做到这一点，最关键的一步是生成可以愚弄跨国公司的输入，同时在各种威胁模型下实现。与其他ML模型相比，发现针对跨国公司的敌意输入更具挑战性，这是因为存在不可区分的组件，例如流量工程，以及需要约束输入以保持语义和确保可靠性。这些因素阻碍了在对抗性ML(AML)中发展的基于梯度的方法的直接使用。为了应对这些挑战，我们引入了PANS，这是一个实用的白盒框架，它独特地将AML技术与可满足性模理论(SMT)求解器相结合，为跨国公司生成对抗性输入。我们还将裤子嵌入到一个迭代的对抗性训练过程中，以增强跨国公司对对抗性输入的健壮性。与最先进的基线，即变形虫和BAP相比，PANS在中位数中找到针对目标跨国公司的敌意输入的可能性分别高出70%和2倍。Pants在不牺牲精确度的情况下，将目标MNC的健壮性提高52.7%(即使是在攻击过程中考虑之外的攻击者)。



## **36. Towards Scalable Topological Regularizers**

迈向可扩展的布局调节器 cs.LG

31 pages, accepted to ICLR 2025

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14641v1) [paper-pdf](http://arxiv.org/pdf/2501.14641v1)

**Authors**: Hiu-Tung Wong, Darrick Lee, Hong Yan

**Abstract**: Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.

摘要: 潜在空间匹配由潜在空间中特征的匹配分布组成，是对抗性攻击和防御、领域自适应和产生式建模等任务的重要组成部分。概率度量指标，如Wasserstein和最大均值差异，通常用于量化此类分布之间的差异。然而，这些通常计算成本很高，或者没有适当地考虑分布的几何和拓扑特征。持久同调是一种从拓扑数据分析中量化点云多尺度拓扑结构的工具，近年来被用作学习任务中的拓扑正则化工具。然而，计算成本排除了更大规模的计算，并且梯度中的不连续会导致不稳定的训练行为，例如在对抗性任务中。在计算大量小样本的持久同调的基础上，我们提出使用主持久度量作为拓扑正则化。我们给出了这种正则化算法的并行GPU实现，并证明了对于光滑的密度，梯度是连续的。此外，我们展示了这种正则化算法在形状匹配、图像生成和半监督学习任务中的有效性，为拓扑特征的可伸缩正则化算法打开了大门。



## **37. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Accepted by NeurIPS 2024

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2404.10642v3) [paper-pdf](http://arxiv.org/pdf/2404.10642v3)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Zheng Yuan, Yong Dai, Lei Han, Nan Du, Xiaolong Li

**Abstract**: We explore the potential of self-play training for large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players must have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by Self-Playing this Adversarial language Game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is available at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中对大型语言模型(LLM)进行自我发挥训练的可能性。在这个游戏中，攻击者和防御者围绕一个只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都必须有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们很好奇，通过自我玩这个对抗性语言游戏(SPAG)，LLMS的推理能力是否会进一步增强。带着这个目标，我们选择了几个开源的LLM，让每个LLM扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS的性能在广泛的推理基准上一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLMS的推理能力。代码可在https://github.com/Linear95/SPAG.上获得



## **38. Real-world Edge Neural Network Implementations Leak Private Interactions Through Physical Side Channel**

现实世界的边缘神经网络实现通过物理侧通道泄露私人交互 cs.CR

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14512v1) [paper-pdf](http://arxiv.org/pdf/2501.14512v1)

**Authors**: Zhuoran Liu, Senna van Hoek, Péter Horváth, Dirk Lauret, Xiaoyun Xu, Lejla Batina

**Abstract**: Neural networks have become a fundamental component of numerous practical applications, and their implementations, which are often accelerated by hardware, are integrated into all types of real-world physical devices. User interactions with neural networks on hardware accelerators are commonly considered privacy-sensitive. Substantial efforts have been made to uncover vulnerabilities and enhance privacy protection at the level of machine learning algorithms, including membership inference attacks, differential privacy, and federated learning. However, neural networks are ultimately implemented and deployed on physical devices, and current research pays comparatively less attention to privacy protection at the implementation level. In this paper, we introduce a generic physical side-channel attack, ScaAR, that extracts user interactions with neural networks by leveraging electromagnetic (EM) emissions of physical devices. Our proposed attack is implementation-agnostic, meaning it does not require the adversary to possess detailed knowledge of the hardware or software implementations, thanks to the capabilities of deep learning-based side-channel analysis (DLSCA). Experimental results demonstrate that, through the EM side channel, ScaAR can effectively extract the class label of user interactions with neural classifiers, including inputs and outputs, on the AMD-Xilinx MPSoC ZCU104 FPGA and Raspberry Pi 3 B. In addition, for the first time, we provide side-channel analysis on edge Large Language Model (LLM) implementations on the Raspberry Pi 5, showing that EM side channel leaks interaction data, and different LLM tokens can be distinguishable from the EM traces.

摘要: 神经网络已经成为许多实际应用的基本组件，其实现通常由硬件加速，并集成到所有类型的现实世界物理设备中。用户在硬件加速器上与神经网络的交互通常被认为是隐私敏感的。在发现漏洞和加强机器学习算法层面的隐私保护方面做出了实质性的努力，包括成员资格推理攻击、差异隐私和联合学习。然而，神经网络最终是在物理设备上实现和部署的，目前的研究相对较少关注实现层面的隐私保护。在本文中，我们介绍了一种通用的物理侧通道攻击，ScaAR，它通过利用物理设备的电磁发射来提取用户与神经网络的交互。我们提出的攻击是与实现无关的，这意味着它不需要攻击者拥有硬件或软件实现的详细知识，这要归功于基于深度学习的旁路分析(DLSCA)能力。实验结果表明，在AMD-Xilinx MPSoC ZCU104和Raspberry PI 3 B上，通过EM侧通道，ScaAR可以有效地提取用户与神经分类器交互的类别标签，包括输入和输出。此外，我们首次对Raspberry PI 5上的EDGE大语言模型(LLM)实现进行了侧通道分析，结果表明EM侧通道泄漏了交互数据，并且可以从EM跟踪中区分不同的LLM标记。



## **39. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

Accepted by ICLR2025

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2406.18849v3) [paper-pdf](http://arxiv.org/pdf/2406.18849v3)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 24 advanced open-source LVLMs and 2 close-source LVLMs are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released at \url{https://github.com/Robin-WZQ/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对24个先进的开源LVLMS和2个闭源LVLMS进行了评估，揭示了现有LVLMS的不足。基准发布地址为\url{https://github.com/Robin-WZQ/Dysca}.



## **40. A Note on Implementation Errors in Recent Adaptive Attacks Against Multi-Resolution Self-Ensembles**

关于最近针对多分辨率自集成的自适应攻击中的实现错误的注释 cs.CR

4 pages, 2 figures, technical note addressing an issue in  arXiv:2411.14834v1

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14496v1) [paper-pdf](http://arxiv.org/pdf/2501.14496v1)

**Authors**: Stanislav Fort

**Abstract**: This note documents an implementation issue in recent adaptive attacks (Zhang et al. [2024]) against the multi-resolution self-ensemble defense (Fort and Lakshminarayanan [2024]). The implementation allowed adversarial perturbations to exceed the standard $L_\infty = 8/255$ bound by up to a factor of 20$\times$, reaching magnitudes of up to $L_\infty = 160/255$. When attacks are properly constrained within the intended bounds, the defense maintains non-trivial robustness. Beyond highlighting the importance of careful validation in adversarial machine learning research, our analysis reveals an intriguing finding: properly bounded adaptive attacks against strong multi-resolution self-ensembles often align with human perception, suggesting the need to reconsider how we measure adversarial robustness.

摘要: 本笔记记录了最近针对多分辨率自集成防御（Fort和Lakshminarayanan [2024]）的自适应攻击（Zhang等人[2024]）中的实现问题。该实现允许对抗性扰动超过标准$L_\infty = 8/255$，其限制因子高达20 $\times $，达到高达$L_\infty = 160/255$的量级。当攻击被适当地限制在预期范围内时，防御系统就会保持非凡的鲁棒性。除了强调对抗性机器学习研究中仔细验证的重要性之外，我们的分析还揭示了一个有趣的发现：针对强大的多分辨率自我集成的适当有界自适应攻击通常与人类的感知一致，这表明需要重新考虑如何衡量对抗性鲁棒性。



## **41. Optimal Strategies for Federated Learning Maintaining Client Privacy**

联合学习维护客户隐私的最佳策略 cs.LG

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14453v1) [paper-pdf](http://arxiv.org/pdf/2501.14453v1)

**Authors**: Uday Bhaskar, Varul Srivastava, Avyukta Manjunatha Vummintala, Naresh Manwani, Sujit Gujar

**Abstract**: Federated Learning (FL) emerged as a learning method to enable the server to train models over data distributed among various clients. These clients are protective about their data being leaked to the server, any other client, or an external adversary, and hence, locally train the model and share it with the server rather than sharing the data. The introduction of sophisticated inferencing attacks enabled the leakage of information about data through access to model parameters. To tackle this challenge, privacy-preserving federated learning aims to achieve differential privacy through learning algorithms like DP-SGD. However, such methods involve adding noise to the model, data, or gradients, reducing the model's performance.   This work provides a theoretical analysis of the tradeoff between model performance and communication complexity of the FL system. We formally prove that training for one local epoch per global round of training gives optimal performance while preserving the same privacy budget. We also investigate the change of utility (tied to privacy) of FL models with a change in the number of clients and observe that when clients are training using DP-SGD and argue that for the same privacy budget, the utility improved with increased clients. We validate our findings through experiments on real-world datasets. The results from this paper aim to improve the performance of privacy-preserving federated learning systems.

摘要: 联合学习(FL)作为一种学习方法应运而生，它使服务器能够针对分布在不同客户端的数据训练模型。这些客户端保护它们的数据不被泄露到服务器、任何其他客户端或外部对手，因此在本地训练模型并与服务器共享它，而不是共享数据。复杂推理攻击的引入使得通过访问模型参数来泄露有关数据的信息成为可能。为了应对这一挑战，隐私保护联合学习旨在通过DP-SGD等学习算法来实现差异隐私。然而，这种方法涉及到向模型、数据或渐变中添加噪声，从而降低模型的性能。这项工作从理论上分析了FL系统的模型性能和通信复杂度之间的权衡。我们正式证明，在保持相同的隐私预算的情况下，每一轮全球培训一个本地纪元的培训可以提供最佳性能。我们还研究了FL模型的效用(与隐私相关)随客户端数量的变化而变化，并观察到当客户端使用DP-SGD进行训练时，对于相同的隐私预算，效用随着客户端的增加而提高。我们通过在真实世界的数据集上进行实验来验证我们的发现。本文的研究结果旨在提高隐私保护联邦学习系统的性能。



## **42. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

Siren：一个基于学习的多回合攻击框架，用于模拟现实世界的人类越狱行为 cs.CL

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14250v1) [paper-pdf](http://arxiv.org/pdf/2501.14250v1)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) training set construction utilizing Turn-Level LLM feedback (Turn-MF), (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.

摘要: 大型语言模型(LLM)在实际应用中被广泛使用，这引发了人们对它们的安全性和可信性的担忧。虽然与越狱提示的红色合作暴露了LLMS的漏洞，但目前的努力主要集中在单回合攻击上，而忽略了现实世界对手使用的多回合策略。现有的多回合攻击方法依赖于静态模式或预定义的逻辑链，无法考虑攻击过程中的动态策略。我们提出了一种基于学习的多回合攻击框架Siren，旨在模拟真实世界的人类越狱行为。SIREN包括三个阶段：(1)利用话轮水平LLM反馈(TURN-MF)构建训练集；(2)使用有监督的精调(SFT)和直接偏好优化(DPO)训练后攻击者；(3)攻击和目标LLM之间的交互。实验表明，SIREN在以骆驼-3-8B为攻击者对抗双子座-1.5-Pro为目标机型时，攻击成功率(ASR)为90%，在以米斯特拉尔-7B为攻击目标时，对GPT-40的攻击成功率为70%，显著超过了单回合基线。此外，拥有7B规模模型的SIREN实现了与利用GPT-40作为攻击者的多回合基线相当的性能，同时需要更少的回合，并采用了与攻击目标更好地语义一致的分解策略。我们希望警报器能在现实场景下启发开发更强大的防御系统来抵御高级多回合越狱攻击。代码可在https://github.com/YiyiyiZhao/siren.上找到警告：本文包含可能有害的文本。



## **43. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

GreedyPixel：通过贪婪算法进行细粒度黑匣子对抗攻击 cs.CV

**SubmitDate**: 2025-01-24    [abs](http://arxiv.org/abs/2501.14230v1) [paper-pdf](http://arxiv.org/pdf/2501.14230v1)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: A critical requirement for deep learning models is ensuring their robustness against adversarial attacks. These attacks commonly introduce noticeable perturbations, compromising the visual fidelity of adversarial examples. Another key challenge is that while white-box algorithms can generate effective adversarial perturbations, they require access to the model gradients, limiting their practicality in many real-world scenarios. Existing attack mechanisms struggle to achieve similar efficacy without access to these gradients. In this paper, we introduce GreedyPixel, a novel pixel-wise greedy algorithm designed to generate high-quality adversarial examples using only query-based feedback from the target model. GreedyPixel improves computational efficiency in what is typically a brute-force process by perturbing individual pixels in sequence, guided by a pixel-wise priority map. This priority map is constructed by ranking gradients obtained from a surrogate model, providing a structured path for perturbation. Our results demonstrate that GreedyPixel achieves attack success rates comparable to white-box methods without the need for gradient information, and surpasses existing algorithms in black-box settings, offering higher success rates, reduced computational time, and imperceptible perturbations. These findings underscore the advantages of GreedyPixel in terms of attack efficacy, time efficiency, and visual quality.

摘要: 深度学习模型的一个关键要求是确保其对对手攻击的健壮性。这些攻击通常会引入明显的干扰，损害对抗性例子的视觉保真度。另一个关键挑战是，尽管白盒算法可以产生有效的对抗性扰动，但它们需要访问模型梯度，限制了它们在许多现实世界场景中的实用性。现有的攻击机制在没有这些梯度的情况下很难达到类似的效果。在本文中，我们介绍了GreedyPixel，这是一种新的像素级贪婪算法，旨在仅使用来自目标模型的基于查询的反馈来生成高质量的对抗性示例。GreedyPixel通过在逐个像素的优先级贴图的指导下按顺序扰乱各个像素，提高了计算效率，这通常是一个蛮力过程。该优先级图是通过对从代理模型获得的梯度进行排序来构建的，提供了用于扰动的结构化路径。结果表明，GreedyPixel在不需要梯度信息的情况下取得了与白盒方法相当的攻击成功率，并在黑盒环境下超过了现有算法，提供了更高的成功率、更少的计算时间和不可察觉的扰动。这些发现强调了GreedyPixel在攻击效率、时间效率和视觉质量方面的优势。



## **44. Reinforcement Learning Platform for Adversarial Black-box Attacks with Custom Distortion Filters**

具有自定义失真过滤器的对抗黑匣子攻击强化学习平台 cs.LG

Under Review for 2025 AAAI Conference on Artificial Intelligence  Proceedings

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.14122v1) [paper-pdf](http://arxiv.org/pdf/2501.14122v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Sahand Ghorbanpour, Avisek Naug, Ricardo Luna Gutierrez, Antonio Guillen

**Abstract**: We present a Reinforcement Learning Platform for Adversarial Black-box untargeted and targeted attacks, RLAB, that allows users to select from various distortion filters to create adversarial examples. The platform uses a Reinforcement Learning agent to add minimum distortion to input images while still causing misclassification by the target model. The agent uses a novel dual-action method to explore the input image at each step to identify sensitive regions for adding distortions while removing noises that have less impact on the target model. This dual action leads to faster and more efficient convergence of the attack. The platform can also be used to measure the robustness of image classification models against specific distortion types. Also, retraining the model with adversarial samples significantly improved robustness when evaluated on benchmark datasets. The proposed platform outperforms state-of-the-art methods in terms of the average number of queries required to cause misclassification. This advances trustworthiness with a positive social impact.

摘要: 我们提出了一个针对对抗性黑盒非目标攻击和目标攻击的强化学习平台RLAB，它允许用户从各种失真过滤器中进行选择以创建对抗性示例。该平台使用强化学习代理向输入图像添加最小失真，同时仍会导致目标模型的错误分类。该代理在每一步都使用一种新颖的双作用方法来探索输入图像，以识别添加失真的敏感区域，同时去除对目标模型影响较小的噪声。这种双重行动导致攻击更快、更有效地收敛。该平台还可用于衡量图像分类模型对特定失真类型的稳健性。此外，当在基准数据集上进行评估时，用对抗性样本重新训练模型显著提高了稳健性。在导致错误分类所需的平均查询数量方面，该平台的性能优于最先进的方法。这提高了可信度，产生了积极的社会影响。



## **45. Noisy Data Meets Privacy: Training Local Models with Post-Processed Remote Queries**

噪音数据与隐私相遇：使用后处理的远程数据库训练本地模型 cs.LG

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2405.16361v2) [paper-pdf](http://arxiv.org/pdf/2405.16361v2)

**Authors**: Kexin Li, Aastha Mehta, David Lie

**Abstract**: The adoption of large cloud-based models for inference in privacy-sensitive domains, such as homeless care systems and medical imaging, raises concerns about end-user data privacy. A common solution is adding locally differentially private (LDP) noise to queries before transmission, but this often reduces utility. LDPKiT, which stands for Local Differentially-Private and Utility-Preserving Inference via Knowledge Transfer, addresses the concern by generating a privacy-preserving inference dataset aligned with the private data distribution. This dataset is used to train a reliable local model for inference on sensitive inputs. LDPKiT employs a two-layer noise injection framework that leverages LDP and its post-processing property to create a privacy-protected inference dataset. The first layer ensures privacy, while the second layer helps to recover utility by creating a sufficiently large dataset for subsequent local model extraction using noisy labels returned from a cloud model on privacy-protected noisy inputs. Our experiments on Fashion-MNIST, SVHN and PathMNIST medical datasets demonstrate that LDPKiT effectively improves utility while preserving privacy. Moreover, the benefits of using LDPKiT increase at higher, more privacy-protective noise levels. For instance, on SVHN, LDPKiT achieves similar inference accuracy with $\epsilon=1.25$ as it does with $\epsilon=2.0$, providing stronger privacy guarantees with less than a 2% drop in accuracy. Furthermore, we perform extensive sensitivity analyses to evaluate the impact of dataset sizes on LDPKiT's effectiveness and systematically analyze the latent space representations to offer a theoretical explanation for its accuracy improvements. Lastly, we qualitatively and quantitatively demonstrate that the type of knowledge distillation performed by LDPKiT is ethical and fundamentally distinct from adversarial model extraction attacks.

摘要: 在无家可归者护理系统和医学成像等隐私敏感领域采用基于云的大型模型进行推断，引发了对最终用户数据隐私的担忧。一种常见的解决方案是在传输之前向查询添加局部差分私有(LDP)噪声，但这通常会降低效用。LDPKiT代表通过知识转移的局部差异私有和效用保护推理，通过生成与私有数据分布一致的隐私保护推理数据集来解决这一问题。该数据集用于训练可靠的本地模型，以便对敏感输入进行推断。LDPKiT采用两层噪声注入框架，该框架利用LDP及其后处理属性来创建隐私保护的推理数据集。第一层确保隐私，而第二层通过在受隐私保护的噪声输入上使用从云模型返回的噪声标签来创建足够大的数据集以用于后续的本地模型提取，从而帮助恢复效用。我们在Fashion-MNIST、SVHN和PathMNIST医学数据集上的实验表明，LDPKiT在保护隐私的同时有效地提高了实用性。此外，使用LDPKiT的好处在更高、更隐私保护的噪音水平下会增加。例如，在SVHN上，LDPKiT在$\epsilon=1.25$时实现了与$\epsilon=2.0$类似的推理精度，提供了更强的隐私保证，精确度下降不到2%。此外，我们还进行了大量的敏感性分析，以评估数据集大小对LDPKiT有效性的影响，并系统地分析了潜在空间表示，为其精度的提高提供了理论解释。最后，我们定性和定量地证明了LDPKiT进行的知识提炼类型是伦理的，并且与对抗性模型提取攻击有根本区别。



## **46. Defending against Adversarial Malware Attacks on ML-based Android Malware Detection Systems**

防御基于ML的Android恶意软件检测系统上的敌对恶意软件攻击 cs.CR

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13782v1) [paper-pdf](http://arxiv.org/pdf/2501.13782v1)

**Authors**: Ping He, Lorenzo Cavallaro, Shouling Ji

**Abstract**: Android malware presents a persistent threat to users' privacy and data integrity. To combat this, researchers have proposed machine learning-based (ML-based) Android malware detection (AMD) systems. However, adversarial Android malware attacks compromise the detection integrity of the ML-based AMD systems, raising significant concerns. Existing defenses against adversarial Android malware provide protections against feature space attacks which generate adversarial feature vectors only, leaving protection against realistic threats from problem space attacks which generate real adversarial malware an open problem. In this paper, we address this gap by proposing ADD, a practical adversarial Android malware defense framework designed as a plug-in to enhance the adversarial robustness of the ML-based AMD systems against problem space attacks. Our extensive evaluation across various ML-based AMD systems demonstrates that ADD is effective against state-of-the-art problem space adversarial Android malware attacks. Additionally, ADD shows the defense effectiveness in enhancing the adversarial robustness of real-world antivirus solutions.

摘要: Android恶意软件对用户隐私和数据完整性构成持续威胁。为了应对这一问题，研究人员提出了基于机器学习(ML-Based)的Android恶意软件检测(AMD)系统。然而，恶意的Android恶意软件攻击损害了基于ML的AMD系统的检测完整性，引发了严重的担忧。现有的针对对抗性Android恶意软件的防御提供了针对仅生成对抗性特征向量的特征空间攻击的保护，使得针对生成真正对抗性恶意软件的问题空间攻击的现实威胁的保护成为一个悬而未决的问题。在本文中，我们通过提出一个实用的恶意软件防御框架ADD来解决这一问题，该框架以插件的形式设计，以增强基于ML的AMD系统对抗问题空间攻击的健壮性。我们对各种基于ML的AMD系统的广泛评估表明，ADD对最先进的问题空间对手Android恶意软件攻击是有效的。此外，ADD显示了在增强现实世界反病毒解决方案的对抗健壮性方面的防御有效性。



## **47. Crossfire: An Elastic Defense Framework for Graph Neural Networks Under Bit Flip Attacks**

Crossfire：位翻转攻击下图神经网络的弹性防御框架 cs.LG

Accepted at AAAI 2025, DOI will be included after publication

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13776v1) [paper-pdf](http://arxiv.org/pdf/2501.13776v1)

**Authors**: Lorenz Kummer, Samir Moustafa, Wilfried Gansterer, Nils Kriege

**Abstract**: Bit Flip Attacks (BFAs) are a well-established class of adversarial attacks, originally developed for Convolutional Neural Networks within the computer vision domain. Most recently, these attacks have been extended to target Graph Neural Networks (GNNs), revealing significant vulnerabilities. This new development naturally raises questions about the best strategies to defend GNNs against BFAs, a challenge for which no solutions currently exist. Given the applications of GNNs in critical fields, any defense mechanism must not only maintain network performance, but also verifiably restore the network to its pre-attack state. Verifiably restoring the network to its pre-attack state also eliminates the need for costly evaluations on test data to ensure network quality. We offer first insights into the effectiveness of existing honeypot- and hashing-based defenses against BFAs adapted from the computer vision domain to GNNs, and characterize the shortcomings of these approaches. To overcome their limitations, we propose Crossfire, a hybrid approach that exploits weight sparsity and combines hashing and honeypots with bit-level correction of out-of-distribution weight elements to restore network integrity. Crossfire is retraining-free and does not require labeled data. Averaged over 2,160 experiments on six benchmark datasets, Crossfire offers a 21.8% higher probability than its competitors of reconstructing a GNN attacked by a BFA to its pre-attack state. These experiments cover up to 55 bit flips from various attacks. Moreover, it improves post-repair prediction quality by 10.85%. Computational and storage overheads are negligible compared to the inherent complexity of even the simplest GNNs.

摘要: 比特翻转攻击(BFA)是一类公认的对抗性攻击，最初是为计算机视觉领域内的卷积神经网络开发的。最近，这些攻击已扩展到目标图神经网络(GNN)，暴露出严重的漏洞。这一新的发展自然提出了关于保护GNN免受BFA攻击的最佳战略的问题，这是一个目前还没有解决方案的挑战。考虑到GNN在关键领域的应用，任何防御机制都必须不仅保持网络的性能，而且要可验证地将网络恢复到攻击前的状态。以可验证的方式将网络恢复到攻击前的状态还消除了对测试数据进行昂贵评估以确保网络质量的需要。我们首次对现有的基于蜜罐和散列的针对从计算机视觉领域改编到GNN的BFA的防御的有效性进行了洞察，并表征了这些方法的缺点。为了克服它们的局限性，我们提出了Crossfire，这是一种混合方法，它利用权重稀疏性，将哈希和蜜罐与对不在分布的权重元素进行比特级校正相结合，以恢复网络完整性。CrossFire是免再培训的，不需要标签数据。Crossfire在六个基准数据集上平均进行了2160次实验，将受到BFA攻击的GNN重建到攻击前状态的可能性比其竞争对手高21.8%。这些实验涵盖了来自各种攻击的多达55个比特翻转。此外，修复后预测质量提高了10.85%。与即使是最简单的GNN的固有复杂性相比，计算和存储开销也微不足道。



## **48. PhishAgent: A Robust Multimodal Agent for Phishing Webpage Detection**

PhishAgent：一种用于网络钓鱼网页检测的鲁棒多模式代理 cs.CR

Accepted at AAAI 2025 (Oral)

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2408.10738v3) [paper-pdf](http://arxiv.org/pdf/2408.10738v3)

**Authors**: Tri Cao, Chengyu Huang, Yuexin Li, Huilin Wang, Amy He, Nay Oo, Bryan Hooi

**Abstract**: Phishing attacks are a major threat to online security, exploiting user vulnerabilities to steal sensitive information. Various methods have been developed to counteract phishing, each with varying levels of accuracy, but they also face notable limitations. In this study, we introduce PhishAgent, a multimodal agent that combines a wide range of tools, integrating both online and offline knowledge bases with Multimodal Large Language Models (MLLMs). This combination leads to broader brand coverage, which enhances brand recognition and recall. Furthermore, we propose a multimodal information retrieval framework designed to extract the relevant top k items from offline knowledge bases, using available information from a webpage, including logos and HTML. Our empirical results, based on three real-world datasets, demonstrate that the proposed framework significantly enhances detection accuracy and reduces both false positives and false negatives, while maintaining model efficiency. Additionally, PhishAgent shows strong resilience against various types of adversarial attacks.

摘要: 网络钓鱼攻击是在线安全的主要威胁，利用用户漏洞窃取敏感信息。已经开发了各种方法来对抗网络钓鱼，每一种方法的精确度都不同，但它们也面临着显著的局限性。在本研究中，我们介绍了PhishAgent，一个结合了广泛工具的多通道代理，将线上和线下知识库与多通道大语言模型(MLLMS)相结合。这一组合导致了更广泛的品牌覆盖，从而提高了品牌认知度和召回率。此外，我们还提出了一个多通道信息检索框架，该框架利用网页中的可用信息，包括标识和超文本标记语言，从离线知识库中提取相关的前k个条目。基于三个真实数据集的实验结果表明，该框架在保持模型效率的同时，显著提高了检测准确率，减少了误报和漏报。此外，PhishAgent对各种类型的对抗性攻击表现出很强的韧性。



## **49. Device-aware Optical Adversarial Attack for a Portable Projector-camera System**

便携式投影相机系统的设备感知光学对抗攻击 cs.CV

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.14005v1) [paper-pdf](http://arxiv.org/pdf/2501.14005v1)

**Authors**: Ning Jiang, Yanhong Liu, Dingheng Zeng, Yue Feng, Weihong Deng, Ying Li

**Abstract**: Deep-learning-based face recognition (FR) systems are susceptible to adversarial examples in both digital and physical domains. Physical attacks present a greater threat to deployed systems as adversaries can easily access the input channel, allowing them to provide malicious inputs to impersonate a victim. This paper addresses the limitations of existing projector-camera-based adversarial light attacks in practical FR setups. By incorporating device-aware adaptations into the digital attack algorithm, such as resolution-aware and color-aware adjustments, we mitigate the degradation from digital to physical domains. Experimental validation showcases the efficacy of our proposed algorithm against real and spoof adversaries, achieving high physical similarity scores in FR models and state-of-the-art commercial systems. On average, there is only a 14% reduction in scores from digital to physical attacks, with high attack success rate in both white- and black-box scenarios.

摘要: 基于深度学习的人脸识别（FR）系统在数字和物理领域都容易受到对抗示例的影响。物理攻击对已部署的系统构成了更大的威胁，因为对手可以轻松访问输入通道，使他们能够提供恶意输入来冒充受害者。本文解决了实际FR设置中现有的基于投影仪摄像机的对抗性光攻击的局限性。通过将设备感知的适应融入数字攻击算法中，例如分辨率感知和颜色感知调整，我们可以减轻从数字域到物理域的降级。实验验证展示了我们提出的算法针对真实和欺骗对手的有效性，在FR模型和最先进的商业系统中实现了很高的物理相似性分数。平均而言，从数字攻击到物理攻击的分数仅降低14%，并且在白盒和黑匣子场景下的攻击成功率都很高。



## **50. Black-Box Adversarial Attack on Vision Language Models for Autonomous Driving**

自动驾驶视觉语言模型的黑匣子对抗攻击 cs.CV

**SubmitDate**: 2025-01-23    [abs](http://arxiv.org/abs/2501.13563v1) [paper-pdf](http://arxiv.org/pdf/2501.13563v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yang Qu, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu, Dacheng Tao

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities; however, these models remain highly susceptible to adversarial attacks. While existing research has explored white-box attacks to some extent, the more practical and challenging black-box scenarios remain largely underexplored due to their inherent difficulty. In this paper, we take the first step toward designing black-box adversarial attacks specifically targeting VLMs in AD. We identify two key challenges for achieving effective black-box attacks in this context: the effectiveness across driving reasoning chains in AD systems and the dynamic nature of driving scenarios. To address this, we propose Cascading Adversarial Disruption (CAD). It first introduces Decision Chain Disruption, which targets low-level reasoning breakdown by generating and injecting deceptive semantics, ensuring the perturbations remain effective across the entire decision-making chain. Building on this, we present Risky Scene Induction, which addresses dynamic adaptation by leveraging a surrogate VLM to understand and construct high-level risky scenarios that are likely to result in critical errors in the current driving contexts. Extensive experiments conducted on multiple AD VLMs and benchmarks demonstrate that CAD achieves state-of-the-art attack effectiveness, significantly outperforming existing methods (+13.43% on average). Moreover, we validate its practical applicability through real-world attacks on AD vehicles powered by VLMs, where the route completion rate drops by 61.11% and the vehicle crashes directly into the obstacle vehicle with adversarial patches. Finally, we release CADA dataset, comprising 18,808 adversarial visual-question-answer pairs, to facilitate further evaluation and research in this critical domain. Our codes and dataset will be available after paper's acceptance.

摘要: 视觉语言模型通过增强推理能力极大地促进了自动驾驶(AD)的发展，但这些模型仍然高度容易受到对手的攻击。虽然现有的研究在一定程度上探索了白盒攻击，但由于其固有的困难，更实用和更具挑战性的黑盒场景在很大程度上仍然没有得到充分的探索。在本文中，我们向设计专门针对AD中的VLM的黑盒对抗性攻击迈出了第一步。在此背景下，我们确定了实现有效的黑盒攻击的两个关键挑战：在AD系统中跨驾驶推理链的有效性和驾驶场景的动态性质。为了解决这个问题，我们提出了级联对抗中断(CAD)。它首先引入决策链中断，通过生成和注入欺骗性语义来针对低级别推理故障，确保扰动在整个决策链中保持有效。在此基础上，我们提出了风险场景归纳，它通过利用代理VLM来理解和构建可能在当前驾驶环境中导致关键错误的高级别风险场景，从而解决动态适应问题。在多个AD VLM和基准上进行的广泛实验表明，CAD实现了最先进的攻击效率，显著优于现有方法(平均+13.43%)。此外，通过对VLMS驱动的AD车辆的实际攻击，路径完成率下降了61.11%，车辆直接撞上了带有对抗性补丁的障碍车辆，验证了该算法的实用性。最后，我们发布了包含18,808个对抗性视觉-问答对的CADA数据集，以便于在这一关键领域进行进一步的评估和研究。我们的代码和数据集将在论文验收后可用。



