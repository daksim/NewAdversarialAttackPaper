# Latest Adversarial Attack Papers
**update at 2022-10-28 06:31:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

偏距离相关在深度学习中的广泛应用 cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2207.09684v2) [paper-pdf](http://arxiv.org/pdf/2207.09684v2)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **2. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

通过威胁建模识别智能城市基础设施中的威胁、网络犯罪和数字取证机会 cs.CR

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14692v1) [paper-pdf](http://arxiv.org/pdf/2210.14692v1)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.

摘要: 技术进步使多个国家能够考虑实施智慧城市基础设施，以深入了解不同的数据点，并改善公民的生活。不幸的是，这些新的技术实施也引诱对手和网络罪犯对这些现代基础设施进行网络攻击和犯罪行为。鉴于网络攻击的无边界性质、对智能城市基础设施的不同程度的了解以及正在进行的调查工作量，执法机构和调查人员将很难对此类网络犯罪做出回应。如果调查人员没有调查能力，这些智能基础设施可能会成为网络犯罪分子青睐的新目标。为了应对调查人员面临的挑战，我们提出了智能城市基础设施的共同定义。在定义的基础上，我们利用STRIDE威胁建模方法和Microsoft威胁建模工具来识别基础设施中存在的威胁，并创建可由感兴趣的各方进一步定制或扩展的威胁模型。接下来，我们将绘制罪行、可能的证据来源和已确定的威胁类型的地图，以帮助调查人员了解哪些罪行可能发生，以及在调查工作中需要哪些证据。最后，注意到智能城市基础设施调查将是一项全球多方面的挑战，我们讨论了智能城市基础设施数字取证的技术和法律机会。



## **3. Certified Robustness in Federated Learning**

联合学习中的认证稳健性 cs.LG

Accepted at Workshop on Federated Learning: Recent Advances and New  Challenges, NeurIPS 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2206.02535v2) [paper-pdf](http://arxiv.org/pdf/2206.02535v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstract**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(i.e. personalized) models, and find that the robustness of local models degrades as they diverge from the global model

摘要: 由于联邦学习在训练分布式数据上的机器学习模型方面的有效性，它最近获得了极大的关注和普及。然而，与单节点监督学习设置中一样，在联合学习中训练的模型容易受到称为对抗性攻击的不可察觉的输入转换的影响，从而质疑其在安全相关应用中的部署。在这项工作中，我们研究了联合训练、个性化和经过认证的健壮性之间的相互作用。特别是，我们采用了随机化平滑，这是一种广泛使用和可扩展的认证方法，用于认证在联合设置上训练的深层网络不受输入扰动和转换的影响。我们发现，与仅基于本地数据进行训练相比，简单的联合平均技术不仅在建立更准确的模型方面是有效的，而且在可证明的健壮性方面也更有效。我们进一步分析了个性化，这是联合训练中的一种流行技术，它增加了模型对本地数据的偏差，并对稳健性进行了分析。我们展示了个性化比这两者(即只在本地数据上训练和联合训练)在建立更健壮的模型和更快的训练方面的几个优势。最后，我们研究了全局模型和局部模型(即个性化模型)的混合模型的稳健性，发现局部模型的稳健性随着偏离全局模型而降低



## **4. Short Paper: Static and Microarchitectural ML-Based Approaches For Detecting Spectre Vulnerabilities and Attacks**

短文：基于静态和微体系结构ML的检测Spectre漏洞和攻击的方法 cs.CR

5 pages, 2 figures. Accepted to the Hardware and Architectural  Support for Security and Privacy (HASP'22), in conjunction with the 55th  IEEE/ACM International Symposium on Microarchitecture (MICRO'22)

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14452v1) [paper-pdf](http://arxiv.org/pdf/2210.14452v1)

**Authors**: Chidera Biringa, Gaspard Baye, Gökhan Kul

**Abstract**: Spectre intrusions exploit speculative execution design vulnerabilities in modern processors. The attacks violate the principles of isolation in programs to gain unauthorized private user information. Current state-of-the-art detection techniques utilize micro-architectural features or vulnerable speculative code to detect these threats. However, these techniques are insufficient as Spectre attacks have proven to be more stealthy with recently discovered variants that bypass current mitigation mechanisms. Side-channels generate distinct patterns in processor cache, and sensitive information leakage is dependent on source code vulnerable to Spectre attacks, where an adversary uses these vulnerabilities, such as branch prediction, which causes a data breach. Previous studies predominantly approach the detection of Spectre attacks using the microarchitectural analysis, a reactive approach. Hence, in this paper, we present the first comprehensive evaluation of static and microarchitectural analysis-assisted machine learning approaches to detect Spectre vulnerable code snippets (preventive) and Spectre attacks (reactive). We evaluate the performance trade-offs in employing classifiers for detecting Spectre vulnerabilities and attacks.

摘要: 幽灵入侵利用现代处理器中的推测性执行设计漏洞。这些攻击违反了程序中的隔离原则，以获取未经授权的私人用户信息。当前最先进的检测技术利用微体系结构特征或易受攻击的推测代码来检测这些威胁。然而，这些技术是不够的，因为Spectre攻击已被证明是更隐蔽的，最近发现的变体绕过了当前的缓解机制。侧通道在处理器缓存中生成不同的模式，敏感信息泄漏依赖于易受Spectre攻击的源代码，其中对手使用这些漏洞，如分支预测，从而导致数据泄露。以前的研究主要是使用微体系结构分析来检测Spectre攻击，这是一种反应性方法。因此，在本文中，我们首次对静态和微体系结构分析辅助的机器学习方法进行了全面评估，以检测Spectre易受攻击的代码片段(预防性的)和Spectre攻击(反应性的)。我们评估了使用分类器来检测Spectre漏洞和攻击时的性能权衡。



## **5. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

21 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14410v1) [paper-pdf](http://arxiv.org/pdf/2210.14410v1)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



## **6. Adaptive Test-Time Defense with the Manifold Hypothesis**

流形假设下的自适应测试时间防御 cs.LG

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14404v1) [paper-pdf](http://arxiv.org/pdf/2210.14404v1)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with our formulation and variational inference. The developed approach combines manifold learning with the Bayesian framework to provide adversarial robustness without the need for adversarial training. We show that our proposed approach can provide adversarial robustness even if attackers are aware of existence of test-time defense. In additions, our approach can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。我们的框架为防御对抗性例子提供了充分的条件。利用我们的公式和变分推理，我们开发了一种测试时间防御方法。该方法将流形学习与贝叶斯框架相结合，在不需要对抗性训练的情况下提供对抗性健壮性。我们证明，即使攻击者知道测试时间防御的存在，我们所提出的方法也可以提供对抗健壮性。此外，我们的方法还可以作为可变自动编码器的测试时间防御机制。



## **7. Robustness of Locally Differentially Private Graph Analysis Against Poisoning**

局部差分私有图分析对中毒的稳健性 cs.CR

22 pages, 6 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14376v1) [paper-pdf](http://arxiv.org/pdf/2210.14376v1)

**Authors**: Jacob Imola, Amrita Roy Chowdhury, Kamalika Chaudhuri

**Abstract**: Locally differentially private (LDP) graph analysis allows private analysis on a graph that is distributed across multiple users. However, such computations are vulnerable to data poisoning attacks where an adversary can skew the results by submitting malformed data. In this paper, we formally study the impact of poisoning attacks for graph degree estimation protocols under LDP. We make two key technical contributions. First, we observe LDP makes a protocol more vulnerable to poisoning -- the impact of poisoning is worse when the adversary can directly poison their (noisy) responses, rather than their input data. Second, we observe that graph data is naturally redundant -- every edge is shared between two users. Leveraging this data redundancy, we design robust degree estimation protocols under LDP that can significantly reduce the impact of data poisoning and compute degree estimates with high accuracy. We evaluate our proposed robust degree estimation protocols under poisoning attacks on real-world datasets to demonstrate their efficacy in practice.

摘要: 局部差分私有(LDP)图分析允许对分布在多个用户之间的图进行私有分析。然而，这种计算很容易受到数据中毒攻击，对手可以通过提交格式错误的数据来扭曲结果。本文形式化地研究了毒化攻击对LDP下图度估计协议的影响。我们做出了两项关键的技术贡献。首先，我们观察到LDP使协议更容易中毒--当对手可以直接中毒他们的(噪声)响应，而不是他们的输入数据时，中毒的影响更严重。其次，我们观察到图形数据自然是冗余的--每条边都在两个用户之间共享。利用这种数据冗余性，我们在LDP下设计了稳健的度估计协议，可以显著降低数据中毒的影响，并能高精度地计算度估计。我们在真实数据集上的中毒攻击下对我们提出的稳健程度估计协议进行了评估，以证明其在实践中的有效性。



## **8. Accelerating Certified Robustness Training via Knowledge Transfer**

通过知识转移加快认证健壮性培训 cs.LG

NeurIPS '22 Camera Ready version (with appendix)

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14283v1) [paper-pdf](http://arxiv.org/pdf/2210.14283v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Training deep neural network classifiers that are certifiably robust against adversarial attacks is critical to ensuring the security and reliability of AI-controlled systems. Although numerous state-of-the-art certified training methods have been developed, they are computationally expensive and scale poorly with respect to both dataset and network complexity. Widespread usage of certified training is further hindered by the fact that periodic retraining is necessary to incorporate new data and network improvements. In this paper, we propose Certified Robustness Transfer (CRT), a general-purpose framework for reducing the computational overhead of any certifiably robust training method through knowledge transfer. Given a robust teacher, our framework uses a novel training loss to transfer the teacher's robustness to the student. We provide theoretical and empirical validation of CRT. Our experiments on CIFAR-10 show that CRT speeds up certified robustness training by $8 \times$ on average across three different architecture generations while achieving comparable robustness to state-of-the-art methods. We also show that CRT can scale to large-scale datasets like ImageNet.

摘要: 训练深度神经网络分类器对于确保人工智能控制系统的安全性和可靠性至关重要。尽管已经开发了许多最先进的认证训练方法，但它们的计算成本很高，并且在数据集和网络复杂性方面可伸缩性较差。定期再培训对于纳入新的数据和网络改进是必要的，这进一步阻碍了认证培训的广泛使用。在本文中，我们提出了认证健壮性转移(CRT)，这是一个通用的框架，通过知识转移来减少任何可证明健壮性训练方法的计算开销。假设有一位健壮的教师，我们的框架使用了一种新的训练损失来将教师的健壮性传递给学生。我们提供了CRT的理论和经验验证。我们在CIFAR-10上的实验表明，CRT在三代不同的体系结构上平均将经过认证的健壮性训练速度提高了8倍，同时获得了与最先进方法相当的健壮性。我们还展示了CRT可以扩展到像ImageNet这样的大规模数据集。



## **9. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

自然语言单位之间的相似性：从粗略到精细的过渡 cs.CL

PhD thesis

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14275v1) [paper-pdf](http://arxiv.org/pdf/2210.14275v1)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.

摘要: 捕捉人类语言单位之间的相似性对于解释人类如何关联不同的对象至关重要，因此其计算得到了广泛的关注、研究和应用。随着我们周围信息量的不断增加，计算相似度变得越来越复杂，特别是在许多情况下，如法律或医疗事务，计算相似度需要格外小心和精确，因为一个语言单位内的小行为可能会产生重大的现实世界影响。我在这篇论文中的研究目标是建立回归模型，以更精细的方式解释语言单位之间的相似性。相似性的计算已经走了很长一段路，但调试这些测量的方法通常是基于不断拟合人类判断值。为此，我的目标是开发一种算法，准确地捕捉相似性计算中的漏洞。此外，大多数方法对它们计算的相似性有模糊的定义，而且往往很难解释。拟议的框架解决了这两个缺点。它通过捕捉不同的漏洞来不断改进模型。此外，模型的每一次细化都提供了合理的解释。本文所介绍的回归模型称为递进精化相似度计算，它将攻击测试和对抗性训练相结合。本文的相似度回归模型在处理边缘情况方面达到了最好的性能。



## **10. Leveraging the Verifier's Dilemma to Double Spend in Bitcoin**

利用验证者的两难境地加倍投入比特币 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14072v1) [paper-pdf](http://arxiv.org/pdf/2210.14072v1)

**Authors**: Tong Cao, Jérémie Decouchant, Jiangshan Yu

**Abstract**: We describe and analyze perishing mining, a novel block-withholding mining strategy that lures profit-driven miners away from doing useful work on the public chain by releasing block headers from a privately maintained chain. We then introduce the dual private chain (DPC) attack, where an adversary that aims at double spending increases its success rate by intermittently dedicating part of its hash power to perishing mining. We detail the DPC attack's Markov decision process, evaluate its double spending success rate using Monte Carlo simulations. We show that the DPC attack lowers Bitcoin's security bound in the presence of profit-driven miners that do not wait to validate the transactions of a block before mining on it.

摘要: 我们描述和分析了正在灭亡的挖掘，这是一种新的块扣留挖掘策略，通过从私人维护的链中释放块头来引诱受利润驱动的矿工远离在公共链上做有用的工作。然后，我们介绍了双重私有链(DPC)攻击，在这种攻击中，一个旨在加倍支出的对手通过断断续续地将其部分散列能力用于消灭挖掘来提高其成功率。详细描述了DPC攻击的马尔可夫决策过程，并利用蒙特卡罗模拟对其双开销成功率进行了评估。我们表明，在利润驱动的矿工在场的情况下，DPC攻击降低了比特币的安全界限，这些矿工在挖掘比特币之前不会等待验证区块的交易。



## **11. A White-Box Adversarial Attack Against a Digital Twin**

一种针对数字双胞胎的白盒对抗性攻击 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14018v1) [paper-pdf](http://arxiv.org/pdf/2210.14018v1)

**Authors**: Wilson Patterson, Ivan Fernandez, Subash Neupane, Milan Parmar, Sudip Mittal, Shahram Rahimi

**Abstract**: Recent research has shown that Machine Learning/Deep Learning (ML/DL) models are particularly vulnerable to adversarial perturbations, which are small changes made to the input data in order to fool a machine learning classifier. The Digital Twin, which is typically described as consisting of a physical entity, a virtual counterpart, and the data connections in between, is increasingly being investigated as a means of improving the performance of physical entities by leveraging computational techniques, which are enabled by the virtual counterpart. This paper explores the susceptibility of Digital Twin (DT), a virtual model designed to accurately reflect a physical object using ML/DL classifiers that operate as Cyber Physical Systems (CPS), to adversarial attacks. As a proof of concept, we first formulate a DT of a vehicular system using a deep neural network architecture and then utilize it to launch an adversarial attack. We attack the DT model by perturbing the input to the trained model and show how easily the model can be broken with white-box attacks.

摘要: 最近的研究表明，机器学习/深度学习(ML/DL)模型特别容易受到对抗性扰动的影响，这些扰动是为了愚弄机器学习分类器而对输入数据进行的微小更改。数字双胞胎通常被描述为由物理实体、虚拟对应物和它们之间的数据连接组成，越来越多的人将其作为一种通过利用由虚拟对等体实现的计算技术来改善物理实体的性能的手段来进行研究。数字孪生(DT)是一种虚拟模型，它使用作为网络物理系统(CPS)的ML/DL分类器来准确地反映物理对象，本文探讨了DT对对手攻击的敏感性。作为概念验证，我们首先使用深度神经网络体系结构来建立车载系统的DT，然后利用它来发起对抗性攻击。我们通过扰动训练模型的输入来攻击DT模型，并展示了该模型可以多么容易地被白盒攻击打破。



## **12. Causal Information Bottleneck Boosts Adversarial Robustness of Deep Neural Network**

因果信息瓶颈增强深度神经网络的对抗健壮性 cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14229v1) [paper-pdf](http://arxiv.org/pdf/2210.14229v1)

**Authors**: Huan Hua, Jun Yan, Xi Fang, Weiquan Huang, Huilin Yin, Wancheng Ge

**Abstract**: The information bottleneck (IB) method is a feasible defense solution against adversarial attacks in deep learning. However, this method suffers from the spurious correlation, which leads to the limitation of its further improvement of adversarial robustness. In this paper, we incorporate the causal inference into the IB framework to alleviate such a problem. Specifically, we divide the features obtained by the IB method into robust features (content information) and non-robust features (style information) via the instrumental variables to estimate the causal effects. With the utilization of such a framework, the influence of non-robust features could be mitigated to strengthen the adversarial robustness. We make an analysis of the effectiveness of our proposed method. The extensive experiments in MNIST, FashionMNIST, and CIFAR-10 show that our method exhibits the considerable robustness against multiple adversarial attacks. Our code would be released.

摘要: 信息瓶颈方法是深度学习中对抗攻击的一种可行的防御方案。然而，该方法存在伪相关问题，限制了其进一步提高对抗健壮性。在本文中，我们将因果推理引入到IB框架中来缓解这一问题。具体地说，我们通过工具变量将IB方法得到的特征划分为稳健特征(内容信息)和非稳健特征(风格信息)来估计因果效应。利用该框架可以减少非稳健特征的影响，增强对抗的稳健性。并对该方法的有效性进行了分析。在MNIST、FashionMNIST和CIFAR-10上的大量实验表明，我们的方法对多个对手攻击具有相当大的鲁棒性。我们的代码就会被发布。



## **13. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

卡尔法特：带有标签偏斜度的校准联合对抗性训练 cs.LG

Accepted to the Conference on the Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2205.14926v2) [paper-pdf](http://arxiv.org/pdf/2205.14926v2)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstract**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability on non-IID data with label skewness, resulting in degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and better convergence points.

摘要: 最近的研究表明，与传统的机器学习一样，联邦学习(FL)也容易受到对手攻击。为了提高FL的对抗健壮性，联合对抗训练(FAT)方法被提出在全局聚集之前局部应用对抗训练。虽然这些方法在独立同分布(IID)数据上显示了良好的结果，但它们在具有标签偏斜的非IID数据上存在训练不稳定性，导致自然精度降低。这往往会阻碍FAT在实际应用中的应用，在现实应用中，跨客户端的标签分布通常是不对称的。本文研究了标签倾斜下的FAT问题，揭示了训练不稳定和自然精度下降的一个根本原因：倾斜的标签会导致类别概率不同和局部模型的异构性。然后，我们提出了一种校准FAT(CALFAT)方法来解决不稳定性问题，方法是自适应地校准逻辑以平衡类别。我们从理论和经验两个方面证明了CALFAT算法的优化可以得到跨客户的同质局部模型和更好的收敛点。



## **14. FocusedCleaner: Sanitizing Poisoned Graphs for Robust GNN-based Node Classification**

FocusedCleaner：用于基于GNN的健壮节点分类的毒图清理 cs.LG

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13815v1) [paper-pdf](http://arxiv.org/pdf/2210.13815v1)

**Authors**: Yulin Zhu, Liang Tong, Kai Zhou

**Abstract**: Recently, a lot of research attention has been devoted to exploring Web security, a most representative topic is the adversarial robustness of graph mining algorithms. Especially, a widely deployed adversarial attacks formulation is the graph manipulation attacks by modifying the relational data to mislead the Graph Neural Networks' (GNNs) predictions. Naturally, an intrinsic question one would ask is whether we can accurately identify the manipulations over graphs - we term this problem as poisoned graph sanitation. In this paper, we present FocusedCleaner, a poisoned graph sanitation framework consisting of two modules: bi-level structural learning and victim node detection. In particular, the structural learning module will reserve the attack process to steadily sanitize the graph while the detection module provides the "focus" - a narrowed and more accurate search region - to structural learning. These two modules will operate in iterations and reinforce each other to sanitize a poisoned graph step by step. Extensive experiments demonstrate that FocusedCleaner outperforms the state-of-the-art baselines both on poisoned graph sanitation and improving robustness.

摘要: 近年来，很多研究都致力于Web安全的探索，图挖掘算法的对抗健壮性就是一个最具代表性的话题。特别是，一种被广泛应用的对抗性攻击方案是通过修改关系数据来误导图神经网络(GNN)预测的图操纵攻击。自然，人们会问一个内在的问题是，我们是否能准确地识别对图的操纵-我们将这个问题称为有毒的图卫生。在本文中，我们提出了FocusedCleaner，一个由两个模块组成的有毒图健康框架：双层结构学习和受害者节点检测。特别是，结构学习模块将保留攻击过程，以稳定地对图进行杀菌，而检测模块则为结构学习提供“焦点”--一个更窄、更准确的搜索区域。这两个模块将在迭代中运行，并相互加强，逐步清理有毒的图形。广泛的实验表明，FocusedCleaner在有毒的图形卫生和提高健壮性方面都优于最先进的基线。



## **15. Flexible Android Malware Detection Model based on Generative Adversarial Networks with Code Tensor**

基于码张量生成对抗网络的灵活Android恶意软件检测模型 cs.CR

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.14225v1) [paper-pdf](http://arxiv.org/pdf/2210.14225v1)

**Authors**: Zhao Yang, Fengyang Deng, Linxi Han

**Abstract**: The behavior of malware threats is gradually increasing, heightened the need for malware detection. However, existing malware detection methods only target at the existing malicious samples, the detection of fresh malicious code and variants of malicious code is limited. In this paper, we propose a novel scheme that detects malware and its variants efficiently. Based on the idea of the generative adversarial networks (GANs), we obtain the `true' sample distribution that satisfies the characteristics of the real malware, use them to deceive the discriminator, thus achieve the defense against malicious code attacks and improve malware detection. Firstly, a new Android malware APK to image texture feature extraction segmentation method is proposed, which is called segment self-growing texture segmentation algorithm. Secondly, tensor singular value decomposition (tSVD) based on the low-tubal rank transforms malicious features with different sizes into a fixed third-order tensor uniformly, which is entered into the neural network for training and learning. Finally, a flexible Android malware detection model based on GANs with code tensor (MTFD-GANs) is proposed. Experiments show that the proposed model can generally surpass the traditional malware detection model, with a maximum improvement efficiency of 41.6\%. At the same time, the newly generated samples of the GANs generator greatly enrich the sample diversity. And retraining malware detector can effectively improve the detection efficiency and robustness of traditional models.

摘要: 恶意软件威胁的行为正在逐渐增加，这加剧了对恶意软件检测的需求。然而，现有的恶意软件检测方法仅针对已有的恶意样本，对新的恶意代码和恶意代码变体的检测有限。本文提出了一种有效检测恶意软件及其变种的新方案。基于生成式对抗网络的思想，我们得到了满足真实恶意软件特征的“真”样本分布，并利用它们来欺骗鉴别器，从而实现了对恶意代码攻击的防御，提高了恶意软件的检测能力。首先，提出了一种新的Android恶意软件APK对图像纹理特征提取的分割方法--分段自增长纹理分割算法。其次，基于低管阶的张量奇异值分解(TSVD)将不同大小的恶意特征统一变换为固定的三阶张量，并输入神经网络进行训练和学习。最后，提出了一种基于编码张量遗传算法的Android恶意软件检测模型(MTFD-GANS)。实验表明，该模型总体上可以超过传统的恶意软件检测模型，最大改进效率为41.6%。同时，Gans生成器新生成的样本极大地丰富了样本的多样性。对恶意软件检测器进行再训练，可以有效提高传统模型的检测效率和鲁棒性。



## **16. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.08870v2) [paper-pdf](http://arxiv.org/pdf/2210.08870v2)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **17. Musings on the HashGraph Protocol: Its Security and Its Limitations**

对哈希图协议的思考：其安全性及其局限性 cs.CR

30 pages, 16 figures

**SubmitDate**: 2022-10-25    [abs](http://arxiv.org/abs/2210.13682v1) [paper-pdf](http://arxiv.org/pdf/2210.13682v1)

**Authors**: Vinesh Sridhar, Erica Blum, Jonathan Katz

**Abstract**: The HashGraph Protocol is a Byzantine fault tolerant atomic broadcast protocol. Its novel use of locally stored metadata allows parties to recover a consistent ordering of their log just by examining their local data, removing the need for a voting protocol. Our paper's first contribution is to present a rewritten proof of security for the HashGraph Protocol that follows the consistency and liveness paradigm used in the atomic broadcast literature. In our second contribution, we show a novel adversarial strategy that stalls the protocol from committing data to the log for an expected exponential number of rounds. This proves tight the exponential upper bound conjectured in the original paper. We believe that our proof of security will make it easier to compare HashGraph with other atomic broadcast protocols and to incorporate its ideas into new constructions. We also believe that our attack might inspire more research into similar attacks for other DAG-based atomic broadcast protocols.

摘要: 哈希图协议是一种拜占庭容错原子广播协议。它新颖地使用了本地存储的元数据，允许各方仅通过检查其本地数据就可以恢复其日志的一致顺序，从而消除了对投票协议的需要。我们的论文的第一个贡献是为HashGraph协议提供了一个重写的安全性证明，它遵循原子广播文献中使用的一致性和活跃性范例。在我们的第二个贡献中，我们展示了一种新的对抗性策略，该策略使协议无法将数据提交到日志中，达到预期的指数轮数。这证明了原论文中所猜想的指数上界是紧的。我们相信，我们的安全性证明将使我们更容易将HashGraph与其他原子广播协议进行比较，并将其思想融入新的构造中。我们还认为，我们的攻击可能会激发更多对其他基于DAG的原子广播协议的类似攻击的研究。



## **18. Analyzing Privacy Leakage in Machine Learning via Multiple Hypothesis Testing: A Lesson From Fano**

用多重假设检验分析机器学习中的隐私泄露--来自Fano的经验 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13662v1) [paper-pdf](http://arxiv.org/pdf/2210.13662v1)

**Authors**: Chuan Guo, Alexandre Sablayrolles, Maziar Sanjabi

**Abstract**: Differential privacy (DP) is by far the most widely accepted framework for mitigating privacy risks in machine learning. However, exactly how small the privacy parameter $\epsilon$ needs to be to protect against certain privacy risks in practice is still not well-understood. In this work, we study data reconstruction attacks for discrete data and analyze it under the framework of multiple hypothesis testing. We utilize different variants of the celebrated Fano's inequality to derive upper bounds on the inferential power of a data reconstruction adversary when the model is trained differentially privately. Importantly, we show that if the underlying private data takes values from a set of size $M$, then the target privacy parameter $\epsilon$ can be $O(\log M)$ before the adversary gains significant inferential power. Our analysis offers theoretical evidence for the empirical effectiveness of DP against data reconstruction attacks even at relatively large values of $\epsilon$.

摘要: 差异隐私(DP)是迄今为止被最广泛接受的减轻机器学习中隐私风险的框架。然而，在实践中，隐私参数$\epsilon$需要多小才能防止某些隐私风险仍然没有得到很好的理解。在本工作中，我们研究了离散数据的数据重构攻击，并在多重假设检验的框架下对其进行了分析。我们利用著名的Fano不等式的不同变体来推导数据重建对手在模型被私人差分训练时的推理能力的上界。重要的是，我们证明了如果底层私有数据取自一组大小为$M$的值，则在对手获得显著的推理能力之前，目标隐私参数$\epsilon$可以是$O(\log M)$。我们的分析为DP抵抗数据重建攻击的经验有效性提供了理论证据，即使在相对较大的$\epsilon$的情况下也是如此。



## **19. SpacePhish: The Evasion-space of Adversarial Attacks against Phishing Website Detectors using Machine Learning**

SpacePhish：利用机器学习对钓鱼网站检测器进行敌意攻击的规避空间 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13660v1) [paper-pdf](http://arxiv.org/pdf/2210.13660v1)

**Authors**: Giovanni Apruzzese, Mauro Conti, Ying Yuan

**Abstract**: Existing literature on adversarial Machine Learning (ML) focuses either on showing attacks that break every ML model, or defenses that withstand most attacks. Unfortunately, little consideration is given to the actual \textit{cost} of the attack or the defense. Moreover, adversarial samples are often crafted in the "feature-space", making the corresponding evaluations of questionable value. Simply put, the current situation does not allow to estimate the actual threat posed by adversarial attacks, leading to a lack of secure ML systems.   We aim to clarify such confusion in this paper. By considering the application of ML for Phishing Website Detection (PWD), we formalize the "evasion-space" in which an adversarial perturbation can be introduced to fool a ML-PWD -- demonstrating that even perturbations in the "feature-space" are useful. Then, we propose a realistic threat model describing evasion attacks against ML-PWD that are cheap to stage, and hence intrinsically more attractive for real phishers. Finally, we perform the first statistically validated assessment of state-of-the-art ML-PWD against 12 evasion attacks. Our evaluation shows (i) the true efficacy of evasion attempts that are more likely to occur; and (ii) the impact of perturbations crafted in different evasion-spaces. Our realistic evasion attempts induce a statistically significant degradation (3-10% at $p\!<$0.05), and their cheap cost makes them a subtle threat. Notably, however, some ML-PWD are immune to our most realistic attacks ($p$=0.22). Our contribution paves the way for a much needed re-assessment of adversarial attacks against ML systems for cybersecurity.

摘要: 现有的关于对抗性机器学习(ML)的文献要么专注于展示打破每个ML模型的攻击，要么专注于抵御大多数攻击的防御。不幸的是，很少考虑攻击或防御的实际成本。此外，对抗性样本往往是在“特征空间”中制作的，使得相应的评估价值值得怀疑。简单地说，目前的情况不允许估计对抗性攻击构成的实际威胁，导致缺乏安全的ML系统。我们的目的是在这篇论文中澄清这种混淆。通过考虑ML在钓鱼网站检测(PWD)中的应用，我们形式化了“规避空间”，在该空间中可以引入敌意扰动来愚弄ML-PWD--表明即使在“特征空间”中的扰动也是有用的。然后，我们提出了一个真实的威胁模型，描述了针对ML-PWD的逃避攻击，这些攻击的实施成本很低，因此本质上对真正的网络钓鱼者更具吸引力。最后，我们对最先进的ML-PWD进行了第一次统计验证评估，以对抗12次逃避攻击。我们的评估显示了(I)更有可能发生的逃避尝试的真实效果；以及(Ii)在不同的逃避空间中制造的扰动的影响。我们的现实规避尝试导致了统计上显著的下降(3%-10%，在$p<$0.05)，而且它们的廉价成本使它们成为一个微妙的威胁。然而，值得注意的是，一些ML-PWD对我们最现实的攻击($p$=0.22)是免疫的。我们的贡献为重新评估针对ML网络安全系统的对抗性攻击铺平了道路。



## **20. On the Robustness of Dataset Inference**

关于数据集推理的稳健性 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13631v1) [paper-pdf](http://arxiv.org/pdf/2210.13631v1)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs. Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI leads to FPs, with high confidence. Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade. Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.

摘要: 机器学习(ML)模型的训练成本很高，因为它们可能需要大量的数据、计算资源和技术专长。因此，它们构成了宝贵的知识产权，需要保护，不受想要窃取它们的对手的攻击。所有权验证技术允许模型盗窃攻击的受害者证明可疑模型实际上是从他们的模型中被盗的。虽然已经提出了一些基于水印或指纹的所有权验证技术，但它们大多在安全保证(装备良好的攻击者可以逃避验证)或计算代价方面存在不足。在ICLR‘21上引入的一种指纹技术，数据集推理(DI)，已经被证明比以前的方法提供了更好的稳健性和效率。DI的作者为线性(可疑)模型提供了正确性证明。然而，在相同的设置中，我们证明了DI存在高误报(FP)--它可能错误地识别使用来自相同分布的非重叠数据训练的独立模型作为被盗。我们进一步证明，在现实的、非线性的可疑模型中，依赖注入也会触发FP。然后，我们以很高的置信度从经验上证实了DI会导致FP。其次，我们证明了DI也存在假阴性(FN)--对手可以通过使用对抗性训练来调整被盗模型的决策边界来愚弄DI，从而导致FN。为此，我们演示了DI无法识别从窃取的数据集中恶意训练的模型--DI最难逃避的设置。最后，我们讨论了我们的发现的含义，基于指纹的所有权验证总体上的可行性，并对未来的工作提出了方向。



## **21. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

Deep VULMAN：一种深度强化学习的网络漏洞管理框架 cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2208.02369v2) [paper-pdf](http://arxiv.org/pdf/2208.02369v2)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstract**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.

摘要: 网络漏洞管理是网络安全运营中心(CSOC)的一项重要职能，有助于保护组织免受对其计算机和网络系统的网络攻击。与CSOC相比，对手拥有不对称的优势，因为与安全团队的扩张率相比，这些系统中的缺陷数量正在以显著更高的速度增加，以在资源受限的环境中缓解这些缺陷。目前的方法是确定性和一次性决策方法，在确定和选择要缓解的脆弱性时，不考虑未来的不确定性。这些办法还受到资源分配次优的限制，无法灵活地调整其对脆弱抵达人数波动的反应。我们提出了一种新的框架--Deep VULMAN，它由深度强化学习代理和整数规划方法组成，以填补网络漏洞管理过程中的这一空白。我们的顺序决策框架首先确定在给定系统状态下的不确定性情况下为缓解而分配的接近最优的资源量，然后确定用于缓解的最优优先级漏洞实例集。我们提出的框架在优先选择重要的特定于组织的漏洞方面优于目前的方法，该方法基于模拟和真实世界的漏洞数据，在一年的时间内观察到。



## **22. Probabilistic Categorical Adversarial Attack & Adversarial Training**

概率分类对抗性攻击与对抗性训练 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.09364v2) [paper-pdf](http://arxiv.org/pdf/2210.09364v2)

**Authors**: Pengfei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.

摘要: 对抗性实例的存在给深度神经网络在安全关键任务中的应用带来了极大的关注。然而，如何利用分类数据生成对抗性实例是一个重要的问题，但缺乏广泛的探索。以前建立的方法利用贪婪搜索方法，进行成功的攻击可能非常耗时。这也限制了对抗性训练的发展和对分类数据的潜在防御。为了解决这个问题，我们提出了概率分类对抗性攻击(PCAA)，它将离散的优化问题转化为一个连续的问题，可以用投影梯度下降法有效地解决。在本文中，我们从理论上分析了它的最优性和时间复杂性，以证明它相对于现有的基于贪婪的攻击具有显著的优势。此外，基于我们的攻击，我们提出了一个有效的对抗性训练框架。通过全面的实证研究，验证了本文提出的攻防算法的有效性。



## **23. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13263v1) [paper-pdf](http://arxiv.org/pdf/2210.13263v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **24. SealClub: Computer-aided Paper Document Authentication**

SealClub：计算机辅助纸质文档认证 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.07884v2) [paper-pdf](http://arxiv.org/pdf/2210.07884v2)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.

摘要: 数字身份验证是一个成熟的领域，提供了一系列具有严格数学保证的解决方案。然而，由于可用性和法律原因，在加密技术不直接适用的情况下，纸质文档仍然被广泛使用。我们提出了一种通过拍摄短视频来使用智能手机对纸质文档进行身份验证的新方法。我们的解决方案结合了加密和图像比较技术，以检测和突出对包含文本和图形的丰富文档的细微语义变化攻击，这些攻击可能不会被人类注意到。我们严格分析了我们的方法，证明了它是安全的，可以抵御能够危害不同系统组件的强大对手。我们还在一组128个纸质文档的视频上对其准确性进行了经验性的测量，其中一半包含微妙的伪造。该算法在平均分析5.13帧(对应于1.28秒的视频)后，准确地发现了所有的伪造(没有虚警)。突出显示的区域足够大，用户可以看到，但也足够小，可以精确定位假货。因此，我们的方法为用户在现实条件下使用传统的智能手机认证纸质文档提供了一种很有前途的方法。



## **25. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.08189v3) [paper-pdf](http://arxiv.org/pdf/2204.08189v3)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **26. Ares: A System-Oriented Wargame Framework for Adversarial ML**

ARES：一种面向系统的对抗性ML战争游戏框架 cs.LG

Presented at the DLS Workshop at S&P 2022

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.12952v1) [paper-pdf](http://arxiv.org/pdf/2210.12952v1)

**Authors**: Farhan Ahmed, Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Since the discovery of adversarial attacks against machine learning models nearly a decade ago, research on adversarial machine learning has rapidly evolved into an eternal war between defenders, who seek to increase the robustness of ML models against adversarial attacks, and adversaries, who seek to develop better attacks capable of weakening or defeating these defenses. This domain, however, has found little buy-in from ML practitioners, who are neither overtly concerned about these attacks affecting their systems in the real world nor are willing to trade off the accuracy of their models in pursuit of robustness against these attacks.   In this paper, we motivate the design and implementation of Ares, an evaluation framework for adversarial ML that allows researchers to explore attacks and defenses in a realistic wargame-like environment. Ares frames the conflict between the attacker and defender as two agents in a reinforcement learning environment with opposing objectives. This allows the introduction of system-level evaluation metrics such as time to failure and evaluation of complex strategies such as moving target defenses. We provide the results of our initial exploration involving a white-box attacker against an adversarially trained defender.

摘要: 自从近十年前发现了针对机器学习模型的对抗性攻击以来，对抗性机器学习的研究迅速演变为防御者和对手之间的一场永恒的战争。防御者试图增加ML模型对对抗性攻击的健壮性，而对手试图开发能够削弱或击败这些防御的更好的攻击。然而，这个领域几乎没有得到ML从业者的认可，他们既不公开担心这些攻击会影响他们在现实世界中的系统，也不愿意牺牲他们模型的准确性来追求对这些攻击的健壮性。在本文中，我们推动了ARES的设计和实现，这是一个针对对抗性ML的评估框架，允许研究人员在现实的类似战争游戏的环境中探索攻击和防御。阿瑞斯将攻击者和防御者之间的冲突框架为具有相反目标的强化学习环境中的两个代理。这允许引入系统级评估指标，如故障发生时间，以及评估复杂战略，如移动目标防御。我们提供了我们的初步探索的结果，涉及一个白盒攻击者对一个对手训练的后卫。



## **27. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

基于稀有嵌入和梯度集成的联合学习后门攻击 cs.LG

Accepted to EMNLP 2022, 9 pages and Appendix

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.14017v2) [paper-pdf](http://arxiv.org/pdf/2204.14017v2)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstract**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through rare word embeddings of NLP models. In text classification, less than 1% of adversary clients suffices to manipulate the model output without any drop in the performance on clean sentences. For a less complex dataset, a mere 0.1% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called Gradient Ensemble, which enhances the backdoor performance in all our experimental settings.

摘要: 联邦学习的最新进展已经证明了它在分散的数据集上学习的前景。然而，由于参与该框架的对手出于对抗目的而破坏全球模式的潜在风险，大量工作引起了关注。通过对自然语言处理模型的稀有词嵌入，研究了模型中毒用于后门攻击的可行性。在文本分类中，只有不到1%的敌意客户端足以在不降低干净句子性能的情况下操纵模型输出。对于不太复杂的数据集，仅0.1%的恶意客户端就足以有效地毒化全球模型。我们还提出了一种专门用于联邦学习方案的技术，称为梯度集成，它在所有实验设置中都提高了后门性能。



## **28. TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack**

TextHacker：用于文本硬标签攻击的基于学习的混合局部搜索算法 cs.CL

Accepted by EMNLP 2022 Findings, Code is available at  https://github.com/JHL-HUST/TextHacker

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2201.08193v2) [paper-pdf](http://arxiv.org/pdf/2201.08193v2)

**Authors**: Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He

**Abstract**: Existing textual adversarial attacks usually utilize the gradient or prediction confidence to generate adversarial examples, making it hard to be deployed in real-world applications. To this end, we consider a rarely investigated but more rigorous setting, namely hard-label attack, in which the attacker can only access the prediction label. In particular, we find we can learn the importance of different words via the change on prediction label caused by word substitutions on the adversarial examples. Based on this observation, we propose a novel adversarial attack, termed Text Hard-label attacker (TextHacker). TextHacker randomly perturbs lots of words to craft an adversarial example. Then, TextHacker adopts a hybrid local search algorithm with the estimation of word importance from the attack history to minimize the adversarial perturbation. Extensive evaluations for text classification and textual entailment show that TextHacker significantly outperforms existing hard-label attacks regarding the attack performance as well as adversary quality.

摘要: 现有的文本对抗性攻击通常利用梯度或预测置信度来生成对抗性实例，这使得它很难应用于实际应用中。为此，我们考虑了一种很少被研究但更严格的环境，即硬标签攻击，在这种攻击中，攻击者只能访问预测标签。特别是，我们发现我们可以通过对抗性例子上的单词替换引起的预测标签的变化来了解不同单词的重要性。基于此，我们提出了一种新的对抗性攻击，称为文本硬标签攻击者(TextHacker)。TextHacker随机扰乱大量单词来制作一个对抗性的例子。然后，TextHacker采用了一种混合局部搜索算法，并从攻击历史中估计单词的重要性，以最小化对手的扰动。对文本分类和文本蕴涵的广泛评估表明，TextHacker在攻击性能和对手质量方面都明显优于现有的硬标签攻击。



## **29. A Secure Design Pattern Approach Toward Tackling Lateral-Injection Attacks**

一种应对横向注入攻击的安全设计模式方法 cs.CR

4 pages, 3 figures. Accepted to The 15th IEEE International  Conference on Security of Information and Networks (SIN)

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12877v1) [paper-pdf](http://arxiv.org/pdf/2210.12877v1)

**Authors**: Chidera Biringa, Gökhan Kul

**Abstract**: Software weaknesses that create attack surfaces for adversarial exploits, such as lateral SQL injection (LSQLi) attacks, are usually introduced during the design phase of software development. Security design patterns are sometimes applied to tackle these weaknesses. However, due to the stealthy nature of lateral-based attacks, employing traditional security patterns to address these threats is insufficient. Hence, we present SEAL, a secure design that extrapolates architectural, design, and implementation abstraction levels to delegate security strategies toward tackling LSQLi attacks. We evaluated SEAL using case study software, where we assumed the role of an adversary and injected several attack vectors tasked with compromising the confidentiality and integrity of its database. Our evaluation of SEAL demonstrated its capacity to address LSQLi attacks.

摘要: 在软件开发的设计阶段，通常会引入为对抗性攻击(如横向SQL注入(LSQLi)攻击)创建攻击面的软件弱点。有时会应用安全设计模式来解决这些弱点。然而，由于基于侧向的攻击的隐蔽性，采用传统的安全模式来应对这些威胁是不够的。因此，我们提出了SEAL，这是一种安全设计，它推断出体系结构、设计和实现抽象级别，以委派安全策略来应对LSQLi攻击。我们使用案例研究软件评估了海豹突击队，我们扮演了一个对手的角色，并注入了几个攻击载体，任务是危及其数据库的机密性和完整性。我们对海豹突击队的评估表明，它有能力应对LSQLi攻击。



## **30. TAPE: Assessing Few-shot Russian Language Understanding**

录像带：评估不太可能的俄语理解 cs.CL

Accepted to EMNLP 2022 Findings

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12813v1) [paper-pdf](http://arxiv.org/pdf/2210.12813v1)

**Authors**: Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

**Abstract**: Recent advances in zero-shot and few-shot learning have shown promise for a scope of research and practical purposes. However, this fast-growing area lacks standardized evaluation suites for non-English languages, hindering progress outside the Anglo-centric paradigm. To address this line of research, we propose TAPE (Text Attack and Perturbation Evaluation), a novel benchmark that includes six more complex NLU tasks for Russian, covering multi-hop reasoning, ethical concepts, logic and commonsense knowledge. The TAPE's design focuses on systematic zero-shot and few-shot NLU evaluation: (i) linguistic-oriented adversarial attacks and perturbations for analyzing robustness, and (ii) subpopulations for nuanced interpretation. The detailed analysis of testing the autoregressive baselines indicates that simple spelling-based perturbations affect the performance the most, while paraphrasing the input has a more negligible effect. At the same time, the results demonstrate a significant gap between the neural and human baselines for most tasks. We publicly release TAPE (tape-benchmark.com) to foster research on robust LMs that can generalize to new tasks when little to no supervision is available.

摘要: 零射击和少射击学习的最新进展显示出了研究范围和实用目的的前景。然而，这个快速发展的领域缺乏针对非英语语言的标准化评估套件，阻碍了以英语为中心的范式之外的进步。针对这一研究方向，我们提出了TAPE(文本攻击和扰动评估)，这是一个新的基准测试，包括六个更复杂的俄语自然语言理解任务，涵盖了多跳推理、伦理概念、逻辑和常识知识。这盘磁带的设计侧重于系统的零镜头和少镜头NLU评估：(I)面向语言的对抗性攻击和扰动，用于分析稳健性；(Ii)亚群，用于细微差别的解释。对自回归基线测试的详细分析表明，基于拼写的简单扰动对成绩的影响最大，而释义输入的影响较小。与此同时，结果表明，在大多数任务中，神经基线和人类基线之间存在着显著的差距。我们公开发布磁带(Tape-Benchmark.com)，以促进对健壮的LMS的研究，这些LMS可以在几乎没有监督的情况下推广到新任务。



## **31. Adversarial Pretraining of Self-Supervised Deep Networks: Past, Present and Future**

自监督深度网络的对抗性预训练：过去、现在和未来 cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.13463v1) [paper-pdf](http://arxiv.org/pdf/2210.13463v1)

**Authors**: Guo-Jun Qi, Mubarak Shah

**Abstract**: In this paper, we review adversarial pretraining of self-supervised deep networks including both convolutional neural networks and vision transformers. Unlike the adversarial training with access to labeled examples, adversarial pretraining is complicated as it only has access to unlabeled examples. To incorporate adversaries into pretraining models on either input or feature level, we find that existing approaches are largely categorized into two groups: memory-free instance-wise attacks imposing worst-case perturbations on individual examples, and memory-based adversaries shared across examples over iterations. In particular, we review several representative adversarial pretraining models based on Contrastive Learning (CL) and Masked Image Modeling (MIM), respectively, two popular self-supervised pretraining methods in literature. We also review miscellaneous issues about computing overheads, input-/feature-level adversaries, as well as other adversarial pretraining approaches beyond the above two groups. Finally, we discuss emerging trends and future directions about the relations between adversarial and cooperative pretraining, unifying adversarial CL and MIM pretraining, and the trade-off between accuracy and robustness in adversarial pretraining.

摘要: 本文回顾了自监督深度网络的对抗性预训练，包括卷积神经网络和视觉转换器。与获得标记样本的对抗性训练不同，对抗性预训练是复杂的，因为它只能访问未标记的样本。为了将对手纳入到输入或特征级别的预训练模型中，我们发现现有的方法主要分为两类：无记忆的实例攻击对单个实例施加最坏情况的扰动，以及基于记忆的对手在迭代过程中跨实例共享。特别是，我们回顾了几种代表性的基于对比学习(CL)和掩蔽图像建模(MIM)的对抗性预训练模型，这两种方法是文献中流行的两种自我监督预训练方法。我们还审查了有关计算管理费用、输入/特征级别的对手以及以上两组以外的其他对抗性预训练方法的杂项问题。最后，我们讨论了对抗性预训练和合作预训练之间的关系，对抗性CL和MIM预训练的统一，以及对抗性预训练中准确性和稳健性之间的权衡等方面的发展趋势和未来方向。



## **32. GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**

GANI：基于不可察觉节点注入的图神经网络全局攻击 cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12598v1) [paper-pdf](http://arxiv.org/pdf/2210.12598v1)

**Authors**: Junyuan Fang, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks (GNNs) have found successful applications in various graph-related tasks. However, recent studies have shown that many GNNs are vulnerable to adversarial attacks. In a vast majority of existing studies, adversarial attacks on GNNs are launched via direct modification of the original graph such as adding/removing links, which may not be applicable in practice. In this paper, we focus on a realistic attack operation via injecting fake nodes. The proposed Global Attack strategy via Node Injection (GANI) is designed under the comprehensive consideration of an unnoticeable perturbation setting from both structure and feature domains. Specifically, to make the node injections as imperceptible and effective as possible, we propose a sampling operation to determine the degree of the newly injected nodes, and then generate features and select neighbors for these injected nodes based on the statistical information of features and evolutionary perturbations obtained from a genetic algorithm, respectively. In particular, the proposed feature generation mechanism is suitable for both binary and continuous node features. Extensive experimental results on benchmark datasets against both general and defended GNNs show strong attack performance of GANI. Moreover, the imperceptibility analyses also demonstrate that GANI achieves a relatively unnoticeable injection on benchmark datasets.

摘要: 图神经网络(GNN)在各种与图相关的任务中得到了成功的应用。然而，最近的研究表明，许多GNN容易受到对抗性攻击。在现有的绝大多数研究中，对GNN的对抗性攻击是通过直接修改原始图形来发起的，例如添加/删除链接，这在实践中可能并不适用。在本文中，我们关注的是一种通过注入伪节点进行的真实攻击操作。通过节点注入的全局攻击策略(GANI)是在综合考虑结构域和特征域中不可察觉的扰动设置的基础上设计的。具体地说，为了使节点注入尽可能隐蔽和有效，我们提出了一种抽样操作来确定新注入节点的程度，然后根据遗传算法获得的特征统计信息和进化扰动分别为这些注入节点生成特征和选择邻居。特别是，所提出的特征生成机制既适用于二进制节点特征，也适用于连续节点特征。在基准数据集上对一般GNN和防御GNN的大量实验结果表明，GANI具有很强的攻击性能。此外，不可感知性分析还表明，GANI在基准数据集上实现了相对不明显的注入。



## **33. Efficient (Soft) Q-Learning for Text Generation with Limited Good Data**

有效(软)Q-学习在有限好数据下的文本生成 cs.CL

Code available at  https://github.com/HanGuo97/soft-Q-learning-for-text-generation

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2106.07704v4) [paper-pdf](http://arxiv.org/pdf/2106.07704v4)

**Authors**: Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu

**Abstract**: Maximum likelihood estimation (MLE) is the predominant algorithm for training text generation models. This paradigm relies on direct supervision examples, which is not applicable to many emerging applications, such as generating adversarial attacks or generating prompts to control language models. Reinforcement learning (RL) on the other hand offers a more flexible solution by allowing users to plug in arbitrary task metrics as reward. Yet previous RL algorithms for text generation, such as policy gradient (on-policy RL) and Q-learning (off-policy RL), are often notoriously inefficient or unstable to train due to the large sequence space and the sparse reward received only at the end of sequences. In this paper, we introduce a new RL formulation for text generation from the soft Q-learning (SQL) perspective. It enables us to draw from the latest RL advances, such as path consistency learning, to combine the best of on-/off-policy updates, and learn effectively from sparse reward. We apply the approach to a wide range of novel text generation tasks, including learning from noisy/negative examples, adversarial attacks, and prompt generation. Experiments show our approach consistently outperforms both task-specialized algorithms and the previous RL methods.

摘要: 最大似然估计(MLE)是训练文本生成模型的主要算法。这种模式依赖于直接监督示例，这不适用于许多新兴的应用程序，例如生成对抗性攻击或生成提示以控制语言模型。另一方面，强化学习(RL)提供了一种更灵活的解决方案，允许用户插入任意任务指标作为奖励。然而，先前用于文本生成的RL算法，例如策略梯度(On-Policy RL)和Q-学习(Off-Policy RL)，由于大的序列空间和仅在序列末尾接收的稀疏回报，训练起来常常是出了名的低效或不稳定。本文从软Q-学习(SQL)的角度出发，提出了一种新的RL文本生成方法。它使我们能够借鉴RL的最新进展，如路径一致性学习，结合最好的开/关策略更新，并从稀疏奖励中有效学习。我们将该方法应用于一系列新颖的文本生成任务，包括从噪声/负面示例中学习、对抗性攻击和提示生成。实验表明，我们的方法的性能始终优于任务专门化算法和以前的RL方法。



## **34. Hindering Adversarial Attacks with Implicit Neural Representations**

用隐式神经表示法阻止敌意攻击 cs.LG

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.13982v1) [paper-pdf](http://arxiv.org/pdf/2210.13982v1)

**Authors**: Andrei A. Rusu, Dan A. Calian, Sven Gowal, Raia Hadsell

**Abstract**: We introduce the Lossy Implicit Network Activation Coding (LINAC) defence, an input transformation which successfully hinders several common adversarial attacks on CIFAR-$10$ classifiers for perturbations up to $\epsilon = 8/255$ in $L_\infty$ norm and $\epsilon = 0.5$ in $L_2$ norm. Implicit neural representations are used to approximately encode pixel colour intensities in $2\text{D}$ images such that classifiers trained on transformed data appear to have robustness to small perturbations without adversarial training or large drops in performance. The seed of the random number generator used to initialise and train the implicit neural representation turns out to be necessary information for stronger generic attacks, suggesting its role as a private key. We devise a Parametric Bypass Approximation (PBA) attack strategy for key-based defences, which successfully invalidates an existing method in this category. Interestingly, our LINAC defence also hinders some transfer and adaptive attacks, including our novel PBA strategy. Our results emphasise the importance of a broad range of customised attacks despite apparent robustness according to standard evaluations. LINAC source code and parameters of defended classifier evaluated throughout this submission are available: https://github.com/deepmind/linac

摘要: 我们引入了有损隐式网络激活编码(LINAC)防御，这是一种输入变换，它成功地阻止了对CIFAR-$10$分类器的几种常见的对抗性攻击，其中$L_INFTY$范数中的$\epsilon=8/255$和$L_2$范数中的$\epsilon=0.5$。隐式神经表示被用来对$2\Text{D}$图像中的像素颜色强度进行近似编码，使得在变换后的数据上训练的分类器对于小的扰动似乎具有鲁棒性，而不需要相反的训练或性能的大幅下降。用于初始化和训练隐式神经表示的随机数生成器的种子被证明是更强大的通用攻击的必要信息，这表明它扮演着私钥的角色。我们设计了一种基于密钥防御的参数旁路近似(PBA)攻击策略，该策略成功地使该类别中的现有方法无效。有趣的是，我们的直线加速器防守也阻碍了一些转移和自适应攻击，包括我们新颖的PBA策略。我们的结果强调了广泛的定制攻击的重要性，尽管根据标准评估，攻击具有明显的健壮性。在整个提交过程中评估的防御型分类器的Linac源代码和参数可用：https://github.com/deepmind/linac



## **35. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

RORL：基于保守平滑的稳健离线强化学习 cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2206.02829v3) [paper-pdf](http://arxiv.org/pdf/2206.02829v3)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstract**: Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.

摘要: 离线强化学习(RL)为利用大量的离线数据进行复杂的决策任务提供了一个很有前途的方向。由于分布平移问题，目前的离线RL算法在价值估计和动作选择上通常被设计为保守的。然而，当在实际条件下遇到观测偏差时，这种保守性会削弱学习策略的稳健性，例如传感器错误和对抗性攻击。为了权衡稳健性和保守性，我们提出了一种新的保守平滑技术的稳健离线强化学习(RORL)。在RORL中，我们明确地引入了关于策略的正则化和数据集附近状态的值函数，以及关于这些状态的附加保守值估计。理论上，我们证明了RORL在线性MDP中享有比最近的理论结果更紧的次优界。我们证明了RORL可以在一般的离线RL基准上获得最先进的性能，并且对对抗性观测扰动具有相当强的鲁棒性。



## **36. ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation**

ADDMU：用数据检测远边界对抗性实例和模型不确定性估计 cs.CL

18 pages, EMNLP 2022, main conference, long paper

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.12396v1) [paper-pdf](http://arxiv.org/pdf/2210.12396v1)

**Authors**: Fan Yin, Yao Li, Cho-Jui Hsieh, Kai-Wei Chang

**Abstract**: Adversarial Examples Detection (AED) is a crucial defense technique against adversarial attacks and has drawn increasing attention from the Natural Language Processing (NLP) community. Despite the surge of new AED methods, our studies show that existing methods heavily rely on a shortcut to achieve good performance. In other words, current search-based adversarial attacks in NLP stop once model predictions change, and thus most adversarial examples generated by those attacks are located near model decision boundaries. To surpass this shortcut and fairly evaluate AED methods, we propose to test AED methods with \textbf{F}ar \textbf{B}oundary (\textbf{FB}) adversarial examples. Existing methods show worse than random guess performance under this scenario. To overcome this limitation, we propose a new technique, \textbf{ADDMU}, \textbf{a}dversary \textbf{d}etection with \textbf{d}ata and \textbf{m}odel \textbf{u}ncertainty, which combines two types of uncertainty estimation for both regular and FB adversarial example detection. Our new method outperforms previous methods by 3.6 and 6.0 \emph{AUC} points under each scenario. Finally, our analysis shows that the two types of uncertainty provided by \textbf{ADDMU} can be leveraged to characterize adversarial examples and identify the ones that contribute most to model's robustness in adversarial training.

摘要: 对抗性实例检测(AED)是对抗敌意攻击的一种重要防御技术，越来越受到自然语言处理(NLP)领域的关注。尽管新的AED方法激增，但我们的研究表明，现有方法严重依赖于一条捷径来获得良好的性能。换言之，当前NLP中基于搜索的对抗性攻击一旦模型预测发生变化就会停止，因此由这些攻击生成的大多数对抗性示例都位于模型决策边界附近。为了超越这一捷径并公平地评价AED方法，我们建议用对抗性的例子来测试AED方法。在这种情况下，现有的方法表现出比随机猜测更差的性能。为了克服这一局限性，我们提出了一种新的方法在每种情况下，我们的新方法比以前的方法分别提高3.6和6.0emph{AUC}点。最后，我们的分析表明，文本bf{ADDMU}提供的两种类型的不确定性可以用来刻画对抗性例子，并确定对对抗性训练中模型的稳健性贡献最大的那些。



## **37. TCAB: A Large-Scale Text Classification Attack Benchmark**

TCAB：一种大规模文本分类攻击基准 cs.LG

32 pages, 7 figures, and 14 tables

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12233v1) [paper-pdf](http://arxiv.org/pdf/2210.12233v1)

**Authors**: Kalyani Asthana, Zhouhang Xie, Wencong You, Adam Noack, Jonathan Brophy, Sameer Singh, Daniel Lowd

**Abstract**: We introduce the Text Classification Attack Benchmark (TCAB), a dataset for analyzing, understanding, detecting, and labeling adversarial attacks against text classifiers. TCAB includes 1.5 million attack instances, generated by twelve adversarial attacks targeting three classifiers trained on six source datasets for sentiment analysis and abuse detection in English. Unlike standard text classification, text attacks must be understood in the context of the target classifier that is being attacked, and thus features of the target classifier are important as well. TCAB includes all attack instances that are successful in flipping the predicted label; a subset of the attacks are also labeled by human annotators to determine how frequently the primary semantics are preserved. The process of generating attacks is automated, so that TCAB can easily be extended to incorporate new text attacks and better classifiers as they are developed. In addition to the primary tasks of detecting and labeling attacks, TCAB can also be used for attack localization, attack target labeling, and attack characterization. TCAB code and dataset are available at https://react-nlp.github.io/tcab/.

摘要: 我们介绍了文本分类攻击基准(TCAB)，这是一个用于分析、理解、检测和标记针对文本分类器的敌意攻击的数据集。TCAB包括150万个攻击实例，由12个针对三个分类器的对抗性攻击生成，这些分类器在六个源数据集上进行训练，用于英语情感分析和滥用检测。与标准文本分类不同，文本攻击必须在被攻击的目标分类器的上下文中理解，因此目标分类器的特征也很重要。TCAB包括成功翻转预测标签的所有攻击实例；人工注释员还标记攻击的子集，以确定保留主要语义的频率。生成攻击的过程是自动化的，因此TCAB可以很容易地进行扩展，以纳入新的文本攻击和开发的更好的分类器。除了检测和标记攻击的主要任务外，TCAB还可以用于攻击定位、攻击目标标记和攻击特征描述。TCAB代码和数据集可在https://react-nlp.github.io/tcab/.上获得



## **38. The Dark Side of AutoML: Towards Architectural Backdoor Search**

AutoML的黑暗面：走向建筑后门搜索 cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12179v1) [paper-pdf](http://arxiv.org/pdf/2210.12179v1)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.

摘要: 这篇论文提出了一个耐人寻味的问题：是否有可能利用神经结构搜索(NAS)作为一种新的攻击载体来发动以前不太可能的攻击？具体地说，我们提出了EVA，这是一种新的攻击，它利用NAS来发现具有固有后门的神经体系结构，并使用输入感知触发器来利用这种漏洞。与现有的攻击相比，EVA表现出许多有趣的性质：(I)它不需要污染训练数据或扰动模型参数；(Ii)它与下游微调甚至从头开始的重新训练无关；(Iii)它自然地避开了依赖于检查模型参数或训练数据的防御。通过在基准数据集上的广泛评估，我们发现EVA具有高度的规避、可转移性和健壮性，从而扩展了对手的设计范围。我们进一步描述了EVA背后的机制，这可能可以通过识别触发模式的架构级“捷径”来解释。这项工作引起了人们对当前NAS实践的关注，并指出了制定有效对策的潜在方向。



## **39. Evolution of Neural Tangent Kernels under Benign and Adversarial Training**

良性训练和对抗性训练下神经正切核的演化 cs.LG

Accepted to the Conference on Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12030v1) [paper-pdf](http://arxiv.org/pdf/2210.12030v1)

**Authors**: Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus

**Abstract**: Two key challenges facing modern deep learning are mitigating deep networks' vulnerability to adversarial attacks and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen, and the underlying feature map is fixed. In finite widths, however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work that studies the adversarial robustness of the empirical/finite NTK during training. In this work, we perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with $76.1\%$ robust accuracy under PGD attacks with $\varepsilon = 4/255$ on CIFAR-10.

摘要: 现代深度学习面临的两个关键挑战是减轻深度网络对对手攻击的脆弱性和理解深度学习的泛化能力。对于第一个问题，已经开发了许多防御策略，其中最常见的是对手训练(AT)。对于第二个挑战，已经出现的占主导地位的理论之一是神经切线核(NTK)--一种对无限宽度限制中的神经网络行为的描述。在此限制下，内核被冻结，底层功能映射被修复。然而，在有限的范围内，有证据表明，特征学习发生在训练的早期阶段(核学习)，然后是核保持固定的第二阶段(懒惰训练)。虽然以前的工作旨在通过冻结的无限宽度NTK的透镜来研究对抗脆弱性，但还没有研究经验/有限NTK在训练过程中的对抗稳健性的工作。在这项工作中，我们对经验NTK在标准训练和对抗性训练下的演化进行了实证研究，旨在消除对抗性训练对核学习和懒惰训练的影响。我们发现，在对抗性训练下，经验NTK迅速收敛到与标准训练不同的核(和特征映射)。这个新的内核提供了对手的健壮性，即使在它上面执行了非健壮的训练。此外，我们发现在固定核上的对抗性训练可以产生一个在CIFAR-10上$varepsilon=4/255$的PGD攻击下具有$76.1\$稳健精度的分类器。



## **40. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.03755v2) [paper-pdf](http://arxiv.org/pdf/2209.03755v2)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息现在是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，一个可行的解决方案是通过检索和核实相关证据来自动化索赔的事实核查。虽然最近在推动自动事实核查方面取得了重大进展，但仍然缺乏对针对这类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，除了生成多样化的和与索赔一致的证据外，还可以微妙地修改证据中突出索赔的片段。因此，在分类维度的许多不同排列下，我们会极大地降低事实检查性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，我们最后讨论了未来防御的挑战和方向。



## **41. Assaying Out-Of-Distribution Generalization in Transfer Learning**

迁移学习中的分布外泛化分析 cs.LG

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2207.09239v2) [paper-pdf](http://arxiv.org/pdf/2207.09239v2)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstract**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.

摘要: 由于分布外泛化是一个通常不适定的问题，因此对不同的代理目标(例如，校准、对手健壮性、算法腐败、跨班次不变性)进行了研究，得出了不同的建议。虽然有着相同的理想目标，但这些方法从未在相同的实验条件下对真实数据进行过测试。在本文中，我们对以前的工作进行了统一的审查，强调了我们通过经验解决的信息差异，并就如何衡量模型的稳健性以及如何改进模型提供了建议。为此，我们收集了172个公开可用的数据集对，用于训练和分布外评估准确性、校准误差、对抗性攻击、环境不变性和合成腐败。我们微调了超过31k的网络，这些网络来自9种不同的架构，在多发和少发的情况下。我们的发现证实，分布内和分布外的精度往往会共同增加，但表明它们的关系在很大程度上依赖于数据集，总体上比之前的较小规模的研究假设的更细微和更复杂。



## **42. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

discuss new and recent works as well as proof-reading

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.02299v5) [paper-pdf](http://arxiv.org/pdf/2209.02299v5)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 今天，计算机系统保存着大量的个人数据。然而，尽管如此丰富的数据使人工智能，特别是机器学习(ML)取得了突破，但它的存在可能会对用户隐私构成威胁，并可能削弱人类与人工智能之间的信任纽带。最近的法规现在要求，根据请求，必须从计算机系统和ML模型中删除关于用户的私人信息，即“被遗忘权”)。虽然从后端数据库中删除数据应该是直接的，但在人工智能上下文中这是不够的，因为ML模型经常‘记住’旧数据。当代针对训练模型的对抗性攻击已经证明，我们可以学习到一个实例或一个属性是否属于训练数据。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。因此，本文致力于对机器遗忘的概念、场景、方法和应用进行全面的考察。具体地说，作为尖端研究的类别集合，本文背后的目的是为寻求介绍机器遗忘及其公式、设计标准、移除请求、算法和应用的研究人员和从业者提供全面的资源。此外，我们的目标是强调关键的发现、当前的趋势和新的研究领域，这些领域还没有使用机器遗忘，但可以从中受益匪浅。我们希望这项调查对ML研究人员和那些寻求创新隐私技术的人来说是一个有价值的资源。我们的资源可在https://github.com/tamlhp/awesome-machine-unlearning.上公开获取



## **43. Identifying Human Strategies for Generating Word-Level Adversarial Examples**

确定生成词级对抗性实例的人类策略 cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11598v1) [paper-pdf](http://arxiv.org/pdf/2210.11598v1)

**Authors**: Maximilian Mozes, Bennett Kleinberg, Lewis D. Griffin

**Abstract**: Adversarial examples in NLP are receiving increasing research attention. One line of investigation is the generation of word-level adversarial examples against fine-tuned Transformer models that preserve naturalness and grammaticality. Previous work found that human- and machine-generated adversarial examples are comparable in their naturalness and grammatical correctness. Most notably, humans were able to generate adversarial examples much more effortlessly than automated attacks. In this paper, we provide a detailed analysis of exactly how humans create these adversarial examples. By exploring the behavioural patterns of human workers during the generation process, we identify statistically significant tendencies based on which words humans prefer to select for adversarial replacement (e.g., word frequencies, word saliencies, sentiment) as well as where and when words are replaced in an input sequence. With our findings, we seek to inspire efforts that harness human strategies for more robust NLP models.

摘要: 自然语言处理中的对抗性例子正受到越来越多的研究关注。一条研究路线是生成词级的对抗性例子，反对保持自然性和语法的微调变形金刚模型。以前的工作发现，人类和机器生成的对抗性例子在自然性和语法正确性方面具有可比性。最值得注意的是，人类能够比自动攻击更轻松地生成对抗性例子。在这篇文章中，我们对人类如何创造这些对抗性的例子提供了详细的分析。通过探索人类工作者在生成过程中的行为模式，我们基于人类更喜欢选择哪些单词作为对抗性替换(例如，单词频率、单词显著程度、情绪)以及输入序列中单词被替换的位置和时间来识别统计上显著的倾向。通过我们的发现，我们寻求激励人们努力利用人类战略来建立更强大的NLP模型。



## **44. Similarity of Neural Architectures Based on Input Gradient Transferability**

基于输入梯度传递的神经结构相似性研究 cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.

摘要: 在本文中，我们的目标是设计两个神经结构之间的定量相似性函数。具体地说，我们定义了一种使用输入梯度可转移性的模型相似性。我们生成两个网络的对抗性样本，并在对方的对抗性样本上测量网络的平均准确率。如果两个网络高度相关，那么攻击的可传递性就高，从而导致高相似度。使用相似性得分，我们研究了两个主题：(1)哪个网络组件对模型多样性有贡献？(2)模型多样性如何影响实际场景？我们通过提供特征重要性分析和聚类分析来回答第一个问题。第二个问题通过两个不同的场景进行验证：模型集成和知识提炼。我们的发现表明，在与不同的神经结构相互作用时，模型多样性起着关键作用。例如，我们发现，更多的多样性会带来更好的合奏表现。我们还观察到，教师和学生网络与蒸馏性能之间的关系取决于教师和学生网络基础架构的选择。我们希望我们的分析工具能够帮助我们更高层次地了解不同神经架构之间的差异，并在使用多种架构时提供实用指导。



## **45. Surprises in adversarially-trained linear regression**

逆训练线性回归中的惊喜 stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.

摘要: 最先进的机器学习模型很容易受到相反构造的非常小的输入扰动的影响。对抗性训练是抵御这类例子的一种有效方法。它被描述为一个最小-最大问题，当训练数据被最坏情况的攻击破坏时，搜索最优解。对于线性回归问题，对抗性训练可以表示为一个凸问题。首先，我们将训练问题描述为稳健回归的一个实例，以揭示它与参数收缩方法的联系，具体地说，对手训练产生稀疏解。其次，我们研究了参数多于数据的过度参数条件下的对抗性训练。我们证明了小干扰下的对抗性训练给出了用最小范数对训练数据进行内插的解。岭回归和套索逼近这种插值解，因为它们的正则化参数为零。相比之下，对于对抗性训练，向插补机制的转变是突然的，并且对于非零值的扰动。这一结果得到了证明，并用数值算例加以说明。



## **46. Attacking Motion Estimation with Adversarial Snow**

对抗性降雪下的攻击运动估计 cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.

摘要: 当前针对运动估计(光流)的敌意攻击优化了每像素的微小扰动，这在现实世界中不太可能出现。相反，我们利用真实世界的天气现象，用相反的优化降雪进行了一次新颖的攻击。我们攻击的核心是一个可区分的渲染器，它一致地将照片级逼真的雪花和逼真的运动整合到3D场景中。通过优化，我们得到了对抗性雪，它对光流有明显的影响，但与普通雪没有什么区别。令人惊讶的是，我们的新攻击对以前对小Lp扰动表现出高度稳健性的方法的影响最大。



## **47. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

UKP-Square v2：可解析性和针对可信QA的对抗性攻击 cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.

摘要: 问答(QA)系统越来越多地部署在支持现实世界决策的应用程序中。然而，最先进的模型依赖于深度神经网络，这很难被人类解释。本质上可解释的模型或事后可解释的方法可以帮助用户理解模型如何达到其预测，如果成功，则增加他们对系统的信任。此外，研究人员可以利用这些洞察力来开发更准确、更少偏见的新方法。在本文中，我们引入了Square的新版本Square v2，以提供基于显著图和基于图的解释等方法的模型比较的可解释性基础设施。虽然显著图有助于检查每个输入标记对于模型预测的重要性，但来自外部知识图的基于图形的解释使用户能够验证模型预测背后的推理。此外，我们还提供了多个对抗性攻击来比较QA模型的健壮性。通过这些可解释性方法和对抗性攻击，我们的目标是简化可信QA模型的研究。Square在https://square.ukp-lab.de.上可用



## **48. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

非IID场景下分布式水平和垂直联合学习体系结构的健壮性分析 cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.

摘要: 联合学习(FL)允许参与者协作训练机器和深度学习模型，同时保护数据隐私。然而，FL范式仍然存在影响其可信度的缺陷，因为恶意参与者可能会对培训过程发动对抗性攻击。相关工作研究了水平FL场景在不同攻击下的稳健性。然而，缺乏对分布式垂直FL的健壮性进行评估，并将其与水平FL体系结构进行比较的工作。因此，本文提出了三种去中心化FL架构，一种用于水平场景，两种用于垂直场景，即HoriChain、VertiChain和VertiComb。这些体系结构提供了适用于水平和垂直场景的不同神经网络和训练协议。然后，部署了一个分散的、保护隐私的联合用例，使用非IID数据对手写数字进行分类，以评估三种体系结构的性能。最后，通过一组实验计算并比较了基于图像水印的数据中毒和基于梯度中毒的敌意攻击对所提体系结构的健壮性的影响。实验表明，尽管两种攻击的特定配置都会破坏体系结构的分类性能，但HoriChain是最健壮的一种。



## **49. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

利用通用防御框架防御敌方补丁攻击的人检测 cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.

摘要: 人体检测在计算机视觉领域引起了极大的关注，是以人为中心的计算机视觉的重要组成部分。虽然个人检测网络的预测性能有了很大的提高，但它们很容易受到对手补丁的攻击。在自动驾驶和安全系统等安全关键应用中，更改受限区域的像素很容易欺骗人员检测网络。尽管有必要对抗对抗性补丁攻击，但很少有人致力于防御对抗性补丁攻击的人检测。在本文中，我们提出了一种新的防御策略，通过优化个人检测的防御框架来防御对抗性补丁攻击。该防御框架在保持人与干净的人的检测性能的同时，减轻了对手补丁的影响。利用竞争学习算法生成人体检测中的防御帧，使得人体检测中的检测威胁模块和检测屏蔽模块之间进行迭代竞争。综合实验结果表明，该方法能够有效地防御敌意补丁攻击。



## **50. Chaos Theory and Adversarial Robustness**

混沌理论与对抗稳健性 cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.13235v1) [paper-pdf](http://arxiv.org/pdf/2210.13235v1)

**Authors**: Jonathan S. Kent

**Abstract**: Neural Networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which Neural Networks are susceptible to or robust against adversarial attacks. Our results show that susceptibility to attack grows significantly with the depth of the model, which has significant safety implications for the design of Neural Networks for production environments. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly, as well as show a clear relationship between our new susceptibility metric and post-attack accuracy.

摘要: 神经网络容易受到对抗性攻击，在部署到关键或对抗性应用程序之前，应该面临严格的审查。本文使用混沌理论的思想来解释、分析和量化神经网络对敌意攻击的敏感程度或健壮性。我们的结果表明，随着模型深度的增加，对攻击的敏感性显著增加，这对于生产环境下的神经网络的设计具有重要的安全意义。我们还演示了如何快速、轻松地近似极大模型的认证稳健性半径，到目前为止，该半径在计算上无法直接计算，并显示了我们的新敏感度度量和攻击后精度之间的明确关系。



