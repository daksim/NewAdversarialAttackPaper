# Latest Adversarial Attack Papers
**update at 2022-04-14 13:30:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Overparameterized Linear Regression under Adversarial Attacks**

对抗性攻击下的超参数线性回归 stat.ML

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06274v1)

**Authors**: Antônio H. Ribeiro, Thomas B. Schön

**Abstracts**: As machine learning models start to be used in critical applications, their vulnerabilities and brittleness become a pressing concern. Adversarial attacks are a popular framework for studying these vulnerabilities. In this work, we study the error of linear regression in the face of adversarial attacks. We provide bounds of the error in terms of the traditional risk and the parameter norm and show how these bounds can be leveraged and make it possible to use analysis from non-adversarial setups to study the adversarial risk. The usefulness of these results is illustrated by shedding light on whether or not overparameterized linear models can be adversarially robust. We show that adding features to linear models might be either a source of additional robustness or brittleness. We show that these differences appear due to scaling and how the $\ell_1$ and $\ell_2$ norms of random projections concentrate. We also show how the reformulation we propose allows for solving adversarial training as a convex optimization problem. This is then used as a tool to study how adversarial training and other regularization methods might affect the robustness of the estimated models.

摘要: 随着机器学习模型开始在关键应用中使用，它们的脆弱性和脆性成为一个紧迫的问题。对抗性攻击是研究这些漏洞的流行框架。在这项工作中，我们研究了线性回归在面对对手攻击时的误差。我们给出了关于传统风险和参数范数的误差的界，并说明了如何利用这些界来利用这些界，使得从非对抗性设置的分析来研究对抗性风险成为可能。这些结果的有用之处在于揭示了过度参数化线性模型是否具有相反的稳健性。我们表明，向线性模型添加特征可能是额外的稳健性或脆性的来源。我们证明了这些差异是由于尺度以及随机投影的$\ell_1$和$\ell_2$范数是如何集中的。我们还展示了我们提出的重新公式如何允许将对抗性训练作为一个凸优化问题来解决。然后将其用作研究对抗性训练和其他正则化方法如何影响估计模型的稳健性的工具。



## **2. Towards A Critical Evaluation of Robustness for Deep Learning Backdoor Countermeasures**

深度学习后门对策的稳健性评测 cs.CR

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06273v1)

**Authors**: Huming Qiu, Hua Ma, Zhi Zhang, Alsharif Abuadbba, Wei Kang, Anmin Fu, Yansong Gao

**Abstracts**: Since Deep Learning (DL) backdoor attacks have been revealed as one of the most insidious adversarial attacks, a number of countermeasures have been developed with certain assumptions defined in their respective threat models. However, the robustness of these countermeasures is inadvertently ignored, which can introduce severe consequences, e.g., a countermeasure can be misused and result in a false implication of backdoor detection.   For the first time, we critically examine the robustness of existing backdoor countermeasures with an initial focus on three influential model-inspection ones that are Neural Cleanse (S&P'19), ABS (CCS'19), and MNTD (S&P'21). Although the three countermeasures claim that they work well under their respective threat models, they have inherent unexplored non-robust cases depending on factors such as given tasks, model architectures, datasets, and defense hyper-parameter, which are \textit{not even rooted from delicate adaptive attacks}. We demonstrate how to trivially bypass them aligned with their respective threat models by simply varying aforementioned factors. Particularly, for each defense, formal proofs or empirical studies are used to reveal its two non-robust cases where it is not as robust as it claims or expects, especially the recent MNTD. This work highlights the necessity of thoroughly evaluating the robustness of backdoor countermeasures to avoid their misleading security implications in unknown non-robust cases.

摘要: 由于深度学习(DL)后门攻击已被发现是最隐蔽的敌意攻击之一，因此已经开发了一些对策，并在各自的威胁模型中定义了某些假设。然而，这些对策的稳健性被无意中忽视了，这可能会带来严重的后果，例如，对策可能被误用，并导致后门检测的错误含义。我们首次批判性地检验了现有后门对策的稳健性，最初集中在三个有影响力的模型检查对策上，它们是神经清洗(S&P‘19)、ABS(CCS’19)和MNTD(S&P‘21)。虽然这三种对策声称它们在各自的威胁模型下都工作得很好，但它们固有的非稳健性情况取决于给定的任务、模型体系结构、数据集和防御超参数等因素，而这些因素甚至不是源于脆弱的自适应攻击}。我们将演示如何通过简单地改变上述因素来绕过它们，使其与各自的威胁模型保持一致。特别是，对于每一种辩护，形式证明或实证研究都被用来揭示它的两种不稳健的情况，其中它并不像它声称或期望的那样稳健，特别是最近的MNTD。这项工作强调了彻底评估后门对策的稳健性的必要性，以避免在未知的非稳健性情况下它们具有误导性的安全影响。



## **3. Towards Practical Robustness Analysis for DNNs based on PAC-Model Learning**

基于PAC模型学习的DNN实用健壮性分析 cs.LG

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2101.10102v2)

**Authors**: Renjue Li, Pengfei Yang, Cheng-Chao Huang, Youcheng Sun, Bai Xue, Lijun Zhang

**Abstracts**: To analyse local robustness properties of deep neural networks (DNNs), we present a practical framework from a model learning perspective. Based on black-box model learning with scenario optimisation, we abstract the local behaviour of a DNN via an affine model with the probably approximately correct (PAC) guarantee. From the learned model, we can infer the corresponding PAC-model robustness property. The innovation of our work is the integration of model learning into PAC robustness analysis: that is, we construct a PAC guarantee on the model level instead of sample distribution, which induces a more faithful and accurate robustness evaluation. This is in contrast to existing statistical methods without model learning. We implement our method in a prototypical tool named DeepPAC. As a black-box method, DeepPAC is scalable and efficient, especially when DNNs have complex structures or high-dimensional inputs. We extensively evaluate DeepPAC, with 4 baselines (using formal verification, statistical methods, testing and adversarial attack) and 20 DNN models across 3 datasets, including MNIST, CIFAR-10, and ImageNet. It is shown that DeepPAC outperforms the state-of-the-art statistical method PROVERO, and it achieves more practical robustness analysis than the formal verification tool ERAN. Also, its results are consistent with existing DNN testing work like DeepGini.

摘要: 为了分析深度神经网络(DNN)的局部稳健性，从模型学习的角度提出了一个实用的框架。基于场景优化的黑盒模型学习，我们通过仿射模型在可能近似正确(PAC)的保证下抽象DNN的局部行为。从学习的模型中，我们可以推断出相应的PAC模型的稳健性。我们工作的创新之处在于将模型学习融入到PAC稳健性分析中：即在模型级别而不是样本分布上构造PAC保证，从而得到更真实和准确的稳健性评估。这与没有模型学习的现有统计方法形成了鲜明对比。我们在一个原型工具DeepPAC中实现了我们的方法。作为一种黑盒方法，DeepPAC具有可扩展性和高效率，特别是当DNN具有复杂的结构或高维输入时。我们对DeepPAC进行了广泛的评估，使用了4个基线(使用正式验证、统计方法、测试和对抗性攻击)和20个DNN模型，涉及MNIST、CIFAR-10和ImageNet等3个数据集。结果表明，DeepPAC的性能优于目前最先进的统计方法PROVERO，并且比形式化验证工具ERAN实现了更实用的健壮性分析。此外，它的结果与DeepGini等现有的DNN测试工作是一致的。



## **4. Stealing Malware Classifiers and AVs at Low False Positive Conditions**

在低误报条件下窃取恶意软件分类器和反病毒软件 cs.CR

12 pages, 8 figures, 6 tables. Under review

**SubmitDate**: 2022-04-13    [paper-pdf](http://arxiv.org/pdf/2204.06241v1)

**Authors**: Maria Rigaki, Sebastian Garcia

**Abstracts**: Model stealing attacks have been successfully used in many machine learning domains, but there is little understanding of how these attacks work in the malware detection domain. Malware detection and, in general, security domains have very strong requirements of low false positive rates (FPR). However, these requirements are not the primary focus of the existing model stealing literature. Stealing attacks create surrogate models that perform similarly to a target model using a limited amount of queries to the target. The first stage of this study is the evaluation of active learning model stealing attacks against publicly available stand-alone machine learning malware classifiers and antivirus products (AVs). We propose a new neural network architecture for surrogate models that outperforms the existing state of the art on low FPR conditions. The surrogates were evaluated on their agreement with the targeted models. Good surrogates of the stand-alone classifiers were created with up to 99% agreement with the target models, using less than 4% of the original training dataset size. Good AV surrogates were also possible to train, but with a lower agreement. The second stage used the best surrogates as well as the target models to generate adversarial malware using the MAB framework to test stand-alone models and AVs (offline and online). Results showed that surrogate models could generate adversarial samples that evade the targets but are less successful than the targets themselves. Using surrogates, however, is a necessity for attackers, given that attacks against AVs are extremely time-consuming and easily detected when the AVs are connected to the internet.

摘要: 模型窃取攻击已经成功地应用于许多机器学习领域，但对于这些攻击在恶意软件检测领域的工作原理却知之甚少。通常，恶意软件检测和安全域对低误报比率(FPR)有非常强烈的要求。然而，这些要求并不是现有窃取文献模型的主要关注点。窃取攻击创建代理模型，该代理模型使用对目标的有限数量的查询来执行类似于目标模型的操作。本研究的第一阶段是评估针对公开可用的独立机器学习恶意软件分类器和反病毒产品(AV)的主动学习模型窃取攻击。我们提出了一种新的神经网络体系结构，用于代理模型，在低FPR条件下性能优于现有技术。根据其与目标模型的一致性对代用品进行评估。使用不到原始训练数据集大小的4%，创建了独立分类器的良好代理，与目标模型的一致性高达99%。好的反病毒代言人也有可能接受培训，但协议的一致性较低。第二阶段使用最好的代理以及目标模型来生成敌意恶意软件，使用MAB框架测试独立模型和AVs(离线和在线)。结果表明，代理模型可以生成避开目标的对抗性样本，但不如目标本身那么成功。然而，使用代理对于攻击者来说是必要的，因为针对AVs的攻击非常耗时，并且在AVs连接到互联网时很容易被检测到。



## **5. Liuer Mihou: A Practical Framework for Generating and Evaluating Grey-box Adversarial Attacks against NIDS**

六二密侯：一种实用的生成和评估针对NIDS的灰盒攻击的框架 cs.CR

16 pages, 8 figures, planning on submitting to ACM CCS 2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06113v1)

**Authors**: Ke He, Dan Dongseong Kim, Jing Sun, Jeong Do Yoo, Young Hun Lee, Huy Kang Kim

**Abstracts**: Due to its high expressiveness and speed, Deep Learning (DL) has become an increasingly popular choice as the detection algorithm for Network-based Intrusion Detection Systems (NIDSes). Unfortunately, DL algorithms are vulnerable to adversarial examples that inject imperceptible modifications to the input and cause the DL algorithm to misclassify the input. Existing adversarial attacks in the NIDS domain often manipulate the traffic features directly, which hold no practical significance because traffic features cannot be replayed in a real network. It remains a research challenge to generate practical and evasive adversarial attacks.   This paper presents the Liuer Mihou attack that generates practical and replayable adversarial network packets that can bypass anomaly-based NIDS deployed in the Internet of Things (IoT) networks. The core idea behind Liuer Mihou is to exploit adversarial transferability and generate adversarial packets on a surrogate NIDS constrained by predefined mutation operations to ensure practicality. We objectively analyse the evasiveness of Liuer Mihou against four ML-based algorithms (LOF, OCSVM, RRCF, and SOM) and the state-of-the-art NIDS, Kitsune. From the results of our experiment, we gain valuable insights into necessary conditions on the adversarial transferability of anomaly detection algorithms. Going beyond a theoretical setting, we replay the adversarial attack in a real IoT testbed to examine the practicality of Liuer Mihou. Furthermore, we demonstrate that existing feature-level adversarial defence cannot defend against Liuer Mihou and constructively criticise the limitations of feature-level adversarial defences.

摘要: 深度学习以其高的表现力和速度成为基于网络的入侵检测系统(NIDSS)的检测算法之一。不幸的是，DL算法容易受到敌意示例的攻击，这些示例向输入注入难以察觉的修改，并导致DL算法错误地对输入进行分类。现有的NIDS域中的对抗性攻击往往直接操纵流量特征，由于流量特征不能在真实网络中再现，因此没有实际意义。产生实用的和躲避的对抗性攻击仍然是一个研究挑战。针对物联网网络中部署的基于异常的网络入侵检测系统，提出了一种能够生成实用的、可重放的敌意网络数据包的六二米侯攻击方法。六儿后的核心思想是利用敌意可转移性，在预定义的变异操作约束下的代理网络入侵检测系统上生成对抗性的报文，以确保实用性。我们客观地分析了六儿密侯对四种基于ML的算法(LOF、OCSVM、RRCF和SOM)以及最新的网络入侵检测系统Kitsune的规避能力。从实验结果中，我们对异常检测算法的对抗性转移的必要条件得到了有价值的见解。超越理论设置，我们在真实的IoT试验台上重播对抗性攻击，以检验六儿后的实用性。此外，我们证明了现有的特征级对抗性防御不能抵抗六二密侯，并建设性地批评了特征级对抗性防御的局限性。



## **6. Optimal Membership Inference Bounds for Adaptive Composition of Sampled Gaussian Mechanisms**

采样高斯机构自适应组合的最优成员推理界 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2204.06106v1)

**Authors**: Saeed Mahloujifar, Alexandre Sablayrolles, Graham Cormode, Somesh Jha

**Abstracts**: Given a trained model and a data sample, membership-inference (MI) attacks predict whether the sample was in the model's training set. A common countermeasure against MI attacks is to utilize differential privacy (DP) during model training to mask the presence of individual examples. While this use of DP is a principled approach to limit the efficacy of MI attacks, there is a gap between the bounds provided by DP and the empirical performance of MI attacks. In this paper, we derive bounds for the \textit{advantage} of an adversary mounting a MI attack, and demonstrate tightness for the widely-used Gaussian mechanism. We further show bounds on the \textit{confidence} of MI attacks. Our bounds are much stronger than those obtained by DP analysis. For example, analyzing a setting of DP-SGD with $\epsilon=4$ would obtain an upper bound on the advantage of $\approx0.36$ based on our analyses, while getting bound of $\approx 0.97$ using the analysis of previous work that convert $\epsilon$ to membership inference bounds.   Finally, using our analysis, we provide MI metrics for models trained on CIFAR10 dataset. To the best of our knowledge, our analysis provides the state-of-the-art membership inference bounds for the privacy.

摘要: 在给定训练模型和数据样本的情况下，成员推理(MI)攻击预测样本是否在模型的训练集中。针对MI攻击的一种常见对策是在模型训练期间利用差异隐私(DP)来掩盖单个示例的存在。虽然使用DP是一种原则性的方法来限制MI攻击的有效性，但是DP提供的界限和MI攻击的经验性能之间存在差距。在这篇文章中，我们得到了敌手发起MI攻击的优势的界，并证明了广泛使用的Gauss机制的紧性。我们进一步给出了MI攻击的置信度的界。我们的边界比DP分析得到的边界要强得多。例如，根据我们的分析，分析具有$\epsilon=4$的DP-SGD的设置将得到$\约0.36$的优势的上界，而使用将$\epsilon$转换为成员推理界的前人工作的分析，得到$\约0.97$的上界。最后，利用我们的分析，我们提供了在CIFAR10数据集上训练的模型的MI度量。据我们所知，我们的分析为隐私提供了最先进的成员关系推断界限。



## **7. Membership Inference Attacks From First Principles**

从第一性原理出发的成员推理攻击 cs.CR

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.03570v2)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

摘要: 成员关系推理攻击允许对手查询经过训练的机器学习模型，以预测特定示例是否包含在该模型的训练数据集中。目前，这些攻击是使用平均案例“准确性”度量来评估的，该度量无法确定攻击是否可以自信地识别训练集的任何成员。我们认为，应该通过在较低的(例如<0.1%)假阳性率下计算真阳性率来评估攻击，并发现大多数先前的攻击在以这种方式评估时表现很差。为了解决这个问题，我们开发了一种似然比攻击(LIRA)，它仔细地结合了文献中的多种想法。我们的攻击在低假阳性率下的威力要高出10倍，而且还严格控制了之前对现有指标的攻击。



## **8. Rate Coding or Direct Coding: Which One is Better for Accurate, Robust, and Energy-efficient Spiking Neural Networks?**

速率编码和直接编码：哪种编码更适合准确、健壮和节能的尖峰神经网络？ cs.NE

Accepted to ICASSP2022

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2202.03133v2)

**Authors**: Youngeun Kim, Hyoungseob Park, Abhishek Moitra, Abhiroop Bhattacharjee, Yeshwanth Venkatesha, Priyadarshini Panda

**Abstracts**: Recent Spiking Neural Networks (SNNs) works focus on an image classification task, therefore various coding techniques have been proposed to convert an image into temporal binary spikes. Among them, rate coding and direct coding are regarded as prospective candidates for building a practical SNN system as they show state-of-the-art performance on large-scale datasets. Despite their usage, there is little attention to comparing these two coding schemes in a fair manner. In this paper, we conduct a comprehensive analysis of the two codings from three perspectives: accuracy, adversarial robustness, and energy-efficiency. First, we compare the performance of two coding techniques with various architectures and datasets. Then, we measure the robustness of the coding techniques on two adversarial attack methods. Finally, we compare the energy-efficiency of two coding schemes on a digital hardware platform. Our results show that direct coding can achieve better accuracy especially for a small number of timesteps. In contrast, rate coding shows better robustness to adversarial attacks owing to the non-differentiable spike generation process. Rate coding also yields higher energy-efficiency than direct coding which requires multi-bit precision for the first layer. Our study explores the characteristics of two codings, which is an important design consideration for building SNNs. The code is made available at https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.

摘要: 最近的尖峰神经网络(SNN)专注于图像分类任务，因此已经提出了各种编码技术来将图像转换为时间二进制尖峰。其中，码率编码和直接编码由于在大规模数据集上表现出最先进的性能，被认为是构建实用SNN系统的潜在候选者。尽管使用了这两种编码方案，但很少有人注意以公平的方式比较这两种编码方案。在本文中，我们从准确性、对手健壮性和能量效率三个角度对这两种编码进行了全面的分析。首先，我们比较了两种编码技术在不同架构和不同数据集下的性能。然后，我们测量了编码技术在两种对抗性攻击方法上的健壮性。最后，在数字硬件平台上对两种编码方案的能量效率进行了比较。我们的结果表明，直接编码可以获得更好的精度，特别是在少量时间步长的情况下。相反，由于不可微的尖峰产生过程，速率编码表现出更好的对对手攻击的稳健性。速率编码还产生比直接编码更高的能量效率，直接编码需要第一层的多比特精度。我们的研究探索了两种编码的特点，这是构建SNN的重要设计考虑因素。代码可在https://github.com/Intelligent-Computing-Lab-Yale/Rate-vs-Direct.上获得



## **9. Masked Faces with Faced Masks**

戴着面具的蒙面人 cs.CV

8 pages

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2201.06427v2)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstracts**: Modern face recognition systems (FRS) still fall short when the subjects are wearing facial masks, a common theme in the age of respiratory pandemics. An intuitive partial remedy is to add a mask detector to flag any masked faces so that the FRS can act accordingly for those low-confidence masked faces. In this work, we set out to investigate the potential vulnerability of such FRS equipped with a mask detector, on large-scale masked faces, which might trigger a serious risk, e.g., letting a suspect evade the FRS where both facial identity and mask are undetected. As existing face recognizers and mask detectors have high performance in their respective tasks, it is significantly challenging to simultaneously fool them and preserve the transferability of the attack. We formulate the new task as the generation of realistic & adversarial-faced mask and make three main contributions: First, we study the naive Delanunay-based masking method (DM) to simulate the process of wearing a faced mask that is cropped from a template image, which reveals the main challenges of this new task. Second, we further equip the DM with the adversarial noise attack and propose the adversarial noise Delaunay-based masking method (AdvNoise-DM) that can fool the face recognition and mask detection effectively but make the face less natural. Third, we propose the adversarial filtering Delaunay-based masking method denoted as MF2M by employing the adversarial filtering for AdvNoise-DM and obtain more natural faces. With the above efforts, the final version not only leads to significant performance deterioration of the state-of-the-art (SOTA) deep learning-based FRS, but also remains undetected by the SOTA facial mask detector, thus successfully fooling both systems at the same time.

摘要: 当受试者戴着口罩时，现代人脸识别系统(FRS)仍然不足，这在呼吸道大流行的时代是一个常见的主题。一种直观的部分补救方法是添加一个掩模检测器来标记任何掩蔽的人脸，以便FRS可以对这些低置信度的掩蔽人脸采取相应的行动。在这项工作中，我们开始调查这种配备了面具检测器的FRS在大规模蒙面人脸上的潜在脆弱性，这可能会引发严重的风险，例如，让嫌疑人逃避FRS，其中面部身份和面具都没有被检测到。由于现有的人脸识别器和面具检测器在各自的任务中具有很高的性能，因此要同时欺骗它们并保持攻击的可转移性是非常具有挑战性的。首先，我们研究了基于朴素Delanunay的掩蔽方法(DM)来模拟从模板图像中裁剪出来的人脸面具的佩戴过程，揭示了这一新任务的主要挑战。其次，进一步为DM提供了对抗性噪声攻击，并提出了基于对抗性噪声Delaunay的掩蔽方法(AdvNoise-DM)，该方法可以有效地欺骗人脸识别和掩模检测，但会使人脸变得不自然。第三，针对AdvNoise-DM，提出了一种基于对抗过滤的Delaunay掩蔽方法MF2M，得到了更自然的人脸。在上述努力下，最终版本不仅导致基于最先进的(SOTA)深度学习的FRS的性能显著恶化，而且仍然没有被SOTA面膜检测器检测到，从而成功地同时愚弄了两个系统。



## **10. Automated Attacker Synthesis for Distributed Protocols**

分布式协议的自动攻击者综合 cs.CR

24 pages, 15 figures

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2004.01220v4)

**Authors**: Max von Hippel, Cole Vick, Stavros Tripakis, Cristina Nita-Rotaru

**Abstracts**: Distributed protocols should be robust to both benign malfunction (e.g. packet loss or delay) and attacks (e.g. message replay) from internal or external adversaries. In this paper we take a formal approach to the automated synthesis of attackers, i.e. adversarial processes that can cause the protocol to malfunction. Specifically, given a formal threat model capturing the distributed protocol model and network topology, as well as the placement, goals, and interface (inputs and outputs) of potential attackers, we automatically synthesize an attacker. We formalize four attacker synthesis problems - across attackers that always succeed versus those that sometimes fail, and attackers that attack forever versus those that do not - and we propose algorithmic solutions to two of them. We report on a prototype implementation called KORG and its application to TCP as a case-study. Our experiments show that KORG can automatically generate well-known attacks for TCP within seconds or minutes.

摘要: 分布式协议应该对来自内部或外部对手的良性故障(例如，分组丢失或延迟)和攻击(例如，消息重放)具有健壮性。在本文中，我们采用了一种形式化的方法来自动合成攻击者，即可能导致协议故障的对抗性过程。具体地说，给定一个捕获分布式协议模型和网络拓扑以及潜在攻击者的位置、目标和接口(输入和输出)的正式威胁模型，我们将自动合成攻击者。我们将四个攻击者合成问题形式化--总是成功的攻击者与有时失败的攻击者，以及永远攻击的攻击者与不成功的攻击者--并针对其中两个问题提出了算法解决方案。我们报告了一个称为KORG的原型实现以及它在TCP中的应用作为案例研究。我们的实验表明，KORG可以在几秒或几分钟内自动生成针对TCP的知名攻击。



## **11. Catch Me If You Can: Blackbox Adversarial Attacks on Automatic Speech Recognition using Frequency Masking**

抓住我：使用频率掩蔽对自动语音识别进行黑箱对抗性攻击 cs.SD

11 pages, 7 figures and 3 tables

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2112.01821v2)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) models are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the security and robustnesss of ASRS, we propose techniques that generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. This is in contrast to existing work that focuses on whitebox targeted attacks that are time consuming and lack portability.   Our techniques generate adversarial attacks that have no human audible difference by manipulating the audio signal using a psychoacoustic model that maintains the audio perturbations below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and two input audio datasets using the metrics - Word Error Rate (WER) of output transcription, Similarity to original audio, attack Success Rate on different ASRs and Detection score by a defense system. We found our adversarial attacks were portable across ASRs, not easily detected by a state-of-the-art defense system, and had significant difference in output transcriptions while sounding similar to original audio.

摘要: 自动语音识别(ASR)模型非常普遍，特别是在家用电器的语音导航和语音控制应用中。ASR的计算核心是深度神经网络(DNN)，已被证明容易受到对手扰动的影响；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的安全性和健壮性，我们提出了生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。这与现有的专注于白盒目标攻击的工作形成了鲜明对比，这些攻击既耗时又缺乏可移植性。我们的技术通过使用将音频扰动保持在人类感知阈值以下的心理声学模型来操纵音频信号，从而产生没有人类听觉差异的对抗性攻击。我们使用三个流行的ASR和两个输入音频数据集，使用输出转录的错误率(WER)、与原始音频的相似性、对不同ASR的攻击成功率和防御系统的检测分数来评估我们的技术的可移植性和有效性。我们发现，我们的敌意攻击可以跨ASR进行移植，不容易被最先进的防御系统检测到，而且在输出转录方面有显著差异，但听起来与原始音频相似。



## **12. Staircase Sign Method for Boosting Adversarial Attacks**

一种加强对抗性攻击的阶梯标记法 cs.CV

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2104.09722v2)

**Authors**: Qilong Zhang, Xiaosu Zhu, Jingkuan Song, Lianli Gao, Heng Tao Shen

**Abstracts**: Crafting adversarial examples for the transfer-based attack is challenging and remains a research hot spot. Currently, such attack methods are based on the hypothesis that the substitute model and the victim model learn similar decision boundaries, and they conventionally apply Sign Method (SM) to manipulate the gradient as the resultant perturbation. Although SM is efficient, it only extracts the sign of gradient units but ignores their value difference, which inevitably leads to a deviation. Therefore, we propose a novel Staircase Sign Method (S$^2$M) to alleviate this issue, thus boosting attacks. Technically, our method heuristically divides the gradient sign into several segments according to the values of the gradient units, and then assigns each segment with a staircase weight for better crafting adversarial perturbation. As a result, our adversarial examples perform better in both white-box and black-box manner without being more visible. Since S$^2$M just manipulates the resultant gradient, our method can be generally integrated into the family of FGSM algorithms, and the computational overhead is negligible. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed methods, which significantly improve the transferability (i.e., on average, \textbf{5.1\%} for normally trained models and \textbf{12.8\%} for adversarially trained defenses). Our code is available at \url{https://github.com/qilong-zhang/Staircase-sign-method}.

摘要: 为基于转移的攻击制作敌意例子是具有挑战性的，也是一个研究热点。目前，这种攻击方法是基于替换模型和受害者模型学习相似的决策边界的假设，并且它们通常应用符号方法(SM)来操纵梯度作为结果扰动。虽然SM方法是有效的，但它只提取梯度单元的符号，而忽略了它们的值差异，这不可避免地会导致偏差。因此，我们提出了一种新的阶梯符号方法(S$^2$M)来缓解这个问题，从而增强了攻击。从技术上讲，我们的方法根据梯度单元的值启发式地将梯度符号分成几个段，然后为每个段分配阶梯权重，以便更好地制作对抗扰动。因此，我们的对抗性例子在白盒和黑盒方式下都表现得更好，而不是更明显。由于S$^2$M只是对合成的梯度进行操作，因此我们的方法一般可以集成到FGSM算法家族中，并且计算开销可以忽略不计。在ImageNet数据集上的大量实验表明，我们提出的方法是有效的，显著提高了可转移性(即，对于正常训练的模型，平均为extbf{5.1}，对于经过相反训练的防御，平均为\extbf{12.8})。我们的代码可以在\url{https://github.com/qilong-zhang/Staircase-sign-method}.上找到



## **13. A survey in Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御和稳健性研究综述 cs.CL

**SubmitDate**: 2022-04-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v2)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.

摘要: 近年来，人们已经看到，深度神经网络缺乏健壮性，在输入数据发生对抗性扰动的情况下很容易崩溃。强对抗性攻击是计算机视觉和自然语言处理领域的研究热点。作为应对措施，还提出了几种防御机制，以避免这些网络出现故障。与图像数据相比，由于文本数据的离散性，在自然语言处理中生成对抗性攻击并对这些模型进行防御并不容易。然而，最近针对文本分类、命名实体识别、自然语言推理等不同的NLP任务提出了许多对抗性防御方法，这些方法不仅用于保护神经网络免受对抗性攻击，而且在训练过程中作为一种正则化机制，避免了模型的过度拟合。这项拟议的调查试图通过提出一种新的分类法来回顾最近在NLP中提出的不同的对抗性防御方法。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及在保护它们方面的挑战。



## **14. Generalizing Adversarial Explanations with Grad-CAM**

基于Grad-CAM的对抗性解释泛化 cs.CV

Accepted in CVPRw ArtofRobustness workshop

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05427v1)

**Authors**: Tanmay Chakraborty, Utkarsh Trehan, Khawla Mallat, Jean-Luc Dugelay

**Abstracts**: Gradient-weighted Class Activation Mapping (Grad- CAM), is an example-based explanation method that provides a gradient activation heat map as an explanation for Convolution Neural Network (CNN) models. The drawback of this method is that it cannot be used to generalize CNN behaviour. In this paper, we present a novel method that extends Grad-CAM from example-based explanations to a method for explaining global model behaviour. This is achieved by introducing two new metrics, (i) Mean Observed Dissimilarity (MOD) and (ii) Variation in Dissimilarity (VID), for model generalization. These metrics are computed by comparing a Normalized Inverted Structural Similarity Index (NISSIM) metric of the Grad-CAM generated heatmap for samples from the original test set and samples from the adversarial test set. For our experiment, we study adversarial attacks on deep models such as VGG16, ResNet50, and ResNet101, and wide models such as InceptionNetv3 and XceptionNet using Fast Gradient Sign Method (FGSM). We then compute the metrics MOD and VID for the automatic face recognition (AFR) use case with the VGGFace2 dataset. We observe a consistent shift in the region highlighted in the Grad-CAM heatmap, reflecting its participation to the decision making, across all models under adversarial attacks. The proposed method can be used to understand adversarial attacks and explain the behaviour of black box CNN models for image analysis.

摘要: 梯度加权类激活映射(Grad-CAM)是一种基于实例的解释方法，它提供了一个梯度激活热图作为对卷积神经网络(CNN)模型的解释。这种方法的缺点是，它不能用来概括CNN的行为。在本文中，我们提出了一种新的方法，将Grad-CAM从基于实例的解释扩展为一种解释全局模型行为的方法。这是通过引入两个新的度量来实现的，(I)平均观察相异度(MOD)和(Ii)相异度变化(VID)，用于模型泛化。这些度量是通过比较Grad-CAM为原始测试集中的样本和对手测试集中的样本生成的热图的归一化反向结构相似性指数(Nissim)度量来计算的。在我们的实验中，我们使用快速梯度符号方法(FGSM)研究了对VGG16、ResNet50和ResNet101等深层模型以及InceptionNetv3和XceptionNet等宽模型的敌意攻击。然后，我们使用VGGFace2数据集计算自动人脸识别(AFR)用例的度量MOD和VID。我们观察到，在Grad-CAM热图中突出显示的区域出现了持续的变化，反映了其参与决策的情况，涵盖了所有受到敌对攻击的模型。该方法可用于理解对抗性攻击和解释用于图像分析的黑盒CNN模型的行为。



## **15. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05276v1)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **16. Exploring the Universal Vulnerability of Prompt-based Learning Paradigm**

探索基于提示的学习范式的普遍脆弱性 cs.CL

Accepted to Findings of NAACL 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05239v1)

**Authors**: Lei Xu, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Zhiyuan Liu

**Abstracts**: Prompt-based learning paradigm bridges the gap between pre-training and fine-tuning, and works effectively under the few-shot setting. However, we find that this learning paradigm inherits the vulnerability from the pre-training stage, where model predictions can be misled by inserting certain triggers into the text. In this paper, we explore this universal vulnerability by either injecting backdoor triggers or searching for adversarial triggers on pre-trained language models using only plain text. In both scenarios, we demonstrate that our triggers can totally control or severely decrease the performance of prompt-based models fine-tuned on arbitrary downstream tasks, reflecting the universal vulnerability of the prompt-based learning paradigm. Further experiments show that adversarial triggers have good transferability among language models. We also find conventional fine-tuning models are not vulnerable to adversarial triggers constructed from pre-trained language models. We conclude by proposing a potential solution to mitigate our attack methods. Code and data are publicly available at https://github.com/leix28/prompt-universal-vulnerability

摘要: 基于提示的学习范式在预训练和微调之间架起了一座桥梁，并在少数情况下有效地工作。然而，我们发现这种学习范式继承了预训练阶段的脆弱性，在预训练阶段，通过在文本中插入某些触发器可能会误导模型预测。在本文中，我们通过注入后门触发器或仅使用纯文本在预先训练的语言模型上搜索敌意触发器来探索这一普遍漏洞。在这两种情况下，我们的触发器可以完全控制或严重降低基于提示的模型对任意下游任务进行微调的性能，反映了基于提示的学习范式的普遍脆弱性。进一步的实验表明，对抗性触发词在语言模型之间具有良好的可移植性。我们还发现，传统的微调模型不容易受到从预先训练的语言模型构建的对抗性触发的影响。最后，我们提出了一个潜在的解决方案来减轻我们的攻击方法。代码和数据可在https://github.com/leix28/prompt-universal-vulnerability上公开获得



## **17. Analysis of a blockchain protocol based on LDPC codes**

一种基于LDPC码的区块链协议分析 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2202.07265v2)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check (LDPC) codes to counter DAAs. We show that the protocol is less secure than expected, owing to a redefinition of the adversarial success probability.

摘要: 在区块链数据可用性攻击(DAA)中，恶意节点发布块标头，但保留包含无效事务的部分块。可以下载并存储完整区块链的诚实全节点，知道有些数据不可用，但没有正式的方法向轻节点证明，即资源有限、无法访问整个区块链数据的节点。对抗这些攻击的常见解决方案使用线性纠错码来编码块内容。最近的一种称为SPAR的协议使用编码Merkle树和低密度奇偶校验(LDPC)码来对抗DAA。我们表明，由于重新定义了对抗性成功概率，该协议的安全性低于预期。



## **18. Measuring and Mitigating the Risk of IP Reuse on Public Clouds**

衡量和降低公共云上IP重用的风险 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05122v1)

**Authors**: Eric Pauley, Ryan Sheatsley, Blaine Hoak, Quinn Burke, Yohan Beugin, Patrick McDaniel

**Abstracts**: Public clouds provide scalable and cost-efficient computing through resource sharing. However, moving from traditional on-premises service management to clouds introduces new challenges; failure to correctly provision, maintain, or decommission elastic services can lead to functional failure and vulnerability to attack. In this paper, we explore a broad class of attacks on clouds which we refer to as cloud squatting. In a cloud squatting attack, an adversary allocates resources in the cloud (e.g., IP addresses) and thereafter leverages latent configuration to exploit prior tenants. To measure and categorize cloud squatting we deployed a custom Internet telescope within the Amazon Web Services us-east-1 region. Using this apparatus, we deployed over 3 million servers receiving 1.5 million unique IP addresses (56% of the available pool) over 101 days beginning in March of 2021. We identified 4 classes of cloud services, 7 classes of third-party services, and DNS as sources of exploitable latent configurations. We discovered that exploitable configurations were both common and in many cases extremely dangerous; we received over 5 million cloud messages, many containing sensitive data such as financial transactions, GPS location, and PII. Within the 7 classes of third-party services, we identified dozens of exploitable software systems spanning hundreds of servers (e.g., databases, caches, mobile applications, and web services). Lastly, we identified 5446 exploitable domains spanning 231 eTLDs-including 105 in the top 10,000 and 23 in the top 1000 popular domains. Through tenant disclosures we have identified several root causes, including (a) a lack of organizational controls, (b) poor service hygiene, and (c) failure to follow best practices. We conclude with a discussion of the space of possible mitigations and describe the mitigations to be deployed by Amazon in response to this study.

摘要: 公共云通过资源共享提供可扩展且经济高效的计算。然而，从传统的本地服务管理转移到云带来了新的挑战；未能正确调配、维护或停用弹性服务可能会导致功能故障和易受攻击。在本文中，我们探索了一大类针对云的攻击，我们称之为云蹲攻击。在云蹲守攻击中，对手在云中分配资源(例如，IP地址)，然后利用潜在配置来利用先前的租户。为了测量和分类云蹲点，我们在亚马逊网络服务US-East-1地区部署了一个定制的互联网望远镜。使用此设备，我们部署了300多万台服务器，从2021年3月开始，在101天内接收150万个唯一IP地址(占可用池的56%)。我们确定了4类云服务、7类第三方服务和DNS作为可利用的潜在配置来源。我们发现，可利用的配置很常见，而且在许多情况下极其危险；我们收到了500多万条云消息，其中许多包含金融交易、GPS位置和PII等敏感数据。在7类第三方服务中，我们确定了跨越数百台服务器(例如数据库、缓存、移动应用程序和Web服务)的数十个可利用的软件系统。最后，我们确定了覆盖231个eTLD的5446个可利用域名-其中105个在前10,000个域名中，23个在前1000个热门域名中。通过对租户的披露，我们确定了几个根本原因，包括(A)缺乏组织控制，(B)糟糕的服务卫生，以及(C)未能遵循最佳做法。最后，我们讨论了可能的缓解措施的空间，并描述了Amazon将针对这项研究部署的缓解措施。



## **19. Anti-Adversarially Manipulated Attributions for Weakly Supervised Semantic Segmentation and Object Localization**

用于弱监督语义分割和对象定位的反恶意操纵属性 cs.CV

IEEE TPAMI, 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.04890v1)

**Authors**: Jungbeom Lee, Eunji Kim, Jisoo Mok, Sungroh Yoon

**Abstracts**: Obtaining accurate pixel-level localization from class labels is a crucial process in weakly supervised semantic segmentation and object localization. Attribution maps from a trained classifier are widely used to provide pixel-level localization, but their focus tends to be restricted to a small discriminative region of the target object. An AdvCAM is an attribution map of an image that is manipulated to increase the classification score produced by a classifier before the final softmax or sigmoid layer. This manipulation is realized in an anti-adversarial manner, so that the original image is perturbed along pixel gradients in directions opposite to those used in an adversarial attack. This process enhances non-discriminative yet class-relevant features, which make an insufficient contribution to previous attribution maps, so that the resulting AdvCAM identifies more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and the excessive concentration of attributions on a small region of the target object. Our method achieves a new state-of-the-art performance in weakly and semi-supervised semantic segmentation, on both the PASCAL VOC 2012 and MS COCO 2014 datasets. In weakly supervised object localization, it achieves a new state-of-the-art performance on the CUB-200-2011 and ImageNet-1K datasets.

摘要: 从类标签中获得准确的像素级定位是弱监督语义分割和目标定位中的关键步骤。来自训练好的分类器的属性图被广泛用于提供像素级定位，但它们的焦点往往被限制在目标对象的一个小的区分区域。AdvCAM是图像的属性图，其被处理以在最终的Softmax或Sigmoid层之前增加由分类器产生的分类分数。这种操作是以反对抗性的方式实现的，使得原始图像沿着与对抗性攻击中使用的方向相反的像素梯度被扰动。该过程增强了对先前属性图贡献不足的非歧视但与类相关的特征，从而所产生的AdvCAM识别目标对象的更多区域。此外，我们引入了一种新的正则化过程，该过程抑制了与目标对象无关的区域的错误归属以及目标对象的小区域属性的过度集中。在PASCAL VOC 2012和MS Coco 2014数据集上，我们的方法在弱监督和半监督语义分割方面取得了新的最先进的性能。在弱监督目标定位方面，它在CUB-200-2011和ImageNet-1K数据集上取得了最新的性能。



## **20. Adversarial Robustness of Deep Sensor Fusion Models**

深度传感器融合模型的对抗稳健性 cs.CV

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2006.13192v3)

**Authors**: Shaojie Wang, Tong Wu, Ayan Chakrabarti, Yevgeniy Vorobeychik

**Abstracts**: We experimentally study the robustness of deep camera-LiDAR fusion architectures for 2D object detection in autonomous driving. First, we find that the fusion model is usually both more accurate, and more robust against single-source attacks than single-sensor deep neural networks. Furthermore, we show that without adversarial training, early fusion is more robust than late fusion, whereas the two perform similarly after adversarial training. However, we note that single-channel adversarial training of deep fusion is often detrimental even to robustness. Moreover, we observe cross-channel externalities, where single-channel adversarial training reduces robustness to attacks on the other channel. Additionally, we observe that the choice of adversarial model in adversarial training is critical: using attacks restricted to cars' bounding boxes is more effective in adversarial training and exhibits less significant cross-channel externalities. Finally, we find that joint-channel adversarial training helps mitigate many of the issues above, but does not significantly boost adversarial robustness.

摘要: 实验研究了深度摄像机-LiDAR融合结构在自动驾驶中检测2D目标的稳健性。首先，我们发现融合模型通常比单传感器深度神经网络更准确，并且对单源攻击具有更强的鲁棒性。此外，我们还表明，在没有对抗性训练的情况下，早期融合比后期融合更稳健，而在对抗性训练后，两者的表现相似。然而，我们注意到，深度融合的单通道对抗性训练往往甚至对健壮性有害。此外，我们观察到了跨通道外部性，其中单通道对抗性训练降低了对另一通道攻击的稳健性。此外，我们观察到，在对抗性训练中选择对抗性模型是至关重要的：在对抗性训练中，使用仅限于汽车包围盒的攻击更有效，并且表现出较少的跨通道外部性。最后，我们发现联合通道对抗性训练有助于缓解上述许多问题，但并不显著提高对抗性健壮性。



## **21. Measuring the False Sense of Security**

测量虚假的安全感 cs.LG

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04778v1)

**Authors**: Carlos Gomes

**Abstracts**: Recently, several papers have demonstrated how widespread gradient masking is amongst proposed adversarial defenses. Defenses that rely on this phenomenon are considered failed, and can easily be broken. Despite this, there has been little investigation into ways of measuring the phenomenon of gradient masking and enabling comparisons of its extent amongst different networks. In this work, we investigate gradient masking under the lens of its mensurability, departing from the idea that it is a binary phenomenon. We propose and motivate several metrics for it, performing extensive empirical tests on defenses suspected of exhibiting different degrees of gradient masking. These are computationally cheaper than strong attacks, enable comparisons between models, and do not require the large time investment of tailor-made attacks for specific models. Our results reveal metrics that are successful in measuring the extent of gradient masking across different networks

摘要: 最近，几篇论文已经证明了梯度掩蔽在所提出的对抗防御中是如何广泛存在的。依赖这种现象的防御被认为是失败的，很容易被打破。尽管如此，关于如何测量梯度掩蔽现象并能够在不同网络之间比较其程度的研究很少。在这项工作中，我们从梯度掩蔽是一种二元现象的观点出发，研究了它的可测性透镜下的梯度掩蔽。我们为它提出并激励了几个衡量标准，对被怀疑表现出不同程度的梯度掩蔽的防御进行了广泛的经验测试。这些攻击在计算上比强攻击更便宜，可以在模型之间进行比较，并且不需要为特定模型定制攻击的大量时间投资。我们的结果揭示了成功测量不同网络之间的梯度掩蔽程度的度量标准



## **22. Analysis of Power-Oriented Fault Injection Attacks on Spiking Neural Networks**

尖峰神经网络面向能量的故障注入攻击分析 cs.AI

Design, Automation and Test in Europe Conference (DATE) 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04768v1)

**Authors**: Karthikeyan Nagarajan, Junde Li, Sina Sayyah Ensan, Mohammad Nasim Imtiaz Khan, Sachhidh Kannan, Swaroop Ghosh

**Abstracts**: Spiking Neural Networks (SNN) are quickly gaining traction as a viable alternative to Deep Neural Networks (DNN). In comparison to DNNs, SNNs are more computationally powerful and provide superior energy efficiency. SNNs, while exciting at first appearance, contain security-sensitive assets (e.g., neuron threshold voltage) and vulnerabilities (e.g., sensitivity of classification accuracy to neuron threshold voltage change) that adversaries can exploit. We investigate global fault injection attacks by employing external power supplies and laser-induced local power glitches to corrupt crucial training parameters such as spike amplitude and neuron's membrane threshold potential on SNNs developed using common analog neurons. We also evaluate the impact of power-based attacks on individual SNN layers for 0% (i.e., no attack) to 100% (i.e., whole layer under attack). We investigate the impact of the attacks on digit classification tasks and find that in the worst-case scenario, classification accuracy is reduced by 85.65%. We also propose defenses e.g., a robust current driver design that is immune to power-oriented attacks, improved circuit sizing of neuron components to reduce/recover the adversarial accuracy degradation at the cost of negligible area and 25% power overhead. We also present a dummy neuron-based voltage fault injection detection system with 1% power and area overhead.

摘要: 尖峰神经网络(SNN)作为深度神经网络(DNN)的一种可行的替代方案正在迅速获得发展。与DNN相比，SNN的计算能力更强，并提供更高的能源效率。SNN虽然乍看上去令人兴奋，但包含对安全敏感的资产(例如，神经元阈值电压)和漏洞(例如，分类精度对神经元阈值电压变化的敏感性)，攻击者可以利用这些漏洞。我们通过使用外部电源和激光诱导的局部功率毛刺来破坏使用普通模拟神经元开发的SNN上的关键训练参数，如棘波幅度和神经元的膜阈值电位，来调查全局故障注入攻击。我们还评估了基于能量的攻击对单个SNN层的影响，从0%(即没有攻击)到100%(即整个层受到攻击)。我们研究了攻击对数字分类任务的影响，发现在最坏的情况下，分类准确率下降了85.65%。我们还提出了防御措施，例如，稳健的电流驱动器设计，它不受面向功率的攻击，改进了神经元组件的电路大小，以可忽略的面积和25%的功率开销为代价来减少/恢复对抗性精度的下降。我们还提出了一个基于虚拟神经元的电压故障注入检测系统，该系统具有1%的功率和面积开销。



## **23. "That Is a Suspicious Reaction!": Interpreting Logits Variation to Detect NLP Adversarial Attacks**

“这是一个可疑的反应！”：解读Logits变量以检测NLP对手攻击 cs.AI

ACL 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04636v1)

**Authors**: Edoardo Mosca, Shreyash Agarwal, Javier Rando-Ramirez, Georg Groh

**Abstracts**: Adversarial attacks are a major challenge faced by current machine learning research. These purposely crafted inputs fool even the most advanced models, precluding their deployment in safety-critical applications. Extensive research in computer vision has been carried to develop reliable defense strategies. However, the same issue remains less explored in natural language processing. Our work presents a model-agnostic detector of adversarial text examples. The approach identifies patterns in the logits of the target classifier when perturbing the input text. The proposed detector improves the current state-of-the-art performance in recognizing adversarial inputs and exhibits strong generalization capabilities across different NLP models, datasets, and word-level attacks.

摘要: 对抗性攻击是当前机器学习研究面临的一大挑战。这些刻意制作的输入甚至欺骗了最先进的型号，使它们无法部署在安全关键应用程序中。为了制定可靠的防御策略，人们在计算机视觉方面进行了广泛的研究。然而，在自然语言处理中，同样的问题仍然被较少地探讨。我们的工作提出了一个模型不可知的对抗性文本例子的检测器。该方法在干扰输入文本时识别目标分类器的逻辑中的模式。提出的检测器提高了当前在识别敌意输入方面的最新性能，并在不同的NLP模型、数据集和词级攻击中显示出强大的泛化能力。



## **24. LTD: Low Temperature Distillation for Robust Adversarial Training**

LTD：低温蒸馏用于强大的对抗性训练 cs.CV

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2111.02331v2)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstracts**: Adversarial training has been widely used to enhance the robustness of the neural network models against adversarial attacks. However, there still a notable gap between the nature accuracy and the robust accuracy. We found one of the reasons is the commonly used labels, one-hot vectors, hinder the learning process for image recognition. In this paper, we proposed a method, called Low Temperature Distillation (LTD), which is based on the knowledge distillation framework to generate the desired soft labels. Unlike the previous work, LTD uses relatively low temperature in the teacher model, and employs different, but fixed, temperatures for the teacher model and the student model. Moreover, we have investigated the methods to synergize the use of nature data and adversarial ones in LTD. Experimental results show that without extra unlabeled data, the proposed method combined with the previous work can achieve 57.72\% and 30.36\% robust accuracy on CIFAR-10 and CIFAR-100 dataset respectively, which is about 1.21\% improvement of the state-of-the-art methods in average.

摘要: 对抗性训练已被广泛应用于增强神经网络模型对对抗性攻击的鲁棒性。然而，自然精度与稳健精度之间仍存在着显著的差距。我们发现，其中一个原因是常用的标签，一个热点向量，阻碍了图像识别的学习过程。本文提出了一种基于知识蒸馏框架来生成所需软标签的方法，称为低温蒸馏(LTD)。与以前的工作不同，LTD在教师模型中使用相对较低的温度，并为教师模型和学生模型使用不同但固定的温度。此外，我们还研究了在LTD协同使用自然数据和对抗性数据的方法。实验结果表明，在不增加额外未标注数据的情况下，该方法在CIFAR-10和CIFAR-100数据集上分别达到了57.72和30.36的稳健准确率，平均比现有方法提高了1.21倍。



## **25. Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification**

文本分类中非分布样本和敌意样本的理解、检测和分离 cs.CL

Preprint. Work in progress

**SubmitDate**: 2022-04-09    [paper-pdf](http://arxiv.org/pdf/2204.04458v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstracts**: In this paper, we study the differences and commonalities between statistically out-of-distribution (OOD) samples and adversarial (Adv) samples, both of which hurting a text classification model's performance. We conduct analyses to compare the two types of anomalies (OOD and Adv samples) with the in-distribution (ID) ones from three aspects: the input features, the hidden representations in each layer of the model, and the output probability distributions of the classifier. We find that OOD samples expose their aberration starting from the first layer, while the abnormalities of Adv samples do not emerge until the deeper layers of the model. We also illustrate that the models' output probabilities for Adv samples tend to be more unconfident. Based on our observations, we propose a simple method to separate ID, OOD, and Adv samples using the hidden representations and output probabilities of the model. On multiple combinations of ID, OOD datasets, and Adv attacks, our proposed method shows exceptional results on distinguishing ID, OOD, and Adv samples.

摘要: 本文研究了统计分布(OOD)样本和对抗性(ADV)样本之间的差异和共同点，这两种样本都影响了文本分类模型的性能。我们从输入特征、模型每一层的隐含表示和分类器的输出概率分布三个方面对两类异常(OOD和ADV样本)和非分布异常(ID)进行了分析比较。我们发现，OOD样本从第一层开始暴露出它们的异常，而ADV样本的异常直到模型的更深层才出现。我们还说明，对于ADV样本，模型的输出概率往往更不可信。基于我们的观察，我们提出了一种简单的方法，利用模型的隐含表示和输出概率来分离ID、OOD和ADV样本。在ID、OOD数据集和ADV攻击的多种组合上，我们提出的方法在区分ID、OOD和ADV样本方面表现出了出色的结果。



## **26. PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier**

PatchCleanser：针对任何图像分类器的恶意补丁的可靠防御 cs.CV

USENIX Security Symposium 2022; extended technical report

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2108.09135v2)

**Authors**: Chong Xiang, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: The adversarial patch attack against image classification models aims to inject adversarially crafted pixels within a restricted image region (i.e., a patch) for inducing model misclassification. This attack can be realized in the physical world by printing and attaching the patch to the victim object; thus, it imposes a real-world threat to computer vision systems. To counter this threat, we design PatchCleanser as a certifiably robust defense against adversarial patches. In PatchCleanser, we perform two rounds of pixel masking on the input image to neutralize the effect of the adversarial patch. This image-space operation makes PatchCleanser compatible with any state-of-the-art image classifier for achieving high accuracy. Furthermore, we can prove that PatchCleanser will always predict the correct class labels on certain images against any adaptive white-box attacker within our threat model, achieving certified robustness. We extensively evaluate PatchCleanser on the ImageNet, ImageNette, CIFAR-10, CIFAR-100, SVHN, and Flowers-102 datasets and demonstrate that our defense achieves similar clean accuracy as state-of-the-art classification models and also significantly improves certified robustness from prior works. Remarkably, PatchCleanser achieves 83.9% top-1 clean accuracy and 62.1% top-1 certified robust accuracy against a 2%-pixel square patch anywhere on the image for the 1000-class ImageNet dataset.

摘要: 针对图像分类模型的对抗性补丁攻击的目的是在受限的图像区域(即补丁)内注入恶意创建的像素，以导致模型误分类。这种攻击可以通过将补丁打印并附加到受害者对象上在物理世界中实现；因此，它对计算机视觉系统构成了现实世界的威胁。为了应对这种威胁，我们将PatchCleanser设计为针对恶意补丁的可靠可靠防御。在PatchCleanser中，我们对输入图像执行两轮像素掩蔽，以中和对手补丁的影响。这种图像空间操作使PatchCleanser与任何最先进的图像分类器兼容，以实现高精度。此外，我们可以证明PatchCleanser将始终预测特定图像上的正确类别标签，以对抗我们威胁模型中的任何自适应白盒攻击者，从而实现经过验证的健壮性。我们在ImageNet、ImageNette、CIFAR-10、CIFAR-100、SVHN和Flowers-102数据集上对PatchCleanser进行了广泛的评估，并展示了我们的防御实现了与最先进的分类模型类似的干净准确性，并显著提高了先前工作中经过认证的稳健性。值得注意的是，对于1000级ImageNet数据集，PatchCleanser针对图像上任何位置2%像素的正方形补丁实现了83.9%的TOP-1清洁准确率和62.1%的TOP-1认证的稳健准确率。



## **27. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

有限信息动态防御者-攻击者Blotto博弈(DDAB)中的路径防御 cs.GT

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.04176v1)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstracts**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.

摘要: 我们考虑了动态防御者-攻击者Blotto博弈(DDAB)中的路径保护问题，其中一组机器人必须防御图中的一条路径以对抗对手代理。多机器人系统特别适合这一应用，因为最近的研究表明，这些系统在周边防御和监视等相关领域是有效的。在设计保证路径防御的防御方策略时，有关对手和环境的信息可能会有所帮助，并且可以减少防御方实现足够安全级别所需的资源数量。在这项工作中，我们刻画了当防御者只能检测到最短路径$k$-跳内的资产时，保证dDAB博弈中两个节点之间最短路径的防御所需的必要且足够数量的资产。通过描述感知范围和所需资源之间的关系，我们表明，增加防御者的感知能力可以极大地减少防御路径所需的防御者资产的数量。



## **28. DAD: Data-free Adversarial Defense at Test Time**

DAD：测试时的无数据对抗性防御 cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.01568v2)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.

摘要: 深度模型非常容易受到对抗性攻击。这类攻击是精心设计的难以察觉的噪音，可以愚弄网络，在部署时可能会造成严重后果。为了遇到它们，该模型需要用于对抗性训练的训练数据或明确的基于正则化的技术。然而，隐私已经成为一个重要的问题，只限制对训练模型的访问，而不限制对训练数据(例如生物特征数据)的访问。此外，数据管理成本高昂，公司可能对其拥有专有权。为了处理这种情况，我们提出了一个全新的问题，即在没有训练数据甚至其统计数据的情况下进行测试时间对抗性防御。我们分两个阶段来解决这个问题：a)对手样本的检测和b)对手样本的校正。我们的对抗性样本检测框架首先在任意数据上进行训练，然后通过无监督的领域自适应来适应未标记的测试数据。通过对检测到的敌意样本进行傅立叶变换，并在我们提出的适合模型预测的半径处获得它们的低频分量，进一步修正了预测。我们通过针对几种对抗性攻击以及针对不同模型架构和数据集的广泛实验，证明了我们所提出的技术的有效性。对于在CIFAR-10上预先训练的非健壮RESNET-18模型，我们的检测方法正确识别了91.42%的对手。此外，在不需要重新训练模型的情况下，我们显著地将对手准确率从0%提高到37.37%，而对最先进的自动攻击的干净准确率最小下降了0.02%。



## **29. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2203.17031v3)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **30. Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures**

旋转的语言模型：宣传即服务的风险和对策 cs.CR

IEEE S&P 2022. arXiv admin note: text overlap with arXiv:2107.10443

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2112.05224v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstracts**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their outputs so as to support an adversary-chosen sentiment or point of view -- but only when the input contains adversary-chosen trigger words. For example, a spinned summarization model outputs positive summaries of any text that mentions the name of some individual or organization.   Model spinning introduces a "meta-backdoor" into a model. Whereas conventional backdoors cause models to produce incorrect outputs on inputs with the trigger, outputs of spinned models preserve context and maintain standard accuracy metrics, yet also satisfy a meta-task chosen by the adversary.   Model spinning enables propaganda-as-a-service, where propaganda is defined as biased speech. An adversary can create customized language models that produce desired spins for chosen triggers, then deploy these models to generate disinformation (a platform attack), or else inject them into ML training pipelines (a supply-chain attack), transferring malicious functionality to downstream models trained by victims.   To demonstrate the feasibility of model spinning, we develop a new backdooring technique. It stacks an adversarial meta-task onto a seq2seq model, backpropagates the desired meta-task output to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. We evaluate this attack on language generation, summarization, and translation models with different triggers and meta-tasks such as sentiment, toxicity, and entailment. Spinned models largely maintain their accuracy metrics (ROUGE and BLEU) while shifting their outputs to satisfy the adversary's meta-task. We also show that, in the case of a supply-chain attack, the spin functionality transfers to downstream models.

摘要: 我们调查了对神经序列到序列(Seq2seq)模型的一种新威胁：训练时间攻击，该攻击使模型的输出“旋转”以支持对手选择的情绪或观点--但仅当输入包含对手选择的触发词时。例如，旋转摘要模型输出提及某个个人或组织名称的任何文本的正面摘要。模型旋转在模型中引入了“元后门”。传统的后门会导致模型在带有触发器的输入上产生不正确的输出，而旋转模型的输出保留了上下文并保持了标准的准确性度量，但也满足了对手选择的元任务。模型旋转使宣传成为一种服务，其中宣传被定义为有偏见的言论。对手可以创建自定义语言模型，为选定的触发器生成所需的旋转，然后部署这些模型以生成虚假信息(平台攻击)，或者将它们注入ML训练管道(供应链攻击)，将恶意功能转移到受害者训练的下游模型。为了论证模型旋转的可行性，我们开发了一种新的回溯技术。它将一个对抗性元任务堆叠到seq2seq模型上，将所需的元任务输出反向传播到我们称为“伪词”的单词嵌入空间中的点，并使用伪词来移动seq2seq模型的整个输出分布。我们用不同的触发因素和元任务，如情感、毒性和蕴涵来评估这种对语言生成、摘要和翻译模型的攻击。旋转模型在很大程度上保持了它们的精度指标(Rouge和BLEU)，同时改变了它们的输出以满足对手的元任务。我们还表明，在供应链攻击的情况下，自旋功能转移到下游模型。



## **31. Backdoor Attack against NLP models with Robustness-Aware Perturbation defense**

对具有鲁棒性感知扰动防御的NLP模型的后门攻击 cs.CR

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.05758v1)

**Authors**: Shaik Mohammed Maqsood, Viveros Manuela Ceron, Addluri GowthamKrishna

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), such that the attacked model performs well on benign samples, whereas its prediction will be maliciously changed if the hidden backdoor is activated by the attacker defined trigger. This threat could happen when the training process is not fully controlled, such as training on third-party data-sets or adopting third-party models. There has been a lot of research and different methods to defend such type of backdoor attacks, one being robustness-aware perturbation-based defense method. This method mainly exploits big gap of robustness between poisoned and clean samples. In our work, we break this defense by controlling the robustness gap between poisoned and clean samples using adversarial training step.

摘要: 后门攻击的目的是将隐藏的后门嵌入到深度神经网络(DNN)中，使得攻击模型在良性样本上表现良好，而如果隐藏的后门被攻击者定义的触发器激活，则其预测将被恶意更改。当培训过程没有得到完全控制时，例如在第三方数据集上进行培训或采用第三方模型时，可能会出现这种威胁。已经有很多研究和不同的方法来防御这种类型的后门攻击，其中一种是基于健壮性感知扰动的防御方法。该方法主要利用了有毒样本和干净样本之间存在的较大的稳健性差距。在我们的工作中，我们通过使用对抗性训练步骤来控制有毒样本和干净样本之间的稳健性差距，从而打破了这种防御。



## **32. Defense against Adversarial Attacks on Hybrid Speech Recognition using Joint Adversarial Fine-tuning with Denoiser**

基于联合对抗性微调和去噪的混合语音识别抗敌意攻击 eess.AS

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03851v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Yiwen Shao, Piotr Zelasko, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak

**Abstracts**: Adversarial attacks are a threat to automatic speech recognition (ASR) systems, and it becomes imperative to propose defenses to protect them. In this paper, we perform experiments to show that K2 conformer hybrid ASR is strongly affected by white-box adversarial attacks. We propose three defenses--denoiser pre-processor, adversarially fine-tuning ASR model, and adversarially fine-tuning joint model of ASR and denoiser. Our evaluation shows denoiser pre-processor (trained on offline adversarial examples) fails to defend against adaptive white-box attacks. However, adversarially fine-tuning the denoiser using a tandem model of denoiser and ASR offers more robustness. We evaluate two variants of this defense--one updating parameters of both models and the second keeping ASR frozen. The joint model offers a mean absolute decrease of 19.3\% ground truth (GT) WER with reference to baseline against fast gradient sign method (FGSM) attacks with different $L_\infty$ norms. The joint model with frozen ASR parameters gives the best defense against projected gradient descent (PGD) with 7 iterations, yielding a mean absolute increase of 22.3\% GT WER with reference to baseline; and against PGD with 500 iterations, yielding a mean absolute decrease of 45.08\% GT WER and an increase of 68.05\% adversarial target WER.

摘要: 敌意攻击是对自动语音识别(ASR)系统的一种威胁，提出防御措施势在必行。在本文中，我们通过实验证明K2一致性混合ASR受到白盒对抗攻击的强烈影响。我们提出了三个防御措施--去噪预处理器、反向微调ASR模型、反向微调ASR和去噪联合模型。我们的评估表明，去噪预处理器(针对离线对手示例进行训练)无法防御自适应白盒攻击。然而，相反地，使用去噪器和ASR的串联模型来微调去噪器可提供更强的稳健性。我们评估了这种防御的两种变体--一种是更新两个模型的参数，另一种是保持ASR不变。该联合模型对于具有不同$L_inty$范数的快速梯度符号法(FGSM)攻击，相对于基线平均绝对减少了19.3%的地面真实(GT)WER。采用冻结ASR参数的联合模型对7次迭代的投影梯度下降(PGD)提供了最好的防御，相对于基线平均绝对增加了2 2.3 GT WER，对5 0 0次的PGD给出了最好的防御，平均绝对减少4 5.0 8 GT WER，增加了68.0 5个目标WER。



## **33. AdvEst: Adversarial Perturbation Estimation to Classify and Detect Adversarial Attacks against Speaker Identification**

AdvEst：对抗性扰动估计分类检测针对说话人识别的对抗性攻击 eess.AS

Submitted to InterSpeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03848v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Jesus Villalba, Najim Dehak

**Abstracts**: Adversarial attacks pose a severe security threat to the state-of-the-art speaker identification systems, thereby making it vital to propose countermeasures against them. Building on our previous work that used representation learning to classify and detect adversarial attacks, we propose an improvement to it using AdvEst, a method to estimate adversarial perturbation. First, we prove our claim that training the representation learning network using adversarial perturbations as opposed to adversarial examples (consisting of the combination of clean signal and adversarial perturbation) is beneficial because it eliminates nuisance information. At inference time, we use a time-domain denoiser to estimate the adversarial perturbations from adversarial examples. Using our improved representation learning approach to obtain attack embeddings (signatures), we evaluate their performance for three applications: known attack classification, attack verification, and unknown attack detection. We show that common attacks in the literature (Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Carlini-Wagner (CW) with different Lp threat models) can be classified with an accuracy of ~96%. We also detect unknown attacks with an equal error rate (EER) of ~9%, which is absolute improvement of ~12% from our previous work.

摘要: 对抗性攻击对最先进的说话人识别系统构成了严重的安全威胁，因此提出针对它们的对策是至关重要的。在利用表征学习对敌方攻击进行分类和检测的基础上，我们提出了一种基于AdvEst的改进方法，该方法是一种估计对抗性扰动的方法。首先，我们证明了我们的主张，即使用对抗性扰动而不是对抗性示例(由干净的信号和对抗性扰动的组合组成)来训练表示学习网络是有益的，因为它消除了滋扰信息。在推理时，我们使用一个时间域去噪器来估计对抗性样本中的对抗性扰动。使用改进的表示学习方法获得攻击嵌入(签名)，我们评估了它们在三个应用中的性能：已知攻击分类、攻击验证和未知攻击检测。我们表明，文献中常见的攻击(快速梯度符号法(FGSM)、投影梯度下降法(PGD)、Carlini-Wagner(CW)和不同的LP威胁模型)可以被分类，准确率为96%。我们还检测未知攻击，等错误率(EER)为~9%，比我们以前的工作绝对提高了~12%。



## **34. Using Multiple Self-Supervised Tasks Improves Model Robustness**

使用多个自监督任务可提高模型的稳健性 cs.CV

Accepted to ICLR 2022 Workshop on PAIR^2Struct: Privacy,  Accountability, Interpretability, Robustness, Reasoning on Structured Data

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03714v1)

**Authors**: Matthew Lawhon, Chengzhi Mao, Junfeng Yang

**Abstracts**: Deep networks achieve state-of-the-art performance on computer vision tasks, yet they fail under adversarial attacks that are imperceptible to humans. In this paper, we propose a novel defense that can dynamically adapt the input using the intrinsic structure from multiple self-supervised tasks. By simultaneously using many self-supervised tasks, our defense avoids over-fitting the adapted image to one specific self-supervised task and restores more intrinsic structure in the image compared to a single self-supervised task approach. Our approach further improves robustness and clean accuracy significantly compared to the state-of-the-art single task self-supervised defense. Our work is the first to connect multiple self-supervised tasks to robustness, and suggests that we can achieve better robustness with more intrinsic signal from visual data.

摘要: 深度网络在计算机视觉任务中实现了最先进的性能，但它们在人类无法察觉的敌意攻击下失败了。在本文中，我们提出了一种新的防御机制，它可以利用多个自监督任务的内在结构来动态调整输入。通过同时使用多个自监督任务，我们的防御方法避免了将适应的图像过度匹配到一个特定的自监督任务，并且与单一的自监督任务方法相比，恢复了图像中更多的内在结构。与最先进的单任务自我监督防御相比，我们的方法进一步提高了健壮性和干净的准确性。我们的工作首次将多个自监督任务与稳健性联系起来，并表明我们可以通过从视觉数据中获得更多的内在信号来获得更好的稳健性。



## **35. Adaptive-Gravity: A Defense Against Adversarial Samples**

自适应重力：对抗对手样本的一种防御 cs.LG

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03694v1)

**Authors**: Ali Mirzaeian, Zhi Tian, Sai Manoj P D, Banafsheh S. Latibari, Ioannis Savidis, Houman Homayoun, Avesta Sasan

**Abstracts**: This paper presents a novel model training solution, denoted as Adaptive-Gravity, for enhancing the robustness of deep neural network classifiers against adversarial examples. We conceptualize the model parameters/features associated with each class as a mass characterized by its centroid location and the spread (standard deviation of the distance) of features around the centroid. We use the centroid associated with each cluster to derive an anti-gravity force that pushes the centroids of different classes away from one another during network training. Then we customized an objective function that aims to concentrate each class's features toward their corresponding new centroid, which has been obtained by anti-gravity force. This methodology results in a larger separation between different masses and reduces the spread of features around each centroid. As a result, the samples are pushed away from the space that adversarial examples could be mapped to, effectively increasing the degree of perturbation needed for making an adversarial example. We have implemented this training solution as an iterative method consisting of four steps at each iteration: 1) centroid extraction, 2) anti-gravity force calculation, 3) centroid relocation, and 4) gravity training. Gravity's efficiency is evaluated by measuring the corresponding fooling rates against various attack models, including FGSM, MIM, BIM, and PGD using LeNet and ResNet110 networks, benchmarked against MNIST and CIFAR10 classification problems. Test results show that Gravity not only functions as a powerful instrument to robustify a model against state-of-the-art adversarial attacks but also effectively improves the model training accuracy.

摘要: 提出了一种新的模型训练方法--自适应重力法，以增强深度神经网络分类器对敌意样本的鲁棒性。我们将与每一类相关联的模型参数/特征概念化为质量，其特征由其质心位置和特征围绕质心的扩散(距离的标准差)来表征。我们使用与每个簇相关联的质心来推导出在网络训练期间将不同类别的质心彼此推开的反重力。然后，我们定制了一个目标函数，目标是将每一类的特征集中到它们对应的新质心上，该质心是通过反重力获得的。这种方法导致不同质量之间的更大分离，并减少了特征在每个质心周围的扩散。结果，样本被推离对抗性示例可以映射到的空间，有效地增加了制作对抗性示例所需的扰动程度。我们将这个训练方案实现为迭代方法，每次迭代包括四个步骤：1)质心提取，2)反重力计算，3)质心重定位，4)重力训练。通过使用LeNet和ResNet110网络测量针对各种攻击模型(包括FGSM、MIM、BIM和PGD)的相应愚骗率，并以MNIST和CIFAR10分类问题为基准，来评估Graight的效率。测试结果表明，该方法不仅可以有效地提高模型的训练精度，而且可以有效地提高模型的训练精度。



## **36. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

线性二次控制的强化学习在成本操纵下的脆弱性 eess.SY

This paper is yet to be peer-reviewed; Typos are corrected in ver 2

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2203.05774v2)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification of the cost parameters will only lead to a bounded change in the optimal policy. The bound is linear on the amount of falsification the attacker can apply to the cost parameters. We propose an attack model where the attacker aims to mislead the agent into learning a `nefarious' policy by intentionally falsifying the cost parameters. We formulate the attack's problem as a convex optimization problem and develop necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the actual cost signal. The paper aims to raise people's awareness of the security threats faced by RL-enabled control systems.

摘要: 在这项工作中，我们通过操纵费用信号来研究线性二次高斯(LQG)代理的欺骗。我们表明，对成本参数的微小篡改只会导致最优策略的有限变化。该界限与攻击者可以应用于成本参数的伪造量呈线性关系。我们提出了一个攻击模型，其中攻击者旨在通过故意伪造成本参数来误导代理学习“邪恶的”策略。我们将攻击问题描述为一个凸优化问题，并给出了检验攻击者目标可达性的充要条件。我们展示了在两种类型的LQG学习器上的对抗操作：批处理RL学习器和自适应动态规划(ADP)学习器。我们的结果表明，在只有2.296%的成本数据被篡改的情况下，攻击者误导批次RL学习将车辆引向危险位置的“邪恶”策略。攻击者还可以通过始终如一地向学习者提供接近实际成本信号的伪造成本信号，逐渐诱骗ADP学习者学习相同的“邪恶”策略。本文旨在提高人们对启用RL的控制系统所面临的安全威胁的认识。



## **37. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线侦听的对策 cs.CR

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2112.01967v2)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Markus Heinrichs, Rainer Kronberger, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线电信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，如今无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过窃听标准通信信号，窃听者获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为一种新的对抗敌意无线传感的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行混淆。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **38. Transfer Attacks Revisited: A Large-Scale Empirical Study in Real Computer Vision Settings**

传输攻击重现：真实计算机视觉环境下的大规模实证研究 cs.CV

Accepted to IEEE Security & Privacy 2022

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.04063v1)

**Authors**: Yuhao Mao, Chong Fu, Saizhuo Wang, Shouling Ji, Xuhong Zhang, Zhenguang Liu, Jun Zhou, Alex X. Liu, Raheem Beyah, Ting Wang

**Abstracts**: One intriguing property of adversarial attacks is their "transferability" -- an adversarial example crafted with respect to one deep neural network (DNN) model is often found effective against other DNNs as well. Intensive research has been conducted on this phenomenon under simplistic controlled conditions. Yet, thus far, there is still a lack of comprehensive understanding about transferability-based attacks ("transfer attacks") in real-world environments.   To bridge this critical gap, we conduct the first large-scale systematic empirical study of transfer attacks against major cloud-based MLaaS platforms, taking the components of a real transfer attack into account. The study leads to a number of interesting findings which are inconsistent to the existing ones, including: (1) Simple surrogates do not necessarily improve real transfer attacks. (2) No dominant surrogate architecture is found in real transfer attacks. (3) It is the gap between posterior (output of the softmax layer) rather than the gap between logit (so-called $\kappa$ value) that increases transferability. Moreover, by comparing with prior works, we demonstrate that transfer attacks possess many previously unknown properties in real-world environments, such as (1) Model similarity is not a well-defined concept. (2) $L_2$ norm of perturbation can generate high transferability without usage of gradient and is a more powerful source than $L_\infty$ norm. We believe this work sheds light on the vulnerabilities of popular MLaaS platforms and points to a few promising research directions.

摘要: 对抗性攻击的一个耐人寻味的特性是它们的“可转移性”--针对一个深度神经网络(DNN)模型制作的对抗性示例通常也被发现对其他DNN有效。人们在简单化的控制条件下对这一现象进行了深入的研究。然而，到目前为止，对现实环境中基于可转移性的攻击(“传输攻击”)仍然缺乏全面的了解。为了弥补这一关键差距，我们首次进行了针对主要基于云的MLaaS平台的传输攻击的大规模系统实证研究，考虑了真实传输攻击的组件。这项研究导致了一些有趣的发现，这些发现与现有的发现不一致，包括：(1)简单的代理并不一定能改善真实的转移攻击。(2)在真实传输攻击中没有发现具有优势的代理体系结构。(3)提高可转移性的是后验之间的差距(Softmax层的输出)，而不是Logit之间的差距(所谓的$\kappa$值)。此外，通过与已有工作的比较，我们证明了传输攻击在现实环境中具有许多以前未知的性质，例如：(1)模型相似性不是一个定义良好的概念。(2)摄动的$L_2$范数可以在不使用梯度的情况下产生很高的可转移性，是一个比$L_inty$范数更强大的来源。我们相信，这项工作揭示了流行的MLaaS平台的漏洞，并指出了一些有前途的研究方向。



## **39. Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems**

针对视频异常检测系统的对抗性机器学习攻击 cs.CV

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03141v1)

**Authors**: Furkan Mumcu, Keval Doshi, Yasin Yilmaz

**Abstracts**: Anomaly detection in videos is an important computer vision problem with various applications including automated video surveillance. Although adversarial attacks on image understanding models have been heavily investigated, there is not much work on adversarial machine learning targeting video understanding models and no previous work which focuses on video anomaly detection. To this end, we investigate an adversarial machine learning attack against video anomaly detection systems, that can be implemented via an easy-to-perform cyber-attack. Since surveillance cameras are usually connected to the server running the anomaly detection model through a wireless network, they are prone to cyber-attacks targeting the wireless connection. We demonstrate how Wi-Fi deauthentication attack, a notoriously easy-to-perform and effective denial-of-service (DoS) attack, can be utilized to generate adversarial data for video anomaly detection systems. Specifically, we apply several effects caused by the Wi-Fi deauthentication attack on video quality (e.g., slow down, freeze, fast forward, low resolution) to the popular benchmark datasets for video anomaly detection. Our experiments with several state-of-the-art anomaly detection models show that the attackers can significantly undermine the reliability of video anomaly detection systems by causing frequent false alarms and hiding physical anomalies from the surveillance system.

摘要: 视频中的异常检测是一个重要的计算机视觉问题，在包括自动视频监控在内的各种应用中都有应用。虽然针对图像理解模型的对抗性攻击已经得到了大量的研究，但针对视频理解模型的对抗性机器学习的研究还很少，也没有专门针对视频异常检测的工作。为此，我们研究了一种针对视频异常检测系统的对抗性机器学习攻击，该攻击可以通过易于执行的网络攻击来实现。由于监控摄像头通常通过无线网络连接到运行异常检测模型的服务器，因此它们容易受到针对无线连接的网络攻击。我们演示了如何利用Wi-Fi解除身份验证攻击，这是一种众所周知的易于执行和有效的拒绝服务(DoS)攻击，可以为视频异常检测系统生成敌意数据。具体地说，我们将Wi-Fi解除身份验证攻击对视频质量造成的几种影响(例如，减速、冻结、快进、低分辨率)应用于流行的基准数据集，用于视频异常检测。我们用几种最先进的异常检测模型进行的实验表明，攻击者可以通过频繁的误报警和对监控系统隐藏物理异常来显著破坏视频异常检测系统的可靠性。



## **40. Control barrier function based attack-recovery with provable guarantees**

具有可证明保证的基于控制屏障函数的攻击恢复 cs.SY

8 pages, 6 figures

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.03077v1)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstracts**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack-detection mechanism based on a zeroing control barrier function (ZCBF) condition. In addition we design an adaptive recovery mechanism based on how close the system is from violating safety. We show that the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. Finally, we use a Quadratic Programming (QP) approach for online recovery (and nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.

摘要: 研究了网络物理系统在执行器攻击下的可证明安全保证问题。特别是，我们考虑了CPS的安全性，提出了一种新的基于归零控制屏障函数(ZCBF)条件的攻击检测机制。此外，我们还设计了一种基于系统与违反安全的距离的自适应恢复机制。我们证明了攻击检测机制是健全的，即对于对抗性攻击没有漏报。最后，我们使用二次规划(QP)方法进行在线恢复(和标称)控制综合。在一个四旋翼发动机受到攻击的仿真案例研究中，我们证明了所提方法的有效性。



## **41. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02887v1)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 深度神经网络已被证明非常容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒攻击中取得了令人印象深刻的攻击成功率之后，更多的注意力转移到了黑盒攻击上。在这两种情况下，常见的基于梯度的方法通常使用$SIGN$函数在过程结束时生成扰动。然而，只有少数著作注意到$SIGN$函数的局限性。原始梯度与产生的噪声之间的偏差可能会导致不准确的梯度更新估计和对抗性转移的次优解，这是黑盒攻击的关键。针对这一问题，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)来提高恶意例子的可转移性。具体地说，在基于梯度的攻击中，我们使用数据重缩放来代替低效的$sign$函数，而不需要额外的计算代价。我们还提出了深度优先采样的方法，消除了重缩放的波动，稳定了梯度更新。我们的方法可以用于任何基于梯度的优化，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗性转移。在标准ImageNet数据集上的大量实验表明，我们的S-FGRM可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **42. Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck**

利用信息瓶颈从对抗性实例中提取稳健和非稳健特征 cs.LG

NeurIPS 2021

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02735v1)

**Authors**: Junho Kim, Byung-Kwan Lee, Yong Man Ro

**Abstracts**: Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.

摘要: 由精心设计的扰动产生的对抗性例子在研究领域引起了相当大的关注。最近的工作认为，稳健特征和非稳健特征的存在是造成对抗性例子的主要原因，并研究了它们在特征空间中的内在交互作用。在本文中，我们提出了一种利用信息瓶颈将特征表示显式提取为稳健和非稳健特征的方法。具体地说，我们将噪声变化注入到每个特征单元中，并评估特征表示中的信息流，以基于噪声变化的大小来区分稳健或非稳健的特征单元。通过综合实验，我们证明所提取的特征与对抗性预测高度相关，并且它们本身就具有人类可感知的语义信息。此外，提出了一种增强与模型预测直接相关的非稳健特征梯度的攻击机制，并验证了其打破模型稳健性的有效性。



## **43. Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition**

滚动颜色：对抗红绿灯识别的激光攻击 cs.CV

To be published in USENIX Security 2022

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02675v1)

**Authors**: Chen Yan, Zhijian Xu, Zhanyuan Yin, Xiaoyu Ji, Wenyuan Xu

**Abstracts**: Traffic light recognition is essential for fully autonomous driving in urban areas. In this paper, we investigate the feasibility of fooling traffic light recognition mechanisms by shedding laser interference on the camera. By exploiting the rolling shutter of CMOS sensors, we manage to inject a color stripe overlapped on the traffic light in the image, which can cause a red light to be recognized as a green light or vice versa. To increase the success rate, we design an optimization method to search for effective laser parameters based on empirical models of laser interference. Our evaluation in emulated and real-world setups on 2 state-of-the-art recognition systems and 5 cameras reports a maximum success rate of 30% and 86.25% for Red-to-Green and Green-to-Red attacks. We observe that the attack is effective in continuous frames from more than 40 meters away against a moving vehicle, which may cause end-to-end impacts on self-driving such as running a red light or emergency stop. To mitigate the threat, we propose redesigning the rolling shutter mechanism.

摘要: 红绿灯识别是城市地区实现全自动驾驶的关键。在本文中，我们研究了通过在摄像机上散布激光干涉来欺骗交通灯识别机制的可行性。通过利用CMOS传感器的滚动快门，我们成功地在图像中的交通灯上注入了重叠的彩色条纹，这可以使红灯被识别为绿灯，反之亦然。为了提高成功率，我们设计了一种基于激光干涉经验模型的优化方法来搜索有效的激光参数。我们在2个最先进的识别系统和5个摄像头上的模拟和真实设置中的评估报告，红色到绿色和绿色到红色攻击的最大成功率分别为30%和86.25%。我们观察到，攻击在40米以外的连续帧中对移动的车辆有效，这可能会对自动驾驶造成端到端的影响，如闯红灯或紧急停车。为了减轻威胁，我们建议重新设计滚动快门机构。



## **44. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

网络物理关键基础设施中非私有联邦学习的对抗性分析 cs.CR

11 pages, 5 figures, 4 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02654v1)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung, La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstracts**: Differential privacy (DP) is considered to be an effective privacy-preservation method to secure the promising distributed machine learning (ML) paradigm-federated learning (FL) from privacy attacks (e.g., membership inference attack). Nevertheless, while the DP mechanism greatly alleviates privacy concerns, recent studies have shown that it can be exploited to conduct security attacks (e.g., false data injection attacks). To address such attacks on FL-based applications in critical infrastructures, in this paper, we perform the first systematic study on the DP-exploited poisoning attacks from an adversarial point of view. We demonstrate that the DP method, despite providing a level of privacy guarantee, can effectively open a new poisoning attack vector for the adversary. Our theoretical analysis and empirical evaluation of a smart grid dataset show the FL performance degradation (sub-optimal model generation) scenario due to the differential noise-exploited selective model poisoning attacks. As a countermeasure, we propose a reinforcement learning-based differential privacy level selection (rDP) process. The rDP process utilizes the differential privacy parameters (privacy loss, information leakage probability, etc.) and the losses to intelligently generate an optimal privacy level for the nodes. The evaluation shows the accumulated reward and errors of the proposed technique converge to an optimal privacy policy.

摘要: 差分隐私(DP)被认为是一种有效的隐私保护方法，可以保护分布式机器学习(ML)范型联合学习(FL)免受隐私攻击(如成员推理攻击)。然而，虽然DP机制极大地缓解了对隐私的担忧，但最近的研究表明，它可以被利用来进行安全攻击(例如，虚假数据注入攻击)。为了解决这类针对关键基础设施中基于FL的应用程序的攻击，本文首次从对抗的角度对DP利用的中毒攻击进行了系统的研究。我们证明，虽然DP方法提供了一定程度的隐私保障，但可以有效地为攻击者打开一个新的中毒攻击载体。我们的理论分析和对智能电网数据集的经验评估表明，差值噪声利用的选择性模型中毒攻击导致FL性能下降(次优模型生成)。作为对策，我们提出了一种基于强化学习的差异隐私级别选择(RDP)过程。RDP过程使用不同的隐私参数(隐私丢失、信息泄露概率等)。以及智能地为节点生成最佳隐私级别的损失。评估结果表明，该技术的累积奖赏和误差收敛于最优隐私策略。



## **45. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？面向最优高效逃避攻击的Deep RL cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2106.05087v4)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)代理在状态观测(在某些约束范围内)的最强/最优对抗扰动下的最坏情况下的性能对于理解RL代理的稳健性至关重要。然而，就我们是否能找到最佳攻击以及找到最佳攻击的效率而言，找到最佳对手是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将智能体视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作更有效。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法的性能普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验稳健性。



## **46. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

补丁-愚人：视觉变形金刚在对抗敌方干扰时总是健壮吗？ cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.08392v2)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.

摘要: 视觉转换器(VITS)最近掀起了神经结构设计的新浪潮，这要归功于它们在各种视觉任务中的创纪录表现。与此同时，为了实现将VITS部署到现实世界视觉应用中的目标，它们对潜在恶意攻击的健壮性得到了越来越多的关注。特别是，最近的研究表明，与卷积神经网络(CNN)相比，VITS对对抗攻击具有更强的鲁棒性，推测这是因为VITS更注重捕捉不同输入/特征块之间的全局交互，从而提高了它们对敌对攻击造成的局部扰动的鲁棒性。在这项工作中，我们提出了一个耐人寻味的问题：“在什么样的扰动下，VITS比CNN更容易成为学习者？”在这个问题的驱动下，我们首先对VITS和CNN在各种现有的对抗性攻击下的健壮性进行了全面的实验，以了解有利于其健壮性的潜在原因。在此基础上，我们提出了一个专门的攻击框架，称为Patch-Fool，它通过使用一系列注意力感知优化技术来攻击自我注意机制的基本组成部分(即单个补丁)来愚弄自我注意机制。有趣的是，我们的Patch-Fool框架首次表明，VITS在对抗对手扰动时并不一定比CNN更健壮。特别是，我们发现VITS比CNN更容易学习，这在广泛的实验中是一致的，并且来自Patch-Fool的两个变种稀疏/温和Patch-Fool的观察表明，每个补丁上的扰动密度和强度似乎是影响VITS和CNN之间健壮性排名的关键因素。



## **47. Exploring Robust Architectures for Deep Artificial Neural Networks**

探索深度人工神经网络的健壮体系结构 cs.LG

27 pages, 16 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2106.15850v2)

**Authors**: Asim Waqas, Ghulam Rasool, Hamza Farooq, Nidhal C. Bouaynaya

**Abstracts**: The architectures of deep artificial neural networks (DANNs) are routinely studied to improve their predictive performance. However, the relationship between the architecture of a DANN and its robustness to noise and adversarial attacks is less explored. We investigate how the robustness of DANNs relates to their underlying graph architectures or structures. This study: (1) starts by exploring the design space of architectures of DANNs using graph-theoretic robustness measures; (2) transforms the graphs to DANN architectures to train/validate/test on various image classification tasks; (3) explores the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures. We show that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness performance of DANNs. The said relationship is stronger for complex tasks and large DANNs. Our work will allow autoML and neural architecture search community to explore design spaces of robust and accurate DANNs.

摘要: 人们经常研究深度人工神经网络(DEN)的结构，以提高其预测性能。然而，DANN的体系结构与其对噪声和敌意攻击的稳健性之间的关系却鲜有人研究。我们研究了DNA的健壮性如何与其底层的图体系结构或结构相关。本研究：(1)使用图论稳健性度量方法探索DANN体系结构的设计空间；(2)将图转换为DANN体系结构，以对各种图像分类任务进行训练/验证/测试；(3)探索训练的DANN体系结构对噪声和敌对攻击的健壮性与其底层体系结构的健壮性之间的关系。我们证明了基础图的拓扑熵和Olivier-Ricci曲率可以量化DANS的稳健性。对于复杂的任务和大的丹尼，上述关系更加牢固。我们的工作将允许AutoML和神经架构搜索社区探索健壮和准确的DAN的设计空间。



## **48. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

联合学习中抵抗语音情感识别属性推理攻击的用户级差分隐私 cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02500v1)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.

摘要: 许多现有的隐私增强型语音情感识别(SER)框架专注于通过集中式机器学习设置中的对抗性训练来扰乱原始语音数据。然而，这种隐私保护方案可能会失败，因为攻击者仍然可以访问受干扰的数据。近年来，分布式学习算法，特别是联邦学习(FL)算法在机器学习应用中保护隐私得到了广泛的应用。虽然FL通过将数据保存在本地设备上来提供良好的直觉来保护隐私，但先前的工作表明，使用FL训练的SER系统可以实现隐私攻击，例如属性推理攻击。在这项工作中，我们建议评估用户级差异隐私(UDP)在缓解FL中SER系统的隐私泄漏方面的作用。UDP通过隐私参数$\epsilon$和$\Delta$提供理论上的隐私保证。实验结果表明，UDP协议在保持SER系统可用性的同时，有效地减少了属性信息泄露，且攻击者只需访问一次模型更新。然而，当FL系统向对手泄露更多的模型更新时，UDP的效率会受到影响。我们将代码公开，以便在https://github.com/usc-sail/fed-ser-leakage.中重现结果



## **49. Training-Free Robust Multimodal Learning via Sample-Wise Jacobian Regularization**

基于样本明智雅可比正则化的免训练鲁棒多模学习 cs.CV

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02485v1)

**Authors**: Zhengqi Gao, Sucheng Ren, Zihui Xue, Siting Li, Hang Zhao

**Abstracts**: Multimodal fusion emerges as an appealing technique to improve model performances on many tasks. Nevertheless, the robustness of such fusion methods is rarely involved in the present literature. In this paper, we propose a training-free robust late-fusion method by exploiting conditional independence assumption and Jacobian regularization. Our key is to minimize the Frobenius norm of a Jacobian matrix, where the resulting optimization problem is relaxed to a tractable Sylvester equation. Furthermore, we provide a theoretical error bound of our method and some insights about the function of the extra modality. Several numerical experiments on AV-MNIST, RAVDESS, and VGGsound demonstrate the efficacy of our method under both adversarial attacks and random corruptions.

摘要: 多通道融合是提高模型在许多任务上性能的一种很有吸引力的技术。然而，这种融合方法的稳健性在目前的文献中很少涉及。本文利用条件独立性假设和雅可比正则化，提出了一种无需训练的鲁棒晚融合方法。我们的关键是最小化雅可比矩阵的Frobenius范数，由此产生的优化问题被松弛到一个容易处理的Sylvester方程。此外，我们还给出了该方法的理论误差界，并对额外通道的作用提出了一些见解。在AV-MNIST、RAVDESS和VGGound上的几个数值实验证明了我们的方法在对抗攻击和随机破坏下的有效性。



## **50. Hear No Evil: Towards Adversarial Robustness of Automatic Speech Recognition via Multi-Task Learning**

听而不闻：通过多任务学习实现自动语音识别的对抗健壮性 eess.AS

Submitted to Insterspeech 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02381v1)

**Authors**: Nilaksh Das, Duen Horng Chau

**Abstracts**: As automatic speech recognition (ASR) systems are now being widely deployed in the wild, the increasing threat of adversarial attacks raises serious questions about the security and reliability of using such systems. On the other hand, multi-task learning (MTL) has shown success in training models that can resist adversarial attacks in the computer vision domain. In this work, we investigate the impact of performing such multi-task learning on the adversarial robustness of ASR models in the speech domain. We conduct extensive MTL experimentation by combining semantically diverse tasks such as accent classification and ASR, and evaluate a wide range of adversarial settings. Our thorough analysis reveals that performing MTL with semantically diverse tasks consistently makes it harder for an adversarial attack to succeed. We also discuss in detail the serious pitfalls and their related remedies that have a significant impact on the robustness of MTL models. Our proposed MTL approach shows considerable absolute improvements in adversarially targeted WER ranging from 17.25 up to 59.90 compared to single-task learning baselines (attention decoder and CTC respectively). Ours is the first in-depth study that uncovers adversarial robustness gains from multi-task learning for ASR.

摘要: 随着自动语音识别(ASR)系统的广泛应用，日益增长的对抗性攻击威胁对使用这类系统的安全性和可靠性提出了严重的问题。另一方面，多任务学习(MTL)在训练模型抵抗计算机视觉领域中的敌意攻击方面取得了成功。在这项工作中，我们研究了执行这种多任务学习对ASR模型在语音域的对抗健壮性的影响。我们通过结合重音分类和ASR等语义多样化的任务来进行广泛的MTL实验，并评估了广泛的对抗性环境。我们的全面分析表明，以语义多样化的任务执行MTL始终会使敌方攻击更难成功。我们还详细讨论了对MTL模型的稳健性有重大影响的严重陷阱及其相关补救措施。与单任务学习基线(注意解码器和CTC)相比，我们提出的MTL方法在相反的目标WER上有相当大的绝对改善，从17.25%到59.90%。我们的研究是第一次深入研究ASR从多任务学习中获得的对手健壮性收益。



