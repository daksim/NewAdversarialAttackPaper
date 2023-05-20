# Latest Adversarial Attack Papers
**update at 2023-05-20 15:39:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attacks on Online Learners: a Teacher-Student Analysis**

对网络学习者的攻击：一种师生分析 stat.ML

15 pages, 6 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11132v1) [paper-pdf](http://arxiv.org/pdf/2305.11132v1)

**Authors**: Riccardo Giuseppe Margiotta, Sebastian Goldt, Guido Sanguinetti

**Abstract**: Machine learning models are famously vulnerable to adversarial attacks: small ad-hoc perturbations of the data that can catastrophically alter the model predictions. While a large literature has studied the case of test-time attacks on pre-trained models, the important case of attacks in an online learning setting has received little attention so far. In this work, we use a control-theoretical perspective to study the scenario where an attacker may perturb data labels to manipulate the learning dynamics of an online learner. We perform a theoretical analysis of the problem in a teacher-student setup, considering different attack strategies, and obtaining analytical results for the steady state of simple linear learners. These results enable us to prove that a discontinuous transition in the learner's accuracy occurs when the attack strength exceeds a critical threshold. We then study empirically attacks on learners with complex architectures using real data, confirming the insights of our theoretical analysis. Our findings show that greedy attacks can be extremely efficient, especially when data stream in small batches.

摘要: 众所周知，机器学习模型容易受到对抗性攻击：对数据的微小特别扰动可能会灾难性地改变模型预测。虽然有大量文献研究了测试时间攻击预先训练的模型的案例，但到目前为止，在线学习环境中的重要攻击案例很少受到关注。在这项工作中，我们使用控制理论的观点来研究攻击者可能扰乱数据标签以操纵在线学习者的学习动态的场景。我们在教师-学生系统中对该问题进行了理论分析，考虑了不同的攻击策略，得到了简单线性学习者稳态的解析结果。这些结果使我们能够证明，当攻击强度超过临界阈值时，学习者的准确率会发生不连续的转变。然后，我们使用真实数据对具有复杂架构的学习者进行了实证研究，证实了我们的理论分析的真知灼见。我们的发现表明，贪婪攻击可以非常有效，特别是当数据流以小批量传输时。



## **2. Deep PackGen: A Deep Reinforcement Learning Framework for Adversarial Network Packet Generation**

Deep PackGen：一种用于对抗性网络数据包生成的深度强化学习框架 cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.11039v1) [paper-pdf](http://arxiv.org/pdf/2305.11039v1)

**Authors**: Soumyadeep Hore, Jalal Ghadermazi, Diwas Paudel, Ankit Shah, Tapas K. Das, Nathaniel D. Bastian

**Abstract**: Recent advancements in artificial intelligence (AI) and machine learning (ML) algorithms, coupled with the availability of faster computing infrastructure, have enhanced the security posture of cybersecurity operations centers (defenders) through the development of ML-aided network intrusion detection systems (NIDS). Concurrently, the abilities of adversaries to evade security have also increased with the support of AI/ML models. Therefore, defenders need to proactively prepare for evasion attacks that exploit the detection mechanisms of NIDS. Recent studies have found that the perturbation of flow-based and packet-based features can deceive ML models, but these approaches have limitations. Perturbations made to the flow-based features are difficult to reverse-engineer, while samples generated with perturbations to the packet-based features are not playable.   Our methodological framework, Deep PackGen, employs deep reinforcement learning to generate adversarial packets and aims to overcome the limitations of approaches in the literature. By taking raw malicious network packets as inputs and systematically making perturbations on them, Deep PackGen camouflages them as benign packets while still maintaining their functionality. In our experiments, using publicly available data, Deep PackGen achieved an average adversarial success rate of 66.4\% against various ML models and across different attack types. Our investigation also revealed that more than 45\% of the successful adversarial samples were out-of-distribution packets that evaded the decision boundaries of the classifiers. The knowledge gained from our study on the adversary's ability to make specific evasive perturbations to different types of malicious packets can help defenders enhance the robustness of their NIDS against evolving adversarial attacks.

摘要: 人工智能(AI)和机器学习(ML)算法的最新进展，加上更快的计算基础设施的可用性，通过开发ML辅助的网络入侵检测系统(NID)，增强了网络安全运营中心(防御者)的安全态势。同时，在AI/ML模型的支持下，攻击者逃避安全的能力也有所增强。因此，防御者需要主动准备利用网络入侵检测系统的检测机制进行规避攻击。最近的研究发现，基于流和基于分组的特征的扰动可以欺骗ML模型，但这些方法都有局限性。对基于流的特征进行的扰动很难进行反向工程，而利用对基于分组的特征的扰动生成的样本是不可播放的。我们的方法框架，Deep PackGen，使用深度强化学习来生成对抗性分组，旨在克服文献中方法的局限性。通过将原始恶意网络数据包作为输入并系统地对其进行干扰，Deep PackGen将其伪装成良性数据包，同时仍保持其功能。在我们的实验中，使用公开的数据，Deep PackGen在不同的ML模型和不同的攻击类型上获得了66.4%的平均攻击成功率。我们的调查还发现，超过45%的成功对抗样本是绕过分类器决策边界的非分发分组。我们从研究对手对不同类型的恶意数据包进行特定规避扰动的能力中获得的知识可以帮助防御者增强其网络入侵检测系统对不断演变的敌意攻击的健壮性。



## **3. SoK: Data Privacy in Virtual Reality**

SOK：虚拟现实中的数据隐私 cs.HC

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2301.05940v2) [paper-pdf](http://arxiv.org/pdf/2301.05940v2)

**Authors**: Gonzalo Munilla Garrido, Vivek Nair, Dawn Song

**Abstract**: The adoption of virtual reality (VR) technologies has rapidly gained momentum in recent years as companies around the world begin to position the so-called "metaverse" as the next major medium for accessing and interacting with the internet. While consumers have become accustomed to a degree of data harvesting on the web, the real-time nature of data sharing in the metaverse indicates that privacy concerns are likely to be even more prevalent in the new "Web 3.0." Research into VR privacy has demonstrated that a plethora of sensitive personal information is observable by various would-be adversaries from just a few minutes of telemetry data. On the other hand, we have yet to see VR parallels for many privacy-preserving tools aimed at mitigating threats on conventional platforms. This paper aims to systematize knowledge on the landscape of VR privacy threats and countermeasures by proposing a comprehensive taxonomy of data attributes, protections, and adversaries based on the study of 68 collected publications. We complement our qualitative discussion with a statistical analysis of the risk associated with various data sources inherent to VR in consideration of the known attacks and defenses. By focusing on highlighting the clear outstanding opportunities, we hope to motivate and guide further research into this increasingly important field.

摘要: 近年来，虚拟现实(VR)技术的采用势头迅速增强，世界各地的公司开始将所谓的“虚拟现实”定位为访问互联网和与互联网互动的下一个主要媒介。虽然消费者已经习惯了在一定程度上从网络上获取数据，但虚拟世界中数据共享的实时性质表明，对隐私的担忧可能会在新的“Web 3.0”中更加普遍。对VR隐私的研究表明，各种潜在的对手只需几分钟的遥测数据就可以观察到过多的敏感个人信息。另一方面，我们还没有看到许多旨在缓解传统平台上威胁的隐私保护工具的VR相似之处。本文旨在通过对收集到的68种出版物的研究，提出数据属性、保护和对手的全面分类，以系统化关于虚拟现实隐私威胁和对策的知识。考虑到已知的攻击和防御，我们用与VR固有的各种数据源相关的风险的统计分析来补充我们的定性讨论。通过重点突出明确的突出机遇，我们希望激励和指导对这一日益重要的领域的进一步研究。



## **4. Certified Robust Neural Networks: Generalization and Corruption Resistance**

认证的稳健神经网络：泛化和抗腐蚀性 stat.ML

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2303.02251v2) [paper-pdf](http://arxiv.org/pdf/2303.02251v2)

**Authors**: Amine Bennouna, Ryan Lucas, Bart Van Parys

**Abstract**: Recent work have demonstrated that robustness (to "corruption") can be at odds with generalization. Adversarial training, for instance, aims to reduce the problematic susceptibility of modern neural networks to small data perturbations. Surprisingly, overfitting is a major concern in adversarial training despite being mostly absent in standard training. We provide here theoretical evidence for this peculiar "robust overfitting" phenomenon. Subsequently, we advance a novel distributionally robust loss function bridging robustness and generalization. We demonstrate both theoretically as well as empirically the loss to enjoy a certified level of robustness against two common types of corruption--data evasion and poisoning attacks--while ensuring guaranteed generalization. We show through careful numerical experiments that our resulting holistic robust (HR) training procedure yields SOTA performance. Finally, we indicate that HR training can be interpreted as a direct extension of adversarial training and comes with a negligible additional computational burden. A ready-to-use python library implementing our algorithm is available at https://github.com/RyanLucas3/HR_Neural_Networks.

摘要: 最近的研究表明，健壮性(对“腐败”)可能与泛化不一致。例如，对抗性训练旨在降低现代神经网络对小数据扰动的问题敏感度。令人惊讶的是，尽管在标准训练中大多缺席，但过度适应是对抗性训练中的一个主要问题。我们在这里为这一特殊的“稳健过拟合”现象提供了理论证据。随后，我们提出了一种新的分布稳健损失函数，它在稳健性和泛化之间架起了桥梁。我们在理论上和经验上都证明了在确保泛化的同时，对两种常见的腐败类型--数据逃避和中毒攻击--具有经过认证的健壮性水平。我们通过仔细的数值实验表明，我们由此产生的整体稳健(HR)训练过程产生了SOTA性能。最后，我们指出，HR训练可以被解释为对抗性训练的直接扩展，并且伴随着可以忽略的额外计算负担。在https://github.com/RyanLucas3/HR_Neural_Networks.上有一个实现我们的算法的现成的Python库



## **5. Architecture-agnostic Iterative Black-box Certified Defense against Adversarial Patches**

架构不可知的迭代黑盒认证防御对手补丁 cs.CV

9 pages

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10929v1) [paper-pdf](http://arxiv.org/pdf/2305.10929v1)

**Authors**: Di Yang, Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Yang Liu, Geguang Pu

**Abstract**: The adversarial patch attack aims to fool image classifiers within a bounded, contiguous region of arbitrary changes, posing a real threat to computer vision systems (e.g., autonomous driving, content moderation, biometric authentication, medical imaging) in the physical world. To address this problem in a trustworthy way, proposals have been made for certified patch defenses that ensure the robustness of classification models and prevent future patch attacks from breaching the defense. State-of-the-art certified defenses can be compatible with any model architecture, as well as achieve high clean and certified accuracy. Although the methods are adaptive to arbitrary patch positions, they inevitably need to access the size of the adversarial patch, which is unreasonable and impractical in real-world attack scenarios. To improve the feasibility of the architecture-agnostic certified defense in a black-box setting (i.e., position and size of the patch are both unknown), we propose a novel two-stage Iterative Black-box Certified Defense method, termed IBCD.In the first stage, it estimates the patch size in a search-based manner by evaluating the size relationship between the patch and mask with pixel masking. In the second stage, the accuracy results are calculated by the existing white-box certified defense methods with the estimated patch size. The experiments conducted on two popular model architectures and two datasets verify the effectiveness and efficiency of IBCD.

摘要: 敌意补丁攻击旨在愚弄任意变化的有界连续区域内的图像分类器，对物理世界中的计算机视觉系统(例如，自动驾驶、内容审核、生物特征验证、医学成像)构成真正的威胁。为了以可信的方式解决这个问题，已经提出了认证补丁防御的建议，以确保分类模型的健壮性，并防止未来的补丁攻击破坏防御。最先进的认证防御可以兼容任何型号的架构，以及实现高清洁和认证的准确性。虽然这些方法能够适应任意的补丁位置，但不可避免地需要访问敌方补丁的大小，这在现实世界的攻击场景中是不合理和不切实际的。为了提高在黑盒环境下(即补丁的位置和大小都未知)下基于体系结构的认证防御的可行性，提出了一种新的两阶段迭代黑盒认证防御方法IBCD。在第二阶段，利用现有的白盒认证防御方法和估计的补丁大小来计算精度结果。在两个流行的模型架构和两个数据集上进行的实验验证了IBCD的有效性和高效性。



## **6. Free Lunch for Privacy Preserving Distributed Graph Learning**

保护隐私的分布式图学习的免费午餐 cs.LG

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10869v1) [paper-pdf](http://arxiv.org/pdf/2305.10869v1)

**Authors**: Nimesh Agrawal, Nikita Malik, Sandeep Kumar

**Abstract**: Learning on graphs is becoming prevalent in a wide range of applications including social networks, robotics, communication, medicine, etc. These datasets belonging to entities often contain critical private information. The utilization of data for graph learning applications is hampered by the growing privacy concerns from users on data sharing. Existing privacy-preserving methods pre-process the data to extract user-side features, and only these features are used for subsequent learning. Unfortunately, these methods are vulnerable to adversarial attacks to infer private attributes. We present a novel privacy-respecting framework for distributed graph learning and graph-based machine learning. In order to perform graph learning and other downstream tasks on the server side, this framework aims to learn features as well as distances without requiring actual features while preserving the original structural properties of the raw data. The proposed framework is quite generic and highly adaptable. We demonstrate the utility of the Euclidean space, but it can be applied with any existing method of distance approximation and graph learning for the relevant spaces. Through extensive experimentation on both synthetic and real datasets, we demonstrate the efficacy of the framework in terms of comparing the results obtained without data sharing to those obtained with data sharing as a benchmark. This is, to our knowledge, the first privacy-preserving distributed graph learning framework.

摘要: 基于图的学习在社交网络、机器人、通信、医学等广泛的应用中变得普遍。这些属于实体的数据集通常包含关键的私人信息。由于用户对数据共享的隐私性日益关注，阻碍了用于图形学习应用的数据的利用。现有的隐私保护方法对数据进行预处理，提取用户侧特征，只有这些特征才能用于后续学习。不幸的是，这些方法容易受到敌意攻击来推断私有属性。提出了一种新的隐私保护框架，用于分布式图学习和基于图的机器学习。为了在服务器端执行图学习和其他下游任务，该框架旨在学习特征和距离，而不需要实际特征，同时保持原始数据的原始结构属性。该框架具有较强的通用性和较强的适应性。我们演示了欧几里得空间的效用，但它可以应用于相关空间的任何现有的距离逼近和图学习方法。通过在合成数据集和真实数据集上的广泛实验，我们证明了该框架在比较没有数据共享的结果和以数据共享为基准的结果方面的有效性。据我们所知，这是第一个隐私保护的分布式图学习框架。



## **7. How Deep Learning Sees the World: A Survey on Adversarial Attacks & Defenses**

深度学习如何看待世界：对抗性攻防研究综述 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10862v1) [paper-pdf](http://arxiv.org/pdf/2305.10862v1)

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Deep Learning is currently used to perform multiple tasks, such as object recognition, face recognition, and natural language processing. However, Deep Neural Networks (DNNs) are vulnerable to perturbations that alter the network prediction (adversarial examples), raising concerns regarding its usage in critical areas, such as self-driving vehicles, malware detection, and healthcare. This paper compiles the most recent adversarial attacks, grouped by the attacker capacity, and modern defenses clustered by protection strategies. We also present the new advances regarding Vision Transformers, summarize the datasets and metrics used in the context of adversarial settings, and compare the state-of-the-art results under different attacks, finishing with the identification of open issues.

摘要: 深度学习目前用于执行多个任务，如对象识别、人脸识别和自然语言处理。然而，深度神经网络(DNN)很容易受到改变网络预测的扰动(对手的例子)，这引发了人们对其在关键领域的使用的担忧，如自动驾驶车辆、恶意软件检测和医疗保健。这篇论文汇编了最新的对抗性攻击，按攻击者的能力分组，以及按保护策略分组的现代防御。我们还介绍了关于Vision Transformers的新进展，总结了在对抗性环境下使用的数据集和度量，并比较了不同攻击下的最新结果，最后确定了有待解决的问题。



## **8. Towards an Accurate and Secure Detector against Adversarial Perturbations**

走向准确和安全的检测器以抵御对手的扰动 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10856v1) [paper-pdf](http://arxiv.org/pdf/2305.10856v1)

**Authors**: Chao Wang, Shuren Qi, Zhiqiu Huang, Yushu Zhang, Xiaochun Cao

**Abstract**: The vulnerability of deep neural networks to adversarial perturbations has been widely perceived in the computer vision community. From a security perspective, it poses a critical risk for modern vision systems, e.g., the popular Deep Learning as a Service (DLaaS) frameworks. For protecting off-the-shelf deep models while not modifying them, current algorithms typically detect adversarial patterns through discriminative decomposition of natural-artificial data. However, these decompositions are biased towards frequency or spatial discriminability, thus failing to capture subtle adversarial patterns comprehensively. More seriously, they are typically invertible, meaning successful defense-aware (secondary) adversarial attack (i.e., evading the detector as well as fooling the model) is practical under the assumption that the adversary is fully aware of the detector (i.e., the Kerckhoffs's principle). Motivated by such facts, we propose an accurate and secure adversarial example detector, relying on a spatial-frequency discriminative decomposition with secret keys. It expands the above works on two aspects: 1) the introduced Krawtchouk basis provides better spatial-frequency discriminability and thereby is more suitable for capturing adversarial patterns than the common trigonometric or wavelet basis; 2) the extensive parameters for decomposition are generated by a pseudo-random function with secret keys, hence blocking the defense-aware adversarial attack. Theoretical and numerical analysis demonstrates the increased accuracy and security of our detector w.r.t. a number of state-of-the-art algorithms.

摘要: 深度神经网络对对抗性扰动的脆弱性已经在计算机视觉领域得到了广泛的认识。从安全的角度来看，它对现代视觉系统构成了严重的风险，例如流行的深度学习即服务(DLaaS)框架。为了在不修改现有深度模型的同时保护它们，目前的算法通常通过对自然-人工数据的区别性分解来检测对抗性模式。然而，这些分解偏向于频率或空间可区分性，因此无法全面地捕捉到微妙的对抗性模式。更严重的是，它们通常是可逆的，这意味着在假设对手完全知道检测器(即Kerckhoff原理)的情况下，成功的防御感知(次要)对手攻击(即，躲避检测器以及愚弄模型)是实用的。在此基础上，提出了一种基于密钥的空频判别分解的准确、安全的对抗性样本检测器。它从两个方面对上述工作进行了扩展：1)引入的Krawtchouk基提供了更好的空频分辨能力，因此比普通的三角或小波基更适合于捕获敌意模式；2)分解的广泛参数是由带有密钥的伪随机函数产生的，从而阻止了具有防御意识的敌意攻击。理论和数值分析表明，我们的探测器的准确度和安全性都有所提高。一些最先进的算法。



## **9. Adversarial Scratches: Deployable Attacks to CNN Classifiers**

对抗性抓痕：对CNN分类器的可部署攻击 cs.LG

This work is published at Pattern Recognition (Elsevier). This paper  stems from 'Scratch that! An Evolution-based Adversarial Attack against  Neural Networks' for which an arXiv preprint is available at  arXiv:1912.02316. Further studies led to a complete overhaul of the work,  resulting in this paper

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2204.09397v3) [paper-pdf](http://arxiv.org/pdf/2204.09397v3)

**Authors**: Loris Giulivi, Malhar Jere, Loris Rossi, Farinaz Koushanfar, Gabriela Ciocarlie, Briland Hitaj, Giacomo Boracchi

**Abstract**: A growing body of work has shown that deep neural networks are susceptible to adversarial examples. These take the form of small perturbations applied to the model's input which lead to incorrect predictions. Unfortunately, most literature focuses on visually imperceivable perturbations to be applied to digital images that often are, by design, impossible to be deployed to physical targets. We present Adversarial Scratches: a novel L0 black-box attack, which takes the form of scratches in images, and which possesses much greater deployability than other state-of-the-art attacks. Adversarial Scratches leverage B\'ezier Curves to reduce the dimension of the search space and possibly constrain the attack to a specific location. We test Adversarial Scratches in several scenarios, including a publicly available API and images of traffic signs. Results show that, often, our attack achieves higher fooling rate than other deployable state-of-the-art methods, while requiring significantly fewer queries and modifying very few pixels.

摘要: 越来越多的研究表明，深度神经网络很容易受到敌意例子的影响。这些采用的形式是应用于模型输入的小扰动，从而导致不正确的预测。不幸的是，大多数文献关注的是应用于数字图像的视觉上不可察觉的扰动，而根据设计，数字图像通常不可能被部署到物理目标上。我们提出了对抗性划痕：一种新颖的L0黑盒攻击，它采用图像划痕的形式，并且比其他最先进的攻击具有更大的可部署性。对抗性划痕利用B‘ezier曲线来减少搜索空间的维度，并可能将攻击限制在特定位置。我们在几个场景中测试了对抗性划痕，包括公开可用的API和交通标志图像。结果表明，我们的攻击通常比其他可部署的最先进方法获得更高的愚骗率，同时需要的查询和修改的像素也非常少。



## **10. Adversarial Amendment is the Only Force Capable of Transforming an Enemy into a Friend**

对抗性修正案是唯一能化敌为友的力量 cs.AI

Accepted to IJCAI 2023, 10 pages, 5 figures

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10766v1) [paper-pdf](http://arxiv.org/pdf/2305.10766v1)

**Authors**: Chong Yu, Tao Chen, Zhongxue Gan

**Abstract**: Adversarial attack is commonly regarded as a huge threat to neural networks because of misleading behavior. This paper presents an opposite perspective: adversarial attacks can be harnessed to improve neural models if amended correctly. Unlike traditional adversarial defense or adversarial training schemes that aim to improve the adversarial robustness, the proposed adversarial amendment (AdvAmd) method aims to improve the original accuracy level of neural models on benign samples. We thoroughly analyze the distribution mismatch between the benign and adversarial samples. This distribution mismatch and the mutual learning mechanism with the same learning ratio applied in prior art defense strategies is the main cause leading the accuracy degradation for benign samples. The proposed AdvAmd is demonstrated to steadily heal the accuracy degradation and even leads to a certain accuracy boost of common neural models on benign classification, object detection, and segmentation tasks. The efficacy of the AdvAmd is contributed by three key components: mediate samples (to reduce the influence of distribution mismatch with a fine-grained amendment), auxiliary batch norm (to solve the mutual learning mechanism and the smoother judgment surface), and AdvAmd loss (to adjust the learning ratios according to different attack vulnerabilities) through quantitative and ablation experiments.

摘要: 对抗性攻击通常被认为是对神经网络的巨大威胁，因为它具有误导性。本文提出了一种相反的观点：如果修正正确，可以利用对抗性攻击来改进神经模型。与传统的对抗性防御或对抗性训练方案不同，提出的对抗性修正(AdvAmd)方法旨在提高神经模型对良性样本的原始精度水平。我们深入分析了良性样本和恶意样本之间的分布不匹配。这种分布失配和现有技术防御策略中采用的具有相同学习比率的相互学习机制是导致良性样本精度下降的主要原因。实验结果表明，该算法能够稳定地修复神经网络模型在良性分类、目标检测和分割等任务中的精度下降，甚至可以在一定程度上提高神经模型的精度。通过定量和烧蚀实验，AdvAmd的有效性由三个关键成分贡献：中间样本(通过细粒度修正减少分布失配的影响)、辅助批次范数(解决相互学习机制和更光滑的判断曲面)和AdvAmd损失(根据不同的攻击漏洞调整学习比率)。



## **11. Re-thinking Data Availablity Attacks Against Deep Neural Networks**

对深度神经网络数据可用性攻击的再思考 cs.CR

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10691v1) [paper-pdf](http://arxiv.org/pdf/2305.10691v1)

**Authors**: Bin Fang, Bo Li, Shuang Wu, Ran Yi, Shouhong Ding, Lizhuang Ma

**Abstract**: The unauthorized use of personal data for commercial purposes and the clandestine acquisition of private data for training machine learning models continue to raise concerns. In response to these issues, researchers have proposed availability attacks that aim to render data unexploitable. However, many current attack methods are rendered ineffective by adversarial training. In this paper, we re-examine the concept of unlearnable examples and discern that the existing robust error-minimizing noise presents an inaccurate optimization objective. Building on these observations, we introduce a novel optimization paradigm that yields improved protection results with reduced computational time requirements. We have conducted extensive experiments to substantiate the soundness of our approach. Moreover, our method establishes a robust foundation for future research in this area.

摘要: 未经授权将个人数据用于商业目的以及秘密获取用于训练机器学习模型的私人数据继续引起关注。针对这些问题，研究人员提出了旨在使数据无法利用的可用性攻击。然而，目前的许多进攻方法由于对抗性训练而变得无效。在本文中，我们重新检查了不可学习示例的概念，并发现现有的鲁棒误差最小化噪声呈现了一个不准确的优化目标。在这些观察的基础上，我们引入了一种新的优化范例，该范例可以在减少计算时间的情况下产生更好的保护结果。我们已经进行了广泛的实验，以证实我们的方法的合理性。此外，我们的方法为这一领域的未来研究奠定了坚实的基础。



## **12. Content-based Unrestricted Adversarial Attack**

基于内容的无限制对抗性攻击 cs.CV

**SubmitDate**: 2023-05-18    [abs](http://arxiv.org/abs/2305.10665v1) [paper-pdf](http://arxiv.org/pdf/2305.10665v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

**Abstract**: Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic, demonstrating their ability to deceive human perception and deep neural networks with stealth and success. However, current works usually sacrifice unrestricted degrees and subjectively select some image content to guarantee the photorealism of unrestricted adversarial examples, which limits its attack performance. To ensure the photorealism of adversarial examples and boost attack performance, we propose a novel unrestricted attack framework called Content-based Unrestricted Adversarial Attack. By leveraging a low-dimensional manifold that represents natural images, we map the images onto the manifold and optimize them along its adversarial direction. Therefore, within this framework, we implement Adversarial Content Attack based on Stable Diffusion and can generate high transferable unrestricted adversarial examples with various adversarial contents. Extensive experimentation and visualization demonstrate the efficacy of ACA, particularly in surpassing state-of-the-art attacks by an average of 13.3-50.4% and 16.8-48.0% in normally trained models and defense methods, respectively.

摘要: 不受限制的对抗性攻击通常会操纵图像的语义内容(例如，颜色或纹理)，以创建既有效又逼真的对抗性示例，展示它们以隐蔽和成功的方式欺骗人类感知和深层神经网络的能力。然而，目前的作品往往牺牲不受限制的程度，主观地选择一些图像内容来保证不受限制的对抗性例子的照片真实感，这限制了其攻击性能。为了保证对抗性实例的真实感，提高攻击性能，我们提出了一种新的无限制攻击框架，称为基于内容的无限对抗性攻击。通过利用表示自然图像的低维流形，我们将图像映射到流形上，并沿着其相反的方向进行优化。因此，在该框架下，我们实现了基于稳定扩散的对抗性内容攻击，并且可以生成具有多种对抗性内容的高可转移性的无限制对抗性实例。广泛的实验和可视化证明了蚁群算法的有效性，特别是在正常训练的模型和防御方法上，平均分别超过最先进的攻击13.3%-50.4%和16.8%-48.0%。



## **13. Exact Recovery for System Identification with More Corrupt Data than Clean Data**

使用比干净数据更多的损坏数据准确恢复系统标识 cs.LG

24 pages, 2 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10506v1) [paper-pdf](http://arxiv.org/pdf/2305.10506v1)

**Authors**: Baturalp Yalcin, Javad Lavaei, Murat Arcak

**Abstract**: In this paper, we study the system identification problem for linear discrete-time systems under adversaries and analyze two lasso-type estimators. We study both asymptotic and non-asymptotic properties of these estimators in two separate scenarios, corresponding to deterministic and stochastic models for the attack times. Since the samples collected from the system are correlated, the existing results on lasso are not applicable. We show that when the system is stable and the attacks are injected periodically, the sample complexity for the exact recovery of the system dynamics is O(n), where n is the dimension of the states. When the adversarial attacks occur at each time instance with probability p, the required sample complexity for the exact recovery scales as O(\log(n)p/(1-p)^2). This result implies the almost sure convergence to the true system dynamics under the asymptotic regime. As a by-product, even when more than half of the data is compromised, our estimators still learn the system correctly. This paper provides the first mathematical guarantee in the literature on learning from correlated data for dynamical systems in the case when there is less clean data than corrupt data.

摘要: 本文研究了对手作用下线性离散时间系统的系统辨识问题，分析了两种套索型估值器。在两种不同的情形下，我们研究了这些估计量的渐近和非渐近性质，分别对应于攻击时间的确定性和随机性模型。由于从系统采集的样本是相关的，现有的套索结果不适用。我们证明了当系统稳定且攻击被周期性注入时，精确恢复系统动力学的样本复杂度为O(N)，其中n是状态的维度。当对抗性攻击发生在概率为p的每个时刻时，精确恢复所需的样本复杂度为O(\log(N)p/(1-p)^2)。这一结果意味着在渐近状态下几乎必然收敛于真实的系统动力学。作为副产品，即使超过一半的数据被泄露，我们的估计者仍然正确地学习系统。本文首次为动态系统在干净数据少于损坏数据的情况下从相关数据中学习提供了数学保证。



## **14. Raising the Bar for Certified Adversarial Robustness with Diffusion Models**

利用扩散模型提高认证对抗性稳健性的标准 cs.LG

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10388v1) [paper-pdf](http://arxiv.org/pdf/2305.10388v1)

**Authors**: Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn

**Abstract**: Certified defenses against adversarial attacks offer formal guarantees on the robustness of a model, making them more reliable than empirical methods such as adversarial training, whose effectiveness is often later reduced by unseen attacks. Still, the limited certified robustness that is currently achievable has been a bottleneck for their practical adoption. Gowal et al. and Wang et al. have shown that generating additional training data using state-of-the-art diffusion models can considerably improve the robustness of adversarial training. In this work, we demonstrate that a similar approach can substantially improve deterministic certified defenses. In addition, we provide a list of recommendations to scale the robustness of certified training approaches. One of our main insights is that the generalization gap, i.e., the difference between the training and test accuracy of the original model, is a good predictor of the magnitude of the robustness improvement when using additional generated data. Our approach achieves state-of-the-art deterministic robustness certificates on CIFAR-10 for the $\ell_2$ ($\epsilon = 36/255$) and $\ell_\infty$ ($\epsilon = 8/255$) threat models, outperforming the previous best results by $+3.95\%$ and $+1.39\%$, respectively. Furthermore, we report similar improvements for CIFAR-100.

摘要: 针对对抗性攻击的认证防御为模型的健壮性提供了正式保证，使它们比对抗性训练等经验方法更可靠，后者的有效性后来往往因看不见的攻击而降低。尽管如此，目前可以实现的有限的认证健壮性一直是它们实际采用的瓶颈。Gowal等人。和Wang等人。已经表明，使用最先进的扩散模型生成额外的训练数据可以显著提高对抗性训练的稳健性。在这项工作中，我们证明了类似的方法可以实质性地改进确定性认证防御。此外，我们还提供了一系列建议，以衡量认证培训方法的健壮性。我们的主要见解之一是，泛化差距，即原始模型的训练精度和测试精度之间的差异，是使用额外生成的数据时稳健性改善幅度的一个很好的预测指标。我们的方法在CIFAR-10上为$\ell_2$($\epsilon=36/255$)和$\ell_inty$($\epsilon=8/255$)威胁模型获得了最先进的确定性稳健性证书，分别比之前的最好结果高出$+3.95\$和$+1.39\$。此外，我们还报告了CIFAR-100的类似改进。



## **15. Certified Invertibility in Neural Networks via Mixed-Integer Programming**

基于混合整数规划的神经网络可逆性证明 cs.LG

22 pages, 7 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2301.11783v2) [paper-pdf](http://arxiv.org/pdf/2301.11783v2)

**Authors**: Tianqi Cui, Thomas Bertalan, George J. Pappas, Manfred Morari, Ioannis G. Kevrekidis, Mahyar Fazlyab

**Abstract**: Neural networks are known to be vulnerable to adversarial attacks, which are small, imperceptible perturbations that can significantly alter the network's output. Conversely, there may exist large, meaningful perturbations that do not affect the network's decision (excessive invariance). In our research, we investigate this latter phenomenon in two contexts: (a) discrete-time dynamical system identification, and (b) the calibration of a neural network's output to that of another network. We examine noninvertibility through the lens of mathematical optimization, where the global solution measures the ``safety" of the network predictions by their distance from the non-invertibility boundary. We formulate mixed-integer programs (MIPs) for ReLU networks and $L_p$ norms ($p=1,2,\infty$) that apply to neural network approximators of dynamical systems. We also discuss how our findings can be useful for invertibility certification in transformations between neural networks, e.g. between different levels of network pruning.

摘要: 众所周知，神经网络容易受到对抗性攻击，这种攻击是微小的、不可察觉的扰动，可以显著改变网络的输出。相反，可能存在不影响网络决策的大的、有意义的扰动(过度的不变性)。在我们的研究中，我们在两个背景下研究后一种现象：(A)离散时间动态系统辨识，和(B)神经网络输出到另一网络的校准。我们通过数学最优化的视角研究不可逆性，其中全局解通过网络预测到不可逆边界的距离来衡量网络预测的“安全性”。我们为RELU网络和$L_p$范数($p=1，2，\inty$)建立了混合整数规划(MIP)。我们还讨论了我们的结果如何用于神经网络之间的变换，例如在不同级别的网络剪枝之间的可逆性证明。



## **16. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

操纵视觉感知的联邦推荐系统及其对策 cs.IR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.08183v2) [paper-pdf](http://arxiv.org/pdf/2305.08183v2)

**Authors**: Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.

摘要: 联邦推荐系统(FedRecs)由于具有保护用户数据隐私的能力，近年来得到了广泛的研究。在FedRecs中，中央服务器通过与客户共享模型公共参数来协作学习推荐模型，从而提供隐私保护解决方案。不幸的是，模型参数的曝光为对手操纵FedRecs留下了后门。已有的关于FedRec安全的研究已经表明，恶意用户很容易通过模型中毒攻击来推销物品，但这些研究主要集中在只有协作信息的FedRecs上(即用户与物品的交互)。我们认为，由于协同信号的数据稀疏性，这些攻击是有效的。在实际应用中，产品的视觉描述等辅助信息被用来缓解协同过滤数据的稀疏性。因此，当在FedRecs中加入视觉信息时，所有现有的模型中毒攻击的有效性都会受到质疑。在本文中，我们进行了大量的实验，以验证在合理的设置下，结合视觉信息可以抵抗现有的最先进的攻击。然而，由于可视信息通常由外部来源提供，简单地将其包括在内将会产生新的安全问题。具体地说，我们提出了一种新的针对视觉感知FedRecs的中毒攻击，即图像中毒攻击，在FedRecs的训练过程中，攻击者可以逐渐修改上传的图像来操纵物品等级。此外，我们揭示了图像中毒攻击和模型中毒攻击之间的潜在合作将使视觉感知的FedRecs更容易被操纵。为了安全地使用视觉信息，我们在视觉感知的FedRecs中使用了扩散模型来净化每一张上传的图像并检测出恶意图像。



## **17. A theoretical basis for Blockchain Extractable Value**

区块链可提取价值的理论基础 cs.CR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02154v3) [paper-pdf](http://arxiv.org/pdf/2302.02154v3)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.

摘要: 可提取价值指的是对公共区块链的一大类经济攻击，在这些攻击中，有能力在区块中重新排序、丢弃或插入交易的对手可以从智能合约中“提取”价值。经验研究表明，主流协议，如分散交换，是这些攻击的大规模目标，对其用户和区块链网络造成有害影响。尽管这些袭击在现实世界中的影响越来越大，但理论基础仍然缺乏。基于区块链和智能合约的一般抽象模型，我们提出了可提取价值的形式理论。我们的理论是针对可提取值攻击的安全性证明的基础。



## **18. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2304.04033v3) [paper-pdf](http://arxiv.org/pdf/2304.04033v3)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **19. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09241v1) [paper-pdf](http://arxiv.org/pdf/2305.09241v1)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。



## **20. Ortho-ODE: Enhancing Robustness and of Neural ODEs against Adversarial Attacks**

正交法：增强神经网络对敌方攻击的稳健性和稳健性 cs.LG

Final project paper

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09179v1) [paper-pdf](http://arxiv.org/pdf/2305.09179v1)

**Authors**: Vishal Purohit

**Abstract**: Neural Ordinary Differential Equations (NODEs) probed the usage of numerical solvers to solve the differential equation characterized by a Neural Network (NN), therefore initiating a new paradigm of deep learning models with infinite depth. NODEs were designed to tackle the irregular time series problem. However, NODEs have demonstrated robustness against various noises and adversarial attacks. This paper is about the natural robustness of NODEs and examines the cause behind such surprising behaviour. We show that by controlling the Lipschitz constant of the ODE dynamics the robustness can be significantly improved. We derive our approach from Grownwall's inequality. Further, we draw parallels between contractivity theory and Grownwall's inequality. Experimentally we corroborate the enhanced robustness on numerous datasets - MNIST, CIFAR-10, and CIFAR 100. We also present the impact of adaptive and non-adaptive solvers on the robustness of NODEs.

摘要: 神经常微分方程组(节点)探索了用数值求解器来求解以神经网络(NN)为特征的微分方程，从而开创了一种无限深度深度学习模型的新范式。节点的设计是为了处理不规则的时间序列问题。然而，节点对各种噪声和敌意攻击表现出了健壮性。这篇论文是关于节点的自然健壮性的，并研究了这种令人惊讶的行为背后的原因。结果表明，通过控制常微分方程组的Lipschitz常数，可以显着提高系统的鲁棒性。我们的方法源于Grownwall的不等式性。此外，我们还比较了可逆性理论和Grownwall不等式之间的异同。在实验上，我们在许多数据集--MNIST、CIFAR-10和CIFAR 100上证实了增强的稳健性。我们还给出了自适应和非自适应求解器对节点健壮性的影响。



## **21. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

决选：改进了针对数据中毒攻击的可证明防御 cs.LG

Accepted to ICML 2023

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02300v3) [paper-pdf](http://arxiv.org/pdf/2302.02300v3)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.

摘要: 在数据中毒攻击中，对手试图通过添加、修改或删除训练数据中的样本来更改模型的预测。最近，已经提出了基于集成的方法来获得针对数据中毒的可证明防御，其中预测是通过在多个基础模型上获得多数票来完成的。在这项工作中，我们表明，仅仅在集成防御中考虑多数投票是浪费的，因为它没有有效地利用基本模型的Logits层中的可用信息。相反，我们提出了决选选举(ROE)，这是一种基于基础模型之间的两轮选举的新型聚合方法：在第一轮中，模型投票选择他们喜欢的类，然后在第一轮中前两个类之间举行第二次决选。在此基础上，提出了基于深度划分聚集(DPA)和有限聚集(FA)的DPA+ROE和FA+ROE防御方法。我们在MNIST、CIFAR-10和GTSRB上对我们的方法进行了评估，并在认证的准确性方面获得了高达3%-4%的改进。此外，通过在增强版本的DPA上应用ROE，与当前最先进的版本相比，我们获得了约12%-27%的改进，从而建立了针对数据中毒的(按点)经认证的新的最先进的健壮性。在许多情况下，我们的方法优于最先进的方法，即使在使用32倍的计算能力时也是如此。



## **22. Training Neural Networks without Backpropagation: A Deeper Dive into the Likelihood Ratio Method**

无反向传播训练神经网络：对似然比方法的深入探讨 cs.LG

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08960v1) [paper-pdf](http://arxiv.org/pdf/2305.08960v1)

**Authors**: Jinyang Jiang, Zeliang Zhang, Chenliang Xu, Zhaofei Yu, Yijie Peng

**Abstract**: Backpropagation (BP) is the most important gradient estimation method for training neural networks in deep learning. However, the literature shows that neural networks trained by BP are vulnerable to adversarial attacks. We develop the likelihood ratio (LR) method, a new gradient estimation method, for training a broad range of neural network architectures, including convolutional neural networks, recurrent neural networks, graph neural networks, and spiking neural networks, without recursive gradient computation. We propose three methods to efficiently reduce the variance of the gradient estimation in the neural network training process. Our experiments yield numerical results for training different neural networks on several datasets. All results demonstrate that the LR method is effective for training various neural networks and significantly improves the robustness of the neural networks under adversarial attacks relative to the BP method.

摘要: 反向传播(BP)是深度学习中训练神经网络最重要的梯度估计方法。然而，文献表明，BP训练的神经网络很容易受到对手的攻击。我们发展了一种新的梯度估计方法-似然比方法，用于训练包括卷积神经网络、递归神经网络、图神经网络和尖峰神经网络在内的广泛的神经网络结构，而不需要递归梯度计算。在神经网络训练过程中，我们提出了三种有效降低梯度估计方差的方法。我们的实验给出了在几个数据集上训练不同神经网络的数值结果。结果表明，与BP方法相比，LR方法对训练各种神经网络是有效的，并且显著提高了神经网络在对抗攻击下的鲁棒性。



## **23. Attacking Perceptual Similarity Metrics**

攻击感知相似性度量 cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.

摘要: 知觉相似性指标已逐渐与人类对知觉相似性的判断更加相关；然而，尽管最近取得了进展，添加了不可察觉的失真仍然可能损害这些指标。在我们的研究中，我们系统地检查了这些度量对不可察觉的对抗性扰动的稳健性。在两个失真图像和一个参考图像的两种选择强迫选择实验设计之后，我们通过对抗性攻击使失真图像更接近参考图像，直到度量颠倒其判断。我们首先表明，我们研究中的所有指标都容易受到常见的对抗性攻击(如FGSM、PGD和单像素攻击)产生的扰动的影响。接下来，我们在白盒环境中使用基于空间变换的对抗性扰动(StAdv)来攻击广泛采用的LPIPS度量，以创建可以有效地转换到黑盒环境中的其他相似性度量的对抗性示例。我们还将空间攻击stAdv与pgd($\ell_\inty$-bound)攻击相结合，以增加可转移性，并使用这些对抗性示例来对传统度量和最近开发的度量的健壮性进行基准测试。我们的基准为讨论和进一步研究度量对不可察觉的对抗性扰动的健壮性提供了一个很好的起点。



## **24. Defending Against Misinformation Attacks in Open-Domain Question Answering**

开放领域答疑中防误报攻击的研究 cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.

摘要: 最近在开放领域问答(ODQA)方面的研究表明，搜索集合的敌意中毒会导致产生式系统的准确率大幅下降。然而，几乎没有工作提出了防御这些攻击的方法。要做到这一点，我们依赖于这样一种直觉，即大型语料库中往往存在冗余信息。为了找到它，我们引入了一种方法，使用查询增强来搜索一组不同的段落，这些段落可以回答原始问题，但不太可能被毒化。我们通过设计一种新的置信度方法将这些新的段落集成到模型中，将预测的答案与其在检索到的上下文中的表现进行比较(我们称其为来自答案冗余的置信度)，即CAR。这些方法结合在一起，提供了一种简单但有效的方法来防御中毒攻击，在不同级别的数据中毒/知识冲突中提供了近20%的精确匹配。



## **25. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

不可察觉和可转移对抗性攻击的扩散模型 cs.CV

Code Page: https://github.com/WindVChen/DiffAttack

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08192v1) [paper-pdf](http://arxiv.org/pdf/2305.08192v1)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an additional recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into adversarial attack field. Extensive experiments on various model structures (including CNNs, Transformers, MLPs) and defense methods have demonstrated our superiority over other attack methods.

摘要: 许多现有的对抗性攻击在图像RGB空间上产生$L_p$-范数扰动。尽管在可转移性和攻击成功率方面取得了一些成就，但制作的对抗性例子很容易被人眼察觉。对于视觉不可感知性，最近的一些工作探索了没有$L_p$-范数约束的无限攻击，但缺乏攻击黑盒模型的可转移性。在这项工作中，我们提出了一种新的不可察觉和可转移的攻击，利用扩散模型的生成性和区分性。具体地说，我们不是在像素空间中直接操作，而是在扩散模型的潜在空间中制造扰动。与设计良好的内容保持结构相结合，我们可以生成嵌入语义线索的人类不敏感的扰动。为了获得更好的可转移性，我们通过将扩散模型的注意力从目标区域转移开，进一步欺骗了扩散模型，该模型可以被视为一个额外的识别代理。据我们所知，我们提出的DiffAttack方法是第一个将扩散模型引入对抗性攻击领域的方法。在各种模型结构(包括CNN、Transformers、MLP)和防御方法上的广泛实验证明了该攻击方法相对于其他攻击方法的优越性。



## **26. Watermarking Text Generated by Black-Box Language Models**

黑盒语言模型生成的文本水印 cs.CL

Code will be available at  https://github.com/Kiode/Text_Watermark_Language_Models

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08883v1) [paper-pdf](http://arxiv.org/pdf/2305.08883v1)

**Authors**: Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, Nenghai Yu

**Abstract**: LLMs now exhibit human-like skills in various fields, leading to worries about misuse. Thus, detecting generated text is crucial. However, passive detection methods are stuck in domain specificity and limited adversarial robustness. To achieve reliable detection, a watermark-based method was proposed for white-box LLMs, allowing them to embed watermarks during text generation. The method involves randomly dividing the model vocabulary to obtain a special list and adjusting the probability distribution to promote the selection of words in the list. A detection algorithm aware of the list can identify the watermarked text. However, this method is not applicable in many real-world scenarios where only black-box language models are available. For instance, third-parties that develop API-based vertical applications cannot watermark text themselves because API providers only supply generated text and withhold probability distributions to shield their commercial interests. To allow third-parties to autonomously inject watermarks into generated text, we develop a watermarking framework for black-box language model usage scenarios. Specifically, we first define a binary encoding function to compute a random binary encoding corresponding to a word. The encodings computed for non-watermarked text conform to a Bernoulli distribution, wherein the probability of a word representing bit-1 being approximately 0.5. To inject a watermark, we alter the distribution by selectively replacing words representing bit-0 with context-based synonyms that represent bit-1. A statistical test is then used to identify the watermark. Experiments demonstrate the effectiveness of our method on both Chinese and English datasets. Furthermore, results under re-translation, polishing, word deletion, and synonym substitution attacks reveal that it is arduous to remove the watermark without compromising the original semantics.

摘要: LLM现在各个领域都展示了类似人类的技能，这导致了人们对滥用的担忧。因此，检测生成的文本至关重要。然而，被动检测方法停留在领域特异性和有限的对抗稳健性。为了实现可靠的检测，提出了一种基于水印的白盒LLMS检测方法，允许白盒LLMS在文本生成过程中嵌入水印。该方法对模型词汇进行随机划分，得到一个特殊的词汇表，并通过调整概率分布来促进词汇表中的词的选择。知道该列表的检测算法可以识别加水印的文本。然而，这种方法不适用于许多实际场景，因为只有黑盒语言模型可用。例如，开发基于API的垂直应用程序的第三方不能自己为文本添加水印，因为API提供商只提供生成的文本，并扣留概率分布以保护他们的商业利益。为了允许第三方自主地向生成的文本中注入水印，我们开发了一个适用于黑盒语言模型使用场景的水印框架。具体地说，我们首先定义一个二进制编码函数来计算对应于一个单词的随机二进制编码。为未加水印的文本计算的编码符合伯努利分布，其中表示位1的字的概率约为0.5。为了注入水印，我们通过有选择地将表示位0的单词替换为表示位1的基于上下文的同义词来改变分布。然后使用统计测试来识别水印。在中文和英文数据集上的实验证明了该方法的有效性。此外，在重译、润色、单词删除和同义词替换攻击下的结果表明，在不损害原始语义的情况下去除水印是困难的。



## **27. Improving Defensive Distillation using Teacher Assistant**

利用助教提高防守蒸馏能力 cs.CV

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08076v1) [paper-pdf](http://arxiv.org/pdf/2305.08076v1)

**Authors**: Maniratnam Mandal, Suna Gao

**Abstract**: Adversarial attacks pose a significant threat to the security and safety of deep neural networks being applied to modern applications. More specifically, in computer vision-based tasks, experts can use the knowledge of model architecture to create adversarial samples imperceptible to the human eye. These attacks can lead to security problems in popular applications such as self-driving cars, face recognition, etc. Hence, building networks which are robust to such attacks is highly desirable and essential. Among the various methods present in literature, defensive distillation has shown promise in recent years. Using knowledge distillation, researchers have been able to create models robust against some of those attacks. However, more attacks have been developed exposing weakness in defensive distillation. In this project, we derive inspiration from teacher assistant knowledge distillation and propose that introducing an assistant network can improve the robustness of the distilled model. Through a series of experiments, we evaluate the distilled models for different distillation temperatures in terms of accuracy, sensitivity, and robustness. Our experiments demonstrate that the proposed hypothesis can improve robustness in most cases. Additionally, we show that multi-step distillation can further improve robustness with very little impact on model accuracy.

摘要: 对抗性攻击对应用于现代应用的深度神经网络的安全性和安全性构成了严重威胁。更具体地说，在基于计算机视觉的任务中，专家可以利用模型体系结构的知识来创建人眼看不到的对抗性样本。这些攻击可能会导致自动驾驶汽车、人脸识别等热门应用中的安全问题。因此，构建对此类攻击具有健壮性的网络是非常必要的。在文献中出现的各种方法中，防御性蒸馏在最近几年显示出了希望。使用知识蒸馏，研究人员已经能够创建针对其中一些攻击的稳健模型。然而，更多的攻击暴露了防守蒸馏的弱点。在本项目中，我们从教师辅助知识提取中得到启发，并提出引入辅助网络可以提高提取模型的健壮性。通过一系列的实验，我们评估了不同蒸馏温度下的蒸馏模型的准确性、灵敏度和稳健性。我们的实验表明，该假设在大多数情况下都能提高稳健性。此外，我们还表明，多步精馏可以在对模型精度影响很小的情况下进一步提高稳健性。



## **28. DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack**

DNN-Defender：一种DRAM深度神经网络对抗性权重攻击防御机制 cs.CR

10 pages, 11 figures

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08034v1) [paper-pdf](http://arxiv.org/pdf/2305.08034v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring additional hardware overhead.

摘要: 随着深度学习在许多安全敏感领域的部署，机器学习的安全性正变得越来越重要。最近的研究表明，攻击者可以利用系统级技术，利用DRAM的RowHammer漏洞来确定并精确地翻转深度神经网络(DNN)模型中的位，以影响推理精度。现有的防御机制是基于软件的，例如需要昂贵的训练开销的权重重建或性能下降。另一方面，通用的基于硬件的以受害者/攻击者为中心的机制增加了昂贵的硬件开销，并保持了受害者和攻击者行之间的空间连接。在本文中，我们提出了第一个基于DRAM的针对量化DNN的以受害者为中心的防御机制，称为DNN-Defender，它利用DRAM内交换的潜力来抵御目标位翻转攻击。我们的结果表明，DNN-Defender可以提供高级别的保护，将目标RowHammer攻击的性能降低到随机攻击级别。此外，拟议的防御在CIFAR-10和ImageNet数据集上没有精度下降，不需要任何软件培训或产生额外的硬件开销。



## **29. On enhancing the robustness of Vision Transformers: Defensive Diffusion**

增强视觉变形金刚的稳健性：防御性扩散 cs.CV

Our code is publicly available at  https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08031v1) [paper-pdf](http://arxiv.org/pdf/2305.08031v1)

**Authors**: Raza Imam, Muhammad Huzaifa, Mohammed El-Amine Azz

**Abstract**: Privacy and confidentiality of medical data are of utmost importance in healthcare settings. ViTs, the SOTA vision model, rely on large amounts of patient data for training, which raises concerns about data security and the potential for unauthorized access. Adversaries may exploit vulnerabilities in ViTs to extract sensitive patient information and compromising patient privacy. This work address these vulnerabilities to ensure the trustworthiness and reliability of ViTs in medical applications. In this work, we introduced a defensive diffusion technique as an adversarial purifier to eliminate adversarial noise introduced by attackers in the original image. By utilizing the denoising capabilities of the diffusion model, we employ a reverse diffusion process to effectively eliminate the adversarial noise from the attack sample, resulting in a cleaner image that is then fed into the ViT blocks. Our findings demonstrate the effectiveness of the diffusion model in eliminating attack-agnostic adversarial noise from images. Additionally, we propose combining knowledge distillation with our framework to obtain a lightweight student model that is both computationally efficient and robust against gray box attacks. Comparison of our method with a SOTA baseline method, SEViT, shows that our work is able to outperform the baseline. Extensive experiments conducted on a publicly available Tuberculosis X-ray dataset validate the computational efficiency and improved robustness achieved by our proposed architecture.

摘要: 医疗数据的隐私和机密性在医疗保健环境中至关重要。VITS是SOTA的视觉模型，它依赖于大量的患者数据进行培训，这引发了人们对数据安全和未经授权访问的可能性的担忧。攻击者可能会利用VITS中的漏洞来提取敏感的患者信息，从而危及患者隐私。这项工作解决了这些漏洞，以确保VITS在医疗应用中的可信性和可靠性。在这项工作中，我们引入了一种防御扩散技术作为对抗性净化器来消除攻击者在原始图像中引入的对抗性噪声。通过利用扩散模型的去噪能力，我们采用反向扩散过程来有效地消除攻击样本中的对抗性噪声，从而得到更干净的图像，然后将其送入VIT块。我们的发现证明了扩散模型在消除图像中与攻击无关的对抗性噪声方面的有效性。此外，我们建议将知识提炼与我们的框架相结合，以获得一个轻量级的学生模型，该模型在计算效率上是有效的，并且对灰盒攻击具有健壮性。我们的方法与SOTA基线方法SEViT的比较表明，我们的工作能够超过基线。在公开可用的结核病X光数据集上进行的大量实验验证了我们所提出的体系结构的计算效率和提高的稳健性。



## **30. Detection and Mitigation of Byzantine Attacks in Distributed Training**

分布式训练中拜占庭攻击的检测与缓解 cs.LG

21 pages, 17 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2208.08085v4) [paper-pdf](http://arxiv.org/pdf/2208.08085v4)

**Authors**: Konstantinos Konstantinidis, Namrata Vaswani, Aditya Ramamoorthy

**Abstract**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. We also show the convergence of our method to the optimal point under common assumptions and settings considered in literature. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.

摘要: 过多的现代机器学习任务需要利用大规模分布式集群作为培训管道的关键组成部分。然而，工作者节点的异常拜占庭行为会破坏训练，影响推理的质量。此类行为可归因于无意的系统故障或精心策划的攻击；因此，某些节点可能会向协调训练的参数服务器(PS)返回任意结果。最近的工作考虑了广泛的攻击模型，并探索了稳健的聚集和/或计算冗余来纠正扭曲的梯度。在这项工作中，我们考虑了从强到强的攻击模型：$q$全知的对手，完全了解防御协议，可以从一个迭代到另一个迭代变化；$q$随机选择的对手，合谋能力有限，一次只有几个迭代改变。我们的算法依赖于冗余的任务分配以及对敌对行为的检测。我们还证明了在文献中常见的假设和设置下，我们的方法收敛到最优点。对于强攻击，我们展示了与以前最先进的技术相比，扭曲梯度的比例降低了16%-99%。我们在CIFAR-10数据集上的TOP-1分类精度结果显示，在最复杂的攻击下，与最先进的方法相比，准确率(在强和弱场景下平均)提高了25%。



## **31. Decision-based iterative fragile watermarking for model integrity verification**

用于模型完整性验证的基于决策的迭代脆弱水印 cs.CR

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2305.09684v1) [paper-pdf](http://arxiv.org/pdf/2305.09684v1)

**Authors**: Zhaoxia Yin, Heng Yin, Hang Su, Xinpeng Zhang, Zhenzhe Gao

**Abstract**: Typically, foundation models are hosted on cloud servers to meet the high demand for their services. However, this exposes them to security risks, as attackers can modify them after uploading to the cloud or transferring from a local system. To address this issue, we propose an iterative decision-based fragile watermarking algorithm that transforms normal training samples into fragile samples that are sensitive to model changes. We then compare the output of sensitive samples from the original model to that of the compromised model during validation to assess the model's completeness.The proposed fragile watermarking algorithm is an optimization problem that aims to minimize the variance of the predicted probability distribution outputed by the target model when fed with the converted sample.We convert normal samples to fragile samples through multiple iterations. Our method has some advantages: (1) the iterative update of samples is done in a decision-based black-box manner, relying solely on the predicted probability distribution of the target model, which reduces the risk of exposure to adversarial attacks, (2) the small-amplitude multiple iterations approach allows the fragile samples to perform well visually, with a PSNR of 55 dB in TinyImageNet compared to the original samples, (3) even with changes in the overall parameters of the model of magnitude 1e-4, the fragile samples can detect such changes, and (4) the method is independent of the specific model structure and dataset. We demonstrate the effectiveness of our method on multiple models and datasets, and show that it outperforms the current state-of-the-art.

摘要: 通常，基础模型托管在云服务器上，以满足对其服务的高需求。然而，这会使它们面临安全风险，因为攻击者可以在上传到云或从本地系统传输后修改它们。为了解决这个问题，我们提出了一种基于迭代判决的脆弱水印算法，将正常的训练样本转换为对模型变化敏感的脆弱样本。然后在验证过程中比较原始模型和受损模型的敏感样本的输出，以评估模型的完备性。脆弱水印算法是一个优化问题，旨在最小化目标模型输入转换后的样本时预测概率分布的方差，通过多次迭代将正常样本转换为脆弱样本。我们的方法有一些优点：(1)样本的迭代更新是以基于决策的黑盒方式进行的，仅依赖于目标模型的预测概率分布，从而降低了暴露于对抗性攻击的风险；(2)小幅度多次迭代方法允许脆弱样本在视觉上表现良好，与原始样本相比，TinyImageNet的PSNR为55dB；(3)即使1e-4量级的模型的总体参数发生变化，脆弱样本也可以检测到这种变化；(4)该方法独立于特定的模型结构和数据集。我们在多个模型和数据集上验证了我们的方法的有效性，并表明它的性能优于目前最先进的方法。



## **32. Quantum Lock: A Provable Quantum Communication Advantage**

量子锁：一种可证明的量子通信优势 quant-ph

47 pages, 13 figures

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2110.09469v4) [paper-pdf](http://arxiv.org/pdf/2110.09469v4)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstract**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.

摘要: 物理不可克隆函数(PUF)通过利用固有的物理随机性为物理实体提供唯一指纹。高等人。讨论了当前大多数PUF对复杂的基于机器学习的攻击的脆弱性。我们通过将经典的PUF和现有的量子通信技术相结合来解决这个问题。具体地说，本文提出了一种可证明安全的PUF的通用设计，称为混合锁定PUF(HLPUF)，为保护经典PUF提供了一种实用的解决方案。HLPUF使用经典的PUF(CPUF)，并将输出编码为非正交的量子态，以向任何对手隐藏底层CPUF的结果。在这里，我们引入量子锁来保护HLPUF免受任何一般对手的攻击。非正交量子态的不可分辨特性，加上量子锁定技术，阻止了攻击者访问CPUF的结果。此外，我们证明了通过利用量子态的非经典属性，HLPUF允许服务器重用挑战-响应对来进行进一步的客户端认证。这一结果为长期运行基于PUF的客户端身份验证提供了一个有效的解决方案，同时在服务器端维护一个小型的挑战-响应对数据库。随后，我们通过使用可访问的真实CPUF来实例化HLPUF设计来支持我们的理论贡献。我们使用最优经典机器学习攻击来伪造CPUF和HLPUF，并证明了我们的构造数值模拟中的安全漏洞。



## **33. Two-in-One: A Model Hijacking Attack Against Text Generation Models**

二合一：一种针对文本生成模型的模型劫持攻击 cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07406v1) [paper-pdf](http://arxiv.org/pdf/2305.07406v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Machine learning has progressed significantly in various applications ranging from face recognition to text generation. However, its success has been accompanied by different attacks. Recently a new attack has been proposed which raises both accountability and parasitic computing risks, namely the model hijacking attack. Nevertheless, this attack has only focused on image classification tasks. In this work, we broaden the scope of this attack to include text generation and classification models, hence showing its broader applicability. More concretely, we propose a new model hijacking attack, Ditto, that can hijack different text classification tasks into multiple generation ones, e.g., language translation, text summarization, and language modeling. We use a range of text benchmark datasets such as SST-2, TweetEval, AGnews, QNLI, and IMDB to evaluate the performance of our attacks. Our results show that by using Ditto, an adversary can successfully hijack text generation models without jeopardizing their utility.

摘要: 机器学习在从人脸识别到文本生成的各种应用中都取得了显著的进展。然而，它的成功伴随着不同的攻击。最近提出了一种新的攻击，它同时增加了可追究性和寄生计算的风险，即模型劫持攻击。尽管如此，这次攻击只集中在图像分类任务上。在这项工作中，我们扩大了该攻击的范围，将文本生成和分类模型包括在内，从而显示了其更广泛的适用性。更具体地说，我们提出了一种新的劫持攻击模型Ditto，该模型可以将不同的文本分类任务劫持为多个世代任务，例如语言翻译、文本摘要和语言建模。我们使用一系列文本基准数据集，如SST-2、TweetEval、AgNews、QNLI和IMDB来评估我们的攻击性能。我们的结果表明，通过使用Ditto，攻击者可以在不损害其实用性的情况下成功劫持文本生成模型。



## **34. Novel bribery mining attacks in the bitcoin system and the bribery miner's dilemma**

比特币系统中的新型贿赂挖掘攻击与贿赂挖掘者的困境 cs.GT

26 pages, 16 figures, 3 tables

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07381v1) [paper-pdf](http://arxiv.org/pdf/2305.07381v1)

**Authors**: Junjie Hu, Chunxiang Xu, Zhe Jiang, Jiwu Cao

**Abstract**: Mining attacks allow adversaries to obtain a disproportionate share of the mining reward by deviating from the honest mining strategy in the Bitcoin system. Among them, the most well-known are selfish mining (SM), block withholding (BWH), fork after withholding (FAW) and bribery mining. In this paper, we propose two novel mining attacks: bribery semi-selfish mining (BSSM) and bribery stubborn mining (BSM). Both of them can increase the relative extra reward of the adversary and will make the target bribery miners suffer from the bribery miner dilemma. All targets earn less under the Nash equilibrium. For each target, their local optimal strategy is to accept the bribes. However, they will suffer losses, comparing with denying the bribes. Furthermore, for all targets, their global optimal strategy is to deny the bribes. Quantitative analysis and simulation have been verified our theoretical analysis. We propose practical measures to mitigate more advanced mining attack strategies based on bribery mining, and provide new ideas for addressing bribery mining attacks in the future. However, how to completely and effectively prevent these attacks is still needed on further research.

摘要: 挖矿攻击允许对手通过偏离比特币系统中诚实的挖矿策略，获得不成比例的挖矿回报。其中，最广为人知的是自私挖矿(SM)、集体扣留(BWH)、扣后分叉(FAW)和贿赂挖矿。本文提出了两种新的挖掘攻击：贿赂半自私挖掘(BSSM)和贿赂顽固挖掘(BSM)。两者都能增加对手的相对额外报酬，使受贿目标矿工陷入受贿矿工困境。在纳什均衡下，所有目标的收入都较低。对于每个目标，他们在当地的最优策略是收受贿赂。然而，与拒绝贿赂相比，他们将蒙受损失。此外，对于所有目标来说，他们的全球最佳策略是否认贿赂。定量分析和仿真验证了我们的理论分析。提出了缓解基于贿赂挖掘的更高级挖掘攻击策略的实用措施，为未来应对贿赂挖掘攻击提供了新的思路。然而，如何完全有效地防范这些攻击还需要进一步的研究。



## **35. Efficient Search of Comprehensively Robust Neural Architectures via Multi-fidelity Evaluation**

基于多保真度评价的综合稳健神经网络高效搜索 cs.CV

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07308v1) [paper-pdf](http://arxiv.org/pdf/2305.07308v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Neural architecture search (NAS) has emerged as one successful technique to find robust deep neural network (DNN) architectures. However, most existing robustness evaluations in NAS only consider $l_{\infty}$ norm-based adversarial noises. In order to improve the robustness of DNN models against multiple types of noises, it is necessary to consider a comprehensive evaluation in NAS for robust architectures. But with the increasing number of types of robustness evaluations, it also becomes more time-consuming to find comprehensively robust architectures. To alleviate this problem, we propose a novel efficient search of comprehensively robust neural architectures via multi-fidelity evaluation (ES-CRNA-ME). Specifically, we first search for comprehensively robust architectures under multiple types of evaluations using the weight-sharing-based NAS method, including different $l_{p}$ norm attacks, semantic adversarial attacks, and composite adversarial attacks. In addition, we reduce the number of robustness evaluations by the correlation analysis, which can incorporate similar evaluations and decrease the evaluation cost. Finally, we propose a multi-fidelity online surrogate during optimization to further decrease the search cost. On the basis of the surrogate constructed by low-fidelity data, the online high-fidelity data is utilized to finetune the surrogate. Experiments on CIFAR10 and CIFAR100 datasets show the effectiveness of our proposed method.

摘要: 神经体系结构搜索(NAS)已成为一种发现健壮的深度神经网络(DNN)体系结构的成功技术。然而，现有的NAS健壮性评估大多只考虑基于$L范数的对抗性噪声。为了提高DNN模型对多种类型噪声的鲁棒性，有必要考虑在NAS中对健壮体系结构进行综合评估。但随着健壮性评估类型的增加，寻找全面健壮的体系结构也变得更加耗时。为了缓解这一问题，我们提出了一种新的高效的通过多保真度评估(ES-CRNA-ME)来寻找全面稳健的神经结构的方法。具体地说，我们首先使用基于权重共享的NAS方法在多种评估类型下搜索全面的健壮性体系结构，包括不同的$L_{p}$范数攻击、语义对抗攻击和复合对抗攻击。另外，通过相关性分析，减少了健壮性评价的次数，可以融合相似评价，降低评价成本。最后，在优化过程中提出了一种多保真的在线代理，进一步降低了搜索成本。在低保真数据构建代理的基础上，利用在线高保真数据对代理进行微调。在CIFAR10和CIFAR100数据集上的实验表明了该方法的有效性。



## **36. Parameter identifiability of a deep feedforward ReLU neural network**

深度前馈RELU神经网络的参数可辨识性 math.ST

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2112.12982v2) [paper-pdf](http://arxiv.org/pdf/2112.12982v2)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstract**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.

摘要: 由于知道神经网络在输入空间的子集上的功能，人们能够恢复神经网络的参数--权重和偏差--的可能性可能是诅咒，也可能是祝福，具体取决于具体情况。一方面，恢复参数可以进行更好的对抗性攻击，还可能泄露用于构建网络的数据集的敏感信息。另一方面，如果可以恢复网络的参数，就可以保证用户可以解释潜在空间中的特征。它还为获得对网络性能的正式保证提供了基础。因此，重要的是要确定其参数可以识别和参数不能识别的网络的特征。本文给出了一个深度全连通的前馈RELU神经网络的一组条件，在该条件下，网络的参数可以从它在输入空间的一个子集上实现的函数中唯一地识别出来--模置换和正重标度。



## **37. Adversarial Security and Differential Privacy in mmWave Beam Prediction in 6G networks**

6G网络毫米波波束预测中的对抗安全和差分保密 cs.CR

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.09679v1) [paper-pdf](http://arxiv.org/pdf/2305.09679v1)

**Authors**: Ghanta Sai Krishna, Kundrapu Supriya, Sanskar Singh, Sabur Baidya

**Abstract**: In the forthcoming era of 6G, the mmWave communication is envisioned to be used in dense user scenarios with high bandwidth requirements, that necessitate efficient and accurate beam prediction. Machine learning (ML) based approaches are ushering as a critical solution for achieving such efficient beam prediction for 6G mmWave communications. However, most contemporary ML classifiers are quite susceptible to adversarial inputs. Attackers can easily perturb the methodology through noise addition in the model itself. To mitigate this, the current work presents a defensive mechanism for attenuating the adversarial attacks against projected ML-based models for mmWave beam anticipation by incorporating adversarial training. Furthermore, as training 6G mmWave beam prediction model necessitates the use of large and comprehensive datasets that could include sensitive information regarding the user's location, differential privacy (DP) has been introduced as a technique to preserve the confidentiality of the information by purposefully adding a low sensitivity controlled noise in the datasets. It ensures that even if the information about a user location could be retrieved, the attacker would have no means to determine whether the information is significant or meaningless. With ray-tracing simulations for various outdoor and indoor scenarios, we illustrate the advantage of our proposed novel framework in terms of beam prediction accuracy and effective achievable rate while ensuring the security and privacy in communications.

摘要: 在即将到来的6G时代，毫米波通信被设想用于高带宽要求的密集用户场景，这就需要高效和准确的波束预测。基于机器学习(ML)的方法正在成为实现6G毫米波通信如此高效的波束预测的关键解决方案。然而，大多数当代ML量词都很容易受到对抗性输入的影响。攻击者可以很容易地通过在模型本身中添加噪声来扰乱该方法。为了缓解这一问题，当前的工作提出了一种防御机制，通过结合对抗性训练来减弱针对基于ML的毫米波波束预测投影模型的对抗性攻击。此外，由于训练6G毫米波波束预测模型需要使用可能包括关于用户位置的敏感信息的大型和全面的数据集，因此引入了差分隐私(DP)作为一种技术，通过在数据集中有目的地添加低灵敏度受控噪声来保护信息的机密性。它确保即使可以检索到有关用户位置的信息，攻击者也无法确定该信息是重要的还是毫无意义的。通过对各种室外和室内场景的光线跟踪仿真，说明了该框架在保证通信安全性和保密性的同时，在波束预测精度和有效可达速率方面的优势。



## **38. Physical-layer Adversarial Robustness for Deep Learning-based Semantic Communications**

基于深度学习的语义通信物理层对抗健壮性 eess.SP

17 pages, 28 figures, accepted by IEEE jsac

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07220v1) [paper-pdf](http://arxiv.org/pdf/2305.07220v1)

**Authors**: Guoshun Nan, Zhichun Li, Jinli Zhai, Qimei Cui, Gong Chen, Xin Du, Xuefei Zhang, Xiaofeng Tao, Zhu Han, Tony Q. S. Quek

**Abstract**: End-to-end semantic communications (ESC) rely on deep neural networks (DNN) to boost communication efficiency by only transmitting the semantics of data, showing great potential for high-demand mobile applications. We argue that central to the success of ESC is the robust interpretation of conveyed semantics at the receiver side, especially for security-critical applications such as automatic driving and smart healthcare. However, robustifying semantic interpretation is challenging as ESC is extremely vulnerable to physical-layer adversarial attacks due to the openness of wireless channels and the fragileness of neural models. Toward ESC robustness in practice, we ask the following two questions: Q1: For attacks, is it possible to generate semantic-oriented physical-layer adversarial attacks that are imperceptible, input-agnostic and controllable? Q2: Can we develop a defense strategy against such semantic distortions and previously proposed adversaries? To this end, we first present MobileSC, a novel semantic communication framework that considers the computation and memory efficiency in wireless environments. Equipped with this framework, we propose SemAdv, a physical-layer adversarial perturbation generator that aims to craft semantic adversaries over the air with the abovementioned criteria, thus answering the Q1. To better characterize the realworld effects for robust training and evaluation, we further introduce a novel adversarial training method SemMixed to harden the ESC against SemAdv attacks and existing strong threats, thus answering the Q2. Extensive experiments on three public benchmarks verify the effectiveness of our proposed methods against various physical adversarial attacks. We also show some interesting findings, e.g., our MobileSC can even be more robust than classical block-wise communication systems in the low SNR regime.

摘要: 端到端语义通信(ESC)依靠深度神经网络(DNN)来提高通信效率，只传输数据的语义，在高需求的移动应用中显示出巨大的潜力。我们认为，ESC成功的核心是在接收方对所传达的语义进行强有力的解释，特别是对于自动驾驶和智能医疗等安全关键型应用。然而，由于无线信道的开放性和神经模型的脆弱性，ESC极易受到物理层的敌意攻击，因此增强语义解释是具有挑战性的。对于ESC在实践中的健壮性，我们提出了以下两个问题：Q1：对于攻击，是否有可能产生不可察觉的、与输入无关的、可控的、面向语义的物理层对抗性攻击？问题2：我们能否针对这种语义扭曲和之前提出的对手制定防御策略？为此，我们首先提出了一种新的语义通信框架MobileSC，该框架考虑了无线环境下的计算和存储效率。在此框架的基础上，我们提出了一种物理层敌意扰动生成器SemAdv，旨在利用上述标准在空中构建语义对手，从而回答问题1。为了更好地表征稳健训练和评估的真实效果，我们进一步引入了一种新的对抗性训练方法SemMixed来强化ESC对SemAdv攻击和现有的强威胁的攻击，从而回答了Q2。在三个公共基准上的大量实验验证了我们提出的方法对各种物理攻击的有效性。我们还发现了一些有趣的发现，例如，在低信噪比条件下，我们的MobileSC甚至可以比经典的分组通信系统更健壮。



## **39. Stratified Adversarial Robustness with Rejection**

具有拒绝的分层对抗健壮性 cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.01139v2) [paper-pdf](http://arxiv.org/pdf/2305.01139v2)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

摘要: 最近，对抗性地训练具有拒绝选项的分类器(也称为选择性分类器)以增强对抗性健壮性是一种新的兴趣。虽然拒绝在许多应用中可能会导致成本，但现有研究通常将零成本与拒绝扰动输入联系在一起，这可能导致拒绝许多可以正确分类的轻微扰动输入。在这项工作中，我们研究了分层拒绝环境下的具有拒绝的对抗性鲁棒分类，其中拒绝代价由拒绝损失函数来建模，拒绝损失函数在扰动幅度上单调地不增加。我们从理论上分析了分层拒绝的设置，并提出了一种新的防御方法--基于一致预测拒绝的对抗训练(CPR)--来构建一个健壮的选择性分类器。在图像数据集上的实验表明，该方法在强自适应攻击下的性能明显优于已有方法。例如，在CIFAR-10上，CPR在看得见和看不见的攻击下都将总的稳健损失(针对不同的拒绝损失)减少了至少7.3%。



## **40. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

提高多重攻击下的高光谱对抗健壮性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2210.16346v4) [paper-pdf](http://arxiv.org/pdf/2210.16346v4)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.

摘要: 对高光谱图像进行分类的语义分割模型容易受到敌意例子的影响。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法的性能会下降。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。



## **41. Untargeted Near-collision Attacks in Biometric Recognition**

生物特征识别中的无目标近碰撞攻击 cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2304.01580v2) [paper-pdf](http://arxiv.org/pdf/2304.01580v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.

摘要: 生物识别系统可以在两种截然不同的模式下工作，即识别或验证。在第一种模式中，系统通过在所有用户的注册模板中搜索匹配项来识别个人。在第二种模式中，系统通过将新提供的模板与注册的模板进行比较来验证用户的身份声明。生物特征转换方案通常产生由加密方案更好地处理的二进制模板，并且比较基于泄露关于两个生物特征模板之间的相似性的信息的距离。实验确定的误匹配率和通过调整识别阈值确定的误不匹配率都定义了识别精度，从而决定了系统的安全性。就我们所知，很少有文献在信息泄露最小的情况下提供安全的形式处理，即与阈值比较的二进制结果。在本文中，我们依赖于概率建模来量化二进制模板的安全强度。我们研究了模板大小、数据库大小和阈值对近碰撞概率的影响。我们重点介绍了几种针对生物识别系统的非定向攻击，考虑到了天真和自适应的对手。有趣的是，这些攻击既可以在线上也可以离线发起，也可以在识别模式和验证模式下发起。我们通过一般提出的攻击讨论参数的选择。



## **42. Distracting Downpour: Adversarial Weather Attacks for Motion Estimation**

分散注意力的倾盆大雨：运动估计的对抗性天气攻击 cs.CV

This work is a direct extension of our extended abstract from  arXiv:2210.11242

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06716v1) [paper-pdf](http://arxiv.org/pdf/2305.06716v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks on motion estimation, or optical flow, optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, adverse weather conditions constitute a much more realistic threat scenario. Hence, in this work, we present a novel attack on motion estimation that exploits adversarially optimized particles to mimic weather effects like snowflakes, rain streaks or fog clouds. At the core of our attack framework is a differentiable particle rendering system that integrates particles (i) consistently over multiple time steps (ii) into the 3D space (iii) with a photo-realistic appearance. Through optimization, we obtain adversarial weather that significantly impacts the motion estimation. Surprisingly, methods that previously showed good robustness towards small per-pixel perturbations are particularly vulnerable to adversarial weather. At the same time, augmenting the training with non-optimized weather increases a method's robustness towards weather effects and improves generalizability at almost no additional cost.

摘要: 目前对运动估计或光流的敌意攻击，优化了每像素的小扰动，这在现实世界中不太可能出现。相比之下，不利的天气条件构成了更现实的威胁情景。因此，在这项工作中，我们提出了一种新颖的攻击运动估计的方法，该方法利用反向优化的粒子来模拟雪花、雨带或雾云等天气效果。在我们的攻击框架的核心是一个可区分的粒子渲染系统，它以照片般的外观将粒子(I)在多个时间步骤(Ii)一致地集成到3D空间(Iii)中。通过优化，得到对运动估计有显著影响的对抗性天气。令人惊讶的是，以前对每像素微小扰动表现出良好稳健性的方法特别容易受到恶劣天气的影响。同时，在不增加额外代价的情况下，用非优化的天气来增加训练，增加了方法对天气影响的鲁棒性，并提高了泛化能力。



## **43. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

超越模型：Android应用程序中对深度学习模型的数据预处理攻击 cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.03963v2) [paper-pdf](http://arxiv.org/pdf/2305.03963v2)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.

摘要: 近年来，深度学习模型的日益流行以及计算的优势，包括智能手机上的低延迟和带宽节省，导致了智能移动应用程序的出现，也被称为深度学习应用程序。然而，这种技术的发展也引起了一些安全问题，包括对抗性例子、模型窃取和数据中毒问题。现有的针对设备上DL模型的攻击和对策的研究主要集中在模型本身。然而，数据处理干扰对模型推理的影响还没有引起足够的重视。这种知识差距突出表明，需要进行更多的研究，以充分理解和解决与设备上模型的数据处理相关的安全问题。在本文中，我们介绍了一种基于数据处理的针对现实世界数字图书馆应用程序的攻击。特别是，我们的攻击可能会影响模型的性能和延迟，而不会影响DL应用的操作。为了证明我们攻击的有效性，我们对从Google Play收集的517个真实数字图书馆应用程序进行了实证研究。在320个使用MLkit的应用中，我们发现81.56%的应用可以被成功攻击。这些结果强调了数字图书馆应用程序开发人员从数据处理的角度意识到并采取行动保护设备上模型的重要性。



## **44. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

关于图神经扩散对拓扑扰动的稳健性 cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2209.07754v2) [paper-pdf](http://arxiv.org/pdf/2209.07754v2)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstract**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

摘要: 图上的神经扩散是一类新的图神经网络，近年来受到越来越多的关注。图神经偏微分方程组(PDE)在解决图神经网络(GNN)的常见障碍(如过光滑和瓶颈问题)方面的能力已被研究，但其对对手攻击的稳健性尚未得到研究。在这项工作中，我们研究了图神经偏微分方程的稳健性。我们的经验证明，与其他GNN相比，图神经PDE在本质上对拓扑扰动具有更强的鲁棒性。通过利用图的拓扑扰动下热半群的稳定性，我们提供了对这一现象的见解。我们讨论了各种图扩散算子，并将它们与现有的图神经偏微分方程联系起来。此外，我们还提出了一个通用的图神经偏微分方程框架，基于该框架可以定义一类新的健壮GNN。我们在几个基准数据集上验证了新模型取得了相当于最先进的性能。



## **45. Prevention of shoulder-surfing attacks using shifting condition using digraph substitution rules**

基于有向图替换规则的移位条件防止冲浪攻击 cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06549v1) [paper-pdf](http://arxiv.org/pdf/2305.06549v1)

**Authors**: Amanul Islam, Fazidah Othman, Nazmus Sakib, Hafiz Md. Hasan Babu

**Abstract**: Graphical passwords are implemented as an alternative scheme to replace alphanumeric passwords to help users to memorize their password. However, most of the graphical password systems are vulnerable to shoulder-surfing attack due to the usage of the visual interface. In this research, a method that uses shifting condition with digraph substitution rules is proposed to address shoulder-surfing attack problem. The proposed algorithm uses both password images and decoy images throughout the user authentication procedure to confuse adversaries from obtaining the password images via direct observation or watching from a recorded session. The pass-images generated by this suggested algorithm are random and can only be generated if the algorithm is fully understood. As a result, adversaries will have no clue to obtain the right password images to log in. A user study was undertaken to assess the proposed method's effectiveness to avoid shoulder-surfing attacks. The results of the user study indicate that the proposed approach can withstand shoulder-surfing attacks (both direct observation and video recording method).The proposed method was tested and the results showed that it is able to resist shoulder-surfing and frequency of occurrence analysis attacks. Moreover, the experience gained in this research can be pervaded the gap on the realm of knowledge of the graphical password.

摘要: 图形密码作为替代字母数字密码的替代方案来实施，以帮助用户记住他们的密码。然而，由于可视化界面的使用，大多数图形化密码系统都容易受到肩部冲浪攻击。针对肩部冲浪攻击问题，提出了一种基于有向图替换规则的移位条件攻击方法。该算法在用户认证过程中同时使用口令图像和诱骗图像，以迷惑攻击者通过直接观察或从记录的会话中观看来获得口令图像。该算法生成的通道图像是随机的，只有在充分理解该算法的情况下才能生成。因此，攻击者将没有任何线索来获取正确的密码图像来登录。进行了一项用户研究，以评估所提出的方法在避免肩部冲浪攻击方面的有效性。用户研究结果表明，该方法能够抵抗直接观察法和录像法的肩部冲浪攻击，并对该方法进行了测试，结果表明该方法能够抵抗肩部冲浪和频度分析攻击。此外，在本研究中获得的经验可以填补图形密码知识领域的空白。



## **46. Inter-frame Accelerate Attack against Video Interpolation Models**

针对视频插补模型的帧间加速攻击 cs.CV

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06540v1) [paper-pdf](http://arxiv.org/pdf/2305.06540v1)

**Authors**: Junpei Liao, Zhikai Chen, Liang Yi, Wenyuan Yang, Baoyuan Wu, Xiaochun Cao

**Abstract**: Deep learning based video frame interpolation (VIF) method, aiming to synthesis the intermediate frames to enhance video quality, have been highly developed in the past few years. This paper investigates the adversarial robustness of VIF models. We apply adversarial attacks to VIF models and find that the VIF models are very vulnerable to adversarial examples. To improve attack efficiency, we suggest to make full use of the property of video frame interpolation task. The intuition is that the gap between adjacent frames would be small, leading to the corresponding adversarial perturbations being similar as well. Then we propose a novel attack method named Inter-frame Accelerate Attack (IAA) that initializes the perturbation as the perturbation for the previous adjacent frame and reduces the number of attack iterations. It is shown that our method can improve attack efficiency greatly while achieving comparable attack performance with traditional methods. Besides, we also extend our method to video recognition models which are higher level vision tasks and achieves great attack efficiency.

摘要: 基于深度学习的视频帧内插方法(VIF)旨在合成中间帧以提高视频质量，在过去的几年中得到了很大的发展。本文研究了VIF模型的对抗稳健性。我们将对抗性攻击应用于VIF模型，发现VIF模型非常容易受到对抗性例子的攻击。为了提高攻击效率，我们建议充分利用视频帧内插任务的特性。直觉是，相邻帧之间的间隙会很小，导致相应的对抗性扰动也是相似的。然后，我们提出了一种新的攻击方法--帧间加速攻击(IAA)，该方法将扰动初始化为对前一相邻帧的扰动，并减少了攻击迭代的次数。实验结果表明，该方法在取得与传统方法相当的攻击性能的同时，大大提高了攻击效率。此外，我们还将我们的方法扩展到视频识别模型，这些模型是较高级别的视觉任务，具有很高的攻击效率。



## **47. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

20 pages, 6 figures

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2210.14410v2) [paper-pdf](http://arxiv.org/pdf/2210.14410v2)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



## **48. Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids**

用于电网虚假数据注入攻击检测的对抗性深度神经网络 cs.CR

This paper has been accepted by the the 32nd International Conference  on Computer Communications and Networks (ICCCN 2023)

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2102.09057v2) [paper-pdf](http://arxiv.org/pdf/2102.09057v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Stella Sun, Kevin Tomsovic, Hairong Qi

**Abstract**: False data injection attacks (FDIAs) pose a significant security threat to power system state estimation. To detect such attacks, recent studies have proposed machine learning (ML) techniques, particularly deep neural networks (DNNs). However, most of these methods fail to account for the risk posed by adversarial measurements, which can compromise the reliability of DNNs in various ML applications. In this paper, we present a DNN-based FDIA detection approach that is resilient to adversarial attacks. We first analyze several adversarial defense mechanisms used in computer vision and show their inherent limitations in FDIA detection. We then propose an adversarial-resilient DNN detection framework for FDIA that incorporates random input padding in both the training and inference phases. Our simulations, based on an IEEE standard power system, demonstrate that this framework significantly reduces the effectiveness of adversarial attacks while having a negligible impact on the DNNs' detection performance.

摘要: 虚假数据注入攻击(FDIA)对电力系统状态估计造成了严重的安全威胁。为了检测此类攻击，最近的研究提出了机器学习(ML)技术，特别是深度神经网络(DNN)。然而，这些方法中的大多数都没有考虑到对抗性测量所带来的风险，这可能会损害DNN在各种ML应用中的可靠性。在本文中，我们提出了一种基于DNN的对敌方攻击具有弹性的FDIA检测方法。我们首先分析了计算机视觉中使用的几种对抗性防御机制，并指出了它们在FDIA检测中的固有局限性。然后，我们提出了一种用于FDIA的对抗性DNN检测框架，该框架在训练和推理阶段都加入了随机输入填充。基于IEEE标准电力系统的仿真表明，该框架显著降低了对抗性攻击的有效性，而对DNN的检测性能影响可以忽略不计。



## **49. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

利用动态触发器对个人重新身份进行隐形后门攻击 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2211.10933v2) [paper-pdf](http://arxiv.org/pdf/2211.10933v2)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one or all-to-all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.

摘要: 近年来，身份识别技术发展迅速，在实际应用中得到了广泛的应用，但同时也带来了巨大的对抗性攻击风险。本文主要研究对深度Reid模型的后门攻击。现有的后门攻击方法遵循All-to-One或All-to-All攻击方案，其中测试集中的所有目标类都已在训练集中看到。然而，REID是一个更复杂的细粒度开集识别问题，其中测试集中的身份不包含在训练集中。因此，以前用于分类的后门攻击方法不适用于REID。为了改善这一问题，我们提出了一种新的全未知场景下对深度Reid的后门攻击，称为动态触发器不可见后门攻击(DT-IBA)。DT-IBA不需要从训练集中学习目标类的固定触发器，而是可以为任何未知身份动态生成新的触发器。具体地说，提出了一种身份散列网络，首先从参考图像中提取目标身份信息，然后通过图像隐写将这些身份信息注入到良性图像中。我们在基准数据集上广泛验证了提出的攻击的有效性和隐蔽性，并评估了几种防御方法对我们的攻击的有效性。



## **50. The Robustness of Computer Vision Models against Common Corruptions: a Survey**

计算机视觉模型对常见腐败的稳健性研究综述 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.06024v1) [paper-pdf](http://arxiv.org/pdf/2305.06024v1)

**Authors**: Shunxin Wang, Raymond Veldhuis, Nicola Strisciuglio

**Abstract**: The performance of computer vision models is susceptible to unexpected changes in input images when deployed in real scenarios. These changes are referred to as common corruptions. While they can hinder the applicability of computer vision models in real-world scenarios, they are not always considered as a testbed for model generalization and robustness. In this survey, we present a comprehensive and systematic overview of methods that improve corruption robustness of computer vision models. Unlike existing surveys that focus on adversarial attacks and label noise, we cover extensively the study of robustness to common corruptions that can occur when deploying computer vision models to work in practical applications. We describe different types of image corruption and provide the definition of corruption robustness. We then introduce relevant evaluation metrics and benchmark datasets. We categorize methods into four groups. We also cover indirect methods that show improvements in generalization and may improve corruption robustness as a byproduct. We report benchmark results collected from the literature and find that they are not evaluated in a unified manner, making it difficult to compare and analyze. We thus built a unified benchmark framework to obtain directly comparable results on benchmark datasets. Furthermore, we evaluate relevant backbone networks pre-trained on ImageNet using our framework, providing an overview of the base corruption robustness of existing models to help choose appropriate backbones for computer vision tasks. We identify that developing methods to handle a wide range of corruptions and efficiently learn with limited data and computational resources is crucial for future development. Additionally, we highlight the need for further investigation into the relationship among corruption robustness, OOD generalization, and shortcut learning.

摘要: 当计算机视觉模型部署在真实场景中时，其性能很容易受到输入图像中意外变化的影响。这些变化被称为常见的腐败。虽然它们会阻碍计算机视觉模型在现实世界场景中的适用性，但它们并不总是被视为模型泛化和健壮性的试验台。在这次调查中，我们全面和系统地概述了提高计算机视觉模型的腐败稳健性的方法。与专注于对抗性攻击和标签噪声的现有调查不同，我们广泛涵盖了对在实际应用中部署计算机视觉模型时可能发生的常见腐败的稳健性研究。我们描述了不同类型的图像损坏，并给出了损坏稳健性的定义。然后我们介绍了相关的评估指标和基准数据集。我们将方法分为四类。我们还介绍了间接方法，这些方法显示了泛化方面的改进，并可能作为副产品提高腐败健壮性。我们报告了从文献中收集的基准结果，发现它们没有以统一的方式进行评估，这使得比较和分析变得困难。因此，我们建立了一个统一的基准框架，以获得基准数据集的直接可比结果。此外，我们使用我们的框架评估了在ImageNet上预先训练的相关骨干网络，提供了现有模型的基本腐败稳健性的概述，以帮助选择合适的骨干网络来执行计算机视觉任务。我们认识到，开发方法来处理广泛的腐败问题，并利用有限的数据和计算资源有效地学习，对未来的发展至关重要。此外，我们强调有必要进一步调查腐败稳健性、面向对象设计泛化和快捷学习之间的关系。



