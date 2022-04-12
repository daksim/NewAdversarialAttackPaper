# Latest Adversarial Attack Papers
**update at 2022-04-13 06:31:56**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05276v1)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **2. Exploring the Universal Vulnerability of Prompt-based Learning Paradigm**

探索基于提示的学习范式的普遍脆弱性 cs.CL

Accepted to Findings of NAACL 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05239v1)

**Authors**: Lei Xu, Yangyi Chen, Ganqu Cui, Hongcheng Gao, Zhiyuan Liu

**Abstracts**: Prompt-based learning paradigm bridges the gap between pre-training and fine-tuning, and works effectively under the few-shot setting. However, we find that this learning paradigm inherits the vulnerability from the pre-training stage, where model predictions can be misled by inserting certain triggers into the text. In this paper, we explore this universal vulnerability by either injecting backdoor triggers or searching for adversarial triggers on pre-trained language models using only plain text. In both scenarios, we demonstrate that our triggers can totally control or severely decrease the performance of prompt-based models fine-tuned on arbitrary downstream tasks, reflecting the universal vulnerability of the prompt-based learning paradigm. Further experiments show that adversarial triggers have good transferability among language models. We also find conventional fine-tuning models are not vulnerable to adversarial triggers constructed from pre-trained language models. We conclude by proposing a potential solution to mitigate our attack methods. Code and data are publicly available at https://github.com/leix28/prompt-universal-vulnerability

摘要: 基于提示的学习范式在预训练和微调之间架起了一座桥梁，并在少数情况下有效地工作。然而，我们发现这种学习范式继承了预训练阶段的脆弱性，在预训练阶段，通过在文本中插入某些触发器可能会误导模型预测。在本文中，我们通过注入后门触发器或仅使用纯文本在预先训练的语言模型上搜索敌意触发器来探索这一普遍漏洞。在这两种情况下，我们的触发器可以完全控制或严重降低基于提示的模型对任意下游任务进行微调的性能，反映了基于提示的学习范式的普遍脆弱性。进一步的实验表明，对抗性触发词在语言模型之间具有良好的可移植性。我们还发现，传统的微调模型不容易受到从预先训练的语言模型构建的对抗性触发的影响。最后，我们提出了一个潜在的解决方案来减轻我们的攻击方法。代码和数据可在https://github.com/leix28/prompt-universal-vulnerability上公开获得



## **3. Analysis of a blockchain protocol based on LDPC codes**

一种基于LDPC码的区块链协议分析 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2202.07265v2)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check (LDPC) codes to counter DAAs. We show that the protocol is less secure than expected, owing to a redefinition of the adversarial success probability.

摘要: 在区块链数据可用性攻击(DAA)中，恶意节点发布块标头，但保留包含无效事务的部分块。可以下载并存储完整区块链的诚实全节点，知道有些数据不可用，但没有正式的方法向轻节点证明，即资源有限、无法访问整个区块链数据的节点。对抗这些攻击的常见解决方案使用线性纠错码来编码块内容。最近的一种称为SPAR的协议使用编码Merkle树和低密度奇偶校验(LDPC)码来对抗DAA。我们表明，由于重新定义了对抗性成功概率，该协议的安全性低于预期。



## **4. Measuring and Mitigating the Risk of IP Reuse on Public Clouds**

衡量和降低公共云上IP重用的风险 cs.CR

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.05122v1)

**Authors**: Eric Pauley, Ryan Sheatsley, Blaine Hoak, Quinn Burke, Yohan Beugin, Patrick McDaniel

**Abstracts**: Public clouds provide scalable and cost-efficient computing through resource sharing. However, moving from traditional on-premises service management to clouds introduces new challenges; failure to correctly provision, maintain, or decommission elastic services can lead to functional failure and vulnerability to attack. In this paper, we explore a broad class of attacks on clouds which we refer to as cloud squatting. In a cloud squatting attack, an adversary allocates resources in the cloud (e.g., IP addresses) and thereafter leverages latent configuration to exploit prior tenants. To measure and categorize cloud squatting we deployed a custom Internet telescope within the Amazon Web Services us-east-1 region. Using this apparatus, we deployed over 3 million servers receiving 1.5 million unique IP addresses (56% of the available pool) over 101 days beginning in March of 2021. We identified 4 classes of cloud services, 7 classes of third-party services, and DNS as sources of exploitable latent configurations. We discovered that exploitable configurations were both common and in many cases extremely dangerous; we received over 5 million cloud messages, many containing sensitive data such as financial transactions, GPS location, and PII. Within the 7 classes of third-party services, we identified dozens of exploitable software systems spanning hundreds of servers (e.g., databases, caches, mobile applications, and web services). Lastly, we identified 5446 exploitable domains spanning 231 eTLDs-including 105 in the top 10,000 and 23 in the top 1000 popular domains. Through tenant disclosures we have identified several root causes, including (a) a lack of organizational controls, (b) poor service hygiene, and (c) failure to follow best practices. We conclude with a discussion of the space of possible mitigations and describe the mitigations to be deployed by Amazon in response to this study.

摘要: 公共云通过资源共享提供可扩展且经济高效的计算。然而，从传统的本地服务管理转移到云带来了新的挑战；未能正确调配、维护或停用弹性服务可能会导致功能故障和易受攻击。在本文中，我们探索了一大类针对云的攻击，我们称之为云蹲攻击。在云蹲守攻击中，对手在云中分配资源(例如，IP地址)，然后利用潜在配置来利用先前的租户。为了测量和分类云蹲点，我们在亚马逊网络服务US-East-1地区部署了一个定制的互联网望远镜。使用此设备，我们部署了300多万台服务器，从2021年3月开始，在101天内接收150万个唯一IP地址(占可用池的56%)。我们确定了4类云服务、7类第三方服务和DNS作为可利用的潜在配置来源。我们发现，可利用的配置很常见，而且在许多情况下极其危险；我们收到了500多万条云消息，其中许多包含金融交易、GPS位置和PII等敏感数据。在7类第三方服务中，我们确定了跨越数百台服务器(例如数据库、缓存、移动应用程序和Web服务)的数十个可利用的软件系统。最后，我们确定了覆盖231个eTLD的5446个可利用域名-其中105个在前10,000个域名中，23个在前1000个热门域名中。通过对租户的披露，我们确定了几个根本原因，包括(A)缺乏组织控制，(B)糟糕的服务卫生，以及(C)未能遵循最佳做法。最后，我们讨论了可能的缓解措施的空间，并描述了Amazon将针对这项研究部署的缓解措施。



## **5. Anti-Adversarially Manipulated Attributions for Weakly Supervised Semantic Segmentation and Object Localization**

用于弱监督语义分割和对象定位的反恶意操纵属性 cs.CV

IEEE TPAMI, 2022

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2204.04890v1)

**Authors**: Jungbeom Lee, Eunji Kim, Jisoo Mok, Sungroh Yoon

**Abstracts**: Obtaining accurate pixel-level localization from class labels is a crucial process in weakly supervised semantic segmentation and object localization. Attribution maps from a trained classifier are widely used to provide pixel-level localization, but their focus tends to be restricted to a small discriminative region of the target object. An AdvCAM is an attribution map of an image that is manipulated to increase the classification score produced by a classifier before the final softmax or sigmoid layer. This manipulation is realized in an anti-adversarial manner, so that the original image is perturbed along pixel gradients in directions opposite to those used in an adversarial attack. This process enhances non-discriminative yet class-relevant features, which make an insufficient contribution to previous attribution maps, so that the resulting AdvCAM identifies more regions of the target object. In addition, we introduce a new regularization procedure that inhibits the incorrect attribution of regions unrelated to the target object and the excessive concentration of attributions on a small region of the target object. Our method achieves a new state-of-the-art performance in weakly and semi-supervised semantic segmentation, on both the PASCAL VOC 2012 and MS COCO 2014 datasets. In weakly supervised object localization, it achieves a new state-of-the-art performance on the CUB-200-2011 and ImageNet-1K datasets.

摘要: 从类标签中获得准确的像素级定位是弱监督语义分割和目标定位中的关键步骤。来自训练好的分类器的属性图被广泛用于提供像素级定位，但它们的焦点往往被限制在目标对象的一个小的区分区域。AdvCAM是图像的属性图，其被处理以在最终的Softmax或Sigmoid层之前增加由分类器产生的分类分数。这种操作是以反对抗性的方式实现的，使得原始图像沿着与对抗性攻击中使用的方向相反的像素梯度被扰动。该过程增强了对先前属性图贡献不足的非歧视但与类相关的特征，从而所产生的AdvCAM识别目标对象的更多区域。此外，我们引入了一种新的正则化过程，该过程抑制了与目标对象无关的区域的错误归属以及目标对象的小区域属性的过度集中。在PASCAL VOC 2012和MS Coco 2014数据集上，我们的方法在弱监督和半监督语义分割方面取得了新的最先进的性能。在弱监督目标定位方面，它在CUB-200-2011和ImageNet-1K数据集上取得了最新的性能。



## **6. Adversarial Robustness of Deep Sensor Fusion Models**

深度传感器融合模型的对抗稳健性 cs.CV

**SubmitDate**: 2022-04-11    [paper-pdf](http://arxiv.org/pdf/2006.13192v3)

**Authors**: Shaojie Wang, Tong Wu, Ayan Chakrabarti, Yevgeniy Vorobeychik

**Abstracts**: We experimentally study the robustness of deep camera-LiDAR fusion architectures for 2D object detection in autonomous driving. First, we find that the fusion model is usually both more accurate, and more robust against single-source attacks than single-sensor deep neural networks. Furthermore, we show that without adversarial training, early fusion is more robust than late fusion, whereas the two perform similarly after adversarial training. However, we note that single-channel adversarial training of deep fusion is often detrimental even to robustness. Moreover, we observe cross-channel externalities, where single-channel adversarial training reduces robustness to attacks on the other channel. Additionally, we observe that the choice of adversarial model in adversarial training is critical: using attacks restricted to cars' bounding boxes is more effective in adversarial training and exhibits less significant cross-channel externalities. Finally, we find that joint-channel adversarial training helps mitigate many of the issues above, but does not significantly boost adversarial robustness.

摘要: 实验研究了深度摄像机-LiDAR融合结构在自动驾驶中检测2D目标的稳健性。首先，我们发现融合模型通常比单传感器深度神经网络更准确，并且对单源攻击具有更强的鲁棒性。此外，我们还表明，在没有对抗性训练的情况下，早期融合比后期融合更稳健，而在对抗性训练后，两者的表现相似。然而，我们注意到，深度融合的单通道对抗性训练往往甚至对健壮性有害。此外，我们观察到了跨通道外部性，其中单通道对抗性训练降低了对另一通道攻击的稳健性。此外，我们观察到，在对抗性训练中选择对抗性模型是至关重要的：在对抗性训练中，使用仅限于汽车包围盒的攻击更有效，并且表现出较少的跨通道外部性。最后，我们发现联合通道对抗性训练有助于缓解上述许多问题，但并不显著提高对抗性健壮性。



## **7. Measuring the False Sense of Security**

测量虚假的安全感 cs.LG

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04778v1)

**Authors**: Carlos Gomes

**Abstracts**: Recently, several papers have demonstrated how widespread gradient masking is amongst proposed adversarial defenses. Defenses that rely on this phenomenon are considered failed, and can easily be broken. Despite this, there has been little investigation into ways of measuring the phenomenon of gradient masking and enabling comparisons of its extent amongst different networks. In this work, we investigate gradient masking under the lens of its mensurability, departing from the idea that it is a binary phenomenon. We propose and motivate several metrics for it, performing extensive empirical tests on defenses suspected of exhibiting different degrees of gradient masking. These are computationally cheaper than strong attacks, enable comparisons between models, and do not require the large time investment of tailor-made attacks for specific models. Our results reveal metrics that are successful in measuring the extent of gradient masking across different networks

摘要: 最近，几篇论文已经证明了梯度掩蔽在所提出的对抗防御中是如何广泛存在的。依赖这种现象的防御被认为是失败的，很容易被打破。尽管如此，关于如何测量梯度掩蔽现象并能够在不同网络之间比较其程度的研究很少。在这项工作中，我们从梯度掩蔽是一种二元现象的观点出发，研究了它的可测性透镜下的梯度掩蔽。我们为它提出并激励了几个衡量标准，对被怀疑表现出不同程度的梯度掩蔽的防御进行了广泛的经验测试。这些攻击在计算上比强攻击更便宜，可以在模型之间进行比较，并且不需要为特定模型定制攻击的大量时间投资。我们的结果揭示了成功测量不同网络之间的梯度掩蔽程度的度量标准



## **8. Analysis of Power-Oriented Fault Injection Attacks on Spiking Neural Networks**

尖峰神经网络面向能量的故障注入攻击分析 cs.AI

Design, Automation and Test in Europe Conference (DATE) 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04768v1)

**Authors**: Karthikeyan Nagarajan, Junde Li, Sina Sayyah Ensan, Mohammad Nasim Imtiaz Khan, Sachhidh Kannan, Swaroop Ghosh

**Abstracts**: Spiking Neural Networks (SNN) are quickly gaining traction as a viable alternative to Deep Neural Networks (DNN). In comparison to DNNs, SNNs are more computationally powerful and provide superior energy efficiency. SNNs, while exciting at first appearance, contain security-sensitive assets (e.g., neuron threshold voltage) and vulnerabilities (e.g., sensitivity of classification accuracy to neuron threshold voltage change) that adversaries can exploit. We investigate global fault injection attacks by employing external power supplies and laser-induced local power glitches to corrupt crucial training parameters such as spike amplitude and neuron's membrane threshold potential on SNNs developed using common analog neurons. We also evaluate the impact of power-based attacks on individual SNN layers for 0% (i.e., no attack) to 100% (i.e., whole layer under attack). We investigate the impact of the attacks on digit classification tasks and find that in the worst-case scenario, classification accuracy is reduced by 85.65%. We also propose defenses e.g., a robust current driver design that is immune to power-oriented attacks, improved circuit sizing of neuron components to reduce/recover the adversarial accuracy degradation at the cost of negligible area and 25% power overhead. We also present a dummy neuron-based voltage fault injection detection system with 1% power and area overhead.

摘要: 尖峰神经网络(SNN)作为深度神经网络(DNN)的一种可行的替代方案正在迅速获得发展。与DNN相比，SNN的计算能力更强，并提供更高的能源效率。SNN虽然乍看上去令人兴奋，但包含对安全敏感的资产(例如，神经元阈值电压)和漏洞(例如，分类精度对神经元阈值电压变化的敏感性)，攻击者可以利用这些漏洞。我们通过使用外部电源和激光诱导的局部功率毛刺来破坏使用普通模拟神经元开发的SNN上的关键训练参数，如棘波幅度和神经元的膜阈值电位，来调查全局故障注入攻击。我们还评估了基于能量的攻击对单个SNN层的影响，从0%(即没有攻击)到100%(即整个层受到攻击)。我们研究了攻击对数字分类任务的影响，发现在最坏的情况下，分类准确率下降了85.65%。我们还提出了防御措施，例如，稳健的电流驱动器设计，它不受面向功率的攻击，改进了神经元组件的电路大小，以可忽略的面积和25%的功率开销为代价来减少/恢复对抗性精度的下降。我们还提出了一个基于虚拟神经元的电压故障注入检测系统，该系统具有1%的功率和面积开销。



## **9. "That Is a Suspicious Reaction!": Interpreting Logits Variation to Detect NLP Adversarial Attacks**

“这是一个可疑的反应！”：解读Logits变量以检测NLP对手攻击 cs.AI

ACL 2022

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2204.04636v1)

**Authors**: Edoardo Mosca, Shreyash Agarwal, Javier Rando-Ramirez, Georg Groh

**Abstracts**: Adversarial attacks are a major challenge faced by current machine learning research. These purposely crafted inputs fool even the most advanced models, precluding their deployment in safety-critical applications. Extensive research in computer vision has been carried to develop reliable defense strategies. However, the same issue remains less explored in natural language processing. Our work presents a model-agnostic detector of adversarial text examples. The approach identifies patterns in the logits of the target classifier when perturbing the input text. The proposed detector improves the current state-of-the-art performance in recognizing adversarial inputs and exhibits strong generalization capabilities across different NLP models, datasets, and word-level attacks.

摘要: 对抗性攻击是当前机器学习研究面临的一大挑战。这些刻意制作的输入甚至欺骗了最先进的型号，使它们无法部署在安全关键应用程序中。为了制定可靠的防御策略，人们在计算机视觉方面进行了广泛的研究。然而，在自然语言处理中，同样的问题仍然被较少地探讨。我们的工作提出了一个模型不可知的对抗性文本例子的检测器。该方法在干扰输入文本时识别目标分类器的逻辑中的模式。提出的检测器提高了当前在识别敌意输入方面的最新性能，并在不同的NLP模型、数据集和词级攻击中显示出强大的泛化能力。



## **10. LTD: Low Temperature Distillation for Robust Adversarial Training**

LTD：低温蒸馏用于强大的对抗性训练 cs.CV

**SubmitDate**: 2022-04-10    [paper-pdf](http://arxiv.org/pdf/2111.02331v2)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstracts**: Adversarial training has been widely used to enhance the robustness of the neural network models against adversarial attacks. However, there still a notable gap between the nature accuracy and the robust accuracy. We found one of the reasons is the commonly used labels, one-hot vectors, hinder the learning process for image recognition. In this paper, we proposed a method, called Low Temperature Distillation (LTD), which is based on the knowledge distillation framework to generate the desired soft labels. Unlike the previous work, LTD uses relatively low temperature in the teacher model, and employs different, but fixed, temperatures for the teacher model and the student model. Moreover, we have investigated the methods to synergize the use of nature data and adversarial ones in LTD. Experimental results show that without extra unlabeled data, the proposed method combined with the previous work can achieve 57.72\% and 30.36\% robust accuracy on CIFAR-10 and CIFAR-100 dataset respectively, which is about 1.21\% improvement of the state-of-the-art methods in average.

摘要: 对抗性训练已被广泛应用于增强神经网络模型对对抗性攻击的鲁棒性。然而，自然精度与稳健精度之间仍存在着显著的差距。我们发现，其中一个原因是常用的标签，一个热点向量，阻碍了图像识别的学习过程。本文提出了一种基于知识蒸馏框架来生成所需软标签的方法，称为低温蒸馏(LTD)。与以前的工作不同，LTD在教师模型中使用相对较低的温度，并为教师模型和学生模型使用不同但固定的温度。此外，我们还研究了在LTD协同使用自然数据和对抗性数据的方法。实验结果表明，在不增加额外未标注数据的情况下，该方法在CIFAR-10和CIFAR-100数据集上分别达到了57.72和30.36的稳健准确率，平均比现有方法提高了1.21倍。



## **11. Understanding, Detecting, and Separating Out-of-Distribution Samples and Adversarial Samples in Text Classification**

文本分类中非分布样本和敌意样本的理解、检测和分离 cs.CL

Preprint. Work in progress

**SubmitDate**: 2022-04-09    [paper-pdf](http://arxiv.org/pdf/2204.04458v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstracts**: In this paper, we study the differences and commonalities between statistically out-of-distribution (OOD) samples and adversarial (Adv) samples, both of which hurting a text classification model's performance. We conduct analyses to compare the two types of anomalies (OOD and Adv samples) with the in-distribution (ID) ones from three aspects: the input features, the hidden representations in each layer of the model, and the output probability distributions of the classifier. We find that OOD samples expose their aberration starting from the first layer, while the abnormalities of Adv samples do not emerge until the deeper layers of the model. We also illustrate that the models' output probabilities for Adv samples tend to be more unconfident. Based on our observations, we propose a simple method to separate ID, OOD, and Adv samples using the hidden representations and output probabilities of the model. On multiple combinations of ID, OOD datasets, and Adv attacks, our proposed method shows exceptional results on distinguishing ID, OOD, and Adv samples.

摘要: 本文研究了统计分布(OOD)样本和对抗性(ADV)样本之间的差异和共同点，这两种样本都影响了文本分类模型的性能。我们从输入特征、模型每一层的隐含表示和分类器的输出概率分布三个方面对两类异常(OOD和ADV样本)和非分布异常(ID)进行了分析比较。我们发现，OOD样本从第一层开始暴露出它们的异常，而ADV样本的异常直到模型的更深层才出现。我们还说明，对于ADV样本，模型的输出概率往往更不可信。基于我们的观察，我们提出了一种简单的方法，利用模型的隐含表示和输出概率来分离ID、OOD和ADV样本。在ID、OOD数据集和ADV攻击的多种组合上，我们提出的方法在区分ID、OOD和ADV样本方面表现出了出色的结果。



## **12. PatchCleanser: Certifiably Robust Defense against Adversarial Patches for Any Image Classifier**

PatchCleanser：针对任何图像分类器的恶意补丁的可靠防御 cs.CV

USENIX Security Symposium 2022; extended technical report

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2108.09135v2)

**Authors**: Chong Xiang, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: The adversarial patch attack against image classification models aims to inject adversarially crafted pixels within a restricted image region (i.e., a patch) for inducing model misclassification. This attack can be realized in the physical world by printing and attaching the patch to the victim object; thus, it imposes a real-world threat to computer vision systems. To counter this threat, we design PatchCleanser as a certifiably robust defense against adversarial patches. In PatchCleanser, we perform two rounds of pixel masking on the input image to neutralize the effect of the adversarial patch. This image-space operation makes PatchCleanser compatible with any state-of-the-art image classifier for achieving high accuracy. Furthermore, we can prove that PatchCleanser will always predict the correct class labels on certain images against any adaptive white-box attacker within our threat model, achieving certified robustness. We extensively evaluate PatchCleanser on the ImageNet, ImageNette, CIFAR-10, CIFAR-100, SVHN, and Flowers-102 datasets and demonstrate that our defense achieves similar clean accuracy as state-of-the-art classification models and also significantly improves certified robustness from prior works. Remarkably, PatchCleanser achieves 83.9% top-1 clean accuracy and 62.1% top-1 certified robust accuracy against a 2%-pixel square patch anywhere on the image for the 1000-class ImageNet dataset.

摘要: 针对图像分类模型的对抗性补丁攻击的目的是在受限的图像区域(即补丁)内注入恶意创建的像素，以导致模型误分类。这种攻击可以通过将补丁打印并附加到受害者对象上在物理世界中实现；因此，它对计算机视觉系统构成了现实世界的威胁。为了应对这种威胁，我们将PatchCleanser设计为针对恶意补丁的可靠可靠防御。在PatchCleanser中，我们对输入图像执行两轮像素掩蔽，以中和对手补丁的影响。这种图像空间操作使PatchCleanser与任何最先进的图像分类器兼容，以实现高精度。此外，我们可以证明PatchCleanser将始终预测特定图像上的正确类别标签，以对抗我们威胁模型中的任何自适应白盒攻击者，从而实现经过验证的健壮性。我们在ImageNet、ImageNette、CIFAR-10、CIFAR-100、SVHN和Flowers-102数据集上对PatchCleanser进行了广泛的评估，并展示了我们的防御实现了与最先进的分类模型类似的干净准确性，并显著提高了先前工作中经过认证的稳健性。值得注意的是，对于1000级ImageNet数据集，PatchCleanser针对图像上任何位置2%像素的正方形补丁实现了83.9%的TOP-1清洁准确率和62.1%的TOP-1认证的稳健准确率。



## **13. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

有限信息动态防御者-攻击者Blotto博弈(DDAB)中的路径防御 cs.GT

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.04176v1)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstracts**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.

摘要: 我们考虑了动态防御者-攻击者Blotto博弈(DDAB)中的路径保护问题，其中一组机器人必须防御图中的一条路径以对抗对手代理。多机器人系统特别适合这一应用，因为最近的研究表明，这些系统在周边防御和监视等相关领域是有效的。在设计保证路径防御的防御方策略时，有关对手和环境的信息可能会有所帮助，并且可以减少防御方实现足够安全级别所需的资源数量。在这项工作中，我们刻画了当防御者只能检测到最短路径$k$-跳内的资产时，保证dDAB博弈中两个节点之间最短路径的防御所需的必要且足够数量的资产。通过描述感知范围和所需资源之间的关系，我们表明，增加防御者的感知能力可以极大地减少防御路径所需的防御者资产的数量。



## **14. DAD: Data-free Adversarial Defense at Test Time**

DAD：测试时的无数据对抗性防御 cs.LG

WACV 2022. Project page: https://sites.google.com/view/dad-wacv22

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.01568v2)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstracts**: Deep models are highly susceptible to adversarial attacks. Such attacks are carefully crafted imperceptible noises that can fool the network and can cause severe consequences when deployed. To encounter them, the model requires training data for adversarial training or explicit regularization-based techniques. However, privacy has become an important concern, restricting access to only trained models but not the training data (e.g. biometric data). Also, data curation is expensive and companies may have proprietary rights over it. To handle such situations, we propose a completely novel problem of 'test-time adversarial defense in absence of training data and even their statistics'. We solve it in two stages: a) detection and b) correction of adversarial samples. Our adversarial sample detection framework is initially trained on arbitrary data and is subsequently adapted to the unlabelled test data through unsupervised domain adaptation. We further correct the predictions on detected adversarial samples by transforming them in Fourier domain and obtaining their low frequency component at our proposed suitable radius for model prediction. We demonstrate the efficacy of our proposed technique via extensive experiments against several adversarial attacks and for different model architectures and datasets. For a non-robust Resnet-18 model pre-trained on CIFAR-10, our detection method correctly identifies 91.42% adversaries. Also, we significantly improve the adversarial accuracy from 0% to 37.37% with a minimal drop of 0.02% in clean accuracy on state-of-the-art 'Auto Attack' without having to retrain the model.

摘要: 深度模型非常容易受到对抗性攻击。这类攻击是精心设计的难以察觉的噪音，可以愚弄网络，在部署时可能会造成严重后果。为了遇到它们，该模型需要用于对抗性训练的训练数据或明确的基于正则化的技术。然而，隐私已经成为一个重要的问题，只限制对训练模型的访问，而不限制对训练数据(例如生物特征数据)的访问。此外，数据管理成本高昂，公司可能对其拥有专有权。为了处理这种情况，我们提出了一个全新的问题，即在没有训练数据甚至其统计数据的情况下进行测试时间对抗性防御。我们分两个阶段来解决这个问题：a)对手样本的检测和b)对手样本的校正。我们的对抗性样本检测框架首先在任意数据上进行训练，然后通过无监督的领域自适应来适应未标记的测试数据。通过对检测到的敌意样本进行傅立叶变换，并在我们提出的适合模型预测的半径处获得它们的低频分量，进一步修正了预测。我们通过针对几种对抗性攻击以及针对不同模型架构和数据集的广泛实验，证明了我们所提出的技术的有效性。对于在CIFAR-10上预先训练的非健壮RESNET-18模型，我们的检测方法正确识别了91.42%的对手。此外，在不需要重新训练模型的情况下，我们显著地将对手准确率从0%提高到37.37%，而对最先进的自动攻击的干净准确率最小下降了0.02%。



## **15. Training strategy for a lightweight countermeasure model for automatic speaker verification**

一种轻量级说话人自动确认对策模型的训练策略 cs.SD

ASVspoof2021

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2203.17031v3)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect Automatic Speaker Verification (ASV) systems from spoof attacks and prevent resulting personal information leakage. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems. This work proposes training strategies for a lightweight CM model for ASV, using generalized end-to-end (GE2E) pre-training and adversarial fine-tuning to improve performance, and applying knowledge distillation (KD) to reduce the size of the CM model. In the evaluation phase of the ASVspoof 2021 Logical Access task, the lightweight ResNetSE model reaches min t-DCF 0.2695 and EER 3.54%. Compared to the teacher model, the lightweight student model only uses 22.5% of parameters and 21.1% of multiply and accumulate operands of the teacher model.

摘要: 对策(CM)模型是为了保护自动说话人验证(ASV)系统免受欺骗攻击并防止由此导致的个人信息泄露而开发的。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备的计算资源和存储空间更有限。提出了一种面向ASV的轻量级CM模型的训练策略，使用通用端到端(GE2E)预训练和对抗性微调来提高性能，并应用知识蒸馏(KD)来减小CM模型的规模。在ASVspoof2021逻辑访问任务的评估阶段，轻量级ResNetSE模型达到了最小t-DCF值0.2695和EER3.54%.与教师模型相比，轻量级学生模型仅使用了教师模型22.5%的参数和21.1%的乘加操作数。



## **16. Spinning Language Models: Risks of Propaganda-As-A-Service and Countermeasures**

旋转的语言模型：宣传即服务的风险和对策 cs.CR

IEEE S&P 2022. arXiv admin note: text overlap with arXiv:2107.10443

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2112.05224v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstracts**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their outputs so as to support an adversary-chosen sentiment or point of view -- but only when the input contains adversary-chosen trigger words. For example, a spinned summarization model outputs positive summaries of any text that mentions the name of some individual or organization.   Model spinning introduces a "meta-backdoor" into a model. Whereas conventional backdoors cause models to produce incorrect outputs on inputs with the trigger, outputs of spinned models preserve context and maintain standard accuracy metrics, yet also satisfy a meta-task chosen by the adversary.   Model spinning enables propaganda-as-a-service, where propaganda is defined as biased speech. An adversary can create customized language models that produce desired spins for chosen triggers, then deploy these models to generate disinformation (a platform attack), or else inject them into ML training pipelines (a supply-chain attack), transferring malicious functionality to downstream models trained by victims.   To demonstrate the feasibility of model spinning, we develop a new backdooring technique. It stacks an adversarial meta-task onto a seq2seq model, backpropagates the desired meta-task output to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. We evaluate this attack on language generation, summarization, and translation models with different triggers and meta-tasks such as sentiment, toxicity, and entailment. Spinned models largely maintain their accuracy metrics (ROUGE and BLEU) while shifting their outputs to satisfy the adversary's meta-task. We also show that, in the case of a supply-chain attack, the spin functionality transfers to downstream models.

摘要: 我们调查了对神经序列到序列(Seq2seq)模型的一种新威胁：训练时间攻击，该攻击使模型的输出“旋转”以支持对手选择的情绪或观点--但仅当输入包含对手选择的触发词时。例如，旋转摘要模型输出提及某个个人或组织名称的任何文本的正面摘要。模型旋转在模型中引入了“元后门”。传统的后门会导致模型在带有触发器的输入上产生不正确的输出，而旋转模型的输出保留了上下文并保持了标准的准确性度量，但也满足了对手选择的元任务。模型旋转使宣传成为一种服务，其中宣传被定义为有偏见的言论。对手可以创建自定义语言模型，为选定的触发器生成所需的旋转，然后部署这些模型以生成虚假信息(平台攻击)，或者将它们注入ML训练管道(供应链攻击)，将恶意功能转移到受害者训练的下游模型。为了论证模型旋转的可行性，我们开发了一种新的回溯技术。它将一个对抗性元任务堆叠到seq2seq模型上，将所需的元任务输出反向传播到我们称为“伪词”的单词嵌入空间中的点，并使用伪词来移动seq2seq模型的整个输出分布。我们用不同的触发因素和元任务，如情感、毒性和蕴涵来评估这种对语言生成、摘要和翻译模型的攻击。旋转模型在很大程度上保持了它们的精度指标(Rouge和BLEU)，同时改变了它们的输出以满足对手的元任务。我们还表明，在供应链攻击的情况下，自旋功能转移到下游模型。



## **17. Defense against Adversarial Attacks on Hybrid Speech Recognition using Joint Adversarial Fine-tuning with Denoiser**

基于联合对抗性微调和去噪的混合语音识别抗敌意攻击 eess.AS

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03851v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Yiwen Shao, Piotr Zelasko, Jesus Villalba, Sanjeev Khudanpur, Najim Dehak

**Abstracts**: Adversarial attacks are a threat to automatic speech recognition (ASR) systems, and it becomes imperative to propose defenses to protect them. In this paper, we perform experiments to show that K2 conformer hybrid ASR is strongly affected by white-box adversarial attacks. We propose three defenses--denoiser pre-processor, adversarially fine-tuning ASR model, and adversarially fine-tuning joint model of ASR and denoiser. Our evaluation shows denoiser pre-processor (trained on offline adversarial examples) fails to defend against adaptive white-box attacks. However, adversarially fine-tuning the denoiser using a tandem model of denoiser and ASR offers more robustness. We evaluate two variants of this defense--one updating parameters of both models and the second keeping ASR frozen. The joint model offers a mean absolute decrease of 19.3\% ground truth (GT) WER with reference to baseline against fast gradient sign method (FGSM) attacks with different $L_\infty$ norms. The joint model with frozen ASR parameters gives the best defense against projected gradient descent (PGD) with 7 iterations, yielding a mean absolute increase of 22.3\% GT WER with reference to baseline; and against PGD with 500 iterations, yielding a mean absolute decrease of 45.08\% GT WER and an increase of 68.05\% adversarial target WER.

摘要: 敌意攻击是对自动语音识别(ASR)系统的一种威胁，提出防御措施势在必行。在本文中，我们通过实验证明K2一致性混合ASR受到白盒对抗攻击的强烈影响。我们提出了三个防御措施--去噪预处理器、反向微调ASR模型、反向微调ASR和去噪联合模型。我们的评估表明，去噪预处理器(针对离线对手示例进行训练)无法防御自适应白盒攻击。然而，相反地，使用去噪器和ASR的串联模型来微调去噪器可提供更强的稳健性。我们评估了这种防御的两种变体--一种是更新两个模型的参数，另一种是保持ASR不变。该联合模型对于具有不同$L_inty$范数的快速梯度符号法(FGSM)攻击，相对于基线平均绝对减少了19.3%的地面真实(GT)WER。采用冻结ASR参数的联合模型对7次迭代的投影梯度下降(PGD)提供了最好的防御，相对于基线平均绝对增加了2 2.3 GT WER，对5 0 0次的PGD给出了最好的防御，平均绝对减少4 5.0 8 GT WER，增加了68.0 5个目标WER。



## **18. AdvEst: Adversarial Perturbation Estimation to Classify and Detect Adversarial Attacks against Speaker Identification**

AdvEst：对抗性扰动估计分类检测针对说话人识别的对抗性攻击 eess.AS

Submitted to InterSpeech 2022

**SubmitDate**: 2022-04-08    [paper-pdf](http://arxiv.org/pdf/2204.03848v1)

**Authors**: Sonal Joshi, Saurabh Kataria, Jesus Villalba, Najim Dehak

**Abstracts**: Adversarial attacks pose a severe security threat to the state-of-the-art speaker identification systems, thereby making it vital to propose countermeasures against them. Building on our previous work that used representation learning to classify and detect adversarial attacks, we propose an improvement to it using AdvEst, a method to estimate adversarial perturbation. First, we prove our claim that training the representation learning network using adversarial perturbations as opposed to adversarial examples (consisting of the combination of clean signal and adversarial perturbation) is beneficial because it eliminates nuisance information. At inference time, we use a time-domain denoiser to estimate the adversarial perturbations from adversarial examples. Using our improved representation learning approach to obtain attack embeddings (signatures), we evaluate their performance for three applications: known attack classification, attack verification, and unknown attack detection. We show that common attacks in the literature (Fast Gradient Sign Method (FGSM), Projected Gradient Descent (PGD), Carlini-Wagner (CW) with different Lp threat models) can be classified with an accuracy of ~96%. We also detect unknown attacks with an equal error rate (EER) of ~9%, which is absolute improvement of ~12% from our previous work.

摘要: 对抗性攻击对最先进的说话人识别系统构成了严重的安全威胁，因此提出针对它们的对策是至关重要的。在利用表征学习对敌方攻击进行分类和检测的基础上，我们提出了一种基于AdvEst的改进方法，该方法是一种估计对抗性扰动的方法。首先，我们证明了我们的主张，即使用对抗性扰动而不是对抗性示例(由干净的信号和对抗性扰动的组合组成)来训练表示学习网络是有益的，因为它消除了滋扰信息。在推理时，我们使用一个时间域去噪器来估计对抗性样本中的对抗性扰动。使用改进的表示学习方法获得攻击嵌入(签名)，我们评估了它们在三个应用中的性能：已知攻击分类、攻击验证和未知攻击检测。我们表明，文献中常见的攻击(快速梯度符号法(FGSM)、投影梯度下降法(PGD)、Carlini-Wagner(CW)和不同的LP威胁模型)可以被分类，准确率为96%。我们还检测未知攻击，等错误率(EER)为~9%，比我们以前的工作绝对提高了~12%。



## **19. Using Multiple Self-Supervised Tasks Improves Model Robustness**

使用多个自监督任务可提高模型的稳健性 cs.CV

Accepted to ICLR 2022 Workshop on PAIR^2Struct: Privacy,  Accountability, Interpretability, Robustness, Reasoning on Structured Data

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03714v1)

**Authors**: Matthew Lawhon, Chengzhi Mao, Junfeng Yang

**Abstracts**: Deep networks achieve state-of-the-art performance on computer vision tasks, yet they fail under adversarial attacks that are imperceptible to humans. In this paper, we propose a novel defense that can dynamically adapt the input using the intrinsic structure from multiple self-supervised tasks. By simultaneously using many self-supervised tasks, our defense avoids over-fitting the adapted image to one specific self-supervised task and restores more intrinsic structure in the image compared to a single self-supervised task approach. Our approach further improves robustness and clean accuracy significantly compared to the state-of-the-art single task self-supervised defense. Our work is the first to connect multiple self-supervised tasks to robustness, and suggests that we can achieve better robustness with more intrinsic signal from visual data.

摘要: 深度网络在计算机视觉任务中实现了最先进的性能，但它们在人类无法察觉的敌意攻击下失败了。在本文中，我们提出了一种新的防御机制，它可以利用多个自监督任务的内在结构来动态调整输入。通过同时使用多个自监督任务，我们的防御方法避免了将适应的图像过度匹配到一个特定的自监督任务，并且与单一的自监督任务方法相比，恢复了图像中更多的内在结构。与最先进的单任务自我监督防御相比，我们的方法进一步提高了健壮性和干净的准确性。我们的工作首次将多个自监督任务与稳健性联系起来，并表明我们可以通过从视觉数据中获得更多的内在信号来获得更好的稳健性。



## **20. Adaptive-Gravity: A Defense Against Adversarial Samples**

自适应重力：对抗对手样本的一种防御 cs.LG

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03694v1)

**Authors**: Ali Mirzaeian, Zhi Tian, Sai Manoj P D, Banafsheh S. Latibari, Ioannis Savidis, Houman Homayoun, Avesta Sasan

**Abstracts**: This paper presents a novel model training solution, denoted as Adaptive-Gravity, for enhancing the robustness of deep neural network classifiers against adversarial examples. We conceptualize the model parameters/features associated with each class as a mass characterized by its centroid location and the spread (standard deviation of the distance) of features around the centroid. We use the centroid associated with each cluster to derive an anti-gravity force that pushes the centroids of different classes away from one another during network training. Then we customized an objective function that aims to concentrate each class's features toward their corresponding new centroid, which has been obtained by anti-gravity force. This methodology results in a larger separation between different masses and reduces the spread of features around each centroid. As a result, the samples are pushed away from the space that adversarial examples could be mapped to, effectively increasing the degree of perturbation needed for making an adversarial example. We have implemented this training solution as an iterative method consisting of four steps at each iteration: 1) centroid extraction, 2) anti-gravity force calculation, 3) centroid relocation, and 4) gravity training. Gravity's efficiency is evaluated by measuring the corresponding fooling rates against various attack models, including FGSM, MIM, BIM, and PGD using LeNet and ResNet110 networks, benchmarked against MNIST and CIFAR10 classification problems. Test results show that Gravity not only functions as a powerful instrument to robustify a model against state-of-the-art adversarial attacks but also effectively improves the model training accuracy.

摘要: 提出了一种新的模型训练方法--自适应重力法，以增强深度神经网络分类器对敌意样本的鲁棒性。我们将与每一类相关联的模型参数/特征概念化为质量，其特征由其质心位置和特征围绕质心的扩散(距离的标准差)来表征。我们使用与每个簇相关联的质心来推导出在网络训练期间将不同类别的质心彼此推开的反重力。然后，我们定制了一个目标函数，目标是将每一类的特征集中到它们对应的新质心上，该质心是通过反重力获得的。这种方法导致不同质量之间的更大分离，并减少了特征在每个质心周围的扩散。结果，样本被推离对抗性示例可以映射到的空间，有效地增加了制作对抗性示例所需的扰动程度。我们将这个训练方案实现为迭代方法，每次迭代包括四个步骤：1)质心提取，2)反重力计算，3)质心重定位，4)重力训练。通过使用LeNet和ResNet110网络测量针对各种攻击模型(包括FGSM、MIM、BIM和PGD)的相应愚骗率，并以MNIST和CIFAR10分类问题为基准，来评估Graight的效率。测试结果表明，该方法不仅可以有效地提高模型的训练精度，而且可以有效地提高模型的训练精度。



## **21. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

线性二次控制的强化学习在成本操纵下的脆弱性 eess.SY

This paper is yet to be peer-reviewed; Typos are corrected in ver 2

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2203.05774v2)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification of the cost parameters will only lead to a bounded change in the optimal policy. The bound is linear on the amount of falsification the attacker can apply to the cost parameters. We propose an attack model where the attacker aims to mislead the agent into learning a `nefarious' policy by intentionally falsifying the cost parameters. We formulate the attack's problem as a convex optimization problem and develop necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the actual cost signal. The paper aims to raise people's awareness of the security threats faced by RL-enabled control systems.

摘要: 在这项工作中，我们通过操纵费用信号来研究线性二次高斯(LQG)代理的欺骗。我们表明，对成本参数的微小篡改只会导致最优策略的有限变化。该界限与攻击者可以应用于成本参数的伪造量呈线性关系。我们提出了一个攻击模型，其中攻击者旨在通过故意伪造成本参数来误导代理学习“邪恶的”策略。我们将攻击问题描述为一个凸优化问题，并给出了检验攻击者目标可达性的充要条件。我们展示了在两种类型的LQG学习器上的对抗操作：批处理RL学习器和自适应动态规划(ADP)学习器。我们的结果表明，在只有2.296%的成本数据被篡改的情况下，攻击者误导批次RL学习将车辆引向危险位置的“邪恶”策略。攻击者还可以通过始终如一地向学习者提供接近实际成本信号的伪造成本信号，逐渐诱骗ADP学习者学习相同的“邪恶”策略。本文旨在提高人们对启用RL的控制系统所面临的安全威胁的认识。



## **22. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线侦听的对策 cs.CR

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2112.01967v2)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Markus Heinrichs, Rainer Kronberger, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线电信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，如今无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过窃听标准通信信号，窃听者获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为一种新的对抗敌意无线传感的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行混淆。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **23. Transfer Attacks Revisited: A Large-Scale Empirical Study in Real Computer Vision Settings**

传输攻击重现：真实计算机视觉环境下的大规模实证研究 cs.CV

Accepted to IEEE Security & Privacy 2022

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.04063v1)

**Authors**: Yuhao Mao, Chong Fu, Saizhuo Wang, Shouling Ji, Xuhong Zhang, Zhenguang Liu, Jun Zhou, Alex X. Liu, Raheem Beyah, Ting Wang

**Abstracts**: One intriguing property of adversarial attacks is their "transferability" -- an adversarial example crafted with respect to one deep neural network (DNN) model is often found effective against other DNNs as well. Intensive research has been conducted on this phenomenon under simplistic controlled conditions. Yet, thus far, there is still a lack of comprehensive understanding about transferability-based attacks ("transfer attacks") in real-world environments.   To bridge this critical gap, we conduct the first large-scale systematic empirical study of transfer attacks against major cloud-based MLaaS platforms, taking the components of a real transfer attack into account. The study leads to a number of interesting findings which are inconsistent to the existing ones, including: (1) Simple surrogates do not necessarily improve real transfer attacks. (2) No dominant surrogate architecture is found in real transfer attacks. (3) It is the gap between posterior (output of the softmax layer) rather than the gap between logit (so-called $\kappa$ value) that increases transferability. Moreover, by comparing with prior works, we demonstrate that transfer attacks possess many previously unknown properties in real-world environments, such as (1) Model similarity is not a well-defined concept. (2) $L_2$ norm of perturbation can generate high transferability without usage of gradient and is a more powerful source than $L_\infty$ norm. We believe this work sheds light on the vulnerabilities of popular MLaaS platforms and points to a few promising research directions.

摘要: 对抗性攻击的一个耐人寻味的特性是它们的“可转移性”--针对一个深度神经网络(DNN)模型制作的对抗性示例通常也被发现对其他DNN有效。人们在简单化的控制条件下对这一现象进行了深入的研究。然而，到目前为止，对现实环境中基于可转移性的攻击(“传输攻击”)仍然缺乏全面的了解。为了弥补这一关键差距，我们首次进行了针对主要基于云的MLaaS平台的传输攻击的大规模系统实证研究，考虑了真实传输攻击的组件。这项研究导致了一些有趣的发现，这些发现与现有的发现不一致，包括：(1)简单的代理并不一定能改善真实的转移攻击。(2)在真实传输攻击中没有发现具有优势的代理体系结构。(3)提高可转移性的是后验之间的差距(Softmax层的输出)，而不是Logit之间的差距(所谓的$\kappa$值)。此外，通过与已有工作的比较，我们证明了传输攻击在现实环境中具有许多以前未知的性质，例如：(1)模型相似性不是一个定义良好的概念。(2)摄动的$L_2$范数可以在不使用梯度的情况下产生很高的可转移性，是一个比$L_inty$范数更强大的来源。我们相信，这项工作揭示了流行的MLaaS平台的漏洞，并指出了一些有前途的研究方向。



## **24. Adversarial Machine Learning Attacks Against Video Anomaly Detection Systems**

针对视频异常检测系统的对抗性机器学习攻击 cs.CV

**SubmitDate**: 2022-04-07    [paper-pdf](http://arxiv.org/pdf/2204.03141v1)

**Authors**: Furkan Mumcu, Keval Doshi, Yasin Yilmaz

**Abstracts**: Anomaly detection in videos is an important computer vision problem with various applications including automated video surveillance. Although adversarial attacks on image understanding models have been heavily investigated, there is not much work on adversarial machine learning targeting video understanding models and no previous work which focuses on video anomaly detection. To this end, we investigate an adversarial machine learning attack against video anomaly detection systems, that can be implemented via an easy-to-perform cyber-attack. Since surveillance cameras are usually connected to the server running the anomaly detection model through a wireless network, they are prone to cyber-attacks targeting the wireless connection. We demonstrate how Wi-Fi deauthentication attack, a notoriously easy-to-perform and effective denial-of-service (DoS) attack, can be utilized to generate adversarial data for video anomaly detection systems. Specifically, we apply several effects caused by the Wi-Fi deauthentication attack on video quality (e.g., slow down, freeze, fast forward, low resolution) to the popular benchmark datasets for video anomaly detection. Our experiments with several state-of-the-art anomaly detection models show that the attackers can significantly undermine the reliability of video anomaly detection systems by causing frequent false alarms and hiding physical anomalies from the surveillance system.

摘要: 视频中的异常检测是一个重要的计算机视觉问题，在包括自动视频监控在内的各种应用中都有应用。虽然针对图像理解模型的对抗性攻击已经得到了大量的研究，但针对视频理解模型的对抗性机器学习的研究还很少，也没有专门针对视频异常检测的工作。为此，我们研究了一种针对视频异常检测系统的对抗性机器学习攻击，该攻击可以通过易于执行的网络攻击来实现。由于监控摄像头通常通过无线网络连接到运行异常检测模型的服务器，因此它们容易受到针对无线连接的网络攻击。我们演示了如何利用Wi-Fi解除身份验证攻击，这是一种众所周知的易于执行和有效的拒绝服务(DoS)攻击，可以为视频异常检测系统生成敌意数据。具体地说，我们将Wi-Fi解除身份验证攻击对视频质量造成的几种影响(例如，减速、冻结、快进、低分辨率)应用于流行的基准数据集，用于视频异常检测。我们用几种最先进的异常检测模型进行的实验表明，攻击者可以通过频繁的误报警和对监控系统隐藏物理异常来显著破坏视频异常检测系统的可靠性。



## **25. Control barrier function based attack-recovery with provable guarantees**

具有可证明保证的基于控制屏障函数的攻击恢复 cs.SY

8 pages, 6 figures

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.03077v1)

**Authors**: Kunal Garg, Ricardo G. Sanfelice, Alvaro A. Cardenas

**Abstracts**: This paper studies provable security guarantees for cyber-physical systems (CPS) under actuator attacks. In particular, we consider CPS safety and propose a new attack-detection mechanism based on a zeroing control barrier function (ZCBF) condition. In addition we design an adaptive recovery mechanism based on how close the system is from violating safety. We show that the attack-detection mechanism is sound, i.e., there are no false negatives for adversarial attacks. Finally, we use a Quadratic Programming (QP) approach for online recovery (and nominal) control synthesis. We demonstrate the effectiveness of the proposed method in a simulation case study involving a quadrotor with an attack on its motors.

摘要: 研究了网络物理系统在执行器攻击下的可证明安全保证问题。特别是，我们考虑了CPS的安全性，提出了一种新的基于归零控制屏障函数(ZCBF)条件的攻击检测机制。此外，我们还设计了一种基于系统与违反安全的距离的自适应恢复机制。我们证明了攻击检测机制是健全的，即对于对抗性攻击没有漏报。最后，我们使用二次规划(QP)方法进行在线恢复(和标称)控制综合。在一个四旋翼发动机受到攻击的仿真案例研究中，我们证明了所提方法的有效性。



## **26. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02887v1)

**Authors**: Xu Han, Anmin Liu, Yifeng Xiong, Yanbo Fan, Kun He

**Abstracts**: Deep neural networks have shown to be very vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to benign inputs. After achieving impressive attack success rates in the white-box setting, more focus is shifted to black-box attacks. In either case, the common gradient-based approaches generally use the $sign$ function to generate perturbations at the end of the process. However, only a few works pay attention to the limitation of the $sign$ function. Deviation between the original gradient and the generated noises may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability, which is crucial for black-box attacks. To address this issue, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM) to improve the transferability of the crafted adversarial examples. Specifically, we use data rescaling to substitute the inefficient $sign$ function in gradient-based attacks without extra computational cost. We also propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method can be used in any gradient-based optimizations and is extensible to be integrated with various input transformation or ensemble methods for further improving the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our S-FGRM could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 深度神经网络已被证明非常容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒攻击中取得了令人印象深刻的攻击成功率之后，更多的注意力转移到了黑盒攻击上。在这两种情况下，常见的基于梯度的方法通常使用$SIGN$函数在过程结束时生成扰动。然而，只有少数著作注意到$SIGN$函数的局限性。原始梯度与产生的噪声之间的偏差可能会导致不准确的梯度更新估计和对抗性转移的次优解，这是黑盒攻击的关键。针对这一问题，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)来提高恶意例子的可转移性。具体地说，在基于梯度的攻击中，我们使用数据重缩放来代替低效的$sign$函数，而不需要额外的计算代价。我们还提出了深度优先采样的方法，消除了重缩放的波动，稳定了梯度更新。我们的方法可以用于任何基于梯度的优化，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗性转移。在标准ImageNet数据集上的大量实验表明，我们的S-FGRM可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **27. Distilling Robust and Non-Robust Features in Adversarial Examples by Information Bottleneck**

利用信息瓶颈从对抗性实例中提取稳健和非稳健特征 cs.LG

NeurIPS 2021

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02735v1)

**Authors**: Junho Kim, Byung-Kwan Lee, Yong Man Ro

**Abstracts**: Adversarial examples, generated by carefully crafted perturbation, have attracted considerable attention in research fields. Recent works have argued that the existence of the robust and non-robust features is a primary cause of the adversarial examples, and investigated their internal interactions in the feature space. In this paper, we propose a way of explicitly distilling feature representation into the robust and non-robust features, using Information Bottleneck. Specifically, we inject noise variation to each feature unit and evaluate the information flow in the feature representation to dichotomize feature units either robust or non-robust, based on the noise variation magnitude. Through comprehensive experiments, we demonstrate that the distilled features are highly correlated with adversarial prediction, and they have human-perceptible semantic information by themselves. Furthermore, we present an attack mechanism intensifying the gradient of non-robust features that is directly related to the model prediction, and validate its effectiveness of breaking model robustness.

摘要: 由精心设计的扰动产生的对抗性例子在研究领域引起了相当大的关注。最近的工作认为，稳健特征和非稳健特征的存在是造成对抗性例子的主要原因，并研究了它们在特征空间中的内在交互作用。在本文中，我们提出了一种利用信息瓶颈将特征表示显式提取为稳健和非稳健特征的方法。具体地说，我们将噪声变化注入到每个特征单元中，并评估特征表示中的信息流，以基于噪声变化的大小来区分稳健或非稳健的特征单元。通过综合实验，我们证明所提取的特征与对抗性预测高度相关，并且它们本身就具有人类可感知的语义信息。此外，提出了一种增强与模型预测直接相关的非稳健特征梯度的攻击机制，并验证了其打破模型稳健性的有效性。



## **28. Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition**

滚动颜色：对抗红绿灯识别的激光攻击 cs.CV

To be published in USENIX Security 2022

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02675v1)

**Authors**: Chen Yan, Zhijian Xu, Zhanyuan Yin, Xiaoyu Ji, Wenyuan Xu

**Abstracts**: Traffic light recognition is essential for fully autonomous driving in urban areas. In this paper, we investigate the feasibility of fooling traffic light recognition mechanisms by shedding laser interference on the camera. By exploiting the rolling shutter of CMOS sensors, we manage to inject a color stripe overlapped on the traffic light in the image, which can cause a red light to be recognized as a green light or vice versa. To increase the success rate, we design an optimization method to search for effective laser parameters based on empirical models of laser interference. Our evaluation in emulated and real-world setups on 2 state-of-the-art recognition systems and 5 cameras reports a maximum success rate of 30% and 86.25% for Red-to-Green and Green-to-Red attacks. We observe that the attack is effective in continuous frames from more than 40 meters away against a moving vehicle, which may cause end-to-end impacts on self-driving such as running a red light or emergency stop. To mitigate the threat, we propose redesigning the rolling shutter mechanism.

摘要: 红绿灯识别是城市地区实现全自动驾驶的关键。在本文中，我们研究了通过在摄像机上散布激光干涉来欺骗交通灯识别机制的可行性。通过利用CMOS传感器的滚动快门，我们成功地在图像中的交通灯上注入了重叠的彩色条纹，这可以使红灯被识别为绿灯，反之亦然。为了提高成功率，我们设计了一种基于激光干涉经验模型的优化方法来搜索有效的激光参数。我们在2个最先进的识别系统和5个摄像头上的模拟和真实设置中的评估报告，红色到绿色和绿色到红色攻击的最大成功率分别为30%和86.25%。我们观察到，攻击在40米以外的连续帧中对移动的车辆有效，这可能会对自动驾驶造成端到端的影响，如闯红灯或紧急停车。为了减轻威胁，我们建议重新设计滚动快门机构。



## **29. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

网络物理关键基础设施中非私有联邦学习的对抗性分析 cs.CR

11 pages, 5 figures, 4 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2204.02654v1)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung, La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstracts**: Differential privacy (DP) is considered to be an effective privacy-preservation method to secure the promising distributed machine learning (ML) paradigm-federated learning (FL) from privacy attacks (e.g., membership inference attack). Nevertheless, while the DP mechanism greatly alleviates privacy concerns, recent studies have shown that it can be exploited to conduct security attacks (e.g., false data injection attacks). To address such attacks on FL-based applications in critical infrastructures, in this paper, we perform the first systematic study on the DP-exploited poisoning attacks from an adversarial point of view. We demonstrate that the DP method, despite providing a level of privacy guarantee, can effectively open a new poisoning attack vector for the adversary. Our theoretical analysis and empirical evaluation of a smart grid dataset show the FL performance degradation (sub-optimal model generation) scenario due to the differential noise-exploited selective model poisoning attacks. As a countermeasure, we propose a reinforcement learning-based differential privacy level selection (rDP) process. The rDP process utilizes the differential privacy parameters (privacy loss, information leakage probability, etc.) and the losses to intelligently generate an optimal privacy level for the nodes. The evaluation shows the accumulated reward and errors of the proposed technique converge to an optimal privacy policy.

摘要: 差分隐私(DP)被认为是一种有效的隐私保护方法，可以保护分布式机器学习(ML)范型联合学习(FL)免受隐私攻击(如成员推理攻击)。然而，虽然DP机制极大地缓解了对隐私的担忧，但最近的研究表明，它可以被利用来进行安全攻击(例如，虚假数据注入攻击)。为了解决这类针对关键基础设施中基于FL的应用程序的攻击，本文首次从对抗的角度对DP利用的中毒攻击进行了系统的研究。我们证明，虽然DP方法提供了一定程度的隐私保障，但可以有效地为攻击者打开一个新的中毒攻击载体。我们的理论分析和对智能电网数据集的经验评估表明，差值噪声利用的选择性模型中毒攻击导致FL性能下降(次优模型生成)。作为对策，我们提出了一种基于强化学习的差异隐私级别选择(RDP)过程。RDP过程使用不同的隐私参数(隐私丢失、信息泄露概率等)。以及智能地为节点生成最佳隐私级别的损失。评估结果表明，该技术的累积奖赏和误差收敛于最优隐私策略。



## **30. Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL**

谁是最强大的敌人？面向最优高效逃避攻击的Deep RL cs.LG

In the 10th International Conference on Learning Representations  (ICLR 2022)

**SubmitDate**: 2022-04-06    [paper-pdf](http://arxiv.org/pdf/2106.05087v4)

**Authors**: Yanchao Sun, Ruijie Zheng, Yongyuan Liang, Furong Huang

**Abstracts**: Evaluating the worst-case performance of a reinforcement learning (RL) agent under the strongest/optimal adversarial perturbations on state observations (within some constraints) is crucial for understanding the robustness of RL agents. However, finding the optimal adversary is challenging, in terms of both whether we can find the optimal attack and how efficiently we can find it. Existing works on adversarial RL either use heuristics-based methods that may not find the strongest adversary, or directly train an RL-based adversary by treating the agent as a part of the environment, which can find the optimal adversary but may become intractable in a large state space. This paper introduces a novel attacking method to find the optimal attacks through collaboration between a designed function named "actor" and an RL-based learner named "director". The actor crafts state perturbations for a given policy perturbation direction, and the director learns to propose the best policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically optimal and significantly more efficient than prior RL-based works in environments with large state spaces. Empirical results show that our proposed PA-AD universally outperforms state-of-the-art attacking methods in various Atari and MuJoCo environments. By applying PA-AD to adversarial training, we achieve state-of-the-art empirical robustness in multiple tasks under strong adversaries.

摘要: 评估强化学习(RL)代理在状态观测(在某些约束范围内)的最强/最优对抗扰动下的最坏情况下的性能对于理解RL代理的稳健性至关重要。然而，就我们是否能找到最佳攻击以及找到最佳攻击的效率而言，找到最佳对手是具有挑战性的。现有的对抗性RL研究要么使用基于启发式的方法，可能找不到最强的对手，要么将智能体视为环境的一部分，直接训练基于RL的对手，这可以找到最优的对手，但在大的状态空间中可能变得难以处理。本文提出了一种新的攻击方法，通过设计一个名为“参与者”的函数和一个名为“导演”的基于RL的学习器之间的协作来寻找最优攻击。参与者为给定的政策扰动方向制作状态扰动，导演学习提出最佳政策扰动方向。我们提出的算法PA-AD在理论上是最优的，并且在具有大状态空间的环境中比以前的基于RL的工作更有效。实验结果表明，在不同的Atari和MuJoCo环境下，我们提出的PA-AD攻击方法的性能普遍优于最新的攻击方法。通过将PA-AD应用于对抗性训练，我们在强对手下的多任务中获得了最先进的经验稳健性。



## **31. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

补丁-愚人：视觉变形金刚在对抗敌方干扰时总是健壮吗？ cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2203.08392v2)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.

摘要: 视觉转换器(VITS)最近掀起了神经结构设计的新浪潮，这要归功于它们在各种视觉任务中的创纪录表现。与此同时，为了实现将VITS部署到现实世界视觉应用中的目标，它们对潜在恶意攻击的健壮性得到了越来越多的关注。特别是，最近的研究表明，与卷积神经网络(CNN)相比，VITS对对抗攻击具有更强的鲁棒性，推测这是因为VITS更注重捕捉不同输入/特征块之间的全局交互，从而提高了它们对敌对攻击造成的局部扰动的鲁棒性。在这项工作中，我们提出了一个耐人寻味的问题：“在什么样的扰动下，VITS比CNN更容易成为学习者？”在这个问题的驱动下，我们首先对VITS和CNN在各种现有的对抗性攻击下的健壮性进行了全面的实验，以了解有利于其健壮性的潜在原因。在此基础上，我们提出了一个专门的攻击框架，称为Patch-Fool，它通过使用一系列注意力感知优化技术来攻击自我注意机制的基本组成部分(即单个补丁)来愚弄自我注意机制。有趣的是，我们的Patch-Fool框架首次表明，VITS在对抗对手扰动时并不一定比CNN更健壮。特别是，我们发现VITS比CNN更容易学习，这在广泛的实验中是一致的，并且来自Patch-Fool的两个变种稀疏/温和Patch-Fool的观察表明，每个补丁上的扰动密度和强度似乎是影响VITS和CNN之间健壮性排名的关键因素。



## **32. Exploring Robust Architectures for Deep Artificial Neural Networks**

探索深度人工神经网络的健壮体系结构 cs.LG

27 pages, 16 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2106.15850v2)

**Authors**: Asim Waqas, Ghulam Rasool, Hamza Farooq, Nidhal C. Bouaynaya

**Abstracts**: The architectures of deep artificial neural networks (DANNs) are routinely studied to improve their predictive performance. However, the relationship between the architecture of a DANN and its robustness to noise and adversarial attacks is less explored. We investigate how the robustness of DANNs relates to their underlying graph architectures or structures. This study: (1) starts by exploring the design space of architectures of DANNs using graph-theoretic robustness measures; (2) transforms the graphs to DANN architectures to train/validate/test on various image classification tasks; (3) explores the relationship between the robustness of trained DANNs against noise and adversarial attacks and the robustness of their underlying architectures estimated via graph-theoretic measures. We show that the topological entropy and Olivier-Ricci curvature of the underlying graphs can quantify the robustness performance of DANNs. The said relationship is stronger for complex tasks and large DANNs. Our work will allow autoML and neural architecture search community to explore design spaces of robust and accurate DANNs.

摘要: 人们经常研究深度人工神经网络(DEN)的结构，以提高其预测性能。然而，DANN的体系结构与其对噪声和敌意攻击的稳健性之间的关系却鲜有人研究。我们研究了DNA的健壮性如何与其底层的图体系结构或结构相关。本研究：(1)使用图论稳健性度量方法探索DANN体系结构的设计空间；(2)将图转换为DANN体系结构，以对各种图像分类任务进行训练/验证/测试；(3)探索训练的DANN体系结构对噪声和敌对攻击的健壮性与其底层体系结构的健壮性之间的关系。我们证明了基础图的拓扑熵和Olivier-Ricci曲率可以量化DANS的稳健性。对于复杂的任务和大的丹尼，上述关系更加牢固。我们的工作将允许AutoML和神经架构搜索社区探索健壮和准确的DAN的设计空间。



## **33. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

联合学习中抵抗语音情感识别属性推理攻击的用户级差分隐私 cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02500v1)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.

摘要: 许多现有的隐私增强型语音情感识别(SER)框架专注于通过集中式机器学习设置中的对抗性训练来扰乱原始语音数据。然而，这种隐私保护方案可能会失败，因为攻击者仍然可以访问受干扰的数据。近年来，分布式学习算法，特别是联邦学习(FL)算法在机器学习应用中保护隐私得到了广泛的应用。虽然FL通过将数据保存在本地设备上来提供良好的直觉来保护隐私，但先前的工作表明，使用FL训练的SER系统可以实现隐私攻击，例如属性推理攻击。在这项工作中，我们建议评估用户级差异隐私(UDP)在缓解FL中SER系统的隐私泄漏方面的作用。UDP通过隐私参数$\epsilon$和$\Delta$提供理论上的隐私保证。实验结果表明，UDP协议在保持SER系统可用性的同时，有效地减少了属性信息泄露，且攻击者只需访问一次模型更新。然而，当FL系统向对手泄露更多的模型更新时，UDP的效率会受到影响。我们将代码公开，以便在https://github.com/usc-sail/fed-ser-leakage.中重现结果



## **34. Training-Free Robust Multimodal Learning via Sample-Wise Jacobian Regularization**

基于样本明智雅可比正则化的免训练鲁棒多模学习 cs.CV

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02485v1)

**Authors**: Zhengqi Gao, Sucheng Ren, Zihui Xue, Siting Li, Hang Zhao

**Abstracts**: Multimodal fusion emerges as an appealing technique to improve model performances on many tasks. Nevertheless, the robustness of such fusion methods is rarely involved in the present literature. In this paper, we propose a training-free robust late-fusion method by exploiting conditional independence assumption and Jacobian regularization. Our key is to minimize the Frobenius norm of a Jacobian matrix, where the resulting optimization problem is relaxed to a tractable Sylvester equation. Furthermore, we provide a theoretical error bound of our method and some insights about the function of the extra modality. Several numerical experiments on AV-MNIST, RAVDESS, and VGGsound demonstrate the efficacy of our method under both adversarial attacks and random corruptions.

摘要: 多通道融合是提高模型在许多任务上性能的一种很有吸引力的技术。然而，这种融合方法的稳健性在目前的文献中很少涉及。本文利用条件独立性假设和雅可比正则化，提出了一种无需训练的鲁棒晚融合方法。我们的关键是最小化雅可比矩阵的Frobenius范数，由此产生的优化问题被松弛到一个容易处理的Sylvester方程。此外，我们还给出了该方法的理论误差界，并对额外通道的作用提出了一些见解。在AV-MNIST、RAVDESS和VGGound上的几个数值实验证明了我们的方法在对抗攻击和随机破坏下的有效性。



## **35. Hear No Evil: Towards Adversarial Robustness of Automatic Speech Recognition via Multi-Task Learning**

听而不闻：通过多任务学习实现自动语音识别的对抗健壮性 eess.AS

Submitted to Insterspeech 2022

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.02381v1)

**Authors**: Nilaksh Das, Duen Horng Chau

**Abstracts**: As automatic speech recognition (ASR) systems are now being widely deployed in the wild, the increasing threat of adversarial attacks raises serious questions about the security and reliability of using such systems. On the other hand, multi-task learning (MTL) has shown success in training models that can resist adversarial attacks in the computer vision domain. In this work, we investigate the impact of performing such multi-task learning on the adversarial robustness of ASR models in the speech domain. We conduct extensive MTL experimentation by combining semantically diverse tasks such as accent classification and ASR, and evaluate a wide range of adversarial settings. Our thorough analysis reveals that performing MTL with semantically diverse tasks consistently makes it harder for an adversarial attack to succeed. We also discuss in detail the serious pitfalls and their related remedies that have a significant impact on the robustness of MTL models. Our proposed MTL approach shows considerable absolute improvements in adversarially targeted WER ranging from 17.25 up to 59.90 compared to single-task learning baselines (attention decoder and CTC respectively). Ours is the first in-depth study that uncovers adversarial robustness gains from multi-task learning for ASR.

摘要: 随着自动语音识别(ASR)系统的广泛应用，日益增长的对抗性攻击威胁对使用这类系统的安全性和可靠性提出了严重的问题。另一方面，多任务学习(MTL)在训练模型抵抗计算机视觉领域中的敌意攻击方面取得了成功。在这项工作中，我们研究了执行这种多任务学习对ASR模型在语音域的对抗健壮性的影响。我们通过结合重音分类和ASR等语义多样化的任务来进行广泛的MTL实验，并评估了广泛的对抗性环境。我们的全面分析表明，以语义多样化的任务执行MTL始终会使敌方攻击更难成功。我们还详细讨论了对MTL模型的稳健性有重大影响的严重陷阱及其相关补救措施。与单任务学习基线(注意解码器和CTC)相比，我们提出的MTL方法在相反的目标WER上有相当大的绝对改善，从17.25%到59.90%。我们的研究是第一次深入研究ASR从多任务学习中获得的对手健壮性收益。



## **36. A Survey of Adversarial Learning on Graphs**

图上的对抗性学习研究综述 cs.LG

Preprint; 16 pages, 2 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2003.05730v3)

**Authors**: Liang Chen, Jintang Li, Jiaying Peng, Tao Xie, Zengxu Cao, Kun Xu, Xiangnan He, Zibin Zheng, Bingzhe Wu

**Abstracts**: Deep learning models on graphs have achieved remarkable performance in various graph analysis tasks, e.g., node classification, link prediction, and graph clustering. However, they expose uncertainty and unreliability against the well-designed inputs, i.e., adversarial examples. Accordingly, a line of studies has emerged for both attack and defense addressed in different graph analysis tasks, leading to the arms race in graph adversarial learning. Despite the booming works, there still lacks a unified problem definition and a comprehensive review. To bridge this gap, we investigate and summarize the existing works on graph adversarial learning tasks systemically. Specifically, we survey and unify the existing works w.r.t. attack and defense in graph analysis tasks, and give appropriate definitions and taxonomies at the same time. Besides, we emphasize the importance of related evaluation metrics, investigate and summarize them comprehensively. Hopefully, our works can provide a comprehensive overview and offer insights for the relevant researchers. Latest advances in graph adversarial learning are summarized in our GitHub repository https://github.com/EdisonLeeeee/Graph-Adversarial-Learning.

摘要: 图的深度学习模型在各种图分析任务中取得了显著的性能，如节点分类、链接预测、图聚类等。然而，它们暴露了相对于精心设计的投入的不确定性和不可靠性，即对抗性例子。相应地，在不同的图分析任务中出现了一系列针对攻击和防御的研究，导致了图对抗学习中的军备竞赛。尽管工作开展得如火如荼，但仍缺乏统一的问题定义和全面审查。为了弥补这一差距，我们系统地调查和总结了已有的关于图对抗性学习任务的工作。具体地说，我们对现有的作品进行了调查和统一。图分析任务中的攻击和防御，同时给出相应的定义和分类。此外，我们还强调了相关评价指标的重要性，并对其进行了全面的调查和总结。希望我们的工作能够提供一个全面的概述，并为相关研究人员提供见解。在我们的GitHub知识库https://github.com/EdisonLeeeee/Graph-Adversarial-Learning.中总结了图形对抗性学习的最新进展



## **37. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

通过提高不可察觉来理解和改进图注入攻击 cs.LG

ICLR2022, 42 pages, 22 figures

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.08057v2)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.

摘要: 图注入攻击(GIA)是近年来在图神经网络(GNN)上出现的一种实用攻击方案，即攻击者只能注入少量恶意节点，而不需要修改已有的节点或边，即图修改攻击(GMA)。尽管GIA取得了令人振奋的成果，但人们对它为什么成功以及成功背后是否存在陷阱知之甚少。为了理解GIA的力量，我们将其与GMA进行比较，发现由于其相对较高的灵活性，GIA显然比GMA更具危害性。然而，较高的灵活性也会对原图的同源分布造成很大的破坏，即邻域间的相似性。因此，GIA的威胁可以很容易地减轻，甚至可以通过基于同质性的防御措施来恢复原始的同质性。为了缓解这一问题，我们引入了一种新的约束--同形不可察觉，强制GIA保持同形，并提出了和谐对抗目标(HAO)来实例化它。广泛的实验证明，带有HAO的GIA可以打破基于同源的防御，并显著超过之前的GIA攻击。我们相信，我们的方法可以更可靠地评估GNN的稳健性。



## **38. Adversarial Detection without Model Information**

无模型信息的对抗性检测 cs.CV

This paper has 14 pages of content and 2 pages of references

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2202.04271v2)

**Authors**: Abhishek Moitra, Youngeun Kim, Priyadarshini Panda

**Abstracts**: Prior state-of-the-art adversarial detection works are classifier model dependent, i.e., they require classifier model outputs and parameters for training the detector or during adversarial detection. This makes their detection approach classifier model specific. Furthermore, classifier model outputs and parameters might not always be accessible. To this end, we propose a classifier model independent adversarial detection method using a simple energy function to distinguish between adversarial and natural inputs. We train a standalone detector independent of the classifier model, with a layer-wise energy separation (LES) training to increase the separation between natural and adversarial energies. With this, we perform energy distribution-based adversarial detection. Our method achieves comparable performance with state-of-the-art detection works (ROC-AUC > 0.9) across a wide range of gradient, score and gaussian noise attacks on CIFAR10, CIFAR100 and TinyImagenet datasets. Furthermore, compared to prior works, our detection approach is light-weight, requires less amount of training data (40% of the actual dataset) and is transferable across different datasets. For reproducibility, we provide layer-wise energy separation training code at https://github.com/Intelligent-Computing-Lab-Yale/Energy-Separation-Training

摘要: 以往的对抗性检测工作依赖于分类器模型，即它们需要分类器模型的输出和参数来训练检测器或在对抗性检测过程中。这使得它们的检测方法具有分类器模型的特殊性。此外，分类器模型输出和参数可能并不总是可访问的。为此，我们提出了一种独立于分类器模型的敌意检测方法，该方法使用一个简单的能量函数来区分敌意输入和自然输入。我们训练一个独立于分类器模型的独立检测器，通过分层能量分离(LES)训练来增加自然能量和敌对能量之间的分离。在此基础上，我们进行了基于能量分布的敌意检测。我们的方法在CIFAR10、CIFAR100和TinyImagenet数据集上的各种梯度、得分和高斯噪声攻击下获得了与最先进的检测工作(ROC-AUC>0.9)相当的性能。此外，与以前的工作相比，我们的检测方法是轻量级的，需要更少的训练数据量(实际数据集的40%)，并且可以在不同的数据集之间传输。为了重现性，我们在https://github.com/Intelligent-Computing-Lab-Yale/Energy-Separation-Training上提供了分层能量分离培训代码



## **39. GAIL-PT: A Generic Intelligent Penetration Testing Framework with Generative Adversarial Imitation Learning**

GAIL-PT：一种具有生成性对抗性模仿学习的通用智能渗透测试框架 cs.CR

**SubmitDate**: 2022-04-05    [paper-pdf](http://arxiv.org/pdf/2204.01975v1)

**Authors**: Jinyin Chen, Shulong Hu, Haibin Zheng, Changyou Xing, Guomin Zhang

**Abstracts**: Penetration testing (PT) is an efficient network testing and vulnerability mining tool by simulating a hacker's attack for valuable information applied in some areas. Compared with manual PT, intelligent PT has become a dominating mainstream due to less time-consuming and lower labor costs. Unfortunately, RL-based PT is still challenged in real exploitation scenarios because the agent's action space is usually high-dimensional discrete, thus leading to algorithm convergence difficulty. Besides, most PT methods still rely on the decisions of security experts. Addressing the challenges, for the first time, we introduce expert knowledge to guide the agent to make better decisions in RL-based PT and propose a Generative Adversarial Imitation Learning-based generic intelligent Penetration testing framework, denoted as GAIL-PT, to solve the problems of higher labor costs due to the involvement of security experts and high-dimensional discrete action space. Specifically, first, we manually collect the state-action pairs to construct an expert knowledge base when the pre-trained RL / DRL model executes successful penetration testings. Second, we input the expert knowledge and the state-action pairs generated online by the different RL / DRL models into the discriminator of GAIL for training. At last, we apply the output reward of the discriminator to guide the agent to perform the action with a higher penetration success rate to improve PT's performance. Extensive experiments conducted on the real target host and simulated network scenarios show that GAIL-PT achieves the SOTA penetration performance against DeepExploit in exploiting actual target Metasploitable2 and Q-learning in optimizing penetration path, not only in small-scale with or without honey-pot network environments but also in the large-scale virtual network environment.

摘要: 渗透测试(PT)是一种有效的网络测试和漏洞挖掘工具，通过模拟黑客对某些领域应用的有价值信息的攻击而实现。与人工PT相比，智能PT由于耗时更少、人力成本更低而成为主流。遗憾的是，基于RL的PT在实际开发场景中仍然面临挑战，因为智能体的动作空间通常是高维离散的，从而导致算法收敛困难。此外，大多数PT方法仍然依赖于安全专家的决策。针对这些挑战，我们首次在基于RL的PT中引入专家知识来指导智能体做出更好的决策，并提出了一种基于生成性对抗模仿学习的通用智能渗透测试框架GAIL-PT，以解决由于安全专家的参与和高维离散动作空间而导致的人工成本较高的问题。具体地说，首先，当预先训练的RL/DRL模型执行成功的渗透测试时，我们手动收集状态-动作对来构建专家知识库。其次，将不同RL/DRL模型在线生成的专家知识和状态-动作对输入到GAIL的鉴别器中进行训练。最后，我们利用鉴别器的输出奖励来指导智能体执行具有较高渗透成功率的动作，以提高PT的性能。在真实目标主机和模拟网络场景上进行的大量实验表明，无论是在有或没有蜜罐网络环境中，Gail-PT在利用实际目标元可分性2和Q-学习优化穿透路径方面都达到了DeepDevelopit的SOTA穿透性能。



## **40. Recent improvements of ASR models in the face of adversarial attacks**

面对对抗性攻击的ASR模型的最新改进 cs.CR

Submitted to Interspeech 2022

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2203.16536v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Like many other tasks involving neural networks, Speech Recognition models are vulnerable to adversarial attacks. However recent research has pointed out differences between attacks and defenses on ASR models compared to image models. Improving the robustness of ASR models requires a paradigm shift from evaluating attacks on one or a few models to a systemic approach in evaluation. We lay the ground for such research by evaluating on various architectures a representative set of adversarial attacks: targeted and untargeted, optimization and speech processing-based, white-box, black-box and targeted attacks. Our results show that the relative strengths of different attack algorithms vary considerably when changing the model architecture, and that the results of some attacks are not to be blindly trusted. They also indicate that training choices such as self-supervised pretraining can significantly impact robustness by enabling transferable perturbations. We release our source code as a package that should help future research in evaluating their attacks and defenses.

摘要: 像涉及神经网络的许多其他任务一样，语音识别模型容易受到敌意攻击。然而，最近的研究指出，与图像模型相比，ASR模型在攻击和防御方面存在差异。要提高ASR模型的稳健性，需要从评估针对一个或几个模型的攻击转变为评估的系统性方法。我们通过在不同的体系结构上评估一组具有代表性的对抗性攻击：目标攻击和非目标攻击、基于优化和语音处理的攻击、白盒攻击、黑盒攻击和目标攻击，为这类研究奠定了基础。结果表明，随着模型结构的改变，不同攻击算法的相对强度有很大差异，某些攻击的结果不能盲目信任。它们还表明，自我监督预训练等训练选择可以通过实现可转移的扰动来显著影响稳健性。我们将我们的源代码作为一个包发布，这应该有助于未来的研究评估他们的攻击和防御。



## **41. Experimental quantum adversarial learning with programmable superconducting qubits**

基于可编程超导量子比特的实验量子对抗学习 quant-ph

26 pages, 17 figures, 8 algorithms

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01738v1)

**Authors**: Wenhui Ren, Weikang Li, Shibo Xu, Ke Wang, Wenjie Jiang, Feitong Jin, Xuhao Zhu, Jiachen Chen, Zixuan Song, Pengfei Zhang, Hang Dong, Xu Zhang, Jinfeng Deng, Yu Gao, Chuanyu Zhang, Yaozu Wu, Bing Zhang, Qiujiang Guo, Hekang Li, Zhen Wang, Jacob Biamonte, Chao Song, Dong-Ling Deng, H. Wang

**Abstracts**: Quantum computing promises to enhance machine learning and artificial intelligence. Different quantum algorithms have been proposed to improve a wide spectrum of machine learning tasks. Yet, recent theoretical works show that, similar to traditional classifiers based on deep classical neural networks, quantum classifiers would suffer from the vulnerability problem: adding tiny carefully-crafted perturbations to the legitimate original data samples would facilitate incorrect predictions at a notably high confidence level. This will pose serious problems for future quantum machine learning applications in safety and security-critical scenarios. Here, we report the first experimental demonstration of quantum adversarial learning with programmable superconducting qubits. We train quantum classifiers, which are built upon variational quantum circuits consisting of ten transmon qubits featuring average lifetimes of 150 $\mu$s, and average fidelities of simultaneous single- and two-qubit gates above 99.94% and 99.4% respectively, with both real-life images (e.g., medical magnetic resonance imaging scans) and quantum data. We demonstrate that these well-trained classifiers (with testing accuracy up to 99%) can be practically deceived by small adversarial perturbations, whereas an adversarial training process would significantly enhance their robustness to such perturbations. Our results reveal experimentally a crucial vulnerability aspect of quantum learning systems under adversarial scenarios and demonstrate an effective defense strategy against adversarial attacks, which provide a valuable guide for quantum artificial intelligence applications with both near-term and future quantum devices.

摘要: 量子计算有望增强机器学习和人工智能。已经提出了不同的量子算法来改进广泛的机器学习任务。然而，最近的理论研究表明，与基于深度经典神经网络的传统分类器类似，量子分类器将受到脆弱性问题的困扰：在合法的原始数据样本中添加精心设计的微小扰动，将有助于在相当高的置信度水平下进行错误预测。这将给未来量子机器学习在安全和安保关键场景中的应用带来严重问题。在这里，我们报告了第一个利用可编程超导量子比特进行量子对抗学习的实验演示。我们训练量子分类器，它建立在由10个传态量子比特组成的变分量子电路上，平均寿命为150$\MU$s，同时具有99.94%和99.4%以上的同时单量子比特门和双量子比特门的平均保真度，使用真实图像(例如医学磁共振成像扫描)和量子数据。我们证明了这些训练有素的分类器(测试准确率高达99%)实际上可以被微小的对抗性扰动所欺骗，而对抗性训练过程将显著增强它们对此类扰动的稳健性。我们的结果在实验上揭示了量子学习系统在对抗场景下的一个关键弱点，并展示了一种有效的防御策略，这为量子人工智能在近期和未来的量子设备应用提供了有价值的指导。



## **42. RobustSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

RobustSense：防御恶意攻击，实现安全的无设备人类活动识别 cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01560v1)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstracts**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, RobustSense, to defend common attacks. RobustSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.

摘要: 深度神经网络能够实现准确的无设备人体活动识别，具有广泛的应用前景。深度模型可以从各种传感器中提取稳健的特征，即使在数据不足等具有挑战性的情况下也能很好地推广。然而，这些系统可能容易受到输入扰动，即对抗性攻击。我们的经验证明，无论是黑盒高斯攻击还是现代对抗性白盒攻击，它们的准确率都会直线下降。在本文中，我们首先指出这种现象会给无设备感知系统带来严重的安全隐患，然后提出一种新的学习框架RobustSense来防御常见的攻击。RobustSense的目标是在输入是否存在攻击的情况下实现一致的预测，缓解因对抗性攻击而导致的分布扰动的负面影响。大量实验表明，该方法能够显著增强已有深度模型的模型稳健性，克服可能的攻击。实验结果表明，该方法在无线人体活动识别和身份识别系统中具有较好的效果。据我们所知，这是第一次在移动计算研究中研究对抗性攻击，并进一步开发出一种新的无线人类活动识别防御框架。



## **43. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

Prada：针对神经排序模型的实用黑箱对抗性攻击 cs.IR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01321v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Adversarial Document Ranking Attack (ADRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.

摘要: 近年来，神经网络排序模型(NRM)取得了显著的成功，尤其是使用了预先训练好的语言模型。然而，深层神经模型因其易受敌意例子的攻击而臭名昭著。鉴于我们对神经信息检索模型的日益依赖，对抗性攻击可能成为一种新型的Web垃圾邮件技术。因此，在部署NRM之前，研究潜在的敌意攻击以识别NRM的漏洞是很重要的。在本文中，我们引入了针对NRMS的对抗性文档排名攻击(ADRA)任务，其目的是通过在文本中添加对抗性扰动来提升目标文档的排名。重点研究了基于决策的黑盒攻击环境，其中攻击者无法获取模型参数和梯度，只能通过查询目标模型获得部分检索列表的排名位置。这种攻击设置在现实世界的搜索引擎中是现实的。提出了一种新的基于伪相关性的对抗性排序攻击方法(PRADA)，该方法通过学习基于伪相关反馈(PRF)的代理模型来生成用于发现对抗性扰动的梯度。在两个网络搜索基准数据集上的实验表明，Prada可以超越现有的攻击策略，并成功地利用文本的微小不可分辨扰动来欺骗NRM。



## **44. Captcha Attack: Turning Captchas Against Humanity**

Captcha攻击：使Captchas反人类 cs.CR

Currently under submission

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2201.04014v3)

**Authors**: Mauro Conti, Luca Pajola, Pier Paolo Tricomi

**Abstracts**: Nowadays, people generate and share massive content on online platforms (e.g., social networks, blogs). In 2021, the 1.9 billion daily active Facebook users posted around 150 thousand photos every minute. Content moderators constantly monitor these online platforms to prevent the spreading of inappropriate content (e.g., hate speech, nudity images). Based on deep learning (DL) advances, Automatic Content Moderators (ACM) help human moderators handle high data volume. Despite their advantages, attackers can exploit weaknesses of DL components (e.g., preprocessing, model) to affect their performance. Therefore, an attacker can leverage such techniques to spread inappropriate content by evading ACM.   In this work, we propose CAPtcha Attack (CAPA), an adversarial technique that allows users to spread inappropriate text online by evading ACM controls. CAPA, by generating custom textual CAPTCHAs, exploits ACM's careless design implementations and internal procedures vulnerabilities. We test our attack on real-world ACM, and the results confirm the ferocity of our simple yet effective attack, reaching up to a 100% evasion success in most cases. At the same time, we demonstrate the difficulties in designing CAPA mitigations, opening new challenges in CAPTCHAs research area.

摘要: 如今，人们在在线平台(如社交网络、博客)上生成和分享大量内容。2021年，Facebook的19亿日活跃用户每分钟发布约15万张照片。内容版主不断监控这些在线平台，以防止传播不恰当的内容(例如，仇恨言论、裸体图片)。基于深度学习(DL)的进步，自动内容审核者(ACM)帮助人工审核者处理大量数据。尽管具有优势，攻击者仍可以利用DL组件的弱点(例如，预处理、模型)来影响其性能。因此，攻击者可以利用这些技术通过规避ACM来传播不适当的内容。在这项工作中，我们提出了验证码攻击(CAPA)，这是一种敌意技术，允许用户通过逃避ACM控制来在线传播不适当的文本。通过生成自定义文本验证码，CAPA可以利用ACM粗心的设计实现和内部过程漏洞。我们在真实的ACM上测试了我们的攻击，结果证实了我们简单而有效的攻击的凶猛，在大多数情况下达到了100%的规避成功。同时，我们展示了设计CAPA缓解措施的困难，为CAPTCHAS研究领域开辟了新的挑战。



## **45. Detecting In-vehicle Intrusion via Semi-supervised Learning-based Convolutional Adversarial Autoencoders**

基于半监督学习的卷积对抗性自动编码器车载入侵检测 cs.CR

**SubmitDate**: 2022-04-04    [paper-pdf](http://arxiv.org/pdf/2204.01193v1)

**Authors**: Thien-Nu Hoang, Daehee Kim

**Abstracts**: With the development of autonomous vehicle technology, the controller area network (CAN) bus has become the de facto standard for an in-vehicle communication system because of its simplicity and efficiency. However, without any encryption and authentication mechanisms, the in-vehicle network using the CAN protocol is susceptible to a wide range of attacks. Many studies, which are mostly based on machine learning, have proposed installing an intrusion detection system (IDS) for anomaly detection in the CAN bus system. Although machine learning methods have many advantages for IDS, previous models usually require a large amount of labeled data, which results in high time and labor costs. To handle this problem, we propose a novel semi-supervised learning-based convolutional adversarial autoencoder model in this paper. The proposed model combines two popular deep learning models: autoencoder and generative adversarial networks. First, the model is trained with unlabeled data to learn the manifolds of normal and attack patterns. Then, only a small number of labeled samples are used in supervised training. The proposed model can detect various kinds of message injection attacks, such as DoS, fuzzy, and spoofing, as well as unknown attacks. The experimental results show that the proposed model achieves the highest F1 score of 0.99 and a low error rate of 0.1\% with limited labeled data compared to other supervised methods. In addition, we show that the model can meet the real-time requirement by analyzing the model complexity in terms of the number of trainable parameters and inference time. This study successfully reduced the number of model parameters by five times and the inference time by eight times, compared to a state-of-the-art model.

摘要: 随着自动驾驶汽车技术的发展，控制器局域网(CAN)总线以其简单高效的特点已成为车载通信系统的事实标准。然而，在没有任何加密和认证机制的情况下，使用CAN协议的车载网络容易受到广泛的攻击。许多研究大多基于机器学习，提出在CAN总线系统中安装入侵检测系统(入侵检测系统)进行异常检测。虽然机器学习方法在入侵检测中有很多优点，但是以前的模型通常需要大量的标记数据，这导致了很高的时间和人力成本。针对这一问题，本文提出了一种新的基于半监督学习的卷积对抗性自动编码器模型。该模型结合了两种流行的深度学习模型：自动编码器和产生式对抗网络。首先，用未标记的数据训练模型，学习正常模式和攻击模式的流形。然后，只使用少量的标记样本进行有监督的训练。该模型可以检测各种类型的消息注入攻击，如DoS、模糊攻击、欺骗攻击以及未知攻击。实验结果表明，与其他监督方法相比，该模型在标签数据有限的情况下获得了最高的F1值0.99和较低的错误率0.1。此外，从可训练参数个数和推理时间两个方面分析了模型的复杂性，结果表明该模型能够满足实时性的要求。与最先进的模型相比，本研究成功地将模型参数的数量减少了5倍，推理时间减少了8倍。



## **46. DST: Dynamic Substitute Training for Data-free Black-box Attack**

DST：无数据黑盒攻击的动态替补训练 cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-04-03    [paper-pdf](http://arxiv.org/pdf/2204.00972v1)

**Authors**: Wenxuan Wang, Xuelin Qian, Yanwei Fu, Xiangyang Xue

**Abstracts**: With the wide applications of deep neural network models in various computer vision tasks, more and more works study the model vulnerability to adversarial examples. For data-free black box attack scenario, existing methods are inspired by the knowledge distillation, and thus usually train a substitute model to learn knowledge from the target model using generated data as input. However, the substitute model always has a static network structure, which limits the attack ability for various target models and tasks. In this paper, we propose a novel dynamic substitute training attack method to encourage substitute model to learn better and faster from the target model. Specifically, a dynamic substitute structure learning strategy is proposed to adaptively generate optimal substitute model structure via a dynamic gate according to different target models and tasks. Moreover, we introduce a task-driven graph-based structure information learning constrain to improve the quality of generated training data, and facilitate the substitute model learning structural relationships from the target model multiple outputs. Extensive experiments have been conducted to verify the efficacy of the proposed attack method, which can achieve better performance compared with the state-of-the-art competitors on several datasets.

摘要: 随着深度神经网络模型在各种计算机视觉任务中的广泛应用，越来越多的工作研究了模型对对抗性例子的脆弱性。对于无数据黑盒攻击场景，现有的方法都是受到知识提炼的启发，因此通常训练一个替代模型，以生成的数据作为输入从目标模型学习知识。然而，替代模型总是具有静态的网络结构，这限制了对各种目标模型和任务的攻击能力。本文提出了一种新的动态替补训练攻击方法，以鼓励替补模型更好、更快地向目标模型学习。具体地，提出了一种动态替换结构学习策略，根据不同的目标模型和任务，通过动态门自适应地生成最优替换模型结构。此外，我们还引入了一种任务驱动的基于图的结构信息学习约束，以提高生成的训练数据的质量，并便于替换模型从目标模型的多个输出中学习结构关系。大量的实验验证了所提出的攻击方法的有效性，该方法在多个数据集上取得了比最先进的竞争对手更好的性能。



## **47. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

对抗性霓虹灯：对DNN的强大物理世界对抗性攻击 cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00853v1)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstracts**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.

摘要: 在物理世界中，光线会影响深度神经网络的性能。如今，许多基于深度神经网络的产品已经进入日常生活。关于光照对深度神经网络模型性能影响的研究很少。然而，光产生的对抗性扰动可能会对这些系统产生极其危险的影响。在这项工作中，我们提出了一种称为对抗性霓虹束的攻击方法(AdvNB)，该方法只需很少的查询就可以获得对抗性霓虹束的物理参数来执行物理攻击。实验表明，该算法在数字测试和物理测试中均能达到较好的攻击效果。在数字环境下，攻击成功率达到99.3%，在物理环境下，攻击成功率达到100%。与最先进的物理攻击方法相比，我们的方法可以实现更好的物理扰动隐藏。此外，通过对实验数据的分析，揭示了对抗性霓虹束攻击带来的一些新现象。



## **48. Precise Statistical Analysis of Classification Accuracies for Adversarial Training**

对抗性训练中分类精度的精确统计分析 stat.ML

80 pages; to appear in the Annals of Statistics

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2010.11213v2)

**Authors**: Adel Javanmard, Mahdi Soltanolkotabi

**Abstracts**: Despite the wide empirical success of modern machine learning algorithms and models in a multitude of applications, they are known to be highly susceptible to seemingly small indiscernible perturbations to the input data known as \emph{adversarial attacks}. A variety of recent adversarial training procedures have been proposed to remedy this issue. Despite the success of such procedures at increasing accuracy on adversarially perturbed inputs or \emph{robust accuracy}, these techniques often reduce accuracy on natural unperturbed inputs or \emph{standard accuracy}. Complicating matters further, the effect and trend of adversarial training procedures on standard and robust accuracy is rather counter intuitive and radically dependent on a variety of factors including the perceived form of the perturbation during training, size/quality of data, model overparameterization, etc. In this paper we focus on binary classification problems where the data is generated according to the mixture of two Gaussians with general anisotropic covariance matrices and derive a precise characterization of the standard and robust accuracy for a class of minimax adversarially trained models. We consider a general norm-based adversarial model, where the adversary can add perturbations of bounded $\ell_p$ norm to each input data, for an arbitrary $p\ge 1$. Our comprehensive analysis allows us to theoretically explain several intriguing empirical phenomena and provide a precise understanding of the role of different problem parameters on standard and robust accuracies.

摘要: 尽管现代机器学习算法和模型在许多应用中取得了广泛的经验上的成功，但众所周知，它们对输入数据的看起来很小、难以辨别的扰动非常敏感，称为\emph(对抗性攻击)。最近提出了各种对抗性训练程序来解决这个问题。尽管这些程序成功地提高了反向扰动输入的准确性或\emph{稳健精度}，但这些技术往往会降低自然扰动输入的准确性或\emph{标准精度}。更复杂的是，对抗性训练过程对标准和稳健精度的影响和趋势是相当违反直觉的，并且从根本上依赖于各种因素，包括训练过程中所感知的扰动形式、数据的大小/质量、模型的过度参数化等。在本文中，我们关注根据两个高斯和一般各向异性协方差矩阵的混合来生成数据的二进制分类问题，并推导了一类极小极大对抗性训练模型的标准和稳健精度的精确表征。我们考虑了一个一般的基于范数的对抗性模型，其中对手可以对每个输入数据添加有界的$\p$范数的扰动，对于任意的$p\ge 1$。我们的全面分析使我们能够从理论上解释几个有趣的经验现象，并提供对不同问题参数对标准和稳健精度的作用的精确理解。



## **49. SkeleVision: Towards Adversarial Resiliency of Person Tracking with Multi-Task Learning**

SkeleVision：基于多任务学习的人跟踪的对抗性 cs.CV

**SubmitDate**: 2022-04-02    [paper-pdf](http://arxiv.org/pdf/2204.00734v1)

**Authors**: Nilaksh Das, Sheng-Yun Peng, Duen Horng Chau

**Abstracts**: Person tracking using computer vision techniques has wide ranging applications such as autonomous driving, home security and sports analytics. However, the growing threat of adversarial attacks raises serious concerns regarding the security and reliability of such techniques. In this work, we study the impact of multi-task learning (MTL) on the adversarial robustness of the widely used SiamRPN tracker, in the context of person tracking. Specifically, we investigate the effect of jointly learning with semantically analogous tasks of person tracking and human keypoint detection. We conduct extensive experiments with more powerful adversarial attacks that can be physically realizable, demonstrating the practical value of our approach. Our empirical study with simulated as well as real-world datasets reveals that training with MTL consistently makes it harder to attack the SiamRPN tracker, compared to typically training only on the single task of person tracking.

摘要: 使用计算机视觉技术的人物跟踪具有广泛的应用，如自动驾驶、家庭安全和体育分析。然而，对抗性攻击的威胁越来越大，这引起了人们对这种技术的安全性和可靠性的严重关切。在这项工作中，我们研究了多任务学习(MTL)对广泛使用的SiamRPN跟踪器的对抗健壮性的影响，在个人跟踪的背景下。具体地说，我们考察了联合学习与语义相似的人物跟踪和人体关键点检测任务的效果。我们进行了更强大的对抗性攻击的广泛实验，这些攻击可以在物理上实现，证明了我们方法的实用价值。我们用模拟和真实世界的数据集进行的经验研究表明，与通常只进行单一人跟踪任务的训练相比，使用MTL进行训练始终使攻击SiamRPN追踪器变得更加困难。



## **50. FrequencyLowCut Pooling -- Plug & Play against Catastrophic Overfitting**

FredencyLowCut池--针对灾难性过拟合的即插即用 cs.CV

**SubmitDate**: 2022-04-01    [paper-pdf](http://arxiv.org/pdf/2204.00491v1)

**Authors**: Julia Grabinski, Steffen Jung, Janis Keuper, Margret Keuper

**Abstracts**: Over the last years, Convolutional Neural Networks (CNNs) have been the dominating neural architecture in a wide range of computer vision tasks. From an image and signal processing point of view, this success might be a bit surprising as the inherent spatial pyramid design of most CNNs is apparently violating basic signal processing laws, i.e. Sampling Theorem in their down-sampling operations. However, since poor sampling appeared not to affect model accuracy, this issue has been broadly neglected until model robustness started to receive more attention. Recent work [17] in the context of adversarial attacks and distribution shifts, showed after all, that there is a strong correlation between the vulnerability of CNNs and aliasing artifacts induced by poor down-sampling operations. This paper builds on these findings and introduces an aliasing free down-sampling operation which can easily be plugged into any CNN architecture: FrequencyLowCut pooling. Our experiments show, that in combination with simple and fast FGSM adversarial training, our hyper-parameter free operator significantly improves model robustness and avoids catastrophic overfitting.

摘要: 在过去的几年里，卷积神经网络(CNN)已经成为在广泛的计算机视觉任务中占主导地位的神经结构。从图像和信号处理的角度来看，这一成功可能有点令人惊讶，因为大多数CNN固有的空间金字塔设计显然违反了基本的信号处理定律，即下采样操作中的采样定理。然而，由于较差的采样似乎不会影响模型的精度，所以这个问题一直被广泛忽视，直到模型的稳健性开始受到更多的关注。最近的工作[17]在对抗性攻击和分布转移的背景下，毕竟表明在CNN的脆弱性和糟糕的下采样操作引起的混叠伪像之间存在很强的相关性。本文以这些发现为基础，介绍了一种无混叠的下采样操作，该操作可以很容易地插入到任何CNN架构中：FrequencyLowCut池。我们的实验表明，结合简单快速的FGSM对抗性训练，我们的超参数自由算子显著地提高了模型的稳健性，并避免了灾难性的过拟合。



