# Latest Adversarial Attack Papers
**update at 2022-07-26 06:31:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Practical Privacy Attacks on Vertical Federated Learning**

针对垂直联合学习的实用隐私攻击 cs.CR

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2011.09290v3)

**Authors**: Haiqin Weng, Juntao Zhang, Xingjun Ma, Feng Xue, Tao Wei, Shouling Ji, Zhiyuan Zong

**Abstracts**: Federated learning (FL) is a privacy-preserving learning paradigm that allows multiple parities to jointly train a powerful machine learning model without sharing their private data. According to the form of collaboration, FL can be further divided into horizontal federated learning (HFL) and vertical federated learning (VFL). In HFL, participants share the same feature space and collaborate on data samples, while in VFL, participants share the same sample IDs and collaborate on features. VFL has a broader scope of applications and is arguably more suitable for joint model training between large enterprises.   In this paper, we focus on VFL and investigate potential privacy leakage in real-world VFL frameworks. We design and implement two practical privacy attacks: reverse multiplication attack for the logistic regression VFL protocol; and reverse sum attack for the XGBoost VFL protocol. We empirically show that the two attacks are (1) effective - the adversary can successfully steal the private training data, even when the intermediate outputs are encrypted to protect data privacy; (2) evasive - the attacks do not deviate from the protocol specification nor deteriorate the accuracy of the target model; and (3) easy - the adversary needs little prior knowledge about the data distribution of the target participant. We also show the leaked information is as effective as the raw training data in training an alternative classifier. We further discuss potential countermeasures and their challenges, which we hope can lead to several promising research directions.

摘要: 联合学习(FL)是一种隐私保护的学习范式，允许多个奇偶校验在不共享其私人数据的情况下联合训练一个强大的机器学习模型。根据协作形式的不同，外语学习又可分为水平联合学习和垂直联合学习。在HFL中，参与者共享相同的特征空间并就数据样本进行协作，而在VFL中，参与者共享相同的样本ID并就特征进行协作。VFL的应用范围更广，可以说更适合大型企业之间的联合模式培训。在本文中，我们聚焦于虚拟现实语言，并调查现实世界虚拟现实语言框架中潜在的隐私泄漏。设计并实现了两种实用的隐私攻击：针对Logistic回归VFL协议的反向乘法攻击和针对XGBoost VFL协议的反向求和攻击。我们的经验表明，这两种攻击是有效的--攻击者可以成功窃取私人训练数据，即使中间输出被加密以保护数据隐私；(2)规避-攻击不偏离协议规范，也不会恶化目标模型的准确性；以及(3)易用性--攻击者几乎不需要关于目标参与者的数据分布的先验知识。我们还表明，泄漏的信息在训练另一种分类器时与原始训练数据一样有效。我们进一步讨论了潜在的对策及其挑战，我们希望这可以引导出几个有前途的研究方向。



## **2. On Higher Adversarial Susceptibility of Contrastive Self-Supervised Learning**

关于对比性自我监督学习的高对抗敏感性 cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.10862v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Ajmal Mian, Mubarak Shah

**Abstracts**: Contrastive self-supervised learning (CSL) has managed to match or surpass the performance of supervised learning in image and video classification. However, it is still largely unknown if the nature of the representation induced by the two learning paradigms is similar. We investigate this under the lens of adversarial robustness. Our analytical treatment of the problem reveals intrinsic higher sensitivity of CSL over supervised learning. It identifies the uniform distribution of data representation over a unit hypersphere in the CSL representation space as the key contributor to this phenomenon. We establish that this increases model sensitivity to input perturbations in the presence of false negatives in the training data. Our finding is supported by extensive experiments for image and video classification using adversarial perturbations and other input corruptions. Building on the insights, we devise strategies that are simple, yet effective in improving model robustness with CSL training. We demonstrate up to 68% reduction in the performance gap between adversarially attacked CSL and its supervised counterpart. Finally, we contribute to robust CSL paradigm by incorporating our findings in adversarial self-supervised learning. We demonstrate an average gain of about 5% over two different state-of-the-art methods in this domain.

摘要: 对比自监督学习(CSL)在图像和视频分类中的性能已经达到或超过了监督学习。然而，这两种学习范式诱导的表征的性质是否相似仍在很大程度上是未知的。我们在对抗稳健性的视角下对此进行了研究。我们对问题的分析处理揭示了CSL对监督学习的内在更高的敏感性。它认为CSL表示空间中单位超球面上数据表示的均匀分布是造成这一现象的关键因素。我们证明，在训练数据中存在假阴性的情况下，这增加了模型对输入扰动的敏感性。我们的发现得到了使用对抗性扰动和其他输入损坏的图像和视频分类的广泛实验的支持。在这些见解的基础上，我们制定了简单但有效的策略，通过CSL培训提高模型的健壮性。我们展示了被对手攻击的CSL与其受监督的对手之间的性能差距降低了高达68%。最后，我们通过将我们的发现纳入对抗性自我监督学习中，为稳健的CSL范式做出了贡献。在这一领域，我们通过两种不同的最先进的方法展示了大约5%的平均增益。



## **3. Adversarially-Aware Robust Object Detector**

对抗性感知的鲁棒目标检测器 cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.06202v3)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.

摘要: 随着深度神经网络的出现，目标检测作为一项基本的计算机视觉任务已经取得了显著的进展。然而，很少有研究探讨对象检测器在各种真实场景中的实际应用中抵抗对手攻击的对抗性健壮性。检测器受到了不可察觉的扰动的极大挑战，在干净图像上的性能急剧下降，在对抗性图像上的性能极差。在这项工作中，我们经验性地探索了目标检测中对抗鲁棒性的模型训练，这在很大程度上归因于学习干净图像和对抗图像之间的冲突。为了缓解这一问题，我们提出了一种基于对抗性感知卷积的稳健检测器(RobustDet)，用于在干净图像和对抗性图像上进行模型学习。RobustDet还采用了对抗性图像鉴别器(AID)和重建一致特征(CFR)，以确保可靠的健壮性。在PASCAL、VOC和MS-COCO上的大量实验表明，该模型在保持对干净图像的检测能力的同时，有效地解开了梯度的纠缠，显著提高了检测的鲁棒性。



## **4. Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks**

通过层次化生成网络提高目标对抗性实例的可转移性 cs.LG

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2107.01809v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Transfer-based adversarial attacks can evaluate model robustness in the black-box setting. Several methods have demonstrated impressive untargeted transferability, however, it is still challenging to efficiently produce targeted transferability. To this end, we develop a simple yet effective framework to craft targeted transfer-based adversarial examples, applying a hierarchical generative network. In particular, we contribute to amortized designs that well adapt to multi-class targeted attacks. Extensive experiments on ImageNet show that our method improves the success rates of targeted black-box attacks by a significant margin over the existing methods -- it reaches an average success rate of 29.1\% against six diverse models based only on one substitute white-box model, which significantly outperforms the state-of-the-art gradient-based attack methods. Moreover, the proposed method is also more efficient beyond an order of magnitude than gradient-based methods.

摘要: 基于传输的对抗性攻击可以在黑盒环境下评估模型的稳健性。几种方法已经证明了令人印象深刻的非定向可转移性，然而，有效地产生定向可转移性仍然具有挑战性。为此，我们开发了一个简单而有效的框架，应用分层生成网络来制作有针对性的基于迁移的对抗性例子。特别是，我们为能够很好地适应多类别目标攻击的分期设计做出了贡献。在ImageNet上的大量实验表明，与现有方法相比，该方法显著提高了目标黑盒攻击的成功率--仅基于一个替代白盒模型，对6种不同模型的平均成功率达到29.1，显著优于最先进的基于梯度的攻击方法。此外，该方法的效率也比基于梯度的方法高出一个数量级。



## **5. Synthetic Dataset Generation for Adversarial Machine Learning Research**

用于对抗性机器学习研究的合成数据集生成 cs.CV

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10719v1)

**Authors**: Xiruo Liu, Shibani Singh, Cory Cornelius, Colin Busho, Mike Tan, Anindya Paul, Jason Martin

**Abstracts**: Existing adversarial example research focuses on digitally inserted perturbations on top of existing natural image datasets. This construction of adversarial examples is not realistic because it may be difficult, or even impossible, for an attacker to deploy such an attack in the real-world due to sensing and environmental effects. To better understand adversarial examples against cyber-physical systems, we propose approximating the real-world through simulation. In this paper we describe our synthetic dataset generation tool that enables scalable collection of such a synthetic dataset with realistic adversarial examples. We use the CARLA simulator to collect such a dataset and demonstrate simulated attacks that undergo the same environmental transforms and processing as real-world images. Our tools have been used to collect datasets to help evaluate the efficacy of adversarial examples, and can be found at https://github.com/carla-simulator/carla/pull/4992.

摘要: 现有的对抗性实例研究集中在现有自然图像数据集上的数字插入扰动。这种对抗性示例的构建是不现实的，因为由于传感和环境影响，攻击者在现实世界中部署这样的攻击可能很困难，甚至不可能。为了更好地理解针对网络物理系统的敌意例子，我们建议通过模拟来近似真实世界。在这篇文章中，我们描述了我们的合成数据集生成工具，它能够通过现实的对抗性例子来实现对这样的合成数据集的可伸缩收集。我们使用CALA模拟器来收集这样的数据集，并演示模拟攻击，这些攻击经历了与真实世界图像相同的环境转换和处理。我们的工具已被用于收集数据集，以帮助评估对抗性例子的效果，可在https://github.com/carla-simulator/carla/pull/4992.上找到



## **6. Careful What You Wish For: on the Extraction of Adversarially Trained Models**

小心你想要的：关于敌对训练模型的提取 cs.LG

To be published in the proceedings of the 19th Annual International  Conference on Privacy, Security & Trust (PST 2022). The conference  proceedings will be included in IEEE Xplore as in previous editions of the  conference

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10561v1)

**Authors**: Kacem Khaled, Gabriela Nicolescu, Felipe Gohring de Magalhães

**Abstracts**: Recent attacks on Machine Learning (ML) models such as evasion attacks with adversarial examples and models stealing through extraction attacks pose several security and privacy threats. Prior work proposes to use adversarial training to secure models from adversarial examples that can evade the classification of a model and deteriorate its performance. However, this protection technique affects the model's decision boundary and its prediction probabilities, hence it might raise model privacy risks. In fact, a malicious user using only a query access to the prediction output of a model can extract it and obtain a high-accuracy and high-fidelity surrogate model. To have a greater extraction, these attacks leverage the prediction probabilities of the victim model. Indeed, all previous work on extraction attacks do not take into consideration the changes in the training process for security purposes. In this paper, we propose a framework to assess extraction attacks on adversarially trained models with vision datasets. To the best of our knowledge, our work is the first to perform such evaluation. Through an extensive empirical study, we demonstrate that adversarially trained models are more vulnerable to extraction attacks than models obtained under natural training circumstances. They can achieve up to $\times1.2$ higher accuracy and agreement with a fraction lower than $\times0.75$ of the queries. We additionally find that the adversarial robustness capability is transferable through extraction attacks, i.e., extracted Deep Neural Networks (DNNs) from robust models show an enhanced accuracy to adversarial examples compared to extracted DNNs from naturally trained (i.e. standard) models.

摘要: 最近对机器学习(ML)模型的攻击，如利用对抗性示例的逃避攻击和通过提取攻击窃取模型，构成了几种安全和隐私威胁。以前的工作建议使用对抗性训练来从对抗性示例中保护模型，这些示例可能会逃避模型的分类并降低其性能。然而，这种保护技术影响了模型的决策边界及其预测概率，因此可能会增加模型的隐私风险。事实上，恶意用户只使用对模型的预测输出的查询访问就可以提取它，并获得高精度和高保真的代理模型。为了进行更大的提取，这些攻击利用了受害者模型的预测概率。事实上，以前关于提取攻击的所有工作都没有考虑到出于安全目的在训练过程中的变化。在本文中，我们提出了一种评估提取攻击的框架，该框架使用视觉数据集来评估对反向训练模型的提取攻击。据我们所知，我们的工作是第一次进行这样的评估。通过大量的实证研究，我们证明了逆向训练的模型比自然训练环境下的模型更容易受到抽取攻击。它们可以实现高达$\x 1.2$的准确率和与低于$\x 0.75$的小部分查询的一致性。此外，我们还发现，对抗的稳健性能力可以通过抽取攻击来传递，即从健壮模型中提取的深度神经网络(DNN)与从自然训练(即标准)模型中提取的DNN相比，对对抗性实例的准确率更高。



## **7. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

三角攻击：一种查询高效的基于决策的对抗性攻击 cs.CV

Accepted by ECCV 2022, code is available at  https://github.com/xiaosen-wang/TA

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2112.06569v3)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples can naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on ImageNet dataset show that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further validate the applicability of TA on real-world API, i.e., Tencent Cloud API.

摘要: 基于决策的攻击将目标模型视为黑匣子，只访问硬预测标签，对现实世界的应用构成了严重威胁。最近已经做出了很大的努力来减少查询的数量；然而，现有的基于决策的攻击仍然需要数千个查询才能生成高质量的对抗性例子。在这项工作中，我们发现一个良性样本、当前和下一个对抗性样本可以自然地在子空间中为任何迭代攻击构造一个三角形。基于正弦定律，提出了一种新的三角形攻击算法(TA)，该算法利用任意三角形中长边总是与较大角相对的几何信息来优化扰动。然而，直接将这些信息应用于输入图像是无效的，因为它不能在高维空间中彻底探索输入样本的邻域。为了解决这个问题，由于这种几何性质的普遍性，TA优化了低频空间中的扰动，以实现有效的降维。在ImageNet数据集上的广泛评估表明，与现有的基于决策的攻击相比，TA在1000个查询中实现了更高的攻击成功率，并且在各种扰动预算下需要更少的查询来达到相同的攻击成功率。在如此高的效率下，我们进一步验证了TA在现实世界的API上的适用性，即腾讯云API。



## **8. Knowledge-enhanced Black-box Attacks for Recommendations**

用于推荐的知识增强型黑盒攻击 cs.LG

Accepted in the KDD'22

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10307v1)

**Authors**: Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, Yihua Huang

**Abstracts**: Recent studies have shown that deep neural networks-based recommender systems are vulnerable to adversarial attacks, where attackers can inject carefully crafted fake user profiles (i.e., a set of items that fake users have interacted with) into a target recommender system to achieve malicious purposes, such as promote or demote a set of target items. Due to the security and privacy concerns, it is more practical to perform adversarial attacks under the black-box setting, where the architecture/parameters and training data of target systems cannot be easily accessed by attackers. However, generating high-quality fake user profiles under black-box setting is rather challenging with limited resources to target systems. To address this challenge, in this work, we introduce a novel strategy by leveraging items' attribute information (i.e., items' knowledge graph), which can be publicly accessible and provide rich auxiliary knowledge to enhance the generation of fake user profiles. More specifically, we propose a knowledge graph-enhanced black-box attacking framework (KGAttack) to effectively learn attacking policies through deep reinforcement learning techniques, in which knowledge graph is seamlessly integrated into hierarchical policy networks to generate fake user profiles for performing adversarial black-box attacks. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of the proposed attacking framework under the black-box setting.

摘要: 最近的研究表明，基于深度神经网络的推荐系统容易受到敌意攻击，攻击者可以向目标推荐系统注入精心制作的虚假用户配置文件(即，虚假用户与之交互的一组项目)，以达到恶意目的，如升级或降级一组目标项目。由于安全和隐私方面的考虑，在目标系统的体系结构/参数和训练数据不容易被攻击者访问的黑盒环境下执行对抗性攻击更实用。然而，在目标系统资源有限的情况下，在黑盒环境下生成高质量的虚假用户配置文件是相当具有挑战性的。为了应对这一挑战，在本工作中，我们引入了一种新的策略，利用项目的属性信息(即项目的知识图)，这些信息可以公开访问，并提供丰富的辅助知识来增强虚假用户配置文件的生成。更具体地说，我们提出了一种知识图增强的黑盒攻击框架(KGAttack)，通过深度强化学习技术有效地学习攻击策略，该框架将知识图无缝地集成到分层策略网络中，生成用于执行对抗性黑盒攻击的虚假用户配置文件。在各种真实数据集上的综合实验证明了该攻击框架在黑盒环境下的有效性。



## **9. Image Generation Network for Covert Transmission in Online Social Network**

在线社交网络中用于隐蔽传输的图像生成网络 cs.CV

ACMMM2022 Poster

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10292v1)

**Authors**: Zhengxin You, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Online social networks have stimulated communications over the Internet more than ever, making it possible for secret message transmission over such noisy channels. In this paper, we propose a Coverless Image Steganography Network, called CIS-Net, that synthesizes a high-quality image directly conditioned on the secret message to transfer. CIS-Net is composed of four modules, namely, the Generation, Adversarial, Extraction, and Noise Module. The receiver can extract the hidden message without any loss even the images have been distorted by JPEG compression attacks. To disguise the behaviour of steganography, we collected images in the context of profile photos and stickers and train our network accordingly. As such, the generated images are more inclined to escape from malicious detection and attack. The distinctions from previous image steganography methods are majorly the robustness and losslessness against diverse attacks. Experiments over diverse public datasets have manifested the superior ability of anti-steganalysis.

摘要: 在线社交网络比以往任何时候都更多地刺激了互联网上的交流，使得在这种嘈杂的渠道上传输秘密信息成为可能。在本文中，我们提出了一种无覆盖的图像隐写网络，称为CIS-Net，它直接根据秘密信息合成高质量的图像进行传输。该网络由四个模块组成，即生成模块、对抗性模块、抽取模块和噪声模块。即使图像被JPEG压缩攻击篡改了，接收者也可以无损地提取隐藏信息。为了掩盖隐写术的行为，我们收集了头像照片和贴纸背景下的图像，并对我们的网络进行了相应的培训。因此，生成的图像更容易逃脱恶意检测和攻击。与以往的图像隐写方法不同的是，该方法对各种攻击具有较强的稳健性和无损性能。在不同的公开数据集上的实验表明，该算法具有较好的抗隐写分析能力。



## **10. Switching One-Versus-the-Rest Loss to Increase the Margin of Logits for Adversarial Robustness**

切换一对一损失以增加对战健壮性的Logit裕度 cs.LG

20 pages, 16 figures

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10283v1)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Yasutoshi Ida

**Abstracts**: Defending deep neural networks against adversarial examples is a key challenge for AI safety. To improve the robustness effectively, recent methods focus on important data points near the decision boundary in adversarial training. However, these methods are vulnerable to Auto-Attack, which is an ensemble of parameter-free attacks for reliable evaluation. In this paper, we experimentally investigate the causes of their vulnerability and find that existing methods reduce margins between logits for the true label and the other labels while keeping their gradient norms non-small values. Reduced margins and non-small gradient norms cause their vulnerability since the largest logit can be easily flipped by the perturbation. Our experiments also show that the histogram of the logit margins has two peaks, i.e., small and large logit margins. From the observations, we propose switching one-versus-the-rest loss (SOVR), which uses one-versus-the-rest loss when data have small logit margins so that it increases the margins. We find that SOVR increases logit margins more than existing methods while keeping gradient norms small and outperforms them in terms of the robustness against Auto-Attack.

摘要: 防御深层神经网络以抵御敌意示例是人工智能安全的关键挑战。在对抗性训练中，为了有效地提高鲁棒性，目前的方法主要集中在决策边界附近的重要数据点。然而，这些方法容易受到自动攻击的攻击，自动攻击是用于可靠评估的无参数攻击的集合。在实验中，我们调查了它们易受攻击的原因，发现现有的方法在保持它们的梯度范数非小值的同时，减少了真实标签和其他标签的对数之间的差值。边际减小和非小梯度范数导致了它们的脆弱性，因为最大的logit很容易被扰动翻转。我们的实验还表明，Logit边缘的直方图有两个峰值，即小的和大的Logit边缘。根据观察，我们建议转换一对其余损失(SOVR)，即当数据具有较小的Logit边际时使用一对其余损失，从而增加边际。我们发现，SOVR在保持小的梯度范数的同时，比现有的方法更能提高Logit裕度，并且在抵抗自动攻击方面优于现有的方法。



## **11. FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data**

焦点：异类数据联合学习中基于代理感知的公平性 cs.LG

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10265v1)

**Authors**: Wenda Chu, Chulin Xie, Boxin Wang, Linyi Li, Lang Yin, Han Zhao, Bo Li

**Abstracts**: Federated learning (FL) provides an effective paradigm to train machine learning models over distributed data with privacy protection. However, recent studies show that FL is subject to various security, privacy, and fairness threats due to the potentially malicious and heterogeneous local agents. For instance, it is vulnerable to local adversarial agents who only contribute low-quality data, with the goal of harming the performance of those with high-quality data. This kind of attack hence breaks existing definitions of fairness in FL that mainly focus on a certain notion of performance parity. In this work, we aim to address this limitation and propose a formal definition of fairness via agent-awareness for FL (FAA), which takes the heterogeneous data contributions of local agents into account. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear models and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness measured by FAA compared with standard FedAvg protocol under both linear models and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness based on FAA while maintaining similar or even higher prediction accuracy compared with FedAvg.

摘要: 联合学习(FL)提供了一种有效的范例来训练具有隐私保护的分布式数据上的机器学习模型。然而，最近的研究表明，由于潜在的恶意和异构性的本地代理，FL受到各种安全、隐私和公平的威胁。例如，它很容易受到当地对手代理人的攻击，这些代理人只提供低质量的数据，目的是损害那些拥有高质量数据的人的表现。因此，这种攻击打破了外语教学中对公平的现有定义，这些定义主要集中在某个绩效平等的概念上。在这项工作中，我们旨在解决这一局限性，并提出了一种基于代理感知的公平的形式化定义(FAA)，该定义考虑了本地代理的异质数据贡献。此外，我们还提出了一种基于主体聚类的公平FL训练算法(FOCUS)来实现FAA。从理论上证明了线性模型和具有有界光滑性的一般凸损失函数在较温和的条件下焦点的收敛和最优性。证明了无论是在线性模型下还是在一般的凸损失函数下，Focus协议都比标准FedAvg协议获得了更高的公平性。实验结果表明，基于FAA的Focus算法在保持与FedAvg相似甚至更高的预测精度的同时，获得了显著更高的公平性。



## **12. Illusionary Attacks on Sequential Decision Makers and Countermeasures**

对序贯决策者的幻觉攻击及其对策 cs.AI

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.10170v1)

**Authors**: Tim Franzmeyer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstracts**: Autonomous intelligent agents deployed to the real-world need to be robust against adversarial attacks on sensory inputs. Existing work in reinforcement learning focuses on minimum-norm perturbation attacks, which were originally introduced to mimic a notion of perceptual invariance in computer vision. In this paper, we note that such minimum-norm perturbation attacks can be trivially detected by victim agents, as these result in observation sequences that are not consistent with the victim agent's actions. Furthermore, many real-world agents, such as physical robots, commonly operate under human supervisors, which are not susceptible to such perturbation attacks. As a result, we propose to instead focus on illusionary attacks, a novel form of attack that is consistent with the world model of the victim agent. We provide a formal definition of this novel attack framework, explore its characteristics under a variety of conditions, and conclude that agents must seek realism feedback to be robust to illusionary attacks.

摘要: 部署在现实世界中的自主智能代理需要对感官输入的敌意攻击具有健壮性。强化学习的现有工作集中在最小范数扰动攻击上，最初引入最小范数扰动攻击是为了模仿计算机视觉中的感知不变性的概念。在本文中，我们注意到这种最小范数扰动攻击可以被受害者代理检测到，因为这些攻击导致的观察序列与受害者代理的行为不一致。此外，许多真实世界的代理，如物理机器人，通常在人类监督下操作，而人类监督不容易受到此类扰动攻击。因此，我们建议转而专注于幻觉攻击，这是一种与受害者代理的世界模型一致的新型攻击形式。我们给出了这种新的攻击框架的形式化定义，探讨了它在各种条件下的特征，并得出结论：代理必须寻求现实主义反馈才能对虚幻攻击具有健壮性。



## **13. PFMC: a parallel symbolic model checker for security protocol verification**

PFMC：一种用于安全协议验证的并行符号模型检查器 cs.LO

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09895v1)

**Authors**: Alex James, Alwen Tiu, Nisansala Yatapanage

**Abstracts**: We present an investigation into the design and implementation of a parallel model checker for security protocol verification that is based on a symbolic model of the adversary, where instantiations of concrete terms and messages are avoided until needed to resolve a particular assertion. We propose to build on this naturally lazy approach to parallelise this symbolic state exploration and evaluation. We utilise the concept of strategies in Haskell, which abstracts away from the low-level details of thread management and modularly adds parallel evaluation strategies (encapsulated as a monad in Haskell). We build on an existing symbolic model checker, OFMC, which is already implemented in Haskell. We show that there is a very significant speed up of around 3-5 times improvement when moving from the original single-threaded implementation of OFMC to our multi-threaded version, for both the Dolev-Yao attacker model and more general algebraic attacker models. We identify several issues in parallelising the model checker: among others, controlling growth of memory consumption, balancing lazy vs strict evaluation, and achieving an optimal granularity of parallelism.

摘要: 本文研究了一种用于安全协议验证的并行模型检查器的设计和实现，该模型检查器基于敌手的符号模型，其中避免实例化具体的术语和消息，直到需要解决特定断言。我们建议基于这种天生懒惰的方法来并行化这种象征性的状态探索和评估。我们利用了Haskell中的策略概念，它从线程管理的底层细节中抽象出来，并模块化地添加了并行计算策略(在Haskell中封装为Monad)。我们构建在现有的符号模型检查器OFMC上，该检查器已经在Haskell中实现。我们表明，从最初的单线程OFMC实现转移到我们的多线程版本时，无论是对于Dolev-姚攻击者模型还是更一般的代数攻击者模型，都有非常显著的速度提升约3-5倍。我们确定了模型检查器并行化中的几个问题：控制内存消耗的增长，平衡懒惰和严格计算，以及实现最优的并行粒度。



## **14. Adaptive Image Transformations for Transfer-based Adversarial Attack**

基于传输的对抗性攻击中的自适应图像变换 cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2111.13844v3)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.

摘要: 对抗性攻击为研究深度学习模型的稳健性提供了一种很好的方法。一类基于转移的黑盒攻击方法利用多幅图像变换操作来提高对抗性样本的可转移性，这种方法是有效的，但没有考虑到输入图像的具体特征。在这项工作中，我们提出了一种新的体系结构，称为自适应图像变换学习器(AITL)，它将不同的图像变换操作整合到一个统一的框架中，以进一步提高对抗性例子的可转移性。与现有工作中使用的固定组合变换不同，我们精心设计的变换学习器自适应地选择特定于输入图像的最有效的图像变换组合。在ImageNet上的大量实验表明，该方法在正常训练模型和防御模型上的攻击成功率在各种设置下都有显著提高。



## **15. On the Robustness of Quality Measures for GANs**

论GAN质量度量的稳健性 cs.LG

Accepted at the European Conference in Computer Vision (ECCV 2022)

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2201.13019v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Anna Frühstück, Philip H. S. Torr, Peter Wonka, Bernard Ghanem

**Abstracts**: This work evaluates the robustness of quality measures of generative models such as Inception Score (IS) and Fr\'echet Inception Distance (FID). Analogous to the vulnerability of deep models against a variety of adversarial attacks, we show that such metrics can also be manipulated by additive pixel perturbations. Our experiments indicate that one can generate a distribution of images with very high scores but low perceptual quality. Conversely, one can optimize for small imperceptible perturbations that, when added to real world images, deteriorate their scores. We further extend our evaluation to generative models themselves, including the state of the art network StyleGANv2. We show the vulnerability of both the generative model and the FID against additive perturbations in the latent space. Finally, we show that the FID can be robustified by simply replacing the standard Inception with a robust Inception. We validate the effectiveness of the robustified metric through extensive experiments, showing it is more robust against manipulation.

摘要: 该工作评估了产生式模型的质量度量的稳健性，如初始得分(IS)和Fr回声初始距离(FID)。类似于深度模型对各种对抗性攻击的脆弱性，我们证明了这样的度量也可以被加性像素扰动所操纵。我们的实验表明，我们可以生成得分很高但感知质量较低的图像分布。相反，人们可以针对微小的不可察觉的扰动进行优化，当这些扰动添加到现实世界的图像中时，会降低他们的得分。我们进一步将我们的评估扩展到生成性模型本身，包括最先进的网络StyleGANv2。我们证明了生成模型和FID在潜在空间中对加性扰动的脆弱性。最后，我们展示了FID可以通过简单地用健壮的初始替换标准的初始来增强。我们通过大量的实验验证了鲁棒性度量的有效性，表明它对操纵具有更强的健壮性。



## **16. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

偏距离相关在深度学习中的广泛应用 cs.CV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09684v1)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstracts**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **17. Detecting Textual Adversarial Examples through Randomized Substitution and Vote**

基于随机化替换和投票的文本对抗性实例检测 cs.CL

Accepted by UAI 2022, code is avaliable at  https://github.com/JHL-HUST/RSV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2109.05698v2)

**Authors**: Xiaosen Wang, Yifeng Xiong, Kun He

**Abstracts**: A line of work has shown that natural text processing models are vulnerable to adversarial examples. Correspondingly, various defense methods are proposed to mitigate the threat of textual adversarial examples, eg, adversarial training, input transformations, detection, etc. In this work, we treat the optimization process for synonym substitution based textual adversarial attacks as a specific sequence of word replacement, in which each word mutually influences other words. We identify that we could destroy such mutual interaction and eliminate the adversarial perturbation by randomly substituting a word with its synonyms. Based on this observation, we propose a novel textual adversarial example detection method, termed Randomized Substitution and Vote (RS&V), which votes the prediction label by accumulating the logits of k samples generated by randomly substituting the words in the input text with synonyms. The proposed RS&V is generally applicable to any existing neural networks without modification on the architecture or extra training, and it is orthogonal to prior work on making the classification network itself more robust. Empirical evaluations on three benchmark datasets demonstrate that our RS&V could detect the textual adversarial examples more successfully than the existing detection methods while maintaining the high classification accuracy on benign samples.

摘要: 一系列研究表明，自然文本处理模型很容易受到敌意例子的影响。相应地，人们提出了各种防御方法来缓解文本对抗性实例的威胁，如对抗性训练、输入转换、检测等。在本文中，我们将基于同义词替换的文本对抗性攻击的优化过程视为一个特定的单词替换序列，其中每个单词都会影响其他单词。我们发现，通过随机地用一个词的同义词替换一个词，我们可以破坏这种相互作用，并消除对抗性扰动。基于这一观察结果，我们提出了一种新的文本对抗性实例检测方法，称为随机替换和投票(RS&V)，该方法通过累加k个样本的逻辑来投票预测标签，所述k个样本是通过随机地将输入文本中的单词替换为同义词而产生的。所提出的RS&V算法一般适用于任何现有的神经网络，而不需要修改结构或进行额外的训练，并且它与先前的使分类网络本身更健壮的工作是正交的。在三个基准数据集上的实验结果表明，与现有的检测方法相比，本文的RS&V方法能够更成功地检测出文本中的敌意实例，同时保持了对良性样本的高分类准确率。



## **18. Diversified Adversarial Attacks based on Conjugate Gradient Method**

基于共轭梯度法的多样化对抗性攻击 cs.LG

Proceedings of the 39th International Conference on Machine Learning  (ICML 2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2206.09628v2)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.

摘要: 深度学习模型容易受到对抗性实例的影响，而用于生成此类实例的对抗性攻击已经引起了相当大的研究兴趣。虽然现有的基于最陡下降的方法已经取得了很高的攻击成功率，但条件恶劣的问题有时会降低它们的性能。针对这一局限性，我们利用对这类问题有效的共轭梯度(CG)方法，并在CG方法的启发下提出了一种新的攻击算法，称为自动共轭梯度(ACG)攻击。在最新的稳健模型上进行的大规模评估实验结果表明，对于大多数模型，ACG能够以更少的迭代发现更多的对抗性实例，而不是现有的SOTA算法Auto-PGD(APGD)。我们研究了ACG和APGD在多样化和集约化方面的搜索性能差异，并定义了一个称为多样性指数(DI)的度量来量化多样性程度。从该指标的多样性分析可以看出，该方法搜索的多样性显著提高了其攻击成功率。



## **19. Towards Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

走向稳健的多变量时间序列预测：对抗性攻击和防御机制 cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09572v1)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstracts**: As deep learning models have gradually become the main workhorse of time series forecasting, the potential vulnerability under adversarial attacks to forecasting and decision system accordingly has emerged as a main issue in recent years. Albeit such behaviors and defense mechanisms started to be investigated for the univariate time series forecasting, there are still few studies regarding the multivariate forecasting which is often preferred due to its capacity to encode correlations between different time series. In this work, we study and design adversarial attack on multivariate probabilistic forecasting models, taking into consideration attack budget constraints and the correlation architecture between multiple time series. Specifically, we investigate a sparse indirect attack that hurts the prediction of an item (time series) by only attacking the history of a small number of other items to save attacking cost. In order to combat these attacks, we also develop two defense strategies. First, we adopt randomized smoothing to multivariate time series scenario and verify its effectiveness via empirical experiments. Second, we leverage a sparse attacker to enable end-to-end adversarial training that delivers robust probabilistic forecasters. Extensive experiments on real dataset confirm that our attack schemes are powerful and our defend algorithms are more effective compared with other baseline defense mechanisms.

摘要: 随着深度学习模型逐渐成为时间序列预测的主要工具，预测和决策系统在敌意攻击下的潜在脆弱性也成为近年来的主要问题。虽然单变量时间序列预测的这种行为和防御机制已经开始被研究，但关于多变量预测的研究仍然很少，因为多变量预测往往因为能够编码不同时间序列之间的相关性而受到青睐。在这项工作中，我们研究和设计了基于多变量概率预测模型的对抗性攻击，考虑了攻击预算约束和多个时间序列之间的关联结构。具体地说，我们调查了一种稀疏的间接攻击，该攻击通过仅攻击少量其他项的历史来节省攻击成本，从而损害了一项(时间序列)的预测。为了对抗这些攻击，我们还制定了两种防御策略。首先，将随机平滑方法应用于多变量时间序列情景，并通过实证实验验证其有效性。其次，我们利用稀疏攻击者实现端到端的对抗性训练，从而提供强大的概率预测者。在真实数据集上的大量实验证实了我们的攻击方案是强大的，与其他基线防御机制相比，我们的防御算法更有效。



## **20. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

使用校准的工作证明增加模型提取的成本 cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2201.09243v2)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstracts**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.

摘要: 在模型提取攻击中，攻击者可以通过反复查询通过公共API暴露的机器学习模型，并根据获得的预测调整自己的模型，从而窃取该模型。为了防止模型窃取，现有的防御措施侧重于检测恶意查询、截断或扭曲输出，因此必然会在健壮性和模型实用程序之间为合法用户带来折衷。相反，我们建议通过要求用户在阅读模型预测之前完成工作证明来阻碍模型提取。这大大增加了(甚至高达100倍)利用查询访问进行模型提取所需的计算工作量，从而阻止了攻击者。由于我们对完成每个查询的工作证明所需的工作量进行了校准，因此这只会给普通用户带来很小的开销(最高可达2倍)。为了实现这一点，我们的校准应用了来自差异隐私的工具来衡量查询所揭示的信息。我们的方法不需要修改受害者模型，并且可以被机器学习从业者应用，以保护他们公开曝光的模型不会轻易被窃取。



## **21. Assaying Out-Of-Distribution Generalization in Transfer Learning**

迁移学习中的分布外泛化分析 cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09239v1)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstracts**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.

摘要: 由于分布外泛化是一个通常不适定的问题，因此对不同的代理目标(例如，校准、对手健壮性、算法腐败、跨班次不变性)进行了研究，得出了不同的建议。虽然有着相同的理想目标，但这些方法从未在相同的实验条件下对真实数据进行过测试。在本文中，我们对以前的工作进行了统一的审查，强调了我们通过经验解决的信息差异，并就如何衡量模型的稳健性以及如何改进模型提供了建议。为此，我们收集了172个公开可用的数据集对，用于训练和分布外评估准确性、校准误差、对抗性攻击、环境不变性和合成腐败。我们微调了超过31k的网络，这些网络来自9种不同的架构，在多发和少发的情况下。我们的发现证实，分布内和分布外的精度往往会共同增加，但表明它们的关系在很大程度上依赖于数据集，总体上比之前的较小规模的研究假设的更细微和更复杂。



## **22. MUD-PQFed: Towards Malicious User Detection in Privacy-Preserving Quantized Federated Learning**

MUD-PQFed：隐私保护量化联合学习中的恶意用户检测 cs.CR

13 pages,13 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09080v1)

**Authors**: Hua Ma, Qun Li, Yifeng Zheng, Zhi Zhang, Xiaoning Liu, Yansong Gao, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: Federated Learning (FL), a distributed machine learning paradigm, has been adapted to mitigate privacy concerns for customers. Despite their appeal, there are various inference attacks that can exploit shared-plaintext model updates to embed traces of customer private information, leading to serious privacy concerns. To alleviate this privacy issue, cryptographic techniques such as Secure Multi-Party Computation and Homomorphic Encryption have been used for privacy-preserving FL. However, such security issues in privacy-preserving FL are poorly elucidated and underexplored. This work is the first attempt to elucidate the triviality of performing model corruption attacks on privacy-preserving FL based on lightweight secret sharing. We consider scenarios in which model updates are quantized to reduce communication overhead in this case, where an adversary can simply provide local parameters outside the legal range to corrupt the model. We then propose the MUD-PQFed protocol, which can precisely detect malicious clients performing attacks and enforce fair penalties. By removing the contributions of detected malicious clients, the global model utility is preserved to be comparable to the baseline global model without the attack. Extensive experiments validate effectiveness in maintaining baseline accuracy and detecting malicious clients in a fine-grained manner

摘要: 联邦学习(FL)是一种分布式机器学习范式，已被用于缓解客户的隐私担忧。尽管有吸引力，但仍有各种推理攻击可以利用共享明文模型更新来嵌入客户私人信息的痕迹，从而导致严重的隐私问题。为了缓解这一隐私问题，安全多方计算和同态加密等密码技术被用于隐私保护FL。然而，在保护隐私的FL中，这样的安全问题还没有得到很好的阐述和探讨。这项工作是首次尝试阐明基于轻量级秘密共享对隐私保护FL执行模型腐败攻击的琐碎之处。在这种情况下，我们考虑对模型更新进行量化以减少通信开销的场景，其中对手只需提供合法范围之外的本地参数即可破坏模型。然后，我们提出了MUD-PQFed协议，该协议能够准确地检测执行攻击的恶意客户端，并执行公平的惩罚。通过去除检测到的恶意客户端的贡献，全局模型实用程序被保留为与没有攻击的基准全局模型相当。大量实验验证了在保持基线准确性和细粒度检测恶意客户端方面的有效性



## **23. $\ell_\infty$-Robustness and Beyond: Unleashing Efficient Adversarial Training**

$\ell_\inty$-健壮性和超越：释放高效的对抗性训练 cs.LG

Accepted to the 17th European Conference on Computer Vision (ECCV  2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2112.00378v2)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches in training robust models against such attacks. However, it is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration, hampering its effectiveness. Recently, Fast Adversarial Training (FAT) was proposed that can obtain robust models efficiently. However, the reasons behind its success are not fully understood, and more importantly, it can only train robust models for $\ell_\infty$-bounded attacks as it uses FGSM during training. In this paper, by leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a general, more principled approach toward reducing the time complexity of robust training. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training (PAT). Our experimental results indicate that our approach speeds up adversarial training by 2-3 times while experiencing a slight reduction in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练抵抗此类攻击的稳健模型的最有效方法之一。然而，它比普通的神经网络训练要慢得多，因为它需要在每一次迭代中为整个训练数据构造对抗性样本，这阻碍了它的有效性。最近，快速对抗性训练(FAT)被提出，它可以有效地获得稳健的模型。然而，其成功背后的原因还不完全清楚，更重要的是，由于它在训练过程中使用FGSM，所以它只能为$\ell_\$有界攻击训练健壮的模型。在本文中，通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种通用的、更有原则的方法来降低健壮训练的时间复杂性。与现有方法不同，我们的方法可以适应广泛的训练目标，包括行业、$\ell_p$-PGD和感知对手训练(PAT)。我们的实验结果表明，我们的方法将对抗性训练的速度提高了2-3倍，而干净和健壮的准确率略有下降。



## **24. Decorrelative Network Architecture for Robust Electrocardiogram Classification**

用于稳健心电分类的解相关网络结构 cs.LG

12 pages, 6 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09031v1)

**Authors**: Christopher Wiedeman, Ge Wang

**Abstracts**: Artificial intelligence has made great progresses in medical data analysis, but the lack of robustness and interpretability has kept these methods from being widely deployed. In particular, data-driven models are vulnerable to adversarial attacks, which are small, targeted perturbations that dramatically degrade model performance. As a recent example, while deep learning has shown impressive performance in electrocardiogram (ECG) classification, Han et al. crafted realistic perturbations that fooled the network 74% of the time [2020]. Current adversarial defense paradigms are computationally intensive and impractical for many high dimensional problems. Previous research indicates that a network vulnerability is related to the features learned during training. We propose a novel approach based on ensemble decorrelation and Fourier partitioning for training parallel network arms into a decorrelated architecture to learn complementary features, significantly reducing the chance of a perturbation fooling all arms of the deep learning model. We test our approach in ECG classification, demonstrating a much-improved 77.2% chance of at least one correct network arm on the strongest adversarial attack tested, in contrast to a 21.7% chance from a comparable ensemble. Our approach does not require expensive optimization with adversarial samples, and thus can be scaled to large problems. These methods can easily be applied to other tasks for improved network robustness.

摘要: 人工智能在医疗数据分析方面取得了很大进展，但缺乏健壮性和可解释性，阻碍了这些方法的广泛部署。特别是，数据驱动的模型很容易受到敌意攻击，这种攻击是小的、有针对性的干扰，会显著降低模型的性能。作为最近的一个例子，当深度学习在心电分类中表现出令人印象深刻的表现时，han等人。精心制作的现实扰动，在[2020]的时间里愚弄了网络74%。目前的对抗性防御模式计算量大，不适用于许多高维问题。以前的研究表明，网络漏洞与在培训过程中学习到的功能有关。我们提出了一种基于集成去相关和傅立叶划分的新方法，将并行网络臂训练到去相关体系结构中学习互补特征，显著降低了扰动欺骗深度学习模型所有臂的机会。我们在心电分类中测试了我们的方法，表明在测试的最强敌意攻击中，至少有一个正确的网络臂的概率有77.2%，而在类似的集合中，这一概率为21.7%。我们的方法不需要使用对抗性样本进行昂贵的优化，因此可以扩展到大型问题。这些方法可以很容易地应用于其他任务，以提高网络的健壮性。



## **25. Defending Substitution-Based Profile Pollution Attacks on Sequential Recommenders**

针对顺序推荐器防御基于替换的Profile污染攻击 cs.IR

Accepted to RecSys 2022

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.11237v1)

**Authors**: Zhenrui Yue, Huimin Zeng, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstracts**: While sequential recommender systems achieve significant improvements on capturing user dynamics, we argue that sequential recommenders are vulnerable against substitution-based profile pollution attacks. To demonstrate our hypothesis, we propose a substitution-based adversarial attack algorithm, which modifies the input sequence by selecting certain vulnerable elements and substituting them with adversarial items. In both untargeted and targeted attack scenarios, we observe significant performance deterioration using the proposed profile pollution algorithm. Motivated by such observations, we design an efficient adversarial defense method called Dirichlet neighborhood sampling. Specifically, we sample item embeddings from a convex hull constructed by multi-hop neighbors to replace the original items in input sequences. During sampling, a Dirichlet distribution is used to approximate the probability distribution in the neighborhood such that the recommender learns to combat local perturbations. Additionally, we design an adversarial training method tailored for sequential recommender systems. In particular, we represent selected items with one-hot encodings and perform gradient ascent on the encodings to search for the worst case linear combination of item embeddings in training. As such, the embedding function learns robust item representations and the trained recommender is resistant to test-time adversarial examples. Extensive experiments show the effectiveness of both our attack and defense methods, which consistently outperform baselines by a significant margin across model architectures and datasets.

摘要: 虽然序列推荐器系统在捕获用户动态方面取得了显著的改进，但我们认为序列推荐器很容易受到基于替换的配置文件污染攻击。为了证明我们的假设，我们提出了一种基于替换的对抗性攻击算法，该算法通过选择某些易受攻击的元素并将其替换为对抗性项来修改输入序列。在非目标攻击和目标攻击场景中，我们观察到使用所提出的配置文件污染算法的性能显著下降。受此启发，我们设计了一种有效的对抗性防御方法--Dirichlet邻域抽样。具体地说，我们从多跳邻居构造的凸壳中抽取项嵌入来替换输入序列中的原始项。在采样过程中，Dirichlet分布被用来近似邻域内的概率分布，从而使推荐者学习对抗局部扰动。此外，我们还设计了一种针对顺序推荐系统的对抗性训练方法。特别地，我们用一热编码来表示选择的项，并对编码进行梯度上升，以搜索训练中项嵌入的最坏情况的线性组合。因此，嵌入函数学习健壮的项目表示，并且训练的推荐器抵抗测试时间的对抗性示例。广泛的实验表明，我们的攻击和防御方法都是有效的，在模型体系结构和数据集上，它们的表现一直比基线高出很多。



## **26. Multi-step domain adaptation by adversarial attack to $\mathcal{H} Δ\mathcal{H}$-divergence**

敌意攻击对$\Mathcal{H}Δ\Mathcal{H}$-发散的多步域适应 cs.LG

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08948v1)

**Authors**: Arip Asadulaev, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: Adversarial examples are transferable between different models. In our paper, we propose to use this property for multi-step domain adaptation. In unsupervised domain adaptation settings, we demonstrate that replacing the source domain with adversarial examples to $\mathcal{H} \Delta \mathcal{H}$-divergence can improve source classifier accuracy on the target domain. Our method can be connected to most domain adaptation techniques. We conducted a range of experiments and achieved improvement in accuracy on Digits and Office-Home datasets.

摘要: 对抗性的例子可以在不同的模型之间转移。在我们的论文中，我们建议将这一性质用于多步域自适应。在非监督领域自适应设置中，我们证明了用对抗性的例子替换源域到$\mathcal{H}\Delta\mathcal{H}$-分歧可以提高目标域上的源分类器的准确率。我们的方法可以连接到大多数领域自适应技术。我们进行了一系列实验，并在数字和Office-Home数据集上实现了准确率的提高。



## **27. Benchmarking Machine Learning Robustness in Covid-19 Genome Sequence Classification**

新冠肺炎基因组序列分类中机器学习稳健性的基准测试 q-bio.GN

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08898v1)

**Authors**: Sarwan Ali, Bikram Sahoo, Alexander Zelikovskiy, Pin-Yu Chen, Murray Patterson

**Abstracts**: The rapid spread of the COVID-19 pandemic has resulted in an unprecedented amount of sequence data of the SARS-CoV-2 genome -- millions of sequences and counting. This amount of data, while being orders of magnitude beyond the capacity of traditional approaches to understanding the diversity, dynamics, and evolution of viruses is nonetheless a rich resource for machine learning (ML) approaches as alternatives for extracting such important information from these data. It is of hence utmost importance to design a framework for testing and benchmarking the robustness of these ML models.   This paper makes the first effort (to our knowledge) to benchmark the robustness of ML models by simulating biological sequences with errors. In this paper, we introduce several ways to perturb SARS-CoV-2 genome sequences to mimic the error profiles of common sequencing platforms such as Illumina and PacBio. We show from experiments on a wide array of ML models that some simulation-based approaches are more robust (and accurate) than others for specific embedding methods to certain adversarial attacks to the input sequences. Our benchmarking framework may assist researchers in properly assessing different ML models and help them understand the behavior of the SARS-CoV-2 virus or avoid possible future pandemics.

摘要: 新冠肺炎疫情的迅速蔓延导致了空前数量的SARS-CoV-2基因组序列数据--数以百万计的序列和不断增加的数据。这些数据量虽然超出了传统方法理解病毒多样性、动态和进化的能力的数量级，但仍然是机器学习(ML)方法的丰富资源，作为从这些数据中提取如此重要信息的替代方案。因此，设计一个框架来测试和基准这些ML模型的健壮性是至关重要的。本文首次尝试(据我们所知)通过模拟有误差的生物序列来测试ML模型的稳健性。在本文中，我们介绍了几种扰乱SARS-CoV-2基因组序列的方法，以模拟Illumina和PacBio等常见测序平台的错误轮廓。我们在大量的ML模型上的实验表明，对于特定的嵌入方法，一些基于模拟的方法对输入序列的某些对抗性攻击比其他方法更健壮(也更准确)。我们的基准框架可以帮助研究人员正确评估不同的ML模型，并帮助他们了解SARS-CoV-2病毒的行为或避免未来可能的大流行。



## **28. Prior-Guided Adversarial Initialization for Fast Adversarial Training**

用于快速对战训练的先验引导对战初始化 cs.CV

ECCV 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08859v1)

**Authors**: Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstracts**: Fast adversarial training (FAT) effectively improves the efficiency of standard adversarial training (SAT). However, initial FAT encounters catastrophic overfitting, i.e.,the robust accuracy against adversarial attacks suddenly and dramatically decreases. Though several FAT variants spare no effort to prevent overfitting, they sacrifice much calculation cost. In this paper, we explore the difference between the training processes of SAT and FAT and observe that the attack success rate of adversarial examples (AEs) of FAT gets worse gradually in the late training stage, resulting in overfitting. The AEs are generated by the fast gradient sign method (FGSM) with a zero or random initialization. Based on the observation, we propose a prior-guided FGSM initialization method to avoid overfitting after investigating several initialization strategies, improving the quality of the AEs during the whole training process. The initialization is formed by leveraging historically generated AEs without additional calculation cost. We further provide a theoretical analysis for the proposed initialization method. We also propose a simple yet effective regularizer based on the prior-guided initialization,i.e., the currently generated perturbation should not deviate too much from the prior-guided initialization. The regularizer adopts both historical and current adversarial perturbations to guide the model learning. Evaluations on four datasets demonstrate that the proposed method can prevent catastrophic overfitting and outperform state-of-the-art FAT methods. The code is released at https://github.com/jiaxiaojunQAQ/FGSM-PGI.

摘要: 快速对抗训练(FAT)有效地提高了标准对抗训练(SAT)的效率。然而，初始FAT会遭遇灾难性的过拟合，即对敌方攻击的稳健准确率突然急剧下降。尽管几个胖变种不遗余力地防止过度适应，但它们牺牲了大量的计算成本。本文探讨了SAT和FAT训练过程的差异，发现FAT的对抗性例子(AES)的攻击成功率在训练后期逐渐变差，导致训练过度。快速梯度符号法(FGSM)通过零或随机初始化来生成AE。在此基础上，研究了几种初始化策略，提出了一种先验引导的FGSM初始化方法，避免了过拟合度的问题，提高了训练过程中的训练质量。初始化是通过利用历史生成的AE形成的，而不需要额外的计算成本。我们进一步对所提出的初始化方法进行了理论分析。我们还提出了一种基于先验引导初始化的简单而有效的正则化方法，即当前产生的扰动不应偏离先验引导初始化太多。正则化子采用历史和当前的对抗性扰动来指导模型学习。在四个数据集上的评估结果表明，该方法可以防止灾难性的过拟合，并优于现有的FAT方法。该代码在https://github.com/jiaxiaojunQAQ/FGSM-PGI.上发布



## **29. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

对抗性像素恢复作为可转移扰动的借口任务 cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08803v1)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max objective which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to our adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR

摘要: 可转移对抗性攻击从预先训练的代理模型和已知标签空间中优化对手，以愚弄未知的黑盒模型。因此，这些攻击受到有效代理模型可用性的限制。在这项工作中，我们放松了这一假设，提出了对抗性像素复原作为一种自我监督的替代方案，在没有标签和数据样本的情况下，从零开始训练一个有效的代理模型。我们的训练方法基于最小-最大目标，该目标减少了通过对抗性目标的过度拟合，从而优化了更具通用性的代理模型。我们建议的攻击是对我们的对抗性像素恢复的补充，并且独立于任何特定任务的目标，因为它可以以自我监督的方式发起。我们成功地展示了我们的视觉变形方法以及卷积神经网络方法在分类、目标检测和视频分割任务中的对抗性可转移性。我们的代码和预先培训的代孕模型可在以下网址获得：https://github.com/HashmatShadab/APR



## **30. Are Vision Transformers Robust to Patch Perturbations?**

视觉变形器对补丁扰动有健壮性吗？ cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2111.10659v2)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: Recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-based input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of ViT to patch-wise perturbations. Surprisingly, we find that ViTs are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we discover that the attention mechanism greatly affects the robustness of vision transformers. Specifically, the attention module can help improve the robustness of ViT by effectively ignoring natural corrupted patches. However, when ViTs are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake. Based on our analysis, we propose a simple temperature-scaling based method to improve the robustness of ViT against adversarial patches. Extensive qualitative and quantitative experiments are performed to support our findings, understanding, and improvement of ViT robustness to patch-wise perturbations across a set of transformer-based architectures.

摘要: 近年来，视觉转换器(VIT)在图像分类中表现出了令人印象深刻的性能，这使得它成为卷积神经网络(CNN)的一种有前途的替代方案。与CNN不同，VIT将输入图像表示为一系列图像补丁。基于块的输入图像表示使得以下问题变得有趣：与CNN相比，当单个输入图像块受到自然破坏或敌意扰动时，VIT的性能如何？在这项工作中，我们研究了VIT对面片扰动的稳健性。令人惊讶的是，我们发现VITS比CNN对自然损坏的补丁更健壮，而它们更容易受到对手补丁的攻击。此外，我们发现注意机制对视觉转换器的稳健性有很大的影响。具体地说，注意模块可以通过有效地忽略自然损坏的补丁来帮助提高VIT的健壮性。然而，当VITS受到对手的攻击时，注意力机制很容易被愚弄，使其更多地集中在对手扰乱的补丁上，从而导致错误。基于我们的分析，我们提出了一种简单的基于温度缩放的方法来提高VIT对恶意补丁的健壮性。进行了大量的定性和定量实验，以支持我们的发现、理解和提高VIT对一组基于变压器的体系结构的补丁扰动的稳健性。



## **31. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

对基于投影的可取消生物特征识别方案的认证攻击 cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2110.15163v6)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **32. Detection of Poisoning Attacks with Anomaly Detection in Federated Learning for Healthcare Applications: A Machine Learning Approach**

医疗联合学习中异常检测的中毒攻击检测：一种机器学习方法 cs.LG

We will updated this article soon

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08486v1)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstracts**: The application of Federated Learning (FL) is steadily increasing, especially in privacy-aware applications, such as healthcare. However, its applications have been limited by security concerns due to various adversarial attacks, such as poisoning attacks (model and data poisoning). Such attacks attempt to poison the local models and data to manipulate the global models in order to obtain undue benefits and malicious use. Traditional methods of data auditing to mitigate poisoning attacks find their limited applications in FL because the edge devices never share their raw data directly due to privacy concerns, and are globally distributed with no insight into their training data. Thereafter, it is challenging to develop appropriate strategies to address such attacks and minimize their impact on the global model in federated learning. In order to address such challenges in FL, we proposed a novel framework to detect poisoning attacks using deep neural networks and support vector machines, in the form of anomaly without acquiring any direct access or information about the underlying training data of local edge devices. We illustrate and evaluate the proposed framework using different state of art poisoning attacks for two different healthcare applications: Electrocardiograph classification and human activity recognition. Our experimental analysis shows that the proposed method can efficiently detect poisoning attacks and can remove the identified poisoned updated from the global aggregation. Thereafter can increase the performance of the federated global.

摘要: 联合学习(FL)的应用正在稳步增加，特别是在隐私感知应用中，如医疗保健。然而，由于各种对抗性攻击，如中毒攻击(模型中毒和数据中毒)，其应用受到了安全方面的考虑。这类攻击试图毒化本地模型和数据以操纵全局模型，以获取不正当的利益和恶意使用。缓解中毒攻击的传统数据审计方法在FL中的应用有限，因为由于隐私问题，边缘设备从未直接共享它们的原始数据，并且是全球分布的，无法洞察它们的训练数据。此后，制定适当的策略来应对此类攻击并将其对联合学习中的全球模型的影响降至最低是具有挑战性的。为了解决FL中的这种挑战，我们提出了一种新的框架，使用深度神经网络和支持向量机来检测异常形式的中毒攻击，而不需要获取任何关于本地边缘设备的底层训练数据的直接访问或信息。我们使用针对两种不同医疗应用的不同技术水平的中毒攻击来说明和评估所提出的框架：心电图机分类和人类活动识别。实验分析表明，该方法能够有效地检测中毒攻击，并能从全局聚集中剔除已识别的中毒更新。之后可以提高联合全局的性能。



## **33. Towards Automated Classification of Attackers' TTPs by combining NLP with ML Techniques**

结合NLP和ML技术实现攻击者TTP的自动分类 cs.CR

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08478v1)

**Authors**: Clemens Sauerwein, Alexander Pfohl

**Abstracts**: The increasingly sophisticated and growing number of threat actors along with the sheer speed at which cyber attacks unfold, make timely identification of attacks imperative to an organisations' security. Consequently, persons responsible for security employ a large variety of information sources concerning emerging attacks, attackers' course of actions or indicators of compromise. However, a vast amount of the needed security information is available in unstructured textual form, which complicates the automated and timely extraction of attackers' Tactics, Techniques and Procedures (TTPs). In order to address this problem we systematically evaluate and compare different Natural Language Processing (NLP) and machine learning techniques used for security information extraction in research. Based on our investigations we propose a data processing pipeline that automatically classifies unstructured text according to attackers' tactics and techniques derived from a knowledge base of adversary tactics, techniques and procedures.

摘要: 威胁因素的日益复杂和数量不断增加，以及网络攻击展开的速度之快，使得及时识别攻击对组织的安全至关重要。因此，负责安全的人员使用大量关于新出现的攻击、攻击者的行动过程或妥协迹象的信息来源。然而，大量所需的安全信息是以非结构化文本形式提供的，这使得自动、及时地提取攻击者的战术、技术和程序(TTP)变得复杂。为了解决这个问题，我们系统地评估和比较了不同的自然语言处理(NLP)和机器学习技术在安全信息提取中的应用研究。基于我们的研究，我们提出了一种数据处理流水线，它根据攻击者的战术和技术来自动对非结构化文本进行分类，这些策略和技术是从对手的战术、技术和过程知识库中得出的。



## **34. A Perturbation-Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

一种用于评估光流稳健性的扰动约束对抗性攻击 cs.CV

Accepted at the European Conference on Computer Vision (ECCV) 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2203.13214v2)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods focus on real-world attacking scenarios rather than a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation-Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments demonstrate PCFA's applicability in white- and black-box settings, and show it finds stronger adversarial samples than previous attacks. Based on these strong samples, we provide the first joint ranking of optical flow methods considering both prediction quality and adversarial robustness, which reveals state-of-the-art methods to be particularly vulnerable. Code is available at https://github.com/cv-stuttgart/PCFA.

摘要: 最近的光流方法几乎完全是根据准确性来判断的，而它们的稳健性往往被忽视。尽管对抗性攻击为执行这种分析提供了一个有用的工具，但目前对光流方法的攻击主要集中在真实世界的攻击场景上，而不是最坏情况下的健壮性评估。因此，在这项工作中，我们提出了一种新的对抗性攻击-扰动约束流攻击(PCFA)-强调破坏性而不是适用性作为现实世界的攻击。PCFA是一种全局攻击，它优化对抗性扰动，将预测流向指定的目标流移动，同时将扰动的L2范数保持在选定的界限以下。我们的实验证明了PCFA在白盒和黑盒环境下的适用性，并表明它比以前的攻击发现了更强的敌意样本。基于这些强样本，我们提供了第一个综合考虑预测质量和对抗稳健性的光流方法的联合排名，这揭示了最新的方法特别脆弱。代码可在https://github.com/cv-stuttgart/PCFA.上找到



## **35. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2202.07054v3)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，不应忽视它们在对抗性例子面前的脆弱性。在本研究中，我们首次在没有任何受害者模型知识的情况下，系统地分析了遥感数据中的通用对抗性实例。具体来说，我们提出了一种新的针对遥感数据的黑盒对抗攻击方法，即Mixup攻击及其简单的变种MixCut攻击。提出的方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性样本，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还在名为UAE-RS的数据集上提供了生成的通用对抗性实例，这是遥感领域中第一个提供黑盒对抗性样本的数据集。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗能力的深度神经网络。代码和阿联酋-RS数据集可在网上获得(https://github.com/YonghaoXu/UAE-RS).



## **36. Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**

水印疫苗：防止水印去除的对抗性攻击 cs.CV

ECCV 2022

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08178v1)

**Authors**: Xinwei Liu, Jian Liu, Yang Bai, Jindong Gu, Tao Chen, Xiaojun Jia, Xiaochun Cao

**Abstracts**: As a common security tool, visible watermarking has been widely applied to protect copyrights of digital images. However, recent works have shown that visible watermarks can be removed by DNNs without damaging their host images. Such watermark-removal techniques pose a great threat to the ownership of images. Inspired by the vulnerability of DNNs on adversarial perturbations, we propose a novel defence mechanism by adversarial machine learning for good. From the perspective of the adversary, blind watermark-removal networks can be posed as our target models; then we actually optimize an imperceptible adversarial perturbation on the host images to proactively attack against watermark-removal networks, dubbed Watermark Vaccine. Specifically, two types of vaccines are proposed. Disrupting Watermark Vaccine (DWV) induces to ruin the host image along with watermark after passing through watermark-removal networks. In contrast, Inerasable Watermark Vaccine (IWV) works in another fashion of trying to keep the watermark not removed and still noticeable. Extensive experiments demonstrate the effectiveness of our DWV/IWV in preventing watermark removal, especially on various watermark removal networks.

摘要: 可见水印作为一种常用的安全工具，已被广泛应用于数字图像的版权保护。然而，最近的研究表明，DNN可以在不损害宿主图像的情况下去除可见水印。这种水印去除技术对图像的所有权构成了极大的威胁。受DNN对对抗性扰动的脆弱性的启发，我们提出了一种基于对抗性机器学习的新型防御机制。从敌手的角度来看，盲水印去除网络可以作为我们的目标模型，然后我们实际上优化了宿主图像上的一种不可感知的对抗性扰动，以主动攻击水印去除网络，称为水印疫苗。具体地说，提出了两种疫苗。破坏水印疫苗(DWV)在通过水印去除网络后，会导致宿主图像与水印一起被破坏。相比之下，不可擦除水印疫苗(IWV)的另一种工作方式是试图保持水印不被移除并仍然可见。大量的实验证明了我们的DWV/IWV在防止水印去除方面的有效性，特别是在各种水印去除网络上。



## **37. Modeling Adversarial Noise for Adversarial Training**

对抗性训练中的对抗性噪声建模 cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2109.09901v5)

**Authors**: Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu

**Abstracts**: Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.

摘要: 深度神经网络已被证明对对抗性噪声很敏感，这促进了防御对抗性攻击的发展。对抗性噪声包含了良好的泛化特征，并且对抗性数据与自然数据之间的关系可以帮助推断自然数据并做出可靠的预测，本文通过学习对抗性标签(即用于生成对抗性数据的翻转标签)和自然标签(即自然数据的地面真值标签)之间的转换关系来研究对抗性噪声的建模。具体地说，我们引入了依赖于实例的转移矩阵来关联对抗性标签和自然标签，该转移矩阵可以无缝地嵌入目标模型(使我们能够建模更强的自适应对抗性噪声)。实验结果表明，该方法能够有效地提高对手识别的准确率。



## **38. Automated Repair of Neural Networks**

神经网络的自动修复 cs.LG

Code and results are available at  https://github.com/dorcoh/NNSynthesizer

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08157v1)

**Authors**: Dor Cohen, Ofer Strichman

**Abstracts**: Over the last decade, Neural Networks (NNs) have been widely used in numerous applications including safety-critical ones such as autonomous systems. Despite their emerging adoption, it is well known that NNs are susceptible to Adversarial Attacks. Hence, it is highly important to provide guarantees that such systems work correctly. To remedy these issues we introduce a framework for repairing unsafe NNs w.r.t. safety specification, that is by utilizing satisfiability modulo theories (SMT) solvers. Our method is able to search for a new, safe NN representation, by modifying only a few of its weight values. In addition, our technique attempts to maximize the similarity to original network with regard to its decision boundaries. We perform extensive experiments which demonstrate the capability of our proposed framework to yield safe NNs w.r.t. the Adversarial Robustness property, with only a mild loss of accuracy (in terms of similarity). Moreover, we compare our method with a naive baseline to empirically prove its effectiveness. To conclude, we provide an algorithm to automatically repair NNs given safety properties, and suggest a few heuristics to improve its computational performance. Currently, by following this approach we are capable of producing small-sized (i.e., with up to few hundreds of parameters) correct NNs, composed of the piecewise linear ReLU activation function. Nevertheless, our framework is general in the sense that it can synthesize NNs w.r.t. any decidable fragment of first-order logic specification.

摘要: 在过去的十年里，神经网络(NNS)被广泛地应用于许多应用中，包括诸如自治系统等安全关键的应用。尽管它们正在被采用，但众所周知，NNS很容易受到对手的攻击。因此，提供此类系统正常工作的保证是非常重要的。为了解决这些问题，我们引入了一个修复不安全NNW.r.t.的框架。安全规范，即利用可满足性模理论(SMT)求解器。我们的方法能够搜索一个新的、安全的神经网络表示，只需修改它的几个权值。此外，我们的技术试图在决策边界方面最大化与原始网络的相似性。我们进行了大量的实验，证明了我们所提出的框架能够产生安全的NNW.r.t.对抗性稳健性，只有轻微的准确性损失(就相似性而言)。此外，我们将我们的方法与天真的基线进行了比较，以经验证明其有效性。综上所述，我们提出了一种在给定安全属性的情况下自动修复神经网络的算法，并提出了一些启发式算法来提高其计算性能。目前，通过采用这种方法，我们能够产生由分段线性REU激活函数组成的小尺寸(即，多达数百个参数)的正确神经网络。然而，我们的框架在某种意义上是通用的，它可以合成NNS w.r.t.一阶逻辑规范的任何可判定片段。



## **39. Achieve Optimal Adversarial Accuracy for Adversarial Deep Learning using Stackelberg Game**

利用Stackelberg博弈实现对抗性深度学习的最优对抗性准确率 cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08137v1)

**Authors**: Xiao-Shan Gao, Shuang Liu, Lijia Yu

**Abstracts**: Adversarial deep learning is to train robust DNNs against adversarial attacks, which is one of the major research focuses of deep learning. Game theory has been used to answer some of the basic questions about adversarial deep learning such as the existence of a classifier with optimal robustness and the existence of optimal adversarial samples for a given class of classifiers. In most previous work, adversarial deep learning was formulated as a simultaneous game and the strategy spaces are assumed to be certain probability distributions in order for the Nash equilibrium to exist. But, this assumption is not applicable to the practical situation. In this paper, we give answers to these basic questions for the practical case where the classifiers are DNNs with a given structure, by formulating the adversarial deep learning as sequential games. The existence of Stackelberg equilibria for these games are proved. Furthermore, it is shown that the equilibrium DNN has the largest adversarial accuracy among all DNNs with the same structure, when Carlini-Wagner's margin loss is used. Trade-off between robustness and accuracy in adversarial deep learning is also studied from game theoretical aspect.

摘要: 对抗性深度学习是针对敌意攻击训练健壮的DNN，是深度学习的主要研究热点之一。博弈论已经被用来回答对抗性深度学习的一些基本问题，例如具有最优稳健性的分类器的存在以及给定类别的分类器的最优对抗性样本的存在。在以前的工作中，对抗性深度学习被描述为一个联立博弈，并且假设策略空间是一定的概率分布，以便纳什均衡的存在。但是，这一假设并不适用于实际情况。在本文中，我们通过将对抗性深度学习描述为序列博弈，针对分类器是具有给定结构的DNN的实际情况，回答了这些基本问题。证明了这些博弈的Stackelberg均衡的存在性。此外，当使用Carlini-Wagner边际损失时，均衡DNN在所有相同结构的DNN中具有最大的对抗准确率。从博弈论的角度研究了对抗性深度学习的稳健性和精确度之间的权衡。



## **40. Threat Model-Agnostic Adversarial Defense using Diffusion Models**

威胁模型--使用扩散模型的不可知式对抗防御 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08089v1)

**Authors**: Tsachi Blau, Roy Ganz, Bahjat Kawar, Alex Bronstein, Michael Elad

**Abstracts**: Deep Neural Networks (DNNs) are highly sensitive to imperceptible malicious perturbations, known as adversarial attacks. Following the discovery of this vulnerability in real-world imaging and vision applications, the associated safety concerns have attracted vast research attention, and many defense techniques have been developed. Most of these defense methods rely on adversarial training (AT) -- training the classification network on images perturbed according to a specific threat model, which defines the magnitude of the allowed modification. Although AT leads to promising results, training on a specific threat model fails to generalize to other types of perturbations. A different approach utilizes a preprocessing step to remove the adversarial perturbation from the attacked image. In this work, we follow the latter path and aim to develop a technique that leads to robust classifiers across various realizations of threat models. To this end, we harness the recent advances in stochastic generative modeling, and means to leverage these for sampling from conditional distributions. Our defense relies on an addition of Gaussian i.i.d noise to the attacked image, followed by a pretrained diffusion process -- an architecture that performs a stochastic iterative process over a denoising network, yielding a high perceptual quality denoised outcome. The obtained robustness with this stochastic preprocessing step is validated through extensive experiments on the CIFAR-10 dataset, showing that our method outperforms the leading defense methods under various threat models.

摘要: 深度神经网络(DNN)对不可察觉的恶意扰动高度敏感，这种恶意扰动称为对抗性攻击。随着在真实世界的成像和视觉应用中发现该漏洞，相关的安全问题引起了广泛的研究关注，许多防御技术也被开发出来。这些防御方法中的大多数依赖于对抗性训练(AT)-根据特定的威胁模型对受干扰的图像训练分类网络，该模型定义了允许修改的幅度。虽然AT带来了有希望的结果，但对特定威胁模型的训练无法推广到其他类型的扰动。一种不同的方法利用预处理步骤来从被攻击的图像中移除对抗性扰动。在这项工作中，我们遵循后一条道路，目标是开发一种技术，从而在威胁模型的各种实现中产生健壮的分类器。为此，我们利用了随机生成建模的最新进展，并利用这些方法从条件分布中进行抽样。我们的防御依赖于在受攻击的图像中添加高斯I.I.D噪声，然后是预先训练的扩散过程--一种在去噪网络上执行随机迭代过程的体系结构，产生高感知质量的去噪结果。通过在CIFAR-10数据集上的大量实验，验证了该随机预处理步骤所获得的稳健性，表明该方法在各种威胁模型下的性能优于领先的防御方法。



## **41. DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking**

DIMBA：单目标跟踪中的离散掩蔽黑盒攻击 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08044v1)

**Authors**: Xiangyu Yin, Wenjie Ruan, Jonathan Fieldsend

**Abstracts**: The adversarial attack can force a CNN-based model to produce an incorrect output by craftily manipulating human-imperceptible input. Exploring such perturbations can help us gain a deeper understanding of the vulnerability of neural networks, and provide robustness to deep learning against miscellaneous adversaries. Despite extensive studies focusing on the robustness of image, audio, and NLP, works on adversarial examples of visual object tracking -- especially in a black-box manner -- are quite lacking. In this paper, we propose a novel adversarial attack method to generate noises for single object tracking under black-box settings, where perturbations are merely added on initial frames of tracking sequences, which is difficult to be noticed from the perspective of a whole video clip. Specifically, we divide our algorithm into three components and exploit reinforcement learning for localizing important frame patches precisely while reducing unnecessary computational queries overhead. Compared to existing techniques, our method requires fewer queries on initialized frames of a video to manipulate competitive or even better attack performance. We test our algorithm in both long-term and short-term datasets, including OTB100, VOT2018, UAV123, and LaSOT. Extensive experiments demonstrate the effectiveness of our method on three mainstream types of trackers: discrimination, Siamese-based, and reinforcement learning-based trackers.

摘要: 敌意攻击可以通过巧妙地操纵人类无法察觉的输入，迫使基于CNN的模型产生错误的输出。探索这种扰动可以帮助我们更深入地了解神经网络的脆弱性，并为针对各种对手的深度学习提供健壮性。尽管广泛的研究集中在图像、音频和NLP的健壮性上，但关于视觉对象跟踪的对抗性例子--尤其是以黑盒方式--的工作相当缺乏。本文提出了一种新的对抗性攻击方法，用于黑盒环境下的单目标跟踪，该方法只对跟踪序列的初始帧添加扰动，从整个视频片段的角度来看很难注意到这一点。具体地说，我们将算法分为三个部分，并利用强化学习来精确定位重要的帧补丁，同时减少不必要的计算查询开销。与现有技术相比，我们的方法需要对视频的初始化帧进行更少的查询来操纵竞争甚至更好的攻击性能。我们在长期和短期数据集上测试了我们的算法，包括OTB100、VOT2018、UAV123和LaSOT。大量的实验表明，我们的方法在三种主流类型的跟踪器上是有效的：区分跟踪器、基于暹罗的跟踪器和基于强化学习的跟踪器。



## **42. Optimal Strategic Mining Against Cryptographic Self-Selection in Proof-of-Stake**

基于密码学自我选择的最优策略挖掘 cs.CR

31 pages, ACM EC 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07996v1)

**Authors**: Matheus V. X. Ferreira, Ye Lin Sally Hahn, S. Matthew Weinberg, Catherine Yu

**Abstracts**: Cryptographic Self-Selection is a subroutine used to select a leader for modern proof-of-stake consensus protocols, such as Algorand. In cryptographic self-selection, each round $r$ has a seed $Q_r$. In round $r$, each account owner is asked to digitally sign $Q_r$, hash their digital signature to produce a credential, and then broadcast this credential to the entire network. A publicly-known function scores each credential in a manner so that the distribution of the lowest scoring credential is identical to the distribution of stake owned by each account. The user who broadcasts the lowest-scoring credential is the leader for round $r$, and their credential becomes the seed $Q_{r+1}$. Such protocols leave open the possibility of a selfish-mining style attack: a user who owns multiple accounts that each produce low-scoring credentials in round $r$ can selectively choose which ones to broadcast in order to influence the seed for round $r+1$. Indeed, the user can pre-compute their credentials for round $r+1$ for each potential seed, and broadcast only the credential (among those with a low enough score to be the leader) that produces the most favorable seed.   We consider an adversary who wishes to maximize the expected fraction of rounds in which an account they own is the leader. We show such an adversary always benefits from deviating from the intended protocol, regardless of the fraction of the stake controlled. We characterize the optimal strategy; first by proving the existence of optimal positive recurrent strategies whenever the adversary owns last than $38\%$ of the stake. Then, we provide a Markov Decision Process formulation to compute the optimal strategy.

摘要: 加密自我选择是用于为现代利害关系证明共识协议(如算法)选择领导者的子例程。在加密自我选择中，每轮$r$都有一个种子$q_r$。在$r$中，每个帐户所有者被要求对$q_r$进行数字签名，对他们的数字签名进行散列以生成凭据，然后将该凭据广播到整个网络。公知函数以一种方式对每个凭证进行评分，使得得分最低的凭证的分布与每个帐户拥有的赌注的分布相同。广播得分最低的凭据的用户是$r$的领先者，他们的凭据成为种子$q_{r+1}$。这样的协议为自私挖掘式攻击提供了可能性：拥有多个账户的用户，每个账户都在$r$中产生低得分凭据，可以有选择地选择广播哪些账户，以便在$r+1$中影响种子。事实上，用户可以为每个潜在种子预先计算$r+1$的凭据，并且只广播产生最有利种子的凭据(在那些得分足够低的人中)。我们考虑一个对手，他希望最大化他们拥有的账户是领导者的回合的预期比例。我们表明，这样的对手总是从偏离预期的协议中受益，无论控制的风险有多大。我们刻画了最优策略：首先，证明了当对手拥有最后$38的股份时，最优正回归策略的存在性。然后，我们给出了一个马尔可夫决策过程公式来计算最优策略。



## **43. BOSS: Bidirectional One-Shot Synthesis of Adversarial Examples**

BOSS：对抗性范例的双向一次合成 cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2108.02756v2)

**Authors**: Ismail R. Alkhouri, Alvaro Velasquez, George K. Atia

**Abstracts**: The design of additive imperceptible perturbations to the inputs of deep classifiers to maximize their misclassification rates is a central focus of adversarial machine learning. An alternative approach is to synthesize adversarial examples from scratch using GAN-like structures, albeit with the use of large amounts of training data. By contrast, this paper considers one-shot synthesis of adversarial examples; the inputs are synthesized from scratch to induce arbitrary soft predictions at the output of pre-trained models, while simultaneously maintaining high similarity to specified inputs. To this end, we present a problem that encodes objectives on the distance between the desired and output distributions of the trained model and the similarity between such inputs and the synthesized examples. We prove that the formulated problem is NP-complete. Then, we advance a generative approach to the solution in which the adversarial examples are obtained as the output of a generative network whose parameters are iteratively updated by optimizing surrogate loss functions for the dual-objective. We demonstrate the generality and versatility of the framework and approach proposed through applications to the design of targeted adversarial attacks, generation of decision boundary samples, and synthesis of low confidence classification inputs. The approach is further extended to an ensemble of models with different soft output specifications. The experimental results verify that the targeted and confidence reduction attack methods developed perform on par with state-of-the-art algorithms.

摘要: 设计对深层分类器的输入进行不可察觉的加性扰动以最大化其错误分类率是对抗性机器学习的中心问题。另一种方法是使用GaN类结构从头开始合成对抗性例子，尽管使用了大量的训练数据。相反，本文考虑了对抗性例子的一次合成；输入是从头开始合成的，在预先训练的模型的输出处诱导出任意的软预测，同时保持与指定输入的高度相似。为此，我们提出了一个问题，即根据训练模型的期望分布和输出分布之间的距离以及这些输入和合成样本之间的相似性对目标进行编码。我们证明了所提出的问题是NP完全的。然后，我们提出了一种求解该问题的产生式方法，即通过优化双目标的代理损失函数，将对抗性实例作为产生式网络的输出来迭代地更新其参数。我们通过在目标对抗性攻击的设计、决策边界样本的生成和低置信度分类输入的合成中的应用，展示了所提出的框架和方法的通用性和通用性。该方法进一步扩展到具有不同软输出规格的模型集成。实验结果表明，本文提出的目标攻击和置信度降低攻击方法的性能与目前最先进的算法相当。



## **44. Security and Safety Aspects of AI in Industry Applications**

人工智能在工业应用中的安全和安全问题 cs.CR

As presented at the Embedded World Conference, Nuremberg, 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.10809v1)

**Authors**: Hans Dermot Doran

**Abstracts**: In this relatively informal discussion-paper we summarise issues in the domains of safety and security in machine learning that will affect industry sectors in the next five to ten years. Various products using neural network classification, most often in vision related applications but also in predictive maintenance, have been researched and applied in real-world applications in recent years. Nevertheless, reports of underlying problems in both safety and security related domains, for instance adversarial attacks have unsettled early adopters and are threatening to hinder wider scale adoption of this technology. The problem for real-world applicability lies in being able to assess the risk of applying these technologies. In this discussion-paper we describe the process of arriving at a machine-learnt neural network classifier pointing out safety and security vulnerabilities in that workflow, citing relevant research where appropriate.

摘要: 在这份相对非正式的讨论文件中，我们总结了机器学习中安全和安保领域的问题，这些问题将在未来五到十年影响到工业部门。近年来，各种使用神经网络分类的产品被研究并应用于实际应用中，其中最常见的是与视觉相关的应用，但也用于预测维护。然而，有关安全和安保相关领域的潜在问题的报道，例如对抗性攻击，让早期采用者感到不安，并有可能阻碍这项技术的更广泛采用。现实世界适用性的问题在于能否评估应用这些技术的风险。在这篇讨论论文中，我们描述了获得机器学习的神经网络分类器的过程，指出了该工作流中的安全和安全漏洞，并在适当的地方引用了相关研究。



## **45. Certified Neural Network Watermarks with Randomized Smoothing**

随机平滑认证神经网络水印 cs.LG

ICML 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07972v1)

**Authors**: Arpit Bansal, Ping-yeh Chiang, Michael Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P Dickerson, Tom Goldstein

**Abstracts**: Watermarking is a commonly used strategy to protect creators' rights to digital images, videos and audio. Recently, watermarking methods have been extended to deep learning models -- in principle, the watermark should be preserved when an adversary tries to copy the model. However, in practice, watermarks can often be removed by an intelligent adversary. Several papers have proposed watermarking methods that claim to be empirically resistant to different types of removal attacks, but these new techniques often fail in the face of new or better-tuned adversaries. In this paper, we propose a certifiable watermarking method. Using the randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain l2 threshold. In addition to being certifiable, our watermark is also empirically more robust compared to previous watermarking methods. Our experiments can be reproduced with code at https://github.com/arpitbansal297/Certified_Watermarks

摘要: 水印是保护创作者对数字图像、视频和音频的权利的一种常用策略。最近，水印方法已经扩展到深度学习模型--原则上，当对手试图复制模型时，水印应该被保留。然而，在实践中，水印往往可以被聪明的对手移除。有几篇论文提出了一些水印方法，这些方法声称在经验上可以抵抗不同类型的删除攻击，但这些新技术在面对新的或调整得更好的对手时往往会失败。在本文中，我们提出了一种可认证的水印方法。使用Chiang等人提出的随机平滑技术，我们证明了除非模型参数改变超过一定的L2阈值，否则我们的水印是不可移除的。除了是可认证的，我们的水印在经验上也比以前的水印方法更健壮。我们的实验可以通过https://github.com/arpitbansal297/Certified_Watermarks上的代码重现



## **46. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

MixTailor：针对定制攻击的稳健学习的混合梯度聚合 cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07941v1)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed and multi-GPU systems creates new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.

摘要: SGD在分布式和多GPU系统上的实现会产生新的漏洞，这些漏洞可能会被一个或多个对抗性代理识别和滥用。最近，有研究表明，众所周知的拜占庭弹性梯度聚合方案确实容易受到可以定制攻击的知情攻击者的攻击(方等人，2020；谢等人，2020b)。我们引入了MixTailor，这是一种基于聚合策略随机化的方案，使得攻击者不可能被完全告知。确定性方案可以动态地集成到MixTailor中，而不需要引入任何额外的超参数。随机化降低了强大对手定制其攻击的能力，而由此产生的随机化聚集方案在性能方面仍然具有竞争力。对于iID和非iID设置，我们几乎肯定建立了比文献中提供的更强大和更一般的收敛保证。我们对各种数据集、攻击和环境的经验研究验证了我们的假设，并表明当众所周知的拜占庭容忍方案失败时，MixTailor成功地进行了辩护。



## **47. Masked Spatial-Spectral Autoencoders Are Excellent Hyperspectral Defenders**

屏蔽式空间光谱自动编码器是优秀的高光谱防御者 cs.CV

14 pages, 9 figures

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07803v1)

**Authors**: Jiahao Qi, Zhiqiang Gong, Xingyue Liu, Kangcheng Bin, Chen Chen, Yongqian Li, Wei Xue, Yu Zhang, Ping Zhong

**Abstracts**: Deep learning methodology contributes a lot to the development of hyperspectral image (HSI) analysis community. However, it also makes HSI analysis systems vulnerable to adversarial attacks. To this end, we propose a masked spatial-spectral autoencoder (MSSA) in this paper under self-supervised learning theory, for enhancing the robustness of HSI analysis systems. First, a masked sequence attention learning module is conducted to promote the inherent robustness of HSI analysis systems along spectral channel. Then, we develop a graph convolutional network with learnable graph structure to establish global pixel-wise combinations.In this way, the attack effect would be dispersed by all the related pixels among each combination, and a better defense performance is achievable in spatial aspect.Finally, to improve the defense transferability and address the problem of limited labelled samples, MSSA employs spectra reconstruction as a pretext task and fits the datasets in a self-supervised manner.Comprehensive experiments over three benchmarks verify the effectiveness of MSSA in comparison with the state-of-the-art hyperspectral classification methods and representative adversarial defense strategies.

摘要: 深度学习方法对高光谱图像(HSI)分析社区的发展做出了重要贡献。然而，它也使HSI分析系统容易受到对手攻击。为此，本文提出了一种基于自监督学习理论的屏蔽空间谱自动编码器(MSSA)，以增强HSI分析系统的鲁棒性。首先，采用掩蔽序列注意学习模块来提高HSI分析系统在频谱信道上的固有健壮性。然后，提出了一种具有可学习图结构的图卷积网络来建立全局像素级组合，这样攻击效果将由每个组合之间的所有相关像素来分散，从而在空间方面获得更好的防御性能；最后，为了提高防御的可传递性并解决标记样本有限的问题，MSSA采用谱重建作为借口任务，并以自监督的方式对数据集进行拟合。



## **48. CARBEN: Composite Adversarial Robustness Benchmark**

Carben：复合对抗健壮性基准 cs.CV

IJCAI 2022 Demo Track; The demonstration is at  https://hsiung.cc/CARBEN/

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07797v1)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Prior literature on adversarial attack methods has mainly focused on attacking with and defending against a single threat model, e.g., perturbations bounded in Lp ball. However, multiple threat models can be combined into composite perturbations. One such approach, composite adversarial attack (CAA), not only expands the perturbable space of the image, but also may be overlooked by current modes of robustness evaluation. This paper demonstrates how CAA's attack order affects the resulting image, and provides real-time inferences of different models, which will facilitate users' configuration of the parameters of the attack level and their rapid evaluation of model prediction. A leaderboard to benchmark adversarial robustness against CAA is also introduced.

摘要: 以前关于对抗性攻击方法的文献主要集中在利用单一威胁模型进行攻击和防御，例如，在LP球中有界的扰动。但是，可以将多个威胁模型组合成复合扰动。一种这样的方法，复合对抗攻击(CAA)，不仅扩展了图像的可扰动空间，而且可能被现有的健壮性评估模式所忽视。文中演示了CAA的攻击顺序对结果图像的影响，并提供了不同模型的实时推理，这将方便用户配置攻击级别的参数，并快速评估模型预测。此外，还引入了一个排行榜来衡量对抗CAA的健壮性。



## **49. Towards the Desirable Decision Boundary by Moderate-Margin Adversarial Training**

通过适度的对抗性训练达到理想的决策边界 cs.CV

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07793v1)

**Authors**: Xiaoyu Liang, Yaguan Qian, Jianchang Huang, Xiang Ling, Bin Wang, Chunming Wu, Wassim Swaileh

**Abstracts**: Adversarial training, as one of the most effective defense methods against adversarial attacks, tends to learn an inclusive decision boundary to increase the robustness of deep learning models. However, due to the large and unnecessary increase in the margin along adversarial directions, adversarial training causes heavy cross-over between natural examples and adversarial examples, which is not conducive to balancing the trade-off between robustness and natural accuracy. In this paper, we propose a novel adversarial training scheme to achieve a better trade-off between robustness and natural accuracy. It aims to learn a moderate-inclusive decision boundary, which means that the margins of natural examples under the decision boundary are moderate. We call this scheme Moderate-Margin Adversarial Training (MMAT), which generates finer-grained adversarial examples to mitigate the cross-over problem. We also take advantage of logits from a teacher model that has been well-trained to guide the learning of our model. Finally, MMAT achieves high natural accuracy and robustness under both black-box and white-box attacks. On SVHN, for example, state-of-the-art robustness and natural accuracy are achieved.

摘要: 对抗性训练作为对抗对抗性攻击的最有效的防御方法之一，倾向于学习一个包容的决策边界，以增加深度学习模型的鲁棒性。然而，由于对抗性方向上的差值有很大且不必要的增加，对抗性训练导致自然例子和对抗性例子之间的严重交叉，这不利于在稳健性和自然准确性之间取得平衡。在本文中，我们提出了一种新的对抗性训练方案，以在稳健性和自然准确性之间实现更好的权衡。它的目的是学习一个适度包容的决策边界，这意味着决策边界下的自然例子的边际是适度的。我们称之为中等边际对抗性训练(MMAT)，它生成更细粒度的对抗性例子来缓解交叉问题。我们还利用训练有素的教师模型的Logit来指导我们模型的学习。最后，在黑盒和白盒攻击下，MMAT都达到了很高的自然准确率和稳健性。例如，在SVHN上，实现了最先进的健壮性和自然准确性。



## **50. Demystifying the Adversarial Robustness of Random Transformation Defenses**

揭开随机变换防御对抗健壮性的神秘面纱 cs.CR

ICML 2022 (short presentation), AAAI 2022 AdvML Workshop (best paper,  oral presentation)

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.03574v2)

**Authors**: Chawin Sitawarin, Zachary Golan-Strieb, David Wagner

**Abstracts**: Neural networks' lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT's evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack ($4.3\times$ improvement). Our result indicates that the RT defense on the Imagenette dataset (a ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain. Code is available at https://github.com/wagner-group/demystify-random-transform.

摘要: 神经网络对攻击缺乏稳健性，这在自动驾驶汽车等安全敏感环境中引发了担忧。尽管许多对策看起来很有希望，但只有少数几项经得起严格的评估。使用随机变换(RT)的防御已经显示出令人印象深刻的结果，特别是Bart(Raff等人，2019年)在ImageNet上。然而，这种类型的防御没有得到严格的评估，使得人们对其健壮性属性知之甚少。它们的随机性使评估变得更具挑战性，并使许多已提出的对确定性模型的攻击不适用。首先，我们证明了BART评估中使用的BPDA攻击(Athalye等人，2018a)是无效的，并且可能高估了它的健壮性。然后，我们试图通过对变换的知情选择和贝叶斯优化来调整它们的参数，从而构建尽可能强的RT防御。此外，我们创建了尽可能强的攻击来评估我们的RT防御。我们的新攻击大大超过了基准，与常用的EoT攻击相比，准确率降低了83%($4.3倍$改进)。我们的结果表明，在Imagenette数据集(ImageNet的十类子集)上的RT防御对敌意示例不是健壮的。进一步扩展研究，我们使用我们的新攻击来恶意训练RT防御(称为AdvRT)，从而获得了很大的健壮性收益。代码可在https://github.com/wagner-group/demystify-random-transform.上找到



