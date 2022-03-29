# Latest Adversarial Attack Papers
**update at 2022-03-30 06:31:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Poisoning and Backdooring Contrastive Learning**

中毒与倒退对比学习 cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2106.09667v2)

**Authors**: Nicholas Carlini, Andreas Terzis

**Abstracts**: Multimodal contrastive learning methods like CLIP train on noisy and uncurated training datasets. This is cheaper than labeling datasets manually, and even improves out-of-distribution robustness. We show that this practice makes backdoor and poisoning attacks a significant threat. By poisoning just 0.01% of a dataset (e.g., just 300 images of the 3 million-example Conceptual Captions dataset), we can cause the model to misclassify test images by overlaying a small patch. Targeted poisoning attacks, whereby the model misclassifies a particular test input with an adversarially-desired label, are even easier requiring control of 0.0001% of the dataset (e.g., just three out of the 3 million images). Our attacks call into question whether training on noisy and uncurated Internet scrapes is desirable.

摘要: 多模式对比学习方法，如CLIP，在噪声和未经过处理的训练数据集上进行训练。这比手动标记数据集更便宜，甚至提高了分布外的健壮性。我们表明，这种做法使后门和中毒攻击成为一种重大威胁。通过只毒化0.01%的数据集(例如，仅300万个概念字幕数据集的300张图像)，我们可以通过叠加一个小补丁来导致模型对测试图像进行错误分类。有针对性的中毒攻击，即模型将特定的测试输入与对手希望的标签进行错误分类，甚至更容易，需要控制0.0001的数据集(例如，仅控制300万张图像中的3张)。我们的攻击让人质疑，对嘈杂和未经管理的互联网擦伤进行培训是否可取。



## **2. Boosting Black-Box Adversarial Attacks with Meta Learning**

利用元学习增强黑箱对抗攻击 cs.LG

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.14607v1)

**Authors**: Junjie Fu, Jian Sun, Gang Wang

**Abstracts**: Deep neural networks (DNNs) have achieved remarkable success in diverse fields. However, it has been demonstrated that DNNs are very vulnerable to adversarial examples even in black-box settings. A large number of black-box attack methods have been proposed to in the literature. However, those methods usually suffer from low success rates and large query counts, which cannot fully satisfy practical purposes. In this paper, we propose a hybrid attack method which trains meta adversarial perturbations (MAPs) on surrogate models and performs black-box attacks by estimating gradients of the models. Our method uses the meta adversarial perturbation as an initialization and subsequently trains any black-box attack method for several epochs. Furthermore, the MAPs enjoy favorable transferability and universality, in the sense that they can be employed to boost performance of other black-box adversarial attack methods. Extensive experiments demonstrate that our method can not only improve the attack success rates, but also reduces the number of queries compared to other methods.

摘要: 深度神经网络(DNN)在各个领域都取得了显著的成功。然而，已经证明，即使在黑盒环境中，DNN也非常容易受到敌意示例的攻击。文献中提出了大量的黑盒攻击方法。然而，这些方法通常存在成功率低、查询次数多等问题，不能完全满足实际应用的需要。在本文中，我们提出了一种混合攻击方法，该方法在代理模型上训练元对抗扰动(MAP)，并通过估计模型的梯度来执行黑盒攻击。我们的方法使用元对抗扰动作为初始化，并随后在几个时期训练任何黑盒攻击方法。此外，映射具有良好的可转移性和普适性，可以用来提高其他黑盒对抗攻击方法的性能。大量实验表明，与其他方法相比，该方法不仅可以提高攻击成功率，而且可以减少查询次数。



## **3. Protecting Facial Privacy: Generating Adversarial Identity Masks via Style-robust Makeup Transfer**

保护面部隐私：通过风格稳健的化妆传输生成敌意身份面具 cs.CV

Accepted by CVPR2022. Code is available at  https://github.com/CGCL-codes/AMT-GAN

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.03121v2)

**Authors**: Shengshan Hu, Xiaogeng Liu, Yechao Zhang, Minghui Li, Leo Yu Zhang, Hai Jin, Libing Wu

**Abstracts**: While deep face recognition (FR) systems have shown amazing performance in identification and verification, they also arouse privacy concerns for their excessive surveillance on users, especially for public face images widely spread on social networks. Recently, some studies adopt adversarial examples to protect photos from being identified by unauthorized face recognition systems. However, existing methods of generating adversarial face images suffer from many limitations, such as awkward visual, white-box setting, weak transferability, making them difficult to be applied to protect face privacy in reality. In this paper, we propose adversarial makeup transfer GAN (AMT-GAN), a novel face protection method aiming at constructing adversarial face images that preserve stronger black-box transferability and better visual quality simultaneously. AMT-GAN leverages generative adversarial networks (GAN) to synthesize adversarial face images with makeup transferred from reference images. In particular, we introduce a new regularization module along with a joint training strategy to reconcile the conflicts between the adversarial noises and the cycle consistence loss in makeup transfer, achieving a desirable balance between the attack strength and visual changes. Extensive experiments verify that compared with state of the arts, AMT-GAN can not only preserve a comfortable visual quality, but also achieve a higher attack success rate over commercial FR APIs, including Face++, Aliyun, and Microsoft.

摘要: 虽然深度人脸识别系统在身份识别和验证方面表现出了惊人的性能，但它们也因为对用户的过度监控而引起了隐私问题，特别是对社交网络上广泛传播的公共人脸图像。最近，一些研究采用对抗性的例子来保护照片不被未经授权的人脸识别系统识别。然而，现有的生成敌意人脸图像的方法存在着视觉上的笨拙、白盒设置、可移植性较弱等诸多局限性，难以应用于现实中的人脸隐私保护。在本文中，我们提出了一种新的人脸保护方法--对抗性化妆转移GAN(AMT-GAN)，该方法旨在构建对抗性人脸图像，同时保持较强的黑盒可转移性和较好的视觉质量。AMT-GAN利用生成性对抗性网络(GAN)来合成带有参考图像化妆的对抗性人脸图像。特别是，我们引入了一种新的正则化模型和一种联合训练策略来协调化妆转移中对抗性噪声和循环一致性损失之间的冲突，实现了攻击强度和视觉变化之间的理想平衡。广泛的实验证明，与现有技术相比，AMT-GAN不仅可以保持舒适的视觉质量，而且比Face++、阿里云、微软等商用FR API具有更高的攻击成功率。



## **4. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

自适应自动攻击对敌方健壮性的实用评估 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2203.05154v3)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$

摘要: 对抗对手攻击的防御模型已经显著增长，但缺乏实用的评估方法阻碍了进展。评估可以定义为在给定预算迭代次数和测试数据集的情况下寻找防御模型的健壮性下限。一种实用的评估方法应该是方便(即无参数)、高效(即迭代次数较少)和可靠(即接近鲁棒性的下界)。针对这一目标，我们提出了一种无参数的自适应自动攻击(A$^3$)评估方法，该方法以测试时间训练的方式来解决效率和可靠性问题。具体地说，通过观察特定防御模型的对抗性示例在起始点遵循一定的规律，我们设计了一种自适应方向初始化策略来加快评估速度。此外，为了在预算迭代次数下逼近鲁棒性的下界，我们提出了一种基于在线统计的丢弃策略，自动识别和丢弃不易攻击的图像。广泛的实验证明了我们的澳元^3元的有效性。特别是，我们将澳元^3美元应用于近50种广泛使用的防御模型。通过比现有方法消耗更少的迭代次数，即平均$1/10$(10$\倍$加速)，我们在所有情况下都获得了较低的鲁棒精度。值得注意的是，我们用这种方法赢得了CVPR 2021年白盒对抗性攻击防御模型比赛1681支队伍中的$\textbf{第一名}$。代码可在以下网址获得：$\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **5. Essential Features: Content-Adaptive Pixel Discretization to Improve Model Robustness to Adaptive Adversarial Attacks**

基本特征：内容自适应像素离散化，以提高模型对自适应攻击的稳健性 cs.CV

**SubmitDate**: 2022-03-28    [paper-pdf](http://arxiv.org/pdf/2012.01699v3)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstracts**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets such as MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We propose a per-image adaptive preprocessing defense called Essential Features, which first applies adaptive blurring to push perturbed pixel values back to their original value and then discretizes the image to an image-adaptive codebook to reduce the color space. Essential Features thus constrains the attack space by forcing the adversary to perturb large regions both locally and color-wise for its effects to survive the preprocessing. Against adaptive attacks, we find that our approach increases the $L_2$ and $L_\infty$ robustness on higher resolution datasets.

摘要: 像像素离散化这样的预处理防御由于其简单性而被用来消除对抗性攻击。然而，它们已经被证明是无效的，除非在MNIST这样的简单数据集上。我们假设现有的离散化方法失败了，因为对整个数据集使用固定的码本限制了它们平衡图像表示和码字可分性的能力。我们提出了一种称为基本特征的逐图像自适应预处理防御方法，它首先应用自适应模糊将扰动的像素值恢复到其原始值，然后将图像离散为图像自适应码本以减少颜色空间。因此，基本特征通过迫使对手在局部和颜色上扰乱大片区域来限制攻击空间，以使其效果在预处理过程中幸存下来。对于自适应攻击，我们发现我们的方法在更高分辨率的数据集上提高了$L_2$和$L_INFTY$的稳健性。



## **6. Adversarial Representation Sharing: A Quantitative and Secure Collaborative Learning Framework**

对抗性表征共享：一种量化、安全的协作学习框架 cs.CR

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14299v1)

**Authors**: Jikun Chen, Feng Qiang, Na Ruan

**Abstracts**: The performance of deep learning models highly depends on the amount of training data. It is common practice for today's data holders to merge their datasets and train models collaboratively, which yet poses a threat to data privacy. Different from existing methods such as secure multi-party computation (MPC) and federated learning (FL), we find representation learning has unique advantages in collaborative learning due to the lower communication overhead and task-independency. However, data representations face the threat of model inversion attacks. In this article, we formally define the collaborative learning scenario, and quantify data utility and privacy. Then we present ARS, a collaborative learning framework wherein users share representations of data to train models, and add imperceptible adversarial noise to data representations against reconstruction or attribute extraction attacks. By evaluating ARS in different contexts, we demonstrate that our mechanism is effective against model inversion attacks, and achieves a balance between privacy and utility. The ARS framework has wide applicability. First, ARS is valid for various data types, not limited to images. Second, data representations shared by users can be utilized in different tasks. Third, the framework can be easily extended to the vertical data partitioning scenario.

摘要: 深度学习模型的性能在很大程度上取决于训练数据量。对于今天的数据持有者来说，合并他们的数据集和协作训练模型是一种常见的做法，但这对数据隐私构成了威胁。不同于现有的安全多方计算(MPC)和联合学习(FL)等方法，我们发现表征学习在协作学习中具有独特的优势，因为它具有较低的通信开销和任务无关性。然而，数据表示面临着模型反转攻击的威胁。在本文中，我们正式定义了协作学习场景，并量化了数据效用和隐私。然后，我们提出了一种协作学习框架ARS，在该框架中，用户共享数据表示以训练模型，并在数据表示中添加不可察觉的对抗性噪声以抵抗重构或属性提取攻击。通过在不同环境下对ARS的评估，我们证明了该机制对模型反转攻击是有效的，并在隐私和效用之间取得了平衡。ARS框架具有广泛的适用性。首先，ARS适用于各种数据类型，而不限于图像。其次，用户共享的数据表示可以在不同的任务中使用。第三，该框架可以很容易地扩展到垂直数据分区场景。



## **7. Rebuild and Ensemble: Exploring Defense Against Text Adversaries**

重建与整合：探索对文本对手的防御 cs.CL

work in progress

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14207v1)

**Authors**: Linyang Li, Demin Song, Jiehang Zeng, Ruotian Ma, Xipeng Qiu

**Abstracts**: Adversarial attacks can mislead strong neural models; as such, in NLP tasks, substitution-based attacks are difficult to defend. Current defense methods usually assume that the substitution candidates are accessible, which cannot be widely applied against adversarial attacks unless knowing the mechanism of the attacks. In this paper, we propose a \textbf{Rebuild and Ensemble} Framework to defend against adversarial attacks in texts without knowing the candidates. We propose a rebuild mechanism to train a robust model and ensemble the rebuilt texts during inference to achieve good adversarial defense results. Experiments show that our method can improve accuracy under the current strong attack methods.

摘要: 对抗性攻击会误导强大的神经模型；因此，在NLP任务中，基于替换的攻击很难防御。目前的防御方法通常假设替换候选者是可访问的，除非了解攻击的机制，否则不能广泛应用于对抗攻击。在本文中，我们提出了一个在不知道候选者的情况下防御文本中的敌意攻击的文本重建与集成框架。我们提出了一种重建机制来训练一个健壮的模型，并在推理过程中对重建的文本进行集成，以达到良好的对抗防御效果。实验表明，在现有的强攻击方法下，我们的方法能够提高准确率。



## **8. HINT: Hierarchical Neuron Concept Explainer**

提示：层次化神经元概念解释器 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14196v1)

**Authors**: Andong Wang, Wei-Ning Lee, Xiaojuan Qi

**Abstracts**: To interpret deep networks, one main approach is to associate neurons with human-understandable concepts. However, existing methods often ignore the inherent relationships of different concepts (e.g., dog and cat both belong to animals), and thus lose the chance to explain neurons responsible for higher-level concepts (e.g., animal). In this paper, we study hierarchical concepts inspired by the hierarchical cognition process of human beings. To this end, we propose HIerarchical Neuron concepT explainer (HINT) to effectively build bidirectional associations between neurons and hierarchical concepts in a low-cost and scalable manner. HINT enables us to systematically and quantitatively study whether and how the implicit hierarchical relationships of concepts are embedded into neurons, such as identifying collaborative neurons responsible to one concept and multimodal neurons for different concepts, at different semantic levels from concrete concepts (e.g., dog) to more abstract ones (e.g., animal). Finally, we verify the faithfulness of the associations using Weakly Supervised Object Localization, and demonstrate its applicability in various tasks such as discovering saliency regions and explaining adversarial attacks. Code is available on https://github.com/AntonotnaWang/HINT.

摘要: 要解释深层网络，一种主要的方法是将神经元与人类可以理解的概念联系起来。然而，现有的方法往往忽略了不同概念之间的内在联系(例如，狗和猫都属于动物)，从而失去了解释负责更高层次概念(例如，动物)的神经元的机会。受人类层次化认知过程的启发，本文研究了层次化概念。为此，我们提出了层次化神经元概念解释器(HINT)，以低成本和可扩展的方式有效地建立神经元和层次化概念之间的双向关联。提示使我们能够系统和定量地研究概念的隐含层次关系是否以及如何嵌入到神经元中，例如识别负责一个概念的协作神经元和负责不同概念的多通道神经元，从具体的概念(如狗)到更抽象的概念(如动物)，在不同的语义水平上识别。最后，我们使用弱监督对象定位验证了关联的可信性，并证明了它在发现显著区域和解释对抗性攻击等各种任务中的适用性。代码可在https://github.com/AntonotnaWang/HINT.上找到



## **9. How to Robustify Black-Box ML Models? A Zeroth-Order Optimization Perspective**

如何将黑箱ML模型规模化？零阶最优化视角 cs.LG

Accepted as ICLR'22 Spotlight Paper

**SubmitDate**: 2022-03-27    [paper-pdf](http://arxiv.org/pdf/2203.14195v1)

**Authors**: Yimeng Zhang, Yuguang Yao, Jinghan Jia, Jinfeng Yi, Mingyi Hong, Shiyu Chang, Sijia Liu

**Abstracts**: The lack of adversarial robustness has been recognized as an important issue for state-of-the-art machine learning (ML) models, e.g., deep neural networks (DNNs). Thereby, robustifying ML models against adversarial attacks is now a major focus of research. However, nearly all existing defense methods, particularly for robust training, made the white-box assumption that the defender has the access to the details of an ML model (or its surrogate alternatives if available), e.g., its architectures and parameters. Beyond existing works, in this paper we aim to address the problem of black-box defense: How to robustify a black-box model using just input queries and output feedback? Such a problem arises in practical scenarios, where the owner of the predictive model is reluctant to share model information in order to preserve privacy. To this end, we propose a general notion of defensive operation that can be applied to black-box models, and design it through the lens of denoised smoothing (DS), a first-order (FO) certified defense technique. To allow the design of merely using model queries, we further integrate DS with the zeroth-order (gradient-free) optimization. However, a direct implementation of zeroth-order (ZO) optimization suffers a high variance of gradient estimates, and thus leads to ineffective defense. To tackle this problem, we next propose to prepend an autoencoder (AE) to a given (black-box) model so that DS can be trained using variance-reduced ZO optimization. We term the eventual defense as ZO-AE-DS. In practice, we empirically show that ZO-AE- DS can achieve improved accuracy, certified robustness, and query complexity over existing baselines. And the effectiveness of our approach is justified under both image classification and image reconstruction tasks. Codes are available at https://github.com/damon-demon/Black-Box-Defense.

摘要: 缺乏对抗性已经被认为是最先进的机器学习(ML)模型的一个重要问题，例如，深度神经网络(DNN)。因此，增强ML模型的抗敌意攻击能力是目前研究的重点。然而，几乎所有现有的防御方法，特别是对于稳健训练，都建立在白盒假设下，即防御者可以访问ML模型(或其替代方案，如果可用)的细节，例如其体系结构和参数。在已有工作的基础上，本文旨在解决黑盒防御问题：如何仅使用输入查询和输出反馈来增强黑盒模型的健壮性？这样的问题出现在实际场景中，其中预测模型的所有者不愿共享模型信息以保护隐私。为此，我们提出了适用于黑盒模型的防御操作的一般概念，并通过一阶(FO)认证的防御技术去噪平滑(DS)的透镜来设计它。为了允许只使用模型查询的设计，我们进一步将DS与零阶(无梯度)优化相结合。然而，直接实现零阶(ZO)优化会遇到梯度估计的高方差，从而导致无效防御。为了解决这个问题，我们接下来建议在给定的(黑盒)模型中预先设置一个自动编码器，以便可以使用方差减少的ZO优化来训练DS。我们称最终的防御为ZO-AE-DS。在实践中，我们的经验表明，ZO-AE-DS可以在现有基线上获得更高的准确率、经过验证的健壮性和查询复杂性。并在图像分类和图像重建任务中验证了该方法的有效性。有关代码，请访问https://github.com/damon-demon/Black-Box-Defense.



## **10. Reverse Engineering of Imperceptible Adversarial Image Perturbations**

不可感知的对抗性图像扰动的逆向工程 cs.CV

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.14145v1)

**Authors**: Yifan Gong, Yuguang Yao, Yize Li, Yimeng Zhang, Xiaoming Liu, Xue Lin, Sijia Liu

**Abstracts**: It has been well recognized that neural network based image classifiers are easily fooled by images with tiny perturbations crafted by an adversary. There has been a vast volume of research to generate and defend such adversarial attacks. However, the following problem is left unexplored: How to reverse-engineer adversarial perturbations from an adversarial image? This leads to a new adversarial learning paradigm--Reverse Engineering of Deceptions (RED). If successful, RED allows us to estimate adversarial perturbations and recover the original images. However, carefully crafted, tiny adversarial perturbations are difficult to recover by optimizing a unilateral RED objective. For example, the pure image denoising method may overfit to minimizing the reconstruction error but hardly preserve the classification properties of the true adversarial perturbations. To tackle this challenge, we formalize the RED problem and identify a set of principles crucial to the RED approach design. Particularly, we find that prediction alignment and proper data augmentation (in terms of spatial transformations) are two criteria to achieve a generalizable RED approach. By integrating these RED principles with image denoising, we propose a new Class-Discriminative Denoising based RED framework, termed CDD-RED. Extensive experiments demonstrate the effectiveness of CDD-RED under different evaluation metrics (ranging from the pixel-level, prediction-level to the attribution-level alignment) and a variety of attack generation methods (e.g., FGSM, PGD, CW, AutoAttack, and adaptive attacks).

摘要: 众所周知，基于神经网络的图像分类器很容易被对手制作的带有微小扰动的图像所愚弄。已经有大量的研究来产生和防御这种对抗性攻击。然而，以下问题仍未得到探索：如何从对抗性图像中逆向设计对抗性扰动？这导致了一种新的对抗性学习范式--欺骗的逆向工程(RED)。如果成功，RED允许我们估计敌方干扰并恢复原始图像。然而，精心设计的微小对抗性干扰很难通过优化单边红色目标来恢复。例如，纯图像去噪方法可能过于适合最小化重建误差，但很难保持真实对抗性扰动的分类性质。为了应对这一挑战，我们将RED问题形式化，并确定一组对RED方法设计至关重要的原则。特别是，我们发现预测对齐和适当的数据增强(在空间变换方面)是实现可推广的RED方法的两个标准。通过将这些RED原理与图像去噪相结合，我们提出了一种新的基于类别区分的RED去噪框架，称为CDD-RED。大量的实验证明了CDD-RED在不同的评估指标(从像素级、预测级到属性级对齐)和各种攻击生成方法(如FGSM、PGD、CW、AutoAttack和自适应攻击)下的有效性。



## **11. PiDAn: A Coherence Optimization Approach for Backdoor Attack Detection and Mitigation in Deep Neural Networks**

PIDAN：一种基于一致性优化的深层神经网络后门攻击检测与消除方法 cs.LG

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.09289v2)

**Authors**: Yue Wang, Wenqing Li, Esha Sarkar, Muhammad Shafique, Michail Maniatakos, Saif Eddin Jabari

**Abstracts**: Backdoor attacks impose a new threat in Deep Neural Networks (DNNs), where a backdoor is inserted into the neural network by poisoning the training dataset, misclassifying inputs that contain the adversary trigger. The major challenge for defending against these attacks is that only the attacker knows the secret trigger and the target class. The problem is further exacerbated by the recent introduction of "Hidden Triggers", where the triggers are carefully fused into the input, bypassing detection by human inspection and causing backdoor identification through anomaly detection to fail. To defend against such imperceptible attacks, in this work we systematically analyze how representations, i.e., the set of neuron activations for a given DNN when using the training data as inputs, are affected by backdoor attacks. We propose PiDAn, an algorithm based on coherence optimization purifying the poisoned data. Our analysis shows that representations of poisoned data and authentic data in the target class are still embedded in different linear subspaces, which implies that they show different coherence with some latent spaces. Based on this observation, the proposed PiDAn algorithm learns a sample-wise weight vector to maximize the projected coherence of weighted samples, where we demonstrate that the learned weight vector has a natural "grouping effect" and is distinguishable between authentic data and poisoned data. This enables the systematic detection and mitigation of backdoor attacks. Based on our theoretical analysis and experimental results, we demonstrate the effectiveness of PiDAn in defending against backdoor attacks that use different settings of poisoned samples on GTSRB and ILSVRC2012 datasets. Our PiDAn algorithm can detect more than 90% infected classes and identify 95% poisoned samples.

摘要: 后门攻击给深度神经网络(DNNS)带来了新的威胁，通过毒化训练数据集，错误分类包含对手触发器的输入，将后门插入到神经网络中。防御这些攻击的主要挑战是只有攻击者知道秘密触发器和目标类别。最近引入的“隐藏触发器”进一步加剧了这一问题，即触发器被小心地融合到输入中，绕过人工检查的检测，并导致通过异常检测进行的后门识别失败。为了防御这种不可察觉的攻击，我们系统地分析了后门攻击如何影响表示，即使用训练数据作为输入时给定DNN的神经元激活集。提出了一种基于相干优化的毒化数据净化算法PIDAN。我们的分析表明，有毒数据和真实数据在目标类中的表示仍然嵌入不同的线性子空间，这意味着它们与一些潜在空间表现出不同的一致性。基于这一观察，提出的PIDAN算法学习样本权重向量来最大化加权样本的投影一致性，其中我们证明了学习的权重向量具有自然的分组效应，并且可以区分真实数据和有毒数据。这使得能够系统地检测和缓解后门攻击。在理论分析和实验结果的基础上，我们在GTSRB和ILSVRC2012数据集上验证了PIDAN对使用不同设置的有毒样本的后门攻击的有效性。我们的Pidan算法可以检测到90%以上的感染类别，并识别出95%的中毒样本。



## **12. A Survey of Robust Adversarial Training in Pattern Recognition: Fundamental, Theory, and Methodologies**

模式识别中稳健的对抗性训练：基础、理论和方法综述 cs.CV

**SubmitDate**: 2022-03-26    [paper-pdf](http://arxiv.org/pdf/2203.14046v1)

**Authors**: Zhuang Qian, Kaizhu Huang, Qiu-Feng Wang, Xu-Yao Zhang

**Abstracts**: In the last a few decades, deep neural networks have achieved remarkable success in machine learning, computer vision, and pattern recognition. Recent studies however show that neural networks (both shallow and deep) may be easily fooled by certain imperceptibly perturbed input samples called adversarial examples. Such security vulnerability has resulted in a large body of research in recent years because real-world threats could be introduced due to vast applications of neural networks. To address the robustness issue to adversarial examples particularly in pattern recognition, robust adversarial training has become one mainstream. Various ideas, methods, and applications have boomed in the field. Yet, a deep understanding of adversarial training including characteristics, interpretations, theories, and connections among different models has still remained elusive. In this paper, we present a comprehensive survey trying to offer a systematic and structured investigation on robust adversarial training in pattern recognition. We start with fundamentals including definition, notations, and properties of adversarial examples. We then introduce a unified theoretical framework for defending against adversarial samples - robust adversarial training with visualizations and interpretations on why adversarial training can lead to model robustness. Connections will be also established between adversarial training and other traditional learning theories. After that, we summarize, review, and discuss various methodologies with adversarial attack and defense/training algorithms in a structured way. Finally, we present analysis, outlook, and remarks of adversarial training.

摘要: 在过去的几十年里，深度神经网络在机器学习、计算机视觉和模式识别方面取得了显著的成功。然而，最近的研究表明，神经网络(无论是浅层的还是深层的)可能很容易被某些被称为对抗性例子的潜意识扰动的输入样本所愚弄。近年来，这种安全漏洞导致了大量的研究，因为神经网络的广泛应用可能会带来现实世界的威胁。为了解决对抗实例的稳健性问题，特别是在模式识别中，稳健的对抗训练已经成为一种主流。各种思想、方法和应用在该领域蓬勃发展。然而，对对抗性训练的深入理解，包括特征、解释、理论以及不同模式之间的联系，仍然是难以捉摸的。在这篇文章中，我们提供了一个全面的调查，试图提供一个系统的和结构化的研究在模式识别中的稳健对手训练。我们从基础知识开始，包括对抗性例子的定义、符号和性质。然后，我们介绍了一个统一的理论框架来防御对抗样本-稳健的对抗训练，可视化和解释为什么对抗训练可以导致模型稳健性。对抗性训练和其他传统学习理论之间也将建立联系。然后，我们以结构化的方式总结、回顾和讨论了各种对抗性攻击和防御/训练算法的方法。最后，我们对对抗性训练进行了分析、展望和评论。



## **13. Canary Extraction in Natural Language Understanding Models**

自然语言理解模型中的金丝雀提取 cs.CL

Accepted to ACL 2022, Main Conference

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13920v1)

**Authors**: Rahil Parikh, Christophe Dupuy, Rahul Gupta

**Abstracts**: Natural Language Understanding (NLU) models can be trained on sensitive information such as phone numbers, zip-codes etc. Recent literature has focused on Model Inversion Attacks (ModIvA) that can extract training data from model parameters. In this work, we present a version of such an attack by extracting canaries inserted in NLU training data. In the attack, an adversary with open-box access to the model reconstructs the canaries contained in the model's training set. We evaluate our approach by performing text completion on canaries and demonstrate that by using the prefix (non-sensitive) tokens of the canary, we can generate the full canary. As an example, our attack is able to reconstruct a four digit code in the training dataset of the NLU model with a probability of 0.5 in its best configuration. As countermeasures, we identify several defense mechanisms that, when combined, effectively eliminate the risk of ModIvA in our experiments.

摘要: 自然语言理解(NLU)模型可以针对电话号码、邮政编码等敏感信息进行训练。最近的文献集中在模型反转攻击(MODIVA)上，它可以从模型参数中提取训练数据。在这项工作中，我们通过提取插入到NLU训练数据中的金丝雀来呈现这种攻击的一个版本。在攻击中，拥有模型开箱访问权限的对手重新构建了模型训练集中包含的金丝雀。我们通过对金丝雀执行文本补全来评估我们的方法，并演示了通过使用金丝雀的前缀(非敏感)标记，我们可以生成完整的金丝雀。例如，我们的攻击能够在NLU模型的训练数据集中以0.5的概率在其最佳配置下重建四位数代码。作为对策，我们确定了几种防御机制，当它们结合在一起时，在我们的实验中有效地消除了MODIVA的风险。



## **14. Improving robustness of jet tagging algorithms with adversarial training**

利用对抗性训练提高JET标签算法的稳健性 physics.data-an

14 pages, 11 figures, 2 tables

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13890v1)

**Authors**: Annika Stein, Xavier Coubez, Spandan Mondal, Andrzej Novak, Alexander Schmidt

**Abstracts**: Deep learning is a standard tool in the field of high-energy physics, facilitating considerable sensitivity enhancements for numerous analysis strategies. In particular, in identification of physics objects, such as jet flavor tagging, complex neural network architectures play a major role. However, these methods are reliant on accurate simulations. Mismodeling can lead to non-negligible differences in performance in data that need to be measured and calibrated against. We investigate the classifier response to input data with injected mismodelings and probe the vulnerability of flavor tagging algorithms via application of adversarial attacks. Subsequently, we present an adversarial training strategy that mitigates the impact of such simulated attacks and improves the classifier robustness. We examine the relationship between performance and vulnerability and show that this method constitutes a promising approach to reduce the vulnerability to poor modeling.

摘要: 深度学习是高能物理领域的标准工具，可大大提高许多分析策略的灵敏度。特别是，在喷气香精等物理对象的识别中，复杂的神经网络结构扮演着重要的角色。然而，这些方法依赖于准确的模拟。错误的建模可能会导致需要测量和校准的数据的性能出现不可忽略的差异。我们研究了分类器对带有注入误建模的输入数据的响应，并通过应用对抗性攻击来探索味道标注算法的脆弱性。随后，我们提出了一种对抗性训练策略，减轻了这类模拟攻击的影响，提高了分类器的稳健性。我们研究了性能和脆弱性之间的关系，并表明该方法是一种很有前途的方法，可以降低因建模不当而造成的脆弱性。



## **15. Origins of Low-dimensional Adversarial Perturbations**

低维对抗性扰动的起源 stat.ML

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13779v1)

**Authors**: Elvis Dohmatob, Chuan Guo, Morgane Goibert

**Abstracts**: In this note, we initiate a rigorous study of the phenomenon of low-dimensional adversarial perturbations in classification. These are adversarial perturbations wherein, unlike the classical setting, the attacker's search is limited to a low-dimensional subspace of the feature space. The goal is to fool the classifier into flipping its decision on a nonzero fraction of inputs from a designated class, upon the addition of perturbations from a subspace chosen by the attacker and fixed once and for all. It is desirable that the dimension $k$ of the subspace be much smaller than the dimension $d$ of the feature space, while the norm of the perturbations should be negligible compared to the norm of a typical data point. In this work, we consider binary classification models under very general regularity conditions, which are verified by certain feedforward neural networks (e.g., with sufficiently smooth, or else ReLU activation function), and compute analytical lower-bounds for the fooling rate of any subspace. These bounds explicitly highlight the dependence that the fooling rate has on the margin of the model (i.e., the ratio of the output to its $L_2$-norm of its gradient at a test point), and on the alignment of the given subspace with the gradients of the model w.r.t. inputs. Our results provide a theoretical explanation for the recent success of heuristic methods for efficiently generating low-dimensional adversarial perturbations. Moreover, our theoretical results are confirmed by experiments.

摘要: 在这篇笔记中，我们开始了对分类中低维对抗性扰动现象的严格研究。这些是对抗性扰动，其中，与经典设置不同，攻击者的搜索被限制在特征空间的低维子空间。其目的是愚弄分类器，在添加来自攻击者选择的并一劳永逸地修复的子空间的扰动时，翻转其对指定类别输入的非零分数的决定。希望子空间的维度$k$比特征空间的维度$d$小得多，而与典型数据点的范数相比，扰动的范数应该是可以忽略的。在这项工作中，我们考虑了在非常一般的正则性条件下的二分类模型，并用某些前向神经网络(例如，具有足够光滑的激活函数)进行了验证，并计算了任意子空间的愚弄率的解析下界。这些界明确地突出了愚弄率对模型边际(即，输出与其在测试点的梯度的$L_2$-范数的比率)以及给定子空间与模型的梯度对齐的依赖性。投入。我们的结果为启发式方法最近成功地产生低维对抗性扰动提供了理论上的解释。此外，我们的理论结果也得到了实验的证实。



## **16. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

理解和提高弗兰克-沃尔夫对抗性训练的效率 cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR  2022. Preliminary version ICML 2021 Adversarial Machine Learning Workshop.  Code: https://github.com/TheoT1/FW-AT-Adapt

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2012.12368v5)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal distortion, while standard networks have lower distortion. It is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and closes the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

摘要: 深度神经网络很容易被被称为对抗性攻击的小扰动所愚弄。对抗性训练(AT)是一种近似地解决稳健优化问题以最小化最坏情况损失的技术，被广泛认为是最有效的防御方法。由于在AT过程中生成强对抗性样本需要很高的计算时间，因此提出了一种单步方法来减少训练时间。然而，这些方法在训练过程中存在灾难性的过拟合问题，其中对抗性精度会下降，虽然已经提出了改进，但它们增加了训练时间，并且鲁棒性远远低于多步AT。我们开发了一个基于FW优化的对抗性训练(FW-AT)的理论框架，该框架揭示了损失情况与$\ell_inty$FW攻击的$\ell_2$失真之间的几何关系。我们分析表明，FW攻击的高失真等价于攻击路径上的小梯度变化。在不同深度神经网络结构上的实验结果表明，对健壮模型的攻击可以获得接近最大的失真，而标准网络的失真较小。实验表明，灾难性过拟合与FW攻击的低失真密切相关。这种数学透明度将FW与投影渐变下降(PGD)优化区分开来。为了证明我们的理论框架的有效性，我们开发了一种新的对抗性训练算法FW-AT-Adapt，它使用一个简单的失真度量来调整训练过程中的攻击步数，以在不影响健壮性的情况下提高效率。FW-AT-Adapt提供与单步快速AT方法相当的训练时间，并在白盒和黑盒设置下以最小的对手精度损失缩小了快速AT方法和多步PGD-AT方法之间的差距。



## **17. Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

给我注意：点积注意被认为对对手补丁的健壮性有害 cs.CV

to be published in IEEE/CVF Conference on Computer Vision and Pattern  Recognition 2022, CVPR22

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13639v1)

**Authors**: Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen

**Abstracts**: Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.

摘要: 基于注意力的神经结构，如视觉转换器，正在给图像识别带来革命性的变化。它们的主要好处是，注意力允许对场景的所有部分进行联合推理。在这篇文章中，我们展示了在面对敌意补丁攻击时，(按比例)点积注意力的全局推理如何成为主要漏洞的来源。我们提供了对该漏洞的理论理解，并将其与对手将所有查询的注意力误导到对手补丁控制下的单个密钥令牌的能力相关联。我们提出了新的对抗性目标，用于制作明确针对此漏洞的对抗性补丁。我们展示了所提出的补丁攻击在流行的图像分类(VITS和DeITS)和目标检测模型(DETR)上的有效性。我们发现，敌意补丁占输入的0.5%可以导致VIT在ImageNet上的稳健准确率低至0%，并将DETR在MS CoCo上的MAP降低到3%以下。



## **18. Adversarial Bone Length Attack on Action Recognition**

动作识别的对抗性骨长攻击 cs.CV

12 pages, 8 figures, accepted to AAAI2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2109.05830v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.

摘要: 基于骨架的动作识别模型最近被证明容易受到对手攻击。与对图像的敌意攻击相比，对骨骼的扰动通常被限制在大约每帧100个维度的较低维度。这种较低维度的设置使产生难以察觉的微扰变得更加困难。现有的攻击通过利用骨骼运动的时间结构来解决这个问题，从而使扰动维度增加到数千。在本文中，我们证明了对抗性攻击可以在基于骨架的动作识别模型上执行，即使在显著低维的环境中也不需要任何时间处理。具体地说，我们将扰动限制在骨骼的长度上，这使得对手只能操纵大约30个有效维度。我们在NTU、RGB+D和HDM05数据集上进行了实验，结果表明，该攻击通过微小的扰动成功地欺骗了模型，有时成功率超过90%。此外，我们还发现了一个有趣的现象：在我们的低维环境下，带有骨长攻击的对抗性训练与数据增强具有相似的性质，它不仅提高了对抗性的健壮性，而且提高了对原始数据的分类精度。这是一个有趣的反例，说明了对抗性稳健性和清晰准确性之间的权衡，这在高维体制下对抗性训练的研究中得到了广泛的观察。



## **19. Improving Adversarial Transferability with Spatial Momentum**

利用空间动量提高对抗性转移能力 cs.CV

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13479v1)

**Authors**: Guoqiu Wang, Xingxing Wei, Huanqian Yan

**Abstracts**: Deep Neural Networks (DNN) are vulnerable to adversarial examples. Although many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, they usually show poor transferability when attacking other DNN models. Momentum-based attack (MI-FGSM) is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, in this paper, we propose a novel method named Spatial Momentum Iterative FGSM Attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context gradient information from different regions within the image. SMI-FGSM is then integrated with MI-FGSM to simultaneously stabilize the gradients' update direction from both the temporal and spatial domain. The final method is called SM$^2$I-FGSM. Extensive experiments are conducted on the ImageNet dataset and results show that SM$^2$I-FGSM indeed further enhances the transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art methods by a large margin.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。虽然许多对抗性攻击方法在白盒设置下取得了令人满意的攻击成功率，但在攻击其他DNN模型时往往表现出较差的可移植性。基于动量的攻击(MI-FGSM)是提高可传递性的一种有效方法。该算法将动量项融入到迭代过程中，通过增加每个像素的梯度时间相关性来稳定更新方向。我们认为，仅有这种时间动量是不够的，图像中来自空间域的梯度，即来自以目标像素为中心的上下文像素的梯度，对于稳定也是重要的。为此，本文提出了一种新的方法--空间动量迭代FGSM攻击(SMI-FGSM)，通过考虑图像内不同区域的上下文梯度信息，引入了从时域到空域的动量积累机制。然后将SMI-FGSM与MI-FGSM相结合，从时间域和空间域同时稳定梯度的更新方向。最后一种方法称为SM$^2$I-FGSM。在ImageNet数据集上进行了大量的实验，结果表明SM$^2$I-FGSM确实进一步提高了可移植性。它在多个主流的无防御和有防御的模型上实现了最好的可转移性成功率，远远超过了最先进的方法。



## **20. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的定界训练数据重构 cs.LG

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2201.12383v2)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 在ML中，差异隐私被广泛接受为防止数据泄露的事实上的方法，传统观点认为，它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。本文首先在形式化威胁模型下给出了DP机制抵抗训练数据重构攻击的语义保证。我们发现，两种不同的隐私记账方法--Renyi Differential Privacy和Fisher信息泄漏--都提供了对数据重构攻击的强大语义保护。



## **21. A Perturbation Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

一种用于评估光流稳健性的扰动约束对抗性攻击 cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.13214v1)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while analyzing their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods rather focus on real-world attacking scenarios than on a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. More precisely, PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments not only demonstrate PCFA's applicability in white- and black-box settings, but also show that it finds stronger adversarial samples for optical flow than previous attacking frameworks. Moreover, based on these strong samples, we provide the first common ranking of optical flow methods in the literature considering both prediction quality and adversarial robustness, indicating that high quality methods are not necessarily robust. Our source code will be publicly available.

摘要: 最近的光流方法几乎完全是从精度的角度来判断的，而对其稳健性的分析往往被忽视。尽管对抗性攻击为执行这种分析提供了一个有用的工具，但目前对光流方法的攻击更多地集中在真实世界的攻击场景中，而不是最坏情况下的健壮性评估。因此，在这项工作中，我们提出了一种新的对抗性攻击-扰动约束流攻击(PCFA)-作为现实世界的攻击，强调破坏性而不是适用性。更准确地说，PCFA是一种全局攻击，它优化对抗性扰动，将预测流向指定的目标流移动，同时将扰动的L2范数保持在选定的界限以下。我们的实验不仅证明了PCFA在白盒和黑盒环境中的适用性，而且表明它比以前的攻击框架发现了更强的光流对抗性样本。此外，基于这些强样本，我们提供了文献中第一个同时考虑预测质量和对抗稳健性的光流方法的常见排名，表明高质量的方法不一定是健壮的。我们的源代码将向公众开放。



## **22. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

基于频率驱动的语义相似度潜伏攻击 cs.CV

CVPR 2022 conference (accepted), 18 pages, 17 figure

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.05151v4)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.

摘要: 目前的对抗性攻击研究揭示了基于学习的分类器对精心设计的扰动的脆弱性。然而，大多数现有的攻击方法在跨数据集泛化方面存在固有的局限性，因为它们依赖于具有封闭类别集的分类层。此外，这些方法产生的扰动可能出现在人类视觉系统(HVS)容易察觉的区域。针对上述问题，我们提出了一种攻击特征表示语义相似度的新算法。通过这种方式，我们能够愚弄分类器，而不会将攻击限制在特定的数据集。对于不可感知性，我们引入了低频约束来限制高频分量内的扰动，以确保对抗性示例与原始示例之间的感知相似性。在三个数据集(CIFAR-10、CIFAR-100和ImageNet-1K)和三个公共在线平台上的广泛实验表明，我们的攻击可以产生跨体系结构和数据集的误导性和可转移的敌意示例。此外，可视化结果和量化性能(在四个不同的度量方面)表明，所提出的算法比现有的方法产生更多的不可察觉的扰动。代码可在上获得。



## **23. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

三维点云分类的潜移式攻防 cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2111.10990v2)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.

摘要: 虽然近年来在二维图像领域的攻击和防御方面做了很多努力，但很少有方法研究三维模型的脆弱性。现有的3D攻击者通常对点云进行逐点摄动，产生变形的结构或离群点，这很容易被人类感知到。此外，他们的对抗性例子是在白盒环境下产生的，当转移到攻击远程黑盒模型时，白盒模型的成功率经常很低。本文从两个新的具有挑战性的角度对三维点云攻击进行了研究，提出了一种新的不可感知性转移攻击(ITA)：1)不可感知性：我们约束每个点沿其邻域曲面的法矢的扰动方向，从而生成具有相似几何性质的示例，从而增强了不可感知性。2)可转移性：我们开发了一个对抗性转换模型来产生最有害的扭曲，并强制执行对抗性例子来抵抗它，从而提高了它们到未知黑盒模型的可转移性。此外，我们建议通过学习更具区别性的点云表示来训练更健壮的黑盒3D模型来防御此类ITA攻击。广泛的评估表明，我们的ITA攻击比最先进的攻击更具隐蔽性和可移动性，验证了我们防御战略的优越性。



## **24. Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation**

提高模型稳健性的对抗性训练？预测和解释都要看 cs.CL

AAAI 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12709v1)

**Authors**: Hanjie Chen, Yangfeng Ji

**Abstracts**: Neural language models show vulnerability to adversarial examples which are semantically similar to their original counterparts with a few words replaced by their synonyms. A common way to improve model robustness is adversarial training which follows two steps-collecting adversarial examples by attacking a target model, and fine-tuning the model on the augmented dataset with these adversarial examples. The objective of traditional adversarial training is to make a model produce the same correct predictions on an original/adversarial example pair. However, the consistency between model decision-makings on two similar texts is ignored. We argue that a robust model should behave consistently on original/adversarial example pairs, that is making the same predictions (what) based on the same reasons (how) which can be reflected by consistent interpretations. In this work, we propose a novel feature-level adversarial training method named FLAT. FLAT aims at improving model robustness in terms of both predictions and interpretations. FLAT incorporates variational word masks in neural networks to learn global word importance and play as a bottleneck teaching the model to make predictions based on important words. FLAT explicitly shoots at the vulnerability problem caused by the mismatch between model understandings on the replaced words and their synonyms in original/adversarial example pairs by regularizing the corresponding global word importance scores. Experiments show the effectiveness of FLAT in improving the robustness with respect to both predictions and interpretations of four neural network models (LSTM, CNN, BERT, and DeBERTa) to two adversarial attacks on four text classification tasks. The models trained via FLAT also show better robustness than baseline models on unforeseen adversarial examples across different attacks.

摘要: 神经语言模型显示出对敌意例子的脆弱性，这些例子在语义上与原始对应的例子相似，但有几个单词被它们的同义词取代。提高模型稳健性的一种常见方法是对抗性训练，它遵循两个步骤-通过攻击目标模型来收集对抗性实例，并使用这些对抗性实例在扩充的数据集上对模型进行微调。传统对抗性训练的目标是使模型在原始/对抗性样本对上产生相同的正确预测。然而，两个相似文本上的模型决策之间的一致性被忽略了。我们认为，一个稳健的模型应该在原始/对抗性示例对上表现出一致的行为，即基于相同的原因(如何)做出相同的预测(What)，这些预测可以被一致的解释反映出来。在这项工作中，我们提出了一种新的特征级别的对抗性训练方法--Flat。Flat旨在提高模型在预测和解释方面的稳健性。Flat在神经网络中引入了变量词掩码来学习全局词的重要性，并作为瓶颈来教导模型基于重要词进行预测。Flat通过规则化相应的全局单词重要性分数，明确地解决了原始/对抗性示例对中被替换单词的模型理解与其同义词之间的不匹配所导致的脆弱性问题。实验表明，对于四种文本分类任务上的两种敌意攻击，Flat能有效地提高四种神经网络模型(LSTM、CNN、BERT和DeBERTa)的预测和解释的鲁棒性。通过Flat训练的模型在不同攻击的不可预见的对手例子上也表现出比基线模型更好的稳健性。



## **25. Enhancing Classifier Conservativeness and Robustness by Polynomiality**

利用多项式增强分类器的保守性和稳健性 cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12693v1)

**Authors**: Ziqi Wang, Marco Loog

**Abstracts**: We illustrate the detrimental effect, such as overconfident decisions, that exponential behavior can have in methods like classical LDA and logistic regression. We then show how polynomiality can remedy the situation. This, among others, leads purposefully to random-level performance in the tails, away from the bulk of the training data. A directly related, simple, yet important technical novelty we subsequently present is softRmax: a reasoned alternative to the standard softmax function employed in contemporary (deep) neural networks. It is derived through linking the standard softmax to Gaussian class-conditional models, as employed in LDA, and replacing those by a polynomial alternative. We show that two aspects of softRmax, conservativeness and inherent gradient regularization, lead to robustness against adversarial attacks without gradient obfuscation.

摘要: 我们举例说明了指数行为在经典的LDA和Logistic回归等方法中可能产生的有害影响，如过度自信的决策。然后，我们展示了多项式如何弥补这种情况。这其中，有目的地导致了尾部的随机水平的性能，而不是大部分的训练数据。我们随后提出的一个直接相关、简单但重要的技术创新是SoftRmax：当代(深度)神经网络中使用的标准Softmax函数的合理替代方案。它是通过将标准Softmax与LDA中使用的高斯类条件模型联系起来，并用多项式替代来得到的。我们证明了软Rmax的两个方面，保守性和固有的梯度正则化，导致在没有梯度混淆的情况下对对手攻击具有健壮性。



## **26. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

基于可解释性的点云神经网络单点攻击 cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2110.04158v3)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.

摘要: 随着神经网络在点云领域的提出，深度学习开始在三维物体识别领域闪耀光芒，同时研究人员对利用对抗性攻击来研究点云网络的可靠性也越来越感兴趣。然而，现有的研究大多旨在欺骗人类或防御算法，而少数针对模型本身操作原理的研究在临界点选择方面仍然存在缺陷。在这项工作中，我们提出了两种对抗方法：单点攻击(OPA)和关键遍历攻击(CTA)，它们结合了可解释性技术，旨在探索点云网络的内在工作原理及其对临界点扰动的敏感性。我们的结果表明，只要从输入实例中移动一个点，流行的点云网络就可以被欺骗，成功率接近100美元。此外，我们还展示了不同的点属性分布对点云网络对抗健壮性的影响。最后，我们讨论了我们的方法如何促进点云网络的可解释性研究。据我们所知，这是第一个基于点云的对抗性解释方法。我们的代码可以在https://github.com/Explain3D/Exp-One-Point-Atk-PC.上找到



## **27. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

对抗性后门防御微调：将后门攻击与对抗性攻击联系起来 cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2202.06312v2)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.

摘要: 众所周知，深度神经网络(DNNs)既容易受到后门攻击，也容易受到敌意攻击。在文献中，这两类攻击由于分别属于训练时间攻击和推理时间攻击，通常被视为不同的问题并分别解决。然而，在本文中，我们发现了它们之间一个有趣的联系：对于一个植入后门的模型，我们观察到其敌对示例与其触发样本具有相似的行为，即两者都激活了相同的DNN神经元子集。这表明在模型中植入后门将显著影响模型的对抗性示例。在此基础上，我们设计了一种新的对抗性微调(AFT)算法来防御后门攻击。我们的实验表明，对于5种最先进的后门攻击，我们的AFT可以有效地清除后门触发，而在干净的样本上没有明显的性能下降，并且显著优于现有的防御方法。



## **28. Input-specific Attention Subnetworks for Adversarial Detection**

用于敌意检测的输入特定关注子网络 cs.CL

Accepted at Findings of ACL 2022, 14 pages, 6 Tables and 9 Figures

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12298v1)

**Authors**: Emil Biju, Anirudh Sriram, Pratyush Kumar, Mitesh M Khapra

**Abstracts**: Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. The resultant detector significantly improves (by over 7.5%) the state-of-the-art adversarial detection accuracy for the BERT encoder on 10 NLU datasets with 11 different adversarial attack types. We also demonstrate that our method (a) is more accurate for larger models which are likely to have more spurious correlations and thus vulnerable to adversarial attack, and (b) performs well even with modest training sets of adversarial examples.

摘要: 自我注意头部是变压器模型的特征，在可解释性和剪枝方面已经得到了很好的研究。在这项工作中，我们展示了注意力头部的一种完全不同的用途，即用于敌意检测。具体地说，我们提出了一种构造输入特定关注子网络(IAS)的方法，从中提取三个特征来区分真实输入和敌意输入。在包含11种不同敌意攻击类型的10个NLU数据集上，所得到的检测器显著地提高了(超过7.5%)BERT编码器的最新敌意检测精度。我们还证明了我们的方法(A)对于更大的模型更准确，这些模型可能具有更多的虚假相关性，因此容易受到对抗性攻击，并且(B)即使在适度的对抗性例子训练集的情况下也表现得很好。



## **29. Integrity Fingerprinting of DNN with Double Black-box Design and Verification**

基于双黑盒设计和验证的DNN完整性指纹分析 cs.CR

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.10902v2)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Surya Nepal, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.

摘要: 支持云的机器学习即服务(MLaaS)显示出巨大的潜力，可以改变深度学习模型的开发和部署方式。尽管如此，使用此类服务仍存在潜在风险，因为恶意方可能会对其进行修改以达到不利的结果。因此，模型所有者、服务提供商和最终用户必须验证部署的模型是否未被篡改。这样的验证需要公开的可验证性(即，指纹模式可供各方使用，包括对手)，并需要通过API对部署的模型进行黑盒访问。然而，现有的水印和指纹方法需要白盒知识(如梯度)来设计指纹，并且只支持私密可验证性，即由诚实的一方进行验证。在本文中，我们描述了一种实用的水印技术，该技术能够在指纹设计中提供黑盒知识，并在验证过程中提供黑盒查询。该服务通过公开验证来确保基于云的服务的完整性(即指纹模式可供各方使用，包括对手)。如果对手操纵了一个模型，这将导致决策边界的转变。因此，双黑水印的基本原理是，模型的决策边界可以作为水印的固有指纹。我们的方法通过生成有限数量的包络样本指纹来捕获决策边界，这些样本指纹是围绕模型决策边界的一组自然转换和扩充的输入，以捕获模型的固有指纹。我们针对各种模型完整性攻击和模型压缩攻击对我们的水印方法进行了评估。



## **30. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

阴影可能是危险的：自然现象对物理世界的隐秘而有效的对抗性攻击 cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.03818v3)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.

摘要: 估计对抗性例子的风险水平对于在现实世界中安全地部署机器学习模型是至关重要的。物理世界攻击的一种流行方法是采用“粘贴”策略，但该策略受到一些限制，包括难以访问目标或以有效颜色打印。最近出现了一种新型的非侵入性攻击，它试图通过激光和投影仪等基于光学的工具对目标进行摄动。然而，添加的光学图案是人造的，但不是自然的。因此，它们仍然是引人注目和引人注目的，很容易被人类注意到。本文研究了一种新的光学对抗实例，其中的扰动是由一种非常常见的自然现象阴影产生的，以实现黑箱环境下自然主义的隐身物理世界对抗攻击。我们广泛评估了这种新攻击在模拟和真实环境中的有效性。在交通标志识别上的实验结果表明，该算法能够有效地生成对抗性样本，在LISA和GTSRB测试集上的准确率分别达到98.23%和90.47%，而在真实场景中，95%以上的时间都能连续误导移动的摄像机。我们还讨论了这种攻击的局限性和防御机制。



## **31. Online Adversarial Attacks**

在线对抗性攻击 cs.LG

ICLR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2103.02014v4)

**Authors**: Andjela Mladenovic, Avishek Joey Bose, Hugo Berard, William L. Hamilton, Simon Lacoste-Julien, Pascal Vincent, Gauthier Gidel

**Abstracts**: Adversarial attacks expose important vulnerabilities of deep learning models, yet little attention has been paid to settings where data arrives as a stream. In this paper, we formalize the online adversarial attack problem, emphasizing two key elements found in real-world use-cases: attackers must operate under partial knowledge of the target model, and the decisions made by the attacker are irrevocable since they operate on a transient data stream. We first rigorously analyze a deterministic variant of the online threat model by drawing parallels to the well-studied $k$-secretary problem in theoretical computer science and propose Virtual+, a simple yet practical online algorithm. Our main theoretical result shows Virtual+ yields provably the best competitive ratio over all single-threshold algorithms for $k<5$ -- extending the previous analysis of the $k$-secretary problem. We also introduce the \textit{stochastic $k$-secretary} -- effectively reducing online blackbox transfer attacks to a $k$-secretary problem under noise -- and prove theoretical bounds on the performance of Virtual+ adapted to this setting. Finally, we complement our theoretical results by conducting experiments on MNIST, CIFAR-10, and Imagenet classifiers, revealing the necessity of online algorithms in achieving near-optimal performance and also the rich interplay between attack strategies and online attack selection, enabling simple strategies like FGSM to outperform stronger adversaries.

摘要: 对抗性攻击暴露了深度学习模型的重大漏洞，但很少有人关注数据以流的形式到达的环境。在本文中，我们形式化地描述了在线对抗攻击问题，强调了现实世界用例中的两个关键要素：攻击者必须在目标模型的部分知识下操作，并且攻击者所做的决定是不可撤销的，因为他们操作的是瞬时数据流。我们首先严格分析了在线威胁模型的一个确定性变体，将其与理论计算机科学中研究得很好的$k$秘书问题进行了类比，并提出了一种简单而实用的在线算法--虚拟+。我们的主要理论结果表明，在$k<5$的情况下，与所有单阈值算法相比，虚拟+可以产生最好的竞争比--扩展了之前对$k$-秘书问题的分析。我们还引入了文本{随机$k$-秘书}--有效地将在线黑盒传输攻击归结为噪声下的一个$k$-秘书问题--并证明了适用于这种设置的虚拟+性能的理论界限。最后，我们通过在MNIST、CIFAR-10和Imagenet分类器上进行实验来补充我们的理论结果，揭示了在线算法获得接近最佳性能的必要性，以及攻击策略和在线攻击选择之间的丰富相互作用，使FGSM等简单策略的性能优于更强大的对手。



## **32. NNReArch: A Tensor Program Scheduling Framework Against Neural Network Architecture Reverse Engineering**

NNReArch：一种面向神经网络结构逆向工程的张量程序调度框架 cs.CR

Accepted by FCCM 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.12046v1)

**Authors**: Yukui Luo, Shijin Duan, Cheng Gongye, Yunsi Fei, Xiaolin Xu

**Abstracts**: Architecture reverse engineering has become an emerging attack against deep neural network (DNN) implementations. Several prior works have utilized side-channel leakage to recover the model architecture while the target is executing on a hardware acceleration platform. In this work, we target an open-source deep-learning accelerator, Versatile Tensor Accelerator (VTA), and utilize electromagnetic (EM) side-channel leakage to comprehensively learn the association between DNN architecture configurations and EM emanations. We also consider the holistic system -- including the low-level tensor program code of the VTA accelerator on a Xilinx FPGA and explore the effect of such low-level configurations on the EM leakage. Our study demonstrates that both the optimization and configuration of tensor programs will affect the EM side-channel leakage.   Gaining knowledge of the association between the low-level tensor program and the EM emanations, we propose NNReArch, a lightweight tensor program scheduling framework against side-channel-based DNN model architecture reverse engineering. Specifically, NNReArch targets reshaping the EM traces of different DNN operators, through scheduling the tensor program execution of the DNN model so as to confuse the adversary. NNReArch is a comprehensive protection framework supporting two modes, a balanced mode that strikes a balance between the DNN model confidentiality and execution performance, and a secure mode where the most secure setting is chosen. We implement and evaluate the proposed framework on the open-source VTA with state-of-the-art DNN architectures. The experimental results demonstrate that NNReArch can efficiently enhance the model architecture security with a small performance overhead. In addition, the proposed obfuscation technique makes reverse engineering of the DNN architecture significantly harder.

摘要: 体系结构逆向工程已经成为对深度神经网络(DNN)实现的一种新兴攻击。已有的一些工作利用旁路泄漏来恢复目标在硬件加速平台上执行时的模型体系结构。在这项工作中，我们针对一个开源的深度学习加速器，通用张量加速器(VTA)，并利用电磁(EM)侧通道泄漏来全面了解DNN结构配置和EM发射之间的关联。我们还考虑了整个系统--包括Xilinx FPGA上的VTA加速器的低电平张量程序代码，并探索了这种低电平配置对电磁泄漏的影响。我们的研究表明，张量程序的优化和配置都会影响到EM侧沟道泄漏。通过获取底层张量程序与EM发射之间的关联，我们提出了一种轻量级张量程序调度框架NNReArch，该框架针对基于侧通道的DNN模型体系结构的逆向工程。具体地说，NNReArch通过调度DNN模型的张量程序执行来重塑不同DNN算子的EM轨迹，从而迷惑对手。NNReArch是一个全面的保护框架，支持两种模式，一种是在DNN模型机密性和执行性能之间取得平衡的平衡模式，另一种是选择最安全设置的安全模式。我们在采用最先进的DNN架构的开源VTA上实现了该框架，并对其进行了评估。实验结果表明，NNReArch能够以较小的性能开销有效地增强模型体系结构的安全性。此外，提出的模糊技术使DNN体系结构的逆向工程变得更加困难。



## **33. Shape-invariant 3D Adversarial Point Clouds**

形状不变的三维对抗性点云 cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.04041v2)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to constrain its perturbation with a simple loss or metric properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.

摘要: 对抗性和隐蔽性是对抗性扰动的两个基本但又相互冲突的特征。以前针对3D点云识别的敌意攻击经常因为其显著的点离群值而受到批评，因为它们只是在耗时的优化中涉及诸如全局距离损失这样的“隐含约束”，以限制生成的噪声。虽然点云是一种高度结构化的数据格式，但很难用简单的损失或度量来适当地约束它的扰动。在本文中，我们提出了一种新的点云敏感度图，以提高点扰动的效率和隐蔽性。这张地图揭示了点云识别模型在遇到形状不变的对抗性噪声时的脆弱性。这些噪波是沿着形状表面设计的，并带有“显式约束”，而不是额外的距离损失。具体地说，我们首先对点云输入的每个点应用可逆坐标变换，以减少一个点自由度并限制其在切平面上的运动。然后利用白盒模型得到的变换点云的梯度来计算最佳攻击方向。最后，我们给每个点分配一个非负的分数来构造敏感度图，这既有利于白盒对抗不可见性，也有利于提高黑盒查询效率。广泛的评估表明，该方法能够在各种点云识别模型上取得较好的性能，具有令人满意的对抗性和对不同点云防御设置的较强抵抗力。我们的代码请访问：https://github.com/shikiw/SI-Adv.



## **34. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

基于后向误差分析的联合学习半目标模型中毒攻击 cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11633v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.

摘要: 针对联邦学习(FL)的模型中毒攻击通过破坏边缘模型来侵入整个系统，导致机器学习模型故障。这种被破坏的模型被篡改，以执行对手所希望的行为。特别是，我们考虑了一种半目标的情况，其中源类是预先确定的，而目标类不是。其目的是使全局分类器对源类的数据进行错误分类。虽然已经采用了标签翻转等方法向FL注入有毒参数，但研究表明，它们的性能通常是类敏感的，随着所使用的目标类的不同而变化。通常，当转移到不同的目标类别时，攻击可能会变得不那么有效。为了克服这一挑战，我们提出了攻击距离感知攻击(ADA)，通过在特征空间中找到优化的目标类来增强中毒攻击。此外，我们研究了一种更具挑战性的情况，即对手对客户数据的先验知识有限。为了解决这一问题，ADA基于向后误差分析，从共享的模型参数中推导出潜在特征空间中不同类别之间的成对距离。我们通过在三种不同的图像分类任务中改变攻击频率的因素，对ADA进行了广泛的经验评估。结果，在最具挑战性的情况下，ADA成功地将攻击性能提高了1.8倍，攻击频率为0.01。



## **35. Exploring High-Order Structure for Robust Graph Structure Learning**

高阶结构在稳健图结构学习中的探索 cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11492v1)

**Authors**: Guangqian Yang, Yibing Zhan, Jinlong Li, Baosheng Yu, Liu Liu, Fengxiang He

**Abstracts**: Recent studies show that Graph Neural Networks (GNNs) are vulnerable to adversarial attack, i.e., an imperceptible structure perturbation can fool GNNs to make wrong predictions. Some researches explore specific properties of clean graphs such as the feature smoothness to defense the attack, but the analysis of it has not been well-studied. In this paper, we analyze the adversarial attack on graphs from the perspective of feature smoothness which further contributes to an efficient new adversarial defensive algorithm for GNNs. We discover that the effect of the high-order graph structure is a smoother filter for processing graph structures. Intuitively, the high-order graph structure denotes the path number between nodes, where larger number indicates closer connection, so it naturally contributes to defense the adversarial perturbation. Further, we propose a novel algorithm that incorporates the high-order structural information into the graph structure learning. We perform experiments on three popular benchmark datasets, Cora, Citeseer and Polblogs. Extensive experiments demonstrate the effectiveness of our method for defending against graph adversarial attacks.

摘要: 最近的研究表明，图神经网络(GNN)容易受到敌意攻击，即不可察觉的结构扰动可以欺骗GNN做出错误的预测。一些研究探索了干净图的一些特殊性质，如特征光滑性来防御攻击，但对它的分析还没有得到很好的研究。本文从特征光滑性的角度对图的对抗性攻击进行了分析，进而提出了一种新的高效的GNN对抗性防御算法。我们发现，高阶图结构的影响是处理图结构的更平滑的过滤器。直观地说，高阶图结构表示节点之间的路径数，其中越大表示连接越紧密，因此自然有助于防御对手的扰动。在此基础上，提出了一种将高阶结构信息融入到图结构学习中的新算法。我们在三个流行的基准数据集CORA、Citeseer和Polblog上进行了实验。大量的实验证明了该方法对图攻击的有效防御。



## **36. Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack**

让DeepFake变得更加虚假：通过痕迹移除攻击来逃避深度人脸伪造检测 cs.CV

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11433v1)

**Authors**: Chi Liu, Huajie Chen, Tianqing Zhu, Jun Zhang, Wanlei Zhou

**Abstracts**: DeepFakes are raising significant social concerns. Although various DeepFake detectors have been developed as forensic countermeasures, these detectors are still vulnerable to attacks. Recently, a few attacks, principally adversarial attacks, have succeeded in cloaking DeepFake images to evade detection. However, these attacks have typical detector-specific designs, which require prior knowledge about the detector, leading to poor transferability. Moreover, these attacks only consider simple security scenarios. Less is known about how effective they are in high-level scenarios where either the detectors or the attacker's knowledge varies. In this paper, we solve the above challenges with presenting a novel detector-agnostic trace removal attack for DeepFake anti-forensics. Instead of investigating the detector side, our attack looks into the original DeepFake creation pipeline, attempting to remove all detectable natural DeepFake traces to render the fake images more "authentic". To implement this attack, first, we perform a DeepFake trace discovery, identifying three discernible traces. Then a trace removal network (TR-Net) is proposed based on an adversarial learning framework involving one generator and multiple discriminators. Each discriminator is responsible for one individual trace representation to avoid cross-trace interference. These discriminators are arranged in parallel, which prompts the generator to remove various traces simultaneously. To evaluate the attack efficacy, we crafted heterogeneous security scenarios where the detectors were embedded with different levels of defense and the attackers' background knowledge of data varies. The experimental results show that the proposed attack can significantly compromise the detection accuracy of six state-of-the-art DeepFake detectors while causing only a negligible loss in visual quality to the original DeepFake samples.

摘要: DeepFake引起了重大的社会关注。虽然已经开发了各种DeepFake检测器作为取证对策，但这些检测器仍然容易受到攻击。最近，一些攻击，主要是对抗性攻击，成功地伪装DeepFake图像以逃避检测。然而，这些攻击具有典型的特定于检测器的设计，需要事先了解检测器，导致可移植性较差。此外，这些攻击只考虑简单的安全场景。对于它们在检测器或攻击者的知识各不相同的高级场景中的有效性，我们知之甚少。在本文中，我们提出了一种新的针对DeepFake反取证的与检测器无关的痕迹移除攻击，从而解决了上述挑战。我们的攻击不是调查探测器端，而是查看原始的DeepFake创建管道，试图删除所有可检测到的自然DeepFake痕迹，以使虚假图像更“真实”。要实现此攻击，首先，我们执行DeepFake跟踪发现，识别三个可识别的跟踪。然后，基于一个生成器和多个鉴别器的对抗性学习框架，提出了一种痕迹去除网络(TR-Net)。每个鉴别器负责一个单独的轨迹表示，以避免交叉轨迹干扰。这些鉴别器是并行排列的，这会提示发生器同时移除各种痕迹。为了评估攻击效能，我们精心设计了不同的安全场景，其中检测器嵌入了不同级别的防御，攻击者的数据背景知识各不相同。实验结果表明，该攻击可以显著降低现有的6种DeepFake检测器的检测精度，而对原始DeepFake样本的视觉质量损失可以忽略不计。



## **37. Subspace Adversarial Training**

子空间对抗训练 cs.LG

CVPR2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2111.12229v2)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to 0% during the training. In this paper, we approach this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand robust overfitting in multi-step AT. To control the growth of the gradient, we propose a new AT method, Subspace Adversarial Training (Sub-AT), which constrains AT in a carefully extracted subspace. It successfully resolves both kinds of overfitting and significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, further improving the robustness performance. As a result, we achieve state-of-the-art single-step AT performance. Without any regularization term, our single-step AT can reach over 51% robust accuracy against strong PGD-50 attack of radius 8/255 on CIFAR-10, reaching a competitive performance against standard multi-step PGD-10 AT with huge computational advantages. The code is released at https://github.com/nblt/Sub-AT.

摘要: 单步对抗性训练(AT)因其高效、健壮而受到广泛关注。然而，存在一个严重的灾难性过拟合问题，即在训练过程中，对投影梯度下降(PGD)攻击的鲁棒精度突然下降到0%。本文从一个新的优化角度来研究这一问题，首次揭示了每个样本快速增长的梯度与过拟合之间的密切联系，这也可以用来理解多步AT中的稳健过拟合。为了控制梯度的增长，我们提出了一种新的AT方法--子空间对抗训练(Sub-Space Adversative Trading，Sub-AT)，它将AT约束在一个仔细提取的子空间中。它成功地解决了这两种过拟合问题，并显著提高了鲁棒性。在子空间中，我们还允许单步AT具有更大的步长和更大的半径，进一步提高了算法的鲁棒性。因此，我们实现了最先进的单步AT性能。在没有任何正则项的情况下，我们的单步AT在抵抗CIFAR-10上半径为8/255的强PGD-50攻击时可以达到51%以上的鲁棒准确率，达到了与标准多步PGD-10 AT相当的性能，同时具有巨大的计算优势。该代码在https://github.com/nblt/Sub-AT.上发布



## **38. On The Robustness of Offensive Language Classifiers**

论攻击性语言量词的稳健性 cs.CL

9 pages, 2 figures, Accepted at ACL 2022

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11331v1)

**Authors**: Jonathan Rusert, Zubair Shafiq, Padmini Srinivasan

**Abstracts**: Social media platforms are deploying machine learning based offensive language classification systems to combat hateful, racist, and other forms of offensive speech at scale. However, despite their real-world deployment, we do not yet comprehensively understand the extent to which offensive language classifiers are robust against adversarial attacks. Prior work in this space is limited to studying robustness of offensive language classifiers against primitive attacks such as misspellings and extraneous spaces. To address this gap, we systematically analyze the robustness of state-of-the-art offensive language classifiers against more crafty adversarial attacks that leverage greedy- and attention-based word selection and context-aware embeddings for word replacement. Our results on multiple datasets show that these crafty adversarial attacks can degrade the accuracy of offensive language classifiers by more than 50% while also being able to preserve the readability and meaning of the modified text.

摘要: 社交媒体平台正在部署基于机器学习的攻击性语言分类系统，以大规模打击仇恨、种族主义和其他形式的攻击性言论。然而，尽管它们在现实世界中部署，我们还没有全面了解攻击性语言分类器对对手攻击的健壮程度。以前在这一领域的工作仅限于研究攻击性语言分类器对原始攻击(如拼写错误和无关空格)的稳健性。为了弥补这一差距，我们系统地分析了最先进的攻击性语言分类器对更狡猾的对手攻击的稳健性，这些攻击利用基于贪婪和注意力的单词选择和上下文感知嵌入进行单词替换。我们在多个数据集上的结果表明，这些狡猾的对抗性攻击可以使攻击性语言分类器的准确率降低50%以上，同时还能够保持修改后的文本的可读性和意义。



## **39. FGAN: Federated Generative Adversarial Networks for Anomaly Detection in Network Traffic**

FGAN：用于网络流量异常检测的联合生成对抗网络 cs.CR

8 pages, 2 figures

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11106v1)

**Authors**: Sankha Das

**Abstracts**: Over the last two decades, a lot of work has been done in improving network security, particularly in intrusion detection systems (IDS) and anomaly detection. Machine learning solutions have also been employed in IDSs to detect known and plausible attacks in incoming traffic. Parameters such as packet contents, sender IP and sender port, connection duration, etc. have been previously used to train these machine learning models to learn to differentiate genuine traffic from malicious ones. Generative Adversarial Networks (GANs) have been significantly successful in detecting such anomalies, mostly attributed to the adversarial training of the generator and discriminator in an attempt to bypass each other and in turn increase their own power and accuracy. However, in large networks having a wide variety of traffic at possibly different regions of the network and susceptible to a large number of potential attacks, training these GANs for a particular kind of anomaly may make it oblivious to other anomalies and attacks. In addition, the dataset required to train these models has to be made centrally available and publicly accessible, posing the obvious question of privacy of the communications of the respective participants of the network. The solution proposed in this work aims at tackling the above two issues by using GANs in a federated architecture in networks of such scale and capacity. In such a setting, different users of the network will be able to train and customize a centrally available adversarial model according to their own frequently faced conditions. Simultaneously, the member users of the network will also able to gain from the experiences of the other users in the network.

摘要: 在过去的二十年里，人们在提高网络安全方面做了大量的工作，特别是在入侵检测系统和异常检测方面。机器学习解决方案也被用于入侵检测系统中，以检测传入流量中的已知和可能的攻击。以前已经使用数据包内容、发送方IP和发送方端口、连接持续时间等参数来训练这些机器学习模型，以学习区分真实流量和恶意流量。生成性对抗网络(GANS)在检测这种异常方面取得了很大的成功，这主要归因于对生成器和鉴别器的对抗性训练，试图绕过彼此，进而提高自己的能力和准确性。然而，在可能在网络的不同区域具有各种流量并且容易受到大量潜在攻击的大型网络中，针对特定类型的异常对这些GAN进行训练可能会使其对其他异常和攻击视而不见。此外，训练这些模型所需的数据集必须集中提供并向公众开放，这就提出了网络各参与方通信隐私的明显问题。本工作提出的解决方案旨在通过在如此规模和容量的网络中的联合架构中使用GAN来解决上述两个问题。在这样的设置下，网络的不同用户将能够根据他们自己经常面临的情况来训练和定制中央可用的对抗模型。同时，网络的成员用户也将能够从网络中的其他用户的体验中获益。



## **40. An Intermediate-level Attack Framework on The Basis of Linear Regression**

一种基于线性回归的中级攻击框架 cs.CV

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10723v1)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. We advocate to establish a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to classification prediction loss of the adversarial example. In this paper, we delve deep into the core components of such a framework by performing comprehensive studies and extensive experiments. We show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level discrepancy is linearly correlated with adversarial transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. By leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks.

摘要: 本文大大扩展了我们在ECCV上发表的工作，在该工作中，提出了一种中级攻击来提高一些基线对手例子的可转移性。我们主张建立一个直接的线性映射，从对抗性实例的中间层差异(对抗性特征和良性特征之间)到分类预测损失。在本文中，我们通过全面的研究和广泛的实验，深入研究了这样一个框架的核心组件。我们证明：1)为了建立映射，可以考虑各种线性回归模型；2)最终得到的中间层差异的大小与对抗可转移性线性相关；3)通过随机初始化执行多次基线攻击，可以进一步提高性能。通过利用这些发现，我们实现了针对基于传输的$\ell_\inty$和$\ell_2$攻击的新技术。



## **41. A Prompting-based Approach for Adversarial Example Generation and Robustness Enhancement**

一种基于提示的对抗性实例生成与健壮性增强方法 cs.CL

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10714v1)

**Authors**: Yuting Yang, Pei Huang, Juan Cao, Jintao Li, Yun Lin, Jin Song Dong, Feifei Ma, Jian Zhang

**Abstracts**: Recent years have seen the wide application of NLP models in crucial areas such as finance, medical treatment, and news media, raising concerns of the model robustness and vulnerabilities. In this paper, we propose a novel prompt-based adversarial attack to compromise NLP models and robustness enhancement technique. We first construct malicious prompts for each instance and generate adversarial examples via mask-and-filling under the effect of a malicious purpose. Our attack technique targets the inherent vulnerabilities of NLP models, allowing us to generate samples even without interacting with the victim NLP model, as long as it is based on pre-trained language models (PLMs). Furthermore, we design a prompt-based adversarial training method to improve the robustness of PLMs. As our training method does not actually generate adversarial samples, it can be applied to large-scale training sets efficiently. The experimental results show that our attack method can achieve a high attack success rate with more diverse, fluent and natural adversarial examples. In addition, our robustness enhancement method can significantly improve the robustness of models to resist adversarial attacks. Our work indicates that prompting paradigm has great potential in probing some fundamental flaws of PLMs and fine-tuning them for downstream tasks.

摘要: 近年来，NLP模型在金融、医疗、新闻媒体等关键领域得到了广泛应用，引起了人们对模型健壮性和脆弱性的担忧。在本文中，我们提出了一种新的基于提示的对抗性攻击来折衷NLP模型和健壮性增强技术。我们首先为每个实例构建恶意提示，并在恶意目的的影响下通过掩码和填充生成敌意实例。我们的攻击技术针对NLP模型的固有漏洞，允许我们生成样本，即使不与受害者NLP模型交互，只要它基于预先训练的语言模型(PLM)。此外，我们还设计了一种基于提示的对抗性训练方法来提高PLM的健壮性。由于我们的训练方法并不实际生成对抗性样本，因此可以有效地应用于大规模训练集。实验结果表明，该攻击方法具有更丰富、更流畅、更自然的对抗性实例，攻击成功率较高。此外，我们的稳健性增强方法可以显著提高模型抵抗对手攻击的稳健性。我们的工作表明，激励范式在探测PLM的一些根本缺陷并为下游任务进行微调方面具有巨大的潜力。



## **42. Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition**

利用专家引导的对抗性增强改进命名实体识别中的泛化 cs.CL

ACL 2022 (Findings)

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10693v1)

**Authors**: Aaron Reich, Jiaao Chen, Aastha Agrawal, Yanzhe Zhang, Diyi Yang

**Abstracts**: Named Entity Recognition (NER) systems often demonstrate great performance on in-distribution data, but perform poorly on examples drawn from a shifted distribution. One way to evaluate the generalization ability of NER models is to use adversarial examples, on which the specific variations associated with named entities are rarely considered. To this end, we propose leveraging expert-guided heuristics to change the entity tokens and their surrounding contexts thereby altering their entity types as adversarial attacks. Using expert-guided heuristics, we augmented the CoNLL 2003 test set and manually annotated it to construct a high-quality challenging set. We found that state-of-the-art NER systems trained on CoNLL 2003 training data drop performance dramatically on our challenging set. By training on adversarial augmented training examples and using mixup for regularization, we were able to significantly improve the performance on the challenging set as well as improve out-of-domain generalization which we evaluated by using OntoNotes data. We have publicly released our dataset and code at https://github.com/GT-SALT/Guided-Adversarial-Augmentation.

摘要: 命名实体识别(NER)系统通常在分布内数据上表现出很好的性能，但在来自转移分布的例子上表现得很差。评估NER模型泛化能力的一种方法是使用对抗性例子，在这些例子上很少考虑与命名实体相关的特定变化。为此，我们建议利用专家指导的启发式方法来更改实体令牌及其周围上下文，从而将其实体类型更改为对抗性攻击。使用专家指导的启发式算法，我们扩充了CoNLL2003测试集并手动对其进行注释，以构建高质量的具有挑战性的测试集。我们发现，在我们具有挑战性的集合上，使用CoNLL 2003训练数据训练的最先进的NER系统的性能显著下降。通过对抗性增强训练样本的训练和使用混合正则化，我们能够显著提高在具有挑战性的集合上的性能，以及改善我们使用OntoNotes数据评估的域外泛化。我们已经在https://github.com/GT-SALT/Guided-Adversarial-Augmentation.公开发布了我们的数据集和代码



## **43. RareGAN: Generating Samples for Rare Classes**

RareGan：为稀有类生成样本 cs.LG

Published in AAAI 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10674v1)

**Authors**: Zinan Lin, Hao Liang, Giulia Fanti, Vyas Sekar

**Abstracts**: We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.

摘要: 我们研究了一类罕见的未标记数据集的生成对抗网络(GANS)的学习问题，该数据集受标记预算的约束。这一问题源于领域中的实际应用，包括安全(例如，为DNS放大攻击合成分组)、系统和网络(例如，合成触发高资源使用率的工作负载)以及机器学习(例如，从稀有类生成图像)。现有的方法是不合适的，要么需要完全标记的数据集，要么牺牲稀有类的保真度来换取普通类的保真度。我们提出了RareGAN，这是一种新的综合了三个关键思想的方法：(1)扩展条件Gans以使用标记和非标记数据来更好地泛化；(2)主动学习方法，要求最有用的标签；(3)加权损失函数，有利于学习稀有类。我们表明，在不同的应用、预算、稀有类分数、GaN损耗和体系结构上，RareGAN在稀有类上实现了更好的保真度-分集折衷。



## **44. Does DQN really learn? Exploring adversarial training schemes in Pong**

DQN真的会学习吗？探索乒乓球对抗性训练方案 cs.LG

RLDM 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10614v1)

**Authors**: Bowen He, Sreehari Rammohan, Jessica Forde, Michael Littman

**Abstracts**: In this work, we study two self-play training schemes, Chainer and Pool, and show they lead to improved agent performance in Atari Pong compared to a standard DQN agent -- trained against the built-in Atari opponent. To measure agent performance, we define a robustness metric that captures how difficult it is to learn a strategy that beats the agent's learned policy. Through playing past versions of themselves, Chainer and Pool are able to target weaknesses in their policies and improve their resistance to attack. Agents trained using these methods score well on our robustness metric and can easily defeat the standard DQN agent. We conclude by using linear probing to illuminate what internal structures the different agents develop to play the game. We show that training agents with Chainer or Pool leads to richer network activations with greater predictive power to estimate critical game-state features compared to the standard DQN agent.

摘要: 在这项工作中，我们研究了两种自我发挥训练方案，Chainer和Pool，并证明与标准的DQN代理相比，它们在Atari Pong中的代理性能有所提高--针对内置的Atari对手进行训练。为了衡量代理的性能，我们定义了一个健壮性度量，该度量捕获了学习超过代理的学习策略的策略有多难。通过扮演过去版本的自己，Chainer和Pool能够针对他们政策中的弱点，并提高他们对攻击的抵抗力。使用这些方法训练的代理在我们的稳健性度量上得分很好，可以很容易地击败标准的DQN代理。我们最后使用线性探测来说明不同的代理为玩游戏而发展的内部结构。我们表明，与标准的DQN代理相比，使用Chainer或Pool训练代理可以导致更丰富的网络激活，以及更强的预测能力，以估计关键的游戏状态特征。



## **45. Improved Semi-Quantum Key Distribution with Two Almost-Classical Users**

改进的两个准经典用户的半量子密钥分配 quant-ph

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10567v1)

**Authors**: Saachi Mutreja, Walter O. Krawec

**Abstracts**: Semi-quantum key distribution (SQKD) protocols attempt to establish a shared secret key between users, secure against computationally unbounded adversaries. Unlike standard quantum key distribution protocols, SQKD protocols contain at least one user who is limited in their quantum abilities and is almost "classical" in nature. In this paper, we revisit a mediated semi-quantum key distribution protocol, introduced by Massa et al., in 2019, where users need only the ability to detect a qubit, or reflect a qubit; they do not need to perform any other basis measurement; nor do they need to prepare quantum signals. Users require the services of a quantum server which may be controlled by the adversary. In this paper, we show how this protocol may be extended to improve its efficiency and also its noise tolerance. We discuss an extension which allows more communication rounds to be directly usable; we analyze the key-rate of this extension in the asymptotic scenario for a particular class of attacks and compare with prior work. Finally, we evaluate the protocol's performance in a variety of lossy and noisy channels.

摘要: 半量子密钥分发(SQKD)协议试图在用户之间建立共享密钥，该密钥在计算上不受限制的攻击者是安全的。与标准量子密钥分发协议不同，SQKD协议至少包含一个用户，该用户的量子能力是有限的，并且本质上几乎是“经典的”。在这篇文章中，我们回顾了由Massa等人在2019年提出的一个中介半量子密钥分发协议，其中用户只需要检测或反映量子比特的能力，他们不需要执行任何其他的基测量，也不需要准备量子信号。用户需要可能由对手控制的量子服务器的服务。在本文中，我们展示了如何对该协议进行扩展以提高其效率和抗噪性。我们讨论了一种允许更多通信轮次直接使用的扩展，分析了该扩展在一类特定攻击的渐近场景下的键率，并与以前的工作进行了比较。最后，我们对该协议在各种有损和噪声信道中的性能进行了评估。



## **46. Adversarial Parameter Attack on Deep Neural Networks**

基于深度神经网络的对抗性参数攻击 cs.LG

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10502v1)

**Authors**: Lijia Yu, Yihan Wang, Xiao-Shan Gao

**Abstracts**: In this paper, a new parameter perturbation attack on DNNs, called adversarial parameter attack, is proposed, in which small perturbations to the parameters of the DNN are made such that the accuracy of the attacked DNN does not decrease much, but its robustness becomes much lower. The adversarial parameter attack is stronger than previous parameter perturbation attacks in that the attack is more difficult to be recognized by users and the attacked DNN gives a wrong label for any modified sample input with high probability. The existence of adversarial parameters is proved. For a DNN $F_{\Theta}$ with the parameter set $\Theta$ satisfying certain conditions, it is shown that if the depth of the DNN is sufficiently large, then there exists an adversarial parameter set $\Theta_a$ for $\Theta$ such that the accuracy of $F_{\Theta_a}$ is equal to that of $F_{\Theta}$, but the robustness measure of $F_{\Theta_a}$ is smaller than any given bound. An effective training algorithm is given to compute adversarial parameters and numerical experiments are used to demonstrate that the algorithms are effective to produce high quality adversarial parameters.

摘要: 本文提出了一种新的DNN参数扰动攻击，称为对抗性参数攻击，通过对DNN的参数进行微小的扰动，使得被攻击的DNN的准确率不会有太大的下降，但其鲁棒性却大大降低。对抗性参数攻击比以往的参数扰动攻击更强，因为攻击更难被用户识别，并且被攻击的DNN对任何修改的样本输入都给出了错误的标签，概率很高。证明了对抗性参数的存在性。对于参数集$\Theta$满足一定条件的DNN$F_{\Theta}$，证明了如果DNN的深度足够大，则对于$\Theta$存在对抗参数集$\Theta_a$，使得$F_{\Theta}$的精度等于$F_{\Theta}$，但$F_{\Theta_a}$的稳健性度量小于任何给定界。给出了一种有效的训练算法来计算对抗参数，并通过数值实验证明了该算法能够有效地产生高质量的对抗参数。



## **47. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

对比性对抗训练中认知分离缓解的稳健性 cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08959v2)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.

摘要: 本文提出了一种新的神经网络训练框架，通过将对比学习(CL)和对抗训练(AT)相结合，在保持较高精度的同时，提高了模型对对手攻击的鲁棒性。我们提出通过学习在数据扩充和对抗性扰动下都是一致的特征表示来提高模型对对抗性攻击的稳健性。我们利用对比学习来提高对抗性样本的稳健性，将一个对抗性样本作为另一个正例，目标是最大化随机增加的数据样本与其对抗性样本之间的相似度，同时不断更新分类头，以避免分类头与嵌入空间之间的认知分离。这种分离是由于CL将网络更新到嵌入空间，同时冻结用于生成新的正面对抗性实例的分类头。我们在CIFAR-10数据集上验证了我们的方法，即带有对抗性特征的对比学习(CLAF)，在CIFAR-10数据集上，它的性能优于其他监督和自我监督对抗性学习方法的稳健准确率和干净准确率。



## **48. On Robust Prefix-Tuning for Text Classification**

面向文本分类的稳健前缀调优方法研究 cs.CL

Accepted in ICLR 2022. We release the code at  https://github.com/minicheshire/Robust-Prefix-Tuning

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10378v1)

**Authors**: Zonghan Yang, Yang Liu

**Abstracts**: Recently, prefix-tuning has gained increasing attention as a parameter-efficient finetuning method for large-scale pretrained language models. The method keeps the pretrained models fixed and only updates the prefix token parameters for each downstream task. Despite being lightweight and modular, prefix-tuning still lacks robustness to textual adversarial attacks. However, most currently developed defense techniques necessitate auxiliary model update and storage, which inevitably hamper the modularity and low storage of prefix-tuning. In this work, we propose a robust prefix-tuning framework that preserves the efficiency and modularity of prefix-tuning. The core idea of our framework is leveraging the layerwise activations of the language model by correctly-classified training data as the standard for additional prefix finetuning. During the test phase, an extra batch-level prefix is tuned for each batch and added to the original prefix for robustness enhancement. Extensive experiments on three text classification benchmarks show that our framework substantially improves robustness over several strong baselines against five textual attacks of different types while maintaining comparable accuracy on clean texts. We also interpret our robust prefix-tuning framework from the optimal control perspective and pose several directions for future research.

摘要: 近年来，前缀调优作为一种用于大规模预训练语言模型的参数高效的精调方法，受到越来越多的关注。该方法保持预先训练的模型不变，只更新每个下游任务的前缀令牌参数。尽管前缀调整是轻量级和模块化的，但它仍然缺乏对文本对手攻击的健壮性。然而，目前开发的大多数防御技术需要辅助模型更新和存储，这不可避免地阻碍了前缀调整的模块化和低存储量。在这项工作中，我们提出了一个健壮的前缀调整框架，它保持了前缀调整的效率和模块化。我们框架的核心思想是通过正确分类训练数据来利用语言模型的LayerWise激活作为额外前缀优化的标准。在测试阶段，为每个批次调整额外的批次级别前缀，并将其添加到原始前缀以增强健壮性。在三个文本分类基准上的大量实验表明，我们的框架显著提高了对五种不同类型的文本攻击在几个强基线上的稳健性，同时保持了对干净文本的相对准确性。我们还从最优控制的角度解释了我们的鲁棒前缀调整框架，并对未来的研究提出了几个方向。



## **49. Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense**

荒野中的扰动：利用人类书写的文本扰动进行现实的对抗性攻击和防御 cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22), Findings

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10346v1)

**Authors**: Thai Le, Jooyoung Lee, Kevin Yen, Yifan Hu, Dongwon Lee

**Abstracts**: We proposes a novel algorithm, ANTHRO, that inductively extracts over 600K human-written text perturbations in the wild and leverages them for realistic adversarial attack. Unlike existing character-based attacks which often deductively hypothesize a set of manipulation strategies, our work is grounded on actual observations from real-world texts. We find that adversarial texts generated by ANTHRO achieve the best trade-off between (1) attack success rate, (2) semantic preservation of the original text, and (3) stealthiness--i.e. indistinguishable from human writings hence harder to be flagged as suspicious. Specifically, our attacks accomplished around 83% and 91% attack success rates on BERT and RoBERTa, respectively. Moreover, it outperformed the TextBugger baseline with an increase of 50% and 40% in terms of semantic preservation and stealthiness when evaluated by both layperson and professional human workers. ANTHRO can further enhance a BERT classifier's performance in understanding different variations of human-written toxic texts via adversarial training when compared to the Perspective API.

摘要: 我们提出了一种新的算法Anthro，该算法在野外归纳提取超过60万个人类书写的文本扰动，并利用它们进行现实的对抗性攻击。与现有的基于字符的攻击通常演绎地假设一组操纵策略不同，我们的工作基于对真实世界文本的实际观察。我们发现，Anthro生成的敌意文本在(1)攻击成功率、(2)保持原始文本的语义和(3)隐蔽性--即隐蔽性--之间达到了最好的权衡。与人类的文字难以区分，因此更难被标记为可疑。具体地说，我们对Bert和Roberta的攻击成功率分别约为83%和91%。此外，在外行和专业人员的评估中，它在语义保持和隐蔽性方面的表现优于TextBugger基线，分别提高了50%和40%。与透视API相比，Anthro可以通过对抗性训练进一步提高BERT分类器在理解人类书写的有毒文本的不同变体方面的性能。



## **50. Efficient Neural Network Analysis with Sum-of-Infeasibilities**

不可行和条件下的高效神经网络分析 cs.LG

TACAS'22

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11201v1)

**Authors**: Haoze Wu, Aleksandar Zeljić, Guy Katz, Clark Barrett

**Abstracts**: Inspired by sum-of-infeasibilities methods in convex optimization, we propose a novel procedure for analyzing verification queries on neural networks with piecewise-linear activation functions. Given a convex relaxation which over-approximates the non-convex activation functions, we encode the violations of activation functions as a cost function and optimize it with respect to the convex relaxation. The cost function, referred to as the Sum-of-Infeasibilities (SoI), is designed so that its minimum is zero and achieved only if all the activation functions are satisfied. We propose a stochastic procedure, DeepSoI, to efficiently minimize the SoI. An extension to a canonical case-analysis-based complete search procedure can be achieved by replacing the convex procedure executed at each search state with DeepSoI. Extending the complete search with DeepSoI achieves multiple simultaneous goals: 1) it guides the search towards a counter-example; 2) it enables more informed branching decisions; and 3) it creates additional opportunities for bound derivation. An extensive evaluation across different benchmarks and solvers demonstrates the benefit of the proposed techniques. In particular, we demonstrate that SoI significantly improves the performance of an existing complete search procedure. Moreover, the SoI-based implementation outperforms other state-of-the-art complete verifiers. We also show that our technique can efficiently improve upon the perturbation bound derived by a recent adversarial attack algorithm.

摘要: 受凸优化中不可行和方法的启发，提出了一种分析分段线性激活函数神经网络上验证查询的新方法。给定一个过度逼近非凸激活函数的凸松弛，我们将激活函数的违例编码为代价函数，并相对于凸松弛进行优化。成本函数被称为不可行和(SOI)，其最小值是零，并且只有在满足所有激活函数的情况下才能实现。我们提出了一个随机过程，DeepSoI，以有效地最小化SOI。通过用DeepSoI替换在每个搜索状态下执行的凸过程，可以实现对基于规范案例分析的完全搜索过程的扩展。使用DeepSoI扩展完整搜索可同时实现多个目标：1)它引导搜索指向反例；2)它支持更明智的分支决策；3)它为界限派生创造更多机会。对不同基准和求解器的广泛评估表明了所提出的技术的好处。特别是，我们证明了SOI显著提高了现有完整搜索过程的性能。此外，基于SOI的实现比其他最先进的完全验证器性能更好。我们还表明，我们的技术可以有效地改善最近的对抗性攻击算法得出的扰动界。



