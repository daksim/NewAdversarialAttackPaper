# New Adversarial Attack Papers
**update at 2021-11-16**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. NNoculation: Catching BadNets in the Wild**

NNoculation：野外抓恶网 cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2002.08313v2)

**Authors**: Akshaj Kumar Veldanda, Kang Liu, Benjamin Tan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Brendan Dolan-Gavitt, Siddharth Garg

**Abstracts**: This paper proposes a novel two-stage defense (NNoculation) against backdoored neural networks (BadNets) that, repairs a BadNet both pre-deployment and online in response to backdoored test inputs encountered in the field. In the pre-deployment stage, NNoculation retrains the BadNet with random perturbations of clean validation inputs to partially reduce the adversarial impact of a backdoor. Post-deployment, NNoculation detects and quarantines backdoored test inputs by recording disagreements between the original and pre-deployment patched networks. A CycleGAN is then trained to learn transformations between clean validation and quarantined inputs; i.e., it learns to add triggers to clean validation images. Backdoored validation images along with their correct labels are used to further retrain the pre-deployment patched network, yielding our final defense. Empirical evaluation on a comprehensive suite of backdoor attacks show that NNoculation outperforms all state-of-the-art defenses that make restrictive assumptions and only work on specific backdoor attacks, or fail on adaptive attacks. In contrast, NNoculation makes minimal assumptions and provides an effective defense, even under settings where existing defenses are ineffective due to attackers circumventing their restrictive assumptions.

摘要: 提出了一种针对回溯神经网络(BadNets)的新的两阶段防御(NNoculation)，即对BadNet进行预部署和在线修复，以响应现场遇到的回溯测试输入。在部署前阶段，NNoculation使用干净验证输入的随机扰动重新训练BadNet，以部分降低后门的敌对影响。部署后，NNoculation通过记录原始修补网络和部署前修补网络之间的不一致来检测和隔离反向测试输入。然后，训练CycleGAN学习干净验证和隔离输入之间的转换；即，它学习向干净验证图像添加触发器。后置的验证映像及其正确的标签用于进一步重新训练部署前修补的网络，从而实现我们的最终防御。对一套全面的后门攻击的经验评估表明，NNoculation的性能优于所有做出限制性假设并仅在特定后门攻击上有效，或在自适应攻击上失败的最先进的防御措施。相比之下，NNoculation只做最少的假设并提供有效的防御，即使在现有防御因攻击者绕过其限制性假设而无效的情况下也是如此。



## **2. Generative Dynamic Patch Attack**

生成式动态补丁攻击 cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.04266v2)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.

摘要: 对抗性补丁攻击是一系列攻击算法，通过扰动图像的一部分来欺骗深层神经网络模型。现有的补丁攻击大多考虑在与输入无关的位置(预定义位置或随机位置)注入敌意补丁。这种攻击设置对于攻击来说可能是足够的，但在用于对抗性训练时有相当大的限制。因此，用现有补丁攻击训练的鲁棒模型不能有效防御其他对抗性攻击。本文首先提出了一种端到端的补丁攻击算法--生成性动态补丁攻击(GDPA)，该算法对每幅输入图像分别生成补丁模式和补丁位置。我们证明了GDPA是一个通用的攻击框架，只需少量的配置更改，就可以生成动态/静电和可见/不可见的补丁。其次，GDPA可以很容易地集成到对抗性训练中，以提高模型对各种对抗性攻击的鲁棒性。在VGGFace、交通标志和ImageNet上的大量实验表明，GDPA比最新的补丁攻击具有更高的攻击成功率，而带有GDPA的对抗性训练模型对敌意补丁攻击表现出比竞争方法更好的鲁棒性。我们的源代码可以在https://github.com/lxuniverse/gdpa.上找到



## **3. Website fingerprinting on early QUIC traffic**

早期Quic流量的网站指纹分析 cs.CR

This work has been accepted by Elsevier Computer Networks for  publication

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2101.11871v2)

**Authors**: Pengwei Zhan, Liming Wang, Yi Tang

**Abstracts**: Cryptographic protocols have been widely used to protect the user's privacy and avoid exposing private information. QUIC (Quick UDP Internet Connections), including the version originally designed by Google (GQUIC) and the version standardized by IETF (IQUIC), as alternatives to the traditional HTTP, demonstrate their unique transmission characteristics: based on UDP for encrypted resource transmitting, accelerating web page rendering. However, existing encrypted transmission schemes based on TCP are vulnerable to website fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited websites by eavesdropping on the transmission channel. Whether GQUIC and IQUIC can effectively resist such attacks is worth investigating. In this paper, we study the vulnerabilities of GQUIC, IQUIC, and HTTPS to WFP attacks from the perspective of traffic analysis. Extensive experiments show that, in the early traffic scenario, GQUIC is the most vulnerable to WFP attacks among GQUIC, IQUIC, and HTTPS, while IQUIC is more vulnerable than HTTPS, but the vulnerability of the three protocols is similar in the normal full traffic scenario. Features transferring analysis shows that most features are transferable between protocols when on normal full traffic scenario. However, combining with the qualitative analysis of latent feature representation, we find that the transferring is inefficient when on early traffic, as GQUIC, IQUIC, and HTTPS show the significantly different magnitude of variation in the traffic distribution on early traffic. By upgrading the one-time WFP attacks to multiple WFP Top-a attacks, we find that the attack accuracy on GQUIC and IQUIC reach 95.4% and 95.5%, respectively, with only 40 packets and just using simple features, whereas reach only 60.7% when on HTTPS. We also demonstrate that the vulnerability of IQUIC is only slightly dependent on the network environment.

摘要: 密码协议已被广泛用于保护用户隐私和避免泄露私人信息。Quic(Quick UDP Internet Connections，快速UDP Internet连接)，包括Google(GQUIC)最初设计的版本和IETF(IQUIC)标准化的版本，作为传统HTTP的替代品，展示了它们独特的传输特性：基于UDP进行加密资源传输，加速网页渲染。然而，现有的基于TCP的加密传输方案容易受到网站指纹识别(WFP)攻击，使得攻击者能够通过窃听传输通道来推断用户访问的网站。GQUIC和IQUIC能否有效抵御此类攻击值得研究。本文从流量分析的角度研究了GQUIC、IQUIC和HTTPS对WFP攻击的脆弱性。大量的实验表明，在早期流量场景中，GQUIC是GQUIC、IQUIC和HTTPS中最容易受到WFP攻击的协议，而IQUIC比HTTPS更容易受到攻击，但在正常的全流量场景下，这三种协议的漏洞是相似的。特征转移分析表明，在正常全流量场景下，协议间的大部分特征是可以转移的。然而，结合潜在特征表示的定性分析，我们发现在早期流量上的传输效率较低，因为GQUIC、IQUIC和HTTPS在早期流量上的流量分布表现出明显不同的变化幅度。通过将一次性的WFP攻击升级为多个WFP Top-a攻击，我们发现GQUIC和IQUIC的攻击准确率分别达到了95.4%和95.5%，只有40个数据包，而且只使用了简单的特征，而在HTTPS上的攻击准确率只有60.7%。我们还证明了IQUIC的漏洞对网络环境的依赖性很小。



## **4. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

敌意检测规避攻击：评估基于感知散列的客户端扫描的健壮性 cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2106.09820v2)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9\% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.

摘要: 消息传递平台提供的端到端加密(E2EE)使人们能够安全、私密地相互通信。然而，它的广泛采用引发了人们的担忧，即非法内容现在可能会被分享而不被发现。继全球对密钥托管系统的抵制之后，科技公司、政府和研究人员最近提出了基于感知散列的客户端扫描，以检测E2EE通信中的非法内容。我们在这里提出了第一个框架来评估基于感知散列的客户端扫描对检测规避攻击的健壮性，并表明当前的系统是不健壮的。更具体地说，我们提出了三种针对感知散列算法的对抗性攻击--一种通用的黑盒攻击和两种基于离散余弦变换的算法的白盒攻击。在大规模的评估中，我们发现基于感知散列的客户端扫描机制在黑盒环境下非常容易受到检测回避攻击，在保护图像内容的同时，99.9%以上的图像被成功攻击。此外，我们还展示了我们的攻击会产生不同的扰动，这强烈地表明直接的缓解策略将是无效的。最后，我们指出，增加攻击难度所需的更大门槛可能需要每天标记和解密超过10亿张图像，这引发了强烈的隐私问题。综上所述，我们的结果对目前世界各地的政府、组织和研究人员提出的基于感知散列的客户端扫描机制的健壮性提出了严重的质疑。



## **5. Property Inference Attacks Against GANs**

针对GAN的属性推理攻击 cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07608v1)

**Authors**: Junhao Zhou, Yufei Chen, Chao Shen, Yang Zhang

**Abstracts**: While machine learning (ML) has made tremendous progress during the past decade, recent research has shown that ML models are vulnerable to various security and privacy attacks. So far, most of the attacks in this field focus on discriminative models, represented by classifiers. Meanwhile, little attention has been paid to the security and privacy risks of generative models, such as generative adversarial networks (GANs). In this paper, we propose the first set of training dataset property inference attacks against GANs. Concretely, the adversary aims to infer the macro-level training dataset property, i.e., the proportion of samples used to train a target GAN with respect to a certain attribute. A successful property inference attack can allow the adversary to gain extra knowledge of the target GAN's training dataset, thereby directly violating the intellectual property of the target model owner. Also, it can be used as a fairness auditor to check whether the target GAN is trained with a biased dataset. Besides, property inference can serve as a building block for other advanced attacks, such as membership inference. We propose a general attack pipeline that can be tailored to two attack scenarios, including the full black-box setting and partial black-box setting. For the latter, we introduce a novel optimization framework to increase the attack efficacy. Extensive experiments over four representative GAN models on five property inference tasks show that our attacks achieve strong performance. In addition, we show that our attacks can be used to enhance the performance of membership inference against GANs.

摘要: 虽然机器学习(ML)在过去的十年中取得了巨大的进步，但最近的研究表明，ML模型容易受到各种安全和隐私攻击。到目前为止，该领域的攻击大多集中在以分类器为代表的区分模型上。同时，生成性模型，如生成性对抗性网络(GANS)的安全和隐私风险也很少受到关注。在本文中，我们提出了第一组针对GANS的训练数据集属性推理攻击。具体地说，对手的目标是推断宏观级别的训练数据集属性，即用于训练目标GAN的样本相对于特定属性的比例。成功的属性推理攻击可以让攻击者获得目标GAN训练数据集的额外知识，从而直接侵犯目标模型所有者的知识产权。此外，它还可以用作公平性审核器，以检查目标GAN是否使用有偏差的数据集进行训练。此外，属性推理还可以作为构建挡路的平台，用于其他高级攻击，如成员身份推理。我们提出了一种通用攻击流水线，该流水线可以针对两种攻击场景进行定制，包括完全黑盒设置和部分黑盒设置。对于后者，我们引入了一种新的优化框架来提高攻击效率。在4个典型的GAN模型上对5个属性推理任务进行的大量实验表明，我们的攻击取得了很好的性能。此外，我们还证明了我们的攻击可以用来提高针对GANS的成员关系推理的性能。



## **6. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

accepted at NeurIPS 2021, including the appendix

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07492v1)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法没有超参数和预训练。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **7. Towards Interpretability of Speech Pause in Dementia Detection using Adversarial Learning**

对抗性学习在痴呆检测中言语停顿的可解释性研究 cs.CL

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07454v1)

**Authors**: Youxiang Zhu, Bang Tran, Xiaohui Liang, John A. Batsis, Robert M. Roth

**Abstracts**: Speech pause is an effective biomarker in dementia detection. Recent deep learning models have exploited speech pauses to achieve highly accurate dementia detection, but have not exploited the interpretability of speech pauses, i.e., what and how positions and lengths of speech pauses affect the result of dementia detection. In this paper, we will study the positions and lengths of dementia-sensitive pauses using adversarial learning approaches. Specifically, we first utilize an adversarial attack approach by adding the perturbation to the speech pauses of the testing samples, aiming to reduce the confidence levels of the detection model. Then, we apply an adversarial training approach to evaluate the impact of the perturbation in training samples on the detection model. We examine the interpretability from the perspectives of model accuracy, pause context, and pause length. We found that some pauses are more sensitive to dementia than other pauses from the model's perspective, e.g., speech pauses near to the verb "is". Increasing lengths of sensitive pauses or adding sensitive pauses leads the model inference to Alzheimer's Disease, while decreasing the lengths of sensitive pauses or deleting sensitive pauses leads to non-AD.

摘要: 言语停顿是检测痴呆的有效生物标志物。最近的深度学习模型利用语音停顿来实现高精度的痴呆症检测，但没有利用语音停顿的可解释性，即语音停顿的位置和长度如何以及如何影响痴呆症检测的结果。在本文中，我们将使用对抗性学习方法来研究痴呆症敏感停顿的位置和长度。具体地说，我们首先利用对抗性攻击方法，通过在测试样本的语音停顿中添加扰动来降低检测模型的置信度。然后，我们应用对抗性训练方法来评估训练样本中的扰动对检测模型的影响。我们从模型精度、暂停上下文和暂停长度的角度来检查可解释性。我们发现，从模型的角度来看，一些停顿比其他停顿对痴呆症更敏感，例如，动词“is”附近的言语停顿。增加敏感停顿的长度或增加敏感停顿会导致模型对阿尔茨海默病的推断，而减少敏感停顿的长度或删除敏感停顿会导致非AD。



## **8. Generating Band-Limited Adversarial Surfaces Using Neural Networks**

用神经网络生成带限对抗性曲面 cs.CV

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07424v1)

**Authors**: Roee Ben Shlomo, Yevgeniy Men, Ido Imanuel

**Abstracts**: Generating adversarial examples is the art of creating a noise that is added to an input signal of a classifying neural network, and thus changing the network's classification, while keeping the noise as tenuous as possible. While the subject is well-researched in the 2D regime, it is lagging behind in the 3D regime, i.e. attacking a classifying network that works on 3D point-clouds or meshes and, for example, classifies the pose of people's 3D scans. As of now, the vast majority of papers that describe adversarial attacks in this regime work by methods of optimization. In this technical report we suggest a neural network that generates the attacks. This network utilizes PointNet's architecture with some alterations. While the previous articles on which we based our work on have to optimize each shape separately, i.e. tailor an attack from scratch for each individual input without any learning, we attempt to create a unified model that can deduce the needed adversarial example with a single forward run.

摘要: 生成对抗性示例是创建噪声的艺术，该噪声被添加到分类神经网络的输入信号，从而改变网络的分类，同时保持噪声尽可能微弱。虽然这个主题在2D模式下研究得很好，但在3D模式下却落后了，即攻击工作在3D点云或网格上的分类网络，例如，对人们3D扫描的姿势进行分类。到目前为止，绝大多数描述该制度下的对抗性攻击的论文都是通过优化的方法来工作的。在这份技术报告中，我们建议使用神经网络来生成攻击。这个网络采用了PointNet的架构，但做了一些改动。虽然我们工作所基于的前面的文章必须分别优化每个形状，即在没有任何学习的情况下为每个单独的输入从头开始定制攻击，但我们试图创建一个统一的模型，它可以通过一次向前运行来推导出所需的对抗性示例。



## **9. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

测量多个模型表示在检测对抗性实例中的贡献 cs.LG

**SubmitDate**: 2021-11-13    [paper-pdf](http://arxiv.org/pdf/2111.07035v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.

摘要: 深度学习模型已被广泛用于各种任务。它们广泛应用于计算机视觉、自然语言处理、语音识别等领域。虽然这些模型在许多情况下都工作得很好，但已经表明它们很容易受到对手的攻击。这导致了对如何识别和/或防御此类攻击的研究激增。我们的目标是探索可以归因于使用多个底层模型进行对抗性实例检测的贡献。我们的论文描述了两种方法，它们融合了来自多个模型的表示，用于检测对抗性示例。我们设计了对照实验来衡量增量利用额外模型的检测影响。对于我们考虑的许多场景，结果显示性能随着用于提取表示的底层模型数量的增加而提高。



## **10. Adversarially Robust Learning for Security-Constrained Optimal Power Flow**

安全约束最优潮流的对抗性鲁棒学习 math.OC

Accepted at Neural Information Processing Systems (NeurIPS) 2021

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06961v1)

**Authors**: Priya L. Donti, Aayushya Agarwal, Neeraj Vijay Bedmutha, Larry Pileggi, J. Zico Kolter

**Abstracts**: In recent years, the ML community has seen surges of interest in both adversarially robust learning and implicit layers, but connections between these two areas have seldom been explored. In this work, we combine innovations from these areas to tackle the problem of N-k security-constrained optimal power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical grids, and aims to schedule power generation in a manner that is robust to potentially k simultaneous equipment outages. Inspired by methods in adversarially robust training, we frame N-k SCOPF as a minimax optimization problem - viewing power generation settings as adjustable parameters and equipment outages as (adversarial) attacks - and solve this problem via gradient-based techniques. The loss function of this minimax problem involves resolving implicit equations representing grid physics and operational decisions, which we differentiate through via the implicit function theorem. We demonstrate the efficacy of our framework in solving N-3 SCOPF, which has traditionally been considered as prohibitively expensive to solve given that the problem size depends combinatorially on the number of potential outages.

摘要: 近些年来，ML社区看到了对相反的健壮学习和隐含层的兴趣激增，但这两个领域之间的联系很少被探索。在这项工作中，我们结合这些领域的创新成果，提出了撞击求解N-k安全约束最优潮流问题。n-k SCOPF是电网运行的核心问题，其目标是以一种对潜在的k个同时设备故障具有鲁棒性的方式来调度发电。受对抗性鲁棒训练方法的启发，我们将N-k SCOPF定义为一个极小极大优化问题--将发电设置视为可调参数，将设备故障视为(对抗性)攻击--并通过基于梯度的技术解决该问题。这个极小极大问题的损失函数涉及到求解代表网格物理和操作决策的隐式方程，我们通过隐函数定理来区分这些方程。我们证明了我们的框架在解决N-3 SCOPF方面的有效性，传统上认为解决N-3 SCOPF的成本高得令人望而却步，因为问题的大小组合地取决于潜在的中断次数。



## **11. Resilient Consensus-based Multi-agent Reinforcement Learning**

基于弹性共识的多智能体强化学习 cs.LG

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06776v1)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.

摘要: 训练过程中的对抗性攻击会严重影响多智能体强化学习算法的性能。因此，非常需要对现有算法进行扩充，以便消除或至少有界地消除对抗性攻击对协作网络的影响。在这项工作中，我们考虑了一个完全分散的网络，在这个网络中，每个代理都会获得局部奖励，并观察全局状态和行动。我们提出了一种弹性的基于共识的行动者-批评者算法，其中每个Agent估计团队平均奖励和价值函数，并将相关的参数向量传达给它的直接邻居。我们证明了当拜占庭代理的估计和通信策略完全任意时，假设每个合作代理的邻域中至多有$H$拜占庭代理，并且网络是$(2H+1)$-鲁棒的，则合作代理的估计以概率1收敛到有界的合意值。在假设对抗性Agent的策略渐近平稳的前提下，证明了合作Agent的策略以概率1收敛到其团队平均目标函数的局部极大值附近的有界邻域。



## **12. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06628v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知散列系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知散列的综合实证分析。具体地说，我们表明当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的改变来操纵散列值，这些改变要么是由基于梯度的方法引起的，要么是简单地通过执行标准图像转换来强制或防止散列冲突。这样的攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **13. Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations**

通过背景增强来表征和提高自监督学习的鲁棒性 cs.CV

Technical Report; Additional Results

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2103.12719v2)

**Authors**: Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

**Abstracts**: Recent progress in self-supervised learning has demonstrated promising results in multiple visual tasks. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, ignoring the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Through a systematic investigation, we show that background augmentations lead to substantial improvements in performance across a spectrum of state-of-the-art self-supervised methods (MoCo-v2, BYOL, SwAV) on a variety of tasks, e.g. $\sim$+1-2% gains on ImageNet, enabling performance on par with the supervised baseline. Further, we find the improvement in limited-labels settings is even larger (up to 4.2%). Background augmentations also improve robustness to a number of distribution shifts, including natural adversarial examples, ImageNet-9, adversarial attacks, ImageNet-Renditions. We also make progress in completely unsupervised saliency detection, in the process of generating saliency masks used for background augmentations.

摘要: 自我监督学习的最新进展在多视觉任务中显示出良好的结果。高性能自监督方法的一个重要组成部分是通过训练模型使用数据增强来将同一图像的不同增强视图放置在嵌入空间的附近。然而，常用的增强流水线从整体上对待图像，忽略了图像各部分的语义相关性。主题与背景--这可能导致学习虚假的相关性。我们的工作通过调查一类简单但高效的“背景增强”来解决这个问题，这种“背景增强”通过阻止模型关注图像背景来鼓励模型专注于语义相关的内容。通过系统调查，我们发现背景增强在各种任务(例如$\sim$+ImageNet$\sim$+1-2%)的一系列最先进的自我监督方法(MoCo-v2、BYOL、SwAV)上显著提高了性能，使性能与监督基线持平。此外，我们发现限制标签设置的改进更大(高达4.2%)。背景增强还提高了对许多分布变化的健壮性，包括自然对抗性示例、ImageNet-9、对抗性攻击、ImageNet-Renditions。在生成用于背景增强的显著掩码的过程中，我们还在完全无监督的显著性检测方面取得了进展。



## **14. Distributionally Robust Trajectory Optimization Under Uncertain Dynamics via Relative Entropy Trust-Regions**

基于相对熵信赖域的不确定动态分布鲁棒轨迹优化 eess.SY

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2103.15388v3)

**Authors**: Hany Abdulsamad, Tim Dorau, Boris Belousov, Jia-Jie Zhu, Jan Peters

**Abstracts**: Trajectory optimization and model predictive control are essential techniques underpinning advanced robotic applications, ranging from autonomous driving to full-body humanoid control. State-of-the-art algorithms have focused on data-driven approaches that infer the system dynamics online and incorporate posterior uncertainty during planning and control. Despite their success, such approaches are still susceptible to catastrophic errors that may arise due to statistical learning biases, unmodeled disturbances, or even directed adversarial attacks. In this paper, we tackle the problem of dynamics mismatch and propose a distributionally robust optimal control formulation that alternates between two relative entropy trust-region optimization problems. Our method finds the worst-case maximum entropy Gaussian posterior over the dynamics parameters and the corresponding robust policy. Furthermore, we show that our approach admits a closed-form backward-pass for a certain class of systems. Finally, we demonstrate the resulting robustness on linear and nonlinear numerical examples.

摘要: 轨迹优化和模型预测控制是支撑先进机器人应用的关键技术，从自动驾驶到全身仿人控制。最先进的算法专注于数据驱动的方法，这些方法在线推断系统动态，并在计划和控制过程中纳入后验不确定性。尽管这些方法取得了成功，但它们仍然容易受到灾难性错误的影响，这些错误可能是由于统计学习偏差、未建模的干扰，甚至是定向的对抗性攻击而产生的。本文对动态失配问题进行了撞击研究，提出了一种在两个相对熵信赖域优化问题之间交替的分布鲁棒最优控制公式。我们的方法求出了动力学参数的最坏情况下的最大熵高斯后验分布，并给出了相应的鲁棒策略。此外，我们还证明了我们的方法对于某一类系统允许闭合形式的后向传递(Closed-Form Back-Pass)。最后，我们在线性和非线性数值算例上证明了所得结果的鲁棒性。



## **15. Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

通过关系推理模式毒化知识图嵌入 cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.

摘要: 研究了针对知识图中链接预测任务的知识图嵌入(KGE)模型产生数据中毒攻击的问题。为了毒化KGE模型，我们提出开发KGE模型的归纳能力，这些能力是通过知识图中的对称性、反转和合成等关系模式捕捉到的。具体地说，为了降低模型对目标事实的预测置信度，我们提出了提高模型对一组诱饵事实的预测置信度的方法。因此，我们设计了对抗性的加法，通过不同的推理模式来提高模型对诱饵事实的预测置信度。我们的实验表明，提出的中毒攻击在两个公开可用的数据集的四个KGE模型上的性能优于最新的基线。我们还发现，基于对称模式的攻击在所有模型-数据集组合中都是通用的，这表明了KGE模型对该模式的敏感性。



## **16. Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes**

Qu反化：利用量化伪像实现对抗性结果 cs.LG

Accepted to NeurIPS 2021 [Poster]

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2110.13541v2)

**Authors**: Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yiğitcan Kaya, Tudor Dumitraş

**Abstracts**: Quantization is a popular technique that $transforms$ the parameter representation of a neural network from floating-point numbers into lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in $behavioral$ $disparities$ between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation

摘要: 量化是一种流行的技术，它将神经网络的参数表示从浮点数转换为低精度数字(例如$，8位整数)。它减少了推理时的内存占用和计算成本，便于部署资源匮乏的模型。然而，这种变换引起的参数扰动导致了量化前后模型之间的$行为$$差异$。例如，量化模型可能会错误分类一些本来可以正确分类的测试时间样本。目前尚不清楚这种差异是否会导致新的安全漏洞。我们假设对手可以控制这种差异，以引入量化后激活的特定行为。为了研究这一假设，我们将量化意识训练武器化，并提出了一个新的训练框架来实现对抗性量化结果。在此框架下，我们提出了三种量化的攻击：(I)不加区别的攻击，造成显著的精度损失；(Ii)针对特定样本的有针对性的攻击；以及(Iii)用输入触发器控制模型的后门攻击。我们进一步表明，单一的折衷模型击败了包括鲁棒量化技术在内的多个量化方案。此外，在联合学习场景中，我们演示了一组合谋的恶意参与者可以注入我们的量化激活后门。最后，我们讨论了潜在的对策，并表明只有重新训练才能始终如一地移除攻击伪影。我们的代码可在https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation获得



## **17. Robust Deep Reinforcement Learning through Adversarial Loss**

对抗性损失下的鲁棒深度强化学习 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2008.01976v2)

**Authors**: Tuomas Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng

**Abstracts**: Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.

摘要: 最近的研究表明，深度强化学习Agent容易受到Agent输入的小的对抗性扰动，这引起了人们对在现实世界中部署此类Agent的担忧。为了解决这个问题，我们提出了RADIUS-RL，一个原则性的框架来训练强化学习Agent，使其对$l_p$-范数有界的攻击具有更好的鲁棒性。我们的框架与流行的深度强化学习算法兼容，并通过深度Q-学习、A3C和PPO验证了其性能。我们在三个深度RL基准(Atari、MuJoCo和ProcGen)上进行了实验，以验证我们的鲁棒训练算法的有效性。我们的Radius-RL代理在针对不同强度的攻击进行测试时，性能一直优于以前的方法，并且在训练时的计算效率更高。此外，我们还提出了一种新的评估方法--贪婪最坏情况奖励(GWC)来度量深度RL代理的攻击不可知性。我们表明，GWC可以被有效地评估，并且是在可能的最坏的对抗性攻击序列下的一个很好的奖励估计。我们实验使用的所有代码都可以在https://github.com/tuomaso/radial_rl_v2.上找到



## **18. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **19. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

基于集成密度传播的深度神经网络鲁棒学习 cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.

摘要: 对于深度神经网络(DNNs)来说，在不确定、噪声或敌对环境中学习是一项具有挑战性的任务。在贝叶斯估计和变分推理的基础上，提出了一种新的具有理论基础的、高效的鲁棒学习方法。我们用集合密度传播(ENDP)方案描述了DNN各层间的密度传播问题，并对其进行了求解。ENDP方法允许我们在贝叶斯DNN的各层之间传播变分概率分布的矩，从而能够在模型的输出处估计预测分布的均值和协方差。我们使用MNIST和CIFAR-10数据集进行的实验表明，训练后的模型对随机噪声和敌意攻击的鲁棒性有了显着的提高。



## **20. Audio Attacks and Defenses against AED Systems -- A Practical Study**

AED系统的音频攻击与防御--一项实用研究 cs.SD

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2106.07428v4)

**Authors**: Rodrigo dos Santos, Shirin Nilizadeh

**Abstracts**: In this paper, we evaluate deep learning-enabled AED systems against evasion attacks based on adversarial examples. We test the robustness of multiple security critical AED tasks, implemented as CNNs classifiers, as well as existing third-party Nest devices, manufactured by Google, which run their own black-box deep learning models. Our adversarial examples use audio perturbations made of white and background noises. Such disturbances are easy to create, to perform and to reproduce, and can be accessible to a large number of potential attackers, even non-technically savvy ones.   We show that an adversary can focus on audio adversarial inputs to cause AED systems to misclassify, achieving high success rates, even when we use small levels of a given type of noisy disturbance. For instance, on the case of the gunshot sound class, we achieve nearly 100% success rate when employing as little as 0.05 white noise level. Similarly to what has been previously done by works focusing on adversarial examples from the image domain as well as on the speech recognition domain. We then, seek to improve classifiers' robustness through countermeasures. We employ adversarial training and audio denoising. We show that these countermeasures, when applied to audio input, can be successful, either in isolation or in combination, generating relevant increases of nearly fifty percent in the performance of the classifiers when these are under attack.

摘要: 在这篇文章中，我们评估了深度学习使能的AED系统对逃避攻击的抵抗能力，这是基于敌意的例子。我们测试了多个安全关键AED任务(实现为CNNS分类器)以及由Google制造的现有第三方Nest设备的健壮性，这些设备运行自己的黑盒深度学习模型。我们的对抗性例子使用由白噪声和背景噪声构成的音频扰动。这样的干扰很容易制造、执行和复制，而且可以被大量潜在的攻击者接触到，即使是不懂技术的人也可以接触到。我们表明，即使当我们使用少量的给定类型的噪声干扰时，对手也可以专注于音频对手输入，导致AED系统错误分类，从而实现高成功率。例如，在枪声类的情况下，当使用低至0.05的白噪声水平时，我们获得了近100%的成功率。类似于先前通过关注来自图像域以及语音识别领域的对抗性示例的作品所做的工作。然后，通过对策寻求提高分类器的鲁棒性。我们采用对抗性训练和音频去噪。我们表明，当这些对策应用于音频输入时，无论是单独应用还是组合应用，都可以取得成功，当分类器受到攻击时，分类器的性能会相应提高近50%。



## **21. A black-box adversarial attack for poisoning clustering**

一种针对中毒聚类的黑盒对抗性攻击 cs.LG

18 pages, Pattern Recognition 2022

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2009.05474v4)

**Authors**: Antonio Emanuele Cinà, Alessandro Torcinovich, Marcello Pelillo

**Abstracts**: Clustering algorithms play a fundamental role as tools in decision-making and sensible automation processes. Due to the widespread use of these applications, a robustness analysis of this family of algorithms against adversarial noise has become imperative. To the best of our knowledge, however, only a few works have currently addressed this problem. In an attempt to fill this gap, in this work, we propose a black-box adversarial attack for crafting adversarial samples to test the robustness of clustering algorithms. We formulate the problem as a constrained minimization program, general in its structure and customizable by the attacker according to her capability constraints. We do not assume any information about the internal structure of the victim clustering algorithm, and we allow the attacker to query it as a service only. In the absence of any derivative information, we perform the optimization with a custom approach inspired by the Abstract Genetic Algorithm (AGA). In the experimental part, we demonstrate the sensibility of different single and ensemble clustering algorithms against our crafted adversarial samples on different scenarios. Furthermore, we perform a comparison of our algorithm with a state-of-the-art approach showing that we are able to reach or even outperform its performance. Finally, to highlight the general nature of the generated noise, we show that our attacks are transferable even against supervised algorithms such as SVMs, random forests, and neural networks.

摘要: 聚类算法在决策和明智的自动化过程中起着基础性的作用。由于这些应用的广泛使用，对这类算法进行抗对抗噪声的鲁棒性分析已成为当务之急。然而，就我们所知，目前只有几部著作解决了这个问题。为了填补这一空白，在这项工作中，我们提出了一种用于制作敌意样本的黑盒对抗性攻击，以测试聚类算法的健壮性。我们将问题描述为一个有约束的最小化规划，其结构一般，攻击者可以根据她的能力约束进行定制。我们不假定有关受害者群集算法的内部结构的任何信息，并且我们仅允许攻击者将其作为服务进行查询。在没有任何导数信息的情况下，受抽象遗传算法(AGA)的启发，采用定制的方法进行优化。在实验部分，我们展示了不同的单一聚类算法和集成聚类算法在不同场景下对我们制作的敌意样本的敏感度。此外，我们将我们的算法与最先进的方法进行了比较，结果表明我们能够达到甚至超过它的性能。最后，为了突出生成噪声的一般性质，我们证明了我们的攻击即使是针对支持向量机、随机森林和神经网络等有监督算法也是可以转移的。



## **22. Universal Multi-Party Poisoning Attacks**

普遍存在的多方中毒攻击 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/1809.03474v3)

**Authors**: Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

**Abstracts**: In this work, we demonstrate universal multi-party poisoning attacks that adapt and apply to any multi-party learning process with arbitrary interaction pattern between the parties. More generally, we introduce and study $(k,p)$-poisoning attacks in which an adversary controls $k\in[m]$ of the parties, and for each corrupted party $P_i$, the adversary submits some poisoned data $\mathcal{T}'_i$ on behalf of $P_i$ that is still ``$(1-p)$-close'' to the correct data $\mathcal{T}_i$ (e.g., $1-p$ fraction of $\mathcal{T}'_i$ is still honestly generated). We prove that for any ``bad'' property $B$ of the final trained hypothesis $h$ (e.g., $h$ failing on a particular test example or having ``large'' risk) that has an arbitrarily small constant probability of happening without the attack, there always is a $(k,p)$-poisoning attack that increases the probability of $B$ from $\mu$ to by $\mu^{1-p \cdot k/m} = \mu + \Omega(p \cdot k/m)$. Our attack only uses clean labels, and it is online.   More generally, we prove that for any bounded function $f(x_1,\dots,x_n) \in [0,1]$ defined over an $n$-step random process $\mathbf{X} = (x_1,\dots,x_n)$, an adversary who can override each of the $n$ blocks with even dependent probability $p$ can increase the expected output by at least $\Omega(p \cdot \mathrm{Var}[f(\mathbf{x})])$.

摘要: 在这项工作中，我们展示了通用的多方中毒攻击，适用于任何具有任意交互模式的多方学习过程。更一般地，我们引入和研究了$(k，p)$中毒攻击，在这种攻击中，敌手控制着当事人的$k\in[m]$，并且对于每个被破坏的一方$P_i$，对手代表$P_i$提交一些有毒数据$\数学{T}‘_i$，该数据仍然是’‘$(1-p)$-接近’‘正确数据$\数学{T}_i$(例如，$1-p$分数$\我们证明了对于最终训练假设$h$的任何“坏”性质$B$(例如，$h$在特定的测试用例上失败或具有“大的”风险)，在没有攻击的情况下发生的概率任意小的恒定概率，总是存在$(k，p)$中毒攻击，它将$B$的概率从$\µ$增加到$\µ^{1-p\CDOT k/m}=\Mu+\Omega(p\CDOT k/m)$。我们的攻击只使用干净的标签，而且是在线的。更一般地，我们证明了对于定义在$n$步随机过程$\mathbf{X}=(x_1，\点，x_n)$上的[0，1]$中的任何有界函数$f(x_1，\dots，x_n)\，能够以偶数依赖概率$p$覆盖$n$块中的每个块的对手可以使期望输出至少增加$\Omega(p\cdot\mathm{var}[f(\mam



## **23. Sparse Adversarial Video Attacks with Spatial Transformations**

基于空间变换的稀疏对抗性视频攻击 cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

摘要: 近年来，大量的研究工作集中在图像的对抗性攻击上，而对抗性视频攻击的研究很少。我们提出了一种针对视频的对抗性攻击策略，称为DeepSAVA。我们的模型通过一个统一的优化框架同时包括加性扰动和空间变换，其中采用结构相似指数(SSIM)度量对抗距离。我们设计了一种有效和新颖的优化方案，它交替使用贝叶斯优化来识别视频中最有影响力的帧，以及基于随机梯度下降(SGD)的优化来产生加性和空间变换的扰动。这样做使DeepSAVA能够对视频执行非常稀疏的攻击，以保持人的不可感知性，同时在攻击成功率和对手可转移性方面仍获得最先进的性能。我们在不同类型的深度神经网络和视频数据集上的密集实验证实了DeepSAVA的优越性。



## **24. Are Transformers More Robust Than CNNs?**

变形金刚比CNN更健壮吗？ cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.

摘要: 变压器作为一种强有力的视觉识别工具应运而生。除了展示好胜在广泛的可视基准上的性能外，最近的研究还认为，变形金刚比卷积神经网络(CNN)更健壮。然而，令人惊讶的是，我们发现这些结论是从不公平的实验环境中得出的，在这些实验环境中，变形金刚和CNN在不同的尺度上进行了比较，并应用了不同的训练框架。在本文中，我们的目标是提供变压器和CNN之间的第一次公平和深入的比较，重点是鲁棒性评估。有了我们的统一训练设置，我们首先挑战了以前的信念，即在衡量对手的健壮性时，变形金刚优于CNN。更令人惊讶的是，我们发现，如果CNN恰当地采用了变形金刚的训练食谱，它们在防御对手攻击方面可以很容易地像变形金刚一样健壮。虽然关于分布外样本的泛化，我们表明(外部)大规模数据集的预训练并不是使Transformers获得比CNN更好的性能的基本要求。此外，我们的消融表明，这种更强的概括性在很大程度上得益于变形金刚的自我关注式架构本身，而不是其他培训设置。我们希望这项工作可以帮助社区更好地理解Transformers和CNN的健壮性，并对其进行基准测试。代码和模型可在https://github.com/ytongbai/ViTs-vs-CNNs.上公开获得



## **25. Statistical Perspectives on Reliability of Artificial Intelligence Systems**

人工智能系统可靠性的统计透视 cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.

摘要: 人工智能(AI)系统在许多领域变得越来越受欢迎。然而，人工智能技术仍处于发展阶段，许多问题需要解决。其中，需要证明人工智能系统的可靠性，以便普通公众可以放心地使用人工智能系统。在这篇文章中，我们提供了关于人工智能系统可靠性的统计观点。与其他考虑因素不同，人工智能系统的可靠性侧重于时间维度。也就是说，系统可以在预期的时间段内执行其设计的功能。介绍了一种用于人工智能可靠性研究的智能统计框架，该框架包括五个组成部分：系统结构、可靠性度量、故障原因分析、可靠性评估和测试规划。我们回顾了可靠性数据分析和软件可靠性的传统方法，并讨论了如何将这些现有方法转化为人工智能系统的可靠性建模和评估。我们还描述了人工智能可靠性建模和分析的最新进展，并概述了该领域的统计研究挑战，包括分布失调检测、训练集的影响、对抗性攻击、模型精度和不确定性量化，并用说明性例子讨论了这些主题如何与人工智能可靠性相关。最后，我们讨论了人工智能可靠性评估的数据收集和测试规划，以及如何改进系统设计以提高人工智能可靠性。论文最后以一些结束语结束。



## **26. TDGIA:Effective Injection Attacks on Graph Neural Networks**

TDGIA：对图神经网络的有效注入攻击 cs.LG

KDD 2021 research track paper

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2106.06663v2)

**Authors**: Xu Zou, Qinkai Zheng, Yuxiao Dong, Xinyu Guan, Evgeny Kharlamov, Jialiang Lu, Jie Tang

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. However, recent studies have shown that GNNs are vulnerable to adversarial attacks. In this paper, we study a recently-introduced realistic attack scenario on graphs -- graph injection attack (GIA). In the GIA scenario, the adversary is not able to modify the existing link structure and node attributes of the input graph, instead the attack is performed by injecting adversarial nodes into it. We present an analysis on the topological vulnerability of GNNs under GIA setting, based on which we propose the Topological Defective Graph Injection Attack (TDGIA) for effective injection attacks. TDGIA first introduces the topological defective edge selection strategy to choose the original nodes for connecting with the injected ones. It then designs the smooth feature optimization objective to generate the features for the injected nodes. Extensive experiments on large-scale datasets show that TDGIA can consistently and significantly outperform various attack baselines in attacking dozens of defense GNN models. Notably, the performance drop on target GNNs resultant from TDGIA is more than double the damage brought by the best attack solution among hundreds of submissions on KDD-CUP 2020.

摘要: 图神经网络(GNNs)在各种实际应用中取得了良好的性能。然而，最近的研究表明，GNN很容易受到敌意攻击。本文研究了最近引入的一种图的现实攻击场景--图注入攻击(GIA)。在GIA场景中，敌手不能修改输入图的现有链接结构和节点属性，而是通过向其中注入敌方节点来执行攻击。分析了GIA环境下GNNs的拓扑脆弱性，在此基础上提出了针对有效注入攻击的拓扑缺陷图注入攻击(TDGIA)。TDGIA首先引入拓扑缺陷边选择策略，选择原始节点与注入节点连接。然后设计平滑特征优化目标，为注入节点生成特征。在大规模数据集上的广泛实验表明，TDGIA在攻击数十个防御GNN模型时，可以一致且显着地优于各种攻击基线。值得注意的是，TDGIA对目标GNN造成的性能下降是KDD-Cup 2020上百份提交的最佳攻击解决方案带来的损害的两倍多。



## **27. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

对抗性稳健性的统一博弈论解释 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2103.07364v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, i.e. the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.

摘要: 本文提供了一个统一的视角来解释不同的对抗性攻击和防御方法，即DNNs的输入变量之间的多阶交互的观点。基于多阶交互，我们发现对抗性攻击主要影响高阶交互来欺骗DNN。此外，我们发现对抗性训练的DNN的鲁棒性来自于特定类别的低阶交互。我们的发现为统一对抗性扰动和鲁棒性提供了一种潜在的方法，可以对现有的防御方法进行原则性的解释。此外，我们的发现还修正了以往对对抗性习得特征的形状偏向的不准确理解。



## **28. Membership Inference Attacks Against Self-supervised Speech Models**

针对自监督语音模型的隶属度推理攻击 cs.CR

Submitted to ICASSP 2022. Source code available at  https://github.com/RayTzeng/s3m-membership-inference

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05113v1)

**Authors**: Wei-Cheng Tseng, Wei-Tsung Kao, Hung-yi Lee

**Abstracts**: Recently, adapting the idea of self-supervised learning (SSL) on continuous speech has started gaining attention. SSL models pre-trained on a huge amount of unlabeled audio can generate general-purpose representations that benefit a wide variety of speech processing tasks. Despite their ubiquitous deployment, however, the potential privacy risks of these models have not been well investigated. In this paper, we present the first privacy analysis on several SSL speech models using Membership Inference Attacks (MIA) under black-box access. The experiment results show that these pre-trained models are vulnerable to MIA and prone to membership information leakage with high adversarial advantage scores in both utterance-level and speaker-level. Furthermore, we also conduct several ablation studies to understand the factors that contribute to the success of MIA.

摘要: 最近，将自我监督学习(SSL)的思想应用于连续语音的研究开始受到关注。在大量未标记音频上预先训练的SSL模型可以生成有利于各种语音处理任务的通用表示。然而，尽管它们的部署无处不在，但这些模型的潜在隐私风险还没有得到很好的调查。本文首次对几种SSL语音模型在黑盒访问下使用成员推理攻击(MIA)进行了隐私分析。实验结果表明，这些预训练模型在话语级和说话人级都有较高的对手优势得分，容易受到MIA的影响，容易泄露成员信息。此外，我们还进行了几项消融研究，以了解导致MIA成功的因素。



## **29. A Statistical Difference Reduction Method for Escaping Backdoor Detection**

一种逃避后门检测的统计减差方法 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05077v1)

**Authors**: Pengfei Xia, Hongjing Niu, Ziqiang Li, Bin Li

**Abstracts**: Recent studies show that Deep Neural Networks (DNNs) are vulnerable to backdoor attacks. An infected model behaves normally on benign inputs, whereas its prediction will be forced to an attack-specific target on adversarial data. Several detection methods have been developed to distinguish inputs to defend against such attacks. The common hypothesis that these defenses rely on is that there are large statistical differences between the latent representations of clean and adversarial inputs extracted by the infected model. However, although it is important, comprehensive research on whether the hypothesis must be true is lacking. In this paper, we focus on it and study the following relevant questions: 1) What are the properties of the statistical differences? 2) How to effectively reduce them without harming the attack intensity? 3) What impact does this reduction have on difference-based defenses? Our work is carried out on the three questions. First, by introducing the Maximum Mean Discrepancy (MMD) as the metric, we identify that the statistical differences of multi-level representations are all large, not just the highest level. Then, we propose a Statistical Difference Reduction Method (SDRM) by adding a multi-level MMD constraint to the loss function during training a backdoor model to effectively reduce the differences. Last, three typical difference-based detection methods are examined. The F1 scores of these defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70% on the models trained with SDRM on all two datasets, four model architectures, and four attack methods. The results indicate that the proposed method can be used to enhance existing attacks to escape backdoor detection algorithms.

摘要: 最近的研究表明，深度神经网络(DNNs)很容易受到后门攻击。被感染的模型在良性输入上表现正常，而它的预测将被迫在对抗性数据上针对攻击特定的目标。已经开发了几种检测方法来区分输入以防御此类攻击。这些防御所依赖的共同假设是，由感染模型提取的干净和敌对输入的潜在表示之间存在很大的统计差异。然而，尽管这很重要，但关于这一假设是否一定是真的缺乏全面的研究。本文针对这一问题进行了研究：1)统计差异的性质是什么？2)如何在不影响攻击强度的情况下有效地降低统计差异？3)这种减少对基于差异的防御有什么影响？(2)如何在不影响攻击强度的情况下有效地减少统计差异？3)这种减少对基于差异的防御有什么影响？我们的工作就是围绕这三个问题展开的。首先，通过引入最大平均差异(MMD)作为度量，我们发现多级表示的统计差异都很大，而不仅仅是最高级别。然后，在后门模型训练过程中，通过在损失函数中加入多级MMD约束，提出了一种统计差值缩减方法(SDRM)，有效地减小了差值。最后，分析了三种典型的基于差分的检测方法。在所有两个数据集、四个模型体系结构和四种攻击方法上，这些防御的F1得分从定期训练的后门模型的90%-100%下降到使用SDRM训练的模型的60%-70%。实验结果表明，该方法可用于增强现有的逃避后门检测算法的攻击。



## **30. Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search**

用自动损失函数搜索法缩小对抗性风险的逼近误差 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05063v1)

**Authors**: Pengfei Xia, Ziqiang Li, Bin Li

**Abstracts**: Numerous studies have demonstrated that deep neural networks are easily misled by adversarial examples. Effectively evaluating the adversarial robustness of a model is important for its deployment in practical applications. Currently, a common type of evaluation is to approximate the adversarial risk of a model as a robustness indicator by constructing malicious instances and executing attacks. Unfortunately, there is an error (gap) between the approximate value and the true value. Previous studies manually design attack methods to achieve a smaller error, which is inefficient and may miss a better solution. In this paper, we establish the tightening of the approximation error as an optimization problem and try to solve it with an algorithm. More specifically, we first analyze that replacing the non-convex and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in calculating the approximation, is one of the main reasons for the error. Then we propose AutoLoss-AR, the first method for searching loss functions for tightening the approximation error of adversarial risk. Extensive experiments are conducted in multiple settings. The results demonstrate the effectiveness of the proposed method: the best-discovered loss functions outperform the handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10, respectively. Besides, we also verify that the searched losses can be transferred to other settings and explore why they are better than the baseline by visualizing the local loss landscape.

摘要: 大量研究表明，深度神经网络很容易被对抗性例子所误导。有效地评估模型的对抗健壮性对于其在实际应用中的部署具有重要意义。目前，一种常见的评估方法是通过构建恶意实例和执行攻击来近似模型的敌意风险作为健壮性指标。不幸的是，近似值和真实值之间存在误差(差距)。以往的研究都是通过手工设计攻击方法来实现较小的错误，效率较低，可能会错过更好的解决方案。本文将逼近误差的收紧问题建立为优化问题，并尝试用算法求解。更具体地说，我们首先分析了用替代损失代替非凸的、不连续的0-1损失是造成误差的主要原因之一，这是计算近似时的一种必要的折衷。在此基础上，提出了第一种搜索损失函数的方法AutoLoss-AR，以减小对手风险的逼近误差。在多个环境中进行了广泛的实验。结果证明了该方法的有效性：在MNIST和CIFAR-10上，最好发现的损失函数的性能分别比手工制作的基线高0.9%-2.9%和0.7%-2.0%。此外，我们还验证了搜索到的损失可以转移到其他设置，并通过可视化本地损失情况来探索为什么它们比基线更好。



## **31. GraphAttacker: A General Multi-Task GraphAttack Framework**

GraphAttacker：一个通用的多任务GraphAttack框架 cs.LG

17 pages,9 figeures

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2101.06855v2)

**Authors**: Jinyin Chen, Dunjie Zhang, Zhaoyan Ming, Kejie Huang, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural networks (GNNs) have been successfully exploited in graph analysis tasks in many real-world applications. The competition between attack and defense methods also enhances the robustness of GNNs. In this competition, the development of adversarial training methods put forward higher requirement for the diversity of attack examples. By contrast, most attack methods with specific attack strategies are difficult to satisfy such a requirement. To address this problem, we propose GraphAttacker, a novel generic graph attack framework that can flexibly adjust the structures and the attack strategies according to the graph analysis tasks. GraphAttacker generates adversarial examples through alternate training on three key components: the multi-strategy attack generator (MAG), the similarity discriminator (SD), and the attack discriminator (AD), based on the generative adversarial network (GAN). Furthermore, we introduce a novel similarity modification rate SMR to conduct a stealthier attack considering the change of node similarity distribution. Experiments on various benchmark datasets demonstrate that GraphAttacker can achieve state-of-the-art attack performance on graph analysis tasks of node classification, graph classification, and link prediction, no matter the adversarial training is conducted or not. Moreover, we also analyze the unique characteristics of each task and their specific response in the unified attack framework. The project code is available at https://github.com/honoluluuuu/GraphAttacker.

摘要: 图神经网络(GNNs)已被成功地应用于许多实际应用中的图分析任务中。攻防手段的竞争也增强了GNNs的健壮性。在本次比赛中，对抗性训练方法的发展对进攻实例的多样性提出了更高的要求。相比之下，大多数具有特定攻击策略的攻击方法很难满足这一要求。针对这一问题，我们提出了一种新的通用图攻击框架GraphAttacker，该框架可以根据图分析任务灵活调整攻击结构和攻击策略。GraphAttacker在生成对抗网络(GAN)的基础上，通过交替训练多策略攻击生成器(MAG)、相似判别器(SD)和攻击鉴别器(AD)这三个关键部件来生成对抗性实例。此外，考虑到节点相似度分布的变化，引入了一种新的相似度修改率SMR来进行隐身攻击。在不同的基准数据集上的实验表明，无论是否进行对抗性训练，GraphAttacker都可以在节点分类、图分类和链接预测等图分析任务上获得最先进的攻击性能。此外，我们还分析了在统一攻击框架下每个任务的独特性和它们的具体响应。项目代码可在https://github.com/honoluluuuu/GraphAttacker.上找到



## **32. Reversible Attack based on Local Visual Adversarial Perturbation**

基于局部视觉对抗扰动的可逆攻击 cs.CV

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2110.02700v2)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstracts**: Deep learning is getting more and more outstanding performance in many tasks such as autonomous driving and face recognition and also has been challenged by different kinds of attacks. Adding perturbations that are imperceptible to human vision in an image can mislead the neural network model to get wrong results with high confidence. Adversarial Examples are images that have been added with specific noise to mislead a deep neural network model However, adding noise to images destroys the original data, making the examples useless in digital forensics and other fields. To prevent illegal or unauthorized access of image data such as human faces and ensure no affection to legal use reversible adversarial attack technique is rise. The original image can be recovered from its reversible adversarial example. However, the existing reversible adversarial examples generation strategies are all designed for the traditional imperceptible adversarial perturbation. How to get reversibility for locally visible adversarial perturbation? In this paper, we propose a new method for generating reversible adversarial examples based on local visual adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by reversible data hiding technique. To reduce image distortion and improve visual quality, lossless compression and B-R-G embedding principle are adopted. Experiments on ImageNet dataset show that our method can restore the original images error-free while ensuring the attack performance.

摘要: 深度学习在自动驾驶、人脸识别等任务中表现越来越突出，也受到了各种攻击的挑战。在图像中添加人眼无法察觉的扰动可能会误导神经网络模型，使其在高置信度下得到错误的结果。对抗性的例子是添加了特定噪声的图像，以误导深度神经网络模型。然而，向图像添加噪声会破坏原始数据，使这些示例在数字取证和其他领域中无用。为了防止对人脸等图像数据的非法或未经授权的访问，确保不影响合法使用，可逆对抗攻击技术应运而生。原始图像可以从其可逆的对抗性示例中恢复。然而，现有的可逆对抗性实例生成策略都是针对传统的潜意识对抗性扰动而设计的。如何获得局部可见的对抗性扰动的可逆性？本文提出了一种基于局部可视对抗性扰动的可逆对抗性实例生成方法。利用可逆数据隐藏技术将图像恢复所需的信息嵌入到敌方补丁之外的区域。为了减少图像失真，提高视觉质量，采用无损压缩和B-R-G嵌入原理。在ImageNet数据集上的实验表明，该方法可以在保证攻击性能的前提下无差错地恢复原始图像。



## **33. On Robustness of Neural Ordinary Differential Equations**

关于神经常微分方程的稳健性 cs.LG

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/1910.05513v3)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks. \url{https://github.com/HanshuYAN/TisODE}

摘要: 近年来，神经常微分方程(ODE)在各个研究领域受到越来越多的关注。已有一些研究神经常微分方程的优化问题和逼近能力的工作，但其鲁棒性尚不清楚。在这项工作中，我们通过从经验和理论上探索神经ODE的稳健性来填补这一重要空白。我们首先对基于ODENET的神经网络(ODENet)的鲁棒性进行了实证研究，方法是将ODENet暴露在具有各种类型扰动的输入中，然后研究相应输出的变化。与传统的卷积神经网络(CNNs)相比，我们发现ODENet对随机高斯扰动和敌意攻击示例都具有更强的鲁棒性。然后，我们通过利用连续时间颂歌的流的某些理想性质，即积分曲线是不相交的，来提供对这一现象的深刻理解。我们的工作表明，由于其固有的鲁棒性，使用神经ODE作为构建鲁棒深层网络模型的基础挡路是很有前途的。为了进一步增强香草神经微分方程组的鲁棒性，我们提出了时不变稳态神经微分方程组(TisODE)，它通过时不变性和施加稳态约束来规则化扰动数据上的流动。我们表明，TisODE方法的性能优于香草神经ODE方法，并且还可以与其他最先进的体系结构方法相结合来构建更健壮的深层网络。\url{https://github.com/HanshuYAN/TisODE}



## **34. Bayesian Framework for Gradient Leakage**

梯度泄漏的贝叶斯框架 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04706v1)

**Authors**: Mislav Balunović, Dimitar I. Dimitrov, Robin Staab, Martin Vechev

**Abstracts**: Federated learning is an established method for training machine learning models without sharing training data. However, recent work has shown that it cannot guarantee data privacy as shared gradients can still leak sensitive information. To formalize the problem of gradient leakage, we propose a theoretical framework that enables, for the first time, analysis of the Bayes optimal adversary phrased as an optimization problem. We demonstrate that existing leakage attacks can be seen as approximations of this optimal adversary with different assumptions on the probability distributions of the input data and gradients. Our experiments confirm the effectiveness of the Bayes optimal adversary when it has knowledge of the underlying distribution. Further, our experimental evaluation shows that several existing heuristic defenses are not effective against stronger attacks, especially early in the training process. Thus, our findings indicate that the construction of more effective defenses and their evaluation remains an open problem.

摘要: 联合学习是一种在不共享训练数据的情况下训练机器学习模型的既定方法。然而，最近的研究表明，它不能保证数据隐私，因为共享梯度仍然可能泄露敏感信息。为了形式化梯度泄漏问题，我们提出了一个理论框架，该框架首次能够将贝叶斯最优对手表述为优化问题进行分析。我们证明了现有的泄漏攻击可以看作是对输入数据的概率分布和梯度的不同假设的最优对手的近似。我们的实验证实了贝叶斯最优对手在知道潜在分布的情况下的有效性。此外，我们的实验评估表明，现有的几种启发式防御方法对更强的攻击并不有效，特别是在训练过程的早期。因此，我们的研究结果表明，构建更有效的防御体系及其评估仍然是一个悬而未决的问题。



## **35. HAPSSA: Holistic Approach to PDF Malware Detection Using Signal and Statistical Analysis**

HAPSSA：基于信号和统计分析的PDF恶意软件整体检测方法 cs.CR

Submitted version - MILCOM 2021 IEEE Military Communications  Conference

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04703v1)

**Authors**: Tajuddin Manhar Mohammed, Lakshmanan Nataraj, Satish Chikkagoudar, Shivkumar Chandrasekaran, B. S. Manjunath

**Abstracts**: Malicious PDF documents present a serious threat to various security organizations that require modern threat intelligence platforms to effectively analyze and characterize the identity and behavior of PDF malware. State-of-the-art approaches use machine learning (ML) to learn features that characterize PDF malware. However, ML models are often susceptible to evasion attacks, in which an adversary obfuscates the malware code to avoid being detected by an Antivirus. In this paper, we derive a simple yet effective holistic approach to PDF malware detection that leverages signal and statistical analysis of malware binaries. This includes combining orthogonal feature space models from various static and dynamic malware detection methods to enable generalized robustness when faced with code obfuscations. Using a dataset of nearly 30,000 PDF files containing both malware and benign samples, we show that our holistic approach maintains a high detection rate (99.92%) of PDF malware and even detects new malicious files created by simple methods that remove the obfuscation conducted by malware authors to hide their malware, which are undetected by most antiviruses.

摘要: 恶意PDF文档对各种安全组织构成严重威胁，这些组织需要现代威胁情报平台来有效地分析和表征PDF恶意软件的身份和行为。最先进的方法使用机器学习(ML)来学习PDF恶意软件的特征。然而，ML模型经常容易受到规避攻击，在这种攻击中，敌手混淆恶意软件代码以避免被防病毒程序检测到。在本文中，我们推导了一种简单而有效的整体PDF恶意软件检测方法，该方法利用恶意软件二进制文件的信号和统计分析。这包括组合来自各种静电的正交特征空间模型和动态恶意软件检测方法，以在面临代码混淆时实现普遍的鲁棒性。使用包含近30,000个包含恶意软件和良性样本的PDF文件的数据集，我们显示，我们的整体方法保持了较高的PDF恶意软件检测率(99.92%)，甚至可以检测到通过简单方法创建的新恶意文件，这些新的恶意文件消除了恶意软件作者为隐藏恶意软件而进行的混淆，而这些文件是大多数防病毒软件无法检测到的。



## **36. A Separation Result Between Data-oblivious and Data-aware Poisoning Attacks**

数据迟钝和数据感知中毒攻击的分离结果 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2003.12020v2)

**Authors**: Samuel Deng, Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Abhradeep Thakurta

**Abstracts**: Poisoning attacks have emerged as a significant security threat to machine learning algorithms. It has been demonstrated that adversaries who make small changes to the training set, such as adding specially crafted data points, can hurt the performance of the output model. Some of the stronger poisoning attacks require the full knowledge of the training data. This leaves open the possibility of achieving the same attack results using poisoning attacks that do not have the full knowledge of the clean training set.   In this work, we initiate a theoretical study of the problem above. Specifically, for the case of feature selection with LASSO, we show that full-information adversaries (that craft poisoning examples based on the rest of the training data) are provably stronger than the optimal attacker that is oblivious to the training set yet has access to the distribution of the data. Our separation result shows that the two setting of data-aware and data-oblivious are fundamentally different and we cannot hope to always achieve the same attack or defense results in these scenarios.

摘要: 中毒攻击已经成为机器学习算法的重大安全威胁。已经证明，对训练集进行微小更改的对手，例如添加巧尽心思构建的数据点，可能会损害输出模型的性能。一些更强的中毒攻击需要完全了解训练数据。这使得使用不完全了解干净训练集的中毒攻击获得相同的攻击结果的可能性仍然存在。在这项工作中，我们开始了对上述问题的理论研究。具体地说，对于使用套索进行特征选择的情况，我们证明了全信息对手(基于训练数据的睡觉来制作中毒实例)比最优攻击者(即对训练集是迟钝但可以访问数据分布的攻击者)更强大。我们的分离结果表明，数据感知和数据迟钝这两个设置是根本不同的，我们不能指望在这些场景下总是能达到相同的攻防效果。



## **37. DeepSteal: Advanced Model Extractions Leveraging Efficient Weight Stealing in Memories**

DeepSteal：高级模型提取，利用记忆中有效的重量窃取 cs.CR

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04625v1)

**Authors**: Adnan Siraj Rakin, Md Hafizul Islam Chowdhuryy, Fan Yao, Deliang Fan

**Abstracts**: Recent advancements of Deep Neural Networks (DNNs) have seen widespread deployment in multiple security-sensitive domains. The need of resource-intensive training and use of valuable domain-specific training data have made these models a top intellectual property (IP) for model owners. One of the major threats to the DNN privacy is model extraction attacks where adversaries attempt to steal sensitive information in DNN models. Recent studies show hardware-based side channel attacks can reveal internal knowledge about DNN models (e.g., model architectures) However, to date, existing attacks cannot extract detailed model parameters (e.g., weights/biases). In this work, for the first time, we propose an advanced model extraction attack framework DeepSteal that effectively steals DNN weights with the aid of memory side-channel attack. Our proposed DeepSteal comprises two key stages. Firstly, we develop a new weight bit information extraction method, called HammerLeak, through adopting the rowhammer based hardware fault technique as the information leakage vector. HammerLeak leverages several novel system-level techniques tailed for DNN applications to enable fast and efficient weight stealing. Secondly, we propose a novel substitute model training algorithm with Mean Clustering weight penalty, which leverages the partial leaked bit information effectively and generates a substitute prototype of the target victim model. We evaluate this substitute model extraction method on three popular image datasets (e.g., CIFAR-10/100/GTSRB) and four DNN architectures (e.g., ResNet-18/34/Wide-ResNet/VGG-11). The extracted substitute model has successfully achieved more than 90 % test accuracy on deep residual networks for the CIFAR-10 dataset. Moreover, our extracted substitute model could also generate effective adversarial input samples to fool the victim model.

摘要: 近年来，深度神经网络(DNNs)在多个安全敏感领域得到了广泛的应用。对资源密集型培训的需求和对有价值的特定领域培训数据的使用已使这些模型成为模型所有者的最高知识产权(IP)。DNN隐私面临的主要威胁之一是模型提取攻击，即攻击者试图窃取DNN模型中的敏感信息。最近的研究表明，基于硬件的侧信道攻击可以揭示DNN模型(例如，模型体系结构)的内部知识，然而，到目前为止，现有的攻击不能提取详细的模型参数(例如，权重/偏差)。在这项工作中，我们首次提出了一个高级模型提取攻击框架DeepSteal，该框架可以借助记忆边信道攻击有效地窃取DNN权重。我们建议的DeepSteal包括两个关键阶段。首先，通过采用基于Rowhammer的硬件故障技术作为信息泄漏向量，提出了一种新的加权比特信息提取方法HammerLeak。HammerLeak利用针对DNN应用的几种新颖的系统级技术来实现快速高效的重量盗窃。其次，提出了一种基于均值聚类权重惩罚的替身模型训练算法，该算法有效地利用了部分泄露的比特信息，生成了目标受害者模型的替身原型。我们在三个流行的图像数据集(如CIFAR10/10 0/GTSRB)和四个数字近邻结构(如Resnet-18/34/Wide-Resnet/VGG-11)上对该替身模型提取方法进行了评估。所提取的替身模型在CIFAR-10数据集上的深层残差网络上的测试准确率已成功达到90%以上。此外，我们提取的替身模型还可以生成有效的敌意输入样本来愚弄受害者模型。



## **38. Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training**

安全胜过遗憾：通过对抗性训练防止妄想对手 cs.LG

NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2102.04716v3)

**Authors**: Lue Tao, Lei Feng, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Delusive attacks aim to substantially deteriorate the test accuracy of the learning model by slightly perturbing the features of correctly labeled training examples. By formalizing this malicious attack as finding the worst-case training data within a specific $\infty$-Wasserstein ball, we show that minimizing adversarial risk on the perturbed data is equivalent to optimizing an upper bound of natural risk on the original data. This implies that adversarial training can serve as a principled defense against delusive attacks. Thus, the test accuracy decreased by delusive attacks can be largely recovered by adversarial training. To further understand the internal mechanism of the defense, we disclose that adversarial training can resist the delusive perturbations by preventing the learner from overly relying on non-robust features in a natural setting. Finally, we complement our theoretical findings with a set of experiments on popular benchmark datasets, which show that the defense withstands six different practical attacks. Both theoretical and empirical results vote for adversarial training when confronted with delusive adversaries.

摘要: 妄想攻击的目的是通过对正确标记的训练样本的特征进行轻微扰动来显著降低学习模型的测试精度。通过将这种恶意攻击形式化为在特定的$\infty$-Wasserstein球中寻找最坏情况的训练数据，我们表明最小化扰动数据上的敌意风险等价于优化原始数据上的自然风险上界。这意味着对抗性训练可以作为对抗妄想攻击的原则性防御。因此，通过对抗性训练可以在很大程度上恢复由于妄想攻击而降低的测试精度。为了进一步了解防御的内在机制，我们揭示了对抗性训练可以通过防止学习者在自然环境中过度依赖非鲁棒特征来抵抗妄想干扰。最后，我们通过在流行的基准数据集上的一组实验来补充我们的理论发现，这些实验表明该防御系统可以抵御六种不同的实际攻击。在面对妄想性对手时，无论是理论结果还是经验结果都支持对抗性训练。



## **39. Robust and Information-theoretically Safe Bias Classifier against Adversarial Attacks**

抗敌意攻击的稳健且信息理论安全的偏向分类器 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04404v1)

**Authors**: Lijia Yu, Xiao-Shan Gao

**Abstracts**: In this paper, the bias classifier is introduced, that is, the bias part of a DNN with Relu as the activation function is used as a classifier. The work is motivated by the fact that the bias part is a piecewise constant function with zero gradient and hence cannot be directly attacked by gradient-based methods to generate adversaries such as FGSM. The existence of the bias classifier is proved an effective training method for the bias classifier is proposed. It is proved that by adding a proper random first-degree part to the bias classifier, an information-theoretically safe classifier against the original-model gradient-based attack is obtained in the sense that the attack generates a totally random direction for generating adversaries. This seems to be the first time that the concept of information-theoretically safe classifier is proposed. Several attack methods for the bias classifier are proposed and numerical experiments are used to show that the bias classifier is more robust than DNNs against these attacks in most cases.

摘要: 本文介绍了偏向分类器，即以RELU为激活函数的DNN的偏向部分作为分类器。该工作的动机在于偏差部分是零梯度的分段常数函数，因此不能被基于梯度的方法直接攻击来生成诸如FGSM之类的对手。证明了偏向分类器的存在性，提出了一种有效的偏向分类器训练方法。证明了通过在偏向分类器中增加适当的随机一阶部分，在攻击产生一个完全随机的攻击方向的意义下，得到了一个针对原始模型梯度攻击的信息论安全的分类器。这似乎是首次提出信息理论安全分类器的概念。提出了几种针对偏向分类器的攻击方法，并通过数值实验表明，在大多数情况下，偏向分类器比DNNs对这些攻击具有更强的鲁棒性。



## **40. Get a Model! Model Hijacking Attack Against Machine Learning Models**

找个模特来！针对机器学习模型的模型劫持攻击 cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04394v1)

**Authors**: Ahmed Salem, Michael Backes, Yang Zhang

**Abstracts**: Machine learning (ML) has established itself as a cornerstone for various critical applications ranging from autonomous driving to authentication systems. However, with this increasing adoption rate of machine learning models, multiple attacks have emerged. One class of such attacks is training time attack, whereby an adversary executes their attack before or during the machine learning model training. In this work, we propose a new training time attack against computer vision based machine learning models, namely model hijacking attack. The adversary aims to hijack a target model to execute a different task than its original one without the model owner noticing. Model hijacking can cause accountability and security risks since a hijacked model owner can be framed for having their model offering illegal or unethical services. Model hijacking attacks are launched in the same way as existing data poisoning attacks. However, one requirement of the model hijacking attack is to be stealthy, i.e., the data samples used to hijack the target model should look similar to the model's original training dataset. To this end, we propose two different model hijacking attacks, namely Chameleon and Adverse Chameleon, based on a novel encoder-decoder style ML model, namely the Camouflager. Our evaluation shows that both of our model hijacking attacks achieve a high attack success rate, with a negligible drop in model utility.

摘要: 机器学习(ML)已经成为从自动驾驶到身份验证系统等各种关键应用的基石。然而，随着机器学习模型采用率的不断提高，出现了多种攻击。这种攻击的一类是训练时间攻击，由此对手在机器学习模型训练之前或期间执行他们的攻击。在这项工作中，我们提出了一种新的针对基于计算机视觉的机器学习模型的训练时间攻击，即模型劫持攻击。敌手的目标是劫持目标模型，以便在模型所有者不察觉的情况下执行与其原始任务不同的任务。劫持模特可能会导致责任和安全风险，因为被劫持的模特所有者可能会因为让他们的模特提供非法或不道德的服务而被陷害。模型劫持攻击的发起方式与现有的数据中毒攻击方式相同。然而，模型劫持攻击的一个要求是隐蔽性，即用于劫持目标模型的数据样本应该与模型的原始训练数据集相似。为此，我们基于一种新的编解码器风格的ML模型，即伪装器，提出了两种不同模型的劫持攻击，即变色龙攻击和逆变色龙攻击。我们的评估表明，我们的两种模型劫持攻击都达到了很高的攻击成功率，而模型效用的下降可以忽略不计。



## **41. Geometrically Adaptive Dictionary Attack on Face Recognition**

人脸识别中的几何自适应字典攻击 cs.CV

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04371v1)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: CNN-based face recognition models have brought remarkable performance improvement, but they are vulnerable to adversarial perturbations. Recent studies have shown that adversaries can fool the models even if they can only access the models' hard-label output. However, since many queries are needed to find imperceptible adversarial noise, reducing the number of queries is crucial for these attacks. In this paper, we point out two limitations of existing decision-based black-box attacks. We observe that they waste queries for background noise optimization, and they do not take advantage of adversarial perturbations generated for other images. We exploit 3D face alignment to overcome these limitations and propose a general strategy for query-efficient black-box attacks on face recognition named Geometrically Adaptive Dictionary Attack (GADA). Our core idea is to create an adversarial perturbation in the UV texture map and project it onto the face in the image. It greatly improves query efficiency by limiting the perturbation search space to the facial area and effectively recycling previous perturbations. We apply the GADA strategy to two existing attack methods and show overwhelming performance improvement in the experiments on the LFW and CPLFW datasets. Furthermore, we also present a novel attack strategy that can circumvent query similarity-based stateful detection that identifies the process of query-based black-box attacks.

摘要: 基于CNN的人脸识别模型带来了显著的性能提升，但它们容易受到对手的干扰。最近的研究表明，即使对手只能访问模型的硬标签输出，他们也可以愚弄模型。然而，由于需要大量的查询来发现不可察觉的对抗性噪声，因此减少查询的数量对这些攻击至关重要。在本文中，我们指出了现有基于决策的黑盒攻击的两个局限性。我们观察到，它们将查询浪费在背景噪声优化上，并且它们没有利用为其他图像生成的对抗性扰动。我们利用三维人脸对齐来克服这些限制，并提出了一种通用的人脸识别黑盒攻击策略，称为几何自适应字典攻击(GADA)。我们的核心想法是在UV纹理贴图中创建对抗性扰动，并将其投影到图像中的脸部。通过将扰动搜索空间限制在人脸区域，并有效地循环使用先前的扰动，极大地提高了查询效率。我们将GADA策略应用于现有的两种攻击方法，在LFW和CPLFW数据集上的实验表明，GADA策略的性能有了显著的提高。此外，我们还提出了一种新的攻击策略，可以规避基于查询相似度的状态检测，识别基于查询的黑盒攻击过程。



## **42. Robustness of Graph Neural Networks at Scale**

图神经网络在尺度上的鲁棒性 cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2110.14038v3)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstracts**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.

摘要: 图神经网络(GNNs)因其普及性和应用的多样性而变得越来越重要。然而，现有的关于它们易受敌意攻击的研究依赖于相对较小的图表。我们解决了这一差距，并研究了如何大规模攻击和防御GNN。我们提出了两种稀疏性感知的一阶优化攻击，这两种攻击在对节点数目为二次的多个参数进行优化的情况下仍能保持有效的表示。我们证明了常见的代理损失并不能很好地适用于针对GNNs的全局攻击。我们的替代方案可以使攻击强度加倍。此外，为了提高GNNs的可靠性，我们设计了一个健壮的聚集函数--软中值，从而在所有尺度上都能进行有效的防御。与以前的工作相比，我们在大于100倍的图上使用标准GNN来评估我们的攻击和防御。我们甚至通过将我们的技术扩展到可伸缩的GNN来进一步扩展一个数量级。



## **43. Characterizing the adversarial vulnerability of speech self-supervised learning**

语音自监督学习的对抗性脆弱性表征 cs.SD

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04330v1)

**Authors**: Haibin Wu, Bo Zheng, Xu Li, Xixin Wu, Hung-yi Lee, Helen Meng

**Abstracts**: A leaderboard named Speech processing Universal PERformance Benchmark (SUPERB), which aims at benchmarking the performance of a shared self-supervised learning (SSL) speech model across various downstream speech tasks with minimal modification of architectures and small amount of data, has fueled the research for speech representation learning. The SUPERB demonstrates speech SSL upstream models improve the performance of various downstream tasks through just minimal adaptation. As the paradigm of the self-supervised learning upstream model followed by downstream tasks arouses more attention in the speech community, characterizing the adversarial robustness of such paradigm is of high priority. In this paper, we make the first attempt to investigate the adversarial vulnerability of such paradigm under the attacks from both zero-knowledge adversaries and limited-knowledge adversaries. The experimental results illustrate that the paradigm proposed by SUPERB is seriously vulnerable to limited-knowledge adversaries, and the attacks generated by zero-knowledge adversaries are with transferability. The XAB test verifies the imperceptibility of crafted adversarial attacks.

摘要: 一个名为语音处理通用性能基准(SUBB)的排行榜推动了语音表示学习的研究，该基准测试旨在以最小的体系结构和少量的数据对共享的自监督学习(SSL)语音模型在各种下游语音任务中的性能进行基准测试。出色的演示了语音SSL上行模型通过最小程度的适配提高了各种下行任务的性能。随着上游自我监督学习模型和下游任务的范式越来越受到语言学界的关注，表征这种范式的对抗性鲁棒性是当务之急。在本文中，我们首次尝试研究了该范式在零知识和有限知识两种攻击下的攻击脆弱性。实验结果表明，Superb提出的范式对有限知识的攻击具有很强的脆弱性，零知识攻击产生的攻击具有可移植性。Xab测试验证精心设计的敌意攻击的隐蔽性。



## **44. Graph Robustness Benchmark: Benchmarking the Adversarial Robustness of Graph Machine Learning**

图健壮性基准：对图机器学习的对抗性健壮性进行基准测试 cs.LG

21 pages, 12 figures, NeurIPS 2021 Datasets and Benchmarks Track

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04314v1)

**Authors**: Qinkai Zheng, Xu Zou, Yuxiao Dong, Yukuo Cen, Da Yin, Jiarong Xu, Yang Yang, Jie Tang

**Abstracts**: Adversarial attacks on graphs have posed a major threat to the robustness of graph machine learning (GML) models. Naturally, there is an ever-escalating arms race between attackers and defenders. However, the strategies behind both sides are often not fairly compared under the same and realistic conditions. To bridge this gap, we present the Graph Robustness Benchmark (GRB) with the goal of providing a scalable, unified, modular, and reproducible evaluation for the adversarial robustness of GML models. GRB standardizes the process of attacks and defenses by 1) developing scalable and diverse datasets, 2) modularizing the attack and defense implementations, and 3) unifying the evaluation protocol in refined scenarios. By leveraging the GRB pipeline, the end-users can focus on the development of robust GML models with automated data processing and experimental evaluations. To support open and reproducible research on graph adversarial learning, GRB also hosts public leaderboards across different scenarios. As a starting point, we conduct extensive experiments to benchmark baseline techniques. GRB is open-source and welcomes contributions from the community. Datasets, codes, leaderboards are available at https://cogdl.ai/grb/home.

摘要: 图的敌意攻击已经成为图机器学习(GML)模型健壮性的主要威胁。当然，攻击者和防御者之间的军备竞赛不断升级。然而，在相同的现实条件下，双方背后的战略往往是不公平的比较。为了弥补这一差距，我们提出了图健壮性基准(GRB)，目的是为GML模型的对抗健壮性提供一个可扩展的、统一的、模块化的和可重现的评估。GRB通过1)开发可扩展和多样化的数据集，2)将攻击和防御实现模块化，3)在细化的场景中统一评估协议，从而标准化了攻击和防御的过程。通过利用GRB管道，最终用户可以专注于开发具有自动数据处理和实验评估功能的健壮GML模型。为了支持关于图形对抗性学习的开放和可重复的研究，GRB还在不同的场景中主持公共排行榜。作为起点，我们进行了广泛的实验来对基线技术进行基准测试。GRB是开源的，欢迎来自社区的贡献。有关数据集、代码和排行榜的信息，请访问https://cogdl.ai/grb/home.



## **45. DeepMoM: Robust Deep Learning With Median-of-Means**

DeepMoM：基于均值中值的稳健深度学习 stat.ML

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2105.14035v2)

**Authors**: Shih-Ting Huang, Johannes Lederer

**Abstracts**: Data used in deep learning is notoriously problematic. For example, data are usually combined from diverse sources, rarely cleaned and vetted thoroughly, and sometimes corrupted on purpose. Intentional corruption that targets the weak spots of algorithms has been studied extensively under the label of "adversarial attacks." In contrast, the arguably much more common case of corruption that reflects the limited quality of data has been studied much less. Such "random" corruptions are due to measurement errors, unreliable sources, convenience sampling, and so forth. These kinds of corruption are common in deep learning, because data are rarely collected according to strict protocols -- in strong contrast to the formalized data collection in some parts of classical statistics. This paper concerns such corruption. We introduce an approach motivated by very recent insights into median-of-means and Le Cam's principle, we show that the approach can be readily implemented, and we demonstrate that it performs very well in practice. In conclusion, we believe that our approach is a very promising alternative to standard parameter training based on least-squares and cross-entropy loss.

摘要: 深度学习中使用的数据是出了名的问题。例如，数据通常来自不同的来源，很少被彻底清理和审查，有时还会被故意破坏。针对算法弱点的故意腐败已经在“对抗性攻击”的标签下进行了广泛的研究。相比之下，可以说更常见的反映数据质量有限的腐败案件的研究要少得多。这种“随机”损坏是由于测量误差、来源不可靠、采样方便等原因造成的。这种类型的损坏在深度学习中很常见，因为数据很少根据严格的协议收集--这与经典统计中某些部分的形式化数据收集形成了强烈对比。本文关注的是这样的腐败现象。我们介绍了一种基于对均值中位数和Le Cam原理的最新见解的方法，我们证明了该方法可以很容易地实现，并且我们证明了它在实践中表现得非常好。总之，我们认为我们的方法是一种非常有前途的替代基于最小二乘和交叉熵损失的标准参数训练的方法。



## **46. On the Effectiveness of Small Input Noise for Defending Against Query-based Black-Box Attacks**

小输入噪声对抵抗基于查询的黑盒攻击的有效性研究 cs.CR

Accepted at WACV 2022

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2101.04829v2)

**Authors**: Junyoung Byun, Hyojun Go, Changick Kim

**Abstracts**: While deep neural networks show unprecedented performance in various tasks, the vulnerability to adversarial examples hinders their deployment in safety-critical systems. Many studies have shown that attacks are also possible even in a black-box setting where an adversary cannot access the target model's internal information. Most black-box attacks are based on queries, each of which obtains the target model's output for an input, and many recent studies focus on reducing the number of required queries. In this paper, we pay attention to an implicit assumption of query-based black-box adversarial attacks that the target model's output exactly corresponds to the query input. If some randomness is introduced into the model, it can break the assumption, and thus, query-based attacks may have tremendous difficulty in both gradient estimation and local search, which are the core of their attack process. From this motivation, we observe even a small additive input noise can neutralize most query-based attacks and name this simple yet effective approach Small Noise Defense (SND). We analyze how SND can defend against query-based black-box attacks and demonstrate its effectiveness against eight state-of-the-art attacks with CIFAR-10 and ImageNet datasets. Even with strong defense ability, SND almost maintains the original classification accuracy and computational speed. SND is readily applicable to pre-trained models by adding only one line of code at the inference.

摘要: 虽然深度神经网络在各种任务中表现出前所未有的性能，但对敌意示例的脆弱性阻碍了它们在安全关键系统中的部署。许多研究表明，即使在对手无法访问目标模型内部信息的黑盒设置中，攻击也是可能的。大多数黑盒攻击都是基于查询的，每个查询都会获取目标模型的输出作为输入，最近的许多研究都集中在减少所需查询的数量上。在本文中，我们注意到基于查询的黑盒对抗攻击的一个隐含假设，即目标模型的输出与查询输入精确对应。如果在模型中引入一定的随机性，可能会打破假设，因此，基于查询的攻击在梯度估计和局部搜索这两个攻击过程的核心上都可能会有很大的困难。从这一动机出发，我们观察到即使是很小的加性输入噪声也可以中和大多数基于查询的攻击，并将这种简单而有效的方法命名为小噪声防御(SND)。我们分析了SND如何防御基于查询的黑盒攻击，并使用CIFAR-10和ImageNet数据集验证了它对八种最先进的攻击的有效性。即使具有很强的防御能力，SND也几乎保持了原有的分类精度和计算速度。通过在推理处只添加一行代码，SND很容易适用于预先训练的模型。



## **47. Defense Against Explanation Manipulation**

对解释操纵的防御 cs.LG

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.04303v1)

**Authors**: Ruixiang Tang, Ninghao Liu, Fan Yang, Na Zou, Xia Hu

**Abstracts**: Explainable machine learning attracts increasing attention as it improves transparency of models, which is helpful for machine learning to be trusted in real applications. However, explanation methods have recently been demonstrated to be vulnerable to manipulation, where we can easily change a model's explanation while keeping its prediction constant. To tackle this problem, some efforts have been paid to use more stable explanation methods or to change model configurations. In this work, we tackle the problem from the training perspective, and propose a new training scheme called Adversarial Training on EXplanations (ATEX) to improve the internal explanation stability of a model regardless of the specific explanation method being applied. Instead of directly specifying explanation values over data instances, ATEX only puts requirement on model predictions which avoids involving second-order derivatives in optimization. As a further discussion, we also find that explanation stability is closely related to another property of the model, i.e., the risk of being exposed to adversarial attack. Through experiments, besides showing that ATEX improves model robustness against manipulation targeting explanation, it also brings additional benefits including smoothing explanations and improving the efficacy of adversarial training if applied to the model.

摘要: 可解释机器学习由于提高了模型的透明性而受到越来越多的关注，这有助于机器学习在实际应用中得到信任。然而，最近已经证明解释方法容易受到操纵，在这些方法中，我们可以很容易地改变模型的解释，同时保持其预测不变。为了解决撞击的这一问题，已经做出了一些努力，使用更稳定的解释方法或改变模型配置。在这项工作中，我们从训练的角度对这一问题进行了撞击研究，并提出了一种新的训练方案，称为对抗性解释训练(ATEX)，以提高模型的内部解释稳定性，而不考虑具体的解释方法。ATEX没有直接指定数据实例上的解释值，而是只对模型预测提出了要求，避免了优化中涉及二阶导数的问题。作为进一步的讨论，我们还发现解释稳定性与模型的另一个性质，即暴露于敌意攻击的风险密切相关。通过实验表明，ATEX除了提高了模型对操作目标解释的鲁棒性外，如果将其应用到模型中，还可以带来平滑解释和提高对抗性训练效果等额外的好处。



## **48. A Unified Game-Theoretic Interpretation of Adversarial Robustness**

对抗性稳健性的统一博弈论解释 cs.LG

the previous version is arXiv:2103.07364, but I mistakenly apply a  new ID for the paper

**SubmitDate**: 2021-11-08    [paper-pdf](http://arxiv.org/pdf/2111.03536v2)

**Authors**: Jie Ren, Die Zhang, Yisen Wang, Lu Chen, Zhanpeng Zhou, Yiting Chen, Xu Cheng, Xin Wang, Meng Zhou, Jie Shi, Quanshi Zhang

**Abstracts**: This paper provides a unified view to explain different adversarial attacks and defense methods, \emph{i.e.} the view of multi-order interactions between input variables of DNNs. Based on the multi-order interaction, we discover that adversarial attacks mainly affect high-order interactions to fool the DNN. Furthermore, we find that the robustness of adversarially trained DNNs comes from category-specific low-order interactions. Our findings provide a potential method to unify adversarial perturbations and robustness, which can explain the existing defense methods in a principle way. Besides, our findings also make a revision of previous inaccurate understanding of the shape bias of adversarially learned features.

摘要: 本文提供了一个统一的视角来解释不同的对抗性攻击和防御方法，即DNNs的输入变量之间的多阶交互的观点。基于多阶交互，我们发现对抗性攻击主要影响高阶交互来欺骗DNN。此外，我们发现对抗性训练的DNN的鲁棒性来自于特定类别的低阶交互。我们的发现为统一对抗性扰动和鲁棒性提供了一种潜在的方法，可以对现有的防御方法进行原则性的解释。此外，我们的发现还修正了以往对对抗性习得特征的形状偏向的不准确理解。



## **49. Natural Adversarial Objects**

自然对抗性客体 cs.CV

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2111.04204v1)

**Authors**: Felix Lau, Nishant Subramani, Sasha Harrison, Aerin Kim, Elliot Branson, Rosanne Liu

**Abstracts**: Although state-of-the-art object detection methods have shown compelling performance, models often are not robust to adversarial attacks and out-of-distribution data. We introduce a new dataset, Natural Adversarial Objects (NAO), to evaluate the robustness of object detection models. NAO contains 7,934 images and 9,943 objects that are unmodified and representative of real-world scenarios, but cause state-of-the-art detection models to misclassify with high confidence. The mean average precision (mAP) of EfficientDet-D7 drops 74.5% when evaluated on NAO compared to the standard MSCOCO validation set.   Moreover, by comparing a variety of object detection architectures, we find that better performance on MSCOCO validation set does not necessarily translate to better performance on NAO, suggesting that robustness cannot be simply achieved by training a more accurate model.   We further investigate why examples in NAO are difficult to detect and classify. Experiments of shuffling image patches reveal that models are overly sensitive to local texture. Additionally, using integrated gradients and background replacement, we find that the detection model is reliant on pixel information within the bounding box, and insensitive to the background context when predicting class labels. NAO can be downloaded at https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.

摘要: 虽然最先进的目标检测方法已经显示出令人信服的性能，但模型通常对敌意攻击和分布外的数据并不健壮。我们引入了一个新的数据集--自然对抗性对象(NAO)来评估目标检测模型的健壮性。NAO包含7934张图像和9943个对象，这些图像和对象未经修改，可以代表真实世界的场景，但会导致最先进的检测模型高度可信地错误分类。与标准MSCOCO验证集相比，在NAO上评估EfficientDet-D7的平均平均精度(MAP)下降了74.5%。此外，通过比较各种目标检测体系结构，我们发现在MSCOCO验证集上更好的性能并不一定转化为在NAO上更好的性能，这表明鲁棒性不能简单地通过训练更精确的模型来实现。我们进一步调查了为什么NAO中的例子很难检测和分类。混洗图像块的实验表明，模型对局部纹理过于敏感。此外，通过使用集成梯度和背景替换，我们发现该检测模型依赖于边界框内的像素信息，并且在预测类别标签时对背景上下文不敏感。NAO可从https://drive.google.com/drive/folders/15P8sOWoJku6SSEiHLEts86ORfytGezi8.下载



## **50. Adversarial Attacks on Multi-task Visual Perception for Autonomous Driving**

自主驾驶多任务视觉感知的对抗性攻击 cs.CV

Accepted for publication at Journal of Imaging Science and Technology

**SubmitDate**: 2021-11-07    [paper-pdf](http://arxiv.org/pdf/2107.07449v2)

**Authors**: Ibrahim Sobh, Ahmed Hamed, Varun Ravi Kumar, Senthil Yogamani

**Abstracts**: Deep neural networks (DNNs) have accomplished impressive success in various applications, including autonomous driving perception tasks, in recent years. On the other hand, current deep neural networks are easily fooled by adversarial attacks. This vulnerability raises significant concerns, particularly in safety-critical applications. As a result, research into attacking and defending DNNs has gained much coverage. In this work, detailed adversarial attacks are applied on a diverse multi-task visual perception deep network across distance estimation, semantic segmentation, motion detection, and object detection. The experiments consider both white and black box attacks for targeted and un-targeted cases, while attacking a task and inspecting the effect on all the others, in addition to inspecting the effect of applying a simple defense method. We conclude this paper by comparing and discussing the experimental results, proposing insights and future work. The visualizations of the attacks are available at https://youtu.be/6AixN90budY.

摘要: 近年来，深度神经网络(DNNs)在包括自主驾驶感知任务在内的各种应用中取得了令人印象深刻的成功。另一方面，当前的深度神经网络很容易被敌意攻击所欺骗。此漏洞引起了严重关注，尤其是在安全关键型应用程序中。因此，攻击和防御DNN的研究得到了广泛的报道。在这项工作中，详细的对抗性攻击应用于一个跨越距离估计、语义分割、运动检测和目标检测的多样化的多任务视觉感知深度网络。实验同时考虑了针对目标和非目标情况的白盒攻击和黑盒攻击，同时攻击一个任务并检查对所有其他任务的影响，此外还检查了应用简单防御方法的效果。最后，通过对实验结果的比较和讨论，总结了本文的研究成果，并提出了自己的见解和未来的工作方向。这些攻击的可视化可在https://youtu.be/6AixN90budY.上查看。



