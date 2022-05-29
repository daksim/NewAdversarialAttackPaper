# Latest Adversarial Attack Papers
**update at 2022-05-30 06:31:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

PerDoor：使用对抗性扰动的联合学习中持久的非一致后门 cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13523v1)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.

摘要: 联合学习(FL)使众多参与者能够协作地训练深度学习模型，而不会暴露他们的个人、潜在敏感数据，使其成为协作培训中数据隐私的一种有前途的解决方案。然而，FL和未经审查的数据的分布式性质使其天生就容易受到后门攻击：在这种情况下，对手在训练期间向集中式模型注入后门功能，这可能会被触发，导致对特定对手选择的输入造成所需的错误分类。先前的一系列工作在FL系统中建立了成功的后门注入；然而，这些后门并没有被证明是持久的。如果将对手从训练过程中移除，则后门功能不会保留在系统中，因为集中式模型参数在连续的FL训练轮期间不断变化。因此，在这项工作中，我们提出了PerDoor，这是一种持久的构造后门注入技术，受对手扰动和集中式模型的目标参数的驱动，这些参数在连续的FL轮中偏离较小，对主任务精度的贡献最小。与传统的后门攻击相比，考虑图像分类场景的详尽评估描绘了在多个FL轮上平均花费10.5\x$持久性。通过实验，我们进一步展示了PerDoor在FL系统中存在最先进的后门预防技术时的有效性。此外，对抗性扰动的操作还有助于PerDoor为后门输入开发非统一的触发模式，而不是现有后门技术的统一触发(具有固定的模式和位置)，后者容易被缓解。



## **2. An Analytic Framework for Robust Training of Artificial Neural Networks**

一种神经网络稳健训练的分析框架 cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13502v1)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.

摘要: 学习模型的可靠性是机器学习在各个行业成功部署的关键。创建一个健壮的模型，特别是一个不受对抗性攻击影响的模型，需要对对抗性例子现象有一个全面的了解。然而，由于机器学习中问题的复杂性，这一现象很难描述。因此，许多研究通过提出对抗性例子如何发生的简化模型来研究这一现象，并通过预测该现象的某些方面来验证该模型。虽然这些研究涵盖了对抗性例子的许多不同特征，但它们还没有达成对这一现象的几何和解析建模的整体方法。本文提出了一种学习理论中研究这一现象的形式化框架，并利用复分析和全纯理论为人工神经网络提供了一种稳健的学习规则。在复杂分析的帮助下，我们可以毫不费力地在现象的几何和解析视角之间切换，并通过揭示它与调和函数的联系来提供对该现象的进一步见解。使用我们的模型，我们可以解释对抗性例子的一些最有趣的特征，包括对抗性例子的可转移性，并为缓解这一现象的影响的新方法铺平道路。



## **3. Towards Understanding and Harnessing the Effect of Image Transformation in Adversarial Detection**

关于理解和利用图像变换在对抗检测中的作用 cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2201.01080v3)

**Authors**: Hui Liu, Bo Zhao, Yuefeng Peng, Weidong Li, Peng Liu

**Abstracts**: Deep neural networks (DNNs) are threatened by adversarial examples. Adversarial detection, which distinguishes adversarial images from benign images, is fundamental for robust DNN-based services. Image transformation is one of the most effective approaches to detect adversarial examples. During the last few years, a variety of image transformations have been studied and discussed to design reliable adversarial detectors. In this paper, we systematically synthesize the recent progress on adversarial detection via image transformations with a novel classification method. Then, we conduct extensive experiments to test the detection performance of image transformations against state-of-the-art adversarial attacks. Furthermore, we reveal that each individual transformation is not capable of detecting adversarial examples in a robust way, and propose a DNN-based approach referred to as \emph{AdvJudge}, which combines scores of 9 image transformations. Without knowing which individual scores are misleading or not misleading, AdvJudge can make the right judgment, and achieve a significant improvement in detection rate. Finally, we utilize an explainable AI tool to show the contribution of each image transformation to adversarial detection. Experimental results show that the contribution of image transformations to adversarial detection is significantly different, the combination of them can significantly improve the generic detection ability against state-of-the-art adversarial attacks.

摘要: 深度神经网络(DNN)受到敌意例子的威胁。敌意检测是基于DNN的稳健服务的基础，它区分敌意图像和良性图像。图像变换是检测敌意例子最有效的方法之一。在过去的几年里，人们已经研究和讨论了各种图像变换来设计可靠的对抗性检测器。本文采用一种新的分类方法，系统地综述了基于图像变换的对抗性检测的最新研究进展。然后，我们进行了大量的实验来测试图像变换对最新的敌意攻击的检测性能。此外，我们揭示了每个个体变换并不能稳健地检测敌意示例，并提出了一种基于DNN的方法，该方法结合了9个图像变换的分数。在不知道哪些分数具有误导性或哪些不具有误导性的情况下，AdvJustice能够做出正确的判断，并实现了检测率的显著提高。最后，我们利用一个可解释的人工智能工具来展示每个图像变换对对抗检测的贡献。实验结果表明，图像变换对敌意检测的贡献是不同的，它们的结合可以显著提高对最新敌意攻击的一般检测能力。



## **4. A Physical-World Adversarial Attack Against 3D Face Recognition**

一种针对3D人脸识别的物理世界对抗性攻击 cs.CV

10 pages, 5 figures, Submit to NeurIPS 2022

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13412v1)

**Authors**: Yanjie Li, Yiquan Li, Bin Xiao

**Abstracts**: 3D face recognition systems have been widely employed in intelligent terminals, among which structured light imaging is a common method to measure the 3D shape. However, this method could be easily attacked, leading to inaccurate 3D face recognition. In this paper, we propose a novel, physically-achievable attack on the fringe structured light system, named structured light attack. The attack utilizes a projector to project optical adversarial fringes on faces to generate point clouds with well-designed noises. We firstly propose a 3D transform-invariant loss function to enhance the robustness of 3D adversarial examples in the physical-world attack. Then we reverse the 3D adversarial examples to the projector's input to place noises on phase-shift images, which models the process of structured light imaging. A real-world structured light system is constructed for the attack and several state-of-the-art 3D face recognition neural networks are tested. Experiments show that our method can attack the physical system successfully and only needs minor modifications of projected images.

摘要: 三维人脸识别系统在智能终端中得到了广泛的应用，其中结构光成像是一种常用的三维形状测量方法。然而，这种方法容易受到攻击，导致3D人脸识别不准确。在本文中，我们提出了一种新颖的、物理上可实现的对条纹结构光系统的攻击，称为结构光攻击。该攻击利用投影仪将光学对抗性条纹投射到人脸上，以生成具有精心设计的噪声的点云。本文首次提出了一种3D变换不变损失函数，以增强3D对抗实例在物理世界攻击中的稳健性。然后，我们将3D对抗性的例子反转到投影仪的输入上，在相移图像上放置噪声，这是对结构光成像过程的模拟。针对攻击构建了一个真实世界的结构光系统，并测试了几种最先进的3D人脸识别神经网络。实验表明，该方法可以成功地攻击物理系统，并且只需要对投影图像进行很小的修改。



## **5. BppAttack: Stealthy and Efficient Trojan Attacks against Deep Neural Networks via Image Quantization and Contrastive Adversarial Learning**

BppAttack：基于图像量化和对比性学习的隐蔽高效木马攻击深度神经网络 cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13383v1)

**Authors**: Zhenting Wang, Juan Zhai, Shiqing Ma

**Abstracts**: Deep neural networks are vulnerable to Trojan attacks. Existing attacks use visible patterns (e.g., a patch or image transformations) as triggers, which are vulnerable to human inspection. In this paper, we propose stealthy and efficient Trojan attacks, BppAttack. Based on existing biology literature on human visual systems, we propose to use image quantization and dithering as the Trojan trigger, making imperceptible changes. It is a stealthy and efficient attack without training auxiliary models. Due to the small changes made to images, it is hard to inject such triggers during training. To alleviate this problem, we propose a contrastive learning based approach that leverages adversarial attacks to generate negative sample pairs so that the learned trigger is precise and accurate. The proposed method achieves high attack success rates on four benchmark datasets, including MNIST, CIFAR-10, GTSRB, and CelebA. It also effectively bypasses existing Trojan defenses and human inspection. Our code can be found in https://github.com/RU-System-Software-and-Security/BppAttack.

摘要: 深度神经网络容易受到特洛伊木马的攻击。现有攻击使用可见模式(例如，补丁或图像转换)作为触发器，这容易受到人工检查。本文提出了一种隐蔽高效的特洛伊木马攻击BppAttack。基于现有关于人类视觉系统的生物学文献，我们提出使用图像量化和抖动作为特洛伊木马的触发器，进行潜移默化的改变。这是一种不需要训练辅助模型的隐身而有效的攻击。由于图像的微小变化，在训练中很难注入这样的触发因素。为了缓解这一问题，我们提出了一种基于对比学习的方法，该方法利用对抗性攻击来生成负样本对，从而学习到的触发器是精确和准确的。该方法在MNIST、CIFAR-10、GTSRB和CelebA四个基准数据集上取得了较高的攻击成功率。它还有效地绕过了现有的特洛伊木马防御和人工检查。我们的代码可以在https://github.com/RU-System-Software-and-Security/BppAttack.中找到



## **6. Denial-of-Service Attacks on Learned Image Compression**

针对学习图像压缩的拒绝服务攻击 cs.CV

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13253v1)

**Authors**: Kang Liu, Di Wu, Yiru Wang, Dan Feng, Benjamin Tan, Siddharth Garg

**Abstracts**: Deep learning techniques have shown promising results in image compression, with competitive bitrate and image reconstruction quality from compressed latent. However, while image compression has progressed towards higher peak signal-to-noise ratio (PSNR) and fewer bits per pixel (bpp), their robustness to corner-case images has never received deliberation. In this work, we, for the first time, investigate the robustness of image compression systems where imperceptible perturbation of input images can precipitate a significant increase in the bitrate of their compressed latent. To characterize the robustness of state-of-the-art learned image compression, we mount white and black-box attacks. Our results on several image compression models with various bitrate qualities show that they are surprisingly fragile, where the white-box attack achieves up to 56.326x and black-box 1.947x bpp change. To improve robustness, we propose a novel model which incorporates attention modules and a basic factorized entropy model, resulting in a promising trade-off between the PSNR/bpp ratio and robustness to adversarial attacks that surpasses existing learned image compressors.

摘要: 深度学习技术在图像压缩方面取得了很好的效果，压缩后的潜伏期具有较高的比特率和图像重建质量。然而，虽然图像压缩已经朝着更高的峰值信噪比(PSNR)和更少的每像素位(BPP)发展，但它们对角点图像的稳健性从未得到深思熟虑。在这项工作中，我们首次研究了图像压缩系统的稳健性，在这种情况下，输入图像的不可察觉的扰动可以导致其压缩潜伏期的比特率显著增加。为了表征最先进的学习图像压缩的稳健性，我们安装了白盒和黑盒攻击。我们在几种不同码率质量的图像压缩模型上的结果表明，它们令人惊讶地脆弱，其中白盒攻击达到了56.326倍，黑盒攻击达到了1.947倍的BPP变化。为了提高鲁棒性，我们提出了一种新的模型，它结合了注意力模块和基本的因式分解熵模型，在PSNR/BPP比和对抗攻击的稳健性之间取得了很好的权衡，其性能超过了现有的学习图像压缩器。



## **7. Certified Robustness Against Natural Language Attacks by Causal Intervention**

通过因果干预验证对自然语言攻击的健壮性 cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.12331v2)

**Authors**: Haiteng Zhao, Chang Ma*, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstracts**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.

摘要: 深度学习模型在许多领域都取得了很大的成功，但它们很容易受到对手例子的影响。本文从因果关系的角度分析了敌意攻击的脆弱性，提出了通过语义平滑进行因果干预的方法，这是一种新的针对自然语言攻击的健壮性框架。与其仅仅对观测数据进行拟合，CIS通过在潜在语义空间中进行平滑以做出稳健的预测来学习因果效应p(y|do(X))，该预测可扩展到深层体系结构，并避免针对特定攻击定制的乏味的噪声构造。可以证明，该系统对单词替换攻击具有较强的健壮性，即使在未知攻击算法加强了扰动的情况下，也具有较强的经验性。例如，在Yelp上，在对单词替换的验证健壮性方面，CISS超过亚军6.7%，并且在整合句法攻击时获得了79.4%的经验健壮性。



## **8. Transferable Adversarial Attack based on Integrated Gradients**

基于集成梯度的可转移敌意攻击 cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13152v1)

**Authors**: Yi Huang, Adams Wai-Kin Kong

**Abstracts**: The vulnerability of deep neural networks to adversarial examples has drawn tremendous attention from the community. Three approaches, optimizing standard objective functions, exploiting attention maps, and smoothing decision surfaces, are commonly used to craft adversarial examples. By tightly integrating the three approaches, we propose a new and simple algorithm named Transferable Attack based on Integrated Gradients (TAIG) in this paper, which can find highly transferable adversarial examples for black-box attacks. Unlike previous methods using multiple computational terms or combining with other methods, TAIG integrates the three approaches into one single term. Two versions of TAIG that compute their integrated gradients on a straight-line path and a random piecewise linear path are studied. Both versions offer strong transferability and can seamlessly work together with the previous methods. Experimental results demonstrate that TAIG outperforms the state-of-the-art methods. The code will available at https://github.com/yihuang2016/TAIG

摘要: 深度神经网络对敌意例子的脆弱性已经引起了社会各界的极大关注。优化标准目标函数、利用注意图和平滑决策面这三种方法通常被用来制作对抗性例子。通过将这三种方法紧密地结合在一起，本文提出了一种新的简单的基于集成梯度的可转移攻击算法(TIAG)，该算法可以发现高度可转移的黑盒攻击对手实例。与以往使用多个计算项或与其他方法相结合的方法不同，TAIG将这三种方法集成到一个项中。研究了在直线路径和随机分段线性路径上计算其积分梯度的两个版本的TAIG。这两个版本都提供了很强的可移植性，并可以与之前的方法无缝配合使用。实验结果表明，TAIG的性能优于目前最先进的方法。该代码将在https://github.com/yihuang2016/TAIG上提供



## **9. Textual Backdoor Attacks with Iterative Trigger Injection**

使用迭代触发器注入的文本后门攻击 cs.CL

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12700v1)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstracts**: The backdoor attack has become an emerging threat for Natural Language Processing (NLP) systems. A victim model trained on poisoned data can be embedded with a "backdoor", making it predict the adversary-specified output (e.g., the positive sentiment label) on inputs satisfying the trigger pattern (e.g., containing a certain keyword). In this paper, we demonstrate that it's possible to design an effective and stealthy backdoor attack by iteratively injecting "triggers" into a small set of training data. While all triggers are common words that fit into the context, our poisoning process strongly associates them with the target label, forming the model backdoor. Experiments on sentiment analysis and hate speech detection show that our proposed attack is both stealthy and effective, raising alarm on the usage of untrusted training data. We further propose a defense method to combat this threat.

摘要: 后门攻击已经成为自然语言处理(NLP)系统的新威胁。在有毒数据上训练的受害者模型可以被嵌入有“后门”，使其在满足触发模式(例如，包含特定关键字)的输入上预测对手指定的输出(例如，积极情绪标签)。在这篇文章中，我们证明了通过迭代地向一小组训练数据注入“触发器”来设计有效和隐蔽的后门攻击是可能的。虽然所有触发器都是符合上下文的常见单词，但我们的中毒过程将它们与目标标签紧密关联，形成了模型后门。情感分析和仇恨语音检测实验表明，本文提出的攻击具有隐蔽性和有效性，对不可信训练数据的使用提出了警告。我们进一步提出了应对这种威胁的防御方法。



## **10. Deniable Steganography**

可否认隐写术 cs.CR

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12587v1)

**Authors**: Yong Xu, Zhihua Xia, Zichi Wang, Xinpeng Zhang, Jian Weng

**Abstracts**: Steganography conceals the secret message into the cover media, generating a stego media which can be transmitted on public channels without drawing suspicion. As its countermeasure, steganalysis mainly aims to detect whether the secret message is hidden in a given media. Although the steganography techniques are improving constantly, the sophisticated steganalysis can always break a known steganographic method to some extent. With a stego media discovered, the adversary could find out the sender or receiver and coerce them to disclose the secret message, which we name as coercive attack in this paper. Inspired by the idea of deniable encryption, we build up the concepts of deniable steganography for the first time and discuss the feasible constructions for it. As an example, we propose a receiver-deniable steganographic scheme to deal with the receiver-side coercive attack using deep neural networks (DNN). Specifically, besides the real secret message, a piece of fake message is also embedded into the cover. On the receiver side, the real message can be extracted with an extraction module; while once the receiver has to surrender a piece of secret message under coercive attack, he can extract the fake message to deceive the adversary with another extraction module. Experiments demonstrate the scalability and sensitivity of the DNN-based receiver-deniable steganographic scheme.

摘要: 隐写术将秘密信息隐藏在掩护媒体中，产生了一种可以在公共渠道上传输而不会引起怀疑的隐写媒体。作为其对策，隐写分析的主要目的是检测秘密信息是否隐藏在给定的媒体中。虽然隐写技术在不断改进，但复杂的隐写分析总能在一定程度上破解已知的隐写方法。发现隐写媒体后，攻击者可以找到发送者或接收者，并强迫他们泄露秘密信息，本文称之为强制攻击。受可否认加密思想的启发，我们首次提出了可否认隐写的概念，并讨论了它的可行构造。作为一个例子，我们提出了一种接收方可否认的隐写方案，利用深度神经网络(DNN)来应对接收方的强制攻击。具体地说，除了真实的秘密信息外，封面还嵌入了一条虚假信息。在接收方，可以使用提取模块来提取真实消息；而一旦接收方在强制攻击下不得不交出一条秘密消息，他可以使用另一个提取模块来提取虚假消息来欺骗对手。实验证明了基于DNN的接收方可否认隐写方案的可扩展性和敏感度。



## **11. Misleading Deep-Fake Detection with GAN Fingerprints**

基于GaN指纹的误导性深伪检测 cs.CV

In IEEE Deep Learning and Security Workshop (DLS) 2022

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12543v1)

**Authors**: Vera Wesselkamp, Konrad Rieck, Daniel Arp, Erwin Quiring

**Abstracts**: Generative adversarial networks (GANs) have made remarkable progress in synthesizing realistic-looking images that effectively outsmart even humans. Although several detection methods can recognize these deep fakes by checking for image artifacts from the generation process, multiple counterattacks have demonstrated their limitations. These attacks, however, still require certain conditions to hold, such as interacting with the detection method or adjusting the GAN directly. In this paper, we introduce a novel class of simple counterattacks that overcomes these limitations. In particular, we show that an adversary can remove indicative artifacts, the GAN fingerprint, directly from the frequency spectrum of a generated image. We explore different realizations of this removal, ranging from filtering high frequencies to more nuanced frequency-peak cleansing. We evaluate the performance of our attack with different detection methods, GAN architectures, and datasets. Our results show that an adversary can often remove GAN fingerprints and thus evade the detection of generated images.

摘要: 生成性对抗网络(GAN)在合成看起来逼真的图像方面取得了显着的进步，这些图像甚至有效地超过了人类。虽然几种检测方法可以通过检查生成过程中的图像伪影来识别这些深度伪像，但多次反击已经证明了它们的局限性。然而，这些攻击仍然需要一定的条件才能成立，例如与检测方法交互或直接调整GAN。在本文中，我们介绍了一类新的简单的反击，它克服了这些限制。具体地说，我们证明了攻击者可以直接从生成的图像的频谱中移除指示性伪像，即GaN指纹。我们探索了这种去除的不同实现，从过滤高频到更细微的频率峰值净化。我们使用不同的检测方法、GAN架构和数据集来评估我们的攻击的性能。我们的结果表明，攻击者经常可以移除GaN指纹，从而逃避生成图像的检测。



## **12. A Survey of Graph-Theoretic Approaches for Analyzing the Resilience of Networked Control Systems**

网络控制系统弹性分析的图论方法综述 eess.SY

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2205.12498v1)

**Authors**: Mohammad Pirani, Aritra Mitra, Shreyas Sundaram

**Abstracts**: As the scale of networked control systems increases and interactions between different subsystems become more sophisticated, questions of the resilience of such networks increase in importance. The need to redefine classical system and control-theoretic notions using the language of graphs has recently started to gain attention as a fertile and important area of research. This paper presents an overview of graph-theoretic methods for analyzing the resilience of networked control systems. We discuss various distributed algorithms operating on networked systems and investigate their resilience against adversarial actions by looking at the structural properties of their underlying networks. We present graph-theoretic methods to quantify the attack impact, and reinterpret some system-theoretic notions of robustness from a graph-theoretic standpoint to mitigate the impact of the attacks. Moreover, we discuss miscellaneous problems in the security of networked control systems which use graph-theory as a tool in their analyses. We conclude by introducing some avenues for further research in this field.

摘要: 随着网络控制系统规模的增加和不同子系统之间的交互变得更加复杂，这种网络的弹性问题变得越来越重要。使用图形语言重新定义经典系统和控制论概念的需要最近开始作为一个肥沃而重要的研究领域得到关注。本文综述了网络控制系统弹性分析的图论方法。我们讨论了运行在网络系统上的各种分布式算法，并通过观察其底层网络的结构属性来研究它们对恶意行为的恢复能力。我们提出了图论方法来量化攻击的影响，并从图论的角度重新解释了一些系统理论中的健壮性概念，以减轻攻击的影响。此外，我们还讨论了以图论为分析工具的网络控制系统的各种安全问题。最后，我们介绍了这一领域进一步研究的一些途径。



## **13. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

垂直联合学习中的标签泄漏及前向嵌入保护 cs.LG

**SubmitDate**: 2022-05-25    [paper-pdf](http://arxiv.org/pdf/2203.01451v3)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.

摘要: 垂直联合学习(VFL)近年来得到了广泛的关注，并被用来解决数据隐私问题。然而，最近的一些工作表明，即使参与者之间只传递前向中间嵌入(而不是原始特征)和反向传播的梯度(而不是原始标签)，VFL也容易受到隐私泄漏的影响。由于原始标签往往包含高度敏感的信息，最近的一些工作被提出以有效地防止VFL中反向传播梯度的标签泄漏。然而，这些工作只是识别和防御了反向传播梯度带来的标签泄漏威胁。这些工作都没有注意到中间嵌入带来的标签泄漏问题。本文提出了一种实用的标签推理方法，即使使用了标签差分隐私和梯度扰动等现有保护方法，也能有效地从共享中间嵌入中窃取私有标签。标签攻击的有效性离不开中间嵌入与对应的私有标签之间的关联。为了缓解前向嵌入带来的标签泄漏问题，我们在标签方增加了一个额外的优化目标，通过最小化中间嵌入与对应的私有标签之间的距离相关性来限制对手的标签窃取能力。我们进行了大量的实验来证明我们提出的保护方法的有效性。



## **14. Recipe2Vec: Multi-modal Recipe Representation Learning with Graph Neural Networks**

Recipe2Vec：基于图神经网络的多模式配方表示学习 cs.LG

Accepted by IJCAI 2022

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12396v1)

**Authors**: Yijun Tian, Chuxu Zhang, Zhichun Guo, Yihong Ma, Ronald Metoyer, Nitesh V. Chawla

**Abstracts**: Learning effective recipe representations is essential in food studies. Unlike what has been developed for image-based recipe retrieval or learning structural text embeddings, the combined effect of multi-modal information (i.e., recipe images, text, and relation data) receives less attention. In this paper, we formalize the problem of multi-modal recipe representation learning to integrate the visual, textual, and relational information into recipe embeddings. In particular, we first present Large-RG, a new recipe graph data with over half a million nodes, making it the largest recipe graph to date. We then propose Recipe2Vec, a novel graph neural network based recipe embedding model to capture multi-modal information. Additionally, we introduce an adversarial attack strategy to ensure stable learning and improve performance. Finally, we design a joint objective function of node classification and adversarial learning to optimize the model. Extensive experiments demonstrate that Recipe2Vec outperforms state-of-the-art baselines on two classic food study tasks, i.e., cuisine category classification and region prediction. Dataset and codes are available at https://github.com/meettyj/Recipe2Vec.

摘要: 学习有效的食谱表示法在食品研究中是必不可少的。与已开发的基于图像的配方检索或学习结构化文本嵌入不同，多模式信息(即，配方图像、文本和关系数据)的组合效果受到的关注较少。在本文中，我们将多通道配方表示学习问题形式化，以便将视觉、文本和关系信息集成到配方嵌入中。特别是，我们首先提出了Large-RG，这是一个具有超过50万个节点的新配方图数据，使其成为迄今为止最大的配方图。在此基础上，提出了一种新的基于图神经网络的食谱嵌入模型Recipe2Vec，用于获取多模式信息。此外，我们引入了对抗性攻击策略，以确保稳定的学习和提高性能。最后，我们设计了节点分类和对抗性学习的联合目标函数来优化模型。广泛的实验表明，Recipe2Vec在两个经典的食物研究任务上表现优于最先进的基线，即烹饪类别分类和区域预测。数据集和代码可在https://github.com/meettyj/Recipe2Vec.上获得



## **15. Label Leakage and Protection in Two-party Split Learning**

两方分裂学习中的标签泄漏及防护 cs.LG

Accepted to ICLR 2022 (https://openreview.net/forum?id=cOtBRgsf2fO)

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2102.08504v3)

**Authors**: Oscar Li, Jiankai Sun, Xin Yang, Weihao Gao, Hongyi Zhang, Junyuan Xie, Virginia Smith, Chong Wang

**Abstracts**: Two-party split learning is a popular technique for learning a model across feature-partitioned data. In this work, we explore whether it is possible for one party to steal the private label information from the other party during split training, and whether there are methods that can protect against such attacks. Specifically, we first formulate a realistic threat model and propose a privacy loss metric to quantify label leakage in split learning. We then show that there exist two simple yet effective methods within the threat model that can allow one party to accurately recover private ground-truth labels owned by the other party. To combat these attacks, we propose several random perturbation techniques, including $\texttt{Marvell}$, an approach that strategically finds the structure of the noise perturbation by minimizing the amount of label leakage (measured through our quantification metric) of a worst-case adversary. We empirically demonstrate the effectiveness of our protection techniques against the identified attacks, and show that $\texttt{Marvell}$ in particular has improved privacy-utility tradeoffs relative to baseline approaches.

摘要: 两方分裂学习是一种流行的跨特征分区数据学习模型的技术。在这项工作中，我们探索了在分裂训练过程中，一方是否有可能从另一方窃取私有标签信息，以及是否有方法可以防止此类攻击。具体地说，我们首先建立了一个现实的威胁模型，并提出了一种隐私损失度量来量化分裂学习中的标签泄漏。然后，我们证明了在威胁模型中存在两种简单而有效的方法，它们可以允许一方准确地恢复另一方拥有的私有地面事实标签。为了对抗这些攻击，我们提出了几种随机扰动技术，包括$\exttt{Marvell}$，一种通过最小化最坏情况对手的标签泄漏量(通过我们的量化度量来衡量)来战略性地发现噪声扰动的结构的方法。我们经验性地证明了我们的保护技术对识别的攻击的有效性，并表明与基准方法相比，$\exttt{Marvell}$尤其改进了隐私效用的权衡。



## **16. PORTFILER: Port-Level Network Profiling for Self-Propagating Malware Detection**

PORTFILER：用于自传播恶意软件检测的端口级网络分析 cs.CR

An earlier version is accepted to be published in IEEE Conference on  Communications and Network Security (CNS) 2021

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2112.13798v2)

**Authors**: Talha Ongun, Oliver Spohngellert, Benjamin Miller, Simona Boboila, Alina Oprea, Tina Eliassi-Rad, Jason Hiser, Alastair Nottingham, Jack Davidson, Malathi Veeraraghavan

**Abstracts**: Recent self-propagating malware (SPM) campaigns compromised hundred of thousands of victim machines on the Internet. It is challenging to detect these attacks in their early stages, as adversaries utilize common network services, use novel techniques, and can evade existing detection mechanisms. We propose PORTFILER (PORT-Level Network Traffic ProFILER), a new machine learning system applied to network traffic for detecting SPM attacks. PORTFILER extracts port-level features from the Zeek connection logs collected at a border of a monitored network, applies anomaly detection techniques to identify suspicious events, and ranks the alerts across ports for investigation by the Security Operations Center (SOC). We propose a novel ensemble methodology for aggregating individual models in PORTFILER that increases resilience against several evasion strategies compared to standard ML baselines. We extensively evaluate PORTFILER on traffic collected from two university networks, and show that it can detect SPM attacks with different patterns, such as WannaCry and Mirai, and performs well under evasion. Ranking across ports achieves precision over 0.94 with low false positive rates in the top ranked alerts. When deployed on the university networks, PORTFILER detected anomalous SPM-like activity on one of the campus networks, confirmed by the university SOC as malicious. PORTFILER also detected a Mirai attack recreated on the two university networks with higher precision and recall than deep-learning-based autoencoder methods.

摘要: 最近的自我传播恶意软件(SPM)活动危害了互联网上数十万受攻击的计算机。在攻击的早期阶段检测这些攻击是具有挑战性的，因为攻击者利用常见的网络服务，使用新的技术，并且可以逃避现有的检测机制。提出了一种应用于网络流量检测的机器学习系统PORTFILER(PORTFILER)，用于检测SPM攻击。PORTFILER从在受监控网络边界收集的Zeek连接日志中提取端口级特征，应用异常检测技术来识别可疑事件，并跨端口对警报进行排序，以供安全运营中心(SOC)调查。我们提出了一种新的集成方法，用于在PORTFILER中聚合单个模型，与标准的ML基线相比，该方法提高了对几种规避策略的弹性。我们对PORTFILER在两个大学网络上收集的流量进行了广泛的测试，结果表明，它可以检测出不同模式的SPM攻击，如WannaCry和Mirai，并且在规避情况下表现良好。在排名靠前的警报中，跨端口排名可实现0.94以上的精确度和较低的误警率。当部署在大学网络上时，PORTFILER在其中一个校园网络上检测到异常的类似SPM的活动，并被大学SOC确认为恶意活动。PORTFILER还检测到在两个大学网络上重新创建的Mirai攻击，与基于深度学习的自动编码器方法相比，具有更高的精确度和召回率。



## **17. Self-Supervised Contrastive Learning with Adversarial Perturbations for Defending Word Substitution-based Attacks**

抗单词替换攻击的对抗性扰动自监督对比学习 cs.CL

In Findings of NAACL 2022

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2107.07610v3)

**Authors**: Zhao Meng, Yihan Dong, Mrinmaya Sachan, Roger Wattenhofer

**Abstracts**: In this paper, we present an approach to improve the robustness of BERT language models against word substitution-based adversarial attacks by leveraging adversarial perturbations for self-supervised contrastive learning. We create a word-level adversarial attack generating hard positives on-the-fly as adversarial examples during contrastive learning. In contrast to previous works, our method improves model robustness without using any labeled data. Experimental results show that our method improves robustness of BERT against four different word substitution-based adversarial attacks, and combining our method with adversarial training gives higher robustness than adversarial training alone. As our method improves the robustness of BERT purely with unlabeled data, it opens up the possibility of using large text datasets to train robust language models against word substitution-based adversarial attacks.

摘要: 在本文中，我们提出了一种方法，通过利用对抗性扰动进行自我监督的对比学习来提高ERT语言模型对基于单词替换的对抗性攻击的健壮性。我们创建了一个词级对抗性攻击，在对比学习期间动态生成硬积极词作为对抗性例子。与以前的工作相比，我们的方法在不使用任何标记数据的情况下提高了模型的稳健性。实验结果表明，我们的方法提高了ERT对四种不同的基于单词替换的对抗性攻击的健壮性，并且将我们的方法与对抗性训练相结合比单独对抗性训练具有更高的健壮性。由于我们的方法提高了纯粹使用未标记数据的ERT的稳健性，因此它打开了使用大型文本数据集来训练稳健的语言模型以抵御基于单词替换的对手攻击的可能性。



## **18. Adversarial Attack on Attackers: Post-Process to Mitigate Black-Box Score-Based Query Attacks**

对攻击者的对抗性攻击：减轻基于黑盒分数的查询攻击的后处理 cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12134v1)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Yingwen Wu, Cihang Xie, Xiaolin Huang

**Abstracts**: The score-based query attacks (SQAs) pose practical threats to deep neural networks by crafting adversarial perturbations within dozens of queries, only using the model's output scores. Nonetheless, we note that if the loss trend of the outputs is slightly perturbed, SQAs could be easily misled and thereby become much less effective. Following this idea, we propose a novel defense, namely Adversarial Attack on Attackers (AAA), to confound SQAs towards incorrect attack directions by slightly modifying the output logits. In this way, (1) SQAs are prevented regardless of the model's worst-case robustness; (2) the original model predictions are hardly changed, i.e., no degradation on clean accuracy; (3) the calibration of confidence scores can be improved simultaneously. Extensive experiments are provided to verify the above advantages. For example, by setting $\ell_\infty=8/255$ on CIFAR-10, our proposed AAA helps WideResNet-28 secure $80.59\%$ accuracy under Square attack ($2500$ queries), while the best prior defense (i.e., adversarial training) only attains $67.44\%$. Since AAA attacks SQA's general greedy strategy, such advantages of AAA over 8 defenses can be consistently observed on 8 CIFAR-10/ImageNet models under 6 SQAs, using different attack targets and bounds. Moreover, AAA calibrates better without hurting the accuracy. Our code would be released.

摘要: 基于分数的查询攻击(SQA)通过在数十个查询中精心设计敌意扰动，仅使用模型的输出分数，对深度神经网络构成实际威胁。尽管如此，我们注意到，如果产出的损失趋势受到轻微干扰，质量保证人员很容易受到误导，从而变得不那么有效。根据这一思想，我们提出了一种新的防御方法，即对攻击者的对抗性攻击(AAA)，通过略微修改输出日志来迷惑SQA对错误的攻击方向。这样，(1)无论模型在最坏情况下的稳健性如何，都可以防止SQA；(2)原始模型预测几乎不会改变，即不会降低干净的精度；(3)置信度得分的校准可以同时得到改善。通过大量的实验验证了上述优点。例如，通过在CIFAR-10上设置$\ell_\infty=8/255$，我们提出的AAA帮助WideResNet-28在Square攻击($2500$查询)下获得$80.59\$准确性，而最好的先前防御(即对抗性训练)仅达到$67.44\$。由于AAA攻击SQA的一般贪婪策略，因此在使用不同的攻击目标和边界的6个SQA下的8个CIFAR-10/ImageNet模型上，可以一致地观察到AAA相对于8个防御的优势。此外，AAA在不影响精度的情况下校准得更好。我们的代码就会被发布。



## **19. Defending a Music Recommender Against Hubness-Based Adversarial Attacks**

保护音乐推荐器免受基于Hubness的恶意攻击 eess.AS

6 pages, to be published in Proceedings of the 19th Sound and Music  Computing Conference 2022 (SMC-22)

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.12032v1)

**Authors**: Katharina Hoedt, Arthur Flexer, Gerhard Widmer

**Abstracts**: Adversarial attacks can drastically degrade performance of recommenders and other machine learning systems, resulting in an increased demand for defence mechanisms. We present a new line of defence against attacks which exploit a vulnerability of recommenders that operate in high dimensional data spaces (the so-called hubness problem). We use a global data scaling method, namely Mutual Proximity (MP), to defend a real-world music recommender which previously was susceptible to attacks that inflated the number of times a particular song was recommended. We find that using MP as a defence greatly increases robustness of the recommender against a range of attacks, with success rates of attacks around 44% (before defence) dropping to less than 6% (after defence). Additionally, adversarial examples still able to fool the defended system do so at the price of noticeably lower audio quality as shown by a decreased average SNR.

摘要: 对抗性攻击会极大地降低推荐器和其他机器学习系统的性能，导致对防御机制的需求增加。我们提出了一条新的防线来抵御攻击，这些攻击利用了在高维数据空间中操作的推荐器的漏洞(所谓的Hubness问题)。我们使用一种全局数据缩放方法，即相互邻近(MP)，来保护一个现实世界的音乐推荐机构，它以前容易受到夸大某首歌曲被推荐次数的攻击。我们发现，使用MP作为防御大大提高了推荐器对一系列攻击的健壮性，攻击成功率在44%左右(防御前)下降到不到6%(防御后)。此外，敌意例子仍然能够欺骗防御系统这样做的代价是显著降低的音频质量，如降低的平均SNR所示。



## **20. Phrase-level Textual Adversarial Attack with Label Preservation**

具有标签保留的短语级文本对抗攻击 cs.CL

NAACL-HLT 2022 Findings (Long), 9 pages + 2 pages references + 8  pages appendix

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.10710v2)

**Authors**: Yibin Lei, Yu Cao, Dianqi Li, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy

**Abstracts**: Generating high-quality textual adversarial examples is critical for investigating the pitfalls of natural language processing (NLP) models and further promoting their robustness. Existing attacks are usually realized through word-level or sentence-level perturbations, which either limit the perturbation space or sacrifice fluency and textual quality, both affecting the attack effectiveness. In this paper, we propose Phrase-Level Textual Adversarial aTtack (PLAT) that generates adversarial samples through phrase-level perturbations. PLAT first extracts the vulnerable phrases as attack targets by a syntactic parser, and then perturbs them by a pre-trained blank-infilling model. Such flexible perturbation design substantially expands the search space for more effective attacks without introducing too many modifications, and meanwhile maintaining the textual fluency and grammaticality via contextualized generation using surrounding texts. Moreover, we develop a label-preservation filter leveraging the likelihoods of language models fine-tuned on each class, rather than textual similarity, to rule out those perturbations that potentially alter the original class label for humans. Extensive experiments and human evaluation demonstrate that PLAT has a superior attack effectiveness as well as a better label consistency than strong baselines.

摘要: 生成高质量的文本对抗性实例对于研究自然语言处理(NLP)模型的缺陷并进一步提高其稳健性至关重要。现有的攻击通常是通过词级或句子级的扰动来实现的，这要么限制了扰动空间，要么牺牲了流畅度和文本质量，两者都影响了攻击的有效性。在本文中，我们提出了短语级别的文本对抗攻击(PLAT)，它通过短语级别的扰动来生成敌对样本。PLAT首先通过句法分析器提取易受攻击的短语作为攻击目标，然后通过预先训练的空白填充模型对其进行扰动。这种灵活的扰动设计在不引入太多修改的情况下大大扩展了更有效的攻击的搜索空间，同时通过使用周围文本的上下文生成来保持文本的流畅性和语法性。此外，我们开发了一个标签保存过滤器，利用在每个类上微调的语言模型的可能性，而不是文本相似性，以排除那些可能改变人类原始类标签的扰动。大量的实验和人工评估表明，与强基线相比，PLAT具有更好的攻击效果和更好的标签一致性。



## **21. Can Adversarial Training Be Manipulated By Non-Robust Features?**

对抗性训练能被非强健特征操纵吗？ cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2201.13329v2)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstracts**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attacks, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness.

摘要: 对抗性训练最初是为了抵抗测试时间对抗性的例子，已经被证明在减轻训练时间可用性攻击方面很有希望。然而，这种防御能力在本文中受到了挑战。我们识别了一种名为稳定性攻击的新威胁模型，该模型旨在通过稍微操纵训练数据来阻碍健壮性可用性。在这种威胁下，我们证明了在简单的统计设置下，使用常规国防预算的对抗性训练不能提供测试稳健性，其中训练数据的非稳健性特征可以通过有界扰动来加强。进一步，我们分析了增加国防预算以对抗稳定性攻击的必要性。最后，综合实验表明，稳定性攻击对基准数据集是有害的，因此需要自适应防御来保持稳健性。



## **22. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

智能电网：网络攻击、关键防御方法和数字孪生 cs.CR

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11783v1)

**Authors**: Tianming Zheng, Ming Liu, Deepak Puthal, Ping Yi, Yue Wu, Xiangjian He

**Abstracts**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internetconnected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by either improving the existing defense approaches or introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid. In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.

摘要: 智能电网作为国家的重要基础设施，其网络安全问题引起了广泛关注。智能电网向着智能化、数字化、互联网化的方向发展，吸引了来自外部的恶意活动对手。有必要通过改进现有的防御方法或将新开发的技术引入智能电网来增强其网络安全。作为一项新兴技术，数字孪生(DT)被认为是增强安全性的使能器。然而，实际实施起来却颇具挑战性。这是由于智能电网设计人员、安全专家和DT开发人员之间的知识障碍。每个领域都是一个复杂的系统，涵盖了各种组件和技术。因此，需要对相关内容进行梳理，以便将分布式测试更好地嵌入到智能电网的安全体系结构设计中。为了满足这一需求，本文涵盖了智能电网、网络安全和分布式计算三个领域。具体地，介绍了智能电网的背景；ii)回顾了来自攻击事件和攻击方法的外部网络攻击；iii)介绍了工业网络系统中的关键防御方法，包括设备识别、漏洞发现、入侵检测系统(IDS)、蜜罐、属性和威胁情报(TI)；iv)回顾了DT的相关内容，包括DT的基本概念、在智能电网中的应用以及DT如何增强安全。最后，本文对基于DT的智能电网的未来发展提出了我们的安全考虑。这项调查有望帮助开发人员打破智能电网、网络安全和DT之间的知识壁垒，并为未来基于DT的智能电网的安全设计提供指导。



## **23. Alleviating Robust Overfitting of Adversarial Training With Consistency Regularization**

用一致性正则化缓解对抗性训练的稳健过度 cs.LG

**SubmitDate**: 2022-05-24    [paper-pdf](http://arxiv.org/pdf/2205.11744v1)

**Authors**: Shudong Zhang, Haichang Gao, Tianwei Zhang, Yunyi Zhou, Zihui Wu

**Abstracts**: Adversarial training (AT) has proven to be one of the most effective ways to defend Deep Neural Networks (DNNs) against adversarial attacks. However, the phenomenon of robust overfitting, i.e., the robustness will drop sharply at a certain stage, always exists during AT. It is of great importance to decrease this robust generalization gap in order to obtain a robust model. In this paper, we present an in-depth study towards the robust overfitting from a new angle. We observe that consistency regularization, a popular technique in semi-supervised learning, has a similar goal as AT and can be used to alleviate robust overfitting. We empirically validate this observation, and find a majority of prior solutions have implicit connections to consistency regularization. Motivated by this, we introduce a new AT solution, which integrates the consistency regularization and Mean Teacher (MT) strategy into AT. Specifically, we introduce a teacher model, coming from the average weights of the student models over the training steps. Then we design a consistency loss function to make the prediction distribution of the student models over adversarial examples consistent with that of the teacher model over clean samples. Experiments show that our proposed method can effectively alleviate robust overfitting and improve the robustness of DNN models against common adversarial attacks.

摘要: 对抗训练(AT)已被证明是保护深度神经网络(DNN)免受对手攻击的最有效方法之一。然而，在自动测试过程中，健壮性过拟合现象时有发生，即健壮性在某一阶段会急剧下降。为了得到一个稳健的模型，缩小这种稳健的泛化差距是非常重要的。本文从一个新的角度对稳健过拟合问题进行了深入的研究。我们观察到，一致性正则化是半监督学习中的一种流行技术，具有与AT相似的目标，并可用于缓解稳健过拟合。我们实证地验证了这一观察结果，并发现大多数先前的解决方案都与一致性正则化有隐含的联系。受此启发，我们提出了一种新的AT解决方案，该方案将一致性正则化和均值教师(MT)策略结合到AT中。具体地说，我们引入了一个教师模型，该模型来自学生模型在训练步骤中的平均权重。然后设计了一致性损失函数，使得学生模型在对抗性样本上的预测分布与教师模型在干净样本上的预测分布一致。实验表明，本文提出的方法可以有效地缓解DNN模型的鲁棒性过高问题，提高DNN模型对常见对手攻击的鲁棒性。



## **24. Learning to Ignore Adversarial Attacks**

学会忽视对手的攻击 cs.CL

14 pages, 2 figures

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2205.11551v1)

**Authors**: Yiming Zhang, Yangqiaoyu Zhou, Samuel Carton, Chenhao Tan

**Abstracts**: Despite the strong performance of current NLP models, they can be brittle against adversarial attacks. To enable effective learning against adversarial inputs, we introduce the use of rationale models that can explicitly learn to ignore attack tokens. We find that the rationale models can successfully ignore over 90\% of attack tokens. This approach leads to consistent sizable improvements ($\sim$10\%) over baseline models in robustness on three datasets for both BERT and RoBERTa, and also reliably outperforms data augmentation with adversarial examples alone. In many cases, we find that our method is able to close the gap between model performance on a clean test set and an attacked test set and hence reduce the effect of adversarial attacks.

摘要: 尽管目前的NLP模型表现强劲，但它们在对抗对手攻击时可能很脆弱。为了能够针对敌意输入进行有效学习，我们引入了可以显式学习忽略攻击令牌的基本模型的使用。我们发现，基本模型可以成功地忽略90%以上的攻击令牌。在BERT和Roberta的三个数据集上，这种方法在稳健性方面都比基线模型有了一致的显著改善($SIM$10)，并且可靠地超过了仅使用对抗性例子的数据增强。在许多情况下，我们发现我们的方法能够缩小干净测试集和被攻击测试集上的模型性能之间的差距，从而减少对抗性攻击的影响。



## **25. Graph Layer Security: Encrypting Information via Common Networked Physics**

图形层安全：通过公共网络物理加密信息 eess.SP

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2006.03568v3)

**Authors**: Zhuangkun Wei, Liang Wang, Schyler Chengyao Sun, Bin Li, Weisi Guo

**Abstracts**: The proliferation of low-cost Internet of Things (IoT) devices has led to a race between wireless security and channel attacks. Traditional cryptography requires high-computational power and is not suitable for low-power IoT scenarios. Whist, recently developed physical layer security (PLS) can exploit common wireless channel state information (CSI), its sensitivity to channel estimation makes them vulnerable from attacks. In this work, we exploit an alternative common physics shared between IoT transceivers: the monitored channel-irrelevant physical networked dynamics (e.g., water/oil/gas/electrical signal-flows). Leveraging this, we propose for the first time, graph layer security (GLS), by exploiting the dependency in physical dynamics among network nodes for information encryption and decryption. A graph Fourier transform (GFT) operator is used to characterize such dependency into a graph-bandlimted subspace, which allows the generations of channel-irrelevant cipher keys by maximizing the secrecy rate. We evaluate our GLS against designed active and passive attackers, using IEEE 39-Bus system. Results demonstrate that, GLS is not reliant on wireless CSI, and can combat attackers that have partial networked dynamic knowledge (realistic access to full dynamic and critical nodes remains challenging). We believe this novel GLS has widespread applicability in secure health monitoring and for Digital Twins in adversarial radio environments.

摘要: 低成本物联网(IoT)设备的激增导致了无线安全和渠道攻击之间的竞争。传统的密码学需要高计算能力，不适合低功耗的物联网场景。然而，最近发展起来的物理层安全技术可以利用常见的无线信道状态信息，但其对信道估计的敏感性使其容易受到攻击。在这项工作中，我们利用了物联网收发器之间共享的另一种常见物理：受监控的与通道无关的物理网络动态(例如，水/油/气/电信号流)。利用这一点，我们首次提出了图层安全(GLS)，通过利用网络节点之间的物理动力学依赖性来进行信息加密和解密。图傅里叶变换(GFT)算子被用来将这种依赖描述到图带宽受限的子空间中，该子空间通过最大化保密率来生成与信道无关的密钥。我们使用IEEE 39节点系统，针对设计的主动和被动攻击者评估我们的GLS。结果表明，GLS不依赖于无线CSI，可以对抗具有部分网络动态知识的攻击者(实际访问完全动态和关键节点仍然具有挑战性)。我们相信这种新的GLS在安全的健康监测和对抗性无线电环境中的数字双胞胎中具有广泛的适用性。



## **26. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles***

网络化无人机隐身对手检测技术研究 eess.SY

to appear at the 2022 Int'l Conference on Unmanned Aircraft Systems  (ICUAS)

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2202.09661v2)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.

摘要: 无人机网络在执行复杂的协作任务时提供了分布式覆盖、可重构和机动性。然而，它依赖的无线通信可能会受到网络对手和入侵的影响，扰乱整个网络的运行。提出了一种基于模型的集中式和分散式观测器技术，用于检测编队控制环境中无人机的一类隐身入侵，即零动态攻击和隐蔽攻击。在控制中心运行的集中式观察器利用无人机通信拓扑中的切换来进行攻击检测，而分布式观察器在网络中的每一架无人机上实现，使用联网的无人机模型和本地可用的测量。实验结果表明，所提出的检测方案在不同的案例研究中是有效的。



## **27. Inverse-Inverse Reinforcement Learning. How to Hide Strategy from an Adversarial Inverse Reinforcement Learner**

逆-逆强化学习。如何从对抗性的逆强化学习器中隐藏策略 cs.LG

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2205.10802v1)

**Authors**: Kunal Pattanayak, Vikram Krishnamurthy, Christopher Berry

**Abstracts**: Inverse reinforcement learning (IRL) deals with estimating an agent's utility function from its actions. In this paper, we consider how an agent can hide its strategy and mitigate an adversarial IRL attack; we call this inverse IRL (I-IRL). How should the decision maker choose its response to ensure a poor reconstruction of its strategy by an adversary performing IRL to estimate the agent's strategy? This paper comprises four results: First, we present an adversarial IRL algorithm that estimates the agent's strategy while controlling the agent's utility function. Our second result for I-IRL result spoofs the IRL algorithm used by the adversary. Our I-IRL results are based on revealed preference theory in micro-economics. The key idea is for the agent to deliberately choose sub-optimal responses that sufficiently masks its true strategy. Third, we give a sample complexity result for our main I-IRL result when the agent has noisy estimates of the adversary specified utility function. Finally, we illustrate our I-IRL scheme in a radar problem where a meta-cognitive radar is trying to mitigate an adversarial target.

摘要: 逆强化学习(IRL)处理从主体的动作中估计主体的效用函数。在本文中，我们考虑了代理如何隐藏其策略并缓解对抗性IRL攻击，我们称之为逆IRL(I-IRL)。决策者应该如何选择其响应，以确保对手执行IRL来估计代理人的战略时，对其战略的重建效果不佳？首先，提出了一种对抗性IRL算法，该算法在估计代理策略的同时控制代理的效用函数。对于I-IRL结果，我们的第二个结果伪造了对手使用的IRL算法。我们的I-IRL结果是基于微观经济学中的揭示偏好理论。关键思想是让代理故意选择次优响应，以充分掩盖其真实战略。第三，当代理对对手指定的效用函数有噪声估计时，我们给出了主要I-IRL结果的样本复杂性结果。最后，我们在一个雷达问题中说明了我们的I-IRL方案，其中元认知雷达正试图减轻敌方目标的威胁。



## **28. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

漏洞后恢复：针对泄漏的DNN模型的白盒对抗示例 cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10686v1)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.

摘要: 在当今的互联网上，服务器入侵是一个不幸的现实。在深度神经网络(DNN)模型的背景下，它们尤其有害，因为泄露的模型让攻击者可以使用“白盒”来生成对抗性示例，这是一种没有实际可靠防御措施的威胁模型。对于在专有DNN(例如医学成像)上投入多年和数百万美元的从业者来说，这似乎是一场不可避免的灾难迫在眉睫。在本文中，我们考虑了DNN模型的漏洞后恢复问题。我们提出了Neo，一个新的系统，它创建新版本的泄漏模型，以及一个推理时间过滤器，检测并删除在以前泄漏的模型上生成的敌对示例。不同模型版本的分类面略有偏移(通过引入隐藏分布)，并且Neo检测到对其生成中使用的泄漏模型的攻击过拟合。我们发现，在各种任务和攻击方法中，Neo能够以非常高的准确率从泄露的模型中过滤攻击，并针对反复破坏服务器的攻击者提供强大的保护(7-10次恢复)。NEO在各种强自适应攻击中表现良好，在可恢复的漏洞数量中略有下降，并显示出在野外作为DNN防御的补充潜力。



## **29. Gradient Concealment: Free Lunch for Defending Adversarial Attacks**

梯度隐藏：防御对手攻击的免费午餐 cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10617v1)

**Authors**: Sen Pei, Jiaxi Sun, Xiaopeng Zhang, Gaofeng Meng

**Abstracts**: Recent studies show that the deep neural networks (DNNs) have achieved great success in various tasks. However, even the \emph{state-of-the-art} deep learning based classifiers are extremely vulnerable to adversarial examples, resulting in sharp decay of discrimination accuracy in the presence of enormous unknown attacks. Given the fact that neural networks are widely used in the open world scenario which can be safety-critical situations, mitigating the adversarial effects of deep learning methods has become an urgent need. Generally, conventional DNNs can be attacked with a dramatically high success rate since their gradient is exposed thoroughly in the white-box scenario, making it effortless to ruin a well trained classifier with only imperceptible perturbations in the raw data space. For tackling this problem, we propose a plug-and-play layer that is training-free, termed as \textbf{G}radient \textbf{C}oncealment \textbf{M}odule (GCM), concealing the vulnerable direction of gradient while guaranteeing the classification accuracy during the inference time. GCM reports superior defense results on the ImageNet classification benchmark, improving up to 63.41\% top-1 attack robustness (AR) when faced with adversarial inputs compared to the vanilla DNNs. Moreover, we use GCM in the CVPR 2022 Robust Classification Challenge, currently achieving \textbf{2nd} place in Phase II with only a tiny version of ConvNext. The code will be made available.

摘要: 最近的研究表明，深度神经网络在各种任务中取得了巨大的成功。然而，即使是基于深度学习的分类器也极易受到敌意例子的攻击，导致在存在大量未知攻击的情况下识别精度急剧下降。鉴于神经网络被广泛应用于开放世界场景中，这可能是安全关键的情况，缓解深度学习方法的对抗性已成为迫切需要。通常，传统的DNN可以以极高的成功率受到攻击，因为它们的梯度在白盒场景中被彻底暴露，使得在原始数据空间中只有不可察觉的扰动就可以毫不费力地破坏一个训练有素的分类器。为了解决这一问题，我们提出了一种无需训练的即插即用层，称为文本bf{G}辐射\文本bf{C}模块(GCM)，它隐藏了梯度的脆弱方向，同时保证了推理时的分类精度。GCM在ImageNet分类基准上报告了卓越的防御结果，与普通DNN相比，在面对敌意输入时，最高可提高63.41\%top-1攻击健壮性(AR)。此外，我们在CVPR 2022健壮分类挑战赛中使用了GCM，目前仅使用ConvNext的一个微型版本就在第二阶段获得了\textbf{2}名。代码将可用。



## **30. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

SERVFAIL：DNSSEC中算法敏捷性的意外后果 cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10608v1)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.

摘要: 密码算法灵活性是DNSSEC的一个重要属性：如果现有算法不再安全，它允许轻松部署新算法。大量的操作和研究工作致力于推动在DNSSEC中部署新算法。最近的研究表明，DNSSEC正在逐步实现算法灵活性：大多数支持DNSSEC的解析器可以验证一些不同的算法，并且越来越多的域使用密码强密码签名。在这项工作中，我们第一次展示了DNSSEC的密码敏捷性，尽管对于使用强大的密码学来确保DNS安全至关重要，但也引入了一个严重的漏洞。我们发现，在某些条件下，当新算法在签名的DNS响应中列出时，解析器不会验证DNSSEC。因此，部署新密码的域有可能使验证解析器面临缓存中毒攻击。我们利用这一点来开发DNSSEC降级攻击，并表明在某些情况下，这些攻击甚至可以由偏离路径的对手发起。我们从实验和伦理上评估我们对全球Web客户端使用的流行的DNS解析器实现、公共DNS提供商和DNS服务的攻击。我们通过毒化解析器来验证DNSSEC降级攻击的成功：我们在有符号的域中向验证解析器的缓存中注入虚假记录。我们发现，主要的域名服务提供商，如Google Public DNS和Cloudflare，以及网络客户端使用的70%的域名解析程序，都容易受到我们的攻击。我们追踪了导致这种情况的因素并提出了建议。



## **31. On the Feasibility and Generality of Patch-based Adversarial Attacks on Semantic Segmentation Problems**

基于补丁的对抗性语义切分攻击的可行性和通用性 cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10539v1)

**Authors**: Soma Kontar, Andras Horvath

**Abstracts**: Deep neural networks were applied with success in a myriad of applications, but in safety critical use cases adversarial attacks still pose a significant threat. These attacks were demonstrated on various classification and detection tasks and are usually considered general in a sense that arbitrary network outputs can be generated by them.   In this paper we will demonstrate through simple case studies both in simulation and in real-life, that patch based attacks can be utilised to alter the output of segmentation networks. Through a few examples and the investigation of network complexity, we will also demonstrate that the number of possible output maps which can be generated via patch-based attacks of a given size is typically smaller than the area they effect or areas which should be attacked in case of practical applications.   We will prove that based on these results most patch-based attacks cannot be general in practice, namely they can not generate arbitrary output maps or if they could, they are spatially limited and this limit is significantly smaller than the receptive field of the patches.

摘要: 深度神经网络在许多应用中都取得了成功，但在安全关键用例中，对抗性攻击仍构成重大威胁。这些攻击在各种分类和检测任务中进行了演示，通常被认为是一般性的，因为它们可以生成任意的网络输出。在本文中，我们将通过模拟和现实生活中的简单案例研究来演示，基于补丁的攻击可以用来改变分段网络的输出。通过几个例子和对网络复杂性的调查，我们还将证明，在实际应用中，通过给定大小的基于补丁的攻击可以生成的可能输出地图的数量通常小于它们影响的区域或应该攻击的区域。基于这些结果，我们将证明大多数基于面片的攻击在实践中不是通用的，即它们不能生成任意的输出地图，或者如果它们可以，它们在空间上是受限的，并且这个限制明显小于面片的接受范围。



## **32. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

Prada：针对神经排序模型的实用黑箱对抗性攻击 cs.IR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2204.01321v2)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.

摘要: 近年来，神经网络排序模型(NRM)取得了显著的成功，尤其是使用了预先训练好的语言模型。然而，深层神经模型因其易受敌意例子的攻击而臭名昭著。鉴于我们对神经信息检索模型的日益依赖，对抗性攻击可能成为一种新型的Web垃圾邮件技术。因此，在部署NRM之前，研究潜在的敌意攻击以识别NRM的漏洞是很重要的。在本文中，我们引入了针对NRMS的单词替换排名攻击(WSRA)任务，该任务旨在通过在目标文档的文本中添加对抗性扰动来提升其排名。重点研究了基于决策的黑盒攻击环境，其中攻击者无法获取模型参数和梯度，只能通过查询目标模型获得部分检索列表的排名位置。这种攻击设置在现实世界的搜索引擎中是现实的。提出了一种新的基于伪相关性的对抗性排序攻击方法(PRADA)，该方法通过学习基于伪相关反馈(PRF)的代理模型来生成用于发现对抗性扰动的梯度。在两个网络搜索基准数据集上的实验表明，Prada可以超越现有的攻击策略，并成功地利用文本的微小不可分辨扰动来欺骗NRM。



## **33. Robust Sensible Adversarial Learning of Deep Neural Networks for Image Classification**

用于图像分类的深度神经网络稳健敏感对抗性学习 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10457v1)

**Authors**: Jungeum Kim, Xiao Wang

**Abstracts**: The idea of robustness is central and critical to modern statistical analysis. However, despite the recent advances of deep neural networks (DNNs), many studies have shown that DNNs are vulnerable to adversarial attacks. Making imperceptible changes to an image can cause DNN models to make the wrong classification with high confidence, such as classifying a benign mole as a malignant tumor and a stop sign as a speed limit sign. The trade-off between robustness and standard accuracy is common for DNN models. In this paper, we introduce sensible adversarial learning and demonstrate the synergistic effect between pursuits of standard natural accuracy and robustness. Specifically, we define a sensible adversary which is useful for learning a robust model while keeping high natural accuracy. We theoretically establish that the Bayes classifier is the most robust multi-class classifier with the 0-1 loss under sensible adversarial learning. We propose a novel and efficient algorithm that trains a robust model using implicit loss truncation. We apply sensible adversarial learning for large-scale image classification to a handwritten digital image dataset called MNIST and an object recognition colored image dataset called CIFAR10. We have performed an extensive comparative study to compare our method with other competitive methods. Our experiments empirically demonstrate that our method is not sensitive to its hyperparameter and does not collapse even with a small model capacity while promoting robustness against various attacks and keeping high natural accuracy.

摘要: 稳健性的概念是现代统计分析的核心和关键。然而，尽管深度神经网络(DNN)最近取得了进展，但许多研究表明DNN很容易受到对手的攻击。对图像进行不知不觉的更改可能会导致DNN模型以高置信度做出错误分类，例如将良性葡萄胎归类为恶性肿瘤，将停车标志归类为限速标志。对于DNN模型来说，稳健性和标准精度之间的权衡是很常见的。在本文中，我们引入了敏感的对抗性学习，并证明了追求标准自然准确性和稳健性之间的协同效应。具体地说，我们定义了一个明智的对手，它有助于学习健壮的模型，同时保持较高的自然准确性。我们从理论上证明了贝叶斯分类器是在敏感对抗性学习下具有0-1损失的最健壮的多类分类器。我们提出了一种新颖而高效的算法，该算法使用隐式损失截断来训练稳健的模型。在手写数字图像数据集MNIST和目标识别彩色图像数据集CIFAR10上，我们将敏感的对抗性学习应用于大规模图像分类。我们进行了广泛的比较研究，将我们的方法与其他竞争方法进行比较。实验表明，该方法对超参数不敏感，即使在模型容量较小的情况下也不会崩溃，同时提高了对各种攻击的鲁棒性，并保持了较高的自然准确率。



## **34. Vulnerability Analysis and Performance Enhancement of Authentication Protocol in Dynamic Wireless Power Transfer Systems**

动态无线电能传输系统中认证协议的脆弱性分析与性能提升 cs.CR

16 pages, conference

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10292v1)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstracts**: Recent advancements in wireless charging technology, as well as the possibility of utilizing it in the Electric Vehicle (EV) domain for dynamic charging solutions, have fueled the demand for a secure and usable protocol in the Dynamic Wireless Power Transfer (DWPT) technology. The DWPT must operate in the presence of malicious adversaries that can undermine the charging process and harm the customer service quality, while preserving the privacy of the users. Recently, it was shown that the DWPT system is susceptible to adversarial attacks, including replay, denial-of-service and free-riding attacks, which can lead to the adversary blocking the authorized user from charging, enabling free charging for free riders and exploiting the location privacy of the customers. In this paper, we study the current State-Of-The-Art (SOTA) authentication protocols and make the following two contributions: a) we show that the SOTA is vulnerable to the tracking of the user activity and b) we propose an enhanced authentication protocol that eliminates the vulnerability while providing improved efficiency compared to the SOTA authentication protocols. By adopting authentication messages based only on exclusive OR operations, hashing, and hash chains, we optimize the protocol to achieve a complexity that varies linearly with the number of charging pads, providing improved scalability. Compared to SOTA, the proposed scheme has a performance gain in the computational cost of around 90% on average for each pad.

摘要: 无线充电技术的最新进展，以及在电动汽车(EV)领域将其用于动态充电解决方案的可能性，推动了对动态无线功率传输(DWPT)技术中安全和可用的协议的需求。DWPT必须在恶意对手存在的情况下运行，这些恶意对手可能会破坏收费过程并损害客户服务质量，同时保护用户的隐私。最近，有研究表明，DWPT系统容易受到包括重放、拒绝服务和搭便车在内的敌意攻击，这些攻击可以导致对手阻止授权用户收费，使搭便车的人能够免费收费，并利用客户的位置隐私。在本文中，我们研究了现有的SOTA认证协议，并做了以下两个方面的贡献：a)我们发现SOTA容易受到用户活动跟踪的影响；b)我们提出了一种增强的认证协议，与SOTA认证协议相比，它消除了这个漏洞，同时提供了更高的效率。通过采用仅基于异或运算、哈希和哈希链的身份验证消息，我们对协议进行了优化，以实现随充电板数量线性变化的复杂性，从而提供更高的可扩展性。与SOTA相比，对于每个PAD，所提出的方案的计算代价平均提高了90%左右。



## **35. Adversarial Body Shape Search for Legged Robots**

腿部机器人的对抗性体型搜索 cs.RO

6 pages, 7 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10187v1)

**Authors**: Takaaki Azakami, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We propose an evolutionary computation method for an adversarial attack on the length and thickness of parts of legged robots by deep reinforcement learning. This attack changes the robot body shape and interferes with walking-we call the attacked body as adversarial body shape. The evolutionary computation method searches adversarial body shape by minimizing the expected cumulative reward earned through walking simulation. To evaluate the effectiveness of the proposed method, we perform experiments with three-legged robots, Walker2d, Ant-v2, and Humanoid-v2 in OpenAI Gym. The experimental results reveal that Walker2d and Ant-v2 are more vulnerable to the attack on the length than the thickness of the body parts, whereas Humanoid-v2 is vulnerable to the attack on both of the length and thickness. We further identify that the adversarial body shapes break left-right symmetry or shift the center of gravity of the legged robots. Finding adversarial body shape can be used to proactively diagnose the vulnerability of legged robot walking.

摘要: 提出了一种基于深度强化学习的腿部机器人长度和厚度对抗性攻击的进化计算方法。这种攻击改变了机器人的体型，干扰了行走--我们将被攻击的体型称为对抗性体型。进化计算方法通过最小化通过模拟行走获得的期望累积奖励来搜索对手的体型。为了评估该方法的有效性，我们在OpenAI健身房中用三足机器人Walker2d、Ant-v2和Human-v2进行了实验。实验结果表明，Walker2d和Ant-v2对长度的攻击比对身体部分的厚度更容易受到攻击，而人形v2对长度和厚度的攻击都更容易受到攻击。我们进一步识别出，对抗性的身体形状打破了左右对称或移动了腿部机器人的重心。发现敌对的体型可以用来主动诊断腿部机器人行走的脆弱性。



## **36. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

获得一轮保证：对认证健壮性的浮点攻击 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10159v1)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstracts**: Adversarial examples pose a security risk as they can alter a classifier's decision through slight perturbations to a benign input. Certified robustness has been proposed as a mitigation strategy where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural network verifiers that return a certified lower bound on a robust radius. Our experiments demonstrate over 50% attack success rate on random linear classifiers, up to 35% on a breast cancer dataset for logistic regression, and a 9% attack success rate on the MNIST dataset for a neural network whose certified radius was verified by a prominent bound propagation method. We also show that state-of-the-art random smoothed classifiers for neural networks are also susceptible to adversarial examples (e.g., up to 2% attack rate on CIFAR10)-validating the importance of accounting for the error rate of robustness guarantees of such classifiers in practice. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.

摘要: 敌意的例子会带来安全风险，因为它们可以通过轻微的扰动改变分类器的决定，使其成为良性的输入。证明的稳健性已经被提出作为一种缓解策略，在给定输入$x$的情况下，分类器返回预测和半径，并且可证明地保证在该半径内(例如，在$L_2$范数下)对$x$的任何扰动不会改变分类器的预测。在这项工作中，我们证明了这些保证可能会由于浮点表示的限制而失效，从而导致舍入误差。我们设计了一种四舍五入的搜索方法，可以有效地利用这个漏洞在认证的半径内找到对抗性示例。我们证明了该攻击可以针对具有精确可证明保证的几个线性分类器，以及针对返回关于稳健半径的证明下界的神经网络验证器。我们的实验表明，在随机线性分类器上的攻击成功率超过50%，在用于Logistic回归的乳腺癌数据集上的攻击成功率高达35%，在MNIST数据集上的攻击成功率在MNIST数据集上达到9%，其认证半径通过显著边界传播方法进行验证。我们还表明，最先进的随机平滑神经网络分类器也容易受到敌意例子的影响(例如，对CIFAR10的攻击率高达2%)-验证了在实践中考虑此类分类器的健壮性保证的错误率的重要性。最后，作为一种缓解措施，我们主张使用四舍五入区间算术来计算舍入误差。



## **37. Generating Semantic Adversarial Examples via Feature Manipulation**

通过特征处理生成语义对抗性实例 cs.LG

arXiv admin note: substantial text overlap with arXiv:1705.09064 by  other authors

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2001.02297v2)

**Authors**: Shuo Wang, Surya Nepal, Carsten Rudolph, Marthie Grobler, Shangyu Chen, Tianle Chen

**Abstracts**: The vulnerability of deep neural networks to adversarial attacks has been widely demonstrated (e.g., adversarial example attacks). Traditional attacks perform unstructured pixel-wise perturbation to fool the classifier. An alternative approach is to have perturbations in the latent space. However, such perturbations are hard to control due to the lack of interpretability and disentanglement. In this paper, we propose a more practical adversarial attack by designing structured perturbation with semantic meanings. Our proposed technique manipulates the semantic attributes of images via the disentangled latent codes. The intuition behind our technique is that images in similar domains have some commonly shared but theme-independent semantic attributes, e.g. thickness of lines in handwritten digits, that can be bidirectionally mapped to disentangled latent codes. We generate adversarial perturbation by manipulating a single or a combination of these latent codes and propose two unsupervised semantic manipulation approaches: vector-based disentangled representation and feature map-based disentangled representation, in terms of the complexity of the latent codes and smoothness of the reconstructed images. We conduct extensive experimental evaluations on real-world image data to demonstrate the power of our attacks for black-box classifiers. We further demonstrate the existence of a universal, image-agnostic semantic adversarial example.

摘要: 深度神经网络对对抗性攻击的脆弱性已被广泛证明(例如，对抗性示例攻击)。传统攻击执行非结构化像素级扰动来愚弄分类器。另一种方法是在潜在空间中进行微扰。然而，由于缺乏可解释性和解缠性，这种扰动很难控制。本文通过设计具有语义的结构化扰动，提出了一种更实用的对抗性攻击。我们提出的技术通过解开纠缠的潜在代码来操纵图像的语义属性。我们的技术背后的直觉是，相似域中的图像具有一些共同的但与主题无关的语义属性，例如手写数字中的线条粗细，可以双向映射到解开的潜在代码。针对潜在编码的复杂性和重构图像的平稳性，提出了两种无监督的语义处理方法：基于矢量的去纠缠表示和基于特征映射的去纠缠表示。我们在真实世界的图像数据上进行了广泛的实验评估，以展示我们对黑盒分类器的攻击的能力。我们进一步证明了一个普遍的、与图像无关的语义对抗例子的存在。



## **38. Adversarial joint attacks on legged robots**

针对腿部机器人的对抗性联合攻击 cs.RO

6 pages, 8 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10098v1)

**Authors**: Takuto Otomo, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We address adversarial attacks on the actuators at the joints of legged robots trained by deep reinforcement learning. The vulnerability to the joint attacks can significantly impact the safety and robustness of legged robots. In this study, we demonstrate that the adversarial perturbations to the torque control signals of the actuators can significantly reduce the rewards and cause walking instability in robots. To find the adversarial torque perturbations, we develop black-box adversarial attacks, where, the adversary cannot access the neural networks trained by deep reinforcement learning. The black box attack can be applied to legged robots regardless of the architecture and algorithms of deep reinforcement learning. We employ three search methods for the black-box adversarial attacks: random search, differential evolution, and numerical gradient descent methods. In experiments with the quadruped robot Ant-v2 and the bipedal robot Humanoid-v2, in OpenAI Gym environments, we find that differential evolution can efficiently find the strongest torque perturbations among the three methods. In addition, we realize that the quadruped robot Ant-v2 is vulnerable to the adversarial perturbations, whereas the bipedal robot Humanoid-v2 is robust to the perturbations. Consequently, the joint attacks can be used for proactive diagnosis of robot walking instability.

摘要: 我们解决了对通过深度强化学习训练的腿部机器人关节处的致动器的对抗性攻击。对联合攻击的脆弱性会显著影响腿部机器人的安全性和健壮性。在这项研究中，我们证明了对执行器的力矩控制信号的对抗性扰动可以显著减少机器人的奖励并导致机器人行走不稳定。为了发现对抗性扭矩扰动，我们提出了黑盒对抗性攻击，其中，对手不能访问通过深度强化学习训练的神经网络。无论深度强化学习的体系结构和算法如何，黑盒攻击都可以应用于腿部机器人。对于黑盒对抗性攻击，我们采用了三种搜索方法：随机搜索、差分进化和数值梯度下降方法。在OpenAI健身房环境下，对四足机器人Ant-v2和两足机器人人形v2进行了实验，发现在三种方法中，差分进化方法可以有效地找到最强的扭矩扰动。此外，我们还认识到四足机器人Ant-v2容易受到对抗性扰动的影响，而两足机器人人形v2对扰动具有很强的鲁棒性。因此，联合攻击可用于机器人行走不稳定性的主动诊断。



## **39. SafeNet: Mitigating Data Poisoning Attacks on Private Machine Learning**

SafeNet：缓解针对私人机器学习的数据中毒攻击 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.09986v1)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, the datasets used for training ML models might be under the control of an adversary mounting a data poisoning attack, and MPC prevents inspecting training sets to detect poisoning. We show that multiple MPC frameworks for private ML training are susceptible to backdoor and targeted poisoning attacks. To mitigate this, we propose SafeNet, a framework for building ensemble models in MPC with formal guarantees of robustness to data poisoning attacks. We extend the security definition of private ML training to account for poisoning and prove that our SafeNet design satisfies the definition. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models. For instance, SafeNet reduces backdoor attack success from 100% to 0% for a neural network model, while achieving 39x faster training and 36x less communication than the four-party MPC framework of Dalskov et al.

摘要: 安全多方计算(MPC)已被提出，以允许多个相互不信任的数据所有者联合训练机器学习(ML)模型。然而，用于训练ML模型的数据集可能处于发起数据中毒攻击的对手的控制之下，并且MPC阻止检查训练集以检测中毒。我们表明，用于私人ML训练的多个MPC框架容易受到后门和有针对性的中毒攻击。为了缓解这一问题，我们提出了SafeNet框架，用于在MPC中构建集成模型，并正式保证对数据中毒攻击的健壮性。我们扩展了私人ML训练的安全定义以解释中毒，并证明了我们的SafeNet设计满足该定义。我们在几个机器学习数据集和模型上展示了SafeNet的效率、准确性和对中毒的弹性。例如，对于神经网络模型，SafeNet将后门攻击成功率从100%降低到0%，同时实现了比Dalskov等人的四方MPC框架快39倍的训练和36倍的通信。



## **40. Adversarial Sample Detection for Speaker Verification by Neural Vocoders**

用于神经声码器说话人确认的对抗性样本检测 cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2107.00309v4)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.

摘要: 自动说话人验证(ASV)是生物特征识别的重要技术之一，在安全关键应用中得到了广泛的应用。然而，ASV在最近出现的对抗性攻击中非常脆弱，但针对它们的有效对策有限。在本文中，我们采用神经声码器来识别ASV的对抗性样本。我们使用神经声码器对音频进行重新合成，发现原始音频和重新合成音频的ASV分数之间的差异是区分真实和敌对样本的一个很好的指标。据我们所知，这项工作是为检测ASV的时间域对手样本而采取的最早的技术方向之一，因此缺乏用于比较的既定基线。因此，我们实现了Griffin-Lim算法作为检测基线。所提出的方法在所有设置下都取得了优于基线的有效检测性能。我们还证明了检测框架中采用的神经声码器是与数据集无关的。我们的代码将是开源的，以供将来的工作做公平的比较。



## **41. Focused Adversarial Attacks**

集中的对抗性攻击 cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09624v1)

**Authors**: Thomas Cilloni, Charles Walter, Charles Fleming

**Abstracts**: Recent advances in machine learning show that neural models are vulnerable to minimally perturbed inputs, or adversarial examples. Adversarial algorithms are optimization problems that minimize the accuracy of ML models by perturbing inputs, often using a model's loss function to craft such perturbations. State-of-the-art object detection models are characterized by very large output manifolds due to the number of possible locations and sizes of objects in an image. This leads to their outputs being sparse and optimization problems that use them incur a lot of unnecessary computation.   We propose to use a very limited subset of a model's learned manifold to compute adversarial examples. Our \textit{Focused Adversarial Attacks} (FA) algorithm identifies a small subset of sensitive regions to perform gradient-based adversarial attacks. FA is significantly faster than other gradient-based attacks when a model's manifold is sparsely activated. Also, its perturbations are more efficient than other methods under the same perturbation constraints. We evaluate FA on the COCO 2017 and Pascal VOC 2007 detection datasets.

摘要: 机器学习的最新进展表明，神经模型很容易受到最小扰动输入或对抗性例子的影响。对抗性算法是一种优化问题，它通过扰动输入来最小化ML模型的准确性，通常使用模型的损失函数来设计这种扰动。由于图像中对象的可能位置和大小的数量，最新的对象检测模型的特征在于非常大的输出流形。这导致它们的输出是稀疏的，并且使用它们的优化问题会引起大量不必要的计算。我们建议使用模型的学习流形的一个非常有限的子集来计算对抗性例子。本文提出的聚焦对抗性攻击(FA)算法识别一小部分敏感区域进行基于梯度的对抗性攻击。当模型的流形被稀疏激活时，FA比其他基于梯度的攻击要快得多。在相同的摄动约束下，它的摄动比其他方法更有效。我们在COCO 2017和Pascal VOC 2007检测数据集上评估FA。



## **42. Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification**

通过决策区域量化提高对真实世界和最坏情况分布漂移的稳健性 cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09619v1)

**Authors**: Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca

**Abstracts**: The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.

摘要: 神经网络的可靠性对于它们在安全关键应用中的使用至关重要。现有方法通常旨在提高神经网络对真实世界分布变化(例如，常见的破坏和扰动、空间变换和自然对抗性示例)或最坏情况分布变化(例如，优化的对抗性示例)的稳健性。在这项工作中，我们提出了决策区域量化(DRQ)算法，以提高任何可微预训练模型对真实世界和最坏情况下数据分布漂移的稳健性。DRQ分析给定数据点附近的局部决策区域的稳健性，以做出更可靠的预测。通过证明DRQ算法有效地平滑了决策面上的虚假局部极值，我们在理论上激励了DRQ算法。此外，我们提出了一种使用定向和非定向对抗性攻击的实现方案。一项广泛的经验评估表明，DRQ提高了对抗性和非对抗性训练模型在几个计算机视觉基准数据集上针对真实世界和最坏情况分布变化的稳健性。



## **43. Transferable Physical Attack against Object Detection with Separable Attention**

注意力可分离的可转移物理攻击目标检测 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09592v1)

**Authors**: Yu Zhang, Zhiqiang Gong, Yichuang Zhang, YongQian Li, Kangcheng Bin, Jiahao Qi, Wei Xue, Ping Zhong

**Abstracts**: Transferable adversarial attack is always in the spotlight since deep learning models have been demonstrated to be vulnerable to adversarial samples. However, existing physical attack methods do not pay enough attention on transferability to unseen models, thus leading to the poor performance of black-box attack.In this paper, we put forward a novel method of generating physically realizable adversarial camouflage to achieve transferable attack against detection models. More specifically, we first introduce multi-scale attention maps based on detection models to capture features of objects with various resolutions. Meanwhile, we adopt a sequence of composite transformations to obtain the averaged attention maps, which could curb model-specific noise in the attention and thus further boost transferability. Unlike the general visualization interpretation methods where model attention should be put on the foreground object as much as possible, we carry out attack on separable attention from the opposite perspective, i.e. suppressing attention of the foreground and enhancing that of the background. Consequently, transferable adversarial camouflage could be yielded efficiently with our novel attention-based loss function. Extensive comparison experiments verify the superiority of our method to state-of-the-art methods.

摘要: 可转移的对抗性攻击一直是人们关注的焦点，因为深度学习模型已被证明容易受到对抗性样本的影响。针对现有物理攻击方法对不可见模型的可转移性不够重视，导致黑盒攻击性能较差的问题，提出了一种新的生成物理可实现的对抗伪装的方法来实现对检测模型的可转移攻击。更具体地说，我们首先引入了基于检测模型的多尺度注意图来捕捉不同分辨率目标的特征。同时，我们采用一系列的复合变换来获得平均注意图，从而抑制了注意力中的特定模型噪声，从而进一步提高了注意力的可转移性。与一般的可视化解释方法将模型注意力尽可能地放在前景对象上不同，我们从相反的角度对可分离的注意进行攻击，即抑制前景的注意和增强背景的注意。因此，新的基于注意力的损失函数可以有效地产生可转移的对抗性伪装。大量的对比实验验证了该方法相对于最新方法的优越性。



## **44. On Trace of PGD-Like Adversarial Attacks**

关于类PGD对抗性攻击的踪迹 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09586v1)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstracts**: Adversarial attacks pose safety and security concerns for deep learning applications. Yet largely imperceptible, a strong PGD-like attack may leave strong trace in the adversarial example. Since attack triggers the local linearity of a network, we speculate network behaves in different extents of linearity for benign examples and adversarial examples. Thus, we construct Adversarial Response Characteristics (ARC) features to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it shows a gradually varying pattern from benign example to adversarial example, as the later leads to Sequel Attack Effect (SAE). ARC feature can be used for informed attack detection (perturbation magnitude is known) with binary classifier, or uninformed attack detection (perturbation magnitude is unknown) with ordinal regression. Due to the uniqueness of SAE to PGD-like attacks, ARC is also capable of inferring other attack details such as loss function, or the ground-truth label as a post-processing defense. Qualitative and quantitative evaluations manifest the effectiveness of ARC feature on CIFAR-10 w/ ResNet-18 and ImageNet w/ ResNet-152 and SwinT-B-IN1K with considerable generalization among PGD-like attacks despite domain shift. Our method is intuitive, light-weighted, non-intrusive, and data-undemanding.

摘要: 对抗性攻击给深度学习应用程序带来了安全和安保问题。然而，在很大程度上潜移默化的，一次强大的类似PGD的攻击可能会在对手的例子中留下强烈的痕迹。由于攻击触发了网络的局部线性，我们推测对于良性示例和恶意示例，网络的行为具有不同程度的线性。因此，我们构造了对抗性反应特征(ARC)特征来反映模型在输入附近的梯度一致性，以指示线性程度。在一定条件下，它呈现出从良性范例到对抗性范例的渐变模式，后者会导致后续攻击效应(SAE)。圆弧特征可以用于二值分类器的知情攻击检测(扰动幅度已知)，也可以用于有序回归的不知情攻击检测(扰动幅度未知)。由于SAE到PGD类攻击的独特性，ARC还能够推断其他攻击细节，如损失函数或地面事实标签作为后处理防御。定性和定量评估表明，ARC特征在CIFAR-10 w/ResNet-18和ImageNet w/ResNet-152和Swint-B-IN1K上的有效性，在类似PGD的攻击中具有相当大的泛化能力，尽管域发生了变化。我们的方法是直观的、轻量级的、非侵入性的、不需要数据的。



## **45. Defending Against Adversarial Attacks by Energy Storage Facility**

利用储能设施防御敌意攻击 cs.CR

arXiv admin note: text overlap with arXiv:1904.06606 by other authors

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09522v1)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstracts**: Adversarial attacks on data-driven algorithms applied in pow-er system will be a new type of threat on grid security. Litera-ture has demonstrated the adversarial attack on deep-neural network can significantly misleading the load forecast of a power system. However, it is unclear how the new type of at-tack impact on the operation of grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which account for around 20 million dollars. When wind-energy penetration increases to over 40%, the 5% adver-sarial attack will inflate the generation cost by 23%. Our re-search discovers a novel approach of defending against the adversarial attack: investing on energy-storage system. All current literature focuses on developing algorithm to defending against adversarial attack. We are the first research revealing the capability of using facility in physical system to defending against the adversarial algorithm attack in a system of Internet of Thing, such as smart grid system.

摘要: 针对电力系统中应用的数据驱动算法的对抗性攻击将是一种新型的网格安全威胁。已有文献表明，对深度神经网络的敌意攻击会严重误导电力系统的负荷预测。然而，目前还不清楚这种新型的AT-TACK对电网系统的运行有何影响。在这项研究中，我们证明了对抗性算法攻击导致了显著的成本增加风险，这种风险将随着间歇性可再生能源的日益普及而加剧。在德克萨斯州，5%的对抗性攻击可以在一个季度内使总发电成本增加17%，约占2000万美元。当风能渗透率增加到40%以上时，5%的风能侵袭将使发电成本增加23%。我们的研究发现了一种防御对手攻击的新方法：投资于储能系统。目前所有的文献都集中在开发算法来防御对手攻击。我们首次揭示了在物联网系统中利用物理系统中的便利来防御对抗性算法攻击的能力，例如智能电网系统。



## **46. Enhancing the Transferability of Adversarial Examples via a Few Queries**

通过几个问题增强对抗性例句的可转移性 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09518v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Due to the vulnerability of deep neural networks, the black-box attack has drawn great attention from the community. Though transferable priors decrease the query number of the black-box query attacks in recent efforts, the average number of queries is still larger than 100, which is easily affected by the number of queries limit policy. In this work, we propose a novel method called query prior-based method to enhance the family of fast gradient sign methods and improve their attack transferability by using a few queries. Specifically, for the untargeted attack, we find that the successful attacked adversarial examples prefer to be classified as the wrong categories with higher probability by the victim model. Therefore, the weighted augmented cross-entropy loss is proposed to reduce the gradient angle between the surrogate model and the victim model for enhancing the transferability of the adversarial examples. Theoretical analysis and extensive experiments demonstrate that our method could significantly improve the transferability of gradient-based adversarial attacks on CIFAR10/100 and ImageNet and outperform the black-box query attack with the same few queries.

摘要: 由于深度神经网络的脆弱性，此次黑匣子攻击引起了社会各界的高度关注。虽然可转移先验在最近的努力中减少了黑盒查询攻击的查询数，但平均查询数仍然大于100，这很容易受到查询数限制策略的影响。在这项工作中，我们提出了一种新的方法，称为基于查询优先的方法，以增强一族快速梯度符号方法，并通过使用几个查询来提高它们的攻击可转移性。具体地说，对于非定向攻击，我们发现被攻击成功的对抗性例子更倾向于被受害者模型分类为概率更高的错误类别。因此，为了提高对抗性例子的可转移性，提出了加权增广交叉熵损失来减小代理模型和受害者模型之间的梯度角。理论分析和大量实验表明，该方法可以显著提高基于梯度的攻击在CIFAR10/100和ImageNet上的可转移性，并在相同的较少查询数下优于黑盒查询攻击。



## **47. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2204.09803v2)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 图卷积网络(GCNS)容易受到微小的敌意扰动，这是一种严重的威胁，在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，称为图通用对抗防御(GARD)。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显着提高了几个已建立的GCN对多个对手攻击的稳健性，并且远远超过了最先进的防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **48. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

多智能体强化学习中的稀疏对抗性攻击 cs.AI

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09362v1)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).

摘要: 协作多智能体强化学习(CMARL)有很多实际应用，但已有的cMARL算法训练的策略在实际应用中不够健壮。针对RL系统的对抗性攻击也有很多方法，这意味着RL系统可能会遭受对抗性攻击，但大多数方法都集中在单个代理RL上。本文提出了一种针对cMARL系统的稀疏对抗攻击。我们使用带正则化的(MA)RL来训练攻击策略。我们的实验表明，当团队中只有一个或几个代理(例如，8个代理中的1个或25个代理中的5个)在几个时间步骤(例如，总共40个时间步骤中的攻击3个)受到攻击时，由当前cMARL算法训练的策略会获得较差的性能。



## **49. Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution**

基于反向分布的贝叶斯神经网络后门攻击 cs.CR

9 pages, 7 figures

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09167v1)

**Authors**: Zhixin Pan, Prabhat Mishra

**Abstracts**: Due to cost and time-to-market constraints, many industries outsource the training process of machine learning models (ML) to third-party cloud service providers, popularly known as ML-asa-Service (MLaaS). MLaaS creates opportunity for an adversary to provide users with backdoored ML models to produce incorrect predictions only in extremely rare (attacker-chosen) scenarios. Bayesian neural networks (BNN) are inherently immune against backdoor attacks since the weights are designed to be marginal distributions to quantify the uncertainty. In this paper, we propose a novel backdoor attack based on effective learning and targeted utilization of reverse distribution. This paper makes three important contributions. (1) To the best of our knowledge, this is the first backdoor attack that can effectively break the robustness of BNNs. (2) We produce reverse distributions to cancel the original distributions when the trigger is activated. (3) We propose an efficient solution for merging probability distributions in BNNs. Experimental results on diverse benchmark datasets demonstrate that our proposed attack can achieve the attack success rate (ASR) of 100%, while the ASR of the state-of-the-art attacks is lower than 60%.

摘要: 由于成本和上市时间的限制，许多行业将机器学习模型(ML)的培训过程外包给第三方云服务提供商，通常称为ML-ASA-Service(MLaaS)。MLaaS为对手创造了机会，为用户提供背道而驰的ML模型，只有在极其罕见的(攻击者选择的)情况下才能产生错误的预测。贝叶斯神经网络(BNN)天生不受后门攻击，因为权重被设计为边际分布来量化不确定性。本文提出了一种新的基于有效学习和定向利用反向分布的后门攻击方法。本文有三个重要贡献。(1)据我们所知，这是第一个可以有效破坏BNN健壮性的后门攻击。(2)当触发器被激活时，我们产生反向分布来抵消原始分布。(3)提出了一种在BNN中合并概率分布的有效解决方案。在不同基准数据集上的实验结果表明，我们提出的攻击可以达到100%的攻击成功率(ASR)，而最新的攻击的ASR低于60%。



## **50. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

基于RIS辅助干扰接收机的6G无线网络VLC物理层安全 cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09026v1)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.

摘要: 可见光通信(VLC)是未来6G网络最有前途的使能技术之一，可以克服基于射频(RF)的通信限制，因为它具有更宽的带宽、更高的数据速率和更高的效率。然而，从安全的角度来看，VLC受到所有已知的无线通信安全威胁(例如，窃听和完整性攻击)。为此，安全研究人员提出了创新的物理层安全(PLS)解决方案来保护此类通信。在不同的解决方案中，新型的反射智能表面(RIS)技术与VLC相结合已经在最近的工作中被成功地展示出来，以提高VLC的通信容量。然而，到目前为止，文献仍然缺乏分析和解决方案来展示基于RIS的VLC通信的偏最小二乘能力。在本文中，我们通过水印盲物理层安全(WBPLSec)算法将水印和干扰基元相结合来保护物理层的VLC通信。我们的解决方案利用RIS技术来提高通信的安全属性。通过使用优化框架，我们可以计算RIS相位，以最大化房间中预定义区域内的WBPLSec干扰方案。特别是，与没有RIS的场景相比，我们的方案在保密能力方面提高了性能，而不需要假设对手的位置。我们通过数值评估验证了RIS辅助解决方案对提高VLC室内场景中合法干扰接收机的保密容量的积极影响。我们的结果表明，RIS技术的引入扩展了安全通信发生的区域，并且随着RIS单元数量的增加，中断概率降低。



