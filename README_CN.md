# Latest Adversarial Attack Papers
**update at 2022-12-08 20:52:13**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Multiple Perturbation Attack: Attack Pixelwise Under Different $\ell_p$-norms For Better Adversarial Performance**

多重扰动攻击：在不同的$\ell_p$规范下进行像素攻击以获得更好的对抗性能 cs.CV

18 pages, 8 figures, 7 tables

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2212.03069v2) [paper-pdf](http://arxiv.org/pdf/2212.03069v2)

**Authors**: Ngoc N. Tran, Anh Tuan Bui, Dinh Phung, Trung Le

**Abstract**: Adversarial machine learning has been both a major concern and a hot topic recently, especially with the ubiquitous use of deep neural networks in the current landscape. Adversarial attacks and defenses are usually likened to a cat-and-mouse game in which defenders and attackers evolve over the time. On one hand, the goal is to develop strong and robust deep networks that are resistant to malicious actors. On the other hand, in order to achieve that, we need to devise even stronger adversarial attacks to challenge these defense models. Most of existing attacks employs a single $\ell_p$ distance (commonly, $p\in\{1,2,\infty\}$) to define the concept of closeness and performs steepest gradient ascent w.r.t. this $p$-norm to update all pixels in an adversarial example in the same way. These $\ell_p$ attacks each has its own pros and cons; and there is no single attack that can successfully break through defense models that are robust against multiple $\ell_p$ norms simultaneously. Motivated by these observations, we come up with a natural approach: combining various $\ell_p$ gradient projections on a pixel level to achieve a joint adversarial perturbation. Specifically, we learn how to perturb each pixel to maximize the attack performance, while maintaining the overall visual imperceptibility of adversarial examples. Finally, through various experiments with standardized benchmarks, we show that our method outperforms most current strong attacks across state-of-the-art defense mechanisms, while retaining its ability to remain clean visually.

摘要: 对抗性机器学习近年来一直是一个重要的关注和热点问题，尤其是在当前深度神经网络的普遍使用下。对抗性的攻击和防御通常被比作猫和老鼠的游戏，其中防御者和攻击者随着时间的推移而演变。一方面，目标是开发强大而健壮的深层网络，抵御恶意行为者。另一方面，为了实现这一目标，我们需要设计出更强大的对抗性攻击来挑战这些防御模式。现有的攻击大多使用单个$ell_p$距离(通常是$p in{1，2，inty$)来定义贴近度的概念，并执行最陡的梯度上升w.r.t.此$p$-规范以相同的方式更新对抗性示例中的所有像素。这些$\ell_p$攻击各有优缺点；没有一种攻击可以成功突破同时对多个$ell_p$规范具有健壮性的防御模型。受这些观察结果的启发，我们提出了一种自然的方法：将不同的$\ell_p$梯度投影组合在一个像素级别上，以实现联合对抗性扰动。具体地说，我们学习了如何扰动每个像素以最大化攻击性能，同时保持对抗性例子的整体视觉不可感知性。最后，通过标准化基准的各种实验，我们表明我们的方法在保持视觉清洁的能力的同时，在最先进的防御机制上优于目前大多数的强攻击。



## **2. Universal Backdoor Attacks Detection via Adaptive Adversarial Probe**

基于自适应对抗性探测的通用后门攻击检测 cs.CV

8 pages, 8 figures

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2209.05244v3) [paper-pdf](http://arxiv.org/pdf/2209.05244v3)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstract**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor attacks detection. Most detection methods are designed to verify whether a model is infected with presumed types of backdoor attacks, yet the adversary is likely to generate diverse backdoor attacks in practice that are unforeseen to defenders, which challenge current detection strategies. In this paper, we focus on this more challenging scenario and propose a universal backdoor attacks detection method named Adaptive Adversarial Probe (A2P). Specifically, we posit that the challenge of universal backdoor attacks detection lies in the fact that different backdoor attacks often exhibit diverse characteristics in trigger patterns (i.e., sizes and transparencies). Therefore, our A2P adopts a global-to-local probing framework, which adversarially probes images with adaptive regions/budgets to fit various backdoor triggers of different sizes/transparencies. Regarding the probing region, we propose the attention-guided region generation strategy that generates region proposals with different sizes/locations based on the attention of the target model, since trigger regions often manifest higher model activation. Considering the attack budget, we introduce the box-to-sparsity scheduling that iteratively increases the perturbation budget from box to sparse constraint, so that we could better activate different latent backdoors with different transparencies. Extensive experiments on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins (+12%).

摘要: 大量证据表明，深度神经网络(DNN)很容易受到后门攻击，这推动了后门攻击检测的发展。大多数检测方法旨在验证模型是否感染了假定类型的后门攻击，但对手在实践中可能会产生防御者无法预见的各种后门攻击，这对当前的检测策略构成了挑战。在本文中，我们针对这种更具挑战性的场景，提出了一种通用的后门攻击检测方法--自适应对抗探测(A2P)。具体地说，我们假设通用后门攻击检测的挑战在于不同的后门攻击通常在触发模式(即大小和透明度)上表现出不同的特征。因此，我们的A2P采用了全局到局部探测框架，该框架相反地探测具有自适应区域/预算的图像，以适应不同大小/透明度的各种后门触发。对于探测区域，我们提出了注意力引导的区域生成策略，该策略根据目标模型的关注度生成不同大小/位置的区域建议，因为触发区域通常表现出较高的模型活跃度。考虑到攻击预算，我们引入了盒到稀疏调度，将扰动预算从盒子迭代增加到稀疏约束，以便更好地激活不同透明度的不同潜在后门。在多个数据集(CIFAR-10，GTSRB，Tiny-ImageNet)上的大量实验表明，我们的方法比最先进的基线方法有很大的优势(+12%)。



## **3. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

Leno：具有可学习噪声的对抗性鲁棒显著目标检测网络 cs.CV

8 pages, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-12-07    [abs](http://arxiv.org/abs/2210.15392v2) [paper-pdf](http://arxiv.org/pdf/2210.15392v2)

**Authors**: He Wang, Lin Wan, He Tang

**Abstract**: Pixel-wise prediction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remarkable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust saliency (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected conditional random field (CRF). Different from ROSA that relies on various pre- and post-processings, this paper proposes a light-weight Learnable Noise (LeNo) to defend adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also on clean images, which contributes stronger robustness for SOD. Our code is available at https://github.com/ssecv/LeNo.

摘要: 基于深度神经网络的像素预测已成为显著目标检测的一种有效范例，并取得了显著的效果。然而，很少有SOD模型对人类视觉上不可察觉的对抗性攻击具有健壮性。该算法首先对预分块后的超像素进行置乱处理，然后利用稠密连接的条件随机场(CRF)对粗略显著图进行细化。不同于ROSA依赖于各种前后处理，本文提出了一种轻量级可学习噪声(Leno)来防御对SOD模型的敌意攻击。Leno保持了SOD模型在对抗性图像和干净图像上的准确性，以及推理速度。一般来说，LENO由简单的浅层噪声和噪声估计组成，分别嵌入到任意SOD网络的编码器和译码中。受人类视觉注意机制中心先验的启发，我们用十字形高斯分布对浅层噪声进行初始化，以更好地防御对手的攻击。所提出的噪声估计只需修改解码器的一个通道，而不是为后处理增加额外的网络组件。通过在最新的RGB和RGB-D SOD网络上进行深度监督噪声解耦训练，Leno不仅在对抗性图像上而且在干净图像上都优于以往的工作，这为SOD提供了更强的稳健性。我们的代码可以在https://github.com/ssecv/LeNo.上找到



## **4. Pre-trained Encoders in Self-Supervised Learning Improve Secure and Privacy-preserving Supervised Learning**

自我监督学习中的预训练编码器改进安全和隐私保护的监督学习 cs.CR

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2212.03334v1) [paper-pdf](http://arxiv.org/pdf/2212.03334v1)

**Authors**: Hongbin Liu, Wenjie Qu, Jinyuan Jia, Neil Zhenqiang Gong

**Abstract**: Classifiers in supervised learning have various security and privacy issues, e.g., 1) data poisoning attacks, backdoor attacks, and adversarial examples on the security side as well as 2) inference attacks and the right to be forgotten for the training data on the privacy side. Various secure and privacy-preserving supervised learning algorithms with formal guarantees have been proposed to address these issues. However, they suffer from various limitations such as accuracy loss, small certified security guarantees, and/or inefficiency. Self-supervised learning is an emerging technique to pre-train encoders using unlabeled data. Given a pre-trained encoder as a feature extractor, supervised learning can train a simple yet accurate classifier using a small amount of labeled training data. In this work, we perform the first systematic, principled measurement study to understand whether and when a pre-trained encoder can address the limitations of secure or privacy-preserving supervised learning algorithms. Our key findings are that a pre-trained encoder substantially improves 1) both accuracy under no attacks and certified security guarantees against data poisoning and backdoor attacks of state-of-the-art secure learning algorithms (i.e., bagging and KNN), 2) certified security guarantees of randomized smoothing against adversarial examples without sacrificing its accuracy under no attacks, 3) accuracy of differentially private classifiers, and 4) accuracy and/or efficiency of exact machine unlearning.

摘要: 有监督学习中的分类器存在各种各样的安全和隐私问题，例如：1)安全端的数据中毒攻击、后门攻击和敌意例子；2)隐私端的推理攻击和训练数据的遗忘权。各种形式保证的安全和隐私保护的监督学习算法已经被提出来解决这些问题。然而，它们受到各种限制，例如准确性损失、较小的认证安全保证和/或效率低下。自我监督学习是一种新兴的技术，可以使用未标记的数据来预先训练编码者。给定一个预先训练的编码器作为特征提取者，监督学习可以使用少量的标记训练数据来训练简单而准确的分类器。在这项工作中，我们进行了第一次系统的、原则性的测量研究，以了解预先训练的编码器是否以及何时可以解决安全或隐私保护的监督学习算法的限制。我们的主要发现是，预先训练的编码器显著提高了1)无攻击情况下的准确率，以及针对最先进的安全学习算法(即Bging和KNN)的数据中毒和后门攻击的认证安全保证，2)针对敌意示例的随机平滑的认证安全保证而不牺牲其在无攻击情况下的精度，3)差分私有分类器的准确性，以及4)精确机器遗忘的准确性和/或效率。



## **5. Robust Models are less Over-Confident**

稳健的模型不那么过度自信 cs.CV

accepted at NeurIPS 2022

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.05938v2) [paper-pdf](http://arxiv.org/pdf/2210.05938v2)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation

摘要: 尽管卷积神经网络(CNN)在许多计算机视觉任务的学术基准中取得了成功，但它们在现实世界中的应用仍然面临着根本性的挑战。这些悬而未决的问题之一是固有的健壮性不足，这一点从对抗性攻击的惊人有效性中可见一斑。目前的攻击方法能够通过向输入添加特定但少量的噪声来操纵网络的预测。反过来，对抗性训练(AT)的目的是通过将对抗性样本包括在训练集中来实现对此类攻击的健壮性，并且理想地实现更好的模型泛化能力。然而，对由此产生的超越对抗性稳健性的稳健性模型的深入分析仍然悬而未决。在这篇文章中，我们实证分析了各种对抗训练的模型，这些模型在面对最先进的攻击时获得了很高的稳健精度，我们发现AT有一个有趣的副作用：它导致模型对他们的决策不那么过度自信，即使是在干净的数据上也是如此。此外，我们对稳健模型的分析表明，不仅AT而且模型的构件(如激活函数和池化)对模型的预测置信度有很大的影响。数据与项目网站：https://github.com/GeJulia/robustness_confidences_evaluation



## **6. On the tightness of linear relaxation based robustness certification methods**

基于线性松弛的稳健性证明方法的紧性 cs.LG

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2210.00178v2) [paper-pdf](http://arxiv.org/pdf/2210.00178v2)

**Authors**: Cheng Tang

**Abstract**: There has been a rapid development and interest in adversarial training and defenses in the machine learning community in the recent years. One line of research focuses on improving the performance and efficiency of adversarial robustness certificates for neural networks \cite{gowal:19, wong_zico:18, raghunathan:18, WengTowardsFC:18, wong:scalable:18, singh:convex_barrier:19, Huang_etal:19, single-neuron-relax:20, Zhang2020TowardsSA}. While each providing a certification to lower (or upper) bound the true distortion under adversarial attacks via relaxation, less studied was the tightness of relaxation. In this paper, we analyze a family of linear outer approximation based certificate methods via a meta algorithm, IBP-Lin. The aforementioned works often lack quantitative analysis to answer questions such as how does the performance of the certificate method depend on the network configuration and the choice of approximation parameters. Under our framework, we make a first attempt at answering these questions, which reveals that the tightness of linear approximation based certification can depend heavily on the configuration of the trained networks.

摘要: 近年来，机器学习界对对抗性训练和防御有了迅速的发展和兴趣。其中一项研究集中于提高神经网络对抗健壮性证书的性能和效率。{gowal：19，Wong_zico：18，Raghunathan：18，WengTowardsFC：18，Wong：Scalable：18，Singh：凸障：19，Huang_Etal：19，单神经元松弛：20，Zhang 2020TowardsSA}。虽然每一个都提供了一个证明，通过放松来降低(或上限)在对抗性攻击下的真实失真，但对放松的紧密性的研究较少。本文通过一个元算法IBP-LIN分析了一类基于线性外逼近的证书方法。上述工作往往缺乏定量的分析来回答诸如证书方法的性能如何依赖于网络配置和近似参数的选择等问题。在我们的框架下，我们首次尝试回答这些问题，这表明基于线性近似的认证的紧密性在很大程度上依赖于训练网络的配置。



## **7. CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models**

CodeAttack：针对预先训练的编程语言模型的基于代码的对抗性攻击 cs.CL

**SubmitDate**: 2022-12-06    [abs](http://arxiv.org/abs/2206.00052v2) [paper-pdf](http://arxiv.org/pdf/2206.00052v2)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.

摘要: 预先训练的编程语言(PL)模型(如CodeT5、CodeBERT、GraphCodeBERT等)有可能自动化涉及代码理解和代码生成的软件工程任务。然而，这些模型在代码的自然通道中运行，即它们主要关注人类对代码的理解。它们对输入的变化不是很健壮，因此，在自然通道中可能容易受到对抗性攻击。我们提出了一个简单而有效的黑盒攻击模型CodeAttack，它使用代码结构来生成有效、高效和不可察觉的对抗性代码样本，并展示了最新的PL模型对代码特定的对抗性攻击的脆弱性。我们评估了CodeAttack在几个代码-代码(翻译和修复)和代码-NL(摘要)任务上跨不同编程语言的可移植性。CodeAttack超越了最先进的对抗性NLP攻击模型，在更高效、更隐蔽、更一致和更流畅的同时，实现了最佳的整体性能下降。代码可在https://github.com/reddy-lab-code-research/CodeAttack.上找到



## **8. StyleGAN as a Utility-Preserving Face De-identification Method**

StyleGan作为一种保持效用的人脸去身份识别方法 cs.CV

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02611v1) [paper-pdf](http://arxiv.org/pdf/2212.02611v1)

**Authors**: Seyyed Mohammad Sadegh Moosavi Khorzooghi, Shirin Nilizadeh

**Abstract**: Several face de-identification methods have been proposed to preserve users' privacy by obscuring their faces. These methods, however, can degrade the quality of photos, and they usually do not preserve the utility of faces, e.g., their age, gender, pose, and facial expression. Recently, advanced generative adversarial network models, such as StyleGAN, have been proposed, which generate realistic, high-quality imaginary faces. In this paper, we investigate the use of StyleGAN in generating de-identified faces through style mixing, where the styles or features of the target face and an auxiliary face get mixed to generate a de-identified face that carries the utilities of the target face. We examined this de-identification method with respect to preserving utility and privacy, by implementing several face detection, verification, and identification attacks. Through extensive experiments and also comparing with two state-of-the-art face de-identification methods, we show that StyleGAN preserves the quality and utility of the faces much better than the other approaches and also by choosing the style mixing levels correctly, it can preserve the privacy of the faces much better than other methods.

摘要: 已经提出了几种人脸去识别方法，通过模糊用户的脸来保护他们的隐私。然而，这些方法会降低照片的质量，并且它们通常不保留人脸的实用性，例如他们的年龄、性别、姿势和面部表情。最近，一些先进的生成性对抗网络模型被提出，如StyleGAN，它们可以生成逼真的、高质量的想象人脸。在本文中，我们研究了StyleGAN在通过样式混合生成去身份人脸中的使用，即将目标人脸和辅助人脸的样式或特征混合以生成携带目标人脸效用的去身份人脸。我们通过实施几种人脸检测、验证和身份识别攻击，从保护实用性和隐私方面检查了这种去身份方法。通过大量的实验，并与两种最新的人脸去识别方法进行了比较，结果表明，StyleGAN比其他方法更好地保持了人脸的质量和实用性，并且通过正确选择样式混合级别，比其他方法更好地保护了人脸的隐私。



## **9. Enhancing Quantum Adversarial Robustness by Randomized Encodings**

利用随机编码增强量子对抗的健壮性 quant-ph

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02531v1) [paper-pdf](http://arxiv.org/pdf/2212.02531v1)

**Authors**: Weiyuan Gong, Dong Yuan, Weikang Li, Dong-Ling Deng

**Abstract**: The interplay between quantum physics and machine learning gives rise to the emergent frontier of quantum machine learning, where advanced quantum learning models may outperform their classical counterparts in solving certain challenging problems. However, quantum learning systems are vulnerable to adversarial attacks: adding tiny carefully-crafted perturbations on legitimate input samples can cause misclassifications. To address this issue, we propose a general scheme to protect quantum learning systems from adversarial attacks by randomly encoding the legitimate data samples through unitary or quantum error correction encoders. In particular, we rigorously prove that both global and local random unitary encoders lead to exponentially vanishing gradients (i.e. barren plateaus) for any variational quantum circuits that aim to add adversarial perturbations, independent of the input data and the inner structures of adversarial circuits and quantum classifiers. In addition, we prove a rigorous bound on the vulnerability of quantum classifiers under local unitary adversarial attacks. We show that random black-box quantum error correction encoders can protect quantum classifiers against local adversarial noises and their robustness increases as we concatenate error correction codes. To quantify the robustness enhancement, we adapt quantum differential privacy as a measure of the prediction stability for quantum classifiers. Our results establish versatile defense strategies for quantum classifiers against adversarial perturbations, which provide valuable guidance to enhance the reliability and security for both near-term and future quantum learning technologies.

摘要: 量子物理和机器学习之间的相互作用催生了量子机器学习的新兴前沿，在解决某些具有挑战性的问题方面，先进的量子学习模型可能会比经典模型表现得更好。然而，量子学习系统很容易受到敌意攻击：在合法的输入样本上添加精心设计的微小扰动可能会导致错误分类。为了解决这个问题，我们提出了一个通用的方案来保护量子学习系统免受敌意攻击，该方案通过酉码或量子纠错码对合法数据样本进行随机编码。特别地，我们严格地证明了对于任何旨在添加对抗性扰动的变分量子电路，全局和局部随机么正编码器都会导致梯度指数地消失(即贫瘠的高原)，而与输入数据和对抗性电路和量子分类器的内部结构无关。此外，我们还证明了量子分类器在局部酉性攻击下的脆弱性的一个严格界。我们证明了随机黑盒量子纠错编码器可以保护量子分类器免受局部敌对噪声的影响，并且随着纠错码的级联，它们的健壮性也增强了。为了量化健壮性的增强，我们采用量子差分隐私作为量子分类器预测稳定性的衡量标准。我们的结果为量子分类器建立了针对敌意扰动的通用防御策略，为提高近期和未来量子学习技术的可靠性和安全性提供了有价值的指导。



## **10. Domain Constraints in Feature Space: Strengthening Robustness of Android Malware Detection against Realizable Adversarial Examples**

特征空间中的域约束：增强Android恶意软件检测对可实现的恶意示例的健壮性 cs.LG

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2205.15128v2) [paper-pdf](http://arxiv.org/pdf/2205.15128v2)

**Authors**: Hamid Bostani, Zhuoran Liu, Zhengyu Zhao, Veelasha Moonsamy

**Abstract**: Strengthening the robustness of machine learning-based Android malware detectors in the real world requires incorporating realizable adversarial examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. However, existing work focuses on generating RealAEs in the problem space, which is known to be time-consuming and impractical for adversarial training. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android malware properties in the feature space. More concretely, we extract feature-space domain constraints by learning meaningful feature dependencies from data and applying them by constructing a robust feature space. Our experiments on DREBIN, a well-known Android malware detector, demonstrate that our approach outperforms the state-of-the-art defense, Sec-SVM, against realistic gradient- and query-based attacks. Additionally, we demonstrate that generating feature-space RealAEs is faster than generating problem-space RealAEs, indicating its high applicability in adversarial training. We further validate the ability of our learned feature-space domain constraints in representing the Android malware properties by showing that (i) re-training detectors with our feature-space RealAEs largely improves model performance on similar problem-space RealAEs and (ii) using our feature-space domain constraints can help distinguish RealAEs from unrealizable AEs (unRealAEs).

摘要: 要增强基于机器学习的Android恶意软件检测器在现实世界中的健壮性，需要结合可实现的对抗实例(RealAE)，即满足Android恶意软件领域约束的AEs。然而，现有的工作主要集中于在问题空间中生成RealAEs，这对于对抗性训练来说是耗时和不切实际的。在本文中，我们建议在特征空间中生成RealAE，从而得到一个更简单、更有效的解决方案。我们的方法是由对功能空间中Android恶意软件属性的新解释驱动的。更具体地说，我们通过从数据中学习有意义的特征依赖关系并通过构造稳健的特征空间来应用它们来提取特征空间域约束。我们在著名的Android恶意软件检测器Drebin上的实验表明，我们的方法在抵抗现实的基于梯度和基于查询的攻击方面优于最先进的防御系统SEC-SVM。此外，我们还证明了生成特征空间RealEs的速度比生成问题空间RealEs的速度要快，这表明它在对抗性训练中具有很高的适用性。我们进一步验证了我们学习的特征空间域约束在表示Android恶意软件属性方面的能力，方法是：(I)用我们的特征空间RealAE重新训练检测器，极大地提高了类似问题空间RealAE的模型性能；(Ii)使用我们的特征空间域约束，可以帮助区分RealAE和不可实现的RealAE(UnRealAE)。



## **11. Adversarial Attacks on Spiking Convolutional Neural Networks for Event-based Vision**

基于事件视觉的尖峰卷积神经网络对抗性攻击 cs.CV

9 pages plus Supplementary Material. Accepted in Frontiers in  Neuroscience -- Neuromorphic Engineering

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2110.02929v3) [paper-pdf](http://arxiv.org/pdf/2110.02929v3)

**Authors**: Julian Büchel, Gregor Lenz, Yalun Hu, Sadique Sheik, Martino Sorbaro

**Abstract**: Event-based dynamic vision sensors provide very sparse output in the form of spikes, which makes them suitable for low-power applications. Convolutional spiking neural networks model such event-based data and develop their full energy-saving potential when deployed on asynchronous neuromorphic hardware. Event-based vision being a nascent field, the sensitivity of spiking neural networks to potentially malicious adversarial attacks has received little attention so far. We show how white-box adversarial attack algorithms can be adapted to the discrete and sparse nature of event-based visual data, and demonstrate smaller perturbation magnitudes at higher success rates than the current state-of-the-art algorithms. For the first time, we also verify the effectiveness of these perturbations directly on neuromorphic hardware. Finally, we discuss the properties of the resulting perturbations, the effect of adversarial training as a defense strategy, and future directions.

摘要: 基于事件的动态视觉传感器以尖峰的形式提供非常稀疏的输出，这使得它们适合低功率应用。卷积尖峰神经网络对这种基于事件的数据进行建模，并在部署在异步神经形态硬件上时开发其全部节能潜力。基于事件的视觉是一个新兴领域，到目前为止，尖峰神经网络对潜在恶意对手攻击的敏感度几乎没有受到关注。我们展示了白盒对抗性攻击算法如何适应基于事件的视觉数据的离散和稀疏的性质，并展示了与当前最先进的算法相比，更小的扰动幅度和更高的成功率。第一次，我们还直接在神经形态硬件上验证了这些扰动的有效性。最后，我们讨论了由此产生的扰动的性质，对抗性训练作为一种防御策略的效果，以及未来的发展方向。



## **12. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2203.16000v2) [paper-pdf](http://arxiv.org/pdf/2203.16000v2)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类别置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **13. FaceQAN: Face Image Quality Assessment Through Adversarial Noise Exploration**

基于对抗性噪声探测的人脸图像质量评价 cs.CV

The content of this paper was published in ICPR 2022

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02127v1) [paper-pdf](http://arxiv.org/pdf/2212.02127v1)

**Authors**: Žiga Babnik, Peter Peer, Vitomir Štruc

**Abstract**: Recent state-of-the-art face recognition (FR) approaches have achieved impressive performance, yet unconstrained face recognition still represents an open problem. Face image quality assessment (FIQA) approaches aim to estimate the quality of the input samples that can help provide information on the confidence of the recognition decision and eventually lead to improved results in challenging scenarios. While much progress has been made in face image quality assessment in recent years, computing reliable quality scores for diverse facial images and FR models remains challenging. In this paper, we propose a novel approach to face image quality assessment, called FaceQAN, that is based on adversarial examples and relies on the analysis of adversarial noise which can be calculated with any FR model learned by using some form of gradient descent. As such, the proposed approach is the first to link image quality to adversarial attacks. Comprehensive (cross-model as well as model-specific) experiments are conducted with four benchmark datasets, i.e., LFW, CFP-FP, XQLFW and IJB-C, four FR models, i.e., CosFace, ArcFace, CurricularFace and ElasticFace, and in comparison to seven state-of-the-art FIQA methods to demonstrate the performance of FaceQAN. Experimental results show that FaceQAN achieves competitive results, while exhibiting several desirable characteristics.

摘要: 目前最先进的人脸识别方法已经取得了令人印象深刻的性能，但无约束的人脸识别仍然是一个悬而未决的问题。人脸图像质量评估(FIQA)方法旨在估计输入样本的质量，这些输入样本可以帮助提供关于识别决策的置信度的信息，并最终在具有挑战性的场景中产生更好的结果。虽然近年来在人脸图像质量评估方面取得了很大进展，但为不同的人脸图像和FR模型计算可靠的质量分数仍然具有挑战性。在本文中，我们提出了一种新的人脸图像质量评估方法FaceQAN，该方法基于对抗性样本，依赖于对抗性噪声的分析，通过使用某种形式的梯度下降学习的任何FR模型都可以计算出对抗性噪声。因此，提出的方法是第一个将图像质量与对抗性攻击联系起来的方法。在四个基准数据集LFW、CFP-FP、XQLFW和IJB-C和四个FR模型CosFace、ArcFace、CurousarFace和ElasticFace上进行了全面的(跨模型和特定模型的)实验，并与现有的七种FIQA方法进行了比较，以验证FaceQAN的性能。实验结果表明，FaceQan在表现出几个理想的特征的同时，达到了竞争的效果。



## **14. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

具有信息增益的贝叶斯学习被证明是强健对抗防御的风险界限 cs.LG

Published at ICML 2022

**SubmitDate**: 2022-12-05    [abs](http://arxiv.org/abs/2212.02003v1) [paper-pdf](http://arxiv.org/pdf/2212.02003v1)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.

摘要: 提出了一种新的学习深度神经网络模型的算法，该模型具有较强的抗攻击能力。以前的算法表明，反向训练的贝叶斯神经网络(BNN)提供了更好的鲁棒性。我们认识到，用对抗性学习方法来逼近贝叶斯模型的多模式后验分布可能会导致模式崩溃，因此，该模型在稳健性和性能方面的成就是次优的。相反，我们首先提出防止模式崩溃，以更好地逼近多模式的后验分布。其次，基于健壮模型应该忽略扰动而只考虑输入的信息内容的直觉，我们概念化和制定了一个信息增益目标来衡量和强制从良性和对抗性训练实例中学习到的信息相似。重要的是。我们证明并证明了最小化信息收益目标使对手风险接近于传统的经验风险。我们相信，我们的努力为对抗性训练BNN的原则性方法奠定了基础。在CIFAR-10和STL-10数据集上，与对抗性训练和ADV-BNN相比，在具有0.035失真的PGD攻击下，我们的模型表现出了高达20%的健壮性。



## **15. Recognizing Object by Components with Human Prior Knowledge Enhances Adversarial Robustness of Deep Neural Networks**

利用人类先验知识进行目标识别增强了深度神经网络的对抗鲁棒性 cs.CV

Under review. Submitted to TPAMI on June 10, 2022. Major revision on  September 4, 2022

**SubmitDate**: 2022-12-04    [abs](http://arxiv.org/abs/2212.01806v1) [paper-pdf](http://arxiv.org/pdf/2212.01806v1)

**Authors**: Xiao Li, Ziqi Wang, Bo Zhang, Fuchun Sun, Xiaolin Hu

**Abstract**: Adversarial attacks can easily fool object recognition systems based on deep neural networks (DNNs). Although many defense methods have been proposed in recent years, most of them can still be adaptively evaded. One reason for the weak adversarial robustness may be that DNNs are only supervised by category labels and do not have part-based inductive bias like the recognition process of humans. Inspired by a well-known theory in cognitive psychology -- recognition-by-components, we propose a novel object recognition model ROCK (Recognizing Object by Components with human prior Knowledge). It first segments parts of objects from images, then scores part segmentation results with predefined human prior knowledge, and finally outputs prediction based on the scores. The first stage of ROCK corresponds to the process of decomposing objects into parts in human vision. The second stage corresponds to the decision process of the human brain. ROCK shows better robustness than classical recognition models across various attack settings. These results encourage researchers to rethink the rationality of currently widely-used DNN-based object recognition models and explore the potential of part-based models, once important but recently ignored, for improving robustness.

摘要: 敌意攻击可以很容易地欺骗基于深度神经网络(DNN)的目标识别系统。虽然近年来已经提出了许多防御方法，但大多数方法仍然可以自适应地规避。对抗健壮性较弱的一个原因可能是DNN只受类别标签的监督，并且不像人类的识别过程那样具有基于部分的归纳偏差。受认知心理学中的一种著名理论--构件识别理论的启发，我们提出了一种新的物体识别模型ROCK(利用人类的先验知识进行构件识别)。它首先从图像中分割出部分目标，然后利用预定义的人类先验知识对部分分割结果进行评分，最后根据评分结果输出预测。岩石的第一阶段对应于在人类视觉中将物体分解成部分的过程。第二个阶段对应于人脑的决策过程。ROCK在各种攻击环境下表现出比经典识别模型更好的健壮性。这些结果鼓励研究人员重新思考当前广泛使用的基于DNN的对象识别模型的合理性，并探索基于零件的模型的潜力，这些模型曾经很重要，但最近被忽视，以提高鲁棒性。



## **16. Imperceptible Adversarial Attack via Invertible Neural Networks**

基于逆神经网络的潜伏性敌意攻击 cs.CV

**SubmitDate**: 2022-12-04    [abs](http://arxiv.org/abs/2211.15030v2) [paper-pdf](http://arxiv.org/pdf/2211.15030v2)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.

摘要: 通过利用辅助梯度信息添加扰动或丢弃良性图像的现有细节是生成对抗性示例的两种常见方法。虽然视觉不可感知性是对抗性例子的理想属性，但传统的对抗性攻击仍然产生可追踪的对抗性扰动。在本文中，我们介绍了一种新的基于可逆神经网络(AdvINN)的对抗性攻击方法，以产生健壮且不可察觉的对抗性示例。具体而言，AdvINN充分利用了可逆神经网络的信息保持性，通过同时添加目标类的类特定语义信息和丢弃原类的判别信息来生成对抗性实例。在CIFAR-10、CIFAR-100和ImageNet-1K上的大量实验表明，所提出的AdvINN方法可以产生比现有方法更少的不可察觉的对抗性图像，并且与其他对抗性攻击相比，AdvINN产生更健壮的对抗性例子和更高的置信度。



## **17. LDL: A Defense for Label-Based Membership Inference Attacks**

低密度脂蛋白：一种基于标签的成员推理攻击防御方案 cs.LG

to appear in ACM AsiaCCS 2023

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2212.01688v1) [paper-pdf](http://arxiv.org/pdf/2212.01688v1)

**Authors**: Arezoo Rajabi, Dinuka Sahabandu, Luyao Niu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The data used to train deep neural network (DNN) models in applications such as healthcare and finance typically contain sensitive information. A DNN model may suffer from overfitting. Overfitted models have been shown to be susceptible to query-based attacks such as membership inference attacks (MIAs). MIAs aim to determine whether a sample belongs to the dataset used to train a classifier (members) or not (nonmembers). Recently, a new class of label based MIAs (LAB MIAs) was proposed, where an adversary was only required to have knowledge of predicted labels of samples. Developing a defense against an adversary carrying out a LAB MIA on DNN models that cannot be retrained remains an open problem.   We present LDL, a light weight defense against LAB MIAs. LDL works by constructing a high-dimensional sphere around queried samples such that the model decision is unchanged for (noisy) variants of the sample within the sphere. This sphere of label-invariance creates ambiguity and prevents a querying adversary from correctly determining whether a sample is a member or a nonmember. We analytically characterize the success rate of an adversary carrying out a LAB MIA when LDL is deployed, and show that the formulation is consistent with experimental observations. We evaluate LDL on seven datasets -- CIFAR-10, CIFAR-100, GTSRB, Face, Purchase, Location, and Texas -- with varying sizes of training data. All of these datasets have been used by SOTA LAB MIAs. Our experiments demonstrate that LDL reduces the success rate of an adversary carrying out a LAB MIA in each case. We empirically compare LDL with defenses against LAB MIAs that require retraining of DNN models, and show that LDL performs favorably despite not needing to retrain the DNNs.

摘要: 用于在医疗保健和金融等应用中训练深度神经网络(DNN)模型的数据通常包含敏感信息。DNN模型可能会出现过度拟合的问题。过适应的模型已经被证明容易受到基于查询的攻击，例如成员推理攻击(MIA)。MIA旨在确定样本是否属于用于训练分类器的数据集(成员)或不属于(非成员)。最近，一类新的基于标签的MIA(Label Based MIA)被提出，其中对手只需要知道样本的预测标签。针对在不能再训练的DNN模型上进行实验室MIA的对手开发防御仍然是一个悬而未决的问题。我们提出了低密度脂蛋白，一种针对实验室MIA的轻量级防御措施。低密度脂蛋白的工作原理是围绕查询的样本构建一个高维球体，使得球体内样本的(噪声)变体的模型决策保持不变。这种标签不变性的范围造成了歧义，并阻止查询对手正确地确定样本是成员还是非成员。我们分析了当部署低密度脂蛋白时对手进行实验室MIA的成功率，并表明该公式与实验观察一致。我们在七个数据集--CIFAR-10、CIFAR-100、GTSRB、Face、Purchase、Location和Texas--上评估LDL，并使用不同大小的训练数据。所有这些数据集都已被SOTA实验室MIA使用。我们的实验表明，在每种情况下，低密度脂蛋白都会降低对手进行实验室MIA的成功率。我们经验性地比较了低密度脂蛋白与对需要重新训练DNN模型的实验室MIA的防御，并表明尽管不需要重新训练DNN，但低密度脂蛋白的表现良好。



## **18. Sparta: Spatially Attentive and Adversarially Robust Activation**

斯巴达：空间关注和强大的对抗性激活 cs.LG

25 pages, 5 figures

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2105.08269v2) [paper-pdf](http://arxiv.org/pdf/2105.08269v2)

**Authors**: Qing Guo, Felix Juefei-Xu, Changqing Zhou, Wei Feng, Yang Liu, Song Wang

**Abstract**: Adversarial training (AT) is one of the most effective ways for improving the robustness of deep convolution neural networks (CNNs). Just like common network training, the effectiveness of AT relies on the design of basic network components. In this paper, we conduct an in-depth study on the role of the basic ReLU activation component in AT for robust CNNs. We find that the spatially-shared and input-independent properties of ReLU activation make CNNs less robust to white-box adversarial attacks with either standard or adversarial training. To address this problem, we extend ReLU to a novel Sparta activation function (Spatially attentive and Adversarially Robust Activation), which enables CNNs to achieve both higher robustness, i.e., lower error rate on adversarial examples, and higher accuracy, i.e., lower error rate on clean examples, than the existing state-of-the-art (SOTA) activation functions. We further study the relationship between Sparta and the SOTA activation functions, providing more insights about the advantages of our method. With comprehensive experiments, we also find that the proposed method exhibits superior cross-CNN and cross-dataset transferability. For the former, the adversarially trained Sparta function for one CNN (e.g., ResNet-18) can be fixed and directly used to train another adversarially robust CNN (e.g., ResNet-34). For the latter, the Sparta function trained on one dataset (e.g., CIFAR-10) can be employed to train adversarially robust CNNs on another dataset (e.g., SVHN). In both cases, Sparta leads to CNNs with higher robustness than the vanilla ReLU, verifying the flexibility and versatility of the proposed method.

摘要: 对抗性训练(AT)是提高深层卷积神经网络(CNN)健壮性的最有效方法之一。就像普通的网络训练一样，AT的有效性依赖于基本网络组件的设计。在本文中，我们对基本REU激活成分在AT中的作用进行了深入的研究，以获得稳健的CNN。我们发现，RELU激活的空间共享和输入无关的性质使得CNN对标准或对抗性训练的白盒对抗性攻击的健壮性较差。为了解决这一问题，我们将RELU扩展到一种新的Sparta激活函数(空间关注和对抗性稳健激活)，该函数使CNN能够获得比现有最先进的(SOTA)激活函数更高的稳健性(即对抗性样本更低的错误率)和更高的准确率(即对干净样本的错误率更低)。我们进一步研究了斯巴达和SOTA激活函数之间的关系，为我们的方法的优势提供了更多的见解。通过大量的实验，我们还发现该方法具有更好的跨CNN和跨数据集可转移性。对于前者，针对一个CNN的对抗性训练的斯巴达函数(例如，ResNet-18)可以被固定并且直接用于训练另一种对抗性稳健的CNN(例如，ResNet-34)。对于后者，可以使用在一个数据集(例如，CIFAR-10)上训练的斯巴达函数来在另一个数据集(例如，SVHN)上训练相反的健壮CNN。在这两种情况下，斯巴达神经网络都比普通的RELU算法具有更高的鲁棒性，验证了该方法的灵活性和通用性。



## **19. Task and Model Agnostic Adversarial Attack on Graph Neural Networks**

基于图神经网络的任务和模型不可知的对抗攻击 cs.LG

To appear as a full paper in AAAI 2023

**SubmitDate**: 2022-12-03    [abs](http://arxiv.org/abs/2112.13267v2) [paper-pdf](http://arxiv.org/pdf/2112.13267v2)

**Authors**: Kartik Sharma, Samidha Verma, Sourav Medya, Sayan Ranu, Arnab Bhattacharya

**Abstract**: Adversarial attacks on Graph Neural Networks (GNNs) reveal their security vulnerabilities, limiting their adoption in safety-critical applications. However, existing attack strategies rely on the knowledge of either the GNN model being used or the predictive task being attacked. Is this knowledge necessary? For example, a graph may be used for multiple downstream tasks unknown to a practical attacker. It is thus important to test the vulnerability of GNNs to adversarial perturbations in a model and task agnostic setting. In this work, we study this problem and show that GNNs remain vulnerable even when the downstream task and model are unknown. The proposed algorithm, TANDIS (Targeted Attack via Neighborhood DIStortion) shows that distortion of node neighborhoods is effective in drastically compromising prediction performance. Although neighborhood distortion is an NP-hard problem, TANDIS designs an effective heuristic through a novel combination of Graph Isomorphism Network with deep Q-learning. Extensive experiments on real datasets and state-of-the-art models show that, on average, TANDIS is up to 50% more effective than state-of-the-art techniques, while being more than 1000 times faster.

摘要: 对图神经网络(GNN)的敌意攻击暴露了它们的安全漏洞，限制了它们在安全关键应用中的采用。然而，现有的攻击策略依赖于正在使用的GNN模型或被攻击的预测任务的知识。这方面的知识有必要吗？例如，图可能用于实际攻击者未知的多个下游任务。因此，重要的是在模型和任务不可知的情况下测试GNN对对抗性扰动的脆弱性。在这项工作中，我们研究了这个问题，并证明了即使在下游任务和模型未知的情况下，GNN仍然是脆弱的。本文提出的基于邻域失真的目标攻击算法TANDIS表明，节点邻域失真对预测性能的影响是有效的。虽然邻域失真是一个NP-Hard问题，但Tandis通过图同构网络和深度Q-学习的新组合设计了一个有效的启发式算法。在真实数据集和最先进的模型上进行的广泛实验表明，平均而言，tandis的效率比最先进的技术高出50%，同时速度快1000倍以上。



## **20. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

ROS-KD：一种适用于噪声医学成像的稳健随机知识提取方法 cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2210.08388v2) [paper-pdf](http://arxiv.org/pdf/2210.08388v2)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.

摘要: 人工智能支持的医学成像最近获得了极大的关注，因为它能够提供快节奏的医疗诊断。然而，由于高昂的注释成本、观察者之间的可变性、人为注释员错误以及计算机生成的标签中的错误，它通常缺乏高质量的数据集。在有噪声标记的数据集上训练的深度学习模型对噪声类型敏感，导致对不可见样本的泛化程度较低。为了应对这一挑战，我们提出了一个稳健的随机知识蒸馏(ROS-KD)框架，它模仿了从多个来源学习一个主题的概念，以确保在学习噪声信息时具有威慑作用。更具体地说，ROS-KD通过从多个教师那里提取知识来学习流畅、消息灵通和健壮的学生流形，这些知识来自于在重叠的训练数据子集上训练的多个教师。我们使用真实世界的数据集对流行的医学图像分类任务(心肺疾病和病变分类)进行了广泛的实验，表明了Ros-KD的性能优势，它能够在相对较小的网络中从许多流行的大型网络(ResNet-50，DenseNet-121，MobileNet-V2)中提取知识，以及它对对手攻击(PGD，FSGM)的鲁棒性。更具体地说，当基础学生是ResNet-18而不是最近的竞争性知识蒸馏基线时，ROS-KD在病变分类和心肺疾病分类任务中分别比F1分数提高>2%和>4%。此外，在心肺疾病分类任务上，ROS-KD在AUC评分上比大多数SOTA基线高出约1%。



## **21. Dual Graphs of Polyhedral Decompositions for the Detection of Adversarial Attacks**

检测对抗性攻击的多面体分解对偶图 cs.CV

978-1-6654-8045-1/22/\$31.00 \copyright{}2022 IEEE The 6th Workshop  on Graph Techniques for Adversarial Activity Analytics (GTA 2022)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2211.13305v2) [paper-pdf](http://arxiv.org/pdf/2211.13305v2)

**Authors**: Huma Jamil, Yajing Liu, Christina M. Cole, Nathaniel Blanchard, Emily J. King, Michael Kirby, Christopher Peterson

**Abstract**: Previous work has shown that a neural network with the rectified linear unit (ReLU) activation function leads to a convex polyhedral decomposition of the input space. These decompositions can be represented by a dual graph with vertices corresponding to polyhedra and edges corresponding to polyhedra sharing a facet, which is a subgraph of a Hamming graph. This paper illustrates how one can utilize the dual graph to detect and analyze adversarial attacks in the context of digital images. When an image passes through a network containing ReLU nodes, the firing or non-firing at a node can be encoded as a bit ($1$ for ReLU activation, $0$ for ReLU non-activation). The sequence of all bit activations identifies the image with a bit vector, which identifies it with a polyhedron in the decomposition and, in turn, identifies it with a vertex in the dual graph. We identify ReLU bits that are discriminators between non-adversarial and adversarial images and examine how well collections of these discriminators can ensemble vote to build an adversarial image detector. Specifically, we examine the similarities and differences of ReLU bit vectors for adversarial images, and their non-adversarial counterparts, using a pre-trained ResNet-50 architecture. While this paper focuses on adversarial digital images, ResNet-50 architecture, and the ReLU activation function, our methods extend to other network architectures, activation functions, and types of datasets.

摘要: 前人的工作表明，具有修正的线性单元(RELU)激活函数的神经网络导致输入空间的凸多面体分解。这些分解可以用一个对偶图来表示，对偶图的顶点对应于多面体，边对应于多面体共享一个面，这是Hamming图的一个子图。本文阐述了如何利用对偶图来检测和分析数字图像环境中的敌意攻击。当图像通过包含REU节点的网络时，在节点处的激发或非激发可以被编码为位($1$用于REU激活，$0$用于REU非激活)。所有位激活的序列用位向量标识图像，位向量在分解中用多面体标识图像，进而用对偶图中的顶点标识图像。我们识别作为非对抗性图像和对抗性图像之间的鉴别器的REU比特，并检查这些鉴别器集合的集合如何能够很好地集成投票来构建对抗性图像检测器。具体地说，我们使用预先训练的ResNet-50体系结构来检查对抗性图像的RELU位向量与非对抗性图像的RELU位向量的异同。虽然本文的重点是对抗性数字图像、ResNet-50体系结构和RELU激活功能，但我们的方法扩展到其他网络体系结构、激活功能和数据集类型。



## **22. Finitely Repeated Adversarial Quantum Hypothesis Testing**

有限重复对抗性量子假设检验 quant-ph

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.02314v1) [paper-pdf](http://arxiv.org/pdf/2212.02314v1)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: We formulate a passive quantum detector based on a quantum hypothesis testing framework under the setting of finite sample size. In particular, we exploit the fundamental limits of performance of the passive quantum detector asymptotically. Under the assumption that the attacker adopts separable optimal strategies, we derive that the worst-case average error bound converges to zero exponentially in terms of the number of repeated observations, which serves as a variation of quantum Sanov's theorem. We illustrate the general decaying results of miss rate numerically, depicting that the `naive' detector manages to achieve a miss rate and a false alarm rate both exponentially decaying to zero given infinitely many quantum states, although the miss rate decays to zero at a much slower rate than a quantum non-adversarial counterpart. Finally we adopt our formulations upon a case study of detection with quantum radars.

摘要: 在有限样本量的情况下，我们建立了一个基于量子假设检验框架的被动量子探测器。特别地，我们渐近地开发了被动量子探测器的基本性能极限。在攻击者采取可分最优策略的假设下，我们推导出最坏情况下的平均误差界关于重复观测次数指数收敛到零，这是量子Sanov定理的一个变形。我们用数值方法说明了失误率的一般衰减结果，描述了在给定无限多个量子态的情况下，尽管失误率衰减到零的速度比量子非对抗性检测器慢得多，但是“朴素”检测器设法实现了失误率和虚警率都以指数形式衰减到零。最后，我们将我们的公式应用于量子雷达探测的一个案例研究。



## **23. Diverse Generative Perturbations on Attention Space for Transferable Adversarial Attacks**

可转移对抗性攻击注意空间上的不同生成扰动 cs.CV

ICIP 2022 (Oral)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2208.05650v2) [paper-pdf](http://arxiv.org/pdf/2208.05650v2)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstract**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.

摘要: 具有改进的可转移性的对抗性攻击--在已知模型上制作的对抗性例子也能够愚弄未知模型的能力--由于其实用性最近受到了极大的关注。然而，现有的可转移攻击以确定性的方式制造扰动，往往不能充分探索损失曲面，从而陷入较差的局部最优，且可转移性较低。为了解决这一问题，我们提出了注意力多样性攻击(ADA)，它以随机的方式破坏不同的显著特征，以提高可转移性。首先，我们扰乱图像注意力，以扰乱不同模型共享的通用特征。然后，为了有效地避免局部最优，我们以随机的方式破坏了这些特征，并更详尽地探索了可转移扰动的搜索空间。更具体地说，我们使用生成器来产生对抗性扰动，每个扰动都以不同的方式干扰特征，具体取决于输入的潜在代码。广泛的实验评估表明，我们的方法是有效的，超过了最先进的方法的可转移性。有关代码，请访问https://github.com/wkim97/ADA.



## **24. Membership Inference Attacks Against Semantic Segmentation Models**

针对语义分割模型的隶属度推理攻击 cs.CR

Submitted as conference paper to PETS 2023

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.01082v1) [paper-pdf](http://arxiv.org/pdf/2212.01082v1)

**Authors**: Tomas Chobola, Dmitrii Usynin, Georgios Kaissis

**Abstract**: Membership inference attacks aim to infer whether a data record has been used to train a target model by observing its predictions. In sensitive domains such as healthcare, this can constitute a severe privacy violation. In this work we attempt to address the existing knowledge gap by conducting an exhaustive study of membership inference attacks and defences in the domain of semantic image segmentation. Our findings indicate that for certain threat models, these learning settings can be considerably more vulnerable than the previously considered classification settings. We additionally investigate a threat model where a dishonest adversary can perform model poisoning to aid their inference and evaluate the effects that these adaptations have on the success of membership inference attacks. We quantitatively evaluate the attacks on a number of popular model architectures across a variety of semantic segmentation tasks, demonstrating that membership inference attacks in this domain can achieve a high success rate and defending against them may result in unfavourable privacy-utility trade-offs or increased computational costs.

摘要: 成员关系推理攻击的目的是通过观察数据记录的预测来推断数据记录是否已被用于训练目标模型。在医疗等敏感领域，这可能构成严重侵犯隐私的行为。在这项工作中，我们试图通过对语义图像分割领域中的隶属度推理攻击和防御进行详尽的研究来解决现有的知识鸿沟。我们的发现表明，对于某些威胁模型，这些学习设置可能比之前考虑的分类设置更容易受到攻击。此外，我们还研究了一个威胁模型，在该模型中，不诚实的对手可以执行模型中毒来帮助他们进行推理，并评估这些适应对成员关系推理攻击成功的影响。我们通过各种语义分割任务对一些流行的模型体系结构的攻击进行了定量评估，证明了在该领域中的成员关系推理攻击可以获得高的成功率，而防御它们可能会导致不利的隐私效用权衡或增加计算成本。



## **25. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

Accepted in AAAI 2023

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2203.04713v6) [paper-pdf](http://arxiv.org/pdf/2203.04713v6)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstract**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks. Code is available at https://github.com/realcrane/RobustActionRecogniser.

摘要: 骨骼运动在人类活动识别(HAR)中得到了广泛的应用。最近，基于骨架的HAR在各种分类器和数据中发现了一个普遍的漏洞，需要缓解。为此，我们提出了第一种基于骨架的HAR黑盒防御方法。我们的方法的特点是对干净数据、对手和分类器进行全面的贝叶斯处理，导致(1)新的基于贝叶斯能量的稳健判别分类器的形成，(2)基于自然运动流形的新的对手采样方案，(3)新的训练后贝叶斯策略用于黑盒防御。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的骨架HAR分类器和数据集上展示了令人惊讶的和普遍的有效性。代码可在https://github.com/realcrane/RobustActionRecogniser.上找到



## **26. AccEar: Accelerometer Acoustic Eavesdropping with Unconstrained Vocabulary**

AccEar：无限制词汇的加速度计声学窃听 cs.SD

2022 IEEE Symposium on Security and Privacy (SP)

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2212.01042v1) [paper-pdf](http://arxiv.org/pdf/2212.01042v1)

**Authors**: Pengfei Hu, Hui Zhuang, Panneer Selvam Santhalingamy, Riccardo Spolaor, Parth Pathaky, Guoming Zhang, Xiuzhen Cheng

**Abstract**: With the increasing popularity of voice-based applications, acoustic eavesdropping has become a serious threat to users' privacy. While on smartphones the access to microphones needs an explicit user permission, acoustic eavesdropping attacks can rely on motion sensors (such as accelerometer and gyroscope), which access is unrestricted. However, previous instances of such attacks can only recognize a limited set of pre-trained words or phrases. In this paper, we present AccEar, an accelerometerbased acoustic eavesdropping attack that can reconstruct any audio played on the smartphone's loudspeaker with unconstrained vocabulary. We show that an attacker can employ a conditional Generative Adversarial Network (cGAN) to reconstruct highfidelity audio from low-frequency accelerometer signals. The presented cGAN model learns to recreate high-frequency components of the user's voice from low-frequency accelerometer signals through spectrogram enhancement. We assess the feasibility and effectiveness of AccEar attack in a thorough set of experiments using audio from 16 public personalities. As shown by the results in both objective and subjective evaluations, AccEar successfully reconstructs user speeches from accelerometer signals in different scenarios including varying sampling rate, audio volume, device model, etc.

摘要: 随着基于语音的应用日益普及，声学窃听已经成为对用户隐私的严重威胁。虽然在智能手机上访问麦克风需要明确的用户许可，但声学窃听攻击可以依赖于不受限制的运动传感器(如加速计和陀螺仪)。然而，以前的此类攻击实例只能识别一组有限的预先训练的单词或短语。在本文中，我们提出了AccEar，这是一种基于加速计的声学窃听攻击，可以用不受限制的词汇重建智能手机扬声器上播放的任何音频。我们发现攻击者可以利用条件生成对抗网络(CGAN)从低频加速度计信号中重建高保真音频。提出的cGAN模型通过谱图增强学习从低频加速度计信号中重建用户语音的高频分量。我们使用16位公众人物的音频，通过一组全面的实验评估了AccEar攻击的可行性和有效性。客观评估和主观评估的结果表明，AccEar能够在不同的场景下，包括不同的采样率、音量、设备型号等，成功地从加速度计信号中恢复用户语音。



## **27. SecureSense: Defending Adversarial Attack for Secure Device-Free Human Activity Recognition**

SecureSense：防御恶意攻击，实现安全的无设备人类活动识别 cs.CR

The paper is accepted by IEEE Transactions on Mobile Computing

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2204.01560v2) [paper-pdf](http://arxiv.org/pdf/2204.01560v2)

**Authors**: Jianfei Yang, Han Zou, Lihua Xie

**Abstract**: Deep neural networks have empowered accurate device-free human activity recognition, which has wide applications. Deep models can extract robust features from various sensors and generalize well even in challenging situations such as data-insufficient cases. However, these systems could be vulnerable to input perturbations, i.e. adversarial attacks. We empirically demonstrate that both black-box Gaussian attacks and modern adversarial white-box attacks can render their accuracies to plummet. In this paper, we firstly point out that such phenomenon can bring severe safety hazards to device-free sensing systems, and then propose a novel learning framework, SecureSense, to defend common attacks. SecureSense aims to achieve consistent predictions regardless of whether there exists an attack on its input or not, alleviating the negative effect of distribution perturbation caused by adversarial attacks. Extensive experiments demonstrate that our proposed method can significantly enhance the model robustness of existing deep models, overcoming possible attacks. The results validate that our method works well on wireless human activity recognition and person identification systems. To the best of our knowledge, this is the first work to investigate adversarial attacks and further develop a novel defense framework for wireless human activity recognition in mobile computing research.

摘要: 深度神经网络能够实现准确的无设备人体活动识别，具有广泛的应用前景。深度模型可以从各种传感器中提取稳健的特征，即使在数据不足等具有挑战性的情况下也能很好地泛化。然而，这些系统可能容易受到输入扰动，即对抗性攻击。我们的经验证明，无论是黑盒高斯攻击还是现代对抗性白盒攻击，它们的准确率都会直线下降。在本文中，我们首先指出这种现象会给无设备感知系统带来严重的安全隐患，然后提出一种新的学习框架SecureSense来防御常见的攻击。SecureSense的目标是无论其输入是否存在攻击，都能实现一致的预测，缓解对抗性攻击造成的分发扰动的负面影响。大量实验表明，该方法能够显著增强已有深度模型的模型稳健性，克服可能的攻击。实验结果表明，该方法在无线人体活动识别和身份识别系统中具有较好的效果。据我们所知，这是第一次在移动计算研究中研究对抗性攻击，并进一步开发出一种新的无线人类活动识别防御框架。



## **28. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

RamBoAttack：一种稳健查询高效的深度神经网络决策开发 cs.LG

Published in Network and Distributed System Security (NDSS) Symposium  2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2112.05282v2) [paper-pdf](http://arxiv.org/pdf/2112.05282v2)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.

摘要: 机器学习模型极易受到来自对手例子的逃避攻击。通常，对抗性的例子，修改后的输入欺骗性地类似于原始输入，由具有完全访问模型的敌手在白盒设置下构造。然而，最近的攻击显示，使用黑盒攻击构建敌意例子的查询数量显著减少。特别是，警报是利用由包括谷歌、微软、IBM在内的越来越多的机器学习即服务提供商提供的训练模型的访问接口的分类决策的能力，并被结合这些模型的大量应用程序使用。对手仅利用模型中预测的标签来制作敌意示例的能力被区分为基于决策的攻击。在我们的研究中，我们首先深入研究了ICLR和SP中最新的基于决策的攻击，以强调使用梯度估计方法发现低失真攻击的代价。我们开发了一种健壮的查询高效攻击，能够避免陷入局部最小值和从梯度估计方法中看到的噪声梯度的误导。我们提出的攻击方法RamBoAttack利用随机化块坐标下降的概念来探索隐藏的分类器流形，针对扰动只操纵局部输入特征来解决梯度估计方法的问题。重要的是，RamBoAttack对于对手和目标类可用的不同样本输入更加健壮。总体而言，对于给定的目标类，RamBoAttack被证明在给定的查询预算内实现较低的失真方面更加健壮。我们使用大规模高分辨率ImageNet数据集和在GitHub上开源的我们的攻击、测试样本和人工制品来管理我们广泛的结果。



## **29. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

Garnet：强健可扩展图神经网络的降阶拓扑学习 cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2022-12-02    [abs](http://arxiv.org/abs/2201.12741v4) [paper-pdf](http://arxiv.org/pdf/2201.12741v4)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.

摘要: 图形神经网络(GNN)已被越来越多地应用于涉及非欧几里德数据学习的各种应用中。然而，最近的研究表明，GNN容易受到图的对抗性攻击。虽然有几种防御方法可以通过消除敌对组件来提高GNN的健壮性，但它们也可能损害有助于GNN训练的底层干净的图形结构。此外，这些防御模型中很少有能够扩展到大型图形的，因为它们的计算复杂性和内存使用量很高。在本文中，我们提出了Garnet，一种可伸缩的谱方法来提高GNN模型的对抗健壮性。Garnet First利用加权谱嵌入来构造基图，该基图不仅能抵抗敌方攻击，而且还包含GNN训练所需的关键(干净)图结构。接下来，Garnet基于概率图模型，通过剪枝额外的非关键边来进一步精化基图。石榴石已经在各种数据集上进行了评估，包括一个包含数百万个节点的大型图表。我们的大量实验结果表明，与现有的GNN(防御)模型相比，Garnet的对抗准确率提高了13.27%，运行时加速比提高了14.7倍。



## **30. Pareto Regret Analyses in Multi-objective Multi-armed Bandit**

多目标多臂匪徒的帕累托悔恨分析 cs.LG

18 pages

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00884v1) [paper-pdf](http://arxiv.org/pdf/2212.00884v1)

**Authors**: Mengfan Xu, Diego Klabjan

**Abstract**: We study Pareto optimality in multi-objective multi-armed bandit by providing a formulation of adversarial multi-objective multi-armed bandit and properly defining its Pareto regrets that can be generalized to stochastic settings as well. The regrets do not rely on any scalarization functions and reflect Pareto optimality compared to scalarized regrets. We also present new algorithms assuming both with and without prior information of the multi-objective multi-armed bandit setting. The algorithms are shown optimal in adversarial settings and nearly optimal in stochastic settings simultaneously by our established upper bounds and lower bounds on Pareto regrets. Moreover, the lower bound analyses show that the new regrets are consistent with the existing Pareto regret for stochastic settings and extend an adversarial attack mechanism from bandit to the multi-objective one.

摘要: 通过给出对抗性多目标多臂匪徒的一个公式，并适当地定义了它的Pareto遗憾，从而研究了多目标多臂匪徒的Pareto最优性，并将其推广到随机环境中。遗憾不依赖于任何标量化函数，并且与标量化的遗憾相比，反映了帕累托最优。我们还提出了新的算法，假设有和没有多目标多臂匪徒设置的先验信息。通过我们建立的Pareto后悔的上界和下界，算法在对抗性环境中被证明是最优的，在随机环境中几乎是最优的。此外，下界分析表明，新的遗憾与现有的随机设置下的帕累托遗憾是一致的，并将对抗性攻击机制从强盗扩展到多目标机制。



## **31. Adversarial Analysis of the Differentially-Private Federated Learning in Cyber-Physical Critical Infrastructures**

网络物理关键基础设施中非私有联邦学习的对抗性分析 cs.CR

16 pages, 9 figures, 5 tables. This work has been submitted to IEEE  for possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2204.02654v2) [paper-pdf](http://arxiv.org/pdf/2204.02654v2)

**Authors**: Md Tamjid Hossain, Shahriar Badsha, Hung La, Haoting Shen, Shafkat Islam, Ibrahim Khalil, Xun Yi

**Abstract**: Federated Learning (FL) has become increasingly popular to perform data-driven analysis in cyber-physical critical infrastructures. Since the FL process may involve the client's confidential information, Differential Privacy (DP) has been proposed lately to secure it from adversarial inference. However, we find that while DP greatly alleviates the privacy concerns, the additional DP-noise opens a new threat for model poisoning in FL. Nonetheless, very little effort has been made in the literature to investigate this adversarial exploitation of the DP-noise. To overcome this gap, in this paper, we present a novel adaptive model poisoning technique {\alpha}-MPELM} through which an attacker can exploit the additional DP-noise to evade the state-of-the-art anomaly detection techniques and prevent optimal convergence of the FL model. We evaluate our proposed attack on the state-of-the-art anomaly detection approaches in terms of detection accuracy and validation loss. The main significance of our proposed {\alpha}-MPELM attack is that it reduces the state-of-the-art anomaly detection accuracy by 6.8% for norm detection, 12.6% for accuracy detection, and 13.8% for mix detection. Furthermore, we propose a Reinforcement Learning-based DP level selection process to defend {\alpha}-MPELM attack. The experimental results confirm that our defense mechanism converges to an optimal privacy policy without human maneuver.

摘要: 联合学习(FL)在网络物理关键基础设施中执行数据驱动分析已变得越来越流行。由于FL过程可能涉及客户的机密信息，最近有人提出了差分隐私(DP)来保护它免受对手的推断。然而，我们发现，虽然DP极大地缓解了隐私问题，但额外的DP噪声为FL中的模型中毒打开了新的威胁。尽管如此，文献中很少有人研究这种对抗性地利用DP-噪声。为了克服这一缺陷，本文提出了一种新的自适应模型中毒技术{\α}-MPELM}，通过该技术，攻击者可以利用额外的DP-噪声来避开最新的异常检测技术，防止FL模型的最优收敛。我们从检测准确率和验证损失两个方面对我们提出的对最新异常检测方法的攻击进行了评估。我们提出的{\α}-MPELM攻击的主要意义在于，它使标准检测的异常检测准确率降低了6.8%，准确率检测降低了12.6%，混合检测的异常检测准确率降低了13.8%。此外，我们还提出了一种基于强化学习的DP级别选择过程来防御{\Alpha}-MPELM攻击。实验结果表明，该防御机制在不需要人工干预的情况下收敛到最优隐私策略。



## **32. Purifier: Defending Data Inference Attacks via Transforming Confidence Scores**

净化器：通过变换置信度分数防御数据推理攻击 cs.LG

accepted by AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00612v1) [paper-pdf](http://arxiv.org/pdf/2212.00612v1)

**Authors**: Ziqi Yang, Lijin Wang, Da Yang, Jie Wan, Ziming Zhao, Ee-Chien Chang, Fan Zhang, Kui Ren

**Abstract**: Neural networks are susceptible to data inference attacks such as the membership inference attack, the adversarial model inversion attack and the attribute inference attack, where the attacker could infer useful information such as the membership, the reconstruction or the sensitive attributes of a data sample from the confidence scores predicted by the target classifier. In this paper, we propose a method, namely PURIFIER, to defend against membership inference attacks. It transforms the confidence score vectors predicted by the target classifier and makes purified confidence scores indistinguishable in individual shape, statistical distribution and prediction label between members and non-members. The experimental results show that PURIFIER helps defend membership inference attacks with high effectiveness and efficiency, outperforming previous defense methods, and also incurs negligible utility loss. Besides, our further experiments show that PURIFIER is also effective in defending adversarial model inversion attacks and attribute inference attacks. For example, the inversion error is raised about 4+ times on the Facescrub530 classifier, and the attribute inference accuracy drops significantly when PURIFIER is deployed in our experiment.

摘要: 神经网络容易受到数据推理攻击，如隶属度推理攻击、对抗性模型反转攻击和属性推理攻击，攻击者可以从目标分类器预测的置信度分数推断出数据样本的隶属度、重构或敏感属性等有用信息。在本文中，我们提出了一种防御成员关系推理攻击的方法，即净化器。它对目标分类器预测的置信度向量进行变换，使得纯化后的置信度在个体形状、统计分布和成员与非成员之间的预测标签上无法区分。实验结果表明，该算法能够有效、高效地防御成员关系推理攻击，性能优于以往的防御方法，而效用损失可以忽略不计。此外，我们进一步的实验表明，净化器在防御对抗性模型反转攻击和属性推理攻击方面也是有效的。例如，在FacescRub530分类器上，倒置错误提高了大约4倍以上，当我们的实验中部署了净化器时，属性推理的准确率显著下降。



## **33. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

PointCA：评估3D点云补全模型对敌方示例的稳健性 cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2211.12294v2) [paper-pdf](http://arxiv.org/pdf/2211.12294v2)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.

摘要: 点云补全作为三维识别和分割的上游步骤，已经成为导航、场景理解等许多任务的重要组成部分。虽然各种点云补全模型已经展示了它们强大的能力，但它们对对手攻击的健壮性仍然未知，这些攻击已被证明是对深度神经网络的致命恶意攻击。此外，由于不同的输出形式和攻击目的，现有的针对点云分类器的攻击方法不能应用于完成模型。为了评估补全模型的健壮性，我们提出了针对三维点云补全模型的第一次对抗性攻击PointCA。PointCA可以生成与原始点云保持高度相似的对抗性点云，同时作为另一个对象完成，具有完全不同的语义信息。具体地说，我们最小化对抗性样本和目标点集之间的表示差异，共同探索几何空间和特征空间中的对抗性点云。此外，为了发动更隐蔽的攻击，我们创新性地使用邻域密度信息来定制扰动约束，导致对每个点的几何感知和分布自适应修改。在不同初始点云完成网络上的大量实验表明，在结构倒角距离保持在0.01以下的情况下，PointCA可以导致性能从77.9%下降到16.7%。我们得出的结论是，现有的完备化模型非常容易受到对手例子的攻击，并且最新的点云分类方法在应用于不完整和不均匀的点云数据时将部分无效。



## **34. All You Need Is Hashing: Defending Against Data Reconstruction Attack in Vertical Federated Learning**

您所需要的就是散列：在垂直联合学习中防御数据重建攻击 cs.CR

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00325v1) [paper-pdf](http://arxiv.org/pdf/2212.00325v1)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Yuwen Pu, Ting Wang

**Abstract**: Vertical federated learning is a trending solution for multi-party collaboration in training machine learning models. Industrial frameworks adopt secure multi-party computation methods such as homomorphic encryption to guarantee data security and privacy. However, a line of work has revealed that there are still leakage risks in VFL. The leakage is caused by the correlation between the intermediate representations and the raw data. Due to the powerful approximation ability of deep neural networks, an adversary can capture the correlation precisely and reconstruct the data. To deal with the threat of the data reconstruction attack, we propose a hashing-based VFL framework, called \textit{HashVFL}, to cut off the reversibility directly. The one-way nature of hashing allows our framework to block all attempts to recover data from hash codes. However, integrating hashing also brings some challenges, e.g., the loss of information. This paper proposes and addresses three challenges to integrating hashing: learnability, bit balance, and consistency. Experimental results demonstrate \textit{HashVFL}'s efficiency in keeping the main task's performance and defending against data reconstruction attacks. Furthermore, we also analyze its potential value in detecting abnormal inputs. In addition, we conduct extensive experiments to prove \textit{HashVFL}'s generalization in various settings. In summary, \textit{HashVFL} provides a new perspective on protecting multi-party's data security and privacy in VFL. We hope our study can attract more researchers to expand the application domains of \textit{HashVFL}.

摘要: 垂直联合学习是训练机器学习模型中多方协作的一种趋势解决方案。产业框架采用同态加密等安全多方计算方法，保障数据安全和隐私。然而，有一项工作透露，VFL仍存在泄漏风险。泄漏是由中间表示法和原始数据之间的相关性造成的。由于深度神经网络的强大逼近能力，对手可以准确地捕获相关性并重建数据。为了应对数据重构攻击的威胁，我们提出了一种基于哈希的VFL框架，称为\textit{HashVFL}，直接切断可逆性。哈希的单向特性允许我们的框架阻止所有从哈希码恢复数据的尝试。然而，整合散列也带来了一些挑战，例如信息的丢失。本文提出并解决了整合散列的三个挑战：可学习性、位平衡和一致性。实验结果证明了该算法在保持主任务性能和抵御数据重构攻击方面的有效性，并分析了其在异常输入检测方面的潜在价值。此外，我们还通过大量的实验证明了该算法在不同环境下的泛化能力。综上所述，该算法为保护VFL中多方的数据安全和隐私提供了一个新的视角。我们希望我们的研究能够吸引更多的研究人员来拓展该算法的应用领域。



## **35. Overcoming the Convex Relaxation Barrier for Neural Network Verification via Nonconvex Low-Rank Semidefinite Relaxations**

用非凸低阶半定松弛克服神经网络验证的凸松弛障碍 cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17244v1) [paper-pdf](http://arxiv.org/pdf/2211.17244v1)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: To rigorously certify the robustness of neural networks to adversarial perturbations, most state-of-the-art techniques rely on a triangle-shaped linear programming (LP) relaxation of the ReLU activation. While the LP relaxation is exact for a single neuron, recent results suggest that it faces an inherent "convex relaxation barrier" as additional activations are added, and as the attack budget is increased. In this paper, we propose a nonconvex relaxation for the ReLU relaxation, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. We show that the nonconvex relaxation has a similar complexity to the LP relaxation, but enjoys improved tightness that is comparable to the much more expensive SDP relaxation. Despite nonconvexity, we prove that the verification problem satisfies constraint qualification, and therefore a Riemannian staircase approach is guaranteed to compute a near-globally optimal solution in polynomial time. Our experiments provide evidence that our nonconvex relaxation almost completely overcome the "convex relaxation barrier" faced by the LP relaxation.

摘要: 为了严格证明神经网络对对抗性扰动的稳健性，大多数最先进的技术依赖于REU激活的三角形线性规划(LP)松弛。虽然LP松弛对于单个神经元来说是精确的，但最近的结果表明，随着额外的激活增加和攻击预算的增加，它面临着固有的“凸松弛障碍”。本文基于半定规划(SDP)松弛的低阶限制，提出了RELU松弛的非凸松弛。我们证明了非凸松弛具有与LP松弛相似的复杂性，但具有与更昂贵的SDP松弛相当的紧性。尽管非凸性，我们证明了验证问题满足约束限定，从而保证了黎曼阶梯方法在多项式时间内计算出近全局最优解。我们的实验证明，我们的非凸松弛几乎完全克服了LP松弛所面临的“凸松弛障碍”。



## **36. Differentially Private ADMM-Based Distributed Discrete Optimal Transport for Resource Allocation**

基于差分私有ADMM的分布式离散资源优化传输 cs.SI

6 pages, 4 images, 1 algorithm, IEEE GLOBECOMM 2022

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17070v1) [paper-pdf](http://arxiv.org/pdf/2211.17070v1)

**Authors**: Jason Hughes, Juntao Chen

**Abstract**: Optimal transport (OT) is a framework that can guide the design of efficient resource allocation strategies in a network of multiple sources and targets. To ease the computational complexity of large-scale transport design, we first develop a distributed algorithm based on the alternating direction method of multipliers (ADMM). However, such a distributed algorithm is vulnerable to sensitive information leakage when an attacker intercepts the transport decisions communicated between nodes during the distributed ADMM updates. To this end, we propose a privacy-preserving distributed mechanism based on output variable perturbation by adding appropriate randomness to each node's decision before it is shared with other corresponding nodes at each update instance. We show that the developed scheme is differentially private, which prevents the adversary from inferring the node's confidential information even knowing the transport decisions. Finally, we corroborate the effectiveness of the devised algorithm through case studies.

摘要: 最优传输(OT)是一个框架，可以指导在多个源和目标的网络中设计有效的资源分配策略。为了降低大规模运输设计的计算复杂性，我们首先提出了一种基于交替方向乘子法的分布式算法。然而，当攻击者在分布式ADMM更新期间截获节点之间通信的传输决策时，这样的分布式算法容易受到敏感信息的泄漏。为此，我们提出了一种基于输出变量扰动的隐私保护分布式机制，在每个更新时刻与其他对应节点共享之前，为每个节点的决策添加适当的随机性。我们证明了所提出的方案是差分私密性的，即使知道传输决策，也可以防止对手推断节点的机密信息。最后，通过案例分析验证了所设计算法的有效性。



## **37. A Systematic Evaluation of Node Embedding Robustness**

节点嵌入健壮性的系统评估 cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2209.08064v3) [paper-pdf](http://arxiv.org/pdf/2209.08064v3)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstract**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has grown significantly in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring attacks computed using network properties as well as node labels. We also investigate the performance of popular node classification attack baselines that assume full knowledge of the node labels. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We find that node classification results are impacted more than network reconstruction ones, that degree-based and label-based attacks are on average the most damaging and that label heterophily can strongly influence attack performance.

摘要: 节点嵌入方法将网络节点映射到可随后用于各种下行预测任务的低维向量。近年来，这些方法的普及率显著提高，然而，人们对它们对输入数据扰动的稳健性仍然知之甚少。在本文中，我们评估了节点嵌入模型对随机和对抗性中毒攻击的经验稳健性。我们的系统评价涵盖了基于Skip-Gram的典型嵌入方法、矩阵分解和深度神经网络。我们比较了使用网络属性和节点标签计算的边添加、删除和重新布线攻击。我们还研究了假设完全知道节点标签的流行的节点分类攻击基线的性能。我们通过嵌入可视化和定量结果来报告下游节点分类和网络重构性能方面的定性结果。我们发现，节点分类结果受到的影响比网络重构的影响更大，基于度和基于标签的攻击平均破坏性最大，标签异质性对攻击性能有很强的影响。



## **38. Adaptive adversarial training method for improving multi-scale GAN based on generalization bound theory**

基于泛化界理论的改进多尺度GAN的自适应对抗性训练方法 cs.CV

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.16791v1) [paper-pdf](http://arxiv.org/pdf/2211.16791v1)

**Authors**: Jing Tang, Bo Tao, Zeyu Gong, Zhouping Yin

**Abstract**: In recent years, multi-scale generative adversarial networks (GANs) have been proposed to build generalized image processing models based on single sample. Constraining on the sample size, multi-scale GANs have much difficulty converging to the global optimum, which ultimately leads to limitations in their capabilities. In this paper, we pioneered the introduction of PAC-Bayes generalized bound theory into the training analysis of specific models under different adversarial training methods, which can obtain a non-vacuous upper bound on the generalization error for the specified multi-scale GAN structure. Based on the drastic changes we found of the generalization error bound under different adversarial attacks and different training states, we proposed an adaptive training method which can greatly improve the image manipulation ability of multi-scale GANs. The final experimental results show that our adaptive training method in this paper has greatly contributed to the improvement of the quality of the images generated by multi-scale GANs on several image manipulation tasks. In particular, for the image super-resolution restoration task, the multi-scale GAN model trained by the proposed method achieves a 100% reduction in natural image quality evaluator (NIQE) and a 60% reduction in root mean squared error (RMSE), which is better than many models trained on large-scale datasets.

摘要: 近年来，多尺度生成对抗网络(GANS)被提出用来建立基于单样本的广义图像处理模型。多尺度遗传算法受样本量的限制，很难收敛到全局最优解，这最终导致它们的能力受到限制。在本文中，我们首次将PAC-Bayes广义界理论引入到不同对抗性训练方法下特定模型的训练分析中，可以得到特定多尺度GaN结构的泛化误差的非空上界。基于不同敌方攻击和不同训练状态下泛化误差界的剧烈变化，提出了一种自适应训练方法，可以极大地提高多尺度遗传算法的图像处理能力。最后的实验结果表明，本文提出的自适应训练方法对多尺度GANS算法在多个图像处理任务中生成的图像质量有很大的改善作用。特别是，对于图像超分辨率恢复任务，由该方法训练的多尺度GAN模型在自然图像质量评价(NIQE)和均方根误差(RMSE)方面取得了100%的降低，优于在大规模数据集上训练的许多模型。



## **39. Sludge for Good: Slowing and Imposing Costs on Cyber Attackers**

泥浆向善：减缓网络攻击者的速度并增加其成本 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16626v1) [paper-pdf](http://arxiv.org/pdf/2211.16626v1)

**Authors**: Josiah Dykstra, Kelly Shortridge, Jamie Met, Douglas Hough

**Abstract**: Choice architecture describes the design by which choices are presented to people. Nudges are an aspect intended to make "good" outcomes easy, such as using password meters to encourage strong passwords. Sludge, on the contrary, is friction that raises the transaction cost and is often seen as a negative to users. Turning this concept around, we propose applying sludge for positive cybersecurity outcomes by using it offensively to consume attackers' time and other resources.   To date, most cyber defenses have been designed to be optimally strong and effective and prohibit or eliminate attackers as quickly as possible. Our complimentary approach is to also deploy defenses that seek to maximize the consumption of the attackers' time and other resources while causing as little damage as possible to the victim. This is consistent with zero trust and similar mindsets which assume breach. The Sludge Strategy introduces cost-imposing cyber defense by strategically deploying friction for attackers before, during, and after an attack using deception and authentic design features. We present the characteristics of effective sludge, and show a continuum from light to heavy sludge. We describe the quantitative and qualitative costs to attackers and offer practical considerations for deploying sludge in practice. Finally, we examine real-world examples of U.S. government operations to frustrate and impose cost on cyber adversaries.

摘要: 选择体系结构描述了将选择呈现给人们的设计。轻推是一个旨在使“好”结果变得容易的方面，例如使用密码计量器来鼓励强密码。相反，淤泥是增加交易成本的摩擦，通常被视为对用户不利。扭转这一概念，我们建议将污泥应用于积极的网络安全结果，通过攻击性地使用它来消耗攻击者的时间和其他资源。到目前为止，大多数网络防御都被设计成最强大和有效的，并尽可能快地禁止或消除攻击者。我们的互补方法是部署防御措施，寻求最大限度地消耗攻击者的时间和其他资源，同时尽可能减少对受害者的损害。这与零信任和假设违反的类似心态是一致的。通过使用欺骗和可信的设计功能，在攻击之前、期间和之后战略性地为攻击者部署摩擦，Send Strategy引入了成本高昂的网络防御。我们介绍了有效污泥的特性，并显示了从轻到重的连续统。我们描述了攻击者的定量和定性成本，并为在实践中部署污泥提供了实际考虑。最后，我们考察了美国政府挫败网络对手并将代价强加于他们的真实世界的例子。



## **40. Synthesizing Attack-Aware Control and Active Sensing Strategies under Reactive Sensor Attacks**

反应式传感器攻击下的攻击感知控制与主动感知综合策略 math.OC

7 pages, 3 figure, 1 table, 1 algorithm

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2204.01584v2) [paper-pdf](http://arxiv.org/pdf/2204.01584v2)

**Authors**: Sumukha Udupa, Abhishek N. Kulkarni, Shuo Han, Nandi O. Leslie, Charles A. Kamhoua, Jie Fu

**Abstract**: We consider the probabilistic planning problem for a defender (P1) who can jointly query the sensors and take control actions to reach a set of goal states while being aware of possible sensor attacks by an adversary (P2) who has perfect observations. To synthesize a provably-correct, attack-aware joint control and active sensing strategy for P1, we construct a stochastic game on graph with augmented states that include the actual game state (known only to the attacker), the belief of the defender about the game state (constructed by the attacker based on his knowledge of defender's observations). We present an algorithm to compute a belief-based, randomized strategy for P1 to ensure satisfying the reachability objective with probability one, under the worst-case sensor attack carried out by an informed P2. We prove the correctness of the algorithm and illustrate using an example.

摘要: 我们考虑了防御者(P1)的概率规划问题，该防御者可以联合查询传感器并采取控制行动以达到一组目标状态，同时知道具有完美观察的对手(P2)可能的传感器攻击。为了综合P1的可证明正确的、攻击感知的联合控制和主动感知策略，我们在图上构造了一个带有增广状态的随机游戏，其中包括实际的游戏状态(只有攻击者知道)、防御者对游戏状态的信念(由攻击者根据他对防御者观察的知识构建的)。我们提出了一种算法来计算基于信任的随机策略，以确保在由知情的P2执行的最坏情况的传感器攻击下，以概率1满足可达性目标。证明了该算法的正确性，并用实例进行了说明。



## **41. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

自定步长硬类对重加权提高对手健壮性 cs.CV

AAAI-23

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2210.15068v2) [paper-pdf](http://arxiv.org/pdf/2210.15068v2)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most effective methods. Theoretically, adversarial perturbation in untargeted attacks can be added along arbitrary directions and the predicted labels of untargeted attacks should be unpredictable. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs become virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair losses in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boosts model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.

摘要: 深度神经网络很容易受到敌意攻击。在众多的防守策略中，非靶向攻击的对抗性训练是最有效的方法之一。从理论上讲，非目标攻击中的对抗性扰动可以沿任意方向添加，并且非目标攻击的预测标签应该是不可预测的。然而，我们发现，自然不平衡的类间语义相似度使得这些硬类对成为彼此的虚拟目标。本研究调查了这种紧密耦合的课程对对抗性攻击的影响，并相应地在对抗性训练中开发了一种自定步调重权重策略。具体地说，我们建议在模型优化中增加硬类对损失的权重，从而促进从硬类中学习区分特征。在对抗性训练中，我们进一步引入了一个术语来量化硬类对一致性，这大大提高了模型的稳健性。大量的实验表明，所提出的对抗性训练方法在对抗大范围的对抗性攻击时获得了比最先进的防御方法更好的健壮性性能。



## **42. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

基于弹性风险的自适应身份验证和授权(RAD-AA)框架 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2208.02592v3) [paper-pdf](http://arxiv.org/pdf/2208.02592v3)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstract**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.

摘要: 在最近的网络攻击中，凭据盗窃已成为进入系统的主要载体之一。一旦攻击者在系统中站稳脚跟，他们就会使用包括令牌操作在内的各种技术来提升权限并访问受保护的资源。这使得身份验证和基于令牌的授权成为安全和有弹性的网络系统的关键组件。在本文中，我们讨论了这样一个安全的、具有弹性的认证和授权框架的设计考虑，该框架能够基于风险分数和信任配置文件自适应。我们将该设计与OAuth 2.0、OpenID Connect和SAML 2.0等现有标准进行了比较。然后，我们研究了流行的威胁模型，如STRIDE和PASA，并总结了所提出的体系结构对常见和相关威胁向量的恢复能力。我们将此框架称为基于弹性风险的自适应身份验证和授权(RAD-AA)。拟议的框架过度增加了对手发动和维持任何网络攻击的成本，并为关键基础设施提供了亟需的力量。我们还讨论了机器学习(ML)方法，使自适应引擎能够准确地对交易进行分类，并得出风险分数。



## **43. Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion**

Ada3Diff：通过自适应扩散防御3D对抗性点云 cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16247v1) [paper-pdf](http://arxiv.org/pdf/2211.16247v1)

**Authors**: Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, Nenghai Yu

**Abstract**: Deep 3D point cloud models are sensitive to adversarial attacks, which poses threats to safety-critical applications such as autonomous driving. Robust training and defend-by-denoise are typical strategies for defending adversarial perturbations, including adversarial training and statistical filtering, respectively. However, they either induce massive computational overhead or rely heavily upon specified noise priors, limiting generalized robustness against attacks of all kinds. This paper introduces a new defense mechanism based on denoising diffusion models that can adaptively remove diverse noises with a tailored intensity estimator. Specifically, we first estimate adversarial distortions by calculating the distance of the points to their neighborhood best-fit plane. Depending on the distortion degree, we choose specific diffusion time steps for the input point cloud and perform the forward diffusion to disrupt potential adversarial shifts. Then we conduct the reverse denoising process to restore the disrupted point cloud back to a clean distribution. This approach enables effective defense against adaptive attacks with varying noise budgets, achieving accentuated robustness of existing 3D deep recognition models.

摘要: 深度3D点云模型对对抗性攻击很敏感，这会对自动驾驶等安全关键型应用程序构成威胁。稳健训练和降噪防御分别是防御对抗性扰动的典型策略，包括对抗性训练和统计过滤。然而，它们要么导致巨大的计算开销，要么严重依赖于特定的噪声先验，限制了对所有类型的攻击的普遍健壮性。本文介绍了一种新的基于去噪扩散模型的防御机制，该机制可以通过一个定制的强度估计器自适应地去除各种噪声。具体地说，我们首先通过计算点到其邻域最佳拟合平面的距离来估计对抗性失真。根据失真程度，我们为输入点云选择特定的扩散时间步长，并执行正向扩散来扰乱潜在的对抗性偏移。然后，我们进行反向去噪过程，将中断的点云恢复到干净的分布。这种方法能够有效地防御具有不同噪声预算的自适应攻击，实现了现有3D深度识别模型的增强稳健性。



## **44. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

利用去噪自动编码器防御大规模MIMO中基于深度学习的功率分配的敌意攻击 eess.SP

This work is currently under review for publication

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15365v2) [paper-pdf](http://arxiv.org/pdf/2211.15365v2)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.

摘要: 最近的工作主张使用深度学习在大规模MIMO(MaMIMO)网络的下行链路中执行功率分配。然而，这种深度学习模型很容易受到对手的攻击。在maMIMO功率分配的上下文中，对抗性攻击是指在推理期间将微妙的扰动注入深度学习模型的输入(即，在模型被训练后在部署期间将对抗性扰动注入输入)，这是专门定制的，以迫使训练的回归模型输出不可行的功率分配解。在这项工作中，我们开发了一种基于自动编码器的缓解技术，该技术允许基于深度学习的功率分配模型在对手存在的情况下运行，而不需要重新训练。具体地说，我们开发了一个去噪自动编码器(DAE)，它学习潜在扰动数据与其对应的未扰动输入之间的映射。我们在多个攻击和多个威胁模型中测试了我们的防御，并证明了它的能力：(I)使用两种常见的预编码方案缓解对抗性攻击对功率分配网络的影响；(Ii)优于先前提出的针对maMIMO网络的基于回归的对抗性攻击基准；(Iii)在没有攻击的情况下保持准确的性能；以及(Iv)以较低的计算开销运行。



## **45. Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks**

量化感知区间界传播用于训练可证明稳健的量化神经网络 cs.LG

Accepted at AAAI 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16187v1) [paper-pdf](http://arxiv.org/pdf/2211.16187v1)

**Authors**: Mathias Lechner, Đorđe Žikelić, Krishnendu Chatterjee, Thomas A. Henzinger, Daniela Rus

**Abstract**: We study the problem of training and certifying adversarially robust quantized neural networks (QNNs). Quantization is a technique for making neural networks more efficient by running them using low-bit integer arithmetic and is therefore commonly adopted in industry. Recent work has shown that floating-point neural networks that have been verified to be robust can become vulnerable to adversarial attacks after quantization, and certification of the quantized representation is necessary to guarantee robustness. In this work, we present quantization-aware interval bound propagation (QA-IBP), a novel method for training robust QNNs. Inspired by advances in robust learning of non-quantized networks, our training algorithm computes the gradient of an abstract representation of the actual network. Unlike existing approaches, our method can handle the discrete semantics of QNNs. Based on QA-IBP, we also develop a complete verification procedure for verifying the adversarial robustness of QNNs, which is guaranteed to terminate and produce a correct answer. Compared to existing approaches, the key advantage of our verification procedure is that it runs entirely on GPU or other accelerator devices. We demonstrate experimentally that our approach significantly outperforms existing methods and establish the new state-of-the-art for training and certifying the robustness of QNNs.

摘要: 研究了反向稳健量化神经网络(QNN)的训练和证明问题。量化是一种通过使用低位整数算法运行神经网络来提高神经网络效率的技术，因此在工业中被广泛采用。最近的工作表明，已被验证为健壮性的浮点神经网络在量化后容易受到敌意攻击，而量化表示的证明是保证健壮性的必要条件。在这项工作中，我们提出了量化感知区间界限传播(QA-IBP)，这是一种新的训练稳健QNN的方法。受非量化网络稳健学习的启发，我们的训练算法计算实际网络的抽象表示的梯度。与现有方法不同，我们的方法可以处理QNN的离散语义。在QA-IBP的基础上，我们还开发了一个完整的验证过程来验证QNN的对抗健壮性，该验证过程保证了QNN的终止和产生正确的答案。与现有的方法相比，我们的验证过程的关键优势是它完全运行在GPU或其他加速器设备上。实验表明，我们的方法明显优于现有的方法，并为QNN的训练和证明的健壮性建立了新的技术水平。



## **46. Understanding and Enhancing Robustness of Concept-based Models**

理解和增强基于概念的模型的健壮性 cs.LG

Accepted at AAAI 2023. Extended Version

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16080v1) [paper-pdf](http://arxiv.org/pdf/2211.16080v1)

**Authors**: Sanchit Sinha, Mengdi Huai, Jianhui Sun, Aidong Zhang

**Abstract**: Rising usage of deep neural networks to perform decision making in critical applications like medical diagnosis and financial analysis have raised concerns regarding their reliability and trustworthiness. As automated systems become more mainstream, it is important their decisions be transparent, reliable and understandable by humans for better trust and confidence. To this effect, concept-based models such as Concept Bottleneck Models (CBMs) and Self-Explaining Neural Networks (SENN) have been proposed which constrain the latent space of a model to represent high level concepts easily understood by domain experts in the field. Although concept-based models promise a good approach to both increasing explainability and reliability, it is yet to be shown if they demonstrate robustness and output consistent concepts under systematic perturbations to their inputs. To better understand performance of concept-based models on curated malicious samples, in this paper, we aim to study their robustness to adversarial perturbations, which are also known as the imperceptible changes to the input data that are crafted by an attacker to fool a well-learned concept-based model. Specifically, we first propose and analyze different malicious attacks to evaluate the security vulnerability of concept based models. Subsequently, we propose a potential general adversarial training-based defense mechanism to increase robustness of these systems to the proposed malicious attacks. Extensive experiments on one synthetic and two real-world datasets demonstrate the effectiveness of the proposed attacks and the defense approach.

摘要: 在医疗诊断和金融分析等关键应用中，越来越多的人使用深度神经网络进行决策，这引发了人们对它们的可靠性和可信度的担忧。随着自动化系统变得越来越主流，为了获得更好的信任和信心，重要的是它们的决策要透明、可靠，并能被人类理解。为此，人们提出了基于概念的模型，如概念瓶颈模型(CBMS)和自解释神经网络(SENN)，它们限制了模型的潜在空间，以表示领域专家容易理解的高级概念。尽管基于概念的模型承诺了一种提高可解释性和可靠性的好方法，但它们是否在输入受到系统扰动时表现出稳健性和输出一致的概念还有待证明。为了更好地理解基于概念的模型在经过精选的恶意样本上的性能，在本文中，我们旨在研究它们对对手扰动的鲁棒性，这种扰动也称为输入数据的不可察觉变化，这些变化是攻击者精心制作的，目的是愚弄一个学习良好的基于概念的模型。具体地说，我们首先提出并分析了不同的恶意攻击来评估基于概念的模型的安全漏洞。随后，我们提出了一种潜在的基于一般对抗性训练的防御机制，以增强这些系统对所提出的恶意攻击的健壮性。在一个合成数据集和两个真实数据集上的大量实验证明了所提出的攻击和防御方法的有效性。



## **47. Model Extraction Attack against Self-supervised Speech Models**

针对自监督语音模型的模型提取攻击 cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16044v1) [paper-pdf](http://arxiv.org/pdf/2211.16044v1)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.

摘要: 自监督学习(SSL)语音模型生成给定片段的有意义的表示，并在各种下游任务中获得令人难以置信的性能。模型提取攻击(MEA)通常是指攻击者仅通过查询访问来窃取受害者模型的功能。在这项工作中，我们研究了带有少量查询的SSL语音模型的MEA问题。我们提出了一个两阶段的模型提取框架。在第一阶段，对大规模的未标注语料库进行SSL，以预先训练一个小的语音模型。其次，我们从未标注的语料库中主动采样一小部分片段，并用这些片段查询目标模型，以获得它们的表示作为小模型第二阶段训练的标签。实验结果表明，我们的采样方法可以在不知道目标模型结构的情况下有效地提取目标模型。



## **48. AdvMask: A Sparse Adversarial Attack Based Data Augmentation Method for Image Classification**

AdvMASK：一种基于稀疏对抗性攻击的图像分类数据增强方法 cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16040v1) [paper-pdf](http://arxiv.org/pdf/2211.16040v1)

**Authors**: Suorong Yang, Jinqiao Li, Jian Zhao, Furao Shen

**Abstract**: Data augmentation is a widely used technique for enhancing the generalization ability of convolutional neural networks (CNNs) in image classification tasks. Occlusion is a critical factor that affects on the generalization ability of image classification models. In order to generate new samples, existing data augmentation methods based on information deletion simulate occluded samples by randomly removing some areas in the images. However, those methods cannot delete areas of the images according to their structural features of the images. To solve those problems, we propose a novel data augmentation method, AdvMask, for image classification tasks. Instead of randomly removing areas in the images, AdvMask obtains the key points that have the greatest influence on the classification results via an end-to-end sparse adversarial attack module. Therefore, we can find the most sensitive points of the classification results without considering the diversity of various image appearance and shapes of the object of interest. In addition, a data augmentation module is employed to generate structured masks based on the key points, thus forcing the CNN classification models to seek other relevant content when the most discriminative content is hidden. AdvMask can effectively improve the performance of classification models in the testing process. The experimental results on various datasets and CNN models verify that the proposed method outperforms other previous data augmentation methods in image classification tasks.

摘要: 数据增强是一种广泛使用的增强卷积神经网络(CNN)泛化能力的技术。遮挡是影响图像分类模型泛化能力的关键因素。为了生成新的样本，现有的基于信息删除的数据增强方法通过随机去除图像中的某些区域来模拟遮挡样本。然而，这些方法不能根据图像的结构特征来删除图像区域。为了解决这些问题，我们提出了一种新的数据增强方法AdvMASK，用于图像分类任务。该算法通过端到端稀疏对抗攻击模块获取对分类结果影响最大的关键点，而不是随机去除图像中的区域。因此，我们可以在不考虑感兴趣对象的各种图像外观和形状的多样性的情况下，找到分类结果中最敏感的点。此外，使用数据增强模块根据关键点生成结构化掩码，从而迫使CNN分类模型在最具区别性的内容被隐藏时寻找其他相关内容。在测试过程中，AdvMASK可以有效地提高分类模型的性能。在不同数据集和CNN模型上的实验结果验证了该方法在图像分类任务中的性能优于以往的其他数据增强方法。



## **49. Interpretations Cannot Be Trusted: Stealthy and Effective Adversarial Perturbations against Interpretable Deep Learning**

解释不可信：对可解释深度学习的隐秘而有效的对抗性干扰 cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15926v1) [paper-pdf](http://arxiv.org/pdf/2211.15926v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning methods have gained increased attention in various applications due to their outstanding performance. For exploring how this high performance relates to the proper use of data artifacts and the accurate problem formulation of a given task, interpretation models have become a crucial component in developing deep learning-based systems. Interpretation models enable the understanding of the inner workings of deep learning models and offer a sense of security in detecting the misuse of artifacts in the input data. Similar to prediction models, interpretation models are also susceptible to adversarial inputs. This work introduces two attacks, AdvEdge and AdvEdge$^{+}$, that deceive both the target deep learning model and the coupled interpretation model. We assess the effectiveness of proposed attacks against two deep learning model architectures coupled with four interpretation models that represent different categories of interpretation models. Our experiments include the attack implementation using various attack frameworks. We also explore the potential countermeasures against such attacks. Our analysis shows the effectiveness of our attacks in terms of deceiving the deep learning models and their interpreters, and highlights insights to improve and circumvent the attacks.

摘要: 深度学习方法由于其优异的性能在各种应用中得到了越来越多的关注。为了探索这种高性能如何与数据人工制品的正确使用和给定任务的准确问题表达有关，解释模型已经成为开发基于深度学习的系统的关键组成部分。解释模型能够理解深度学习模型的内部工作原理，并在检测输入数据中的伪像误用时提供一种安全感。与预测模型类似，解释模型也容易受到对抗性输入的影响。本文介绍了两种欺骗目标深度学习模型和耦合解释模型的攻击方法：AdvEdge和AdvEdge。我们评估了针对两个深度学习模型体系结构和代表不同类别解释模型的四个解释模型的攻击的有效性。我们的实验包括使用各种攻击框架的攻击实现。我们还探讨了针对此类攻击的潜在对策。我们的分析显示了我们的攻击在欺骗深度学习模型及其解释器方面的有效性，并强调了改进和规避攻击的见解。



## **50. Training Time Adversarial Attack Aiming the Vulnerability of Continual Learning**

针对持续学习脆弱性的训练时间对抗性攻击 cs.LG

Accepted at NeurIPS 2022 ML Safety Workshop

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15875v1) [paper-pdf](http://arxiv.org/pdf/2211.15875v1)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world setting which has memory and privacy issues. However, this introduces a problem in these models by not being able to track the performance on each task. In other words, current continual learning methods are vulnerable to attacks done on the previous task. We demonstrate the vulnerability of regularization-based continual learning methods by presenting simple task-specific training time adversarial attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. Experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attack.

摘要: 通常，基于正则化的持续学习模型限制对先前任务数据的访问，以模拟存在记忆和隐私问题的真实世界环境。然而，这在这些模型中引入了一个问题，因为无法跟踪每个任务的性能。换句话说，当前的持续学习方法很容易受到对前一任务的攻击。我们通过简单的任务特定训练时间的对抗性攻击来展示基于正则化的持续学习方法的脆弱性，这些攻击可以用于新任务的学习过程。建议的攻击生成的训练数据会导致攻击者针对的特定任务的性能下降。实验结果验证了本文提出的脆弱性，并证明了开发对对手攻击具有健壮性的持续学习模型的重要性。



