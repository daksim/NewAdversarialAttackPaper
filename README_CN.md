# Latest Adversarial Attack Papers
**update at 2022-10-18 16:59:54**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

神枪手后门：任意目标等级的后门攻击 cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.

摘要: 近年来，机器学习模型被证明容易受到后门攻击。在这样的攻击下，对手在训练的模型中嵌入一个秘密的后门，这样受攻击的模型将在干净的输入上正常运行，但将根据对手对带有触发器的恶意构建的输入的控制进行错误分类。虽然这些现有的攻击非常有效，但对手的能力是有限的：在给定输入的情况下，这些攻击只能导致模型错误分类为单个预定义或目标类。相反，本文利用了一种新的后门攻击，具有更强大的有效载荷，表示为射手，其中对手可以任意选择模型将错误分类的目标类别，在推理过程中给定任何输入。为了实现这一目标，我们建议将触发函数表示为类条件生成模型，并在约束优化框架中注入后门，其中触发函数学习生成任意攻击目标类的最优触发模式，同时将该生成后门嵌入到训练的模型中。给定学习的触发器生成函数，在推理期间，对手可以指定任意的后门攻击目标类，并且相应地创建导致模型向该目标类分类的适当触发器。我们在MNIST、CIFAR10、GTSRB和TinyImageNet等几个基准数据集上的实验表明，该框架在保持干净数据性能的同时，实现了高的攻击性能。拟议的射手后门攻击也可以很容易地绕过现有的后门防御，这些后门防御最初是针对单一目标类别的后门攻击而设计的。我们的工作朝着了解后门攻击在实践中的广泛风险又迈出了重要的一步。



## **2. Adversarial Robustness is at Odds with Lazy Training**

对抗健壮性与懒惰训练不一致 cs.CR

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.

摘要: 最近的工作表明，随机神经网络存在对抗性的例子[Daniely和Schacham，2020]，并且这些例子可以使用单一的梯度上升步骤来找到[Bubeck等人，2021]。在这项工作中，我们将这一工作扩展到神经网络的“懒惰训练”--深度学习理论中的一种主要模型，在这种模型中，神经网络被证明是可有效学习的。我们证明了过度参数化的神经网络具有良好的泛化能力和强大的计算保证，但仍然容易受到使用单步梯度上升产生的攻击。



## **3. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

反裁剪：用于预先训练的分类器的数据高效认证稳健性 cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.

摘要: 使用随机化平滑的认证防御是一种流行的技术，可以为深层神经网络提供抵御L2攻击的健壮性保证。现有的工作使用这种技术来通过在整个训练数据上训练自定义去噪器网络来证明预先训练的非稳健模型的安全。然而，由于诸如高传输成本和数据的专有性质等限制，对训练集的访问可能被限制为少数数据样本。因此，我们提出了一个新的问题：如何仅用几个训练样本来证明预先训练的模型的稳健性。我们观察到，直接使用现有技术在有限的样本上培训自定义去噪器会产生较差的认证。为了克服这一问题，我们提出的方法(DE-CROP)生成对应于每个训练样本的类边界样本和内插样本，从而确保预先训练的分类器的特征空间具有高度的多样性。在分类器的Logit空间中，我们通过最大化生成样本的去噪输出与原始训练样本之间的相似度来训练去噪器。我们还使用域鉴别器和最大平均差异来执行分布级别匹配，从而产生进一步的好处。在白盒设置中，我们在多个基准数据集上获得了比基线显著的改进，并且在具有挑战性的黑盒设置下也报告了类似的性能。



## **4. Beyond Model Interpretability: On the Faithfulness and Adversarial Robustness of Contrastive Textual Explanations**

超越模式的可解释性--论对比文本解释的忠实性和对抗性 cs.CL

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08902v1) [paper-pdf](http://arxiv.org/pdf/2210.08902v1)

**Authors**: Julia El Zini, Mariette Awad

**Abstract**: Contrastive explanation methods go beyond transparency and address the contrastive aspect of explanations. Such explanations are emerging as an attractive option to provide actionable change to scenarios adversely impacted by classifiers' decisions. However, their extension to textual data is under-explored and there is little investigation on their vulnerabilities and limitations.   This work motivates textual counterfactuals by laying the ground for a novel evaluation scheme inspired by the faithfulness of explanations. Accordingly, we extend the computation of three metrics, proximity,connectedness and stability, to textual data and we benchmark two successful contrastive methods, POLYJUICE and MiCE, on our suggested metrics. Experiments on sentiment analysis data show that the connectedness of counterfactuals to their original counterparts is not obvious in both models. More interestingly, the generated contrastive texts are more attainable with POLYJUICE which highlights the significance of latent representations in counterfactual search. Finally, we perform the first semantic adversarial attack on textual recourse methods. The results demonstrate the robustness of POLYJUICE and the role that latent input representations play in robustness and reliability.

摘要: 对比解释方法超越了透明度，解决了解释的对比方面。这样的解释正在成为一种有吸引力的选择，可以为受到分类员决定不利影响的情况提供可操作的改变。然而，它们对文本数据的扩展还没有得到充分的探索，对它们的脆弱性和局限性的调查也很少。这项工作通过为一个新的评估方案奠定了基础，从而激发了文本反事实的动机，该方案的灵感来自于解释的真实性。相应地，我们将邻近度、连通度和稳定性这三个指标的计算扩展到文本数据，并在我们建议的指标上对两种成功的对比方法Polyjuus和MICE进行了基准测试。在情感分析数据上的实验表明，在这两个模型中，反事实与原始事实的联系并不明显。更有趣的是，生成的对比文本更容易通过Polyjuus获得，这突显了潜在表征在反事实搜索中的重要性。最后，我们对文本求助方法进行了首次语义对抗性攻击。实验结果证明了Polyjuus的稳健性，以及潜在输入表征在稳健性和可靠性方面所起的作用。



## **5. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08870v1) [paper-pdf](http://arxiv.org/pdf/2210.08870v1)

**Authors**: Jialiang Sun

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **6. ODG-Q: Robust Quantization via Online Domain Generalization**

ODG-Q：基于在线域泛化的稳健量化 cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08701v1) [paper-pdf](http://arxiv.org/pdf/2210.08701v1)

**Authors**: Chaofan Tao, Ngai Wong

**Abstract**: Quantizing neural networks to low-bitwidth is important for model deployment on resource-limited edge hardware. Although a quantized network has a smaller model size and memory footprint, it is fragile to adversarial attacks. However, few methods study the robustness and training efficiency of quantized networks. To this end, we propose a new method by recasting robust quantization as an online domain generalization problem, termed ODG-Q, which generates diverse adversarial data at a low cost during training. ODG-Q consistently outperforms existing works against various adversarial attacks. For example, on CIFAR-10 dataset, ODG-Q achieves 49.2% average improvements under five common white-box attacks and 21.7% average improvements under five common black-box attacks, with a training cost similar to that of natural training (viz. without adversaries). To our best knowledge, this work is the first work that trains both quantized and binary neural networks on ImageNet that consistently improve robustness under different attacks. We also provide a theoretical insight of ODG-Q that accounts for the bound of model risk on attacked data.

摘要: 将神经网络量化为低位宽对于在资源有限的边缘硬件上部署模型具有重要意义。虽然量化网络具有较小的模型大小和内存占用，但它很容易受到对手攻击。然而，很少有方法研究量化网络的稳健性和训练效率。为此，我们提出了一种新的方法，将稳健量化重塑为一个在线领域泛化问题，称为ODG-Q，该方法在训练过程中以较低的代价生成不同的对抗性数据。ODG-Q在抵抗各种对抗性攻击时的表现一直优于现有的工作。例如，在CIFAR-10数据集上，在五种常见的白盒攻击下，ODG-Q的平均性能提高了49.2%，在五种常见的黑盒攻击下，ODG-Q的平均性能提高了21.7%，而训练代价与自然训练(即.没有对手)。据我们所知，这是第一个在ImageNet上训练量化和二进制神经网络的工作，这些网络在不同的攻击下都能持续提高健壮性。我们还提供了ODG-Q的理论见解，它解释了攻击数据上的模型风险的界限。



## **7. A2: Efficient Automated Attacker for Boosting Adversarial Training**

A2：用于加强对抗性训练的高效自动攻击者 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.03543v2) [paper-pdf](http://arxiv.org/pdf/2210.03543v2)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.

摘要: 在对抗训练显著提高模型稳健性的基础上，各种变种被提出以进一步提高性能。公认的方法侧重于AT的不同组成部分(例如，设计损失函数和利用额外的未标记数据)。人们普遍认为，更强的扰动会产生更稳健的模型。然而，如何有效地产生更强的扰动仍然是一个未解决的问题。在本文中，我们提出了一种称为A2的高效自动攻击者，通过在训练过程中生成最优的动态扰动来增强AT。A2是一个参数化的自动攻击者，可以在攻击者空间中搜索最好的攻击者，并针对防御模型和实例进行攻击。在不同数据集上的大量实验表明，A2以较低的额外代价产生更强的扰动，并可靠地提高了各种AT方法对不同攻击的健壮性。



## **8. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

基于机器学习的钓鱼URL检测器的可靠性和稳健性分析 cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2005.08454v2) [paper-pdf](http://arxiv.org/pdf/2005.08454v2)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.

摘要: 基于ML的钓鱼URL(MLPU)检测器是保护用户和组织免受钓鱼攻击的第一级防御。最近，很少有研究针对特定的MLPU检测器发起成功的对抗性攻击，这引发了对其实际可靠性和使用的质疑。然而，这些系统的稳健性还没有得到广泛的研究。因此，这些系统的安全漏洞总体上仍然是未知的，这就需要测试这些系统的健壮性。在本文中，我们提出了一种方法来调查50个最具代表性的MLPU模型的可靠性和稳健性。首先，我们提出了一个高性价比的敌意URL生成器URLBUG，它创建了一个敌意URL数据集。随后，我们复制了50个MLPU(传统ML和深度学习)系统，并记录了它们的基线性能。最后，我们在敌意数据集上对所考虑的MLPU系统进行了测试，并使用盒图和热图分析了它们的健壮性和可靠性。我们的结果表明，生成的恶意URL具有有效的语法，并且可以以11.99美元的中位数年价格注册。在已注册的13个恶意URL中，有63.94个被用于恶意目的。此外，所考虑的MLPU模型马修相关系数(MCC)从中位数0.92下降到0.02，表明基线MLPU模型目前的形式是不可靠的。此外，我们的发现发现了这些系统的几个安全漏洞，并为研究人员设计可靠和安全的MLPU系统提供了未来的方向。



## **9. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

漏洞后恢复：针对泄漏的DNN模型的白盒对抗示例 cs.CR

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2205.10686v2) [paper-pdf](http://arxiv.org/pdf/2205.10686v2)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstract**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.

摘要: 在当今的互联网上，服务器入侵是一个不幸的现实。在深度神经网络(DNN)模型的背景下，它们尤其有害，因为泄露的模型让攻击者可以使用“白盒”来生成对抗性示例，这是一种没有实际可靠防御措施的威胁模型。对于在专有DNN(例如医学成像)上投入多年和数百万美元的从业者来说，这似乎是一场不可避免的灾难迫在眉睫。在本文中，我们考虑了DNN模型的漏洞后恢复问题。我们提出了Neo，一个新的系统，它创建新版本的泄漏模型，以及一个推理时间过滤器，检测并删除在以前泄漏的模型上生成的敌对示例。不同模型版本的分类面略有偏移(通过引入隐藏分布)，并且Neo检测到对其生成中使用的泄漏模型的攻击过拟合。我们发现，在各种任务和攻击方法中，Neo能够以非常高的准确率从泄露的模型中过滤攻击，并针对反复破坏服务器的攻击者提供强大的保护(7-10次恢复)。NEO在各种强自适应攻击中表现良好，在可恢复的漏洞数量中略有下降，并显示出在野外作为DNN防御的补充潜力。



## **10. Robust Feature-Level Adversaries are Interpretability Tools**

强大的功能级对手是可解释的工具 cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2110.03605v5) [paper-pdf](http://arxiv.org/pdf/2110.03605v5)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv

摘要: 关于计算机视觉中的对抗性攻击的文献通常集中在像素级的扰动上。这些往往很难解释。最近的工作是利用图像生成器的潜在表示来创建“特征级别”的对抗性扰动，这给了我们一个探索可感知的、可解释的对抗性攻击的机会。我们有三点贡献。首先，我们观察到特征级别的攻击为学习模型中的表示提供了有用的输入类。其次，我们证明了这些对手是多才多艺的，并且非常健壮。我们证明了它们可以用于在ImageNet规模上产生有针对性的、普遍的、伪装的、物理上可实现的和黑匣子攻击。第三，我们展示了如何将这些对抗性图像用作识别网络漏洞的实用可解释性工具。我们利用这些对手来预测特征和类别之间的虚假关联，然后通过设计“复制/粘贴”攻击来测试这些关联，在这种攻击中，一幅自然图像被粘贴到另一幅图像中，从而导致有针对性的误分类。我们的结果表明，特征级攻击对于严格的可解释性研究是一种很有前途的方法。它们支持工具的设计，以更好地理解模型学习到的内容并诊断脆弱的特征关联。代码可在https://github.com/thestephencasper/feature_level_adv上找到



## **11. Nowhere to Hide: A Lightweight Unsupervised Detector against Adversarial Examples**

无处藏身：针对敌意例子的轻量级无监督检测器 cs.LG

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08579v1) [paper-pdf](http://arxiv.org/pdf/2210.08579v1)

**Authors**: Hui Liu, Bo Zhao, Kehuan Zhang, Peng Liu

**Abstract**: Although deep neural networks (DNNs) have shown impressive performance on many perceptual tasks, they are vulnerable to adversarial examples that are generated by adding slight but maliciously crafted perturbations to benign images. Adversarial detection is an important technique for identifying adversarial examples before they are entered into target DNNs. Previous studies to detect adversarial examples either targeted specific attacks or required expensive computation. How design a lightweight unsupervised detector is still a challenging problem. In this paper, we propose an AutoEncoder-based Adversarial Examples (AEAE) detector, that can guard DNN models by detecting adversarial examples with low computation in an unsupervised manner. The AEAE includes only a shallow autoencoder but plays two roles. First, a well-trained autoencoder has learned the manifold of benign examples. This autoencoder can produce a large reconstruction error for adversarial images with large perturbations, so we can detect significantly perturbed adversarial examples based on the reconstruction error. Second, the autoencoder can filter out the small noise and change the DNN's prediction on adversarial examples with small perturbations. It helps to detect slightly perturbed adversarial examples based on the prediction distance. To cover these two cases, we utilize the reconstruction error and prediction distance from benign images to construct a two-tuple feature set and train an adversarial detector using the isolation forest algorithm. We show empirically that the AEAE is unsupervised and inexpensive against the most state-of-the-art attacks. Through the detection in these two cases, there is nowhere to hide adversarial examples.

摘要: 尽管深度神经网络(DNN)在许多感知任务中表现出了令人印象深刻的性能，但它们很容易受到通过向良性图像添加轻微但恶意制作的扰动而产生的敌意示例。敌意检测是在敌意实例进入目标DNN之前识别它们的一项重要技术。以前检测对抗性例子的研究要么是针对特定攻击，要么是需要昂贵的计算。如何设计一个轻量级的无监督检测器仍然是一个具有挑战性的问题。本文提出了一种基于自动编码器的对抗性实例检测器(AEAE)，该检测器能够以无监督的方式以较低的计算量检测敌意实例，从而保护DNN模型。AEAE只包括一个浅层自动编码器，但它扮演着两个角色。首先，训练有素的自动编码器已经学会了大量良性的例子。该自动编码器对扰动较大的对抗性图像会产生较大的重建误差，因此可以根据重建误差检测出明显扰动的对抗性实例。其次，自动编码器可以滤除小噪声，并在小扰动的情况下改变DNN对对抗性样本的预测。它有助于基于预测距离检测略微扰动的敌意示例。为了覆盖这两种情况，我们利用良性图像的重建误差和预测距离来构造一个二元组特征集，并使用隔离森林算法来训练敌意检测器。我们的经验表明，AEAE在应对最先进的攻击时是无人监督的，而且成本低廉。通过对这两起案件的侦破，对抗性事例无处藏身。



## **12. Object-Attentional Untargeted Adversarial Attack**

目标--注意的非目标对抗性攻击 cs.CV

**SubmitDate**: 2022-10-16    [abs](http://arxiv.org/abs/2210.08472v1) [paper-pdf](http://arxiv.org/pdf/2210.08472v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Guopu Zhu

**Abstract**: Deep neural networks are facing severe threats from adversarial attacks. Most existing black-box attacks fool target model by generating either global perturbations or local patches. However, both global perturbations and local patches easily cause annoying visual artifacts in adversarial example. Compared with some smooth regions of an image, the object region generally has more edges and a more complex texture. Thus small perturbations on it will be more imperceptible. On the other hand, the object region is undoubtfully the decisive part of an image to classification tasks. Motivated by these two facts, we propose an object-attentional adversarial attack method for untargeted attack. Specifically, we first generate an object region by intersecting the object detection region from YOLOv4 with the salient object detection (SOD) region from HVPNet. Furthermore, we design an activation strategy to avoid the reaction caused by the incomplete SOD. Then, we perform an adversarial attack only on the detected object region by leveraging Simple Black-box Adversarial Attack (SimBA). To verify the proposed method, we create a unique dataset by extracting all the images containing the object defined by COCO from ImageNet-1K, named COCO-Reduced-ImageNet in this paper. Experimental results on ImageNet-1K and COCO-Reduced-ImageNet show that under various system settings, our method yields the adversarial example with better perceptual quality meanwhile saving the query budget up to 24.16\% compared to the state-of-the-art approaches including SimBA.

摘要: 深度神经网络正面临着来自对手攻击的严重威胁。现有的大多数黑盒攻击通过产生全局扰动或局部补丁来愚弄目标模型。然而，在对抗性例子中，全局扰动和局部斑块都容易造成令人讨厌的视觉伪影。与图像的一些平滑区域相比，目标区域通常具有更多的边缘和更复杂的纹理。因此，对它的微小扰动将更加难以察觉。另一方面，目标区域无疑是一幅图像进行分类任务的决定性部分。受这两个事实的启发，我们提出了一种针对非定向攻击的对象注意对抗性攻击方法。具体地，我们首先通过将来自YOLOv4的目标检测区域与来自HVPNet的显著目标检测(SOD)区域相交来生成目标区域。此外，我们设计了一种激活策略，以避免由于不完整的SOD而引起的反应。然后，我们利用简单黑盒对抗攻击(Simba)只对检测到的目标区域执行对抗性攻击。为了验证所提出的方法，我们从ImageNet-1K中提取包含CoCo定义的对象的所有图像来创建唯一的数据集，本文将其命名为CoCo-Reduced-ImageNet。在ImageNet-1K和Coco-Reduced-ImageNet上的实验结果表明，在不同的系统设置下，我们的方法生成的对抗性实例具有更好的感知质量，同时与包括SIMBA在内的最新方法相比，可以节省高达24.16\%的查询预算。



## **13. RoS-KD: A Robust Stochastic Knowledge Distillation Approach for Noisy Medical Imaging**

ROS-KD：一种适用于噪声医学成像的稳健随机知识提取方法 cs.CV

Accepted in ICDM 2022

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08388v1) [paper-pdf](http://arxiv.org/pdf/2210.08388v1)

**Authors**: Ajay Jaiswal, Kumar Ashutosh, Justin F Rousseau, Yifan Peng, Zhangyang Wang, Ying Ding

**Abstract**: AI-powered Medical Imaging has recently achieved enormous attention due to its ability to provide fast-paced healthcare diagnoses. However, it usually suffers from a lack of high-quality datasets due to high annotation cost, inter-observer variability, human annotator error, and errors in computer-generated labels. Deep learning models trained on noisy labelled datasets are sensitive to the noise type and lead to less generalization on the unseen samples. To address this challenge, we propose a Robust Stochastic Knowledge Distillation (RoS-KD) framework which mimics the notion of learning a topic from multiple sources to ensure deterrence in learning noisy information. More specifically, RoS-KD learns a smooth, well-informed, and robust student manifold by distilling knowledge from multiple teachers trained on overlapping subsets of training data. Our extensive experiments on popular medical imaging classification tasks (cardiopulmonary disease and lesion classification) using real-world datasets, show the performance benefit of RoS-KD, its ability to distill knowledge from many popular large networks (ResNet-50, DenseNet-121, MobileNet-V2) in a comparatively small network, and its robustness to adversarial attacks (PGD, FSGM). More specifically, RoS-KD achieves >2% and >4% improvement on F1-score for lesion classification and cardiopulmonary disease classification tasks, respectively, when the underlying student is ResNet-18 against recent competitive knowledge distillation baseline. Additionally, on cardiopulmonary disease classification task, RoS-KD outperforms most of the SOTA baselines by ~1% gain in AUC score.

摘要: 人工智能支持的医学成像最近获得了极大的关注，因为它能够提供快节奏的医疗诊断。然而，由于高昂的注释成本、观察者之间的可变性、人为注释员错误以及计算机生成的标签中的错误，它通常缺乏高质量的数据集。在有噪声标记的数据集上训练的深度学习模型对噪声类型敏感，导致对不可见样本的泛化程度较低。为了应对这一挑战，我们提出了一个稳健的随机知识蒸馏(ROS-KD)框架，它模仿了从多个来源学习一个主题的概念，以确保在学习噪声信息时具有威慑作用。更具体地说，ROS-KD通过从多个教师那里提取知识来学习流畅、消息灵通和健壮的学生流形，这些知识来自于在重叠的训练数据子集上训练的多个教师。我们使用真实世界的数据集对流行的医学图像分类任务(心肺疾病和病变分类)进行了广泛的实验，表明了Ros-KD的性能优势，它能够在相对较小的网络中从许多流行的大型网络(ResNet-50，DenseNet-121，MobileNet-V2)中提取知识，以及它对对手攻击(PGD，FSGM)的鲁棒性。更具体地说，当基础学生是ResNet-18而不是最近的竞争性知识蒸馏基线时，ROS-KD在病变分类和心肺疾病分类任务中分别比F1分数提高>2%和>4%。此外，在心肺疾病分类任务上，ROS-KD在AUC评分上比大多数SOTA基线高出约1%。



## **14. GAMA: Generative Adversarial Multi-Object Scene Attacks**

GAMA：生成性对抗性多目标场景攻击 cs.CV

Accepted at NeurIPS 2022; First two authors contributed equally;  Includes Supplementary Material

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2209.09502v2) [paper-pdf](http://arxiv.org/pdf/2209.09502v2)

**Authors**: Abhishek Aich, Calvin-Khang Ta, Akash Gupta, Chengyu Song, Srikanth V. Krishnamurthy, M. Salman Asif, Amit K. Roy-Chowdhury

**Abstract**: The majority of methods for crafting adversarial attacks have focused on scenes with a single dominant object (e.g., images from ImageNet). On the other hand, natural scenes include multiple dominant objects that are semantically related. Thus, it is crucial to explore designing attack strategies that look beyond learning on single-object scenes or attack single-object victim classifiers. Due to their inherent property of strong transferability of perturbations to unknown models, this paper presents the first approach of using generative models for adversarial attacks on multi-object scenes. In order to represent the relationships between different objects in the input scene, we leverage upon the open-sourced pre-trained vision-language model CLIP (Contrastive Language-Image Pre-training), with the motivation to exploit the encoded semantics in the language space along with the visual space. We call this attack approach Generative Adversarial Multi-object scene Attacks (GAMA). GAMA demonstrates the utility of the CLIP model as an attacker's tool to train formidable perturbation generators for multi-object scenes. Using the joint image-text features to train the generator, we show that GAMA can craft potent transferable perturbations in order to fool victim classifiers in various attack settings. For example, GAMA triggers ~16% more misclassification than state-of-the-art generative approaches in black-box settings where both the classifier architecture and data distribution of the attacker are different from the victim. Our code is available here: https://abhishekaich27.github.io/gama.html

摘要: 大多数制作敌意攻击的方法都集中在具有单一主导对象的场景(例如，来自ImageNet的图像)。另一方面，自然场景包括多个语义相关的主导对象。因此，探索设计超越学习单对象场景或攻击单对象受害者分类器的攻击策略是至关重要的。由于产生式模型对未知模型具有很强的可转移性，本文首次提出了利用产生式模型进行多目标场景对抗性攻击的方法。为了表示输入场景中不同对象之间的关系，我们利用开源的预先训练的视觉语言模型剪辑(Contrastive Language-Image Pre-Training)，目的是利用语言空间和视觉空间中的编码语义。我们称这种攻击方式为生成性对抗性多对象场景攻击(GAMA)。GAMA演示了剪辑模型作为攻击者的工具的效用，以训练用于多对象场景的强大的扰动生成器。使用联合图文特征训练生成器，我们证明了GAMA能够在不同的攻击环境下制造有效的可转移扰动来愚弄受害者分类器。例如，在攻击者的分类器体系结构和数据分布都与受害者不同的黑盒环境中，GAMA触发的错误分类方法比最先进的生成性方法高出约16%。我们的代码可在此处获得：https://abhishekaich27.github.io/gama.html



## **15. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

友善噪声对抗敌意噪声：数据中毒攻击的有力防御 cs.CR

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.10224v2) [paper-pdf](http://arxiv.org/pdf/2208.10224v2)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.

摘要: 一种强大的(不可见)数据中毒攻击通过小的对抗性扰动来修改训练样本的子集，以改变对某些测试时间数据的预测。现有的防御机制在实践中并不可取，因为它们通常要么严重损害泛化性能，要么是特定于攻击的，应用速度慢得令人望而却步。在这里，我们提出了一种简单而高效的方法，不同于现有的方法，它以最小的泛化性能下降来破解各种类型的隐形中毒攻击。我们的关键观察是，攻击引入了高训练损失的局部尖锐区域，当训练损失最小化时，导致学习对手的扰动，使攻击成功。要打破毒物攻击，我们的关键思想是减轻毒物引入的急剧损失区域。为此，我们的方法包括两个组件：一个优化的友好噪声，它被生成以在不降低性能的情况下最大限度地扰动示例，以及一个随机变化的噪声组件。这两个组件的组合构建了一个非常轻但极其有效的防御系统，以抵御最强大的无触发器定向和隐藏触发器后门中毒攻击，包括梯度匹配、公牛眼多面体和睡眠代理。我们证明了我们的友好噪声是可以转移到其他体系结构的，而自适应攻击由于其随机噪声成分而不能破坏我们的防御。



## **16. A Scalable Reinforcement Learning Approach for Attack Allocation in Swarm to Swarm Engagement Problems**

群对群交战问题中攻击分配的可扩展强化学习方法 cs.RO

submitted to ICRA 2023

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08319v1) [paper-pdf](http://arxiv.org/pdf/2210.08319v1)

**Authors**: Umut Demir, Nazim Kemal Ure

**Abstract**: In this work we propose a reinforcement learning (RL) framework that controls the density of a large-scale swarm for engaging with adversarial swarm attacks. Although there is a significant amount of existing work in applying artificial intelligence methods to swarm control, analysis of interactions between two adversarial swarms is a rather understudied area. Most of the existing work in this subject develop strategies by making hard assumptions regarding the strategy and dynamics of the adversarial swarm. Our main contribution is the formulation of the swarm to swarm engagement problem as a Markov Decision Process and development of RL algorithms that can compute engagement strategies without the knowledge of strategy/dynamics of the adversarial swarm. Simulation results show that the developed framework can handle a wide array of large-scale engagement scenarios in an efficient manner.

摘要: 在这项工作中，我们提出了一种强化学习(RL)框架，用于控制大规模群的密度，以应对对抗性的群攻击。虽然现有的大量工作是应用人工智能方法来控制种群，但分析两个敌对种群之间的相互作用是一个相当不充分的研究领域。本学科现有的大多数工作都是通过对敌方蜂群的战略和动态做出硬假设来制定战略的。我们的主要贡献是将群到群的参与问题描述为一个马尔可夫决策过程，并开发了RL算法，该算法可以在不了解对手群的策略/动态的情况下计算参与策略。仿真结果表明，该框架能够高效地处理多种大规模交战场景。



## **17. Robust Binary Models by Pruning Randomly-initialized Networks**

剪枝随机初始化网络的稳健二进制模型 cs.LG

Accepted as NeurIPS 2022 paper

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2202.01341v2) [paper-pdf](http://arxiv.org/pdf/2202.01341v2)

**Authors**: Chen Liu, Ziqi Zhao, Sabine Süsstrunk, Mathieu Salzmann

**Abstract**: Robustness to adversarial attacks was shown to require a larger model capacity, and thus a larger memory footprint. In this paper, we introduce an approach to obtain robust yet compact models by pruning randomly-initialized binary networks. Unlike adversarial training, which learns the model parameters, we initialize the model parameters as either +1 or -1, keep them fixed, and find a subnetwork structure that is robust to attacks. Our method confirms the Strong Lottery Ticket Hypothesis in the presence of adversarial attacks, and extends this to binary networks. Furthermore, it yields more compact networks with competitive performance than existing works by 1) adaptively pruning different network layers; 2) exploiting an effective binary initialization scheme; 3) incorporating a last batch normalization layer to improve training stability. Our experiments demonstrate that our approach not only always outperforms the state-of-the-art robust binary networks, but also can achieve accuracy better than full-precision ones on some datasets. Finally, we show the structured patterns of our pruned binary networks.

摘要: 对敌意攻击的稳健性被证明需要更大的模型容量，因此需要更大的内存占用。在本文中，我们介绍了一种通过剪枝随机初始化的二进制网络来获得健壮而紧凑的模型的方法。与学习模型参数的对抗性训练不同，我们将模型参数初始化为+1或-1，保持其固定，并找到一个对攻击具有鲁棒性的子网络结构。我们的方法证实了存在对抗性攻击时的强彩票假设，并将其推广到二进制网络。此外，它通过1)自适应剪枝不同的网络层；2)采用有效的二进制初始化方案；3)加入最后一批归一化层来提高训练稳定性，从而产生了比现有工作更紧凑、性能更具竞争力的网络。我们的实验表明，我们的方法不仅在性能上总是优于最先进的健壮的二进制网络，而且在一些数据集上可以达到比全精度方法更好的准确率。最后，我们展示了我们的剪枝二进制网络的结构模式。



## **18. Overparameterization from Computational Constraints**

计算约束的超参数化 cs.LG

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2208.12926v2) [paper-pdf](http://arxiv.org/pdf/2208.12926v2)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstract**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.

摘要: 具有数百万个参数的过度参数模型已经取得了巨大的成功。在这项工作中，我们问：对大型模型的需求是否至少部分是由于学习者的计算限制？此外，我们问，这种情况是否会因为学习而加剧？我们表明，情况确实可能是这样的。我们展示了计算受限的学习者比信息论学习者需要更多的模型参数的学习任务。此外，我们还表明，稳健学习可能需要更多的模型参数。特别是，对于计算有界的学习者，我们将Bubeck和Sellke[NeurIPS‘2021]的最新结果推广到计算机制，该结果表明健壮模型可能需要更多的参数，并表明有界的学习者可能需要更多的参数。然后，我们解决了以下相关问题：为了获得参数更少的模型，我们是否可以通过限制对手也是计算有界的来纠正健壮的计算有界学习的情况？在这里，我们再次证明了这是可能的。具体地说，在Garg，Jha，MahLoujifar和Mahmoody[Alt‘2020]的工作基础上，我们演示了一种学习任务，该任务可以在计算受限的攻击者面前高效而稳健地学习，而为了对信息论攻击者具有健壮性，需要学习者使用更多的参数。



## **19. Distributionally Robust Multiclass Classification and Applications in Deep Image Classifiers**

分布稳健多类分类及其在深度图像分类器中的应用 cs.CV

arXiv admin note: substantial text overlap with arXiv:2109.12772

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08198v1) [paper-pdf](http://arxiv.org/pdf/2210.08198v1)

**Authors**: Ruidi Chen, Boran Hao, Ioannis Ch. Paschalidis

**Abstract**: We develop a Distributionally Robust Optimization (DRO) formulation for Multiclass Logistic Regression (MLR), which could tolerate data contaminated by outliers. The DRO framework uses a probabilistic ambiguity set defined as a ball of distributions that are close to the empirical distribution of the training set in the sense of the Wasserstein metric. We relax the DRO formulation into a regularized learning problem whose regularizer is a norm of the coefficient matrix. We establish out-of-sample performance guarantees for the solutions to our model, offering insights on the role of the regularizer in controlling the prediction error. We apply the proposed method in rendering deep Vision Transformer (ViT)-based image classifiers robust to random and adversarial attacks. Specifically, using the MNIST and CIFAR-10 datasets, we demonstrate reductions in test error rate by up to 83.5% and loss by up to 91.3% compared with baseline methods, by adopting a novel random training method.

摘要: 针对多分类Logistic回归(MLR)模型，提出了一种分布式稳健优化(DRO)方法，该方法能够容忍数据中的离群点污染。DRO框架使用概率模糊集，该模糊集被定义为在Wasserstein度量意义上接近训练集的经验分布的分布球。我们将DRO公式松弛为一个正则化学习问题，其正则化子是系数矩阵的范数。我们为我们的模型的解建立了样本外性能保证，提供了关于正则化在控制预测误差中的作用的见解。我们将该方法应用于基于深度视觉转换器(VIT)的图像分类器对随机攻击和敌意攻击的稳健性。具体地说，使用MNIST和CIFAR-10数据集，我们证明了通过采用一种新的随机训练方法，与基线方法相比，测试错误率降低了83.5%，损失降低了91.3%。



## **20. Dynamics-aware Adversarial Attack of Adaptive Neural Networks**

自适应神经网络的动态感知敌意攻击 cs.CV

**SubmitDate**: 2022-10-15    [abs](http://arxiv.org/abs/2210.08159v1) [paper-pdf](http://arxiv.org/pdf/2210.08159v1)

**Authors**: An Tao, Yueqi Duan, Yingqi Wang, Jiwen Lu, Jie Zhou

**Abstract**: In this paper, we investigate the dynamics-aware adversarial attack problem of adaptive neural networks. Most existing adversarial attack algorithms are designed under a basic assumption -- the network architecture is fixed throughout the attack process. However, this assumption does not hold for many recently proposed adaptive neural networks, which adaptively deactivate unnecessary execution units based on inputs to improve computational efficiency. It results in a serious issue of lagged gradient, making the learned attack at the current step ineffective due to the architecture change afterward. To address this issue, we propose a Leaded Gradient Method (LGM) and show the significant effects of the lagged gradient. More specifically, we reformulate the gradients to be aware of the potential dynamic changes of network architectures, so that the learned attack better "leads" the next step than the dynamics-unaware methods when network architecture changes dynamically. Extensive experiments on representative types of adaptive neural networks for both 2D images and 3D point clouds show that our LGM achieves impressive adversarial attack performance compared with the dynamic-unaware attack methods.

摘要: 本文研究了自适应神经网络的动态感知对抗攻击问题。大多数现有的对抗性攻击算法都是在一个基本假设下设计的--网络体系结构在整个攻击过程中是固定的。然而，这一假设对最近提出的许多自适应神经网络并不成立，这些自适应神经网络基于输入自适应地停用不必要的执行单元来提高计算效率。它导致了严重的梯度滞后问题，使得当前步骤的学习攻击由于之后的体系结构变化而无效。为了解决这个问题，我们提出了一种引导梯度法(LGM)，并展示了滞后梯度的显著影响。更具体地说，我们重新制定了梯度，以了解网络体系结构的潜在动态变化，以便在网络体系结构动态变化时，学习到的攻击比不知道动态变化的方法更好地“引导”下一步。在典型的自适应神经网络上对2D图像和3D点云进行的大量实验表明，与动态未知攻击方法相比，我们的LGM具有令人印象深刻的对抗性攻击性能。



## **21. Certified Robustness Against Natural Language Attacks by Causal Intervention**

通过因果干预验证对自然语言攻击的健壮性 cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2205.12331v3) [paper-pdf](http://arxiv.org/pdf/2205.12331v3)

**Authors**: Haiteng Zhao, Chang Ma, Xinshuai Dong, Anh Tuan Luu, Zhi-Hong Deng, Hanwang Zhang

**Abstract**: Deep learning models have achieved great success in many fields, yet they are vulnerable to adversarial examples. This paper follows a causal perspective to look into the adversarial vulnerability and proposes Causal Intervention by Semantic Smoothing (CISS), a novel framework towards robustness against natural language attacks. Instead of merely fitting observational data, CISS learns causal effects p(y|do(x)) by smoothing in the latent semantic space to make robust predictions, which scales to deep architectures and avoids tedious construction of noise customized for specific attacks. CISS is provably robust against word substitution attacks, as well as empirically robust even when perturbations are strengthened by unknown attack algorithms. For example, on YELP, CISS surpasses the runner-up by 6.7% in terms of certified robustness against word substitutions, and achieves 79.4% empirical robustness when syntactic attacks are integrated.

摘要: 深度学习模型在许多领域都取得了很大的成功，但它们很容易受到对手例子的影响。本文从因果关系的角度分析了敌意攻击的脆弱性，提出了通过语义平滑进行因果干预的方法，这是一种新的针对自然语言攻击的健壮性框架。与其仅仅对观测数据进行拟合，CIS通过在潜在语义空间中进行平滑以做出稳健的预测来学习因果效应p(y|do(X))，该预测可扩展到深层体系结构，并避免针对特定攻击定制的乏味的噪声构造。可以证明，该系统对单词替换攻击具有较强的健壮性，即使在未知攻击算法加强了扰动的情况下，也具有较强的经验性。例如，在Yelp上，在对单词替换的验证健壮性方面，CISS超过亚军6.7%，并且在整合句法攻击时获得了79.4%的经验健壮性。



## **22. SealClub: Computer-aided Paper Document Authentication**

SealClub：计算机辅助纸质文档认证 cs.CR

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07884v1) [paper-pdf](http://arxiv.org/pdf/2210.07884v1)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.

摘要: 数字身份验证是一个成熟的领域，提供了一系列具有严格数学保证的解决方案。然而，由于可用性和法律原因，在加密技术不直接适用的情况下，纸质文档仍然被广泛使用。我们提出了一种通过拍摄短视频来使用智能手机对纸质文档进行身份验证的新方法。我们的解决方案结合了加密和图像比较技术，以检测和突出对包含文本和图形的丰富文档的细微语义变化攻击，这些攻击可能不会被人类注意到。我们严格分析了我们的方法，证明了它是安全的，可以抵御能够危害不同系统组件的强大对手。我们还在一组128个纸质文档的视频上对其准确性进行了经验性的测量，其中一半包含微妙的伪造。该算法在平均分析5.13帧(对应于1.28秒的视频)后，准确地发现了所有的伪造(没有虚警)。突出显示的区域足够大，用户可以看到，但也足够小，可以精确定位假货。因此，我们的方法为用户在现实条件下使用传统的智能手机认证纸质文档提供了一种很有前途的方法。



## **23. Pre-trained Adversarial Perturbations**

预先训练的对抗性扰动 cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.03372v2) [paper-pdf](http://arxiv.org/pdf/2210.03372v2)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.

摘要: 近年来，自监督预训练因其在经过微调后在众多下游任务中的优异表现而受到越来越多的关注。然而，众所周知，深度学习模型缺乏对敌意示例的健壮性，这也可能会引发预先训练的模型的安全问题，尽管研究较少。在本文中，我们通过引入预训练对抗扰动(PAP)来深入研究预训练模型的稳健性，PAP是为预训练模型设计的通用扰动，用于在攻击精调模型时保持有效性，而不需要了解下游任务。为此，我们提出了一种低层提升攻击(L4A)的方法，通过提升预训练模型低层神经元的激活来生成有效的PAP。配备了增强的噪音增强策略，L4A在针对微调模型生成更多可转移的PAP方面是有效的。在典型的预训练视觉模型和十个下游任务上的大量实验表明，该方法与现有方法相比，攻击成功率有较大幅度的提高。



## **24. Generative Adversarial Learning for Trusted and Secure Clustering in Industrial Wireless Sensor Networks**

工业无线传感器网络可信安全分簇的生成性对抗性学习 cs.NI

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07707v1) [paper-pdf](http://arxiv.org/pdf/2210.07707v1)

**Authors**: Liu Yang, Simon X. Yang, Yun Li, Yinzhi Lu, Tan Guo

**Abstract**: Traditional machine learning techniques have been widely used to establish the trust management systems. However, the scale of training dataset can significantly affect the security performances of the systems, while it is a great challenge to detect malicious nodes due to the absence of labeled data regarding novel attacks. To address this issue, this paper presents a generative adversarial network (GAN) based trust management mechanism for Industrial Wireless Sensor Networks (IWSNs). First, type-2 fuzzy logic is adopted to evaluate the reputation of sensor nodes while alleviating the uncertainty problem. Then, trust vectors are collected to train a GAN-based codec structure, which is used for further malicious node detection. Moreover, to avoid normal nodes being isolated from the network permanently due to error detections, a GAN-based trust redemption model is constructed to enhance the resilience of trust management. Based on the latest detection results, a trust model update method is developed to adapt to the dynamic industrial environment. The proposed trust management mechanism is finally applied to secure clustering for reliable and real-time data transmission, and simulation results show that it achieves a high detection rate up to 96%, as well as a low false positive rate below 8%.

摘要: 传统的机器学习技术已被广泛应用于建立信任管理系统。然而，训练数据集的规模会显著影响系统的安全性能，同时由于缺乏关于新攻击的标记数据，检测恶意节点是一个巨大的挑战。针对这一问题，提出了一种基于产生式对抗网络的工业无线传感器网络信任管理机制。首先，在缓解不确定性问题的同时，采用二型模糊逻辑对传感器节点的信誉度进行评估。然后，收集信任向量训练基于GAN的编解码器结构，用于进一步的恶意节点检测。此外，为了避免正常节点因错误检测而与网络永久隔离，构建了基于GAN的信任赎回模型，增强了信任管理的弹性。基于最新的检测结果，提出了一种适应动态工业环境的信任模型更新方法。最后将提出的信任管理机制应用于安全分簇，实现了可靠、实时的数据传输，仿真结果表明，该信任管理机制的检测率高达96%，误检率低于8%。



## **25. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

对抗性像素恢复作为可转移扰动的借口任务 cs.CV

Accepted at BMVC'22 (Oral)

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2207.08803v3) [paper-pdf](http://arxiv.org/pdf/2207.08803v3)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstract**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR

摘要: 可转移对抗性攻击从预先训练的代理模型和已知标签空间中优化对手，以愚弄未知的黑盒模型。因此，这些攻击受到有效代理模型可用性的限制。在这项工作中，我们放松了这一假设，提出了对抗性像素复原作为一种自我监督的替代方案，在没有标签和数据样本的情况下，从零开始训练一个有效的代理模型。我们的训练方法基于最小-最大方案，该方案减少了通过对抗性目标的过度拟合，从而优化了更具普适性的代理模型。我们提出的攻击是对抗性像素恢复的补充，并且独立于任何特定于任务的目标，因为它可以以自我监督的方式发起。我们成功地展示了我们的视觉变形方法以及卷积神经网络方法在分类、目标检测和视频分割任务中的对抗性可转移性。我们的训练方法将基线无监督训练方法在ImageNet Val上的可转移性提高了16.4%。准备好了。我们的代码和预先培训的代孕模型可在以下网址获得：https://github.com/HashmatShadab/APR



## **26. When Adversarial Training Meets Vision Transformers: Recipes from Training to Architecture**

当对抗训练遇到视觉变形者：从训练到建筑的秘诀 cs.CV

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07540v1) [paper-pdf](http://arxiv.org/pdf/2210.07540v1)

**Authors**: Yichuan Mo, Dongxian Wu, Yifei Wang, Yiwen Guo, Yisen Wang

**Abstract**: Vision Transformers (ViTs) have recently achieved competitive performance in broad vision tasks. Unfortunately, on popular threat models, naturally trained ViTs are shown to provide no more adversarial robustness than convolutional neural networks (CNNs). Adversarial training is still required for ViTs to defend against such adversarial attacks. In this paper, we provide the first and comprehensive study on the adversarial training recipe of ViTs via extensive evaluation of various training techniques across benchmark datasets. We find that pre-training and SGD optimizer are necessary for ViTs' adversarial training. Further considering ViT as a new type of model architecture, we investigate its adversarial robustness from the perspective of its unique architectural components. We find, when randomly masking gradients from some attention blocks or masking perturbations on some patches during adversarial training, the adversarial robustness of ViTs can be remarkably improved, which may potentially open up a line of work to explore the architectural information inside the newly designed models like ViTs. Our code is available at https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers.

摘要: 视觉变形器(VITS)最近在广泛的视觉任务中取得了具有竞争力的表现。不幸的是，在流行的威胁模型上，自然训练的VITS并不比卷积神经网络(CNN)提供更多的对抗健壮性。VITS仍然需要进行对抗性训练，以防御这种对抗性攻击。在本文中，我们通过对基准数据集上各种训练技术的广泛评估，首次对VITS的对抗性训练配方进行了全面的研究。我们发现，对于VITS的对抗性训练，预训练和SGD优化器是必要的。进一步考虑到VIT是一种新型的模型体系结构，我们从其独特的体系结构组件的角度研究了它的对抗性健壮性。我们发现，当在对抗性训练中随机掩蔽某些注意块中的梯度或掩蔽某些斑块上的扰动时，VITS的对抗性稳健性可以显著提高，这可能会为探索VITS等新设计模型中的体系结构信息开辟一条道路。我们的代码可以在https://github.com/mo666666/When-Adversarial-Training-Meets-Vision-Transformers.上找到



## **27. Characterizing the Influence of Graph Elements**

刻画图元素的影响 cs.LG

**SubmitDate**: 2022-10-14    [abs](http://arxiv.org/abs/2210.07441v1) [paper-pdf](http://arxiv.org/pdf/2210.07441v1)

**Authors**: Zizhang Chen, Peizhao Li, Hongfu Liu, Pengyu Hong

**Abstract**: Influence function, a method from robust statistics, measures the changes of model parameters or some functions about model parameters concerning the removal or modification of training instances. It is an efficient and useful post-hoc method for studying the interpretability of machine learning models without the need for expensive model re-training. Recently, graph convolution networks (GCNs), which operate on graph data, have attracted a great deal of attention. However, there is no preceding research on the influence functions of GCNs to shed light on the effects of removing training nodes/edges from an input graph. Since the nodes/edges in a graph are interdependent in GCNs, it is challenging to derive influence functions for GCNs. To fill this gap, we started with the simple graph convolution (SGC) model that operates on an attributed graph and formulated an influence function to approximate the changes in model parameters when a node or an edge is removed from an attributed graph. Moreover, we theoretically analyzed the error bound of the estimated influence of removing an edge. We experimentally validated the accuracy and effectiveness of our influence estimation function. In addition, we showed that the influence function of an SGC model could be used to estimate the impact of removing training nodes/edges on the test performance of the SGC without re-training the model. Finally, we demonstrated how to use influence functions to guide the adversarial attacks on GCNs effectively.

摘要: 影响函数是稳健统计中的一种方法，它度量模型参数的变化或与训练实例的删除或修改有关的模型参数的某些函数。对于研究机器学习模型的可解释性，它是一种有效和有用的后处理方法，而不需要昂贵的模型重新训练。近年来，处理图形数据的图形卷积网络(GCNS)引起了人们的极大关注。然而，以前还没有关于GCNS影响函数的研究来阐明从输入图中移除训练节点/边的效果。由于图中的节点/边在GCNS中是相互依赖的，因此推导GCNS的影响函数是一项具有挑战性的工作。为了填补这一空白，我们从对属性图进行操作的简单图卷积(SGC)模型开始，并建立了一个影响函数来逼近当从属性图中移除节点或边时模型参数的变化。此外，我们还从理论上分析了去除边缘对估计影响的误差界。我们通过实验验证了我们的影响估计函数的准确性和有效性。此外，我们还证明了SGC模型的影响函数可以用来估计去除训练节点/边对SGC测试性能的影响，而不需要重新训练模型。最后，我们演示了如何使用影响函数来有效地指导对GCNS的对抗性攻击。



## **28. Demystifying Self-supervised Trojan Attacks**

揭开自我监督特洛伊木马攻击的神秘面纱 cs.CR

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.07346v1) [paper-pdf](http://arxiv.org/pdf/2210.07346v1)

**Authors**: Changjiang Li, Ren Pang, Zhaohan Xi, Tianyu Du, Shouling Ji, Yuan Yao, Ting Wang

**Abstract**: As an emerging machine learning paradigm, self-supervised learning (SSL) is able to learn high-quality representations for complex data without data labels. Prior work shows that, besides obviating the reliance on labeling, SSL also benefits adversarial robustness by making it more challenging for the adversary to manipulate model prediction. However, whether this robustness benefit generalizes to other types of attacks remains an open question.   We explore this question in the context of trojan attacks by showing that SSL is comparably vulnerable as supervised learning to trojan attacks. Specifically, we design and evaluate CTRL, an extremely simple self-supervised trojan attack. By polluting a tiny fraction of training data (less than 1%) with indistinguishable poisoning samples, CTRL causes any trigger-embedded input to be misclassified to the adversary's desired class with a high probability (over 99%) at inference. More importantly, through the lens of CTRL, we study the mechanisms underlying self-supervised trojan attacks. With both empirical and analytical evidence, we reveal that the representation invariance property of SSL, which benefits adversarial robustness, may also be the very reason making SSL highly vulnerable to trojan attacks. We further discuss the fundamental challenges to defending against self-supervised trojan attacks, pointing to promising directions for future research.

摘要: 作为一种新兴的机器学习范式，自监督学习能够在没有数据标签的情况下学习复杂数据的高质量表示。先前的工作表明，除了避免对标记的依赖外，SSL还通过使对手更具挑战性来操纵模型预测，从而有利于对手的稳健性。然而，这种健壮性优势是否适用于其他类型的攻击仍是一个悬而未决的问题。我们在特洛伊木马攻击的背景下探索这个问题，表明SSL与监督学习相比容易受到特洛伊木马攻击。具体地说，我们设计并评估了CTRL，一种极其简单的自我监督木马攻击。通过用难以区分的中毒样本污染一小部分训练数据(不到1%)，CTRL导致任何嵌入触发器的输入在推理时被错误分类到对手想要的类别的概率很高(超过99%)。更重要的是，通过CTRL的镜头，我们研究了自我监督特洛伊木马攻击的机制。通过经验证据和分析证据，我们揭示了SSL的表示不变性，这一特性有利于攻击的健壮性，也可能是使SSL高度容易受到木马攻击的原因。我们进一步讨论了防御自我监督特洛伊木马攻击的基本挑战，指出了未来研究的方向。



## **29. Autoregressive Perturbations for Data Poisoning**

数据中毒的自回归摄动 cs.LG

Accepted to NeurIPS 2022. Code available at  https://github.com/psandovalsegura/autoregressive-poisoning

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2206.03693v3) [paper-pdf](http://arxiv.org/pdf/2206.03693v3)

**Authors**: Pedro Sandoval-Segura, Vasu Singla, Jonas Geiping, Micah Goldblum, Tom Goldstein, David W. Jacobs

**Abstract**: The prevalence of data scraping from social media as a means to obtain datasets has led to growing concerns regarding unauthorized use of data. Data poisoning attacks have been proposed as a bulwark against scraping, as they make data "unlearnable" by adding small, imperceptible perturbations. Unfortunately, existing methods require knowledge of both the target architecture and the complete dataset so that a surrogate network can be trained, the parameters of which are used to generate the attack. In this work, we introduce autoregressive (AR) poisoning, a method that can generate poisoned data without access to the broader dataset. The proposed AR perturbations are generic, can be applied across different datasets, and can poison different architectures. Compared to existing unlearnable methods, our AR poisons are more resistant against common defenses such as adversarial training and strong data augmentations. Our analysis further provides insight into what makes an effective data poison.

摘要: 从社交媒体上窃取数据作为获取数据集的一种手段的盛行，导致对未经授权使用数据的担忧日益加剧。数据中毒攻击被认为是防止抓取的堡垒，因为它们通过添加微小的、不可察觉的干扰而使数据“无法学习”。不幸的是，现有方法需要目标体系结构和完整数据集的知识，以便可以训练代理网络，其参数用于生成攻击。在这项工作中，我们引入了自回归(AR)中毒，这是一种在不访问更广泛的数据集的情况下生成有毒数据的方法。所提出的AR扰动是通用的，可以应用于不同的数据集，并且可能毒害不同的体系结构。与现有的无法学习的方法相比，我们的AR毒药对常见的防御措施更具抵抗力，例如对抗性训练和强大的数据增强。我们的分析进一步提供了对有效数据毒害的原因的洞察。



## **30. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

Pikachu：通过使用Taproot检查点进入比特币PoW来保护PoS区块链免受远程攻击 cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2208.05408v2) [paper-pdf](http://arxiv.org/pdf/2208.05408v2)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstract**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.

摘要: 基于可重用资源的区块链系统，如风险证明(POS)，提供的安全保证比基于工作证明的系统更弱。具体地说，它们容易受到远程攻击，在远程攻击中，对手可以破坏之前的参与者，以便重写链的完整历史。为了防止这种对PoS链的攻击，我们提出了一种协议，将PoS链的状态检查到工作证明区块链，如比特币。因此，我们的检查点协议不依赖于任何中央机构。我们的工作使用Schnorr签名并利用比特币最近的Taproot升级，使我们能够创建恒定大小的检查点交易。我们为协议的安全性进行了论证，并给出了一个在比特币测试网上进行测试的开源实现。



## **31. AccelAT: A Framework for Accelerating the Adversarial Training of Deep Neural Networks through Accuracy Gradient**

AccelAT：一种通过精度梯度加速深度神经网络对抗性训练的框架 cs.LG

12 pages

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06888v1) [paper-pdf](http://arxiv.org/pdf/2210.06888v1)

**Authors**: Farzad Nikfam, Alberto Marchisio, Maurizio Martina, Muhammad Shafique

**Abstract**: Adversarial training is exploited to develop a robust Deep Neural Network (DNN) model against the malicious altered data. These attacks may have catastrophic effects on DNN models but are indistinguishable for a human being. For example, an external attack can modify an image adding noises invisible for a human eye, but a DNN model misclassified the image. A key objective for developing robust DNN models is to use a learning algorithm that is fast but can also give model that is robust against different types of adversarial attacks. Especially for adversarial training, enormously long training times are needed for obtaining high accuracy under many different types of adversarial samples generated using different adversarial attack techniques.   This paper aims at accelerating the adversarial training to enable fast development of robust DNN models against adversarial attacks. The general method for improving the training performance is the hyperparameters fine-tuning, where the learning rate is one of the most crucial hyperparameters. By modifying its shape (the value over time) and value during the training, we can obtain a model robust to adversarial attacks faster than standard training.   First, we conduct experiments on two different datasets (CIFAR10, CIFAR100), exploring various techniques. Then, this analysis is leveraged to develop a novel fast training methodology, AccelAT, which automatically adjusts the learning rate for different epochs based on the accuracy gradient. The experiments show comparable results with the related works, and in several experiments, the adversarial training of DNNs using our AccelAT framework is conducted up to 2 times faster than the existing techniques. Thus, our findings boost the speed of adversarial training in an era in which security and performance are fundamental optimization objectives in DNN-based applications.

摘要: 利用对抗性训练建立了一种针对恶意篡改数据的稳健深度神经网络(DNN)模型。这些攻击可能会对DNN模型产生灾难性影响，但对人类来说是无法区分的。例如，外部攻击可以修改图像，添加人眼看不见的噪声，但DNN模型错误地分类了图像。开发健壮DNN模型的一个关键目标是使用一种快速的学习算法，并且能够给出对不同类型的对手攻击具有健壮性的模型。特别是对于对抗性训练，在使用不同的对抗性攻击技术生成的许多不同类型的对抗性样本下，需要非常长的训练时间才能获得高的准确率。本文的目的是加速对抗性训练，以便快速开发出抵抗对抗性攻击的稳健DNN模型。提高训练性能的一般方法是超参数微调，其中学习率是最关键的超参数之一。通过在训练过程中修改其形状(随时间变化的值)和值，我们可以得到一个比标准训练更快地抗击对手攻击的模型。首先，我们在两个不同的数据集(CIFAR10，CIFAR100)上进行了实验，探索了各种技术。然后，利用这一分析来开发一种新的快速训练方法AccelAT，该方法根据精度梯度自动调整不同历元的学习率。实验结果与相关工作具有可比性，在多个实验中，使用AccelAT框架对DNN进行对抗性训练的速度比现有技术快2倍。因此，在安全性和性能是基于DNN的应用程序的基本优化目标的时代，我们的发现提高了对抗性训练的速度。



## **32. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

ADV-ATTRIBUTE：对人脸识别的隐蔽且可转移的敌意攻击 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06871v1) [paper-pdf](http://arxiv.org/pdf/2210.06871v1)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.

摘要: 深度学习模型在处理对抗性攻击时显示出了它们的脆弱性。现有的攻击几乎是在低层实例上执行的，例如像素和超像素，很少利用语义线索。对于人脸识别攻击，现有的方法通常会产生像素上的l_p范数扰动，导致攻击可传递性低，对去噪防御模型的脆弱性高。在这项工作中，我们不是对低层像素进行扰动，而是通过对高层语义的扰动来产生攻击，以提高攻击的可转移性。具体地说，设计了一个统一的灵活框架--对抗性属性(ADV-ATTRIBUTE)，用于产生对人脸识别的隐蔽性和可转移性攻击，该框架根据人脸识别特征与目标的差异指导生成对抗性噪声并将其添加到不同的属性中。此外，引入了重要性感知的属性选择和多目标优化策略，进一步保证了隐蔽性和攻击力的平衡。在FFHQ和CelebA-HQ数据集上的大量实验表明，所提出的ADV属性方法达到了最先进的攻击成功率，同时对最近的攻击方法保持了更好的视觉效果。



## **33. Federated Learning for Tabular Data: Exploring Potential Risk to Privacy**

表格数据的联合学习：探索隐私的潜在风险 cs.CR

In the proceedings of The 33rd IEEE International Symposium on  Software Reliability Engineering (ISSRE), November 2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06856v1) [paper-pdf](http://arxiv.org/pdf/2210.06856v1)

**Authors**: Han Wu, Zilong Zhao, Lydia Y. Chen, Aad van Moorsel

**Abstract**: Federated Learning (FL) has emerged as a potentially powerful privacy-preserving machine learning methodology, since it avoids exchanging data between participants, but instead exchanges model parameters. FL has traditionally been applied to image, voice and similar data, but recently it has started to draw attention from domains including financial services where the data is predominantly tabular. However, the work on tabular data has not yet considered potential attacks, in particular attacks using Generative Adversarial Networks (GANs), which have been successfully applied to FL for non-tabular data. This paper is the first to explore leakage of private data in Federated Learning systems that process tabular data. We design a Generative Adversarial Networks (GANs)-based attack model which can be deployed on a malicious client to reconstruct data and its properties from other participants. As a side-effect of considering tabular data, we are able to statistically assess the efficacy of the attack (without relying on human observation such as done for FL for images). We implement our attack model in a recently developed generic FL software framework for tabular data processing. The experimental results demonstrate the effectiveness of the proposed attack model, thus suggesting that further research is required to counter GAN-based privacy attacks.

摘要: 联合学习(FL)已经成为一种潜在的强大的隐私保护机器学习方法，因为它避免了参与者之间交换数据，而是交换模型参数。传统上，FL被应用于图像、语音和类似数据，但最近它开始吸引包括金融服务在内的领域的注意，这些领域的数据主要是表格。然而，关于表格数据的工作还没有考虑到潜在的攻击，特别是使用生成性对抗网络(GANS)的攻击，这些攻击已经成功地应用于非表格数据的FL。本文首次探讨了联邦学习系统中处理表格数据的私有数据泄漏问题。我们设计了一个基于生成性对抗网络(GANS)的攻击模型，该模型可以部署在恶意客户端上，以重构来自其他参与者的数据及其属性。考虑表格数据的一个副作用是，我们能够在统计上评估攻击的效果(不依赖于人的观察，如对图像的FL所做的)。我们在最近开发的用于表格数据处理的通用FL软件框架中实现了我们的攻击模型。实验结果证明了该攻击模型的有效性，表明需要对基于GAN的隐私攻击进行进一步的研究。



## **34. Observed Adversaries in Deep Reinforcement Learning**

深度强化学习中观察到的对手 cs.LG

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06787v1) [paper-pdf](http://arxiv.org/pdf/2210.06787v1)

**Authors**: Eugene Lim, Harold Soh

**Abstract**: In this work, we point out the problem of observed adversaries for deep policies. Specifically, recent work has shown that deep reinforcement learning is susceptible to adversarial attacks where an observed adversary acts under environmental constraints to invoke natural but adversarial observations. This setting is particularly relevant for HRI since HRI-related robots are expected to perform their tasks around and with other agents. In this work, we demonstrate that this effect persists even with low-dimensional observations. We further show that these adversarial attacks transfer across victims, which potentially allows malicious attackers to train an adversary without access to the target victim.

摘要: 在这项工作中，我们指出了深度政策的观察对手的问题。具体地说，最近的工作表明，深度强化学习容易受到对抗性攻击，即被观察到的对手在环境约束下采取行动，援引自然但对抗性的观察。这一设置与HRI特别相关，因为与HRI相关的机器人应该在其他代理周围和与其他代理一起执行任务。在这项工作中，我们证明了即使在低维观测中，这种效应仍然存在。我们进一步表明，这些对抗性攻击在受害者之间转移，这可能允许恶意攻击者在没有访问目标受害者的情况下训练对手。



## **35. COLLIDER: A Robust Training Framework for Backdoor Data**

Collider：一个健壮的后门数据训练框架 cs.LG

Accepted to the 16th Asian Conference on Computer Vision (ACCV 2022)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06704v1) [paper-pdf](http://arxiv.org/pdf/2210.06704v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Deep neural network (DNN) classifiers are vulnerable to backdoor attacks. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect malicious backdoored DNNs. However, a robust, end-to-end training approach, like adversarial training, is yet to be discovered for backdoor poisoned data. In this paper, we take the first step toward such methods by developing a robust training framework, COLLIDER, that selects the most prominent samples by exploiting the underlying geometric structures of the data. Specifically, we effectively filter out candidate poisoned data at each training epoch by solving a geometrical coreset selection objective. We first argue how clean data samples exhibit (1) gradients similar to the clean majority of data and (2) low local intrinsic dimensionality (LID). Based on these criteria, we define a novel coreset selection objective to find such samples, which are used for training a DNN. We show the effectiveness of the proposed method for robust training of DNNs on various poisoned datasets, reducing the backdoor success rate significantly.

摘要: 深度神经网络(DNN)分类器容易受到后门攻击。对手通过安装触发器来毒化此类攻击中的一些训练数据。这样做的目的是让经过训练的DNN在触发器被激活时输出攻击者想要的类，同时像往常一样执行干净的数据。最近提出了各种方法来检测恶意回溯的DNN。然而，一种强大的端到端培训方法，如对抗性培训，尚未发现针对后门有毒数据的方法。在本文中，我们通过开发一个健壮的训练框架Collider来朝着这种方法迈出第一步，该框架通过利用数据的基本几何结构来选择最突出的样本。具体地说，我们通过求解几何核心重置选择目标，有效地过滤出每个训练时期的候选有毒数据。我们首先讨论干净的数据样本如何表现出(1)类似于干净的大多数数据的梯度和(2)低的局部固有维度(LID)。基于这些准则，我们定义了一种新的核心选择目标来寻找用于训练DNN的样本。我们在不同的有毒数据集上展示了所提出的方法用于DNN稳健训练的有效性，显著降低了后门成功率。



## **36. A Game Theoretical vulnerability analysis of Adversarial Attack**

对抗性攻击的博弈论脆弱性分析 cs.GT

Accepted in 17th International Symposium on Visual Computing,2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06670v1) [paper-pdf](http://arxiv.org/pdf/2210.06670v1)

**Authors**: Khondker Fariha Hossain, Alireza Tavakkoli, Shamik Sengupta

**Abstract**: In recent times deep learning has been widely used for automating various security tasks in Cyber Domains. However, adversaries manipulate data in many situations and diminish the deployed deep learning model's accuracy. One notable example is fooling CAPTCHA data to access the CAPTCHA-based Classifier leading to the critical system being vulnerable to cybersecurity attacks. To alleviate this, we propose a computational framework of game theory to analyze the CAPTCHA-based Classifier's vulnerability, strategy, and outcomes by forming a simultaneous two-player game. We apply the Fast Gradient Symbol Method (FGSM) and One Pixel Attack on CAPTCHA Data to imitate real-life scenarios of possible cyber-attack. Subsequently, to interpret this scenario from a Game theoretical perspective, we represent the interaction in the Stackelberg Game in Kuhn tree to study players' possible behaviors and actions by applying our Classifier's actual predicted values. Thus, we interpret potential attacks in deep learning applications while representing viable defense strategies in the game theory prospect.

摘要: 近年来，深度学习被广泛用于自动化网络领域中的各种安全任务。然而，敌手在许多情况下操纵数据，降低了部署的深度学习模型的准确性。一个值得注意的例子是欺骗验证码数据访问基于验证码的分类器，导致关键系统容易受到网络安全攻击。为了缓解这一问题，我们提出了一个博弈论的计算框架，通过形成一个同时的两人博弈来分析基于验证码的分类器的脆弱性、策略和结果。我们对验证码数据应用快速梯度符号方法(FGSM)和单像素攻击来模拟可能的网络攻击的真实场景。随后，为了从博弈论的角度解释这一场景，我们将Stackelberg博弈中的交互表示在Kuhn树中，通过应用我们的分类器的实际预测值来研究玩家可能的行为和行动。因此，我们解释了深度学习应用中的潜在攻击，同时表示了博弈论前景中可行的防御策略。



## **37. Understanding Impacts of Task Similarity on Backdoor Attack and Detection**

了解任务相似度对后门攻击和检测的影响 cs.CR

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06509v1) [paper-pdf](http://arxiv.org/pdf/2210.06509v1)

**Authors**: Di Tang, Rui Zhu, XiaoFeng Wang, Haixu Tang, Yi Chen

**Abstract**: With extensive studies on backdoor attack and detection, still fundamental questions are left unanswered regarding the limits in the adversary's capability to attack and the defender's capability to detect. We believe that answers to these questions can be found through an in-depth understanding of the relations between the primary task that a benign model is supposed to accomplish and the backdoor task that a backdoored model actually performs. For this purpose, we leverage similarity metrics in multi-task learning to formally define the backdoor distance (similarity) between the primary task and the backdoor task, and analyze existing stealthy backdoor attacks, revealing that most of them fail to effectively reduce the backdoor distance and even for those that do, still much room is left to further improve their stealthiness. So we further design a new method, called TSA attack, to automatically generate a backdoor model under a given distance constraint, and demonstrate that our new attack indeed outperforms existing attacks, making a step closer to understanding the attacker's limits. Most importantly, we provide both theoretic results and experimental evidence on various datasets for the positive correlation between the backdoor distance and backdoor detectability, demonstrating that indeed our task similarity analysis help us better understand backdoor risks and has the potential to identify more effective mitigations.

摘要: 随着对后门攻击和检测的广泛研究，仍然没有回答关于对手攻击能力和防御者检测能力的限制的根本问题。我们相信，通过深入理解良性模型应该完成的主要任务和后门模型实际执行的后门任务之间的关系，可以找到这些问题的答案。为此，我们利用多任务学习中的相似性度量来形式化地定义主任务和后门任务之间的后门距离(相似性)，并分析了现有的隐身后门攻击，发现它们中的大多数都无法有效地减少后门距离，即使对于那些能够有效降低后门距离的攻击，仍然有很大的空间来进一步提高它们的隐蔽性。因此，我们进一步设计了一种新的方法，称为TSA攻击，在给定的距离约束下自动生成一个后门模型，并证明了我们的新攻击确实比现有的攻击性能更好，使我们更接近了解攻击者的限制。最重要的是，我们提供了理论结果和在各种数据集上的实验证据，证明了后门距离和后门可检测性之间的正相关关系，表明我们的任务相似性分析确实有助于我们更好地理解后门风险，并有可能识别更有效的缓解措施。



## **38. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

关于深度神经网络攻击域外不确定性估计问题 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02191v2) [paper-pdf](http://arxiv.org/pdf/2210.02191v2)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.

摘要: 在许多具有真实世界后果的应用中，为人工智能决策系统做出的预测开发可靠的不确定性估计是至关重要的。针对不确定性估计的目标，人们提出了各种基于深度神经网络(DNN)的不确定性估计算法。然而，这些算法返回的不确定性的稳健性还没有得到系统的探讨。在这项工作中，为了提高研究界对稳健不确定性估计的认识，我们证明了最新的不确定性估计算法在我们提出的对抗性攻击下可能会灾难性地失败，尽管它们在不确定性估计方面的表现令人印象深刻。特别是，我们的目标是攻击域外的不确定性估计：在我们的攻击下，不确定性模型将被愚弄，以对域外数据做出高度自信的预测，而他们最初会拒绝这些预测。在不同基准图像数据集上的大量实验结果表明，最新方法估计的不确定性很容易被我们的攻击所破坏。



## **39. On Optimal Learning Under Targeted Data Poisoning**

目标数据中毒下的最优学习问题研究 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02713v2) [paper-pdf](http://arxiv.org/pdf/2210.02713v2)

**Authors**: Steve Hanneke, Amin Karbasi, Mohammad Mahmoody, Idan Mehalel, Shay Moran

**Abstract**: Consider the task of learning a hypothesis class $\mathcal{H}$ in the presence of an adversary that can replace up to an $\eta$ fraction of the examples in the training set with arbitrary adversarial examples. The adversary aims to fail the learner on a particular target test point $x$ which is known to the adversary but not to the learner. In this work we aim to characterize the smallest achievable error $\epsilon=\epsilon(\eta)$ by the learner in the presence of such an adversary in both realizable and agnostic settings. We fully achieve this in the realizable setting, proving that $\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $\mathtt{VC}(\mathcal{H})$ is the VC dimension of $\mathcal{H}$. Remarkably, we show that the upper bound can be attained by a deterministic learner. In the agnostic setting we reveal a more elaborate landscape: we devise a deterministic learner with a multiplicative regret guarantee of $\epsilon \leq C\cdot\mathtt{OPT} + O(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $C > 1$ is a universal numerical constant. We complement this by showing that for any deterministic learner there is an attack which worsens its error to at least $2\cdot \mathtt{OPT}$. This implies that a multiplicative deterioration in the regret is unavoidable in this case. Finally, the algorithms we develop for achieving the optimal rates are inherently improper. Nevertheless, we show that for a variety of natural concept classes, such as linear classifiers, it is possible to retain the dependence $\epsilon=\Theta_{\mathcal{H}}(\eta)$ by a proper algorithm in the realizable setting. Here $\Theta_{\mathcal{H}}$ conceals a polynomial dependence on $\mathtt{VC}(\mathcal{H})$.

摘要: 考虑在对手在场的情况下学习假设类$\mathcal{H}$的任务，该对手可以用任意的对抗性例子替换训练集中的$\eta$分数的例子。对手的目标是在对手知道但学习者不知道的特定目标测试点$x$上让学习者不及格。在这项工作中，我们的目标是刻画在可实现和不可知的情况下，学习者在这样的对手存在的情况下所能达到的最小误差。我们在可实现的设置下完全实现了这一点，证明了$\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot\eta)$，其中$\mathtt{VC}(\mathcal{H})$是$\mathcal{H}$的VC维。值得注意的是，我们证明了这一上界可以由确定性学习者获得。在不可知论的背景下，我们展示了一个更精细的场景：我们设计了一个确定性学习者，其乘性后悔保证为$\epsilon\leq C\cdot\mathtt{opt}+O(\mathtt{VC}(\mathcal{H})\cdot\eta)$，其中$C>1$是通用数值常量。我们的补充是，对于任何确定性学习者，都存在将其错误恶化到至少$2\cdot\mathtt{opt}$的攻击。这意味着，在这种情况下，遗憾的成倍恶化是不可避免的。最后，我们开发的用于实现最优速率的算法本质上是不正确的。然而，我们证明了对于各种自然概念类，例如线性分类器，在可实现的设置下，通过适当的算法可以保持依赖关系$\epsilon=\tha_{\mathcal{H}}(\eta)$。这里，$theta_{\mathcal{H}}$隐藏了对$\mathtt{VC}(\mathcal{H})$的多项式依赖关系。



## **40. Alleviating Adversarial Attacks on Variational Autoencoders with MCMC**

利用MCMC减轻对变分自动编码器的敌意攻击 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2203.09940v2) [paper-pdf](http://arxiv.org/pdf/2203.09940v2)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstract**: Variational autoencoders (VAEs) are latent variable models that can generate complex objects and provide meaningful latent representations. Moreover, they could be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attack construction proposed previously and present a solution to alleviate the effect of these attacks. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step that we motivate with a theoretical analysis. Thus, we do not incorporate any extra costs during training, and the performance on non-attacked inputs is not decreased. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, $\beta$-TCVAE), and show that our approach consistently improves the model robustness to adversarial attacks.

摘要: 变分自动编码器(VAE)是一种潜在变量模型，可以生成复杂的对象并提供有意义的潜在表示。此外，它们还可以进一步用于分类等下游任务。正如以前的工作所表明的，人们可以很容易地愚弄VAE，为视觉上稍有修改的输入产生意想不到的潜在表示和重建。在这里，我们检查了几个以前提出的对抗性攻击构造的目标函数，并提出了一个解决方案来减轻这些攻击的影响。我们的方法在推理步骤中使用了马尔科夫链蒙特卡罗(MCMC)技术，并进行了理论分析。因此，我们在训练期间不会纳入任何额外的成本，并且非攻击输入的性能不会降低。我们在各种数据集(MNIST、Fashion MNIST、Color MNIST、CelebA)和VAE配置($\beta$-VAE、NVAE、$\beta$-TCVAE)上验证了我们的方法，并表明我们的方法持续提高了模型对对手攻击的稳健性。



## **41. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

6 pages, 4 figures, 3 tables

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06284v1) [paper-pdf](http://arxiv.org/pdf/2210.06284v1)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **42. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

半监督对抗性鲁棒PAC学习性的一个刻画 cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2202.05420v2) [paper-pdf](http://arxiv.org/pdf/2202.05420v2)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.

摘要: 我们研究了在半监督PAC模型中学习对抗性稳健预测器来测试时间攻击的问题。我们解决了需要多少已标记和未标记的示例才能确保学习的问题。我们证明，有足够的未标记数据(完全监督方法所需的标记样本的大小)，标记样本的复杂度可以比以前的工作任意小，并且明显地被不同的复杂性度量所刻画。我们证明了这一样本复杂性的上下界几乎一致。这表明，即使在最坏情况下无分布的模型中，半监督稳健学习也有显著的好处，并在监督和半监督标签复杂性之间建立了差距，这在标准的非稳健PAC学习中是不存在的。



## **43. Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity**

双重泡沫、辛劳和麻烦：通过传递性增强认证的健壮性 cs.LG

Accepted for Neurips`22, 19 pages, 14 figures, for associated code  see https://github.com/andrew-cullen/DoubleBubble

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06077v1) [paper-pdf](http://arxiv.org/pdf/2210.06077v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In response to subtle adversarial examples flipping classifications of neural network models, recent research has promoted certified robustness as a solution. There, invariance of predictions to all norm-bounded attacks is achieved through randomised smoothing of network inputs. Today's state-of-the-art certifications make optimal use of the class output scores at the input instance under test: no better radius of certification (under the $L_2$ norm) is possible given only these score. However, it is an open question as to whether such lower bounds can be improved using local information around the instance under test. In this work, we demonstrate how today's "optimal" certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than $80\%$ of Tiny-Imagenet instances, yielding an on average $5 \%$ increase in the associated certification. When incorporating training time processes that enhance the certified radius, our technique shows even more promising results, with a uniform $4$ percentage point increase in the achieved certified radius.

摘要: 为了应对微妙的敌意例子颠覆神经网络模型的分类，最近的研究已经将经过认证的稳健性作为解决方案。通过对网络输入的随机平滑，实现了对所有范数有界攻击的预测不变性。当今最先进的认证最好地利用了测试中输入实例的类输出分数：如果只给出这些分数，就没有更好的认证半径(在$L_2$范数下)。然而，是否可以使用测试实例周围的本地信息来提高这种下限，这是一个悬而未决的问题。在这项工作中，我们演示了如何通过利用证书的传递性和输入空间的几何来改进当今的“最佳”证书，从而产生我们所称的几何信息的认证健壮性。通过考虑到一组证书的边界上的点的最小距离，该方法改进了超过$80$的微型Imagenet实例的证书，导致相关证书的平均增加$5\$。当加入了提高认证半径的训练时间过程时，我们的技术显示出更有希望的结果，所获得的认证半径统一增加了$4$百分点。



## **44. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

SA：抗剪裁和自拼接的合成语音检测滑动攻击 cs.SD

Updated description and formula

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2208.13066v2) [paper-pdf](http://arxiv.org/pdf/2208.13066v2)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstract**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.

摘要: 深度神经网络很容易受到敌意例子的影响，这些例子用无法察觉的扰动误导模型。在音频方面，虽然对抗性例子在白盒和黑盒设置上取得了令人难以置信的攻击成功率，但大多数现有的对抗性攻击都受到输入长度的限制。一个更实际的场景是，对抗性的例子必须被剪裁或自我拼接，并输入到黑盒模型中。因此，有必要探索如何在不同的输入长度设置下提高可转移性。在本文中，我们以合成语音检测任务为例，考虑了两个具有代表性的SOTA模型。通过分析剪裁或自剪接后将样本送入模型获得的梯度，我们观察到相同样本值的片段在不同模型中的梯度是相似的。受此启发，我们提出了一种新的对抗性攻击方法--滑动攻击。具体地说，我们使每个采样点知道不同位置的梯度，这可以模拟对抗性例子被输入到具有不同输入长度的黑盒模型的情况。因此，我们不是在梯度计算的每次迭代中直接使用当前梯度，而是经历以下三个步骤。首先，我们使用滑动窗口提取不同长度的子段。然后，我们使用来自相邻域的数据来增强子分段。最后，我们将子片段输入到不同的模型中，以获得聚合梯度来更新对抗性实例。实验结果表明，该方法能够显著提高截取或自拼接后的对抗性样本的可转移性。此外，我们的方法还可以增强基于不同特征的模型之间的可移植性。



## **45. Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation**

利用反向对抗性扰动提高对抗性攻击的可转移性 cs.CV

NeurIPS 2022 conference paper

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05968v1) [paper-pdf](http://arxiv.org/pdf/2210.05968v1)

**Authors**: Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples, which can produce erroneous predictions by injecting imperceptible perturbations. In this work, we study the transferability of adversarial examples, which is significant due to its threat to real-world applications where model architecture or parameters are usually unknown. Many existing works reveal that the adversarial examples are likely to overfit the surrogate model that they are generated from, limiting its transfer attack performance against different target models. To mitigate the overfitting of the surrogate model, we propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem. By integrating RAP into the iterative process for attacks, our method can find more stable adversarial examples which are less sensitive to the changes of decision boundary, mitigating the overfitting of the surrogate model. Comprehensive experimental comparisons demonstrate that RAP can significantly boost adversarial transferability. Furthermore, RAP can be naturally combined with many existing black-box attack techniques, to further boost the transferability. When attacking a real-world image recognition system, Google Cloud Vision API, we obtain 22% performance improvement of targeted attacks over the compared method. Our codes are available at https://github.com/SCLBD/Transfer_attack_RAP.

摘要: 深度神经网络(DNN)已被证明容易受到敌意例子的攻击，这些例子通过注入不可察觉的扰动而产生错误的预测。在这项工作中，我们研究了对抗性例子的可转移性，这一点很重要，因为它对模型结构或参数通常未知的现实世界应用程序构成了威胁。许多已有的工作表明，敌意示例可能会过度匹配生成它们的代理模型，从而限制了其对不同目标模型的传输攻击性能。为了缓解代理模型的过度拟合，我们提出了一种新的攻击方法，称为反向对抗扰动(RAP)。具体地说，我们主张通过为优化过程的每一步注入最坏情况的扰动(反向对抗性扰动)来寻找位于具有统一低损失值的区域的对抗性实例，而不是最小化单个对抗点的损失。基于RAP的对抗性攻击被描述为一个最小-最大双层优化问题。通过将RAP集成到攻击的迭代过程中，我们的方法可以找到更稳定的对抗性实例，这些实例对决策边界的变化不那么敏感，从而缓解了代理模型的过度拟合问题。综合实验比较表明，RAP能够显著提高对抗性转移能力。此外，RAP可以自然地与许多现有的黑盒攻击技术相结合，进一步提高可转移性。在攻击真实世界的图像识别系统Google Cloud Vision API时，与比较的方法相比，我们获得了22%的定向攻击性能提升。我们的代码可在https://github.com/SCLBD/Transfer_attack_RAP.上获得



## **46. Robust Models are less Over-Confident**

稳健的模型不那么过度自信 cs.CV

accepted at NeuRips 2022

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05938v1) [paper-pdf](http://arxiv.org/pdf/2210.05938v1)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation

摘要: 尽管卷积神经网络(CNN)在许多计算机视觉任务的学术基准中取得了成功，但它们在现实世界中的应用仍然面临着根本性的挑战。这些悬而未决的问题之一是固有的健壮性不足，这一点从对抗性攻击的惊人有效性中可见一斑。目前的攻击方法能够通过向输入添加特定但少量的噪声来操纵网络的预测。反过来，对抗性训练(AT)的目的是通过将对抗性样本包括在训练集中来实现对此类攻击的健壮性，并且理想地实现更好的模型泛化能力。然而，对由此产生的超越对抗性稳健性的稳健性模型的深入分析仍然悬而未决。在这篇文章中，我们实证分析了各种对抗训练的模型，这些模型在面对最先进的攻击时获得了很高的稳健精度，我们发现AT有一个有趣的副作用：它导致模型对他们的决策不那么过度自信，即使是在干净的数据上也是如此。此外，我们对稳健模型的分析表明，不仅AT而且模型的构件(如激活函数和池化)对模型的预测置信度有很大的影响。数据与项目网站：https://github.com/GeJulia/robustness_confidences_evaluation



## **47. Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning**

无攻击的高效对抗性训练：最坏情况感知的稳健强化学习 cs.LG

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05927v1) [paper-pdf](http://arxiv.org/pdf/2210.05927v1)

**Authors**: Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang

**Abstract**: Recent studies reveal that a well-trained deep reinforcement learning (RL) policy can be particularly vulnerable to adversarial perturbations on input observations. Therefore, it is crucial to train RL agents that are robust against any attacks with a bounded budget. Existing robust training methods in deep RL either treat correlated steps separately, ignoring the robustness of long-term rewards, or train the agents and RL-based attacker together, doubling the computational burden and sample complexity of the training process. In this work, we propose a strong and efficient robust training framework for RL, named Worst-case-aware Robust RL (WocaR-RL) that directly estimates and optimizes the worst-case reward of a policy under bounded l_p attacks without requiring extra samples for learning an attacker. Experiments on multiple environments show that WocaR-RL achieves state-of-the-art performance under various strong attacks, and obtains significantly higher training efficiency than prior state-of-the-art robust training methods. The code of this work is available at https://github.com/umd-huang-lab/WocaR-RL.

摘要: 最近的研究表明，训练有素的深度强化学习(RL)策略特别容易受到输入观测的对抗性扰动。因此，在有限的预算内培训对任何攻击具有健壮性的RL代理是至关重要的。现有的深度RL稳健训练方法要么单独处理相关步骤，忽略长期回报的稳健性，要么将代理和基于RL的攻击者一起训练，使训练过程的计算负担和样本复杂度翻了一番。在这项工作中，我们提出了一种强而有效的RL稳健训练框架，称为最坏情况感知稳健RL(WocaR-RL)，它可以直接估计和优化策略在有界l_p攻击下的最坏情况奖励，而不需要额外的样本来学习攻击者。在多种环境下的实验表明，WocaR-RL在各种强攻击下具有最好的性能，训练效率明显高于现有的健壮训练方法。这项工作的代码可以在https://github.com/umd-huang-lab/WocaR-RL.上找到



## **48. On the Limitations of Stochastic Pre-processing Defenses**

论随机前处理防御的局限性 cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.09491v3) [paper-pdf](http://arxiv.org/pdf/2206.09491v3)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstract**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.

摘要: 抵御敌意的例子仍然是一个悬而未决的问题。一种普遍的看法是，推理的随机性增加了寻找敌对输入的成本。这种防御的一个例子是在将输入提供给模型之前对它们应用随机转换。在本文中，我们从经验和理论上研究了这种随机预处理防御机制，并证明了它们是有缺陷的。首先，我们证明了大多数随机防御比之前认为的要弱；它们缺乏足够的随机性，即使是像投影梯度下降这样的标准攻击也是如此。这让人对一个长期持有的假设产生了怀疑，即随机防御使旨在逃避确定性防御的攻击无效，并迫使攻击者整合期望过转换(EOT)概念。其次，我们证明了随机防御面临着对抗稳健性和模型不变性之间的权衡；随着被防御模型对其随机化获得更多的不变性，它们变得不那么有效。未来的工作将需要将这两种影响脱钩。我们还讨论了对未来研究的启示和指导。



## **49. Adversarial Attack Against Image-Based Localization Neural Networks**

基于图像的定位神经网络的对抗性攻击 cs.CV

13 pages, 10 figures

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.06589v1) [paper-pdf](http://arxiv.org/pdf/2210.06589v1)

**Authors**: Meir Brand, Itay Naeh, Daniel Teitelman

**Abstract**: In this paper, we present a proof of concept for adversarially attacking the image-based localization module of an autonomous vehicle. This attack aims to cause the vehicle to perform a wrong navigational decisions and prevent it from reaching a desired predefined destination in a simulated urban environment. A database of rendered images allowed us to train a deep neural network that performs a localization task and implement, develop and assess the adversarial pattern. Our tests show that using this adversarial attack we can prevent the vehicle from turning at a given intersection. This is done by manipulating the vehicle's navigational module to falsely estimate its current position and thus fail to initialize the turning procedure until the vehicle misses the last opportunity to perform a safe turn in a given intersection.

摘要: 在这篇文章中，我们提出了一种概念证明，用于恶意攻击自主车辆的基于图像的定位模块。这种攻击旨在导致车辆执行错误的导航决策，并阻止其在模拟城市环境中到达预期的预定目的地。渲染图像的数据库使我们能够训练执行定位任务的深度神经网络，并实施、开发和评估对抗性模式。我们的测试表明，使用这种对抗性攻击，我们可以防止车辆在给定的十字路口转弯。这是通过操纵车辆的导航模块错误地估计其当前位置，从而无法初始化转弯程序，直到车辆错过在给定十字路口执行安全转弯的最后机会来实现的。



## **50. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

攻击失败的指标：对抗性实例的调试和改进优化 cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2106.09947v3) [paper-pdf](http://arxiv.org/pdf/2106.09947v3)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstract**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.

摘要: 评估机器学习模型对对抗性样本的稳健性是一个具有挑战性的问题。事实证明，许多防御措施通过导致基于梯度的攻击失败来提供一种错误的健壮感，这些防御措施已经在更严格的评估下被打破。虽然有人建议采用准则和最佳做法来改进目前的对抗性评估，但由于缺乏自动测试和调试工具，很难系统地适用这些建议。在这项工作中，我们克服了这些局限性：(I)根据攻击失败如何影响基于梯度的攻击的优化进行分类，同时也揭示了影响许多流行攻击实现和过去评估的两个新失败；(Ii)提出了六个新的失败指示器，以自动检测攻击优化过程中此类失败的存在；以及(Iii)提出了一个系统的协议来应用相应的修复。我们广泛的实验分析，涉及3个不同应用领域的15个模型，表明我们的失败指示器可以用于调试和改进当前的对手健壮性评估，从而为实现自动化和系统化迈出了具体的第一步。我们的开源代码可以在https://github.com/pralab/IndicatorsOfAttackFailure.上找到



