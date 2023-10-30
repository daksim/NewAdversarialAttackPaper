# Latest Adversarial Attack Papers
**update at 2023-10-30 09:53:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LipSim: A Provably Robust Perceptual Similarity Metric**

LipSim：一种可证明的稳健感知相似性度量 cs.CV

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18274v1) [paper-pdf](http://arxiv.org/pdf/2310.18274v1)

**Authors**: Sara Ghazanfari, Alexandre Araujo, Prashanth Krishnamurthy, Farshad Khorrami, Siddharth Garg

**Abstract**: Recent years have seen growing interest in developing and applying perceptual similarity metrics. Research has shown the superiority of perceptual metrics over pixel-wise metrics in aligning with human perception and serving as a proxy for the human visual system. On the other hand, as perceptual metrics rely on neural networks, there is a growing concern regarding their resilience, given the established vulnerability of neural networks to adversarial attacks. It is indeed logical to infer that perceptual metrics may inherit both the strengths and shortcomings of neural networks. In this work, we demonstrate the vulnerability of state-of-the-art perceptual similarity metrics based on an ensemble of ViT-based feature extractors to adversarial attacks. We then propose a framework to train a robust perceptual similarity metric called LipSim (Lipschitz Similarity Metric) with provable guarantees. By leveraging 1-Lipschitz neural networks as the backbone, LipSim provides guarded areas around each data point and certificates for all perturbations within an $\ell_2$ ball. Finally, a comprehensive set of experiments shows the performance of LipSim in terms of natural and certified scores and on the image retrieval application. The code is available at https://github.com/SaraGhazanfari/LipSim.

摘要: 近年来，人们对开发和应用感知相似性度量的兴趣与日俱增。研究表明，与像素度量相比，感知度量在与人类感知和作为人类视觉系统的代理方面具有优势。另一方面，由于感知指标依赖于神经网络，鉴于神经网络对对手攻击的公认脆弱性，人们越来越担心其弹性。推断感知指标可能继承了神经网络的长处和短处，这确实是合乎逻辑的。在这项工作中，我们展示了基于基于VIT的特征提取集合的最新感知相似性度量在对抗攻击中的脆弱性。然后，我们提出了一个框架来训练一个健壮的感知相似性度量，称为LipSim(Lipschitz相似性度量)，并具有可证明的保证。通过利用1-Lipschitz神经网络作为主干，LipSim在每个数据点周围提供保护区域，并为$\ell_2$球内的所有扰动提供证书。最后，一组全面的实验显示了LipSim在自然分数和认证分数以及图像检索应用方面的性能。代码可在https://github.com/SaraGhazanfari/LipSim.上获得



## **2. $α$-Mutual Information: A Tunable Privacy Measure for Privacy Protection in Data Sharing**

$α$-互信息：数据共享中隐私保护的可调隐私度量 cs.LG

2023 22nd IEEE International Conference on Machine Learning and  Applications (ICMLA)

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18241v1) [paper-pdf](http://arxiv.org/pdf/2310.18241v1)

**Authors**: MirHamed Jafarzadeh Asl, Mohammadhadi Shateri, Fabrice Labeau

**Abstract**: This paper adopts Arimoto's $\alpha$-Mutual Information as a tunable privacy measure, in a privacy-preserving data release setting that aims to prevent disclosing private data to adversaries. By fine-tuning the privacy metric, we demonstrate that our approach yields superior models that effectively thwart attackers across various performance dimensions. We formulate a general distortion-based mechanism that manipulates the original data to offer privacy protection. The distortion metrics are determined according to the data structure of a specific experiment. We confront the problem expressed in the formulation by employing a general adversarial deep learning framework that consists of a releaser and an adversary, trained with opposite goals. This study conducts empirical experiments on images and time-series data to verify the functionality of $\alpha$-Mutual Information. We evaluate the privacy-utility trade-off of customized models and compare them to mutual information as the baseline measure. Finally, we analyze the consequence of an attacker's access to side information about private data and witness that adapting the privacy measure results in a more refined model than the state-of-the-art in terms of resiliency against side information.

摘要: 本文采用Arimoto的$\Alpha$-Mutual Information作为一种可调的隐私措施，在隐私保护的数据发布环境中，旨在防止将私人数据泄露给攻击者。通过微调隐私度量，我们证明了我们的方法产生了卓越的模型，可以有效地在不同的性能维度上挫败攻击者。我们制定了一个通用的基于失真的机制，对原始数据进行操纵以提供隐私保护。根据特定实验的数据结构确定失真度量。我们通过采用一般对抗性深度学习框架来应对公式中表达的问题，该框架由释放者和对手组成，训练目标相反。本研究在图像和时间序列数据上进行了实证实验，以验证$-α$-相互信息的功能。我们评估定制模型的隐私-效用权衡，并将它们与互信息作为基线衡量标准进行比较。最后，我们分析了攻击者访问有关私有数据的辅助信息的后果，并见证了采用隐私措施在对抗辅助信息方面的弹性方面比最新的模型更精细。



## **3. Adaptive Webpage Fingerprinting from TLS Traces**

基于TLS轨迹的自适应网页指纹识别 cs.CR

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2010.10294v2) [paper-pdf](http://arxiv.org/pdf/2010.10294v2)

**Authors**: Vasilios Mavroudis, Jamie Hayes

**Abstract**: In webpage fingerprinting, an on-path adversary infers the specific webpage loaded by a victim user by analysing the patterns in the encrypted TLS traffic exchanged between the user's browser and the website's servers. This work studies modern webpage fingerprinting adversaries against the TLS protocol; aiming to shed light on their capabilities and inform potential defences. Despite the importance of this research area (the majority of global Internet users rely on standard web browsing with TLS) and the potential real-life impact, most past works have focused on attacks specific to anonymity networks (e.g., Tor). We introduce a TLS-specific model that: 1) scales to an unprecedented number of target webpages, 2) can accurately classify thousands of classes it never encountered during training, and 3) has low operational costs even in scenarios of frequent page updates. Based on these findings, we then discuss TLS-specific countermeasures and evaluate the effectiveness of the existing padding capabilities provided by TLS 1.3.

摘要: 在网页指纹识别中，路径上的攻击者通过分析用户浏览器和网站服务器之间交换的加密TLS流量的模式来推断受攻击用户加载的特定网页。这项工作研究了针对TLS协议的现代网页指纹识别攻击者，旨在揭示他们的能力并为潜在的防御提供信息。尽管这一研究领域的重要性(大多数全球互联网用户依赖于使用TLS的标准网络浏览)及其潜在的现实影响，但过去的大多数工作都集中在针对匿名网络的攻击(例如，ToR)。我们引入了一种TLS特定的模型：1)扩展到前所未有的目标网页数量；2)可以准确地分类它在训练过程中从未遇到的数千个类；3)即使在页面更新频繁的情况下，也具有较低的操作成本。基于这些发现，我们讨论了特定于TLS的对策，并评估了TLS 1.3提供的现有填充功能的有效性。



## **4. Elevating Code-mixed Text Handling through Auditory Information of Words**

利用词汇听觉信息提升语码混合文本处理 cs.CL

Accepted to EMNLP 2023

**SubmitDate**: 2023-10-27    [abs](http://arxiv.org/abs/2310.18155v1) [paper-pdf](http://arxiv.org/pdf/2310.18155v1)

**Authors**: Mamta, Zishan Ahmad, Asif Ekbal

**Abstract**: With the growing popularity of code-mixed data, there is an increasing need for better handling of this type of data, which poses a number of challenges, such as dealing with spelling variations, multiple languages, different scripts, and a lack of resources. Current language models face difficulty in effectively handling code-mixed data as they primarily focus on the semantic representation of words and ignore the auditory phonetic features. This leads to difficulties in handling spelling variations in code-mixed text. In this paper, we propose an effective approach for creating language models for handling code-mixed textual data using auditory information of words from SOUNDEX. Our approach includes a pre-training step based on masked-language-modelling, which includes SOUNDEX representations (SAMLM) and a new method of providing input data to the pre-trained model. Through experimentation on various code-mixed datasets (of different languages) for sentiment, offensive and aggression classification tasks, we establish that our novel language modeling approach (SAMLM) results in improved robustness towards adversarial attacks on code-mixed classification tasks. Additionally, our SAMLM based approach also results in better classification results over the popular baselines for code-mixed tasks. We use the explainability technique, SHAP (SHapley Additive exPlanations) to explain how the auditory features incorporated through SAMLM assist the model to handle the code-mixed text effectively and increase robustness against adversarial attacks \footnote{Source code has been made available on \url{https://github.com/20118/DefenseWithPhonetics}, \url{https://www.iitp.ac.in/~ai-nlp-ml/resources.html\#Phonetics}}.

摘要: 随着代码混合数据的日益流行，越来越需要更好地处理这种类型的数据，这带来了许多挑战，例如处理拼写变化、多种语言、不同的脚本和缺乏资源。现有的语言模型主要关注词语的语义表征，而忽视了听觉语音特征，难以有效地处理混码数据。这导致难以处理代码混合文本中的拼写变化。在本文中，我们提出了一种利用Soundex中单词的听觉信息来建立处理代码混合文本数据的语言模型的有效方法。我们的方法包括基于掩蔽语言建模的预训练步骤，其中包括Soundex表示(SAMLM)和向预训练模型提供输入数据的新方法。通过在情感、攻击性和攻击性分类任务的不同代码混合数据集上的实验，我们建立了新的语言建模方法(SAMLM)，从而提高了对代码混合分类任务的敌意攻击的鲁棒性。此外，我们基于SAMLM的方法还可以在代码混合任务的流行基线上产生更好的分类结果。我们使用可解释性技术Shap(Shap)来解释通过SAMLM整合的听觉特征如何帮助模型有效地处理代码混合的文本并增加对对手攻击的稳健性\脚注{源代码已在\url{https://github.com/20118/DefenseWithPhonetics}，\url{https://www.iitp.ac.in/~ai-nlp-ml/resources.html\#Phonetics}}.上提供



## **5. A Unified Algebraic Perspective on Lipschitz Neural Networks**

关于Lipschitz神经网络的统一代数观点 cs.LG

ICLR 2023. Spotlight paper

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2303.03169v2) [paper-pdf](http://arxiv.org/pdf/2303.03169v2)

**Authors**: Alexandre Araujo, Aaron Havens, Blaise Delattre, Alexandre Allauzen, Bin Hu

**Abstract**: Important research efforts have focused on the design and training of neural networks with a controlled Lipschitz constant. The goal is to increase and sometimes guarantee the robustness against adversarial attacks. Recent promising techniques draw inspirations from different backgrounds to design 1-Lipschitz neural networks, just to name a few: convex potential layers derive from the discretization of continuous dynamical systems, Almost-Orthogonal-Layer proposes a tailored method for matrix rescaling. However, it is today important to consider the recent and promising contributions in the field under a common theoretical lens to better design new and improved layers. This paper introduces a novel algebraic perspective unifying various types of 1-Lipschitz neural networks, including the ones previously mentioned, along with methods based on orthogonality and spectral methods. Interestingly, we show that many existing techniques can be derived and generalized via finding analytical solutions of a common semidefinite programming (SDP) condition. We also prove that AOL biases the scaled weight to the ones which are close to the set of orthogonal matrices in a certain mathematical manner. Moreover, our algebraic condition, combined with the Gershgorin circle theorem, readily leads to new and diverse parameterizations for 1-Lipschitz network layers. Our approach, called SDP-based Lipschitz Layers (SLL), allows us to design non-trivial yet efficient generalization of convex potential layers. Finally, the comprehensive set of experiments on image classification shows that SLLs outperform previous approaches on certified robust accuracy. Code is available at https://github.com/araujoalexandre/Lipschitz-SLL-Networks.

摘要: 重要的研究工作集中在设计和训练具有受控Lipschitz常数的神经网络。其目标是增加有时甚至保证对对手攻击的健壮性。最近很有前途的技术从不同的背景中获得了设计1-Lipschitz神经网络的灵感，仅举几个例子：凸势层源于连续动力系统的离散化，几乎正交层提出了一种定制的矩阵重标度方法。然而，今天，重要的是在共同的理论视角下考虑该领域最近和有希望的贡献，以更好地设计新的和改进的层。本文介绍了一种新的代数观点，统一了各种类型的1-Lipschitz神经网络，包括前面提到的那些，以及基于正交性和谱方法的方法。有趣的是，我们证明了许多现有的技术可以通过寻找一个常见的半定规划(SDP)条件的解析解来推导和推广。我们还证明了AOL以一定的数学方式将标度权重偏向于接近正交矩阵集的权重。此外，我们的代数条件与Gershgorin圆定理相结合，很容易得到新的和不同的1-Lipschitz网络层的参数。我们的方法称为基于SDP的Lipschitz层(SLL)，它允许我们设计非平凡但有效的凸势层的推广。最后，一组综合的图像分类实验表明，SLLS在证明的稳健性精度方面优于以往的方法。代码可在https://github.com/araujoalexandre/Lipschitz-SLL-Networks.上找到



## **6. Defending Against Transfer Attacks From Public Models**

防御来自公共模型的传输攻击 cs.LG

Under submission. Code available at  https://github.com/wagner-group/pubdef

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17645v1) [paper-pdf](http://arxiv.org/pdf/2310.17645v1)

**Authors**: Chawin Sitawarin, Jaewon Chang, David Huang, Wesson Altoyan, David Wagner

**Abstract**: Adversarial attacks have been a looming and unaddressed threat in the industry. However, through a decade-long history of the robustness evaluation literature, we have learned that mounting a strong or optimal attack is challenging. It requires both machine learning and domain expertise. In other words, the white-box threat model, religiously assumed by a large majority of the past literature, is unrealistic. In this paper, we propose a new practical threat model where the adversary relies on transfer attacks through publicly available surrogate models. We argue that this setting will become the most prevalent for security-sensitive applications in the future. We evaluate the transfer attacks in this setting and propose a specialized defense method based on a game-theoretic perspective. The defenses are evaluated under 24 public models and 11 attack algorithms across three datasets (CIFAR-10, CIFAR-100, and ImageNet). Under this threat model, our defense, PubDef, outperforms the state-of-the-art white-box adversarial training by a large margin with almost no loss in the normal accuracy. For instance, on ImageNet, our defense achieves 62% accuracy under the strongest transfer attack vs only 36% of the best adversarially trained model. Its accuracy when not under attack is only 2% lower than that of an undefended model (78% vs 80%). We release our code at https://github.com/wagner-group/pubdef.

摘要: 对抗性攻击一直是该行业一个迫在眉睫且尚未解决的威胁。然而，通过长达十年的健壮性评估文献的历史，我们已经了解到，发动强大或最佳的攻击是具有挑战性的。它既需要机器学习，也需要领域专业知识。换句话说，过去大多数文献虔诚地假设的白盒威胁模型是不现实的。在本文中，我们提出了一个新的实用威胁模型，其中敌手通过公开可用的代理模型依赖于传输攻击。我们认为，此设置将成为未来安全敏感应用程序的最流行设置。在此背景下，我们对传输攻击进行了评估，并从博弈论的角度提出了一种专门的防御方法。这些防御在24个公共模型和11个攻击算法下进行了评估，涉及三个数据集(CIFAR-10、CIFAR-100和ImageNet)。在这种威胁模型下，我们的防御PubDef在正常准确率几乎没有损失的情况下，远远超过最先进的白盒对抗训练。例如，在ImageNet上，我们的防御在最强的传输攻击下达到了62%的准确率，而最好的对手训练模型的准确率只有36%。在没有受到攻击的情况下，它的准确性只比没有防御的模型低2%(78%比80%)。我们在https://github.com/wagner-group/pubdef.发布我们的代码



## **7. A Survey on Transferability of Adversarial Examples across Deep Neural Networks**

对抗性实例在深度神经网络中的可转移性研究 cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17626v1) [paper-pdf](http://arxiv.org/pdf/2310.17626v1)

**Authors**: Jindong Gu, Xiaojun Jia, Pau de Jorge, Wenqain Yu, Xinwei Liu, Avery Ma, Yuan Xun, Anjun Hu, Ashkan Khakzar, Zhijiang Li, Xiaochun Cao, Philip Torr

**Abstract**: The emergence of Deep Neural Networks (DNNs) has revolutionized various domains, enabling the resolution of complex tasks spanning image recognition, natural language processing, and scientific problem-solving. However, this progress has also exposed a concerning vulnerability: adversarial examples. These crafted inputs, imperceptible to humans, can manipulate machine learning models into making erroneous predictions, raising concerns for safety-critical applications. An intriguing property of this phenomenon is the transferability of adversarial examples, where perturbations crafted for one model can deceive another, often with a different architecture. This intriguing property enables "black-box" attacks, circumventing the need for detailed knowledge of the target model. This survey explores the landscape of the adversarial transferability of adversarial examples. We categorize existing methodologies to enhance adversarial transferability and discuss the fundamental principles guiding each approach. While the predominant body of research primarily concentrates on image classification, we also extend our discussion to encompass other vision tasks and beyond. Challenges and future prospects are discussed, highlighting the importance of fortifying DNNs against adversarial vulnerabilities in an evolving landscape.

摘要: 深度神经网络(DNN)的出现给各个领域带来了革命性的变化，使图像识别、自然语言处理和科学问题解决等复杂任务的解决成为可能。然而，这一进展也暴露了一个令人担忧的脆弱性：对抗性的例子。这些精心制作的、人类无法察觉的输入可能会操纵机器学习模型做出错误的预测，从而引发对安全关键应用的担忧。这种现象的一个耐人寻味的属性是对抗性例子的可转移性，在这种情况下，为一个模型精心设计的扰动可以欺骗另一个模型，通常是使用不同的架构。这一耐人寻味的特性使“黑匣子”攻击成为可能，从而避免了对目标模型详细了解的需要。这项调查探讨了对抗性例子的对抗性转移的情况。我们对增强对抗性可转移性的现有方法进行了分类，并讨论了指导每种方法的基本原则。虽然主要的研究主体主要集中在图像分类上，但我们也扩展了我们的讨论，以涵盖其他视觉任务和其他任务。讨论了挑战和未来前景，强调了在不断发展的环境中加强DNN对抗对手脆弱性的重要性。



## **8. Gaussian Membership Inference Privacy**

高斯隶属度推理隐私性 cs.LG

NeurIPS 2023 camera-ready. The first two authors contributed equally

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2306.07273v2) [paper-pdf](http://arxiv.org/pdf/2306.07273v2)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a novel and practical privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. Consequently, $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). In particular, we derive a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP) by theoretically analyzing likelihood ratio-based membership inference attacks on stochastic gradient descent (SGD). Our analysis highlights that models trained with standard SGD already offer an elementary level of MIP. Additionally, we show how $f$-MIP can be amplified by adding noise to gradient updates. Our analysis further yields an analytical membership inference attack that offers two distinct advantages over previous approaches. First, unlike existing state-of-the-art attacks that require training hundreds of shadow models, our attack does not require any shadow model. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, we quantify how various hyperparameters (e.g., batch size, number of model parameters) and specific data characteristics determine an attacker's ability to accurately infer a point's membership in the training set. We demonstrate the effectiveness of our method on models trained on vision and tabular datasets.

摘要: 提出了一种新颖实用的隐私概念$f$-MIP($f$-MIP)，它明确地考虑了现实对手在成员关系推理攻击威胁模型下的能力。因此，$f$-MIP提供了可解释的隐私保证和改进的实用性(例如，更好的分类准确性)。特别地，我们通过理论分析基于似然比的随机梯度下降(SGD)成员关系推理攻击，得到了一个$f$-MIP保证的参数族，我们称之为$\MU$-高斯成员关系推理隐私($\MIP)。我们的分析强调，用标准SGD训练的模型已经提供了初级的MIP。此外，我们还展示了如何通过向梯度更新添加噪声来放大$f$-MIP。我们的分析进一步产生了一种分析性的成员关系推断攻击，与以前的方法相比提供了两个明显的优势。首先，与现有需要训练数百个影子模型的最先进攻击不同，我们的攻击不需要任何影子模型。其次，我们的分析攻击使我们能够直接审计我们的隐私概念$f$-mip。最后，我们量化各种超参数(例如，批次大小、模型参数的数量)和特定数据特征如何确定攻击者准确推断点在训练集中的成员资格的能力。我们在视觉和表格数据集上训练的模型上展示了我们的方法的有效性。



## **9. Instability of computer vision models is a necessary result of the task itself**

计算机视觉模型的不稳定性是任务本身的必然结果 cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17559v1) [paper-pdf](http://arxiv.org/pdf/2310.17559v1)

**Authors**: Oliver Turnbull, George Cevora

**Abstract**: Adversarial examples resulting from instability of current computer vision models are an extremely important topic due to their potential to compromise any application. In this paper we demonstrate that instability is inevitable due to a) symmetries (translational invariance) of the data, b) the categorical nature of the classification task, and c) the fundamental discrepancy of classifying images as objects themselves. The issue is further exacerbated by non-exhaustive labelling of the training data. Therefore we conclude that instability is a necessary result of how the problem of computer vision is currently formulated. While the problem cannot be eliminated, through the analysis of the causes, we have arrived at ways how it can be partially alleviated. These include i) increasing the resolution of images, ii) providing contextual information for the image, iii) exhaustive labelling of training data, and iv) preventing attackers from frequent access to the computer vision system.

摘要: 当前计算机视觉模型的不稳定性导致的对抗性例子是一个极其重要的话题，因为它们可能危及任何应用。在本文中，我们证明了不稳定性是不可避免的，这是由于a)数据的对称性(平移不变性)，b)分类任务的分类性质，以及c)将图像分类为对象本身的根本差异。训练数据的非详尽标签进一步加剧了这一问题。因此，我们得出结论，不稳定性是计算机视觉问题当前表述方式的必然结果。虽然这个问题无法消除，但通过对原因的分析，我们已经找到了如何部分缓解的方法。这些措施包括i)提高图像的分辨率，ii)提供图像的上下文信息，iii)对训练数据进行详尽的标记，以及iv)防止攻击者频繁访问计算机视觉系统。



## **10. SoK: Pitfalls in Evaluating Black-Box Attacks**

SOK：评估黑盒攻击的陷阱 cs.CR

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17534v1) [paper-pdf](http://arxiv.org/pdf/2310.17534v1)

**Authors**: Fnu Suya, Anshuman Suri, Tingwei Zhang, Jingtao Hong, Yuan Tian, David Evans

**Abstract**: Numerous works study black-box attacks on image classifiers. However, these works make different assumptions on the adversary's knowledge and current literature lacks a cohesive organization centered around the threat model. To systematize knowledge in this area, we propose a taxonomy over the threat space spanning the axes of feedback granularity, the access of interactive queries, and the quality and quantity of the auxiliary data available to the attacker. Our new taxonomy provides three key insights. 1) Despite extensive literature, numerous under-explored threat spaces exist, which cannot be trivially solved by adapting techniques from well-explored settings. We demonstrate this by establishing a new state-of-the-art in the less-studied setting of access to top-k confidence scores by adapting techniques from well-explored settings of accessing the complete confidence vector, but show how it still falls short of the more restrictive setting that only obtains the prediction label, highlighting the need for more research. 2) Identification the threat model of different attacks uncovers stronger baselines that challenge prior state-of-the-art claims. We demonstrate this by enhancing an initially weaker baseline (under interactive query access) via surrogate models, effectively overturning claims in the respective paper. 3) Our taxonomy reveals interactions between attacker knowledge that connect well to related areas, such as model inversion and extraction attacks. We discuss how advances in other areas can enable potentially stronger black-box attacks. Finally, we emphasize the need for a more realistic assessment of attack success by factoring in local attack runtime. This approach reveals the potential for certain attacks to achieve notably higher success rates and the need to evaluate attacks in diverse and harder settings, highlighting the need for better selection criteria.

摘要: 许多著作研究了对图像分类器的黑盒攻击。然而，这些著作对对手的知识做出了不同的假设，目前的文献缺乏一个以威胁模型为中心的有凝聚力的组织。为了系统化这一领域的知识，我们提出了一种横跨反馈粒度、交互查询的访问以及攻击者可用的辅助数据的质量和数量轴的威胁空间分类。我们的新分类法提供了三个关键的见解。1)尽管有大量的文献，但仍然存在大量未被充分探索的威胁空间，这些威胁空间不能通过从经过充分探索的环境中采用技术来平凡地解决。我们通过在较少研究的访问Top-k置信度分数的设置中建立新的最先进的设置来证明这一点，方法是采用来自充分探索的访问完整置信度向量的设置的技术，但展示了它如何仍然没有达到仅获得预测标签的更具限制性的设置，从而突出了更多研究的必要性。2)识别不同攻击的威胁模型揭示了挑战先前最先进主张的更强大的基线。我们通过代理模型增强了最初较弱的基线(在交互式查询访问下)，有效地推翻了各自论文中的主张，从而证明了这一点。3)我们的分类揭示了攻击者知识之间的交互作用，这些知识与相关领域联系良好，如模型倒置和提取攻击。我们讨论了其他领域的进步如何使潜在更强大的黑盒攻击成为可能。最后，我们强调需要通过考虑本地攻击运行时来更现实地评估攻击成功。这种方法揭示了某些攻击实现显著更高成功率的潜力，以及在不同和更困难的环境中评估攻击的必要性，突出了需要更好的选择标准。



## **11. CBD: A Certified Backdoor Detector Based on Local Dominant Probability**

CBD：一种基于局部支配概率的认证后门检测器 cs.LG

Accepted to NeurIPS 2023

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17498v1) [paper-pdf](http://arxiv.org/pdf/2310.17498v1)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.

摘要: 后门攻击是深度神经网络的常见威胁。在测试过程中，嵌入后门触发器的样本将被后门模型错误分类为对手目标，而没有后门触发器的样本将被正确分类。在本文中，我们提出了第一个认证的后门检测器(CBD)，它基于我们提出的统计局部主导概率的一种新的、可调整的共形预测方案。对于被检查的任何分类器，CBD提供了1)检测推理，2)保证攻击对于相同的分类域是可检测的条件，以及3)错误阳性率的概率上界。我们的理论结果表明，触发对测试时间噪声更具弹性并且扰动幅度更小的攻击更有可能被检测到。此外，我们在四个基准数据集上进行了广泛的实验，考虑了不同的后门类型，如BadNet、CB和Blend。CBD实现了与最先进的检测器相当甚至更高的检测精度，此外，它还提供检测认证。值得注意的是，对于攻击成功率超过90%的随机扰动触发的后门攻击，CBD在GTSRB、SVHN、CIFAR-10和TinyImageNet四个基准数据集上分别获得了100(98)、100(84)、98(98)和72(40)的经验(认证)检测真阳性，假阳性率较低。



## **12. Uncertainty-weighted Loss Functions for Improved Adversarial Attacks on Semantic Segmentation**

改进对抗性语义分割攻击的不确定性加权损失函数 cs.CV

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17436v1) [paper-pdf](http://arxiv.org/pdf/2310.17436v1)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have been shown to be extremely powerful in a variety of perceptual tasks like semantic segmentation. However, these networks are vulnerable to adversarial perturbations of the input which are imperceptible for humans but lead to incorrect predictions. Treating image segmentation as a sum of pixel-wise classifications, adversarial attacks developed for classification models were shown to be applicable to segmentation models as well. In this work, we present simple uncertainty-based weighting schemes for the loss functions of such attacks that (i) put higher weights on pixel classifications which can more easily perturbed and (ii) zero-out the pixel-wise losses corresponding to those pixels that are already confidently misclassified. The weighting schemes can be easily integrated into the loss function of a range of well-known adversarial attackers with minimal additional computational overhead, but lead to significant improved perturbation performance, as we demonstrate in our empirical analysis on several datasets and models.

摘要: 最先进的深度神经网络已被证明在各种感知任务中非常强大，如语义分割。然而，这些网络容易受到输入的对抗性扰动，这种扰动对人类来说是不可察觉的，但会导致错误的预测。将图像分割看作像素级分类的总和，针对分类模型开发的对抗性攻击同样适用于分割模型。在这项工作中，我们为这类攻击的损失函数提出了简单的基于不确定性的加权方案，该方案(I)对更容易被扰动的像素分类给予更高的权重，(Ii)将与已经被错误分类的像素相对应的像素级损失归零。加权方案可以很容易地集成到一系列知名对手攻击者的损失函数中，而只需最小的额外计算开销，但会显著提高扰动性能，正如我们在几个数据集和模型上的经验分析所展示的那样。



## **13. Stealthy SWAPs: Adversarial SWAP Injection in Multi-Tenant Quantum Computing**

隐形交换：多租户量子计算中的对抗性交换注入 quant-ph

7 pages, VLSID

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17426v1) [paper-pdf](http://arxiv.org/pdf/2310.17426v1)

**Authors**: Suryansh Upadhyay, Swaroop Ghosh

**Abstract**: Quantum computing (QC) holds tremendous promise in revolutionizing problem-solving across various domains. It has been suggested in literature that 50+ qubits are sufficient to achieve quantum advantage (i.e., to surpass supercomputers in solving certain class of optimization problems).The hardware size of existing Noisy Intermediate-Scale Quantum (NISQ) computers have been ever increasing over the years. Therefore, Multi-tenant computing (MTC) has emerged as a potential solution for efficient hardware utilization, enabling shared resource access among multiple quantum programs. However, MTC can also bring new security concerns. This paper proposes one such threat for MTC in superconducting quantum hardware i.e., adversarial SWAP gate injection in victims program during compilation for MTC. We present a representative scheduler designed for optimal resource allocation. To demonstrate the impact of this attack model, we conduct a detailed case study using a sample scheduler. Exhaustive experiments on circuits with varying depths and qubits offer valuable insights into the repercussions of these attacks. We report a max of approximately 55 percent and a median increase of approximately 25 percent in SWAP overhead. As a countermeasure, we also propose a sample machine learning model for detecting any abnormal user behavior and priority adjustment.

摘要: 量子计算（QC）在各个领域彻底改变问题解决方面具有巨大的潜力。在文献中已经提出，50+量子比特足以实现量子优势（即，在解决某些类别的优化问题方面超过超级计算机）。现有的噪声中等规模量子（NISQ）计算机的硬件尺寸多年来一直在增加。因此，多租户计算（MTC）已经成为有效利用硬件的潜在解决方案，使多个量子程序之间能够共享资源访问。然而，MTC也可能带来新的安全问题。本文提出了超导量子硬件中MTC的一个这样的威胁，即，在MTC编译期间，在受害者程序中注入对抗性SWAP门。我们提出了一个有代表性的调度器设计的最佳资源分配。为了证明这种攻击模型的影响，我们进行了详细的案例研究，使用一个样本调度程序。对具有不同深度和量子位的电路进行的详尽实验为这些攻击的影响提供了有价值的见解。我们报告的最大值约为55%，平均值约为25%。作为对策，我们还提出了一个样本机器学习模型，用于检测任何异常用户行为和优先级调整。



## **14. Detection Defenses: An Empty Promise against Adversarial Patch Attacks on Optical Flow**

检测防御：对抗光流对抗性补丁攻击的空头承诺 cs.CV

Accepted to WACV 2024

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17403v1) [paper-pdf](http://arxiv.org/pdf/2310.17403v1)

**Authors**: Erik Scheurer, Jenny Schmalfuss, Alexander Lis, Andrés Bruhn

**Abstract**: Adversarial patches undermine the reliability of optical flow predictions when placed in arbitrary scene locations. Therefore, they pose a realistic threat to real-world motion detection and its downstream applications. Potential remedies are defense strategies that detect and remove adversarial patches, but their influence on the underlying motion prediction has not been investigated. In this paper, we thoroughly examine the currently available detect-and-remove defenses ILP and LGS for a wide selection of state-of-the-art optical flow methods, and illuminate their side effects on the quality and robustness of the final flow predictions. In particular, we implement defense-aware attacks to investigate whether current defenses are able to withstand attacks that take the defense mechanism into account. Our experiments yield two surprising results: Detect-and-remove defenses do not only lower the optical flow quality on benign scenes, in doing so, they also harm the robustness under patch attacks for all tested optical flow methods except FlowNetC. As currently employed detect-and-remove defenses fail to deliver the promised adversarial robustness for optical flow, they evoke a false sense of security. The code is available at https://github.com/cv-stuttgart/DetectionDefenses.

摘要: 当放置在任意场景位置时，对抗性补丁破坏了光流预测的可靠性。因此，它们对真实世界的运动检测及其下游应用构成了现实的威胁。潜在的补救措施是检测和移除对抗性补丁的防御策略，但它们对潜在运动预测的影响尚未被调查。在这篇文章中，我们彻底审查了目前可用的检测和删除防御ILP和LGS的各种最先进的光流方法选择，并说明了它们对最终流动预测的质量和稳健性的副作用。特别是，我们实施防御感知攻击，以调查当前的防御是否能够抵御考虑到防御机制的攻击。我们的实验产生了两个令人惊讶的结果：检测和删除防御不仅降低了良性场景的光流质量，而且还损害了除FlowNetC之外的所有测试的光流方法在补丁攻击下的健壮性。由于目前使用的检测和删除防御系统无法为光流提供承诺的对手健壮性，它们会引起一种错误的安全感。代码可在https://github.com/cv-stuttgart/DetectionDefenses.上获得



## **15. Learning Transferable Adversarial Robust Representations via Multi-view Consistency**

基于多视点一致性的可转移对抗性稳健表示学习 cs.LG

*Equal contribution (Author ordering determined by coin flip).  NeurIPS SafetyML workshop 2022, Under review

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2210.10485v2) [paper-pdf](http://arxiv.org/pdf/2210.10485v2)

**Authors**: Minseon Kim, Hyeonjeong Ha, Dong Bok Lee, Sung Ju Hwang

**Abstract**: Despite the success on few-shot learning problems, most meta-learned models only focus on achieving good performance on clean examples and thus easily break down when given adversarially perturbed samples. While some recent works have shown that a combination of adversarial learning and meta-learning could enhance the robustness of a meta-learner against adversarial attacks, they fail to achieve generalizable adversarial robustness to unseen domains and tasks, which is the ultimate goal of meta-learning. To address this challenge, we propose a novel meta-adversarial multi-view representation learning framework with dual encoders. Specifically, we introduce the discrepancy across the two differently augmented samples of the same data instance by first updating the encoder parameters with them and further imposing a novel label-free adversarial attack to maximize their discrepancy. Then, we maximize the consistency across the views to learn transferable robust representations across domains and tasks. Through experimental validation on multiple benchmarks, we demonstrate the effectiveness of our framework on few-shot learning tasks from unseen domains, achieving over 10\% robust accuracy improvements against previous adversarial meta-learning baselines.

摘要: 尽管在少数学习问题上取得了成功，但大多数元学习模型只专注于在干净的示例上实现良好的性能，因此在给定不利扰动样本时很容易崩溃。虽然最近的一些研究表明，对抗性学习和元学习的结合可以增强元学习者对对抗性攻击的鲁棒性，但它们无法实现对未知领域和任务的可推广的对抗性鲁棒性，这是元学习的最终目标。为了解决这一挑战，我们提出了一种新的元对抗多视图表示学习框架与双编码器。具体来说，我们通过首先使用它们更新编码器参数并进一步实施一种新的无标签对抗攻击来最大化它们的差异，从而在同一数据实例的两个不同增强样本之间引入差异。然后，我们最大限度地提高跨视图的一致性，以学习跨域和任务的可转移鲁棒表示。通过对多个基准测试的实验验证，我们证明了我们的框架对来自未知领域的少量学习任务的有效性，与以前的对抗性元学习基线相比，实现了超过10%的鲁棒准确性改进。



## **16. Effective Targeted Attacks for Adversarial Self-Supervised Learning**

对抗性自监督学习的有效目标攻击 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2210.10482v2) [paper-pdf](http://arxiv.org/pdf/2210.10482v2)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sooel Son, Sung Ju Hwang

**Abstract**: Recently, unsupervised adversarial training (AT) has been highlighted as a means of achieving robustness in models without any label information. Previous studies in unsupervised AT have mostly focused on implementing self-supervised learning (SSL) frameworks, which maximize the instance-wise classification loss to generate adversarial examples. However, we observe that simply maximizing the self-supervised training loss with an untargeted adversarial attack often results in generating ineffective adversaries that may not help improve the robustness of the trained model, especially for non-contrastive SSL frameworks without negative examples. To tackle this problem, we propose a novel positive mining for targeted adversarial attack to generate effective adversaries for adversarial SSL frameworks. Specifically, we introduce an algorithm that selects the most confusing yet similar target example for a given instance based on entropy and similarity, and subsequently perturbs the given instance towards the selected target. Our method demonstrates significant enhancements in robustness when applied to non-contrastive SSL frameworks, and less but consistent robustness improvements with contrastive SSL frameworks, on the benchmark datasets.

摘要: 最近，无监督对抗训练(AT)被认为是在没有任何标签信息的情况下实现模型稳健性的一种手段。以往在无监督AT中的研究大多集中于实现自监督学习(SSL)框架，该框架最大限度地利用实例分类损失来生成对抗性实例。然而，我们观察到，简单地使用无针对性的对抗性攻击最大化自我监督的训练损失往往会导致产生无效的对手，这可能无助于提高训练模型的稳健性，特别是对于没有负面示例的非对比SSL框架。针对这一问题，我们提出了一种新的针对目标敌意攻击的正向挖掘方法，以生成针对对抗性SSL框架的有效敌手。具体地说，我们引入了一种算法，该算法基于熵和相似度为给定实例选择最混乱但相似的目标示例，并随后将给定实例扰动为所选目标。在基准数据集上，我们的方法在应用于非对比SSL框架时在稳健性方面表现出显著的增强，而与对比SSL框架相比，稳健性改善较少但一致。



## **17. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

基于鲁棒性感知CoReset选择的高效对抗性对比学习 cs.LG

NeurIPS 2023 Spotlight

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2302.03857v5) [paper-pdf](http://arxiv.org/pdf/2302.03857v5)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS. Our source code is at https://github.com/GodXuxilie/Efficient_ACL_via_RCS.

摘要: 对抗性对比学习（ACL）不需要昂贵的数据注释，而是输出一个强大的表示，可以承受对抗性攻击，并推广到广泛的下游任务。然而，ACL需要大量的运行时间来生成所有训练数据的对抗变体，这限制了其对大型数据集的可扩展性。为了提高ACL的速度，提出了一种鲁棒性感知的核心集选择（RCS）方法。RCS不需要标签信息，并搜索一个信息子集，最大限度地减少了代表性的分歧，这是自然数据和它们的虚拟对抗变体之间的代表性的距离。通过遍历所有可能的子集的RCS的香草解决方案是计算上禁止的。因此，我们从理论上将RCS转化为一个次模最大化的代理问题，其中的贪婪搜索是一个有效的解决方案，与原问题的最优性保证。从经验上讲，我们的综合结果证实，RCS可以加快ACL的大幅利润，而不会显着损害鲁棒性的可移植性。值得注意的是，据我们所知，我们是第一个在大规模ImageNet-1 K数据集上有效地进行ACL，以通过RCS获得有效的鲁棒表示。我们的源代码位于https://github.com/GodXuxilie/Efficient_ACL_via_RCS。



## **18. Detecting stealthy cyberattacks on adaptive cruise control vehicles: A machine learning approach**

检测自适应巡航控制车辆上的隐形网络攻击：一种机器学习方法 cs.MA

**SubmitDate**: 2023-10-26    [abs](http://arxiv.org/abs/2310.17091v1) [paper-pdf](http://arxiv.org/pdf/2310.17091v1)

**Authors**: Tianyi Li, Mingfeng Shang, Shian Wang, Raphael Stern

**Abstract**: With the advent of vehicles equipped with advanced driver-assistance systems, such as adaptive cruise control (ACC) and other automated driving features, the potential for cyberattacks on these automated vehicles (AVs) has emerged. While overt attacks that force vehicles to collide may be easily identified, more insidious attacks, which only slightly alter driving behavior, can result in network-wide increases in congestion, fuel consumption, and even crash risk without being easily detected. To address the detection of such attacks, we first present a traffic model framework for three types of potential cyberattacks: malicious manipulation of vehicle control commands, false data injection attacks on sensor measurements, and denial-of-service (DoS) attacks. We then investigate the impacts of these attacks at both the individual vehicle (micro) and traffic flow (macro) levels. A novel generative adversarial network (GAN)-based anomaly detection model is proposed for real-time identification of such attacks using vehicle trajectory data. We provide numerical evidence {to demonstrate} the efficacy of our machine learning approach in detecting cyberattacks on ACC-equipped vehicles. The proposed method is compared against some recently proposed neural network models and observed to have higher accuracy in identifying anomalous driving behaviors of ACC vehicles.

摘要: 随着配备先进的驾驶员辅助系统的车辆的出现，如自适应巡航控制(ACC)和其他自动驾驶功能，针对这些自动车辆(AV)的网络攻击的可能性已经出现。虽然迫使车辆相撞的公开攻击可能很容易识别，但更隐蔽的攻击只会轻微改变驾驶行为，可能会导致网络范围内拥堵、燃油消耗甚至碰撞风险的增加，而不容易被发现。为了解决此类攻击的检测，我们首先提出了一个针对三种潜在网络攻击的流量模型框架：对车辆控制命令的恶意操纵、对传感器测量的虚假数据注入攻击和拒绝服务(DoS)攻击。然后，我们调查这些攻击在单个车辆(微观)和交通流量(宏观)两个层面上的影响。为利用车辆轨迹数据实时识别此类攻击，提出了一种基于产生式对抗网络(GAN)的异常检测模型。我们提供了数字证据，以证明我们的机器学习方法在检测针对配备了ACC的车辆的网络攻击方面的有效性。将该方法与最近提出的几种神经网络模型进行了比较，发现该方法在识别ACC车辆的异常驾驶行为方面具有更高的准确率。



## **19. Trust, but Verify: Robust Image Segmentation using Deep Learning**

信任，但要验证：使用深度学习的稳健图像分割 cs.CV

5 Pages, 8 Figures, conference

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16999v1) [paper-pdf](http://arxiv.org/pdf/2310.16999v1)

**Authors**: Fahim Ahmed Zaman, Xiaodong Wu, Weiyu Xu, Milan Sonka, Raghuraman Mudumbai

**Abstract**: We describe a method for verifying the output of a deep neural network for medical image segmentation that is robust to several classes of random as well as worst-case perturbations i.e. adversarial attacks. This method is based on a general approach recently developed by the authors called ``Trust, but Verify" wherein an auxiliary verification network produces predictions about certain masked features in the input image using the segmentation as an input. A well-designed auxiliary network will produce high-quality predictions when the input segmentations are accurate, but will produce low-quality predictions when the segmentations are incorrect. Checking the predictions of such a network with the original image allows us to detect bad segmentations. However, to ensure the verification method is truly robust, we need a method for checking the quality of the predictions that does not itself rely on a black-box neural network. Indeed, we show that previous methods for segmentation evaluation that do use deep neural regression networks are vulnerable to false negatives i.e. can inaccurately label bad segmentations as good. We describe the design of a verification network that avoids such vulnerability and present results to demonstrate its robustness compared to previous methods.

摘要: 我们描述了一种用于医学图像分割的深度神经网络输出的验证方法，该方法对几类随机和最坏情况的扰动，即对抗性攻击具有鲁棒性。这种方法基于作者最近开发的一种名为“信任，但验证”的通用方法，其中一个辅助验证网络使用分割作为输入，对输入图像中的某些掩蔽特征产生预测。一个设计良好的辅助网络在输入分割准确时会产生高质量的预测，但当分割不正确时会产生低质量的预测。用原始图像检查这种网络的预测可以让我们检测到错误的分割。然而，为了确保验证方法真正稳健，我们需要一种方法来检查预测的质量，该方法本身不依赖于黑盒神经网络。事实上，我们表明，以前使用深度神经回归网络的分割评估方法很容易出现假阴性，即可能不准确地将不良分割标记为良好分割。我们描述了一个避免这种漏洞的验证网络的设计，并给出了与以前方法相比的结果来证明它的健壮性。



## **20. Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks**

打破它，模仿它，修复它：通过产生类似人类的攻击来增强健壮性 cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16955v1) [paper-pdf](http://arxiv.org/pdf/2310.16955v1)

**Authors**: Aradhana Sinha, Ananth Balashankar, Ahmad Beirami, Thi Avrahami, Jilin Chen, Alex Beutel

**Abstract**: Real-world natural language processing systems need to be robust to human adversaries. Collecting examples of human adversaries for training is an effective but expensive solution. On the other hand, training on synthetic attacks with small perturbations - such as word-substitution - does not actually improve robustness to human adversaries. In this paper, we propose an adversarial training framework that uses limited human adversarial examples to generate more useful adversarial examples at scale. We demonstrate the advantages of this system on the ANLI and hate speech detection benchmark datasets - both collected via an iterative, adversarial human-and-model-in-the-loop procedure. Compared to training only on observed human attacks, also training on our synthetic adversarial examples improves model robustness to future rounds. In ANLI, we see accuracy gains on the current set of attacks (44.1%$\,\to\,$50.1%) and on two future unseen rounds of human generated attacks (32.5%$\,\to\,$43.4%, and 29.4%$\,\to\,$40.2%). In hate speech detection, we see AUC gains on current attacks (0.76 $\to$ 0.84) and a future round (0.77 $\to$ 0.79). Attacks from methods that do not learn the distribution of existing human adversaries, meanwhile, degrade robustness.

摘要: 真实世界的自然语言处理系统需要对人类对手具有健壮性。收集人类对手的例子进行训练是一种有效但代价高昂的解决方案。另一方面，对带有小扰动的合成攻击进行训练--例如单词替换--实际上并不能提高对人类对手的稳健性。在本文中，我们提出了一个对抗性训练框架，该框架使用有限的人类对抗性实例来生成更多规模上有用的对抗性实例。我们在安利和仇恨语音检测基准数据集上展示了该系统的优势-这两个基准数据都是通过迭代、对抗性的人和模型在循环中收集的过程收集的。与只对观察到的人类攻击进行训练相比，对我们的合成对抗性例子的训练也提高了模型对未来几轮的稳健性。在安利，我们看到当前的一组攻击(44.1%$\，\to\，$50.1%)和未来两轮看不见的人为攻击(32.5%$，\to\，$43.4%和29.4%$，\to\，$40.2%)的准确率有所提高。在仇恨言论检测方面，我们看到AUC在当前攻击(0.76美元至0.84美元)和未来一轮攻击(0.77美元至0.79美元)上获得了收益。同时，来自不了解现有人类对手分布的方法的攻击会降低健壮性。



## **21. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

使用Pre-Softmax分数的归因方法的漏洞 cs.LG

7 pages, 5 figures

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2307.03305v2) [paper-pdf](http://arxiv.org/pdf/2307.03305v2)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.

摘要: 我们讨论了一个漏洞，涉及一类属性方法，用于解释用作分类器的卷积神经网络的输出。众所周知，这种类型的网络容易受到敌意攻击，在这种攻击中，输入的不知不觉的扰动可能会改变模型的输出。相反，这里我们关注的是在不改变模型输出的情况下，模型中的微小修改可能会对归因方法造成的影响。



## **22. Dual Defense: Adversarial, Traceable, and Invisible Robust Watermarking against Face Swapping**

双重防御：抗人脸交换的对抗性、可追踪性和隐蔽性数字水印 cs.CV

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16540v1) [paper-pdf](http://arxiv.org/pdf/2310.16540v1)

**Authors**: Yunming Zhang, Dengpan Ye, Caiyun Xie, Long Tang, Chuanxi Chen, Ziyi Liu, Jiacheng Deng

**Abstract**: The malicious applications of deep forgery, represented by face swapping, have introduced security threats such as misinformation dissemination and identity fraud. While some research has proposed the use of robust watermarking methods to trace the copyright of facial images for post-event traceability, these methods cannot effectively prevent the generation of forgeries at the source and curb their dissemination. To address this problem, we propose a novel comprehensive active defense mechanism that combines traceability and adversariality, called Dual Defense. Dual Defense invisibly embeds a single robust watermark within the target face to actively respond to sudden cases of malicious face swapping. It disrupts the output of the face swapping model while maintaining the integrity of watermark information throughout the entire dissemination process. This allows for watermark extraction at any stage of image tracking for traceability. Specifically, we introduce a watermark embedding network based on original-domain feature impersonation attack. This network learns robust adversarial features of target facial images and embeds watermarks, seeking a well-balanced trade-off between watermark invisibility, adversariality, and traceability through perceptual adversarial encoding strategies. Extensive experiments demonstrate that Dual Defense achieves optimal overall defense success rates and exhibits promising universality in anti-face swapping tasks and dataset generalization ability. It maintains impressive adversariality and traceability in both original and robust settings, surpassing current forgery defense methods that possess only one of these capabilities, including CMUA-Watermark, Anti-Forgery, FakeTagger, or PGD methods.

摘要: 以人脸互换为代表的深度伪造恶意应用，带来了虚假信息传播、身份诈骗等安全威胁。虽然一些研究提出了使用稳健的水印方法来追踪人脸图像的版权，以实现事件后的可追溯性，但这些方法不能有效地从源头上防止伪造的产生和遏制其传播。针对这一问题，我们提出了一种结合可追溯性和对抗性的新型综合主动防御机制，称为双重防御。双重防御在目标人脸内隐形嵌入单个稳健水印，主动应对恶意换脸的突发案例。它破坏了人脸交换模型的输出，同时在整个传播过程中保持了水印信息的完整性。这允许在图像跟踪的任何阶段提取水印以实现可追踪性。具体来说，我们提出了一种基于原始域特征模仿攻击的水印嵌入网络。该网络学习目标人脸图像的健壮对抗性特征并嵌入水印，通过感知对抗性编码策略在水印不可见性、对抗性和可追踪性之间寻求良好的平衡。大量实验表明，DUAL DARTY具有最优的整体防御成功率，在反人脸交换任务和数据集泛化能力方面表现出良好的普适性。它在原始和健壮的设置中都保持了令人印象深刻的对抗性和可追溯性，超过了目前仅具有其中一种功能的伪造防御方法，包括CMUA-水印、防伪造、伪造标记或PGD方法。



## **23. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

量子分类器多分类任务的普遍对抗性扰动 quant-ph

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2306.11974v3) [paper-pdf](http://arxiv.org/pdf/2306.11974v3)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.

摘要: 量子对抗机器学习是一个新兴的研究领域，它研究量子学习系统对对抗扰动的脆弱性，并开发可能的防御策略。量子通用对抗性扰动是一种微小的扰动，它可以使不同的输入样本变成可能欺骗给定量子分类器的对抗性例子。这是一个很少被研究但值得研究的领域，因为普遍的扰动可能会在很大程度上简化恶意攻击，给量子机器学习模型造成意想不到的破坏。在这篇文章中，我们向前迈进了一步，探索了异质分类任务背景下的量子普适微扰。特别是，我们发现，在两个不同的分类任务上获得几乎最先进的精度的量子分类器都可能最终被一个精心设计的普遍扰动所欺骗。这一结果通过设计良好的弹性权重巩固方法的量子连续学习模型来避免灾难性遗忘，以及来自手写数字和医学MRI图像的现实生活中的异质数据集得到了明确的证明。我们的结果提供了一种简单而有效的方法来产生对异类分类任务的普遍扰动，从而为未来的量子学习技术提供了有价值的指导。



## **24. Impartial Games: A Challenge for Reinforcement Learning**

公平博弈：强化学习的挑战 cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2205.12787v2) [paper-pdf](http://arxiv.org/pdf/2205.12787v2)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: AlphaZero-style reinforcement learning (RL) algorithms excel in various board games but face challenges with impartial games, where players share pieces. We present a concrete example of a game - namely the children's game of nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar reinforcement learning algorithms.   Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that nim can be learned on small boards, but AlphaZero-style algorithms learning dramatically slows down when the board size increases. Intuitively, the difference between impartial games like nim and partisan games like Chess and Go can be explained by the fact that if a tiny amount of noise is added to the system (e.g. if a small part of the board is covered), for impartial games, it is typically not possible to predict whether the position is good or bad (won or lost). There is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out configuration typically provides abundant or at least non-trifle information about the value of the fully uncovered position.

摘要: AlphaZero风格的强化学习(RL)算法在各种棋盘游戏中表现出色，但在公平的游戏中面临挑战，玩家分享棋子。我们提供了一个具体的游戏示例--即Nim的儿童游戏--以及其他公平的游戏，这些游戏似乎是AlphaZero风格和类似的强化学习算法的绊脚石。我们的发现与最近的研究一致，这些研究表明AlphaZero风格的算法容易受到对手攻击和对手扰动，这表明在所有合法国家学习掌握游戏都是困难的。我们表明，NIM可以在小电路板上学习，但AlphaZero风格的算法学习随着电路板大小的增加而显著减慢。直觉上，像尼姆这样的公正游戏与像国际象棋和围棋这样的党派游戏之间的区别可以用这样一个事实来解释：如果在系统中添加少量的噪音(例如，如果棋盘的一小部分被覆盖)，对于公正的游戏，通常无法预测形势是好是坏(赢或输)。部分消隐位置的可见部分与其正确评估之间的相关性通常为零。这种情况与党派游戏形成了鲜明对比，在党派游戏中，部分空白的配置通常会提供大量或至少不是无关紧要的信息，以了解完全暴露的头寸的价值。



## **25. Defense Against Model Extraction Attacks on Recommender Systems**

推荐系统对模型提取攻击的防御 cs.LG

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2310.16335v1) [paper-pdf](http://arxiv.org/pdf/2310.16335v1)

**Authors**: Sixiao Zhang, Hongzhi Yin, Hongxu Chen, Cheng Long

**Abstract**: The robustness of recommender systems has become a prominent topic within the research community. Numerous adversarial attacks have been proposed, but most of them rely on extensive prior knowledge, such as all the white-box attacks or most of the black-box attacks which assume that certain external knowledge is available. Among these attacks, the model extraction attack stands out as a promising and practical method, involving training a surrogate model by repeatedly querying the target model. However, there is a significant gap in the existing literature when it comes to defending against model extraction attacks on recommender systems. In this paper, we introduce Gradient-based Ranking Optimization (GRO), which is the first defense strategy designed to counter such attacks. We formalize the defense as an optimization problem, aiming to minimize the loss of the protected target model while maximizing the loss of the attacker's surrogate model. Since top-k ranking lists are non-differentiable, we transform them into swap matrices which are instead differentiable. These swap matrices serve as input to a student model that emulates the surrogate model's behavior. By back-propagating the loss of the student model, we obtain gradients for the swap matrices. These gradients are used to compute a swap loss, which maximizes the loss of the student model. We conducted experiments on three benchmark datasets to evaluate the performance of GRO, and the results demonstrate its superior effectiveness in defending against model extraction attacks.

摘要: 推荐系统的健壮性已经成为研究界的一个重要话题。已经提出了许多对抗性攻击，但其中大多数依赖于广泛的先验知识，例如所有的白盒攻击或大多数假设某些外部知识可用的黑盒攻击。在这些攻击中，模型提取攻击是一种很有前途的实用方法，它通过反复查询目标模型来训练代理模型。然而，现有文献在防御针对推荐系统的模型提取攻击方面存在着很大的差距。在本文中，我们介绍了基于梯度的排名优化(GRO)，这是第一种针对此类攻击而设计的防御策略。我们将防御形式化为一个优化问题，目标是最小化受保护的目标模型的损失，同时最大化攻击者的代理模型的损失。因为top-k排名列表是不可微的，所以我们将它们转换为交换矩阵，而交换矩阵是可微的。这些交换矩阵用作模拟代理模型的行为的学生模型的输入。通过反向传播学生模型的损失，我们得到了交换矩阵的梯度。这些梯度被用来计算掉期损失，从而使学生模型的损失最大化。我们在三个基准数据集上进行了实验，评估了GRO的性能，结果表明它在抵御模型提取攻击方面具有优越的有效性。



## **26. UPTON: Preventing Authorship Leakage from Public Text Release via Data Poisoning**

Upton：通过数据中毒防止公开文本发布的作者泄露 cs.CY

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2211.09717v3) [paper-pdf](http://arxiv.org/pdf/2211.09717v3)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: Consider a scenario where an author-e.g., activist, whistle-blower, with many public writings wishes to write "anonymously" when attackers may have already built an authorship attribution (AA) model based off of public writings including those of the author. To enable her wish, we ask a question "Can one make the publicly released writings, T, unattributable so that AA models trained on T cannot attribute its authorship well?" Toward this question, we present a novel solution, UPTON, that exploits black-box data poisoning methods to weaken the authorship features in training samples and make released texts unlearnable. It is different from previous obfuscation works-e.g., adversarial attacks that modify test samples or backdoor works that only change the model outputs when triggering words occur. Using four authorship datasets (IMDb10, IMDb64, Enron, and WJO), we present empirical validation where UPTON successfully downgrades the accuracy of AA models to the impractical level (~35%) while keeping texts still readable (semantic similarity>0.9). UPTON remains effective to AA models that are already trained on available clean writings of authors.

摘要: 考虑这样一种场景，当攻击者可能已经根据包括作者的公开作品构建了作者归属(AA)模型时，具有许多公开作品的作者--例如，活动家、告密者--希望“匿名”写作。为了实现她的愿望，我们问了一个问题：“我们能不能让公开发布的作品T无法归类，这样接受过T训练的AA模型就不能很好地确定它的作者是谁？”它不同于以前的混淆工作-例如，修改测试样本的对抗性攻击，或者只在触发单词出现时更改模型输出的后门工作。使用四个作者数据集(IMDb10、IMDb64、Enron和WJO)，我们提供了经验验证，其中Upton成功地将AA模型的准确率降低到不切实际的水平(~35%)，同时保持文本的可读性(语义相似度>0.9)。厄普顿仍然对那些已经接受过培训的AA模型有效，这些模型已经接受了作者可用的干净作品的培训。



## **27. Database Matching Under Noisy Synchronization Errors**

噪声同步误差下的数据库匹配 cs.IT

**SubmitDate**: 2023-10-25    [abs](http://arxiv.org/abs/2301.06796v2) [paper-pdf](http://arxiv.org/pdf/2301.06796v2)

**Authors**: Serhat Bakirtas, Elza Erkip

**Abstract**: The re-identification or de-anonymization of users from anonymized data through matching with publicly available correlated user data has raised privacy concerns, leading to the complementary measure of obfuscation in addition to anonymization. Recent research provides a fundamental understanding of the conditions under which privacy attacks, in the form of database matching, are successful in the presence of obfuscation. Motivated by synchronization errors stemming from the sampling of time-indexed databases, this paper presents a unified framework considering both obfuscation and synchronization errors and investigates the matching of databases under noisy entry repetitions. By investigating different structures for the repetition pattern, replica detection and seeded deletion detection algorithms are devised and sufficient and necessary conditions for successful matching are derived. Finally, the impacts of some variations of the underlying assumptions, such as the adversarial deletion model, seedless database matching, and zero-rate regime, on the results are discussed. Overall, our results provide insights into the privacy-preserving publication of anonymized and obfuscated time-indexed data as well as the closely related problem of the capacity of synchronization channels.

摘要: 通过与公开可用的相关用户数据进行匹配来从匿名化数据中重新识别或去匿名化用户已经引起了隐私问题，导致除了匿名化之外的混淆的补充措施。最近的研究提供了一个基本的了解的条件下，隐私攻击，在数据库匹配的形式，是成功的混淆的存在。针对时间索引数据库采样过程中产生的同步错误，提出了一个同时考虑混淆和同步错误的统一框架，并研究了噪声条目重复下的数据库匹配问题.通过研究重复模式的不同结构，设计了副本检测和种子删除检测算法，并推导了成功匹配的充分必要条件。最后，一些变化的基本假设，如敌对删除模型，无籽数据库匹配，和零利率制度，对结果的影响进行了讨论。总的来说，我们的研究结果提供了深入的隐私保护的匿名和模糊的时间索引数据的发布，以及同步通道的容量密切相关的问题。



## **28. A Generative Framework for Low-Cost Result Validation of Outsourced Machine Learning Tasks**

外包机器学习任务低成本结果验证的产生式框架 cs.CR

15 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2304.00083v2) [paper-pdf](http://arxiv.org/pdf/2304.00083v2)

**Authors**: Abhinav Kumar, Miguel A. Guirao Aguilera, Reza Tourani, Satyajayant Misra

**Abstract**: The growing popularity of Machine Learning (ML) has led to its deployment in various sensitive domains, which has resulted in significant research focused on ML security and privacy. However, in some applications, such as autonomous driving, integrity verification of the outsourced ML workload is more critical--a facet that has not received much attention. Existing solutions, such as multi-party computation and proof-based systems, impose significant computation overhead, which makes them unfit for real-time applications. We propose Fides, a novel framework for real-time validation of outsourced ML workloads. Fides features a novel and efficient distillation technique--Greedy Distillation Transfer Learning--that dynamically distills and fine-tunes a space and compute-efficient verification model for verifying the corresponding service model while running inside a trusted execution environment. Fides features a client-side attack detection model that uses statistical analysis and divergence measurements to identify, with a high likelihood, if the service model is under attack. Fides also offers a re-classification functionality that predicts the original class whenever an attack is identified. We devised a generative adversarial network framework for training the attack detection and re-classification models. The evaluation shows that Fides achieves an accuracy of up to 98% for attack detection and 94% for re-classification.

摘要: 机器学习(ML)的日益流行导致了它在各种敏感领域的部署，这导致了对ML安全和隐私的大量研究。然而，在一些应用中，例如自动驾驶，外包的ML工作负载的完整性验证更关键--这一方面没有得到太多关注。现有的解决方案，如多方计算和基于证明的系统，带来了巨大的计算开销，这使得它们不适合实时应用。我们提出了一种新的实时验证外包ML工作负载的框架FIDS。FIDS的特点是一种新颖而高效的蒸馏技术--贪婪蒸馏转移学习--它动态地提取和微调空间和计算效率高的验证模型，以便在可信执行环境中运行时验证相应的服务模型。FIDS具有客户端攻击检测模型，该模型使用统计分析和分歧测量来识别服务模型是否受到攻击的可能性很高。FIDS还提供了重新分类功能，该功能可以在识别攻击时预测原始类别。我们设计了一个生成式对抗性网络框架来训练攻击检测和重分类模型。评估表明，FIDS对攻击检测的准确率高达98%，对重分类的准确率高达94%。



## **29. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

FLTrojan：通过选择性权重篡改对联邦语言模型进行隐私泄露攻击 cs.CR

22 pages (including bibliography and Appendix), Submitted to USENIX  Security '24

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.16152v1) [paper-pdf](http://arxiv.org/pdf/2310.16152v1)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) is becoming a key component in many technology-based applications including language modeling -- where individual FL participants often have privacy-sensitive text data in their local datasets. However, realizing the extent of privacy leakage in federated language models is not straightforward and the existing attacks only intend to extract data regardless of how sensitive or naive it is. To fill this gap, in this paper, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other user in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 70% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.

摘要: 联合学习(FL)正在成为包括语言建模在内的许多基于技术的应用程序中的关键组件--在这些应用程序中，个别FL参与者通常在其本地数据集中拥有隐私敏感的文本数据。然而，在联邦语言模型中实现隐私泄露的程度并不简单，现有的攻击只打算提取数据，无论它是多么敏感或幼稚。为了填补这一空白，在本文中，我们介绍了关于从联邦语言模型泄露隐私敏感用户数据的两个新发现。首先，我们做了一个关键的观察，在FL的中间轮中的模型快照比最终训练的模型会导致更大的隐私泄露。其次，我们发现，通过篡改模型的选择性权重可能会加剧隐私泄露，这些选择性权重专门负责记忆敏感的训练数据。我们展示了恶意客户端如何在没有任何服务器合作的情况下泄露FL中其他用户的隐私敏感数据。该算法的成员关系推理召回率提高了29%，私有数据重构效率高达70%，明显优于现有攻击方法。



## **30. Diffusion-Based Adversarial Purification for Speaker Verification**

基于扩散的对抗性净化说话人确认 eess.AS

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14270v2) [paper-pdf](http://arxiv.org/pdf/2310.14270v2)

**Authors**: Yibo Bai, Xiao-Lei Zhang

**Abstract**: Recently, automatic speaker verification (ASV) based on deep learning is easily contaminated by adversarial attacks, which is a new type of attack that injects imperceptible perturbations to audio signals so as to make ASV produce wrong decisions. This poses a significant threat to the security and reliability of ASV systems. To address this issue, we propose a Diffusion-Based Adversarial Purification (DAP) method that enhances the robustness of ASV systems against such adversarial attacks. Our method leverages a conditional denoising diffusion probabilistic model to effectively purify the adversarial examples and mitigate the impact of perturbations. DAP first introduces controlled noise into adversarial examples, and then performs a reverse denoising process to reconstruct clean audio. Experimental results demonstrate the efficacy of the proposed DAP in enhancing the security of ASV and meanwhile minimizing the distortion of the purified audio signals.

摘要: 目前，基于深度学习的自动说话人确认(ASV)容易受到敌意攻击的污染，这是一种新型的攻击方式，通过在音频信号中注入不可察觉的扰动，从而使ASV产生错误的判决。这对ASV系统的安全性和可靠性构成了重大威胁。为了解决这个问题，我们提出了一种基于扩散的对抗性净化(DAP)方法，该方法增强了ASV系统对此类对抗性攻击的健壮性。我们的方法利用条件去噪扩散概率模型来有效地净化对抗性例子，并减轻扰动的影响。DAP首先将受控噪声引入到敌意样本中，然后进行反向去噪处理来重建干净的音频。实验结果表明，该算法在增强ASV安全性的同时，最大限度地减少了音频信号的失真。



## **31. A Survey on LLM-generated Text Detection: Necessity, Methods, and Future Directions**

LLM生成的文本检测：必要性、方法和未来发展方向综述 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.14724v2) [paper-pdf](http://arxiv.org/pdf/2310.14724v2)

**Authors**: Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Derek F. Wong, Lidia S. Chao

**Abstract**: The powerful ability to understand, follow, and generate complex language emerging from large language models (LLMs) makes LLM-generated text flood many areas of our daily lives at an incredible speed and is widely accepted by humans. As LLMs continue to expand, there is an imperative need to develop detectors that can detect LLM-generated text. This is crucial to mitigate potential misuse of LLMs and safeguard realms like artistic expression and social networks from harmful influence of LLM-generated content. The LLM-generated text detection aims to discern if a piece of text was produced by an LLM, which is essentially a binary classification task. The detector techniques have witnessed notable advancements recently, propelled by innovations in watermarking techniques, zero-shot methods, fine-turning LMs methods, adversarial learning methods, LLMs as detectors, and human-assisted methods. In this survey, we collate recent research breakthroughs in this area and underscore the pressing need to bolster detector research. We also delve into prevalent datasets, elucidating their limitations and developmental requirements. Furthermore, we analyze various LLM-generated text detection paradigms, shedding light on challenges like out-of-distribution problems, potential attacks, and data ambiguity. Conclusively, we highlight interesting directions for future research in LLM-generated text detection to advance the implementation of responsible artificial intelligence (AI). Our aim with this survey is to provide a clear and comprehensive introduction for newcomers while also offering seasoned researchers a valuable update in the field of LLM-generated text detection. The useful resources are publicly available at: https://github.com/NLP2CT/LLM-generated-Text-Detection.

摘要: 大型语言模型(LLM)强大的理解、跟踪和生成复杂语言的能力使得LLM生成的文本以令人难以置信的速度涌入我们日常生活的许多领域，并被人类广泛接受。随着LLMS的不断扩展，迫切需要开发能够检测LLM生成的文本的检测器。这对于减少LLM的潜在滥用以及保护艺术表达和社交网络等领域免受LLM生成的内容的有害影响至关重要。LLM生成的文本检测旨在识别一段文本是否由LLM生成，这本质上是一项二进制分类任务。最近，在水印技术、零镜头方法、精细旋转LMS方法、对抗性学习方法、作为检测器的LLMS以及人工辅助方法的创新的推动下，检测器技术有了显著的进步。在这次调查中，我们整理了这一领域的最新研究突破，并强调了支持探测器研究的迫切需要。我们还深入研究了流行的数据集，阐明了它们的局限性和发展需求。此外，我们分析了各种LLM生成的文本检测范例，揭示了诸如分发外问题、潜在攻击和数据歧义等挑战。最后，我们指出了未来在LLM生成的文本检测方面的有趣研究方向，以推进负责任人工智能(AI)的实施。我们这次调查的目的是为新手提供一个清晰而全面的介绍，同时也为经验丰富的研究人员提供在LLM生成的文本检测领域的有价值的更新。这些有用的资源可在以下网址公开获得：https://github.com/NLP2CT/LLM-generated-Text-Detection.



## **32. Momentum Gradient-based Untargeted Attack on Hypergraph Neural Networks**

基于动量梯度的超图神经网络无目标攻击 cs.LG

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15656v1) [paper-pdf](http://arxiv.org/pdf/2310.15656v1)

**Authors**: Yang Chen, Stjepan Picek, Zhonglin Ye, Zhaoyang Wang, Haixing Zhao

**Abstract**: Hypergraph Neural Networks (HGNNs) have been successfully applied in various hypergraph-related tasks due to their excellent higher-order representation capabilities. Recent works have shown that deep learning models are vulnerable to adversarial attacks. Most studies on graph adversarial attacks have focused on Graph Neural Networks (GNNs), and the study of adversarial attacks on HGNNs remains largely unexplored. In this paper, we try to reduce this gap. We design a new HGNNs attack model for the untargeted attack, namely MGHGA, which focuses on modifying node features. We consider the process of HGNNs training and use a surrogate model to implement the attack before hypergraph modeling. Specifically, MGHGA consists of two parts: feature selection and feature modification. We use a momentum gradient mechanism to choose the attack node features in the feature selection module. In the feature modification module, we use two feature generation approaches (direct modification and sign gradient) to enable MGHGA to be employed on discrete and continuous datasets. We conduct extensive experiments on five benchmark datasets to validate the attack performance of MGHGA in the node and the visual object classification tasks. The results show that MGHGA improves performance by an average of 2% compared to the than the baselines.

摘要: 超图神经网络(HGNN)因其优良的高阶表示能力而被成功地应用于各种与超图相关的任务中。最近的研究表明，深度学习模型容易受到敌意攻击。大多数关于图对抗攻击的研究都集中在图神经网络(GNN)上，而对HGNN上的对抗攻击的研究还很少。在本文中，我们试图缩小这一差距。针对非定向攻击，我们设计了一种新的HGNN攻击模型--MGHGA，该模型侧重于修改节点特征。我们考虑了HGNN的训练过程，并在超图建模之前使用代理模型来实现攻击。具体而言，MGHGA由特征选择和特征修改两部分组成。在特征选择模块中，我们使用动量梯度机制来选择攻击节点特征。在特征修改模块中，我们使用了两种特征生成方法(直接修改和符号梯度)，使得MGHGA可以在离散和连续的数据集上使用。我们在五个基准数据集上进行了大量的实验，以验证MGHGA在节点和视觉对象分类任务中的攻击性能。结果表明，与基线相比，MGHGA的性能平均提高了2%。



## **33. Deceptive Fairness Attacks on Graphs via Meta Learning**

基于元学习的图的欺骗性公平攻击 cs.LG

23 pages, 11 tables

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15653v1) [paper-pdf](http://arxiv.org/pdf/2310.15653v1)

**Authors**: Jian Kang, Yinglong Xia, Ross Maciejewski, Jiebo Luo, Hanghang Tong

**Abstract**: We study deceptive fairness attacks on graphs to answer the following question: How can we achieve poisoning attacks on a graph learning model to exacerbate the bias deceptively? We answer this question via a bi-level optimization problem and propose a meta learning-based framework named FATE. FATE is broadly applicable with respect to various fairness definitions and graph learning models, as well as arbitrary choices of manipulation operations. We further instantiate FATE to attack statistical parity and individual fairness on graph neural networks. We conduct extensive experimental evaluations on real-world datasets in the task of semi-supervised node classification. The experimental results demonstrate that FATE could amplify the bias of graph neural networks with or without fairness consideration while maintaining the utility on the downstream task. We hope this paper provides insights into the adversarial robustness of fair graph learning and can shed light on designing robust and fair graph learning in future studies.

摘要: 我们研究图上的欺骗性公平攻击，以回答以下问题：我们如何在图学习模型上实现中毒攻击，以欺骗性地加剧偏差？我们通过一个双层优化问题来回答这个问题，并提出了一个基于元学习的框架Fate。Fate广泛适用于各种公平性定义和图学习模型，以及任意选择的操作操作。我们进一步实例化命运来攻击图神经网络上的统计等价性和个体公平性。在半监督节点分类任务中，我们在真实数据集上进行了广泛的实验评估。实验结果表明，无论是否考虑公平性，Fate都能放大图神经网络的偏差，同时保持对下游任务的效用。我们希望本文能为公平图学习的对抗稳健性研究提供帮助，并在未来的研究中为设计稳健的公平图学习提供参考。



## **34. Facial Data Minimization: Shallow Model as Your Privacy Filter**

面部数据最小化：浅层模型作为您的隐私过滤器 cs.CR

14 pages, 11 figures

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15590v1) [paper-pdf](http://arxiv.org/pdf/2310.15590v1)

**Authors**: Yuwen Pu, Jiahao Chen, Jiayu Pan, Hao li, Diqun Yan, Xuhong Zhang, Shouling Ji

**Abstract**: Face recognition service has been used in many fields and brings much convenience to people. However, once the user's facial data is transmitted to a service provider, the user will lose control of his/her private data. In recent years, there exist various security and privacy issues due to the leakage of facial data. Although many privacy-preserving methods have been proposed, they usually fail when they are not accessible to adversaries' strategies or auxiliary data. Hence, in this paper, by fully considering two cases of uploading facial images and facial features, which are very typical in face recognition service systems, we proposed a data privacy minimization transformation (PMT) method. This method can process the original facial data based on the shallow model of authorized services to obtain the obfuscated data. The obfuscated data can not only maintain satisfactory performance on authorized models and restrict the performance on other unauthorized models but also prevent original privacy data from leaking by AI methods and human visual theft. Additionally, since a service provider may execute preprocessing operations on the received data, we also propose an enhanced perturbation method to improve the robustness of PMT. Besides, to authorize one facial image to multiple service models simultaneously, a multiple restriction mechanism is proposed to improve the scalability of PMT. Finally, we conduct extensive experiments and evaluate the effectiveness of the proposed PMT in defending against face reconstruction, data abuse, and face attribute estimation attacks. These experimental results demonstrate that PMT performs well in preventing facial data abuse and privacy leakage while maintaining face recognition accuracy.

摘要: 人脸识别服务已经在许多领域得到了应用，给人们带来了极大的便利。然而，一旦用户的面部数据被传输到服务提供商，用户将失去对他/她的私人数据的控制。近年来，由于人脸数据的泄露，存在着各种各样的安全和隐私问题。虽然已经提出了许多隐私保护方法，但当对手的策略或辅助数据无法访问时，这些方法通常会失败。因此，在本文中，通过充分考虑人脸识别服务系统中非常典型的两种上传人脸图像和人脸特征的情况，提出了一种数据隐私最小化转换(PMT)方法。该方法可以基于授权服务的浅模型对原始人脸数据进行处理，得到混淆后的数据。混淆后的数据不仅可以在授权模型上保持满意的性能，在其他非授权模型上也可以限制性能，还可以防止AI方法泄露原始隐私数据和人类视觉窃取。此外，由于服务提供商可以对接收到的数据执行预处理操作，我们还提出了一种增强的扰动方法来提高PMT的稳健性。此外，为了将一幅人脸图像同时授权给多个服务模型，提出了一种多约束机制来提高PMT的可扩展性。最后，我们进行了大量的实验，评估了提出的PMT在抵抗人脸重建、数据滥用和人脸属性估计攻击方面的有效性。这些实验结果表明，PMT在保持人脸识别准确率的同时，很好地防止了人脸数据的滥用和隐私泄露。



## **35. LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked**

LLM自卫：通过自我检查，LLM知道自己被骗了 cs.CL

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2308.07308v3) [paper-pdf](http://arxiv.org/pdf/2308.07308v3)

**Authors**: Mansi Phute, Alec Helbling, Matthew Hull, ShengYun Peng, Sebastian Szyller, Cory Cornelius, Duen Horng Chau

**Abstract**: Large language models (LLMs) are popular for high-quality text generation but can produce harmful content, even when aligned with human values through reinforcement learning. Adversarial prompts can bypass their safety measures. We propose LLM Self Defense, a simple approach to defend against these attacks by having an LLM screen the induced responses. Our method does not require any fine-tuning, input preprocessing, or iterative output generation. Instead, we incorporate the generated content into a pre-defined prompt and employ another instance of an LLM to analyze the text and predict whether it is harmful. We test LLM Self Defense on GPT 3.5 and Llama 2, two of the current most prominent LLMs against various types of attacks, such as forcefully inducing affirmative responses to prompts and prompt engineering attacks. Notably, LLM Self Defense succeeds in reducing the attack success rate to virtually 0 using both GPT 3.5 and Llama 2.

摘要: 大型语言模型(LLM)对于高质量的文本生成很受欢迎，但可能会产生有害的内容，即使通过强化学习与人类的价值观保持一致。对抗性提示可以绕过它们的安全措施。我们提出了LLM自卫，这是一种通过让LLM筛选诱导响应来防御这些攻击的简单方法。我们的方法不需要任何微调、输入预处理或迭代输出生成。相反，我们将生成的内容合并到预定义的提示中，并使用LLM的另一个实例来分析文本并预测它是否有害。我们在GPT 3.5和Llama 2上测试了LLM自卫，这两种当前最著名的LLM针对各种类型的攻击，例如对提示的强制诱导肯定反应和提示工程攻击。值得注意的是，LLM自卫成功地使用GPT 3.5和Llama 2将攻击成功率降低到几乎为0。



## **36. Fast Propagation is Better: Accelerating Single-Step Adversarial Training via Sampling Subnetworks**

快速传播更好：通过抽样子网络加速单步对抗训练 cs.CV

**SubmitDate**: 2023-10-24    [abs](http://arxiv.org/abs/2310.15444v1) [paper-pdf](http://arxiv.org/pdf/2310.15444v1)

**Authors**: Xiaojun Jia, Jianshu Li, Jindong Gu, Yang Bai, Xiaochun Cao

**Abstract**: Adversarial training has shown promise in building robust models against adversarial examples. A major drawback of adversarial training is the computational overhead introduced by the generation of adversarial examples. To overcome this limitation, adversarial training based on single-step attacks has been explored. Previous work improves the single-step adversarial training from different perspectives, e.g., sample initialization, loss regularization, and training strategy. Almost all of them treat the underlying model as a black box. In this work, we propose to exploit the interior building blocks of the model to improve efficiency. Specifically, we propose to dynamically sample lightweight subnetworks as a surrogate model during training. By doing this, both the forward and backward passes can be accelerated for efficient adversarial training. Besides, we provide theoretical analysis to show the model robustness can be improved by the single-step adversarial training with sampled subnetworks. Furthermore, we propose a novel sampling strategy where the sampling varies from layer to layer and from iteration to iteration. Compared with previous methods, our method not only reduces the training cost but also achieves better model robustness. Evaluations on a series of popular datasets demonstrate the effectiveness of the proposed FB-Better. Our code has been released at https://github.com/jiaxiaojunQAQ/FP-Better.

摘要: 对抗性训练在建立针对对抗性例子的健壮模型方面显示出了希望。对抗性训练的一个主要缺点是产生对抗性例子所带来的计算开销。为了克服这一局限性，基于单步攻击的对抗性训练被探索出来。以往的工作从样本初始化、损失正则化、训练策略等不同角度对单步对抗性训练进行了改进。几乎所有人都将基础模型视为一个黑匣子。在这项工作中，我们建议利用模型的内部构建块来提高效率。具体地说，我们提出了在训练过程中动态采样轻型子网络作为代理模型。通过这样做，向前和向后的传球都可以加快速度，以进行有效的对抗性训练。此外，我们还给出了理论分析，表明利用采样子网络进行单步对抗性训练可以提高模型的稳健性。此外，我们还提出了一种新的采样策略，即采样层次分明，迭代不同。与以前的方法相比，我们的方法不仅降低了训练成本，而且获得了更好的模型鲁棒性。在一系列流行数据集上的评估表明，所提出的FB-Better算法是有效的。我们的代码已在https://github.com/jiaxiaojunQAQ/FP-Better.发布



## **37. Unsupervised Federated Learning: A Federated Gradient EM Algorithm for Heterogeneous Mixture Models with Robustness against Adversarial Attacks**

无监督联合学习：一种适用于异质混合模型的联合梯度EM算法 stat.ML

43 pages, 1 figure

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15330v1) [paper-pdf](http://arxiv.org/pdf/2310.15330v1)

**Authors**: Ye Tian, Haolei Weng, Yang Feng

**Abstract**: While supervised federated learning approaches have enjoyed significant success, the domain of unsupervised federated learning remains relatively underexplored. In this paper, we introduce a novel federated gradient EM algorithm designed for the unsupervised learning of mixture models with heterogeneous mixture proportions across tasks. We begin with a comprehensive finite-sample theory that holds for general mixture models, then apply this general theory on Gaussian Mixture Models (GMMs) and Mixture of Regressions (MoRs) to characterize the explicit estimation error of model parameters and mixture proportions. Our proposed federated gradient EM algorithm demonstrates several key advantages: adaptability to unknown task similarity, resilience against adversarial attacks on a small fraction of data sources, protection of local data privacy, and computational and communication efficiency.

摘要: 虽然有监督的联合学习方法已经取得了很大的成功，但无监督的联合学习领域仍然相对较少被探索。本文提出了一种新的联邦梯度EM算法，用于混合模型跨任务混合比例的无监督学习。我们从一个适用于一般混合模型的综合有限样本理论开始，然后将这个一般理论应用于高斯混合模型(GMM)和混合回归模型(MORS)来刻画模型参数和混合比例的显式估计误差。我们提出的联合梯度EM算法具有以下几个关键优点：对未知任务相似性的适应性、对一小部分数据源的恶意攻击的恢复能力、对本地数据隐私的保护以及计算和通信效率。



## **38. GRASP: Accelerating Shortest Path Attacks via Graph Attention**

GRAPH：通过图注意力加速最短路径攻击 cs.LG

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.07980v2) [paper-pdf](http://arxiv.org/pdf/2310.07980v2)

**Authors**: Zohair Shafi, Benjamin A. Miller, Ayan Chatterjee, Tina Eliassi-Rad, Rajmonda S. Caceres

**Abstract**: Recent advances in machine learning (ML) have shown promise in aiding and accelerating classical combinatorial optimization algorithms. ML-based speed ups that aim to learn in an end to end manner (i.e., directly output the solution) tend to trade off run time with solution quality. Therefore, solutions that are able to accelerate existing solvers while maintaining their performance guarantees, are of great interest. We consider an APX-hard problem, where an adversary aims to attack shortest paths in a graph by removing the minimum number of edges. We propose the GRASP algorithm: Graph Attention Accelerated Shortest Path Attack, an ML aided optimization algorithm that achieves run times up to 10x faster, while maintaining the quality of solution generated. GRASP uses a graph attention network to identify a smaller subgraph containing the combinatorial solution, thus effectively reducing the input problem size. Additionally, we demonstrate how careful representation of the input graph, including node features that correlate well with the optimization task, can highlight important structure in the optimization solution.

摘要: 机器学习(ML)的最新进展在辅助和加速经典组合优化算法方面显示出良好的前景。基于ML的加速旨在以端到端的方式学习(即直接输出解决方案)，往往会在运行时间和解决方案质量之间进行权衡。因此，能够在保持现有求解器性能保证的同时加速现有求解器的解决方案是非常有意义的。我们考虑APX-Hard问题，其中对手的目标是通过删除最少的边数来攻击图中的最短路径。我们提出了GRASH算法：图注意加速最短路径攻击，这是一种ML辅助优化算法，在保持所生成解的质量的情况下，运行时间最多可以提高10倍。GRASH使用图注意网络来识别包含组合解的较小的子图，从而有效地减少了输入问题的规模。此外，我们还演示了如何仔细表示输入图，包括与优化任务关联良好的节点特征，以突出优化解决方案中的重要结构。



## **39. AutoDAN: Automatic and Interpretable Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的自动和可解释的对抗性攻击 cs.CR

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15140v1) [paper-pdf](http://arxiv.org/pdf/2310.15140v1)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent work suggests that patching LLMs against these attacks is possible: manual jailbreak attacks are human-readable but often limited and public, making them easy to block; adversarial attacks generate gibberish prompts that can be detected using perplexity-based filters. In this paper, we show that these solutions may be too optimistic. We propose an interpretable adversarial attack, \texttt{AutoDAN}, that combines the strengths of both types of attacks. It automatically generates attack prompts that bypass perplexity-based filters while maintaining a high attack success rate like manual jailbreak attacks. These prompts are interpretable and diverse, exhibiting strategies commonly used in manual jailbreak attacks, and transfer better than their non-readable counterparts when using limited training data or a single proxy model. We also customize \texttt{AutoDAN}'s objective to leak system prompts, another jailbreak application not addressed in the adversarial attack literature. Our work provides a new way to red-team LLMs and to understand the mechanism of jailbreak attacks.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，修补LLM以抵御这些攻击是可能的：手动越狱攻击是人类可读的，但通常是有限的和公开的，使它们很容易被阻止；对抗性攻击生成胡言乱语的提示，可以使用基于困惑的过滤器检测到。在本文中，我们证明了这些解决方案可能过于乐观。我们提出了一种可解释的对抗性攻击，它结合了这两种攻击的优点。它自动生成攻击提示，绕过基于困惑的过滤器，同时保持较高的攻击成功率，如手动越狱攻击。这些提示是可解释的和多样化的，展示了手动越狱攻击中常用的策略，并且在使用有限的训练数据或单一代理模型时，传输效果比不可读的相应提示更好。我们还定制了S的目标来泄露系统提示，这是另一个在对抗性攻击文献中没有涉及的越狱应用。我们的工作为红队低层管理和理解越狱攻击的机制提供了一种新的途径。



## **40. On the Detection of Image-Scaling Attacks in Machine Learning**

机器学习中图像缩放攻击的检测研究 cs.CR

Accepted at ACSAC'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.15085v1) [paper-pdf](http://arxiv.org/pdf/2310.15085v1)

**Authors**: Erwin Quiring, Andreas Müller, Konrad Rieck

**Abstract**: Image scaling is an integral part of machine learning and computer vision systems. Unfortunately, this preprocessing step is vulnerable to so-called image-scaling attacks where an attacker makes unnoticeable changes to an image so that it becomes a new image after scaling. This opens up new ways for attackers to control the prediction or to improve poisoning and backdoor attacks. While effective techniques exist to prevent scaling attacks, their detection has not been rigorously studied yet. Consequently, it is currently not possible to reliably spot these attacks in practice.   This paper presents the first in-depth systematization and analysis of detection methods for image-scaling attacks. We identify two general detection paradigms and derive novel methods from them that are simple in design yet significantly outperform previous work. We demonstrate the efficacy of these methods in a comprehensive evaluation with all major learning platforms and scaling algorithms. First, we show that image-scaling attacks modifying the entire scaled image can be reliably detected even under an adaptive adversary. Second, we find that our methods provide strong detection performance even if only minor parts of the image are manipulated. As a result, we can introduce a novel protection layer against image-scaling attacks.

摘要: 图像缩放是机器学习和计算机视觉系统不可或缺的一部分。不幸的是，这一预处理步骤容易受到所谓的图像缩放攻击，即攻击者对图像进行不明显的更改，使其在缩放后成为新图像。虽然已经有了有效的技术来防止伸缩攻击，但它们的检测还没有得到严格的研究。因此，目前还不可能在实践中可靠地发现这些攻击。因此，我们可以引入一种新的保护层来抵御图像缩放攻击。



## **41. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

对抗性不变正则化增强对抗性对比学习 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2305.00374v2) [paper-pdf](http://arxiv.org/pdf/2305.00374v2)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) is a technique that enhances standard contrastive learning (SCL) by incorporating adversarial data to learn a robust representation that can withstand adversarial attacks and common corruptions without requiring costly annotations. To improve transferability, the existing work introduced the standard invariant regularization (SIR) to impose style-independence property to SCL, which can exempt the impact of nuisance style factors in the standard representation. However, it is unclear how the style-independence property benefits ACL-learned robust representations. In this paper, we leverage the technique of causal reasoning to interpret the ACL and propose adversarial invariant regularization (AIR) to enforce independence from style factors. We regulate the ACL using both SIR and AIR to output the robust representation. Theoretically, we show that AIR implicitly encourages the representational distance between different views of natural data and their adversarial variants to be independent of style factors. Empirically, our experimental results show that invariant regularization significantly improves the performance of state-of-the-art ACL methods in terms of both standard generalization and robustness on downstream tasks. To the best of our knowledge, we are the first to apply causal reasoning to interpret ACL and develop AIR for enhancing ACL-learned robust representations. Our source code is at https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.

摘要: 对抗性对比学习(ACL)是一种增强标准对比学习(SCL)的技术，它通过结合对抗性数据来学习健壮的表示，该表示可以抵抗对抗性攻击和常见的腐败，而不需要昂贵的注释。然而，尚不清楚样式独立属性如何使ACL学习的健壮表示受益。在本文中，我们利用因果推理技术来解释ACL，并提出了对抗不变正则化(AIR)来加强对风格因素的独立性。我们同时使用SIR和AIR来调节ACL，以输出稳健的表示。理论上，我们表明，AIR隐含地鼓励自然数据的不同观点与其敌对变体之间的表征距离独立于风格因素。实验结果表明，不变正则化在标准泛化和下游任务的稳健性方面都显著提高了最新的ACL方法的性能。据我们所知，我们是第一个应用因果推理来解释ACL并开发AIR来增强ACL学习的健壮表示的公司。我们的源代码在https://github.com/GodXuxilie/Enhancing_ACL_via_AIR.



## **42. Kidnapping Deep Learning-based Multirotors using Optimized Flying Adversarial Patches**

优化飞行对抗性补丁绑架基于深度学习的多旋翼机器人 cs.RO

Accepted at MRS 2023, 7 pages, 5 figures. arXiv admin note:  substantial text overlap with arXiv:2305.12859

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2308.00344v2) [paper-pdf](http://arxiv.org/pdf/2308.00344v2)

**Authors**: Pia Hanfeld, Khaled Wahba, Marina M. -C. Höhne, Michael Bussmann, Wolfgang Hönig

**Abstract**: Autonomous flying robots, such as multirotors, often rely on deep learning models that make predictions based on a camera image, e.g. for pose estimation. These models can predict surprising results if applied to input images outside the training domain. This fault can be exploited by adversarial attacks, for example, by computing small images, so-called adversarial patches, that can be placed in the environment to manipulate the neural network's prediction. We introduce flying adversarial patches, where multiple images are mounted on at least one other flying robot and therefore can be placed anywhere in the field of view of a victim multirotor. By introducing the attacker robots, the system is extended to an adversarial multi-robot system. For an effective attack, we compare three methods that simultaneously optimize multiple adversarial patches and their position in the input image. We show that our methods scale well with the number of adversarial patches. Moreover, we demonstrate physical flights with two robots, where we employ a novel attack policy that uses the computed adversarial patches to kidnap a robot that was supposed to follow a human.

摘要: 自主飞行机器人，如多旋翼，通常依赖深度学习模型，根据相机图像进行预测，例如用于姿势估计。如果将这些模型应用于训练域之外的输入图像，则可以预测令人惊讶的结果。这一缺陷可被对抗性攻击利用，例如，通过计算小图像，即所谓的对抗性补丁，可以放置在环境中来操纵神经网络的预测。我们引入了飞行对抗性补丁，其中至少一个其他飞行机器人上安装了多幅图像，因此可以放置在受害者多旋翼的视野中的任何地方。通过引入攻击者机器人，将系统扩展为对抗性多机器人系统。对于有效的攻击，我们比较了三种同时优化多个对抗性补丁及其在输入图像中的位置的方法。我们表明，我们的方法可以很好地适应对抗性补丁的数量。此外，我们还演示了两个机器人的物理飞行，其中我们采用了一种新的攻击策略，使用计算的对抗性补丁来绑架本应跟踪人类的机器人。



## **43. Beyond Hard Samples: Robust and Effective Grammatical Error Correction with Cycle Self-Augmenting**

超越硬样本：循环自增强的稳健有效的语法纠错 cs.CL

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.13321v2) [paper-pdf](http://arxiv.org/pdf/2310.13321v2)

**Authors**: Zecheng Tang, Kaifeng Qi, Juntao Li, Min Zhang

**Abstract**: Recent studies have revealed that grammatical error correction methods in the sequence-to-sequence paradigm are vulnerable to adversarial attack, and simply utilizing adversarial examples in the pre-training or post-training process can significantly enhance the robustness of GEC models to certain types of attack without suffering too much performance loss on clean data. In this paper, we further conduct a thorough robustness evaluation of cutting-edge GEC methods for four different types of adversarial attacks and propose a simple yet very effective Cycle Self-Augmenting (CSA) method accordingly. By leveraging the augmenting data from the GEC models themselves in the post-training process and introducing regularization data for cycle training, our proposed method can effectively improve the model robustness of well-trained GEC models with only a few more training epochs as an extra cost. More concretely, further training on the regularization data can prevent the GEC models from over-fitting on easy-to-learn samples and thus can improve the generalization capability and robustness towards unseen data (adversarial noise/samples). Meanwhile, the self-augmented data can provide more high-quality pseudo pairs to improve model performance on the original testing data. Experiments on four benchmark datasets and seven strong models indicate that our proposed training method can significantly enhance the robustness of four types of attacks without using purposely built adversarial examples in training. Evaluation results on clean data further confirm that our proposed CSA method significantly improves the performance of four baselines and yields nearly comparable results with other state-of-the-art models. Our code is available at https://github.com/ZetangForward/CSA-GEC.

摘要: 最近的研究表明，序列到序列范式中的语法纠错方法容易受到对抗性攻击，在训练前或训练后简单地使用对抗性例子可以显著增强GEC模型对某些类型攻击的鲁棒性，而不会在干净数据上造成太大的性能损失。在本文中，我们进一步对四种不同类型的对抗性攻击的前沿GEC方法进行了深入的健壮性评估，并相应地提出了一种简单但非常有效的循环自增强(CSA)方法。通过在训练后的过程中利用GEC模型本身增加的数据，并引入正则化数据进行循环训练，我们提出的方法可以有效地提高训练有素的GEC模型的模型稳健性，而只需要增加几个训练周期作为额外的代价。更具体地说，对正则化数据的进一步训练可以防止GEC模型对容易学习的样本进行过度拟合，从而提高对未知数据(对抗性噪声/样本)的泛化能力和鲁棒性。同时，自增强后的数据可以提供更多高质量的伪对，以提高模型在原始测试数据上的性能。在四个基准数据集和七个强模型上的实验表明，我们提出的训练方法可以显著提高四种类型攻击的稳健性，而不需要在训练中使用刻意构建的对抗性例子。对CLEAN数据的评估结果进一步证实，我们提出的CSA方法显著提高了四条基线的性能，并产生了与其他最先进模型几乎相当的结果。我们的代码可以在https://github.com/ZetangForward/CSA-GEC.上找到



## **44. Semantic-Aware Adversarial Training for Reliable Deep Hashing Retrieval**

用于可靠深度哈希检索的语义感知对抗训练 cs.CV

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14637v1) [paper-pdf](http://arxiv.org/pdf/2310.14637v1)

**Authors**: Xu Yuan, Zheng Zhang, Xunguang Wang, Lin Wu

**Abstract**: Deep hashing has been intensively studied and successfully applied in large-scale image retrieval systems due to its efficiency and effectiveness. Recent studies have recognized that the existence of adversarial examples poses a security threat to deep hashing models, that is, adversarial vulnerability. Notably, it is challenging to efficiently distill reliable semantic representatives for deep hashing to guide adversarial learning, and thereby it hinders the enhancement of adversarial robustness of deep hashing-based retrieval models. Moreover, current researches on adversarial training for deep hashing are hard to be formalized into a unified minimax structure. In this paper, we explore Semantic-Aware Adversarial Training (SAAT) for improving the adversarial robustness of deep hashing models. Specifically, we conceive a discriminative mainstay features learning (DMFL) scheme to construct semantic representatives for guiding adversarial learning in deep hashing. Particularly, our DMFL with the strict theoretical guarantee is adaptively optimized in a discriminative learning manner, where both discriminative and semantic properties are jointly considered. Moreover, adversarial examples are fabricated by maximizing the Hamming distance between the hash codes of adversarial samples and mainstay features, the efficacy of which is validated in the adversarial attack trials. Further, we, for the first time, formulate the formalized adversarial training of deep hashing into a unified minimax optimization under the guidance of the generated mainstay codes. Extensive experiments on benchmark datasets show superb attack performance against the state-of-the-art algorithms, meanwhile, the proposed adversarial training can effectively eliminate adversarial perturbations for trustworthy deep hashing-based retrieval. Our code is available at https://github.com/xandery-geek/SAAT.

摘要: 深度散列算法以其高效、高效的特点在大规模图像检索系统中得到了广泛的研究和成功的应用。最近的研究已经认识到，对抗性例子的存在对深度哈希模型构成了安全威胁，即对抗性漏洞。值得注意的是，有效地提取可靠的语义代表用于深度散列以指导对抗性学习是具有挑战性的，从而阻碍了基于深度散列的检索模型对抗性健壮性的增强。此外，目前针对深度散列的对抗性训练的研究很难被形式化成一个统一的极大极小结构。本文探讨了语义感知对抗训练(SAAT)来提高深度哈希模型的对抗健壮性。具体地说，我们设想了一种区分主干特征学习(DMFL)方案来构建语义表示，以指导深度哈希中的对抗性学习。特别是，我们的DMFL在严格的理论保证下，以区分学习的方式进行了自适应优化，同时考虑了区分属性和语义属性。此外，通过最大化对抗性样本的哈希码与主流特征之间的汉明距离来构造对抗性样本，并在对抗性攻击试验中验证了该方法的有效性。在生成的主干代码的指导下，首次将深度散列的形式化对抗性训练转化为统一的极大极小优化问题。在基准数据集上的大量实验表明，该算法对现有算法具有良好的攻击性能，同时，本文提出的对抗性训练能够有效地消除对抗性扰动，实现基于深度散列的可信检索。我们的代码可以在https://github.com/xandery-geek/SAAT.上找到



## **45. TrojLLM: A Black-box Trojan Prompt Attack on Large Language Models**

TrojLLM：一种针对大型语言模型的黑盒木马提示攻击 cs.CR

Accepted by NeurIPS'23

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2306.06815v2) [paper-pdf](http://arxiv.org/pdf/2306.06815v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Ting Hua, Yilin Shen, Yepeng Liu, Ladislau Boloni, Qian Lou

**Abstract**: Large Language Models (LLMs) are progressively being utilized as machine learning services and interface tools for various applications. However, the security implications of LLMs, particularly in relation to adversarial and Trojan attacks, remain insufficiently examined. In this paper, we propose TrojLLM, an automatic and black-box framework to effectively generate universal and stealthy triggers. When these triggers are incorporated into the input data, the LLMs' outputs can be maliciously manipulated. Moreover, the framework also supports embedding Trojans within discrete prompts, enhancing the overall effectiveness and precision of the triggers' attacks. Specifically, we propose a trigger discovery algorithm for generating universal triggers for various inputs by querying victim LLM-based APIs using few-shot data samples. Furthermore, we introduce a novel progressive Trojan poisoning algorithm designed to generate poisoned prompts that retain efficacy and transferability across a diverse range of models. Our experiments and results demonstrate TrojLLM's capacity to effectively insert Trojans into text prompts in real-world black-box LLM APIs including GPT-3.5 and GPT-4, while maintaining exceptional performance on clean test sets. Our work sheds light on the potential security risks in current models and offers a potential defensive approach. The source code of TrojLLM is available at https://github.com/UCF-ML-Research/TrojLLM.

摘要: 大型语言模型(LLM)正逐渐被用作各种应用的机器学习服务和接口工具。然而，LLMS的安全影响，特别是与对抗性攻击和特洛伊木马攻击有关的影响，仍然没有得到充分的研究。在本文中，我们提出了一个自动黑盒框架TrojLLM，它可以有效地生成通用的、隐蔽的触发器。当这些触发器被合并到输入数据中时，LLMS的输出可能被恶意操纵。此外，该框架还支持在离散提示中嵌入特洛伊木马，增强了触发器攻击的整体有效性和精确度。具体地说，我们提出了一种触发器发现算法，通过使用少量数据样本查询受害者基于LLM的API来为各种输入生成通用触发器。此外，我们引入了一种新的渐进式特洛伊木马中毒算法，旨在生成中毒提示，从而在不同的模型中保持有效性和可转移性。我们的实验和结果表明，TrojLLM能够在包括GPT-3.5和GPT-4在内的真实黑盒LLMAPI中有效地将特洛伊木马程序插入到文本提示中，同时在干净的测试集上保持出色的性能。我们的工作揭示了当前模型中的潜在安全风险，并提供了一种潜在的防御方法。TrojLLm的源代码可在https://github.com/UCF-ML-Research/TrojLLM.上找到



## **46. ADoPT: LiDAR Spoofing Attack Detection Based on Point-Level Temporal Consistency**

采用：基于点级时间一致性的激光雷达欺骗攻击检测 cs.CV

BMVC 2023 (17 pages, 13 figures, and 1 table)

**SubmitDate**: 2023-10-23    [abs](http://arxiv.org/abs/2310.14504v1) [paper-pdf](http://arxiv.org/pdf/2310.14504v1)

**Authors**: Minkyoung Cho, Yulong Cao, Zixiang Zhou, Z. Morley Mao

**Abstract**: Deep neural networks (DNNs) are increasingly integrated into LiDAR (Light Detection and Ranging)-based perception systems for autonomous vehicles (AVs), requiring robust performance under adversarial conditions. We aim to address the challenge of LiDAR spoofing attacks, where attackers inject fake objects into LiDAR data and fool AVs to misinterpret their environment and make erroneous decisions. However, current defense algorithms predominantly depend on perception outputs (i.e., bounding boxes) thus face limitations in detecting attackers given the bounding boxes are generated by imperfect perception models processing limited points, acquired based on the ego vehicle's viewpoint. To overcome these limitations, we propose a novel framework, named ADoPT (Anomaly Detection based on Point-level Temporal consistency), which quantitatively measures temporal consistency across consecutive frames and identifies abnormal objects based on the coherency of point clusters. In our evaluation using the nuScenes dataset, our algorithm effectively counters various LiDAR spoofing attacks, achieving a low (< 10%) false positive ratio (FPR) and high (> 85%) true positive ratio (TPR), outperforming existing state-of-the-art defense methods, CARLO and 3D-TC2. Furthermore, our evaluation demonstrates the promising potential for accurate attack detection across various road environments.

摘要: 深度神经网络(DNN)越来越多地被集成到基于LiDAR(光检测和测距)的自主车辆(AV)感知系统中，要求在对抗条件下具有稳健的性能。我们的目标是应对LiDAR欺骗攻击的挑战，即攻击者向LiDAR数据中注入虚假对象，并愚弄AVs曲解其环境并做出错误的决定。然而，当前的防御算法主要依赖于感知输出(即包围盒)，因此在检测攻击者时面临限制，因为包围盒是由基于自我车辆的视角获得的处理有限点的不完美感知模型生成的。为了克服这些局限性，我们提出了一种新的框架，称为基于点级时间一致性的异常检测框架，该框架定量地测量连续帧的时间一致性，并根据点簇的一致性来识别异常对象。在我们使用nuScenes数据集进行的评估中，我们的算法有效地抵抗了各种LiDAR欺骗攻击，获得了低(<10%)的假正确率(FPR)和高(>85%)的真正确率(TPR)，性能优于现有的最先进的防御方法CALO和3D-TC2。此外，我们的评估显示了在各种道路环境中进行准确攻击检测的前景。



## **47. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

26 pages, 17 figures, 1 table

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2307.02881v4) [paper-pdf](http://arxiv.org/pdf/2307.02881v4)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating image probability density functions that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space-not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. We therefore consider popular generative models. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: the possibility to sample from this distribution with the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute its probability, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show how semantic interpretations are used to describe points on the manifold. To achieve this, we consider an emergent language framework that uses variational encoders for a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described as evolving semantic descriptions. We also show that such probabilistic descriptions (bounded) can be used to improve semantic consistency by constructing defences against adversarial attacks. We evaluate our methods with improved semantic robustness and OoD detection capability, explainable and editable semantic interpolation, and improved classification accuracy under patch attacks. We also discuss the limitation in diffusion models.

摘要: 本文首先描述了用于估计图像概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域-并不是每种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。因此，我们考虑流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下属性：1)样本生成：使用建模的密度函数从该分布中进行样本的可能性；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应能够计算其概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们展示了如何使用语义解释来描述流形上的点。为了实现这一点，我们考虑一种新的语言框架，它使用变分编码器来解开驻留在给定流形上的点的表示。然后，流形上的点之间的轨迹可以被描述为不断演变的语义描述。我们还表明，这种概率描述(有界)可以通过构建对对手攻击的防御来提高语义一致性。我们通过改进的语义健壮性和OOD检测能力、可解释和可编辑的语义内插以及在补丁攻击下改进的分类精度来评估我们的方法。我们还讨论了扩散模型的局限性。



## **48. Attacks Meet Interpretability (AmI) Evaluation and Findings**

攻击符合可解释性(AMI)评估和调查结果 cs.CR

Need to withdraw it. The current work needs to be changed at a large  extent which would take a longer time

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.08808v3) [paper-pdf](http://arxiv.org/pdf/2310.08808v3)

**Authors**: Qian Ma, Ziping Ye, Shagufta Mehnaz

**Abstract**: To investigate the effectiveness of the model explanation in detecting adversarial examples, we reproduce the results of two papers, Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples and Is AmI (Attacks Meet Interpretability) Robust to Adversarial Examples. And then conduct experiments and case studies to identify the limitations of both works. We find that Attacks Meet Interpretability(AmI) is highly dependent on the selection of hyperparameters. Therefore, with a different hyperparameter choice, AmI is still able to detect Nicholas Carlini's attack. Finally, we propose recommendations for future work on the evaluation of defense techniques such as AmI.

摘要: 为了考察模型解释在检测敌意实例方面的有效性，我们复制了两篇论文的结果：攻击满足解释性：对抗性样本的属性导向检测和AMI(攻击满足解释性)对对抗性实例的稳健性。然后进行实验和案例研究，找出两部作品的局限性。我们发现攻击满足可解释性(AMI)高度依赖于超参数的选择。因此，通过不同的超参数选择，阿米仍然能够检测到尼古拉斯·卡里尼的攻击。最后，我们对AMI等防御技术的未来评估工作提出了建议。



## **49. HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with Adaptive Firing Thresholds**

HoSNN：具有自适应放电阈值的逆鲁棒自适应稳态尖峰神经网络 cs.NE

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2308.10373v2) [paper-pdf](http://arxiv.org/pdf/2308.10373v2)

**Authors**: Hejia Geng, Peng Li

**Abstract**: Spiking neural networks (SNNs) offer promise for efficient and powerful neurally inspired computation. Common to other types of neural networks, however, SNNs face the severe issue of vulnerability to adversarial attacks. We present the first study that draws inspiration from neural homeostasis to develop a bio-inspired solution that counters the susceptibilities of SNNs to adversarial onslaughts. At the heart of our approach is a novel threshold-adapting leaky integrate-and-fire (TA-LIF) neuron model, which we adopt to construct the proposed adversarially robust homeostatic SNN (HoSNN). Distinct from traditional LIF models, our TA-LIF model incorporates a self-stabilizing dynamic thresholding mechanism, curtailing adversarial noise propagation and safeguarding the robustness of HoSNNs in an unsupervised manner. Theoretical analysis is presented to shed light on the stability and convergence properties of the TA-LIF neurons, underscoring their superior dynamic robustness under input distributional shifts over traditional LIF neurons. Remarkably, without explicit adversarial training, our HoSNNs demonstrate inherent robustness on CIFAR-10, with accuracy improvements to 72.6% and 54.19% against FGSM and PGD attacks, up from 20.97% and 0.6%, respectively. Furthermore, with minimal FGSM adversarial training, our HoSNNs surpass previous models by 29.99% under FGSM and 47.83% under PGD attacks on CIFAR-10. Our findings offer a new perspective on harnessing biological principles for bolstering SNNs adversarial robustness and defense, paving the way to more resilient neuromorphic computing.

摘要: 尖峰神经网络(SNN)为高效和强大的神经启发计算提供了希望。然而，与其他类型的神经网络一样，SNN面临着易受对手攻击的严重问题。我们介绍了第一项从神经动态平衡中获得灵感的研究，以开发一种生物启发的解决方案，以对抗SNN对对手攻击的敏感性。我们方法的核心是一种新的阈值自适应泄漏积分与点火(TA-LIF)神经元模型，我们采用该模型来构造所提出的对抗性鲁棒自稳态SNN(HoSNN)。与传统的LIF模型不同，TA-LIF模型引入了一种自稳定的动态阈值机制，在无监督的情况下抑制了敌对噪声的传播，保护了HoSNN的健壮性。理论分析揭示了TA-LIF神经元的稳定性和收敛特性，强调了其在输入分布漂移下优于传统LIF神经元的动态鲁棒性。值得注意的是，在没有明确的对抗性训练的情况下，我们的HoSNN对CIFAR-10表现出固有的鲁棒性，对FGSM和PGD攻击的准确率分别从20.97%和0.6%提高到72.6%和54.19%。此外，在最少的FGSM对抗训练下，我们的HoSNN在FGSM下超过了以前的模型29.99%，在PGD攻击CIFAR-10下超过了47.83%。我们的发现为利用生物学原理加强SNN对抗的稳健性和防御提供了一个新的视角，为更具弹性的神经形态计算铺平了道路。



## **50. DANAA: Towards transferable attacks with double adversarial neuron attribution**

DANAA：具有双重对抗神经元属性的可转移攻击 cs.CV

Accepted by 19th International Conference on Advanced Data Mining and  Applications. (ADMA 2023)

**SubmitDate**: 2023-10-22    [abs](http://arxiv.org/abs/2310.10427v2) [paper-pdf](http://arxiv.org/pdf/2310.10427v2)

**Authors**: Zhibo Jin, Zhiyu Zhu, Xinyi Wang, Jiayu Zhang, Jun Shen, Huaming Chen

**Abstract**: While deep neural networks have excellent results in many fields, they are susceptible to interference from attacking samples resulting in erroneous judgments. Feature-level attacks are one of the effective attack types, which targets the learnt features in the hidden layers to improve its transferability across different models. Yet it is observed that the transferability has been largely impacted by the neuron importance estimation results. In this paper, a double adversarial neuron attribution attack method, termed `DANAA', is proposed to obtain more accurate feature importance estimation. In our method, the model outputs are attributed to the middle layer based on an adversarial non-linear path. The goal is to measure the weight of individual neurons and retain the features that are more important towards transferability. We have conducted extensive experiments on the benchmark datasets to demonstrate the state-of-the-art performance of our method. Our code is available at: https://github.com/Davidjinzb/DANAA

摘要: 虽然深度神经网络在许多领域都有很好的效果，但它们容易受到攻击样本的干扰，从而导致错误的判断。特征级攻击是一种有效的攻击类型，它针对隐含层中的学习特征，以提高其在不同模型上的可移植性。然而，观察到神经元重要性估计结果在很大程度上影响了神经网络的可转移性。为了获得更准确的特征重要性估计，本文提出了一种双重对抗神经元属性攻击方法--DANAA。在我们的方法中，模型输出被归因于基于对抗性非线性路径的中间层。其目标是测量单个神经元的重量，并保留对可转移性更重要的特征。我们在基准数据集上进行了广泛的实验，以证明我们的方法具有最先进的性能。我们的代码请访问：https://github.com/Davidjinzb/DANAA



