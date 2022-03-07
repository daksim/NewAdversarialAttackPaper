# Latest Adversarial Attack Papers
**update at 2022-03-08 06:31:30**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Medical Aegis: Robust adversarial protectors for medical images**

医学宙斯盾：强大的医学图像对抗性保护者 cs.CV

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2111.10969v4)

**Authors**: Qingsong Yao, Zecheng He, S. Kevin Zhou

**Abstracts**: Deep neural network based medical image systems are vulnerable to adversarial examples. Many defense mechanisms have been proposed in the literature, however, the existing defenses assume a passive attacker who knows little about the defense system and does not change the attack strategy according to the defense. Recent works have shown that a strong adaptive attack, where an attacker is assumed to have full knowledge about the defense system, can easily bypass the existing defenses. In this paper, we propose a novel adversarial example defense system called Medical Aegis. To the best of our knowledge, Medical Aegis is the first defense in the literature that successfully addresses the strong adaptive adversarial example attacks to medical images. Medical Aegis boasts two-tier protectors: The first tier of Cushion weakens the adversarial manipulation capability of an attack by removing its high-frequency components, yet posing a minimal effect on classification performance of the original image; the second tier of Shield learns a set of per-class DNN models to predict the logits of the protected model. Deviation from the Shield's prediction indicates adversarial examples. Shield is inspired by the observations in our stress tests that there exist robust trails in the shallow layers of a DNN model, which the adaptive attacks can hardly destruct. Experimental results show that the proposed defense accurately detects adaptive attacks, with negligible overhead for model inference.

摘要: 基于深度神经网络的医学图像系统容易受到敌意例子的攻击。文献中提出了很多防御机制，但现有的防御机制都假设攻击者是被动的，对防御系统知之甚少，不会根据防御情况改变攻击策略。最近的研究表明，强自适应攻击(假设攻击者完全了解防御系统)可以很容易地绕过现有的防御系统。在本文中，我们提出了一种新的对抗性实例防御系统--医学宙斯盾(Medical Aegis)。据我们所知，医学宙斯盾是文献中第一个成功解决了对医学图像的强适应性敌意攻击的防御方案。医学宙斯盾拥有两层保护器：第一层缓冲通过移除攻击的高频分量来削弱攻击的敌意操纵能力，但对原始图像的分类性能影响最小；第二层Shield学习一组按类的DNN模型来预测受保护模型的逻辑。与盾牌的预测背道而驰表明了敌对的例子。Shield的灵感来自于我们在压力测试中观察到的DNN模型的浅层存在健壮的踪迹，而自适应攻击很难破坏这些踪迹。实验结果表明，该防御方法能够准确检测出自适应攻击，而模型推理开销可以忽略不计。



## **2. Adversarial Patterns: Building Robust Android Malware Classifiers**

对抗性模式：构建健壮的Android恶意软件分类器 cs.CR

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02121v1)

**Authors**: Dipkamal Bhusal, Nidhi Rastogi

**Abstracts**: Deep learning-based classifiers have substantially improved recognition of malware samples. However, these classifiers can be vulnerable to adversarial input perturbations. Any vulnerability in malware classifiers poses significant threats to the platforms they defend. Therefore, to create stronger defense models against malware, we must understand the patterns in input perturbations caused by an adversary. This survey paper presents a comprehensive study on adversarial machine learning for android malware classifiers. We first present an extensive background in building a machine learning classifier for android malware, covering both image-based and text-based feature extraction approaches. Then, we examine the pattern and advancements in the state-of-the-art research in evasion attacks and defenses. Finally, we present guidelines for designing robust malware classifiers and enlist research directions for the future.

摘要: 基于深度学习的分类器大大提高了恶意软件样本的识别能力。然而，这些分类器可能容易受到对抗性输入扰动的影响。恶意软件分类器中的任何漏洞都会对其防御的平台构成重大威胁。因此，要创建针对恶意软件的更强大的防御模型，我们必须了解由对手造成的输入扰动的模式。本文对Android恶意软件分类器的对抗性机器学习进行了全面的研究。我们首先介绍了构建Android恶意软件机器学习分类器的广泛背景，包括基于图像和基于文本的特征提取方法。然后，我们考察了在躲避攻击和防御方面的最新研究模式和进展。最后，我们提出了设计健壮的恶意软件分类器的指导原则，并提出了未来的研究方向。



## **3. Label Leakage and Protection from Forward Embedding in Vertical Federated Learning**

垂直联合学习中的标签泄漏与前向嵌入保护 cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.01451v2)

**Authors**: Jiankai Sun, Xin Yang, Yuanshun Yao, Chong Wang

**Abstracts**: Vertical federated learning (vFL) has gained much attention and been deployed to solve machine learning problems with data privacy concerns in recent years. However, some recent work demonstrated that vFL is vulnerable to privacy leakage even though only the forward intermediate embedding (rather than raw features) and backpropagated gradients (rather than raw labels) are communicated between the involved participants. As the raw labels often contain highly sensitive information, some recent work has been proposed to prevent the label leakage from the backpropagated gradients effectively in vFL. However, these work only identified and defended the threat of label leakage from the backpropagated gradients. None of these work has paid attention to the problem of label leakage from the intermediate embedding. In this paper, we propose a practical label inference method which can steal private labels effectively from the shared intermediate embedding even though some existing protection methods such as label differential privacy and gradients perturbation are applied. The effectiveness of the label attack is inseparable from the correlation between the intermediate embedding and corresponding private labels. To mitigate the issue of label leakage from the forward embedding, we add an additional optimization goal at the label party to limit the label stealing ability of the adversary by minimizing the distance correlation between the intermediate embedding and corresponding private labels. We conducted massive experiments to demonstrate the effectiveness of our proposed protection methods.

摘要: 垂直联合学习(VFL)近年来得到了广泛的关注，并被应用于解决数据隐私问题中的机器学习问题。然而，最近的一些工作表明，即使参与者之间只传递前向中间嵌入(而不是原始特征)和反向传播梯度(而不是原始标签)，VFL也容易受到隐私泄露的影响。由于原始标签往往包含高度敏感的信息，最近已有一些工作被提出以有效地防止VFL中反向传播梯度引起的标签泄漏。然而，这些工作仅仅识别和防御了反向传播梯度带来的标签泄漏威胁。这些工作都没有注意到中间嵌入带来的标签泄漏问题。本文提出了一种实用的标签推理方法，即使采用了标签差分隐私、梯度扰动等保护方法，也能有效地从共享中间嵌入中窃取私有标签。标签攻击的有效性离不开中间嵌入与对应的私有标签之间的关联。为了缓解前向嵌入带来的标签泄漏问题，我们在标签方增加了一个额外的优化目标，通过最小化中间嵌入与相应私有标签之间的距离相关性来限制敌手的标签窃取能力。我们进行了大量的实验来验证我们提出的保护方法的有效性。



## **4. Differentially Private Label Protection in Split Learning**

分裂学习中的差异化私有标签保护 cs.LG

**SubmitDate**: 2022-03-04    [paper-pdf](http://arxiv.org/pdf/2203.02073v1)

**Authors**: Xin Yang, Jiankai Sun, Yuanshun Yao, Junyuan Xie, Chong Wang

**Abstracts**: Split learning is a distributed training framework that allows multiple parties to jointly train a machine learning model over vertically partitioned data (partitioned by attributes). The idea is that only intermediate computation results, rather than private features and labels, are shared between parties so that raw training data remains private. Nevertheless, recent works showed that the plaintext implementation of split learning suffers from severe privacy risks that a semi-honest adversary can easily reconstruct labels. In this work, we propose \textsf{TPSL} (Transcript Private Split Learning), a generic gradient perturbation based split learning framework that provides provable differential privacy guarantee. Differential privacy is enforced on not only the model weights, but also the communicated messages in the distributed computation setting. Our experiments on large-scale real-world datasets demonstrate the robustness and effectiveness of \textsf{TPSL} against label leakage attacks. We also find that \textsf{TPSL} have a better utility-privacy trade-off than baselines.

摘要: 分裂学习是一种分布式训练框架，允许多方在垂直划分的数据(按属性划分)上联合训练机器学习模型。其想法是，各方之间只共享中间计算结果，而不是私有特征和标签，因此原始训练数据保持私有。然而，最近的研究表明，分裂学习的明文实现存在严重的隐私风险，半诚实的对手可以很容易地重构标签。在这项工作中，我们提出了一个通用的基于梯度扰动的分裂学习框架--textsf{TPSL}(Transcript Private Split Learning)，它提供了可证明的差分隐私保证。在分布式计算环境中，不仅对模型权重，而且对通信消息实施差分隐私。我们在大规模真实数据集上的实验证明了Textsf{TPSL}对标签泄漏攻击的鲁棒性和有效性。我们还发现，\textsf{tpsl}比基线有更好的效用-隐私权衡。



## **5. Can Authoritative Governments Abuse the Right to Access?**

权威政府会滥用访问权吗？ cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02068v1)

**Authors**: Cédric Lauradoux

**Abstracts**: The right to access is a great tool provided by the GDPR to empower data subjects with their data. However, it needs to be implemented properly otherwise it could turn subject access requests against the subjects privacy. Indeed, recent works have shown that it is possible to abuse the right to access using impersonation attacks. We propose to extend those impersonation attacks by considering that the adversary has an access to governmental resources. In this case, the adversary can forge official documents or exploit copy of them. Our attack affects more people than one may expect. To defeat the attacks from this kind of adversary, several solutions are available like multi-factors or proof of aliveness. Our attacks highlight the need for strong procedures to authenticate subject access requests.

摘要: 访问权是GDPR提供的一个很好的工具，用来赋予数据主体数据权力。但是，它需要正确实现，否则可能会使主体访问请求与主体隐私背道而驰。事实上，最近的研究表明，使用冒充攻击来滥用访问权限是可能的。我们建议扩展这些冒充攻击，因为考虑到对手可以访问政府资源。在这种情况下，对手可以伪造官方文件或利用其副本。我们的袭击影响的人比人们预料的要多。要击败这类对手的攻击，有几种解决方案可用，比如多因素或活性证明。我们的攻击突显了需要强大的程序来验证主体访问请求。



## **6. Autonomous and Resilient Control for Optimal LEO Satellite Constellation Coverage Against Space Threats**

空间威胁下LEO卫星星座最优覆盖的自主弹性控制 eess.SY

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02050v1)

**Authors**: Yuhan Zhao, Quanyan Zhu

**Abstracts**: LEO satellite constellation coverage has served as the base platform for various space applications. However, the rapidly evolving security environment such as orbit debris and adversarial space threats are greatly endangering the security of satellite constellation and integrity of the satellite constellation coverage. As on-orbit repairs are challenging, a distributed and autonomous protection mechanism is necessary to ensure the adaptation and self-healing of the satellite constellation coverage from different attacks. To this end, we establish an integrative and distributed framework to enable resilient satellite constellation coverage planning and control in a single orbit. Each satellite can make decisions individually to recover from adversarial and non-adversarial attacks and keep providing coverage service. We first provide models and methodologies to measure the coverage performance. Then, we formulate the joint resilient coverage planning-control problem as a two-stage problem. A coverage game is proposed to find the equilibrium constellation deployment for resilient coverage planning and an agent-based algorithm is developed to compute the equilibrium. The multi-waypoint Model Predictive Control (MPC) methodology is adopted to achieve autonomous self-healing control. Finally, we use a typical LEO satellite constellation as a case study to corroborate the results.

摘要: 低轨卫星星座复盖已成为各种空间应用的基础平台。然而，快速发展的轨道碎片和对抗性空间威胁等安全环境极大地威胁着卫星星座的安全性和卫星星座覆盖的完整性。由于在轨修复具有挑战性，需要一种分布式、自主的保护机制来确保卫星星座覆盖在不同攻击下的适应性和自愈性。为此，我们建立了一个一体化和分布式的框架，使弹性卫星星座覆盖规划和控制能够在单一轨道上进行。每颗卫星都可以单独做出决定，从对抗性和非对抗性攻击中恢复，并继续提供覆盖服务。我们首先提供了衡量覆盖性能的模型和方法。然后，我们将联合弹性覆盖规划-控制问题描述为一个两阶段问题。提出了一种基于覆盖博弈的弹性覆盖规划均衡星座部署方法，并提出了一种基于Agent的均衡算法。采用多路点模型预测控制(MPC)方法实现自主自愈控制。最后，以一个典型的低轨卫星星座为例进行了验证。



## **7. Why adversarial training can hurt robust accuracy**

为什么对抗性训练会损害稳健的准确性 cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.02006v1)

**Authors**: Jacob Clarysse, Julia Hörmann, Fanny Yang

**Abstracts**: Machine learning classifiers with high test accuracy often perform poorly under adversarial attacks. It is commonly believed that adversarial training alleviates this issue. In this paper, we demonstrate that, surprisingly, the opposite may be true -- Even though adversarial training helps when enough data is available, it may hurt robust generalization in the small sample size regime. We first prove this phenomenon for a high-dimensional linear classification setting with noiseless observations. Our proof provides explanatory insights that may also transfer to feature learning models. Further, we observe in experiments on standard image datasets that the same behavior occurs for perceptible attacks that effectively reduce class information such as mask attacks and object corruptions.

摘要: 测试精度高的机器学习分类器在敌意攻击下往往表现不佳。一般认为对抗性训练可以缓解这个问题。在这篇文章中，我们证明，令人惊讶的是，相反的情况可能是真的--尽管对抗性训练在足够的数据可用时有所帮助，但它可能会损害小样本制度下的鲁棒泛化。我们首先在高维线性分类环境中用无声观测证明了这一现象。我们的证明提供了解释性的见解，也可以转移到特征学习模型中。此外，我们在标准图像数据集上的实验中观察到，对于可感知的攻击，也会出现同样的行为，这些攻击有效地减少了掩码攻击和对象损坏等类别信息。



## **8. Dynamic Backdoor Attacks Against Machine Learning Models**

针对机器学习模型的动态后门攻击 cs.CR

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2003.03675v2)

**Authors**: Ahmed Salem, Rui Wen, Michael Backes, Shiqing Ma, Yang Zhang

**Abstracts**: Machine learning (ML) has made tremendous progress during the past decade and is being adopted in various critical real-world applications. However, recent research has shown that ML models are vulnerable to multiple security and privacy attacks. In particular, backdoor attacks against ML models have recently raised a lot of awareness. A successful backdoor attack can cause severe consequences, such as allowing an adversary to bypass critical authentication systems.   Current backdooring techniques rely on adding static triggers (with fixed patterns and locations) on ML model inputs which are prone to detection by the current backdoor detection mechanisms. In this paper, we propose the first class of dynamic backdooring techniques against deep neural networks (DNN), namely Random Backdoor, Backdoor Generating Network (BaN), and conditional Backdoor Generating Network (c-BaN). Triggers generated by our techniques can have random patterns and locations, which reduce the efficacy of the current backdoor detection mechanisms. In particular, BaN and c-BaN based on a novel generative network are the first two schemes that algorithmically generate triggers. Moreover, c-BaN is the first conditional backdooring technique that given a target label, it can generate a target-specific trigger. Both BaN and c-BaN are essentially a general framework which renders the adversary the flexibility for further customizing backdoor attacks.   We extensively evaluate our techniques on three benchmark datasets: MNIST, CelebA, and CIFAR-10. Our techniques achieve almost perfect attack performance on backdoored data with a negligible utility loss. We further show that our techniques can bypass current state-of-the-art defense mechanisms against backdoor attacks, including ABS, Februus, MNTD, Neural Cleanse, and STRIP.

摘要: 机器学习(ML)在过去的十年中取得了巨大的进步，并被应用于各种关键的现实世界应用中。然而，最近的研究表明，ML模型容易受到多种安全和隐私攻击。特别值得一提的是，针对ML模型的后门攻击最近引起了很多关注。成功的后门攻击可能导致严重后果，例如允许攻击者绕过关键身份验证系统。当前的回溯技术依赖于在ML模型输入上添加静电触发器(具有固定的模式和位置)，这容易被当前的后门检测机制检测到。本文提出了针对深度神经网络(DNN)的第一类动态回溯技术，即随机后门、后门生成网络(BAN)和条件后门生成网络(C-BAN)。我们的技术生成的触发器可能具有随机的模式和位置，这降低了当前后门检测机制的效率。特别地，基于新型产生式网络的BAN和C-BAN是算法上生成触发器的前两个方案。此外，C-BAN是第一种在给定目标标签的情况下可以生成特定于目标的触发器的条件回溯技术。BAND和C-BAN本质上都是一个通用框架，为对手提供了进一步定制后门攻击的灵活性。我们在三个基准数据集上广泛评估了我们的技术：MNIST、CelebA和CIFAR-10。我们的技术在几乎可以忽略效用损失的情况下实现了对后置数据的近乎完美的攻击性能。我们进一步表明，我们的技术可以绕过当前最先进的后门攻击防御机制，包括ABS、Februus、MNTD、NeuroCleanse和STRINE。



## **9. Assessing the Robustness of Visual Question Answering Models**

视觉问答模型的稳健性评估 cs.CV

24 pages, 13 figures, International Journal of Computer Vision (IJCV)  [under review]. arXiv admin note: substantial text overlap with  arXiv:1711.06232, arXiv:1709.04625

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1912.01452v2)

**Authors**: Jia-Hong Huang, Modar Alfadly, Bernard Ghanem, Marcel Worring

**Abstracts**: Deep neural networks have been playing an essential role in the task of Visual Question Answering (VQA). Until recently, their accuracy has been the main focus of research. Now there is a trend toward assessing the robustness of these models against adversarial attacks by evaluating the accuracy of these models under increasing levels of noisiness in the inputs of VQA models. In VQA, the attack can target the image and/or the proposed query question, dubbed main question, and yet there is a lack of proper analysis of this aspect of VQA. In this work, we propose a new method that uses semantically related questions, dubbed basic questions, acting as noise to evaluate the robustness of VQA models. We hypothesize that as the similarity of a basic question to the main question decreases, the level of noise increases. To generate a reasonable noise level for a given main question, we rank a pool of basic questions based on their similarity with this main question. We cast this ranking problem as a LASSO optimization problem. We also propose a novel robustness measure Rscore and two large-scale basic question datasets in order to standardize robustness analysis of VQA models. The experimental results demonstrate that the proposed evaluation method is able to effectively analyze the robustness of VQA models. To foster the VQA research, we will publish our proposed datasets.

摘要: 深度神经网络在视觉问答(VQA)中起着至关重要的作用。直到最近，它们的准确性一直是研究的主要焦点。现在有一种趋势是通过评估在VQA模型输入的噪声水平增加时这些模型的准确性来评估这些模型对敌方攻击的鲁棒性。在VQA中，攻击可以以图像和/或建议的查询问题(称为主问题)为目标，但缺乏对VQA这一方面的适当分析。在这项工作中，我们提出了一种新的方法，使用语义相关的问题，称为基本问题，作为噪声来评估VQA模型的稳健性。我们假设，随着基本问题与主要问题的相似度降低，噪音水平就会增加。为了为给定的主要问题生成合理的噪声水平，我们根据基本问题与该主要问题的相似性对基本问题池进行排名。我们把这个排序问题归结为套索优化问题。为了规范VQA模型的健壮性分析，我们还提出了一种新的健壮性度量RSCORE和两个大规模的基本问题数据集。实验结果表明，该评估方法能够有效地分析VQA模型的鲁棒性。为了促进VQA研究，我们将公布我们建议的数据集。



## **10. Detection of Word Adversarial Examples in Text Classification: Benchmark and Baseline via Robust Density Estimation**

文本分类中词语对抗性实例的检测：基于鲁棒密度估计的基准和基线 cs.CL

Findings of ACL 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01677v1)

**Authors**: KiYoon Yoo, Jangho Kim, Jiho Jang, Nojun Kwak

**Abstracts**: Word-level adversarial attacks have shown success in NLP models, drastically decreasing the performance of transformer-based models in recent years. As a countermeasure, adversarial defense has been explored, but relatively few efforts have been made to detect adversarial examples. However, detecting adversarial examples may be crucial for automated tasks (e.g. review sentiment analysis) that wish to amass information about a certain population and additionally be a step towards a robust defense system. To this end, we release a dataset for four popular attack methods on four datasets and four models to encourage further research in this field. Along with it, we propose a competitive baseline based on density estimation that has the highest AUC on 29 out of 30 dataset-attack-model combinations. Source code is available in https://github.com/anoymous92874838/text-adv-detection.

摘要: 词级敌意攻击在NLP模型中已显示出成功，近年来极大地降低了基于变压器的模型的性能。作为一种对策，对抗性防御已经被探索，但对发现对抗性例子的努力相对较少。然而，检测敌意的例子对于自动化任务(例如，评论情绪分析)可能是至关重要的，这些自动化任务希望收集关于特定人群的信息，并且另外是迈向健壮防御系统的一步。为此，我们在四个数据集和四个模型上发布了四种流行攻击方法的数据集，以鼓励该领域的进一步研究。在此基础上，我们提出了一个基于密度估计的好胜基线，该基线在30个数据集攻击模型组合中的29个组合上具有最高的AUC值。源代码在https://github.com/anoymous92874838/text-adv-detection.中提供



## **11. On Improving Adversarial Transferability of Vision Transformers**

关于提高视觉变形金刚对抗性转换性的探讨 cs.CV

ICLR'22 (Spotlight), the first two authors contributed equally. Code:  https://t.ly/hBbW

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2106.04169v3)

**Authors**: Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, Fatih Porikli

**Abstracts**: Vision transformers (ViTs) process input images as sequences of patches via self-attention; a radically different architecture than convolutional neural networks (CNNs). This makes it interesting to study the adversarial feature space of ViT models and their transferability. In particular, we observe that adversarial patterns found via conventional adversarial attacks show very \emph{low} black-box transferability even for large ViT models. We show that this phenomenon is only due to the sub-optimal attack procedures that do not leverage the true representation potential of ViTs. A deep ViT is composed of multiple blocks, with a consistent architecture comprising of self-attention and feed-forward layers, where each block is capable of independently producing a class token. Formulating an attack using only the last class token (conventional approach) does not directly leverage the discriminative information stored in the earlier tokens, leading to poor adversarial transferability of ViTs. Using the compositional nature of ViT models, we enhance transferability of existing attacks by introducing two novel strategies specific to the architecture of ViT models. (i) Self-Ensemble: We propose a method to find multiple discriminative pathways by dissecting a single ViT model into an ensemble of networks. This allows explicitly utilizing class-specific information at each ViT block. (ii) Token Refinement: We then propose to refine the tokens to further enhance the discriminative capacity at each block of ViT. Our token refinement systematically combines the class tokens with structural information preserved within the patch tokens.

摘要: 视觉转换器(VITS)通过自我注意将输入图像处理为补丁序列；这是一种与卷积神经网络(CNN)完全不同的体系结构。这使得研究VIT模型的对抗性特征空间及其可移植性变得很有意义。特别地，我们观察到，即使对于大型VIT模型，通过传统的对抗性攻击发现的对抗性模式也表现出非常低的黑箱可转移性。我们表明，这种现象仅仅是由于次优攻击过程没有充分利用VITS的真实表现潜力所致。深度VIT由多个块组成，具有一致的架构，由自观层和前馈层组成，每个挡路可以独立生成一个类Token。仅使用最后一个类令牌(传统方法)来制定攻击没有直接利用存储在较早令牌中的区别性信息，从而导致VITS的对抗性差的可转移性。利用VIT模型的组合特性，通过引入两种针对VIT模型体系结构的新策略，增强了现有攻击的可转移性。(I)自集成：我们提出了一种通过将单个VIT模型分解成一个网络集成来寻找多条区分路径的方法。这允许在每个VIT挡路上显式地利用特定于类的信息。(Ii)优化代币：然后，我们建议优化代币，以进一步增强每个挡路的识别能力。我们的令牌精化将类令牌与保存在补丁令牌中的结构信息系统地结合在一起。



## **12. On Robustness of Neural Ordinary Differential Equations**

关于神经常微分方程的稳健性 cs.LG

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/1910.05513v4)

**Authors**: Hanshu Yan, Jiawei Du, Vincent Y. F. Tan, Jiashi Feng

**Abstracts**: Neural ordinary differential equations (ODEs) have been attracting increasing attention in various research domains recently. There have been some works studying optimization issues and approximation capabilities of neural ODEs, but their robustness is still yet unclear. In this work, we fill this important gap by exploring robustness properties of neural ODEs both empirically and theoretically. We first present an empirical study on the robustness of the neural ODE-based networks (ODENets) by exposing them to inputs with various types of perturbations and subsequently investigating the changes of the corresponding outputs. In contrast to conventional convolutional neural networks (CNNs), we find that the ODENets are more robust against both random Gaussian perturbations and adversarial attack examples. We then provide an insightful understanding of this phenomenon by exploiting a certain desirable property of the flow of a continuous-time ODE, namely that integral curves are non-intersecting. Our work suggests that, due to their intrinsic robustness, it is promising to use neural ODEs as a basic block for building robust deep network models. To further enhance the robustness of vanilla neural ODEs, we propose the time-invariant steady neural ODE (TisODE), which regularizes the flow on perturbed data via the time-invariant property and the imposition of a steady-state constraint. We show that the TisODE method outperforms vanilla neural ODEs and also can work in conjunction with other state-of-the-art architectural methods to build more robust deep networks.

摘要: 近年来，神经常微分方程(ODE)在各个研究领域受到越来越多的关注。已有一些研究神经常微分方程的优化问题和逼近能力的工作，但其鲁棒性尚不清楚。在这项工作中，我们通过从经验和理论上探索神经ODE的稳健性来填补这一重要空白。我们首先对基于ODENET的神经网络(ODENet)的鲁棒性进行了实证研究，方法是将ODENet暴露在具有各种类型扰动的输入中，然后研究相应输出的变化。与传统的卷积神经网络(CNNs)相比，我们发现ODENet对随机高斯扰动和敌意攻击示例都具有更强的鲁棒性。然后，我们通过利用连续时间颂歌的流的某些理想性质，即积分曲线是不相交的，来提供对这一现象的深刻理解。我们的工作表明，由于其固有的鲁棒性，使用神经ODE作为构建鲁棒深层网络模型的基础挡路是很有前途的。为了进一步增强香草神经微分方程组的鲁棒性，我们提出了时不变稳态神经微分方程组(TisODE)，它通过时不变性和施加稳态约束来规则化扰动数据上的流动。我们表明，TisODE方法的性能优于香草神经ODE方法，并且还可以与其他最先进的体系结构方法相结合来构建更健壮的深层网络。



## **13. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

对基于投影的可取消生物特征识别方案的认证攻击 cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2110.15163v2)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物测定方案旨在通过将诸如密码、存储的秘密或盐等用户特定令牌与生物测定数据相结合来生成安全的生物测定模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近几个方案在这些要求方面受到攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未被证明。本文利用整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以便冒充任何个人。此外，在更严重的情况下，可以同时冒充几个人。



## **14. Ad2Attack: Adaptive Adversarial Attack on Real-Time UAV Tracking**

Ad2Attack：无人机实时跟踪的自适应对抗攻击 cs.CV

7 pages, 7 figures, accepted by ICRA 2022

**SubmitDate**: 2022-03-03    [paper-pdf](http://arxiv.org/pdf/2203.01516v1)

**Authors**: Changhong Fu, Sihang Li, Xinnan Yuan, Junjie Ye, Ziang Cao, Fangqiang Ding

**Abstracts**: Visual tracking is adopted to extensive unmanned aerial vehicle (UAV)-related applications, which leads to a highly demanding requirement on the robustness of UAV trackers. However, adding imperceptible perturbations can easily fool the tracker and cause tracking failures. This risk is often overlooked and rarely researched at present. Therefore, to help increase awareness of the potential risk and the robustness of UAV tracking, this work proposes a novel adaptive adversarial attack approach, i.e., Ad$^2$Attack, against UAV object tracking. Specifically, adversarial examples are generated online during the resampling of the search patch image, which leads trackers to lose the target in the following frames. Ad$^2$Attack is composed of a direct downsampling module and a super-resolution upsampling module with adaptive stages. A novel optimization function is proposed for balancing the imperceptibility and efficiency of the attack. Comprehensive experiments on several well-known benchmarks and real-world conditions show the effectiveness of our attack method, which dramatically reduces the performance of the most advanced Siamese trackers.

摘要: 无人机相关应用广泛采用视觉跟踪，这对无人机跟踪器的健壮性提出了很高的要求。然而，添加不可察觉的扰动很容易欺骗跟踪器并导致跟踪失败。目前，这一风险往往被忽视，研究甚少。因此，为了提高对无人机跟踪潜在风险和鲁棒性的认识，本文提出了一种新的针对无人机目标跟踪的自适应对抗性攻击方法，即Ad$^2$攻击。具体地说，在搜索补丁图像的重采样过程中，在线生成对抗性示例，这会导致跟踪者在随后的帧中丢失目标。AD$^2$攻击由直接下采样模块和带自适应阶段的超分辨率上采样模块组成。为了平衡攻击的隐蔽性和效率，提出了一种新的优化函数。在几个著名的基准和真实世界条件下的综合实验表明，我们的攻击方法是有效的，这大大降低了最先进的暹罗跟踪器的性能。



## **15. Two Attacks On Proof-of-Stake GHOST/Ethereum**

对证明鬼/以太的两次攻击 cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01315v1)

**Authors**: Joachim Neu, Ertem Nusret Tas, David Tse

**Abstracts**: We present two attacks targeting the Proof-of-Stake (PoS) Ethereum consensus protocol. The first attack suggests a fundamental conceptual incompatibility between PoS and the Greedy Heaviest-Observed Sub-Tree (GHOST) fork choice paradigm employed by PoS Ethereum. In a nutshell, PoS allows an adversary with a vanishing amount of stake to produce an unlimited number of equivocating blocks. While most equivocating blocks will be orphaned, such orphaned `uncle blocks' still influence fork choice under the GHOST paradigm, bestowing upon the adversary devastating control over the canonical chain. While the Latest Message Driven (LMD) aspect of current PoS Ethereum prevents a straightforward application of this attack, our second attack shows how LMD specifically can be exploited to obtain a new variant of the balancing attack that overcomes a recent protocol addition that was intended to mitigate balancing-type attacks. Thus, in its current form, PoS Ethereum without and with LMD is vulnerable to our first and second attack, respectively.

摘要: 我们提出了两种针对以太共识协议的攻击。第一个攻击表明，pos和pos Etherum采用的贪婪最重的子树(GHOST)分叉选择范例之间存在根本的概念上的不兼容。简而言之，POS允许赌注逐渐消失的对手产生无限数量的模棱两可的块。虽然大多数模棱两可的块将是孤立的，但这种孤立的“叔叔块”仍然影响着幽灵范式下的叉子选择，赋予对手对正则链的毁灭性控制。虽然当前PoS Etherum的最新消息驱动(LMD)方面阻止了此攻击的直接应用，但我们的第二个攻击显示了如何专门利用LMD来获得平衡攻击的新变体，该变体克服了最近添加的旨在缓解平衡型攻击的协议。因此，在目前的形式下，没有LMD和有LMD的POS Etherum分别容易受到我们的第一次和第二次攻击。



## **16. Detecting Adversarial Perturbations in Multi-Task Perception**

多任务感知中的敌意扰动检测 cs.CV

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.01177v1)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code will be available on github. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.

摘要: 虽然深度神经网络(DNNs)在环境感知任务中取得了令人印象深刻的性能，但其对对抗性扰动的敏感性限制了其在实际应用中的应用。本文(I)提出了一种新的基于复杂视觉任务多任务感知(即深度估计和语义分割)的对抗性扰动检测方案。具体地说，通过所提取的输入图像的边缘、深度输出和分割输出之间的不一致来检测对抗性扰动。为了进一步改进这一技术，我们(Ii)在所有三种模态之间开发了一种新的边缘一致性损失，从而提高了它们的初始一致性，这反过来又支持我们的检测方案。我们通过使用各种已知攻击和图像噪声来验证我们的检测方案的有效性。此外，我们(Iii)开发了一种多任务对抗性攻击，旨在欺骗任务和我们的检测方案。在CITYSCAPES和KITTI数据集上的实验评估表明，在假阳性率为5%的假设下，高达100%的图像被正确检测为恶意扰动，这取决于扰动的强度。代码将在GitHub上提供。https://youtu.be/KKa6gOyWmH4上的一段简短视频提供了定性结果。



## **17. How to Inject Backdoors with Better Consistency: Logit Anchoring on Clean Data**

如何以更好的一致性注入后门：基于干净数据的Logit锚定 cs.LG

Accepted by ICLR 2022

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2109.01300v2)

**Authors**: Zhiyuan Zhang, Lingjuan Lyu, Weiqiang Wang, Lichao Sun, Xu Sun

**Abstracts**: Since training a large-scale backdoored model from scratch requires a large training dataset, several recent attacks have considered to inject backdoors into a trained clean model without altering model behaviors on the clean data. Previous work finds that backdoors can be injected into a trained clean model with Adversarial Weight Perturbation (AWP). Here AWPs refers to the variations of parameters that are small in backdoor learning. In this work, we observe an interesting phenomenon that the variations of parameters are always AWPs when tuning the trained clean model to inject backdoors. We further provide theoretical analysis to explain this phenomenon. We formulate the behavior of maintaining accuracy on clean data as the consistency of backdoored models, which includes both global consistency and instance-wise consistency. We extensively analyze the effects of AWPs on the consistency of backdoored models. In order to achieve better consistency, we propose a novel anchoring loss to anchor or freeze the model behaviors on the clean data, with a theoretical guarantee. Both the analytical and the empirical results validate the effectiveness of the anchoring loss in improving the consistency, especially the instance-wise consistency.

摘要: 由于从零开始训练大规模回溯模型需要大量的训练数据集，最近的几次攻击已经考虑在不改变干净数据上的模型行为的情况下向训练过的干净模型注入后门。以前的工作发现，后门可以被注入到具有对抗性权重扰动(AWP)的训练有素的干净模型中。这里的AWP指的是在后门学习中较小的参数变化。在这项工作中，我们观察到一个有趣的现象，即在调整训练好的干净模型进行后门注入时，参数的变化总是AWP。我们进一步对这一现象进行了理论分析。我们将在干净数据上保持准确性的行为表述为回溯模型的一致性，包括全局一致性和实例一致性。我们广泛地分析了AWP对回溯模型一致性的影响。为了达到更好的一致性，我们提出了一种新的锚定损失来锚定或冻结干净数据上的模型行为，并提供了理论上的保证。分析和实验结果都验证了锚定损失对提高一致性，特别是实例一致性的有效性。



## **18. Video is All You Need: Attacking PPG-based Biometric Authentication**

视频就是您需要的一切：攻击基于PPG的生物识别身份验证 cs.CR

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00928v1)

**Authors**: Lin Li, Chao Chen, Lei Pan, Jun Zhang, Yang Xiang

**Abstracts**: Unobservable physiological signals enhance biometric authentication systems. Photoplethysmography (PPG) signals are convenient owning to its ease of measurement and are usually well protected against remote adversaries in authentication. Any leaked PPG signals help adversaries compromise the biometric authentication systems, and the advent of remote PPG (rPPG) enables adversaries to acquire PPG signals through restoration. While potentially dangerous, rPPG-based attacks are overlooked because existing methods require the victim's PPG signals. This paper proposes a novel spoofing attack approach that uses the waveforms of rPPG signals extracted from video clips to fool the PPG-based biometric authentication. We develop a new PPG restoration model that does not require leaked PPG signals for adversarial attacks. Test results on state-of-art PPG-based biometric authentication show that the signals recovered through rPPG pose a severe threat to PPG-based biometric authentication.

摘要: 不可观测的生理信号增强了生物特征认证系统。光体积描记(PPG)信号由于其易于测量而非常方便，并且通常在认证时能够很好地防止远程攻击。任何泄漏的PPG信号都会帮助攻击者危害生物特征认证系统，而远程PPG(RPPG)的出现使攻击者能够通过恢复来获取PPG信号。虽然存在潜在危险，但基于rPPG的攻击被忽略了，因为现有方法需要受害者的PPG信号。提出了一种利用从视频片段中提取的rPPG信号波形来欺骗基于PPG的生物特征认证的欺骗攻击方法。我们开发了一种新的PPG恢复模型，该模型不需要泄漏的PPG信号来进行对抗性攻击。对现有的基于PPG的生物特征认证的测试结果表明，通过rPPG恢复的信号对基于PPG的生物特征认证构成了严重的威胁。



## **19. Canonical foliations of neural networks: application to robustness**

神经网络的标准叶：在鲁棒性方面的应用 stat.ML

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00922v1)

**Authors**: Eliot Tron, Nicolas Couellan, Stéphane Puechmorel

**Abstracts**: Adversarial attack is an emerging threat to the trustability of machine learning. Understanding these attacks is becoming a crucial task. We propose a new vision on neural network robustness using Riemannian geometry and foliation theory, and create a new adversarial attack by taking into account the curvature of the data space. This new adversarial attack called the "dog-leg attack" is a two-step approximation of a geodesic in the data space. The data space is treated as a (pseudo) Riemannian manifold equipped with the pullback of the Fisher Information Metric (FIM) of the neural network. In most cases, this metric is only semi-definite and its kernel becomes a central object to study. A canonical foliation is derived from this kernel. The curvature of the foliation's leaves gives the appropriate correction to get a two-step approximation of the geodesic and hence a new efficient adversarial attack. Our attack is tested on a toy example, a neural network trained to mimic the $\texttt{Xor}$ function, and demonstrates better results that the state of the art attack presented by Zhao et al. (2019).

摘要: 对抗性攻击是对机器学习可信性的一种新兴威胁。了解这些攻击正在成为一项至关重要的任务。我们利用黎曼几何和分层理论提出了一种新的神经网络鲁棒性的观点，并通过考虑数据空间的曲率来创建一种新的对抗性攻击。这种新的对抗性攻击被称为“狗腿攻击”，它是数据空间中测地线的两步近似。将数据空间处理为带有神经网络Fisher信息度量(FIM)回撤的(伪)黎曼流形。在大多数情况下，该度量只是半定的，其内核成为研究的中心对象。一个典型的叶理就是从这个核中衍生出来的。叶面的曲率给出了适当的修正，以得到测地线的两步近似，从而得到一种新的有效的对抗性攻击。我们的攻击在一个玩具示例上进行了测试，该神经网络被训练成模仿$\texttt{XOR}$函数，并显示了比赵等人提出的最新攻击更好的结果。(2019年)。



## **20. MIAShield: Defending Membership Inference Attacks via Preemptive Exclusion of Members**

MIAShield：通过抢占排除成员来防御成员推断攻击 cs.CR

21 pages, 17 figures, 10 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2203.00915v1)

**Authors**: Ismat Jarin, Birhanu Eshete

**Abstracts**: In membership inference attacks (MIAs), an adversary observes the predictions of a model to determine whether a sample is part of the model's training data. Existing MIA defenses conceal the presence of a target sample through strong regularization, knowledge distillation, confidence masking, or differential privacy.   We propose MIAShield, a new MIA defense based on preemptive exclusion of member samples instead of masking the presence of a member. The key insight in MIAShield is weakening the strong membership signal that stems from the presence of a target sample by preemptively excluding it at prediction time without compromising model utility. To that end, we design and evaluate a suite of preemptive exclusion oracles leveraging model-confidence, exact or approximate sample signature, and learning-based exclusion of member data points. To be practical, MIAShield splits a training data into disjoint subsets and trains each subset to build an ensemble of models. The disjointedness of subsets ensures that a target sample belongs to only one subset, which isolates the sample to facilitate the preemptive exclusion goal.   We evaluate MIAShield on three benchmark image classification datasets. We show that MIAShield effectively mitigates membership inference (near random guess) for a wide range of MIAs, achieves far better privacy-utility trade-off compared with state-of-the-art defenses, and remains resilient against an adaptive adversary.

摘要: 在成员关系推断攻击(MIA)中，对手通过观察模型的预测来确定样本是否为模型训练数据的一部分。现有的MIA防御通过强正则化、知识提炼、置信度掩蔽或差分隐私来隐藏目标样本的存在。我们提出了MIAShield，一种新的基于抢占排除成员样本的MIA防御，而不是掩盖成员的存在。MIAShield的关键洞察力是在不影响模型效用的情况下，通过在预测时先发制人地排除目标样本，来削弱由于目标样本的存在而产生的强烈成员资格信号。为此，我们设计并评估了一套抢占式排除预言，利用模型置信度、精确或近似样本签名以及基于学习的成员数据点排除。实际上，MIAShield将训练数据分割成不相交的子集，并训练每个子集来构建模型集成。子集的不相交性保证了目标样本只属于一个子集，从而隔离了样本，有利于抢占排除目标的实现。我们在三个基准图像分类数据集上对MIAShield进行了评估。我们表明，MIAShield有效地缓解了大范围MIA的成员推断(近乎随机猜测)，与最先进的防御措施相比，实现了更好的隐私效用权衡，并且对自适应对手保持了弹性。



## **21. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

ECG-ATK-GAN：基于条件生成对抗网络的心电对抗攻击鲁棒性 eess.SP

10 pages, 3 figures, 3 tables

**SubmitDate**: 2022-03-02    [paper-pdf](http://arxiv.org/pdf/2110.09983v2)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce a novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and two novel blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.

摘要: 从ECG中自动检测心律失常需要一个健壮可靠的系统，该系统在电干扰下仍能保持高精度。许多机器学习方法在从心电图中分类心律失常方面已经达到了人类的水平。然而，这些体系结构容易受到敌意攻击，这些攻击会降低模型的准确性，从而导致心电信号的误分类。对抗性攻击是注入到原始数据中的小的、精心设计的扰动，它显示了信号的非分布转移，以误分类正确的类别。因此，滥用这些扰动的虚假住院和保险欺诈引起了安全担忧。为了缓解这一问题，我们引入了一种新的条件生成对抗网络(GAN)，该网络对敌意攻击的心电信号具有较强的鲁棒性，并保持了较高的准确率。我们的体系结构集成了一个新的类别加权目标函数来识别对抗性扰动，以及两个新的块来识别和组合学习过程中信号的非分布偏移，以准确地分类各种心律失常类型。此外，我们在六种不同的白盒和黑盒攻击上对我们的体系结构进行了基准测试，并在两个公开可用的ECG心律失常数据集上与其他最近提出的心律失常分类模型进行了比较。实验证明，该模型对心律失常的分类准确率较高，对这种对抗性攻击具有较强的鲁棒性。



## **22. Proceedings of the Artificial Intelligence for Cyber Security (AICS) Workshop at AAAI 2022**

2022年AAAI 2022年网络安全人工智能(AICS)研讨会论文集 cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.14010v2)

**Authors**: James Holt, Edward Raff, Ahmad Ridley, Dennis Ross, Arunesh Sinha, Diane Staheli, William Streilen, Milind Tambe, Yevgeniy Vorobeychik, Allan Wollaber

**Abstracts**: The workshop will focus on the application of AI to problems in cyber security. Cyber systems generate large volumes of data, utilizing this effectively is beyond human capabilities. Additionally, adversaries continue to develop new attacks. Hence, AI methods are required to understand and protect the cyber domain. These challenges are widely studied in enterprise networks, but there are many gaps in research and practice as well as novel problems in other domains.   In general, AI techniques are still not widely adopted in the real world. Reasons include: (1) a lack of certification of AI for security, (2) a lack of formal study of the implications of practical constraints (e.g., power, memory, storage) for AI systems in the cyber domain, (3) known vulnerabilities such as evasion, poisoning attacks, (4) lack of meaningful explanations for security analysts, and (5) lack of analyst trust in AI solutions. There is a need for the research community to develop novel solutions for these practical issues.

摘要: 研讨会将重点讨论人工智能在网络安全问题上的应用。网络系统产生了大量的数据，有效地利用这些数据超出了人类的能力范围。此外，对手还在继续开发新的攻击。因此，需要人工智能方法来理解和保护网络领域。这些挑战在企业网络中得到了广泛的研究，但在研究和实践中还存在许多差距，在其他领域也出现了一些新的问题。总的来说，人工智能技术在现实世界中仍然没有被广泛采用。原因包括：(1)缺乏对人工智能安全的认证，(2)缺乏对网络领域中实际限制(例如，电力、内存、存储)对人工智能系统的影响的正式研究，(3)已知的漏洞，如逃避、中毒攻击，(4)缺乏对安全分析师的有意义的解释，以及(5)分析师对人工智能解决方案缺乏信任。研究界需要为这些实际问题开发新的解决方案。



## **23. Beyond Gradients: Exploiting Adversarial Priors in Model Inversion Attacks**

超越梯度：在模型反转攻击中利用对抗性先验 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00481v1)

**Authors**: Dmitrii Usynin, Daniel Rueckert, Georgios Kaissis

**Abstracts**: Collaborative machine learning settings like federated learning can be susceptible to adversarial interference and attacks. One class of such attacks is termed model inversion attacks, characterised by the adversary reverse-engineering the model to extract representations and thus disclose the training data. Prior implementations of this attack typically only rely on the captured data (i.e. the shared gradients) and do not exploit the data the adversary themselves control as part of the training consortium. In this work, we propose a novel model inversion framework that builds on the foundations of gradient-based model inversion attacks, but additionally relies on matching the features and the style of the reconstructed image to data that is controlled by an adversary. Our technique outperforms existing gradient-based approaches both qualitatively and quantitatively, while still maintaining the same honest-but-curious threat model, allowing the adversary to obtain enhanced reconstructions while remaining concealed.

摘要: 协作式机器学习环境(如联合学习)很容易受到敌意干扰和攻击。一类这样的攻击被称为模型反转攻击，其特征是对手对模型进行逆向工程以提取表示，从而泄露训练数据。该攻击的先前实现通常仅依赖于捕获的数据(即共享梯度)，并且不利用对手自己控制的数据作为训练联盟的一部分。在这项工作中，我们提出了一种新的模型反演框架，它建立在基于梯度的模型反演攻击的基础上，但另外还依赖于将重建图像的特征和样式与对手控制的数据进行匹配。我们的技术在质量和数量上都优于现有的基于梯度的方法，同时仍然保持相同的诚实但好奇的威胁模型，允许攻击者在保持隐蔽的情况下获得增强的重建。



## **24. RAB: Provable Robustness Against Backdoor Attacks**

RAB：针对后门攻击的可证明的健壮性 cs.LG

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2003.08904v6)

**Authors**: Maurice Weber, Xiaojun Xu, Bojan Karlaš, Ce Zhang, Bo Li

**Abstracts**: Recent studies have shown that deep neural networks are vulnerable to adversarial attacks, including evasion and backdoor (poisoning) attacks. On the defense side, there have been intensive efforts on improving both empirical and provable robustness against evasion attacks; however, provable robustness against backdoor attacks still remains largely unexplored. In this paper, we focus on certifying the machine learning model robustness against general threat models, especially backdoor attacks. We first provide a unified framework via randomized smoothing techniques and show how it can be instantiated to certify the robustness against both evasion and backdoor attacks. We then propose the first robust training process, RAB, to smooth the trained model and certify its robustness against backdoor attacks. We theoretically prove the robustness bound for machine learning models trained with RAB, and prove that our robustness bound is tight. We derive the robustness conditions for different smoothing distributions including Gaussian and uniform distributions. In addition, we theoretically show that it is possible to train the robust smoothed models efficiently for simple models such as K-nearest neighbor classifiers, and we propose an exact smooth-training algorithm which eliminates the need to sample from a noise distribution for such models. Empirically, we conduct comprehensive experiments for different machine learning models such as DNNs and K-NN models on MNIST, CIFAR-10, and ImageNette datasets and provide the first benchmark for certified robustness against backdoor attacks. In addition, we evaluate K-NN models on a spambase tabular dataset to demonstrate the advantages of the proposed exact algorithm. Both the theoretic analysis and the comprehensive evaluation on diverse ML models and datasets shed lights on further robust learning strategies against general training time attacks.

摘要: 最近的研究表明，深层神经网络容易受到敌意攻击，包括逃避和后门(中毒)攻击。在防御方面，已经进行了密集的努力来提高针对规避攻击的经验性和可证明的健壮性；然而，针对后门攻击的可证明的健壮性在很大程度上仍未得到探索。在本文中，我们重点验证机器学习模型对一般威胁模型，特别是后门攻击的鲁棒性。我们首先通过随机平滑技术提供了一个统一的框架，并展示了如何将其实例化来证明对规避和后门攻击的鲁棒性。然后，我们提出了第一个鲁棒训练过程RAB，以平滑训练的模型并证明其对后门攻击的鲁棒性。从理论上证明了RAB训练的机器学习模型的稳健界，并证明了我们的稳健界是紧的。我们推导了不同平滑分布(包括高斯分布和均匀分布)的稳健性条件。此外，我们从理论上证明了对于K近邻分类器等简单模型，可以有效地训练鲁棒平滑模型，并提出了一种精确的平滑训练算法，该算法消除了对此类模型从噪声分布中采样的需要。经验上，我们在MNIST、CIFAR-10和ImageNette数据集上对不同的机器学习模型(如DNNS和K-NN模型)进行了全面的实验，并提供了第一个经验证的针对后门攻击的健壮性基准。此外，我们在垃圾邮件库表格数据集上对K-NN模型进行了评估，以展示所提出的精确算法的优势。理论分析和对不同ML模型和数据集的综合评价，为进一步研究抗一般训练时间攻击的鲁棒学习策略提供了理论依据。



## **25. Adversarial samples for deep monocular 6D object pose estimation**

用于深部单目6维目标姿态估计的对抗性样本 cs.CV

15 pages. arXiv admin note: text overlap with arXiv:2105.14291 by  other authors

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2203.00302v1)

**Authors**: Jinlai Zhang, Weiming Li, Shuang Liang, Hao Wang, Jihong Zhu

**Abstracts**: Estimating object 6D pose from an RGB image is important for many real-world applications such as autonomous driving and robotic grasping, where robustness of the estimation is crucial. In this work, for the first time, we study adversarial samples that can fool state-of-the-art (SOTA) deep learning based 6D pose estimation models. In particular, we propose a Unified 6D pose estimation Attack, namely U6DA, which can successfully attack all the three main categories of models for 6D pose estimation. The key idea of our U6DA is to fool the models to predict wrong results for object shapes that are essential for correct 6D pose estimation. Specifically, we explore a transfer-based black-box attack to 6D pose estimation. By shifting the segmentation attention map away from its original position, adversarial samples are crafted. We show that such adversarial samples are not only effective for the direct 6D pose estimation models, but also able to attack the two-stage based models regardless of their robust RANSAC modules. Extensive experiments were conducted to demonstrate the effectiveness of our U6DA with large-scale public benchmarks. We also introduce a new U6DA-Linemod dataset for robustness study of the 6D pose estimation task. Our codes and dataset will be available at \url{https://github.com/cuge1995/U6DA}.

摘要: 从RGB图像估计物体6D姿态对于许多真实世界的应用非常重要，例如自动驾驶和机器人抓取，其中估计的健壮性至关重要。在这项工作中，我们首次研究了可以欺骗基于SOTA深度学习的6D姿态估计模型的对抗性样本。特别地，我们提出了一种统一的6D位姿估计攻击，即U6DA，它可以成功地攻击所有三种主要的6D位姿估计模型。我们的U6DA的关键思想是愚弄模型来预测对象形状的错误结果，这对于正确的6D姿势估计是必不可少的。具体地说，我们探索了一种基于传输的黑盒攻击来进行6D位姿估计。通过将分割注意图从其原始位置移开，可以制作对抗性样本。结果表明，这种对抗性样本不仅对直接6D姿态估计模型有效，而且能够攻击基于两阶段的模型，而不考虑其稳健的RANSAC模型。通过大规模的公共基准测试，验证了我们的U6DA算法的有效性。我们还介绍了一个新的U6DA-Linemod数据集，用于6D位姿估计任务的鲁棒性研究。我们的代码和数据集将在\url{https://github.com/cuge1995/U6DA}.



## **26. Towards Robust Stacked Capsule Autoencoder with Hybrid Adversarial Training**

基于混合对抗训练的鲁棒堆叠式胶囊自动编码器 cs.CV

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.13755v2)

**Authors**: Jiazhu Dai, Siwei Xiong

**Abstracts**: Capsule networks (CapsNets) are new neural networks that classify images based on the spatial relationships of features. By analyzing the pose of features and their relative positions, it is more capable to recognize images after affine transformation. The stacked capsule autoencoder (SCAE) is a state-of-the-art CapsNet, and achieved unsupervised classification of CapsNets for the first time. However, the security vulnerabilities and the robustness of the SCAE has rarely been explored. In this paper, we propose an evasion attack against SCAE, where the attacker can generate adversarial perturbations based on reducing the contribution of the object capsules in SCAE related to the original category of the image. The adversarial perturbations are then applied to the original images, and the perturbed images will be misclassified. Furthermore, we propose a defense method called Hybrid Adversarial Training (HAT) against such evasion attacks. HAT makes use of adversarial training and adversarial distillation to achieve better robustness and stability. We evaluate the defense method and the experimental results show that the refined SCAE model can achieve 82.14% classification accuracy under evasion attack. The source code is available at https://github.com/FrostbiteXSW/SCAE_Defense.

摘要: 胶囊网络(CapsNets)是一种基于特征空间关系对图像进行分类的新型神经网络。通过分析特征的姿态及其相对位置，使仿射变换后的图像具有更强的识别能力。堆叠式胶囊自动编码器(SCAE)是一种先进的CapsNet，首次实现了CapsNet的无监督分类。然而，SCAE的安全漏洞和健壮性很少被研究。本文提出了一种针对SCAE的规避攻击，攻击者可以通过减少SCAE中对象胶囊相对于图像原始类别的贡献来产生敌意扰动。然后将对抗性扰动应用于原始图像，并且扰动图像将被错误分类。此外，针对此类逃避攻击，我们提出了一种称为混合对抗训练(HAT)的防御方法。HAT利用对抗性训练和对抗性蒸馏来实现更好的健壮性和稳定性。实验结果表明，改进后的SCAE模型在规避攻击下可以达到82.14%的分类正确率。源代码可以在https://github.com/FrostbiteXSW/SCAE_Defense.上找到



## **27. Towards Effective and Robust Neural Trojan Defenses via Input Filtering**

通过输入过滤实现高效、健壮的神经木马防御 cs.CR

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2202.12154v2)

**Authors**: Kien Do, Haripriya Harikumar, Hung Le, Dung Nguyen, Truyen Tran, Santu Rana, Dang Nguyen, Willy Susilo, Svetha Venkatesh

**Abstracts**: Trojan attacks on deep neural networks are both dangerous and surreptitious. Over the past few years, Trojan attacks have advanced from using only a simple trigger and targeting only one class to using many sophisticated triggers and targeting multiple classes. However, Trojan defenses have not caught up with this development. Most defense methods still make out-of-date assumptions about Trojan triggers and target classes, thus, can be easily circumvented by modern Trojan attacks. In this paper, we advocate general defenses that are effective and robust against various Trojan attacks and propose two novel "filtering" defenses with these characteristics called Variational Input Filtering (VIF) and Adversarial Input Filtering (AIF). VIF and AIF leverage variational inference and adversarial training respectively to purify all potential Trojan triggers in the input at run time without making any assumption about their numbers and forms. We further extend "filtering" to "filtering-then-contrasting" - a new defense mechanism that helps avoid the drop in classification accuracy on clean data caused by filtering. Extensive experimental results show that our proposed defenses significantly outperform 4 well-known defenses in mitigating 5 different Trojan attacks including the two state-of-the-art which defeat many strong defenses.

摘要: 特洛伊木马对深层神经网络的攻击既危险又隐蔽。在过去的几年里，特洛伊木马攻击已经从只使用一个简单的触发器，只针对一个类，发展到使用许多复杂的触发器，针对多个类。然而，特洛伊木马防御并没有跟上这一发展。大多数防御方法仍然对木马触发器和目标类做出过时的假设，因此很容易被现代木马攻击所规避。在本文中，我们提倡对各种特洛伊木马攻击有效和健壮的通用防御，并提出了两种具有这些特性的新型“过滤”防御方案，称为变量输入过滤(VIF)和对抗性输入过滤(AIF)。VIF和AIF分别利用变分推理和对抗性训练在运行时净化输入中所有潜在的特洛伊木马触发器，而不对其数量和形式做出任何假设。我们将“过滤”进一步扩展为“过滤-然后对比”-一种新的防御机制，它有助于避免过滤导致的干净数据分类准确率的下降。广泛的实验结果表明，我们提出的防御方案在缓解5种不同的特洛伊木马攻击方面明显优于4种众所周知的防御方案，其中包括两种最先进的防御方案，它们击败了许多强大的防御方案。



## **28. Adversarial Attack Framework on Graph Embedding Models with Limited Knowledge**

基于有限知识的图嵌入模型的对抗性攻击框架 cs.LG

Journal extension of GF-Attack, accepted by TKDE

**SubmitDate**: 2022-03-01    [paper-pdf](http://arxiv.org/pdf/2105.12419v2)

**Authors**: Heng Chang, Yu Rong, Tingyang Xu, Wenbing Huang, Honglei Zhang, Peng Cui, Xin Wang, Wenwu Zhu, Junzhou Huang

**Abstracts**: With the success of the graph embedding model in both academic and industry areas, the robustness of graph embedding against adversarial attack inevitably becomes a crucial problem in graph learning. Existing works usually perform the attack in a white-box fashion: they need to access the predictions/labels to construct their adversarial loss. However, the inaccessibility of predictions/labels makes the white-box attack impractical to a real graph learning system. This paper promotes current frameworks in a more general and flexible sense -- we demand to attack various kinds of graph embedding models with black-box driven. We investigate the theoretical connections between graph signal processing and graph embedding models and formulate the graph embedding model as a general graph signal process with a corresponding graph filter. Therefore, we design a generalized adversarial attacker: GF-Attack. Without accessing any labels and model predictions, GF-Attack can perform the attack directly on the graph filter in a black-box fashion. We further prove that GF-Attack can perform an effective attack without knowing the number of layers of graph embedding models. To validate the generalization of GF-Attack, we construct the attacker on four popular graph embedding models. Extensive experiments validate the effectiveness of GF-Attack on several benchmark datasets.

摘要: 随着图嵌入模型在学术界和工业界的成功应用，图嵌入对敌意攻击的鲁棒性不可避免地成为图学习中的一个关键问题。现有的作品通常以白盒方式进行攻击：它们需要访问预测/标签来构建它们的对抗性损失。然而，预测/标签的不可访问性使得白盒攻击对于真实的图学习系统来说是不切实际的。本文在更一般、更灵活的意义上提升了现有的框架--我们要求攻击各种黑盒驱动的图嵌入模型。我们研究了图信号处理和图嵌入模型之间的理论联系，并将图嵌入模型表示为具有相应图过滤的一般图信号过程。因此，我们设计了一种广义对抗性攻击者：GF-攻击。GF-Attack在不访问任何标签和模型预测的情况下，可以黑盒方式直接对图过滤进行攻击。进一步证明了GF-攻击可以在不知道图嵌入模型层数的情况下进行有效的攻击。为了验证GF-攻击的泛化能力，我们在四种流行的图嵌入模型上构造了攻击者。在几个基准数据集上的大量实验验证了GF-攻击的有效性。



## **29. Load-Altering Attacks Against Power Grids under COVID-19 Low-Inertia Conditions**

冠状病毒低惯性条件下电网的变负荷攻击 cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2201.10505v2)

**Authors**: Subhash Lakshminarayana, Juan Ospina, Charalambos Konstantinou

**Abstracts**: The COVID-19 pandemic has impacted our society by forcing shutdowns and shifting the way people interacted worldwide. In relation to the impacts on the electric grid, it created a significant decrease in energy demands across the globe. Recent studies have shown that the low demand conditions caused by COVID-19 lockdowns combined with large renewable generation have resulted in extremely low-inertia grid conditions. In this work, we examine how an attacker could exploit these {scenarios} to cause unsafe grid operating conditions by executing load-altering attacks (LAAs) targeted at compromising hundreds of thousands of IoT-connected high-wattage loads in low-inertia power systems. Our study focuses on analyzing the impact of the COVID-19 mitigation measures on U.S. regional transmission operators (RTOs), formulating a plausible and realistic least-effort LAA targeted at transmission systems with low-inertia conditions, and evaluating the probability of these large-scale LAAs. Theoretical and simulation results are presented based on the WSCC 9-bus {and IEEE 118-bus} test systems. Results demonstrate how adversaries could provoke major frequency disturbances by targeting vulnerable load buses in low-inertia systems and offer insights into how the temporal fluctuations of renewable energy sources, considering generation scheduling, impact the grid's vulnerability to LAAs.

摘要: 冠状病毒大流行已经通过迫使政府关门和改变世界各地人们互动的方式影响了我们的社会。在对电网的影响方面，它造成了全球能源需求的显著下降。最近的研究表明，冠状病毒关闭造成的低需求条件，加上大量的可再生能源发电，导致了极低的惯性电网条件。在这项工作中，我们研究了攻击者如何利用这些{场景}通过执行负载改变攻击(LAA)来造成不安全的电网运行条件，LAA的目标是危害低惯性电力系统中数十万个物联网连接的高瓦数负载。我们的研究重点是分析冠状病毒缓解措施对美国区域传输运营商(RTO)的影响，针对低惯性条件下的传输系统制定合理而现实的最小工作量LAA，并评估这些大规模LAA的可能性。给出了基于WSCC9母线{和IEEE118母线}测试系统的理论和仿真结果。结果表明，对手如何通过瞄准低惯性系统中脆弱的负载母线来引发重大频率扰动，并为考虑发电调度的可再生能源的时间波动如何影响电网的LAAS脆弱性提供了深入的见解。



## **30. MaMaDroid2.0 -- The Holes of Control Flow Graphs**

MaMaDroid2.0--控制流图的漏洞 cs.CR

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13922v1)

**Authors**: Harel Berger, Chen Hajaj, Enrico Mariconti, Amit Dvir

**Abstracts**: Android malware is a continuously expanding threat to billions of mobile users around the globe. Detection systems are updated constantly to address these threats. However, a backlash takes the form of evasion attacks, in which an adversary changes malicious samples such that those samples will be misclassified as benign. This paper fully inspects a well-known Android malware detection system, MaMaDroid, which analyzes the control flow graph of the application. Changes to the portion of benign samples in the train set and models are considered to see their effect on the classifier. The changes in the ratio between benign and malicious samples have a clear effect on each one of the models, resulting in a decrease of more than 40% in their detection rate. Moreover, adopted ML models are implemented as well, including 5-NN, Decision Tree, and Adaboost. Exploration of the six models reveals a typical behavior in different cases, of tree-based models and distance-based models. Moreover, three novel attacks that manipulate the CFG and their detection rates are described for each one of the targeted models. The attacks decrease the detection rate of most of the models to 0%, with regards to different ratios of benign to malicious apps. As a result, a new version of MaMaDroid is engineered. This model fuses the CFG of the app and static analysis of features of the app. This improved model is proved to be robust against evasion attacks targeting both CFG-based models and static analysis models, achieving a detection rate of more than 90% against each one of the attacks.

摘要: Android恶意软件正在对全球数十亿移动用户构成持续扩大的威胁。探测系统会不断更新，以应对这些威胁。然而，反弹采取逃避攻击的形式，在这种攻击中，敌手更改恶意样本，使这些样本被错误地归类为良性样本。本文全面考察了著名的Android恶意软件检测系统MaMaDroid，该系统分析了应用程序的控制流图。考虑对训练集和模型中良性样本部分的改变，以查看它们对分类器的影响。良性样本和恶意样本比例的变化对每个模型都有明显的影响，导致它们的检测率下降了40%以上。此外，还实现了所采用的ML模型，包括5-NN、决策树和Adboost。对这六种模型的研究揭示了基于树的模型和基于距离的模型在不同情况下的典型行为。此外，对于每个目标模型，描述了操纵CFG的三种新攻击及其检测率。这些攻击将大多数模型的检测率降低到0%，涉及到不同比例的良性应用程序和恶意应用程序。因此，一个新版本的MaMaDroid被设计出来。这种模式融合了APP的CFG和静电对APP功能的分析。实验证明，该改进模型对基于cfg模型和静电分析模型的规避攻击均具有较强的鲁棒性，对每种攻击的检测率均在90%以上。



## **31. Formally verified asymptotic consensus in robust networks**

鲁棒网络中渐近一致性的形式化验证 cs.PL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13833v1)

**Authors**: Mohit Tekriwal, Avi Tachna-Fram, Jean-Baptiste Jeannin, Manos Kapritsos, Dimitra Panagou

**Abstracts**: Distributed architectures are used to improve performance and reliability of various systems. An important capability of a distributed architecture is the ability to reach consensus among all its nodes. To achieve this, several consensus algorithms have been proposed for various scenarii, and many of these algorithms come with proofs of correctness that are not mechanically checked. Unfortunately, those proofs are known to be intricate and prone to errors.   In this paper, we formalize and mechanically check a consensus algorithm widely used in the distributed controls community: the Weighted-Mean Subsequence Reduced (W-MSR) algorithm proposed by Le Blanc et al. This algorithm provides a way to achieve asymptotic consensus in a distributed controls scenario in the presence of adversarial agents (attackers) that may not update their states based on the nominal consensus protocol, and may share inaccurate information with their neighbors. Using the Coq proof assistant, we formalize the necessary and sufficient conditions required to achieve resilient asymptotic consensus under the assumed attacker model. We leverage the existing Coq formalizations of graph theory, finite sets and sequences of the mathcomp library for our development. To our knowledge, this is the first mechanical proof of an asymptotic consensus algorithm. During the formalization, we clarify several imprecisions in the paper proof, including an imprecision on quantifiers in the main theorem.

摘要: 分布式体系结构被用来提高各种系统的性能和可靠性。分布式体系结构的一项重要功能是在其所有节点之间达成共识的能力。为了实现这一点，已经针对不同的场景提出了几种共识算法，其中许多算法都带有不经过机械检查的正确性证明。不幸的是，众所周知，这些证明错综复杂，容易出错。本文对一种广泛应用于分布式控制领域的一致性算法--由Le Blanc等人提出的加权平均子序列简化(W-MSR)算法进行了形式化和机械检验。该算法提供了一种在分布式控制场景中获得渐近共识的方法，在存在可能不会基于名义共识协议更新其状态并且可能与其邻居共享不准确信息的敌对代理(攻击者)存在的情况下，该算法提供了一种获得渐近共识的方法。利用CoQ证明助手，我们形式化了在假设的攻击者模型下实现弹性渐近共识所需的充要条件。我们利用现有的图论、有限集和Mathcomp库序列的CoQ形式化进行开发。据我们所知，这是渐近一致算法的第一个机械证明。在形式化过程中，我们澄清了论文证明中的几个不精确之处，包括主要定理中关于量词的不精确。



## **32. Robust Textual Embedding against Word-level Adversarial Attacks**

抵抗词级敌意攻击的鲁棒文本嵌入 cs.CL

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13817v1)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull the words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows the great potential of improving the textual robustness through robust word embedding.

摘要: 我们将自然语言处理模型的脆弱性归因于相似的输入在嵌入空间被转换为不相似的表示，从而导致输出不一致，并提出了一种新的鲁棒训练方法，称为快速三重度量学习(Fast Triplet Metric Learning，FTML)。具体地说，我们认为原始样本应该与对手样本具有相似的表示，并将其表示与其他样本区分开来，以获得更好的鲁棒性。为此，我们将三元组度量学习引入标准训练中，将单词拉近其正样本(即同义词)，并在嵌入空间中推开其负样本(即非同义词)。大量实验表明，该算法在保持好胜原始样本分类准确率的同时，能显着提高模型对各种高级敌意攻击的鲁棒性。此外，我们的方法是高效的，因为它只需要调整嵌入，并且对标准训练的开销很小。我们的工作显示了通过鲁棒的词嵌入来提高文本鲁棒性的巨大潜力。



## **33. On the Robustness of CountSketch to Adaptive Inputs**

关于CountSketch对自适应输入的鲁棒性 cs.DS

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13736v1)

**Authors**: Edith Cohen, Xin Lyu, Jelani Nelson, Tamás Sarlós, Moshe Shechner, Uri Stemmer

**Abstracts**: CountSketch is a popular dimensionality reduction technique that maps vectors to a lower dimension using randomized linear measurements. The sketch supports recovering $\ell_2$-heavy hitters of a vector (entries with $v[i]^2 \geq \frac{1}{k}\|\boldsymbol{v}\|^2_2$). We study the robustness of the sketch in adaptive settings where input vectors may depend on the output from prior inputs. Adaptive settings arise in processes with feedback or with adversarial attacks. We show that the classic estimator is not robust, and can be attacked with a number of queries of the order of the sketch size. We propose a robust estimator (for a slightly modified sketch) that allows for quadratic number of queries in the sketch size, which is an improvement factor of $\sqrt{k}$ (for $k$ heavy hitters) over prior work.

摘要: CountSketch是一种流行的降维技术，它使用随机化的线性测量将向量映射到较低的维度。草图支持恢复向量的$\ell_2$重打击数(具有$v[i]^2\geq\frac{1}{k}\|\boldSymbol{v}\^2_2$的条目)。我们研究了草图在自适应环境下的鲁棒性，其中输入向量可能依赖于先前输入的输出。自适应设置出现在具有反馈或敌意攻击的过程中。我们证明了经典的估计器是不稳健的，并且可以用草图大小的数量级的一些查询来攻击。我们提出了一个稳健的估计器(对于略微修改的草图)，它允许草图大小的二次查询数，这比以前的工作提高了$\sqrt{k}$(对于$k$重命中者是$\sqrt{k}$)。



## **34. An Empirical Study on the Intrinsic Privacy of SGD**

关于SGD内在隐私性的实证研究 cs.LG

21 pages, 11 figures, 8 tables

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/1912.02919v4)

**Authors**: Stephanie L. Hyland, Shruti Tople

**Abstracts**: Introducing noise in the training of machine learning systems is a powerful way to protect individual privacy via differential privacy guarantees, but comes at a cost to utility. This work looks at whether the inherent randomness of stochastic gradient descent (SGD) could contribute to privacy, effectively reducing the amount of \emph{additional} noise required to achieve a given privacy guarantee. We conduct a large-scale empirical study to examine this question. Training a grid of over 120,000 models across four datasets (tabular and images) on convex and non-convex objectives, we demonstrate that the random seed has a larger impact on model weights than any individual training example. We test the distribution over weights induced by the seed, finding that the simple convex case can be modelled with a multivariate Gaussian posterior, while neural networks exhibit multi-modal and non-Gaussian weight distributions. By casting convex SGD as a Gaussian mechanism, we then estimate an `intrinsic' data-dependent $\epsilon_i(\mathcal{D})$, finding values as low as 6.3, dropping to 1.9 using empirical estimates. We use a membership inference attack to estimate $\epsilon$ for non-convex SGD and demonstrate that hiding the random seed from the adversary results in a statistically significant reduction in attack performance, corresponding to a reduction in the effective $\epsilon$. These results provide empirical evidence that SGD exhibits appreciable variability relative to its dataset sensitivity, and this `intrinsic noise' has the potential to be leveraged to improve the utility of privacy-preserving machine learning.

摘要: 在机器学习系统的训练中引入噪声是通过不同的隐私保证来保护个人隐私的一种强有力的方式，但这是以实用为代价的。这项工作着眼于随机梯度下降(SGD)固有的随机性是否有助于隐私，从而有效地减少实现给定隐私保证所需的{附加}噪声量。为了检验这一问题，我们进行了大规模的实证研究。通过对四个数据集(表格和图像)上超过12万个模型的网格进行凸和非凸目标的训练，我们证明了随机种子对模型权重的影响比任何单个训练示例都要大。我们对种子引起的权值分布进行了测试，发现简单的凸情况可以用多元高斯后验分布来建模，而神经网络则表现出多峰和非高斯权重分布。通过把凸SGD看作一个高斯机制，然后我们估计一个“固有的”依赖于数据的$\epsilon_i(\mathcal{D})$，得到低至6.3的值，用经验估计降到1.9。我们使用成员关系推理攻击来估计非凸SGD的$\epsilon$，并证明了对对手隐藏随机种子会导致攻击性能在统计上显著降低，相应地，有效的$\epsilon$也会降低。这些结果提供了经验证据，表明SGD相对于其数据集敏感度表现出明显的变异性，并且这种“固有噪声”有可能被用来提高隐私保护机器学习的效用。



## **35. Enhance transferability of adversarial examples with model architecture**

利用模型架构增强对抗性实例的可移植性 cs.LG

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13625v1)

**Authors**: Mingyuan Fan, Wenzhong Guo, Shengxing Yu, Zuobin Ying, Ximeng Liu

**Abstracts**: Transferability of adversarial examples is of critical importance to launch black-box adversarial attacks, where attackers are only allowed to access the output of the target model. However, under such a challenging but practical setting, the crafted adversarial examples are always prone to overfitting to the proxy model employed, presenting poor transferability. In this paper, we suggest alleviating the overfitting issue from a novel perspective, i.e., designing a fitted model architecture. Specifically, delving the bottom of the cause of poor transferability, we arguably decompose and reconstruct the existing model architecture into an effective model architecture, namely multi-track model architecture (MMA). The adversarial examples crafted on the MMA can maximumly relieve the effect of model-specified features to it and toward the vulnerable directions adopted by diverse architectures. Extensive experimental evaluation demonstrates that the transferability of adversarial examples based on the MMA significantly surpass other state-of-the-art model architectures by up to 40% with comparable overhead.

摘要: 对抗性示例的可转移性对于发起黑盒对抗性攻击至关重要，在黑盒对抗性攻击中，攻击者只能访问目标模型的输出。然而，在这样一个具有挑战性但实用的背景下，精心制作的对抗性例子往往容易与所采用的代理模型过度拟合，表现出较差的可移植性。在本文中，我们建议从一个新的角度来缓解过度拟合问题，即设计一个合适的模型体系结构。具体地说，通过深入挖掘可移植性差的底层原因，我们可以将现有的模型体系结构分解和重构为一种有效的模型体系结构，即多轨道模型体系结构(MMA)。在MMA上制作的对抗性示例可以最大限度地缓解模型指定功能对MMA的影响，以及对不同体系结构采用的易受攻击方向的影响。广泛的实验评估表明，基于MMA的对抗性示例的可转移性在开销相当的情况下大大超过了其他最先进的模型体系结构，最高可达40%。



## **36. GRAPHITE: Generating Automatic Physical Examples for Machine-Learning Attacks on Computer Vision Systems**

石墨：为机器学习攻击计算机视觉系统生成自动物理示例 cs.CR

IEEE European Symposium on Security and Privacy 2022 (EuroS&P 2022)

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2002.07088v6)

**Authors**: Ryan Feng, Neal Mangaokar, Jiefeng Chen, Earlence Fernandes, Somesh Jha, Atul Prakash

**Abstracts**: This paper investigates an adversary's ease of attack in generating adversarial examples for real-world scenarios. We address three key requirements for practical attacks for the real-world: 1) automatically constraining the size and shape of the attack so it can be applied with stickers, 2) transform-robustness, i.e., robustness of a attack to environmental physical variations such as viewpoint and lighting changes, and 3) supporting attacks in not only white-box, but also black-box hard-label scenarios, so that the adversary can attack proprietary models. In this work, we propose GRAPHITE, an efficient and general framework for generating attacks that satisfy the above three key requirements. GRAPHITE takes advantage of transform-robustness, a metric based on expectation over transforms (EoT), to automatically generate small masks and optimize with gradient-free optimization. GRAPHITE is also flexible as it can easily trade-off transform-robustness, perturbation size, and query count in black-box settings. On a GTSRB model in a hard-label black-box setting, we are able to find attacks on all possible 1,806 victim-target class pairs with averages of 77.8% transform-robustness, perturbation size of 16.63% of the victim images, and 126K queries per pair. For digital-only attacks where achieving transform-robustness is not a requirement, GRAPHITE is able to find successful small-patch attacks with an average of only 566 queries for 92.2% of victim-target pairs. GRAPHITE is also able to find successful attacks using perturbations that modify small areas of the input image against PatchGuard, a recently proposed defense against patch-based attacks.

摘要: 本文调查了一个对手在为真实世界场景生成敌意示例时的攻击易用性。我们解决了现实世界中实际攻击的三个关键要求：1)自动约束攻击的大小和形状，使其可以与贴纸一起应用；2)变换鲁棒性，即攻击对环境物理变化(如视点和光照变化)的鲁棒性；3)不仅支持白盒攻击，而且支持黑盒硬标签场景下的攻击，以便攻击者可以攻击专有模型。在这项工作中，我们提出了石墨，这是一个高效和通用的框架，用于生成满足上述三个关键要求的攻击。石墨利用变换稳健性(一种基于期望重于变换(EoT)的度量)自动生成小掩码，并通过无梯度优化进行优化。石墨还很灵活，因为它可以很容易地权衡黑盒设置中的转换健壮性、扰动大小和查询计数。在硬标签黑盒环境下的GTSRB模型上，我们能够发现对所有可能的1806个受害者-目标类对的攻击，平均变换鲁棒性为77.8%，扰动大小为16.63%的受害者图像，每对126K查询。对于不要求实现变换鲁棒性的纯数字攻击，Graphic能够发现成功的小补丁攻击，92.2%的受害者-目标对平均只有566个查询。石墨还能够通过针对PatchGuard(最近提出的一种针对基于补丁的攻击的一种防御措施)修改输入图像的小区域来发现成功的攻击。



## **37. Markov Chain Monte Carlo-Based Machine Unlearning: Unlearning What Needs to be Forgotten**

基于马尔可夫链蒙特卡罗的机器遗忘：遗忘需要遗忘的东西 cs.LG

Proceedings of the 2022 ACM Asia Conference on Computer and  Communications Security (ASIA CCS '22), May 30-June 3, 2022, Nagasaki, Japan

**SubmitDate**: 2022-02-28    [paper-pdf](http://arxiv.org/pdf/2202.13585v1)

**Authors**: Quoc Phong Nguyen, Ryutaro Oikawa, Dinil Mon Divakaran, Mun Choon Chan, Bryan Kian Hsiang Low

**Abstracts**: As the use of machine learning (ML) models is becoming increasingly popular in many real-world applications, there are practical challenges that need to be addressed for model maintenance. One such challenge is to 'undo' the effect of a specific subset of dataset used for training a model. This specific subset may contain malicious or adversarial data injected by an attacker, which affects the model performance. Another reason may be the need for a service provider to remove data pertaining to a specific user to respect the user's privacy. In both cases, the problem is to 'unlearn' a specific subset of the training data from a trained model without incurring the costly procedure of retraining the whole model from scratch. Towards this goal, this paper presents a Markov chain Monte Carlo-based machine unlearning (MCU) algorithm. MCU helps to effectively and efficiently unlearn a trained model from subsets of training dataset. Furthermore, we show that with MCU, we are able to explain the effect of a subset of a training dataset on the model prediction. Thus, MCU is useful for examining subsets of data to identify the adversarial data to be removed. Similarly, MCU can be used to erase the lineage of a user's personal data from trained ML models, thus upholding a user's "right to be forgotten". We empirically evaluate the performance of our proposed MCU algorithm on real-world phishing and diabetes datasets. Results show that MCU can achieve a desirable performance by efficiently removing the effect of a subset of training dataset and outperform an existing algorithm that utilizes the remaining dataset.

摘要: 随着机器学习(ML)模型的使用在许多现实世界的应用程序中变得越来越流行，需要解决模型维护方面的实际挑战。一个这样的挑战是“取消”用于训练模型的数据集的特定子集的效果。此特定子集可能包含攻击者注入的恶意或敌意数据，这会影响模型性能。另一个原因可能是服务提供商需要移除属于特定用户的数据以尊重该用户的隐私。在这两种情况下，问题都是从训练的模型中“忘却”训练数据的特定子集，而不会招致从零开始重新训练整个模型的昂贵过程。针对这一目标，提出了一种基于马尔可夫链蒙特卡罗的机器遗忘(MCU)算法。MCU帮助有效且高效地从训练数据集子集中去除训练模型。此外，我们还表明，使用MCU，我们能够解释训练数据集的子集对模型预测的影响。因此，MCU对于检查数据子集以识别要移除的敌意数据是有用的。同样，MCU可以用来从经过训练的ML模型中删除用户个人数据的谱系，从而维护用户的“被遗忘权”。我们在真实的网络钓鱼和糖尿病数据集上对我们提出的MCU算法的性能进行了实证评估。结果表明，MCU通过有效地去除训练数据集子集的影响，取得了理想的性能，并优于现有的利用剩余数据集的算法。



## **38. A Unified Wasserstein Distributional Robustness Framework for Adversarial Training**

一种用于对抗性训练的统一Wasserstein分布健壮性框架 cs.LG

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2202.13437v1)

**Authors**: Tuan Anh Bui, Trung Le, Quan Tran, He Zhao, Dinh Phung

**Abstracts**: It is well-known that deep neural networks (DNNs) are susceptible to adversarial attacks, exposing a severe fragility of deep learning systems. As the result, adversarial training (AT) method, by incorporating adversarial examples during training, represents a natural and effective approach to strengthen the robustness of a DNN-based classifier. However, most AT-based methods, notably PGD-AT and TRADES, typically seek a pointwise adversary that generates the worst-case adversarial example by independently perturbing each data sample, as a way to "probe" the vulnerability of the classifier. Arguably, there are unexplored benefits in considering such adversarial effects from an entire distribution. To this end, this paper presents a unified framework that connects Wasserstein distributional robustness with current state-of-the-art AT methods. We introduce a new Wasserstein cost function and a new series of risk functions, with which we show that standard AT methods are special cases of their counterparts in our framework. This connection leads to an intuitive relaxation and generalization of existing AT methods and facilitates the development of a new family of distributional robustness AT-based algorithms. Extensive experiments show that our distributional robustness AT algorithms robustify further their standard AT counterparts in various settings.

摘要: 众所周知，深度神经网络(DNNs)容易受到敌意攻击，暴露出深度学习系统的严重脆弱性。因此，对抗性训练(AT)方法通过在训练过程中加入对抗性实例，是增强基于DNN的分类器鲁棒性的一种自然而有效的方法。然而，大多数基于AT的方法，特别是PGD-AT和TRADS，通常通过独立扰动每个数据样本来寻找一个点状对手，该对手通过独立扰动每个数据样本来生成最坏情况下的对手示例，作为“探测”分类器漏洞的一种方式。可以说，从整个发行版考虑这样的对抗性影响有未开发的好处。为此，本文提出了一个将Wasserstein分布稳健性与当前最先进的AT方法相结合的统一框架。我们引入了一个新的Wasserstein成本函数和一系列新的风险函数，证明了标准AT方法是我们框架中相应方法的特例。这种联系导致了现有AT方法的直观松弛和泛化，并促进了一类新的基于分布健壮性的AT算法的开发。大量的实验表明，我们的分布式健壮性AT算法在不同的设置下进一步增强了标准AT算法的健壮性。



## **39. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

Accepted at NeurIPS 2021. The missing square term in Eqn.(13), as  well as many other mistakes of the previous version, have been fixed in the  current version

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2111.07492v5)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法是免费的前期培训。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **40. CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks**

CC-Cert：一种验证神经网络一般鲁棒性的概率方法 cs.LG

In Proceedings of AAAI-22, the Thirty-Sixth AAAI Conference on  Artificial Intelligence

**SubmitDate**: 2022-02-27    [paper-pdf](http://arxiv.org/pdf/2109.10696v2)

**Authors**: Mikhail Pautov, Nurislam Tursynbek, Marina Munkhoeva, Nikita Muravev, Aleksandr Petiushko, Ivan Oseledets

**Abstracts**: In safety-critical machine learning applications, it is crucial to defend models against adversarial attacks -- small modifications of the input that change the predictions. Besides rigorously studied $\ell_p$-bounded additive perturbations, recently proposed semantic perturbations (e.g. rotation, translation) raise a serious concern on deploying ML systems in real-world. Therefore, it is important to provide provable guarantees for deep learning models against semantically meaningful input transformations. In this paper, we propose a new universal probabilistic certification approach based on Chernoff-Cramer bounds that can be used in general attack settings. We estimate the probability of a model to fail if the attack is sampled from a certain distribution. Our theoretical findings are supported by experimental results on different datasets.

摘要: 在安全关键型机器学习应用程序中，保护模型免受敌意攻击--对输入的微小修改会改变预测--是至关重要的。除了严格研究$\ellp$-有界的加性扰动之外，最近提出的语义扰动(如旋转、平移)也引起了人们对在现实世界中部署ML系统的严重关注。因此，针对语义上有意义的输入转换为深度学习模型提供可证明的保证是很重要的。本文提出了一种新的基于Chernoff-Cramer界的通用概率认证方法，该方法适用于一般攻击环境。我们估计了如果攻击是从特定分布中抽样的，模型失败的概率。在不同数据集上的实验结果支持了我们的理论发现。



## **41. Socialbots on Fire: Modeling Adversarial Behaviors of Socialbots via Multi-Agent Hierarchical Reinforcement Learning**

着火的社交机器人：基于多智能体分层强化学习的社交机器人对抗行为建模 cs.SI

Accepted to The ACM Web Conference 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2110.10655v2)

**Authors**: Thai Le, Long Tran-Thanh, Dongwon Lee

**Abstracts**: Socialbots are software-driven user accounts on social platforms, acting autonomously (mimicking human behavior), with the aims to influence the opinions of other users or spread targeted misinformation for particular goals. As socialbots undermine the ecosystem of social platforms, they are often considered harmful. As such, there have been several computational efforts to auto-detect the socialbots. However, to our best knowledge, the adversarial nature of these socialbots has not yet been studied. This begs a question "can adversaries, controlling socialbots, exploit AI techniques to their advantage?" To this question, we successfully demonstrate that indeed it is possible for adversaries to exploit computational learning mechanism such as reinforcement learning (RL) to maximize the influence of socialbots while avoiding being detected. We first formulate the adversarial socialbot learning as a cooperative game between two functional hierarchical RL agents. While one agent curates a sequence of activities that can avoid the detection, the other agent aims to maximize network influence by selectively connecting with right users. Our proposed policy networks train with a vast amount of synthetic graphs and generalize better than baselines on unseen real-life graphs both in terms of maximizing network influence (up to +18%) and sustainable stealthiness (up to +40% undetectability) under a strong bot detector (with 90% detection accuracy). During inference, the complexity of our approach scales linearly, independent of a network's structure and the virality of news. This makes our approach a practical adversarial attack when deployed in a real-life setting.

摘要: 社交机器人是社交平台上由软件驱动的用户账户，自主行动(模仿人类行为)，目的是影响其他用户的意见或为特定目标传播有针对性的错误信息。由于社交机器人破坏了社交平台的生态系统，它们通常被认为是有害的。因此，已经有几种计算努力来自动检测社交机器人。然而，据我们所知，这些社交机器人的对抗性还没有被研究过。这就引出了一个问题：“控制社交机器人的对手能不能利用人工智能技术对他们有利？”对于这个问题，我们成功地证明了攻击者确实有可能利用计算学习机制，如强化学习(RL)来最大化社交机器人的影响力，同时避免被发现。我们首先将对抗性的社会机器人学习描述为两个功能层次化的RL Agent之间的合作博弈。当一个代理策划一系列可以避免检测的活动时，另一个代理的目标是通过有选择地与正确的用户连接来最大化网络影响力。我们提出的策略网络使用大量的合成图形进行训练，并在强大的BOT检测器(具有90%的检测准确率)下，在最大化网络影响(高达+18%)和可持续隐蔽性(高达+40%不可检测性)方面，对不可见的真实图形进行了更好的概括。在推理过程中，我们方法的复杂性呈线性增长，与网络结构和新闻的病毒度无关。这使得我们的方法在实际环境中部署时成为一种实际的对抗性攻击。



## **42. Natural Attack for Pre-trained Models of Code**

针对预先训练的代码模型的自然攻击 cs.SE

To appear in the Technical Track of ICSE 2022

**SubmitDate**: 2022-02-26    [paper-pdf](http://arxiv.org/pdf/2201.08698v2)

**Authors**: Zhou Yang, Jieke Shi, Junda He, David Lo

**Abstracts**: Pre-trained models of code have achieved success in many important software engineering tasks. However, these powerful models are vulnerable to adversarial attacks that slightly perturb model inputs to make a victim model produce wrong outputs. Current works mainly attack models of code with examples that preserve operational program semantics but ignore a fundamental requirement for adversarial example generation: perturbations should be natural to human judges, which we refer to as naturalness requirement.   In this paper, we propose ALERT (nAturaLnEss AwaRe ATtack), a black-box attack that adversarially transforms inputs to make victim models produce wrong outputs. Different from prior works, this paper considers the natural semantic of generated examples at the same time as preserving the operational semantic of original inputs. Our user study demonstrates that human developers consistently consider that adversarial examples generated by ALERT are more natural than those generated by the state-of-the-art work by Zhang et al. that ignores the naturalness requirement. On attacking CodeBERT, our approach can achieve attack success rates of 53.62%, 27.79%, and 35.78% across three downstream tasks: vulnerability prediction, clone detection and code authorship attribution. On GraphCodeBERT, our approach can achieve average success rates of 76.95%, 7.96% and 61.47% on the three tasks. The above outperforms the baseline by 14.07% and 18.56% on the two pre-trained models on average. Finally, we investigated the value of the generated adversarial examples to harden victim models through an adversarial fine-tuning procedure and demonstrated the accuracy of CodeBERT and GraphCodeBERT against ALERT-generated adversarial examples increased by 87.59% and 92.32%, respectively.

摘要: 预先训练的代码模型在许多重要的软件工程任务中取得了成功。然而，这些强大的模型容易受到对抗性攻击，这些攻击略微扰动模型输入，使受害者模型产生错误的输出。目前的工作主要是用保持操作程序语义的示例攻击代码模型，而忽略了生成对抗性示例的一个基本要求：扰动对于人类判断来说应该是自然的，我们称之为自然性要求。在本文中，我们提出了ALERT(自然度感知攻击)，这是一种黑盒攻击，它对输入进行恶意转换，使受害者模型产生错误的输出。与以往的工作不同，本文在保留原始输入操作语义的同时，考虑了生成示例的自然语义。我们的用户研究表明，人类开发人员一致认为，ALERT生成的对抗性示例比由Zhang等人的最新工作生成的示例更自然。这忽略了自然度的要求。在攻击CodeBERT上，我们的方法可以在漏洞预测、克隆检测和代码作者归属三个下游任务上获得53.62%、27.79%和35.78%的攻击成功率。在GraphCodeBERT上，我们的方法在三个任务上的平均成功率分别为76.95%、7.96%和61.47%。在两个预先训练的模型上，上述两个模型的平均性能分别比基线高14.07%和18.56%。最后，我们通过对抗性微调过程考察了生成的对抗性实例对硬化受害者模型的价值，并证明了CodeBERT和GraphCodeBERT对警报生成的对抗性实例的准确率分别提高了87.59%和92.32%。结果表明，CodeBERT和GraphCodeBERT对警报生成的对抗性实例的准确率分别提高了87.59%和92.32%。



## **43. Projective Ranking-based GNN Evasion Attacks**

基于投影排名的GNN逃避攻击 cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12993v1)

**Authors**: He Zhang, Xingliang Yuan, Chuan Zhou, Shirui Pan

**Abstracts**: Graph neural networks (GNNs) offer promising learning methods for graph-related tasks. However, GNNs are at risk of adversarial attacks. Two primary limitations of the current evasion attack methods are highlighted: (1) The current GradArgmax ignores the "long-term" benefit of the perturbation. It is faced with zero-gradient and invalid benefit estimates in certain situations. (2) In the reinforcement learning-based attack methods, the learned attack strategies might not be transferable when the attack budget changes. To this end, we first formulate the perturbation space and propose an evaluation framework and the projective ranking method. We aim to learn a powerful attack strategy then adapt it as little as possible to generate adversarial samples under dynamic budget settings. In our method, based on mutual information, we rank and assess the attack benefits of each perturbation for an effective attack strategy. By projecting the strategy, our method dramatically minimizes the cost of learning a new attack strategy when the attack budget changes. In the comparative assessment with GradArgmax and RL-S2V, the results show our method owns high attack performance and effective transferability. The visualization of our method also reveals various attack patterns in the generation of adversarial samples.

摘要: 图神经网络(GNNs)为与图相关的任务提供了很有前途的学习方法。然而，GNN面临着遭到敌意攻击的风险。强调了当前规避攻击方法的两个主要局限性：(1)当前的GradArgmax忽略了扰动的“长期”益处。在某些情况下，它面临着零梯度和无效的效益估计。(2)在基于强化学习的攻击方法中，当攻击预算发生变化时，学习到的攻击策略可能无法迁移。为此，我们首先定义了扰动空间，并提出了评价框架和投影排序方法。我们的目标是学习一种强大的攻击策略，然后尽可能少地对其进行调整，以便在动态预算设置下生成对抗性样本。在我们的方法中，我们基于互信息，对每个扰动的攻击收益进行排序和评估，以确定有效的攻击策略。通过投射策略，当攻击预算发生变化时，我们的方法极大地降低了学习新攻击策略的成本。在与GradArgmax和RL-S2V的对比评估中，结果表明该方法具有较高的攻击性能和有效的可移植性。该方法的可视化还揭示了敌意样本生成过程中的各种攻击模式。



## **44. Attacks and Faults Injection in Self-Driving Agents on the Carla Simulator -- Experience Report**

自动驾驶智能体在CALA模拟器上的攻击和故障注入--经验报告 cs.AI

submitted version; appeared at: International Conference on Computer  Safety, Reliability, and Security. Springer, Cham, 2021

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12991v1)

**Authors**: Niccolò Piazzesi, Massimo Hong, Andrea Ceccarelli

**Abstracts**: Machine Learning applications are acknowledged at the foundation of autonomous driving, because they are the enabling technology for most driving tasks. However, the inclusion of trained agents in automotive systems exposes the vehicle to novel attacks and faults, that can result in safety threats to the driv-ing tasks. In this paper we report our experimental campaign on the injection of adversarial attacks and software faults in a self-driving agent running in a driving simulator. We show that adversarial attacks and faults injected in the trained agent can lead to erroneous decisions and severely jeopardize safety. The paper shows a feasible and easily-reproducible approach based on open source simula-tor and tools, and the results clearly motivate the need of both protective measures and extensive testing campaigns.

摘要: 机器学习应用在自动驾驶的基础上得到认可，因为它们是大多数驾驶任务的使能技术。然而，在汽车系统中加入训练有素的代理会使车辆暴露在新的攻击和故障下，这可能会对驾驶任务造成安全威胁。在本文中，我们报告了我们在驾驶模拟器中运行的自动驾驶代理中注入对抗性攻击和软件故障的实验活动。我们表明，对抗性攻击和错误注入训练有素的代理可能导致错误的决策，并严重危及安全。本文展示了一种基于开源模拟器和工具的可行且易于重现的方法，其结果清楚地激励了保护措施和广泛的测试活动的需要。



## **45. Does Label Differential Privacy Prevent Label Inference Attacks?**

标签差分隐私能防止标签推理攻击吗？ cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12968v1)

**Authors**: Ruihan Wu, Jin Peng Zhou, Kilian Q. Weinberger, Chuan Guo

**Abstracts**: Label differential privacy (LDP) is a popular framework for training private ML models on datasets with public features and sensitive private labels. Despite its rigorous privacy guarantee, it has been observed that in practice LDP does not preclude label inference attacks (LIAs): Models trained with LDP can be evaluated on the public training features to recover, with high accuracy, the very private labels that it was designed to protect. In this work, we argue that this phenomenon is not paradoxical and that LDP merely limits the advantage of an LIA adversary compared to predicting training labels using the Bayes classifier. At LDP $\epsilon=0$ this advantage is zero, hence the optimal attack is to predict according to the Bayes classifier and is independent of the training labels. Finally, we empirically demonstrate that our result closely captures the behavior of simulated attacks on both synthetic and real world datasets.

摘要: 标签差异隐私(Label Differential Privacy，LDP)是一种流行的框架，用于在具有公共特征和敏感私有标签的数据集上训练私有ML模型。尽管LDP提供了严格的隐私保障，但已经观察到，在实践中，LDP并不排除标签推断攻击(LIA)：使用LDP训练的模型可以在公共训练特征上进行评估，以高精度地恢复其设计来保护的非常私有的标签。在这项工作中，我们认为这种现象并不矛盾，与使用贝叶斯分类器预测训练标签相比，LDP只是限制了LIA对手的优势。当LDP$\εsilon=0$时，这一优势为零，因此最优攻击是根据贝叶斯分类器进行预测，并且与训练标签无关。最后，我们通过实验证明，我们的结果很好地捕捉了模拟攻击在合成数据集和真实世界数据集上的行为。



## **46. Robust and Accurate Authorship Attribution via Program Normalization**

基于程序归一化的稳健准确的作者归属 cs.LG

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.00772v3)

**Authors**: Yizhen Wang, Mohannad Alhanahnah, Ke Wang, Mihai Christodorescu, Somesh Jha

**Abstracts**: Source code attribution approaches have achieved remarkable accuracy thanks to the rapid advances in deep learning. However, recent studies shed light on their vulnerability to adversarial attacks. In particular, they can be easily deceived by adversaries who attempt to either create a forgery of another author or to mask the original author. To address these emerging issues, we formulate this security challenge into a general threat model, the $\textit{relational adversary}$, that allows an arbitrary number of the semantics-preserving transformations to be applied to an input in any problem space. Our theoretical investigation shows the conditions for robustness and the trade-off between robustness and accuracy in depth. Motivated by these insights, we present a novel learning framework, $\textit{normalize-and-predict}$ ($\textit{N&P}$), that in theory guarantees the robustness of any authorship-attribution approach. We conduct an extensive evaluation of $\textit{N&P}$ in defending two of the latest authorship-attribution approaches against state-of-the-art attack methods. Our evaluation demonstrates that $\textit{N&P}$ improves the accuracy on adversarial inputs by as much as 70% over the vanilla models. More importantly, $\textit{N&P}$ also increases robust accuracy to 45% higher than adversarial training while running over 40 times faster.

摘要: 由于深度学习的快速发展，源代码归属方法已经取得了显着的准确性。然而，最近的研究揭示了它们在对抗性攻击中的脆弱性。特别是，他们很容易被试图伪造另一位作者或掩盖原作者的对手欺骗。为了解决这些新出现的问题，我们将此安全挑战表述为一个通用威胁模型，即$\textit{关系对手}$，该模型允许将任意数量的语义保留转换应用于任何问题空间中的输入。我们的理论研究深入地给出了稳健性的条件以及稳健性和准确性之间的权衡。在这些观点的启发下，我们提出了一个新的学习框架$\textit{Normize-and-Predicate}$($\textit{N&P}$)，该框架在理论上保证了任何作者归因方法的健壮性。我们对$\textit{N&P}$在防御两种最新的作者归属方法方面进行了广泛的评估，以抵御最先进的攻击方法。我们的评估表明，与普通模型相比，$\textit{N&P}$将对手输入的准确率提高了70%。更重要的是，$\textit{N&P}$还将健壮的准确率提高到比对抗性训练高出45%，同时运行速度快40倍以上。



## **47. ARIA: Adversarially Robust Image Attribution for Content Provenance**

ARIA：内容来源的逆向稳健图像归因 cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12860v1)

**Authors**: Maksym Andriushchenko, Xiaoyang Rebecca Li, Geoffrey Oxholm, Thomas Gittings, Tu Bui, Nicolas Flammarion, John Collomosse

**Abstracts**: Image attribution -- matching an image back to a trusted source -- is an emerging tool in the fight against online misinformation. Deep visual fingerprinting models have recently been explored for this purpose. However, they are not robust to tiny input perturbations known as adversarial examples. First we illustrate how to generate valid adversarial images that can easily cause incorrect image attribution. Then we describe an approach to prevent imperceptible adversarial attacks on deep visual fingerprinting models, via robust contrastive learning. The proposed training procedure leverages training on $\ell_\infty$-bounded adversarial examples, it is conceptually simple and incurs only a small computational overhead. The resulting models are substantially more robust, are accurate even on unperturbed images, and perform well even over a database with millions of images. In particular, we achieve 91.6% standard and 85.1% adversarial recall under $\ell_\infty$-bounded perturbations on manipulated images compared to 80.1% and 0.0% from prior work. We also show that robustness generalizes to other types of imperceptible perturbations unseen during training. Finally, we show how to train an adversarially robust image comparator model for detecting editorial changes in matched images.

摘要: 图像归属--将图像与可信来源相匹配--是打击在线错误信息的一种新兴工具。最近已经为此目的探索了深度视觉指纹模型。然而，它们对被称为对抗性示例的微小输入扰动并不健壮。首先，我们说明了如何生成有效的敌意图像，这些图像很容易导致错误的图像属性。然后，我们描述了一种通过稳健的对比学习来防止对深度视觉指纹模型的不可察觉的敌意攻击的方法。所提出的训练过程利用了对$\ELL_\INFTY$-有界的对抗性示例的训练，概念上很简单，并且只产生很小的计算开销。由此产生的模型更加健壮，即使在不受干扰的图像上也是准确的，即使在拥有数百万图像的数据库上也表现得很好。特别地，在$\ell_\ininfty$-有界扰动下，我们获得了91.6%的标准召回率和85.1%的敌意召回率，而以前的工作分别为80.1%和0.0%。我们还表明，鲁棒性推广到其他类型的在训练过程中看不到的不可察觉的扰动。最后，我们展示了如何训练一个对抗性的鲁棒图像比较器模型来检测匹配图像中的编辑变化。



## **48. Short Paper: Device- and Locality-Specific Fingerprinting of Shared NISQ Quantum Computers**

短文：共享NISQ量子计算机的特定于设备和位置的指纹识别 cs.CR

5 pages, 8 figures, HASP 2021 author version

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12731v1)

**Authors**: Allen Mi, Shuwen Deng, Jakub Szefer

**Abstracts**: Fingerprinting of quantum computer devices is a new threat that poses a challenge to shared, cloud-based quantum computers. Fingerprinting can allow adversaries to map quantum computer infrastructures, uniquely identify cloud-based devices which otherwise have no public identifiers, and it can assist other adversarial attacks. This work shows idle tomography-based fingerprinting method based on crosstalk-induced errors in NISQ quantum computers. The device- and locality-specific fingerprinting results show prediction accuracy values of $99.1\%$ and $95.3\%$, respectively.

摘要: 量子计算机设备的指纹识别是一个新的威胁，对共享的、基于云的量子计算机构成了挑战。指纹识别可以让攻击者映射量子计算机基础设施，唯一识别没有公共标识符的基于云的设备，还可以协助其他敌意攻击。这项工作展示了NISQ量子计算机中基于串扰引起的错误的基于空闲层析成像的指纹识别方法。设备特定指纹和位置特定指纹的预测准确值分别为99.1美元和95.3美元。



## **49. Detection as Regression: Certified Object Detection by Median Smoothing**

回归检测：基于中值平滑的认证目标检测 cs.CV

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2007.03730v4)

**Authors**: Ping-yeh Chiang, Michael J. Curry, Ahmed Abdelkader, Aounon Kumar, John Dickerson, Tom Goldstein

**Abstracts**: Despite the vulnerability of object detectors to adversarial attacks, very few defenses are known to date. While adversarial training can improve the empirical robustness of image classifiers, a direct extension to object detection is very expensive. This work is motivated by recent progress on certified classification by randomized smoothing. We start by presenting a reduction from object detection to a regression problem. Then, to enable certified regression, where standard mean smoothing fails, we propose median smoothing, which is of independent interest. We obtain the first model-agnostic, training-free, and certified defense for object detection against $\ell_2$-bounded attacks. The code for all experiments in the paper is available at http://github.com/Ping-C/CertifiedObjectDetection .

摘要: 尽管物体探测器对敌方攻击很脆弱，但到目前为止，人们所知的防御措施很少。虽然对抗性训练可以提高图像分类器的经验鲁棒性，但直接扩展到目标检测是非常昂贵的。这项工作是由随机平滑认证分类的最新进展推动的。我们首先介绍从目标检测到回归问题的简化。然后，为了实现认证回归，在标准均值平滑失败的情况下，我们提出了中值平滑，这是独立感兴趣的。我们获得了第一个模型不可知的、无需训练的、经过认证的针对$\ELL_2$有界攻击的目标检测防御。论文中所有实验的代码都可以在http://github.com/Ping-C/CertifiedObjectDetection上找到。



## **50. On the Effectiveness of Dataset Watermarking in Adversarial Settings**

数据集水印在对抗性环境中的有效性研究 cs.CR

7 pages, 2 figures. Will appear in the proceedings of CODASPY-IWSPA  2022

**SubmitDate**: 2022-02-25    [paper-pdf](http://arxiv.org/pdf/2202.12506v1)

**Authors**: Buse Gul Atli Tekgul, N. Asokan

**Abstracts**: In a data-driven world, datasets constitute a significant economic value. Dataset owners who spend time and money to collect and curate the data are incentivized to ensure that their datasets are not used in ways that they did not authorize. When such misuse occurs, dataset owners need technical mechanisms for demonstrating their ownership of the dataset in question. Dataset watermarking provides one approach for ownership demonstration which can, in turn, deter unauthorized use. In this paper, we investigate a recently proposed data provenance method, radioactive data, to assess if it can be used to demonstrate ownership of (image) datasets used to train machine learning (ML) models. The original paper reported that radioactive data is effective in white-box settings. We show that while this is true for large datasets with many classes, it is not as effective for datasets where the number of classes is low $(\leq 30)$ or the number of samples per class is low $(\leq 500)$. We also show that, counter-intuitively, the black-box verification technique is effective for all datasets used in this paper, even when white-box verification is not. Given this observation, we show that the confidence in white-box verification can be improved by using watermarked samples directly during the verification process. We also highlight the need to assess the robustness of radioactive data if it were to be used for ownership demonstration since it is an adversarial setting unlike provenance identification.   Compared to dataset watermarking, ML model watermarking has been explored more extensively in recent literature. However, most of the model watermarking techniques can be defeated via model extraction. We show that radioactive data can effectively survive model extraction attacks, which raises the possibility that it can be used for ML model ownership verification robust against model extraction.

摘要: 在一个数据驱动的世界里，数据集构成了重要的经济价值。花费时间和金钱收集和管理数据的数据集所有者受到激励，以确保他们的数据集不会以未经授权的方式使用。当这种误用发生时，数据集所有者需要技术机制来证明他们对相关数据集的所有权。数据集水印为所有权证明提供了一种方法，进而可以阻止未经授权的使用。在这篇文章中，我们调查了最近提出的一种数据来源方法，放射性数据，以评估它是否可以用来证明用于训练机器学习(ML)模型的(图像)数据集的所有权。最初的论文报道说，放射性数据在白盒设置中是有效的。我们表明，虽然这对于具有多个类的大型数据集是正确的，但是对于类数量低$(\leq 30)$或每个类的样本数低$(\leq 500)$的数据集就不那么有效了。我们还表明，与直觉相反，黑盒验证技术对本文使用的所有数据集都是有效的，即使白盒验证不是有效的。基于这一观察结果，我们证明了在验证过程中直接使用带水印的样本可以提高白盒验证的置信度。我们还强调，如果放射性数据要用于所有权证明，则需要评估其稳健性，因为这是一种与来源鉴定不同的对抗性环境。与数据集水印相比，ML模型水印在最近的文献中得到了更广泛的研究。然而，大多数模型水印技术都可以通过模型提取来破解。结果表明，放射性数据能够有效抵御模型提取攻击，提高了其用于ML模型所有权验证对模型提取的鲁棒性的可能性。



