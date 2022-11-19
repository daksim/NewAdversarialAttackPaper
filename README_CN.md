# Latest Adversarial Attack Papers
**update at 2022-11-19 15:42:13**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adaptive Test-Time Defense with the Manifold Hypothesis**

流形假设下的自适应测试时间防御 cs.LG

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2210.14404v3) [paper-pdf](http://arxiv.org/pdf/2210.14404v3)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with variational inference and our formulation. The developed approach combines manifold learning with variational inference to provide adversarial robustness without the need for adversarial training. We show that our approach can provide adversarial robustness even if attackers are aware of the existence of test-time defense. In addition, our approach can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。我们的框架为防御对抗性例子提供了充分的条件。我们提出了一种带有变分推理的测试时间防御方法和我们的公式。该方法将流形学习和变分推理相结合，在不需要对抗性训练的情况下提供对抗性稳健性。我们表明，即使攻击者知道测试时间防御的存在，我们的方法也可以提供对抗健壮性。此外，我们的方法还可以作为可变自动编码器的测试时间防御机制。



## **2. UPTON: Unattributable Authorship Text via Data Poisoning**

Upton：数据中毒导致的不明作者身份文本 cs.CY

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09717v1) [paper-pdf](http://arxiv.org/pdf/2211.09717v1)

**Authors**: Ziyao Wang, Thai Le, Dongwon Lee

**Abstract**: In online medium such as opinion column in Bloomberg, The Guardian and Western Journal, aspiring writers post their writings for various reasons with their names often proudly open. However, it may occur that such a writer wants to write in other venues anonymously or under a pseudonym (e.g., activist, whistle-blower). However, if an attacker has already built an accurate authorship attribution (AA) model based off of the writings from such platforms, attributing an anonymous writing to the known authorship is possible. Therefore, in this work, we ask a question "can one make the writings and texts, T, in the open spaces such as opinion sharing platforms unattributable so that AA models trained from T cannot attribute authorship well?" Toward this question, we present a novel solution, UPTON, that exploits textual data poisoning method to disturb the training process of AA models. UPTON uses data poisoning to destroy the authorship feature only in training samples by perturbing them, and try to make released textual data unlearnable on deep neuron networks. It is different from previous obfuscation works, that use adversarial attack to modify the test samples and mislead an AA model, and also the backdoor works, which use trigger words both in test and training samples and only change the model output when trigger words occur. Using four authorship datasets (e.g., IMDb10, IMDb64, Enron and WJO), then, we present empirical validation where: (1)UPTON is able to downgrade the test accuracy to about 30% with carefully designed target-selection methods. (2)UPTON poisoning is able to preserve most of the original semantics. The BERTSCORE between the clean and UPTON poisoned texts are higher than 0.95. The number is very closed to 1.00, which means no sematic change. (3)UPTON is also robust towards spelling correction systems.

摘要: 在彭博社、《卫报》、《西部日报》的观点专栏等网络媒体上，有抱负的作家出于各种原因发布自己的作品，经常自豪地打开自己的名字。然而，可能会发生这样的作者想要在其他场所匿名或以化名(例如，活动家、告密者)写作的情况。然而，如果攻击者已经基于来自这些平台的作品构建了准确的作者归属(AA)模型，则可以将匿名作品归因于已知的作者。因此，在这项工作中，我们提出了一个问题：是否可以让意见分享平台等开放空间中的文字和文本T无法归因于T，从而使从T训练的AA模型无法很好地归属作者？针对这一问题，我们提出了一种新的解决方案Upton，它利用文本数据毒化方法来干扰AA模型的训练过程。厄普顿使用数据中毒来破坏只有在训练样本中才有的作者特征，并试图使已发布的文本数据在深层神经元网络上无法学习。不同于以往的混淆工作，它使用对抗性攻击来修改测试样本，误导AA模型；后门工作，在测试和训练样本中都使用触发词，只有在触发词出现时才改变模型输出。然后，使用四个作者的数据集(例如，IMDb10，IMDb64，Enron和WJO)，我们提供了经验验证，其中：(1)Upton能够通过精心设计的目标选择方法将测试准确率降低到约30%。(2)Upton中毒能够保留大部分原始语义。CLEAN文本和Upton中毒文本之间的BERTSCORE均大于0.95。这个数字非常接近1.00，这意味着没有语义变化。(3)厄普顿对拼写纠正系统也很感兴趣。



## **3. An efficient combination of quantum error correction and authentication**

一种量子纠错和认证的有效组合 quant-ph

30 pages, 10 figures

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09686v1) [paper-pdf](http://arxiv.org/pdf/2211.09686v1)

**Authors**: Yfke Dulek, Garazi Muguruza, Florian Speelman

**Abstract**: When sending quantum information over a channel, we want to ensure that the message remains intact. Quantum error correction and quantum authentication both aim to protect (quantum) information, but approach this task from two very different directions: error-correcting codes protect against probabilistic channel noise and are meant to be very robust against small errors, while authentication codes prevent adversarial attacks and are designed to be very sensitive against any error, including small ones.   In practice, when sending an authenticated state over a noisy channel, one would have to wrap it in an error-correcting code to counterbalance the sensitivity of the underlying authentication scheme. We study the question of whether this can be done more efficiently by combining the two functionalities in a single code. To illustrate the potential of such a combination, we design the threshold code, a modification of the trap authentication code which preserves that code's authentication properties, but which is naturally robust against depolarizing channel noise. We show that the threshold code needs polylogarithmically fewer qubits to achieve the same level of security and robustness, compared to the naive composition of the trap code with any concatenated CSS code. We believe our analysis opens the door to combining more general error-correction and authentication codes, which could improve the practicality of the resulting scheme.

摘要: 在通过通道发送量子信息时，我们希望确保消息保持不变。量子纠错和量子认证都旨在保护(量子)信息，但从两个非常不同的方向来处理这项任务：纠错码保护不受概率信道噪声的影响，并且对小错误非常健壮，而认证码防止对手攻击，并且对任何错误(包括小错误)非常敏感。在实践中，当通过噪声信道发送认证状态时，必须将其包装在纠错码中以平衡基础认证方案的敏感性。我们研究的问题是，是否可以通过在单个代码中组合这两个功能来更有效地完成这项工作。为了说明这种组合的可能性，我们设计了门限码，这是对陷阱认证码的修改，它保留了该码的认证属性，但对去极化信道噪声具有天然的健壮性。我们证明，与任何级联的CSS码相比，门限码需要更少的多对数量子比特来实现相同级别的安全性和稳健性。我们相信，我们的分析为结合更一般的纠错和认证码打开了大门，这可能会提高所得到的方案的实用性。



## **4. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

在评估转会对抗性攻击方面的良好做法 cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09565v1) [paper-pdf](http://arxiv.org/pdf/2211.09565v1)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of attack methods is difficult to assess due to two main limitations in existing evaluations. First, existing evaluations are unsystematic and sometimes unfair since new methods are often directly added to old ones without complete comparisons to similar methods. Second, existing evaluations mainly focus on transferability but overlook another key attack property: stealthiness. In this work, we design good practices to address these limitations. We first introduce a new attack categorization, which enables our systematic analyses of similar attacks in each specific category. Our analyses lead to new findings that complement or even challenge existing knowledge. Furthermore, we comprehensively evaluate 23 representative attacks against 9 defenses on ImageNet. We pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Our evaluation reveals new important insights: 1) Transferability is highly contextual, and some white-box defenses may give a false sense of security since they are actually vulnerable to (black-box) transfer attacks; 2) All transfer attacks are less stealthy, and their stealthiness can vary dramatically under the same $L_{\infty}$ bound.

摘要: 在现实世界的黑盒场景中，传输敌意攻击会引发严重的安全问题。然而，由于现有评估中的两个主要限制，攻击方法的实际进展很难评估。首先，现有的评价是不系统的，有时是不公平的，因为新的方法往往直接添加到旧的方法中，而没有与类似的方法进行完全的比较。其次，现有的评估主要集中在可转移性上，而忽略了另一个关键的攻击属性：隐蔽性。在这项工作中，我们设计了良好的实践来解决这些限制。我们首先介绍了一种新的攻击分类，它使我们能够对每个特定类别中的类似攻击进行系统分析。我们的分析导致了补充甚至挑战现有知识的新发现。此外，我们还综合评估了ImageNet上针对9个防御的23种代表性攻击。我们特别关注隐蔽性，采用了不同的隐蔽性度量标准，并研究了新的、更细粒度的特征。我们的评估揭示了新的重要见解：1)可转移性与上下文高度相关，一些白盒防御可能会给人一种错误的安全感，因为它们实际上容易受到(黑盒)传输攻击；2)所有传输攻击的隐蔽性都较低，并且它们的隐蔽性在相同的$L_(\infty)$界限下可能会有很大的变化。



## **5. Ignore Previous Prompt: Attack Techniques For Language Models**

忽略前面的提示：语言模型的攻击技巧 cs.CL

ML Safety Workshop NeurIPS 2022

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09527v1) [paper-pdf](http://arxiv.org/pdf/2211.09527v1)

**Authors**: Fábio Perez, Ian Ribeiro

**Abstract**: Transformer-based large language models (LLMs) provide a powerful foundation for natural language tasks in large-scale customer-facing applications. However, studies that explore their vulnerabilities emerging from malicious user interaction are scarce. By proposing PromptInject, a prosaic alignment framework for mask-based iterative adversarial prompt composition, we examine how GPT-3, the most widely deployed language model in production, can be easily misaligned by simple handcrafted inputs. In particular, we investigate two types of attacks -- goal hijacking and prompt leaking -- and demonstrate that even low-aptitude, but sufficiently ill-intentioned agents, can easily exploit GPT-3's stochastic nature, creating long-tail risks. The code for PromptInject is available at https://github.com/agencyenterprise/PromptInject.

摘要: 基于转换器的大型语言模型(LLM)为大规模面向客户应用中的自然语言任务提供了强大的基础。然而，探索恶意用户交互中出现的漏洞的研究很少。通过提出PromptInject，一个基于掩码的迭代对抗性提示合成的平淡无奇的对齐框架，我们研究了GPT-3，生产中应用最广泛的语言模型，如何通过简单的手工制作的输入很容易地错位。特别是，我们调查了两种类型的攻击--目标劫持和快速泄漏--并证明即使是低能力但足够恶意的代理也可以很容易地利用GPT-3的随机性质，创造长尾风险。PromptInject的代码可在https://github.com/agencyenterprise/PromptInject.上获得



## **6. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2207.13381v3) [paper-pdf](http://arxiv.org/pdf/2207.13381v3)

**Authors**: Mingjie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstract**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.

摘要: 本文旨在通过读取敌人的心理(Vm)来生成真实的人重新识别的攻击样本，Reid。本文提出了一种新的隐蔽可控的Reid攻击基线--LCYE，用于生成敌意查询图像。具体来说，LCYE首先通过模仿代理任务中的师生记忆来提取VM的知识。然后，这种先验知识就像一个明确的密码，传达了被VM认为是必要和现实的东西，以实现准确的对抗性误导。此外，得益于LCYE的多重对立任务框架，我们从对抗性攻击的角度进一步考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。我们的代码现已在https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.上提供



## **7. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

幻影海绵：利用非最大抑制攻击深度对象探测器 cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2205.13618v3) [paper-pdf](http://arxiv.org/pdf/2205.13618v3)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects which allows the attack to go undetected for a longer period of time.

摘要: 针对基于深度学习的目标检测器的对抗性攻击在过去的几年中得到了广泛的研究。大多数提出的攻击都是针对模型的完整性(即导致模型做出错误的预测)，而针对模型可用性的对抗性攻击(自动驾驶等安全关键领域的一个关键方面)尚未被机器学习研究社区探索。在本文中，我们提出了一种新的攻击，它对端到端对象检测流水线的决策延迟产生负面影响。我们设计了一种通用对抗摄动(UAP)，目标是集成在许多对象探测器流水线中的一种广泛使用的技术--非最大抑制(NMS)。我们的实验证明了所提出的UAP能够通过添加重载NMS算法的“幻影”对象来增加单个帧的处理时间，同时保持对原始对象的检测，从而允许攻击在更长的时间内被检测到。



## **8. Reasons for the Superiority of Stochastic Estimators over Deterministic Ones: Robustness, Consistency and Perceptual Quality**

随机估计量优于确定性估计量的原因：稳健性、一致性和知觉质量 eess.IV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.08944v2) [paper-pdf](http://arxiv.org/pdf/2211.08944v2)

**Authors**: Guy Ohayon, Theo Adrai, Michael Elad, Tomer Michaeli

**Abstract**: Stochastic restoration algorithms allow to explore the space of solutions that correspond to the degraded input. In this paper we reveal additional fundamental advantages of stochastic methods over deterministic ones, which further motivate their use. First, we prove that any restoration algorithm that attains perfect perceptual quality and whose outputs are consistent with the input must be a posterior sampler, and is thus required to be stochastic. Second, we illustrate that while deterministic restoration algorithms may attain high perceptual quality, this can be achieved only by filling up the space of all possible source images using an extremely sensitive mapping, which makes them highly vulnerable to adversarial attacks. Indeed, we show that enforcing deterministic models to be robust to such attacks profoundly hinders their perceptual quality, while robustifying stochastic models hardly influences their perceptual quality, and improves their output variability. These findings provide a motivation to foster progress in stochastic restoration methods, paving the way to better recovery algorithms.

摘要: 随机恢复算法允许探索对应于退化输入的解的空间。在这篇文章中，我们揭示了随机方法相对于确定性方法的其他基本优势，这进一步促进了它们的使用。首先，我们证明了任何获得完美感知质量并且其输出与输入一致的恢复算法一定是后验采样器，因此要求它是随机的。其次，我们说明虽然确定性恢复算法可以获得高感知质量，但这只能通过使用极其敏感的映射来填充所有可能的源图像的空间来实现，这使得它们非常容易受到对手的攻击。的确，我们表明，强制确定性模型对此类攻击具有健壮性会严重阻碍它们的感知质量，而健壮化的随机模型几乎不会影响它们的感知质量，并改善它们的输出变异性。这些发现为促进随机恢复方法的进步提供了动力，为更好的恢复算法铺平了道路。



## **9. Generalizable Deepfake Detection with Phase-Based Motion Analysis**

基于相位运动分析的泛化深伪检测 cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09363v1) [paper-pdf](http://arxiv.org/pdf/2211.09363v1)

**Authors**: Ekta Prashnani, Michael Goebel, B. S. Manjunath

**Abstract**: We propose PhaseForensics, a DeepFake (DF) video detection method that leverages a phase-based motion representation of facial temporal dynamics. Existing methods relying on temporal inconsistencies for DF detection present many advantages over the typical frame-based methods. However, they still show limited cross-dataset generalization and robustness to common distortions. These shortcomings are partially due to error-prone motion estimation and landmark tracking, or the susceptibility of the pixel intensity-based features to spatial distortions and the cross-dataset domain shifts. Our key insight to overcome these issues is to leverage the temporal phase variations in the band-pass components of the Complex Steerable Pyramid on face sub-regions. This not only enables a robust estimate of the temporal dynamics in these regions, but is also less prone to cross-dataset variations. Furthermore, the band-pass filters used to compute the local per-frame phase form an effective defense against the perturbations commonly seen in gradient-based adversarial attacks. Overall, with PhaseForensics, we show improved distortion and adversarial robustness, and state-of-the-art cross-dataset generalization, with 91.2% video-level AUC on the challenging CelebDFv2 (a recent state-of-the-art compares at 86.9%).

摘要: 我们提出了一种DeepFake(DF)视频检测方法PhaseForensics，该方法利用了人脸时间动力学的基于相位的运动表示。与典型的基于帧的方法相比，现有的依赖于时间不一致性的DF检测方法显示出许多优点。然而，它们仍然表现出有限的跨数据集泛化和对常见失真的稳健性。这些缺点部分是由于容易出错的运动估计和标志点跟踪，或者是基于像素强度的特征对空间失真和跨数据集域漂移的敏感性。我们克服这些问题的关键洞察力是利用复杂的可控金字塔面子区域的带通分量中的时间相位变化。这不仅能够对这些区域的时间动态进行稳健的估计，而且不太容易发生跨数据集变化。此外，用于计算局部每帧相位的带通滤波器形成了对基于梯度的对抗性攻击中常见的扰动的有效防御。总体而言，使用PhaseForensics，我们显示出更好的失真和对抗健壮性，以及最先进的跨数据集泛化，具有挑战性的CelebDFv2的视频级AUC为91.2%(最近的最新AUC为86.9%)。



## **10. Targeted Attention for Generalized- and Zero-Shot Learning**

针对泛化学习和零射击学习的定向注意 cs.CV

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2211.09322v1) [paper-pdf](http://arxiv.org/pdf/2211.09322v1)

**Authors**: Abhijit Suprem

**Abstract**: The Zero-Shot Learning (ZSL) task attempts to learn concepts without any labeled data. Unlike traditional classification/detection tasks, the evaluation environment is provided unseen classes never encountered during training. As such, it remains both challenging, and promising on a variety of fronts, including unsupervised concept learning, domain adaptation, and dataset drift detection. Recently, there have been a variety of approaches towards solving ZSL, including improved metric learning methods, transfer learning, combinations of semantic and image domains using, e.g. word vectors, and generative models to model the latent space of known classes to classify unseen classes. We find many approaches require intensive training augmentation with attributes or features that may be commonly unavailable (attribute-based learning) or susceptible to adversarial attacks (generative learning). We propose combining approaches from the related person re-identification task for ZSL, with key modifications to ensure sufficiently improved performance in the ZSL setting without the need for feature or training dataset augmentation. We are able to achieve state-of-the-art performance on the CUB200 and Cars196 datasets in the ZSL setting compared to recent works, with NMI (normalized mutual inference) of 63.27 and top-1 of 61.04 for CUB200, and NMI 66.03 with top-1 82.75% in Cars196. We also show state-of-the-art results in the Generalized Zero-Shot Learning (GZSL) setting, with Harmonic Mean R-1 of 66.14% on the CUB200 dataset.

摘要: 零镜头学习(ZSL)任务试图在没有任何标签数据的情况下学习概念。与传统的分类/检测任务不同，评估环境提供了在培训期间从未遇到过的看不见的类。因此，它在许多方面仍然具有挑战性和前景，包括无监督概念学习、域自适应和数据集漂移检测。最近，已经有各种方法来解决ZSL，包括改进的度量学习方法、迁移学习、使用例如单词向量的语义域和图像域的组合，以及生成模型来对已知类的潜在空间进行建模以对未见类进行分类。我们发现，许多方法需要强化训练，增加通常无法获得的属性或特征(基于属性的学习)或容易受到对手攻击的属性或特征(生成性学习)。我们建议将ZSL的相关人员重新识别任务的方法与关键修改相结合，以确保在ZSL环境中充分提高性能，而不需要增加特征或训练数据集。与最近的工作相比，我们能够在ZSL环境下在CUB200和Cars196数据集上获得最先进的性能，CUB200的NMI(归一化相互推理)为63.27，TOP-1为61.04，Cars196的NMI为66.03，TOP-182.75%。我们还展示了在广义零点学习(GZSL)设置下的最新结果，在CUB200数据集上的调和平均R-1为66.14%。



## **11. Fair Robust Active Learning by Joint Inconsistency**

基于联合不一致性的公平鲁棒主动学习 cs.LG

11 pages, 2 figures, 8 tables

**SubmitDate**: 2022-11-17    [abs](http://arxiv.org/abs/2209.10729v2) [paper-pdf](http://arxiv.org/pdf/2209.10729v2)

**Authors**: Tsung-Han Wu, Hung-Ting Su, Shang-Tse Chen, Winston H. Hsu

**Abstract**: Fairness and robustness play vital roles in trustworthy machine learning. Observing safety-critical needs in various annotation-expensive vision applications, we introduce a novel learning framework, Fair Robust Active Learning (FRAL), generalizing conventional active learning to fair and adversarial robust scenarios. This framework allows us to achieve standard and robust minimax fairness with limited acquired labels. In FRAL, we then observe existing fairness-aware data selection strategies suffer from either ineffectiveness under severe data imbalance or inefficiency due to huge computations of adversarial training. To address these two problems, we develop a novel Joint INconsistency (JIN) method exploiting prediction inconsistencies between benign and adversarial inputs as well as between standard and robust models. These two inconsistencies can be used to identify potential fairness gains and data imbalance mitigations. Thus, by performing label acquisition with our inconsistency-based ranking metrics, we can alleviate the class imbalance issue and enhance minimax fairness with limited computation. Extensive experiments on diverse datasets and sensitive groups demonstrate that our method obtains the best results in standard and robust fairness under white-box PGD attacks compared with existing active data selection baselines.

摘要: 公平性和稳健性播放在可信机器学习中起着至关重要的作用。通过观察各种注解开销大的视觉应用中的安全关键需求，我们引入了一种新的学习框架--公平稳健主动学习(FRAL)，将传统的主动学习推广到公平和对抗性的稳健场景。该框架允许我们在获得有限标签的情况下实现标准和稳健的极大极小公平性。在Fral中，我们观察到现有的公平感知数据选择策略要么在严重的数据不平衡情况下效率低下，要么由于对抗性训练的巨大计算量而效率低下。为了解决这两个问题，我们开发了一种新的联合不一致性(JIN)方法，该方法利用良性和敌对输入之间以及标准和稳健模型之间的预测不一致性。这两个不一致可用于识别潜在的公平收益和数据失衡缓解。因此，通过使用基于不一致性的排序度量进行标签获取，我们可以在有限的计算中缓解类不平衡问题并提高极小极大公平性。在不同的数据集和敏感组上的大量实验表明，与现有的主动数据选择基准相比，该方法在白盒PGD攻击下获得了最好的标准和稳健的公平性结果。



## **12. Differentially Private Optimizers Can Learn Adversarially Robust Models**

不同的私有优化器可以学习相反的健壮模型 cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08942v1) [paper-pdf](http://arxiv.org/pdf/2211.08942v1)

**Authors**: Yuan Zhang, Zhiqi Bu

**Abstract**: Machine learning models have shone in a variety of domains and attracted increasing attention from both the security and the privacy communities. One important yet worrying question is: will training models under the differential privacy (DP) constraint unfavorably impact on the adversarial robustness? While previous works have postulated that privacy comes at the cost of worse robustness, we give the first theoretical analysis to show that DP models can indeed be robust and accurate, even sometimes more robust than their naturally-trained non-private counterparts. We observe three key factors that influence the privacy-robustness-accuracy tradeoff: (1) hyperparameters for DP optimizers are critical; (2) pre-training on public data significantly mitigates the accuracy and robustness drop; (3) choice of DP optimizers makes a difference. With these factors set properly, we achieve 90\% natural accuracy, 72\% robust accuracy ($+9\%$ than the non-private model) under $l_2(0.5)$ attack, and 69\% robust accuracy ($+16\%$ than the non-private model) with pre-trained SimCLRv2 model under $l_\infty(4/255)$ attack on CIFAR10 with $\epsilon=2$. In fact, we show both theoretically and empirically that DP models are Pareto optimal on the accuracy-robustness tradeoff. Empirically, the robustness of DP models is consistently observed on MNIST, Fashion MNIST and CelebA datasets, with ResNet and Vision Transformer. We believe our encouraging results are a significant step towards training models that are private as well as robust.

摘要: 机器学习模型已经在各个领域大放异彩，越来越受到安全和隐私界的关注。一个重要但令人担忧的问题是：差异隐私(DP)约束下的训练模型是否会对对手的健壮性产生不利影响？虽然以前的研究假设隐私是以更差的稳健性为代价的，但我们首次给出了理论分析，表明DP模型确实可以是健壮和准确的，有时甚至比自然训练的非私有模型更健壮。我们观察到影响隐私-稳健性-准确度权衡的三个关键因素：(1)DP优化器的超参数至关重要；(2)对公共数据的预训练显著缓解了准确率和稳健性下降；(3)DP优化器的选择产生了影响。在适当设置这些因素的情况下，我们在$l_2(0.5)$攻击下获得了90%的自然精度和72%的稳健精度(比非私有模型高出9美元)，在$l_Infty(4/255)美元攻击下，用预先训练好的SimCLRv2模型获得了69%的稳健精度(比非私有模型高出16美元)。事实上，我们在理论和经验上都证明了DP模型在精度和稳健性之间的权衡是帕累托最优的。从经验上看，DP模型的稳健性在MNIST、Fashion MNIST和CelebA数据集以及ResNet和Vision Transformer上得到了一致的观察。我们认为，我们令人鼓舞的结果是朝着私人和稳健的培训模式迈出了重要的一步。



## **13. Attacking Object Detector Using A Universal Targeted Label-Switch Patch**

基于通用靶标开关贴片的攻击目标探测器 cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08859v1) [paper-pdf](http://arxiv.org/pdf/2211.08859v1)

**Authors**: Avishag Shapira, Ron Bitton, Dan Avraham, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: Adversarial attacks against deep learning-based object detectors (ODs) have been studied extensively in the past few years. These attacks cause the model to make incorrect predictions by placing a patch containing an adversarial pattern on the target object or anywhere within the frame. However, none of prior research proposed a misclassification attack on ODs, in which the patch is applied on the target object. In this study, we propose a novel, universal, targeted, label-switch attack against the state-of-the-art object detector, YOLO. In our attack, we use (i) a tailored projection function to enable the placement of the adversarial patch on multiple target objects in the image (e.g., cars), each of which may be located a different distance away from the camera or have a different view angle relative to the camera, and (ii) a unique loss function capable of changing the label of the attacked objects. The proposed universal patch, which is trained in the digital domain, is transferable to the physical domain. We performed an extensive evaluation using different types of object detectors, different video streams captured by different cameras, and various target classes, and evaluated different configurations of the adversarial patch in the physical domain.

摘要: 在过去的几年里，针对基于深度学习的对象检测器(OD)的对抗性攻击已经被广泛研究。这些攻击通过在目标对象上或帧内的任何位置放置包含对抗性模式的补丁，导致模型做出错误的预测。然而，以往的研究都没有提出对OD的误分类攻击，即将补丁应用于目标对象。在这项研究中，我们提出了一种新的、通用的、有针对性的、针对最先进的对象检测器YOLO的标签切换攻击。在我们的攻击中，我们使用(I)定制的投影函数来实现在图像中的多个目标对象(例如汽车)上放置对抗性补丁，每个目标对象可能位于距离摄像机不同的距离或相对于摄像机具有不同的视角，以及(Ii)能够改变被攻击对象的标签的独特的损失函数。所提出的通用补丁是在数字域中训练的，可以转移到物理域。我们使用不同类型的目标检测器、不同摄像头捕获的不同视频流和不同的目标类别进行了广泛的评估，并在物理域评估了不同配置的对抗性补丁。



## **14. T-SEA: Transfer-based Self-Ensemble Attack on Object Detection**

T-SEA：基于传输的目标检测自集成攻击 cs.CV

10 pages, 5 figures

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.09773v1) [paper-pdf](http://arxiv.org/pdf/2211.09773v1)

**Authors**: Hao Huang, Ziyan Chen, Huanran Chen, Yongtao Wang, Kevin Zhang

**Abstract**: Compared to query-based black-box attacks, transfer-based black-box attacks do not require any information of the attacked models, which ensures their secrecy. However, most existing transfer-based approaches rely on ensembling multiple models to boost the attack transferability, which is time- and resource-intensive, not to mention the difficulty of obtaining diverse models on the same task. To address this limitation, in this work, we focus on the single-model transfer-based black-box attack on object detection, utilizing only one model to achieve a high-transferability adversarial attack on multiple black-box detectors. Specifically, we first make observations on the patch optimization process of the existing method and propose an enhanced attack framework by slightly adjusting its training strategies. Then, we analogize patch optimization with regular model optimization, proposing a series of self-ensemble approaches on the input data, the attacked model, and the adversarial patch to efficiently make use of the limited information and prevent the patch from overfitting. The experimental results show that the proposed framework can be applied with multiple classical base attack methods (e.g., PGD and MIM) to greatly improve the black-box transferability of the well-optimized patch on multiple mainstream detectors, meanwhile boosting white-box performance. Our code is available at https://github.com/VDIGPKU/T-SEA.

摘要: 与基于查询的黑盒攻击相比，基于传输的黑盒攻击不需要被攻击模型的任何信息，保证了被攻击模型的保密性。然而，现有的大多数基于转移的方法依赖于集成多个模型来提高攻击的可转移性，这是时间和资源密集型的，更不用说在同一任务上获得不同的模型的难度。针对这一局限性，本文重点研究了基于单模型转移的黑盒目标检测攻击，只利用一个模型实现了对多个黑盒检测器的高可转移性对抗性攻击。具体地说，我们首先对现有方法的补丁优化过程进行了观察，并通过对其训练策略进行略微调整，提出了一个增强的攻击框架。然后，我们将补丁优化类比为常规模型优化，提出了一系列针对输入数据、被攻击模型和敌意补丁的自集成方法，以有效利用有限的信息，防止补丁过拟合。实验结果表明，该框架可以与多种经典的基本攻击方法(如PGD和MIM)相结合，极大地提高优化后的补丁在多个主流检测器上的黑盒可转移性，同时提高白盒性能。我们的代码可以在https://github.com/VDIGPKU/T-SEA.上找到



## **15. Adversarial Camouflage for Node Injection Attack on Graphs**

图上节点注入攻击的对抗性伪装 cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2208.01819v2) [paper-pdf](http://arxiv.org/pdf/2208.01819v2)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Xueqi Cheng

**Abstract**: Node injection attacks against Graph Neural Networks (GNNs) have received emerging attention as a practical attack scenario, where the attacker injects malicious nodes instead of modifying node features or edges to degrade the performance of GNNs. Despite the initial success of node injection attacks, we find that the injected nodes by existing methods are easy to be distinguished from the original normal nodes by defense methods and limiting their attack performance in practice. To solve the above issues, we devote to camouflage node injection attack, i.e., camouflaging injected malicious nodes (structure/attributes) as the normal ones that appear legitimate/imperceptible to defense methods. The non-Euclidean nature of graph data and the lack of human prior brings great challenges to the formalization, implementation, and evaluation of camouflage on graphs. In this paper, we first propose and formulate the camouflage of injected nodes from both the fidelity and diversity of the ego networks centered around injected nodes. Then, we design an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve the camouflage while ensuring the attack performance. Several novel indicators for graph camouflage are further designed for a comprehensive evaluation. Experimental results demonstrate that when equipping existing node injection attack methods with our proposed CANA framework, the attack performance against defense methods as well as node camouflage is significantly improved.

摘要: 针对图神经网络的节点注入攻击作为一种实用的攻击场景受到了越来越多的关注，即攻击者注入恶意节点而不是修改节点特征或边来降低图神经网络的性能。尽管节点注入攻击取得了初步的成功，但我们发现，现有方法注入的节点很容易通过防御方法与原来的正常节点区分开来，限制了它们在实践中的攻击性能。为了解决上述问题，我们致力于伪装节点注入攻击，即伪装注入的恶意节点(结构/属性)作为正常的合法/不可察觉的防御方法。图数据的非欧几里得性质和人类先验知识的缺乏给图上伪装的形式化、实现和评估带来了巨大的挑战。本文首先从以注入节点为中心的EGO网络的保真度和多样性两个方面提出并构造了注入节点的伪装。然后，设计了一种节点注入攻击的对抗性伪装框架CANA，在保证攻击性能的同时提高伪装性能。进一步设计了几种新的图形伪装指标，进行了综合评价。实验结果表明，在现有的节点注入攻击方法中加入CANA框架后，对防御方法的攻击性能以及对节点伪装的攻击性能都得到了显著提高。



## **16. Semi-supervised Conditional GAN for Simultaneous Generation and Detection of Phishing URLs: A Game theoretic Perspective**

基于博弈论的半监督条件遗传算法同时生成和检测钓鱼URL cs.CR

5 Pages, 4 figures, 2 tables

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2108.01852v3) [paper-pdf](http://arxiv.org/pdf/2108.01852v3)

**Authors**: Sharif Amit Kamran, Shamik Sengupta, Alireza Tavakkoli

**Abstract**: Spear Phishing is a type of cyber-attack where the attacker sends hyperlinks through email on well-researched targets. The objective is to obtain sensitive information by imitating oneself as a trustworthy website. In recent times, deep learning has become the standard for defending against such attacks. However, these architectures were designed with only defense in mind. Moreover, the attacker's perspective and motivation are absent while creating such models. To address this, we need a game-theoretic approach to understand the perspective of the attacker (Hacker) and the defender (Phishing URL detector). We propose a Conditional Generative Adversarial Network with novel training strategy for real-time phishing URL detection. Additionally, we train our architecture in a semi-supervised manner to distinguish between adversarial and real examples, along with detecting malicious and benign URLs. We also design two games between the attacker and defender in training and deployment settings by utilizing the game-theoretic perspective. Our experiments confirm that the proposed architecture surpasses recent state-of-the-art architectures for phishing URLs detection.

摘要: 鱼叉式网络钓鱼是一种网络攻击，攻击者通过电子邮件向经过充分研究的目标发送超链接。其目标是通过将自己伪装成一个值得信赖的网站来获取敏感信息。最近，深度学习已成为防御此类攻击的标准。然而，这些架构在设计时只考虑到了防御。此外，在创建这样的模型时，攻击者的视角和动机是缺失的。为了解决这个问题，我们需要一个博弈论的方法来理解攻击者(黑客)和防御者(网络钓鱼URL检测器)的角度。提出了一种具有新颖训练策略的条件生成对抗网络，用于实时网络钓鱼URL检测。此外，我们以半监督的方式训练我们的体系结构，以区分敌意和真实的示例，以及检测恶意和良性URL。我们还利用博弈论的观点设计了攻防双方在训练和部署环境下的两场比赛。我们的实验证实，该体系结构在网络钓鱼URL检测方面优于目前最先进的体系结构。



## **17. Improving Interpretability via Regularization of Neural Activation Sensitivity**

通过神经激活敏感度的正则化提高可读性 cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08686v1) [paper-pdf](http://arxiv.org/pdf/2211.08686v1)

**Authors**: Ofir Moshe, Gil Fidel, Ron Bitton, Asaf Shabtai

**Abstract**: State-of-the-art deep neural networks (DNNs) are highly effective at tackling many real-world tasks. However, their wide adoption in mission-critical contexts is hampered by two major weaknesses - their susceptibility to adversarial attacks and their opaqueness. The former raises concerns about the security and generalization of DNNs in real-world conditions, whereas the latter impedes users' trust in their output. In this research, we (1) examine the effect of adversarial robustness on interpretability and (2) present a novel approach for improving the interpretability of DNNs that is based on regularization of neural activation sensitivity. We evaluate the interpretability of models trained using our method to that of standard models and models trained using state-of-the-art adversarial robustness techniques. Our results show that adversarially robust models are superior to standard models and that models trained using our proposed method are even better than adversarially robust models in terms of interpretability.

摘要: 最先进的深度神经网络(DNN)在处理许多现实世界的任务时非常有效。然而，它们在任务关键型环境中的广泛采用受到两个主要弱点的阻碍--它们易受对抗性攻击和它们的不透明。前者引起了人们对DNN在现实世界条件下的安全性和普适性的担忧，而后者阻碍了用户对其输出的信任。在本研究中，我们(1)考察了敌对性健壮性对可解释性的影响，(2)提出了一种基于神经激活敏感度正则化的改进DNN可解释性的新方法。我们评估了使用我们的方法训练的模型与标准模型和使用最先进的对抗健壮性技术训练的模型的可解释性。我们的结果表明，逆稳健模型优于标准模型，并且用我们提出的方法训练的模型在可解释性方面甚至优于逆稳健模型。



## **18. Nano-Resolution Visual Identifiers Enable Secure Monitoring in Next-Generation Cyber-Physical Systems**

纳米分辨率视觉识别器在下一代网络物理系统中实现安全监控 cs.CR

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08678v1) [paper-pdf](http://arxiv.org/pdf/2211.08678v1)

**Authors**: Hao Wang, Xiwen Chen, Abolfazl Razi, Michael Kozicki, Rahul Amin, Mark Manfredo

**Abstract**: Today's supply chains heavily rely on cyber-physical systems such as intelligent transportation, online shopping, and E-commerce. It is advantageous to track goods in real-time by web-based registration and authentication of products after any substantial change or relocation. Despite recent advantages in technology-based tracking systems, most supply chains still rely on plainly printed tags such as barcodes and Quick Response (QR) codes for tracking purposes. Although affordable and efficient, these tags convey no security against counterfeit and cloning attacks, raising privacy concerns. It is a critical matter since a few security breaches in merchandise databases in recent years has caused crucial social and economic impacts such as identity loss, social panic, and loss of trust in the community. This paper considers an end-to-end system using dendrites as nano-resolution visual identifiers to secure supply chains. Dendrites are formed by generating fractal metallic patterns on transparent substrates through an electrochemical process, which can be used as secure identifiers due to their natural randomness, high entropy, and unclonable features. The proposed framework compromises the back-end program for identification and authentication, a web-based application for mobile devices, and a cloud database. We review architectural design, dendrite operational phases (personalization, registration, inspection), a lightweight identification method based on 2D graph-matching, and a deep 3D image authentication method based on Digital Holography (DH). A two-step search is proposed to make the system scalable by limiting the search space to samples with high similarity scores in a lower-dimensional space. We conclude by presenting our solution to make dendrites secure against adversarial attacks.

摘要: 今天的供应链严重依赖于智能交通、在线购物和电子商务等网络物理系统。有利的是，在任何重大更改或搬迁之后，通过产品的基于网络的注册和认证来实时跟踪货物。尽管最近基于技术的跟踪系统具有优势，但大多数供应链仍依赖条形码和快速响应(QR)码等简单打印的标签进行跟踪。尽管价格实惠且高效，但这些标签没有提供针对假冒和克隆攻击的安全性，这引发了隐私问题。这是一个关键问题，因为近年来商品数据库中的几个安全漏洞已经造成了严重的社会和经济影响，如身份丧失、社会恐慌和对社区的信任丧失。本文考虑了一种端到端系统，该系统使用树枝作为纳米分辨率的视觉识别器来保护供应链。树枝晶是通过电化学过程在透明的衬底上产生分形的金属图案而形成的，树枝晶具有天然的随机性、高熵和不可克隆的特点，可以用作安全识别符。拟议的框架包括身份识别和身份验证的后端程序、移动设备的基于Web的应用程序和云数据库。我们回顾了建筑设计、树枝操作阶段(个性化、注册、检查)、基于二维图匹配的轻量级识别方法和基于数字全息术的深度三维图像认证方法。提出了一种两步搜索，通过将搜索空间限制在较低维空间中具有高相似性得分的样本来使系统具有可伸缩性。最后，我们提出了我们的解决方案，以确保树状结构安全，不受对手攻击。



## **19. Person Text-Image Matching via Text-Featur Interpretability Embedding and External Attack Node Implantation**

基于文本特征可解释性嵌入和外部攻击节点植入的人文本图像匹配 cs.CV

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2211.08657v1) [paper-pdf](http://arxiv.org/pdf/2211.08657v1)

**Authors**: Fan Li, Hang Zhou, Huafeng Li, Yafei Zhang, Zhengtao Yu

**Abstract**: Person text-image matching, also known as textbased person search, aims to retrieve images of specific pedestrians using text descriptions. Although person text-image matching has made great research progress, existing methods still face two challenges. First, the lack of interpretability of text features makes it challenging to effectively align them with their corresponding image features. Second, the same pedestrian image often corresponds to multiple different text descriptions, and a single text description can correspond to multiple different images of the same identity. The diversity of text descriptions and images makes it difficult for a network to extract robust features that match the two modalities. To address these problems, we propose a person text-image matching method by embedding text-feature interpretability and an external attack node. Specifically, we improve the interpretability of text features by providing them with consistent semantic information with image features to achieve the alignment of text and describe image region features.To address the challenges posed by the diversity of text and the corresponding person images, we treat the variation caused by diversity to features as caused by perturbation information and propose a novel adversarial attack and defense method to solve it. In the model design, graph convolution is used as the basic framework for feature representation and the adversarial attacks caused by text and image diversity on feature extraction is simulated by implanting an additional attack node in the graph convolution layer to improve the robustness of the model against text and image diversity. Extensive experiments demonstrate the effectiveness and superiority of text-pedestrian image matching over existing methods. The source code of the method is published at

摘要: 人的文本-图像匹配，也称为基于文本的人搜索，旨在使用文本描述检索特定行人的图像。虽然人的文本-图像匹配已经取得了很大的研究进展，但现有的方法仍然面临着两个方面的挑战。首先，文本特征缺乏可解释性，这使得有效地将它们与其对应的图像特征对齐具有挑战性。其次，相同的行人图像往往对应于多个不同的文本描述，并且单个文本描述可以对应于相同身份的多个不同图像。文本描述和图像的多样性使得网络很难提取匹配这两种模式的稳健特征。针对这些问题，我们提出了一种嵌入文本特征可解释性和外部攻击节点的人文本图像匹配方法。具体而言，通过为文本特征提供与图像特征一致的语义信息来提高文本特征的可解释性，从而实现文本对齐和描述图像区域特征；针对文本和对应人物图像的多样性带来的挑战，我们将特征多样性引起的变异看作是扰动信息引起的，并提出了一种新的对抗性攻防方法。在模型设计中，使用图卷积作为特征表示的基本框架，通过在图卷积层中增加一个攻击节点来模拟文本和图像多样性对特征提取的敌意攻击，以提高模型对文本和图像多样性的鲁棒性。大量的实验证明了文本-行人图像匹配方法的有效性和优越性。该方法的源代码发布在



## **20. Membership Inference Attacks Against Temporally Correlated Data in Deep Reinforcement Learning**

深度强化学习中对时间相关数据的隶属度推理攻击 cs.LG

**SubmitDate**: 2022-11-16    [abs](http://arxiv.org/abs/2109.03975v3) [paper-pdf](http://arxiv.org/pdf/2109.03975v3)

**Authors**: Maziar Gomrokchi, Susan Amin, Hossein Aboutalebi, Alexander Wong, Doina Precup

**Abstract**: While significant research advances have been made in the field of deep reinforcement learning, there have been no concrete adversarial attack strategies in literature tailored for studying the vulnerability of deep reinforcement learning algorithms to membership inference attacks. In such attacking systems, the adversary targets the set of collected input data on which the deep reinforcement learning algorithm has been trained. To address this gap, we propose an adversarial attack framework designed for testing the vulnerability of a state-of-the-art deep reinforcement learning algorithm to a membership inference attack. In particular, we design a series of experiments to investigate the impact of temporal correlation, which naturally exists in reinforcement learning training data, on the probability of information leakage. Moreover, we compare the performance of \emph{collective} and \emph{individual} membership attacks against the deep reinforcement learning algorithm. Experimental results show that the proposed adversarial attack framework is surprisingly effective at inferring data with an accuracy exceeding $84\%$ in individual and $97\%$ in collective modes in three different continuous control Mujoco tasks, which raises serious privacy concerns in this regard. Finally, we show that the learning state of the reinforcement learning algorithm influences the level of privacy breaches significantly.

摘要: 虽然在深度强化学习领域已经取得了重要的研究进展，但还没有专门为研究深度强化学习算法对成员推理攻击的脆弱性而定制的具体的对抗性攻击策略。在这样的攻击系统中，对手的目标是已在其上训练深度强化学习算法的所收集的输入数据集。为了弥补这一缺陷，我们提出了一个对抗性攻击框架，用于测试现有的深度强化学习算法对成员关系推理攻击的脆弱性。特别是，我们设计了一系列实验来研究强化学习训练数据中自然存在的时间相关性对信息泄漏概率的影响。此外，我们还比较了成员身份攻击和深度强化学习算法的性能。实验结果表明，在三种不同的连续控制Mujoco任务中，提出的对抗性攻击框架在推断数据时具有惊人的有效性，在个体模式下的准确率超过84美元，在集体模式下的准确率超过97美元，这引发了严重的隐私问题。最后，我们发现强化学习算法的学习状态对隐私泄露的程度有显著的影响。



## **21. Universal Distributional Decision-based Black-box Adversarial Attack with Reinforcement Learning**

基于强化学习的通用分布式决策黑盒对抗攻击 cs.LG

10 pages, 2 figures, conference

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08384v1) [paper-pdf](http://arxiv.org/pdf/2211.08384v1)

**Authors**: Yiran Huang, Yexu Zhou, Michael Hefenbrock, Till Riedel, Likun Fang, Michael Beigl

**Abstract**: The vulnerability of the high-performance machine learning models implies a security risk in applications with real-world consequences. Research on adversarial attacks is beneficial in guiding the development of machine learning models on the one hand and finding targeted defenses on the other. However, most of the adversarial attacks today leverage the gradient or logit information from the models to generate adversarial perturbation. Works in the more realistic domain: decision-based attacks, which generate adversarial perturbation solely based on observing the output label of the targeted model, are still relatively rare and mostly use gradient-estimation strategies. In this work, we propose a pixel-wise decision-based attack algorithm that finds a distribution of adversarial perturbation through a reinforcement learning algorithm. We call this method Decision-based Black-box Attack with Reinforcement learning (DBAR). Experiments show that the proposed approach outperforms state-of-the-art decision-based attacks with a higher attack success rate and greater transferability.

摘要: 高性能机器学习模型的漏洞意味着应用程序中存在安全风险，并会产生现实世界的后果。对抗性攻击的研究一方面有利于指导机器学习模型的发展，另一方面有助于发现有针对性的防御措施。然而，今天的大多数对抗性攻击利用模型中的梯度或Logit信息来产生对抗性扰动。工作在更现实的领域：基于决策的攻击，仅根据观察目标模型的输出标签产生对抗性扰动，仍然相对较少，主要使用梯度估计策略。在这项工作中，我们提出了一种基于像素的基于决策的攻击算法，该算法通过强化学习算法找到对抗性扰动的分布。我们称这种方法为带强化学习的基于决策的黑盒攻击(DBAR)。实验表明，该方法比现有的基于决策的攻击方法具有更高的攻击成功率和更强的可移植性。



## **22. Resisting Graph Adversarial Attack via Cooperative Homophilous Augmentation**

基于协作式同构增强的图对抗攻击 cs.LG

The paper has been accepted for presentation at ECML PKDD 2022

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08068v1) [paper-pdf](http://arxiv.org/pdf/2211.08068v1)

**Authors**: Zhihao Zhu, Chenwang Wu, Min Zhou, Hao Liao, Defu Lian, Enhong Chen

**Abstract**: Recent studies show that Graph Neural Networks(GNNs) are vulnerable and easily fooled by small perturbations, which has raised considerable concerns for adapting GNNs in various safety-critical applications. In this work, we focus on the emerging but critical attack, namely, Graph Injection Attack(GIA), in which the adversary poisons the graph by injecting fake nodes instead of modifying existing structures or node attributes. Inspired by findings that the adversarial attacks are related to the increased heterophily on perturbed graphs (the adversary tends to connect dissimilar nodes), we propose a general defense framework CHAGNN against GIA through cooperative homophilous augmentation of graph data and model. Specifically, the model generates pseudo-labels for unlabeled nodes in each round of training to reduce heterophilous edges of nodes with distinct labels. The cleaner graph is fed back to the model, producing more informative pseudo-labels. In such an iterative manner, model robustness is then promisingly enhanced. We present the theoretical analysis of the effect of homophilous augmentation and provide the guarantee of the proposal's validity. Experimental results empirically demonstrate the effectiveness of CHAGNN in comparison with recent state-of-the-art defense methods on diverse real-world datasets.

摘要: 最近的研究表明，图神经网络(GNN)是脆弱的，容易被微小的扰动所愚弄，这使得GNN在各种安全关键应用中的应用受到了相当大的关注。在这项工作中，我们关注的是一种新兴但关键的攻击，即图注入攻击(GIA)，在该攻击中，对手通过注入虚假节点而不是修改现有的结构或节点属性来毒化图。受敌意攻击与扰动图上的异质性增加(对手倾向于连接不同的节点)有关的发现的启发，我们通过图数据和模型的协同性增强，提出了一个针对GIA的通用防御框架CHAGNN。具体地说，该模型在每一轮训练中为未标记的节点生成伪标签，以减少具有不同标签的节点的异嗜性边缘。更干净的图形被反馈给模型，产生更多信息更丰富的伪标签。在这样的迭代方式下，模型的稳健性有望得到增强。我们从理论上分析了同源增强的效果，并为该方案的有效性提供了保证。实验结果表明，在不同的真实数据集上，CHAGNN与目前最先进的防御方法相比是有效的。



## **23. MORA: Improving Ensemble Robustness Evaluation with Model-Reweighing Attack**

MORA：利用模型重加权攻击改进集成健壮性评估 cs.LG

To appear in NeurIPS 2022. Project repository:  https://github.com/lafeat/mora

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.08008v1) [paper-pdf](http://arxiv.org/pdf/2211.08008v1)

**Authors**: Yunrui Yu, Xitong Gao, Cheng-Zhong Xu

**Abstract**: Adversarial attacks can deceive neural networks by adding tiny perturbations to their input data. Ensemble defenses, which are trained to minimize attack transferability among sub-models, offer a promising research direction to improve robustness against such attacks while maintaining a high accuracy on natural inputs. We discover, however, that recent state-of-the-art (SOTA) adversarial attack strategies cannot reliably evaluate ensemble defenses, sizeably overestimating their robustness. This paper identifies the two factors that contribute to this behavior. First, these defenses form ensembles that are notably difficult for existing gradient-based method to attack, due to gradient obfuscation. Second, ensemble defenses diversify sub-model gradients, presenting a challenge to defeat all sub-models simultaneously, simply summing their contributions may counteract the overall attack objective; yet, we observe that ensemble may still be fooled despite most sub-models being correct. We therefore introduce MORA, a model-reweighing attack to steer adversarial example synthesis by reweighing the importance of sub-model gradients. MORA finds that recent ensemble defenses all exhibit varying degrees of overestimated robustness. Comparing it against recent SOTA white-box attacks, it can converge orders of magnitude faster while achieving higher attack success rates across all ensemble models examined with three different ensemble modes (i.e., ensembling by either softmax, voting or logits). In particular, most ensemble defenses exhibit near or exactly 0% robustness against MORA with $\ell^\infty$ perturbation within 0.02 on CIFAR-10, and 0.01 on CIFAR-100. We make MORA open source with reproducible results and pre-trained models; and provide a leaderboard of ensemble defenses under various attack strategies.

摘要: 对抗性攻击可以通过向神经网络的输入数据添加微小扰动来欺骗神经网络。集成防御被训练为最小化子模型之间的攻击转移，为在保持对自然输入的高精度的同时提高对此类攻击的稳健性提供了一个有前途的研究方向。然而，我们发现，最近最先进的(SOTA)对抗性攻击策略不能可靠地评估整体防御，严重高估了它们的健壮性。本文确定了导致这一行为的两个因素。首先，由于梯度混淆，这些防御形成了集合，对于现有的基于梯度的方法来说，这些集合很难攻击。其次，集合防御使子模型梯度多样化，提出了同时击败所有子模型的挑战，简单地计算它们的贡献可能会抵消整体攻击目标；然而，我们观察到，尽管大多数子模型是正确的，但集合仍然可能被愚弄。因此，我们引入了Mora，一种模型重加权攻击，通过重新权重子模型梯度的重要性来指导对抗性实例合成。莫拉发现，最近的整体防御都表现出不同程度的高估了稳健性。与最近的Sota白盒攻击相比，它可以更快地收敛数量级，同时在使用三种不同集成模式(即，通过Softmax、投票或逻辑进行集成)检查的所有集成模型上实现更高的攻击成功率。特别是，大多数集合防御系统对Mora表现出接近或恰好0%的鲁棒性，在CIFAR-10上的摄动在0.02以内，在CIFAR-100上在0.01以内。我们使Mora开源，具有可重现的结果和预先训练的模型；并提供各种攻击策略下的整体防御排行榜。



## **24. Security Closure of IC Layouts Against Hardware Trojans**

针对硬件特洛伊木马的IC布局的安全关闭 cs.CR

To appear in ISPD'23

**SubmitDate**: 2022-11-15    [abs](http://arxiv.org/abs/2211.07997v1) [paper-pdf](http://arxiv.org/pdf/2211.07997v1)

**Authors**: Fangzhou Wang, Qijing Wang, Bangqi Fu, Shui Jiang, Xiaopeng Zhang, Lilas Alrahis, Ozgur Sinanoglu, Johann Knechtel, Tsung-Yi Ho, Evangeline F. Y. Young

**Abstract**: Due to cost benefits, supply chains of integrated circuits (ICs) are largely outsourced nowadays. However, passing ICs through various third-party providers gives rise to many threats, like piracy of IC intellectual property or insertion of hardware Trojans, i.e., malicious circuit modifications.   In this work, we proactively and systematically harden the physical layouts of ICs against post-design insertion of Trojans. Toward that end, we propose a multiplexer-based logic-locking scheme that is (i) devised for layout-level Trojan prevention, (ii) resilient against state-of-the-art, oracle-less machine learning attacks, and (iii) fully integrated into a tailored, yet generic, commercial-grade design flow. Our work provides in-depth security and layout analysis on a challenging benchmark suite. We show that ours can render layouts resilient, with reasonable overheads, against Trojan insertion in general and also against second-order attacks (i.e., adversaries seeking to bypass the locking defense in an oracle-less setting).   We release our layout artifacts for independent verification [29] and we will release our methodology's source code.

摘要: 由于成本效益，如今集成电路(IC)的供应链大多被外包。然而，通过各种第三方提供商传递IC会带来许多威胁，如盗版IC知识产权或插入硬件特洛伊木马程序，即恶意电路修改。在这项工作中，我们主动和系统地加强了IC的物理布局，以防止设计后插入特洛伊木马。为此，我们提出了一种基于多路复用器的逻辑锁定方案，该方案(I)针对布局级特洛伊木马程序的预防而设计，(Ii)对最先进的、无预言的机器学习攻击具有弹性，(Iii)完全集成到定制的、但通用的商业级设计流程中。我们的工作为具有挑战性的基准套件提供深入的安全和布局分析。我们表明，我们的布局可以使布局具有弹性，具有合理的开销，一般可以抵御特洛伊木马插入，也可以抵御二次攻击(即，寻求在无Oracle设置中绕过锁定防御的对手)。我们将发布我们的布局构件以供独立验证[29]，我们还将发布我们方法的源代码。



## **25. Towards Robust Numerical Question Answering: Diagnosing Numerical Capabilities of NLP Systems**

走向稳健的数值问题回答：诊断NLP系统的数值能力 cs.CL

Accepted by EMNLP'2022

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07455v1) [paper-pdf](http://arxiv.org/pdf/2211.07455v1)

**Authors**: Jialiang Xu, Mengyu Zhou, Xinyi He, Shi Han, Dongmei Zhang

**Abstract**: Numerical Question Answering is the task of answering questions that require numerical capabilities. Previous works introduce general adversarial attacks to Numerical Question Answering, while not systematically exploring numerical capabilities specific to the topic. In this paper, we propose to conduct numerical capability diagnosis on a series of Numerical Question Answering systems and datasets. A series of numerical capabilities are highlighted, and corresponding dataset perturbations are designed. Empirical results indicate that existing systems are severely challenged by these perturbations. E.g., Graph2Tree experienced a 53.83% absolute accuracy drop against the ``Extra'' perturbation on ASDiv-a, and BART experienced 13.80% accuracy drop against the ``Language'' perturbation on the numerical subset of DROP. As a counteracting approach, we also investigate the effectiveness of applying perturbations as data augmentation to relieve systems' lack of robust numerical capabilities. With experiment analysis and empirical studies, it is demonstrated that Numerical Question Answering with robust numerical capabilities is still to a large extent an open question. We discuss future directions of Numerical Question Answering and summarize guidelines on future dataset collection and system design.

摘要: 数字问题回答是回答需要数字能力的问题的任务。以前的工作介绍了一般的对抗性攻击数字问题回答，但没有系统地探索特定于主题的数字能力。在本文中，我们提出对一系列数值问答系统和数据集进行数值能力诊断。突出了一系列的数值能力，并设计了相应的数据集扰动。实证结果表明，现有系统受到这些扰动的严重挑战。例如，Graph2Tree在ASDiv-a上经历了53.83%的绝对准确率下降，而BART在Drop的数值子集上经历了13.80%的准确率下降。作为一种抵消方法，我们还研究了应用扰动作为数据增强来缓解系统缺乏健壮的数值能力的有效性。实验分析和实证研究表明，具有较强数字能力的数字问题回答在很大程度上仍然是一个开放的问题。我们讨论了未来数字问题回答的方向，并总结了未来数据集收集和系统设计的指导方针。



## **26. Privacy and Security in Network Controlled Systems via Dynamic Masking**

基于动态掩蔽的网络控制系统中的隐私与安全 eess.SY

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07328v1) [paper-pdf](http://arxiv.org/pdf/2211.07328v1)

**Authors**: Mohamed Abdalmoaty, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: In this paper, we propose a new architecture to enhance the privacy and security of networked control systems against malicious adversaries. We consider an adversary which first learns the system dynamics (privacy) using system identification techniques, and then performs a data injection attack (security). In particular, we consider an adversary conducting zero-dynamics attacks (ZDA) which maximizes the performance cost of the system whilst staying undetected. However, using the proposed architecture, we show that it is possible to (i) introduce significant bias in the system estimates of the adversary: thus providing privacy of the system parameters, and (ii) efficiently detect attacks when the adversary performs a ZDA using the identified system: thus providing security. Through numerical simulations, we illustrate the efficacy of the proposed architecture.

摘要: 在本文中，我们提出了一种新的体系结构来增强网络控制系统的隐私性和安全性，以抵御恶意攻击者。我们考虑这样一个对手，他首先使用系统识别技术学习系统动态(隐私)，然后执行数据注入攻击(安全)。特别是，我们考虑了一个进行零动态攻击(ZDA)的对手，该攻击最大化了系统的性能成本，同时保持不被发现。然而，使用所提出的体系结构，我们表明：(I)在敌手的系统估计中引入显著的偏差：从而提供系统参数的私密性；(Ii)当敌手使用所识别的系统执行ZDA时，有效地检测攻击：从而提供安全性。通过数值模拟，验证了该体系结构的有效性。



## **27. Jacobian Norm with Selective Input Gradient Regularization for Improved and Interpretable Adversarial Defense**

改进的可解释对抗防御的选择输入梯度正则化雅可比范数 cs.LG

Under review

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2207.13036v4) [paper-pdf](http://arxiv.org/pdf/2207.13036v4)

**Authors**: Deyin Liu, Lin Wu, Haifeng Zhao, Farid Boussaid, Mohammed Bennamoun, Xianghua Xie

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve robustness through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with transferred adversarial examples which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose a novel approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.

摘要: 众所周知，深度神经网络(DNN)容易受到带有不可察觉扰动的敌意示例的影响，即输入图像的微小变化就会导致误分类，从而威胁到基于深度学习的部署系统的可靠性。对抗性训练(AT)经常被用来通过训练被破坏的和被清理的数据的混合来提高稳健性。然而，大多数基于AT的方法都不能有效地处理转移的对抗性例子，这些例子是为了愚弄广泛的防御模型而产生的，因此不能满足现实场景中提出的泛化要求。此外，对抗性地训练防御模型一般不能产生对带有扰动的输入的可解释预测，而不同领域的专家需要高度可解释的稳健模型来理解DNN的行为。在本文中，我们提出了一种基于雅可比范数和选择性输入梯度正则化(J-SIGR)的新方法，该方法通过雅可比归一化提供线性化的稳健性，并将基于扰动的显著图正则化以模拟模型的可解释预测。因此，我们实现了DNN的改进的防御性和高度的可解释性。最后，我们在不同的体系结构上对我们的方法进行了评估，以对抗强大的对手攻击。实验表明，所提出的J-SIGR算法对转移攻击具有较好的稳健性，并且神经网络的预测结果易于解释。



## **28. Securing Access to Untrusted Services From TEEs with GateKeeper**

通过带有网守的T恤保护对不受信任服务的访问 cs.CR

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2211.07185v1) [paper-pdf](http://arxiv.org/pdf/2211.07185v1)

**Authors**: Meni Orenbach, Bar Raveh, Alon Berkenstadt, Yan Michalevsky, Shachar Itzhaky, Mark Silberstein

**Abstract**: Applications running in Trusted Execution Environments (TEEs) commonly use untrusted external services such as host File System. Adversaries may maliciously alter the normal service behavior to trigger subtle application bugs that would have never occurred under correct service operation, causing data leaks and integrity violations. Unfortunately, existing manual protections are incomplete and ad-hoc, whereas formally-verified ones require special expertise.   We introduce GateKeeper, a framework to develop mitigations and vulnerability checkers for such attacks by leveraging lightweight formal models of untrusted services. With the attack seen as a violation of a services' functional correctness, GateKeeper takes a novel approach to develop a comprehensive model of a service without requiring formal methods expertise. We harness available testing suites routinely used in service development to tighten the model to known correct service implementation. GateKeeper uses the resulting model to automatically generate (1) a correct-by-construction runtime service validator in C that is linked with a trusted application and guards each service invocation to conform to the model; and (2) a targeted model-driven vulnerability checker for analyzing black-box applications.   We evaluate GateKeeper on Intel SGX enclaves. We develop comprehensive models of a POSIX file system and OS synchronization primitives while using thousands of existing test suites to tighten their models to the actual Linux implementations. We generate the validator and integrate it with Graphene-SGX, and successfully protect unmodified Memcached and SQLite with negligible overheads. The generated vulnerability checker detects novel vulnerabilities in the Graphene-SGX protection layer and production applications.

摘要: 在可信执行环境(TEE)中运行的应用程序通常使用不可信的外部服务，如主机文件系统。攻击者可能会恶意改变正常的服务行为，以触发在正确的服务操作下永远不会发生的微妙的应用程序错误，从而导致数据泄漏和完整性破坏。不幸的是，现有的人工保护是不完整的和临时的，而正式验证的保护需要特殊的专业知识。我们引入了GateKeeper，这是一个框架，通过利用不可信服务的轻量级正式模型来开发此类攻击的缓解和漏洞检查器。由于攻击被视为对服务功能正确性的违反，GateKeeper采用了一种新的方法来开发服务的综合模型，而不需要正式的方法专业知识。我们利用服务开发中常规使用的可用测试套件来加强模型，使之成为已知正确的服务实现。GateKeeper使用生成的模型自动生成(1)C语言的按构造正确的运行时服务验证器，该验证器与受信任的应用程序链接，并保护每个服务调用以符合模型；以及(2)用于分析黑盒应用程序的目标模型驱动的漏洞检查器。我们在Intel SGX Enclaves上评估网守。我们开发了POSIX文件系统和操作系统同步原语的全面模型，同时使用数千个现有的测试套件来加强它们的模型，使其符合实际的Linux实现。我们生成了验证器，并将其与Graphene-SGX集成在一起，成功地保护了未修改的Memcach和SQLite，而开销可以忽略不计。生成的漏洞检查器检测石墨烯-SGX保护层和生产应用程序中的新漏洞。



## **29. Robust Deep Semi-Supervised Learning: A Brief Introduction**

稳健深度半监督学习：简介 cs.LG

We will rewrite this paper

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2202.05975v2) [paper-pdf](http://arxiv.org/pdf/2202.05975v2)

**Authors**: Lan-Zhe Guo, Zhi Zhou, Yu-Feng Li

**Abstract**: Semi-supervised learning (SSL) is the branch of machine learning that aims to improve learning performance by leveraging unlabeled data when labels are insufficient. Recently, SSL with deep models has proven to be successful on standard benchmark tasks. However, they are still vulnerable to various robustness threats in real-world applications as these benchmarks provide perfect unlabeled data, while in realistic scenarios, unlabeled data could be corrupted. Many researchers have pointed out that after exploiting corrupted unlabeled data, SSL suffers severe performance degradation problems. Thus, there is an urgent need to develop SSL algorithms that could work robustly with corrupted unlabeled data. To fully understand robust SSL, we conduct a survey study. We first clarify a formal definition of robust SSL from the perspective of machine learning. Then, we classify the robustness threats into three categories: i) distribution corruption, i.e., unlabeled data distribution is mismatched with labeled data; ii) feature corruption, i.e., the features of unlabeled examples are adversarially attacked; and iii) label corruption, i.e., the label distribution of unlabeled data is imbalanced. Under this unified taxonomy, we provide a thorough review and discussion of recent works that focus on these issues. Finally, we propose possible promising directions within robust SSL to provide insights for future research.

摘要: 半监督学习(半监督学习)是机器学习的一个分支，旨在通过在标签不足时利用未标记的数据来提高学习性能。最近，具有深度模型的SSL在标准基准任务中被证明是成功的。然而，它们在实际应用中仍然容易受到各种健壮性威胁，因为这些基准提供了完美的未标记数据，而在现实场景中，未标记数据可能会被破坏。许多研究人员指出，在利用损坏的未标记数据后，SSL面临着严重的性能下降问题。因此，迫切需要开发能够稳健地处理被破坏的未标记数据的SSL算法。为了全面了解健壮的SSL，我们进行了一项调查研究。我们首先从机器学习的角度阐明了健壮SSL的形式化定义。然后，我们将健壮性威胁分为三类：i)分布破坏，即未标记数据分布与已标记数据不匹配；ii)特征破坏，即未标记样本的特征受到相反攻击；以及iii)标签破坏，即未标记数据的标记分布不平衡。在这个统一的分类法下，我们对最近关注这些问题的工作进行了彻底的回顾和讨论。最后，我们提出了在健壮的SSL中可能有前景的方向，为未来的研究提供见解。



## **30. Optimization for Robustness Evaluation beyond $\ell_p$ Metrics**

超越$\ell_p$度量的健壮性评估优化 cs.LG

5 pages, 1 figure, 3 tables, accepted by the 14th International OPT  Workshop on Optimization for Machine Learning, and submitted to the 2023 IEEE  International Conference on Acoustics, Speech, and Signal Processing (ICASSP  2023)

**SubmitDate**: 2022-11-14    [abs](http://arxiv.org/abs/2210.00621v2) [paper-pdf](http://arxiv.org/pdf/2210.00621v2)

**Authors**: Hengyue Liang, Buyun Liang, Ying Cui, Tim Mitchell, Ju Sun

**Abstract**: Empirical evaluation of deep learning models against adversarial attacks entails solving nontrivial constrained optimization problems. Popular algorithms for solving these constrained problems rely on projected gradient descent (PGD) and require careful tuning of multiple hyperparameters. Moreover, PGD can only handle $\ell_1$, $\ell_2$, and $\ell_\infty$ attack models due to the use of analytical projectors. In this paper, we introduce a novel algorithmic framework that blends a general-purpose constrained-optimization solver PyGRANSO, With Constraint-Folding (PWCF), to add reliability and generality to robustness evaluation. PWCF 1) finds good-quality solutions without the need of delicate hyperparameter tuning, and 2) can handle general attack models, e.g., general $\ell_p$ ($p \geq 0$) and perceptual attacks, which are inaccessible to PGD-based algorithms.

摘要: 针对敌意攻击的深度学习模型的经验评估需要解决非平凡的约束优化问题。解决这些约束问题的流行算法依赖于投影梯度下降(PGD)，并且需要仔细调整多个超参数。此外，由于分析投影仪的使用，PGD只能处理$\ell_1$、$\ell_2$和$\ell_\inty$攻击模型。在本文中，我们提出了一个新的算法框架，将通用的约束优化求解器PyGRANSO与约束折叠(PWCF)相结合，以增加健壮性评估的可靠性和通用性。PWCF1)在不需要调整超参数的情况下找到高质量的解，2)可以处理一般攻击模型，例如一般的$\ell_p$($p\geq 0$)和感知攻击，这是基于PGD的算法无法访问的。



## **31. Watermarking Graph Neural Networks based on Backdoor Attacks**

基于后门攻击的数字水印图神经网络 cs.LG

18 pages, 9 figures

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2110.11024v5) [paper-pdf](http://arxiv.org/pdf/2110.11024v5)

**Authors**: Jing Xu, Stefanos Koffas, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. Moreover, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, verifying the ownership of the GNN models is necessary.   This paper presents a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (up to $99\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against a state-of-the-art model extraction technique and four state-of-the-art defenses against backdoor attacks.

摘要: 图神经网络(GNN)在各种实际应用中取得了良好的性能。构建一个强大的GNN模型不是一项简单的任务，因为它需要大量的训练数据、强大的计算资源和微调模型的人力专业知识。此外，随着敌意攻击的发展，例如模型窃取攻击，GNN对模型认证提出了挑战。为了避免对GNN的版权侵权，有必要核实GNN模型的所有权。本文提出了一种适用于图和节点分类任务的GNN水印框架。我们设计了两种策略来为图分类任务和节点分类任务生成水印数据，2)通过训练将水印嵌入到宿主模型中，得到带水印的GNN模型，3)在黑盒环境下验证可疑模型的所有权。实验表明，我们的框架能够以很高的概率(高达99美元)验证这两个任务的GNN模型的所有权。最后，我们的实验表明，我们的水印方法对于一种最先进的模型提取技术和四种最先进的后门攻击防御方法是健壮的。



## **32. Physical-World Optical Adversarial Attacks on 3D Face Recognition**

3D人脸识别的物理世界光学对抗性攻击 cs.CV

Submitted to CVPR 2023

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2205.13412v3) [paper-pdf](http://arxiv.org/pdf/2205.13412v3)

**Authors**: Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao

**Abstract**: 2D face recognition has been proven insecure for physical adversarial attacks. However, few studies have investigated the possibility of attacking real-world 3D face recognition systems. 3D-printed attacks recently proposed cannot generate adversarial points in the air. In this paper, we attack 3D face recognition systems through elaborate optical noises. We took structured light 3D scanners as our attack target. End-to-end attack algorithms are designed to generate adversarial illumination for 3D faces through the inherent or an additional projector to produce adversarial points at arbitrary positions. Nevertheless, face reflectance is a complex procedure because the skin is translucent. To involve this projection-and-capture procedure in optimization loops, we model it by Lambertian rendering model and use SfSNet to estimate the albedo. Moreover, to improve the resistance to distance and angle changes while maintaining the perturbation unnoticeable, a 3D transform invariant loss and two kinds of sensitivity maps are introduced. Experiments are conducted in both simulated and physical worlds. We successfully attacked point-cloud-based and depth-image-based 3D face recognition algorithms while needing fewer perturbations than previous state-of-the-art physical-world 3D adversarial attacks.

摘要: 2D人脸识别已被证明对于物理对手攻击是不安全的。然而，很少有研究调查攻击真实世界3D人脸识别系统的可能性。最近提出的3D打印攻击不能在空中产生敌方点数。在本文中，我们通过复杂的光学噪声攻击3D人脸识别系统。我们以结构光3D扫描仪为攻击目标。端到端攻击算法被设计为通过固有的或附加的投影仪为3D人脸生成对抗性光照，以在任意位置产生对抗性点。然而，面部反射是一个复杂的过程，因为皮肤是半透明的。为了将这种投影和捕获过程引入优化循环，我们用Lambertian渲染模型对其进行建模，并使用SfSNet估计反照率。此外，为了在保持摄动不明显的同时提高对距离和角度变化的抵抗能力，引入了一种3D变换不变损失和两种灵敏度图。实验既在模拟世界中进行，也在物理世界中进行。我们成功地攻击了基于点云和基于深度图像的3D人脸识别算法，并且比以前最先进的物理世界3D对抗性攻击需要更少的扰动。



## **33. Adversarial Attacks and Defenses in Physiological Computing: A Systematic Review**

生理计算中的对抗性攻击与防御：系统综述 cs.LG

National Science Open, 2022

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2102.02729v4) [paper-pdf](http://arxiv.org/pdf/2102.02729v4)

**Authors**: Dongrui Wu, Jiaxin Xu, Weili Fang, Yi Zhang, Liuqing Yang, Xiaodong Xu, Hanbin Luo, Xiang Yu

**Abstract**: Physiological computing uses human physiological data as system inputs in real time. It includes, or significantly overlaps with, brain-computer interfaces, affective computing, adaptive automation, health informatics, and physiological signal based biometrics. Physiological computing increases the communication bandwidth from the user to the computer, but is also subject to various types of adversarial attacks, in which the attacker deliberately manipulates the training and/or test examples to hijack the machine learning algorithm output, leading to possible user confusion, frustration, injury, or even death. However, the vulnerability of physiological computing systems has not been paid enough attention to, and there does not exist a comprehensive review on adversarial attacks to them. This paper fills this gap, by providing a systematic review on the main research areas of physiological computing, different types of adversarial attacks and their applications to physiological computing, and the corresponding defense strategies. We hope this review will attract more research interests on the vulnerability of physiological computing systems, and more importantly, defense strategies to make them more secure.

摘要: 生理计算使用人体生理数据作为系统的实时输入。它包括脑机接口、情感计算、自适应自动化、健康信息学和基于生理信号的生物测定，或与之显著重叠。生理计算增加了从用户到计算机的通信带宽，但也容易受到各种类型的对抗性攻击，攻击者故意操纵训练和/或测试用例来劫持机器学习算法的输出，导致可能的用户困惑、沮丧、受伤，甚至死亡。然而，生理计算系统的脆弱性还没有得到足够的重视，对它们的对抗性攻击也没有一个全面的综述。本文对生理计算的主要研究领域、不同类型的对抗性攻击及其在生理计算中的应用以及相应的防御策略进行了系统的综述，填补了这一空白。我们希望这篇综述将吸引更多关于生理计算系统脆弱性的研究兴趣，更重要的是，为了使它们更安全，防御策略将引起更多的兴趣。



## **34. TrojViT: Trojan Insertion in Vision Transformers**

TrojViT：视觉变形金刚中的特洛伊木马插入 cs.LG

10 pages, 4 figures, 10 tables

**SubmitDate**: 2022-11-13    [abs](http://arxiv.org/abs/2208.13049v2) [paper-pdf](http://arxiv.org/pdf/2208.13049v2)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstract**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.

摘要: 视觉变形金刚(VITS)在各种与视觉相关的任务中展示了最先进的性能。VITS的成功促使对手对VITS进行后门攻击。虽然传统的CNN对后门攻击的脆弱性是众所周知的，但对VITS的后门攻击很少被研究。与通过卷积获取像素级局部特征的CNN相比，VITS通过块和关注点来提取全局上下文信息。将CNN特定的后门攻击活生生地移植到VITS只会产生低的干净数据准确性和低的攻击成功率。在本文中，我们提出了一种隐形和实用的特定于VIT的后门攻击$TrojViT$。与CNN特定后门攻击使用的区域触发不同，TrojViT生成修补程序触发，旨在通过修补程序显著程度排名和注意力目标丢失来构建由存储在DRAM内存中的VIT参数上的一些易受攻击位组成的特洛伊木马程序。TrojViT进一步使用最小调整的参数更新来减少特洛伊木马的比特数。一旦攻击者通过翻转易受攻击的比特将特洛伊木马程序插入到VIT模型中，VIT模型仍然会使用良性输入产生正常的推理准确性。但是，当攻击者将触发器嵌入到输入中时，VIT模型被迫将输入分类到预定义的目标类。我们表明，只需使用著名的RowHammer在VIT模型上翻转TrojViT识别的少数易受攻击的位，就可以将该模型转换为后置模型。我们在不同的VIT模型上对多个数据集进行了广泛的实验。TrojViT可以通过在ImageNet的VIT上翻转$345$比特，将$99.64\$测试图像分类到目标类别。



## **35. SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks**

SoftHebb：无监督Hebbian软赢家通吃网络中的贝叶斯推理 cs.LG

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2107.05747v4) [paper-pdf](http://arxiv.org/pdf/2107.05747v4)

**Authors**: Timoleon Moraitis, Dmitry Toichkin, Adrien Journé, Yansong Chua, Qinghai Guo

**Abstract**: Hebbian plasticity in winner-take-all (WTA) networks is highly attractive for neuromorphic on-chip learning, owing to its efficient, local, unsupervised, and on-line nature. Moreover, its biological plausibility may help overcome important limitations of artificial algorithms, such as their susceptibility to adversarial attacks, and their high demands for training-example quantity and repetition. However, Hebbian WTA learning has found little use in machine learning (ML), likely because it has been missing an optimization theory compatible with deep learning (DL). Here we show rigorously that WTA networks constructed by standard DL elements, combined with a Hebbian-like plasticity that we derive, maintain a Bayesian generative model of the data. Importantly, without any supervision, our algorithm, SoftHebb, minimizes cross-entropy, i.e. a common loss function in supervised DL. We show this theoretically and in practice. The key is a "soft" WTA where there is no absolute "hard" winner neuron. Strikingly, in shallow-network comparisons with backpropagation (BP), SoftHebb shows advantages beyond its Hebbian efficiency. Namely, it converges in fewer iterations, and is significantly more robust to noise and adversarial attacks. Notably, attacks that maximally confuse SoftHebb are also confusing to the human eye, potentially linking human perceptual robustness, with Hebbian WTA circuits of cortex. Finally, SoftHebb can generate synthetic objects as interpolations of real object classes. All in all, Hebbian efficiency, theoretical underpinning, cross-entropy-minimization, and surprising empirical advantages, suggest that SoftHebb may inspire highly neuromorphic and radically different, but practical and advantageous learning algorithms and hardware accelerators.

摘要: Winner-Take-All(WTA)网络中的Hebbian可塑性由于其高效、局部、无监督和在线的性质，对神经形态芯片上学习具有极大的吸引力。此外，它在生物学上的合理性可能有助于克服人工算法的重要局限性，例如它们对对抗性攻击的敏感性，以及它们对训练样本量和重复性的高要求。然而，Hebbian WTA学习在机器学习(ML)中几乎没有发现，可能是因为它一直缺少与深度学习(DL)兼容的优化理论。在这里，我们严格地证明了由标准的DL元素构造的WTA网络，与我们推导的Hebbian类可塑性相结合，维持了数据的贝叶斯生成模型。重要的是，在没有任何监督的情况下，我们的算法SoftHebb最小化了交叉熵，即有监督DL中的一个常见损失函数。我们从理论和实践上证明了这一点。关键是没有绝对的“硬”赢家神经元的“软”WTA。值得注意的是，在浅层网络与反向传播(BP)的比较中，SoftHebb显示出了比Hebbian效率更高的优势。也就是说，它在更少的迭代中收敛，并且对噪声和对手攻击具有明显的健壮性。值得注意的是，最大限度地混淆SoftHebb的攻击也会混淆人眼，潜在地将人类感知的健壮性与大脑皮质的Hebbian WTA回路联系起来。最后，SoftHebb可以生成合成对象作为真实对象类的内插。总而言之，Hebbian效率、理论基础、交叉熵最小化和令人惊讶的经验优势表明，SoftHebb可能会激发高度神经形态和根本不同的、但实用和有利的学习算法和硬件加速器。



## **36. Practical No-box Adversarial Attacks with Training-free Hybrid Image Transformation**

一种实用的免训练混合图像变换的无盒对抗性攻击 cs.CV

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2203.04607v2) [paper-pdf](http://arxiv.org/pdf/2203.04607v2)

**Authors**: Qilong Zhang, Chaoning Zhang, Chaoqun Li, Jingkuan Song, Lianli Gao

**Abstract**: In recent years, the adversarial vulnerability of deep neural networks (DNNs) has raised increasing attention. Among all the threat models, no-box attacks are the most practical but extremely challenging since they neither rely on any knowledge of the target model or similar substitute model, nor access the dataset for training a new substitute model. Although a recent method has attempted such an attack in a loose sense, its performance is not good enough and computational overhead of training is expensive. In this paper, we move a step forward and show the existence of a \textbf{training-free} adversarial perturbation under the no-box threat model, which can be successfully used to attack different DNNs in real-time. Motivated by our observation that high-frequency component (HFC) domains in low-level features and plays a crucial role in classification, we attack an image mainly by manipulating its frequency components. Specifically, the perturbation is manipulated by suppression of the original HFC and adding of noisy HFC. We empirically and experimentally analyze the requirements of effective noisy HFC and show that it should be regionally homogeneous, repeating and dense. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our proposed no-box method. It attacks ten well-known models with a success rate of \textbf{98.13\%} on average, which outperforms state-of-the-art no-box attacks by \textbf{29.39\%}. Furthermore, our method is even competitive to mainstream transfer-based black-box attacks.

摘要: 近年来，深度神经网络(DNN)的攻击脆弱性引起了越来越多的关注。在所有的威胁模型中，非盒子攻击是最实用但极具挑战性的攻击，因为它们既不依赖于任何目标模型或类似的替代模型的任何知识，也不需要访问数据集来训练新的替代模型。虽然最近的一种方法在松散意义上尝试了这种攻击，但其性能不够好，并且训练的计算开销很高。在这篇文章中，我们进一步证明了在非盒子威胁模型下存在一个对抗性扰动，它可以成功地用来实时攻击不同的DNN。由于我们观察到高频分量(HFC)域位于低层特征并且在分类中起着关键作用，我们主要通过操纵其频率分量来攻击图像。具体地说，通过抑制原始HFC和添加噪声HFC来操纵扰动。我们从经验和实验上分析了有效的噪声HFC的要求，表明它应该是区域均匀的、重复的和密集的。在ImageNet数据集上的大量实验证明了我们提出的非盒子方法的有效性。它攻击十个著名的模型，平均成功率为\extbf{98.13\%}，比最先进的非盒子攻击的\extbf{29.39\%}要好。此外，我们的方法甚至可以与主流的基于传输的黑盒攻击相竞争。



## **37. Generating Textual Adversaries with Minimal Perturbation**

以最小的扰动生成文本对手 cs.CL

To appear in EMNLP Findings 2022. The code is available at  https://github.com/xingyizhao/TAMPERS

**SubmitDate**: 2022-11-12    [abs](http://arxiv.org/abs/2211.06571v1) [paper-pdf](http://arxiv.org/pdf/2211.06571v1)

**Authors**: Xingyi Zhao, Lu Zhang, Depeng Xu, Shuhan Yuan

**Abstract**: Many word-level adversarial attack approaches for textual data have been proposed in recent studies. However, due to the massive search space consisting of combinations of candidate words, the existing approaches face the problem of preserving the semantics of texts when crafting adversarial counterparts. In this paper, we develop a novel attack strategy to find adversarial texts with high similarity to the original texts while introducing minimal perturbation. The rationale is that we expect the adversarial texts with small perturbation can better preserve the semantic meaning of original texts. Experiments show that, compared with state-of-the-art attack approaches, our approach achieves higher success rates and lower perturbation rates in four benchmark datasets.

摘要: 最近的研究提出了许多针对文本数据的词级对抗性攻击方法。然而，由于候选词的组合构成了巨大的搜索空间，现有的方法在构建对抗性对应词时面临着保留文本语义的问题。在本文中，我们提出了一种新的攻击策略，在引入最小扰动的同时，找到与原始文本具有较高相似度的对抗性文本。其基本原理是，我们期望经过微小扰动的对抗性文本能够更好地保留原始文本的语义。实验表明，与最新的攻击方法相比，我们的方法在四个基准数据集上获得了更高的成功率和更低的扰动率。



## **38. An investigation of security controls and MITRE ATT\&CK techniques**

安全控制与MITRE ATT-CK技术的研究 cs.CR

**SubmitDate**: 2022-11-11    [abs](http://arxiv.org/abs/2211.06500v1) [paper-pdf](http://arxiv.org/pdf/2211.06500v1)

**Authors**: Md Rayhanur Rahman, Laurie Williams

**Abstract**: Attackers utilize a plethora of adversarial techniques in cyberattacks to compromise the confidentiality, integrity, and availability of the target organizations and systems. Information security standards such as NIST, ISO/IEC specify hundreds of security controls that organizations can enforce to protect and defend the information systems from adversarial techniques. However, implementing all the available controls at the same time can be infeasible and security controls need to be investigated in terms of their mitigation ability over adversarial techniques used in cyberattacks as well. The goal of this research is to aid organizations in making informed choices on security controls to defend against cyberthreats through an investigation of adversarial techniques used in current cyberattacks. In this study, we investigated the extent of mitigation of 298 NIST SP800-53 controls over 188 adversarial techniques used in 669 cybercrime groups and malware cataloged in the MITRE ATT\&CK framework based upon an existing mapping between the controls and techniques. We identify that, based on the mapping, only 101 out of 298 control are capable of mitigating adversarial techniques. However, we also identify that 53 adversarial techniques cannot be mitigated by any existing controls, and these techniques primarily aid adversaries in bypassing system defense and discovering targeted system information. We identify a set of 20 critical controls that can mitigate 134 adversarial techniques, and on average, can mitigate 72\% of all techniques used by 98\% of the cataloged adversaries in MITRE ATT\&CK. We urge organizations, that do not have any controls enforced in place, to implement the top controls identified in the study.

摘要: 攻击者在网络攻击中利用过多的对抗性技术来危害目标组织和系统的机密性、完整性和可用性。NIST、ISO/IEC等信息安全标准规定了数百种安全控制，组织可以实施这些控制来保护和保护信息系统免受恶意技术的攻击。然而，同时实施所有可用的控制措施可能是不可行的，需要调查安全控制措施对网络攻击中使用的敌对技术的缓解能力。这项研究的目标是通过调查当前网络攻击中使用的对抗性技术，帮助组织在安全控制方面做出明智的选择，以防御网络威胁。在这项研究中，我们调查了298个NIST SP800-53控件对669个网络犯罪组织中使用的188种对抗性技术和MITRE ATT\&CK框架中编目的恶意软件的缓解程度，这些控件和技术之间的现有映射。我们发现，基于映射，在298个控制中只有101个能够减轻对抗性技术。然而，我们也发现，53种对抗性技术不能通过任何现有的控制来缓解，这些技术主要帮助对手绕过系统防御并发现目标系统信息。我们确定了一组20种关键控制措施，可以减少134种对抗技术，平均而言，可以减少MITRE ATT-CK中98个对手所使用的所有技术的72%。我们敦促没有实施任何控制措施的组织实施研究中确定的顶级控制措施。



## **39. Blockchain Technology to Secure Bluetooth**

区块链技术保护蓝牙安全 cs.CR

7 pages, 6 figures

**SubmitDate**: 2022-11-11    [abs](http://arxiv.org/abs/2211.06451v1) [paper-pdf](http://arxiv.org/pdf/2211.06451v1)

**Authors**: Athanasios Kalogiratos, Ioanna Kantzavelou

**Abstract**: Bluetooth is a communication technology used to wirelessly exchange data between devices. In the last few years there have been found a great number of security vulnerabilities, and adversaries are taking advantage of them causing harm and significant loss. Numerous system security updates have been approved and installed in order to sort out security holes and bugs, and prevent attacks that could expose personal or other valuable information. But those updates are not sufficient and appropriate and new bugs keep showing up. In Bluetooth technology, pairing is identified as the step where most bugs are found and most attacks target this particular process part of Bluetooth. A new technology that has been proved bulletproof when it comes to security and the exchange of sensitive information is Blockchain. Blockchain technology is promising to be incorporated well in a network of smart devices, and secure an Internet of Things (IoT), where Bluetooth technology is being extensively used. This work presents a vulnerability discovered in Bluetooth pairing process, and proposes a Blockchain solution approach to secure pairing and mitigate this vulnerability. The paper first introduces the Bluetooth technology and delves into how Blockchain technology can be a solution to certain security problems. Then a solution approach shows how Blockchain can be integrated and implemented to ensure the required level of security. Certain attack incidents on Bluetooth vulnerable points are examined and discussion and conclusions give the extension of the security related problems.

摘要: 蓝牙是一种用于在设备之间无线交换数据的通信技术。在过去的几年里，发现了大量的安全漏洞，攻击者正在利用这些漏洞造成危害和重大损失。已经批准和安装了许多系统安全更新，以找出安全漏洞和错误，并防止可能暴露个人或其他有价值信息的攻击。但这些更新是不够和适当的，新的错误不断出现。在蓝牙技术中，配对被认为是发现大多数错误的步骤，并且大多数攻击都针对蓝牙的这一特定进程部分。在安全和敏感信息交换方面，一项已被证明是防弹的新技术是区块链。区块链技术有望很好地融入智能设备网络，并确保物联网(IoT)的安全，蓝牙技术正在物联网中得到广泛应用。本文介绍了蓝牙配对过程中发现的一个漏洞，并提出了一种区块链解决方案来保护配对并缓解该漏洞。本文首先介绍了蓝牙技术，并深入探讨了区块链技术是如何解决某些安全问题的。然后，一个解决方案方法展示了如何集成和实施区块链，以确保所需的安全级别。对蓝牙易受攻击点的攻击事件进行了分析，并对相关的安全问题进行了讨论和结论。



## **40. Test-time adversarial detection and robustness for localizing humans using ultra wide band channel impulse responses**

使用超宽带信道脉冲响应定位人类的测试时间敌意检测和稳健性 cs.LG

5 pages, 4 figures, ICASSP Conference

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05854v1) [paper-pdf](http://arxiv.org/pdf/2211.05854v1)

**Authors**: Abhiram Kolli, Muhammad Jehanzeb Mirza, Horst Possegger, Horst Bischof

**Abstract**: Keyless entry systems in cars are adopting neural networks for localizing its operators. Using test-time adversarial defences equip such systems with the ability to defend against adversarial attacks without prior training on adversarial samples. We propose a test-time adversarial example detector which detects the input adversarial example through quantifying the localized intermediate responses of a pre-trained neural network and confidence scores of an auxiliary softmax layer. Furthermore, in order to make the network robust, we extenuate the non-relevant features by non-iterative input sample clipping. Using our approach, mean performance over 15 levels of adversarial perturbations is increased by 55.33% for the fast gradient sign method (FGSM) and 6.3% for both the basic iterative method (BIM) and the projected gradient method (PGD).

摘要: 汽车的无钥匙进入系统正在采用神经网络来定位其操作员。利用测试时间的对抗性防御，使这些系统具备防御对抗性攻击的能力，而无需事先对对抗性样本进行培训。我们提出了一种测试时间敌意实例检测器，它通过量化预先训练的神经网络的局部化中间响应和辅助Softmax层的置信度分数来检测输入的敌意实例。此外，为了使网络具有健壮性，我们通过非迭代的输入样本裁剪来消除不相关的特征。使用我们的方法，在15个对抗性扰动级别上，快速梯度符号方法(FGSM)的平均性能提高了55.33%，基本迭代方法(BIM)和投影梯度方法(PGD)的平均性能都提高了6.3%。



## **41. A Practical Introduction to Side-Channel Extraction of Deep Neural Network Parameters**

一种实用的旁通道深度神经网络参数提取方法 cs.CR

Accepted at Smart Card Research and Advanced Application Conference  (CARDIS 2022)

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05590v1) [paper-pdf](http://arxiv.org/pdf/2211.05590v1)

**Authors**: Raphael Joud, Pierre-Alain Moellic, Simon Pontie, Jean-Baptiste Rigaud

**Abstract**: Model extraction is a major threat for embedded deep neural network models that leverages an extended attack surface. Indeed, by physically accessing a device, an adversary may exploit side-channel leakages to extract critical information of a model (i.e., its architecture or internal parameters). Different adversarial objectives are possible including a fidelity-based scenario where the architecture and parameters are precisely extracted (model cloning). We focus this work on software implementation of deep neural networks embedded in a high-end 32-bit microcontroller (Cortex-M7) and expose several challenges related to fidelity-based parameters extraction through side-channel analysis, from the basic multiplication operation to the feed-forward connection through the layers. To precisely extract the value of parameters represented in the single-precision floating point IEEE-754 standard, we propose an iterative process that is evaluated with both simulations and traces from a Cortex-M7 target. To our knowledge, this work is the first to target such an high-end 32-bit platform. Importantly, we raise and discuss the remaining challenges for the complete extraction of a deep neural network model, more particularly the critical case of biases.

摘要: 模型提取是利用扩展攻击面的嵌入式深度神经网络模型的主要威胁。事实上，通过物理访问设备，攻击者可以利用侧通道泄漏来提取模型的关键信息(即其体系结构或内部参数)。不同的对抗性目标是可能的，包括基于保真度的场景，其中精确地提取了体系结构和参数(模型克隆)。我们的工作重点是在高端32位微控制器(Cortex-M7)中嵌入深度神经网络的软件实现，并揭示了从基本的乘法运算到通过各层的前馈连接，通过侧通道分析提取基于保真度的参数的几个挑战。为了精确提取单精度浮点IEEE-754标准中表示的参数值，我们提出了一种迭代过程，并通过仿真和Cortex-M7目标的跟踪进行了评估。据我们所知，这项工作是首次瞄准如此高端的32位平台。重要的是，我们提出并讨论了完整提取深度神经网络模型的剩余挑战，尤其是在偏差的关键情况下。



## **42. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

对抗性训练对语言模型稳健性和泛化能力的影响 cs.CL

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05523v1) [paper-pdf](http://arxiv.org/pdf/2211.05523v1)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of BERT-like language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveal that the improved generalization is due to `more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.

摘要: 对抗性训练被广泛认为是对抗对抗性攻击的最有效的防御方法。然而，众所周知，在对抗性训练的模型中实现稳健性和泛化都需要权衡。这项工作的目标是深入比较语言模型中对抗性训练的不同方法。具体地说，我们研究了训练前数据扩充以及训练时间输入扰动与嵌入空间扰动对类BERT语言模型的鲁棒性和泛化的影响。我们的发现表明，通过训练前数据增强或通过输入空间扰动训练可以获得更好的稳健性。然而，嵌入空间扰动的训练显著提高了泛化能力。对学习模型的神经元进行的语言相关性分析表明，改进的泛化是由于“更专门的”神经元。据我们所知，这是第一次对语言模型对抗性训练中生成对抗性实例的不同方法进行深入的定性分析。



## **43. On the Privacy Risks of Algorithmic Recourse**

论算法追索权的隐私风险 cs.LG

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05427v1) [paper-pdf](http://arxiv.org/pdf/2211.05427v1)

**Authors**: Martin Pawelczyk, Himabindu Lakkaraju, Seth Neel

**Abstract**: As predictive models are increasingly being employed to make consequential decisions, there is a growing emphasis on developing techniques that can provide algorithmic recourse to affected individuals. While such recourses can be immensely beneficial to affected individuals, potential adversaries could also exploit these recourses to compromise privacy. In this work, we make the first attempt at investigating if and how an adversary can leverage recourses to infer private information about the underlying model's training data. To this end, we propose a series of novel membership inference attacks which leverage algorithmic recourse. More specifically, we extend the prior literature on membership inference attacks to the recourse setting by leveraging the distances between data instances and their corresponding counterfactuals output by state-of-the-art recourse methods. Extensive experimentation with real world and synthetic datasets demonstrates significant privacy leakage through recourses. Our work establishes unintended privacy leakage as an important risk in the widespread adoption of recourse methods.

摘要: 随着预测模型越来越多地被用来做出相应的决策，人们越来越重视开发能够为受影响的个人提供算法追索的技术。虽然这些资源对受影响的个人可能是非常有益的，但潜在的对手也可能利用这些资源来损害隐私。在这项工作中，我们第一次尝试调查对手是否以及如何利用资源来推断有关底层模型的训练数据的私人信息。为此，我们提出了一系列利用算法资源的新型成员推理攻击。更具体地说，我们通过利用数据实例与其由最先进的求助方法输出的对应反事实之间的距离，将先前关于成员关系推理攻击的文献扩展到求助设置。对真实世界和合成数据集的广泛实验表明，资源中存在严重的隐私泄露。我们的工作确立了意外的隐私泄露是广泛采用追索权方法的一个重要风险。



## **44. Stay Home Safe with Starving Federated Data**

联邦数据匮乏，足不出户 cs.LG

11 pages, 12 figures, 7 tables, accepted as a conference paper at  IEEE UV 2022, Boston, USA

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2211.05410v1) [paper-pdf](http://arxiv.org/pdf/2211.05410v1)

**Authors**: Jaechul Roh, Yajun Fang

**Abstract**: Over the past few years, the field of adversarial attack received numerous attention from various researchers with the help of successful attack success rate against well-known deep neural networks that were acknowledged to achieve high classification ability in various tasks. However, majority of the experiments were completed under a single model, which we believe it may not be an ideal case in a real-life situation. In this paper, we introduce a novel federated adversarial training method for smart home face recognition, named FLATS, where we observed some interesting findings that may not be easily noticed in a traditional adversarial attack to federated learning experiments. By applying different variations to the hyperparameters, we have spotted that our method can make the global model to be robust given a starving federated environment. Our code can be found on https://github.com/jcroh0508/FLATS.

摘要: 在过去的几年里，借助对公认在各种任务中具有高分类能力的知名深度神经网络的攻击成功率，对抗性攻击领域受到了众多研究人员的关注。然而，大多数实验都是在单一模型下完成的，我们认为这在现实生活中可能不是理想的情况。本文介绍了一种新的用于智能家居人脸识别的联合对抗性训练方法--Flats，我们在该方法中观察到了一些有趣的发现，这些发现在传统的对抗性攻击联合学习实验中可能不容易被注意到。通过对超参数应用不同的变化，我们已经发现，我们的方法可以使全局模型在饥饿的联邦环境下具有健壮性。我们的代码可以在https://github.com/jcroh0508/FLATS.上找到



## **45. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

30 pages, 7 figures, NeurIPS camera-ready

**SubmitDate**: 2022-11-10    [abs](http://arxiv.org/abs/2205.01663v5) [paper-pdf](http://arxiv.org/pdf/2205.01663v5)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstract**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a safe language generation task (``avoid injuries'') as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. We found that adversarial training increased robustness to the adversarial attacks that we trained on -- doubling the time for our contractors to find adversarial examples both with our tool (from 13 to 26 minutes) and without (from 20 to 44 minutes) -- without affecting in-distribution performance.   We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用了一个安全的语言生成任务(`避免受伤‘)作为通过对抗性训练获得高可靠性的试验床。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们的任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。我们发现对抗性训练增加了对我们训练的对抗性攻击的健壮性--使我们的承包商在使用我们的工具(从13分钟到26分钟)和不使用我们的工具(从20分钟到44分钟)的情况下找到对抗性例子的时间翻了一番--而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **46. Using Deception in Markov Game to Understand Adversarial Behaviors through a Capture-The-Flag Environment**

利用马尔可夫博弈中的欺骗来理解捕获旗帜环境中的敌方行为 cs.GT

Accepted at GameSec 2022

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.15011v2) [paper-pdf](http://arxiv.org/pdf/2210.15011v2)

**Authors**: Siddhant Bhambri, Purv Chauhan, Frederico Araujo, Adam Doupé, Subbarao Kambhampati

**Abstract**: Identifying the actual adversarial threat against a system vulnerability has been a long-standing challenge for cybersecurity research. To determine an optimal strategy for the defender, game-theoretic based decision models have been widely used to simulate the real-world attacker-defender scenarios while taking the defender's constraints into consideration. In this work, we focus on understanding human attacker behaviors in order to optimize the defender's strategy. To achieve this goal, we model attacker-defender engagements as Markov Games and search for their Bayesian Stackelberg Equilibrium. We validate our modeling approach and report our empirical findings using a Capture-The-Flag (CTF) setup, and we conduct user studies on adversaries with varying skill-levels. Our studies show that application-level deceptions are an optimal mitigation strategy against targeted attacks -- outperforming classic cyber-defensive maneuvers, such as patching or blocking network requests. We use this result to further hypothesize over the attacker's behaviors when trapped in an embedded honeypot environment and present a detailed analysis of the same.

摘要: 识别针对系统漏洞的实际对手威胁一直是网络安全研究的长期挑战。为了确定防御者的最优策略，基于博弈论的决策模型被广泛用于模拟现实世界中的攻防场景，同时考虑了防御者的约束。在这项工作中，我们重点了解人类攻击者的行为，以便优化防御者的策略。为了实现这一目标，我们将攻防双方的交战建模为马尔可夫博弈，并寻找他们的贝叶斯Stackelberg均衡。我们验证了我们的建模方法，并使用捕获旗帜(CTF)设置报告了我们的经验结果，并对具有不同技能水平的对手进行了用户研究。我们的研究表明，应用程序级别的欺骗是针对目标攻击的最佳缓解策略--性能优于修补或阻止网络请求等传统的网络防御策略。我们利用这一结果进一步假设攻击者在被困在嵌入式蜜罐环境中时的行为，并对此进行了详细的分析。



## **47. Are All Edges Necessary? A Unified Framework for Graph Purification**

所有的边都是必要的吗？一种统一的图净化框架 cs.SI

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.05184v1) [paper-pdf](http://arxiv.org/pdf/2211.05184v1)

**Authors**: Zishan Gu, Jintang Li, Liang Chen

**Abstract**: Graph Neural Networks (GNNs) as deep learning models working on graph-structure data have achieved advanced performance in many works. However, it has been proved repeatedly that, not all edges in a graph are necessary for the training of machine learning models. In other words, some of the connections between nodes may bring redundant or even misleading information to downstream tasks. In this paper, we try to provide a method to drop edges in order to purify the graph data from a new perspective. Specifically, it is a framework to purify graphs with the least loss of information, under which the core problems are how to better evaluate the edges and how to delete the relatively redundant edges with the least loss of information. To address the above two problems, we propose several measurements for the evaluation and different judges and filters for the edge deletion. We also introduce a residual-iteration strategy and a surrogate model for measurements requiring unknown information. The experimental results show that our proposed measurements for KL divergence with constraints to maintain the connectivity of the graph and delete edges in an iterative way can find out the most edges while keeping the performance of GNNs. What's more, further experiments show that this method also achieves the best defense performance against adversarial attacks.

摘要: 图神经网络作为一种处理图结构数据的深度学习模型，在许多工作中取得了很好的性能。然而，已经被反复证明，并非图中的所有边都是机器学习模型训练所必需的。换句话说，节点之间的一些连接可能会给下游任务带来冗余甚至误导的信息。本文试图从一个新的角度提供一种边删除的方法，以达到对图形数据进行净化的目的。具体地说，它是一个信息损失最小的图净化框架，其核心问题是如何更好地评价边以及如何在信息损失最小的情况下删除相对冗余的边。针对上述两个问题，我们提出了几种评价方法和不同的边缘删除判断和滤波方法。对于需要未知信息的测量，我们还引入了残差迭代策略和代理模型。实验结果表明，在保持图的连通性和以迭代方式删除边的约束条件下，我们提出的KL发散度度量方法能够在保持GNN性能的同时找到最多的边。进一步的实验表明，该方法也取得了最好的对抗攻击防御性能。



## **48. Accountable and Explainable Methods for Complex Reasoning over Text**

基于文本的复杂推理的可靠和可解释方法 cs.LG

PhD Thesis

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04946v1) [paper-pdf](http://arxiv.org/pdf/2211.04946v1)

**Authors**: Pepa Atanasova

**Abstract**: A major concern of Machine Learning (ML) models is their opacity. They are deployed in an increasing number of applications where they often operate as black boxes that do not provide explanations for their predictions. Among others, the potential harms associated with the lack of understanding of the models' rationales include privacy violations, adversarial manipulations, and unfair discrimination. As a result, the accountability and transparency of ML models have been posed as critical desiderata by works in policy and law, philosophy, and computer science.   In computer science, the decision-making process of ML models has been studied by developing accountability and transparency methods. Accountability methods, such as adversarial attacks and diagnostic datasets, expose vulnerabilities of ML models that could lead to malicious manipulations or systematic faults in their predictions. Transparency methods explain the rationales behind models' predictions gaining the trust of relevant stakeholders and potentially uncovering mistakes and unfairness in models' decisions. To this end, transparency methods have to meet accountability requirements as well, e.g., being robust and faithful to the underlying rationales of a model.   This thesis presents my research that expands our collective knowledge in the areas of accountability and transparency of ML models developed for complex reasoning tasks over text.

摘要: 机器学习(ML)模型的一个主要问题是其不透明性。它们被部署在越来越多的应用程序中，在这些应用程序中，它们通常作为黑匣子运行，不能为他们的预测提供解释。其中，与缺乏对模型原理的理解相关的潜在危害包括侵犯隐私、敌意操纵和不公平歧视。因此，ML模型的问责制和透明度已被政策和法律、哲学和计算机科学领域的著作视为迫切需要。在计算机科学中，人们通过发展问责和透明方法来研究ML模型的决策过程。诸如对抗性攻击和诊断数据集等问责方法暴露了ML模型的漏洞，这些漏洞可能导致恶意操纵或预测中的系统性错误。透明度方法解释了模型预测背后的原理，赢得了相关利益相关者的信任，并潜在地揭示了模型决策中的错误和不公平。为此目的，透明度方法还必须满足问责制要求，例如，稳健并忠实于模式的基本理由。这篇论文介绍了我的研究，旨在扩大我们在ML模型的责任和透明度领域的集体知识，这些模型是为复杂的文本推理任务开发的。



## **49. Lipschitz Continuous Algorithms for Graph Problems**

图问题的Lipschitz连续算法 cs.DS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2211.04674v1) [paper-pdf](http://arxiv.org/pdf/2211.04674v1)

**Authors**: Soh Kumabe, Yuichi Yoshida

**Abstract**: It has been widely observed in the machine learning community that a small perturbation to the input can cause a large change in the prediction of a trained model, and such phenomena have been intensively studied in the machine learning community under the name of adversarial attacks. Because graph algorithms also are widely used for decision making and knowledge discovery, it is important to design graph algorithms that are robust against adversarial attacks. In this study, we consider the Lipschitz continuity of algorithms as a robustness measure and initiate a systematic study of the Lipschitz continuity of algorithms for (weighted) graph problems.   Depending on how we embed the output solution to a metric space, we can think of several Lipschitzness notions. We mainly consider the one that is invariant under scaling of weights, and we provide Lipschitz continuous algorithms and lower bounds for the minimum spanning tree problem, the shortest path problem, and the maximum weight matching problem. In particular, our shortest path algorithm is obtained by first designing an algorithm for unweighted graphs that are robust against edge contractions and then applying it to the unweighted graph constructed from the original weighted graph.   Then, we consider another Lipschitzness notion induced by a natural mapping that maps the output solution to its characteristic vector. It turns out that no Lipschitz continuous algorithm exists for this Lipschitz notion, and we instead design algorithms with bounded pointwise Lipschitz constants for the minimum spanning tree problem and the maximum weight bipartite matching problem. Our algorithm for the latter problem is based on an LP relaxation with entropy regularization.

摘要: 机器学习界已经广泛观察到，输入的微小扰动会导致训练模型的预测发生很大变化，这种现象已经在机器学习界以对抗攻击的名义进行了深入的研究。由于图算法也被广泛用于决策和知识发现，因此设计对对手攻击具有健壮性的图算法是很重要的。在这项研究中，我们将算法的Lipschitz连续性作为一个稳健性度量，并开始系统地研究(加权)图问题的算法的Lipschitz连续性。根据我们将输出解嵌入到度量空间的方式，我们可以想到几个Lipschitzness概念。我们主要考虑在权值缩放下不变的问题，给出了最小生成树问题、最短路问题和最大权匹配问题的Lipschitz连续算法和下界。特别是，我们的最短路径算法是通过设计一个对边收缩具有健壮性的未加权图的算法来获得的，然后将其应用于由原始加权图构造的未加权图。然后，我们考虑由自然映射诱导的另一个Lipschitzness概念，该映射将输出解映射到其特征向量。结果表明，对于这种Lipschitz概念，不存在Lipschitz连续算法，而是针对最小生成树问题和最大权二部匹配问题设计了逐点Lipschitz常数有界的算法。对于后一个问题，我们的算法是基于带熵正则化的LP松弛算法。



## **50. FedDef: Defense Against Gradient Leakage in Federated Learning-based Network Intrusion Detection Systems**

FedDef：基于联邦学习的网络入侵检测系统的梯度泄漏防御 cs.CR

14 pages, 9 figures, submitted to TIFS

**SubmitDate**: 2022-11-09    [abs](http://arxiv.org/abs/2210.04052v2) [paper-pdf](http://arxiv.org/pdf/2210.04052v2)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Xuewei Feng, Ke Xu

**Abstract**: Deep learning (DL) methods have been widely applied to anomaly-based network intrusion detection system (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows multiple users to train a global model on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, we propose two privacy evaluation metrics designed for FL-based NIDSs, including (1) privacy score that evaluates the similarity between the original and recovered traffic features using reconstruction attacks, and (2) evasion rate against NIDSs using Generative Adversarial Network-based adversarial attack with the reconstructed benign traffic. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To defend against such attacks and build a more robust FL-based NIDS, we further propose FedDef, a novel optimization-based input perturbation defense strategy with theoretical guarantee. It achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines in terms of privacy protection with up to 7 times higher privacy score, while maintaining model accuracy loss within 3% under optimal parameter combination.

摘要: 深度学习方法已被广泛应用于基于异常的网络入侵检测系统中，以检测恶意流量。为了扩展基于DL的方法的使用场景，联邦学习(FL)框架允许多个用户在尊重个人数据隐私的基础上训练全局模型。然而，还没有系统地评估基于FL的NIDS在现有防御系统下对现有隐私攻击的健壮性。针对这一问题，我们提出了两个针对FL网络入侵检测系统的隐私评估指标，包括：(1)隐私评分，通过重构攻击评估原始流量特征和恢复流量特征之间的相似性；(2)利用重构的良性流量对基于网络的生成性对抗性攻击的逃避率。我们进行的实验表明，现有的防御措施提供的保护很少，相应的敌意流量甚至可以避开Sota NIDS Kitsune。为了防御此类攻击，构建一个更健壮的基于FL的网络入侵检测系统，我们进一步提出了一种新的基于优化的输入扰动防御策略FedDef，并提供了理论上的保证。它既通过最小化梯度距离实现了高效用，又通过最大化输入距离实现了强大的隐私保护。我们在四个数据集上对四种现有的防御措施进行了实验评估，结果表明，我们的防御措施在隐私保护方面的表现优于所有基线，隐私得分高达7倍，同时在最优参数组合下将模型精度损失保持在3%以内。



