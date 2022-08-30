# Latest Adversarial Attack Papers
**update at 2022-08-31 06:31:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Demystifying Arch-hints for Model Extraction: An Attack in Unified Memory System**

揭开模型提取拱门的神秘面纱：统一存储系统中的攻击 cs.CR

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2208.13720v1)

**Authors**: Zhendong Wang, Xiaoming Zeng, Xulong Tang, Danfeng Zhang, Xing Hu, Yang Hu

**Abstracts**: The deep neural network (DNN) models are deemed confidential due to their unique value in expensive training efforts, privacy-sensitive training data, and proprietary network characteristics. Consequently, the model value raises incentive for adversary to steal the model for profits, such as the representative model extraction attack. Emerging attack can leverage timing-sensitive architecture-level events (i.e., Arch-hints) disclosed in hardware platforms to extract DNN model layer information accurately. In this paper, we take the first step to uncover the root cause of such Arch-hints and summarize the principles to identify them. We then apply these principles to emerging Unified Memory (UM) management system and identify three new Arch-hints caused by UM's unique data movement patterns. We then develop a new extraction attack, UMProbe. We also create the first DNN benchmark suite in UM and utilize the benchmark suite to evaluate UMProbe. Our evaluation shows that UMProbe can extract the layer sequence with an accuracy of 95% for almost all victim test models, which thus calls for more attention to the DNN security in UM system.

摘要: 深度神经网络(DNN)模型被认为是机密的，因为它们在昂贵的训练工作、隐私敏感的训练数据和专有的网络特征方面具有独特的价值。因此，模型值增加了攻击者窃取模型以获取利润的动机，例如典型的模型提取攻击。新出现的攻击可以利用硬件平台中披露的计时敏感的架构级事件(即Arch-hints)来准确提取DNN模型层信息。在这篇文章中，我们首先要找出这些暗示语产生的根本原因，并总结出识别这些暗示语的原则。然后，我们将这些原则应用到新兴的统一内存(UM)管理系统中，并确定了由UM独特的数据移动模式引起的三个新的拱形提示。然后，我们开发了一种新的提取攻击UMProbe。我们还创建了UM中的第一个DNN基准测试套件，并利用该基准测试套件对UMProbe进行了评估。我们的评估表明，UMProbe对几乎所有的受害者测试模型都能以95%的准确率提取层序列，这就要求我们更多地关注UM系统中的DNN安全性。



## **2. Understanding the Limits of Poisoning Attacks in Episodic Reinforcement Learning**

在情景强化学习中理解中毒攻击的限度 cs.LG

Accepted at International Joint Conferences on Artificial  Intelligence (IJCAI) 2022

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2208.13663v1)

**Authors**: Anshuka Rangi, Haifeng Xu, Long Tran-Thanh, Massimo Franceschetti

**Abstracts**: To understand the security threats to reinforcement learning (RL) algorithms, this paper studies poisoning attacks to manipulate \emph{any} order-optimal learning algorithm towards a targeted policy in episodic RL and examines the potential damage of two natural types of poisoning attacks, i.e., the manipulation of \emph{reward} and \emph{action}. We discover that the effect of attacks crucially depend on whether the rewards are bounded or unbounded. In bounded reward settings, we show that only reward manipulation or only action manipulation cannot guarantee a successful attack. However, by combining reward and action manipulation, the adversary can manipulate any order-optimal learning algorithm to follow any targeted policy with $\tilde{\Theta}(\sqrt{T})$ total attack cost, which is order-optimal, without any knowledge of the underlying MDP. In contrast, in unbounded reward settings, we show that reward manipulation attacks are sufficient for an adversary to successfully manipulate any order-optimal learning algorithm to follow any targeted policy using $\tilde{O}(\sqrt{T})$ amount of contamination. Our results reveal useful insights about what can or cannot be achieved by poisoning attacks, and are set to spur more works on the design of robust RL algorithms.

摘要: 为了了解强化学习(RL)算法面临的安全威胁，研究了剧情强化学习(RL)中针对目标策略操纵任意阶优化学习算法的中毒攻击，并考察了两种自然类型的中毒攻击，即操纵EMPH{REWART}和EMPH{ACTION}的潜在危害.我们发现，攻击的效果关键取决于回报是有界的还是无界的。在有界奖赏设置下，我们证明了仅有奖赏操纵或仅有动作操纵不能保证攻击成功。然而，通过结合奖励和动作操纵，攻击者可以操纵任何顺序最优的学习算法来遵循任何具有$tide{\theta}(\Sqrt{T})$攻击总代价的目标策略，这是顺序最优的，而不需要知道潜在的MDP。相反，在无界奖赏环境下，我们证明了奖赏操纵攻击足以使敌手成功操纵任何阶数最优学习算法来遵循任何目标策略，并且使用$tide{O}(\sqrt{T})$污染量。我们的结果揭示了关于中毒攻击可以或不能实现什么的有用见解，并将刺激更多关于健壮RL算法设计的工作。



## **3. HAT4RD: Hierarchical Adversarial Training for Rumor Detection on Social Media**

HAT4RD：针对社交媒体谣言检测的分级对抗性训练 cs.CL

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2110.00425v2)

**Authors**: Shiwen Ni, Jiawen Li, Hung-Yu Kao

**Abstracts**: With the development of social media, social communication has changed. While this facilitates people's communication and access to information, it also provides an ideal platform for spreading rumors. In normal or critical situations, rumors will affect people's judgment and even endanger social security. However, natural language is high-dimensional and sparse, and the same rumor may be expressed in hundreds of ways on social media. As such, the robustness and generalization of the current rumor detection model are put into question. We proposed a novel \textbf{h}ierarchical \textbf{a}dversarial \textbf{t}raining method for \textbf{r}umor \textbf{d}etection (HAT4RD) on social media. Specifically, HAT4RD is based on gradient ascent by adding adversarial perturbations to the embedding layers of post-level and event-level modules to deceive the detector. At the same time, the detector uses stochastic gradient descent to minimize the adversarial risk to learn a more robust model. In this way, the post-level and event-level sample spaces are enhanced, and we have verified the robustness of our model under a variety of adversarial attacks. Moreover, visual experiments indicate that the proposed model drifts into an area with a flat loss landscape, leading to better generalization. We evaluate our proposed method on three public rumors datasets from two commonly used social platforms (Twitter and Weibo). Experiment results demonstrate that our model achieves better results than state-of-the-art methods.

摘要: 随着社交媒体的发展，社交传播也发生了变化。这在方便人们交流和获取信息的同时，也为谣言传播提供了一个理想的平台。在正常或危急的情况下，谣言会影响人们的判断，甚至危害社会治安。然而，自然语言是高维和稀疏的，同样的谣言可能会在社交媒体上以数百种方式表达。因此，当前谣言检测模型的健壮性和泛化能力受到了质疑。针对社交媒体上的文本bf{r}和文本bf{d}评论，我们提出了一种新的层次式文本bf{a}分布式文本bf{t}挖掘方法(HAT4RD)。具体地说，HAT4RD是基于梯度上升的，通过在后期和事件级模块的嵌入层添加对抗性扰动来欺骗检测器。同时，检测器使用随机梯度下降来最小化对抗风险，以学习更健壮的模型。通过这种方式，增强了后级和事件级样本空间，验证了该模型在各种对抗性攻击下的健壮性。此外，视觉实验表明，所提出的模型漂移到损失平坦的区域，具有更好的泛化能力。我们在两个常用社交平台(Twitter和微博)的三个公开谣言数据集上对我们提出的方法进行了评估。实验结果表明，我们的模型取得了比现有方法更好的结果。



## **4. Towards Both Accurate and Robust Neural Networks without Extra Data**

无额外数据的精确和稳健的神经网络 cs.CV

**SubmitDate**: 2022-08-29    [paper-pdf](http://arxiv.org/pdf/2103.13124v2)

**Authors**: Faqiang Liu, Rong Zhao

**Abstracts**: Deep neural networks have achieved remarkable performance in various applications but are extremely vulnerable to adversarial perturbation. The most representative and promising methods that can enhance model robustness, such as adversarial training and its variants, substantially degrade model accuracy on benign samples, limiting practical utility. Although incorporating extra training data can alleviate the trade-off to a certain extent, it remains unsolved to achieve both robustness and accuracy under limited training data. Here, we demonstrate the feasibility of overcoming the trade-off, by developing an adversarial feature stacking (AFS) model, which combines multiple independent feature extractors with varied levels of robustness and accuracy. Theoretical analysis is further conducted, and general principles for the selection of basic feature extractors are provided. We evaluate the AFS model on CIFAR-10 and CIFAR-100 datasets with strong adaptive attack methods, significantly advancing the state-of-the-art in terms of the trade-off. The AFS model achieves a benign accuracy improvement of ~6% on CIFAR-10 and ~10% on CIFAR-100 with comparable or even stronger robustness than the state-of-the-art adversarial training methods.

摘要: 深度神经网络在各种应用中取得了显著的性能，但极易受到对抗性扰动的影响。最具代表性和最有前景的增强模型稳健性的方法，如对抗性训练及其变种，大大降低了良性样本的模型精度，限制了实际应用。虽然加入额外的训练数据可以在一定程度上缓解这一权衡，但在有限的训练数据下达到稳健性和准确性的问题仍然没有解决。在这里，我们通过开发一种对抗性特征堆栈(AFS)模型来证明克服这种权衡的可行性，该模型结合了多个具有不同水平的稳健性和准确性的独立特征提取器。在此基础上进行了理论分析，给出了基本特征提取算子选择的一般原则。我们在CIFAR-10和CIFAR-100数据集上使用强自适应攻击方法对AFS模型进行了评估，在权衡方面显著提高了最新水平。AFS模型在CIFAR-10和CIFAR-100上的准确率分别提高了~6%和~10%，与最先进的对抗性训练方法相比，具有相当甚至更强的鲁棒性。



## **5. Tricking the Hashing Trick: A Tight Lower Bound on the Robustness of CountSketch to Adaptive Inputs**

欺骗散列技巧：CountSketch对自适应输入的稳健性的紧致下界 cs.DS

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2207.00956v2)

**Authors**: Edith Cohen, Jelani Nelson, Tamás Sarlós, Uri Stemmer

**Abstracts**: CountSketch and Feature Hashing (the "hashing trick") are popular randomized dimensionality reduction methods that support recovery of $\ell_2$-heavy hitters (keys $i$ where $v_i^2 > \epsilon \|\boldsymbol{v}\|_2^2$) and approximate inner products. When the inputs are {\em not adaptive} (do not depend on prior outputs), classic estimators applied to a sketch of size $O(\ell/\epsilon)$ are accurate for a number of queries that is exponential in $\ell$. When inputs are adaptive, however, an adversarial input can be constructed after $O(\ell)$ queries with the classic estimator and the best known robust estimator only supports $\tilde{O}(\ell^2)$ queries. In this work we show that this quadratic dependence is in a sense inherent: We design an attack that after $O(\ell^2)$ queries produces an adversarial input vector whose sketch is highly biased. Our attack uses "natural" non-adaptive inputs (only the final adversarial input is chosen adaptively) and universally applies with any correct estimator, including one that is unknown to the attacker. In that, we expose inherent vulnerability of this fundamental method.

摘要: CountSketch和Feature Hash(散列技巧)是流行的随机降维方法，支持恢复$\ell_2$-重打击者(key$i$where$v_i^2>\epsilon\boldsign{v}\_2^2$)和近似内积。当输入是{\em不自适应的}(不依赖于先前的输出)时，应用于大小为$O(\ell/\epsilon)$的草图的经典估计器对于许多以$\ell$为指数的查询是准确的。然而，当输入是自适应的时，可以用经典估计在$O(\ell)$查询之后构造敌意输入，而最著名的稳健估计只支持$T{O}(\ell^2)$查询。在这项工作中，我们证明了这种二次依赖在某种意义上是固有的：我们设计了一个攻击，在$O(\ell^2)$查询后产生一个高度有偏的敌对输入向量。我们的攻击使用“自然的”非自适应输入(只有最终的对抗性输入是自适应选择的)，并且普遍适用于任何正确的估计器，包括攻击者未知的估计器。在这一点上，我们暴露了这一基本方法的固有弱点。



## **6. Categorical composable cryptography: extended version**

范畴可合成密码学：扩展版本 cs.CR

Extended version of arXiv:2105.05949 which appeared in FoSSaCS 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2208.13232v1)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstracts**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. W We conclude by using string diagrams to rederive the security of the one-time pad and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting. On the way, we exhibit two categorical constructions of resource theories that might be of independent interest: one capturing resources shared among multiple parties and one capturing resource conversions that succeed asymptotically.

摘要: 我们用范畴理论形式化了密码学的模拟范型，证明了对抽象攻击安全的协议形成了对称的么半范畴，从而给出了密码学中可组合安全定义的抽象模型。我们的模型能够以模块化、灵活的方式结合计算安全性、设置假设和各种攻击模型，例如串通或独立行动的对手子集。W我们最后使用字符串图重新推导了关于两方和三方密码术限制的一次一密和不进行结果的安全性，排除了例如可组合承诺和广播。在此过程中，我们展示了两种可能独立感兴趣的资源理论范畴结构：一种是捕获多方共享的资源，另一种是捕获渐近成功的资源转换。



## **7. Categorical composable cryptography**

范畴可合成密码学 cs.CR

Updated to match the proceedings version

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2105.05949v3)

**Authors**: Anne Broadbent, Martti Karvonen

**Abstracts**: We formalize the simulation paradigm of cryptography in terms of category theory and show that protocols secure against abstract attacks form a symmetric monoidal category, thus giving an abstract model of composable security definitions in cryptography. Our model is able to incorporate computational security, set-up assumptions and various attack models such as colluding or independently acting subsets of adversaries in a modular, flexible fashion. We conclude by using string diagrams to rederive the security of the one-time pad and no-go results concerning the limits of bipartite and tripartite cryptography, ruling out e.g., composable commitments and broadcasting.

摘要: 我们用范畴理论形式化了密码学的模拟范型，证明了对抽象攻击安全的协议形成了对称的么半范畴，从而给出了密码学中可组合安全定义的抽象模型。我们的模型能够以模块化、灵活的方式结合计算安全性、设置假设和各种攻击模型，例如串通或独立行动的对手子集。最后，我们使用字符串图重新推导出关于两方和三方密码术限制的一次性密码本和禁止结果的安全性，排除了例如可组合承诺和广播。



## **8. Self-Supervised Adversarial Example Detection by Disentangled Representation**

基于解缠表示的自监督敌意范例检测 cs.CV

to appear in TrustCom 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2105.03689v4)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Jinyu Tian, Jiantao Zhou

**Abstracts**: Deep learning models are known to be vulnerable to adversarial examples that are elaborately designed for malicious purposes and are imperceptible to the human perceptual system. Autoencoder, when trained solely over benign examples, has been widely used for (self-supervised) adversarial detection based on the assumption that adversarial examples yield larger reconstruction errors. However, because lacking adversarial examples in its training and the too strong generalization ability of autoencoder, this assumption does not always hold true in practice. To alleviate this problem, we explore how to detect adversarial examples with disentangled label/semantic features under the autoencoder structure. Specifically, we propose Disentangled Representation-based Reconstruction (DRR). In DRR, we train an autoencoder over both correctly paired label/semantic features and incorrectly paired label/semantic features to reconstruct benign and counterexamples. This mimics the behavior of adversarial examples and can reduce the unnecessary generalization ability of autoencoder. We compare our method with the state-of-the-art self-supervised detection methods under different adversarial attacks and different victim models, and it exhibits better performance in various metrics (area under the ROC curve, true positive rate, and true negative rate) for most attack settings. Though DRR is initially designed for visual tasks only, we demonstrate that it can be easily extended for natural language tasks as well. Notably, different from other autoencoder-based detectors, our method can provide resistance to the adaptive adversary.

摘要: 众所周知，深度学习模型容易受到敌意例子的攻击，这些例子是为恶意目的精心设计的，人类感知系统无法察觉。当仅对良性样本进行训练时，自动编码器已被广泛用于(自监督)敌意检测，其基础是假设对抗性样本产生更大的重建误差。然而，由于在训练中缺乏对抗性的例子，而且自动编码器的泛化能力太强，这一假设在实践中并不总是成立的。为了缓解这一问题，我们探索了如何在自动编码器结构下检测具有分离的标签/语义特征的对抗性示例。具体地说，我们提出了基于解缠表示的重建算法(DRR)。在DRR中，我们训练一个自动编码器，包括正确配对的标签/语义特征和错误配对的标签/语义特征，以重建良性和反例。这模仿了对抗性例子的行为，并且可以降低自动编码器不必要的泛化能力。在不同的对手攻击和不同的受害者模型下，我们的方法与最新的自监督检测方法进行了比较，在大多数攻击环境下，它在各种指标(ROC曲线下面积、真正确率和真负率)上都表现出了更好的性能。虽然DRR最初是为视觉任务设计的，但我们演示了它也可以很容易地扩展到自然语言任务。值得注意的是，与其他基于自动编码器的检测器不同，我们的方法可以提供对自适应对手的抵抗。



## **9. Cross-domain Cross-architecture Black-box Attacks on Fine-tuned Models with Transferred Evolutionary Strategies**

基于转移进化策略的精调模型跨域跨体系结构黑盒攻击 cs.LG

To appear in CIKM 2022

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2208.13182v1)

**Authors**: Yinghua Zhang, Yangqiu Song, Kun Bai, Qiang Yang

**Abstracts**: Fine-tuning can be vulnerable to adversarial attacks. Existing works about black-box attacks on fine-tuned models (BAFT) are limited by strong assumptions. To fill the gap, we propose two novel BAFT settings, cross-domain and cross-domain cross-architecture BAFT, which only assume that (1) the target model for attacking is a fine-tuned model, and (2) the source domain data is known and accessible. To successfully attack fine-tuned models under both settings, we propose to first train an adversarial generator against the source model, which adopts an encoder-decoder architecture and maps a clean input to an adversarial example. Then we search in the low-dimensional latent space produced by the encoder of the adversarial generator. The search is conducted under the guidance of the surrogate gradient obtained from the source model. Experimental results on different domains and different network architectures demonstrate that the proposed attack method can effectively and efficiently attack the fine-tuned models.

摘要: 微调很容易受到对手的攻击。现有的关于精调模型(BAFT)上的黑盒攻击的工作都受到强假设的限制。为了填补这一空白，我们提出了两种新的BAFT设置，跨域和跨域跨架构BAFT，它们只假设(1)攻击的目标模型是微调的模型，(2)源域数据是已知的和可访问的。为了在这两种情况下成功攻击微调模型，我们建议首先针对源模型训练敌意生成器，该模型采用编解码器架构，并将干净的输入映射到对抗性示例。然后在对抗性生成器的编码器产生的低维潜在空间中进行搜索。搜索是在从源模型获得的代理梯度的指导下进行的。在不同域和不同网络体系结构上的实验结果表明，该攻击方法能够有效地攻击微调模型。



## **10. Improved and Interpretable Defense to Transferred Adversarial Examples by Jacobian Norm with Selective Input Gradient Regularization**

基于选择输入梯度正则化的雅可比范数对转移对抗性实例的改进和可解释防御 cs.LG

Under review

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2207.13036v3)

**Authors**: Deyin Liu, Lin Wu, Lingqiao Liu, Haifeng Zhao, Farid Boussaid, Mohammed Bennamoun

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve robustness through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with transferred adversarial examples which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose a novel approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.

摘要: 众所周知，深度神经网络(DNN)容易受到带有不可察觉扰动的敌意示例的影响，即输入图像的微小变化就会导致误分类，从而威胁到基于深度学习的部署系统的可靠性。对抗性训练(AT)经常被用来通过训练被破坏和被破坏的数据的混合来提高稳健性。然而，大多数基于AT的方法都不能有效地处理转移的对抗性例子，这些例子是为了愚弄广泛的防御模型而产生的，因此不能满足现实场景中提出的泛化要求。此外，对抗性地训练防御模型一般不能产生对带有扰动的输入的可解释预测，而不同领域的专家需要高度可解释的稳健模型来理解DNN的行为。在本文中，我们提出了一种基于雅可比范数和选择性输入梯度正则化(J-SIGR)的新方法，该方法通过雅可比归一化提供线性化的稳健性，并将基于扰动的显著图正则化以模拟模型的可解释预测。因此，我们实现了DNN的改进的防御性和高度的可解释性。最后，我们在不同的体系结构上对我们的方法进行了评估，以对抗强大的对手攻击。实验表明，所提出的J-SIGR算法对转移攻击具有较好的稳健性，并且神经网络的预测结果易于解释。



## **11. Covariate Balancing Methods for Randomized Controlled Trials Are Not Adversarially Robust**

随机对照试验的协变量平衡方法不是逆稳健的 econ.EM

12 pages, double column, 4 figures

**SubmitDate**: 2022-08-28    [paper-pdf](http://arxiv.org/pdf/2110.13262v3)

**Authors**: Hossein Babaei, Sina Alemohammad, Richard Baraniuk

**Abstracts**: The first step towards investigating the effectiveness of a treatment via a randomized trial is to split the population into control and treatment groups then compare the average response of the treatment group receiving the treatment to the control group receiving the placebo.   In order to ensure that the difference between the two groups is caused only by the treatment, it is crucial that the control and the treatment groups have similar statistics. Indeed, the validity and reliability of a trial are determined by the similarity of two groups' statistics. Covariate balancing methods increase the similarity between the distributions of the two groups' covariates. However, often in practice, there are not enough samples to accurately estimate the groups' covariate distributions. In this paper, we empirically show that covariate balancing with the Standardized Means Difference (SMD) covariate balancing measure, as well as Pocock's sequential treatment assignment method, are susceptible to worst-case treatment assignments. Worst-case treatment assignments are those admitted by the covariate balance measure, but result in highest possible ATE estimation errors. We developed an adversarial attack to find adversarial treatment assignment for any given trial. Then, we provide an index to measure how close the given trial is to the worst-case. To this end, we provide an optimization-based algorithm, namely Adversarial Treatment ASsignment in TREatment Effect Trials (ATASTREET), to find the adversarial treatment assignments.

摘要: 通过随机试验调查治疗有效性的第一步是将人群分为对照组和治疗组，然后比较接受治疗的治疗组和接受安慰剂的对照组的平均反应。为了确保两组之间的差异只由治疗引起，至关重要的是对照组和治疗组有类似的统计数据。事实上，试验的有效性和可靠性取决于两组统计数据的相似性。协变量平衡方法增加了两组协变量分布之间的相似性。然而，在实践中，往往没有足够的样本来准确估计群体的协变量分布。在这篇文章中，我们的经验表明，协变量平衡与标准化均值差(SMD)协变量平衡度量，以及Pocock的序贯处理分配方法，都容易受到最坏情况处理分配的影响。最坏情况下的处理分配是协变量平衡测量允许的，但会导致最大可能的ATE估计误差。我们开发了一种对抗性攻击，以便为任何给定的试验找到对抗性治疗任务。然后，我们提供一个指数来衡量给定的试验离最坏情况有多近。为此，我们提出了一种基于优化的算法，即治疗效果试验中的对抗性处理分配算法(ATASTREET)来寻找对抗性处理分配。



## **12. Overcoming Data Availability Attacks in Blockchain Systems: Short Code-Length LDPC Code Design for Coded Merkle Tree**

克服区块链系统中的数据可用性攻击：基于编码Merkle树的短码长LDPC码设计 cs.IT

18 pages, 7 figures, 3 tables, accepted at IEEE Transactions on  Communications (TCOM) 2022. This version reflects comments from reviewers at  TCOM

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2108.13332v3)

**Authors**: Debarnab Mitra, Lev Tauz, Lara Dolecek

**Abstracts**: Light nodes are clients in blockchain systems that only store a small portion of the blockchain ledger. In certain blockchains, light nodes are vulnerable to a data availability (DA) attack where a malicious node makes the light nodes accept an invalid block by hiding the invalid portion of the block from the nodes in the system. Recently, a technique based on LDPC codes called Coded Merkle Tree was proposed by Yu et al. that enables light nodes to detect a DA attack by randomly requesting/sampling portions of the block from the malicious node. However, light nodes fail to detect a DA attack with high probability if a malicious node hides a small stopping set of the LDPC code. In this paper, we demonstrate that a suitable co-design of specialized LDPC codes and the light node sampling strategy leads to a high probability of detection of DA attacks. We consider different adversary models based on their computational capabilities of finding stopping sets. For the different adversary models, we provide new specialized LDPC code constructions and coupled light node sampling strategies and demonstrate that they lead to a higher probability of detection of DA attacks compared to approaches proposed in earlier literature.

摘要: 轻节点是区块链系统中的客户端，只存储区块链账簿的一小部分。在某些区块链中，轻节点容易受到数据可用性(DA)攻击，其中恶意节点通过向系统中的节点隐藏块的无效部分来使轻节点接受无效块。最近，Yu等人提出了一种基于LDPC码的编码Merkle树技术。这使得轻节点能够通过随机请求/采样来自恶意节点的块的部分来检测DA攻击。然而，如果恶意节点隐藏了LDPC码的一小部分停止集，则轻节点无法检测到DA攻击的概率很高。在本文中，我们证明了适当的专用LDPC码和光节点采样策略的联合设计可以导致高概率的DA攻击被检测。我们考虑了不同的对手模型，基于它们寻找停止集的计算能力。对于不同的敌方模型，我们提出了新的专用LDPC码构造和耦合光节点采样策略，并证明了它们比以往文献中提出的方法具有更高的DA攻击检测概率。



## **13. Cooperative Distributed State Estimation: Resilient Topologies against Smart Spoofers**

协作分布式状态估计：针对智能欺骗器的弹性拓扑 cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/1909.04172v4)

**Authors**: Mostafa Safi

**Abstracts**: A network of observers is considered, where through asynchronous (with bounded delay) communications, they cooperatively estimate the states of a Linear Time-Invariant (LTI) system. In such a setting, a new type of adversary might affect the observation process by impersonating the identity of the regular node, which is a violation of communication authenticity. These adversaries also inherit the capabilities of Byzantine nodes, making them more powerful threats called smart spoofers. We show how asynchronous networks are vulnerable to smart spoofing attack. In the estimation scheme considered in this paper, information flows from the sets of source nodes, which can detect a portion of the state variables each, to the other follower nodes. The regular nodes, to avoid being misguided by the threats, distributively filter the extreme values received from the nodes in their neighborhood. Topological conditions based on strong robustness are proposed to guarantee the convergence. Two simulation scenarios are provided to verify the results.

摘要: 考虑了一个观测器网络，其中通过异步(有界时延)通信，它们协作估计线性时不变(LTI)系统的状态。在这种情况下，新类型的对手可能会通过冒充常规节点的身份来影响观察过程，这违反了通信的真实性。这些对手还继承了拜占庭节点的能力，使它们成为更强大的威胁，称为智能欺骗程序。我们展示了异步网络如何容易受到智能欺骗攻击。在本文所考虑的估计方案中，信息从源节点集合流向其他跟随者节点，每个源节点集合可以检测到一部分状态变量。规则节点为了避免被威胁误导，对邻居节点接收到的极值进行分布式过滤。为了保证算法的收敛，提出了基于强鲁棒性的拓扑条件。文中给出了两个仿真场景来验证结果。



## **14. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

SA：抗剪裁和自拼接的合成语音检测滑动攻击 cs.SD

12 pages, Neurocomputing

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13066v1)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstracts**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.

摘要: 深度神经网络很容易受到敌意例子的影响，这些例子用无法察觉的扰动误导模型。在音频方面，虽然对抗性例子在白盒和黑盒设置上取得了令人难以置信的攻击成功率，但大多数现有的对抗性攻击都受到输入长度的限制。一个更实际的场景是，对抗性的例子必须被剪裁或自我拼接，并输入到黑盒模型中。因此，有必要探索如何在不同的输入长度设置下提高可转移性。在本文中，我们以合成语音检测任务为例，考虑了两个具有代表性的SOTA模型。通过分析剪裁或自剪接后将样本送入模型获得的梯度，我们观察到相同样本值的片段在不同模型中的梯度是相似的。受此启发，我们提出了一种新的对抗性攻击方法--滑动攻击。具体地说，我们使每个采样点知道不同位置的梯度，这可以模拟对抗性例子被输入到具有不同输入长度的黑盒模型的情况。因此，我们不是在梯度计算的每次迭代中直接使用当前梯度，而是经历以下三个步骤。首先，我们使用滑动窗口提取不同长度的子段。然后，我们使用来自相邻域的数据来增强子分段。最后，我们将子片段输入到不同的模型中，以获得聚合梯度来更新对抗性实例。实验结果表明，该方法能够显著提高截取或自拼接后的对抗性样本的可转移性。此外，我们的方法还可以增强基于不同特征的模型之间的可移植性。



## **15. Adversarial Robustness for Tabular Data through Cost and Utility Awareness**

通过成本和效用意识实现表格数据的对抗稳健性 cs.LG

* authors contributed equally

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13058v1)

**Authors**: Klim Kireev, Bogdan Kulynych, Carmela Troncoso

**Abstracts**: Many machine learning problems use data in the tabular domains. Adversarial examples can be especially damaging for these applications. Yet, existing works on adversarial robustness mainly focus on machine-learning models in the image and text domains. We argue that due to the differences between tabular data and images or text, existing threat models are inappropriate for tabular domains. These models do not capture that cost can be more important than imperceptibility, nor that the adversary could ascribe different value to the utility obtained from deploying different adversarial examples. We show that due to these differences the attack and defence methods used for images and text cannot be directly applied to the tabular setup. We address these issues by proposing new cost and utility-aware threat models tailored to the adversarial capabilities and constraints of attackers targeting tabular domains. We introduce a framework that enables us to design attack and defence mechanisms which result in models protected against cost or utility-aware adversaries, e.g., adversaries constrained by a certain dollar budget. We show that our approach is effective on three tabular datasets corresponding to applications for which adversarial examples can have economic and social implications.

摘要: 许多机器学习问题都使用表格域中的数据。对抗性的例子可能对这些应用程序特别有害。然而，现有的关于对抗稳健性的研究主要集中在图像和文本领域的机器学习模型。我们认为，由于表格数据与图像或文本之间的差异，现有的威胁模型不适用于表格领域。这些模型没有捕捉到成本可能比不可察觉更重要，也没有捕捉到对手可以将不同的价值归因于通过部署不同的对抗性例子而获得的效用。我们表明，由于这些差异，用于图像和文本的攻击和防御方法不能直接应用于表格设置。我们通过提出新的成本和效用感知威胁模型来解决这些问题，该模型针对针对表格域的攻击者的对抗能力和约束而量身定做。我们引入了一个框架，使我们能够设计攻击和防御机制，从而产生针对成本或效用意识的对手的模型，例如，受特定美元预算限制的对手。我们表明，我们的方法在三个表格数据集上是有效的，这些数据集对应于对抗性例子可能具有经济和社会影响的应用程序。



## **16. TrojViT: Trojan Insertion in Vision Transformers**

TrojViT：视觉变形金刚中的特洛伊木马插入 cs.LG

9 pages, 4 figures, 9 tables

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13049v1)

**Authors**: Mengxin Zheng, Qian Lou, Lei Jiang

**Abstracts**: Vision Transformers (ViTs) have demonstrated the state-of-the-art performance in various vision-related tasks. The success of ViTs motivates adversaries to perform backdoor attacks on ViTs. Although the vulnerability of traditional CNNs to backdoor attacks is well-known, backdoor attacks on ViTs are seldom-studied. Compared to CNNs capturing pixel-wise local features by convolutions, ViTs extract global context information through patches and attentions. Na\"ively transplanting CNN-specific backdoor attacks to ViTs yields only a low clean data accuracy and a low attack success rate. In this paper, we propose a stealth and practical ViT-specific backdoor attack $TrojViT$. Rather than an area-wise trigger used by CNN-specific backdoor attacks, TrojViT generates a patch-wise trigger designed to build a Trojan composed of some vulnerable bits on the parameters of a ViT stored in DRAM memory through patch salience ranking and attention-target loss. TrojViT further uses minimum-tuned parameter update to reduce the bit number of the Trojan. Once the attacker inserts the Trojan into the ViT model by flipping the vulnerable bits, the ViT model still produces normal inference accuracy with benign inputs. But when the attacker embeds a trigger into an input, the ViT model is forced to classify the input to a predefined target class. We show that flipping only few vulnerable bits identified by TrojViT on a ViT model using the well-known RowHammer can transform the model into a backdoored one. We perform extensive experiments of multiple datasets on various ViT models. TrojViT can classify $99.64\%$ of test images to a target class by flipping $345$ bits on a ViT for ImageNet.

摘要: 视觉变形金刚(VITS)在各种与视觉相关的任务中展示了最先进的性能。VITS的成功促使对手对VITS进行后门攻击。虽然传统的CNN对后门攻击的脆弱性是众所周知的，但对VITS的后门攻击很少被研究。与通过卷积获取像素级局部特征的CNN相比，VITS通过块和关注点来提取全局上下文信息。将CNN特定的后门攻击活生生地移植到VITS只会产生低的干净数据准确性和低的攻击成功率。在本文中，我们提出了一种隐形和实用的特定于VIT的后门攻击$TrojViT$。与CNN特定后门攻击使用的区域触发不同，TrojViT生成修补程序触发，旨在通过修补程序显著程度排名和注意力目标丢失来构建由存储在DRAM内存中的VIT参数上的一些易受攻击位组成的特洛伊木马程序。TrojViT进一步使用最小调整的参数更新来减少特洛伊木马的比特数。一旦攻击者通过翻转易受攻击的比特将特洛伊木马程序插入到VIT模型中，VIT模型仍然会使用良性输入产生正常的推理准确性。但是，当攻击者将触发器嵌入到输入中时，VIT模型被迫将输入分类到预定义的目标类。我们表明，只需使用著名的RowHammer在VIT模型上翻转TrojViT识别的少数易受攻击的位，就可以将该模型转换为后置模型。我们在不同的VIT模型上对多个数据集进行了广泛的实验。TrojViT可以通过在ImageNet的VIT上翻转$345$比特，将$99.64\$测试图像分类到目标类别。



## **17. SoK: Decentralized Finance (DeFi) Incidents**

SOK：分散金融(Defi)事件 cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.13035v1)

**Authors**: Liyi Zhou, Xihan Xiong, Jens Ernstberger, Stefanos Chaliasos, Zhipeng Wang, Ye Wang, Kaihua Qin, Roger Wattenhofer, Dawn Song, Arthur Gervais

**Abstracts**: Within just four years, the blockchain-based Decentralized Finance (DeFi) ecosystem has accumulated a peak total value locked (TVL) of more than 253 billion USD. This surge in DeFi's popularity has, unfortunately, been accompanied by many impactful incidents. According to our data, users, liquidity providers, speculators, and protocol operators suffered a total loss of at least 3.24 USD from Apr 30, 2018 to Apr 30, 2022. Given the blockchain's transparency and increasing incident frequency, two questions arise: How can we systematically measure, evaluate, and compare DeFi incidents? How can we learn from past attacks to strengthen DeFi security?   In this paper, we introduce a common reference frame to systematically evaluate and compare DeFi incidents. We investigate 77 academic papers, 30 audit reports, and 181 real-world incidents. Our open data reveals several gaps between academia and the practitioners' community. For example, few academic papers address "price oracle attacks" and "permissonless interactions", while our data suggests that they are the two most frequent incident types (15% and 10.5% correspondingly). We also investigate potential defenses, and find that: (i) 103 (56%) of the attacks are not executed atomically, granting a rescue time frame for defenders; (ii) SoTA bytecode similarity analysis can at least detect 31 vulnerable/23 adversarial contracts; and (iii) 33 (15.3%) of the adversaries leak potentially identifiable information by interacting with centralized exchanges.

摘要: 短短四年时间，基于区块链的去中心化金融(DEFI)生态系统已经积累了超过2530亿美元的峰值总价值锁定(TVL)。不幸的是，Defi人气的飙升伴随着许多有影响力的事件。根据我们的数据，从2018年4月30日到2022年4月30日，用户、流动性提供商、投机者和协议运营商总共遭受了至少3.24美元的损失。鉴于区块链的透明度和不断增加的事件频率，出现了两个问题：我们如何系统地衡量、评估和比较Defi事件？我们如何从过去的袭击中吸取教训，以加强Defi安全？在本文中，我们引入了一个通用的参照系来系统地评估和比较DEFI事件。我们调查了77篇学术论文，30份审计报告和181起真实世界的事件。我们的公开数据揭示了学术界和从业者社区之间的几个差距。举例来说，很少有学术论文涉及“价格先知攻击”和“不允许的相互作用”，而我们的数据显示，它们是最常见的两种事件类型(分别为15%和10.5%)。我们还调查了潜在的防御措施，发现：(I)103(56%)的攻击不是自动执行的，这为防御者提供了救援时间框架；(Ii)Sota字节码相似性分析至少可以检测到31个VULNERABLE/23个对手合同；以及(Iii)33个(15.3%)的对手通过与中央交易所的交互泄露了潜在的可识别信息。



## **18. Overparameterized (robust) models from computational constraints**

计算约束条件下的超参数(稳健)模型 cs.LG

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.12926v1)

**Authors**: Sanjam Garg, Somesh Jha, Saeed Mahloujifar, Mohammad Mahmoody, Mingyuan Wang

**Abstracts**: Overparameterized models with millions of parameters have been hugely successful. In this work, we ask: can the need for large models be, at least in part, due to the \emph{computational} limitations of the learner? Additionally, we ask, is this situation exacerbated for \emph{robust} learning? We show that this indeed could be the case. We show learning tasks for which computationally bounded learners need \emph{significantly more} model parameters than what information-theoretic learners need. Furthermore, we show that even more model parameters could be necessary for robust learning. In particular, for computationally bounded learners, we extend the recent result of Bubeck and Sellke [NeurIPS'2021] which shows that robust models might need more parameters, to the computational regime and show that bounded learners could provably need an even larger number of parameters. Then, we address the following related question: can we hope to remedy the situation for robust computationally bounded learning by restricting \emph{adversaries} to also be computationally bounded for sake of obtaining models with fewer parameters? Here again, we show that this could be possible. Specifically, building on the work of Garg, Jha, Mahloujifar, and Mahmoody [ALT'2020], we demonstrate a learning task that can be learned efficiently and robustly against a computationally bounded attacker, while to be robust against an information-theoretic attacker requires the learner to utilize significantly more parameters.

摘要: 具有数百万个参数的过度参数模型已经取得了巨大的成功。在这项工作中，我们问：对大型模型的需求是否至少部分是由于学习者的计算限制？此外，我们问，这种情况是否会因为学习而加剧？我们表明，情况确实可能是这样的。我们展示了计算受限的学习者比信息论学习者需要更多的模型参数的学习任务。此外，我们还表明，稳健学习可能需要更多的模型参数。特别是，对于计算有界的学习者，我们将Bubeck和Sellke[NeurIPS‘2021]的最新结果推广到计算机制，该结果表明健壮模型可能需要更多的参数，并表明有界的学习者可能需要更多的参数。然后，我们解决了以下相关问题：为了获得参数更少的模型，我们是否可以通过限制对手也是计算有界的来纠正健壮的计算有界学习的情况？在这里，我们再次证明了这是可能的。具体地说，在Garg，Jha，MahLoujifar和Mahmoody[Alt‘2020]的工作基础上，我们演示了一种学习任务，该任务可以在计算受限的攻击者面前高效而稳健地学习，而为了对信息论攻击者具有健壮性，需要学习者使用更多的参数。



## **19. Bitcoin's Latency--Security Analysis Made Simple**

比特币的潜伏期--安全分析变得简单 cs.CR

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2203.06357v3)

**Authors**: Dongning Guo, Ling Ren

**Abstracts**: Simple closed-form upper and lower bounds are developed for the security of the Nakamoto consensus as a function of the confirmation depth, the honest and adversarial block mining rates, and an upper bound on the block propagation delay. The bounds are exponential in the confirmation depth and apply regardless of the adversary's attack strategy. The gap between the upper and lower bounds is small for Bitcoin's parameters. For example, assuming an average block interval of 10 minutes, a network delay bound of ten seconds, and 10% adversarial mining power, the widely used 6-block confirmation rule yields a safety violation between 0.11% and 0.35% probability.

摘要: 对于Nakamoto共识的安全性，给出了简单的闭合上下界，作为确认深度、诚实和对抗性块挖掘率的函数，以及块传播延迟的上界。这些界限在确认深度上是指数级的，无论对手的攻击策略如何，都适用。就比特币的参数而言，上下限之间的差距很小。例如，假设平均阻塞间隔为10分钟，网络延迟界限为10秒，对抗性挖掘能力为10%，则广泛使用的6-块确认规则产生的安全违规概率在0.11%到0.35%之间。



## **20. Network-Level Adversaries in Federated Learning**

联合学习中的网络级对手 cs.CR

12 pages. Appearing at IEEE CNS 2022

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2208.12911v1)

**Authors**: Giorgio Severi, Matthew Jagielski, Gökberk Yar, Yuxuan Wang, Alina Oprea, Cristina Nita-Rotaru

**Abstracts**: Federated learning is a popular strategy for training models on distributed, sensitive data, while preserving data privacy. Prior work identified a range of security threats on federated learning protocols that poison the data or the model. However, federated learning is a networked system where the communication between clients and server plays a critical role for the learning task performance. We highlight how communication introduces another vulnerability surface in federated learning and study the impact of network-level adversaries on training federated learning models. We show that attackers dropping the network traffic from carefully selected clients can significantly decrease model accuracy on a target population. Moreover, we show that a coordinated poisoning campaign from a few clients can amplify the dropping attacks. Finally, we develop a server-side defense which mitigates the impact of our attacks by identifying and up-sampling clients likely to positively contribute towards target accuracy. We comprehensively evaluate our attacks and defenses on three datasets, assuming encrypted communication channels and attackers with partial visibility of the network.

摘要: 联合学习是一种流行的策略，用于训练分布式、敏感数据的模型，同时保护数据隐私。以前的工作确定了联合学习协议上的一系列安全威胁，这些威胁毒害了数据或模型。然而，联合学习是一个网络系统，客户端和服务器之间的通信对学习任务的性能起着至关重要的作用。我们重点介绍了通信如何在联合学习中引入另一个脆弱性表面，并研究了网络级对手对训练联合学习模型的影响。我们表明，攻击者丢弃来自精心选择的客户端的网络流量会显著降低对目标人群的模型精度。此外，我们还表明，来自几个客户的协调投毒活动可以放大丢弃攻击。最后，我们开发了一种服务器端防御，通过识别和向上采样可能对目标准确性做出积极贡献的客户端来减轻攻击的影响。我们综合评估了我们对三个数据集的攻击和防御，假设加密的通信渠道和攻击者对网络具有部分可见性。



## **21. Adversarial Relighting Against Face Recognition**

对抗人脸识别的对抗性重发 cs.CV

**SubmitDate**: 2022-08-27    [paper-pdf](http://arxiv.org/pdf/2108.07920v4)

**Authors**: Qian Zhang, Qing Guo, Ruijun Gao, Felix Juefei-Xu, Hongkai Yu, Wei Feng

**Abstracts**: Deep face recognition (FR) has achieved significantly high accuracy on several challenging datasets and fosters successful real-world applications, even showing high robustness to the illumination variation that is usually regarded as a main threat to the FR system. However, in the real world, illumination variation caused by diverse lighting conditions cannot be fully covered by the limited face dataset. In this paper, we study the threat of lighting against FR from a new angle, i.e., adversarial attack, and identify a new task, i.e., adversarial relighting. Given a face image, adversarial relighting aims to produce a naturally relighted counterpart while fooling the state-of-the-art deep FR methods. To this end, we first propose the physical modelbased adversarial relighting attack (ARA) denoted as albedoquotient-based adversarial relighting attack (AQ-ARA). It generates natural adversarial light under the physical lighting model and guidance of FR systems and synthesizes adversarially relighted face images. Moreover, we propose the auto-predictive adversarial relighting attack (AP-ARA) by training an adversarial relighting network (ARNet) to automatically predict the adversarial light in a one-step manner according to different input faces, allowing efficiency-sensitive applications. More importantly, we propose to transfer the above digital attacks to physical ARA (PhyARA) through a precise relighting device, making the estimated adversarial lighting condition reproducible in the real world. We validate our methods on three state-of-the-art deep FR methods, i.e., FaceNet, ArcFace, and CosFace, on two public datasets. The extensive and insightful results demonstrate our work can generate realistic adversarial relighted face images fooling face recognition tasks easily, revealing the threat of specific light directions and strengths.

摘要: 深度人脸识别(FR)已经在几个具有挑战性的数据集上取得了显著的高精度，并促进了现实世界的成功应用，甚至对通常被视为FR系统主要威胁的光照变化表现出高度的稳健性。然而，在现实世界中，有限的人脸数据集不能完全覆盖由于光照条件的变化而引起的光照变化。本文从对抗性攻击这一新的角度研究了闪电对火箭弹的威胁，并提出了一种新的任务，即对抗性重发。给定脸部图像，对抗性重光旨在产生自然重光的对应物，同时愚弄最先进的深度FR方法。为此，我们首先提出了基于物理模型的对抗性重亮攻击(ARA)，称为基于反商的对抗性重亮攻击(AQ-ARA)。它在物理照明模型和FR系统的指导下产生自然的对抗性光，并合成对抗性重光的人脸图像。此外，我们提出了自动预测对抗性重光攻击(AP-ARA)，通过训练对抗性重光网络(ARNet)来根据不同的输入人脸一步自动预测对抗性光，从而允许对效率敏感的应用。更重要的是，我们建议通过精确的重光装置将上述数字攻击转移到物理ARA(PhyARA)，使估计的对抗性照明条件在现实世界中可重现。在两个公开的数据集上，我们在三种最先进的深度FR方法，即FaceNet，ArcFace和CosFace上对我们的方法进行了验证。广泛而有洞察力的结果表明，我们的工作可以生成真实的对抗性重光照人脸图像，轻松地愚弄人脸识别任务，揭示特定光照方向和强度的威胁。



## **22. ATTRITION: Attacking Static Hardware Trojan Detection Techniques Using Reinforcement Learning**

消耗性：基于强化学习的攻击静态硬件木马检测技术 cs.CR

To Appear in 2022 ACM SIGSAC Conference on Computer and  Communications Security (CCS), November 2022

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12897v1)

**Authors**: Vasudev Gohil, Hao Guo, Satwik Patnaik, Jeyavijayan, Rajendran

**Abstracts**: Stealthy hardware Trojans (HTs) inserted during the fabrication of integrated circuits can bypass the security of critical infrastructures. Although researchers have proposed many techniques to detect HTs, several limitations exist, including: (i) a low success rate, (ii) high algorithmic complexity, and (iii) a large number of test patterns. Furthermore, the most pertinent drawback of prior detection techniques stems from an incorrect evaluation methodology, i.e., they assume that an adversary inserts HTs randomly. Such inappropriate adversarial assumptions enable detection techniques to claim high HT detection accuracy, leading to a "false sense of security." Unfortunately, to the best of our knowledge, despite more than a decade of research on detecting HTs inserted during fabrication, there have been no concerted efforts to perform a systematic evaluation of HT detection techniques.   In this paper, we play the role of a realistic adversary and question the efficacy of HT detection techniques by developing an automated, scalable, and practical attack framework, ATTRITION, using reinforcement learning (RL). ATTRITION evades eight detection techniques across two HT detection categories, showcasing its agnostic behavior. ATTRITION achieves average attack success rates of $47\times$ and $211\times$ compared to randomly inserted HTs against state-of-the-art HT detection techniques. We demonstrate ATTRITION's ability to evade detection techniques by evaluating designs ranging from the widely-used academic suites to larger designs such as the open-source MIPS and mor1kx processors to AES and a GPS module. Additionally, we showcase the impact of ATTRITION-generated HTs through two case studies (privilege escalation and kill switch) on the mor1kx processor. We envision that our work, along with our released HT benchmarks and models, fosters the development of better HT detection techniques.

摘要: 在集成电路制造过程中插入的隐形硬件特洛伊木马(HTS)可以绕过关键基础设施的安全。尽管研究人员已经提出了许多技术来检测HTS，但仍然存在一些局限性，包括：(I)成功率低，(Ii)算法复杂性高，以及(Iii)大量的测试模式。此外，现有检测技术的最大缺陷源于不正确的评估方法，即它们假设对手随机插入HTS。这种不恰当的敌意假设使检测技术能够声称高HT检测精度，从而导致“错误的安全感”。不幸的是，就我们所知，尽管对在制造过程中插入的高温超导的检测进行了十多年的研究，但还没有一致的努力来对高温超导检测技术进行系统的评估。在本文中，我们扮演一个现实对手的角色，并通过使用强化学习(RL)开发一个自动化的、可扩展的、实用的攻击框架Astrition来质疑HT检测技术的有效性。磨损在两个HT检测类别中避开了八种检测技术，展示了它的不可知性行为。与随机插入的HTS相比，与最先进的HTS检测技术相比，损耗攻击的平均攻击成功率分别为47倍和211倍。我们通过评估从广泛使用的学术套件到更大的设计(如开放源代码的MIPS和mor1kx处理器，再到AES和GPS模块)的设计来展示磨损逃避检测技术的能力。此外，我们还通过两个案例研究(权限提升和终止开关)展示了损耗产生的HTS对mor1kx处理器的影响。我们设想，我们的工作，连同我们发布的HT基准和模型，将促进更好的HT检测技术的发展。



## **23. SoftHebb: Bayesian Inference in Unsupervised Hebbian Soft Winner-Take-All Networks**

SoftHebb：无监督Hebbian软赢家通吃网络中的贝叶斯推理 cs.LG

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2107.05747v3)

**Authors**: Timoleon Moraitis, Dmitry Toichkin, Adrien Journé, Yansong Chua, Qinghai Guo

**Abstracts**: Hebbian plasticity in winner-take-all (WTA) networks is highly attractive for neuromorphic on-chip learning, owing to its efficient, local, unsupervised, and on-line nature. Moreover, its biological plausibility may help overcome important limitations of artificial algorithms, such as their susceptibility to adversarial attacks and long training time. However, Hebbian WTA learning has found little use in machine learning (ML), likely because it has been missing an optimization theory compatible with deep learning (DL). Here we show rigorously that WTA networks constructed by standard DL elements, combined with a Hebbian-like plasticity that we derive, maintain a Bayesian generative model of the data. Importantly, without any supervision, our algorithm, SoftHebb, minimizes cross-entropy, i.e. a common loss function in supervised DL. We show this theoretically and in practice. The key is a "soft" WTA where there is no absolute "hard" winner neuron. Strikingly, in shallow-network comparisons with backpropagation (BP), SoftHebb shows advantages beyond its Hebbian efficiency. Namely, it converges faster and is significantly more robust to noise and adversarial attacks. Notably, attacks that maximally confuse SoftHebb are also confusing to the human eye, potentially linking human perceptual robustness, with Hebbian WTA circuits of cortex. Finally, SoftHebb can generate synthetic objects as interpolations of real object classes. All in all, Hebbian efficiency, theoretical underpinning, cross-entropy-minimization, and surprising empirical advantages, suggest that SoftHebb may inspire highly neuromorphic and radically different, but practical and advantageous learning algorithms and hardware accelerators.

摘要: Winner-Take-All(WTA)网络中的Hebbian可塑性由于其高效、局部、无监督和在线的性质，对神经形态芯片上学习具有极大的吸引力。此外，它在生物学上的合理性可能有助于克服人工算法的重要局限性，例如它们对对手攻击的敏感性和较长的训练时间。然而，Hebbian WTA学习在机器学习(ML)中几乎没有发现，可能是因为它一直缺少与深度学习(DL)兼容的优化理论。在这里，我们严格地证明了由标准的DL元素构造的WTA网络，与我们推导的Hebbian类可塑性相结合，维持了数据的贝叶斯生成模型。重要的是，在没有任何监督的情况下，我们的算法SoftHebb最小化了交叉熵，即有监督DL中的一个常见损失函数。我们从理论和实践上证明了这一点。关键是没有绝对的“硬”赢家神经元的“软”WTA。值得注意的是，在浅层网络与反向传播(BP)的比较中，SoftHebb显示出了比Hebbian效率更高的优势。也就是说，它的收敛速度更快，对噪声和对手攻击的健壮性要强得多。值得注意的是，最大限度地混淆SoftHebb的攻击也会混淆人眼，潜在地将人类感知的健壮性与大脑皮质的Hebbian WTA回路联系起来。最后，SoftHebb可以生成合成对象作为真实对象类的内插。总而言之，Hebbian效率、理论基础、交叉熵最小化和令人惊讶的经验优势表明，SoftHebb可能会激发高度神经形态和根本不同的、但实用和有利的学习算法和硬件加速器。



## **24. What Does the Gradient Tell When Attacking the Graph Structure**

当攻击图形结构时，渐变说明了什么 cs.LG

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12815v1)

**Authors**: Zihan Liu, Ge Wang, Yun Luo, Stan Z. Li

**Abstracts**: Recent studies have proven that graph neural networks are vulnerable to adversarial attacks. Attackers can rely solely on the training labels to disrupt the performance of the agnostic victim model by edge perturbations. Researchers observe that the saliency-based attackers tend to add edges rather than delete them, which is previously explained by the fact that adding edges pollutes the nodes' features by aggregation while removing edges only leads to some loss of information. In this paper, we further prove that the attackers perturb graphs by adding inter-class edges, which also manifests as a reduction in the homophily of the perturbed graph. From this point of view, saliency-based attackers still have room for improvement in capability and imperceptibility. The message passing of the GNN-based surrogate model leads to the oversmoothing of nodes connected by inter-class edges, preventing attackers from obtaining the distinctiveness of node features. To solve this issue, we introduce a multi-hop aggregated message passing to preserve attribute differences between nodes. In addition, we propose a regularization term to restrict the homophily variance to enhance the attack imperceptibility. Experiments verify that our proposed surrogate model improves the attacker's versatility and the regularization term helps to limit the homophily of the perturbed graph.

摘要: 最近的研究证明，图神经网络很容易受到敌意攻击。攻击者可以完全依赖训练标签来通过边缘扰动来破坏不可知性受害者模型的性能。研究人员观察到，基于显著性的攻击者倾向于添加边而不是删除边，这之前的解释是添加边会通过聚集污染节点的特征，而删除边只会导致一些信息丢失。在这篇文章中，我们进一步证明了攻击者通过添加类间边来扰乱图，这也表现为扰动图的同伦降低。从这个角度来看，基于显著性的攻击者在能力和隐蔽性方面仍有提升的空间。基于GNN的代理模型的消息传递导致类间边连接的节点过度平滑，阻止攻击者获取节点特征的区分性。为了解决这个问题，我们引入了一种多跳聚合消息传递来保持节点之间的属性差异。此外，为了增强攻击的不可见性，我们还提出了一个正则化项来限制同态方差。实验证明，我们提出的代理模型提高了攻击者的通用性，并且正则化项有助于限制扰动图的同质性。



## **25. Robust Prototypical Few-Shot Organ Segmentation with Regularized Neural-ODEs**

基于正则化神经节点的典型少发器官分割 cs.CV

**SubmitDate**: 2022-08-26    [paper-pdf](http://arxiv.org/pdf/2208.12428v1)

**Authors**: Prashant Pandey, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstracts**: Despite the tremendous progress made by deep learning models in image semantic segmentation, they typically require large annotated examples, and increasing attention is being diverted to problem settings like Few-Shot Learning (FSL) where only a small amount of annotation is needed for generalisation to novel classes. This is especially seen in medical domains where dense pixel-level annotations are expensive to obtain. In this paper, we propose Regularized Prototypical Neural Ordinary Differential Equation (R-PNODE), a method that leverages intrinsic properties of Neural-ODEs, assisted and enhanced by additional cluster and consistency losses to perform Few-Shot Segmentation (FSS) of organs. R-PNODE constrains support and query features from the same classes to lie closer in the representation space thereby improving the performance over the existing Convolutional Neural Network (CNN) based FSS methods. We further demonstrate that while many existing Deep CNN based methods tend to be extremely vulnerable to adversarial attacks, R-PNODE exhibits increased adversarial robustness for a wide array of these attacks. We experiment with three publicly available multi-organ segmentation datasets in both in-domain and cross-domain FSS settings to demonstrate the efficacy of our method. In addition, we perform experiments with seven commonly used adversarial attacks in various settings to demonstrate R-PNODE's robustness. R-PNODE outperforms the baselines for FSS by significant margins and also shows superior performance for a wide array of attacks varying in intensity and design.

摘要: 尽管深度学习模型在图像语义分割方面取得了巨大的进步，但它们通常需要大量的注释示例，并且越来越多的注意力被转移到像少镜头学习(FSL)这样的问题环境中，其中只需要少量的注释就可以概括到新的类。这在医学领域中尤其常见，在医学领域中，密集像素级注释的获取成本很高。在本文中，我们提出了正则化的原型神经常微分方程(R-PNODE)，该方法利用神经节点的固有特性，通过额外的聚类和一致性损失来辅助和增强器官的少镜头分割(FSS)。R-PNODE约束支持和查询来自同一类的特征在表示空间中更接近，从而提高了现有基于卷积神经网络(CNN)的FSS方法的性能。我们进一步证明，虽然许多现有的基于Deep CNN的方法往往非常容易受到对抗性攻击，但R-PNODE对一系列此类攻击表现出更强的对抗性。我们用三个公开可用的多器官分割数据集在域内和跨域的FSS环境中进行了实验，以证明我们方法的有效性。此外，我们在不同的环境下对七种常用的对抗性攻击进行了实验，以验证R-PNODE的健壮性。R-PNODE的表现远远超过FSS的基线，并在各种强度和设计的攻击中显示出卓越的性能。



## **26. SNAP: Efficient Extraction of Private Properties with Poisoning**

Snap：高效提取带有毒物的私有财产 cs.LG

27 pages, 13 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12348v1)

**Authors**: Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman

**Abstracts**: Property inference attacks allow an adversary to extract global properties of the training dataset from a machine learning model. Such attacks have privacy implications for data owners who share their datasets to train machine learning models. Several existing approaches for property inference attacks against deep neural networks have been proposed, but they all rely on the attacker training a large number of shadow models, which induces large computational overhead.   In this paper, we consider the setting of property inference attacks in which the attacker can poison a subset of the training dataset and query the trained target model. Motivated by our theoretical analysis of model confidences under poisoning, we design an efficient property inference attack, SNAP, which obtains higher attack success and requires lower amounts of poisoning than the state-of-the-art poisoning-based property inference attack by Mahloujifar et al. For example, on the Census dataset, SNAP achieves 34% higher success rate than Mahloujifar et al. while being 56.5x faster. We also extend our attack to determine if a certain property is present at all in training, and estimate the exact proportion of a property of interest efficiently. We evaluate our attack on several properties of varying proportions from four datasets, and demonstrate SNAP's generality and effectiveness.

摘要: 属性推理攻击允许对手从机器学习模型中提取训练数据集的全局属性。此类攻击对共享数据集以训练机器学习模型的数据所有者具有隐私影响。已有的几种针对深层神经网络的属性推理攻击方法都依赖于攻击者训练大量的影子模型，导致计算开销较大。在本文中，我们考虑了属性推理攻击的设置，在该攻击中，攻击者可以毒化训练数据集的子集并查询训练的目标模型。在对中毒下的模型可信度进行理论分析的基础上，设计了一种高效的属性推理攻击SNAP，它比MahLoujifar等人提出的基于中毒的属性推理攻击具有更高的攻击成功率和更低的投毒量。例如，在人口普查数据集上，SNAP的成功率比MahLoujifar等人高34%。同时速度提高了56.5倍。我们还扩展了我们的攻击，以确定在训练中是否存在某个属性，并有效地估计感兴趣的属性的确切比例。我们对来自四个数据集的几个不同比例的属性进行了评估，并证明了SNAP的通用性和有效性。



## **27. Semantic Preserving Adversarial Attack Generation with Autoencoder and Genetic Algorithm**

基于自动编码和遗传算法的语义保持敌意攻击生成 cs.LG

8 pages conference paper, accepted for publication in IEEE GLOBECOM  2022

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12230v1)

**Authors**: Xinyi Wang, Simon Yusuf Enoch, Dong Seong Kim

**Abstracts**: Widely used deep learning models are found to have poor robustness. Little noises can fool state-of-the-art models into making incorrect predictions. While there is a great deal of high-performance attack generation methods, most of them directly add perturbations to original data and measure them using L_p norms; this can break the major structure of data, thus, creating invalid attacks. In this paper, we propose a black-box attack, which, instead of modifying original data, modifies latent features of data extracted by an autoencoder; then, we measure noises in semantic space to protect the semantics of data. We trained autoencoders on MNIST and CIFAR-10 datasets and found optimal adversarial perturbations using a genetic algorithm. Our approach achieved a 100% attack success rate on the first 100 data of MNIST and CIFAR-10 datasets with less perturbation than FGSM.

摘要: 广泛使用的深度学习模型具有较差的稳健性。微小的噪音可以愚弄最先进的模型做出错误的预测。虽然有很多高性能的攻击生成方法，但它们大多直接对原始数据添加扰动，并使用L_p范数进行度量，这会破坏数据的主要结构，从而产生无效攻击。本文提出了一种黑盒攻击，它不修改原始数据，而是修改由自动编码器提取的数据的潜在特征，然后在语义空间中测量噪声来保护数据的语义。我们在MNIST和CIFAR-10数据集上训练自动编码器，并使用遗传算法找到最优的对抗性扰动。我们的方法在MNIST和CIFAR-10数据集的前100个数据上取得了100%的攻击成功率，并且比FGSM具有更小的扰动。



## **28. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12216v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **29. Automatic Mapping of Unstructured Cyber Threat Intelligence: An Experimental Study**

非结构化网络威胁情报自动测绘的实验研究 cs.CR

2022 IEEE 33rd International Symposium on Software Reliability  Engineering (ISSRE)

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.12144v1)

**Authors**: Vittorio Orbinato, Mariarosaria Barbaraci, Roberto Natella, Domenico Cotroneo

**Abstracts**: Proactive approaches to security, such as adversary emulation, leverage information about threat actors and their techniques (Cyber Threat Intelligence, CTI). However, most CTI still comes in unstructured forms (i.e., natural language), such as incident reports and leaked documents. To support proactive security efforts, we present an experimental study on the automatic classification of unstructured CTI into attack techniques using machine learning (ML). We contribute with two new datasets for CTI analysis, and we evaluate several ML models, including both traditional and deep learning-based ones. We present several lessons learned about how ML can perform at this task, which classifiers perform best and under which conditions, which are the main causes of classification errors, and the challenges ahead for CTI analysis.

摘要: 主动的安全方法，如对手模拟，利用有关威胁参与者及其技术的信息(网络威胁情报，CTI)。然而，大多数CTI仍然是非结构化的形式(即自然语言)，例如事件报告和泄露的文件。为了支持主动安全工作，我们提出了一项使用机器学习(ML)将非结构化CTI自动分类为攻击技术的实验研究。我们为CTI分析提供了两个新的数据集，并评估了几个ML模型，包括传统的和基于深度学习的模型。我们提供了几个经验教训，关于ML如何在这项任务中执行，哪些分类器在哪些条件下表现最好，哪些是分类错误的主要原因，以及CTI分析未来的挑战。



## **30. ECG-ATK-GAN: Robustness against Adversarial Attacks on ECGs using Conditional Generative Adversarial Networks**

ECG-ATK-GAN：使用条件生成对抗网络对ECG的对抗攻击的稳健性 eess.SP

Accepted to MICCAI2022 Applications of Medical AI (AMAI) Workshop

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2110.09983v3)

**Authors**: Khondker Fariha Hossain, Sharif Amit Kamran, Alireza Tavakkoli, Xingjun Ma

**Abstracts**: Automating arrhythmia detection from ECG requires a robust and trusted system that retains high accuracy under electrical disturbances. Many machine learning approaches have reached human-level performance in classifying arrhythmia from ECGs. However, these architectures are vulnerable to adversarial attacks, which can misclassify ECG signals by decreasing the model's accuracy. Adversarial attacks are small crafted perturbations injected in the original data which manifest the out-of-distribution shifts in signal to misclassify the correct class. Thus, security concerns arise for false hospitalization and insurance fraud abusing these perturbations. To mitigate this problem, we introduce the first novel Conditional Generative Adversarial Network (GAN), robust against adversarial attacked ECG signals and retaining high accuracy. Our architecture integrates a new class-weighted objective function for adversarial perturbation identification and new blocks for discerning and combining out-of-distribution shifts in signals in the learning process for accurately classifying various arrhythmia types. Furthermore, we benchmark our architecture on six different white and black-box attacks and compare them with other recently proposed arrhythmia classification models on two publicly available ECG arrhythmia datasets. The experiment confirms that our model is more robust against such adversarial attacks for classifying arrhythmia with high accuracy.

摘要: 从心电中自动检测心律失常需要一个健壮和可信的系统，在电子干扰下保持高精度。许多机器学习方法在区分心律失常和心电信号方面已经达到了人类的水平。然而，这些体系结构容易受到敌意攻击，这些攻击可能会降低模型的准确性，从而导致心电信号的误分类。对抗性攻击是注入到原始数据中的小的精心设计的扰动，它显示了信号的不分布转移，以错误地分类正确的类别。因此，出现了对虚假住院和滥用这些扰动的保险欺诈的安全担忧。为了缓解这一问题，我们引入了第一个新的条件生成对抗网络(GAN)，它对对手攻击的心电信号具有鲁棒性，并保持了较高的准确率。我们的体系结构集成了一个新的类别加权目标函数来识别对抗性扰动，以及新的块来识别和组合学习过程中信号的非分布变化，以准确地分类各种心律失常类型。此外，我们在六种不同的白盒和黑盒攻击上测试了我们的体系结构，并将它们与最近提出的其他心律失常分类模型在两个公开可用的心电心律失常数据集上进行了比较。实验证明，该模型对心律失常的分类具有较强的鲁棒性，分类准确率较高。



## **31. A Perturbation Resistant Transformation and Classification System for Deep Neural Networks**

一种抗扰动的深度神经网络变换与分类系统 cs.CV

12 pages, 4 figures

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.11839v1)

**Authors**: Nathaniel Dean, Dilip Sarkar

**Abstracts**: Deep convolutional neural networks accurately classify a diverse range of natural images, but may be easily deceived when designed, imperceptible perturbations are embedded in the images. In this paper, we design a multi-pronged training, input transformation, and image ensemble system that is attack agnostic and not easily estimated. Our system incorporates two novel features. The first is a transformation layer that computes feature level polynomial kernels from class-level training data samples and iteratively updates input image copies at inference time based on their feature kernel differences to create an ensemble of transformed inputs. The second is a classification system that incorporates the prediction of the undefended network with a hard vote on the ensemble of filtered images. Our evaluations on the CIFAR10 dataset show our system improves the robustness of an undefended network against a variety of bounded and unbounded white-box attacks under different distance metrics, while sacrificing little accuracy on clean images. Against adaptive full-knowledge attackers creating end-to-end attacks, our system successfully augments the existing robustness of adversarially trained networks, for which our methods are most effectively applied.

摘要: 深层卷积神经网络可以准确地对多种自然图像进行分类，但在设计时很容易被欺骗，图像中嵌入了难以察觉的扰动。在本文中，我们设计了一个多管齐下的训练、输入变换和图像集成系统，该系统与攻击无关，不易估计。我们的系统有两个新颖的特点。第一个是变换层，它从类级训练数据样本计算特征级多项式核，并在推理时基于它们的特征核差异迭代地更新输入图像副本，以创建变换输入的集成。第二种是一种分类系统，它结合了对无防御网络的预测和对过滤图像集合的硬投票。我们在CIFAR10数据集上的评估表明，我们的系统提高了无防御网络在不同距离度量下对各种有界和无界白盒攻击的健壮性，而对干净图像的准确性几乎没有牺牲。针对自适应全知识攻击者制造的端到端攻击，我们的系统成功地增强了对手训练网络的现有健壮性，我们的方法在这些网络中得到了最有效的应用。



## **32. A New Kind of Adversarial Example**

一种新的对抗性例证 cs.CV

**SubmitDate**: 2022-08-25    [paper-pdf](http://arxiv.org/pdf/2208.02430v2)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.

摘要: 几乎所有的对抗性攻击都是为了给图像添加一个难以察觉的扰动，以愚弄模型。在这里，我们考虑的是相反的情况，即可以愚弄人类但不能愚弄模型的对抗性例子。一个足够大和可感知的扰动被添加到图像中，使得模型保持其原始决定，而如果被迫做出决定(或者选择根本不决定)，人类很可能会犯错误。现有的有针对性的攻击可以重新制定，以合成这种对抗性的例子。我们提出的名为NKE的攻击在本质上类似于愚弄图像，但由于它使用了梯度下降而不是进化算法，因此效率更高。它还为敌方脆弱性问题提供了一个新的统一视角。在MNIST和CIFAR-10数据集上的实验结果表明，我们的攻击在欺骗深度神经网络方面是相当有效的。代码可在https://github.com/aliborji/NKE.上找到



## **33. Attacking Neural Binary Function Detection**

攻击神经二进制函数检测 cs.CR

18 pages

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11667v1)

**Authors**: Joshua Bundt, Michael Davinroy, Ioannis Agadakos, Alina Oprea, William Robertson

**Abstracts**: Binary analyses based on deep neural networks (DNNs), or neural binary analyses (NBAs), have become a hotly researched topic in recent years. DNNs have been wildly successful at pushing the performance and accuracy envelopes in the natural language and image processing domains. Thus, DNNs are highly promising for solving binary analysis problems that are typically hard due to a lack of complete information resulting from the lossy compilation process. Despite this promise, it is unclear that the prevailing strategy of repurposing embeddings and model architectures originally developed for other problem domains is sound given the adversarial contexts under which binary analysis often operates.   In this paper, we empirically demonstrate that the current state of the art in neural function boundary detection is vulnerable to both inadvertent and deliberate adversarial attacks. We proceed from the insight that current generation NBAs are built upon embeddings and model architectures intended to solve syntactic problems. We devise a simple, reproducible, and scalable black-box methodology for exploring the space of inadvertent attacks - instruction sequences that could be emitted by common compiler toolchains and configurations - that exploits this syntactic design focus. We then show that these inadvertent misclassifications can be exploited by an attacker, serving as the basis for a highly effective black-box adversarial example generation process. We evaluate this methodology against two state-of-the-art neural function boundary detectors: XDA and DeepDi. We conclude with an analysis of the evaluation data and recommendations for how future research might avoid succumbing to similar attacks.

摘要: 基于深度神经网络(DNN)或神经二进制分析(NBAs)的二进制分析是近年来研究的热点。在自然语言和图像处理领域，DNN在提高性能和准确率方面取得了巨大的成功。因此，DNN在解决二进制分析问题方面非常有希望，这些问题通常很难解决，因为有损编译过程导致缺乏完整的信息。尽管有这样的承诺，但考虑到二元分析经常在敌对的环境下运行，目前尚不清楚重新调整最初为其他问题领域开发的嵌入和模型体系结构的用途的流行策略是否合理。在这篇文章中，我们经验地证明，神经功能边界检测的当前技术水平容易受到无意和故意的敌意攻击。我们的出发点是，当前一代的NBA是建立在旨在解决语法问题的嵌入和模型体系结构之上的。我们设计了一种简单、可重复和可扩展的黑盒方法，用于探索意外攻击的空间-可能由常见编译器工具链和配置发出的指令序列-利用了这一语法设计重点。然后，我们展示了这些无意的错误分类可以被攻击者利用，作为高效的黑盒对抗性示例生成过程的基础。我们用两种最先进的神经功能边界检测器：XDA和DeepDi对该方法进行了评估。最后，我们对评估数据进行了分析，并就未来的研究如何避免屈服于类似的攻击提出了建议。



## **34. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2103.09151v3)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As the research in deep neural networks advances, deep convolutional networks become feasible for automated driving tasks. There is an emerging trend of employing end-to-end models in the automation of driving tasks. However, previous research unveils that deep neural networks are vulnerable to adversarial attacks in classification tasks. While for regression tasks such as autonomous driving, the effect of these attacks remains rarely explored. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving model takes an image as input and outputs the steering angle. Our attacks can manipulate the behavior of the autonomous driving system only by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. This research aims to raise concerns over applications of end-to-end models in safety-critical systems.

摘要: 随着深度神经网络研究的深入，深度卷积网络在自动驾驶任务中变得可行。在驾驶任务的自动化中使用端到端模型是一种新兴的趋势。然而，以往的研究表明，深度神经网络在分类任务中容易受到敌意攻击。而对于自动驾驶等回归任务，这些攻击的影响仍然很少被研究。在本研究中，我们设计了两种针对端到端自动驾驶系统的白盒针对性攻击。驾驶模型以图像为输入，输出转向角。我们的攻击只能通过干扰输入图像来操纵自动驾驶系统的行为。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。这项研究旨在引起人们对端到端模型在安全关键系统中的应用的关注。



## **35. Unrestricted Black-box Adversarial Attack Using GAN with Limited Queries**

基于有限查询GAN的无限制黑盒对抗性攻击 cs.CV

Accepted to the ECCV 2022 Workshop on Adversarial Robustness in the  Real World

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11613v1)

**Authors**: Dongbin Na, Sangwoo Ji, Jong Kim

**Abstracts**: Adversarial examples are inputs intentionally generated for fooling a deep neural network. Recent studies have proposed unrestricted adversarial attacks that are not norm-constrained. However, the previous unrestricted attack methods still have limitations to fool real-world applications in a black-box setting. In this paper, we present a novel method for generating unrestricted adversarial examples using GAN where an attacker can only access the top-1 final decision of a classification model. Our method, Latent-HSJA, efficiently leverages the advantages of a decision-based attack in the latent space and successfully manipulates the latent vectors for fooling the classification model.   With extensive experiments, we demonstrate that our proposed method is efficient in evaluating the robustness of classification models with limited queries in a black-box setting. First, we demonstrate that our targeted attack method is query-efficient to produce unrestricted adversarial examples for a facial identity recognition model that contains 307 identities. Then, we demonstrate that the proposed method can also successfully attack a real-world celebrity recognition service.

摘要: 对抗性例子是为愚弄深度神经网络而故意生成的输入。最近的研究提出了不受规范约束的无限制对抗性攻击。然而，以前的不受限制的攻击方法仍然有局限性，无法在黑盒设置中愚弄现实世界的应用程序。在本文中，我们提出了一种利用GAN生成无限制敌意实例的新方法，其中攻击者只能访问分类模型的TOP-1最终决策。该方法有效地利用了基于决策的攻击在潜在空间中的优势，并成功地操纵了潜在向量来欺骗分类模型。通过大量的实验，我们证明了我们提出的方法在黑盒环境下评估有限查询的分类模型的稳健性是有效的。首先，我们证明了我们的定向攻击方法是查询高效的，可以为包含307个身份的面部身份识别模型生成不受限制的对抗性示例。然后，我们证明了所提出的方法也可以成功地攻击真实世界的名人识别服务。



## **36. Robustness of the Tangle 2.0 Consensus**

Tangle 2.0共识的健壮性 cs.DC

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.08254v2)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.

摘要: 本文研究了Tange2.0一致性协议在拜占庭环境下的性能。我们使用了一个基于代理的仿真模型，该模型结合了Tangel2.0共识协议的主要特征。实验结果表明，Tangel2.0协议对诱饵切换攻击具有较强的鲁棒性，达到了敌手33%投票权重的理论上限。我们进一步证明了Tange2.0中的普通硬币机制对于抵抗强大的对手是必要的。实验结果表明，该协议在典型场景下可以达到1s左右的确认时间，且无冲突事务的确认时间不受冲突的影响。



## **37. LPF-Defense: 3D Adversarial Defense based on Frequency Analysis**

LPF-Defense：基于频率分析的三维对抗性防御 cs.CV

15 pages, 7 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2202.11287v2)

**Authors**: Hanieh Naderi, Kimia Noorbakhsh, Arian Etemadi, Shohreh Kasaei

**Abstracts**: Although 3D point cloud classification has recently been widely deployed in different application scenarios, it is still very vulnerable to adversarial attacks. This increases the importance of robust training of 3D models in the face of adversarial attacks. Based on our analysis on the performance of existing adversarial attacks, more adversarial perturbations are found in the mid and high-frequency components of input data. Therefore, by suppressing the high-frequency content in the training phase, the models robustness against adversarial examples is improved. Experiments showed that the proposed defense method decreases the success rate of six attacks on PointNet, PointNet++ ,, and DGCNN models. In particular, improvements are achieved with an average increase of classification accuracy by 3.8 % on drop100 attack and 4.26 % on drop200 attack compared to the state-of-the-art methods. The method also improves models accuracy on the original dataset compared to other available methods.

摘要: 虽然三维点云分类最近在不同的应用场景中得到了广泛的部署，但它仍然非常容易受到对抗性攻击。这增加了在面对对手攻击时对3D模型进行稳健训练的重要性。基于对现有对抗性攻击性能的分析，在输入数据的中高频成分中发现了更多的对抗性扰动。因此，通过抑制训练阶段的高频内容，提高了模型对敌意样本的稳健性。实验表明，该防御方法降低了对PointNet、PointNet++、和DGCNN模型的6次攻击的成功率。特别是，与现有方法相比，Drop100攻击的分类准确率平均提高了3.8%，Drop200攻击的分类准确率平均提高了4.26%。与其他可用的方法相比，该方法还提高了原始数据集上的模型精度。



## **38. Trace and Detect Adversarial Attacks on CNNs using Feature Response Maps**

使用特征响应映射跟踪和检测对CNN的敌意攻击 cs.CV

13 pages, 6 figures

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11436v1)

**Authors**: Mohammadreza Amirian, Friedhelm Schwenker, Thilo Stadelmann

**Abstracts**: The existence of adversarial attacks on convolutional neural networks (CNN) questions the fitness of such models for serious applications. The attacks manipulate an input image such that misclassification is evoked while still looking normal to a human observer -- they are thus not easily detectable. In a different context, backpropagated activations of CNN hidden layers -- "feature responses" to a given input -- have been helpful to visualize for a human "debugger" what the CNN "looks at" while computing its output. In this work, we propose a novel detection method for adversarial examples to prevent attacks. We do so by tracking adversarial perturbations in feature responses, allowing for automatic detection using average local spatial entropy. The method does not alter the original network architecture and is fully human-interpretable. Experiments confirm the validity of our approach for state-of-the-art attacks on large-scale models trained on ImageNet.

摘要: 对卷积神经网络(CNN)的敌意攻击的存在质疑了这种模型在严肃应用中的适用性。这些攻击操作输入图像，从而在人类观察者看来仍然正常的情况下引发错误分类--因此它们不容易被检测到。在另一种情况下，对CNN隐藏层的反向传播激活--对给定输入的“特征响应”--有助于人类“调试者”在计算其输出时可视化CNN“所看到的”。在这项工作中，我们提出了一种新的对抗性实例检测方法来防止攻击。我们通过跟踪特征响应中的对抗性扰动来实现这一点，允许使用平均局部空间熵进行自动检测。该方法不改变原有的网络体系结构，完全是人类可理解的。实验证实了该方法对基于ImageNet训练的大规模模型进行最先进攻击的有效性。



## **39. Towards an Awareness of Time Series Anomaly Detection Models' Adversarial Vulnerability**

认识时间序列异常检测模型的对抗性漏洞 cs.LG

Part of Proceedings of the 31st ACM International Conference on  Information and Knowledge Management (CIKM '22)

**SubmitDate**: 2022-08-24    [paper-pdf](http://arxiv.org/pdf/2208.11264v1)

**Authors**: Shahroz Tariq, Binh M. Le, Simon S. Woo

**Abstracts**: Time series anomaly detection is extensively studied in statistics, economics, and computer science. Over the years, numerous methods have been proposed for time series anomaly detection using deep learning-based methods. Many of these methods demonstrate state-of-the-art performance on benchmark datasets, giving the false impression that these systems are robust and deployable in many practical and industrial real-world scenarios. In this paper, we demonstrate that the performance of state-of-the-art anomaly detection methods is degraded substantially by adding only small adversarial perturbations to the sensor data. We use different scoring metrics such as prediction errors, anomaly, and classification scores over several public and private datasets ranging from aerospace applications, server machines, to cyber-physical systems in power plants. Under well-known adversarial attacks from Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) methods, we demonstrate that state-of-the-art deep neural networks (DNNs) and graph neural networks (GNNs) methods, which claim to be robust against anomalies and have been possibly integrated in real-life systems, have their performance drop to as low as 0%. To the best of our understanding, we demonstrate, for the first time, the vulnerabilities of anomaly detection systems against adversarial attacks. The overarching goal of this research is to raise awareness towards the adversarial vulnerabilities of time series anomaly detectors.

摘要: 时间序列异常检测在统计学、经济学和计算机科学中都有广泛的研究。多年来，已经提出了许多基于深度学习的时间序列异常检测方法。其中许多方法在基准数据集上展示了最先进的性能，给人一种错误的印象，即这些系统在许多实际和工业真实世界的场景中都是健壮的和可部署的。在这篇文章中，我们证明了最新的异常检测方法的性能在很大程度上由于只向传感器数据添加小的对抗性扰动而降低。我们在从航空航天应用程序、服务器机器到发电厂的网络物理系统等多个公共和私有数据集上使用不同的评分指标，例如预测误差、异常和分类分数。在已知的快速梯度符号方法(FGSM)和投影梯度下降(PGD)方法的敌意攻击下，我们证明了最新的深度神经网络(DNNS)和图神经网络(GNNS)方法的性能下降到了0%，这两种方法声称对异常具有健壮性，并可能集成到现实系统中。据我们所知，我们第一次展示了异常检测系统对对手攻击的脆弱性。这项研究的首要目标是提高人们对时间序列异常检测器的对抗性脆弱性的认识。



## **40. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

ObfuNAS：一种基于神经结构搜索的DNN混淆方法 cs.CR

9 pages

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.08569v2)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.

摘要: 恶意体系结构提取已经成为深度神经网络(DNN)安全的一个重要问题。作为防御，体系结构混淆被提出将受害者DNN重新映射到不同的体系结构。尽管如此，我们观察到，只要提取一个混淆的DNN体系结构，攻击者仍然可以高性能(例如，准确性)重新训练替代模型，使得混淆技术无效。为了缓解这一未被充分挖掘的漏洞，我们提出了ObfuNAS，它将DNN体系结构的混淆转化为神经体系结构搜索(NAS)问题。ObfuNAS结合了函数保留混淆策略，确保了混淆后的DNN架构只能达到比受害者更低的准确率。我们使用NAS-BENCH-101和NAS-BENCH-301等开源架构数据集验证了ObfuNAS的性能。实验结果表明，ObfuNAS能够在给定的FLOPS约束下成功地找到受害者模型的最优掩码，使得仅需0.14倍FLOPS开销的攻击者的推理准确率降低2.6%。代码可从以下网址获得：https://github.com/Tongzhou0101/ObfuNAS.



## **41. Auditing Membership Leakages of Multi-Exit Networks**

审计多出口网络的成员泄漏 cs.CR

Accepted by CCS 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.11180v1)

**Authors**: Zheng Li, Yiyong Liu, Xinlei He, Ning Yu, Michael Backes, Yang Zhang

**Abstracts**: Relying on the fact that not all inputs require the same amount of computation to yield a confident prediction, multi-exit networks are gaining attention as a prominent approach for pushing the limits of efficient deployment. Multi-exit networks endow a backbone model with early exits, allowing to obtain predictions at intermediate layers of the model and thus save computation time and/or energy. However, current various designs of multi-exit networks are only considered to achieve the best trade-off between resource usage efficiency and prediction accuracy, the privacy risks stemming from them have never been explored. This prompts the need for a comprehensive investigation of privacy risks in multi-exit networks.   In this paper, we perform the first privacy analysis of multi-exit networks through the lens of membership leakages. In particular, we first leverage the existing attack methodologies to quantify the multi-exit networks' vulnerability to membership leakages. Our experimental results show that multi-exit networks are less vulnerable to membership leakages and the exit (number and depth) attached to the backbone model is highly correlated with the attack performance. Furthermore, we propose a hybrid attack that exploits the exit information to improve the performance of existing attacks. We evaluate membership leakage threat caused by our hybrid attack under three different adversarial setups, ultimately arriving at a model-free and data-free adversary. These results clearly demonstrate that our hybrid attacks are very broadly applicable, thereby the corresponding risks are much more severe than shown by existing membership inference attacks. We further present a defense mechanism called TimeGuard specifically for multi-exit networks and show that TimeGuard mitigates the newly proposed attacks perfectly.

摘要: 由于并非所有的输入都需要相同的计算量才能得出可信的预测，多出口网络作为一种突破有效部署极限的重要方法正受到人们的关注。多出口网络为主干模型提供了早期出口，允许在模型的中间层获得预测，从而节省计算时间和/或能量。然而，目前各种设计的多出口网络只考虑在资源使用效率和预测精度之间实现最佳折衷，并没有探讨它们所带来的隐私风险。这促使需要对多出口网络中的隐私风险进行全面调查。本文首次从成员泄漏的角度对多出口网络进行了隐私分析。特别是，我们首先利用现有的攻击方法来量化多出口网络对成员泄漏的脆弱性。我们的实验结果表明，多出口网络不太容易受到成员泄漏的影响，并且连接到主干模型的出口(数量和深度)与攻击性能高度相关。此外，我们还提出了一种利用出口信息的混合攻击来提高现有攻击的性能。我们评估了我们的混合攻击在三种不同的对手设置下所造成的成员泄漏威胁，最终得出了一个无模型、无数据的对手。这些结果清楚地表明，我们的混合攻击具有非常广泛的适用性，因此相应的风险比现有的成员推理攻击要严重得多。我们进一步提出了一种专门针对多出口网络的防御机制TimeGuard，并证明了TimeGuard可以很好地缓解新提出的攻击。



## **42. Adversarial Speaker Distillation for Countermeasure Model on Automatic Speaker Verification**

说话人自动确认对抗模型中的对抗性说话人提取 cs.SD

Accepted by ISCA SPSC 2022

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2203.17031v5)

**Authors**: Yen-Lun Liao, Xuanjun Chen, Chung-Che Wang, Jyh-Shing Roger Jang

**Abstracts**: The countermeasure (CM) model is developed to protect ASV systems from spoof attacks and prevent resulting personal information leakage in Automatic Speaker Verification (ASV) system. Based on practicality and security considerations, the CM model is usually deployed on edge devices, which have more limited computing resources and storage space than cloud-based systems, confining the model size under a limitation. To better trade off the CM model sizes and performance, we proposed an adversarial speaker distillation method, which is an improved version of knowledge distillation method combined with generalized end-to-end (GE2E) pre-training and adversarial fine-tuning. In the evaluation phase of the ASVspoof 2021 Logical Access task, our proposed adversarial speaker distillation ResNetSE (ASD-ResNetSE) model reaches 0.2695 min t-DCF and 3.54\% EER. ASD-ResNetSE only used 22.5\% of parameters and 19.4\% of multiply and accumulate operands of ResNetSE model.

摘要: 在自动说话人确认(ASV)系统中，为了保护ASV系统免受欺骗攻击，并防止由此导致的个人信息泄露，提出了对策(CM)模型。基于实用性和安全性考虑，CM模型通常部署在边缘设备上，与基于云的系统相比，边缘设备具有更有限的计算资源和存储空间，从而将模型大小限制在一定范围内。为了更好地权衡CM模型的规模和性能，我们提出了一种对抗性说话人蒸馏方法，它是一种改进的知识蒸馏方法，结合了广义端到端(GE2E)预训练和对抗性微调。在ASVspoof 2021逻辑访问任务的评估阶段，我们提出的对抗性说话人蒸馏ResNetSE(ASD-ResNetSE)模型达到了0.2695分钟的t-DCF和3.54EER。ASD-ResNetSE只使用了ResNetSE模型的22.5个参数和19.4个乘法和累加操作数。



## **43. Privacy Enhancement for Cloud-Based Few-Shot Learning**

增强基于云的极少机会学习的隐私 cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2205.07864v2)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.

摘要: 对于精确的模型，少镜头学习需要较少的数据，在许多应用领域都表现出了健壮性和通用性。然而，在不受信任的环境中部署极少的模型可能会引起隐私问题，例如，可能会破坏用户提供的数据的隐私的攻击或对手。通过建立一种新的隐私保护嵌入空间来保护数据隐私并保持模型的准确性，研究了在不可信环境(如云)下的少机会学习的隐私增强问题。研究了模糊、像素化、高斯噪声、差分隐私像素化等图像隐私保护方法对少镜头图像分类的影响，提出了一种通过联合损失学习隐私保护表示的方法。实证结果表明，对于隐私增强型少镜头学习，隐私性能与性能之间的权衡是如何协商的。



## **44. A Comprehensive Study of Real-Time Object Detection Networks Across Multiple Domains: A Survey**

跨域实时目标检测网络研究综述 cs.CV

Published in Transactions on Machine Learning Research (TMLR) with  Survey Certification

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10895v1)

**Authors**: Elahe Arani, Shruthi Gowda, Ratnajit Mukherjee, Omar Magdy, Senthilkumar Kathiresan, Bahram Zonooz

**Abstracts**: Deep neural network based object detectors are continuously evolving and are used in a multitude of applications, each having its own set of requirements. While safety-critical applications need high accuracy and reliability, low-latency tasks need resource and energy-efficient networks. Real-time detectors, which are a necessity in high-impact real-world applications, are continuously proposed, but they overemphasize the improvements in accuracy and speed while other capabilities such as versatility, robustness, resource and energy efficiency are omitted. A reference benchmark for existing networks does not exist, nor does a standard evaluation guideline for designing new networks, which results in ambiguous and inconsistent comparisons. We, thus, conduct a comprehensive study on multiple real-time detectors (anchor-, keypoint-, and transformer-based) on a wide range of datasets and report results on an extensive set of metrics. We also study the impact of variables such as image size, anchor dimensions, confidence thresholds, and architecture layers on the overall performance. We analyze the robustness of detection networks against distribution shifts, natural corruptions, and adversarial attacks. Also, we provide a calibration analysis to gauge the reliability of the predictions. Finally, to highlight the real-world impact, we conduct two unique case studies, on autonomous driving and healthcare applications. To further gauge the capability of networks in critical real-time applications, we report the performance after deploying the detection networks on edge devices. Our extensive empirical study can act as a guideline for the industrial community to make an informed choice on the existing networks. We also hope to inspire the research community towards a new direction in the design and evaluation of networks that focuses on a bigger and holistic overview for a far-reaching impact.

摘要: 基于深度神经网络的目标检测器正在不断发展，并在许多应用中使用，每个应用都有其自己的一组要求。安全关键型应用程序需要高准确性和可靠性，而低延迟任务则需要资源和能效高的网络。实时检测器在高影响的现实世界应用中是必不可少的，不断被提出，但它们过分强调精度和速度的提高，而忽略了其他功能，如通用性、健壮性、资源和能源效率。现有网络没有参考基准，也没有设计新网络的标准评估指南，这导致比较不明确和不一致。因此，我们在广泛的数据集上对多个实时检测器(锚点、关键点和变压器)进行了全面的研究，并报告了一组广泛的指标的结果。我们还研究了图像大小、锚点维度、置信度阈值和架构层等变量对整体性能的影响。我们分析了检测网络对分布偏移、自然破坏和敌意攻击的稳健性。此外，我们还提供了校准分析，以衡量预测的可靠性。最后，为了突出现实世界的影响，我们进行了两个独特的案例研究，分别是自动驾驶和医疗保健应用。为了进一步衡量网络在关键实时应用中的能力，我们报告了在边缘设备上部署检测网络后的性能。我们广泛的实证研究可以为工业界在现有网络上做出明智的选择提供指导。我们还希望激励研究界在网络的设计和评估方面朝着一个新的方向前进，专注于更大和更全面的概览，以产生深远的影响。



## **45. Transferability Ranking of Adversarial Examples**

对抗性例证的可转移性排名 cs.LG

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10878v1)

**Authors**: Mosh Levy, Yuval Elovici, Yisroel Mirsky

**Abstracts**: Adversarial examples can be used to maliciously and covertly change a model's prediction. It is known that an adversarial example designed for one model can transfer to other models as well. This poses a major threat because it means that attackers can target systems in a blackbox manner.   In the domain of transferability, researchers have proposed ways to make attacks more transferable and to make models more robust to transferred examples. However, to the best of our knowledge, there are no works which propose a means for ranking the transferability of an adversarial example in the perspective of a blackbox attacker. This is an important task because an attacker is likely to use only a select set of examples, and therefore will want to select the samples which are most likely to transfer.   In this paper we suggest a method for ranking the transferability of adversarial examples without access to the victim's model. To accomplish this, we define and estimate the expected transferability of a sample given limited information about the victim. We also explore practical scenarios: where the adversary can select the best sample to attack and where the adversary must use a specific sample but can choose different perturbations. Through our experiments, we found that our ranking method can increase an attacker's success rate by up to 80% compared to the baseline (random selection without ranking).

摘要: 敌意的例子可能被用来恶意和秘密地改变模型的预测。众所周知，为一种模型设计的对抗性例子也可以转移到其他模型上。这构成了一个重大威胁，因为这意味着攻击者可以以黑盒方式将系统作为目标。在可转移性领域，研究人员已经提出了一些方法来使攻击更具可转移性，并使模型对已转移的示例更健壮。然而，就我们所知，目前还没有文献提出从黑盒攻击者的角度对对抗性例子的可转移性进行排序的方法。这是一项重要的任务，因为攻击者可能只使用一组选定的示例，因此希望选择最有可能传输的样本。在这篇文章中，我们提出了一种在不使用受害者模型的情况下对对抗性例子的可转移性进行排序的方法。为了实现这一点，我们定义并估计了给定关于受害者的有限信息的样本的预期可转移性。我们还探索了实际场景：其中对手可以选择最佳样本进行攻击，以及对手必须使用特定样本但可以选择不同的扰动。通过实验，我们发现我们的排序方法可以使攻击者的成功率比基线(随机选择而不排序)提高高达80%。



## **46. Complete Traceability Multimedia Fingerprinting Codes Resistant to Averaging Attack and Adversarial Noise with Optimal Rate**

具有最优码率的抗平均攻击和对抗噪声的完全可追溯性多媒体指纹编码 cs.IT

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2108.09015v4)

**Authors**: Ilya Vorobyev

**Abstracts**: In this paper we consider complete traceability multimedia fingerprinting codes resistant to averaging attacks and adversarial noise. Recently it was shown that there are no such codes for the case of an arbitrary linear attack. However, for the case of averaging attacks complete traceability multimedia fingerprinting codes of exponential cardinality resistant to constant adversarial noise were constructed in 2020 by Egorova et al. We continue this work and provide an improved lower bound on the rate of these codes.

摘要: 在本文中，我们考虑了完全可追踪性多媒体指纹码，它能抵抗平均攻击和对抗噪声。最近的研究表明，对于任意线性攻击的情况，不存在这样的码。然而，对于平均攻击的情况，Egorova等人于2020年构造了抵抗恒定对抗性噪声的指数基数完全可追溯性多媒体指纹码。我们继续这项工作，并提供了这些码率的一个改进的下界。



## **47. Evaluating Machine Unlearning via Epistemic Uncertainty**

基于认知不确定性的机器遗忘评估 cs.LG

Rejected at ECML 2021. Even though the paper was rejected, we want to  "publish" it on arxiv, since we believe that it is nevertheless interesting  to investigate the connections between unlearning and uncertainty

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10836v1)

**Authors**: Alexander Becker, Thomas Liebig

**Abstracts**: There has been a growing interest in Machine Unlearning recently, primarily due to legal requirements such as the General Data Protection Regulation (GDPR) and the California Consumer Privacy Act. Thus, multiple approaches were presented to remove the influence of specific target data points from a trained model. However, when evaluating the success of unlearning, current approaches either use adversarial attacks or compare their results to the optimal solution, which usually incorporates retraining from scratch. We argue that both ways are insufficient in practice. In this work, we present an evaluation metric for Machine Unlearning algorithms based on epistemic uncertainty. This is the first definition of a general evaluation metric for Machine Unlearning to our best knowledge.

摘要: 最近，人们对机器遗忘的兴趣与日俱增，主要是因为法律要求，如一般数据保护法规(GDPR)和加州消费者隐私法。因此，人们提出了多种方法来消除特定目标数据点对训练模型的影响。然而，在评估遗忘的成功时，目前的方法要么使用对抗性攻击，要么将结果与最优解决方案进行比较，后者通常包括从头开始的再培训。我们认为，这两种方式在实践中都是不够的。在这项工作中，我们提出了一种基于认知不确定性的机器遗忘算法评价指标。据我们所知，这是对机器遗忘的一般评估指标的第一次定义。



## **48. UKP-SQuARE v2 Explainability and Adversarial Attacks for Trustworthy QA**

UKP-Square v2可解析性和可信QA的对抗性攻击 cs.CL

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.09316v2)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstracts**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.

摘要: 问答(QA)系统越来越多地部署在支持现实世界决策的应用程序中。然而，最先进的模型依赖于深度神经网络，这很难被人类解释。本质上可解释的模型或事后可解释的方法可以帮助用户理解模型如何达到其预测，如果成功，则增加他们对系统的信任。此外，研究人员可以利用这些洞察力来开发更准确、更少偏见的新方法。在本文中，我们引入了Square的新版本Square v2，以提供基于显著图和基于图的解释等方法的模型比较的可解释性基础设施。虽然显著图有助于检查每个输入标记对于模型预测的重要性，但来自外部知识图的基于图形的解释使用户能够验证模型预测背后的推理。此外，我们还提供了多个对抗性攻击来比较QA模型的健壮性。通过这些可解释性方法和对抗性攻击，我们的目标是简化可信QA模型的研究。Square在https://square.ukp-lab.de.上可用



## **49. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2009.04131v7)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **50. Adversarial Vulnerability of Temporal Feature Networks for Object Detection**

用于目标检测的时态特征网络的对抗脆弱性 cs.CV

Accepted for publication at ECCV 2022 SAIAD workshop

**SubmitDate**: 2022-08-23    [paper-pdf](http://arxiv.org/pdf/2208.10773v1)

**Authors**: Svetlana Pavlitskaya, Nikolai Polley, Michael Weber, J. Marius Zöllner

**Abstracts**: Taking into account information across the temporal domain helps to improve environment perception in autonomous driving. However, it has not been studied so far whether temporally fused neural networks are vulnerable to deliberately generated perturbations, i.e. adversarial attacks, or whether temporal history is an inherent defense against them. In this work, we study whether temporal feature networks for object detection are vulnerable to universal adversarial attacks. We evaluate attacks of two types: imperceptible noise for the whole image and locally-bound adversarial patch. In both cases, perturbations are generated in a white-box manner using PGD. Our experiments confirm, that attacking even a portion of a temporal input suffices to fool the network. We visually assess generated perturbations to gain insights into the functioning of attacks. To enhance the robustness, we apply adversarial training using 5-PGD. Our experiments on KITTI and nuScenes datasets demonstrate, that a model robustified via K-PGD is able to withstand the studied attacks while keeping the mAP-based performance comparable to that of an unattacked model.

摘要: 考虑跨时间域的信息有助于改善自动驾驶中的环境感知。然而，到目前为止，时间融合的神经网络是否容易受到故意产生的扰动，即对抗性攻击，或者时间历史是否是对它们的固有防御，还没有被研究过。在这项工作中，我们研究了用于目标检测的时态特征网络是否容易受到普遍的敌意攻击。我们评估了两种类型的攻击：整个图像的不可感知噪声攻击和局部绑定的敌意补丁攻击。在这两种情况下，使用PGD以白盒方式产生扰动。我们的实验证实，攻击即使是时间输入的一部分，也足以愚弄网络。我们直观地评估产生的扰动，以深入了解攻击的功能。为了增强算法的稳健性，我们使用了5-PGD进行对抗性训练。我们在Kitti和nuScenes数据集上的实验表明，通过K-PGD健壮的模型能够抵抗所研究的攻击，同时保持基于地图的性能与未受攻击的模型相当。



