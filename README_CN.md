# Latest Adversarial Attack Papers
**update at 2022-07-02 06:31:32**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. MEAD: A Multi-Armed Approach for Evaluation of Adversarial Examples Detectors**

Mead：一种评估对抗性范例检测器的多臂方法 cs.CV

This paper has been accepted to appear in the Proceedings of the 2022  European Conference on Machine Learning and Data Mining (ECML-PKDD), 19th to  the 23rd of September, Grenoble, France

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15415v1)

**Authors**: Federica Granese, Marine Picot, Marco Romanelli, Francisco Messina, Pablo Piantanida

**Abstracts**: Detection of adversarial examples has been a hot topic in the last years due to its importance for safely deploying machine learning algorithms in critical applications. However, the detection methods are generally validated by assuming a single implicitly known attack strategy, which does not necessarily account for real-life threats. Indeed, this can lead to an overoptimistic assessment of the detectors' performance and may induce some bias in the comparison between competing detection schemes. We propose a novel multi-armed framework, called MEAD, for evaluating detectors based on several attack strategies to overcome this limitation. Among them, we make use of three new objectives to generate attacks. The proposed performance metric is based on the worst-case scenario: detection is successful if and only if all different attacks are correctly recognized. Empirically, we show the effectiveness of our approach. Moreover, the poor performance obtained for state-of-the-art detectors opens a new exciting line of research.

摘要: 对抗性样本的检测是近年来的一个热门话题，因为它对于在关键应用中安全地部署机器学习算法具有重要意义。然而，检测方法通常是通过假设单个隐式已知的攻击策略来验证的，这不一定考虑现实生活中的威胁。事实上，这可能会导致对检测器性能的过度乐观评估，并可能在相互竞争的检测方案之间的比较中导致一些偏见。为了克服这一局限性，我们提出了一种新的多臂框架，称为MEAD，用于基于几种攻击策略来评估检测器。其中，我们利用三个新的目标来产生攻击。建议的性能指标基于最坏的情况：当且仅当正确识别所有不同的攻击时，检测才成功。在经验上，我们展示了我们方法的有效性。此外，最先进的探测器获得的糟糕性能开启了一条新的令人兴奋的研究路线。



## **2. The Topological BERT: Transforming Attention into Topology for Natural Language Processing**

拓扑学BERT：将注意力转化为自然语言处理的拓扑学 cs.CL

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.15195v1)

**Authors**: Ilan Perez, Raphael Reinauer

**Abstracts**: In recent years, the introduction of the Transformer models sparked a revolution in natural language processing (NLP). BERT was one of the first text encoders using only the attention mechanism without any recurrent parts to achieve state-of-the-art results on many NLP tasks.   This paper introduces a text classifier using topological data analysis. We use BERT's attention maps transformed into attention graphs as the only input to that classifier. The model can solve tasks such as distinguishing spam from ham messages, recognizing whether a sentence is grammatically correct, or evaluating a movie review as negative or positive. It performs comparably to the BERT baseline and outperforms it on some tasks.   Additionally, we propose a new method to reduce the number of BERT's attention heads considered by the topological classifier, which allows us to prune the number of heads from 144 down to as few as ten with no reduction in performance. Our work also shows that the topological model displays higher robustness against adversarial attacks than the original BERT model, which is maintained during the pruning process. To the best of our knowledge, this work is the first to confront topological-based models with adversarial attacks in the context of NLP.

摘要: 近年来，Transformer模型的引入引发了自然语言处理(NLP)的革命。Bert是第一批只使用注意力机制而不使用任何重复部分的文本编码者之一，以在许多NLP任务中获得最先进的结果。本文介绍了一种基于拓扑数据分析的文本分类器。我们使用Bert转换为注意图的注意图作为该分类器的唯一输入。该模型可以解决一些任务，比如区分垃圾邮件和垃圾邮件，识别句子的语法是否正确，或者评估电影评论是负面的还是正面的。它的表现与BERT基线相当，并在某些任务上超过它。此外，我们还提出了一种新的方法来减少拓扑分类器所考虑的BERT注意头数，该方法允许我们在不降低性能的情况下将注意头数从144个减少到10个。我们的工作还表明，与在剪枝过程中保持的原始BERT模型相比，该拓扑模型对敌意攻击表现出更高的稳健性。据我们所知，这是第一个在NLP环境下对抗基于拓扑模型的攻击的工作。



## **3. FIDO2 With Two Displays$\unicode{x2013}$Or How to Protect Security-Critical Web Transactions Against Malware Attacks**

带两个显示屏的FIDO2$\Unicode{x2013}$或如何保护安全关键型Web交易免受恶意软件攻击 cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2206.13358v2)

**Authors**: Timon Hackenjos, Benedikt Wagner, Julian Herr, Jochen Rill, Marek Wehmer, Niklas Goerke, Ingmar Baumgart

**Abstracts**: With the rise of attacks on online accounts in the past years, more and more services offer two-factor authentication for their users. Having factors out of two of the three categories something you know, something you have and something you are should ensure that an attacker cannot compromise two of them at once. Thus, an adversary should not be able to maliciously interact with one's account. However, this is only true if one considers a weak adversary. In particular, since most current solutions only authenticate a session and not individual transactions, they are noneffective if one's device is infected with malware. For online banking, the banking industry has long since identified the need for authenticating transactions. However, specifications of such authentication schemes are not public and implementation details vary wildly from bank to bank with most still being unable to protect against malware. In this work, we present a generic approach to tackle the problem of malicious account takeovers, even in the presence of malware. To this end, we define a new paradigm to improve two-factor authentication that involves the concepts of one-out-of-two security and transaction authentication. Web authentication schemes following this paradigm can protect security-critical transactions against manipulation, even if one of the factors is completely compromised. Analyzing existing authentication schemes, we find that they do not realize one-out-of-two security. We give a blueprint of how to design secure web authentication schemes in general. Based on this blueprint we propose FIDO2 With Two Displays (FIDO2D), a new web authentication scheme based on the FIDO2 standard and prove its security using Tamarin. We hope that our work inspires a new wave of more secure web authentication schemes, which protect security-critical transactions even against attacks with malware.

摘要: 随着过去几年针对在线账户的攻击事件的增加，越来越多的服务为其用户提供双因素身份验证。拥有三个类别中的两个因素，你知道的，你拥有的和你是的，应该确保攻击者不能同时危害其中的两个。因此，对手不应该能够恶意地与自己的帐户交互。然而，只有当一个人考虑到一个弱小的对手时，这才是正确的。特别是，由于大多数当前的解决方案只对会话进行身份验证，而不是对单个事务进行身份验证，因此如果设备感染了恶意软件，这些解决方案就会无效。对于网上银行，银行业早就认识到了对交易进行身份验证的必要性。然而，此类身份验证方案的规范并未公开，各银行的实施细节也存在很大差异，大多数银行仍无法防范恶意软件。在这项工作中，我们提出了一种通用的方法来解决恶意帐户接管问题，即使在存在恶意软件的情况下也是如此。为此，我们定义了一个新的范例来改进双因素身份验证，它涉及二选一安全和事务身份验证的概念。遵循此范例的Web身份验证方案可以保护安全关键型交易免受操纵，即使其中一个因素完全受损。分析现有的认证方案，发现它们并没有实现二选一的安全性。我们给出了一个总体上如何设计安全的Web认证方案的蓝图。在此基础上，我们提出了一种新的基于FIDO2标准的网络认证方案FIDO2 with Two Display(FIDO2D)，并用Tamarin对其安全性进行了证明。我们希望我们的工作激发出新一波更安全的网络身份验证方案，这些方案甚至可以保护安全关键交易免受恶意软件的攻击。



## **4. An Intermediate-level Attack Framework on The Basis of Linear Regression**

一种基于线性回归的中级攻击框架 cs.CV

Accepted by TPAMI; Code is available at  https://github.com/qizhangli/ila-plus-plus-lr

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2203.10723v2)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. Specifically, we advocate a framework in which a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to prediction loss of the adversarial example is established. By delving deep into the core components of such a framework, we show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level adversarial discrepancy is correlated with the transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. In addition, by leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks. Our code is publicly available at https://github.com/qizhangli/ila-plus-plus-lr.

摘要: 本文大大扩展了我们在ECCV上发表的工作，在该工作中，提出了一种中级攻击来提高一些基线对手例子的可转移性。具体地说，我们主张建立一个框架，在这个框架中，建立从对抗性例子的中间级差异(对抗性特征和良性特征之间)到预测损失的直接线性映射。通过深入研究该框架的核心部分，我们发现：1)为了建立映射，可以考虑多种线性回归模型；2)最终获得的中级敌方差异的大小与可转移性相关；3)通过随机初始化执行多次基线攻击，可以进一步提高性能。此外，通过利用这些发现，我们实现了针对基于传输的$\ell_\inty$和$\ell_2$攻击的新技术。我们的代码在https://github.com/qizhangli/ila-plus-plus-lr.上公开提供



## **5. On the Challenges of Detecting Side-Channel Attacks in SGX**

关于在SGX中检测旁路攻击的挑战 cs.CR

**SubmitDate**: 2022-06-30    [paper-pdf](http://arxiv.org/pdf/2011.14599v2)

**Authors**: Jianyu Jiang, Claudio Soriente, Ghassan Karame

**Abstracts**: Existing tools to detect side-channel attacks on Intel SGX are grounded on the observation that attacks affect the performance of the victim application. As such, all detection tools monitor the potential victim and raise an alarm if the witnessed performance (in terms of runtime, enclave interruptions, cache misses, etc.) is out of the ordinary.   In this paper, we show that monitoring the performance of enclaves to detect side-channel attacks may not be effective. Our core intuition is that all monitoring tools are geared towards an adversary that interferes with the victim's execution in order to extract the most number of secret bits (e.g., the entire secret) in one or few runs. They cannot, however, detect an adversary that leaks smaller portions of the secret - as small as a single bit - at each execution of the victim. In particular, by minimizing the information leaked at each run, the impact of any side-channel attack on the application's performance is significantly lowered - ensuring that the detection tool does not detect an attack. By repeating the attack multiple times, each time on a different part of the secret, the adversary can recover the whole secret and remain undetected. Based on this intuition, we adapt known attacks leveraging page-tables and L3 cache to bypass existing detection mechanisms. We show experimentally how an attacker can successfully exfiltrate the secret key used in an enclave running various cryptographic routines of libgcrypt. Beyond cryptographic libraries, we also show how to compromise the predictions of enclaves running decision-tree routines of OpenCV. Our evaluation results suggest that performance-based detection tools do not deter side-channel attacks on SGX enclaves and that effective detection mechanisms are yet to be designed.

摘要: 现有工具用于检测针对Intel SGX的旁路攻击，其基础是观察到攻击会影响受攻击应用程序的性能。因此，所有检测工具都会监视潜在受害者，并在发现性能(在运行时、飞地中断、缓存未命中等方面)时发出警报是不寻常的。在本文中，我们表明，通过监控Enclaves的性能来检测旁路攻击可能并不有效。我们的核心直觉是，所有监控工具都是针对干扰受害者执行的对手，以便在一次或几次运行中提取最多数量的秘密比特(例如，整个秘密)。然而，他们无法检测到在每次处决受害者时泄露较小部分秘密的对手。特别是，通过最大限度地减少每次运行时泄漏的信息，任何侧通道攻击对应用程序性能的影响都会显著降低，从而确保检测工具不会检测到攻击。通过多次重复攻击，每次对秘密的不同部分进行攻击，攻击者可以恢复整个秘密并保持不被发现。基于这一直觉，我们采用了利用页表和L3缓存的已知攻击来绕过现有的检测机制。我们通过实验展示了攻击者如何成功地渗出在运行各种libgcrypt加密例程的飞地中使用的秘密密钥。除了密码库之外，我们还展示了如何折衷运行OpenCV决策树例程的Enclaves的预测。我们的评估结果表明，基于性能的检测工具不能阻止对SGX飞地的旁路攻击，并且还没有设计有效的检测机制。



## **6. Depth-2 Neural Networks Under a Data-Poisoning Attack**

数据中毒攻击下的深度-2神经网络 cs.LG

32 page, 7 figures

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2005.01699v3)

**Authors**: Sayar Karmakar, Anirbit Mukherjee, Theodore Papamarkou

**Abstracts**: In this work, we study the possibility of defending against data-poisoning attacks while training a shallow neural network in a regression setup. We focus on doing supervised learning for a class of depth-2 finite-width neural networks, which includes single-filter convolutional networks. In this class of networks, we attempt to learn the network weights in the presence of a malicious oracle doing stochastic, bounded and additive adversarial distortions on the true output during training. For the non-gradient stochastic algorithm that we construct, we prove worst-case near-optimal trade-offs among the magnitude of the adversarial attack, the weight approximation accuracy, and the confidence achieved by the proposed algorithm. As our algorithm uses mini-batching, we analyze how the mini-batch size affects convergence. We also show how to utilize the scaling of the outer layer weights to counter output-poisoning attacks depending on the probability of attack. Lastly, we give experimental evidence demonstrating how our algorithm outperforms stochastic gradient descent under different input data distributions, including instances of heavy-tailed distributions.

摘要: 在这项工作中，我们研究了在回归设置中训练浅层神经网络的同时防御数据中毒攻击的可能性。重点研究了一类深度为2的有限宽度神经网络的监督学习问题，其中包括单滤波卷积网络。在这类网络中，我们试图在恶意预言存在的情况下学习网络权重，该预言在训练期间对真实输出进行随机的、有界的和相加的对抗性扭曲。对于我们构造的非梯度随机算法，我们证明了该算法在对抗性攻击的强度、权重逼近精度和所获得的置信度之间的最坏情况下的近优折衷。由于我们的算法使用了小批量，我们分析了小批量大小对收敛的影响。我们还展示了如何利用外层权重的比例来对抗依赖于攻击概率的输出中毒攻击。最后，我们给出了实验证据，展示了在不同的输入数据分布下，包括重尾分布的情况下，我们的算法的性能如何优于随机梯度下降。



## **7. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

基于分枝定界的IBP正则化算法 cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14772v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstracts**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.

摘要: 最近的工作试图通过在比原始扰动更大的域上运行攻击并在目标中添加各种正则化项来增加恶意训练网络的可验证性。然而，这些算法要么表现不佳，要么需要复杂而昂贵的阶段性训练过程，从而阻碍了它们的实际适用性。提出了一种简单有效的新的验证训练算法IBP-R。IBP-R通过将扩展域上的敌意攻击与基于廉价区间界传播的正则化项相结合来诱导网络可验证性，从而最小化非凸验证问题与其近似问题之间的差距。通过利用最近的分支定界框架，我们表明IBP-R在CIFAR-10上的小扰动下获得了经过验证的最先进的健壮性和准确性折衷，同时训练速度比相关以前的工作快得多。此外，我们提出了一种新的分支策略UPB，它依赖于基于$\beta$-Crown的简单启发式算法，在产生类似质量的分裂的同时，降低了最先进的分支算法的成本。



## **8. longhorns at DADC 2022: How many linguists does it take to fool a Question Answering model? A systematic approach to adversarial attacks**

DADC 2022上的长角人：需要多少语言学家才能愚弄一个问题回答模型？应对对抗性攻击的系统方法 cs.CL

Accepted at DADC2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14729v1)

**Authors**: Venelin Kovatchev, Trina Chatterjee, Venkata S Govindarajan, Jifan Chen, Eunsol Choi, Gabriella Chronis, Anubrata Das, Katrin Erk, Matthew Lease, Junyi Jessy Li, Yating Wu, Kyle Mahowald

**Abstracts**: Developing methods to adversarially challenge NLP systems is a promising avenue for improving both model performance and interpretability. Here, we describe the approach of the team "longhorns" on Task 1 of the The First Workshop on Dynamic Adversarial Data Collection (DADC), which asked teams to manually fool a model on an Extractive Question Answering task. Our team finished first, with a model error rate of 62%. We advocate for a systematic, linguistically informed approach to formulating adversarial questions, and we describe the results of our pilot experiments, as well as our official submission.

摘要: 开发反挑战NLP系统的方法是提高模型性能和可解释性的一条很有前途的途径。在这里，我们描述了“长角人”团队在第一次动态对手数据收集(DADC)研讨会(DADC)的任务1上的方法，该方法要求团队在提取问答任务中手动愚弄模型。我们的团队以62%的模型错误率获得第一名。我们提倡一种系统的、在语言上知情的方法来提出对抗性的问题，我们描述了我们的试点实验的结果，以及我们的正式提交。



## **9. Private Graph Extraction via Feature Explanations**

基于特征解释的专用图提取 cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14724v1)

**Authors**: Iyiola E. Olatunji, Mandeep Rathee, Thorben Funke, Megha Khosla

**Abstracts**: Privacy and interpretability are two of the important ingredients for achieving trustworthy machine learning. We study the interplay of these two aspects in graph machine learning through graph reconstruction attacks. The goal of the adversary here is to reconstruct the graph structure of the training data given access to model explanations. Based on the different kinds of auxiliary information available to the adversary, we propose several graph reconstruction attacks. We show that additional knowledge of post-hoc feature explanations substantially increases the success rate of these attacks. Further, we investigate in detail the differences between attack performance with respect to three different classes of explanation methods for graph neural networks: gradient-based, perturbation-based, and surrogate model-based methods. While gradient-based explanations reveal the most in terms of the graph structure, we find that these explanations do not always score high in utility. For the other two classes of explanations, privacy leakage increases with an increase in explanation utility. Finally, we propose a defense based on a randomized response mechanism for releasing the explanations which substantially reduces the attack success rate. Our anonymized code is available.

摘要: 隐私和可解释性是实现可信机器学习的两个重要因素。我们通过图重构攻击来研究这两个方面在图机器学习中的相互作用。在这里，对手的目标是在获得模型解释的情况下重建训练数据的图形结构。基于敌手可获得的各种辅助信息，我们提出了几种图重构攻击。我们表明，额外的事后特征解释知识大大提高了这些攻击的成功率。此外，我们还详细研究了图神经网络的三种不同解释方法：基于梯度、基于扰动和基于代理模型的解释方法在攻击性能上的差异。虽然基于梯度的解释在图表结构方面揭示了最多，但我们发现这些解释并不总是在实用方面得分很高。对于其他两类解释，隐私泄露随着解释效用的增加而增加。最后，我们提出了一种基于随机化响应机制的防御机制，用于发布解释，大大降低了攻击成功率。我们的匿名码是可用的。



## **10. Enhancing Security of Memristor Computing System Through Secure Weight Mapping**

通过安全权重映射提高忆阻器计算系统的安全性 cs.ET

6 pages, 4 figures, accepted by IEEE ISVLSI 2022

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14498v1)

**Authors**: Minhui Zou, Junlong Zhou, Xiaotong Cui, Wei Wang, Shahar Kvatinsky

**Abstracts**: Emerging memristor computing systems have demonstrated great promise in improving the energy efficiency of neural network (NN) algorithms. The NN weights stored in memristor crossbars, however, may face potential theft attacks due to the nonvolatility of the memristor devices. In this paper, we propose to protect the NN weights by mapping selected columns of them in the form of 1's complements and leaving the other columns in their original form, preventing the adversary from knowing the exact representation of each weight. The results show that compared with prior work, our method achieves effectiveness comparable to the best of them and reduces the hardware overhead by more than 18X.

摘要: 新兴的忆阻器计算系统在提高神经网络(NN)算法的能量效率方面显示出巨大的前景。然而，由于忆阻器器件的非易失性，存储在忆阻器纵横杆中的NN权重可能面临潜在的盗窃攻击。在本文中，我们建议通过以1的补码的形式映射选定的列来保护NN权重，而将其他列保持其原始形式，以防止对手知道每个权重的准确表示。实验结果表明，与前人的工作相比，我们的方法取得了与之相当的效果，硬件开销减少了18倍以上。



## **11. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

联合学习标签依赖关系和成员模型的对抗性集成训练 cs.LG

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14477v1)

**Authors**: Lele Wang, Bin Liu

**Abstracts**: Training an ensemble of different sub-models has empirically proven to be an effective strategy to improve deep neural networks' adversarial robustness. Current ensemble training methods for image recognition usually encode the image labels by one-hot vectors, which neglect dependency relationships between the labels. Here we propose a novel adversarial training approach that learns the conditional dependencies between labels and the model ensemble jointly. We test our approach on widely used datasets MNIST, FasionMNIST and CIFAR-10. Results show that our approach is more robust against black-box attacks compared with state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.

摘要: 实验证明，训练不同子模型的集成是提高深度神经网络对抗健壮性的有效策略。目前用于图像识别的集成训练方法通常将图像标签编码为单热点向量，忽略了标签之间的依赖关系。这里我们提出了一种新的对抗性训练方法，该方法联合学习标签和模型集成之间的条件依赖关系。我们在广泛使用的数据集MNIST、FasionMNIST和CIFAR-10上测试了我们的方法。结果表明，与最新的方法相比，我们的方法对黑盒攻击具有更强的鲁棒性。我们的代码可以在https://github.com/ZJLAB-AMMI/LSD.上找到



## **12. Guided Diffusion Model for Adversarial Purification**

对抗性净化中的引导扩散模型 cs.CV

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2205.14969v3)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.

摘要: 随着深度神经网络(DNN)在各种算法和框架中的广泛应用，安全威胁已成为人们关注的问题之一。对抗性攻击干扰了基于DNN的图像分类器，攻击者可以故意在输入图像上添加不可察觉的对抗性扰动来愚弄分类器。在本文中，我们提出了一种新的净化方法，称为引导扩散净化模型(GDMP)，以帮助保护分类器免受对手攻击。该方法的核心是将净化嵌入到去噪扩散概率模型(DDPM)的扩散去噪过程中，使其扩散过程能够淹没带有逐渐增加的高斯噪声的对抗性扰动，并在引导去噪过程后同时去除这两种噪声。在不同数据集上的综合实验表明，所提出的GDMP将对抗性攻击引起的扰动减少到较小的范围，从而显著提高了分类的正确性。GDMP在CIFAR10数据集上的稳健准确率提高了5%，在PGD攻击下达到了90.1%。此外，GDMP在具有挑战性的ImageNet数据集上获得了70.94%的健壮性。



## **13. A Deep Learning Approach to Create DNS Amplification Attacks**

一种创建域名系统放大攻击的深度学习方法 cs.CR

12 pages, 6 figures, Conference: to 2022 4th International Conference  on Management Science and Industrial Engineering (MSIE) (MSIE 2022), DOI:  https://doi.org/10.1145/3535782.3535838, accepted to conference above, not  yet published

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2206.14346v1)

**Authors**: Jared Mathews, Prosenjit Chatterjee, Shankar Banik, Cory Nance

**Abstracts**: In recent years, deep learning has shown itself to be an incredibly valuable tool in cybersecurity as it helps network intrusion detection systems to classify attacks and detect new ones. Adversarial learning is the process of utilizing machine learning to generate a perturbed set of inputs to then feed to the neural network to misclassify it. Much of the current work in the field of adversarial learning has been conducted in image processing and natural language processing with a wide variety of algorithms. Two algorithms of interest are the Elastic-Net Attack on Deep Neural Networks and TextAttack. In our experiment the EAD and TextAttack algorithms are applied to a Domain Name System amplification classifier. The algorithms are used to generate malicious Distributed Denial of Service adversarial examples to then feed as inputs to the network intrusion detection systems neural network to classify as valid traffic. We show in this work that both image processing and natural language processing adversarial learning algorithms can be applied against a network intrusion detection neural network.

摘要: 近年来，深度学习已被证明是网络安全中一个极其有价值的工具，因为它有助于网络入侵检测系统对攻击进行分类并检测新的攻击。对抗性学习是利用机器学习生成一组扰动的输入，然后馈送到神经网络进行错误分类的过程。目前在对抗性学习领域的许多工作都是在图像处理和自然语言处理方面进行的，并使用了各种算法。两个有趣的算法是对深度神经网络的弹性网络攻击和TextAttack。在我们的实验中，我们将EAD和TextAttack算法应用于域名系统放大分类器。这些算法被用来生成恶意的分布式拒绝服务攻击实例，然后将其作为输入输入到网络入侵检测系统的神经网络中，以分类为有效流量。在这项工作中，我们证明了图像处理和自然语言处理对抗性学习算法都可以应用于网络入侵检测神经网络。



## **14. Linear Model Against Malicious Adversaries with Local Differential Privacy**

基于局部差分隐私的对抗恶意攻击的线性模型 cs.CR

**SubmitDate**: 2022-06-29    [paper-pdf](http://arxiv.org/pdf/2202.02448v2)

**Authors**: Guanhong Miao, A. Adam Ding, Samuel S. Wu

**Abstracts**: Scientific collaborations benefit from collaborative learning of distributed sources, but remain difficult to achieve when data are sensitive. In recent years, privacy preserving techniques have been widely studied to analyze distributed data across different agencies while protecting sensitive information. Most existing privacy preserving techniques are designed to resist semi-honest adversaries and require intense computation to perform data analysis. Secure collaborative learning is significantly difficult with the presence of malicious adversaries who may deviates from the secure protocol. Another challenge is to maintain high computation efficiency with privacy protection. In this paper, matrix encryption is applied to encrypt data such that the secure schemes are against malicious adversaries, including chosen plaintext attack, known plaintext attack, and collusion attack. The encryption scheme also achieves local differential privacy. Moreover, cross validation is studied to prevent overfitting without additional communication cost. Empirical experiments on real-world datasets demonstrate that the proposed schemes are computationally efficient compared to existing techniques against malicious adversary and semi-honest model.

摘要: 科学协作受益于分布式来源的协作学习，但在数据敏感时仍难以实现。近年来，隐私保护技术被广泛研究，以在保护敏感信息的同时分析跨不同机构的分布式数据。大多数现有的隐私保护技术都是为了抵抗半诚实的对手而设计的，并且需要大量的计算来执行数据分析。在存在可能偏离安全协议的恶意攻击者的情况下，安全协作学习非常困难。另一个挑战是在隐私保护的情况下保持高计算效率。本文采用矩阵加密的方法对数据进行加密，使安全方案能够抵抗选择明文攻击、已知明文攻击和合谋攻击等恶意攻击。该加密方案还实现了局部差分保密。此外，为了在不增加通信成本的情况下防止过拟合，还研究了交叉验证。在真实数据集上的实验表明，与已有的对抗恶意攻击和半诚实模型的技术相比，所提出的方案具有较高的计算效率。



## **15. An Empirical Study of Challenges in Converting Deep Learning Models**

深度学习模式转换挑战的实证研究 cs.LG

Accepted for publication in ICSME 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14322v1)

**Authors**: Moses Openja, Amin Nikanjam, Ahmed Haj Yahmed, Foutse Khomh, Zhen Ming, Jiang

**Abstracts**: There is an increase in deploying Deep Learning (DL)-based software systems in real-world applications. Usually DL models are developed and trained using DL frameworks that have their own internal mechanisms/formats to represent and train DL models, and usually those formats cannot be recognized by other frameworks. Moreover, trained models are usually deployed in environments different from where they were developed. To solve the interoperability issue and make DL models compatible with different frameworks/environments, some exchange formats are introduced for DL models, like ONNX and CoreML. However, ONNX and CoreML were never empirically evaluated by the community to reveal their prediction accuracy, performance, and robustness after conversion. Poor accuracy or non-robust behavior of converted models may lead to poor quality of deployed DL-based software systems. We conduct, in this paper, the first empirical study to assess ONNX and CoreML for converting trained DL models. In our systematic approach, two popular DL frameworks, Keras and PyTorch, are used to train five widely used DL models on three popular datasets. The trained models are then converted to ONNX and CoreML and transferred to two runtime environments designated for such formats, to be evaluated. We investigate the prediction accuracy before and after conversion. Our results unveil that the prediction accuracy of converted models are at the same level of originals. The performance (time cost and memory consumption) of converted models are studied as well. The size of models are reduced after conversion, which can result in optimized DL-based software deployment. Converted models are generally assessed as robust at the same level of originals. However, obtained results show that CoreML models are more vulnerable to adversarial attacks compared to ONNX.

摘要: 在现实世界的应用程序中部署基于深度学习(DL)的软件系统的情况越来越多。通常，使用具有自己的内部机制/格式来表示和训练DL模型的DL框架来开发和训练DL模型，并且通常这些格式不能被其他框架识别。此外，经过训练的模型通常部署在与开发环境不同的环境中。为了解决互操作问题，使DL模型与不同的框架/环境兼容，引入了一些用于DL模型的交换格式，如ONNX和CoreML。然而，ONNX和CoreML从未得到社区的经验性评估，以揭示它们在转换后的预测准确性、性能和稳健性。转换后模型的准确性或非健壮性可能会导致已部署的基于DL的软件系统的质量较差。在本文中，我们进行了第一次实证研究，以评估ONNX和CoreML用于转换训练的DL模型的能力。在我们的系统方法中，两个流行的DL框架Kera和PyTorch被用来在三个流行的数据集上训练五个广泛使用的DL模型。然后，训练的模型被转换为ONNX和CoreML，并被传输到为这些格式指定的两个运行时环境，以进行评估。我们考察了转换前后的预测精度。我们的结果表明，转换后的模型的预测精度与原始模型相同。并对转换后模型的性能(时间开销和内存消耗)进行了研究。转换后模型的大小会减小，从而可以优化基于DL的软件部署。转换后的模型通常被评估为与原始模型具有相同水平的健壮性。然而，研究结果表明，与ONNX相比，CoreML模型更容易受到敌意攻击。



## **16. Collecting high-quality adversarial data for machine reading comprehension tasks with humans and models in the loop**

为机器阅读理解任务收集高质量的对抗性数据，其中人和模型处于循环中 cs.CL

8 pages, 3 figures, for more information about the shared task please  go to https://dadcworkshop.github.io/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14272v1)

**Authors**: Damian Y. Romero Diaz, Magdalena Anioł, John Culnan

**Abstracts**: We present our experience as annotators in the creation of high-quality, adversarial machine-reading-comprehension data for extractive QA for Task 1 of the First Workshop on Dynamic Adversarial Data Collection (DADC). DADC is an emergent data collection paradigm with both models and humans in the loop. We set up a quasi-experimental annotation design and perform quantitative analyses across groups with different numbers of annotators focusing on successful adversarial attacks, cost analysis, and annotator confidence correlation. We further perform a qualitative analysis of our perceived difficulty of the task given the different topics of the passages in our dataset and conclude with recommendations and suggestions that might be of value to people working on future DADC tasks and related annotation interfaces.

摘要: 我们介绍了我们作为注释员在创建高质量的对抗性机器阅读理解数据方面的经验，这些数据用于第一次动态对抗性数据收集(DADC)研讨会的任务1的摘录QA。DADC是一种模型和人类都在循环中的紧急数据收集范例。我们建立了一个准实验性的标注设计，并对不同数量的注释者进行了量化分析，重点是成功的对抗性攻击、代价分析和注释者的置信度相关性。在给定数据集中段落的不同主题的情况下，我们进一步对任务的感知难度进行了定性分析，并提出了可能对从事未来DADC任务和相关注释接口工作的人有价值的建议和建议。



## **17. How to Steer Your Adversary: Targeted and Efficient Model Stealing Defenses with Gradient Redirection**

如何引导你的对手：有针对性和高效的模型窃取防御和渐变重定向 cs.LG

ICML 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14157v1)

**Authors**: Mantas Mazeika, Bo Li, David Forsyth

**Abstracts**: Model stealing attacks present a dilemma for public machine learning APIs. To protect financial investments, companies may be forced to withhold important information about their models that could facilitate theft, including uncertainty estimates and prediction explanations. This compromise is harmful not only to users but also to external transparency. Model stealing defenses seek to resolve this dilemma by making models harder to steal while preserving utility for benign users. However, existing defenses have poor performance in practice, either requiring enormous computational overheads or severe utility trade-offs. To meet these challenges, we present a new approach to model stealing defenses called gradient redirection. At the core of our approach is a provably optimal, efficient algorithm for steering an adversary's training updates in a targeted manner. Combined with improvements to surrogate networks and a novel coordinated defense strategy, our gradient redirection defense, called GRAD${}^2$, achieves small utility trade-offs and low computational overhead, outperforming the best prior defenses. Moreover, we demonstrate how gradient redirection enables reprogramming the adversary with arbitrary behavior, which we hope will foster work on new avenues of defense.

摘要: 模型窃取攻击给公共机器学习API带来了两难境地。为了保护金融投资，公司可能会被迫隐瞒有关其模型的重要信息，这些信息可能会为盗窃提供便利，包括不确定性估计和预测解释。这种妥协不仅损害了用户，也损害了外部透明度。模型窃取防御试图通过使模型更难被窃取，同时保护良性用户的实用性来解决这一困境。然而，现有的防御在实践中表现不佳，要么需要巨大的计算开销，要么需要严重的实用权衡。为了应对这些挑战，我们提出了一种新的方法来模拟窃取防御，称为梯度重定向。我们方法的核心是一种可证明是最优的、高效的算法，用于以有针对性的方式引导对手的训练更新。结合对代理网络的改进和一种新的协调防御策略，我们的梯度重定向防御，称为Grad$^2$，实现了小的效用权衡和较低的计算开销，性能优于最好的先前防御。此外，我们还演示了梯度重定向如何使用任意行为对对手进行重新编程，我们希望这将促进新防御途径的工作。



## **18. Debiasing Learning for Membership Inference Attacks Against Recommender Systems**

推荐系统成员关系推理攻击的去偏学习 cs.IR

Accepted by KDD 2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.12401v2)

**Authors**: Zihan Wang, Na Huang, Fei Sun, Pengjie Ren, Zhumin Chen, Hengliang Luo, Maarten de Rijke, Zhaochun Ren

**Abstracts**: Learned recommender systems may inadvertently leak information about their training data, leading to privacy violations. We investigate privacy threats faced by recommender systems through the lens of membership inference. In such attacks, an adversary aims to infer whether a user's data is used to train the target recommender. To achieve this, previous work has used a shadow recommender to derive training data for the attack model, and then predicts the membership by calculating difference vectors between users' historical interactions and recommended items. State-of-the-art methods face two challenging problems: (1) training data for the attack model is biased due to the gap between shadow and target recommenders, and (2) hidden states in recommenders are not observational, resulting in inaccurate estimations of difference vectors. To address the above limitations, we propose a Debiasing Learning for Membership Inference Attacks against recommender systems (DL-MIA) framework that has four main components: (1) a difference vector generator, (2) a disentangled encoder, (3) a weight estimator, and (4) an attack model. To mitigate the gap between recommenders, a variational auto-encoder (VAE) based disentangled encoder is devised to identify recommender invariant and specific features. To reduce the estimation bias, we design a weight estimator, assigning a truth-level score for each difference vector to indicate estimation accuracy. We evaluate DL-MIA against both general recommenders and sequential recommenders on three real-world datasets. Experimental results show that DL-MIA effectively alleviates training and estimation biases simultaneously, and achieves state-of-the-art attack performance.

摘要: 学习推荐系统可能会无意中泄露有关其训练数据的信息，导致侵犯隐私。我们通过成员关系推理的视角来研究推荐系统所面临的隐私威胁。在这类攻击中，对手的目标是推断用户的数据是否被用来训练目标推荐者。为此，以前的工作使用影子推荐器来获取攻击模型的训练数据，然后通过计算用户历史交互与推荐项目之间的差异向量来预测成员资格。最新的方法面临两个具有挑战性的问题：(1)由于阴影和目标推荐器之间的差距，攻击模型的训练数据存在偏差；(2)推荐器中的隐藏状态不是可观测的，导致对差异向量的估计不准确。针对上述局限性，我们提出了一个针对推荐系统成员推理攻击的去偏学习框架(DL-MIA)，该框架包括四个主要部分：(1)差分向量生成器，(2)解缠编码器，(3)权重估计器，(4)攻击模型。为了缩小推荐者之间的差距，设计了一种基于变分自动编码器(VAE)的解缠编码器来识别推荐者的不变性和特定特征。为了减少估计偏差，我们设计了一个权重估计器，为每个差异向量分配一个真实度分数来表示估计的准确性。在三个真实数据集上，我们对比了一般推荐器和顺序推荐器对DL-MIA进行了评估。实验结果表明，DL-MIA有效地同时缓解了训练偏差和估计偏差，达到了最好的攻击性能。



## **19. On the amplification of security and privacy risks by post-hoc explanations in machine learning models**

机器学习模型中事后解释对安全和隐私风险的放大 cs.LG

9 pages, appendix: 2 pages

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.14004v1)

**Authors**: Pengrui Quan, Supriyo Chakraborty, Jeya Vikranth Jeyakumar, Mani Srivastava

**Abstracts**: A variety of explanation methods have been proposed in recent years to help users gain insights into the results returned by neural networks, which are otherwise complex and opaque black-boxes. However, explanations give rise to potential side-channels that can be leveraged by an adversary for mounting attacks on the system. In particular, post-hoc explanation methods that highlight input dimensions according to their importance or relevance to the result also leak information that weakens security and privacy. In this work, we perform the first systematic characterization of the privacy and security risks arising from various popular explanation techniques. First, we propose novel explanation-guided black-box evasion attacks that lead to 10 times reduction in query count for the same success rate. We show that the adversarial advantage from explanations can be quantified as a reduction in the total variance of the estimated gradient. Second, we revisit the membership information leaked by common explanations. Contrary to observations in prior studies, via our modified attacks we show significant leakage of membership information (above 100% improvement over prior results), even in a much stricter black-box setting. Finally, we study explanation-guided model extraction attacks and demonstrate adversarial gains through a large reduction in query count.

摘要: 近年来，人们提出了各种解释方法，以帮助用户深入了解神经网络返回的结果，否则这些结果就是复杂和不透明的黑匣子。然而，解释会导致潜在的旁路，对手可以利用这些旁路对系统进行攻击。特别是，根据输入维度的重要性或与结果的相关性来强调输入维度的事后解释方法也会泄露削弱安全和隐私的信息。在这项工作中，我们首次系统地描述了各种流行的解释技术所产生的隐私和安全风险。首先，我们提出了一种新的解释引导的黑盒逃避攻击，在相同的成功率下使查询次数减少了10倍。我们表明，从解释中获得的对手优势可以量化为估计梯度的总方差的减少。其次，我们重新审视了由常见解释泄露的成员信息。与先前研究中的观察相反，通过我们修改的攻击，我们显示了显著的成员信息泄露(比先前的结果提高了100%以上)，即使在更严格的黑盒设置中也是如此。最后，我们研究了解释引导的模型提取攻击，并通过大幅减少查询次数展示了敌意收益。



## **20. Increasing Confidence in Adversarial Robustness Evaluations**

增加对对手健壮性评估的信心 cs.LG

Oral at CVPR 2022 Workshop (Art of Robustness). Project website  https://zimmerrol.github.io/active-tests/

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2206.13991v1)

**Authors**: Roland S. Zimmermann, Wieland Brendel, Florian Tramer, Nicholas Carlini

**Abstracts**: Hundreds of defenses have been proposed to make deep neural networks robust against minimal (adversarial) input perturbations. However, only a handful of these defenses held up their claims because correctly evaluating robustness is extremely challenging: Weak attacks often fail to find adversarial examples even if they unknowingly exist, thereby making a vulnerable network look robust. In this paper, we propose a test to identify weak attacks, and thus weak defense evaluations. Our test slightly modifies a neural network to guarantee the existence of an adversarial example for every sample. Consequentially, any correct attack must succeed in breaking this modified network. For eleven out of thirteen previously-published defenses, the original evaluation of the defense fails our test, while stronger attacks that break these defenses pass it. We hope that attack unit tests - such as ours - will be a major component in future robustness evaluations and increase confidence in an empirical field that is currently riddled with skepticism.

摘要: 已经提出了数百种防御措施，以使深度神经网络对最小(对抗性)输入扰动具有健壮性。然而，这些防御中只有少数几个站得住脚，因为正确评估健壮性极具挑战性：弱攻击往往无法找到对抗性示例，即使它们在不知情的情况下存在，从而使易受攻击的网络看起来很健壮。在本文中，我们提出了一种测试来识别弱攻击，从而对弱防御进行评估。我们的测试略微修改了神经网络，以确保每个样本都存在对抗性示例。因此，任何正确的攻击都必须成功地破坏这个修改后的网络。在之前公布的13个防御措施中，有11个没有通过我们的测试，而打破这些防御措施的更强大的攻击通过了我们的测试。我们希望攻击单元测试--比如我们的测试--将成为未来健壮性评估的主要组成部分，并增加对目前充满怀疑的经验领域的信心。



## **21. Ownership Verification of DNN Architectures via Hardware Cache Side Channels**

通过硬件缓存侧通道验证DNN体系结构的所有权 cs.CR

The paper has been accepted by IEEE Transactions on Circuits and  Systems for Video Technology

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2102.03523v4)

**Authors**: Xiaoxuan Lou, Shangwei Guo, Jiwei Li, Tianwei Zhang

**Abstracts**: Deep Neural Networks (DNN) are gaining higher commercial values in computer vision applications, e.g., image classification, video analytics, etc. This calls for urgent demands of the intellectual property (IP) protection of DNN models. In this paper, we present a novel watermarking scheme to achieve the ownership verification of DNN architectures. Existing works all embedded watermarks into the model parameters while treating the architecture as public property. These solutions were proven to be vulnerable by an adversary to detect or remove the watermarks. In contrast, we claim the model architectures as an important IP for model owners, and propose to implant watermarks into the architectures. We design new algorithms based on Neural Architecture Search (NAS) to generate watermarked architectures, which are unique enough to represent the ownership, while maintaining high model usability. Such watermarks can be extracted via side-channel-based model extraction techniques with high fidelity. We conduct comprehensive experiments on watermarked CNN models for image classification tasks and the experimental results show our scheme has negligible impact on the model performance, and exhibits strong robustness against various model transformations and adaptive attacks.

摘要: 深度神经网络(DNN)在计算机视觉应用中获得了更高的商业价值，如图像分类、视频分析等。这就对DNN模型的知识产权保护提出了迫切的要求。本文提出了一种新的数字水印方案来实现DNN体系结构的所有权验证。现有的工作都是将水印嵌入到模型参数中，同时将建筑视为公共财产。事实证明，这些解决方案在检测或删除水印时容易受到攻击。相反，我们声称模型体系结构是模型所有者的重要IP，并提出在模型体系结构中嵌入水印。我们设计了基于神经结构搜索(NAS)的新算法来生成水印体系结构，这些体系结构具有足够的唯一性来表示所有权，同时保持了高模型可用性。这样的水印可以通过基于侧信道的高保真模型提取技术来提取。我们对带水印的CNN模型进行了全面的图像分类实验，实验结果表明，该算法对模型性能的影响可以忽略不计，并且对各种模型变换和自适应攻击具有很强的鲁棒性。



## **22. Deep Image Destruction: Vulnerability of Deep Image-to-Image Models against Adversarial Attacks**

深度图像破坏：深度图像到图像模型抵抗敌意攻击的脆弱性 cs.CV

ICPR2022

**SubmitDate**: 2022-06-28    [paper-pdf](http://arxiv.org/pdf/2104.15022v2)

**Authors**: Jun-Ho Choi, Huan Zhang, Jun-Hyuk Kim, Cho-Jui Hsieh, Jong-Seok Lee

**Abstracts**: Recently, the vulnerability of deep image classification models to adversarial attacks has been investigated. However, such an issue has not been thoroughly studied for image-to-image tasks that take an input image and generate an output image (e.g., colorization, denoising, deblurring, etc.) This paper presents comprehensive investigations into the vulnerability of deep image-to-image models to adversarial attacks. For five popular image-to-image tasks, 16 deep models are analyzed from various standpoints such as output quality degradation due to attacks, transferability of adversarial examples across different tasks, and characteristics of perturbations. We show that unlike image classification tasks, the performance degradation on image-to-image tasks largely differs depending on various factors, e.g., attack methods and task objectives. In addition, we analyze the effectiveness of conventional defense methods used for classification models in improving the robustness of the image-to-image models.

摘要: 最近，深度图像分类模型对敌意攻击的脆弱性进行了研究。然而，对于获取输入图像并生成输出图像(例如，彩色化、去噪、去模糊等)的图像到图像任务，这样的问题尚未被彻底研究。本文对深度图像到图像模型在对抗攻击中的脆弱性进行了全面的研究。对于五种常见的图像到图像任务，从攻击导致的输出质量下降、对抗性样本在不同任务之间的可转移性以及扰动的特征等不同的角度分析了16个深度模型。我们表明，与图像分类任务不同，图像到图像任务的性能下降在很大程度上取决于各种因素，例如攻击方法和任务目标。此外，我们还分析了用于分类模型的常规防御方法在提高图像到图像模型的稳健性方面的有效性。



## **23. Improving Privacy and Security in Unmanned Aerial Vehicles Network using Blockchain**

利用区块链提高无人机网络的保密性和安全性 cs.CR

18 Pages; 14 Figures; 2 Tables

**SubmitDate**: 2022-06-27    [paper-pdf](http://arxiv.org/pdf/2201.06100v2)

**Authors**: Hardik Sachdeva, Shivam Gupta, Anushka Misra, Khushbu Chauhan, Mayank Dave

**Abstracts**: Unmanned Aerial Vehicles (UAVs), also known as drones, have exploded in every segment present in todays business industry. They have scope in reinventing old businesses, and they are even developing new opportunities for various brands and franchisors. UAVs are used in the supply chain, maintaining surveillance and serving as mobile hotspots. Although UAVs have potential applications, they bring several societal concerns and challenges that need addressing in public safety, privacy, and cyber security. UAVs are prone to various cyber-attacks and vulnerabilities; they can also be hacked and misused by malicious entities resulting in cyber-crime. The adversaries can exploit these vulnerabilities, leading to data loss, property, and destruction of life. One can partially detect the attacks like false information dissemination, jamming, gray hole, blackhole, and GPS spoofing by monitoring the UAV behavior, but it may not resolve privacy issues. This paper presents secure communication between UAVs using blockchain technology. Our approach involves building smart contracts and making a secure and reliable UAV adhoc network. This network will be resilient to various network attacks and is secure against malicious intrusions.

摘要: 无人机(UAVs)，也被称为无人机，在当今商业行业的每一个细分领域都出现了爆炸式增长。他们有重塑旧业务的余地，甚至正在为各种品牌和特许经营商开发新的机会。无人机在供应链中使用，维持监控，并作为移动热点。虽然无人机有潜在的应用，但它们带来了一些社会关切和挑战，需要在公共安全、隐私和网络安全方面加以解决。无人机容易受到各种网络攻击和漏洞；它们也可能被恶意实体黑客攻击和滥用，从而导致网络犯罪。攻击者可以利用这些漏洞，导致数据丢失、财产损失和生命损失。通过对无人机行为的监控，可以部分检测到虚假信息传播、干扰、灰洞、黑洞、GPS欺骗等攻击，但不一定能解决隐私问题。本文介绍了利用区块链技术实现无人机之间的安全通信。我们的方法包括建立智能合同和建立安全可靠的无人机临时网络。该网络将对各种网络攻击具有弹性，并且能够安全地抵御恶意入侵。



## **24. Adversarially Robust Learning of Real-Valued Functions**

实值函数的逆鲁棒学习 cs.LG

**SubmitDate**: 2022-06-26    [paper-pdf](http://arxiv.org/pdf/2206.12977v1)

**Authors**: Idan Attias, Steve Hanneke

**Abstracts**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. We also discuss extensions to agnostic learning. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension.

摘要: 在具有$\ell_p$损失和任意扰动集的回归环境下，研究了对测试时间敌意攻击的稳健性。我们解决了在此设置中哪些函数类是PAC可学习的问题。我们证明了有限脂肪粉碎维类是可学习的。此外，对于凸函数类，它们甚至是可正规学习的。相反，一些非凸函数类显然需要不正确的学习算法。我们还讨论了不可知论学习的扩展。我们的主要技术是基于构造一个反向稳健的样本压缩方案，其大小由脂肪粉碎维度确定。



## **25. Cascading Failures in Smart Grids under Random, Targeted and Adaptive Attacks**

随机、定向和自适应攻击下智能电网的连锁故障 cs.SI

Accepted for publication as a book chapter. arXiv admin note:  substantial text overlap with arXiv:1402.6809

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12735v1)

**Authors**: Sushmita Ruj, Arindam Pal

**Abstracts**: We study cascading failures in smart grids, where an attacker selectively compromises the nodes with probabilities proportional to their degrees, betweenness, or clustering coefficient. This implies that nodes with high degrees, betweenness, or clustering coefficients are attacked with higher probability. We mathematically and experimentally analyze the sizes of the giant components of the networks under different types of targeted attacks, and compare the results with the corresponding sizes under random attacks. We show that networks disintegrate faster for targeted attacks compared to random attacks. A targeted attack on a small fraction of high degree nodes disintegrates one or both of the networks, whereas both the networks contain giant components for random attack on the same fraction of nodes. An important observation is that an attacker has an advantage if it compromises nodes based on their betweenness, rather than based on degree or clustering coefficient.   We next study adaptive attacks, where an attacker compromises nodes in rounds. Here, some nodes are compromised in each round based on their degree, betweenness or clustering coefficients, instead of compromising all nodes together. In this case, the degree, betweenness, or clustering coefficient is calculated before the start of each round, instead of at the beginning. We show experimentally that an adversary has an advantage in this adaptive approach, compared to compromising the same number of nodes all at once.

摘要: 我们研究了智能电网中的连锁故障，在这种情况下，攻击者有选择地以与节点的度、介数或聚类系数成正比的概率危害节点。这意味着具有高度、介数或聚类系数的节点被攻击的概率更高。我们从数学和实验上分析了不同类型的目标攻击下网络巨型组件的大小，并将结果与随机攻击下的相应大小进行了比较。我们发现，与随机攻击相比，定向攻击的网络瓦解速度更快。对一小部分高度节点的定向攻击会瓦解一个或两个网络，而这两个网络都包含对相同部分节点进行随机攻击的巨大组件。一个重要的观察结果是，如果攻击者基于节点的介入性而不是基于度或聚类系数来危害节点，则攻击者具有优势。接下来，我们研究自适应攻击，即攻击者对节点进行轮次攻击。这里，一些节点在每一轮中根据它们的度、介数或聚类系数进行妥协，而不是将所有节点一起妥协。在这种情况下，度数、介数或聚类系数在每轮开始之前计算，而不是在开始时计算。我们通过实验证明，与一次危害相同数量的节点相比，对手在这种自适应方法中具有优势。



## **26. Empirical Evaluation of Physical Adversarial Patch Attacks Against Overhead Object Detection Models**

基于头顶目标检测模型的物理对抗性补丁攻击的经验评估 cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12725v1)

**Authors**: Gavin S. Hartnett, Li Ang Zhang, Caolionn O'Connell, Andrew J. Lohn, Jair Aguirre

**Abstracts**: Adversarial patches are images designed to fool otherwise well-performing neural network-based computer vision models. Although these attacks were initially conceived of and studied digitally, in that the raw pixel values of the image were perturbed, recent work has demonstrated that these attacks can successfully transfer to the physical world. This can be accomplished by printing out the patch and adding it into scenes of newly captured images or video footage. In this work we further test the efficacy of adversarial patch attacks in the physical world under more challenging conditions. We consider object detection models trained on overhead imagery acquired through aerial or satellite cameras, and we test physical adversarial patches inserted into scenes of a desert environment. Our main finding is that it is far more difficult to successfully implement the adversarial patch attacks under these conditions than in the previously considered conditions. This has important implications for AI safety as the real-world threat posed by adversarial examples may be overstated.

摘要: 对抗性补丁是旨在愚弄其他表现良好的基于神经网络的计算机视觉模型的图像。虽然这些攻击最初是以数字方式构思和研究的，因为图像的原始像素值受到了干扰，但最近的研究表明，这些攻击可以成功地转移到物理世界。这可以通过打印补丁并将其添加到新捕获的图像或视频片段的场景中来实现。在这项工作中，我们进一步测试了对抗性补丁攻击在更具挑战性的条件下在物理世界中的有效性。我们考虑在通过航空或卫星摄像机获取的头顶图像上训练的目标检测模型，并测试插入到沙漠环境场景中的物理对抗性补丁。我们的主要发现是，在这些条件下成功实施对抗性补丁攻击比在先前考虑的条件下要困难得多。这对人工智能安全具有重要影响，因为对抗性例子构成的现实世界威胁可能被夸大了。



## **27. Defending Multimodal Fusion Models against Single-Source Adversaries**

防御单源攻击的多通道融合模型 cs.CV

CVPR 2021

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12714v1)

**Authors**: Karren Yang, Wan-Yi Lin, Manash Barman, Filipe Condessa, Zico Kolter

**Abstracts**: Beyond achieving high performance across many vision tasks, multimodal models are expected to be robust to single-source faults due to the availability of redundant information between modalities. In this paper, we investigate the robustness of multimodal neural networks against worst-case (i.e., adversarial) perturbations on a single modality. We first show that standard multimodal fusion models are vulnerable to single-source adversaries: an attack on any single modality can overcome the correct information from multiple unperturbed modalities and cause the model to fail. This surprising vulnerability holds across diverse multimodal tasks and necessitates a solution. Motivated by this finding, we propose an adversarially robust fusion strategy that trains the model to compare information coming from all the input sources, detect inconsistencies in the perturbed modality compared to the other modalities, and only allow information from the unperturbed modalities to pass through. Our approach significantly improves on state-of-the-art methods in single-source robustness, achieving gains of 7.8-25.2% on action recognition, 19.7-48.2% on object detection, and 1.6-6.7% on sentiment analysis, without degrading performance on unperturbed (i.e., clean) data.

摘要: 除了在许多视觉任务中实现高性能之外，由于多模式之间存在冗余信息，因此预计多模式对单源故障具有健壮性。在本文中，我们研究了多通道神经网络对单一通道上最坏情况(即对抗性)扰动的稳健性。我们首先证明了标准的多模式融合模型容易受到单一来源的攻击：对任何单一模式的攻击都可以克服来自多个未受干扰的模式的正确信息，从而导致模型失败。这种令人惊讶的漏洞存在于各种多模式任务中，需要一个解决方案。受这一发现的启发，我们提出了一种对抗性鲁棒的融合策略，该策略训练模型比较来自所有输入源的信息，检测与其他通道相比扰动通道中的不一致性，并且只允许来自未扰动通道的信息通过。我们的方法在单源稳健性方面明显优于现有的方法，在动作识别上获得了7.8-25.2%的收益，在目标检测上获得了19.7-48.2%的收益，在情感分析上获得了1.6-6.7%的收益，而在未受干扰(即干净的)数据上的性能没有下降。



## **28. Defense against adversarial attacks on deep convolutional neural networks through nonlocal denoising**

基于非局部去噪的深层卷积神经网络对抗攻击 cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12685v1)

**Authors**: Sandhya Aneja, Nagender Aneja, Pg Emeroylariffion Abas, Abdul Ghani Naim

**Abstracts**: Despite substantial advances in network architecture performance, the susceptibility of adversarial attacks makes deep learning challenging to implement in safety-critical applications. This paper proposes a data-centric approach to addressing this problem. A nonlocal denoising method with different luminance values has been used to generate adversarial examples from the Modified National Institute of Standards and Technology database (MNIST) and Canadian Institute for Advanced Research (CIFAR-10) data sets. Under perturbation, the method provided absolute accuracy improvements of up to 9.3% in the MNIST data set and 13% in the CIFAR-10 data set. Training using transformed images with higher luminance values increases the robustness of the classifier. We have shown that transfer learning is disadvantageous for adversarial machine learning. The results indicate that simple adversarial examples can improve resilience and make deep learning easier to apply in various applications.

摘要: 尽管网络架构的性能有了很大的进步，但敌意攻击的敏感性使得深度学习在安全关键型应用中的实施具有挑战性。本文提出了一种以数据为中心的方法来解决这个问题。一种不同亮度值的非局部去噪方法被用来从修改的国家标准与技术研究所(MNIST)数据库(MNIST)和加拿大高级研究院(CIFAR-10)数据集生成对抗性样本。在摄动下，该方法在MNIST数据集和CIFAR-10数据集上的绝对精度分别提高了9.3%和13%。使用具有较高亮度值的变换图像进行训练增加了分类器的稳健性。我们已经证明，迁移学习对对抗性机器学习是不利的。结果表明，简单的对抗性例子可以提高韧性，使深度学习更容易应用于各种应用中。



## **29. RSTAM: An Effective Black-Box Impersonation Attack on Face Recognition using a Mobile and Compact Printer**

RSTAM：一种有效的移动紧凑型打印机人脸识别黑盒模拟攻击 cs.CV

**SubmitDate**: 2022-06-25    [paper-pdf](http://arxiv.org/pdf/2206.12590v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao, Changhai Nie

**Abstracts**: Face recognition has achieved considerable progress in recent years thanks to the development of deep neural networks, but it has recently been discovered that deep neural networks are vulnerable to adversarial examples. This means that face recognition models or systems based on deep neural networks are also susceptible to adversarial examples. However, the existing methods of attacking face recognition models or systems with adversarial examples can effectively complete white-box attacks but not black-box impersonation attacks, physical attacks, or convenient attacks, particularly on commercial face recognition systems. In this paper, we propose a new method to attack face recognition models or systems called RSTAM, which enables an effective black-box impersonation attack using an adversarial mask printed by a mobile and compact printer. First, RSTAM enhances the transferability of the adversarial masks through our proposed random similarity transformation strategy. Furthermore, we propose a random meta-optimization strategy for ensembling several pre-trained face models to generate more general adversarial masks. Finally, we conduct experiments on the CelebA-HQ, LFW, Makeup Transfer (MT), and CASIA-FaceV5 datasets. The performance of the attacks is also evaluated on state-of-the-art commercial face recognition systems: Face++, Baidu, Aliyun, Tencent, and Microsoft. Extensive experiments show that RSTAM can effectively perform black-box impersonation attacks on face recognition models or systems.

摘要: 近年来，由于深度神经网络的发展，人脸识别取得了长足的进步，但最近发现，深度神经网络很容易受到对手例子的影响。这意味着，基于深度神经网络的人脸识别模型或系统也容易受到敌意例子的影响。然而，现有的利用对抗性例子攻击人脸识别模型或系统的方法可以有效地完成白盒攻击，而不能完成黑盒冒充攻击、物理攻击或便利攻击，特别是对商业人脸识别系统。在本文中，我们提出了一种新的攻击人脸识别模型或系统的方法RSTAM，它使用移动和紧凑型打印机打印的敌意面具来实现有效的黑盒模仿攻击。首先，RSTAM通过我们提出的随机相似变换策略增强了敌方面具的可转移性。此外，我们还提出了一种随机元优化策略来集成多个预先训练好的人脸模型，以生成更一般的对抗性面具。最后，我们在CelebA-HQ、LFW、Makeup Transfer(MT)和CASIA-FaceV5数据集上进行了实验。攻击的性能还在最先进的商业人脸识别系统上进行了评估：Face++、百度、阿里云、腾讯和微软。大量实验表明，RSTAM能够有效地对人脸识别模型或系统进行黑盒模拟攻击。



## **30. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

利用补丁处理防御视觉转换器的后门攻击 cs.CV

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12381v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstracts**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.

摘要: 与卷积神经网络相比，视觉转换器(VITS)具有完全不同的体系结构，具有明显更少的感应偏差。随着性能的提高，VITS的安全性和健壮性也具有重要的研究意义。与最近许多利用VITS对敌意例子的健壮性的工作不同，本文研究了一种典型的致因攻击，即后门攻击。我们首先检查VITS对各种后门攻击的脆弱性，发现VITS也很容易受到现有攻击的攻击。然而，我们观察到VITS的干净数据准确性和后门攻击成功率对位置编码之前的补丁变换有明显的响应。然后，基于这一发现，我们提出了一种VITS通过补丁处理来防御基于补丁和基于混合的触发后门攻击的有效方法。在包括CIFAR10、GTSRB和TinyImageNet在内的几个基准数据集上进行了性能评估，表明所提出的新型防御在缓解VITS后门攻击方面是非常成功的。据我们所知，本文提出了第一种利用VITS的独特特性来抵御后门攻击的防御策略。



## **31. Robustness of Explanation Methods for NLP Models**

NLP模型解释方法的稳健性 cs.CL

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12284v1)

**Authors**: Shriya Atmakuri, Tejas Chheda, Dinesh Kandula, Nishant Yadav, Taesung Lee, Hessel Tuinhof

**Abstracts**: Explanation methods have emerged as an important tool to highlight the features responsible for the predictions of neural networks. There is mounting evidence that many explanation methods are rather unreliable and susceptible to malicious manipulations. In this paper, we particularly aim to understand the robustness of explanation methods in the context of text modality. We provide initial insights and results towards devising a successful adversarial attack against text explanations. To our knowledge, this is the first attempt to evaluate the adversarial robustness of an explanation method. Our experiments show the explanation method can be largely disturbed for up to 86% of the tested samples with small changes in the input sentence and its semantics.

摘要: 解释方法已成为突出神经网络预测特征的重要工具。越来越多的证据表明，许多解释方法相当不可靠，容易受到恶意操纵。在这篇文章中，我们特别致力于理解解释方法在语篇情态语境中的稳健性。我们为设计对文本解释的成功的对抗性攻击提供了初步的见解和结果。据我们所知，这是第一次尝试评估解释方法的对抗性稳健性。我们的实验表明，在输入句子及其语义稍有变化的情况下，该解释方法可以对高达86%的测试样本产生很大的干扰。



## **32. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

属性遗忘：一种防御属性推理攻击的策略 cs.CR

Please note: As of June 24, 2022, we have discovered some flaws in  our experimental setup. The defense mechanism property unlearning is not as  strong as the experimental results in the current version of the paper  suggest. We will provide an updated version soon

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2205.08821v2)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.

摘要: 在机器学习模型的训练过程中，它们可能存储或“学习”比预测或分类任务实际需要的更多关于训练数据的信息。这被属性推理攻击所利用，该属性推理攻击的目的是从给定模型的训练数据中提取统计属性，而不访问训练数据本身。这些属性可以包括用于识别相机型号的图片质量、用于揭示产品目标受众的年龄分布、或用于改进计算机网络中的恶意软件攻击的所包括的主机类型。当攻击者有权访问所有模型参数时，即在白盒情况下，此攻击尤其准确。通过防御此类攻击，模型所有者能够确保他们的训练数据、相关属性以及他们的知识产权是保密的，即使他们故意共享他们的模型，例如协作训练，或者如果模型被泄露。在本文中，我们引入了属性遗忘，这是一种有效的防御白盒属性推理攻击的机制，独立于训练数据类型、模型任务或属性数量。属性遗忘通过系统地改变目标模型的训练权重和偏差来减轻属性推断攻击，使得对手无法提取所选的属性。我们在三个不同的数据集上经验地评估了属性遗忘，包括表格和图像数据，以及两种类型的人工神经网络。我们的结果表明，属性忘却在保护机器学习模型免受属性推理攻击方面是有效和可靠的，并且具有良好的隐私效用权衡。此外，我们的方法表明，该机制也有效地忘却了多个属性。



## **33. Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective**

深度神经网络的对抗健壮性：从形式验证的角度综述 cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12227v1)

**Authors**: Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong

**Abstracts**: Neural networks have been widely applied in security applications such as spam and phishing detection, intrusion prevention, and malware detection. This black-box method, however, often has uncertainty and poor explainability in applications. Furthermore, neural networks themselves are often vulnerable to adversarial attacks. For those reasons, there is a high demand for trustworthy and rigorous methods to verify the robustness of neural network models. Adversarial robustness, which concerns the reliability of a neural network when dealing with maliciously manipulated inputs, is one of the hottest topics in security and machine learning. In this work, we survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies. We also demonstrate representative techniques that have been applied in existing studies with a sample model. Finally, we discuss open questions for future research.

摘要: 神经网络已广泛应用于垃圾邮件和网络钓鱼检测、入侵防御和恶意软件检测等安全应用中。然而，这种黑箱方法在应用中往往具有不确定性和较差的可解释性。此外，神经网络本身往往容易受到敌意攻击。因此，对神经网络模型的稳健性验证方法提出了更高的要求。对抗健壮性是安全和机器学习领域中最热门的话题之一，它涉及到神经网络在处理恶意操作的输入时的可靠性。在这项工作中，我们综述了现有的神经网络对抗健壮性验证的文献，并收集了39个不同的研究工作，涉及机器学习、安全和软件工程领域。我们系统地分析了他们的方法，包括健壮性是如何形成的，使用了什么验证技术，以及每种技术的优点和局限性。为了全面理解这一主题，我们从正式验证的角度提供了一个分类法。我们根据属性规范、问题约简和推理策略对现有技术进行分类。我们还用一个样本模型演示了已在现有研究中应用的代表性技术。最后，我们讨论了未来研究的有待解决的问题。



## **34. Cluster Attack: Query-based Adversarial Attacks on Graphs with Graph-Dependent Priors**

簇攻击：图依赖先验图上基于查询的敌意攻击 cs.LG

IJCAI 2022 (Long Presentation)

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2109.13069v2)

**Authors**: Zhengyi Wang, Zhongkai Hao, Ziqiao Wang, Hang Su, Jun Zhu

**Abstracts**: While deep neural networks have achieved great success in graph analysis, recent work has shown that they are vulnerable to adversarial attacks. Compared with adversarial attacks on image classification, performing adversarial attacks on graphs is more challenging because of the discrete and non-differential nature of the adjacent matrix for a graph. In this work, we propose Cluster Attack -- a Graph Injection Attack (GIA) on node classification, which injects fake nodes into the original graph to degenerate the performance of graph neural networks (GNNs) on certain victim nodes while affecting the other nodes as little as possible. We demonstrate that a GIA problem can be equivalently formulated as a graph clustering problem; thus, the discrete optimization problem of the adjacency matrix can be solved in the context of graph clustering. In particular, we propose to measure the similarity between victim nodes by a metric of Adversarial Vulnerability, which is related to how the victim nodes will be affected by the injected fake node, and to cluster the victim nodes accordingly. Our attack is performed in a practical and unnoticeable query-based black-box manner with only a few nodes on the graphs that can be accessed. Theoretical analysis and extensive experiments demonstrate the effectiveness of our method by fooling the node classifiers with only a small number of queries.

摘要: 虽然深度神经网络在图分析方面取得了巨大的成功，但最近的研究表明，它们很容易受到对手的攻击。与图像分类中的对抗性攻击相比，由于图的邻接矩阵的离散和非可微性质，对图进行对抗性攻击具有更大的挑战性。在这项工作中，我们提出了一种针对节点分类的图注入攻击(GIA)--图注入攻击(GIA)，它将伪节点注入到原始图中，以降低图神经网络(GNN)在某些受害节点上的性能，同时尽可能地减少对其他节点的影响。我们证明了GIA问题可以等价地表示为图聚类问题，从而可以在图聚类的背景下解决邻接矩阵的离散优化问题。特别是，我们提出了通过敌意脆弱性来衡量受害节点之间的相似性，该度量与注入的伪节点将如何影响受害节点有关，并据此对受害节点进行聚类。我们的攻击是以一种实用的、不可察觉的基于查询的黑盒方式进行的，图上只有几个节点可以访问。理论分析和大量实验证明了该方法的有效性，仅用少量的查询就可以愚弄节点分类器。



## **35. An Improved Lattice-Based Ring Signature with Unclaimable Anonymity in the Standard Model**

一种改进的标准模型下不可否认匿名性的格环签名 cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12093v1)

**Authors**: Mingxing Hu, Weijiong Zhang, Zhen Liu

**Abstracts**: Ring signatures enable a user to sign messages on behalf of an arbitrary set of users, called the ring, without revealing exactly which member of that ring actually generated the signature. The signer-anonymity property makes ring signatures have been an active research topic. Recently, Park and Sealfon (CRYPTO 19) presented an important anonymity notion named signer-unclaimability and constructed a lattice-based ring signature scheme with unclaimable anonymity in the standard model, however, it did not consider the unforgeable w.r.t. adversarially-chosen-key attack (the public key ring of a signature may contain keys created by an adversary) and the signature size grows quadratically in the size of ring and message. In this work, we propose a new lattice-based ring signature scheme with unclaimable anonymity in the standard model. In particular, our work improves the security and efficiency of Park and Sealfons work, which is unforgeable w.r.t. adversarially-chosen-key attack, and the ring signature size grows linearly in the ring size.

摘要: 环签名使用户能够代表称为环的任意一组用户对消息进行签名，而不会确切地揭示该环中的哪个成员实际生成了签名。签名者匿名性使得环签名成为一个活跃的研究课题。最近，Park和Sealfon(Crypto 19)提出了一个重要的匿名性概念：签名者不可否认性，并在标准模型下构造了一个不可否认匿名性的格型环签名方案，但它没有考虑不可伪造性。恶意选择密钥攻击(签名的公钥环可能包含对手创建的密钥)，签名大小随着环和消息的大小呈二次曲线增长。在这项工作中，我们提出了一个新的基于格的环签名方案，在标准模型下具有不可否认的匿名性。特别是，我们的工作提高了公园和Sealfons工作的安全性和效率，这是无法伪造的W.r.t.恶意选择密钥攻击，且环签名大小随环大小线性增长。



## **36. Keep Your Transactions On Short Leashes**

在短时间内控制你的交易 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11974v1)

**Authors**: Bennet Yee

**Abstracts**: The adversary's goal in mounting Long Range Attacks (LRAs) is to fool potential victims into using and relying on a side chain, i.e., a false, alternate history of transactions, and into proposing transactions that end up harming themselves or others. Previous research work on LRAs on blockchain systems have used, at a high level, one of two approaches. They either try to (1) prevent the creation of a bogus side chain or (2) make it possible to distinguish such a side chain from the main consensus chain.   In this paper, we take a different approach. We start with the indistinguishability of side chains from the consensus chain -- for the eclipsed victim -- as a given and assume the potential victim will be fooled. Instead, we protect the victim via harm reduction applying "short leashes" to transactions. The leashes prevent transactions from being used in the wrong context.   The primary contribution of this paper is the design and analysis of leashes. A secondary contribution is the careful explication of the LRA threat model in the context of BAR fault tolerance, and using it to analyze related work to identify their limitations.

摘要: 对手发起远程攻击(LRA)的目的是欺骗潜在受害者使用和依赖侧链，即虚假的、替代的交易历史，并提出最终伤害自己或他人的交易。以前关于区块链系统上的LRA的研究工作在高水平上使用了两种方法之一。他们要么试图(1)防止虚假侧链的产生，要么(2)使这种侧链与主要共识链区分开来成为可能。在本文中，我们采取了一种不同的方法。我们从侧链和共识链的不可区分开始--对于黯然失色的受害者--作为给定的假设，并假设潜在的受害者将被愚弄。取而代之的是，我们通过减少伤害来保护受害者，对交易施加“短皮带”。捆绑可以防止交易在错误的上下文中使用。本文的主要贡献是对牵引带的设计和分析。第二个贡献是在BAR容错的背景下仔细解释了LRA威胁模型，并使用它来分析相关工作以确定它们的局限性。



## **37. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

将你的力量转向你：检测和减轻健壮的和通用的对抗性补丁攻击 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2108.05075v3)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks that inject arbitrary distortions within a bounded region of an image, can trigger misclassification in deep neural networks (DNNs). These attacks are robust (i.e., physically realizable) and universally malicious, and hence represent a severe security threat to real-world DNN-based systems.   This work proposes Jujutsu, a two-stage technique to detect and mitigate robust and universal adversarial patch attacks. We first observe that patch attacks often yield large influence on the prediction output in order to dominate the prediction on any input, and Jujutsu is built to expose this behavior for effective attack detection. For mitigation, we observe that patch attacks corrupt only a localized region while the remaining contents are unperturbed, based on which Jujutsu leverages GAN-based image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image for correct prediction.   We evaluate Jujutsu on four diverse datasets and show that it achieves superior performance and significantly outperforms four leading defenses. Jujutsu can further defend against physical-world attacks, attacks that target diverse classes, and adaptive attacks. Our code is available at https://github.com/DependableSystemsLab/Jujutsu.

摘要: 对抗性补丁攻击在图像的有界区域内注入任意扭曲，可在深度神经网络(DNN)中引发错误分类。这些攻击是健壮的(即，物理上可实现的)并且普遍是恶意的，因此对现实世界中基于DNN的系统构成了严重的安全威胁。这项工作提出了Jujutsu，这是一种两阶段技术，用于检测和缓解健壮且通用的敌意补丁攻击。我们首先观察到补丁攻击通常会对预测输出产生很大的影响，以便在任何输入上控制预测，Jujutsu被构建来暴露这种行为以进行有效的攻击检测。为了缓解攻击，我们观察到补丁攻击只破坏了局部区域，而其余内容不受干扰，基于此，Jujutsu利用基于GaN的图像修复来合成被攻击破坏的像素中的语义内容，并重建出正确的预测。我们在四个不同的数据集上对Jujutsu进行了评估，结果表明它取得了优越的性能，并显著超过了四个领先的防御系统。魔术可以进一步防御物理世界的攻击，针对不同职业的攻击，以及适应性攻击。我们的代码可以在https://github.com/DependableSystemsLab/Jujutsu.上找到



## **38. Probabilistically Resilient Multi-Robot Informative Path Planning**

概率弹性多机器人信息路径规划 cs.RO

9 pages, 6 figures, submitted to IEEE Robotics and Automation Letters  (RA-L)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11789v1)

**Authors**: Remy Wehbe, Ryan K. Williams

**Abstracts**: In this paper, we solve a multi-robot informative path planning (MIPP) task under the influence of uncertain communication and adversarial attackers. The goal is to create a multi-robot system that can learn and unify its knowledge of an unknown environment despite the presence of corrupted robots sharing malicious information. We use a Gaussian Process (GP) to model our unknown environment and define informativeness using the metric of mutual information. The objectives of our MIPP task is to maximize the amount of information collected by the team while maximizing the probability of resilience to attack. Unfortunately, these objectives are at odds especially when exploring large environments which necessitates disconnections between robots. As a result, we impose a probabilistic communication constraint that allows robots to meet intermittently and resiliently share information, and then act to maximize collected information during all other times. To solve our problem, we select meeting locations with the highest probability of resilience and use a sequential greedy algorithm to optimize paths for robots to explore. Finally, we show the validity of our results by comparing the learning ability of well-behaving robots applying resilient vs. non-resilient MIPP algorithms.

摘要: 本文解决了通信不确定和敌方攻击者影响下的多机器人信息路径规划问题。其目标是创建一种多机器人系统，即使存在共享恶意信息的被破坏的机器人，也可以学习和统一其对未知环境的知识。我们使用高斯过程(GP)对未知环境进行建模，并使用互信息度量来定义信息量。我们的MIPP任务的目标是最大化团队收集的信息量，同时最大化抵抗攻击的可能性。不幸的是，这些目标是不一致的，特别是在探索需要断开机器人之间连接的大型环境时。因此，我们施加了一个概率通信约束，允许机器人间歇性地会面并弹性地共享信息，然后在所有其他时间采取行动最大限度地收集信息。为了解决我们的问题，我们选择具有最高弹性的会议地点，并使用顺序贪婪算法来优化供机器人探索的路径。最后，我们通过比较弹性和非弹性MIPP算法对行为良好的机器人的学习能力，证明了我们的结果的有效性。



## **39. Towards End-to-End Private Automatic Speaker Recognition**

走向端到端的私人自动说话人识别 eess.AS

Accepted for publication at Interspeech 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11750v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstracts**: The development of privacy-preserving automatic speaker verification systems has been the focus of a number of studies with the intent of allowing users to authenticate themselves without risking the privacy of their voice. However, current privacy-preserving methods assume that the template voice representations (or speaker embeddings) used for authentication are extracted locally by the user. This poses two important issues: first, knowledge of the speaker embedding extraction model may create security and robustness liabilities for the authentication system, as this knowledge might help attackers in crafting adversarial examples able to mislead the system; second, from the point of view of a service provider the speaker embedding extraction model is arguably one of the most valuable components in the system and, as such, disclosing it would be highly undesirable. In this work, we show how speaker embeddings can be extracted while keeping both the speaker's voice and the service provider's model private, using Secure Multiparty Computation. Further, we show that it is possible to obtain reasonable trade-offs between security and computational cost. This work is complementary to those showing how authentication may be performed privately, and thus can be considered as another step towards fully private automatic speaker recognition.

摘要: 保护隐私的自动说话人验证系统的开发一直是许多研究的重点，目的是允许用户在不危及其语音隐私的情况下验证自己。然而，当前的隐私保护方法假设用于认证的模板语音表示(或说话人嵌入)是由用户本地提取的。这提出了两个重要问题：第一，知道说话人嵌入提取模型可能会给认证系统带来安全和健壮性方面的风险，因为这种知识可能帮助攻击者制作能够误导系统的敌意例子；第二，从服务提供商的角度来看，说话人嵌入提取模型可能是系统中最有价值的组件之一，因此，公开它将是非常不可取的。在这项工作中，我们展示了如何使用安全多方计算在保持说话人的语音和服务提供商的模型隐私的同时提取说话人嵌入。此外，我们还证明了在安全性和计算成本之间取得合理的折衷是可能的。这项工作是对那些展示如何私下执行身份验证的工作的补充，因此可以被视为迈向完全私密自动说话人识别的又一步。



## **40. BERT Rankers are Brittle: a Study using Adversarial Document Perturbations**

Bert Rankers是脆弱的：一项使用对抗性文件扰动的研究 cs.IR

To appear in ICTIR 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11724v1)

**Authors**: Yumeng Wang, Lijun Lyu, Avishek Anand

**Abstracts**: Contextual ranking models based on BERT are now well established for a wide range of passage and document ranking tasks. However, the robustness of BERT-based ranking models under adversarial inputs is under-explored. In this paper, we argue that BERT-rankers are not immune to adversarial attacks targeting retrieved documents given a query. Firstly, we propose algorithms for adversarial perturbation of both highly relevant and non-relevant documents using gradient-based optimization methods. The aim of our algorithms is to add/replace a small number of tokens to a highly relevant or non-relevant document to cause a large rank demotion or promotion. Our experiments show that a small number of tokens can already result in a large change in the rank of a document. Moreover, we find that BERT-rankers heavily rely on the document start/head for relevance prediction, making the initial part of the document more susceptible to adversarial attacks. More interestingly, we find a small set of recurring adversarial words that when added to documents result in successful rank demotion/promotion of any relevant/non-relevant document respectively. Finally, our adversarial tokens also show particular topic preferences within and across datasets, exposing potential biases from BERT pre-training or downstream datasets.

摘要: 基于BERT的上下文排名模型现在已经很好地建立了用于广泛的段落和文档排名任务。然而，基于BERT的排序模型在对抗性输入下的稳健性还没有得到充分的研究。在这篇文章中，我们认为BERT排名者也不能幸免于针对给定查询的检索文档的对抗性攻击。首先，我们使用基于梯度的优化方法提出了针对高度相关和无关文档的对抗性扰动的算法。我们的算法的目的是在高度相关或不相关的文档中添加/替换少量令牌，从而导致较大的排名降级或提升。我们的实验表明，少量的标记已经可以导致文档的排名发生很大的变化。此外，我们发现BERT排名者严重依赖文档START/HEAD进行相关性预测，使得文档的开头部分更容易受到对手攻击。更有趣的是，我们发现了一小部分重复出现的敌意单词，当添加到文档中时，这些单词会分别成功地对任何相关/不相关的文档进行排名降级/提升。最后，我们的对抗性令牌还显示了数据集内和数据集之间的特定主题偏好，暴露了来自BERT预训练或下游数据集的潜在偏差。



## **41. Adversarial Zoom Lens: A Novel Physical-World Attack to DNNs**

对抗性变焦镜头：一种新的物理世界对DNN的攻击 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12251v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Although deep neural networks (DNNs) are known to be fragile, no one has studied the effects of zooming-in and zooming-out of images in the physical world on DNNs performance. In this paper, we demonstrate a novel physical adversarial attack technique called Adversarial Zoom Lens (AdvZL), which uses a zoom lens to zoom in and out of pictures of the physical world, fooling DNNs without changing the characteristics of the target object. The proposed method is so far the only adversarial attack technique that does not add physical adversarial perturbation attack DNNs. In a digital environment, we construct a data set based on AdvZL to verify the antagonism of equal-scale enlarged images to DNNs. In the physical environment, we manipulate the zoom lens to zoom in and out of the target object, and generate adversarial samples. The experimental results demonstrate the effectiveness of AdvZL in both digital and physical environments. We further analyze the antagonism of the proposed data set to the improved DNNs. On the other hand, we provide a guideline for defense against AdvZL by means of adversarial training. Finally, we look into the threat possibilities of the proposed approach to future autonomous driving and variant attack ideas similar to the proposed attack.

摘要: 尽管深度神经网络(DNN)被认为是脆弱的，但还没有人研究物理世界中图像的放大和缩小对DNN性能的影响。在本文中，我们展示了一种新的物理对抗性攻击技术，称为对抗性变焦镜头(AdvZL)，它使用变焦镜头来放大和缩小物理世界的图像，在不改变目标对象特征的情况下愚弄DNN。该方法是迄今为止唯一一种不添加物理对抗性扰动攻击DNN的对抗性攻击技术。在数字环境下，我们构建了一个基于AdvZL的数据集，以验证等比例放大图像对DNN的对抗。在物理环境中，我们操纵变焦镜头来放大和缩小目标对象，并生成对抗性样本。实验结果证明了AdvZL在数字和物理环境中的有效性。我们进一步分析了所提出的数据集对改进的DNN的对抗性。另一方面，我们通过对抗性训练的方式提供了防御AdvZL的指导方针。最后，我们展望了所提出的方法对未来自动驾驶的威胁可能性，以及类似于所提出的攻击的不同攻击思想。



## **42. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.06761v2)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **43. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的边界训练数据重构 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2201.12383v4)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 在ML中，差异隐私被广泛接受为防止数据泄露的事实上的方法，传统观点认为，它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。本文首先在形式化威胁模型下给出了DP机制抵抗训练数据重构攻击的语义保证。我们发现，两种不同的隐私记账方法--Renyi Differential Privacy和Fisher信息泄漏--都提供了对数据重构攻击的强大语义保护。



## **44. A Framework for Understanding Model Extraction Attack and Defense**

一种理解模型提取攻击与防御的框架 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11480v1)

**Authors**: Xun Xian, Mingyi Hong, Jie Ding

**Abstracts**: The privacy of machine learning models has become a significant concern in many emerging Machine-Learning-as-a-Service applications, where prediction services based on well-trained models are offered to users via pay-per-query. The lack of a defense mechanism can impose a high risk on the privacy of the server's model since an adversary could efficiently steal the model by querying only a few `good' data points. The interplay between a server's defense and an adversary's attack inevitably leads to an arms race dilemma, as commonly seen in Adversarial Machine Learning. To study the fundamental tradeoffs between model utility from a benign user's view and privacy from an adversary's view, we develop new metrics to quantify such tradeoffs, analyze their theoretical properties, and develop an optimization problem to understand the optimal adversarial attack and defense strategies. The developed concepts and theory match the empirical findings on the `equilibrium' between privacy and utility. In terms of optimization, the key ingredient that enables our results is a unified representation of the attack-defense problem as a min-max bi-level problem. The developed results will be demonstrated by examples and experiments.

摘要: 在许多新兴的机器学习即服务应用中，机器学习模型的隐私已经成为一个重要的问题，其中基于训练有素的模型的预测服务通过按查询付费的方式提供给用户。缺乏防御机制可能会给服务器模型的隐私带来很高的风险，因为攻击者只需查询几个“好”的数据点就可以有效地窃取模型。服务器的防御和对手的攻击之间的相互作用不可避免地导致了军备竞赛的两难境地，就像在对抗性机器学习中常见的那样。为了从良性用户的角度研究模型效用和从对手的角度研究隐私之间的基本权衡，我们开发了新的度量来量化这种权衡，分析了它们的理论性质，并开发了一个优化问题来理解最优的对抗性攻击和防御策略。所发展的概念和理论与关于隐私和效用之间的“平衡”的经验研究结果相吻合。在优化方面，使我们的结果得以实现的关键因素是将攻防问题统一表示为最小-最大双层问题。所开发的结果将通过实例和实验进行验证。



## **45. InfoAT: Improving Adversarial Training Using the Information Bottleneck Principle**

InfoAT：利用信息瓶颈原理改进对抗性训练 cs.LG

Published in: IEEE Transactions on Neural Networks and Learning  Systems ( Early Access )

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12292v1)

**Authors**: Mengting Xu, Tao Zhang, Zhongnian Li, Daoqiang Zhang

**Abstracts**: Adversarial training (AT) has shown excellent high performance in defending against adversarial examples. Recent studies demonstrate that examples are not equally important to the final robustness of models during AT, that is, the so-called hard examples that can be attacked easily exhibit more influence than robust examples on the final robustness. Therefore, guaranteeing the robustness of hard examples is crucial for improving the final robustness of the model. However, defining effective heuristics to search for hard examples is still difficult. In this article, inspired by the information bottleneck (IB) principle, we uncover that an example with high mutual information of the input and its associated latent representation is more likely to be attacked. Based on this observation, we propose a novel and effective adversarial training method (InfoAT). InfoAT is encouraged to find examples with high mutual information and exploit them efficiently to improve the final robustness of models. Experimental results show that InfoAT achieves the best robustness among different datasets and models in comparison with several state-of-the-art methods.

摘要: 对抗性训练(AT)在防御对抗性例子方面表现出了出色的高性能。最近的研究表明，在AT过程中，样本对模型的最终稳健性并不是同样重要，即所谓的易受攻击的硬样本对最终稳健性的影响比健壮样本更大。因此，保证硬样本的稳健性是提高模型最终稳健性的关键。然而，定义有效的启发式算法来搜索困难的例子仍然是困难的。在本文中，受信息瓶颈(IB)原理的启发，我们发现输入及其关联的潜在表示具有高互信息的示例更容易受到攻击。基于此，我们提出了一种新颖而有效的对抗性训练方法(InfoAT)。InfoAT被鼓励寻找具有高互信息的例子，并有效地利用它们来提高模型的最终稳健性。实验结果表明，与几种最先进的方法相比，InfoAT在不同的数据集和模型中获得了最好的稳健性。



## **46. Incorporating Hidden Layer representation into Adversarial Attacks and Defences**

在对抗性攻击和防御中引入隐藏层表示 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2011.14045v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: In this paper, we propose a defence strategy to improve adversarial robustness by incorporating hidden layer representation. The key of this defence strategy aims to compress or filter input information including adversarial perturbation. And this defence strategy can be regarded as an activation function which can be applied to any kind of neural network. We also prove theoretically the effectiveness of this defense strategy under certain conditions. Besides, incorporating hidden layer representation we propose three types of adversarial attacks to generate three types of adversarial examples, respectively. The experiments show that our defence method can significantly improve the adversarial robustness of deep neural networks which achieves the state-of-the-art performance even though we do not adopt adversarial training.

摘要: 在本文中，我们提出了一种防御策略，通过引入隐含层表示来提高对手的稳健性。这种防御策略的关键是压缩或过滤输入信息，包括对抗性扰动。这种防御策略可以看作是一种激活函数，可以应用于任何类型的神经网络。在一定条件下，我们还从理论上证明了该防御策略的有效性。此外，结合隐含层表示，我们提出了三种类型的对抗性攻击，分别生成三种类型的对抗性实例。实验表明，在不采用对抗性训练的情况下，我们的防御方法能够显着提高深层神经网络的对抗性，达到了最好的性能。



## **47. Adversarial Learning with Cost-Sensitive Classes**

成本敏感类的对抗性学习 cs.LG

12 pages

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2101.12372v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: It is necessary to improve the performance of some special classes or to particularly protect them from attacks in adversarial learning. This paper proposes a framework combining cost-sensitive classification and adversarial learning together to train a model that can distinguish between protected and unprotected classes, such that the protected classes are less vulnerable to adversarial examples. We find in this framework an interesting phenomenon during the training of deep neural networks, called Min-Max property, that is, the absolute values of most parameters in the convolutional layer approach zero while the absolute values of a few parameters are significantly larger becoming bigger. Based on this Min-Max property which is formulated and analyzed in a view of random distribution, we further build a new defense model against adversarial examples for adversarial robustness improvement. An advantage of the built model is that it performs better than the standard one and can combine with adversarial training to achieve an improved performance. It is experimentally confirmed that, regarding the average accuracy of all classes, our model is almost as same as the existing models when an attack does not occur and is better than the existing models when an attack occurs. Specifically, regarding the accuracy of protected classes, the proposed model is much better than the existing models when an attack occurs.

摘要: 在对抗性学习中，有必要提高某些特殊类的表现，或特别保护它们免受攻击。本文提出了一种结合代价敏感分类和对抗性学习的框架，以训练一个能够区分保护类和非保护类的模型，从而使受保护类不太容易受到对抗性例子的影响。在该框架中，我们发现了深层神经网络训练过程中一个有趣的现象，称为Min-Max性质，即卷积层中大部分参数的绝对值趋于零，而少数参数的绝对值明显变大。基于这种从随机分布的角度来描述和分析的Min-Max性质，我们进一步构建了一种新的对抗实例的防御模型，以提高对抗的健壮性。建立的模型的一个优点是它的性能比标准模型更好，并且可以与对抗性训练相结合，以实现更好的性能。实验证实，在所有类别的平均准确率方面，当攻击不发生时，我们的模型与现有模型几乎相同，而当攻击发生时，我们的模型优于现有模型。具体地说，在保护类的准确性方面，当攻击发生时，所提出的模型比现有的模型要好得多。



## **48. Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles**

通过学习生成虚假用户配置文件来攻击黑盒推荐系统 cs.IR

Accepted by TNNLS. 15 pages, 8 figures

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11433v1)

**Authors**: Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, Hui Li

**Abstracts**: Due to the pivotal role of Recommender Systems (RS) in guiding customers towards the purchase, there is a natural motivation for unscrupulous parties to spoof RS for profits. In this paper, we study Shilling Attack where an adversarial party injects a number of fake user profiles for improper purposes. Conventional Shilling Attack approaches lack attack transferability (i.e., attacks are not effective on some victim RS models) and/or attack invisibility (i.e., injected profiles can be easily detected). To overcome these issues, we present Leg-UP, a novel attack model based on the Generative Adversarial Network. Leg-UP learns user behavior patterns from real users in the sampled ``templates'' and constructs fake user profiles. To simulate real users, the generator in Leg-UP directly outputs discrete ratings. To enhance attack transferability, the parameters of the generator are optimized by maximizing the attack performance on a surrogate RS model. To improve attack invisibility, Leg-UP adopts a discriminator to guide the generator to generate undetectable fake user profiles. Experiments on benchmarks have shown that Leg-UP exceeds state-of-the-art Shilling Attack methods on a wide range of victim RS models. The source code of our work is available at: https://github.com/XMUDM/ShillingAttack.

摘要: 由于推荐系统在引导消费者购买方面起着举足轻重的作用，不道德的人有一个自然的动机来欺骗推荐系统以获取利润。在这篇文章中，我们研究了恶意用户出于不正当目的注入大量虚假用户配置文件的先令攻击。传统的先令攻击方法缺乏攻击的可转移性(即，攻击在某些受害者RS模型上无效)和/或攻击的不可见性(即，可以很容易地检测到注入的配置文件)。为了克服这些问题，我们提出了一种基于产生式对抗网络的新型攻击模型--LEG-UP。Leg-Up从采样的“模板”中的真实用户那里学习用户行为模式，并构建虚假的用户配置文件。为了模拟真实用户，立式发电机直接输出离散的额定值。为了增强攻击的可转移性，在代理RS模型上通过最大化攻击性能来优化生成器的参数。为了提高攻击的不可见性，Leg-Up采用了一个鉴别器来引导生成器生成无法检测的虚假用户配置文件。基准测试实验表明，在广泛的受害者RS模型上，Leg-Up攻击方法超过了最先进的先令攻击方法。我们工作的源代码可以在https://github.com/XMUDM/ShillingAttack.上找到



## **49. Making Generated Images Hard To Spot: A Transferable Attack On Synthetic Image Detectors**

使生成的图像难以识别：对合成图像检测器的可转移攻击 cs.CV

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2104.12069v2)

**Authors**: Xinwei Zhao, Matthew C. Stamm

**Abstracts**: Visually realistic GAN-generated images have recently emerged as an important misinformation threat. Research has shown that these synthetic images contain forensic traces that are readily identifiable by forensic detectors. Unfortunately, these detectors are built upon neural networks, which are vulnerable to recently developed adversarial attacks. In this paper, we propose a new anti-forensic attack capable of fooling GAN-generated image detectors. Our attack uses an adversarially trained generator to synthesize traces that these detectors associate with real images. Furthermore, we propose a technique to train our attack so that it can achieve transferability, i.e. it can fool unknown CNNs that it was not explicitly trained against. We evaluate our attack through an extensive set of experiments, where we show that our attack can fool eight state-of-the-art detection CNNs with synthetic images created using seven different GANs, and outperform other alternative attacks.

摘要: 视觉逼真的GaN生成的图像最近已经成为一种重要的错误信息威胁。研究表明，这些合成图像包含法医探测器可以很容易识别的法医痕迹。不幸的是，这些探测器是建立在神经网络的基础上的，而神经网络很容易受到最近发展起来的对抗性攻击。在本文中，我们提出了一种新的反取证攻击，能够欺骗GaN生成的图像检测器。我们的攻击使用一个经过敌意训练的生成器来合成这些检测器与真实图像相关联的痕迹。此外，我们还提出了一种技术来训练我们的攻击，以便它能够实现可转移性，即它可以欺骗它没有明确训练针对的未知CNN。我们通过一组广泛的实验来评估我们的攻击，其中我们表明我们的攻击可以通过使用7个不同的GAN创建的合成图像来欺骗8个最先进的检测CNN，并且性能优于其他替代攻击。



## **50. AdvSmo: Black-box Adversarial Attack by Smoothing Linear Structure of Texture**

AdvSmo：平滑纹理线性结构的黑盒对抗性攻击 cs.CV

6 pages,3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10988v1)

**Authors**: Hui Xia, Rui Zhang, Shuliang Jiang, Zi Kang

**Abstracts**: Black-box attacks usually face two problems: poor transferability and the inability to evade the adversarial defense. To overcome these shortcomings, we create an original approach to generate adversarial examples by smoothing the linear structure of the texture in the benign image, called AdvSmo. We construct the adversarial examples without relying on any internal information to the target model and design the imperceptible-high attack success rate constraint to guide the Gabor filter to select appropriate angles and scales to smooth the linear texture from the input images to generate adversarial examples. Benefiting from the above design concept, AdvSmo will generate adversarial examples with strong transferability and solid evasiveness. Finally, compared to the four advanced black-box adversarial attack methods, for the eight target models, the results show that AdvSmo improves the average attack success rate by 9% on the CIFAR-10 and 16% on the Tiny-ImageNet dataset compared to the best of these attack methods.

摘要: 黑盒攻击通常面临两个问题：可转移性差和无法躲避对手的防御。为了克服这些缺点，我们创建了一种新颖的方法，通过平滑良性图像中纹理的线性结构来生成对抗性示例，称为AdvSmo。我们在不依赖目标模型任何内部信息的情况下构造敌意样本，并设计了不可察觉的高攻击成功率约束来指导Gabor滤波器选择合适的角度和尺度来平滑输入图像中的线性纹理来生成敌意样本。受益于上述设计理念，AdvSmo将生成具有很强的可转移性和坚实的规避能力的对抗性范例。最后，与四种先进的黑盒对抗攻击方法相比，对于8个目标模型，结果表明，AdvSmo在CIFAR-10上的平均攻击成功率比这些攻击方法中最好的方法提高了9%，在Tiny-ImageNet数据集上的平均攻击成功率提高了16%。



