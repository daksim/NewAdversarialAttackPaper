# Latest Adversarial Attack Papers
**update at 2024-02-18 11:12:29**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents**

颤抖的纸牌屋？映射针对语言代理的对抗性攻击 cs.CL

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10196v1) [paper-pdf](http://arxiv.org/pdf/2402.10196v1)

**Authors**: Lingbo Mo, Zeyi Liao, Boyuan Zheng, Yu Su, Chaowei Xiao, Huan Sun

**Abstract**: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for agents with three major components: Perception, Brain, and Action. Under this framework, we present a comprehensive discussion and propose 12 potential attack scenarios against different components of an agent, covering different attack strategies (e.g., input manipulation, adversarial demonstrations, jailbreaking, backdoors). We also draw connections to successful attack strategies previously applied to LLMs. We emphasize the urgency to gain a thorough understanding of language agent risks before their widespread deployment.

摘要: 由大型语言模型(LLM)驱动的语言代理经历了爆炸性的发展。他们将语言作为思维和交流的媒介的能力，带来了令人难以置信的灵活性和多功能性。人们迅速利用这一能力将LLMS连接到各种外部组件和环境：数据库、工具、互联网、机器人化身等。许多人认为，一种前所未有的强大自动化技术正在出现。然而，新的自动化技术也伴随着新的安全风险，特别是对于语言代理这样复杂的系统。它们的发展和部署的速度和规模与我们对其安全风险的理解之间存在着令人惊讶的巨大差距。我们是在建造一座纸牌房子吗？在这份立场文件中，我们提出了第一次系统地绘制针对语言代理的对抗性攻击的努力。我们首先提出了一个统一的概念框架，包括三个主要组成部分：感知、大脑和行动。在这个框架下，我们对代理的不同组件进行了全面的讨论，并提出了12种潜在的攻击方案，涵盖了不同的攻击策略(例如，输入操纵、对抗性演示、越狱、后门)。我们还将其与以前应用于LLM的成功攻击策略联系起来。我们强调，在广泛部署语言诱导剂风险之前，迫切需要彻底了解这些风险。



## **2. Transaction Capacity, Security and Latency in Blockchains**

区块链中的交易容量、安全性和延迟 cs.CR

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10138v1) [paper-pdf](http://arxiv.org/pdf/2402.10138v1)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: We analyze how secure a block is after the block becomes k-deep, i.e., security-latency, for Nakamoto consensus under an exponential network delay model. We give parameter regimes for which transactions are safe when sufficiently deep in the chain. We compare our results for Nakamoto consensus under bounded network delay models and obtain analogous bounds for safety violation threshold. Next, modeling the blockchain system as a batch service queue with exponential network delay, we connect the security-latency analysis to sustainable transaction rate of the queue system. As our model assumes exponential network delay, batch service queue models give a meaningful trade-off between transaction capacity, security and latency. As adversary can attack the queue service to hamper the service process, we consider two different attacks for adversary. In an extreme scenario, we modify the selfish-mining attack for this purpose and consider its effect on the sustainable transaction rate of the queue.

摘要: 在指数网络延迟模型下，我们分析了块变得k深后的安全程度，即Nakamoto共识下的安全延迟。我们给出了交易在链中足够深时是安全的参数机制。我们比较了有界网络延迟模型下Nakamoto共识的结果，得到了类似的安全违规阈值的界。其次，将区块链系统建模为具有指数网络延迟的批处理服务队列，将安全延迟分析与队列系统的可持续交易率联系起来。由于我们的模型假设网络延迟为指数型，批处理服务队列模型在事务处理能力、安全性和延迟之间提供了有意义的折衷。由于敌手可以攻击队列服务来阻碍服务过程，我们考虑了两种不同的攻击方式。在一个极端的场景中，我们为此目的修改了自私挖掘攻击，并考虑了它对队列可持续事务率的影响。



## **3. Indiscriminate Data Poisoning Attacks on Neural Networks**

对神经网络的不分青红皂白的数据中毒攻击 cs.LG

Accepted to TMLR in 2022

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2204.09092v2) [paper-pdf](http://arxiv.org/pdf/2204.09092v2)

**Authors**: Yiwei Lu, Gautam Kamath, Yaoliang Yu

**Abstract**: Data poisoning attacks, in which a malicious adversary aims to influence a model by injecting "poisoned" data into the training process, have attracted significant recent attention. In this work, we take a closer look at existing poisoning attacks and connect them with old and new algorithms for solving sequential Stackelberg games. By choosing an appropriate loss function for the attacker and optimizing with algorithms that exploit second-order information, we design poisoning attacks that are effective on neural networks. We present efficient implementations that exploit modern auto-differentiation packages and allow simultaneous and coordinated generation of tens of thousands of poisoned points, in contrast to existing methods that generate poisoned points one by one. We further perform extensive experiments that empirically explore the effect of data poisoning attacks on deep neural networks.

摘要: 数据中毒攻击是指恶意对手通过在训练过程中注入“有毒”数据来影响模型的攻击，最近引起了极大的关注。在这项工作中，我们仔细研究现有的中毒攻击，并将它们与解决连续Stackelberg博弈的旧算法和新算法联系起来。通过为攻击者选择合适的损失函数，并利用二阶信息的算法进行优化，设计出对神经网络有效的中毒攻击。我们提供了利用现代自动区分包并允许同时和协调地生成数万个毒点的高效实现，与逐个生成毒点的现有方法形成对比。我们进一步进行了大量的实验，经验地探索了数据中毒攻击对深度神经网络的影响。



## **4. How Much Does Each Datapoint Leak Your Privacy? Quantifying the Per-datum Membership Leakage**

每个数据点泄露了多少您的隐私？量化每一数据的成员泄漏 cs.LG

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.10065v1) [paper-pdf](http://arxiv.org/pdf/2402.10065v1)

**Authors**: Achraf Azize, Debabrota Basu

**Abstract**: We study the per-datum Membership Inference Attacks (MIAs), where an attacker aims to infer whether a fixed target datum has been included in the input dataset of an algorithm and thus, violates privacy. First, we define the membership leakage of a datum as the advantage of the optimal adversary targeting to identify it. Then, we quantify the per-datum membership leakage for the empirical mean, and show that it depends on the Mahalanobis distance between the target datum and the data-generating distribution. We further assess the effect of two privacy defences, i.e. adding Gaussian noise and sub-sampling. We quantify exactly how both of them decrease the per-datum membership leakage. Our analysis builds on a novel proof technique that combines an Edgeworth expansion of the likelihood ratio test and a Lindeberg-Feller central limit theorem. Our analysis connects the existing likelihood ratio and scalar product attacks, and also justifies different canary selection strategies used in the privacy auditing literature. Finally, our experiments demonstrate the impacts of the leakage score, the sub-sampling ratio and the noise scale on the per-datum membership leakage as indicated by the theory.

摘要: 研究了基于数据的成员关系推理攻击(MIA)，攻击者的目的是推断算法的输入数据集中是否包含了固定的目标数据，从而侵犯了隐私。首先，我们将数据的成员泄漏定义为最优对手目标识别的优势。然后，我们量化了经验平均值的每数据成员泄漏，并表明它取决于目标数据和数据生成分布之间的马氏距离。我们进一步评估了两种隐私防御措施的效果，即添加高斯噪声和二次采样。我们精确地量化了它们是如何减少每数据成员泄漏的。我们的分析建立在一种新的证明技术上，该技术结合了似然比检验的Edgeworth展开和Lindeberg-Feller中心极限定理。我们的分析将现有的似然比和标量积攻击联系起来，并证明了隐私审计文献中使用的不同金丝雀选择策略的合理性。最后，通过实验验证了泄漏分数、子采样率和噪声尺度对每数据成员泄漏的影响。



## **5. Protect Your Score: Contact Tracing With Differential Privacy Guarantees**

保护您的分数：具有差异隐私保证的联系人跟踪 cs.CR

Accepted to The 38th Annual AAAI Conference on Artificial  Intelligence (AAAI 2024)

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2312.11581v2) [paper-pdf](http://arxiv.org/pdf/2312.11581v2)

**Authors**: Rob Romijnders, Christos Louizos, Yuki M. Asano, Max Welling

**Abstract**: The pandemic in 2020 and 2021 had enormous economic and societal consequences, and studies show that contact tracing algorithms can be key in the early containment of the virus. While large strides have been made towards more effective contact tracing algorithms, we argue that privacy concerns currently hold deployment back. The essence of a contact tracing algorithm constitutes the communication of a risk score. Yet, it is precisely the communication and release of this score to a user that an adversary can leverage to gauge the private health status of an individual. We pinpoint a realistic attack scenario and propose a contact tracing algorithm with differential privacy guarantees against this attack. The algorithm is tested on the two most widely used agent-based COVID19 simulators and demonstrates superior performance in a wide range of settings. Especially for realistic test scenarios and while releasing each risk score with epsilon=1 differential privacy, we achieve a two to ten-fold reduction in the infection rate of the virus. To the best of our knowledge, this presents the first contact tracing algorithm with differential privacy guarantees when revealing risk scores for COVID19.

摘要: 2020年和2021年的大流行造成了巨大的经济和社会后果，研究表明，接触者追踪算法可能是早期遏制病毒的关键。虽然在更有效的接触追踪算法方面已经取得了很大进展，但我们认为，隐私问题目前阻碍了部署。接触者跟踪算法的本质构成了风险分值的通信。然而，对手恰恰可以利用该分数向用户传达和发布该分数来衡量个人的私人健康状态。我们针对一个真实的攻击场景，提出了一种针对这种攻击的具有不同隐私保证的联系人跟踪算法。该算法在两个最广泛使用的基于代理的COVID19模拟器上进行了测试，并在广泛的设置范围内展示了优越的性能。特别是对于现实的测试场景，在发布具有epsilon=1差异隐私的每个风险分数的同时，我们将病毒的感染率降低了两到十倍。据我们所知，这是第一个在揭示COVID19风险分数时具有不同隐私保证的接触者跟踪算法。



## **6. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别敌意音频样本 cs.SD

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2305.17000v2) [paper-pdf](http://arxiv.org/pdf/2305.17000v2)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99\% and 97\%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在接收器操作特征下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **7. Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis**

对抗性量子机器学习：信息论泛化分析 quant-ph

10 pages, 2 figures. Fixed a typo (wrong inequality sign) in lemma 2  and extended to cover the whole range of values of p. Added reference on  inequalities in trace norms

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.00176v2) [paper-pdf](http://arxiv.org/pdf/2402.00176v2)

**Authors**: Petros Georgiou, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: In a manner analogous to their classical counterparts, quantum classifiers are vulnerable to adversarial attacks that perturb their inputs. A promising countermeasure is to train the quantum classifier by adopting an attack-aware, or adversarial, loss function. This paper studies the generalization properties of quantum classifiers that are adversarially trained against bounded-norm white-box attacks. Specifically, a quantum adversary maximizes the classifier's loss by transforming an input state $\rho(x)$ into a state $\lambda$ that is $\epsilon$-close to the original state $\rho(x)$ in $p$-Schatten distance. Under suitable assumptions on the quantum embedding $\rho(x)$, we derive novel information-theoretic upper bounds on the generalization error of adversarially trained quantum classifiers for $p = 1$ and $p = \infty$. The derived upper bounds consist of two terms: the first is an exponential function of the 2-R\'enyi mutual information between classical data and quantum embedding, while the second term scales linearly with the adversarial perturbation size $\epsilon$. Both terms are shown to decrease as $1/\sqrt{T}$ over the training set size $T$ . An extension is also considered in which the adversary assumed during training has different parameters $p$ and $\epsilon$ as compared to the adversary affecting the test inputs. Finally, we validate our theoretical findings with numerical experiments for a synthetic setting.

摘要: 与经典分类器类似，量子分类器很容易受到对手的攻击，扰乱它们的输入。一种有希望的对策是通过采用攻击感知或对抗性损失函数来训练量子分类器。研究了反向训练的量子分类器抵抗有界范数白盒攻击的泛化性质。具体地说，量子对手通过将输入状态$\rho(X)$转换为$\epsilon$-接近$p$-Schatten距离中的原始状态$\rho(X)$来最大化分类器的损失。在适当的量子嵌入假设下，我们得到了对抗性训练的量子分类器对$p=1$和$p=inty$的泛化误差的新的信息论上界。得到的上界由两项组成：第一项是经典数据和量子嵌入之间的2-R‘Enyi互信息的指数函数，第二项与对抗性扰动的大小成线性关系。这两项在训练集大小为$T$的情况下都减小了$1/\Sqrt{T}$。我们还考虑了一种扩展，其中假设的对手方在训练过程中与影响测试输入的对手方相比具有不同的参数$p$和$\epsilon$。最后，我们用数值实验验证了我们的理论结果。



## **8. Camouflage is all you need: Evaluating and Enhancing Language Model Robustness Against Camouflage Adversarial Attacks**

伪装就是您所需要的：评估和增强语言模型对伪装对手攻击的健壮性 cs.CL

19 pages, 8 figures, 5 tables

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09874v1) [paper-pdf](http://arxiv.org/pdf/2402.09874v1)

**Authors**: Álvaro Huertas-García, Alejandro Martín, Javier Huertas-Tato, David Camacho

**Abstract**: Adversarial attacks represent a substantial challenge in Natural Language Processing (NLP). This study undertakes a systematic exploration of this challenge in two distinct phases: vulnerability evaluation and resilience enhancement of Transformer-based models under adversarial attacks.   In the evaluation phase, we assess the susceptibility of three Transformer configurations, encoder-decoder, encoder-only, and decoder-only setups, to adversarial attacks of escalating complexity across datasets containing offensive language and misinformation. Encoder-only models manifest a 14% and 21% performance drop in offensive language detection and misinformation detection tasks, respectively. Decoder-only models register a 16% decrease in both tasks, while encoder-decoder models exhibit a maximum performance drop of 14% and 26% in the respective tasks.   The resilience-enhancement phase employs adversarial training, integrating pre-camouflaged and dynamically altered data. This approach effectively reduces the performance drop in encoder-only models to an average of 5% in offensive language detection and 2% in misinformation detection tasks. Decoder-only models, occasionally exceeding original performance, limit the performance drop to 7% and 2% in the respective tasks. Although not surpassing the original performance, Encoder-decoder models can reduce the drop to an average of 6% and 2% respectively.   Results suggest a trade-off between performance and robustness, with some models maintaining similar performance while gaining robustness. Our study and adversarial training techniques have been incorporated into an open-source tool for generating camouflaged datasets. However, methodology effectiveness depends on the specific camouflage technique and data encountered, emphasizing the need for continued exploration.

摘要: 对抗性攻击是自然语言处理(NLP)中的一个重大挑战。本研究分两个不同的阶段对这一挑战进行了系统的探讨：基于Transformer的模型在对抗攻击下的脆弱性评估和弹性增强。在评估阶段，我们评估了三种Transformer配置，编解码器、仅编码器和仅解码器设置对包含攻击性语言和错误信息的数据集的不断升级的复杂性的对抗性攻击的敏感度。只有编码者的模型在攻击性语言检测和错误信息检测任务中的性能分别下降了14%和21%。只有解码器的模型在两个任务中都表现出16%的下降，而编解码器模型在各自的任务中表现出14%和26%的最大性能下降。增强复原力阶段采用对抗性训练，整合预先伪装和动态改变的数据。这种方法有效地降低了仅编码者模型的性能下降，在攻击性语言检测任务中平均下降5%，在错误信息检测任务中平均下降2%。只有解码器的模型，有时会超过原始性能，将各自任务中的性能降幅限制在7%和2%。虽然没有超过原来的性能，但编解码器模型可以将平均降幅分别降低到6%和2%。结果表明，在性能和稳健性之间进行了权衡，一些模型在获得稳健性的同时保持了类似的性能。我们的研究和对抗性训练技术已经被整合到一个用于生成伪装数据集的开源工具中。然而，方法的有效性取决于具体的伪装技术和遇到的数据，强调需要继续探索。



## **9. Fooling the Image Dehazing Models by First Order Gradient**

利用一阶梯度愚弄图像去雾模型 cs.CV

This paper is accepted by IEEE Transactions on Circuits and Systems  for Video Technology (TCSVT)

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2303.17255v2) [paper-pdf](http://arxiv.org/pdf/2303.17255v2)

**Authors**: Jie Gui, Xiaofeng Cong, Chengwei Peng, Yuan Yan Tang, James Tin-Yau Kwok

**Abstract**: The research on the single image dehazing task has been widely explored. However, as far as we know, no comprehensive study has been conducted on the robustness of the well-trained dehazing models. Therefore, there is no evidence that the dehazing networks can resist malicious attacks. In this paper, we focus on designing a group of attack methods based on first order gradient to verify the robustness of the existing dehazing algorithms. By analyzing the general purpose of image dehazing task, four attack methods are proposed, which are predicted dehazed image attack, hazy layer mask attack, haze-free image attack and haze-preserved attack. The corresponding experiments are conducted on six datasets with different scales. Further, the defense strategy based on adversarial training is adopted for reducing the negative effects caused by malicious attacks. In summary, this paper defines a new challenging problem for the image dehazing area, which can be called as adversarial attack on dehazing networks (AADN). Code and Supplementary Material are available at https://github.com/Xiaofeng-life/AADN Dehazing.

摘要: 单幅图像去霾任务的研究已经得到了广泛的探索。然而，据我们所知，目前还没有对训练有素的去霾模型的稳健性进行全面的研究。因此，没有证据表明除霾网络能够抵御恶意攻击。在本文中，我们重点设计了一组基于一阶梯度的攻击方法，以验证现有去霾算法的健壮性。通过分析图像去霾任务的一般目的，提出了四种攻击方法，即预测去霾图像攻击、雾霾层遮罩攻击、无霾图像攻击和保霾攻击。在6个不同尺度的数据集上进行了相应的实验。此外，还采用了基于对抗性训练的防御策略，以减少恶意攻击带来的负面影响。综上所述，本文为图像去污领域定义了一个新的具有挑战性的问题，称为对抗性网络攻击。代码和补充材料可在https://github.com/Xiaofeng-life/AADN除霾网站上找到。



## **10. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09132v2) [paper-pdf](http://arxiv.org/pdf/2402.09132v2)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **11. Exploiting Alpha Transparency In Language And Vision-Based AI Systems**

在基于语言和视觉的人工智能系统中开发Alpha透明度 cs.CV

**SubmitDate**: 2024-02-15    [abs](http://arxiv.org/abs/2402.09671v1) [paper-pdf](http://arxiv.org/pdf/2402.09671v1)

**Authors**: David Noever, Forrest McKee

**Abstract**: This investigation reveals a novel exploit derived from PNG image file formats, specifically their alpha transparency layer, and its potential to fool multiple AI vision systems. Our method uses this alpha layer as a clandestine channel invisible to human observers but fully actionable by AI image processors. The scope tested for the vulnerability spans representative vision systems from Apple, Microsoft, Google, Salesforce, Nvidia, and Facebook, highlighting the attack's potential breadth. This vulnerability challenges the security protocols of existing and fielded vision systems, from medical imaging to autonomous driving technologies. Our experiments demonstrate that the affected systems, which rely on convolutional neural networks or the latest multimodal language models, cannot quickly mitigate these vulnerabilities through simple patches or updates. Instead, they require retraining and architectural changes, indicating a persistent hole in multimodal technologies without some future adversarial hardening against such vision-language exploits.

摘要: 这项调查揭示了一种源自PNG图像文件格式的新漏洞，特别是它们的alpha透明层，以及它欺骗多个AI视觉系统的潜力。我们的方法使用这个alpha层作为人类观察者不可见的秘密通道，但AI图像处理器完全可以操作。该漏洞的测试范围涵盖了来自Apple，Microsoft，Google，Salesforce，Nvidia和Facebook的代表性视觉系统，突出了攻击的潜在广度。该漏洞挑战了现有和现场视觉系统的安全协议，从医学成像到自动驾驶技术。我们的实验表明，依赖于卷积神经网络或最新多模态语言模型的受影响系统无法通过简单的补丁或更新快速缓解这些漏洞。相反，它们需要重新训练和架构更改，这表明多模式技术中存在一个持久的漏洞，而未来没有针对这种视觉语言漏洞的对抗性强化。



## **12. Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks**

打破它，模仿它，修复它：通过产生类似人类的攻击来增强健壮性 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.16955v2) [paper-pdf](http://arxiv.org/pdf/2310.16955v2)

**Authors**: Aradhana Sinha, Ananth Balashankar, Ahmad Beirami, Thi Avrahami, Jilin Chen, Alex Beutel

**Abstract**: Real-world natural language processing systems need to be robust to human adversaries. Collecting examples of human adversaries for training is an effective but expensive solution. On the other hand, training on synthetic attacks with small perturbations - such as word-substitution - does not actually improve robustness to human adversaries. In this paper, we propose an adversarial training framework that uses limited human adversarial examples to generate more useful adversarial examples at scale. We demonstrate the advantages of this system on the ANLI and hate speech detection benchmark datasets - both collected via an iterative, adversarial human-and-model-in-the-loop procedure. Compared to training only on observed human attacks, also training on our synthetic adversarial examples improves model robustness to future rounds. In ANLI, we see accuracy gains on the current set of attacks (44.1%$\,\to\,$50.1%) and on two future unseen rounds of human generated attacks (32.5%$\,\to\,$43.4%, and 29.4%$\,\to\,$40.2%). In hate speech detection, we see AUC gains on current attacks (0.76 $\to$ 0.84) and a future round (0.77 $\to$ 0.79). Attacks from methods that do not learn the distribution of existing human adversaries, meanwhile, degrade robustness.

摘要: 现实世界的自然语言处理系统需要对人类对手具有鲁棒性。收集人类对手的例子进行训练是一种有效但昂贵的解决方案。另一方面，对具有小扰动的合成攻击进行训练-例如单词替换-实际上并不能提高对人类对手的鲁棒性。在本文中，我们提出了一个对抗性训练框架，该框架使用有限的人类对抗性示例来大规模生成更有用的对抗性示例。我们在ANLI和仇恨语音检测基准数据集上展示了该系统的优势-这两个数据集都是通过迭代，对抗性的人类和模型在环过程收集的。与仅在观察到的人类攻击上进行训练相比，在我们的合成对抗性示例上进行训练也提高了模型对未来几轮的鲁棒性。在ANLI中，我们看到当前攻击集（44.1%$\，\to\，$50.1%）和未来两轮不可见的人类生成攻击（32.5%$\，\to\，$43.4%和29.4%$\，\to\，$40.2%）的准确率提高。在仇恨言论检测中，我们看到当前攻击（0.76 $\到0.84 $）和未来一轮（0.77 $\到0.79 $）的AUC收益。同时，来自不学习现有人类对手分布的方法的攻击会降低鲁棒性。



## **13. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

大型语言模型(LLM)在城市环境中的导航安全性如何？ cs.RO

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09546v1) [paper-pdf](http://arxiv.org/pdf/2402.09546v1)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently shown impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the technology's widespread application in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Suffix (NPS) Attack that manipulates LLM-based navigation models by appending gradient-derived suffixes to the original navigational prompt, leading to incorrect actions. We conducted comprehensive experiments on an LLMs-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across three metrics in the face of both white-box and black-box attacks. These results highlight the generalizability and transferability of the NPS Attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, concentrating on navigation-relevant keywords to reduce the impact of adversarial suffixes. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.

摘要: 在机器人学和自动化领域，基于大语言模型(LLMS)的导航系统最近表现出令人印象深刻的性能。然而，这些系统的安全方面受到的关注相对较少。本文率先探索了城市户外环境中基于LLM的导航模型的漏洞，鉴于该技术在自动驾驶、物流和紧急服务中的广泛应用，这是一个关键领域。具体地说，我们引入了一种新的导航提示后缀(NPS)攻击，该攻击通过在原始导航提示中添加梯度派生后缀来操纵基于LLM的导航模型，从而导致不正确的操作。我们在一个基于LLMS的导航模型上进行了全面的实验，该模型使用了不同的LLMS进行推理。我们的结果来自Touchdown和Map2Seq街景数据集，在少镜头学习和微调配置下，在面对白盒和黑盒攻击时，三个指标的性能都有显著下降。这些结果突出了NPS攻击的通用性和可转移性，强调了在基于LLM的导航系统中增强安全性的必要性。作为初步对策，我们提出了导航提示工程(NPE)防御策略，将重点放在与导航相关的关键字上，以减少对抗性后缀的影响。虽然初步研究结果表明，这一战略增强了导航安全，但更广泛的研究界仍然迫切需要开发更强大的防御方法，以有效地应对这些系统所面临的现实世界挑战。



## **14. NeuroBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroBitch：脑电波认证研究方法学标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08656v2) [paper-pdf](http://arxiv.org/pdf/2402.08656v2)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroB边来研究文献中提出的浅分类器和基于深度学习的方法，并在多个会话中测试健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法实践进一步推进基于脑电波的身份验证。



## **15. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

对抗性攻击下文本到图像生成中的非对称偏向 cs.LG

preprint version

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2312.14440v2) [paper-pdf](http://arxiv.org/pdf/2312.14440v2)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.

摘要: 文本到图像(T2I)模型在内容生成中的广泛使用需要仔细检查它们的安全性，包括它们对对手攻击的健壮性。尽管对对抗性攻击进行了广泛的研究，但其有效性的原因仍未得到充分探讨。本文对T2I模型的对抗性攻击进行了实证研究，重点分析了影响攻击成功率的因素。提出了一种新的攻击目标实体交换算法，利用对抗性后缀和两种基于梯度的攻击算法。人工评估和自动评估揭示了ASR在实体交换上的不对称性质：例如，在提示符“a Human in the雨中跳舞”中，更容易将“Human”替换为“bot”。使用对抗性后缀，但反向替换要困难得多。我们进一步提出了探测度量来建立从模型信念到对抗性ASR的指示性信号。我们确定了对抗性攻击成功概率为60%的条件，以及其他可能性降至5%以下的条件。



## **16. Only My Model On My Data: A Privacy Preserving Approach Protecting one Model and Deceiving Unauthorized Black-Box Models**

Only My Model on My Data：保护一个模型并欺骗未经授权的黑盒模型的隐私保护方法 cs.CV

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09316v1) [paper-pdf](http://arxiv.org/pdf/2402.09316v1)

**Authors**: Weiheng Chai, Brian Testa, Huantao Ren, Asif Salekin, Senem Velipasalar

**Abstract**: Deep neural networks are extensively applied to real-world tasks, such as face recognition and medical image classification, where privacy and data protection are critical. Image data, if not protected, can be exploited to infer personal or contextual information. Existing privacy preservation methods, like encryption, generate perturbed images that are unrecognizable to even humans. Adversarial attack approaches prohibit automated inference even for authorized stakeholders, limiting practical incentives for commercial and widespread adaptation. This pioneering study tackles an unexplored practical privacy preservation use case by generating human-perceivable images that maintain accurate inference by an authorized model while evading other unauthorized black-box models of similar or dissimilar objectives, and addresses the previous research gaps. The datasets employed are ImageNet, for image classification, Celeba-HQ dataset, for identity classification, and AffectNet, for emotion classification. Our results show that the generated images can successfully maintain the accuracy of a protected model and degrade the average accuracy of the unauthorized black-box models to 11.97%, 6.63%, and 55.51% on ImageNet, Celeba-HQ, and AffectNet datasets, respectively.

摘要: 深度神经网络被广泛应用于真实世界的任务，如人脸识别和医学图像分类，其中隐私和数据保护至关重要。如果不受保护，图像数据可以被利用来推断个人或上下文信息。现有的隐私保护方法，如加密，会产生甚至人类都无法识别的受干扰的图像。对抗性攻击方法甚至禁止对授权的利益攸关方进行自动推理，限制了商业和广泛适应的实际激励。这项开创性的研究解决了一个未探索的隐私保护实用用例，方法是生成人类可感知的图像，该图像保持授权模型的准确推断，同时避开其他目标相似或不相似的未经授权的黑盒模型，并填补了之前研究的空白。使用的数据集是ImageNet，用于图像分类，Celeba-HQ数据集，用于身份分类，以及AffectNet，用于情感分类。实验结果表明，生成的图像在ImageNet、Celeba-HQ和AffectNet数据集上能够成功地保持受保护模型的准确率，并将未经授权的黑盒模型的平均准确率分别降低到11.97%、6.63%和55.51%。



## **17. SoK: Pitfalls in Evaluating Black-Box Attacks**

SOK：评估黑盒攻击的陷阱 cs.CR

Accepted at SaTML 2024

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.17534v2) [paper-pdf](http://arxiv.org/pdf/2310.17534v2)

**Authors**: Fnu Suya, Anshuman Suri, Tingwei Zhang, Jingtao Hong, Yuan Tian, David Evans

**Abstract**: Numerous works study black-box attacks on image classifiers. However, these works make different assumptions on the adversary's knowledge and current literature lacks a cohesive organization centered around the threat model. To systematize knowledge in this area, we propose a taxonomy over the threat space spanning the axes of feedback granularity, the access of interactive queries, and the quality and quantity of the auxiliary data available to the attacker. Our new taxonomy provides three key insights. 1) Despite extensive literature, numerous under-explored threat spaces exist, which cannot be trivially solved by adapting techniques from well-explored settings. We demonstrate this by establishing a new state-of-the-art in the less-studied setting of access to top-k confidence scores by adapting techniques from well-explored settings of accessing the complete confidence vector, but show how it still falls short of the more restrictive setting that only obtains the prediction label, highlighting the need for more research. 2) Identification the threat model of different attacks uncovers stronger baselines that challenge prior state-of-the-art claims. We demonstrate this by enhancing an initially weaker baseline (under interactive query access) via surrogate models, effectively overturning claims in the respective paper. 3) Our taxonomy reveals interactions between attacker knowledge that connect well to related areas, such as model inversion and extraction attacks. We discuss how advances in other areas can enable potentially stronger black-box attacks. Finally, we emphasize the need for a more realistic assessment of attack success by factoring in local attack runtime. This approach reveals the potential for certain attacks to achieve notably higher success rates and the need to evaluate attacks in diverse and harder settings, highlighting the need for better selection criteria.

摘要: 许多著作研究了对图像分类器的黑盒攻击。然而，这些著作对对手的知识做出了不同的假设，目前的文献缺乏一个以威胁模型为中心的有凝聚力的组织。为了系统化这一领域的知识，我们提出了一种横跨反馈粒度、交互查询的访问以及攻击者可用的辅助数据的质量和数量轴的威胁空间分类。我们的新分类法提供了三个关键的见解。1)尽管有大量的文献，但仍然存在大量未被充分探索的威胁空间，这些威胁空间不能通过从经过充分探索的环境中采用技术来平凡地解决。我们通过在较少研究的访问Top-k置信度分数的设置中建立新的最先进的设置来证明这一点，方法是采用来自充分探索的访问完整置信度向量的设置的技术，但展示了它如何仍然没有达到仅获得预测标签的更具限制性的设置，从而突出了更多研究的必要性。2)识别不同攻击的威胁模型揭示了挑战先前最先进主张的更强大的基线。我们通过代理模型增强了最初较弱的基线(在交互式查询访问下)，有效地推翻了各自论文中的主张，从而证明了这一点。3)我们的分类揭示了攻击者知识之间的交互作用，这些知识与相关领域联系良好，如模型倒置和提取攻击。我们讨论了其他领域的进步如何使潜在更强大的黑盒攻击成为可能。最后，我们强调需要通过考虑本地攻击运行时来更现实地评估攻击成功。这种方法揭示了某些攻击实现显著更高成功率的潜力，以及在不同和更困难的环境中评估攻击的必要性，突出了需要更好的选择标准。



## **18. Evading Black-box Classifiers Without Breaking Eggs**

不破鸡蛋躲避黑盒分类器 cs.CR

Code at https://github.com/ethz-privsec/realistic-adv-examples.  Accepted at IEEE SaTML 2024

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2306.02895v2) [paper-pdf](http://arxiv.org/pdf/2306.02895v2)

**Authors**: Edoardo Debenedetti, Nicholas Carlini, Florian Tramèr

**Abstract**: Decision-based evasion attacks repeatedly query a black-box classifier to generate adversarial examples. Prior work measures the cost of such attacks by the total number of queries made to the classifier. We argue this metric is flawed. Most security-critical machine learning systems aim to weed out "bad" data (e.g., malware, harmful content, etc). Queries to such systems carry a fundamentally asymmetric cost: queries detected as "bad" come at a higher cost because they trigger additional security filters, e.g., usage throttling or account suspension. Yet, we find that existing decision-based attacks issue a large number of "bad" queries, which likely renders them ineffective against security-critical systems. We then design new attacks that reduce the number of bad queries by $1.5$-$7.3\times$, but often at a significant increase in total (non-bad) queries. We thus pose it as an open problem to build black-box attacks that are more effective under realistic cost metrics.

摘要: 基于决策的规避攻击反复查询黑盒分类器以生成对抗性示例。以前的工作通过对分类器进行的查询总数来衡量此类攻击的成本。我们认为这个指标是有缺陷的。大多数安全关键型机器学习系统的目标是剔除“坏”数据(例如，恶意软件、有害内容等)。对这类系统的查询会带来根本不对称的成本：被检测为“坏”的查询成本更高，因为它们会触发额外的安全过滤器，例如，使用节流或帐户暂停。然而，我们发现现有的基于决策的攻击发出了大量的“坏”查询，这可能会使它们对安全关键系统无效。然后，我们设计新的攻击，将错误查询的数量减少$1.5$-$7.3倍$，但通常会显著增加总(非错误)查询的数量。因此，我们认为构建在现实成本指标下更有效的黑盒攻击是一个悬而未决的问题。



## **19. Attacking Large Language Models with Projected Gradient Descent**

用投影梯度下降攻击大型语言模型 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09154v1) [paper-pdf](http://arxiv.org/pdf/2402.09154v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Johannes Gasteiger, Stephan Günnemann

**Abstract**: Current LLM alignment methods are readily broken through specifically crafted adversarial prompts. While crafting adversarial prompts using discrete optimization is highly effective, such attacks typically use more than 100,000 LLM calls. This high computational cost makes them unsuitable for, e.g., quantitative analyses and adversarial training. To remedy this, we revisit Projected Gradient Descent (PGD) on the continuously relaxed input prompt. Although previous attempts with ordinary gradient-based attacks largely failed, we show that carefully controlling the error introduced by the continuous relaxation tremendously boosts their efficacy. Our PGD for LLMs is up to one order of magnitude faster than state-of-the-art discrete optimization to achieve the same devastating attack results.

摘要: 目前的LLM对齐方法很容易突破专门制作的对抗性提示。虽然使用离散优化精心编制敌意提示是非常有效的，但此类攻击通常使用超过100,000个LLM调用。这种高的计算成本使得它们不适合例如定量分析和对抗性训练。为了纠正这个问题，我们重新审视了持续放松的输入提示上的预测梯度下降(PGD)。虽然以前使用普通的基于梯度的攻击的尝试基本上都失败了，但我们表明，仔细控制连续放松带来的错误可以极大地提高它们的效率。我们针对LLMS的PGD比最先进的离散优化快一个数量级，以实现相同的毁灭性攻击结果。



## **20. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

软提示威胁：通过嵌入空间攻击开源LLMS中的安全对齐和遗忘 cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09063v1) [paper-pdf](http://arxiv.org/pdf/2402.09063v1)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.

摘要: 目前LLM对抗鲁棒性的研究主要集中在自然语言空间中的离散输入操作，这些操作可以直接转移到闭源模型。然而，这种方法忽略了开源模型的稳步发展。随着开源模型在功能上的进步，确保其安全性也变得越来越重要。然而，针对利用完整模型访问的开源LLM的攻击在很大程度上仍未被探索。我们解决了这一研究空白，并提出了嵌入空间攻击，它直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击规避模型对齐和触发有害行为比离散攻击或模型微调更有效。此外，我们提出了一种新的威胁模型，在unlearning的背景下，并表明嵌入空间攻击可以从多个数据集和模型的unlearned LLM中提取被删除的信息。我们的研究结果强调了嵌入空间攻击作为开源LLM中的一个重要威胁模型。触发警告：附录包含LLM生成的暴力和骚扰文本。



## **21. Review-Incorporated Model-Agnostic Profile Injection Attacks on Recommender Systems**

综述-针对推荐系统的合并模型不可知配置文件注入攻击 cs.CR

Accepted by ICDM 2023

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.09023v1) [paper-pdf](http://arxiv.org/pdf/2402.09023v1)

**Authors**: Shiyi Yang, Lina Yao, Chen Wang, Xiwei Xu, Liming Zhu

**Abstract**: Recent studies have shown that recommender systems (RSs) are highly vulnerable to data poisoning attacks. Understanding attack tactics helps improve the robustness of RSs. We intend to develop efficient attack methods that use limited resources to generate high-quality fake user profiles to achieve 1) transferability among black-box RSs 2) and imperceptibility among detectors. In order to achieve these goals, we introduce textual reviews of products to enhance the generation quality of the profiles. Specifically, we propose a novel attack framework named R-Trojan, which formulates the attack objectives as an optimization problem and adopts a tailored transformer-based generative adversarial network (GAN) to solve it so that high-quality attack profiles can be produced. Comprehensive experiments on real-world datasets demonstrate that R-Trojan greatly outperforms state-of-the-art attack methods on various victim RSs under black-box settings and show its good imperceptibility.

摘要: 最近的研究表明，推荐系统(RSS)非常容易受到数据中毒攻击。了解攻击策略有助于提高RSS的健壮性。我们打算开发高效的攻击方法，利用有限的资源来生成高质量的虚假用户配置文件，以实现1)黑盒RSS之间的可传递性和检测器之间的不可见性。为了实现这些目标，我们引入了产品的文本审查，以提高配置文件的生成质量。具体地说，我们提出了一种新的攻击框架R-特洛伊木马，该框架将攻击目标描述为一个优化问题，并采用定制的基于变压器的生成性对抗网络(GAN)进行求解，从而生成高质量的攻击轮廓。在真实数据集上的综合实验表明，R-特洛伊木马在黑盒环境下对各种受害者RS的攻击性能大大优于现有的攻击方法，并显示出良好的不可感知性。



## **22. Prompted Contextual Vectors for Spear-Phishing Detection**

鱼叉式网络钓鱼检测的提示上下文向量 cs.LG

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08309v2) [paper-pdf](http://arxiv.org/pdf/2402.08309v2)

**Authors**: Daniel Nahmias, Gal Engelberg, Dan Klein, Asaf Shabtai

**Abstract**: Spear-phishing attacks present a significant security challenge, with large language models (LLMs) escalating the threat by generating convincing emails and facilitating target reconnaissance. To address this, we propose a detection approach based on a novel document vectorization method that utilizes an ensemble of LLMs to create representation vectors. By prompting LLMs to reason and respond to human-crafted questions, we quantify the presence of common persuasion principles in the email's content, producing prompted contextual document vectors for a downstream supervised machine learning model. We evaluate our method using a unique dataset generated by a proprietary system that automates target reconnaissance and spear-phishing email creation. Our method achieves a 91% F1 score in identifying LLM-generated spear-phishing emails, with the training set comprising only traditional phishing and benign emails. Key contributions include an innovative document vectorization method utilizing LLM reasoning, a publicly available dataset of high-quality spear-phishing emails, and the demonstrated effectiveness of our method in detecting such emails. This methodology can be utilized for various document classification tasks, particularly in adversarial problem domains.

摘要: 鱼叉式网络钓鱼攻击是一个重大的安全挑战，大型语言模型（LLM）通过生成令人信服的电子邮件和促进目标侦察来升级威胁。为了解决这个问题，我们提出了一种检测方法的基础上，一种新的文档矢量化方法，利用一个合奏的LLM创建表示向量。通过促使LLM推理并响应人工问题，我们量化了电子邮件内容中常见说服原则的存在，为下游监督机器学习模型生成提示的上下文文档向量。我们使用专有系统生成的独特数据集来评估我们的方法，该系统可自动进行目标侦察和鱼叉式网络钓鱼电子邮件创建。我们的方法在识别LLM生成的鱼叉式网络钓鱼电子邮件方面达到了91%的F1分数，训练集仅包括传统的网络钓鱼和良性电子邮件。主要贡献包括利用LLM推理的创新文档矢量化方法，高质量鱼叉式网络钓鱼电子邮件的公开数据集，以及我们的方法在检测此类电子邮件方面的有效性。该方法可用于各种文档分类任务，特别是在对抗性问题领域。



## **23. Robustness of AI-Image Detectors: Fundamental Limits and Practical Attacks**

人工智能图像检测器的稳健性：基本限制和实用攻击 cs.CV

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2310.00076v2) [paper-pdf](http://arxiv.org/pdf/2310.00076v2)

**Authors**: Mehrdad Saberi, Vinu Sankar Sadasivan, Keivan Rezaei, Aounon Kumar, Atoosa Chegini, Wenxiao Wang, Soheil Feizi

**Abstract**: In light of recent advancements in generative AI models, it has become essential to distinguish genuine content from AI-generated one to prevent the malicious usage of fake materials as authentic ones and vice versa. Various techniques have been introduced for identifying AI-generated images, with watermarking emerging as a promising approach. In this paper, we analyze the robustness of various AI-image detectors including watermarking and classifier-based deepfake detectors. For watermarking methods that introduce subtle image perturbations (i.e., low perturbation budget methods), we reveal a fundamental trade-off between the evasion error rate (i.e., the fraction of watermarked images detected as non-watermarked ones) and the spoofing error rate (i.e., the fraction of non-watermarked images detected as watermarked ones) upon an application of diffusion purification attack. To validate our theoretical findings, we also provide empirical evidence demonstrating that diffusion purification effectively removes low perturbation budget watermarks by applying minimal changes to images. The diffusion purification attack is ineffective for high perturbation watermarking methods where notable changes are applied to images. In this case, we develop a model substitution adversarial attack that can successfully remove watermarks. Moreover, we show that watermarking methods are vulnerable to spoofing attacks where the attacker aims to have real images identified as watermarked ones, damaging the reputation of the developers. In particular, with black-box access to the watermarking method, a watermarked noise image can be generated and added to real images, causing them to be incorrectly classified as watermarked. Finally, we extend our theory to characterize a fundamental trade-off between the robustness and reliability of classifier-based deep fake detectors and demonstrate it through experiments.

摘要: 鉴于生成性人工智能模型最近的进展，区分真正的内容和人工智能生成的内容变得至关重要，以防止恶意使用假冒材料作为真品，反之亦然。人们已经引入了各种技术来识别人工智能生成的图像，其中水印是一种很有前途的方法。在本文中，我们分析了各种人工智能图像检测器的鲁棒性，包括水印和基于分类器的深度伪检测器。对于引入微妙图像扰动的水印方法(即，低扰动预算方法)，我们揭示了在应用扩散净化攻击时，规避错误率(即，检测为非水印图像的分数)和欺骗错误率(即，检测为水印图像的非水印图像的分数)之间的基本权衡。为了验证我们的理论发现，我们还提供了实验证据，证明扩散净化通过对图像应用最小的改变来有效地去除低扰动预算的水印。扩散净化攻击对于对图像应用显著变化的高扰动水印方法是无效的。在这种情况下，我们开发了一种模型替换对抗性攻击，可以成功地去除水印。此外，我们发现水印方法容易受到欺骗攻击，攻击者的目标是将真实图像识别为带水印的图像，从而损害开发人员的声誉。具体地说，通过黑盒访问水印方法，可以生成加水印的噪声图像并将其添加到真实图像，从而导致它们被错误地分类为加水印。最后，我们将我们的理论扩展到基于分类器的深度伪检测器的稳健性和可靠性之间的一个基本权衡，并通过实验进行了验证。



## **24. Detecting Adversarial Spectrum Attacks via Distance to Decision Boundary Statistics**

基于距离决策边界统计量的敌意频谱攻击检测 cs.CR

10 pages, 11 figures

**SubmitDate**: 2024-02-14    [abs](http://arxiv.org/abs/2402.08986v1) [paper-pdf](http://arxiv.org/pdf/2402.08986v1)

**Authors**: Wenwei Zhao, Xiaowen Li, Shangqing Zhao, Jie Xu, Yao Liu, Zhuo Lu

**Abstract**: Machine learning has been adopted for efficient cooperative spectrum sensing. However, it incurs an additional security risk due to attacks leveraging adversarial machine learning to create malicious spectrum sensing values to deceive the fusion center, called adversarial spectrum attacks. In this paper, we propose an efficient framework for detecting adversarial spectrum attacks. Our design leverages the concept of the distance to the decision boundary (DDB) observed at the fusion center and compares the training and testing DDB distributions to identify adversarial spectrum attacks. We create a computationally efficient way to compute the DDB for machine learning based spectrum sensing systems. Experimental results based on realistic spectrum data show that our method, under typical settings, achieves a high detection rate of up to 99\% and maintains a low false alarm rate of less than 1\%. In addition, our method to compute the DDB based on spectrum data achieves 54\%--64\% improvements in computational efficiency over existing distance calculation methods. The proposed DDB-based detection framework offers a practical and efficient solution for identifying malicious sensing values created by adversarial spectrum attacks.

摘要: 机器学习被用于高效的协作频谱感知。然而，它会招致额外的安全风险，因为攻击利用对抗性机器学习来创建恶意频谱感知值来欺骗融合中心，称为对抗性频谱攻击。在本文中，我们提出了一个有效的框架来检测对抗性频谱攻击。我们的设计利用了在融合中心观察到的到决策边界(DDB)的距离的概念，并比较了训练和测试DDB分布以识别对抗性频谱攻击。我们为基于机器学习的频谱感知系统创建了一种计算分布式数据库的高效方法。基于实际频谱数据的实验结果表明，在典型设置下，该方法的检测率高达99%，虚警率低于1%。此外，基于频谱数据计算分布式数据库的方法与现有的距离计算方法相比，计算效率提高了54-64。提出的基于分布式数据库的检测框架为识别敌意频谱攻击产生的恶意感知值提供了一种实用而有效的解决方案。



## **25. Adversarially Robust Feature Learning for Breast Cancer Diagnosis**

逆稳健特征学习在乳腺癌诊断中的应用 eess.IV

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08768v1) [paper-pdf](http://arxiv.org/pdf/2402.08768v1)

**Authors**: Degan Hao, Dooman Arefan, Margarita Zuley, Wendie Berg, Shandong Wu

**Abstract**: Adversarial data can lead to malfunction of deep learning applications. It is essential to develop deep learning models that are robust to adversarial data while accurate on standard, clean data. In this study, we proposed a novel adversarially robust feature learning (ARFL) method for a real-world application of breast cancer diagnosis. ARFL facilitates adversarial training using both standard data and adversarial data, where a feature correlation measure is incorporated as an objective function to encourage learning of robust features and restrain spurious features. To show the effects of ARFL in breast cancer diagnosis, we built and evaluated diagnosis models using two independent clinically collected breast imaging datasets, comprising a total of 9,548 mammogram images. We performed extensive experiments showing that our method outperformed several state-of-the-art methods and that our method can enhance safer breast cancer diagnosis against adversarial attacks in clinical settings.

摘要: 敌意数据可能会导致深度学习应用程序出现故障。开发深度学习模型是至关重要的，这种模型对对抗性数据是健壮的，而对标准、干净的数据是准确的。在这项研究中，我们提出了一种新的对抗性稳健特征学习(ARFL)方法，用于乳腺癌诊断的实际应用。ARFL使用标准数据和对抗性数据来促进对抗性训练，其中特征相关性度量被合并为目标函数，以鼓励稳健特征的学习并抑制虚假特征。为了展示ARFL在乳腺癌诊断中的作用，我们使用两个独立的临床收集的乳房成像数据集，包括总共9,548张乳房X光图像，建立并评估了诊断模型。我们进行了广泛的实验，表明我们的方法优于几种最先进的方法，并且我们的方法可以提高乳腺癌诊断的安全性，使其在临床环境中免受对抗性攻击。



## **26. Enhancing Robustness of Indoor Robotic Navigation with Free-Space Segmentation Models Against Adversarial Attacks**

利用自由空间分割模型增强室内机器人导航对敌方攻击的鲁棒性 cs.CV

Accepted to 2023 IEEE International Conference on Robotic Computing  (IRC). arXiv admin note: text overlap with arXiv:2311.01966

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08763v1) [paper-pdf](http://arxiv.org/pdf/2402.08763v1)

**Authors**: Qiyuan An, Christos Sevastopoulos, Fillia Makedon

**Abstract**: Endeavors in indoor robotic navigation rely on the accuracy of segmentation models to identify free space in RGB images. However, deep learning models are vulnerable to adversarial attacks, posing a significant challenge to their real-world deployment. In this study, we identify vulnerabilities within the hidden layers of neural networks and introduce a practical approach to reinforce traditional adversarial training. Our method incorporates a novel distance loss function, minimizing the gap between hidden layers in clean and adversarial images. Experiments demonstrate satisfactory performance in improving the model's robustness against adversarial perturbations.

摘要: 室内机器人导航的努力依赖于分割模型的准确性来识别RGB图像中的自由空间。然而，深度学习模型很容易受到对抗性攻击，这对其在现实世界中的部署构成了重大挑战。在这项研究中，我们识别了神经网络隐藏层中的漏洞，并引入了一种实用的方法来加强传统的对抗训练。我们的方法结合了一种新的距离损失函数，最大限度地减少了干净和对抗图像中隐藏层之间的差距。实验结果表明，该模型在提高对抗性扰动的鲁棒性方面具有令人满意的性能。



## **27. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐身性和可控性的LLM越狱 cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08679v1) [paper-pdf](http://arxiv.org/pdf/2402.08679v1)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on Large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent suffix attacks, but also allow us to address new controllable attack settings such as revising a user query adversarially with minimal paraphrasing, and inserting stealthy attacks in context with left-right-coherence. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击带来的可控性导致了不同的新越狱场景，这些场景不仅覆盖了生成流畅后缀攻击的标准设置，而且允许我们解决新的可控攻击设置，例如以最小的转述以相反的方式修改用户查询，以及以左右一致的方式在上下文中插入隐蔽攻击。我们在不同的LLMS(骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5)上的广泛实验表明，冷攻击具有广泛的适用性、较强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **28. SAGMAN: Stability Analysis of Graph Neural Networks on the Manifolds**

Sagman：流形上图神经网络的稳定性分析 cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08653v1) [paper-pdf](http://arxiv.org/pdf/2402.08653v1)

**Authors**: Wuxinlin Cheng, Chenhui Deng, Ali Aghdaei, Zhiru Zhang, Zhuo Feng

**Abstract**: Modern graph neural networks (GNNs) can be sensitive to changes in the input graph structure and node features, potentially resulting in unpredictable behavior and degraded performance. In this work, we introduce a spectral framework known as SAGMAN for examining the stability of GNNs. This framework assesses the distance distortions that arise from the nonlinear mappings of GNNs between the input and output manifolds: when two nearby nodes on the input manifold are mapped (through a GNN model) to two distant ones on the output manifold, it implies a large distance distortion and thus a poor GNN stability. We propose a distance-preserving graph dimension reduction (GDR) approach that utilizes spectral graph embedding and probabilistic graphical models (PGMs) to create low-dimensional input/output graph-based manifolds for meaningful stability analysis. Our empirical evaluations show that SAGMAN effectively assesses the stability of each node when subjected to various edge or feature perturbations, offering a scalable approach for evaluating the stability of GNNs, extending to applications within recommendation systems. Furthermore, we illustrate its utility in downstream tasks, notably in enhancing GNN stability and facilitating adversarial targeted attacks.

摘要: 现代图神经网络(GNN)对输入图结构和节点特征的变化很敏感，可能导致不可预测的行为和性能下降。在这项工作中，我们引入了一个称为Sagman的光谱框架来检查GNN的稳定性。该框架评估了输入和输出流形之间GNN之间的非线性映射引起的距离失真：当输入流形上的两个邻近节点(通过GNN模型)映射到输出流形上的两个相距较远的节点时，这意味着较大的距离失真，因此GNN稳定性较差。我们提出了一种保持距离的图降维方法(GDR)，该方法利用谱图嵌入和概率图模型(PGMS)来创建基于低维输入/输出图的流形，用于有意义的稳定性分析。实验结果表明，Sagman算法能够有效地评估每个节点在受到各种边缘或特征扰动时的稳定性，为评估GNN的稳定性提供了一种可扩展的方法，并扩展到推荐系统中的应用。此外，我们还说明了它在下游任务中的效用，特别是在增强GNN稳定性和促进对抗性定向攻击方面。



## **29. Generating Universal Adversarial Perturbations for Quantum Classifiers**

为量子分类器生成通用对抗性扰动 cs.LG

Accepted at AAAI 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08648v1) [paper-pdf](http://arxiv.org/pdf/2402.08648v1)

**Authors**: Gautham Anil, Vishnu Vinod, Apurva Narayan

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising field of research, aiming to leverage the capabilities of quantum computing to enhance existing machine learning methodologies. Recent studies have revealed that, like their classical counterparts, QML models based on Parametrized Quantum Circuits (PQCs) are also vulnerable to adversarial attacks. Moreover, the existence of Universal Adversarial Perturbations (UAPs) in the quantum domain has been demonstrated theoretically in the context of quantum classifiers. In this work, we introduce QuGAP: a novel framework for generating UAPs for quantum classifiers. We conceptualize the notion of additive UAPs for PQC-based classifiers and theoretically demonstrate their existence. We then utilize generative models (QuGAP-A) to craft additive UAPs and experimentally show that quantum classifiers are susceptible to such attacks. Moreover, we formulate a new method for generating unitary UAPs (QuGAP-U) using quantum generative models and a novel loss function based on fidelity constraints. We evaluate the performance of the proposed framework and show that our method achieves state-of-the-art misclassification rates, while maintaining high fidelity between legitimate and adversarial samples.

摘要: 量子机器学习(QML)已经成为一个很有前途的研究领域，旨在利用量子计算的能力来增强现有的机器学习方法。最近的研究表明，与经典模型一样，基于参数化量子电路(PQCs)的QML模型也容易受到敌意攻击。此外，在量子分类器的背景下，从理论上证明了量子域中普遍对抗扰动(UAP)的存在。在这项工作中，我们介绍了QuGAP：一个为量子分类器生成UAP的新框架。我们对基于PQC的分类器提出了加性UAP的概念，并从理论上证明了加性UAP的存在性。然后，我们利用生成模型(QuGAP-A)来制作可添加的UAP，并通过实验表明量子分类器容易受到此类攻击。此外，我们利用量子生成模型和基于保真度约束的损失函数，提出了一种新的生成酉UAP的方法(QuGAP-U)。我们对所提出的框架的性能进行了评估，结果表明，我们的方法在保持合法样本和敌对样本之间的高保真度的同时，获得了最先进的错误分类率。



## **30. SWAP: Sparse Entropic Wasserstein Regression for Robust Network Pruning**

SWAP：用于稳健网络剪枝的稀疏熵Wasserstein回归 cs.AI

Published as a conference paper at ICLR 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2310.04918v3) [paper-pdf](http://arxiv.org/pdf/2310.04918v3)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study tackles the issue of neural network pruning that inaccurate gradients exist when computing the empirical Fisher Information Matrix (FIM). We introduce SWAP, an Entropic Wasserstein regression (EWR) network pruning formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. The "swap" of a commonly used standard linear regression (LR) with the EWR in optimization is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points, yet incurs marginal extra computational cost. The unique strength of SWAP is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of SWAP with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.

摘要: 该研究解决了在计算经验Fisher信息矩阵(FIM)时存在梯度不准确的神经网络修剪问题。利用最优传输(OT)问题的几何属性，我们引入了SWAP，这是一种熵Wasserstein回归(EWR)网络剪枝公式。分析表明，常用的标准线性回归(LR)与EWR在优化中的“互换”，通过采用跨数据点的邻域内插在噪声抑制方面表现得更好，但会产生边际额外的计算成本。SWAP的独特优势在于其在降噪和保留协方差信息之间取得平衡的内在能力。在不同网络上进行的大量实验表明，SWAP的性能与最新的网络修剪算法(SOTA)相当。当网络规模或目标稀疏度较大时，我们提出的方法的性能优于SOTA，当存在噪声梯度时，增益甚至更大，可能来自噪声数据、模拟记忆或敌对攻击。值得注意的是，我们提出的方法在剩余不到四分之一的网络参数的情况下，对MobileNetV1实现了6%的准确率提高和8%的测试损失改善。



## **31. System-level Analysis of Adversarial Attacks and Defenses on Intelligence in O-RAN based Cellular Networks**

基于O-RAN的蜂窝网络智能对抗攻防的系统级分析 cs.CR

his paper has been accepted for publication in ACM WiSec 2024

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.06846v2) [paper-pdf](http://arxiv.org/pdf/2402.06846v2)

**Authors**: Azuka Chiejina, Brian Kim, Kaushik Chowhdury, Vijay K. Shah

**Abstract**: While the open architecture, open interfaces, and integration of intelligence within Open Radio Access Network technology hold the promise of transforming 5G and 6G networks, they also introduce cybersecurity vulnerabilities that hinder its widespread adoption. In this paper, we conduct a thorough system-level investigation of cyber threats, with a specific focus on machine learning (ML) intelligence components known as xApps within the O-RAN's near-real-time RAN Intelligent Controller (near-RT RIC) platform. Our study begins by developing a malicious xApp designed to execute adversarial attacks on two types of test data - spectrograms and key performance metrics (KPMs), stored in the RIC database within the near-RT RIC. To mitigate these threats, we utilize a distillation technique that involves training a teacher model at a high softmax temperature and transferring its knowledge to a student model trained at a lower softmax temperature, which is deployed as the robust ML model within xApp. We prototype an over-the-air LTE/5G O-RAN testbed to assess the impact of these attacks and the effectiveness of the distillation defense technique by leveraging an ML-based Interference Classification (InterClass) xApp as an example. We examine two versions of InterClass xApp under distinct scenarios, one based on Convolutional Neural Networks (CNNs) and another based on Deep Neural Networks (DNNs) using spectrograms and KPMs as input data respectively. Our findings reveal up to 100% and 96.3% degradation in the accuracy of both the CNN and DNN models respectively resulting in a significant decline in network performance under considered adversarial attacks. Under the strict latency constraints of the near-RT RIC closed control loop, our analysis shows that the distillation technique outperforms classical adversarial training by achieving an accuracy of up to 98.3% for mitigating such attacks.

摘要: 虽然开放式无线接入网络技术中的开放架构、开放接口和智能集成有望实现5G和6G网络的转型，但它们也引入了网络安全漏洞，阻碍了其广泛采用。在本文中，我们对网络威胁进行了彻底的系统级调查，重点是O-RAN的近实时RAN智能控制器(Near-RTRIC)平台中称为xApp的机器学习(ML)智能组件。我们的研究首先开发了一个恶意xApp，旨在对两种类型的测试数据-频谱图和关键性能指标(KPM)-执行对抗性攻击，这些数据存储在近RT RIC的RIC数据库中。为了缓解这些威胁，我们利用了一种蒸馏技术，包括在高Softmax温度下培训教师模型，并将其知识传输到在较低Softmax温度下培训的学生模型，该学生模型在xApp中部署为健壮的ML模型。我们以一个基于ML的干扰分类(类间)xApp为例，构建了一个空中LTE/5G O-RAN试验台，以评估这些攻击的影响和蒸馏防御技术的有效性。我们在不同的场景下考察了两个版本的类间xApp，一个基于卷积神经网络(CNN)，另一个基于深度神经网络(DNN)，分别使用谱图和KPM作为输入数据。我们的研究结果显示，在被认为是对抗性攻击的情况下，CNN和DNN模型的准确率分别下降了100%和96.3%，导致网络性能显著下降。分析表明，在近实时RIC闭合控制环的严格延迟约束下，蒸馏技术的性能优于经典的对抗性训练，在缓解此类攻击方面的准确率高达98.3%。



## **32. Test-Time Backdoor Attacks on Multimodal Large Language Models**

对多通道大型语言模型的测试时间后门攻击 cs.CL

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.08577v1) [paper-pdf](http://arxiv.org/pdf/2402.08577v1)

**Authors**: Dong Lu, Tianyu Pang, Chao Du, Qian Liu, Xianjun Yang, Min Lin

**Abstract**: Backdoor attacks are commonly executed by contaminating training data, such that a trigger can activate predetermined harmful effects during the test phase. In this work, we present AnyDoor, a test-time backdoor attack against multimodal large language models (MLLMs), which involves injecting the backdoor into the textual modality using adversarial test images (sharing the same universal perturbation), without requiring access to or modification of the training data. AnyDoor employs similar techniques used in universal adversarial attacks, but distinguishes itself by its ability to decouple the timing of setup and activation of harmful effects. In our experiments, we validate the effectiveness of AnyDoor against popular MLLMs such as LLaVA-1.5, MiniGPT-4, InstructBLIP, and BLIP-2, as well as provide comprehensive ablation studies. Notably, because the backdoor is injected by a universal perturbation, AnyDoor can dynamically change its backdoor trigger prompts/harmful effects, exposing a new challenge for defending against backdoor attacks. Our project page is available at https://sail-sg.github.io/AnyDoor/.

摘要: 后门攻击通常通过污染训练数据来执行，使得触发器可以在测试阶段激活预定的有害影响。在这项工作中，我们提出了AnyDoor，一种针对多模式大型语言模型(MLLMS)的测试时间后门攻击，它使用敌对的测试图像(共享相同的通用扰动)将后门注入到文本通道中，而不需要访问或修改训练数据。AnyDoor使用了通用对抗性攻击中使用的类似技术，但其与众不同之处在于它能够将设置和激活有害效果的时间脱钩。在我们的实验中，我们验证了AnyDoor对LLaVA-1.5、MiniGPT-4、InstructBLIP和BLIP-2等常用MLLMS的有效性，并提供了全面的消融研究。值得注意的是，由于后门是由通用扰动注入的，AnyDoor可以动态更改其后门触发提示/有害影响，从而暴露出防御后门攻击的新挑战。我们的项目页面可在https://sail-sg.github.io/AnyDoor/.上查看



## **33. Adversarial attacks and defenses in explainable artificial intelligence: A survey**

可解释人工智能中的对抗性攻击和防御：综述 cs.CR

Accepted by Information Fusion

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2306.06123v3) [paper-pdf](http://arxiv.org/pdf/2306.06123v3)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: Explainable artificial intelligence (XAI) methods are portrayed as a remedy for debugging and trusting statistical and deep learning models, as well as interpreting their predictions. However, recent advances in adversarial machine learning (AdvML) highlight the limitations and vulnerabilities of state-of-the-art explanation methods, putting their security and trustworthiness into question. The possibility of manipulating, fooling or fairwashing evidence of the model's reasoning has detrimental consequences when applied in high-stakes decision-making and knowledge discovery. This survey provides a comprehensive overview of research concerning adversarial attacks on explanations of machine learning models, as well as fairness metrics. We introduce a unified notation and taxonomy of methods facilitating a common ground for researchers and practitioners from the intersecting research fields of AdvML and XAI. We discuss how to defend against attacks and design robust interpretation methods. We contribute a list of existing insecurities in XAI and outline the emerging research directions in adversarial XAI (AdvXAI). Future work should address improving explanation methods and evaluation protocols to take into account the reported safety issues.

摘要: 可解释人工智能（XAI）方法被描述为调试和信任统计和深度学习模型以及解释其预测的补救措施。然而，对抗性机器学习（AdvML）的最新进展突出了最先进解释方法的局限性和脆弱性，使其安全性和可信度受到质疑。操纵、愚弄或粉饰模型推理证据的可能性在高风险决策和知识发现中具有有害后果。这项调查全面概述了关于机器学习模型解释的对抗性攻击以及公平性指标的研究。我们介绍了一个统一的符号和分类的方法，促进共同的基础，研究人员和从业人员从AdvML和XAI的交叉研究领域。我们讨论如何抵御攻击和设计强大的解释方法。我们提供了XAI中现有的不安全因素列表，并概述了对抗性XAI（AdvXAI）的新兴研究方向。今后的工作应致力于改进解释方法和评价方案，以考虑到所报告的安全问题。



## **34. Your Diffusion Model is Secretly a Certifiably Robust Classifier**

你的扩散模型实际上是一个可靠的稳健分类器 cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.02316v2) [paper-pdf](http://arxiv.org/pdf/2402.02316v2)

**Authors**: Huanran Chen, Yinpeng Dong, Shitong Shao, Zhongkai Hao, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: Diffusion models are recently employed as generative classifiers for robust classification. However, a comprehensive theoretical understanding of the robustness of diffusion classifiers is still lacking, leading us to question whether they will be vulnerable to future stronger attacks. In this study, we propose a new family of diffusion classifiers, named Noised Diffusion Classifiers~(NDCs), that possess state-of-the-art certified robustness. Specifically, we generalize the diffusion classifiers to classify Gaussian-corrupted data by deriving the evidence lower bounds (ELBOs) for these distributions, approximating the likelihood using the ELBO, and calculating classification probabilities via Bayes' theorem. We integrate these generalized diffusion classifiers with randomized smoothing to construct smoothed classifiers possessing non-constant Lipschitzness. Experimental results demonstrate the superior certified robustness of our proposed NDCs. Notably, we are the first to achieve 80\%+ and 70\%+ certified robustness on CIFAR-10 under adversarial perturbations with $\ell_2$ norm less than 0.25 and 0.5, respectively, using a single off-the-shelf diffusion model without any additional data.

摘要: 近年来，扩散模型被用作产生式分类器来实现稳健分类。然而，对扩散分类器的稳健性在理论上仍然缺乏全面的理解，这使得我们质疑它们是否会容易受到未来更强大的攻击。在这项研究中，我们提出了一类新的扩散分类器，称为带噪声扩散分类器(NDC)，它具有最先进的证明的稳健性。具体地说，我们通过推导这些分布的证据下界(ELBO)，使用ELBO近似似然估计，并通过贝叶斯定理计算分类概率，来推广扩散分类器来分类受高斯污染的数据。我们将这些广义扩散分类器与随机平滑相结合，构造出具有非常数Lipschitz性的平滑分类器。实验结果表明，该算法具有较好的稳健性。值得注意的是，在没有任何额外数据的情况下，我们首次使用单一的现成扩散模型在对抗性扰动下获得了CIFAR-10的80+和70+证明的稳健性，其中$2$范数分别小于0.25和0.5。



## **35. Discriminative Adversarial Unlearning**

辨别性对抗性遗忘 cs.LG

13 pages including references, 2 tables, 2 figures and 1 algorithm

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2402.06864v2) [paper-pdf](http://arxiv.org/pdf/2402.06864v2)

**Authors**: Rohan Sharma, Shijie Zhou, Kaiyi Ji, Changyou Chen

**Abstract**: We introduce a novel machine unlearning framework founded upon the established principles of the min-max optimization paradigm. We capitalize on the capabilities of strong Membership Inference Attacks (MIA) to facilitate the unlearning of specific samples from a trained model. We consider the scenario of two networks, the attacker $\mathbf{A}$ and the trained defender $\mathbf{D}$ pitted against each other in an adversarial objective, wherein the attacker aims at teasing out the information of the data to be unlearned in order to infer membership, and the defender unlearns to defend the network against the attack, whilst preserving its general performance. The algorithm can be trained end-to-end using backpropagation, following the well known iterative min-max approach in updating the attacker and the defender. We additionally incorporate a self-supervised objective effectively addressing the feature space discrepancies between the forget set and the validation set, enhancing unlearning performance. Our proposed algorithm closely approximates the ideal benchmark of retraining from scratch for both random sample forgetting and class-wise forgetting schemes on standard machine-unlearning datasets. Specifically, on the class unlearning scheme, the method demonstrates near-optimal performance and comprehensively overcomes known methods over the random sample forgetting scheme across all metrics and multiple network pruning strategies.

摘要: 我们介绍了一种新的机器遗忘框架，该框架建立在最小-最大优化范例的既定原则上。我们利用强成员推理攻击(MIA)的能力来帮助从训练的模型中忘记特定样本。我们考虑了两个网络的场景，攻击者$\mathbf{A}$和训练有素的防御者$\mathbf{D}$相互对抗，其中攻击者的目标是梳理出待学习的数据信息以推断成员身份，而防御者则不学习保护网络免受攻击，同时保持其总体性能。该算法可以使用反向传播进行端到端的训练，遵循众所周知的更新攻击者和防御者的迭代最小-最大方法。此外，我们还引入了一个自监督目标，有效地解决了遗忘集和验证集之间的特征空间差异，提高了遗忘性能。对于标准机器遗忘数据集上的随机样本遗忘和分类遗忘方案，我们提出的算法非常接近从零开始重新训练的理想基准。具体地说，在类遗忘方案上，该方法表现出接近最优的性能，并在所有度量和多种网络剪枝策略上全面克服了随机样本遗忘方案的已知方法。



## **36. Input Validation for Neural Networks via Runtime Local Robustness Verification**

基于局部鲁棒性检验的神经网络输入确认 cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2002.03339v2) [paper-pdf](http://arxiv.org/pdf/2002.03339v2)

**Authors**: Jiangchao Liu, Liqian Chen, Antoine Mine, Ji Wang

**Abstract**: Local robustness verification can verify that a neural network is robust wrt. any perturbation to a specific input within a certain distance. We call this distance Robustness Radius. We observe that the robustness radii of correctly classified inputs are much larger than that of misclassified inputs which include adversarial examples, especially those from strong adversarial attacks. Another observation is that the robustness radii of correctly classified inputs often follow a normal distribution. Based on these two observations, we propose to validate inputs for neural networks via runtime local robustness verification. Experiments show that our approach can protect neural networks from adversarial examples and improve their accuracies.

摘要: 局部稳健性验证可以验证神经网络是稳健的WRT。在一定距离内对特定输入的任何扰动。我们称这一距离为稳健半径。我们观察到，正确分类的输入比错误分类的输入的鲁棒性半径要大得多，错误分类的输入包括对抗性例子，特别是来自强对抗性攻击的例子。另一个观察是，正确分类的输入的稳健性半径通常服从正态分布。基于这两个观察结果，我们建议通过运行时的局部健壮性验证来验证神经网络的输入。实验表明，该方法能够保护神经网络不受敌意样本的影响，提高了神经网络的准确率。



## **37. Bridging Optimal Transport and Jacobian Regularization by Optimal Trajectory for Enhanced Adversarial Defense**

增强对抗防御的最优轨迹桥接最优传输和雅可比正则化 cs.CV

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2303.11793v3) [paper-pdf](http://arxiv.org/pdf/2303.11793v3)

**Authors**: Binh M. Le, Shahroz Tariq, Simon S. Woo

**Abstract**: Deep neural networks, particularly in vision tasks, are notably susceptible to adversarial perturbations. To overcome this challenge, developing a robust classifier is crucial. In light of the recent advancements in the robustness of classifiers, we delve deep into the intricacies of adversarial training and Jacobian regularization, two pivotal defenses. Our work is the first carefully analyzes and characterizes these two schools of approaches, both theoretically and empirically, to demonstrate how each approach impacts the robust learning of a classifier. Next, we propose our novel Optimal Transport with Jacobian regularization method, dubbed OTJR, bridging the input Jacobian regularization with the a output representation alignment by leveraging the optimal transport theory. In particular, we employ the Sliced Wasserstein distance that can efficiently push the adversarial samples' representations closer to those of clean samples, regardless of the number of classes within the dataset. The SW distance provides the adversarial samples' movement directions, which are much more informative and powerful for the Jacobian regularization. Our empirical evaluations set a new standard in the domain, with our method achieving commendable accuracies of 52.57% on CIFAR-10 and 28.3% on CIFAR-100 datasets under the AutoAttack. Further validating our model's practicality, we conducted real-world tests by subjecting internet-sourced images to online adversarial attacks. These demonstrations highlight our model's capability to counteract sophisticated adversarial perturbations, affirming its significance and applicability in real-world scenarios.

摘要: 深度神经网络，特别是在视觉任务中，特别容易受到对抗性扰动的影响。要克服这一挑战，开发一个健壮的分类器至关重要。鉴于最近在分类器稳健性方面的进展，我们深入研究了对抗性训练和雅可比正则化这两个关键防御措施的复杂性。我们的工作是第一次从理论和经验上仔细分析和表征这两种方法，以证明每种方法是如何影响分类器的稳健学习的。接下来，我们提出了一种新的基于雅可比正则化的最优传输方法，称为OTJR，利用最优传输理论将输入雅可比正则化与输出表示对齐连接起来。特别是，我们使用了切片Wasserstein距离，该距离可以有效地将对抗性样本的表示更接近于干净样本的表示，而不考虑数据集中的类数量。Sw距离提供了对抗性样本的运动方向，为雅可比正则化提供了更多的信息和更强大的能力。我们的经验评估在该领域建立了一个新的标准，我们的方法在AutoAttack下在CIFAR-10和CIFAR-100数据集上的准确率分别达到了52.57%和28.3%。为了进一步验证我们模型的实用性，我们进行了真实世界的测试，对来自互联网的图像进行了在线敌意攻击。这些演示突出了我们的模型对抗复杂对抗性扰动的能力，肯定了它在现实世界场景中的重要性和适用性。



## **38. Agnostic Multi-Robust Learning Using ERM**

基于ERM的不可知多鲁棒学习 cs.LG

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2303.08944v2) [paper-pdf](http://arxiv.org/pdf/2303.08944v2)

**Authors**: Saba Ahmadi, Avrim Blum, Omar Montasser, Kevin Stangl

**Abstract**: A fundamental problem in robust learning is asymmetry: a learner needs to correctly classify every one of exponentially-many perturbations that an adversary might make to a test-time natural example. In contrast, the attacker only needs to find one successful perturbation. Xiang et al.[2022] proposed an algorithm that in the context of patch attacks for image classification, reduces the effective number of perturbations from an exponential to a polynomial number of perturbations and learns using an ERM oracle. However, to achieve its guarantee, their algorithm requires the natural examples to be robustly realizable. This prompts the natural question; can we extend their approach to the non-robustly-realizable case where there is no classifier with zero robust error?   Our first contribution is to answer this question affirmatively by reducing this problem to a setting in which an algorithm proposed by Feige et al.[2015] can be applied, and in the process extend their guarantees. Next, we extend our results to a multi-group setting and introduce a novel agnostic multi-robust learning problem where the goal is to learn a predictor that achieves low robust loss on a (potentially) rich collection of subgroups.

摘要: 鲁棒学习中的一个基本问题是不对称性：学习者需要正确地分类对手可能对测试时自然示例做出的指数级干扰中的每一个。相反，攻击者只需要找到一个成功的扰动。Xiang等[2022]提出了一种算法，在图像分类的补丁攻击的背景下，将扰动的有效数量从指数减少到多项式数量的扰动，并使用ERM oracle进行学习。然而，为了实现它的保证，他们的算法要求自然的例子是鲁棒可实现的。这就引出了一个自然的问题：我们能否将他们的方法扩展到非鲁棒可实现的情况，即没有鲁棒误差为零的分类器？   我们的第一个贡献是肯定地回答这个问题，减少这个问题的设置中，Feige等人提出的算法。[2015年]可以适用，并在此过程中延长其保障。接下来，我们将我们的结果扩展到多组设置，并引入一个新的不可知多鲁棒学习问题，其目标是学习一个预测器，该预测器在（潜在）丰富的子组集合上实现低鲁棒损失。



## **39. Adversarial Robustness on Image Classification with $k$-means**

基于$k$-均值的图像分类的对抗稳健性 cs.LG

6 pages, 3 figures, 2 equations, 1 algorithm

**SubmitDate**: 2024-02-13    [abs](http://arxiv.org/abs/2312.09533v2) [paper-pdf](http://arxiv.org/pdf/2312.09533v2)

**Authors**: Rollin Omari, Junae Kim, Paul Montague

**Abstract**: In this paper we explore the challenges and strategies for enhancing the robustness of $k$-means clustering algorithms against adversarial manipulations. We evaluate the vulnerability of clustering algorithms to adversarial attacks, emphasising the associated security risks. Our study investigates the impact of incremental attack strength on training, introduces the concept of transferability between supervised and unsupervised models, and highlights the sensitivity of unsupervised models to sample distributions. We additionally introduce and evaluate an adversarial training method that improves testing performance in adversarial scenarios, and we highlight the importance of various parameters in the proposed training method, such as continuous learning, centroid initialisation, and adversarial step-count.

摘要: 在这篇文章中，我们探讨了增强$k$-Means聚类算法对对手操纵的健壮性的挑战和策略。我们评估了集群算法对敌意攻击的脆弱性，强调了相关的安全风险。我们的研究考察了增量攻击强度对训练的影响，引入了监督模型和非监督模型之间可转换性的概念，并强调了非监督模型对样本分布的敏感性。此外，我们还介绍和评估了一种提高对抗场景下测试性能的对抗训练方法，并强调了所提出的训练方法中各种参数的重要性，如连续学习、质心初始化和对抗步数。



## **40. Multi-Attribute Vision Transformers are Efficient and Robust Learners**

多属性视觉变换器是高效和鲁棒的学习器 cs.CV

Code: https://github.com/hananshafi/MTL-ViT. arXiv admin note: text  overlap with arXiv:2207.08677 by other authors

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.08070v1) [paper-pdf](http://arxiv.org/pdf/2402.08070v1)

**Authors**: Hanan Gani, Nada Saadi, Noor Hussein, Karthik Nandakumar

**Abstract**: Since their inception, Vision Transformers (ViTs) have emerged as a compelling alternative to Convolutional Neural Networks (CNNs) across a wide spectrum of tasks. ViTs exhibit notable characteristics, including global attention, resilience against occlusions, and adaptability to distribution shifts. One underexplored aspect of ViTs is their potential for multi-attribute learning, referring to their ability to simultaneously grasp multiple attribute-related tasks. In this paper, we delve into the multi-attribute learning capability of ViTs, presenting a straightforward yet effective strategy for training various attributes through a single ViT network as distinct tasks. We assess the resilience of multi-attribute ViTs against adversarial attacks and compare their performance against ViTs designed for single attributes. Moreover, we further evaluate the robustness of multi-attribute ViTs against a recent transformer based attack called Patch-Fool. Our empirical findings on the CelebA dataset provide validation for our assertion.

摘要: 自诞生以来，视觉变压器(VITS)已经成为卷积神经网络(CNN)在广泛任务范围内的一种引人注目的替代方案。VITS表现出显著的特征，包括全球注意力、对闭塞的弹性和对分布变化的适应性。VITS的一个未被开发的方面是其多属性学习的潜力，指的是它们同时掌握多个与属性相关的任务的能力。在本文中，我们深入研究了VITS的多属性学习能力，提出了一种简单而有效的策略，通过单个VIT网络将各种属性作为不同的任务进行训练。我们评估了多属性VITS抵抗敌意攻击的能力，并与单属性VITS的性能进行了比较。此外，我们进一步评估了多属性VITS对最近一种称为Patch-Fool的基于变压器的攻击的健壮性。我们在CelebA数据集上的经验发现为我们的断言提供了验证。



## **41. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2309.02705v3) [paper-pdf](http://arxiv.org/pdf/2309.02705v3)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, Himabindu Lakkaraju

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that add malicious tokens to an input prompt to bypass the safety guardrails of an LLM and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework for defending against adversarial prompts with certifiable safety guarantees. Given a prompt, our procedure erases tokens individually and inspects the resulting subsequences using a safety filter. Our safety certificate guarantees that harmful prompts are not mislabeled as safe due to an adversarial attack up to a certain size. We implement the safety filter in two ways, using Llama 2 and DistilBERT, and compare the performance of erase-and-check for the two cases. We defend against three attack modes: i) adversarial suffix, where an adversarial sequence is appended at the end of a harmful prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Our experimental results demonstrate that this procedure can obtain strong certified safety guarantees on harmful prompts while maintaining good empirical performance on safe prompts. Additionally, we propose three efficient empirical defenses: i) RandEC, a randomized subsampling version of erase-and-check; ii) GreedyEC, which greedily erases tokens that maximize the softmax score of the harmful class; and iii) GradEC, which uses gradient information to optimize tokens to erase. We demonstrate their effectiveness against adversarial prompts generated by the Greedy Coordinate Gradient (GCG) attack algorithm. The code for our experiments is available at https://github.com/aounon/certified-llm-safety.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击会向输入提示添加恶意令牌，以绕过LLM的安全护栏，导致其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个用于防御具有可证明安全保证的对抗性提示的框架。在给定提示的情况下，我们的过程将逐个擦除令牌，并使用安全过滤器检查结果子序列。我们的安全证书保证有害提示不会因为达到一定大小的敌意攻击而被错误地标记为安全。我们用Llama 2和DistilBERT两种方法实现了安全过滤器，并比较了两种情况下的擦除和检查性能。我们防御三种攻击模式：i)对抗性后缀，其中对抗性序列被附加在有害提示的末尾；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该方法在对安全提示保持较好经验性能的同时，对有害提示具有较强的认证安全保障。此外，我们还提出了三种有效的经验防御：i)RandEC，一种随机化的擦除和检查版本；ii)GreedyEC，它贪婪地擦除使有害类别的Softmax得分最大化的标记；以及iii)Gradec，它使用梯度信息来优化要擦除的标记。我们证明了它们对贪婪坐标梯度(GCG)攻击算法生成的敌意提示的有效性。我们实验的代码可以在https://github.com/aounon/certified-llm-safety.上找到



## **42. Privacy-Preserving Gaze Data Streaming in Immersive Interactive Virtual Reality: Robustness and User Experience**

沉浸式交互式虚拟现实中保护隐私的凝视数据流：健壮性和用户体验 cs.HC

To appear in IEEE Transactions on Visualization and Computer Graphics

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07687v1) [paper-pdf](http://arxiv.org/pdf/2402.07687v1)

**Authors**: Ethan Wilson, Azim Ibragimov, Michael J. Proulx, Sai Deep Tetali, Kevin Butler, Eakta Jain

**Abstract**: Eye tracking is routinely being incorporated into virtual reality (VR) systems. Prior research has shown that eye tracking data can be used for re-identification attacks. The state of our knowledge about currently existing privacy mechanisms is limited to privacy-utility trade-off curves based on data-centric metrics of utility, such as prediction error, and black-box threat models. We propose that for interactive VR applications, it is essential to consider user-centric notions of utility and a variety of threat models. We develop a methodology to evaluate real-time privacy mechanisms for interactive VR applications that incorporate subjective user experience and task performance metrics. We evaluate selected privacy mechanisms using this methodology and find that re-identification accuracy can be decreased to as low as 14% while maintaining a high usability score and reasonable task performance. Finally, we elucidate three threat scenarios (black-box, black-box with exemplars, and white-box) and assess how well the different privacy mechanisms hold up to these adversarial scenarios. This work advances the state of the art in VR privacy by providing a methodology for end-to-end assessment of the risk of re-identification attacks and potential mitigating solutions.

摘要: 眼球跟踪通常会被整合到虚拟现实(VR)系统中。先前的研究表明，眼睛跟踪数据可以用于重新识别攻击。我们对当前现有隐私机制的了解仅限于基于以数据为中心的效用度量的隐私效用权衡曲线，如预测误差和黑盒威胁模型。我们认为，对于交互式VR应用，必须考虑以用户为中心的效用概念和各种威胁模型。我们开发了一种方法来评估交互式VR应用程序的实时隐私机制，该机制结合了主观用户体验和任务性能度量。我们使用这种方法对选定的隐私机制进行了评估，发现在保持较高的可用性分数和合理的任务性能的同时，重新识别的准确率可以降低到14%。最后，我们阐明了三种威胁场景(黑盒、带样本的黑盒和白盒)，并评估了不同的隐私机制对这些对抗性场景的支持程度。这项工作通过提供一种端到端的方法来评估重新识别攻击的风险和潜在的缓解解决方案，从而推动了VR隐私的发展。



## **43. Tighter Bounds on the Information Bottleneck with Application to Deep Learning**

信息瓶颈的更严格边界及其在深度学习中的应用 cs.LG

10 pages, 5 figures, code included in github repo

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07639v1) [paper-pdf](http://arxiv.org/pdf/2402.07639v1)

**Authors**: Nir Weingarten, Zohar Yakhini, Moshe Butman, Ran Gilad-Bachrach

**Abstract**: Deep Neural Nets (DNNs) learn latent representations induced by their downstream task, objective function, and other parameters. The quality of the learned representations impacts the DNN's generalization ability and the coherence of the emerging latent space. The Information Bottleneck (IB) provides a hypothetically optimal framework for data modeling, yet it is often intractable. Recent efforts combined DNNs with the IB by applying VAE-inspired variational methods to approximate bounds on mutual information, resulting in improved robustness to adversarial attacks. This work introduces a new and tighter variational bound for the IB, improving performance of previous IB-inspired DNNs. These advancements strengthen the case for the IB and its variational approximations as a data modeling framework, and provide a simple method to significantly enhance the adversarial robustness of classifier DNNs.

摘要: 深度神经网络(DNN)学习由其下游任务、目标函数和其他参数引起的潜在表示。学习表示的质量直接影响DNN的泛化能力和隐含空间的连贯性。信息瓶颈(IB)为数据建模提供了一个假设最优的框架，但它通常是难以解决的。最近的努力通过应用VAE启发的变分方法来近似互信息界，将DNN和IB结合起来，从而提高了对对手攻击的鲁棒性。这项工作为IB引入了一个新的更紧密的变分界，改进了以前IB启发的DNN的性能。这些改进加强了IB及其变分近似作为数据建模框架的理由，并提供了一种简单的方法来显著增强分类器DNN的对抗性健壮性。



## **44. BadLabel: A Robust Perspective on Evaluating and Enhancing Label-noise Learning**

BadLabel：评估和加强标签噪声学习的稳健视角 cs.LG

IEEE T-PAMI 2024 Accept

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2305.18377v2) [paper-pdf](http://arxiv.org/pdf/2305.18377v2)

**Authors**: Jingfeng Zhang, Bo Song, Haohan Wang, Bo Han, Tongliang Liu, Lei Liu, Masashi Sugiyama

**Abstract**: Label-noise learning (LNL) aims to increase the model's generalization given training data with noisy labels. To facilitate practical LNL algorithms, researchers have proposed different label noise types, ranging from class-conditional to instance-dependent noises. In this paper, we introduce a novel label noise type called BadLabel, which can significantly degrade the performance of existing LNL algorithms by a large margin. BadLabel is crafted based on the label-flipping attack against standard classification, where specific samples are selected and their labels are flipped to other labels so that the loss values of clean and noisy labels become indistinguishable. To address the challenge posed by BadLabel, we further propose a robust LNL method that perturbs the labels in an adversarial manner at each epoch to make the loss values of clean and noisy labels again distinguishable. Once we select a small set of (mostly) clean labeled data, we can apply the techniques of semi-supervised learning to train the model accurately. Empirically, our experimental results demonstrate that existing LNL algorithms are vulnerable to the newly introduced BadLabel noise type, while our proposed robust LNL method can effectively improve the generalization performance of the model under various types of label noise. The new dataset of noisy labels and the source codes of robust LNL algorithms are available at https://github.com/zjfheart/BadLabels.

摘要: 标签噪声学习(LNL)的目的是在给定含有噪声标签的训练数据的情况下提高模型的泛化能力。为了便于实用的LNL算法，研究人员提出了不同的标签噪声类型，从类条件噪声到实例相关噪声。本文引入了一种新的标签噪声类型BadLabel，它可以显著降低现有LNL算法的性能。BadLabel是基于针对标准分类的标签翻转攻击而构建的，在标准分类中，选择特定的样本，并将其标签翻转到其他标签，从而使干净和噪声标签的损失值变得无法区分。为了应对BadLabel带来的挑战，我们进一步提出了一种稳健的LNL方法，该方法在每个历元以对抗性的方式扰动标签，使干净标签和噪声标签的丢失值再次可区分。一旦我们选择了一小部分(大部分)干净的标签数据，我们就可以应用半监督学习技术来准确地训练模型。实验结果表明，现有的LNL算法容易受到新引入的BadLabel噪声类型的影响，而本文提出的稳健LNL方法可以有效地提高模型在各种类型标签噪声下的泛化性能。新的噪声标注数据集和健壮的LNL算法的源代码可在https://github.com/zjfheart/BadLabels.获得



## **45. Universal Jailbreak Backdoors from Poisoned Human Feedback**

来自有毒人类反馈的通用越狱后门 cs.AI

Accepted as conference paper in ICLR 2024

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2311.14455v3) [paper-pdf](http://arxiv.org/pdf/2311.14455v3)

**Authors**: Javier Rando, Florian Tramèr

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak backdoors.

摘要: 来自人类反馈的强化学习(RLHF)被用来对齐大型语言模型以产生有益和无害的响应。然而，先前的工作表明，这些模型可以通过找到敌意提示来越狱，这些提示可以将模型恢复到其不一致的行为。在本文中，我们考虑了一种新的威胁，即攻击者在RLHF训练数据中下毒，以便在模型中嵌入“越狱后门”。后门将触发词嵌入到模型中，其作用类似于通用的“sudo命令”：将触发词添加到任何提示中都可以实现有害的响应，而无需搜索敌意提示。通用越狱后门比之前在语言模型上研究的后门要强大得多，我们发现使用常见的后门攻击技术来植入它们要困难得多。我们调查了RLHF中有助于其所谓的健壮性的设计决策，并发布了一个有毒模型的基准，以刺激未来对通用越狱后门的研究。



## **46. Accelerated Smoothing: A Scalable Approach to Randomized Smoothing**

加速平滑：一种可扩展的随机平滑方法 cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07498v1) [paper-pdf](http://arxiv.org/pdf/2402.07498v1)

**Authors**: Devansh Bhardwaj, Kshitiz Kaushik, Sarthak Gupta

**Abstract**: Randomized smoothing has emerged as a potent certifiable defense against adversarial attacks by employing smoothing noises from specific distributions to ensure the robustness of a smoothed classifier. However, the utilization of Monte Carlo sampling in this process introduces a compute-intensive element, which constrains the practicality of randomized smoothing on a larger scale. To address this limitation, we propose a novel approach that replaces Monte Carlo sampling with the training of a surrogate neural network. Through extensive experimentation in various settings, we demonstrate the efficacy of our approach in approximating the smoothed classifier with remarkable precision. Furthermore, we demonstrate that our approach significantly accelerates the robust radius certification process, providing nearly $600$X improvement in computation time, overcoming the computational bottlenecks associated with traditional randomized smoothing.

摘要: 随机平滑已经成为对抗性攻击的一种有效的可证明的防御方法，它通过使用来自特定分布的平滑噪声来确保平滑分类器的鲁棒性。然而，在这个过程中利用蒙特卡罗抽样引入了计算密集型元素，这限制了更大规模的随机平滑的实用性。为了解决这个问题，我们提出了一种新的方法，取代蒙特卡洛采样与代理神经网络的训练。通过在各种设置中进行广泛的实验，我们证明了我们的方法在近似平滑分类器具有显着的精度的功效。此外，我们证明了我们的方法显着加快了强大的半径认证过程，提供了近600$X的计算时间的改善，克服了与传统的随机平滑相关的计算瓶颈。



## **47. Understanding Deep Learning defenses Against Adversarial Examples Through Visualizations for Dynamic Risk Assessment**

通过动态风险评估的可视化来了解对抗性示例的深度学习防御 cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07496v1) [paper-pdf](http://arxiv.org/pdf/2402.07496v1)

**Authors**: Xabier Echeberria-Barrio, Amaia Gil-Lerchundi, Jon Egana-Zubia, Raul Orduna-Urrutia

**Abstract**: In recent years, Deep Neural Network models have been developed in different fields, where they have brought many advances. However, they have also started to be used in tasks where risk is critical. A misdiagnosis of these models can lead to serious accidents or even death. This concern has led to an interest among researchers to study possible attacks on these models, discovering a long list of vulnerabilities, from which every model should be defended. The adversarial example attack is a widely known attack among researchers, who have developed several defenses to avoid such a threat. However, these defenses are as opaque as a deep neural network model, how they work is still unknown. This is why visualizing how they change the behavior of the target model is interesting in order to understand more precisely how the performance of the defended model is being modified. For this work, some defenses, against adversarial example attack, have been selected in order to visualize the behavior modification of each of them in the defended model. Adversarial training, dimensionality reduction and prediction similarity were the selected defenses, which have been developed using a model composed by convolution neural network layers and dense neural network layers. In each defense, the behavior of the original model has been compared with the behavior of the defended model, representing the target model by a graph in a visualization.

摘要: 近年来，深度神经网络模型在不同的领域得到了发展，带来了许多新的进展。然而，它们也开始被用于风险严重的任务。这些模型的误诊可能会导致严重的事故甚至死亡。这种担忧引发了研究人员的兴趣，他们研究这些模型可能受到的攻击，发现了一长串漏洞，每个模型都应该防御这些漏洞。对抗性攻击是研究人员中广为人知的攻击，他们已经开发了几种防御措施来避免这样的威胁。然而，这些防御措施就像深度神经网络模型一样不透明，它们如何工作仍是未知的。这就是为什么可视化它们如何更改目标模型的行为是有趣的，以便更准确地了解防御模型的性能是如何被修改的。在这项工作中，针对对手例子攻击选择了一些防御措施，以便在防御模型中可视化每个防御措施的行为修改。对抗性训练、降维和预测相似性是采用卷积神经网络层和密集神经网络层组成的模型开发的防御措施。在每个防御中，将原始模型的行为与防御模型的行为进行比较，在可视化中用图形表示目标模型。



## **48. Malicious Package Detection using Metadata Information**

基于元数据信息的恶意包检测 cs.CR

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07444v1) [paper-pdf](http://arxiv.org/pdf/2402.07444v1)

**Authors**: S. Halder, M. Bewong, A. Mahboubi, Y. Jiang, R. Islam, Z. Islam, R. Ip, E. Ahmed, G. Ramachandran, A. Babar

**Abstract**: Protecting software supply chains from malicious packages is paramount in the evolving landscape of software development. Attacks on the software supply chain involve attackers injecting harmful software into commonly used packages or libraries in a software repository. For instance, JavaScript uses Node Package Manager (NPM), and Python uses Python Package Index (PyPi) as their respective package repositories. In the past, NPM has had vulnerabilities such as the event-stream incident, where a malicious package was introduced into a popular NPM package, potentially impacting a wide range of projects. As the integration of third-party packages becomes increasingly ubiquitous in modern software development, accelerating the creation and deployment of applications, the need for a robust detection mechanism has become critical. On the other hand, due to the sheer volume of new packages being released daily, the task of identifying malicious packages presents a significant challenge. To address this issue, in this paper, we introduce a metadata-based malicious package detection model, MeMPtec. This model extracts a set of features from package metadata information. These extracted features are classified as either easy-to-manipulate (ETM) or difficult-to-manipulate (DTM) features based on monotonicity and restricted control properties. By utilising these metadata features, not only do we improve the effectiveness of detecting malicious packages, but also we demonstrate its resistance to adversarial attacks in comparison with existing state-of-the-art. Our experiments indicate a significant reduction in both false positives (up to 97.56%) and false negatives (up to 91.86%).

摘要: 在不断发展的软件开发环境中，保护软件供应链免受恶意程序包的攻击是至关重要的。对软件供应链的攻击涉及攻击者将有害软件注入软件存储库中常用的包或库中。例如，JavaScript使用Node Package Manager(NPM)，而Python使用Python Package Index(PyPI)作为它们各自的包库。在过去，NPM存在诸如事件流事件之类的漏洞，在该事件流事件中，恶意包被引入到流行的NPM包中，潜在地影响了广泛的项目。随着第三方包的集成在现代软件开发中变得越来越普遍，加速了应用程序的创建和部署，对强大检测机制的需求变得至关重要。另一方面，由于每天发布的新程序包数量巨大，识别恶意程序包的任务是一个巨大的挑战。针对这一问题，本文提出了一种基于元数据的恶意包检测模型MeMPtec。该模型从包元数据信息中提取一组特征。这些提取的特征根据单调性和受限控制属性被分类为易于操作(ETM)或难以操作(DTM)特征。通过利用这些元数据功能，我们不仅提高了检测恶意包的有效性，而且与现有的最先进水平相比，我们还展示了它对对手攻击的抵抗力。我们的实验表明，无论是假阳性(高达97.56%)还是假阴性(高达91.86%)都有显著的降低。



## **49. Accuracy of TextFooler black box adversarial attacks on 01 loss sign activation neural network ensemble**

TextFooler黑盒对抗性攻击01丢失信号激活神经网络集成的准确性 cs.LG

**SubmitDate**: 2024-02-12    [abs](http://arxiv.org/abs/2402.07347v1) [paper-pdf](http://arxiv.org/pdf/2402.07347v1)

**Authors**: Yunzhe Xue, Usman Roshan

**Abstract**: Recent work has shown the defense of 01 loss sign activation neural networks against image classification adversarial attacks. A public challenge to attack the models on CIFAR10 dataset remains undefeated. We ask the following question in this study: are 01 loss sign activation neural networks hard to deceive with a popular black box text adversarial attack program called TextFooler? We study this question on four popular text classification datasets: IMDB reviews, Yelp reviews, MR sentiment classification, and AG news classification. We find that our 01 loss sign activation network is much harder to attack with TextFooler compared to sigmoid activation cross entropy and binary neural networks. We also study a 01 loss sign activation convolutional neural network with a novel global pooling step specific to sign activation networks. With this new variation we see a significant gain in adversarial accuracy rendering TextFooler practically useless against it. We make our code freely available at \url{https://github.com/zero-one-loss/wordcnn01} and \url{https://github.com/xyzacademic/mlp01example}. Our work here suggests that 01 loss sign activation networks could be further developed to create fool proof models against text adversarial attacks.

摘要: 最近的工作表明，01丢失标记激活神经网络能够抵抗图像分类攻击。攻击CIFAR10数据集上的模型的公开挑战仍然是不败的。在这项研究中，我们提出了以下问题：01丢失符号激活神经网络是否很难使用流行的黑盒文本对抗性攻击程序TextFooler进行欺骗？我们在四个流行的文本分类数据集上研究了这个问题：IMDb评论、Yelp评论、MR情感分类和AG新闻分类。我们发现，与Sigmoid激活、交叉熵和二进制神经网络相比，我们的01丢失符号激活网络更难用TextFooler攻击。我们还研究了一种01丢失符号激活卷积神经网络，该网络具有一种新的针对符号激活网络的全局汇集步骤。在这个新的变种中，我们看到了对手精确度的显着提高，使得TextFooler对它几乎毫无用处。我们在\url{https://github.com/zero-one-loss/wordcnn01}和\url{https://github.com/xyzacademic/mlp01example}.免费提供我们的代码我们的工作表明，01丢失符号激活网络可以进一步发展，以创建抵抗文本对手攻击的傻瓜模型。



## **50. Intrinsic Biologically Plausible Adversarial Training**

内在生物学上看似合理的对抗性训练 cs.LG

**SubmitDate**: 2024-02-11    [abs](http://arxiv.org/abs/2309.17348v4) [paper-pdf](http://arxiv.org/pdf/2309.17348v4)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) excel in different daily tasks but have a dangerous vulnerability: inputs with small targeted perturbations, also known as adversarial samples, can drastically disrupt their performance. Adversarial training, a technique in which the training dataset is augmented with exemplary adversarial samples, is proven to mitigate this problem but comes at a high computational cost. In contrast to ANNs, humans are not susceptible to misclassifying these same adversarial samples, so one can postulate that biologically-plausible trained ANNs might be more robust against adversarial attacks. Choosing as a case study the biologically-plausible learning algorithm Present the Error to Perturb the Input To modulate Activity (PEPITA), we investigate this question through a comparative analysis with BP-trained ANNs on various computer vision tasks. We observe that PEPITA has a higher intrinsic adversarial robustness and, when adversarially trained, has a more favorable natural-vs-adversarial performance trade-off since, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average only by 0.26% while BP's decrease by 8.05%.

摘要: 用反向传播（BP）训练的人工神经网络（ANN）在不同的日常任务中表现出色，但有一个危险的漏洞：具有小目标扰动的输入，也称为对抗样本，可以严重破坏其性能。对抗训练是一种使用示例性对抗样本增强训练数据集的技术，已被证明可以缓解这个问题，但计算成本很高。与人工神经网络相比，人类不容易对这些相同的对抗性样本进行错误分类，因此可以假设生物学上合理的训练人工神经网络可能对对抗性攻击更鲁棒。作为一个案例研究的生物学合理的学习算法提出错误干扰输入调节活动（PEPITA），我们通过比较分析与BP训练的人工神经网络在各种计算机视觉任务调查这个问题。我们观察到，PEPITA具有更高的固有对抗鲁棒性，并且在对抗训练时，具有更有利的自然与对抗性能权衡，因为对于相同的自然准确率，PEPITA的对抗准确率平均仅下降0.26%，而BP下降8.05%。



