# Latest Adversarial Attack Papers
**update at 2022-05-25 06:31:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Graph Layer Security: Encrypting Information via Common Networked Physics**

图形层安全：通过公共网络物理加密信息 eess.SP

**SubmitDate**: 2022-05-23    [paper-pdf](http://arxiv.org/pdf/2006.03568v3)

**Authors**: Zhuangkun Wei, Liang Wang, Schyler Chengyao Sun, Bin Li, Weisi Guo

**Abstracts**: The proliferation of low-cost Internet of Things (IoT) devices has led to a race between wireless security and channel attacks. Traditional cryptography requires high-computational power and is not suitable for low-power IoT scenarios. Whist, recently developed physical layer security (PLS) can exploit common wireless channel state information (CSI), its sensitivity to channel estimation makes them vulnerable from attacks. In this work, we exploit an alternative common physics shared between IoT transceivers: the monitored channel-irrelevant physical networked dynamics (e.g., water/oil/gas/electrical signal-flows). Leveraging this, we propose for the first time, graph layer security (GLS), by exploiting the dependency in physical dynamics among network nodes for information encryption and decryption. A graph Fourier transform (GFT) operator is used to characterize such dependency into a graph-bandlimted subspace, which allows the generations of channel-irrelevant cipher keys by maximizing the secrecy rate. We evaluate our GLS against designed active and passive attackers, using IEEE 39-Bus system. Results demonstrate that, GLS is not reliant on wireless CSI, and can combat attackers that have partial networked dynamic knowledge (realistic access to full dynamic and critical nodes remains challenging). We believe this novel GLS has widespread applicability in secure health monitoring and for Digital Twins in adversarial radio environments.

摘要: 低成本物联网(IoT)设备的激增导致了无线安全和渠道攻击之间的竞争。传统的密码学需要高计算能力，不适合低功耗的物联网场景。然而，最近发展起来的物理层安全技术可以利用常见的无线信道状态信息，但其对信道估计的敏感性使其容易受到攻击。在这项工作中，我们利用了物联网收发器之间共享的另一种常见物理：受监控的与通道无关的物理网络动态(例如，水/油/气/电信号流)。利用这一点，我们首次提出了图层安全(GLS)，通过利用网络节点之间的物理动力学依赖性来进行信息加密和解密。图傅里叶变换(GFT)算子被用来将这种依赖描述到图带宽受限的子空间中，该子空间通过最大化保密率来生成与信道无关的密钥。我们使用IEEE 39节点系统，针对设计的主动和被动攻击者评估我们的GLS。结果表明，GLS不依赖于无线CSI，可以对抗具有部分网络动态知识的攻击者(实际访问完全动态和关键节点仍然具有挑战性)。我们相信这种新的GLS在安全的健康监测和对抗性无线电环境中的数字双胞胎中具有广泛的适用性。



## **2. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles***

网络化无人机隐身对手检测技术研究 eess.SY

to appear at the 2022 Int'l Conference on Unmanned Aircraft Systems  (ICUAS)

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2202.09661v2)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.

摘要: 无人机网络在执行复杂的协作任务时提供了分布式覆盖、可重构和机动性。然而，它依赖的无线通信可能会受到网络对手和入侵的影响，扰乱整个网络的运行。提出了一种基于模型的集中式和分散式观测器技术，用于检测编队控制环境中无人机的一类隐身入侵，即零动态攻击和隐蔽攻击。在控制中心运行的集中式观察器利用无人机通信拓扑中的切换来进行攻击检测，而分布式观察器在网络中的每一架无人机上实现，使用联网的无人机模型和本地可用的测量。实验结果表明，所提出的检测方案在不同的案例研究中是有效的。



## **3. Inverse-Inverse Reinforcement Learning. How to Hide Strategy from an Adversarial Inverse Reinforcement Learner**

逆-逆强化学习。如何从对抗性的逆强化学习器中隐藏策略 cs.LG

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2205.10802v1)

**Authors**: Kunal Pattanayak, Vikram Krishnamurthy, Christopher Berry

**Abstracts**: Inverse reinforcement learning (IRL) deals with estimating an agent's utility function from its actions. In this paper, we consider how an agent can hide its strategy and mitigate an adversarial IRL attack; we call this inverse IRL (I-IRL). How should the decision maker choose its response to ensure a poor reconstruction of its strategy by an adversary performing IRL to estimate the agent's strategy? This paper comprises four results: First, we present an adversarial IRL algorithm that estimates the agent's strategy while controlling the agent's utility function. Our second result for I-IRL result spoofs the IRL algorithm used by the adversary. Our I-IRL results are based on revealed preference theory in micro-economics. The key idea is for the agent to deliberately choose sub-optimal responses that sufficiently masks its true strategy. Third, we give a sample complexity result for our main I-IRL result when the agent has noisy estimates of the adversary specified utility function. Finally, we illustrate our I-IRL scheme in a radar problem where a meta-cognitive radar is trying to mitigate an adversarial target.

摘要: 逆强化学习(IRL)处理从主体的动作中估计主体的效用函数。在本文中，我们考虑了代理如何隐藏其策略并缓解对抗性IRL攻击，我们称之为逆IRL(I-IRL)。决策者应该如何选择其响应，以确保对手执行IRL来估计代理人的战略时，对其战略的重建效果不佳？首先，提出了一种对抗性IRL算法，该算法在估计代理策略的同时控制代理的效用函数。对于I-IRL结果，我们的第二个结果伪造了对手使用的IRL算法。我们的I-IRL结果是基于微观经济学中的揭示偏好理论。关键思想是让代理故意选择次优响应，以充分掩盖其真实战略。第三，当代理对对手指定的效用函数有噪声估计时，我们给出了主要I-IRL结果的样本复杂性结果。最后，我们在一个雷达问题中说明了我们的I-IRL方案，其中元认知雷达正试图减轻敌方目标的威胁。



## **4. Phrase-level Textual Adversarial Attack with Label Preservation**

具有标签保留的短语级文本对抗攻击 cs.CL

9 pages + 2 pages references + 8 pages appendix

**SubmitDate**: 2022-05-22    [paper-pdf](http://arxiv.org/pdf/2205.10710v1)

**Authors**: Yibin Lei, Yu Cao, Dianqi Li, Tianyi Zhou, Meng Fang, Mykola Pechenizkiy

**Abstracts**: Generating high-quality textual adversarial examples is critical for investigating the pitfalls of natural language processing (NLP) models and further promoting their robustness. Existing attacks are usually realized through word-level or sentence-level perturbations, which either limit the perturbation space or sacrifice fluency and textual quality, both affecting the attack effectiveness. In this paper, we propose Phrase-Level Textual Adversarial aTtack (PLAT) that generates adversarial samples through phrase-level perturbations. PLAT first extracts the vulnerable phrases as attack targets by a syntactic parser, and then perturbs them by a pre-trained blank-infilling model. Such flexible perturbation design substantially expands the search space for more effective attacks without introducing too many modifications, and meanwhile maintaining the textual fluency and grammaticality via contextualized generation using surrounding texts. Moreover, we develop a label-preservation filter leveraging the likelihoods of language models fine-tuned on each class, rather than textual similarity, to rule out those perturbations that potentially alter the original class label for humans. Extensive experiments and human evaluation demonstrate that PLAT has a superior attack effectiveness as well as a better label consistency than strong baselines.

摘要: 生成高质量的文本对抗性实例对于研究自然语言处理(NLP)模型的缺陷并进一步提高其稳健性至关重要。现有的攻击通常是通过词级或句子级的扰动来实现的，这要么限制了扰动空间，要么牺牲了流畅度和文本质量，两者都影响了攻击的有效性。在本文中，我们提出了短语级别的文本对抗攻击(PLAT)，它通过短语级别的扰动来生成敌对样本。PLAT首先通过句法分析器提取易受攻击的短语作为攻击目标，然后通过预先训练的空白填充模型对其进行扰动。这种灵活的扰动设计在不引入太多修改的情况下大大扩展了更有效的攻击的搜索空间，同时通过使用周围文本的上下文生成来保持文本的流畅性和语法性。此外，我们开发了一个标签保存过滤器，利用在每个类上微调的语言模型的可能性，而不是文本相似性，以排除那些可能改变人类原始类标签的扰动。大量的实验和人工评估表明，与强基线相比，PLAT具有更好的攻击效果和更好的标签一致性。



## **5. Post-breach Recovery: Protection against White-box Adversarial Examples for Leaked DNN Models**

漏洞后恢复：针对泄漏的DNN模型的白盒对抗示例 cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10686v1)

**Authors**: Shawn Shan, Wenxin Ding, Emily Wenger, Haitao Zheng, Ben Y. Zhao

**Abstracts**: Server breaches are an unfortunate reality on today's Internet. In the context of deep neural network (DNN) models, they are particularly harmful, because a leaked model gives an attacker "white-box" access to generate adversarial examples, a threat model that has no practical robust defenses. For practitioners who have invested years and millions into proprietary DNNs, e.g. medical imaging, this seems like an inevitable disaster looming on the horizon.   In this paper, we consider the problem of post-breach recovery for DNN models. We propose Neo, a new system that creates new versions of leaked models, alongside an inference time filter that detects and removes adversarial examples generated on previously leaked models. The classification surfaces of different model versions are slightly offset (by introducing hidden distributions), and Neo detects the overfitting of attacks to the leaked model used in its generation. We show that across a variety of tasks and attack methods, Neo is able to filter out attacks from leaked models with very high accuracy, and provides strong protection (7--10 recoveries) against attackers who repeatedly breach the server. Neo performs well against a variety of strong adaptive attacks, dropping slightly in # of breaches recoverable, and demonstrates potential as a complement to DNN defenses in the wild.

摘要: 在当今的互联网上，服务器入侵是一个不幸的现实。在深度神经网络(DNN)模型的背景下，它们尤其有害，因为泄露的模型让攻击者可以使用“白盒”来生成对抗性示例，这是一种没有实际可靠防御措施的威胁模型。对于在专有DNN(例如医学成像)上投入多年和数百万美元的从业者来说，这似乎是一场不可避免的灾难迫在眉睫。在本文中，我们考虑了DNN模型的漏洞后恢复问题。我们提出了Neo，一个新的系统，它创建新版本的泄漏模型，以及一个推理时间过滤器，检测并删除在以前泄漏的模型上生成的敌对示例。不同模型版本的分类面略有偏移(通过引入隐藏分布)，并且Neo检测到对其生成中使用的泄漏模型的攻击过拟合。我们发现，在各种任务和攻击方法中，Neo能够以非常高的准确率从泄露的模型中过滤攻击，并针对反复破坏服务器的攻击者提供强大的保护(7-10次恢复)。NEO在各种强自适应攻击中表现良好，在可恢复的漏洞数量中略有下降，并显示出在野外作为DNN防御的补充潜力。



## **6. Gradient Concealment: Free Lunch for Defending Adversarial Attacks**

梯度隐藏：防御对手攻击的免费午餐 cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10617v1)

**Authors**: Sen Pei, Jiaxi Sun, Xiaopeng Zhang, Gaofeng Meng

**Abstracts**: Recent studies show that the deep neural networks (DNNs) have achieved great success in various tasks. However, even the \emph{state-of-the-art} deep learning based classifiers are extremely vulnerable to adversarial examples, resulting in sharp decay of discrimination accuracy in the presence of enormous unknown attacks. Given the fact that neural networks are widely used in the open world scenario which can be safety-critical situations, mitigating the adversarial effects of deep learning methods has become an urgent need. Generally, conventional DNNs can be attacked with a dramatically high success rate since their gradient is exposed thoroughly in the white-box scenario, making it effortless to ruin a well trained classifier with only imperceptible perturbations in the raw data space. For tackling this problem, we propose a plug-and-play layer that is training-free, termed as \textbf{G}radient \textbf{C}oncealment \textbf{M}odule (GCM), concealing the vulnerable direction of gradient while guaranteeing the classification accuracy during the inference time. GCM reports superior defense results on the ImageNet classification benchmark, improving up to 63.41\% top-1 attack robustness (AR) when faced with adversarial inputs compared to the vanilla DNNs. Moreover, we use GCM in the CVPR 2022 Robust Classification Challenge, currently achieving \textbf{2nd} place in Phase II with only a tiny version of ConvNext. The code will be made available.

摘要: 最近的研究表明，深度神经网络在各种任务中取得了巨大的成功。然而，即使是基于深度学习的分类器也极易受到敌意例子的攻击，导致在存在大量未知攻击的情况下识别精度急剧下降。鉴于神经网络被广泛应用于开放世界场景中，这可能是安全关键的情况，缓解深度学习方法的对抗性已成为迫切需要。通常，传统的DNN可以以极高的成功率受到攻击，因为它们的梯度在白盒场景中被彻底暴露，使得在原始数据空间中只有不可察觉的扰动就可以毫不费力地破坏一个训练有素的分类器。为了解决这一问题，我们提出了一种无需训练的即插即用层，称为文本bf{G}辐射\文本bf{C}模块(GCM)，它隐藏了梯度的脆弱方向，同时保证了推理时的分类精度。GCM在ImageNet分类基准上报告了卓越的防御结果，与普通DNN相比，在面对敌意输入时，最高可提高63.41\%top-1攻击健壮性(AR)。此外，我们在CVPR 2022健壮分类挑战赛中使用了GCM，目前仅使用ConvNext的一个微型版本就在第二阶段获得了\textbf{2}名。代码将可用。



## **7. SERVFAIL: The Unintended Consequences of Algorithm Agility in DNSSEC**

SERVFAIL：DNSSEC中算法敏捷性的意外后果 cs.CR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10608v1)

**Authors**: Elias Heftrig, Jean-Pierre Seifert, Haya Shulman, Peter Thomassen, Michael Waidner, Nils Wisiol

**Abstracts**: Cryptographic algorithm agility is an important property for DNSSEC: it allows easy deployment of new algorithms if the existing ones are no longer secure. Significant operational and research efforts are dedicated to pushing the deployment of new algorithms in DNSSEC forward. Recent research shows that DNSSEC is gradually achieving algorithm agility: most DNSSEC supporting resolvers can validate a number of different algorithms and domains are increasingly signed with cryptographically strong ciphers.   In this work we show for the first time that the cryptographic agility in DNSSEC, although critical for making DNS secure with strong cryptography, also introduces a severe vulnerability. We find that under certain conditions, when new algorithms are listed in signed DNS responses, the resolvers do not validate DNSSEC. As a result, domains that deploy new ciphers, risk exposing the validating resolvers to cache poisoning attacks.   We use this to develop DNSSEC-downgrade attacks and show that in some situations these attacks can be launched even by off-path adversaries. We experimentally and ethically evaluate our attacks against popular DNS resolver implementations, public DNS providers, and DNS services used by web clients worldwide. We validate the success of DNSSEC-downgrade attacks by poisoning the resolvers: we inject fake records, in signed domains, into the caches of validating resolvers. We find that major DNS providers, such as Google Public DNS and Cloudflare, as well as 70% of DNS resolvers used by web clients are vulnerable to our attacks.   We trace the factors that led to this situation and provide recommendations.

摘要: 密码算法灵活性是DNSSEC的一个重要属性：如果现有算法不再安全，它允许轻松部署新算法。大量的操作和研究工作致力于推动在DNSSEC中部署新算法。最近的研究表明，DNSSEC正在逐步实现算法灵活性：大多数支持DNSSEC的解析器可以验证一些不同的算法，并且越来越多的域使用密码强密码签名。在这项工作中，我们第一次展示了DNSSEC的密码敏捷性，尽管对于使用强大的密码学来确保DNS安全至关重要，但也引入了一个严重的漏洞。我们发现，在某些条件下，当新算法在签名的DNS响应中列出时，解析器不会验证DNSSEC。因此，部署新密码的域有可能使验证解析器面临缓存中毒攻击。我们利用这一点来开发DNSSEC降级攻击，并表明在某些情况下，这些攻击甚至可以由偏离路径的对手发起。我们从实验和伦理上评估我们对全球Web客户端使用的流行的DNS解析器实现、公共DNS提供商和DNS服务的攻击。我们通过毒化解析器来验证DNSSEC降级攻击的成功：我们在有符号的域中向验证解析器的缓存中注入虚假记录。我们发现，主要的域名服务提供商，如Google Public DNS和Cloudflare，以及网络客户端使用的70%的域名解析程序，都容易受到我们的攻击。我们追踪了导致这种情况的因素并提出了建议。



## **8. On the Feasibility and Generality of Patch-based Adversarial Attacks on Semantic Segmentation Problems**

基于补丁的对抗性语义切分攻击的可行性和通用性 cs.CV

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2205.10539v1)

**Authors**: Soma Kontar, Andras Horvath

**Abstracts**: Deep neural networks were applied with success in a myriad of applications, but in safety critical use cases adversarial attacks still pose a significant threat. These attacks were demonstrated on various classification and detection tasks and are usually considered general in a sense that arbitrary network outputs can be generated by them.   In this paper we will demonstrate through simple case studies both in simulation and in real-life, that patch based attacks can be utilised to alter the output of segmentation networks. Through a few examples and the investigation of network complexity, we will also demonstrate that the number of possible output maps which can be generated via patch-based attacks of a given size is typically smaller than the area they effect or areas which should be attacked in case of practical applications.   We will prove that based on these results most patch-based attacks cannot be general in practice, namely they can not generate arbitrary output maps or if they could, they are spatially limited and this limit is significantly smaller than the receptive field of the patches.

摘要: 深度神经网络在许多应用中都取得了成功，但在安全关键用例中，对抗性攻击仍构成重大威胁。这些攻击在各种分类和检测任务中进行了演示，通常被认为是一般性的，因为它们可以生成任意的网络输出。在本文中，我们将通过模拟和现实生活中的简单案例研究来演示，基于补丁的攻击可以用来改变分段网络的输出。通过几个例子和对网络复杂性的调查，我们还将证明，在实际应用中，通过给定大小的基于补丁的攻击可以生成的可能输出地图的数量通常小于它们影响的区域或应该攻击的区域。基于这些结果，我们将证明大多数基于面片的攻击在实践中不是通用的，即它们不能生成任意的输出地图，或者如果它们可以，它们在空间上是受限的，并且这个限制明显小于面片的接受范围。



## **9. PRADA: Practical Black-Box Adversarial Attacks against Neural Ranking Models**

Prada：针对神经排序模型的实用黑箱对抗性攻击 cs.IR

**SubmitDate**: 2022-05-21    [paper-pdf](http://arxiv.org/pdf/2204.01321v2)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have shown remarkable success in recent years, especially with pre-trained language models. However, deep neural models are notorious for their vulnerability to adversarial examples. Adversarial attacks may become a new type of web spamming technique given our increased reliance on neural information retrieval models. Therefore, it is important to study potential adversarial attacks to identify vulnerabilities of NRMs before they are deployed.   In this paper, we introduce the Word Substitution Ranking Attack (WSRA) task against NRMs, which aims to promote a target document in rankings by adding adversarial perturbations to its text. We focus on the decision-based black-box attack setting, where the attackers have no access to the model parameters and gradients, but can only acquire the rank positions of the partial retrieved list by querying the target model. This attack setting is realistic in real-world search engines. We propose a novel Pseudo Relevance-based ADversarial ranking Attack method (PRADA) that learns a surrogate model based on Pseudo Relevance Feedback (PRF) to generate gradients for finding the adversarial perturbations.   Experiments on two web search benchmark datasets show that PRADA can outperform existing attack strategies and successfully fool the NRM with small indiscernible perturbations of text.

摘要: 近年来，神经网络排序模型(NRM)取得了显著的成功，尤其是使用了预先训练好的语言模型。然而，深层神经模型因其易受敌意例子的攻击而臭名昭著。鉴于我们对神经信息检索模型的日益依赖，对抗性攻击可能成为一种新型的Web垃圾邮件技术。因此，在部署NRM之前，研究潜在的敌意攻击以识别NRM的漏洞是很重要的。在本文中，我们引入了针对NRMS的单词替换排名攻击(WSRA)任务，该任务旨在通过在目标文档的文本中添加对抗性扰动来提升其排名。重点研究了基于决策的黑盒攻击环境，其中攻击者无法获取模型参数和梯度，只能通过查询目标模型获得部分检索列表的排名位置。这种攻击设置在现实世界的搜索引擎中是现实的。提出了一种新的基于伪相关性的对抗性排序攻击方法(PRADA)，该方法通过学习基于伪相关反馈(PRF)的代理模型来生成用于发现对抗性扰动的梯度。在两个网络搜索基准数据集上的实验表明，Prada可以超越现有的攻击策略，并成功地利用文本的微小不可分辨扰动来欺骗NRM。



## **10. Robust Sensible Adversarial Learning of Deep Neural Networks for Image Classification**

用于图像分类的深度神经网络稳健敏感对抗性学习 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10457v1)

**Authors**: Jungeum Kim, Xiao Wang

**Abstracts**: The idea of robustness is central and critical to modern statistical analysis. However, despite the recent advances of deep neural networks (DNNs), many studies have shown that DNNs are vulnerable to adversarial attacks. Making imperceptible changes to an image can cause DNN models to make the wrong classification with high confidence, such as classifying a benign mole as a malignant tumor and a stop sign as a speed limit sign. The trade-off between robustness and standard accuracy is common for DNN models. In this paper, we introduce sensible adversarial learning and demonstrate the synergistic effect between pursuits of standard natural accuracy and robustness. Specifically, we define a sensible adversary which is useful for learning a robust model while keeping high natural accuracy. We theoretically establish that the Bayes classifier is the most robust multi-class classifier with the 0-1 loss under sensible adversarial learning. We propose a novel and efficient algorithm that trains a robust model using implicit loss truncation. We apply sensible adversarial learning for large-scale image classification to a handwritten digital image dataset called MNIST and an object recognition colored image dataset called CIFAR10. We have performed an extensive comparative study to compare our method with other competitive methods. Our experiments empirically demonstrate that our method is not sensitive to its hyperparameter and does not collapse even with a small model capacity while promoting robustness against various attacks and keeping high natural accuracy.

摘要: 稳健性的概念是现代统计分析的核心和关键。然而，尽管深度神经网络(DNN)最近取得了进展，但许多研究表明DNN很容易受到对手的攻击。对图像进行不知不觉的更改可能会导致DNN模型以高置信度做出错误分类，例如将良性葡萄胎归类为恶性肿瘤，将停车标志归类为限速标志。对于DNN模型来说，稳健性和标准精度之间的权衡是很常见的。在本文中，我们引入了敏感的对抗性学习，并证明了追求标准自然准确性和稳健性之间的协同效应。具体地说，我们定义了一个明智的对手，它有助于学习健壮的模型，同时保持较高的自然准确性。我们从理论上证明了贝叶斯分类器是在敏感对抗性学习下具有0-1损失的最健壮的多类分类器。我们提出了一种新颖而高效的算法，该算法使用隐式损失截断来训练稳健的模型。在手写数字图像数据集MNIST和目标识别彩色图像数据集CIFAR10上，我们将敏感的对抗性学习应用于大规模图像分类。我们进行了广泛的比较研究，将我们的方法与其他竞争方法进行比较。实验表明，该方法对超参数不敏感，即使在模型容量较小的情况下也不会崩溃，同时提高了对各种攻击的鲁棒性，并保持了较高的自然准确率。



## **11. Vulnerability Analysis and Performance Enhancement of Authentication Protocol in Dynamic Wireless Power Transfer Systems**

动态无线电能传输系统中认证协议的脆弱性分析与性能提升 cs.CR

16 pages, conference

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10292v1)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstracts**: Recent advancements in wireless charging technology, as well as the possibility of utilizing it in the Electric Vehicle (EV) domain for dynamic charging solutions, have fueled the demand for a secure and usable protocol in the Dynamic Wireless Power Transfer (DWPT) technology. The DWPT must operate in the presence of malicious adversaries that can undermine the charging process and harm the customer service quality, while preserving the privacy of the users. Recently, it was shown that the DWPT system is susceptible to adversarial attacks, including replay, denial-of-service and free-riding attacks, which can lead to the adversary blocking the authorized user from charging, enabling free charging for free riders and exploiting the location privacy of the customers. In this paper, we study the current State-Of-The-Art (SOTA) authentication protocols and make the following two contributions: a) we show that the SOTA is vulnerable to the tracking of the user activity and b) we propose an enhanced authentication protocol that eliminates the vulnerability while providing improved efficiency compared to the SOTA authentication protocols. By adopting authentication messages based only on exclusive OR operations, hashing, and hash chains, we optimize the protocol to achieve a complexity that varies linearly with the number of charging pads, providing improved scalability. Compared to SOTA, the proposed scheme has a performance gain in the computational cost of around 90% on average for each pad.

摘要: 无线充电技术的最新进展，以及在电动汽车(EV)领域将其用于动态充电解决方案的可能性，推动了对动态无线功率传输(DWPT)技术中安全和可用的协议的需求。DWPT必须在恶意对手存在的情况下运行，这些恶意对手可能会破坏收费过程并损害客户服务质量，同时保护用户的隐私。最近，有研究表明，DWPT系统容易受到包括重放、拒绝服务和搭便车在内的敌意攻击，这些攻击可以导致对手阻止授权用户收费，使搭便车的人能够免费收费，并利用客户的位置隐私。在本文中，我们研究了现有的SOTA认证协议，并做了以下两个方面的贡献：a)我们发现SOTA容易受到用户活动跟踪的影响；b)我们提出了一种增强的认证协议，与SOTA认证协议相比，它消除了这个漏洞，同时提供了更高的效率。通过采用仅基于异或运算、哈希和哈希链的身份验证消息，我们对协议进行了优化，以实现随充电板数量线性变化的复杂性，从而提供更高的可扩展性。与SOTA相比，对于每个PAD，所提出的方案的计算代价平均提高了90%左右。



## **12. Adversarial Body Shape Search for Legged Robots**

腿部机器人的对抗性体型搜索 cs.RO

6 pages, 7 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10187v1)

**Authors**: Takaaki Azakami, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We propose an evolutionary computation method for an adversarial attack on the length and thickness of parts of legged robots by deep reinforcement learning. This attack changes the robot body shape and interferes with walking-we call the attacked body as adversarial body shape. The evolutionary computation method searches adversarial body shape by minimizing the expected cumulative reward earned through walking simulation. To evaluate the effectiveness of the proposed method, we perform experiments with three-legged robots, Walker2d, Ant-v2, and Humanoid-v2 in OpenAI Gym. The experimental results reveal that Walker2d and Ant-v2 are more vulnerable to the attack on the length than the thickness of the body parts, whereas Humanoid-v2 is vulnerable to the attack on both of the length and thickness. We further identify that the adversarial body shapes break left-right symmetry or shift the center of gravity of the legged robots. Finding adversarial body shape can be used to proactively diagnose the vulnerability of legged robot walking.

摘要: 提出了一种基于深度强化学习的腿部机器人长度和厚度对抗性攻击的进化计算方法。这种攻击改变了机器人的体型，干扰了行走--我们将被攻击的体型称为对抗性体型。进化计算方法通过最小化通过模拟行走获得的期望累积奖励来搜索对手的体型。为了评估该方法的有效性，我们在OpenAI健身房中用三足机器人Walker2d、Ant-v2和Human-v2进行了实验。实验结果表明，Walker2d和Ant-v2对长度的攻击比对身体部分的厚度更容易受到攻击，而人形v2对长度和厚度的攻击都更容易受到攻击。我们进一步识别出，对抗性的身体形状打破了左右对称或移动了腿部机器人的重心。发现敌对的体型可以用来主动诊断腿部机器人行走的脆弱性。



## **13. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

获得一轮保证：对认证健壮性的浮点攻击 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10159v1)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstracts**: Adversarial examples pose a security risk as they can alter a classifier's decision through slight perturbations to a benign input. Certified robustness has been proposed as a mitigation strategy where given an input $x$, a classifier returns a prediction and a radius with a provable guarantee that any perturbation to $x$ within this radius (e.g., under the $L_2$ norm) will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples within the certified radius. We show that the attack can be carried out against several linear classifiers that have exact certifiable guarantees and against neural network verifiers that return a certified lower bound on a robust radius. Our experiments demonstrate over 50% attack success rate on random linear classifiers, up to 35% on a breast cancer dataset for logistic regression, and a 9% attack success rate on the MNIST dataset for a neural network whose certified radius was verified by a prominent bound propagation method. We also show that state-of-the-art random smoothed classifiers for neural networks are also susceptible to adversarial examples (e.g., up to 2% attack rate on CIFAR10)-validating the importance of accounting for the error rate of robustness guarantees of such classifiers in practice. Finally, as a mitigation, we advocate the use of rounded interval arithmetic to account for rounding errors.

摘要: 敌意的例子会带来安全风险，因为它们可以通过轻微的扰动改变分类器的决定，使其成为良性的输入。证明的稳健性已经被提出作为一种缓解策略，在给定输入$x$的情况下，分类器返回预测和半径，并且可证明地保证在该半径内(例如，在$L_2$范数下)对$x$的任何扰动不会改变分类器的预测。在这项工作中，我们证明了这些保证可能会由于浮点表示的限制而失效，从而导致舍入误差。我们设计了一种四舍五入的搜索方法，可以有效地利用这个漏洞在认证的半径内找到对抗性示例。我们证明了该攻击可以针对具有精确可证明保证的几个线性分类器，以及针对返回关于稳健半径的证明下界的神经网络验证器。我们的实验表明，在随机线性分类器上的攻击成功率超过50%，在用于Logistic回归的乳腺癌数据集上的攻击成功率高达35%，在MNIST数据集上的攻击成功率在MNIST数据集上达到9%，其认证半径通过显著边界传播方法进行验证。我们还表明，最先进的随机平滑神经网络分类器也容易受到敌意例子的影响(例如，对CIFAR10的攻击率高达2%)-验证了在实践中考虑此类分类器的健壮性保证的错误率的重要性。最后，作为一种缓解措施，我们主张使用四舍五入区间算术来计算舍入误差。



## **14. Generating Semantic Adversarial Examples via Feature Manipulation**

通过特征处理生成语义对抗性实例 cs.LG

arXiv admin note: substantial text overlap with arXiv:1705.09064 by  other authors

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2001.02297v2)

**Authors**: Shuo Wang, Surya Nepal, Carsten Rudolph, Marthie Grobler, Shangyu Chen, Tianle Chen

**Abstracts**: The vulnerability of deep neural networks to adversarial attacks has been widely demonstrated (e.g., adversarial example attacks). Traditional attacks perform unstructured pixel-wise perturbation to fool the classifier. An alternative approach is to have perturbations in the latent space. However, such perturbations are hard to control due to the lack of interpretability and disentanglement. In this paper, we propose a more practical adversarial attack by designing structured perturbation with semantic meanings. Our proposed technique manipulates the semantic attributes of images via the disentangled latent codes. The intuition behind our technique is that images in similar domains have some commonly shared but theme-independent semantic attributes, e.g. thickness of lines in handwritten digits, that can be bidirectionally mapped to disentangled latent codes. We generate adversarial perturbation by manipulating a single or a combination of these latent codes and propose two unsupervised semantic manipulation approaches: vector-based disentangled representation and feature map-based disentangled representation, in terms of the complexity of the latent codes and smoothness of the reconstructed images. We conduct extensive experimental evaluations on real-world image data to demonstrate the power of our attacks for black-box classifiers. We further demonstrate the existence of a universal, image-agnostic semantic adversarial example.

摘要: 深度神经网络对对抗性攻击的脆弱性已被广泛证明(例如，对抗性示例攻击)。传统攻击执行非结构化像素级扰动来愚弄分类器。另一种方法是在潜在空间中进行微扰。然而，由于缺乏可解释性和解缠性，这种扰动很难控制。本文通过设计具有语义的结构化扰动，提出了一种更实用的对抗性攻击。我们提出的技术通过解开纠缠的潜在代码来操纵图像的语义属性。我们的技术背后的直觉是，相似域中的图像具有一些共同的但与主题无关的语义属性，例如手写数字中的线条粗细，可以双向映射到解开的潜在代码。针对潜在编码的复杂性和重构图像的平稳性，提出了两种无监督的语义处理方法：基于矢量的去纠缠表示和基于特征映射的去纠缠表示。我们在真实世界的图像数据上进行了广泛的实验评估，以展示我们对黑盒分类器的攻击的能力。我们进一步证明了一个普遍的、与图像无关的语义对抗例子的存在。



## **15. Adversarial joint attacks on legged robots**

针对腿部机器人的对抗性联合攻击 cs.RO

6 pages, 8 figures

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.10098v1)

**Authors**: Takuto Otomo, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: We address adversarial attacks on the actuators at the joints of legged robots trained by deep reinforcement learning. The vulnerability to the joint attacks can significantly impact the safety and robustness of legged robots. In this study, we demonstrate that the adversarial perturbations to the torque control signals of the actuators can significantly reduce the rewards and cause walking instability in robots. To find the adversarial torque perturbations, we develop black-box adversarial attacks, where, the adversary cannot access the neural networks trained by deep reinforcement learning. The black box attack can be applied to legged robots regardless of the architecture and algorithms of deep reinforcement learning. We employ three search methods for the black-box adversarial attacks: random search, differential evolution, and numerical gradient descent methods. In experiments with the quadruped robot Ant-v2 and the bipedal robot Humanoid-v2, in OpenAI Gym environments, we find that differential evolution can efficiently find the strongest torque perturbations among the three methods. In addition, we realize that the quadruped robot Ant-v2 is vulnerable to the adversarial perturbations, whereas the bipedal robot Humanoid-v2 is robust to the perturbations. Consequently, the joint attacks can be used for proactive diagnosis of robot walking instability.

摘要: 我们解决了对通过深度强化学习训练的腿部机器人关节处的致动器的对抗性攻击。对联合攻击的脆弱性会显著影响腿部机器人的安全性和健壮性。在这项研究中，我们证明了对执行器的力矩控制信号的对抗性扰动可以显著减少机器人的奖励并导致机器人行走不稳定。为了发现对抗性扭矩扰动，我们提出了黑盒对抗性攻击，其中，对手不能访问通过深度强化学习训练的神经网络。无论深度强化学习的体系结构和算法如何，黑盒攻击都可以应用于腿部机器人。对于黑盒对抗性攻击，我们采用了三种搜索方法：随机搜索、差分进化和数值梯度下降方法。在OpenAI健身房环境下，对四足机器人Ant-v2和两足机器人人形v2进行了实验，发现在三种方法中，差分进化方法可以有效地找到最强的扭矩扰动。此外，我们还认识到四足机器人Ant-v2容易受到对抗性扰动的影响，而两足机器人人形v2对扰动具有很强的鲁棒性。因此，联合攻击可用于机器人行走不稳定性的主动诊断。



## **16. SafeNet: Mitigating Data Poisoning Attacks on Private Machine Learning**

SafeNet：缓解针对私人机器学习的数据中毒攻击 cs.CR

**SubmitDate**: 2022-05-20    [paper-pdf](http://arxiv.org/pdf/2205.09986v1)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, the datasets used for training ML models might be under the control of an adversary mounting a data poisoning attack, and MPC prevents inspecting training sets to detect poisoning. We show that multiple MPC frameworks for private ML training are susceptible to backdoor and targeted poisoning attacks. To mitigate this, we propose SafeNet, a framework for building ensemble models in MPC with formal guarantees of robustness to data poisoning attacks. We extend the security definition of private ML training to account for poisoning and prove that our SafeNet design satisfies the definition. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models. For instance, SafeNet reduces backdoor attack success from 100% to 0% for a neural network model, while achieving 39x faster training and 36x less communication than the four-party MPC framework of Dalskov et al.

摘要: 安全多方计算(MPC)已被提出，以允许多个相互不信任的数据所有者联合训练机器学习(ML)模型。然而，用于训练ML模型的数据集可能处于发起数据中毒攻击的对手的控制之下，并且MPC阻止检查训练集以检测中毒。我们表明，用于私人ML训练的多个MPC框架容易受到后门和有针对性的中毒攻击。为了缓解这一问题，我们提出了SafeNet框架，用于在MPC中构建集成模型，并正式保证对数据中毒攻击的健壮性。我们扩展了私人ML训练的安全定义以解释中毒，并证明了我们的SafeNet设计满足该定义。我们在几个机器学习数据集和模型上展示了SafeNet的效率、准确性和对中毒的弹性。例如，对于神经网络模型，SafeNet将后门攻击成功率从100%降低到0%，同时实现了比Dalskov等人的四方MPC框架快39倍的训练和36倍的通信。



## **17. Adversarial Sample Detection for Speaker Verification by Neural Vocoders**

用于神经声码器说话人确认的对抗性样本检测 cs.SD

Accepted by ICASSP 2022

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2107.00309v4)

**Authors**: Haibin Wu, Po-chun Hsu, Ji Gao, Shanshan Zhang, Shen Huang, Jian Kang, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Automatic speaker verification (ASV), one of the most important technology for biometric identification, has been widely adopted in security-critical applications. However, ASV is seriously vulnerable to recently emerged adversarial attacks, yet effective countermeasures against them are limited. In this paper, we adopt neural vocoders to spot adversarial samples for ASV. We use the neural vocoder to re-synthesize audio and find that the difference between the ASV scores for the original and re-synthesized audio is a good indicator for discrimination between genuine and adversarial samples. This effort is, to the best of our knowledge, among the first to pursue such a technical direction for detecting time-domain adversarial samples for ASV, and hence there is a lack of established baselines for comparison. Consequently, we implement the Griffin-Lim algorithm as the detection baseline. The proposed approach achieves effective detection performance that outperforms the baselines in all the settings. We also show that the neural vocoder adopted in the detection framework is dataset-independent. Our codes will be made open-source for future works to do fair comparison.

摘要: 自动说话人验证(ASV)是生物特征识别的重要技术之一，在安全关键应用中得到了广泛的应用。然而，ASV在最近出现的对抗性攻击中非常脆弱，但针对它们的有效对策有限。在本文中，我们采用神经声码器来识别ASV的对抗性样本。我们使用神经声码器对音频进行重新合成，发现原始音频和重新合成音频的ASV分数之间的差异是区分真实和敌对样本的一个很好的指标。据我们所知，这项工作是为检测ASV的时间域对手样本而采取的最早的技术方向之一，因此缺乏用于比较的既定基线。因此，我们实现了Griffin-Lim算法作为检测基线。所提出的方法在所有设置下都取得了优于基线的有效检测性能。我们还证明了检测框架中采用的神经声码器是与数据集无关的。我们的代码将是开源的，以供将来的工作做公平的比较。



## **18. Focused Adversarial Attacks**

集中的对抗性攻击 cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09624v1)

**Authors**: Thomas Cilloni, Charles Walter, Charles Fleming

**Abstracts**: Recent advances in machine learning show that neural models are vulnerable to minimally perturbed inputs, or adversarial examples. Adversarial algorithms are optimization problems that minimize the accuracy of ML models by perturbing inputs, often using a model's loss function to craft such perturbations. State-of-the-art object detection models are characterized by very large output manifolds due to the number of possible locations and sizes of objects in an image. This leads to their outputs being sparse and optimization problems that use them incur a lot of unnecessary computation.   We propose to use a very limited subset of a model's learned manifold to compute adversarial examples. Our \textit{Focused Adversarial Attacks} (FA) algorithm identifies a small subset of sensitive regions to perform gradient-based adversarial attacks. FA is significantly faster than other gradient-based attacks when a model's manifold is sparsely activated. Also, its perturbations are more efficient than other methods under the same perturbation constraints. We evaluate FA on the COCO 2017 and Pascal VOC 2007 detection datasets.

摘要: 机器学习的最新进展表明，神经模型很容易受到最小扰动输入或对抗性例子的影响。对抗性算法是一种优化问题，它通过扰动输入来最小化ML模型的准确性，通常使用模型的损失函数来设计这种扰动。由于图像中对象的可能位置和大小的数量，最新的对象检测模型的特征在于非常大的输出流形。这导致它们的输出是稀疏的，并且使用它们的优化问题会引起大量不必要的计算。我们建议使用模型的学习流形的一个非常有限的子集来计算对抗性例子。本文提出的聚焦对抗性攻击(FA)算法识别一小部分敏感区域进行基于梯度的对抗性攻击。当模型的流形被稀疏激活时，FA比其他基于梯度的攻击要快得多。在相同的摄动约束下，它的摄动比其他方法更有效。我们在COCO 2017和Pascal VOC 2007检测数据集上评估FA。



## **19. Improving Robustness against Real-World and Worst-Case Distribution Shifts through Decision Region Quantification**

通过决策区域量化提高对真实世界和最坏情况分布漂移的稳健性 cs.LG

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09619v1)

**Authors**: Leo Schwinn, Leon Bungert, An Nguyen, René Raab, Falk Pulsmeyer, Doina Precup, Björn Eskofier, Dario Zanca

**Abstracts**: The reliability of neural networks is essential for their use in safety-critical applications. Existing approaches generally aim at improving the robustness of neural networks to either real-world distribution shifts (e.g., common corruptions and perturbations, spatial transformations, and natural adversarial examples) or worst-case distribution shifts (e.g., optimized adversarial examples). In this work, we propose the Decision Region Quantification (DRQ) algorithm to improve the robustness of any differentiable pre-trained model against both real-world and worst-case distribution shifts in the data. DRQ analyzes the robustness of local decision regions in the vicinity of a given data point to make more reliable predictions. We theoretically motivate the DRQ algorithm by showing that it effectively smooths spurious local extrema in the decision surface. Furthermore, we propose an implementation using targeted and untargeted adversarial attacks. An extensive empirical evaluation shows that DRQ increases the robustness of adversarially and non-adversarially trained models against real-world and worst-case distribution shifts on several computer vision benchmark datasets.

摘要: 神经网络的可靠性对于它们在安全关键应用中的使用至关重要。现有方法通常旨在提高神经网络对真实世界分布变化(例如，常见的破坏和扰动、空间变换和自然对抗性示例)或最坏情况分布变化(例如，优化的对抗性示例)的稳健性。在这项工作中，我们提出了决策区域量化(DRQ)算法，以提高任何可微预训练模型对真实世界和最坏情况下数据分布漂移的稳健性。DRQ分析给定数据点附近的局部决策区域的稳健性，以做出更可靠的预测。通过证明DRQ算法有效地平滑了决策面上的虚假局部极值，我们在理论上激励了DRQ算法。此外，我们提出了一种使用定向和非定向对抗性攻击的实现方案。一项广泛的经验评估表明，DRQ提高了对抗性和非对抗性训练模型在几个计算机视觉基准数据集上针对真实世界和最坏情况分布变化的稳健性。



## **20. Transferable Physical Attack against Object Detection with Separable Attention**

注意力可分离的可转移物理攻击目标检测 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09592v1)

**Authors**: Yu Zhang, Zhiqiang Gong, Yichuang Zhang, YongQian Li, Kangcheng Bin, Jiahao Qi, Wei Xue, Ping Zhong

**Abstracts**: Transferable adversarial attack is always in the spotlight since deep learning models have been demonstrated to be vulnerable to adversarial samples. However, existing physical attack methods do not pay enough attention on transferability to unseen models, thus leading to the poor performance of black-box attack.In this paper, we put forward a novel method of generating physically realizable adversarial camouflage to achieve transferable attack against detection models. More specifically, we first introduce multi-scale attention maps based on detection models to capture features of objects with various resolutions. Meanwhile, we adopt a sequence of composite transformations to obtain the averaged attention maps, which could curb model-specific noise in the attention and thus further boost transferability. Unlike the general visualization interpretation methods where model attention should be put on the foreground object as much as possible, we carry out attack on separable attention from the opposite perspective, i.e. suppressing attention of the foreground and enhancing that of the background. Consequently, transferable adversarial camouflage could be yielded efficiently with our novel attention-based loss function. Extensive comparison experiments verify the superiority of our method to state-of-the-art methods.

摘要: 可转移的对抗性攻击一直是人们关注的焦点，因为深度学习模型已被证明容易受到对抗性样本的影响。针对现有物理攻击方法对不可见模型的可转移性不够重视，导致黑盒攻击性能较差的问题，提出了一种新的生成物理可实现的对抗伪装的方法来实现对检测模型的可转移攻击。更具体地说，我们首先引入了基于检测模型的多尺度注意图来捕捉不同分辨率目标的特征。同时，我们采用一系列的复合变换来获得平均注意图，从而抑制了注意力中的特定模型噪声，从而进一步提高了注意力的可转移性。与一般的可视化解释方法将模型注意力尽可能地放在前景对象上不同，我们从相反的角度对可分离的注意进行攻击，即抑制前景的注意和增强背景的注意。因此，新的基于注意力的损失函数可以有效地产生可转移的对抗性伪装。大量的对比实验验证了该方法相对于最新方法的优越性。



## **21. On Trace of PGD-Like Adversarial Attacks**

关于类PGD对抗性攻击的踪迹 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09586v1)

**Authors**: Mo Zhou, Vishal M. Patel

**Abstracts**: Adversarial attacks pose safety and security concerns for deep learning applications. Yet largely imperceptible, a strong PGD-like attack may leave strong trace in the adversarial example. Since attack triggers the local linearity of a network, we speculate network behaves in different extents of linearity for benign examples and adversarial examples. Thus, we construct Adversarial Response Characteristics (ARC) features to reflect the model's gradient consistency around the input to indicate the extent of linearity. Under certain conditions, it shows a gradually varying pattern from benign example to adversarial example, as the later leads to Sequel Attack Effect (SAE). ARC feature can be used for informed attack detection (perturbation magnitude is known) with binary classifier, or uninformed attack detection (perturbation magnitude is unknown) with ordinal regression. Due to the uniqueness of SAE to PGD-like attacks, ARC is also capable of inferring other attack details such as loss function, or the ground-truth label as a post-processing defense. Qualitative and quantitative evaluations manifest the effectiveness of ARC feature on CIFAR-10 w/ ResNet-18 and ImageNet w/ ResNet-152 and SwinT-B-IN1K with considerable generalization among PGD-like attacks despite domain shift. Our method is intuitive, light-weighted, non-intrusive, and data-undemanding.

摘要: 对抗性攻击给深度学习应用程序带来了安全和安保问题。然而，在很大程度上潜移默化的，一次强大的类似PGD的攻击可能会在对手的例子中留下强烈的痕迹。由于攻击触发了网络的局部线性，我们推测对于良性示例和恶意示例，网络的行为具有不同程度的线性。因此，我们构造了对抗性反应特征(ARC)特征来反映模型在输入附近的梯度一致性，以指示线性程度。在一定条件下，它呈现出从良性范例到对抗性范例的渐变模式，后者会导致后续攻击效应(SAE)。圆弧特征可以用于二值分类器的知情攻击检测(扰动幅度已知)，也可以用于有序回归的不知情攻击检测(扰动幅度未知)。由于SAE到PGD类攻击的独特性，ARC还能够推断其他攻击细节，如损失函数或地面事实标签作为后处理防御。定性和定量评估表明，ARC特征在CIFAR-10 w/ResNet-18和ImageNet w/ResNet-152和Swint-B-IN1K上的有效性，在类似PGD的攻击中具有相当大的泛化能力，尽管域发生了变化。我们的方法是直观的、轻量级的、非侵入性的、不需要数据的。



## **22. Defending Against Adversarial Attacks by Energy Storage Facility**

利用储能设施防御敌意攻击 cs.CR

arXiv admin note: text overlap with arXiv:1904.06606 by other authors

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09522v1)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstracts**: Adversarial attacks on data-driven algorithms applied in pow-er system will be a new type of threat on grid security. Litera-ture has demonstrated the adversarial attack on deep-neural network can significantly misleading the load forecast of a power system. However, it is unclear how the new type of at-tack impact on the operation of grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which account for around 20 million dollars. When wind-energy penetration increases to over 40%, the 5% adver-sarial attack will inflate the generation cost by 23%. Our re-search discovers a novel approach of defending against the adversarial attack: investing on energy-storage system. All current literature focuses on developing algorithm to defending against adversarial attack. We are the first research revealing the capability of using facility in physical system to defending against the adversarial algorithm attack in a system of Internet of Thing, such as smart grid system.

摘要: 针对电力系统中应用的数据驱动算法的对抗性攻击将是一种新型的网格安全威胁。已有文献表明，对深度神经网络的敌意攻击会严重误导电力系统的负荷预测。然而，目前还不清楚这种新型的AT-TACK对电网系统的运行有何影响。在这项研究中，我们证明了对抗性算法攻击导致了显著的成本增加风险，这种风险将随着间歇性可再生能源的日益普及而加剧。在德克萨斯州，5%的对抗性攻击可以在一个季度内使总发电成本增加17%，约占2000万美元。当风能渗透率增加到40%以上时，5%的风能侵袭将使发电成本增加23%。我们的研究发现了一种防御对手攻击的新方法：投资于储能系统。目前所有的文献都集中在开发算法来防御对手攻击。我们首次揭示了在物联网系统中利用物理系统中的便利来防御对抗性算法攻击的能力，例如智能电网系统。



## **23. Enhancing the Transferability of Adversarial Examples via a Few Queries**

通过几个问题增强对抗性例句的可转移性 cs.CV

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09518v1)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Due to the vulnerability of deep neural networks, the black-box attack has drawn great attention from the community. Though transferable priors decrease the query number of the black-box query attacks in recent efforts, the average number of queries is still larger than 100, which is easily affected by the number of queries limit policy. In this work, we propose a novel method called query prior-based method to enhance the family of fast gradient sign methods and improve their attack transferability by using a few queries. Specifically, for the untargeted attack, we find that the successful attacked adversarial examples prefer to be classified as the wrong categories with higher probability by the victim model. Therefore, the weighted augmented cross-entropy loss is proposed to reduce the gradient angle between the surrogate model and the victim model for enhancing the transferability of the adversarial examples. Theoretical analysis and extensive experiments demonstrate that our method could significantly improve the transferability of gradient-based adversarial attacks on CIFAR10/100 and ImageNet and outperform the black-box query attack with the same few queries.

摘要: 由于深度神经网络的脆弱性，此次黑匣子攻击引起了社会各界的高度关注。虽然可转移先验在最近的努力中减少了黑盒查询攻击的查询数，但平均查询数仍然大于100，这很容易受到查询数限制策略的影响。在这项工作中，我们提出了一种新的方法，称为基于查询优先的方法，以增强一族快速梯度符号方法，并通过使用几个查询来提高它们的攻击可转移性。具体地说，对于非定向攻击，我们发现被攻击成功的对抗性例子更倾向于被受害者模型分类为概率更高的错误类别。因此，为了提高对抗性例子的可转移性，提出了加权增广交叉熵损失来减小代理模型和受害者模型之间的梯度角。理论分析和大量实验表明，该方法可以显著提高基于梯度的攻击在CIFAR10/100和ImageNet上的可转移性，并在相同的较少查询数下优于黑盒查询攻击。



## **24. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2204.09803v2)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 图卷积网络(GCNS)容易受到微小的敌意扰动，这是一种严重的威胁，在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，称为图通用对抗防御(GARD)。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显着提高了几个已建立的GCN对多个对手攻击的稳健性，并且远远超过了最先进的防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **25. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

多智能体强化学习中的稀疏对抗性攻击 cs.AI

**SubmitDate**: 2022-05-19    [paper-pdf](http://arxiv.org/pdf/2205.09362v1)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).

摘要: 协作多智能体强化学习(CMARL)有很多实际应用，但已有的cMARL算法训练的策略在实际应用中不够健壮。针对RL系统的对抗性攻击也有很多方法，这意味着RL系统可能会遭受对抗性攻击，但大多数方法都集中在单个代理RL上。本文提出了一种针对cMARL系统的稀疏对抗攻击。我们使用带正则化的(MA)RL来训练攻击策略。我们的实验表明，当团队中只有一个或几个代理(例如，8个代理中的1个或25个代理中的5个)在几个时间步骤(例如，总共40个时间步骤中的攻击3个)受到攻击时，由当前cMARL算法训练的策略会获得较差的性能。



## **26. Backdoor Attacks on Bayesian Neural Networks using Reverse Distribution**

基于反向分布的贝叶斯神经网络后门攻击 cs.CR

9 pages, 7 figures

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09167v1)

**Authors**: Zhixin Pan, Prabhat Mishra

**Abstracts**: Due to cost and time-to-market constraints, many industries outsource the training process of machine learning models (ML) to third-party cloud service providers, popularly known as ML-asa-Service (MLaaS). MLaaS creates opportunity for an adversary to provide users with backdoored ML models to produce incorrect predictions only in extremely rare (attacker-chosen) scenarios. Bayesian neural networks (BNN) are inherently immune against backdoor attacks since the weights are designed to be marginal distributions to quantify the uncertainty. In this paper, we propose a novel backdoor attack based on effective learning and targeted utilization of reverse distribution. This paper makes three important contributions. (1) To the best of our knowledge, this is the first backdoor attack that can effectively break the robustness of BNNs. (2) We produce reverse distributions to cancel the original distributions when the trigger is activated. (3) We propose an efficient solution for merging probability distributions in BNNs. Experimental results on diverse benchmark datasets demonstrate that our proposed attack can achieve the attack success rate (ASR) of 100%, while the ASR of the state-of-the-art attacks is lower than 60%.

摘要: 由于成本和上市时间的限制，许多行业将机器学习模型(ML)的培训过程外包给第三方云服务提供商，通常称为ML-ASA-Service(MLaaS)。MLaaS为对手创造了机会，为用户提供背道而驰的ML模型，只有在极其罕见的(攻击者选择的)情况下才能产生错误的预测。贝叶斯神经网络(BNN)天生不受后门攻击，因为权重被设计为边际分布来量化不确定性。本文提出了一种新的基于有效学习和定向利用反向分布的后门攻击方法。本文有三个重要贡献。(1)据我们所知，这是第一个可以有效破坏BNN健壮性的后门攻击。(2)当触发器被激活时，我们产生反向分布来抵消原始分布。(3)提出了一种在BNN中合并概率分布的有效解决方案。在不同基准数据集上的实验结果表明，我们提出的攻击可以达到100%的攻击成功率(ASR)，而最新的攻击的ASR低于60%。



## **27. VLC Physical Layer Security through RIS-aided Jamming Receiver for 6G Wireless Networks**

基于RIS辅助干扰接收机的6G无线网络VLC物理层安全 cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.09026v1)

**Authors**: Simone Soderi, Alessandro Brighente, Federico Turrin, Mauro Conti

**Abstracts**: Visible Light Communication (VLC) is one the most promising enabling technology for future 6G networks to overcome Radio-Frequency (RF)-based communication limitations thanks to a broader bandwidth, higher data rate, and greater efficiency. However, from the security perspective, VLCs suffer from all known wireless communication security threats (e.g., eavesdropping and integrity attacks). For this reason, security researchers are proposing innovative Physical Layer Security (PLS) solutions to protect such communication. Among the different solutions, the novel Reflective Intelligent Surface (RIS) technology coupled with VLCs has been successfully demonstrated in recent work to improve the VLC communication capacity. However, to date, the literature still lacks analysis and solutions to show the PLS capability of RIS-based VLC communication. In this paper, we combine watermarking and jamming primitives through the Watermark Blind Physical Layer Security (WBPLSec) algorithm to secure VLC communication at the physical layer. Our solution leverages RIS technology to improve the security properties of the communication. By using an optimization framework, we can calculate RIS phases to maximize the WBPLSec jamming interference schema over a predefined area in the room. In particular, compared to a scenario without RIS, our solution improves the performance in terms of secrecy capacity without any assumption about the adversary's location. We validate through numerical evaluations the positive impact of RIS-aided solution to increase the secrecy capacity of the legitimate jamming receiver in a VLC indoor scenario. Our results show that the introduction of RIS technology extends the area where secure communication occurs and that by increasing the number of RIS elements the outage probability decreases.

摘要: 可见光通信(VLC)是未来6G网络最有前途的使能技术之一，可以克服基于射频(RF)的通信限制，因为它具有更宽的带宽、更高的数据速率和更高的效率。然而，从安全的角度来看，VLC受到所有已知的无线通信安全威胁(例如，窃听和完整性攻击)。为此，安全研究人员提出了创新的物理层安全(PLS)解决方案来保护此类通信。在不同的解决方案中，新型的反射智能表面(RIS)技术与VLC相结合已经在最近的工作中被成功地展示出来，以提高VLC的通信容量。然而，到目前为止，文献仍然缺乏分析和解决方案来展示基于RIS的VLC通信的偏最小二乘能力。在本文中，我们通过水印盲物理层安全(WBPLSec)算法将水印和干扰基元相结合来保护物理层的VLC通信。我们的解决方案利用RIS技术来提高通信的安全属性。通过使用优化框架，我们可以计算RIS相位，以最大化房间中预定义区域内的WBPLSec干扰方案。特别是，与没有RIS的场景相比，我们的方案在保密能力方面提高了性能，而不需要假设对手的位置。我们通过数值评估验证了RIS辅助解决方案对提高VLC室内场景中合法干扰接收机的保密容量的积极影响。我们的结果表明，RIS技术的引入扩展了安全通信发生的区域，并且随着RIS单元数量的增加，中断概率降低。



## **28. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

增量对抗性(IMA)训练提高神经网络对抗性鲁棒性 cs.CV

13 pages

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2005.09147v8)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial noises. By adding adversarial noises to training samples, adversarial training can improve the model's robustness against adversarial noises. However, adversarial training samples with excessive noises can harm standard accuracy, which may be unacceptable for many medical image analysis applications. This issue has been termed the trade-off between standard accuracy and adversarial robustness. In this paper, we hypothesize that this issue may be alleviated if the adversarial samples for training are placed right on the decision boundaries. Based on this hypothesis, we design an adaptive adversarial training method, named IMA. For each individual training sample, IMA makes a sample-wise estimation of the upper bound of the adversarial perturbation. In the training process, each of the sample-wise adversarial perturbations is gradually increased to match the margin. Once an equilibrium state is reached, the adversarial perturbations will stop increasing. IMA is evaluated on publicly available datasets under two popular adversarial attacks, PGD and IFGSM. The results show that: (1) IMA significantly improves adversarial robustness of DNN classifiers, which achieves the state-of-the-art performance; (2) IMA has a minimal reduction in clean accuracy among all competing defense methods; (3) IMA can be applied to pretrained models to reduce time cost; (4) IMA can be applied to the state-of-the-art medical image segmentation networks, with outstanding performance. We hope our work may help to lift the trade-off between adversarial robustness and clean accuracy and facilitate the development of robust applications in the medical field. The source code will be released when this paper is published.

摘要: 深度神经网络(DNN)很容易受到对抗性噪声的影响。通过在训练样本中加入对抗性噪声，对抗性训练可以提高模型对对抗性噪声的鲁棒性。然而，含有过多噪声的对抗性训练样本可能会损害标准精度，这对于许多医学图像分析应用来说可能是不可接受的。这个问题被称为标准准确性和对抗性稳健性之间的权衡。在本文中，我们假设，如果用于训练的对手样本正确地放置在决策边界上，这个问题可能会得到缓解。基于这一假设，我们设计了一种自适应对抗性训练方法IMA。对于每个单独的训练样本，IMA逐个估计对抗性扰动的上界。在训练过程中，每个样本对抗性扰动都会逐渐增加，以匹配差值。一旦达到均衡状态，对抗性扰动将停止增加。IMA是在两种流行的对抗性攻击PGD和IFGSM下基于公开可用的数据集进行评估的。结果表明：(1)IMA显著提高了DNN分类器的对抗健壮性，达到了最先进的性能；(2)IMA在所有竞争防御方法中干净准确率的降幅最小；(3)IMA可以应用于预先训练的模型，减少了时间开销；(4)IMA可以应用于最先进的医学图像分割网络，性能优异。我们希望我们的工作可以帮助消除对抗性健壮性和干净准确性之间的权衡，并促进医疗领域健壮性应用的发展。源代码将在这篇论文发表时发布。



## **29. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

属性遗忘：一种防御属性推理攻击的策略 cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08821v1)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.

摘要: 在机器学习模型的训练过程中，它们可能存储或“学习”比预测或分类任务实际需要的更多关于训练数据的信息。这被属性推理攻击所利用，该属性推理攻击的目的是从给定模型的训练数据中提取统计属性，而不访问训练数据本身。这些属性可以包括用于识别相机型号的图片质量、用于揭示产品目标受众的年龄分布、或用于改进计算机网络中的恶意软件攻击的所包括的主机类型。当攻击者有权访问所有模型参数时，即在白盒情况下，此攻击尤其准确。通过防御此类攻击，模型所有者能够确保他们的训练数据、相关属性以及他们的知识产权是保密的，即使他们故意共享他们的模型，例如协作训练，或者如果模型被泄露。在本文中，我们引入了属性遗忘，这是一种有效的防御白盒属性推理攻击的机制，独立于训练数据类型、模型任务或属性数量。属性遗忘通过系统地改变目标模型的训练权重和偏差来减轻属性推断攻击，使得对手无法提取所选的属性。我们在三个不同的数据集上经验地评估了属性遗忘，包括表格和图像数据，以及两种类型的人工神经网络。我们的结果表明，属性忘却在保护机器学习模型免受属性推理攻击方面是有效和可靠的，并且具有良好的隐私效用权衡。此外，我们的方法表明，该机制也有效地忘却了多个属性。



## **30. Passive Defense Against 3D Adversarial Point Clouds Through the Lens of 3D Steganalysis**

基于3D隐写分析镜头的3D对抗点云被动防御 cs.MM

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08738v1)

**Authors**: Jiahao Zhu

**Abstracts**: Nowadays, 3D data plays an indelible role in the computer vision field. However, extensive studies have proved that deep neural networks (DNNs) fed with 3D data, such as point clouds, are susceptible to adversarial examples, which aim to misguide DNNs and might bring immeasurable losses. Currently, 3D adversarial point clouds are chiefly generated in three fashions, i.e., point shifting, point adding, and point dropping. These point manipulations would modify geometrical properties and local correlations of benign point clouds more or less. Motivated by this basic fact, we propose to defend such adversarial examples with the aid of 3D steganalysis techniques. Specifically, we first introduce an adversarial attack and defense model adapted from the celebrated Prisoners' Problem in steganography to help us comprehend 3D adversarial attack and defense more generally. Then we rethink two significant but vague concepts in the field of adversarial example, namely, active defense and passive defense, from the perspective of steganalysis. Most importantly, we design a 3D adversarial point cloud detector through the lens of 3D steganalysis. Our detector is double-blind, that is to say, it does not rely on the exact knowledge of the adversarial attack means and victim models. To enable the detector to effectively detect malicious point clouds, we craft a 64-D discriminant feature set, including features related to first-order and second-order local descriptions of point clouds. To our knowledge, this work is the first to apply 3D steganalysis to 3D adversarial example defense. Extensive experimental results demonstrate that the proposed 3D adversarial point cloud detector can achieve good detection performance on multiple types of 3D adversarial point clouds.

摘要: 如今，3D数据在计算机视觉领域发挥着不可磨灭的作用。然而，大量的研究已经证明，以点云等三维数据为基础的深度神经网络(DNN)很容易受到敌意例子的影响，这些例子旨在误导DNN，并可能带来不可估量的损失。目前，三维对抗性点云的生成主要有三种方式，即点移位、点加点和点删除。这些点操作会或多或少地改变良性点云的几何性质和局部相关性。在这一基本事实的推动下，我们建议借助3D隐写分析技术来为这些对抗性例子进行辩护。具体地说，我们首先介绍了一种改编自隐写术中著名囚犯问题的对抗性攻防模型，以帮助我们更全面地理解3D对抗性攻防。然后，我们从隐写分析的角度重新思考了对抗性例证领域中两个重要而模糊的概念，即主动防御和被动防御。最重要的是，我们通过3D隐写分析的镜头设计了一个3D对抗点云检测器。我们的检测器是双盲的，也就是说，它不依赖于对抗性攻击手段和受害者模型的准确知识。为了使检测器能够有效地检测恶意点云，我们构造了一个64维判别特征集，包括与点云的一阶和二阶局部描述相关的特征。据我们所知，这项工作是首次将3D隐写分析应用于3D对抗实例防御。大量的实验结果表明，本文提出的三维对抗性点云检测器对多种类型的三维对抗性点云具有较好的检测性能。



## **31. Policy Distillation with Selective Input Gradient Regularization for Efficient Interpretability**

用于有效解释的选择性输入梯度正则化策略精馏 cs.LG

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2205.08685v1)

**Authors**: Jinwei Xing, Takashi Nagata, Xinyun Zou, Emre Neftci, Jeffrey L. Krichmar

**Abstracts**: Although deep Reinforcement Learning (RL) has proven successful in a wide range of tasks, one challenge it faces is interpretability when applied to real-world problems. Saliency maps are frequently used to provide interpretability for deep neural networks. However, in the RL domain, existing saliency map approaches are either computationally expensive and thus cannot satisfy the real-time requirement of real-world scenarios or cannot produce interpretable saliency maps for RL policies. In this work, we propose an approach of Distillation with selective Input Gradient Regularization (DIGR) which uses policy distillation and input gradient regularization to produce new policies that achieve both high interpretability and computation efficiency in generating saliency maps. Our approach is also found to improve the robustness of RL policies to multiple adversarial attacks. We conduct experiments on three tasks, MiniGrid (Fetch Object), Atari (Breakout) and CARLA Autonomous Driving, to demonstrate the importance and effectiveness of our approach.

摘要: 虽然深度强化学习(RL)已被证明在广泛的任务中取得了成功，但它面临的一个挑战是当应用于现实世界的问题时的可解释性。显著图经常被用来为深度神经网络提供可解释性。然而，在RL领域，现有的显著图方法要么计算量大，不能满足现实场景的实时要求，要么无法为RL策略生成可解释的显著图。在这项工作中，我们提出了一种选择输入梯度正则化的蒸馏方法(DIGR)，它使用策略精馏和输入梯度正则化来产生新的策略，在生成显著图的过程中实现了高的可解释性和计算效率。我们的方法也被发现提高了RL策略对多个对手攻击的健壮性。通过对MinigRid(取回对象)、Atari(突破)和Carla自主驾驶三个任务的实验，验证了该方法的重要性和有效性。



## **32. Longest Chain Consensus Under Bandwidth Constraint**

带宽约束下的最长链共识 cs.CR

**SubmitDate**: 2022-05-18    [paper-pdf](http://arxiv.org/pdf/2111.12332v3)

**Authors**: Joachim Neu, Srivatsan Sridhar, Lei Yang, David Tse, Mohammad Alizadeh

**Abstracts**: Spamming attacks are a serious concern for consensus protocols, as witnessed by recent outages of a major blockchain, Solana. They cause congestion and excessive message delays in a real network due to its bandwidth constraints. In contrast, longest chain (LC), an important family of consensus protocols, has previously only been proven secure assuming an idealized network model in which all messages are delivered within bounded delay. This model-reality mismatch is further aggravated for Proof-of-Stake (PoS) LC where the adversary can spam the network with equivocating blocks. Hence, we extend the network model to capture bandwidth constraints, under which nodes now need to choose carefully which blocks to spend their limited download budget on. To illustrate this point, we show that 'download along the longest header chain', a natural download rule for Proof-of-Work (PoW) LC, is insecure for PoS LC. We propose a simple rule 'download towards the freshest block', formalize two common heuristics 'not downloading equivocations' and 'blocklisting', and prove in a unified framework that PoS LC with any one of these download rules is secure in bandwidth-constrained networks. In experiments, we validate our claims and showcase the behavior of these download rules under attack. By composing multiple instances of a PoS LC protocol with a suitable download rule in parallel, we obtain a PoS consensus protocol that achieves a constant fraction of the network's throughput limit even under worst-case adversarial strategies.

摘要: 垃圾邮件攻击是共识协议的一个严重问题，主要区块链Solana最近的中断就证明了这一点。由于带宽的限制，它们会在实际网络中造成拥塞和过多的消息延迟。相比之下，最长链(LC)是一类重要的共识协议，以前只有在假设所有消息在有限延迟内传递的理想化网络模型下才被证明是安全的。这种模型与现实的不匹配在风险证明(POS)LC中进一步加剧，在这种LC中，对手可以使用模棱两可的块向网络发送垃圾邮件。因此，我们扩展了网络模型以捕获带宽限制，在这种情况下，节点现在需要仔细选择将其有限的下载预算花费在哪些块上。为了说明这一点，我们证明了用于工作证明(PoW)LC的自然下载规则--“沿着最长的报头链下载”对于PoS LC是不安全的。我们提出了一个简单的规则“下载到最新的块”，形式化了两个常见的启发式规则“不下载歧义”和“块列表”，并在一个统一的框架中证明了在带宽受限的网络中，使用这些下载规则中的任何一个的POS LC是安全的。在实验中，我们验证了我们的说法，并展示了这些下载规则在攻击下的行为。通过将PoS LC协议的多个实例与合适的下载规则并行组合，我们得到了一个PoS共识协议，该协议即使在最坏的对抗策略下也能达到网络吞吐量极限的恒定分数。



## **33. An Integrated Approach for Energy Efficient Handover and Key Distribution Protocol for Secure NC-enabled Small Cells**

一种集成的安全NC小蜂窝切换和密钥分发协议 cs.NI

Preprint of the paper accepted at Computer Networks

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08641v1)

**Authors**: Vipindev Adat Vasudevan, Muhammad Tayyab, George P. Koudouridis, Xavier Gelabert, Ilias Politis

**Abstracts**: Future wireless networks must serve dense mobile networks with high data rates, keeping energy requirements to a possible minimum. The small cell-based network architecture and device-to-device (D2D) communication are already being considered part of 5G networks and beyond. In such environments, network coding (NC) can be employed to achieve both higher throughput and energy efficiency. However, NC-enabled systems need to address security challenges specific to NC, such as pollution attacks. All integrity schemes against pollution attacks generally require proper key distribution and management to ensure security in a mobile environment. Additionally, the mobility requirements in small cell environments are more challenging and demanding in terms of signaling overhead. This paper proposes a blockchain-assisted key distribution protocol tailored for MAC-based integrity schemes, which combined with an uplink reference signal (UL RS) handover mechanism, enables energy efficient secure NC. The performance analysis of the protocol during handover scenarios indicates its suitability for ensuring high level of security against pollution attacks in dense small cell environments with multiple adversaries being present. Furthermore, the proposed scheme achieves lower bandwidth and signaling overhead during handover compared to legacy schemes and the signaling cost reduces significantly as the communication progresses, thus enhancing the network's cumulative energy efficiency.

摘要: 未来的无线网络必须以高数据速率服务于密集的移动网络，将能源需求保持在尽可能低的水平。基于小蜂窝的网络架构和设备到设备(D2D)通信已经被认为是5G网络和更远的网络的一部分。在这样的环境中，可以使用网络编码(NC)来实现更高的吞吐量和能量效率。然而，支持NC的系统需要解决特定于NC的安全挑战，例如污染攻击。所有针对污染攻击的完整性方案通常都需要适当的密钥分发和管理，以确保移动环境中的安全。此外，小蜂窝环境中的移动性要求在信令开销方面更具挑战性和要求。提出了一种适合基于MAC的完整性方案的区块链辅助密钥分发协议，该协议结合上行参考信号(UL RS)切换机制，实现了能量高效的安全NC。在切换场景中的性能分析表明，该协议适合在密集的小蜂窝环境中，在存在多个对手的情况下，确保针对污染攻击的高级别安全。此外，与传统方案相比，该方案在切换过程中获得了更低的带宽和信令开销，并且随着通信的进行，信令开销显著降低，从而提高了网络的累积能量效率。



## **34. Hierarchical Distribution-Aware Testing of Deep Learning**

深度学习的分层分布感知测试 cs.SE

Under Review

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08589v1)

**Authors**: Wei Huang, Xingyu Zhao, Alec Banks, Victoria Cox, Xiaowei Huang

**Abstracts**: With its growing use in safety/security-critical applications, Deep Learning (DL) has raised increasing concerns regarding its dependability. In particular, DL has a notorious problem of lacking robustness. Despite recent efforts made in detecting Adversarial Examples (AEs) via state-of-the-art attacking and testing methods, they are normally input distribution agnostic and/or disregard the perception quality of AEs. Consequently, the detected AEs are irrelevant inputs in the application context or unnatural/unrealistic that can be easily noticed by humans. This may lead to a limited effect on improving the DL model's dependability, as the testing budget is likely to be wasted on detecting AEs that are encountered very rarely in its real-life operations. In this paper, we propose a new robustness testing approach for detecting AEs that considers both the input distribution and the perceptual quality of inputs. The two considerations are encoded by a novel hierarchical mechanism. First, at the feature level, the input data distribution is extracted and approximated by data compression techniques and probability density estimators. Such quantified feature level distribution, together with indicators that are highly correlated with local robustness, are considered in selecting test seeds. Given a test seed, we then develop a two-step genetic algorithm for local test case generation at the pixel level, in which two fitness functions work alternatively to control the quality of detected AEs. Finally, extensive experiments confirm that our holistic approach considering hierarchical distributions at feature and pixel levels is superior to state-of-the-arts that either disregard any input distribution or only consider a single (non-hierarchical) distribution, in terms of not only the quality of detected AEs but also improving the overall robustness of the DL model under testing.

摘要: 随着深度学习在安全/安保关键应用中的应用越来越多，人们越来越关注它的可靠性。特别是，DL有一个臭名昭著的问题，即缺乏健壮性。尽管最近通过最先进的攻击和测试方法来检测对抗性实例(AEs)，但它们通常是输入分布不可知的和/或忽略了AEs的感知质量。因此，检测到的AE是应用上下文中不相关的输入，或者是人类容易注意到的不自然/不现实的输入。这可能会对提高DL模型的可靠性产生有限的影响，因为测试预算很可能被浪费在检测在其实际操作中很少遇到的AE上。在本文中，我们提出了一种新的稳健性测试方法，该方法同时考虑了输入的分布和输入的感知质量。这两个考虑因素通过一种新的分层机制进行编码。首先，在特征层，利用数据压缩技术和概率密度估计器提取输入数据的分布，并对其进行近似。在选择测试种子时，考虑了这种量化的特征级别分布，以及与局部稳健性高度相关的指标。在给定测试种子的情况下，我们提出了一种在像素级局部生成测试用例的两步遗传算法，其中两个适应度函数交替工作来控制检测到的测试用例的质量。最后，大量的实验证实，我们的整体方法在考虑特征和像素级别的分层分布方面优于现有的忽略任何输入分布或只考虑单个(非分层)分布的方法，不仅在检测到的AE的质量方面，而且在提高测试下的DL模型的整体稳健性方面。



## **35. F3B: A Low-Latency Commit-and-Reveal Architecture to Mitigate Blockchain Front-Running**

F3B：一种降低区块链前运行的低延迟提交与揭示体系结构 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08529v1)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Vero Estrada-Galinanes, Bryan Ford

**Abstracts**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the cryptocurrency space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants$\unicode{x2013}$estimated at \$280M each month$\unicode{x2013}$and endangers the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture to address front-running attacks by relying on a commit-and-reveal scheme where the contents of transactions are encrypted and later revealed by a decentralized secret-management committee once the underlying consensus layer has committed the transaction. F3B mitigates front-running attacks because an adversary can no longer read the content of a transaction before commitment, thus preventing the adversary from benefiting from advance knowledge of pending transactions. We design F3B to be agnostic to the underlying consensus algorithm and compatible with legacy smart contracts by addressing front-running at the blockchain architecture level. Unlike existing commit-and-reveal approaches, F3B only requires writing data onto the underlying blockchain once, establishing a significant overhead reduction. An exploration of F3B shows that with a secret-management committee consisting of 8 and 128 members, F3B presents between 0.1 and 1.8 seconds of transaction-processing latency, respectively.

摘要: 自去中心化金融出现以来，受益于待完成交易的先进知识的前沿攻击在加密货币领域激增。领跑给诚实的参与者造成了毁灭性的损失，估计每月损失2.8亿美元，并危及生态系统的公平性。我们提出了Flash冷冻Flash Boys(F3B)，这是一种区块链架构，通过依赖提交并披露方案来应对前沿攻击，其中交易的内容被加密，一旦底层共识层提交交易，随后由分散的秘密管理委员会披露。F3B减轻了前置攻击，因为对手在提交之前不能再读取交易的内容，从而防止对手受益于对未决交易的预先了解。我们将F3B设计为与底层共识算法无关，并通过在区块链架构级别解决先期运行问题与传统智能合同兼容。与现有的提交和揭示方法不同，F3B只需将数据写入底层区块链一次，从而显著降低了开销。对F3B的研究表明，对于由8名和128名成员组成的秘密管理委员会，F3B的事务处理延迟分别在0.1秒到1.8秒之间。



## **36. On the Privacy of Decentralized Machine Learning**

关于分散式机器学习的隐私性 cs.CR

17 pages

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08443v1)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstracts**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at circumventing the main limitations of federated learning. We identify the decentralized learning properties that affect users' privacy and we introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantages over more practical approaches such as federated learning. Rather, it tends to degrade users' privacy by increasing the attack surface and enabling any user in the system to perform powerful privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also reveal that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require abandoning any possible advantage over the federated setup, completely defeating the objective of the decentralized approach.

摘要: 在这项工作中，我们进行了第一次，深入的，隐私分析的分散学习--一个合作的机器学习框架，旨在绕过联邦学习的主要限制。我们识别了影响用户隐私的分散学习属性，并针对被动和主动的分散攻击引入了一套新的攻击。我们证明，与去中心化学习提出者所声称的相反，去中心化学习并不比联邦学习等更实用的方法提供任何安全优势。相反，它往往会通过增加攻击面来降低用户的隐私，使系统中的任何用户都可以执行强大的隐私攻击，如梯度反转，甚至获得对诚实用户的本地模型的完全控制。我们还揭示，考虑到保护措施的最新水平，去中心化学习的隐私保护配置要求放弃任何可能的优势，而不是联邦设置，完全违背了去中心化方法的目标。



## **37. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

你还能看到我吗？：在端到端加密通道上重建机器人操作 cs.CR

13 pages, 7 figures, poster presented at wisec'22

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08426v1)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around \textasciitilde60\% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.

摘要: 互联机器人在工业4.0中扮演着关键角色，为许多工业工作流程提供自动化和更高的效率。不幸的是，这些机器人可能会将有关这些操作工作流程的敏感信息泄露给远程对手。虽然在这种情况下有使用端到端加密进行数据传输的规定，但被动攻击者完全有可能对正在执行的整个工作流程进行指纹识别和重建--建立对设施如何运行的理解。在本文中，我们调查远程攻击者是否能够准确地识别机器人的运动并最终重建操作工作流。使用神经网络方法对流量进行分析，我们发现可以预测TLS加密的移动，精度约为60%，在现实网络条件下提高到接近完美的精度。此外，我们还发现攻击者可以成功地重构仓储工作流。归根结底，简单地采用最佳网络安全实践显然不足以阻止即使是弱小的(被动的)对手。



## **38. Bankrupting DoS Attackers Despite Uncertainty**

尽管存在不确定性，但仍使DoS攻击者破产 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08287v1)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as economic denial-of-sustainability (EDoS), where the cost for defending a service is higher than that of attacking.   A natural approach for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior approaches with security guarantees do not account for the cost on-demand provisioning.   Another valuable defensive tool is to use a classifier in order to discern good jobs from a legitimate client, versus bad jobs from the adversary. However, while useful, uncertainty arises from classification error, which still allows bad jobs to consume server resources. Thus, classification is not a solution by itself.   Here, we propose an EDoS defense, RootDef, that leverages both RB and classification, while accounting for both the costs of resource burning and on-demand provisioning. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $\tilde{O}( \sqrt{B\,g} + B^{2/3} + g)$, where $g$ is the number of good jobs and $\tilde{O}$ refers to hidden logarithmic factors in the total number of jobs $n$. Notably, for large $B$ relative to $g$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound showing that RootDef has total costs that are asymptotically tight up to logarithmic factors in $n$.

摘要: 云中的按需配置允许服务在遭受大规模拒绝服务(DoS)攻击时保持可用。不幸的是，按需配置的成本很高，必须权衡对手所产生的成本。这导致了最近一种被称为经济拒绝可持续性(EDOS)的威胁，在这种威胁下，防御服务的成本高于攻击。打击EDO的一个自然方法是通过资源燃烧(RB)来施加成本。在这里，在提供服务之前，客户端必须可验证地消耗资源--例如，通过解决计算挑战。然而，以前的具有安全保证的方法不考虑按需供应的成本。另一个有价值的防御工具是使用分类器，以便区分合法客户的好工作，而不是对手的坏工作。但是，尽管分类错误很有用，但不确定性源于分类错误，分类错误仍然允许不良作业消耗服务器资源。因此，分类本身并不是一个解决方案。在这里，我们提出了EDOS防御措施RootDef，它利用RB和分类，同时考虑了资源消耗和按需配置的成本。具体地说，对于花费$B$资源进行攻击的对手，防御的总成本为$\tide{O}(\sqrt{B\，g}+B^{2/3}+g)$，其中$g$是好工作的数目，$\tide{O}$是工作总数$n$中的隐藏对数因子。值得注意的是，对于较大的$B$相对于$G$，对手具有更高的成本，这意味着该算法具有经济优势。最后，我们证明了一个下界，证明了RootDef的总成本是渐近紧到$n$的对数因子的。



## **39. How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy**

如何不处理密钥：对FIDO验证器隐私的计时攻击 cs.CR

to be published in the 22nd Privacy Enhancing Technologies Symposium  (PETS 2022)

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08071v1)

**Authors**: Michal Kepkowski, Lucjan Hanzlik, Ian Wood, Mohamed Ali Kaafar

**Abstracts**: This paper presents a timing attack on the FIDO2 (Fast IDentity Online) authentication protocol that allows attackers to link user accounts stored in vulnerable authenticators, a serious privacy concern. FIDO2 is a new standard specified by the FIDO industry alliance for secure token online authentication. It complements the W3C WebAuthn specification by providing means to use a USB token or other authenticator as a second factor during the authentication process. From a cryptographic perspective, the protocol is a simple challenge-response where the elliptic curve digital signature algorithm is used to sign challenges. To protect the privacy of the user the token uses unique key pairs per service. To accommodate for small memory, tokens use various techniques that make use of a special parameter called a key handle sent by the service to the token. We identify and analyse a vulnerability in the way the processing of key handles is implemented that allows attackers to remotely link user accounts on multiple services. We show that for vulnerable authenticators there is a difference between the time it takes to process a key handle for a different service but correct authenticator, and for a different authenticator but correct service. This difference can be used to perform a timing attack allowing an adversary to link user's accounts across services. We present several real world examples of adversaries that are in a position to execute our attack and can benefit from linking accounts. We found that two of the eight hardware authenticators we tested were vulnerable despite FIDO level 1 certification. This vulnerability cannot be easily mitigated on authenticators because, for security reasons, they usually do not allow firmware updates. In addition, we show that due to the way existing browsers implement the WebAuthn standard, the attack can be executed remotely.

摘要: 提出了一种对FIDO2(Fast Identity Online)认证协议的计时攻击，使得攻击者能够链接存储在易受攻击的认证器中的用户帐户，这是一个严重的隐私问题。FIDO2是由FIDO行业联盟为安全令牌在线身份验证指定的新标准。它补充了W3CWebAuthn规范，提供了在身份验证过程中使用USB令牌或其他身份验证器作为第二因素的方法。从密码学的角度来看，该协议是一个简单的挑战-响应协议，其中使用椭圆曲线数字签名算法来签署挑战。为了保护用户的隐私，令牌对每个服务使用唯一的密钥对。为了适应较小的内存，令牌使用各种技术，这些技术利用由服务发送到令牌的称为密钥句柄的特殊参数。我们发现并分析了密钥句柄处理实现方式中的一个漏洞，该漏洞允许攻击者远程链接多个服务上的用户帐户。我们证明，对于易受攻击的验证器，处理不同服务但正确的验证器的密钥句柄所花费的时间与处理不同的验证器但正确的服务的密钥句柄所花费的时间是不同的。这种差异可用于执行计时攻击，从而允许对手跨服务链接用户帐户。我们提供了几个真实世界的例子，这些对手处于执行我们的攻击的位置，并可以从链接帐户中受益。我们发现，尽管通过了FIDO级别1认证，但我们测试的八个硬件验证器中有两个容易受到攻击。在验证器上无法轻松缓解此漏洞，因为出于安全原因，验证器通常不允许固件更新。此外，我们还表明，由于现有浏览器实现WebAuthn标准的方式，攻击可以远程执行。



## **40. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

联合学习中抵抗语音情感识别属性推理攻击的用户级差分隐私 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2204.02500v2)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.

摘要: 许多现有的隐私增强型语音情感识别(SER)框架专注于通过集中式机器学习设置中的对抗性训练来扰乱原始语音数据。然而，这种隐私保护方案可能会失败，因为攻击者仍然可以访问受干扰的数据。近年来，分布式学习算法，特别是联邦学习(FL)算法在机器学习应用中保护隐私得到了广泛的应用。虽然FL通过将数据保存在本地设备上来提供良好的直觉来保护隐私，但先前的工作表明，使用FL训练的SER系统可以实现隐私攻击，例如属性推理攻击。在这项工作中，我们建议评估用户级差异隐私(UDP)在缓解FL中SER系统的隐私泄漏方面的作用。UDP通过隐私参数$\epsilon$和$\Delta$提供理论上的隐私保证。实验结果表明，UDP协议在保持SER系统可用性的同时，有效地减少了属性信息泄露，且攻击者只需访问一次模型更新。然而，当FL系统向对手泄露更多的模型更新时，UDP的效率会受到影响。我们将代码公开，以便在https://github.com/usc-sail/fed-ser-leakage.中重现结果



## **41. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

RoVISQ：通过对基于深度学习的视频压缩进行对抗性攻击来降低视频服务质量 cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2203.10183v2)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion (R-D) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to 2.4 times while achieving over 90% attack success rate on a downstream classifier.

摘要: 视频压缩在视频流和分类系统中起着至关重要的作用，它在给定的带宽预算下最大化最终用户的体验质量(QOE)。本文首次对基于深度学习的视频压缩和下行分类系统的敌意攻击进行了系统的研究。我们的攻击框架RoVISQ通过操纵视频压缩模型的率失真(R-D)关系来实现以下一个或两个目标：(1)增加网络带宽；(2)降低最终用户的视频质量。我们进一步制定了针对下游视频分类服务的定向和非定向攻击的新目标。最后，我们设计了一种输入不变的扰动，该扰动普遍地扰乱了视频压缩和分类系统的实时。与之前提出的针对视频分类的攻击不同，我们的对抗性扰动最先经受住了压缩。我们经验地展示了RoVISQ攻击对各种防御措施的弹性，即对抗性训练、视频去噪和JPEG压缩。我们在不同视频数据集上的大量实验结果表明，RoVISQ攻击使峰值信噪比下降5.6dB，比特率下降2.4倍，同时在下游分类器上获得90%以上的攻击成功率。



## **42. Classification Auto-Encoder based Detector against Diverse Data Poisoning Attacks**

基于分类自动编码器的抗多种数据中毒攻击检测器 cs.LG

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2108.04206v2)

**Authors**: Fereshteh Razmi, Li Xiong

**Abstracts**: Poisoning attacks are a category of adversarial machine learning threats in which an adversary attempts to subvert the outcome of the machine learning systems by injecting crafted data into training data set, thus increasing the machine learning model's test error. The adversary can tamper with the data feature space, data labels, or both, each leading to a different attack strategy with different strengths. Various detection approaches have recently emerged, each focusing on one attack strategy. The Achilles heel of many of these detection approaches is their dependence on having access to a clean, untampered data set. In this paper, we propose CAE, a Classification Auto-Encoder based detector against diverse poisoned data. CAE can detect all forms of poisoning attacks using a combination of reconstruction and classification errors without having any prior knowledge of the attack strategy. We show that an enhanced version of CAE (called CAE+) does not have to employ a clean data set to train the defense model. Our experimental results on three real datasets MNIST, Fashion-MNIST and CIFAR demonstrate that our proposed method can maintain its functionality under up to 30% contaminated data and help the defended SVM classifier to regain its best accuracy.

摘要: 中毒攻击是一种对抗性机器学习威胁，对手试图通过将精心设计的数据注入训练数据集来颠覆机器学习系统的结果，从而增加机器学习模型的测试误差。对手可以篡改数据特征空间、数据标签或两者，每一种都会导致具有不同强度的不同攻击策略。最近出现了各种检测方法，每种方法都专注于一种攻击策略。许多这些检测方法的致命弱点是它们依赖于能够访问干净、未经篡改的数据集。在本文中，我们提出了一种针对不同有毒数据的基于分类自动编码器的检测器CAE。CAE可以使用重构和分类错误的组合来检测所有形式的中毒攻击，而不需要事先了解攻击策略。我们证明了CAE的增强版本(称为CAE+)不必使用干净的数据集来训练防御模型。我们在三个真实数据集MNIST、Fashion-MNIST和CIFAR上的实验结果表明，我们的方法可以在高达30%的污染数据下保持其功能，并帮助被防御的支持向量机分类器恢复其最佳精度。



## **43. Transferability of Adversarial Attacks on Synthetic Speech Detection**

合成语音检测中对抗性攻击的可转移性 cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.

摘要: 合成语音检测是音频安全领域的重要研究课题之一。与此同时，深度神经网络很容易受到敌意攻击。因此，我们建立了一个综合的基准来评估对抗性攻击对合成语音检测任务的可转移性。具体地说，我们试图研究：1)对抗性攻击在不同特征之间的可转移性。2)不同特征提取超参数对对抗性攻击可转移性的影响。3)截断或自填充操作对对抗性攻击可转移性的影响。通过这些分析，我们总结了合成语音检测器的弱点和对抗性攻击的可转移性行为，为未来的研究提供了见解。欲了解更多详情，请访问https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.。



## **44. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.01287v2)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **45. SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks**

SGBA：针对深度神经网络的隐形替罪羊后门攻击 cs.CR

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2104.01026v3)

**Authors**: Ying He, Zhili Shen, Chang Xia, Jingyu Hua, Wei Tong, Sheng Zhong

**Abstracts**: Outsourced deep neural networks have been demonstrated to suffer from patch-based trojan attacks, in which an adversary poisons the training sets to inject a backdoor in the obtained model so that regular inputs can be still labeled correctly while those carrying a specific trigger are falsely given a target label. Due to the severity of such attacks, many backdoor detection and containment systems have recently, been proposed for deep neural networks. One major category among them are various model inspection schemes, which hope to detect backdoors before deploying models from non-trusted third-parties. In this paper, we show that such state-of-the-art schemes can be defeated by a so-called Scapegoat Backdoor Attack, which introduces a benign scapegoat trigger in data poisoning to prevent the defender from reversing the real abnormal trigger. In addition, it confines the values of network parameters within the same variances of those from clean model during training, which further significantly enhances the difficulty of the defender to learn the differences between legal and illegal models through machine-learning approaches. Our experiments on 3 popular datasets show that it can escape detection by all five state-of-the-art model inspection schemes. Moreover, this attack brings almost no side-effects on the attack effectiveness and guarantees the universal feature of the trigger compared with original patch-based trojan attacks.

摘要: 外包的深度神经网络已经被证明遭受基于补丁的特洛伊木马攻击，在这种攻击中，对手毒化训练集，在所获得的模型中注入后门，以便仍然可以正确地标记常规输入，而那些带有特定触发器的输入被错误地给予目标标签。由于这种攻击的严重性，最近提出了许多用于深度神经网络的后门检测和遏制系统。其中一个主要类别是各种模型检查方案，它们希望在部署来自不可信第三方的模型之前检测后门。在本文中，我们证明了这种最先进的方案可以被所谓的替罪羊后门攻击所击败，即在数据中毒中引入良性的替罪羊触发器，以防止防御者逆转真正的异常触发器。此外，在训练过程中，它将网络参数的值限制在与CLEAN模型相同的方差内，这进一步增加了防御者通过机器学习方法学习合法和非法模型之间的差异的难度。我们在3个流行的数据集上的实验表明，它可以逃脱所有五种最先进的模型检测方案的检测。此外，与原有的基于补丁的木马攻击相比，该攻击几乎不会对攻击效果产生副作用，并保证了触发器的通用特性。



## **46. Unreasonable Effectiveness of Last Hidden Layer Activations for Adversarial Robustness**

最后隐含层激活对对抗健壮性的不合理有效性 cs.LG

IEEE COMPSAC 2022 publication full version

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.07342v2)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的惯例是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来获得每一类的概率分数。在这种类型的体系结构中，分类器相对于任何输出类的损失值与最终概率得分和关联类的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用该模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的稳健性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法，如Deepfoo攻击，有一些额外的好处。



## **47. Attacking and Defending Deep Reinforcement Learning Policies**

攻击和防御深度强化学习策略 cs.LG

nine pages

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07626v1)

**Authors**: Chao Wang

**Abstracts**: Recent studies have shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial attacks, which raise concerns about applications of DRL to safety-critical systems. In this work, we adopt a principled way and study the robustness of DRL policies to adversarial attacks from the perspective of robust optimization. Within the framework of robust optimization, optimal adversarial attacks are given by minimizing the expected return of the policy, and correspondingly a good defense mechanism should be realized by improving the worst-case performance of the policy. Considering that attackers generally have no access to the training environment, we propose a greedy attack algorithm, which tries to minimize the expected return of the policy without interacting with the environment, and a defense algorithm, which performs adversarial training in a max-min form. Experiments on Atari game environments show that our attack algorithm is more effective and leads to worse return of the policy than existing attack algorithms, and our defense algorithm yields policies more robust than existing defense methods to a range of adversarial attacks (including our proposed attack algorithm).

摘要: 最近的研究表明，深度强化学习(DRL)策略容易受到敌意攻击，这引发了人们对DRL在安全关键系统中的应用的担忧。在这项工作中，我们采用原则性的方法，从稳健优化的角度研究了DRL策略对对手攻击的稳健性。在稳健优化的框架下，通过最小化策略的预期收益来给出最优的对抗性攻击，并通过提高策略的最坏情况性能来实现良好的防御机制。考虑到攻击者一般不能访问训练环境，我们提出了贪婪攻击算法和防御算法，贪婪攻击算法试图在不与环境交互的情况下最小化策略的期望回报，防御算法以max-min的形式执行对抗性训练。在Atari游戏环境下的实验表明，我们的攻击算法比现有的攻击算法更有效，策略回报更差，而我们的防御算法对一系列对抗性攻击(包括我们提出的攻击算法)产生的策略比现有的防御方法更健壮。



## **48. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

17 pages, 13 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.03195v2)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training datasets of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。CBA是通过在每个恶意方的训练过程中嵌入相同的全局触发器来进行的，而DBA是通过将全局触发器分解为单独的局部触发器并将其分别嵌入到不同恶意方的训练数据集中来进行的。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步研究联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对两种最先进的防御措施的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **49. Learning Classical Readout Quantum PUFs based on single-qubit gates**

基于单量子比特门的经典读出量子PUF学习 quant-ph

12 pages, 9 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2112.06661v2)

**Authors**: Niklas Pirnay, Anna Pappa, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.

摘要: 物理不可克隆功能(PUF)已经被提出作为识别和认证电子设备的一种方式。最近，已经提出了几个旨在实现同样的量子设备的想法。其中一些结构应用了单量子比特门，以便提供量子设备的安全指纹。在这项工作中，我们使用统计查询(SQ)模型形式化了经典读出量子PUF(CR-QPUF)，并显式地证明了当攻击者可以访问CR-QPUF时，基于单量子比特旋转门的CR-QPUF是不安全的。我们演示了恶意方如何学习CR-QPUF特征，并通过使用简单的低次多项式回归的建模攻击来伪造量子设备的签名。所提出的模型化攻击在真实的IBM Q量子机上的真实场景中被成功地实现。我们深入讨论了利用量子器件缺陷作为安全指纹的CR-QPUF的前景和存在的问题。



## **50. Manifold Characteristics That Predict Downstream Task Performance**

预测下游任务绩效的多种特征 cs.LG

Currently under review

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07477v1)

**Authors**: Ruan van der Merwe, Gregory Newman, Etienne Barnard

**Abstracts**: Pretraining methods are typically compared by evaluating the accuracy of linear classifiers, transfer learning performance, or visually inspecting the representation manifold's (RM) lower-dimensional projections. We show that the differences between methods can be understood more clearly by investigating the RM directly, which allows for a more detailed comparison. To this end, we propose a framework and new metric to measure and compare different RMs. We also investigate and report on the RM characteristics for various pretraining methods. These characteristics are measured by applying sequentially larger local alterations to the input data, using white noise injections and Projected Gradient Descent (PGD) adversarial attacks, and then tracking each datapoint. We calculate the total distance moved for each datapoint and the relative change in distance between successive alterations. We show that self-supervised methods learn an RM where alterations lead to large but constant size changes, indicating a smoother RM than fully supervised methods. We then combine these measurements into one metric, the Representation Manifold Quality Metric (RMQM), where larger values indicate larger and less variable step sizes, and show that RMQM correlates positively with performance on downstream tasks.

摘要: 通常通过评估线性分类器的准确性、转移学习性能或视觉检查表示流形(RM)的低维投影来比较预训练方法。我们表明，通过直接调查RM，可以更清楚地理解方法之间的差异，这允许更详细的比较。为此，我们提出了一个框架和新的度量来衡量和比较不同的均方根。我们还调查和报告了各种预训练方法的RM特征。这些特征是通过对输入数据应用顺序更大的局部改变、使用白噪声注入和预测的梯度下降(PGD)对抗性攻击，然后跟踪每个数据点来衡量的。我们计算每个数据点移动的总距离以及连续更改之间的相对距离变化。我们表明，自我监督方法学习的RM中，变化导致较大但恒定的大小变化，表明比完全监督方法更平滑的RM。然后，我们将这些测量组合成一个度量，表示流形质量度量(RMQM)，其中较大的值表示更大且变化较小的步长，并表明RMQM与下游任务的性能呈正相关。



