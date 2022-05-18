# Latest Adversarial Attack Papers
**update at 2022-05-19 06:31:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. F3B: A Low-Latency Commit-and-Reveal Architecture to Mitigate Blockchain Front-Running**

F3B：一种降低区块链前运行的低延迟提交与揭示体系结构 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08529v1)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Vero Estrada-Galinanes, Bryan Ford

**Abstracts**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the cryptocurrency space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants$\unicode{x2013}$estimated at \$280M each month$\unicode{x2013}$and endangers the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture to address front-running attacks by relying on a commit-and-reveal scheme where the contents of transactions are encrypted and later revealed by a decentralized secret-management committee once the underlying consensus layer has committed the transaction. F3B mitigates front-running attacks because an adversary can no longer read the content of a transaction before commitment, thus preventing the adversary from benefiting from advance knowledge of pending transactions. We design F3B to be agnostic to the underlying consensus algorithm and compatible with legacy smart contracts by addressing front-running at the blockchain architecture level. Unlike existing commit-and-reveal approaches, F3B only requires writing data onto the underlying blockchain once, establishing a significant overhead reduction. An exploration of F3B shows that with a secret-management committee consisting of 8 and 128 members, F3B presents between 0.1 and 1.8 seconds of transaction-processing latency, respectively.

摘要: 自去中心化金融出现以来，受益于待完成交易的先进知识的前沿攻击在加密货币领域激增。领跑给诚实的参与者造成了毁灭性的损失，估计每月损失2.8亿美元，并危及生态系统的公平性。我们提出了Flash冷冻Flash Boys(F3B)，这是一种区块链架构，通过依赖提交并披露方案来应对前沿攻击，其中交易的内容被加密，一旦底层共识层提交交易，随后由分散的秘密管理委员会披露。F3B减轻了前置攻击，因为对手在提交之前不能再读取交易的内容，从而防止对手受益于对未决交易的预先了解。我们将F3B设计为与底层共识算法无关，并通过在区块链架构级别解决先期运行问题与传统智能合同兼容。与现有的提交和揭示方法不同，F3B只需将数据写入底层区块链一次，从而显著降低了开销。对F3B的研究表明，对于由8名和128名成员组成的秘密管理委员会，F3B的事务处理延迟分别在0.1秒到1.8秒之间。



## **2. On the Privacy of Decentralized Machine Learning**

关于分散式机器学习的隐私性 cs.CR

17 pages

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08443v1)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstracts**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at circumventing the main limitations of federated learning. We identify the decentralized learning properties that affect users' privacy and we introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantages over more practical approaches such as federated learning. Rather, it tends to degrade users' privacy by increasing the attack surface and enabling any user in the system to perform powerful privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also reveal that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require abandoning any possible advantage over the federated setup, completely defeating the objective of the decentralized approach.

摘要: 在这项工作中，我们进行了第一次，深入的，隐私分析的分散学习--一个合作的机器学习框架，旨在绕过联邦学习的主要限制。我们识别了影响用户隐私的分散学习属性，并针对被动和主动的分散攻击引入了一套新的攻击。我们证明，与去中心化学习提出者所声称的相反，去中心化学习并不比联邦学习等更实用的方法提供任何安全优势。相反，它往往会通过增加攻击面来降低用户的隐私，使系统中的任何用户都可以执行强大的隐私攻击，如梯度反转，甚至获得对诚实用户的本地模型的完全控制。我们还揭示，考虑到保护措施的最新水平，去中心化学习的隐私保护配置要求放弃任何可能的优势，而不是联邦设置，完全违背了去中心化方法的目标。



## **3. Can You Still See Me?: Reconstructing Robot Operations Over End-to-End Encrypted Channels**

你还能看到我吗？：在端到端加密通道上重建机器人操作 cs.CR

13 pages, 7 figures, poster presented at wisec'22

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08426v1)

**Authors**: Ryan Shah, Chuadhry Mujeeb Ahmed, Shishir Nagaraja

**Abstracts**: Connected robots play a key role in Industry 4.0, providing automation and higher efficiency for many industrial workflows. Unfortunately, these robots can leak sensitive information regarding these operational workflows to remote adversaries. While there exists mandates for the use of end-to-end encryption for data transmission in such settings, it is entirely possible for passive adversaries to fingerprint and reconstruct entire workflows being carried out -- establishing an understanding of how facilities operate. In this paper, we investigate whether a remote attacker can accurately fingerprint robot movements and ultimately reconstruct operational workflows. Using a neural network approach to traffic analysis, we find that one can predict TLS-encrypted movements with around \textasciitilde60\% accuracy, increasing to near-perfect accuracy under realistic network conditions. Further, we also find that attackers can reconstruct warehousing workflows with similar success. Ultimately, simply adopting best cybersecurity practices is clearly not enough to stop even weak (passive) adversaries.

摘要: 互联机器人在工业4.0中扮演着关键角色，为许多工业工作流程提供自动化和更高的效率。不幸的是，这些机器人可能会将有关这些操作工作流程的敏感信息泄露给远程对手。虽然在这种情况下有使用端到端加密进行数据传输的规定，但被动攻击者完全有可能对正在执行的整个工作流程进行指纹识别和重建--建立对设施如何运行的理解。在本文中，我们调查远程攻击者是否能够准确地识别机器人的运动并最终重建操作工作流。使用神经网络方法对流量进行分析，我们发现可以预测TLS加密的移动，精度约为60%，在现实网络条件下提高到接近完美的精度。此外，我们还发现攻击者可以成功地重构仓储工作流。归根结底，简单地采用最佳网络安全实践显然不足以阻止即使是弱小的(被动的)对手。



## **4. Bankrupting DoS Attackers Despite Uncertainty**

尽管存在不确定性，但仍使DoS攻击者破产 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08287v1)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as economic denial-of-sustainability (EDoS), where the cost for defending a service is higher than that of attacking.   A natural approach for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior approaches with security guarantees do not account for the cost on-demand provisioning.   Another valuable defensive tool is to use a classifier in order to discern good jobs from a legitimate client, versus bad jobs from the adversary. However, while useful, uncertainty arises from classification error, which still allows bad jobs to consume server resources. Thus, classification is not a solution by itself.   Here, we propose an EDoS defense, RootDef, that leverages both RB and classification, while accounting for both the costs of resource burning and on-demand provisioning. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $\tilde{O}( \sqrt{B\,g} + B^{2/3} + g)$, where $g$ is the number of good jobs and $\tilde{O}$ refers to hidden logarithmic factors in the total number of jobs $n$. Notably, for large $B$ relative to $g$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound showing that RootDef has total costs that are asymptotically tight up to logarithmic factors in $n$.

摘要: 云中的按需配置允许服务在遭受大规模拒绝服务(DoS)攻击时保持可用。不幸的是，按需配置的成本很高，必须权衡对手所产生的成本。这导致了最近一种被称为经济拒绝可持续性(EDOS)的威胁，在这种威胁下，防御服务的成本高于攻击。打击EDO的一个自然方法是通过资源燃烧(RB)来施加成本。在这里，在提供服务之前，客户端必须可验证地消耗资源--例如，通过解决计算挑战。然而，以前的具有安全保证的方法不考虑按需供应的成本。另一个有价值的防御工具是使用分类器，以便区分合法客户的好工作，而不是对手的坏工作。但是，尽管分类错误很有用，但不确定性源于分类错误，分类错误仍然允许不良作业消耗服务器资源。因此，分类本身并不是一个解决方案。在这里，我们提出了EDOS防御措施RootDef，它利用RB和分类，同时考虑了资源消耗和按需配置的成本。具体地说，对于花费$B$资源进行攻击的对手，防御的总成本为$\tide{O}(\sqrt{B\，g}+B^{2/3}+g)$，其中$g$是好工作的数目，$\tide{O}$是工作总数$n$中的隐藏对数因子。值得注意的是，对于较大的$B$相对于$G$，对手具有更高的成本，这意味着该算法具有经济优势。最后，我们证明了一个下界，证明了RootDef的总成本是渐近紧到$n$的对数因子的。



## **5. How Not to Handle Keys: Timing Attacks on FIDO Authenticator Privacy**

如何不处理密钥：对FIDO验证器隐私的计时攻击 cs.CR

to be published in the 22nd Privacy Enhancing Technologies Symposium  (PETS 2022)

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2205.08071v1)

**Authors**: Michal Kepkowski, Lucjan Hanzlik, Ian Wood, Mohamed Ali Kaafar

**Abstracts**: This paper presents a timing attack on the FIDO2 (Fast IDentity Online) authentication protocol that allows attackers to link user accounts stored in vulnerable authenticators, a serious privacy concern. FIDO2 is a new standard specified by the FIDO industry alliance for secure token online authentication. It complements the W3C WebAuthn specification by providing means to use a USB token or other authenticator as a second factor during the authentication process. From a cryptographic perspective, the protocol is a simple challenge-response where the elliptic curve digital signature algorithm is used to sign challenges. To protect the privacy of the user the token uses unique key pairs per service. To accommodate for small memory, tokens use various techniques that make use of a special parameter called a key handle sent by the service to the token. We identify and analyse a vulnerability in the way the processing of key handles is implemented that allows attackers to remotely link user accounts on multiple services. We show that for vulnerable authenticators there is a difference between the time it takes to process a key handle for a different service but correct authenticator, and for a different authenticator but correct service. This difference can be used to perform a timing attack allowing an adversary to link user's accounts across services. We present several real world examples of adversaries that are in a position to execute our attack and can benefit from linking accounts. We found that two of the eight hardware authenticators we tested were vulnerable despite FIDO level 1 certification. This vulnerability cannot be easily mitigated on authenticators because, for security reasons, they usually do not allow firmware updates. In addition, we show that due to the way existing browsers implement the WebAuthn standard, the attack can be executed remotely.

摘要: 提出了一种对FIDO2(Fast Identity Online)认证协议的计时攻击，使得攻击者能够链接存储在易受攻击的认证器中的用户帐户，这是一个严重的隐私问题。FIDO2是由FIDO行业联盟为安全令牌在线身份验证指定的新标准。它补充了W3CWebAuthn规范，提供了在身份验证过程中使用USB令牌或其他身份验证器作为第二因素的方法。从密码学的角度来看，该协议是一个简单的挑战-响应协议，其中使用椭圆曲线数字签名算法来签署挑战。为了保护用户的隐私，令牌对每个服务使用唯一的密钥对。为了适应较小的内存，令牌使用各种技术，这些技术利用由服务发送到令牌的称为密钥句柄的特殊参数。我们发现并分析了密钥句柄处理实现方式中的一个漏洞，该漏洞允许攻击者远程链接多个服务上的用户帐户。我们证明，对于易受攻击的验证器，处理不同服务但正确的验证器的密钥句柄所花费的时间与处理不同的验证器但正确的服务的密钥句柄所花费的时间是不同的。这种差异可用于执行计时攻击，从而允许对手跨服务链接用户帐户。我们提供了几个真实世界的例子，这些对手处于执行我们的攻击的位置，并可以从链接帐户中受益。我们发现，尽管通过了FIDO级别1认证，但我们测试的八个硬件验证器中有两个容易受到攻击。在验证器上无法轻松缓解此漏洞，因为出于安全原因，验证器通常不允许固件更新。此外，我们还表明，由于现有浏览器实现WebAuthn标准的方式，攻击可以远程执行。



## **6. User-Level Differential Privacy against Attribute Inference Attack of Speech Emotion Recognition in Federated Learning**

联合学习中抵抗语音情感识别属性推理攻击的用户级差分隐私 cs.CR

**SubmitDate**: 2022-05-17    [paper-pdf](http://arxiv.org/pdf/2204.02500v2)

**Authors**: Tiantian Feng, Raghuveer Peri, Shrikanth Narayanan

**Abstracts**: Many existing privacy-enhanced speech emotion recognition (SER) frameworks focus on perturbing the original speech data through adversarial training within a centralized machine learning setup. However, this privacy protection scheme can fail since the adversary can still access the perturbed data. In recent years, distributed learning algorithms, especially federated learning (FL), have gained popularity to protect privacy in machine learning applications. While FL provides good intuition to safeguard privacy by keeping the data on local devices, prior work has shown that privacy attacks, such as attribute inference attacks, are achievable for SER systems trained using FL. In this work, we propose to evaluate the user-level differential privacy (UDP) in mitigating the privacy leaks of the SER system in FL. UDP provides theoretical privacy guarantees with privacy parameters $\epsilon$ and $\delta$. Our results show that the UDP can effectively decrease attribute information leakage while keeping the utility of the SER system with the adversary accessing one model update. However, the efficacy of the UDP suffers when the FL system leaks more model updates to the adversary. We make the code publicly available to reproduce the results in https://github.com/usc-sail/fed-ser-leakage.

摘要: 许多现有的隐私增强型语音情感识别(SER)框架专注于通过集中式机器学习设置中的对抗性训练来扰乱原始语音数据。然而，这种隐私保护方案可能会失败，因为攻击者仍然可以访问受干扰的数据。近年来，分布式学习算法，特别是联邦学习(FL)算法在机器学习应用中保护隐私得到了广泛的应用。虽然FL通过将数据保存在本地设备上来提供良好的直觉来保护隐私，但先前的工作表明，使用FL训练的SER系统可以实现隐私攻击，例如属性推理攻击。在这项工作中，我们建议评估用户级差异隐私(UDP)在缓解FL中SER系统的隐私泄漏方面的作用。UDP通过隐私参数$\epsilon$和$\Delta$提供理论上的隐私保证。实验结果表明，UDP协议在保持SER系统可用性的同时，有效地减少了属性信息泄露，且攻击者只需访问一次模型更新。然而，当FL系统向对手泄露更多的模型更新时，UDP的效率会受到影响。我们将代码公开，以便在https://github.com/usc-sail/fed-ser-leakage.中重现结果



## **7. RoVISQ: Reduction of Video Service Quality via Adversarial Attacks on Deep Learning-based Video Compression**

RoVISQ：通过对基于深度学习的视频压缩进行对抗性攻击来降低视频服务质量 cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2203.10183v2)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in video streaming and classification systems by maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning-based video compression and downstream classification systems. Our attack framework, dubbed RoVISQ, manipulates the Rate-Distortion (R-D) relationship of a video compression model to achieve one or both of the following goals: (1) increasing the network bandwidth, (2) degrading the video quality for end-users. We further devise new objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of RoVISQ attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets show RoVISQ attacks deteriorate peak signal-to-noise ratio by up to 5.6dB and the bit-rate by up to 2.4 times while achieving over 90% attack success rate on a downstream classifier.

摘要: 视频压缩在视频流和分类系统中起着至关重要的作用，它在给定的带宽预算下最大化最终用户的体验质量(QOE)。本文首次对基于深度学习的视频压缩和下行分类系统的敌意攻击进行了系统的研究。我们的攻击框架RoVISQ通过操纵视频压缩模型的率失真(R-D)关系来实现以下一个或两个目标：(1)增加网络带宽；(2)降低最终用户的视频质量。我们进一步制定了针对下游视频分类服务的定向和非定向攻击的新目标。最后，我们设计了一种输入不变的扰动，该扰动普遍地扰乱了视频压缩和分类系统的实时。与之前提出的针对视频分类的攻击不同，我们的对抗性扰动最先经受住了压缩。我们经验地展示了RoVISQ攻击对各种防御措施的弹性，即对抗性训练、视频去噪和JPEG压缩。我们在不同视频数据集上的大量实验结果表明，RoVISQ攻击使峰值信噪比下降5.6dB，比特率下降2.4倍，同时在下游分类器上获得90%以上的攻击成功率。



## **8. Classification Auto-Encoder based Detector against Diverse Data Poisoning Attacks**

基于分类自动编码器的抗多种数据中毒攻击检测器 cs.LG

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2108.04206v2)

**Authors**: Fereshteh Razmi, Li Xiong

**Abstracts**: Poisoning attacks are a category of adversarial machine learning threats in which an adversary attempts to subvert the outcome of the machine learning systems by injecting crafted data into training data set, thus increasing the machine learning model's test error. The adversary can tamper with the data feature space, data labels, or both, each leading to a different attack strategy with different strengths. Various detection approaches have recently emerged, each focusing on one attack strategy. The Achilles heel of many of these detection approaches is their dependence on having access to a clean, untampered data set. In this paper, we propose CAE, a Classification Auto-Encoder based detector against diverse poisoned data. CAE can detect all forms of poisoning attacks using a combination of reconstruction and classification errors without having any prior knowledge of the attack strategy. We show that an enhanced version of CAE (called CAE+) does not have to employ a clean data set to train the defense model. Our experimental results on three real datasets MNIST, Fashion-MNIST and CIFAR demonstrate that our proposed method can maintain its functionality under up to 30% contaminated data and help the defended SVM classifier to regain its best accuracy.

摘要: 中毒攻击是一种对抗性机器学习威胁，对手试图通过将精心设计的数据注入训练数据集来颠覆机器学习系统的结果，从而增加机器学习模型的测试误差。对手可以篡改数据特征空间、数据标签或两者，每一种都会导致具有不同强度的不同攻击策略。最近出现了各种检测方法，每种方法都专注于一种攻击策略。许多这些检测方法的致命弱点是它们依赖于能够访问干净、未经篡改的数据集。在本文中，我们提出了一种针对不同有毒数据的基于分类自动编码器的检测器CAE。CAE可以使用重构和分类错误的组合来检测所有形式的中毒攻击，而不需要事先了解攻击策略。我们证明了CAE的增强版本(称为CAE+)不必使用干净的数据集来训练防御模型。我们在三个真实数据集MNIST、Fashion-MNIST和CIFAR上的实验结果表明，我们的方法可以在高达30%的污染数据下保持其功能，并帮助被防御的支持向量机分类器恢复其最佳精度。



## **9. Transferability of Adversarial Attacks on Synthetic Speech Detection**

合成语音检测中对抗性攻击的可转移性 cs.SD

5 pages, submit to Interspeech2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07711v1)

**Authors**: Jiacheng Deng, Shunyi Chen, Li Dong, Diqun Yan, Rangding Wang

**Abstracts**: Synthetic speech detection is one of the most important research problems in audio security. Meanwhile, deep neural networks are vulnerable to adversarial attacks. Therefore, we establish a comprehensive benchmark to evaluate the transferability of adversarial attacks on the synthetic speech detection task. Specifically, we attempt to investigate: 1) The transferability of adversarial attacks between different features. 2) The influence of varying extraction hyperparameters of features on the transferability of adversarial attacks. 3) The effect of clipping or self-padding operation on the transferability of adversarial attacks. By performing these analyses, we summarise the weaknesses of synthetic speech detectors and the transferability behaviours of adversarial attacks, which provide insights for future research. More details can be found at https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.

摘要: 合成语音检测是音频安全领域的重要研究课题之一。与此同时，深度神经网络很容易受到敌意攻击。因此，我们建立了一个综合的基准来评估对抗性攻击对合成语音检测任务的可转移性。具体地说，我们试图研究：1)对抗性攻击在不同特征之间的可转移性。2)不同特征提取超参数对对抗性攻击可转移性的影响。3)截断或自填充操作对对抗性攻击可转移性的影响。通过这些分析，我们总结了合成语音检测器的弱点和对抗性攻击的可转移性行为，为未来的研究提供了见解。欲了解更多详情，请访问https://gitee.com/djc_QRICK/Attack-Transferability-On-Synthetic-Detection.。



## **10. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.01287v2)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **11. SGBA: A Stealthy Scapegoat Backdoor Attack against Deep Neural Networks**

SGBA：针对深度神经网络的隐形替罪羊后门攻击 cs.CR

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2104.01026v3)

**Authors**: Ying He, Zhili Shen, Chang Xia, Jingyu Hua, Wei Tong, Sheng Zhong

**Abstracts**: Outsourced deep neural networks have been demonstrated to suffer from patch-based trojan attacks, in which an adversary poisons the training sets to inject a backdoor in the obtained model so that regular inputs can be still labeled correctly while those carrying a specific trigger are falsely given a target label. Due to the severity of such attacks, many backdoor detection and containment systems have recently, been proposed for deep neural networks. One major category among them are various model inspection schemes, which hope to detect backdoors before deploying models from non-trusted third-parties. In this paper, we show that such state-of-the-art schemes can be defeated by a so-called Scapegoat Backdoor Attack, which introduces a benign scapegoat trigger in data poisoning to prevent the defender from reversing the real abnormal trigger. In addition, it confines the values of network parameters within the same variances of those from clean model during training, which further significantly enhances the difficulty of the defender to learn the differences between legal and illegal models through machine-learning approaches. Our experiments on 3 popular datasets show that it can escape detection by all five state-of-the-art model inspection schemes. Moreover, this attack brings almost no side-effects on the attack effectiveness and guarantees the universal feature of the trigger compared with original patch-based trojan attacks.

摘要: 外包的深度神经网络已经被证明遭受基于补丁的特洛伊木马攻击，在这种攻击中，对手毒化训练集，在所获得的模型中注入后门，以便仍然可以正确地标记常规输入，而那些带有特定触发器的输入被错误地给予目标标签。由于这种攻击的严重性，最近提出了许多用于深度神经网络的后门检测和遏制系统。其中一个主要类别是各种模型检查方案，它们希望在部署来自不可信第三方的模型之前检测后门。在本文中，我们证明了这种最先进的方案可以被所谓的替罪羊后门攻击所击败，即在数据中毒中引入良性的替罪羊触发器，以防止防御者逆转真正的异常触发器。此外，在训练过程中，它将网络参数的值限制在与CLEAN模型相同的方差内，这进一步增加了防御者通过机器学习方法学习合法和非法模型之间的差异的难度。我们在3个流行的数据集上的实验表明，它可以逃脱所有五种最先进的模型检测方案的检测。此外，与原有的基于补丁的木马攻击相比，该攻击几乎不会对攻击效果产生副作用，并保证了触发器的通用特性。



## **12. Unreasonable Effectiveness of Last Hidden Layer Activations for Adversarial Robustness**

最后隐含层激活对对抗健壮性的不合理有效性 cs.LG

IEEE COMPSAC 2022 publication full version

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.07342v2)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的惯例是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来获得每一类的概率分数。在这种类型的体系结构中，分类器相对于任何输出类的损失值与最终概率得分和关联类的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用该模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的稳健性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法，如Deepfoo攻击，有一些额外的好处。



## **13. Attacking and Defending Deep Reinforcement Learning Policies**

攻击和防御深度强化学习策略 cs.LG

nine pages

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07626v1)

**Authors**: Chao Wang

**Abstracts**: Recent studies have shown that deep reinforcement learning (DRL) policies are vulnerable to adversarial attacks, which raise concerns about applications of DRL to safety-critical systems. In this work, we adopt a principled way and study the robustness of DRL policies to adversarial attacks from the perspective of robust optimization. Within the framework of robust optimization, optimal adversarial attacks are given by minimizing the expected return of the policy, and correspondingly a good defense mechanism should be realized by improving the worst-case performance of the policy. Considering that attackers generally have no access to the training environment, we propose a greedy attack algorithm, which tries to minimize the expected return of the policy without interacting with the environment, and a defense algorithm, which performs adversarial training in a max-min form. Experiments on Atari game environments show that our attack algorithm is more effective and leads to worse return of the policy than existing attack algorithms, and our defense algorithm yields policies more robust than existing defense methods to a range of adversarial attacks (including our proposed attack algorithm).

摘要: 最近的研究表明，深度强化学习(DRL)策略容易受到敌意攻击，这引发了人们对DRL在安全关键系统中的应用的担忧。在这项工作中，我们采用原则性的方法，从稳健优化的角度研究了DRL策略对对手攻击的稳健性。在稳健优化的框架下，通过最小化策略的预期收益来给出最优的对抗性攻击，并通过提高策略的最坏情况性能来实现良好的防御机制。考虑到攻击者一般不能访问训练环境，我们提出了贪婪攻击算法和防御算法，贪婪攻击算法试图在不与环境交互的情况下最小化策略的期望回报，防御算法以max-min的形式执行对抗性训练。在Atari游戏环境下的实验表明，我们的攻击算法比现有的攻击算法更有效，策略回报更差，而我们的防御算法对一系列对抗性攻击(包括我们提出的攻击算法)产生的策略比现有的防御方法更健壮。



## **14. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

17 pages, 13 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2202.03195v2)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstracts**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). CBA is conducted by embedding the same global trigger during training for every malicious party, while DBA is conducted by decomposing a global trigger into separate local triggers and embedding them into the training datasets of different malicious parties, respectively. Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against two state-of-the-art defenses. We find that both attacks are robust against the investigated defenses, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。CBA是通过在每个恶意方的训练过程中嵌入相同的全局触发器来进行的，而DBA是通过将全局触发器分解为单独的局部触发器并将其分别嵌入到不同恶意方的训练数据集中来进行的。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步研究联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对两种最先进的防御措施的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **15. Learning Classical Readout Quantum PUFs based on single-qubit gates**

基于单量子比特门的经典读出量子PUF学习 quant-ph

12 pages, 9 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2112.06661v2)

**Authors**: Niklas Pirnay, Anna Pappa, Jean-Pierre Seifert

**Abstracts**: Physical Unclonable Functions (PUFs) have been proposed as a way to identify and authenticate electronic devices. Recently, several ideas have been presented that aim to achieve the same for quantum devices. Some of these constructions apply single-qubit gates in order to provide a secure fingerprint of the quantum device. In this work, we formalize the class of Classical Readout Quantum PUFs (CR-QPUFs) using the statistical query (SQ) model and explicitly show insufficient security for CR-QPUFs based on single qubit rotation gates, when the adversary has SQ access to the CR-QPUF. We demonstrate how a malicious party can learn the CR-QPUF characteristics and forge the signature of a quantum device through a modelling attack using a simple regression of low-degree polynomials. The proposed modelling attack was successfully implemented in a real-world scenario on real IBM Q quantum machines. We thoroughly discuss the prospects and problems of CR-QPUFs where quantum device imperfections are used as a secure fingerprint.

摘要: 物理不可克隆功能(PUF)已经被提出作为识别和认证电子设备的一种方式。最近，已经提出了几个旨在实现同样的量子设备的想法。其中一些结构应用了单量子比特门，以便提供量子设备的安全指纹。在这项工作中，我们使用统计查询(SQ)模型形式化了经典读出量子PUF(CR-QPUF)，并显式地证明了当攻击者可以访问CR-QPUF时，基于单量子比特旋转门的CR-QPUF是不安全的。我们演示了恶意方如何学习CR-QPUF特征，并通过使用简单的低次多项式回归的建模攻击来伪造量子设备的签名。所提出的模型化攻击在真实的IBM Q量子机上的真实场景中被成功地实现。我们深入讨论了利用量子器件缺陷作为安全指纹的CR-QPUF的前景和存在的问题。



## **16. Manifold Characteristics That Predict Downstream Task Performance**

预测下游任务绩效的多种特征 cs.LG

Currently under review

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07477v1)

**Authors**: Ruan van der Merwe, Gregory Newman, Etienne Barnard

**Abstracts**: Pretraining methods are typically compared by evaluating the accuracy of linear classifiers, transfer learning performance, or visually inspecting the representation manifold's (RM) lower-dimensional projections. We show that the differences between methods can be understood more clearly by investigating the RM directly, which allows for a more detailed comparison. To this end, we propose a framework and new metric to measure and compare different RMs. We also investigate and report on the RM characteristics for various pretraining methods. These characteristics are measured by applying sequentially larger local alterations to the input data, using white noise injections and Projected Gradient Descent (PGD) adversarial attacks, and then tracking each datapoint. We calculate the total distance moved for each datapoint and the relative change in distance between successive alterations. We show that self-supervised methods learn an RM where alterations lead to large but constant size changes, indicating a smoother RM than fully supervised methods. We then combine these measurements into one metric, the Representation Manifold Quality Metric (RMQM), where larger values indicate larger and less variable step sizes, and show that RMQM correlates positively with performance on downstream tasks.

摘要: 通常通过评估线性分类器的准确性、转移学习性能或视觉检查表示流形(RM)的低维投影来比较预训练方法。我们表明，通过直接调查RM，可以更清楚地理解方法之间的差异，这允许更详细的比较。为此，我们提出了一个框架和新的度量来衡量和比较不同的均方根。我们还调查和报告了各种预训练方法的RM特征。这些特征是通过对输入数据应用顺序更大的局部改变、使用白噪声注入和预测的梯度下降(PGD)对抗性攻击，然后跟踪每个数据点来衡量的。我们计算每个数据点移动的总距离以及连续更改之间的相对距离变化。我们表明，自我监督方法学习的RM中，变化导致较大但恒定的大小变化，表明比完全监督方法更平滑的RM。然后，我们将这些测量组合成一个度量，表示流形质量度量(RMQM)，其中较大的值表示更大且变化较小的步长，并表明RMQM与下游任务的性能呈正相关。



## **17. Robust Representation via Dynamic Feature Aggregation**

基于动态特征聚合的稳健表示 cs.CV

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07466v1)

**Authors**: Haozhe Liu, Haoqin Ji, Yuexiang Li, Nanjun He, Haoqian Wu, Feng Liu, Linlin Shen, Yefeng Zheng

**Abstracts**: Deep convolutional neural network (CNN) based models are vulnerable to the adversarial attacks. One of the possible reasons is that the embedding space of CNN based model is sparse, resulting in a large space for the generation of adversarial samples. In this study, we propose a method, denoted as Dynamic Feature Aggregation, to compress the embedding space with a novel regularization. Particularly, the convex combination between two samples are regarded as the pivot for aggregation. In the embedding space, the selected samples are guided to be similar to the representation of the pivot. On the other side, to mitigate the trivial solution of such regularization, the last fully-connected layer of the model is replaced by an orthogonal classifier, in which the embedding codes for different classes are processed orthogonally and separately. With the regularization and orthogonal classifier, a more compact embedding space can be obtained, which accordingly improves the model robustness against adversarial attacks. An averaging accuracy of 56.91% is achieved by our method on CIFAR-10 against various attack methods, which significantly surpasses a solid baseline (Mixup) by a margin of 37.31%. More surprisingly, empirical results show that, the proposed method can also achieve the state-of-the-art performance for out-of-distribution (OOD) detection, due to the learned compact feature space. An F1 score of 0.937 is achieved by the proposed method, when adopting CIFAR-10 as in-distribution (ID) dataset and LSUN as OOD dataset. Code is available at https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.

摘要: 基于深度卷积神经网络(CNN)的模型容易受到敌意攻击。可能的原因之一是基于CNN的模型的嵌入空间稀疏，导致产生对抗性样本的空间很大。在这项研究中，我们提出了一种动态特征聚集的方法，用一种新的正则化方法压缩嵌入空间。特别地，两个样本之间的凸组合被认为是聚集的支点。在嵌入空间中，所选样本被引导为类似于枢轴的表示。另一方面，为了减少这种正则化的平凡解，将模型的最后一层完全连通替换为一个正交分类器，其中不同类别的嵌入码分别进行正交和单独处理。利用正则化和正交分类器，可以得到更紧凑的嵌入空间，从而提高了模型对敌意攻击的鲁棒性。我们的方法在CIFAR-10上对各种攻击方法的平均准确率达到了56.91%，显著超过了坚实的基线(Mixup)37.31%。更令人惊讶的是，实验结果表明，由于学习到的紧凑特征空间，所提出的方法还可以获得最先进的OOD检测性能。当使用CIFAR-10作为分布内数据集，LSUN作为面向对象的数据集时，该方法的F1得分为0.937。代码可在https://github.com/HaozheLiu-ST/DynamicFeatureAggregation.上找到



## **18. Diffusion Models for Adversarial Purification**

对抗性净化的扩散模型 cs.LG

ICML 2022

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07460v1)

**Authors**: Weili Nie, Brandon Guo, Yujia Huang, Chaowei Xiao, Arash Vahdat, Anima Anandkumar

**Abstracts**: Adversarial purification refers to a class of defense methods that remove adversarial perturbations using a generative model. These methods do not make assumptions on the form of attack and the classification model, and thus can defend pre-existing classifiers against unseen threats. However, their performance currently falls behind adversarial training methods. In this work, we propose DiffPure that uses diffusion models for adversarial purification: Given an adversarial example, we first diffuse it with a small amount of noise following a forward diffusion process, and then recover the clean image through a reverse generative process. To evaluate our method against strong adaptive attacks in an efficient and scalable way, we propose to use the adjoint method to compute full gradients of the reverse generative process. Extensive experiments on three image datasets including CIFAR-10, ImageNet and CelebA-HQ with three classifier architectures including ResNet, WideResNet and ViT demonstrate that our method achieves the state-of-the-art results, outperforming current adversarial training and adversarial purification methods, often by a large margin. Project page: https://diffpure.github.io.

摘要: 对抗性净化是指利用生成模型消除对抗性扰动的一类防御方法。这些方法不对攻击的形式和分类模型做出假设，因此可以保护预先存在的分类器免受未知威胁的攻击。然而，他们目前的表现落后于对抗性训练方法。在这项工作中，我们提出了DiffPure，它使用扩散模型来进行对抗性净化：给定一个对抗性例子，我们首先在正向扩散过程中对其进行少量噪声扩散，然后通过反向生成过程恢复干净的图像。为了有效和可扩展地评估我们的方法抵抗强自适应攻击，我们提出了使用伴随方法来计算反向生成过程的全梯度。在CIFAR-10、ImageNet和CelebA-HQ三种分类器结构(包括ResNet、WideResNet和Vit)上的大量实验表明，我们的方法达到了最先进的结果，远远超过了现有的对手训练和对手净化方法。项目页面：https://diffpure.github.io.



## **19. Trustworthy Graph Neural Networks: Aspects, Methods and Trends**

值得信赖的图神经网络：特点、方法和发展趋势 cs.LG

36 pages, 7 tables, 4 figures

**SubmitDate**: 2022-05-16    [paper-pdf](http://arxiv.org/pdf/2205.07424v1)

**Authors**: He Zhang, Bang Wu, Xingliang Yuan, Shirui Pan, Hanghang Tong, Jian Pei

**Abstracts**: Graph neural networks (GNNs) have emerged as a series of competent graph learning methods for diverse real-world scenarios, ranging from daily applications like recommendation systems and question answering to cutting-edge technologies such as drug discovery in life sciences and n-body simulation in astrophysics. However, task performance is not the only requirement for GNNs. Performance-oriented GNNs have exhibited potential adverse effects like vulnerability to adversarial attacks, unexplainable discrimination against disadvantaged groups, or excessive resource consumption in edge computing environments. To avoid these unintentional harms, it is necessary to build competent GNNs characterised by trustworthiness. To this end, we propose a comprehensive roadmap to build trustworthy GNNs from the view of the various computing technologies involved. In this survey, we introduce basic concepts and comprehensively summarise existing efforts for trustworthy GNNs from six aspects, including robustness, explainability, privacy, fairness, accountability, and environmental well-being. Additionally, we highlight the intricate cross-aspect relations between the above six aspects of trustworthy GNNs. Finally, we present a thorough overview of trending directions for facilitating the research and industrialisation of trustworthy GNNs.

摘要: 图形神经网络(GNN)已经成为一系列适用于各种现实世界场景的有能力的图形学习方法，范围从推荐系统和问答等日常应用到生命科学中的药物发现和天体物理中的n体模拟等尖端技术。然而，任务执行情况并不是对GNN的唯一要求。面向性能的GNN表现出潜在的不利影响，如易受对手攻击、对弱势群体的莫名其妙的歧视、或边缘计算环境中过度的资源消耗。为了避免这些无意的伤害，有必要建立以可信为特征的合格的GNN。为此，我们从涉及的各种计算技术的角度提出了一个全面的路线图来构建可信的GNN。在这次调查中，我们介绍了基本概念，并从稳健性、可解释性、隐私、公平性、问责性和环境福利等六个方面全面总结了现有的可信网络努力。此外，我们还重点介绍了上述六个方面的可信赖GNN之间复杂的交叉关系。最后，我们对促进值得信赖的网络的研究和产业化的趋势方向进行了全面的概述。



## **20. Parameter Adaptation for Joint Distribution Shifts**

联合分布移位的参数自适应 cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.07315v1)

**Authors**: Siddhartha Datta

**Abstracts**: While different methods exist to tackle distinct types of distribution shift, such as label shift (in the form of adversarial attacks) or domain shift, tackling the joint shift setting is still an open problem. Through the study of a joint distribution shift manifesting both adversarial and domain-specific perturbations, we not only show that a joint shift worsens model performance compared to their individual shifts, but that the use of a similar domain worsens performance than a dissimilar domain. To curb the performance drop, we study the use of perturbation sets motivated by input and parameter space bounds, and adopt a meta learning strategy (hypernetworks) to model parameters w.r.t. test-time inputs to recover performance.

摘要: 虽然存在不同的方法来处理不同类型的分布转移，例如标签转移(以对抗性攻击的形式)或域转移，但处理联合转移设置仍然是一个悬而未决的问题。通过对同时表现为对抗性和特定于域的扰动的联合分布移位的研究，我们不仅表明联合移位比它们各自的移位降低了模型的性能，而且使用相似的域比使用不同的域的性能更差。为了抑制性能下降，我们研究了由输入和参数空间边界激励的扰动集的使用，并采用元学习策略(超网络)来建模参数。恢复性能的测试时间输入。



## **21. CE-based white-box adversarial attacks will not work using super-fitting**

基于CE的白盒对抗性攻击将不会使用超级拟合 cs.LG

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2205.02741v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep neural networks are widely used in various fields because of their powerful performance. However, recent studies have shown that deep learning models are vulnerable to adversarial attacks, i.e., adding a slight perturbation to the input will make the model obtain wrong results. This is especially dangerous for some systems with high-security requirements, so this paper proposes a new defense method by using the model super-fitting state to improve the model's adversarial robustness (i.e., the accuracy under adversarial attacks). This paper mathematically proves the effectiveness of super-fitting and enables the model to reach this state quickly by minimizing unrelated category scores (MUCS). Theoretically, super-fitting can resist any existing (even future) CE-based white-box adversarial attacks. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting, and the proposed method is compared with nearly 50 defense models from recent conferences. The experimental results show that the super-fitting method in this paper can make the trained model obtain the highest adversarial robustness.

摘要: 深度神经网络以其强大的性能被广泛应用于各个领域。然而，最近的研究表明，深度学习模型容易受到对抗性攻击，即对输入进行微小的扰动就会使模型得到错误的结果。这对于一些对安全性要求很高的系统来说尤其危险，因此本文提出了一种新的防御方法，利用模型的超拟合状态来提高模型的对抗性稳健性(即在对抗性攻击下的准确性)。本文从数学上证明了超拟合的有效性，并通过最小化不相关类别得分(MUC)使模型快速达到这一状态。从理论上讲，超级拟合可以抵抗任何现有的(甚至是未来的)基于CE的白盒对抗性攻击。此外，本文使用了多种强大的攻击算法来评估超拟合的对抗健壮性，并与最近会议上的近50个防御模型进行了比较。实验结果表明，本文提出的超拟合方法可以使训练后的模型获得最高的对抗鲁棒性。



## **22. Measuring Vulnerabilities of Malware Detectors with Explainability-Guided Evasion Attacks**

利用可解析性引导的逃避攻击测量恶意软件检测器的漏洞 cs.CR

**SubmitDate**: 2022-05-15    [paper-pdf](http://arxiv.org/pdf/2111.10085v3)

**Authors**: Ruoxi Sun, Wei Wang, Tian Dong, Shaofeng Li, Minhui Xue, Gareth Tyson, Haojin Zhu, Mingyu Guo, Surya Nepal

**Abstracts**: Numerous open-source and commercial malware detectors are available. However, their efficacy is threatened by new adversarial attacks, whereby malware attempts to evade detection, e.g., by performing feature-space manipulation. In this work, we propose an explainability-guided and model-agnostic framework for measuring the efficacy of malware detectors when confronted with adversarial attacks. The framework introduces the concept of Accrued Malicious Magnitude (AMM) to identify which malware features should be manipulated to maximize the likelihood of evading detection. We then use this framework to test several state-of-the-art malware detectors' ability to detect manipulated malware. We find that (i) commercial antivirus engines are vulnerable to AMM-guided manipulated samples; (ii) the ability of a manipulated malware generated using one detector to evade detection by another detector (i.e., transferability) depends on the overlap of features with large AMM values between the different detectors; and (iii) AMM values effectively measure the importance of features and explain the ability to evade detection. Our findings shed light on the weaknesses of current malware detectors, as well as how they can be improved.

摘要: 有许多开源和商业恶意软件检测器可用。然而，它们的有效性受到新的敌意攻击的威胁，借此恶意软件试图通过例如执行特征空间操纵来逃避检测。在这项工作中，我们提出了一个可解释性指导和模型不可知的框架来衡量恶意软件检测器在面临敌意攻击时的有效性。该框架引入了累积恶意量级(AMM)的概念，以确定应操纵哪些恶意软件功能以最大限度地提高逃避检测的可能性。然后，我们使用这个框架来测试几个最先进的恶意软件检测器检测操纵恶意软件的能力。我们发现(I)商业反病毒引擎容易受到AMM引导的操纵样本的攻击；(Ii)使用一个检测器生成的操纵恶意软件逃避另一个检测器检测的能力(即可转移性)取决于不同检测器之间具有大AMM值的特征的重叠；以及(Iii)AMM值有效地衡量了特征的重要性并解释了逃避检测的能力。我们的发现揭示了当前恶意软件检测器的弱点，以及如何改进它们。



## **23. Unsupervised Abnormal Traffic Detection through Topological Flow Analysis**

基于拓扑流分析的非监督异常流量检测 cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.07109v1)

**Authors**: Paul Irofti, Andrei Pătraşcu, Andrei Iulian Hîji

**Abstracts**: Cyberthreats are a permanent concern in our modern technological world. In the recent years, sophisticated traffic analysis techniques and anomaly detection (AD) algorithms have been employed to face the more and more subversive adversarial attacks. A malicious intrusion, defined as an invasive action intending to illegally exploit private resources, manifests through unusual data traffic and/or abnormal connectivity pattern. Despite the plethora of statistical or signature-based detectors currently provided in the literature, the topological connectivity component of a malicious flow is less exploited. Furthermore, a great proportion of the existing statistical intrusion detectors are based on supervised learning, that relies on labeled data. By viewing network flows as weighted directed interactions between a pair of nodes, in this paper we present a simple method that facilitate the use of connectivity graph features in unsupervised anomaly detection algorithms. We test our methodology on real network traffic datasets and observe several improvements over standard AD.

摘要: 在我们的现代科技世界中，网络威胁是一个永久性的问题。近年来，复杂的流量分析技术和异常检测(AD)算法被用来应对越来越多的颠覆性敌意攻击。恶意入侵被定义为意图非法利用私有资源的入侵行为，通过异常数据流量和/或异常连接模式表现出来。尽管目前文献中提供了过多的统计或基于签名的检测器，但恶意流的拓扑连通性组件较少被利用。此外，现有的统计入侵检测器有很大一部分是基于监督学习的，而监督学习依赖于标记数据。通过将网络流视为两个节点之间的加权有向交互，本文提出了一种简单的方法，便于在无监督异常检测算法中使用连通图特征。我们在实际网络流量数据集上测试了我们的方法，并观察到与标准AD相比有几个改进。



## **24. Learning Coated Adversarial Camouflages for Object Detectors**

用于目标探测器的学习涂层对抗性伪装 cs.CV

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2109.00124v3)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Jin Zhang, Wu Zhang, Zhisong Pan

**Abstracts**: An adversary can fool deep neural network object detectors by generating adversarial noises. Most of the existing works focus on learning local visible noises in an adversarial "patch" fashion. However, the 2D patch attached to a 3D object tends to suffer from an inevitable reduction in attack performance as the viewpoint changes. To remedy this issue, this work proposes the Coated Adversarial Camouflage (CAC) to attack the detectors in arbitrary viewpoints. Unlike the patch trained in the 2D space, our camouflage generated by a conceptually different training framework consists of 3D rendering and dense proposals attack. Specifically, we make the camouflage perform 3D spatial transformations according to the pose changes of the object. Based on the multi-view rendering results, the top-n proposals of the region proposal network are fixed, and all the classifications in the fixed dense proposals are attacked simultaneously to output errors. In addition, we build a virtual 3D scene to fairly and reproducibly evaluate different attacks. Extensive experiments demonstrate the superiority of CAC over the existing attacks, and it shows impressive performance both in the virtual scene and the real world. This poses a potential threat to the security-critical computer vision systems.

摘要: 敌手可以通过产生敌意噪音来愚弄深度神经网络对象检测器。已有的工作大多集中于以对抗性的“补丁”方式学习局部可见噪声。然而，随着视点的改变，附着到3D对象的2D面片往往会不可避免地降低攻击性能。为了解决这一问题，本文提出了一种覆盖对抗伪装(CAC)来攻击任意视点下的检测器。与在2D空间中训练的补丁不同，我们由概念上不同的训练框架生成的伪装包括3D渲染和密集提议攻击。具体地说，我们让伪装者根据物体的姿态变化进行3D空间变换。基于多视点绘制结果，固定区域建议网络的前n个建议，并同时攻击固定的密集建议中的所有分类以输出错误。此外，我们还构建了一个虚拟的3D场景，以公平和可重复性地评估不同的攻击。大量的实验证明了CAC算法的优越性，并且在虚拟场景和真实世界中都表现出了令人印象深刻的性能。这对安全关键的计算机视觉系统构成了潜在的威胁。



## **25. Rethinking Classifier and Adversarial Attack**

对量词与对抗性攻击的再思考 cs.LG

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.02743v2)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e., not approaching the lower bound of robustness). To solve this problem, this paper uses the proposed decouple space method to divide the classifier into two parts: non-linear and linear. Then, this paper defines the representation vector of the original example (and its space, i.e., the representation space) and uses the iterative optimization of Absolute Classification Boundaries Initialization (ACBI) to obtain a better attack starting point. Particularly, this paper applies ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.

摘要: 人们提出了各种防御模型来抵抗对抗性攻击算法，但现有的对抗性健壮性评估方法总是高估了这些模型的对抗性健壮性(即没有接近鲁棒性的下界)。为了解决这一问题，本文使用提出的解耦空间方法将分类器分为两部分：非线性部分和线性部分。然后，本文定义了原始样本的表示向量(及其空间，即表示空间)，并采用绝对分类边界初始化(ACBI)的迭代优化方法来获得更好的攻击起点。特别是，本文将ACBI应用于近50个广泛使用的防御模型(包括8个体系结构)。实验结果表明，ACBI在所有情况下都表现出较低的稳健性。



## **26. Evaluating Membership Inference Through Adversarial Robustness**

用对抗性稳健性评价隶属度推理 cs.CR

Accepted by The Computer Journal. Pre-print version

**SubmitDate**: 2022-05-14    [paper-pdf](http://arxiv.org/pdf/2205.06986v1)

**Authors**: Zhaoxi Zhang, Leo Yu Zhang, Xufei Zheng, Bilal Hussain Abbasi, Shengshan Hu

**Abstracts**: The usage of deep learning is being escalated in many applications. Due to its outstanding performance, it is being used in a variety of security and privacy-sensitive areas in addition to conventional applications. One of the key aspects of deep learning efficacy is to have abundant data. This trait leads to the usage of data which can be highly sensitive and private, which in turn causes wariness with regard to deep learning in the general public. Membership inference attacks are considered lethal as they can be used to figure out whether a piece of data belongs to the training dataset or not. This can be problematic with regards to leakage of training data information and its characteristics. To highlight the significance of these types of attacks, we propose an enhanced methodology for membership inference attacks based on adversarial robustness, by adjusting the directions of adversarial perturbations through label smoothing under a white-box setting. We evaluate our proposed method on three datasets: Fashion-MNIST, CIFAR-10, and CIFAR-100. Our experimental results reveal that the performance of our method surpasses that of the existing adversarial robustness-based method when attacking normally trained models. Additionally, through comparing our technique with the state-of-the-art metric-based membership inference methods, our proposed method also shows better performance when attacking adversarially trained models. The code for reproducing the results of this work is available at \url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.

摘要: 深度学习的使用在许多应用中都在升级。由于其出色的性能，除了常规应用外，它还被用于各种安全和隐私敏感领域。深度学习效能的一个关键方面是拥有丰富的数据。这一特点导致使用高度敏感和隐私的数据，这反过来又导致公众对深度学习持谨慎态度。成员关系推断攻击被认为是致命的，因为它们可以用来确定一段数据是否属于训练数据集。这在训练数据信息及其特征的泄漏方面可能是问题。为了突出这类攻击的重要性，我们提出了一种基于对抗性稳健性的改进方法，通过白盒设置下的标签平滑来调整对抗性扰动的方向。我们在三个数据集上对我们提出的方法进行了评估：FORM-MNIST、CIFAR-10和CIFAR-100。我们的实验结果表明，在攻击正常训练的模型时，该方法的性能优于现有的基于对抗性稳健性的方法。此外，通过与最新的基于度量的隶属度推理方法的比较，我们提出的方法在攻击恶意训练的模型时也表现出了更好的性能。复制这项工作结果的代码可在\url{https://github.com/plll4zzx/Evaluating-Membership-Inference-Through-Adversarial-Robustness}.上获得



## **27. Universal Post-Training Backdoor Detection**

通用训练后后门检测 cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06900v1)

**Authors**: Hang Wang, Zhen Xiang, David J. Miller, George Kesidis

**Abstracts**: A Backdoor attack (BA) is an important type of adversarial attack against deep neural network classifiers, wherein test samples from one or more source classes will be (mis)classified to the attacker's target class when a backdoor pattern (BP) is embedded. In this paper, we focus on the post-training backdoor defense scenario commonly considered in the literature, where the defender aims to detect whether a trained classifier was backdoor attacked, without any access to the training set. To the best of our knowledge, existing post-training backdoor defenses are all designed for BAs with presumed BP types, where each BP type has a specific embedding function. They may fail when the actual BP type used by the attacker (unknown to the defender) is different from the BP type assumed by the defender. In contrast, we propose a universal post-training defense that detects BAs with arbitrary types of BPs, without making any assumptions about the BP type. Our detector leverages the influence of the BA, independently of the BP type, on the landscape of the classifier's outputs prior to the softmax layer. For each class, a maximum margin statistic is estimated using a set of random vectors; detection inference is then performed by applying an unsupervised anomaly detector to these statistics. Thus, our detector is also an advance relative to most existing post-training methods by not needing any legitimate clean samples, and can efficiently detect BAs with arbitrary numbers of source classes. These advantages of our detector over several state-of-the-art methods are demonstrated on four datasets, for three different types of BPs, and for a variety of attack configurations. Finally, we propose a novel, general approach for BA mitigation once a detection is made.

摘要: 后门攻击(BA)是针对深度神经网络分类器的一种重要的对抗性攻击，当嵌入后门模式(BP)时，来自一个或多个源类的测试样本将被(错误地)分类为攻击者的目标类。在本文中，我们关注文献中通常考虑的训练后后门防御场景，其中防御者的目标是检测训练的分类器是否被后门攻击，而不需要访问训练集。据我们所知，现有的训练后后门防御都是为假定BP类型的BA设计的，其中每种BP类型都有特定的嵌入功能。当攻击者使用的实际BP类型(防御者未知)不同于防御者假定的BP类型时，它们可能失败。相反，我们提出了一种通用的训练后防御，它检测具有任意类型BP的BA，而不对BP类型做出任何假设。我们的检测器利用BA的影响，独立于BP类型，在Softmax层之前对分类器输出的景观进行影响。对于每一类，使用一组随机向量来估计最大边缘统计量；然后通过将无监督异常检测器应用于这些统计量来执行检测推理。因此，相对于大多数已有的后置训练方法，我们的检测器不需要任何合法的干净样本，并且可以有效地检测具有任意数量的信源类的BA。我们的检测器相对于几种最先进的方法的这些优势在四个数据集上进行了演示，这些数据集针对三种不同类型的BPS和各种攻击配置。最后，我们提出了一种新颖的、通用的方法，一旦进行了检测，就可以进行BA缓解。



## **28. secml: A Python Library for Secure and Explainable Machine Learning**

SecML：一种用于安全和可解释的机器学习的Python库 cs.LG

Accepted for publication to SoftwareX. Published version can be found  at: https://doi.org/10.1016/j.softx.2022.101095

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/1912.10013v2)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Marco Melis, Ambra Demontis, Battista Biggio

**Abstracts**: We present \texttt{secml}, an open-source Python library for secure and explainable machine learning. It implements the most popular attacks against machine learning, including test-time evasion attacks to generate adversarial examples against deep neural networks and training-time poisoning attacks against support vector machines and many other algorithms. These attacks enable evaluating the security of learning algorithms and the corresponding defenses under both white-box and black-box threat models. To this end, \texttt{secml} provides built-in functions to compute security evaluation curves, showing how quickly classification performance decreases against increasing adversarial perturbations of the input data. \texttt{secml} also includes explainability methods to help understand why adversarial attacks succeed against a given model, by visualizing the most influential features and training prototypes contributing to each decision. It is distributed under the Apache License 2.0 and hosted at \url{https://github.com/pralab/secml}.

摘要: 我们介绍了\exttt{secml}，这是一个用于安全和可解释的机器学习的开放源码的Python库。它实现了针对机器学习的最流行的攻击，包括测试时间逃避攻击以生成针对深度神经网络的敌意示例，以及针对支持向量机和许多其他算法的训练时间中毒攻击。这些攻击可以在白盒和黑盒威胁模型下评估学习算法和相应防御的安全性。为此，\exttt{secml}提供了计算安全评估曲线的内置函数，显示了针对输入数据日益增加的对抗性扰动，分类性能下降的速度有多快。\exttt{secml}还包括可解释性方法，通过可视化最有影响力的功能和训练对每个决策有贡献的原型，帮助理解针对给定模型的对抗性攻击成功的原因。它是在Apachelicsion2.0下分发的，并托管在\url{https://github.com/pralab/secml}.



## **29. Privacy Preserving Release of Mobile Sensor Data**

保护隐私的移动传感器数据发布 cs.CR

12 pages, 10 figures, 1 table

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06641v1)

**Authors**: Rahat Masood, Wing Yan Cheng, Dinusha Vatsalan, Deepak Mishra, Hassan Jameel Asghar, Mohamed Ali Kaafar

**Abstracts**: Sensors embedded in mobile smart devices can monitor users' activity with high accuracy to provide a variety of services to end-users ranging from precise geolocation, health monitoring, and handwritten word recognition. However, this involves the risk of accessing and potentially disclosing sensitive information of individuals to the apps that may lead to privacy breaches. In this paper, we aim to minimize privacy leakages that may lead to user identification on mobile devices through user tracking and distinguishability while preserving the functionality of apps and services. We propose a privacy-preserving mechanism that effectively handles the sensor data fluctuations (e.g., inconsistent sensor readings while walking, sitting, and running at different times) by formulating the data as time-series modeling and forecasting. The proposed mechanism also uses the notion of correlated noise-series against noise filtering attacks from an adversary, which aims to filter out the noise from the perturbed data to re-identify the original data. Unlike existing solutions, our mechanism keeps running in isolation without the interaction of a user or a service provider. We perform rigorous experiments on benchmark datasets and show that our proposed mechanism limits user tracking and distinguishability threats to a significant extent compared to the original data while maintaining a reasonable level of utility of functionalities. In general, we show that our obfuscation mechanism reduces the user trackability threat by 60\% across all the datasets while maintaining the utility loss below 0.5 Mean Absolute Error (MAE). We also observe that our mechanism is more effective in large datasets. For example, with the Swipes dataset, the distinguishability risk is reduced by 60\% on average while the utility loss is below 0.5 MAE.

摘要: 嵌入到移动智能设备中的传感器可以高精度地监控用户的活动，为最终用户提供从精确地理定位、健康监测到手写单词识别的各种服务。然而，这涉及到访问并可能向应用程序泄露个人敏感信息的风险，这可能会导致隐私被侵犯。在本文中，我们的目标是在保持应用程序和服务的功能的同时，通过用户跟踪和区分将可能导致移动设备上的用户身份识别的隐私泄漏降至最低。我们提出了一种隐私保护机制，通过将传感器数据描述为时间序列建模和预测，有效地处理了传感器数据的波动(例如，不同时间行走、坐着和跑步时传感器读数的不一致)。该机制还使用了相关噪声序列的概念来抵抗来自对手的噪声过滤攻击，其目的是从扰动数据中滤除噪声，以重新识别原始数据。与现有的解决方案不同，我们的机制在没有用户或服务提供商交互的情况下保持隔离运行。我们在基准数据集上进行了严格的实验，结果表明，与原始数据相比，我们提出的机制在很大程度上限制了用户跟踪和区分威胁，同时保持了合理的功能效用水平。总体而言，我们的混淆机制在将效用损失保持在0.5个平均绝对误差(MAE)以下的同时，将所有数据集的用户可跟踪性威胁降低了60%。我们还观察到，我们的机制在大型数据集上更有效。例如，使用SWIPES数据集，当效用损失低于0.5MAE时，可区分性风险平均降低60%。



## **30. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

对基于投影的可取消生物识别方案的身份验证攻击(长版) cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2110.15163v5)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **31. Uncertify: Attacks Against Neural Network Certification**

未认证：针对神经网络认证的攻击 cs.LG

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2108.11299v3)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstracts**: A key concept towards reliable, robust, and safe AI systems is the idea to implement fallback strategies when predictions of the AI cannot be trusted. Certifiers for neural networks have made great progress towards provable robustness guarantees against evasion attacks using adversarial examples. These methods guarantee for some predictions that a certain class of manipulations or attacks could not have changed the outcome. For the remaining predictions without guarantees, the method abstains from making a prediction and a fallback strategy needs to be invoked, which is typically more costly, less accurate, or even involves a human operator. While this is a key concept towards safe and secure AI, we show for the first time that this strategy comes with its own security risks, as such fallback strategies can be deliberately triggered by an adversary. In particular, we conduct the first systematic analysis of training-time attacks against certifiers in practical application pipelines, identifying new threat vectors that can be exploited to degrade the overall system. Using these insights, we design two backdoor attacks against network certifiers, which can drastically reduce certified robustness. For example, adding 1% poisoned data during training is sufficient to reduce certified robustness by up to 95 percentage points, effectively rendering the certifier useless. We analyze how such novel attacks can compromise the overall system's integrity or availability. Our extensive experiments across multiple datasets, model architectures, and certifiers demonstrate the wide applicability of these attacks. A first investigation into potential defenses shows that current approaches are insufficient to mitigate the issue, highlighting the need for new, more specific solutions.

摘要: 迈向可靠、健壮和安全的人工智能系统的一个关键概念是，当人工智能的预测不可信时，实施后备策略。神经网络的认证器已经取得了很大的进展，通过使用对抗性的例子来证明对逃避攻击的健壮性保证。这些方法保证了某些预测，即某一类操纵或攻击不可能改变结果。对于没有保证的其余预测，该方法放弃进行预测，并且需要调用后备策略，这通常更昂贵、更不准确，甚至涉及人工操作员。虽然这是一个安全可靠的人工智能的关键概念，但我们第一次表明，这一战略带有自身的安全风险，因为这种后备战略可能会被对手故意触发。特别是，我们首次对实际应用管道中针对认证器的训练时间攻击进行了系统分析，识别了可以用来降低整个系统性能的新威胁向量。利用这些见解，我们设计了两种针对网络认证者的后门攻击，它们可以极大地降低认证的健壮性。例如，在训练期间添加1%的有毒数据就足以将认证的健壮性降低高达95个百分点，从而有效地使认证器毫无用处。我们分析了这种新的攻击如何危害整个系统的完整性或可用性。我们在多个数据集、模型体系结构和认证器上的广泛实验证明了这些攻击的广泛适用性。对潜在防御措施的首次调查显示，目前的方法不足以缓解这一问题，这突显了需要新的、更具体的解决方案。



## **32. Millimeter-Wave Automotive Radar Spoofing**

毫米波汽车雷达欺骗 cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06567v1)

**Authors**: Mihai Ordean, Flavio D. Garcia

**Abstracts**: Millimeter-wave radar systems are one of the core components of the safety-critical Advanced Driver Assistant System (ADAS) of a modern vehicle. Due to their ability to operate efficiently despite bad weather conditions and poor visibility, they are often the only reliable sensor a car has to detect and evaluate potential dangers in the surrounding environment. In this paper, we propose several attacks against automotive radars for the purposes of assessing their reliability in real-world scenarios. Using COTS hardware, we are able to successfully interfere with automotive-grade FMCW radars operating in the commonly used 77GHz frequency band, deployed in real-world, truly wireless environments. Our strongest type of interference is able to trick the victim into detecting virtual (moving) objects. We also extend this attack with a novel method that leverages noise to remove real-world objects, thus complementing the aforementioned object spoofing attack. We evaluate the viability of our attacks in two ways. First, we establish a baseline by implementing and evaluating an unrealistically powerful adversary which requires synchronization to the victim in a limited setup that uses wire-based chirp synchronization. Later, we implement, for the first time, a truly wireless attack that evaluates a weaker but realistic adversary which is non-synchronized and does not require any adjustment feedback from the victim. Finally, we provide theoretical fundamentals for our findings, and discuss the efficiency of potential countermeasures against the proposed attacks. We plan to release our software as open-source.

摘要: 毫米波雷达系统是现代车辆安全关键的高级驾驶员辅助系统(ADAS)的核心部件之一。由于它们能够在恶劣的天气条件和低能见度的情况下高效运行，它们往往是汽车检测和评估周围环境潜在危险的唯一可靠传感器。在本文中，我们提出了几种针对汽车雷达的攻击，目的是评估它们在现实世界场景中的可靠性。使用COTS硬件，我们能够成功干扰在常用的77 GHz频段运行的车载级FMCW雷达，部署在真实世界、真正的无线环境中。我们最强的干扰类型是能够诱骗受害者检测虚拟(移动)对象。我们还用一种新的方法来扩展这种攻击，该方法利用噪声来移除真实世界的对象，从而补充了前面提到的对象欺骗攻击。我们通过两种方式评估我们的攻击的可行性。首先，我们通过实现和评估一个不现实的强大对手来建立基准，该对手需要在使用有线chirp同步的有限设置中与受害者同步。后来，我们第一次实现了一个真正的无线攻击，它评估一个较弱但现实的对手，它是非同步的，不需要受害者提供任何调整反馈。最后，我们为我们的发现提供了理论基础，并讨论了针对拟议攻击的潜在对策的效率。我们计划将我们的软件作为开源软件发布。



## **33. l-Leaks: Membership Inference Attacks with Logits**

L-泄漏：带Logit的成员关系推断攻击 cs.LG

10pages,6figures

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2205.06469v1)

**Authors**: Shuhao Li, Yajie Wang, Yuanzhang Li, Yu-an Tan

**Abstracts**: Machine Learning (ML) has made unprecedented progress in the past several decades. However, due to the memorability of the training data, ML is susceptible to various attacks, especially Membership Inference Attacks (MIAs), the objective of which is to infer the model's training data. So far, most of the membership inference attacks against ML classifiers leverage the shadow model with the same structure as the target model. However, empirical results show that these attacks can be easily mitigated if the shadow model is not clear about the network structure of the target model.   In this paper, We present attacks based on black-box access to the target model. We name our attack \textbf{l-Leaks}. The l-Leaks follows the intuition that if an established shadow model is similar enough to the target model, then the adversary can leverage the shadow model's information to predict a target sample's membership.The logits of the trained target model contain valuable sample knowledge. We build the shadow model by learning the logits of the target model and making the shadow model more similar to the target model. Then shadow model will have sufficient confidence in the member samples of the target model. We also discuss the effect of the shadow model's different network structures to attack results. Experiments over different networks and datasets demonstrate that both of our attacks achieve strong performance.

摘要: 在过去的几十年里，机器学习取得了前所未有的进步。然而，由于训练数据的记忆性，ML容易受到各种攻击，尤其是成员关系推理攻击(MIA)，其目的是推断模型的训练数据。到目前为止，针对ML分类器的成员关系推理攻击大多利用与目标模型具有相同结构的影子模型。然而，实验结果表明，如果影子模型不清楚目标模型的网络结构，则可以很容易地缓解这些攻击。在本文中，我们提出了基于黑盒访问目标模型的攻击。我们将我们的攻击命名为\extbf{l-leaks}。L-泄漏遵循这样的直觉，即如果建立的阴影模型与目标模型足够相似，则对手可以利用阴影模型的信息来预测目标样本的成员资格。训练后的目标模型的逻辑包含有价值的样本知识。我们通过学习目标模型的逻辑并使阴影模型更接近目标模型来构建阴影模型。那么影子模型将对目标模型的成员样本具有足够的置信度。讨论了影子模型的不同网络结构对攻击结果的影响。在不同的网络和数据集上的实验表明，我们的两种攻击都取得了很好的性能。



## **34. Bitcoin's Latency--Security Analysis Made Simple**

比特币的潜伏期--安全分析变得简单 cs.CR

**SubmitDate**: 2022-05-13    [paper-pdf](http://arxiv.org/pdf/2203.06357v2)

**Authors**: Dongning Guo, Ling Ren

**Abstracts**: Simple closed-form upper and lower bounds are developed for the security of the Nakamoto consensus as a function of the confirmation depth, the honest and adversarial block mining rates, and an upper bound on the block propagation delay. The bounds are exponential in the confirmation depth and apply regardless of the adversary's attack strategy. The gap between the upper and lower bounds is small for Bitcoin's parameters. For example, assuming an average block interval of 10 minutes, a network delay bound of ten seconds, and 10% adversarial mining power, the widely used 6-block confirmation rule yields a safety violation between 0.11% and 0.35% probability.

摘要: 对于Nakamoto共识的安全性，给出了简单的闭合上下界，作为确认深度、诚实和对抗性块挖掘率的函数，以及块传播延迟的上界。这些界限在确认深度上是指数级的，无论对手的攻击策略如何，都适用。就比特币的参数而言，上下限之间的差距很小。例如，假设平均阻塞间隔为10分钟，网络延迟界限为10秒，对抗性挖掘能力为10%，则广泛使用的6-块确认规则产生的安全违规概率在0.11%到0.35%之间。



## **35. How to Combine Membership-Inference Attacks on Multiple Updated Models**

如何在多个更新的模型上组合成员推理攻击 cs.LG

31 pages, 9 figures

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06369v1)

**Authors**: Matthew Jagielski, Stanley Wu, Alina Oprea, Jonathan Ullman, Roxana Geambasu

**Abstracts**: A large body of research has shown that machine learning models are vulnerable to membership inference (MI) attacks that violate the privacy of the participants in the training data. Most MI research focuses on the case of a single standalone model, while production machine-learning platforms often update models over time, on data that often shifts in distribution, giving the attacker more information. This paper proposes new attacks that take advantage of one or more model updates to improve MI. A key part of our approach is to leverage rich information from standalone MI attacks mounted separately against the original and updated models, and to combine this information in specific ways to improve attack effectiveness. We propose a set of combination functions and tuning methods for each, and present both analytical and quantitative justification for various options. Our results on four public datasets show that our attacks are effective at using update information to give the adversary a significant advantage over attacks on standalone models, but also compared to a prior MI attack that takes advantage of model updates in a related machine-unlearning setting. We perform the first measurements of the impact of distribution shift on MI attacks with model updates, and show that a more drastic distribution shift results in significantly higher MI risk than a gradual shift. Our code is available at https://www.github.com/stanleykywu/model-updates.

摘要: 大量研究表明，机器学习模型容易受到成员推理(MI)攻击，这些攻击侵犯了训练数据中参与者的隐私。大多数MI研究集中在单个独立模型的情况下，而生产型机器学习平台经常随着时间的推移更新模型，更新数据的分布经常发生变化，为攻击者提供更多信息。本文提出了利用一个或多个模型更新来改进MI的新攻击。我们方法的一个关键部分是利用来自独立MI攻击的丰富信息，针对原始和更新的模型单独安装，并以特定的方式组合这些信息以提高攻击效率。我们提出了一套组合函数和调整方法，并对不同的选项进行了分析和定量论证。我们在四个公共数据集上的结果表明，我们的攻击在使用更新信息为对手提供显著优势方面比对独立模型的攻击具有显著优势，但也比之前的MI攻击在相关的机器遗忘环境中利用模型更新的优势更大。我们通过模型更新首次测量了分布漂移对MI攻击的影响，并表明更剧烈的分布漂移导致的MI风险显著高于渐变。我们的代码可以在https://www.github.com/stanleykywu/model-updates.上找到



## **36. Anomaly Detection of Adversarial Examples using Class-conditional Generative Adversarial Networks**

基于类条件生成对抗性网络的对抗性实例异常检测 cs.LG

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2105.10101v2)

**Authors**: Hang Wang, David J. Miller, George Kesidis

**Abstracts**: Deep Neural Networks (DNNs) have been shown vulnerable to Test-Time Evasion attacks (TTEs, or adversarial examples), which, by making small changes to the input, alter the DNN's decision. We propose an unsupervised attack detector on DNN classifiers based on class-conditional Generative Adversarial Networks (GANs). We model the distribution of clean data conditioned on the predicted class label by an Auxiliary Classifier GAN (AC-GAN). Given a test sample and its predicted class, three detection statistics are calculated based on the AC-GAN Generator and Discriminator. Experiments on image classification datasets under various TTE attacks show that our method outperforms previous detection methods. We also investigate the effectiveness of anomaly detection using different DNN layers (input features or internal-layer features) and demonstrate, as one might expect, that anomalies are harder to detect using features closer to the DNN's output layer.

摘要: 深度神经网络(DNN)已经被证明容易受到测试时间逃避攻击(TTE，或对抗性例子)，这些攻击通过对输入进行微小的改变来改变DNN的决策。提出了一种基于类别条件生成对抗网络(GANS)的DNN分类器的无监督攻击检测器。我们以辅助分类器GaN(AC-GaN)预测的类别标签为条件，对清洁数据的分布进行建模。在给定测试样本及其预测类别的情况下，基于AC-GaN生成器和鉴别器计算了三个检测统计量。在不同TTE攻击下的图像分类数据集上的实验表明，该方法的性能优于以往的检测方法。我们还研究了使用不同的DNN层(输入特征或内部层特征)进行异常检测的有效性，并证明了，正如人们所预期的那样，使用离DNN输出层更近的特征更难检测到异常。



## **37. Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks**

抗规避攻击的稳健学习决策表的样本复杂性界 cs.LG

To appear in the proceedings of International Joint Conference on  Artificial Intelligence (2022)

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06127v1)

**Authors**: Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell

**Abstracts**: A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed $k$ the class of $k$-decision lists has polynomial sample complexity against a $\log(n)$-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient $\log(n)$-robust learning algorithm under the uniform distribution.

摘要: 对抗性机器学习中的一个基本问题是量化在存在逃避攻击的情况下需要多少训练数据。在本文中，我们在PAC学习的框架内解决这个问题，重点是决策列表的类。鉴于分布假设在对抗性环境中是必不可少的，我们在满足Lipschitz条件的输入数据上使用概率分布：邻近的点具有类似的概率。我们的关键结果表明，对手的预算(即，它可以在每一次输入上扰动的比特数)是决定稳健学习的样本复杂性的基本量。我们的第一个主要结果是一个样本复杂性下界：单调合取类(本质上是布尔超立方体上最简单的非平凡假设类)和任何超类在对手的预算中至少具有指数级的样本复杂性。我们的第二个主要结果是相应的上界：对于每一个固定的$k$，对于$\log(N)$有界的对手，这类$k$-决策列表具有多项式样本复杂性。这进一步揭示了在均匀分布下，有效的PAC学习算法是否总是可以用作有效的$\log(N)$稳健学习算法的问题。



## **38. From IP to transport and beyond: cross-layer attacks against applications**

从IP到传输乃至更远：针对应用程序的跨层攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06085v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: We perform the first analysis of methodologies for launching DNS cache poisoning: manipulation at the IP layer, hijack of the inter-domain routing and probing open ports via side channels. We evaluate these methodologies against DNS resolvers in the Internet and compare them with respect to effectiveness, applicability and stealth. Our study shows that DNS cache poisoning is a practical and pervasive threat.   We then demonstrate cross-layer attacks that leverage DNS cache poisoning for attacking popular systems, ranging from security mechanisms, such as RPKI, to applications, such as VoIP. In addition to more traditional adversarial goals, most notably impersonation and Denial of Service, we show for the first time that DNS cache poisoning can even enable adversaries to bypass cryptographic defences: we demonstrate how DNS cache poisoning can facilitate BGP prefix hijacking of networks protected with RPKI even when all the other networks apply route origin validation to filter invalid BGP announcements. Our study shows that DNS plays a much more central role in the Internet security than previously assumed.   We recommend mitigations for securing the applications and for preventing cache poisoning.

摘要: 我们对发起DNS缓存中毒的方法进行了第一次分析：在IP层操纵、劫持域间路由和通过侧通道探测开放端口。我们针对互联网中的域名解析程序对这些方法进行评估，并在有效性、适用性和隐蔽性方面对它们进行比较。我们的研究表明，DNS缓存中毒是一种实际且普遍存在的威胁。然后，我们演示了利用DNS缓存毒化来攻击流行系统的跨层攻击，攻击范围从安全机制(如RPKI)到应用程序(如VoIP)。除了更传统的敌意目标之外，最显著的是模拟和拒绝服务，我们首次展示了DNS缓存中毒甚至可以使攻击者绕过加密防御：我们演示了DNS缓存中毒如何促进对受RPKI保护的网络的BGP前缀劫持，即使所有其他网络都应用路由来源验证来过滤无效的BGP通告。我们的研究表明，在互联网安全中，域名系统扮演的角色比之前设想的要重要得多。我们建议采取缓解措施来保护应用程序和防止缓存中毒。



## **39. Segmentation-Consistent Probabilistic Lesion Counting**

分割一致的概率病变计数 eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2204.05276v2)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.

摘要: 病灶计数是疾病严重程度、患者预后和治疗效果的重要指标，但在医学成像中，作为一项任务的计数往往被忽视，而有利于分割。这项工作引入了一种新的连续可微函数，它以一致的方式将病变分割预测映射到病变计数概率分布。所提出的端到端方法--包括体素聚类、病变级体素概率聚合和泊松二项计数--是非参数的，因此提供了一种稳健且一致的方法来增强具有后自组织计数能力的病变分割模型。对Gd增强病变计数的实验表明，我们的方法输出准确且校准良好的计数分布，捕捉到有意义的不确定信息。结果还表明，该模型适用于病变分割的多任务学习，在低数据量环境下是有效的，并且对敌意攻击具有较强的鲁棒性。



## **40. Stalloris: RPKI Downgrade Attack**

Stalloris：RPKI降级攻击 cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06064v1)

**Authors**: Tomas Hlavacek, Philipp Jeitner, Donika Mirdita, Haya Shulman, Michael Waidner

**Abstracts**: We demonstrate the first downgrade attacks against RPKI. The key design property in RPKI that allows our attacks is the tradeoff between connectivity and security: when networks cannot retrieve RPKI information from publication points, they make routing decisions in BGP without validating RPKI. We exploit this tradeoff to develop attacks that prevent the retrieval of the RPKI objects from the public repositories, thereby disabling RPKI validation and exposing the RPKI-protected networks to prefix hijack attacks.   We demonstrate experimentally that at least 47% of the public repositories are vulnerable against a specific version of our attacks, a rate-limiting off-path downgrade attack. We also show that all the current RPKI relying party implementations are vulnerable to attacks by a malicious publication point. This translates to 20.4% of the IPv4 address space.   We provide recommendations for preventing our downgrade attacks. However, resolving the fundamental problem is not straightforward: if the relying parties prefer security over connectivity and insist on RPKI validation when ROAs cannot be retrieved, the victim AS may become disconnected from many more networks than just the one that the adversary wishes to hijack. Our work shows that the publication points are a critical infrastructure for Internet connectivity and security. Our main recommendation is therefore that the publication points should be hosted on robust platforms guaranteeing a high degree of connectivity.

摘要: 我们演示了针对RPKI的第一次降级攻击。RPKI中允许我们攻击的关键设计属性是连接性和安全性之间的权衡：当网络无法从发布点检索RPKI信息时，它们在BGP中做出路由决定，而不验证RPKI。我们利用这一权衡来开发攻击，以阻止从公共存储库中检索RPKI对象，从而禁用RPKI验证并使受RPKI保护的网络暴露于前缀劫持攻击。我们通过实验证明，至少47%的公共存储库容易受到我们的特定版本的攻击，这是一种限速的非路径降级攻击。我们还表明，所有当前的RPKI依赖方实现都容易受到恶意发布点的攻击。这相当于IPv4地址空间的20.4%。我们提供了防止降级攻击的建议。然而，解决根本问题并不简单：如果依赖方更看重安全而不是连接，并在无法检索ROA时坚持RPKI验证，受害者AS可能会断开与更多网络的连接，而不仅仅是对手希望劫持的网络。我们的工作表明，发布点是互联网连接和安全的关键基础设施。因此，我们的主要建议是，发布点应设在保证高度连通性的强大平台上。



## **41. Infrared Invisible Clothing:Hiding from Infrared Detectors at Multiple Angles in Real World**

红外隐身衣：在现实世界中从多个角度躲避红外探测器 cs.CV

Accepted by CVPR 2022, ORAL

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.05909v1)

**Authors**: Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, Xiaolin Hu

**Abstracts**: Thermal infrared imaging is widely used in body temperature measurement, security monitoring, and so on, but its safety research attracted attention only in recent years. We proposed the infrared adversarial clothing, which could fool infrared pedestrian detectors at different angles. We simulated the process from cloth to clothing in the digital world and then designed the adversarial "QR code" pattern. The core of our method is to design a basic pattern that can be expanded periodically, and make the pattern after random cropping and deformation still have an adversarial effect, then we can process the flat cloth with an adversarial pattern into any 3D clothes. The results showed that the optimized "QR code" pattern lowered the Average Precision (AP) of YOLOv3 by 87.7%, while the random "QR code" pattern and blank pattern lowered the AP of YOLOv3 by 57.9% and 30.1%, respectively, in the digital world. We then manufactured an adversarial shirt with a new material: aerogel. Physical-world experiments showed that the adversarial "QR code" pattern clothing lowered the AP of YOLOv3 by 64.6%, while the random "QR code" pattern clothing and fully heat-insulated clothing lowered the AP of YOLOv3 by 28.3% and 22.8%, respectively. We used the model ensemble technique to improve the attack transferability to unseen models.

摘要: 热红外成像广泛应用于体温测量、安防监测等领域，但其安全性研究直到最近几年才引起人们的重视。我们提出了红外防御服，可以从不同角度欺骗红外行人探测器。我们模拟了数字世界中从布料到衣物的过程，然后设计了对抗性的“二维码”图案。该方法的核心是设计一种可周期性扩展的基本图案，使任意裁剪和变形后的图案仍然具有对抗效果，从而可以将带有对抗图案的平面布加工成任何3D服装。结果表明，在数字世界中，优化的二维码模式使YOLOv3的平均准确率下降了87.7%，而随机的二维码模式和空白模式分别使YOLOv3的平均准确率下降了57.9%和30.1%。然后，我们用一种新材料制作了一件对抗性衬衫：气凝胶。实物实验表明，对抗性的“二维码”图案服装使YOLOv3的AP降低了64.6%，而随机的“二维码”图案服装和完全隔热的服装分别使YOLOv3的AP降低了28.3%和22.8%。我们使用模型集成技术来提高攻击到不可见模型的可转移性。



## **42. Using Frequency Attention to Make Adversarial Patch Powerful Against Person Detector**

利用频率注意使对抗性补丁成为强大的抗人检测器 cs.CV

10pages, 4 figures

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.04638v2)

**Authors**: Xiaochun Lei, Chang Lu, Zetao Jiang, Zhaoting Gong, Xiang Cai, Linjun Lu

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, object detectors may be attacked by applying a particular adversarial patch to the image. However, because the patch shrinks during preprocessing, most existing approaches that employ adversarial patches to attack object detectors would diminish the attack success rate on small and medium targets. This paper proposes a Frequency Module(FRAN), a frequency-domain attention module for guiding patch generation. This is the first study to introduce frequency domain attention to optimize the attack capabilities of adversarial patches. Our method increases the attack success rates of small and medium targets by 4.18% and 3.89%, respectively, over the state-of-the-art attack method for fooling the human detector while assaulting YOLOv3 without reducing the attack success rate of big targets.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。具体地，可以通过将特定的敌意补丁应用于图像来攻击对象检测器。然而，由于补丁在预处理过程中会缩小，现有的大多数使用对抗性补丁攻击目标检测器的方法都会降低对中小目标的攻击成功率。提出了一种用于指导补丁生成的频域注意模块FRAN。这是首次引入频域注意力来优化敌方补丁攻击能力的研究。该方法在不降低大目标攻击成功率的前提下，将小目标和中型目标的攻击成功率分别提高了4.18%和3.89%。



## **43. The Hijackers Guide To The Galaxy: Off-Path Taking Over Internet Resources**

《银河劫机者指南：越轨接管互联网资源》 cs.CR

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.05473v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: Internet resources form the basic fabric of the digital society. They provide the fundamental platform for digital services and assets, e.g., for critical infrastructures, financial services, government. Whoever controls that fabric effectively controls the digital society.   In this work we demonstrate that the current practices of Internet resources management, of IP addresses, domains, certificates and virtual platforms are insecure. Over long periods of time adversaries can maintain control over Internet resources which they do not own and perform stealthy manipulations, leading to devastating attacks. We show that network adversaries can take over and manipulate at least 68% of the assigned IPv4 address space as well as 31% of the top Alexa domains. We demonstrate such attacks by hijacking the accounts associated with the digital resources.   For hijacking the accounts we launch off-path DNS cache poisoning attacks, to redirect the password recovery link to the adversarial hosts. We then demonstrate that the adversaries can manipulate the resources associated with these accounts. We find all the tested providers vulnerable to our attacks.   We recommend mitigations for blocking the attacks that we present in this work. Nevertheless, the countermeasures cannot solve the fundamental problem - the management of the Internet resources should be revised to ensure that applying transactions cannot be done so easily and stealthily as is currently possible.

摘要: 互联网资源构成了数字社会的基本结构。它们为数字服务和资产提供基础平台，例如关键基础设施、金融服务、政府。无论谁控制了这种结构，谁就有效地控制了数字社会。在这项工作中，我们证明了当前互联网资源管理的做法，即IP地址、域、证书和虚拟平台是不安全的。在很长一段时间内，对手可以保持对他们不拥有的互联网资源的控制，并执行秘密操作，导致毁灭性的攻击。我们发现，网络攻击者可以接管和操纵至少68%的分配的IPv4地址空间以及31%的顶级Alexa域。我们通过劫持与数字资源相关的帐户来演示此类攻击。对于劫持帐户，我们发起非路径的DNS缓存中毒攻击，将密码恢复链接重定向到恶意主机。然后，我们将演示对手可以操纵与这些帐户关联的资源。我们发现所有经过测试的供应商都容易受到我们的攻击。我们建议采取缓解措施来阻止我们在此工作中提出的攻击。然而，这些对策不能解决根本问题--对互联网资源的管理应加以修订，以确保申请交易不能像目前那样容易和秘密地进行。



## **44. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2204.08189v2)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **45. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

利用计算机视觉技术开发隐形敌方补丁来伪装军事资产 cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2202.08892v2)

**Authors**: Chris Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.

摘要: 卷积神经网络(CNN)在目标检测方面取得了快速的进展和很高的成功率。然而，最近的证据突显了它们在对抗性攻击中的脆弱性。这些攻击是经过计算的图像扰动或对抗性补丁，导致目标错误分类或检测抑制。传统的伪装方法用于伪装飞机和其他大型机动资产，使其免受情报、监视和侦察技术以及第五代导弹的自主探测，是不切实际的。在这篇文章中，我们提出了一种独特的方法，可以从计算机视觉启用的技术中产生能够伪装大型军事资产的隐形补丁。我们开发了这些补丁，通过最大化目标检测损失，同时限制补丁的颜色敏感度。这项工作还旨在进一步理解对抗性例子及其对目标检测算法的影响。



## **46. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

一句话抵得上一千美元：敌意攻击推特傻瓜股预测 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.01094v2)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **47. Privacy Enhancement for Cloud-Based Few-Shot Learning**

增强基于云的极少机会学习的隐私 cs.LG

14 pages, 13 figures, 3 tables. Preprint. Accepted in IEEE WCCI 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.07864v1)

**Authors**: Archit Parnami, Muhammad Usama, Liyue Fan, Minwoo Lee

**Abstracts**: Requiring less data for accurate models, few-shot learning has shown robustness and generality in many application domains. However, deploying few-shot models in untrusted environments may inflict privacy concerns, e.g., attacks or adversaries that may breach the privacy of user-supplied data. This paper studies the privacy enhancement for the few-shot learning in an untrusted environment, e.g., the cloud, by establishing a novel privacy-preserved embedding space that preserves the privacy of data and maintains the accuracy of the model. We examine the impact of various image privacy methods such as blurring, pixelization, Gaussian noise, and differentially private pixelization (DP-Pix) on few-shot image classification and propose a method that learns privacy-preserved representation through the joint loss. The empirical results show how privacy-performance trade-off can be negotiated for privacy-enhanced few-shot learning.

摘要: 对于精确的模型，少镜头学习需要较少的数据，在许多应用领域都表现出了健壮性和通用性。然而，在不受信任的环境中部署极少的模型可能会引起隐私问题，例如，可能会破坏用户提供的数据的隐私的攻击或对手。通过建立一种新的隐私保护嵌入空间来保护数据隐私并保持模型的准确性，研究了在不可信环境(如云)下的少机会学习的隐私增强问题。研究了模糊、像素化、高斯噪声、差分隐私像素化等图像隐私保护方法对少镜头图像分类的影响，提出了一种通过联合损失学习隐私保护表示的方法。实证结果表明，对于隐私增强型少镜头学习，隐私性能与性能之间的权衡是如何协商的。



## **48. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

SYNFI：一种开源安全元件的硅前故障分析 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.04775v1)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.

摘要: 故障攻击是一种主动的物理攻击，攻击者可以利用这些攻击来改变嵌入式设备的控制流，从而获得对敏感信息的访问权限或绕过保护机制。由于这些攻击的严重性，制造商将基于硬件的故障防御部署到安全关键系统中，例如安全元件。这些对策的开发是一项具有挑战性的任务，因为电路元件之间的复杂相互作用，以及现代设计自动化工具倾向于优化插入的结构，从而违背了它们的目的。因此，至关重要的是，这些对策在合成后得到严格验证。由于传统的功能验证技术无法评估对策的有效性，开发人员不得不求助于能够在模拟测试台或物理芯片中注入故障的方法。然而，开发测试序列以在模拟中注入故障是一项容易出错的任务，在芯片上执行故障攻击需要专门的设备，并且非常耗时。为此，本文引入了SYNFI，这是一个运行在合成网表上的形式化的预硅故障验证框架。SYNFI可以用来分析故障对电路输入输出关系的一般影响及其故障对策，从而使硬件设计者能够以系统和半自动的方式评估和验证嵌入式对策的有效性。为了证明SYNFI能够处理使用商业和开放工具合成的未经修改的工业级网表，我们分析了第一个开源安全元素OpenTitan。在我们的分析中，我们确定了未受保护的AES块中的关键安全漏洞，开发了有针对性的对策，重新评估了它们的安全性，并将这些对策贡献给了OpenTitan存储库。



## **49. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

基于后向误差分析的联合学习半目标模型中毒攻击 cs.LG

Published in IJCNN 2022

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.11633v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.

摘要: 针对联邦学习(FL)的模型中毒攻击通过破坏边缘模型来侵入整个系统，导致机器学习模型故障。这种被破坏的模型被篡改，以执行对手所希望的行为。特别是，我们考虑了一种半目标的情况，其中源类是预先确定的，而目标类不是。其目的是使全局分类器对源类的数据进行错误分类。虽然已经采用了标签翻转等方法向FL注入有毒参数，但研究表明，它们的性能通常是类敏感的，随着所使用的目标类的不同而变化。通常，当转移到不同的目标类别时，攻击可能会变得不那么有效。为了克服这一挑战，我们提出了攻击距离感知攻击(ADA)，通过在特征空间中找到优化的目标类来增强中毒攻击。此外，我们研究了一种更具挑战性的情况，即对手对客户数据的先验知识有限。为了解决这一问题，ADA基于向后误差分析，从共享的模型参数中推导出潜在特征空间中不同类别之间的成对距离。我们通过在三种不同的图像分类任务中改变攻击频率的因素，对ADA进行了广泛的经验评估。结果，在最具挑战性的情况下，ADA成功地将攻击性能提高了1.8倍，攻击频率为0.01。



## **50. Fingerprinting of DNN with Black-box Design and Verification**

基于黑盒设计和验证的DNN指纹识别 cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.10902v3)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Ruoxi Sun, Minhui Xue, Surya Nepal, Seyit Camtepe, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.

摘要: 支持云的机器学习即服务(MLaaS)显示出巨大的潜力，可以改变深度学习模型的开发和部署方式。尽管如此，使用此类服务仍存在潜在风险，因为恶意方可能会对其进行修改以达到不利的结果。因此，模型所有者、服务提供商和最终用户必须验证部署的模型是否未被篡改。这样的验证需要公开的可验证性(即，指纹模式可供各方使用，包括对手)，并需要通过API对部署的模型进行黑盒访问。然而，现有的水印和指纹方法需要白盒知识(如梯度)来设计指纹，并且只支持私密可验证性，即由诚实的一方进行验证。在本文中，我们描述了一种实用的水印技术，该技术能够在指纹设计中提供黑盒知识，并在验证过程中提供黑盒查询。该服务通过公开验证来确保基于云的服务的完整性(即指纹模式可供各方使用，包括对手)。如果对手操纵了一个模型，这将导致决策边界的转变。因此，双黑水印的基本原理是，模型的决策边界可以作为水印的固有指纹。我们的方法通过生成有限数量的包络样本指纹来捕获决策边界，这些样本指纹是围绕模型决策边界的一组自然转换和扩充的输入，以捕获模型的固有指纹。我们针对各种模型完整性攻击和模型压缩攻击对我们的水印方法进行了评估。



