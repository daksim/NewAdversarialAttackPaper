# Latest Adversarial Attack Papers
**update at 2023-01-14 11:59:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Phase-shifted Adversarial Training**

相移对抗性训练 cs.LG

**SubmitDate**: 2023-01-12    [abs](http://arxiv.org/abs/2301.04785v1) [paper-pdf](http://arxiv.org/pdf/2301.04785v1)

**Authors**: Yeachan Kim, Seongyeon Kim, Ihyeok Seo, Bonggun Shin

**Abstract**: Adversarial training has been considered an imperative component for safely deploying neural network-based applications to the real world. To achieve stronger robustness, existing methods primarily focus on how to generate strong attacks by increasing the number of update steps, regularizing the models with the smoothed loss function, and injecting the randomness into the attack. Instead, we analyze the behavior of adversarial training through the lens of response frequency. We empirically discover that adversarial training causes neural networks to have low convergence to high-frequency information, resulting in highly oscillated predictions near each data. To learn high-frequency contents efficiently and effectively, we first prove that a universal phenomenon of frequency principle, i.e., \textit{lower frequencies are learned first}, still holds in adversarial training. Based on that, we propose phase-shifted adversarial training (PhaseAT) in which the model learns high-frequency components by shifting these frequencies to the low-frequency range where the fast convergence occurs. For evaluations, we conduct the experiments on CIFAR-10 and ImageNet with the adaptive attack carefully designed for reliable evaluation. Comprehensive results show that PhaseAT significantly improves the convergence for high-frequency information. This results in improved adversarial robustness by enabling the model to have smoothed predictions near each data.

摘要: 对抗性训练一直被认为是将基于神经网络的应用安全地部署到现实世界中的一个必不可少的组成部分。为了获得更强的稳健性，现有的方法主要集中在如何通过增加更新步骤、用平滑的损失函数对模型进行正则化以及在攻击中注入随机性来产生强攻击。相反，我们通过反应频率的镜头来分析对抗性训练的行为。我们经验发现，对抗性训练导致神经网络对高频信息的收敛程度较低，导致每个数据附近的预测高度振荡。为了高效有效地学习高频内容，我们首先证明了频率原理中的一个普遍现象，即先学习较低的频率在对抗性训练中仍然成立。在此基础上，我们提出了相移对抗性训练(PhaseAT)，该模型通过将高频成分转移到发生快速收敛的低频范围来学习高频成分。在评估方面，我们在CIFAR-10和ImageNet上进行了实验，并使用精心设计的自适应攻击进行了可靠的评估。综合结果表明，PhaseAT算法显著提高了高频信息的收敛速度。这使得模型能够在每个数据附近平滑预测，从而提高了对手的稳健性。



## **2. Reducing Exploitability with Population Based Training**

通过基于人口的培训减少可利用性 cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2208.05083v3) [paper-pdf](http://arxiv.org/pdf/2208.05083v3)

**Authors**: Pavel Czempin, Adam Gleave

**Abstract**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We analyze a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. This defense increases robustness against adversaries, as measured by the number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.

摘要: 自我发挥强化学习在各种零和游戏中实现了最先进的，往往是超人的表现。然而，先前的工作已经发现，对常规对手具有高度能力的政策，可能会在对抗对手的政策上灾难性地失败：一个明确针对受害者的对手。使用对抗性训练的先前防御能够使受害者对特定的对手变得健壮，但受害者仍然容易受到新对手的攻击。我们推测，这一限制是由于训练过程中看到的对手多样性不足所致。我们使用基于人口的训练来分析防御，让受害者与不同的对手对抗。我们在两个低维环境中评估了该防御对新对手的健壮性。这种防御提高了对抗对手的健壮性，通过攻击者训练攻击受害者的时间步数来衡量。此外，我们还证明了健壮性与对手种群的大小相关。



## **3. Resynthesis-based Attacks Against Logic Locking**

基于再合成的逻辑锁定攻击 cs.CR

8 pages, 7 figures, conference

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2301.04400v1) [paper-pdf](http://arxiv.org/pdf/2301.04400v1)

**Authors**: F. Almeida, L. Aksoy, Q-L. Nguyen, S. Dupuis, M-L. Flottes, S. Pagliarini

**Abstract**: Logic locking has been a promising solution to many hardware security threats, such as intellectual property infringement and overproduction. Due to the increased attention that threats have received, many efficient specialized attacks against logic locking have been introduced over the years. However, the ability of an adversary to manipulate a locked netlist prior to mounting an attack has not been investigated thoroughly. This paper introduces a resynthesis-based strategy that utilizes the strength of a commercial electronic design automation (EDA) tool to reveal the vulnerabilities of a locked circuit. To do so, in a pre-attack step, a locked netlist is resynthesized using different synthesis parameters in a systematic way, leading to a large number of functionally equivalent but structurally different locked circuits. Then, under the oracle-less threat model, where it is assumed that the adversary only possesses the locked circuit, not the original circuit to query, a prominent attack is applied to these generated netlists collectively, from which a large number of key bits are deciphered. Nevertheless, this paper also describes how the proposed oracle-less attack can be integrated with an oracle-guided attack. The feasibility of the proposed approach is demonstrated for several benchmarks, including remarkable results for breaking a recently proposed provably secure logic locking method and deciphering values of a large number of key bits of the CSAW'19 circuits with very high accuracy.

摘要: 逻辑锁定是解决许多硬件安全威胁的一种很有前途的解决方案，例如侵犯知识产权和生产过剩。由于威胁受到越来越多的关注，多年来已经引入了许多针对逻辑锁定的高效专门攻击。然而，对手在发动攻击之前操纵锁定网表的能力尚未得到彻底调查。本文介绍了一种基于再综合的策略，该策略利用商业电子设计自动化(EDA)工具的优势来揭示锁定电路的脆弱性。为此，在攻击前步骤中，使用不同的合成参数以系统的方式重新合成锁定网表，导致大量功能等价但结构不同的锁定电路。然后，在无预言机威胁模型下，假设攻击者只拥有锁定的电路，而不是要查询的原始电路，对这些生成的网表进行显著攻击，从中解密大量的密钥位。然而，本文还描述了提出的无甲骨文攻击如何与甲骨文引导攻击相结合。在几个基准测试中验证了该方法的可行性，包括破解最近提出的可证明安全的逻辑锁定方法的显著结果，以及以非常高的精度解密CSAW‘19电路的大量密钥位。



## **4. Towards Backdoor Attacks and Defense in Robust Machine Learning Models**

稳健机器学习模型中的后门攻击与防御 cs.CV

Accepted in Computers & Security, 2023

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2003.00865v4) [paper-pdf](http://arxiv.org/pdf/2003.00865v4)

**Authors**: Ezekiel Soremekun, Sakshi Udeshi, Sudipta Chattopadhyay

**Abstract**: The introduction of robust optimisation has pushed the state-of-the-art in defending against adversarial attacks. Notably, the state-of-the-art projected gradient descent (PGD)-based training method has been shown to be universally and reliably effective in defending against adversarial inputs. This robustness approach uses PGD as a reliable and universal "first-order adversary". However, the behaviour of such optimisation has not been studied in the light of a fundamentally different class of attacks called backdoors. In this paper, we study how to inject and defend against backdoor attacks for robust models trained using PGD-based robust optimisation. We demonstrate that these models are susceptible to backdoor attacks. Subsequently, we observe that backdoors are reflected in the feature representation of such models. Then, this observation is leveraged to detect such backdoor-infected models via a detection technique called AEGIS. Specifically, given a robust Deep Neural Network (DNN) that is trained using PGD-based first-order adversarial training approach, AEGIS uses feature clustering to effectively detect whether such DNNs are backdoor-infected or clean.   In our evaluation of several visible and hidden backdoor triggers on major classification tasks using CIFAR-10, MNIST and FMNIST datasets, AEGIS effectively detects PGD-trained robust DNNs infected with backdoors. AEGIS detects such backdoor-infected models with 91.6% accuracy (11 out of 12 tested models), without any false positives. Furthermore, AEGIS detects the targeted class in the backdoor-infected model with a reasonably low (11.1%) false positive rate. Our investigation reveals that salient features of adversarially robust DNNs could be promising to break the stealthy nature of backdoor attacks.

摘要: 强健优化的引入推动了防御对手攻击的最先进水平。值得注意的是，基于最先进的投影梯度下降(PGD)的训练方法已被证明在防御对手输入方面普遍而可靠地有效。这种健壮性方法使用PGD作为可靠且通用的“一阶对手”。然而，这种优化的行为还没有从一种被称为后门的根本不同的攻击类别中进行研究。在本文中，我们研究了如何对基于PGD的稳健优化训练的稳健模型注入和防御后门攻击。我们证明这些模型容易受到后门攻击。随后，我们观察到后门反映在这些模型的特征表示中。然后，利用这一观察结果，通过一种名为宙斯盾的检测技术来检测这种后门感染的模型。具体地说，对于使用基于PGD的一阶对抗训练方法训练的健壮深度神经网络(DNN)，Aegis使用特征聚类来有效地检测此类DNN是后门感染的还是干净的。在我们使用CIFAR-10、MNIST和FMNIST数据集对主要分类任务中的几个可见和隐藏的后门触发器进行评估时，Aegis有效地检测到感染了后门的PGD训练的健壮DNN。Aegis检测这类后门感染模型的准确率为91.6%(12个测试模型中有11个)，没有任何假阳性。此外，Aegis在后门感染模型中检测到目标类，假阳性率相当低(11.1%)。我们的调查显示，敌方健壮的DNN的显著特征可能有望打破后门攻击的隐蔽性。



## **5. SoK: Adversarial Machine Learning Attacks and Defences in Multi-Agent Reinforcement Learning**

SOK：多智能体强化学习中的对抗性机器学习攻击与防御 cs.LG

**SubmitDate**: 2023-01-11    [abs](http://arxiv.org/abs/2301.04299v1) [paper-pdf](http://arxiv.org/pdf/2301.04299v1)

**Authors**: Maxwell Standen, Junae Kim, Claudia Szabo

**Abstract**: Multi-Agent Reinforcement Learning (MARL) is vulnerable to Adversarial Machine Learning (AML) attacks and needs adequate defences before it can be used in real world applications. We have conducted a survey into the use of execution-time AML attacks against MARL and the defences against those attacks. We surveyed related work in the application of AML in Deep Reinforcement Learning (DRL) and Multi-Agent Learning (MAL) to inform our analysis of AML for MARL. We propose a novel perspective to understand the manner of perpetrating an AML attack, by defining Attack Vectors. We develop two new frameworks to address a gap in current modelling frameworks, focusing on the means and tempo of an AML attack against MARL, and identify knowledge gaps and future avenues of research.

摘要: 多智能体强化学习(MAIL)容易受到对抗性机器学习(AML)的攻击，需要足够的防御才能应用于现实世界。我们已经对针对Marl的执行时AML攻击的使用以及对这些攻击的防御进行了调查。我们综述了AML在深度强化学习(DRL)和多智能体学习(MAL)中应用的相关工作，为我们对MAIL的AML分析提供了信息。通过定义攻击向量，我们提出了一个新的视角来理解实施AML攻击的方式。我们开发了两个新的框架来解决当前建模框架中的差距，重点放在AML攻击Marl的手段和节奏上，并确定知识差距和未来的研究途径。



## **6. User-Centered Security in Natural Language Processing**

自然语言处理中以用户为中心的安全问题 cs.CL

PhD thesis, ISBN 978-94-6458-867-5

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.04230v1) [paper-pdf](http://arxiv.org/pdf/2301.04230v1)

**Authors**: Chris Emmery

**Abstract**: This dissertation proposes a framework of user-centered security in Natural Language Processing (NLP), and demonstrates how it can improve the accessibility of related research. Accordingly, it focuses on two security domains within NLP with great public interest. First, that of author profiling, which can be employed to compromise online privacy through invasive inferences. Without access and detailed insight into these models' predictions, there is no reasonable heuristic by which Internet users might defend themselves from such inferences. Secondly, that of cyberbullying detection, which by default presupposes a centralized implementation; i.e., content moderation across social platforms. As access to appropriate data is restricted, and the nature of the task rapidly evolves (both through lexical variation, and cultural shifts), the effectiveness of its classifiers is greatly diminished and thereby often misrepresented.   Under the proposed framework, we predominantly investigate the use of adversarial attacks on language; i.e., changing a given input (generating adversarial samples) such that a given model does not function as intended. These attacks form a common thread between our user-centered security problems; they are highly relevant for privacy-preserving obfuscation methods against author profiling, and adversarial samples might also prove useful to assess the influence of lexical variation and augmentation on cyberbullying detection.

摘要: 本文提出了一种以用户为中心的自然语言处理安全框架，并论证了该框架如何提高相关研究的可达性。因此，它侧重于NLP内的两个安全领域，具有极大的公共利益。首先是作者侧写，它可以被用来通过侵入性推理来危害在线隐私。如果没有对这些模型预测的访问和详细的洞察，互联网用户就没有合理的启发式方法来为自己辩护，不受此类推论的影响。第二，网络欺凌检测，这在默认情况下是以集中实施为前提的；即跨社交平台的内容审核。由于对适当数据的获取受到限制，任务的性质迅速演变(通过词汇变化和文化变化)，其量词的有效性大大降低，因此经常被歪曲。在所提出的框架下，我们主要调查对语言的对抗性攻击的使用；即，改变给定的输入(生成对抗性样本)，使得给定的模型不起预期的作用。这些攻击形成了我们以用户为中心的安全问题之间的共同线索；它们与隐私保护混淆方法和作者剖析高度相关，而敌意样本也可能被证明有助于评估词汇变异和增强对网络欺凌检测的影响。



## **7. AdvBiom: Adversarial Attacks on Biometric Matchers**

AdvBiom：对生物特征匹配器的敌意攻击 cs.CV

arXiv admin note: text overlap with arXiv:1908.05008

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.03966v1) [paper-pdf](http://arxiv.org/pdf/2301.03966v1)

**Authors**: Debayan Deb, Vishesh Mistry, Rahul Parthe

**Abstract**: With the advent of deep learning models, face recognition systems have achieved impressive recognition rates. The workhorses behind this success are Convolutional Neural Networks (CNNs) and the availability of large training datasets. However, we show that small human-imperceptible changes to face samples can evade most prevailing face recognition systems. Even more alarming is the fact that the same generator can be extended to other traits in the future. In this work, we present how such a generator can be trained and also extended to other biometric modalities, such as fingerprint recognition systems.

摘要: 随着深度学习模型的出现，人脸识别系统已经取得了令人印象深刻的识别率。这一成功背后的主力是卷积神经网络(CNN)和大型训练数据集的可用性。然而，我们表明，对人脸样本进行人类无法察觉的微小变化可以躲避大多数流行的人脸识别系统。更令人担忧的是，同样的基因在未来可以扩展到其他特征。在这项工作中，我们介绍了如何训练这样的生成器，并将其扩展到其他生物识别模式，如指纹识别系统。



## **8. Learned Systems Security**

学习的系统安全 cs.CR

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2212.10318v3) [paper-pdf](http://arxiv.org/pdf/2212.10318v3)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned systems under active development. To empirically validate the many vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against prominent learned-system instances. We show that the use of ML caused leakage of past queries in a database, enabled a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enabled index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is a universal threat against learned systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.

摘要: 学习系统在内部使用机器学习(ML)来提高性能。我们可以预计，这样的系统容易受到一些对抗性的ML攻击。通常，学习的组件在相互不信任的用户或进程之间共享，这与缓存等微体系结构资源非常相似，这可能会导致高度逼真的攻击者模型。然而，与对其他基于ML的系统的攻击相比，攻击者面临着一定程度的间接性，因为他们不能直接与学习的模型交互。此外，同一系统的学习版本和非学习版本的攻击面之间的差异通常是微妙的。这些因素混淆了合并ML带来的事实上的风险。我们分析了学习系统中潜在增加的攻击面的根本原因，并开发了一个框架来识别源于ML的使用的漏洞。我们将我们的框架应用于一系列正在积极开发的学习系统。为了经验性地验证我们的框架中出现的许多漏洞，我们选择其中3个漏洞，并针对突出的学习系统实例实施和评估利用漏洞。我们证明了ML的使用导致了数据库中过去查询的泄漏，启用了导致索引结构中指数级内存爆炸并在几秒钟内崩溃的中毒攻击，并使索引用户能够通过对他们自己的键的定时查询来窥探彼此的键分布。我们发现对抗性ML是对学习系统的普遍威胁，指出在我们对学习系统安全的理解中打开了研究缺口，并通过讨论缓解来结束，同时注意到数据泄漏是在其学习组件由多方共享的系统中固有的。



## **9. Over-The-Air Adversarial Attacks on Deep Learning Wi-Fi Fingerprinting**

深度学习Wi-Fi指纹识别的空中对抗性攻击 cs.CR

To appear in the IEEE Internet of Things Journal

**SubmitDate**: 2023-01-10    [abs](http://arxiv.org/abs/2301.03760v1) [paper-pdf](http://arxiv.org/pdf/2301.03760v1)

**Authors**: Fei Xiao, Yong Huang, Yingying Zuo, Wei Kuang, Wei Wang

**Abstract**: Empowered by deep neural networks (DNNs), Wi-Fi fingerprinting has recently achieved astonishing localization performance to facilitate many security-critical applications in wireless networks, but it is inevitably exposed to adversarial attacks, where subtle perturbations can mislead DNNs to wrong predictions. Such vulnerability provides new security breaches to malicious devices for hampering wireless network security, such as malfunctioning geofencing or asset management. The prior adversarial attack on localization DNNs uses additive perturbations on channel state information (CSI) measurements, which is impractical in Wi-Fi transmissions. To transcend this limitation, this paper presents FooLoc, which fools Wi-Fi CSI fingerprinting DNNs over the realistic wireless channel between the attacker and the victim access point (AP). We observe that though uplink CSIs are unknown to the attacker, the accessible downlink CSIs could be their reasonable substitutes at the same spot. We thoroughly investigate the multiplicative and repetitive properties of over-the-air perturbations and devise an efficient optimization problem to generate imperceptible yet robust adversarial perturbations. We implement FooLoc using commercial Wi-Fi APs and Wireless Open-Access Research Platform (WARP) v3 boards in offline and online experiments, respectively. The experimental results show that FooLoc achieves overall attack success rates of about 70% in targeted attacks and of above 90% in untargeted attacks with small perturbation-to-signal ratios of about -18dB.

摘要: 在深度神经网络(DNN)的支持下，Wi-Fi指纹识别最近取得了惊人的本地化性能，为无线网络中的许多安全关键应用提供了便利，但它不可避免地面临对手攻击，其中细微的扰动可能会误导DNN做出错误的预测。这种漏洞为恶意设备提供了新的安全漏洞，以阻碍无线网络安全，例如出现故障的地理围栏或资产管理。以前针对本地化DNN的敌意攻击使用对信道状态信息(CSI)测量的加性扰动，这在Wi-Fi传输中是不切实际的。为了突破这一局限，本文提出了FooLoc，它在攻击者和受害者接入点(AP)之间的真实无线通道上欺骗Wi-Fi CSI指纹识别DNN。我们观察到，尽管上行链路CSI对于攻击者来说是未知的，但可访问的下行CSIS可能是他们在同一地点的合理替代。我们深入研究了空中扰动的乘性和重复性，并设计了一个有效的优化问题来产生不可察觉但健壮的对抗性扰动。我们使用商用Wi-Fi AP和无线开放访问研究平台(WARP)v3板分别在离线和在线实验中实现了FooLoc。实验结果表明，FooLoc在目标攻击中的总体攻击成功率约为70%，在非目标攻击中的整体攻击成功率在90%以上，小扰动信噪比约为-18dB。



## **10. On the Susceptibility and Robustness of Time Series Models through Adversarial Attack and Defense**

对抗性攻防下时间序列模型的敏感性和稳健性 cs.LG

8 pages, 3 figures, 7 tables

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2301.03703v1) [paper-pdf](http://arxiv.org/pdf/2301.03703v1)

**Authors**: Asadullah Hill Galib, Bidhan Bashyal

**Abstract**: Under adversarial attacks, time series regression and classification are vulnerable. Adversarial defense, on the other hand, can make the models more resilient. It is important to evaluate how vulnerable different time series models are to attacks and how well they recover using defense. The sensitivity to various attacks and the robustness using the defense of several time series models are investigated in this study. Experiments are run on seven-time series models with three adversarial attacks and one adversarial defense. According to the findings, all models, particularly GRU and RNN, appear to be vulnerable. LSTM and GRU also have better defense recovery. FGSM exceeds the competitors in terms of attacks. PGD attacks are more difficult to recover from than other sorts of attacks.

摘要: 在对抗性攻击下，时间序列回归和分类是脆弱的。另一方面，对抗性防御可以使模型更具弹性。重要的是要评估不同的时间序列模型对攻击的脆弱性以及它们使用防御的恢复情况。研究了几种时间序列模型对各种攻击的敏感度和防御的稳健性。实验是在具有三个对抗性攻击和一个对抗性防御的七时间序列模型上进行的。根据调查结果，所有模型，特别是GRU和RNN，似乎都容易受到攻击。LSTM和GRU的防御恢复也更好。FGSM在攻击方面超过了竞争对手。与其他类型的攻击相比，PGD攻击更难恢复。



## **11. Adversarial Policies Beat Superhuman Go AIs**

对抗性政策击败了超人围棋 cs.LG

36 pages, 19 figures, see paper for changelog

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2211.00241v2) [paper-pdf](http://arxiv.org/pdf/2211.00241v2)

**Authors**: Tony Tong Wang, Adam Gleave, Nora Belrose, Tom Tseng, Joseph Miller, Kellin Pelrine, Michael D Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system, KataGo, by training adversarial policies that play against frozen KataGo victims. Our attack achieves a >99% win rate when KataGo uses no tree-search, and a >77% win rate when KataGo uses enough search to be superhuman. Notably, our adversaries do not win by learning to play Go better than KataGo -- in fact, our adversaries are easily beaten by human amateurs. Instead, our adversaries win by tricking KataGo into making serious blunders. Our results demonstrate that even superhuman AI systems may harbor surprising failure modes. Example games are available at https://goattack.far.ai/.

摘要: 我们通过训练对抗冻结的KataGo受害者的对抗性策略来攻击最先进的围棋人工智能系统KataGo。当KataGo不使用树搜索时，我们的攻击胜率超过99%，当KataGo使用足够的搜索来成为超人时，我们的攻击胜率超过77%。值得注意的是，我们的对手并不是通过学习比KataGo下得更好的围棋来取胜的--事实上，我们的对手很容易被人类业余选手击败。相反，我们的对手通过欺骗KataGo犯下严重的错误而获胜。我们的结果表明，即使是超人人工智能系统也可能存在令人惊讶的故障模式。示例游戏可在https://goattack.far.ai/.上获得



## **12. F3B: A Low-Overhead Blockchain Architecture with Per-Transaction Front-Running Protection**

F3B：一种具有每事务前置保护的低开销区块链架构 cs.CR

25 pages, 6 figures

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2205.08529v2) [paper-pdf](http://arxiv.org/pdf/2205.08529v2)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Mahsa Bastankhah, Vero Estrada-Galinanes, Bryan Ford

**Abstract**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the blockchain space, since the emergence of decentralized finance. Front-running causes devastating losses to honest participants and continues to endanger the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture that addresses front-running attacks by using threshold cryptography. In F3B, a user generates a symmetric key to encrypt their transaction, and once the underlying consensus layer has committed the transaction, a decentralized secret-management committee reveals this key. F3B mitigates front-running attacks because, before the consensus group commits it, an adversary can no longer read the content of a transaction, thus preventing the adversary from benefiting from advanced knowledge of pending transactions. Unlike other threshold-based approaches, where users encrypt their transactions with a key derived from a future block, F3B enables users to generate a unique key for each transaction. This feature ensures that all uncommitted transactions remain private, even if they are delayed. Furthermore, F3B addresses front-running at the execution layer; thus, our solution is agnostic to the underlying consensus algorithm and compatible with existing smart contracts. We evaluated F3B based on Ethereum, demonstrating a 0.05% transaction latency overhead with a secret-management committee of 128 members, thus indicating our solution is practical at a low cost.

摘要: 自去中心化金融出现以来，受益于待完成交易的先进知识的前沿攻击在区块链领域激增。领跑给诚实的参与者造成了毁灭性的损失，并继续危及生态系统的公平性。我们提出了Flash冻结Flash Boys(F3B)，这是一种区块链架构，通过使用门限密码来应对前沿攻击。在F3B中，用户生成对称密钥来加密他们的交易，一旦底层共识层提交了交易，分散的秘密管理委员会就会公布该密钥。F3B减轻了前置攻击，因为在共识组提交之前，对手不能再读取交易的内容，从而阻止对手受益于有关未决交易的高级知识。与其他基于阈值的方法不同，在这些方法中，用户使用从未来块派生的密钥来加密他们的交易，F3B使用户能够为每笔交易生成唯一的密钥。此功能确保所有未提交的事务保持私有，即使它们被延迟。此外，F3B解决了执行层的先行问题；因此，我们的解决方案与底层共识算法无关，并与现有的智能合约兼容。我们基于Etherum对F3B进行了评估，在一个由128名成员组成的秘密管理委员会的情况下，交易延迟开销为0.05%，因此表明我们的解决方案是实用的，成本较低。



## **13. Distributed Estimation over Directed Graphs Resilient to Sensor Spoofing**

抗传感器欺骗的有向图上的分布式估计 eess.SY

12 pages, 9 figures

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2104.04680v2) [paper-pdf](http://arxiv.org/pdf/2104.04680v2)

**Authors**: Shamik Bhattacharyya, Kiran Rokade, Rachel Kalpana Kalaimani

**Abstract**: This paper addresses the problem of distributed estimation of an unknown dynamic parameter by a multi-agent system over a directed communication network in the presence of an adversarial attack on the agents' sensors. The mode of attack of the adversaries is to corrupt the sensor measurements of some of the agents, while the communication and information processing capabilities of those agents remain unaffected. To ensure that all the agents, both normal as well as those under attack, are able to correctly estimate the parameter value, the Resilient Estimation through Weight Balancing (REWB) algorithm is introduced. The only condition required for the REWB algorithm to guarantee resilient estimation is that at any given point in time, less than half of the total number of agents are under attack. The paper discusses the development of the REWB algorithm using the concepts of weight balancing of directed graphs, and the consensus+innovations approach for linear estimation. Numerical simulations are presented to illustrate the performance of our algorithm over directed graphs under different conditions of adversarial attacks.

摘要: 研究了当多智能体系统的传感器受到敌意攻击时，多智能体系统在有向通信网络上对未知动态参数进行分布式估计的问题。对手的攻击模式是破坏一些代理的传感器测量，而这些代理的通信和信息处理能力保持不变。为了确保所有正常的和被攻击的代理都能够正确地估计参数值，引入了弹性加权平衡估计(REWB)算法。REWB算法保证弹性估计的唯一条件是在任何给定的时间点，只有不到一半的代理受到攻击。本文利用有向图权重平衡的概念，讨论了REWB算法的发展，以及线性估计的共识+新息方法。数值模拟结果表明，在不同的对抗性攻击条件下，该算法在有向图上具有较好的性能。



## **14. Structural Equivalence in Subgraph Matching**

子图匹配中的结构等价 cs.DS

To appear in IEEE Transactions on Network Science and Engineering

**SubmitDate**: 2023-01-09    [abs](http://arxiv.org/abs/2301.03161v1) [paper-pdf](http://arxiv.org/pdf/2301.03161v1)

**Authors**: Dominic Yang, Yurun Ge, Thien Nguyen, Jacob Moorman, Denali Molitor, Andrea Bertozzi

**Abstract**: Symmetry plays a major role in subgraph matching both in the description of the graphs in question and in how it confounds the search process. This work addresses how to quantify these effects and how to use symmetries to increase the efficiency of subgraph isomorphism algorithms. We introduce rigorous definitions of structural equivalence and establish conditions for when it can be safely used to generate more solutions. We illustrate how to adapt standard search routines to utilize these symmetries to accelerate search and compactly describe the solution space. We then adapt a state-of-the-art solver and perform a comprehensive series of tests to demonstrate these methods' efficacy on a standard benchmark set. We extend these methods to multiplex graphs and present results on large multiplex networks drawn from transportation systems, social media, adversarial attacks, and knowledge graphs.

摘要: 对称性在子图匹配中扮演着重要的角色，无论是在对所讨论的图的描述中，还是在它如何混淆搜索过程中。这项工作解决了如何量化这些影响以及如何利用对称性来提高子图同构算法的效率。我们引入了结构等价的严格定义，并为什么时候可以安全地使用它来生成更多的解建立了条件。我们说明了如何调整标准搜索例程以利用这些对称性来加速搜索并紧凑地描述解空间。然后，我们采用最先进的求解器并执行一系列全面的测试，以在标准基准集上演示这些方法的有效性。我们将这些方法扩展到多重图，并在来自交通系统、社交媒体、对手攻击和知识图的大型多重网络上给出了结果。



## **15. RobArch: Designing Robust Architectures against Adversarial Attacks**

RobArch：设计针对对手攻击的健壮体系结构 cs.CV

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2301.03110v1) [paper-pdf](http://arxiv.org/pdf/2301.03110v1)

**Authors**: ShengYun Peng, Weilin Xu, Cory Cornelius, Kevin Li, Rahul Duggal, Duen Horng Chau, Jason Martin

**Abstract**: Adversarial Training is the most effective approach for improving the robustness of Deep Neural Networks (DNNs). However, compared to the large body of research in optimizing the adversarial training process, there are few investigations into how architecture components affect robustness, and they rarely constrain model capacity. Thus, it is unclear where robustness precisely comes from. In this work, we present the first large-scale systematic study on the robustness of DNN architecture components under fixed parameter budgets. Through our investigation, we distill 18 actionable robust network design guidelines that empower model developers to gain deep insights. We demonstrate these guidelines' effectiveness by introducing the novel Robust Architecture (RobArch) model that instantiates the guidelines to build a family of top-performing models across parameter capacities against strong adversarial attacks. RobArch achieves the new state-of-the-art AutoAttack accuracy on the RobustBench ImageNet leaderboard. The code is available at $\href{https://github.com/ShengYun-Peng/RobArch}{\text{this url}}$.

摘要: 对抗性训练是提高深度神经网络(DNN)鲁棒性的最有效方法。然而，与优化对抗性训练过程的大量研究相比，关于体系结构组件如何影响健壮性的研究很少，而且它们很少限制模型的容量。因此，还不清楚健壮性到底从何而来。在这项工作中，我们首次对DNN体系结构组件在固定参数预算下的健壮性进行了大规模系统研究。通过我们的调查，我们提炼出18个可操作的健壮网络设计指南，使模型开发人员能够获得深入的见解。我们通过引入新的健壮体系结构(RobArch)模型来证明这些指南的有效性，该模型实例化了构建跨参数能力的抗强对手攻击的一系列最高性能模型的指南。RobArch在RobustBuchImageNet排行榜上实现了最新的AutoAttack精度。代码可以在$\href{https://github.com/ShengYun-Peng/RobArch}{\text{this url上找到。



## **16. A Bayesian Robust Regression Method for Corrupted Data Reconstruction**

一种基于贝叶斯稳健回归的数据重构方法 cs.LG

22 pages

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2212.12787v2) [paper-pdf](http://arxiv.org/pdf/2212.12787v2)

**Authors**: Zheyi Fan, Zhaohui Li, Jingyan Wang, Dennis K. J. Lin, Xiao Xiong, Qingpei Hu

**Abstract**: Because of the widespread existence of noise and data corruption, recovering the true regression parameters with a certain proportion of corrupted response variables is an essential task. Methods to overcome this problem often involve robust least-squares regression, but few methods perform well when confronted with severe adaptive adversarial attacks. In many applications, prior knowledge is often available from historical data or engineering experience, and by incorporating prior information into a robust regression method, we develop an effective robust regression method that can resist adaptive adversarial attacks. First, we propose the novel TRIP (hard Thresholding approach to Robust regression with sImple Prior) algorithm, which improves the breakdown point when facing adaptive adversarial attacks. Then, to improve the robustness and reduce the estimation error caused by the inclusion of priors, we use the idea of Bayesian reweighting to construct the more robust BRHT (robust Bayesian Reweighting regression via Hard Thresholding) algorithm. We prove the theoretical convergence of the proposed algorithms under mild conditions, and extensive experiments show that under different types of dataset attacks, our algorithms outperform other benchmark ones. Finally, we apply our methods to a data-recovery problem in a real-world application involving a space solar array, demonstrating their good applicability.

摘要: 由于噪声和数据损坏的普遍存在，用一定比例的损坏响应变量恢复真实的回归参数是一项重要的任务。克服这一问题的方法通常涉及稳健的最小二乘回归，但很少有方法在面对严重的自适应对手攻击时表现良好。在许多应用中，先验知识往往来自历史数据或工程经验，通过将先验信息融入到稳健回归方法中，我们提出了一种有效的稳健回归方法，可以抵抗自适应对手攻击。首先，提出了一种新的基于简单先验的硬阈值稳健回归算法TRIP(Hard Threshold Approach To Robust Regregation With Simple Prior)，改善了自适应攻击时的故障点。然后，为了提高算法的稳健性并减小先验信息的引入所带来的估计误差，利用贝叶斯加权的思想构造了更稳健的基于硬阈值的稳健贝叶斯重加权回归算法。我们在温和的条件下证明了所提算法的理论收敛，并且大量的实验表明，在不同类型的数据集攻击下，我们的算法的性能优于其他基准算法。最后，我们将我们的方法应用于一个涉及空间太阳能电池板的实际应用中的数据恢复问题，展示了它们良好的适用性。



## **17. Deepfake CAPTCHA: A Method for Preventing Fake Calls**

深伪验证码：一种防止虚假呼叫的方法 cs.CR

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2301.03064v1) [paper-pdf](http://arxiv.org/pdf/2301.03064v1)

**Authors**: Lior Yasur, Guy Frankovits, Fred M. Grabovski, Yisroel Mirsky

**Abstract**: Deep learning technology has made it possible to generate realistic content of specific individuals. These `deepfakes' can now be generated in real-time which enables attackers to impersonate people over audio and video calls. Moreover, some methods only need a few images or seconds of audio to steal an identity. Existing defenses perform passive analysis to detect fake content. However, with the rapid progress of deepfake quality, this may be a losing game.   In this paper, we propose D-CAPTCHA: an active defense against real-time deepfakes. The approach is to force the adversary into the spotlight by challenging the deepfake model to generate content which exceeds its capabilities. By doing so, passive detection becomes easier since the content will be distorted. In contrast to existing CAPTCHAs, we challenge the AI's ability to create content as opposed to its ability to classify content. In this work we focus on real-time audio deepfakes and present preliminary results on video.   In our evaluation we found that D-CAPTCHA outperforms state-of-the-art audio deepfake detectors with an accuracy of 91-100% depending on the challenge (compared to 71% without challenges). We also performed a study on 41 volunteers to understand how threatening current real-time deepfake attacks are. We found that the majority of the volunteers could not tell the difference between real and fake audio.

摘要: 深度学习技术使生成特定个人的现实内容成为可能。这些“深度假冒”现在可以实时生成，使攻击者能够通过音频和视频通话冒充他人。此外，一些方法只需要几张图像或几秒钟的音频就可以窃取身份。现有的防御系统执行被动分析来检测虚假内容。然而，随着深度假货品质的快速进步，这可能是一场失败的游戏。在本文中，我们提出了D-CAPTCHA：一种针对实时深度假冒的主动防御。方法是通过挑战深度假冒模型来生成超出其能力的内容，从而迫使对手成为聚光灯下的焦点。通过这样做，被动检测变得更容易，因为内容将被扭曲。与现有的验证码不同，我们挑战人工智能创造内容的能力，而不是对内容进行分类的能力。在这项工作中，我们专注于实时音频深伪，并给出了视频的初步结果。在我们的评估中，我们发现D-CAPTCHA的性能优于最先进的音频深伪检测器，根据挑战的不同，准确率为91%-100%(而没有挑战的准确率为71%)。我们还对41名志愿者进行了一项研究，以了解当前实时深度虚假攻击的威胁性有多大。我们发现，大多数志愿者无法区分真实和虚假的音频。



## **18. Byzantine Multiple Access Channels -- Part I: Reliable Communication**

拜占庭式多址接入信道--第一部分：可靠通信 cs.IT

This supercedes Part I of arxiv:1904.11925

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2211.12769v2) [paper-pdf](http://arxiv.org/pdf/2211.12769v2)

**Authors**: Neha Sangwan, Mayank Bakshi, Bikash Kumar Dey, Vinod M. Prabhakaran

**Abstract**: We study communication over a Multiple Access Channel (MAC) where users can possibly be adversarial. The receiver is unaware of the identity of the adversarial users (if any). When all users are non-adversarial, we want their messages to be decoded reliably. When a user behaves adversarially, we require that the honest users' messages be decoded reliably. An adversarial user can mount an attack by sending any input into the channel rather than following the protocol. It turns out that the $2$-user MAC capacity region follows from the point-to-point Arbitrarily Varying Channel (AVC) capacity. For the $3$-user MAC in which at most one user may be malicious, we characterize the capacity region for deterministic codes and randomized codes (where each user shares an independent random secret key with the receiver). These results are then generalized for the $k$-user MAC where the adversary may control all users in one out of a collection of given subsets.

摘要: 我们研究了多路访问信道(MAC)上的通信，其中用户可能是对抗性的。接收方不知道敌对用户(如果有的话)的身份。当所有用户都是非对抗性的时，我们希望他们的消息被可靠地解码。当用户做出恶意行为时，我们要求可靠地解码诚实用户的消息。敌意用户可以通过向通道发送任何输入而不是遵循协议来发动攻击。事实证明，$2$-用户MAC容量区域紧随点对点任意变化信道(AVC)容量。对于最多一个用户可能是恶意用户的$3$-用户MAC，我们刻画了确定码和随机码的容量域(其中每个用户与接收方共享一个独立的随机密钥)。然后将这些结果推广到$k$-用户MAC，其中对手可以控制给定子集集合中的一个用户。



## **19. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

Garnet：强健可扩展图神经网络的降阶拓扑学习 cs.LG

Published as a conference paper at LoG 2022

**SubmitDate**: 2023-01-08    [abs](http://arxiv.org/abs/2201.12741v6) [paper-pdf](http://arxiv.org/pdf/2201.12741v6)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.

摘要: 图形神经网络(GNN)已被越来越多地应用于涉及非欧几里德数据学习的各种应用中。然而，最近的研究表明，GNN容易受到图的对抗性攻击。虽然有几种防御方法可以通过消除敌对组件来提高GNN的健壮性，但它们也可能损害有助于GNN训练的底层干净的图形结构。此外，这些防御模型中很少有能够扩展到大型图形的，因为它们的计算复杂性和内存使用量很高。在本文中，我们提出了Garnet，一种可伸缩的谱方法来提高GNN模型的对抗健壮性。Garnet First利用加权谱嵌入来构造基图，该基图不仅能抵抗敌方攻击，而且还包含GNN训练所需的关键(干净)图结构。接下来，Garnet基于概率图模型，通过剪枝额外的非关键边来进一步精化基图。石榴石已经在各种数据集上进行了评估，包括一个包含数百万个节点的大型图表。我们的大量实验结果表明，与现有的GNN(防御)模型相比，Garnet的对抗准确率提高了13.27%，运行时加速比提高了14.7倍。



## **20. Robust Feature-Level Adversaries are Interpretability Tools**

强大的功能级对手是可解释的工具 cs.LG

Code available at  https://github.com/thestephencasper/feature_level_adv

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2110.03605v6) [paper-pdf](http://arxiv.org/pdf/2110.03605v6)

**Authors**: Stephen Casper, Max Nadeau, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstract**: The literature on adversarial attacks in computer vision typically focuses on pixel-level perturbations. These tend to be very difficult to interpret. Recent work that manipulates the latent representations of image generators to create "feature-level" adversarial perturbations gives us an opportunity to explore perceptible, interpretable adversarial attacks. We make three contributions. First, we observe that feature-level attacks provide useful classes of inputs for studying representations in models. Second, we show that these adversaries are uniquely versatile and highly robust. We demonstrate that they can be used to produce targeted, universal, disguised, physically-realizable, and black-box attacks at the ImageNet scale. Third, we show how these adversarial images can be used as a practical interpretability tool for identifying bugs in networks. We use these adversaries to make predictions about spurious associations between features and classes which we then test by designing "copy/paste" attacks in which one natural image is pasted into another to cause a targeted misclassification. Our results suggest that feature-level attacks are a promising approach for rigorous interpretability research. They support the design of tools to better understand what a model has learned and diagnose brittle feature associations. Code is available at https://github.com/thestephencasper/feature_level_adv

摘要: 关于计算机视觉中的对抗性攻击的文献通常集中在像素级的扰动上。这些往往很难解释。最近的工作是利用图像生成器的潜在表示来创建“特征级别”的对抗性扰动，这给了我们一个探索可感知的、可解释的对抗性攻击的机会。我们有三点贡献。首先，我们观察到特征级别的攻击为学习模型中的表示提供了有用的输入类。其次，我们展示了这些对手独一无二的多才多艺和高度健壮。我们证明了它们可以用于在ImageNet规模上产生有针对性的、普遍的、伪装的、物理上可实现的和黑匣子攻击。第三，我们展示了如何将这些对抗性图像用作识别网络漏洞的实用可解释性工具。我们利用这些对手来预测特征和类别之间的虚假关联，然后通过设计“复制/粘贴”攻击来测试这些关联，在这种攻击中，一幅自然图像被粘贴到另一幅图像中，从而导致有针对性的误分类。我们的结果表明，特征级攻击对于严格的可解释性研究是一种很有前途的方法。它们支持工具的设计，以更好地理解模型学习到的内容并诊断脆弱的特征关联。代码可在https://github.com/thestephencasper/feature_level_adv上找到



## **21. Adversarial training with informed data selection**

用知情的数据选择进行对抗性训练 cs.LG

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2301.04472v1) [paper-pdf](http://arxiv.org/pdf/2301.04472v1)

**Authors**: Marcele O. K. Mendonça, Javier Maroto, Pascal Frossard, Paulo S. R. Diniz

**Abstract**: With the increasing amount of available data and advances in computing capabilities, deep neural networks (DNNs) have been successfully employed to solve challenging tasks in various areas, including healthcare, climate, and finance. Nevertheless, state-of-the-art DNNs are susceptible to quasi-imperceptible perturbed versions of the original images -- adversarial examples. These perturbations of the network input can lead to disastrous implications in critical areas where wrong decisions can directly affect human lives. Adversarial training is the most efficient solution to defend the network against these malicious attacks. However, adversarial trained networks generally come with lower clean accuracy and higher computational complexity. This work proposes a data selection (DS) strategy to be applied in the mini-batch training. Based on the cross-entropy loss, the most relevant samples in the batch are selected to update the model parameters in the backpropagation. The simulation results show that a good compromise can be obtained regarding robustness and standard accuracy, whereas the computational complexity of the backpropagation pass is reduced.

摘要: 随着可用数据量的增加和计算能力的提高，深度神经网络(DNN)已被成功地用于解决包括医疗保健、气候和金融在内的各个领域的挑战性任务。然而，最先进的DNN很容易受到原始图像的准不可察觉的扰动版本的影响--对抗性的例子。网络输入的这些扰动可能会在关键领域造成灾难性影响，在这些领域，错误的决策可能会直接影响人类的生命。对抗性训练是保护网络免受这些恶意攻击的最有效解决方案。然而，对抗性训练的网络通常具有较低的干净准确率和较高的计算复杂性。本文提出了一种用于小批量训练的数据选择(DS)策略。基于交叉熵损失，选择批次中最相关的样本来更新反向传播中的模型参数。仿真结果表明，该算法在稳健性和标准精度方面取得了较好的折衷，同时降低了反向传播通路的计算复杂度。



## **22. PatchUp: A Feature-Space Block-Level Regularization Technique for Convolutional Neural Networks**

PatchUp：卷积神经网络的特征空间块级正则化技术 cs.LG

AAAI - 2022

**SubmitDate**: 2023-01-07    [abs](http://arxiv.org/abs/2006.07794v2) [paper-pdf](http://arxiv.org/pdf/2006.07794v2)

**Authors**: Mojtaba Faramarzi, Mohammad Amini, Akilesh Badrinaaraayanan, Vikas Verma, Sarath Chandar

**Abstract**: Large capacity deep learning models are often prone to a high generalization gap when trained with a limited amount of labeled training data. A recent class of methods to address this problem uses various ways to construct a new training sample by mixing a pair (or more) of training samples. We propose PatchUp, a hidden state block-level regularization technique for Convolutional Neural Networks (CNNs), that is applied on selected contiguous blocks of feature maps from a random pair of samples. Our approach improves the robustness of CNN models against the manifold intrusion problem that may occur in other state-of-the-art mixing approaches. Moreover, since we are mixing the contiguous block of features in the hidden space, which has more dimensions than the input space, we obtain more diverse samples for training towards different dimensions. Our experiments on CIFAR10/100, SVHN, Tiny-ImageNet, and ImageNet using ResNet architectures including PreActResnet18/34, WRN-28-10, ResNet101/152 models show that PatchUp improves upon, or equals, the performance of current state-of-the-art regularizers for CNNs. We also show that PatchUp can provide a better generalization to deformed samples and is more robust against adversarial attacks.

摘要: 大容量深度学习模型在使用有限数量的标签训练数据训练时，往往容易出现较高的泛化差距。最近一类解决该问题的方法使用各种方式来通过混合一对(或更多)训练样本来构建新的训练样本。提出了一种用于卷积神经网络(CNN)的隐藏状态块级正则化技术PatchUp，该技术应用于从随机样本对中选择连续的特征映射块。我们的方法提高了CNN模型对其他最先进的混合方法中可能出现的流形入侵问题的稳健性。此外，由于我们将相邻的特征块混合在具有比输入空间更多维度的隐藏空间中，因此我们获得了更多针对不同维度的训练样本。我们在CIFAR10/100、SVHN、Tiny-ImageNet和ImageNet上使用ResNet架构(包括PreActResnet18/34、WRN-28-10、ResNet101/152)进行的实验表明，PatchUp改进了或等于当前最先进的CNN正则化算法的性能。我们还表明，PatchUp能够对变形样本提供更好的泛化，并且对对手攻击具有更强的鲁棒性。



## **23. Stealthy Backdoor Attack for Code Models**

针对代码模型的隐蔽后门攻击 cs.CR

Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02496v1) [paper-pdf](http://arxiv.org/pdf/2301.02496v1)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.

摘要: 代码模型，如CodeBERT和CodeT5，提供了代码的通用表示，并在支持下游自动化软件工程任务方面发挥了至关重要的作用。最近，代码模型被发现容易受到后门攻击。被后门攻击的代码模型可以在干净的示例上正常运行，但会在注入了激活后门的触发器的示例上生成预定义的恶意输出。现有对代码模型的后门攻击使用隐蔽且易于检测的触发器。本文旨在研究具有隐蔽后门攻击的代码模型的脆弱性。为此，我们提出了AFRAIDOOR(对抗性特征作为自适应后门)。AFRAIDOOR通过利用对抗性扰动将自适应触发器注入不同的输入来实现隐蔽性。我们在三个广泛采用的代码模型(CodeBERT、PLBART和CodeT5)和两个下游任务(代码摘要和方法名称预测)上对AFRAIDOOR进行了评估。我们发现，AFRAIDOOR中约85%的自适应触发器在防御过程中绕过了检测。相比之下，只有不到12%的以前工作中的触发因素绕过了防御。当不应用防御方法时，AFRAIDOOR和基线都具有几乎完美的攻击成功率。然而，一旦实施防御，基线的成功率急剧下降到10.47%和12.06%，而AFRAIDOOR在两个任务上的成功率分别为77.05%和92.98%。我们的发现暴露了代码模型在秘密后门攻击下的安全漏洞，并表明最先进的防御方法不能提供足够的保护。我们呼吁在了解代码模型的安全威胁和开发更有效的对策方面做出更多研究努力。



## **24. Watching your call: Breaking VoLTE Privacy in LTE/5G Networks**

关注您的电话：打破LTE/5G网络中的VoLTE隐私 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02487v1) [paper-pdf](http://arxiv.org/pdf/2301.02487v1)

**Authors**: Zishuai Cheng, Mihai Ordean, Flavio D. Garcia, Baojiang Cui, Dominik Rys

**Abstract**: Voice over LTE (VoLTE) and Voice over NR (VoNR) are two similar technologies that have been widely deployed by operators to provide a better calling experience in LTE and 5G networks, respectively. The VoLTE/NR protocols rely on the security features of the underlying LTE/5G network to protect users' privacy such that nobody can monitor calls and learn details about call times, duration, and direction. In this paper, we introduce a new privacy attack which enables adversaries to analyse encrypted LTE/5G traffic and recover any VoLTE/NR call details. We achieve this by implementing a novel mobile-relay adversary which is able to remain undetected by using an improved physical layer parameter guessing procedure. This adversary facilitates the recovery of encrypted configuration messages exchanged between victim devices and the mobile network. We further propose an identity mapping method which enables our mobile-relay adversary to link a victim's network identifiers to the phone number efficiently, requiring a single VoLTE protocol message. We evaluate the real-world performance of our attacks using four modern commercial off-the-shelf phones and two representative, commercial network carriers. We collect over 60 hours of traffic between the phones and the mobile networks and execute 160 VoLTE calls, which we use to successfully identify patterns in the physical layer parameter allocation and in VoLTE traffic, respectively. Our real-world experiments show that our mobile-relay works as expected in all test cases, and the VoLTE activity logs recovered describe the actual communication with 100% accuracy. Finally, we show that we can link network identifiers such as International Mobile Subscriber Identities (IMSI), Subscriber Concealed Identifiers (SUCI) and/or Globally Unique Temporary Identifiers (GUTI) to phone numbers while remaining undetected by the victim.

摘要: LTE语音(VoLTE)和NR语音(VoNR)是运营商广泛部署的两项类似技术，分别在LTE和5G网络中提供更好的呼叫体验。VoLTE/NR协议依靠底层LTE/5G网络的安全功能来保护用户隐私，因此没有人可以监控通话并了解有关通话时间、时长和方向的详细信息。在本文中，我们介绍了一种新的隐私攻击，使攻击者能够分析加密的LTE/5G流量，并恢复任何VoLTE/NR呼叫细节。我们通过使用改进的物理层参数猜测过程实现了一种新的移动中继对手，该对手能够保持不被检测到。该敌手便于恢复在受害者设备和移动网络之间交换的加密配置消息。我们进一步提出了一种身份映射方法，使我们的移动中继攻击者能够有效地将受害者的网络标识符链接到电话号码，只需要一条VoLTE协议消息。我们使用四部现代商用现成手机和两家具有代表性的商用网络运营商来评估我们的攻击的真实性能。我们收集电话和移动网络之间超过60小时的流量，并执行160个VoLTE呼叫，我们使用这些呼叫分别成功识别物理层参数分配和VoLTE流量中的模式。实际测试表明，我们的移动中继在所有测试用例中都能正常工作，恢复的VoLTE活动日志对实际通信的描述准确率达到100%。最后，我们展示了我们可以将诸如国际移动用户标识(IMSI)、用户隐藏标识(SUCI)和/或全球唯一临时标识(GUTI)之类的网络标识符链接到电话号码，而不被受害者检测到。



## **25. Adversarial Attacks on Neural Models of Code via Code Difference Reduction**

码差缩减对码神经模型的敌意攻击 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02412v1) [paper-pdf](http://arxiv.org/pdf/2301.02412v1)

**Authors**: Zhao Tian, Junjie Chen, Zhi Jin

**Abstract**: Deep learning has been widely used to solve various code-based tasks by building deep code models based on a large number of code snippets. However, deep code models are still vulnerable to adversarial attacks. As source code is discrete and has to strictly stick to the grammar and semantics constraints, the adversarial attack techniques in other domains are not applicable. Moreover, the attack techniques specific to deep code models suffer from the effectiveness issue due to the enormous attack space. In this work, we propose a novel adversarial attack technique (i.e., CODA). Its key idea is to use the code differences between the target input and reference inputs (that have small code differences but different prediction results with the target one) to guide the generation of adversarial examples. It considers both structure differences and identifier differences to preserve the original semantics. Hence, the attack space can be largely reduced as the one constituted by the two kinds of code differences, and thus the attack process can be largely improved by designing corresponding equivalent structure transformations and identifier renaming transformations. Our experiments on 10 deep code models (i.e., two pre trained models with five code-based tasks) demonstrate the effectiveness and efficiency of CODA, the naturalness of its generated examples, and its capability of defending against attacks after adversarial fine-tuning. For example, CODA improves the state-of-the-art techniques (i.e., CARROT and ALERT) by 79.25% and 72.20% on average in terms of the attack success rate, respectively.

摘要: 深度学习通过基于大量代码片段构建深度代码模型，被广泛用于解决各种基于代码的任务。然而，深层代码模型仍然容易受到敌意攻击。由于源代码是离散的，并且必须严格遵守语法和语义的约束，因此其他领域的对抗性攻击技术不适用。此外，针对深层代码模型的攻击技术由于攻击空间巨大而存在有效性问题。在这项工作中，我们提出了一种新的对抗性攻击技术(即CODA)。它的核心思想是利用目标输入和参考输入之间的编码差异(编码差异很小，但预测结果与目标输入不同)来指导对抗性实例的生成。它同时考虑了结构差异和标识差异，以保持原有的语义。因此，通过设计相应的等价结构变换和标识符重命名变换，可以将攻击空间大大缩减为由两种代码差异构成的攻击空间，从而大大改善了攻击过程。我们在10个深层代码模型(即两个具有5个基于代码的任务的预训练模型)上的实验证明了CODA的有效性和高效性、生成的示例的自然性以及经过对抗性微调后的抵御攻击的能力。例如，在攻击成功率方面，CODA将最先进的技术(即胡萝卜和警报)平均分别提高了79.25%和72.20%。



## **26. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

特洛伊木马之谜：秘密中毒代码-建议模型 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02344v1) [paper-pdf](http://arxiv.org/pdf/2301.02344v1)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training or fine-tuning phases by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior poisoning attacks explicitly inject the insecure code payload into the training data, making the poisoning data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel data poisoning attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poisoning data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poisoning data by never including certain (suspicious) parts of the payload in the poisoned data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that identify and filter out suspicious sequences from the training data. Our evaluation against two model sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for how practitioners should select code used to train or tune code-suggestion models.

摘要: 有了GitHub Copilot这样的工具，自动代码建议不再是软件工程中的梦想。这些工具基于大型语言模型，通常针对从未经审查的公共来源挖掘的大量代码语料库进行培训。因此，这些模型容易受到数据中毒攻击，即对手通过注入恶意数据来操纵模型的训练或微调阶段。毒化攻击可以被设计成影响模型在运行时对所选上下文的建议，例如诱导模型建议不安全的代码有效负载。为了实现这一点，先前的中毒攻击显式地将不安全的代码有效载荷注入到训练数据中，使得可以从训练集中移除此类恶意数据的静态分析工具可以检测到中毒数据。在这项工作中，我们展示了两种新型的数据中毒攻击：ASTIFT和TROJANPUZLE，它们可以通过在文档字符串等脱离上下文的区域植入恶意中毒数据来绕过静态分析。我们最新颖的攻击TROJANPUZLE在生成不那么可疑的中毒数据方面又向前迈进了一步，它从未在有毒数据中包括有效负载的某些(可疑)部分，同时仍诱导出一个模型，该模型在完成代码时(即在文档字符串外部)建议整个有效负载。这使得TROJANPUZLE对于基于签名的数据集清理方法具有健壮性，这些方法从训练数据中识别和过滤可疑序列。我们对两个模型大小的评估表明，COVERT和TROJANPUZLE对于实践者应该如何选择用于训练或调优代码建议模型的代码具有重要影响。



## **27. DRL-GAN: A Hybrid Approach for Binary and Multiclass Network Intrusion Detection**

DRL-GAN：一种混合的二类和多类网络入侵检测方法 cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.03368v1) [paper-pdf](http://arxiv.org/pdf/2301.03368v1)

**Authors**: Caroline Strickland, Chandrika Saha, Muhammad Zakar, Sareh Nejad, Noshin Tasnim, Daniel Lizotte, Anwar Haque

**Abstract**: Our increasingly connected world continues to face an ever-growing amount of network-based attacks. Intrusion detection systems (IDS) are an essential security technology for detecting these attacks. Although numerous machine learning-based IDS have been proposed for the detection of malicious network traffic, the majority have difficulty properly detecting and classifying the more uncommon attack types. In this paper, we implement a novel hybrid technique using synthetic data produced by a Generative Adversarial Network (GAN) to use as input for training a Deep Reinforcement Learning (DRL) model. Our GAN model is trained with the NSL-KDD dataset for four attack categories as well as normal network flow. Ultimately, our findings demonstrate that training the DRL on specific synthetic datasets can result in better performance in correctly classifying minority classes over training on the true imbalanced dataset.

摘要: 我们日益互联的世界继续面临着越来越多的基于网络的攻击。入侵检测系统(入侵检测系统)是检测这些攻击的重要安全技术。虽然已经提出了许多基于机器学习的入侵检测系统来检测恶意网络流量，但大多数都很难正确地检测和分类更常见的攻击类型。在本文中，我们实现了一种新的混合技术，使用生成性对抗网络(GAN)产生的合成数据作为输入来训练深度强化学习(DRL)模型。我们使用NSL-KDD数据集对四种攻击类别和正常网络流量进行了GAN模型训练。最终，我们的发现表明，在特定的合成数据集上训练DRL可以导致在正确分类少数类方面比在真正的不平衡数据集上训练的性能更好。



## **28. Silent Killer: Optimizing Backdoor Trigger Yields a Stealthy and Powerful Data Poisoning Attack**

无声杀手：优化后门触发器可产生隐形且强大的数据中毒攻击 cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02615v1) [paper-pdf](http://arxiv.org/pdf/2301.02615v1)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: We propose a stealthy and powerful backdoor attack on neural networks based on data poisoning (DP). In contrast to previous attacks, both the poison and the trigger in our method are stealthy. We are able to change the model's classification of samples from a source class to a target class chosen by the attacker. We do so by using a small number of poisoned training samples with nearly imperceptible perturbations, without changing their labels. At inference time, we use a stealthy perturbation added to the attacked samples as a trigger. This perturbation is crafted as a universal adversarial perturbation (UAP), and the poison is crafted using gradient alignment coupled to this trigger. Our method is highly efficient in crafting time compared to previous methods and requires only a trained surrogate model without additional retraining. Our attack achieves state-of-the-art results in terms of attack success rate while maintaining high accuracy on clean samples.

摘要: 提出了一种基于数据毒化(DP)的隐蔽而强大的神经网络后门攻击方法。与以前的攻击不同，我们方法中的毒药和触发器都是隐蔽的。我们能够将模型的样本分类从源类更改为攻击者选择的目标类。我们通过使用少量具有几乎不可察觉的扰动的有毒训练样本来做到这一点，而不改变它们的标签。在推断时，我们使用添加到被攻击样本的隐形扰动作为触发器。该扰动被精心设计为通用对抗性扰动(UAP)，并且毒药是使用与该触发器耦合的梯度对齐来定制的。与以前的方法相比，我们的方法在计算时间上具有很高的效率，并且只需要一个经过训练的代理模型，而不需要额外的重新训练。我们的攻击在攻击成功率方面实现了最先进的结果，同时保持了对干净样本的高精度。



## **29. Holistic Adversarial Robustness of Deep Learning Models**

深度学习模型的整体对抗稳健性 cs.LG

survey paper on holistic adversarial robustness for deep learning;  published at AAAI 2023 Senior Member Presentation Track

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2202.07201v3) [paper-pdf](http://arxiv.org/pdf/2202.07201v3)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstract**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning-based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.

摘要: 对抗健壮性研究机器学习模型的最坏情况下的性能，以确保安全性和可靠性。随着基于深度学习的技术的激增，与模型开发和部署相关的潜在风险可能会放大，并成为可怕的漏洞。本文全面综述了深度学习模型对抗性稳健性的研究主题和基本原理，包括攻击、防御、验证和新的应用。



## **30. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

随机消息拦截平滑：图神经网络的灰箱证书 cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02039v1) [paper-pdf](http://arxiv.org/pdf/2301.02039v1)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.

摘要: 随机化平滑是证明机器学习模型(包括图神经网络)对抗稳健性的最有前途的框架之一。然而，现有的用于GNN的随机化平滑证书过于悲观，因为它们将模型视为黑匣子，忽略了底层架构。为了解决这个问题，我们提出了一种新的灰盒证书，它利用了GNN的消息传递原理：我们随机截获消息，并仔细分析来自恶意控制节点的消息到达目标节点的概率。与现有的证书相比，我们证明了对控制图中的整个节点并可以任意操纵节点特征的更强大的攻击者的健壮性。我们的证书为更远距离的攻击提供了更强有力的保证，因为来自较远节点的消息更有可能被拦截。我们在不同的模型和数据集上演示了我们的方法的有效性。由于我们的灰盒证书考虑了底层的图结构，所以我们可以通过应用图稀疏来显著提高可证明的健壮性。



## **31. Beckman Defense**

贝克曼辩护 cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01495v2) [paper-pdf](http://arxiv.org/pdf/2301.01495v2)

**Authors**: A. V. Subramanyam

**Abstract**: Optimal transport (OT) based distributional robust optimisation (DRO) has received some traction in the recent past. However, it is at a nascent stage but has a sound potential in robustifying the deep learning models. Interestingly, OT barycenters demonstrate a good robustness against adversarial attacks. Owing to the computationally expensive nature of OT barycenters, they have not been investigated under DRO framework. In this work, we propose a new barycenter, namely Beckman barycenter, which can be computed efficiently and used for training the network to defend against adversarial attacks in conjunction with adversarial training. We propose a novel formulation of Beckman barycenter and analytically obtain the barycenter using the marginals of the input image. We show that the Beckman barycenter can be used to train adversarially trained networks to improve the robustness. Our training is extremely efficient as it requires only a single epoch of training. Elaborate experiments on CIFAR-10, CIFAR-100 and Tiny ImageNet demonstrate that training an adversarially robust network with Beckman barycenter can significantly increase the performance. Under auto attack, we get a a maximum boost of 10\% in CIFAR-10, 8.34\% in CIFAR-100 and 11.51\% in Tiny ImageNet. Our code is available at https://github.com/Visual-Conception-Group/test-barycentric-defense.

摘要: 最近，基于最优传输(OT)的分布式稳健优化(DRO)受到了一些关注。然而，它还处于初级阶段，但在推动深度学习模式方面具有良好的潜力。有趣的是，OT重心对敌方攻击表现出良好的健壮性。由于OT重心的计算代价很高，因此尚未在DRO框架下对其进行研究。在这项工作中，我们提出了一种新的重心，即Beckman重心，它可以被有效地计算出来，并用于训练网络在对抗训练的同时防御对手攻击。我们提出了一种新的Beckman重心公式，并利用输入图像的边缘来解析地获得重心。我们证明了Beckman重心可以用于训练对抗性训练的网络，以提高网络的健壮性。我们的训练非常有效，因为它只需要一个时期的训练。在CIFAR-10、CIFAR-100和Tiny ImageNet上的详细实验表明，使用Beckman重心训练一个对抗健壮的网络可以显著提高性能。在AUTO攻击下，CIFAR-10、CIFAR-100和TING ImageNet的最大性能提升分别为10%、8.34%和11.51%。我们的代码可以在https://github.com/Visual-Conception-Group/test-barycentric-defense.上找到



## **32. Enhancement attacks in biomedical machine learning**

生物医学机器学习中的增强攻击 stat.ML

13 pages, 3 figures

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01885v1) [paper-pdf](http://arxiv.org/pdf/2301.01885v1)

**Authors**: Matthew Rosenblatt, Javid Dadashkarimi, Dustin Scheinost

**Abstract**: The prevalence of machine learning in biomedical research is rapidly growing, yet the trustworthiness of such research is often overlooked. While some previous works have investigated the ability of adversarial attacks to degrade model performance in medical imaging, the ability to falsely improve performance via recently-developed "enhancement attacks" may be a greater threat to biomedical machine learning. In the spirit of developing attacks to better understand trustworthiness, we developed three techniques to drastically enhance prediction performance of classifiers with minimal changes to features, including the enhancement of 1) within-dataset predictions, 2) a particular method over another, and 3) cross-dataset generalization. Our within-dataset enhancement framework falsely improved classifiers' accuracy from 50% to almost 100% while maintaining high feature similarities between original and enhanced data (Pearson's r's>0.99). Similarly, the method-specific enhancement framework was effective in falsely improving the performance of one method over another. For example, a simple neural network outperformed LR by 50% on our enhanced dataset, although no performance differences were present in the original dataset. Crucially, the original and enhanced data were still similar (r=0.95). Finally, we demonstrated that enhancement is not specific to within-dataset predictions but can also be adapted to enhance the generalization accuracy of one dataset to another by up to 38%. Overall, our results suggest that more robust data sharing and provenance tracking pipelines are necessary to maintain data integrity in biomedical machine learning research.

摘要: 机器学习在生物医学研究中的盛行正在迅速增长，但此类研究的可信度往往被忽视。虽然以前的一些工作已经研究了对抗性攻击降低医学成像中模型性能的能力，但通过最近开发的增强攻击来错误地提高性能的能力可能会对生物医学机器学习构成更大的威胁。本着开发攻击以更好地理解可信度的精神，我们开发了三种技术来在对特征进行最小更改的情况下显著提高分类器的预测性能，包括1)数据集内预测的增强，2)一种特定方法优于另一种方法，以及3)跨数据集泛化。我们的数据集内增强框架错误地将分类器的准确率从50%提高到几乎100%，同时保持了原始数据和增强数据之间的高度特征相似性(Pearson‘s r’s>0.99)。类似地，特定于方法的增强框架在错误地改进一种方法的性能方面是有效的。例如，一个简单的神经网络在我们的增强数据集上的性能比LR高50%，尽管在原始数据集中没有表现出性能差异。重要的是，原始数据和增强数据仍然相似(r=0.95)。最后，我们证明了增强并不特定于数据集内的预测，但也可以用于将一个数据集到另一个数据集的泛化精度提高高达38%。总体而言，我们的结果表明，在生物医学机器学习研究中，为了保持数据的完整性，需要更健壮的数据共享和来源跟踪管道。



## **33. Availability Adversarial Attack and Countermeasures for Deep Learning-based Load Forecasting**

基于深度学习的负荷预测可用性攻击与对策 cs.LG

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01832v1) [paper-pdf](http://arxiv.org/pdf/2301.01832v1)

**Authors**: Wangkun Xu, Fei Teng

**Abstract**: The forecast of electrical loads is essential for the planning and operation of the power system. Recently, advances in deep learning have enabled more accurate forecasts. However, deep neural networks are prone to adversarial attacks. Although most of the literature focuses on integrity-based attacks, this paper proposes availability-based adversarial attacks, which can be more easily implemented by attackers. For each forecast instance, the availability attack position is optimally solved by mixed-integer reformulation of the artificial neural network. To tackle this attack, an adversarial training algorithm is proposed. In simulation, a realistic load forecasting dataset is considered and the attack performance is compared to the integrity-based attack. Meanwhile, the adversarial training algorithm is shown to significantly improve robustness against availability attacks. All codes are available at https://github.com/xuwkk/AAA_Load_Forecast.

摘要: 电力负荷预测对于电力系统的规划和运行是至关重要的。最近，深度学习的进步使预测更加准确。然而，深度神经网络容易受到对抗性攻击。虽然大多数文献关注的是基于完整性的攻击，但本文提出的基于可用性的对抗性攻击更容易被攻击者实现。对于每个预测实例，通过人工神经网络的混合整数重构来最优地求解可用攻击位置。针对这种攻击，提出了一种对抗性训练算法。在仿真中，考虑了真实的负载预测数据集，并将其攻击性能与基于完整性的攻击进行了比较。同时，对抗性训练算法显著提高了对可用性攻击的健壮性。所有代码均可在https://github.com/xuwkk/AAA_Load_Forecast.上获得。



## **34. GUAP: Graph Universal Attack Through Adversarial Patching**

GUAP：通过对抗性补丁实现通用攻击 cs.LG

8 pages

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01731v1) [paper-pdf](http://arxiv.org/pdf/2301.01731v1)

**Authors**: Xiao Zang, Jie Chen, Bo Yuan

**Abstract**: Graph neural networks (GNNs) are a class of effective deep learning models for node classification tasks; yet their predictive capability may be severely compromised under adversarially designed unnoticeable perturbations to the graph structure and/or node data. Most of the current work on graph adversarial attacks aims at lowering the overall prediction accuracy, but we argue that the resulting abnormal model performance may catch attention easily and invite quick counterattack. Moreover, attacks through modification of existing graph data may be hard to conduct if good security protocols are implemented. In this work, we consider an easier attack harder to be noticed, through adversarially patching the graph with new nodes and edges. The attack is universal: it targets a single node each time and flips its connection to the same set of patch nodes. The attack is unnoticeable: it does not modify the predictions of nodes other than the target. We develop an algorithm, named GUAP, that achieves high attack success rate but meanwhile preserves the prediction accuracy. GUAP is fast to train by employing a sampling strategy. We demonstrate that a 5% sampling in each epoch yields 20x speedup in training, with only a slight degradation in attack performance. Additionally, we show that the adversarial patch trained with the graph convolutional network transfers well to other GNNs, such as the graph attention network.

摘要: 图神经网络(GNN)是一类用于节点分类任务的有效深度学习模型，但在图结构和/或节点数据受到恶意设计的不可察觉扰动时，其预测能力可能会受到严重影响。目前关于图对抗攻击的大部分工作都是为了降低整体预测精度，但我们认为，由此产生的异常模型性能可能容易引起注意并引发快速反击。此外，如果实施了良好的安全协议，通过修改现有图形数据进行的攻击可能很难进行。在这项工作中，我们认为更容易的攻击更难被注意，通过用新的节点和边恶意修补图。这种攻击是通用的：它每次只针对一个节点，并将其连接反转到同一组补丁节点。攻击是不可察觉的：它不会修改除目标之外的其他节点的预测。提出了一种在保持预测精度的同时获得较高攻击成功率的GUAP算法。通过采用抽样策略，GAP的训练速度很快。我们证明，在每个时期进行5%的采样可以在训练中获得20倍的加速，而攻击性能只有轻微的下降。此外，我们还证明了用图卷积网络训练的敌意补丁可以很好地移植到其他GNN上，例如图注意网络。



## **35. A Survey on Physical Adversarial Attack in Computer Vision**

计算机视觉中的身体对抗攻击研究综述 cs.CV

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2209.14262v2) [paper-pdf](http://arxiv.org/pdf/2209.14262v2)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijian Tang, Xiaoqian Chen

**Abstract**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.

摘要: 在过去的十年里，深度学习以其强大的特征学习能力，极大地改变了传统的手工特征学习方式，使常规任务得到了极大的改善。然而，深度神经网络最近被证明容易受到敌意例子的攻击，这是一种由精心设计的小噪声制作的恶意样本，它误导DNN做出错误的决定，同时保持对人类的不可察觉。对抗性攻击可分为数字对抗性攻击和物理对抗性攻击。数字对抗攻击大多在实验室环境中进行，致力于提高对抗攻击算法的性能。相比之下，物理对抗性攻击侧重于攻击物理世界中部署的DNN系统，由于物理环境复杂(即亮度、遮挡等)，这是一项更具挑战性的任务。虽然数字对抗例子和物理对抗例子之间的差异很小，但物理对抗例子有一个特定的设计来克服复杂物理环境的影响。本文回顾了基于DNN的计算机视觉任务中物理对抗攻击的发展，包括图像识别任务、目标检测任务和语义分割任务。为了算法演化的完备性，我们将简要介绍不涉及物理对抗攻击的工作。我们首先提出了一种分类方案来总结当前的物理对抗性攻击。然后讨论了现有物理对抗性攻击的优缺点，并重点介绍了应用于物理环境中维护对抗性的技术。最后，指出了当前物理对抗性攻击需要解决的问题，并提出了有前景的研究方向。



## **36. Validity in Music Information Research Experiments**

音乐信息研究实验中的效度 cs.SD

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01578v1) [paper-pdf](http://arxiv.org/pdf/2301.01578v1)

**Authors**: Bob L. T. Sturm, Arthur Flexer

**Abstract**: Validity is the truth of an inference made from evidence, such as data collected in an experiment, and is central to working scientifically. Given the maturity of the domain of music information research (MIR), validity in our opinion should be discussed and considered much more than it has been so far. Considering validity in one's work can improve its scientific and engineering value. Puzzling MIR phenomena like adversarial attacks and performance glass ceilings become less mysterious through the lens of validity. In this article, we review the subject of validity in general, considering the four major types of validity from a key reference: Shadish et al. 2002. We ground our discussion of these types with a prototypical MIR experiment: music classification using machine learning. Through this MIR experimentalists can be guided to make valid inferences from data collected from their experiments.

摘要: 有效性是从证据中做出的推论的真实性，例如在实验中收集的数据，它是科学工作的核心。鉴于音乐信息研究(MIR)领域的成熟，我们认为应该比目前更多地讨论和考虑有效性。在工作中考虑有效性可以提高工作的科学价值和工程价值。令人费解的MIR现象，如对抗性攻击和性能玻璃天花板，通过有效性的镜头变得不那么神秘。在这篇文章中，我们大体回顾了有效性的主题，从一个关键的参考文献考虑了四种主要的有效性类型：Shaish等人。2002年。我们用一个典型的MIR实验来讨论这些类型：使用机器学习的音乐分类。通过这一实验，可以指导实验者从他们的实验中收集的数据中做出有效的推断。



## **37. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2208.12216v3) [paper-pdf](http://arxiv.org/pdf/2208.12216v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **38. The Feasibility and Inevitability of Stealth Attacks**

隐形攻击的可行性和必然性 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2106.13997v4) [paper-pdf](http://arxiv.org/pdf/2106.13997v4)

**Authors**: Ivan Y. Tyukin, Desmond J. Higham, Alexander Bastounis, Eliyas Woldegeorgis, Alexander N. Gorban

**Abstract**: We develop and study new adversarial perturbations that enable an attacker to gain control over decisions in generic Artificial Intelligence (AI) systems including deep learning neural networks. In contrast to adversarial data modification, the attack mechanism we consider here involves alterations to the AI system itself. Such a stealth attack could be conducted by a mischievous, corrupt or disgruntled member of a software development team. It could also be made by those wishing to exploit a ``democratization of AI'' agenda, where network architectures and trained parameter sets are shared publicly. We develop a range of new implementable attack strategies with accompanying analysis, showing that with high probability a stealth attack can be made transparent, in the sense that system performance is unchanged on a fixed validation set which is unknown to the attacker, while evoking any desired output on a trigger input of interest. The attacker only needs to have estimates of the size of the validation set and the spread of the AI's relevant latent space. In the case of deep learning neural networks, we show that a one neuron attack is possible - a modification to the weights and bias associated with a single neuron - revealing a vulnerability arising from over-parameterization. We illustrate these concepts using state of the art architectures on two standard image data sets. Guided by the theory and computational results, we also propose strategies to guard against stealth attacks.

摘要: 我们开发和研究了新的对抗性扰动，使攻击者能够控制通用人工智能(AI)系统中的决策，包括深度学习神经网络。与对抗性数据修改不同，我们在这里考虑的攻击机制涉及对AI系统本身的更改。这种隐形攻击可能是由软件开发团队中调皮的、腐败的或心怀不满的成员实施的。它也可以由那些希望利用“人工智能民主化”议程的人提出，在这种议程中，网络架构和经过训练的参数集是公开共享的。我们开发了一系列新的可实现的攻击策略，并进行了分析，表明在高概率情况下，隐形攻击可以变得透明，也就是说，在攻击者未知的固定验证集上，系统性能不变，同时在感兴趣的触发器输入上唤起任何期望的输出。攻击者只需要对验证集的大小和人工智能相关潜在空间的传播进行估计。在深度学习神经网络的情况下，我们证明了一个神经元攻击是可能的--对与单个神经元相关的权重和偏差的修改--揭示了由于过度参数化而产生的漏洞。我们在两个标准图像数据集上使用最先进的体系结构来说明这些概念。在理论和计算结果的指导下，我们还提出了防范隐身攻击的策略。



## **39. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2210.13263v3) [paper-pdf](http://arxiv.org/pdf/2210.13263v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **40. Organised Firestorm as strategy for business cyber-attacks**

有组织的火暴作为商业网络攻击的战略 cs.CY

9 pages, 3 figures, 2 table

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01518v1) [paper-pdf](http://arxiv.org/pdf/2301.01518v1)

**Authors**: Andrea Russo

**Abstract**: Having a good reputation is paramount for most organisations and companies. In fact, having an optimal corporate image allows them to have better transaction relationships with various customers and partners. However, such reputation is hard to build and easy to destroy for all kind of business commercial activities (B2C, B2B, B2B2C, B2G). A misunderstanding during the communication process to the customers, or just a bad communication strategy, can lead to a disaster for the entire company. This is emphasised by the reaction of millions of people on social networks, which can be very detrimental for the corporate image if they react negatively to a certain event. This is called a firestorm.   In this paper, I propose a well-organised strategy for firestorm attacks on organisations, also showing how an adversary can leverage them to obtain private information on the attacked firm. Standard business security procedures are not designed to operate against multi-domain attacks; therefore, I will show how it is possible to bypass the classic and advised security procedures by operating different kinds of attack. I also propose a different firestorm attack, targeting a specific business company network in an efficient way. Finally, I present defensive procedures to reduce the negative effect of firestorms on a company.

摘要: 对大多数组织和公司来说，拥有良好的声誉是最重要的。事实上，拥有最佳的公司形象可以让他们与各种客户和合作伙伴建立更好的交易关系。然而，对于所有类型的商业商业活动(B2C、B2B、B2B2C、B2G)来说，这样的声誉很难建立，也很容易被摧毁。在与客户沟通的过程中产生误解，或者只是沟通策略不当，都可能给整个公司带来灾难。数以百万计的人在社交网络上的反应突显了这一点，如果他们对某个事件做出负面反应，这可能会对公司形象造成非常不利的影响。这被称为大火风暴。在这篇文章中，我提出了一种组织严密的战略，以应对对组织的火暴攻击，并展示了对手如何利用它们来获取被攻击公司的私人信息。标准的业务安全程序不是为抵御多域攻击而设计的；因此，我将展示如何通过操作不同类型的攻击来绕过经典的和建议的安全程序。我还提出了一种不同的FireStorm攻击，以一种高效的方式针对特定的商业公司网络。最后，我提出了一些防御措施，以减少风暴对公司的负面影响。



## **41. Universal adversarial perturbation for remote sensing images**

遥感图像的普遍对抗性摄动 cs.CV

Published in the Twenty-Fourth International Workshop on Multimedia  Signal Processing, MMSP 2022

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2202.10693v2) [paper-pdf](http://arxiv.org/pdf/2202.10693v2)

**Authors**: Qingyu Wang, Guorui Feng, Zhaoxia Yin, Bin Luo

**Abstract**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been dramatically improved compared with traditional technology. However, even the state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). The research on UAP is mostly limited to ordinary images, and RSIs have not been studied. To explore the basic characteristics of UAPs of RSIs, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism to generate the UAP of RSIs. Firstly, the former is used to generate the UAP, which can learn the distribution of perturbations better, and then the latter is used to find the sensitive regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbation making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the classification model misclassify, and the attack success rate of our proposed method on the RSI data set is as high as 97.09%.

摘要: 近年来，随着深度学习在遥感图像领域的应用，遥感图像的分类精度与传统技术相比有了很大的提高。然而，即使是最先进的目标识别卷积神经网络也被通用对抗性摄动(UAP)愚弄了。对UAP的研究大多局限于普通图像，对RIS的研究尚未见报道。为了探索RSIS的UAP的基本特征，提出了一种将编解码器网络和注意力机制相结合的RSIS UAP生成方法。首先利用前者生成能更好地学习扰动分布的UAP，然后利用后者寻找RSI分类模型所关注的敏感区域。最后，生成的区域被用来微调扰动，使得模型在较少扰动的情况下被误分类。实验结果表明，UAP能够使分类模型发生误分类，本文提出的方法在RSI数据集上的攻击成功率高达97.09%。



## **42. Surveillance Face Anti-spoofing**

监控面反欺骗 cs.CV

15 pages, 9 figures

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00975v1) [paper-pdf](http://arxiv.org/pdf/2301.00975v1)

**Authors**: Hao Fang, Ajian Liu, Jun Wan, Sergio Escalera, Chenxu Zhao, Xu Zhang, Stan Z. Li, Zhen Lei

**Abstract**: Face Anti-spoofing (FAS) is essential to secure face recognition systems from various physical attacks. However, recent research generally focuses on short-distance applications (i.e., phone unlocking) while lacking consideration of long-distance scenes (i.e., surveillance security checks). In order to promote relevant research and fill this gap in the community, we collect a large-scale Surveillance High-Fidelity Mask (SuHiFiMask) dataset captured under 40 surveillance scenes, which has 101 subjects from different age groups with 232 3D attacks (high-fidelity masks), 200 2D attacks (posters, portraits, and screens), and 2 adversarial attacks. In this scene, low image resolution and noise interference are new challenges faced in surveillance FAS. Together with the SuHiFiMask dataset, we propose a Contrastive Quality-Invariance Learning (CQIL) network to alleviate the performance degradation caused by image quality from three aspects: (1) An Image Quality Variable module (IQV) is introduced to recover image information associated with discrimination by combining the super-resolution network. (2) Using generated sample pairs to simulate quality variance distributions to help contrastive learning strategies obtain robust feature representation under quality variation. (3) A Separate Quality Network (SQN) is designed to learn discriminative features independent of image quality. Finally, a large number of experiments verify the quality of the SuHiFiMask dataset and the superiority of the proposed CQIL.

摘要: 人脸反欺骗技术是保护人脸识别系统免受各种物理攻击的重要手段。然而，目前的研究大多集中在短距离应用(如手机解锁)上，而缺乏对远程场景(如监控安检)的考虑。为了推动相关研究，填补社区这一空白，我们收集了40个监控场景下的大规模监控高保真面具(SuHiFiMASK)数据集，其中包含来自不同年龄段的101名受试者，分别进行了232次3D攻击(高保真面具)、200次2D攻击(海报、肖像和屏幕)和2次对抗性攻击。在这种情况下，图像分辨率低和噪声干扰是自动监控系统面临的新挑战。结合SuHiFiMASK数据集，我们从三个方面提出了一种对比质量-不变性学习(CQIL)网络来缓解图像质量引起的性能下降：(1)引入图像质量变量模块(IQV)，通过结合超分辨率网络来恢复与区分相关的图像信息。(2)使用生成的样本对来模拟质量方差分布，以帮助对比学习策略在质量变化下获得稳健的特征表示。(3)设计了一个独立的质量网络(SQN)来学习与图像质量无关的区分特征。最后，通过大量实验验证了SuHiFiMASK数据集的质量和CQIL算法的优越性。



## **43. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

对抗性时空聚焦视频的高效稳健性评估 cs.CV

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00896v1) [paper-pdf](http://arxiv.org/pdf/2301.00896v1)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.

摘要: 视频识别模型的对抗性健壮性评估由于其在安全关键任务中的广泛应用而引起了人们的关注。与图像相比，视频的维度要高得多，这在生成对抗性视频时带来了巨大的计算代价。这对于基于查询的黑盒攻击尤为严重，这种攻击通常使用威胁模型的梯度估计，高维将导致大量的查询。为了缓解这一问题，我们提出同时消除视频中的时间和空间冗余，在缩减的搜索空间上实现有效和高效的梯度估计，从而减少查询数量。为了实现这一思想，我们设计了一种新颖的对抗性时空聚焦(AstFocus)攻击，它从视频的帧间和帧内对同时聚焦的关键帧和关键区域进行攻击。AstFocus攻击基于协作多智能体强化学习(MAIL)框架。一个代理负责选择关键帧，另一个代理负责选择关键区域。这两个代理通过从黑盒威胁模型获得的共同奖励来联合训练，以执行合作预测。通过连续查询，缩小了由关键帧和关键区域组成的搜索空间，变得更加精确，整个查询次数比原始视频上的少。在四个主流视频识别模型和三个广泛使用的动作识别数据集上的大量实验表明，AstFocus攻击的性能优于SOTA方法，后者在愚弄率、查询次数、时间和扰动幅度方面都优于SOTA方法。



## **44. Adaptive Perturbation for Adversarial Attack**

对抗性攻击的自适应摄动 cs.CV

13 pages, 5 figures, 9 tables

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2111.13841v2) [paper-pdf](http://arxiv.org/pdf/2111.13841v2)

**Authors**: Zheng Yuan, Jie Zhang, Zhaoyan Jiang, Liangliang Li, Shiguang Shan

**Abstract**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. Instead of using the sign function, we propose to directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. At the same time, we also theoretically prove that this method can achieve better black-box transferability. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve their attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.

摘要: 近年来，随着神经网络的快速发展，深度学习模型的安全性越来越受到人们的关注，因为神经网络容易受到敌意例子的攻击。几乎所有现有的基于梯度的攻击方法都在生成时使用符号函数，以满足$L_INFTY$范数上的扰动预算要求。然而，我们发现符号函数可能不适合于生成对抗性示例，因为它修改了精确的梯度方向。我们不使用符号函数，而是直接利用带有比例因子的精确梯度方向来产生对抗性扰动，从而在扰动较少的情况下提高了对抗性实例的攻击成功率。同时，我们还从理论上证明了该方法可以达到更好的黑盒可转移性。此外，考虑到不同图像的最佳比例因子不同，我们提出了一种自适应比例因子生成器来为每幅图像寻找合适的比例因子，从而避免了手动搜索比例因子的计算代价。我们的方法可以与几乎所有现有的基于梯度的攻击方法相集成，进一步提高它们的攻击成功率。在CIFAR10和ImageNet数据集上的大量实验表明，我们的方法表现出更高的可转移性，并且性能优于最先进的方法。



## **45. Differentiable Search of Accurate and Robust Architectures**

精确且健壮的体系结构的可微搜索 cs.LG

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2212.14049v2) [paper-pdf](http://arxiv.org/pdf/2212.14049v2)

**Authors**: Yuwei Ou, Xiangning Xie, Shangce Gao, Yanan Sun, Kay Chen Tan, Jiancheng Lv

**Abstract**: Deep neural networks (DNNs) are found to be vulnerable to adversarial attacks, and various methods have been proposed for the defense. Among these methods, adversarial training has been drawing increasing attention because of its simplicity and effectiveness. However, the performance of the adversarial training is greatly limited by the architectures of target DNNs, which often makes the resulting DNNs with poor accuracy and unsatisfactory robustness. To address this problem, we propose DSARA to automatically search for the neural architectures that are accurate and robust after adversarial training. In particular, we design a novel cell-based search space specially for adversarial training, which improves the accuracy and the robustness upper bound of the searched architectures by carefully designing the placement of the cells and the proportional relationship of the filter numbers. Then we propose a two-stage search strategy to search for both accurate and robust neural architectures. At the first stage, the architecture parameters are optimized to minimize the adversarial loss, which makes full use of the effectiveness of the adversarial training in enhancing the robustness. At the second stage, the architecture parameters are optimized to minimize both the natural loss and the adversarial loss utilizing the proposed multi-objective adversarial training method, so that the searched neural architectures are both accurate and robust. We evaluate the proposed algorithm under natural data and various adversarial attacks, which reveals the superiority of the proposed method in terms of both accurate and robust architectures. We also conclude that accurate and robust neural architectures tend to deploy very different structures near the input and the output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust neural architectures.

摘要: 深度神经网络(DNN)被发现容易受到敌意攻击，并且已经提出了各种防御方法。在这些方法中，对抗性训练因其简单性和有效性而受到越来越多的关注。然而，对抗性训练的性能很大程度上受到目标DNN结构的限制，这往往使所得到的DNN具有较差的准确性和较差的稳健性。为了解决这一问题，我们提出了DSARA在对抗性训练后自动搜索准确和健壮的神经结构。特别是，我们设计了一种专门用于对抗性训练的基于单元的搜索空间，通过仔细设计单元的位置和过滤器数量的比例关系，提高了搜索结构的准确性和鲁棒性上界。然后，我们提出了一种两阶段搜索策略来搜索准确和健壮的神经结构。在第一阶段，对结构参数进行优化，使对抗性损失最小，充分利用对抗性训练在增强鲁棒性方面的有效性。在第二阶段，利用所提出的多目标对抗性训练方法对结构参数进行优化，使自然损失和对抗性损失最小化，从而使搜索到的神经结构既准确又健壮。我们在自然数据和各种敌意攻击下对该算法进行了测试，结果表明该算法在准确性和健壮性方面都具有一定的优势。我们还得出结论，精确和健壮的神经结构往往在输入和输出附近部署截然不同的结构，这对于手工制作和自动设计准确和健壮的神经结构都具有重要的现实意义。



## **46. Reversible Attack based on Local Visual Adversarial Perturbation**

基于局部视觉对抗扰动的可逆攻击 cs.CV

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2110.02700v3) [paper-pdf](http://arxiv.org/pdf/2110.02700v3)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstract**: Adding perturbations to images can mislead classification models to produce incorrect results. Recently, researchers exploited adversarial perturbations to protect image privacy from retrieval by intelligent models. However, adding adversarial perturbations to images destroys the original data, making images useless in digital forensics and other fields. To prevent illegal or unauthorized access to sensitive image data such as human faces without impeding legitimate users, the use of reversible adversarial attack techniques is increasing. The original image can be recovered from its reversible adversarial examples. However, existing reversible adversarial attack methods are designed for traditional imperceptible adversarial perturbations and ignore the local visible adversarial perturbation. In this paper, we propose a new method for generating reversible adversarial examples based on local visible adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by the reversible data hiding technique. To reduce image distortion, lossless compression and the B-R-G (bluered-green) embedding principle are adopted. Experiments on CIFAR-10 and ImageNet datasets show that the proposed method can restore the original images error-free while ensuring good attack performance.

摘要: 向图像添加扰动可能会误导分类模型产生不正确的结果。最近，研究人员利用对抗性扰动来保护图像隐私，使其不受智能模型的检索。然而，向图像添加对抗性扰动会破坏原始数据，使图像在数字取证等领域毫无用处。为了防止在不妨碍合法用户的情况下非法或未经授权地访问人脸等敏感图像数据，可逆对抗性攻击技术的使用正在增加。原始图像可以从其可逆的对抗性例子中恢复。然而，现有的可逆对抗性攻击方法是针对传统的不可察觉的对抗性扰动而设计的，忽略了局部可见的对抗性扰动。本文提出了一种基于局部可视对抗性扰动的可逆对抗性实例生成方法。利用可逆数据隐藏技术将图像恢复所需的信息嵌入到敌方补丁之外的区域。为了减少图像失真，采用了无损压缩和B-R-G(蓝绿色)嵌入原理。在CIFAR-10和ImageNet数据集上的实验表明，该方法能够在保证良好攻击性能的前提下无差错地恢复原始图像。



## **47. Trojaning semi-supervised learning model via poisoning wild images on the web**

中毒网络野图的特洛伊木马半监督学习模型 cs.CY

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00435v1) [paper-pdf](http://arxiv.org/pdf/2301.00435v1)

**Authors**: Le Feng, Zhenxing Qian, Sheng Li, Xinpeng Zhang

**Abstract**: Wild images on the web are vulnerable to backdoor (also called trojan) poisoning, causing machine learning models learned on these images to be injected with backdoors. Most previous attacks assumed that the wild images are labeled. In reality, however, most images on the web are unlabeled. Specifically, we study the effects of unlabeled backdoor images under semi-supervised learning (SSL) on widely studied deep neural networks. To be realistic, we assume that the adversary is zero-knowledge and that the semi-supervised learning model is trained from scratch. Firstly, we find the fact that backdoor poisoning always fails when poisoned unlabeled images come from different classes, which is different from poisoning the labeled images. The reason is that the SSL algorithms always strive to correct them during training. Therefore, for unlabeled images, we implement backdoor poisoning on images from the target class. Then, we propose a gradient matching strategy to craft poisoned images such that their gradients match the gradients of target images on the SSL model, which can fit poisoned images to the target class and realize backdoor injection. To the best of our knowledge, this may be the first approach to backdoor poisoning on unlabeled images of trained-from-scratch SSL models. Experiments show that our poisoning achieves state-of-the-art attack success rates on most SSL algorithms while bypassing modern backdoor defenses.

摘要: 网络上的狂野图像很容易受到后门(也称为特洛伊木马)的毒害，导致从这些图像上学习的机器学习模型被注入后门。以前的大多数攻击都假设野生图像被标记了。然而，在现实中，网络上的大多数图片都是没有标签的。具体地说，我们研究了半监督学习(半监督学习)下的未标记后门图像对广泛研究的深度神经网络的影响。为了现实，我们假设对手是零知识，半监督学习模型是从头开始训练的。首先，我们发现，当中毒的未标记图像来自不同类别时，后门攻击总是失败的，这与对标记图像进行中毒是不同的。这是因为，在训练过程中，SSL算法总是努力纠正它们。因此，对于未标记的图像，我们对来自目标类的图像进行后门毒化。然后，我们提出了一种梯度匹配策略来构造中毒图像，使其梯度与目标图像在SSL模型上的梯度相匹配，从而将中毒图像匹配到目标类，实现后门注入。据我们所知，这可能是第一种对从头开始训练的SSL模型的未标记图像进行后门中毒的方法。实验表明，我们的毒剂在绕过现代后门防御的同时，在大多数SSL算法上实现了最先进的攻击成功率。



## **48. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2210.08870v3) [paper-pdf](http://arxiv.org/pdf/2210.08870v3)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **49. Generalizable Black-Box Adversarial Attack with Meta Learning**

基于元学习的泛化黑箱对抗攻击 cs.LG

T-PAMI 2022. Project Page is at https://github.com/SCLBD/MCG-Blackbox

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00364v1) [paper-pdf](http://arxiv.org/pdf/2301.00364v1)

**Authors**: Fei Yin, Yong Zhang, Baoyuan Wu, Yan Feng, Jingyi Zhang, Yanbo Fan, Yujiu Yang

**Abstract**: In the scenario of black-box adversarial attack, the target model's parameters are unknown, and the attacker aims to find a successful adversarial perturbation based on query feedback under a query budget. Due to the limited feedback information, existing query-based black-box attack methods often require many queries for attacking each benign example. To reduce query cost, we propose to utilize the feedback information across historical attacks, dubbed example-level adversarial transferability. Specifically, by treating the attack on each benign example as one task, we develop a meta-learning framework by training a meta-generator to produce perturbations conditioned on benign examples. When attacking a new benign example, the meta generator can be quickly fine-tuned based on the feedback information of the new task as well as a few historical attacks to produce effective perturbations. Moreover, since the meta-train procedure consumes many queries to learn a generalizable generator, we utilize model-level adversarial transferability to train the meta-generator on a white-box surrogate model, then transfer it to help the attack against the target model. The proposed framework with the two types of adversarial transferability can be naturally combined with any off-the-shelf query-based attack methods to boost their performance, which is verified by extensive experiments.

摘要: 在黑盒对抗性攻击场景中，目标模型的参数未知，攻击者的目标是在查询预算内根据查询反馈找到一个成功的对抗性扰动。由于反馈信息有限，现有的基于查询的黑盒攻击方法往往需要多次查询才能攻击每个良性实例。为了降低查询代价，我们提出了利用历史攻击中的反馈信息，称为实例级对抗性转移。具体地说，通过将对每个良性样本的攻击视为一个任务，我们通过训练元生成器来产生以良性样本为条件的扰动，从而开发了一个元学习框架。当攻击新的良性示例时，元生成器可以根据新任务的反馈信息以及一些历史攻击来快速微调，以产生有效的扰动。此外，由于元训练过程需要消耗大量的查询来学习可泛化的生成器，我们利用模型级的对抗性转移来训练白盒代理模型上的元生成器，然后将其转移以帮助对目标模型的攻击。该框架具有两种对抗性可转移性，可以自然地与任何现有的基于查询的攻击方法相结合，从而提高其性能，这一点得到了广泛的实验验证。



## **50. ExploreADV: Towards exploratory attack for Neural Networks**

ExplreADV：对神经网络的探索性攻击 cs.CR

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.01223v1) [paper-pdf](http://arxiv.org/pdf/2301.01223v1)

**Authors**: Tianzuo Luo, Yuyi Zhong, Siaucheng Khoo

**Abstract**: Although deep learning has made remarkable progress in processing various types of data such as images, text and speech, they are known to be susceptible to adversarial perturbations: perturbations specifically designed and added to the input to make the target model produce erroneous output. Most of the existing studies on generating adversarial perturbations attempt to perturb the entire input indiscriminately. In this paper, we propose ExploreADV, a general and flexible adversarial attack system that is capable of modeling regional and imperceptible attacks, allowing users to explore various kinds of adversarial examples as needed. We adapt and combine two existing boundary attack methods, DeepFool and Brendel\&Bethge Attack, and propose a mask-constrained adversarial attack system, which generates minimal adversarial perturbations under the pixel-level constraints, namely ``mask-constraints''. We study different ways of generating such mask-constraints considering the variance and importance of the input features, and show that our adversarial attack system offers users good flexibility to focus on sub-regions of inputs, explore imperceptible perturbations and understand the vulnerability of pixels/regions to adversarial attacks. We demonstrate our system to be effective based on extensive experiments and user study.

摘要: 虽然深度学习在处理图像、文本和语音等各种类型的数据方面取得了显著的进展，但众所周知，它们容易受到对抗性扰动的影响：这些扰动是专门设计并添加到输入中的，以使目标模型产生错误的输出。现有的关于产生对抗性扰动的研究大多试图不加区别地扰乱整个输入。本文提出了一种通用的、灵活的对抗性攻击系统DevelopreADV，它能够对局部的、不可察觉的攻击进行建模，允许用户根据需要探索各种对抗性的例子。本文对现有的两种边界攻击方法DeepFool和Brendel-Bethge进行了改进和结合，提出了一种基于掩码约束的对抗性攻击系统，该系统在像素级约束下产生最小的对抗性扰动，即“掩码约束”。考虑到输入特征的方差和重要性，我们研究了不同的生成掩码约束的方法，并表明我们的对抗性攻击系统为用户提供了良好的灵活性，使用户能够专注于输入子区域，探索不可察觉的扰动，并了解像素/区域对对抗性攻击的脆弱性。通过大量的实验和用户研究，我们证明了我们的系统是有效的。



