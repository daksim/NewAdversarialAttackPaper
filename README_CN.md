# Latest Adversarial Attack Papers
**update at 2022-10-26 06:31:39**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Probabilistic Categorical Adversarial Attack & Adversarial Training**

概率分类对抗性攻击与对抗性训练 cs.LG

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.09364v2) [paper-pdf](http://arxiv.org/pdf/2210.09364v2)

**Authors**: Pengfei He, Han Xu, Jie Ren, Yuxuan Wan, Zitao Liu, Jiliang Tang

**Abstract**: The existence of adversarial examples brings huge concern for people to apply Deep Neural Networks (DNNs) in safety-critical tasks. However, how to generate adversarial examples with categorical data is an important problem but lack of extensive exploration. Previously established methods leverage greedy search method, which can be very time-consuming to conduct successful attack. This also limits the development of adversarial training and potential defenses for categorical data. To tackle this problem, we propose Probabilistic Categorical Adversarial Attack (PCAA), which transfers the discrete optimization problem to a continuous problem that can be solved efficiently by Projected Gradient Descent. In our paper, we theoretically analyze its optimality and time complexity to demonstrate its significant advantage over current greedy based attacks. Moreover, based on our attack, we propose an efficient adversarial training framework. Through a comprehensive empirical study, we justify the effectiveness of our proposed attack and defense algorithms.

摘要: 对抗性实例的存在给深度神经网络在安全关键任务中的应用带来了极大的关注。然而，如何利用分类数据生成对抗性实例是一个重要的问题，但缺乏广泛的探索。以前建立的方法利用贪婪搜索方法，进行成功的攻击可能非常耗时。这也限制了对抗性训练的发展和对分类数据的潜在防御。为了解决这个问题，我们提出了概率分类对抗性攻击(PCAA)，它将离散的优化问题转化为一个连续的问题，可以用投影梯度下降法有效地解决。在本文中，我们从理论上分析了它的最优性和时间复杂性，以证明它相对于现有的基于贪婪的攻击具有显著的优势。此外，基于我们的攻击，我们提出了一个有效的对抗性训练框架。通过全面的实证研究，验证了本文提出的攻防算法的有效性。



## **2. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.13263v1) [paper-pdf](http://arxiv.org/pdf/2210.13263v1)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **3. SealClub: Computer-aided Paper Document Authentication**

SealClub：计算机辅助纸质文档认证 cs.CR

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.07884v2) [paper-pdf](http://arxiv.org/pdf/2210.07884v2)

**Authors**: Martín Ochoa, Jorge Toro-Pozo, David Basin

**Abstract**: Digital authentication is a mature field, offering a range of solutions with rigorous mathematical guarantees. Nevertheless, paper documents, where cryptographic techniques are not directly applicable, are still widely utilized due to usability and legal reasons. We propose a novel approach to authenticating paper documents using smartphones by taking short videos of them. Our solution combines cryptographic and image comparison techniques to detect and highlight subtle semantic-changing attacks on rich documents, containing text and graphics, that could go unnoticed by humans. We rigorously analyze our approach, proving that it is secure against strong adversaries capable of compromising different system components. We also measure its accuracy empirically on a set of 128 videos of paper documents, half containing subtle forgeries. Our algorithm finds all forgeries accurately (no false alarms) after analyzing 5.13 frames on average (corresponding to 1.28 seconds of video). Highlighted regions are large enough to be visible to users, but small enough to precisely locate forgeries. Thus, our approach provides a promising way for users to authenticate paper documents using conventional smartphones under realistic conditions.

摘要: 数字身份验证是一个成熟的领域，提供了一系列具有严格数学保证的解决方案。然而，由于可用性和法律原因，在加密技术不直接适用的情况下，纸质文档仍然被广泛使用。我们提出了一种通过拍摄短视频来使用智能手机对纸质文档进行身份验证的新方法。我们的解决方案结合了加密和图像比较技术，以检测和突出对包含文本和图形的丰富文档的细微语义变化攻击，这些攻击可能不会被人类注意到。我们严格分析了我们的方法，证明了它是安全的，可以抵御能够危害不同系统组件的强大对手。我们还在一组128个纸质文档的视频上对其准确性进行了经验性的测量，其中一半包含微妙的伪造。该算法在平均分析5.13帧(对应于1.28秒的视频)后，准确地发现了所有的伪造(没有虚警)。突出显示的区域足够大，用户可以看到，但也足够小，可以精确定位假货。因此，我们的方法为用户在现实条件下使用传统的智能手机认证纸质文档提供了一种很有前途的方法。



## **4. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.08189v3) [paper-pdf](http://arxiv.org/pdf/2204.08189v3)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **5. Ares: A System-Oriented Wargame Framework for Adversarial ML**

ARES：一种面向系统的对抗性ML战争游戏框架 cs.LG

Presented at the DLS Workshop at S&P 2022

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2210.12952v1) [paper-pdf](http://arxiv.org/pdf/2210.12952v1)

**Authors**: Farhan Ahmed, Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstract**: Since the discovery of adversarial attacks against machine learning models nearly a decade ago, research on adversarial machine learning has rapidly evolved into an eternal war between defenders, who seek to increase the robustness of ML models against adversarial attacks, and adversaries, who seek to develop better attacks capable of weakening or defeating these defenses. This domain, however, has found little buy-in from ML practitioners, who are neither overtly concerned about these attacks affecting their systems in the real world nor are willing to trade off the accuracy of their models in pursuit of robustness against these attacks.   In this paper, we motivate the design and implementation of Ares, an evaluation framework for adversarial ML that allows researchers to explore attacks and defenses in a realistic wargame-like environment. Ares frames the conflict between the attacker and defender as two agents in a reinforcement learning environment with opposing objectives. This allows the introduction of system-level evaluation metrics such as time to failure and evaluation of complex strategies such as moving target defenses. We provide the results of our initial exploration involving a white-box attacker against an adversarially trained defender.

摘要: 自从近十年前发现了针对机器学习模型的对抗性攻击以来，对抗性机器学习的研究迅速演变为防御者和对手之间的一场永恒的战争。防御者试图增加ML模型对对抗性攻击的健壮性，而对手试图开发能够削弱或击败这些防御的更好的攻击。然而，这个领域几乎没有得到ML从业者的认可，他们既不公开担心这些攻击会影响他们在现实世界中的系统，也不愿意牺牲他们模型的准确性来追求对这些攻击的健壮性。在本文中，我们推动了ARES的设计和实现，这是一个针对对抗性ML的评估框架，允许研究人员在现实的类似战争游戏的环境中探索攻击和防御。阿瑞斯将攻击者和防御者之间的冲突框架为具有相反目标的强化学习环境中的两个代理。这允许引入系统级评估指标，如故障发生时间，以及评估复杂战略，如移动目标防御。我们提供了我们的初步探索的结果，涉及一个白盒攻击者对一个对手训练的后卫。



## **6. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

基于稀有嵌入和梯度集成的联合学习后门攻击 cs.LG

Accepted to EMNLP 2022, 9 pages and Appendix

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2204.14017v2) [paper-pdf](http://arxiv.org/pdf/2204.14017v2)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstract**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through rare word embeddings of NLP models. In text classification, less than 1% of adversary clients suffices to manipulate the model output without any drop in the performance on clean sentences. For a less complex dataset, a mere 0.1% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called Gradient Ensemble, which enhances the backdoor performance in all our experimental settings.

摘要: 联邦学习的最新进展已经证明了它在分散的数据集上学习的前景。然而，由于参与该框架的对手出于对抗目的而破坏全球模式的潜在风险，大量工作引起了关注。通过对自然语言处理模型的稀有词嵌入，研究了模型中毒用于后门攻击的可行性。在文本分类中，只有不到1%的敌意客户端足以在不降低干净句子性能的情况下操纵模型输出。对于不太复杂的数据集，仅0.1%的恶意客户端就足以有效地毒化全球模型。我们还提出了一种专门用于联邦学习方案的技术，称为梯度集成，它在所有实验设置中都提高了后门性能。



## **7. TextHacker: Learning based Hybrid Local Search Algorithm for Text Hard-label Adversarial Attack**

TextHacker：用于文本硬标签攻击的基于学习的混合局部搜索算法 cs.CL

Accepted by EMNLP 2022 Findings, Code is available at  https://github.com/JHL-HUST/TextHacker

**SubmitDate**: 2022-10-24    [abs](http://arxiv.org/abs/2201.08193v2) [paper-pdf](http://arxiv.org/pdf/2201.08193v2)

**Authors**: Zhen Yu, Xiaosen Wang, Wanxiang Che, Kun He

**Abstract**: Existing textual adversarial attacks usually utilize the gradient or prediction confidence to generate adversarial examples, making it hard to be deployed in real-world applications. To this end, we consider a rarely investigated but more rigorous setting, namely hard-label attack, in which the attacker can only access the prediction label. In particular, we find we can learn the importance of different words via the change on prediction label caused by word substitutions on the adversarial examples. Based on this observation, we propose a novel adversarial attack, termed Text Hard-label attacker (TextHacker). TextHacker randomly perturbs lots of words to craft an adversarial example. Then, TextHacker adopts a hybrid local search algorithm with the estimation of word importance from the attack history to minimize the adversarial perturbation. Extensive evaluations for text classification and textual entailment show that TextHacker significantly outperforms existing hard-label attacks regarding the attack performance as well as adversary quality.

摘要: 现有的文本对抗性攻击通常利用梯度或预测置信度来生成对抗性实例，这使得它很难应用于实际应用中。为此，我们考虑了一种很少被研究但更严格的环境，即硬标签攻击，在这种攻击中，攻击者只能访问预测标签。特别是，我们发现我们可以通过对抗性例子上的单词替换引起的预测标签的变化来了解不同单词的重要性。基于此，我们提出了一种新的对抗性攻击，称为文本硬标签攻击者(TextHacker)。TextHacker随机扰乱大量单词来制作一个对抗性的例子。然后，TextHacker采用了一种混合局部搜索算法，并从攻击历史中估计单词的重要性，以最小化对手的扰动。对文本分类和文本蕴涵的广泛评估表明，TextHacker在攻击性能和对手质量方面都明显优于现有的硬标签攻击。



## **8. A Secure Design Pattern Approach Toward Tackling Lateral-Injection Attacks**

一种应对横向注入攻击的安全设计模式方法 cs.CR

4 pages, 3 figures. Accepted to The 15th IEEE International  Conference on Security of Information and Networks (SIN)

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12877v1) [paper-pdf](http://arxiv.org/pdf/2210.12877v1)

**Authors**: Chidera Biringa, Gökhan Kul

**Abstract**: Software weaknesses that create attack surfaces for adversarial exploits, such as lateral SQL injection (LSQLi) attacks, are usually introduced during the design phase of software development. Security design patterns are sometimes applied to tackle these weaknesses. However, due to the stealthy nature of lateral-based attacks, employing traditional security patterns to address these threats is insufficient. Hence, we present SEAL, a secure design that extrapolates architectural, design, and implementation abstraction levels to delegate security strategies toward tackling LSQLi attacks. We evaluated SEAL using case study software, where we assumed the role of an adversary and injected several attack vectors tasked with compromising the confidentiality and integrity of its database. Our evaluation of SEAL demonstrated its capacity to address LSQLi attacks.

摘要: 在软件开发的设计阶段，通常会引入为对抗性攻击(如横向SQL注入(LSQLi)攻击)创建攻击面的软件弱点。有时会应用安全设计模式来解决这些弱点。然而，由于基于侧向的攻击的隐蔽性，采用传统的安全模式来应对这些威胁是不够的。因此，我们提出了SEAL，这是一种安全设计，它推断出体系结构、设计和实现抽象级别，以委派安全策略来应对LSQLi攻击。我们使用案例研究软件评估了海豹突击队，我们扮演了一个对手的角色，并注入了几个攻击载体，任务是危及其数据库的机密性和完整性。我们对海豹突击队的评估表明，它有能力应对LSQLi攻击。



## **9. TAPE: Assessing Few-shot Russian Language Understanding**

录像带：评估不太可能的俄语理解 cs.CL

Accepted to EMNLP 2022 Findings

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12813v1) [paper-pdf](http://arxiv.org/pdf/2210.12813v1)

**Authors**: Ekaterina Taktasheva, Tatiana Shavrina, Alena Fenogenova, Denis Shevelev, Nadezhda Katricheva, Maria Tikhonova, Albina Akhmetgareeva, Oleg Zinkevich, Anastasiia Bashmakova, Svetlana Iordanskaia, Alena Spiridonova, Valentina Kurenshchikova, Ekaterina Artemova, Vladislav Mikhailov

**Abstract**: Recent advances in zero-shot and few-shot learning have shown promise for a scope of research and practical purposes. However, this fast-growing area lacks standardized evaluation suites for non-English languages, hindering progress outside the Anglo-centric paradigm. To address this line of research, we propose TAPE (Text Attack and Perturbation Evaluation), a novel benchmark that includes six more complex NLU tasks for Russian, covering multi-hop reasoning, ethical concepts, logic and commonsense knowledge. The TAPE's design focuses on systematic zero-shot and few-shot NLU evaluation: (i) linguistic-oriented adversarial attacks and perturbations for analyzing robustness, and (ii) subpopulations for nuanced interpretation. The detailed analysis of testing the autoregressive baselines indicates that simple spelling-based perturbations affect the performance the most, while paraphrasing the input has a more negligible effect. At the same time, the results demonstrate a significant gap between the neural and human baselines for most tasks. We publicly release TAPE (tape-benchmark.com) to foster research on robust LMs that can generalize to new tasks when little to no supervision is available.

摘要: 零射击和少射击学习的最新进展显示出了研究范围和实用目的的前景。然而，这个快速发展的领域缺乏针对非英语语言的标准化评估套件，阻碍了以英语为中心的范式之外的进步。针对这一研究方向，我们提出了TAPE(文本攻击和扰动评估)，这是一个新的基准测试，包括六个更复杂的俄语自然语言理解任务，涵盖了多跳推理、伦理概念、逻辑和常识知识。这盘磁带的设计侧重于系统的零镜头和少镜头NLU评估：(I)面向语言的对抗性攻击和扰动，用于分析稳健性；(Ii)亚群，用于细微差别的解释。对自回归基线测试的详细分析表明，基于拼写的简单扰动对成绩的影响最大，而释义输入的影响较小。与此同时，结果表明，在大多数任务中，神经基线和人类基线之间存在着显著的差距。我们公开发布磁带(Tape-Benchmark.com)，以促进对健壮的LMS的研究，这些LMS可以在几乎没有监督的情况下推广到新任务。



## **10. GANI: Global Attacks on Graph Neural Networks via Imperceptible Node Injections**

GANI：基于不可察觉节点注入的图神经网络全局攻击 cs.LG

**SubmitDate**: 2022-10-23    [abs](http://arxiv.org/abs/2210.12598v1) [paper-pdf](http://arxiv.org/pdf/2210.12598v1)

**Authors**: Junyuan Fang, Haixian Wen, Jiajing Wu, Qi Xuan, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks (GNNs) have found successful applications in various graph-related tasks. However, recent studies have shown that many GNNs are vulnerable to adversarial attacks. In a vast majority of existing studies, adversarial attacks on GNNs are launched via direct modification of the original graph such as adding/removing links, which may not be applicable in practice. In this paper, we focus on a realistic attack operation via injecting fake nodes. The proposed Global Attack strategy via Node Injection (GANI) is designed under the comprehensive consideration of an unnoticeable perturbation setting from both structure and feature domains. Specifically, to make the node injections as imperceptible and effective as possible, we propose a sampling operation to determine the degree of the newly injected nodes, and then generate features and select neighbors for these injected nodes based on the statistical information of features and evolutionary perturbations obtained from a genetic algorithm, respectively. In particular, the proposed feature generation mechanism is suitable for both binary and continuous node features. Extensive experimental results on benchmark datasets against both general and defended GNNs show strong attack performance of GANI. Moreover, the imperceptibility analyses also demonstrate that GANI achieves a relatively unnoticeable injection on benchmark datasets.

摘要: 图神经网络(GNN)在各种与图相关的任务中得到了成功的应用。然而，最近的研究表明，许多GNN容易受到对抗性攻击。在现有的绝大多数研究中，对GNN的对抗性攻击是通过直接修改原始图形来发起的，例如添加/删除链接，这在实践中可能并不适用。在本文中，我们关注的是一种通过注入伪节点进行的真实攻击操作。通过节点注入的全局攻击策略(GANI)是在综合考虑结构域和特征域中不可察觉的扰动设置的基础上设计的。具体地说，为了使节点注入尽可能隐蔽和有效，我们提出了一种抽样操作来确定新注入节点的程度，然后根据遗传算法获得的特征统计信息和进化扰动分别为这些注入节点生成特征和选择邻居。特别是，所提出的特征生成机制既适用于二进制节点特征，也适用于连续节点特征。在基准数据集上对一般GNN和防御GNN的大量实验结果表明，GANI具有很强的攻击性能。此外，不可感知性分析还表明，GANI在基准数据集上实现了相对不明显的注入。



## **11. Efficient (Soft) Q-Learning for Text Generation with Limited Good Data**

有效(软)Q-学习在有限好数据下的文本生成 cs.CL

Code available at  https://github.com/HanGuo97/soft-Q-learning-for-text-generation

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2106.07704v4) [paper-pdf](http://arxiv.org/pdf/2106.07704v4)

**Authors**: Han Guo, Bowen Tan, Zhengzhong Liu, Eric P. Xing, Zhiting Hu

**Abstract**: Maximum likelihood estimation (MLE) is the predominant algorithm for training text generation models. This paradigm relies on direct supervision examples, which is not applicable to many emerging applications, such as generating adversarial attacks or generating prompts to control language models. Reinforcement learning (RL) on the other hand offers a more flexible solution by allowing users to plug in arbitrary task metrics as reward. Yet previous RL algorithms for text generation, such as policy gradient (on-policy RL) and Q-learning (off-policy RL), are often notoriously inefficient or unstable to train due to the large sequence space and the sparse reward received only at the end of sequences. In this paper, we introduce a new RL formulation for text generation from the soft Q-learning (SQL) perspective. It enables us to draw from the latest RL advances, such as path consistency learning, to combine the best of on-/off-policy updates, and learn effectively from sparse reward. We apply the approach to a wide range of novel text generation tasks, including learning from noisy/negative examples, adversarial attacks, and prompt generation. Experiments show our approach consistently outperforms both task-specialized algorithms and the previous RL methods.

摘要: 最大似然估计(MLE)是训练文本生成模型的主要算法。这种模式依赖于直接监督示例，这不适用于许多新兴的应用程序，例如生成对抗性攻击或生成提示以控制语言模型。另一方面，强化学习(RL)提供了一种更灵活的解决方案，允许用户插入任意任务指标作为奖励。然而，先前用于文本生成的RL算法，例如策略梯度(On-Policy RL)和Q-学习(Off-Policy RL)，由于大的序列空间和仅在序列末尾接收的稀疏回报，训练起来常常是出了名的低效或不稳定。本文从软Q-学习(SQL)的角度出发，提出了一种新的RL文本生成方法。它使我们能够借鉴RL的最新进展，如路径一致性学习，结合最好的开/关策略更新，并从稀疏奖励中有效学习。我们将该方法应用于一系列新颖的文本生成任务，包括从噪声/负面示例中学习、对抗性攻击和提示生成。实验表明，我们的方法的性能始终优于任务专门化算法和以前的RL方法。



## **12. RORL: Robust Offline Reinforcement Learning via Conservative Smoothing**

RORL：基于保守平滑的稳健离线强化学习 cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS) 2022

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2206.02829v3) [paper-pdf](http://arxiv.org/pdf/2206.02829v3)

**Authors**: Rui Yang, Chenjia Bai, Xiaoteng Ma, Zhaoran Wang, Chongjie Zhang, Lei Han

**Abstract**: Offline reinforcement learning (RL) provides a promising direction to exploit massive amount of offline data for complex decision-making tasks. Due to the distribution shift issue, current offline RL algorithms are generally designed to be conservative in value estimation and action selection. However, such conservatism can impair the robustness of learned policies when encountering observation deviation under realistic conditions, such as sensor errors and adversarial attacks. To trade off robustness and conservatism, we propose Robust Offline Reinforcement Learning (RORL) with a novel conservative smoothing technique. In RORL, we explicitly introduce regularization on the policy and the value function for states near the dataset, as well as additional conservative value estimation on these states. Theoretically, we show RORL enjoys a tighter suboptimality bound than recent theoretical results in linear MDPs. We demonstrate that RORL can achieve state-of-the-art performance on the general offline RL benchmark and is considerably robust to adversarial observation perturbations.

摘要: 离线强化学习(RL)为利用大量的离线数据进行复杂的决策任务提供了一个很有前途的方向。由于分布平移问题，目前的离线RL算法在价值估计和动作选择上通常被设计为保守的。然而，当在实际条件下遇到观测偏差时，这种保守性会削弱学习策略的稳健性，例如传感器错误和对抗性攻击。为了权衡稳健性和保守性，我们提出了一种新的保守平滑技术的稳健离线强化学习(RORL)。在RORL中，我们明确地引入了关于策略的正则化和数据集附近状态的值函数，以及关于这些状态的附加保守值估计。理论上，我们证明了RORL在线性MDP中享有比最近的理论结果更紧的次优界。我们证明了RORL可以在一般的离线RL基准上获得最先进的性能，并且对对抗性观测扰动具有相当强的鲁棒性。



## **13. ADDMU: Detection of Far-Boundary Adversarial Examples with Data and Model Uncertainty Estimation**

ADDMU：用数据检测远边界对抗性实例和模型不确定性估计 cs.CL

18 pages, EMNLP 2022, main conference, long paper

**SubmitDate**: 2022-10-22    [abs](http://arxiv.org/abs/2210.12396v1) [paper-pdf](http://arxiv.org/pdf/2210.12396v1)

**Authors**: Fan Yin, Yao Li, Cho-Jui Hsieh, Kai-Wei Chang

**Abstract**: Adversarial Examples Detection (AED) is a crucial defense technique against adversarial attacks and has drawn increasing attention from the Natural Language Processing (NLP) community. Despite the surge of new AED methods, our studies show that existing methods heavily rely on a shortcut to achieve good performance. In other words, current search-based adversarial attacks in NLP stop once model predictions change, and thus most adversarial examples generated by those attacks are located near model decision boundaries. To surpass this shortcut and fairly evaluate AED methods, we propose to test AED methods with \textbf{F}ar \textbf{B}oundary (\textbf{FB}) adversarial examples. Existing methods show worse than random guess performance under this scenario. To overcome this limitation, we propose a new technique, \textbf{ADDMU}, \textbf{a}dversary \textbf{d}etection with \textbf{d}ata and \textbf{m}odel \textbf{u}ncertainty, which combines two types of uncertainty estimation for both regular and FB adversarial example detection. Our new method outperforms previous methods by 3.6 and 6.0 \emph{AUC} points under each scenario. Finally, our analysis shows that the two types of uncertainty provided by \textbf{ADDMU} can be leveraged to characterize adversarial examples and identify the ones that contribute most to model's robustness in adversarial training.

摘要: 对抗性实例检测(AED)是对抗敌意攻击的一种重要防御技术，越来越受到自然语言处理(NLP)领域的关注。尽管新的AED方法激增，但我们的研究表明，现有方法严重依赖于一条捷径来获得良好的性能。换言之，当前NLP中基于搜索的对抗性攻击一旦模型预测发生变化就会停止，因此由这些攻击生成的大多数对抗性示例都位于模型决策边界附近。为了超越这一捷径并公平地评价AED方法，我们建议用对抗性的例子来测试AED方法。在这种情况下，现有的方法表现出比随机猜测更差的性能。为了克服这一局限性，我们提出了一种新的方法在每种情况下，我们的新方法比以前的方法分别提高3.6和6.0emph{AUC}点。最后，我们的分析表明，文本bf{ADDMU}提供的两种类型的不确定性可以用来刻画对抗性例子，并确定对对抗性训练中模型的稳健性贡献最大的那些。



## **14. TCAB: A Large-Scale Text Classification Attack Benchmark**

TCAB：一种大规模文本分类攻击基准 cs.LG

32 pages, 7 figures, and 14 tables

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12233v1) [paper-pdf](http://arxiv.org/pdf/2210.12233v1)

**Authors**: Kalyani Asthana, Zhouhang Xie, Wencong You, Adam Noack, Jonathan Brophy, Sameer Singh, Daniel Lowd

**Abstract**: We introduce the Text Classification Attack Benchmark (TCAB), a dataset for analyzing, understanding, detecting, and labeling adversarial attacks against text classifiers. TCAB includes 1.5 million attack instances, generated by twelve adversarial attacks targeting three classifiers trained on six source datasets for sentiment analysis and abuse detection in English. Unlike standard text classification, text attacks must be understood in the context of the target classifier that is being attacked, and thus features of the target classifier are important as well. TCAB includes all attack instances that are successful in flipping the predicted label; a subset of the attacks are also labeled by human annotators to determine how frequently the primary semantics are preserved. The process of generating attacks is automated, so that TCAB can easily be extended to incorporate new text attacks and better classifiers as they are developed. In addition to the primary tasks of detecting and labeling attacks, TCAB can also be used for attack localization, attack target labeling, and attack characterization. TCAB code and dataset are available at https://react-nlp.github.io/tcab/.

摘要: 我们介绍了文本分类攻击基准(TCAB)，这是一个用于分析、理解、检测和标记针对文本分类器的敌意攻击的数据集。TCAB包括150万个攻击实例，由12个针对三个分类器的对抗性攻击生成，这些分类器在六个源数据集上进行训练，用于英语情感分析和滥用检测。与标准文本分类不同，文本攻击必须在被攻击的目标分类器的上下文中理解，因此目标分类器的特征也很重要。TCAB包括成功翻转预测标签的所有攻击实例；人工注释员还标记攻击的子集，以确定保留主要语义的频率。生成攻击的过程是自动化的，因此TCAB可以很容易地进行扩展，以纳入新的文本攻击和开发的更好的分类器。除了检测和标记攻击的主要任务外，TCAB还可以用于攻击定位、攻击目标标记和攻击特征描述。TCAB代码和数据集可在https://react-nlp.github.io/tcab/.上获得



## **15. The Dark Side of AutoML: Towards Architectural Backdoor Search**

AutoML的黑暗面：走向建筑后门搜索 cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12179v1) [paper-pdf](http://arxiv.org/pdf/2210.12179v1)

**Authors**: Ren Pang, Changjiang Li, Zhaohan Xi, Shouling Ji, Ting Wang

**Abstract**: This paper asks the intriguing question: is it possible to exploit neural architecture search (NAS) as a new attack vector to launch previously improbable attacks? Specifically, we present EVAS, a new attack that leverages NAS to find neural architectures with inherent backdoors and exploits such vulnerability using input-aware triggers. Compared with existing attacks, EVAS demonstrates many interesting properties: (i) it does not require polluting training data or perturbing model parameters; (ii) it is agnostic to downstream fine-tuning or even re-training from scratch; (iii) it naturally evades defenses that rely on inspecting model parameters or training data. With extensive evaluation on benchmark datasets, we show that EVAS features high evasiveness, transferability, and robustness, thereby expanding the adversary's design spectrum. We further characterize the mechanisms underlying EVAS, which are possibly explainable by architecture-level ``shortcuts'' that recognize trigger patterns. This work raises concerns about the current practice of NAS and points to potential directions to develop effective countermeasures.

摘要: 这篇论文提出了一个耐人寻味的问题：是否有可能利用神经结构搜索(NAS)作为一种新的攻击载体来发动以前不太可能的攻击？具体地说，我们提出了EVA，这是一种新的攻击，它利用NAS来发现具有固有后门的神经体系结构，并使用输入感知触发器来利用这种漏洞。与现有的攻击相比，EVA表现出许多有趣的性质：(I)它不需要污染训练数据或扰动模型参数；(Ii)它与下游微调甚至从头开始的重新训练无关；(Iii)它自然地避开了依赖于检查模型参数或训练数据的防御。通过在基准数据集上的广泛评估，我们发现EVA具有高度的规避、可转移性和健壮性，从而扩展了对手的设计范围。我们进一步描述了EVA背后的机制，这可能可以通过识别触发模式的架构级“捷径”来解释。这项工作引起了人们对当前NAS实践的关注，并指出了制定有效对策的潜在方向。



## **16. Evolution of Neural Tangent Kernels under Benign and Adversarial Training**

良性训练和对抗性训练下神经正切核的演化 cs.LG

Accepted to the Conference on Advances in Neural Information  Processing Systems (NeurIPS) 2022

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2210.12030v1) [paper-pdf](http://arxiv.org/pdf/2210.12030v1)

**Authors**: Noel Loo, Ramin Hasani, Alexander Amini, Daniela Rus

**Abstract**: Two key challenges facing modern deep learning are mitigating deep networks' vulnerability to adversarial attacks and understanding deep learning's generalization capabilities. Towards the first issue, many defense strategies have been developed, with the most common being Adversarial Training (AT). Towards the second challenge, one of the dominant theories that has emerged is the Neural Tangent Kernel (NTK) -- a characterization of neural network behavior in the infinite-width limit. In this limit, the kernel is frozen, and the underlying feature map is fixed. In finite widths, however, there is evidence that feature learning happens at the earlier stages of the training (kernel learning) before a second phase where the kernel remains fixed (lazy training). While prior work has aimed at studying adversarial vulnerability through the lens of the frozen infinite-width NTK, there is no work that studies the adversarial robustness of the empirical/finite NTK during training. In this work, we perform an empirical study of the evolution of the empirical NTK under standard and adversarial training, aiming to disambiguate the effect of adversarial training on kernel learning and lazy training. We find under adversarial training, the empirical NTK rapidly converges to a different kernel (and feature map) than standard training. This new kernel provides adversarial robustness, even when non-robust training is performed on top of it. Furthermore, we find that adversarial training on top of a fixed kernel can yield a classifier with $76.1\%$ robust accuracy under PGD attacks with $\varepsilon = 4/255$ on CIFAR-10.

摘要: 现代深度学习面临的两个关键挑战是减轻深度网络对对手攻击的脆弱性和理解深度学习的泛化能力。对于第一个问题，已经开发了许多防御策略，其中最常见的是对手训练(AT)。对于第二个挑战，已经出现的占主导地位的理论之一是神经切线核(NTK)--一种对无限宽度限制中的神经网络行为的描述。在此限制下，内核被冻结，底层功能映射被修复。然而，在有限的范围内，有证据表明，特征学习发生在训练的早期阶段(核学习)，然后是核保持固定的第二阶段(懒惰训练)。虽然以前的工作旨在通过冻结的无限宽度NTK的透镜来研究对抗脆弱性，但还没有研究经验/有限NTK在训练过程中的对抗稳健性的工作。在这项工作中，我们对经验NTK在标准训练和对抗性训练下的演化进行了实证研究，旨在消除对抗性训练对核学习和懒惰训练的影响。我们发现，在对抗性训练下，经验NTK迅速收敛到与标准训练不同的核(和特征映射)。这个新的内核提供了对手的健壮性，即使在它上面执行了非健壮的训练。此外，我们发现在固定核上的对抗性训练可以产生一个在CIFAR-10上$varepsilon=4/255$的PGD攻击下具有$76.1\$稳健精度的分类器。



## **17. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.03755v2) [paper-pdf](http://arxiv.org/pdf/2209.03755v2)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息现在是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，一个可行的解决方案是通过检索和核实相关证据来自动化索赔的事实核查。虽然最近在推动自动事实核查方面取得了重大进展，但仍然缺乏对针对这类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，除了生成多样化的和与索赔一致的证据外，还可以微妙地修改证据中突出索赔的片段。因此，在分类维度的许多不同排列下，我们会极大地降低事实检查性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，我们最后讨论了未来防御的挑战和方向。



## **18. Assaying Out-Of-Distribution Generalization in Transfer Learning**

迁移学习中的分布外泛化分析 cs.LG

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2207.09239v2) [paper-pdf](http://arxiv.org/pdf/2207.09239v2)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstract**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.

摘要: 由于分布外泛化是一个通常不适定的问题，因此对不同的代理目标(例如，校准、对手健壮性、算法腐败、跨班次不变性)进行了研究，得出了不同的建议。虽然有着相同的理想目标，但这些方法从未在相同的实验条件下对真实数据进行过测试。在本文中，我们对以前的工作进行了统一的审查，强调了我们通过经验解决的信息差异，并就如何衡量模型的稳健性以及如何改进模型提供了建议。为此，我们收集了172个公开可用的数据集对，用于训练和分布外评估准确性、校准误差、对抗性攻击、环境不变性和合成腐败。我们微调了超过31k的网络，这些网络来自9种不同的架构，在多发和少发的情况下。我们的发现证实，分布内和分布外的精度往往会共同增加，但表明它们的关系在很大程度上依赖于数据集，总体上比之前的较小规模的研究假设的更细微和更复杂。



## **19. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

discuss new and recent works as well as proof-reading

**SubmitDate**: 2022-10-21    [abs](http://arxiv.org/abs/2209.02299v5) [paper-pdf](http://arxiv.org/pdf/2209.02299v5)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstract**: Today, computer systems hold large amounts of personal data. Yet while such an abundance of data allows breakthroughs in artificial intelligence, and especially machine learning (ML), its existence can be a threat to user privacy, and it can weaken the bonds of trust between humans and AI. Recent regulations now require that, on request, private information about a user must be removed from both computer systems and from ML models, i.e. ``the right to be forgotten''). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often `remember' the old data. Contemporary adversarial attacks on trained models have proven that we can learn whether an instance or an attribute belonged to the training data. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to completely solve the problem due to the lack of common frameworks and resources. Therefore, this paper aspires to present a comprehensive examination of machine unlearning's concepts, scenarios, methods, and applications. Specifically, as a category collection of cutting-edge studies, the intention behind this article is to serve as a comprehensive resource for researchers and practitioners seeking an introduction to machine unlearning and its formulations, design criteria, removal requests, algorithms, and applications. In addition, we aim to highlight the key findings, current trends, and new research areas that have not yet featured the use of machine unlearning but could benefit greatly from it. We hope this survey serves as a valuable resource for ML researchers and those seeking to innovate privacy technologies. Our resources are publicly available at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 今天，计算机系统保存着大量的个人数据。然而，尽管如此丰富的数据使人工智能，特别是机器学习(ML)取得了突破，但它的存在可能会对用户隐私构成威胁，并可能削弱人类与人工智能之间的信任纽带。最近的法规现在要求，根据请求，必须从计算机系统和ML模型中删除关于用户的私人信息，即“被遗忘权”)。虽然从后端数据库中删除数据应该是直接的，但在人工智能上下文中这是不够的，因为ML模型经常‘记住’旧数据。当代针对训练模型的对抗性攻击已经证明，我们可以学习到一个实例或一个属性是否属于训练数据。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。因此，本文致力于对机器遗忘的概念、场景、方法和应用进行全面的考察。具体地说，作为尖端研究的类别集合，本文背后的目的是为寻求介绍机器遗忘及其公式、设计标准、移除请求、算法和应用的研究人员和从业者提供全面的资源。此外，我们的目标是强调关键的发现、当前的趋势和新的研究领域，这些领域还没有使用机器遗忘，但可以从中受益匪浅。我们希望这项调查对ML研究人员和那些寻求创新隐私技术的人来说是一个有价值的资源。我们的资源可在https://github.com/tamlhp/awesome-machine-unlearning.上公开获取



## **20. Identifying Human Strategies for Generating Word-Level Adversarial Examples**

确定生成词级对抗性实例的人类策略 cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11598v1) [paper-pdf](http://arxiv.org/pdf/2210.11598v1)

**Authors**: Maximilian Mozes, Bennett Kleinberg, Lewis D. Griffin

**Abstract**: Adversarial examples in NLP are receiving increasing research attention. One line of investigation is the generation of word-level adversarial examples against fine-tuned Transformer models that preserve naturalness and grammaticality. Previous work found that human- and machine-generated adversarial examples are comparable in their naturalness and grammatical correctness. Most notably, humans were able to generate adversarial examples much more effortlessly than automated attacks. In this paper, we provide a detailed analysis of exactly how humans create these adversarial examples. By exploring the behavioural patterns of human workers during the generation process, we identify statistically significant tendencies based on which words humans prefer to select for adversarial replacement (e.g., word frequencies, word saliencies, sentiment) as well as where and when words are replaced in an input sequence. With our findings, we seek to inspire efforts that harness human strategies for more robust NLP models.

摘要: 自然语言处理中的对抗性例子正受到越来越多的研究关注。一条研究路线是生成词级的对抗性例子，反对保持自然性和语法的微调变形金刚模型。以前的工作发现，人类和机器生成的对抗性例子在自然性和语法正确性方面具有可比性。最值得注意的是，人类能够比自动攻击更轻松地生成对抗性例子。在这篇文章中，我们对人类如何创造这些对抗性的例子提供了详细的分析。通过探索人类工作者在生成过程中的行为模式，我们基于人类更喜欢选择哪些单词作为对抗性替换(例如，单词频率、单词显著程度、情绪)以及输入序列中单词被替换的位置和时间来识别统计上显著的倾向。通过我们的发现，我们寻求激励人们努力利用人类战略来建立更强大的NLP模型。



## **21. Similarity of Neural Architectures Based on Input Gradient Transferability**

基于输入梯度传递的神经结构相似性研究 cs.LG

21pages, 10 figures, 1.5MB

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11407v1) [paper-pdf](http://arxiv.org/pdf/2210.11407v1)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In this paper, we aim to design a quantitative similarity function between two neural architectures. Specifically, we define a model similarity using input gradient transferability. We generate adversarial samples of two networks and measure the average accuracy of the networks on adversarial samples of each other. If two networks are highly correlated, then the attack transferability will be high, resulting in high similarity. Using the similarity score, we investigate two topics: (1) Which network component contributes to the model diversity? (2) How does model diversity affect practical scenarios? We answer the first question by providing feature importance analysis and clustering analysis. The second question is validated by two different scenarios: model ensemble and knowledge distillation. Our findings show that model diversity takes a key role when interacting with different neural architectures. For example, we found that more diversity leads to better ensemble performance. We also observe that the relationship between teacher and student networks and distillation performance depends on the choice of the base architecture of the teacher and student networks. We expect our analysis tool helps a high-level understanding of differences between various neural architectures as well as practical guidance when using multiple architectures.

摘要: 在本文中，我们的目标是设计两个神经结构之间的定量相似性函数。具体地说，我们定义了一种使用输入梯度可转移性的模型相似性。我们生成两个网络的对抗性样本，并在对方的对抗性样本上测量网络的平均准确率。如果两个网络高度相关，那么攻击的可传递性就高，从而导致高相似度。使用相似性得分，我们研究了两个主题：(1)哪个网络组件对模型多样性有贡献？(2)模型多样性如何影响实际场景？我们通过提供特征重要性分析和聚类分析来回答第一个问题。第二个问题通过两个不同的场景进行验证：模型集成和知识提炼。我们的发现表明，在与不同的神经结构相互作用时，模型多样性起着关键作用。例如，我们发现，更多的多样性会带来更好的合奏表现。我们还观察到，教师和学生网络与蒸馏性能之间的关系取决于教师和学生网络基础架构的选择。我们希望我们的分析工具能够帮助我们更高层次地了解不同神经架构之间的差异，并在使用多种架构时提供实用指导。



## **22. Surprises in adversarially-trained linear regression**

逆训练线性回归中的惊喜 stat.ML

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2205.12695v2) [paper-pdf](http://arxiv.org/pdf/2205.12695v2)

**Authors**: Antônio H. Ribeiro, Dave Zachariah, Thomas B. Schön

**Abstract**: State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against such examples. It is formulated as a min-max problem, searching for the best solution when the training data was corrupted by the worst-case attacks. For linear regression problems, adversarial training can be formulated as a convex problem. We use this reformulation to make two technical contributions: First, we formulate the training problem as an instance of robust regression to reveal its connection to parameter-shrinking methods, specifically that $\ell_\infty$-adversarial training produces sparse solutions. Secondly, we study adversarial training in the overparameterized regime, i.e. when there are more parameters than data. We prove that adversarial training with small disturbances gives the solution with the minimum-norm that interpolates the training data. Ridge regression and lasso approximate such interpolating solutions as their regularization parameter vanishes. By contrast, for adversarial training, the transition into the interpolation regime is abrupt and for non-zero values of disturbance. This result is proved and illustrated with numerical examples.

摘要: 最先进的机器学习模型很容易受到相反构造的非常小的输入扰动的影响。对抗性训练是抵御这类例子的一种有效方法。它被描述为一个最小-最大问题，当训练数据被最坏情况的攻击破坏时，搜索最优解。对于线性回归问题，对抗性训练可以表示为一个凸问题。首先，我们将训练问题描述为稳健回归的一个实例，以揭示它与参数收缩方法的联系，具体地说，对手训练产生稀疏解。其次，我们研究了参数多于数据的过度参数条件下的对抗性训练。我们证明了小干扰下的对抗性训练给出了用最小范数对训练数据进行内插的解。岭回归和套索逼近这种插值解，因为它们的正则化参数为零。相比之下，对于对抗性训练，向插补机制的转变是突然的，并且对于非零值的扰动。这一结果得到了证明，并用数值算例加以说明。



## **23. Attacking Motion Estimation with Adversarial Snow**

对抗性降雪下的攻击运动估计 cs.CV

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11242v1) [paper-pdf](http://arxiv.org/pdf/2210.11242v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks for motion estimation (optical flow) optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, we exploit a real-world weather phenomenon for a novel attack with adversarially optimized snow. At the core of our attack is a differentiable renderer that consistently integrates photorealistic snowflakes with realistic motion into the 3D scene. Through optimization we obtain adversarial snow that significantly impacts the optical flow while being indistinguishable from ordinary snow. Surprisingly, the impact of our novel attack is largest on methods that previously showed a high robustness to small L_p perturbations.

摘要: 当前针对运动估计(光流)的敌意攻击优化了每像素的微小扰动，这在现实世界中不太可能出现。相反，我们利用真实世界的天气现象，用相反的优化降雪进行了一次新颖的攻击。我们攻击的核心是一个可区分的渲染器，它一致地将照片级逼真的雪花和逼真的运动整合到3D场景中。通过优化，我们得到了对抗性雪，它对光流有明显的影响，但与普通雪没有什么区别。令人惊讶的是，我们的新攻击对以前对小Lp扰动表现出高度稳健性的方法的影响最大。



## **24. UKP-SQuARE v2: Explainability and Adversarial Attacks for Trustworthy QA**

UKP-Square v2：可解析性和针对可信QA的对抗性攻击 cs.CL

Accepted at AACL 2022 as Demo Paper

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2208.09316v3) [paper-pdf](http://arxiv.org/pdf/2208.09316v3)

**Authors**: Rachneet Sachdeva, Haritz Puerto, Tim Baumgärtner, Sewin Tariverdian, Hao Zhang, Kexin Wang, Hossain Shaikh Saadi, Leonardo F. R. Ribeiro, Iryna Gurevych

**Abstract**: Question Answering (QA) systems are increasingly deployed in applications where they support real-world decisions. However, state-of-the-art models rely on deep neural networks, which are difficult to interpret by humans. Inherently interpretable models or post hoc explainability methods can help users to comprehend how a model arrives at its prediction and, if successful, increase their trust in the system. Furthermore, researchers can leverage these insights to develop new methods that are more accurate and less biased. In this paper, we introduce SQuARE v2, the new version of SQuARE, to provide an explainability infrastructure for comparing models based on methods such as saliency maps and graph-based explanations. While saliency maps are useful to inspect the importance of each input token for the model's prediction, graph-based explanations from external Knowledge Graphs enable the users to verify the reasoning behind the model prediction. In addition, we provide multiple adversarial attacks to compare the robustness of QA models. With these explainability methods and adversarial attacks, we aim to ease the research on trustworthy QA models. SQuARE is available on https://square.ukp-lab.de.

摘要: 问答(QA)系统越来越多地部署在支持现实世界决策的应用程序中。然而，最先进的模型依赖于深度神经网络，这很难被人类解释。本质上可解释的模型或事后可解释的方法可以帮助用户理解模型如何达到其预测，如果成功，则增加他们对系统的信任。此外，研究人员可以利用这些洞察力来开发更准确、更少偏见的新方法。在本文中，我们引入了Square的新版本Square v2，以提供基于显著图和基于图的解释等方法的模型比较的可解释性基础设施。虽然显著图有助于检查每个输入标记对于模型预测的重要性，但来自外部知识图的基于图形的解释使用户能够验证模型预测背后的推理。此外，我们还提供了多个对抗性攻击来比较QA模型的健壮性。通过这些可解释性方法和对抗性攻击，我们的目标是简化可信QA模型的研究。Square在https://square.ukp-lab.de.上可用



## **25. Analyzing the Robustness of Decentralized Horizontal and Vertical Federated Learning Architectures in a Non-IID Scenario**

非IID场景下分布式水平和垂直联合学习体系结构的健壮性分析 cs.LG

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.11061v1) [paper-pdf](http://arxiv.org/pdf/2210.11061v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Enrique Tomás Martínez Beltrán, Daniel Demeter, Gérôme Bovet, Gregorio Martínez Pérez, Burkhard Stiller

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine and deep learning models while protecting data privacy. However, the FL paradigm still presents drawbacks affecting its trustworthiness since malicious participants could launch adversarial attacks against the training process. Related work has studied the robustness of horizontal FL scenarios under different attacks. However, there is a lack of work evaluating the robustness of decentralized vertical FL and comparing it with horizontal FL architectures affected by adversarial attacks. Thus, this work proposes three decentralized FL architectures, one for horizontal and two for vertical scenarios, namely HoriChain, VertiChain, and VertiComb. These architectures present different neural networks and training protocols suitable for horizontal and vertical scenarios. Then, a decentralized, privacy-preserving, and federated use case with non-IID data to classify handwritten digits is deployed to evaluate the performance of the three architectures. Finally, a set of experiments computes and compares the robustness of the proposed architectures when they are affected by different data poisoning based on image watermarks and gradient poisoning adversarial attacks. The experiments show that even though particular configurations of both attacks can destroy the classification performance of the architectures, HoriChain is the most robust one.

摘要: 联合学习(FL)允许参与者协作训练机器和深度学习模型，同时保护数据隐私。然而，FL范式仍然存在影响其可信度的缺陷，因为恶意参与者可能会对培训过程发动对抗性攻击。相关工作研究了水平FL场景在不同攻击下的稳健性。然而，缺乏对分布式垂直FL的健壮性进行评估，并将其与水平FL体系结构进行比较的工作。因此，本文提出了三种去中心化FL架构，一种用于水平场景，两种用于垂直场景，即HoriChain、VertiChain和VertiComb。这些体系结构提供了适用于水平和垂直场景的不同神经网络和训练协议。然后，部署了一个分散的、保护隐私的联合用例，使用非IID数据对手写数字进行分类，以评估三种体系结构的性能。最后，通过一组实验计算并比较了基于图像水印的数据中毒和基于梯度中毒的敌意攻击对所提体系结构的健壮性的影响。实验表明，尽管两种攻击的特定配置都会破坏体系结构的分类性能，但HoriChain是最健壮的一种。



## **26. Defending Person Detection Against Adversarial Patch Attack by using Universal Defensive Frame**

利用通用防御框架防御敌方补丁攻击的人检测 cs.CV

Accepted at IEEE Transactions on Image Processing (TIP), 2022

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2204.13004v2) [paper-pdf](http://arxiv.org/pdf/2204.13004v2)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstract**: Person detection has attracted great attention in the computer vision area and is an imperative element in human-centric computer vision. Although the predictive performances of person detection networks have been improved dramatically, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the person detection network in safety-critical applications such as autonomous driving and security systems. Despite the necessity of countering adversarial patch attacks, very few efforts have been dedicated to defending person detection against adversarial patch attack. In this paper, we propose a novel defense strategy that defends against an adversarial patch attack by optimizing a defensive frame for person detection. The defensive frame alleviates the effect of the adversarial patch while maintaining person detection performance with clean person. The proposed defensive frame in the person detection is generated with a competitive learning algorithm which makes an iterative competition between detection threatening module and detection shielding module in person detection. Comprehensive experimental results demonstrate that the proposed method effectively defends person detection against adversarial patch attacks.

摘要: 人体检测在计算机视觉领域引起了极大的关注，是以人为中心的计算机视觉的重要组成部分。虽然个人检测网络的预测性能有了很大的提高，但它们很容易受到对手补丁的攻击。在自动驾驶和安全系统等安全关键应用中，更改受限区域的像素很容易欺骗人员检测网络。尽管有必要对抗对抗性补丁攻击，但很少有人致力于防御对抗性补丁攻击的人检测。在本文中，我们提出了一种新的防御策略，通过优化个人检测的防御框架来防御对抗性补丁攻击。该防御框架在保持人与干净的人的检测性能的同时，减轻了对手补丁的影响。利用竞争学习算法生成人体检测中的防御帧，使得人体检测中的检测威胁模块和检测屏蔽模块之间进行迭代竞争。综合实验结果表明，该方法能够有效地防御敌意补丁攻击。



## **27. Chaos Theory and Adversarial Robustness**

混沌理论与对抗稳健性 cs.LG

13 pages, 6 figures

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2210.13235v1) [paper-pdf](http://arxiv.org/pdf/2210.13235v1)

**Authors**: Jonathan S. Kent

**Abstract**: Neural Networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which Neural Networks are susceptible to or robust against adversarial attacks. Our results show that susceptibility to attack grows significantly with the depth of the model, which has significant safety implications for the design of Neural Networks for production environments. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly, as well as show a clear relationship between our new susceptibility metric and post-attack accuracy.

摘要: 神经网络容易受到对抗性攻击，在部署到关键或对抗性应用程序之前，应该面临严格的审查。本文使用混沌理论的思想来解释、分析和量化神经网络对敌意攻击的敏感程度或健壮性。我们的结果表明，随着模型深度的增加，对攻击的敏感性显著增加，这对于生产环境下的神经网络的设计具有重要的安全意义。我们还演示了如何快速、轻松地近似极大模型的认证稳健性半径，到目前为止，该半径在计算上无法直接计算，并显示了我们的新敏感度度量和攻击后精度之间的明确关系。



## **28. Towards Adversarial Attack on Vision-Language Pre-training Models**

视觉语言预训练模型的对抗性攻击 cs.LG

Accepted by ACM MM2022. Code is available in GitHub

**SubmitDate**: 2022-10-20    [abs](http://arxiv.org/abs/2206.09391v2) [paper-pdf](http://arxiv.org/pdf/2206.09391v2)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstract**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios. Code is available at https://github.com/adversarial-for-goodness/Co-Attack.

摘要: 虽然视觉-语言预训练模型(VLP)在各种视觉-语言(V+L)任务上有了革命性的改进，但关于其对抗健壮性的研究仍然很少。研究了对流行的VLP模型和V+L任务的对抗性攻击。首先，我们分析了不同环境下对抗性攻击的性能。通过考察不同扰动对象和攻击目标的影响，我们总结了一些关键的观察结果，作为设计强多通道对抗性攻击和构建稳健VLP模型的指导。其次，我们提出了一种新的针对VLP模型的多模式攻击方法，称为协作式多模式对抗攻击(Co-Attack)，它共同对图像通道和文本通道进行攻击。实验结果表明，该方法在不同的V+L下游任务和VLP模型下均能获得较好的攻击性能。分析观察和新颖的攻击方法有望对VLP模型的对抗健壮性提供新的理解，从而有助于在更真实的场景中安全可靠地部署VLP模型。代码可在https://github.com/adversarial-for-goodness/Co-Attack.上找到



## **29. Rewriting Meaningful Sentences via Conditional BERT Sampling and an application on fooling text classifiers**

基于条件BERT抽样的有意义句子重写及其在愚弄文本分类器上的应用 cs.CL

Please see an updated version of this paper at arXiv:2104.08453

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2010.11869v2) [paper-pdf](http://arxiv.org/pdf/2010.11869v2)

**Authors**: Lei Xu, Ivan Ramirez, Kalyan Veeramachaneni

**Abstract**: Most adversarial attack methods that are designed to deceive a text classifier change the text classifier's prediction by modifying a few words or characters. Few try to attack classifiers by rewriting a whole sentence, due to the difficulties inherent in sentence-level rephrasing as well as the problem of setting the criteria for legitimate rewriting.   In this paper, we explore the problem of creating adversarial examples with sentence-level rewriting. We design a new sampling method, named ParaphraseSampler, to efficiently rewrite the original sentence in multiple ways. Then we propose a new criteria for modification, called a sentence-level threaten model. This criteria allows for both word- and sentence-level changes, and can be adjusted independently in two dimensions: semantic similarity and grammatical quality. Experimental results show that many of these rewritten sentences are misclassified by the classifier. On all 6 datasets, our ParaphraseSampler achieves a better attack success rate than our baseline.

摘要: 大多数旨在欺骗文本分类器的对抗性攻击方法都是通过修改几个单词或字符来改变文本分类器的预测。很少有人试图通过重写整个句子来攻击量词，这是因为句子级重写固有的困难，以及为合法重写设定标准的问题。在本文中，我们探讨了使用句子级重写来创建对抗性实例的问题。我们设计了一种新的抽样方法，称为ParaphraseSsamer，以多种方式高效地重写原始句子。然后，我们提出了一种新的修改标准，称为语句级威胁模型。这一标准允许单词和句子级别的变化，并且可以在两个维度上独立调整：语义相似度和语法质量。实验结果表明，许多改写后的句子被分类器误分类。在所有6个数据集上，我们的ParaphraseSsamer实现了比我们的基线更高的攻击成功率。



## **30. R&R: Metric-guided Adversarial Sentence Generation**

R&R：度量制导的对抗性句子生成 cs.CL

Accepted to Finding of AACL-IJCNLP2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2104.08453v3) [paper-pdf](http://arxiv.org/pdf/2104.08453v3)

**Authors**: Lei Xu, Alfredo Cuesta-Infante, Laure Berti-Equille, Kalyan Veeramachaneni

**Abstract**: Adversarial examples are helpful for analyzing and improving the robustness of text classifiers. Generating high-quality adversarial examples is a challenging task as it requires generating fluent adversarial sentences that are semantically similar to the original sentences and preserve the original labels, while causing the classifier to misclassify them. Existing methods prioritize misclassification by maximizing each perturbation's effectiveness at misleading a text classifier; thus, the generated adversarial examples fall short in terms of fluency and similarity. In this paper, we propose a rewrite and rollback (R&R) framework for adversarial attack. It improves the quality of adversarial examples by optimizing a critique score which combines the fluency, similarity, and misclassification metrics. R&R generates high-quality adversarial examples by allowing exploration of perturbations that do not have immediate impact on the misclassification metric but can improve fluency and similarity metrics. We evaluate our method on 5 representative datasets and 3 classifier architectures. Our method outperforms current state-of-the-art in attack success rate by +16.2%, +12.8%, and +14.0% on the classifiers respectively. Code is available at https://github.com/DAI-Lab/fibber

摘要: 对抗性例子有助于分析和提高文本分类器的健壮性。生成高质量的对抗性例子是一项具有挑战性的任务，因为它需要生成流畅的对抗性句子，这些句子在语义上与原始句子相似，并保留原始标签，同时导致分类器对它们进行错误分类。现有的方法通过最大化每个扰动在误导文本分类器方面的有效性来区分错误分类的优先级；因此，生成的对抗性示例在流畅性和相似性方面存在不足。提出了一种用于对抗攻击的重写和回滚(R&R)框架。它通过优化结合了流畅度、相似度和误分类度量的批评性分数来提高对抗性例子的质量。R&R通过允许探索对错误分类度量没有直接影响但可以提高流畅性和相似性度量的扰动来生成高质量的对抗性示例。我们在5个有代表性的数据集和3个分类器体系结构上对我们的方法进行了评估。我们的方法在攻击成功率上分别比当前最先进的分类器高出16.2%、+12.8%和+14.0%。代码可在https://github.com/DAI-Lab/fibber上找到



## **31. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

基于联邦生成对抗网络的医学图像合成后门攻击与防御 cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10886v1) [paper-pdf](http://arxiv.org/pdf/2210.10886v1)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.

摘要: 基于深度学习的图像合成技术已被应用于医疗保健研究中，以生成医学图像以支持开放研究和扩充医学数据集。生成性对抗神经网络的训练通常需要大量的训练数据。联合学习(FL)提供了一种在本地保留原始数据的同时使用分布式数据训练中央模型的方法。然而，鉴于FL服务器无法访问原始数据，它很容易受到后门攻击，这是通过毒化训练数据而产生的对抗性攻击。大多数后门攻击策略侧重于分类模型和集中域。现有的后门攻击是否会影响GAN的训练，如果是的话，在FL环境下如何防御攻击，仍然是一个悬而未决的问题。在这项工作中，我们研究了联邦GAN(FedGAN)中被忽视的后门攻击问题。这种攻击的成功随后被确定为一些本地鉴别器过度拟合有毒数据并破坏本地GaN平衡的结果，这随后在平均发电机参数时进一步污染其他客户端，并产生高发电机损耗。因此，我们提出了FedDetect，这是一种在FL环境下有效防御后门攻击的方法，它允许服务器根据客户端的损失来检测客户端的敌对行为，并阻止恶意客户端。我们在两个不同模式的医学数据集上的广泛实验表明，对FedGan的后门攻击可以导致合成图像的低保真度。在使用该防御策略检测和抑制检测到的恶意客户端后，我们证明了FedGans能够合成高质量的医学数据集(带标签)用于数据增强，从而提高分类模型的性能。



## **32. On the Perils of Cascading Robust Classifiers**

关于级联稳健分类器的危险 cs.LG

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2206.00278v2) [paper-pdf](http://arxiv.org/pdf/2206.00278v2)

**Authors**: Ravi Mangal, Zifan Wang, Chi Zhang, Klas Leino, Corina Pasareanu, Matt Fredrikson

**Abstract**: Ensembling certifiably robust neural networks is a promising approach for improving the \emph{certified robust accuracy} of neural models. Black-box ensembles that assume only query-access to the constituent models (and their robustness certifiers) during prediction are particularly attractive due to their modular structure. Cascading ensembles are a popular instance of black-box ensembles that appear to improve certified robust accuracies in practice. However, we show that the robustness certifier used by a cascading ensemble is unsound. That is, when a cascading ensemble is certified as locally robust at an input $x$ (with respect to $\epsilon$), there can be inputs $x'$ in the $\epsilon$-ball centered at $x$, such that the cascade's prediction at $x'$ is different from $x$ and thus the ensemble is not locally robust. Our theoretical findings are accompanied by empirical results that further demonstrate this unsoundness. We present \emph{cascade attack} (CasA), an adversarial attack against cascading ensembles, and show that: (1) there exists an adversarial input for up to 88\% of the samples where the ensemble claims to be certifiably robust and accurate; and (2) the accuracy of a cascading ensemble under our attack is as low as 11\% when it claims to be certifiably robust and accurate on 97\% of the test set. Our work reveals a critical pitfall of cascading certifiably robust models by showing that the seemingly beneficial strategy of cascading can actually hurt the robustness of the resulting ensemble. Our code is available at \url{https://github.com/TristaChi/ensembleKW}.

摘要: 集成可证明稳健神经网络是提高神经模型证明稳健精度的一种很有前途的方法。由于其模块化结构，在预测期间假设只对组成模型(及其稳健性证明器)进行查询访问的黑盒集成特别有吸引力。级联组合是黑盒组合的一个受欢迎的例子，它似乎在实践中提高了经过认证的稳健精度。然而，我们证明了级联系综所使用的稳健性证明是不可靠的。也就是说，当级联系综在输入$x$(相对于$\epsilon$)被证明为局部稳健时，在以$x$为中心的$\epsilon$球中可以有输入$x‘$，使得在$x’$处的级联预测不同于$x$，因此该系综不是局部稳健的。我们的理论发现伴随着进一步证明这一不合理的经验结果。我们提出了一种对级联集的对抗性攻击(CASA)，并证明了：(1)对于高达88%的样本，存在对抗性输入，其中集成声称是可证明的健壮和准确的；(2)在我们的攻击下，当级联集成声称在97%的测试集上是可证明的健壮和准确时，其准确率低至11%。我们的工作揭示了级联可证明健壮性模型的一个关键陷阱，即看似有益的级联策略实际上可能会损害由此产生的整体的健壮性。我们的代码可以在\url{https://github.com/TristaChi/ensembleKW}.上找到



## **33. Why Should Adversarial Perturbations be Imperceptible? Rethink the Research Paradigm in Adversarial NLP**

为什么对抗性的扰动应该是不可察觉的？对抗性自然语言处理研究范式的再思考 cs.CL

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10683v1) [paper-pdf](http://arxiv.org/pdf/2210.10683v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Fanchao Qi, Longtao Huang, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual adversarial samples play important roles in multiple subfields of NLP research, including security, evaluation, explainability, and data augmentation. However, most work mixes all these roles, obscuring the problem definitions and research goals of the security role that aims to reveal the practical concerns of NLP models. In this paper, we rethink the research paradigm of textual adversarial samples in security scenarios. We discuss the deficiencies in previous work and propose our suggestions that the research on the Security-oriented adversarial NLP (SoadNLP) should: (1) evaluate their methods on security tasks to demonstrate the real-world concerns; (2) consider real-world attackers' goals, instead of developing impractical methods. To this end, we first collect, process, and release a security datasets collection Advbench. Then, we reformalize the task and adjust the emphasis on different goals in SoadNLP. Next, we propose a simple method based on heuristic rules that can easily fulfill the actual adversarial goals to simulate real-world attack methods. We conduct experiments on both the attack and the defense sides on Advbench. Experimental results show that our method has higher practical value, indicating that the research paradigm in SoadNLP may start from our new benchmark. All the code and data of Advbench can be obtained at \url{https://github.com/thunlp/Advbench}.

摘要: 文本敌意样本在自然语言处理研究的多个子领域中扮演着重要的角色，包括安全性、评估、可解释性和数据增强。然而，大多数工作混合了所有这些角色，模糊了安全角色的问题定义和研究目标，旨在揭示NLP模型的实际问题。在本文中，我们重新思考了安全场景中文本对抗样本的研究范式。我们讨论了前人工作中的不足，并提出了我们的建议，面向安全的对抗性NLP(SoadNLP)的研究应该：(1)评估他们在安全任务上的方法以展示现实世界的关注点；(2)考虑真实世界攻击者的目标，而不是开发不切实际的方法。为此，我们首先收集、处理和发布安全数据集集合Advbench。然后，我们对SoadNLP的任务进行了改革，并针对不同的目标调整了侧重点。接下来，我们提出了一种简单的基于启发式规则的方法来模拟真实世界的攻击方法，该方法可以很容易地实现实际的对抗目标。我们在Advbench上进行了攻防双方的实验。实验结果表明，我们的方法具有较高的实用价值，表明SoadNLP的研究范式可以从我们的新基准开始。有关安进的所有代码和数据，请访问\url{https://github.com/thunlp/Advbench}.



## **34. Textual Backdoor Attacks Can Be More Harmful via Two Simple Tricks**

文本后门攻击可能通过两个简单的技巧造成更大的危害 cs.CR

Accepted to EMNLP 2022, main conference

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2110.08247v2) [paper-pdf](http://arxiv.org/pdf/2110.08247v2)

**Authors**: Yangyi Chen, Fanchao Qi, Hongcheng Gao, Zhiyuan Liu, Maosong Sun

**Abstract**: Backdoor attacks are a kind of emergent security threat in deep learning. After being injected with a backdoor, a deep neural model will behave normally on standard inputs but give adversary-specified predictions once the input contains specific backdoor triggers. In this paper, we find two simple tricks that can make existing textual backdoor attacks much more harmful. The first trick is to add an extra training task to distinguish poisoned and clean data during the training of the victim model, and the second one is to use all the clean training data rather than remove the original clean data corresponding to the poisoned data. These two tricks are universally applicable to different attack models. We conduct experiments in three tough situations including clean data fine-tuning, low-poisoning-rate, and label-consistent attacks. Experimental results show that the two tricks can significantly improve attack performance. This paper exhibits the great potential harmfulness of backdoor attacks. All the code and data can be obtained at \url{https://github.com/thunlp/StyleAttack}.

摘要: 后门攻击是深度学习中一种突发的安全威胁。在被注入后门后，深层神经模型将在标准输入上正常运行，但一旦输入包含特定的后门触发器，就会给出对手指定的预测。在本文中，我们发现了两个简单的技巧，可以使现有的文本后门攻击更具危害性。第一个技巧是在受害者模型的训练过程中增加一个额外的训练任务来区分有毒和干净的数据，第二个技巧是使用所有的干净训练数据而不是删除有毒数据对应的原始干净数据。这两个技巧普遍适用于不同的攻击模式。我们在三种艰难的情况下进行了实验，包括干净的数据微调、低中毒率和标签一致攻击。实验结果表明，这两种策略都能显著提高攻击性能。本文展示了后门攻击的巨大潜在危害性。所有代码和数据均可在\url{https://github.com/thunlp/StyleAttack}.



## **35. Few-shot Transferable Robust Representation Learning via Bilevel Attacks**

基于两层攻击的少射转移稳健表示学习 cs.LG

*Equal contribution. Author ordering determined by coin flip

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10485v1) [paper-pdf](http://arxiv.org/pdf/2210.10485v1)

**Authors**: Minseon Kim, Hyeonjeong Ha, Sung Ju Hwang

**Abstract**: Existing adversarial learning methods for enhancing the robustness of deep neural networks assume the availability of a large amount of data from which we can generate adversarial examples. However, in an adversarial meta-learning setting, the model needs to train with only a few adversarial examples to learn a robust model for unseen tasks, which is a very difficult goal to achieve. Further, learning transferable robust representations for unseen domains is a difficult problem even with a large amount of data. To tackle such a challenge, we propose a novel adversarial self-supervised meta-learning framework with bilevel attacks which aims to learn robust representations that can generalize across tasks and domains. Specifically, in the inner loop, we update the parameters of the given encoder by taking inner gradient steps using two different sets of augmented samples, and generate adversarial examples for each view by maximizing the instance classification loss. Then, in the outer loop, we meta-learn the encoder parameter to maximize the agreement between the two adversarial examples, which enables it to learn robust representations. We experimentally validate the effectiveness of our approach on unseen domain adaptation tasks, on which it achieves impressive performance. Specifically, our method significantly outperforms the state-of-the-art meta-adversarial learning methods on few-shot learning tasks, as well as self-supervised learning baselines in standard learning settings with large-scale datasets.

摘要: 现有的用于增强深度神经网络鲁棒性的对抗性学习方法假设有大量数据可用，我们可以从这些数据中生成对抗性示例。然而，在对抗性的元学习环境下，该模型只需要用几个对抗性的例子来训练，以学习一个针对未知任务的健壮模型，这是一个很难实现的目标。此外，即使在有大量数据的情况下，学习不可见领域的可转移的稳健表示也是一个困难的问题。为了应对这样的挑战，我们提出了一种新颖的具有双层攻击的对抗性自监督元学习框架，旨在学习能够跨任务和领域泛化的健壮表示。具体地说，在内循环中，我们通过使用两组不同的增广样本来采取内梯度步骤来更新给定编码器的参数，并通过最大化实例分类损失来为每个视图生成对抗性示例。然后，在外部循环中，我们元学习编码器参数以最大化两个敌对示例之间的一致性，从而使其能够学习稳健的表示。我们在实验上验证了我们方法在看不见的领域适应任务上的有效性，在这些任务上它取得了令人印象深刻的性能。具体地说，在大规模数据集的标准学习环境中，我们的方法在少镜头学习任务上显著优于最先进的元对抗学习方法，并且在自我监督学习基线上也是如此。



## **36. Emerging Threats in Deep Learning-Based Autonomous Driving: A Comprehensive Survey**

基于深度学习的自动驾驶中新出现的威胁：全面调查 cs.CR

28 pages,10 figures

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.11237v1) [paper-pdf](http://arxiv.org/pdf/2210.11237v1)

**Authors**: Hui Cao, Wenlong Zou, Yinkun Wang, Ting Song, Mengjun Liu

**Abstract**: Since the 2004 DARPA Grand Challenge, the autonomous driving technology has witnessed nearly two decades of rapid development. Particularly, in recent years, with the application of new sensors and deep learning technologies extending to the autonomous field, the development of autonomous driving technology has continued to make breakthroughs. Thus, many carmakers and high-tech giants dedicated to research and system development of autonomous driving. However, as the foundation of autonomous driving, the deep learning technology faces many new security risks. The academic community has proposed deep learning countermeasures against the adversarial examples and AI backdoor, and has introduced them into the autonomous driving field for verification. Deep learning security matters to autonomous driving system security, and then matters to personal safety, which is an issue that deserves attention and research.This paper provides an summary of the concepts, developments and recent research in deep learning security technologies in autonomous driving. Firstly, we briefly introduce the deep learning framework and pipeline in the autonomous driving system, which mainly include the deep learning technologies and algorithms commonly used in this field. Moreover, we focus on the potential security threats of the deep learning based autonomous driving system in each functional layer in turn. We reviews the development of deep learning attack technologies to autonomous driving, investigates the State-of-the-Art algorithms, and reveals the potential risks. At last, we provides an outlook on deep learning security in the autonomous driving field and proposes recommendations for building a safe and trustworthy autonomous driving system.

摘要: 自2004年DARPA大赛以来，自动驾驶技术经历了近二十年的快速发展。特别是近年来，随着新型传感器和深度学习技术的应用向自主领域延伸，自动驾驶技术的发展不断取得突破。因此，许多汽车制造商和高科技巨头都致力于自动驾驶的研究和系统开发。然而，深度学习技术作为自动驾驶的基础，也面临着许多新的安全隐患。学术界针对对抗性范例和AI后门提出了深度学习对策，并将其引入自动驾驶领域进行验证。深度学习安全关系到自动驾驶系统的安全，进而关系到人身安全，是一个值得关注和研究的问题。本文综述了自动驾驶中深度学习安全技术的概念、发展和研究现状。首先，简要介绍了自主驾驶系统中深度学习的框架和流水线，主要包括该领域常用的深度学习技术和算法。在此基础上，依次对基于深度学习的自主驾驶系统在各个功能层中存在的潜在安全威胁进行了分析。我们回顾了深度学习攻击技术对自动驾驶的发展，研究了最新的算法，并揭示了潜在的风险。最后，对自主驾驶领域的深度学习安全进行了展望，并对构建安全可信的自动驾驶系统提出了建议。



## **37. On the Adversarial Robustness of Mixture of Experts**

关于专家混合的对抗性稳健性 cs.LG

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-19    [abs](http://arxiv.org/abs/2210.10253v1) [paper-pdf](http://arxiv.org/pdf/2210.10253v1)

**Authors**: Joan Puigcerver, Rodolphe Jenatton, Carlos Riquelme, Pranjal Awasthi, Srinadh Bhojanapalli

**Abstract**: Adversarial robustness is a key desirable property of neural networks. It has been empirically shown to be affected by their sizes, with larger networks being typically more robust. Recently, Bubeck and Sellke proved a lower bound on the Lipschitz constant of functions that fit the training data in terms of their number of parameters. This raises an interesting open question, do -- and can -- functions with more parameters, but not necessarily more computational cost, have better robustness? We study this question for sparse Mixture of Expert models (MoEs), that make it possible to scale up the model size for a roughly constant computational cost. We theoretically show that under certain conditions on the routing and the structure of the data, MoEs can have significantly smaller Lipschitz constants than their dense counterparts. The robustness of MoEs can suffer when the highest weighted experts for an input implement sufficiently different functions. We next empirically evaluate the robustness of MoEs on ImageNet using adversarial attacks and show they are indeed more robust than dense models with the same computational cost. We make key observations showing the robustness of MoEs to the choice of experts, highlighting the redundancy of experts in models trained in practice.

摘要: 对抗健壮性是神经网络的一个关键的理想性质。经验表明，它受到网络规模的影响，更大的网络通常更健壮。最近，Bubeck和Sellke证明了根据参数个数拟合训练数据的函数的Lipschitz常数的一个下界。这提出了一个有趣的开放问题，参数更多但计算成本不一定更高的函数是否--以及能否--具有更好的健壮性？我们研究了稀疏混合专家模型(MOE)的这个问题，它使得在计算成本大致不变的情况下扩大模型的规模成为可能。我们从理论上证明，在一定的布线和数据结构条件下，MOE的Lipschitz常数可以比稠密的MoE小得多。当输入的权重最高的专家实现完全不同的功能时，MOE的稳健性可能会受到影响。接下来，我们在ImageNet上使用对抗性攻击对MOE的健壮性进行了经验性评估，并表明在相同的计算代价下，它们确实比密集模型更健壮。我们进行了关键的观察，显示了MOE对专家选择的稳健性，强调了在实践中训练的模型中专家的冗余。



## **38. Parsimonious Black-Box Adversarial Attacks via Efficient Combinatorial Optimization**

基于高效组合优化的节约型黑箱对抗攻击 cs.LG

Accepted and to appear at ICML 2019

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/1905.06635v2) [paper-pdf](http://arxiv.org/pdf/1905.06635v2)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstract**: Solving for adversarial examples with projected gradient descent has been demonstrated to be highly effective in fooling the neural network based classifiers. However, in the black-box setting, the attacker is limited only to the query access to the network and solving for a successful adversarial example becomes much more difficult. To this end, recent methods aim at estimating the true gradient signal based on the input queries but at the cost of excessive queries. We propose an efficient discrete surrogate to the optimization problem which does not require estimating the gradient and consequently becomes free of the first order update hyperparameters to tune. Our experiments on Cifar-10 and ImageNet show the state of the art black-box attack performance with significant reduction in the required queries compared to a number of recently proposed methods. The source code is available at https://github.com/snu-mllab/parsimonious-blackbox-attack.

摘要: 用投影梯度下降的对抗性例子的求解已经被证明在愚弄基于神经网络的分类器方面是非常有效的。然而，在黑盒环境下，攻击者仅限于对网络的查询访问，求解一个成功的对抗性例子变得困难得多。为此，最近的方法旨在基于输入查询来估计真实的梯度信号，但代价是过多的查询。对于优化问题，我们提出了一种有效的离散代理，它不需要估计梯度，因此不需要一阶更新超参数来调整。我们在CIFAR-10和ImageNet上的实验显示了最先进的黑盒攻击性能，与最近提出的一些方法相比，所需的查询显著减少。源代码可在https://github.com/snu-mllab/parsimonious-blackbox-attack.上找到



## **39. Scaling Adversarial Training to Large Perturbation Bounds**

将对抗性训练扩展到大扰动界 cs.LG

ECCV 2022

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09852v1) [paper-pdf](http://arxiv.org/pdf/2210.09852v1)

**Authors**: Sravanti Addepalli, Samyak Jain, Gaurang Sriramanan, R. Venkatesh Babu

**Abstract**: The vulnerability of Deep Neural Networks to Adversarial Attacks has fuelled research towards building robust models. While most Adversarial Training algorithms aim at defending attacks constrained within low magnitude Lp norm bounds, real-world adversaries are not limited by such constraints. In this work, we aim to achieve adversarial robustness within larger bounds, against perturbations that may be perceptible, but do not change human (or Oracle) prediction. The presence of images that flip Oracle predictions and those that do not makes this a challenging setting for adversarial robustness. We discuss the ideal goals of an adversarial defense algorithm beyond perceptual limits, and further highlight the shortcomings of naively extending existing training algorithms to higher perturbation bounds. In order to overcome these shortcomings, we propose a novel defense, Oracle-Aligned Adversarial Training (OA-AT), to align the predictions of the network with that of an Oracle during adversarial training. The proposed approach achieves state-of-the-art performance at large epsilon bounds (such as an L-inf bound of 16/255 on CIFAR-10) while outperforming existing defenses (AWP, TRADES, PGD-AT) at standard bounds (8/255) as well.

摘要: 深度神经网络对敌意攻击的脆弱性推动了建立健壮模型的研究。虽然大多数对抗性训练算法的目标是防御被限制在低幅度LP范数范围内的攻击，但现实世界中的对手并不受这种限制的限制。在这项工作中，我们的目标是在更大的范围内实现对抗鲁棒性，对抗可能可感知但不改变人类(或Oracle)预测的扰动。与甲骨文预测相反的图像和非预测图像的存在，使得这对对手的稳健性来说是一个具有挑战性的环境。我们讨论了超越感知极限的对抗性防御算法的理想目标，并进一步强调了将现有训练算法天真地扩展到更高扰动界的缺点。为了克服这些不足，我们提出了一种新的防御方法--Oracle-Align对抗性训练(OA-AT)，以在对抗性训练中使网络的预测与Oracle的预测保持一致。所提出的方法在很大的epsilon界限(例如，CIFAR-10上的L-inf界限为16/255)下实现了最先进的性能，同时在标准界限(8/255)下也超过了现有的防御措施(AWP、TRADS、PGD-AT)。



## **40. Provably Robust Detection of Out-of-distribution Data (almost) for free**

可证明稳健的(几乎)免费的非分布数据检测 cs.LG

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2106.04260v2) [paper-pdf](http://arxiv.org/pdf/2106.04260v2)

**Authors**: Alexander Meinke, Julian Bitterwolf, Matthias Hein

**Abstract**: The application of machine learning in safety-critical systems requires a reliable assessment of uncertainty. However, deep neural networks are known to produce highly overconfident predictions on out-of-distribution (OOD) data. Even if trained to be non-confident on OOD data, one can still adversarially manipulate OOD data so that the classifier again assigns high confidence to the manipulated samples. We show that two previously published defenses can be broken by better adapted attacks, highlighting the importance of robustness guarantees around OOD data. Since the existing method for this task is hard to train and significantly limits accuracy, we construct a classifier that can simultaneously achieve provably adversarially robust OOD detection and high clean accuracy. Moreover, by slightly modifying the classifier's architecture our method provably avoids the asymptotic overconfidence problem of standard neural networks. We provide code for all our experiments.

摘要: 机器学习在安全关键系统中的应用需要对不确定性进行可靠的评估。然而，众所周知，深度神经网络对分布外(OOD)数据产生高度过度自信的预测。即使被训练成对OOD数据不可信，人们仍然可以相反地操纵OOD数据，以便分类器再次为被操纵的样本赋予高置信度。我们证明了之前发表的两个防御措施可以被更好地适应的攻击打破，强调了围绕OOD数据的健壮性保证的重要性。针对现有的分类方法训练难度大、精度低的问题，我们构造了一个分类器，它可以同时实现相对稳健的面向对象检测和较高的清洁准确率。此外，通过略微修改分类器的结构，我们的方法被证明避免了标准神经网络的渐近过度自信问题。我们为我们所有的实验提供代码。



## **41. ROSE: Robust Selective Fine-tuning for Pre-trained Language Models**

ROSE：对预先训练的语言模型进行稳健的选择性微调 cs.CL

Accepted to EMNLP 2022. Code is available at  https://github.com/jiangllan/ROSE

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09658v1) [paper-pdf](http://arxiv.org/pdf/2210.09658v1)

**Authors**: Lan Jiang, Hao Zhou, Yankai Lin, Peng Li, Jie Zhou, Rui Jiang

**Abstract**: Even though the large-scale language models have achieved excellent performances, they suffer from various adversarial attacks. A large body of defense methods has been proposed. However, they are still limited due to redundant attack search spaces and the inability to defend against various types of attacks. In this work, we present a novel fine-tuning approach called \textbf{RO}bust \textbf{SE}letive fine-tuning (\textbf{ROSE}) to address this issue. ROSE conducts selective updates when adapting pre-trained models to downstream tasks, filtering out invaluable and unrobust updates of parameters. Specifically, we propose two strategies: the first-order and second-order ROSE for selecting target robust parameters. The experimental results show that ROSE achieves significant improvements in adversarial robustness on various downstream NLP tasks, and the ensemble method even surpasses both variants above. Furthermore, ROSE can be easily incorporated into existing fine-tuning methods to improve their adversarial robustness further. The empirical analysis confirms that ROSE eliminates unrobust spurious updates during fine-tuning, leading to solutions corresponding to flatter and wider optima than the conventional method. Code is available at \url{https://github.com/jiangllan/ROSE}.

摘要: 尽管大规模语言模型取得了优异的性能，但它们也受到了各种对抗性攻击。已经提出了大量的防御方法。然而，由于多余的攻击搜索空间和无法防御各种类型的攻击，它们仍然有限。在这项工作中，我们提出了一种新的微调方法-.ROSE在调整预先训练的模型以适应下游任务时进行选择性更新，过滤掉无价和不可靠的参数更新。具体地说，我们提出了两种选择目标稳健参数的策略：一阶ROSE和二阶ROSE。实验结果表明，ROSE在不同下游NLP任务上的对抗性健壮性有了显著的提高，集成方法甚至超过了上述两种方法。此外，ROSE可以很容易地结合到现有的微调方法中，以进一步提高它们的对抗健壮性。实证分析证实，ROSE消除了微调过程中不稳健的虚假更新，得到了与传统方法相比更平坦和更广泛的最优解。代码位于\url{https://github.com/jiangllan/ROSE}.



## **42. Analysis of Master Vein Attacks on Finger Vein Recognition Systems**

主静脉攻击对手指静脉识别系统的影响分析 cs.CV

Accepted to be Published in Proceedings of the IEEE/CVF Winter  Conference on Applications of Computer Vision (WACV) 2023

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.10667v1) [paper-pdf](http://arxiv.org/pdf/2210.10667v1)

**Authors**: Huy H. Nguyen, Trung-Nghia Le, Junichi Yamagishi, Isao Echizen

**Abstract**: Finger vein recognition (FVR) systems have been commercially used, especially in ATMs, for customer verification. Thus, it is essential to measure their robustness against various attack methods, especially when a hand-crafted FVR system is used without any countermeasure methods. In this paper, we are the first in the literature to introduce master vein attacks in which we craft a vein-looking image so that it can falsely match with as many identities as possible by the FVR systems. We present two methods for generating master veins for use in attacking these systems. The first uses an adaptation of the latent variable evolution algorithm with a proposed generative model (a multi-stage combination of beta-VAE and WGAN-GP models). The second uses an adversarial machine learning attack method to attack a strong surrogate CNN-based recognition system. The two methods can be easily combined to boost their attack ability. Experimental results demonstrated that the proposed methods alone and together achieved false acceptance rates up to 73.29% and 88.79%, respectively, against Miura's hand-crafted FVR system. We also point out that Miura's system is easily compromised by non-vein-looking samples generated by a WGAN-GP model with false acceptance rates up to 94.21%. The results raise the alarm about the robustness of such systems and suggest that master vein attacks should be considered an important security measure.

摘要: 手指静脉识别(FVR)系统已经在商业上使用，特别是在自动取款机中，用于客户验证。因此，测量它们对各种攻击方法的稳健性是至关重要的，特别是当使用手工制作的FVR系统而没有任何对抗方法时。在本文中，我们在文献中首次引入了主静脉攻击，在这种攻击中，我们制作了一张看起来像静脉的图像，以便它可以通过FVR系统与尽可能多的身份进行虚假匹配。我们提出了两种生成主脉以用于攻击这些系统的方法。第一种是将潜在变量进化算法与所提出的生成模型(Beta-VAE和WGAN-GP模型的多阶段组合)相结合。第二个攻击是使用对抗性机器学习攻击方法来攻击基于CNN的强代理识别系统。这两种方法可以很容易地结合起来，以提高他们的攻击能力。实验结果表明，与Miura手工制作的FVR系统相比，提出的方法单独和联合使用的错误接受率分别高达73.29%和88.79%。我们还指出，Miura的系统很容易被WGAN-GP模型生成的非静脉样本所危害，错误接受率高达94.21%。这一结果对这类系统的健壮性提出了警告，并建议应将主静脉攻击视为一项重要的安全措施。



## **43. Making Split Learning Resilient to Label Leakage by Potential Energy Loss**

利用潜在能量损失使分裂学习具有抗泄漏能力 cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2210.09617v1) [paper-pdf](http://arxiv.org/pdf/2210.09617v1)

**Authors**: Fei Zheng, Chaochao Chen, Binhui Yao, Xiaolin Zheng

**Abstract**: As a practical privacy-preserving learning method, split learning has drawn much attention in academia and industry. However, its security is constantly being questioned since the intermediate results are shared during training and inference. In this paper, we focus on the privacy leakage problem caused by the trained split model, i.e., the attacker can use a few labeled samples to fine-tune the bottom model, and gets quite good performance. To prevent such kind of privacy leakage, we propose the potential energy loss to make the output of the bottom model become a more `complicated' distribution, by pushing outputs of the same class towards the decision boundary. Therefore, the adversary suffers a large generalization error when fine-tuning the bottom model with only a few leaked labeled samples. Experiment results show that our method significantly lowers the attacker's fine-tuning accuracy, making the split model more resilient to label leakage.

摘要: 分裂学习作为一种实用的隐私保护学习方法，受到了学术界和工业界的广泛关注。然而，由于中间结果是在训练和推理过程中共享的，其安全性不断受到质疑。本文重点研究了训练好的分裂模型带来的隐私泄露问题，即攻击者可以利用少量的标签样本对底层模型进行微调，取得了较好的性能。为了防止这种隐私泄露，我们提出了潜在能量损失，通过将同一类的输出推向决策边界，使底层模型的输出成为更复杂的分布。因此，对手在仅用几个泄漏的标签样本对底层模型进行微调时，会遭受较大的泛化误差。实验结果表明，该方法显著降低了攻击者的微调精度，使分裂模型具有更强的抗标签泄漏能力。



## **44. Friendly Noise against Adversarial Noise: A Powerful Defense against Data Poisoning Attacks**

友善噪声对抗敌意噪声：数据中毒攻击的有力防御 cs.CR

**SubmitDate**: 2022-10-18    [abs](http://arxiv.org/abs/2208.10224v3) [paper-pdf](http://arxiv.org/pdf/2208.10224v3)

**Authors**: Tian Yu Liu, Yu Yang, Baharan Mirzasoleiman

**Abstract**: A powerful category of (invisible) data poisoning attacks modify a subset of training examples by small adversarial perturbations to change the prediction of certain test-time data. Existing defense mechanisms are not desirable to deploy in practice, as they often either drastically harm the generalization performance, or are attack-specific, and prohibitively slow to apply. Here, we propose a simple but highly effective approach that unlike existing methods breaks various types of invisible poisoning attacks with the slightest drop in the generalization performance. We make the key observation that attacks introduce local sharp regions of high training loss, which when minimized, results in learning the adversarial perturbations and makes the attack successful. To break poisoning attacks, our key idea is to alleviate the sharp loss regions introduced by poisons. To do so, our approach comprises two components: an optimized friendly noise that is generated to maximally perturb examples without degrading the performance, and a randomly varying noise component. The combination of both components builds a very light-weight but extremely effective defense against the most powerful triggerless targeted and hidden-trigger backdoor poisoning attacks, including Gradient Matching, Bulls-eye Polytope, and Sleeper Agent. We show that our friendly noise is transferable to other architectures, and adaptive attacks cannot break our defense due to its random noise component.

摘要: 一种强大的(不可见)数据中毒攻击通过小的对抗性扰动来修改训练样本的子集，以改变对某些测试时间数据的预测。现有的防御机制在实践中并不可取，因为它们通常要么严重损害泛化性能，要么是特定于攻击的，应用速度慢得令人望而却步。在这里，我们提出了一种简单而高效的方法，不同于现有的方法，它以最小的泛化性能下降来破解各种类型的隐形中毒攻击。我们的关键观察是，攻击引入了高训练损失的局部尖锐区域，当训练损失最小化时，导致学习对手的扰动，使攻击成功。要打破毒物攻击，我们的关键思想是减轻毒物引入的急剧损失区域。为此，我们的方法包括两个组件：一个优化的友好噪声，它被生成以在不降低性能的情况下最大限度地扰动示例，以及一个随机变化的噪声组件。这两个组件的组合构建了一个非常轻但极其有效的防御系统，以抵御最强大的无触发器定向和隐藏触发器后门中毒攻击，包括梯度匹配、公牛眼多面体和睡眠代理。我们证明了我们的友好噪声是可以转移到其他体系结构的，而自适应攻击由于其随机噪声成分而不能破坏我们的防御。



## **45. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

制造一些噪音：可靠而高效的单步对抗性训练 cs.LG

Published in NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2202.01181v3) [paper-pdf](http://arxiv.org/pdf/2202.01181v3)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstract**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named Catastrophic Overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. Experimentally they showed that simply adding a random perturbation prior to FGSM (RS-FGSM) could prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed a computationally expensive regularizer (GradAlign) to avoid it. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with \textit{not clipping} is highly effective in avoiding CO for large perturbation radii. We then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous state-of-the-art GradAlign, while achieving 3x speed-up. Code can be found in https://github.com/pdejorge/N-FGSM

摘要: 最近，Wong et al.研究表明，采用单步FGSM的对抗性训练会导致一种称为灾难性过匹配(CO)的特征故障模式，在这种模式下，模型突然变得容易受到多步攻击。在实验上，他们表明，在FGSM(RS-FGSM)之前简单地添加随机扰动可以防止CO。然而，Andriushchenko和Flammarion观察到，对于较大的扰动，RS-FGSM仍然会导致CO，并提出了一种计算代价很高的正则化方法(GradAlign)来避免这种情况。在这项工作中，我们有条不紊地重新审视噪声和剪辑在单步对抗性训练中的作用。与以前的直觉相反，我们发现在清洁的样品周围使用更强的噪声并结合纹理{不剪裁}在大扰动半径下避免CO是非常有效的。然后，我们提出了噪声-FGSM(N-FGSM)，它在提供单步对抗性训练的好处的同时，不会受到CO的影响。大量实验的实验分析表明，N-FGSM能够在性能上赶上或超过以往最先进的GradAlign，同时获得3倍的加速。代码可在https://github.com/pdejorge/N-FGSM中找到



## **46. Deepfake Text Detection: Limitations and Opportunities**

深度假冒文本检测：局限与机遇 cs.CR

Accepted to IEEE S&P 2023; First two authors contributed equally to  this work; 18 pages, 7 figures

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09421v1) [paper-pdf](http://arxiv.org/pdf/2210.09421v1)

**Authors**: Jiameng Pu, Zain Sarwar, Sifat Muhammad Abdullah, Abdullah Rehman, Yoonjin Kim, Parantapa Bhattacharya, Mobin Javed, Bimal Viswanath

**Abstract**: Recent advances in generative models for language have enabled the creation of convincing synthetic text or deepfake text. Prior work has demonstrated the potential for misuse of deepfake text to mislead content consumers. Therefore, deepfake text detection, the task of discriminating between human and machine-generated text, is becoming increasingly critical. Several defenses have been proposed for deepfake text detection. However, we lack a thorough understanding of their real-world applicability. In this paper, we collect deepfake text from 4 online services powered by Transformer-based tools to evaluate the generalization ability of the defenses on content in the wild. We develop several low-cost adversarial attacks, and investigate the robustness of existing defenses against an adaptive attacker. We find that many defenses show significant degradation in performance under our evaluation scenarios compared to their original claimed performance. Our evaluation shows that tapping into the semantic information in the text content is a promising approach for improving the robustness and generalization performance of deepfake text detection schemes.

摘要: 语言生成模型的最新进展使得创造令人信服的合成文本或深度假文本成为可能。先前的工作已经证明了滥用深度虚假文本来误导内容消费者的可能性。因此，区分人类和机器生成的文本的任务--深度虚假文本检测变得越来越关键。已经提出了几种针对深度虚假文本检测的防御措施。然而，我们对它们在现实世界中的适用性缺乏透彻的了解。在本文中，我们从基于Transformer的工具支持的4个在线服务中收集深度虚假文本，以评估这些防御措施对野生内容的泛化能力。我们开发了几种低成本的对抗性攻击，并研究了现有防御措施对自适应攻击者的健壮性。我们发现，与最初声称的性能相比，在我们的评估情景下，许多防御系统的性能显著下降。我们的评估表明，挖掘文本内容中的语义信息是一种很有前途的方法，可以提高深度虚假文本检测方案的稳健性和泛化性能。



## **47. Towards Generating Adversarial Examples on Mixed-type Data**

在混合类型数据上生成对抗性实例 cs.LG

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09405v1) [paper-pdf](http://arxiv.org/pdf/2210.09405v1)

**Authors**: Han Xu, Menghai Pan, Zhimeng Jiang, Huiyuan Chen, Xiaoting Li, Mahashweta Das, Hao Yang

**Abstract**: The existence of adversarial attacks (or adversarial examples) brings huge concern about the machine learning (ML) model's safety issues. For many safety-critical ML tasks, such as financial forecasting, fraudulent detection, and anomaly detection, the data samples are usually mixed-type, which contain plenty of numerical and categorical features at the same time. However, how to generate adversarial examples with mixed-type data is still seldom studied. In this paper, we propose a novel attack algorithm M-Attack, which can effectively generate adversarial examples in mixed-type data. Based on M-Attack, attackers can attempt to mislead the targeted classification model's prediction, by only slightly perturbing both the numerical and categorical features in the given data samples. More importantly, by adding designed regularizations, our generated adversarial examples can evade potential detection models, which makes the attack indeed insidious. Through extensive empirical studies, we validate the effectiveness and efficiency of our attack method and evaluate the robustness of existing classification models against our proposed attack. The experimental results highlight the feasibility of generating adversarial examples toward machine learning models in real-world applications.

摘要: 对抗性攻击(或对抗性例子)的存在给机器学习(ML)模型的安全问题带来了巨大的担忧。对于金融预测、欺诈检测、异常检测等许多安全关键的ML任务，数据样本通常是混合类型的，同时包含大量的数值和分类特征。然而，如何利用混合类型数据生成对抗性实例的研究还很少。本文提出了一种新的攻击算法M-Attack，该算法能够有效地生成混合类型数据中的对抗性实例。基于M-攻击，攻击者只需对给定数据样本中的数值特征和分类特征稍加干扰，就可以试图误导目标分类模型的预测。更重要的是，通过添加设计的正则化，我们生成的敌意示例可以避开潜在的检测模型，这使得攻击实际上是隐蔽的。通过大量的实验研究，我们验证了我们的攻击方法的有效性和效率，并评估了现有分类模型对我们提出的攻击的健壮性。实验结果突出了在实际应用中生成针对机器学习模型的对抗性示例的可行性。



## **48. Marksman Backdoor: Backdoor Attacks with Arbitrary Target Class**

神枪手后门：任意目标等级的后门攻击 cs.CR

Accepted to NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.09194v1) [paper-pdf](http://arxiv.org/pdf/2210.09194v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Ping Li

**Abstract**: In recent years, machine learning models have been shown to be vulnerable to backdoor attacks. Under such attacks, an adversary embeds a stealthy backdoor into the trained model such that the compromised models will behave normally on clean inputs but will misclassify according to the adversary's control on maliciously constructed input with a trigger. While these existing attacks are very effective, the adversary's capability is limited: given an input, these attacks can only cause the model to misclassify toward a single pre-defined or target class. In contrast, this paper exploits a novel backdoor attack with a much more powerful payload, denoted as Marksman, where the adversary can arbitrarily choose which target class the model will misclassify given any input during inference. To achieve this goal, we propose to represent the trigger function as a class-conditional generative model and to inject the backdoor in a constrained optimization framework, where the trigger function learns to generate an optimal trigger pattern to attack any target class at will while simultaneously embedding this generative backdoor into the trained model. Given the learned trigger-generation function, during inference, the adversary can specify an arbitrary backdoor attack target class, and an appropriate trigger causing the model to classify toward this target class is created accordingly. We show empirically that the proposed framework achieves high attack performance while preserving the clean-data performance in several benchmark datasets, including MNIST, CIFAR10, GTSRB, and TinyImageNet. The proposed Marksman backdoor attack can also easily bypass existing backdoor defenses that were originally designed against backdoor attacks with a single target class. Our work takes another significant step toward understanding the extensive risks of backdoor attacks in practice.

摘要: 近年来，机器学习模型被证明容易受到后门攻击。在这样的攻击下，对手在训练的模型中嵌入一个秘密的后门，这样受攻击的模型将在干净的输入上正常运行，但将根据对手对带有触发器的恶意构建的输入的控制进行错误分类。虽然这些现有的攻击非常有效，但对手的能力是有限的：在给定输入的情况下，这些攻击只能导致模型错误分类为单个预定义或目标类。相反，本文利用了一种新的后门攻击，具有更强大的有效载荷，表示为射手，其中对手可以任意选择模型将错误分类的目标类别，在推理过程中给定任何输入。为了实现这一目标，我们建议将触发函数表示为类条件生成模型，并在约束优化框架中注入后门，其中触发函数学习生成任意攻击目标类的最优触发模式，同时将该生成后门嵌入到训练的模型中。给定学习的触发器生成函数，在推理期间，对手可以指定任意的后门攻击目标类，并且相应地创建导致模型向该目标类分类的适当触发器。我们在MNIST、CIFAR10、GTSRB和TinyImageNet等几个基准数据集上的实验表明，该框架在保持干净数据性能的同时，实现了高的攻击性能。拟议的射手后门攻击也可以很容易地绕过现有的后门防御，这些后门防御最初是针对单一目标类别的后门攻击而设计的。我们的工作朝着了解后门攻击在实践中的广泛风险又迈出了重要的一步。



## **49. Adversarial Robustness is at Odds with Lazy Training**

对抗健壮性与懒惰训练不一致 cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2207.00411v2) [paper-pdf](http://arxiv.org/pdf/2207.00411v2)

**Authors**: Yunjuan Wang, Enayat Ullah, Poorya Mianjy, Raman Arora

**Abstract**: Recent works show that adversarial examples exist for random neural networks [Daniely and Schacham, 2020] and that these examples can be found using a single step of gradient ascent [Bubeck et al., 2021]. In this work, we extend this line of work to "lazy training" of neural networks -- a dominant model in deep learning theory in which neural networks are provably efficiently learnable. We show that over-parametrized neural networks that are guaranteed to generalize well and enjoy strong computational guarantees remain vulnerable to attacks generated using a single step of gradient ascent.

摘要: 最近的工作表明，随机神经网络存在对抗性的例子[Daniely和Schacham，2020]，并且这些例子可以使用单一的梯度上升步骤来找到[Bubeck等人，2021]。在这项工作中，我们将这一工作扩展到神经网络的“懒惰训练”--深度学习理论中的一种主要模型，在这种模型中，神经网络被证明是可有效学习的。我们证明了过度参数化的神经网络具有良好的泛化能力和强大的计算保证，但仍然容易受到使用单步梯度上升产生的攻击。



## **50. DE-CROP: Data-efficient Certified Robustness for Pretrained Classifiers**

反裁剪：用于预先训练的分类器的数据高效认证稳健性 cs.LG

WACV 2023. Project page: https://sites.google.com/view/decrop

**SubmitDate**: 2022-10-17    [abs](http://arxiv.org/abs/2210.08929v1) [paper-pdf](http://arxiv.org/pdf/2210.08929v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Certified defense using randomized smoothing is a popular technique to provide robustness guarantees for deep neural networks against l2 adversarial attacks. Existing works use this technique to provably secure a pretrained non-robust model by training a custom denoiser network on entire training data. However, access to the training set may be restricted to a handful of data samples due to constraints such as high transmission cost and the proprietary nature of the data. Thus, we formulate a novel problem of "how to certify the robustness of pretrained models using only a few training samples". We observe that training the custom denoiser directly using the existing techniques on limited samples yields poor certification. To overcome this, our proposed approach (DE-CROP) generates class-boundary and interpolated samples corresponding to each training sample, ensuring high diversity in the feature space of the pretrained classifier. We train the denoiser by maximizing the similarity between the denoised output of the generated sample and the original training sample in the classifier's logit space. We also perform distribution level matching using domain discriminator and maximum mean discrepancy that yields further benefit. In white box setup, we obtain significant improvements over the baseline on multiple benchmark datasets and also report similar performance under the challenging black box setup.

摘要: 使用随机化平滑的认证防御是一种流行的技术，可以为深层神经网络提供抵御L2攻击的健壮性保证。现有的工作使用这种技术来通过在整个训练数据上训练自定义去噪器网络来证明预先训练的非稳健模型的安全。然而，由于诸如高传输成本和数据的专有性质等限制，对训练集的访问可能被限制为少数数据样本。因此，我们提出了一个新的问题：如何仅用几个训练样本来证明预先训练的模型的稳健性。我们观察到，直接使用现有技术在有限的样本上培训自定义去噪器会产生较差的认证。为了克服这一问题，我们提出的方法(DE-CROP)生成对应于每个训练样本的类边界样本和内插样本，从而确保预先训练的分类器的特征空间具有高度的多样性。在分类器的Logit空间中，我们通过最大化生成样本的去噪输出与原始训练样本之间的相似度来训练去噪器。我们还使用域鉴别器和最大平均差异来执行分布级别匹配，从而产生进一步的好处。在白盒设置中，我们在多个基准数据集上获得了比基线显著的改进，并且在具有挑战性的黑盒设置下也报告了类似的性能。



