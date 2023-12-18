# Latest Adversarial Attack Papers
**update at 2023-12-18 10:04:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Dynamic Gradient Balancing for Enhanced Adversarial Attacks on Multi-Task Models**

多任务模型增强对抗性攻击的动态梯度平衡算法 cs.LG

19 pages, 6 figures; AAAI24

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2305.12066v2) [paper-pdf](http://arxiv.org/pdf/2305.12066v2)

**Authors**: Lijun Zhang, Xiao Liu, Kaleel Mahmood, Caiwen Ding, Hui Guan

**Abstract**: Multi-task learning (MTL) creates a single machine learning model called multi-task model to simultaneously perform multiple tasks. Although the security of single task classifiers has been extensively studied, there are several critical security research questions for multi-task models including 1) How secure are multi-task models to single task adversarial machine learning attacks, 2) Can adversarial attacks be designed to attack multiple tasks simultaneously, and 3) Does task sharing and adversarial training increase multi-task model robustness to adversarial attacks? In this paper, we answer these questions through careful analysis and rigorous experimentation. First, we develop na\"ive adaptation of single-task white-box attacks and analyze their inherent drawbacks. We then propose a novel attack framework, Dynamic Gradient Balancing Attack (DGBA). Our framework poses the problem of attacking a multi-task model as an optimization problem based on averaged relative loss change, which can be solved by approximating the problem as an integer linear programming problem. Extensive evaluation on two popular MTL benchmarks, NYUv2 and Tiny-Taxonomy, demonstrates the effectiveness of DGBA compared to na\"ive multi-task attack baselines on both clean and adversarially trained multi-task models. The results also reveal a fundamental trade-off between improving task accuracy by sharing parameters across tasks and undermining model robustness due to increased attack transferability from parameter sharing. DGBA is open-sourced and available at https://github.com/zhanglijun95/MTLAttack-DGBA.

摘要: 多任务学习（MTL）创建一个称为多任务模型的单一机器学习模型，以同时执行多个任务。虽然单任务分类器的安全性已经得到了广泛的研究，但多任务模型仍然存在几个关键的安全研究问题，包括1）多任务模型对单任务对抗性机器学习攻击的安全性如何，2）对抗性攻击是否可以被设计为同时攻击多个任务，以及3）任务共享和对抗性训练是否增加了多任务模型对对抗性攻击的鲁棒性？在本文中，我们回答这些问题，通过仔细的分析和严格的实验。首先，我们开发了单任务白盒攻击的朴素适应，并分析了它们固有的缺点。然后，我们提出了一种新的攻击框架，动态梯度平衡攻击（DGBA）。我们的框架提出的问题，攻击的多任务模型作为一个优化问题的基础上平均相对损失的变化，这可以通过近似的整数线性规划问题的问题来解决。对两个流行的MTL基准测试NYUv 2和Tiny-Taxonomy的广泛评估表明，DGBA与干净和对抗训练的多任务模型上的天真多任务攻击基线相比是有效的。结果还揭示了一个根本的权衡提高任务的准确性，通过共享参数的任务和破坏模型的鲁棒性，由于增加攻击的可转移性，从参数共享。DGBA是开源的，可在https://github.com/zhanglijun95/MTLAttack-DGBA上获得。



## **2. Exploring Adversarial Robustness of Vision Transformers in the Spectral Perspective**

从频谱角度探讨视觉转换器的对抗稳健性 cs.CV

Accepted in IEEE/CVF Winter Conference on Applications of Computer  Vision (WACV) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2208.09602v2) [paper-pdf](http://arxiv.org/pdf/2208.09602v2)

**Authors**: Gihyun Kim, Juyeop Kim, Jong-Seok Lee

**Abstract**: The Vision Transformer has emerged as a powerful tool for image classification tasks, surpassing the performance of convolutional neural networks (CNNs). Recently, many researchers have attempted to understand the robustness of Transformers against adversarial attacks. However, previous researches have focused solely on perturbations in the spatial domain. This paper proposes an additional perspective that explores the adversarial robustness of Transformers against frequency-selective perturbations in the spectral domain. To facilitate comparison between these two domains, an attack framework is formulated as a flexible tool for implementing attacks on images in the spatial and spectral domains. The experiments reveal that Transformers rely more on phase and low frequency information, which can render them more vulnerable to frequency-selective attacks than CNNs. This work offers new insights into the properties and adversarial robustness of Transformers.

摘要: 视觉转换器已经成为一种强大的图像分类工具，其性能超过了卷积神经网络(CNN)。最近，许多研究人员试图了解变形金刚对对手攻击的健壮性。然而，以往的研究主要集中在空间域的扰动上。本文提出了一个额外的视角，探讨了在谱域中变压器对频率选择性扰动的对抗稳健性。为了便于这两个域之间的比较，提出了一种攻击框架，作为在空域和谱域对图像实施攻击的灵活工具。实验表明，变形金刚更依赖于相位和低频信息，这使得它们比CNN更容易受到频率选择性攻击。这项工作为了解变形金刚的特性和对抗健壮性提供了新的见解。



## **3. LogoStyleFool: Vitiating Video Recognition Systems via Logo Style Transfer**

LogoStyleFool：通过徽标样式转换破坏视频识别系统 cs.CV

13 pages, 3 figures. Accepted to AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09935v1) [paper-pdf](http://arxiv.org/pdf/2312.09935v1)

**Authors**: Yuxin Cao, Ziyu Zhao, Xi Xiao, Derui Wang, Minhui Xue, Jin Lu

**Abstract**: Video recognition systems are vulnerable to adversarial examples. Recent studies show that style transfer-based and patch-based unrestricted perturbations can effectively improve attack efficiency. These attacks, however, face two main challenges: 1) Adding large stylized perturbations to all pixels reduces the naturalness of the video and such perturbations can be easily detected. 2) Patch-based video attacks are not extensible to targeted attacks due to the limited search space of reinforcement learning that has been widely used in video attacks recently. In this paper, we focus on the video black-box setting and propose a novel attack framework named LogoStyleFool by adding a stylized logo to the clean video. We separate the attack into three stages: style reference selection, reinforcement-learning-based logo style transfer, and perturbation optimization. We solve the first challenge by scaling down the perturbation range to a regional logo, while the second challenge is addressed by complementing an optimization stage after reinforcement learning. Experimental results substantiate the overall superiority of LogoStyleFool over three state-of-the-art patch-based attacks in terms of attack performance and semantic preservation. Meanwhile, LogoStyleFool still maintains its performance against two existing patch-based defense methods. We believe that our research is beneficial in increasing the attention of the security community to such subregional style transfer attacks.

摘要: 视频识别系统很容易受到敌意例子的攻击。最近的研究表明，基于风格迁移和基于补丁的无限制扰动可以有效地提高攻击效率。然而，这些攻击面临两个主要挑战：1)向所有像素添加大的风格化扰动会降低视频的自然度，并且这种扰动很容易被检测到。2)基于补丁的视频攻击不能扩展到有针对性的攻击，因为强化学习的搜索空间有限，这是近年来在视频攻击中广泛使用的。本文针对视频黑盒的设置，通过在干净的视频中添加一个风格化的标识，提出了一种新的攻击框架--LogoStyleFool。我们将攻击分为三个阶段：样式参考选择、基于强化学习的标识样式迁移和扰动优化。我们通过将扰动范围缩小到区域标志来解决第一个挑战，而第二个挑战是通过在强化学习后补充优化阶段来解决的。实验结果表明，在攻击性能和语义保持方面，LogoStyleFool在攻击性能和语义保持方面都优于三种最先进的基于补丁的攻击。同时，与现有的两种基于补丁的防御方法相比，LogoStyleFool仍然保持其性能。我们认为，我们的研究有助于提高安全界对这种次区域风格的转移袭击的关注。



## **4. A Game-theoretic Framework for Privacy-preserving Federated Learning**

保护隐私的联邦学习的博弈论框架 cs.LG

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2304.05836v2) [paper-pdf](http://arxiv.org/pdf/2304.05836v2)

**Authors**: Xiaojin Zhang, Lixin Fan, Siwei Wang, Wenjie Li, Kai Chen, Qiang Yang

**Abstract**: In federated learning, benign participants aim to optimize a global model collaboratively. However, the risk of \textit{privacy leakage} cannot be ignored in the presence of \textit{semi-honest} adversaries. Existing research has focused either on designing protection mechanisms or on inventing attacking mechanisms. While the battle between defenders and attackers seems never-ending, we are concerned with one critical question: is it possible to prevent potential attacks in advance? To address this, we propose the first game-theoretic framework that considers both FL defenders and attackers in terms of their respective payoffs, which include computational costs, FL model utilities, and privacy leakage risks. We name this game the federated learning privacy game (FLPG), in which neither defenders nor attackers are aware of all participants' payoffs.   To handle the \textit{incomplete information} inherent in this situation, we propose associating the FLPG with an \textit{oracle} that has two primary responsibilities. First, the oracle provides lower and upper bounds of the payoffs for the players. Second, the oracle acts as a correlation device, privately providing suggested actions to each player. With this novel framework, we analyze the optimal strategies of defenders and attackers. Furthermore, we derive and demonstrate conditions under which the attacker, as a rational decision-maker, should always follow the oracle's suggestion \textit{not to attack}.

摘要: 在联合学习中，良性参与者的目标是协作优化全球模型。然而，在存在半诚实的对手的情况下，隐私泄露的风险是不容忽视的。现有的研究要么集中在设计保护机制上，要么集中在发明攻击机制上。虽然防御者和攻击者之间的战斗似乎永无止境，但我们关心的是一个关键问题：是否有可能提前防止潜在的攻击？为了解决这一问题，我们提出了第一个博弈论框架，该框架考虑了FL防御者和攻击者各自的收益，其中包括计算成本、FL模型效用和隐私泄露风险。我们将这款游戏命名为联邦学习隐私游戏(FLPG)，在该游戏中，防御者和攻击者都不知道所有参与者的收益。为了处理这种情况下固有的不完整信息，我们建议将FLPG与具有两个主要职责的\textit{Oracle}相关联。首先，先知为玩家提供了收益的上下限。其次，先知充当了关联设备，私下向每个玩家提供建议的动作。在此框架下，我们分析了防御者和攻击者的最优策略。此外，我们还推导并证明了攻击者作为理性决策者应始终遵循神谕的建议的条件。



## **5. FlowMur: A Stealthy and Practical Audio Backdoor Attack with Limited Knowledge**

FlowMur：一种隐蔽实用的有限知识音频后门攻击 cs.CR

To appear at lEEE Symposium on Security & Privacy (Oakland) 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09665v1) [paper-pdf](http://arxiv.org/pdf/2312.09665v1)

**Authors**: Jiahe Lan, Jie Wang, Baochen Yan, Zheng Yan, Elisa Bertino

**Abstract**: Speech recognition systems driven by DNNs have revolutionized human-computer interaction through voice interfaces, which significantly facilitate our daily lives. However, the growing popularity of these systems also raises special concerns on their security, particularly regarding backdoor attacks. A backdoor attack inserts one or more hidden backdoors into a DNN model during its training process, such that it does not affect the model's performance on benign inputs, but forces the model to produce an adversary-desired output if a specific trigger is present in the model input. Despite the initial success of current audio backdoor attacks, they suffer from the following limitations: (i) Most of them require sufficient knowledge, which limits their widespread adoption. (ii) They are not stealthy enough, thus easy to be detected by humans. (iii) Most of them cannot attack live speech, reducing their practicality. To address these problems, in this paper, we propose FlowMur, a stealthy and practical audio backdoor attack that can be launched with limited knowledge. FlowMur constructs an auxiliary dataset and a surrogate model to augment adversary knowledge. To achieve dynamicity, it formulates trigger generation as an optimization problem and optimizes the trigger over different attachment positions. To enhance stealthiness, we propose an adaptive data poisoning method according to Signal-to-Noise Ratio (SNR). Furthermore, ambient noise is incorporated into the process of trigger generation and data poisoning to make FlowMur robust to ambient noise and improve its practicality. Extensive experiments conducted on two datasets demonstrate that FlowMur achieves high attack performance in both digital and physical settings while remaining resilient to state-of-the-art defenses. In particular, a human study confirms that triggers generated by FlowMur are not easily detected by participants.

摘要: 由DNN驱动的语音识别系统通过语音接口使人机交互发生了革命性的变化，极大地方便了我们的日常生活。然而，这些系统越来越受欢迎，也引发了对其安全性的特别关注，特别是关于后门攻击的问题。后门攻击在其训练过程中将一个或多个隐藏的后门插入到DNN模型中，使得它不会影响模型在良性输入上的性能，但如果模型输入中存在特定触发器，则迫使模型产生对手期望的输出。尽管目前的音频后门攻击取得了初步的成功，但它们受到以下限制：(I)大多数音频后门攻击需要足够的知识，这限制了它们的广泛采用。(Ii)它们的隐蔽性不够，因此很容易被人发现。(3)他们中的大多数不能攻击现场演讲，降低了他们的实用性。为了解决这些问题，在本文中，我们提出了FlowMur，一种隐蔽而实用的音频后门攻击，可以在有限的知识下发起。FlowMur构建了一个辅助数据集和一个代理模型来增强对手的知识。为了实现动态化，它将触发器的生成描述为一个优化问题，并在不同的附着位置上对触发器进行优化。为了提高隐蔽性，提出了一种基于信噪比的自适应数据毒化方法。此外，在触发产生和数据毒化过程中引入了环境噪声，使FlowMur对环境噪声具有较强的鲁棒性，提高了其实用性。在两个数据集上进行的广泛实验表明，FlowMur在数字和物理环境中都实现了高攻击性能，同时对最先进的防御保持了弹性。特别是，一项人类研究证实，FlowMur产生的触发因素不容易被参与者检测到。



## **6. Unsupervised and Supervised learning by Dense Associative Memory under replica symmetry breaking**

副本对称破缺下稠密联想记忆的无监督和有监督学习 cond-mat.dis-nn

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09638v1) [paper-pdf](http://arxiv.org/pdf/2312.09638v1)

**Authors**: Linda Albanese, Andrea Alessandrelli, Alessia Annibale, Adriano Barra

**Abstract**: Statistical mechanics of spin glasses is one of the main strands toward a comprehension of information processing by neural networks and learning machines. Tackling this approach, at the fairly standard replica symmetric level of description, recently Hebbian attractor networks with multi-node interactions (often called Dense Associative Memories) have been shown to outperform their classical pairwise counterparts in a number of tasks, from their robustness against adversarial attacks and their capability to work with prohibitively weak signals to their supra-linear storage capacities. Focusing on mathematical techniques more than computational aspects, in this paper we relax the replica symmetric assumption and we derive the one-step broken-replica-symmetry picture of supervised and unsupervised learning protocols for these Dense Associative Memories: a phase diagram in the space of the control parameters is achieved, independently, both via the Parisi's hierarchy within then replica trick as well as via the Guerra's telescope within the broken-replica interpolation. Further, an explicit analytical investigation is provided to deepen both the big-data and ground state limits of these networks as well as a proof that replica symmetry breaking does not alter the thresholds for learning and slightly increases the maximal storage capacity. Finally the De Almeida and Thouless line, depicting the onset of instability of a replica symmetric description, is also analytically derived highlighting how, crossed this boundary, the broken replica description should be preferred.

摘要: 自旋玻璃的统计力学是通过神经网络和学习机理解信息处理的主要途径之一。为了处理这种方法，在相当标准的副本对称描述水平上，最近具有多节点交互的Hebbian吸引子网络(通常被称为密集联想记忆)已经被证明在许多任务中表现出比它们的经典成对同行更好的性能，从它们对对手攻击的健壮性和它们处理令人望而却步的弱信号的能力到它们的超线性存储能力。本文更多地关注数学技术而不是计算方面，放松了副本对称假设，导出了这些密集联想记忆的监督和非监督学习协议的一步破坏副本对称图：控制参数空间中的相图既可以通过副本技巧中的Parisi层次结构独立获得，也可以通过破碎副本内插中的Guera望远镜独立获得。此外，还提供了一个明确的分析研究，以加深这些网络的大数据和基态限制，并证明了副本对称性破坏不会改变学习阈值，并略微增加了最大存储容量。最后，描述复制品对称描述的不稳定性开始的de Almeida和Thouless线也被解析地推导出来，突出了如何跨越这一边界，破碎的复制品描述应该是首选的。



## **7. A Malware Classification Survey on Adversarial Attacks and Defences**

对抗性攻击与防御的恶意软件分类综述 cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09636v1) [paper-pdf](http://arxiv.org/pdf/2312.09636v1)

**Authors**: Mahesh Datta Sai Ponnuru, Likhitha Amasala, Tanu Sree Bhimavarapu, Guna Chaitanya Garikipati

**Abstract**: As the number and complexity of malware attacks continue to increase, there is an urgent need for effective malware detection systems. While deep learning models are effective at detecting malware, they are vulnerable to adversarial attacks. Attacks like this can create malicious files that are resistant to detection, creating a significant cybersecurity risk. Recent research has seen the development of several adversarial attack and response approaches aiming at strengthening deep learning models' resilience to such attacks. This survey study offers an in-depth look at current research in adversarial attack and defensive strategies for malware classification in cybersecurity. The methods are classified into four categories: generative models, feature-based approaches, ensemble methods, and hybrid tactics. The article outlines cutting-edge procedures within each area, assessing their benefits and drawbacks. Each topic presents cutting-edge approaches and explores their advantages and disadvantages. In addition, the study discusses the datasets and assessment criteria that are often utilized on this subject. Finally, it identifies open research difficulties and suggests future study options. This document is a significant resource for malware categorization and cyber security researchers and practitioners.

摘要: 随着恶意软件攻击的数量和复杂性不断增加，迫切需要有效的恶意软件检测系统。虽然深度学习模型在检测恶意软件方面很有效，但它们很容易受到对手攻击。像这样的攻击可以创建抵抗检测的恶意文件，从而造成重大的网络安全风险。最近的研究已经看到了几种对抗性攻击和响应方法的发展，旨在加强深度学习模型对此类攻击的弹性。这项调查研究对当前网络安全中恶意软件分类的对抗性攻击和防御策略的研究进行了深入的探讨。这些方法分为四类：生成模型、基于特征的方法、集成方法和混合策略。这篇文章概述了每个领域的尖端程序，评估了它们的优点和缺点。每个主题都介绍了最新的方法，并探讨了它们的优缺点。此外，这项研究还讨论了经常用于这一主题的数据集和评估标准。最后，指出了尚待解决的研究困难，并提出了未来的研究选择。本文档是恶意软件分类和网络安全研究人员和从业者的重要资源。



## **8. Understanding and Improving Adversarial Attacks on Latent Diffusion Model**

对潜在扩散模型的敌意攻击的理解与改进 cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2310.04687v2) [paper-pdf](http://arxiv.org/pdf/2310.04687v2)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Latent Diffusion Model (LDM) achieves state-of-the-art performances in image generation yet raising copyright and privacy concerns. Adversarial attacks on LDM are then born to protect unauthorized images from being used in LDM-driven few-shot generation. However, these attacks suffer from moderate performance and excessive computational cost, especially in GPU memory. In this paper, we propose an effective adversarial attack on LDM that shows superior performance against state-of-the-art few-shot generation pipeline of LDM, for example, LoRA. We implement the attack with memory efficiency by introducing several mechanisms and decrease the memory cost of the attack to less than 6GB, which allows individual users to run the attack on a majority of consumer GPUs. Our proposed attack can be a practical tool for people facing the copyright and privacy risk brought by LDM to protect themselves.

摘要: 潜在扩散模型(LDM)在图像生成方面实现了最先进的性能，但也引发了版权和隐私问题。然后，针对LDM的对抗性攻击诞生，以保护未经授权的图像不被用于LDM驱动的少镜头生成。然而，这些攻击的性能中等，计算成本过高，特别是在GPU内存中。在本文中，我们提出了一种对LDM的有效的对抗性攻击，该攻击相对于目前最先进的LDM的少镜头生成流水线，例如LORA，表现出了优越的性能。我们通过引入几种机制来实现具有内存效率的攻击，并将攻击的内存成本降低到6 GB以下，从而允许单个用户在大多数消费类GPU上运行攻击。我们提出的攻击可以为面临LDM带来的版权和隐私风险的人提供一个实用的工具来保护自己。



## **9. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

在物理世界中走向可转移的定向3D对抗攻击 cs.CV

11 pages, 7 figures

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09558v1) [paper-pdf](http://arxiv.org/pdf/2312.09558v1)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.

摘要: 与可转移的非定向攻击相比，可转移的定向攻击可以指定对手样本的错误分类类别，对安全关键任务构成更大的威胁。同时，3D对抗性样本由于其潜在的多视点稳健性，可以更全面地识别现有深度学习系统中的弱点，具有很大的应用价值。然而，可转移的定向3D对抗性攻击领域仍然空白。这项工作的目标是开发一种更有效的技术，可以生成可转移的目标3D对抗性实例，填补这一领域的空白。为了实现这一目标，我们设计了一种新的框架TT3D，它可以从少量的多视角图像快速重建为可转移的目标3D纹理网格。针对现有的基于网格的纹理优化方法在高维网格空间中计算梯度，容易陷入局部最优，导致可移植性差和失真明显的问题，TT3D创新性地在基于网格的NERF空间中对特征网格和多层感知器(MLP)参数进行双重优化，在享受自然感的同时显著增强了黑盒的可传递性。实验结果表明，TT3D不仅表现出了良好的跨模型可移植性，而且在不同的渲染和视觉任务之间保持了相当大的适应性。更重要的是，我们用3D打印技术在真实世界中生成了3D对抗性例子，并验证了它们在各种场景下的健壮性。



## **10. Embodied Adversarial Attack: A Dynamic Robust Physical Attack in Autonomous Driving**

具体化对抗性攻击：自主驾驶中的一种动态健壮身体攻击 cs.CV

10 pages, 7 figures

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09554v1) [paper-pdf](http://arxiv.org/pdf/2312.09554v1)

**Authors**: Yitong Sun, Yao Huang, Xingxing Wei

**Abstract**: As physical adversarial attacks become extensively applied in unearthing the potential risk of security-critical scenarios, especially in autonomous driving, their vulnerability to environmental changes has also been brought to light. The non-robust nature of physical adversarial attack methods brings less-than-stable performance consequently. To enhance the robustness of physical adversarial attacks in the real world, instead of statically optimizing a robust adversarial example via an off-line training manner like the existing methods, this paper proposes a brand new robust adversarial attack framework: Embodied Adversarial Attack (EAA) from the perspective of dynamic adaptation, which aims to employ the paradigm of embodied intelligence: Perception-Decision-Control to dynamically adjust the optimal attack strategy according to the current situations in real time. For the perception module, given the challenge of needing simulation for the victim's viewpoint, EAA innovatively devises a Perspective Transformation Network to estimate the target's transformation from the attacker's perspective. For the decision and control module, EAA adopts the laser-a highly manipulable medium to implement physical attacks, and further trains an attack agent with reinforcement learning to make it capable of instantaneously determining the best attack strategy based on the perceived information. Finally, we apply our framework to the autonomous driving scenario. A variety of experiments verify the high effectiveness of our method under complex scenes.

摘要: 随着物理对抗性攻击在挖掘安全关键场景的潜在风险方面得到广泛应用，特别是在自动驾驶中，它们对环境变化的脆弱性也暴露了出来。因此，物理对抗性攻击方法的非健壮性带来了不稳定的性能。为了提高物理对抗攻击在现实世界中的健壮性，不像现有方法那样通过离线训练的方式静态地优化健壮的对抗实例，从动态适应的角度提出了一种全新的健壮对抗攻击框架：具体化对抗性攻击(EAA)，旨在利用具体化智能的范式：感知-决策-控制来实时地根据当前情况动态地调整最优攻击策略。对于感知模块，针对受害者视点需要模拟的挑战，EAA创新性地设计了一个视角转换网络来从攻击者的角度估计目标的变化。对于决策和控制模块，EAA采用激光这一高度可操控的介质来实施物理攻击，并进一步使用强化学习来训练攻击代理，使其能够根据感知的信息即时确定最佳攻击策略。最后，我们将我们的框架应用于自动驾驶场景。大量实验验证了该方法在复杂场景下的高效性。



## **11. Adversarial Robustness on Image Classification with $k$-means**

基于$k$-均值的图像分类的对抗稳健性 cs.LG

6 pages, 3 figures, 2 equations, 1 algorithm

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09533v1) [paper-pdf](http://arxiv.org/pdf/2312.09533v1)

**Authors**: Rollin Omari, Junae Kim, Paul Montague

**Abstract**: In this paper we explore the challenges and strategies for enhancing the robustness of $k$-means clustering algorithms against adversarial manipulations. We evaluate the vulnerability of clustering algorithms to adversarial attacks, emphasising the associated security risks. Our study investigates the impact of incremental attack strength on training, introduces the concept of transferability between supervised and unsupervised models, and highlights the sensitivity of unsupervised models to sample distributions. We additionally introduce and evaluate an adversarial training method that improves testing performance in adversarial scenarios, and we highlight the importance of various parameters in the proposed training method, such as continuous learning, centroid initialisation, and adversarial step-count.

摘要: 在这篇文章中，我们探讨了增强$k$-Means聚类算法对对手操纵的健壮性的挑战和策略。我们评估了集群算法对敌意攻击的脆弱性，强调了相关的安全风险。我们的研究考察了增量攻击强度对训练的影响，引入了监督模型和非监督模型之间可转换性的概念，并强调了非监督模型对样本分布的敏感性。此外，我们还介绍和评估了一种提高对抗场景下测试性能的对抗训练方法，并强调了所提出的训练方法中各种参数的重要性，如连续学习、质心初始化和对抗步数。



## **12. SlowTrack: Increasing the Latency of Camera-based Perception in Autonomous Driving Using Adversarial Examples**

SlowTrack：使用对抗性例子增加自动驾驶中基于摄像头的感知的延迟 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09520v1) [paper-pdf](http://arxiv.org/pdf/2312.09520v1)

**Authors**: Chen Ma, Ningfei Wang, Qi Alfred Chen, Chao Shen

**Abstract**: In Autonomous Driving (AD), real-time perception is a critical component responsible for detecting surrounding objects to ensure safe driving. While researchers have extensively explored the integrity of AD perception due to its safety and security implications, the aspect of availability (real-time performance) or latency has received limited attention. Existing works on latency-based attack have focused mainly on object detection, i.e., a component in camera-based AD perception, overlooking the entire camera-based AD perception, which hinders them to achieve effective system-level effects, such as vehicle crashes. In this paper, we propose SlowTrack, a novel framework for generating adversarial attacks to increase the execution time of camera-based AD perception. We propose a novel two-stage attack strategy along with the three new loss function designs. Our evaluation is conducted on four popular camera-based AD perception pipelines, and the results demonstrate that SlowTrack significantly outperforms existing latency-based attacks while maintaining comparable imperceptibility levels. Furthermore, we perform the evaluation on Baidu Apollo, an industry-grade full-stack AD system, and LGSVL, a production-grade AD simulator, with two scenarios to compare the system-level effects of SlowTrack and existing attacks. Our evaluation results show that the system-level effects can be significantly improved, i.e., the vehicle crash rate of SlowTrack is around 95% on average while existing works only have around 30%.

摘要: 在自动驾驶中，实时感知是负责检测周围物体以确保安全驾驶的关键部件。虽然由于AD感知的安全性和安全性，研究人员已经对其完整性进行了广泛的探索，但可用性(实时性能)或延迟方面的关注有限。现有的基于延迟攻击的研究主要集中于目标检测，即基于摄像头的广告感知中的一个组件，而忽略了整个基于摄像头的广告感知，这阻碍了它们达到有效的系统级效果，如车辆碰撞。在本文中，我们提出了一种新的生成敌意攻击的框架SlowTrack，以增加基于摄像机的广告感知的执行时间。我们提出了一种新的两阶段攻击策略以及三种新的损失函数设计。我们在四个流行的基于摄像头的AD感知管道上进行了评估，结果表明，SlowTrack在保持相当的不可感知性水平的同时，显著优于现有的基于延迟的攻击。此外，我们在工业级全栈AD系统百度Apollo和生产级AD模拟器LGSVL上进行了评估，并通过两个场景比较了SlowTrack和现有攻击的系统级影响。我们的评估结果表明，系统级效果可以得到显著提高，即SlowTrack的车辆撞击率平均在95%左右，而现有的工作只有30%左右。



## **13. Effective and Imperceptible Adversarial Textual Attack via Multi-objectivization**

基于多对象化的有效隐蔽对抗性文本攻击 cs.CL

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2111.01528v4) [paper-pdf](http://arxiv.org/pdf/2111.01528v4)

**Authors**: Shengcai Liu, Ning Lu, Wenjing Hong, Chao Qian, Ke Tang

**Abstract**: The field of adversarial textual attack has significantly grown over the last few years, where the commonly considered objective is to craft adversarial examples (AEs) that can successfully fool the target model. However, the imperceptibility of attacks, which is also essential for practical attackers, is often left out by previous studies. In consequence, the crafted AEs tend to have obvious structural and semantic differences from the original human-written text, making them easily perceptible. In this work, we advocate leveraging multi-objectivization to address such issue. Specifically, we reformulate the problem of crafting AEs as a multi-objective optimization problem, where the attack imperceptibility is considered as an auxiliary objective. Then, we propose a simple yet effective evolutionary algorithm, dubbed HydraText, to solve this problem. To the best of our knowledge, HydraText is currently the only approach that can be effectively applied to both score-based and decision-based attack settings. Exhaustive experiments involving 44237 instances demonstrate that HydraText consistently achieves competitive attack success rates and better attack imperceptibility than the recently proposed attack approaches. A human evaluation study also shows that the AEs crafted by HydraText are more indistinguishable from human-written text. Finally, these AEs exhibit good transferability and can bring notable robustness improvement to the target model by adversarial training.

摘要: 对抗性文本攻击领域在过去几年中显著增长，其中通常被认为的目标是制作能够成功愚弄目标模型的对抗性示例(AE)。然而，攻击的不可感知性对于实际攻击者来说也是必不可少的，但以往的研究往往忽略了这一点。因此，精心制作的AE往往与原始的人类书写的文本在结构和语义上有明显的差异，使它们很容易被察觉。在这项工作中，我们主张利用多对象化来解决这一问题。具体地说，我们将攻击隐蔽性作为辅助目标，将攻击不可感知性问题转化为一个多目标优化问题。然后，我们提出了一个简单而有效的进化算法，称为HydraText，来解决这个问题。据我们所知，HydraText是目前唯一可以有效应用于基于分数和基于决策的攻击设置的方法。44237个实例的详尽实验表明，与最近提出的攻击方法相比，HydraText始终具有与之相当的攻击成功率和更好的攻击不可见性。一项人类评估研究还表明，HydraText制作的AE与人类书写的文本更难区分。最后，这些训练引擎表现出良好的可移植性，通过对抗性训练可以显著提高目标模型的稳健性。



## **14. No-Skim: Towards Efficiency Robustness Evaluation on Skimming-based Language Models**

No-Skim：基于略读的语言模型的效率稳健性评价 cs.CR

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09494v1) [paper-pdf](http://arxiv.org/pdf/2312.09494v1)

**Authors**: Shengyao Zhang, Mi Zhang, Xudong Pan, Min Yang

**Abstract**: To reduce the computation cost and the energy consumption in large language models (LLM), skimming-based acceleration dynamically drops unimportant tokens of the input sequence progressively along layers of the LLM while preserving the tokens of semantic importance. However, our work for the first time reveals the acceleration may be vulnerable to Denial-of-Service (DoS) attacks. In this paper, we propose No-Skim, a general framework to help the owners of skimming-based LLM to understand and measure the robustness of their acceleration scheme. Specifically, our framework searches minimal and unnoticeable perturbations at character-level and token-level to generate adversarial inputs that sufficiently increase the remaining token ratio, thus increasing the computation cost and energy consumption. We systematically evaluate the vulnerability of the skimming acceleration in various LLM architectures including BERT and RoBERTa on the GLUE benchmark. In the worst case, the perturbation found by No-Skim substantially increases the running cost of LLM by over 145% on average. Moreover, No-Skim extends the evaluation framework to various scenarios, making the evaluation conductible with different level of knowledge.

摘要: 为了降低大型语言模型(LLM)的计算代价和能量消耗，基于略读的加速算法在保留语义重要的标记的同时，动态地沿着LLM的层次逐步丢弃输入序列中不重要的标记。然而，我们的工作首次揭示了加速可能容易受到拒绝服务(DoS)攻击。在本文中，我们提出了一个通用的框架No-Skim，以帮助基于略读的LLM的所有者理解和度量其加速方案的健壮性。具体地说，我们的框架在字符级和令牌级搜索最小和不可察觉的扰动，以生成足以增加剩余令牌率的对抗性输入，从而增加计算成本和能量消耗。我们在GLUE基准上系统地评估了包括Bert和Roberta在内的各种LLM架构中掠读加速的脆弱性。在最坏的情况下，No-Skim发现的扰动大大增加了LLM的运行成本，平均超过145%。此外，No-Skim将评估框架扩展到各种场景，使评估可以在不同的知识水平下进行。



## **15. Continual Adversarial Defense**

持续的对抗性防御 cs.CV

**SubmitDate**: 2023-12-15    [abs](http://arxiv.org/abs/2312.09481v1) [paper-pdf](http://arxiv.org/pdf/2312.09481v1)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that can generalize to all types of attacks, including unseen ones, is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks used by many attackers. The defense system needs to upgrade itself by utilizing few-shot defense feedback and efficient memory. Therefore, we propose the first continual adversarial defense (CAD) framework that adapts to any attacks in a dynamic scenario, where various attacks emerge stage by stage. In practice, CAD is modeled under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial images. We leverage cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Experiments conducted on CIFAR-10 and ImageNet-100 validate the effectiveness of our approach against multiple stages of 10 modern adversarial attacks and significant improvements over 10 baseline methods. In particular, CAD is capable of quickly adapting with minimal feedback and a low cost of defense failure, while maintaining good performance against old attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

摘要: 为了应对每月迅速演变的对抗性攻击的性质，提出了许多防御措施，以概括尽可能多的已知攻击。然而，设计一种可以概括所有类型的攻击的防御方法是不现实的，包括看不见的攻击，因为防御系统运行的环境是动态的，包括许多攻击者使用的各种独特攻击。防御系统需要利用少发的防御反馈和高效的内存进行自我升级。因此，我们提出了第一个连续对抗防御(CAD)框架，该框架能够适应动态场景中的任何攻击，其中各种攻击是逐步出现的。在实践中，CAD的建模遵循四个原则：(1)持续适应新的攻击而不发生灾难性遗忘；(2)少镜头适应；(3)内存高效适应；(4)干净图像和对抗性图像的高精度。我们利用尖端的持续学习、少机会学习和整体学习技术来验证这些原则。在CIFAR-10和ImageNet-100上进行的实验验证了该方法对10个现代对抗性攻击的多个阶段的有效性，并比10个基准方法有显著的改进。特别是，CAD能够以最小的反馈和较低的防御失败成本快速适应，同时保持对旧攻击的良好性能。我们的研究揭示了一种针对动态和不断变化的攻击进行持续防御适应的全新范式。



## **16. Coevolutionary Algorithm for Building Robust Decision Trees under Minimax Regret**

极小极大后悔条件下构建稳健决策树的协进化算法 cs.LG

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09078v1) [paper-pdf](http://arxiv.org/pdf/2312.09078v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: In recent years, there has been growing interest in developing robust machine learning (ML) models that can withstand adversarial attacks, including one of the most widely adopted, efficient, and interpretable ML algorithms-decision trees (DTs). This paper proposes a novel coevolutionary algorithm (CoEvoRDT) designed to create robust DTs capable of handling noisy high-dimensional data in adversarial contexts. Motivated by the limitations of traditional DT algorithms, we leverage adaptive coevolution to allow DTs to evolve and learn from interactions with perturbed input data. CoEvoRDT alternately evolves competing populations of DTs and perturbed features, enabling construction of DTs with desired properties. CoEvoRDT is easily adaptable to various target metrics, allowing the use of tailored robustness criteria such as minimax regret. Furthermore, CoEvoRDT has potential to improve the results of other state-of-the-art methods by incorporating their outcomes (DTs they produce) into the initial population and optimize them in the process of coevolution. Inspired by the game theory, CoEvoRDT utilizes mixed Nash equilibrium to enhance convergence. The method is tested on 20 popular datasets and shows superior performance compared to 4 state-of-the-art algorithms. It outperformed all competing methods on 13 datasets with adversarial accuracy metrics, and on all 20 considered datasets with minimax regret. Strong experimental results and flexibility in choosing the error measure make CoEvoRDT a promising approach for constructing robust DTs in real-world applications.

摘要: 近年来，人们对开发能够抵抗对手攻击的健壮机器学习(ML)模型越来越感兴趣，其中包括最广泛采用的、高效的和可解释的ML算法之一-决策树(DTD)。提出了一种新的协同进化算法(CoEvoRDT)，旨在创建能够在对抗性环境中处理噪声高维数据的健壮DT。由于传统DT算法的局限性，我们利用自适应协同进化来允许DT进化，并从与扰动输入数据的交互中学习。CoEvoRDT交替进化相互竞争的DT种群和受干扰的特征，从而能够构建具有所需特性的DT。CoEvoRDT很容易适应各种目标指标，允许使用定制的健壮性标准，如最小最大遗憾。此外，CoEvoRDT通过将其他最先进方法的结果(它们产生的DT)合并到初始种群中并在共同进化过程中对其进行优化，具有改进其他最先进方法的结果的潜力。受博弈论的启发，CoEvoRDT利用混合纳什均衡来增强收敛。该方法在20个流行的数据集上进行了测试，与4种最先进的算法相比，表现出了更好的性能。它在13个带有对抗性准确度指标的数据集上的表现优于所有竞争方法，在所有20个考虑的数据集上的表现都是最小最大遗憾。强大的实验结果和选择误差度量的灵活性使CoEvoRDT成为在实际应用中构造健壮DTD的一种很有前途的方法。



## **17. Concealing Sensitive Samples against Gradient Leakage in Federated Learning**

联合学习中防止梯度泄漏的敏感样本隐藏 cs.LG

Defence against model inversion attack in federated learning

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2209.05724v2) [paper-pdf](http://arxiv.org/pdf/2209.05724v2)

**Authors**: Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi

**Abstract**: Federated Learning (FL) is a distributed learning paradigm that enhances users privacy by eliminating the need for clients to share raw, private data with the server. Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance.

摘要: 联合学习(FL)是一种分布式学习范例，它通过消除客户端与服务器共享原始私人数据的需要来增强用户隐私。尽管取得了成功，但最近的研究暴露了FL在模拟反转攻击中的脆弱性，在这种攻击中，攻击者通过窃听共享的梯度信息来重建用户的私人数据。我们假设，这种攻击成功的一个关键因素是在随机优化过程中批次内每个数据的梯度之间的低纠缠。这造成了一个漏洞，攻击者可以利用该漏洞来重建敏感数据。基于这一见解，我们提出了一种简单而有效的防御策略，该策略将敏感数据的梯度与隐藏的样本混淆。为了实现这一点，我们建议合成隐藏样本来模拟梯度级别的敏感数据，同时确保它们与实际敏感数据的视觉差异。与以前的技术相比，我们的经验评估表明，所提出的技术在保持FL性能的同时提供了最强的保护。



## **18. DRAM-Locker: A General-Purpose DRAM Protection Mechanism against Adversarial DNN Weight Attacks**

DRAM-Locker：一种抵抗敌意DNN权重攻击的通用DRAM保护机制 cs.AR

7 pages. arXiv admin note: text overlap with arXiv:2305.08034

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09027v1) [paper-pdf](http://arxiv.org/pdf/2312.09027v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Arman Roohi, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: In this work, we propose DRAM-Locker as a robust general-purpose defense mechanism that can protect DRAM against various adversarial Deep Neural Network (DNN) weight attacks affecting data or page tables. DRAM-Locker harnesses the capabilities of in-DRAM swapping combined with a lock-table to prevent attackers from singling out specific DRAM rows to safeguard DNN's weight parameters. Our results indicate that DRAM-Locker can deliver a high level of protection downgrading the performance of targeted weight attacks to a random attack level. Furthermore, the proposed defense mechanism demonstrates no reduction in accuracy when applied to CIFAR-10 and CIFAR-100. Importantly, DRAM-Locker does not necessitate any software retraining or result in extra hardware burden.

摘要: 在这项工作中，我们提出了DRAM-Locker作为一种健壮的通用防御机制，可以保护DRAM免受各种影响数据或页表的对抗性深度神经网络(DNN)权重攻击。DRAM-Locker利用DRAM内交换和锁定表相结合的功能，防止攻击者挑出特定的DRAM行来保护DNN的重量参数。我们的结果表明，DRAM-Locker可以提供高级别的保护，将目标权重攻击的性能降低到随机攻击级别。此外，建议的防御机制在应用于CIFAR-10和CIFAR-100时不会降低精度。重要的是，DRAM-Locker不需要任何软件再培训或导致额外的硬件负担。



## **19. Amicable Aid: Perturbing Images to Improve Classification Performance**

Amicable Aid：扰动图像以提高分类性能 cs.CV

ICASSP 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2112.04720v4) [paper-pdf](http://arxiv.org/pdf/2112.04720v4)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstract**: While adversarial perturbation of images to attack deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of image perturbation can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be modified to yield higher classification confidence and even a misclassified image can be made correctly classified. This can be also achieved with a large amount of perturbation by which the image is made unrecognizable by human eyes. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. Furthermore, we investigate the universal amicable aid, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found.

摘要: 虽然在实践中对图像的敌意扰动攻击深度图像分类模型带来了严重的安全问题，但本文提出了一种新的范例，其中图像扰动的概念有助于提高分类性能，我们称之为友好辅助。我们证明，通过采取与扰动相反的搜索方向，可以对图像进行修改以产生更高的分类置信度，甚至可以对错误分类的图像进行正确分类。这也可以通过大量的扰动来实现，通过这种扰动，人眼无法识别图像。从潜在的自然意象流形的角度解释了友好相助的机制。此外，我们还研究了一种通用的友好辅助方法，即对多幅图像施加一个固定的扰动来改善它们的分类结果。虽然很难找到这样的扰动，但我们表明，通过用修改后的数据训练使决策边界尽可能垂直于图像流形，可以有效地获得一个更容易找到普遍友好扰动的模型。



## **20. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08793v1) [paper-pdf](http://arxiv.org/pdf/2312.08793v1)

**Authors**: Tony T. Wang, Miles Wang, Kaivu Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: 低收入国家经常面临相互竞争的压力(例如，有益与无害)。为了理解模型如何解决此类冲突，我们研究了关于禁止事实任务的Llama-2-Chat模型。具体地说，我们指示骆驼2号如实完成事实回忆声明，同时禁止它说出正确的答案。这经常使模型给出错误的答案。我们将Llama-2分解成1000多个成分，并根据它们对阻止正确答案的作用程度对每个成分进行排名。我们发现，总共大约35个组件就足以可靠地实现完全抑制行为。然而，这些组件具有相当大的异构性，许多组件使用错误的启发式方法进行操作。我们发现，其中一个启发式攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚州攻击。我们的结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站为https://forbiddenfacts.github.io。



## **21. AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection**

AVA：绕过DeepFake检测的基于不显眼属性变异的对抗性攻击 cs.CV

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08675v1) [paper-pdf](http://arxiv.org/pdf/2312.08675v1)

**Authors**: Xiangtao Meng, Li Wang, Shanqing Guo, Lei Ju, Qingchuan Zhao

**Abstract**: While DeepFake applications are becoming popular in recent years, their abuses pose a serious privacy threat. Unfortunately, most related detection algorithms to mitigate the abuse issues are inherently vulnerable to adversarial attacks because they are built atop DNN-based classification models, and the literature has demonstrated that they could be bypassed by introducing pixel-level perturbations. Though corresponding mitigation has been proposed, we have identified a new attribute-variation-based adversarial attack (AVA) that perturbs the latent space via a combination of Gaussian prior and semantic discriminator to bypass such mitigation. It perturbs the semantics in the attribute space of DeepFake images, which are inconspicuous to human beings (e.g., mouth open) but can result in substantial differences in DeepFake detection. We evaluate our proposed AVA attack on nine state-of-the-art DeepFake detection algorithms and applications. The empirical results demonstrate that AVA attack defeats the state-of-the-art black box attacks against DeepFake detectors and achieves more than a 95% success rate on two commercial DeepFake detectors. Moreover, our human study indicates that AVA-generated DeepFake images are often imperceptible to humans, which presents huge security and privacy concerns.

摘要: 虽然DeepFake应用程序近年来变得流行起来，但它们的滥用构成了严重的隐私威胁。不幸的是，大多数用于缓解滥用问题的相关检测算法天生就容易受到敌意攻击，因为它们建立在基于DNN的分类模型之上，并且文献证明可以通过引入像素级扰动来绕过它们。虽然已经提出了相应的缓解措施，但我们发现了一种新的基于属性变异的对抗攻击(AVA)，它通过结合高斯先验和语义鉴别器来扰动潜在空间，从而绕过了这种缓解措施。它扰乱了DeepFake图像的属性空间中的语义，这些图像对于人类来说是不明显的(例如，嘴巴张开)，但会导致DeepFake检测的显著差异。我们在九种最先进的DeepFake检测算法和应用上对我们提出的AVA攻击进行了评估。实验结果表明，AVA攻击击败了目前最先进的针对DeepFake探测器的黑盒攻击，并在两个商用DeepFake探测器上获得了95%以上的成功率。此外，我们的人体研究表明，AVA生成的DeepFake图像通常是人类无法察觉的，这带来了巨大的安全和隐私问题。



## **22. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的可解释的基于梯度的对抗性攻击 cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，防御这些攻击是可能的：对抗性攻击生成无限但不可读的胡言乱语提示，可通过基于困惑的过滤器检测；手动越狱攻击创建可读的提示，但由于人类创造力的必要性，其数量有限，允许轻松阻止。在本文中，我们证明了这些解决方案可能过于乐观。我们介绍了AutoDAN，一种可解释的、基于梯度的对抗性攻击，它融合了这两种攻击类型的优点。在越狱和可读性双重目标的指导下，AutoDAN从左到右一个接一个地优化和生成令牌，产生可读的提示，绕过困惑过滤器，同时保持高攻击成功率。值得注意的是，这些使用渐变从零开始生成的提示是可解释的和多样化的，新出现的策略通常出现在手动越狱攻击中。当使用有限的训练数据或单一代理模型时，它们还概括到不可预见的有害行为，并比不可读的同行更好地转移到黑盒LLM。此外，我们通过使用定制目标自动泄漏系统提示来展示AutoDAN的多功能性。我们的工作为红色团队LLM提供了一种新的方法，并通过可解释性来理解越狱机制。



## **23. Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks**

走向归纳鲁棒性：提取和培养抗图对抗攻击的传导GCN中的波诱导共振 cs.LG

AAAI 2024

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08651v1) [paper-pdf](http://arxiv.org/pdf/2312.08651v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou

**Abstract**: Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs.

摘要: 图神经网络(GNN)最近被证明容易受到敌意攻击，图结构中的微小扰动可能导致错误的预测。然而，目前用于防御此类攻击的健壮模型继承了图卷积网络(GCNS)的传导性限制。因此，它们受到固定结构的约束，不会自然地概括为不可见的节点。在这里，我们发现转导的GCNS天生就具有可蒸馏的健壮性，通过波诱导的共振过程实现。基于此，我们促进了这种共鸣，以促进归纳和稳健的学习。具体地说，我们首先证明了由GCN驱动的消息传递(MP)形成的信号等价于基于边缘的拉普拉斯波，其中，在一个波动系统中，信号与其传输介质之间可以自然地产生共振。这种共振提供了对对信号系统施加的恶意干扰的内在抵抗力。然后，我们证明了在GCNS内只需三次MP迭代就可以在节点和边之间产生信号共振，表现为节点与其可提取的局部子图之间的耦合。因此，我们提出了图共振培育网络(GRN)，通过从提取的共振子图中学习节点表示来促进这种共振。通过捕获该子图中的边传输信号并将其与节点信号集成，GRN将这些组合信号嵌入到中心节点的表示中。这种基于节点的嵌入方法允许对不可见节点进行泛化。我们通过实验验证了我们的理论发现，并证明了GRN对不可见节点的鲁棒性，同时保持了对扰动图的最新分类精度。



## **24. Guarding the Grid: Enhancing Resilience in Automated Residential Demand Response Against False Data Injection Attacks**

保护网格：增强住宅需求自动响应对虚假数据注入攻击的弹性 eess.SY

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08646v1) [paper-pdf](http://arxiv.org/pdf/2312.08646v1)

**Authors**: Thusitha Dayaratne, Carsten Rudolph, Ariel Liebman, Mahsa Salehi

**Abstract**: Utility companies are increasingly leveraging residential demand flexibility and the proliferation of smart/IoT devices to enhance the effectiveness of residential demand response (DR) programs through automated device scheduling. However, the adoption of distributed architectures in these systems exposes them to the risk of false data injection attacks (FDIAs), where adversaries can manipulate decision-making processes by injecting false data. Given the limited control utility companies have over these distributed systems and data, the need for reliable implementations to enhance the resilience of residential DR schemes against FDIAs is paramount. In this work, we present a comprehensive framework that combines DR optimisation, anomaly detection, and strategies for mitigating the impacts of attacks to create a resilient and automated device scheduling system. To validate the robustness of our framework against FDIAs, we performed an evaluation using real-world data sets, highlighting its effectiveness in securing residential DR systems.

摘要: 公用事业公司越来越多地利用住宅需求灵活性和智能/物联网设备的激增，通过自动化设备调度来增强住宅需求响应(DR)计划的有效性。然而，在这些系统中采用分布式体系结构使它们面临虚假数据注入攻击(FDIA)的风险，在FDIA中，对手可以通过注入虚假数据来操纵决策过程。鉴于公用事业公司对这些分布式系统和数据的控制有限，需要可靠的实施来增强住宅灾难恢复计划对FDIA的弹性。在这项工作中，我们提出了一个全面的框架，该框架结合了灾难恢复优化、异常检测和减轻攻击影响的策略，以创建一个弹性和自动化的设备调度系统。为了验证我们的框架针对FDIA的稳健性，我们使用真实世界的数据集进行了评估，突出了其在保护住宅DR系统方面的有效性。



## **25. Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification**

基于可扩展集成的说话人确认对抗攻击检测方法 eess.AS

Submitted to 2024 ICASSP

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08622v1) [paper-pdf](http://arxiv.org/pdf/2312.08622v1)

**Authors**: Haibin Wu, Heng-Cheng Kuo, Yu Tsao, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) is highly susceptible to adversarial attacks. Purification modules are usually adopted as a pre-processing to mitigate adversarial noise. However, they are commonly implemented across diverse experimental settings, rendering direct comparisons challenging. This paper comprehensively compares mainstream purification techniques in a unified framework. We find these methods often face a trade-off between user experience and security, as they struggle to simultaneously maintain genuine sample performance and reduce adversarial perturbations. To address this challenge, some efforts have extended purification modules to encompass detection capabilities, aiming to alleviate the trade-off. However, advanced purification modules will always come into the stage to surpass previous detection method. As a result, we further propose an easy-to-follow ensemble approach that integrates advanced purification modules for detection, achieving state-of-the-art (SOTA) performance in countering adversarial noise. Our ensemble method has great potential due to its compatibility with future advanced purification techniques.

摘要: 自动说话人验证(ASV)是一种易受敌意攻击的技术。为了减少对抗性噪声，通常采用净化模块作为预处理。然而，它们通常是在不同的实验环境中实施的，这使得直接比较具有挑战性。本文在统一的框架内对主流净化技术进行了综合比较。我们发现，这些方法经常面临用户体验和安全性之间的权衡，因为它们难以同时保持真正的样本性能和减少对抗性干扰。为了应对这一挑战，一些努力扩展了净化模块，以包含检测能力，旨在缓解权衡。然而，先进的净化模块总会出现，以超越以往的检测方法。因此，我们进一步提出了一种易于遵循的集成方法，该方法集成了用于检测的高级净化模块，在对抗对抗性噪声方面实现了最先进的性能(SOTA)。我们的集成方法具有很大的潜力，因为它与未来的先进纯化技术兼容。



## **26. Exploring the Privacy Risks of Adversarial VR Game Design**

对抗性VR游戏设计中的隐私风险探讨 cs.CR

Learn more at https://rdi.berkeley.edu/metaverse/metadata

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2207.13176v4) [paper-pdf](http://arxiv.org/pdf/2207.13176v4)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song, James F. O'Brien

**Abstract**: Fifty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Within just a few minutes, an adversarial program had accurately inferred over 25 of their personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. Since the Cambridge Analytica scandal of 2018, adversarially designed gamified elements have been known to constitute a significant privacy threat in conventional social platforms. In this work, we present a case study of how metaverse environments can similarly be adversarially constructed to covertly infer dozens of personal data attributes from seemingly anonymous users. While existing VR privacy research largely focuses on passive observation, we argue that because individuals subconsciously reveal personal information via their motion in response to specific stimuli, active attacks pose an outsized risk in VR environments.

摘要: 50名研究参与者在虚拟现实(VR)中玩了一个看起来很无辜的“逃生室”游戏。在短短几分钟内，一个对抗性程序就准确地推断出了25个以上的个人数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计学数据。随着以渴望数据著称的公司越来越多地参与到VR开发中来，这种实验场景可能很快就会代表一种典型的VR用户体验。自2018年剑桥分析丑闻以来，已知的是，恶意设计的游戏化元素在传统社交平台上构成了严重的隐私威胁。在这项工作中，我们提供了一个案例研究，即如何类似地以对抗性的方式构建虚拟世界环境，以秘密地从看似匿名的用户那里推断出数十个个人数据属性。虽然现有的虚拟现实隐私研究主要集中在被动观察上，但我们认为，由于个人潜意识地通过对特定刺激的动作来泄露个人信息，主动攻击在虚拟现实环境中构成了巨大的风险。



## **27. Defenses in Adversarial Machine Learning: A Survey**

对抗性机器学习中的防御机制：综述 cs.CV

21 pages, 5 figures, 2 tables, 237 reference papers

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08890v1) [paper-pdf](http://arxiv.org/pdf/2312.08890v1)

**Authors**: Baoyuan Wu, Shaokui Wei, Mingli Zhu, Meixi Zheng, Zihao Zhu, Mingda Zhang, Hongrui Chen, Danni Yuan, Li Liu, Qingshan Liu

**Abstract**: Adversarial phenomenon has been widely observed in machine learning (ML) systems, especially in those using deep neural networks, describing that ML systems may produce inconsistent and incomprehensible predictions with humans at some particular cases. This phenomenon poses a serious security threat to the practical application of ML systems, and several advanced attack paradigms have been developed to explore it, mainly including backdoor attacks, weight attacks, and adversarial examples. For each individual attack paradigm, various defense paradigms have been developed to improve the model robustness against the corresponding attack paradigm. However, due to the independence and diversity of these defense paradigms, it is difficult to examine the overall robustness of an ML system against different kinds of attacks.This survey aims to build a systematic review of all existing defense paradigms from a unified perspective. Specifically, from the life-cycle perspective, we factorize a complete machine learning system into five stages, including pre-training, training, post-training, deployment, and inference stages, respectively. Then, we present a clear taxonomy to categorize and review representative defense methods at each individual stage. The unified perspective and presented taxonomies not only facilitate the analysis of the mechanism of each defense paradigm but also help us to understand connections and differences among different defense paradigms, which may inspire future research to develop more advanced, comprehensive defenses.

摘要: 对抗性现象在机器学习系统中被广泛观察到，特别是在那些使用深度神经网络的系统中，描述了在某些特定情况下，机器学习系统可能会产生与人类不一致的、不可理解的预测。这一现象对ML系统的实际应用构成了严重的安全威胁，人们已经开发了几种先进的攻击范型来探索它，主要包括后门攻击、权重攻击和对抗性例子。对于每个单独的攻击范例，已经开发了各种防御范例来提高模型对相应攻击范例的稳健性。然而，由于这些防御范例的独立性和多样性，很难检验ML系统对不同类型攻击的整体稳健性，该调查旨在从统一的角度对所有现有的防御范例进行系统回顾。具体地说，从生命周期的角度，我们将一个完整的机器学习系统分解为五个阶段，分别包括训练前、训练、训练后、部署和推理阶段。然后，我们提出了一个明确的分类方法，对每个阶段的代表性防御方法进行分类和回顾。这种统一的视角和提出的分类不仅有助于分析每种防御范式的机制，而且有助于我们了解不同防御范式之间的联系和差异，这可能会启发未来的研究开发更先进、更全面的防御。



## **28. Universal Adversarial Framework to Improve Adversarial Robustness for Diabetic Retinopathy Detection**

提高糖尿病视网膜病变检测中对抗稳健性的通用对抗框架 eess.IV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08193v1) [paper-pdf](http://arxiv.org/pdf/2312.08193v1)

**Authors**: Samrat Mukherjee, Dibyanayan Bandyopadhyay, Baban Gain, Asif Ekbal

**Abstract**: Diabetic Retinopathy (DR) is a prevalent illness associated with Diabetes which, if left untreated, can result in irreversible blindness. Deep Learning based systems are gradually being introduced as automated support for clinical diagnosis. Since healthcare has always been an extremely important domain demanding error-free performance, any adversaries could pose a big threat to the applicability of such systems. In this work, we use Universal Adversarial Perturbations (UAPs) to quantify the vulnerability of Medical Deep Neural Networks (DNNs) for detecting DR. To the best of our knowledge, this is the very first attempt that works on attacking complete fine-grained classification of DR images using various UAPs. Also, as a part of this work, we use UAPs to fine-tune the trained models to defend against adversarial samples. We experiment on several models and observe that the performance of such models towards unseen adversarial attacks gets boosted on average by $3.41$ Cohen-kappa value and maximum by $31.92$ Cohen-kappa value. The performance degradation on normal data upon ensembling the fine-tuned models was found to be statistically insignificant using t-test, highlighting the benefits of UAP-based adversarial fine-tuning.

摘要: 糖尿病视网膜病变（DR）是一种与糖尿病相关的常见疾病，如果不及时治疗，可导致不可逆的失明。基于深度学习的系统正逐渐被引入作为临床诊断的自动化支持。由于医疗保健一直是一个要求无差错性能的极其重要的领域，任何对手都可能对此类系统的适用性构成巨大威胁。在这项工作中，我们使用通用对抗扰动（UAP）来量化医学深度神经网络（DNN）检测DR的脆弱性。据我们所知，这是第一次尝试使用各种UAP攻击DR图像的完整细粒度分类。此外，作为这项工作的一部分，我们使用UAP来微调训练模型，以抵御对抗性样本。我们在几个模型上进行了实验，观察到这些模型对不可见的对抗性攻击的性能平均提高了3.41美元的Cohen-kappa值，最大提高了31.92美元的Cohen-kappa值。使用t检验发现，在集成微调模型时，正常数据的性能下降在统计上是不显著的，突出了基于UAP的对抗性微调的好处。



## **29. Adversarial Attacks on Graph Neural Networks based Spatial Resource Management in P2P Wireless Communications**

P2P无线通信中基于图神经网络空间资源管理的对抗性攻击 eess.SP

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08181v1) [paper-pdf](http://arxiv.org/pdf/2312.08181v1)

**Authors**: Ahmad Ghasemi, Ehsan Zeraatkar, Majid Moradikia, Seyed, Zekavat

**Abstract**: This paper introduces adversarial attacks targeting a Graph Neural Network (GNN) based radio resource management system in point to point (P2P) communications. Our focus lies on perturbing the trained GNN model during the test phase, specifically targeting its vertices and edges. To achieve this, four distinct adversarial attacks are proposed, each accounting for different constraints, and aiming to manipulate the behavior of the system. The proposed adversarial attacks are formulated as optimization problems, aiming to minimize the system's communication quality. The efficacy of these attacks is investigated against the number of users, signal-to-noise ratio (SNR), and adversary power budget. Furthermore, we address the detection of such attacks from the perspective of the Central Processing Unit (CPU) of the system. To this end, we formulate an optimization problem that involves analyzing the distribution of channel eigenvalues before and after the attacks are applied. This formulation results in a Min-Max optimization problem, allowing us to detect the presence of attacks. Through extensive simulations, we observe that in the absence of adversarial attacks, the eigenvalues conform to Johnson's SU distribution. However, the attacks significantly alter the characteristics of the eigenvalue distribution, and in the most effective attack, they even change the type of the eigenvalue distribution.

摘要: 介绍了点对点(P2P)通信中基于图神经网络(GNN)的无线资源管理系统的对抗性攻击。我们的重点在于在测试阶段对训练好的GNN模型进行扰动，特别是针对其顶点和边。为了实现这一点，提出了四种不同的对抗性攻击，每一种攻击都考虑了不同的约束，旨在操纵系统的行为。所提出的对抗性攻击被描述为优化问题，目标是最小化系统的通信质量。这些攻击的有效性是根据用户数、信噪比(SNR)和对手功率预算进行调查的。此外，我们从系统的中央处理单元(CPU)的角度来解决此类攻击的检测问题。为此，我们提出了一个优化问题，包括分析攻击实施前后信道特征值的分布。这个公式导致了最小-最大优化问题，使我们能够检测攻击的存在。通过大量的仿真，我们观察到在没有对手攻击的情况下，特征值服从Johnson的SU分布。然而，攻击显著地改变了特征值分布的特征，在最有效的攻击中，它们甚至改变了特征值分布的类型。



## **30. Efficient Representation of the Activation Space in Deep Neural Networks**

深度神经网络中激活空间的有效表示 cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.

摘要: 深度神经网络(DNN)的激活空间表示被广泛用于自然语言处理、异常检测和语音识别等任务。由于这些任务的多样性和DNN的巨大规模，高效和独立于任务的激活表示变得至关重要。经验p值被用来量化与已知输入产生的激活相比，观察到的节点激活的相对强度。尽管如此，为这些计算保留原始数据会增加内存资源消耗，并引发隐私问题。为此，我们提出了一个与模型无关的框架，用于使用节点特定的直方图来创建DNN中的激活表示，以计算观察到的激活的p值，而不保留已知的输入。我们提出的方法在不同下游任务的多个网络架构上进行验证，并与核密度估计和蛮力经验基线进行比较，显示出良好的潜力。此外，该框架减少了30%的内存使用量，p值计算时间最多提高了4倍，同时在下游任务中保持了最先进的检测能力，例如检测对抗性攻击和合成内容。此外，由于我们不在推理时保留原始数据，因此我们可能会降低对攻击和隐私问题的易感性。



## **31. Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification**

基于边界判别和关联提纯的健壮少镜头命名实体识别 cs.CL

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07961v1) [paper-pdf](http://arxiv.org/pdf/2312.07961v1)

**Authors**: Xiaojun Xue, Chunxia Zhang, Tianxiang Xu, Zhendong Niu

**Abstract**: Few-shot named entity recognition (NER) aims to recognize novel named entities in low-resource domains utilizing existing knowledge. However, the present few-shot NER models assume that the labeled data are all clean without noise or outliers, and there are few works focusing on the robustness of the cross-domain transfer learning ability to textual adversarial attacks in Few-shot NER. In this work, we comprehensively explore and assess the robustness of few-shot NER models under textual adversarial attack scenario, and found the vulnerability of existing few-shot NER models. Furthermore, we propose a robust two-stage few-shot NER method with Boundary Discrimination and Correlation Purification (BDCP). Specifically, in the span detection stage, the entity boundary discriminative module is introduced to provide a highly distinguishing boundary representation space to detect entity spans. In the entity typing stage, the correlations between entities and contexts are purified by minimizing the interference information and facilitating correlation generalization to alleviate the perturbations caused by textual adversarial attacks. In addition, we construct adversarial examples for few-shot NER based on public datasets Few-NERD and Cross-Dataset. Comprehensive evaluations on those two groups of few-shot NER datasets containing adversarial examples demonstrate the robustness and superiority of the proposed method.

摘要: 少镜头命名实体识别(NER)旨在利用现有知识识别低资源领域中的新命名实体。然而，目前的少射NER模型都假设标记的数据都是干净的，没有噪声或离群点，而很少有人关注在少射NER中跨域迁移学习能力对文本攻击的健壮性。在这项工作中，我们全面探索和评估了文本对抗攻击场景下的少镜头NER模型的健壮性，发现了现有的少镜头NER模型的脆弱性。此外，我们还提出了一种具有边界识别和相关净化(BDCP)的健壮两阶段少镜头NER方法。具体来说，在跨度检测阶段，引入了实体边界判别模块，为检测实体跨度提供了一个高分辨率的边界表示空间。在实体分类阶段，通过最小化干扰信息和促进关联泛化来净化实体与上下文之间的相关性，以缓解文本对抗性攻击造成的扰动。此外，我们还在公开的数据集Low-Nerd和Cross-DataSet上构建了对抗性的例子。对这两组包含对抗性实例的少镜头NER数据集的综合评价表明了该方法的稳健性和优越性。



## **32. DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space**

DifAttack：基于解缠特征空间的查询高效黑盒攻击 cs.CV

Accepted in AAAI'24

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2309.14585v3) [paper-pdf](http://arxiv.org/pdf/2309.14585v3)

**Authors**: Liu Jun, Zhou Jiantao, Zeng Jiandian, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a Disentangled Feature space, called DifAttack, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack firstly disentangles an image's latent feature into an adversarial feature and a visual feature, where the former dominates the adversarial capability of an image, while the latter largely determines its visual appearance. We train an autoencoder for the disentanglement by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, DifAttack iteratively optimizes the adversarial feature according to the query feedback from the victim model until a successful AE is generated, while keeping the visual feature unaltered. In addition, due to the avoidance of using surrogate models' gradient information when optimizing AEs for black-box models, our proposed DifAttack inherently possesses better attack capability in the open-set scenario, where the training dataset of the victim model is unknown. Extensive experimental results demonstrate that our method achieves significant improvements in ASR and query efficiency simultaneously, especially in the targeted attack and open-set scenarios. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(ASR)和良好的泛化能力。我们设计了一种新的基于解缠特征空间的攻击方法DifAttack，它与现有的操作在整个特征空间上的攻击方法有很大的不同。具体地说，DifAttack首先将图像的潜在特征分解为对抗性特征和视觉特征，其中前者主导图像的对抗性能力，而后者在很大程度上决定了图像的视觉外观。我们通过白盒攻击方法，使用已有的代理模型生成的干净图像对和它们的对抗性实例(AE)来训练自动编码器来进行解缠。最后，DifAttack根据受害者模型的查询反馈迭代地优化对抗性特征，直到生成成功的AE，同时保持视觉特征不变。此外，由于在优化黑盒模型的AES时避免了使用代理模型的梯度信息，因此在受害者模型的训练数据集未知的开集场景下，我们提出的DifAttack具有更好的攻击能力。大量的实验结果表明，该方法在ASR和查询效率上都有显著的提高，特别是在目标攻击和开集场景下。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **33. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **34. Causality Analysis for Evaluating the Security of Large Language Models**

大型语言模型安全性评估的因果分析 cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

摘要: 大型语言模型(LLM)，如GPT和Llama2，在许多安全关键型应用中越来越多地被采用。因此，他们的安全至关重要。即使在从人类反馈中强化学习(RLHF)方面花费了大量的努力，最近的研究表明LLMS仍然受到诸如对抗性扰动和特洛伊木马攻击的攻击。因此，需要进一步研究，以评估其安全性和/或了解其缺乏安全性。在这项工作中，我们提出了一个框架，用于在标记、层和神经元水平上进行LLMS的轻量级因果分析。我们将我们的框架应用于开源LLM，如Llama2和Vicuna，并有多个有趣的发现。基于层级因果关系分析，我们发现RLHF具有对有害提示的模型过度拟合的效果。这意味着这种安全很容易被“不寻常的”有害提示所克服。作为证据，我们提出了一种对抗性扰动方法，在2023年木马检测大赛的红队任务上达到了100%的攻击成功率。此外，我们证明了在Llama2和Vicuna2中都存在一个神秘的神经元，它对输出具有不合理的高因果效应。虽然我们不确定为什么会有这样的神经元存在，但我们证明了有可能进行针对该特定神经元的“特洛伊木马”攻击，以完全削弱LLM，即我们可以为提示生成可转移的后缀，这些后缀经常使LLM产生无意义的响应。



## **35. Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based Integrity Verification**

图神经网络在MLaaS中的安全：基于查询完整性验证的综合实现 cs.CR

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07870v1) [paper-pdf](http://arxiv.org/pdf/2312.07870v1)

**Authors**: Bang Wu, Xingliang Yuan, Shuo Wang, Qi Li, Minhui Xue, Shirui Pan

**Abstract**: The deployment of Graph Neural Networks (GNNs) within Machine Learning as a Service (MLaaS) has opened up new attack surfaces and an escalation in security concerns regarding model-centric attacks. These attacks can directly manipulate the GNN model parameters during serving, causing incorrect predictions and posing substantial threats to essential GNN applications. Traditional integrity verification methods falter in this context due to the limitations imposed by MLaaS and the distinct characteristics of GNN models.   In this research, we introduce a groundbreaking approach to protect GNN models in MLaaS from model-centric attacks. Our approach includes a comprehensive verification schema for GNN's integrity, taking into account both transductive and inductive GNNs, and accommodating varying pre-deployment knowledge of the models. We propose a query-based verification technique, fortified with innovative node fingerprint generation algorithms. To deal with advanced attackers who know our mechanisms in advance, we introduce randomized fingerprint nodes within our design. The experimental evaluation demonstrates that our method can detect five representative adversarial model-centric attacks, displaying 2 to 4 times greater efficiency compared to baselines.

摘要: 图神经网络(GNN)在机器学习即服务(MLaaS)中的部署开辟了新的攻击面，并加剧了对以模型为中心的攻击的安全担忧。这些攻击可以在服务期间直接操纵GNN模型参数，导致错误的预测，并对必要的GNN应用构成实质性威胁。在这种情况下，由于MLaaS的限制和GNN模型的独特特性，传统的完整性验证方法步履蹒跚。在这项研究中，我们介绍了一种突破性的方法来保护MLaaS中的GNN模型免受以模型为中心的攻击。我们的方法包括对GNN完整性的全面验证方案，同时考虑了传导性和感应性GNN，并容纳了不同的模型部署前知识。我们提出了一种基于查询的验证技术，并采用了创新的节点指纹生成算法。为了应对提前知道我们的机制的高级攻击者，我们在设计中引入了随机指纹节点。实验评估表明，我们的方法可以检测到五种典型的以模型为中心的对抗性攻击，与基线相比，效率提高了2到4倍。



## **36. SimAC: A Simple Anti-Customization Method against Text-to-Image Synthesis of Diffusion Models**

SIMAC：一种针对扩散模型图文合成的简单反定制方法 cs.CV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07865v1) [paper-pdf](http://arxiv.org/pdf/2312.07865v1)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps. In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization. Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby enhancing user privacy and security.

摘要: 尽管基于扩散的定制方法在视觉内容创作上取得了成功，但从隐私和政治的角度来看，人们对这种技术的关注越来越多。为了解决这个问题，最近几个月提出了几种反定制方法，主要基于对抗性攻击。遗憾的是，这些方法大多采用简单的设计，例如端到端优化，关注的是相反地最大化原始训练损失，从而忽略了扩散模型固有的微妙的内在属性，甚至导致在一些扩散时间步长内的无效优化。在本文中，我们努力通过对这些固有属性的全面探索来弥合这一差距，以提高当前反定制方法的性能。研究了两个方面的性质：1)在图像的频域中，我们考察了时间步长选择与模型感知之间的关系，发现时间步长越低，对对抗性噪声的贡献越大。这启发了我们提出了一种自适应贪婪搜索来寻找最优时间步长，并与现有的反定制方法无缝集成。2)我们仔细研究了不同层次的特征在去噪过程中的作用，并设计了一个复杂的基于特征的反定制优化框架。在面部基准上的实验表明，我们的方法显著增加了身份破坏，从而增强了用户隐私和安全性。



## **37. Radio Signal Classification by Adversarially Robust Quantum Machine Learning**

逆稳健量子机器学习在无线电信号分类中的应用 quant-ph

12 pages, 6 figures

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07821v1) [paper-pdf](http://arxiv.org/pdf/2312.07821v1)

**Authors**: Yanqiu Wu, Eromanga Adermann, Chandra Thapa, Seyit Camtepe, Hajime Suzuki, Muhammad Usman

**Abstract**: Radio signal classification plays a pivotal role in identifying the modulation scheme used in received radio signals, which is essential for demodulation and proper interpretation of the transmitted information. Researchers have underscored the high susceptibility of ML algorithms for radio signal classification to adversarial attacks. Such vulnerability could result in severe consequences, including misinterpretation of critical messages, interception of classified information, or disruption of communication channels. Recent advancements in quantum computing have revolutionized theories and implementations of computation, bringing the unprecedented development of Quantum Machine Learning (QML). It is shown that quantum variational classifiers (QVCs) provide notably enhanced robustness against classical adversarial attacks in image classification. However, no research has yet explored whether QML can similarly mitigate adversarial threats in the context of radio signal classification. This work applies QVCs to radio signal classification and studies their robustness to various adversarial attacks. We also propose the novel application of the approximate amplitude encoding (AAE) technique to encode radio signal data efficiently. Our extensive simulation results present that attacks generated on QVCs transfer well to CNN models, indicating that these adversarial examples can fool neural networks that they are not explicitly designed to attack. However, the converse is not true. QVCs primarily resist the attacks generated on CNNs. Overall, with comprehensive simulations, our results shed new light on the growing field of QML by bridging knowledge gaps in QAML in radio signal classification and uncovering the advantages of applying QML methods in practical applications.

摘要: 无线电信号分类在识别接收无线电信号中使用的调制方案方面起着关键作用，这对于解调和正确解释传输的信息是必不可少的。研究人员强调了用于无线电信号分类的ML算法对敌方攻击的高度敏感性。此类漏洞可能导致严重后果，包括误解关键消息、截取机密信息或中断通信渠道。量子计算的最新进展使计算的理论和实现发生了革命性的变化，带来了量子机器学习(QML)前所未有的发展。结果表明，量子变分分类器(QVC)在图像分类中对经典的敌意攻击具有显著的鲁棒性。然而，还没有研究探索QML是否可以类似地在无线电信号分类的背景下减轻对抗性威胁。本文将QVC应用于无线电信号分类，研究了QVC对各种攻击的稳健性。我们还提出了近似幅度编码(AAE)技术在无线电信号数据编码中的新应用。我们广泛的模拟结果表明，对QVC产生的攻击很好地转移到了CNN模型中，表明这些敌对的例子可以欺骗不是明确设计来攻击的神经网络。然而，相反的情况并非如此。QVC主要抵抗CNN上产生的攻击。总体而言，通过全面的仿真，我们的结果通过弥合无线电信号分类中QAML的知识差距，揭示了在实际应用中应用QML方法的优势，从而为QML不断发展的领域提供了新的线索。



## **38. BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs**

BarraCUDA：利用电磁侧通道从NVIDIA GPU窃取神经网络的权重 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07783v1) [paper-pdf](http://arxiv.org/pdf/2312.07783v1)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks have spread to cover all aspects of life. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and autonomous cars. They are being used in safety and security-critical applications like high definition maps and medical wristbands, or in globally used products like Google Translate and ChatGPT. Much of the intellectual property underpinning these products is encoded in the exact configuration of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product.   Past work has demonstrated that with physical access, attackers can reverse engineer neural networks that run on scalar microcontrollers, like ARM Cortex M3. However, for performance reasons, neural networks are often implemented on highly-parallel general purpose graphics processing units (GPGPUs), and so far, attacks on these have only recovered course-grained information on the structure of the neural network, but failed to retrieve the weights and biases.   In this work, we present BarraCUDA, a novel attack on GPGPUs that can completely extract the parameters of neural networks. BarraCUDA uses correlation electromagnetic analysis to recover the weights and biases in the convolutional layers of neural networks. We use BarraCUDA to attack the popular NVIDIA Jetson Nano device, demonstrating successful parameter extraction of neural networks in a highly parallel and noisy environment.

摘要: 在过去的十年里，神经网络的应用已经扩展到生活的方方面面。许多公司的业务基础是开发使用神经网络执行人脸识别、机器翻译和自动驾驶汽车等任务的产品。它们正被用于高清晰度地图和医疗腕带等安全和安保关键应用程序，或谷歌翻译和ChatGPT等全球使用的产品。支撑这些产品的大部分知识产权都编码在神经网络的准确配置中。因此，保护这些信息对企业来说是最重要的。同时，这些产品中的许多都需要在强大的威胁模式下运行，在这种模式下，对手可以不受约束地对产品进行物理控制。过去的研究表明，通过物理访问，攻击者可以对在ARM Cortex M3等标量微控制器上运行的神经网络进行反向工程。然而，由于性能原因，神经网络通常是在高度并行的通用图形处理单元(GPGPU)上实现的，到目前为止，对这些单元的攻击只恢复了关于神经网络结构的过程粒度信息，但无法恢复权重和偏差。在这项工作中，我们提出了一种新的针对GPGPU的攻击Barracuda，它可以完全提取神经网络的参数。梭鱼使用相关电磁分析来恢复神经网络卷积层中的权重和偏差。我们使用梭鱼攻击流行的NVIDIA Jetson Nano设备，演示了在高度并行和噪声环境中成功提取神经网络的参数。



## **39. Majority is Not Required: A Rational Analysis of the Private Double-Spend Attack from a Sub-Majority Adversary**

不需要多数：对一个次多数对手的私人双重支出攻击的理性分析 cs.GT

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07709v1) [paper-pdf](http://arxiv.org/pdf/2312.07709v1)

**Authors**: Yanni Georghiades, Rajesh Mishra, Karl Kreder, Sriram Vishwanath

**Abstract**: We study the incentives behind double-spend attacks on Nakamoto-style Proof-of-Work cryptocurrencies. In these systems, miners are allowed to choose which transactions to reference with their block, and a common strategy for selecting transactions is to simply choose those with the highest fees. This can be problematic if these transactions originate from an adversary with substantial (but less than 50\%) computational power, as high-value transactions can present an incentive for a rational adversary to attempt a double-spend attack if they expect to profit. The most common mechanism for deterring double-spend attacks is for the recipients of large transactions to wait for additional block confirmations (i.e., to increase the attack cost). We argue that this defense mechanism is not satisfactory, as the security of the system is contingent on the actions of its users. Instead, we propose that defending against double-spend attacks should be the responsibility of the miners; specifically, miners should limit the amount of transaction value they include in a block (i.e., reduce the attack reward). To this end, we model cryptocurrency mining as a mean-field game in which we augment the standard mining reward function to simulate the presence of a rational, double-spending adversary. We design and implement an algorithm which characterizes the behavior of miners at equilibrium, and we show that miners who use the adversary-aware reward function accumulate more wealth than those who do not. We show that the optimal strategy for honest miners is to limit the amount of value transferred by each block such that the adversary's expected profit is 0. Additionally, we examine Bitcoin's resilience to double-spend attacks. Assuming a 6 block confirmation time, we find that an attacker with at least 25% of the network mining power can expect to profit from a double-spend attack.

摘要: 我们研究了对Nakamoto风格的工作量证明加密货币进行双重花费攻击背后的动机。在这些系统中，矿工可以选择哪些交易与他们的区块相关联，选择交易的常见策略是简单地选择那些费用最高的交易。如果这些交易来自具有大量（但小于50%）计算能力的对手，这可能是有问题的，因为高价值交易可能会激励理性的对手尝试双重花费攻击，如果他们期望获利的话。阻止双重花费攻击的最常见机制是让大型交易的接收者等待额外的块确认（即，增加攻击成本）。我们认为，这种防御机制是不令人满意的，因为系统的安全性是视用户的行为。相反，我们建议防御双重花费攻击应该是矿工的责任;具体来说，矿工应该限制他们在区块中包含的交易价值的数量（即，减少攻击奖励）。为此，我们将加密货币挖掘建模为平均场游戏，在该游戏中，我们增加了标准的挖掘奖励函数，以模拟理性的双重支出对手的存在。我们设计并实现了一个算法，该算法描述了矿工在平衡状态下的行为，并且我们表明，使用对手意识奖励函数的矿工比不使用的矿工积累了更多的财富。我们表明，诚实矿工的最佳策略是限制每个区块转移的价值量，使对手的预期利润为0。此外，我们还研究了比特币对双重支出攻击的弹性。假设一个6块的确认时间，我们发现一个攻击者至少有25%的网络挖掘能力可以期望从双重花费攻击中获利。



## **40. Defending Our Privacy With Backdoors**

使用后门保护我们的隐私 cs.LG

14 pages, 10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2310.08320v2) [paper-pdf](http://arxiv.org/pdf/2310.08320v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的简单而有效的防御方法，将个人姓名等私人信息从模型中移除，并将重点放在文本编码器上。具体地说，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语--“人”而不是人的名字--保持一致。我们的实验结果证明了我们的基于后门的防御在CLIP上的有效性，通过使用专门的针对零镜头分类器的隐私攻击来评估其性能。我们的方法不仅为后门攻击提供了一种新的“两用”视角，而且还提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **41. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **42. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

ReRoGCRL：目标条件强化学习中基于表示的稳健性 cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07392v1) [paper-pdf](http://arxiv.org/pdf/2312.07392v1)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness, particularly against adversarial perturbations, remains unexplored. Unfortunately, the attacks and robust representation training methods specifically designed for traditional RL are not so effective when applied to GCRL. To address this challenge, we propose the \textit{Semi-Contrastive Representation} attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Furthermore, to mitigate the vulnerability of existing GCRL algorithms, we introduce \textit{Adversarial Representation Tactics}. This strategy combines \textit{Semi-Contrastive Adversarial Augmentation} with \textit{Sensitivity-Aware Regularizer}. It improves the adversarial robustness of the underlying agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence mechanism across multiple state-of-the-art GCRL algorithms. Our tool {\bf ReRoGCRL} is available at \url{https://github.com/TrustAI/ReRoGCRL}.

摘要: 虽然目标条件强化学习(GCRL)已经引起了人们的关注，但它的算法健壮性，特别是对对抗性扰动的鲁棒性，仍然没有得到探索。不幸的是，专门针对传统RL设计的攻击和稳健表示训练方法在应用于GCRL时并不是很有效。为了应对这一挑战，我们提出了半对比表示攻击，这是一种受对抗性对比攻击启发的新方法。与RL中现有的攻击不同，它只需要来自策略功能的信息，并且可以在部署期间无缝实施。此外，为了缓解现有GCRL算法的脆弱性，我们引入了对抗性表示策略。该策略结合了半对比式对抗性增强和敏感度感知调节器。它提高了底层代理对各种类型扰动的对抗健壮性。广泛的实验验证了我们的攻击和防御机制在多种最先进的GCRL算法上的卓越性能。我们的工具{\bf ReRoGCRL}位于\url{https://github.com/TrustAI/ReRoGCRL}.



## **43. Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems**

航空图像中信任的侵蚀：地理空间系统中对抗性攻击的综合分析和评估 cs.CV

Accepted at IEEE AIRP 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07389v1) [paper-pdf](http://arxiv.org/pdf/2312.07389v1)

**Authors**: Michael Lanier, Aayush Dhakal, Zhexiao Xiong, Arthur Li, Nathan Jacobs, Yevgeniy Vorobeychik

**Abstract**: In critical operations where aerial imagery plays an essential role, the integrity and trustworthiness of data are paramount. The emergence of adversarial attacks, particularly those that exploit control over labels or employ physically feasible trojans, threatens to erode that trust, making the analysis and mitigation of these attacks a matter of urgency. We demonstrate how adversarial attacks can degrade confidence in geospatial systems, specifically focusing on scenarios where the attacker's control over labels is restricted and the use of realistic threat vectors. Proposing and evaluating several innovative attack methodologies, including those tailored to overhead images, we empirically show their threat to remote sensing systems using high-quality SpaceNet datasets. Our experimentation reflects the unique challenges posed by aerial imagery, and these preliminary results not only reveal the potential risks but also highlight the non-trivial nature of the problem compared to recent works.

摘要: 在航空图像发挥重要作用的关键行动中，数据的完整性和可信度是最重要的。敌意攻击的出现，特别是那些利用对标签的控制或使用物理上可行的特洛伊木马的攻击，有可能侵蚀这种信任，使分析和缓解这些攻击成为当务之急。我们演示了敌意攻击如何降低对地理空间系统的信心，特别是在攻击者对标签的控制受到限制的情况下，以及使用现实的威胁向量。提出并评估了几种创新的攻击方法，包括那些针对高空图像量身定做的攻击方法，我们使用高质量的Spacenet数据集经验地展示了它们对遥感系统的威胁。我们的实验反映了航空图像带来的独特挑战，这些初步结果不仅揭示了潜在的风险，而且与最近的工作相比，也突出了问题的非同小可的性质。



## **44. SSTA: Salient Spatially Transformed Attack**

SSTA：显著的空间变换攻击 cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07258v1) [paper-pdf](http://arxiv.org/pdf/2312.07258v1)

**Authors**: Renyang Liu, Wei Zhou, Sixin Wu, Jun Zhao, Kwok-Yan Lam

**Abstract**: Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial attacks, which brings a huge security risk to the further application of DNNs, especially for the AI models developed in the real world. Despite the significant progress that has been made recently, existing attack methods still suffer from the unsatisfactory performance of escaping from being detected by naked human eyes due to the formulation of adversarial example (AE) heavily relying on a noise-adding manner. Such mentioned challenges will significantly increase the risk of exposure and result in an attack to be failed. Therefore, in this paper, we propose the Salient Spatially Transformed Attack (SSTA), a novel framework to craft imperceptible AEs, which enhance the stealthiness of AEs by estimating a smooth spatial transform metric on a most critical area to generate AEs instead of adding external noise to the whole image. Compared to state-of-the-art baselines, extensive experiments indicated that SSTA could effectively improve the imperceptibility of the AEs while maintaining a 100\% attack success rate.

摘要: 大量的研究表明，深度神经网络（DNN）容易受到对抗性攻击，这给DNN的进一步应用带来了巨大的安全风险，特别是对于现实世界中开发的AI模型。尽管最近已经取得了显着的进展，现有的攻击方法仍然遭受逃避被检测到的裸眼由于制定对抗性的例子（AE）严重依赖于噪声添加方式的性能不令人满意。上述挑战将显著增加暴露的风险，并导致攻击失败。因此，在本文中，我们提出了显着的空间变换攻击（SSTA），一种新的框架来制作不可感知的AE，它通过在最关键的区域上估计平滑的空间变换度量来生成AE，而不是向整个图像添加外部噪声，从而增强了AE的隐蔽性。实验结果表明，SSTA算法在保持100%攻击成功率的同时，有效地提高了AE的不可感知性。



## **45. DTA: Distribution Transform-based Attack for Query-Limited Scenario**

DTA：查询受限场景下基于分布变换的攻击 cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07245v1) [paper-pdf](http://arxiv.org/pdf/2312.07245v1)

**Authors**: Renyang Liu, Wei Zhou, Xin Jin, Song Gao, Yuanyu Wang, Ruxin Wang

**Abstract**: In generating adversarial examples, the conventional black-box attack methods rely on sufficient feedback from the to-be-attacked models by repeatedly querying until the attack is successful, which usually results in thousands of trials during an attack. This may be unacceptable in real applications since Machine Learning as a Service Platform (MLaaS) usually only returns the final result (i.e., hard-label) to the client and a system equipped with certain defense mechanisms could easily detect malicious queries. By contrast, a feasible way is a hard-label attack that simulates an attacked action being permitted to conduct a limited number of queries. To implement this idea, in this paper, we bypass the dependency on the to-be-attacked model and benefit from the characteristics of the distributions of adversarial examples to reformulate the attack problem in a distribution transform manner and propose a distribution transform-based attack (DTA). DTA builds a statistical mapping from the benign example to its adversarial counterparts by tackling the conditional likelihood under the hard-label black-box settings. In this way, it is no longer necessary to query the target model frequently. A well-trained DTA model can directly and efficiently generate a batch of adversarial examples for a certain input, which can be used to attack un-seen models based on the assumed transferability. Furthermore, we surprisingly find that the well-trained DTA model is not sensitive to the semantic spaces of the training dataset, meaning that the model yields acceptable attack performance on other datasets. Extensive experiments validate the effectiveness of the proposed idea and the superiority of DTA over the state-of-the-art.

摘要: 在生成对抗性实例时，传统的黑盒攻击方法依赖于被攻击模型的充分反馈，通过反复查询直到攻击成功，这通常导致在一次攻击中进行数千次尝试。这在实际应用中可能是不可接受的，因为机器学习作为服务平台(MLaaS)通常只向客户端返回最终结果(即硬标签)，并且配备了某些防御机制的系统可以很容易地检测到恶意查询。相比之下，一种可行的方法是硬标签攻击，它模拟允许执行有限数量的查询的攻击操作。为了实现这一思想，本文绕过了对待攻击模型的依赖，利用对抗性实例分布的特点，用分布变换的方式重新描述攻击问题，提出了一种基于分布变换的攻击(DTA)。DTA通过处理硬标签黑盒设置下的条件似然，建立了从良性例子到对抗性例子的统计映射。这样，就不再需要频繁地查询目标模型。一个训练有素的DTA模型可以直接有效地为某一输入生成一批对抗性的例子，这些例子可以用来攻击基于假设的可转移性的不可见模型。此外，我们惊讶地发现，经过良好训练的DTA模型对训练数据集的语义空间不敏感，这意味着该模型在其他数据集上的攻击性能是可以接受的。大量的实验验证了所提出的思想的有效性以及DTA相对于最先进技术的优越性。



## **46. Reward Certification for Policy Smoothed Reinforcement Learning**

策略平滑强化学习的奖励认证 cs.LG

This paper will be presented in AAAI2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06436v2) [paper-pdf](http://arxiv.org/pdf/2312.06436v2)

**Authors**: Ronghui Mu, Leandro Soriano Marcolino, Tianle Zhang, Yanghao Zhang, Xiaowei Huang, Wenjie Ruan

**Abstract**: Reinforcement Learning (RL) has achieved remarkable success in safety-critical areas, but it can be weakened by adversarial attacks. Recent studies have introduced "smoothed policies" in order to enhance its robustness. Yet, it is still challenging to establish a provable guarantee to certify the bound of its total reward. Prior methods relied primarily on computing bounds using Lipschitz continuity or calculating the probability of cumulative reward above specific thresholds. However, these techniques are only suited for continuous perturbations on the RL agent's observations and are restricted to perturbations bounded by the $l_2$-norm. To address these limitations, this paper proposes a general black-box certification method capable of directly certifying the cumulative reward of the smoothed policy under various $l_p$-norm bounded perturbations. Furthermore, we extend our methodology to certify perturbations on action spaces. Our approach leverages f-divergence to measure the distinction between the original distribution and the perturbed distribution, subsequently determining the certification bound by solving a convex optimisation problem. We provide a comprehensive theoretical analysis and run sufficient experiments in multiple environments. Our results show that our method not only improves the certified lower bound of mean cumulative reward but also demonstrates better efficiency than state-of-the-art techniques.

摘要: 强化学习(RL)在安全关键领域取得了显著的成功，但它可能会被对手攻击所削弱。最近的研究引入了“平滑政策”，以增强其稳健性。然而，建立一个可证明的保证来证明其总回报的界限仍然是具有挑战性的。以前的方法主要依赖于使用Lipschitz连续性来计算界限，或者计算超过特定阈值的累积奖励的概率。然而，这些技术只适用于对RL代理观测的连续扰动，并且限于$L_2$-范数的扰动。针对这些局限性，本文提出了一种通用的黑盒证明方法，该方法能够直接证明平滑策略在各种$L_p$-范数有界扰动下的累积报酬。此外，我们将我们的方法扩展到证明行动空间上的扰动。我们的方法利用f-散度来度量原始分布和扰动分布之间的区别，然后通过求解一个凸优化问题来确定认证界。我们提供了全面的理论分析，并在多个环境下进行了大量的实验。我们的结果表明，我们的方法不仅改善了平均累积奖励的证明下界，而且比最新的技术表现出更好的效率。



## **47. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击端到端自动驾驶 cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2103.09151v8) [paper-pdf](http://arxiv.org/pdf/2103.09151v8)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的进展，深度卷积网络在自动驾驶任务中变得很有前途。特别是，有一个新兴的趋势，采用端到端的神经网络模型的自动驾驶。然而，之前的研究表明，深度神经网络分类器容易受到对抗性攻击。而对于回归任务，对抗性攻击的影响还没有得到很好的理解。在这项研究中，我们设计了两种针对端到端自动驾驶模型的白盒攻击。我们的攻击通过扰动输入图像来操纵自动驾驶系统的行为。在平均800次具有相同攻击强度（λ =1）的攻击中，图像特定攻击和图像不可知攻击分别使转向角偏离原始输出0.478和0.111，这比仅使转向角扰动0.002的随机噪声强得多（转向角范围为[-1，1]）。这两种攻击都可以在CPU上实时发起，而无需使用GPU。演示视频：https://youtu.be/I0i8uN2oOP0。



## **48. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2209.01962v6) [paper-pdf](http://arxiv.org/pdf/2209.01962v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内实现了约90%的成功率。该演示视频可在https://youtu.be/zJZ1aNlXsMU.上查看



## **49. Cost Aware Untargeted Poisoning Attack against Graph Neural Networks,**

针对图神经网络的成本意识非目标中毒攻击， cs.AI

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07158v1) [paper-pdf](http://arxiv.org/pdf/2312.07158v1)

**Authors**: Yuwei Han, Yuni Lai, Yulin Zhu, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have become widely used in the field of graph mining. However, these networks are vulnerable to structural perturbations. While many research efforts have focused on analyzing vulnerability through poisoning attacks, we have identified an inefficiency in current attack losses. These losses steer the attack strategy towards modifying edges targeting misclassified nodes or resilient nodes, resulting in a waste of structural adversarial perturbation. To address this issue, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins. Our experiments demonstrate that the proposed CA-attack significantly enhances existing attack strategies

摘要: 图神经网络在图挖掘领域得到了广泛的应用。然而，这些网络很容易受到结构扰动的影响。虽然许多研究工作都集中在通过中毒攻击来分析脆弱性上，但我们已经发现了当前攻击损失的低效。这些损失使攻击策略倾向于修改针对错误分类节点或弹性节点的边，从而浪费了结构上的对抗性扰动。针对这一问题，我们提出了一种新的攻击损失框架，称为代价感知中毒攻击(CA-Attack)，通过动态考虑节点的分类裕度来提高攻击预算的分配。具体地说，它优先考虑正边距较小的节点，而推迟边距为负值的节点。实验表明，所提出的CA-攻击显著增强了现有的攻击策略



## **50. Data-Free Hard-Label Robustness Stealing Attack**

无数据硬标签健壮性窃取攻击 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.05924v2) [paper-pdf](http://arxiv.org/pdf/2312.05924v2)

**Authors**: Xiaojian Yuan, Kejiang Chen, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: The popularity of Machine Learning as a Service (MLaaS) has led to increased concerns about Model Stealing Attacks (MSA), which aim to craft a clone model by querying MLaaS. Currently, most research on MSA assumes that MLaaS can provide soft labels and that the attacker has a proxy dataset with a similar distribution. However, this fails to encapsulate the more practical scenario where only hard labels are returned by MLaaS and the data distribution remains elusive. Furthermore, most existing work focuses solely on stealing the model accuracy, neglecting the model robustness, while robustness is essential in security-sensitive scenarios, e.g., face-scan payment. Notably, improving model robustness often necessitates the use of expensive techniques such as adversarial training, thereby further making stealing robustness a more lucrative prospect. In response to these identified gaps, we introduce a novel Data-Free Hard-Label Robustness Stealing (DFHL-RS) attack in this paper, which enables the stealing of both model accuracy and robustness by simply querying hard labels of the target model without the help of any natural data. Comprehensive experiments demonstrate the effectiveness of our method. The clone model achieves a clean accuracy of 77.86% and a robust accuracy of 39.51% against AutoAttack, which are only 4.71% and 8.40% lower than the target model on the CIFAR-10 dataset, significantly exceeding the baselines. Our code is available at: https://github.com/LetheSec/DFHL-RS-Attack.

摘要: 机器学习即服务(MLaaS)的流行引起了人们对模型窃取攻击(MSA)的越来越多的关注，MSA旨在通过查询MLaaS来创建克隆模型。目前，大多数关于MSA的研究都假设MLaaS可以提供软标签，并且攻击者拥有一个具有类似分布的代理数据集。然而，这无法封装更实际的场景，即MLaaS只返回硬标签，数据分布仍然难以捉摸。此外，现有的大多数工作只关注窃取模型的准确性，而忽略了模型的健壮性，而健壮性在安全敏感的场景中是必不可少的，例如人脸扫描支付。值得注意的是，提高模型的稳健性通常需要使用昂贵的技术，如对抗性训练，从而进一步使窃取稳健性成为更有利可图的前景。针对这些缺陷，本文提出了一种新的无数据硬标签健壮性窃取攻击(DFHL-RS)，该攻击通过简单地查询目标模型的硬标签来实现对模型精度和稳健性的窃取，而不需要任何自然数据。综合实验证明了该方法的有效性。克隆模型在AutoAttack上的清洁准确率为77.86%，健壮性准确率为39.51%，仅比目标模型在CIFAR-10数据集上低4.71%和8.40%，显著超过基线。我们的代码请访问：https://github.com/LetheSec/DFHL-RS-Attack.



