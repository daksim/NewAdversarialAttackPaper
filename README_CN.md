# Latest Adversarial Attack Papers
**update at 2024-05-24 10:15:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Overcoming the Challenges of Batch Normalization in Federated Learning**

克服联邦学习中批量规范化的挑战 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14670v1) [paper-pdf](http://arxiv.org/pdf/2405.14670v1)

**Authors**: Rachid Guerraoui, Rafael Pinot, Geovani Rizk, John Stephan, François Taiani

**Abstract**: Batch normalization has proven to be a very beneficial mechanism to accelerate the training and improve the accuracy of deep neural networks in centralized environments. Yet, the scheme faces significant challenges in federated learning, especially under high data heterogeneity. Essentially, the main challenges arise from external covariate shifts and inconsistent statistics across clients. We introduce in this paper Federated BatchNorm (FBN), a novel scheme that restores the benefits of batch normalization in federated learning. Essentially, FBN ensures that the batch normalization during training is consistent with what would be achieved in a centralized execution, hence preserving the distribution of the data, and providing running statistics that accurately approximate the global statistics. FBN thereby reduces the external covariate shift and matches the evaluation performance of the centralized setting. We also show that, with a slight increase in complexity, we can robustify FBN to mitigate erroneous statistics and potentially adversarial attacks.

摘要: 批量归一化已被证明是一种非常有益的机制，可以在集中式环境下加快训练速度，提高深度神经网络的精度。然而，该方案在联合学习方面面临着巨大的挑战，特别是在数据高度异构性的情况下。从本质上讲，主要挑战来自外部协变量变化和客户之间不一致的统计数据。本文介绍了联邦BatchNorm(FBN)算法，它恢复了联合学习中批量归一化的优点。本质上，FBN确保训练期间的批归一化与在集中执行中实现的一致，从而保持数据的分布，并提供准确接近全局统计的运行统计。因此，FBN减少了外部协变量漂移，并且与集中式设置的评估性能匹配。我们还表明，在复杂性略有增加的情况下，我们可以粗暴地使用FBN来减少错误的统计数据和潜在的对抗性攻击。



## **2. A New Formulation for Zeroth-Order Optimization of Adversarial EXEmples in Malware Detection**

恶意软件检测中对抗实例零阶优化的新公式 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14519v1) [paper-pdf](http://arxiv.org/pdf/2405.14519v1)

**Authors**: Marco Rando, Luca Demetrio, Lorenzo Rosasco, Fabio Roli

**Abstract**: Machine learning malware detectors are vulnerable to adversarial EXEmples, i.e. carefully-crafted Windows programs tailored to evade detection. Unlike other adversarial problems, attacks in this context must be functionality-preserving, a constraint which is challenging to address. As a consequence heuristic algorithms are typically used, that inject new content, either randomly-picked or harvested from legitimate programs. In this paper, we show how learning malware detectors can be cast within a zeroth-order optimization framework which allows to incorporate functionality-preserving manipulations. This permits the deployment of sound and efficient gradient-free optimization algorithms, which come with theoretical guarantees and allow for minimal hyper-parameters tuning. As a by-product, we propose and study ZEXE, a novel zero-order attack against Windows malware detection. Compared to state-of-the-art techniques, ZEXE provides drastic improvement in the evasion rate, while reducing to less than one third the size of the injected content.

摘要: 机器学习恶意软件检测器容易受到敌意例子的攻击，即精心设计的Windows程序，旨在逃避检测。与其他对抗性问题不同，这种情况下的攻击必须保留功能，这是一个具有挑战性的限制。因此，通常使用启发式算法，注入新内容，无论是随机挑选的还是从合法程序中获取的。在这篇文章中，我们展示了如何在零阶优化框架内转换学习恶意软件检测器，该框架允许合并保留功能的操作。这允许部署合理和高效的无梯度优化算法，这些算法具有理论上的保证，并允许最小限度的超参数调整。作为副产品，我们提出并研究了一种针对Windows恶意软件检测的新型零序攻击方法ZEXE。与最先进的技术相比，ZEXE在逃逸率方面有了很大的改进，同时将注入内容的大小减少到不到三分之一。



## **3. SLIFER: Investigating Performance and Robustness of Malware Detection Pipelines**

SIFER：调查恶意软件检测管道的性能和稳健性 cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14478v1) [paper-pdf](http://arxiv.org/pdf/2405.14478v1)

**Authors**: Andrea Ponte, Dmitrijs Trizna, Luca Demetrio, Battista Biggio, Fabio Roli

**Abstract**: As a result of decades of research, Windows malware detection is approached through a plethora of techniques. However, there is an ongoing mismatch between academia -- which pursues an optimal performances in terms of detection rate and low false alarms -- and the requirements of real-world scenarios. In particular, academia focuses on combining static and dynamic analysis within a single or ensemble of models, falling into several pitfalls like (i) firing dynamic analysis without considering the computational burden it requires; (ii) discarding impossible-to-analyse samples; and (iii) analysing robustness against adversarial attacks without considering that malware detectors are complemented with more non-machine-learning components. Thus, in this paper we propose SLIFER, a novel Windows malware detection pipeline sequentially leveraging both static and dynamic analysis, interrupting computations as soon as one module triggers an alarm, requiring dynamic analysis only when needed. Contrary to the state of the art, we investigate how to deal with samples resistance to analysis, showing how much they impact performances, concluding that it is better to flag them as legitimate to not drastically increase false alarms. Lastly, we perform a robustness evaluation of SLIFER leveraging content-injections attacks, and we show that, counter-intuitively, attacks are blocked more by YARA rules than dynamic analysis due to byte artifacts created while optimizing the adversarial strategy.

摘要: 作为数十年研究的结果，Windows恶意软件检测是通过大量技术实现的。然而，学术界--追求在检测率和低虚警方面的最佳表现--与现实世界场景的要求之间存在着持续的不匹配。特别是，学术界专注于在单个或集成模型中结合静态和动态分析，陷入了几个陷阱，如(I)触发动态分析而不考虑其所需的计算负担；(Ii)丢弃无法分析的样本；以及(Iii)分析针对敌意攻击的稳健性，而不考虑恶意软件检测器补充了更多的非机器学习组件。因此，在本文中，我们提出了一种新颖的Windows恶意软件检测流水线Slifer，它顺序地利用静态和动态分析，在一个模块触发警报时立即中断计算，仅在需要时才需要动态分析。与最新技术相反，我们调查了如何处理抗拒分析的样本，显示了它们对性能的影响程度，得出的结论是，最好将它们标记为合法，不要大幅增加错误警报。最后，我们利用内容注入攻击对Slifer进行了健壮性评估，并且我们表明，与直觉相反，由于优化对抗策略时产生的字节伪影，攻击更多地被Yara规则阻止而不是动态分析。



## **4. BadPart: Unified Black-box Adversarial Patch Attacks against Pixel-wise Regression Tasks**

BadPart：针对像素式回归任务的统一黑匣子对抗补丁攻击 cs.CV

Paper accepted at ICML 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.00924v2) [paper-pdf](http://arxiv.org/pdf/2404.00924v2)

**Authors**: Zhiyuan Cheng, Zhaoyi Liu, Tengda Guo, Shiwei Feng, Dongfang Liu, Mingjie Tang, Xiangyu Zhang

**Abstract**: Pixel-wise regression tasks (e.g., monocular depth estimation (MDE) and optical flow estimation (OFE)) have been widely involved in our daily life in applications like autonomous driving, augmented reality and video composition. Although certain applications are security-critical or bear societal significance, the adversarial robustness of such models are not sufficiently studied, especially in the black-box scenario. In this work, we introduce the first unified black-box adversarial patch attack framework against pixel-wise regression tasks, aiming to identify the vulnerabilities of these models under query-based black-box attacks. We propose a novel square-based adversarial patch optimization framework and employ probabilistic square sampling and score-based gradient estimation techniques to generate the patch effectively and efficiently, overcoming the scalability problem of previous black-box patch attacks. Our attack prototype, named BadPart, is evaluated on both MDE and OFE tasks, utilizing a total of 7 models. BadPart surpasses 3 baseline methods in terms of both attack performance and efficiency. We also apply BadPart on the Google online service for portrait depth estimation, causing 43.5% relative distance error with 50K queries. State-of-the-art (SOTA) countermeasures cannot defend our attack effectively.

摘要: 像素级回归任务(如单目深度估计(MDE)和光流估计(OFE))在自动驾驶、增强现实和视频合成等应用中广泛应用于我们的日常生活中。虽然某些应用是安全关键的或具有社会意义的，但这些模型的对抗健壮性没有得到充分的研究，特别是在黑盒场景中。在这项工作中，我们引入了第一个针对像素回归任务的统一黑盒对抗性补丁攻击框架，旨在识别这些模型在基于查询的黑盒攻击下的脆弱性。提出了一种新的基于平方的对抗性补丁优化框架，并利用概率平方采样和基于分数的梯度估计技术有效地生成了补丁，克服了以往黑盒补丁攻击的可扩展性问题。我们的攻击原型名为BadPart，在MDE和OFE任务上进行了评估，总共使用了7个模型。BadPart在攻击性能和效率方面都超过了3种基线方法。我们还将BadPart应用于Google在线服务上进行人像深度估计，在50K查询中导致了43.5%的相对距离误差。最先进的(SOTA)对策不能有效地防御我们的攻击。



## **5. Self-playing Adversarial Language Game Enhances LLM Reasoning**

自玩对抗语言游戏增强LLM推理 cs.CL

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2404.10642v2) [paper-pdf](http://arxiv.org/pdf/2404.10642v2)

**Authors**: Pengyu Cheng, Tianhao Hu, Han Xu, Zhisong Zhang, Yong Dai, Lei Han, Nan Du

**Abstract**: We explore the self-play training procedure of large language models (LLMs) in a two-player adversarial language game called Adversarial Taboo. In this game, an attacker and a defender communicate around a target word only visible to the attacker. The attacker aims to induce the defender to speak the target word unconsciously, while the defender tries to infer the target word from the attacker's utterances. To win the game, both players should have sufficient knowledge about the target word and high-level reasoning ability to infer and express in this information-reserved conversation. Hence, we are curious about whether LLMs' reasoning ability can be further enhanced by self-play in this adversarial language game (SPAG). With this goal, we select several open-source LLMs and let each act as the attacker and play with a copy of itself as the defender on an extensive range of target words. Through reinforcement learning on the game outcomes, we observe that the LLMs' performances uniformly improve on a broad range of reasoning benchmarks. Furthermore, iteratively adopting this self-play process can continuously promote LLMs' reasoning abilities. The code is at https://github.com/Linear95/SPAG.

摘要: 我们探索了在一个名为对抗性禁忌的两人对抗性语言游戏中，大语言模型(LLM)的自我发挥训练过程。在这个游戏中，攻击者和防御者围绕一个只有攻击者才能看到的目标单词进行交流。攻击者的目的是诱导防御者无意识地说出目标词，而防御者则试图从攻击者的话语中推断出目标词。要赢得这场比赛，双方都应该有足够的目标词知识和高级推理能力，以便在这种信息储备的对话中进行推理和表达。因此，我们好奇的是，在这场对抗性语言游戏(SPAG)中，LLMS的推理能力能否通过自我游戏进一步增强。带着这个目标，我们选择了几个开源的LLM，让每个LLM扮演攻击者的角色，并在广泛的目标词上扮演自己的防御者。通过对游戏结果的强化学习，我们观察到LLMS的性能在广泛的推理基准上一致提高。此外，迭代地采用这种自我发挥过程可以不断提升LLMS的推理能力。代码在https://github.com/Linear95/SPAG.



## **6. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

Eidos：高效、不可感知的对抗性3D点云 cs.CV

Preprint

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14210v1) [paper-pdf](http://arxiv.org/pdf/2405.14210v1)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.

摘要: 三维点云的分类是一项具有挑战性的机器学习(ML)任务，在从自动驾驶和机器人辅助手术到低轨道对地观测等一系列实际应用中具有重要的应用。与其他ML任务一样，分类模型在存在对抗性攻击时是出了名的脆弱。这些问题根源于对投入的潜移默化的改变，其结果是，一个看似训练有素的模型最终会错误地对投入进行分类。本文通过介绍EIDOS来加深对敌意攻击的理解，EIDOS是一种在3D点云上提供高效的隐形攻击的框架。Eidos支持一组不同的不可感知性指标。它使用迭代的两步过程来确定最佳对抗性示例，从而实现了运行时不可感知性的权衡。我们提供了与几种流行的三维点云分类模型和几种已建立的三维攻击方法相关的经验证据，表明了Eidos在效率和不可感知性方面的优势。



## **7. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

S-Eval：用于大型语言模型基准安全评估的自动和自适应测试生成 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14191v1) [paper-pdf](http://arxiv.org/pdf/2405.14191v1)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of a LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200, 000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.

摘要: 大型语言模型因其革命性的能力而获得了相当大的关注。然而，人们也越来越担心它们的安全影响，这使得在模型部署之前迫切需要对LLMS进行全面的安全评估。在这项工作中，我们提出了一种新的全面、多维、开放式的安全评价基准S-EVAL。S-EVAL的核心是一种新颖的基于LLM的测试提示自动生成和选择框架，该框架训练一名测试专家，结合一系列测试选择策略，自动构建用于安全评估的高质量测试用例集。这一过程自动化的关键是一种新颖的专家安全评论LLm Mc，它能够量化LLm响应的风险分数，并另外产生风险标签和解释。此外，生成过程还遵循了精心设计的四个不同级别的风险分类，涵盖了令人关注的全面和多维度的安全风险。在此基础上，我们系统地构建了一个新的大规模的低层管理系统安全评估基准，该基准由22万条评估提示组成，其中包括2万条基本风险提示(中文10000条，英文10000条)和来自10种流行的对抗性指令攻击的20万条相应的攻击提示。此外，考虑到LLM的快速演化和伴随的安全威胁，S-EVAL可以灵活配置和调整，以包括新的风险、攻击和模型。S-EVAL在20个流行和有代表性的低成本模型上进行了广泛的评估。结果证实，与现有基准相比，S-EVAL能够更好地反映和告知低成本机械的安全风险。我们还探讨了参数尺度、语言环境和解码参数对评估的影响，为评估LLMS的安全性提供了一种系统的方法。



## **8. Certified Robustness against Sparse Adversarial Perturbations via Data Localization**

通过数据本地化认证针对稀疏对抗扰动的鲁棒性 cs.LG

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14176v1) [paper-pdf](http://arxiv.org/pdf/2405.14176v1)

**Authors**: Ambar Pal, René Vidal, Jeremias Sulam

**Abstract**: Recent work in adversarial robustness suggests that natural data distributions are localized, i.e., they place high probability in small volume regions of the input space, and that this property can be utilized for designing classifiers with improved robustness guarantees for $\ell_2$-bounded perturbations. Yet, it is still unclear if this observation holds true for more general metrics. In this work, we extend this theory to $\ell_0$-bounded adversarial perturbations, where the attacker can modify a few pixels of the image but is unrestricted in the magnitude of perturbation, and we show necessary and sufficient conditions for the existence of $\ell_0$-robust classifiers. Theoretical certification approaches in this regime essentially employ voting over a large ensemble of classifiers. Such procedures are combinatorial and expensive or require complicated certification techniques. In contrast, a simple classifier emerges from our theory, dubbed Box-NN, which naturally incorporates the geometry of the problem and improves upon the current state-of-the-art in certified robustness against sparse attacks for the MNIST and Fashion-MNIST datasets.

摘要: 最近在对抗性稳健性方面的工作表明，自然数据分布是局部化的，即它们将高概率放置在输入空间的小体积区域，并且这一性质可以用于设计具有更好的对$\ell_2有界扰动的稳健性保证的分类器。然而，目前仍不清楚这一观察结果是否适用于更普遍的指标。在这项工作中，我们将这一理论推广到$\ell_0$有界的对抗性扰动，其中攻击者可以修改图像的几个像素，但扰动的大小是不受限制的，我们给出了$\ell_0$-稳健分类器存在的充要条件。这一制度中的理论认证方法实质上是对大量分类器进行投票。这种程序既复杂又昂贵，或者需要复杂的认证技术。相反，从我们的理论中出现了一个简单的分类器，称为Box-NN，它自然地结合了问题的几何结构，并在MNIST和Fashion-MNIST数据集经过验证的针对稀疏攻击的健壮性方面改进了当前的最先进水平。



## **9. Towards Transferable Attacks Against Vision-LLMs in Autonomous Driving with Typography**

通过印刷术在自动驾驶中针对视觉LLM的可转移攻击 cs.CV

12 pages, 5 tables, 5 figures, work in progress

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14169v1) [paper-pdf](http://arxiv.org/pdf/2405.14169v1)

**Authors**: Nhat Chung, Sensen Gao, Tuan-Anh Vu, Jie Zhang, Aishan Liu, Yun Lin, Jin Song Dong, Qing Guo

**Abstract**: Vision-Large-Language-Models (Vision-LLMs) are increasingly being integrated into autonomous driving (AD) systems due to their advanced visual-language reasoning capabilities, targeting the perception, prediction, planning, and control mechanisms. However, Vision-LLMs have demonstrated susceptibilities against various types of adversarial attacks, which would compromise their reliability and safety. To further explore the risk in AD systems and the transferability of practical threats, we propose to leverage typographic attacks against AD systems relying on the decision-making capabilities of Vision-LLMs. Different from the few existing works developing general datasets of typographic attacks, this paper focuses on realistic traffic scenarios where these attacks can be deployed, on their potential effects on the decision-making autonomy, and on the practical ways in which these attacks can be physically presented. To achieve the above goals, we first propose a dataset-agnostic framework for automatically generating false answers that can mislead Vision-LLMs' reasoning. Then, we present a linguistic augmentation scheme that facilitates attacks at image-level and region-level reasoning, and we extend it with attack patterns against multiple reasoning tasks simultaneously. Based on these, we conduct a study on how these attacks can be realized in physical traffic scenarios. Through our empirical study, we evaluate the effectiveness, transferability, and realizability of typographic attacks in traffic scenes. Our findings demonstrate particular harmfulness of the typographic attacks against existing Vision-LLMs (e.g., LLaVA, Qwen-VL, VILA, and Imp), thereby raising community awareness of vulnerabilities when incorporating such models into AD systems. We will release our source code upon acceptance.

摘要: 视觉大语言模型(Vision-LLMS)由于其先进的视觉语言推理能力，针对感知、预测、规划和控制机制，越来越多地被集成到自动驾驶(AD)系统中。然而，Vision-LLM已经显示出对各种类型的对抗性攻击的敏感性，这将损害它们的可靠性和安全性。为了进一步探索AD系统中的风险和实际威胁的可转移性，我们建议依靠Vision-LLMS的决策能力来利用排版攻击AD系统。与现有开发排版攻击通用数据集的少数工作不同，本文关注的是可以部署这些攻击的现实交通场景，它们对决策自主性的潜在影响，以及这些攻击可以物理呈现的实际方式。为了实现上述目标，我们首先提出了一个与数据集无关的框架，用于自动生成可能误导Vision-LLMS推理的错误答案。在此基础上，提出了一种支持图像级和区域级推理攻击的语言扩充方案，并将其扩展为同时针对多个推理任务的攻击模式。在此基础上，对如何在物理流量场景下实现这些攻击进行了研究。通过我们的实证研究，我们评估了交通场景中排版攻击的有效性、可转移性和可实现性。我们的发现证明了针对现有Vision-LLM(例如LLaVA、Qwen-VL、Vila和IMP)的排版攻击的特殊危害性，从而提高了社区对将此类模型整合到AD系统中时的漏洞的认识。我们将在接受后公布我们的源代码。



## **10. Carefully Blending Adversarial Training and Purification Improves Adversarial Robustness**

仔细混合对抗训练和净化提高对抗稳健性 cs.CV

21 pages, 1 figure, 15 tables

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2306.06081v4) [paper-pdf](http://arxiv.org/pdf/2306.06081v4)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and an aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for CIFAR-10, CIFAR-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at https://github.com/emaballarin/CARSO .

摘要: 在这项工作中，我们提出了一种新的用于图像分类的对抗性防御机制-CASO-以协同增强鲁棒性的方式融合了对抗性训练和对抗性净化的范例。该方法建立在对抗性训练的分类器之上，并学习将其与潜在扰动输入相关联的内部表示映射到试探性干净重构的分布上。来自这种分布的多个样本被相同的对抗性训练模型分类，其输出的聚集最终构成感兴趣的稳健预测。基于不同图像数据集的强自适应攻击基准的实验评估表明，CARSO能够抵抗针对随机防御而设计的自适应端到端白盒攻击。付出适度的干净精度代价，我们的方法显著提高了CIFAR-10、CIFAR-100和TinyImageNet-200$\ell_\inty$相对于AutoAttack的稳健分类精度。代码，以及获取预先训练的模型的说明，请访问https://github.com/emaballarin/CARSO。



## **11. Breaking Free: How to Hack Safety Guardrails in Black-Box Diffusion Models!**

挣脱束缚：如何破解黑匣子扩散模型中的安全护栏！ cs.CV

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2402.04699v2) [paper-pdf](http://arxiv.org/pdf/2402.04699v2)

**Authors**: Shashank Kotyan, Po-Yuan Mao, Pin-Yu Chen, Danilo Vasconcellos Vargas

**Abstract**: Deep neural networks can be exploited using natural adversarial samples, which do not impact human perception. Current approaches often rely on deep neural networks' white-box nature to generate these adversarial samples or synthetically alter the distribution of adversarial samples compared to the training distribution. In contrast, we propose EvoSeed, a novel evolutionary strategy-based algorithmic framework for generating photo-realistic natural adversarial samples. Our EvoSeed framework uses auxiliary Conditional Diffusion and Classifier models to operate in a black-box setting. We employ CMA-ES to optimize the search for an initial seed vector, which, when processed by the Conditional Diffusion Model, results in the natural adversarial sample misclassified by the Classifier Model. Experiments show that generated adversarial images are of high image quality, raising concerns about generating harmful content bypassing safety classifiers. Our research opens new avenues to understanding the limitations of current safety mechanisms and the risk of plausible attacks against classifier systems using image generation. Project Website can be accessed at: https://shashankkotyan.github.io/EvoSeed.

摘要: 深度神经网络可以使用自然的对抗性样本来开发，这些样本不会影响人类的感知。目前的方法往往依赖于深度神经网络的白盒性质来生成这些对抗性样本，或者综合改变对抗性样本相对于训练分布的分布。相反，我们提出了EvoSeed，一个新的基于进化策略的算法框架，用于生成照片级的自然对抗性样本。我们的EvoSeed框架使用辅助的条件扩散和分类器模型在黑盒环境中运行。我们使用CMA-ES来优化初始种子向量的搜索，当该初始种子向量被条件扩散模型处理时，导致自然对抗性样本被分类器模型误分类。实验表明，生成的敌意图像具有高图像质量，这引发了人们对绕过安全分类器生成有害内容的担忧。我们的研究为理解当前安全机制的局限性以及使用图像生成对分类器系统进行可信攻击的风险开辟了新的途径。项目网站可访问：https://shashankkotyan.github.io/EvoSeed.



## **12. Nearly Tight Black-Box Auditing of Differentially Private Machine Learning**

差异私有机器学习的近乎严格的黑匣子审计 cs.CR

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14106v1) [paper-pdf](http://arxiv.org/pdf/2405.14106v1)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Emiliano De Cristofaro

**Abstract**: This paper presents a nearly tight audit of the Differentially Private Stochastic Gradient Descent (DP-SGD) algorithm in the black-box model. Our auditing procedure empirically estimates the privacy leakage from DP-SGD using membership inference attacks; unlike prior work, the estimates are appreciably close to the theoretical DP bounds. The main intuition is to craft worst-case initial model parameters, as DP-SGD's privacy analysis is agnostic to the choice of the initial model parameters. For models trained with theoretical $\varepsilon=10.0$ on MNIST and CIFAR-10, our auditing procedure yields empirical estimates of $7.21$ and $6.95$, respectively, on 1,000-record samples and $6.48$ and $4.96$ on the full datasets. By contrast, previous work achieved tight audits only in stronger (i.e., less realistic) white-box models that allow the adversary to access the model's inner parameters and insert arbitrary gradients. Our auditing procedure can be used to detect bugs and DP violations more easily and offers valuable insight into how the privacy analysis of DP-SGD can be further improved.

摘要: 本文对黑盒模型中的差分私有随机梯度下降(DP-SGD)算法进行了严格的审计。我们的审计过程使用成员推理攻击经验地估计了DP-SGD的隐私泄漏；与以前的工作不同，该估计相当接近理论DP界。主要的直觉是制作最坏情况的初始模型参数，因为DP-SGD的隐私分析与初始模型参数的选择无关。对于在MNIST和CIFAR-10上用理论上的$varepsilon=10.0$训练的模型，我们的审计程序在1,000个记录样本上产生的经验估计值分别为7.21美元和6.95美元，在完整数据集上产生的经验估计值分别为6.48美元和4.96美元。相比之下，以前的工作只在更强大(即不太现实)的白盒模型中实现了严格的审计，这些白盒模型允许对手访问模型的内部参数并插入任意渐变。我们的审计程序可用于更轻松地检测错误和DP违规，并为如何进一步改进DP-SGD的隐私分析提供宝贵的见解。



## **13. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

视觉感知推荐系统上的对抗项目推广 cs.IR

Accepted by TOIS 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2312.15826v4) [paper-pdf](http://arxiv.org/pdf/2312.15826v4)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.

摘要: 视觉感知推荐系统在视觉元素对用户潜在偏好的推断有重要作用的领域得到了广泛的应用。虽然加入视觉信息有望提高推荐的准确性和缓解冷启动问题，但必须指出的是，纳入物品图像可能会带来重大的安全挑战。一些已有的研究表明，物品提供者可以通过构建对抗性图像来操纵物品曝光率。然而，这些工作并不能揭示视觉感知推荐系统的真正弱点，因为(1)生成的敌意图像明显失真，使得人类很容易发现它们；(2)攻击的有效性在某些场景下是不一致的，甚至无效的。为了揭示视觉感知推荐系统在面对敌意图像时的真正弱点，提出了一种新的攻击方法--IPDGI(Item Promotion By Diffumation Generated Image)。具体地说，IPDGI使用引导扩散模型来生成敌意样本，旨在欺骗视觉感知的推荐系统。利用扩散模型精确模拟良性图像的分布，生成的对抗性图像与原始图像具有较高的保真度，保证了IPDGI的隐蔽性。为了验证我们提出的方法的有效性，我们在两个常用的电子商务推荐数据集(Amazon Beauty和Amazon Baby)上进行了广泛的实验，并使用几个典型的视觉感知推荐系统进行了实验。实验结果表明，我们的攻击方法在提升长尾(即不受欢迎)项的性能和生成对抗性图像的质量方面都有显著的提高。



## **14. Learning to Transform Dynamically for Better Adversarial Transferability**

学习动态转型以获得更好的对抗可移植性 cs.CV

accepted as a poster in CVPR 2024

**SubmitDate**: 2024-05-23    [abs](http://arxiv.org/abs/2405.14077v1) [paper-pdf](http://arxiv.org/pdf/2405.14077v1)

**Authors**: Rongyi Zhu, Zeliang Zhang, Susan Liang, Zhuo Liu, Chenliang Xu

**Abstract**: Adversarial examples, crafted by adding perturbations imperceptible to humans, can deceive neural networks. Recent studies identify the adversarial transferability across various models, \textit{i.e.}, the cross-model attack ability of adversarial samples. To enhance such adversarial transferability, existing input transformation-based methods diversify input data with transformation augmentation. However, their effectiveness is limited by the finite number of available transformations. In our study, we introduce a novel approach named Learning to Transform (L2T). L2T increases the diversity of transformed images by selecting the optimal combination of operations from a pool of candidates, consequently improving adversarial transferability. We conceptualize the selection of optimal transformation combinations as a trajectory optimization problem and employ a reinforcement learning strategy to effectively solve the problem. Comprehensive experiments on the ImageNet dataset, as well as practical tests with Google Vision and GPT-4V, reveal that L2T surpasses current methodologies in enhancing adversarial transferability, thereby confirming its effectiveness and practical significance. The code is available at https://github.com/RongyiZhu/L2T.

摘要: 通过添加人类察觉不到的扰动而精心制作的对抗性例子可以欺骗神经网络。最近的研究发现了各种模型之间的对抗性转移，即对抗性样本的跨模型攻击能力。为了增强这种对抗性的可转移性，现有的基于输入变换的方法通过变换增强来使输入数据多样化。然而，它们的有效性受到可用变换数量有限的限制。在我们的研究中，我们引入了一种名为学习转化(L2T)的新方法。L2T通过从候选集合中选择最优的操作组合来增加变换图像的多样性，从而提高了对抗性转移。我们将最优变换组合的选择概念化为一个轨迹优化问题，并采用强化学习策略来有效地解决该问题。在ImageNet数据集上的综合实验以及与Google Vision和GPT-4V的实际测试表明，L2T在增强对抗性可转移性方面优于现有方法，从而证实了其有效性和现实意义。代码可在https://github.com/RongyiZhu/L2T.上获得



## **15. Remote Keylogging Attacks in Multi-user VR Applications**

多用户VR应用程序中的远程键盘记录攻击 cs.CR

Accepted for Usenix 2024

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14036v1) [paper-pdf](http://arxiv.org/pdf/2405.14036v1)

**Authors**: Zihao Su, Kunlin Cai, Reuben Beeler, Lukas Dresel, Allan Garcia, Ilya Grishchenko, Yuan Tian, Christopher Kruegel, Giovanni Vigna

**Abstract**: As Virtual Reality (VR) applications grow in popularity, they have bridged distances and brought users closer together. However, with this growth, there have been increasing concerns about security and privacy, especially related to the motion data used to create immersive experiences. In this study, we highlight a significant security threat in multi-user VR applications, which are applications that allow multiple users to interact with each other in the same virtual space. Specifically, we propose a remote attack that utilizes the avatar rendering information collected from an adversary's game clients to extract user-typed secrets like credit card information, passwords, or private conversations. We do this by (1) extracting motion data from network packets, and (2) mapping motion data to keystroke entries. We conducted a user study to verify the attack's effectiveness, in which our attack successfully inferred 97.62% of the keystrokes. Besides, we performed an additional experiment to underline that our attack is practical, confirming its effectiveness even when (1) there are multiple users in a room, and (2) the attacker cannot see the victims. Moreover, we replicated our proposed attack on four applications to demonstrate the generalizability of the attack. These results underscore the severity of the vulnerability and its potential impact on millions of VR social platform users.

摘要: 随着虚拟现实(VR)应用越来越受欢迎，它们弥合了距离，拉近了用户之间的距离。然而，随着这种增长，人们对安全和隐私的担忧也越来越多，特别是与用于创建身临其境体验的运动数据有关。在这项研究中，我们强调了多用户VR应用中的一个重大安全威胁，即允许多个用户在同一虚拟空间中相互交互的应用。具体地说，我们提出了一种远程攻击，它利用从对手的游戏客户端收集的化身渲染信息来提取用户键入的秘密，如信用卡信息、密码或私人对话。我们通过(1)从网络分组中提取运动数据，以及(2)将运动数据映射到击键条目来实现这一点。我们进行了用户研究来验证攻击的有效性，其中我们的攻击成功推断了97.62%的击键。此外，我们还执行了一个额外的实验，以强调我们的攻击是实用的，即使在(1)一个房间有多个用户，以及(2)攻击者看不到受害者的情况下，也证实了它的有效性。此外，我们在四个应用程序上复制了我们提出的攻击，以证明该攻击的泛化能力。这些结果突显了该漏洞的严重性及其对数百万VR社交平台用户的潜在影响。



## **16. Adversarial Training of Two-Layer Polynomial and ReLU Activation Networks via Convex Optimization**

通过凸优化进行两层多边形和ReLU激活网络的对抗训练 cs.LG

6 pages, 4 figures

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14033v1) [paper-pdf](http://arxiv.org/pdf/2405.14033v1)

**Authors**: Daniel Kuelbs, Sanjay Lall, Mert Pilanci

**Abstract**: Training neural networks which are robust to adversarial attacks remains an important problem in deep learning, especially as heavily overparameterized models are adopted in safety-critical settings. Drawing from recent work which reformulates the training problems for two-layer ReLU and polynomial activation networks as convex programs, we devise a convex semidefinite program (SDP) for adversarial training of polynomial activation networks via the S-procedure. We also derive a convex SDP to compute the minimum distance from a correctly classified example to the decision boundary of a polynomial activation network. Adversarial training for two-layer ReLU activation networks has been explored in the literature, but, in contrast to prior work, we present a scalable approach which is compatible with standard machine libraries and GPU acceleration. The adversarial training SDP for polynomial activation networks leads to large increases in robust test accuracy against $\ell^\infty$ attacks on the Breast Cancer Wisconsin dataset from the UCI Machine Learning Repository. For two-layer ReLU networks, we leverage our scalable implementation to retrain the final two fully connected layers of a Pre-Activation ResNet-18 model on the CIFAR-10 dataset. Our 'robustified' model achieves higher clean and robust test accuracies than the same architecture trained with sharpness-aware minimization.

摘要: 训练对敌意攻击稳健的神经网络仍然是深度学习中的一个重要问题，特别是在安全关键环境中采用严重过度参数模型的情况下。借鉴最近将两层RELU和多项式激活网络的训练问题转化为凸规划的工作，我们设计了一个用于多项式激活网络对抗训练的凸半定规划(SDP)，该规划采用S过程。我们还推导了一个凸SDP来计算一个正确分类的例子到多项式激活网络的决策边界的最小距离。两层REU激活网络的对抗性训练已经在文献中得到了探索，但与以前的工作不同的是，我们提出了一种与标准机器库和GPU加速兼容的可扩展方法。多项式激活网络的对抗性训练SDP导致了针对来自UCI机器学习储存库的威斯康星州乳腺癌数据集的$\ell^\inty$攻击的稳健测试精度的大幅提高。对于两层RELU网络，我们利用我们的可扩展实施在CIFAR-10数据集上重新训练预激活ResNet-18模型的最后两个完全连接的层。与采用锐度感知最小化训练的相同架构相比，我们的“强健”模型实现了更高的清洁和健壮的测试精度。



## **17. WordGame: Efficient & Effective LLM Jailbreak via Simultaneous Obfuscation in Query and Response**

WordGame：通过查询和响应中的同时混淆高效且有效的LLM越狱 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.14023v1) [paper-pdf](http://arxiv.org/pdf/2405.14023v1)

**Authors**: Tianrong Zhang, Bochuan Cao, Yuanpu Cao, Lu Lin, Prasenjit Mitra, Jinghui Chen

**Abstract**: The recent breakthrough in large language models (LLMs) such as ChatGPT has revolutionized production processes at an unprecedented pace. Alongside this progress also comes mounting concerns about LLMs' susceptibility to jailbreaking attacks, which leads to the generation of harmful or unsafe content. While safety alignment measures have been implemented in LLMs to mitigate existing jailbreak attempts and force them to become increasingly complicated, it is still far from perfect. In this paper, we analyze the common pattern of the current safety alignment and show that it is possible to exploit such patterns for jailbreaking attacks by simultaneous obfuscation in queries and responses. Specifically, we propose WordGame attack, which replaces malicious words with word games to break down the adversarial intent of a query and encourage benign content regarding the games to precede the anticipated harmful content in the response, creating a context that is hardly covered by any corpus used for safety alignment. Extensive experiments demonstrate that WordGame attack can break the guardrails of the current leading proprietary and open-source LLMs, including the latest Claude-3, GPT-4, and Llama-3 models. Further ablation studies on such simultaneous obfuscation in query and response provide evidence of the merits of the attack strategy beyond an individual attack.

摘要: 最近ChatGPT等大型语言模型(LLM)的突破以前所未有的速度彻底改变了生产流程。在取得这一进展的同时，人们也越来越担心LLMS容易受到越狱攻击，这会导致产生有害或不安全的内容。虽然LLMS已经实施了安全调整措施，以减轻现有的越狱企图，并迫使它们变得越来越复杂，但它仍然远未达到完美。在本文中，我们分析了当前安全对齐的常见模式，并证明了通过在查询和响应中同时混淆来利用这种模式来进行越狱攻击是可能的。具体地说，我们提出了文字游戏攻击，它用文字游戏取代恶意单词，以打破查询的敌对意图，并鼓励与游戏有关的良性内容在响应中位于预期的有害内容之前，从而创造出任何用于安全对齐的语料库几乎无法覆盖的上下文。大量实验表明，文字游戏攻击可以打破目前领先的专有和开源LLM的屏障，包括最新的Claude-3、GPT-4和Llama-3型号。对查询和响应中的这种同时混淆的进一步消融研究提供了除单个攻击之外的攻击策略的优点的证据。



## **18. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

LookHere：具有定向注意力的视觉变形者概括和推断 cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13985v1) [paper-pdf](http://arxiv.org/pdf/2405.13985v1)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.

摘要: 高分辨率图像提供了有关场景的更多信息，可以提高模型精度。然而，计算机视觉中占主导地位的模型体系结构视觉转换器(VIT)在没有精细调整的情况下无法有效地利用更大的图像-VIT在测试时很难外推到更多的补丁，尽管转换器提供了序列长度的灵活性。我们将这一缺陷归因于目前的补丁位置编码方法，这些方法在外推时会产生分布偏移。我们提出了一种替代普通VITS的位置编码的方法，它使用2D注意掩码将注意力头部限制在指向不同方向的固定视野中。我们的新方法LookHere提供了平移等差性，确保了注意力头部的多样性，并限制了注意力头部在外推时面临的分布变化。我们证明LookHere提高了分类性能(平均1.6%)，抗对手攻击(Avg.5.4%)，降低了校准误差(平均1.5%)--在ImageNet上，没有外推。通过外推，LookHere在ImageNet上以$224^2$px进行训练并以$1024^2$px进行测试时，在ImageNet上的性能比当前的SOTA位置编码方法2D-ROPE高21.7%。此外，我们还发布了一个高分辨率测试集来改进高分辨率图像分类器的评估，称为ImageNet-HR。



## **19. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

对抗攻击下的不确定性校准认证 cs.LG

11 pages main paper, appendix included

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13922v1) [paper-pdf](http://arxiv.org/pdf/2405.13922v1)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.

摘要: 由于众所周知，神经分类器对改变其准确性的对抗性扰动敏感，因此\textit{认证方法}的开发是为了提供可证明的保证其预测对此类扰动的不敏感性。此外，在安全关键应用中，分类器置信度的频率主义解释（也称为模型校准）可能至关重要。该属性可以通过Brier评分或预期的校准误差来测量。我们表明，攻击可能会严重损害校准，因此建议将经过认证的校准作为对抗性扰动下校准的最坏情况界限。具体来说，我们通过对预期校准误差求解混合整数程序来产生Brier分数的分析界限和近似界限。最后，我们提出了新颖的校准攻击，并演示了它们如何通过\textit{对抗校准训练}来改进模型校准。



## **20. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2305.17000v4) [paper-pdf](http://arxiv.org/pdf/2305.17000v4)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **21. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

L-AutoDA：利用大型语言模型进行自动化基于决策的对抗性攻击 cs.CR

Camera ready version for GECCO'24 workshop

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2401.15335v2) [paper-pdf](http://arxiv.org/pdf/2401.15335v2)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

摘要: 在快速发展的机器学习领域，敌意攻击对模型的健壮性和安全性提出了重大挑战。基于决策的攻击，只需要对模型的决策进行反馈，而不需要详细的概率或分数，特别隐蔽，很难防御。本文介绍了L-AUTODA(基于大语言模型的自动决策对抗性攻击)，它是一种利用大语言模型的生成能力来自动化攻击设计的新方法。通过在进化框架中迭代地与LLM交互，L自动DA无需太多人力即可自动高效地设计竞争攻击算法。我们在CIFAR-10数据集上验证了L-AutoDA的有效性，在成功率和计算效率上都比基线方法有了显著的提高。我们的发现强调了语言模型作为对抗性攻击生成工具的潜力，并强调了开发健壮的人工智能系统的新途径。



## **22. TrojanRAG: Retrieval-Augmented Generation Can Be Backdoor Driver in Large Language Models**

TrojanRAG：检索增强生成可以成为大型语言模型中的后门驱动程序 cs.CR

18 pages, 13 figures, 4 tables

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13401v1) [paper-pdf](http://arxiv.org/pdf/2405.13401v1)

**Authors**: Pengzhou Cheng, Yidong Ding, Tianjie Ju, Zongru Wu, Wei Du, Ping Yi, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Large language models (LLMs) have raised concerns about potential security threats despite performing significantly in Natural Language Processing (NLP). Backdoor attacks initially verified that LLM is doing substantial harm at all stages, but the cost and robustness have been criticized. Attacking LLMs is inherently risky in security review, while prohibitively expensive. Besides, the continuous iteration of LLMs will degrade the robustness of backdoors. In this paper, we propose TrojanRAG, which employs a joint backdoor attack in the Retrieval-Augmented Generation, thereby manipulating LLMs in universal attack scenarios. Specifically, the adversary constructs elaborate target contexts and trigger sets. Multiple pairs of backdoor shortcuts are orthogonally optimized by contrastive learning, thus constraining the triggering conditions to a parameter subspace to improve the matching. To improve the recall of the RAG for the target contexts, we introduce a knowledge graph to construct structured data to achieve hard matching at a fine-grained level. Moreover, we normalize the backdoor scenarios in LLMs to analyze the real harm caused by backdoors from both attackers' and users' perspectives and further verify whether the context is a favorable tool for jailbreaking models. Extensive experimental results on truthfulness, language understanding, and harmfulness show that TrojanRAG exhibits versatility threats while maintaining retrieval capabilities on normal queries.

摘要: 尽管大型语言模型(LLM)在自然语言处理(NLP)中表现出色，但仍引发了人们对潜在安全威胁的担忧。后门攻击最初证实了LLM在所有阶段都在造成实质性的危害，但其成本和健壮性受到了批评。在安全审查中，攻击LLMS固有的风险，同时代价高得令人望而却步。此外，LLMS的连续迭代会降低后门的健壮性。在本文中，我们提出了TrojanRAG，它在检索-增强生成中使用联合后门攻击，从而在通用攻击场景下操纵LLMS。具体地说，对手构建了精心设计的目标上下文和触发集。通过对比学习对多对后门捷径进行正交化优化，从而将触发条件约束到一个参数子空间以提高匹配性。为了提高RAG对目标上下文的查全率，我们引入了知识图来构建结构化数据，以实现细粒度的硬匹配。此外，我们对LLMS中的后门场景进行了规范化，从攻击者和用户的角度分析了后门造成的真实危害，并进一步验证了上下文是否为越狱模型的有利工具。在真实性、语言理解和危害性方面的大量实验结果表明，TrojanRAG在保持对正常查询的检索能力的同时，表现出通用性威胁。



## **23. Adversarial Training via Adaptive Knowledge Amalgamation of an Ensemble of Teachers**

通过教师群体的适应性知识融合进行对抗性培训 cs.LG

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.13324v1) [paper-pdf](http://arxiv.org/pdf/2405.13324v1)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Adversarial training (AT) is a popular method for training robust deep neural networks (DNNs) against adversarial attacks. Yet, AT suffers from two shortcomings: (i) the robustness of DNNs trained by AT is highly intertwined with the size of the DNNs, posing challenges in achieving robustness in smaller models; and (ii) the adversarial samples employed during the AT process exhibit poor generalization, leaving DNNs vulnerable to unforeseen attack types. To address these dual challenges, this paper introduces adversarial training via adaptive knowledge amalgamation of an ensemble of teachers (AT-AKA). In particular, we generate a diverse set of adversarial samples as the inputs to an ensemble of teachers; and then, we adaptively amalgamate the logtis of these teachers to train a generalized-robust student. Through comprehensive experiments, we illustrate the superior efficacy of AT-AKA over existing AT methods and adversarial robustness distillation techniques against cutting-edge attacks, including AutoAttack.

摘要: 对抗性训练(AT)是训练稳健的深层神经网络(DNN)抵抗敌意攻击的一种常用方法。然而，AT存在两个缺点：(I)由AT训练的DNN的健壮性与DNN的大小高度交织在一起，这给在较小模型中实现健壮性带来了挑战；(Ii)在AT过程中使用的敌意样本表现出较差的泛化能力，使得DNN容易受到不可预见的攻击类型的影响。为了应对这些双重挑战，本文引入了教师自适应知识融合的对抗性训练(AT-AKA)。特别是，我们生成了一组不同的对抗性样本作为教师集成的输入，然后，我们自适应地合并这些教师的逻辑，以训练一个普遍健壮的学生。通过综合实验，证明了AT-AKA算法对包括AutoAttack在内的前沿攻击具有优于现有的AT方法和对手健壮性蒸馏技术的有效性。



## **24. Box-Free Model Watermarks Are Prone to Black-Box Removal Attacks**

无框模型水印容易受到黑匣子删除攻击 cs.CV

**SubmitDate**: 2024-05-22    [abs](http://arxiv.org/abs/2405.09863v2) [paper-pdf](http://arxiv.org/pdf/2405.09863v2)

**Authors**: Haonan An, Guang Hua, Zhiping Lin, Yuguang Fang

**Abstract**: Box-free model watermarking is an emerging technique to safeguard the intellectual property of deep learning models, particularly those for low-level image processing tasks. Existing works have verified and improved its effectiveness in several aspects. However, in this paper, we reveal that box-free model watermarking is prone to removal attacks, even under the real-world threat model such that the protected model and the watermark extractor are in black boxes. Under this setting, we carry out three studies. 1) We develop an extractor-gradient-guided (EGG) remover and show its effectiveness when the extractor uses ReLU activation only. 2) More generally, for an unknown extractor, we leverage adversarial attacks and design the EGG remover based on the estimated gradients. 3) Under the most stringent condition that the extractor is inaccessible, we design a transferable remover based on a set of private proxy models. In all cases, the proposed removers can successfully remove embedded watermarks while preserving the quality of the processed images, and we also demonstrate that the EGG remover can even replace the watermarks. Extensive experimental results verify the effectiveness and generalizability of the proposed attacks, revealing the vulnerabilities of the existing box-free methods and calling for further research.

摘要: 无盒模型水印是一种新兴的保护深度学习模型知识产权的技术，尤其是用于低层图像处理任务的模型。已有的工作在几个方面验证和改进了它的有效性。然而，在本文中，我们揭示了无盒模型水印容易受到移除攻击，即使在真实世界的威胁模型下，受保护的模型和水印抽取器都在黑盒中。在此背景下，我们开展了三个方面的研究。1)我们开发了一种萃取器-梯度引导(EGG)去除器，并在仅使用RELU激活的情况下展示了其有效性。2)更一般地，对于未知的提取者，我们利用对抗性攻击，并基于估计的梯度来设计鸡蛋去除器。3)在抽取器不可访问的最严格条件下，基于一组私有代理模型设计了一个可转移的抽取器。在所有情况下，所提出的去除器都可以在保持处理图像质量的情况下成功地去除嵌入的水印，并且我们还证明了鸡蛋去除器甚至可以替换水印。大量的实验结果验证了所提出的攻击方法的有效性和泛化能力，揭示了现有去盒方法的弱点，需要进一步研究。



## **25. Cloud-based XAI Services for Assessing Open Repository Models Under Adversarial Attacks**

基于云的XAI服务，用于评估对抗性攻击下的开放存储库模型 cs.CR

Accepted by IEEE International Conference on Software Services  Engineering (SSE) 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2401.12261v3) [paper-pdf](http://arxiv.org/pdf/2401.12261v3)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: The opacity of AI models necessitates both validation and evaluation before their integration into services. To investigate these models, explainable AI (XAI) employs methods that elucidate the relationship between input features and output predictions. The operations of XAI extend beyond the execution of a single algorithm, involving a series of activities that include preprocessing data, adjusting XAI to align with model parameters, invoking the model to generate predictions, and summarizing the XAI results. Adversarial attacks are well-known threats that aim to mislead AI models. The assessment complexity, especially for XAI, increases when open-source AI models are subject to adversarial attacks, due to various combinations. To automate the numerous entities and tasks involved in XAI-based assessments, we propose a cloud-based service framework that encapsulates computing components as microservices and organizes assessment tasks into pipelines. The current XAI tools are not inherently service-oriented. This framework also integrates open XAI tool libraries as part of the pipeline composition. We demonstrate the application of XAI services for assessing five quality attributes of AI models: (1) computational cost, (2) performance, (3) robustness, (4) explanation deviation, and (5) explanation resilience across computer vision and tabular cases. The service framework generates aggregated analysis that showcases the quality attributes for more than a hundred combination scenarios.

摘要: 人工智能模型的不透明性需要在将其集成到服务中之前进行验证和评估。为了研究这些模型，可解释人工智能(XAI)使用了一些方法来阐明输入特征和输出预测之间的关系。XAI的操作超出了单一算法的执行范围，涉及一系列活动，包括数据预处理、调整XAI以与模型参数保持一致、调用模型以生成预测，以及汇总XAI结果。对抗性攻击是众所周知的旨在误导人工智能模型的威胁。当开源AI模型由于各种组合而受到对抗性攻击时，评估的复杂性，特别是对XAI来说，会增加。为了自动化基于XAI的评估中涉及的众多实体和任务，我们提出了一个基于云的服务框架，该框架将计算组件封装为微服务，并将评估任务组织到管道中。当前的XAI工具本身并不是面向服务的。该框架还集成了开放的XAI工具库，作为管道组合的一部分。我们展示了XAI服务在评估人工智能模型的五个质量属性中的应用：(1)计算成本，(2)性能，(3)稳健性，(4)解释偏差，(5)跨计算机视觉和表格案例的解释弹性。服务框架生成聚合分析，展示一百多个组合场景的质量属性。



## **26. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不精确的遗忘需要更仔细的评估，以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.01218v3) [paper-pdf](http://arxiv.org/pdf/2403.01218v3)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their "U-MIA" counterparts). We propose a categorization of existing U-MIAs into "population U-MIAs", where the same attacker is instantiated for all examples, and "per-example U-MIAs", where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高昂成本使得开发忘却学习的技术变得越来越受欢迎。这些技术寻求消除训练示例的影响，而不必从头开始重新训练模型。直观地说，一旦模型取消学习，与该模型交互的对手应该不再能够判断未学习的示例是否包括在该模型的训练集中。在隐私文献中，这被称为成员关系推断。在这项工作中，我们讨论了成员关系推理攻击(MIA)对遗忘环境的适应(导致它们的U-MIA对应)。我们建议将现有的U-MIA分类为“群体U-MIA”，其中针对所有示例实例化相同的攻击者，以及“每示例U-MIA”，其中针对每个示例实例化一个专门的攻击者。我们表明，后一类，其中攻击者根据每个被攻击的例子定制其成员预测，明显更强。事实上，我们的结果表明，遗忘文献中常用的U-MIA高估了现有遗忘技术在视觉和语言模型上提供的隐私保护。我们的调查显示，不同示例对每个示例的U-MIA的脆弱性存在很大差异。事实上，几种忘记算法降低了我们希望忘记的一些(但不是所有)示例的脆弱性，但代价是增加了其他示例的脆弱性。值得注意的是，我们发现，由于遗忘，其余训练样本的隐私保护可能会恶化。我们还讨论了使用现有的遗忘方案平等地保护所有例子的基本困难，因为例子被遗忘的比率不同。我们证明，根据不同的例子调整遗忘停止标准的天真尝试无法缓解这些问题。



## **27. Adversarial Attacks and Defenses in Automated Control Systems: A Comprehensive Benchmark**

自动控制系统中的对抗性攻击和防御：全面的基准 cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2403.13502v3) [paper-pdf](http://arxiv.org/pdf/2403.13502v3)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.

摘要: 将机器学习集成到自动化控制系统（ACS）中，增强了工业过程管理中的决策。这些技术在工业中广泛采用的局限性之一是神经网络容易受到对抗攻击。本研究探索了使用田纳西州伊士曼Process数据集在ACS中部署深度学习模型进行故障诊断的威胁。通过评估具有不同架构的三个神经网络，我们将它们置于六种类型的对抗攻击中，并探索五种不同的防御方法。我们的结果凸显了模型对对抗样本的强烈脆弱性以及防御策略的不同有效性。我们还提出了一种通过结合多种防御方法的新型保护方法，并证明了其有效性。这项研究为确保ACS内的机器学习提供了多项见解，确保工业流程中的稳健故障诊断。



## **28. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

重新思考面部识别系统的漏洞：从实践的角度 cs.CR

19 pages

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12786v1) [paper-pdf](http://arxiv.org/pdf/2405.12786v1)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.

摘要: 人脸识别系统(FRS)越来越多地集成到包括监控和用户身份验证在内的关键应用中，突显了它们在现代安全系统中的关键作用。最近的研究发现，FRS对对抗性攻击(例如，对抗性补丁攻击)和后门攻击(例如，训练数据中毒)的脆弱性，引起了人们对其可靠性和可信性的严重担忧。以往的研究主要集中于传统的对抗性攻击或后门攻击，忽略了此类威胁的资源密集型或特权操纵性，从而限制了它们的实用通用性、隐蔽性、普遍性和健壮性。相应地，在本文中，我们通过用户研究和初步探索，深入研究了FRS的固有漏洞。通过利用这些漏洞，我们确定了一种新型的攻击，即面部识别后门攻击，称为FIBA，它揭示了对FRS的一个潜在的更具破坏性的威胁：注册阶段的后门攻击。FIBA绕过了传统攻击的限制，允许任何使用特定触发器的攻击者绕过这些系统，从而实现广泛的破坏。这意味着在将单个有毒示例插入数据库后，相应的触发器将成为任何攻击者欺骗FRS的通用密钥。该策略实质上是通过在注册阶段发起攻击来挑战传统攻击，通过毒化特征数据库而不是训练数据来极大地改变威胁格局。



## **29. Generative AI and Large Language Models for Cyber Security: All Insights You Need**

网络安全的生成性人工智能和大型语言模型：您需要的所有见解 cs.CR

50 pages, 8 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12750v1) [paper-pdf](http://arxiv.org/pdf/2405.12750v1)

**Authors**: Mohamed Amine Ferrag, Fatima Alwahedi, Ammar Battah, Bilel Cherif, Abdechakour Mechri, Norbert Tihanyi

**Abstract**: This paper provides a comprehensive review of the future of cybersecurity through Generative AI and Large Language Models (LLMs). We explore LLM applications across various domains, including hardware design security, intrusion detection, software engineering, design verification, cyber threat intelligence, malware detection, and phishing detection. We present an overview of LLM evolution and its current state, focusing on advancements in models such as GPT-4, GPT-3.5, Mixtral-8x7B, BERT, Falcon2, and LLaMA. Our analysis extends to LLM vulnerabilities, such as prompt injection, insecure output handling, data poisoning, DDoS attacks, and adversarial instructions. We delve into mitigation strategies to protect these models, providing a comprehensive look at potential attack scenarios and prevention techniques. Furthermore, we evaluate the performance of 42 LLM models in cybersecurity knowledge and hardware security, highlighting their strengths and weaknesses. We thoroughly evaluate cybersecurity datasets for LLM training and testing, covering the lifecycle from data creation to usage and identifying gaps for future research. In addition, we review new strategies for leveraging LLMs, including techniques like Half-Quadratic Quantization (HQQ), Reinforcement Learning with Human Feedback (RLHF), Direct Preference Optimization (DPO), Quantized Low-Rank Adapters (QLoRA), and Retrieval-Augmented Generation (RAG). These insights aim to enhance real-time cybersecurity defenses and improve the sophistication of LLM applications in threat detection and response. Our paper provides a foundational understanding and strategic direction for integrating LLMs into future cybersecurity frameworks, emphasizing innovation and robust model deployment to safeguard against evolving cyber threats.

摘要: 本文通过生成式人工智能和大型语言模型(LLMS)对网络安全的未来进行了全面的回顾。我们探索了LLM在不同领域的应用，包括硬件设计安全、入侵检测、软件工程、设计验证、网络威胁情报、恶意软件检测和网络钓鱼检测。我们概述了LLM的演化和现状，重点介绍了GPT-4、GPT-3.5、Mixtral-8x7B、BERT、Falcon2和Llama等模型的进展。我们的分析扩展到LLM漏洞，如快速注入、不安全的输出处理、数据中毒、DDoS攻击和敌意指令。我们深入研究缓解策略以保护这些模型，提供对潜在攻击场景和预防技术的全面了解。此外，我们评估了42个LLM模型在网络安全知识和硬件安全方面的性能，突出了它们的优势和劣势。我们为LLM培训和测试彻底评估网络安全数据集，涵盖从数据创建到使用的整个生命周期，并为未来的研究确定差距。此外，我们还回顾了利用LLMS的新策略，包括半二次量化(HQQ)、带人反馈的强化学习(RLHF)、直接偏好优化(DPO)、量化低阶适配器(QLoRA)和检索增强生成(RAG)。这些见解旨在增强实时网络安全防御，并提高LLM应用程序在威胁检测和响应方面的复杂性。我们的论文为将低成本管理系统整合到未来的网络安全框架中提供了一个基础性的理解和战略方向，强调创新和稳健的模型部署，以防范不断演变的网络威胁。



## **30. A GAN-Based Data Poisoning Attack Against Federated Learning Systems and Its Countermeasure**

针对联邦学习系统的基于GAN的数据中毒攻击及其对策 cs.CR

18 pages, 16 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.11440v2) [paper-pdf](http://arxiv.org/pdf/2405.11440v2)

**Authors**: Wei Sun, Bo Gao, Ke Xiong, Yuwei Wang

**Abstract**: As a distributed machine learning paradigm, federated learning (FL) is collaboratively carried out on privately owned datasets but without direct data access. Although the original intention is to allay data privacy concerns, "available but not visible" data in FL potentially brings new security threats, particularly poisoning attacks that target such "not visible" local data. Initial attempts have been made to conduct data poisoning attacks against FL systems, but cannot be fully successful due to their high chance of causing statistical anomalies. To unleash the potential for truly "invisible" attacks and build a more deterrent threat model, in this paper, a new data poisoning attack model named VagueGAN is proposed, which can generate seemingly legitimate but noisy poisoned data by untraditionally taking advantage of generative adversarial network (GAN) variants. Capable of manipulating the quality of poisoned data on demand, VagueGAN enables to trade-off attack effectiveness and stealthiness. Furthermore, a cost-effective countermeasure named Model Consistency-Based Defense (MCD) is proposed to identify GAN-poisoned data or models after finding out the consistency of GAN outputs. Extensive experiments on multiple datasets indicate that our attack method is generally much more stealthy as well as more effective in degrading FL performance with low complexity. Our defense method is also shown to be more competent in identifying GAN-poisoned data or models. The source codes are publicly available at \href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.

摘要: 作为一种分布式机器学习范式，联合学习(FL)是在私有数据集上协作进行的，但不需要直接访问数据。虽然初衷是为了缓解数据隐私问题，但FL中的“可用但不可见”数据可能会带来新的安全威胁，特别是针对此类“不可见”本地数据的中毒攻击。已经进行了针对FL系统的数据中毒攻击的初步尝试，但由于造成统计异常的可能性很高，因此不能完全成功。为了释放真正隐形攻击的可能性，建立更具威慑力的威胁模型，提出了一种新的数据中毒攻击模型VagueGAN，该模型通过非传统地利用生成性对手网络(GAN)变体来生成看似合法但含有噪声的有毒数据。VagueGAN能够按需操纵有毒数据的质量，从而能够在攻击效率和隐蔽性之间进行权衡。此外，还提出了一种基于模型一致性防御(MCD)的高性价比对策，用于在发现GaN输出的一致性之后识别GaN中毒数据或模型。在多个数据集上的大量实验表明，我们的攻击方法通常更隐蔽，并且在降低复杂度的情况下更有效地降低了FL性能。我们的防御方法也被证明在识别GaN中毒数据或模型方面更有能力。源代码可在\href{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}{https://github.com/SSssWEIssSS/VagueGAN-Data-Poisoning-Attack-and-Its-Countermeasure}.上公开获取



## **31. How to Train a Backdoor-Robust Model on a Poisoned Dataset without Auxiliary Data?**

如何在没有辅助数据的情况下在中毒数据集中训练后门稳健模型？ cs.CR

13 pages, under review

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12719v1) [paper-pdf](http://arxiv.org/pdf/2405.12719v1)

**Authors**: Yuwen Pu, Jiahao Chen, Chunyi Zhou, Zhou Feng, Qingming Li, Chunqiang Hu, Shouling Ji

**Abstract**: Backdoor attacks have attracted wide attention from academia and industry due to their great security threat to deep neural networks (DNN). Most of the existing methods propose to conduct backdoor attacks by poisoning the training dataset with different strategies, so it's critical to identify the poisoned samples and then train a clean model on the unreliable dataset in the context of defending backdoor attacks. Although numerous backdoor countermeasure researches are proposed, their inherent weaknesses render them limited in practical scenarios, such as the requirement of enough clean samples, unstable defense performance under various attack conditions, poor defense performance against adaptive attacks, and so on.Therefore, in this paper, we are committed to overcome the above limitations and propose a more practical backdoor defense method. Concretely, we first explore the inherent relationship between the potential perturbations and the backdoor trigger, and the theoretical analysis and experimental results demonstrate that the poisoned samples perform more robustness to perturbation than the clean ones. Then, based on our key explorations, we introduce AdvrBD, an Adversarial perturbation-based and robust Backdoor Defense framework, which can effectively identify the poisoned samples and train a clean model on the poisoned dataset. Constructively, our AdvrBD eliminates the requirement for any clean samples or knowledge about the poisoned dataset (e.g., poisoning ratio), which significantly improves the practicality in real-world scenarios.

摘要: 后门攻击因其对深度神经网络(DNN)的巨大安全威胁而受到学术界和工业界的广泛关注。现有的大多数方法都是通过对训练数据集使用不同的策略进行中毒来进行后门攻击，因此在防御后门攻击的背景下，识别中毒样本并在不可靠的数据集上训练一个干净的模型是至关重要的。虽然已经提出了大量的后门对抗研究，但其固有的缺陷使其在实际场景中受到限制，如需要足够的清洁样本，在各种攻击条件下的防御性能不稳定，对自适应攻击的防御性能较差等，因此，本文致力于克服上述局限性，提出一种更实用的后门防御方法。具体地说，我们首先探讨了潜在扰动和后门触发之间的内在联系，理论分析和实验结果表明，中毒样本比干净样本对扰动具有更强的鲁棒性。然后，在重点探索的基础上，提出了一种基于对抗性扰动的健壮后门防御框架AdvrBD，它可以有效地识别有毒样本，并在有毒数据集上训练一个干净的模型。建设性地说，我们的AdvrBD不需要任何干净的样本或关于有毒数据集的知识(例如，投毒率)，这显著提高了现实世界场景中的实用性。



## **32. Robust Classification via a Single Diffusion Model**

通过单一扩散模型的稳健分类 cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2305.15241v2) [paper-pdf](http://arxiv.org/pdf/2305.15241v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Diffusion models have been applied to improve adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, this paper proposes Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. RDC first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood estimated by the diffusion model through Bayes' theorem. To further reduce the computational cost, we propose a new diffusion backbone called multi-head diffusion and develop efficient sampling strategies. As RDC does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $75.67\%$ robust accuracy against various $\ell_\infty$ norm-bounded adaptive attacks with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+4.77\%$. The results highlight the potential of generative classifiers by employing pre-trained diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers. Code is available at \url{https://github.com/huanranchen/DiffusionClassifier}.

摘要: 扩散模型已被应用于通过净化对抗性噪声或生成用于对抗性训练的真实数据来提高图像分类器的对抗性鲁棒性。然而，基于扩散的净化方法可以通过更强的自适应攻击来规避，而对抗性训练在看不见的威胁下表现不佳，显示出这些方法不可避免的局限性。为了更好地利用扩散模型的表达能力，提出了稳健扩散分类器(RDC)，它是一种生成式分类器，由预先训练的扩散模型构造而成，具有相反的鲁棒性。RDC首先最大化给定输入的数据似然，然后利用扩散模型通过贝叶斯定理估计的条件似然来预测优化输入的类别概率。为了进一步降低计算成本，我们提出了一种新的扩散骨干，称为多头扩散，并开发了高效的采样策略。由于RDC不需要关于特定对手攻击的培训，我们证明了防御多个看不见的威胁更具普遍性。特别是，在CIFAR-10上，RDC对各种有界自适应攻击的稳健准确率达到了75.67美元，其中$epsilon_INFTY=8/255$，比以前最先进的对抗性训练模型高出$+4.77$。与通常研究的判别分类器相比，这些结果突出了生成式分类器通过使用预训练的扩散模型来提高对抗稳健性的潜力。代码可在\url{https://github.com/huanranchen/DiffusionClassifier}.上找到



## **33. EmInspector: Combating Backdoor Attacks in Federated Self-Supervised Learning Through Embedding Inspection**

EmInspector：通过嵌入检查打击联邦自我监督学习中的后门攻击 cs.CR

18 pages, 12 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13080v1) [paper-pdf](http://arxiv.org/pdf/2405.13080v1)

**Authors**: Yuwen Qian, Shuchi Wu, Kang Wei, Ming Ding, Di Xiao, Tao Xiang, Chuan Ma, Song Guo

**Abstract**: Federated self-supervised learning (FSSL) has recently emerged as a promising paradigm that enables the exploitation of clients' vast amounts of unlabeled data while preserving data privacy. While FSSL offers advantages, its susceptibility to backdoor attacks, a concern identified in traditional federated supervised learning (FSL), has not been investigated. To fill the research gap, we undertake a comprehensive investigation into a backdoor attack paradigm, where unscrupulous clients conspire to manipulate the global model, revealing the vulnerability of FSSL to such attacks. In FSL, backdoor attacks typically build a direct association between the backdoor trigger and the target label. In contrast, in FSSL, backdoor attacks aim to alter the global model's representation for images containing the attacker's specified trigger pattern in favor of the attacker's intended target class, which is less straightforward. In this sense, we demonstrate that existing defenses are insufficient to mitigate the investigated backdoor attacks in FSSL, thus finding an effective defense mechanism is urgent. To tackle this issue, we dive into the fundamental mechanism of backdoor attacks on FSSL, proposing the Embedding Inspector (EmInspector) that detects malicious clients by inspecting the embedding space of local models. In particular, EmInspector assesses the similarity of embeddings from different local models using a small set of inspection images (e.g., ten images of CIFAR100) without specific requirements on sample distribution or labels. We discover that embeddings from backdoored models tend to cluster together in the embedding space for a given inspection image. Evaluation results show that EmInspector can effectively mitigate backdoor attacks on FSSL across various adversary settings. Our code is avaliable at https://github.com/ShuchiWu/EmInspector.

摘要: 联合自我监督学习(FSSL)最近成为一种很有前途的范例，它能够在保护数据隐私的同时利用客户的大量未标记数据。虽然FSSL提供了优势，但它对后门攻击的敏感性尚未得到调查，这是传统联邦监督学习(FSL)中发现的一个问题。为了填补这一研究空白，我们对后门攻击范式进行了全面调查，在这种范式中，不择手段的客户合谋操纵全球模型，揭示了FSSL对此类攻击的脆弱性。在FSL中，后门攻击通常在后门触发器和目标标签之间建立直接关联。相比之下，在FSSL中，后门攻击的目标是更改包含攻击者指定的触发模式的图像的全局模型表示，以支持攻击者预期的目标类，这不是那么直接。从这个意义上说，我们证明了现有的防御措施不足以缓解FSSL中被调查的后门攻击，因此迫切需要找到一种有效的防御机制。为了解决这个问题，我们深入研究了FSSL后门攻击的基本机制，提出了嵌入检查器(EmInspector)，它通过检查本地模型的嵌入空间来检测恶意客户端。特别是，EmInspector使用一小组检查图像(例如，CIFAR100的10张图像)来评估来自不同本地模型的嵌入的相似性，而对样本分布或标签没有特定要求。我们发现，对于给定的检查图像，来自后置模型的嵌入往往在嵌入空间中聚集在一起。评估结果表明，EmInspector能够有效地缓解各种敌方环境下对FSSL的后门攻击。我们的代码可在https://github.com/ShuchiWu/EmInspector.上获得



## **34. Fully Randomized Pointers**

完全随机指针 cs.CR

24 pages, 3 figures

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12513v1) [paper-pdf](http://arxiv.org/pdf/2405.12513v1)

**Authors**: Gregory J. Duck, Sai Dhawal Phaye, Roland H. C. Yap, Trevor E. Carlson

**Abstract**: Software security continues to be a critical concern for programs implemented in low-level programming languages such as C and C++. Many defenses have been proposed in the current literature, each with different trade-offs including performance, compatibility, and attack resistance. One general class of defense is pointer randomization or authentication, where invalid object access (e.g., memory errors) is obfuscated or denied. Many defenses rely on the program termination (e.g., crashing) to abort attacks, with the implicit assumption that an adversary cannot "brute force" the defense with multiple attack attempts. However, such assumptions do not always hold, such as hardware speculative execution attacks or network servers configured to restart on error. In such cases, we argue that most existing defenses provide only weak effective security.   In this paper, we propose Fully Randomized Pointers (FRP) as a stronger memory error defense that is resistant to even brute force attacks. The key idea is to fully randomize pointer bits -- as much as possible while also preserving binary compatibility -- rendering the relationships between pointers highly unpredictable. Furthermore, the very high degree of randomization renders brute force attacks impractical -- providing strong effective security compared to existing work. We design a new FRP encoding that is: (1) compatible with existing binary code (without recompilation); (2) decoupled from the underlying object layout; and (3) can be efficiently decoded on-the-fly to the underlying memory address. We prototype FRP in the form of a software implementation (BlueFat) to test security and compatibility, and a proof-of-concept hardware implementation (GreenFat) to evaluate performance. We show that FRP is secure, practical, and compatible at the binary level, while a hardware implementation can achieve low performance overheads (<10%).

摘要: 对于使用低级编程语言(如C和C++)实现的程序来说，软件安全性仍然是一个关键问题。在当前的文献中已经提出了许多防御措施，每种防御措施都具有不同的权衡，包括性能、兼容性和抗攻击能力。一种常见的防御类别是指针随机化或身份验证，在这种情况下，无效对象访问(例如，内存错误)被混淆或拒绝。许多防御依赖于程序终止(例如崩溃)来中止攻击，并隐含地假设对手不能通过多次攻击尝试来“野蛮地强迫”防御。然而，这样的假设并不总是成立的，例如硬件推测性执行攻击或配置为在出错时重新启动的网络服务器。在这种情况下，我们认为，大多数现有的防御措施只提供了薄弱的有效安全。在本文中，我们提出了完全随机化指针(FRP)作为一种更强的内存错误防御机制，它甚至可以抵抗暴力攻击。其关键思想是完全随机化指针位--尽可能多地同时保持二进制兼容性--使指针之间的关系高度不可预测。此外，非常高的随机性使暴力攻击变得不切实际--与现有工作相比，提供了强大的有效安全性。我们设计了一种新的FRP编码：(1)与现有的二进制代码兼容(无需重新编译)；(2)与底层对象布局解耦；(3)可以高效地动态解码到底层内存地址。我们以软件实现(BlueFat)的形式构建FRP原型以测试安全性和兼容性，并以概念验证硬件实现(GreenFat)的形式评估性能。我们证明了FRP是安全的、实用的和二进制级兼容的，而硬件实现可以获得较低的性能开销(<10%)。



## **35. GPT-4 Jailbreaks Itself with Near-Perfect Success Using Self-Explanation**

GPT-4通过自我解释以近乎完美的成功越狱 cs.CR

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.13077v1) [paper-pdf](http://arxiv.org/pdf/2405.13077v1)

**Authors**: Govind Ramesh, Yao Dou, Wei Xu

**Abstract**: Research on jailbreaking has been valuable for testing and understanding the safety and security issues of large language models (LLMs). In this paper, we introduce Iterative Refinement Induced Self-Jailbreak (IRIS), a novel approach that leverages the reflective capabilities of LLMs for jailbreaking with only black-box access. Unlike previous methods, IRIS simplifies the jailbreaking process by using a single model as both the attacker and target. This method first iteratively refines adversarial prompts through self-explanation, which is crucial for ensuring that even well-aligned LLMs obey adversarial instructions. IRIS then rates and enhances the output given the refined prompt to increase its harmfulness. We find IRIS achieves jailbreak success rates of 98% on GPT-4 and 92% on GPT-4 Turbo in under 7 queries. It significantly outperforms prior approaches in automatic, black-box and interpretable jailbreaking, while requiring substantially fewer queries, thereby establishing a new standard for interpretable jailbreaking methods.

摘要: 越狱研究对于测试和理解大型语言模型(LLM)的安全和安保问题具有重要价值。在本文中，我们介绍了迭代精化诱导的自越狱(IRIS)，这是一种新的方法，它利用LLMS的反射能力来实现只访问黑盒的越狱。与以前的方法不同，IRIS通过将单一模型用作攻击者和目标来简化越狱过程。这种方法首先通过自我解释迭代地精炼对抗性提示，这对于确保即使是排列良好的LLM也遵守对抗性指令至关重要。然后，虹膜给出精致的提示，对产量进行评级并提高产量，以增加其危害性。我们发现IRIS在GPT-4上的越狱成功率为98%，在GPT-4 Turbo上的越狱成功率为92%。它在自动、黑盒和可解释越狱方面显著优于现有方法，同时需要的查询大大减少，从而建立了可解释越狱方法的新标准。



## **36. Rethinking Robustness Assessment: Adversarial Attacks on Learning-based Quadrupedal Locomotion Controllers**

重新思考稳健性评估：对基于学习的四足运动控制器的对抗攻击 cs.RO

RSS 2024

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2405.12424v1) [paper-pdf](http://arxiv.org/pdf/2405.12424v1)

**Authors**: Fan Shi, Chong Zhang, Takahiro Miki, Joonho Lee, Marco Hutter, Stelian Coros

**Abstract**: Legged locomotion has recently achieved remarkable success with the progress of machine learning techniques, especially deep reinforcement learning (RL). Controllers employing neural networks have demonstrated empirical and qualitative robustness against real-world uncertainties, including sensor noise and external perturbations. However, formally investigating the vulnerabilities of these locomotion controllers remains a challenge. This difficulty arises from the requirement to pinpoint vulnerabilities across a long-tailed distribution within a high-dimensional, temporally sequential space. As a first step towards quantitative verification, we propose a computational method that leverages sequential adversarial attacks to identify weaknesses in learned locomotion controllers. Our research demonstrates that, even state-of-the-art robust controllers can fail significantly under well-designed, low-magnitude adversarial sequence. Through experiments in simulation and on the real robot, we validate our approach's effectiveness, and we illustrate how the results it generates can be used to robustify the original policy and offer valuable insights into the safety of these black-box policies.

摘要: 近年来，随着机器学习技术的进步，特别是深度强化学习(RL)的发展，腿部运动已经取得了显著的成功。采用神经网络的控制器对真实世界的不确定性表现出了经验和定性的鲁棒性，包括传感器噪声和外部扰动。然而，正式调查这些运动控制器的漏洞仍然是一个挑战。这一困难源于需要在高维的、时间顺序的空间内精确定位跨长尾分布的漏洞。作为定量验证的第一步，我们提出了一种计算方法，该方法利用顺序对抗性攻击来识别学习的运动控制器中的弱点。我们的研究表明，即使是最先进的鲁棒控制器，在设计良好的低幅度对抗性序列下也会显著失效。通过仿真实验和在真实机器人上的实验，我们验证了该方法的有效性，并说明了它所产生的结果如何被用来证明原始策略的健壮性，并为这些黑盒策略的安全性提供了有价值的见解。



## **37. Rethinking PGD Attack: Is Sign Function Necessary?**

重新思考PVD攻击：符号功能是否必要？ cs.LG

**SubmitDate**: 2024-05-21    [abs](http://arxiv.org/abs/2312.01260v2) [paper-pdf](http://arxiv.org/pdf/2312.01260v2)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.

摘要: 神经网络已经在各个领域取得了成功，但即使是很小的输入扰动也会显著降低其性能。因此，这种被称为对抗性攻击的扰动的构造得到了极大的关注，其中许多都属于我们可以完全访问神经网络的“白盒”情景。现有的攻击算法，如投影梯度下降(PGD)算法，通常在更新敌方输入之前对原始梯度取符号函数，从而忽略了梯度大小信息。本文从理论上分析了这种基于符号的更新算法对分步攻击性能的影响，并给出了相应的警告。我们还解释了为什么以前直接使用原始梯度的尝试失败了。在此基础上，进一步提出了一种新的原始梯度下降(RGD)算法，该算法省去了符号的使用。具体地说，我们通过引入一个可以超越约束的非剪裁扰动的新的隐变量，将约束优化问题转化为无约束优化问题。所提出的RGD算法的有效性已经在实验中得到了广泛的证明，在不引起任何额外计算开销的情况下，在不同环境下的性能优于PGD和其他竞争对手。这些代码可以在https://github.com/JunjieYang97/RGD.中找到



## **38. Hacking Predictors Means Hacking Cars: Using Sensitivity Analysis to Identify Trajectory Prediction Vulnerabilities for Autonomous Driving Security**

黑客预测器意味着黑客汽车：使用敏感性分析来识别自动驾驶安全的轨迹预测漏洞 cs.CR

10 pages, 5 figures, 1 tables

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2401.10313v2) [paper-pdf](http://arxiv.org/pdf/2401.10313v2)

**Authors**: Marsalis Gibson, David Babazadeh, Claire Tomlin, Shankar Sastry

**Abstract**: Adversarial attacks on learning-based multi-modal trajectory predictors have already been demonstrated. However, there are still open questions about the effects of perturbations on inputs other than state histories, and how these attacks impact downstream planning and control. In this paper, we conduct a sensitivity analysis on two trajectory prediction models, Trajectron++ and AgentFormer. The analysis reveals that between all inputs, almost all of the perturbation sensitivities for both models lie only within the most recent position and velocity states. We additionally demonstrate that, despite dominant sensitivity on state history perturbations, an undetectable image map perturbation made with the Fast Gradient Sign Method can induce large prediction error increases in both models, revealing that these trajectory predictors are, in fact, susceptible to image-based attacks. Using an optimization-based planner and example perturbations crafted from sensitivity results, we show how these attacks can cause a vehicle to come to a sudden stop from moderate driving speeds.

摘要: 对基于学习的多模式轨迹预测器的对抗性攻击已经被证明。然而，对于扰动对除状态历史之外的输入的影响，以及这些攻击如何影响下游规划和控制，仍然存在悬而未决的问题。本文对两种弹道预测模型Trajectron++和AgentFormer进行了灵敏度分析。分析表明，在所有输入之间，两个模型的几乎所有摄动灵敏度都只存在于最近的位置和速度状态。此外，我们还证明了，尽管对状态历史扰动的主要敏感性，但用快速梯度符号方法进行的不可检测的图像映射扰动可以在两个模型中导致预测误差的大幅增加，这表明这些轨迹预测器实际上容易受到基于图像的攻击。使用基于优化的计划器和根据敏感度结果制作的示例扰动，我们展示了这些攻击如何导致车辆在中等速度下突然停止。



## **39. Optimizing Sensor Network Design for Multiple Coverage**

优化传感器网络设计以实现多覆盖 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.09096v2) [paper-pdf](http://arxiv.org/pdf/2405.09096v2)

**Authors**: Lukas Taus, Yen-Hsi Richard Tsai

**Abstract**: Sensor placement optimization methods have been studied extensively. They can be applied to a wide range of applications, including surveillance of known environments, optimal locations for 5G towers, and placement of missile defense systems. However, few works explore the robustness and efficiency of the resulting sensor network concerning sensor failure or adversarial attacks. This paper addresses this issue by optimizing for the least number of sensors to achieve multiple coverage of non-simply connected domains by a prescribed number of sensors. We introduce a new objective function for the greedy (next-best-view) algorithm to design efficient and robust sensor networks and derive theoretical bounds on the network's optimality. We further introduce a Deep Learning model to accelerate the algorithm for near real-time computations. The Deep Learning model requires the generation of training examples. Correspondingly, we show that understanding the geometric properties of the training data set provides important insights into the performance and training process of deep learning techniques. Finally, we demonstrate that a simple parallel version of the greedy approach using a simpler objective can be highly competitive.

摘要: 传感器布局优化方法得到了广泛的研究。它们可以应用于广泛的应用，包括对已知环境的监视，5G塔的最佳位置，以及导弹防御系统的布置。然而，很少有文献探讨传感器网络在传感器故障或敌意攻击下的健壮性和有效性。本文通过优化最少的传感器数量来解决这一问题，从而在规定的传感器数量下实现对非单连通区域的多次覆盖。为了设计高效、健壮的传感器网络，我们为贪婪(Next-Best-view)算法引入了一个新的目标函数，并给出了网络最优性的理论界。我们进一步引入了深度学习模型来加速算法，以实现近实时计算。深度学习模型需要生成训练实例。相应地，我们表明，理解训练数据集的几何属性可以为深度学习技术的性能和训练过程提供重要的见解。最后，我们证明了贪婪方法的简单并行版本使用更简单的目标可以具有很强的竞争力。



## **40. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的有效模型窃取攻击 cs.LG

arXiv admin note: text overlap with arXiv:2112.08331 by other authors

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12295v1) [paper-pdf](http://arxiv.org/pdf/2405.12295v1)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which enable the processing of graph-structured data without relying on predefined graph structures, are gaining importance in an increasingly wide variety of applications. As these networks demonstrate proficiency across a range of tasks, they become lucrative targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. A large effort has been made to develop model-stealing attacks that focus on models trained with images and texts. However, little attention has been paid to GNNs trained on graph data. This paper introduces a novel method for unsupervised model-stealing attacks against inductive GNNs, based on graph contrasting learning and spectral graph augmentations to efficiently extract information from the target model. The proposed attack is thoroughly evaluated on six datasets. The results show that this approach demonstrates a higher level of efficiency compared to existing stealing attacks. More concretely, our attack outperforms the baseline on all benchmarks achieving higher fidelity and downstream accuracy of the stolen model while requiring fewer queries sent to the target model.

摘要: 图神经网络(GNN)被认为是处理以图结构组织的真实世界数据的有力工具。尤其是感应式GNN，它能够在不依赖于预定义的图结构的情况下处理图结构的数据，在越来越广泛的应用中正变得越来越重要。由于这些网络在一系列任务中表现出熟练程度，它们成为窃取模型攻击的有利可图的目标，在这种攻击中，对手试图复制目标网络的功能。已经做出了大量努力来开发窃取模型的攻击，这些攻击集中在使用图像和文本训练的模型上。然而，对以图表数据为基础的全球网络的关注很少。提出了一种基于图对比学习和谱图扩充的非监督模型窃取方法，有效地从目标模型中提取信息。对提出的攻击在六个数据集上进行了彻底的评估。实验结果表明，与现有的窃取攻击相比，该方法具有更高的效率。更具体地说，我们的攻击在所有基准上都超过了基线，实现了被盗模型的更高保真度和下游精度，同时需要发送到目标模型的查询更少。



## **41. EGAN: Evolutional GAN for Ransomware Evasion**

EGAN：勒索软件规避的进化GAN cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12266v1) [paper-pdf](http://arxiv.org/pdf/2405.12266v1)

**Authors**: Daniel Commey, Benjamin Appiah, Bill K. Frimpong, Isaac Osei, Ebenezer N. A. Hammond, Garth V. Crosby

**Abstract**: Adversarial Training is a proven defense strategy against adversarial malware. However, generating adversarial malware samples for this type of training presents a challenge because the resulting adversarial malware needs to remain evasive and functional. This work proposes an attack framework, EGAN, to address this limitation. EGAN leverages an Evolution Strategy and Generative Adversarial Network to select a sequence of attack actions that can mutate a Ransomware file while preserving its original functionality. We tested this framework on popular AI-powered commercial antivirus systems listed on VirusTotal and demonstrated that our framework is capable of bypassing the majority of these systems. Moreover, we evaluated whether the EGAN attack framework can evade other commercial non-AI antivirus solutions. Our results indicate that the adversarial ransomware generated can increase the probability of evading some of them.

摘要: 对抗性训练是一种行之有效的针对对抗性恶意软件的防御策略。然而，为此类训练生成对抗性恶意软件样本存在挑战，因为产生的对抗性恶意软件需要保持规避性和功能性。这项工作提出了一个攻击框架EGAN来解决这一限制。EGAN利用进化策略和生成对抗网络来选择一系列攻击动作，这些动作可以变异勒索软件文件，同时保留其原始功能。我们在Virus Total上列出的流行人工智能驱动的商业防病毒系统上测试了这个框架，并证明我们的框架能够绕过大多数这些系统。此外，我们还评估了EGAN攻击框架是否可以规避其他商业非AI防病毒解决方案。我们的结果表明，生成的敌对勒索软件可以增加逃避其中一些勒索软件的可能性。



## **42. GAN-GRID: A Novel Generative Attack on Smart Grid Stability Prediction**

GAN-GRID：对智能电网稳定性预测的新型生成攻击 cs.CR

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.12076v1) [paper-pdf](http://arxiv.org/pdf/2405.12076v1)

**Authors**: Emad Efatinasab, Alessandro Brighente, Mirco Rampazzo, Nahal Azadi, Mauro Conti

**Abstract**: The smart grid represents a pivotal innovation in modernizing the electricity sector, offering an intelligent, digitalized energy network capable of optimizing energy delivery from source to consumer. It hence represents the backbone of the energy sector of a nation. Due to its central role, the availability of the smart grid is paramount and is hence necessary to have in-depth control of its operations and safety. To this aim, researchers developed multiple solutions to assess the smart grid's stability and guarantee that it operates in a safe state. Artificial intelligence and Machine learning algorithms have proven to be effective measures to accurately predict the smart grid's stability. Despite the presence of known adversarial attacks and potential solutions, currently, there exists no standardized measure to protect smart grids against this threat, leaving them open to new adversarial attacks. In this paper, we propose GAN-GRID a novel adversarial attack targeting the stability prediction system of a smart grid tailored to real-world constraints. Our findings reveal that an adversary armed solely with the stability model's output, devoid of data or model knowledge, can craft data classified as stable with an Attack Success Rate (ASR) of 0.99. Also by manipulating authentic data and sensor values, the attacker can amplify grid issues, potentially undetected due to a compromised stability prediction system. These results underscore the imperative of fortifying smart grid security mechanisms against adversarial manipulation to uphold system stability and reliability.

摘要: 智能电网代表着电力部门现代化的一项关键创新，提供了一个智能、数字化的能源网络，能够优化从来源到用户的能源输送。因此，它代表着一个国家能源部门的中坚力量。由于其核心作用，智能电网的可用性是至关重要的，因此有必要对其运行和安全进行深入控制。为此，研究人员开发了多种解决方案来评估智能电网的稳定性，并确保其在安全状态下运行。人工智能和机器学习算法已被证明是准确预测智能电网稳定性的有效手段。尽管存在已知的对抗性攻击和潜在的解决方案，但目前还没有标准化措施来保护智能电网免受这种威胁，使其容易受到新的对抗性攻击。在本文中，我们提出了一种新的针对智能电网稳定性预测系统的对抗性攻击。我们的发现表明，仅用稳定性模型的输出武装的对手，在缺乏数据或模型知识的情况下，可以伪造被归类为稳定的数据，攻击成功率(ASR)为0.99。此外，通过操纵真实的数据和传感器值，攻击者可以放大网格问题，这些问题可能由于稳定性预测系统受损而未被检测到。这些结果突显了加强智能电网安全机制以防止恶意操纵以维护系统稳定性和可靠性的必要性。



## **43. A Constraint-Enforcing Reward for Adversarial Attacks on Text Classifiers**

文本分类器对抗攻击的约束强制奖励 cs.CL

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11904v1) [paper-pdf](http://arxiv.org/pdf/2405.11904v1)

**Authors**: Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi

**Abstract**: Text classifiers are vulnerable to adversarial examples -- correctly-classified examples that are deliberately transformed to be misclassified while satisfying acceptability constraints. The conventional approach to finding adversarial examples is to define and solve a combinatorial optimisation problem over a space of allowable transformations. While effective, this approach is slow and limited by the choice of transformations. An alternate approach is to directly generate adversarial examples by fine-tuning a pre-trained language model, as is commonly done for other text-to-text tasks. This approach promises to be much quicker and more expressive, but is relatively unexplored. For this reason, in this work we train an encoder-decoder paraphrase model to generate a diverse range of adversarial examples. For training, we adopt a reinforcement learning algorithm and propose a constraint-enforcing reward that promotes the generation of valid adversarial examples. Experimental results over two text classification datasets show that our model has achieved a higher success rate than the original paraphrase model, and overall has proved more effective than other competitive attacks. Finally, we show how key design choices impact the generated examples and discuss the strengths and weaknesses of the proposed approach.

摘要: 文本分类器很容易受到对抗性示例的影响--正确分类的示例在满足可接受性约束的同时被故意转换为错误分类。寻找对抗性例子的传统方法是在允许变换的空间上定义和解决组合优化问题。虽然这种方法很有效，但速度很慢，而且受到转换选择的限制。另一种方法是通过微调预先训练的语言模型来直接生成对抗性示例，这是其他文本到文本任务的常见做法。这种方法承诺会更快、更有表现力，但相对来说还没有被探索过。为此，在这项工作中，我们训练一个编码器-解码者转述模型，以生成不同范围的对抗性例子。对于训练，我们采用了一种强化学习算法，并提出了一种约束强制奖励，以促进有效对抗性实例的生成。在两个文本分类数据集上的实验结果表明，我们的模型取得了比原始释义模型更高的成功率，总体上被证明比其他竞争攻击更有效。最后，我们展示了关键的设计选择如何影响生成的实例，并讨论了所提出的方法的优点和缺点。



## **44. Adversarially Diversified Rehearsal Memory (ADRM): Mitigating Memory Overfitting Challenge in Continual Learning**

敌对多元化排练记忆（ADRM）：缓解持续学习中的记忆过度匹配挑战 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11829v1) [paper-pdf](http://arxiv.org/pdf/2405.11829v1)

**Authors**: Hikmat Khan, Ghulam Rasool, Nidhal Carla Bouaynaya

**Abstract**: Continual learning focuses on learning non-stationary data distribution without forgetting previous knowledge. Rehearsal-based approaches are commonly used to combat catastrophic forgetting. However, these approaches suffer from a problem called "rehearsal memory overfitting, " where the model becomes too specialized on limited memory samples and loses its ability to generalize effectively. As a result, the effectiveness of the rehearsal memory progressively decays, ultimately resulting in catastrophic forgetting of the learned tasks.   We introduce the Adversarially Diversified Rehearsal Memory (ADRM) to address the memory overfitting challenge. This novel method is designed to enrich memory sample diversity and bolster resistance against natural and adversarial noise disruptions. ADRM employs the FGSM attacks to introduce adversarially modified memory samples, achieving two primary objectives: enhancing memory diversity and fostering a robust response to continual feature drifts in memory samples.   Our contributions are as follows: Firstly, ADRM addresses overfitting in rehearsal memory by employing FGSM to diversify and increase the complexity of the memory buffer. Secondly, we demonstrate that ADRM mitigates memory overfitting and significantly improves the robustness of CL models, which is crucial for safety-critical applications. Finally, our detailed analysis of features and visualization demonstrates that ADRM mitigates feature drifts in CL memory samples, significantly reducing catastrophic forgetting and resulting in a more resilient CL model. Additionally, our in-depth t-SNE visualizations of feature distribution and the quantification of the feature similarity further enrich our understanding of feature representation in existing CL approaches. Our code is publically available at https://github.com/hikmatkhan/ADRM.

摘要: 持续学习侧重于学习非平稳数据分布，而不会忘记先前的知识。基于排练的方法通常被用来对抗灾难性遗忘。然而，这些方法存在一个称为“预演记忆过度匹配”的问题，该模型对有限的记忆样本过于专门化，失去了有效推广的能力。结果，排练记忆的有效性逐渐衰退，最终导致对所学任务的灾难性遗忘。我们引入了对抗性多元化预演记忆(ADRM)来解决记忆过度匹配的挑战。这种新的方法旨在丰富记忆样本的多样性，并增强对自然和对抗性噪声干扰的抵抗力。ADRM利用FGSM攻击引入恶意修改的记忆样本，实现两个主要目标：增强记忆多样性和培养对记忆样本中持续特征漂移的稳健响应。我们的贡献如下：首先，ADRM通过使用FGSM来多样化和增加存储缓冲区的复杂度来解决预演记忆中的过度匹配问题。其次，我们证明了ADRM缓解了内存过度匹配，并显著提高了CL模型的健壮性，这对于安全关键型应用是至关重要的。最后，我们对特征和可视化的详细分析表明，ADRM减少了CL记忆样本中的特征漂移，显著减少了灾难性遗忘，并导致了更具弹性的CL模型。此外，我们对特征分布的深入t-SNE可视化和特征相似性的量化进一步丰富了我们对现有CL方法中的特征表示的理解。我们的代码在https://github.com/hikmatkhan/ADRM.上公开提供



## **45. Fed-Credit: Robust Federated Learning with Credibility Management**

Fed-Credit：具有可信度管理的稳健联邦学习 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11758v1) [paper-pdf](http://arxiv.org/pdf/2405.11758v1)

**Authors**: Jiayan Chen, Zhirong Qian, Tianhui Meng, Xitong Gao, Tian Wang, Weijia Jia

**Abstract**: Aiming at privacy preservation, Federated Learning (FL) is an emerging machine learning approach enabling model training on decentralized devices or data sources. The learning mechanism of FL relies on aggregating parameter updates from individual clients. However, this process may pose a potential security risk due to the presence of malicious devices. Existing solutions are either costly due to the use of compute-intensive technology, or restrictive for reasons of strong assumptions such as the prior knowledge of the number of attackers and how they attack. Few methods consider both privacy constraints and uncertain attack scenarios. In this paper, we propose a robust FL approach based on the credibility management scheme, called Fed-Credit. Unlike previous studies, our approach does not require prior knowledge of the nodes and the data distribution. It maintains and employs a credibility set, which weighs the historical clients' contributions based on the similarity between the local models and global model, to adjust the global model update. The subtlety of Fed-Credit is that the time decay and attitudinal value factor are incorporated into the dynamic adjustment of the reputation weights and it boasts a computational complexity of O(n) (n is the number of the clients). We conducted extensive experiments on the MNIST and CIFAR-10 datasets under 5 types of attacks. The results exhibit superior accuracy and resilience against adversarial attacks, all while maintaining comparatively low computational complexity. Among these, on the Non-IID CIFAR-10 dataset, our algorithm exhibited performance enhancements of 19.5% and 14.5%, respectively, in comparison to the state-of-the-art algorithm when dealing with two types of data poisoning attacks.

摘要: 针对隐私保护，联合学习(FL)是一种新兴的机器学习方法，能够在分散的设备或数据源上进行模型训练。FL的学习机制依赖于聚合来自单个客户端的参数更新。然而，由于恶意设备的存在，此过程可能会带来潜在的安全风险。现有的解决方案要么由于使用计算密集型技术而成本高昂，要么由于事先知道攻击者的数量及其攻击方式等强有力的假设而受到限制。很少有方法同时考虑隐私约束和不确定的攻击场景。在本文中，我们提出了一种基于可信度管理方案的稳健FL方法，称为FED-Credit。与以前的研究不同，我们的方法不需要节点和数据分布的先验知识。它维护并使用了一个可信度集合，该集合根据局部模型和全局模型之间的相似度来权衡历史客户的贡献，以调整全局模型的更新。FED-Credit的微妙之处在于，它将时间衰减和态度价值因素引入到声誉权重的动态调整中，其计算复杂度为O(N)(n为客户数量)。我们在MNIST和CIFAR-10数据集上对5种类型的攻击进行了广泛的实验。结果显示，在保持相对较低的计算复杂性的同时，对对手攻击表现出了卓越的准确性和弹性。其中，在非IID CIFAR-10数据集上，与现有算法相比，在处理两种类型的数据中毒攻击时，我们的算法分别表现出19.5%和14.5%的性能提升。



## **46. Towards Optimal Adversarial Robust Q-learning with Bellman Infinity-error**

采用Bellman无限误差实现最佳对抗鲁棒Q学习 cs.LG

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2402.02165v2) [paper-pdf](http://arxiv.org/pdf/2402.02165v2)

**Authors**: Haoran Li, Zicheng Zhang, Wang Luo, Congying Han, Yudong Hu, Tiande Guo, Shichen Liao

**Abstract**: Establishing robust policies is essential to counter attacks or disturbances affecting deep reinforcement learning (DRL) agents. Recent studies explore state-adversarial robustness and suggest the potential lack of an optimal robust policy (ORP), posing challenges in setting strict robustness constraints. This work further investigates ORP: At first, we introduce a consistency assumption of policy (CAP) stating that optimal actions in the Markov decision process remain consistent with minor perturbations, supported by empirical and theoretical evidence. Building upon CAP, we crucially prove the existence of a deterministic and stationary ORP that aligns with the Bellman optimal policy. Furthermore, we illustrate the necessity of $L^{\infty}$-norm when minimizing Bellman error to attain ORP. This finding clarifies the vulnerability of prior DRL algorithms that target the Bellman optimal policy with $L^{1}$-norm and motivates us to train a Consistent Adversarial Robust Deep Q-Network (CAR-DQN) by minimizing a surrogate of Bellman Infinity-error. The top-tier performance of CAR-DQN across various benchmarks validates its practical effectiveness and reinforces the soundness of our theoretical analysis.

摘要: 建立稳健的策略对于对抗影响深度强化学习(DRL)代理的攻击或干扰至关重要。最近的研究探索了状态对抗的健壮性，并表明可能缺乏最优的健壮性策略(ORP)，这给设置严格的健壮性约束带来了挑战。首先，我们引入了策略一致性假设(CAP)，指出马尔可夫决策过程中的最优行为在微小扰动下保持一致，并得到了经验和理论证据的支持。在CAP的基础上，我们关键地证明了与Bellman最优策略一致的确定性且平稳的ORP的存在。此外，我们还说明了在最小化Bellman误差以达到ORP时，$L^$-范数的必要性。这一发现澄清了以前以$L^{1}$范数为目标的Bellman最优策略的DRL算法的脆弱性，并激励我们通过最小化Bellman无穷错误的代理来训练一致的对抗性鲁棒深度Q-网络(CAR-DQN)。CAR-DQN在各种基准测试中的顶级性能验证了它的实际有效性，并加强了我们理论分析的合理性。



## **47. Adaptive Batch Normalization Networks for Adversarial Robustness**

对抗鲁棒性的自适应批量正规化网络 cs.LG

Accepted at IEEE International Conference on Advanced Video and  Signal-based Surveillance (AVSS) 2024

**SubmitDate**: 2024-05-20    [abs](http://arxiv.org/abs/2405.11708v1) [paper-pdf](http://arxiv.org/pdf/2405.11708v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Deep networks are vulnerable to adversarial examples. Adversarial Training (AT) has been a standard foundation of modern adversarial defense approaches due to its remarkable effectiveness. However, AT is extremely time-consuming, refraining it from wide deployment in practical applications. In this paper, we aim at a non-AT defense: How to design a defense method that gets rid of AT but is still robust against strong adversarial attacks? To answer this question, we resort to adaptive Batch Normalization (BN), inspired by the recent advances in test-time domain adaptation. We propose a novel defense accordingly, referred to as the Adaptive Batch Normalization Network (ABNN). ABNN employs a pre-trained substitute model to generate clean BN statistics and sends them to the target model. The target model is exclusively trained on clean data and learns to align the substitute model's BN statistics. Experimental results show that ABNN consistently improves adversarial robustness against both digital and physically realizable attacks on both image and video datasets. Furthermore, ABNN can achieve higher clean data performance and significantly lower training time complexity compared to AT-based approaches.

摘要: 深层网络很容易受到敌意例子的攻击。对抗训练(AT)因其显著的有效性而成为现代对抗防御方法的标准基础。然而，AT非常耗时，阻碍了它在实际应用中的广泛应用。在本文中，我们针对的是一种非AT防御：如何设计一种既能去除AT，又能对强对手攻击保持健壮性的防御方法？为了回答这个问题，我们求助于自适应批处理归一化(BN)，灵感来自于测试-时间域自适应的最新进展。因此，我们提出了一种新的防御方法，称为自适应批处理归一化网络(ABNN)。ABNN使用预先训练的替代模型来生成干净的BN统计数据，并将其发送到目标模型。目标模型专门接受关于干净数据的培训，并学习如何调整替代模型的BN统计数据。实验结果表明，ABNN在抵抗图像和视频数据集上的数字攻击和物理可实现攻击时，都一致地提高了对手的健壮性。此外，与基于AT的方法相比，ABNN可以获得更高的清洁数据性能和更低的训练时间复杂度。



## **48. Geometry-Aware Instrumental Variable Regression**

几何感知工具变量回归 cs.LG

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11633v1) [paper-pdf](http://arxiv.org/pdf/2405.11633v1)

**Authors**: Heiner Kremer, Bernhard Schölkopf

**Abstract**: Instrumental variable (IV) regression can be approached through its formulation in terms of conditional moment restrictions (CMR). Building on variants of the generalized method of moments, most CMR estimators are implicitly based on approximating the population data distribution via reweightings of the empirical sample. While for large sample sizes, in the independent identically distributed (IID) setting, reweightings can provide sufficient flexibility, they might fail to capture the relevant information in presence of corrupted data or data prone to adversarial attacks. To address these shortcomings, we propose the Sinkhorn Method of Moments, an optimal transport-based IV estimator that takes into account the geometry of the data manifold through data-derivative information. We provide a simple plug-and-play implementation of our method that performs on par with related estimators in standard settings but improves robustness against data corruption and adversarial attacks.

摘要: 工具变量（IV）回归可以通过条件矩限制（RCM）的公式来进行。在广义矩法的变体的基础上，大多数MCR估计量隐含地基于通过对经验样本的重新加权来逼近人口数据分布。虽然对于大样本量，在独立同分布（IID）设置中，重新加权可以提供足够的灵活性，但在存在损坏的数据或容易遭受对抗攻击的数据的情况下，它们可能无法捕获相关信息。为了解决这些缺点，我们提出了Sinkhorn矩法，这是一种基于传输的最佳IV估计器，它通过数据衍生信息考虑数据流的几何形状。我们提供了我们的方法的简单即插即用实现，其性能与标准设置中的相关估计器相同，但提高了针对数据损坏和对抗性攻击的鲁棒性。



## **49. Searching Realistic-Looking Adversarial Objects For Autonomous Driving Systems**

为自动驾驶系统搜索外观逼真的对抗对象 cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2405.11629v1) [paper-pdf](http://arxiv.org/pdf/2405.11629v1)

**Authors**: Shengxiang Sun, Shenzhe Zhu

**Abstract**: Numerous studies on adversarial attacks targeting self-driving policies fail to incorporate realistic-looking adversarial objects, limiting real-world applicability. Building upon prior research that facilitated the transition of adversarial objects from simulations to practical applications, this paper discusses a modified gradient-based texture optimization method to discover realistic-looking adversarial objects. While retaining the core architecture and techniques of the prior research, the proposed addition involves an entity termed the 'Judge'. This agent assesses the texture of a rendered object, assigning a probability score reflecting its realism. This score is integrated into the loss function to encourage the NeRF object renderer to concurrently learn realistic and adversarial textures. The paper analyzes four strategies for developing a robust 'Judge': 1) Leveraging cutting-edge vision-language models. 2) Fine-tuning open-sourced vision-language models. 3) Pretraining neurosymbolic systems. 4) Utilizing traditional image processing techniques. Our findings indicate that strategies 1) and 4) yield less reliable outcomes, pointing towards strategies 2) or 3) as more promising directions for future research.

摘要: 许多针对自动驾驶政策的对抗性攻击研究未能纳入看起来逼真的对抗性对象，从而限制了现实世界的适用性。在前人研究的基础上，讨论了一种改进的基于梯度的纹理优化方法，以发现外观逼真的对抗性对象。在保留先前研究的核心架构和技术的同时，拟议的增加涉及一个被称为“法官”的实体。该代理评估渲染对象的纹理，指定反映其真实感的概率分数。这个分数被集成到损失函数中，以鼓励NERF对象渲染器同时学习现实和对抗性纹理。本文分析了开发一个健壮的‘裁判’的四个策略：1)利用尖端的视觉语言模型。2)微调开源的视觉语言模型。3)训练前的神经象征系统。4)利用传统的图像处理技术。我们的发现表明，策略1)和4)产生的结果不太可靠，指出策略2)或3)是未来研究的更有前途的方向。



## **50. Struggle with Adversarial Defense? Try Diffusion**

与对抗性防御作斗争？尝试扩散 cs.CV

**SubmitDate**: 2024-05-19    [abs](http://arxiv.org/abs/2404.08273v3) [paper-pdf](http://arxiv.org/pdf/2404.08273v3)

**Authors**: Yujie Li, Yanbin Wang, Haitao Xu, Bin Liu, Jianguo Sun, Zhenhao Guo, Wenrui Ma

**Abstract**: Adversarial attacks induce misclassification by introducing subtle perturbations. Recently, diffusion models are applied to the image classifiers to improve adversarial robustness through adversarial training or by purifying adversarial noise. However, diffusion-based adversarial training often encounters convergence challenges and high computational expenses. Additionally, diffusion-based purification inevitably causes data shift and is deemed susceptible to stronger adaptive attacks. To tackle these issues, we propose the Truth Maximization Diffusion Classifier (TMDC), a generative Bayesian classifier that builds upon pre-trained diffusion models and the Bayesian theorem. Unlike data-driven classifiers, TMDC, guided by Bayesian principles, utilizes the conditional likelihood from diffusion models to determine the class probabilities of input images, thereby insulating against the influences of data shift and the limitations of adversarial training. Moreover, to enhance TMDC's resilience against more potent adversarial attacks, we propose an optimization strategy for diffusion classifiers. This strategy involves post-training the diffusion model on perturbed datasets with ground-truth labels as conditions, guiding the diffusion model to learn the data distribution and maximizing the likelihood under the ground-truth labels. The proposed method achieves state-of-the-art performance on the CIFAR10 dataset against heavy white-box attacks and strong adaptive attacks. Specifically, TMDC achieves robust accuracies of 82.81% against $l_{\infty}$ norm-bounded perturbations and 86.05% against $l_{2}$ norm-bounded perturbations, respectively, with $\epsilon=0.05$.

摘要: 对抗性攻击通过引入微妙的扰动来导致错误分类。近年来，扩散模型被应用到图像分类器中，通过对抗性训练或净化对抗性噪声来提高对抗性稳健性。然而，基于扩散的对抗性训练往往会遇到收敛挑战和较高的计算开销。此外，基于扩散的净化不可避免地会导致数据转移，并被认为容易受到更强的适应性攻击。为了解决这些问题，我们提出了真值最大化扩散分类器(TMDC)，这是一种生成式贝叶斯分类器，它建立在预先训练的扩散模型和贝叶斯定理的基础上。与数据驱动的分类器不同，TMDC在贝叶斯原理的指导下，利用扩散模型的条件似然来确定输入图像的类别概率，从而避免了数据迁移的影响和对抗性训练的限制。此外，为了增强TMDC对更强大的对手攻击的韧性，我们提出了一种扩散分类器的优化策略。该策略包括在扰动数据集上对扩散模型进行后训练，以地面真实标签为条件，引导扩散模型学习数据分布，并最大化地面真实标签下的似然。该方法在CIFAR10数据集上取得了较好的抗重白盒攻击和强自适应攻击的性能。具体地说，TMDC对$L范数有界摄动和L范数有界摄动的稳健精度分别为82.81%和86.05%，其中$epsilon=0.05$。



