# Latest Adversarial Attack Papers
**update at 2023-09-11 14:41:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Avoid Adversarial Adaption in Federated Learning by Multi-Metric Investigations**

通过多度量调查避免联合学习中的对抗性适应 cs.LG

25 pages, 14 figures, 23 tables, 11 equations

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2306.03600v2) [paper-pdf](http://arxiv.org/pdf/2306.03600v2)

**Authors**: Torsten Krauß, Alexandra Dmitrienko

**Abstract**: Federated Learning (FL) facilitates decentralized machine learning model training, preserving data privacy, lowering communication costs, and boosting model performance through diversified data sources. Yet, FL faces vulnerabilities such as poisoning attacks, undermining model integrity with both untargeted performance degradation and targeted backdoor attacks. Preventing backdoors proves especially challenging due to their stealthy nature.   Prominent mitigation techniques against poisoning attacks rely on monitoring certain metrics and filtering malicious model updates. While shown effective in evaluations, we argue that previous works didn't consider realistic real-world adversaries and data distributions. We define a new notion of strong adaptive adversaries, capable of adapting to multiple objectives simultaneously. Through extensive empirical tests, we show that existing defense methods can be easily circumvented in this adversary model. We also demonstrate, that existing defenses have limited effectiveness when no assumptions are made about underlying data distributions.   We introduce Metric-Cascades (MESAS), a novel defense method for more realistic scenarios and adversary models. MESAS employs multiple detection metrics simultaneously to identify poisoned model updates, creating a complex multi-objective optimization problem for adaptive attackers. In our extensive evaluation featuring nine backdoors and three datasets, MESAS consistently detects even strong adaptive attackers. Furthermore, MESAS outperforms existing defenses in distinguishing backdoors from data distribution-related distortions within and across clients. MESAS is the first defense robust against strong adaptive adversaries, effective in real-world data scenarios, with an average overhead of just 24.37 seconds.

摘要: 联合学习(FL)有助于分散机器学习模型的训练，保护数据隐私，降低通信成本，并通过多样化的数据源提高模型性能。然而，FL面临着中毒攻击、无目标性能降级和有针对性的后门攻击等漏洞，破坏了模型的完整性。事实证明，由于后门的隐蔽性，防止后门特别具有挑战性。针对中毒攻击的突出缓解技术依赖于监控某些指标和过滤恶意模型更新。虽然在评估中显示了有效性，但我们认为以前的工作没有考虑现实世界中的对手和数据分布。我们定义了一种新的概念，即强适应性对手，能够同时适应多个目标。通过广泛的实证测试，我们表明现有的防御方法可以很容易地在这个对手模型中被绕过。我们还证明，当没有对潜在的数据分布做出假设时，现有的防御措施的有效性有限。我们介绍了Metric-Cascade(MEAS)，这是一种新的防御方法，适用于更真实的场景和对手模型。MEAS同时使用多个检测指标来识别有毒的模型更新，为适应性攻击者创造了一个复杂的多目标优化问题。在我们广泛的评估中，包括九个后门和三个数据集，MEAS一致地检测到即使是强大的适应性攻击者。此外，MEAS在区分后门与客户内部和客户之间与数据分发相关的扭曲方面优于现有的防御措施。MEAS是第一个针对强大自适应对手的稳健防御，在真实数据场景中有效，平均开销仅为24.37秒。



## **2. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

19 pages, 5 figures; full version of the revised paper accepted at  ACM CCS 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2305.03626v2) [paper-pdf](http://arxiv.org/pdf/2305.03626v2)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on public datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, at the cost of an acceptable loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公共数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准的商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵抗躲避攻击，代价是在非对抗性环境中损失可接受的准确性。



## **3. FIVA: Facial Image and Video Anonymization and Anonymization Defense**

FIVA：人脸图像和视频的匿名化及匿名化防御 cs.CV

Accepted to ICCVW 2023 - DFAD 2023

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04228v1) [paper-pdf](http://arxiv.org/pdf/2309.04228v1)

**Authors**: Felix Rosberg, Eren Erdal Aksoy, Cristofer Englund, Fernando Alonso-Fernandez

**Abstract**: In this paper, we present a new approach for facial anonymization in images and videos, abbreviated as FIVA. Our proposed method is able to maintain the same face anonymization consistently over frames with our suggested identity-tracking and guarantees a strong difference from the original face. FIVA allows for 0 true positives for a false acceptance rate of 0.001. Our work considers the important security issue of reconstruction attacks and investigates adversarial noise, uniform noise, and parameter noise to disrupt reconstruction attacks. In this regard, we apply different defense and protection methods against these privacy threats to demonstrate the scalability of FIVA. On top of this, we also show that reconstruction attack models can be used for detection of deep fakes. Last but not least, we provide experimental results showing how FIVA can even enable face swapping, which is purely trained on a single target image.

摘要: 在本文中，我们提出了一种新的图像和视频中的人脸匿名方法，简称FIVA。我们提出的方法能够在使用我们建议的身份跟踪的帧中一致地保持相同的人脸匿名化，并保证与原始人脸有很大的不同。FIVA允许0个真阳性，错误接受率为0.001。我们的工作考虑了重构攻击的重要安全问题，并研究了对抗噪声、均匀噪声和参数噪声对重构攻击的干扰。在这方面，我们针对这些隐私威胁采用了不同的防御和保护方法，以展示FIVA的可扩展性。此外，我们还证明了重构攻击模型可以用于深度伪装的检测。最后但并非最不重要的是，我们提供了实验结果，展示了FIVA甚至可以实现人脸交换，这是纯粹针对单个目标图像进行训练的。



## **4. Counterfactual Explanations via Locally-guided Sequential Algorithmic Recourse**

基于局部引导的序贯算法资源的反事实解释 cs.LG

7 pages, 5 figures, 3 appendix pages

**SubmitDate**: 2023-09-08    [abs](http://arxiv.org/abs/2309.04211v1) [paper-pdf](http://arxiv.org/pdf/2309.04211v1)

**Authors**: Edward A. Small, Jeffrey N. Clark, Christopher J. McWilliams, Kacper Sokol, Jeffrey Chan, Flora D. Salim, Raul Santos-Rodriguez

**Abstract**: Counterfactuals operationalised through algorithmic recourse have become a powerful tool to make artificial intelligence systems explainable. Conceptually, given an individual classified as y -- the factual -- we seek actions such that their prediction becomes the desired class y' -- the counterfactual. This process offers algorithmic recourse that is (1) easy to customise and interpret, and (2) directly aligned with the goals of each individual. However, the properties of a "good" counterfactual are still largely debated; it remains an open challenge to effectively locate a counterfactual along with its corresponding recourse. Some strategies use gradient-driven methods, but these offer no guarantees on the feasibility of the recourse and are open to adversarial attacks on carefully created manifolds. This can lead to unfairness and lack of robustness. Other methods are data-driven, which mostly addresses the feasibility problem at the expense of privacy, security and secrecy as they require access to the entire training data set. Here, we introduce LocalFACE, a model-agnostic technique that composes feasible and actionable counterfactual explanations using locally-acquired information at each step of the algorithmic recourse. Our explainer preserves the privacy of users by only leveraging data that it specifically requires to construct actionable algorithmic recourse, and protects the model by offering transparency solely in the regions deemed necessary for the intervention.

摘要: 通过算法资源操作的反事实已成为使人工智能系统变得可解释的强大工具。在概念上，给出一个被归类为y的个体--事实--我们寻求行动，使他们的预测成为所需的类别y‘--反事实。这一过程提供了(1)易于定制和解释的算法资源，(2)直接与每个人的目标保持一致。然而，“好的”反事实的性质仍然在很大程度上存在争议；有效地确定反事实及其相应的追索权仍然是一项悬而未决的挑战。一些策略使用梯度驱动的方法，但这些方法不能保证这种方法的可行性，而且容易受到精心创建的流形的对抗性攻击。这可能会导致不公平和缺乏稳健性。其他方法是以数据为导向的，主要是以牺牲隐私、安全和保密性为代价来解决可行性问题，因为它们需要访问整个训练数据集。在这里，我们介绍LocalFACE，这是一种与模型无关的技术，它在算法资源的每个步骤使用本地获取的信息组成可行和可操作的反事实解释。我们的解释器通过只利用它特别需要的数据来保护用户的隐私，以构建可操作的算法资源，并通过仅在被认为对干预必要的区域提供透明度来保护模型。



## **5. Blades: A Unified Benchmark Suite for Byzantine Attacks and Defenses in Federated Learning**

刀片：联邦学习中拜占庭攻击和防御的统一基准套件 cs.CR

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2206.05359v2) [paper-pdf](http://arxiv.org/pdf/2206.05359v2)

**Authors**: Shenghui Li, Edith Ngai, Fanghua Ye, Li Ju, Tianru Zhang, Thiemo Voigt

**Abstract**: Federated learning (FL) facilitates distributed training across clients, safeguarding the privacy of their data. The inherent distributed structure of FL introduces vulnerabilities, especially from adversarial (Byzantine) clients aiming to skew local updates to their advantage. Despite the plethora of research focusing on Byzantine-resilient FL, the academic community has yet to establish a comprehensive benchmark suite, pivotal for impartial assessment and comparison of different techniques.   This paper investigates existing techniques in Byzantine-resilient FL and introduces an open-source benchmark suite for convenient and fair performance comparisons. Our investigation begins with a systematic study of Byzantine attack and defense strategies. Subsequently, we present \ours, a scalable, extensible, and easily configurable benchmark suite that supports researchers and developers in efficiently implementing and validating novel strategies against baseline algorithms in Byzantine-resilient FL. The design of \ours incorporates key characteristics derived from our systematic study, encompassing the attacker's capabilities and knowledge, defense strategy categories, and factors influencing robustness. Blades contains built-in implementations of representative attack and defense strategies and offers user-friendly interfaces for seamlessly integrating new ideas.

摘要: 联合学习(FL)促进了跨客户的分布式培训，保护了他们的数据隐私。FL固有的分布式结构引入了漏洞，特别是来自敌意(拜占庭)客户端的漏洞，旨在歪曲本地更新以利于其优势。尽管有太多关于拜占庭式外语的研究，但学术界还没有建立一个全面的基准套件，这是公正评估和比较不同技术的关键。本文研究了拜占庭弹性FL中的现有技术，并介绍了一个开源的基准测试套件，用于方便和公平地进行性能比较。我们的调查始于对拜占庭攻防战略的系统研究。随后，我们提出了一个可伸缩、可扩展且易于配置的基准测试套件，它支持研究人员和开发人员在拜占庭弹性FL中有效地实现和验证针对基线算法的新策略。我们的设计包含了来自我们系统研究的关键特征，包括攻击者的能力和知识、防御策略类别和影响健壮性的因素。Blade包含典型攻击和防御策略的内置实现，并提供用户友好的界面，以无缝集成新想法。



## **6. Node Injection for Class-specific Network Poisoning**

针对特定类别的网络中毒的节点注入 cs.LG

28 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2301.12277v2) [paper-pdf](http://arxiv.org/pdf/2301.12277v2)

**Authors**: Ansh Kumar Sharma, Rahul Kukreja, Mayank Kharbanda, Tanmoy Chakraborty

**Abstract**: Graph Neural Networks (GNNs) are powerful in learning rich network representations that aid the performance of downstream tasks. However, recent studies showed that GNNs are vulnerable to adversarial attacks involving node injection and network perturbation. Among these, node injection attacks are more practical as they don't require manipulation in the existing network and can be performed more realistically. In this paper, we propose a novel problem statement - a class-specific poison attack on graphs in which the attacker aims to misclassify specific nodes in the target class into a different class using node injection. Additionally, nodes are injected in such a way that they camouflage as benign nodes. We propose NICKI, a novel attacking strategy that utilizes an optimization-based approach to sabotage the performance of GNN-based node classifiers. NICKI works in two phases - it first learns the node representation and then generates the features and edges of the injected nodes. Extensive experiments and ablation studies on four benchmark networks show that NICKI is consistently better than four baseline attacking strategies for misclassifying nodes in the target class. We also show that the injected nodes are properly camouflaged as benign, thus making the poisoned graph indistinguishable from its clean version w.r.t various topological properties.

摘要: 图形神经网络(GNN)在学习丰富的网络表示方面功能强大，有助于下游任务的执行。然而，最近的研究表明，GNN很容易受到包括节点注入和网络扰动在内的对抗性攻击。其中，节点注入攻击更实用，因为它们不需要在现有网络中进行操作，并且可以更真实地执行。在本文中，我们提出了一种新的问题陈述--针对图的特定类的毒物攻击，攻击者的目的是通过节点注入将目标类中的特定节点错误地分类到不同的类中。此外，注入结节的方式是伪装成良性结节。我们提出了一种新的攻击策略Nicki，它利用基于优化的方法来破坏基于GNN的节点分类器的性能。Nicki分两个阶段工作--它首先学习节点表示，然后生成注入节点的特征和边。在四个基准网络上的大量实验和烧蚀研究表明，对于目标类中的节点误分类，Nicki一致优于四种基线攻击策略。我们还证明了注入的节点被适当伪装成良性的，从而使得中毒的图与其干净的版本无法区分各种拓扑性质。



## **7. Experimental Study of Adversarial Attacks on ML-based xApps in O-RAN**

O-Range环境下基于ML的xApp对抗性攻击实验研究 cs.NI

Accepted for Globecom 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03844v1) [paper-pdf](http://arxiv.org/pdf/2309.03844v1)

**Authors**: Naveen Naik Sapavath, Brian Kim, Kaushik Chowdhury, Vijay K Shah

**Abstract**: Open Radio Access Network (O-RAN) is considered as a major step in the evolution of next-generation cellular networks given its support for open interfaces and utilization of artificial intelligence (AI) into the deployment, operation, and maintenance of RAN. However, due to the openness of the O-RAN architecture, such AI models are inherently vulnerable to various adversarial machine learning (ML) attacks, i.e., adversarial attacks which correspond to slight manipulation of the input to the ML model. In this work, we showcase the vulnerability of an example ML model used in O-RAN, and experimentally deploy it in the near-real time (near-RT) RAN intelligent controller (RIC). Our ML-based interference classifier xApp (extensible application in near-RT RIC) tries to classify the type of interference to mitigate the interference effect on the O-RAN system. We demonstrate the first-ever scenario of how such an xApp can be impacted through an adversarial attack by manipulating the data stored in a shared database inside the near-RT RIC. Through a rigorous performance analysis deployed on a laboratory O-RAN testbed, we evaluate the performance in terms of capacity and the prediction accuracy of the interference classifier xApp using both clean and perturbed data. We show that even small adversarial attacks can significantly decrease the accuracy of ML application in near-RT RIC, which can directly impact the performance of the entire O-RAN deployment.

摘要: 开放式无线接入网(O-RAN)被认为是下一代蜂窝网络演进的重要一步，因为它支持开放接口，并利用人工智能(AI)来部署、运营和维护RAN。然而，由于O-RAN体系结构的开放性，这种人工智能模型天生就容易受到各种对抗性机器学习(ML)攻击，即对应于对ML模型的输入进行轻微操纵的对抗性攻击。在这项工作中，我们展示了一个用于O-RAN的示例ML模型的漏洞，并将其实验部署在近实时(Near-RT)RAN智能控制器(RIC)中。我们的基于ML的干扰分类器xApp(eXtensible Application in Near-RT RIC)试图对干扰类型进行分类，以减轻干扰对O-RAN系统的影响。我们通过操作存储在近RTRIC内的共享数据库中的数据，首次演示了这样的xApp如何通过敌意攻击受到影响的场景。通过在实验室O-RAN试验台上部署的严格性能分析，我们使用干净和扰动数据评估了干扰分类器xApp的容量和预测精度方面的性能。结果表明，即使是较小的敌意攻击也会显著降低ML在近RTRIC中应用的准确性，这将直接影响整个O-RAN部署的性能。



## **8. Adversarially Robust Deep Learning with Optimal-Transport-Regularized Divergences**

基于最优传输正则化发散的对抗性稳健深度学习 cs.LG

30 pages, 5 figures

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03791v1) [paper-pdf](http://arxiv.org/pdf/2309.03791v1)

**Authors**: Jeremiah Birrell, Mohammadreza Ebrahimi

**Abstract**: We introduce the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These methods are based on a new class of optimal-transport-regularized divergences, constructed via an infimal convolution between an information divergence and an optimal-transport (OT) cost. We use these as tools to enhance adversarial robustness by maximizing the expected loss over a neighborhood of distributions, a technique known as distributionally robust optimization. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence. We demonstrate the effectiveness of our method on malware detection and image recognition applications and find that, to our knowledge, it outperforms existing methods at enhancing the robustness against adversarial attacks. $ARMOR_D$ yields the robustified accuracy of $98.29\%$ against $FGSM$ and $98.18\%$ against $PGD^{40}$ on the MNIST dataset, reducing the error rate by more than $19.7\%$ and $37.2\%$ respectively compared to prior methods. Similarly, in malware detection, a discrete (binary) data domain, $ARMOR_D$ improves the robustified accuracy under $rFGSM^{50}$ attack compared to the previous best-performing adversarial training methods by $37.0\%$ while lowering false negative and false positive rates by $51.1\%$ and $57.53\%$, respectively.

摘要: 我们引入了$ARMOR_D$方法作为增强深度学习模型对抗性稳健性的新方法。这些方法是基于一类新的最优传输正则化发散，通过信息发散和最优传输(OT)成本之间的逐次卷积构造的。我们使用它们作为工具，通过最大化分布邻域的预期损失来增强对手的稳健性，这是一种称为分布稳健优化的技术。作为一种构建对抗性样本的工具，我们的方法允许样本既可以根据OT成本进行传输，又可以根据信息分歧进行重新加权。我们在恶意软件检测和图像识别应用中验证了该方法的有效性，并发现，据我们所知，该方法在增强对对手攻击的稳健性方面优于现有方法。在MNIST数据集上，$ARMOR_D$对$FGSM$和$PGD^{40}$的粗化精度分别为98.29美元和98.18美元，与以前的方法相比，错误率分别降低了19.7美元和37.2美元以上。类似地，在恶意软件检测方面，离散(二进制)数据域$ARMOR_D$在$rFGSM^{50}$攻击下，与以前性能最好的对抗性训练方法相比，提高了$rFGSM^{50}$攻击的粗暴准确率$37.0$，同时降低了漏检率和误警率分别为$51.1和$57.53$。



## **9. DiffDefense: Defending against Adversarial Attacks via Diffusion Models**

扩散防御：通过扩散模型防御敌意攻击 cs.LG

Paper published at ICIAP23

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03702v1) [paper-pdf](http://arxiv.org/pdf/2309.03702v1)

**Authors**: Hondamunige Prasanna Silva, Lorenzo Seidenari, Alberto Del Bimbo

**Abstract**: This paper presents a novel reconstruction method that leverages Diffusion Models to protect machine learning classifiers against adversarial attacks, all without requiring any modifications to the classifiers themselves. The susceptibility of machine learning models to minor input perturbations renders them vulnerable to adversarial attacks. While diffusion-based methods are typically disregarded for adversarial defense due to their slow reverse process, this paper demonstrates that our proposed method offers robustness against adversarial threats while preserving clean accuracy, speed, and plug-and-play compatibility. Code at: https://github.com/HondamunigePrasannaSilva/DiffDefence.

摘要: 本文提出了一种新的重建方法，该方法利用扩散模型来保护机器学习分类器免受对手攻击，而不需要对分类器本身进行任何修改。机器学习模型对微小输入扰动的敏感性使其容易受到敌意攻击。虽然基于扩散的方法由于其缓慢的反向过程而通常被忽略用于对抗性防御，但本文证明了我们提出的方法在保持干净的准确性、速度和即插即用兼容性的同时，对对抗性威胁提供了健壮性。代码：https://github.com/HondamunigePrasannaSilva/DiffDefence.



## **10. Improving Visual Quality and Transferability of Adversarial Attacks on Face Recognition Simultaneously with Adversarial Restoration**

在对抗性恢复的同时提高人脸识别对抗性攻击的视觉质量和可转移性 cs.CV

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.01582v2) [paper-pdf](http://arxiv.org/pdf/2309.01582v2)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Ping Li

**Abstract**: Adversarial face examples possess two critical properties: Visual Quality and Transferability. However, existing approaches rarely address these properties simultaneously, leading to subpar results. To address this issue, we propose a novel adversarial attack technique known as Adversarial Restoration (AdvRestore), which enhances both visual quality and transferability of adversarial face examples by leveraging a face restoration prior. In our approach, we initially train a Restoration Latent Diffusion Model (RLDM) designed for face restoration. Subsequently, we employ the inference process of RLDM to generate adversarial face examples. The adversarial perturbations are applied to the intermediate features of RLDM. Additionally, by treating RLDM face restoration as a sibling task, the transferability of the generated adversarial face examples is further improved. Our experimental results validate the effectiveness of the proposed attack method.

摘要: 对抗性人脸样例具有两个重要属性：视觉质量和可转移性。然而，现有的方法很少同时处理这些属性，导致结果低于平均水平。为了解决这个问题，我们提出了一种新的对抗性攻击技术，称为对抗性恢复(AdvRestore)，它通过利用事先的人脸恢复来提高对抗性人脸样本的视觉质量和可转移性。在我们的方法中，我们首先训练一个用于人脸恢复的恢复潜在扩散模型(RLDM)。随后，我们利用RLDM的推理过程来生成对抗性人脸样本。将对抗性扰动应用于RLDM的中间特征。此外，通过将RLDM人脸恢复视为兄弟任务，进一步提高了生成的对抗性人脸样本的可转移性。实验结果验证了该攻击方法的有效性。



## **11. Impact Sensitivity Analysis of Cooperative Adaptive Cruise Control Against Resource-Limited Adversaries**

协同自适应巡航控制对抗资源受限对手的撞击敏感性分析 eess.SY

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2304.02395v2) [paper-pdf](http://arxiv.org/pdf/2304.02395v2)

**Authors**: Mischa Huisman, Carlos Murguia, Erjen Lefeber, Nathan van de Wouw

**Abstract**: Cooperative Adaptive Cruise Control (CACC) is a technology that allows groups of vehicles to form in automated, tightly-coupled platoons. CACC schemes exploit Vehicle-to-Vehicle (V2V) wireless communications to exchange information between vehicles. However, the use of communication networks brings security concerns as it exposes network access points that the adversary can exploit to disrupt the vehicles' operation and even cause crashes. In this manuscript, we present a sensitivity analysis of CACC schemes against a class of resource-limited attacks. We present a modelling framework that allows us to systematically compute outer ellipsoidal approximations of reachable sets induced by attacks. We use the size of these sets as a security metric to quantify the potential damage of attacks affecting different signals in a CACC-controlled vehicle and study how two key system parameters change this metric. We carry out a sensitivity analysis for two different controller implementations (as given the available sensors there is an infinite number of realizations of the same controller) and show how different controller realizations can significantly affect the impact of attacks. We present extensive simulation experiments to illustrate the results.

摘要: 合作自适应巡航控制(CACC)是一种允许车辆组成自动、紧密耦合排的技术。CACC方案利用车辆对车辆(V2V)无线通信来在车辆之间交换信息。然而，通信网络的使用带来了安全问题，因为它暴露了网络接入点，对手可以利用这些接入点来扰乱车辆的操作，甚至导致撞车。在本文中，我们给出了CACC方案对一类资源受限攻击的敏感度分析。我们提出了一个模型框架，它允许我们系统地计算由攻击诱导的可达集的外椭球近似。我们使用这些集合的大小作为安全度量，以量化影响CACC控制的车辆中不同信号的攻击的潜在损害，并研究两个关键系统参数如何改变该度量。我们对两种不同的控制器实现进行了敏感度分析(在给定可用传感器的情况下，同一控制器有无限多个实现)，并展示了不同的控制器实现如何显著影响攻击的影响。我们给出了大量的模拟实验来说明结果。



## **12. How adversarial attacks can disrupt seemingly stable accurate classifiers**

敌意攻击如何扰乱看似稳定的准确分类器 cs.LG

11 pages, 8 figures, additional supplementary materials

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03665v1) [paper-pdf](http://arxiv.org/pdf/2309.03665v1)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Ivan Y. Tyukin, Alexander N. Gorban, Alexander Bastounis, Desmond J. Higham

**Abstract**: Adversarial attacks dramatically change the output of an otherwise accurate learning system using a seemingly inconsequential modification to a piece of input data. Paradoxically, empirical evidence indicates that even systems which are robust to large random perturbations of the input data remain susceptible to small, easily constructed, adversarial perturbations of their inputs. Here, we show that this may be seen as a fundamental feature of classifiers working with high dimensional input data. We introduce a simple generic and generalisable framework for which key behaviours observed in practical systems arise with high probability -- notably the simultaneous susceptibility of the (otherwise accurate) model to easily constructed adversarial attacks, and robustness to random perturbations of the input data. We confirm that the same phenomena are directly observed in practical neural networks trained on standard image classification problems, where even large additive random noise fails to trigger the adversarial instability of the network. A surprising takeaway is that even small margins separating a classifier's decision surface from training and testing data can hide adversarial susceptibility from being detected using randomly sampled perturbations. Counterintuitively, using additive noise during training or testing is therefore inefficient for eradicating or detecting adversarial examples, and more demanding adversarial training is required.

摘要: 对抗性攻击通过对一段输入数据进行看似无关紧要的修改，极大地改变了原本准确的学习系统的输出。矛盾的是，经验证据表明，即使是对输入数据的大随机扰动具有健壮性的系统，也仍然容易受到其输入的小的、容易构造的、对抗性的扰动。在这里，我们展示了这可以被视为使用高维输入数据的分类器的基本特征。我们引入了一个简单的通用和可推广的框架，对于该框架，在实际系统中观察到的关键行为以高概率出现-特别是(否则准确的)模型对容易构造的对抗性攻击的同时敏感性，以及对输入数据的随机扰动的稳健性。我们证实，在标准图像分类问题上训练的实际神经网络中也直接观察到了同样的现象，其中即使是较大的加性随机噪声也不能触发网络的对抗性不稳定性。令人惊讶的是，即使是将分类器的决策面与训练和测试数据分开的很小的边距，也可以隐藏对手的易感性，使其不会被随机抽样的扰动检测到。因此，与直觉相反的是，在训练或测试期间使用加性噪声对于消除或检测对抗性例子是低效的，并且需要更苛刻的对抗性训练。



## **13. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

Accepted in MM 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2305.09241v4) [paper-pdf](http://arxiv.org/pdf/2305.09241v4)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning. Our code is available at \url{https://github.com/jiangw-0/LE_JCDP}.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。我们的代码可在\url{https://github.com/jiangw-0/LE_JCDP}.



## **14. Password-Stealing without Hacking: Wi-Fi Enabled Practical Keystroke Eavesdropping**

无需黑客攻击即可窃取密码：支持Wi-Fi的实用击键窃听 cs.CR

14 pages, 27 figures, in ACM Conference on Computer and  Communications Security 2023

**SubmitDate**: 2023-09-07    [abs](http://arxiv.org/abs/2309.03492v1) [paper-pdf](http://arxiv.org/pdf/2309.03492v1)

**Authors**: Jingyang Hu, Hongbo Wang, Tianyue Zheng, Jingzhi Hu, Zhe Chen, Hongbo Jiang, Jun Luo

**Abstract**: The contact-free sensing nature of Wi-Fi has been leveraged to achieve privacy breaches, yet existing attacks relying on Wi-Fi CSI (channel state information) demand hacking Wi-Fi hardware to obtain desired CSIs. Since such hacking has proven prohibitively hard due to compact hardware, its feasibility in keeping up with fast-developing Wi-Fi technology becomes very questionable. To this end, we propose WiKI-Eve to eavesdrop keystrokes on smartphones without the need for hacking. WiKI-Eve exploits a new feature, BFI (beamforming feedback information), offered by latest Wi-Fi hardware: since BFI is transmitted from a smartphone to an AP in clear-text, it can be overheard (hence eavesdropped) by any other Wi-Fi devices switching to monitor mode. As existing keystroke inference methods offer very limited generalizability, WiKI-Eve further innovates in an adversarial learning scheme to enable its inference generalizable towards unseen scenarios. We implement WiKI-Eve and conduct extensive evaluation on it; the results demonstrate that WiKI-Eve achieves 88.9% inference accuracy for individual keystrokes and up to 65.8% top-10 accuracy for stealing passwords of mobile applications (e.g., WeChat).

摘要: Wi-Fi的免接触传感特性已被用来实现隐私泄露，但依赖Wi-Fi CSI(通道状态信息)的现有攻击需要黑客攻击Wi-Fi硬件以获得所需的CSI。由于硬件紧凑，此类黑客攻击已被证明非常困难，因此其跟上快速发展的Wi-Fi技术的可行性变得非常值得怀疑。为此，我们建议Wiki-Eve在不需要黑客的情况下窃听智能手机上的按键。Wiki-Eve利用了最新Wi-Fi硬件提供的新功能BFI(波束形成反馈信息)：由于BFI以明文形式从智能手机传输到AP，因此它可以被切换到监控模式的任何其他Wi-Fi设备窃听(因此被窃听)。由于现有的击键推理方法提供的泛化能力非常有限，Wiki-Eve进一步创新了对抗性学习方案，使其推理能够针对未知场景进行泛化。我们实现了Wiki-Eve并对其进行了广泛的评估，结果表明，Wiki-Eve对单个击键的推理准确率为88.9%，对于移动应用(如微信)的密码窃取前10名的准确率高达65.8%。



## **15. Cyber Recovery from Dynamic Load Altering Attacks: Linking Electricity, Transportation, and Cyber Networks**

动态负载变化攻击的网络恢复：连接电力、交通和网络 eess.SY

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03380v1) [paper-pdf](http://arxiv.org/pdf/2309.03380v1)

**Authors**: Mengxiang Liu, Zhongda Chu, Fei Teng

**Abstract**: To address the increasing vulnerability of power grids, significant attention has been focused on the attack detection and impact mitigation. However, it is still unclear how to effectively and quickly recover the cyber and physical networks from a cyberattack. In this context, this paper presents the first investigation of the Cyber Recovery from Dynamic load altering Attack (CRDA). Considering the interconnection among electricity, transportation, and cyber networks, two essential sub-tasks are formulated for the CRDA: i) Optimal design of repair crew routes to remove installed malware and ii) Adaptive adjustment of system operation to eliminate the mitigation costs while guaranteeing stability. To achieve this, linear stability constraints are obtained by estimating the related eigenvalues under the variation of multiple IBR droop gains based on the sensitivity information of strategically selected sampling points. Moreover, to obtain the robust recovery strategy, the potential counter-measures from the adversary during the recovery process are modeled as maximizing the attack impact of remaining compromised resources in each step. A Mixed-Integer Linear Programming (MILP) problem can be finally formulated for the CRDA with the primary objective to reset involved droop gains and secondarily to repair all compromised loads. Case studies are performed in the modified IEEE 39-bus power system to illustrate the effectiveness of the proposed CRDA compared to the benchmark case.

摘要: 为了解决电网日益增长的脆弱性，攻击检测和影响缓解已经引起了极大的关注。然而，目前仍不清楚如何有效、快速地从网络攻击中恢复网络和物理网络。在此背景下，本文首次对动态负载变化攻击的网络恢复进行了研究。考虑到电力、交通和网络之间的互联，CRDA制定了两个必要的子任务：i)优化设计维修人员路线以删除安装的恶意软件；ii)自适应调整系统运行，在保证稳定性的同时消除缓解成本。为了实现这一点，基于策略选择的采样点的灵敏度信息，通过估计多个IBR下垂增益变化下的相关特征值来获得线性稳定性约束。此外，为了获得稳健的恢复策略，在恢复过程中来自对手的潜在对抗措施被建模为最大化每一步中剩余受损资源的攻击影响。对于CRDA，最终可以建立一个混合整数线性规划(MILP)问题，主要目标是重置涉及的下垂增益，其次是修复所有受损的负载。以IEEE 39节点电力系统为例进行了算例分析，并与基准算例进行了比较，验证了该方法的有效性。



## **16. The Space of Adversarial Strategies**

对抗性策略的空间 cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2209.04521v2) [paper-pdf](http://arxiv.org/pdf/2209.04521v2)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstract**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.

摘要: 对抗性例子是机器学习模型中旨在诱导最坏情况行为的输入，在过去的十年里得到了广泛的研究。然而，我们对这一现象的理解源于相当零散的知识池；目前，有几种攻击，每一种攻击在威胁模型中都有不同的假设，对最优的定义也是无与伦比的。在本文中，我们提出了一种系统的方法来刻画最坏情况(即最佳)对手的特征。我们首先介绍了对抗性机器学习中攻击的一种可扩展分解，将攻击组件原子化到表面和旅行者中。通过我们的分解，我们列举了组件以创建576次攻击(其中568次是以前未曾探索过的)。接下来，我们提出了Pareto系综攻击(PEA)：一种上界攻击性能的理论攻击。在我们的新攻击中，我们在以下方面衡量相对于PEA的性能：稳健和非稳健模型、七个数据集和三个包含计算成本的扩展的基于LP的威胁模型，正式确定了对手战略空间。从我们的评估中，我们发现攻击性能与上下文高度相关：域、模型健壮性和威胁模型可以对攻击效率产生深远的影响。我们的调查表明，未来衡量机器学习安全性的研究应该：(1)从域和威胁模型出发，(2)超越目前使用的少数已知攻击。



## **17. My Art My Choice: Adversarial Protection Against Unruly AI**

我的艺术我的选择：对抗不守规矩的人工智能 cs.CV

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03198v1) [paper-pdf](http://arxiv.org/pdf/2309.03198v1)

**Authors**: Anthony Rhodes, Ram Bhagat, Umur Aybars Ciftci, Ilke Demir

**Abstract**: Generative AI is on the rise, enabling everyone to produce realistic content via publicly available interfaces. Especially for guided image generation, diffusion models are changing the creator economy by producing high quality low cost content. In parallel, artists are rising against unruly AI, since their artwork are leveraged, distributed, and dissimulated by large generative models. Our approach, My Art My Choice (MAMC), aims to empower content owners by protecting their copyrighted materials from being utilized by diffusion models in an adversarial fashion. MAMC learns to generate adversarially perturbed "protected" versions of images which can in turn "break" diffusion models. The perturbation amount is decided by the artist to balance distortion vs. protection of the content. MAMC is designed with a simple UNet-based generator, attacking black box diffusion models, combining several losses to create adversarial twins of the original artwork. We experiment on three datasets for various image-to-image tasks, with different user control values. Both protected image and diffusion output results are evaluated in visual, noise, structure, pixel, and generative spaces to validate our claims. We believe that MAMC is a crucial step for preserving ownership information for AI generated content in a flawless, based-on-need, and human-centric way.

摘要: 生成性人工智能正在崛起，使每个人都可以通过公开可用的界面来制作逼真的内容。特别是对于引导图像的生成，传播模式正在通过生产高质量、低成本的内容来改变创作者的经济。与此同时，艺术家们正在崛起，反对不守规矩的人工智能，因为他们的艺术作品是通过大型生成模型来利用、分发和伪装的。我们的方法，我的艺术我的选择(MAMC)，旨在通过保护他们的受版权保护的材料不被传播模型以对抗的方式利用来赋予内容所有者权力。MAMC学会了生成受到恶意干扰的“受保护”版本的图像，这反过来又可以“破坏”扩散模型。扰动量由艺术家决定，以平衡失真和对内容的保护。MAMC是用一个简单的基于UNT的生成器设计的，攻击黑盒扩散模型，结合几个损失来创建原始艺术品的对抗性孪生兄弟。我们在三个数据集上对不同的图像到图像任务进行了实验，这些数据集具有不同的用户控制值。在视觉、噪声、结构、像素和生成空间中对受保护图像和扩散输出结果进行了评估，以验证我们的主张。我们认为，MAMC是以完美、基于需求和以人为中心的方式保存人工智能生成内容的所有权信息的关键一步。



## **18. J-Guard: Journalism Guided Adversarially Robust Detection of AI-generated News**

J-Guard：新闻引导的恶意稳健检测人工智能生成的新闻 cs.CL

This Paper is Accepted to The 13th International Joint Conference on  Natural Language Processing and the 3rd Conference of the Asia-Pacific  Chapter of the Association for Computational Linguistics (IJCNLP-AACL 2023)

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.03164v1) [paper-pdf](http://arxiv.org/pdf/2309.03164v1)

**Authors**: Tharindu Kumarage, Amrita Bhattacharjee, Djordje Padejski, Kristy Roschke, Dan Gillmor, Scott Ruston, Huan Liu, Joshua Garland

**Abstract**: The rapid proliferation of AI-generated text online is profoundly reshaping the information landscape. Among various types of AI-generated text, AI-generated news presents a significant threat as it can be a prominent source of misinformation online. While several recent efforts have focused on detecting AI-generated text in general, these methods require enhanced reliability, given concerns about their vulnerability to simple adversarial attacks. Furthermore, due to the eccentricities of news writing, applying these detection methods for AI-generated news can produce false positives, potentially damaging the reputation of news organizations. To address these challenges, we leverage the expertise of an interdisciplinary team to develop a framework, J-Guard, capable of steering existing supervised AI text detectors for detecting AI-generated news while boosting adversarial robustness. By incorporating stylistic cues inspired by the unique journalistic attributes, J-Guard effectively distinguishes between real-world journalism and AI-generated news articles. Our experiments on news articles generated by a vast array of AI models, including ChatGPT (GPT3.5), demonstrate the effectiveness of J-Guard in enhancing detection capabilities while maintaining an average performance decrease of as low as 7% when faced with adversarial attacks.

摘要: 在线人工智能生成的文本的迅速激增正在深刻地重塑信息格局。在各种类型的人工智能生成的文本中，人工智能生成的新闻构成了一个重大威胁，因为它可能是网上错误信息的主要来源。虽然最近的几项努力主要集中在检测人工智能生成的文本，但这些方法需要增强可靠性，因为人们担心它们容易受到简单的对手攻击。此外，由于新闻写作的古怪，将这些检测方法应用于人工智能生成的新闻可能会产生假阳性，可能会损害新闻机构的声誉。为了应对这些挑战，我们利用一个跨学科团队的专业知识来开发一个名为J-Guard的框架，该框架能够指导现有的监督AI文本检测器检测AI生成的新闻，同时提高对手的健壮性。通过融入受独特新闻属性启发的文体线索，J-Guard有效地区分了真实世界的新闻和人工智能生成的新闻文章。我们对包括ChatGPT(GPT3.5)在内的大量人工智能模型生成的新闻文章进行了实验，证明了J-Guard在增强检测能力方面的有效性，同时在面对对手攻击时保持了低至7%的平均性能下降。



## **19. Defense-Prefix for Preventing Typographic Attacks on CLIP**

防御-用于防止对剪辑进行排版攻击的前缀 cs.CV

ICCV2023 Workshop

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2304.04512v3) [paper-pdf](http://arxiv.org/pdf/2304.04512v3)

**Authors**: Hiroki Azuma, Yusuke Matsui

**Abstract**: Vision-language pre-training models (VLPs) have exhibited revolutionary improvements in various vision-language tasks. In VLP, some adversarial attacks fool a model into false or absurd classifications. Previous studies addressed these attacks by fine-tuning the model or changing its architecture. However, these methods risk losing the original model's performance and are difficult to apply to downstream tasks. In particular, their applicability to other tasks has not been considered. In this study, we addressed the reduction of the impact of typographic attacks on CLIP without changing the model parameters. To achieve this, we expand the idea of "prefix learning" and introduce our simple yet effective method: Defense-Prefix (DP), which inserts the DP token before a class name to make words "robust" against typographic attacks. Our method can be easily applied to downstream tasks, such as object detection, because the proposed method is independent of the model parameters. Our method significantly improves the accuracy of classification tasks for typographic attack datasets, while maintaining the zero-shot capabilities of the model. In addition, we leverage our proposed method for object detection, demonstrating its high applicability and effectiveness. The codes and datasets are available at https://github.com/azuma164/Defense-Prefix.

摘要: 视觉语言预训练模型(VLP)在各种视觉语言任务中表现出革命性的改进。在VLP中，一些对抗性攻击欺骗模型进行错误或荒谬的分类。以前的研究通过微调模型或更改其体系结构来解决这些攻击。然而，这些方法可能会失去原始模型的性能，并且很难应用于下游任务。特别是，它们对其他任务的适用性没有得到考虑。在这项研究中，我们解决了在不改变模型参数的情况下减少排版攻击对CLIP的影响。为了实现这一点，我们扩展了前缀学习的思想，并引入了我们简单但有效的方法：防御前缀(DP)，它在类名之前插入DP标记，以使单词对排版攻击具有健壮性。我们的方法可以很容易地应用于下游任务，如目标检测，因为所提出的方法与模型参数无关。我们的方法显著地提高了排版攻击数据集的分类任务的准确性，同时保持了模型的零命中能力。此外，我们利用我们提出的方法进行目标检测，证明了其高度的适用性和有效性。代码和数据集可在https://github.com/azuma164/Defense-Prefix.上获得



## **20. Enhancing Adversarial Attacks: The Similar Target Method**

加强对抗性攻击：相似靶法 cs.CV

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2308.10743v2) [paper-pdf](http://arxiv.org/pdf/2308.10743v2)

**Authors**: Shuo Zhang, Ziruo Wang, Zikai Zhou, Huanran Chen

**Abstract**: Deep neural networks are vulnerable to adversarial examples, posing a threat to the models' applications and raising security concerns. An intriguing property of adversarial examples is their strong transferability. Several methods have been proposed to enhance transferability, including ensemble attacks which have demonstrated their efficacy. However, prior approaches simply average logits, probabilities, or losses for model ensembling, lacking a comprehensive analysis of how and why model ensembling significantly improves transferability. In this paper, we propose a similar targeted attack method named Similar Target~(ST). By promoting cosine similarity between the gradients of each model, our method regularizes the optimization direction to simultaneously attack all surrogate models. This strategy has been proven to enhance generalization ability. Experimental results on ImageNet validate the effectiveness of our approach in improving adversarial transferability. Our method outperforms state-of-the-art attackers on 18 discriminative classifiers and adversarially trained models.

摘要: 深度神经网络很容易受到敌意例子的攻击，这对模型的应用构成了威胁，并引发了安全担忧。对抗性例子的一个耐人寻味的特点是它们具有很强的可转移性。已经提出了几种提高可转移性的方法，包括已经证明其有效性的集合攻击。然而，以前的方法只是对模型集成的对数、概率或损失进行平均，缺乏对模型集成如何以及为什么显著提高可转移性的全面分析。本文提出了一种类似的目标攻击方法--相似目标~(ST)。通过提高每个模型梯度之间的余弦相似度，我们的方法将优化方向正则化以同时攻击所有代理模型。实践证明，该策略提高了泛化能力。在ImageNet上的实验结果验证了该方法在提高对手可转移性方面的有效性。我们的方法在18个区分分类器和对抗性训练的模型上优于最先进的攻击者。



## **21. Efficient Query-Based Attack against ML-Based Android Malware Detection under Zero Knowledge Setting**

零知识环境下基于ML的Android恶意软件检测的高效查询攻击 cs.CR

To Appear in the ACM Conference on Computer and Communications  Security, November, 2023

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.01866v2) [paper-pdf](http://arxiv.org/pdf/2309.01866v2)

**Authors**: Ping He, Yifan Xia, Xuhong Zhang, Shouling Ji

**Abstract**: The widespread adoption of the Android operating system has made malicious Android applications an appealing target for attackers. Machine learning-based (ML-based) Android malware detection (AMD) methods are crucial in addressing this problem; however, their vulnerability to adversarial examples raises concerns. Current attacks against ML-based AMD methods demonstrate remarkable performance but rely on strong assumptions that may not be realistic in real-world scenarios, e.g., the knowledge requirements about feature space, model parameters, and training dataset. To address this limitation, we introduce AdvDroidZero, an efficient query-based attack framework against ML-based AMD methods that operates under the zero knowledge setting. Our extensive evaluation shows that AdvDroidZero is effective against various mainstream ML-based AMD methods, in particular, state-of-the-art such methods and real-world antivirus solutions.

摘要: Android操作系统的广泛采用使恶意Android应用程序成为攻击者的诱人目标。基于机器学习(ML-Based)的Android恶意软件检测(AMD)方法对于解决这一问题至关重要；然而，它们对敌意例子的脆弱性引起了人们的担忧。目前针对基于ML的AMD方法的攻击表现出显著的性能，但依赖于强假设，这些假设在现实场景中可能不现实，例如关于特征空间、模型参数和训练数据集的知识要求。为了解决这一局限性，我们引入了AdvDroidZero，这是一个针对基于ML的AMD方法的高效的基于查询的攻击框架，它运行在零知识环境下。我们的广泛评估表明，AdvDroidZero对各种主流的基于ML的AMD方法，特别是最先进的此类方法和现实世界的反病毒解决方案都是有效的。



## **22. SWAP: Exploiting Second-Ranked Logits for Adversarial Attacks on Time Series**

SWAP：利用第二级日志对时间序列进行对抗性攻击 cs.LG

10 pages, 8 figures

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02752v1) [paper-pdf](http://arxiv.org/pdf/2309.02752v1)

**Authors**: Chang George Dong, Liangwei Nathan Zheng, Weitong Chen, Wei Emma Zhang, Lin Yue

**Abstract**: Time series classification (TSC) has emerged as a critical task in various domains, and deep neural models have shown superior performance in TSC tasks. However, these models are vulnerable to adversarial attacks, where subtle perturbations can significantly impact the prediction results. Existing adversarial methods often suffer from over-parameterization or random logit perturbation, hindering their effectiveness. Additionally, increasing the attack success rate (ASR) typically involves generating more noise, making the attack more easily detectable. To address these limitations, we propose SWAP, a novel attacking method for TSC models. SWAP focuses on enhancing the confidence of the second-ranked logits while minimizing the manipulation of other logits. This is achieved by minimizing the Kullback-Leibler divergence between the target logit distribution and the predictive logit distribution. Experimental results demonstrate that SWAP achieves state-of-the-art performance, with an ASR exceeding 50% and an 18% increase compared to existing methods.

摘要: 时间序列分类(TSC)已经成为各个领域的一项关键任务，而深度神经模型在TSC任务中表现出了优越的性能。然而，这些模型容易受到敌意攻击，其中细微的扰动会显著影响预测结果。现有的对抗性方法经常受到过度参数化或随机Logit扰动的影响，阻碍了它们的有效性。此外，提高攻击成功率(ASR)通常涉及生成更多噪声，从而使攻击更容易被检测到。针对这些局限性，我们提出了一种新的针对TSC模型的攻击方法--SWAP。SWAP侧重于增强排名第二的逻辑的可信度，同时最大限度地减少对其他逻辑的操纵。这是通过最小化目标Logit分布和预测Logit分布之间的Kullback-Leibler发散来实现的。实验结果表明，SWAP达到了最好的性能，ASR超过50%，与现有方法相比提高了18%。



## **23. Certifying LLM Safety against Adversarial Prompting**

针对敌意提示认证LLM安全 cs.CL

**SubmitDate**: 2023-09-06    [abs](http://arxiv.org/abs/2309.02705v1) [paper-pdf](http://arxiv.org/pdf/2309.02705v1)

**Authors**: Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Soheil Feizi, Hima Lakkaraju

**Abstract**: Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial sequence at the end of the prompt; ii) adversarial insertion, where the adversarial sequence is inserted anywhere in the middle of the prompt; and iii) adversarial infusion, where adversarial tokens are inserted at arbitrary positions in the prompt, not necessarily as a contiguous block. Empirical results demonstrate that our technique obtains strong certified safety guarantees on harmful prompts while maintaining good performance on safe prompts. For example, against adversarial suffixes of length 20, it certifiably detects 93% of the harmful prompts and labels 94% of the safe prompts as safe using the open source language model Llama 2 as the safety filter.

摘要: 发布给公众使用的大型语言模型(LLM)包括护栏，以确保其输出是安全的，通常被称为“模型对齐”。统一的语言模型应该拒绝用户制作有害内容的请求。然而，这样的安全措施很容易受到敌意提示的攻击，这些提示包含恶意设计的令牌序列，以绕过模型的安全警卫，使其产生有害内容。在这项工作中，我们引入了Erase-and-Check，这是第一个通过可验证的安全保证来防御敌意提示的框架。我们逐个擦除令牌，并使用安全过滤器检查得到的子序列。如果过滤器检测到任何子序列或输入提示有害，则我们的过程将输入提示标记为有害。这保证了对有害提示的任何敌意修改达到一定的大小也被标记为有害的。我们防御三种攻击模式：i)对抗性后缀，其在提示的末尾附加对抗性序列；ii)对抗性插入，其中对抗性序列被插入在提示中间的任何位置；以及iii)对抗性注入，其中对抗性标记被插入在提示中的任意位置，而不一定作为连续的块。实验结果表明，该技术在保证安全提示性能的同时，对有害提示提供了较强的认证安全保障。例如，对于长度为20的敌意后缀，它使用开源语言模型Llama 2作为安全过滤器，可证明检测到93%的有害提示和94%的安全提示是安全的。



## **24. Experimental quantum key distribution certified by Bell's theorem**

贝尔定理证明的实验性量子密钥分配 quant-ph

5+1 pages in main text and methods with 4 figures and 1 table; 42  pages of supplementary material (replaced with revision accepted for  publication in Nature; original title: "Device-Independent Quantum Key  Distribution")

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2109.14600v2) [paper-pdf](http://arxiv.org/pdf/2109.14600v2)

**Authors**: D. P. Nadlinger, P. Drmota, B. C. Nichol, G. Araneda, D. Main, R. Srinivas, D. M. Lucas, C. J. Ballance, K. Ivanov, E. Y-Z. Tan, P. Sekatski, R. L. Urbanke, R. Renner, N. Sangouard, J-D. Bancal

**Abstract**: Cryptographic key exchange protocols traditionally rely on computational conjectures such as the hardness of prime factorisation to provide security against eavesdropping attacks. Remarkably, quantum key distribution protocols like the one proposed by Bennett and Brassard provide information-theoretic security against such attacks, a much stronger form of security unreachable by classical means. However, quantum protocols realised so far are subject to a new class of attacks exploiting implementation defects in the physical devices involved, as demonstrated in numerous ingenious experiments. Following the pioneering work of Ekert proposing the use of entanglement to bound an adversary's information from Bell's theorem, we present here the experimental realisation of a complete quantum key distribution protocol immune to these vulnerabilities. We achieve this by combining theoretical developments on finite-statistics analysis, error correction, and privacy amplification, with an event-ready scheme enabling the rapid generation of high-fidelity entanglement between two trapped-ion qubits connected by an optical fibre link. The secrecy of our key is guaranteed device-independently: it is based on the validity of quantum theory, and certified by measurement statistics observed during the experiment. Our result shows that provably secure cryptography with real-world devices is possible, and paves the way for further quantum information applications based on the device-independence principle.

摘要: 密码密钥交换协议传统上依赖于计算猜想，例如素数分解的难度，以提供针对窃听攻击的安全性。值得注意的是，像Bennett和Brassard提出的量子密钥分发协议提供了针对此类攻击的信息论安全，这是一种传统方法无法达到的更强大的安全形式。然而，迄今为止实现的量子协议受到一类新的攻击，利用所涉及的物理设备中的实现缺陷，正如许多巧妙的实验所证明的那样。在Ekert的开创性工作提出使用纠缠将对手的信息从Bell定理绑定之后，我们在这里提出了一个完整的量子密钥分配协议的实验实现，该协议不受这些漏洞的影响。我们通过将有限统计分析、纠错和隐私放大方面的理论发展与事件就绪方案相结合来实现这一点，该方案能够在通过光纤连接的两个俘获离子量子比特之间快速产生高保真纠缠。我们的密钥的保密性是与设备无关的：它基于量子理论的有效性，并通过在实验中观察到的测量统计数据进行验证。我们的结果表明，使用真实设备进行可证明安全的密码学是可能的，并为基于设备无关原理的进一步量子信息应用铺平了道路。



## **25. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

图神经网络图分类中的敌意攻击再探 cs.SI

13 pages, 7 figures

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2208.06651v2) [paper-pdf](http://arxiv.org/pdf/2208.06651v2)

**Authors**: Xin Wang, Heng Chang, Beini Xie, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstract**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and its diverse downstream real-world applications. Despite the huge success in learning graph representations, current GNN models have demonstrated their vulnerability to potentially existent adversarial examples on graph-structured data. Existing approaches are either limited to structure attacks or restricted to local information, urging for the design of a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" attack challenge, we present a novel and general framework to generate adversarial examples via manipulating graph structure and node features. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.

摘要: 图神经网络(GNN)在图分类及其各种下游实际应用中取得了巨大的成功。尽管在学习图表示方面取得了巨大的成功，但当前的GNN模型已经证明了它们对图结构数据上潜在存在的对抗性示例的脆弱性。现有的方法要么局限于结构攻击，要么局限于局部信息，迫切需要设计一个更通用的图分类攻击框架，由于利用全局图级信息生成局部节点级对抗性实例的复杂性，该框架面临着巨大的挑战。为了应对这种“从全局到局部”的攻击挑战，我们提出了一个新颖而通用的框架，通过操纵图结构和节点特征来生成敌意示例。具体地说，我们利用图类激活映射及其变体来产生与图分类任务相对应的节点级重要性。然后通过算法的启发式设计，在节点级和子图级重要性的帮助下，在不可察觉的扰动预算下执行特征攻击和结构攻击。在六个真实世界基准上对四个最先进的图分类模型进行了攻击实验，验证了该框架的灵活性和有效性。



## **26. E-DPNCT: An Enhanced Attack Resilient Differential Privacy Model For Smart Grids Using Split Noise Cancellation**

E-DPNCT：一种基于分裂噪声抵消的智能电网抗攻击增强差分隐私模型 cs.CR

13 pages, 8 figues, 1 tables

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2110.11091v4) [paper-pdf](http://arxiv.org/pdf/2110.11091v4)

**Authors**: Khadija Hafeez, Donna OShea, Thomas Newe, Mubashir Husain Rehmani

**Abstract**: High frequency reporting of energy consumption data in smart grids can be used to infer sensitive information regarding the consumer's life style and poses serious security and privacy threats. Differential privacy (DP) based privacy models for smart grids ensure privacy when analysing energy consumption data for billing and load monitoring. However, DP models for smart grids are vulnerable to collusion attack where an adversary colludes with malicious smart meters and un-trusted aggregator in order to get private information from other smart meters. We first show the vulnerability of DP based privacy model for smart grids against collusion attacks to establish the need of a collusion resistant model privacy model. Then, we propose an Enhanced Differential Private Noise Cancellation Model for Load Monitoring and Billing for Smart Meters (E-DPNCT) which not only provides resistance against collusion attacks but also protects the privacy of the smart grid data while providing accurate billing and load monitoring. We use differential privacy with a split noise cancellation protocol with multiple master smart meters (MSMs) to achieve colluison resistance. We did extensive comparison of our E-DPNCT model with state of the art attack resistant privacy preserving models such as EPIC for collusion attack. We simulate our E-DPNCT model with real time data which shows significant improvement in privacy attack scenarios. Further, we analyze the impact of selecting different sensitivity parameters for calibrating DP noise over the privacy of customer electricity profile and accuracy of electricity data aggregation such as load monitoring and billing.

摘要: 智能电网中能源消耗数据的高频报告可用于推断有关消费者生活方式的敏感信息，并构成严重的安全和隐私威胁。基于差分隐私(DP)的智能电网隐私模型可在分析用于计费和负荷监控的能耗数据时确保隐私。然而，智能电网的DP模型容易受到合谋攻击，即对手与恶意智能电表和不可信的聚合器串通，以便从其他智能电表获取私人信息。本文首先分析了基于DP的智能电网隐私模型对合谋攻击的脆弱性，建立了一种防合谋攻击的隐私模型。在此基础上，提出了一种用于智能电表负荷监测和计费的增强型差分私有噪声抵消模型(E-DPNCT)，该模型在提供准确的计费和负荷监测的同时，不仅可以抵抗合谋攻击，而且可以保护智能电网数据的隐私。我们使用差分隐私和拆分噪声消除协议与多个主智能电表(MSM)来实现抗合谋。我们对我们的E-DPNCT模型和最新的抗攻击隐私保护模型进行了广泛的比较，例如EPIC用于合谋攻击。我们用实时数据模拟了我们的E-DPNCT模型，结果表明该模型在隐私攻击场景中有明显的改善。在此基础上，分析了不同灵敏度参数的选取对用户用电信息的隐私性和负荷监测、计费等用电数据汇总的准确性的影响。



## **27. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

24 pages, 17 figures, 1 table

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2307.02881v3) [paper-pdf](http://arxiv.org/pdf/2307.02881v3)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Yiwei Fu, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.

摘要: 本文首先描述了用于估计图像的概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域--并不是每一种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，尽管图像可能位于这样的低维流形上，但流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。在追求这一目标的过程中，我们考虑了人工智能和计算机视觉领域中流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下特性：1)样本生成：应该能够根据建模的密度函数从该分布中进行样本；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应该能够计算该样本的概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们证明了这种概率描述可以用来构建对抗攻击的防御。除了用密度来描述流形之外，我们还考虑了如何使用语义解释来描述流形上的点。为此，我们考虑了一种新的语言框架，它利用变分编码器来产生驻留在给定流形上的点的无纠缠表示。然后，流形上的点之间的轨迹可以通过不断演变的语义描述来描述。



## **28. Mayhem: Targeted Corruption of Register and Stack Variables**

破坏：寄存器和堆栈变量的定向损坏 cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02545v1) [paper-pdf](http://arxiv.org/pdf/2309.02545v1)

**Authors**: Andrew J. Adiletta, M. Caner Tol, Yarkın Doröz, Berk Sunar

**Abstract**: In the past decade, many vulnerabilities were discovered in microarchitectures which yielded attack vectors and motivated the study of countermeasures. Further, architectural and physical imperfections in DRAMs led to the discovery of Rowhammer attacks which give an adversary power to introduce bit flips in a victim's memory space. Numerous studies analyzed Rowhammer and proposed techniques to prevent it altogether or to mitigate its effects.   In this work, we push the boundary and show how Rowhammer can be further exploited to inject faults into stack variables and even register values in a victim's process. We achieve this by targeting the register value that is stored in the process's stack, which subsequently is flushed out into the memory, where it becomes vulnerable to Rowhammer. When the faulty value is restored into the register, it will end up used in subsequent iterations. The register value can be stored in the stack via latent function calls in the source or by actively triggering signal handlers. We demonstrate the power of the findings by applying the techniques to bypass SUDO and SSH authentication. We further outline how MySQL and other cryptographic libraries can be targeted with the new attack vector. There are a number of challenges this work overcomes with extensive experimentation before coming together to yield an end-to-end attack on an OpenSSL digital signature: achieving co-location with stack and register variables, with synchronization provided via a blocking window. We show that stack and registers are no longer safe from the Rowhammer attack.

摘要: 在过去的十年中，微体系结构中发现了许多漏洞，这些漏洞产生了攻击载体，并推动了对抗措施的研究。此外，DRAM的结构和物理缺陷导致了Rowhammer攻击的发现，这种攻击使对手有能力在受害者的记忆空间中引入比特翻转。许多研究分析了Rowhammer，并提出了完全预防或减轻其影响的技术。在这项工作中，我们突破了界限，并展示了如何进一步利用Rowhammer向受害者进程中的堆栈变量甚至寄存器值注入错误。我们通过锁定存储在进程堆栈中的寄存器值来实现这一点，该寄存器值随后被刷新到内存中，在内存中变得容易受到Rowhammer的攻击。当故障值恢复到寄存器中时，它将在后续迭代中使用。寄存器值可以通过源代码中的潜在函数调用或通过主动触发信号处理程序存储在堆栈中。我们通过应用绕过SUDO和SSH身份验证的技术来演示这些发现的威力。我们进一步概述了如何利用新的攻击载体将MySQL和其他加密库作为目标。这项工作通过广泛的实验克服了许多挑战，然后结合在一起对OpenSSL数字签名进行端到端攻击：实现堆栈和寄存器变量的协同定位，并通过阻塞窗口提供同步。我们表明堆栈和寄存器不再安全，不再受到Rowhammer攻击。



## **29. Adaptive Adversarial Training Does Not Increase Recourse Costs**

适应性对抗性训练不会增加追索权成本 cs.LG

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02528v1) [paper-pdf](http://arxiv.org/pdf/2309.02528v1)

**Authors**: Ian Hardy, Jayanth Yetukuri, Yang Liu

**Abstract**: Recent work has connected adversarial attack methods and algorithmic recourse methods: both seek minimal changes to an input instance which alter a model's classification decision. It has been shown that traditional adversarial training, which seeks to minimize a classifier's susceptibility to malicious perturbations, increases the cost of generated recourse; with larger adversarial training radii correlating with higher recourse costs. From the perspective of algorithmic recourse, however, the appropriate adversarial training radius has always been unknown. Another recent line of work has motivated adversarial training with adaptive training radii to address the issue of instance-wise variable adversarial vulnerability, showing success in domains with unknown attack radii. This work studies the effects of adaptive adversarial training on algorithmic recourse costs. We establish that the improvements in model robustness induced by adaptive adversarial training show little effect on algorithmic recourse costs, providing a potential avenue for affordable robustness in domains where recoursability is critical.

摘要: 最近的工作将对抗性攻击方法和算法求助方法联系在一起：两者都寻求对输入实例进行最小程度的更改，从而改变模型的分类决策。已有研究表明，传统的对抗性训练试图将分类器对恶意干扰的敏感性降至最低，从而增加了生成资源的成本；较大的对抗性训练半径与较高的资源成本相关。然而，从算法资源的角度来看，合适的对抗性训练半径一直是未知的。最近的另一项工作是用自适应训练半径激励对手训练，以解决实例可变对手脆弱性的问题，在攻击半径未知的领域显示出成功。本文研究了适应性对抗性训练对算法资源成本的影响。我们发现，自适应对抗性训练对模型稳健性的改善对算法资源成本几乎没有影响，这为可资源性至关重要的领域提供了一种负担得起的稳健性的潜在途径。



## **30. Black-Box Attacks against Signed Graph Analysis via Balance Poisoning**

利用平衡毒化对符号图分析的黑盒攻击 cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02396v1) [paper-pdf](http://arxiv.org/pdf/2309.02396v1)

**Authors**: Jialong Zhou, Yuni Lai, Jian Ren, Kai Zhou

**Abstract**: Signed graphs are well-suited for modeling social networks as they capture both positive and negative relationships. Signed graph neural networks (SGNNs) are commonly employed to predict link signs (i.e., positive and negative) in such graphs due to their ability to handle the unique structure of signed graphs. However, real-world signed graphs are vulnerable to malicious attacks by manipulating edge relationships, and existing adversarial graph attack methods do not consider the specific structure of signed graphs. SGNNs often incorporate balance theory to effectively model the positive and negative links. Surprisingly, we find that the balance theory that they rely on can ironically be exploited as a black-box attack. In this paper, we propose a novel black-box attack called balance-attack that aims to decrease the balance degree of the signed graphs. We present an efficient heuristic algorithm to solve this NP-hard optimization problem. We conduct extensive experiments on five popular SGNN models and four real-world datasets to demonstrate the effectiveness and wide applicability of our proposed attack method. By addressing these challenges, our research contributes to a better understanding of the limitations and resilience of robust models when facing attacks on SGNNs. This work contributes to enhancing the security and reliability of signed graph analysis in social network modeling. Our PyTorch implementation of the attack is publicly available on GitHub: https://github.com/JialongZhou666/Balance-Attack.git.

摘要: 带符号的图非常适合为社交网络建模，因为它们捕捉到了积极和消极的关系。带符号图神经网络(SGNN)由于能够处理带符号图的独特结构，通常被用来预测这类图中的链接符号(即正和负)。然而，现实世界中的签名图通过操纵边关系容易受到恶意攻击，现有的对抗性图攻击方法没有考虑签名图的具体结构。SGNN经常结合平衡理论来有效地模拟积极和消极的联系。令人惊讶的是，我们发现他们所依赖的平衡理论可以讽刺地被用作黑箱攻击。在本文中，我们提出了一种新的黑盒攻击，称为平衡攻击，旨在降低签名图的平衡度。我们给出了一个有效的启发式算法来解决这个NP-Hard优化问题。我们在五个流行的SGNN模型和四个真实世界的数据集上进行了大量的实验，以验证我们所提出的攻击方法的有效性和广泛的适用性。通过解决这些挑战，我们的研究有助于更好地理解稳健模型在面对SGNN攻击时的局限性和弹性。这项工作有助于提高社会网络建模中签名图分析的安全性和可靠性。我们对攻击的PyTorch实现在GitHub上公开可用：https://github.com/JialongZhou666/Balance-Attack.git.



## **31. Adaversarial Issue of Machine Learning Approaches Applied in Smart Grid: A Survey**

机器学习方法在智能电网中的应用研究综述 cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2308.15736v2) [paper-pdf](http://arxiv.org/pdf/2308.15736v2)

**Authors**: Zhenyong Zhang, Mengxiang Liu

**Abstract**: The machine learning (ML) sees an increasing prevalence of being used in the internet-of-things enabled smart grid. However, the trustworthiness of ML is a severe issue that must be addressed to accommodate the trend of ML-based smart grid applications (MLsgAPPs). The adversarial distortion injected into the power signal will greatly affect the system's normal control and operation. Therefore, it is imperative to conduct vulnerability assessment for MLsgAPPs applied in the context of safety-critical power systems. In this paper, we provide a comprehensive review of the recent progress in designing attack and defense methods for MLsgAPPs. Unlike the traditional survey about ML security, this is the first review work about the security of MLsgAPPs that focuses on the characteristics of power systems. The survey is organized from the aspects of adversarial assumptions, targeted applications, evaluation metrics, defending approaches, physics-related constraints, and applied datasets. We also highlight future directions on this topic to encourage more researchers to conduct further research on adversarial attacks and defending approaches for MLsgAPPs.

摘要: 机器学习(ML)在物联网智能电网中的应用越来越普遍。然而，ML的可信性是一个必须解决的严重问题，以适应基于ML的智能电网应用(MLsgAPP)的趋势。注入到电源信号中的对抗性失真将极大地影响系统的正常控制和运行。因此，对应用于安全关键电力系统背景下的MLsgAPP进行脆弱性评估势在必行。在本文中，我们提供了一个全面的进展，设计攻击和防御方法的MLsgAPP。与传统的ML安全研究不同，本文首次针对电力系统的特点对MLsgAPP的安全问题进行了综述。调查从对抗性假设、目标应用、评估指标、防御方法、与物理相关的约束和应用数据集等方面进行组织。我们还指出了这一主题的未来方向，以鼓励更多的研究人员对MLsgAPP的对抗性攻击和防御方法进行进一步的研究。



## **32. When Measures are Unreliable: Imperceptible Adversarial Perturbations toward Top-$k$ Multi-Label Learning**

当测量不可靠时：对Top-$k$多标签学习的潜移默化的对抗性扰动 cs.CV

22 pages, 7 figures, accepted by ACM MM 2023

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.00007v2) [paper-pdf](http://arxiv.org/pdf/2309.00007v2)

**Authors**: Yuchen Sun, Qianqian Xu, Zitai Wang, Qingming Huang

**Abstract**: With the great success of deep neural networks, adversarial learning has received widespread attention in various studies, ranging from multi-class learning to multi-label learning. However, existing adversarial attacks toward multi-label learning only pursue the traditional visual imperceptibility but ignore the new perceptible problem coming from measures such as Precision@$k$ and mAP@$k$. Specifically, when a well-trained multi-label classifier performs far below the expectation on some samples, the victim can easily realize that this performance degeneration stems from attack, rather than the model itself. Therefore, an ideal multi-labeling adversarial attack should manage to not only deceive visual perception but also evade monitoring of measures. To this end, this paper first proposes the concept of measure imperceptibility. Then, a novel loss function is devised to generate such adversarial perturbations that could achieve both visual and measure imperceptibility. Furthermore, an efficient algorithm, which enjoys a convex objective, is established to optimize this objective. Finally, extensive experiments on large-scale benchmark datasets, such as PASCAL VOC 2012, MS COCO, and NUS WIDE, demonstrate the superiority of our proposed method in attacking the top-$k$ multi-label systems.

摘要: 随着深度神经网络的巨大成功，对抗性学习在从多类学习到多标签学习的各种研究中受到了广泛的关注。然而，现有的针对多标签学习的对抗性攻击只追求传统的视觉不可感知性，而忽略了精度@$k$和MAP@$k$等度量带来的新的可感知问题。具体地说，当训练有素的多标签分类器在某些样本上的性能远远低于预期时，受害者可以很容易地意识到这种性能退化源于攻击，而不是模型本身。因此，一个理想的多标签对抗性攻击不仅应该能够欺骗视觉感知，而且应该能够逃避措施的监控。为此，本文首先提出了度量不可感知性的概念。然后，设计了一种新的损失函数来产生这样的对抗性扰动，可以同时实现视觉和测量不可感知性。此外，还建立了一个具有凸性目标的有效算法来优化该目标。最后，在大规模基准数据集(如Pascal VOC 2012、MS Coco和NUS Wide)上的大量实验证明了该方法在攻击top-$k$多标签系统方面的优越性。



## **33. The Adversarial Implications of Variable-Time Inference**

可变时间推理的对抗性含义 cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02159v1) [paper-pdf](http://arxiv.org/pdf/2309.02159v1)

**Authors**: Dudi Biton, Aditi Misra, Efrat Levy, Jaidip Kotak, Ron Bitton, Roei Schuster, Nicolas Papernot, Yuval Elovici, Ben Nassi

**Abstract**: Machine learning (ML) models are known to be vulnerable to a number of attacks that target the integrity of their predictions or the privacy of their training data. To carry out these attacks, a black-box adversary must typically possess the ability to query the model and observe its outputs (e.g., labels). In this work, we demonstrate, for the first time, the ability to enhance such decision-based attacks. To accomplish this, we present an approach that exploits a novel side channel in which the adversary simply measures the execution time of the algorithm used to post-process the predictions of the ML model under attack. The leakage of inference-state elements into algorithmic timing side channels has never been studied before, and we have found that it can contain rich information that facilitates superior timing attacks that significantly outperform attacks based solely on label outputs. In a case study, we investigate leakage from the non-maximum suppression (NMS) algorithm, which plays a crucial role in the operation of object detectors. In our examination of the timing side-channel vulnerabilities associated with this algorithm, we identified the potential to enhance decision-based attacks. We demonstrate attacks against the YOLOv3 detector, leveraging the timing leakage to successfully evade object detection using adversarial examples, and perform dataset inference. Our experiments show that our adversarial examples exhibit superior perturbation quality compared to a decision-based attack. In addition, we present a new threat model in which dataset inference based solely on timing leakage is performed. To address the timing leakage vulnerability inherent in the NMS algorithm, we explore the potential and limitations of implementing constant-time inference passes as a mitigation strategy.

摘要: 众所周知，机器学习(ML)模型容易受到一些攻击，这些攻击的目标是它们预测的完整性或它们的训练数据的隐私。要执行这些攻击，黑盒对手通常必须具备查询模型并观察其输出(例如，标签)的能力。在这项工作中，我们首次展示了增强这种基于决策的攻击的能力。为了实现这一点，我们提出了一种利用一种新的侧通道的方法，其中攻击者只需测量算法的执行时间，该算法用于在攻击下对ML模型的预测进行后处理。推理状态元素泄漏到算法定时侧通道之前从未被研究过，我们发现它可以包含丰富的信息，这些信息有助于优越的定时攻击，这些攻击的性能显著优于仅基于标签输出的攻击。在一个案例研究中，我们调查了非最大抑制(NMS)算法的泄漏，该算法在目标探测器的操作中起着至关重要的作用。在我们对与该算法相关的计时侧通道漏洞的检查中，我们确定了增强基于决策的攻击的可能性。我们演示了对YOLOv3检测器的攻击，利用定时泄漏成功地逃避了目标检测，并执行了数据集推理。我们的实验表明，与基于决策的攻击相比，我们的对抗性例子显示出更好的扰动质量。此外，我们还提出了一种新的威胁模型，在该模型中，仅基于定时泄漏来执行数据集推理。为了解决NMS算法中固有的定时泄漏漏洞，我们探索了实现常量时间推理传递作为一种缓解策略的潜力和局限性。



## **34. Robust Recommender System: A Survey and Future Directions**

健壮推荐系统的研究现状与发展方向 cs.IR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2309.02057v1) [paper-pdf](http://arxiv.org/pdf/2309.02057v1)

**Authors**: Kaike Zhang, Qi Cao, Fei Sun, Yunfan Wu, Shuchang Tao, Huawei Shen, Xueqi Cheng

**Abstract**: With the rapid growth of information, recommender systems have become integral for providing personalized suggestions and overcoming information overload. However, their practical deployment often encounters "dirty" data, where noise or malicious information can lead to abnormal recommendations. Research on improving recommender systems' robustness against such dirty data has thus gained significant attention. This survey provides a comprehensive review of recent work on recommender systems' robustness. We first present a taxonomy to organize current techniques for withstanding malicious attacks and natural noise. We then explore state-of-the-art methods in each category, including fraudster detection, adversarial training, certifiable robust training against malicious attacks, and regularization, purification, self-supervised learning against natural noise. Additionally, we summarize evaluation metrics and common datasets used to assess robustness. We discuss robustness across varying recommendation scenarios and its interplay with other properties like accuracy, interpretability, privacy, and fairness. Finally, we delve into open issues and future research directions in this emerging field. Our goal is to equip readers with a holistic understanding of robust recommender systems and spotlight pathways for future research and development.

摘要: 随着信息的快速增长，推荐系统已经成为提供个性化推荐和克服信息过载的不可或缺的系统。然而，他们在实际部署中经常会遇到“脏”数据，其中的噪音或恶意信息可能会导致异常推荐。因此，提高推荐系统对这些脏数据的稳健性的研究受到了极大的关注。这项调查全面回顾了最近关于推荐系统健壮性的研究工作。我们首先提出了一种分类来组织当前抵御恶意攻击和自然噪声的技术。然后，我们在每个类别中探索最先进的方法，包括欺诈者检测、对抗性训练、针对恶意攻击的可认证的健壮训练，以及针对自然噪声的正则化、净化和自我监督学习。此外，我们还总结了用于评估稳健性的评估指标和常见数据集。我们讨论了在不同的推荐场景中的健壮性，以及它与其他属性的相互作用，如准确性、可解释性、隐私和公平性。最后，我们对这一新兴领域的开放问题和未来研究方向进行了深入的探讨。我们的目标是让读者对强大的推荐系统有一个全面的了解，并为未来的研究和开发指明方向。



## **35. F3B: A Low-Overhead Blockchain Architecture with Per-Transaction Front-Running Protection**

F3B：一种具有每事务前置保护的低开销区块链架构 cs.CR

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2205.08529v3) [paper-pdf](http://arxiv.org/pdf/2205.08529v3)

**Authors**: Haoqian Zhang, Louis-Henri Merino, Ziyan Qu, Mahsa Bastankhah, Vero Estrada-Galinanes, Bryan Ford

**Abstract**: Front-running attacks, which benefit from advanced knowledge of pending transactions, have proliferated in the blockchain space since the emergence of decentralized finance. Front-running causes devastating losses to honest participants and continues to endanger the fairness of the ecosystem. We present Flash Freezing Flash Boys (F3B), a blockchain architecture that addresses front-running attacks by using threshold cryptography. In F3B, a user generates a symmetric key to encrypt their transaction, and once the underlying consensus layer has finalized the transaction, a decentralized secret-management committee reveals this key. F3B mitigates front-running attacks because, before the consensus group finalizes it, an adversary can no longer read the content of a transaction, thus preventing the adversary from benefiting from advanced knowledge of pending transactions. Unlike other mitigation systems, F3B properly ensures that all unfinalized transactions, even with significant delays, remain private by adopting per-transaction protection. Furthermore, F3B addresses front-running at the execution layer; thus, our solution is agnostic to the underlying consensus algorithm and compatible with existing smart contracts. We evaluated F3B on Ethereum with a modified execution layer and found only a negligible (0.026%) increase in transaction latency, specifically due to running threshold decryption with a 128-member secret-management committee after a transaction is finalized; this indicates that F3B is both practical and low-cost.

摘要: 自去中心化金融出现以来，受益于待完成交易的先进知识的前沿攻击在区块链领域激增。领跑给诚实的参与者造成了毁灭性的损失，并继续危及生态系统的公平性。我们提出了Flash冻结Flash Boys(F3B)，这是一种区块链架构，通过使用门限密码来应对前沿攻击。在F3B中，用户生成对称密钥来加密他们的交易，一旦底层共识层完成交易，分散的秘密管理委员会就会公布该密钥。F3B减轻了前置攻击，因为在共识小组最终确定之前，对手不再能够读取交易的内容，从而阻止对手受益于有关未决交易的高级知识。与其他缓解系统不同，F3B通过采用每笔交易保护来适当地确保所有未完成的交易，即使有显著的延迟，也是保密的。此外，F3B解决了执行层的先行问题；因此，我们的解决方案与底层共识算法无关，并与现有的智能合约兼容。我们在具有修改的执行层的Etherum上评估了F3B，发现交易延迟仅略有增加(0.026%)，特别是由于在交易完成后与一个由128名成员组成的秘密管理委员会运行阈值解密；这表明F3B既实用又低成本。



## **36. Boosting the Adversarial Transferability of Surrogate Models with Dark Knowledge**

提高具有暗知识的代理模型的对抗性转移 cs.LG

Accepted at 2023 International Conference on Tools with Artificial  Intelligence (ICTAI)

**SubmitDate**: 2023-09-05    [abs](http://arxiv.org/abs/2206.08316v2) [paper-pdf](http://arxiv.org/pdf/2206.08316v2)

**Authors**: Dingcheng Yang, Zihao Xiao, Wenjian Yu

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial examples. And, the adversarial examples have transferability, which means that an adversarial example for a DNN model can fool another model with a non-trivial probability. This gave birth to the transfer-based attack where the adversarial examples generated by a surrogate model are used to conduct black-box attacks. There are some work on generating the adversarial examples from a given surrogate model with better transferability. However, training a special surrogate model to generate adversarial examples with better transferability is relatively under-explored. This paper proposes a method for training a surrogate model with dark knowledge to boost the transferability of the adversarial examples generated by the surrogate model. This trained surrogate model is named dark surrogate model (DSM). The proposed method for training a DSM consists of two key components: a teacher model extracting dark knowledge, and the mixing augmentation skill enhancing dark knowledge of training data. We conducted extensive experiments to show that the proposed method can substantially improve the adversarial transferability of surrogate models across different architectures of surrogate models and optimizers for generating adversarial examples, and it can be applied to other scenarios of transfer-based attack that contain dark knowledge, like face verification. Our code is publicly available at \url{https://github.com/ydc123/Dark_Surrogate_Model}.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。而且，对抗性例子具有可转移性，这意味着一个DNN模型的对抗性例子可以以非平凡的概率愚弄另一个模型。这就产生了基于转移的攻击，其中通过代理模型生成的对抗性例子被用来进行黑盒攻击。已有一些工作是从给定的代理模型中生成具有较好可转移性的对抗性实例。然而，训练一种特殊的代理模型来生成具有更好可转移性的对抗性实例的研究相对较少。本文提出了一种利用暗知识训练代理模型的方法，以提高代理模型生成的对抗性实例的可转移性。这种经过训练的代理模型被称为暗代理模型(DSM)。提出的训练DSM的方法由两个关键部分组成：提取暗知识的教师模型和增强训练数据的暗知识的混合增强技术。通过大量的实验表明，该方法可以有效地提高代理模型在不同结构的代理模型和优化器之间的对抗性可转移性，并且可以应用到其他基于转移的攻击场景中，比如人脸验证。我们的代码在\url{https://github.com/ydc123/Dark_Surrogate_Model}.上公开提供



## **37. Efficient Defense Against Model Stealing Attacks on Convolutional Neural Networks**

卷积神经网络模型窃取攻击的有效防御 cs.LG

Accepted for publication at 2023 International Conference on Machine  Learning and Applications (ICMLA)

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01838v1) [paper-pdf](http://arxiv.org/pdf/2309.01838v1)

**Authors**: Kacem Khaled, Mouna Dhaouadi, Felipe Gohring de Magalhães, Gabriela Nicolescu

**Abstract**: Model stealing attacks have become a serious concern for deep learning models, where an attacker can steal a trained model by querying its black-box API. This can lead to intellectual property theft and other security and privacy risks. The current state-of-the-art defenses against model stealing attacks suggest adding perturbations to the prediction probabilities. However, they suffer from heavy computations and make impracticable assumptions about the adversary. They often require the training of auxiliary models. This can be time-consuming and resource-intensive which hinders the deployment of these defenses in real-world applications. In this paper, we propose a simple yet effective and efficient defense alternative. We introduce a heuristic approach to perturb the output probabilities. The proposed defense can be easily integrated into models without additional training. We show that our defense is effective in defending against three state-of-the-art stealing attacks. We evaluate our approach on large and quantized (i.e., compressed) Convolutional Neural Networks (CNNs) trained on several vision datasets. Our technique outperforms the state-of-the-art defenses with a $\times37$ faster inference latency without requiring any additional model and with a low impact on the model's performance. We validate that our defense is also effective for quantized CNNs targeting edge devices.

摘要: 模型窃取攻击已经成为深度学习模型的一个严重问题，在深度学习模型中，攻击者可以通过查询黑盒API来窃取训练的模型。这可能会导致知识产权被盗以及其他安全和隐私风险。目前针对模型窃取攻击的最先进防御措施建议增加预测概率的扰动。然而，他们遭受着繁重的计算，并对对手做出不切实际的假设。它们往往需要辅助模型的培训。这可能会耗费时间和资源，从而阻碍在实际应用程序中部署这些防御措施。在本文中，我们提出了一种简单而有效的防御方案。我们引入了一种启发式方法来扰动输出概率。建议的防御可以很容易地集成到模型中，而不需要额外的培训。我们表明，我们的防御在防御三种最先进的窃取攻击方面是有效的。我们在几个视觉数据集上训练的大型量化(即压缩)卷积神经网络(CNN)上对我们的方法进行了评估。我们的技术优于最先进的防御技术，在不需要任何额外模型的情况下，推理延迟快37倍，并且对模型性能的影响很小。我们验证了我们的防御对于针对边缘设备的量化CNN也是有效的。



## **38. Baseline Defenses for Adversarial Attacks Against Aligned Language Models**

针对对齐语言模型的对抗性攻击的基线防御 cs.LG

12 pages

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.00614v2) [paper-pdf](http://arxiv.org/pdf/2309.00614v2)

**Authors**: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein

**Abstract**: As Large Language Models quickly become ubiquitous, it becomes critical to understand their security vulnerabilities. Recent work shows that text optimizers can produce jailbreaking prompts that bypass moderation and alignment. Drawing from the rich body of work on adversarial machine learning, we approach these attacks with three questions: What threat models are practically useful in this domain? How do baseline defense techniques perform in this new domain? How does LLM security differ from computer vision?   We evaluate several baseline defense strategies against leading adversarial attacks on LLMs, discussing the various settings in which each is feasible and effective. Particularly, we look at three types of defenses: detection (perplexity based), input preprocessing (paraphrase and retokenization), and adversarial training. We discuss white-box and gray-box settings and discuss the robustness-performance trade-off for each of the defenses considered. We find that the weakness of existing discrete optimizers for text, combined with the relatively high costs of optimization, makes standard adaptive attacks more challenging for LLMs. Future research will be needed to uncover whether more powerful optimizers can be developed, or whether the strength of filtering and preprocessing defenses is greater in the LLMs domain than it has been in computer vision.

摘要: 随着大型语言模型迅速变得无处不在，了解它们的安全漏洞变得至关重要。最近的研究表明，文本优化器可以生成绕过审核和对齐的越狱提示。从对抗性机器学习的丰富工作中，我们用三个问题来处理这些攻击：什么威胁模型在这个领域实际上是有用的？基线防御技术在这个新领域的表现如何？LLM安全与计算机视觉有何不同？我们评估了几种针对LLMS的主要对手攻击的基线防御策略，讨论了每种策略可行和有效的各种设置。特别是，我们研究了三种类型的防御：检测(基于困惑)、输入预处理(释义和重新标记化)和对抗性训练。我们讨论了白盒和灰盒设置，并讨论了所考虑的每种防御的稳健性和性能之间的权衡。我们发现，现有的文本离散优化器的弱点，再加上相对较高的优化成本，使得标准的自适应攻击对LLMS来说更具挑战性。未来的研究将需要揭示是否可以开发出更强大的优化器，或者在LLMS领域中过滤和预处理防御的强度是否比在计算机视觉领域更强。



## **39. Machine Learning (In) Security: A Stream of Problems**

机器学习(In)安全：一系列问题 cs.CR

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2010.16045v2) [paper-pdf](http://arxiv.org/pdf/2010.16045v2)

**Authors**: Fabrício Ceschin, Marcus Botacin, Albert Bifet, Bernhard Pfahringer, Luiz S. Oliveira, Heitor Murilo Gomes, André Grégio

**Abstract**: Machine Learning (ML) has been widely applied to cybersecurity and is considered state-of-the-art for solving many of the open issues in that field. However, it is very difficult to evaluate how good the produced solutions are, since the challenges faced in security may not appear in other areas. One of these challenges is the concept drift, which increases the existing arms race between attackers and defenders: malicious actors can always create novel threats to overcome the defense solutions, which may not consider them in some approaches. Due to this, it is essential to know how to properly build and evaluate an ML-based security solution. In this paper, we identify, detail, and discuss the main challenges in the correct application of ML techniques to cybersecurity data. We evaluate how concept drift, evolution, delayed labels, and adversarial ML impact the existing solutions. Moreover, we address how issues related to data collection affect the quality of the results presented in the security literature, showing that new strategies are needed to improve current solutions. Finally, we present how existing solutions may fail under certain circumstances, and propose mitigations to them, presenting a novel checklist to help the development of future ML solutions for cybersecurity.

摘要: 机器学习(ML)已被广泛应用于网络安全领域，被认为是解决该领域许多开放问题的最新技术。然而，很难评估产生的解决方案有多好，因为在安全方面面临的挑战可能不会出现在其他领域。其中一个挑战是概念漂移，这加剧了攻击者和防御者之间现有的军备竞赛：恶意行为者总是可以制造新的威胁来克服防御解决方案，而防御解决方案可能在某些方法中不考虑它们。因此，了解如何正确构建和评估基于ML的安全解决方案是至关重要的。在这篇文章中，我们确定、详细和讨论了在将ML技术正确应用于网络安全数据方面的主要挑战。我们评估了概念漂移、演化、延迟标签和对抗性ML对现有解决方案的影响。此外，我们讨论了与数据收集相关的问题如何影响安全文献中提出的结果的质量，表明需要新的战略来改进现有的解决方案。最后，我们介绍了现有解决方案在某些情况下可能失败的原因，并提出了缓解措施，提出了一种新颖的检查表，以帮助开发未来的网络安全ML解决方案。



## **40. MathAttack: Attacking Large Language Models Towards Math Solving Ability**

MathAttack：攻击大型语言模型的数学解题能力 cs.CL

11 pages, 6 figures

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01686v1) [paper-pdf](http://arxiv.org/pdf/2309.01686v1)

**Authors**: Zihao Zhou, Qiufeng Wang, Mingyu Jin, Jie Yao, Jianan Ye, Wei Liu, Wei Wang, Xiaowei Huang, Kaizhu Huang

**Abstract**: With the boom of Large Language Models (LLMs), the research of solving Math Word Problem (MWP) has recently made great progress. However, there are few studies to examine the security of LLMs in math solving ability. Instead of attacking prompts in the use of LLMs, we propose a MathAttack model to attack MWP samples which are closer to the essence of security in solving math problems. Compared to traditional text adversarial attack, it is essential to preserve the mathematical logic of original MWPs during the attacking. To this end, we propose logical entity recognition to identify logical entries which are then frozen. Subsequently, the remaining text are attacked by adopting a word-level attacker. Furthermore, we propose a new dataset RobustMath to evaluate the robustness of LLMs in math solving ability. Extensive experiments on our RobustMath and two another math benchmark datasets GSM8K and MultiAirth show that MathAttack could effectively attack the math solving ability of LLMs. In the experiments, we observe that (1) Our adversarial samples from higher-accuracy LLMs are also effective for attacking LLMs with lower accuracy (e.g., transfer from larger to smaller-size LLMs, or from few-shot to zero-shot prompts); (2) Complex MWPs (such as more solving steps, longer text, more numbers) are more vulnerable to attack; (3) We can improve the robustness of LLMs by using our adversarial samples in few-shot prompts. Finally, we hope our practice and observation can serve as an important attempt towards enhancing the robustness of LLMs in math solving ability. We will release our code and dataset.

摘要: 近年来，随着大型语言模型的兴起，数学应用题的研究取得了长足的进步。然而，很少有研究考察LLMS在数学解题能力上的安全性。在解决数学问题时，我们提出了一种MathAttack模型来攻击更接近安全本质的MWP样本，而不是使用LLMS来攻击提示。与传统的文本对抗性攻击相比，在攻击过程中必须保留原始MWP的数学逻辑。为此，我们提出了逻辑实体识别来识别然后被冻结的逻辑条目。随后，采用词级攻击者对剩余文本进行攻击。此外，我们还提出了一种新的数据集RobustMath来评估LLMS在数学求解能力方面的稳健性。在我们的RobustMath和另外两个数学基准数据集GSM8K和MultiAirth上的大量实验表明，MathAttack可以有效地攻击LLMS的数学求解能力。在实验中，我们观察到：(1)我们从高准确度的LLMS中得到的敌意样本对于攻击准确率较低的LLMS也是有效的(例如，从较大的LLMS转移到较小的LLMS，或者从少枪到零枪的提示)；(2)复杂的MWP(如更多的求解步骤、更长的文本、更多的数字)更容易受到攻击；(3)通过在少枪提示中使用我们的对手样本可以提高LLMS的健壮性。最后，我们希望我们的实践和观察能够为增强LLMS在数学求解能力方面的稳健性提供重要的尝试。我们将发布我们的代码和数据集。



## **41. Hindering Adversarial Attacks with Multiple Encrypted Patch Embeddings**

利用多个加密补丁嵌入阻止敌意攻击 cs.CV

To appear in APSIPA ASC 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01620v1) [paper-pdf](http://arxiv.org/pdf/2309.01620v1)

**Authors**: AprilPyone MaungMaung, Isao Echizen, Hitoshi Kiya

**Abstract**: In this paper, we propose a new key-based defense focusing on both efficiency and robustness. Although the previous key-based defense seems effective in defending against adversarial examples, carefully designed adaptive attacks can bypass the previous defense, and it is difficult to train the previous defense on large datasets like ImageNet. We build upon the previous defense with two major improvements: (1) efficient training and (2) optional randomization. The proposed defense utilizes one or more secret patch embeddings and classifier heads with a pre-trained isotropic network. When more than one secret embeddings are used, the proposed defense enables randomization on inference. Experiments were carried out on the ImageNet dataset, and the proposed defense was evaluated against an arsenal of state-of-the-art attacks, including adaptive ones. The results show that the proposed defense achieves a high robust accuracy and a comparable clean accuracy compared to the previous key-based defense.

摘要: 在本文中，我们提出了一种兼顾效率和稳健性的基于密钥的防御方案。虽然以前的基于密钥的防御方法在对抗对手例子中似乎是有效的，但精心设计的自适应攻击可以绕过以前的防御，并且很难在ImageNet这样的大数据集上训练以前的防御。我们在先前防御的基础上进行了两个主要改进：(1)有效的训练和(2)可选的随机化。该防御方法利用了一个或多个秘密补丁嵌入和带有预训练的各向同性网络的分类器头部。当使用多个秘密嵌入时，所提出的防御实现了推理的随机化。在ImageNet数据集上进行了实验，并针对包括自适应攻击在内的各种最先进的攻击对所提出的防御进行了评估。结果表明，与以往的基于密钥的防御相比，该防御方案具有较高的稳健性和相当的清洁准确率。



## **42. Robustness of SAM: Segment Anything Under Corruptions and Beyond**

SAM的健壮性：在腐败和其他方面分割任何东西 cs.CV

The first work evaluates the robustness of SAM under various  corruptions such as style transfer, local occlusion, and adversarial patch  attack

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2306.07713v3) [paper-pdf](http://arxiv.org/pdf/2306.07713v3)

**Authors**: Yu Qiao, Chaoning Zhang, Taegoo Kang, Donghun Kim, Chenshuang Zhang, Choong Seon Hong

**Abstract**: Segment anything model (SAM), as the name suggests, is claimed to be capable of cutting out any object and demonstrates impressive zero-shot transfer performance with the guidance of prompts. However, there is currently a lack of comprehensive evaluation regarding its robustness under various corruptions. Understanding the robustness of SAM across different corruption scenarios is crucial for its real-world deployment. Prior works show that SAM is biased towards texture (style) rather than shape, motivated by which we start by investigating its robustness against style transfer, which is synthetic corruption. Following by interpreting the effects of synthetic corruption as style changes, we proceed to conduct a comprehensive evaluation for its robustness against 15 types of common corruption. These corruptions mainly fall into categories such as digital, noise, weather, and blur, and within each corruption category, we explore 5 severity levels to simulate real-world corruption scenarios. Beyond the corruptions, we further assess the robustness of SAM against local occlusion and local adversarial patch attacks. To the best of our knowledge, our work is the first of its kind to evaluate the robustness of SAM under style change, local occlusion, and local adversarial patch attacks. Given that patch attacks visible to human eyes are easily detectable, we further assess its robustness against global adversarial attacks that are imperceptible to human eyes. Overall, this work provides a comprehensive empirical study of the robustness of SAM, evaluating its performance under various corruptions and extending the assessment to critical aspects such as local occlusion, local adversarial patch attacks, and global adversarial attacks. These evaluations yield valuable insights into the practical applicability and effectiveness of SAM in addressing real-world challenges.

摘要: 片断任何模型(SAM)，顾名思义，声称能够裁剪出任何对象，并在提示的指导下展示了令人印象深刻的零射传输性能。然而，目前还缺乏对其在各种腐败情况下的稳健性的全面评估。了解SAM在不同腐败场景中的健壮性对于其实际部署至关重要。以前的工作表明，SAM偏向于纹理(风格)而不是形状，出于这个原因，我们从研究它对风格转移的稳健性开始，这是一种合成的腐败。在将合成腐败的影响解释为风格变化的基础上，我们继续对其对15种常见腐败类型的稳健性进行了综合评估。这些腐败主要分为数字、噪声、天气和模糊等类别，在每个腐败类别中，我们探索5个严重程度来模拟真实世界的腐败场景。除了腐败之外，我们还进一步评估了SAM对局部遮挡和局部恶意补丁攻击的健壮性。据我们所知，我们的工作是第一次评估SAM在风格变化、局部遮挡和局部敌意补丁攻击下的稳健性。考虑到人眼可见的补丁攻击很容易被检测到，我们进一步评估了它对人眼无法感知的全局对手攻击的稳健性。总之，这项工作对SAM的健壮性进行了全面的实证研究，评估了其在各种腐败情况下的性能，并将评估扩展到关键方面，如局部遮挡、局部对抗性补丁攻击和全局对抗性攻击。这些评估对SAM在应对现实世界挑战方面的实际适用性和有效性提供了宝贵的见解。



## **43. OUTFOX: LLM-generated Essay Detection through In-context Learning with Adversarially Generated Examples**

Outfox：基于上下文学习的LLM生成的文章检测与恶意生成的示例 cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2307.11729v2) [paper-pdf](http://arxiv.org/pdf/2307.11729v2)

**Authors**: Ryuto Koike, Masahiro Kaneko, Naoaki Okazaki

**Abstract**: Large Language Models (LLMs) have achieved human-level fluency in text generation, making it difficult to distinguish between human-written and LLM-generated texts. This poses a growing risk of misuse of LLMs and demands the development of detectors to identify LLM-generated texts. However, existing detectors lack robustness against attacks: they degrade detection accuracy by simply paraphrasing LLM-generated texts. Furthermore, a malicious user might attempt to deliberately evade the detectors based on detection results, but this has not been assumed in previous studies. In this paper, we propose OUTFOX, a framework that improves the robustness of LLM-generated-text detectors by allowing both the detector and the attacker to consider each other's output. In this framework, the attacker uses the detector's prediction labels as examples for in-context learning and adversarially generates essays that are harder to detect, while the detector uses the adversarially generated essays as examples for in-context learning to learn to detect essays from a strong attacker. Experiments in the domain of student essays show that the proposed detector improves the detection performance on the attacker-generated texts by up to +41.3 points in F1-score. Furthermore, the proposed detector shows a state-of-the-art detection performance: up to 96.9 points in F1-score, beating existing detectors on non-attacked texts. Finally, the proposed attacker drastically degrades the performance of detectors by up to -57.0 points F1-score, massively outperforming the baseline paraphrasing method for evading detection.

摘要: 大型语言模型(LLM)在文本生成方面达到了人类水平的流畅性，使得区分人类编写的文本和LLM生成的文本变得困难。这带来了滥用LLMS的越来越大的风险，并要求开发检测器来识别LLM生成的文本。然而，现有的检测器缺乏对攻击的稳健性：它们通过简单地解释LLM生成的文本来降低检测精度。此外，恶意用户可能试图根据检测结果故意躲避检测器，但在之前的研究中没有假设这一点。在本文中，我们提出了Outfox框架，它通过允许检测器和攻击者考虑彼此的输出来提高LLM生成的文本检测器的健壮性。在该框架中，攻击者使用检测器的预测标签作为上下文中学习的示例，并恶意生成更难检测的文章，而检测器使用恶意生成的文章作为上下文中学习的示例，以学习检测来自强大攻击者的文章。在学生作文领域的实验表明，该检测器在F1-Score上将攻击者生成的文本的检测性能提高了41.3分。此外，提出的检测器具有最先进的检测性能：在F1分数上高达96.9分，在非攻击文本上击败现有检测器。最后，提出的攻击者极大地降低了检测器的性能，最高可达-57.0分F1-Score，大大超过了用于逃避检测的基线改述方法。



## **44. Communication Lower Bounds for Cryptographic Broadcast Protocols**

密码广播协议的通信下限 cs.CR

A preliminary version of this work appeared in DISC 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01466v1) [paper-pdf](http://arxiv.org/pdf/2309.01466v1)

**Authors**: Erica Blum, Elette Boyle, Ran Cohen, Chen-Da Liu-Zhang

**Abstract**: Broadcast protocols enable a set of $n$ parties to agree on the input of a designated sender, even facing attacks by malicious parties. In the honest-majority setting, randomization and cryptography were harnessed to achieve low-communication broadcast with sub-quadratic total communication and balanced sub-linear cost per party. However, comparatively little is known in the dishonest-majority setting. Here, the most communication-efficient constructions are based on Dolev and Strong (SICOMP '83), and sub-quadratic broadcast has not been achieved. On the other hand, the only nontrivial $\omega(n)$ communication lower bounds are restricted to deterministic protocols, or against strong adaptive adversaries that can perform "after the fact" removal of messages.   We provide new communication lower bounds in this space, which hold against arbitrary cryptography and setup assumptions, as well as a simple protocol showing near tightness of our first bound.   1) We demonstrate a tradeoff between resiliency and communication for protocols secure against $n-o(n)$ static corruptions. For example, $\Omega(n\cdot {\sf polylog}(n))$ messages are needed when the number of honest parties is $n/{\sf polylog}(n)$; $\Omega(n\sqrt{n})$ messages are needed for $O(\sqrt{n})$ honest parties; and $\Omega(n^2)$ messages are needed for $O(1)$ honest parties.   Complementarily, we demonstrate broadcast with $O(n\cdot{\sf polylog}(n))$ total communication facing any constant fraction of static corruptions.   2) Our second bound considers $n/2 + k$ corruptions and a weakly adaptive adversary that cannot remove messages "after the fact." We show that any broadcast protocol within this setting can be attacked to force an arbitrary party to send messages to $k$ other parties. This rules out, for example, broadcast facing 51% corruptions in which all non-sender parties have sublinear communication locality.

摘要: 广播协议使一组$n$方能够就指定发送方的输入达成一致，即使面临恶意方的攻击。在诚实多数的设置下，利用随机化和密码学来实现低通信广播，具有次二次总通信量和均衡的次线性单方成本。然而，在不诚实的多数人的背景下，人们知之甚少。这里，通信效率最高的构造是基于Dolev和Strong(SICOMP‘83)，并且没有实现亚二次广播。另一方面，仅有的非平凡的$\omega(N)$通信下限仅限于确定性协议，或者是针对能够执行“事后”删除消息的强适应性对手。我们在这个空间中提供了新的通信下界，它反对任意的密码学和设置假设，以及一个简单的协议，证明了我们的第一个界几乎是紧的。1)我们证明了针对$n-o(N)$静态破坏的安全协议的弹性和通信之间的折衷。例如，当诚实方的数量为$n/{\SF PolyLog}(N)$时，需要$\Omega(n\cot{\sf PolyLog}(N))$Messages；当$O(\SQRT{n})$诚实方需要$\Omega(n\sf PolyLog)$Messages时；$O(1)$诚实方需要$\Omega(n^2)$Messages。作为补充，我们演示了具有$O(n\cdot{\sf PolyLog}(N))$总通信的广播面临任何恒定比例的静态损坏。2)我们的第二个边界考虑了$n/2+k$损坏和一个不能在事后删除消息的弱适应性对手。我们证明了此设置中的任何广播协议都可以被攻击，从而迫使任意一方向$k$其他方发送消息。例如，这排除了面临51%的损坏的广播，其中所有非发送方都具有次线性通信位置。



## **45. Toward Defensive Letter Design**

走向防御性信函设计 cs.CV

14 pages, 8 figures, accepted at ACPR 2023

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01452v1) [paper-pdf](http://arxiv.org/pdf/2309.01452v1)

**Authors**: Rentaro Kataoka, Akisato Kimura, Seiichi Uchida

**Abstract**: A major approach for defending against adversarial attacks aims at controlling only image classifiers to be more resilient, and it does not care about visual objects, such as pandas and cars, in images. This means that visual objects themselves cannot take any defensive actions, and they are still vulnerable to adversarial attacks. In contrast, letters are artificial symbols, and we can freely control their appearance unless losing their readability. In other words, we can make the letters more defensive to the attacks. This paper poses three research questions related to the adversarial vulnerability of letter images: (1) How defensive are the letters against adversarial attacks? (2) Can we estimate how defensive a given letter image is before attacks? (3) Can we control the letter images to be more defensive against adversarial attacks? For answering the first and second questions, we measure the defensibility of letters by employing Iterative Fast Gradient Sign Method (I-FGSM) and then build a deep regression model for estimating the defensibility of each letter image. We also propose a two-step method based on a generative adversarial network (GAN) for generating character images with higher defensibility, which solves the third research question.

摘要: 防御敌意攻击的一种主要方法旨在仅控制图像分类器具有更强的弹性，而不关心图像中的可视对象，如熊猫和汽车。这意味着视觉对象本身不能采取任何防御行动，它们仍然容易受到对抗性攻击。相比之下，字母是人工符号，我们可以自由控制它们的外观，而不会失去它们的可读性。换句话说，我们可以让信件对攻击更具防御性。本文提出了三个与字母图像的攻击脆弱性相关的研究问题：(1)字母对对手攻击的防御性有多强？(2)我们能估计给定的字母图像在攻击前的防御程度吗？(3)我们能控制字母图像对对手攻击具有更强的防御能力吗？为了回答第一个和第二个问题，我们使用迭代快速梯度符号方法(I-FGSM)来衡量字母的防御性，然后建立一个深度回归模型来估计每个字母图像的防御性。我们还提出了一种基于生成对抗网络(GAN)的两步生成具有更高防御性的字符图像的方法，解决了第三个研究问题。



## **46. Open Sesame! Universal Black Box Jailbreaking of Large Language Models**

芝麻开门！大型语言模型的通用黑盒越狱 cs.CL

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01446v1) [paper-pdf](http://arxiv.org/pdf/2309.01446v1)

**Authors**: Raz Lapid, Ron Langberg, Moshe Sipper

**Abstract**: Large language models (LLMs), designed to provide helpful and safe responses, often rely on alignment techniques to align with user intent and social guidelines. Unfortunately, this alignment can be exploited by malicious actors seeking to manipulate an LLM's outputs for unintended purposes. In this paper we introduce a novel approach that employs a genetic algorithm (GA) to manipulate LLMs when model architecture and parameters are inaccessible. The GA attack works by optimizing a universal adversarial prompt that -- when combined with a user's query -- disrupts the attacked model's alignment, resulting in unintended and potentially harmful outputs. Our novel approach systematically reveals a model's limitations and vulnerabilities by uncovering instances where its responses deviate from expected behavior. Through extensive experiments we demonstrate the efficacy of our technique, thus contributing to the ongoing discussion on responsible AI development by providing a diagnostic tool for evaluating and enhancing alignment of LLMs with human intent. To our knowledge this is the first automated universal black box jailbreak attack.

摘要: 大型语言模型(LLM)旨在提供有用和安全的响应，它们通常依靠对齐技术来与用户意图和社交指南保持一致。遗憾的是，恶意行为者可能会利用这种对齐，试图出于非预期目的操纵LLM的输出。在本文中，我们介绍了一种新的方法，即在模型结构和参数不可访问的情况下，使用遗传算法(GA)来操作LLM。GA攻击的工作原理是优化一个通用的对抗性提示，当与用户的查询结合在一起时，会扰乱被攻击模型的对齐，导致意外的和潜在的有害输出。我们的新方法通过揭示模型响应偏离预期行为的实例，系统地揭示了模型的局限性和漏洞。通过广泛的实验，我们展示了我们技术的有效性，从而通过提供一种诊断工具来评估和增强LLM与人类意图的一致性，从而为正在进行的关于负责任的人工智能开发的讨论做出贡献。据我们所知，这是第一次自动通用黑匣子越狱攻击。



## **47. Adv3D: Generating 3D Adversarial Examples in Driving Scenarios with NeRF**

Adv3D：使用NERF在驾驶场景中生成3D对抗性示例 cs.CV

**SubmitDate**: 2023-09-04    [abs](http://arxiv.org/abs/2309.01351v1) [paper-pdf](http://arxiv.org/pdf/2309.01351v1)

**Authors**: Leheng Li, Qing Lian, Ying-Cong Chen

**Abstract**: Deep neural networks (DNNs) have been proven extremely susceptible to adversarial examples, which raises special safety-critical concerns for DNN-based autonomous driving stacks (i.e., 3D object detection). Although there are extensive works on image-level attacks, most are restricted to 2D pixel spaces, and such attacks are not always physically realistic in our 3D world. Here we present Adv3D, the first exploration of modeling adversarial examples as Neural Radiance Fields (NeRFs). Advances in NeRF provide photorealistic appearances and 3D accurate generation, yielding a more realistic and realizable adversarial example. We train our adversarial NeRF by minimizing the surrounding objects' confidence predicted by 3D detectors on the training set. Then we evaluate Adv3D on the unseen validation set and show that it can cause a large performance reduction when rendering NeRF in any sampled pose. To generate physically realizable adversarial examples, we propose primitive-aware sampling and semantic-guided regularization that enable 3D patch attacks with camouflage adversarial texture. Experimental results demonstrate that the trained adversarial NeRF generalizes well to different poses, scenes, and 3D detectors. Finally, we provide a defense method to our attacks that involves adversarial training through data augmentation. Project page: https://len-li.github.io/adv3d-web

摘要: 深度神经网络(DNN)已被证明极易受到敌意例子的影响，这对基于DNN的自动驾驶堆栈(即3D对象检测)提出了特殊的安全关键问题。虽然有大量关于图像级攻击的工作，但大多数局限于2D像素空间，并且在我们的3D世界中，此类攻击并不总是物理上真实的。在这里，我们介绍了Adv3D，这是将对抗性例子建模为神经辐射场(NERF)的第一次探索。NERF的进步提供了照片逼真的外观和3D准确的生成，产生了一个更真实和可实现的对抗性例子。我们通过最小化3D检测器在训练集上预测的周围对象的置信度来训练我们的对手NERF。然后，我们在不可见的验证集上对Adv3D进行了评估，结果表明，在任何采样姿势下绘制NERF时，它都会导致性能大幅下降。为了生成物理上可实现的对抗性示例，我们提出了基元感知采样和语义引导正则化，使得3D补丁攻击具有伪装对抗性纹理。实验结果表明，训练好的对抗性神经网络对不同的姿态、场景和3D检测器都具有很好的泛化能力。最后，我们提供了一种通过数据增强进行对抗性训练的攻击防御方法。项目页面：https://len-li.github.io/adv3d-web



## **48. Everyone Can Attack: Repurpose Lossy Compression as a Natural Backdoor Attack**

每个人都可以攻击：将有损压缩重新用作自然的后门攻击 cs.CR

14 pages. This paper shows everyone can mount a powerful and stealthy  backdoor attack with the widely-used lossy image compression

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2308.16684v2) [paper-pdf](http://arxiv.org/pdf/2308.16684v2)

**Authors**: Sze Jue Yang, Quang Nguyen, Chee Seng Chan, Khoa D. Doan

**Abstract**: The vulnerabilities to backdoor attacks have recently threatened the trustworthiness of machine learning models in practical applications. Conventional wisdom suggests that not everyone can be an attacker since the process of designing the trigger generation algorithm often involves significant effort and extensive experimentation to ensure the attack's stealthiness and effectiveness. Alternatively, this paper shows that there exists a more severe backdoor threat: anyone can exploit an easily-accessible algorithm for silent backdoor attacks. Specifically, this attacker can employ the widely-used lossy image compression from a plethora of compression tools to effortlessly inject a trigger pattern into an image without leaving any noticeable trace; i.e., the generated triggers are natural artifacts. One does not require extensive knowledge to click on the "convert" or "save as" button while using tools for lossy image compression. Via this attack, the adversary does not need to design a trigger generator as seen in prior works and only requires poisoning the data. Empirically, the proposed attack consistently achieves 100% attack success rate in several benchmark datasets such as MNIST, CIFAR-10, GTSRB and CelebA. More significantly, the proposed attack can still achieve almost 100% attack success rate with very small (approximately 10%) poisoning rates in the clean label setting. The generated trigger of the proposed attack using one lossy compression algorithm is also transferable across other related compression algorithms, exacerbating the severity of this backdoor threat. This work takes another crucial step toward understanding the extensive risks of backdoor attacks in practice, urging practitioners to investigate similar attacks and relevant backdoor mitigation methods.

摘要: 后门攻击的漏洞最近威胁到了机器学习模型在实际应用中的可信度。传统的观点认为，并不是每个人都能成为攻击者，因为设计触发生成算法的过程通常涉及大量的工作和广泛的实验，以确保攻击的隐蔽性和有效性。或者，本文显示存在更严重的后门威胁：任何人都可以利用一种易于访问的算法进行静默后门攻击。具体地说，该攻击者可以利用大量压缩工具中广泛使用的有损图像压缩，毫不费力地向图像中注入触发图案，而不会留下任何明显的痕迹；即，生成的触发是自然伪像。在使用有损图像压缩工具时，不需要广博的知识就可以点击“转换”或“另存为”按钮。通过这种攻击，攻击者不需要像以前的工作中看到的那样设计触发发生器，只需要毒化数据即可。实验证明，该攻击在MNIST、CIFAR-10、GTSRB和CelebA等多个基准数据集上一致达到100%的攻击成功率。更重要的是，在干净的标签设置下，建议的攻击仍然可以实现几乎100%的攻击成功率，而投毒率非常低(约10%)。使用一种有损压缩算法生成的拟议攻击的触发器也可以在其他相关压缩算法之间传输，从而加剧了这种后门威胁的严重性。这项工作朝着了解实践中后门攻击的广泛风险又迈出了关键的一步，敦促实践者调查类似的攻击和相关的后门缓解方法。



## **49. A Comprehensive Study and Comparison of the Robustness of 3D Object Detectors Against Adversarial Attacks**

3D目标检测器抗敌意攻击能力的综合研究与比较 cs.CV

30 pages, 14 figures

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2212.10230v2) [paper-pdf](http://arxiv.org/pdf/2212.10230v2)

**Authors**: Yifan Zhang, Junhui Hou, Yixuan Yuan

**Abstract**: Recent years have witnessed significant advancements in deep learning-based 3D object detection, leading to its widespread adoption in numerous applications. As 3D object detectors become increasingly crucial for security-critical tasks, it is imperative to understand their robustness against adversarial attacks. This paper presents the first comprehensive evaluation and analysis of the robustness of LiDAR-based 3D detectors under adversarial attacks. Specifically, we extend three distinct adversarial attacks to the 3D object detection task, benchmarking the robustness of state-of-the-art LiDAR-based 3D object detectors against attacks on the KITTI and Waymo datasets. We further analyze the relationship between robustness and detector properties. Additionally, we explore the transferability of cross-model, cross-task, and cross-data attacks. Thorough experiments on defensive strategies for 3D detectors are conducted, demonstrating that simple transformations like flipping provide little help in improving robustness when the applied transformation strategy is exposed to attackers. Finally, we propose balanced adversarial focal training, based on conventional adversarial training, to strike a balance between accuracy and robustness. Our findings will facilitate investigations into understanding and defending against adversarial attacks on LiDAR-based 3D object detectors, thus advancing the field. The source code is publicly available at \url{https://github.com/Eaphan/Robust3DOD}.

摘要: 近年来，基于深度学习的3D目标检测技术取得了长足的进步，在众多应用中得到了广泛的应用。随着3D对象检测器对安全关键任务变得越来越重要，了解它们对对手攻击的稳健性是当务之急。本文首次对基于LiDAR的3D探测器在对抗攻击下的健壮性进行了全面的评估和分析。具体地说，我们将三种不同的对抗性攻击扩展到3D对象检测任务，以基准测试最先进的基于LiDAR的3D对象检测器针对Kitti和Waymo数据集的攻击的健壮性。我们进一步分析了稳健性与检测器特性之间的关系。此外，我们还探讨了跨模型、跨任务和跨数据攻击的可转移性。对3D探测器的防御策略进行了深入的实验，表明当应用的变换策略暴露给攻击者时，像翻转这样的简单变换对提高稳健性几乎没有帮助。最后，我们在传统的对抗性训练的基础上，提出了平衡的对抗性焦点训练，以在准确性和稳健性之间取得平衡。我们的发现将有助于研究理解和防御基于LiDAR的3D对象探测器的敌意攻击，从而推动该领域的发展。源代码可在\url{https://github.com/Eaphan/Robust3DOD}.}上公开获取



## **50. AdvMono3D: Advanced Monocular 3D Object Detection with Depth-Aware Robust Adversarial Training**

AdvMono3D：先进的单目3D目标检测，具有深度感知的强大对抗训练 cs.CV

**SubmitDate**: 2023-09-03    [abs](http://arxiv.org/abs/2309.01106v1) [paper-pdf](http://arxiv.org/pdf/2309.01106v1)

**Authors**: Xingyuan Li, Jinyuan Liu, Long Ma, Xin Fan, Risheng Liu

**Abstract**: Monocular 3D object detection plays a pivotal role in the field of autonomous driving and numerous deep learning-based methods have made significant breakthroughs in this area. Despite the advancements in detection accuracy and efficiency, these models tend to fail when faced with such attacks, rendering them ineffective. Therefore, bolstering the adversarial robustness of 3D detection models has become a crucial issue that demands immediate attention and innovative solutions. To mitigate this issue, we propose a depth-aware robust adversarial training method for monocular 3D object detection, dubbed DART3D. Specifically, we first design an adversarial attack that iteratively degrades the 2D and 3D perception capabilities of 3D object detection models(IDP), serves as the foundation for our subsequent defense mechanism. In response to this attack, we propose an uncertainty-based residual learning method for adversarial training. Our adversarial training approach capitalizes on the inherent uncertainty, enabling the model to significantly improve its robustness against adversarial attacks. We conducted extensive experiments on the KITTI 3D datasets, demonstrating that DART3D surpasses direct adversarial training (the most popular approach) under attacks in 3D object detection $AP_{R40}$ of car category for the Easy, Moderate, and Hard settings, with improvements of 4.415%, 4.112%, and 3.195%, respectively.

摘要: 单目三维目标检测在自动驾驶领域占有举足轻重的地位，许多基于深度学习的方法在这一领域取得了重大突破。尽管检测的准确性和效率有所提高，但这些模型在面临此类攻击时往往会失败，从而导致它们无效。因此，增强3D检测模型的对抗性已经成为一个迫切需要关注和创新解决方案的关键问题。为了缓解这一问题，我们提出了一种深度感知的稳健对抗训练方法DART3D，用于单目3D目标检测。具体地说，我们首先设计了一种对抗性攻击，迭代地降低了3D对象检测模型(IDP)的2D和3D感知能力，作为后续防御机制的基础。针对这种攻击，我们提出了一种基于不确定性的残差学习方法用于对抗性训练。我们的对抗性训练方法利用了固有的不确定性，使该模型能够显著提高其对对抗性攻击的健壮性。我们在KITTI3D数据集上进行了大量的实验，结果表明，DART3D在简单、中等和困难的设置下，在3D目标检测方面都优于CAR类别的直接对抗性训练(最流行的方法)，分别提高了4.415%、4.112%和3.195%。



