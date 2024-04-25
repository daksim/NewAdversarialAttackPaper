# Latest Adversarial Attack Papers
**update at 2024-04-25 09:46:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. OffRAMPS: An FPGA-based Intermediary for Analysis and Modification of Additive Manufacturing Control Systems**

OffRAMPS：一家基于PGA的中介机构，用于分析和修改增材制造控制系统 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15446v1) [paper-pdf](http://arxiv.org/pdf/2404.15446v1)

**Authors**: Jason Blocklove, Md Raz, Prithwish Basu Roy, Hammond Pearce, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Cybersecurity threats in Additive Manufacturing (AM) are an increasing concern as AM adoption continues to grow. AM is now being used for parts in the aerospace, transportation, and medical domains. Threat vectors which allow for part compromise are particularly concerning, as any failure in these domains would have life-threatening consequences. A major challenge to investigation of AM part-compromises comes from the difficulty in evaluating and benchmarking both identified threat vectors as well as methods for detecting adversarial actions. In this work, we introduce a generalized platform for systematic analysis of attacks against and defenses for 3D printers. Our "OFFRAMPS" platform is based on the open-source 3D printer control board "RAMPS." OFFRAMPS allows analysis, recording, and modification of all control signals and I/O for a 3D printer. We show the efficacy of OFFRAMPS by presenting a series of case studies based on several Trojans, including ones identified in the literature, and show that OFFRAMPS can both emulate and detect these attacks, i.e., it can both change and detect arbitrary changes to the g-code print commands.

摘要: 随着添加剂制造(AM)的采用持续增长，AM中的网络安全威胁日益受到关注。AM现在被用于航空航天、交通运输和医疗领域的部件。允许部分妥协的威胁载体尤其令人担忧，因为这些领域的任何失败都将产生危及生命的后果。调查AM部分妥协的一个主要挑战来自于评估和基准识别的威胁向量以及检测敌对行为的方法的困难。在这项工作中，我们介绍了一个通用的平台，用于系统地分析针对3D打印机的攻击和防御。我们的“出坡道”平台是基于开源3D打印机控制板“坡道”。Outramps允许对3D打印机的所有控制信号和I/O进行分析、记录和修改。我们通过基于几种特洛伊木马程序的一系列案例研究展示了出站攻击的有效性，其中包括文献中识别的木马程序，并表明出站出站可以模拟和检测这些攻击，即它既可以更改g代码打印命令，也可以检测对g代码打印命令的任意更改。



## **2. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

超越文本：利用人声线索改善LLM机器人导航任务的决策 cs.AI

28 pages, 7 figures

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2402.03494v2) [paper-pdf](http://arxiv.org/pdf/2402.03494v2)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present \emph{Beyond Text}; an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26\% winning rate, outperforming existing LLMs by 22.16\% to 48.30\% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44\% less decrease ratio than the text-only language model in winning rate. ``\textit{Beyond Text}'' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.

摘要: 虽然LLM在处理这些人类对话中的文本方面表现出色，但它们在社交导航等场景中难以处理语言指令的细微差别，在这些场景中，模棱两可和不确定性可能会侵蚀人们对机器人和其他人工智能系统的信任。我们可以通过超越文本并另外关注这些音频反应的副语言特征来解决这一缺点。这些特征是口语交际的方面，不涉及字面上的措辞(词汇内容)，但通过说话方式传达意义和细微差别。我们提出了一种改进LLM决策的方法，该方法通过集成音频转录和这些特征的一部分来改进LLM决策，这些特征集中在人-机器人对话中的影响和更相关的方面。该方法不仅获得了70.26\%的胜率，比现有的LLM(分别为Gemini-1.5-Pro和GPT-3.5)提高了22.16\%到48.30\%，而且还增强了对令牌操纵恶意攻击的健壮性，其突出表现是胜率比纯文本语言模型降低22.44\%。这标志着社交机器人导航和更广泛的人-机器人交互方面的进步，将基于文本的指导与人-音频信息语言模型无缝地结合在一起。



## **3. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.01318v2) [paper-pdf](http://arxiv.org/pdf/2404.01318v2)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work -- which align with OpenAI's usage policies; (3) a standardized evaluation framework that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community. Over time, we will expand and adapt the benchmark to reflect technical and methodological advances in the research community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak Bch，这是一款开源基准测试，具有以下组件：(1)不断发展的最新对抗性提示存储库，我们称之为越狱人工产物；(2)包含100种行为的越狱数据集，包括原始行为和源自先前工作的行为，这些行为与OpenAI的使用策略保持一致；(3)标准化评估框架，其中包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)跟踪各种LLM攻击和防御性能的排行榜。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。随着时间的推移，我们将扩大和调整基准，以反映研究界的技术和方法进步。



## **4. Differentially-Private Data Synthetisation for Efficient Re-Identification Risk Control**

差异私密数据合成，实现高效的重新识别风险控制 cs.LG

21 pages, 6 figures and 2 tables

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2212.00484v3) [paper-pdf](http://arxiv.org/pdf/2212.00484v3)

**Authors**: Tânia Carvalho, Nuno Moniz, Luís Antunes, Nitesh Chawla

**Abstract**: Protecting user data privacy can be achieved via many methods, from statistical transformations to generative models. However, all of them have critical drawbacks. For example, creating a transformed data set using traditional techniques is highly time-consuming. Also, recent deep learning-based solutions require significant computational resources in addition to long training phases, and differentially private-based solutions may undermine data utility. In this paper, we propose $\epsilon$-PrivateSMOTE, a technique designed for safeguarding against re-identification and linkage attacks, particularly addressing cases with a high \sloppy re-identification risk. Our proposal combines synthetic data generation via noise-induced interpolation with differential privacy principles to obfuscate high-risk cases. We demonstrate how $\epsilon$-PrivateSMOTE is capable of achieving competitive results in privacy risk and better predictive performance when compared to multiple traditional and state-of-the-art privacy-preservation methods, including generative adversarial networks, variational autoencoders, and differential privacy baselines. We also show how our method improves time requirements by at least a factor of 9 and is a resource-efficient solution that ensures high performance without specialised hardware.

摘要: 保护用户数据隐私可以通过许多方法实现，从统计转换到生成模型。然而，它们都有严重的缺陷。例如，使用传统技术创建转换后的数据集非常耗时。此外，最近基于深度学习的解决方案除了需要较长的培训阶段外，还需要大量的计算资源，而不同的基于私人的解决方案可能会破坏数据效用。在本文中，我们提出了$-PrivateSMOTE，这是一种旨在防止重新识别和链接攻击的技术，特别是针对重新识别风险较高的情况。我们的方案将通过噪声诱导内插生成的合成数据与差分隐私原则相结合来混淆高危案例。我们展示了$\epsilon$-PrivateSMOTE与多种传统和最先进的隐私保护方法相比，如何能够在隐私风险和更好的预测性能方面实现竞争结果，这些方法包括生成性对抗网络、可变自动编码器和差异隐私基线。我们还展示了我们的方法如何将时间需求提高至少9倍，并且是一种资源高效的解决方案，无需专门的硬件即可确保高性能。



## **5. Problem space structural adversarial attacks for Network Intrusion Detection Systems based on Graph Neural Networks**

基于图神经网络的网络入侵检测系统的问题空间结构对抗攻击 cs.CR

preprint submitted to IEEE TIFS, under review

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2403.11830v2) [paper-pdf](http://arxiv.org/pdf/2403.11830v2)

**Authors**: Andrea Venturi, Dario Stabili, Mirco Marchetti

**Abstract**: Machine Learning (ML) algorithms have become increasingly popular for supporting Network Intrusion Detection Systems (NIDS). Nevertheless, extensive research has shown their vulnerability to adversarial attacks, which involve subtle perturbations to the inputs of the models aimed at compromising their performance. Recent proposals have effectively leveraged Graph Neural Networks (GNN) to produce predictions based also on the structural patterns exhibited by intrusions to enhance the detection robustness. However, the adoption of GNN-based NIDS introduces new types of risks. In this paper, we propose the first formalization of adversarial attacks specifically tailored for GNN in network intrusion detection. Moreover, we outline and model the problem space constraints that attackers need to consider to carry out feasible structural attacks in real-world scenarios. As a final contribution, we conduct an extensive experimental campaign in which we launch the proposed attacks against state-of-the-art GNN-based NIDS. Our findings demonstrate the increased robustness of the models against classical feature-based adversarial attacks, while highlighting their susceptibility to structure-based attacks.

摘要: 机器学习(ML)算法因支持网络入侵检测系统(NIDS)而变得越来越流行。然而，广泛的研究表明，它们在对抗性攻击中的脆弱性，这涉及到对模型的输入进行微妙的扰动，目的是降低它们的性能。最近的建议已经有效地利用图神经网络(GNN)来产生基于入侵表现出的结构模式的预测，以增强检测的稳健性。然而，采用基于GNN的网络入侵检测系统带来了新的风险类型。在本文中，我们首次提出了网络入侵检测中专门针对GNN的对抗性攻击的形式化描述。此外，我们概述并模拟了攻击者在现实世界场景中执行可行的结构性攻击所需考虑的问题空间约束。作为最后的贡献，我们进行了一项广泛的实验活动，在该活动中，我们对最先进的基于GNN的网络入侵检测系统发起了拟议的攻击。我们的研究结果表明，该模型在抵抗经典的基于特征的对抗性攻击时具有更强的稳健性，同时突出了它们对基于结构的攻击的敏感性。



## **6. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

ALI-DPFL：具有自适应本地迭代的差异化私人联邦学习 cs.LG

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2308.10457v6) [paper-pdf](http://arxiv.org/pdf/2308.10457v6)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.

摘要: 联合学习(FL)是一种分布式机器学习技术，通过共享训练参数而不是原始数据，允许在多个设备或组织之间进行模型训练。然而，攻击者仍然可以通过对这些训练参数的推理攻击(例如差异攻击)来推断个人信息。因此，差分隐私(DP)被广泛应用于FL中以防止此类攻击。我们考虑在资源受限的情况下进行不同的私有联合学习，其中隐私预算和通信回合都受到限制。通过对收敛的理论分析，我们可以找到任意两个连续全局更新之间客户端的最优局部DPSGD迭代次数。在此基础上，设计了一种基于自适应局部迭代的差分私有联邦学习算法(ALI-DPFL)。我们在MNIST、FashionMNIST和Cifar10数据集上测试了我们的算法，并在资源受限的情况下展示了比以前的工作更好的性能。代码可在https://github.com/KnightWan/ALI-DPFL.上找到



## **7. Perturbing Attention Gives You More Bang for the Buck: Subtle Imaging Perturbations That Efficiently Fool Customized Diffusion Models**

扰动注意力为您带来更多好处：有效愚弄定制扩散模型的微妙成像扰动 cs.CV

Published at CVPR 2024

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15081v1) [paper-pdf](http://arxiv.org/pdf/2404.15081v1)

**Authors**: Jingyao Xu, Yuetong Lu, Yandong Li, Siyang Lu, Dongdong Wang, Xiang Wei

**Abstract**: Diffusion models (DMs) embark a new era of generative modeling and offer more opportunities for efficient generating high-quality and realistic data samples. However, their widespread use has also brought forth new challenges in model security, which motivates the creation of more effective adversarial attackers on DMs to understand its vulnerability. We propose CAAT, a simple but generic and efficient approach that does not require costly training to effectively fool latent diffusion models (LDMs). The approach is based on the observation that cross-attention layers exhibits higher sensitivity to gradient change, allowing for leveraging subtle perturbations on published images to significantly corrupt the generated images. We show that a subtle perturbation on an image can significantly impact the cross-attention layers, thus changing the mapping between text and image during the fine-tuning of customized diffusion models. Extensive experiments demonstrate that CAAT is compatible with diverse diffusion models and outperforms baseline attack methods in a more effective (more noise) and efficient (twice as fast as Anti-DreamBooth and Mist) manner.

摘要: 扩散模型开启了产生式建模的新时代，为高效地生成高质量和真实的数据样本提供了更多的机会。然而，它们的广泛使用也给模型安全带来了新的挑战，这促使在DM上创建更有效的对抗性攻击者来了解其脆弱性。我们提出了CAAT，这是一种简单但通用和高效的方法，不需要昂贵的培训来有效地愚弄潜在扩散模型(LDM)。该方法的基础是观察到交叉注意层对梯度变化表现出更高的敏感性，允许利用发布图像上的细微扰动来显著破坏生成的图像。我们发现，在定制扩散模型的微调过程中，图像上的细微扰动会显著影响交叉注意层，从而改变文本和图像之间的映射。大量的实验表明，CAAT与多种扩散模型兼容，并且在更有效(更多噪声)和更高效(速度是Anti-DreamBooth和Mist的两倍)方面优于基线攻击方法。



## **8. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

具有不确定节点特征和不确定图结构的图卷积网络的形式化验证 cs.LG

under review

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15065v1) [paper-pdf](http://arxiv.org/pdf/2404.15065v1)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.

摘要: 图神经网络因其处理以图结构化的数据的独特能力而在机器学习领域变得越来越受欢迎。它们还应用于固有地发生扰动的安全关键环境中。然而，这些扰动需要我们在将神经网络部署到安全关键环境中之前对其进行正式验证，因为神经网络容易受到对抗性攻击。虽然存在关于神经网络形式验证的研究，但还没有任何工作验证通用图卷积网络架构的稳健性，因为节点特征和多个消息传递步骤中的图结构存在不确定性。这项工作通过使用（矩阵）多项分区的可达性分析明确保留基础计算中所有元素的非凸依赖性来解决这一研究空白。我们在三个流行的基准数据集上展示了我们的方法。



## **9. Leverage Variational Graph Representation For Model Poisoning on Federated Learning**

利用变分图表示解决联邦学习中的模型中毒问题 cs.CR

12 pages, 8 figures, 2 tables

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.15042v1) [paper-pdf](http://arxiv.org/pdf/2404.15042v1)

**Authors**: Kai Li, Xin Yuan, Jingjing Zheng, Wei Ni, Falko Dressler, Abbas Jamalipour

**Abstract**: This paper puts forth a new training data-untethered model poisoning (MP) attack on federated learning (FL). The new MP attack extends an adversarial variational graph autoencoder (VGAE) to create malicious local models based solely on the benign local models overheard without any access to the training data of FL. Such an advancement leads to the VGAE-MP attack that is not only efficacious but also remains elusive to detection. VGAE-MP attack extracts graph structural correlations among the benign local models and the training data features, adversarially regenerates the graph structure, and generates malicious local models using the adversarial graph structure and benign models' features. Moreover, a new attacking algorithm is presented to train the malicious local models using VGAE and sub-gradient descent, while enabling an optimal selection of the benign local models for training the VGAE. Experiments demonstrate a gradual drop in FL accuracy under the proposed VGAE-MP attack and the ineffectiveness of existing defense mechanisms in detecting the attack, posing a severe threat to FL.

摘要: 提出了一种新的训练数据--非拴系模型中毒(MP)联合学习攻击。新的MP攻击扩展了对抗性变分图自动编码器(VGAE)，仅基于在没有访问FL训练数据的情况下偷听到的良性局部模型来创建恶意局部模型。这样的进步导致了VGAE-MP攻击，这种攻击不仅有效，而且仍然难以检测。VGAE-MP攻击提取良性局部模型和训练数据特征之间的图结构相关性，恶意重建图结构，并利用对抗性图结构和良性模型特征生成恶意局部模型。此外，提出了一种新的攻击算法，利用VGAE和次梯度下降来训练恶意局部模型，同时使良性局部模型能够最优地选择用于训练VGAE的局部模型。实验表明，在提出的VGAE-MP攻击下，FL的准确率逐渐下降，并且现有的防御机制在检测攻击时效率低下，对FL构成了严重的威胁。



## **10. Manipulating Recommender Systems: A Survey of Poisoning Attacks and Countermeasures**

操纵推荐系统：中毒攻击及对策调查 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14942v1) [paper-pdf](http://arxiv.org/pdf/2404.14942v1)

**Authors**: Thanh Toan Nguyen, Quoc Viet Hung Nguyen, Thanh Tam Nguyen, Thanh Trung Huynh, Thanh Thi Nguyen, Matthias Weidlich, Hongzhi Yin

**Abstract**: Recommender systems have become an integral part of online services to help users locate specific information in a sea of data. However, existing studies show that some recommender systems are vulnerable to poisoning attacks, particularly those that involve learning schemes. A poisoning attack is where an adversary injects carefully crafted data into the process of training a model, with the goal of manipulating the system's final recommendations. Based on recent advancements in artificial intelligence, such attacks have gained importance recently. While numerous countermeasures to poisoning attacks have been developed, they have not yet been systematically linked to the properties of the attacks. Consequently, assessing the respective risks and potential success of mitigation strategies is difficult, if not impossible. This survey aims to fill this gap by primarily focusing on poisoning attacks and their countermeasures. This is in contrast to prior surveys that mainly focus on attacks and their detection methods. Through an exhaustive literature review, we provide a novel taxonomy for poisoning attacks, formalise its dimensions, and accordingly organise 30+ attacks described in the literature. Further, we review 40+ countermeasures to detect and/or prevent poisoning attacks, evaluating their effectiveness against specific types of attacks. This comprehensive survey should serve as a point of reference for protecting recommender systems against poisoning attacks. The article concludes with a discussion on open issues in the field and impactful directions for future research. A rich repository of resources associated with poisoning attacks is available at https://github.com/tamlhp/awesome-recsys-poisoning.

摘要: 推荐系统已经成为帮助用户在海量数据中定位特定信息的在线服务的组成部分。然而，现有的研究表明，一些推荐系统容易受到中毒攻击，特别是那些涉及学习方案的系统。中毒攻击是指对手在训练模型的过程中注入精心制作的数据，目的是操纵系统的最终建议。基于人工智能的最新进展，这类攻击最近变得越来越重要。虽然已经制定了许多针对中毒攻击的对策，但它们尚未系统地与攻击的性质联系起来。因此，评估缓解战略的各自风险和潜在成功是困难的，如果不是不可能的话。这项调查旨在通过主要关注中毒攻击及其对策来填补这一空白。这与以往主要关注攻击及其检测方法的调查形成了鲜明对比。通过详尽的文献回顾，我们为中毒攻击提供了一种新的分类，确定了其规模，并相应地组织了文献中描述的30+次攻击。此外，我们还回顾了40多种检测和/或预防中毒攻击的对策，评估了它们对特定类型攻击的有效性。这一全面的调查应该作为保护推荐系统免受中毒攻击的参考点。文章最后对该领域存在的问题进行了讨论，并指出了未来研究的方向。Https://github.com/tamlhp/awesome-recsys-poisoning.上提供了与中毒攻击相关的丰富资源存储库



## **11. Adaptive Hybrid Masking Strategy for Privacy-Preserving Face Recognition Against Model Inversion Attack**

抗模型倒置攻击的隐私保护人脸识别自适应混合掩蔽策略 cs.CV

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2403.10558v2) [paper-pdf](http://arxiv.org/pdf/2403.10558v2)

**Authors**: Yinggui Wang, Yuanqing Huang, Jianshu Li, Le Yang, Kai Song, Lei Wang

**Abstract**: The utilization of personal sensitive data in training face recognition (FR) models poses significant privacy concerns, as adversaries can employ model inversion attacks (MIA) to infer the original training data. Existing defense methods, such as data augmentation and differential privacy, have been employed to mitigate this issue. However, these methods often fail to strike an optimal balance between privacy and accuracy. To address this limitation, this paper introduces an adaptive hybrid masking algorithm against MIA. Specifically, face images are masked in the frequency domain using an adaptive MixUp strategy. Unlike the traditional MixUp algorithm, which is predominantly used for data augmentation, our modified approach incorporates frequency domain mixing. Previous studies have shown that increasing the number of images mixed in MixUp can enhance privacy preservation but at the expense of reduced face recognition accuracy. To overcome this trade-off, we develop an enhanced adaptive MixUp strategy based on reinforcement learning, which enables us to mix a larger number of images while maintaining satisfactory recognition accuracy. To optimize privacy protection, we propose maximizing the reward function (i.e., the loss function of the FR system) during the training of the strategy network. While the loss function of the FR network is minimized in the phase of training the FR network. The strategy network and the face recognition network can be viewed as antagonistic entities in the training process, ultimately reaching a more balanced trade-off. Experimental results demonstrate that our proposed hybrid masking scheme outperforms existing defense algorithms in terms of privacy preservation and recognition accuracy against MIA.

摘要: 在训练人脸识别(FR)模型中使用个人敏感数据引起了严重的隐私问题，因为攻击者可以使用模型反转攻击(MIA)来推断原始训练数据。现有的防御方法，如数据增强和差异隐私，已经被用来缓解这个问题。然而，这些方法往往无法在隐私和准确性之间取得最佳平衡。针对这一局限性，提出了一种针对MIA的自适应混合掩蔽算法。具体地说，人脸图像在频域中使用自适应混合策略进行掩蔽。与主要用于数据增强的传统混合算法不同，我们的改进方法结合了频域混合。以前的研究表明，增加混合图像的数量可以增强隐私保护，但代价是降低人脸识别的准确性。为了克服这种权衡，我们开发了一种基于强化学习的增强的自适应混合策略，它使我们能够混合更多的图像，同时保持令人满意的识别精度。为了优化隐私保护，我们提出在策略网络的训练过程中最大化奖励函数(即FR系统的损失函数)。在训练FR网络的过程中，使FR网络的损失函数最小。在训练过程中，策略网络和人脸识别网络可以被视为对立的实体，最终达到更平衡的权衡。实验结果表明，本文提出的混合掩蔽方案在隐私保护和MIA识别准确率方面优于已有的防御算法。



## **12. Double Privacy Guard: Robust Traceable Adversarial Watermarking against Face Recognition**

双重隐私保护：针对人脸识别的稳健可追溯对抗水印 cs.CR

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14693v1) [paper-pdf](http://arxiv.org/pdf/2404.14693v1)

**Authors**: Yunming Zhang, Dengpan Ye, Sipeng Shen, Caiyun Xie, Ziyi Liu, Jiacheng Deng, Long Tang

**Abstract**: The wide deployment of Face Recognition (FR) systems poses risks of privacy leakage. One countermeasure to address this issue is adversarial attacks, which deceive malicious FR searches but simultaneously interfere the normal identity verification of trusted authorizers. In this paper, we propose the first Double Privacy Guard (DPG) scheme based on traceable adversarial watermarking. DPG employs a one-time watermark embedding to deceive unauthorized FR models and allows authorizers to perform identity verification by extracting the watermark. Specifically, we propose an information-guided adversarial attack against FR models. The encoder embeds an identity-specific watermark into the deep feature space of the carrier, guiding recognizable features of the image to deviate from the source identity. We further adopt a collaborative meta-optimization strategy compatible with sub-tasks, which regularizes the joint optimization direction of the encoder and decoder. This strategy enhances the representation of universal carrier features, mitigating multi-objective optimization conflicts in watermarking. Experiments confirm that DPG achieves significant attack success rates and traceability accuracy on state-of-the-art FR models, exhibiting remarkable robustness that outperforms the existing privacy protection methods using adversarial attacks and deep watermarking, or simple combinations of the two. Our work potentially opens up new insights into proactive protection for FR privacy.

摘要: 人脸识别(FR)系统的广泛应用带来了隐私泄露的风险。解决这个问题的一种对策是对抗性攻击，它欺骗恶意FR搜索，但同时干扰受信任授权者的正常身份验证。本文提出了第一个基于可追踪对抗水印的双重隐私保护(DPG)方案。DPG采用一次性水印嵌入来欺骗未经授权的FR模型，并允许授权者通过提取水印来进行身份验证。具体地说，我们提出了一种针对FR模型的信息制导的对抗性攻击。编码器将特定于身份的水印嵌入到载体的深层特征空间中，引导图像的可识别特征偏离源身份。进一步采用了和子任务兼容的协作元优化策略，规范了编解码器的联合优化方向。该策略增强了对通用载体特征的表示，缓解了水印中的多目标优化冲突。实验证实，DPG在最先进的FR模型上获得了显著的攻击成功率和可追踪性准确性，表现出显著的稳健性，其性能优于现有的使用对抗性攻击和深度水印的隐私保护方法，或两者的简单组合。我们的工作可能为主动保护FR隐私打开新的洞察力。



## **13. Pseudorandom Permutations from Random Reversible Circuits**

随机可逆电路的伪随机排列 cs.CC

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2404.14648v1) [paper-pdf](http://arxiv.org/pdf/2404.14648v1)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).

摘要: 我们研究了由可逆$3$位门($0，1^3$上的置换)构成的随机电路计算的$0，1^n上置换的伪随机性。我们的主要结果是，一个深度为$n\cot\tide{O}(k^2)$的随机电路，每一层由固定最近邻体系结构中的$\约n/3$随机门组成，产生几乎$k$方向的独立排列。主要的技术内容是证明了由单个随机的$3$比特最近邻门产生的$n$比特串的$k$-元组上的马尔可夫链至少有$1/n\cdot\tilde{O}(K)$。这比Gowers[Gowers96]的原始工作有所改进，Gowers[Gowers96]对一个随机门(具有非相邻输入)显示了$1/\mathm{pol}(n，k)$的差距；在随后的工作[HMMR05，BH08]中，在相同设置下将差距改进为$\Omega(1/n^2k)$。从密码学的角度来看，我们的结果可以看作是一种特别简单实用的分组密码构造，它提供了针对在几轮内访问$k$~输入输出对的攻击者的可证明的统计安全性。我们还证明了伪随机函数的伪随机置换的Luby-Rackoff构造可以用可逆电路实现。由此，我们在最小可逆电路大小问题(MRCSP)的复杂性方面取得了进展，表明在假设存在单向函数(OWF)的情况下，固定多项式大小的分组密码在计算上是安全的，可以抵抗任意多项式时间的攻击者。



## **14. RETVec: Resilient and Efficient Text Vectorizer**

RETVec：弹性且高效的文本Vectorizer cs.CL

37th Conference on Neural Information Processing Systems (NeurIPS  2023)

**SubmitDate**: 2024-04-23    [abs](http://arxiv.org/abs/2302.09207v3) [paper-pdf](http://arxiv.org/pdf/2302.09207v3)

**Authors**: Elie Bursztein, Marina Zhang, Owen Vallis, Xinyu Jia, Alexey Kurakin

**Abstract**: This paper describes RETVec, an efficient, resilient, and multilingual text vectorizer designed for neural-based text processing. RETVec combines a novel character encoding with an optional small embedding model to embed words into a 256-dimensional vector space. The RETVec embedding model is pre-trained using pair-wise metric learning to be robust against typos and character-level adversarial attacks. In this paper, we evaluate and compare RETVec to state-of-the-art vectorizers and word embeddings on popular model architectures and datasets. These comparisons demonstrate that RETVec leads to competitive, multilingual models that are significantly more resilient to typos and adversarial text attacks. RETVec is available under the Apache 2 license at https://github.com/google-research/retvec.

摘要: 本文描述了RETVec，这是一种高效、有弹性、多语言的文本载体器，专为基于神经的文本处理而设计。RETVec将新颖的字符编码与可选的小型嵌入模型相结合，将单词嵌入到256维载体空间中。RETVec嵌入模型是使用成对度量学习进行预训练的，以对抗错别字和字符级对抗攻击。在本文中，我们评估并比较了RETVec与流行模型架构和数据集上的最先进的载体器和单词嵌入。这些比较表明，RETVec带来了有竞争力的多语言模型，这些模型对拼写错误和对抗性文本攻击的弹性明显更强。RETVec可在https://github.com/google-research/retvec上使用Apach2许可证。



## **15. Image Hijacks: Adversarial Images can Control Generative Models at Runtime**

图像劫持：对抗图像可以随时控制生成模型 cs.LG

Project page at https://image-hijacks.github.io

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2309.00236v3) [paper-pdf](http://arxiv.org/pdf/2309.00236v3)

**Authors**: Luke Bailey, Euan Ong, Stuart Russell, Scott Emmons

**Abstract**: Are foundation models secure against malicious actors? In this work, we focus on the image input to a vision-language model (VLM). We discover image hijacks, adversarial images that control the behaviour of VLMs at inference time, and introduce the general Behaviour Matching algorithm for training image hijacks. From this, we derive the Prompt Matching method, allowing us to train hijacks matching the behaviour of an arbitrary user-defined text prompt (e.g. 'the Eiffel Tower is now located in Rome') using a generic, off-the-shelf dataset unrelated to our choice of prompt. We use Behaviour Matching to craft hijacks for four types of attack, forcing VLMs to generate outputs of the adversary's choice, leak information from their context window, override their safety training, and believe false statements. We study these attacks against LLaVA, a state-of-the-art VLM based on CLIP and LLaMA-2, and find that all attack types achieve a success rate of over 80%. Moreover, our attacks are automated and require only small image perturbations.

摘要: 基础模型针对恶意行为者是否安全？在这项工作中，我们关注的是图像输入到视觉语言模型(VLM)。我们发现了图像劫持，即在推理时控制VLM行为的敌意图像，并介绍了用于训练图像劫持的通用行为匹配算法。从这里，我们得到了提示匹配方法，允许我们使用与我们选择的提示无关的通用现成数据集来训练与任意用户定义的文本提示(例如‘埃菲尔铁塔现在位于罗马’)的行为匹配的劫持者。我们使用行为匹配来为四种类型的攻击制作劫持，迫使VLM生成对手选择的输出，从他们的上下文窗口泄露信息，覆盖他们的安全培训，并相信虚假陈述。我们对基于CLIP和LLAMA-2的最新VLM LLaVA进行了研究，发现所有类型的攻击都达到了80%以上的成功率。此外，我们的攻击是自动化的，只需要很小的图像扰动。



## **16. An Adversarial Approach to Evaluating the Robustness of Event Identification Models**

评估事件识别模型稳健性的对抗方法 eess.SY

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2402.12338v2) [paper-pdf](http://arxiv.org/pdf/2402.12338v2)

**Authors**: Obai Bahwal, Oliver Kosut, Lalitha Sankar

**Abstract**: Intelligent machine learning approaches are finding active use for event detection and identification that allow real-time situational awareness. Yet, such machine learning algorithms have been shown to be susceptible to adversarial attacks on the incoming telemetry data. This paper considers a physics-based modal decomposition method to extract features for event classification and focuses on interpretable classifiers including logistic regression and gradient boosting to distinguish two types of events: load loss and generation loss. The resulting classifiers are then tested against an adversarial algorithm to evaluate their robustness. The adversarial attack is tested in two settings: the white box setting, wherein the attacker knows exactly the classification model; and the gray box setting, wherein the attacker has access to historical data from the same network as was used to train the classifier, but does not know the classification model. Thorough experiments on the synthetic South Carolina 500-bus system highlight that a relatively simpler model such as logistic regression is more susceptible to adversarial attacks than gradient boosting.

摘要: 智能机器学习方法正在积极应用于事件检测和识别，从而实现实时的态势感知。然而，这种机器学习算法已被证明容易受到对传入遥测数据的敌意攻击。本文考虑了一种基于物理的模式分解方法来提取事件分类的特征，并重点使用Logistic回归和梯度提升等可解释分类器来区分两种类型的事件：负荷损失和发电损失。然后，将得到的分类器与对抗性算法进行测试，以评估它们的稳健性。在两种设置中测试对抗性攻击：白盒设置，其中攻击者确切地知道分类模型；以及灰盒设置，其中攻击者可以访问来自用于训练分类器的相同网络的历史数据，但不知道分类模型。在合成的南卡罗来纳州500母线系统上进行的彻底实验表明，相对简单的模型，如Logistic回归，比梯度助推更容易受到对抗性攻击。



## **17. Automatic Discovery of Visual Circuits**

视觉回路的自动发现 cs.CV

14 pages, 11 figures

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14349v1) [paper-pdf](http://arxiv.org/pdf/2404.14349v1)

**Authors**: Achyuta Rajaram, Neil Chowdhury, Antonio Torralba, Jacob Andreas, Sarah Schwettmann

**Abstract**: To date, most discoveries of network subcomponents that implement human-interpretable computations in deep vision models have involved close study of single units and large amounts of human labor. We explore scalable methods for extracting the subgraph of a vision model's computational graph that underlies recognition of a specific visual concept. We introduce a new method for identifying these subgraphs: specifying a visual concept using a few examples, and then tracing the interdependence of neuron activations across layers, or their functional connectivity. We find that our approach extracts circuits that causally affect model output, and that editing these circuits can defend large pretrained models from adversarial attacks.

摘要: 迄今为止，大多数在深度视觉模型中实现人类可解释计算的网络子组件的发现都涉及对单个单元和大量人力的密切研究。我们探索提取视觉模型计算图的子图的可扩展方法，该计算图是特定视觉概念识别的基础。我们引入了一种识别这些子图的新方法：使用几个例子指定视觉概念，然后追踪各层神经元激活的相互依赖性，或其功能连接性。我们发现我们的方法提取了对模型输出产生因果影响的电路，并且编辑这些电路可以保护大型预训练模型免受对抗攻击。



## **18. Towards Better Adversarial Purification via Adversarial Denoising Diffusion Training**

通过对抗去噪扩散训练实现更好的对抗净化 cs.CV

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14309v1) [paper-pdf](http://arxiv.org/pdf/2404.14309v1)

**Authors**: Yiming Liu, Kezhao Liu, Yao Xiao, Ziyi Dong, Xiaogang Xu, Pengxu Wei, Liang Lin

**Abstract**: Recently, diffusion-based purification (DBP) has emerged as a promising approach for defending against adversarial attacks. However, previous studies have used questionable methods to evaluate the robustness of DBP models, their explanations of DBP robustness also lack experimental support. We re-examine DBP robustness using precise gradient, and discuss the impact of stochasticity on DBP robustness. To better explain DBP robustness, we assess DBP robustness under a novel attack setting, Deterministic White-box, and pinpoint stochasticity as the main factor in DBP robustness. Our results suggest that DBP models rely on stochasticity to evade the most effective attack direction, rather than directly countering adversarial perturbations. To improve the robustness of DBP models, we propose Adversarial Denoising Diffusion Training (ADDT). This technique uses Classifier-Guided Perturbation Optimization (CGPO) to generate adversarial perturbation through guidance from a pre-trained classifier, and uses Rank-Based Gaussian Mapping (RBGM) to convert adversarial pertubation into a normal Gaussian distribution. Empirical results show that ADDT improves the robustness of DBP models. Further experiments confirm that ADDT equips DBP models with the ability to directly counter adversarial perturbations.

摘要: 近年来，基于扩散的纯化技术(DBP)已成为一种很有前途的防御敌意攻击的方法。然而，以前的研究使用了有问题的方法来评估DBP模型的稳健性，它们对DBP模型的稳健性的解释也缺乏实验支持。我们使用精确梯度重新检验了DBP稳健性，并讨论了随机性对DBP稳健性的影响。为了更好地解释DBP的稳健性，我们评估了一种新的攻击环境下的DBP稳健性，即确定性白盒，并指出随机性是影响DBP稳健性的主要因素。我们的结果表明，DBP模型依赖于随机性来避开最有效的攻击方向，而不是直接对抗对手的扰动。为了提高DBP模型的稳健性，我们提出了对抗性去噪扩散训练方法。该技术使用分类器引导的扰动优化(CGPO)通过预先训练的分类器的引导来产生对抗性扰动，并使用基于等级的高斯映射(RBGM)将对抗性扰动转换为正态高斯分布。实证结果表明，ADDT提高了DBP模型的稳健性。进一步的实验证实，ADDT使DBP模型具有直接对抗对抗性扰动的能力。



## **19. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14250v1) [paper-pdf](http://arxiv.org/pdf/2404.14250v1)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **20. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

利用强化学习进行黑匣子图像、视频和心电图信号分类的鲁棒性和视觉解释 cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2403.18985v2) [paper-pdf](http://arxiv.org/pdf/2403.18985v2)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.

摘要: 我们提出了一个通用的强化学习（RL）框架，经过优化，用于对来自心电图信号分析（1D）、图像分类（2D）和视频分类（3D）的不同模型类型进行对抗攻击。该框架的重点是识别敏感区域并以最小的失真和各种失真类型引发错误分类。对于所有三种应用，新型RL方法的性能都优于最先进的方法，证明了其效率。我们的RL方法产生了卓越的定位模板，增强了图像分类和心电图分析模型的可解释性。对于心电图分析等应用，我们的平台为临床医生突出显示关键的心电图段，同时确保针对普遍失真的弹性。这个全面的工具旨在通过对抗性培训和各种应用程序和数据类型的透明度来增强弹性。



## **21. Secure compilation of rich smart contracts on poor UTXO blockchains**

在较差的UTXO区块链上安全编写丰富的智能合同 cs.CR

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2305.09545v3) [paper-pdf](http://arxiv.org/pdf/2305.09545v3)

**Authors**: Massimo Bartoletti, Riccardo Marchesin, Roberto Zunino

**Abstract**: Most blockchain platforms from Ethereum onwards render smart contracts as stateful reactive objects that update their state and transfer crypto-assets in response to transactions. A drawback of this design is that when users submit a transaction, they cannot predict in which state it will be executed. This exposes them to transaction-ordering attacks, a widespread class of attacks where adversaries with the power to construct blocks of transactions can extract value from smart contracts (the so-called MEV attacks). The UTXO model is an alternative blockchain design that thwarts these attacks by requiring new transactions to spend past ones: since transactions have unique identifiers, reordering attacks are ineffective. Currently, the blockchains following the UTXO model either provide contracts with limited expressiveness (Bitcoin), or require complex run-time environments (Cardano). We present ILLUM , an Intermediate-Level Language for the UTXO Model. ILLUM can express real-world smart contracts, e.g. those found in Decentralized Finance. We define a compiler from ILLUM to a bare-bone UTXO blockchain with loop-free scripts. Our compilation target only requires minimal extensions to Bitcoin Script: in particular, we exploit covenants, a mechanism for preserving scripts along chains of transactions. We prove the security of our compiler: namely, any attack targeting the compiled contract is also observable at the ILLUM level. Hence, the compiler does not introduce new vulnerabilities that were not already present in the source ILLUM contract. We evaluate the practicality of ILLUM as a compilation target for higher-level languages. To this purpose, we implement a compiler from a contract language inspired by Solidity to ILLUM, and we apply it to a benchmark or real-world smart contracts.

摘要: 从Etherum开始，大多数区块链平台都将智能合约呈现为有状态的反应对象，这些对象更新其状态并传输加密资产以响应交易。这种设计的一个缺点是，当用户提交事务时，他们无法预测该事务将在哪种状态下执行。这使他们面临交易顺序攻击，这是一种广泛存在的攻击类别，在这种攻击中，有能力构建交易块的对手可以从智能合约中提取价值(所谓的MEV攻击)。UTXO模型是一种替代区块链设计，通过要求新交易花费过去的交易来挫败这些攻击：由于交易具有唯一标识符，重新排序攻击是无效的。目前，遵循UTXO模式的区块链要么提供表现力有限的合同(比特币)，要么需要复杂的运行时环境(Cardano)。我们介绍了Illum，一种用于UTXO模型的中级语言。Illum可以表示现实世界中的智能合约，例如在去中心化金融中找到的那些。我们定义了一个从Illum到具有无循环脚本的基本UTXO区块链的编译器。我们的编译目标只需要对比特币脚本进行最小程度的扩展：尤其是，我们利用了契诺，这是一种在交易链上保留脚本的机制。我们证明了我们的编译器的安全性：也就是说，任何针对已编译约定的攻击也可以在Illum级别上观察到。因此，编译器不会引入源Illum协定中尚未存在的新漏洞。我们评估了Illum作为高级语言编译目标的实用性。为此，我们实现了一个编译器，从一种受Solidity启发的契约语言到Illum，并将其应用于基准或现实世界的智能合约。



## **22. Protecting Your LLMs with Information Bottleneck**

通过信息瓶颈保护您的LLC cs.CL

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13968v1) [paper-pdf](http://arxiv.org/pdf/2404.13968v1)

**Authors**: Zichuan Liu, Zefan Wang, Linjie Xu, Jinyu Wang, Lei Song, Tianchun Wang, Chunlin Chen, Wei Cheng, Jiang Bian

**Abstract**: The advent of large language models (LLMs) has revolutionized the field of natural language processing, yet they might be attacked to produce harmful content. Despite efforts to ethically align LLMs, these are often fragile and can be circumvented by jailbreaking attacks through optimized or manual adversarial prompts. To address this, we introduce the Information Bottleneck Protector (IBProtector), a defense mechanism grounded in the information bottleneck principle, and we modify the objective to avoid trivial solutions. The IBProtector selectively compresses and perturbs prompts, facilitated by a lightweight and trainable extractor, preserving only essential information for the target LLMs to respond with the expected answer. Moreover, we further consider a situation where the gradient is not visible to be compatible with any LLM. Our empirical evaluations show that IBProtector outperforms current defense methods in mitigating jailbreak attempts, without overly affecting response quality or inference speed. Its effectiveness and adaptability across various attack methods and target LLMs underscore the potential of IBProtector as a novel, transferable defense that bolsters the security of LLMs without requiring modifications to the underlying models.

摘要: 大型语言模型的出现给自然语言处理领域带来了革命性的变化，但它们可能会受到攻击，产生有害的内容。尽管努力在道德上调整LLM，但这些往往是脆弱的，可以通过优化或手动对抗性提示通过越狱攻击来绕过。为了解决这个问题，我们引入了信息瓶颈保护器(IBProtector)，这是一种基于信息瓶颈原理的防御机制，我们修改了目标以避免琐碎的解决方案。IBProtector有选择地压缩和干扰提示，由一个轻量级和可训练的提取程序促进，只保留目标LLMS的基本信息，以响应预期的答案。此外，我们还进一步考虑了梯度不可见的情况，以与任何LLM相容。我们的经验评估表明，在不过度影响响应质量或推理速度的情况下，IBProtector在缓解越狱企图方面优于现有的防御方法。它对各种攻击方法和目标LLM的有效性和适应性突显了IBProtector作为一种新型、可转移的防御系统的潜力，无需修改底层模型即可增强LLM的安全性。



## **23. Audio Anti-Spoofing Detection: A Survey**

音频反欺骗检测：调查 cs.SD

submitted to ACM Computing Surveys

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13914v1) [paper-pdf](http://arxiv.org/pdf/2404.13914v1)

**Authors**: Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang

**Abstract**: The availability of smart devices leads to an exponential increase in multimedia content. However, the rapid advancements in deep learning have given rise to sophisticated algorithms capable of manipulating or creating multimedia fake content, known as Deepfake. Audio Deepfakes pose a significant threat by producing highly realistic voices, thus facilitating the spread of misinformation. To address this issue, numerous audio anti-spoofing detection challenges have been organized to foster the development of anti-spoofing countermeasures. This survey paper presents a comprehensive review of every component within the detection pipeline, including algorithm architectures, optimization techniques, application generalizability, evaluation metrics, performance comparisons, available datasets, and open-source availability. For each aspect, we conduct a systematic evaluation of the recent advancements, along with discussions on existing challenges. Additionally, we also explore emerging research topics on audio anti-spoofing, including partial spoofing detection, cross-dataset evaluation, and adversarial attack defence, while proposing some promising research directions for future work. This survey paper not only identifies the current state-of-the-art to establish strong baselines for future experiments but also guides future researchers on a clear path for understanding and enhancing the audio anti-spoofing detection mechanisms.

摘要: 智能设备的出现导致多媒体内容呈指数级增长。然而，深度学习的快速发展催生了能够操纵或创建多媒体假内容的复杂算法，即Deepfac。音频Deepfake会产生高度逼真的声音，从而为错误信息的传播提供便利，从而构成重大威胁。为了解决这个问题，已经组织了许多音频反欺骗检测挑战，以促进反欺骗对策的发展。本文对检测流水线中的每个组件进行了全面的回顾，包括算法体系结构、优化技术、应用程序通用性、评估指标、性能比较、可用的数据集和开放源码可用性。对于每个方面，我们都对最近的进展进行了系统的评估，同时讨论了现有的挑战。此外，我们还探讨了音频反欺骗的新兴研究课题，包括部分欺骗检测、跨数据集评估和对抗性攻击防御，并对未来的工作提出了一些有前景的研究方向。这份调查报告不仅确定了当前的最新水平，为未来的实验建立了坚实的基线，而且也为未来的研究人员提供了一条理解和增强音频反欺骗检测机制的明确途径。



## **24. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.14461v1) [paper-pdf](http://arxiv.org/pdf/2404.14461v1)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **25. Distributional Black-Box Model Inversion Attack with Multi-Agent Reinforcement Learning**

基于多智能体强化学习的分布式黑匣子模型翻转攻击 cs.LG

**SubmitDate**: 2024-04-22    [abs](http://arxiv.org/abs/2404.13860v1) [paper-pdf](http://arxiv.org/pdf/2404.13860v1)

**Authors**: Huan Bao, Kaimin Wei, Yongdong Wu, Jin Qian, Robert H. Deng

**Abstract**: A Model Inversion (MI) attack based on Generative Adversarial Networks (GAN) aims to recover the private training data from complex deep learning models by searching codes in the latent space. However, they merely search a deterministic latent space such that the found latent code is usually suboptimal. In addition, the existing distributional MI schemes assume that an attacker can access the structures and parameters of the target model, which is not always viable in practice. To overcome the above shortcomings, this paper proposes a novel Distributional Black-Box Model Inversion (DBB-MI) attack by constructing the probabilistic latent space for searching the target privacy data. Specifically, DBB-MI does not need the target model parameters or specialized GAN training. Instead, it finds the latent probability distribution by combining the output of the target model with multi-agent reinforcement learning techniques. Then, it randomly chooses latent codes from the latent probability distribution for recovering the private data. As the latent probability distribution closely aligns with the target privacy data in latent space, the recovered data will leak the privacy of training samples of the target model significantly. Abundant experiments conducted on diverse datasets and networks show that the present DBB-MI has better performance than state-of-the-art in attack accuracy, K-nearest neighbor feature distance, and Peak Signal-to-Noise Ratio.

摘要: 基于产生式对抗网络(GAN)的模型反转(MI)攻击旨在通过在潜在空间中搜索码来恢复复杂深度学习模型中的私有训练数据。然而，它们只是搜索确定性的潜在空间，因此发现的潜在代码通常是次优的。此外，现有的分布式MI方案假设攻击者可以访问目标模型的结构和参数，这在实践中并不总是可行的。为了克服上述不足，通过构造搜索目标隐私数据的概率潜在空间，提出了一种新的分布式黑盒模型反转(DBB-MI)攻击。具体地说，DBB-MI不需要目标模型参数或专门的GaN训练。相反，它通过将目标模型的输出与多智能体强化学习技术相结合来寻找潜在概率分布。然后，从潜在概率分布中随机选择潜在代码来恢复私有数据。由于潜在概率分布在潜在空间中与目标隐私数据密切相关，恢复后的数据会严重泄露目标模型训练样本的隐私。在不同的数据集和网络上进行的大量实验表明，本文的DBB-MI在攻击准确率、K近邻特征距离和峰值信噪比方面都优于现有的DBB-MI。



## **26. Concept Arithmetics for Circumventing Concept Inhibition in Diffusion Models**

避免扩散模型中概念抑制的概念算法 cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13706v1) [paper-pdf](http://arxiv.org/pdf/2404.13706v1)

**Authors**: Vitali Petsiuk, Kate Saenko

**Abstract**: Motivated by ethical and legal concerns, the scientific community is actively developing methods to limit the misuse of Text-to-Image diffusion models for reproducing copyrighted, violent, explicit, or personal information in the generated images. Simultaneously, researchers put these newly developed safety measures to the test by assuming the role of an adversary to find vulnerabilities and backdoors in them. We use compositional property of diffusion models, which allows to leverage multiple prompts in a single image generation. This property allows us to combine other concepts, that should not have been affected by the inhibition, to reconstruct the vector, responsible for target concept generation, even though the direct computation of this vector is no longer accessible. We provide theoretical and empirical evidence why the proposed attacks are possible and discuss the implications of these findings for safe model deployment. We argue that it is essential to consider all possible approaches to image generation with diffusion models that can be employed by an adversary. Our work opens up the discussion about the implications of concept arithmetics and compositional inference for safety mechanisms in diffusion models.   Content Advisory: This paper contains discussions and model-generated content that may be considered offensive. Reader discretion is advised.   Project page: https://cs-people.bu.edu/vpetsiuk/arc

摘要: 出于伦理和法律方面的考虑，科学界正在积极开发方法，以限制滥用文本到图像的传播模式，在生成的图像中复制受版权保护的、暴力的、露骨的或个人信息。与此同时，研究人员通过扮演对手的角色来测试这些新开发的安全措施，以发现其中的漏洞和后门。我们使用扩散模型的合成属性，允许在单个图像生成中利用多个提示。这一性质允许我们组合不应该受到抑制影响的其他概念，以重建负责目标概念生成的向量，即使不再能够直接计算该向量。我们提供了理论和经验证据，解释了为什么建议的攻击是可能的，并讨论了这些发现对安全模型部署的影响。我们认为，重要的是要考虑所有可能的方法，利用扩散模型生成图像，以供对手使用。我们的工作开启了关于扩散模型中的安全机制的概念、算法和组合推理的含义的讨论。内容建议：本文包含可能被视为冒犯性的讨论和模型生成的内容。建议读者酌情阅读。项目页面：https://cs-people.bu.edu/vpetsiuk/arc



## **27. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统性文献综述 cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2403.14280v3) [paper-pdf](http://arxiv.org/pdf/2403.14280v3)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **28. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13621v1) [paper-pdf](http://arxiv.org/pdf/2404.13621v1)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. Robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants show a higher vulnerability for the optical flow networks.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。



## **29. Robust EEG-based Emotion Recognition Using an Inception and Two-sided Perturbation Model**

使用初始和双边扰动模型的鲁棒性基于脑电波的情绪识别 eess.SP

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.15373v1) [paper-pdf](http://arxiv.org/pdf/2404.15373v1)

**Authors**: Shadi Sartipi, Mujdat Cetin

**Abstract**: Automated emotion recognition using electroencephalogram (EEG) signals has gained substantial attention. Although deep learning approaches exhibit strong performance, they often suffer from vulnerabilities to various perturbations, like environmental noise and adversarial attacks. In this paper, we propose an Inception feature generator and two-sided perturbation (INC-TSP) approach to enhance emotion recognition in brain-computer interfaces. INC-TSP integrates the Inception module for EEG data analysis and employs two-sided perturbation (TSP) as a defensive mechanism against input perturbations. TSP introduces worst-case perturbations to the model's weights and inputs, reinforcing the model's elasticity against adversarial attacks. The proposed approach addresses the challenge of maintaining accurate emotion recognition in the presence of input uncertainties. We validate INC-TSP in a subject-independent three-class emotion recognition scenario, demonstrating robust performance.

摘要: 使用脑电波（EEG）信号的自动情感识别已引起广泛关注。尽管深度学习方法表现出出色的性能，但它们往往容易受到各种干扰的影响，例如环境噪音和对抗性攻击。在本文中，我们提出了一种初始特征生成器和双边扰动（INC-TBC）方法来增强脑机接口中的情感识别。INC-TPS集成了Incement模块用于脑电数据分析，并采用双边扰动（TBC）作为针对输入扰动的防御机制。TPS向模型的权重和输入引入了最坏情况的扰动，增强了模型对抗对抗攻击的弹性。所提出的方法解决了在存在输入不确定性的情况下保持准确情感识别的挑战。我们在与对象无关的三级情感识别场景中验证了INC-TPS，展示了稳健的性能。



## **30. How to Evaluate Semantic Communications for Images with ViTScore Metric?**

如何使用ViTScore指标评估图像的语义通信？ cs.CV

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2309.04891v2) [paper-pdf](http://arxiv.org/pdf/2309.04891v2)

**Authors**: Tingting Zhu, Bo Peng, Jifan Liang, Tingchen Han, Hai Wan, Jingqiao Fu, Junjie Chen

**Abstract**: Semantic communications (SC) have been expected to be a new paradigm shifting to catalyze the next generation communication, whose main concerns shift from accurate bit transmission to effective semantic information exchange in communications. However, the previous and widely-used metrics for images are not applicable to evaluate the image semantic similarity in SC. Classical metrics to measure the similarity between two images usually rely on the pixel level or the structural level, such as the PSNR and the MS-SSIM. Straightforwardly using some tailored metrics based on deep-learning methods in CV community, such as the LPIPS, is infeasible for SC. To tackle this, inspired by BERTScore in NLP community, we propose a novel metric for evaluating image semantic similarity, named Vision Transformer Score (ViTScore). We prove theoretically that ViTScore has 3 important properties, including symmetry, boundedness, and normalization, which make ViTScore convenient and intuitive for image measurement. To evaluate the performance of ViTScore, we compare ViTScore with 3 typical metrics (PSNR, MS-SSIM, and LPIPS) through 4 classes of experiments: (i) correlation with BERTScore through evaluation of image caption downstream CV task, (ii) evaluation in classical image communications, (iii) evaluation in image semantic communication systems, and (iv) evaluation in image semantic communication systems with semantic attack. Experimental results demonstrate that ViTScore is robust and efficient in evaluating the semantic similarity of images. Particularly, ViTScore outperforms the other 3 typical metrics in evaluating the image semantic changes by semantic attack, such as image inverse with Generative Adversarial Networks (GANs). This indicates that ViTScore is an effective performance metric when deployed in SC scenarios.

摘要: 语义通信(SC)被认为是一种新的范式转换，以催化下一代通信，其主要关注点从准确的比特传输转向通信中有效的语义信息交换。然而，以往广泛使用的图像度量方法不适用于SC中的图像语义相似度评价。衡量两幅图像之间相似性的经典度量通常依赖于像素级或结构级，如PSNR和MS-SSIM。对于供应链来说，直接使用一些基于深度学习方法的量身定做的度量方法，如LPIPS，是不可行的。针对这一问题，受自然语言处理领域BERTScore的启发，我们提出了一种新的图像语义相似度评价指标--视觉变换得分(ViTScore)。从理论上证明了ViTScore具有对称性、有界性和归一化三个重要性质，这使得ViTScore能够方便直观地进行图像测量。为了评估ViTScore的性能，我们通过4类实验将ViTScore与3种典型的度量指标(PSNR、MS-SSIM和LPIPS)进行了比较：(I)通过评估图像字幕下行CV任务与BERTScore的相关性；(Ii)在经典图像通信中的评估；(Iii)在图像语义通信系统中的评估；(Iv)在语义攻击下的图像语义通信系统中的评估。实验结果表明，ViTScore在评价图像语义相似度方面具有较强的鲁棒性和较高的效率。特别是，ViTScore在通过语义攻击来评估图像语义变化方面优于其他3种典型的度量标准，例如基于生成性对抗网络的图像逆(GANS)。这表明当部署在SC场景中时，ViTScore是一个有效的性能指标。



## **31. Reliable Model Watermarking: Defending Against Theft without Compromising on Evasion**

可靠的模型水印：在不妥协的情况下抵御盗窃 cs.CR

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.13518v1) [paper-pdf](http://arxiv.org/pdf/2404.13518v1)

**Authors**: Hongyu Zhu, Sichu Liang, Wentao Hu, Fangqi Li, Ju Jia, Shilin Wang

**Abstract**: With the rise of Machine Learning as a Service (MLaaS) platforms,safeguarding the intellectual property of deep learning models is becoming paramount. Among various protective measures, trigger set watermarking has emerged as a flexible and effective strategy for preventing unauthorized model distribution. However, this paper identifies an inherent flaw in the current paradigm of trigger set watermarking: evasion adversaries can readily exploit the shortcuts created by models memorizing watermark samples that deviate from the main task distribution, significantly impairing their generalization in adversarial settings. To counteract this, we leverage diffusion models to synthesize unrestricted adversarial examples as trigger sets. By learning the model to accurately recognize them, unique watermark behaviors are promoted through knowledge injection rather than error memorization, thus avoiding exploitable shortcuts. Furthermore, we uncover that the resistance of current trigger set watermarking against removal attacks primarily relies on significantly damaging the decision boundaries during embedding, intertwining unremovability with adverse impacts. By optimizing the knowledge transfer properties of protected models, our approach conveys watermark behaviors to extraction surrogates without aggressively decision boundary perturbation. Experimental results on CIFAR-10/100 and Imagenette datasets demonstrate the effectiveness of our method, showing not only improved robustness against evasion adversaries but also superior resistance to watermark removal attacks compared to state-of-the-art solutions.

摘要: 随着机器学习即服务(MLaaS)平台的兴起，保护深度学习模型的知识产权变得至关重要。在各种保护措施中，触发集水印已经成为防止未经授权的模型传播的一种灵活而有效的策略。然而，本文指出了当前触发集水印的一个固有缺陷：逃避攻击者可以很容易地利用模型记忆偏离主任务分布的水印样本所创建的快捷方式，从而显著削弱其在对抗性环境中的泛化能力。为了抵消这一点，我们利用扩散模型来合成不受限制的对抗性例子作为触发集。通过学习模型来准确地识别它们，通过知识注入而不是错误记忆来促进独特的水印行为，从而避免了可利用的快捷方式。此外，我们发现当前触发器集水印抵抗移除攻击的能力主要依赖于在嵌入过程中显著破坏决策边界，将不可去除性与不利影响交织在一起。通过优化受保护模型的知识传递特性，该方法在不影响决策边界的情况下，将水印行为传递给提取代理。在CIFAR-10/100和Imagenette数据集上的实验结果证明了该方法的有效性，与最先进的解决方案相比，该方法不仅提高了对规避攻击的鲁棒性，而且对水印去除攻击具有更好的抵抗能力。



## **32. Machine Learning Robustness: A Primer**

机器学习鲁棒性：入门 cs.LG

**SubmitDate**: 2024-04-21    [abs](http://arxiv.org/abs/2404.00897v2) [paper-pdf](http://arxiv.org/pdf/2404.00897v2)

**Authors**: Houssem Ben Braiek, Foutse Khomh

**Abstract**: This chapter explores the foundational concept of robustness in Machine Learning (ML) and its integral role in establishing trustworthiness in Artificial Intelligence (AI) systems. The discussion begins with a detailed definition of robustness, portraying it as the ability of ML models to maintain stable performance across varied and unexpected environmental conditions. ML robustness is dissected through several lenses: its complementarity with generalizability; its status as a requirement for trustworthy AI; its adversarial vs non-adversarial aspects; its quantitative metrics; and its indicators such as reproducibility and explainability. The chapter delves into the factors that impede robustness, such as data bias, model complexity, and the pitfalls of underspecified ML pipelines. It surveys key techniques for robustness assessment from a broad perspective, including adversarial attacks, encompassing both digital and physical realms. It covers non-adversarial data shifts and nuances of Deep Learning (DL) software testing methodologies. The discussion progresses to explore amelioration strategies for bolstering robustness, starting with data-centric approaches like debiasing and augmentation. Further examination includes a variety of model-centric methods such as transfer learning, adversarial training, and randomized smoothing. Lastly, post-training methods are discussed, including ensemble techniques, pruning, and model repairs, emerging as cost-effective strategies to make models more resilient against the unpredictable. This chapter underscores the ongoing challenges and limitations in estimating and achieving ML robustness by existing approaches. It offers insights and directions for future research on this crucial concept, as a prerequisite for trustworthy AI systems.

摘要: 本章探讨了机器学习(ML)中稳健性的基本概念及其在人工智能(AI)系统中建立可信度的不可或缺的作用。讨论开始于对稳健性的详细定义，将其描述为ML模型在不同和意外的环境条件下保持稳定性能的能力。ML稳健性通过几个方面进行剖析：它与通用性的互补性；它作为值得信赖的人工智能的要求的地位；它的对抗性与非对抗性方面；它的量化指标；以及它的可再现性和可解释性等指标。本章深入探讨了阻碍健壮性的因素，如数据偏差、模型复杂性和未指定的ML管道的陷阱。它从广泛的角度考察了健壮性评估的关键技术，包括涵盖数字和物理领域的对抗性攻击。它涵盖了深度学习(DL)软件测试方法的非对抗性数据转移和细微差别。讨论继续探索增强健壮性的改进策略，从去偏向和增强等以数据为中心的方法开始。进一步的考试包括各种以模型为中心的方法，如转移学习、对抗性训练和随机平滑。最后，讨论了训练后的方法，包括集合技术、修剪和模型修复，这些方法成为使模型对不可预测的情况更具弹性的成本效益策略。本章强调了在通过现有方法估计和实现ML健壮性方面的持续挑战和限制。它为未来对这一关键概念的研究提供了见解和方向，这是值得信赖的人工智能系统的先决条件。



## **33. Deepfake Generation and Detection: A Benchmark and Survey**

Deepfake生成和检测：基准和调查 cs.CV

We closely follow the latest developments in  https://github.com/flyingby/Awesome-Deepfake-Generation-and-Detection

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2403.17881v3) [paper-pdf](http://arxiv.org/pdf/2403.17881v3)

**Authors**: Gan Pei, Jiangning Zhang, Menghan Hu, Zhenyu Zhang, Chengjie Wang, Yunsheng Wu, Guangtao Zhai, Jian Yang, Chunhua Shen, Dacheng Tao

**Abstract**: Deepfake is a technology dedicated to creating highly realistic facial images and videos under specific conditions, which has significant application potential in fields such as entertainment, movie production, digital human creation, to name a few. With the advancements in deep learning, techniques primarily represented by Variational Autoencoders and Generative Adversarial Networks have achieved impressive generation results. More recently, the emergence of diffusion models with powerful generation capabilities has sparked a renewed wave of research. In addition to deepfake generation, corresponding detection technologies continuously evolve to regulate the potential misuse of deepfakes, such as for privacy invasion and phishing attacks. This survey comprehensively reviews the latest developments in deepfake generation and detection, summarizing and analyzing current state-of-the-arts in this rapidly evolving field. We first unify task definitions, comprehensively introduce datasets and metrics, and discuss developing technologies. Then, we discuss the development of several related sub-fields and focus on researching four representative deepfake fields: face swapping, face reenactment, talking face generation, and facial attribute editing, as well as forgery detection. Subsequently, we comprehensively benchmark representative methods on popular datasets for each field, fully evaluating the latest and influential published works. Finally, we analyze challenges and future research directions of the discussed fields.

摘要: 深伪是一项致力于在特定条件下创建高真实感面部图像和视频的技术，在娱乐、电影制作、数字人类创作等领域具有巨大的应用潜力。随着深度学习的进步，以变式自动编码器和生成式对抗性网络为主要代表的技术已经取得了令人印象深刻的生成结果。最近，具有强大发电能力的扩散模型的出现引发了新一轮的研究浪潮。除了深度假冒的生成，相应的检测技术也在不断发展，以规范深度假冒的潜在滥用，例如用于侵犯隐私和网络钓鱼攻击。这项调查全面回顾了深度伪码生成和检测的最新进展，总结和分析了这一快速发展领域的最新技术。我们首先统一任务定义，全面介绍数据集和指标，并讨论开发技术。然后，讨论了几个相关的子领域的发展，重点研究了四个有代表性的深度伪领域：人脸交换、人脸重演、说话人脸生成、人脸属性编辑以及伪造检测。随后，我们在每个领域的热门数据集上综合基准有代表性的方法，充分评价最新和有影响力的已发表作品。最后，分析了所讨论领域面临的挑战和未来的研究方向。



## **34. Pixel is a Barrier: Diffusion Models Are More Adversarially Robust Than We Think**

像素是一个障碍：扩散模型比我们想象的更具对抗性 cs.CV

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13320v1) [paper-pdf](http://arxiv.org/pdf/2404.13320v1)

**Authors**: Haotian Xue, Yongxin Chen

**Abstract**: Adversarial examples for diffusion models are widely used as solutions for safety concerns. By adding adversarial perturbations to personal images, attackers can not edit or imitate them easily. However, it is essential to note that all these protections target the latent diffusion model (LDMs), the adversarial examples for diffusion models in the pixel space (PDMs) are largely overlooked. This may mislead us to think that the diffusion models are vulnerable to adversarial attacks like most deep models. In this paper, we show novel findings that: even though gradient-based white-box attacks can be used to attack the LDMs, they fail to attack PDMs. This finding is supported by extensive experiments of almost a wide range of attacking methods on various PDMs and LDMs with different model structures, which means diffusion models are indeed much more robust against adversarial attacks. We also find that PDMs can be used as an off-the-shelf purifier to effectively remove the adversarial patterns that were generated on LDMs to protect the images, which means that most protection methods nowadays, to some extent, cannot protect our images from malicious attacks. We hope that our insights will inspire the community to rethink the adversarial samples for diffusion models as protection methods and move forward to more effective protection. Codes are available in https://github.com/xavihart/PDM-Pure.

摘要: 扩散模型的对抗性例子被广泛用作安全问题的解决方案。通过向个人图像添加敌意干扰，攻击者无法轻松编辑或模仿它们。然而，值得注意的是，所有这些保护都是针对潜在扩散模型(LDMS)的，而像素空间扩散模型(PDMS)的对抗性例子在很大程度上被忽视了。这可能会误导我们认为扩散模型像大多数深度模型一样容易受到对手攻击。在本文中，我们发现了新的发现：即使基于梯度的白盒攻击可以用于攻击LDMS，它们也不能攻击PDMS。这一发现得到了对不同模型结构的PDMS和LDM上几乎各种攻击方法的广泛实验的支持，这意味着扩散模型确实对对手攻击具有更强的鲁棒性。我们还发现，PDMS可以作为现成的净化器来有效地去除LDM上产生的恶意模式来保护图像，这意味着目前的大多数保护方法在某种程度上不能保护我们的图像免受恶意攻击。我们希望我们的见解将激励社会重新考虑将扩散模型的对抗性样本作为保护方法，并朝着更有效的保护迈进。代码在https://github.com/xavihart/PDM-Pure.中可用



## **35. Backdoor Attacks and Defenses on Semantic-Symbol Reconstruction in Semantic Communications**

语义传播中语义符号重建的后门攻击与防御 cs.CR

This paper has been accepted by IEEE ICC 2024

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13279v1) [paper-pdf](http://arxiv.org/pdf/2404.13279v1)

**Authors**: Yuan Zhou, Rose Qingyang Hu, Yi Qian

**Abstract**: Semantic communication is of crucial importance for the next-generation wireless communication networks. The existing works have developed semantic communication frameworks based on deep learning. However, systems powered by deep learning are vulnerable to threats such as backdoor attacks and adversarial attacks. This paper delves into backdoor attacks targeting deep learning-enabled semantic communication systems. Since current works on backdoor attacks are not tailored for semantic communication scenarios, a new backdoor attack paradigm on semantic symbols (BASS) is introduced, based on which the corresponding defense measures are designed. Specifically, a training framework is proposed to prevent BASS. Additionally, reverse engineering-based and pruning-based defense strategies are designed to protect against backdoor attacks in semantic communication. Simulation results demonstrate the effectiveness of both the proposed attack paradigm and the defense strategies.

摘要: 语义通信对于下一代无线通信网络至关重要。现有作品开发了基于深度学习的语义通信框架。然而，由深度学习驱动的系统很容易受到后门攻击和对抗攻击等威胁。本文深入研究了针对支持深度学习的语义通信系统的后门攻击。由于当前的后门攻击研究并不适合语义通信场景，因此引入了一种新的语义符号后门攻击范式（BASS），并在此基础上设计了相应的防御措施。具体来说，提出了一个训练框架来防止BASS。此外，基于反向工程和基于修剪的防御策略旨在防止语义通信中的后门攻击。仿真结果证明了所提出的攻击范式和防御策略的有效性。



## **36. Beyond Score Changes: Adversarial Attack on No-Reference Image Quality Assessment from Two Perspectives**

超越分数变化：从两个角度对无参考图像质量评估的对抗攻击 eess.IV

Submitted to a conference

**SubmitDate**: 2024-04-20    [abs](http://arxiv.org/abs/2404.13277v1) [paper-pdf](http://arxiv.org/pdf/2404.13277v1)

**Authors**: Chenxi Yang, Yujia Liu, Dingquan Li, Yan Zhong, Tingting Jiang

**Abstract**: Deep neural networks have demonstrated impressive success in No-Reference Image Quality Assessment (NR-IQA). However, recent researches highlight the vulnerability of NR-IQA models to subtle adversarial perturbations, leading to inconsistencies between model predictions and subjective ratings. Current adversarial attacks, however, focus on perturbing predicted scores of individual images, neglecting the crucial aspect of inter-score correlation relationships within an entire image set. Meanwhile, it is important to note that the correlation, like ranking correlation, plays a significant role in NR-IQA tasks. To comprehensively explore the robustness of NR-IQA models, we introduce a new framework of correlation-error-based attacks that perturb both the correlation within an image set and score changes on individual images. Our research primarily focuses on ranking-related correlation metrics like Spearman's Rank-Order Correlation Coefficient (SROCC) and prediction error-related metrics like Mean Squared Error (MSE). As an instantiation, we propose a practical two-stage SROCC-MSE-Attack (SMA) that initially optimizes target attack scores for the entire image set and then generates adversarial examples guided by these scores. Experimental results demonstrate that our SMA method not only significantly disrupts the SROCC to negative values but also maintains a considerable change in the scores of individual images. Meanwhile, it exhibits state-of-the-art performance across metrics with different categories. Our method provides a new perspective on the robustness of NR-IQA models.

摘要: 深度神经网络在无参考图像质量评估(NR-IQA)中取得了令人印象深刻的成功。然而，最近的研究突显了NR-IQA模型在微妙的对抗性扰动下的脆弱性，导致模型预测与主观评分之间的不一致。然而，当前的对抗性攻击集中于扰乱单个图像的预测分数，而忽略了整个图像集内分数间相关性的关键方面。同时，值得注意的是，这种相关性和排名相关性一样，在NR-IQA任务中扮演着重要的角色。为了全面探索NR-IQA模型的稳健性，我们引入了一种新的基于相关性误差的攻击框架，该框架既干扰了图像集合内的相关性，也干扰了单个图像上的分数变化。我们的研究主要集中在与排名相关的指标，如Spearman的秩次相关系数(SROCC)和与预测误差相关的指标，如均方误差(MSE)。作为一个实例，我们提出了一种实用的两阶段SROCC-MSE攻击(SMA)，它首先优化整个图像集的目标攻击分数，然后根据这些分数生成对抗性实例。实验结果表明，我们的SMA方法不仅显著地将SROCC打乱到负值，而且还保持了单个图像得分的较大变化。同时，它在不同类别的指标上展示了最先进的性能。我们的方法为NR-IQA模型的稳健性提供了一个新的视角。



## **37. The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions**

教学层次结构：培训LLM以优先考虑授权的教学 cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.13208v1) [paper-pdf](http://arxiv.org/pdf/2404.13208v1)

**Authors**: Eric Wallace, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, Alex Beutel

**Abstract**: Today's LLMs are susceptible to prompt injections, jailbreaks, and other attacks that allow adversaries to overwrite a model's original instructions with their own malicious prompts. In this work, we argue that one of the primary vulnerabilities underlying these attacks is that LLMs often consider system prompts (e.g., text from an application developer) to be the same priority as text from untrusted users and third parties. To address this, we propose an instruction hierarchy that explicitly defines how models should behave when instructions of different priorities conflict. We then propose a data generation method to demonstrate this hierarchical instruction following behavior, which teaches LLMs to selectively ignore lower-privileged instructions. We apply this method to GPT-3.5, showing that it drastically increases robustness -- even for attack types not seen during training -- while imposing minimal degradations on standard capabilities.

摘要: 当今的LLM很容易受到提示注入、越狱和其他攻击，这些攻击允许对手用自己的恶意提示覆盖模型的原始指令。在这项工作中，我们认为这些攻击背后的主要漏洞之一是LLM经常考虑系统提示（例如，来自应用程序开发人员的文本）与来自不受信任用户和第三方的文本具有相同的优先级。为了解决这个问题，我们提出了一个指令层次结构，它明确定义了当不同优先级的指令发生冲突时模型应该如何表现。然后，我们提出了一种数据生成方法来演示这种分层指令遵循行为，该方法教导LLM选择性地忽略较低特权的指令。我们将这种方法应用于GPT-3.5，表明它极大地提高了稳健性（即使对于训练期间未看到的攻击类型），同时对标准能力的降级降到最低。



## **38. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

使用局部自适应对抗色彩攻击保护艺术品免受神经风格转移 cs.CV

9 pages, 5 figures, 4 tables

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.09673v2) [paper-pdf](http://arxiv.org/pdf/2401.09673v2)

**Authors**: Zhongliang Guo, Junhao Dong, Yifei Qian, Kaixuan Wang, Weiye Li, Ziheng Guo, Yuheng Wang, Yanli Li, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) generates new images by combining the style of one image with the content of another. However, unauthorized NST can exploit artwork, raising concerns about artists' rights and motivating the development of proactive protection methods. We propose Locally Adaptive Adversarial Color Attack (LAACA), empowering artists to protect their artwork from unauthorized style transfer by processing before public release. By delving into the intricacies of human visual perception and the role of different frequency components, our method strategically introduces frequency-adaptive perturbations in the image. These perturbations significantly degrade the generation quality of NST while maintaining an acceptable level of visual change in the original image, ensuring that potential infringers are discouraged from using the protected artworks, because of its bad NST generation quality. Additionally, existing metrics often overlook the importance of color fidelity in evaluating color-mattered tasks, such as the quality of NST-generated images, which is crucial in the context of artistic works. To comprehensively assess the color-mattered tasks, we propose the Adversarial Color Distance Metric (ACDM), designed to quantify the color difference of images pre- and post-manipulations. Experimental results confirm that attacking NST using LAACA results in visually inferior style transfer, and the ACDM can efficiently measure color-mattered tasks. By providing artists with a tool to safeguard their intellectual property, our work relieves the socio-technical challenges posed by the misuse of NST in the art community.

摘要: 神经样式转换(NST)通过将一幅图像的样式与另一幅图像的内容相结合来生成新图像。然而，未经授权的NST可以利用艺术品，这引发了人们对艺术家权利的担忧，并推动了主动保护方法的发展。我们提出了局部自适应对抗性色彩攻击(LAACA)，授权艺术家通过在公开发布之前进行处理来保护他们的作品免受未经授权的样式转移。通过深入研究人类视觉感知的复杂性和不同频率分量的作用，我们的方法战略性地在图像中引入了频率自适应扰动。这些干扰显著降低了NST的生成质量，同时保持了原始图像中可接受的视觉变化水平，确保了潜在侵权者因其糟糕的NST生成质量而不愿使用受保护的艺术品。此外，现有的衡量标准往往忽略了颜色保真度在评估与颜色有关的任务时的重要性，例如NST生成的图像的质量，这在艺术作品的背景下是至关重要的。为了全面评估颜色相关任务，我们提出了对抗性颜色距离度量(ACDM)，旨在量化图像处理前后的颜色差异。实验结果证实，使用LAACA攻击NST会导致视觉劣势风格迁移，ACDM可以有效地测量颜色相关任务。通过为艺术家提供保护其知识产权的工具，我们的工作缓解了艺术界滥用NST带来的社会技术挑战。



## **39. Set-Based Training for Neural Network Verification**

神经网络验证的基于集的训练 cs.LG

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.14961v2) [paper-pdf](http://arxiv.org/pdf/2401.14961v2)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can significantly affect the outputs of a neural network. In safety-critical environments, the inputs often contain noisy sensor data; hence, in this case, neural networks that are robust against input perturbations are required. To ensure safety, the robustness of a neural network must be formally verified. However, training and formally verifying robust neural networks is challenging. We address both of these challenges by employing, for the first time, an end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure trains neural networks, which can be easily verified using simple polynomial-time verification algorithms. Moreover, our extensive evaluation demonstrates that our set-based training procedure effectively trains robust neural networks, which are easier to verify. Set-based trained neural networks consistently match or outperform those trained with state-of-the-art robust training approaches.

摘要: 神经网络容易受到敌意攻击，即微小的输入扰动会显著影响神经网络的输出。在安全关键环境中，输入通常包含有噪声的传感器数据；因此，在这种情况下，需要对输入扰动具有鲁棒性的神经网络。为了确保安全，神经网络的健壮性必须经过正式验证。然而，训练和正式验证稳健的神经网络是具有挑战性的。我们通过首次采用端到端基于集合的训练程序来解决这两个挑战，该程序训练用于正式验证的稳健神经网络。我们的训练程序训练神经网络，可以很容易地使用简单的多项式时间验证算法进行验证。此外，我们的广泛评估表明，我们的基于集合的训练过程有效地训练了健壮的神经网络，更容易验证。基于集合的训练神经网络始终与使用最先进的稳健训练方法训练的神经网络相匹配或优于那些训练方法。



## **40. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

量子机器学习安全性的主要方面：文献评论 quant-ph

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2401.07774v2) [paper-pdf](http://arxiv.org/pdf/2401.07774v2)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. We point out specific risks, such as cross-talk in superconducting systems and forced repeated shuttle operations in ion-trap systems, which threaten QML's reliability. However, approaches like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.

摘要: 量子机器学习(QML)已经成为量子计算和经典机器学习的一个有前途的交叉点，有望推动计算任务的突破。本文通过系统的文献综述，探讨了QML的安全关注点和优势所在。我们对QML模型的安全性、量子体系结构固有的脆弱性以及提出的缓解策略进行了分类和回顾。调查显示，虽然QML具有独特的优势，但它也引入了经典系统中未曾见过的新攻击载体。我们指出了具体的风险，如超导系统中的串扰和离子陷阱系统中被迫重复的穿梭操作，这些风险威胁到了QML的可靠性。然而，对抗性训练、量子噪声利用和量子差分隐私等方法已经显示出在增强QML稳健性方面的潜力。我们的综述讨论了持续和严格研究的必要性，以确保QML在现实世界应用程序中的安全部署。这项工作为旨在导航QML安全方面的研究人员和实践者提供了基础性参考。



## **41. A Proactive Decoy Selection Scheme for Cyber Deception using MITRE ATT&CK**

使用MITRE ATT & CK的网络欺骗主动诱饵选择方案 cs.CR

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12783v1) [paper-pdf](http://arxiv.org/pdf/2404.12783v1)

**Authors**: Marco Zambianco, Claudio Facchinetti, Domenico Siracusa

**Abstract**: Cyber deception allows compensating the late response of defenders countermeasures to the ever evolving tactics, techniques, and procedures (TTPs) of attackers. This proactive defense strategy employs decoys resembling legitimate system components to lure stealthy attackers within the defender environment, slowing and/or denying the accomplishment of their goals. In this regard, the selection of decoys that can expose the techniques used by malicious users plays a central role to incentivize their engagement. However, this is a difficult task to achieve in practice, since it requires an accurate and realistic modeling of the attacker capabilities and his possible targets. In this work, we tackle this challenge and we design a decoy selection scheme that is supported by an adversarial modeling based on empirical observation of real-world attackers. We take advantage of a domain-specific threat modelling language using MITRE ATT&CK framework as source of attacker TTPs targeting enterprise systems. In detail, we extract the information about the execution preconditions of each technique as well as its possible effects on the environment to generate attack graphs modeling the adversary capabilities. Based on this, we formulate a graph partition problem that minimizes the number of decoys detecting a corresponding number of techniques employed in various attack paths directed to specific targets. We compare our optimization-based decoy selection approach against several benchmark schemes that ignore the preconditions between the various attack steps. Results reveal that the proposed scheme provides the highest interception rate of attack paths using the lowest amount of decoys.

摘要: 网络欺骗可以补偿防御者对攻击者不断演变的战术、技术和程序(TTP)的反应迟缓。这种主动防御策略使用类似于合法系统组件的诱饵，在防御者环境中引诱隐形攻击者，减缓和/或拒绝他们目标的实现。在这方面，选择能够揭露恶意用户使用的技术的诱饵对激励他们的参与起着核心作用。然而，这在实践中是一项困难的任务，因为它需要对攻击者的能力及其可能的目标进行准确和现实的建模。在这项工作中，我们解决了这一挑战，并设计了一个诱饵选择方案，该方案由基于对真实世界攻击者的经验观察的对抗性建模来支持。我们利用一种特定于领域的威胁建模语言，使用MITRE ATT&CK框架作为针对企业系统的攻击者TTP的来源。详细地，我们提取了关于每种技术的执行前提及其对环境的可能影响的信息，以生成模拟对手能力的攻击图。在此基础上，我们提出了一个图划分问题，该问题最小化了在针对特定目标的各种攻击路径中检测到相应数量的技术的诱饵数量。我们将我们基于优化的诱饵选择方法与几个忽略攻击步骤之间的前提条件的基准方案进行了比较。结果表明，该方案以最少的诱饵获得了最高的攻击路径拦截率。



## **42. Defending against Data Poisoning Attacks in Federated Learning via User Elimination**

通过用户消除防御联邦学习中的数据中毒攻击 cs.CR

To be submitted in AISEC 2024

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12778v1) [paper-pdf](http://arxiv.org/pdf/2404.12778v1)

**Authors**: Nick Galanis

**Abstract**: In the evolving landscape of Federated Learning (FL), a new type of attacks concerns the research community, namely Data Poisoning Attacks, which threaten the model integrity by maliciously altering training data. This paper introduces a novel defensive framework focused on the strategic elimination of adversarial users within a federated model. We detect those anomalies in the aggregation phase of the Federated Algorithm, by integrating metadata gathered by the local training instances with Differential Privacy techniques, to ensure that no data leakage is possible. To our knowledge, this is the first proposal in the field of FL that leverages metadata other than the model's gradients in order to ensure honesty in the reported local models. Our extensive experiments demonstrate the efficacy of our methods, significantly mitigating the risk of data poisoning while maintaining user privacy and model performance. Our findings suggest that this new user elimination approach serves us with a great balance between privacy and utility, thus contributing to the arsenal of arguments in favor of the safe adoption of FL in safe domains, both in academic setting and in the industry.

摘要: 在联邦学习(FL)的发展中，一种新型的攻击引起了研究界的关注，即数据中毒攻击，它通过恶意篡改训练数据来威胁模型的完整性。本文介绍了一种新的防御框架，该框架侧重于在联邦模型中对敌意用户进行战略消除。在联合算法的聚合阶段，我们通过将本地训练实例收集的元数据与差异隐私技术相结合来检测这些异常，以确保数据不会泄露。据我们所知，这是FL领域中第一个利用元数据而不是模型的梯度来确保所报告的本地模型的真实性的建议。我们的广泛实验证明了我们方法的有效性，显著降低了数据中毒的风险，同时保持了用户隐私和模型性能。我们的发现表明，这种新的用户淘汰方法在隐私和效用之间取得了很好的平衡，因此有助于在学术环境和行业中支持在安全领域安全采用FL的论点。



## **43. A Clean-graph Backdoor Attack against Graph Convolutional Networks with Poisoned Label Only**

针对仅带有毒标签的图卷积网络的干净图后门攻击 cs.AI

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12704v1) [paper-pdf](http://arxiv.org/pdf/2404.12704v1)

**Authors**: Jiazhu Dai, Haoyu Sun

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in dealing with various graph structures such as node classification, graph classification and other tasks. However,recent studies have shown that GCNs are vulnerable to a novel threat known as backdoor attacks. However, all existing backdoor attacks in the graph domain require modifying the training samples to accomplish the backdoor injection, which may not be practical in many realistic scenarios where adversaries have no access to modify the training samples and may leads to the backdoor attack being detected easily. In order to explore the backdoor vulnerability of GCNs and create a more practical and stealthy backdoor attack method, this paper proposes a clean-graph backdoor attack against GCNs (CBAG) in the node classification task,which only poisons the training labels without any modification to the training samples, revealing that GCNs have this security vulnerability. Specifically, CBAG designs a new trigger exploration method to find important feature dimensions as the trigger patterns to improve the attack performance. By poisoning the training labels, a hidden backdoor is injected into the GCNs model. Experimental results show that our clean graph backdoor can achieve 99% attack success rate while maintaining the functionality of the GCNs model on benign samples.

摘要: 图卷积网络(GCNS)在处理各种图结构，如节点分类、图分类等方面表现出了优异的性能。然而，最近的研究表明，GCNS容易受到一种名为后门攻击的新威胁的攻击。然而，图域中现有的所有后门攻击都需要修改训练样本来完成后门注入，这在许多现实场景中可能并不实用，因为攻击者无法修改训练样本，从而可能导致后门攻击很容易被检测到。为了探索GCNS的后门漏洞，创造一种更实用、更隐蔽的后门攻击方法，提出了一种在节点分类任务中针对GCNS的清洁图后门攻击(CBAG)，该攻击只毒化训练标签而不对训练样本进行任何修改，从而揭示了GCNS存在该安全漏洞。具体地说，CBAG设计了一种新的触发探测方法，找到重要的特征维度作为触发模式，以提高攻击性能。通过毒化训练标签，GCNS模型被注入了一个隐藏的后门。实验结果表明，我们的清洁图后门在保持GCNS模型在良性样本上的功能的同时，可以达到99%的攻击成功率。



## **44. MLSD-GAN -- Generating Strong High Quality Face Morphing Attacks using Latent Semantic Disentanglement**

MLSD-GAN --使用潜在语义解纠缠生成强大的高质量人脸变形攻击 cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12679v1) [paper-pdf](http://arxiv.org/pdf/2404.12679v1)

**Authors**: Aravinda Reddy PN, Raghavendra Ramachandra, Krothapalli Sreenivasa Rao, Pabitra Mitra

**Abstract**: Face-morphing attacks are a growing concern for biometric researchers, as they can be used to fool face recognition systems (FRS). These attacks can be generated at the image level (supervised) or representation level (unsupervised). Previous unsupervised morphing attacks have relied on generative adversarial networks (GANs). More recently, researchers have used linear interpolation of StyleGAN-encoded images to generate morphing attacks. In this paper, we propose a new method for generating high-quality morphing attacks using StyleGAN disentanglement. Our approach, called MLSD-GAN, spherically interpolates the disentangled latents to produce realistic and diverse morphing attacks. We evaluate the vulnerability of MLSD-GAN on two deep-learning-based FRS techniques. The results show that MLSD-GAN poses a significant threat to FRS, as it can generate morphing attacks that are highly effective at fooling these systems.

摘要: 面部变形攻击越来越受到生物识别研究人员的关注，因为它们可以用来欺骗面部识别系统（FSG）。这些攻击可以在图像级别（受监督）或表示级别（无监督）生成。之前的无监督变形攻击依赖于生成对抗网络（GAN）。最近，研究人员使用StyleGAN编码图像的线性插值来生成变形攻击。在本文中，我们提出了一种使用StyleGAN去纠缠生成高质量变形攻击的新方法。我们的方法称为MLSD-GAN，对解开的潜伏进行球形内插，以产生真实且多样化的变形攻击。我们在两种基于深度学习的FSG技术上评估了MLSD-GAN的脆弱性。结果表明，MLSD-GAN对FSG构成了重大威胁，因为它可以生成能够非常有效地欺骗这些系统的变形攻击。



## **45. How Real Is Real? A Human Evaluation Framework for Unrestricted Adversarial Examples**

真实有多真实？无限制对抗性例子的人类评估框架 cs.AI

3 pages, 3 figures, AAAI 2024 Spring Symposium on User-Aligned  Assessment of Adaptive AI Systems

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12653v1) [paper-pdf](http://arxiv.org/pdf/2404.12653v1)

**Authors**: Dren Fazlija, Arkadij Orlov, Johanna Schrader, Monty-Maximilian Zühlke, Michael Rohs, Daniel Kudenko

**Abstract**: With an ever-increasing reliance on machine learning (ML) models in the real world, adversarial examples threaten the safety of AI-based systems such as autonomous vehicles. In the image domain, they represent maliciously perturbed data points that look benign to humans (i.e., the image modification is not noticeable) but greatly mislead state-of-the-art ML models. Previously, researchers ensured the imperceptibility of their altered data points by restricting perturbations via $\ell_p$ norms. However, recent publications claim that creating natural-looking adversarial examples without such restrictions is also possible. With much more freedom to instill malicious information into data, these unrestricted adversarial examples can potentially overcome traditional defense strategies as they are not constrained by the limitations or patterns these defenses typically recognize and mitigate. This allows attackers to operate outside of expected threat models. However, surveying existing image-based methods, we noticed a need for more human evaluations of the proposed image modifications. Based on existing human-assessment frameworks for image generation quality, we propose SCOOTER - an evaluation framework for unrestricted image-based attacks. It provides researchers with guidelines for conducting statistically significant human experiments, standardized questions, and a ready-to-use implementation. We propose a framework that allows researchers to analyze how imperceptible their unrestricted attacks truly are.

摘要: 随着现实世界中对机器学习(ML)模型的日益依赖，敌对的例子威胁到了自动驾驶汽车等基于人工智能的系统的安全性。在图像域中，它们表示恶意扰动的数据点，这些数据点看起来对人类无害(即图像修改不明显)，但却极大地误导了最先进的ML模型。在此之前，研究人员通过使用$\ell_p$规范来限制扰动，以确保更改后的数据点的不可见性。然而，最近的出版物声称，创造看起来自然的对抗性例子也是可能的，而不受这样的限制。随着将恶意信息注入数据的自由度大大增加，这些不受限制的敌意例子可能会克服传统的防御策略，因为它们不受这些防御通常识别和缓解的限制或模式的限制。这允许攻击者在预期的威胁模型之外操作。然而，审视现有的基于图像的方法，我们注意到需要对拟议的图像修改进行更多的人工评估。在已有的图像生成质量人工评估框架的基础上，提出了一种针对无限制图像攻击的评估框架--SCOOTER。它为研究人员提供了进行具有统计意义的人体实验、标准化问题和现成实施的指导方针。我们提出了一个框架，允许研究人员分析他们的不受限制的攻击到底有多难以察觉。



## **46. AED-PADA:Improving Generalizability of Adversarial Example Detection via Principal Adversarial Domain Adaptation**

AED-PADA：通过主要对抗领域适应提高对抗示例检测的通用性 cs.CV

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12635v1) [paper-pdf](http://arxiv.org/pdf/2404.12635v1)

**Authors**: Heqi Peng, Yunhong Wang, Ruijie Yang, Beichen Li, Rui Wang, Yuanfang Guo

**Abstract**: Adversarial example detection, which can be conveniently applied in many scenarios, is important in the area of adversarial defense. Unfortunately, existing detection methods suffer from poor generalization performance, because their training process usually relies on the examples generated from a single known adversarial attack and there exists a large discrepancy between the training and unseen testing adversarial examples. To address this issue, we propose a novel method, named Adversarial Example Detection via Principal Adversarial Domain Adaptation (AED-PADA). Specifically, our approach identifies the Principal Adversarial Domains (PADs), i.e., a combination of features of the adversarial examples from different attacks, which possesses large coverage of the entire adversarial feature space. Then, we pioneer to exploit multi-source domain adaptation in adversarial example detection with PADs as source domains. Experiments demonstrate the superior generalization ability of our proposed AED-PADA. Note that this superiority is particularly achieved in challenging scenarios characterized by employing the minimal magnitude constraint for the perturbations.

摘要: 对抗性实例检测在对抗性防御领域具有重要意义，可以方便地应用于多种场景。遗憾的是，现有的检测方法泛化性能较差，因为它们的训练过程通常依赖于单一已知对手攻击产生的样本，并且训练样本和未见的测试对手样本之间存在着很大的差异。为了解决这一问题，我们提出了一种新的方法，称为基于主对抗性领域适应的对抗性范例检测(AED-PADA)。具体地说，我们的方法识别主要的对抗性领域(PADS)，即来自不同攻击的对抗性实例的特征的组合，它具有很大的覆盖整个对抗性特征空间。在此基础上，首次将多源域自适应技术应用于以PADS为源域的对抗性实例检测中。实验结果表明，AED-PADA算法具有较好的泛化能力。请注意，这种优势尤其在具有挑战性的场景中实现，该场景的特征是对扰动采用最小幅度约束。



## **47. Watermark-embedded Adversarial Examples for Copyright Protection against Diffusion Models**

针对扩散模型的版权保护嵌入水印的对抗示例 cs.CV

updated references

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.09401v2) [paper-pdf](http://arxiv.org/pdf/2404.09401v2)

**Authors**: Peifei Zhu, Tsubasa Takahashi, Hirokatsu Kataoka

**Abstract**: Diffusion Models (DMs) have shown remarkable capabilities in various image-generation tasks. However, there are growing concerns that DMs could be used to imitate unauthorized creations and thus raise copyright issues. To address this issue, we propose a novel framework that embeds personal watermarks in the generation of adversarial examples. Such examples can force DMs to generate images with visible watermarks and prevent DMs from imitating unauthorized images. We construct a generator based on conditional adversarial networks and design three losses (adversarial loss, GAN loss, and perturbation loss) to generate adversarial examples that have subtle perturbation but can effectively attack DMs to prevent copyright violations. Training a generator for a personal watermark by our method only requires 5-10 samples within 2-3 minutes, and once the generator is trained, it can generate adversarial examples with that watermark significantly fast (0.2s per image). We conduct extensive experiments in various conditional image-generation scenarios. Compared to existing methods that generate images with chaotic textures, our method adds visible watermarks on the generated images, which is a more straightforward way to indicate copyright violations. We also observe that our adversarial examples exhibit good transferability across unknown generative models. Therefore, this work provides a simple yet powerful way to protect copyright from DM-based imitation.

摘要: 扩散模型(DM)在各种图像生成任务中表现出了卓越的能力。然而，人们越来越担心，DM可能被用来模仿未经授权的创作，从而引发版权问题。为了解决这个问题，我们提出了一种新的框架，在对抗性例子的生成中嵌入个人水印。这样的例子可以迫使DM生成带有可见水印的图像，并防止DM模仿未经授权的图像。我们构造了一个基于条件对抗网络的生成器，并设计了三种损失(对抗损失、GAN损失和扰动损失)来生成具有微妙扰动但可以有效攻击DM以防止版权侵犯的对抗实例。在2-3分钟内，我们的方法只需要5-10个样本就可以训练生成个人水印的生成器，并且一旦训练生成器，它可以显著地快速地生成带有该水印的对抗性示例(每幅图像0.2s)。我们在各种有条件的图像生成场景中进行了广泛的实验。与现有的带有混沌纹理的图像生成方法相比，我们的方法在生成的图像上添加了可见的水印，这是一种更直接的方式来指示侵犯版权的行为。我们还观察到，我们的对抗性例子显示出良好的跨未知生成模型的可转移性。因此，这部作品提供了一种简单而强大的方式来保护版权免受基于DM的模仿。



## **48. SA-Attack: Speed-adaptive stealthy adversarial attack on trajectory prediction**

SA攻击：对轨迹预测的速度自适应隐形对抗攻击 cs.LG

This work is published in IEEE IV Symposium

**SubmitDate**: 2024-04-19    [abs](http://arxiv.org/abs/2404.12612v1) [paper-pdf](http://arxiv.org/pdf/2404.12612v1)

**Authors**: Huilin Yin, Jiaxiang Li, Pengju Zhen, Jun Yan

**Abstract**: Trajectory prediction is critical for the safe planning and navigation of automated vehicles. The trajectory prediction models based on the neural networks are vulnerable to adversarial attacks. Previous attack methods have achieved high attack success rates but overlook the adaptability to realistic scenarios and the concealment of the deceits. To address this problem, we propose a speed-adaptive stealthy adversarial attack method named SA-Attack. This method searches the sensitive region of trajectory prediction models and generates the adversarial trajectories by using the vehicle-following method and incorporating information about forthcoming trajectories. Our method has the ability to adapt to different speed scenarios by reconstructing the trajectory from scratch. Fusing future trajectory trends and curvature constraints can guarantee the smoothness of adversarial trajectories, further ensuring the stealthiness of attacks. The empirical study on the datasets of nuScenes and Apolloscape demonstrates the attack performance of our proposed method. Finally, we also demonstrate the adaptability and stealthiness of SA-Attack for different speed scenarios. Our code is available at the repository: https://github.com/eclipse-bot/SA-Attack.

摘要: 轨迹预测对于自动车辆的安全规划和导航是至关重要的。基于神经网络的弹道预测模型容易受到敌方攻击。以往的攻击方法取得了较高的攻击成功率，但忽略了对现实场景的适应性和欺骗的隐蔽性。针对这一问题，我们提出了一种速度自适应的隐身对抗攻击方法SA-Attack。该方法搜索轨迹预测模型的敏感区域，利用车辆跟踪法并结合即将到来的轨迹信息生成对抗性轨迹。我们的方法通过从头开始重建轨迹，能够适应不同的速度场景。融合未来轨迹趋势和曲率约束，可以保证对抗性轨迹的光滑性，进一步保证攻击的隐蔽性。在nuScenes和Apollosscape数据集上的实证研究证明了该方法的攻击性能。最后，我们还展示了SA-攻击对不同速度场景的适应性和隐蔽性。我们的代码可以在存储库中找到：https://github.com/eclipse-bot/SA-Attack.



## **49. Proteus: Preserving Model Confidentiality during Graph Optimizations**

Proteus：在图优化期间保持模型机密性 cs.CR

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12512v1) [paper-pdf](http://arxiv.org/pdf/2404.12512v1)

**Authors**: Yubo Gao, Maryam Haghifam, Christina Giannoula, Renbo Tu, Gennady Pekhimenko, Nandita Vijaykumar

**Abstract**: Deep learning (DL) models have revolutionized numerous domains, yet optimizing them for computational efficiency remains a challenging endeavor. Development of new DL models typically involves two parties: the model developers and performance optimizers. The collaboration between the parties often necessitates the model developers exposing the model architecture and computational graph to the optimizers. However, this exposure is undesirable since the model architecture is an important intellectual property, and its innovations require significant investments and expertise. During the exchange, the model is also vulnerable to adversarial attacks via model stealing.   This paper presents Proteus, a novel mechanism that enables model optimization by an independent party while preserving the confidentiality of the model architecture. Proteus obfuscates the protected model by partitioning its computational graph into subgraphs and concealing each subgraph within a large pool of generated realistic subgraphs that cannot be easily distinguished from the original. We evaluate Proteus on a range of DNNs, demonstrating its efficacy in preserving confidentiality without compromising performance optimization opportunities. Proteus effectively hides the model as one alternative among up to $10^{32}$ possible model architectures, and is resilient against attacks with a learning-based adversary. We also demonstrate that heuristic based and manual approaches are ineffective in identifying the protected model. To our knowledge, Proteus is the first work that tackles the challenge of model confidentiality during performance optimization. Proteus will be open-sourced for direct use and experimentation, with easy integration with compilers such as ONNXRuntime.

摘要: 深度学习模型已经给许多领域带来了革命性的变化，但为了计算效率而优化它们仍然是一项具有挑战性的努力。新的动态链接库模型的开发通常涉及两方：模型开发者和性能优化器。各方之间的协作通常需要模型开发人员向优化器公开模型体系结构和计算图。然而，这种暴露是不可取的，因为模型体系结构是一项重要的知识产权，其创新需要大量投资和专业知识。在交换过程中，模型也容易受到模型窃取的对手攻击。提出了一种新的模型优化机制Proteus，它能够在保证模型体系结构机密性的同时，由独立的一方对模型进行优化。Proteus通过将其计算图分割成子图并将每个子图隐藏在生成的不容易与原始区分的真实子图的大池中来混淆受保护的模型。我们在一系列DNN上对Proteus进行了评估，展示了其在不影响性能优化机会的情况下保护机密性的有效性。Proteus有效地隐藏了该模型，将其作为高达10^{32}$可能的模型体系结构中的一种替代方案，并通过基于学习的对手对攻击具有弹性。我们还证明了基于启发式方法和人工方法在识别受保护模型方面是无效的。据我们所知，Proteus是第一个在性能优化过程中解决模型保密性挑战的工作。Proteus将是开源的，可以直接使用和试验，可以很容易地与ONNXRuntime等编译器集成。



## **50. KDk: A Defense Mechanism Against Label Inference Attacks in Vertical Federated Learning**

KDk：垂直联邦学习中针对标签推理攻击的防御机制 cs.LG

**SubmitDate**: 2024-04-18    [abs](http://arxiv.org/abs/2404.12369v1) [paper-pdf](http://arxiv.org/pdf/2404.12369v1)

**Authors**: Marco Arazzi, Serena Nicolazzo, Antonino Nocera

**Abstract**: Vertical Federated Learning (VFL) is a category of Federated Learning in which models are trained collaboratively among parties with vertically partitioned data. Typically, in a VFL scenario, the labels of the samples are kept private from all the parties except for the aggregating server, that is the label owner. Nevertheless, recent works discovered that by exploiting gradient information returned by the server to bottom models, with the knowledge of only a small set of auxiliary labels on a very limited subset of training data points, an adversary can infer the private labels. These attacks are known as label inference attacks in VFL. In our work, we propose a novel framework called KDk, that combines Knowledge Distillation and k-anonymity to provide a defense mechanism against potential label inference attacks in a VFL scenario. Through an exhaustive experimental campaign we demonstrate that by applying our approach, the performance of the analyzed label inference attacks decreases consistently, even by more than 60%, maintaining the accuracy of the whole VFL almost unaltered.

摘要: 垂直联合学习(VFL)是联合学习的一个类别，其中模型在具有垂直分割数据的各方之间协作训练。通常，在VFL场景中，除了聚合服务器(即标签所有者)之外，样本的标签对所有各方都是私有的。然而，最近的工作发现，通过利用服务器返回到底层模型的梯度信息，只要知道非常有限的训练数据点子集上的一小部分辅助标签，对手就可以推断出私有标签。这些攻击在VFL中被称为标签推理攻击。在我们的工作中，我们提出了一种新的框架KDK，它结合了知识蒸馏和k-匿名性来提供一种防御VFL场景中潜在的标签推理攻击的机制。通过详尽的实验证明，应用我们的方法，分析的标签推理攻击的性能持续下降，甚至下降了60%以上，几乎保持了整个VFL的准确率不变。



