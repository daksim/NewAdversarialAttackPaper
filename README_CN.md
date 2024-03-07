# Latest Adversarial Attack Papers
**update at 2024-03-07 10:25:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Adversarial Attacks on Latent Diffusion Model**

基于潜在扩散模型的对抗性攻击改进 cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.

摘要: 针对当前最先进的图像生成模型--潜在扩散模型(LDM)的敌意攻击已被用作对未经授权的图像进行恶意微调的有效保护。在这些对抗性例子上精调的LDM学习通过偏差来降低误差，由此对模型进行攻击并预测带有偏差的得分函数。ACE统一了添加到预测得分函数的额外误差的模式。然后，我们引入一个精心设计的模式来改进攻击。



## **2. A Survey on Adversarial Contention Resolution**

对抗性争议解决机制研究综述 cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 争用解决方案解决了协调多个进程对共享资源（如内存、磁盘存储或通信信道）的访问的挑战。竞争解决最初是由数据库系统和总线网络的挑战所激发的，尽管经历了几十年的技术变革，但它仍然是资源共享的重要抽象。在这里，我们调查的文献解决最坏情况下的竞争，其中进程的数量和每个进程可能开始寻求访问资源的时间是由对手。我们强调了争用解决方案的演变，其中新的问题-如安全性，服务质量和能源效率-是由现代系统的动机。这些努力已经深入了解了随机和确定性方法的局限性，以及不同模型假设的影响，如全局时钟同步，处理器数量的知识，访问尝试的反馈，以及对共享资源可用性的攻击。



## **3. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

NeuroExec：学习(和学习)快速注入攻击的执行触发器 cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.

摘要: 我们介绍了一类新的快速注入攻击，称为神经执行攻击。与依赖手工创建的字符串(例如，“忽略先前的指令和...”)的已知攻击不同，我们展示了将创建执行触发器概念化为可区分的搜索问题并使用基于学习的方法自主生成它们是可能的。我们的结果表明，有动机的对手可以伪造触发器，不仅比目前手工制作的触发器有效得多，而且在形状、属性和功能上表现出固有的灵活性。在这个方向上，我们展示了攻击者可以设计和生成能够在多阶段预处理管道中持久存在的神经Execs，例如在基于检索-增强生成(RAG)的应用程序的情况下。更关键的是，我们的发现表明，攻击者可以产生在形式和形状上与任何已知攻击显著偏离的触发器，绕过现有的基于黑名单的检测和卫生方法。



## **4. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

PPTC-R基准测试：评估大型语言模型在PowerPoint任务完成中的鲁棒性 cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.

摘要: 越来越多地依赖大型语言模型(LLM)来完成用户指令，这就需要全面了解它们在现实世界中完成复杂任务时的健壮性。为了解决这一关键需求，我们提出了PowerPoint任务完成健壮性基准(PPTC-R)来测量LLMS对用户PPT任务指令和软件版本的健壮性。具体地说，我们通过在句子、语义和多语言级别攻击用户指令来构建对抗性用户指令。为了评估语言模型对软件版本的稳健性，我们改变了提供的API的数量，以模拟最新版本和较早版本的设置。随后，我们使用结合了这些健壮性设置的基准测试了3个封闭源代码LLMS和4个开放源代码LLMS，旨在评估偏差如何影响LLMS完成任务的API调用。我们发现GPT-4在我们的基准测试中表现出了最高的性能和强大的健壮性，特别是在版本更新和多语言设置方面。然而，我们发现，当同时面对多个挑战(例如，多回合)时，所有的LLM都失去了它们的健壮性，导致性能显著下降。



## **5. Verification of Neural Networks' Global Robustness**

神经网络的全局健壮性验证 cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.

摘要: 神经网络在各种应用中都很成功，但也容易受到对抗性攻击。为了证明网络分类器的安全性，已经引入了许多验证器来推理给定输入对给定扰动的局部鲁棒性。虽然成功，局部鲁棒性不能推广到看不见的输入。一些作品分析了全局鲁棒性，但是，都不能提供一个精确的保证的情况下，网络分类器不改变其分类。在这项工作中，我们提出了一个新的全局鲁棒性的分类器，旨在寻找最小的全局鲁棒性的界限，这自然扩展了流行的局部鲁棒性的分类器。我们引入VHAGaR，一个随时验证计算这个界限。VHAGaR依赖于三个主要思想：将问题编码为混合整数规划，通过识别源自扰动或网络计算的依赖关系来修剪搜索空间，并将对抗性攻击推广到未知输入。我们在几个数据集和分类器上评估VHAGaR，并表明，给定三个小时的超时，VHAGaR计算的最小全局鲁棒性边界的下限和上限之间的平均差距为1.9，而现有的全局鲁棒性验证器的差距为154.7。此外，VHAGaR比这个验证器快130.6倍。我们的研究结果进一步表明，利用依赖性和对抗性攻击使VHAGaR的速度提高了78.6倍。



## **6. Simplified PCNet with Robustness**

具有健壮性的简化PCNet cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.

摘要: 图神经网络(GNN)因其在学习同亲图或异亲图的表示方面的成功而受到极大的关注。然而，它们不能很好地推广到具有不同同质性水平的真实世界的图。作为回应，Possion-Charlier Network(PCNet)引用了以前的工作{li2024pc}，允许从异形到同形学习图表示。虽然PCNet缓解了异质性问题，但在进一步提高疗效和效率方面仍存在一些挑战。在本文中，我们简化了PCNet，增强了它的健壮性。我们首先将滤波阶扩展到连续值，并对其参数进行降阶。实现了两种具有自适应邻域大小的变体。理论分析表明，该模型对图结构扰动或敌意攻击具有较强的稳健性。我们通过在不同数据集上的半监督学习任务来验证我们的方法，这些数据集既代表同嗜图，也代表异嗜图。



## **7. Adversarial Infrared Geometry: Using Geometry to Perform Adversarial Attack against Infrared Pedestrian Detectors**

对抗红外几何：利用几何对红外行人探测器进行对抗攻击 cs.CV

arXiv admin note: text overlap with arXiv:2312.14217 by other authors

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03674v1) [paper-pdf](http://arxiv.org/pdf/2403.03674v1)

**Authors**: Kalibinuer Tiliwalidi

**Abstract**: Currently, infrared imaging technology enjoys widespread usage, with infrared object detection technology experiencing a surge in prominence. While previous studies have delved into physical attacks on infrared object detectors, the implementation of these techniques remains complex. For instance, some approaches entail the use of bulb boards or infrared QR suits as perturbations to execute attacks, which entail costly optimization and cumbersome deployment processes. Other methodologies involve the utilization of irregular aerogel as physical perturbations for infrared attacks, albeit at the expense of optimization expenses and perceptibility issues. In this study, we propose a novel infrared physical attack termed Adversarial Infrared Geometry (\textbf{AdvIG}), which facilitates efficient black-box query attacks by modeling diverse geometric shapes (lines, triangles, ellipses) and optimizing their physical parameters using Particle Swarm Optimization (PSO). Extensive experiments are conducted to evaluate the effectiveness, stealthiness, and robustness of AdvIG. In digital attack experiments, line, triangle, and ellipse patterns achieve attack success rates of 93.1\%, 86.8\%, and 100.0\%, respectively, with average query times of 71.7, 113.1, and 2.57, respectively, thereby confirming the efficiency of AdvIG. Physical attack experiments are conducted to assess the attack success rate of AdvIG at different distances. On average, the line, triangle, and ellipse achieve attack success rates of 61.1\%, 61.2\%, and 96.2\%, respectively. Further experiments are conducted to comprehensively analyze AdvIG, including ablation experiments, transfer attack experiments, and adversarial defense mechanisms. Given the superior performance of our method as a simple and efficient black-box adversarial attack in both digital and physical environments, we advocate for widespread attention to AdvIG.

摘要: 目前，红外成像技术得到了广泛的应用，其中红外目标检测技术正在经历突出的激增。虽然以前的研究已经深入研究了对红外物体探测器的物理攻击，但这些技术的实现仍然很复杂。例如，一些方法需要使用灯泡板或红外QR套装作为执行攻击的扰动，这需要昂贵的优化和繁琐的部署过程。其他方法涉及利用不规则气凝胶作为红外攻击的物理扰动，尽管以优化费用和感知问题为代价。在这项研究中，我们提出了一种新的红外物理攻击称为对抗红外几何（\textbf{AdvIG}），它有利于有效的黑盒查询攻击建模不同的几何形状（线，三角形，椭圆形）和优化其物理参数，使用粒子群优化（PSO）。大量的实验进行评估的有效性，隐蔽性，和鲁棒性的AdvIG。在数字攻击实验中，直线、三角形和椭圆形模式的攻击成功率分别为93.1%、86.8%和100.0%，平均查询次数分别为71.7、113.1和2.57，验证了AdvIG的有效性。通过物理攻击实验，评估了AdvIG在不同距离下的攻击成功率。平均而言，直线、三角形和椭圆形的攻击成功率分别为61.1%、61.2%和96.2%。进一步的实验进行了全面的分析AdvIG，包括消融实验，转移攻击实验，和对抗性防御机制。鉴于我们的方法在数字和物理环境中作为一种简单有效的黑盒对抗攻击的优越性能，我们主张广泛关注AdvIG。



## **8. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

乐透：联合学习中对抗敌意服务器的安全参与者选择 cs.CR

This article has been accepted to USENIX Security '24

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2401.02880v2) [paper-pdf](http://arxiv.org/pdf/2401.02880v2)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-enhancing techniques, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively aligns the proportion of server-selected compromised participants with the base rate of dishonest clients in the population. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.

摘要: 在联邦学习(FL)中，常见的隐私增强技术，如安全聚合和分布式差异隐私，依赖于参与者之间诚实多数的关键假设来抵御各种攻击。然而，在实践中，服务器并不总是可信的，敌意服务器可以策略性地选择受攻击的客户端来制造不诚实的多数，从而破坏系统的安全保证。在本文中，我们提出了乐透，一个FL系统，解决了这个基本的，但探索不足的问题，通过提供安全的参与者选择对抗敌对的服务器。乐透支持两种选择算法：随机和通知。为了确保在没有可信服务器的情况下随机选择，乐透使每个客户端能够使用可验证的随机性自主确定他们的参与。对于更容易受到操纵的知情选择，乐透通过在改进的客户机池中使用随机选择来近似算法。我们的理论分析表明，乐透有效地将服务器选择的受攻击参与者的比例与人口中不诚实客户端的基本比率保持一致。大规模实验进一步表明，乐透算法的时间精度性能与非安全选择方法相当，表明安全选择方法具有较低的计算开销。



## **9. Noise-BERT: A Unified Perturbation-Robust Framework with Noise Alignment Pre-training for Noisy Slot Filling Task**

Noise-BERT：一种带噪声对齐预训练的统一扰动-稳健框架 cs.CL

Accepted by ICASSP 2024

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.14494v3) [paper-pdf](http://arxiv.org/pdf/2402.14494v3)

**Authors**: Jinxu Zhao, Guanting Dong, Yueyan Qiu, Tingfeng Hui, Xiaoshuai Song, Daichi Guo, Weiran Xu

**Abstract**: In a realistic dialogue system, the input information from users is often subject to various types of input perturbations, which affects the slot-filling task. Although rule-based data augmentation methods have achieved satisfactory results, they fail to exhibit the desired generalization when faced with unknown noise disturbances. In this study, we address the challenges posed by input perturbations in slot filling by proposing Noise-BERT, a unified Perturbation-Robust Framework with Noise Alignment Pre-training. Our framework incorporates two Noise Alignment Pre-training tasks: Slot Masked Prediction and Sentence Noisiness Discrimination, aiming to guide the pre-trained language model in capturing accurate slot information and noise distribution. During fine-tuning, we employ a contrastive learning loss to enhance the semantic representation of entities and labels. Additionally, we introduce an adversarial attack training strategy to improve the model's robustness. Experimental results demonstrate the superiority of our proposed approach over state-of-the-art models, and further analysis confirms its effectiveness and generalization ability.

摘要: 在现实的对话系统中，用户输入的信息往往会受到各种输入扰动的影响，从而影响到槽填充任务。虽然基于规则的数据增强方法已经取得了令人满意的结果，但当面对未知的噪声干扰时，它们未能表现出期望的泛化能力。在这项研究中，我们提出了噪声BERT，一个统一的扰动鲁棒框架与噪声对齐预训练槽填充输入扰动所带来的挑战。我们的框架结合了两个噪声对齐预训练任务：槽掩蔽预测和句子噪声识别，旨在指导预训练的语言模型捕获准确的槽信息和噪声分布。在微调过程中，我们采用对比学习损失来增强实体和标签的语义表示。此外，我们还引入了一种对抗性攻击训练策略来提高模型的鲁棒性。实验结果表明，我们提出的方法优于国家的最先进的模型，进一步的分析证实了其有效性和泛化能力。



## **10. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14197v2) [paper-pdf](http://arxiv.org/pdf/2312.14197v2)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.

摘要: 大型语言模型(LLM)与外部内容的集成使LLM能够更新、更广泛地应用，如Microsoft Copilot。然而，这种集成也使LLMS面临间接提示注入攻击的风险，攻击者可以在外部内容中嵌入恶意指令，损害LLM输出并导致响应偏离用户预期。为了研究这一重要但未被探索的问题，我们引入了第一个间接即时注入攻击基准，称为BIPIA，以评估此类攻击的风险。在评估的基础上，我们的工作重点分析了攻击成功的根本原因，即LLMS无法区分指令和外部内容，以及LLMS缺乏不执行外部内容中的指令的意识。在此基础上，我们提出了两种基于快速学习的黑盒防御方法和一种基于微调对抗性训练的白盒防御方法。实验结果表明，黑盒防御对于缓解这些攻击是非常有效的，而白盒防御将攻击成功率降低到接近于零的水平。总体而言，我们的工作通过引入基准、分析攻击成功的根本原因以及开发一套初始防御措施来系统地调查间接即时注入攻击。



## **11. TTPXHunter: Actionable Threat Intelligence Extraction as TTPs form Finished Cyber Threat Reports**

TTPXHunter：在TTP形成已完成的网络威胁报告时提取可操作的威胁情报 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.03267v1) [paper-pdf](http://arxiv.org/pdf/2403.03267v1)

**Authors**: Nanda Rani, Bikash Saha, Vikas Maurya, Sandeep Kumar Shukla

**Abstract**: Understanding the modus operandi of adversaries aids organizations in employing efficient defensive strategies and sharing intelligence in the community. This knowledge is often present in unstructured natural language text within threat analysis reports. A translation tool is needed to interpret the modus operandi explained in the sentences of the threat report and translate it into a structured format. This research introduces a methodology named TTPXHunter for the automated extraction of threat intelligence in terms of Tactics, Techniques, and Procedures (TTPs) from finished cyber threat reports. It leverages cyber domain-specific state-of-the-art natural language processing (NLP) to augment sentences for minority class TTPs and refine pinpointing the TTPs in threat analysis reports significantly. The knowledge of threat intelligence in terms of TTPs is essential for comprehensively understanding cyber threats and enhancing detection and mitigation strategies. We create two datasets: an augmented sentence-TTP dataset of 39,296 samples and a 149 real-world cyber threat intelligence report-to-TTP dataset. Further, we evaluate TTPXHunter on the augmented sentence dataset and the cyber threat reports. The TTPXHunter achieves the highest performance of 92.42% f1-score on the augmented dataset, and it also outperforms existing state-of-the-art solutions in TTP extraction by achieving an f1-score of 97.09% when evaluated over the report dataset. TTPXHunter significantly improves cybersecurity threat intelligence by offering quick, actionable insights into attacker behaviors. This advancement automates threat intelligence analysis, providing a crucial tool for cybersecurity professionals fighting cyber threats.

摘要: 了解对手的作案手法有助于组织采用有效的防御策略，并在社区中分享情报。这种知识通常出现在威胁分析报告中的非结构化自然语言文本中。需要一个翻译工具来解释威胁报告句子中解释的工作方式，并将其翻译成结构化格式。本研究介绍了一种名为TTPXHunter的方法，用于从已完成的网络威胁报告中自动提取策略、技术和过程(TTP)方面的威胁情报。它利用特定于网络领域的最先进的自然语言处理(NLP)来增加少数族裔类TTP的句子，并显著细化威胁分析报告中的TTP。TTP方面的威胁情报知识对于全面了解网络威胁和加强检测和缓解战略至关重要。我们创建了两个数据集：一个包含39,296个样本的增强句-TTP数据集，以及149个真实世界网络威胁情报报告到TTP的数据集。此外，我们在增加句子数据集和网络威胁报告上对TTPXHunter进行了评估。TTPXHunter在增强的数据集上获得了92.42%的F1分数的最高性能，在TTP提取方面也超过了现有的最先进的解决方案，在报告数据集上的F1分数达到了97.09%。TTPXHunter通过提供对攻击者行为的快速、可操作的洞察，显著提高了网络安全威胁情报。这一进步使威胁情报分析自动化，为应对网络威胁的网络安全专业人员提供了一个重要工具。



## **12. Attacks on Node Attributes in Graph Neural Networks**

图神经网络中节点属性的攻击 cs.SI

Accepted to AAAI 2024 AICS workshop

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2402.12426v2) [paper-pdf](http://arxiv.org/pdf/2402.12426v2)

**Authors**: Ying Xu, Michael Lanier, Anindya Sarkar, Yevgeniy Vorobeychik

**Abstract**: Graphs are commonly used to model complex networks prevalent in modern social media and literacy applications. Our research investigates the vulnerability of these graphs through the application of feature based adversarial attacks, focusing on both decision time attacks and poisoning attacks. In contrast to state of the art models like Net Attack and Meta Attack, which target node attributes and graph structure, our study specifically targets node attributes. For our analysis, we utilized the text dataset Hellaswag and graph datasets Cora and CiteSeer, providing a diverse basis for evaluation. Our findings indicate that decision time attacks using Projected Gradient Descent (PGD) are more potent compared to poisoning attacks that employ Mean Node Embeddings and Graph Contrastive Learning strategies. This provides insights for graph data security, pinpointing where graph-based models are most vulnerable and thereby informing the development of stronger defense mechanisms against such attacks.

摘要: 图通常用于对现代社交媒体和识字应用中普遍存在的复杂网络进行建模。我们的研究通过应用基于特征的对抗性攻击来研究这些图的脆弱性，重点研究了决策时攻击和中毒攻击。与网络攻击和元攻击等针对节点属性和图结构的最新模型不同，我们的研究专门针对节点属性。在我们的分析中，我们使用了文本数据集Hellaswag和图形数据集Cora和CiteSeer，为评估提供了多样化的基础。我们的发现表明，与使用均值节点嵌入和图对比学习策略的中毒攻击相比，使用投影梯度下降(PGD)的决策时间攻击更有效。这为图形数据安全提供了洞察力，准确地指出了基于图形的模型最易受攻击的位置，从而为开发针对此类攻击的更强大的防御机制提供了信息。



## **13. Mitigating Label Flipping Attacks in Malicious URL Detectors Using Ensemble Trees**

利用集成树减轻恶意URL检测器中的标签翻转攻击 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02995v1) [paper-pdf](http://arxiv.org/pdf/2403.02995v1)

**Authors**: Ehsan Nowroozi, Nada Jadalla, Samaneh Ghelichkhani, Alireza Jolfaei

**Abstract**: Malicious URLs provide adversarial opportunities across various industries, including transportation, healthcare, energy, and banking which could be detrimental to business operations. Consequently, the detection of these URLs is of crucial importance; however, current Machine Learning (ML) models are susceptible to backdoor attacks. These attacks involve manipulating a small percentage of training data labels, such as Label Flipping (LF), which changes benign labels to malicious ones and vice versa. This manipulation results in misclassification and leads to incorrect model behavior. Therefore, integrating defense mechanisms into the architecture of ML models becomes an imperative consideration to fortify against potential attacks.   The focus of this study is on backdoor attacks in the context of URL detection using ensemble trees. By illuminating the motivations behind such attacks, highlighting the roles of attackers, and emphasizing the critical importance of effective defense strategies, this paper contributes to the ongoing efforts to fortify ML models against adversarial threats within the ML domain in network security. We propose an innovative alarm system that detects the presence of poisoned labels and a defense mechanism designed to uncover the original class labels with the aim of mitigating backdoor attacks on ensemble tree classifiers. We conducted a case study using the Alexa and Phishing Site URL datasets and showed that LF attacks can be addressed using our proposed defense mechanism. Our experimental results prove that the LF attack achieved an Attack Success Rate (ASR) between 50-65% within 2-5%, and the innovative defense method successfully detected poisoned labels with an accuracy of up to 100%.

摘要: 恶意URL为包括交通、医疗、能源和银行在内的多个行业提供了敌意机会，可能会对业务运营造成不利影响。因此，对这些URL的检测至关重要；然而，当前的机器学习(ML)模型容易受到后门攻击。这些攻击涉及操纵一小部分训练数据标签，例如标签翻转(LF)，它将良性标签更改为恶意标签，反之亦然。这种操作会导致错误分类，并导致不正确的模型行为。因此，将防御机制集成到ML模型的体系结构中成为防御潜在攻击的当务之急。这项研究的重点是利用集成树检测URL上下文中的后门攻击。通过阐明此类攻击背后的动机，突出攻击者的作用，并强调有效防御策略的关键重要性，本文有助于在网络安全中加强ML域内对抗对手威胁的ML模型的持续努力。我们提出了一种创新的警报系统来检测有毒标签的存在，并提出了一种旨在发现原始类别标签的防御机制，目的是减少对集成树分类器的后门攻击。我们使用Alexa和网络钓鱼网站的URL数据集进行了一个案例研究，并表明可以使用我们提出的防御机制来应对LF攻击。实验结果表明，LF攻击在2%-5%的范围内达到了50%-65%的攻击成功率(ASR)，新的防御方法成功地检测到了有毒标签，准确率达到100%。



## **14. Federated Learning Under Attack: Exposing Vulnerabilities through Data Poisoning Attacks in Computer Networks**

攻击下的联合学习：通过计算机网络中的数据中毒攻击暴露漏洞 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02983v1) [paper-pdf](http://arxiv.org/pdf/2403.02983v1)

**Authors**: Ehsan Nowroozi, Imran Haider, Rahim Taheri, Mauro Conti

**Abstract**: Federated Learning (FL) is a machine learning (ML) approach that enables multiple decentralized devices or edge servers to collaboratively train a shared model without exchanging raw data. During the training and sharing of model updates between clients and servers, data and models are susceptible to different data-poisoning attacks.   In this study, our motivation is to explore the severity of data poisoning attacks in the computer network domain because they are easy to implement but difficult to detect. We considered two types of data-poisoning attacks, label flipping (LF) and feature poisoning (FP), and applied them with a novel approach. In LF, we randomly flipped the labels of benign data and trained the model on the manipulated data. For FP, we randomly manipulated the highly contributing features determined using the Random Forest algorithm. The datasets used in this experiment were CIC and UNSW related to computer networks. We generated adversarial samples using the two attacks mentioned above, which were applied to a small percentage of datasets. Subsequently, we trained and tested the accuracy of the model on adversarial datasets. We recorded the results for both benign and manipulated datasets and observed significant differences between the accuracy of the models on different datasets. From the experimental results, it is evident that the LF attack failed, whereas the FP attack showed effective results, which proved its significance in fooling a server. With a 1% LF attack on the CIC, the accuracy was approximately 0.0428 and the ASR was 0.9564; hence, the attack is easily detectable, while with a 1% FP attack, the accuracy and ASR were both approximately 0.9600, hence, FP attacks are difficult to detect. We repeated the experiment with different poisoning percentages.

摘要: 联合学习(FL)是一种机器学习(ML)方法，它使多个分散的设备或边缘服务器能够在不交换原始数据的情况下协作地训练共享模型。在客户端和服务器之间模型更新的训练和共享过程中，数据和模型容易受到不同的数据中毒攻击。在这项研究中，我们的动机是探索计算机网络领域中的数据中毒攻击的严重性，因为它们易于实现但难以检测。我们考虑了两种类型的数据中毒攻击，标签翻转(LF)和特征中毒(FP)，并将它们应用于一种新的方法。在LF中，我们随机翻转良性数据的标签，并在被操纵的数据上训练模型。对于FP，我们随机处理使用随机森林算法确定的高贡献特征。本实验中使用的数据集是与计算机网络相关的CIC和新南威尔士大学。我们使用上面提到的两种攻击生成了对抗性样本，这些攻击应用于一小部分数据集。随后，我们在对抗性数据集上训练和测试了该模型的准确性。我们记录了良性数据集和操纵数据集的结果，并观察到不同数据集上模型的准确性存在显著差异。从实验结果可以看出，LF攻击是失败的，而FP攻击是有效的，这证明了它在欺骗服务器方面的重要意义。在对CIC进行1%的LF攻击时，准确率约为0.0428，ASR为0.9564；因此，攻击很容易被检测到；而对于1%的FP攻击，准确率和ASR都约为0.9600，因此，FP攻击很难被检测到。我们用不同的中毒百分比重复了实验。



## **15. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度袖口：通过探索拒绝损失场景来检测对大型语言模型的越狱攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.00867v2) [paper-pdf](http://arxiv.org/pdf/2403.00867v2)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **16. XAI-Based Detection of Adversarial Attacks on Deepfake Detectors**

基于XAI的深伪检测器对抗性攻击检测 cs.CR

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02955v1) [paper-pdf](http://arxiv.org/pdf/2403.02955v1)

**Authors**: Ben Pinhasov, Raz Lapid, Rony Ohayon, Moshe Sipper, Yehudit Aperstein

**Abstract**: We introduce a novel methodology for identifying adversarial attacks on deepfake detectors using eXplainable Artificial Intelligence (XAI). In an era characterized by digital advancement, deepfakes have emerged as a potent tool, creating a demand for efficient detection systems. However, these systems are frequently targeted by adversarial attacks that inhibit their performance. We address this gap, developing a defensible deepfake detector by leveraging the power of XAI. The proposed methodology uses XAI to generate interpretability maps for a given method, providing explicit visualizations of decision-making factors within the AI models. We subsequently employ a pretrained feature extractor that processes both the input image and its corresponding XAI image. The feature embeddings extracted from this process are then used for training a simple yet effective classifier. Our approach contributes not only to the detection of deepfakes but also enhances the understanding of possible adversarial attacks, pinpointing potential vulnerabilities. Furthermore, this approach does not change the performance of the deepfake detector. The paper demonstrates promising results suggesting a potential pathway for future deepfake detection mechanisms. We believe this study will serve as a valuable contribution to the community, sparking much-needed discourse on safeguarding deepfake detectors.

摘要: 我们介绍了一种利用可解释人工智能(XAI)来识别针对深度假冒检测器的对抗性攻击的新方法。在一个以数字进步为特征的时代，深度假冒已经成为一种强有力的工具，创造了对高效检测系统的需求。然而，这些系统经常成为抑制其性能的对抗性攻击的目标。我们解决了这个问题，通过利用XAI的能力开发了一个可防御的深度伪检测器。所提出的方法使用XAI为给定的方法生成可解释性地图，提供人工智能模型中决策因素的显式可视化。随后，我们采用了一个预先训练的特征抽取器来处理输入图像及其对应的XAI图像。然后使用从该过程中提取的特征嵌入来训练简单而有效的分类器。我们的方法不仅有助于深度假冒的检测，还有助于增强对可能的敌意攻击的理解，准确地定位潜在的漏洞。此外，该方法不会改变深度伪检测器的性能。这篇论文展示了令人振奋的结果，为未来的深度伪检测机制提供了一条潜在的途径。我们相信，这项研究将对社区做出有价值的贡献，引发关于保护深度假冒探测器的迫切需要的讨论。



## **17. Precise Extraction of Deep Learning Models via Side-Channel Attacks on Edge/Endpoint Devices**

基于边缘/端点设备旁通道攻击的深度学习模型精确提取 cs.AI

Accepted by 27th European Symposium on Research in Computer Security  (ESORICS 2022)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02870v1) [paper-pdf](http://arxiv.org/pdf/2403.02870v1)

**Authors**: Younghan Lee, Sohee Jun, Yungi Cho, Woorim Han, Hyungon Moon, Yunheung Paek

**Abstract**: With growing popularity, deep learning (DL) models are becoming larger-scale, and only the companies with vast training datasets and immense computing power can manage their business serving such large models. Most of those DL models are proprietary to the companies who thus strive to keep their private models safe from the model extraction attack (MEA), whose aim is to steal the model by training surrogate models. Nowadays, companies are inclined to offload the models from central servers to edge/endpoint devices. As revealed in the latest studies, adversaries exploit this opportunity as new attack vectors to launch side-channel attack (SCA) on the device running victim model and obtain various pieces of the model information, such as the model architecture (MA) and image dimension (ID). Our work provides a comprehensive understanding of such a relationship for the first time and would benefit future MEA studies in both offensive and defensive sides in that they may learn which pieces of information exposed by SCA are more important than the others. Our analysis additionally reveals that by grasping the victim model information from SCA, MEA can get highly effective and successful even without any prior knowledge of the model. Finally, to evince the practicality of our analysis results, we empirically apply SCA, and subsequently, carry out MEA under realistic threat assumptions. The results show up to 5.8 times better performance than when the adversary has no model information about the victim model.

摘要: 随着深度学习的日益普及，深度学习模型的规模越来越大，只有拥有海量训练数据集和巨大计算能力的公司才能管理自己的业务，为如此大规模的模型服务。这些DL模型中的大多数都是公司的专利，这些公司因此努力使他们的私人模型免受模型提取攻击(MEA)，其目的是通过训练代理模型来窃取模型。如今，公司倾向于将模型从中央服务器转移到边缘/终端设备。最新研究表明，攻击者利用这一机会作为新的攻击载体，对运行受害者模型的设备发起侧通道攻击(SCA)，获得模型的各种信息，如模型体系结构(MA)和图像维度(ID)。我们的工作首次提供了对这种关系的全面理解，并将有助于未来攻防双方的MEA研究，因为他们可以了解到SCA暴露的哪些信息比其他信息更重要。我们的分析还表明，通过从SCA中获取受害者模型信息，MEA即使在没有任何模型先验知识的情况下也可以获得高效和成功的信息。最后，为了证明我们的分析结果的实用性，我们实证地应用了SCA，并随后在现实的威胁假设下进行了MEA。结果表明，与对手没有关于受害者模型的模型信息时相比，性能最高可提高5.8倍。



## **18. FLGuard: Byzantine-Robust Federated Learning via Ensemble of Contrastive Models**

FLGuard：通过对比模型的嵌入实现拜占庭鲁棒联邦学习 cs.LG

Accepted by 28th European Symposium on Research in Computer Security  (ESORICS 2023)

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02846v1) [paper-pdf](http://arxiv.org/pdf/2403.02846v1)

**Authors**: Younghan Lee, Yungi Cho, Woorim Han, Ho Bae, Yunheung Paek

**Abstract**: Federated Learning (FL) thrives in training a global model with numerous clients by only sharing the parameters of their local models trained with their private training datasets. Therefore, without revealing the private dataset, the clients can obtain a deep learning (DL) model with high performance. However, recent research proposed poisoning attacks that cause a catastrophic loss in the accuracy of the global model when adversaries, posed as benign clients, are present in a group of clients. Therefore, recent studies suggested byzantine-robust FL methods that allow the server to train an accurate global model even with the adversaries present in the system. However, many existing methods require the knowledge of the number of malicious clients or the auxiliary (clean) dataset or the effectiveness reportedly decreased hugely when the private dataset was non-independently and identically distributed (non-IID). In this work, we propose FLGuard, a novel byzantine-robust FL method that detects malicious clients and discards malicious local updates by utilizing the contrastive learning technique, which showed a tremendous improvement as a self-supervised learning method. With contrastive models, we design FLGuard as an ensemble scheme to maximize the defensive capability. We evaluate FLGuard extensively under various poisoning attacks and compare the accuracy of the global model with existing byzantine-robust FL methods. FLGuard outperforms the state-of-the-art defense methods in most cases and shows drastic improvement, especially in non-IID settings. https://github.com/201younghanlee/FLGuard

摘要: 联合学习(FL)通过仅共享使用其私有训练数据集训练的本地模型的参数，在与众多客户训练全球模型方面蓬勃发展。因此，在不透露私有数据集的情况下，客户端可以获得高性能的深度学习(DL)模型。然而，最近的研究提出，当一组客户中存在伪装成良性客户的对手时，中毒攻击会导致全球模型准确性的灾难性损失。因此，最近的研究建议拜占庭稳健的FL方法，允许服务器即使在系统中存在对手的情况下也能训练准确的全局模型。然而，许多现有的方法需要知道恶意客户端或辅助(干净)数据集的数量，或者当私有数据集是非独立且相同分布(非IID)时，据报道其有效性大大降低。在这项工作中，我们提出了一种新的拜占庭稳健FL方法--FLGuard，它利用对比学习技术检测恶意客户端并丢弃恶意本地更新，作为一种自我监督学习方法，显示出巨大的改进。通过对比模型，我们将FLGuard设计为一个整体方案，以最大限度地提高防御能力。我们在各种中毒攻击下对FLGuard进行了广泛的评估，并将全局模型的准确性与现有的拜占庭稳健FL方法进行了比较。在大多数情况下，FLGuard的表现优于最先进的防御方法，并显示出显著的改进，特别是在非IID设置中。Https://github.com/201younghanlee/FLGuard



## **19. Here Comes The AI Worm: Unleashing Zero-click Worms that Target GenAI-Powered Applications**

AI蠕虫来了：释放针对GenAI支持的应用的零点击蠕虫 cs.CR

Website: https://sites.google.com/view/compromptmized

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02817v1) [paper-pdf](http://arxiv.org/pdf/2403.02817v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In the past year, numerous companies have incorporated Generative AI (GenAI) capabilities into new and existing applications, forming interconnected Generative AI (GenAI) ecosystems consisting of semi/fully autonomous agents powered by GenAI services. While ongoing research highlighted risks associated with the GenAI layer of agents (e.g., dialog poisoning, membership inference, prompt leaking, jailbreaking), a critical question emerges: Can attackers develop malware to exploit the GenAI component of an agent and launch cyber-attacks on the entire GenAI ecosystem? This paper introduces Morris II, the first worm designed to target GenAI ecosystems through the use of adversarial self-replicating prompts. The study demonstrates that attackers can insert such prompts into inputs that, when processed by GenAI models, prompt the model to replicate the input as output (replication), engaging in malicious activities (payload). Additionally, these inputs compel the agent to deliver them (propagate) to new agents by exploiting the connectivity within the GenAI ecosystem. We demonstrate the application of Morris II against GenAIpowered email assistants in two use cases (spamming and exfiltrating personal data), under two settings (black-box and white-box accesses), using two types of input data (text and images). The worm is tested against three different GenAI models (Gemini Pro, ChatGPT 4.0, and LLaVA), and various factors (e.g., propagation rate, replication, malicious activity) influencing the performance of the worm are evaluated.

摘要: 在过去的一年里，许多公司将生成性人工智能(GenAI)功能整合到新的和现有的应用程序中，形成了由GenAI服务支持的半/全自主代理组成的互联生成性AI(GenAI)生态系统。虽然正在进行的研究突出了与GenAI代理层相关的风险(例如，对话中毒、成员关系推断、提示泄漏、越狱)，但一个关键问题出现了：攻击者是否可以开发恶意软件来利用代理的GenAI组件，并对整个GenAI生态系统发动网络攻击？本文介绍了Morris II，它是第一个通过使用对抗性自我复制提示来攻击GenAI生态系统的蠕虫。这项研究表明，攻击者可以将这样的提示插入到输入中，当被GenAI模型处理时，提示模型将输入复制为输出(复制)，参与恶意活动(有效负载)。此外，这些输入迫使代理通过利用GenAI生态系统中的连接将它们交付(传播)给新的代理。我们使用两种类型的输入数据(文本和图像)，在两种设置(黑盒和白盒访问)下，在两种用例(垃圾邮件和渗漏个人数据)中演示了Morris II对GenAI支持的电子邮件助理的应用。该蠕虫针对三种不同的GenAI模型(Gemini Pro、ChatGPT 4.0和LLaVA)进行了测试，并评估了影响该蠕虫性能的各种因素(例如，传播速度、复制、恶意活动)。



## **20. Towards Robust Federated Learning via Logits Calibration on Non-IID Data**

基于非IID数据Logits校正的稳健联合学习 cs.CV

Accepted by IEEE NOMS 2024

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02803v1) [paper-pdf](http://arxiv.org/pdf/2403.02803v1)

**Authors**: Yu Qiao, Apurba Adhikary, Chaoning Zhang, Choong Seon Hong

**Abstract**: Federated learning (FL) is a privacy-preserving distributed management framework based on collaborative model training of distributed devices in edge networks. However, recent studies have shown that FL is vulnerable to adversarial examples (AEs), leading to a significant drop in its performance. Meanwhile, the non-independent and identically distributed (non-IID) challenge of data distribution between edge devices can further degrade the performance of models. Consequently, both AEs and non-IID pose challenges to deploying robust learning models at the edge. In this work, we adopt the adversarial training (AT) framework to improve the robustness of FL models against adversarial example (AE) attacks, which can be termed as federated adversarial training (FAT). Moreover, we address the non-IID challenge by implementing a simple yet effective logits calibration strategy under the FAT framework, which can enhance the robustness of models when subjected to adversarial attacks. Specifically, we employ a direct strategy to adjust the logits output by assigning higher weights to classes with small samples during training. This approach effectively tackles the class imbalance in the training data, with the goal of mitigating biases between local and global models. Experimental results on three dataset benchmarks, MNIST, Fashion-MNIST, and CIFAR-10 show that our strategy achieves competitive results in natural and robust accuracy compared to several baselines.

摘要: 联合学习(FL)是一种基于边缘网络中分布式设备协作模型训练的隐私保护分布式管理框架。然而，最近的研究表明，外语容易受到对抗性例子的影响，导致其成绩显著下降。同时，边缘设备之间数据分发的非独立和同分布(Non-IID)挑战可能会进一步降低模型的性能。因此，企业环境和非独立企业都对在边缘部署强大的学习模型提出了挑战。在这项工作中，我们采用对抗性训练(AT)框架来提高FL模型对对抗性范例(AE)攻击的健壮性，这可以被称为联合对抗性训练(FAT)。此外，我们通过在FAT框架下实现一种简单而有效的Logits校准策略来应对非IID的挑战，该策略可以增强模型在受到对抗性攻击时的健壮性。具体地说，我们采用一种直接策略来调整LOGITS输出，方法是在训练期间为样本较小的类分配更高的权重。这种方法有效地解决了训练数据中的类不平衡问题，目标是减轻局部模型和全局模型之间的偏差。在MNIST、Fashion-MNIST和CIFAR-10三个数据集基准上的实验结果表明，与几个基准相比，我们的策略在自然和稳健的准确率方面取得了与之相当的结果。



## **21. On the Alignment of Group Fairness with Attribute Privacy**

关于组公平性与属性隐私的匹配问题 cs.LG

arXiv admin note: text overlap with arXiv:2202.02242

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2211.10209v3) [paper-pdf](http://arxiv.org/pdf/2211.10209v3)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Group fairness and privacy are fundamental aspects in designing trustworthy machine learning models. Previous research has highlighted conflicts between group fairness and different privacy notions. We are the first to demonstrate the alignment of group fairness with the specific privacy notion of attribute privacy in a blackbox setting. Attribute privacy, quantified by the resistance to attribute inference attacks (AIAs), requires indistinguishability in the target model's output predictions. Group fairness guarantees this thereby mitigating AIAs and achieving attribute privacy. To demonstrate this, we first introduce AdaptAIA, an enhancement of existing AIAs, tailored for real-world datasets with class imbalances in sensitive attributes. Through theoretical and extensive empirical analyses, we demonstrate the efficacy of two standard group fairness algorithms (i.e., adversarial debiasing and exponentiated gradient descent) against AdaptAIA. Additionally, since using group fairness results in attribute privacy, it acts as a defense against AIAs, which is currently lacking. Overall, we show that group fairness aligns with attribute privacy at no additional cost other than the already existing trade-off with model utility.

摘要: 群体公平和隐私是设计可信机器学习模型的基本方面。之前的研究已经强调了群体公平和不同隐私观念之间的冲突。我们首先展示了在黑箱设置中，组公平与属性隐私的特定隐私概念的一致性。属性隐私由对属性推理攻击的抵抗力(AIAS)来量化，要求目标模型的输出预测不可区分。组公平性保证了这一点，从而减轻了AIAS并实现了属性隐私。为了说明这一点，我们首先引入了AdaptAIA，它是现有AIAS的增强，专为具有敏感属性中的类不平衡的真实世界数据集而定制。通过理论和大量的实证分析，我们证明了两种标准的群体公平算法(即对抗性去偏向算法和指数梯度下降算法)对AdaptAIA的有效性。此外，由于使用组公平会导致属性隐私，因此它充当了对AIAS的防御，而AIAS目前是缺乏的。总体而言，我们表明，除了已经存在的与模型实用程序的权衡外，组公平与属性隐私保持一致，而不需要额外的成本。



## **22. Minimum Topology Attacks for Graph Neural Networks**

图神经网络的最小拓扑攻击 cs.AI

Published on WWW 2023. Proceedings of the ACM Web Conference 2023

**SubmitDate**: 2024-03-05    [abs](http://arxiv.org/abs/2403.02723v1) [paper-pdf](http://arxiv.org/pdf/2403.02723v1)

**Authors**: Mengmei Zhang, Xiao Wang, Chuan Shi, Lingjuan Lyu, Tianchi Yang, Junping Du

**Abstract**: With the great popularity of Graph Neural Networks (GNNs), their robustness to adversarial topology attacks has received significant attention. Although many attack methods have been proposed, they mainly focus on fixed-budget attacks, aiming at finding the most adversarial perturbations within a fixed budget for target node. However, considering the varied robustness of each node, there is an inevitable dilemma caused by the fixed budget, i.e., no successful perturbation is found when the budget is relatively small, while if it is too large, the yielding redundant perturbations will hurt the invisibility. To break this dilemma, we propose a new type of topology attack, named minimum-budget topology attack, aiming to adaptively find the minimum perturbation sufficient for a successful attack on each node. To this end, we propose an attack model, named MiBTack, based on a dynamic projected gradient descent algorithm, which can effectively solve the involving non-convex constraint optimization on discrete topology. Extensive results on three GNNs and four real-world datasets show that MiBTack can successfully lead all target nodes misclassified with the minimum perturbation edges. Moreover, the obtained minimum budget can be used to measure node robustness, so we can explore the relationships of robustness, topology, and uncertainty for nodes, which is beyond what the current fixed-budget topology attacks can offer.

摘要: 随着图神经网络(GNN)的广泛应用，其对敌意拓扑攻击的健壮性受到了广泛的关注。虽然已经提出了许多攻击方法，但它们主要集中在固定预算攻击上，旨在为目标节点在固定预算内找到最具对抗性的扰动。然而，考虑到每个节点的健壮性不同，固定预算不可避免地造成了一个两难境地，即当预算相对较小时，没有找到成功的扰动，而如果预算太大，则产生的冗余扰动将损害不可见性。为了打破这一困境，我们提出了一种新的拓扑攻击，称为最小预算拓扑攻击，旨在自适应地找到对每个节点进行成功攻击所需的最小扰动。为此，我们提出了一种基于动态投影梯度下降算法的攻击模型MiBTack，该模型可以有效地解决离散拓扑上涉及的非凸约束优化问题。在3个GNN和4个真实数据集上的广泛结果表明，MiBTack能够成功地以最小扰动边引导所有目标节点的错误分类。此外，所得到的最小预算可以用来衡量节点的健壮性，从而可以探索节点的健壮性、拓扑性和不确定性之间的关系，这是目前固定预算拓扑攻击所不能提供的。



## **23. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

SCAR：激光雷达目标检测的对抗性缩放算法 cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2312.03085v2) [paper-pdf](http://arxiv.org/pdf/2312.03085v2)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method. Our code is available at https://github.com/xiaohulugo/ScAR-IROS2023.

摘要: 模型的对抗性健壮性在于其抵抗以输入数据的小扰动形式的对抗性攻击的能力。快速符号梯度法(FSGM)和投影梯度下降法(PGD)等通用对抗攻击方法是激光雷达目标检测的常用方法，但与特定任务的对抗攻击相比往往存在不足。此外，这些通用方法通常需要不受限制地访问模型的信息，这在现实世界的应用程序中是很难获得的。为了解决这些局限性，我们提出了一种用于激光雷达目标检测的黑盒尺度对抗稳健性(SCAR)方法。通过分析Kitti、Waymo和nuScenes等3D目标检测数据集的统计特性，我们发现模型的预测对3D实例的缩放很敏感。我们根据已有的信息提出了三种黑盒尺度对抗性攻击方法：模型感知攻击、分布感知攻击和盲目攻击。我们还介绍了一种生成伸缩敌意实例的策略，以提高模型对这三种伸缩对手攻击的稳健性。在不同3D目标检测体系结构下的公共数据集上与其他方法进行了比较，验证了该方法的有效性。我们的代码可以在https://github.com/xiaohulugo/ScAR-IROS2023.上找到



## **24. Towards Poisoning Fair Representations**

走向毒害公平代表 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2309.16487v2) [paper-pdf](http://arxiv.org/pdf/2309.16487v2)

**Authors**: Tianci Liu, Haoyu Wang, Feijie Wu, Hengtong Zhang, Pan Li, Lu Su, Jing Gao

**Abstract**: Fair machine learning seeks to mitigate model prediction bias against certain demographic subgroups such as elder and female. Recently, fair representation learning (FRL) trained by deep neural networks has demonstrated superior performance, whereby representations containing no demographic information are inferred from the data and then used as the input to classification or other downstream tasks. Despite the development of FRL methods, their vulnerability under data poisoning attack, a popular protocol to benchmark model robustness under adversarial scenarios, is under-explored. Data poisoning attacks have been developed for classical fair machine learning methods which incorporate fairness constraints into shallow-model classifiers. Nonetheless, these attacks fall short in FRL due to notably different fairness goals and model architectures. This work proposes the first data poisoning framework attacking FRL. We induce the model to output unfair representations that contain as much demographic information as possible by injecting carefully crafted poisoning samples into the training data. This attack entails a prohibitive bilevel optimization, wherefore an effective approximated solution is proposed. A theoretical analysis on the needed number of poisoning samples is derived and sheds light on defending against the attack. Experiments on benchmark fairness datasets and state-of-the-art fair representation learning models demonstrate the superiority of our attack.

摘要: 公平的机器学习寻求减轻模型预测对某些人口子组的偏差，例如老年人和女性。最近，由深度神经网络训练的公平表征学习(FRL)表现出了优越的性能，即从数据中推断出不包含人口统计信息的表征，然后将其用作分类或其他下游任务的输入。尽管FRL方法已经得到了发展，但它们在数据中毒攻击下的脆弱性还没有得到充分的探索。数据中毒攻击是一种流行的协议，用于在对抗场景下对模型的健壮性进行基准测试。数据中毒攻击是针对将公平性约束引入浅模型分类器的经典公平机器学习方法而开发的。尽管如此，由于公平目标和模型架构的显著不同，这些攻击在FRL上仍存在不足。本文提出了第一个攻击FRL的数据中毒框架。我们通过将精心制作的中毒样本注入训练数据来诱导模型输出包含尽可能多的人口统计信息的不公平表示。这种攻击需要一个禁止的两层优化，因此提出了一个有效的近似解。对所需中毒样本数量进行了理论分析，为防御攻击提供了理论依据。在基准公平性数据集和最新的公平表示学习模型上的实验证明了该攻击的优越性。



## **25. COMMIT: Certifying Robustness of Multi-Sensor Fusion Systems against Semantic Attacks**

Commit：证明多传感器融合系统对语义攻击的健壮性 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02329v1) [paper-pdf](http://arxiv.org/pdf/2403.02329v1)

**Authors**: Zijian Huang, Wenda Chu, Linyi Li, Chejian Xu, Bo Li

**Abstract**: Multi-sensor fusion systems (MSFs) play a vital role as the perception module in modern autonomous vehicles (AVs). Therefore, ensuring their robustness against common and realistic adversarial semantic transformations, such as rotation and shifting in the physical world, is crucial for the safety of AVs. While empirical evidence suggests that MSFs exhibit improved robustness compared to single-modal models, they are still vulnerable to adversarial semantic transformations. Despite the proposal of empirical defenses, several works show that these defenses can be attacked again by new adaptive attacks. So far, there is no certified defense proposed for MSFs. In this work, we propose the first robustness certification framework COMMIT certify robustness of multi-sensor fusion systems against semantic attacks. In particular, we propose a practical anisotropic noise mechanism that leverages randomized smoothing with multi-modal data and performs a grid-based splitting method to characterize complex semantic transformations. We also propose efficient algorithms to compute the certification in terms of object detection accuracy and IoU for large-scale MSF models. Empirically, we evaluate the efficacy of COMMIT in different settings and provide a comprehensive benchmark of certified robustness for different MSF models using the CARLA simulation platform. We show that the certification for MSF models is at most 48.39% higher than that of single-modal models, which validates the advantages of MSF models. We believe our certification framework and benchmark will contribute an important step towards certifiably robust AVs in practice.

摘要: 多传感器融合系统作为感知模块在现代自动驾驶汽车中发挥着重要作用。因此，确保它们对常见和现实的对抗性语义转换（例如物理世界中的旋转和移位）的鲁棒性对于AV的安全性至关重要。虽然经验证据表明，与单模态模型相比，MSF表现出更好的鲁棒性，但它们仍然容易受到对抗性语义转换的影响。尽管建议的经验防御，一些作品表明，这些防御可以再次攻击新的自适应攻击。到目前为止，还没有为MSF提出的认证辩护。在这项工作中，我们提出了第一个鲁棒性认证框架COMMIT证明多传感器融合系统对语义攻击的鲁棒性。特别是，我们提出了一个实用的各向异性噪声机制，利用多模态数据的随机平滑，并执行基于网格的分裂方法来表征复杂的语义转换。我们还提出了有效的算法来计算大规模MSF模型的对象检测精度和IoU的认证。从经验上讲，我们评估了在不同的设置COMMIT的功效，并提供了一个全面的基准认证的鲁棒性不同的MSF模型使用CARLA仿真平台。我们发现，MSF模型的认证是最多48.39%，高于单模态模型，这验证了MSF模型的优势。我们相信，我们的认证框架和基准将有助于在实践中向可认证的强大AV迈出重要一步。



## **26. Mirage: Defense against CrossPath Attacks in Software Defined Networks**

幻影：软件定义网络中的交叉路径攻击防御 cs.CR

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.02172v1) [paper-pdf](http://arxiv.org/pdf/2403.02172v1)

**Authors**: Shariq Murtuza, Krishna Asawa

**Abstract**: The Software-Defined Networks (SDNs) face persistent threats from various adversaries that attack them using different methods to mount Denial of Service attacks. These attackers have different motives and follow diverse tactics to achieve their nefarious objectives. In this work, we focus on the impact of CrossPath attacks in SDNs and introduce our framework, Mirage, which not only detects but also mitigates this attack. Our framework, Mirage, detects SDN switches that become unreachable due to being under attack, takes proactive measures to prevent Adversarial Path Reconnaissance, and effectively mitigates CrossPath attacks in SDNs. A CrossPath attack is a form of link flood attack that indirectly attacks the control plane by overwhelming the shared links that connect the data and control planes with data plane traffic. This attack is exclusive to in band SDN, where the data and the control plane, both utilize the same physical links for transmitting and receiving traffic. Our framework, Mirage, prevents attackers from launching adversarial path reconnaissance to identify shared links in a network, thereby thwarting their abuse and preventing this attack. Mirage not only stops adversarial path reconnaissance but also includes features to quickly counter ongoing attacks once detected. Mirage uses path diversity to reroute network packet to prevent timing based measurement. Mirage can also enforce short lived flow table rules to prevent timing attacks. These measures are carefully designed to enhance the security of the SDN environment. Moreover, we share the results of our experiments, which clearly show Mirage's effectiveness in preventing path reconnaissance, detecting CrossPath attacks, and mitigating ongoing threats. Our framework successfully protects the network from these harmful activities, giving valuable insights into SDN security.

摘要: 软件定义网络(SDN)面临来自各种对手的持续威胁，这些对手使用不同的方法发动拒绝服务攻击。这些袭击者有不同的动机，采取不同的战术来实现他们的邪恶目的。在这项工作中，我们重点研究了跨路径攻击在SDNS中的影响，并介绍了我们的框架Mige，它不仅可以检测到这种攻击，而且可以缓解这种攻击。我们的框架MIRAGE检测由于受到攻击而变得不可达的SDN交换机，采取主动措施防止恶意路径侦察，并有效地缓解SDN中的CrossPath攻击。交叉路径攻击是链路泛洪攻击的一种形式，它通过压倒连接数据平面和控制平面与数据平面流量的共享链路来间接攻击控制平面。此攻击专用于带内SDN，其中数据和控制平面都使用相同的物理链路来传输和接收流量。我们的框架，幻影，防止攻击者发起敌对的路径侦察来识别网络中的共享链路，从而挫败他们的滥用，防止这种攻击。幻影不仅停止敌对路径侦察，还包括一旦检测到快速反击正在进行的攻击的功能。幻影使用路径分集来重新路由网络数据包，以防止基于时序的测量。幻影还可以强制执行短期流表规则，以防止计时攻击。这些措施都是精心设计的，以加强SDN环境的安全。此外，我们还分享了我们的实验结果，这些结果清楚地表明了幻影在防止路径侦察、检测交叉路径攻击和缓解持续威胁方面的有效性。我们的框架成功地保护网络免受这些有害活动的影响，为SDN安全提供了有价值的见解。



## **27. Rethinking Model Ensemble in Transfer-based Adversarial Attacks**

基于迁移的对抗性攻击中模型集成的再思考 cs.CV

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2303.09105v2) [paper-pdf](http://arxiv.org/pdf/2303.09105v2)

**Authors**: Huanran Chen, Yichi Zhang, Yinpeng Dong, Xiao Yang, Hang Su, Jun Zhu

**Abstract**: It is widely recognized that deep learning models lack robustness to adversarial examples. An intriguing property of adversarial examples is that they can transfer across different models, which enables black-box attacks without any knowledge of the victim model. An effective strategy to improve the transferability is attacking an ensemble of models. However, previous works simply average the outputs of different models, lacking an in-depth analysis on how and why model ensemble methods can strongly improve the transferability. In this paper, we rethink the ensemble in adversarial attacks and define the common weakness of model ensemble with two properties: 1) the flatness of loss landscape; and 2) the closeness to the local optimum of each model. We empirically and theoretically show that both properties are strongly correlated with the transferability and propose a Common Weakness Attack (CWA) to generate more transferable adversarial examples by promoting these two properties. Experimental results on both image classification and object detection tasks validate the effectiveness of our approach to improving the adversarial transferability, especially when attacking adversarially trained models. We also successfully apply our method to attack a black-box large vision-language model -- Google's Bard, showing the practical effectiveness. Code is available at \url{https://github.com/huanranchen/AdversarialAttacks}.

摘要: 人们普遍认为，深度学习模型对对抗性例子缺乏稳健性。对抗性例子的一个耐人寻味的特性是，它们可以在不同的模型之间传输，这使得在不知道受害者模型的情况下进行黑盒攻击。提高可转移性的一个有效策略是攻击一系列模型。然而，以往的工作只是简单地对不同模型的输出进行平均，而缺乏对模型集成方法如何以及为什么能够显著提高可转移性的深入分析。在本文中，我们重新考虑了对抗性攻击中的集成，并定义了模型集成的两个共同弱点：1)损失图景的平坦性；2)每个模型接近局部最优。我们从经验和理论上证明了这两个性质与可转移性有很强的相关性，并提出了一种共同弱点攻击(CWA)，通过提升这两个性质来生成更多可转移的对抗性实例。在图像分类和目标检测任务上的实验结果验证了该方法的有效性，特别是在攻击对抗性训练模型时。我们还成功地应用我们的方法攻击了一个黑盒大视觉语言模型--Google的BARD，显示了它的实际有效性。代码可在\url{https://github.com/huanranchen/AdversarialAttacks}.上找到



## **28. Robustness Bounds on the Successful Adversarial Examples: Theory and Practice**

成功对抗性例子的稳健性界限：理论与实践 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01896v1) [paper-pdf](http://arxiv.org/pdf/2403.01896v1)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification. We proved a new upper bound that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.

摘要: 对抗性例子(AE)是一种机器学习的攻击方法，它是通过对导致错误分类的数据添加不可察觉的扰动来构建的。在本文中，我们研究了基于高斯过程(GP)分类的AES成功概率的上界。我们证明了一个新的上界，它依赖于AE的扰动范数，GP中使用的核函数，以及训练数据集中具有不同标签的最近对的距离。令人惊讶的是，无论样本数据集的分布如何，上限都是确定的。我们通过使用ImageNet的实验验证了我们的理论结果。此外，我们还证明了改变核函数的参数会引起成功事件概率的上界的变化。



## **29. One Prompt Word is Enough to Boost Adversarial Robustness for Pre-trained Vision-Language Models**

一个提示词就足以提高预先训练的视觉语言模型的对抗性 cs.CV

CVPR2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2403.01849v1) [paper-pdf](http://arxiv.org/pdf/2403.01849v1)

**Authors**: Lin Li, Haoyan Guan, Jianing Qiu, Michael Spratling

**Abstract**: Large pre-trained Vision-Language Models (VLMs) like CLIP, despite having remarkable generalization ability, are highly vulnerable to adversarial examples. This work studies the adversarial robustness of VLMs from the novel perspective of the text prompt instead of the extensively studied model weights (frozen in this work). We first show that the effectiveness of both adversarial attack and defense are sensitive to the used text prompt. Inspired by this, we propose a method to improve resilience to adversarial attacks by learning a robust text prompt for VLMs. The proposed method, named Adversarial Prompt Tuning (APT), is effective while being both computationally and data efficient. Extensive experiments are conducted across 15 datasets and 4 data sparsity schemes (from 1-shot to full training data settings) to show APT's superiority over hand-engineered prompts and other state-of-the-art adaption methods. APT demonstrated excellent abilities in terms of the in-distribution performance and the generalization under input distribution shift and across datasets. Surprisingly, by simply adding one learned word to the prompts, APT can significantly boost the accuracy and robustness (epsilon=4/255) over the hand-engineered prompts by +13% and +8.5% on average respectively. The improvement further increases, in our most effective setting, to +26.4% for accuracy and +16.7% for robustness. Code is available at https://github.com/TreeLLi/APT.

摘要: 像CLIP这样的大型预先训练的视觉语言模型(VLM)，尽管具有显著的泛化能力，但很容易受到对手例子的攻击。该工作从文本提示的新角度来研究VLMS的对抗健壮性，而不是广泛研究的模型权重(在本工作中是冻结的)。我们首先证明了对抗性攻击和防御的有效性都对所使用的文本提示敏感。受此启发，我们提出了一种通过学习VLM的健壮文本提示来提高对对手攻击的恢复能力的方法。该方法称为对抗性提示调优(APT)，在计算效率和数据效率上都是有效的。在15个数据集和4个数据稀疏方案(从单镜头到全训练数据设置)上进行了广泛的实验，以展示APT相对于人工设计提示和其他最先进的适应方法的优势。在输入分布平移和跨数据集情况下，APT在分布内性能和泛化能力方面表现出优异的性能。令人惊讶的是，通过简单地在提示中添加一个学习的单词，APT可以显著提高人工设计提示的准确率和稳健性(epsilon=4/255)，平均分别提高+13%和+8.5%。在我们最有效的设置中，改进进一步增加了准确性的+26.4%和健壮性的+16.7%。代码可在https://github.com/TreeLLi/APT.上找到



## **30. Coupling Graph Neural Networks with Fractional Order Continuous Dynamics: A Robustness Study**

分数阶连续动态耦合图神经网络的稳健性研究 cs.LG

in Proc. AAAI Conference on Artificial Intelligence, Vancouver,  Canada, Feb. 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2401.04331v2) [paper-pdf](http://arxiv.org/pdf/2401.04331v2)

**Authors**: Qiyu Kang, Kai Zhao, Yang Song, Yihang Xie, Yanan Zhao, Sijie Wang, Rui She, Wee Peng Tay

**Abstract**: In this work, we rigorously investigate the robustness of graph neural fractional-order differential equation (FDE) models. This framework extends beyond traditional graph neural (integer-order) ordinary differential equation (ODE) models by implementing the time-fractional Caputo derivative. Utilizing fractional calculus allows our model to consider long-term memory during the feature updating process, diverging from the memoryless Markovian updates seen in traditional graph neural ODE models. The superiority of graph neural FDE models over graph neural ODE models has been established in environments free from attacks or perturbations. While traditional graph neural ODE models have been verified to possess a degree of stability and resilience in the presence of adversarial attacks in existing literature, the robustness of graph neural FDE models, especially under adversarial conditions, remains largely unexplored. This paper undertakes a detailed assessment of the robustness of graph neural FDE models. We establish a theoretical foundation outlining the robustness characteristics of graph neural FDE models, highlighting that they maintain more stringent output perturbation bounds in the face of input and graph topology disturbances, compared to their integer-order counterparts. Our empirical evaluations further confirm the enhanced robustness of graph neural FDE models, highlighting their potential in adversarially robust applications.

摘要: 在这项工作中，我们严格研究了图神经分数阶微分方程(FDE)模型的稳健性。该框架通过实现时间分数Caputo导数，扩展了传统的图神经(整数阶)常微分方程(ODE)模型。利用分数阶微积分，我们的模型可以在特征更新过程中考虑长期记忆，不同于传统的图神经节点模型中看到的无记忆的马尔可夫更新。图神经FDE模型相对于图神经ODE模型的优越性已经在没有攻击或扰动的环境中得到了证实。虽然已有文献证明传统的图神经FDE模型在对抗攻击下具有一定程度的稳定性和韧性，但图神经FDE模型的稳健性，特别是在对抗条件下的稳健性，在很大程度上还没有被探索。本文对图神经FDE模型的稳健性进行了详细的评估。我们建立了一个理论基础，概述了图神经FDE模型的稳健性特征，强调了它们在面对输入和图的拓扑扰动时，与其整数阶对应模型相比，保持了更严格的输出摄动界。我们的经验评估进一步证实了图神经FDE模型的增强的稳健性，突出了它们在相反的健壮应用中的潜力。



## **31. LLMs Can Defend Themselves Against Jailbreaking in a Practical Manner: A Vision Paper**

LLMS能够以实际的方式保护自己免受越狱：一份愿景文件 cs.CR

Fixed the bibliography reference issue in our LLM jailbreak defense  vision paper submitted on 24 Feb 2024

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.15727v2) [paper-pdf](http://arxiv.org/pdf/2402.15727v2)

**Authors**: Daoyuan Wu, Shuai Wang, Yang Liu, Ning Liu

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs). A considerable amount of research exists proposing more effective jailbreak attacks, including the recent Greedy Coordinate Gradient (GCG) attack, jailbreak template-based attacks such as using "Do-Anything-Now" (DAN), and multilingual jailbreak. In contrast, the defensive side has been relatively less explored. This paper proposes a lightweight yet practical defense called SELFDEFEND, which can defend against all existing jailbreak attacks with minimal delay for jailbreak prompts and negligible delay for normal user prompts. Our key insight is that regardless of the kind of jailbreak strategies employed, they eventually need to include a harmful prompt (e.g., "how to make a bomb") in the prompt sent to LLMs, and we found that existing LLMs can effectively recognize such harmful prompts that violate their safety policies. Based on this insight, we design a shadow stack that concurrently checks whether a harmful prompt exists in the user prompt and triggers a checkpoint in the normal stack once a token of "No" or a harmful prompt is output. The latter could also generate an explainable LLM response to adversarial prompts. We demonstrate our idea of SELFDEFEND works in various jailbreak scenarios through manual analysis in GPT-3.5/4. We also list three future directions to further enhance SELFDEFEND.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐。已有大量研究提出了更有效的越狱攻击方案，包括最近的贪婪坐标梯度(GCG)攻击、基于模板的越狱攻击(例如使用“Do-Anything-Now”(DAN))和多语言越狱。相比之下，防守方面的探索相对较少。本文提出了一种轻量级而实用的防御方法SELFDEFEND，它可以防御所有现有的越狱攻击，而越狱提示的延迟最小，正常用户提示的延迟可以忽略不计。我们的主要见解是，无论采用哪种越狱策略，他们最终都需要在发送给LLMS的提示中包含有害提示(例如，如何制造炸弹)，我们发现现有LLMS可以有效地识别此类违反其安全政策的有害提示。基于这一观点，我们设计了一个影子堆栈，该堆栈同时检查用户提示中是否存在有害提示，并在输出令牌“否”或有害提示时触发正常堆栈中的检查点。后者还可以对对抗性提示产生可解释的LLM响应。我们通过GPT-3.5/4中的手动分析，展示了我们的SELFDEFEND在各种越狱场景中的工作原理。我们还列出了进一步增强SELFDEFEND的三个未来方向。



## **32. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

共享扩散模型中的隐私和公平风险：对抗性视角 cs.LG

**SubmitDate**: 2024-03-04    [abs](http://arxiv.org/abs/2402.18607v2) [paper-pdf](http://arxiv.org/pdf/2402.18607v2)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.

摘要: 扩散模型由于其在抽样质量和分布复盖率方面令人印象深刻的生成性能，最近在学术界和工业界都得到了极大的关注。因此，提出了在不同组织之间共享预先培训的传播模式的建议，以此作为提高数据利用率的一种方式，同时通过避免直接共享私人数据来加强隐私保护。然而，与这种方法相关的潜在风险尚未得到全面审查。在这篇文章中，我们采取对抗性的视角来调查与共享扩散模型相关的潜在的隐私和公平风险。具体地说，我们调查了一方(共享者)使用私有数据训练扩散模型，并为另一方(接收者)提供对下游任务的预训练模型的黑箱访问的情况。我们证明了共享者可以通过操纵扩散模型的训练数据分布来执行公平毒化攻击来破坏接收者的下游模型。同时，接收者可以执行属性推断攻击，以揭示共享者数据集中敏感特征的分布。我们在真实数据集上进行的实验表明，在不同类型的扩散模型上具有显著的攻击性能，这突显了健壮的数据审计和隐私保护协议在相关应用中的关键重要性。



## **33. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

Accepted at AISTATS 2024, a preliminary version appeared at ICML 2023  AdvML Workshop

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2008.09312v7) [paper-pdf](http://arxiv.org/pdf/2008.09312v7)

**Authors**: Shiliang Zuo

**Abstract**: I study adversarial attacks against stochastic bandit algorithms. At each round, the learner chooses an arm, and a stochastic reward is generated. The adversary strategically adds corruption to the reward, and the learner is only able to observe the corrupted reward at each round. Two sets of results are presented in this paper. The first set studies the optimal attack strategies for the adversary. The adversary has a target arm he wishes to promote, and his goal is to manipulate the learner into choosing this target arm $T - o(T)$ times. I design attack strategies against UCB and Thompson Sampling that only spends $\widehat{O}(\sqrt{\log T})$ cost. Matching lower bounds are presented, and the vulnerability of UCB, Thompson sampling and $\varepsilon$-greedy are exactly characterized. The second set studies how the learner can defend against the adversary. Inspired by literature on smoothed analysis and behavioral economics, I present two simple algorithms that achieve a competitive ratio arbitrarily close to 1.

摘要: 我研究针对随机盗贼算法的敌意攻击。在每一轮，学习者选择一只手臂，并产生随机奖励。对手策略性地将腐败添加到奖励中，而学习者在每一轮只能观察到腐败的奖励。本文给出了两组结果。第一组研究对手的最优攻击策略。对手有一个他想要提升的目标手臂，他的目标是操纵学习者选择这个目标手臂$T-O(T)$次。我设计了针对UCB和Thompson抽样的攻击策略，该策略只花费$\widehat{O}(\sqrt{\log T})$。给出了匹配下界，并准确刻画了UCB、Thompson抽样和$varepsilon$-贪婪的脆弱性。第二组研究学习者如何防御对手。受平滑分析和行为经济学文献的启发，我提出了两个简单的算法，它们的竞争比率任意接近1。



## **34. Adversarial Attacks on Fairness of Graph Neural Networks**

图神经网络公平性的对抗性攻击 cs.LG

Accepted at ICLR 2024

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2310.13822v2) [paper-pdf](http://arxiv.org/pdf/2310.13822v2)

**Authors**: Binchi Zhang, Yushun Dong, Chen Chen, Yada Zhu, Minnan Luo, Jundong Li

**Abstract**: Fairness-aware graph neural networks (GNNs) have gained a surge of attention as they can reduce the bias of predictions on any demographic group (e.g., female) in graph-based applications. Although these methods greatly improve the algorithmic fairness of GNNs, the fairness can be easily corrupted by carefully designed adversarial attacks. In this paper, we investigate the problem of adversarial attacks on fairness of GNNs and propose G-FairAttack, a general framework for attacking various types of fairness-aware GNNs in terms of fairness with an unnoticeable effect on prediction utility. In addition, we propose a fast computation technique to reduce the time complexity of G-FairAttack. The experimental study demonstrates that G-FairAttack successfully corrupts the fairness of different types of GNNs while keeping the attack unnoticeable. Our study on fairness attacks sheds light on potential vulnerabilities in fairness-aware GNNs and guides further research on the robustness of GNNs in terms of fairness.

摘要: 在基于图的应用中，公平感知图神经网络(GNN)可以减少对任何人口统计群体(例如，女性)的预测偏差，因此受到了广泛的关注。虽然这些方法极大地提高了GNN算法的公平性，但这种公平性很容易被精心设计的对抗性攻击所破坏。本文研究了对GNN公平性的敌意攻击问题，提出了G-FairAttack框架，该框架从公平性的角度攻击各种类型的公平感知GNN，并且对预测效用没有明显的影响。此外，我们还提出了一种快速计算技术来降低G-FairAttack的时间复杂度。实验研究表明，G-FairAttack在保持攻击不可察觉的同时，成功地破坏了不同类型GNN的公平性。我们对公平攻击的研究揭示了公平感知网络中潜在的漏洞，并指导了进一步研究公平感知网络的健壮性。



## **35. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAprompt：通过全球规模的即时黑客竞争暴露LLMs的系统漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2024-03-03    [abs](http://arxiv.org/abs/2311.16119v3) [paper-pdf](http://arxiv.org/pdf/2311.16119v3)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型(LLM)部署在具有直接用户参与的交互上下文中，例如聊天机器人和写作助手。这些部署容易受到即时注入和越狱(统称为即时黑客)的攻击，在这些情况下，模型被操纵以忽略其原始指令并遵循潜在的恶意指令。尽管被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和量化研究。为了弥补这一漏洞，我们发起了一场全球即时黑客竞赛，允许自由形式的人工输入攻击。我们在三个最先进的LLM上获得了600K+的对抗性提示。我们描述了数据集，这从经验上验证了当前的LLM确实可以通过即时黑客来操纵。我们还提出了对抗性提示类型的全面分类本体。



## **36. Gradient Shaping: Enhancing Backdoor Attack Against Reverse Engineering**

渐变成形：增强对逆向工程的后门攻击 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2301.12318v2) [paper-pdf](http://arxiv.org/pdf/2301.12318v2)

**Authors**: Rui Zhu, Di Tang, Siyuan Tang, Guanhong Tao, Shiqing Ma, Xiaofeng Wang, Haixu Tang

**Abstract**: Most existing methods to detect backdoored machine learning (ML) models take one of the two approaches: trigger inversion (aka. reverse engineer) and weight analysis (aka. model diagnosis). In particular, the gradient-based trigger inversion is considered to be among the most effective backdoor detection techniques, as evidenced by the TrojAI competition, Trojan Detection Challenge and backdoorBench. However, little has been done to understand why this technique works so well and, more importantly, whether it raises the bar to the backdoor attack. In this paper, we report the first attempt to answer this question by analyzing the change rate of the backdoored model around its trigger-carrying inputs. Our study shows that existing attacks tend to inject the backdoor characterized by a low change rate around trigger-carrying inputs, which are easy to capture by gradient-based trigger inversion. In the meantime, we found that the low change rate is not necessary for a backdoor attack to succeed: we design a new attack enhancement called \textit{Gradient Shaping} (GRASP), which follows the opposite direction of adversarial training to reduce the change rate of a backdoored model with regard to the trigger, without undermining its backdoor effect. Also, we provide a theoretic analysis to explain the effectiveness of this new technique and the fundamental weakness of gradient-based trigger inversion. Finally, we perform both theoretical and experimental analysis, showing that the GRASP enhancement does not reduce the effectiveness of the stealthy attacks against the backdoor detection methods based on weight analysis, as well as other backdoor mitigation methods without using detection.

摘要: 大多数现有的检测回溯机器学习(ML)模型的方法都采用两种方法之一：触发反转(又名。逆向工程)和权重分析(又名模型诊断)。特别是，基于梯度的触发器反转被认为是最有效的后门检测技术之一，特洛伊木马竞赛、特洛伊木马检测挑战赛和后门B边就是证明。然而，对于这种技术为什么如此有效，以及更重要的是，它是否提高了后门攻击的门槛，人们几乎没有做过什么。在这篇文章中，我们首次尝试通过分析回溯模型围绕其触发输入的变化率来回答这个问题。我们的研究表明，现有的攻击倾向于注入带有触发器的输入周围变化率低的后门，这很容易通过基于梯度的触发器反转来捕获。同时，我们发现低更改率并不是后门攻击成功的必要条件：我们设计了一种新的攻击增强机制，称为文本{梯度整形}(GRASH)，它遵循对抗性训练的相反方向，在不破坏后门效应的情况下，降低后门模型关于触发的更改率。此外，我们还从理论上分析了这一新技术的有效性以及基于梯度的触发反演的根本缺陷。最后，我们进行了理论和实验分析，结果表明抓取能力的增强不会降低基于权重分析的后门检测方法以及其他不使用检测的后门防御方法的隐身攻击的有效性。



## **37. Fusion is Not Enough: Single Modal Attacks on Fusion Models for 3D Object Detection**

融合还不够：用于3D目标检测的融合模型的单模式攻击 cs.CV

Accepted at ICLR'2024

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2304.14614v3) [paper-pdf](http://arxiv.org/pdf/2304.14614v3)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely used in autonomous vehicles (AVs) for perception, particularly for 3D object detection with camera and LiDAR sensors. The purpose of fusion is to capitalize on the advantages of each modality while minimizing its weaknesses. Advanced deep neural network (DNN)-based fusion techniques have demonstrated the exceptional and industry-leading performance. Due to the redundant information in multiple modalities, MSF is also recognized as a general defence strategy against adversarial attacks. In this paper, we attack fusion models from the camera modality that is considered to be of lesser importance in fusion but is more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion-based 3D object detection models through camera-only adversarial attacks. Our approach employs a two-stage optimization-based strategy that first thoroughly evaluates vulnerable image areas under adversarial attacks, and then applies dedicated attack strategies for different fusion models to generate deployable patches. The evaluations with six advanced camera-LiDAR fusion models and one camera-only model indicate that our attacks successfully compromise all of them. Our approach can either decrease the mean average precision (mAP) of detection performance from 0.824 to 0.353, or degrade the detection score of a target object from 0.728 to 0.156, demonstrating the efficacy of our proposed attack framework. Code is available.

摘要: 多传感器融合(MSF)被广泛应用于自动驾驶车辆(AVs)中的感知，特别是在带有摄像头和LiDAR传感器的三维目标检测中。融合的目的是利用每种模式的优点，同时将其缺点降至最低。先进的基于深度神经网络(DNN)的融合技术展示了卓越的行业领先性能。由于多通道的冗余信息，MSF也被认为是对抗对手攻击的一种一般防御策略。在本文中，我们从摄像机模式来攻击融合模型，这种模式被认为在融合中不那么重要，但对于攻击者来说更容易负担得起。我们认为融合模型的最薄弱环节取决于它们最脆弱的通道，并提出了一种攻击框架，通过仅针对摄像机的对抗性攻击来攻击先进的基于摄像机-LiDAR融合的三维目标检测模型。该方法采用基于优化的两阶段策略，首先对易受攻击的图像区域进行全面评估，然后针对不同的融合模型应用专门的攻击策略生成可部署的补丁。对6个先进的摄像机-激光雷达融合模型和1个仅摄像机模型的评估表明，我们的攻击成功地折衷了所有这些模型。我们的方法可以将检测性能的平均精度(MAP)从0.824降低到0.353，或者将目标对象的检测得分从0.728降低到0.156，从而证明了我们所提出的攻击框架的有效性。代码可用。



## **38. Accelerating Greedy Coordinate Gradient via Probe Sampling**

利用探头采样加速贪婪坐标梯度 cs.CL

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01251v1) [paper-pdf](http://arxiv.org/pdf/2403.01251v1)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a central issue given their rapid progress and wide applications. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing prompts containing adversarial suffixes to break the presumingly safe LLMs, but the optimization of GCG is time-consuming and limits its practicality. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$ to accelerate the GCG algorithm. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates to reduce the computation time. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b and leads to equal or improved attack success rate (ASR) on the AdvBench.

摘要: 由于大型语言模型的快速发展和广泛应用，其安全性已成为一个核心问题。贪婪坐标梯度(GCG)被证明能有效地构造含有敌意后缀的提示，以打破假定安全的LLMS，但GCG的优化耗时长，限制了其实用性。为了减少GCG算法的时间开销，更全面地研究LLM的安全性，本文研究了一种新的加速GCG算法的算法该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者，以减少计算时间。使用Llama2-7b，探头采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。



## **39. Inexact Unlearning Needs More Careful Evaluations to Avoid a False Sense of Privacy**

不准确的遗忘需要更仔细的评估以避免错误的隐私感 cs.LG

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01218v1) [paper-pdf](http://arxiv.org/pdf/2403.01218v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Eleni Triantafillou, Amr Khalifa, Nicolas Papernot

**Abstract**: The high cost of model training makes it increasingly desirable to develop techniques for unlearning. These techniques seek to remove the influence of a training example without having to retrain the model from scratch. Intuitively, once a model has unlearned, an adversary that interacts with the model should no longer be able to tell whether the unlearned example was included in the model's training set or not. In the privacy literature, this is known as membership inference. In this work, we discuss adaptations of Membership Inference Attacks (MIAs) to the setting of unlearning (leading to their ``U-MIA'' counterparts). We propose a categorization of existing U-MIAs into ``population U-MIAs'', where the same attacker is instantiated for all examples, and ``per-example U-MIAs'', where a dedicated attacker is instantiated for each example. We show that the latter category, wherein the attacker tailors its membership prediction to each example under attack, is significantly stronger. Indeed, our results show that the commonly used U-MIAs in the unlearning literature overestimate the privacy protection afforded by existing unlearning techniques on both vision and language models. Our investigation reveals a large variance in the vulnerability of different examples to per-example U-MIAs. In fact, several unlearning algorithms lead to a reduced vulnerability for some, but not all, examples that we wish to unlearn, at the expense of increasing it for other examples. Notably, we find that the privacy protection for the remaining training examples may worsen as a consequence of unlearning. We also discuss the fundamental difficulty of equally protecting all examples using existing unlearning schemes, due to the different rates at which examples are unlearned. We demonstrate that naive attempts at tailoring unlearning stopping criteria to different examples fail to alleviate these issues.

摘要: 模型训练的高成本使得人们越来越希望开发用于遗忘的技术。这些技术试图消除训练示例的影响，而不必从头开始重新训练模型。直觉上，一旦模型被取消学习，与模型交互的对手就不应该再能够分辨出未学习的示例是否包含在模型的训练集中。在隐私文献中，这被称为成员推断。在这项工作中，我们讨论了成员推理攻击（MIA）对unlearning设置的适应（导致其“U-MIA”对应物）。我们提出了一个分类现有的U-MIA到“人口U-MIA”，其中相同的攻击者被实例化的所有例子，和“每个例子U-MIA”，其中一个专用的攻击者被实例化的每个例子。我们发现，后者类别，其中攻击者量身定制的成员预测，每个例子下的攻击，是显着更强。事实上，我们的研究结果表明，unlearning文献中常用的U-MIA高估了现有unlearning技术在视觉和语言模型上提供的隐私保护。我们的调查揭示了一个很大的差异，不同的例子，每个例子U-MIA的脆弱性。事实上，几种unlearning算法可以减少我们希望unlearn的一些（但不是全部）示例的漏洞，但代价是增加其他示例的漏洞。值得注意的是，我们发现剩余训练示例的隐私保护可能会因遗忘而恶化。我们还讨论了基本的困难，同样保护所有的例子使用现有的unlearning计划，由于不同的速度，在这些例子是unlearned。我们证明，天真的尝试在不同的例子定制遗忘停止标准无法缓解这些问题。



## **40. SAR-AE-SFP: SAR Imagery Adversarial Example in Real Physics domain with Target Scattering Feature Parameters**

SAR-AE-SFP：具有目标散射特征参数的真实物理域SAR图像对抗实例 cs.CV

10 pages, 9 figures, 2 tables

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01210v1) [paper-pdf](http://arxiv.org/pdf/2403.01210v1)

**Authors**: Jiahao Cui, Jiale Duan, Binyan Luo, Hang Cao, Wang Guo, Haifeng Li

**Abstract**: Deep neural network-based Synthetic Aperture Radar (SAR) target recognition models are susceptible to adversarial examples. Current adversarial example generation methods for SAR imagery primarily operate in the 2D digital domain, known as image adversarial examples. Recent work, while considering SAR imaging scatter mechanisms, fails to account for the actual imaging process, rendering attacks in the three-dimensional physical domain infeasible, termed pseudo physics adversarial examples. To address these challenges, this paper proposes SAR-AE-SFP-Attack, a method to generate real physics adversarial examples by altering the scattering feature parameters of target objects. Specifically, we iteratively optimize the coherent energy accumulation of the target echo by perturbing the reflection coefficient and scattering coefficient in the scattering feature parameters of the three-dimensional target object, and obtain the adversarial example after echo signal processing and imaging processing in the RaySAR simulator. Experimental results show that compared to digital adversarial attack methods, SAR-AE-SFP Attack significantly improves attack efficiency on CNN-based models (over 30\%) and Transformer-based models (over 13\%), demonstrating significant transferability of attack effects across different models and perspectives.

摘要: 基于深度神经网络的合成孔径雷达(SAR)目标识别模型容易受到敌意例子的影响。目前的合成孔径雷达图像对抗性样本生成方法主要是在2D数字域内进行的，称为图像对抗性样本。最近的工作虽然考虑了SAR成像的散射机制，但未能解释实际的成像过程，使得在三维物理领域进行攻击是不可行的，被称为伪物理对抗性例子。为了应对这些挑战，本文提出了一种通过改变目标对象的散射特征参数来生成真实物理对抗实例的方法--SAR-AE-SFP-Attack。具体而言，通过摄动三维目标目标散射特征参数中的反射系数和散射系数，迭代优化目标回波的相干能量积累，并在RaySAR模拟器中进行回波信号处理和成像处理后，得到了反例。实验结果表明，与数字对抗性攻击方法相比，SAR-AE-SFP攻击显著提高了基于CNN模型(超过30个模型)和基于Transformer模型(超过13个模型)的攻击效率，表现出显著的攻击效果在不同模型和视角之间的可转移性。



## **41. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

利用机器学习的速度和准确性提高网络安全 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2302.12415v3) [paper-pdf](http://arxiv.org/pdf/2302.12415v3)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity

摘要: 随着网络攻击的频率和复杂性不断增加，检测恶意软件已成为维护计算机系统安全的关键任务。传统的基于特征码的恶意软件检测方法在检测复杂和不断变化的威胁方面存在局限性。近年来，机器学习作为一种有效检测恶意软件的解决方案应运而生。ML算法能够分析大型数据集，并识别人类难以识别的模式。本文对恶意软件检测中使用的最大似然学习技术进行了全面的综述，包括监督学习和非监督学习、深度学习和强化学习。我们还研究了基于ML的恶意软件检测的挑战和局限性，例如潜在的对抗性攻击和对大量标记数据的需求。此外，我们还讨论了基于ML的恶意软件检测的未来发展方向，包括集成多种ML算法和使用可解释人工智能技术来增强基于ML的检测系统的解释能力。我们的研究突出了基于ML的技术在提高恶意软件检测的速度和准确性方面的潜力，并有助于增强网络安全



## **42. False Claims against Model Ownership Resolution**

针对模型所有权解决方案的虚假声明 cs.CR

13pages,3 figures. To appear in the 33rd USENIX Security Symposium  (USENIX Security '24)

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2304.06607v5) [paper-pdf](http://arxiv.org/pdf/2304.06607v5)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation, we demonstrate that our false claim attacks always succeed in the MOR schemes that follow our generalization, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的实证评估，我们证明了我们的虚假声明攻击在遵循我们的推广的MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **43. Self-Guided Robust Graph Structure Refinement**

自引导鲁棒图结构求精 cs.LG

This paper has been accepted by TheWebConf 2024 (Oral Presentation)

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2402.11837v2) [paper-pdf](http://arxiv.org/pdf/2402.11837v2)

**Authors**: Yeonjun In, Kanghoon Yoon, Kibum Kim, Kijung Shin, Chanyoung Park

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. To defend against such attacks, robust graph structure refinement (GSR) methods aim at minimizing the effect of adversarial edges based on node features, graph structure, or external information. However, we have discovered that existing GSR methods are limited by narrowassumptions, such as assuming clean node features, moderate structural attacks, and the availability of external clean graphs, resulting in the restricted applicability in real-world scenarios. In this paper, we propose a self-guided GSR framework (SG-GSR), which utilizes a clean sub-graph found within the given attacked graph itself. Furthermore, we propose a novel graph augmentation and a group-training strategy to handle the two technical challenges in the clean sub-graph extraction: 1) loss of structural information, and 2) imbalanced node degree distribution. Extensive experiments demonstrate the effectiveness of SG-GSR under various scenarios including non-targeted attacks, targeted attacks, feature attacks, e-commerce fraud, and noisy node labels. Our code is available at https://github.com/yeonjun-in/torch-SG-GSR.

摘要: 最近的研究表明，GNN很容易受到对抗性攻击。为了防御这类攻击，基于节点特征、图结构或外部信息的健壮图结构精化(GSR)方法旨在最小化敌方边的影响。在本文中，我们提出了一种自导GSR框架(SG-GSR)，它利用了在给定的攻击图本身中发现的干净的子图。大量实验验证了SG-GSR在非定向攻击、定向攻击、特征攻击、电子商务欺诈、节点标签噪声等场景下的有效性。我们的代码可以在https://github.com/yeonjun-in/torch-SG-GSR.上找到



## **44. Attacking the Diebold Signature Variant -- RSA Signatures with Unverified High-order Padding**

攻击Diebold型签名变种--使用未经验证的高阶填充的RSA签名 cs.CR

**SubmitDate**: 2024-03-02    [abs](http://arxiv.org/abs/2403.01048v1) [paper-pdf](http://arxiv.org/pdf/2403.01048v1)

**Authors**: Ryan W. Gardner, Tadayoshi Kohno, Alec Yasinsac

**Abstract**: We examine a natural but improper implementation of RSA signature verification deployed on the widely used Diebold Touch Screen and Optical Scan voting machines. In the implemented scheme, the verifier fails to examine a large number of the high-order bits of signature padding and the public exponent is three. We present an very mathematically simple attack that enables an adversary to forge signatures on arbitrary messages in a negligible amount of time.

摘要: 我们研究了一个自然的，但不适当的RSA签名验证部署在广泛使用的Diebold触摸屏和光学扫描投票机的实现。在实现的方案中，验证者无法检查签名填充的大量高阶位，并且公共指数为3。我们提出了一个数学上非常简单的攻击，使对手伪造签名的任意消息在一个微不足道的时间。



## **45. Resilience of Entropy Model in Distributed Neural Networks**

分布式神经网络熵模型的弹性 cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00942v1) [paper-pdf](http://arxiv.org/pdf/2403.00942v1)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.

摘要: 分布式深度神经网络(DNN)已成为边缘计算系统中在不牺牲性能的前提下减少通信开销的关键技术。最近，引入了熵编码来进一步降低通信开销。该算法的核心思想是将分布的DNN与一个熵模型联合训练，作为推理时的辅助信息，自适应地将潜在的表示编码成可变长度的比特流。就我们所知，熵模型的弹性还有待研究。因此，在本文中，我们建立并研究了熵模型对有意干扰(例如，对抗性攻击)和无意干扰(例如，天气变化和运动模糊)的弹性。通过使用3种不同的DNN结构、2种熵模型和4种率失真权衡因子的广泛实验活动，我们证明了熵攻击可以使通信开销增加高达95%。通过在频域和空间域分离压缩特征，我们提出了一种新的防御机制，与未受干扰的数据相比，该机制可以使被攻击输入的传输开销减少约9%，而精确度损失仅约2%。重要的是，建议的防御机制是一种独立的方法，可以与对抗性训练等方法结合使用，以进一步提高健壮性。代码将被共享，以实现重现性。



## **46. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

扩散模型流形中对抗性例子的错位 cs.CV

under review

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2401.06637v4) [paper-pdf](http://arxiv.org/pdf/2401.06637v4)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

摘要: 近年来，扩散模型(DM)因其在近似数据分布方面的成功而引起了人们的极大关注，产生了最先进的生成结果。然而，这些模型的多功能性超出了它们的生成能力，涵盖了各种视觉应用，如图像修复、分割、对抗性鲁棒性等。本研究致力于从扩散模型的角度研究对抗性攻击。然而，我们的目标不涉及增强图像分类器的对抗性稳健性。相反，我们的重点在于利用扩散模型来检测和分析这些攻击对图像带来的异常。为此，我们使用扩散模型系统地考察了对抗性例子在经历转换过程时的分布的一致性。在CIFAR-10和ImageNet数据集上评估了这种方法的有效性，包括在后者中不同的图像大小。实验结果表明，该方法能够有效地区分良性图像和被攻击图像，提供了令人信服的证据，表明敌意实例与学习到的DM流形并不一致。



## **47. Unfolding Local Growth Rate Estimates for (Almost) Perfect Adversarial Detection**

关于(几乎)完美敌意检测的局部增长率估计 cs.CV

accepted at VISAPP23

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2212.06776v5) [paper-pdf](http://arxiv.org/pdf/2212.06776v5)

**Authors**: Peter Lorenz, Margret Keuper, Janis Keuper

**Abstract**: Convolutional neural networks (CNN) define the state-of-the-art solution on many perceptual tasks. However, current CNN approaches largely remain vulnerable against adversarial perturbations of the input that have been crafted specifically to fool the system while being quasi-imperceptible to the human eye. In recent years, various approaches have been proposed to defend CNNs against such attacks, for example by model hardening or by adding explicit defence mechanisms. Thereby, a small "detector" is included in the network and trained on the binary classification task of distinguishing genuine data from data containing adversarial perturbations. In this work, we propose a simple and light-weight detector, which leverages recent findings on the relation between networks' local intrinsic dimensionality (LID) and adversarial attacks. Based on a re-interpretation of the LID measure and several simple adaptations, we surpass the state-of-the-art on adversarial detection by a significant margin and reach almost perfect results in terms of F1-score for several networks and datasets. Sources available at: https://github.com/adverML/multiLID

摘要: 卷积神经网络(CNN)定义了许多感知任务的最先进的解决方案。然而，目前的CNN方法在很大程度上仍然容易受到输入的对抗性扰动，这些扰动是专门为愚弄系统而设计的，而人眼几乎察觉不到。近年来，已经提出了各种方法来保护CNN免受此类攻击，例如通过模型硬化或通过添加显式防御机制。因此，在网络中包括一个小的“检测器”，并在区分真实数据和包含对抗性扰动的数据的二进制分类任务上进行训练。在这项工作中，我们提出了一个简单而轻量级的检测器，它利用了最近关于网络的局部固有维度(LID)与对手攻击之间关系的研究结果。基于对LID度量的重新解释和几个简单的适应，我们在对手检测方面远远超过了最先进的水平，并在几个网络和数据集的F1得分方面取得了几乎完美的结果。资料来源：https://github.com/adverML/multiLID



## **48. Protect and Extend -- Using GANs for Synthetic Data Generation of Time-Series Medical Records**

保护与延伸--使用GANS生成时间序列病历的合成数据 cs.LG

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2402.14042v2) [paper-pdf](http://arxiv.org/pdf/2402.14042v2)

**Authors**: Navid Ashrafi, Vera Schmitt, Robert P. Spang, Sebastian Möller, Jan-Niklas Voigt-Antons

**Abstract**: Preservation of private user data is of paramount importance for high Quality of Experience (QoE) and acceptability, particularly with services treating sensitive data, such as IT-based health services. Whereas anonymization techniques were shown to be prone to data re-identification, synthetic data generation has gradually replaced anonymization since it is relatively less time and resource-consuming and more robust to data leakage. Generative Adversarial Networks (GANs) have been used for generating synthetic datasets, especially GAN frameworks adhering to the differential privacy phenomena. This research compares state-of-the-art GAN-based models for synthetic data generation to generate time-series synthetic medical records of dementia patients which can be distributed without privacy concerns. Predictive modeling, autocorrelation, and distribution analysis are used to assess the Quality of Generating (QoG) of the generated data. The privacy preservation of the respective models is assessed by applying membership inference attacks to determine potential data leakage risks. Our experiments indicate the superiority of the privacy-preserving GAN (PPGAN) model over other models regarding privacy preservation while maintaining an acceptable level of QoG. The presented results can support better data protection for medical use cases in the future.

摘要: 保存私人用户数据对于高质量体验和可接受性至关重要，特别是对于处理敏感数据的服务，如基于IT的医疗服务。由于匿名化技术被证明易于数据重新识别，合成数据生成因其相对较少的时间和资源消耗以及对数据泄露的健壮性而逐渐取代匿名化。生成性对抗网络(GANS)已被用于生成合成数据集，特别是遵循差异隐私现象的GAN框架。这项研究比较了最先进的基于GaN的合成数据生成模型，以生成痴呆症患者的时间序列合成病历，这些病历可以在没有隐私问题的情况下分发。预测建模、自相关和分布分析用于评估所生成数据的生成质量(QOG)。通过应用成员推理攻击来确定潜在的数据泄露风险，来评估各个模型的隐私保护。我们的实验表明，隐私保护GAN(PPGAN)模型在保持可接受的QOG水平的同时，在隐私保护方面优于其他模型。所给出的结果可以为未来医疗用例提供更好的数据保护。



## **49. Attacking Delay-based PUFs with Minimal Adversary Model**

利用最小对手模型攻击基于时延的PUF cs.CR

13 pages, 6 figures, journal

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00464v1) [paper-pdf](http://arxiv.org/pdf/2403.00464v1)

**Authors**: Hongming Fei, Owen Millwood, Prosanta Gope, Jack Miskelly, Biplab Sikdar

**Abstract**: Physically Unclonable Functions (PUFs) provide a streamlined solution for lightweight device authentication. Delay-based Arbiter PUFs, with their ease of implementation and vast challenge space, have received significant attention; however, they are not immune to modelling attacks that exploit correlations between their inputs and outputs. Research is therefore polarized between developing modelling-resistant PUFs and devising machine learning attacks against them. This dichotomy often results in exaggerated concerns and overconfidence in PUF security, primarily because there lacks a universal tool to gauge a PUF's security. In many scenarios, attacks require additional information, such as PUF type or configuration parameters. Alarmingly, new PUFs are often branded `secure' if they lack a specific attack model upon introduction. To impartially assess the security of delay-based PUFs, we present a generic framework featuring a Mixture-of-PUF-Experts (MoPE) structure for mounting attacks on various PUFs with minimal adversarial knowledge, which provides a way to compare their performance fairly and impartially. We demonstrate the capability of our model to attack different PUF types, including the first successful attack on Heterogeneous Feed-Forward PUFs using only a reasonable amount of challenges and responses. We propose an extension version of our model, a Multi-gate Mixture-of-PUF-Experts (MMoPE) structure, facilitating multi-task learning across diverse PUFs to recognise commonalities across PUF designs. This allows a streamlining of training periods for attacking multiple PUFs simultaneously. We conclude by showcasing the potent performance of MoPE and MMoPE across a spectrum of PUF types, employing simulated, real-world unbiased, and biased data sets for analysis.

摘要: 物理不可克隆功能（PUF）为轻量级设备身份验证提供了简化的解决方案。基于延迟的Arbiter PUF由于其易于实现和巨大的挑战空间而受到了极大的关注;然而，它们并不能免受利用其输入和输出之间的相关性的建模攻击。因此，研究在开发抗建模的PUF和设计针对它们的机器学习攻击之间两极分化。这种二分法通常导致对PUF安全性的过度关注和过度自信，主要是因为缺乏通用工具来衡量PUF的安全性。在许多情况下，攻击需要额外的信息，如PUF类型或配置参数。令人震惊的是，如果新的PUF在推出时缺乏特定的攻击模式，它们往往被贴上“安全”的标签。为了公正地评估基于延迟的PUF的安全性，我们提出了一个通用框架，该框架具有混合PUF专家（MoPE）结构，用于以最少的对抗知识对各种PUF进行攻击，这提供了一种公平公正地比较其性能的方法。我们展示了我们的模型攻击不同PUF类型的能力，包括仅使用合理数量的挑战和响应首次成功攻击异构前馈PUF。我们提出了我们的模型的扩展版本，一个多门混合PUF专家（MMoPE）结构，促进多任务学习不同的PUF识别PUF设计的共性。这允许同时攻击多个PUF的训练周期的流线化。最后，我们展示了MoPE和MMoPE在一系列PUF类型中的强大性能，采用模拟的，真实世界的无偏和有偏数据集进行分析。



## **50. Robust Deep Reinforcement Learning Through Adversarial Attacks and Training : A Survey**

基于对抗性攻击和训练的稳健深度强化学习研究综述 cs.LG

57 pages, 16 figues, 2 tables

**SubmitDate**: 2024-03-01    [abs](http://arxiv.org/abs/2403.00420v1) [paper-pdf](http://arxiv.org/pdf/2403.00420v1)

**Authors**: Lucas Schott, Josephine Delas, Hatem Hajri, Elies Gherbi, Reda Yaich, Nora Boulahia-Cuppens, Frederic Cuppens, Sylvain Lamprier

**Abstract**: Deep Reinforcement Learning (DRL) is an approach for training autonomous agents across various complex environments. Despite its significant performance in well known environments, it remains susceptible to minor conditions variations, raising concerns about its reliability in real-world applications. To improve usability, DRL must demonstrate trustworthiness and robustness. A way to improve robustness of DRL to unknown changes in the conditions is through Adversarial Training, by training the agent against well suited adversarial attacks on the dynamics of the environment. Addressing this critical issue, our work presents an in-depth analysis of contemporary adversarial attack methodologies, systematically categorizing them and comparing their objectives and operational mechanisms. This classification offers a detailed insight into how adversarial attacks effectively act for evaluating the resilience of DRL agents, thereby paving the way for enhancing their robustness.

摘要: 深度强化学习(DRL)是一种在各种复杂环境中训练自主智能体的方法。尽管它在众所周知的环境中表现出色，但它仍然容易受到微小条件变化的影响，这引发了人们对其在现实世界应用中的可靠性的担忧。为了提高可用性，DRL必须证明可信性和健壮性。提高DRL对未知条件变化的稳健性的一种方法是通过对抗性训练，通过训练代理抵御对环境动态的很好的对抗性攻击。针对这一关键问题，我们的工作对当代对抗攻击方法进行了深入分析，系统地对它们进行了分类，并比较了它们的目标和运行机制。这一分类提供了对抗性攻击如何有效地用于评估DRL代理的弹性的详细洞察，从而为增强其健壮性铺平了道路。



