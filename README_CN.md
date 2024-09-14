# Latest Adversarial Attack Papers
**update at 2024-09-14 09:33:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. LoRID: Low-Rank Iterative Diffusion for Adversarial Purification**

LoDID：对抗净化的低等级迭代扩散 cs.LG

LA-UR-24-28834

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08255v1) [paper-pdf](http://arxiv.org/pdf/2409.08255v1)

**Authors**: Geigh Zollicoffer, Minh Vu, Ben Nebgen, Juan Castorena, Boian Alexandrov, Manish Bhattarai

**Abstract**: This work presents an information-theoretic examination of diffusion-based purification methods, the state-of-the-art adversarial defenses that utilize diffusion models to remove malicious perturbations in adversarial examples. By theoretically characterizing the inherent purification errors associated with the Markov-based diffusion purifications, we introduce LoRID, a novel Low-Rank Iterative Diffusion purification method designed to remove adversarial perturbation with low intrinsic purification errors. LoRID centers around a multi-stage purification process that leverages multiple rounds of diffusion-denoising loops at the early time-steps of the diffusion models, and the integration of Tucker decomposition, an extension of matrix factorization, to remove adversarial noise at high-noise regimes. Consequently, LoRID increases the effective diffusion time-steps and overcomes strong adversarial attacks, achieving superior robustness performance in CIFAR-10/100, CelebA-HQ, and ImageNet datasets under both white-box and black-box settings.

摘要: 这项工作提出了一种基于扩散的净化方法的信息论检验，这种方法是一种最先进的对抗防御方法，它利用扩散模型来消除对抗例子中的恶意扰动。通过对基于马尔可夫扩散净化的固有净化误差进行理论分析，提出了一种新的低阶迭代扩散净化方法LoRID，旨在以较低的固有净化误差去除对抗性扰动。LoRID以多阶段净化过程为中心，在扩散模型的早期步骤利用多轮扩散去噪循环，并整合Tucker分解(矩阵因式分解的扩展)，以在高噪声区域消除对抗性噪声。因此，LoRID增加了有效的扩散时间步长，并克服了强大的对手攻击，在CIFAR-10/100、CelebA-HQ和ImageNet数据集上实现了在白盒和黑盒设置下的卓越鲁棒性性能。



## **2. High-Frequency Anti-DreamBooth: Robust Defense Against Image Synthesis**

高频反DreamBooth：针对图像合成的强大防御 cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08167v1) [paper-pdf](http://arxiv.org/pdf/2409.08167v1)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.

摘要: 最近，文本到图像的生成模型被滥用来创建未经授权的恶意个人图像，造成了日益严重的社会问题。以前的解决方案（例如Anti-DreamBooth）会向图像添加对抗性噪音，以保护它们不被用作恶意生成的训练数据。然而，我们发现对抗性噪音可以通过迪夫Pure等对抗性净化方法去除。因此，我们提出了一种新的对抗性攻击方法，该方法在图像的高频区域添加强扰动，使其对对抗性净化更稳健。我们的实验表明，即使在对抗净化之后，对抗图像也会保留噪音，从而阻碍了恶意图像的生成。



## **3. Unleashing Worms and Extracting Data: Escalating the Outcome of Attacks against RAG-based Inference in Scale and Severity Using Jailbreaking**

释放蠕虫和提取数据：使用越狱从规模和严重性上升级针对基于RAG的推理的攻击结果 cs.CR

for Github, see  https://github.com/StavC/UnleashingWorms-ExtractingData

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.08045v1) [paper-pdf](http://arxiv.org/pdf/2409.08045v1)

**Authors**: Stav Cohen, Ron Bitton, Ben Nassi

**Abstract**: In this paper, we show that with the ability to jailbreak a GenAI model, attackers can escalate the outcome of attacks against RAG-based GenAI-powered applications in severity and scale. In the first part of the paper, we show that attackers can escalate RAG membership inference attacks and RAG entity extraction attacks to RAG documents extraction attacks, forcing a more severe outcome compared to existing attacks. We evaluate the results obtained from three extraction methods, the influence of the type and the size of five embeddings algorithms employed, the size of the provided context, and the GenAI engine. We show that attackers can extract 80%-99.8% of the data stored in the database used by the RAG of a Q&A chatbot. In the second part of the paper, we show that attackers can escalate the scale of RAG data poisoning attacks from compromising a single GenAI-powered application to compromising the entire GenAI ecosystem, forcing a greater scale of damage. This is done by crafting an adversarial self-replicating prompt that triggers a chain reaction of a computer worm within the ecosystem and forces each affected application to perform a malicious activity and compromise the RAG of additional applications. We evaluate the performance of the worm in creating a chain of confidential data extraction about users within a GenAI ecosystem of GenAI-powered email assistants and analyze how the performance of the worm is affected by the size of the context, the adversarial self-replicating prompt used, the type and size of the embeddings algorithm employed, and the number of hops in the propagation. Finally, we review and analyze guardrails to protect RAG-based inference and discuss the tradeoffs.

摘要: 在本文中，我们展示了通过越狱GenAI模型的能力，攻击者可以在严重性和规模上升级对基于RAG的GenAI支持的应用程序的攻击结果。在论文的第一部分，我们证明了攻击者可以将RAG成员关系推理攻击和RAG实体提取攻击升级为RAG文档提取攻击，从而迫使出现比现有攻击更严重的后果。我们评估了三种提取方法获得的结果，所采用的五种嵌入算法的类型和大小、所提供的上下文的大小以及GenAI引擎的影响。我们表明，攻击者可以提取80%-99.8%的数据存储在数据库中的问答聊天机器人使用的RAG。在本文的第二部分中，我们展示了攻击者可以将RAG数据中毒攻击的规模从危害单个GenAI支持的应用程序升级到危害整个GenAI生态系统，从而迫使更大规模的破坏。这是通过精心编制一个敌意的自我复制提示来实现的，该提示会在生态系统中触发计算机蠕虫的连锁反应，并迫使每个受影响的应用程序执行恶意活动，并危及其他应用程序的安全。我们评估了蠕虫在由GenAI支持的电子邮件助手组成的GenAI生态系统中创建关于用户的机密数据提取链的性能，并分析了上下文大小、使用的敌意自复制提示、所使用的嵌入算法的类型和大小以及传播中的跳数对蠕虫性能的影响。最后，我们回顾和分析了保护基于RAG的推理的护栏，并讨论了其权衡。



## **4. Detecting and Defending Against Adversarial Attacks on Automatic Speech Recognition via Diffusion Models**

利用扩散模型检测和防御自动语音识别中的对抗攻击 eess.AS

Under review at ICASSP 2025

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07936v1) [paper-pdf](http://arxiv.org/pdf/2409.07936v1)

**Authors**: Nikolai L. Kühne, Astrid H. F. Kitchen, Marie S. Jensen, Mikkel S. L. Brøndt, Martin Gonzalez, Christophe Biscio, Zheng-Hua Tan

**Abstract**: Automatic speech recognition (ASR) systems are known to be vulnerable to adversarial attacks. This paper addresses detection and defence against targeted white-box attacks on speech signals for ASR systems. While existing work has utilised diffusion models (DMs) to purify adversarial examples, achieving state-of-the-art results in keyword spotting tasks, their effectiveness for more complex tasks such as sentence-level ASR remains unexplored. Additionally, the impact of the number of forward diffusion steps on performance is not well understood. In this paper, we systematically investigate the use of DMs for defending against adversarial attacks on sentences and examine the effect of varying forward diffusion steps. Through comprehensive experiments on the Mozilla Common Voice dataset, we demonstrate that two forward diffusion steps can completely defend against adversarial attacks on sentences. Moreover, we introduce a novel, training-free approach for detecting adversarial attacks by leveraging a pre-trained DM. Our experimental results show that this method can detect adversarial attacks with high accuracy.

摘要: 众所周知，自动语音识别(ASR)系统容易受到对手攻击。本文研究了ASR系统中语音信号白盒攻击的检测和防御。虽然现有的工作已经利用扩散模型(DM)来净化对抗性例子，在关键字识别任务中取得了最先进的结果，但它们对更复杂任务(如句子级ASR)的有效性仍未被探索。此外，前向扩散步数对性能的影响还不是很清楚。在这篇文章中，我们系统地研究了DMS在防御对抗性句子攻击中的应用，并考察了不同的前向扩散步骤的效果。通过在Mozilla公共语音数据集上的综合实验，我们证明了两个前向扩散步骤可以完全防御针对句子的敌意攻击。此外，我们引入了一种新的、无需训练的方法来利用预先训练的DM来检测对抗性攻击。实验结果表明，该方法具有较高的检测准确率。



## **5. What Matters to Enhance Traffic Rule Compliance of Imitation Learning for End-to-End Autonomous Driving**

增强端到端自动驾驶模仿学习的交通规则合规性的重要性 cs.CV

14 pages, 3 figures

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2309.07808v3) [paper-pdf](http://arxiv.org/pdf/2309.07808v3)

**Authors**: Hongkuan Zhou, Wei Cao, Aifen Sui, Zhenshan Bing

**Abstract**: End-to-end autonomous driving, where the entire driving pipeline is replaced with a single neural network, has recently gained research attention because of its simpler structure and faster inference time. Despite this appealing approach largely reducing the complexity in the driving pipeline, it also leads to safety issues because the trained policy is not always compliant with the traffic rules. In this paper, we proposed P-CSG, a penalty-based imitation learning approach with contrastive-based cross semantics generation sensor fusion technologies to increase the overall performance of end-to-end autonomous driving. In this method, we introduce three penalties - red light, stop sign, and curvature speed penalty to make the agent more sensitive to traffic rules. The proposed cross semantics generation helps to align the shared information of different input modalities. We assessed our model's performance using the CARLA Leaderboard - Town 05 Long Benchmark and Longest6 Benchmark, achieving 8.5% and 2.0% driving score improvement compared to the baselines. Furthermore, we conducted robustness evaluations against adversarial attacks like FGSM and Dot attacks, revealing a substantial increase in robustness compared to other baseline models. More detailed information can be found at https://hk-zh.github.io/p-csg-plus.

摘要: 端到端自动驾驶，即整个驾驶管道被单一的神经网络取代，由于其结构更简单，推理时间更快，最近得到了研究人员的关注。尽管这种吸引人的方法极大地降低了驾驶管道的复杂性，但它也导致了安全问题，因为经过培训的政策并不总是符合交通规则。为了提高端到端自主驾驶的整体性能，本文提出了一种基于惩罚的模仿学习方法P-CSG，并结合基于对比的交叉语义生成传感器融合技术。在该方法中，我们引入了三种惩罚--红灯、停车标志和曲率速度惩罚，以使智能体对交通规则更加敏感。提出的交叉语义生成有助于对齐不同输入通道的共享信息。我们使用Carla Leaderboard-town 05 Long基准和Longest6基准评估了我们的模型的性能，与基准相比，驾驶分数分别提高了8.5%和2.0%。此外，我们对FGSM和Dot攻击等对手攻击进行了健壮性评估，显示出与其他基线模型相比，健壮性有了显著的提高。欲了解更多详细信息，请访问https://hk-zh.github.io/p-csg-plus.。



## **6. A Spatiotemporal Stealthy Backdoor Attack against Cooperative Multi-Agent Deep Reinforcement Learning**

针对协作多智能体深度强化学习的时空隐形后门攻击 cs.AI

6 pages, IEEE Globecom 2024

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07775v1) [paper-pdf](http://arxiv.org/pdf/2409.07775v1)

**Authors**: Yinbo Yu, Saihao Yan, Jiajia Liu

**Abstract**: Recent studies have shown that cooperative multi-agent deep reinforcement learning (c-MADRL) is under the threat of backdoor attacks. Once a backdoor trigger is observed, it will perform abnormal actions leading to failures or malicious goals. However, existing proposed backdoors suffer from several issues, e.g., fixed visual trigger patterns lack stealthiness, the backdoor is trained or activated by an additional network, or all agents are backdoored. To this end, in this paper, we propose a novel backdoor attack against c-MADRL, which attacks the entire multi-agent team by embedding the backdoor only in a single agent. Firstly, we introduce adversary spatiotemporal behavior patterns as the backdoor trigger rather than manual-injected fixed visual patterns or instant status and control the attack duration. This method can guarantee the stealthiness and practicality of injected backdoors. Secondly, we hack the original reward function of the backdoored agent via reward reverse and unilateral guidance during training to ensure its adverse influence on the entire team. We evaluate our backdoor attacks on two classic c-MADRL algorithms VDN and QMIX, in a popular c-MADRL environment SMAC. The experimental results demonstrate that our backdoor attacks are able to reach a high attack success rate (91.6\%) while maintaining a low clean performance variance rate (3.7\%).

摘要: 最近的研究表明，协作多智能体深度强化学习(c-MADRL)受到后门攻击的威胁。一旦观察到后门触发器，它将执行导致失败或恶意目标的异常操作。然而，现有的拟议后门受到几个问题的困扰，例如，固定的视觉触发模式缺乏隐蔽性，后门被额外的网络训练或激活，或者所有代理都被后门。为此，本文提出了一种新的针对c-MADRL的后门攻击，通过在单个代理中嵌入后门来攻击整个多代理团队。首先，引入敌方时空行为模式作为后门触发，而不是人工注入固定的视觉模式或即时状态，并控制攻击持续时间。该方法可以保证注入后门的隐蔽性和实用性。其次，在训练过程中，通过奖励反转和单边指导，破解背靠背代理人原有的奖励功能，以确保其对整个团队的不利影响。我们在一个流行的c-MADRL环境SMAC中评估了我们对两个经典c-MADRL算法VDN和QMIX的后门攻击。实验结果表明，我们的后门攻击能够在保持较低的干净性能变异率(3.7)的同时达到较高的攻击成功率(91.6)。



## **7. Attack End-to-End Autonomous Driving through Module-Wise Noise**

通过模块化噪音攻击端到端自动驾驶 cs.LG

**SubmitDate**: 2024-09-12    [abs](http://arxiv.org/abs/2409.07706v1) [paper-pdf](http://arxiv.org/pdf/2409.07706v1)

**Authors**: Lu Wang, Tianyuan Zhang, Yikai Han, Muyang Fang, Ting Jin, Jiaqi Kang

**Abstract**: With recent breakthroughs in deep neural networks, numerous tasks within autonomous driving have exhibited remarkable performance. However, deep learning models are susceptible to adversarial attacks, presenting significant security risks to autonomous driving systems. Presently, end-to-end architectures have emerged as the predominant solution for autonomous driving, owing to their collaborative nature across different tasks. Yet, the implications of adversarial attacks on such models remain relatively unexplored. In this paper, we conduct comprehensive adversarial security research on the modular end-to-end autonomous driving model for the first time. We thoroughly consider the potential vulnerabilities in the model inference process and design a universal attack scheme through module-wise noise injection. We conduct large-scale experiments on the full-stack autonomous driving model and demonstrate that our attack method outperforms previous attack methods. We trust that our research will offer fresh insights into ensuring the safety and reliability of autonomous driving systems.

摘要: 随着最近深度神经网络的突破，自动驾驶中的许多任务表现出了显著的性能。然而，深度学习模型容易受到对抗性攻击，给自动驾驶系统带来了巨大的安全风险。目前，端到端架构已经成为自动驾驶的主要解决方案，因为它们跨不同任务的协作性质。然而，对抗性攻击对这类模型的影响仍相对未被探索。本文首次对模块化端到端自主驾驶模型进行了全面的对抗性安全研究。我们充分考虑了模型推理过程中的潜在漏洞，并通过模块化噪声注入设计了一种通用的攻击方案。我们在全栈自主驾驶模型上进行了大规模的实验，证明了我们的攻击方法比以前的攻击方法要好。我们相信，我们的研究将为确保自动驾驶系统的安全和可靠性提供新的见解。



## **8. A Training Rate and Survival Heuristic for Inference and Robustness Evaluation (TRASHFIRE)**

推理和稳健性评估的训练率和生存启发式（TRASHFIRE） cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2401.13751v2) [paper-pdf](http://arxiv.org/pdf/2401.13751v2)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Machine learning models -- deep neural networks in particular -- have performed remarkably well on benchmark datasets across a wide variety of domains. However, the ease of finding adversarial counter-examples remains a persistent problem when training times are measured in hours or days and the time needed to find a successful adversarial counter-example is measured in seconds. Much work has gone into generating and defending against these adversarial counter-examples, however the relative costs of attacks and defences are rarely discussed. Additionally, machine learning research is almost entirely guided by test/train metrics, but these would require billions of samples to meet industry standards. The present work addresses the problem of understanding and predicting how particular model hyper-parameters influence the performance of a model in the presence of an adversary. The proposed approach uses survival models, worst-case examples, and a cost-aware analysis to precisely and accurately reject a particular model change during routine model training procedures rather than relying on real-world deployment, expensive formal verification methods, or accurate simulations of very complicated systems (\textit{e.g.}, digitally recreating every part of a car or a plane). Through an evaluation of many pre-processing techniques, adversarial counter-examples, and neural network configurations, the conclusion is that deeper models do offer marginal gains in survival times compared to more shallow counterparts. However, we show that those gains are driven more by the model inference time than inherent robustness properties. Using the proposed methodology, we show that ResNet is hopelessly insecure against even the simplest of white box attacks.

摘要: 机器学习模型--尤其是深度神经网络--在各种领域的基准数据集上表现得非常好。然而，当训练时间以小时或天衡量，而找到成功的对抗性反例所需的时间以秒衡量时，寻找对抗性反例的容易程度仍然是一个持久的问题。在生成和防御这些对抗性反例方面已经做了很多工作，然而攻击和防御的相对成本很少被讨论。此外，机器学习研究几乎完全由测试/训练指标指导，但这些指标需要数十亿个样本才能满足行业标准。目前的工作解决的问题是理解和预测特定的模型超参数如何在对手存在的情况下影响模型的性能。该方法使用生存模型、最坏情况示例和成本意识分析，在常规模型训练过程中准确和准确地拒绝特定的模型更改，而不是依赖于真实世界的部署、昂贵的形式验证方法或非常复杂的系统的准确模拟(例如，以数字方式重建汽车或飞机的每个部件)。通过对许多预处理技术、对抗性反例和神经网络配置的评估，结论是，与较浅的模型相比，较深的模型确实提供了生存时间的边际收益。然而，我们表明，这些收益更多地是由模型推理时间驱动的，而不是固有的稳健性。使用提出的方法，我们表明ResNet对于即使是最简单的白盒攻击也是无可救药的不安全的。



## **9. A Cost-Aware Approach to Adversarial Robustness in Neural Networks**

神经网络中对抗鲁棒性的一种具有成本意识的方法 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07609v1) [paper-pdf](http://arxiv.org/pdf/2409.07609v1)

**Authors**: Charles Meyers, Mohammad Reza Saleh Sedghpour, Tommy Löfstedt, Erik Elmroth

**Abstract**: Considering the growing prominence of production-level AI and the threat of adversarial attacks that can evade a model at run-time, evaluating the robustness of models to these evasion attacks is of critical importance. Additionally, testing model changes likely means deploying the models to (e.g. a car or a medical imaging device), or a drone to see how it affects performance, making un-tested changes a public problem that reduces development speed, increases cost of development, and makes it difficult (if not impossible) to parse cause from effect. In this work, we used survival analysis as a cloud-native, time-efficient and precise method for predicting model performance in the presence of adversarial noise. For neural networks in particular, the relationships between the learning rate, batch size, training time, convergence time, and deployment cost are highly complex, so researchers generally rely on benchmark datasets to assess the ability of a model to generalize beyond the training data. To address this, we propose using accelerated failure time models to measure the effect of hardware choice, batch size, number of epochs, and test-set accuracy by using adversarial attacks to induce failures on a reference model architecture before deploying the model to the real world. We evaluate several GPU types and use the Tree Parzen Estimator to maximize model robustness and minimize model run-time simultaneously. This provides a way to evaluate the model and optimise it in a single step, while simultaneously allowing us to model the effect of model parameters on training time, prediction time, and accuracy. Using this technique, we demonstrate that newer, more-powerful hardware does decrease the training time, but with a monetary and power cost that far outpaces the marginal gains in accuracy.

摘要: 考虑到产生级人工智能的日益突出以及运行时可以逃避模型的对抗性攻击的威胁，评估模型对这些逃避攻击的稳健性至关重要。此外，测试模型更改可能意味着将模型部署到(例如，汽车或医疗成像设备)或无人机，以了解它如何影响性能，使未经测试的更改成为一个公共问题，从而降低开发速度、增加开发成本，并使分析原因和结果变得困难(如果不是不可能的话)。在这项工作中，我们使用生存分析作为一种云本地的、时间高效和精确的方法来预测存在对抗性噪声存在的模型性能。尤其对于神经网络，学习速度、批次大小、训练时间、收敛时间和部署成本之间的关系非常复杂，因此研究人员通常依赖基准数据集来评估模型在训练数据之外的泛化能力。为了解决这个问题，我们建议在将参考模型部署到现实世界之前，使用加速故障时间模型来衡量硬件选择、批量大小、历元数和测试集精度的影响，方法是使用对抗性攻击在参考模型体系结构上诱导故障。我们评估了几种类型的GPU，并使用Tree Parzen Estimator来最大化模型的健壮性，同时最小化模型的运行时间。这提供了一种在单个步骤中评估和优化模型的方法，同时允许我们对模型参数对训练时间、预测时间和精度的影响进行建模。使用这项技术，我们证明了更新的、功能更强大的硬件确实减少了训练时间，但在金钱和电力成本上远远超过了精度的边际收益。



## **10. Resilient Graph Neural Networks: A Coupled Dynamical Systems Approach**

弹性图神经网络：一种耦合动态系统方法 cs.LG

ECAI 2024

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2311.06942v3) [paper-pdf](http://arxiv.org/pdf/2311.06942v3)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of coupled dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

摘要: 图形神经网络(GNN)已经成为解决各种基于图形的任务的关键组件。尽管GNN取得了显著的成功，但它们仍然容易受到对抗性攻击形式的投入扰动的影响。本文介绍了一种通过耦合动力系统的透镜来增强GNN抵抗敌意扰动的创新方法。我们的方法引入了基于具有压缩性质的微分方程的图神经层，从而提高了GNN的稳健性。该方法的一个显著特点是节点特征和邻接矩阵的同时学习进化，从而内在地增强了模型对输入特征扰动和图的连通性的稳健性。我们从数学上推导出我们的新体系结构的基础，并提供理论见解来推理其预期行为。我们通过许多真实世界的基准测试来证明我们的方法的有效性，与现有的方法相比，我们的阅读是平分的，或者是性能有所提高。



## **11. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

引入扰动能力评分（PS）增强ML-NIDS对抗规避攻击的鲁棒性 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07448v1) [paper-pdf](http://arxiv.org/pdf/2409.07448v1)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: This paper proposes a novel Perturb-ability Score (PS) that can be used to identify Network Intrusion Detection Systems (NIDS) features that can be easily manipulated by attackers in the problem-space. We demonstrate that using PS to select only non-perturb-able features for ML-based NIDS maintains detection performance while enhancing robustness against adversarial attacks.

摘要: 本文提出了一种新颖的扰动能力评分（PS），可用于识别攻击者在问题空间中容易操纵的网络入侵检测系统（NIDS）特征。我们证明，使用PS仅为基于ML的NIDS选择不可扰动的特征可以保持检测性能，同时增强针对对抗性攻击的鲁棒性。



## **12. Enhancing adversarial robustness in Natural Language Inference using explanations**

使用解释增强自然语言推理中的对抗稳健性 cs.CL

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07423v1) [paper-pdf](http://arxiv.org/pdf/2409.07423v1)

**Authors**: Alexandros Koulakos, Maria Lymperaiou, Giorgos Filandrianos, Giorgos Stamou

**Abstract**: The surge of state-of-the-art Transformer-based models has undoubtedly pushed the limits of NLP model performance, excelling in a variety of tasks. We cast the spotlight on the underexplored task of Natural Language Inference (NLI), since models trained on popular well-suited datasets are susceptible to adversarial attacks, allowing subtle input interventions to mislead the model. In this work, we validate the usage of natural language explanation as a model-agnostic defence strategy through extensive experimentation: only by fine-tuning a classifier on the explanation rather than premise-hypothesis inputs, robustness under various adversarial attacks is achieved in comparison to explanation-free baselines. Moreover, since there is no standard strategy of testing the semantic validity of the generated explanations, we research the correlation of widely used language generation metrics with human perception, in order for them to serve as a proxy towards robust NLI models. Our approach is resource-efficient and reproducible without significant computational limitations.

摘要: 最先进的基于变形金刚的模型的激增无疑已经突破了NLP模型的性能极限，在各种任务中表现出色。我们将注意力集中在自然语言推理(NLI)这一未被探索的任务上，因为在流行的匹配良好的数据集上训练的模型容易受到对抗性攻击，允许微妙的输入干预误导模型。在这项工作中，我们通过广泛的实验验证了自然语言解释作为一种与模型无关的防御策略的使用：只有通过微调解释而不是前提假设输入的分类器，才能实现与无解释基线相比在各种对手攻击下的健壮性。此外，由于没有标准的策略来测试生成的解释的语义有效性，我们研究了广泛使用的语言生成度量与人类感知的相关性，以便它们能够作为稳健的NLI模型的代理。我们的方法是资源高效和可重复性的，没有明显的计算限制。



## **13. SoK: Security and Privacy Risks of Medical AI**

SoK：医疗人工智能的安全和隐私风险 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07415v1) [paper-pdf](http://arxiv.org/pdf/2409.07415v1)

**Authors**: Yuanhaur Chang, Han Liu, Evin Jaff, Chenyang Lu, Ning Zhang

**Abstract**: The integration of technology and healthcare has ushered in a new era where software systems, powered by artificial intelligence and machine learning, have become essential components of medical products and services. While these advancements hold great promise for enhancing patient care and healthcare delivery efficiency, they also expose sensitive medical data and system integrity to potential cyberattacks. This paper explores the security and privacy threats posed by AI/ML applications in healthcare. Through a thorough examination of existing research across a range of medical domains, we have identified significant gaps in understanding the adversarial attacks targeting medical AI systems. By outlining specific adversarial threat models for medical settings and identifying vulnerable application domains, we lay the groundwork for future research that investigates the security and resilience of AI-driven medical systems. Through our analysis of different threat models and feasibility studies on adversarial attacks in different medical domains, we provide compelling insights into the pressing need for cybersecurity research in the rapidly evolving field of AI healthcare technology.

摘要: 技术与医疗的融合开启了一个新时代，以人工智能和机器学习为动力的软件系统已成为医疗产品和服务的重要组成部分。虽然这些进步为提高患者护理和医疗保健提供效率带来了巨大的希望，但它们也使敏感的医疗数据和系统完整性面临潜在的网络攻击。本文探讨了医疗保健中AI/ML应用程序所带来的安全和隐私威胁。通过对一系列医学领域现有研究的彻底检查，我们发现了在理解针对医疗人工智能系统的对抗性攻击方面存在的重大差距。通过概述医疗环境的特定对抗性威胁模型并识别易受攻击的应用领域，我们为未来调查人工智能驱动的医疗系统的安全性和弹性的研究奠定了基础。通过我们对不同威胁模型的分析和对不同医疗领域对抗性攻击的可行性研究，我们对快速发展的人工智能医疗技术领域对网络安全研究的迫切需求提供了令人信服的见解。



## **14. D-CAPTCHA++: A Study of Resilience of Deepfake CAPTCHA under Transferable Imperceptible Adversarial Attack**

D-CAPTCHA++：Deepfake CAPTCHA在可转移不可感知对抗攻击下的弹性研究 cs.CR

14 pages

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07390v1) [paper-pdf](http://arxiv.org/pdf/2409.07390v1)

**Authors**: Hong-Hanh Nguyen-Le, Van-Tuan Tran, Dinh-Thuc Nguyen, Nhien-An Le-Khac

**Abstract**: The advancements in generative AI have enabled the improvement of audio synthesis models, including text-to-speech and voice conversion. This raises concerns about its potential misuse in social manipulation and political interference, as synthetic speech has become indistinguishable from natural human speech. Several speech-generation programs are utilized for malicious purposes, especially impersonating individuals through phone calls. Therefore, detecting fake audio is crucial to maintain social security and safeguard the integrity of information. Recent research has proposed a D-CAPTCHA system based on the challenge-response protocol to differentiate fake phone calls from real ones. In this work, we study the resilience of this system and introduce a more robust version, D-CAPTCHA++, to defend against fake calls. Specifically, we first expose the vulnerability of the D-CAPTCHA system under transferable imperceptible adversarial attack. Secondly, we mitigate such vulnerability by improving the robustness of the system by using adversarial training in D-CAPTCHA deepfake detectors and task classifiers.

摘要: 生成性人工智能的进步使音频合成模型得以改进，包括文本到语音和语音转换。这引发了人们对其在社会操纵和政治干预中可能被滥用的担忧，因为合成语音已变得与自然人类语音难以区分。几个语音生成程序被用于恶意目的，特别是通过电话冒充个人。因此，检测虚假音频对于维护社会安全、维护信息完整性至关重要。最近的研究提出了一种基于挑战-响应协议的D-CAPTCHA系统来区分虚假电话和真实电话。在这项工作中，我们研究了该系统的弹性，并引入了一个更健壮的版本，D-CAPTCHA++，以防御虚假呼叫。具体地说，我们首先暴露了D-CAPTCHA系统在可转移的不可察觉的对手攻击下的脆弱性。其次，我们通过在D-CAPTCHA深度伪检测器和任务分类器中使用对抗性训练来提高系统的健壮性，从而缓解了这种脆弱性。



## **15. Securing Vision-Language Models with a Robust Encoder Against Jailbreak and Adversarial Attacks**

使用稳健的编码器保护视觉语言模型免受越狱和对抗攻击 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07353v1) [paper-pdf](http://arxiv.org/pdf/2409.07353v1)

**Authors**: Md Zarif Hossain, Ahmed Imteaj

**Abstract**: Large Vision-Language Models (LVLMs), trained on multimodal big datasets, have significantly advanced AI by excelling in vision-language tasks. However, these models remain vulnerable to adversarial attacks, particularly jailbreak attacks, which bypass safety protocols and cause the model to generate misleading or harmful responses. This vulnerability stems from both the inherent susceptibilities of LLMs and the expanded attack surface introduced by the visual modality. We propose Sim-CLIP+, a novel defense mechanism that adversarially fine-tunes the CLIP vision encoder by leveraging a Siamese architecture. This approach maximizes cosine similarity between perturbed and clean samples, facilitating resilience against adversarial manipulations. Sim-CLIP+ offers a plug-and-play solution, allowing seamless integration into existing LVLM architectures as a robust vision encoder. Unlike previous defenses, our method requires no structural modifications to the LVLM and incurs minimal computational overhead. Sim-CLIP+ demonstrates effectiveness against both gradient-based adversarial attacks and various jailbreak techniques. We evaluate Sim-CLIP+ against three distinct jailbreak attack strategies and perform clean evaluations using standard downstream datasets, including COCO for image captioning and OKVQA for visual question answering. Extensive experiments demonstrate that Sim-CLIP+ maintains high clean accuracy while substantially improving robustness against both gradient-based adversarial attacks and jailbreak techniques. Our code and robust vision encoders are available at https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.

摘要: 在多模式大数据集上训练的大型视觉语言模型(LVLM)通过在视觉语言任务中脱颖而出，极大地促进了人工智能的发展。然而，这些模型仍然容易受到对抗性攻击，特别是越狱攻击，这些攻击绕过了安全协议，并导致模型生成误导性或有害的响应。该漏洞既源于LLMS固有的易感性，也源于视觉通道引入的扩展攻击面。我们提出了Sim-Clip+，这是一种新颖的防御机制，它利用暹罗体系结构对剪辑视觉编码器进行了相反的微调。这种方法最大限度地提高了扰动样本和干净样本之间的余弦相似性，促进了对对手操纵的弹性。SIM-Clip+提供了一种即插即用的解决方案，允许无缝集成到现有的LVLM架构中，作为一种强大的视觉编码器。与以前的防御方法不同，我们的方法不需要对LVLM进行结构修改，并且产生的计算开销最小。SIM-CLIP+展示了对抗基于梯度的对抗性攻击和各种越狱技术的有效性。我们针对三种不同的越狱攻击策略对Sim-Clip+进行了评估，并使用标准下游数据集进行了干净的评估，包括用于图像字幕的COCO和用于视觉问题回答的OKVQA。广泛的实验表明，Sim-Clip+保持了很高的干净准确率，同时显著提高了对基于梯度的对手攻击和越狱技术的稳健性。我们的代码和健壮的视觉编码器可在https://github.com/speedlab-git/Robust-Encoder-against-Jailbreak-attack.git.上获得



## **16. Module-wise Adaptive Adversarial Training for End-to-end Autonomous Driving**

端到端自动驾驶的模块自适应对抗训练 cs.CV

14 pages

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07321v1) [paper-pdf](http://arxiv.org/pdf/2409.07321v1)

**Authors**: Tianyuan Zhang, Lu Wang, Jiaqi Kang, Xinwei Zhang, Siyuan Liang, Yuwei Chen, Aishan Liu, Xianglong Liu

**Abstract**: Recent advances in deep learning have markedly improved autonomous driving (AD) models, particularly end-to-end systems that integrate perception, prediction, and planning stages, achieving state-of-the-art performance. However, these models remain vulnerable to adversarial attacks, where human-imperceptible perturbations can disrupt decision-making processes. While adversarial training is an effective method for enhancing model robustness against such attacks, no prior studies have focused on its application to end-to-end AD models. In this paper, we take the first step in adversarial training for end-to-end AD models and present a novel Module-wise Adaptive Adversarial Training (MA2T). However, extending conventional adversarial training to this context is highly non-trivial, as different stages within the model have distinct objectives and are strongly interconnected. To address these challenges, MA2T first introduces Module-wise Noise Injection, which injects noise before the input of different modules, targeting training models with the guidance of overall objectives rather than each independent module loss. Additionally, we introduce Dynamic Weight Accumulation Adaptation, which incorporates accumulated weight changes to adaptively learn and adjust the loss weights of each module based on their contributions (accumulated reduction rates) for better balance and robust training. To demonstrate the efficacy of our defense, we conduct extensive experiments on the widely-used nuScenes dataset across several end-to-end AD models under both white-box and black-box attacks, where our method outperforms other baselines by large margins (+5-10%). Moreover, we validate the robustness of our defense through closed-loop evaluation in the CARLA simulation environment, showing improved resilience even against natural corruption.

摘要: 深度学习的最新进展显著改进了自动驾驶(AD)模型，特别是集成了感知、预测和规划阶段的端到端系统，实现了最先进的性能。然而，这些模型仍然容易受到对抗性攻击，在这种攻击中，人类无法察觉的扰动可能会扰乱决策过程。虽然对抗性训练是增强模型对此类攻击的稳健性的有效方法，但以前的研究还没有将其应用于端到端的AD模型。在本文中，我们对端到端的AD模型进行了对抗性训练的第一步，并提出了一种新的模块化自适应对抗性训练(MA2T)。然而，将传统的对抗性训练扩展到这一背景下是非常重要的，因为该模式中的不同阶段有不同的目标，并且相互之间有很强的联系。为了应对这些挑战，MA2T首先引入了模块化噪声注入，即在不同模块输入之前注入噪声，在总体目标的指导下针对训练模型，而不是每个独立的模块损失。此外，我们引入了动态权重累积自适应，它结合累积权重变化来自适应地学习和调整每个模块的权重，基于它们的贡献(累积减少率)，以实现更好的平衡和稳健的训练。为了验证我们防御的有效性，我们在几个端到端的AD模型上对广泛使用的nuScenes数据集进行了广泛的实验，在白盒和黑盒攻击下，我们的方法的性能远远超过了其他基线(+5%-10%)。此外，我们通过在CALA仿真环境中的闭环评估来验证我们的防御的健壮性，表现出对自然腐败的更好的韧性。



## **17. Optimizing Neural Network Performance and Interpretability with Diophantine Equation Encoding**

利用丢番图方程编码优化神经网络性能和可解释性 cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07310v1) [paper-pdf](http://arxiv.org/pdf/2409.07310v1)

**Authors**: Ronald Katende

**Abstract**: This paper explores the integration of Diophantine equations into neural network (NN) architectures to improve model interpretability, stability, and efficiency. By encoding and decoding neural network parameters as integer solutions to Diophantine equations, we introduce a novel approach that enhances both the precision and robustness of deep learning models. Our method integrates a custom loss function that enforces Diophantine constraints during training, leading to better generalization, reduced error bounds, and enhanced resilience against adversarial attacks. We demonstrate the efficacy of this approach through several tasks, including image classification and natural language processing, where improvements in accuracy, convergence, and robustness are observed. This study offers a new perspective on combining mathematical theory and machine learning to create more interpretable and efficient models.

摘要: 本文探讨了将丢番图方程集成到神经网络（NN）架构中以提高模型的可解释性、稳定性和效率。通过将神经网络参数编码和解码为丢番图方程的整解，我们引入了一种新颖的方法，可以增强深度学习模型的精确性和鲁棒性。我们的方法集成了一个自定义损失函数，该函数在训练期间强制执行丢番图约束，从而实现更好的概括、减少错误界限并增强对抗性攻击的弹性。我们通过多项任务（包括图像分类和自然语言处理）证明了这种方法的有效性，其中观察到准确性、收敛性和鲁棒性的提高。这项研究为将数学理论和机器学习结合起来创建更可解释和高效的模型提供了一个新的视角。



## **18. Potion: Towards Poison Unlearning**

药剂：走向毒药的学习 cs.LG

Accepted for publication in the Journal of Data-centric Machine  Learning Research (DMLR) https://openreview.net/forum?id=4eSiRnWWaF

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.09173v3) [paper-pdf](http://arxiv.org/pdf/2406.09173v3)

**Authors**: Stefan Schoepf, Jack Foster, Alexandra Brintrup

**Abstract**: Adversarial attacks by malicious actors on machine learning systems, such as introducing poison triggers into training datasets, pose significant risks. The challenge in resolving such an attack arises in practice when only a subset of the poisoned data can be identified. This necessitates the development of methods to remove, i.e. unlearn, poison triggers from already trained models with only a subset of the poison data available. The requirements for this task significantly deviate from privacy-focused unlearning where all of the data to be forgotten by the model is known. Previous work has shown that the undiscovered poisoned samples lead to a failure of established unlearning methods, with only one method, Selective Synaptic Dampening (SSD), showing limited success. Even full retraining, after the removal of the identified poison, cannot address this challenge as the undiscovered poison samples lead to a reintroduction of the poison trigger in the model. Our work addresses two key challenges to advance the state of the art in poison unlearning. First, we introduce a novel outlier-resistant method, based on SSD, that significantly improves model protection and unlearning performance. Second, we introduce Poison Trigger Neutralisation (PTN) search, a fast, parallelisable, hyperparameter search that utilises the characteristic "unlearning versus model protection" trade-off to find suitable hyperparameters in settings where the forget set size is unknown and the retain set is contaminated. We benchmark our contributions using ResNet-9 on CIFAR10 and WideResNet-28x10 on CIFAR100. Experimental results show that our method heals 93.72% of poison compared to SSD with 83.41% and full retraining with 40.68%. We achieve this while also lowering the average model accuracy drop caused by unlearning from 5.68% (SSD) to 1.41% (ours).

摘要: 恶意行为者对机器学习系统的对抗性攻击，如将有毒触发器引入训练数据集，构成了巨大的风险。解决此类攻击的挑战出现在实践中，当只能识别有毒数据的子集时。这就需要开发方法来从仅有有毒数据的子集的已训练模型中移除(即取消学习)有毒触发器。这项任务的要求与关注隐私的遗忘有很大不同，在隐私遗忘中，模型要忘记的所有数据都是已知的。以前的工作表明，未发现的中毒样本会导致已有的遗忘方法的失败，只有一种方法-选择性突触抑制(SSD)-显示出有限的成功。即使在去除已识别的毒物之后进行全面的再培训，也不能解决这一挑战，因为未发现的毒物样本会导致在模型中重新引入毒物触发器。我们的工作解决了两个关键挑战，以推进毒物忘却学习的艺术水平。首先，我们提出了一种新的基于SSD的抗孤立点方法，该方法显著改善了模型保护和遗忘性能。其次，我们引入了毒药触发中和(PTN)搜索，这是一种快速、可并行的超参数搜索，它利用“遗忘与模型保护”的权衡特性，在忘记集大小未知且保留集受到污染的情况下找到合适的超参数。我们使用CIFAR10上的ResNet-9和CIFAR100上的WideResNet-28x10对我们的贡献进行基准测试。实验结果表明，与SSD的83.41%和完全再训练的40.68%相比，我们的方法可以治愈93.72%的毒物。我们实现了这一点，同时也将因遗忘而导致的平均模型精度下降从5.68%(SSD)降至1.41%(我们的)。



## **19. Countering adversarial perturbations in graphs using error correcting codes**

使用错误纠正码对抗图中的对抗性扰动 cs.CR

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.14245v2) [paper-pdf](http://arxiv.org/pdf/2406.14245v2)

**Authors**: Saif Eddin Jabari

**Abstract**: We consider the problem of a graph subjected to adversarial perturbations, such as those arising from cyber-attacks, where edges are covertly added or removed. The adversarial perturbations occur during the transmission of the graph between a sender and a receiver. To counteract potential perturbations, this study explores a repetition coding scheme with sender-assigned noise and majority voting on the receiver's end to rectify the graph's structure. The approach operates without prior knowledge of the attack's characteristics. We analytically derive a bound on the number of repetitions needed to satisfy probabilistic constraints on the quality of the reconstructed graph. The method can accurately and effectively decode Erd\H{o}s-R\'{e}nyi graphs that were subjected to non-random edge removal, namely, those connected to vertices with the highest eigenvector centrality, in addition to random addition and removal of edges by the attacker. The method is also effective against attacks on large scale-free graphs generated using the Barab\'{a}si-Albert model but require a larger number of repetitions than needed to correct Erd\H{o}s-R\'{e}nyi graphs.

摘要: 我们考虑了图受到对抗性扰动的问题，例如网络攻击引起的扰动，其中边被秘密地添加或删除。对抗性扰动发生在发送者和接收者之间的图形传输期间。为了抵消潜在的干扰，本研究探索了一种重复编码方案，该方案使用发送方分配的噪声和接收方的多数投票来纠正图的结构。该方法在事先不知道攻击特征的情况下进行操作。我们解析地推导出满足重构图质量的概率约束所需的重复次数的界限。该方法能够准确有效地破译非随机去边后的ErdHo S-R‘enyi图，即那些连接到特征向量中心度最高的顶点的图，以及攻击者对边的随机添加和删除.该方法对Barab‘a si-Albert模型生成的大规模无标度图的攻击也是有效的，但所需的重复次数比校正Erd{o}S-R’{e}nyi图所需的重复次数多.



## **20. The Philosopher's Stone: Trojaning Plugins of Large Language Models**

哲学家之石：大型语言模型的特洛伊插件 cs.CR

Accepted by NDSS Symposium 2025. Please cite this paper as "Tian  Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen  Liu, Haojin Zhu. The Philosopher's Stone: Trojaning Plugins of Large Language  Models. In the 32nd Annual Network and Distributed System Security Symposium  (NDSS 2025)."

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2312.00374v3) [paper-pdf](http://arxiv.org/pdf/2312.00374v3)

**Authors**: Tian Dong, Minhui Xue, Guoxing Chen, Rayne Holland, Yan Meng, Shaofeng Li, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers,an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses a superior LLM to align na\"ively poisoned data based on our insight that it can better inject poisoning knowledge during training. In contrast, FUSION leverages a novel over-poisoning procedure to transform a benign adapter into a malicious one by magnifying the attention between trigger and target in model weights. In our experiments, we first conduct two case studies to demonstrate that a compromised LLM agent can use malware to control the system (e.g., a LLM-driven robot) or to launch a spear-phishing attack. Then, in terms of targeted misinformation, we show that our attacks provide higher attack effectiveness than the existing baseline and, for the purpose of attracting downloads, preserve or improve the adapter's utility. Finally, we designed and evaluated three potential defenses. However, none proved entirely effective in safeguarding against our attacks, highlighting the need for more robust defenses supporting a secure LLM supply chain.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。基于我们对在训练过程中可以更好地注入中毒知识的洞察力，波兰德使用了一种高级的LLM来对齐严重中毒的数据。相比之下，Fusion利用一种新的过度中毒程序，通过放大模型权重中触发器和目标之间的注意力，将良性适配器转换为恶意适配器。在我们的实验中，我们首先进行了两个案例研究，以证明受攻击的LLM代理可以使用恶意软件控制系统(例如，LLM驱动的机器人)或发起鱼叉式网络钓鱼攻击。然后，在有针对性的错误信息方面，我们表明我们的攻击提供了比现有基线更高的攻击效率，并且出于吸引下载的目的，保留或提高了适配器的实用性。最后，我们设计并评估了三种可能的防御措施。然而，没有一种被证明在防御我们的攻击方面是完全有效的，这突显了需要更强大的防御来支持安全的LLM供应链。



## **21. FullCert: Deterministic End-to-End Certification for Training and Inference of Neural Networks**

FullCert：神经网络训练和推理的确定性端到端认证 cs.LG

This preprint has not undergone peer review or any post-submission  improvements or corrections. The Version of Record of this contribution is  published in DAGM GCPR 2024

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2406.11522v2) [paper-pdf](http://arxiv.org/pdf/2406.11522v2)

**Authors**: Tobias Lorenz, Marta Kwiatkowska, Mario Fritz

**Abstract**: Modern machine learning models are sensitive to the manipulation of both the training data (poisoning attacks) and inference data (adversarial examples). Recognizing this issue, the community has developed many empirical defenses against both attacks and, more recently, certification methods with provable guarantees against inference-time attacks. However, such guarantees are still largely lacking for training-time attacks. In this work, we present FullCert, the first end-to-end certifier with sound, deterministic bounds, which proves robustness against both training-time and inference-time attacks. We first bound all possible perturbations an adversary can make to the training data under the considered threat model. Using these constraints, we bound the perturbations' influence on the model's parameters. Finally, we bound the impact of these parameter changes on the model's prediction, resulting in joint robustness guarantees against poisoning and adversarial examples. To facilitate this novel certification paradigm, we combine our theoretical work with a new open-source library BoundFlow, which enables model training on bounded datasets. We experimentally demonstrate FullCert's feasibility on two datasets.

摘要: 现代机器学习模型对训练数据(中毒攻击)和推理数据(对抗性例子)的操纵都很敏感。认识到这一问题，社区已经开发了许多针对这两种攻击的经验防御方法，最近还开发了具有针对推理时间攻击的可证明保证的认证方法。然而，这样的保障在很大程度上仍然缺乏对训练时间攻击的保障。在这项工作中，我们提出了FullCert，这是第一个端到端证书，具有良好的确定性界，它证明了对训练时间和推理时间攻击的健壮性。我们首先在考虑的威胁模型下限制了对手可以对训练数据进行的所有可能的扰动。利用这些约束，我们限制了扰动对模型参数的影响。最后，我们结合了这些参数变化对模型预测的影响，从而对中毒和敌意示例提供了联合稳健性保证。为了促进这一新的认证范式，我们将我们的理论工作与新的开源库BordFlow相结合，该库能够对有界数据集进行模型训练。我们在两个数据集上实验验证了FullCert的可行性。



## **22. Adversarial Doodles: Interpretable and Human-drawable Attacks Provide Describable Insights**

对抗涂鸦：可解释和人类可绘制的攻击提供可描述的见解 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2311.15994v3) [paper-pdf](http://arxiv.org/pdf/2311.15994v3)

**Authors**: Ryoya Nara, Yusuke Matsui

**Abstract**: DNN-based image classifiers are susceptible to adversarial attacks. Most previous adversarial attacks do not have clear patterns, making it difficult to interpret attacks' results and gain insights into classifiers' mechanisms. Therefore, we propose Adversarial Doodles, which have interpretable shapes. We optimize black bezier curves to fool the classifier by overlaying them onto the input image. By introducing random affine transformation and regularizing the doodled area, we obtain small-sized attacks that cause misclassification even when humans replicate them by hand. Adversarial doodles provide describable insights into the relationship between the human-drawn doodle's shape and the classifier's output, such as "When we add three small circles on a helicopter image, the ResNet-50 classifier mistakenly classifies it as an airplane."

摘要: 基于DNN的图像分类器容易受到对抗攻击。之前的大多数对抗性攻击都没有明确的模式，因此很难解释攻击结果并深入了解分类器的机制。因此，我们提出了具有可解释形状的对抗涂鸦。我们优化黑色贝塞尔曲线，通过将它们叠加到输入图像上来欺骗分类器。通过引入随机仿射变换并规范化涂鸦区域，我们获得了即使人类手工复制也会导致错误分类的小规模攻击。对抗性涂鸦提供了有关人类绘制涂鸦形状与分类器输出之间关系的可描述的见解，例如“当我们在直升机图像上添加三个小圆圈时，ResNet-50分类器错误地将其分类为飞机。"



## **23. Privacy-oriented manipulation of speaker representations**

以隐私为导向的说话者表示操纵 eess.AS

Article published in IEEE Access

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2310.06652v2) [paper-pdf](http://arxiv.org/pdf/2310.06652v2)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.

摘要: 说话人嵌入无处不在，其应用范围从说话人识别和数字化到语音合成和语音匿名化。这些嵌入所包含的信息量使它们具有多功能性，但也引发了隐私问题。发言者嵌入已被证明包含有关年龄、性别、健康状况等信息，发言者可能希望保密，特别是当目标任务不需要这些信息时。在这项工作中，我们提出了一种从说话者嵌入中删除和操纵私人属性的方法，该方法利用了Vector-Quantized Variational Autoencoder架构，结合了对抗分类器和新型互信息丢失。我们根据两个属性（性别和年龄）验证我们的模型，并对无知且完全知情的攻击者以及域内和域外数据进行实验。



## **24. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2404.13621v6) [paper-pdf](http://arxiv.org/pdf/2404.13621v6)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks. Code is available at https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。代码可在https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.上找到



## **25. CPSample: Classifier Protected Sampling for Guarding Training Data During Diffusion**

CPSample：扩散期间警卫训练数据的分类器保护抽样 cs.LG

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07025v1) [paper-pdf](http://arxiv.org/pdf/2409.07025v1)

**Authors**: Joshua Kazdan, Hao Sun, Jiaqi Han, Felix Petersen, Stefano Ermon

**Abstract**: Diffusion models have a tendency to exactly replicate their training data, especially when trained on small datasets. Most prior work has sought to mitigate this problem by imposing differential privacy constraints or masking parts of the training data, resulting in a notable substantial decrease in image quality. We present CPSample, a method that modifies the sampling process to prevent training data replication while preserving image quality. CPSample utilizes a classifier that is trained to overfit on random binary labels attached to the training data. CPSample then uses classifier guidance to steer the generation process away from the set of points that can be classified with high certainty, a set that includes the training data. CPSample achieves FID scores of 4.97 and 2.97 on CIFAR-10 and CelebA-64, respectively, without producing exact replicates of the training data. Unlike prior methods intended to guard the training images, CPSample only requires training a classifier rather than retraining a diffusion model, which is computationally cheaper. Moreover, our technique provides diffusion models with greater robustness against membership inference attacks, wherein an adversary attempts to discern which images were in the model's training dataset. We show that CPSample behaves like a built-in rejection sampler, and we demonstrate its capabilities to prevent mode collapse in Stable Diffusion.

摘要: 扩散模型有精确复制其训练数据的趋势，尤其是在小数据集上进行训练时。大多数以前的工作都试图通过施加不同的隐私限制或屏蔽部分训练数据来缓解这个问题，导致图像质量显著下降。我们提出了CPSample方法，该方法修改了采样过程，在保持图像质量的同时防止了训练数据的复制。CPSample利用一个分类器，该分类器被训练成在附加到训练数据的随机二进制标签上过度匹配。然后，CPSample使用分类器指导来引导生成过程远离可被高度确定地分类的点集，该点集包括训练数据。CPSample在CIFAR-10和CelebA-上的FID得分分别为4.97和2.97，但没有产生训练数据的准确副本。与以往用于保护训练图像的方法不同，CPSample只需要训练分类器，而不需要重新训练扩散模型，这在计算上更便宜。此外，我们的技术为扩散模型提供了更好的抵抗成员推理攻击的稳健性，其中对手试图辨别哪些图像在模型的训练数据集中。我们证明了CPSample的行为就像一个内置的拒绝采样器，并且我们展示了它在稳定扩散中防止模式崩溃的能力。



## **26. AdvLogo: Adversarial Patch Attack against Object Detectors based on Diffusion Models**

AdvLogo：针对基于扩散模型的对象检测器的对抗补丁攻击 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2409.07002v1) [paper-pdf](http://arxiv.org/pdf/2409.07002v1)

**Authors**: Boming Miao, Chunxiao Li, Yao Zhu, Weixiang Sun, Zizhe Wang, Xiaoyi Wang, Chuanlong Xie

**Abstract**: With the rapid development of deep learning, object detectors have demonstrated impressive performance; however, vulnerabilities still exist in certain scenarios. Current research exploring the vulnerabilities using adversarial patches often struggles to balance the trade-off between attack effectiveness and visual quality. To address this problem, we propose a novel framework of patch attack from semantic perspective, which we refer to as AdvLogo. Based on the hypothesis that every semantic space contains an adversarial subspace where images can cause detectors to fail in recognizing objects, we leverage the semantic understanding of the diffusion denoising process and drive the process to adversarial subareas by perturbing the latent and unconditional embeddings at the last timestep. To mitigate the distribution shift that exposes a negative impact on image quality, we apply perturbation to the latent in frequency domain with the Fourier Transform. Experimental results demonstrate that AdvLogo achieves strong attack performance while maintaining high visual quality.

摘要: 随着深度学习的快速发展，目标检测器表现出了令人印象深刻的性能，但在某些场景下仍然存在漏洞。目前使用对抗性补丁探索漏洞的研究往往难以在攻击效率和视觉质量之间取得平衡。针对这一问题，我们从语义的角度提出了一种新的补丁攻击框架，称为AdvLogo。基于每个语义空间包含一个对抗性的子空间的假设，在这个子空间中，图像可能导致检测器无法识别目标，我们利用扩散去噪过程的语义理解，通过在最后一个时间步扰动潜在的和无条件的嵌入来驱动该过程到对抗性的子区域。为了减少分布漂移对图像质量的负面影响，我们利用傅里叶变换对频域中的潜伏点进行扰动。实验结果表明，AdvLogo在保持较高视觉质量的同时，具有较强的攻击性能。



## **27. Privacy Leakage on DNNs: A Survey of Model Inversion Attacks and Defenses**

DNN上的隐私泄露：模型倒置攻击和防御的调查 cs.CV

**SubmitDate**: 2024-09-11    [abs](http://arxiv.org/abs/2402.04013v2) [paper-pdf](http://arxiv.org/pdf/2402.04013v2)

**Authors**: Hao Fang, Yixiang Qiu, Hongyao Yu, Wenbo Yu, Jiawei Kong, Baoli Chong, Bin Chen, Xuan Wang, Shu-Tao Xia, Ke Xu

**Abstract**: Deep Neural Networks (DNNs) have revolutionized various domains with their exceptional performance across numerous applications. However, Model Inversion (MI) attacks, which disclose private information about the training dataset by abusing access to the trained models, have emerged as a formidable privacy threat. Given a trained network, these attacks enable adversaries to reconstruct high-fidelity data that closely aligns with the private training samples, posing significant privacy concerns. Despite the rapid advances in the field, we lack a comprehensive and systematic overview of existing MI attacks and defenses. To fill this gap, this paper thoroughly investigates this realm and presents a holistic survey. Firstly, our work briefly reviews early MI studies on traditional machine learning scenarios. We then elaborately analyze and compare numerous recent attacks and defenses on Deep Neural Networks (DNNs) across multiple modalities and learning tasks. By meticulously analyzing their distinctive features, we summarize and classify these methods into different categories and provide a novel taxonomy. Finally, this paper discusses promising research directions and presents potential solutions to open issues. To facilitate further study on MI attacks and defenses, we have implemented an open-source model inversion toolbox on GitHub (https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox).

摘要: 深度神经网络(DNN)以其在众多应用中的卓越性能给各个领域带来了革命性的变化。然而，模型反转(MI)攻击通过滥用对训练模型的访问来泄露关于训练数据集的私人信息，已经成为一种强大的隐私威胁。给定一个训练有素的网络，这些攻击使攻击者能够重建与私人训练样本紧密一致的高保真数据，从而造成严重的隐私问题。尽管该领域取得了快速进展，但我们缺乏对现有MI攻击和防御的全面和系统的概述。为了填补这一空白，本文对这一领域进行了深入的研究，并进行了全面的调查。首先，我们的工作简要回顾了早期关于传统机器学习场景的MI研究。然后，我们详细地分析和比较了最近针对深度神经网络(DNN)的众多攻击和防御措施，这些攻击和防御涉及多个通道和学习任务。通过仔细分析它们的特点，我们对这些方法进行了总结和分类，并提供了一种新的分类方法。最后，本文讨论了未来的研究方向，并提出了可能的解决方案。为了方便对MI攻击和防御的进一步研究，我们在GitHub(https://github.com/ffhibnese/Model-Inversion-Attack-ToolBox).上实现了一个开源的模型反演工具箱



## **28. Well, that escalated quickly: The Single-Turn Crescendo Attack (STCA)**

嗯，情况迅速升级：单转渐强攻击（STCA） cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.03131v2) [paper-pdf](http://arxiv.org/pdf/2409.03131v2)

**Authors**: Alan Aqrawi, Arian Abbasi

**Abstract**: This paper introduces a new method for adversarial attacks on large language models (LLMs) called the Single-Turn Crescendo Attack (STCA). Building on the multi-turn crescendo attack method introduced by Russinovich, Salem, and Eldan (2024), which gradually escalates the context to provoke harmful responses, the STCA achieves similar outcomes in a single interaction. By condensing the escalation into a single, well-crafted prompt, the STCA bypasses typical moderation filters that LLMs use to prevent inappropriate outputs. This technique reveals vulnerabilities in current LLMs and emphasizes the importance of stronger safeguards in responsible AI (RAI). The STCA offers a novel method that has not been previously explored.

摘要: 本文介绍了一种针对大型语言模型（LLM）的对抗性攻击的新方法，称为单轮渐强攻击（STCA）。STCA基于Russinovich、Salem和Eldan（2024）引入的多回合渐强攻击方法（该方法逐渐升级上下文以引发有害反应），在单次交互中实现了类似的结果。通过将升级浓缩为一个精心设计的提示，STCA绕过了LLM用来防止不当输出的典型审核过滤器。该技术揭示了当前LLM中的漏洞，并强调了负责任人工智能（RAI）中更强有力的保障措施的重要性。STCA提供了一种以前尚未探索过的新颖方法。



## **29. Personalized Federated Learning Techniques: Empirical Analysis**

个性化联邦学习技术：实证分析 cs.LG

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06805v1) [paper-pdf](http://arxiv.org/pdf/2409.06805v1)

**Authors**: Azal Ahmad Khan, Ahmad Faraz Khan, Haider Ali, Ali Anwar

**Abstract**: Personalized Federated Learning (pFL) holds immense promise for tailoring machine learning models to individual users while preserving data privacy. However, achieving optimal performance in pFL often requires a careful balancing act between memory overhead costs and model accuracy. This paper delves into the trade-offs inherent in pFL, offering valuable insights for selecting the right algorithms for diverse real-world scenarios. We empirically evaluate ten prominent pFL techniques across various datasets and data splits, uncovering significant differences in their performance. Our study reveals interesting insights into how pFL methods that utilize personalized (local) aggregation exhibit the fastest convergence due to their efficiency in communication and computation. Conversely, fine-tuning methods face limitations in handling data heterogeneity and potential adversarial attacks while multi-objective learning methods achieve higher accuracy at the cost of additional training and resource consumption. Our study emphasizes the critical role of communication efficiency in scaling pFL, demonstrating how it can significantly affect resource usage in real-world deployments.

摘要: 个性化联合学习(PFL)在保护数据隐私的同时，为个人用户定制机器学习模型有着巨大的前景。然而，要在PFL中实现最佳性能，通常需要在内存开销成本和模型准确性之间仔细权衡。本文深入研究了PFL固有的权衡，为为不同的现实世界场景选择正确的算法提供了有价值的见解。我们在不同的数据集和数据分割中对十种重要的PFL技术进行了实证评估，发现它们在性能上存在显著差异。我们的研究揭示了有趣的见解，即利用个性化(本地)聚合的PFL方法如何由于其在通信和计算方面的效率而表现出最快的收敛速度。相反，微调方法在处理数据异构性和潜在的对抗性攻击方面面临局限性，而多目标学习方法以额外的训练和资源消耗为代价获得更高的准确率。我们的研究强调了通信效率在扩展PFL中的关键作用，展示了它如何在实际部署中显著影响资源使用。



## **30. Adversarial Attacks to Multi-Modal Models**

对多模式模型的对抗攻击 cs.CR

To appear in the ACM Workshop on Large AI Systems and Models with  Privacy and Safety Analysis 2024 (LAMPS '24)

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06793v1) [paper-pdf](http://arxiv.org/pdf/2409.06793v1)

**Authors**: Zhihao Dou, Xin Hu, Haibo Yang, Zhuqing Liu, Minghong Fang

**Abstract**: Multi-modal models have gained significant attention due to their powerful capabilities. These models effectively align embeddings across diverse data modalities, showcasing superior performance in downstream tasks compared to their unimodal counterparts. Recent study showed that the attacker can manipulate an image or audio file by altering it in such a way that its embedding matches that of an attacker-chosen targeted input, thereby deceiving downstream models. However, this method often underperforms due to inherent disparities in data from different modalities. In this paper, we introduce CrossFire, an innovative approach to attack multi-modal models. CrossFire begins by transforming the targeted input chosen by the attacker into a format that matches the modality of the original image or audio file. We then formulate our attack as an optimization problem, aiming to minimize the angular deviation between the embeddings of the transformed input and the modified image or audio file. Solving this problem determines the perturbations to be added to the original media. Our extensive experiments on six real-world benchmark datasets reveal that CrossFire can significantly manipulate downstream tasks, surpassing existing attacks. Additionally, we evaluate six defensive strategies against CrossFire, finding that current defenses are insufficient to counteract our CrossFire.

摘要: 多通道模型因其强大的性能而备受关注。这些模型有效地调整了跨不同数据模式的嵌入，在下游任务中展示了与单峰对应的卓越性能。最近的研究表明，攻击者可以通过改变图像或音频文件的嵌入方式来操纵它，使其嵌入与攻击者选择的目标输入相匹配，从而欺骗下游模型。然而，由于来自不同模式的数据的内在差异，该方法常常表现不佳。在本文中，我们介绍了一种创新的攻击多通道模型的方法--CrossFire。CrossFire首先将攻击者选择的目标输入转换为与原始图像或音频文件的形态相匹配的格式。然后，我们将攻击描述为一个优化问题，旨在最小化转换后的输入和修改后的图像或音频文件的嵌入之间的角度偏差。解决此问题将确定要添加到原始介质的扰动。我们在六个真实世界基准数据集上的广泛实验表明，CrossFire可以显著操纵下游任务，超过现有的攻击。此外，我们评估了六种防御交叉火力的策略，发现目前的防御不足以对抗我们的交叉火力。



## **31. DNN-Defender: A Victim-Focused In-DRAM Defense Mechanism for Taming Adversarial Weight Attack on DNNs**

DNN防御者：一种以受害者为中心的内存防御机制，用于驯服DNN上的对抗权重攻击 cs.CR

6 pages, 9 figures

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2305.08034v2) [paper-pdf](http://arxiv.org/pdf/2305.08034v2)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks with a priority protection mechanism. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring hardware overhead.

摘要: 随着深度学习在许多安全敏感领域的部署，机器学习的安全性正变得越来越重要。最近的研究表明，攻击者可以利用系统级技术，利用DRAM的RowHammer漏洞来确定并精确地翻转深度神经网络(DNN)模型中的位，以影响推理精度。现有的防御机制是基于软件的，例如需要昂贵的训练开销的权重重建或性能下降。另一方面，通用的基于硬件的以受害者/攻击者为中心的机制增加了昂贵的硬件开销，并保持了受害者和攻击者行之间的空间连接。在本文中，我们提出了第一个基于DRAM的针对量化DNN的以受害者为中心的防御机制，称为DNN-Defender，它利用DRAM内交换的潜力来抵御目标位翻转攻击，并具有优先保护机制。我们的结果表明，DNN-Defender可以提供高级别的保护，将目标RowHammer攻击的性能降低到随机攻击级别。此外，建议的防御在CIFAR-10和ImageNet数据集上没有精度下降，不需要任何软件培训或产生硬件开销。



## **32. DV-FSR: A Dual-View Target Attack Framework for Federated Sequential Recommendation**

DV-FSR：一种用于联合顺序推荐的双视图目标攻击框架 cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.07500v1) [paper-pdf](http://arxiv.org/pdf/2409.07500v1)

**Authors**: Qitao Qin, Yucong Luo, Mingyue Cheng, Qingyang Mao, Chenyi Lei

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation (FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models.

摘要: 联邦推荐(FedRec)通过支持个性化模型的分散训练来保护用户隐私，但这种体系结构天生就容易受到敌意攻击。出于商业和社会影响的考虑，对FedRec系统中的目标攻击进行了重要的研究。然而，这些工作在很大程度上忽略了推荐模型的差异化稳健性。此外，我们的实验结果表明，现有的定向攻击方法在联邦顺序推荐(FSR)任务中只能取得有限的效果。在此基础上，我们重点研究了FSR中的目标攻击，并提出了一种新的DualView攻击框架DV-FSR。该攻击方法独特地结合了基于采样的显式策略和基于对比学习的隐式梯度策略来协调攻击。此外，我们在FSR中引入了一种针对目标攻击的特定防御机制，旨在评估我们提出的攻击方法的缓解效果。在典型的序列模型上进行了大量的实验，验证了该方法的有效性。



## **33. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

在基于能量的模型的视角下更多地关注稳健分类器 cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2407.06315v3) [paper-pdf](http://arxiv.org/pdf/2407.06315v3)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .

摘要: 通过将稳健的判别分类器重新解释为基于能量的模型(EBM)，我们提供了一种新的方法来研究对手训练(AT)的动态。我们对AT过程中的能量格局的分析表明，从模型的角度来看，非目标攻击产生的敌意图像比原始数据更不均匀(能量更低)。相反，我们在有针对性的攻击中观察到相反的情况。在我们深入分析的基础上，我们提出了新的理论和实践结果，表明解释AT能量动力学如何揭示更好的理解：(1)AT动态由三个阶段控制，鲁棒过拟合发生在第三阶段，自然能量和对抗能量之间存在巨大差异(2)通过代理损失最小化(交易)在能量方面改写了权衡激发的对抗性防御的损失，我们表明，交易通过将自然能量与对手能量对齐的方式隐含地缓解了过度匹配。(3)我们的经验表明，所有最近最先进的稳健分类器都在平滑能量格局，我们协调了关于理解AT和在EBM保护伞下加权损失函数的各种研究。在严格证据的激励下，我们提出了加权能量对抗训练(Weat)，这是一种新的样本加权方案，其精度与CIFAR-10和SVHN等多个基准测试的最新水平相当，并超过CIFAR-100和Tiny-ImageNet。我们进一步证明了健壮分类器在其生成能力的强度和质量上存在差异，并提供了一种简单的方法来推动这一能力，使用健壮分类器而不需要为生成性建模进行训练就可以达到显著的初始得分(IS)和FID。复制我们结果的代码可以在http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/上找到。



## **34. Frosty: Bringing strong liveness guarantees to the Snow family of consensus protocols**

Frosty：为Snow家族的共识协议带来强大的活力保证 cs.DC

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2404.14250v5) [paper-pdf](http://arxiv.org/pdf/2404.14250v5)

**Authors**: Aaron Buchwald, Stephen Buttolph, Andrew Lewis-Pye, Patrick O'Grady, Kevin Sekniqi

**Abstract**: Snowman is the consensus protocol implemented by the Avalanche blockchain and is part of the Snow family of protocols, first introduced through the original Avalanche leaderless consensus protocol. A major advantage of Snowman is that each consensus decision only requires an expected constant communication overhead per processor in the `common' case that the protocol is not under substantial Byzantine attack, i.e. it provides a solution to the scalability problem which ensures that the expected communication overhead per processor is independent of the total number of processors $n$ during normal operation. This is the key property that would enable a consensus protocol to scale to 10,000 or more independent validators (i.e. processors). On the other hand, the two following concerns have remained:   (1) Providing formal proofs of consistency for Snowman has presented a formidable challenge.   (2) Liveness attacks exist in the case that a Byzantine adversary controls more than $O(\sqrt{n})$ processors, slowing termination to more than a logarithmic number of steps.   In this paper, we address the two issues above. We consider a Byzantine adversary that controls at most $f<n/5$ processors. First, we provide a simple proof of consistency for Snowman. Then we supplement Snowman with a `liveness module' that can be triggered in the case that a substantial adversary launches a liveness attack, and which guarantees liveness in this event by temporarily forgoing the communication complexity advantages of Snowman, but without sacrificing these low communication complexity advantages during normal operation.

摘要: 雪人是雪崩区块链实施的共识协议，是雪诺协议家族的一部分，最初是通过最初的雪崩无领导共识协议引入的。Snowman的一个主要优势是，在协议没有受到实质性拜占庭攻击的情况下，每个协商一致的决定只需要每个处理器预期的恒定通信开销，即它提供了对可伸缩性问题的解决方案，该解决方案确保在正常操作期间每个处理器的预期通信开销与处理器总数$n$无关。这是使共识协议能够扩展到10,000个或更多独立验证器(即处理器)的关键属性。另一方面，以下两个问题仍然存在：(1)为雪人提供一致性的正式证据是一个巨大的挑战。(2)当拜占庭敌手控制超过$O(\Sqrt{n})$个处理器时，存在活性攻击，从而将终止速度减慢到超过对数步数。在本文中，我们解决了上述两个问题。我们考虑一个拜占庭对手，它至多控制$f<n/5$处理器。首先，我们为雪人提供了一个简单的一致性证明。然后，我们给Snowman增加了一个活跃度模块，该模块可以在强大的对手发起活跃度攻击的情况下触发，并通过暂时放弃Snowman的通信复杂性优势来保证在这种情况下的活跃性，但在正常运行时不会牺牲这些低通信复杂性的优势。



## **35. Unrevealed Threats: A Comprehensive Study of the Adversarial Robustness of Underwater Image Enhancement Models**

未揭露的威胁：水下图像增强模型对抗鲁棒性的综合研究 eess.IV

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06420v1) [paper-pdf](http://arxiv.org/pdf/2409.06420v1)

**Authors**: Siyu Zhai, Zhibo He, Xiaofeng Cong, Junming Hou, Jie Gui, Jian Wei You, Xin Gong, James Tin-Yau Kwok, Yuan Yan Tang

**Abstract**: Learning-based methods for underwater image enhancement (UWIE) have undergone extensive exploration. However, learning-based models are usually vulnerable to adversarial examples so as the UWIE models. To the best of our knowledge, there is no comprehensive study on the adversarial robustness of UWIE models, which indicates that UWIE models are potentially under the threat of adversarial attacks. In this paper, we propose a general adversarial attack protocol. We make a first attempt to conduct adversarial attacks on five well-designed UWIE models on three common underwater image benchmark datasets. Considering the scattering and absorption of light in the underwater environment, there exists a strong correlation between color correction and underwater image enhancement. On the basis of that, we also design two effective UWIE-oriented adversarial attack methods Pixel Attack and Color Shift Attack targeting different color spaces. The results show that five models exhibit varying degrees of vulnerability to adversarial attacks and well-designed small perturbations on degraded images are capable of preventing UWIE models from generating enhanced results. Further, we conduct adversarial training on these models and successfully mitigated the effectiveness of adversarial attacks. In summary, we reveal the adversarial vulnerability of UWIE models and propose a new evaluation dimension of UWIE models.

摘要: 基于学习的水下图像增强方法已经得到了广泛的探索。然而，与UWIE模型一样，基于学习的模型通常容易受到对抗性例子的影响。据我们所知，目前还没有对UWIE模型的对抗稳健性进行全面的研究，这表明UWIE模型可能受到对抗性攻击的威胁。本文提出了一种通用的对抗性攻击协议。我们首次尝试在三个常见的水下图像基准数据集上对五个精心设计的UWIE模型进行对抗性攻击。考虑到光在水下环境中的散射和吸收，颜色校正与水下图像增强之间存在着很强的相关性。在此基础上，针对不同的颜色空间，设计了两种有效的面向UWIE的对抗性攻击方法Pixel攻击和Color Shift攻击。结果表明，5种模型对敌意攻击表现出不同程度的脆弱性，对降质图像进行精心设计的小扰动能够阻止UWIE模型产生增强结果。此外，我们在这些模型上进行了对抗性训练，并成功地缓解了对抗性攻击的有效性。综上所述，我们揭示了UWIE模型的对抗性弱点，并提出了一种新的UWIE模型评价维度。



## **36. Passive Inference Attacks on Split Learning via Adversarial Regularization**

通过对抗正规化对分裂学习的被动推理攻击 cs.CR

To appear at NDSS 2025; 25 pages, 27 figures

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2310.10483v5) [paper-pdf](http://arxiv.org/pdf/2310.10483v5)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more capable attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves significantly superior attack performance, even comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更有能力的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在现有被动攻击难以有效重建客户端私有数据的挑战性场景中，SDAR始终实现显著优越的攻击性能，甚至可以与主动攻击相媲美。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **37. Influence-based Attributions can be Manipulated**

基于影响力的归因可以被操纵 cs.LG

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.05208v2) [paper-pdf](http://arxiv.org/pdf/2409.05208v2)

**Authors**: Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Influence Functions are a standard tool for attributing predictions to training data in a principled manner and are widely used in applications such as data valuation and fairness. In this work, we present realistic incentives to manipulate influencebased attributions and investigate whether these attributions can be systematically tampered by an adversary. We show that this is indeed possible and provide efficient attacks with backward-friendly implementations. Our work raises questions on the reliability of influence-based attributions under adversarial circumstances.

摘要: 影响力函数是一种标准工具，用于以有原则的方式将预测归因于训练数据，并广泛用于数据评估和公平性等应用中。在这项工作中，我们提出了操纵基于影响力的属性的现实激励，并调查这些属性是否可以被对手系统性篡改。我们证明这确实是可能的，并通过向后友好的实现提供高效的攻击。我们的工作对敌对情况下基于影响力的归因的可靠性提出了质疑。



## **38. On the Weaknesses of Backdoor-based Model Watermarking: An Information-theoretic Perspective**

基于后门的模型水印的弱点：信息论的角度 cs.CR

**SubmitDate**: 2024-09-10    [abs](http://arxiv.org/abs/2409.06130v1) [paper-pdf](http://arxiv.org/pdf/2409.06130v1)

**Authors**: Aoting Hu, Yanzhi Chen, Renjie Xie, Adrian Weller

**Abstract**: Safeguarding the intellectual property of machine learning models has emerged as a pressing concern in AI security. Model watermarking is a powerful technique for protecting ownership of machine learning models, yet its reliability has been recently challenged by recent watermark removal attacks. In this work, we investigate why existing watermark embedding techniques particularly those based on backdooring are vulnerable. Through an information-theoretic analysis, we show that the resilience of watermarking against erasure attacks hinges on the choice of trigger-set samples, where current uses of out-distribution trigger-set are inherently vulnerable to white-box adversaries. Based on this discovery, we propose a novel model watermarking scheme, In-distribution Watermark Embedding (IWE), to overcome the limitations of existing method. To further minimise the gap to clean models, we analyze the role of logits as watermark information carriers and propose a new approach to better conceal watermark information within the logits. Experiments on real-world datasets including CIFAR-100 and Caltech-101 demonstrate that our method robustly defends against various adversaries with negligible accuracy loss (< 0.1%).

摘要: 保护机器学习模型的知识产权已成为人工智能安全领域的一个紧迫问题。模型水印是一种保护机器学习模型所有权的强大技术，但其可靠性最近受到了最近的水印移除攻击的挑战。在这项工作中，我们调查了为什么现有的水印嵌入技术，特别是那些基于回溯的水印嵌入技术是脆弱的。通过信息论分析，我们证明了水印对擦除攻击的抵抗能力取决于触发集样本的选择，而当前使用的外部分布触发集本身就容易受到白盒攻击。基于这一发现，我们提出了一种新的模型水印方案--分布内水印嵌入(IWE)，以克服现有方法的局限性。为了进一步缩小与CLEAN模型的差距，我们分析了Logits作为水印信息载体的作用，并提出了一种新的方法来更好地隐藏Logits中的水印信息。在包括CIFAR-100和CALTECH-101在内的真实世界数据集上的实验表明，我们的方法在几乎可以忽略不计的准确率损失(<0.1%)的情况下，稳健地防御各种对手。



## **39. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

通过触发优化的数据中毒隐藏联邦学习中后门模型更新 cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2405.06206v2) [paper-pdf](http://arxiv.org/pdf/2405.06206v2)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on analyzing clients' model updates exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign client model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.

摘要: 联合学习(FL)是一种去中心化的机器学习方法，允许参与者在不共享私人数据的情况下协作训练模型。尽管FL具有隐私和可扩展性方面的优势，但它很容易受到后门攻击，即攻击者使用后门触发器毒化部分客户端的本地训练数据，目的是在推理时输入满足相同的后门条件时，使聚合模型产生恶意结果。FL中现有的后门攻击存在共同的缺陷：固定的触发模式和依赖模型中毒的辅助。基于分析客户端模型更新的最新防御技术在这些攻击中表现出良好的防御性能，因为恶意客户端模型更新和良性客户端模型更新之间存在显著差异。为了有效地隐藏良性模型更新中的恶意模型更新，我们提出了一种FL中的后门攻击策略DPOT，它通过优化后门触发器来动态构建后门目标，使后门数据对模型更新的影响最小。我们为DPOT的攻击原理提供了理论依据，并展示了实验结果表明，DPOT仅通过一次数据中毒攻击就可以有效地破坏最先进的防御措施，并在各种数据集上优于现有的后门攻击技术。



## **40. Cross-Input Certified Training for Universal Perturbations**

针对普遍扰动的交叉输入认证培训 cs.LG

23 pages, 6 figures, ECCV '24

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2405.09176v2) [paper-pdf](http://arxiv.org/pdf/2405.09176v2)

**Authors**: Changming Xu, Gagandeep Singh

**Abstract**: Existing work in trustworthy machine learning primarily focuses on single-input adversarial perturbations. In many real-world attack scenarios, input-agnostic adversarial attacks, e.g. universal adversarial perturbations (UAPs), are much more feasible. Current certified training methods train models robust to single-input perturbations but achieve suboptimal clean and UAP accuracy, thereby limiting their applicability in practical applications. We propose a novel method, CITRUS, for certified training of networks robust against UAP attackers. We show in an extensive evaluation across different datasets, architectures, and perturbation magnitudes that our method outperforms traditional certified training methods on standard accuracy (up to 10.3\%) and achieves SOTA performance on the more practical certified UAP accuracy metric.

摘要: 可信机器学习的现有工作主要集中在单输入对抗性扰动上。在许多现实世界的攻击场景中，输入不可知的对抗性攻击，例如通用对抗性扰动（UPC），更为可行。当前经过认证的训练方法训练模型对单输入扰动具有鲁棒性，但实现了次优的干净和UAP准确性，从而限制了其在实际应用中的适用性。我们提出了一种新颖的方法CITRUS，用于对抵御UAP攻击者的强大网络进行认证训练。我们在对不同数据集、架构和扰动幅度的广泛评估中表明，我们的方法在标准准确性方面优于传统认证训练方法（高达10.3%），并在更实用的认证UAP准确性指标方面实现了SOTA性能。



## **41. Espresso: Robust Concept Filtering in Text-to-Image Models**

浓缩咖啡：文本到图像模型中的稳健概念过滤 cs.CV

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2404.19227v5) [paper-pdf](http://arxiv.org/pdf/2404.19227v5)

**Authors**: Anudeep Das, Vasisht Duddu, Rui Zhang, N. Asokan

**Abstract**: Diffusion based text-to-image models are trained on large datasets scraped from the Internet, potentially containing unacceptable concepts (e.g., copyright infringing or unsafe). We need concept removal techniques (CRTs) which are effective in preventing the generation of images with unacceptable concepts, utility-preserving on acceptable concepts, and robust against evasion with adversarial prompts. None of the prior CRTs satisfy all these requirements simultaneously. We introduce Espresso, the first robust concept filter based on Contrastive Language-Image Pre-Training (CLIP). We configure CLIP to identify unacceptable concepts in generated images using the distance of their embeddings to the text embeddings of both unacceptable and acceptable concepts. This lets us fine-tune for robustness by separating the text embeddings of unacceptable and acceptable concepts while preserving their pairing with image embeddings for utility. We present a pipeline to evaluate various CRTs, attacks against them, and show that Espresso, is more effective and robust than prior CRTs, while retaining utility.

摘要: 基于扩散的文本到图像模型是在从互联网上收集的大数据集上进行训练的，这些数据集可能包含不可接受的概念(例如，侵犯版权或不安全)。我们需要概念移除技术(CRT)，它能有效地防止生成包含不可接受概念的图像，保留可接受概念的效用，并对带有对抗性提示的规避具有健壮性。以前的CRT没有一种同时满足所有这些要求。介绍了第一个基于对比语言-图像预训练(CLIP)的稳健概念过滤器Espresso。我们将CLIP配置为在生成的图像中识别不可接受的概念，使用其嵌入到不可接受和可接受概念的文本嵌入的距离。这使我们可以通过分离不可接受和可接受概念的文本嵌入，同时保留它们与图像嵌入的配对以实现实用，从而对健壮性进行微调。我们提出了一种评估各种CRT的流水线，对它们的攻击，并表明Espresso，比以前的CRT更有效和健壮，同时保持了实用性。



## **42. Unlearning or Concealment? A Critical Analysis and Evaluation Metrics for Unlearning in Diffusion Models**

忘记还是隐瞒？扩散模型中放弃学习的批判性分析和评估工具包 cs.LG

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05668v1) [paper-pdf](http://arxiv.org/pdf/2409.05668v1)

**Authors**: Aakash Sen Sharma, Niladri Sarkar, Vikram Chundawat, Ankur A Mali, Murari Mandal

**Abstract**: Recent research has seen significant interest in methods for concept removal and targeted forgetting in diffusion models. In this paper, we conduct a comprehensive white-box analysis to expose significant vulnerabilities in existing diffusion model unlearning methods. We show that the objective functions used for unlearning in the existing methods lead to decoupling of the targeted concepts (meant to be forgotten) for the corresponding prompts. This is concealment and not actual unlearning, which was the original goal. The ineffectiveness of current methods stems primarily from their narrow focus on reducing generation probabilities for specific prompt sets, neglecting the diverse modalities of intermediate guidance employed during the inference process. The paper presents a rigorous theoretical and empirical examination of four commonly used techniques for unlearning in diffusion models. We introduce two new evaluation metrics: Concept Retrieval Score (CRS) and Concept Confidence Score (CCS). These metrics are based on a successful adversarial attack setup that can recover forgotten concepts from unlearned diffusion models. The CRS measures the similarity between the latent representations of the unlearned and fully trained models after unlearning. It reports the extent of retrieval of the forgotten concepts with increasing amount of guidance. The CCS quantifies the confidence of the model in assigning the target concept to the manipulated data. It reports the probability of the unlearned model's generations to be aligned with the original domain knowledge with increasing amount of guidance. Evaluating existing unlearning methods with our proposed stringent metrics for diffusion models reveals significant shortcomings in their ability to truly unlearn concepts. Source Code: https://respailab.github.io/unlearning-or-concealment

摘要: 最近的研究对扩散模型中的概念移除和目标遗忘的方法产生了浓厚的兴趣。在本文中，我们进行了全面的白盒分析，以揭示现有扩散模型遗忘方法中的重大漏洞。我们证明了现有方法中用于遗忘的目标函数导致了对应提示的目标概念(意图被遗忘)的解耦。这是隐藏，而不是真正的遗忘，这是最初的目标。当前方法的无效主要是因为它们狭隘地侧重于降低特定提示集的生成概率，而忽视了推理过程中采用的中间指导的各种形式。本文对扩散模型中四种常用的遗忘技术进行了严格的理论和实证检验。我们引入了两个新的评价指标：概念检索得分(CRS)和概念置信度得分(CCS)。这些指标基于成功的对抗性攻击设置，可以从未学习的扩散模型中恢复忘记的概念。CRS度量未学习的模型和完全训练的模型在忘记后的潜在表示之间的相似性。它报告了对被遗忘的概念的检索程度，并提供了越来越多的指导。CCS在将目标概念分配给被操纵的数据时量化模型的置信度。它报告了随着指导量的增加，未学习模型的生成与原始领域知识保持一致的概率。用我们提出的扩散模型的严格度量来评估现有的遗忘方法，发现它们在真正忘记概念的能力方面存在重大缺陷。源代码：https://respailab.github.io/unlearning-or-concealment



## **43. Adversarial Attacks on Data Attribution**

对数据归因的对抗性攻击 cs.LG

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05657v1) [paper-pdf](http://arxiv.org/pdf/2409.05657v1)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities, and by proposing principled adversarial attack methods on data attribution. We present two such methods, Shadow Attack and Outlier Attack, both of which generate manipulated datasets to adversarially inflate the compensation. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%.

摘要: 数据属性旨在量化单个训练数据点对人工智能模型输出的贡献，该模型已被用于衡量训练数据的价值并补偿数据提供者。考虑到对财务决策和补偿机制的影响，数据归因方法的对抗性稳健性出现了一个关键问题。然而，很少或根本没有针对这一问题的系统研究。在这项工作中，我们旨在通过详细描述威胁模型来弥合这一差距，该模型具有关于对手目标和能力的明确假设，并提出了关于数据归因的原则性对抗性攻击方法。我们提出了两种这样的方法，影子攻击和离群点攻击，这两种方法都会生成被操纵的数据集，以相反地夸大补偿。影子攻击利用人工智能应用程序中数据分布的知识，通过成员关系推理攻击中常用的一种技术“影子训练”来获得对抗性扰动。相比之下，离群点攻击不假设任何关于数据分布的知识，并且仅依赖于对目标模型的预测的黑盒查询。它利用了许多数据属性方法中存在的归纳偏差--离群值数据点更有可能具有影响力--并使用对抗性例子来生成被操纵的数据集。实验表明，在图像分类和文本生成任务中，阴影攻击可以将基于数据属性的补偿膨胀至少200%，而离群点攻击可以实现185%到高达643%的补偿膨胀。



## **44. A Framework for Differential Privacy Against Timing Attacks**

针对时间攻击的差异隐私框架 cs.CR

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.05623v1) [paper-pdf](http://arxiv.org/pdf/2409.05623v1)

**Authors**: Zachary Ratliff, Salil Vadhan

**Abstract**: The standard definition of differential privacy (DP) ensures that a mechanism's output distribution on adjacent datasets is indistinguishable. However, real-world implementations of DP can, and often do, reveal information through their runtime distributions, making them susceptible to timing attacks. In this work, we establish a general framework for ensuring differential privacy in the presence of timing side channels. We define a new notion of timing privacy, which captures programs that remain differentially private to an adversary that observes the program's runtime in addition to the output. Our framework enables chaining together component programs that are timing-stable followed by a random delay to obtain DP programs that achieve timing privacy. Importantly, our definitions allow for measuring timing privacy and output privacy using different privacy measures. We illustrate how to instantiate our framework by giving programs for standard DP computations in the RAM and Word RAM models of computation. Furthermore, we show how our framework can be realized in code through a natural extension of the OpenDP Programming Framework.

摘要: 差分隐私(DP)的标准定义确保了机制在相邻数据集上的输出分布是不可区分的。然而，DP的真实实现可以而且经常通过它们的运行时分发泄露信息，从而使它们容易受到计时攻击。在这项工作中，我们建立了一个通用的框架，以确保在存在定时侧信道的情况下的差异隐私。我们定义了一个新的时间隐私的概念，它捕获了对对手保持不同隐私的程序，除了输出之外，还观察程序的运行时。我们的框架允许将定时稳定的组件程序链接在一起，然后是随机延迟，以获得实现定时隐私的DP程序。重要的是，我们的定义允许使用不同的隐私度量来测量定时隐私和输出隐私。我们通过给出计算的RAM和Word RAM模型中的标准DP计算程序来说明如何实例化我们的框架。此外，我们还展示了如何通过OpenDP编程框架的自然扩展在代码中实现我们的框架。



## **45. How adversarial attacks can disrupt seemingly stable accurate classifiers**

对抗性攻击如何扰乱看似稳定的准确分类器 cs.LG

11 pages, 8 figures, additional supplementary materials

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2309.03665v2) [paper-pdf](http://arxiv.org/pdf/2309.03665v2)

**Authors**: Oliver J. Sutton, Qinghua Zhou, Ivan Y. Tyukin, Alexander N. Gorban, Alexander Bastounis, Desmond J. Higham

**Abstract**: Adversarial attacks dramatically change the output of an otherwise accurate learning system using a seemingly inconsequential modification to a piece of input data. Paradoxically, empirical evidence indicates that even systems which are robust to large random perturbations of the input data remain susceptible to small, easily constructed, adversarial perturbations of their inputs. Here, we show that this may be seen as a fundamental feature of classifiers working with high dimensional input data. We introduce a simple generic and generalisable framework for which key behaviours observed in practical systems arise with high probability -- notably the simultaneous susceptibility of the (otherwise accurate) model to easily constructed adversarial attacks, and robustness to random perturbations of the input data. We confirm that the same phenomena are directly observed in practical neural networks trained on standard image classification problems, where even large additive random noise fails to trigger the adversarial instability of the network. A surprising takeaway is that even small margins separating a classifier's decision surface from training and testing data can hide adversarial susceptibility from being detected using randomly sampled perturbations. Counterintuitively, using additive noise during training or testing is therefore inefficient for eradicating or detecting adversarial examples, and more demanding adversarial training is required.

摘要: 对抗性攻击通过对一段输入数据进行看似无关紧要的修改，极大地改变了原本准确的学习系统的输出。矛盾的是，经验证据表明，即使是对输入数据的大随机扰动具有健壮性的系统，也仍然容易受到其输入的小的、容易构造的、对抗性的扰动。在这里，我们展示了这可以被视为使用高维输入数据的分类器的基本特征。我们引入了一个简单的通用和可推广的框架，对于该框架，在实际系统中观察到的关键行为以高概率出现-特别是(否则准确的)模型对容易构造的对抗性攻击的同时敏感性，以及对输入数据的随机扰动的稳健性。我们证实，在标准图像分类问题上训练的实际神经网络中也直接观察到了同样的现象，其中即使是较大的加性随机噪声也不能触发网络的对抗性不稳定性。令人惊讶的是，即使是将分类器的决策面与训练和测试数据分开的很小的边距，也可以隐藏对手的易感性，使其不会被随机抽样的扰动检测到。因此，与直觉相反的是，在训练或测试期间使用加性噪声对于消除或检测对抗性例子是低效的，并且需要更苛刻的对抗性训练。



## **46. Getting a-Round Guarantees: Floating-Point Attacks on Certified Robustness**

获得全面保证：对认证稳健性的浮点攻击 cs.CR

In Proceedings of the 2024 Workshop on Artificial Intelligence and  Security (AISec '24)

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2205.10159v5) [paper-pdf](http://arxiv.org/pdf/2205.10159v5)

**Authors**: Jiankai Jin, Olga Ohrimenko, Benjamin I. P. Rubinstein

**Abstract**: Adversarial examples pose a security risk as they can alter decisions of a machine learning classifier through slight input perturbations. Certified robustness has been proposed as a mitigation where given an input $\mathbf{x}$, a classifier returns a prediction and a certified radius $R$ with a provable guarantee that any perturbation to $\mathbf{x}$ with $R$-bounded norm will not alter the classifier's prediction. In this work, we show that these guarantees can be invalidated due to limitations of floating-point representation that cause rounding errors. We design a rounding search method that can efficiently exploit this vulnerability to find adversarial examples against state-of-the-art certifications in two threat models, that differ in how the norm of the perturbation is computed. We show that the attack can be carried out against linear classifiers that have exact certifiable guarantees and against neural networks that have conservative certifications. In the weak threat model, our experiments demonstrate attack success rates over 50% on random linear classifiers, up to 23% on the MNIST dataset for linear SVM, and up to 15% for a neural network. In the strong threat model, the success rates are lower but positive. The floating-point errors exploited by our attacks can range from small to large (e.g., $10^{-13}$ to $10^{3}$) - showing that even negligible errors can be systematically exploited to invalidate guarantees provided by certified robustness. Finally, we propose a formal mitigation approach based on rounded interval arithmetic, encouraging future implementations of robustness certificates to account for limitations of modern computing architecture to provide sound certifiable guarantees.

摘要: 对抗性的例子构成了安全风险，因为它们可以通过轻微的输入扰动来改变机器学习分类器的决定。证明的稳健性已经被提出作为一种缓解方法，其中给定一个输入$\mathbf{x}$，分类器返回一个预测和一个证明的半径$R$，并且可证明地保证，对$\mathbf{x}$的任何扰动都不会改变分类器的预测。在这项工作中，我们证明了这些保证可能会由于浮点表示的限制而失效，从而导致舍入误差。我们设计了一个四舍五入的搜索方法，可以有效地利用这个漏洞在两个威胁模型中找到针对最新认证的敌意例子，这两个威胁模型的扰动范数的计算方式不同。我们证明了该攻击可以针对具有精确可证明保证的线性分类器和具有保守认证的神经网络来执行。在弱威胁模型中，我们的实验表明，在随机线性分类器上的攻击成功率超过50%，线性支持向量机在MNIST数据集上的攻击成功率高达23%，而神经网络的攻击成功率高达15%。在强威胁模型中，成功率较低，但却是积极的。我们的攻击利用的浮点错误可以从小到大(例如，$10^{-13}$到$10^{3}$)-表明即使是可以忽略的错误也可以被系统地利用来使经证明的健壮性提供的保证无效。最后，我们提出了一种基于四舍五入区间算法的形式化缓解方法，鼓励未来实现健壮性证书，以解决现代计算体系结构的局限性，提供可靠的可证明保证。



## **47. Boosting Certificate Robustness for Time Series Classification with Efficient Self-Ensemble**

通过高效的自集成提高时间序列分类的证书稳健性 cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2409.02802v2) [paper-pdf](http://arxiv.org/pdf/2409.02802v2)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.

摘要: 最近，时间序列域中的对抗性稳健性问题引起了人们的广泛关注。然而，现有的防御机制仍然有限，对抗性训练是主要的方法，尽管它不提供理论上的保证。由于随机化平滑方法能够证明在$ell_p$-ball攻击下的健壮性半径的一个可证明的下界，所以它已经成为一种优秀的方法。认识到它的成功，时间序列领域的研究已经开始集中在这些方面。然而，现有的研究主要集中在时间序列预测，或在统计特征增强对时间序列分类具有非埃尔p稳健性的情况下。我们的综述发现，随机平滑在TSC中表现平平，难以对稳健性较差的数据集提供有效的保证。因此，我们提出了一种自集成方法，通过减小分类裕度的方差来提高预测标签的概率置信度下界，从而证明更大的半径。这种方法还解决了深层集成~(DE)的计算开销问题，同时保持了竞争力，在某些情况下，在健壮性方面优于它。理论分析和实验结果都验证了该方法的有效性，在稳健性测试中表现出了优于基线方法的性能。



## **48. A Study on Prompt Injection Attack Against LLM-Integrated Mobile Robotic Systems**

针对LLM集成移动机器人系统的即时注入攻击研究 cs.RO

**SubmitDate**: 2024-09-09    [abs](http://arxiv.org/abs/2408.03515v2) [paper-pdf](http://arxiv.org/pdf/2408.03515v2)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Braunl, Jin B. Hong

**Abstract**: The integration of Large Language Models (LLMs) like GPT-4o into robotic systems represents a significant advancement in embodied artificial intelligence. These models can process multi-modal prompts, enabling them to generate more context-aware responses. However, this integration is not without challenges. One of the primary concerns is the potential security risks associated with using LLMs in robotic navigation tasks. These tasks require precise and reliable responses to ensure safe and effective operation. Multi-modal prompts, while enhancing the robot's understanding, also introduce complexities that can be exploited maliciously. For instance, adversarial inputs designed to mislead the model can lead to incorrect or dangerous navigational decisions. This study investigates the impact of prompt injections on mobile robot performance in LLM-integrated systems and explores secure prompt strategies to mitigate these risks. Our findings demonstrate a substantial overall improvement of approximately 30.8% in both attack detection and system performance with the implementation of robust defence mechanisms, highlighting their critical role in enhancing security and reliability in mission-oriented tasks.

摘要: 将像GPT-40这样的大型语言模型(LLM)集成到机器人系统中，代表着体现的人工智能的重大进步。这些模型可以处理多模式提示，使它们能够生成更多情景感知响应。然而，这种整合并不是没有挑战。其中一个主要问题是在机器人导航任务中使用LLMS存在潜在的安全风险。这些任务需要准确可靠的反应，以确保安全有效的运行。多模式提示在增强机器人理解能力的同时，也引入了可能被恶意利用的复杂性。例如，旨在误导模型的对抗性输入可能导致错误或危险的导航决策。这项研究调查了快速注射对LLM集成系统中移动机器人性能的影响，并探索了安全的提示策略来缓解这些风险。我们的研究结果表明，随着强大的防御机制的实施，攻击检测和系统性能都有了大约30.8%的大幅整体改进，突出了它们在增强面向任务的任务的安全性和可靠性方面的关键作用。



## **49. Pseudorandom Permutations from Random Reversible Circuits**

随机可逆电路的伪随机排列 cs.CC

v3: fixed minor errors

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2404.14648v3) [paper-pdf](http://arxiv.org/pdf/2404.14648v3)

**Authors**: William He, Ryan O'Donnell

**Abstract**: We study pseudorandomness properties of permutations on $\{0,1\}^n$ computed by random circuits made from reversible $3$-bit gates (permutations on $\{0,1\}^3$). Our main result is that a random circuit of depth $n \cdot \tilde{O}(k^2)$, with each layer consisting of $\approx n/3$ random gates in a fixed nearest-neighbor architecture, yields almost $k$-wise independent permutations. The main technical component is showing that the Markov chain on $k$-tuples of $n$-bit strings induced by a single random $3$-bit nearest-neighbor gate has spectral gap at least $1/n \cdot \tilde{O}(k)$. This improves on the original work of Gowers [Gowers96], who showed a gap of $1/\mathrm{poly}(n,k)$ for one random gate (with non-neighboring inputs); and, on subsequent work [HMMR05,BH08] improving the gap to $\Omega(1/n^2k)$ in the same setting.   From the perspective of cryptography, our result can be seen as a particularly simple/practical block cipher construction that gives provable statistical security against attackers with access to $k$~input-output pairs within few rounds. We also show that the Luby--Rackoff construction of pseudorandom permutations from pseudorandom functions can be implemented with reversible circuits. From this, we make progress on the complexity of the Minimum Reversible Circuit Size Problem (MRCSP), showing that block ciphers of fixed polynomial size are computationally secure against arbitrary polynomial-time adversaries, assuming the existence of one-way functions (OWFs).

摘要: 我们研究了由可逆$3$位门($0，1^3$上的置换)构成的随机电路计算的$0，1^n上置换的伪随机性。我们的主要结果是，一个深度为$n\cot\tide{O}(k^2)$的随机电路，每一层由固定最近邻体系结构中的$\约n/3$随机门组成，产生几乎$k$方向的独立排列。主要的技术内容是证明了由单个随机的$3$比特最近邻门产生的$n$比特串的$k$-元组上的马尔可夫链至少有$1/n\cdot\tilde{O}(K)$。这比Gowers[Gowers96]的原始工作有所改进，Gowers[Gowers96]对一个随机门(具有非相邻输入)显示了$1/\mathm{pol}(n，k)$的差距；在随后的工作[HMMR05，BH08]中，在相同设置下将差距改进为$\Omega(1/n^2k)$。从密码学的角度来看，我们的结果可以看作是一种特别简单实用的分组密码构造，它提供了针对在几轮内访问$k$~输入输出对的攻击者的可证明的统计安全性。我们还证明了伪随机函数的伪随机置换的Luby-Rackoff构造可以用可逆电路实现。由此，我们在最小可逆电路大小问题(MRCSP)的复杂性方面取得了进展，表明在假设存在单向函数(OWF)的情况下，固定多项式大小的分组密码在计算上是安全的，可以抵抗任意多项式时间的攻击者。



## **50. PIP: Detecting Adversarial Examples in Large Vision-Language Models via Attention Patterns of Irrelevant Probe Questions**

PIP：通过不相关探索问题的注意力模式检测大型视觉语言模型中的对抗示例 cs.CV

Accepted by ACM Multimedia 2024 BNI track (Oral)

**SubmitDate**: 2024-09-08    [abs](http://arxiv.org/abs/2409.05076v1) [paper-pdf](http://arxiv.org/pdf/2409.05076v1)

**Authors**: Yudong Zhang, Ruobing Xie, Jiansheng Chen, Xingwu Sun, Yu Wang

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated their powerful multimodal capabilities. However, they also face serious safety problems, as adversaries can induce robustness issues in LVLMs through the use of well-designed adversarial examples. Therefore, LVLMs are in urgent need of detection tools for adversarial examples to prevent incorrect responses. In this work, we first discover that LVLMs exhibit regular attention patterns for clean images when presented with probe questions. We propose an unconventional method named PIP, which utilizes the attention patterns of one randomly selected irrelevant probe question (e.g., "Is there a clock?") to distinguish adversarial examples from clean examples. Regardless of the image to be tested and its corresponding question, PIP only needs to perform one additional inference of the image to be tested and the probe question, and then achieves successful detection of adversarial examples. Even under black-box attacks and open dataset scenarios, our PIP, coupled with a simple SVM, still achieves more than 98% recall and a precision of over 90%. Our PIP is the first attempt to detect adversarial attacks on LVLMs via simple irrelevant probe questions, shedding light on deeper understanding and introspection within LVLMs. The code is available at https://github.com/btzyd/pip.

摘要: 大型视觉语言模型(LVLM)已经展示了其强大的多通道能力。然而，它们也面临着严重的安全问题，因为攻击者可以通过使用设计良好的对抗性示例在LVLM中引发健壮性问题。因此，LVLMS迫切需要针对对抗性实例的检测工具来防止错误响应。在这项工作中，我们首先发现，当被呈现探索性问题时，LVLMS对干净的图像表现出规则的注意模式。我们提出了一种非常规的方法，称为PIP，它利用了一个随机选择的无关探测问题的注意模式(例如，“有时钟吗？”)区分敌意的例子和干净的例子。无论待测试图像及其对应的问题是什么，PIP只需要对待测试图像和探测问题进行一次额外的推理，即可实现对抗性实例的成功检测。即使在黑盒攻击和开放数据集场景下，我们的PIP结合简单的支持向量机，仍然可以达到98%以上的召回率和90%以上的准确率。我们的PIP是首次尝试通过简单的无关紧要的探索性问题来检测对LVLMS的敌意攻击，从而揭示了LVLMS内部更深层次的理解和反省。代码可在https://github.com/btzyd/pip.上获得



