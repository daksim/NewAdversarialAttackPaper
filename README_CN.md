# Latest Adversarial Attack Papers
**update at 2024-11-13 17:30:03**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

7 pages, 1 figure

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.08003v1) [paper-pdf](http://arxiv.org/pdf/2411.08003v1)

**Authors**: Manuel Cebrian, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation-presents significant challenges that are likely to grow in importance. We investigate this attribution problem using formal language theory, specifically language identification in the limit as introduced by Gold and extended by Angluin. By modeling LLM outputs as formal languages, we analyze whether finite text samples can uniquely pinpoint the originating model. Our results show that due to the non-identifiability of certain language classes, under some mild assumptions about overlapping outputs from fine-tuned models it is theoretically impossible to attribute outputs to specific LLMs with certainty. This holds also when accounting for expressivity limitations of Transformer architectures. Even with direct model access or comprehensive monitoring, significant computational hurdles impede attribution efforts. These findings highlight an urgent need for proactive measures to mitigate risks posed by adversarial LLM use as their influence continues to expand.

摘要: 将大型语言模型(LLM)的输出归因于敌对环境--如网络攻击和虚假信息--带来了重大挑战，而这些挑战的重要性可能会越来越大。我们使用形式化语言理论来研究这一归因问题，特别是Gold提出并由Anluin推广的极限语言识别问题。通过将LLM输出建模为形式语言，我们分析了有限文本样本是否能够唯一地定位原始模型。我们的结果表明，由于某些语言类别的不可识别性，在微调模型的输出重叠的一些温和假设下，理论上不可能确定地将输出归因于特定的LLM。当考虑到Transformer架构的表现力限制时，这也是成立的。即使有了直接的模型访问或全面的监测，重大的计算障碍也阻碍了归因努力。这些调查结果突出表明，迫切需要采取积极主动的措施，以减轻敌对使用LLM所带来的风险，因为它们的影响继续扩大。



## **2. IAE: Irony-based Adversarial Examples for Sentiment Analysis Systems**

IAE：情感分析系统的基于讽刺的对抗示例 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07850v1) [paper-pdf](http://arxiv.org/pdf/2411.07850v1)

**Authors**: Xiaoyin Yi, Jiacheng Huang

**Abstract**: Adversarial examples, which are inputs deliberately perturbed with imperceptible changes to induce model errors, have raised serious concerns for the reliability and security of deep neural networks (DNNs). While adversarial attacks have been extensively studied in continuous data domains such as images, the discrete nature of text presents unique challenges. In this paper, we propose Irony-based Adversarial Examples (IAE), a method that transforms straightforward sentences into ironic ones to create adversarial text. This approach exploits the rhetorical device of irony, where the intended meaning is opposite to the literal interpretation, requiring a deeper understanding of context to detect. The IAE method is particularly challenging due to the need to accurately locate evaluation words, substitute them with appropriate collocations, and expand the text with suitable ironic elements while maintaining semantic coherence. Our research makes the following key contributions: (1) We introduce IAE, a strategy for generating textual adversarial examples using irony. This method does not rely on pre-existing irony corpora, making it a versatile tool for creating adversarial text in various NLP tasks. (2) We demonstrate that the performance of several state-of-the-art deep learning models on sentiment analysis tasks significantly deteriorates when subjected to IAE attacks. This finding underscores the susceptibility of current NLP systems to adversarial manipulation through irony. (3) We compare the impact of IAE on human judgment versus NLP systems, revealing that humans are less susceptible to the effects of irony in text.

摘要: 对抗性的例子，即输入故意被不可察觉的变化扰动以引起模型错误，已经引起了对深度神经网络(DNN)的可靠性和安全性的严重关注。虽然对抗性攻击已经在图像等连续数据领域得到了广泛的研究，但文本的离散性质带来了独特的挑战。在本文中，我们提出了一种基于反讽的对抗性范例(IAE)，它是一种将直白的句子转换成反讽句子来创建对抗性文本的方法。这一方法利用了反讽的修辞手段，其意图与字面解释相反，需要对语境进行更深层次的理解才能发现。IAE方法特别具有挑战性，因为需要准确定位评价词，用适当的搭配取代它们，并在保持语义连贯的同时用合适的讽刺元素扩展文本。我们的研究取得了以下主要贡献：(1)介绍了IAE，这是一种使用反讽生成文本对抗性实例的策略。这种方法不依赖于预先存在的反讽语料库，使其成为在各种自然语言处理任务中创建敌意文本的通用工具。(2)研究表明，当情感分析任务受到IAE攻击时，几种最新的深度学习模型的性能会显著下降。这一发现突显了当前NLP系统通过反讽进行对抗性操纵的敏感性。(3)我们比较了IAE和NLP系统对人类判断的影响，发现人类不太容易受到语篇中反讽的影响。



## **3. Chain Association-based Attacking and Shielding Natural Language Processing Systems**

基于链关联的攻击和屏蔽自然语言处理系统 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07843v1) [paper-pdf](http://arxiv.org/pdf/2411.07843v1)

**Authors**: Jiacheng Huang, Long Chen

**Abstract**: Association as a gift enables people do not have to mention something in completely straightforward words and allows others to understand what they intend to refer to. In this paper, we propose a chain association-based adversarial attack against natural language processing systems, utilizing the comprehension gap between humans and machines. We first generate a chain association graph for Chinese characters based on the association paradigm for building search space of potential adversarial examples. Then, we introduce an discrete particle swarm optimization algorithm to search for the optimal adversarial examples. We conduct comprehensive experiments and show that advanced natural language processing models and applications, including large language models, are vulnerable to our attack, while humans appear good at understanding the perturbed text. We also explore two methods, including adversarial training and associative graph-based recovery, to shield systems from chain association-based attack. Since a few examples that use some derogatory terms, this paper contains materials that may be offensive or upsetting to some people.

摘要: 联想作为一种礼物，使人们不必用完全直截了当的语言来提及某事，并让其他人理解他们想指的是什么。本文利用人与机器之间的理解鸿沟，提出了一种基于链式联想的对抗性自然语言处理系统攻击方法。首先在联想范式的基础上生成汉字的链式联想图，构建潜在对抗性实例的搜索空间。然后，我们引入了离散粒子群优化算法来搜索最优的对抗性实例。我们进行了全面的实验，并表明高级自然语言处理模型和应用程序，包括大型语言模型，容易受到我们的攻击，而人类似乎很擅长理解受干扰的文本。我们还探索了两种方法，包括对抗性训练和基于联想图的恢复，以保护系统免受基于链关联的攻击。由于有几个例子使用了一些贬义性的术语，因此本文包含的材料可能会冒犯某些人或使某些人不安。



## **4. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2410.23091v3) [paper-pdf](http://arxiv.org/pdf/2410.23091v3)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。



## **5. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

17 pages, 13 figures

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2404.19287v3) [paper-pdf](http://arxiv.org/pdf/2404.19287v3)

**Authors**: Wanqi Zhou, Shuanghao Bai, Danilo P. Mandic, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP exhibit exceptional generalization across diverse downstream tasks. While recent studies reveal their vulnerability to adversarial attacks, research to date has primarily focused on enhancing the robustness of image encoders against image-based attacks, with defenses against text-based and multimodal attacks remaining largely unexplored. To this end, this work presents the first comprehensive study on improving the adversarial robustness of VLMs against attacks targeting image, text, and multimodal inputs. This is achieved by proposing multimodal contrastive adversarial training (MMCoA). Such an approach strengthens the robustness of both image and text encoders by aligning the clean text embeddings with adversarial image embeddings, and adversarial text embeddings with clean image embeddings. The robustness of the proposed MMCoA is examined against existing defense methods over image, text, and multimodal attacks on the CLIP model. Extensive experiments on 15 datasets across two tasks reveal the characteristics of different adversarial defense methods under distinct distribution shifts and dataset complexities across the three attack types. This paves the way for a unified framework of adversarial robustness against different modality attacks, opening up new possibilities for securing VLMs against multimodal attacks. The code is available at https://github.com/ElleZWQ/MMCoA.git.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在不同的下游任务中表现出非凡的通用性。虽然最近的研究揭示了它们对对手攻击的脆弱性，但到目前为止的研究主要集中在增强图像编码器对基于图像的攻击的稳健性上，对基于文本的攻击和多模式攻击的防御在很大程度上仍未被探索。为此，本文首次全面研究了如何提高VLMS对图像、文本和多模式输入的攻击健壮性。这是通过提出多模式对比对抗训练(MMCoA)来实现的。这种方法通过将干净的文本嵌入与对抗性的图像嵌入以及对抗性的文本嵌入与干净的图像嵌入对齐来增强图像和文本编码器的稳健性。针对已有的针对图像、文本和多模式攻击的防御方法，对提出的MMCoA算法的鲁棒性进行了测试。在两个任务的15个数据集上进行了大量的实验，揭示了三种攻击类型在不同的分布变化和数据集复杂性下不同的对抗防御方法的特点。这为对抗不同模式攻击的对抗健壮性的统一框架铺平了道路，为保护VLM免受多模式攻击开辟了新的可能性。代码可在https://github.com/ElleZWQ/MMCoA.git.上获得



## **6. Data-Driven Graph Switching for Cyber-Resilient Control in Microgrids**

数据驱动的图形交换用于微电网中的网络弹性控制 eess.SY

Accepted in IEEE Design Methodologies Conference (DMC) 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07686v1) [paper-pdf](http://arxiv.org/pdf/2411.07686v1)

**Authors**: Suman Rath, Subham Sahoo

**Abstract**: Distributed microgrids are conventionally dependent on communication networks to achieve secondary control objectives. This dependence makes them vulnerable to stealth data integrity attacks (DIAs) where adversaries may perform manipulations via infected transmitters and repeaters to jeopardize stability. This paper presents a physics-guided, supervised Artificial Neural Network (ANN)-based framework that identifies communication-level cyberattacks in microgrids by analyzing whether incoming measurements will cause abnormal behavior of the secondary control layer. If abnormalities are detected, an iteration through possible spanning tree graph topologies that can be used to fulfill secondary control objectives is done. Then, a communication network topology that would not create secondary control abnormalities is identified and enforced for maximum stability. By altering the communication graph topology, the framework eliminates the dependence of the secondary control layer on inputs from compromised cyber devices helping it achieve resilience without instability. Several case studies are provided showcasing the robustness of the framework against False Data Injections and repeater-level Man-in-the-Middle attacks. To understand practical feasibility, robustness is also verified against larger microgrid sizes and in the presence of varying noise levels. Our findings indicate that performance can be affected when attempting scalability in the presence of noise. However, the framework operates robustly in low-noise settings.

摘要: 传统上，分布式微电网依靠通信网络来实现二次控制目标。这种依赖使它们容易受到隐形数据完整性攻击(DIA)，攻击者可能会通过受感染的发射器和中继器执行操作，从而危及稳定性。提出了一种基于物理引导的有监督人工神经网络(ANN)框架，该框架通过分析输入测量是否会导致二次控制层的异常行为来识别微电网中的通信级网络攻击。如果检测到异常，则对可用于实现二级控制目标的可能的生成树图拓扑进行迭代。然后，识别并实施不会产生二次控制异常的通信网络拓扑以实现最大稳定性。通过改变通信图拓扑，该框架消除了二次控制层对来自受损网络设备的输入的依赖，帮助它实现了弹性，而不会出现不稳定。提供了几个案例研究，展示了该框架对虚假数据注入和中继器级别的中间人攻击的健壮性。为了了解实际可行性，还针对较大的微电网规模和不同的噪声水平验证了稳健性。我们的发现表明，在存在噪声的情况下尝试可伸缩性时，性能可能会受到影响。然而，该框架在低噪声设置下运行稳健。



## **7. A Survey on Adversarial Machine Learning for Code Data: Realistic Threats, Countermeasures, and Interpretations**

代码数据对抗性机器学习调查：现实威胁、对策和解释 cs.CR

Under a reviewing process since Sep. 3, 2024

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07597v1) [paper-pdf](http://arxiv.org/pdf/2411.07597v1)

**Authors**: Yulong Yang, Haoran Fan, Chenhao Lin, Qian Li, Zhengyu Zhao, Chao Shen, Xiaohong Guan

**Abstract**: Code Language Models (CLMs) have achieved tremendous progress in source code understanding and generation, leading to a significant increase in research interests focused on applying CLMs to real-world software engineering tasks in recent years. However, in realistic scenarios, CLMs are exposed to potential malicious adversaries, bringing risks to the confidentiality, integrity, and availability of CLM systems. Despite these risks, a comprehensive analysis of the security vulnerabilities of CLMs in the extremely adversarial environment has been lacking. To close this research gap, we categorize existing attack techniques into three types based on the CIA triad: poisoning attacks (integrity \& availability infringement), evasion attacks (integrity infringement), and privacy attacks (confidentiality infringement). We have collected so far the most comprehensive (79) papers related to adversarial machine learning for CLM from the research fields of artificial intelligence, computer security, and software engineering. Our analysis covers each type of risk, examining threat model categorization, attack techniques, and countermeasures, while also introducing novel perspectives on eXplainable AI (XAI) and exploring the interconnections between different risks. Finally, we identify current challenges and future research opportunities. This study aims to provide a comprehensive roadmap for both researchers and practitioners and pave the way towards more reliable CLMs for practical applications.

摘要: 代码语言模型(CLMS)在源代码理解和生成方面取得了巨大的进步，导致近年来将代码语言模型应用于实际软件工程任务的研究兴趣显著增加。然而，在现实场景中，CLM暴露在潜在的恶意攻击者面前，给CLM系统的机密性、完整性和可用性带来了风险。尽管存在这些风险，但在极端敌对的环境中，缺乏对CLMS安全漏洞的全面分析。为了缩小这一研究空白，我们根据CIA三合会将现有的攻击技术分为三类：中毒攻击(完整性和可用性破坏)、逃避攻击(完整性破坏)和隐私攻击(保密侵犯)。到目前为止，我们已经从人工智能、计算机安全和软件工程的研究领域收集了与CLM的对抗性机器学习相关的最全面的(79)篇论文。我们的分析涵盖了每种类型的风险，研究了威胁模型分类、攻击技术和对策，同时也引入了关于可解释人工智能(XAI)的新视角，并探索了不同风险之间的相互联系。最后，我们确定了当前的挑战和未来的研究机会。这项研究旨在为研究人员和实践者提供一个全面的路线图，并为更可靠的CLMS的实际应用铺平道路。



## **8. Graph Agent Network: Empowering Nodes with Inference Capabilities for Adversarial Resilience**

图代理网络：赋予节点推理能力以对抗复原力 cs.LG

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2306.06909v4) [paper-pdf](http://arxiv.org/pdf/2306.06909v4)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Guangquan Xu, Pan Zhou, Wengang Ma, Hanyuan Huang

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **9. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

快速抢占：前向-后向级联学习，实现高效且可转移的主动对抗防御 cs.CR

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2407.15524v4) [paper-pdf](http://arxiv.org/pdf/2407.15524v4)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy due to its sensitivity to adversarial attacks. Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, existing anti-adversarial methods typically counteract adversarial perturbations post-attack, while we have devised a proactive strategy that preempts by safeguarding media upfront, effectively neutralizing potential adversarial effects before the third-party attacks occur. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first, to our knowledge, effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对对手攻击的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，现有的反对手方法通常在攻击后抵消对手干扰，而我们设计了一种主动战略，通过预先保护媒体来抢占先机，有效地在第三方攻击发生之前消除潜在的对手影响。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。据我们所知，我们还设计了第一个有效的白盒自适应恢复攻击，并证明了除非主干模型、算法和设置完全受损，否则我们的防御策略添加的保护是不可逆转的。这项工作为主动防御对抗性攻击提供了新的方向。



## **10. Rapid Response: Mitigating LLM Jailbreaks with a Few Examples**

快速反应：通过一些例子缓解LLM越狱 cs.CL

**SubmitDate**: 2024-11-12    [abs](http://arxiv.org/abs/2411.07494v1) [paper-pdf](http://arxiv.org/pdf/2411.07494v1)

**Authors**: Alwin Peng, Julian Michael, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: As large language models (LLMs) grow more powerful, ensuring their safety against misuse becomes crucial. While researchers have focused on developing robust defenses, no method has yet achieved complete invulnerability to attacks. We propose an alternative approach: instead of seeking perfect adversarial robustness, we develop rapid response techniques to look to block whole classes of jailbreaks after observing only a handful of attacks. To study this setting, we develop RapidResponseBench, a benchmark that measures a defense's robustness against various jailbreak strategies after adapting to a few observed examples. We evaluate five rapid response methods, all of which use jailbreak proliferation, where we automatically generate additional jailbreaks similar to the examples observed. Our strongest method, which fine-tunes an input classifier to block proliferated jailbreaks, reduces attack success rate by a factor greater than 240 on an in-distribution set of jailbreaks and a factor greater than 15 on an out-of-distribution set, having observed just one example of each jailbreaking strategy. Moreover, further studies suggest that the quality of proliferation model and number of proliferated examples play an key role in the effectiveness of this defense. Overall, our results highlight the potential of responding rapidly to novel jailbreaks to limit LLM misuse.

摘要: 随着大型语言模型(LLM)变得越来越强大，确保它们的安全性以防止误用变得至关重要。虽然研究人员专注于开发强大的防御系统，但还没有一种方法能够完全抵御攻击。我们提出了另一种方法：我们不是寻求完美的对手健壮性，而是开发快速响应技术，在仅观察到少数几次攻击后，寻求阻止整个类别的越狱。为了研究这种情况，我们开发了RapidResponseBch，这是一个基准，在适应了几个观察到的例子后，衡量了防御对各种越狱策略的健壮性。我们评估了五种快速响应方法，所有这些方法都使用越狱扩散，在这些方法中，我们自动生成与观察到的示例类似的额外越狱。我们最强大的方法是微调输入分类器以阻止越狱激增，在仅观察到每个越狱策略的一个示例后，在分布内越狱集合上将攻击成功率降低240倍以上，在分布外集合上降低15倍以上。此外，进一步的研究表明，扩散模型的质量和扩散实例的数量在这一防御措施的有效性中起着关键作用。总体而言，我们的结果突出了对新型越狱做出快速反应以限制LLM滥用的潜力。



## **11. DrAttack: Prompt Decomposition and Reconstruction Makes Powerful LLM Jailbreakers**

DrAttack：强大的快速分解和重建让LLM越狱者 cs.CR

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.16914v3) [paper-pdf](http://arxiv.org/pdf/2402.16914v3)

**Authors**: Xirui Li, Ruochen Wang, Minhao Cheng, Tianyi Zhou, Cho-Jui Hsieh

**Abstract**: The safety alignment of Large Language Models (LLMs) is vulnerable to both manual and automated jailbreak attacks, which adversarially trigger LLMs to output harmful content. However, current methods for jailbreaking LLMs, which nest entire harmful prompts, are not effective at concealing malicious intent and can be easily identified and rejected by well-aligned LLMs. This paper discovers that decomposing a malicious prompt into separated sub-prompts can effectively obscure its underlying malicious intent by presenting it in a fragmented, less detectable form, thereby addressing these limitations. We introduce an automatic prompt \textbf{D}ecomposition and \textbf{R}econstruction framework for jailbreak \textbf{Attack} (DrAttack). DrAttack includes three key components: (a) `Decomposition' of the original prompt into sub-prompts, (b) `Reconstruction' of these sub-prompts implicitly by in-context learning with semantically similar but harmless reassembling demo, and (c) a `Synonym Search' of sub-prompts, aiming to find sub-prompts' synonyms that maintain the original intent while jailbreaking LLMs. An extensive empirical study across multiple open-source and closed-source LLMs demonstrates that, with a significantly reduced number of queries, DrAttack obtains a substantial gain of success rate over prior SOTA prompt-only attackers. Notably, the success rate of 78.0\% on GPT-4 with merely 15 queries surpassed previous art by 33.1\%. The project is available at https://github.com/xirui-li/DrAttack.

摘要: 大型语言模型(LLM)的安全一致性容易受到手动和自动越狱攻击的攻击，这些攻击会相反地触发LLM输出有害内容。然而，当前的越狱LLM方法嵌套了整个有害的提示，在隐藏恶意意图方面并不有效，很容易被排列良好的LLM识别和拒绝。本文发现，通过将恶意提示分解为单独的子提示，可以通过以零散的、较难检测的形式来表示恶意提示，从而有效地掩盖潜在的恶意意图，从而解决这些限制。介绍了一种自动提示的文本bf{D}分解和文本bf{R}重构框架(DrAttack)。DrAttack包括三个关键组件：(A)将原始提示‘分解’成子提示，(B)通过使用语义相似但无害的重组演示的上下文学习隐式地‘重构’这些子提示，以及(C)对子提示进行‘同步搜索’，目的是在越狱LLM的同时找到保持原意的子提示的同义词。一项针对多个开源和闭源LLM的广泛经验研究表明，DrAttack在查询次数显著减少的情况下，与之前的Sota仅提示攻击者相比，获得了显著的成功率。值得注意的是，仅用15个查询在GPT-4上的成功率为78.0\%，比以前的ART高出33.1\%。该项目的网址为：https://github.com/xirui-li/DrAttack.。



## **12. The Inherent Adversarial Robustness of Analog In-Memory Computing**

模拟内存计算固有的对抗鲁棒性 cs.ET

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.07023v1) [paper-pdf](http://arxiv.org/pdf/2411.07023v1)

**Authors**: Corey Lammie, Julian Büchel, Athanasios Vasilopoulos, Manuel Le Gallo, Abu Sebastian

**Abstract**: A key challenge for Deep Neural Network (DNN) algorithms is their vulnerability to adversarial attacks. Inherently non-deterministic compute substrates, such as those based on Analog In-Memory Computing (AIMC), have been speculated to provide significant adversarial robustness when performing DNN inference. In this paper, we experimentally validate this conjecture for the first time on an AIMC chip based on Phase Change Memory (PCM) devices. We demonstrate higher adversarial robustness against different types of adversarial attacks when implementing an image classification network. Additional robustness is also observed when performing hardware-in-the-loop attacks, for which the attacker is assumed to have full access to the hardware. A careful study of the various noise sources indicate that a combination of stochastic noise sources (both recurrent and non-recurrent) are responsible for the adversarial robustness and that their type and magnitude disproportionately effects this property. Finally, it is demonstrated, via simulations, that when a much larger transformer network is used to implement a Natural Language Processing (NLP) task, additional robustness is still observed.

摘要: 深度神经网络(DNN)算法面临的一个关键挑战是它们对对手攻击的脆弱性。固有的非确定性计算基板，例如基于模拟内存计算(AIMC)的基板，被推测在执行DNN推理时提供显著的对抗性健壮性。在本文中，我们首次在基于相变存储(PCM)器件的AIMC芯片上实验验证了这一猜想。在实现图像分类网络时，我们表现出对不同类型的对抗性攻击的更高的对抗性鲁棒性。在执行硬件在环攻击时，还可以观察到额外的稳健性，假定攻击者对硬件具有完全访问权限。对各种噪声源的仔细研究表明，随机噪声源(循环和非循环)的组合是造成对抗鲁棒性的原因，并且它们的类型和大小不成比例地影响这一特性。最后，通过仿真证明，当使用更大的变压器网络来实现自然语言处理(NLP)任务时，仍然可以观察到额外的稳健性。



## **13. Computable Model-Independent Bounds for Adversarial Quantum Machine Learning**

对抗性量子机器学习的可计算模型独立边界 cs.LG

21 pages, 9 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06863v1) [paper-pdf](http://arxiv.org/pdf/2411.06863v1)

**Authors**: Bacui Li, Tansu Alpcan, Chandra Thapa, Udaya Parampalli

**Abstract**: By leveraging the principles of quantum mechanics, QML opens doors to novel approaches in machine learning and offers potential speedup. However, machine learning models are well-documented to be vulnerable to malicious manipulations, and this susceptibility extends to the models of QML. This situation necessitates a thorough understanding of QML's resilience against adversarial attacks, particularly in an era where quantum computing capabilities are expanding. In this regard, this paper examines model-independent bounds on adversarial performance for QML. To the best of our knowledge, we introduce the first computation of an approximate lower bound for adversarial error when evaluating model resilience against sophisticated quantum-based adversarial attacks. Experimental results are compared to the computed bound, demonstrating the potential of QML models to achieve high robustness. In the best case, the experimental error is only 10% above the estimated bound, offering evidence of the inherent robustness of quantum models. This work not only advances our theoretical understanding of quantum model resilience but also provides a precise reference bound for the future development of robust QML algorithms.

摘要: 通过利用量子力学的原理，QML为机器学习中的新方法打开了大门，并提供了潜在的加速比。然而，机器学习模型很容易受到恶意操作，这种易感性延伸到QML模型。这种情况需要彻底了解QML对对手攻击的韧性，特别是在量子计算能力不断扩大的时代。在这方面，本文研究了QML对抗性能的与模型无关的界。据我们所知，在评估模型对复杂的基于量子的敌意攻击的弹性时，我们引入了对抗性错误的近似下界的第一次计算。实验结果与计算界进行了比较，证明了QML模型具有较高的鲁棒性。在最好的情况下，实验误差仅比估计值高出10%，这为量子模型的内在稳健性提供了证据。这项工作不仅加深了我们对量子模型弹性的理论理解，而且为未来健壮的QML算法的发展提供了一个精确的参考界。



## **14. Boosting the Targeted Transferability of Adversarial Examples via Salient Region & Weighted Feature Drop**

通过显著区域和加权特征下降提高对抗性示例的目标可移植性 cs.IR

9 pages

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06784v1) [paper-pdf](http://arxiv.org/pdf/2411.06784v1)

**Authors**: Shanjun Xu, Linghui Li, Kaiguo Yuan, Bingyu Li

**Abstract**: Deep neural networks can be vulnerable to adversarially crafted examples, presenting significant risks to practical applications. A prevalent approach for adversarial attacks relies on the transferability of adversarial examples, which are generated from a substitute model and leveraged to attack unknown black-box models. Despite various proposals aimed at improving transferability, the success of these attacks in targeted black-box scenarios is often hindered by the tendency for adversarial examples to overfit to the surrogate models. In this paper, we introduce a novel framework based on Salient region & Weighted Feature Drop (SWFD) designed to enhance the targeted transferability of adversarial examples. Drawing from the observation that examples with higher transferability exhibit smoother distributions in the deep-layer outputs, we propose the weighted feature drop mechanism to modulate activation values according to weights scaled by norm distribution, effectively addressing the overfitting issue when generating adversarial examples. Additionally, by leveraging salient region within the image to construct auxiliary images, our method enables the adversarial example's features to be transferred to the target category in a model-agnostic manner, thereby enhancing the transferability. Comprehensive experiments confirm that our approach outperforms state-of-the-art methods across diverse configurations. On average, the proposed SWFD raises the attack success rate for normally trained models and robust models by 16.31% and 7.06% respectively.

摘要: 深度神经网络可能容易受到恶意构建的示例的攻击，从而给实际应用带来重大风险。对抗性攻击的一种普遍方法依赖于对抗性例子的可转移性，这些对抗性例子由替代模型生成并被用来攻击未知的黑盒模型。尽管提出了各种旨在提高可转移性的建议，但这些攻击在有针对性的黑盒场景中的成功往往受到对抗性例子过度适应代理模型的趋势的阻碍。在本文中，我们介绍了一种新的框架，该框架基于突出区域&加权特征丢弃(SWFD)，旨在增强对抗性例子的定向可转移性。根据可转移性较高的样本在深层输出中表现出更平滑的分布这一观察结果，我们提出了加权特征丢弃机制，根据范数分布衡量的权重来调整激活值，有效地解决了生成对抗性样本时的过拟合问题。此外，通过利用图像中的显著区域构造辅助图像，我们的方法能够以模型无关的方式将对抗性例子的特征转移到目标类别，从而增强了可转移性。综合实验证实，我们的方法在不同的配置上比最先进的方法性能更好。平均而言，SWFD使正常训练模型和稳健模型的攻击成功率分别提高了16.31%和7.06%。



## **15. Beyond Text: Utilizing Vocal Cues to Improve Decision Making in LLMs for Robot Navigation Tasks**

超越文本：利用人声线索改善LLM机器人导航任务的决策 cs.AI

30 pages, 7 figures

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2402.03494v3) [paper-pdf](http://arxiv.org/pdf/2402.03494v3)

**Authors**: Xingpeng Sun, Haoming Meng, Souradip Chakraborty, Amrit Singh Bedi, Aniket Bera

**Abstract**: While LLMs excel in processing text in these human conversations, they struggle with the nuances of verbal instructions in scenarios like social navigation, where ambiguity and uncertainty can erode trust in robotic and other AI systems. We can address this shortcoming by moving beyond text and additionally focusing on the paralinguistic features of these audio responses. These features are the aspects of spoken communication that do not involve the literal wording (lexical content) but convey meaning and nuance through how something is said. We present Beyond Text: an approach that improves LLM decision-making by integrating audio transcription along with a subsection of these features, which focus on the affect and more relevant in human-robot conversations.This approach not only achieves a 70.26% winning rate, outperforming existing LLMs by 22.16% to 48.30% (gemini-1.5-pro and gpt-3.5 respectively), but also enhances robustness against token manipulation adversarial attacks, highlighted by a 22.44% less decrease ratio than the text-only language model in winning rate. Beyond Text' marks an advancement in social robot navigation and broader Human-Robot interactions, seamlessly integrating text-based guidance with human-audio-informed language models.

摘要: 虽然LLM在处理这些人类对话中的文本方面表现出色，但它们在社交导航等场景中难以处理语言指令的细微差别，在这些场景中，模棱两可和不确定性可能会侵蚀人们对机器人和其他人工智能系统的信任。我们可以通过超越文本并另外关注这些音频反应的副语言特征来解决这一缺点。这些特征是口语交际的方面，不涉及字面上的措辞(词汇内容)，但通过说话方式传达意义和细微差别。我们提出了Beyond Text：一种改进LLM决策的方法，它集成了音频转录和这些特征的一部分，这些特征集中在人-机器人对话中的影响和更相关的方面。该方法不仅获得了70.26%的优胜率，比现有的LLM分别提高了22.16%到48.30%(分别为Gemini-1.5-Pro和GPT-3.5)，而且还增强了对令牌操纵对手攻击的鲁棒性，其优胜率比纯文本语言模型降低了22.44%。Beyond Text‘标志着社交机器人导航和更广泛的人-机器人交互方面的进步，无缝地将基于文本的指导与人-音频信息语言模型相结合。



## **16. Adversarial Detection with a Dynamically Stable System**

具有动态稳定系统的对抗性检测 cs.AI

**SubmitDate**: 2024-11-11    [abs](http://arxiv.org/abs/2411.06666v1) [paper-pdf](http://arxiv.org/pdf/2411.06666v1)

**Authors**: Xiaowei Long, Jie Lin, Xiangyuan Yang

**Abstract**: Adversarial detection is designed to identify and reject maliciously crafted adversarial examples(AEs) which are generated to disrupt the classification of target models.   Presently, various input transformation-based methods have been developed on adversarial example detection, which typically rely on empirical experience and lead to unreliability against new attacks.   To address this issue, we propose and conduct a Dynamically Stable System (DSS), which can effectively detect the adversarial examples from normal examples according to the stability of input examples.   Particularly, in our paper, the generation of adversarial examples is considered as the perturbation process of a Lyapunov dynamic system, and we propose an example stability mechanism, in which a novel control term is added in adversarial example generation to ensure that the normal examples can achieve dynamic stability while the adversarial examples cannot achieve the stability.   Then, based on the proposed example stability mechanism, a Dynamically Stable System (DSS) is proposed, which can utilize the disruption and restoration actions to determine the stability of input examples and detect the adversarial examples through changes in the stability of the input examples.   In comparison with existing methods in three benchmark datasets(MNIST, CIFAR10, and CIFAR100), our evaluation results show that our proposed DSS can achieve ROC-AUC values of 99.83%, 97.81% and 94.47%, surpassing the state-of-the-art(SOTA) values of 97.35%, 91.10% and 93.49% in the other 7 methods.

摘要: 敌意检测旨在识别和拒绝恶意构建的敌意示例(AE)，这些AE是为了扰乱目标模型的分类而生成的。目前，已有多种基于输入变换的对抗性样本检测方法，这些方法通常依赖于经验，对新的攻击不可靠。针对这一问题，我们提出并实现了一个动态稳定系统(DSS)，该系统能够根据输入样本的稳定性有效地从正常样本中检测出敌意样本。特别地，本文将对抗性实例的生成看作是一个Lyapunov动态系统的摄动过程，并提出了一种实例稳定机制，在对抗性实例生成过程中增加了一个新的控制项，以保证正常实例能够实现动态稳定，而对抗性实例不能实现稳定性。然后，基于所提出的实例稳定机制，提出了一个动态稳定系统(DSS)，该系统可以利用中断和恢复行为来确定输入实例的稳定性，并通过输入实例稳定性的变化来检测敌意实例。在三个基准数据集(MNIST、CIFAR10和CIFAR100)上的评估结果表明，我们提出的决策支持系统ROC-AUC值分别为99.83%、97.81%和94.47%，超过了其他7种方法的ROC-AUC值97.35%、91.10%和93.49%。



## **17. Do Unlearning Methods Remove Information from Language Model Weights?**

取消学习方法会从语言模型权重中删除信息吗？ cs.LG

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2410.08827v2) [paper-pdf](http://arxiv.org/pdf/2410.08827v2)

**Authors**: Aghyad Deeb, Fabien Roger

**Abstract**: Large Language Models' knowledge of how to perform cyber-security attacks, create bioweapons, and manipulate humans poses risks of misuse. Previous work has proposed methods to unlearn this knowledge. Historically, it has been unclear whether unlearning techniques are removing information from the model weights or just making it harder to access. To disentangle these two objectives, we propose an adversarial evaluation method to test for the removal of information from model weights: we give an attacker access to some facts that were supposed to be removed, and using those, the attacker tries to recover other facts from the same distribution that cannot be guessed from the accessible facts. We show that using fine-tuning on the accessible facts can recover 88% of the pre-unlearning accuracy when applied to current unlearning methods, revealing the limitations of these methods in removing information from the model weights.

摘要: 大型语言模型关于如何执行网络安全攻击、制造生物武器和操纵人类的知识带来了滥用的风险。之前的工作提出了忘记这些知识的方法。从历史上看，目前尚不清楚取消学习技术是否正在从模型权重中删除信息，或者只是使其更难访问。为了解开这两个目标，我们提出了一种对抗评估方法来测试从模型权重中删除信息的情况：我们让攻击者访问一些应该删除的事实，并使用这些事实，攻击者试图从无法从可访问的事实中猜测到的相同分布中恢复其他事实。我们表明，当应用于当前的取消学习方法时，对可访问的事实进行微调可以恢复取消学习前的88%的准确性，揭示了这些方法在从模型权重中删除信息方面的局限性。



## **18. HidePrint: Hiding the Radio Fingerprint via Random Noise**

HidePrint：通过随机噪音隐藏无线电指纹 cs.CR

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2411.06417v1) [paper-pdf](http://arxiv.org/pdf/2411.06417v1)

**Authors**: Gabriele Oligeri, Savio Sciancalepore

**Abstract**: Radio Frequency Fingerprinting (RFF) techniques allow a receiver to authenticate a transmitter by analyzing the physical layer of the radio spectrum. Although the vast majority of scientific contributions focus on improving the performance of RFF considering different parameters and scenarios, in this work, we consider RFF as an attack vector to identify and track a target device.   We propose, implement, and evaluate HidePrint, a solution to prevent tracking through RFF without affecting the quality of the communication link between the transmitter and the receiver. HidePrint hides the transmitter's fingerprint against an illegitimate eavesdropper by injecting controlled noise in the transmitted signal. We evaluate our solution against state-of-the-art image-based RFF techniques considering different adversarial models, different communication links (wired and wireless), and different configurations. Our results show that the injection of a Gaussian noise pattern with a standard deviation of (at least) 0.02 prevents device fingerprinting in all the considered scenarios, thus making the performance of the identification process indistinguishable from the random guess while affecting the Signal-to-Noise Ratio (SNR) of the received signal by only 0.1 dB. Moreover, we introduce selective radio fingerprint disclosure, a new technique that allows the transmitter to disclose the radio fingerprint to only a subset of intended receivers. This technique allows the transmitter to regain anonymity, thus preventing identification and tracking while allowing authorized receivers to authenticate the transmitter without affecting the quality of the transmitted signal.

摘要: 射频指纹(RFF)技术允许接收器通过分析无线电频谱的物理层来验证发射器。虽然绝大多数的科学贡献都集中在考虑不同参数和场景的情况下提高RFF的性能，但在这项工作中，我们将RFF视为识别和跟踪目标设备的攻击矢量。我们提出、实现和评估了HidePrint，这是一种在不影响发送器和接收器之间的通信链路质量的情况下防止通过RFF进行跟踪的解决方案。HidePrint通过在传输的信号中注入受控噪声来隐藏发射器的指纹，以防止非法窃听者。我们针对最先进的基于图像的RFF技术对我们的解决方案进行了评估，考虑了不同的对抗模型、不同的通信链路(有线和无线)和不同的配置。我们的结果表明，在所有考虑的场景中，注入标准差为(至少)0.02的高斯噪声模式防止了设备指纹识别，从而使识别过程的性能与随机猜测难以区分，而对接收信号的信噪比(SNR)的影响仅为0.1dB。此外，我们引入了选择性无线电指纹披露，这是一种新的技术，允许发射机只向目标接收者的子集披露无线电指纹。该技术允许发射机重新获得匿名性，从而防止识别和跟踪，同时允许授权的接收机在不影响传输信号质量的情况下认证发射机。



## **19. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

随机消息拦截平滑：图神经网络的灰箱证书 cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2024-11-10    [abs](http://arxiv.org/abs/2301.02039v2) [paper-pdf](http://arxiv.org/pdf/2301.02039v2)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.

摘要: 随机化平滑是证明机器学习模型(包括图神经网络)对抗稳健性的最有前途的框架之一。然而，现有的用于GNN的随机化平滑证书过于悲观，因为它们将模型视为黑匣子，忽略了底层架构。为了解决这个问题，我们提出了一种新的灰盒证书，它利用了GNN的消息传递原理：我们随机截获消息，并仔细分析来自恶意控制节点的消息到达目标节点的概率。与现有的证书相比，我们证明了对控制图中的整个节点并可以任意操纵节点特征的更强大的攻击者的健壮性。我们的证书为更远距离的攻击提供了更强有力的保证，因为来自较远节点的消息更有可能被拦截。我们在不同的模型和数据集上演示了我们的方法的有效性。由于我们的灰盒证书考虑了底层的图结构，所以我们可以通过应用图稀疏来显著提高可证明的健壮性。



## **20. Robust Detection of LLM-Generated Text: A Comparative Analysis**

LLM生成文本的稳健检测：比较分析 cs.CL

8 pages

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06248v1) [paper-pdf](http://arxiv.org/pdf/2411.06248v1)

**Authors**: Yongye Su, Yuqing Wu

**Abstract**: The ability of large language models to generate complex texts allows them to be widely integrated into many aspects of life, and their output can quickly fill all network resources. As the impact of LLMs grows, it becomes increasingly important to develop powerful detectors for the generated text. This detector is essential to prevent the potential misuse of these technologies and to protect areas such as social media from the negative effects of false content generated by LLMS. The main goal of LLM-generated text detection is to determine whether text is generated by an LLM, which is a basic binary classification task. In our work, we mainly use three different classification methods based on open source datasets: traditional machine learning techniques such as logistic regression, k-means clustering, Gaussian Naive Bayes, support vector machines, and methods based on converters such as BERT, and finally algorithms that use LLMs to detect LLM-generated text. We focus on model generalization, potential adversarial attacks, and accuracy of model evaluation. Finally, the possible research direction in the future is proposed, and the current experimental results are summarized.

摘要: 大型语言模型生成复杂文本的能力使它们能够广泛融入生活的许多方面，它们的输出可以迅速填满所有网络资源。随着LLMS的影响越来越大，为生成的文本开发强大的检测器变得越来越重要。这种检测器对于防止这些技术的潜在滥用以及保护社交媒体等领域免受LLMS产生的虚假内容的负面影响至关重要。LLM生成的文本检测的主要目标是确定文本是否由LLM生成，这是一项基本的二进制分类任务。在我们的工作中，我们主要使用了三种不同的基于开源数据集的分类方法：传统的机器学习技术，如Logistic回归，k-均值聚类，高斯朴素贝叶斯，支持向量机，以及基于转换器的方法，如BERT，最后是使用LLMS来检测LLM生成的文本的算法。我们主要关注模型的泛化、潜在的敌意攻击和模型评估的准确性。最后，提出了未来可能的研究方向，并对目前的实验结果进行了总结。



## **21. Target-driven Attack for Large Language Models**

针对大型语言模型的目标驱动攻击 cs.CL

12 pages, 7 figures. arXiv admin note: substantial text overlap with  arXiv:2404.07234

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.07268v1) [paper-pdf](http://arxiv.org/pdf/2411.07268v1)

**Authors**: Chong Zhang, Mingyu Jin, Dong Shu, Taowen Wang, Dongfang Liu, Xiaobo Jin

**Abstract**: Current large language models (LLM) provide a strong foundation for large-scale user-oriented natural language tasks. Many users can easily inject adversarial text or instructions through the user interface, thus causing LLM model security challenges like the language model not giving the correct answer. Although there is currently a large amount of research on black-box attacks, most of these black-box attacks use random and heuristic strategies. It is unclear how these strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we propose our target-driven black-box attack method to maximize the KL divergence between the conditional probabilities of the clean text and the attack text to redefine the attack's goal. We transform the distance maximization problem into two convex optimization problems based on the attack goal to solve the attack text and estimate the covariance. Furthermore, the projected gradient descent algorithm solves the vector corresponding to the attack text. Our target-driven black-box attack approach includes two attack strategies: token manipulation and misinformation attack. Experimental results on multiple Large Language Models and datasets demonstrate the effectiveness of our attack method.

摘要: 现有的大型语言模型(LLM)为大规模面向用户的自然语言任务提供了坚实的基础。许多用户可以很容易地通过用户界面注入敌意文本或指令，从而导致LLM模型的安全挑战，如语言模型无法给出正确的答案。虽然目前有大量关于黑盒攻击的研究，但这些黑盒攻击大多采用随机和启发式策略。目前尚不清楚这些策略如何与攻击成功率相关，从而有效地提高模型的健壮性。为了解决这一问题，我们提出了目标驱动的黑盒攻击方法，以最大化明文和攻击文本的条件概率之间的KL偏差，从而重新定义攻击的目标。将距离最大化问题转化为基于攻击目标的两个凸优化问题来求解攻击文本并估计协方差。此外，投影梯度下降算法求解与攻击文本对应的向量。我们的目标驱动的黑盒攻击方法包括两种攻击策略：令牌操纵和错误信息攻击。在多个大型语言模型和数据集上的实验结果证明了该攻击方法的有效性。



## **22. BM-PAW: A Profitable Mining Attack in the PoW-based Blockchain System**

BM-PAW：基于PoW的区块链系统中的有利可图的采矿攻击 cs.CR

21 pages, 4 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06187v1) [paper-pdf](http://arxiv.org/pdf/2411.06187v1)

**Authors**: Junjie Hu, Xunzhi Chen, Huan Yan, Na Ruan

**Abstract**: Mining attacks enable an adversary to procure a disproportionately large portion of mining rewards by deviating from honest mining practices within the PoW-based blockchain system. In this paper, we demonstrate that the security vulnerabilities of PoW-based blockchain extend beyond what these mining attacks initially reveal. We introduce a novel mining strategy, named BM-PAW, which yields superior rewards for both the attacker and the targeted pool compared to the state-of-the-art mining attack: PAW. Our analysis reveals that BM-PAW attackers are incentivized to offer appropriate bribe money to other targets, as they comply with the attacker's directives upon receiving payment. We find the BM-PAW attacker can circumvent the "miner's dilemma" through equilibrium analysis in a two-pool BM-PAW game scenario, wherein the outcome is determined by the attacker's mining power. We finally propose practical countermeasures to mitigate these novel pool attacks.

摘要: 采矿攻击使对手能够通过偏离基于PoW的区块链系统内的诚实采矿实践来获得不成比例的大部分采矿奖励。在本文中，我们证明了基于PoW的区块链的安全漏洞超出了这些采矿攻击最初揭示的范围。我们引入了一种名为BM-PAW的新型采矿策略，与最先进的采矿攻击PAW相比，该策略为攻击者和目标池提供了更高的回报。我们的分析表明，BM-PAW攻击者受到激励向其他目标提供适当的贿赂资金，因为他们在收到付款后遵守攻击者的指示。我们发现，BM-PAW攻击者可以通过两池BM-PAW游戏场景中的均衡分析来规避“矿工困境”，其中结果取决于攻击者的采矿能力。我们最后提出了实用的对策来减轻这些新型池攻击。



## **23. AI-Compass: A Comprehensive and Effective Multi-module Testing Tool for AI Systems**

AI-Compass：一款全面有效的人工智能系统多模块测试工具 cs.AI

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2411.06146v1) [paper-pdf](http://arxiv.org/pdf/2411.06146v1)

**Authors**: Zhiyu Zhu, Zhibo Jin, Hongsheng Hu, Minhui Xue, Ruoxi Sun, Seyit Camtepe, Praveen Gauravaram, Huaming Chen

**Abstract**: AI systems, in particular with deep learning techniques, have demonstrated superior performance for various real-world applications. Given the need for tailored optimization in specific scenarios, as well as the concerns related to the exploits of subsurface vulnerabilities, a more comprehensive and in-depth testing AI system becomes a pivotal topic. We have seen the emergence of testing tools in real-world applications that aim to expand testing capabilities. However, they often concentrate on ad-hoc tasks, rendering them unsuitable for simultaneously testing multiple aspects or components. Furthermore, trustworthiness issues arising from adversarial attacks and the challenge of interpreting deep learning models pose new challenges for developing more comprehensive and in-depth AI system testing tools. In this study, we design and implement a testing tool, \tool, to comprehensively and effectively evaluate AI systems. The tool extensively assesses multiple measurements towards adversarial robustness, model interpretability, and performs neuron analysis. The feasibility of the proposed testing tool is thoroughly validated across various modalities, including image classification, object detection, and text classification. Extensive experiments demonstrate that \tool is the state-of-the-art tool for a comprehensive assessment of the robustness and trustworthiness of AI systems. Our research sheds light on a general solution for AI systems testing landscape.

摘要: 人工智能系统，特别是具有深度学习技术的系统，在各种现实世界的应用中表现出了优越的性能。鉴于在特定场景中需要量身定做的优化，以及与地下漏洞利用相关的担忧，更全面和深入的测试人工智能系统成为一个关键话题。我们已经看到，在真实世界的应用程序中出现了旨在扩展测试能力的测试工具。然而，它们通常专注于特别任务，使得它们不适合同时测试多个方面或组件。此外，对抗性攻击产生的可信性问题以及解释深度学习模型的挑战为开发更全面和深入的AI系统测试工具提出了新的挑战。在这项研究中，我们设计并实现了一个测试工具，\Tool，以全面有效地评估AI系统。该工具广泛评估对抗性稳健性、模型可解释性的多个测量，并执行神经元分析。所提出的测试工具的可行性在包括图像分类、目标检测和文本分类在内的各种模式上得到了彻底的验证。大量的实验表明，该工具是全面评估人工智能系统健壮性和可信性的最先进的工具。我们的研究为人工智能系统测试领域提供了一个通用的解决方案。



## **24. Robust Graph Neural Networks via Unbiased Aggregation**

通过无偏聚集的鲁棒图神经网络 cs.LG

NeurIPS 2024 poster. 28 pages, 14 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2311.14934v2) [paper-pdf](http://arxiv.org/pdf/2311.14934v2)

**Authors**: Zhichao Hou, Ruiqi Feng, Tyler Derr, Xiaorui Liu

**Abstract**: The adversarial robustness of Graph Neural Networks (GNNs) has been questioned due to the false sense of security uncovered by strong adaptive attacks despite the existence of numerous defenses. In this work, we delve into the robustness analysis of representative robust GNNs and provide a unified robust estimation point of view to understand their robustness and limitations. Our novel analysis of estimation bias motivates the design of a robust and unbiased graph signal estimator. We then develop an efficient Quasi-Newton Iterative Reweighted Least Squares algorithm to solve the estimation problem, which is unfolded as robust unbiased aggregation layers in GNNs with theoretical guarantees. Our comprehensive experiments confirm the strong robustness of our proposed model under various scenarios, and the ablation study provides a deep understanding of its advantages. Our code is available at https://github.com/chris-hzc/RUNG.

摘要: 图形神经网络（GNN）的对抗鲁棒性受到质疑，因为尽管存在多种防御措施，但强自适应攻击却暴露了错误的安全感。在这项工作中，我们深入研究了代表性稳健GNN的稳健性分析，并提供统一的稳健性估计观点来了解其稳健性和局限性。我们对估计偏差的新颖分析激励了设计稳健且无偏的图信号估计器。然后，我们开发了一种高效的准牛顿迭代重加权最小平方算法来解决估计问题，该算法在理论保证的情况下被展开为GNN中的鲁棒无偏聚集层。我们全面的实验证实了我们提出的模型在各种场景下具有强大的鲁棒性，并且消融研究深入了解了其优势。我们的代码可在https://github.com/chris-hzc/RUNG上获取。



## **25. Goal-guided Generative Prompt Injection Attack on Large Language Models**

对大型语言模型的目标引导生成提示注入攻击 cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-11-09    [abs](http://arxiv.org/abs/2404.07234v4) [paper-pdf](http://arxiv.org/pdf/2404.07234v4)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.

摘要: 现有的大型语言模型为大规模面向用户的自然语言任务提供了坚实的基础。大量用户可以很容易地通过用户界面注入敌意文本或指令，从而造成LLMS模型的安全挑战。虽然目前有大量关于即时注入攻击的研究，但这些黑盒攻击大多采用启发式策略。目前尚不清楚这些启发式策略如何与攻击成功率相关，从而有效地提高模型的稳健性。为了解决这个问题，我们重新定义了攻击的目标：最大化纯文本和敌意文本的条件概率之间的KL偏差。此外，我们证明了当条件概率为高斯分布时，最大化KL发散度等价于最大化明文和敌意文本的嵌入表示$x$和$x‘$之间的马氏距离，并给出了关于$x$和$x’$的定量关系。然后，设计了一种简单有效的目标引导生成性提示注入策略(G2PIA)，找到满足特定约束的注入文本，以近似达到最优的攻击效果。特别值得注意的是，我们的攻击方法是一种计算代价低的无查询黑盒攻击方法。在7个LLM模型和4个数据集上的实验结果表明了该攻击方法的有效性。



## **26. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

走向更真实的提取攻击：对抗的角度 cs.CR

Presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2407.02596v2) [paper-pdf](http://arxiv.org/pdf/2407.02596v2)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing parts of their training data which makes them vulnerable to extraction attacks. Existing research often examines isolated setups--such as evaluating extraction risks from a single model or with a fixed prompt design. However, a real-world adversary could access models across various sizes and checkpoints, as well as exploit prompt sensitivity, resulting in a considerably larger attack surface than previously studied. In this paper, we revisit extraction attacks from an adversarial perspective, focusing on how to leverage the brittleness of language models and the multi-faceted access to the underlying data. We find significant churn in extraction trends, i.e., even unintuitive changes to the prompt, or targeting smaller models and earlier checkpoints, can extract distinct information. By combining information from multiple attacks, our adversary is able to increase the extraction risks by up to $2 \times$. Furthermore, even with mitigation strategies like data deduplication, we find the same escalation of extraction risks against a real-world adversary. We conclude with a set of case studies, including detecting pre-training data, copyright violations, and extracting personally identifiable information, showing how our more realistic adversary can outperform existing adversaries in the literature.

摘要: 语言模型容易记住其训练数据的一部分，这使得它们容易受到提取攻击。现有的研究经常考察孤立的设置--例如评估从单一模型或固定提示设计中提取的风险。然而，现实世界中的对手可以访问不同大小和检查点的模型，并利用即时敏感性，导致比之前研究的更大的攻击面。在本文中，我们从敌意的角度重新审视提取攻击，重点放在如何利用语言模型的脆弱性和对底层数据的多方面访问。我们发现提取趋势中存在显著的波动，即即使对提示进行了不直观的更改，或者针对较小的模型和较早的检查点，也可以提取不同的信息。通过组合来自多个攻击的信息，我们的对手能够将提取风险增加高达$2\x$。此外，即使使用重复数据删除等缓解策略，我们也发现现实世界中的对手面临同样的提取风险升级。我们以一组案例研究结束，包括检测训练前数据、侵犯版权和提取个人身份信息，展示我们更现实的对手如何超越文献中现有的对手。



## **27. A Survey of AI-Related Cyber Security Risks and Countermeasures in Mobility-as-a-Service**

移动即服务中人工智能相关网络安全风险及对策调查 cs.CR

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05681v1) [paper-pdf](http://arxiv.org/pdf/2411.05681v1)

**Authors**: Kai-Fung Chu, Haiyue Yuan, Jinsheng Yuan, Weisi Guo, Nazmiye Balta-Ozkan, Shujun Li

**Abstract**: Mobility-as-a-Service (MaaS) integrates different transport modalities and can support more personalisation of travellers' journey planning based on their individual preferences, behaviours and wishes. To fully achieve the potential of MaaS, a range of AI (including machine learning and data mining) algorithms are needed to learn personal requirements and needs, to optimise journey planning of each traveller and all travellers as a whole, to help transport service operators and relevant governmental bodies to operate and plan their services, and to detect and prevent cyber attacks from various threat actors including dishonest and malicious travellers and transport operators. The increasing use of different AI and data processing algorithms in both centralised and distributed settings opens the MaaS ecosystem up to diverse cyber and privacy attacks at both the AI algorithm level and the connectivity surfaces. In this paper, we present the first comprehensive review on the coupling between AI-driven MaaS design and the diverse cyber security challenges related to cyber attacks and countermeasures. In particular, we focus on how current and emerging AI-facilitated privacy risks (profiling, inference, and third-party threats) and adversarial AI attacks (evasion, extraction, and gamification) may impact the MaaS ecosystem. These risks often combine novel attacks (e.g., inverse learning) with traditional attack vectors (e.g., man-in-the-middle attacks), exacerbating the risks for the wider participation actors and the emergence of new business models.

摘要: 移动即服务(MaAS)集成了不同的交通方式，可以根据旅行者的个人偏好、行为和意愿支持更个性化的旅行计划。为了充分发挥MAAS的潜力，需要一系列人工智能(包括机器学习和数据挖掘)算法来了解个人需求和需求，优化每个旅行者和所有旅行者的旅行计划，帮助运输服务运营商和相关政府机构运营和规划他们的服务，以及检测和防止来自各种威胁参与者的网络攻击，包括不诚实和恶意的旅行者和运输运营商。在集中式和分布式环境中越来越多地使用不同的人工智能和数据处理算法，使MAAS生态系统在人工智能算法级别和连接面上都面临不同的网络和隐私攻击。在这篇文章中，我们首次全面回顾了人工智能驱动的MAAS设计与与网络攻击相关的各种网络安全挑战和对策之间的耦合。特别是，我们关注当前和正在出现的人工智能促进的隐私风险(剖析、推理和第三方威胁)和对抗性人工智能攻击(逃避、提取和游戏化)可能如何影响MAAS生态系统。这些风险通常将新的攻击(例如反向学习)与传统的攻击向量(例如中间人攻击)结合在一起，加剧了更广泛的参与者和新商业模式出现的风险。



## **28. DeepDRK: Deep Dependency Regularized Knockoff for Feature Selection**

DeepDRK：功能选择的深度依赖正规化仿制品 cs.LG

33 pages, 15 figures, 9 tables

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2402.17176v2) [paper-pdf](http://arxiv.org/pdf/2402.17176v2)

**Authors**: Hongyu Shen, Yici Yan, Zhizhen Zhao

**Abstract**: Model-X knockoff has garnered significant attention among various feature selection methods due to its guarantees for controlling the false discovery rate (FDR). Since its introduction in parametric design, knockoff techniques have evolved to handle arbitrary data distributions using deep learning-based generative models. However, we have observed limitations in the current implementations of the deep Model-X knockoff framework. Notably, the "swap property" that knockoffs require often faces challenges at the sample level, resulting in diminished selection power. To address these issues, we develop "Deep Dependency Regularized Knockoff (DeepDRK)," a distribution-free deep learning method that effectively balances FDR and power. In DeepDRK, we introduce a novel formulation of the knockoff model as a learning problem under multi-source adversarial attacks. By employing an innovative perturbation technique, we achieve lower FDR and higher power. Our model outperforms existing benchmarks across synthetic, semi-synthetic, and real-world datasets, particularly when sample sizes are small and data distributions are non-Gaussian.

摘要: 在各种特征选择方法中，Model-X假冒因其在控制错误发现率(FDR)方面的保证而受到广泛关注。自从它被引入到参数设计中以来，仿冒技术已经发展到使用基于深度学习的生成性模型来处理任意数据分布。然而，我们观察到深度Model-X仿冒框架的当前实现存在局限性。值得注意的是，仿冒品所需的“互换属性”经常在样本层面面临挑战，导致选择能力减弱。为了解决这些问题，我们开发了“深度依赖正规化仿冒(DeepDRK)”，这是一种无需分发的深度学习方法，可以有效地平衡FDR和功率。在DeepDRK中，我们引入了一种新的仿冒模型作为多源攻击下的学习问题。通过采用创新的微扰技术，我们实现了更低的FDR和更高的功率。我们的模型在合成、半合成和真实世界数据集上的表现优于现有基准，特别是在样本量较小且数据分布为非高斯的情况下。



## **29. Towards a Re-evaluation of Data Forging Attacks in Practice**

重新评估实践中的数据伪造攻击 cs.CR

18 pages

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05658v1) [paper-pdf](http://arxiv.org/pdf/2411.05658v1)

**Authors**: Mohamed Suliman, Anisa Halimi, Swanand Kadhe, Nathalie Baracaldo, Douglas Leith

**Abstract**: Data forging attacks provide counterfactual proof that a model was trained on a given dataset, when in fact, it was trained on another. These attacks work by forging (replacing) mini-batches with ones containing distinct training examples that produce nearly identical gradients. Data forging appears to break any potential avenues for data governance, as adversarial model owners may forge their training set from a dataset that is not compliant to one that is. Given these serious implications on data auditing and compliance, we critically analyse data forging from both a practical and theoretical point of view, finding that a key practical limitation of current attack methods makes them easily detectable by a verifier; namely that they cannot produce sufficiently identical gradients. Theoretically, we analyse the question of whether two distinct mini-batches can produce the same gradient. Generally, we find that while there may exist an infinite number of distinct mini-batches with real-valued training examples and labels that produce the same gradient, finding those that are within the allowed domain e.g. pixel values between 0-255 and one hot labels is a non trivial task. Our results call for the reevaluation of the strength of existing attacks, and for additional research into successful data forging, given the serious consequences it may have on machine learning and privacy.

摘要: 数据伪造攻击提供了反事实的证据，证明一个模型是在给定的数据集上训练的，而实际上，它是在另一个数据集上训练的。这些攻击的工作原理是用包含不同训练样本的小批次来伪造(替换)小批次，这些训练样本产生几乎相同的梯度。数据伪造似乎打破了数据治理的任何潜在途径，因为对抗性模型所有者可能会从与之不符的数据集伪造他们的训练集。鉴于这些对数据审计和合规性的严重影响，我们从实践和理论的角度对数据伪造进行了批判性分析，发现当前攻击方法的一个关键实际限制使它们很容易被验证者检测到；即它们不能产生足够相同的梯度。理论上，我们分析了两个不同的小批次是否可以产生相同的梯度的问题。通常，我们发现虽然可能存在无限数量的不同的小批次，其实值训练样本和标签产生相同的梯度，但找到那些在允许的域内的小批次，例如，像素值在0-255和一个热点标签之间是一项不平凡的任务。我们的结果要求重新评估现有攻击的强度，并考虑到它可能对机器学习和隐私造成的严重后果，对成功的数据伪造进行更多研究。



## **30. BAN: Detecting Backdoors Activated by Adversarial Neuron Noise**

BAN：检测由对抗性神经元噪音激活的后门 cs.LG

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2405.19928v2) [paper-pdf](http://arxiv.org/pdf/2405.19928v2)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Shujian Yu, Stjepan Picek

**Abstract**: Backdoor attacks on deep learning represent a recent threat that has gained significant attention in the research community. Backdoor defenses are mainly based on backdoor inversion, which has been shown to be generic, model-agnostic, and applicable to practical threat scenarios. State-of-the-art backdoor inversion recovers a mask in the feature space to locate prominent backdoor features, where benign and backdoor features can be disentangled. However, it suffers from high computational overhead, and we also find that it overly relies on prominent backdoor features that are highly distinguishable from benign features. To tackle these shortcomings, this paper improves backdoor feature inversion for backdoor detection by incorporating extra neuron activation information. In particular, we adversarially increase the loss of backdoored models with respect to weights to activate the backdoor effect, based on which we can easily differentiate backdoored and clean models. Experimental results demonstrate our defense, BAN, is 1.37$\times$ (on CIFAR-10) and 5.11$\times$ (on ImageNet200) more efficient with an average 9.99\% higher detect success rate than the state-of-the-art defense BTI-DBF. Our code and trained models are publicly available at~\url{https://github.com/xiaoyunxxy/ban}.

摘要: 对深度学习的后门攻击是最近的一种威胁，在研究界得到了极大的关注。后门防御主要基于后门倒置，这已被证明是通用的、与模型无关的，并且适用于实际的威胁场景。最先进的后门反转在特征空间中恢复掩码，以定位突出的后门特征，其中良性和后门特征可以被解开。然而，它的计算开销很高，我们还发现它过度依赖显著的后门功能，这些功能与良性功能有很大的区别。针对这些不足，本文通过引入额外的神经元激活信息，改进了后门特征倒置的后门检测方法。特别是，我们相反地增加了后门模型相对于权重的损失，以激活后门效应，基于此，我们可以很容易地区分后门模型和干净模型。实验结果表明，我们的防御算法BAN在CIFAR-10和ImageNet200上的检测效率分别为1.37和5.11，平均检测成功率比最先进的防御算法BTI-DBF高9.99倍。我们的代码和经过培训的模型可在~\url{https://github.com/xiaoyunxxy/ban}.



## **31. Post-Hoc Robustness Enhancement in Graph Neural Networks with Conditional Random Fields**

具有条件随机场的图神经网络的后组织鲁棒性增强 cs.LG

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05399v1) [paper-pdf](http://arxiv.org/pdf/2411.05399v1)

**Authors**: Yassine Abbahaddou, Sofiane Ennadir, Johannes F. Lutzeyer, Fragkiskos D. Malliaros, Michalis Vazirgiannis

**Abstract**: Graph Neural Networks (GNNs), which are nowadays the benchmark approach in graph representation learning, have been shown to be vulnerable to adversarial attacks, raising concerns about their real-world applicability. While existing defense techniques primarily concentrate on the training phase of GNNs, involving adjustments to message passing architectures or pre-processing methods, there is a noticeable gap in methods focusing on increasing robustness during inference. In this context, this study introduces RobustCRF, a post-hoc approach aiming to enhance the robustness of GNNs at the inference stage. Our proposed method, founded on statistical relational learning using a Conditional Random Field, is model-agnostic and does not require prior knowledge about the underlying model architecture. We validate the efficacy of this approach across various models, leveraging benchmark node classification datasets.

摘要: 图神经网络（GNN）是当今图表示学习的基准方法，已被证明容易受到对抗攻击，这引发了对其现实世界适用性的担忧。虽然现有的防御技术主要集中在GNN的训练阶段，涉及对消息传递架构或预处理方法的调整，但专注于提高推理期间稳健性的方法存在明显差距。在此背景下，本研究引入了RobustCF，这是一种事后方法，旨在增强GNN在推理阶段的稳健性。我们提出的方法基于使用条件随机场的统计关系学习，是模型不可知的，并且不需要有关底层模型架构的先验知识。我们利用基准节点分类数据集在各种模型中验证了这种方法的有效性。



## **32. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

保护语言模型免受越狱攻击的鲁棒即时优化 cs.LG

NeurIPS 2024 Spotlight; code available at  https://github.com/lapisrocks/rpo

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2401.17263v5) [paper-pdf](http://arxiv.org/pdf/2401.17263v5)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, large language models (LLMs) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries can modify prompts to induce unwanted behavior. While some defenses have been proposed, they have not been adapted to newly proposed attacks and more challenging threat models. To address this, we propose an optimization-based objective for defending LLMs against jailbreaking attacks and an algorithm, Robust Prompt Optimization (RPO) to create robust system-level defenses. Our approach directly incorporates the adversary into the defensive objective and optimizes a lightweight and transferable suffix, enabling RPO to adapt to worst-case adaptive attacks. Our theoretical and experimental results show improved robustness to both jailbreaks seen during optimization and unknown jailbreaks, reducing the attack success rate (ASR) on GPT-4 to 6% and Llama-2 to 0% on JailbreakBench, setting the state-of-the-art. Code can be found at https://github.com/lapisrocks/rpo

摘要: 尽管在人工智能对齐方面取得了进展，但大型语言模型(LLM)仍然容易受到对手攻击或越狱的攻击，在这些攻击或越狱中，对手可以修改提示以诱导不想要的行为。虽然已经提出了一些防御措施，但它们还没有适应新提出的攻击和更具挑战性的威胁模型。为了解决这个问题，我们提出了一个基于优化的目标来保护LLMS免受越狱攻击，并提出了一个算法--稳健提示优化(RPO)来创建强大的系统级防御。我们的方法直接将对手合并到防御目标中，并优化了一个轻量级和可转移的后缀，使RPO能够适应最坏情况的自适应攻击。我们的理论和实验结果表明，对于优化期间看到的越狱和未知越狱，我们都提高了健壮性，将GPT-4上的攻击成功率(ASR)降低到6%，将Llama-2上的攻击成功率降低到0%，从而达到了最先进的水平。代码可在https://github.com/lapisrocks/rpo上找到



## **33. Region-Guided Attack on the Segment Anything Model (SAM)**

对分段任意模型（Sam）的区域引导攻击 cs.CV

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.02974v2) [paper-pdf](http://arxiv.org/pdf/2411.02974v2)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.

摘要: Segment Anything Model(SAM)是图像分割的基石，在各种应用中表现出卓越的性能，特别是在自动驾驶和医学成像中，准确的分割至关重要。然而，SAM很容易受到对抗性攻击，这些攻击可能会通过微小的输入扰动显著损害其功能。传统的分割技术，如FGSM和PGD，在分割任务中往往是无效的，因为它们依赖于忽略空间细微差别的全局扰动。最近的方法，如攻击-SAM-K和UAD已经开始解决这些挑战，但它们经常依赖外部线索，并且没有充分利用分割过程中的结构相互依赖。这一限制强调了需要一种利用分段任务的独特特征的新的对抗性策略。作为回应，我们引入了专门为SAM设计的区域制导攻击(RGA)。RGA利用区域引导地图(RGM)来处理分割的区域，从而实现了将大片段分割并扩展小片段的有针对性的扰动，从而导致SAM的错误输出。我们的实验表明，RGA在白盒和黑盒场景中都取得了很高的成功率，强调了对这种复杂攻击的稳健防御的必要性。RGA不仅揭示了SAM的漏洞，而且为开发更具弹性的防御图像分割中的对抗性威胁奠定了基础。



## **34. Accelerating Greedy Coordinate Gradient and General Prompt Optimization via Probe Sampling**

通过探针采样加速贪婪坐标梯度和一般提示优化 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2403.01251v3) [paper-pdf](http://arxiv.org/pdf/2403.01251v3)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.

摘要: 随着大型语言模型的快速发展，其安全性已成为一个关键问题。贪婪坐标梯度(GCG)在构造敌意提示以打破排列的LLM方面是有效的，但GCG的优化是耗时的。为了减少GCG的时间开销，更全面地研究LLM的安全性，本文研究了一种新的算法--$\exttt{Probe Samples}$。该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者。使用Llama2-7b-Chat，探测采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。此外，探针采样还能够加速其他即时优化技术和对抗方法，导致AutoPrompt、APE和AutoDAN的加速分别为1.8倍$、2.4倍$和2.4倍$。



## **35. Reasoning Robustness of LLMs to Adversarial Typographical Errors**

LLM对对抗性印刷错误的推理鲁棒性 cs.CL

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.05345v1) [paper-pdf](http://arxiv.org/pdf/2411.05345v1)

**Authors**: Esther Gan, Yiran Zhao, Liying Cheng, Yancan Mao, Anirudh Goyal, Kenji Kawaguchi, Min-Yen Kan, Michael Shieh

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning using Chain-of-Thought (CoT) prompting. However, CoT can be biased by users' instruction. In this work, we study the reasoning robustness of LLMs to typographical errors, which can naturally occur in users' queries. We design an Adversarial Typo Attack ($\texttt{ATA}$) algorithm that iteratively samples typos for words that are important to the query and selects the edit that is most likely to succeed in attacking. It shows that LLMs are sensitive to minimal adversarial typographical changes. Notably, with 1 character edit, Mistral-7B-Instruct's accuracy drops from 43.7% to 38.6% on GSM8K, while with 8 character edits the performance further drops to 19.2%. To extend our evaluation to larger and closed-source LLMs, we develop the $\texttt{R$^2$ATA}$ benchmark, which assesses models' $\underline{R}$easoning $\underline{R}$obustness to $\underline{\texttt{ATA}}$. It includes adversarial typographical questions derived from three widely used reasoning datasets-GSM8K, BBH, and MMLU-by applying $\texttt{ATA}$ to open-source LLMs. $\texttt{R$^2$ATA}$ demonstrates remarkable transferability and causes notable performance drops across multiple super large and closed-source LLMs.

摘要: 大型语言模型(LLM)在使用思维链(CoT)提示进行推理方面表现出了令人印象深刻的能力。然而，COT可能会因用户的指示而产生偏差。在这项工作中，我们研究了LLMS对用户查询中自然发生的打字错误的推理健壮性。我们设计了一个对抗性的Typo攻击($\exttt{ATA}$)算法，该算法迭代地采样对查询重要的单词的打字错误，并选择最有可能成功攻击的编辑。这表明LLM对最小的对抗性排版变化很敏感。值得注意的是，在GSM8K上，1个字符编辑时，米斯特拉尔-7B指令的准确率从43.7%下降到38.6%，而8个字符编辑时，性能进一步下降到19.2%。为了将我们的评估扩展到更大的封闭源代码的LLM，我们开发了$\exttt{R$^2$ATA}$基准，它评估模型的$\下划线{R}$季节$\下划线{R}$热闹到$\下划线{Texttt{ATA}}$。它包括来自三个广泛使用的推理数据集-GSM8K、BBH和MMLU-的对抗性排版问题，方法是将$\exttt{ATA}$应用于开源LLM。$\exttt{R$^2$ATA}$表现出显著的可转移性，并在多个超大型和闭源LLM上导致显著的性能下降。



## **36. Fight Fire with Fire: Combating Adversarial Patch Attacks using Pattern-randomized Defensive Patches**

以毒攻毒：使用模式随机防御补丁对抗对抗补丁攻击 cs.CV

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2311.06122v2) [paper-pdf](http://arxiv.org/pdf/2311.06122v2)

**Authors**: Jianan Feng, Jiachun Li, Changqing Miao, Jianjun Huang, Wei You, Wenchang Shi, Bin Liang

**Abstract**: Object detection has found extensive applications in various tasks, but it is also susceptible to adversarial patch attacks. The ideal defense should be effective, efficient, easy to deploy, and capable of withstanding adaptive attacks. In this paper, we adopt a counterattack strategy to propose a novel and general methodology for defending adversarial attacks. Two types of defensive patches, canary and woodpecker, are specially-crafted and injected into the model input to proactively probe or counteract potential adversarial patches. In this manner, adversarial patch attacks can be effectively detected by simply analyzing the model output, without the need to alter the target model. Moreover, we employ randomized canary and woodpecker injection patterns to defend against defense-aware attacks. The effectiveness and practicality of the proposed method are demonstrated through comprehensive experiments. The results illustrate that canary and woodpecker achieve high performance, even when confronted with unknown attack methods, while incurring limited time overhead. Furthermore, our method also exhibits sufficient robustness against defense-aware attacks, as evidenced by adaptive attack experiments.

摘要: 目标检测在各种任务中得到了广泛的应用，但它也容易受到对抗性补丁的攻击。理想的防御应该是有效的、高效的、易于部署的、能够抵御适应性攻击的。在本文中，我们采用一种反击策略，提出了一种新颖而通用的防御对抗性攻击的方法。两种类型的防御补丁，金丝雀和啄木鸟，是专门制作的，并注入到模型输入中，以主动探测或对抗潜在的敌方补丁。通过这种方式，可以通过简单地分析模型输出来有效地检测对抗性补丁攻击，而不需要改变目标模型。此外，我们使用随机的金丝雀和啄木鸟注射模式来防御防御意识攻击。通过综合实验，验证了该方法的有效性和实用性。实验结果表明，金丝雀和啄木鸟在攻击方式未知的情况下也能获得较高的性能，同时具有有限的时间开销。此外，自适应攻击实验表明，该方法对防御感知攻击也表现出了足够的鲁棒性。



## **37. Towards Secured Smart Grid 2.0: Exploring Security Threats, Protection Models, and Challenges**

迈向安全智能电网2.0：探索安全威胁、保护模型和挑战 cs.NI

30 pages, 21 figures, 5 tables, accepted to appear in IEEE COMST

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2411.04365v2) [paper-pdf](http://arxiv.org/pdf/2411.04365v2)

**Authors**: Lan-Huong Nguyen, Van-Linh Nguyen, Ren-Hung Hwang, Jian-Jhih Kuo, Yu-Wen Chen, Chien-Chung Huang, Ping-I Pan

**Abstract**: Many nations are promoting the green transition in the energy sector to attain neutral carbon emissions by 2050. Smart Grid 2.0 (SG2) is expected to explore data-driven analytics and enhance communication technologies to improve the efficiency and sustainability of distributed renewable energy systems. These features are beyond smart metering and electric surplus distribution in conventional smart grids. Given the high dependence on communication networks to connect distributed microgrids in SG2, potential cascading failures of connectivity can cause disruption to data synchronization to the remote control systems. This paper reviews security threats and defense tactics for three stakeholders: power grid operators, communication network providers, and consumers. Through the survey, we found that SG2's stakeholders are particularly vulnerable to substation attacks/vandalism, malware/ransomware threats, blockchain vulnerabilities and supply chain breakdowns. Furthermore, incorporating artificial intelligence (AI) into autonomous energy management in distributed energy resources of SG2 creates new challenges. Accordingly, adversarial samples and false data injection on electricity reading and measurement sensors at power plants can fool AI-powered control functions and cause messy error-checking operations in energy storage, wrong energy estimation in electric vehicle charging, and even fraudulent transactions in peer-to-peer energy trading models. Scalable blockchain-based models, physical unclonable function, interoperable security protocols, and trustworthy AI models designed for managing distributed microgrids in SG2 are typical promising protection models for future research.

摘要: 许多国家正在推动能源领域的绿色转型，以期在2050年前实现碳排放中性。智能电网2.0(SG2)预计将探索数据驱动的分析和增强通信技术，以提高分布式可再生能源系统的效率和可持续性。这些功能超越了传统智能电网中的智能计量和剩余电量分配。考虑到在SG2中高度依赖通信网络来连接分布式微电网，潜在的级联连接故障可能会导致与远程控制系统的数据同步中断。本文回顾了电网运营商、通信网络提供商和消费者这三个利益相关者面临的安全威胁和防御策略。通过调查，我们发现SG2的S利益相关者特别容易受到变电站攻击/破坏、恶意软件/勒索软件威胁、区块链漏洞和供应链故障的影响。此外，将人工智能(AI)融入SG2分布式能源的自主能源管理中也带来了新的挑战。因此，发电厂电量读数和测量传感器上的敌意样本和虚假数据注入可能会愚弄人工智能支持的控制功能，并导致储能中混乱的错误检查操作，电动汽车充电中的错误能量估计，甚至P2P能源交易模式中的欺诈性交易。基于区块链的可扩展模型、物理不可克隆功能、可互操作的安全协议以及SG2中为管理分布式微网格而设计的可信AI模型是未来研究的典型保护模型。



## **38. Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling**

物理可实现的自然外观服装纹理通过3D建模躲避人体探测器 cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2024-11-08    [abs](http://arxiv.org/abs/2307.01778v2) [paper-pdf](http://arxiv.org/pdf/2307.01778v2)

**Authors**: Zhanhao Hu, Wenda Chu, Xiaopei Zhu, Hui Zhang, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.

摘要: 最近的研究提出了为躲避人体探测器而制作对抗服装，而这些服装要么只在有限的视角下有效，要么对人类来说非常显眼。我们的目标是基于3D建模为衣服制作对抗性纹理，这一想法已被用于制作刚性对抗性对象，如3D打印的乌龟。与刚性物体不同，人和衣服是非刚性的，这导致了物理实现的困难。为了制作出看起来自然的、能够在多个视角下躲避人体探测的对抗性服装，我们提出了一种类似于日常服装的典型纹理--伪装纹理的对抗性伪装纹理(AdvCaT)。我们利用Voronoi图和Gumbel-Softmax技巧对伪装纹理进行参数化，并通过3D建模优化参数。此外，我们还提出了一种结合拓扑似然投影(TOPO Proj)和薄板样条线(TPS)的三维网格增强流水线，以缩小数字对象和真实对象之间的差距。我们将开发的3D纹理块打印在面料上，并将它们裁剪成T恤和裤子。实验表明，这些衣服对多个探测器的攻击成功率很高。



## **39. A Barrier Certificate-based Simplex Architecture for Systems with Approximate and Hybrid Dynamics**

用于具有近似和混合动力学的系统的基于屏障证书的单纯形架构 eess.SY

This version includes the following new contributions. (1) We extend  Bb-Simplex to hybrid systems and prove the correctness of this extension. (2)  We extend Bb-Simplex to support the use of approximate dynamics. (3) We  combine these two extensions of Bb-Simplex. (4) We present new experiments  evaluating Bb-Simplex and its extensions using a complex model of a real  microgrid

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2202.09710v3) [paper-pdf](http://arxiv.org/pdf/2202.09710v3)

**Authors**: Amol Damare, Shouvik Roy, Roshan Sharma, Keith DSouza, Scott A. Smolka, Scott D. Stoller

**Abstract**: We present Barrier-based Simplex (Bb-Simplex), a new, provably correct design for runtime assurance of continuous dynamical systems. Bb-Simplex is centered around the Simplex control architecture, which consists of a high-performance advanced controller that is not guaranteed to maintain safety of the plant, a verified-safe baseline controller, and a decision module that switches control of the plant between the two controllers to ensure safety without sacrificing performance. In Bb-Simplex, Barrier certificates are used to prove that the baseline controller ensures safety. Furthermore, Bb-Simplex features a new automated method for deriving, from the barrier certificate, the conditions for switching between the controllers. Our method is based on the Taylor expansion of the barrier certificate and yields computationally inexpensive switching conditions.   We also propose extensions to Bb-Simplex to enable its use in hybrid systems, which have multiple modes each with its own dynamics, and to support its use when only approximate dynamics (not exact dynamics) are available, for both continuous-time and hybrid dynamical systems.   We consider significant applications of Bb-Simplex to microgrids featuring advanced controllers in the form of neural networks trained using reinforcement learning. These microgrids are modeled in RTDS, an industry-standard high-fidelity, real-time power systems simulator. Our results demonstrate that Bb-Simplex can automatically derive switching conditions for complex continuous-time and hybrid systems, the switching conditions are not overly conservative, and Bb-Simplex ensures safety even in the presence of adversarial attacks on the neural controller when only approximate dynamics (with an error bound) are available.

摘要: 本文提出了基于屏障的单纯形(BB-Simplex)，这是一种新的、可证明是正确的连续动态系统运行时保证设计。BB-Simplex以Simplex控制架构为中心，由不能保证维护工厂安全的高性能高级控制器、经过验证的安全基准控制器以及在两个控制器之间切换工厂控制以确保安全而不牺牲性能的决策模块组成。在BB-Simplex中，屏障证书被用来证明基线控制器确保了安全性。此外，BB-Simplex具有一种新的自动方法，用于从屏障证书推导控制器之间切换的条件。我们的方法是基于障碍证书的泰勒展开式，并且产生了计算上不昂贵的切换条件。我们还提出了对BB-单纯形的扩展，使其能够用于具有多个模式的混杂系统，每个模式都有自己的动态，并支持当连续时间和混杂动态系统只有近似动态(而不是精确动态)时使用BB-单纯形。我们考虑了BB-单纯形在微电网中的重要应用，该微网具有先进的控制器，其形式是使用强化学习训练的神经网络。这些微电网是在RTDS中建模的，RTDS是一种行业标准的高保真、实时电力系统仿真器。结果表明，BB-单纯形可以自动推导出复杂连续时间和混杂系统的切换条件，切换条件不是过于保守，并且在只有近似动力学(有误差界)的情况下，BB-单纯形即使在神经控制器受到对抗性攻击的情况下也能确保安全性。



## **40. Adversarial Robustness of In-Context Learning in Transformers for Linear Regression**

线性回归变换器中上下文学习的对抗鲁棒性 cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.05189v1) [paper-pdf](http://arxiv.org/pdf/2411.05189v1)

**Authors**: Usman Anwar, Johannes Von Oswald, Louis Kirsch, David Krueger, Spencer Frei

**Abstract**: Transformers have demonstrated remarkable in-context learning capabilities across various domains, including statistical learning tasks. While previous work has shown that transformers can implement common learning algorithms, the adversarial robustness of these learned algorithms remains unexplored. This work investigates the vulnerability of in-context learning in transformers to \textit{hijacking attacks} focusing on the setting of linear regression tasks. Hijacking attacks are prompt-manipulation attacks in which the adversary's goal is to manipulate the prompt to force the transformer to generate a specific output. We first prove that single-layer linear transformers, known to implement gradient descent in-context, are non-robust and can be manipulated to output arbitrary predictions by perturbing a single example in the in-context training set. While our experiments show these attacks succeed on linear transformers, we find they do not transfer to more complex transformers with GPT-2 architectures. Nonetheless, we show that these transformers can be hijacked using gradient-based adversarial attacks. We then demonstrate that adversarial training enhances transformers' robustness against hijacking attacks, even when just applied during finetuning. Additionally, we find that in some settings, adversarial training against a weaker attack model can lead to robustness to a stronger attack model. Lastly, we investigate the transferability of hijacking attacks across transformers of varying scales and initialization seeds, as well as between transformers and ordinary least squares (OLS). We find that while attacks transfer effectively between small-scale transformers, they show poor transferability in other scenarios (small-to-large scale, large-to-large scale, and between transformers and OLS).

摘要: 变形金刚在不同领域展示了卓越的情境学习能力，包括统计学习任务。虽然以前的工作已经表明，转换器可以实现常见的学习算法，但这些学习算法的对抗性健壮性仍未被探索。本文研究了在线性回归任务的设置下，变压器中的上下文学习对劫持攻击的脆弱性。劫持攻击是一种提示操纵攻击，对手的目标是操纵提示，迫使转换器生成特定的输出。我们首先证明了已知在上下文中实现梯度下降的单层线性变压器是非稳健的，并且可以通过扰动上下文训练集中的单个示例来操作以输出任意预测。虽然我们的实验表明这些攻击在线性变压器上成功，但我们发现它们不会转移到具有GPT-2架构的更复杂的变压器上。尽管如此，我们证明了可以使用基于梯度的对抗性攻击来劫持这些变压器。然后，我们证明对抗性训练增强了变压器对劫持攻击的稳健性，即使在微调期间才应用。此外，我们发现在某些情况下，针对较弱攻击模型的对抗性训练可以导致对较强攻击模型的鲁棒性。最后，我们研究了劫持攻击在不同规模和初始化种子的变压器之间以及在变压器和普通最小二乘之间的可转移性。我们发现，虽然攻击在小型变压器之间有效地转移，但在其他场景(从小到大、从大到大以及在变压器和OLS之间)表现出较差的可传递性。



## **41. Seeing is Deceiving: Exploitation of Visual Pathways in Multi-Modal Language Models**

看到就是欺骗：多模式语言模型中视觉路径的开发 cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.05056v1) [paper-pdf](http://arxiv.org/pdf/2411.05056v1)

**Authors**: Pete Janowczyk, Linda Laurier, Ave Giulietta, Arlo Octavia, Meade Cleti

**Abstract**: Multi-Modal Language Models (MLLMs) have transformed artificial intelligence by combining visual and text data, making applications like image captioning, visual question answering, and multi-modal content creation possible. This ability to understand and work with complex information has made MLLMs useful in areas such as healthcare, autonomous systems, and digital content. However, integrating multiple types of data also creates security risks. Attackers can manipulate either the visual or text inputs, or both, to make the model produce unintended or even harmful responses. This paper reviews how visual inputs in MLLMs can be exploited by various attack strategies. We break down these attacks into categories: simple visual tweaks and cross-modal manipulations, as well as advanced strategies like VLATTACK, HADES, and Collaborative Multimodal Adversarial Attack (Co-Attack). These attacks can mislead even the most robust models while looking nearly identical to the original visuals, making them hard to detect. We also discuss the broader security risks, including threats to privacy and safety in important applications. To counter these risks, we review current defense methods like the SmoothVLM framework, pixel-wise randomization, and MirrorCheck, looking at their strengths and limitations. We also discuss new methods to make MLLMs more secure, including adaptive defenses, better evaluation tools, and security approaches that protect both visual and text data. By bringing together recent developments and identifying key areas for improvement, this review aims to support the creation of more secure and reliable multi-modal AI systems for real-world use.

摘要: 多模式语言模型(MLLMS)将视觉和文本数据结合在一起，改变了人工智能，使图像字幕、视觉问答和多模式内容创建等应用成为可能。这种理解和处理复杂信息的能力使MLLMS在医疗保健、自主系统和数字内容等领域非常有用。然而，集成多种类型的数据也会带来安全风险。攻击者可以操纵视觉或文本输入，或者两者兼而有之，以使模型产生意想不到的甚至有害的响应。本文回顾了MLLMS中的视觉输入如何被各种攻击策略所利用。我们将这些攻击分为几类：简单的视觉调整和跨模式操作，以及VLATTACK、HADES和协作性多模式对抗攻击(Co-Attack)等高级策略。这些攻击甚至可以误导最健壮的模型，而它们看起来与原始视觉效果几乎相同，使得它们很难被检测到。我们还讨论了更广泛的安全风险，包括对重要应用程序中隐私和安全的威胁。为了应对这些风险，我们回顾了当前的防御方法，如SmoothVLM框架、像素随机化和MirrorCheck，并分析了它们的优点和局限性。我们还讨论了使MLLMS更安全的新方法，包括自适应防御、更好的评估工具以及保护视觉和文本数据的安全方法。通过汇集最近的发展并确定需要改进的关键领域，这项审查旨在支持创建更安全和可靠的多模式人工智能系统，供现实世界使用。



## **42. FRACTURED-SORRY-Bench: Framework for Revealing Attacks in Conversational Turns Undermining Refusal Efficacy and Defenses over SORRY-Bench (Automated Multi-shot Jailbreaks)**

骨折-抱歉-长凳：揭露对话回合中攻击的框架，这些攻击削弱了SORRY长凳（自动多枪越狱）的拒绝功效和防御 cs.CL

4 pages, 2 tables

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2408.16163v2) [paper-pdf](http://arxiv.org/pdf/2408.16163v2)

**Authors**: Aman Priyanshu, Supriti Vijay

**Abstract**: This paper introduces FRACTURED-SORRY-Bench, a framework for evaluating the safety of Large Language Models (LLMs) against multi-turn conversational attacks. Building upon the SORRY-Bench dataset, we propose a simple yet effective method for generating adversarial prompts by breaking down harmful queries into seemingly innocuous sub-questions. Our approach achieves a maximum increase of +46.22\% in Attack Success Rates (ASRs) across GPT-4, GPT-4o, GPT-4o-mini, and GPT-3.5-Turbo models compared to baseline methods. We demonstrate that this technique poses a challenge to current LLM safety measures and highlights the need for more robust defenses against subtle, multi-turn attacks.

摘要: 本文介绍了FRACTURED-SORRY-Bench，这是一个用于评估大型语言模型（LLM）针对多轮对话攻击的安全性的框架。基于SORRY-Bench数据集，我们提出了一种简单而有效的方法，通过将有害的查询分解为看似无害的子问题来生成对抗性提示。与基线方法相比，我们的方法在GPT-4、GPT-4 o、GPT-4 o-mini和GPT-3.5-Turbo模型中实现了+46.22%的攻击成功率（SVR）最大增加。我们证明这种技术对当前的LLM安全措施构成了挑战，并强调了对微妙的多回合攻击进行更强大的防御的必要性。



## **43. Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes**

梯度Cuff：通过探索拒绝损失景观来检测对大型语言模型的越狱攻击 cs.CR

Accepted by NeurIPS 2024. Project page:  https://huggingface.co/spaces/TrustSafeAI/GradientCuff-Jailbreak-Defense

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2403.00867v3) [paper-pdf](http://arxiv.org/pdf/2403.00867v3)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Large Language Models (LLMs) are becoming a prominent generative AI tool, where the user enters a query and the LLM generates an answer. To reduce harm and misuse, efforts have been made to align these LLMs to human values using advanced training techniques such as Reinforcement Learning from Human Feedback (RLHF). However, recent studies have highlighted the vulnerability of LLMs to adversarial jailbreak attempts aiming at subverting the embedded safety guardrails. To address this challenge, this paper defines and investigates the Refusal Loss of LLMs and then proposes a method called Gradient Cuff to detect jailbreak attempts. Gradient Cuff exploits the unique properties observed in the refusal loss landscape, including functional values and its smoothness, to design an effective two-step detection strategy. Experimental results on two aligned LLMs (LLaMA-2-7B-Chat and Vicuna-7B-V1.5) and six types of jailbreak attacks (GCG, AutoDAN, PAIR, TAP, Base64, and LRL) show that Gradient Cuff can significantly improve the LLM's rejection capability for malicious jailbreak queries, while maintaining the model's performance for benign user queries by adjusting the detection threshold.

摘要: 大型语言模型(LLM)正在成为一种重要的生成性人工智能工具，用户输入一个查询，LLM生成一个答案。为了减少危害和误用，已经做出努力，使用先进的培训技术，如从人类反馈中强化学习(RLHF)，使这些LLM与人类价值保持一致。然而，最近的研究强调了LLMS在旨在颠覆嵌入的安全护栏的对抗性越狱企图中的脆弱性。为了应对这一挑战，本文定义并研究了LLMS的拒绝丢失，提出了一种检测越狱企图的梯度袖口方法。梯度袖口利用拒绝丢失场景中观察到的独特属性，包括函数值及其光滑性，设计了一种有效的两步检测策略。在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B-V1.5)和六种越狱攻击(GCG、AutoDAN、Pair、TAP、Base64和LRL)上的实验结果表明，梯度袖口可以显著提高LLM对恶意越狱查询的拒绝能力，同时通过调整检测阈值保持该模型对良性用户查询的性能。



## **44. Attention Masks Help Adversarial Attacks to Bypass Safety Detectors**

注意力口罩帮助对抗攻击绕过安全检测器 cs.CR

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04772v1) [paper-pdf](http://arxiv.org/pdf/2411.04772v1)

**Authors**: Yunfan Shi

**Abstract**: Despite recent research advancements in adversarial attack methods, current approaches against XAI monitors are still discoverable and slower. In this paper, we present an adaptive framework for attention mask generation to enable stealthy, explainable and efficient PGD image classification adversarial attack under XAI monitors. Specifically, we utilize mutation XAI mixture and multitask self-supervised X-UNet for attention mask generation to guide PGD attack. Experiments on MNIST (MLP), CIFAR-10 (AlexNet) have shown that our system can outperform benchmark PGD, Sparsefool and SOTA SINIFGSM in balancing among stealth, efficiency and explainability which is crucial for effectively fooling SOTA defense protected classifiers.

摘要: 尽管最近研究在对抗攻击方法方面取得了进展，但当前针对XAI监视器的方法仍然是可行的，而且速度较慢。在本文中，我们提出了一个用于注意力屏蔽生成的自适应框架，以在XAI监视器下实现隐蔽、可解释和高效的PVD图像分类对抗攻击。具体来说，我们利用突变XAI混合物和多任务自我监督X-UNet来生成注意力屏蔽来引导PVD攻击。在MNIST（MLP）、CIFAR-10（AlexNet）上的实验表明，我们的系统在隐身性、效率和可解释性之间的平衡方面优于基准PVD、Sparsefool和SOTA SINIFGSM，这对于有效欺骗SOTA防御保护分类器至关重要。



## **45. MISGUIDE: Security-Aware Attack Analytics for Smart Grid Load Frequency Control**

MISGUIDE：用于智能电网负载频率控制的安全感知攻击分析 cs.CE

12 page journal

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04731v1) [paper-pdf](http://arxiv.org/pdf/2411.04731v1)

**Authors**: Nur Imtiazul Haque, Prabin Mali, Mohammad Zakaria Haider, Mohammad Ashiqur Rahman, Sumit Paudyal

**Abstract**: Incorporating advanced information and communication technologies into smart grids (SGs) offers substantial operational benefits while increasing vulnerability to cyber threats like false data injection (FDI) attacks. Current SG attack analysis tools predominantly employ formal methods or adversarial machine learning (ML) techniques with rule-based bad data detectors to analyze the attack space. However, these attack analytics either generate simplistic attack vectors detectable by the ML-based anomaly detection models (ADMs) or fail to identify critical attack vectors from complex controller dynamics in a feasible time. This paper introduces MISGUIDE, a novel defense-aware attack analytics designed to extract verifiable multi-time slot-based FDI attack vectors from complex SG load frequency control dynamics and ADMs, utilizing the Gurobi optimizer. MISGUIDE can identify optimal (maliciously triggering under/over frequency relays in minimal time) and stealthy attack vectors. Using real-world load data, we validate the MISGUIDE-identified attack vectors through real-time hardware-in-the-loop (OPALRT) simulations of the IEEE 39-bus system.

摘要: 将先进的信息和通信技术融入智能电网(SGS)可以带来巨大的运营效益，同时增加了对虚假数据注入(FDI)攻击等网络威胁的脆弱性。目前的SG攻击分析工具主要使用形式化方法或对抗性机器学习(ML)技术和基于规则的坏数据检测器来分析攻击空间。然而，这些攻击分析要么生成基于ML的异常检测模型(ADMS)可以检测到的简单攻击向量，要么无法在可行的时间内从复杂的控制器动态中识别关键攻击向量。本文介绍了一种新的防御感知攻击分析工具MisGuide，它利用Gurobi优化器从复杂的SG负载频率控制动态和ADMS中提取可验证的基于多时隙的FDI攻击向量。误导可以识别最优(在最短时间内恶意触发频率下/频率上的继电器)和隐蔽攻击载体。使用真实负荷数据，通过IEEE 39节点系统的实时硬件在环仿真(OPALRT)验证了误导识别的攻击向量。



## **46. Neural Fingerprints for Adversarial Attack Detection**

用于对抗性攻击检测的神经指纹 cs.CV

14 pages

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04533v1) [paper-pdf](http://arxiv.org/pdf/2411.04533v1)

**Authors**: Haim Fisher, Moni Shahar, Yehezkel S. Resheff

**Abstract**: Deep learning models for image classification have become standard tools in recent years. A well known vulnerability of these models is their susceptibility to adversarial examples. These are generated by slightly altering an image of a certain class in a way that is imperceptible to humans but causes the model to classify it wrongly as another class. Many algorithms have been proposed to address this problem, falling generally into one of two categories: (i) building robust classifiers (ii) directly detecting attacked images. Despite the good performance of these detectors, we argue that in a white-box setting, where the attacker knows the configuration and weights of the network and the detector, they can overcome the detector by running many examples on a local copy, and sending only those that were not detected to the actual model. This problem is common in security applications where even a very good model is not sufficient to ensure safety. In this paper we propose to overcome this inherent limitation of any static defence with randomization. To do so, one must generate a very large family of detectors with consistent performance, and select one or more of them randomly for each input. For the individual detectors, we suggest the method of neural fingerprints. In the training phase, for each class we repeatedly sample a tiny random subset of neurons from certain layers of the network, and if their average is sufficiently different between clean and attacked images of the focal class they are considered a fingerprint and added to the detector bank. During test time, we sample fingerprints from the bank associated with the label predicted by the model, and detect attacks using a likelihood ratio test. We evaluate our detectors on ImageNet with different attack methods and model architectures, and show near-perfect detection with low rates of false detection.

摘要: 近年来，用于图像分类的深度学习模型已成为标准工具。这些模型的一个众所周知的弱点是它们容易受到对抗性例子的影响。它们是通过以人类无法察觉的方式稍微改变某个类的图像来生成的，但会导致模型将其错误地归类为另一个类。已经提出了许多算法来解决这个问题，通常分为两类：(I)构建稳健的分类器(Ii)直接检测受攻击的图像。尽管这些检测器的性能很好，但我们认为，在白盒设置中，攻击者知道网络和检测器的配置和权重，他们可以通过在本地副本上运行许多示例，并仅将未检测到的示例发送到实际模型来克服检测器。这个问题在安全应用程序中很常见，即使是非常好的模型也不足以确保安全。在这篇文章中，我们建议用随机化来克服任何静态防御的固有局限性。要做到这一点，必须生成具有一致性能的非常大的检测器家族，并为每个输入随机选择一个或多个检测器。对于单个检测器，我们建议采用神经指纹的方法。在训练阶段，对于每一类，我们重复从网络的某些层对神经元的微小随机子集进行采样，如果它们的平均值在焦点类的干净图像和被攻击的图像之间有足够的差异，则它们被认为是指纹并添加到检测器库中。在测试期间，我们从与模型预测的标签相关联的银行中采样指纹，并使用似然比检验来检测攻击。我们在ImageNet上用不同的攻击方法和模型架构对我们的检测器进行了评估，结果表明我们的检测器在低误检率的情况下接近完美的检测。



## **47. Undermining Image and Text Classification Algorithms Using Adversarial Attacks**

使用对抗攻击削弱图像和文本分类算法 cs.CR

Accepted for presentation at Electronic Imaging Conference 2025

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.03348v2) [paper-pdf](http://arxiv.org/pdf/2411.03348v2)

**Authors**: Langalibalele Lunga, Suhas Sreehari

**Abstract**: Machine learning models are prone to adversarial attacks, where inputs can be manipulated in order to cause misclassifications. While previous research has focused on techniques like Generative Adversarial Networks (GANs), there's limited exploration of GANs and Synthetic Minority Oversampling Technique (SMOTE) in text and image classification models to perform adversarial attacks. Our study addresses this gap by training various machine learning models and using GANs and SMOTE to generate additional data points aimed at attacking text classification models. Furthermore, we extend our investigation to face recognition models, training a Convolutional Neural Network(CNN) and subjecting it to adversarial attacks with fast gradient sign perturbations on key features identified by GradCAM, a technique used to highlight key image characteristics CNNs use in classification. Our experiments reveal a significant vulnerability in classification models. Specifically, we observe a 20 % decrease in accuracy for the top-performing text classification models post-attack, along with a 30 % decrease in facial recognition accuracy. This highlights the susceptibility of these models to manipulation of input data. Adversarial attacks not only compromise the security but also undermine the reliability of machine learning systems. By showcasing the impact of adversarial attacks on both text classification and face recognition models, our study underscores the urgent need for develop robust defenses against such vulnerabilities.

摘要: 机器学习模型容易受到对抗性攻击，在这种攻击中，输入可能被操纵以导致错误分类。虽然以前的研究主要集中在生成性对抗网络(GANS)等技术上，但在文本和图像分类模型中使用生成性对抗网络(GANS)和合成少数过采样技术(SMOTE)来执行敌意攻击的探索有限。我们的研究通过训练各种机器学习模型并使用Gans和Smote生成旨在攻击文本分类模型的额外数据点来解决这一差距。此外，我们将研究扩展到人脸识别模型，训练卷积神经网络(CNN)，并对GradCAM识别的关键特征进行快速梯度符号扰动的对抗性攻击，这是一种用于突出CNN用于分类的关键图像特征的技术。我们的实验揭示了分类模型中的一个重大漏洞。具体地说，我们观察到攻击后表现最好的文本分类模型的准确率下降了20%，面部识别准确率下降了30%。这突显了这些模型对输入数据操纵的敏感性。对抗性攻击不仅破坏了机器学习系统的安全性，而且破坏了系统的可靠性。通过展示对抗性攻击对文本分类和人脸识别模型的影响，我们的研究强调了开发针对此类漏洞的强大防御的迫切需要。



## **48. Game-Theoretic Defenses for Robust Conformal Prediction Against Adversarial Attacks in Medical Imaging**

针对医学成像中对抗攻击的鲁棒共形预测的游戏理论防御 cs.LG

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.04376v1) [paper-pdf](http://arxiv.org/pdf/2411.04376v1)

**Authors**: Rui Luo, Jie Bao, Zhixin Zhou, Chuangyin Dang

**Abstract**: Adversarial attacks pose significant threats to the reliability and safety of deep learning models, especially in critical domains such as medical imaging. This paper introduces a novel framework that integrates conformal prediction with game-theoretic defensive strategies to enhance model robustness against both known and unknown adversarial perturbations. We address three primary research questions: constructing valid and efficient conformal prediction sets under known attacks (RQ1), ensuring coverage under unknown attacks through conservative thresholding (RQ2), and determining optimal defensive strategies within a zero-sum game framework (RQ3). Our methodology involves training specialized defensive models against specific attack types and employing maximum and minimum classifiers to aggregate defenses effectively. Extensive experiments conducted on the MedMNIST datasets, including PathMNIST, OrganAMNIST, and TissueMNIST, demonstrate that our approach maintains high coverage guarantees while minimizing prediction set sizes. The game-theoretic analysis reveals that the optimal defensive strategy often converges to a singular robust model, outperforming uniform and simple strategies across all evaluated datasets. This work advances the state-of-the-art in uncertainty quantification and adversarial robustness, providing a reliable mechanism for deploying deep learning models in adversarial environments.

摘要: 对抗性攻击对深度学习模型的可靠性和安全性构成了严重威胁，特别是在医学成像等关键领域。本文介绍了一种新的框架，它将保形预测与博弈论防御策略相结合，以增强模型对已知和未知对手扰动的稳健性。我们主要研究了三个问题：在已知攻击(RQ1)下构造有效且高效的共形预测集，通过保守阈值(RQ2)确保未知攻击下的覆盖，以及在零和博弈框架(RQ3)下确定最优防御策略。我们的方法包括针对特定的攻击类型训练专门的防御模型，并使用最大和最小分类器来有效地聚合防御。在包括PathMNIST、OrganAMNIST和TIseMNIST在内的MedMNIST数据集上进行的大量实验表明，我们的方法在保持高覆盖率的同时最小化了预测集的大小。博弈论分析表明，最优防御策略往往收敛到一个奇异的稳健模型，在所有评估的数据集上表现优于统一和简单的策略。这项工作推进了不确定性量化和对抗稳健性方面的最新进展，为在对抗环境中部署深度学习模型提供了可靠的机制。



## **49. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

$B ' 4 $：对LLM水印的黑匣子清除攻击 cs.CL

**SubmitDate**: 2024-11-07    [abs](http://arxiv.org/abs/2411.01222v3) [paper-pdf](http://arxiv.org/pdf/2411.01222v3)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $B^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $B^4$ compared with other baselines.

摘要: 通过嵌入不可察觉的模式，水印已经成为LLM生成的内容检测的一种重要技术。尽管具有最高的性能，但它对对手攻击的健壮性仍未得到充分开发。以前的工作通常考虑灰盒攻击设置，其中特定类型的水印已知。有些甚至需要关于水印方法的超参数的知识。这样的先决条件在现实世界的场景中是无法实现的。针对一种假设更少、更逼真的黑盒威胁模型，本文提出了一种针对水印的黑盒擦除攻击--$B^4$。具体地说，我们将水印洗涤攻击描述为一个约束优化问题，通过两个分布来捕获其目标，即水印分布和保真度分布。这个优化问题可以使用两个代理分布近似地解决。在12个不同设置下的实验结果表明，与其他基线相比，$B^4$具有更好的性能。



## **50. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

可转移习得图像抗压缩对抗扰动 cs.CV

Accepted by BMVC 2024

**SubmitDate**: 2024-11-06    [abs](http://arxiv.org/abs/2401.03115v2) [paper-pdf](http://arxiv.org/pdf/2401.03115v2)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.

摘要: 敌意攻击很容易破坏图像分类系统，暴露了基于DNN的识别任务的脆弱性。虽然现有的对抗性扰动主要应用于未压缩图像或使用传统图像压缩方法(即JPEG)压缩的图像，但在基于DNN的图像压缩环境下，已有有限的研究调查了图像分类模型的稳健性。随着先进图像压缩技术的迅速发展，基于DNN的学习图像压缩技术以其优于传统压缩的性能，在基于云的人脸识别和自动驾驶等安全关键应用中成为一种很有前途的图像传输方法。因此，迫切需要充分研究学习图像压缩后处理的分类系统的稳健性。为了弥补这一研究空白，我们探索了一种新的管道上的敌意攻击，该管道的目标是使用学习的图像压缩器作为预处理模块的图像分类模型。此外，为了增强扰动在学习图像压缩模型的不同质量水平和体系结构上的可转移性，我们引入了基于显著分数的采样方法来快速生成可转移的扰动。对常用攻击方法的大量实验表明，当攻击经过不同学习图像压缩模型后处理的图像时，所提出的方法具有更强的可转移性。



