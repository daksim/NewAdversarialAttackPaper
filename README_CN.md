# Latest Adversarial Attack Papers
**update at 2024-05-15 11:02:33**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. S3C2 Summit 2024-03: Industry Secure Supply Chain Summit**

S3 C2峰会2024-03：行业安全供应链峰会 cs.CR

This is our WIP paper on the Summit. More versions will be released  soon

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08762v1) [paper-pdf](http://arxiv.org/pdf/2405.08762v1)

**Authors**: Greg Tystahl, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: Supply chain security has become a very important vector to consider when defending against adversary attacks. Due to this, more and more developers are keen on improving their supply chains to make them more robust against future threats. On March 7th, 2024 researchers from the Secure Software Supply Chain Center (S3C2) gathered 14 industry leaders, developers and consumers of the open source ecosystem to discuss the state of supply chain security. The goal of the summit is to share insights between companies and developers alike to foster new collaborations and ideas moving forward. Through this meeting, participants were questions on best practices and thoughts how to improve things for the future. In this paper we summarize the responses and discussions of the summit. The panel questions can be found in the appendix.

摘要: 供应链安全已成为防御对手攻击时需要考虑的一个非常重要的载体。因此，越来越多的开发商热衷于改善其供应链，使其更强大地应对未来的威胁。2024年3月7日，安全软件供应链中心（S3 C2）的研究人员聚集了开源生态系统的14位行业领导者、开发人员和消费者，讨论供应链安全状况。峰会的目标是在公司和开发人员之间分享见解，以促进新的合作和前进的想法。通过这次会议，与会者就最佳实践和如何为未来改进的想法提出了问题。本文总结了峰会的回应和讨论。小组问题可在附录中找到。



## **2. Design and Analysis of Resilient Vehicular Platoon Systems over Wireless Networks**

无线网络上弹性车辆排系统的设计与分析 eess.SY

6 pages, 4 figures, in submission of Globecom 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08706v1) [paper-pdf](http://arxiv.org/pdf/2405.08706v1)

**Authors**: Tingyu Shui, Walid Saad

**Abstract**: Connected vehicular platoons provide a promising solution to improve traffic efficiency and ensure road safety. Vehicles in a platoon utilize on-board sensors and wireless vehicle-to-vehicle (V2V) links to share traffic information for cooperative adaptive cruise control. To process real-time control and alert information, there is a need to ensure clock synchronization among the platoon's vehicles. However, adversaries can jeopardize the operation of the platoon by attacking the local clocks of vehicles, leading to clock offsets with the platoon's reference clock. In this paper, a novel framework is proposed for analyzing the resilience of vehicular platoons that are connected using V2V links. In particular, a resilient design based on a diffusion protocol is proposed to re-synchronize the attacked vehicle through wireless V2V links thereby mitigating the impact of variance of the transmission delay during recovery. Then, a novel metric named temporal conditional mean exceedance is defined and analyzed in order to characterize the resilience of the platoon. Subsequently, the conditions pertaining to the V2V links and recovery time needed for a resilient design are derived. Numerical results show that the proposed resilient design is feasible in face of a nine-fold increase in the variance of transmission delay compared to a baseline designed for reliability. Moreover, the proposed approach improves the reliability, defined as the probability of meeting a desired clock offset error requirement, by 45% compared to the baseline.

摘要: 联网的车辆排为提高交通效率和确保道路安全提供了一个很有前途的解决方案。排中的车辆利用车载传感器和无线车载(V2V)链路共享交通信息，以实现协作自适应巡航控制。为了处理实时的控制和警报信息，需要确保排内车辆之间的时钟同步。然而，敌人可以通过攻击车辆的本地时钟来危及排的运行，导致时钟与排的参考时钟发生偏差。本文提出了一种新的分析车辆排弹性的框架，这些排通过V2V链路连接。特别是，提出了一种基于扩散协议的弹性设计，通过无线V2V链路重新同步受攻击的车辆，从而减轻恢复过程中传输延迟变化的影响。然后，定义并分析了一种新的度量--时间条件平均超越度，以刻画排的抗弹能力。随后，推导出了弹性设计所需的V2V链路和恢复时间的条件。数值结果表明，当传输时延的方差比可靠性设计的基线增加9倍时，所提出的弹性设计是可行的。此外，与基线相比，所提出的方法将可靠性(定义为满足期望的时钟偏移误差要求的概率)提高了45%。



## **3. Quantum Oblivious LWE Sampling and Insecurity of Standard Model Lattice-Based SNARKs**

量子不经意LWE采样和标准模型基于网格的SNARK的不安全性 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2401.03807v2) [paper-pdf](http://arxiv.org/pdf/2401.03807v2)

**Authors**: Thomas Debris-Alazard, Pouria Fallahpour, Damien Stehlé

**Abstract**: The Learning With Errors ($\mathsf{LWE}$) problem asks to find $\mathbf{s}$ from an input of the form $(\mathbf{A}, \mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}) \in (\mathbb{Z}/q\mathbb{Z})^{m \times n} \times (\mathbb{Z}/q\mathbb{Z})^{m}$, for a vector $\mathbf{e}$ that has small-magnitude entries. In this work, we do not focus on solving $\mathsf{LWE}$ but on the task of sampling instances. As these are extremely sparse in their range, it may seem plausible that the only way to proceed is to first create $\mathbf{s}$ and $\mathbf{e}$ and then set $\mathbf{b} = \mathbf{A}\mathbf{s}+\mathbf{e}$. In particular, such an instance sampler knows the solution. This raises the question whether it is possible to obliviously sample $(\mathbf{A}, \mathbf{A}\mathbf{s}+\mathbf{e})$, namely, without knowing the underlying $\mathbf{s}$. A variant of the assumption that oblivious $\mathsf{LWE}$ sampling is hard has been used in a series of works to analyze the security of candidate constructions of Succinct Non interactive Arguments of Knowledge (SNARKs). As the assumption is related to $\mathsf{LWE}$, these SNARKs have been conjectured to be secure in the presence of quantum adversaries.   Our main result is a quantum polynomial-time algorithm that samples well-distributed $\mathsf{LWE}$ instances while provably not knowing the solution, under the assumption that $\mathsf{LWE}$ is hard. Moreover, the approach works for a vast range of $\mathsf{LWE}$ parametrizations, including those used in the above-mentioned SNARKs. This invalidates the assumptions used in their security analyses, although it does not yield attacks against the constructions themselves.

摘要: 错误学习($\mathbf{lwe}$)问题要求从以下形式的输入中查找$\mathbf{S}$，对于具有小震级条目的向量$\mathbf{e}$\mathbf{e}$/mathbf{e}$\mathbf{e}$\mathbf{e}$(mathbf{Z}/q\mathbb{Z})^{m}$。在这项工作中，我们关注的不是$\mathsf{LWE}$，而是采样实例的任务。因为它们在它们的范围内非常稀疏，所以似乎唯一的继续的方法是首先创建$\mathbf{S}$和$\mathbf{e}$，然后设置$\mathbf{b}=\mathbf{A}\mathbf{S}+\mathbf{e}$。特别是，这样的实例采样器知道解决方案。这就提出了一个问题：是否有可能在不知道潜在的$\mathbf{S}$的情况下，对$(\mathbf{A}，\mathbf{A}\mathbf{S}+\mathbf{e})$进行不经意的采样。在一系列工作中，忽略的抽样是困难的这一假设的变体被用于分析简明的非交互知识论元(SNARK)候选构造的安全性。由于该假设与$\mathsf{lwe}$有关，因此人们猜测这些snarks在量子对手的存在下是安全的。我们的主要结果是一个量子多项式时间算法，它在假设$\mathsf{LWE}$是困难的假设下，对均匀分布的$\mathsf{LWE}$实例进行采样，但可证明不知道解。此外，该方法适用于大量的$\mathsf{lwe}$参数化，包括在上述snarks中使用的那些。这使他们在安全分析中使用的假设无效，尽管它不会产生针对建筑本身的攻击。



## **4. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.06823v2) [paper-pdf](http://arxiv.org/pdf/2405.06823v2)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型(LLM)支持一个新的生态系统，该生态系统具有许多下游应用程序，称为LLM应用程序，具有不同的自然语言处理任务。LLM应用程序的功能和性能高度依赖于其系统提示符，系统提示符指示后端LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统提示保密，以保护其知识产权。因此，一种称为提示泄漏的自然攻击是从LLM应用程序中窃取系统提示，这会损害开发人员的知识产权。现有的即时泄漏攻击主要依赖于手动创建的查询，因此效果有限。在本文中，我们设计了一个新颖的封闭盒提示泄漏攻击框架PLeak，用于优化敌意查询，使其在攻击者将其发送到目标LLM应用程序时，其响应显示其自己的系统提示。我们将寻找这样一个敌意查询描述为一个优化问题，并用基于梯度的方法近似求解。我们的核心思想是通过对系统提示的敌意查询进行增量优化来打破优化目标，即从每个系统提示的前几个令牌开始逐步优化，直到系统提示的整个长度。我们在离线设置和现实世界的LLM应用程序(例如，托管此类应用程序的流行平台PoE上的应用程序)中对PLeak进行评估。我们的结果表明，PLeak能够有效地泄露系统提示，不仅显著优于手动管理查询的基线，而且显著优于从现有越狱攻击中修改和调整的优化查询的基线。我们负责任地向PoE报告了这些问题，并仍在等待他们的回应。我们的实现可从以下存储库获得：https://github.com/BHui97/PLeak.



## **5. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

用多边形抽象解释证明图卷积网络对节点扰动的鲁棒性 cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08645v1) [paper-pdf](http://arxiv.org/pdf/2405.08645v1)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.

摘要: 图卷积神经网络(GCNS)是从训练数据中学习基于图的知识表示的有力工具。然而，它们容易受到输入图中的小扰动的影响，这使得它们容易受到输入错误或对手攻击的影响。这对希望用于关键应用的GCNS提出了一个重大问题，即使在存在对抗性扰动的情况下，GCNS也需要提供可证明的健壮性服务。针对存在节点特征扰动的节点分类问题，提出了一种改进的GCN健壮性认证技术。我们引入了一种新的基于多面体的抽象解释方法来解决图形数据的特定挑战，并为GCN的健壮性提供了严格的上下界。实验表明，我们的方法同时提高了健壮界的紧密性和认证的运行时性能。此外，我们的方法还可以在训练过程中使用，以进一步提高GCNS的鲁棒性。



## **6. HookChain: A new perspective for Bypassing EDR Solutions**

HookChain：询问EDR解决方案的新视角 cs.CR

46 pages, 22 figures, HookChain, Bypass EDR, Evading EDR, IAT Hook,  Halo's Gate

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2404.16856v2) [paper-pdf](http://arxiv.org/pdf/2404.16856v2)

**Authors**: Helvio Carvalho Junior

**Abstract**: In the current digital security ecosystem, where threats evolve rapidly and with complexity, companies developing Endpoint Detection and Response (EDR) solutions are in constant search for innovations that not only keep up but also anticipate emerging attack vectors. In this context, this article introduces the HookChain, a look from another perspective at widely known techniques, which when combined, provide an additional layer of sophisticated evasion against traditional EDR systems. Through a precise combination of IAT Hooking techniques, dynamic SSN resolution, and indirect system calls, HookChain redirects the execution flow of Windows subsystems in a way that remains invisible to the vigilant eyes of EDRs that only act on Ntdll.dll, without requiring changes to the source code of the applications and malwares involved. This work not only challenges current conventions in cybersecurity but also sheds light on a promising path for future protection strategies, leveraging the understanding that continuous evolution is key to the effectiveness of digital security. By developing and exploring the HookChain technique, this study significantly contributes to the body of knowledge in endpoint security, stimulating the development of more robust and adaptive solutions that can effectively address the ever-changing dynamics of digital threats. This work aspires to inspire deep reflection and advancement in the research and development of security technologies that are always several steps ahead of adversaries.   UNDER CONSTRUCTION RESEARCH: This paper is not the final version, as it is currently undergoing final tests against several EDRs. We expect to release the final version by August 2024.

摘要: 在当前的数字安全生态系统中，威胁发展迅速且复杂，开发终端检测和响应(EDR)解决方案的公司正在不断寻找创新，不仅要跟上形势，还要预测新出现的攻击媒介。在此背景下，本文介绍了HookChain，从另一个角度介绍了广为人知的技术，这些技术结合在一起时，提供了针对传统EDR系统的另一层复杂规避。通过IAT挂钩技术、动态SSN解析和间接系统调用的精确组合，HookChain以一种仅作用于Ntdll.dll的EDR保持警惕的眼睛看不到的方式重定向Windows子系统的执行流，而不需要更改所涉及的应用程序和恶意软件的源代码。这项工作不仅挑战了目前的网络安全惯例，而且还揭示了未来保护战略的一条有希望的道路，充分利用了对持续演变是数字安全有效性的关键的理解。通过开发和探索HookChain技术，这项研究对终端安全方面的知识体系做出了重大贡献，刺激了能够有效应对不断变化的数字威胁动态的更健壮和适应性更强的解决方案的开发。这项工作旨在激发人们对安全技术研究和开发的深刻反思和进步，这些技术总是领先于对手几步。正在进行的研究：这篇论文不是最终版本，因为它目前正在接受针对几个EDR的最终测试。我们预计在2024年8月之前发布最终版本。



## **7. Secure Aggregation Meets Sparsification in Decentralized Learning**

安全聚合遇到去中心化学习中的稀疏化 cs.LG

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.07708v2) [paper-pdf](http://arxiv.org/pdf/2405.07708v2)

**Authors**: Sayan Biswas, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma, Milos Vujasinovic

**Abstract**: Decentralized learning (DL) faces increased vulnerability to privacy breaches due to sophisticated attacks on machine learning (ML) models. Secure aggregation is a computationally efficient cryptographic technique that enables multiple parties to compute an aggregate of their private data while keeping their individual inputs concealed from each other and from any central aggregator. To enhance communication efficiency in DL, sparsification techniques are used, selectively sharing only the most crucial parameters or gradients in a model, thereby maintaining efficiency without notably compromising accuracy. However, applying secure aggregation to sparsified models in DL is challenging due to the transmission of disjoint parameter sets by distinct nodes, which can prevent masks from canceling out effectively. This paper introduces CESAR, a novel secure aggregation protocol for DL designed to be compatible with existing sparsification mechanisms. CESAR provably defends against honest-but-curious adversaries and can be formally adapted to counteract collusion between them. We provide a foundational understanding of the interaction between the sparsification carried out by the nodes and the proportion of the parameters shared under CESAR in both colluding and non-colluding environments, offering analytical insight into the working and applicability of the protocol. Experiments on a network with 48 nodes in a 3-regular topology show that with random subsampling, CESAR is always within 0.5% accuracy of decentralized parallel stochastic gradient descent (D-PSGD), while adding only 11% of data overhead. Moreover, it surpasses the accuracy on TopK by up to 0.3% on independent and identically distributed (IID) data.

摘要: 由于对机器学习(ML)模型的复杂攻击，分散学习(DL)面临着更多的隐私泄露漏洞。安全聚合是一种计算高效的加密技术，它使多方能够计算他们的私有数据的聚合，同时保持他们的个人输入对彼此和任何中央聚集器隐藏。为了提高DL中的通信效率，使用了稀疏化技术，选择性地仅共享模型中最关键的参数或梯度，从而在不显著影响精度的情况下保持效率。然而，将安全聚合应用于DL中的稀疏模型是具有挑战性的，这是因为不同的节点传输不相交的参数集，这会阻止掩码有效地抵消。本文介绍了一种新的面向下行链路的安全聚集协议--CESAR，该协议与现有的稀疏机制兼容。塞萨尔被证明可以防御诚实但好奇的对手，并可以正式修改以抵消他们之间的勾结。我们提供了对节点执行的稀疏化和在CESAR下在共谋和非共谋环境下共享的参数比例之间的交互的基础性理解，为协议的工作和适用性提供了分析洞察力。在一个具有48个节点的3正则拓扑网络上的实验表明，在随机子采样的情况下，CESAR算法的精度始终在分散并行随机梯度下降算法(D-PSGD)的0.5%以内，而增加的数据开销仅为11%。此外，在独立同分布(IID)数据上，它比TOPK的准确率高出0.3%。



## **8. UnMarker: A Universal Attack on Defensive Watermarking**

UnMarker：对防御性水印的普遍攻击 cs.CR

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08363v1) [paper-pdf](http://arxiv.org/pdf/2405.08363v1)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of $\textit{Generative AI}$ ($\textit{GenAI}$) to create harmful deepfakes are emerging daily. Recently, defensive watermarking, which enables $\textit{GenAI}$ providers to hide fingerprints in their images to later use for deepfake detection, has been on the rise. Yet, its potential has not been fully explored. We present $\textit{UnMarker}$ -- the first practical $\textit{universal}$ attack on defensive watermarking. Unlike existing attacks, $\textit{UnMarker}$ requires no detector feedback, no unrealistic knowledge of the scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, $\textit{UnMarker}$ employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against the $\textit{SOTA}$ prove its effectiveness, not only defeating traditional schemes while retaining superior quality compared to existing attacks but also breaking $\textit{semantic}$ watermarks that alter the image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, $\textit{UnMarker}$ is the first practical attack on $\textit{semantic}$ watermarks, which have been deemed the future of robust watermarking. $\textit{UnMarker}$ casts doubts on the very penitential of this countermeasure and exposes its paradoxical nature as designing schemes for robustness inevitably compromises other robustness aspects.

摘要: 关于滥用$\textit{生成性人工智能}$($\textit{GenAI}$)来创建有害的深度假冒的报告每天都在出现。最近，防御性水印正在兴起，它使$\textit{GenAI}$提供商能够隐藏他们图像中的指纹，以便以后用于深度假冒检测。然而，它的潜力还没有得到充分的开发。我们提出了第一个实用的针对防御性水印的$\textit{UnMarker}$攻击。与现有的攻击不同，$\textit{UnMarker}$不需要检测器反馈，不需要不切实际的方案或类似模型的知识，也不需要可能无法获得的高级去噪管道。相反，作为对水印范例的深入分析的产物，稳健方案必须在频谱幅度上构造水印，它采用了两种新的对抗性优化来扰乱水印图像的频谱，从而消除水印。对该算法的评估证明了该算法的有效性，该算法不仅能在保持现有攻击质量的同时击败传统方案，还能破解改变图像结构的水印，使最佳检测率降至43美元，并使其毫无用处。据我们所知，$\textit{UnMarker}$是针对被认为是未来稳健水印的$\textit{语义}$水印的第一个实用攻击。由于健壮性设计方案不可避免地会折衷于其他健壮性方面，因此对这一对策的悔过性提出了质疑，并暴露了它的悖论性质。



## **9. SpeechGuard: Exploring the Adversarial Robustness of Multimodal Large Language Models**

SpeechGuard：探索多模式大型语言模型的对抗鲁棒性 cs.CL

9+6 pages, Submitted to ACL 2024

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2405.08317v1) [paper-pdf](http://arxiv.org/pdf/2405.08317v1)

**Authors**: Raghuveer Peri, Sai Muralidhar Jayanthi, Srikanth Ronanki, Anshu Bhatia, Karel Mundnich, Saket Dingliwal, Nilaksh Das, Zejiang Hou, Goeric Huybrechts, Srikanth Vishnubhotla, Daniel Garcia-Romero, Sundararajan Srinivasan, Kyu J Han, Katrin Kirchhoff

**Abstract**: Integrated Speech and Large Language Models (SLMs) that can follow speech instructions and generate relevant text responses have gained popularity lately. However, the safety and robustness of these models remains largely unclear. In this work, we investigate the potential vulnerabilities of such instruction-following speech-language models to adversarial attacks and jailbreaking. Specifically, we design algorithms that can generate adversarial examples to jailbreak SLMs in both white-box and black-box attack settings without human involvement. Additionally, we propose countermeasures to thwart such jailbreaking attacks. Our models, trained on dialog data with speech instructions, achieve state-of-the-art performance on spoken question-answering task, scoring over 80% on both safety and helpfulness metrics. Despite safety guardrails, experiments on jailbreaking demonstrate the vulnerability of SLMs to adversarial perturbations and transfer attacks, with average attack success rates of 90% and 10% respectively when evaluated on a dataset of carefully designed harmful questions spanning 12 different toxic categories. However, we demonstrate that our proposed countermeasures reduce the attack success significantly.

摘要: 集成的语音和大型语言模型(SLM)可以遵循语音指令并生成相关的文本响应，最近得到了广泛的应用。然而，这些模型的安全性和稳健性在很大程度上仍不清楚。在这项工作中，我们调查了这种遵循指令的语音语言模型在对抗攻击和越狱时的潜在脆弱性。具体地说，我们设计的算法可以生成白盒和黑盒攻击环境下的越狱SLM的对抗性示例，而不需要人工参与。此外，我们还提出了挫败此类越狱攻击的对策。我们的模型在对话数据和语音指令上进行了训练，在口语问答任务中实现了最先进的性能，在安全性和有助性指标上都获得了80%以上的分数。尽管有安全护栏，但越狱实验证明了SLM在对抗性扰动和转移攻击中的脆弱性，当对12个不同有毒类别的精心设计的有害问题集进行评估时，平均攻击成功率分别为90%和10%。然而，我们证明我们提出的对策显著降低了攻击的成功率。



## **10. Adversarial Nibbler: An Open Red-Teaming Method for Identifying Diverse Harms in Text-to-Image Generation**

对抗性Nibbler：一种用于识别文本到图像生成中各种伤害的开放式红团队方法 cs.CY

10 pages, 6 figures

**SubmitDate**: 2024-05-14    [abs](http://arxiv.org/abs/2403.12075v3) [paper-pdf](http://arxiv.org/pdf/2403.12075v3)

**Authors**: Jessica Quaye, Alicia Parrish, Oana Inel, Charvi Rastogi, Hannah Rose Kirk, Minsuk Kahng, Erin van Liemt, Max Bartolo, Jess Tsang, Justin White, Nathan Clement, Rafael Mosquera, Juan Ciro, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: With the rise of text-to-image (T2I) generative AI models reaching wide audiences, it is critical to evaluate model robustness against non-obvious attacks to mitigate the generation of offensive images. By focusing on ``implicitly adversarial'' prompts (those that trigger T2I models to generate unsafe images for non-obvious reasons), we isolate a set of difficult safety issues that human creativity is well-suited to uncover. To this end, we built the Adversarial Nibbler Challenge, a red-teaming methodology for crowdsourcing a diverse set of implicitly adversarial prompts. We have assembled a suite of state-of-the-art T2I models, employed a simple user interface to identify and annotate harms, and engaged diverse populations to capture long-tail safety issues that may be overlooked in standard testing. The challenge is run in consecutive rounds to enable a sustained discovery and analysis of safety pitfalls in T2I models.   In this paper, we present an in-depth account of our methodology, a systematic study of novel attack strategies and discussion of safety failures revealed by challenge participants. We also release a companion visualization tool for easy exploration and derivation of insights from the dataset. The first challenge round resulted in over 10k prompt-image pairs with machine annotations for safety. A subset of 1.5k samples contains rich human annotations of harm types and attack styles. We find that 14% of images that humans consider harmful are mislabeled as ``safe'' by machines. We have identified new attack strategies that highlight the complexity of ensuring T2I model robustness. Our findings emphasize the necessity of continual auditing and adaptation as new vulnerabilities emerge. We are confident that this work will enable proactive, iterative safety assessments and promote responsible development of T2I models.

摘要: 随着文本到图像(T2I)生成式人工智能模型的兴起，评估模型对非明显攻击的稳健性以减少攻击性图像的生成至关重要。通过关注“隐含的对抗性”提示(那些由于不明显的原因触发T2I模型生成不安全图像的提示)，我们隔离了一组人类创造力非常适合揭示的困难安全问题。为此，我们建立了对抗性Nibbler挑战赛，这是一种用于众包各种隐含对抗性提示的红团队方法论。我们组装了一套最先进的T2I模型，使用简单的用户界面来识别和注释危害，并让不同的人群参与捕获标准测试中可能被忽视的长尾安全问题。该挑战赛分连续几轮进行，以持续发现和分析T2I型号的安全隐患。在这篇文章中，我们介绍了我们的方法，对新的攻击策略进行了系统的研究，并讨论了挑战参与者揭示的安全故障。我们还发布了一个配套的可视化工具，用于轻松探索和从数据集获得洞察力。第一轮挑战赛产生了10000多个带有机器注释的提示图像对，以确保安全。1.5K样本的子集包含丰富的危害类型和攻击风格的人类注释。我们发现，在人类认为有害的图像中，14%被机器错误地贴上了“安全”的标签。我们已经确定了新的攻击策略，这些策略突出了确保T2I模型健壮性的复杂性。我们的发现强调了随着新漏洞的出现而持续审计和适应的必要性。我们相信，这项工作将使主动、迭代的安全评估成为可能，并促进负责任的T2I模型的开发。



## **11. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

RAGE：机器生成文本检测器稳健评估的共享基准 cs.CL

To appear at ACL 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07940v1) [paper-pdf](http://arxiv.org/pdf/2405.07940v1)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with very high accuracy (99\% or higher). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging -- lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our dataset and tools to encourage further exploration into detector robustness.

摘要: 许多商业和开源模型声称可以以非常高的准确性（99%或更高）检测机器生成的文本。然而，这些检测器中很少有在共享基准数据集上进行评估，即使如此，用于评估的数据集也不够具有挑战性--缺乏采样策略、对抗性攻击和开源生成模型的变化。在这项工作中，我们介绍了RAIDA：用于机器生成文本检测的最大、最具挑战性的基准数据集。磁盘阵列包含超过600万代，涵盖11个模型、8个域、11种对抗性攻击和4种解码策略。使用RAIDGE，我们评估了8个开源检测器和4个开源检测器的域外和对抗稳健性，发现当前的检测器很容易被对抗攻击、采样策略的变化、重复惩罚和看不见的生成模型所愚弄。我们发布了我们的数据集和工具，以鼓励进一步探索检测器的稳健性。



## **12. On the Adversarial Robustness of Learning-based Image Compression Against Rate-Distortion Attacks**

基于学习的图像压缩对率失真攻击的对抗鲁棒性 eess.IV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07717v1) [paper-pdf](http://arxiv.org/pdf/2405.07717v1)

**Authors**: Chenhao Wu, Qingbo Wu, Haoran Wei, Shuai Chen, Lei Wang, King Ngi Ngan, Fanman Meng, Hongliang Li

**Abstract**: Despite demonstrating superior rate-distortion (RD) performance, learning-based image compression (LIC) algorithms have been found to be vulnerable to malicious perturbations in recent studies. Adversarial samples in these studies are designed to attack only one dimension of either bitrate or distortion, targeting a submodel with a specific compression ratio. However, adversaries in real-world scenarios are neither confined to singular dimensional attacks nor always have control over compression ratios. This variability highlights the inadequacy of existing research in comprehensively assessing the adversarial robustness of LIC algorithms in practical applications. To tackle this issue, this paper presents two joint rate-distortion attack paradigms at both submodel and algorithm levels, i.e., Specific-ratio Rate-Distortion Attack (SRDA) and Agnostic-ratio Rate-Distortion Attack (ARDA). Additionally, a suite of multi-granularity assessment tools is introduced to evaluate the attack results from various perspectives. On this basis, extensive experiments on eight prominent LIC algorithms are conducted to offer a thorough analysis of their inherent vulnerabilities. Furthermore, we explore the efficacy of two defense techniques in improving the performance under joint rate-distortion attacks. The findings from these experiments can provide a valuable reference for the development of compression algorithms with enhanced adversarial robustness.

摘要: 尽管基于学习的图像压缩(LIC)算法表现出优异的率失真(RD)性能，但在最近的研究中发现LIC算法容易受到恶意干扰。这些研究中的对抗性样本被设计为仅攻击比特率或失真的一个维度，以具有特定压缩比的子模型为目标。然而，现实世界场景中的对手既不限于单维攻击，也不总是控制压缩比。这种可变性突出了现有研究在全面评估LIC算法在实际应用中的对抗性稳健性方面的不足。针对这一问题，本文从子模型和算法两个层面提出了两种联合的率失真攻击范式，即特定比率率失真攻击(SRDA)和不可知率失真攻击(ARDA)。此外，还引入了一套多粒度评估工具，从不同角度对攻击结果进行评估。在此基础上，对8种重要的LIC算法进行了广泛的实验，深入分析了它们的固有漏洞。此外，我们还探讨了两种防御技术在提高联合码率失真攻击下性能的有效性。这些实验结果可以为开发具有增强对抗性的压缩算法提供有价值的参考。



## **13. DP-DCAN: Differentially Private Deep Contrastive Autoencoder Network for Single-cell Clustering**

DP-DCAN：用于单细胞集群的差异私有深度对比自动编码器网络 cs.LG

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2311.03410v2) [paper-pdf](http://arxiv.org/pdf/2311.03410v2)

**Authors**: Huifa Li, Jie Fu, Zhili Chen, Xiaomin Yang, Haitao Liu, Xinpeng Ling

**Abstract**: Single-cell RNA sequencing (scRNA-seq) is important to transcriptomic analysis of gene expression. Recently, deep learning has facilitated the analysis of high-dimensional single-cell data. Unfortunately, deep learning models may leak sensitive information about users. As a result, Differential Privacy (DP) is increasingly used to protect privacy. However, existing DP methods usually perturb whole neural networks to achieve differential privacy, and hence result in great performance overheads. To address this challenge, in this paper, we take advantage of the uniqueness of the autoencoder that it outputs only the dimension-reduced vector in the middle of the network, and design a Differentially Private Deep Contrastive Autoencoder Network (DP-DCAN) by partial network perturbation for single-cell clustering. Since only partial network is added with noise, the performance improvement is obvious and twofold: one part of network is trained with less noise due to a bigger privacy budget, and the other part is trained without any noise. Experimental results of six datasets have verified that DP-DCAN is superior to the traditional DP scheme with whole network perturbation. Moreover, DP-DCAN demonstrates strong robustness to adversarial attacks.

摘要: 单细胞RNA测序(scRNA-seq)对于基因表达的转录分析具有重要意义。最近，深度学习为高维单细胞数据的分析提供了便利。不幸的是，深度学习模型可能会泄露用户的敏感信息。因此，差异隐私(DP)越来越多地被用来保护隐私。然而，现有的DP方法通常会对整个神经网络进行扰动以实现差分隐私，从而导致很大的性能开销。为了应对这一挑战，本文利用自动编码器只输出网络中间降维向量的独特性，设计了一种基于部分网络扰动的差分私有深度对比自动编码器网络(DP-DCAN)，用于单小区聚类。由于只有部分网络添加了噪声，因此性能改善是明显的，而且是双重的：一部分网络的训练由于较大的隐私预算而噪声较小，而另一部分网络训练时没有任何噪声。在6个数据集上的实验结果表明，DP-DCAN算法优于传统的全网扰动下的DP算法。此外，DP-DCAN对敌方攻击表现出很强的鲁棒性。



## **14. CrossCert: A Cross-Checking Detection Approach to Patch Robustness Certification for Deep Learning Models**

CrossCert：一种交叉检查检测方法，为深度学习模型修补鲁棒性认证 cs.SE

23 pages, 2 figures, accepted by FSE 2024 (The ACM International  Conference on the Foundations of Software Engineering)

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07668v1) [paper-pdf](http://arxiv.org/pdf/2405.07668v1)

**Authors**: Qilin Zhou, Zhengyuan Wei, Haipeng Wang, Bo Jiang, W. K. Chan

**Abstract**: Patch robustness certification is an emerging kind of defense technique against adversarial patch attacks with provable guarantees. There are two research lines: certified recovery and certified detection. They aim to label malicious samples with provable guarantees correctly and issue warnings for malicious samples predicted to non-benign labels with provable guarantees, respectively. However, existing certified detection defenders suffer from protecting labels subject to manipulation, and existing certified recovery defenders cannot systematically warn samples about their labels. A certified defense that simultaneously offers robust labels and systematic warning protection against patch attacks is desirable. This paper proposes a novel certified defense technique called CrossCert. CrossCert formulates a novel approach by cross-checking two certified recovery defenders to provide unwavering certification and detection certification. Unwavering certification ensures that a certified sample, when subjected to a patched perturbation, will always be returned with a benign label without triggering any warnings with a provable guarantee. To our knowledge, CrossCert is the first certified detection technique to offer this guarantee. Our experiments show that, with a slightly lower performance than ViP and comparable performance with PatchCensor in terms of detection certification, CrossCert certifies a significant proportion of samples with the guarantee of unwavering certification.

摘要: 补丁健壮性认证是一种新兴的防御恶意补丁攻击的技术，具有可证明的保证。有两条研究路线：认证的回收和认证的检测。它们的目标是正确地标记具有可证明保证的恶意样本，并分别对预测为具有可证明保证的非良性标签的恶意样本发出警告。然而，现有的认证检测防御者遭受着保护受操纵的标签的困扰，并且现有的认证恢复防御者不能系统地警告样本有关其标签的信息。同时提供坚固标签和针对补丁攻击的系统警告保护的认证防御是可取的。提出了一种新的认证防御技术CrossCert。CrossCert通过交叉检查两个经过认证的恢复防御者来制定一种新的方法，以提供坚定不移的认证和检测认证。坚定不移的认证确保了经过认证的样品在受到修补扰动时，始终会被退回带有良性标签的产品，而不会触发任何带有可证明保证的警告。据我们所知，CrossCert是第一个提供这一保证的认证检测技术。我们的实验表明，在检测认证方面，CrossCert的性能略低于VIP，但在检测认证方面与补丁检查器相当，可以在保证认证坚定不移的情况下认证相当比例的样本。



## **15. Backdoor Removal for Generative Large Language Models**

生成性大型语言模型的后门删除 cs.CR

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07667v1) [paper-pdf](http://arxiv.org/pdf/2405.07667v1)

**Authors**: Haoran Li, Yulin Chen, Zihao Zheng, Qi Hu, Chunkit Chan, Heshan Liu, Yangqiu Song

**Abstract**: With rapid advances, generative large language models (LLMs) dominate various Natural Language Processing (NLP) tasks from understanding to reasoning. Yet, language models' inherent vulnerabilities may be exacerbated due to increased accessibility and unrestricted model training on massive textual data from the Internet. A malicious adversary may publish poisoned data online and conduct backdoor attacks on the victim LLMs pre-trained on the poisoned data. Backdoored LLMs behave innocuously for normal queries and generate harmful responses when the backdoor trigger is activated. Despite significant efforts paid to LLMs' safety issues, LLMs are still struggling against backdoor attacks. As Anthropic recently revealed, existing safety training strategies, including supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF), fail to revoke the backdoors once the LLM is backdoored during the pre-training stage. In this paper, we present Simulate and Eliminate (SANDE) to erase the undesired backdoored mappings for generative LLMs. We initially propose Overwrite Supervised Fine-tuning (OSFT) for effective backdoor removal when the trigger is known. Then, to handle the scenarios where the trigger patterns are unknown, we integrate OSFT into our two-stage framework, SANDE. Unlike previous works that center on the identification of backdoors, our safety-enhanced LLMs are able to behave normally even when the exact triggers are activated. We conduct comprehensive experiments to show that our proposed SANDE is effective against backdoor attacks while bringing minimal harm to LLMs' powerful capability without any additional access to unbackdoored clean models. We will release the reproducible code.

摘要: 随着研究的深入，从理解到推理的各种自然语言处理任务都被生成性大语言模型(LLMS)所支配。然而，语言模型固有的脆弱性可能会因为可访问性的提高和对来自互联网的海量文本数据的不受限制的模型训练而加剧。恶意对手可能会在网上发布有毒数据，并对受害者LLM进行后门攻击，这些LLM预先训练了有毒数据。后门LLM在正常查询中的行为是无害的，并在激活后门触发器时生成有害的响应。尽管在LLMS的安全问题上付出了巨大的努力，但LLMS仍在努力应对后门攻击。正如人类最近揭示的那样，现有的安全培训策略，包括监督微调(SFT)和从人类反馈的强化学习(RLHF)，一旦LLM在培训前阶段后退，就无法取消后门。在这篇文章中，我们提出了模拟和消除(SANDE)来消除生成式LLMS中不需要的回溯映射。我们最初提出了覆盖监督精调(OSFT)，用于在已知触发器的情况下有效地删除后门。然后，为了处理触发模式未知的场景，我们将OSFT集成到我们的两阶段框架SANDE中。与以前以识别后门为中心的工作不同，我们的安全增强型LLM即使在准确的触发器被激活时也能够正常运行。我们进行了全面的实验，以表明我们提出的SANDE可以有效地抵御后门攻击，同时对LLMS的强大功能造成的损害最小，而不需要额外访问未后门的干净模型。我们将发布可重现的代码。



## **16. Environmental Matching Attack Against Unmanned Aerial Vehicles Object Detection**

针对无人机目标检测的环境匹配攻击 cs.CV

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07595v1) [paper-pdf](http://arxiv.org/pdf/2405.07595v1)

**Authors**: Dehong Kong, Siyuan Liang, Wenqi Ren

**Abstract**: Object detection techniques for Unmanned Aerial Vehicles (UAVs) rely on Deep Neural Networks (DNNs), which are vulnerable to adversarial attacks. Nonetheless, adversarial patches generated by existing algorithms in the UAV domain pay very little attention to the naturalness of adversarial patches. Moreover, imposing constraints directly on adversarial patches makes it difficult to generate patches that appear natural to the human eye while ensuring a high attack success rate. We notice that patches are natural looking when their overall color is consistent with the environment. Therefore, we propose a new method named Environmental Matching Attack(EMA) to address the issue of optimizing the adversarial patch under the constraints of color. To the best of our knowledge, this paper is the first to consider natural patches in the domain of UAVs. The EMA method exploits strong prior knowledge of a pretrained stable diffusion to guide the optimization direction of the adversarial patch, where the text guidance can restrict the color of the patch. To better match the environment, the contrast and brightness of the patch are appropriately adjusted. Instead of optimizing the adversarial patch itself, we optimize an adversarial perturbation patch which initializes to zero so that the model can better trade off attacking performance and naturalness. Experiments conducted on the DroneVehicle and Carpk datasets have shown that our work can reach nearly the same attack performance in the digital attack(no greater than 2 in mAP$\%$), surpass the baseline method in the physical specific scenarios, and exhibit a significant advantage in terms of naturalness in visualization and color difference with the environment.

摘要: 无人机(UAV)的目标检测技术依赖于深度神经网络(DNN)，而DNN容易受到对手的攻击。尽管如此，无人机领域的现有算法生成的对抗性补丁很少关注对抗性补丁的自然性。此外，直接对敌方补丁施加限制，使得在确保高攻击成功率的同时，很难生成人眼看起来很自然的补丁。我们注意到，当补丁的整体颜色与环境一致时，它们看起来很自然。因此，我们提出了一种新的方法--环境匹配攻击(EMA)来解决颜色约束下敌方补丁的优化问题。据我们所知，本文是第一次考虑无人机领域中的自然斑块。EMA方法利用预先训练的稳定扩散的强先验知识来指导对抗性补丁的优化方向，其中文本指导可以限制补丁的颜色。为了更好地匹配环境，贴片的对比度和亮度进行了适当的调整。我们没有优化对抗性补丁本身，而是优化了一个初始化为零的对抗性扰动补丁，使模型能够更好地在攻击性能和自然性之间进行权衡。在DroneVehicle和Carpk数据集上进行的实验表明，我们的工作在数字攻击中可以达到几乎相同的攻击性能(MAP$不大于2)，在物理特定场景中超过基线方法，在可视化和与环境的色差方面显示出显著的优势。



## **17. Towards Rational Consensus in Honest Majority**

在诚实的多数中走向理性共识 cs.GT

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07557v1) [paper-pdf](http://arxiv.org/pdf/2405.07557v1)

**Authors**: Varul Srivastava, Sujit Gujar

**Abstract**: Distributed consensus protocols reach agreement among $n$ players in the presence of $f$ adversaries; different protocols support different values of $f$. Existing works study this problem for different adversary types (captured by threat models). There are three primary threat models: (i) Crash fault tolerance (CFT), (ii) Byzantine fault tolerance (BFT), and (iii) Rational fault tolerance (RFT), each more general than the previous. Agreement in repeated rounds on both (1) the proposed value in each round and (2) the ordering among agreed-upon values across multiple rounds is called Atomic BroadCast (ABC). ABC is more generalized than consensus and is employed in blockchains.   This work studies ABC under the RFT threat model. We consider $t$ byzantine and $k$ rational adversaries among $n$ players. We also study different types of rational players based on their utility towards (1) liveness attack, (2) censorship or (3) disagreement (forking attack). We study the problem of ABC under this general threat model in partially-synchronous networks. We show (1) ABC is impossible for $n/3< (t+k) <n/2$ if rational players prefer liveness or censorship attacks and (2) the consensus protocol proposed by Ranchal-Pedrosa and Gramoli cannot be generalized to solve ABC due to insecure Nash equilibrium (resulting in disagreement). For ABC in partially synchronous network settings, we propose a novel protocol \textsf{pRFT}(practical Rational Fault Tolerance). We show \textsf{pRFT} achieves ABC if (a) rational players prefer only disagreement attacks and (b) $t < \frac{n}{4}$ and $(t + k) < \frac{n}{2}$. In \textsf{pRFT}, we incorporate accountability (capturing deviating players) within the protocol by leveraging honest players. We also show that the message complexity of \textsf{pRFT} is at par with the best consensus protocols that guarantee accountability.

摘要: 分布式共识协议在存在$f$对手的情况下，在$n$参与者之间达成协议；不同的协议支持不同的$f$值。现有的研究工作针对不同的对手类型(通过威胁模型捕获)来研究这一问题。有三种主要的威胁模型：(I)崩溃容错(CFT)，(Ii)拜占庭容错(BFT)和(Iii)理性容错(RFT)，每一种模型都比以前的模型更具一般性。在重复回合中就(1)每轮中的建议值和(2)多轮中商定的值之间的排序这两个方面达成一致称为原子广播(ABC)。ABC比共识更一般化，并被应用于区块链。本文研究了RFT威胁模型下的ABC。我们在$n$玩家中考虑$t$拜占庭和$k$理性对手。我们还研究了不同类型的理性玩家，根据他们对(1)活性攻击，(2)审查或(3)分歧(分叉攻击)的效用。我们研究了部分同步网络中这种一般威胁模型下的ABC问题。我们证明了：(1)如果理性参与者喜欢活跃度或审查攻击，则$n/3<(t+k)<n/2$时ABC是不可能的；(2)由于不安全的纳什均衡(导致不一致)，Ranchal-Pedrosa和Gramoli提出的共识协议不能推广到求解ABC。对于部分同步网络环境下的ABC，我们提出了一种新的协议-.我们证明了，如果(A)理性参与者只喜欢不一致攻击，并且(B)$t<\frac{n}{4}$和$(t+k)<\frac{n}{2}$，则文本sf{pRFT}达到ABC。在\extsf{pRFT}中，我们通过利用诚实的参与者将责任(捕获偏离规则的参与者)纳入到协议中。我们还证明了Textsf{pRFT}的消息复杂性与保证可问责性的最佳共识协议相当。



## **18. On Securing Analog Lagrange Coded Computing from Colluding Adversaries**

保护模拟拉格朗日编码计算免受共谋对手的侵害 cs.IT

To appear in the proceedings of IEEE ISIT 2024

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2405.07454v1) [paper-pdf](http://arxiv.org/pdf/2405.07454v1)

**Authors**: Rimpi Borah, J. Harshan

**Abstract**: Analog Lagrange Coded Computing (ALCC) is a recently proposed coded computing paradigm wherein certain computations over analog datasets can be efficiently performed using distributed worker nodes through floating point implementation. While ALCC is known to preserve privacy of data from the workers, it is not resilient to adversarial workers that return erroneous computation results. Pointing at this security vulnerability, we focus on securing ALCC from a wide range of non-colluding and colluding adversarial workers. As a foundational step, we make use of error-correction algorithms for Discrete Fourier Transform (DFT) codes to build novel algorithms to nullify the erroneous computations returned from the adversaries. Furthermore, when such a robust ALCC is implemented in practical settings, we show that the presence of precision errors in the system can be exploited by the adversaries to propose novel colluding attacks to degrade the computation accuracy. As the main takeaway, we prove a counter-intuitive result that not all the adversaries should inject noise in their computations in order to optimally degrade the accuracy of the ALCC framework. This is the first work of its kind to address the vulnerability of ALCC against colluding adversaries.

摘要: 模拟拉格朗日编码计算(ALCC)是最近提出的一种编码计算范例，其中模拟数据集上的某些计算可以通过浮点实现使用分布式工作节点来高效地执行。虽然众所周知，ALCC可以保护工作人员的数据隐私，但它对返回错误计算结果的敌意工作人员没有弹性。针对这一安全漏洞，我们专注于保护ALCC免受广泛的非串通和串通敌方工作人员的攻击。作为基础步骤，我们利用离散傅里叶变换(DFT)码的纠错算法来构建新的算法来抵消从对手返回的错误计算。此外，当这种健壮的ALCC在实际环境中实现时，我们证明了攻击者可以利用系统中存在的精度误差来提出新的合谋攻击来降低计算精度。作为主要的结论，我们证明了一个与直觉相反的结果，即并不是所有的对手都应该在他们的计算中注入噪声，以便最佳地降低ALCC框架的准确性。这是解决ALCC针对串通对手的脆弱性的第一项同类工作。



## **19. Universal Coding for Shannon Ciphers under Side-Channel Attacks**

侧通道攻击下香农密码的通用编码 cs.IT

6 pages, 3 figures. arXiv admin note: substantial text overlap with  arXiv:1801.02563, arXiv:2201.11670, arXiv:1901.05940

**SubmitDate**: 2024-05-13    [abs](http://arxiv.org/abs/2302.01314v3) [paper-pdf](http://arxiv.org/pdf/2302.01314v3)

**Authors**: Yasutada Oohama, Bagus Santoso

**Abstract**: We study the universal coding under side-channel attacks posed and investigated by Oohama and Santoso (2022). They proposed a theoretical security model for Shannon cipher system under side-channel attacks, where the adversary is not only allowed to collect ciphertexts by eavesdropping the public communication channel, but is also allowed to collect the physical information leaked by the devices where the cipher system is implemented on such as running time, power consumption, electromagnetic radiation, etc. For any distributions of the plain text, any noisy channels through which the adversary observe the corrupted version of the key, and any measurement device used for collecting the physical information, we can derive an achievable rate region for reliability and security such that if we compress the ciphertext with rate within the achievable rate region, then: (1) anyone with secret key will be able to decrypt and decode the ciphertext correctly, but (2) any adversary who obtains the ciphertext and also the side physical information will not be able to obtain any information about the hidden source as long as the leaked physical information is encoded with a rate within the rate region.

摘要: 我们研究了Oohama和Santoso(2022)提出和研究的边信道攻击下的通用编码。他们提出了一种侧信道攻击下Shannon密码系统的理论安全模型，其中不仅允许攻击者通过窃听公共通信信道来收集密文，还允许攻击者收集实现密码系统的设备泄露的物理信息，如运行时间、功耗、电磁辐射等。对于明文的任何分布、攻击者观察密钥被破坏的任何噪声信道以及用于收集物理信息的任何测量设备，我们可以推导出可靠性和安全性的可达速率区域，使得如果将密文的速率压缩在可达速率区域内，那么：(1)任何拥有秘密密钥的人都将能够正确地解密和解码密文，但是(2)只要以速率区域内的速率对泄漏的物理信息进行编码，任何获得密文以及侧物理信息的对手都将不能获得关于隐藏源的任何信息。



## **20. The Janus Interface: How Fine-Tuning in Large Language Models Amplifies the Privacy Risks**

Janus界面：大型语言模型中的微调如何放大隐私风险 cs.CR

**SubmitDate**: 2024-05-12    [abs](http://arxiv.org/abs/2310.15469v2) [paper-pdf](http://arxiv.org/pdf/2310.15469v2)

**Authors**: Xiaoyi Chen, Siyuan Tang, Rui Zhu, Shijun Yan, Lei Jin, Zihao Wang, Liya Su, Zhikun Zhang, XiaoFeng Wang, Haixu Tang

**Abstract**: The rapid advancements of large language models (LLMs) have raised public concerns about the privacy leakage of personally identifiable information (PII) within their extensive training datasets. Recent studies have demonstrated that an adversary could extract highly sensitive privacy data from the training data of LLMs with carefully designed prompts. However, these attacks suffer from the model's tendency to hallucinate and catastrophic forgetting (CF) in the pre-training stage, rendering the veracity of divulged PIIs negligible. In our research, we propose a novel attack, Janus, which exploits the fine-tuning interface to recover forgotten PIIs from the pre-training data in LLMs. We formalize the privacy leakage problem in LLMs and explain why forgotten PIIs can be recovered through empirical analysis on open-source language models. Based upon these insights, we evaluate the performance of Janus on both open-source language models and two latest LLMs, i.e., GPT-3.5-Turbo and LLaMA-2-7b. Our experiment results show that Janus amplifies the privacy risks by over 10 times in comparison with the baseline and significantly outperforms the state-of-the-art privacy extraction attacks including prefix attacks and in-context learning (ICL). Furthermore, our analysis validates that existing fine-tuning APIs provided by OpenAI and Azure AI Studio are susceptible to our Janus attack, allowing an adversary to conduct such an attack at a low cost.

摘要: 大型语言模型(LLM)的快速发展引起了公众对其广泛训练数据集中个人身份信息(PII)隐私泄露的担忧。最近的研究表明，攻击者可以通过精心设计的提示从LLMS的训练数据中提取高度敏感的隐私数据。然而，这些攻击受到模型在预训练阶段的幻觉和灾难性遗忘(CF)的倾向的影响，使得泄露的PII的真实性可以忽略不计。在我们的研究中，我们提出了一种新的攻击，Janus，它利用微调接口从LLMS的训练前数据中恢复被遗忘的PII。我们形式化地描述了LLMS中的隐私泄露问题，并通过对开源语言模型的实证分析解释了为什么被遗忘的PII可以恢复。基于这些见解，我们评估了Janus在开源语言模型和两个最新的LLMS上的性能，即GPT-3.5-Turbo和Llama-2-7b。我们的实验结果表明，Janus将隐私风险放大了10倍以上，并且显著优于目前最先进的隐私提取攻击，包括前缀攻击和上下文中学习(ICL)。此外，我们的分析验证了OpenAI和Azure AI Studio提供的现有微调API容易受到我们的Janus攻击，允许对手以低成本进行此类攻击。



## **21. Synthesizing Iris Images using Generative Adversarial Networks: Survey and Comparative Analysis**

使用生成对抗网络合成虹膜图像：调查和比较分析 cs.CV

**SubmitDate**: 2024-05-11    [abs](http://arxiv.org/abs/2404.17105v2) [paper-pdf](http://arxiv.org/pdf/2404.17105v2)

**Authors**: Shivangi Yadav, Arun Ross

**Abstract**: Biometric systems based on iris recognition are currently being used in border control applications and mobile devices. However, research in iris recognition is stymied by various factors such as limited datasets of bonafide irides and presentation attack instruments; restricted intra-class variations; and privacy concerns. Some of these issues can be mitigated by the use of synthetic iris data. In this paper, we present a comprehensive review of state-of-the-art GAN-based synthetic iris image generation techniques, evaluating their strengths and limitations in producing realistic and useful iris images that can be used for both training and testing iris recognition systems and presentation attack detectors. In this regard, we first survey the various methods that have been used for synthetic iris generation and specifically consider generators based on StyleGAN, RaSGAN, CIT-GAN, iWarpGAN, StarGAN, etc. We then analyze the images generated by these models for realism, uniqueness, and biometric utility. This comprehensive analysis highlights the pros and cons of various GANs in the context of developing robust iris matchers and presentation attack detectors.

摘要: 基于虹膜识别的生物识别系统目前正被用于边境管制应用和移动设备。然而，虹膜识别的研究受到各种因素的阻碍，例如真实虹膜和呈现攻击工具的数据集有限；类内变异有限；以及隐私问题。其中一些问题可以通过使用合成虹膜数据来缓解。本文对最新的基于GaN的合成虹膜图像生成技术进行了全面的综述，评价了它们在生成逼真和有用的虹膜图像方面的优势和局限性，这些图像可以用于虹膜识别系统和呈现攻击检测器的训练和测试。在这方面，我们首先综述了用于合成虹膜生成的各种方法，并具体考虑了基于StyleGAN、RaSGAN、CIT-GAN、iWarpGAN、StarGAN等的生成器。然后，我们分析了这些模型生成的图像的真实感、唯一性和生物特征实用价值。这一全面的分析强调了在开发健壮的虹膜匹配器和呈现攻击检测器的背景下各种GAN的优缺点。



## **22. Tree Proof-of-Position Algorithms**

树位置证明算法 cs.DS

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06761v1) [paper-pdf](http://arxiv.org/pdf/2405.06761v1)

**Authors**: Aida Manzano Kharman, Pietro Ferraro, Homayoun Hamedmoghadam, Robert Shorten

**Abstract**: We present a novel class of proof-of-position algorithms: Tree-Proof-of-Position (T-PoP). This algorithm is decentralised, collaborative and can be computed in a privacy preserving manner, such that agents do not need to reveal their position publicly. We make no assumptions of honest behaviour in the system, and consider varying ways in which agents may misbehave. Our algorithm is therefore resilient to highly adversarial scenarios. This makes it suitable for a wide class of applications, namely those in which trust in a centralised infrastructure may not be assumed, or high security risk scenarios. Our algorithm has a worst case quadratic runtime, making it suitable for hardware constrained IoT applications. We also provide a mathematical model that summarises T-PoP's performance for varying operating conditions. We then simulate T-PoP's behaviour with a large number of agent-based simulations, which are in complete agreement with our mathematical model, thus demonstrating its validity. T-PoP can achieve high levels of reliability and security by tuning its operating conditions, both in high and low density environments. Finally, we also present a mathematical model to probabilistically detect platooning attacks.

摘要: 提出了一类新的位置证明算法：树位置证明算法(T-POP)。该算法是分散的、协作的，并且可以以保护隐私的方式进行计算，因此代理不需要公开透露他们的位置。我们没有对系统中的诚实行为做出假设，并考虑了代理人可能不当行为的各种方式。因此，我们的算法对高度对抗性的场景具有弹性。这使得它适合于广泛类别的应用程序，即那些可能不信任集中式基础设施的应用程序，或高安全风险场景。该算法具有最坏情况下的二次运行时间，适用于硬件受限的物联网应用。我们还提供了一个数学模型，总结了T-POP在不同工作条件下的性能。然后，我们用大量基于代理的模拟来模拟T-POP的行为，这与我们的数学模型完全一致，从而证明了它的有效性。T-POP可以通过调整其在高密度和低密度环境中的运行条件来实现高水平的可靠性和安全性。最后，我们还给出了一个概率检测排队攻击的数学模型。



## **23. Certified $\ell_2$ Attribution Robustness via Uniformly Smoothed Attributions**

通过均匀平滑的归因认证$\ell_2$归因稳健性 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06361v1) [paper-pdf](http://arxiv.org/pdf/2405.06361v1)

**Authors**: Fan Wang, Adams Wai-Kin Kong

**Abstract**: Model attribution is a popular tool to explain the rationales behind model predictions. However, recent work suggests that the attributions are vulnerable to minute perturbations, which can be added to input samples to fool the attributions while maintaining the prediction outputs. Although empirical studies have shown positive performance via adversarial training, an effective certified defense method is eminently needed to understand the robustness of attributions. In this work, we propose to use uniform smoothing technique that augments the vanilla attributions by noises uniformly sampled from a certain space. It is proved that, for all perturbations within the attack region, the cosine similarity between uniformly smoothed attribution of perturbed sample and the unperturbed sample is guaranteed to be lower bounded. We also derive alternative formulations of the certification that is equivalent to the original one and provides the maximum size of perturbation or the minimum smoothing radius such that the attribution can not be perturbed. We evaluate the proposed method on three datasets and show that the proposed method can effectively protect the attributions from attacks, regardless of the architecture of networks, training schemes and the size of the datasets.

摘要: 模型归因是解释模型预测背后的理论基础的流行工具。然而，最近的工作表明，属性容易受到微小扰动的影响，可以将这些微小扰动添加到输入样本中，以在保持预测输出的同时愚弄属性。虽然实证研究表明，通过对抗性训练取得了积极的效果，但需要一种有效的认证防御方法来理解归因的稳健性。在这项工作中，我们提出使用均匀平滑技术，通过从特定空间均匀采样的噪声来增强香草属性。证明了对于攻击区域内的所有扰动，扰动样本的一致光滑属性与未扰动样本的余弦相似保证是下界的。我们还推导出了与原始证明等价的证明的替代公式，并且提供了最大扰动大小或最小光滑半径，使得属性不能被扰动。我们在三个数据集上对该方法进行了评估，结果表明，无论网络结构、训练方案和数据集的大小如何，该方法都能有效地保护属性免受攻击。



## **24. Evaluating Adversarial Robustness in the Spatial Frequency Domain**

空间频域中的对抗鲁棒性评估 cs.CV

14 pages

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06345v1) [paper-pdf](http://arxiv.org/pdf/2405.06345v1)

**Authors**: Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen

**Abstract**: Convolutional Neural Networks (CNNs) have dominated the majority of computer vision tasks. However, CNNs' vulnerability to adversarial attacks has raised concerns about deploying these models to safety-critical applications. In contrast, the Human Visual System (HVS), which utilizes spatial frequency channels to process visual signals, is immune to adversarial attacks. As such, this paper presents an empirical study exploring the vulnerability of CNN models in the frequency domain. Specifically, we utilize the discrete cosine transform (DCT) to construct the Spatial-Frequency (SF) layer to produce a block-wise frequency spectrum of an input image and formulate Spatial Frequency CNNs (SF-CNNs) by replacing the initial feature extraction layers of widely-used CNN backbones with the SF layer. Through extensive experiments, we observe that SF-CNN models are more robust than their CNN counterparts under both white-box and black-box attacks. To further explain the robustness of SF-CNNs, we compare the SF layer with a trainable convolutional layer with identical kernel sizes using two mixing strategies to show that the lower frequency components contribute the most to the adversarial robustness of SF-CNNs. We believe our observations can guide the future design of robust CNN models.

摘要: 卷积神经网络(CNN)已经主导了计算机视觉的大部分任务。然而，CNN对对手攻击的脆弱性已经引起了人们对将这些模型部署到安全关键应用程序的担忧。相比之下，人类视觉系统(HVS)利用空间频率通道来处理视觉信号，不受对手攻击。因此，本文提出了一项实证研究，探索CNN模型在频域中的脆弱性。具体地说，我们利用离散余弦变换(DCT)来构造空间频率(SF)层来产生输入图像的块状频谱，并通过用SF层替换广泛使用的CNN骨干的初始特征提取层来构造空间频率CNN(SF-CNN)。通过大量的实验，我们观察到SF-CNN模型在白盒和黑盒攻击下都比CNN模型更健壮。为了进一步解释SF-CNN的健壮性，我们使用两种混合策略将SF层与具有相同核大小的可训练卷积层进行了比较，结果表明低频分量对SF-CNN的对抗健壮性贡献最大。我们相信，我们的观察可以指导未来稳健的CNN模型的设计。



## **25. Improving Transferable Targeted Adversarial Attack via Normalized Logit Calibration and Truncated Feature Mixing**

通过规范化Logit校准和截断特征混合改进可转移有针对性的对抗攻击 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06340v1) [paper-pdf](http://arxiv.org/pdf/2405.06340v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: This paper aims to enhance the transferability of adversarial samples in targeted attacks, where attack success rates remain comparatively low. To achieve this objective, we propose two distinct techniques for improving the targeted transferability from the loss and feature aspects. First, in previous approaches, logit calibrations used in targeted attacks primarily focus on the logit margin between the targeted class and the untargeted classes among samples, neglecting the standard deviation of the logit. In contrast, we introduce a new normalized logit calibration method that jointly considers the logit margin and the standard deviation of logits. This approach effectively calibrates the logits, enhancing the targeted transferability. Second, previous studies have demonstrated that mixing the features of clean samples during optimization can significantly increase transferability. Building upon this, we further investigate a truncated feature mixing method to reduce the impact of the source training model, resulting in additional improvements. The truncated feature is determined by removing the Rank-1 feature associated with the largest singular value decomposed from the high-level convolutional layers of the clean sample. Extensive experiments conducted on the ImageNet-Compatible and CIFAR-10 datasets demonstrate the individual and mutual benefits of our proposed two components, which outperform the state-of-the-art methods by a large margin in black-box targeted attacks.

摘要: 本文旨在提高攻击成功率相对较低的定向攻击中对抗性样本的可转移性。为了实现这一目标，我们从损失和特征两个方面提出了两种不同的技术来提高目标可转移性。首先，在以往的方法中，用于目标攻击的Logit校准主要集中在样本中目标类和非目标类之间的Logit差值，而忽略了Logit的标准差。相反，我们引入了一种新的归一化Logit校准方法，该方法同时考虑了Logit裕度和Logit的标准差。这种方法有效地校准了LOGITS，增强了目标可转移性。其次，以往的研究表明，在优化过程中混合清洁样本的特征可以显著提高可转移性。在此基础上，我们进一步研究了一种截断特征混合方法，以减少源训练模型的影响，从而得到进一步的改进。通过去除与从清洁样本的高级卷积层分解的最大奇异值相关联的Rank-1特征来确定截断特征。在ImageNet兼容和CIFAR-10数据集上进行的广泛实验表明，我们提出的两个组件具有单独和共同的好处，在黑盒定向攻击中远远超过最先进的方法。



## **26. PUMA: margin-based data pruning**

SEARCH A：基于利润的数据修剪 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06298v1) [paper-pdf](http://arxiv.org/pdf/2405.06298v1)

**Authors**: Javier Maroto, Pascal Frossard

**Abstract**: Deep learning has been able to outperform humans in terms of classification accuracy in many tasks. However, to achieve robustness to adversarial perturbations, the best methodologies require to perform adversarial training on a much larger training set that has been typically augmented using generative models (e.g., diffusion models). Our main objective in this work, is to reduce these data requirements while achieving the same or better accuracy-robustness trade-offs. We focus on data pruning, where some training samples are removed based on the distance to the model classification boundary (i.e., margin). We find that the existing approaches that prune samples with low margin fails to increase robustness when we add a lot of synthetic data, and explain this situation with a perceptron learning task. Moreover, we find that pruning high margin samples for better accuracy increases the harmful impact of mislabeled perturbed data in adversarial training, hurting both robustness and accuracy. We thus propose PUMA, a new data pruning strategy that computes the margin using DeepFool, and prunes the training samples of highest margin without hurting performance by jointly adjusting the training attack norm on the samples of lowest margin. We show that PUMA can be used on top of the current state-of-the-art methodology in robustness, and it is able to significantly improve the model performance unlike the existing data pruning strategies. Not only PUMA achieves similar robustness with less data, but it also significantly increases the model accuracy, improving the performance trade-off.

摘要: 在许多任务中，深度学习在分类准确率方面已经能够超过人类。然而，为了实现对对抗性扰动的稳健性，最好的方法需要在通常使用生成模型(例如，扩散模型)扩充的大得多的训练集上执行对抗性训练。我们在这项工作中的主要目标是减少这些数据要求，同时实现相同或更好的精度-稳健性权衡。我们的重点是数据剪枝，即根据到模型分类边界的距离(即边界)来删除一些训练样本。我们发现，当我们添加大量的合成数据时，现有的对低边际样本进行剪枝的方法不能提高鲁棒性，并用感知器学习任务来解释这种情况。此外，我们发现，为了更好的准确性而修剪高边缘样本会增加错误标记的扰动数据在对抗性训练中的有害影响，损害稳健性和准确性。因此，我们提出了一种新的数据剪枝策略PUMA，它使用DeepFool计算差值，并在差值最小的样本上联合调整训练攻击范数，在不影响性能的情况下修剪差值最高的训练样本。我们表明，PUMA可以在当前最先进的方法的健壮性上使用，并且它能够显著提高模型的性能，而不是现有的数据剪枝策略。PUMA不仅用更少的数据实现了类似的稳健性，而且还显著提高了模型的精度，改善了性能权衡。



## **27. Exploring the Interplay of Interpretability and Robustness in Deep Neural Networks: A Saliency-guided Approach**

探索深度神经网络中可解释性和鲁棒性的相互作用：显着性引导的方法 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06278v1) [paper-pdf](http://arxiv.org/pdf/2405.06278v1)

**Authors**: Amira Guesmi, Nishant Suresh Aswani, Muhammad Shafique

**Abstract**: Adversarial attacks pose a significant challenge to deploying deep learning models in safety-critical applications. Maintaining model robustness while ensuring interpretability is vital for fostering trust and comprehension in these models. This study investigates the impact of Saliency-guided Training (SGT) on model robustness, a technique aimed at improving the clarity of saliency maps to deepen understanding of the model's decision-making process. Experiments were conducted on standard benchmark datasets using various deep learning architectures trained with and without SGT. Findings demonstrate that SGT enhances both model robustness and interpretability. Additionally, we propose a novel approach combining SGT with standard adversarial training to achieve even greater robustness while preserving saliency map quality. Our strategy is grounded in the assumption that preserving salient features crucial for correctly classifying adversarial examples enhances model robustness, while masking non-relevant features improves interpretability. Our technique yields significant gains, achieving a 35\% and 20\% improvement in robustness against PGD attack with noise magnitudes of $0.2$ and $0.02$ for the MNIST and CIFAR-10 datasets, respectively, while producing high-quality saliency maps.

摘要: 对抗性攻击对在安全关键型应用中部署深度学习模型提出了重大挑战。在确保可解释性的同时保持模型的健壮性对于培养对这些模型的信任和理解至关重要。本研究调查显著引导训练(SGT)对模型稳健性的影响，这是一种旨在提高显著图的清晰度以加深对模型决策过程的理解的技术。实验在标准基准数据集上进行，使用各种深度学习体系结构，在有和没有SGT的情况下进行训练。研究结果表明，SGT既增强了模型的稳健性，又增强了模型的可解释性。此外，我们提出了一种结合SGT和标准对抗性训练的新方法，在保持显著图质量的同时获得更好的稳健性。我们的策略基于这样的假设，即保留对于正确分类对抗性示例至关重要的显著特征可以增强模型的稳健性，而屏蔽不相关的特征可以提高可解释性。我们的技术产生了显著的收益，在MNIST和CIFAR-10数据集的噪声幅度分别为0.2美元和0.02美元的情况下，对PGD攻击的稳健性分别提高了35%和20%，同时生成了高质量的显著图。



## **28. Disttack: Graph Adversarial Attacks Toward Distributed GNN Training**

区别：针对分布式GNN培训的图形对抗攻击 cs.LG

Accepted by 30th International European Conference on Parallel and  Distributed Computing(Euro-Par 2024)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06247v1) [paper-pdf](http://arxiv.org/pdf/2405.06247v1)

**Authors**: Yuxiang Zhang, Xin Liu, Meng Wu, Wei Yan, Mingyu Yan, Xiaochun Ye, Dongrui Fan

**Abstract**: Graph Neural Networks (GNNs) have emerged as potent models for graph learning. Distributing the training process across multiple computing nodes is the most promising solution to address the challenges of ever-growing real-world graphs. However, current adversarial attack methods on GNNs neglect the characteristics and applications of the distributed scenario, leading to suboptimal performance and inefficiency in attacking distributed GNN training.   In this study, we introduce Disttack, the first framework of adversarial attacks for distributed GNN training that leverages the characteristics of frequent gradient updates in a distributed system. Specifically, Disttack corrupts distributed GNN training by injecting adversarial attacks into one single computing node. The attacked subgraphs are precisely perturbed to induce an abnormal gradient ascent in backpropagation, disrupting gradient synchronization between computing nodes and thus leading to a significant performance decline of the trained GNN. We evaluate Disttack on four large real-world graphs by attacking five widely adopted GNNs. Compared with the state-of-the-art attack method, experimental results demonstrate that Disttack amplifies the model accuracy degradation by 2.75$\times$ and achieves speedup by 17.33$\times$ on average while maintaining unnoticeability.

摘要: 图神经网络(GNN)已经成为图学习的有力模型。将训练过程分布在多个计算节点上是解决不断增长的真实世界图的挑战的最有前途的解决方案。然而，目前针对GNN的对抗性攻击方法忽略了分布式场景的特点和应用，导致在攻击分布式GNN训练时性能不佳且效率低下。在这项研究中，我们介绍了Disttack，这是第一个用于分布式GNN训练的对抗性攻击框架，它利用了分布式系统中频繁梯度更新的特点。具体地说，Disttack通过将敌意攻击注入到单个计算节点来破坏分布式GNN训练。被攻击的子图被精确地扰动，导致反向传播中的异常梯度上升，扰乱了计算节点之间的梯度同步，从而导致训练后的GNN的性能显著下降。我们通过攻击五个广泛使用的GNN来评估四个大型真实世界图上的Disttack。实验结果表明，与最新的攻击方法相比，Disttack在保持不可察觉的情况下，使模型的准确率降低了2.75倍，平均加速比提高了17.33倍。



## **29. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

通过触发优化的数据中毒隐藏联邦学习中后门模型更新 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06206v1) [paper-pdf](http://arxiv.org/pdf/2405.06206v1)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on Byzantine-robust aggregation exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.

摘要: 联合学习(FL)是一种去中心化的机器学习方法，允许参与者在不共享私人数据的情况下协作训练模型。尽管FL具有隐私和可扩展性方面的优势，但它很容易受到后门攻击，即攻击者使用后门触发器毒化部分客户端的本地训练数据，目的是在推理时输入满足相同的后门条件时，使聚合模型产生恶意结果。FL中现有的后门攻击存在共同的缺陷：固定的触发模式和依赖模型中毒的辅助。由于恶意模型更新和良性模型更新之间的显著差异，基于拜占庭稳健聚合的最新防御技术在这些攻击中表现出良好的防御性能。为了有效地隐藏良性模型更新中的恶意模型更新，我们提出了一种FL中的后门攻击策略DPOT，它通过优化后门触发器来动态构建后门目标，使后门数据对模型更新的影响最小。我们为DPOT的攻击原理提供了理论依据，并展示了实验结果表明，DPOT仅通过一次数据中毒攻击就可以有效地破坏最先进的防御措施，并在各种数据集上优于现有的后门攻击技术。



## **30. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **31. Hard Work Does Not Always Pay Off: Poisoning Attacks on Neural Architecture Search**

努力工作并不总是有回报：对神经架构搜索的毒害攻击 cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06073v1) [paper-pdf](http://arxiv.org/pdf/2405.06073v1)

**Authors**: Zachary Coalson, Huazheng Wang, Qingyun Wu, Sanghyun Hong

**Abstract**: In this paper, we study the robustness of "data-centric" approaches to finding neural network architectures (known as neural architecture search) to data distribution shifts. To audit this robustness, we present a data poisoning attack, when injected to the training data used for architecture search that can prevent the victim algorithm from finding an architecture with optimal accuracy. We first define the attack objective for crafting poisoning samples that can induce the victim to generate sub-optimal architectures. To this end, we weaponize existing search algorithms to generate adversarial architectures that serve as our objectives. We also present techniques that the attacker can use to significantly reduce the computational costs of crafting poisoning samples. In an extensive evaluation of our poisoning attack on a representative architecture search algorithm, we show its surprising robustness. Because our attack employs clean-label poisoning, we also evaluate its robustness against label noise. We find that random label-flipping is more effective in generating sub-optimal architectures than our clean-label attack. Our results suggests that care must be taken for the data this emerging approach uses, and future work is needed to develop robust algorithms.

摘要: 在本文中，我们研究了“以数据为中心”的方法寻找神经网络结构(称为神经结构搜索)对数据分布变化的稳健性。为了检验这种健壮性，我们提出了一种数据中毒攻击，当注入用于体系结构搜索的训练数据时，可以阻止受害者算法以最佳精度找到体系结构。我们首先定义了制作中毒样本的攻击目标，这些样本可以诱导受害者生成次优的体系结构。为此，我们将现有的搜索算法武器化，以生成作为我们目标的对抗性架构。我们还提供了攻击者可以用来显著降低制作中毒样本的计算成本的技术。在对我们对一个典型架构搜索算法的毒化攻击的广泛评估中，我们展示了其惊人的健壮性。因为我们的攻击使用了干净标签中毒，所以我们还评估了它对标签噪声的稳健性。我们发现随机标签翻转在生成次优体系结构方面比我们的干净标签攻击更有效。我们的结果表明，必须注意这种新兴方法使用的数据，并且需要进一步的工作来开发健壮的算法。



## **32. BB-Patch: BlackBox Adversarial Patch-Attack using Zeroth-Order Optimization**

BB-patch：使用零阶优化的黑匣子对抗补丁攻击 cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06049v1) [paper-pdf](http://arxiv.org/pdf/2405.06049v1)

**Authors**: Satyadwyoom Kumar, Saurabh Gupta, Arun Balaji Buduru

**Abstract**: Deep Learning has become popular due to its vast applications in almost all domains. However, models trained using deep learning are prone to failure for adversarial samples and carry a considerable risk in sensitive applications. Most of these adversarial attack strategies assume that the adversary has access to the training data, the model parameters, and the input during deployment, hence, focus on perturbing the pixel level information present in the input image.   Adversarial Patches were introduced to the community which helped in bringing out the vulnerability of deep learning models in a much more pragmatic manner but here the attacker has a white-box access to the model parameters. Recently, there has been an attempt to develop these adversarial attacks using black-box techniques. However, certain assumptions such as availability large training data is not valid for a real-life scenarios. In a real-life scenario, the attacker can only assume the type of model architecture used from a select list of state-of-the-art architectures while having access to only a subset of input dataset. Hence, we propose an black-box adversarial attack strategy that produces adversarial patches which can be applied anywhere in the input image to perform an adversarial attack.

摘要: 深度学习由于其在几乎所有领域的广泛应用而变得流行起来。然而，使用深度学习训练的模型对于对抗性样本容易失败，并且在敏感应用中具有相当大的风险。这些对抗性攻击策略大多假设对手在部署过程中可以访问训练数据、模型参数和输入，因此，专注于干扰输入图像中存在的像素级信息。社区中引入了对抗性补丁，这有助于以更实用的方式暴露深度学习模型的漏洞，但在这里，攻击者可以通过白盒访问模型参数。最近，有人试图使用黑盒技术来开发这些对抗性攻击。然而，某些假设，如大量训练数据的可用性，对于现实生活场景是不成立的。在现实生活场景中，攻击者只能假定从最先进的体系结构的精选列表中使用的模型体系结构的类型，同时只能访问输入数据集的子集。因此，我们提出了一种黑盒对抗性攻击策略，该策略产生对抗性补丁，可以应用于输入图像中的任何位置来执行对抗性攻击。



## **33. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

智能6G网络中值得信赖的人工智能生成内容：对抗性、隐私性和公平性 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.

摘要: 以大语言模型(LLM)为代表的AI-Generated Content(AIGC)模型给内容生成领域带来了革命性的变化。高速和广泛的6G技术是提供强大的AIGC移动业务应用的理想平台，而未来的6G移动网络也需要支持智能化和个性化的移动生成服务。然而，当前AIGC模型存在的重大伦理和安全问题，如对抗性攻击、隐私和公平性，极大地影响了6G智能网络的可信度，特别是在确保安全、私有和公平的AIGC应用方面。本文提出了一种新的6G网络可信AIGC模型TrustGAIN，以保证未来6G网络中可信赖的大规模AIGC服务。我们首先讨论了AIGC系统在6G网络中面临的敌意攻击和隐私威胁，以及相应的保护问题。随后，我们强调了在未来的智能网中确保移动生成业务的无偏性和公平性的重要性。特别是，我们进行了一个用例来证明TrustGAIN可以有效地指导对恶意或生成的虚假信息的抵抗。我们认为，TrustGAIN是智能可信6G网络支持AIGC服务的必备范式，确保AIGC网络服务的安全性、私密性和公平性。



## **34. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

针对合成数据的属性推理攻击的线性重建方法 cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2301.10053v3) [paper-pdf](http://arxiv.org/pdf/2301.10053v3)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.

摘要: 合成数据生成(SDG)的最新进展被誉为在保护隐私的同时共享敏感数据这一难题的解决方案。SDG旨在学习真实数据的统计属性，以便生成在结构和统计上与敏感数据相似的“人造”数据。然而，先前的研究表明，对合成数据的推理攻击可能会破坏隐私，但仅限于特定的离群值记录。在这项工作中，我们引入了一种新的针对合成数据的属性推理攻击。该攻击基于聚合统计的线性重建方法，其目标是数据集中的所有记录，而不仅仅是离群值。我们评估了我们对最先进的SDG算法的攻击，包括概率图形模型、生成性对手网络和最近的差异私有SDG机制。通过定义一个正式的隐私游戏，我们证明了我们的攻击即使在任意记录上也可以非常准确，并且这是个人信息泄露的结果(与总体级别的推断相反)。然后，我们系统地评估了保护隐私和保护统计效用之间的权衡。我们的发现表明，现有的SDG方法在保持合理效用的同时，不能始终如一地提供足够的隐私保护来抵御推理攻击。评估的最佳方法是一种不同的私有SDG机制，它可以提供对推理攻击的保护和合理的实用程序，但只能在非常特定的环境中提供。最后，我们表明，发布更多的合成记录可以提高实用性，但代价是使攻击更加有效。



## **35. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

通过注意力细化实现针对基于补丁的攻击的鲁棒语义分割 cs.CV

Accepted by International Journal of Computer Vision (IJCV).34 pages,  5 figures, 16 tables

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2401.01750v2) [paper-pdf](http://arxiv.org/pdf/2401.01750v2)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.

摘要: 近年来，注意机制在各种视觉任务中被证明是有效的。在语义分割任务中，注意力机制被应用于各种方法，包括卷积神经网络(CNN)和视觉转换器(VIT)作为骨干的情况。然而，我们观察到注意机制很容易受到基于补丁的对抗性攻击。通过对有效接受场的分析，我们将其归因于全球注意带来的广泛接受场可能导致对抗性斑块的传播。针对这一问题，本文提出了一种健壮的注意力机制(RAM)来提高语义分割模型的健壮性，该机制可以显著缓解语义分割模型对基于补丁攻击的脆弱性。与Vallina注意机制相比，RAM引入了两个新的模块：最大注意抑制和随机注意丢弃，这两个模块的目的都是为了细化注意矩阵，并限制单个敌意补丁对其他位置语义分割结果的影响。大量实验表明，在不同的攻击环境下，该算法能够有效地提高语义分割模型对各种基于补丁的攻击方法的稳健性。



## **36. TroLLoc: Logic Locking and Layout Hardening for IC Security Closure against Hardware Trojans**

TroLLoc：逻辑锁定和布局硬化，以防止硬件特洛伊木马的IC安全关闭 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05590v1) [paper-pdf](http://arxiv.org/pdf/2405.05590v1)

**Authors**: Fangzhou Wang, Qijing Wang, Lilas Alrahis, Bangqi Fu, Shui Jiang, Xiaopeng Zhang, Ozgur Sinanoglu, Tsung-Yi Ho, Evangeline F. Y. Young, Johann Knechtel

**Abstract**: Due to cost benefits, supply chains of integrated circuits (ICs) are largely outsourced nowadays. However, passing ICs through various third-party providers gives rise to many security threats, like piracy of IC intellectual property or insertion of hardware Trojans, i.e., malicious circuit modifications.   In this work, we proactively and systematically protect the physical layouts of ICs against post-design insertion of Trojans. Toward that end, we propose TroLLoc, a novel scheme for IC security closure that employs, for the first time, logic locking and layout hardening in unison. TroLLoc is fully integrated into a commercial-grade design flow, and TroLLoc is shown to be effective, efficient, and robust. Our work provides in-depth layout and security analysis considering the challenging benchmarks of the ISPD'22/23 contests for security closure. We show that TroLLoc successfully renders layouts resilient, with reasonable overheads, against (i) general prospects for Trojan insertion as in the ISPD'22 contest, (ii) actual Trojan insertion as in the ISPD'23 contest, and (iii) potential second-order attacks where adversaries would first (i.e., before Trojan insertion) try to bypass the locking defense, e.g., using advanced machine learning attacks. Finally, we release all our artifacts for independent verification [2].

摘要: 由于成本效益，如今集成电路(IC)的供应链大多被外包。然而，通过各种第三方提供商传递IC会带来许多安全威胁，如盗版IC知识产权或插入硬件特洛伊木马程序，即恶意电路修改。在这项工作中，我们主动和系统地保护IC的物理布局不受设计后插入特洛伊木马的影响。为此，我们提出了一种新的IC安全闭包方案--TroLLoc，它首次采用了逻辑锁定和版图加固相结合的方法。TroLLoc被完全集成到商业级设计流程中，并被证明是有效、高效和健壮的。考虑到ISPD‘22/23安全关闭竞赛的挑战性基准，我们的工作提供了深入的布局和安全分析。我们表明，TroLLoc成功地以合理的开销使布局具有弹性，以对抗(I)如在ISPD‘22比赛中那样的一般特洛伊木马插入前景，(Ii)如在ISPD’23比赛中那样的实际特洛伊木马插入，以及(Iii)潜在的二阶攻击，其中对手首先(即，在木马插入之前)试图绕过锁定防御，例如，使用高级机器学习攻击。最后，我们发布所有构件以进行独立验证[2]。



## **37. Poisoning-based Backdoor Attacks for Arbitrary Target Label with Positive Triggers**

针对具有阳性触发的任意目标标签的基于中毒的后门攻击 cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05573v1) [paper-pdf](http://arxiv.org/pdf/2405.05573v1)

**Authors**: Binxiao Huang, Jason Chun Lok, Chang Liu, Ngai Wong

**Abstract**: Poisoning-based backdoor attacks expose vulnerabilities in the data preparation stage of deep neural network (DNN) training. The DNNs trained on the poisoned dataset will be embedded with a backdoor, making them behave well on clean data while outputting malicious predictions whenever a trigger is applied. To exploit the abundant information contained in the input data to output label mapping, our scheme utilizes the network trained from the clean dataset as a trigger generator to produce poisons that significantly raise the success rate of backdoor attacks versus conventional approaches. Specifically, we provide a new categorization of triggers inspired by the adversarial technique and develop a multi-label and multi-payload Poisoning-based backdoor attack with Positive Triggers (PPT), which effectively moves the input closer to the target label on benign classifiers. After the classifier is trained on the poisoned dataset, we can generate an input-label-aware trigger to make the infected classifier predict any given input to any target label with a high possibility. Under both dirty- and clean-label settings, we show empirically that the proposed attack achieves a high attack success rate without sacrificing accuracy across various datasets, including SVHN, CIFAR10, GTSRB, and Tiny ImageNet. Furthermore, the PPT attack can elude a variety of classical backdoor defenses, proving its effectiveness.

摘要: 基于中毒的后门攻击暴露了深度神经网络(DNN)训练的数据准备阶段的漏洞。在有毒数据集上训练的DNN将嵌入一个后门，使它们在干净的数据上表现良好，同时在应用触发器时输出恶意预测。为了利用输入数据中包含的丰富信息来输出标签映射，我们的方案利用从干净数据集训练的网络作为触发生成器来产生毒药，与传统方法相比，显著提高了后门攻击的成功率。具体地说，我们在对抗性技术的启发下提出了一种新的触发器分类方法，并提出了一种基于多标签和多负载中毒的正触发器后门攻击(PPT)，有效地使输入更接近良性分类器上的目标标签。当分类器在中毒的数据集上训练后，我们可以生成一个输入标签感知触发器，使受感染的分类器预测任何给定的输入到任何目标标签的可能性很高。在脏标签和干净标签两种设置下，我们的经验表明，所提出的攻击在不牺牲包括SVHN、CIFAR10、GTSRB和Tiny ImageNet在内的各种数据集的精度的情况下获得了高的攻击成功率。此外，PPT攻击可以避开各种经典的后门防御，证明了其有效性。



## **38. Universal Adversarial Perturbations for Vision-Language Pre-trained Models**

视觉语言预训练模型的普遍对抗扰动 cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05524v1) [paper-pdf](http://arxiv.org/pdf/2405.05524v1)

**Authors**: Peng-Fei Zhang, Zi Huang, Guangdong Bai

**Abstract**: Vision-language pre-trained (VLP) models have been the foundation of numerous vision-language tasks. Given their prevalence, it becomes imperative to assess their adversarial robustness, especially when deploying them in security-crucial real-world applications. Traditionally, adversarial perturbations generated for this assessment target specific VLP models, datasets, and/or downstream tasks. This practice suffers from low transferability and additional computation costs when transitioning to new scenarios.   In this work, we thoroughly investigate whether VLP models are commonly sensitive to imperceptible perturbations of a specific pattern for the image modality. To this end, we propose a novel black-box method to generate Universal Adversarial Perturbations (UAPs), which is so called the Effective and T ransferable Universal Adversarial Attack (ETU), aiming to mislead a variety of existing VLP models in a range of downstream tasks. The ETU comprehensively takes into account the characteristics of UAPs and the intrinsic cross-modal interactions to generate effective UAPs. Under this regime, the ETU encourages both global and local utilities of UAPs. This benefits the overall utility while reducing interactions between UAP units, improving the transferability. To further enhance the effectiveness and transferability of UAPs, we also design a novel data augmentation method named ScMix. ScMix consists of self-mix and cross-mix data transformations, which can effectively increase the multi-modal data diversity while preserving the semantics of the original data. Through comprehensive experiments on various downstream tasks, VLP models, and datasets, we demonstrate that the proposed method is able to achieve effective and transferrable universal adversarial attacks.

摘要: 视觉语言预训练(VLP)模型是众多视觉语言任务的基础。鉴于它们的普遍存在，评估它们的对手健壮性变得势在必行，特别是在将它们部署在安全关键的现实世界应用程序中时。传统上，为该评估生成的对抗性扰动针对特定的VLP模型、数据集和/或下游任务。这种做法的缺点是可转移性低，在过渡到新方案时需要额外的计算成本。在这项工作中，我们彻底调查了VLP模型是否通常对图像通道的特定模式的不可察觉的扰动敏感。为此，我们提出了一种新的生成通用对抗扰动(UAP)的黑箱方法，即有效且可传递的通用对抗攻击(ETU)，其目的是在一系列下游任务中误导现有的各种VLP模型。ETU综合考虑了UAP的特点和固有的跨模式交互作用，以生成有效的UAP。在这一制度下，ETU鼓励全球和当地的UAP公用事业。这有利于整体效用，同时减少了UAP单元之间的交互，提高了可转移性。为了进一步提高UAP的有效性和可转移性，我们还设计了一种新的数据增强方法ScMix。ScMix包括自混合和交叉混合数据转换，在保持原始数据语义的同时，有效地增加了多模式数据的多样性。通过在各种下游任务、VLP模型和数据集上的综合实验，我们证明了该方法能够实现有效的、可转移的通用对抗性攻击。



## **39. Towards Accurate and Robust Architectures via Neural Architecture Search**

通过神经架构搜索实现准确和稳健的架构 cs.CV

Accepted by CVPR2024. arXiv admin note: substantial text overlap with  arXiv:2212.14049

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05502v1) [paper-pdf](http://arxiv.org/pdf/2405.05502v1)

**Authors**: Yuwei Ou, Yuqi Feng, Yanan Sun

**Abstract**: To defend deep neural networks from adversarial attacks, adversarial training has been drawing increasing attention for its effectiveness. However, the accuracy and robustness resulting from the adversarial training are limited by the architecture, because adversarial training improves accuracy and robustness by adjusting the weight connection affiliated to the architecture. In this work, we propose ARNAS to search for accurate and robust architectures for adversarial training. First we design an accurate and robust search space, in which the placement of the cells and the proportional relationship of the filter numbers are carefully determined. With the design, the architectures can obtain both accuracy and robustness by deploying accurate and robust structures to their sensitive positions, respectively. Then we propose a differentiable multi-objective search strategy, performing gradient descent towards directions that are beneficial for both natural loss and adversarial loss, thus the accuracy and robustness can be guaranteed at the same time. We conduct comprehensive experiments in terms of white-box attacks, black-box attacks, and transferability. Experimental results show that the searched architecture has the strongest robustness with the competitive accuracy, and breaks the traditional idea that NAS-based architectures cannot transfer well to complex tasks in robustness scenarios. By analyzing outstanding architectures searched, we also conclude that accurate and robust neural architectures tend to deploy different structures near the input and output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust architectures.

摘要: 为了保护深层神经网络免受对抗性攻击，对抗性训练因其有效性而受到越来越多的关注。然而，对抗训练产生的准确性和稳健性受到体系结构的限制，因为对抗训练通过调整附属于体系结构的权重连接来提高准确性和稳健性。在这项工作中，我们建议ARNAS为对抗训练寻找准确和健壮的体系结构。首先，我们设计了一个精确且稳健的搜索空间，在这个空间中，我们仔细地确定了单元的位置和过滤器数量的比例关系。通过这种设计，结构可以通过将精确和稳健的结构分别部署到其敏感位置来获得精度和稳健性。然后提出了一种可微多目标搜索策略，向有利于自然损失和对手损失的方向进行梯度下降，保证了搜索的准确性和稳健性。我们在白盒攻击、黑盒攻击和可转移性方面进行了全面的实验。实验结果表明，搜索到的体系结构具有最强的稳健性，具有与之相当的准确率，打破了基于NAS的体系结构在健壮性场景下不能很好地迁移到复杂任务的传统思想。通过对搜索到的优秀体系结构的分析，我们还得出结论：准确和健壮的神经体系结构往往在输入和输出附近部署不同的结构，这对于手工制作和自动设计准确健壮的体系结构都具有重要的现实意义。



## **40. Adversary-Guided Motion Retargeting for Skeleton Anonymization**

对抗引导的骨架模拟运动重定向 cs.CV

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05428v1) [paper-pdf](http://arxiv.org/pdf/2405.05428v1)

**Authors**: Thomas Carr, Depeng Xu, Aidong Lu

**Abstract**: Skeleton-based motion visualization is a rising field in computer vision, especially in the case of virtual reality (VR). With further advancements in human-pose estimation and skeleton extracting sensors, more and more applications that utilize skeleton data have come about. These skeletons may appear to be anonymous but they contain embedded personally identifiable information (PII). In this paper we present a new anonymization technique that is based on motion retargeting, utilizing adversary classifiers to further remove PII embedded in the skeleton. Motion retargeting is effective in anonymization as it transfers the movement of the user onto the a dummy skeleton. In doing so, any PII linked to the skeleton will be based on the dummy skeleton instead of the user we are protecting. We propose a Privacy-centric Deep Motion Retargeting model (PMR) which aims to further clear the retargeted skeleton of PII through adversarial learning. In our experiments, PMR achieves motion retargeting utility performance on par with state of the art models while also reducing the performance of privacy attacks.

摘要: 基于骨架的运动可视化是计算机视觉领域的一个新兴领域，尤其是在虚拟现实(VR)领域。随着人体姿态估计和骨骼提取传感器的进一步发展，利用骨骼数据的应用越来越多。这些骨架可能看起来是匿名的，但它们包含嵌入的个人身份信息(PII)。本文提出了一种新的匿名技术，该技术基于运动重定向，利用敌方分类器进一步去除嵌入在骨架中的PII。运动重定目标在匿名化中是有效的，因为它将用户的运动转移到虚拟骨骼上。这样，链接到骨架的任何PII都将基于虚拟骨架，而不是我们正在保护的用户。我们提出了一种以隐私为中心的深度运动重定向模型(PMR)，旨在通过对抗性学习进一步清除PII的重定向骨架。在我们的实验中，PMR实现了与最先进模型相当的运动重定目标实用性能，同时还降低了隐私攻击的性能。



## **41. Air Gap: Protecting Privacy-Conscious Conversational Agents**

空气间隙：保护有隐私意识的对话代理人 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **42. Filtering and smoothing estimation algorithms from uncertain nonlinear observations with time-correlated additive noise and random deception attacks**

来自具有时间相关添加性噪音和随机欺骗攻击的不确定非线性观测的过滤和平滑估计算法 eess.SP

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05157v1) [paper-pdf](http://arxiv.org/pdf/2405.05157v1)

**Authors**: R. Caballero-Águila, J. Hu, J. Linares-Pérez

**Abstract**: This paper discusses the problem of estimating a stochastic signal from nonlinear uncertain observations with time-correlated additive noise described by a first-order Markov process. Random deception attacks are assumed to be launched by an adversary, and both this phenomenon and the uncertainty in the observations are modelled by two sets of Bernoulli random variables. Under the assumption that the evolution model generating the signal to be estimated is unknown and only the mean and covariance functions of the processes involved in the observation equation are available, recursive algorithms based on linear approximations of the real observations are proposed for the least-squares filtering and fixed-point smoothing problems. Finally, the feasibility and effectiveness of the developed estimation algorithms are verified by a numerical simulation example, where the impact of uncertain observation and deception attack probabilities on estimation accuracy is evaluated.

摘要: 本文讨论了由一阶马尔科夫过程描述的具有时间相关添加性噪音的非线性不确定观测估计随机信号的问题。假设随机欺骗攻击是由对手发起的，这种现象和观察中的不确定性都是由两组伯努里随机变量建模的。在生成待估计信号的进化模型未知且只有观测方程中涉及的过程的均值和协方差函数可用的假设下，提出了基于真实观测值线性逼近的回归算法来解决最小平方过滤和定点平滑问题。最后，通过数值仿真算例验证了所开发的估计算法的可行性和有效性，评估了不确定观测和欺骗攻击概率对估计准确性的影响。



## **43. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

针对1_0美元有界对抗性扰动的稳健模型的有效训练和评估 cs.LG

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05075v1) [paper-pdf](http://arxiv.org/pdf/2405.05075v1)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.

摘要: 这项工作研究了以$l_0$规范为界的稀疏对抗扰动。我们提出了一种名为sparse-PVD的白盒类PGD攻击方法，以有效且高效地生成此类扰动。此外，我们将稀疏PVD与黑匣子攻击相结合，以全面、更可靠地评估模型对1_0美元有界对抗扰动的鲁棒性。此外，稀疏PVD的效率使我们能够进行对抗训练，以构建针对稀疏扰动的稳健模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。代码可访问https://github.com/CityU-MLO/sPGD。



## **44. Adversarial Threats to Automatic Modulation Open Set Recognition in Wireless Networks**

无线网络中自动调制开集识别的对抗威胁 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05022v1) [paper-pdf](http://arxiv.org/pdf/2405.05022v1)

**Authors**: Yandie Yang, Sicheng Zhang, Kuixian Li, Qiao Tian, Yun Lin

**Abstract**: Automatic Modulation Open Set Recognition (AMOSR) is a crucial technological approach for cognitive radio communications, wireless spectrum management, and interference monitoring within wireless networks. Numerous studies have shown that AMR is highly susceptible to minimal perturbations carefully designed by malicious attackers, leading to misclassification of signals. However, the adversarial security issue of AMOSR has not yet been explored. This paper adopts the perspective of attackers and proposes an Open Set Adversarial Attack (OSAttack), aiming at investigating the adversarial vulnerabilities of various AMOSR methods. Initially, an adversarial threat model for AMOSR scenarios is established. Subsequently, by analyzing the decision criteria of both discriminative and generative open set recognition, OSFGSM and OSPGD are proposed to reduce the performance of AMOSR. Finally, the influence of OSAttack on AMOSR is evaluated utilizing a range of qualitative and quantitative indicators. The results indicate that despite the increased resistance of AMOSR models to conventional interference signals, they remain vulnerable to attacks by adversarial examples.

摘要: 自动调制开集识别(AMOSR)是认知无线电通信、无线频谱管理和无线网络干扰监测的重要技术手段。大量研究表明，AMR非常容易受到恶意攻击者精心设计的微小扰动的影响，从而导致信号的错误分类。然而，AMOSR的对抗性安全问题尚未被探讨。本文从攻击者的角度出发，提出了一种开放集对抗性攻击(OSAttack)，旨在研究各种AMOSR方法的对抗性漏洞。首先，建立了AMOSR场景的对抗性威胁模型。随后，通过分析判别性和生成性开集识别的决策准则，提出了OSFGSM和OSPGD来降低AMOSR的性能。最后，利用一系列定性和定量指标对OSAttack对AMOSR的影响进行了评估。结果表明，尽管AMOSR模型对常规干扰信号的抵抗力有所增强，但它们仍然容易受到对手例子的攻击。



## **45. Deep Reinforcement Learning with Spiking Q-learning**

具有峰值Q学习的深度强化学习 cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2201.09754v3) [paper-pdf](http://arxiv.org/pdf/2201.09754v3)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.

摘要: 在特殊的神经形态硬件的帮助下，脉冲神经网络(SNN)有望以更少的能量消耗实现人工智能(AI)。它将神经网络和深度强化学习相结合，为实际控制任务提供了一种很有前途的节能方法。目前已有的基于SNN的RL方法很少。大多数人要么缺乏泛化能力，要么在训练中使用人工神经网络(ANN)来估计价值函数。前者需要针对每个场景调整大量的超参数，而后者限制了不同类型RL算法的应用，忽略了训练过程中的巨大能量消耗。为了开发一种稳健的基于棘波的RL方法，我们从昆虫中发现的非尖峰中间神经元中吸取灵感，提出了深度尖峰Q-网络(DSQN)，它使用非尖峰神经元的膜电压作为Q值的表示，可以使用端到端RL直接从高维感觉输入中学习鲁棒策略。在17个Atari游戏上的实验表明，DSQN是有效的，甚至在大多数游戏中都优于基于神经网络的深度Q网络(DQN)。实验表明，DSQN具有良好的学习稳定性和对敌意攻击的健壮性。



## **46. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2401.04929v2) [paper-pdf](http://arxiv.org/pdf/2401.04929v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



## **47. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

BiasKG：对抗性知识图在大型语言模型中诱导偏见 cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.

摘要: 现代大型语言模型（LLM）拥有大量的世界知识，如果利用得当，可以在常识推理和知识密集型任务中取得出色的性能。语言模型还可以学习社会偏见，这具有巨大的社会危害潜力。人们为LLM安全提出了许多缓解策略，但目前尚不清楚它们对于消除社会偏见的有效性如何。在这项工作中，我们提出了一种利用知识图增强生成来攻击语言模型的新方法。我们将自然语言刻板印象重新构建到知识图谱中，并使用对抗性攻击策略来诱导几个开放和封闭源语言模型的偏见反应。我们发现我们的方法增加了所有模型的偏差，甚至是那些接受过安全护栏训练的模型。这表明需要对人工智能安全进行进一步研究，并在这个新的对抗空间中进一步开展工作。



## **48. Demonstration of an Adversarial Attack Against a Multimodal Vision Language Model for Pathology Imaging**

演示针对病理成像多模式视觉语言模型的对抗攻击 eess.IV

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2401.02565v3) [paper-pdf](http://arxiv.org/pdf/2401.02565v3)

**Authors**: Poojitha Thota, Jai Prakash Veerla, Partha Sai Guttikonda, Mohammad S. Nasr, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the context of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted attacks. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial perturbation attacks to induce misclassifications intentionally. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models. The source codes for this experiment can be found at https://github.com/jaiprakash1824/VLM_Adv_Attack.

摘要: 在医学人工智能的背景下，本研究探索了视觉语言基础模型-病理语言-图像预训练(PLIP)模型在有针对性攻击下的脆弱性。利用Kather Colon数据集和9种组织类型的7,180张H&E图像，我们的研究使用了投影梯度下降(PGD)对抗性扰动攻击来故意诱导错误分类。结果显示，PLIP操纵预测的成功率为100%，突显出其易受对手干扰的影响。对抗性例子的定性分析深入到了可解释性的挑战，揭示了对抗性操纵导致的预测的细微变化。这些发现为医学成像中视觉语言模型的可解释性、领域适应性和可信性提供了重要的见解。该研究强调，迫切需要强大的防御措施，以确保人工智能模型的可靠性。这个实验的源代码可以在https://github.com/jaiprakash1824/VLM_Adv_Attack.上找到



## **49. Fully Automated Selfish Mining Analysis in Efficient Proof Systems Blockchains**

高效证明系统区块链中的全自动自私挖掘分析 cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04420v1) [paper-pdf](http://arxiv.org/pdf/2405.04420v1)

**Authors**: Krishnendu Chatterjee, Amirali Ebrahimzadeh, Mehrdad Karrabi, Krzysztof Pietrzak, Michelle Yeo, Đorđe Žikelić

**Abstract**: We study selfish mining attacks in longest-chain blockchains like Bitcoin, but where the proof of work is replaced with efficient proof systems -- like proofs of stake or proofs of space -- and consider the problem of computing an optimal selfish mining attack which maximizes expected relative revenue of the adversary, thus minimizing the chain quality. To this end, we propose a novel selfish mining attack that aims to maximize this objective and formally model the attack as a Markov decision process (MDP). We then present a formal analysis procedure which computes an $\epsilon$-tight lower bound on the optimal expected relative revenue in the MDP and a strategy that achieves this $\epsilon$-tight lower bound, where $\epsilon>0$ may be any specified precision. Our analysis is fully automated and provides formal guarantees on the correctness. We evaluate our selfish mining attack and observe that it achieves superior expected relative revenue compared to two considered baselines.   In concurrent work [Sarenche FC'24] does an automated analysis on selfish mining in predictable longest-chain blockchains based on efficient proof systems. Predictable means the randomness for the challenges is fixed for many blocks (as used e.g., in Ouroboros), while we consider unpredictable (Bitcoin-like) chains where the challenge is derived from the previous block.

摘要: 我们研究了比特币等最长链区块链中的自私挖掘攻击，但工作证明被高效的证明系统取代--如赌注证明或空间证明--并考虑计算最优自私挖掘攻击的问题，该攻击最大化对手的预期相对收益，从而最小化链质量。为此，我们提出了一种新的自私挖掘攻击，旨在最大化这一目标，并将攻击形式化地建模为马尔可夫决策过程(MDP)。然后，我们给出了一个形式的分析程序，它计算了MDP中最优预期相对收益的$\epsilon$-紧下界，并给出了一个实现这个$\epsilon$-紧下界的策略，其中$\epsilon>0$可以是任意指定的精度。我们的分析是完全自动化的，并为正确性提供正式保证。我们评估了我们的自私挖掘攻击，并观察到与两个考虑的基线相比，它实现了更好的预期相对收益。在并发工作[Sarhene FC‘24]中，基于高效的证明系统，对可预测的最长链区块链中的自私挖掘进行了自动化分析。可预测意味着挑战的随机性对于许多区块是固定的(例如，在Ouroboros中使用)，而我们认为挑战来自前一个区块的不可预测(类似比特币的)链。



## **50. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：基于脑电波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2402.08656v4) [paper-pdf](http://arxiv.org/pdf/2402.08656v4)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



