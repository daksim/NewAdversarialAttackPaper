# Latest Adversarial Attack Papers
**update at 2023-12-18 09:51:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Coevolutionary Algorithm for Building Robust Decision Trees under Minimax Regret**

极小极大后悔条件下构建稳健决策树的协进化算法 cs.LG

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09078v1) [paper-pdf](http://arxiv.org/pdf/2312.09078v1)

**Authors**: Adam Żychowski, Andrew Perrault, Jacek Mańdziuk

**Abstract**: In recent years, there has been growing interest in developing robust machine learning (ML) models that can withstand adversarial attacks, including one of the most widely adopted, efficient, and interpretable ML algorithms-decision trees (DTs). This paper proposes a novel coevolutionary algorithm (CoEvoRDT) designed to create robust DTs capable of handling noisy high-dimensional data in adversarial contexts. Motivated by the limitations of traditional DT algorithms, we leverage adaptive coevolution to allow DTs to evolve and learn from interactions with perturbed input data. CoEvoRDT alternately evolves competing populations of DTs and perturbed features, enabling construction of DTs with desired properties. CoEvoRDT is easily adaptable to various target metrics, allowing the use of tailored robustness criteria such as minimax regret. Furthermore, CoEvoRDT has potential to improve the results of other state-of-the-art methods by incorporating their outcomes (DTs they produce) into the initial population and optimize them in the process of coevolution. Inspired by the game theory, CoEvoRDT utilizes mixed Nash equilibrium to enhance convergence. The method is tested on 20 popular datasets and shows superior performance compared to 4 state-of-the-art algorithms. It outperformed all competing methods on 13 datasets with adversarial accuracy metrics, and on all 20 considered datasets with minimax regret. Strong experimental results and flexibility in choosing the error measure make CoEvoRDT a promising approach for constructing robust DTs in real-world applications.

摘要: 近年来，人们对开发能够抵抗对手攻击的健壮机器学习(ML)模型越来越感兴趣，其中包括最广泛采用的、高效的和可解释的ML算法之一-决策树(DTD)。提出了一种新的协同进化算法(CoEvoRDT)，旨在创建能够在对抗性环境中处理噪声高维数据的健壮DT。由于传统DT算法的局限性，我们利用自适应协同进化来允许DT进化，并从与扰动输入数据的交互中学习。CoEvoRDT交替进化相互竞争的DT种群和受干扰的特征，从而能够构建具有所需特性的DT。CoEvoRDT很容易适应各种目标指标，允许使用定制的健壮性标准，如最小最大遗憾。此外，CoEvoRDT通过将其他最先进方法的结果(它们产生的DT)合并到初始种群中并在共同进化过程中对其进行优化，具有改进其他最先进方法的结果的潜力。受博弈论的启发，CoEvoRDT利用混合纳什均衡来增强收敛。该方法在20个流行的数据集上进行了测试，与4种最先进的算法相比，表现出了更好的性能。它在13个带有对抗性准确度指标的数据集上的表现优于所有竞争方法，在所有20个考虑的数据集上的表现都是最小最大遗憾。强大的实验结果和选择误差度量的灵活性使CoEvoRDT成为在实际应用中构造健壮DTD的一种很有前途的方法。



## **2. Concealing Sensitive Samples against Gradient Leakage in Federated Learning**

联合学习中防止梯度泄漏的敏感样本隐藏 cs.LG

Defence against model inversion attack in federated learning

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2209.05724v2) [paper-pdf](http://arxiv.org/pdf/2209.05724v2)

**Authors**: Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi

**Abstract**: Federated Learning (FL) is a distributed learning paradigm that enhances users privacy by eliminating the need for clients to share raw, private data with the server. Despite the success, recent studies expose the vulnerability of FL to model inversion attacks, where adversaries reconstruct users private data via eavesdropping on the shared gradient information. We hypothesize that a key factor in the success of such attacks is the low entanglement among gradients per data within the batch during stochastic optimization. This creates a vulnerability that an adversary can exploit to reconstruct the sensitive data. Building upon this insight, we present a simple, yet effective defense strategy that obfuscates the gradients of the sensitive data with concealed samples. To achieve this, we propose synthesizing concealed samples to mimic the sensitive data at the gradient level while ensuring their visual dissimilarity from the actual sensitive data. Compared to the previous art, our empirical evaluations suggest that the proposed technique provides the strongest protection while simultaneously maintaining the FL performance.

摘要: Federated Learning（FL）是一种分布式学习范式，通过消除客户端与服务器共享原始私有数据的需要来增强用户隐私。尽管取得了成功，但最近的研究暴露了FL对模型反演攻击的脆弱性，其中对手通过窃听共享的梯度信息来重建用户的私人数据。我们假设，这种攻击成功的一个关键因素是在随机优化过程中，批次内每个数据的梯度之间的低纠缠。这就产生了一个漏洞，攻击者可以利用这个漏洞来重建敏感数据。基于这一认识，我们提出了一个简单而有效的防御策略，用隐藏的样本模糊敏感数据的梯度。为了实现这一目标，我们建议合成隐藏的样本，以模仿梯度级别的敏感数据，同时确保它们与实际敏感数据的视觉差异。与以前的技术相比，我们的经验评估表明，所提出的技术提供了最强的保护，同时保持FL性能。



## **3. DRAM-Locker: A General-Purpose DRAM Protection Mechanism against Adversarial DNN Weight Attacks**

DRAM-Locker：一种抵抗敌意DNN权重攻击的通用DRAM保护机制 cs.AR

7 pages. arXiv admin note: text overlap with arXiv:2305.08034

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.09027v1) [paper-pdf](http://arxiv.org/pdf/2312.09027v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Arman Roohi, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: In this work, we propose DRAM-Locker as a robust general-purpose defense mechanism that can protect DRAM against various adversarial Deep Neural Network (DNN) weight attacks affecting data or page tables. DRAM-Locker harnesses the capabilities of in-DRAM swapping combined with a lock-table to prevent attackers from singling out specific DRAM rows to safeguard DNN's weight parameters. Our results indicate that DRAM-Locker can deliver a high level of protection downgrading the performance of targeted weight attacks to a random attack level. Furthermore, the proposed defense mechanism demonstrates no reduction in accuracy when applied to CIFAR-10 and CIFAR-100. Importantly, DRAM-Locker does not necessitate any software retraining or result in extra hardware burden.

摘要: 在这项工作中，我们提出了DRAM-Locker作为一种健壮的通用防御机制，可以保护DRAM免受各种影响数据或页表的对抗性深度神经网络(DNN)权重攻击。DRAM-Locker利用DRAM内交换和锁定表相结合的功能，防止攻击者挑出特定的DRAM行来保护DNN的重量参数。我们的结果表明，DRAM-Locker可以提供高级别的保护，将目标权重攻击的性能降低到随机攻击级别。此外，建议的防御机制在应用于CIFAR-10和CIFAR-100时不会降低精度。重要的是，DRAM-Locker不需要任何软件再培训或导致额外的硬件负担。



## **4. Amicable Aid: Perturbing Images to Improve Classification Performance**

友好的帮助：扰动图像以提高分类性能 cs.CV

ICASSP 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2112.04720v4) [paper-pdf](http://arxiv.org/pdf/2112.04720v4)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstract**: While adversarial perturbation of images to attack deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of image perturbation can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be modified to yield higher classification confidence and even a misclassified image can be made correctly classified. This can be also achieved with a large amount of perturbation by which the image is made unrecognizable by human eyes. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. Furthermore, we investigate the universal amicable aid, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found.

摘要: 虽然在实践中对图像的敌意扰动攻击深度图像分类模型带来了严重的安全问题，但本文提出了一种新的范例，其中图像扰动的概念有助于提高分类性能，我们称之为友好辅助。我们证明，通过采取与扰动相反的搜索方向，可以对图像进行修改以产生更高的分类置信度，甚至可以对错误分类的图像进行正确分类。这也可以通过大量的扰动来实现，通过这种扰动，人眼无法识别图像。从潜在的自然意象流形的角度解释了友好相助的机制。此外，我们还研究了一种通用的友好辅助方法，即对多幅图像施加一个固定的扰动来改善它们的分类结果。虽然很难找到这样的扰动，但我们表明，通过用修改后的数据训练使决策边界尽可能垂直于图像流形，可以有效地获得一个更容易找到普遍友好扰动的模型。



## **5. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

禁忌事实：骆驼2号中相互竞争的目标的调查 cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08793v1) [paper-pdf](http://arxiv.org/pdf/2312.08793v1)

**Authors**: Tony T. Wang, Miles Wang, Kaivu Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .

摘要: LLM经常面临竞争压力（例如有益与无害）。为了理解模型如何解决这种冲突，我们研究了Llama-2-chat模型的禁止事实任务。具体来说，我们指导Llama-2如实地完成一个事实回忆陈述，同时禁止它说出正确的答案。这通常会导致模型给出错误的答案。我们将Llama-2分解为1000多个组件，并根据每个组件对禁止正确答案的有用程度进行排名。我们发现，总的来说，大约35个组件足以可靠地实现完全抑制行为。然而，这些组件是相当异构的，并且许多组件使用错误的语法操作。我们发现其中一种攻击可以通过手动设计的对抗性攻击来利用，我们称之为加利福尼亚攻击。我们的研究结果突出了一些阻碍成功解释高级ML系统的障碍。项目网站：https://forbiddenfacts.github.io。



## **6. AVA: Inconspicuous Attribute Variation-based Adversarial Attack bypassing DeepFake Detection**

AVA：绕过DeepFake检测的基于不明显属性变化的敌意攻击 cs.CV

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08675v1) [paper-pdf](http://arxiv.org/pdf/2312.08675v1)

**Authors**: Xiangtao Meng, Li Wang, Shanqing Guo, Lei Ju, Qingchuan Zhao

**Abstract**: While DeepFake applications are becoming popular in recent years, their abuses pose a serious privacy threat. Unfortunately, most related detection algorithms to mitigate the abuse issues are inherently vulnerable to adversarial attacks because they are built atop DNN-based classification models, and the literature has demonstrated that they could be bypassed by introducing pixel-level perturbations. Though corresponding mitigation has been proposed, we have identified a new attribute-variation-based adversarial attack (AVA) that perturbs the latent space via a combination of Gaussian prior and semantic discriminator to bypass such mitigation. It perturbs the semantics in the attribute space of DeepFake images, which are inconspicuous to human beings (e.g., mouth open) but can result in substantial differences in DeepFake detection. We evaluate our proposed AVA attack on nine state-of-the-art DeepFake detection algorithms and applications. The empirical results demonstrate that AVA attack defeats the state-of-the-art black box attacks against DeepFake detectors and achieves more than a 95% success rate on two commercial DeepFake detectors. Moreover, our human study indicates that AVA-generated DeepFake images are often imperceptible to humans, which presents huge security and privacy concerns.

摘要: 虽然DeepFake应用程序近年来越来越流行，但它们的滥用构成了严重的隐私威胁。不幸的是，大多数缓解滥用问题的相关检测算法本质上容易受到对抗性攻击，因为它们是建立在基于DNN的分类模型之上的，并且文献已经证明，它们可以通过引入像素级扰动来绕过。虽然已经提出了相应的缓解措施，但我们已经确定了一种新的基于属性变化的对抗性攻击（AVA），该攻击通过高斯先验和语义匹配的组合来扰乱潜在空间，以绕过这种缓解措施。它扰乱了DeepFake图像属性空间中的语义，这些图像对人类来说是不显眼的（例如，嘴巴张开），但可能导致DeepFake检测的实质性差异。我们对九种最先进的DeepFake检测算法和应用程序评估了我们提出的AVA攻击。实验结果表明，AVA攻击击败了针对DeepFake检测器的最先进的黑盒攻击，并在两个商业DeepFake检测器上实现了超过95%的成功率。此外，我们的人类研究表明，AVA生成的DeepFake图像通常是人类无法感知的，这带来了巨大的安全和隐私问题。



## **7. AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models**

AutoDAN：对大型语言模型的可解释的基于梯度的对抗性攻击 cs.CR

Version 2 updates: Added comparison of three more evaluation methods  and their reliability check using human labeling. Added results for  jailbreaking Llama2 (individual behavior) and included complexity and  hyperparameter analysis. Revised objectives for prompt leaking. Other minor  changes made

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2310.15140v2) [paper-pdf](http://arxiv.org/pdf/2310.15140v2)

**Authors**: Sicheng Zhu, Ruiyi Zhang, Bang An, Gang Wu, Joe Barrow, Zichao Wang, Furong Huang, Ani Nenkova, Tong Sun

**Abstract**: Safety alignment of Large Language Models (LLMs) can be compromised with manual jailbreak attacks and (automatic) adversarial attacks. Recent studies suggest that defending against these attacks is possible: adversarial attacks generate unlimited but unreadable gibberish prompts, detectable by perplexity-based filters; manual jailbreak attacks craft readable prompts, but their limited number due to the necessity of human creativity allows for easy blocking. In this paper, we show that these solutions may be too optimistic. We introduce AutoDAN, an interpretable, gradient-based adversarial attack that merges the strengths of both attack types. Guided by the dual goals of jailbreak and readability, AutoDAN optimizes and generates tokens one by one from left to right, resulting in readable prompts that bypass perplexity filters while maintaining high attack success rates. Notably, these prompts, generated from scratch using gradients, are interpretable and diverse, with emerging strategies commonly seen in manual jailbreak attacks. They also generalize to unforeseen harmful behaviors and transfer to black-box LLMs better than their unreadable counterparts when using limited training data or a single proxy model. Furthermore, we show the versatility of AutoDAN by automatically leaking system prompts using a customized objective. Our work offers a new way to red-team LLMs and understand jailbreak mechanisms via interpretability.

摘要: 大型语言模型(LLM)的安全对齐可能会受到手动越狱攻击和(自动)对抗性攻击的影响。最近的研究表明，防御这些攻击是可能的：对抗性攻击生成无限但不可读的胡言乱语提示，可通过基于困惑的过滤器检测；手动越狱攻击创建可读的提示，但由于人类创造力的必要性，其数量有限，允许轻松阻止。在本文中，我们证明了这些解决方案可能过于乐观。我们介绍了AutoDAN，一种可解释的、基于梯度的对抗性攻击，它融合了这两种攻击类型的优点。在越狱和可读性双重目标的指导下，AutoDAN从左到右一个接一个地优化和生成令牌，产生可读的提示，绕过困惑过滤器，同时保持高攻击成功率。值得注意的是，这些使用渐变从零开始生成的提示是可解释的和多样化的，新出现的策略通常出现在手动越狱攻击中。当使用有限的训练数据或单一代理模型时，它们还概括到不可预见的有害行为，并比不可读的同行更好地转移到黑盒LLM。此外，我们通过使用定制目标自动泄漏系统提示来展示AutoDAN的多功能性。我们的工作为红色团队LLM提供了一种新的方法，并通过可解释性来理解越狱机制。



## **8. Towards Inductive Robustness: Distilling and Fostering Wave-induced Resonance in Transductive GCNs Against Graph Adversarial Attacks**

通向诱导健壮性：提取和培育传导性GCNS中对抗图攻击的波诱导共振 cs.LG

AAAI 2024

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08651v1) [paper-pdf](http://arxiv.org/pdf/2312.08651v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Pan Zhou

**Abstract**: Graph neural networks (GNNs) have recently been shown to be vulnerable to adversarial attacks, where slight perturbations in the graph structure can lead to erroneous predictions. However, current robust models for defending against such attacks inherit the transductive limitations of graph convolutional networks (GCNs). As a result, they are constrained by fixed structures and do not naturally generalize to unseen nodes. Here, we discover that transductive GCNs inherently possess a distillable robustness, achieved through a wave-induced resonance process. Based on this, we foster this resonance to facilitate inductive and robust learning. Specifically, we first prove that the signal formed by GCN-driven message passing (MP) is equivalent to the edge-based Laplacian wave, where, within a wave system, resonance can naturally emerge between the signal and its transmitting medium. This resonance provides inherent resistance to malicious perturbations inflicted on the signal system. We then prove that merely three MP iterations within GCNs can induce signal resonance between nodes and edges, manifesting as a coupling between nodes and their distillable surrounding local subgraph. Consequently, we present Graph Resonance-fostering Network (GRN) to foster this resonance via learning node representations from their distilled resonating subgraphs. By capturing the edge-transmitted signals within this subgraph and integrating them with the node signal, GRN embeds these combined signals into the central node's representation. This node-wise embedding approach allows for generalization to unseen nodes. We validate our theoretical findings with experiments, and demonstrate that GRN generalizes robustness to unseen nodes, whilst maintaining state-of-the-art classification accuracy on perturbed graphs.

摘要: 图神经网络（GNN）最近被证明容易受到对抗性攻击，图结构中的轻微扰动可能导致错误的预测。然而，目前用于防御此类攻击的鲁棒模型继承了图卷积网络（GCN）的转换限制。因此，它们受到固定结构的约束，并且不会自然地推广到不可见的节点。在这里，我们发现，transductive GCN固有地具有可伸缩的鲁棒性，通过波诱导的共振过程实现。在此基础上，我们培养这种共鸣，以促进归纳和鲁棒学习。具体来说，我们首先证明了GCN驱动的消息传递（MP）形成的信号是等价的基于边缘的拉普拉斯波，其中，在波系统中，谐振可以自然地出现在信号和它的传输介质之间。这种谐振提供了对施加在信号系统上的恶意扰动的固有抵抗力。然后，我们证明，只有三个MP迭代GCN内可以诱导节点和边缘之间的信号共振，表现为节点和它们的周围的局部子图之间的耦合。因此，我们提出了图共振促进网络（GRN），以促进这种共振，通过学习节点表示从他们的蒸馏共振子图。通过捕获该子图中的边缘传输信号并将其与节点信号集成，GRN将这些组合信号嵌入到中心节点的表示中。这种逐节点嵌入方法允许推广到不可见的节点。我们用实验验证了我们的理论研究结果，并证明了GRN将鲁棒性推广到看不见的节点，同时保持了扰动图的最新分类精度。



## **9. Guarding the Grid: Enhancing Resilience in Automated Residential Demand Response Against False Data Injection Attacks**

保护网格：增强住宅需求自动响应对虚假数据注入攻击的弹性 eess.SY

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08646v1) [paper-pdf](http://arxiv.org/pdf/2312.08646v1)

**Authors**: Thusitha Dayaratne, Carsten Rudolph, Ariel Liebman, Mahsa Salehi

**Abstract**: Utility companies are increasingly leveraging residential demand flexibility and the proliferation of smart/IoT devices to enhance the effectiveness of residential demand response (DR) programs through automated device scheduling. However, the adoption of distributed architectures in these systems exposes them to the risk of false data injection attacks (FDIAs), where adversaries can manipulate decision-making processes by injecting false data. Given the limited control utility companies have over these distributed systems and data, the need for reliable implementations to enhance the resilience of residential DR schemes against FDIAs is paramount. In this work, we present a comprehensive framework that combines DR optimisation, anomaly detection, and strategies for mitigating the impacts of attacks to create a resilient and automated device scheduling system. To validate the robustness of our framework against FDIAs, we performed an evaluation using real-world data sets, highlighting its effectiveness in securing residential DR systems.

摘要: 公用事业公司越来越多地利用住宅需求灵活性和智能/物联网设备的激增，通过自动化设备调度来增强住宅需求响应(DR)计划的有效性。然而，在这些系统中采用分布式体系结构使它们面临虚假数据注入攻击(FDIA)的风险，在FDIA中，对手可以通过注入虚假数据来操纵决策过程。鉴于公用事业公司对这些分布式系统和数据的控制有限，需要可靠的实施来增强住宅灾难恢复计划对FDIA的弹性。在这项工作中，我们提出了一个全面的框架，该框架结合了灾难恢复优化、异常检测和减轻攻击影响的策略，以创建一个弹性和自动化的设备调度系统。为了验证我们的框架针对FDIA的稳健性，我们使用真实世界的数据集进行了评估，突出了其在保护住宅DR系统方面的有效性。



## **10. Scalable Ensemble-based Detection Method against Adversarial Attacks for speaker verification**

基于可扩展集成的说话人确认对抗攻击检测方法 eess.AS

Submitted to 2024 ICASSP

**SubmitDate**: 2023-12-14    [abs](http://arxiv.org/abs/2312.08622v1) [paper-pdf](http://arxiv.org/pdf/2312.08622v1)

**Authors**: Haibin Wu, Heng-Cheng Kuo, Yu Tsao, Hung-yi Lee

**Abstract**: Automatic speaker verification (ASV) is highly susceptible to adversarial attacks. Purification modules are usually adopted as a pre-processing to mitigate adversarial noise. However, they are commonly implemented across diverse experimental settings, rendering direct comparisons challenging. This paper comprehensively compares mainstream purification techniques in a unified framework. We find these methods often face a trade-off between user experience and security, as they struggle to simultaneously maintain genuine sample performance and reduce adversarial perturbations. To address this challenge, some efforts have extended purification modules to encompass detection capabilities, aiming to alleviate the trade-off. However, advanced purification modules will always come into the stage to surpass previous detection method. As a result, we further propose an easy-to-follow ensemble approach that integrates advanced purification modules for detection, achieving state-of-the-art (SOTA) performance in countering adversarial noise. Our ensemble method has great potential due to its compatibility with future advanced purification techniques.

摘要: 自动说话人验证(ASV)是一种易受敌意攻击的技术。为了减少对抗性噪声，通常采用净化模块作为预处理。然而，它们通常是在不同的实验环境中实施的，这使得直接比较具有挑战性。本文在统一的框架内对主流净化技术进行了综合比较。我们发现，这些方法经常面临用户体验和安全性之间的权衡，因为它们难以同时保持真正的样本性能和减少对抗性干扰。为了应对这一挑战，一些努力扩展了净化模块，以包含检测能力，旨在缓解权衡。然而，先进的净化模块总会出现，以超越以往的检测方法。因此，我们进一步提出了一种易于遵循的集成方法，该方法集成了用于检测的高级净化模块，在对抗对抗性噪声方面实现了最先进的性能(SOTA)。我们的集成方法具有很大的潜力，因为它与未来的先进纯化技术兼容。



## **11. Exploring the Privacy Risks of Adversarial VR Game Design**

对抗性VR游戏设计中的隐私风险探讨 cs.CR

Learn more at https://rdi.berkeley.edu/metaverse/metadata

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2207.13176v4) [paper-pdf](http://arxiv.org/pdf/2207.13176v4)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song, James F. O'Brien

**Abstract**: Fifty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Within just a few minutes, an adversarial program had accurately inferred over 25 of their personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. Since the Cambridge Analytica scandal of 2018, adversarially designed gamified elements have been known to constitute a significant privacy threat in conventional social platforms. In this work, we present a case study of how metaverse environments can similarly be adversarially constructed to covertly infer dozens of personal data attributes from seemingly anonymous users. While existing VR privacy research largely focuses on passive observation, we argue that because individuals subconsciously reveal personal information via their motion in response to specific stimuli, active attacks pose an outsized risk in VR environments.

摘要: 50名研究参与者在虚拟现实（VR）中玩了一个看似无辜的“密室逃脱”游戏。在短短几分钟内，一个对抗程序就准确地推断出了他们超过25个个人数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计数据。随着数据饥渴的公司越来越多地参与VR开发，这个实验场景可能很快就会代表典型的VR用户体验。自2018年剑桥分析公司丑闻以来，已知对抗性设计的游戏化元素在传统社交平台中构成了重大的隐私威胁。在这项工作中，我们提出了一个案例研究，如何metaverse环境可以类似地被构造成从看似匿名的用户中秘密地推断出数十个个人数据属性。虽然现有的VR隐私研究主要集中在被动观察上，但我们认为，由于个人通过对特定刺激的反应而下意识地透露个人信息，因此主动攻击在VR环境中构成了巨大的风险。



## **12. Defenses in Adversarial Machine Learning: A Survey**

对抗性机器学习中的防御：综述 cs.CV

21 pages, 5 figures, 2 tables, 237 reference papers

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08890v1) [paper-pdf](http://arxiv.org/pdf/2312.08890v1)

**Authors**: Baoyuan Wu, Shaokui Wei, Mingli Zhu, Meixi Zheng, Zihao Zhu, Mingda Zhang, Hongrui Chen, Danni Yuan, Li Liu, Qingshan Liu

**Abstract**: Adversarial phenomenon has been widely observed in machine learning (ML) systems, especially in those using deep neural networks, describing that ML systems may produce inconsistent and incomprehensible predictions with humans at some particular cases. This phenomenon poses a serious security threat to the practical application of ML systems, and several advanced attack paradigms have been developed to explore it, mainly including backdoor attacks, weight attacks, and adversarial examples. For each individual attack paradigm, various defense paradigms have been developed to improve the model robustness against the corresponding attack paradigm. However, due to the independence and diversity of these defense paradigms, it is difficult to examine the overall robustness of an ML system against different kinds of attacks.This survey aims to build a systematic review of all existing defense paradigms from a unified perspective. Specifically, from the life-cycle perspective, we factorize a complete machine learning system into five stages, including pre-training, training, post-training, deployment, and inference stages, respectively. Then, we present a clear taxonomy to categorize and review representative defense methods at each individual stage. The unified perspective and presented taxonomies not only facilitate the analysis of the mechanism of each defense paradigm but also help us to understand connections and differences among different defense paradigms, which may inspire future research to develop more advanced, comprehensive defenses.

摘要: 对抗现象在机器学习（ML）系统中被广泛观察到，特别是在使用深度神经网络的系统中，描述了ML系统在某些特定情况下可能会产生与人类不一致和不可理解的预测。这种现象对ML系统的实际应用构成了严重的安全威胁，已经开发了几种先进的攻击范式来探索它，主要包括后门攻击，权重攻击和对抗性示例。对于每个单独的攻击范例，已经开发了各种防御范例，以提高模型对相应攻击范例的鲁棒性。然而，由于这些防御范式的独立性和多样性，很难检查ML系统对不同类型的攻击的整体鲁棒性。本调查旨在从统一的角度对所有现有的防御范式进行系统的回顾。具体来说，从生命周期的角度来看，我们将一个完整的机器学习系统分解为五个阶段，分别包括预训练、训练、后训练、部署和推理阶段。然后，我们提出了一个明确的分类法，分类和审查代表性的防御方法在每个单独的阶段。这种统一的视角和分类不仅有助于分析每种防御范式的机制，而且有助于我们理解不同防御范式之间的联系和差异，这可能会启发未来的研究开发更先进，更全面的防御。



## **13. Universal Adversarial Framework to Improve Adversarial Robustness for Diabetic Retinopathy Detection**

提高糖尿病视网膜病变检测中对抗稳健性的通用对抗框架 eess.IV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08193v1) [paper-pdf](http://arxiv.org/pdf/2312.08193v1)

**Authors**: Samrat Mukherjee, Dibyanayan Bandyopadhyay, Baban Gain, Asif Ekbal

**Abstract**: Diabetic Retinopathy (DR) is a prevalent illness associated with Diabetes which, if left untreated, can result in irreversible blindness. Deep Learning based systems are gradually being introduced as automated support for clinical diagnosis. Since healthcare has always been an extremely important domain demanding error-free performance, any adversaries could pose a big threat to the applicability of such systems. In this work, we use Universal Adversarial Perturbations (UAPs) to quantify the vulnerability of Medical Deep Neural Networks (DNNs) for detecting DR. To the best of our knowledge, this is the very first attempt that works on attacking complete fine-grained classification of DR images using various UAPs. Also, as a part of this work, we use UAPs to fine-tune the trained models to defend against adversarial samples. We experiment on several models and observe that the performance of such models towards unseen adversarial attacks gets boosted on average by $3.41$ Cohen-kappa value and maximum by $31.92$ Cohen-kappa value. The performance degradation on normal data upon ensembling the fine-tuned models was found to be statistically insignificant using t-test, highlighting the benefits of UAP-based adversarial fine-tuning.

摘要: 糖尿病视网膜病变(DR)是一种与糖尿病相关的流行疾病，如果不治疗，可能导致不可逆转的失明。基于深度学习的系统正逐渐被引入，作为临床诊断的自动化支持。由于医疗保健一直是一个要求无差错性能的极其重要的领域，任何对手都可能对此类系统的适用性构成巨大威胁。在这项工作中，我们使用通用对抗扰动(UAP)来量化医学深层神经网络(DNNS)检测DR的脆弱性。据我们所知，这是使用各种UAP攻击DR图像的完全细粒度分类的第一次尝试。此外，作为这项工作的一部分，我们使用UAP来微调训练的模型，以防御对手样本。我们在几个模型上进行了实验，观察到这些模型对看不见的对抗性攻击的性能平均提高了3.41美元Cohen-kappa值，最大值提高了31.92美元Cohen-kappa值。使用t检验发现，集成微调模型后对正常数据的性能降级在统计上微不足道，突出了基于UAP的对抗性微调的好处。



## **14. Adversarial Attacks on Graph Neural Networks based Spatial Resource Management in P2P Wireless Communications**

P2P无线通信中基于图神经网络空间资源管理的对抗性攻击 eess.SP

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08181v1) [paper-pdf](http://arxiv.org/pdf/2312.08181v1)

**Authors**: Ahmad Ghasemi, Ehsan Zeraatkar, Majid Moradikia, Seyed, Zekavat

**Abstract**: This paper introduces adversarial attacks targeting a Graph Neural Network (GNN) based radio resource management system in point to point (P2P) communications. Our focus lies on perturbing the trained GNN model during the test phase, specifically targeting its vertices and edges. To achieve this, four distinct adversarial attacks are proposed, each accounting for different constraints, and aiming to manipulate the behavior of the system. The proposed adversarial attacks are formulated as optimization problems, aiming to minimize the system's communication quality. The efficacy of these attacks is investigated against the number of users, signal-to-noise ratio (SNR), and adversary power budget. Furthermore, we address the detection of such attacks from the perspective of the Central Processing Unit (CPU) of the system. To this end, we formulate an optimization problem that involves analyzing the distribution of channel eigenvalues before and after the attacks are applied. This formulation results in a Min-Max optimization problem, allowing us to detect the presence of attacks. Through extensive simulations, we observe that in the absence of adversarial attacks, the eigenvalues conform to Johnson's SU distribution. However, the attacks significantly alter the characteristics of the eigenvalue distribution, and in the most effective attack, they even change the type of the eigenvalue distribution.

摘要: 介绍了点对点(P2P)通信中基于图神经网络(GNN)的无线资源管理系统的对抗性攻击。我们的重点在于在测试阶段对训练好的GNN模型进行扰动，特别是针对其顶点和边。为了实现这一点，提出了四种不同的对抗性攻击，每一种攻击都考虑了不同的约束，旨在操纵系统的行为。所提出的对抗性攻击被描述为优化问题，目标是最小化系统的通信质量。这些攻击的有效性是根据用户数、信噪比(SNR)和对手功率预算进行调查的。此外，我们从系统的中央处理单元(CPU)的角度来解决此类攻击的检测问题。为此，我们提出了一个优化问题，包括分析攻击实施前后信道特征值的分布。这个公式导致了最小-最大优化问题，使我们能够检测攻击的存在。通过大量的仿真，我们观察到在没有对手攻击的情况下，特征值服从Johnson的SU分布。然而，攻击显著地改变了特征值分布的特征，在最有效的攻击中，它们甚至改变了特征值分布的类型。



## **15. Efficient Representation of the Activation Space in Deep Neural Networks**

深度神经网络中激活空间的有效表示 cs.LG

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.08143v1) [paper-pdf](http://arxiv.org/pdf/2312.08143v1)

**Authors**: Tanya Akumu, Celia Cintas, Girmaw Abebe Tadesse, Adebayo Oshingbesan, Skyler Speakman, Edward McFowland III

**Abstract**: The representations of the activation space of deep neural networks (DNNs) are widely utilized for tasks like natural language processing, anomaly detection and speech recognition. Due to the diverse nature of these tasks and the large size of DNNs, an efficient and task-independent representation of activations becomes crucial. Empirical p-values have been used to quantify the relative strength of an observed node activation compared to activations created by already-known inputs. Nonetheless, keeping raw data for these calculations increases memory resource consumption and raises privacy concerns. To this end, we propose a model-agnostic framework for creating representations of activations in DNNs using node-specific histograms to compute p-values of observed activations without retaining already-known inputs. Our proposed approach demonstrates promising potential when validated with multiple network architectures across various downstream tasks and compared with the kernel density estimates and brute-force empirical baselines. In addition, the framework reduces memory usage by 30% with up to 4 times faster p-value computing time while maintaining state of-the-art detection power in downstream tasks such as the detection of adversarial attacks and synthesized content. Moreover, as we do not persist raw data at inference time, we could potentially reduce susceptibility to attacks and privacy issues.

摘要: 深度神经网络（DNN）的激活空间表示被广泛用于自然语言处理、异常检测和语音识别等任务。由于这些任务的多样性和DNN的大规模，激活的有效和任务无关的表示变得至关重要。经验p值已用于量化观察到的节点激活与已知输入创建的激活相比的相对强度。尽管如此，保留这些计算的原始数据会增加内存资源消耗并引发隐私问题。为此，我们提出了一个与模型无关的框架，用于使用节点特定的直方图在DNN中创建激活的表示，以计算观察到的激活的p值，而不保留已知的输入。我们提出的方法在跨各种下游任务的多个网络架构进行验证时，并与内核密度估计和蛮力经验基线进行比较时，表现出了很好的潜力。此外，该框架将内存使用量减少了30%，p值计算时间加快了4倍，同时在下游任务中保持了最先进的检测能力，例如检测对抗性攻击和合成内容。此外，由于我们在推理时不保存原始数据，因此可能会降低对攻击和隐私问题的敏感性。



## **16. Robust Few-Shot Named Entity Recognition with Boundary Discrimination and Correlation Purification**

基于边界判别和关联提纯的健壮少镜头命名实体识别 cs.CL

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07961v1) [paper-pdf](http://arxiv.org/pdf/2312.07961v1)

**Authors**: Xiaojun Xue, Chunxia Zhang, Tianxiang Xu, Zhendong Niu

**Abstract**: Few-shot named entity recognition (NER) aims to recognize novel named entities in low-resource domains utilizing existing knowledge. However, the present few-shot NER models assume that the labeled data are all clean without noise or outliers, and there are few works focusing on the robustness of the cross-domain transfer learning ability to textual adversarial attacks in Few-shot NER. In this work, we comprehensively explore and assess the robustness of few-shot NER models under textual adversarial attack scenario, and found the vulnerability of existing few-shot NER models. Furthermore, we propose a robust two-stage few-shot NER method with Boundary Discrimination and Correlation Purification (BDCP). Specifically, in the span detection stage, the entity boundary discriminative module is introduced to provide a highly distinguishing boundary representation space to detect entity spans. In the entity typing stage, the correlations between entities and contexts are purified by minimizing the interference information and facilitating correlation generalization to alleviate the perturbations caused by textual adversarial attacks. In addition, we construct adversarial examples for few-shot NER based on public datasets Few-NERD and Cross-Dataset. Comprehensive evaluations on those two groups of few-shot NER datasets containing adversarial examples demonstrate the robustness and superiority of the proposed method.

摘要: 少镜头命名实体识别(NER)旨在利用现有知识识别低资源领域中的新命名实体。然而，目前的少射NER模型都假设标记的数据都是干净的，没有噪声或离群点，而很少有人关注在少射NER中跨域迁移学习能力对文本攻击的健壮性。在这项工作中，我们全面探索和评估了文本对抗攻击场景下的少镜头NER模型的健壮性，发现了现有的少镜头NER模型的脆弱性。此外，我们还提出了一种具有边界识别和相关净化(BDCP)的健壮两阶段少镜头NER方法。具体来说，在跨度检测阶段，引入了实体边界判别模块，为检测实体跨度提供了一个高分辨率的边界表示空间。在实体分类阶段，通过最小化干扰信息和促进关联泛化来净化实体与上下文之间的相关性，以缓解文本对抗性攻击造成的扰动。此外，我们还在公开的数据集Low-Nerd和Cross-DataSet上构建了对抗性的例子。对这两组包含对抗性实例的少镜头NER数据集的综合评价表明了该方法的稳健性和优越性。



## **17. DifAttack: Query-Efficient Black-Box Attack via Disentangled Feature Space**

DifAttack：基于解缠特征空间的查询高效黑盒攻击 cs.CV

Accepted in AAAI'24

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2309.14585v3) [paper-pdf](http://arxiv.org/pdf/2309.14585v3)

**Authors**: Liu Jun, Zhou Jiantao, Zeng Jiandian, Jinyu Tian

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (ASR) and good generalizability. We design a novel attack method based on a Disentangled Feature space, called DifAttack, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack firstly disentangles an image's latent feature into an adversarial feature and a visual feature, where the former dominates the adversarial capability of an image, while the latter largely determines its visual appearance. We train an autoencoder for the disentanglement by using pairs of clean images and their Adversarial Examples (AEs) generated from available surrogate models via white-box attack methods. Eventually, DifAttack iteratively optimizes the adversarial feature according to the query feedback from the victim model until a successful AE is generated, while keeping the visual feature unaltered. In addition, due to the avoidance of using surrogate models' gradient information when optimizing AEs for black-box models, our proposed DifAttack inherently possesses better attack capability in the open-set scenario, where the training dataset of the victim model is unknown. Extensive experimental results demonstrate that our method achieves significant improvements in ASR and query efficiency simultaneously, especially in the targeted attack and open-set scenarios. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(ASR)和良好的泛化能力。我们设计了一种新的基于解缠特征空间的攻击方法DifAttack，它与现有的操作在整个特征空间上的攻击方法有很大的不同。具体地说，DifAttack首先将图像的潜在特征分解为对抗性特征和视觉特征，其中前者主导图像的对抗性能力，而后者在很大程度上决定了图像的视觉外观。我们通过白盒攻击方法，使用已有的代理模型生成的干净图像对和它们的对抗性实例(AE)来训练自动编码器来进行解缠。最后，DifAttack根据受害者模型的查询反馈迭代地优化对抗性特征，直到生成成功的AE，同时保持视觉特征不变。此外，由于在优化黑盒模型的AES时避免了使用代理模型的梯度信息，因此在受害者模型的训练数据集未知的开集场景下，我们提出的DifAttack具有更好的攻击能力。大量的实验结果表明，该方法在ASR和查询效率上都有显著的提高，特别是在目标攻击和开集场景下。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **18. PromptBench: A Unified Library for Evaluation of Large Language Models**

PromptBitch：大型语言模型评估的统一库 cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07910v1) [paper-pdf](http://arxiv.org/pdf/2312.07910v1)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.

摘要: 大型语言模型(LLM)的评估对于评估其性能和降低潜在的安全风险至关重要。在本文中，我们介绍了一个用于评估LLMS的统一库PromptBitch.它由几个易于研究人员使用和扩展的关键组件组成：即时构建、即时工程、数据集和模型加载、对抗性即时攻击、动态评估协议和分析工具。PromptBitch是一个开放的、通用的、灵活的研究代码库，可以在创建新的基准、部署下游应用程序和设计新的评估协议方面促进原创研究。该代码可在https://github.com/microsoft/promptbench上获得，并将继续受到支持。



## **19. Causality Analysis for Evaluating the Security of Large Language Models**

大型语言模型安全性评估的因果分析 cs.AI

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07876v1) [paper-pdf](http://arxiv.org/pdf/2312.07876v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) such as GPT and Llama2 are increasingly adopted in many safety-critical applications. Their security is thus essential. Even with considerable efforts spent on reinforcement learning from human feedback (RLHF), recent studies have shown that LLMs are still subject to attacks such as adversarial perturbation and Trojan attacks. Further research is thus needed to evaluate their security and/or understand the lack of it. In this work, we propose a framework for conducting light-weight causality-analysis of LLMs at the token, layer, and neuron level. We applied our framework to open-source LLMs such as Llama2 and Vicuna and had multiple interesting discoveries. Based on a layer-level causality analysis, we show that RLHF has the effect of overfitting a model to harmful prompts. It implies that such security can be easily overcome by `unusual' harmful prompts. As evidence, we propose an adversarial perturbation method that achieves 100\% attack success rate on the red-teaming tasks of the Trojan Detection Competition 2023. Furthermore, we show the existence of one mysterious neuron in both Llama2 and Vicuna that has an unreasonably high causal effect on the output. While we are uncertain on why such a neuron exists, we show that it is possible to conduct a ``Trojan'' attack targeting that particular neuron to completely cripple the LLM, i.e., we can generate transferable suffixes to prompts that frequently make the LLM produce meaningless responses.

摘要: 大型语言模型(LLM)，如GPT和Llama2，在许多安全关键型应用中越来越多地被采用。因此，他们的安全至关重要。即使在从人类反馈中强化学习(RLHF)方面花费了大量的努力，最近的研究表明LLMS仍然受到诸如对抗性扰动和特洛伊木马攻击的攻击。因此，需要进一步研究，以评估其安全性和/或了解其缺乏安全性。在这项工作中，我们提出了一个框架，用于在标记、层和神经元水平上进行LLMS的轻量级因果分析。我们将我们的框架应用于开源LLM，如Llama2和Vicuna，并有多个有趣的发现。基于层级因果关系分析，我们发现RLHF具有对有害提示的模型过度拟合的效果。这意味着这种安全很容易被“不寻常的”有害提示所克服。作为证据，我们提出了一种对抗性扰动方法，在2023年木马检测大赛的红队任务上达到了100%的攻击成功率。此外，我们证明了在Llama2和Vicuna2中都存在一个神秘的神经元，它对输出具有不合理的高因果效应。虽然我们不确定为什么会有这样的神经元存在，但我们证明了有可能进行针对该特定神经元的“特洛伊木马”攻击，以完全削弱LLM，即我们可以为提示生成可转移的后缀，这些后缀经常使LLM产生无意义的响应。



## **20. Securing Graph Neural Networks in MLaaS: A Comprehensive Realization of Query-based Integrity Verification**

图神经网络在MLaaS中的安全：基于查询完整性验证的综合实现 cs.CR

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07870v1) [paper-pdf](http://arxiv.org/pdf/2312.07870v1)

**Authors**: Bang Wu, Xingliang Yuan, Shuo Wang, Qi Li, Minhui Xue, Shirui Pan

**Abstract**: The deployment of Graph Neural Networks (GNNs) within Machine Learning as a Service (MLaaS) has opened up new attack surfaces and an escalation in security concerns regarding model-centric attacks. These attacks can directly manipulate the GNN model parameters during serving, causing incorrect predictions and posing substantial threats to essential GNN applications. Traditional integrity verification methods falter in this context due to the limitations imposed by MLaaS and the distinct characteristics of GNN models.   In this research, we introduce a groundbreaking approach to protect GNN models in MLaaS from model-centric attacks. Our approach includes a comprehensive verification schema for GNN's integrity, taking into account both transductive and inductive GNNs, and accommodating varying pre-deployment knowledge of the models. We propose a query-based verification technique, fortified with innovative node fingerprint generation algorithms. To deal with advanced attackers who know our mechanisms in advance, we introduce randomized fingerprint nodes within our design. The experimental evaluation demonstrates that our method can detect five representative adversarial model-centric attacks, displaying 2 to 4 times greater efficiency compared to baselines.

摘要: 图神经网络(GNN)在机器学习即服务(MLaaS)中的部署开辟了新的攻击面，并加剧了对以模型为中心的攻击的安全担忧。这些攻击可以在服务期间直接操纵GNN模型参数，导致错误的预测，并对必要的GNN应用构成实质性威胁。在这种情况下，由于MLaaS的限制和GNN模型的独特特性，传统的完整性验证方法步履蹒跚。在这项研究中，我们介绍了一种突破性的方法来保护MLaaS中的GNN模型免受以模型为中心的攻击。我们的方法包括对GNN完整性的全面验证方案，同时考虑了传导性和感应性GNN，并容纳了不同的模型部署前知识。我们提出了一种基于查询的验证技术，并采用了创新的节点指纹生成算法。为了应对提前知道我们的机制的高级攻击者，我们在设计中引入了随机指纹节点。实验评估表明，我们的方法可以检测到五种典型的以模型为中心的对抗性攻击，与基线相比，效率提高了2到4倍。



## **21. SimAC: A Simple Anti-Customization Method against Text-to-Image Synthesis of Diffusion Models**

SIMAC：一种针对扩散模型图文合成的简单反定制方法 cs.CV

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07865v1) [paper-pdf](http://arxiv.org/pdf/2312.07865v1)

**Authors**: Feifei Wang, Zhentao Tan, Tianyi Wei, Yue Wu, Qidong Huang

**Abstract**: Despite the success of diffusion-based customization methods on visual content creation, increasing concerns have been raised about such techniques from both privacy and political perspectives. To tackle this issue, several anti-customization methods have been proposed in very recent months, predominantly grounded in adversarial attacks. Unfortunately, most of these methods adopt straightforward designs, such as end-to-end optimization with a focus on adversarially maximizing the original training loss, thereby neglecting nuanced internal properties intrinsic to the diffusion model, and even leading to ineffective optimization in some diffusion time steps. In this paper, we strive to bridge this gap by undertaking a comprehensive exploration of these inherent properties, to boost the performance of current anti-customization approaches. Two aspects of properties are investigated: 1) We examine the relationship between time step selection and the model's perception in the frequency domain of images and find that lower time steps can give much more contributions to adversarial noises. This inspires us to propose an adaptive greedy search for optimal time steps that seamlessly integrates with existing anti-customization methods. 2) We scrutinize the roles of features at different layers during denoising and devise a sophisticated feature-based optimization framework for anti-customization. Experiments on facial benchmarks demonstrate that our approach significantly increases identity disruption, thereby enhancing user privacy and security.

摘要: 尽管基于扩散的定制方法在视觉内容创建上取得了成功，但从隐私和政治角度来看，对这种技术的关注越来越多。为了解决这个问题，最近几个月提出了几种反定制方法，主要基于对抗性攻击。不幸的是，大多数这些方法都采用简单的设计，例如端到端优化，重点是对抗性地最大化原始训练损失，从而忽略了扩散模型固有的细微内部属性，甚至导致在某些扩散时间步长中的无效优化。在本文中，我们努力弥合这一差距，进行全面的探索，这些固有的属性，以提高目前的反定制方法的性能。研究了两个方面的性质：1）研究了时间步长的选择与模型在图像频域感知的关系，发现较低的时间步长对对抗性噪声的贡献更大。这启发我们提出一个自适应贪婪搜索的最佳时间步长，无缝集成与现有的反定制方法。2)我们仔细研究了在去噪过程中不同层的特征的作用，并设计了一个复杂的基于特征的反定制优化框架。面部基准测试的实验表明，我们的方法显着增加身份破坏，从而提高用户的隐私和安全性。



## **22. Radio Signal Classification by Adversarially Robust Quantum Machine Learning**

逆稳健量子机器学习在无线电信号分类中的应用 quant-ph

12 pages, 6 figures

**SubmitDate**: 2023-12-13    [abs](http://arxiv.org/abs/2312.07821v1) [paper-pdf](http://arxiv.org/pdf/2312.07821v1)

**Authors**: Yanqiu Wu, Eromanga Adermann, Chandra Thapa, Seyit Camtepe, Hajime Suzuki, Muhammad Usman

**Abstract**: Radio signal classification plays a pivotal role in identifying the modulation scheme used in received radio signals, which is essential for demodulation and proper interpretation of the transmitted information. Researchers have underscored the high susceptibility of ML algorithms for radio signal classification to adversarial attacks. Such vulnerability could result in severe consequences, including misinterpretation of critical messages, interception of classified information, or disruption of communication channels. Recent advancements in quantum computing have revolutionized theories and implementations of computation, bringing the unprecedented development of Quantum Machine Learning (QML). It is shown that quantum variational classifiers (QVCs) provide notably enhanced robustness against classical adversarial attacks in image classification. However, no research has yet explored whether QML can similarly mitigate adversarial threats in the context of radio signal classification. This work applies QVCs to radio signal classification and studies their robustness to various adversarial attacks. We also propose the novel application of the approximate amplitude encoding (AAE) technique to encode radio signal data efficiently. Our extensive simulation results present that attacks generated on QVCs transfer well to CNN models, indicating that these adversarial examples can fool neural networks that they are not explicitly designed to attack. However, the converse is not true. QVCs primarily resist the attacks generated on CNNs. Overall, with comprehensive simulations, our results shed new light on the growing field of QML by bridging knowledge gaps in QAML in radio signal classification and uncovering the advantages of applying QML methods in practical applications.

摘要: 无线电信号分类在识别接收无线电信号中使用的调制方案方面起着关键作用，这对于解调和正确解释传输的信息是必不可少的。研究人员强调了用于无线电信号分类的ML算法对敌方攻击的高度敏感性。此类漏洞可能导致严重后果，包括误解关键消息、截取机密信息或中断通信渠道。量子计算的最新进展使计算的理论和实现发生了革命性的变化，带来了量子机器学习(QML)前所未有的发展。结果表明，量子变分分类器(QVC)在图像分类中对经典的敌意攻击具有显著的鲁棒性。然而，还没有研究探索QML是否可以类似地在无线电信号分类的背景下减轻对抗性威胁。本文将QVC应用于无线电信号分类，研究了QVC对各种攻击的稳健性。我们还提出了近似幅度编码(AAE)技术在无线电信号数据编码中的新应用。我们广泛的模拟结果表明，对QVC产生的攻击很好地转移到了CNN模型中，表明这些敌对的例子可以欺骗不是明确设计来攻击的神经网络。然而，相反的情况并非如此。QVC主要抵抗CNN上产生的攻击。总体而言，通过全面的仿真，我们的结果通过弥合无线电信号分类中QAML的知识差距，揭示了在实际应用中应用QML方法的优势，从而为QML不断发展的领域提供了新的线索。



## **23. BarraCUDA: Bringing Electromagnetic Side Channel Into Play to Steal the Weights of Neural Networks from NVIDIA GPUs**

梭鱼：发挥电磁侧通道的作用，从NVIDIA图形处理器中窃取神经网络的权重 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07783v1) [paper-pdf](http://arxiv.org/pdf/2312.07783v1)

**Authors**: Peter Horvath, Lukasz Chmielewski, Leo Weissbart, Lejla Batina, Yuval Yarom

**Abstract**: Over the last decade, applications of neural networks have spread to cover all aspects of life. A large number of companies base their businesses on building products that use neural networks for tasks such as face recognition, machine translation, and autonomous cars. They are being used in safety and security-critical applications like high definition maps and medical wristbands, or in globally used products like Google Translate and ChatGPT. Much of the intellectual property underpinning these products is encoded in the exact configuration of the neural networks. Consequently, protecting these is of utmost priority to businesses. At the same time, many of these products need to operate under a strong threat model, in which the adversary has unfettered physical control of the product.   Past work has demonstrated that with physical access, attackers can reverse engineer neural networks that run on scalar microcontrollers, like ARM Cortex M3. However, for performance reasons, neural networks are often implemented on highly-parallel general purpose graphics processing units (GPGPUs), and so far, attacks on these have only recovered course-grained information on the structure of the neural network, but failed to retrieve the weights and biases.   In this work, we present BarraCUDA, a novel attack on GPGPUs that can completely extract the parameters of neural networks. BarraCUDA uses correlation electromagnetic analysis to recover the weights and biases in the convolutional layers of neural networks. We use BarraCUDA to attack the popular NVIDIA Jetson Nano device, demonstrating successful parameter extraction of neural networks in a highly parallel and noisy environment.

摘要: 在过去的十年里，神经网络的应用已经扩展到生活的方方面面。许多公司的业务基础是开发使用神经网络执行人脸识别、机器翻译和自动驾驶汽车等任务的产品。它们正被用于高清晰度地图和医疗腕带等安全和安保关键应用程序，或谷歌翻译和ChatGPT等全球使用的产品。支撑这些产品的大部分知识产权都编码在神经网络的准确配置中。因此，保护这些信息对企业来说是最重要的。同时，这些产品中的许多都需要在强大的威胁模式下运行，在这种模式下，对手可以不受约束地对产品进行物理控制。过去的研究表明，通过物理访问，攻击者可以对在ARM Cortex M3等标量微控制器上运行的神经网络进行反向工程。然而，由于性能原因，神经网络通常是在高度并行的通用图形处理单元(GPGPU)上实现的，到目前为止，对这些单元的攻击只恢复了关于神经网络结构的过程粒度信息，但无法恢复权重和偏差。在这项工作中，我们提出了一种新的针对GPGPU的攻击Barracuda，它可以完全提取神经网络的参数。梭鱼使用相关电磁分析来恢复神经网络卷积层中的权重和偏差。我们使用梭鱼攻击流行的NVIDIA Jetson Nano设备，演示了在高度并行和噪声环境中成功提取神经网络的参数。



## **24. Majority is Not Required: A Rational Analysis of the Private Double-Spend Attack from a Sub-Majority Adversary**

不需要多数：对一个次多数对手的私人双重支出攻击的理性分析 cs.GT

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07709v1) [paper-pdf](http://arxiv.org/pdf/2312.07709v1)

**Authors**: Yanni Georghiades, Rajesh Mishra, Karl Kreder, Sriram Vishwanath

**Abstract**: We study the incentives behind double-spend attacks on Nakamoto-style Proof-of-Work cryptocurrencies. In these systems, miners are allowed to choose which transactions to reference with their block, and a common strategy for selecting transactions is to simply choose those with the highest fees. This can be problematic if these transactions originate from an adversary with substantial (but less than 50\%) computational power, as high-value transactions can present an incentive for a rational adversary to attempt a double-spend attack if they expect to profit. The most common mechanism for deterring double-spend attacks is for the recipients of large transactions to wait for additional block confirmations (i.e., to increase the attack cost). We argue that this defense mechanism is not satisfactory, as the security of the system is contingent on the actions of its users. Instead, we propose that defending against double-spend attacks should be the responsibility of the miners; specifically, miners should limit the amount of transaction value they include in a block (i.e., reduce the attack reward). To this end, we model cryptocurrency mining as a mean-field game in which we augment the standard mining reward function to simulate the presence of a rational, double-spending adversary. We design and implement an algorithm which characterizes the behavior of miners at equilibrium, and we show that miners who use the adversary-aware reward function accumulate more wealth than those who do not. We show that the optimal strategy for honest miners is to limit the amount of value transferred by each block such that the adversary's expected profit is 0. Additionally, we examine Bitcoin's resilience to double-spend attacks. Assuming a 6 block confirmation time, we find that an attacker with at least 25% of the network mining power can expect to profit from a double-spend attack.

摘要: 我们研究了对Nakamoto风格的工作量证明加密货币进行双重花费攻击背后的动机。在这些系统中，矿工可以选择哪些交易与他们的区块相关联，选择交易的常见策略是简单地选择那些费用最高的交易。如果这些交易来自具有大量（但小于50%）计算能力的对手，这可能是有问题的，因为高价值交易可能会激励理性的对手尝试双重花费攻击，如果他们期望获利的话。阻止双重花费攻击的最常见机制是让大型交易的接收者等待额外的块确认（即，增加攻击成本）。我们认为，这种防御机制是不令人满意的，因为系统的安全性是视用户的行为。相反，我们建议防御双重花费攻击应该是矿工的责任;具体来说，矿工应该限制他们在区块中包含的交易价值的数量（即，减少攻击奖励）。为此，我们将加密货币挖掘建模为平均场游戏，在该游戏中，我们增加了标准的挖掘奖励函数，以模拟理性的双重支出对手的存在。我们设计并实现了一个算法，该算法描述了矿工在平衡状态下的行为，并且我们表明，使用对手意识奖励函数的矿工比不使用的矿工积累了更多的财富。我们表明，诚实矿工的最佳策略是限制每个区块转移的价值量，使对手的预期利润为0。此外，我们还研究了比特币对双重支出攻击的弹性。假设一个6块的确认时间，我们发现一个攻击者至少有25%的网络挖掘能力可以期望从双重花费攻击中获利。



## **25. Defending Our Privacy With Backdoors**

使用后门保护我们的隐私 cs.LG

14 pages, 10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2310.08320v2) [paper-pdf](http://arxiv.org/pdf/2310.08320v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Daniel Neider, Kristian Kersting

**Abstract**: The proliferation of large AI models trained on uncurated, often sensitive web-scraped data has raised significant privacy concerns. One of the concerns is that adversaries can extract information about the training data using privacy attacks. Unfortunately, the task of removing specific information from the models without sacrificing performance is not straightforward and has proven to be challenging. We propose a rather easy yet effective defense based on backdoor attacks to remove private information such as names of individuals from models, and focus in this work on text encoders. Specifically, through strategic insertion of backdoors, we align the embeddings of sensitive phrases with those of neutral terms-"a person" instead of the person's name. Our empirical results demonstrate the effectiveness of our backdoor-based defense on CLIP by assessing its performance using a specialized privacy attack for zero-shot classifiers. Our approach provides not only a new "dual-use" perspective on backdoor attacks, but also presents a promising avenue to enhance the privacy of individuals within models trained on uncurated web-scraped data.

摘要: 大型人工智能模型的激增引发了人们对隐私的严重担忧。这些模型针对未经管理的、往往是敏感的网络数据进行培训。其中一个令人担忧的问题是，攻击者可以使用隐私攻击来提取有关训练数据的信息。不幸的是，在不牺牲性能的情况下从模型中删除特定信息的任务并不简单，而且已被证明是具有挑战性的。我们提出了一种基于后门攻击的简单而有效的防御方法，将个人姓名等私人信息从模型中移除，并将重点放在文本编码器上。具体地说，通过策略性地插入后门，我们将敏感短语的嵌入与中性术语--“人”而不是人的名字--保持一致。我们的实验结果证明了我们的基于后门的防御在CLIP上的有效性，通过使用专门的针对零镜头分类器的隐私攻击来评估其性能。我们的方法不仅为后门攻击提供了一种新的“两用”视角，而且还提供了一种在未经管理的网络抓取数据的培训模型中增强个人隐私的有前景的途径。



## **26. DeceptPrompt: Exploiting LLM-driven Code Generation via Adversarial Natural Language Instructions**

DeceptPrompt：通过对抗性自然语言指令利用LLM驱动的代码生成 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.04730v2) [paper-pdf](http://arxiv.org/pdf/2312.04730v2)

**Authors**: Fangzhou Wu, Xiaogeng Liu, Chaowei Xiao

**Abstract**: With the advancement of Large Language Models (LLMs), significant progress has been made in code generation, enabling LLMs to transform natural language into programming code. These Code LLMs have been widely accepted by massive users and organizations. However, a dangerous nature is hidden in the code, which is the existence of fatal vulnerabilities. While some LLM providers have attempted to address these issues by aligning with human guidance, these efforts fall short of making Code LLMs practical and robust. Without a deep understanding of the performance of the LLMs under the practical worst cases, it would be concerning to apply them to various real-world applications. In this paper, we answer the critical issue: Are existing Code LLMs immune to generating vulnerable code? If not, what is the possible maximum severity of this issue in practical deployment scenarios? In this paper, we introduce DeceptPrompt, a novel algorithm that can generate adversarial natural language instructions that drive the Code LLMs to generate functionality correct code with vulnerabilities. DeceptPrompt is achieved through a systematic evolution-based algorithm with a fine grain loss design. The unique advantage of DeceptPrompt enables us to find natural prefix/suffix with totally benign and non-directional semantic meaning, meanwhile, having great power in inducing the Code LLMs to generate vulnerable code. This feature can enable us to conduct the almost-worstcase red-teaming on these LLMs in a real scenario, where users are using natural language. Our extensive experiments and analyses on DeceptPrompt not only validate the effectiveness of our approach but also shed light on the huge weakness of LLMs in the code generation task. When applying the optimized prefix/suffix, the attack success rate (ASR) will improve by average 50% compared with no prefix/suffix applying.

摘要: 随着大型语言模型(LLMS)的发展，在代码生成方面取得了重大进展，使LLMS能够将自然语言转换为编程代码。这些CodeLLM已被广大用户和组织广泛接受。然而，代码中隐藏着一个危险的性质，那就是存在致命的漏洞。虽然一些LLM提供商试图通过与人类的指导保持一致来解决这些问题，但这些努力并不能使Code LLM实用和健壮。如果不深入了解LLMS在实际最坏情况下的性能，将它们应用于各种现实世界应用将是令人担忧的。在这篇文章中，我们回答了一个关键问题：现有的代码LLM是否不会生成易受攻击的代码？如果不是，此问题在实际部署方案中可能的最大严重程度是多少？在本文中，我们介绍了DeceptPrompt算法，它可以生成敌意的自然语言指令，这些指令驱动Code LLMS生成有漏洞的功能正确的代码。DeceptPrompt是通过基于系统进化的算法实现的，具有细粒度的损耗设计。DeceptPrompt的独特优势使我们能够找到具有完全良性和非方向性语义的自然前缀/后缀，同时对诱使Code LLMS生成易受攻击的代码具有强大的能力。这一功能使我们能够在用户使用自然语言的真实场景中对这些LLM进行几乎最糟糕的红色团队。我们在DeceptPrompt上的大量实验和分析不仅验证了我们方法的有效性，而且揭示了LLMS在代码生成任务中的巨大弱点。当应用优化的前缀/后缀时，与不应用前缀/后缀相比，攻击成功率(ASR)将平均提高50%。



## **27. ReRoGCRL: Representation-based Robustness in Goal-Conditioned Reinforcement Learning**

ReRoGCRL：目标条件强化学习中基于表示的稳健性 cs.LG

This paper has been accepted in AAAI24  (https://aaai.org/aaai-conference/)

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07392v1) [paper-pdf](http://arxiv.org/pdf/2312.07392v1)

**Authors**: Xiangyu Yin, Sihao Wu, Jiaxu Liu, Meng Fang, Xingyu Zhao, Xiaowei Huang, Wenjie Ruan

**Abstract**: While Goal-Conditioned Reinforcement Learning (GCRL) has gained attention, its algorithmic robustness, particularly against adversarial perturbations, remains unexplored. Unfortunately, the attacks and robust representation training methods specifically designed for traditional RL are not so effective when applied to GCRL. To address this challenge, we propose the \textit{Semi-Contrastive Representation} attack, a novel approach inspired by the adversarial contrastive attack. Unlike existing attacks in RL, it only necessitates information from the policy function and can be seamlessly implemented during deployment. Furthermore, to mitigate the vulnerability of existing GCRL algorithms, we introduce \textit{Adversarial Representation Tactics}. This strategy combines \textit{Semi-Contrastive Adversarial Augmentation} with \textit{Sensitivity-Aware Regularizer}. It improves the adversarial robustness of the underlying agent against various types of perturbations. Extensive experiments validate the superior performance of our attack and defence mechanism across multiple state-of-the-art GCRL algorithms. Our tool {\bf ReRoGCRL} is available at \url{https://github.com/TrustAI/ReRoGCRL}.

摘要: 虽然目标条件强化学习(GCRL)已经引起了人们的关注，但它的算法健壮性，特别是对对抗性扰动的鲁棒性，仍然没有得到探索。不幸的是，专门针对传统RL设计的攻击和稳健表示训练方法在应用于GCRL时并不是很有效。为了应对这一挑战，我们提出了半对比表示攻击，这是一种受对抗性对比攻击启发的新方法。与RL中现有的攻击不同，它只需要来自策略功能的信息，并且可以在部署期间无缝实施。此外，为了缓解现有GCRL算法的脆弱性，我们引入了对抗性表示策略。该策略结合了半对比式对抗性增强和敏感度感知调节器。它提高了底层代理对各种类型扰动的对抗健壮性。广泛的实验验证了我们的攻击和防御机制在多种最先进的GCRL算法上的卓越性能。我们的工具{\bf ReRoGCRL}位于\url{https://github.com/TrustAI/ReRoGCRL}.



## **28. Eroding Trust In Aerial Imagery: Comprehensive Analysis and Evaluation Of Adversarial Attacks In Geospatial Systems**

航空图像中信任的侵蚀：地理空间系统中对抗性攻击的综合分析和评估 cs.CV

Accepted at IEEE AIRP 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07389v1) [paper-pdf](http://arxiv.org/pdf/2312.07389v1)

**Authors**: Michael Lanier, Aayush Dhakal, Zhexiao Xiong, Arthur Li, Nathan Jacobs, Yevgeniy Vorobeychik

**Abstract**: In critical operations where aerial imagery plays an essential role, the integrity and trustworthiness of data are paramount. The emergence of adversarial attacks, particularly those that exploit control over labels or employ physically feasible trojans, threatens to erode that trust, making the analysis and mitigation of these attacks a matter of urgency. We demonstrate how adversarial attacks can degrade confidence in geospatial systems, specifically focusing on scenarios where the attacker's control over labels is restricted and the use of realistic threat vectors. Proposing and evaluating several innovative attack methodologies, including those tailored to overhead images, we empirically show their threat to remote sensing systems using high-quality SpaceNet datasets. Our experimentation reflects the unique challenges posed by aerial imagery, and these preliminary results not only reveal the potential risks but also highlight the non-trivial nature of the problem compared to recent works.

摘要: 在航空图像发挥重要作用的关键行动中，数据的完整性和可信度至关重要。对抗性攻击的出现，特别是那些利用对标签的控制或使用物理上可行的特洛伊木马的攻击，可能会削弱这种信任，使分析和缓解这些攻击成为当务之急。我们展示了对抗性攻击如何降低地理空间系统的信心，特别是专注于攻击者对标签的控制受到限制的场景以及现实威胁向量的使用。提出并评估了几种创新的攻击方法，包括针对头顶图像的攻击方法，我们使用高质量的SpaceNet数据集以经验的方式展示了它们对遥感系统的威胁。我们的实验反映了航拍图像所带来的独特挑战，这些初步结果不仅揭示了潜在的风险，而且与最近的工作相比，突出了问题的重要性。



## **29. SSTA: Salient Spatially Transformed Attack**

SSTA：显著的空间变换攻击 cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07258v1) [paper-pdf](http://arxiv.org/pdf/2312.07258v1)

**Authors**: Renyang Liu, Wei Zhou, Sixin Wu, Jun Zhao, Kwok-Yan Lam

**Abstract**: Extensive studies have demonstrated that deep neural networks (DNNs) are vulnerable to adversarial attacks, which brings a huge security risk to the further application of DNNs, especially for the AI models developed in the real world. Despite the significant progress that has been made recently, existing attack methods still suffer from the unsatisfactory performance of escaping from being detected by naked human eyes due to the formulation of adversarial example (AE) heavily relying on a noise-adding manner. Such mentioned challenges will significantly increase the risk of exposure and result in an attack to be failed. Therefore, in this paper, we propose the Salient Spatially Transformed Attack (SSTA), a novel framework to craft imperceptible AEs, which enhance the stealthiness of AEs by estimating a smooth spatial transform metric on a most critical area to generate AEs instead of adding external noise to the whole image. Compared to state-of-the-art baselines, extensive experiments indicated that SSTA could effectively improve the imperceptibility of the AEs while maintaining a 100\% attack success rate.

摘要: 大量的研究表明，深度神经网络(DNN)容易受到敌意攻击，这给DNN的进一步应用带来了巨大的安全风险，特别是对于现实世界中开发的人工智能模型。尽管最近取得了很大的进展，但现有的攻击方法由于严重依赖于噪声添加的方式而形成的对抗性范例(AE)，仍然存在逃脱肉眼检测的不尽人意的表现。上述挑战将极大地增加暴露的风险，并导致攻击失败。因此，在本文中，我们提出了一种新的隐身攻击框架--突显空间变换攻击(SSTA)，它通过在最关键的区域估计一个平滑的空间变换度量来生成隐蔽攻击，而不是在整个图像中添加外部噪声，从而增强了隐身攻击的隐蔽性。大量的实验表明，与最新的基线相比，SSTA在保持100%攻击成功率的同时，可以有效地提高AEs的隐蔽性。



## **30. DTA: Distribution Transform-based Attack for Query-Limited Scenario**

DTA：查询受限场景下基于分布变换的攻击 cs.CV

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07245v1) [paper-pdf](http://arxiv.org/pdf/2312.07245v1)

**Authors**: Renyang Liu, Wei Zhou, Xin Jin, Song Gao, Yuanyu Wang, Ruxin Wang

**Abstract**: In generating adversarial examples, the conventional black-box attack methods rely on sufficient feedback from the to-be-attacked models by repeatedly querying until the attack is successful, which usually results in thousands of trials during an attack. This may be unacceptable in real applications since Machine Learning as a Service Platform (MLaaS) usually only returns the final result (i.e., hard-label) to the client and a system equipped with certain defense mechanisms could easily detect malicious queries. By contrast, a feasible way is a hard-label attack that simulates an attacked action being permitted to conduct a limited number of queries. To implement this idea, in this paper, we bypass the dependency on the to-be-attacked model and benefit from the characteristics of the distributions of adversarial examples to reformulate the attack problem in a distribution transform manner and propose a distribution transform-based attack (DTA). DTA builds a statistical mapping from the benign example to its adversarial counterparts by tackling the conditional likelihood under the hard-label black-box settings. In this way, it is no longer necessary to query the target model frequently. A well-trained DTA model can directly and efficiently generate a batch of adversarial examples for a certain input, which can be used to attack un-seen models based on the assumed transferability. Furthermore, we surprisingly find that the well-trained DTA model is not sensitive to the semantic spaces of the training dataset, meaning that the model yields acceptable attack performance on other datasets. Extensive experiments validate the effectiveness of the proposed idea and the superiority of DTA over the state-of-the-art.

摘要: 在生成对抗性实例时，传统的黑盒攻击方法依赖于被攻击模型的充分反馈，通过反复查询直到攻击成功，这通常导致在一次攻击中进行数千次尝试。这在实际应用中可能是不可接受的，因为机器学习作为服务平台(MLaaS)通常只向客户端返回最终结果(即硬标签)，并且配备了某些防御机制的系统可以很容易地检测到恶意查询。相比之下，一种可行的方法是硬标签攻击，它模拟允许执行有限数量的查询的攻击操作。为了实现这一思想，本文绕过了对待攻击模型的依赖，利用对抗性实例分布的特点，用分布变换的方式重新描述攻击问题，提出了一种基于分布变换的攻击(DTA)。DTA通过处理硬标签黑盒设置下的条件似然，建立了从良性例子到对抗性例子的统计映射。这样，就不再需要频繁地查询目标模型。一个训练有素的DTA模型可以直接有效地为某一输入生成一批对抗性的例子，这些例子可以用来攻击基于假设的可转移性的不可见模型。此外，我们惊讶地发现，经过良好训练的DTA模型对训练数据集的语义空间不敏感，这意味着该模型在其他数据集上的攻击性能是可以接受的。大量的实验验证了所提出的思想的有效性以及DTA相对于最先进技术的优越性。



## **31. Reward Certification for Policy Smoothed Reinforcement Learning**

策略平滑强化学习的奖励认证 cs.LG

This paper will be presented in AAAI2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06436v2) [paper-pdf](http://arxiv.org/pdf/2312.06436v2)

**Authors**: Ronghui Mu, Leandro Soriano Marcolino, Tianle Zhang, Yanghao Zhang, Xiaowei Huang, Wenjie Ruan

**Abstract**: Reinforcement Learning (RL) has achieved remarkable success in safety-critical areas, but it can be weakened by adversarial attacks. Recent studies have introduced "smoothed policies" in order to enhance its robustness. Yet, it is still challenging to establish a provable guarantee to certify the bound of its total reward. Prior methods relied primarily on computing bounds using Lipschitz continuity or calculating the probability of cumulative reward above specific thresholds. However, these techniques are only suited for continuous perturbations on the RL agent's observations and are restricted to perturbations bounded by the $l_2$-norm. To address these limitations, this paper proposes a general black-box certification method capable of directly certifying the cumulative reward of the smoothed policy under various $l_p$-norm bounded perturbations. Furthermore, we extend our methodology to certify perturbations on action spaces. Our approach leverages f-divergence to measure the distinction between the original distribution and the perturbed distribution, subsequently determining the certification bound by solving a convex optimisation problem. We provide a comprehensive theoretical analysis and run sufficient experiments in multiple environments. Our results show that our method not only improves the certified lower bound of mean cumulative reward but also demonstrates better efficiency than state-of-the-art techniques.

摘要: 强化学习(RL)在安全关键领域取得了显著的成功，但它可能会被对手攻击所削弱。最近的研究引入了“平滑政策”，以增强其稳健性。然而，建立一个可证明的保证来证明其总回报的界限仍然是具有挑战性的。以前的方法主要依赖于使用Lipschitz连续性来计算界限，或者计算超过特定阈值的累积奖励的概率。然而，这些技术只适用于对RL代理观测的连续扰动，并且限于$L_2$-范数的扰动。针对这些局限性，本文提出了一种通用的黑盒证明方法，该方法能够直接证明平滑策略在各种$L_p$-范数有界扰动下的累积报酬。此外，我们将我们的方法扩展到证明行动空间上的扰动。我们的方法利用f-散度来度量原始分布和扰动分布之间的区别，然后通过求解一个凸优化问题来确定认证界。我们提供了全面的理论分析，并在多个环境下进行了大量的实验。我们的结果表明，我们的方法不仅改善了平均累积奖励的证明下界，而且比最新的技术表现出更好的效率。



## **32. Adversarial Driving: Attacking End-to-End Autonomous Driving**

对抗性驾驶：攻击型端到端自动驾驶 cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2103.09151v8) [paper-pdf](http://arxiv.org/pdf/2103.09151v8)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.

摘要: 随着深度神经网络研究的深入，深度卷积网络在自动驾驶任务中变得很有前途。特别是，使用端到端神经网络模型进行自动驾驶是一种新兴的趋势。然而，以往的研究表明，深度神经网络分类器容易受到敌意攻击。而对于回归任务，对抗性攻击的效果并没有被很好地理解。在本研究中，我们设计了两种针对端到端自动驾驶模型的白盒针对性攻击。我们的攻击通过干扰输入图像来操纵自动驾驶系统的行为。在相同攻击强度(epsilon=1)的平均800次攻击中，图像特定攻击和图像无关攻击使转向角与原始输出的偏差分别为0.478和0.111，远远强于仅扰动转向角0.002的随机噪声(转向角范围为[-1，1])。这两种攻击都可以在不使用GPU的情况下在CPU上实时发起。演示视频：https://youtu.be/I0i8uN2oOP0.



## **33. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2209.01962v6) [paper-pdf](http://arxiv.org/pdf/2209.01962v6)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.

摘要: 智能机器人依靠物体检测模型来感知环境。随着深度学习安全性的进步，人们发现目标检测模型容易受到敌意攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。因此，目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。本文通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。这些攻击在大约20次迭代内实现了约90%的成功率。该演示视频可在https://youtu.be/zJZ1aNlXsMU.上查看



## **34. Cost Aware Untargeted Poisoning Attack against Graph Neural Networks,**

针对图神经网络的成本意识非目标中毒攻击， cs.AI

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07158v1) [paper-pdf](http://arxiv.org/pdf/2312.07158v1)

**Authors**: Yuwei Han, Yuni Lai, Yulin Zhu, Kai Zhou

**Abstract**: Graph Neural Networks (GNNs) have become widely used in the field of graph mining. However, these networks are vulnerable to structural perturbations. While many research efforts have focused on analyzing vulnerability through poisoning attacks, we have identified an inefficiency in current attack losses. These losses steer the attack strategy towards modifying edges targeting misclassified nodes or resilient nodes, resulting in a waste of structural adversarial perturbation. To address this issue, we propose a novel attack loss framework called the Cost Aware Poisoning Attack (CA-attack) to improve the allocation of the attack budget by dynamically considering the classification margins of nodes. Specifically, it prioritizes nodes with smaller positive margins while postponing nodes with negative margins. Our experiments demonstrate that the proposed CA-attack significantly enhances existing attack strategies

摘要: 图神经网络在图挖掘领域得到了广泛的应用。然而，这些网络很容易受到结构扰动的影响。虽然许多研究工作都集中在通过中毒攻击来分析脆弱性上，但我们已经发现了当前攻击损失的低效。这些损失使攻击策略倾向于修改针对错误分类节点或弹性节点的边，从而浪费了结构上的对抗性扰动。针对这一问题，我们提出了一种新的攻击损失框架，称为代价感知中毒攻击(CA-Attack)，通过动态考虑节点的分类裕度来提高攻击预算的分配。具体地说，它优先考虑正边距较小的节点，而推迟边距为负值的节点。实验表明，所提出的CA-攻击显著增强了现有的攻击策略



## **35. Data-Free Hard-Label Robustness Stealing Attack**

无数据硬标签健壮性窃取攻击 cs.CV

Accepted by AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.05924v2) [paper-pdf](http://arxiv.org/pdf/2312.05924v2)

**Authors**: Xiaojian Yuan, Kejiang Chen, Wen Huang, Jie Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: The popularity of Machine Learning as a Service (MLaaS) has led to increased concerns about Model Stealing Attacks (MSA), which aim to craft a clone model by querying MLaaS. Currently, most research on MSA assumes that MLaaS can provide soft labels and that the attacker has a proxy dataset with a similar distribution. However, this fails to encapsulate the more practical scenario where only hard labels are returned by MLaaS and the data distribution remains elusive. Furthermore, most existing work focuses solely on stealing the model accuracy, neglecting the model robustness, while robustness is essential in security-sensitive scenarios, e.g., face-scan payment. Notably, improving model robustness often necessitates the use of expensive techniques such as adversarial training, thereby further making stealing robustness a more lucrative prospect. In response to these identified gaps, we introduce a novel Data-Free Hard-Label Robustness Stealing (DFHL-RS) attack in this paper, which enables the stealing of both model accuracy and robustness by simply querying hard labels of the target model without the help of any natural data. Comprehensive experiments demonstrate the effectiveness of our method. The clone model achieves a clean accuracy of 77.86% and a robust accuracy of 39.51% against AutoAttack, which are only 4.71% and 8.40% lower than the target model on the CIFAR-10 dataset, significantly exceeding the baselines. Our code is available at: https://github.com/LetheSec/DFHL-RS-Attack.

摘要: 机器学习即服务(MLaaS)的流行引起了人们对模型窃取攻击(MSA)的越来越多的关注，MSA旨在通过查询MLaaS来创建克隆模型。目前，大多数关于MSA的研究都假设MLaaS可以提供软标签，并且攻击者拥有一个具有类似分布的代理数据集。然而，这无法封装更实际的场景，即MLaaS只返回硬标签，数据分布仍然难以捉摸。此外，现有的大多数工作只关注窃取模型的准确性，而忽略了模型的健壮性，而健壮性在安全敏感的场景中是必不可少的，例如人脸扫描支付。值得注意的是，提高模型的稳健性通常需要使用昂贵的技术，如对抗性训练，从而进一步使窃取稳健性成为更有利可图的前景。针对这些缺陷，本文提出了一种新的无数据硬标签健壮性窃取攻击(DFHL-RS)，该攻击通过简单地查询目标模型的硬标签来实现对模型精度和稳健性的窃取，而不需要任何自然数据。综合实验证明了该方法的有效性。克隆模型在AutoAttack上的清洁准确率为77.86%，健壮性准确率为39.51%，仅比目标模型在CIFAR-10数据集上低4.71%和8.40%，显著超过基线。我们的代码请访问：https://github.com/LetheSec/DFHL-RS-Attack.



## **36. Divide-and-Conquer Attack: Harnessing the Power of LLM to Bypass the Censorship of Text-to-Image Generation Model**

分治攻击：利用LLM的力量绕过文本到图像生成模型的审查 cs.AI

20 pages,6 figures, under review

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07130v1) [paper-pdf](http://arxiv.org/pdf/2312.07130v1)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: Text-to-image generative models offer many innovative services but also raise ethical concerns due to their potential to generate unethical images. Most publicly available text-to-image models employ safety filters to prevent unintended generation intents. In this work, we introduce the Divide-and-Conquer Attack to circumvent the safety filters of state-of-the-art text-to-image models. Our attack leverages LLMs as agents for text transformation, creating adversarial prompts from sensitive ones. We have developed effective helper prompts that enable LLMs to break down sensitive drawing prompts into multiple harmless descriptions, allowing them to bypass safety filters while still generating sensitive images. This means that the latent harmful meaning only becomes apparent when all individual elements are drawn together. Our evaluation demonstrates that our attack successfully circumvents the closed-box safety filter of SOTA DALLE-3 integrated natively into ChatGPT to generate unethical images. This approach, which essentially uses LLM-generated adversarial prompts against GPT-4-assisted DALLE-3, is akin to using one's own spear to breach their shield. It could have more severe security implications than previous manual crafting or iterative model querying methods, and we hope it stimulates more attention towards similar efforts. Our code and data are available at: https://github.com/researchcode001/Divide-and-Conquer-Attack

摘要: 文本到图像生成模型提供了许多创新服务，但也引起了道德问题，因为它们可能会生成不道德的图像。大多数公开可用的文本到图像模型都使用安全过滤器来防止无意的生成意图。在这项工作中，我们引入了分治攻击来规避最先进的文本到图像模型的安全过滤器。我们的攻击利用LLM作为文本转换的代理，从敏感的文本中创建对抗性提示。我们已经开发了有效的辅助提示，使LLM能够将敏感的绘图提示分解为多个无害的描述，使它们能够绕过安全过滤器，同时仍然生成敏感图像。这意味着，只有当所有单个元素被聚集在一起时，潜在的有害含义才会变得明显。我们的评估表明，我们的攻击成功地绕过了SOTA DALLE-3的封闭式安全过滤器，该过滤器原生集成到ChatGPT中，以生成不道德的图像。这种方法基本上使用LLM生成的对抗性提示来对抗GPT-4辅助的DALLE-3，类似于使用自己的矛来突破自己的盾。它可能比以前的手工制作或迭代模型查询方法具有更严重的安全性影响，我们希望它能激发更多的关注。我们的代码和数据可在https://github.com/researchcode001/Divide-and-Conquer-Attack上获得



## **37. Promoting Counterfactual Robustness through Diversity**

通过多样性促进反事实稳健性 cs.LG

Accepted at AAAI 2024

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06564v2) [paper-pdf](http://arxiv.org/pdf/2312.06564v2)

**Authors**: Francesco Leofante, Nico Potyka

**Abstract**: Counterfactual explanations shed light on the decisions of black-box models by explaining how an input can be altered to obtain a favourable decision from the model (e.g., when a loan application has been rejected). However, as noted recently, counterfactual explainers may lack robustness in the sense that a minor change in the input can cause a major change in the explanation. This can cause confusion on the user side and open the door for adversarial attacks. In this paper, we study some sources of non-robustness. While there are fundamental reasons for why an explainer that returns a single counterfactual cannot be robust in all instances, we show that some interesting robustness guarantees can be given by reporting multiple rather than a single counterfactual. Unfortunately, the number of counterfactuals that need to be reported for the theoretical guarantees to hold can be prohibitively large. We therefore propose an approximation algorithm that uses a diversity criterion to select a feasible number of most relevant explanations and study its robustness empirically. Our experiments indicate that our method improves the state-of-the-art in generating robust explanations, while maintaining other desirable properties and providing competitive computational performance.

摘要: 反事实的解释通过解释如何改变输入以从模型中获得有利的决定(例如，当贷款申请被拒绝时)，揭示了黑箱模型的决定。然而，正如最近指出的那样，反事实解释者可能缺乏稳健性，因为输入的微小变化可能会导致解释的重大变化。这可能会在用户端造成混乱，并为对抗性攻击打开大门。在本文中，我们研究了非稳健性的一些来源。虽然返回单个反事实的解释器不能在所有情况下都是健壮的是有根本原因的，但我们证明了一些有趣的健壮性保证可以通过报告多个而不是单个反事实来提供。不幸的是，需要报告的反事实数量可能会令人望而却步，才能让理论保证成立。因此，我们提出了一种近似算法，该算法使用多样性准则来选择最相关的可行数量的解释，并对其稳健性进行了实证研究。我们的实验表明，我们的方法在生成健壮的解释方面提高了最先进的水平，同时保持了其他所需的性质，并提供了具有竞争力的计算性能。



## **38. Patch-MI: Enhancing Model Inversion Attacks via Patch-Based Reconstruction**

Patch-MI：通过基于Patch的重构增强模型反转攻击 cs.AI

11 pages

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07040v1) [paper-pdf](http://arxiv.org/pdf/2312.07040v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Hyun Jong Yang

**Abstract**: Model inversion (MI) attacks aim to reveal sensitive information in training datasets by solely accessing model weights. Generative MI attacks, a prominent strand in this field, utilize auxiliary datasets to recreate target data attributes, restricting the images to remain photo-realistic, but their success often depends on the similarity between auxiliary and target datasets. If the distributions are dissimilar, existing MI attack attempts frequently fail, yielding unrealistic or target-unrelated results. In response to these challenges, we introduce a groundbreaking approach named Patch-MI, inspired by jigsaw puzzle assembly. To this end, we build upon a new probabilistic interpretation of MI attacks, employing a generative adversarial network (GAN)-like framework with a patch-based discriminator. This approach allows the synthesis of images that are similar to the target dataset distribution, even in cases of dissimilar auxiliary dataset distribution. Moreover, we artfully employ a random transformation block, a sophisticated maneuver that crafts generalized images, thus enhancing the efficacy of the target classifier. Our numerical and graphical findings demonstrate that Patch-MI surpasses existing generative MI methods in terms of accuracy, marking significant advancements while preserving comparable statistical dataset quality. For reproducibility of our results, we make our source code publicly available in https://github.com/jonggyujang0123/Patch-Attack.

摘要: 模型反转攻击的目的是通过仅访问模型权重来揭示训练数据集中的敏感信息。生成性MI攻击是该领域的一个重要分支，它利用辅助数据集来重建目标数据属性，限制图像保持照片真实感，但它们的成功往往取决于辅助数据集和目标数据集之间的相似性。如果分布不同，现有的MI攻击尝试经常失败，从而产生不切实际或与目标无关的结果。为了应对这些挑战，我们引入了一种名为Patch-MI的开创性方法，灵感来自拼图拼图组装。为此，我们建立了对MI攻击的一种新的概率解释，采用了一个基于补丁的鉴别器的生成性对手网络(GAN)类框架。该方法允许合成与目标数据集分布相似的图像，即使在不同辅助数据集分布的情况下也是如此。此外，我们巧妙地使用了随机变换块，这是一种复杂的策略，可以制作通用图像，从而提高了目标分类器的效率。我们的数值和图形结果表明，Patch-MI在准确性方面超过了现有的生成性MI方法，在保持可比较的统计数据集质量的同时标志着显著的进步。为了使结果重现，我们在https://github.com/jonggyujang0123/Patch-Attack.中公开了我们的源代码



## **39. EdgePruner: Poisoned Edge Pruning in Graph Contrastive Learning**

EdgePruner：图对比学习中的毒边剪枝 cs.CR

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.07022v1) [paper-pdf](http://arxiv.org/pdf/2312.07022v1)

**Authors**: Hiroya Kato, Kento Hasegawa, Seira Hidano, Kazuhide Fukushima

**Abstract**: Graph Contrastive Learning (GCL) is unsupervised graph representation learning that can obtain useful representation of unknown nodes. The node representation can be utilized as features of downstream tasks. However, GCL is vulnerable to poisoning attacks as with existing learning models. A state-of-the-art defense cannot sufficiently negate adverse effects by poisoned graphs although such a defense introduces adversarial training in the GCL. To achieve further improvement, pruning adversarial edges is important. To the best of our knowledge, the feasibility remains unexplored in the GCL domain. In this paper, we propose a simple defense for GCL, EdgePruner. We focus on the fact that the state-of-the-art poisoning attack on GCL tends to mainly add adversarial edges to create poisoned graphs, which means that pruning edges is important to sanitize the graphs. Thus, EdgePruner prunes edges that contribute to minimizing the contrastive loss based on the node representation obtained after training on poisoned graphs by GCL. Furthermore, we focus on the fact that nodes with distinct features are connected by adversarial edges in poisoned graphs. Thus, we introduce feature similarity between neighboring nodes to help more appropriately determine adversarial edges. This similarity is helpful in further eliminating adverse effects from poisoned graphs on various datasets. Finally, EdgePruner outputs a graph that yields the minimum contrastive loss as the sanitized graph. Our results demonstrate that pruning adversarial edges is feasible on six datasets. EdgePruner can improve the accuracy of node classification under the attack by up to 5.55% compared with that of the state-of-the-art defense. Moreover, we show that EdgePruner is immune to an adaptive attack.

摘要: 图对比学习（GCL）是一种无监督的图表示学习，可以获得未知节点的有用表示。节点表示可以用作下游任务的特征。然而，GCL与现有的学习模型一样容易受到中毒攻击。一个国家的最先进的防御不能充分否定有毒的图形的不利影响，虽然这样的防御介绍了对抗性的训练在GCL。为了实现进一步的改进，修剪对抗边缘是重要的。据我们所知，在GCL领域的可行性尚未探索。在本文中，我们提出了一个简单的防御GCL，EdgePruner。我们关注的事实是，对GCL的最先进的中毒攻击往往主要是添加对抗性的边缘来创建中毒的图，这意味着修剪边缘对净化图很重要。因此，EdgePruner基于GCL在中毒图上训练后获得的节点表示来修剪有助于最小化对比度损失的边缘。此外，我们专注于这样一个事实，即具有不同功能的节点连接的敌对边缘中毒图。因此，我们在相邻节点之间引入特征相似性，以帮助更适当地确定对抗边缘。这种相似性有助于进一步消除中毒图对各种数据集的不利影响。最后，EdgePruner输出一个产生最小对比损失的图作为净化图。我们的研究结果表明，修剪敌对的边缘是可行的六个数据集。EdgePruner在攻击下的节点分类准确率比最先进的防御方法提高了5.55%。此外，我们表明，EdgePruner是免疫自适应攻击。



## **40. Attacking the Loop: Adversarial Attacks on Graph-based Loop Closure Detection**

攻击循环：基于图的循环闭合检测的对抗性攻击 cs.CV

Accepted at VISIGRAPP 2024, 8 pages

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06991v1) [paper-pdf](http://arxiv.org/pdf/2312.06991v1)

**Authors**: Jonathan J. Y. Kim, Martin Urschler, Patricia J. Riddle, Jorg S. Wicker

**Abstract**: With the advancement in robotics, it is becoming increasingly common for large factories and warehouses to incorporate visual SLAM (vSLAM) enabled automated robots that operate closely next to humans. This makes any adversarial attacks on vSLAM components potentially detrimental to humans working alongside them. Loop Closure Detection (LCD) is a crucial component in vSLAM that minimizes the accumulation of drift in mapping, since even a small drift can accumulate into a significant drift over time. A prior work by Kim et al., SymbioLCD2, unified visual features and semantic objects into a single graph structure for finding loop closure candidates. While this provided a performance improvement over visual feature-based LCD, it also created a single point of vulnerability for potential graph-based adversarial attacks. Unlike previously reported visual-patch based attacks, small graph perturbations are far more challenging to detect, making them a more significant threat. In this paper, we present Adversarial-LCD, a novel black-box evasion attack framework that employs an eigencentrality-based perturbation method and an SVM-RBF surrogate model with a Weisfeiler-Lehman feature extractor for attacking graph-based LCD. Our evaluation shows that the attack performance of Adversarial-LCD with the SVM-RBF surrogate model was superior to that of other machine learning surrogate algorithms, including SVM-linear, SVM-polynomial, and Bayesian classifier, demonstrating the effectiveness of our attack framework. Furthermore, we show that our eigencentrality-based perturbation method outperforms other algorithms, such as Random-walk and Shortest-path, highlighting the efficiency of Adversarial-LCD's perturbation selection method.

摘要: 随着机器人技术的进步，大型工厂和仓库越来越普遍地采用支持视觉SLAM（vSLAM）的自动化机器人，这些机器人在人类旁边工作。这使得对vSLAM组件的任何对抗性攻击都可能对与它们一起工作的人类有害。环路闭合检测（LCD）是vSLAM中的一个关键组件，可最大限度地减少标测中的漂移累积，因为即使是很小的漂移也会随着时间的推移累积成显著的漂移。Kim等人先前的工作，SymbioLCD 2将视觉特征和语义对象统一到一个单一的图结构中，用于查找循环闭合候选项。虽然这提供了基于视觉特征的LCD的性能改进，但它也为潜在的基于图的对抗性攻击创建了单点漏洞。与以前报道的基于视觉补丁的攻击不同，小图扰动更难检测，使其成为更重要的威胁。在本文中，我们提出了Adversarial-LCD，一种新的黑盒规避攻击框架，采用基于特征中心的扰动方法和SVM-RBF代理模型与Weisfeiler-Lehman特征提取攻击基于图形的LCD。实验结果表明，采用SVM-RBF代理模型的Adversarial-LCD攻击性能优于其他机器学习代理算法，包括SVM线性、SVM多项式和贝叶斯分类器，证明了该攻击框架的有效性。此外，我们表明，我们的特征中心为基础的扰动方法优于其他算法，如随机行走和最短路径，突出的效率，对抗LCD的扰动选择方法。



## **41. Task-Agnostic Privacy-Preserving Representation Learning for Federated Learning Against Attribute Inference Attacks**

针对属性推理攻击的任务无关隐私保护表示学习 cs.CR

Accepted by AAAI 2024; Full version

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06989v1) [paper-pdf](http://arxiv.org/pdf/2312.06989v1)

**Authors**: Caridad Arroyo Arevalo, Sayedeh Leila Noorbakhsh, Yun Dong, Yuan Hong, Binghui Wang

**Abstract**: Federated learning (FL) has been widely studied recently due to its property to collaboratively train data from different devices without sharing the raw data. Nevertheless, recent studies show that an adversary can still be possible to infer private information about devices' data, e.g., sensitive attributes such as income, race, and sexual orientation. To mitigate the attribute inference attacks, various existing privacy-preserving FL methods can be adopted/adapted. However, all these existing methods have key limitations: they need to know the FL task in advance, or have intolerable computational overheads or utility losses, or do not have provable privacy guarantees.   We address these issues and design a task-agnostic privacy-preserving presentation learning method for FL ({\bf TAPPFL}) against attribute inference attacks. TAPPFL is formulated via information theory. Specifically, TAPPFL has two mutual information goals, where one goal learns task-agnostic data representations that contain the least information about the private attribute in each device's data, and the other goal ensures the learnt data representations include as much information as possible about the device data to maintain FL utility. We also derive privacy guarantees of TAPPFL against worst-case attribute inference attacks, as well as the inherent tradeoff between utility preservation and privacy protection. Extensive results on multiple datasets and applications validate the effectiveness of TAPPFL to protect data privacy, maintain the FL utility, and be efficient as well. Experimental results also show that TAPPFL outperforms the existing defenses\footnote{Source code and full version: \url{https://github.com/TAPPFL}}.

摘要: 联合学习(FL)因其能够在不共享原始数据的情况下协作训练来自不同设备的数据而受到广泛研究。然而，最近的研究表明，对手仍然有可能推断出有关设备数据的私人信息，例如收入、种族和性取向等敏感属性。为了减轻属性推理攻击，可以采用/修改现有的各种隐私保护FL方法。然而，所有这些现有的方法都有关键的局限性：它们需要提前知道FL任务，或者具有无法容忍的计算开销或效用损失，或者没有可证明的隐私保证。针对这些问题，我们设计了一种与任务无关的隐私保护的FL呈现学习方法({\bf TAPPFL})来抵抗属性推理攻击。TAPPFL是通过信息论来制定的。具体地说，TAPPFL具有两个互信息目标，其中一个目标学习与任务无关的数据表示，该数据表示包含关于每个设备的数据中的私有属性的最少信息，而另一个目标确保所学习的数据表示包括尽可能多的关于设备数据的信息以维持FL效用。我们还推导了TAPPFL对最坏情况下的属性推理攻击的隐私保证，以及效用保护和隐私保护之间的内在权衡。在多个数据集和应用程序上的广泛结果验证了TAPPFL在保护数据隐私、维护FL效用以及高效方面的有效性。实验结果还表明，TAPPFL的性能优于现有的防御措施\脚注{源代码和完整版本：\url{https://github.com/TAPPFL}}.



## **42. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

基于自提示校正的针对精调大型语言模型的实用隶属度推理攻击 cs.CL

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2311.06062v2) [paper-pdf](http://arxiv.org/pdf/2311.06062v2)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **43. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全对齐：作为上下文攻击的弱对齐总结 cs.CL

17 pages,10 figures

**SubmitDate**: 2023-12-12    [abs](http://arxiv.org/abs/2312.06924v1) [paper-pdf](http://arxiv.org/pdf/2312.06924v1)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integraty of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型（LLM）的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否充分符合安全考虑？我们的研究专注于通过对抗性攻击获得的安全敏感文档，揭示了各种NLP任务的安全性差异。例如，LLM可以有效地总结恶意的长文档，但通常拒绝翻译它们。这种差异突出了一个以前未发现的漏洞：利用安全性较弱的任务（如摘要）的攻击可能会损害传统上被认为更强大的任务（如翻译和问答（QA））的完整性。此外，同时使用多个NLP任务，安全性较低，增加了LLM无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama 2模型和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **44. Adversarial Estimation of Topological Dimension with Harmonic Score Maps**

利用调和分数图对拓扑维的对抗性估计 cs.LG

Accepted to the NeurIPS'23 Workshop on Diffusion Models

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06869v1) [paper-pdf](http://arxiv.org/pdf/2312.06869v1)

**Authors**: Eric Yeats, Cameron Darwin, Frank Liu, Hai Li

**Abstract**: Quantification of the number of variables needed to locally explain complex data is often the first step to better understanding it. Existing techniques from intrinsic dimension estimation leverage statistical models to glean this information from samples within a neighborhood. However, existing methods often rely on well-picked hyperparameters and ample data as manifold dimension and curvature increases. Leveraging insight into the fixed point of the score matching objective as the score map is regularized by its Dirichlet energy, we show that it is possible to retrieve the topological dimension of the manifold learned by the score map. We then introduce a novel method to measure the learned manifold's topological dimension (i.e., local intrinsic dimension) using adversarial attacks, thereby generating useful interpretations of the learned manifold.

摘要: 量化局部解释复杂数据所需的变量数量往往是更好地理解数据的第一步。现有的内在维度估计技术利用统计模型从邻域内的样本中收集这些信息。然而，随着流形维度和曲率的增加，现有的方法往往依赖于精选的超参数和大量的数据。利用对分数匹配目标的不动点的洞察，我们证明了通过分数映射学习的流形的拓扑维度是可能的。然后，我们介绍了一种使用对抗性攻击来测量学习流形的拓扑维(即局部本征维度)的新方法，从而产生对学习流形的有用解释。



## **45. Adversarial Purification with the Manifold Hypothesis**

流形假设下的对抗性净化 cs.LG

Extended version of paper accepted at AAAI 2024 with supplementary  materials

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2210.14404v4) [paper-pdf](http://arxiv.org/pdf/2210.14404v4)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework for adversarial robustness using the manifold hypothesis. This framework provides sufficient conditions for defending against adversarial examples. We develop an adversarial purification method with this framework. Our method combines manifold learning with variational inference to provide adversarial robustness without the need for expensive adversarial training. Experimentally, our approach can provide adversarial robustness even if attackers are aware of the existence of the defense. In addition, our method can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。这一框架为对抗对手的例子提供了充分的条件。在此框架下，我们提出了一种对抗性净化方法。我们的方法结合了流形学习和变分推理，在不需要昂贵的对抗性训练的情况下提供对抗性健壮性。在实验上，即使攻击者知道防御的存在，我们的方法也可以提供对抗的健壮性。此外，我们的方法还可以作为变分自动编码器的测试时间防御机制。



## **46. Sparse but Strong: Crafting Adversarially Robust Graph Lottery Tickets**

稀疏但强大：精心制作异常健壮的图形彩票 cs.LG

Accepted at NeurIPS 2023 GLFrontiers Workshop

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06568v1) [paper-pdf](http://arxiv.org/pdf/2312.06568v1)

**Authors**: Subhajit Dutta Chowdhury, Zhiyu Ni, Qingyuan Peng, Souvik Kundu, Pierluigi Nuzzo

**Abstract**: Graph Lottery Tickets (GLTs), comprising a sparse adjacency matrix and a sparse graph neural network (GNN), can significantly reduce the inference latency and compute footprint compared to their dense counterparts. Despite these benefits, their performance against adversarial structure perturbations remains to be fully explored. In this work, we first investigate the resilience of GLTs against different structure perturbation attacks and observe that they are highly vulnerable and show a large drop in classification accuracy. Based on this observation, we then present an adversarially robust graph sparsification (ARGS) framework that prunes the adjacency matrix and the GNN weights by optimizing a novel loss function capturing the graph homophily property and information associated with both the true labels of the train nodes and the pseudo labels of the test nodes. By iteratively applying ARGS to prune both the perturbed graph adjacency matrix and the GNN model weights, we can find adversarially robust graph lottery tickets that are highly sparse yet achieve competitive performance under different untargeted training-time structure attacks. Evaluations conducted on various benchmarks, considering different poisoning structure attacks, namely, PGD, MetaAttack, Meta-PGD, and PR-BCD demonstrate that the GLTs generated by ARGS can significantly improve the robustness, even when subjected to high levels of sparsity.

摘要: 图彩票(GLTS)由一个稀疏邻接矩阵和一个稀疏图神经网络(GNN)组成，与稠密彩票相比，它可以显著减少推理延迟和计算量。尽管有这些好处，但它们在对抗对抗性结构扰动时的性能仍有待充分探讨。在这项工作中，我们首先研究了GLT对不同结构扰动攻击的弹性，并观察到它们具有高度的脆弱性，并且分类精度下降很大。在此基础上，我们提出了一种对抗性的图稀疏化框架，该框架通过优化一种新的损失函数来剪枝邻接矩阵和GNN权重，该损失函数捕捉了图的同伦性质以及与训练节点的真实标签和测试节点的伪标签相关联的信息。通过迭代应用ARGS修剪扰动图邻接矩阵和GNN模型权重，我们可以找到高度稀疏但在不同非目标训练时间结构攻击下具有竞争性性能的对抗性健壮图彩票。考虑到不同的中毒结构攻击，即PGD、MetaAttack、Meta-PGD和PR-BCD，在不同的基准上进行的评估表明，即使在高度稀疏的情况下，由ARGs生成的GLT也可以显著提高鲁棒性。



## **47. Robust Graph Neural Network based on Graph Denoising**

基于图去噪的稳健图神经网络 cs.LG

Presented in the 2023 Asilomar Conference on Signals, Systems, and  Computers (Oct. 29th - Nov 1st, 2023)

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06557v1) [paper-pdf](http://arxiv.org/pdf/2312.06557v1)

**Authors**: Victor M. Tenorio, Samuel Rey, Antonio G. Marques

**Abstract**: Graph Neural Networks (GNNs) have emerged as a notorious alternative to address learning problems dealing with non-Euclidean datasets. However, although most works assume that the graph is perfectly known, the observed topology is prone to errors stemming from observational noise, graph-learning limitations, or adversarial attacks. If ignored, these perturbations may drastically hinder the performance of GNNs. To address this limitation, this work proposes a robust implementation of GNNs that explicitly accounts for the presence of perturbations in the observed topology. For any task involving GNNs, our core idea is to i) solve an optimization problem not only over the learnable parameters of the GNN but also over the true graph, and ii) augment the fitting cost with a term accounting for discrepancies on the graph. Specifically, we consider a convolutional GNN based on graph filters and follow an alternating optimization approach to handle the (non-differentiable and constrained) optimization problem by combining gradient descent and projected proximal updates. The resulting algorithm is not limited to a particular type of graph and is amenable to incorporating prior information about the perturbations. Finally, we assess the performance of the proposed method through several numerical experiments.

摘要: 图形神经网络(GNN)已经成为解决非欧几里得数据集学习问题的一种臭名昭著的选择。然而，尽管大多数工作都假设图是完全已知的，但所观察到的拓扑结构容易由于观测噪声、图学习限制或敌意攻击而产生错误。如果忽视，这些扰动可能会极大地阻碍GNN的性能。为了解决这一限制，这项工作提出了一种健壮的GNN实现，明确地说明了观测到的拓扑中存在的扰动。对于任何涉及GNN的任务，我们的核心思想是i)不仅解决GNN的可学习参数的优化问题，而且解决真实图上的优化问题，以及ii)在图上增加一个考虑差异的项来增加拟合成本。具体地说，我们考虑了基于图过滤器的卷积GNN，并遵循交替优化方法通过结合梯度下降和投影近邻更新来处理(不可微且受约束的)优化问题。所得到的算法不限于特定类型的图，并且可以合并关于扰动的先验信息。最后，我们通过几个数值实验对该方法的性能进行了评估。



## **48. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

DIFFender：基于扩散的补丁攻击对抗性防御 cs.CV

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2306.09124v3) [paper-pdf](http://arxiv.org/pdf/2306.09124v3)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is unsatisfactory. In this paper, we propose DIFFender, a novel defense method that leverages a text-guided diffusion model to defend against adversarial patches. DIFFender includes two main stages: patch localization and patch restoration. In the localization stage, we find and exploit an intriguing property of the diffusion model to precisely identify the locations of adversarial patches. In the restoration stage, we employ the diffusion model to reconstruct the adversarial regions in the images while preserving the integrity of the visual content. Thanks to the former finding, these two stages can be simultaneously guided by a unified diffusion model. Thus, we can utilize the close interaction between them to improve the whole defense performance. Moreover, we propose a few-shot prompt-tuning algorithm to fine-tune the diffusion model, enabling the pre-trained diffusion model to adapt to the defense task easily. We conduct extensive experiments on image classification, face recognition, and further in the physical world, demonstrating that our proposed method exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple patch attack methods.

摘要: 对抗性攻击，特别是补丁攻击，对深度学习模型的鲁棒性和可靠性构成了重大威胁。针对补丁攻击开发可靠的防御对于现实世界的应用至关重要，但目前在这方面的研究并不令人满意。在本文中，我们提出了一种新的防御方法，它利用文本引导的扩散模型来防御对抗性补丁。复原器包括两个主要阶段：斑块定位和斑块恢复。在定位阶段，我们发现并利用扩散模型的一个有趣的特性来精确地识别对抗补丁的位置。在恢复阶段，我们采用扩散模型来重建图像中的敌对区域，同时保持视觉内容的完整性。由于前一个发现，这两个阶段可以同时指导一个统一的扩散模型。因此，我们可以利用它们之间的密切互动，以提高整体的防御性能。此外，我们提出了一个少镜头的快速调整算法来微调扩散模型，使预先训练的扩散模型，以适应防御任务很容易。我们对图像分类，人脸识别，并进一步在物理世界中进行了广泛的实验，证明了我们提出的方法在强自适应攻击下具有出色的鲁棒性，并在各种场景，不同的分类器和多种补丁攻击方法中具有很好的泛化能力。



## **49. MalPurifier: Enhancing Android Malware Detection with Adversarial Purification against Evasion Attacks**

MalPurier：通过对抗逃避攻击的对抗性净化增强Android恶意软件检测 cs.CR

14 pages; In submission

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2312.06423v1) [paper-pdf](http://arxiv.org/pdf/2312.06423v1)

**Authors**: Yuyang Zhou, Guang Cheng, Zongyao Chen, Shui Yu

**Abstract**: Machine learning (ML) has gained significant adoption in Android malware detection to address the escalating threats posed by the rapid proliferation of malware attacks. However, recent studies have revealed the inherent vulnerabilities of ML-based detection systems to evasion attacks. While efforts have been made to address this critical issue, many of the existing defensive methods encounter challenges such as lower effectiveness or reduced generalization capabilities. In this paper, we introduce a novel Android malware detection method, MalPurifier, which exploits adversarial purification to eliminate perturbations independently, resulting in attack mitigation in a light and flexible way. Specifically, MalPurifier employs a Denoising AutoEncoder (DAE)-based purification model to preprocess input samples, removing potential perturbations from them and then leading to correct classification. To enhance defense effectiveness, we propose a diversified adversarial perturbation mechanism that strengthens the purification model against different manipulations from various evasion attacks. We also incorporate randomized "protective noises" onto benign samples to prevent excessive purification. Furthermore, we customize a loss function for improving the DAE model, combining reconstruction loss and prediction loss, to enhance feature representation learning, resulting in accurate reconstruction and classification. Experimental results on two Android malware datasets demonstrate that MalPurifier outperforms the state-of-the-art defenses, and it significantly strengthens the vulnerable malware detector against 37 evasion attacks, achieving accuracies over 90.91%. Notably, MalPurifier demonstrates easy scalability to other detectors, offering flexibility and robustness in its implementation.

摘要: 机器学习（ML）在Android恶意软件检测中获得了广泛采用，以解决恶意软件攻击快速扩散所带来的不断升级的威胁。然而，最近的研究已经揭示了基于ML的检测系统对规避攻击的固有脆弱性。虽然已经做出努力来解决这个关键问题，但许多现有的防御方法遇到了诸如有效性较低或泛化能力降低等挑战。在本文中，我们介绍了一种新的Android恶意软件检测方法MalPurifier，该方法利用对抗性净化来独立消除扰动，从而以轻松灵活的方式减轻攻击。具体来说，MalPurifier采用基于去噪自动编码器（DAE）的纯化模型来预处理输入样本，从其中去除潜在的扰动，然后进行正确的分类。为了提高防御效果，我们提出了一个多样化的对抗扰动机制，加强了净化模型对不同的操作，从各种规避攻击。我们还将随机的“保护性噪音”加入到良性样本中，以防止过度纯化。此外，我们自定义了一个损失函数来改进DAE模型，将重建损失和预测损失结合起来，以增强特征表示学习，从而实现准确的重建和分类。在两个Android恶意软件数据集上的实验结果表明，MalPurifier的性能优于最先进的防御方法，它显著增强了易受攻击的恶意软件检测器对37种规避攻击的抵御能力，准确率超过90.91%。值得注意的是，MalPurifier展示了对其他检测器的轻松可扩展性，在其实现中提供了灵活性和鲁棒性。



## **50. Bidirectional Contrastive Split Learning for Visual Question Answering**

视觉问答中的双向对比分裂学习 cs.CV

Accepted for AAAI 2024

**SubmitDate**: 2023-12-11    [abs](http://arxiv.org/abs/2208.11435v4) [paper-pdf](http://arxiv.org/pdf/2208.11435v4)

**Authors**: Yuwei Sun, Hideya Ochiai

**Abstract**: Visual Question Answering (VQA) based on multi-modal data facilitates real-life applications such as home robots and medical diagnoses. One significant challenge is to devise a robust decentralized learning framework for various client models where centralized data collection is refrained due to confidentiality concerns. This work aims to tackle privacy-preserving VQA by decoupling a multi-modal model into representation modules and a contrastive module and leveraging inter-module gradients sharing and inter-client weight sharing. To this end, we propose Bidirectional Contrastive Split Learning (BiCSL) to train a global multi-modal model on the entire data distribution of decentralized clients. We employ the contrastive loss that enables a more efficient self-supervised learning of decentralized modules. Comprehensive experiments are conducted on the VQA-v2 dataset based on five SOTA VQA models, demonstrating the effectiveness of the proposed method. Furthermore, we inspect BiCSL's robustness against a dual-key backdoor attack on VQA. Consequently, BiCSL shows much better robustness to the multi-modal adversarial attack compared to the centralized learning method, which provides a promising approach to decentralized multi-modal learning.

摘要: 基于多模式数据的视觉问答(VQA)为家庭机器人和医疗诊断等实际应用提供了便利。一个重大挑战是为各种客户模型设计一个强大的分散学习框架，在这些客户模型中，出于保密考虑，不再进行集中的数据收集。该工作旨在通过将多通道模型解耦为表示模块和对比模块，并利用模块间梯度共享和客户间权重分担来解决隐私保护VQA问题。为此，我们提出了双向对比分裂学习(BiCSL)来训练一个全局多模式的分布式客户端的整体数据分布模型。我们采用对比损失，使去中心化模块的自我监督学习更有效。在基于五种SOTA VQA模型的VQA-v2数据集上进行了全面的实验，验证了该方法的有效性。此外，我们还检查了BiCSL对VQA的双密钥后门攻击的健壮性。因此，与集中式学习方法相比，BiCSL具有更好的抗多模式攻击的鲁棒性，为分布式多模式学习提供了一种很有前途的方法。



