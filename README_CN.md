# Latest Adversarial Attack Papers
**update at 2024-11-22 10:20:01**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Indiscriminate Disruption of Conditional Inference on Multivariate Gaussians**

不加区别地破坏多元高斯的条件推理 stat.ML

30 pages, 6 figures; 4 tables

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14351v1) [paper-pdf](http://arxiv.org/pdf/2411.14351v1)

**Authors**: William N. Caballero, Matthew LaRosa, Alexander Fisher, Vahid Tarokh

**Abstract**: The multivariate Gaussian distribution underpins myriad operations-research, decision-analytic, and machine-learning models (e.g., Bayesian optimization, Gaussian influence diagrams, and variational autoencoders). However, despite recent advances in adversarial machine learning (AML), inference for Gaussian models in the presence of an adversary is notably understudied. Therefore, we consider a self-interested attacker who wishes to disrupt a decisionmaker's conditional inference and subsequent actions by corrupting a set of evidentiary variables. To avoid detection, the attacker also desires the attack to appear plausible wherein plausibility is determined by the density of the corrupted evidence. We consider white- and grey-box settings such that the attacker has complete and incomplete knowledge about the decisionmaker's underlying multivariate Gaussian distribution, respectively. Select instances are shown to reduce to quadratic and stochastic quadratic programs, and structural properties are derived to inform solution methods. We assess the impact and efficacy of these attacks in three examples, including, real estate evaluation, interest rate estimation and signals processing. Each example leverages an alternative underlying model, thereby highlighting the attacks' broad applicability. Through these applications, we also juxtapose the behavior of the white- and grey-box attacks to understand how uncertainty and structure affect attacker behavior.

摘要: 多变量高斯分布支撑着无数的操作研究、决策分析和机器学习模型(例如，贝叶斯优化、高斯影响图和变分自动编码器)。然而，尽管最近在对抗性机器学习(AML)方面取得了进展，但在对手存在的情况下对高斯模型的推理研究明显不足。因此，我们考虑一个自私自利的攻击者，他希望通过破坏一组证据变量来扰乱决策者的条件推理和后续行动。为了避免被发现，攻击者还希望攻击看起来像是可信的，其中可信程度取决于被破坏证据的密度。我们考虑白盒和灰盒设置，使得攻击者对决策者潜在的多变量高斯分布分别有完全和不完全的了解。文中给出了精选实例，将其归结为二次规划和随机二次规划，并给出了求解方法的结构性质。我们通过三个例子来评估这些攻击的影响和效果，包括房地产评估、利率估计和信号处理。每个示例都利用了一个可选的底层模型，从而突出了攻击的广泛适用性。通过这些应用，我们还比较了白盒攻击和灰盒攻击的行为，以了解不确定性和结构如何影响攻击者的行为。



## **2. Layer Pruning with Consensus: A Triple-Win Solution**

达成共识的分层修剪：三赢解决方案 cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14345v1) [paper-pdf](http://arxiv.org/pdf/2411.14345v1)

**Authors**: Leandro Giusti Mugnaini, Carolina Tavares Duarte, Anna H. Reali Costa, Artur Jordao

**Abstract**: Layer pruning offers a promising alternative to standard structured pruning, effectively reducing computational costs, latency, and memory footprint. While notable layer-pruning approaches aim to detect unimportant layers for removal, they often rely on single criteria that may not fully capture the complex, underlying properties of layers. We propose a novel approach that combines multiple similarity metrics into a single expressive measure of low-importance layers, called the Consensus criterion. Our technique delivers a triple-win solution: low accuracy drop, high-performance improvement, and increased robustness to adversarial attacks. With up to 78.80% FLOPs reduction and performance on par with state-of-the-art methods across different benchmarks, our approach reduces energy consumption and carbon emissions by up to 66.99% and 68.75%, respectively. Additionally, it avoids shortcut learning and improves robustness by up to 4 percentage points under various adversarial attacks. Overall, the Consensus criterion demonstrates its effectiveness in creating robust, efficient, and environmentally friendly pruned models.

摘要: 层修剪提供了标准结构化修剪的一种很有前途的替代方案，有效地降低了计算成本、延迟和内存占用。虽然显著的层修剪方法旨在检测要移除的不重要的层，但它们通常依赖于单一的标准，可能不能完全捕捉层的复杂的潜在属性。我们提出了一种新的方法，该方法将多个相似性度量结合到一个低重要性层的表达度量中，称为共识准则。我们的技术提供了一个三赢的解决方案：低精确度下降，高性能改进，以及增强对对手攻击的健壮性。在不同的基准测试中，我们的方法减少了高达78.80%的FLOPS，性能与最先进的方法持平，能耗和碳排放分别减少了66.99%和68.75%。此外，它避免了快捷学习，并在各种对抗性攻击下将健壮性提高了高达4个百分点。总体而言，共识标准证明了其在创建健壮、高效和环境友好的修剪模型方面的有效性。



## **3. Generating Realistic Adversarial Examples for Business Processes using Variational Autoencoders**

使用变分自动编码器生成业务流程的现实对抗示例 cs.LG

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14263v1) [paper-pdf](http://arxiv.org/pdf/2411.14263v1)

**Authors**: Alexander Stevens, Jari Peeperkorn, Johannes De Smedt, Jochen De Weerdt

**Abstract**: In predictive process monitoring, predictive models are vulnerable to adversarial attacks, where input perturbations can lead to incorrect predictions. Unlike in computer vision, where these perturbations are designed to be imperceptible to the human eye, the generation of adversarial examples in predictive process monitoring poses unique challenges. Minor changes to the activity sequences can create improbable or even impossible scenarios to occur due to underlying constraints such as regulatory rules or process constraints. To address this, we focus on generating realistic adversarial examples tailored to the business process context, in contrast to the imperceptible, pixel-level changes commonly seen in computer vision adversarial attacks. This paper introduces two novel latent space attacks, which generate adversaries by adding noise to the latent space representation of the input data, rather than directly modifying the input attributes. These latent space methods are domain-agnostic and do not rely on process-specific knowledge, as we restrict the generation of adversarial examples to the learned class-specific data distributions by directly perturbing the latent space representation of the business process executions. We evaluate these two latent space methods with six other adversarial attacking methods on eleven real-life event logs and four predictive models. The first three attacking methods directly permute the activities of the historically observed business process executions. The fourth method constrains the adversarial examples to lie within the same data distribution as the original instances, by projecting the adversarial examples to the original data distribution.

摘要: 在预测过程监控中，预测模型容易受到对抗性攻击，输入扰动会导致不正确的预测。与计算机视觉不同，在计算机视觉中，这些扰动被设计为人眼无法察觉，而在预测过程监控中生成对抗性示例构成了独特的挑战。活动序列的微小更改可能会由于潜在的约束(如法规规则或流程约束)而创建不太可能甚至不可能发生的场景。为了解决这一问题，我们专注于生成针对业务流程上下文量身定做的现实对抗性示例，这与计算机视觉对抗性攻击中常见的难以察觉的像素级变化形成对比。本文介绍了两种新的潜在空间攻击，它们通过在输入数据的潜在空间表示中添加噪声来产生攻击，而不是直接修改输入属性。这些潜在空间方法是领域不可知的，不依赖于特定于流程的知识，因为我们通过直接扰动业务流程执行的潜在空间表示，将对抗性示例的生成限制在学习的特定类的数据分布上。我们在11个真实事件日志和4个预测模型上对这两种潜在空间方法和其他6种对抗性攻击方法进行了评估。前三种攻击方法直接改变了历史上观察到的业务流程执行的活动。第四种方法通过将对抗性实例投影到原始数据分布来约束对抗性实例位于与原始实例相同的数据分布内。



## **4. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

AnywhereDoor：对象检测的多目标后门攻击 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14243v1) [paper-pdf](http://arxiv.org/pdf/2411.14243v1)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a significant threat by implanting hidden backdoor in a victim model, which adversaries can later exploit to trigger malicious behaviors during inference. However, current backdoor techniques are limited to static scenarios where attackers must define a malicious objective before training, locking the attack into a predetermined action without inference-time adaptability. Given the expressive output space in object detection, including object existence detection, bounding box estimation, and object classification, the feasibility of implanting a backdoor that provides inference-time control with a high degree of freedom remains unexplored. This paper introduces AnywhereDoor, a flexible backdoor attack tailored for object detection. Once implanted, AnywhereDoor enables adversaries to specify different attack types (object vanishing, fabrication, or misclassification) and configurations (untargeted or targeted with specific classes) to dynamically control detection behavior. This flexibility is achieved through three key innovations: (i) objective disentanglement to support a broader range of attack combinations well beyond what existing methods allow; (ii) trigger mosaicking to ensure backdoor activations are robust, even against those object detectors that extract localized regions from the input image for recognition; and (iii) strategic batching to address object-level data imbalances that otherwise hinders a balanced manipulation. Extensive experiments demonstrate that AnywhereDoor provides attackers with a high degree of control, achieving an attack success rate improvement of nearly 80% compared to adaptations of existing methods for such flexible control.

摘要: 随着对象检测成为许多安全关键型应用程序不可或缺的一部分，了解其漏洞至关重要。尤其是后门攻击，通过在受害者模型中植入隐藏的后门，构成了重大威胁，攻击者稍后可以利用这些后门在推理过程中触发恶意行为。然而，目前的后门技术仅限于静态场景，其中攻击者必须在训练前定义恶意目标，将攻击锁定在预定的动作中，而不是推理时间适应性。考虑到目标检测中的表达输出空间，包括目标存在检测、包围盒估计和目标分类，植入具有高度自由度的推理时间控制的后门的可行性仍未被探索。本文介绍了Anywhere Door，一种为目标检测量身定做的灵活的后门攻击。一旦被植入，Anywhere Door使攻击者能够指定不同的攻击类型(对象消失、捏造或错误分类)和配置(非目标或特定类别的目标)，以动态控制检测行为。这种灵活性是通过三项关键创新实现的：(I)客观解缠，以支持远远超出现有方法所允许的更广泛的攻击组合；(Ii)触发马赛克，以确保后门激活是稳健的，即使针对那些从输入图像中提取局部区域以进行识别的对象检测器；以及(Iii)战略批处理，以解决否则阻碍平衡操纵的对象级别的数据失衡。广泛的实验表明，Anywhere Door为攻击者提供了高度的控制，与采用现有方法进行这种灵活的控制相比，攻击成功率提高了近80%。



## **5. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

GISP：针对越狱LLM的对抗性后缀的高效黑匣子生成 cs.LG

28 pages, 9 tables, 13 figures; under review at CVPR '25

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14133v1) [paper-pdf](http://arxiv.org/pdf/2411.14133v1)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: Large Language Models (LLMs) have shown impressive proficiency across a range of natural language processing tasks yet remain vulnerable to adversarial prompts, known as jailbreak attacks, carefully designed to elicit harmful responses from LLMs. Traditional methods rely on manual heuristics, which suffer from limited generalizability. While being automatic, optimization-based attacks often produce unnatural jailbreak prompts that are easy to detect by safety filters or require high computational overhead due to discrete token optimization. Witnessing the limitations of existing jailbreak methods, we introduce Generative Adversarial Suffix Prompter (GASP), a novel framework that combines human-readable prompt generation with Latent Bayesian Optimization (LBO) to improve adversarial suffix creation in a fully black-box setting. GASP leverages LBO to craft adversarial suffixes by efficiently exploring continuous embedding spaces, gradually optimizing the model to improve attack efficacy while balancing prompt coherence through a targeted iterative refinement procedure. Our experiments show that GASP can generate natural jailbreak prompts, significantly improving attack success rates, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.

摘要: 大型语言模型(LLM)在一系列自然语言处理任务中表现出令人印象深刻的熟练程度，但仍然容易受到对手提示的攻击，这种提示被称为越狱攻击，这些提示是精心设计的，旨在引起LLM的有害反应。传统的方法依赖于人工启发式方法，泛化能力有限。虽然基于优化的攻击是自动的，但通常会产生不自然的越狱提示，这些提示很容易被安全过滤器检测到，或者由于离散令牌优化而需要很高的计算开销。鉴于现有越狱方法的局限性，我们引入了生成性对抗性后缀提示器(GAP)，这是一种将人类可读的提示生成与潜在贝叶斯优化(LBO)相结合的新框架，以改进完全黑盒环境下的对抗性后缀创建。GASP利用LBO通过有效地探索连续嵌入空间来创建对抗性后缀，逐步优化模型以提高攻击效率，同时通过有针对性的迭代细化过程平衡即时一致性。我们的实验表明，GAP能够生成自然的越狱提示，显著提高了攻击成功率，减少了训练次数，加快了推理速度，从而使其成为红队LLMS的一种高效和可扩展的解决方案。



## **6. RAG-Thief: Scalable Extraction of Private Data from Retrieval-Augmented Generation Applications with Agent-based Attacks**

RAG-Thief：利用基于代理的攻击从检索增强生成应用程序中可扩展地提取私人数据 cs.CR

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.14110v1) [paper-pdf](http://arxiv.org/pdf/2411.14110v1)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Min Yang

**Abstract**: While large language models (LLMs) have achieved notable success in generative tasks, they still face limitations, such as lacking up-to-date knowledge and producing hallucinations. Retrieval-Augmented Generation (RAG) enhances LLM performance by integrating external knowledge bases, providing additional context which significantly improves accuracy and knowledge coverage. However, building these external knowledge bases often requires substantial resources and may involve sensitive information. In this paper, we propose an agent-based automated privacy attack called RAG-Thief, which can extract a scalable amount of private data from the private database used in RAG applications. We conduct a systematic study on the privacy risks associated with RAG applications, revealing that the vulnerability of LLMs makes the private knowledge bases suffer significant privacy risks. Unlike previous manual attacks which rely on traditional prompt injection techniques, RAG-Thief starts with an initial adversarial query and learns from model responses, progressively generating new queries to extract as many chunks from the knowledge base as possible. Experimental results show that our RAG-Thief can extract over 70% information from the private knowledge bases within customized RAG applications deployed on local machines and real-world platforms, including OpenAI's GPTs and ByteDance's Coze. Our findings highlight the privacy vulnerabilities in current RAG applications and underscore the pressing need for stronger safeguards.

摘要: 虽然大型语言模型在生成性任务中取得了显著的成功，但它们仍然面临着局限性，如缺乏最新知识和产生幻觉。检索-增强生成(RAG)通过集成外部知识库来增强LLM性能，提供额外的上下文，从而显著提高准确性和知识覆盖率。然而，建立这些外部知识库往往需要大量资源，并可能涉及敏感信息。本文提出了一种基于代理的自动隐私攻击方法RAG-Thief，它可以从RAG应用中使用的私有数据库中提取大量可伸缩的私有数据。我们对RAG应用相关的隐私风险进行了系统的研究，揭示了LLMS的漏洞使私人知识库面临着重大的隐私风险。与以前依赖传统提示注入技术的手动攻击不同，RAG-Thief从最初的对抗性查询开始，并从模型响应中学习，逐步生成新的查询以从知识库中提取尽可能多的块。实验结果表明，我们的RAG-Thief可以从本地机器和真实平台上部署的定制RAG应用程序的私有知识库中提取70%以上的信息，包括OpenAI的GPTS和ByteDance的Coze。我们的发现突显了当前RAG应用程序中的隐私漏洞，并强调了加强保护的迫切需要。



## **7. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

32 pages, 12 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2406.08298v5) [paper-pdf](http://arxiv.org/pdf/2406.08298v5)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) demonstrate remarkable performance in image classification through visual-token interaction learning, particularly when equipped with local information via region attention or convolutions. Although such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global visual-token representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy input. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformers that uses NCA as plug-and-play adaptors between ViT layers, thus enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Using our analysis of AdaNCA placement and robustness improvement, we also develop an algorithm for identifying the most effective insertion points for AdaNCA. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across eight robustness benchmarks and four ViT architectures that AdaNCA, as a plug-and-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)通过视觉-表征交互学习在图像分类中表现出显著的性能，特别是在通过区域注意或卷积获得局部信息的情况下。虽然这样的体系结构从不同的粒度提高了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)通过局部交互实现全局视觉表征的建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，从而增强了VIT的性能和对敌意样本以及分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。利用我们对AdaNCA布局和健壮性改进的分析，我们还开发了一种算法来识别AdaNCA最有效的插入点。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对八个健壮性基准和四个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **8. Verifying the Robustness of Automatic Credibility Assessment**

验证自动可信度评估的稳健性 cs.CL

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2303.08032v3) [paper-pdf](http://arxiv.org/pdf/2303.08032v3)

**Authors**: Piotr Przybyła, Alexander Shvets, Horacio Saggion

**Abstract**: Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we systematically test the robustness of common text classifiers against available attacking techniques and discover that, indeed, meaning-preserving changes in input text can mislead the models. The approaches we test focus on finding vulnerable spans in text and replacing individual characters or words, taking into account the similarity between the original and replacement content. We also introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. The attacked tasks include (1) fact checking and detection of (2) hyperpartisan news, (3) propaganda and (4) rumours. Our experimental results show that modern large language models are often more vulnerable to attacks than previous, smaller solutions, e.g. attacks on GEMMA being up to 27\% more successful than those on BERT. Finally, we manually analyse a subset adversarial examples and check what kinds of modifications are used in successful attacks.

摘要: 文本分类方法被广泛研究为检测可信度较低的内容的一种方式：假新闻、社交媒体机器人、宣传等。相当准确的模型(可能基于深度神经网络)有助于调节公共电子平台，并经常导致内容创建者面临提交的拒绝或已发布的文本的删除。出于逃避进一步检测的动机，内容创建者试图对文本进行稍微修改的版本(称为带有敌意的示例的攻击)，以利用分类器的弱点并产生不同的输出。在这里，我们系统地测试了常见文本分类器对现有攻击技术的健壮性，并发现确实，输入文本中保持意义的变化会误导模型。我们测试的方法侧重于查找文本中易受攻击的范围，并替换单个字符或单词，同时考虑到原始内容和替换内容之间的相似性。我们还引入了Bodega：一个基准，用于在四个错误信息检测任务中测试受害者模型和攻击方法，该评估框架旨在模拟真实的内容审核用例。被攻击的任务包括(1)事实核查和检测(2)超党派新闻，(3)宣传和(4)谣言。我们的实验结果表明，现代大语言模型往往比以前的较小的解决方案更容易受到攻击，例如，对Gema的攻击比对Bert的攻击成功高达27%.最后，我们手动分析了一个子集的敌意例子，并检查了在成功的攻击中使用了哪些修改。



## **9. Robust Data-Driven Predictive Control for Mixed Platoons under Noise and Attacks**

噪音和攻击下混合排的鲁棒数据驱动预测控制 eess.SY

16 pages, 7 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13924v1) [paper-pdf](http://arxiv.org/pdf/2411.13924v1)

**Authors**: Shuai Li, Chaoyi Chen, Haotian Zheng, Jiawei Wang, Qing Xu, Jianqiang Wang, Keqiang Li

**Abstract**: Controlling mixed platoons, which consist of both connected and automated vehicles (CAVs) and human-driven vehicles (HDVs), poses significant challenges due to the uncertain and unknown human driving behaviors. Data-driven control methods offer promising solutions by leveraging available trajectory data, but their performance can be compromised by process noise and adversarial attacks. To address this issue, this paper proposes a Robust Data-EnablEd Predictive Leading Cruise Control (RDeeP-LCC) framework based on data-driven reachability analysis. The framework over-approximates system dynamics under noise and attack using a matrix zonotope set derived from data, and develops a stabilizing feedback control law. By decoupling the mixed platoon system into nominal and error components, we employ data-driven reachability sets to recursively compute error reachable sets that account for noise and attacks, and obtain tightened safety constraints of the nominal system. This leads to a robust data-driven predictive control framework, solved in a tube-based control manner. Numerical simulations and human-in-the-loop experiments validate that the RDeeP-LCC method significantly enhances the robustness of mixed platoons, improving mixed traffic stability and safety against practical noise and attacks.

摘要: 由于人类驾驶行为的不确定性和未知性，控制由互联和自动车辆(CAV)和人类驾驶车辆(HDV)组成的混合排构成了巨大的挑战。数据驱动的控制方法通过利用可用的轨迹数据提供了有希望的解决方案，但其性能可能会受到过程噪声和对抗性攻击的影响。针对这一问题，提出了一种基于数据驱动可达性分析的健壮数据启用预测领先巡航控制(RDeeP-LCC)框架。该框架利用从数据中得到的矩阵区域集来过度逼近噪声和攻击下的系统动态，并发展了一种镇定反馈控制律。通过将混合排系统分解为标称部分和错误部分，利用数据驱动的可达集递归计算考虑噪声和攻击的错误可达集，得到标称系统严格的安全约束。这导致了一个稳健的数据驱动的预测控制框架，以基于管子的控制方式解决。数值模拟和人在环实验验证了RDeeP-LCC方法显著增强了混合排的稳健性，改善了混合交通的稳定性和对实际噪声和攻击的安全性。



## **10. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

Magmaw：对基于机器学习的无线通信系统的模式不可知的对抗攻击 cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2311.00207v3) [paper-pdf](http://arxiv.org/pdf/2311.00207v3)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.

摘要: 机器学习(ML)通过合并端到端无线通信系统的所有物理层块，在实现联合收发器优化方面发挥了重要作用。尽管已经有一些针对基于ML的无线系统的对抗性攻击，但现有的方法不能提供包括源数据的多模态、公共物理层协议和无线域限制在内的全面视角。本文提出了一种新的无线攻击方法Magmaw，它能够对无线信道上传输的任何多模信号产生通用的对抗性扰动。我们进一步引入了针对下游应用程序的对抗性攻击的新目标。我们采用了广泛使用的防御措施来验证Magmaw的弹性。对于概念验证评估，我们使用软件定义的无线电系统构建了一个实时无线攻击平台。实验结果表明，即使在强防御机制存在的情况下，Magmaw也会导致性能显著下降。此外，我们在加密通信通道和基于通道通道的ML模型两个案例中验证了MAGMAW的性能。



## **11. Towards Understanding Adversarial Transferability in Federated Learning**

了解联邦学习中的对抗可移植性 cs.LG

Published in Transactions on Machine Learning Research (TMLR)  (11/2024)

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2310.00616v2) [paper-pdf](http://arxiv.org/pdf/2310.00616v2)

**Authors**: Yijiang Li, Ying Gao, Haohan Wang

**Abstract**: We investigate a specific security risk in FL: a group of malicious clients has impacted the model during training by disguising their identities and acting as benign clients but later switching to an adversarial role. They use their data, which was part of the training set, to train a substitute model and conduct transferable adversarial attacks against the federated model. This type of attack is subtle and hard to detect because these clients initially appear to be benign.   The key question we address is: How robust is the FL system to such covert attacks, especially compared to traditional centralized learning systems? We empirically show that the proposed attack imposes a high security risk to current FL systems. By using only 3\% of the client's data, we achieve the highest attack rate of over 80\%. To further offer a full understanding of the challenges the FL system faces in transferable attacks, we provide a comprehensive analysis over the transfer robustness of FL across a spectrum of configurations. Surprisingly, FL systems show a higher level of robustness than their centralized counterparts, especially when both systems are equally good at handling regular, non-malicious data.   We attribute this increased robustness to two main factors: 1) Decentralized Data Training: Each client trains the model on its own data, reducing the overall impact of any single malicious client. 2) Model Update Averaging: The updates from each client are averaged together, further diluting any malicious alterations. Both practical experiments and theoretical analysis support our conclusions. This research not only sheds light on the resilience of FL systems against hidden attacks but also raises important considerations for their future application and development.

摘要: 我们调查了FL中的一个特定安全风险：一群恶意客户在培训期间通过伪装他们的身份并充当良性客户，但后来切换到对手角色，影响了模型。他们使用他们的数据，这是训练集的一部分，训练一个替代模型，并对联邦模型进行可转移的对抗性攻击。这种类型的攻击很隐蔽，很难检测到，因为这些客户端最初看起来是良性的。我们解决的关键问题是：FL系统对这种隐蔽攻击的健壮性如何，特别是与传统的集中式学习系统相比？实验表明，该攻击对现有的FL系统构成了很高的安全风险。在仅使用3个客户端数据的情况下，我们获得了超过80%的最高攻击率。为了进一步全面了解FL系统在可转移攻击中面临的挑战，我们对FL在各种配置下的传输健壮性进行了全面的分析。令人惊讶的是，FL系统表现出比集中式对应系统更高的健壮性，特别是当两个系统在处理常规、非恶意数据方面同样出色时。我们将这种增强的健壮性归因于两个主要因素：1)分散的数据训练：每个客户端根据自己的数据训练模型，减少任何单个恶意客户端的总体影响。2)模型更新平均：将来自每个客户端的更新平均在一起，进一步稀释任何恶意更改。实际实验和理论分析都支持我们的结论。这一研究不仅揭示了FL系统抵抗隐藏攻击的能力，而且为其未来的应用和发展提出了重要的考虑。



## **12. TransLinkGuard: Safeguarding Transformer Models Against Model Stealing in Edge Deployment**

TransLinkGuard：保护Transformer模型，防止边缘部署中的模型窃取 cs.CR

Accepted by ACM MM24 Conference

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2404.11121v2) [paper-pdf](http://arxiv.org/pdf/2404.11121v2)

**Authors**: Qinfeng Li, Zhiqiang Shen, Zhenghan Qin, Yangfan Xie, Xuhong Zhang, Tianyu Du, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) have been widely applied in various scenarios. Additionally, deploying LLMs on edge devices is trending for efficiency and privacy reasons. However, edge deployment of proprietary LLMs introduces new security challenges: edge-deployed models are exposed as white-box accessible to users, enabling adversaries to conduct effective model stealing (MS) attacks. Unfortunately, existing defense mechanisms fail to provide effective protection. Specifically, we identify four critical protection properties that existing methods fail to simultaneously satisfy: (1) maintaining protection after a model is physically copied; (2) authorizing model access at request level; (3) safeguarding runtime reverse engineering; (4) achieving high security with negligible runtime overhead. To address the above issues, we propose TransLinkGuard, a plug-and-play model protection approach against model stealing on edge devices. The core part of TransLinkGuard is a lightweight authorization module residing in a secure environment, e.g., TEE. The authorization module can freshly authorize each request based on its input. Extensive experiments show that TransLinkGuard achieves the same security protection as the black-box security guarantees with negligible overhead.

摘要: 专有的大型语言模型(LLM)已广泛应用于各种场景。此外，出于效率和隐私的原因，在边缘设备上部署LLM是一种趋势。然而，专有LLMS的边缘部署带来了新的安全挑战：边缘部署的模型暴露为用户可访问的白盒，使对手能够进行有效的模型窃取(MS)攻击。不幸的是，现有的防御机制未能提供有效的保护。具体地说，我们确定了现有方法无法同时满足的四个关键保护性质：(1)在物理复制模型后保持保护；(2)在请求级授权模型访问；(3)保护运行时逆向工程；(4)以可忽略的运行时开销实现高安全性。为了解决上述问题，我们提出了一种针对边缘设备上的模型窃取的即插即用模型保护方法TransLinkGuard。TransLinkGuard的核心部分是驻留在安全环境中的轻量级授权模块，例如TEE。授权模块可以基于其输入对每个请求进行新的授权。大量实验表明，TransLinkGuard实现了与黑盒安全保证相同的安全保护，而开销可以忽略不计。



## **13. Physical Adversarial Attack meets Computer Vision: A Decade Survey**

物理对抗攻击与计算机视觉：十年调查 cs.CV

Published at IEEE TPAMI. GitHub:https://github.com/weihui1308/PAA

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2209.15179v4) [paper-pdf](http://arxiv.org/pdf/2209.15179v4)

**Authors**: Hui Wei, Hao Tang, Xuemei Jia, Zhixiang Wang, Hanxun Yu, Zhubo Li, Shin'ichi Satoh, Luc Van Gool, Zheng Wang

**Abstract**: Despite the impressive achievements of Deep Neural Networks (DNNs) in computer vision, their vulnerability to adversarial attacks remains a critical concern. Extensive research has demonstrated that incorporating sophisticated perturbations into input images can lead to a catastrophic degradation in DNNs' performance. This perplexing phenomenon not only exists in the digital space but also in the physical world. Consequently, it becomes imperative to evaluate the security of DNNs-based systems to ensure their safe deployment in real-world scenarios, particularly in security-sensitive applications. To facilitate a profound understanding of this topic, this paper presents a comprehensive overview of physical adversarial attacks. Firstly, we distill four general steps for launching physical adversarial attacks. Building upon this foundation, we uncover the pervasive role of artifacts carrying adversarial perturbations in the physical world. These artifacts influence each step. To denote them, we introduce a new term: adversarial medium. Then, we take the first step to systematically evaluate the performance of physical adversarial attacks, taking the adversarial medium as a first attempt. Our proposed evaluation metric, hiPAA, comprises six perspectives: Effectiveness, Stealthiness, Robustness, Practicability, Aesthetics, and Economics. We also provide comparative results across task categories, together with insightful observations and suggestions for future research directions.

摘要: 尽管深度神经网络(DNN)在计算机视觉方面取得了令人印象深刻的成就，但它们对对手攻击的脆弱性仍然是一个令人担忧的问题。大量研究表明，在输入图像中加入复杂的扰动会导致DNN性能的灾难性下降。这种令人困惑的现象不仅存在于数字空间，也存在于物理世界。因此，迫切需要评估基于DNNS的系统的安全性，以确保它们在现实世界场景中的安全部署，特别是在安全敏感的应用中。为了促进对这一主题的深入理解，本文对物理对抗性攻击进行了全面的概述。首先，我们提炼出发动身体对抗攻击的四个一般步骤。在此基础上，我们揭示了在物理世界中携带对抗性扰动的人工制品的普遍作用。这些人工制品会影响每一步。为了表示它们，我们引入了一个新的术语：对抗性媒介。然后，以对抗性媒介为第一次尝试，对物理对抗性攻击的性能进行了系统的评估。我们提出的评估指标HIPAA包括六个角度：有效性、隐蔽性、健壮性、实用性、美观性和经济性。我们还提供了跨任务类别的比较结果，以及有洞察力的观察结果和对未来研究方向的建议。



## **14. A Survey on Adversarial Robustness of LiDAR-based Machine Learning Perception in Autonomous Vehicles**

自动驾驶车辆中基于LiDART的机器学习感知的对抗鲁棒性调查 cs.LG

20 pages, 2 figures

**SubmitDate**: 2024-11-21    [abs](http://arxiv.org/abs/2411.13778v1) [paper-pdf](http://arxiv.org/pdf/2411.13778v1)

**Authors**: Junae Kim, Amardeep Kaur

**Abstract**: In autonomous driving, the combination of AI and vehicular technology offers great potential. However, this amalgamation comes with vulnerabilities to adversarial attacks. This survey focuses on the intersection of Adversarial Machine Learning (AML) and autonomous systems, with a specific focus on LiDAR-based systems. We comprehensively explore the threat landscape, encompassing cyber-attacks on sensors and adversarial perturbations. Additionally, we investigate defensive strategies employed in countering these threats. This paper endeavors to present a concise overview of the challenges and advances in securing autonomous driving systems against adversarial threats, emphasizing the need for robust defenses to ensure safety and security.

摘要: 在自动驾驶方面，人工智能和车载技术的结合提供了巨大的潜力。然而，这种合并存在对抗攻击的脆弱性。这项调查的重点是对抗性机器学习（ML）和自治系统的交叉点，特别关注基于LiDART的系统。我们全面探索威胁格局，包括对传感器的网络攻击和对抗性扰动。此外，我们还调查了用于应对这些威胁的防御策略。本文试图简要概述保护自动驾驶系统免受对抗威胁的挑战和进展，强调需要强大的防御措施来确保安全。



## **15. WaterPark: A Robustness Assessment of Language Model Watermarking**

WaterPark：语言模型水印的稳健性评估 cs.CR

22 pages

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13425v1) [paper-pdf](http://arxiv.org/pdf/2411.13425v1)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: To mitigate the misuse of large language models (LLMs), such as disinformation, automated phishing, and academic cheating, there is a pressing need for the capability of identifying LLM-generated texts. Watermarking emerges as one promising solution: it plants statistical signals into LLMs' generative processes and subsequently verifies whether LLMs produce given texts. Various watermarking methods (``watermarkers'') have been proposed; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments?   To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. For instance, a watermarker's resilience to increasingly intensive attacks hinges on its context dependency. We further explore the best practices to operate watermarkers in adversarial environments. For instance, using a generic detector alongside a watermark-specific detector improves the security of vulnerable watermarkers. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.

摘要: 为了减少对大型语言模型(LLM)的滥用，如虚假信息、自动网络钓鱼和学术作弊，迫切需要识别LLM生成的文本的能力。数字水印作为一种很有前途的解决方案出现了：它将统计信号植入LLMS的生成过程中，随后验证LLMS是否生成给定的文本。人们已经提出了各种水印方法，然而，由于缺乏统一的评估平台，许多关键问题仍然没有得到充分的探讨：i)各种水印的优点/局限性是什么，特别是它们的攻击稳健性？Ii)各种设计选择对其健壮性有何影响？三)如何在对抗性环境中以最佳方式使用水印？为了填补这一空白，我们对现有的LLM水印和水印移除攻击进行了系统化，规划了它们的设计空间。然后我们开发了Water Park，这是一个统一的平台，集成了10个最先进的水印和12个具有代表性的攻击。更重要的是，利用水上公园，我们对现有的水印进行了全面的评估，揭示了各种设计选择对其攻击健壮性的影响。例如，水印对日益激烈的攻击的适应能力取决于它的上下文依赖性。我们进一步探索在对抗性环境中操作水印的最佳实践。例如，在水印专用检测器旁边使用通用检测器可以提高易受攻击的水印的安全性。我们相信我们的研究对当前的LLM数字水印技术有一定的启发作用，同时也为以后的研究提供了一个有价值的实验平台。



## **16. CopyrightMeter: Revisiting Copyright Protection in Text-to-image Models**

CopyrightMeter：重新审视文本到图像模型中的版权保护 cs.CR

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13144v1) [paper-pdf](http://arxiv.org/pdf/2411.13144v1)

**Authors**: Naen Xu, Changjiang Li, Tianyu Du, Minxi Li, Wenjie Luo, Jiacheng Liang, Yuyuan Li, Xuhong Zhang, Meng Han, Jianwei Yin, Ting Wang

**Abstract**: Text-to-image diffusion models have emerged as powerful tools for generating high-quality images from textual descriptions. However, their increasing popularity has raised significant copyright concerns, as these models can be misused to reproduce copyrighted content without authorization. In response, recent studies have proposed various copyright protection methods, including adversarial perturbation, concept erasure, and watermarking techniques. However, their effectiveness and robustness against advanced attacks remain largely unexplored. Moreover, the lack of unified evaluation frameworks has hindered systematic comparison and fair assessment of different approaches. To bridge this gap, we systematize existing copyright protection methods and attacks, providing a unified taxonomy of their design spaces. We then develop CopyrightMeter, a unified evaluation framework that incorporates 17 state-of-the-art protections and 16 representative attacks. Leveraging CopyrightMeter, we comprehensively evaluate protection methods across multiple dimensions, thereby uncovering how different design choices impact fidelity, efficacy, and resilience under attacks. Our analysis reveals several key findings: (i) most protections (16/17) are not resilient against attacks; (ii) the "best" protection varies depending on the target priority; (iii) more advanced attacks significantly promote the upgrading of protections. These insights provide concrete guidance for developing more robust protection methods, while its unified evaluation protocol establishes a standard benchmark for future copyright protection research in text-to-image generation.

摘要: 文本到图像扩散模型已经成为从文本描述生成高质量图像的强大工具。然而，它们越来越受欢迎也引发了严重的版权问题，因为这些模型可能被滥用来未经授权复制受版权保护的内容。对此，最近的研究提出了多种版权保护方法，包括对抗性扰动、概念删除和水印技术。然而，它们对高级攻击的有效性和健壮性在很大程度上仍有待研究。此外，缺乏统一的评价框架妨碍了对不同方法的系统比较和公平评估。为了弥补这一差距，我们对现有的版权保护方法和攻击进行了系统化，提供了它们的设计空间的统一分类。然后我们开发了CopyrightMeter，这是一个统一的评估框架，包含17种最先进的保护和16种典型的攻击。利用CopyrightMeter，我们跨多个维度全面评估保护方法，从而揭示不同的设计选择如何影响攻击下的保真度、有效性和弹性。我们的分析揭示了几个关键发现：(1)大多数保护(16/17)对攻击没有弹性；(2)“最佳”保护因目标优先而异；(3)更高级的攻击极大地促进了保护的升级。这些见解为开发更稳健的保护方法提供了具体指导，而其统一的评估协议为未来文本到图像生成中的版权保护研究建立了标准基准。



## **17. TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models**

TAPT：测试时对抗快速调整视觉语言模型中的鲁棒推理 cs.CV

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.13136v1) [paper-pdf](http://arxiv.org/pdf/2411.13136v1)

**Authors**: Xin Wang, Kai Chen, Jiaming Zhang, Jingjing Chen, Xingjun Ma

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as CLIP have demonstrated excellent zero-shot generalizability across various downstream tasks. However, recent studies have shown that the inference performance of CLIP can be greatly degraded by small adversarial perturbations, especially its visual modality, posing significant safety threats. To mitigate this vulnerability, in this paper, we propose a novel defense method called Test-Time Adversarial Prompt Tuning (TAPT) to enhance the inference robustness of CLIP against visual adversarial attacks. TAPT is a test-time defense method that learns defensive bimodal (textual and visual) prompts to robustify the inference process of CLIP. Specifically, it is an unsupervised method that optimizes the defensive prompts for each test sample by minimizing a multi-view entropy and aligning adversarial-clean distributions. We evaluate the effectiveness of TAPT on 11 benchmark datasets, including ImageNet and 10 other zero-shot datasets, demonstrating that it enhances the zero-shot adversarial robustness of the original CLIP by at least 48.9% against AutoAttack (AA), while largely maintaining performance on clean examples. Moreover, TAPT outperforms existing adversarial prompt tuning methods across various backbones, achieving an average robustness improvement of at least 36.6%.

摘要: 大型预先训练的视觉语言模型(VLM)，如CLIP，已经在各种下游任务中表现出出色的零射击泛化能力。然而，最近的研究表明，CLIP的推理性能会因小的对抗性扰动而大大降低，特别是它的视觉通道，构成了严重的安全威胁。为了缓解这一漏洞，本文提出了一种新的防御方法，称为测试时间对抗性提示调整(TAPT)，以增强CLIP对视觉对抗性攻击的推理健壮性。TAPT是一种测试时防御方法，它学习防御性双峰(文本和视觉)提示，以巩固CLIP的推理过程。具体地说，它是一种无监督的方法，通过最小化多视图熵和对齐对抗性干净的分布来优化每个测试样本的防御提示。我们在11个基准数据集上对TAPT的有效性进行了评估，包括ImageNet和其他10个零镜头数据集，结果表明，它在很大程度上保持了在干净样本上的性能，但相对于AutoAttack(AA)，它至少提高了原始剪辑的零镜头对抗健壮性48.9%。此外，TAPT在不同主干上的性能优于现有的对抗性提示调优方法，实现了平均至少36.6%的健壮性改进。



## **18. Disco Intelligent Omni-Surfaces: 360-degree Fully-Passive Jamming Attacks**

迪斯科智能全方位：360度全被动干扰攻击 eess.SP

This paper has been submitted to IEEE TWC for possible publication

**SubmitDate**: 2024-11-20    [abs](http://arxiv.org/abs/2411.12985v1) [paper-pdf](http://arxiv.org/pdf/2411.12985v1)

**Authors**: Huan Huang, Hongliang Zhang, Jide Yuan, Luyao Sun, Yitian Wang, Weidong Mei, Boya Di, Yi Cai, Zhu Han

**Abstract**: Intelligent omni-surfaces (IOSs) with 360-degree electromagnetic radiation significantly improves the performance of wireless systems, while an adversarial IOS also poses a significant potential risk for physical layer security. In this paper, we propose a "DISCO" IOS (DIOS) based fully-passive jammer (FPJ) that can launch omnidirectional fully-passive jamming attacks. In the proposed DIOS-based FPJ, the interrelated refractive and reflective (R&R) coefficients of the adversarial IOS are randomly generated, acting like a "DISCO" that distributes wireless energy radiated by the base station. By introducing active channel aging (ACA) during channel coherence time, the DIOS-based FPJ can perform omnidirectional fully-passive jamming without neither jamming power nor channel knowledge of legitimate users (LUs). To characterize the impact of the DIOS-based PFJ, we derive the statistical characteristics of DIOS-jammed channels based on two widely-used IOS models, i.e., the constant-amplitude model and the variable-amplitude model. Consequently, the asymptotic analysis of the ergodic achievable sum rates under the DIOS-based omnidirectional fully-passive jamming is given based on the derived stochastic characteristics for both the two IOS models. Based on the derived analysis, the omnidirectional jamming impact of the proposed DIOS-based FPJ implemented by a constant-amplitude IOS does not depend on either the quantization number or the stochastic distribution of the DIOS coefficients, while the conclusion does not hold on when a variable-amplitude IOS is used. Numerical results based on one-bit quantization of the IOS phase shifts are provided to verify the effectiveness of the derived theoretical analysis. The proposed DIOS-based FPJ can not only launch omnidirectional fully-passive jamming, but also improve the jamming impact by about 55% at 10 dBm transmit power per LU.

摘要: 具有360度电磁辐射的智能全表面(IOSS)显著提高了无线系统的性能，而敌意的IOS也对物理层安全构成了重大的潜在风险。本文提出了一种基于“迪斯科”IOS(DIOS)的全无源干扰机(FPJ)，它可以发起全方位的全无源干扰攻击。在拟议的基于DIOS的FPJ中，敌方IOS的相互关联的折射和反射(R&R)系数是随机生成的，其作用类似于分发基站辐射的无线能量的“迪斯科舞厅”。通过在信道相干时间引入主动信道老化(ACA)，基于DIOS的FPJ可以在没有干扰功率和合法用户(LU)的信道知识的情况下执行全方位全无源干扰。为了刻画基于DIOS的调频干扰的影响，我们基于两种广泛使用的IOS模型，即恒幅模型和变幅模型，推导出了DIOS干扰信道的统计特性。基于推导出的两种IOS模型的随机特性，给出了基于Dios的全向全无源干扰下遍历可达和速率的渐近分析。在此基础上，分析了采用恒幅IOS实现的基于DIOS的FPJ的全向干扰效果不依赖于量化次数或DIOS系数的随机分布，而当采用变幅IOS时，这一结论不成立。给出了基于IOS相移一位量化的数值结果，验证了理论分析的有效性。所提出的基于DIOS的FPJ不仅可以实现全方位的全无源干扰，而且在每逻辑单元发射功率为10dBm时，干扰效果可提高约55%。



## **19. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的有效模型窃取攻击 cs.LG

Accepted at ECAI - 27th European Conference on Artificial  Intelligence

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2405.12295v4) [paper-pdf](http://arxiv.org/pdf/2405.12295v4)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska, Tomasz P. Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which allow for the processing of graph-structured data without relying on predefined graph structures, are becoming increasingly important in a wide range of applications. As such these networks become attractive targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. Significant efforts have been devoted to developing model-stealing attacks that extract models trained on images and texts. However, little attention has been given to stealing GNNs trained on graph data. This paper identifies a new method of performing unsupervised model-stealing attacks against inductive GNNs, utilizing graph contrastive learning and spectral graph augmentations to efficiently extract information from the targeted model. The new type of attack is thoroughly evaluated on six datasets and the results show that our approach outperforms the current state-of-the-art by Shen et al. (2021). In particular, our attack surpasses the baseline across all benchmarks, attaining superior fidelity and downstream accuracy of the stolen model while necessitating fewer queries directed toward the target model.

摘要: 图神经网络(GNN)被认为是处理以图结构组织的真实世界数据的有力工具。尤其是允许在不依赖预定义的图结构的情况下处理图结构数据的感应式GNN，在广泛的应用中正变得越来越重要。因此，这些网络成为模型窃取攻击的有吸引力的目标，在这种攻击中，对手试图复制目标网络的功能。已经投入了大量的努力来开发窃取模型的攻击，提取针对图像和文本训练的模型。然而，对窃取针对图表数据训练的GNN的关注很少。本文提出了一种新的无监督窃取模型攻击方法，利用图对比学习和谱图扩充来有效地从目标模型中提取信息。在六个数据集上对新类型的攻击进行了全面的评估，结果表明，我们的方法的性能优于沈等人目前的最新技术。(2021年)。特别是，我们的攻击在所有基准测试中都超过了基线，获得了被盗模型的卓越保真度和下游准确性，同时需要更少的针对目标模型的查询。



## **20. Attribute Inference Attacks for Federated Regression Tasks**

针对联邦回归任务的属性推理攻击 cs.LG

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12697v1) [paper-pdf](http://arxiv.org/pdf/2411.12697v1)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.

摘要: 联合学习(FL)使多个客户端(如移动电话和物联网设备)能够协作训练全球机器学习模型，同时保持其数据的本地化。然而，最近的研究表明，FL的训练阶段容易受到重构攻击，如属性推理攻击(AIA)，即攻击者利用交换的消息和辅助公共信息来发现目标客户的敏感属性。虽然这些攻击已经在分类任务的背景下进行了广泛的研究，但它们对回归任务的影响在很大程度上仍未被探索。在本文中，我们通过提出专门为FL环境中的回归任务设计的新的基于模型的AIAS来解决这一差距。我们的方法考虑了攻击者可以窃听交换的消息或直接干扰训练过程的场景。我们使用真实世界的数据集，根据最先进的方法对我们提出的攻击进行基准测试。结果表明，重建精度显著提高，特别是在异类客户端数据集，这是FL中的常见场景。我们基于模型的AIAS的有效性使它们更适合于经验性地量化联合回归任务的隐私泄露。



## **21. Stochastic BIQA: Median Randomized Smoothing for Certified Blind Image Quality Assessment**

随机BIQA：用于认证盲图像质量评估的随机中位数平滑 eess.IV

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12575v1) [paper-pdf](http://arxiv.org/pdf/2411.12575v1)

**Authors**: Ekaterina Shumitskaya, Mikhail Pautov, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Most modern No-Reference Image-Quality Assessment (NR-IQA) metrics are based on neural networks vulnerable to adversarial attacks. Attacks on such metrics lead to incorrect image/video quality predictions, which poses significant risks, especially in public benchmarks. Developers of image processing algorithms may unfairly increase the score of a target IQA metric without improving the actual quality of the adversarial image. Although some empirical defenses for IQA metrics were proposed, they do not provide theoretical guarantees and may be vulnerable to adaptive attacks. This work focuses on developing a provably robust no-reference IQA metric. Our method is based on Median Smoothing (MS) combined with an additional convolution denoiser with ranking loss to improve the SROCC and PLCC scores of the defended IQA metric. Compared with two prior methods on three datasets, our method exhibited superior SROCC and PLCC scores while maintaining comparable certified guarantees.

摘要: 大多数现代无参考图像质量评估（NR-IQA）指标都基于容易受到对抗攻击的神经网络。对此类指标的攻击会导致图像/视频质量预测错误，从而带来重大风险，尤其是在公共基准中。图像处理算法的开发人员可能会不公平地增加目标IQA指标的分数，而不提高对抗图像的实际质量。尽管提出了一些针对IQA指标的经验防御措施，但它们并不提供理论保证，并且可能容易受到自适应攻击。这项工作的重点是开发一个可证明稳健的无参考IQA指标。我们的方法基于中位数平滑（MS），结合具有排名损失的额外卷积去噪器，以提高受保护的IQA指标的SROCC和PLCC分数。与三个数据集上的两种先前方法相比，我们的方法表现出更好的SROCC和PLCC评分，同时保持了相当的认证保证。



## **22. Variational Bayesian Bow tie Neural Networks with Shrinkage**

具有收缩性的变分Bayesian领结神经网络 stat.ML

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.11132v2) [paper-pdf](http://arxiv.org/pdf/2411.11132v2)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by constructing a relaxed version of the standard feed-forward rectified neural network, and employing Polya-Gamma data augmentation tricks to render a conditionally linear and Gaussian model. Additionally, we use sparsity-promoting priors on the weights of the neural network for data-driven architectural design. To approximate the posterior, we derive a variational inference algorithm that avoids distributional assumptions and independence across layers and is a faster alternative to the usual Markov Chain Monte Carlo schemes.

摘要: 尽管深度模型在机器学习中起着主导作用，但局限性依然存在，包括过度自信的预测、对对抗性攻击的敏感性以及对预测中的可变性的低估。贝叶斯范式为克服这些问题提供了一个自然的框架，并已成为深度模型不确定性估计的黄金标准，还提供了改进的精度和调整关键超参数的框架。然而，准确的贝叶斯推理是具有挑战性的，通常涉及施加强独立性和分布假设的变分算法。此外，现有的方法对网络的架构选择很敏感。我们通过构造一个松弛版本的标准前馈校正神经网络来解决这些问题，并使用Polya-Gamma数据增强技巧来呈现条件线性和高斯模型。此外，对于数据驱动的建筑设计，我们在神经网络的权值上使用了稀疏性提升的先验。为了逼近后验概率，我们推导了一种变分推理算法，它避免了分布假设和层间独立性，是通常的马尔可夫链蒙特卡罗格式的一个更快的替代方案。



## **23. NMT-Obfuscator Attack: Ignore a sentence in translation with only one word**

NMT-Obfuscator攻击：忽略翻译中只有一个单词的句子 cs.CL

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12473v1) [paper-pdf](http://arxiv.org/pdf/2411.12473v1)

**Authors**: Sahar Sadrizadeh, César Descalzo, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Neural Machine Translation systems are used in diverse applications due to their impressive performance. However, recent studies have shown that these systems are vulnerable to carefully crafted small perturbations to their inputs, known as adversarial attacks. In this paper, we propose a new type of adversarial attack against NMT models. In this attack, we find a word to be added between two sentences such that the second sentence is ignored and not translated by the NMT model. The word added between the two sentences is such that the whole adversarial text is natural in the source language. This type of attack can be harmful in practical scenarios since the attacker can hide malicious information in the automatic translation made by the target NMT model. Our experiments show that different NMT models and translation tasks are vulnerable to this type of attack. Our attack can successfully force the NMT models to ignore the second part of the input in the translation for more than 50% of all cases while being able to maintain low perplexity for the whole input.

摘要: 神经机器翻译系统因其令人印象深刻的性能而被广泛应用。然而，最近的研究表明，这些系统很容易受到精心设计的对其输入的微小扰动，即所谓的对抗性攻击。本文提出了一种新型的针对NMT模型的对抗性攻击。在这种攻击中，我们发现在两个句子之间添加了一个单词，使得第二个句子被忽略，并且不被NMT模型翻译。在两个句子之间添加的单词是这样的，即整个对抗性文本在源语言中是自然的。这种类型的攻击在实际情况下可能是有害的，因为攻击者可以在目标NMT模型所做的自动翻译中隐藏恶意信息。我们的实验表明，不同的NMT模型和翻译任务都容易受到此类攻击。我们的攻击可以成功地迫使NMT模型在超过50%的情况下忽略翻译中输入的第二部分，同时能够保持整个输入的低困惑。



## **24. Efficient Verifiable Differential Privacy with Input Authenticity in the Local and Shuffle Model**

本地和洗牌模型中具有输入真实性的高效可验证差异隐私 cs.CR

21 pages, 13 figures, 2 tables; accepted for publication in the  Proceedings on the 25th Privacy Enhancing Technologies Symposium (PoPETs)  2025

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2406.18940v2) [paper-pdf](http://arxiv.org/pdf/2406.18940v2)

**Authors**: Tariq Bontekoe, Hassan Jameel Asghar, Fatih Turkmen

**Abstract**: Local differential privacy (LDP) enables the efficient release of aggregate statistics without having to trust the central server (aggregator), as in the central model of differential privacy, and simultaneously protects a client's sensitive data. The shuffle model with LDP provides an additional layer of privacy, by disconnecting the link between clients and the aggregator. However, LDP has been shown to be vulnerable to malicious clients who can perform both input and output manipulation attacks, i.e., before and after applying the LDP mechanism, to skew the aggregator's results. In this work, we show how to prevent malicious clients from compromising LDP schemes. Our only realistic assumption is that the initial raw input is authenticated; the rest of the processing pipeline, e.g., formatting the input and applying the LDP mechanism, may be under adversarial control. We give several real-world examples where this assumption is justified. Our proposed schemes for verifiable LDP (VLDP), prevent both input and output manipulation attacks against generic LDP mechanisms, requiring only one-time interaction between client and server, unlike existing alternatives [37, 43]. Most importantly, we are the first to provide an efficient scheme for VLDP in the shuffle model. We describe, and prove security of, two schemes for VLDP in the local model, and one in the shuffle model. We show that all schemes are highly practical, with client run times of less than 2 seconds, and server run times of 5-7 milliseconds per client.

摘要: 本地差异隐私(LDP)支持高效发布汇总统计数据，而不必像差异隐私的中央模型那样信任中央服务器(聚合器)，同时保护客户端的敏感数据。带有LDP的随机模式通过断开客户端和聚合器之间的链路，提供了额外的保密层。然而，LDP已被证明容易受到恶意客户端的攻击，这些客户端可以执行输入和输出操纵攻击，即在应用LDP机制之前和之后，以歪曲聚合器的结果。在这项工作中，我们展示了如何防止恶意客户端危害LDP方案。我们唯一现实的假设是初始原始输入是经过认证的；处理流水线的其余部分，例如格式化输入和应用LDP机制，可能处于敌对控制之下。我们给出了几个真实世界的例子，证明这一假设是合理的。我们提出的可验证LDP方案(VLDP)，防止了针对通用LDP机制的输入和输出操纵攻击，与现有的替代方案不同，它只需要客户端和服务器之间的一次性交互。最重要的是，我们首次在混洗模型中为VLDP提供了一种有效的方案。我们描述并证明了VLDP在局部模型下的两个方案和在置乱模型下的一个方案的安全性。我们证明了所有方案都是非常实用的，客户端运行时间不到2秒，每个客户端的服务器运行时间为5-7毫秒。



## **25. DeTrigger: A Gradient-Centric Approach to Backdoor Attack Mitigation in Federated Learning**

DeTrigger：联邦学习中以用户为中心的后门攻击缓解方法 cs.LG

14 pages

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12220v1) [paper-pdf](http://arxiv.org/pdf/2411.12220v1)

**Authors**: Kichang Lee, Yujin Shin, Jonghyuk Yun, Jun Han, JeongGil Ko

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed devices while preserving local data privacy, making it ideal for mobile and embedded systems. However, the decentralized nature of FL also opens vulnerabilities to model poisoning attacks, particularly backdoor attacks, where adversaries implant trigger patterns to manipulate model predictions. In this paper, we propose DeTrigger, a scalable and efficient backdoor-robust federated learning framework that leverages insights from adversarial attack methodologies. By employing gradient analysis with temperature scaling, DeTrigger detects and isolates backdoor triggers, allowing for precise model weight pruning of backdoor activations without sacrificing benign model knowledge. Extensive evaluations across four widely used datasets demonstrate that DeTrigger achieves up to 251x faster detection than traditional methods and mitigates backdoor attacks by up to 98.9%, with minimal impact on global model accuracy. Our findings establish DeTrigger as a robust and scalable solution to protect federated learning environments against sophisticated backdoor threats.

摘要: 联合学习(FL)支持跨分布式设备进行协作模型培训，同时保护本地数据隐私，使其成为移动和嵌入式系统的理想选择。然而，FL的分散性也为建模中毒攻击打开了漏洞，特别是后门攻击，对手植入触发模式来操纵模型预测。在本文中，我们提出了DeTrigger，一个可扩展的高效后门健壮的联邦学习框架，它利用了对手攻击方法的见解。通过使用带有温度缩放的梯度分析，DeTrigger检测并隔离后门触发器，从而在不牺牲良性模型知识的情况下精确削减后门激活的模型权重。对四个广泛使用的数据集的广泛评估表明，DeTrigger的检测速度比传统方法快251倍，后门攻击减少高达98.9%，对全局模型精度的影响最小。我们的发现将DeTrigger确立为一个强大且可扩展的解决方案，可以保护联合学习环境免受复杂的后门威胁。



## **26. Architectural Patterns for Designing Quantum Artificial Intelligence Systems**

设计量子人工智能系统的架构模式 cs.SE

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.10487v2) [paper-pdf](http://arxiv.org/pdf/2411.10487v2)

**Authors**: Mykhailo Klymenko, Thong Hoang, Xiwei Xu, Zhenchang Xing, Muhammad Usman, Qinghua Lu, Liming Zhu

**Abstract**: Utilising quantum computing technology to enhance artificial intelligence systems is expected to improve training and inference times, increase robustness against noise and adversarial attacks, and reduce the number of parameters without compromising accuracy. However, moving beyond proof-of-concept or simulations to develop practical applications of these systems while ensuring high software quality faces significant challenges due to the limitations of quantum hardware and the underdeveloped knowledge base in software engineering for such systems. In this work, we have conducted a systematic mapping study to identify the challenges and solutions associated with the software architecture of quantum-enhanced artificial intelligence systems. Our review uncovered several architectural patterns that describe how quantum components can be integrated into inference engines, as well as middleware patterns that facilitate communication between classical and quantum components. These insights have been compiled into a catalog of architectural patterns. Each pattern realises a trade-off between efficiency, scalability, trainability, simplicity, portability and deployability, and other software quality attributes.

摘要: 利用量子计算技术来增强人工智能系统，预计将改善训练和推理时间，提高对噪音和对手攻击的稳健性，并在不影响准确性的情况下减少参数数量。然而，由于量子硬件的限制和此类系统的软件工程知识库的不发达，超越概念验证或模拟来开发这些系统的实际应用，同时确保高软件质量面临着重大挑战。在这项工作中，我们进行了系统的映射研究，以确定与量子增强型人工智能系统的软件体系结构相关的挑战和解决方案。我们的审查揭示了几种描述量子组件如何集成到推理引擎中的体系结构模式，以及促进经典组件和量子组件之间通信的中间件模式。这些见解已被汇编成体系结构模式的目录。每个模式都实现了效率、可伸缩性、可训练性、简单性、可移植性和可部署性以及其他软件质量属性之间的权衡。



## **27. Adversarial Multi-Agent Reinforcement Learning for Proactive False Data Injection Detection**

用于主动错误数据注入检测的对抗性多智能体强化学习 eess.SY

**SubmitDate**: 2024-11-19    [abs](http://arxiv.org/abs/2411.12130v1) [paper-pdf](http://arxiv.org/pdf/2411.12130v1)

**Authors**: Kejun Chen, Truc Nguyen, Malik Hassanaly

**Abstract**: Smart inverters are instrumental in the integration of renewable and distributed energy resources (DERs) into the electric grid. Such inverters rely on communication layers for continuous control and monitoring, potentially exposing them to cyber-physical attacks such as false data injection attacks (FDIAs). We propose to construct a defense strategy against a priori unknown FDIAs with a multi-agent reinforcement learning (MARL) framework. The first agent is an adversary that simulates and discovers various FDIA strategies, while the second agent is a defender in charge of detecting and localizing FDIAs. This approach enables the defender to be trained against new FDIAs continuously generated by the adversary. The numerical results demonstrate that the proposed MARL defender outperforms a supervised offline defender. Additionally, we show that the detection skills of an MARL defender can be combined with that of an offline defender through a transfer learning approach.

摘要: 智能逆变器对于将可再生能源和分布式能源（BER）集成到电网中至关重要。此类逆变器依赖通信层进行持续控制和监控，可能会使它们面临虚假数据注入攻击（FDIA）等网络物理攻击。我们建议通过多智能体强化学习（MARL）框架构建针对先验未知FDIA的防御策略。第一个代理是模拟和发现各种FDIA策略的对手，而第二个代理是负责检测和定位FDIA的防御者。这种方法使防御者能够针对对手不断产生的新FDIA进行训练。数值结果表明，提出的MARL防御器优于有监督的离线防御器。此外，我们还表明，MARL防守者的检测技能可以通过迁移学习方法与离线防守者的检测技能相结合。



## **28. Theoretical Corrections and the Leveraging of Reinforcement Learning to Enhance Triangle Attack**

理论修正和利用强化学习增强三角攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12071v1) [paper-pdf](http://arxiv.org/pdf/2411.12071v1)

**Authors**: Nicole Meng, Caleb Manicke, David Chen, Yingjie Lao, Caiwen Ding, Pengyu Hong, Kaleel Mahmood

**Abstract**: Adversarial examples represent a serious issue for the application of machine learning models in many sensitive domains. For generating adversarial examples, decision based black-box attacks are one of the most practical techniques as they only require query access to the model. One of the most recently proposed state-of-the-art decision based black-box attacks is Triangle Attack (TA). In this paper, we offer a high-level description of TA and explain potential theoretical limitations. We then propose a new decision based black-box attack, Triangle Attack with Reinforcement Learning (TARL). Our new attack addresses the limits of TA by leveraging reinforcement learning. This creates an attack that can achieve similar, if not better, attack accuracy than TA with half as many queries on state-of-the-art classifiers and defenses across ImageNet and CIFAR-10.

摘要: 对抗性示例代表了机器学习模型在许多敏感领域的应用的一个严重问题。对于生成对抗性示例，基于决策的黑匣子攻击是最实用的技术之一，因为它们只需要对模型进行查询访问。最近提出的最先进的基于决策的黑匣子攻击之一是三角攻击（TA）。在本文中，我们对TA进行了高级描述并解释了潜在的理论局限性。然后，我们提出了一种新的基于决策的黑匣子攻击，即带强化学习的三角攻击（TARL）。我们的新攻击通过利用强化学习来解决TA的局限性。这会创建一种攻击，它可以实现与TA类似（甚至更好）的攻击准确性，只需对ImageNet和CIFAR-10中最先进的分类器和防御系统进行一半的查询。



## **29. Exploring adversarial robustness of JPEG AI: methodology, comparison and new methods**

探索JPEG AI的对抗鲁棒性：方法论、比较和新方法 eess.IV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11795v1) [paper-pdf](http://arxiv.org/pdf/2411.11795v1)

**Authors**: Egor Kovalev, Georgii Bychkov, Khaled Abud, Aleksandr Gushchin, Anna Chistyakova, Sergey Lavrushkin, Dmitriy Vatolin, Anastasia Antsiferova

**Abstract**: Adversarial robustness of neural networks is an increasingly important area of research, combining studies on computer vision models, large language models (LLMs), and others. With the release of JPEG AI - the first standard for end-to-end neural image compression (NIC) methods - the question of its robustness has become critically significant. JPEG AI is among the first international, real-world applications of neural-network-based models to be embedded in consumer devices. However, research on NIC robustness has been limited to open-source codecs and a narrow range of attacks. This paper proposes a new methodology for measuring NIC robustness to adversarial attacks. We present the first large-scale evaluation of JPEG AI's robustness, comparing it with other NIC models. Our evaluation results and code are publicly available online (link is hidden for a blind review).

摘要: 神经网络的对抗鲁棒性是一个越来越重要的研究领域，结合了对计算机视觉模型、大型语言模型（LLM）等的研究。随着JPEG AI（端到端神经图像压缩（NIC）方法的第一个标准）的发布，其稳健性问题变得至关重要。JPEG AI是首批嵌入消费设备的基于神经网络的模型的国际现实应用之一。然而，关于NIC稳健性的研究仅限于开源编解码器和范围狭窄的攻击。本文提出了一种新的方法来衡量NIC对对抗性攻击的稳健性。我们首次对JPEG AI的稳健性进行了大规模评估，并将其与其他NIC模型进行了比较。我们的评估结果和代码可在线公开（链接已隐藏，以供盲目审查）。



## **30. Robust Subgraph Learning by Monitoring Early Training Representations**

通过监控早期训练表示进行稳健的子图学习 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2403.09901v2) [paper-pdf](http://arxiv.org/pdf/2403.09901v2)

**Authors**: Sepideh Neshatfar, Salimeh Yasaei Sekeh

**Abstract**: Graph neural networks (GNNs) have attracted significant attention for their outstanding performance in graph learning and node classification tasks. However, their vulnerability to adversarial attacks, particularly through susceptible nodes, poses a challenge in decision-making. The need for robust graph summarization is evident in adversarial challenges resulting from the propagation of attacks throughout the entire graph. In this paper, we address both performance and adversarial robustness in graph input by introducing the novel technique SHERD (Subgraph Learning Hale through Early Training Representation Distances). SHERD leverages information from layers of a partially trained graph convolutional network (GCN) to detect susceptible nodes during adversarial attacks using standard distance metrics. The method identifies "vulnerable (bad)" nodes and removes such nodes to form a robust subgraph while maintaining node classification performance. Through our experiments, we demonstrate the increased performance of SHERD in enhancing robustness by comparing the network's performance on original and subgraph inputs against various baselines alongside existing adversarial attacks. Our experiments across multiple datasets, including citation datasets such as Cora, Citeseer, and Pubmed, as well as microanatomical tissue structures of cell graphs in the placenta, highlight that SHERD not only achieves substantial improvement in robust performance but also outperforms several baselines in terms of node classification accuracy and computational complexity.

摘要: 图神经网络(GNN)因其在图学习和节点分类任务中的优异性能而备受关注。然而，它们对敌意攻击的脆弱性，特别是通过易受攻击的节点，对决策构成了挑战。在攻击在整个图中传播所导致的对抗性挑战中，对健壮图摘要的需求是显而易见的。在本文中，我们通过引入新的技术SHERD(子图通过早期训练表示距离学习Hale)来解决图形输入中的性能和对手健壮性。SHERD利用来自部分训练的图卷积网络(GCN)各层的信息，使用标准距离度量在敌意攻击期间检测易受攻击的节点。该方法在保持节点分类性能的同时，识别“易受攻击(坏)”的节点，并删除这些节点以形成一个健壮的子图。通过我们的实验，我们通过比较网络在原始和子图输入上的性能与不同基线的性能以及现有的对抗性攻击，证明了SHERD在增强健壮性方面的性能提高。我们在多个数据集上的实验，包括引用数据集，如Cora，Citeseer和Pubmed，以及胎盘细胞图的显微解剖组织结构，突出了Sherd不仅在健壮性性能方面取得了实质性的改进，而且在节点分类精度和计算复杂性方面也超过了几个基线。



## **31. Eidos: Efficient, Imperceptible Adversarial 3D Point Clouds**

Eidos：高效、不可感知的对抗性3D点云 cs.CV

Preprint

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2405.14210v2) [paper-pdf](http://arxiv.org/pdf/2405.14210v2)

**Authors**: Hanwei Zhang, Luo Cheng, Qisong He, Wei Huang, Renjue Li, Ronan Sicre, Xiaowei Huang, Holger Hermanns, Lijun Zhang

**Abstract**: Classification of 3D point clouds is a challenging machine learning (ML) task with important real-world applications in a spectrum from autonomous driving and robot-assisted surgery to earth observation from low orbit. As with other ML tasks, classification models are notoriously brittle in the presence of adversarial attacks. These are rooted in imperceptible changes to inputs with the effect that a seemingly well-trained model ends up misclassifying the input. This paper adds to the understanding of adversarial attacks by presenting Eidos, a framework providing Efficient Imperceptible aDversarial attacks on 3D pOint cloudS. Eidos supports a diverse set of imperceptibility metrics. It employs an iterative, two-step procedure to identify optimal adversarial examples, thereby enabling a runtime-imperceptibility trade-off. We provide empirical evidence relative to several popular 3D point cloud classification models and several established 3D attack methods, showing Eidos' superiority with respect to efficiency as well as imperceptibility.

摘要: 三维点云的分类是一项具有挑战性的机器学习(ML)任务，在从自动驾驶和机器人辅助手术到低轨道对地观测等一系列实际应用中具有重要的应用。与其他ML任务一样，分类模型在存在对抗性攻击时是出了名的脆弱。这些问题根源于对投入的潜移默化的改变，其结果是，一个看似训练有素的模型最终会错误地对投入进行分类。本文通过介绍EIDOS来加深对敌意攻击的理解，EIDOS是一种在3D点云上提供高效的隐形攻击的框架。Eidos支持一组不同的不可感知性指标。它使用迭代的两步过程来确定最佳对抗性示例，从而实现了运行时不可感知性的权衡。我们提供了与几种流行的三维点云分类模型和几种已建立的三维攻击方法相关的经验证据，表明了Eidos在效率和不可感知性方面的优势。



## **32. Bitcoin Under Volatile Block Rewards: How Mempool Statistics Can Influence Bitcoin Mining**

波动性区块奖励下的比特币：Mempool统计数据如何影响比特币采矿 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11702v1) [paper-pdf](http://arxiv.org/pdf/2411.11702v1)

**Authors**: Roozbeh Sarenche, Alireza Aghabagherloo, Svetla Nikova, Bart Preneel

**Abstract**: As Bitcoin experiences more halving events, the protocol reward converges to zero, making transaction fees the primary source of miner rewards. This shift in Bitcoin's incentivization mechanism, which introduces volatility into block rewards, could lead to the emergence of new security threats or intensify existing ones. Previous security analyses of Bitcoin have either considered a fixed block reward model or a highly simplified volatile model, overlooking the complexities of Bitcoin's mempool behavior.   In this paper, we present a reinforcement learning-based tool designed to analyze mining strategies under a more realistic volatile model. Our tool uses the Asynchronous Advantage Actor-Critic (A3C) algorithm to derive near-optimal mining strategies while interacting with an environment that models the complexity of the Bitcoin mempool. This tool enables the analysis of adversarial mining strategies, such as selfish mining and undercutting, both before and after difficulty adjustments, providing insights into the effects of mining attacks in both the short and long term.   Our analysis reveals that Bitcoin users' trend of offering higher fees to speed up the inclusion of their transactions in the chain can incentivize payoff-maximizing miners to deviate from the honest strategy. In the fixed reward model, a disincentive for the selfish mining attack is the initial loss period of at least two weeks, during which the attack is not profitable. However, our analysis shows that once the protocol reward diminishes to zero in the future, or even currently on days when transaction fees are comparable to the protocol reward, mining pools might be incentivized to abandon honest mining to gain an immediate profit.

摘要: 随着比特币经历更多减半事件，协议奖励趋于零，使交易费成为矿工奖励的主要来源。比特币激励机制的这种转变，在大宗奖励中引入了波动性，可能会导致新的安全威胁的出现，或者加剧现有的安全威胁。此前对比特币的安全分析要么考虑了固定的区块奖励模型，要么考虑了高度简化的波动性模型，忽视了比特币成员池行为的复杂性。在本文中，我们提出了一个基于强化学习的工具，用于在更真实的易变模型下分析挖掘策略。我们的工具使用异步优势参与者-批评者(A3C)算法来推导出接近最优的挖掘策略，同时与模拟比特币记忆池复杂性的环境交互。这一工具能够分析难度调整前后的对抗性采矿战略，如自私采矿和削价，从而深入了解采矿攻击在短期和长期的影响。我们的分析显示，比特币用户提供更高费用以加快将他们的交易纳入链中的趋势，可以激励收益最大化的矿工偏离诚实策略。在固定报酬模型中，对自私挖矿攻击的抑制是至少两周的初始损失期，在此期间攻击是不盈利的。然而，我们的分析表明，一旦协议奖励在未来减少到零，甚至目前在交易费与协议奖励相当的日子里，采矿池可能会受到激励，放弃诚实的开采，以获得直接利润。



## **33. TrojanRobot: Backdoor Attacks Against Robotic Manipulation in the Physical World**

特洛伊机器人：针对物理世界中机器人操纵的后门攻击 cs.RO

Initial version with preliminary results. We welcome any feedback or  suggestions

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11683v1) [paper-pdf](http://arxiv.org/pdf/2411.11683v1)

**Authors**: Xianlong Wang, Hewen Pan, Hangtao Zhang, Minghui Li, Shengshan Hu, Ziqi Zhou, Lulu Xue, Peijin Guo, Yichen Wang, Wei Wan, Aishan Liu, Leo Yu Zhang

**Abstract**: Robotic manipulation refers to the autonomous handling and interaction of robots with objects using advanced techniques in robotics and artificial intelligence. The advent of powerful tools such as large language models (LLMs) and large vision-language models (LVLMs) has significantly enhanced the capabilities of these robots in environmental perception and decision-making. However, the introduction of these intelligent agents has led to security threats such as jailbreak attacks and adversarial attacks.   In this research, we take a further step by proposing a backdoor attack specifically targeting robotic manipulation and, for the first time, implementing backdoor attack in the physical world. By embedding a backdoor visual language model into the visual perception module within the robotic system, we successfully mislead the robotic arm's operation in the physical world, given the presence of common items as triggers. Experimental evaluations in the physical world demonstrate the effectiveness of the proposed backdoor attack.

摘要: 机器人操纵是指使用机器人学和人工智能的先进技术，自主处理机器人与物体的交互。大型语言模型(LLM)和大型视觉语言模型(LVLM)等强大工具的出现，大大增强了这些机器人在环境感知和决策方面的能力。然而，这些智能代理的引入导致了越狱攻击和对抗性攻击等安全威胁。在这项研究中，我们进一步提出了专门针对机器人操作的后门攻击，并首次在物理世界中实现了后门攻击。通过将后门视觉语言模型嵌入机器人系统的视觉感知模块中，我们成功地误导了机械臂在物理世界中的操作，因为存在共同的物品作为触发器。物理世界中的实验评估证明了所提出的后门攻击的有效性。



## **34. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

针对顺序推荐系统的少镜头模型提取攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11677v1) [paper-pdf](http://arxiv.org/pdf/2411.11677v1)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.

摘要: 在针对序列推荐系统的对抗性攻击中，模型提取攻击是一种在没有先验知识的情况下攻击序列推荐模型的方法。现有的研究主要集中在对手通过无数据模型提取来执行黑盒攻击。然而，关于对手开发代理模型的文献中仍然存在着一个显著的差距，这些对手可以访问很少的原始数据(10\%甚至更少)。如何在稀疏数据场景下构建功能相似度高的代理模型是一个亟待解决的问题，本研究通过引入一种针对顺序推荐者的稀疏模型提取框架来解决这一问题，该框架旨在利用稀疏数据构建一个更优的代理模型。所提出的少镜头模型提取框架由两部分组成：自回归增广生成策略和双向修复损失促进模型精馏过程。具体地说，为了生成接近原始数据分布的合成数据，自回归增强生成策略集成了一个概率交互采样器来提取固有依赖关系和一个合成行列式信号模块来表征用户行为模式。随后，针对推荐列表之间的差异，设计了双向修复损失作为辅助损失来纠正代理模型中的错误预测，有效地将受害者模型中的知识传递到代理模型中。在三个数据集上的实验表明，所提出的少镜头模型提取框架产生了更好的代理模型。



## **35. Formal Verification of Deep Neural Networks for Object Detection**

用于对象检测的深度神经网络的形式化验证 cs.CV

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2407.01295v5) [paper-pdf](http://arxiv.org/pdf/2407.01295v5)

**Authors**: Yizhak Y. Elboher, Avraham Raviv, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep neural networks (DNNs) are widely used in real-world applications, yet they remain vulnerable to errors and adversarial attacks. Formal verification offers a systematic approach to identify and mitigate these vulnerabilities, enhancing model robustness and reliability. While most existing verification methods focus on image classification models, this work extends formal verification to the more complex domain of emph{object detection} models. We propose a formulation for verifying the robustness of such models and demonstrate how state-of-the-art verification tools, originally developed for classification, can be adapted for this purpose. Our experiments, conducted on various datasets and networks, highlight the ability of formal verification to uncover vulnerabilities in object detection models, underscoring the need to extend verification efforts to this domain. This work lays the foundation for further research into formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中得到了广泛的应用，但它们仍然容易受到错误和敌意攻击。正式验证提供了一种系统的方法来识别和缓解这些漏洞，从而增强了模型的健壮性和可靠性。虽然现有的验证方法大多集中在图像分类模型上，但该工作将形式验证扩展到更复杂的领域，即目标检测模型。我们提出了一种验证此类模型的稳健性的公式，并演示了最初为分类而开发的最先进的验证工具如何适用于此目的。我们在各种数据集和网络上进行的实验，突出了正式验证发现对象检测模型中漏洞的能力，强调了将验证工作扩展到这一领域的必要性。这项工作为在更广泛的计算机视觉应用中进一步研究形式验证奠定了基础。



## **36. The Dark Side of Trust: Authority Citation-Driven Jailbreak Attacks on Large Language Models**

信任的阴暗面：权威引用驱动的对大型语言模型的越狱攻击 cs.LG

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11407v1) [paper-pdf](http://arxiv.org/pdf/2411.11407v1)

**Authors**: Xikang Yang, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: The widespread deployment of large language models (LLMs) across various domains has showcased their immense potential while exposing significant safety vulnerabilities. A major concern is ensuring that LLM-generated content aligns with human values. Existing jailbreak techniques reveal how this alignment can be compromised through specific prompts or adversarial suffixes. In this study, we introduce a new threat: LLMs' bias toward authority. While this inherent bias can improve the quality of outputs generated by LLMs, it also introduces a potential vulnerability, increasing the risk of producing harmful content. Notably, the biases in LLMs is the varying levels of trust given to different types of authoritative information in harmful queries. For example, malware development often favors trust GitHub. To better reveal the risks with LLM, we propose DarkCite, an adaptive authority citation matcher and generator designed for a black-box setting. DarkCite matches optimal citation types to specific risk types and generates authoritative citations relevant to harmful instructions, enabling more effective jailbreak attacks on aligned LLMs.Our experiments show that DarkCite achieves a higher attack success rate (e.g., LLama-2 at 76% versus 68%) than previous methods. To counter this risk, we propose an authenticity and harm verification defense strategy, raising the average defense pass rate (DPR) from 11% to 74%. More importantly, the ability to link citations to the content they encompass has become a foundational function in LLMs, amplifying the influence of LLMs' bias toward authority.

摘要: 大型语言模型(LLM)在不同领域的广泛部署展示了它们的巨大潜力，同时也暴露了重大的安全漏洞。一个主要的问题是确保LLM生成的内容符合人类的价值观。现有的越狱技术揭示了如何通过特定的提示或对抗性后缀来破坏这种对齐。在这项研究中，我们引入了一个新的威胁：LLMS对权威的偏见。虽然这种固有的偏见可以提高低成本管理产生的产出的质量，但它也引入了一个潜在的脆弱性，增加了产生有害内容的风险。值得注意的是，LLMS中的偏差是在有害查询中对不同类型的权威信息给予的不同程度的信任。例如，恶意软件开发通常偏向信任GitHub。为了更好地揭示LLM的风险，我们提出了DarkCite，这是一个为黑箱设置而设计的自适应权威引用匹配器和生成器。DarkCite将最佳引用类型与特定的风险类型相匹配，并生成与有害指令相关的权威引用，从而对对齐的LLMS进行更有效的越狱攻击。我们的实验表明，与以前的方法相比，DarkCite实现了更高的攻击成功率(例如，骆驼-2为76%，而不是68%)。为了应对这种风险，我们提出了真实性和危害性验证防御策略，将平均防御通过率(DPR)从11%提高到74%。更重要的是，将引文与它们所包含的内容相联系的能力已经成为LLMS的一项基本功能，放大了LLMS对权威的偏见的影响。



## **37. Hacking Back the AI-Hacker: Prompt Injection as a Defense Against LLM-driven Cyberattacks**

黑客攻击人工智能黑客：即时注入作为抵御LLM驱动的网络攻击的防御 cs.CR

v0.2 (evaluated on more agents)

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.20911v2) [paper-pdf](http://arxiv.org/pdf/2410.20911v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese

**Abstract**: Large language models (LLMs) are increasingly being harnessed to automate cyberattacks, making sophisticated exploits more accessible and scalable. In response, we propose a new defense strategy tailored to counter LLM-driven cyberattacks. We introduce Mantis, a defensive framework that exploits LLMs' susceptibility to adversarial inputs to undermine malicious operations. Upon detecting an automated cyberattack, Mantis plants carefully crafted inputs into system responses, leading the attacker's LLM to disrupt their own operations (passive defense) or even compromise the attacker's machine (active defense). By deploying purposefully vulnerable decoy services to attract the attacker and using dynamic prompt injections for the attacker's LLM, Mantis can autonomously hack back the attacker. In our experiments, Mantis consistently achieved over 95% effectiveness against automated LLM-driven attacks. To foster further research and collaboration, Mantis is available as an open-source tool: https://github.com/pasquini-dario/project_mantis

摘要: 大型语言模型(LLM)越来越多地被用来自动化网络攻击，使复杂的利用更容易获得和可扩展。作为回应，我们提出了一种新的防御战略，以对抗LLM驱动的网络攻击。我们引入了Mantis，这是一个防御框架，利用LLMS对对手输入的敏感性来破坏恶意操作。在检测到自动网络攻击后，螳螂工厂会精心设计输入到系统响应中，导致攻击者的LLM扰乱自己的操作(被动防御)，甚至危害攻击者的机器(主动防御)。通过部署故意易受攻击的诱骗服务来吸引攻击者，并对攻击者的LLM使用动态提示注入，螳螂可以自主地攻击攻击者。在我们的实验中，螳螂对自动LLM驱动的攻击始终取得了95%以上的效率。为了促进进一步的研究和合作，Mantis以开源工具的形式提供：https://github.com/pasquini-dario/project_mantis



## **38. Adapting to Cyber Threats: A Phishing Evolution Network (PEN) Framework for Phishing Generation and Analyzing Evolution Patterns using Large Language Models**

适应网络威胁：用于使用大型语言模型进行网络钓鱼生成和分析进化模式的网络钓鱼进化网络（PEN）框架 cs.CR

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.11389v1) [paper-pdf](http://arxiv.org/pdf/2411.11389v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Hongsheng Hu, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), particularly deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains their effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems vulnerable to an ever-growing array of attacks. Addressing this gap is essential to strengthening defenses in an increasingly hostile cyber landscape. To address this gap, we propose the Phishing Evolution Network (PEN), a framework leveraging large language models (LLMs) and adversarial training mechanisms to continuously generate high quality and realistic diverse phishing samples, and analyze features of LLM-provided phishing to understand evolving phishing patterns. We evaluate the quality and diversity of phishing samples generated by PEN and find that it produces over 80% realistic phishing samples, effectively expanding phishing datasets across seven dominant types. These PEN-generated samples enhance the performance of current phishing detectors, leading to a 40% improvement in detection accuracy. Additionally, the use of PEN significantly boosts model robustness, reducing detectors' sensitivity to perturbations by up to 60%, thereby decreasing attack success rates under adversarial conditions. When we analyze the phishing patterns that are used in LLM-generated phishing, the cognitive complexity and the tone of time limitation are detected with statistically significant differences compared with existing phishing.

摘要: 网络钓鱼仍然是一个普遍存在的网络威胁，因为攻击者精心制作了欺骗性电子邮件，以引诱受害者泄露敏感信息。虽然人工智能(AI)，特别是深度学习，已经成为防御网络钓鱼攻击的关键组件，但这些方法面临着严重的限制。由于缺乏公开可用的、多样化的和更新的数据，这主要是由于隐私问题，限制了它们的有效性。随着钓鱼策略的快速发展，基于有限、过时数据的模型很难检测出新的、复杂的欺骗策略，这使得系统容易受到越来越多的攻击。在日益充满敌意的网络环境中，解决这一差距对于加强防御至关重要。为了弥补这一差距，我们提出了钓鱼进化网络(PEN)，这是一个利用大型语言模型(LLMS)和对手训练机制来持续生成高质量和真实的多样化钓鱼样本的框架，并分析LLM提供的钓鱼特征以了解不断演变的钓鱼模式。我们评估了PEN生成的网络钓鱼样本的质量和多样性，发现它产生了超过80%的真实网络钓鱼样本，有效地扩展了七种主要类型的网络钓鱼数据集。这些笔生成的样本增强了当前网络钓鱼检测器的性能，导致检测准确率提高了40%。此外，PEN的使用显著提高了模型的稳健性，将检测器对扰动的敏感度降低了高达60%，从而降低了对抗性条件下的攻击成功率。当我们分析LLM生成的网络钓鱼中使用的网络钓鱼模式时，我们检测到了认知复杂性和时间限制的基调，与现有的网络钓鱼相比具有统计学意义上的差异。



## **39. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

CROW：通过内部一致性规范化消除大型语言模型的后门 cs.CL

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.12768v1) [paper-pdf](http://arxiv.org/pdf/2411.12768v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Recent studies reveal that Large Language Models (LLMs) are susceptible to backdoor attacks, where adversaries embed hidden triggers that manipulate model responses. Existing backdoor defense methods are primarily designed for vision or classification tasks, and are thus ineffective for text generation tasks, leaving LLMs vulnerable. We introduce Internal Consistency Regularization (CROW), a novel defense using consistency regularization finetuning to address layer-wise inconsistencies caused by backdoor triggers. CROW leverages the intuition that clean models exhibit smooth, consistent transitions in hidden representations across layers, whereas backdoored models show noticeable fluctuation when triggered. By enforcing internal consistency through adversarial perturbations and regularization, CROW neutralizes backdoor effects without requiring clean reference models or prior trigger knowledge, relying only on a small set of clean data. This makes it practical for deployment across various LLM architectures. Experimental results demonstrate that CROW consistently achieves a significant reductions in attack success rates across diverse backdoor strategies and tasks, including negative sentiment, targeted refusal, and code injection, on models such as Llama-2 (7B, 13B), CodeLlama (7B, 13B) and Mistral-7B, while preserving the model's generative capabilities.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到后门攻击，即对手嵌入操纵模型响应的隐藏触发器。现有的后门防御方法主要是为视觉或分类任务设计的，因此对于文本生成任务无效，从而使LLMS容易受到攻击。我们引入了内部一致性正则化(CROW)，这是一种使用一致性正则化精调来解决后门触发器引起的层级不一致的新防御机制。Crow利用了这样一种直觉，即干净的模型在各层的隐藏表示中显示出平滑、一致的过渡，而回溯的模型在触发时会显示出明显的波动。通过对抗性扰动和正则化来强制内部一致性，Crow中和了后门效应，而不需要干净的参考模型或事先的触发知识，只依赖于一小部分干净的数据。这使得它适用于跨各种LLM体系结构进行部署。实验结果表明，在Llama-2(7B，13B)、CodeLlama(7B，13B)和Mistral-7B等模型上，Crow在不同的后门策略和任务(包括负面情绪、定向拒绝和代码注入)上持续显著降低攻击成功率，同时保持了模型的生成能力。



## **40. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2410.23091v4) [paper-pdf](http://arxiv.org/pdf/2410.23091v4)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at \href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。代码可在\href{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}{https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff}上获得



## **41. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

**SubmitDate**: 2024-11-18    [abs](http://arxiv.org/abs/2411.13587v1) [paper-pdf](http://arxiv.org/pdf/2411.13587v1)

**Authors**: Taowen Wang, Dongfang Liu, James Chenhao Liang, Wenhao Yang, Qifan Wang, Cheng Han, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce an untargeted position-aware attack objective that leverages spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, this work advances both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了一个利用空间基础来破坏机器人动作稳定性的无目标位置感知攻击目标，以及一个操纵机器人轨迹的目标攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，这项工作促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前开发强大的防御策略的必要性。



## **42. Countering Backdoor Attacks in Image Recognition: A Survey and Evaluation of Mitigation Strategies**

对抗图像识别中的后门攻击：缓解策略的调查和评估 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11200v1) [paper-pdf](http://arxiv.org/pdf/2411.11200v1)

**Authors**: Kealan Dunnett, Reza Arablouei, Dimity Miller, Volkan Dedeoglu, Raja Jurdak

**Abstract**: The widespread adoption of deep learning across various industries has introduced substantial challenges, particularly in terms of model explainability and security. The inherent complexity of deep learning models, while contributing to their effectiveness, also renders them susceptible to adversarial attacks. Among these, backdoor attacks are especially concerning, as they involve surreptitiously embedding specific triggers within training data, causing the model to exhibit aberrant behavior when presented with input containing the triggers. Such attacks often exploit vulnerabilities in outsourced processes, compromising model integrity without affecting performance on clean (trigger-free) input data. In this paper, we present a comprehensive review of existing mitigation strategies designed to counter backdoor attacks in image recognition. We provide an in-depth analysis of the theoretical foundations, practical efficacy, and limitations of these approaches. In addition, we conduct an extensive benchmarking of sixteen state-of-the-art approaches against eight distinct backdoor attacks, utilizing three datasets, four model architectures, and three poisoning ratios. Our results, derived from 122,236 individual experiments, indicate that while many approaches provide some level of protection, their performance can vary considerably. Furthermore, when compared to two seminal approaches, most newer approaches do not demonstrate substantial improvements in overall performance or consistency across diverse settings. Drawing from these findings, we propose potential directions for developing more effective and generalizable defensive mechanisms in the future.

摘要: 深度学习在各个行业的广泛采用带来了巨大的挑战，特别是在模型的可解释性和安全性方面。深度学习模型固有的复杂性，虽然有助于它们的有效性，但也使它们容易受到对手的攻击。其中，后门攻击尤其令人担忧，因为它们涉及在训练数据中秘密嵌入特定触发器，导致在输入包含触发器的输入时导致模型表现出异常行为。此类攻击通常利用外包流程中的漏洞，在不影响干净(无触发器)输入数据的性能的情况下损害模型完整性。在这篇文章中，我们提出了一个全面的审查现有的缓解策略，旨在对抗后门攻击的图像识别。我们对这些方法的理论基础、实践有效性和局限性进行了深入分析。此外，我们利用三个数据集、四个模型体系结构和三个投毒率，对针对八种不同后门攻击的16种最先进方法进行了广泛的基准测试。我们的结果来自122,236个单独的实验，表明虽然许多方法提供了一定程度的保护，但它们的性能可能会有很大的差异。此外，与两种开创性的方法相比，大多数较新的方法在总体性能或跨不同环境的一致性方面没有显示出实质性的改进。根据这些发现，我们提出了未来发展更有效和更具普遍性的防御机制的潜在方向。



## **43. Exploiting the Uncoordinated Privacy Protections of Eye Tracking and VR Motion Data for Unauthorized User Identification**

利用眼动追踪和VR运动数据的不协调隐私保护来识别未经授权的用户 cs.HC

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.12766v1) [paper-pdf](http://arxiv.org/pdf/2411.12766v1)

**Authors**: Samantha Aziz, Oleg Komogortsev

**Abstract**: Virtual reality (VR) devices use a variety of sensors to capture a rich body of user-generated data, which can be misused by malicious parties to covertly infer information about the user. Privacy-enhancing techniques seek to reduce the amount of personally identifying information in sensor data, but these techniques are typically developed for a subset of data streams that are available on the platform, without consideration for the auxiliary information that may be readily available from other sensors. In this paper, we evaluate whether body motion data can be used to circumvent the privacy protections applied to eye tracking data to enable user identification on a VR platform, and vice versa. We empirically show that eye tracking, headset tracking, and hand tracking data are not only informative for inferring user identity on their own, but contain complementary information that can increase the rate of successful user identification. Most importantly, we demonstrate that applying privacy protections to only a subset of the data available in VR can create an opportunity for an adversary to bypass those privacy protections by using other unprotected data streams that are available on the platform, performing a user identification attack as accurately as though a privacy mechanism was never applied. These results highlight a new privacy consideration at the intersection between eye tracking and VR, and emphasizes the need for privacy-enhancing techniques that address multiple technologies comprehensively.

摘要: 虚拟现实(VR)设备使用各种传感器来捕获大量用户生成的数据，这些数据可能被恶意方滥用来秘密推断用户的信息。隐私增强技术试图减少传感器数据中的个人识别信息量，但这些技术通常是针对平台上可用的数据流的子集开发的，而不考虑可能从其他传感器容易获得的辅助信息。在本文中，我们评估了身体运动数据是否可以用来规避应用于眼睛跟踪数据的隐私保护，以便在VR平台上进行用户识别，反之亦然。我们的经验表明，眼睛跟踪、耳机跟踪和手部跟踪数据不仅对推断用户身份本身具有信息性，而且包含补充信息，可以提高用户识别的成功率。最重要的是，我们证明，仅对VR中可用数据的子集应用隐私保护可以为攻击者创造机会，通过使用平台上可用的其他未受保护的数据流来绕过这些隐私保护，就像从未应用隐私机制一样准确地执行用户识别攻击。这些结果突显了眼睛跟踪和VR之间的交叉点上的一个新的隐私考虑，并强调了全面解决多种技术的隐私增强技术的必要性。



## **44. Optimal Denial-of-Service Attacks Against Partially-Observable Real-Time Monitoring Systems**

针对部分可观察实时监控系统的最佳拒绝服务攻击 cs.IT

arXiv admin note: text overlap with arXiv:2403.04489

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2409.16794v2) [paper-pdf](http://arxiv.org/pdf/2409.16794v2)

**Authors**: Saad Kriouile, Mohamad Assaad, Amira Alloum, Touraj Soleymani

**Abstract**: In this paper, we investigate the impact of denial-of-service attacks on the status updating of a cyber-physical system with one or more sensors connected to a remote monitor via unreliable channels. We approach the problem from the perspective of an adversary that can strategically jam a subset of the channels. The sources are modeled as Markov chains, and the performance of status updating is measured based on the age of incorrect information at the monitor. Our objective is to derive jamming policies that strike a balance between the degradation of the system's performance and the conservation of the adversary's energy. For a single-source scenario, we formulate the problem as a partially-observable Markov decision process, and rigorously prove that the optimal jamming policy is of a threshold form. We then extend the problem to a multi-source scenario. We formulate this problem as a restless multi-armed bandit, and provide a jamming policy based on the Whittle's index. Our numerical results highlight the performance of our policies compared to baseline policies.

摘要: 在本文中，我们研究了拒绝服务攻击对一个或多个传感器通过不可靠的信道连接到远程监视器的网络物理系统状态更新的影响。我们从一个对手的角度来处理这个问题，这个对手可以战略性地堵塞部分渠道。信源被建模为马尔可夫链，状态更新的性能基于监视器处错误信息的年龄来衡量。我们的目标是制定干扰策略，在系统性能下降和保存对手能量之间取得平衡。对于单源情况，我们将问题描述为部分可观测的马尔可夫决策过程，并严格证明了最优干扰策略是门限形式的。然后，我们将问题扩展到多源场景。我们将该问题描述为一个躁动的多臂强盗问题，并基于惠特尔指标提出了一种干扰策略。我们的数字结果突出了我们的政策与基准政策相比的表现。



## **45. CLMIA: Membership Inference Attacks via Unsupervised Contrastive Learning**

CLMIA：通过无监督对比学习的成员推断攻击 cs.LG

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11144v1) [paper-pdf](http://arxiv.org/pdf/2411.11144v1)

**Authors**: Depeng Chen, Xiao Liu, Jie Cui, Hong Zhong

**Abstract**: Since machine learning model is often trained on a limited data set, the model is trained multiple times on the same data sample, which causes the model to memorize most of the training set data. Membership Inference Attacks (MIAs) exploit this feature to determine whether a data sample is used for training a machine learning model. However, in realistic scenarios, it is difficult for the adversary to obtain enough qualified samples that mark accurate identity information, especially since most samples are non-members in real world applications. To address this limitation, in this paper, we propose a new attack method called CLMIA, which uses unsupervised contrastive learning to train an attack model without using extra membership status information. Meanwhile, in CLMIA, we require only a small amount of data with known membership status to fine-tune the attack model. Experimental results demonstrate that CLMIA performs better than existing attack methods for different datasets and model structures, especially with data with less marked identity information. In addition, we experimentally find that the attack performs differently for different proportions of labeled identity information for member and non-member data. More analysis proves that our attack method performs better with less labeled identity information, which applies to more realistic scenarios.

摘要: 由于机器学习模型通常是在有限的数据集上训练的，所以该模型在同一数据样本上被多次训练，这使得该模型记住了大部分训练集数据。成员资格推理攻击(MIA)利用这一特征来确定数据样本是否用于训练机器学习模型。然而，在现实场景中，攻击者很难获得足够的合格样本来标记准确的身份信息，特别是在现实世界应用中大多数样本都是非成员的情况下。针对这一局限性，本文提出了一种新的攻击方法CLMIA，该方法使用无监督对比学习来训练攻击模型，而不使用额外的成员状态信息。同时，在CLMIA中，我们只需要少量已知成员状态的数据来微调攻击模型。实验结果表明，对于不同的数据集和模型结构，CLMIA的攻击性能优于现有的攻击方法，尤其是对于身份信息标记较少的数据。此外，我们还通过实验发现，对于成员和非成员数据，对于不同比例的标签身份信息，该攻击的表现是不同的。更多的分析证明，我们的攻击方法在标签身份信息较少的情况下性能更好，适用于更真实的场景。



## **46. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

18 pages, 10 figures

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2411.11114v1) [paper-pdf](http://arxiv.org/pdf/2411.11114v1)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Rui Zheng, Kui Ren, Chun Chen

**Abstract**: Despite the outstanding performance of Large language models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses.Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explain typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing the representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of these attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives (which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on four mainstream LLMs under seven jailbreak strategies. Our evaluation finds that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. Although this manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals, it still produce abnormal activation which can be caught in the circuit analysis.

摘要: 尽管大型语言模型(LLM)在不同的任务中表现出色，但它们很容易受到越狱攻击，在这些攻击中，敌意提示被精心制作以绕过其安全机制并引发意外响应。尽管越狱攻击非常普遍，但对其潜在机制的了解仍然有限。最近的研究已经通过分析越狱提示引起的潜伏空间的表征变化或识别有助于这些攻击成功的关键神经元来解释LLM的典型越狱行为(例如，模型拒绝响应的程度)。然而，这些研究既没有探索多样化的越狱模式，也没有提供从电路故障到表征变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailBreakLens，一个解释框架，它从表示(揭示越狱如何改变模型的危害性感知)和电路角度(通过识别导致漏洞的关键电路来揭示这些欺骗的原因)来分析越狱机制，跟踪它们在整个响应生成过程中的演变。然后，我们在七种越狱策略下对四种主流的低成本移动模型的越狱行为进行了深入的评估。我们的评估发现，越狱提示放大了那些强化肯定反应的成分，同时抑制了那些产生拒绝的成分。尽管这种操作将模型表示转移到安全簇以欺骗LLM，导致它提供详细的响应而不是拒绝，但它仍然产生可以在电路分析中发现的异常激活。



## **47. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

探索对抗前沿：通过对抗超容量量化稳健性 cs.CR

**SubmitDate**: 2024-11-17    [abs](http://arxiv.org/abs/2403.05100v2) [paper-pdf](http://arxiv.org/pdf/2403.05100v2)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.

摘要: 对深度学习模型的敌意攻击的威胁不断升级，特别是在安全关键领域，这突显了需要强大的深度学习系统。传统的稳健性评估依赖于对抗精度，该精度衡量模型在特定扰动强度下的性能。然而，这种单一的度量并不能完全概括模型对不同程度扰动的总体弹性。为了弥补这一差距，我们提出了一种新的度量标准，称为对抗性超体积，从多目标优化的角度全面评估深度学习模型在一系列扰动强度下的稳健性。这一指标允许对防御机制进行深入比较，并认识到较弱的防御策略在健壮性方面的微小改进。此外，我们采用了一种新的训练算法，该算法在不同的扰动强度下均匀地增强了对抗的稳健性，而不是狭隘地专注于优化对抗的准确性。我们广泛的实证研究验证了对抗性超卷度量的有效性，证明了它能够揭示对抗性准确性忽略的稳健性的细微差异。这项研究提供了一种新的稳健性衡量标准，并为评估和基准当前和未来防御模型对对手威胁的弹性建立了标准。



## **48. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

游戏理论的内曼-皮尔森检测对抗战略规避 cs.CR

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2206.05276v3) [paper-pdf](http://arxiv.org/pdf/2206.05276v3)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.

摘要: 网络系统的安全性在很大程度上取决于对敌方行为的识别和识别。传统的检测方法侧重于特定类别的攻击，已不适用于日益隐蔽和欺骗性的攻击，这些攻击旨在从战略上绕过检测。这项工作旨在开发一种整体理论来对抗这种规避攻击。基于Neyman-Pearson(NP)假设检验公式，我们重点扩展了一类基本的基于统计的检测方法。我们提出了博弈论框架来捕捉战略规避攻击者和规避感知NP检测器之间的冲突关系。通过分析攻击者和NP检测器的均衡行为，我们用均衡接收-操作-特征(EROC)曲线来表征它们的性能。我们证明了逃避感知NP检测器的性能优于被动NP检测器，前者可以针对攻击者的行为采取策略性行动，并根据收到的消息自适应地修改其决策规则。此外，我们将我们的框架扩展到顺序设置，在该设置中，用户发送相同分布的消息。我们通过一个异常检测的案例验证了分析结果。



## **49. A Survey of Graph Unlearning**

图形遗忘研究综述 cs.LG

22 page review paper on graph unlearning

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2310.02164v3) [paper-pdf](http://arxiv.org/pdf/2310.02164v3)

**Authors**: Anwar Said, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the right to be forgotten. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.

摘要: 在追求负责任的人工智能方面，图形遗忘成为一个关键的进步，提供了从训练的模型中移除敏感数据痕迹的手段，从而维护了被遗忘的权利。显然，图机器学习表现出对数据隐私和敌意攻击的敏感性，因此有必要应用图遗忘技术来有效地解决这些问题。在这篇全面的调查论文中，我们提出了第一次系统地回顾图形遗忘方法，包括一系列不同的方法，并提供了详细的分类和最新的文献综述，以促进新进入该领域的研究人员的理解。为了确保清晰，我们对图形遗忘中使用的基本概念和评估措施进行了清晰的解释，以迎合具有不同专业水平的更广泛的受众。深入挖掘潜在的应用，我们探索了图遗忘在不同领域的多功能性，包括但不限于社交网络、对手环境、推荐系统和物联网等资源受限环境，说明了它在保护数据隐私和增强AI系统健壮性方面的潜在影响。最后，我们阐明了有前途的研究方向，鼓励在图忘却学习领域内的进一步进步和创新。通过奠定坚实的基础和促进持续进步，这项调查旨在激励研究人员进一步推进图形遗忘领域，从而灌输对人工智能系统伦理增长的信心，并加强机器学习技术在各个领域的负责任应用。



## **50. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-16    [abs](http://arxiv.org/abs/2405.18942v3) [paper-pdf](http://arxiv.org/pdf/2405.18942v3)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。本文介绍了一种新的框架VRCP，它利用最新的神经网络验证方法来恢复对抗性攻击下的覆盖保证。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



