# Latest Adversarial Attack Papers
**update at 2024-06-11 10:38:11**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robust Distribution Learning with Local and Global Adversarial Corruptions**

具有本地和全球对抗性腐蚀的稳健分布学习 cs.LG

Accepted for presentation at the Conference on Learning Theory (COLT)  2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06509v1) [paper-pdf](http://arxiv.org/pdf/2406.06509v1)

**Authors**: Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**Abstract**: We consider learning in an adversarial environment, where an $\varepsilon$-fraction of samples from a distribution $P$ are arbitrarily modified (*global* corruptions) and the remaining perturbations have average magnitude bounded by $\rho$ (*local* corruptions). Given access to $n$ such corrupted samples, we seek a computationally efficient estimator $\hat{P}_n$ that minimizes the Wasserstein distance $\mathsf{W}_1(\hat{P}_n,P)$. In fact, we attack the fine-grained task of minimizing $\mathsf{W}_1(\Pi_\# \hat{P}_n, \Pi_\# P)$ for all orthogonal projections $\Pi \in \mathbb{R}^{d \times d}$, with performance scaling with $\mathrm{rank}(\Pi) = k$. This allows us to account simultaneously for mean estimation ($k=1$), distribution estimation ($k=d$), as well as the settings interpolating between these two extremes. We characterize the optimal population-limit risk for this task and then develop an efficient finite-sample algorithm with error bounded by $\sqrt{\varepsilon k} + \rho + d^{O(1)}\tilde{O}(n^{-1/k})$ when $P$ has bounded moments of order $2+\delta$, for constant $\delta > 0$. For data distributions with bounded covariance, our finite-sample bounds match the minimax population-level optimum for large sample sizes. Our efficient procedure relies on a novel trace norm approximation of an ideal yet intractable 2-Wasserstein projection estimator. We apply this algorithm to robust stochastic optimization, and, in the process, uncover a new method for overcoming the curse of dimensionality in Wasserstein distributionally robust optimization.

摘要: 我们考虑在对抗性环境中学习，其中来自分布$P$的$\varepsilon$-分数样本被任意修改(*全局*损坏)，而其余扰动的平均幅度由$\rho$(*局部*损坏)限定。在给定$n$这样的破坏样本的情况下，我们寻找一个计算上有效的估计量$\hat{P}_n$以最小化Wasserstein距离$\mathsf{W}_1(\hat{P}_n，P)$。事实上，我们对所有的正交投影$\pI\in\mathbb{R}^{d\time d}$发起了最小化$\mathsf{W}_1(\pI_#\hat{P}_n，\pI_#P)$的细粒度任务，并且性能伸缩为$\mathm{RANK}(\pI)=k$。这允许我们同时考虑平均值估计($k=1$)、分布估计($k=d$)以及在这两个极值之间插入的设置。我们刻画了该任务的最优总体极限风险，并在此基础上发展了一个有效的有限样本算法，当$P$具有$2+β$的有界矩时，误差有界于$Sqrt{varepsilon k}+Rho+d^{O(1)}(n^{-1/k})$。对于协方差有界的数据分布，我们的有限样本界与大样本量的最小最大总体水平最优匹配。我们的有效过程依赖于一个理想但难以处理的2-Wasserstein投影估计量的一个新的迹范数近似。我们将该算法应用于稳健随机优化中，并在此过程中发现了一种克服Wasserstein分布稳健优化中的维度灾难的新方法。



## **2. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.04313v2) [paper-pdf](http://arxiv.org/pdf/2406.04313v2)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **3. Evolving Assembly Code in an Adversarial Environment**

对抗环境中发展汇编代码 cs.NE

20 pages, 6 figures, 6 listings, 5 tables

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2403.19489v2) [paper-pdf](http://arxiv.org/pdf/2403.19489v2)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve Assembly code for the CodeGuru competition. The goal is to create a survivor -- an Assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the Assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. To push evolution further, we implemented memetic operators that utilize machine learning to explore the solution space effectively. This work has important applications for cyber-security as we utilize evolution to detect weaknesses in survivors. The Assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。为了进一步推进进化，我们实现了模因算子，利用机器学习来有效地探索解空间。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。Assembly BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **4. Explainable Graph Neural Networks Under Fire**

受攻击的可解释图神经网络 cs.LG

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06417v1) [paper-pdf](http://arxiv.org/pdf/2406.06417v1)

**Authors**: Zhong Li, Simon Geisler, Yuhang Wang, Stephan Günnemann, Matthijs van Leeuwen

**Abstract**: Predictions made by graph neural networks (GNNs) usually lack interpretability due to their complex computational behavior and the abstract nature of graphs. In an attempt to tackle this, many GNN explanation methods have emerged. Their goal is to explain a model's predictions and thereby obtain trust when GNN models are deployed in decision critical applications. Most GNN explanation methods work in a post-hoc manner and provide explanations in the form of a small subset of important edges and/or nodes. In this paper we demonstrate that these explanations can unfortunately not be trusted, as common GNN explanation methods turn out to be highly susceptible to adversarial perturbations. That is, even small perturbations of the original graph structure that preserve the model's predictions may yield drastically different explanations. This calls into question the trustworthiness and practical utility of post-hoc explanation methods for GNNs. To be able to attack GNN explanation models, we devise a novel attack method dubbed \textit{GXAttack}, the first \textit{optimization-based} adversarial attack method for post-hoc GNN explanations under such settings. Due to the devastating effectiveness of our attack, we call for an adversarial evaluation of future GNN explainers to demonstrate their robustness.

摘要: 由于图的复杂的计算行为和图的抽象性质，图神经网络(GNN)的预测通常缺乏可解释性。为了解决这个问题，出现了许多GNN解释方法。他们的目标是解释模型的预测，从而在决策关键应用程序中部署GNN模型时获得信任。大多数GNN解释方法以后自组织的方式工作，并以重要边和/或节点的小子集的形式提供解释。在本文中，我们证明了不幸的是，这些解释不能被信任，因为常见的GNN解释方法被证明非常容易受到对抗性扰动的影响。也就是说，即使是对原始图表结构的微小扰动，保留了模型的预测，也可能产生截然不同的解释。这使人们对特别解释GNN的方法的可信性和实用性产生了疑问。为了能够攻击GNN解释模型，我们设计了一种新的攻击方法，称为文本{GXAttack}，这是第一个在这种情况下针对后自组织GNN解释的对抗性攻击方法。由于我们的攻击具有毁灭性的效果，我们呼吁对未来的GNN解释器进行对抗性评估，以证明其健壮性。



## **5. RAID: A Shared Benchmark for Robust Evaluation of Machine-Generated Text Detectors**

RAGE：机器生成文本检测器稳健评估的共享基准 cs.CL

ACL 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2405.07940v2) [paper-pdf](http://arxiv.org/pdf/2405.07940v2)

**Authors**: Liam Dugan, Alyssa Hwang, Filip Trhlik, Josh Magnus Ludan, Andrew Zhu, Hainiu Xu, Daphne Ippolito, Chris Callison-Burch

**Abstract**: Many commercial and open-source models claim to detect machine-generated text with extremely high accuracy (99% or more). However, very few of these detectors are evaluated on shared benchmark datasets and even when they are, the datasets used for evaluation are insufficiently challenging-lacking variations in sampling strategy, adversarial attacks, and open-source generative models. In this work we present RAID: the largest and most challenging benchmark dataset for machine-generated text detection. RAID includes over 6 million generations spanning 11 models, 8 domains, 11 adversarial attacks and 4 decoding strategies. Using RAID, we evaluate the out-of-domain and adversarial robustness of 8 open- and 4 closed-source detectors and find that current detectors are easily fooled by adversarial attacks, variations in sampling strategies, repetition penalties, and unseen generative models. We release our data along with a leaderboard to encourage future research.

摘要: 许多商业和开源模型声称可以以极高的准确性（99%或更高）检测机器生成的文本。然而，这些检测器中很少有在共享基准数据集上进行评估，即使如此，用于评估的数据集也不够具有挑战性--缺乏采样策略、对抗性攻击和开源生成模型的变化。在这项工作中，我们介绍了RAIDA：用于机器生成文本检测的最大、最具挑战性的基准数据集。磁盘阵列包含超过600万代，涵盖11个模型、8个域、11种对抗性攻击和4种解码策略。使用RAIDGE，我们评估了8个开源检测器和4个开源检测器的域外和对抗稳健性，发现当前的检测器很容易被对抗攻击、采样策略的变化、重复惩罚和看不见的生成模型所愚弄。我们发布我们的数据和排行榜，以鼓励未来的研究。



## **6. Towards Transferable Targeted 3D Adversarial Attack in the Physical World**

迈向物理世界中的可转移定向3D对抗攻击 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2312.09558v3) [paper-pdf](http://arxiv.org/pdf/2312.09558v3)

**Authors**: Yao Huang, Yinpeng Dong, Shouwei Ruan, Xiao Yang, Hang Su, Xingxing Wei

**Abstract**: Compared with transferable untargeted attacks, transferable targeted adversarial attacks could specify the misclassification categories of adversarial samples, posing a greater threat to security-critical tasks. In the meanwhile, 3D adversarial samples, due to their potential of multi-view robustness, can more comprehensively identify weaknesses in existing deep learning systems, possessing great application value. However, the field of transferable targeted 3D adversarial attacks remains vacant. The goal of this work is to develop a more effective technique that could generate transferable targeted 3D adversarial examples, filling the gap in this field. To achieve this goal, we design a novel framework named TT3D that could rapidly reconstruct from few multi-view images into Transferable Targeted 3D textured meshes. While existing mesh-based texture optimization methods compute gradients in the high-dimensional mesh space and easily fall into local optima, leading to unsatisfactory transferability and distinct distortions, TT3D innovatively performs dual optimization towards both feature grid and Multi-layer Perceptron (MLP) parameters in the grid-based NeRF space, which significantly enhances black-box transferability while enjoying naturalness. Experimental results show that TT3D not only exhibits superior cross-model transferability but also maintains considerable adaptability across different renders and vision tasks. More importantly, we produce 3D adversarial examples with 3D printing techniques in the real world and verify their robust performance under various scenarios.

摘要: 与可转移的非定向攻击相比，可转移的定向攻击可以指定对手样本的错误分类类别，对安全关键任务构成更大的威胁。同时，3D对抗性样本由于其潜在的多视点稳健性，可以更全面地识别现有深度学习系统中的弱点，具有很大的应用价值。然而，可转移的定向3D对抗性攻击领域仍然空白。这项工作的目标是开发一种更有效的技术，可以生成可转移的目标3D对抗性实例，填补这一领域的空白。为了实现这一目标，我们设计了一种新的框架TT3D，它可以从少量的多视角图像快速重建为可转移的目标3D纹理网格。针对现有的基于网格的纹理优化方法在高维网格空间中计算梯度，容易陷入局部最优，导致可移植性差和失真明显的问题，TT3D创新性地在基于网格的NERF空间中对特征网格和多层感知器(MLP)参数进行双重优化，在享受自然感的同时显著增强了黑盒的可传递性。实验结果表明，TT3D不仅表现出了良好的跨模型可移植性，而且在不同的渲染和视觉任务之间保持了相当大的适应性。更重要的是，我们用3D打印技术在真实世界中生成了3D对抗性例子，并验证了它们在各种场景下的健壮性。



## **7. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

警报器--通过欺骗和适应性分析推进网络安全 cs.CR

7 pages, 6 figures

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06225v1) [paper-pdf](http://arxiv.org/pdf/2406.06225v1)

**Authors**: Girish Kulathumani, Samruth Ananthanarayanan, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for real-time analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.

摘要: SIREN代表了一项开创性的研究成果，旨在通过欺骗、机器学习和主动威胁分析的战略集成来加强网络安全。这个项目从神话中的警报器中获得灵感，使用复杂的方法将潜在的威胁引诱到受控环境中。该系统采用动态机器学习模型进行实时分析和分类，确保对新出现的网络威胁持续适应。该体系结构框架包括链接监控代理、用于动态链接分析的专门构建的机器学习模型，以及丰富了模拟用户交互以加强威胁参与的蜜罐。蜜罐内的数据保护通过概率加密得到加强。此外，模拟用户活动的加入扩展了系统的能力，即使在用户退出后也能捕获潜在攻击者并从他们那里学习。SIREN在网络安全方面引入了一种范式转变，将传统的防御机制转变为主动参与并向潜在对手学习的系统。这项研究努力加强用户保护，同时为不断完善以应对不断变化的网络安全威胁提供有价值的见解。



## **8. Defending Against Physical Adversarial Patch Attacks on Infrared Human Detection**

红外人体检测防御物理对抗补丁攻击 cs.CV

Accepted at ICIP2024. Lukas Strack and Futa Waseda contributed  equally

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2309.15519v3) [paper-pdf](http://arxiv.org/pdf/2309.15519v3)

**Authors**: Lukas Strack, Futa Waseda, Huy H. Nguyen, Yinqiang Zheng, Isao Echizen

**Abstract**: Infrared detection is an emerging technique for safety-critical tasks owing to its remarkable anti-interference capability. However, recent studies have revealed that it is vulnerable to physically-realizable adversarial patches, posing risks in its real-world applications. To address this problem, we are the first to investigate defense strategies against adversarial patch attacks on infrared detection, especially human detection. We propose a straightforward defense strategy, patch-based occlusion-aware detection (POD), which efficiently augments training samples with random patches and subsequently detects them. POD not only robustly detects people but also identifies adversarial patch locations. Surprisingly, while being extremely computationally efficient, POD easily generalizes to state-of-the-art adversarial patch attacks that are unseen during training. Furthermore, POD improves detection precision even in a clean (i.e., no-attack) situation due to the data augmentation effect. Our evaluation demonstrates that POD is robust to adversarial patches of various shapes and sizes. The effectiveness of our baseline approach is shown to be a viable defense mechanism for real-world infrared human detection systems, paving the way for exploring future research directions.

摘要: 红外探测是一种新兴的安全关键任务检测技术，具有显著的抗干扰性。然而，最近的研究表明，它很容易受到物理上可实现的对抗性补丁的攻击，这给它在现实世界的应用带来了风险。针对这一问题，我们首次研究了针对红外探测，尤其是人体探测的对抗性补丁攻击的防御策略。我们提出了一种简单的防御策略，基于补丁的遮挡感知检测(POD)，它有效地利用随机补丁来增加训练样本并随后对其进行检测。Pod不仅可以稳健地检测人员，还可以识别敌方的补丁位置。令人惊讶的是，虽然POD在计算上非常高效，但它很容易概括为最先进的对抗性补丁攻击，这些攻击在训练中是看不到的。此外，由于数据增强效应，即使在干净(即，无攻击)的情况下，POD也提高了检测精度。我们的评估表明，POD对不同形状和大小的敌方补丁具有很强的鲁棒性。我们的基线方法的有效性被证明是一种可行的防御机制，用于真实世界的红外人体探测系统，为探索未来的研究方向铺平了道路。



## **9. Making Them Ask and Answer: Jailbreaking Large Language Models in Few Queries via Disguise and Reconstruction**

让他们问答：通过伪装和重建在很短的时间内越狱大型语言模型 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.18104v2) [paper-pdf](http://arxiv.org/pdf/2402.18104v2)

**Authors**: Tong Liu, Yingjie Zhang, Zhe Zhao, Yinpeng Dong, Guozhu Meng, Kai Chen

**Abstract**: In recent years, large language models (LLMs) have demonstrated notable success across various tasks, but the trustworthiness of LLMs is still an open problem. One specific threat is the potential to generate toxic or harmful responses. Attackers can craft adversarial prompts that induce harmful responses from LLMs. In this work, we pioneer a theoretical foundation in LLMs security by identifying bias vulnerabilities within the safety fine-tuning and design a black-box jailbreak method named DRA (Disguise and Reconstruction Attack), which conceals harmful instructions through disguise and prompts the model to reconstruct the original harmful instruction within its completion. We evaluate DRA across various open-source and closed-source models, showcasing state-of-the-art jailbreak success rates and attack efficiency. Notably, DRA boasts a 91.1% attack success rate on OpenAI GPT-4 chatbot.

摘要: 近年来，大型语言模型（LLM）在各种任务中取得了显着的成功，但LLM的可信度仍然是一个悬而未决的问题。一个具体的威胁是可能产生有毒或有害反应。攻击者可以设计对抗性提示，引发LLM的有害反应。在这项工作中，我们通过识别安全微调中的偏见漏洞，开创了LLM安全的理论基础，并设计了一种名为“伪装和重建攻击”的黑匣子越狱方法，通过伪装隐藏有害指令，并促使模型在完成时重建原始有害指令。我们评估各种开源和开源模型的NPS，展示最先进的越狱成功率和攻击效率。值得注意的是，Inbox对OpenAI GPT-4聊天机器人的攻击成功率为91.1%。



## **10. Texture Re-scalable Universal Adversarial Perturbation**

纹理可重新扩展的通用对抗扰动 cs.CV

14 pages (accepted by TIFS2024)

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.06089v1) [paper-pdf](http://arxiv.org/pdf/2406.06089v1)

**Authors**: Yihao Huang, Qing Guo, Felix Juefei-Xu, Ming Hu, Xiaojun Jia, Xiaochun Cao, Geguang Pu, Yang Liu

**Abstract**: Universal adversarial perturbation (UAP), also known as image-agnostic perturbation, is a fixed perturbation map that can fool the classifier with high probabilities on arbitrary images, making it more practical for attacking deep models in the real world. Previous UAP methods generate a scale-fixed and texture-fixed perturbation map for all images, which ignores the multi-scale objects in images and usually results in a low fooling ratio. Since the widely used convolution neural networks tend to classify objects according to semantic information stored in local textures, it seems a reasonable and intuitive way to improve the UAP from the perspective of utilizing local contents effectively. In this work, we find that the fooling ratios significantly increase when we add a constraint to encourage a small-scale UAP map and repeat it vertically and horizontally to fill the whole image domain. To this end, we propose texture scale-constrained UAP (TSC-UAP), a simple yet effective UAP enhancement method that automatically generates UAPs with category-specific local textures that can fool deep models more easily. Through a low-cost operation that restricts the texture scale, TSC-UAP achieves a considerable improvement in the fooling ratio and attack transferability for both data-dependent and data-free UAP methods. Experiments conducted on two state-of-the-art UAP methods, eight popular CNN models and four classical datasets show the remarkable performance of TSC-UAP.

摘要: 通用对抗摄动(UAP)，又称图像不可知摄动，是一种固定的摄动映射，可以在任意图像上以高概率欺骗分类器，使其更适用于攻击现实世界中的深层模型。以往的UAP方法为所有图像生成一个比例固定和纹理固定的扰动图，忽略了图像中的多尺度对象，通常会导致较低的欺骗率。由于广泛使用的卷积神经网络倾向于根据存储在局部纹理中的语义信息来对对象进行分类，从有效利用局部内容的角度来提高UAP似乎是一种合理而直观的方法。在这项工作中，我们发现，当我们添加一个约束来鼓励一个小比例的UAP地图并垂直和水平地重复它来填充整个图像域时，愚弄比率显著增加。为此，我们提出了纹理比例受限的UAP(TSC-UAP)，这是一种简单而有效的UAP增强方法，它自动生成具有特定类别局部纹理的UAP，从而更容易欺骗深层模型。TSC-UAP通过一种限制纹理规模的低成本操作，在依赖数据和无数据的UAP方法的欺骗比率和攻击可传递性方面都有了相当大的改善。在两种最新的UAP方法、8个流行的CNN模型和4个经典数据集上的实验表明，TSC-UAP具有显著的性能。



## **11. When Authentication Is Not Enough: On the Security of Behavioral-Based Driver Authentication Systems**

当认证还不够时：基于行为的驾驶员认证系统的安全性 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2306.05923v4) [paper-pdf](http://arxiv.org/pdf/2306.05923v4)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: Many research papers have recently focused on behavioral-based driver authentication systems in vehicles. Pushed by Artificial Intelligence (AI) advancements, these works propose powerful models to identify drivers through their unique biometric behavior. However, these models have never been scrutinized from a security point of view, rather focusing on the performance of the AI algorithms. Several limitations and oversights make implementing the state-of-the-art impractical, such as their secure connection to the vehicle's network and the management of security alerts. Furthermore, due to the extensive use of AI, these systems may be vulnerable to adversarial attacks. However, there is currently no discussion on the feasibility and impact of such attacks in this scenario.   Driven by the significant gap between research and practical application, this paper seeks to connect these two domains. We propose the first security-aware system model for behavioral-based driver authentication. We develop two lightweight driver authentication systems based on Random Forest and Recurrent Neural Network architectures designed for our constrained environments. We formalize a realistic system and threat model reflecting a real-world vehicle's network for their implementation. When evaluated on real driving data, our models outclass the state-of-the-art with an accuracy of up to 0.999 in identification and authentication. Moreover, we are the first to propose attacks against these systems by developing two novel evasion attacks, SMARTCAN and GANCAN. We show how attackers can still exploit these systems with a perfect attack success rate (up to 1.000). Finally, we discuss requirements for deploying driver authentication systems securely. Through our contributions, we aid practitioners in safely adopting these systems, help reduce car thefts, and enhance driver security.

摘要: 最近，许多研究论文都集中在基于行为的车辆驾驶员身份验证系统上。在人工智能(AI)进步的推动下，这些工作提出了强大的模型，通过司机独特的生物识别行为来识别他们。然而，这些模型从来没有从安全的角度进行过审查，而是专注于人工智能算法的性能。一些限制和疏忽使得实施最先进的技术不切实际，例如它们安全地连接到车辆的网络和安全警报的管理。此外，由于人工智能的广泛使用，这些系统可能容易受到对手攻击。然而，目前还没有关于这种情况下此类攻击的可行性和影响的讨论。在研究和实际应用之间的巨大差距的推动下，本文试图将这两个领域联系起来。提出了第一个基于行为的驾驶员身份认证的安全感知系统模型。我们开发了两个基于随机森林和递归神经网络架构的轻量级司机身份验证系统，这些架构是为我们的受限环境设计的。我们形式化了一个反映真实世界车辆网络的现实系统和威胁模型，以便实现它们。当在实际驾驶数据上进行评估时，我们的模型在识别和验证方面的准确率高达0.999，超过了最先进的模型。此外，我们还首次提出了针对这些系统的攻击，开发了两种新型的逃避攻击：Smartcan和Gancan。我们展示了攻击者如何仍然能够以完美的攻击成功率(高达1.000)利用这些系统。最后，我们将讨论安全部署驱动程序身份验证系统的要求。通过我们的贡献，我们帮助从业者安全地采用这些系统，帮助减少汽车盗窃，并提高司机的安全性。



## **12. A High Dimensional Statistical Model for Adversarial Training: Geometry and Trade-Offs**

对抗训练的多维统计模型：几何结构和权衡 stat.ML

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2402.05674v2) [paper-pdf](http://arxiv.org/pdf/2402.05674v2)

**Authors**: Kasimir Tanner, Matteo Vilucchio, Bruno Loureiro, Florent Krzakala

**Abstract**: This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust features during training, identifying a uniform protection as an inherently effective defence mechanism.

摘要: 该工作研究了高维环境下基于差值的线性分类器的对抗性训练，其中维度$d$和数据点数目$n$以固定的比率$\α=n/d$发散。我们引入了一个易于处理的数学模型，其中可以研究数据和敌意攻击者几何之间的相互作用，同时捕获在对抗性健壮性文献中观察到的核心现象学。我们的主要理论贡献是给出了一般凸损失和非增加损失下对抗性经验风险最小化充分统计量的精确渐近描述。我们的结果使我们能够准确地描述数据中的哪些方向与更高的泛化/稳健性权衡相关，如稳健性和有用性度量所定义的那样。特别是，我们揭示了方向的存在，这些方向可以在不影响准确性的情况下得到辩护。最后，我们展示了在训练过程中防御非健壮特征的优势，确定了统一保护作为一种内在有效的防御机制。



## **13. Safety Alignment Should Be Made More Than Just a Few Tokens Deep**

安全调整不应仅仅深入一些代币 cs.CR

**SubmitDate**: 2024-06-10    [abs](http://arxiv.org/abs/2406.05946v1) [paper-pdf](http://arxiv.org/pdf/2406.05946v1)

**Authors**: Xiangyu Qi, Ashwinee Panda, Kaifeng Lyu, Xiao Ma, Subhrajit Roy, Ahmad Beirami, Prateek Mittal, Peter Henderson

**Abstract**: The safety alignment of current Large Language Models (LLMs) is vulnerable. Relatively simple attacks, or even benign fine-tuning, can jailbreak aligned models. We argue that many of these vulnerabilities are related to a shared underlying issue: safety alignment can take shortcuts, wherein the alignment adapts a model's generative distribution primarily over only its very first few output tokens. We refer to this issue as shallow safety alignment. In this paper, we present case studies to explain why shallow safety alignment can exist and provide evidence that current aligned LLMs are subject to this issue. We also show how these findings help explain multiple recently discovered vulnerabilities in LLMs, including the susceptibility to adversarial suffix attacks, prefilling attacks, decoding parameter attacks, and fine-tuning attacks. Importantly, we discuss how this consolidated notion of shallow safety alignment sheds light on promising research directions for mitigating these vulnerabilities. For instance, we show that deepening the safety alignment beyond just the first few tokens can often meaningfully improve robustness against some common exploits. Finally, we design a regularized finetuning objective that makes the safety alignment more persistent against fine-tuning attacks by constraining updates on initial tokens. Overall, we advocate that future safety alignment should be made more than just a few tokens deep.

摘要: 当前大型语言模型(LLM)的安全对齐是易受攻击的。相对简单的攻击，甚至是温和的微调，都可以让结盟的模型越狱。我们认为，这些漏洞中的许多都与一个共同的潜在问题有关：安全对齐可以走捷径，其中对齐主要适应模型的生成性分布，仅在其最初的几个输出令牌上。我们将这个问题称为浅层安全对准。在这篇文章中，我们提供了案例研究来解释为什么浅层安全对准可以存在，并提供证据表明当前对准的LLM受到这个问题的影响。我们还展示了这些发现如何帮助解释LLMS中最近发现的多个漏洞，包括对敌意后缀攻击、预填充攻击、解码参数攻击和微调攻击的敏感性。重要的是，我们讨论了浅层安全对齐这一统一概念如何揭示了缓解这些漏洞的有前途的研究方向。例如，我们表明，除了最初的几个令牌之外，深化安全对齐通常可以有意义地提高对一些常见漏洞的健壮性。最后，我们设计了一个正则化的精调目标，通过限制对初始令牌的更新，使安全对齐更持久地抵抗微调攻击。总体而言，我们主张未来的安全调整应该不仅仅是几个标志的深度。



## **14. A Relevance Model for Threat-Centric Ranking of Cybersecurity Vulnerabilities**

以威胁为中心的网络安全漏洞排名的相关模型 cs.CR

24 pages, 8 figures, 14 tables

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05933v1) [paper-pdf](http://arxiv.org/pdf/2406.05933v1)

**Authors**: Corren McCoy, Ross Gore, Michael L. Nelson, Michele C. Weigle

**Abstract**: The relentless process of tracking and remediating vulnerabilities is a top concern for cybersecurity professionals. The key challenge is trying to identify a remediation scheme specific to in-house, organizational objectives. Without a strategy, the result is a patchwork of fixes applied to a tide of vulnerabilities, any one of which could be the point of failure in an otherwise formidable defense. Given that few vulnerabilities are a focus of real-world attacks, a practical remediation strategy is to identify vulnerabilities likely to be exploited and focus efforts towards remediating those vulnerabilities first. The goal of this research is to demonstrate that aggregating and synthesizing readily accessible, public data sources to provide personalized, automated recommendations for organizations to prioritize their vulnerability management strategy will offer significant improvements over using the Common Vulnerability Scoring System (CVSS). We provide a framework for vulnerability management specifically focused on mitigating threats using adversary criteria derived from MITRE ATT&CK. We test our approach by identifying vulnerabilities in software associated with six universities and four government facilities. Ranking policy performance is measured using the Normalized Discounted Cumulative Gain (nDCG). Our results show an average 71.5% - 91.3% improvement towards the identification of vulnerabilities likely to be targeted and exploited by cyber threat actors. The return on investment (ROI) of patching using our policies results in a savings of 23.3% - 25.5% in annualized costs. Our results demonstrate the efficacy of creating knowledge graphs to link large data sets to facilitate semantic queries and create data-driven, flexible ranking policies.

摘要: 无情的漏洞跟踪和修复过程是网络安全专业人士最关心的问题。关键的挑战是试图确定一个专门针对内部组织目标的补救方案。如果没有战略，结果是对大量漏洞进行拼凑的修复，其中任何一个都可能成为原本令人敬畏的防御措施的失败点。鉴于很少有漏洞是现实世界攻击的焦点，一个实用的补救策略是识别可能被利用的漏洞，并将重点放在首先补救这些漏洞上。这项研究的目的是证明，聚合和综合易于访问的公共数据源，为组织提供个性化、自动化的建议，以确定其漏洞管理战略的优先顺序，将比使用通用漏洞评分系统(CVSS)提供显著改进。我们使用从MITRE ATT&CK派生的敌意标准，提供了一个专门针对缓解威胁的漏洞管理框架。我们通过识别与六所大学和四个政府机构相关的软件中的漏洞来测试我们的方法。排名策略绩效使用归一化贴现累积收益(NDCG)来衡量。我们的结果显示，在识别可能被网络威胁参与者瞄准和利用的漏洞方面，平均提高了71.5%-91.3%。使用我们的策略进行修补的投资回报率(ROI)可节省23.3%-25.5%的年化成本。我们的结果证明了创建知识图来链接大数据集以促进语义查询和创建数据驱动的、灵活的排名策略的有效性。



## **15. MeanSparse: Post-Training Robustness Enhancement Through Mean-Centered Feature Sparsification**

MeanSparse：通过以均值为中心的特征稀疏化来增强训练后的鲁棒性 cs.CV

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05927v1) [paper-pdf](http://arxiv.org/pdf/2406.05927v1)

**Authors**: Sajjad Amini, Mohammadreza Teymoorianfard, Shiqing Ma, Amir Houmansadr

**Abstract**: We present a simple yet effective method to improve the robustness of Convolutional Neural Networks (CNNs) against adversarial examples by post-processing an adversarially trained model. Our technique, MeanSparse, cascades the activation functions of a trained model with novel operators that sparsify mean-centered feature vectors. This is equivalent to reducing feature variations around the mean, and we show that such reduced variations merely affect the model's utility, yet they strongly attenuate the adversarial perturbations and decrease the attacker's success rate. Our experiments show that, when applied to the top models in the RobustBench leaderboard, it achieves a new robustness record of 72.08% (from 71.07%) and 59.64% (from 59.56%) on CIFAR-10 and ImageNet, respectively, in term of AutoAttack accuracy. Code is available at https://github.com/SPIN-UMass/MeanSparse

摘要: 我们提出了一种简单而有效的方法，通过后处理对抗训练的模型来提高卷积神经网络（CNN）对对抗示例的鲁棒性。我们的技术MeanSparse通过新颖的运算符级联经过训练的模型的激活函数，这些运算符稀疏化以均值为中心的特征载体。这相当于减少均值附近的特征变化，我们表明，这种减少的变化只会影响模型的效用，但它们会强烈削弱对抗性扰动并降低攻击者的成功率。我们的实验表明，当应用于RobustBench排行榜上的顶级模型时，在AutoAttack准确性方面，它在CIFAR-10和ImageNet上分别实现了72.08%（从71.07%开始）和59.64%（从59.56%开始）的新稳健性记录。代码可访问https://github.com/SPIN-UMass/MeanSparse



## **16. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

机器对抗RAG：用阻止器文档干扰检索增强生成 cs.CR

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05870v1) [paper-pdf](http://arxiv.org/pdf/2406.05870v1)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database, then generating an answer by applying an LLM to the retrieved documents.   We demonstrate that RAG systems that operate on databases with potentially untrusted content are vulnerable to a new class of denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and, furthermore, result in the RAG system not answering the query - ostensibly because it lacks the information or because the answer is unsafe.   We describe and analyze several methods for generating blocker documents, including a new method based on black-box optimization that does not require the adversary to know the embedding or LLM used by the target RAG system, nor access to an auxiliary LLM to generate blocker documents. We measure the efficacy of the considered methods against several LLMs and embeddings, and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.

摘要: 检索-增强生成(RAG)系统通过从知识数据库中检索相关文档，然后通过将LLM应用于所检索的文档来生成答案来响应查询。我们证明，在含有潜在不可信内容的数据库上运行的RAG系统容易受到一种新的拒绝服务攻击，我们称之为干扰。敌手可以在数据库中添加一个“拦截器”文档，该文档将响应于特定查询而被检索，并进一步导致RAG系统不回答查询--表面上是因为它缺乏信息或因为答案不安全。我们描述和分析了几种生成拦截器文档的方法，包括一种基于黑盒优化的新方法，该方法不需要攻击者知道目标RAG系统使用的嵌入或LLM，也不需要访问辅助LLM来生成拦截器文档。我们测量了所考虑的方法在几个LLM和嵌入上的有效性，并证明了现有的LLM的安全度量不能捕捉到它们对干扰的脆弱性。然后我们讨论针对拦截器文档的防御。



## **17. Self-supervised Adversarial Training of Monocular Depth Estimation against Physical-World Attacks**

针对物理世界攻击的单目深度估计的自我监督对抗训练 cs.CV

Accepted in TPAMI'24. Extended from our ICLR'23 publication  (arXiv:2301.13487). arXiv admin note: substantial text overlap with  arXiv:2301.13487

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05857v1) [paper-pdf](http://arxiv.org/pdf/2406.05857v1)

**Authors**: Zhiyuan Cheng, Cheng Han, James Liang, Qifan Wang, Xiangyu Zhang, Dongfang Liu

**Abstract**: Monocular Depth Estimation (MDE) plays a vital role in applications such as autonomous driving. However, various attacks target MDE models, with physical attacks posing significant threats to system security. Traditional adversarial training methods, which require ground-truth labels, are not directly applicable to MDE models that lack ground-truth depth. Some self-supervised model hardening techniques (e.g., contrastive learning) overlook the domain knowledge of MDE, resulting in suboptimal performance. In this work, we introduce a novel self-supervised adversarial training approach for MDE models, leveraging view synthesis without the need for ground-truth depth. We enhance adversarial robustness against real-world attacks by incorporating L_0-norm-bounded perturbation during training. We evaluate our method against supervised learning-based and contrastive learning-based approaches specifically designed for MDE. Our experiments with two representative MDE networks demonstrate improved robustness against various adversarial attacks, with minimal impact on benign performance.

摘要: 单目深度估计(MDE)在自动驾驶等应用中起着至关重要的作用。然而，各种攻击针对的是MDE模型，其中物理攻击对系统安全构成了重大威胁。传统的对抗性训练方法需要地面真相标签，不能直接适用于缺乏地面真相深度的MDE模型。一些自监督模型硬化技术(如对比学习)忽略了MDE的领域知识，导致性能不佳。在这项工作中，我们介绍了一种新的自我监督的MDE模型对抗性训练方法，利用视图合成而不需要地面真实深度。通过在训练过程中引入L_0范数有界扰动来增强对手对真实世界攻击的健壮性。我们用基于监督学习的方法和专门为MDE设计的基于对比学习的方法来评估我们的方法。我们用两个典型的MDE网络进行的实验表明，在对良性性能影响最小的情况下，提高了对各种敌意攻击的稳健性。



## **18. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2402.06255v2) [paper-pdf](http://arxiv.org/pdf/2402.06255v2)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both black-box and white-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在有害内容过滤或启发式防御提示设计上。然而，如何通过提示实现内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对黑盒攻击和白盒攻击都是有效的，在保持模型对良性任务的实用性的同时，将高级攻击的成功率降低到近0。所提出的防御策略只需要很少的计算开销，为未来在LLM安全方面的探索开辟了新的前景。我们的代码可以在https://github.com/rain152/PAT.上找到



## **19. PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection**

PSBD：预测转变不确定性解锁后门检测 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05826v1) [paper-pdf](http://arxiv.org/pdf/2406.05826v1)

**Authors**: Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang

**Abstract**: Deep neural networks are susceptible to backdoor attacks, where adversaries manipulate model predictions by inserting malicious samples into the training data. Currently, there is still a lack of direct filtering methods for identifying suspicious training data to unveil potential backdoor samples. In this paper, we propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. PSBD is motivated by an intriguing Prediction Shift (PS) phenomenon, where poisoned models' predictions on clean data often shift away from true labels towards certain other labels with dropout applied during inference, while backdoor samples exhibit less PS. We hypothesize PS results from neuron bias effect, making neurons favor features of certain classes. PSBD identifies backdoor training samples by computing the Prediction Shift Uncertainty (PSU), the variance in probability values when dropout layers are toggled on and off during model inference. Extensive experiments have been conducted to verify the effectiveness and efficiency of PSBD, which achieves state-of-the-art results among mainstream detection methods.

摘要: 深度神经网络很容易受到后门攻击，对手通过在训练数据中插入恶意样本来操纵模型预测。目前，仍然缺乏识别可疑培训数据以揭示潜在后门样本的直接过滤方法。在本文中，我们提出了一种新的方法，预测移位后门检测(PSBD)，利用了一种基于不确定性的方法，需要最少的未标记干净验证数据。PSBD的动机是一种有趣的预测漂移(PS)现象，在这种现象中，有毒模型对干净数据的预测经常从真实标签转移到其他某些标签，并在推断过程中应用丢弃，而借壳样本显示较少的PS。我们假设PS是神经元偏向效应的结果，使神经元偏爱某些类别的特征。PSBD通过计算预测漂移不确定性(PSU)来识别后门训练样本，PSU是在模型推理过程中切换退出层时概率值的方差。大量的实验验证了PSBD的有效性和效率，在主流检测方法中取得了最先进的结果。



## **20. ControlLoc: Physical-World Hijacking Attack on Visual Perception in Autonomous Driving**

Control Loc：自动驾驶中视觉感知的物理世界劫持攻击 cs.CV

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05810v1) [paper-pdf](http://arxiv.org/pdf/2406.05810v1)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qian Wang, Qi Alfred Chen, Chao Shen

**Abstract**: Recent research in adversarial machine learning has focused on visual perception in Autonomous Driving (AD) and has shown that printed adversarial patches can attack object detectors. However, it is important to note that AD visual perception encompasses more than just object detection; it also includes Multiple Object Tracking (MOT). MOT enhances the robustness by compensating for object detection errors and requiring consistent object detection results across multiple frames before influencing tracking results and driving decisions. Thus, MOT makes attacks on object detection alone less effective. To attack such robust AD visual perception, a digital hijacking attack has been proposed to cause dangerous driving scenarios. However, this attack has limited effectiveness.   In this paper, we introduce a novel physical-world adversarial patch attack, ControlLoc, designed to exploit hijacking vulnerabilities in entire AD visual perception. ControlLoc utilizes a two-stage process: initially identifying the optimal location for the adversarial patch, and subsequently generating the patch that can modify the perceived location and shape of objects with the optimal location. Extensive evaluations demonstrate the superior performance of ControlLoc, achieving an impressive average attack success rate of around 98.1% across various AD visual perceptions and datasets, which is four times greater effectiveness than the existing hijacking attack. The effectiveness of ControlLoc is further validated in physical-world conditions, including real vehicle tests under different conditions such as outdoor light conditions with an average attack success rate of 77.5%. AD system-level impact assessments are also included, such as vehicle collision, using industry-grade AD systems and production-grade AD simulators with an average vehicle collision rate and unnecessary emergency stop rate of 81.3%.

摘要: 最近对抗性机器学习的研究集中在自动驾驶(AD)中的视觉感知，并表明打印的对抗性补丁可以攻击对象检测器。然而，重要的是要注意，AD视觉感知不仅包括对象检测；它还包括多对象跟踪(MOT)。MOT通过补偿目标检测错误并要求在影响跟踪结果和驱动决策之前在多个帧上获得一致的目标检测结果来增强稳健性。因此，MOT使得仅针对对象检测的攻击效果较差。为了攻击这种强大的广告视觉感知，人们提出了一种数字劫持攻击，以造成危险的驾驶场景。然而，这次袭击的效果有限。在本文中，我们介绍了一种新的物理世界对抗性补丁攻击，ControlLoc，旨在利用整个AD视觉感知中的劫持漏洞。ControlLoc使用两个阶段的过程：首先确定对抗性补丁的最佳位置，然后生成可以修改具有最佳位置的对象的感知位置和形状的补丁。广泛的评估显示了ControlLoc的卓越性能，在各种AD视觉感知和数据集上实现了令人印象深刻的平均攻击成功率约98.1%，比现有的劫持攻击效率高出四倍。在物理世界条件下进一步验证了ControlLoc的有效性，包括在不同条件下的真实车辆测试，如室外光照条件，平均攻击成功率为77.5%。AD系统级影响评估也包括在内，例如车辆碰撞，使用工业级AD系统和生产级AD模拟器进行评估，平均车辆碰撞率和不必要的紧急停机率为81.3%。



## **21. SlowPerception: Physical-World Latency Attack against Visual Perception in Autonomous Driving**

慢感知：自动驾驶中对视觉感知的物理世界延迟攻击 cs.CV

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05800v1) [paper-pdf](http://arxiv.org/pdf/2406.05800v1)

**Authors**: Chen Ma, Ningfei Wang, Zhengyu Zhao, Qi Alfred Chen, Chao Shen

**Abstract**: Autonomous Driving (AD) systems critically depend on visual perception for real-time object detection and multiple object tracking (MOT) to ensure safe driving. However, high latency in these visual perception components can lead to significant safety risks, such as vehicle collisions. While previous research has extensively explored latency attacks within the digital realm, translating these methods effectively to the physical world presents challenges. For instance, existing attacks rely on perturbations that are unrealistic or impractical for AD, such as adversarial perturbations affecting areas like the sky, or requiring large patches that obscure most of a camera's view, thus making them impossible to be conducted effectively in the real world.   In this paper, we introduce SlowPerception, the first physical-world latency attack against AD perception, via generating projector-based universal perturbations. SlowPerception strategically creates numerous phantom objects on various surfaces in the environment, significantly increasing the computational load of Non-Maximum Suppression (NMS) and MOT, thereby inducing substantial latency. Our SlowPerception achieves second-level latency in physical-world settings, with an average latency of 2.5 seconds across different AD perception systems, scenarios, and hardware configurations. This performance significantly outperforms existing state-of-the-art latency attacks. Additionally, we conduct AD system-level impact assessments, such as vehicle collisions, using industry-grade AD systems with production-grade AD simulators with a 97% average rate. We hope that our analyses can inspire further research in this critical domain, enhancing the robustness of AD systems against emerging vulnerabilities.

摘要: 自动驾驶(AD)系统在很大程度上依赖于视觉感知进行实时目标检测和多目标跟踪(MOT)来确保安全驾驶。然而，这些视觉感知组件的高延迟可能会导致重大安全风险，如车辆碰撞。虽然之前的研究已经广泛地探索了数字领域内的延迟攻击，但将这些方法有效地转换到物理世界是一项挑战。例如，现有的攻击依赖于对AD来说不现实或不切实际的扰动，例如影响天空等区域的对抗性扰动，或者需要遮挡大部分摄像机视野的大补丁，从而使它们不可能在现实世界中有效地进行。在本文中，我们通过产生基于投影仪的普遍扰动，引入了第一个针对AD感知的物理世界延迟攻击SlowPercept。SlowPercept战略性地在环境中的不同表面创建大量幻影对象，显著增加非最大抑制(NMS)和MOT的计算负荷，从而导致显著的延迟。我们的SlowPercept在物理世界设置中实现了二级延迟，跨不同AD感知系统、场景和硬件配置的平均延迟为2.5秒。这一性能大大超过了现有最先进的延迟攻击。此外，我们使用带有生产级AD模拟器的工业级AD系统进行AD系统级影响评估，例如车辆碰撞，平均比率为97%。我们希望我们的分析能够启发这一关键领域的进一步研究，增强AD系统对新出现的漏洞的健壮性。



## **22. ProFeAT: Projected Feature Adversarial Training for Self-Supervised Learning of Robust Representations**

ProFeAT：用于鲁棒表示自我监督学习的投影特征对抗训练 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05796v1) [paper-pdf](http://arxiv.org/pdf/2406.05796v1)

**Authors**: Sravanti Addepalli, Priyam Dey, R. Venkatesh Babu

**Abstract**: The need for abundant labelled data in supervised Adversarial Training (AT) has prompted the use of Self-Supervised Learning (SSL) techniques with AT. However, the direct application of existing SSL methods to adversarial training has been sub-optimal due to the increased training complexity of combining SSL with AT. A recent approach, DeACL, mitigates this by utilizing supervision from a standard SSL teacher in a distillation setting, to mimic supervised AT. However, we find that there is still a large performance gap when compared to supervised adversarial training, specifically on larger models. In this work, investigate the key reason for this gap and propose Projected Feature Adversarial Training (ProFeAT) to bridge the same. We show that the sub-optimal distillation performance is a result of mismatch in training objectives of the teacher and student, and propose to use a projection head at the student, that allows it to leverage weak supervision from the teacher while also being able to learn adversarially robust representations that are distinct from the teacher. We further propose appropriate attack and defense losses at the feature and projector, alongside a combination of weak and strong augmentations for the teacher and student respectively, to improve the training data diversity without increasing the training complexity. Through extensive experiments on several benchmark datasets and models, we demonstrate significant improvements in both clean and robust accuracy when compared to existing SSL-AT methods, setting a new state-of-the-art. We further report on-par/ improved performance when compared to TRADES, a popular supervised-AT method.

摘要: 在有监督的对抗性训练(AT)中需要大量的标签数据，这促使了在AT中使用自我监督学习(SSL)技术。然而，由于将SSL与AT相结合增加了训练复杂性，现有的SSL方法在对抗性训练中的直接应用一直是次优的。最近的一种方法，DeACL，通过在蒸馏设置中利用标准SSL教师的监督来模拟受监督的AT来缓解这种情况。然而，我们发现，与监督对抗性训练相比，尤其是在较大的模型上，仍然存在很大的性能差距。在这项工作中，调查这一差距的关键原因，并提出预测特征对抗性训练(ProFeAT)来弥合这一差距。我们证明了次优的蒸馏性能是由于教师和学生的培训目标不匹配的结果，并建议在学生身上使用投影头，这使得它能够利用教师的弱监督，同时也能够学习与教师不同的相反的稳健表示。为了在不增加训练复杂度的情况下提高训练数据的多样性，我们进一步提出了在特征和投影器上适当的攻击和防御损失，以及分别针对教师和学生的弱增强和强增强的组合。通过在几个基准数据集和模型上的广泛实验，我们证明了与现有的SSL-AT方法相比，我们在干净和健壮的准确性方面都有了显著的改进，创下了新的艺术水平。我们进一步报告了与广受监督的AT方法TRADS相比，性能持平/改进的情况。



## **23. Injecting Undetectable Backdoors in Deep Learning and Language Models**

在深度学习和语言模型中注入无法检测的后门 cs.LG

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2406.05660v1) [paper-pdf](http://arxiv.org/pdf/2406.05660v1)

**Authors**: Alkis Kalavasis, Amin Karbasi, Argyris Oikonomou, Katerina Sotiraki, Grigoris Velegkas, Manolis Zampetakis

**Abstract**: As ML models become increasingly complex and integral to high-stakes domains such as finance and healthcare, they also become more susceptible to sophisticated adversarial attacks. We investigate the threat posed by undetectable backdoors in models developed by insidious external expert firms. When such backdoors exist, they allow the designer of the model to sell information to the users on how to carefully perturb the least significant bits of their input to change the classification outcome to a favorable one. We develop a general strategy to plant a backdoor to neural networks while ensuring that even if the model's weights and architecture are accessible, the existence of the backdoor is still undetectable. To achieve this, we utilize techniques from cryptography such as cryptographic signatures and indistinguishability obfuscation. We further introduce the notion of undetectable backdoors to language models and extend our neural network backdoor attacks to such models based on the existence of steganographic functions.

摘要: 随着ML模型变得越来越复杂，成为金融和医疗等高风险领域不可或缺的一部分，它们也变得更容易受到复杂的对抗性攻击。我们在潜伏的外部专家公司开发的模型中调查了不可检测的后门构成的威胁。当存在这样的后门时，它们允许模型的设计者向用户出售信息，告诉用户如何小心地扰乱他们输入的最低有效位，以将分类结果更改为有利的结果。我们开发了一个通用策略来植入神经网络的后门，同时确保即使模型的权重和体系结构是可访问的，后门的存在仍然是不可检测的。为了实现这一点，我们利用了密码学中的技术，如密码签名和不可区分混淆。我们进一步将不可检测后门的概念引入到语言模型中，并基于隐写函数的存在将神经网络后门攻击扩展到这类模型。



## **24. Fooling the Textual Fooler via Randomizing Latent Representations**

通过随机化潜在表示来愚弄文本预见 cs.CL

Accepted to Findings of ACL 2024

**SubmitDate**: 2024-06-09    [abs](http://arxiv.org/abs/2310.01452v2) [paper-pdf](http://arxiv.org/pdf/2310.01452v2)

**Authors**: Duy C. Hoang, Quang H. Nguyen, Saurav Manchanda, MinLong Peng, Kok-Seng Wong, Khoa D. Doan

**Abstract**: Despite outstanding performance in a variety of NLP tasks, recent studies have revealed that NLP models are vulnerable to adversarial attacks that slightly perturb the input to cause the models to misbehave. Among these attacks, adversarial word-level perturbations are well-studied and effective attack strategies. Since these attacks work in black-box settings, they do not require access to the model architecture or model parameters and thus can be detrimental to existing NLP applications. To perform an attack, the adversary queries the victim model many times to determine the most important words in an input text and to replace these words with their corresponding synonyms. In this work, we propose a lightweight and attack-agnostic defense whose main goal is to perplex the process of generating an adversarial example in these query-based black-box attacks; that is to fool the textual fooler. This defense, named AdvFooler, works by randomizing the latent representation of the input at inference time. Different from existing defenses, AdvFooler does not necessitate additional computational overhead during training nor relies on assumptions about the potential adversarial perturbation set while having a negligible impact on the model's accuracy. Our theoretical and empirical analyses highlight the significance of robustness resulting from confusing the adversary via randomizing the latent space, as well as the impact of randomization on clean accuracy. Finally, we empirically demonstrate near state-of-the-art robustness of AdvFooler against representative adversarial word-level attacks on two benchmark datasets.

摘要: 尽管在各种自然语言处理任务中表现出色，但最近的研究表明，自然语言处理模型容易受到对抗性攻击，这些攻击会轻微扰乱输入，导致模型行为不当。在这些攻击中，对抗性词级扰动是研究较多、效果较好的攻击策略。由于这些攻击在黑盒设置中工作，因此它们不需要访问模型体系结构或模型参数，因此可能对现有的NLP应用程序有害。为了执行攻击，对手多次查询受害者模型，以确定输入文本中最重要的单词，并用它们对应的同义词替换这些单词。在这项工作中，我们提出了一个轻量级和攻击不可知的防御，其主要目标是在这些基于查询的黑盒攻击中迷惑生成敌对示例的过程，即愚弄文本傻瓜。这种防御被称为AdvFooler，其工作原理是在推理时随机化输入的潜在表示。与现有的防御方法不同，AdvFooler在训练过程中不需要额外的计算开销，也不依赖于对潜在对手扰动集的假设，同时对模型的精度影响可以忽略不计。我们的理论和经验分析强调了通过随机化潜在空间来迷惑对手而产生的稳健性的重要性，以及随机化对干净精度的影响。最后，我们在两个基准数据集上通过实验验证了AdvFooler对典型的对抗性词级攻击的近乎最先进的健壮性。



## **25. Perturbation Towards Easy Samples Improves Targeted Adversarial Transferability**

对简单样本的扰动提高了有针对性的对抗转移能力 cs.LG

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05535v1) [paper-pdf](http://arxiv.org/pdf/2406.05535v1)

**Authors**: Junqi Gao, Biqing Qi, Yao Li, Zhichang Guo, Dong Li, Yuming Xing, Dazhi Zhang

**Abstract**: The transferability of adversarial perturbations provides an effective shortcut for black-box attacks. Targeted perturbations have greater practicality but are more difficult to transfer between models. In this paper, we experimentally and theoretically demonstrated that neural networks trained on the same dataset have more consistent performance in High-Sample-Density-Regions (HSDR) of each class instead of low sample density regions. Therefore, in the target setting, adding perturbations towards HSDR of the target class is more effective in improving transferability. However, density estimation is challenging in high-dimensional scenarios. Further theoretical and experimental verification demonstrates that easy samples with low loss are more likely to be located in HSDR. Perturbations towards such easy samples in the target class can avoid density estimation for HSDR location. Based on the above facts, we verified that adding perturbations to easy samples in the target class improves targeted adversarial transferability of existing attack methods. A generative targeted attack strategy named Easy Sample Matching Attack (ESMA) is proposed, which has a higher success rate for targeted attacks and outperforms the SOTA generative method. Moreover, ESMA requires only 5% of the storage space and much less computation time comparing to the current SOTA, as ESMA attacks all classes with only one model instead of seperate models for each class. Our code is available at https://github.com/gjq100/ESMA.

摘要: 对抗性扰动的可转移性为黑盒攻击提供了一条有效的捷径。有针对性的摄动具有更大的实用性，但更难在模型之间传递。在本文中，我们从实验和理论上证明了在同一数据集上训练的神经网络在每一类的高样本密度区域(HSDR)比在低样本密度区域具有更一致的性能。因此，在目标设置中，增加对目标类别的HSDR的扰动在提高可转移性方面更有效。然而，密度估计在高维场景中是具有挑战性的。进一步的理论和实验验证表明，低损耗的简单样品更有可能位于高速SDR中。对于目标类中这样容易的样本的扰动可以避免用于HSDR定位的密度估计。基于上述事实，我们验证了在目标类的简单样本中添加扰动提高了现有攻击方法的目标对抗性转移能力。提出了一种生成式目标攻击策略--简单样本匹配攻击(ESMA)，该策略对目标攻击具有较高的成功率，并优于SOTA生成法。此外，与当前的SOTA相比，ESMA只需要5%的存储空间和更少的计算时间，因为ESMA只使用一个模型攻击所有类，而不是每个类使用单独的模型。我们的代码可以在https://github.com/gjq100/ESMA.上找到



## **26. Enhancing Adversarial Transferability via Information Bottleneck Constraints**

通过信息瓶颈约束增强对抗性可转让性 cs.LG

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05531v1) [paper-pdf](http://arxiv.org/pdf/2406.05531v1)

**Authors**: Biqing Qi, Junqi Gao, Jianxing Liu, Ligang Wu, Bowen Zhou

**Abstract**: From the perspective of information bottleneck (IB) theory, we propose a novel framework for performing black-box transferable adversarial attacks named IBTA, which leverages advancements in invariant features. Intuitively, diminishing the reliance of adversarial perturbations on the original data, under equivalent attack performance constraints, encourages a greater reliance on invariant features that contributes most to classification, thereby enhancing the transferability of adversarial attacks. Building on this motivation, we redefine the optimization of transferable attacks using a novel theoretical framework that centers around IB. Specifically, to overcome the challenge of unoptimizable mutual information, we propose a simple and efficient mutual information lower bound (MILB) for approximating computation. Moreover, to quantitatively evaluate mutual information, we utilize the Mutual Information Neural Estimator (MINE) to perform a thorough analysis. Our experiments on the ImageNet dataset well demonstrate the efficiency and scalability of IBTA and derived MILB. Our code is available at https://github.com/Biqing-Qi/Enhancing-Adversarial-Transferability-via-Information-Bottleneck-Constraints.

摘要: 从信息瓶颈(IB)理论的角度出发，利用不变特征的优势，提出了一种新的黑盒可转移对抗攻击框架IBTA。直观地说，在等价的攻击性能约束下，减少对抗性扰动对原始数据的依赖，鼓励更多地依赖对分类贡献最大的不变特征，从而增强对抗性攻击的可转移性。在这一动机的基础上，我们使用以IB为中心的新理论框架重新定义了可转移攻击的优化。具体地说，为了克服互信息不可优化的挑战，我们提出了一种简单有效的互信息下界(MILB)来近似计算。此外，为了定量评估互信息，我们利用互信息神经估计器(MIME)进行了深入的分析。我们在ImageNet数据集上的实验很好地证明了IBTA和派生的MILB算法的有效性和可扩展性。我们的代码可以在https://github.com/Biqing-Qi/Enhancing-Adversarial-Transferability-via-Information-Bottleneck-Constraints.上找到



## **27. The Perception-Robustness Tradeoff in Deterministic Image Restoration**

确定性图像恢复中的感知与鲁棒性权衡 eess.IV

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2311.09253v4) [paper-pdf](http://arxiv.org/pdf/2311.09253v4)

**Authors**: Guy Ohayon, Tomer Michaeli, Michael Elad

**Abstract**: We study the behavior of deterministic methods for solving inverse problems in imaging. These methods are commonly designed to achieve two goals: (1) attaining high perceptual quality, and (2) generating reconstructions that are consistent with the measurements. We provide a rigorous proof that the better a predictor satisfies these two requirements, the larger its Lipschitz constant must be, regardless of the nature of the degradation involved. In particular, to approach perfect perceptual quality and perfect consistency, the Lipschitz constant of the model must grow to infinity. This implies that such methods are necessarily more susceptible to adversarial attacks. We demonstrate our theory on single image super-resolution algorithms, addressing both noisy and noiseless settings. We also show how this undesired behavior can be leveraged to explore the posterior distribution, thereby allowing the deterministic model to imitate stochastic methods.

摘要: 我们研究解决成像反问题的确定性方法的行为。这些方法通常旨在实现两个目标：（1）获得高感知质量，以及（2）生成与测量结果一致的重建。我们提供了一个严格的证据，证明预测器满足这两个要求越好，其利普希茨常数就必须越大，无论所涉及的退化的性质如何。特别是，为了达到完美的感知质量和完美的一致性，模型的利普希茨常数必须增长到无穷大。这意味着此类方法必然更容易受到对抗攻击。我们展示了我们关于单图像超分辨率算法的理论，解决有噪和无噪设置。我们还展示了如何利用这种不受欢迎的行为来探索后验分布，从而允许确定性模型模仿随机方法。



## **28. SelfDefend: LLMs Can Defend Themselves against Jailbreaking in a Practical Manner**

SelfDefend：LLM可以以实用的方式保护自己免受越狱的侵害 cs.CR

This paper completes its earlier vision paper, available at  arXiv:2402.15727

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05498v1) [paper-pdf](http://arxiv.org/pdf/2406.05498v1)

**Authors**: Xunguang Wang, Daoyuan Wu, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Shuai Wang, Yingjiu Li, Yang Liu, Ning Liu, Juergen Rahmel

**Abstract**: Jailbreaking is an emerging adversarial attack that bypasses the safety alignment deployed in off-the-shelf large language models (LLMs) and has evolved into four major categories: optimization-based attacks such as Greedy Coordinate Gradient (GCG), jailbreak template-based attacks such as "Do-Anything-Now", advanced indirect attacks like DrAttack, and multilingual jailbreaks. However, delivering a practical jailbreak defense is challenging because it needs to not only handle all the above jailbreak attacks but also incur negligible delay to user prompts, as well as be compatible with both open-source and closed-source LLMs.   Inspired by how the traditional security concept of shadow stacks defends against memory overflow attacks, this paper introduces a generic LLM jailbreak defense framework called SelfDefend, which establishes a shadow LLM defense instance to concurrently protect the target LLM instance in the normal stack and collaborate with it for checkpoint-based access control. The effectiveness of SelfDefend builds upon our observation that existing LLMs (both target and defense LLMs) have the capability to identify harmful prompts or intentions in user queries, which we empirically validate using the commonly used GPT-3.5/4 models across all major jailbreak attacks. Our measurements show that SelfDefend enables GPT-3.5 to suppress the attack success rate (ASR) by 8.97-95.74% (average: 60%) and GPT-4 by even 36.36-100% (average: 83%), while incurring negligible effects on normal queries. To further improve the defense's robustness and minimize costs, we employ a data distillation approach to tune dedicated open-source defense models. These models outperform four SOTA defenses and match the performance of GPT-4-based SelfDefend, with significantly lower extra delays. We also empirically show that the tuned models are robust to targeted GCG and prompt injection attacks.

摘要: 越狱是一种新兴的对抗性攻击，它绕过了现成的大型语言模型(LLM)中部署的安全对齐，并已演变为四大类：基于优化的攻击(如贪婪坐标梯度(GCG))、基于模板的攻击(如Do-Anything-Now)、高级间接攻击(如DrAttack)和多语言越狱。然而，提供实际的越狱防御是具有挑战性的，因为它不仅需要处理所有上述越狱攻击，还需要对用户提示造成可以忽略不计的延迟，以及与开源和闭源LLM兼容。受传统影子堆栈安全概念防御内存溢出攻击的启发，提出了一种通用的LLM越狱防御框架--SelfDefend，该框架通过建立一个影子LLM防御实例来同时保护正常堆栈中的目标LLM实例，并与其协作进行基于检查点的访问控制。SelfDefend的有效性建立在我们的观察基础上，即现有的LLM(目标和防御LLM)能够识别用户查询中的有害提示或意图，我们使用所有主要越狱攻击中常用的GPT-3.5/4模型进行了经验验证。我们的测试表明，SelfDefend使GPT-3.5的攻击成功率(ASR)降低了8.97-95.74%(平均：60%)，GPT-4的攻击成功率(ASR)降低了36.36-100%(平均：83%)，而对正常查询的影响可以忽略不计。为了进一步提高防御的健壮性并将成本降至最低，我们使用数据蒸馏方法来优化专用的开源防御模型。这些型号的性能超过了四种SOTA防御系统，并与基于GPT-4的SelfDefend的性能相当，额外延迟明显更低。我们的实验还表明，调整后的模型对目标GCG攻击和即时注入攻击具有较强的鲁棒性。



## **29. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05491v1) [paper-pdf](http://arxiv.org/pdf/2406.05491v1)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models trained on large-scale image-text pairs have demonstrated unprecedented capability in many practical applications. However, previous studies have revealed that VLP models are vulnerable to adversarial samples crafted by a malicious adversary. While existing attacks have achieved great success in improving attack effect and transferability, they all focus on instance-specific attacks that generate perturbations for each input sample. In this paper, we show that VLP models can be vulnerable to a new class of universal adversarial perturbation (UAP) for all input samples. Although initially transplanting existing UAP algorithms to perform attacks showed effectiveness in attacking discriminative models, the results were unsatisfactory when applied to VLP models. To this end, we revisit the multimodal alignments in VLP model training and propose the Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC). Specifically, we first design a generator that incorporates cross-modal information as conditioning input to guide the training. To further exploit cross-modal interactions, we propose to formulate the training objective as a multimodal contrastive learning paradigm based on our constructed positive and negative image-text pairs. By training the conditional generator with the designed loss, we successfully force the adversarial samples to move away from its original area in the VLP model's feature space, and thus essentially enhance the attacks. Extensive experiments show that our method achieves remarkable attack performance across various VLP models and Vision-and-Language (V+L) tasks. Moreover, C-PGC exhibits outstanding black-box transferability and achieves impressive results in fooling prevalent large VLP models including LLaVA and Qwen-VL.

摘要: 在大规模图文对上训练的视觉语言预训练(VLP)模型在许多实际应用中表现出了前所未有的能力。然而，以前的研究表明，VLP模型容易受到恶意对手制作的敌意样本的攻击。虽然现有的攻击在提高攻击效果和可转移性方面取得了很大的成功，但它们都专注于针对特定实例的攻击，这些攻击会对每个输入样本产生扰动。在本文中，我们证明了VLP模型对于所有输入样本都可能受到一类新的通用对抗性摄动(UAP)的影响。虽然最初移植现有的UAP算法进行攻击对区分模型的攻击是有效的，但将其应用于VLP模型时，效果并不理想。为此，我们回顾了VLP模型训练中的多模式对齐，并提出了具有跨模式条件的对比训练扰动生成器(C-PGC)。具体地说，我们首先设计了一个生成器，它结合了跨通道信息作为条件输入来指导训练。为了进一步开发跨通道互动，我们建议将培训目标制定为基于我们构建的正面和负面图文对的多通道对比学习范式。通过用设计的损失训练条件生成器，我们成功地迫使敌方样本在VLP模型的特征空间中离开其原始区域，从而从根本上增强了攻击。大量的实验表明，我们的方法在不同的视觉和语言(V+L)任务上取得了显著的攻击性能。此外，C-PGC表现出出色的黑盒可转移性，并在愚弄LLaVA和Qwen-VL等流行的大型VLP模型方面取得了令人印象深刻的结果。



## **30. Novel Approach to Intrusion Detection: Introducing GAN-MSCNN-BILSTM with LIME Predictions**

入侵检测的新方法：引入GAN-MSC NN-BILSTM和LIME预测 cs.CR

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05443v1) [paper-pdf](http://arxiv.org/pdf/2406.05443v1)

**Authors**: Asmaa Benchama, Khalid Zebbara

**Abstract**: This paper introduces an innovative intrusion detection system that harnesses Generative Adversarial Networks (GANs), Multi-Scale Convolutional Neural Networks (MSCNNs), and Bidirectional Long Short-Term Memory (BiLSTM) networks, supplemented by Local Interpretable Model-Agnostic Explanations (LIME) for interpretability. Employing a GAN, the system generates realistic network traffic data, encompassing both normal and attack patterns. This synthesized data is then fed into an MSCNN-BiLSTM architecture for intrusion detection. The MSCNN layer extracts features from the network traffic data at different scales, while the BiLSTM layer captures temporal dependencies within the traffic sequences. Integration of LIME allows for explaining the model's decisions. Evaluation on the Hogzilla dataset, a standard benchmark, showcases an impressive accuracy of 99.16\% for multi-class classification and 99.10\% for binary classification, while ensuring interpretability through LIME. This fusion of deep learning and interpretability presents a promising avenue for enhancing intrusion detection systems by improving transparency and decision support in network security.

摘要: 本文介绍了一种创新的入侵检测系统，它利用产生式对抗网络(GANS)、多尺度卷积神经网络(MSCNN)和双向长期短期记忆(BiLSTM)网络，并辅以局部可解释模型无关解释(LIME)来实现可解释性。该系统使用GAN生成真实的网络流量数据，包括正常模式和攻击模式。这些合成的数据然后被馈送到MSCNN-BiLSTM体系结构中用于入侵检测。MSCNN层从不同尺度的网络流量数据中提取特征，而BiLSTM层捕获流量序列中的时间相关性。石灰的整合可以解释模型的决定。对标准基准Hogzilla数据集的评估显示，多类分类的准确率高达99.16%，二类分类的准确率高达99.10%，同时确保了LIME的可解释性。这种深度学习和可解释性的融合为通过提高网络安全的透明度和决策支持来增强入侵检测系统提供了一条很有前途的途径。



## **31. Rethinking the Vulnerabilities of Face Recognition Systems:From a Practical Perspective**

重新思考面部识别系统的漏洞：从实践的角度 cs.CR

19 pages,version 3

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2405.12786v3) [paper-pdf](http://arxiv.org/pdf/2405.12786v3)

**Authors**: Jiahao Chen, Zhiqiang Shen, Yuwen Pu, Chunyi Zhou, Changjiang Li, Jiliang Li, Ting Wang, Shouling Ji

**Abstract**: Face Recognition Systems (FRS) have increasingly integrated into critical applications, including surveillance and user authentication, highlighting their pivotal role in modern security systems. Recent studies have revealed vulnerabilities in FRS to adversarial (e.g., adversarial patch attacks) and backdoor attacks (e.g., training data poisoning), raising significant concerns about their reliability and trustworthiness. Previous studies primarily focus on traditional adversarial or backdoor attacks, overlooking the resource-intensive or privileged-manipulation nature of such threats, thus limiting their practical generalization, stealthiness, universality and robustness. Correspondingly, in this paper, we delve into the inherent vulnerabilities in FRS through user studies and preliminary explorations. By exploiting these vulnerabilities, we identify a novel attack, facial identity backdoor attack dubbed FIBA, which unveils a potentially more devastating threat against FRS:an enrollment-stage backdoor attack. FIBA circumvents the limitations of traditional attacks, enabling broad-scale disruption by allowing any attacker donning a specific trigger to bypass these systems. This implies that after a single, poisoned example is inserted into the database, the corresponding trigger becomes a universal key for any attackers to spoof the FRS. This strategy essentially challenges the conventional attacks by initiating at the enrollment stage, dramatically transforming the threat landscape by poisoning the feature database rather than the training data.

摘要: 人脸识别系统(FRS)越来越多地集成到包括监控和用户身份验证在内的关键应用中，突显了它们在现代安全系统中的关键作用。最近的研究发现，FRS对对抗性攻击(例如，对抗性补丁攻击)和后门攻击(例如，训练数据中毒)的脆弱性，引起了人们对其可靠性和可信性的严重担忧。以往的研究主要集中于传统的对抗性攻击或后门攻击，忽略了此类威胁的资源密集型或特权操纵性，从而限制了它们的实用通用性、隐蔽性、普遍性和健壮性。相应地，在本文中，我们通过用户研究和初步探索，深入研究了FRS的固有漏洞。通过利用这些漏洞，我们确定了一种新型的攻击，即面部识别后门攻击，称为FIBA，它揭示了对FRS的一个潜在的更具破坏性的威胁：注册阶段的后门攻击。FIBA绕过了传统攻击的限制，允许任何使用特定触发器的攻击者绕过这些系统，从而实现广泛的破坏。这意味着在将单个有毒示例插入数据库后，相应的触发器将成为任何攻击者欺骗FRS的通用密钥。该策略实质上是通过在注册阶段发起攻击来挑战传统攻击，通过毒化特征数据库而不是训练数据来极大地改变威胁格局。



## **32. Adversarial flows: A gradient flow characterization of adversarial attacks**

对抗流：对抗攻击的梯度流特征 cs.LG

**SubmitDate**: 2024-06-08    [abs](http://arxiv.org/abs/2406.05376v1) [paper-pdf](http://arxiv.org/pdf/2406.05376v1)

**Authors**: Lukas Weigand, Tim Roith, Martin Burger

**Abstract**: A popular method to perform adversarial attacks on neuronal networks is the so-called fast gradient sign method and its iterative variant. In this paper, we interpret this method as an explicit Euler discretization of a differential inclusion, where we also show convergence of the discretization to the associated gradient flow. To do so, we consider the concept of p-curves of maximal slope in the case $p=\infty$. We prove existence of $\infty$-curves of maximum slope and derive an alternative characterization via differential inclusions. Furthermore, we also consider Wasserstein gradient flows for potential energies, where we show that curves in the Wasserstein space can be characterized by a representing measure on the space of curves in the underlying Banach space, which fulfill the differential inclusion. The application of our theory to the finite-dimensional setting is twofold: On the one hand, we show that a whole class of normalized gradient descent methods (in particular signed gradient descent) converge, up to subsequences, to the flow, when sending the step size to zero. On the other hand, in the distributional setting, we show that the inner optimization task of adversarial training objective can be characterized via $\infty$-curves of maximum slope on an appropriate optimal transport space.

摘要: 对神经元网络进行敌意攻击的一种流行方法是所谓的快速梯度符号方法及其迭代变体。在本文中，我们将该方法解释为微分包含的显式Euler离散化，并证明了该离散化收敛于相应的梯度流。为此，我们考虑了最大斜率的p-曲线的概念。我们证明了最大斜率的$-曲线的存在性，并通过微分包含得到了另一种刻画。此外，我们还考虑了势能的Wasserstein梯度流，其中我们证明了Wasserstein空间中的曲线可以用基本Banach空间中的曲线空间上的表示测度来刻画，从而满足微分包含.我们的理论在有限维环境中的应用有两个方面：一方面，我们证明了当步长为零时，一整类归一化梯度下降方法(特别是符号梯度下降方法)收敛到流，至上子序列。另一方面，在分布环境下，我们证明了对抗性训练目标的内部优化任务可以用适当的最优运输空间上的最大斜率曲线来刻画。



## **33. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.03230v2) [paper-pdf](http://arxiv.org/pdf/2406.03230v2)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 大型语言模型(LLM)的广泛采用，如OpenAI的ChatGPT，使防御这些模型上的对手威胁成为当务之急。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性和用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，该策略利用LLM变压器层之间的剩余激活分析。我们应用了一种新的方法来分析残留流中独特的激活模式，以进行攻击提示分类。我们精选了多个数据集，以演示此分类方法如何在多种类型的攻击场景中具有高精度，包括我们新创建的攻击数据集。此外，我们通过集成LLMS的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击的能力的影响。这些结果强调了我们的方法在加强对敌对输入的检测和缓解、推进LLMS运作的安全框架方面的有效性。



## **34. Compositional Curvature Bounds for Deep Neural Networks**

深度神经网络的组成弯曲界 cs.LG

Proceedings of the 41 st International Conference on Machine Learning  (ICML 2024)

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.05119v1) [paper-pdf](http://arxiv.org/pdf/2406.05119v1)

**Authors**: Taha Entesari, Sina Sharifi, Mahyar Fazlyab

**Abstract**: A key challenge that threatens the widespread use of neural networks in safety-critical applications is their vulnerability to adversarial attacks. In this paper, we study the second-order behavior of continuously differentiable deep neural networks, focusing on robustness against adversarial perturbations. First, we provide a theoretical analysis of robustness and attack certificates for deep classifiers by leveraging local gradients and upper bounds on the second derivative (curvature constant). Next, we introduce a novel algorithm to analytically compute provable upper bounds on the second derivative of neural networks. This algorithm leverages the compositional structure of the model to propagate the curvature bound layer-by-layer, giving rise to a scalable and modular approach. The proposed bound can serve as a differentiable regularizer to control the curvature of neural networks during training, thereby enhancing robustness. Finally, we demonstrate the efficacy of our method on classification tasks using the MNIST and CIFAR-10 datasets.

摘要: 威胁神经网络在安全关键应用中广泛使用的一个关键挑战是它们易受对手攻击。本文研究了连续可微深度神经网络的二阶行为，重点研究了其对敌意扰动的鲁棒性。首先，我们利用局部梯度和二阶导数(曲率常数)的上界，对深度分类器的稳健性和攻击证书进行了理论分析。接下来，我们介绍了一种新的算法来解析地计算神经网络二阶导数的可证明上界。该算法利用模型的组成结构逐层传播曲率界限，产生了一种可伸缩的模块化方法。所提出的界可以作为可微正则化器，在训练过程中控制神经网络的曲率，从而增强了鲁棒性。最后，我们使用MNIST和CIFAR-10数据集验证了我们的方法在分类任务上的有效性。



## **35. Corpus Poisoning via Approximate Greedy Gradient Descent**

通过近似贪婪梯度下降来中毒 cs.IR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.05087v1) [paper-pdf](http://arxiv.org/pdf/2406.05087v1)

**Authors**: Jinyan Su, John X. Morris, Preslav Nakov, Claire Cardie

**Abstract**: Dense retrievers are widely used in information retrieval and have also been successfully extended to other knowledge intensive areas such as language models, e.g., Retrieval-Augmented Generation (RAG) systems. Unfortunately, they have recently been shown to be vulnerable to corpus poisoning attacks in which a malicious user injects a small fraction of adversarial passages into the retrieval corpus to trick the system into returning these passages among the top-ranked results for a broad set of user queries. Further study is needed to understand the extent to which these attacks could limit the deployment of dense retrievers in real-world applications. In this work, we propose Approximate Greedy Gradient Descent (AGGD), a new attack on dense retrieval systems based on the widely used HotFlip method for efficiently generating adversarial passages. We demonstrate that AGGD can select a higher quality set of token-level perturbations than HotFlip by replacing its random token sampling with a more structured search. Experimentally, we show that our method achieves a high attack success rate on several datasets and using several retrievers, and can generalize to unseen queries and new domains. Notably, our method is extremely effective in attacking the ANCE retrieval model, achieving attack success rates that are 17.6\% and 13.37\% higher on the NQ and MS MARCO datasets, respectively, compared to HotFlip. Additionally, we demonstrate AGGD's potential to replace HotFlip in other adversarial attacks, such as knowledge poisoning of RAG systems.\footnote{Code can be find in \url{https://github.com/JinyanSu1/AGGD}}

摘要: 密集检索器被广泛应用于信息检索，也被成功地扩展到其他知识密集型领域，例如语言模型，例如检索-增强生成(RAG)系统。不幸的是，它们最近被证明容易受到语料库中毒攻击，在这种攻击中，恶意用户将一小部分对抗性段落注入检索语料库，以欺骗系统返回针对广泛的用户查询集合的排名靠前的结果中的这些段落。需要进一步的研究来了解这些攻击在多大程度上会限制密集检索器在现实世界应用中的部署。在这项工作中，我们提出了近似贪婪梯度下降(AGGD)，一种新的攻击密集检索系统的基础上，广泛使用的HotFlip方法，以有效地生成敌意段落。我们证明，通过用更结构化的搜索取代随机令牌抽样，AGGD可以选择比HotFlip更高质量的令牌级扰动集。实验表明，我们的方法在多个数据集和多个检索器上取得了很高的攻击成功率，并且可以推广到未知的查询和新的领域。值得注意的是，我们的方法在攻击ANCE检索模型方面非常有效，在NQ和MS Marco数据集上的攻击成功率分别比HotFlip高17.6和13.37。此外，我们还展示了AGGD在其他对手攻击中取代HotFlip的潜力，例如RAG系统的知识中毒。\脚注{代码可以在\URL{https://github.com/JinyanSu1/AGGD}}中找到



## **36. ADBA:Approximation Decision Boundary Approach for Black-Box Adversarial Attacks**

ADBA：黑匣子对抗攻击的逼近决策边界方法 cs.LG

10 pages, 5 figures, conference

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04998v1) [paper-pdf](http://arxiv.org/pdf/2406.04998v1)

**Authors**: Feiyang Wang, Xingquan Zuo, Hai Huang, Gang Chen

**Abstract**: Many machine learning models are susceptible to adversarial attacks, with decision-based black-box attacks representing the most critical threat in real-world applications. These attacks are extremely stealthy, generating adversarial examples using hard labels obtained from the target machine learning model. This is typically realized by optimizing perturbation directions, guided by decision boundaries identified through query-intensive exact search, significantly limiting the attack success rate. This paper introduces a novel approach using the Approximation Decision Boundary (ADB) to efficiently and accurately compare perturbation directions without precisely determining decision boundaries. The effectiveness of our ADB approach (ADBA) hinges on promptly identifying suitable ADB, ensuring reliable differentiation of all perturbation directions. For this purpose, we analyze the probability distribution of decision boundaries, confirming that using the distribution's median value as ADB can effectively distinguish different perturbation directions, giving rise to the development of the ADBA-md algorithm. ADBA-md only requires four queries on average to differentiate any pair of perturbation directions, which is highly query-efficient. Extensive experiments on six well-known image classifiers clearly demonstrate the superiority of ADBA and ADBA-md over multiple state-of-the-art black-box attacks.

摘要: 许多机器学习模型容易受到对抗性攻击，其中基于决策的黑盒攻击是现实世界应用程序中最关键的威胁。这些攻击非常隐蔽，使用从目标机器学习模型获得的硬标签生成敌意示例。这通常是通过优化扰动方向来实现的，由通过查询密集型精确搜索识别的决策边界来指导，从而显著限制攻击成功率。本文提出了一种新的方法，利用近似决策边界(ADB)来高效、准确地比较扰动方向，而无需精确地确定决策边界。我们的ADB方法(ADBA)的有效性取决于迅速找到合适的ADB，确保可靠地区分所有扰动方向。为此，我们分析了决策边界的概率分布，证实了用该分布的中值作为ADB可以有效地区分不同的扰动方向，从而导致了ADBA-MD算法的发展。ADBA-MD平均只需要4个查询就可以区分任意一对扰动方向，查询效率很高。在六个著名的图像分类器上的广泛实验清楚地证明了ADBA和ADBA-MD相对于多种最先进的黑盒攻击的优越性。



## **37. Adversarial Attacks and Defenses in Fault Detection and Diagnosis: A Comprehensive Benchmark on the Tennessee Eastman Process**

故障检测和诊断中的对抗性攻击和防御：田纳西州伊士曼进程的综合基准 cs.LG

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2403.13502v4) [paper-pdf](http://arxiv.org/pdf/2403.13502v4)

**Authors**: Vitaliy Pozdnyakov, Aleksandr Kovalenko, Ilya Makarov, Mikhail Drobyshevskiy, Kirill Lukyanov

**Abstract**: Integrating machine learning into Automated Control Systems (ACS) enhances decision-making in industrial process management. One of the limitations to the widespread adoption of these technologies in industry is the vulnerability of neural networks to adversarial attacks. This study explores the threats in deploying deep learning models for fault diagnosis in ACS using the Tennessee Eastman Process dataset. By evaluating three neural networks with different architectures, we subject them to six types of adversarial attacks and explore five different defense methods. Our results highlight the strong vulnerability of models to adversarial samples and the varying effectiveness of defense strategies. We also propose a novel protection approach by combining multiple defense methods and demonstrate it's efficacy. This research contributes several insights into securing machine learning within ACS, ensuring robust fault diagnosis in industrial processes.

摘要: 将机器学习集成到自动化控制系统（ACS）中，增强了工业过程管理中的决策。这些技术在工业中广泛采用的局限性之一是神经网络容易受到对抗攻击。本研究探索了使用田纳西州伊士曼Process数据集在ACS中部署深度学习模型进行故障诊断的威胁。通过评估具有不同架构的三个神经网络，我们将它们置于六种类型的对抗攻击中，并探索五种不同的防御方法。我们的结果凸显了模型对对抗样本的强烈脆弱性以及防御策略的不同有效性。我们还提出了一种通过结合多种防御方法的新型保护方法，并证明了其有效性。这项研究为确保ACS内的机器学习提供了多项见解，确保工业流程中的稳健故障诊断。



## **38. Fragile Model Watermarking: A Comprehensive Survey of Evolution, Characteristics, and Classification**

脆弱模型水印：演变、特征和分类的全面调查 cs.CR

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04809v1) [paper-pdf](http://arxiv.org/pdf/2406.04809v1)

**Authors**: Zhenzhe Gao, Yu Cheng, Zhaoxia Yin

**Abstract**: Model fragile watermarking, inspired by both the field of adversarial attacks on neural networks and traditional multimedia fragile watermarking, has gradually emerged as a potent tool for detecting tampering, and has witnessed rapid development in recent years. Unlike robust watermarks, which are widely used for identifying model copyrights, fragile watermarks for models are designed to identify whether models have been subjected to unexpected alterations such as backdoors, poisoning, compression, among others. These alterations can pose unknown risks to model users, such as misidentifying stop signs as speed limit signs in classic autonomous driving scenarios. This paper provides an overview of the relevant work in the field of model fragile watermarking since its inception, categorizing them and revealing the developmental trajectory of the field, thus offering a comprehensive survey for future endeavors in model fragile watermarking.

摘要: 模型脆弱水印受到神经网络对抗攻击领域和传统多媒体脆弱水印的启发，逐渐成为检测篡改的有力工具，并在近年来得到了快速发展。与广泛用于识别模型版权的稳健水印不同，模型的脆弱水印旨在识别模型是否遭受了意外更改，例如后门、中毒、压缩等。这些更改可能会给模型用户带来未知的风险，例如在经典自动驾驶场景中将停车标志误识别为限速标志。本文概述了模型脆弱水印领域自诞生以来的相关工作，对其进行了分类，揭示了该领域的发展轨迹，从而为模型脆弱水印的未来工作提供了全面的综述。



## **39. Probabilistic Perspectives on Error Minimization in Adversarial Reinforcement Learning**

对抗强化学习中错误最小化的概率观点 cs.LG

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04724v1) [paper-pdf](http://arxiv.org/pdf/2406.04724v1)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are critically vulnerable to adversarial noise in observations, posing severe risks in safety-critical scenarios. For example, a self-driving car receiving manipulated sensory inputs about traffic signs could lead to catastrophic outcomes. Existing strategies to fortify RL algorithms against such adversarial perturbations generally fall into two categories: (a) using regularization methods that enhance robustness by incorporating adversarial loss terms into the value objectives, and (b) adopting "maximin" principles, which focus on maximizing the minimum value to ensure robustness. While regularization methods reduce the likelihood of successful attacks, their effectiveness drops significantly if an attack does succeed. On the other hand, maximin objectives, although robust, tend to be overly conservative. To address this challenge, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), which naturally balances optimizing value and robustness against adversarial attacks. To optimize ACoE in a scalable manner in model-free settings, we propose a theoretically justified surrogate objective known as Cumulative-ACoE (C-ACoE). The core idea of optimizing C-ACoE is utilizing the belief about the underlying true state given the adversarially perturbed observation. Our empirical evaluations demonstrate that our method outperforms current state-of-the-art approaches for addressing adversarial RL problems across all established benchmarks (MuJoCo, Atari, and Highway) used in the literature.

摘要: 深度强化学习(DRL)策略在观察中极易受到对抗性噪声的影响，在安全关键场景中构成严重风险。例如，自动驾驶汽车接收到关于交通标志的受操纵的感官输入可能会导致灾难性的结果。现有的增强RL算法抵抗这种对抗性扰动的策略通常分为两类：(A)使用正则化方法，通过将对抗性损失项合并到价值目标中来增强稳健性；以及(B)采用“最大”原则，其重点是最大化最小值以确保稳健性。虽然正规化方法降低了攻击成功的可能性，但如果攻击确实成功，其有效性会显著下降。另一方面，最大化目标虽然稳健，但往往过于保守。为了应对这一挑战，我们引入了一个新的目标，称为对抗性反事实错误(ACoE)，它自然地平衡了优化值和对抗攻击的健壮性。为了在无模型环境下以可扩展的方式优化ACoE，我们提出了一个理论上合理的代理目标，称为累积ACoE(C-ACoE)。优化C-ACoE的核心思想是利用对给定相反扰动观测的潜在真实状态的信念。我们的经验评估表明，我们的方法在文献中使用的所有已建立的基准(MuJoCo、Atari和Road)上都优于当前最先进的方法来解决对抗性RL问题。



## **40. WAVES: Benchmarking the Robustness of Image Watermarks**

WAVES：图像水印的鲁棒性基准 cs.CV

Accepted by ICML 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2401.08573v3) [paper-pdf](http://arxiv.org/pdf/2401.08573v3)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: In the burgeoning age of generative AI, watermarks act as identifiers of provenance and artificial content. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a benchmark for assessing image watermark robustness, overcoming the limitations of current evaluation methods. WAVES integrates detection and identification tasks and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced, novel variations of diffusive, and adversarial attacks. Our evaluation examines two pivotal dimensions: the degree of image quality degradation and the efficacy of watermark detection after attacks. Our novel, comprehensive evaluation reveals previously undetected vulnerabilities of several modern watermarking algorithms. We envision WAVES as a toolkit for the future development of robust watermarks. The project is available at https://wavesbench.github.io/

摘要: 在生成人工智能的蓬勃发展时代，水印充当出处和人工内容的标识符。我们提出了WAVES（通过增强压力测试进行水印分析），这是评估图像水印稳健性的基准，克服了当前评估方法的局限性。WAVES集成了检测和识别任务，并建立了由各种压力测试组成的标准化评估协议。WAVES中的攻击范围从传统的图像失真到扩散攻击和对抗攻击的高级、新颖变体。我们的评估检查了两个关键维度：图像质量退化程度和攻击后水印检测的有效性。我们新颖、全面的评估揭示了几种现代水印算法之前未检测到的漏洞。我们将WAVES视为未来开发稳健水印的工具包。该项目可访问https://wavesbench.github.io/



## **41. NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise**

NoisyGL：标签噪音下图神经网络的综合基准 cs.LG

28 pages, 15 figures

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04299v2) [paper-pdf](http://arxiv.org/pdf/2406.04299v2)

**Authors**: Zhonghao Wang, Danyu Sun, Sheng Zhou, Haobo Wang, Jiapei Fan, Longtao Huang, Jiajun Bu

**Abstract**: Graph Neural Networks (GNNs) exhibit strong potential in node classification task through a message-passing mechanism. However, their performance often hinges on high-quality node labels, which are challenging to obtain in real-world scenarios due to unreliable sources or adversarial attacks. Consequently, label noise is common in real-world graph data, negatively impacting GNNs by propagating incorrect information during training. To address this issue, the study of Graph Neural Networks under Label Noise (GLN) has recently gained traction. However, due to variations in dataset selection, data splitting, and preprocessing techniques, the community currently lacks a comprehensive benchmark, which impedes deeper understanding and further development of GLN. To fill this gap, we introduce NoisyGL in this paper, the first comprehensive benchmark for graph neural networks under label noise. NoisyGL enables fair comparisons and detailed analyses of GLN methods on noisy labeled graph data across various datasets, with unified experimental settings and interface. Our benchmark has uncovered several important insights that were missed in previous research, and we believe these findings will be highly beneficial for future studies. We hope our open-source benchmark library will foster further advancements in this field. The code of the benchmark can be found in https://github.com/eaglelab-zju/NoisyGL.

摘要: 图神经网络(GNN)通过一种消息传递机制在节点分类任务中显示出很强的潜力。然而，它们的性能往往取决于高质量的节点标签，而在现实世界的场景中，由于不可靠的来源或对手攻击，这些标签很难获得。因此，标签噪声在真实世界的图形数据中很常见，通过在训练期间传播不正确的信息而对GNN产生负面影响。为了解决这个问题，标签噪声下的图神经网络(GLN)的研究最近得到了重视。然而，由于数据集选择、数据拆分和预处理技术的差异，目前社区缺乏一个全面的基准，这阻碍了对GLN的深入理解和进一步发展。为了填补这一空白，我们在本文中引入了NoisyGL，这是第一个在标签噪声下对图神经网络进行全面测试的基准。NoisyGL可以通过统一的实验设置和界面，对不同数据集上的噪声标记图形数据进行公平的GLN方法比较和详细分析。我们的基准发现了之前研究中遗漏的几个重要见解，我们相信这些发现将对未来的研究非常有益。我们希望我们的开源基准库将促进这一领域的进一步发展。基准测试的代码可以在https://github.com/eaglelab-zju/NoisyGL.中找到



## **42. Neural Codec-based Adversarial Sample Detection for Speaker Verification**

基于神经编解码器的对抗样本检测用于说话人验证 eess.AS

Accepted by Interspeech 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2406.04582v1) [paper-pdf](http://arxiv.org/pdf/2406.04582v1)

**Authors**: Xuanjun Chen, Jiawei Du, Haibin Wu, Jyh-Shing Roger Jang, Hung-yi Lee

**Abstract**: Automatic Speaker Verification (ASV), increasingly used in security-critical applications, faces vulnerabilities from rising adversarial attacks, with few effective defenses available. In this paper, we propose a neural codec-based adversarial sample detection method for ASV. The approach leverages the codec's ability to discard redundant perturbations and retain essential information. Specifically, we distinguish between genuine and adversarial samples by comparing ASV score differences between original and re-synthesized audio (by codec models). This comprehensive study explores all open-source neural codecs and their variant models for experiments. The Descript-audio-codec model stands out by delivering the highest detection rate among 15 neural codecs and surpassing seven prior state-of-the-art (SOTA) detection methods. Note that, our single-model method even outperforms a SOTA ensemble method by a large margin.

摘要: 自动说话人验证（ASV）越来越多地用于安全关键应用程序，但它面临着不断增加的对抗攻击的漏洞，可用的有效防御措施很少。本文提出了一种基于神经编解码器的ASV对抗样本检测方法。该方法利用了编解码器丢弃冗余扰动并保留重要信息的能力。具体来说，我们通过比较原始和重新合成音频之间的ASV评分差异（通过编解码器模型）来区分真实样本和对抗样本。这项全面的研究探索了所有开源神经编解码器及其变体模型进行实验。描述音频编解码器模型在15种神经编解码器中提供最高的检测率，并超过了7种先前最先进的（SOTA）检测方法，脱颖而出。请注意，我们的单模型方法甚至远远优于SOTA集成方法。



## **43. COLD-Attack: Jailbreaking LLMs with Stealthiness and Controllability**

冷攻击：具有隐蔽性和可控性的越狱LLM cs.LG

Accepted to ICML 2024

**SubmitDate**: 2024-06-07    [abs](http://arxiv.org/abs/2402.08679v2) [paper-pdf](http://arxiv.org/pdf/2402.08679v2)

**Authors**: Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, Bin Hu

**Abstract**: Jailbreaks on large language models (LLMs) have recently received increasing attention. For a comprehensive assessment of LLM safety, it is essential to consider jailbreaks with diverse attributes, such as contextual coherence and sentiment/stylistic variations, and hence it is beneficial to study controllable jailbreaking, i.e. how to enforce control on LLM attacks. In this paper, we formally formulate the controllable attack generation problem, and build a novel connection between this problem and controllable text generation, a well-explored topic of natural language processing. Based on this connection, we adapt the Energy-based Constrained Decoding with Langevin Dynamics (COLD), a state-of-the-art, highly efficient algorithm in controllable text generation, and introduce the COLD-Attack framework which unifies and automates the search of adversarial LLM attacks under a variety of control requirements such as fluency, stealthiness, sentiment, and left-right-coherence. The controllability enabled by COLD-Attack leads to diverse new jailbreak scenarios which not only cover the standard setting of generating fluent (suffix) attack with continuation constraint, but also allow us to address new controllable attack settings such as revising a user query adversarially with paraphrasing constraint, and inserting stealthy attacks in context with position constraint. Our extensive experiments on various LLMs (Llama-2, Mistral, Vicuna, Guanaco, GPT-3.5, and GPT-4) show COLD-Attack's broad applicability, strong controllability, high success rate, and attack transferability. Our code is available at https://github.com/Yu-Fangxu/COLD-Attack.

摘要: 大型语言模型(LLM)的越狱最近受到越来越多的关注。为了全面评估LLM的安全性，必须考虑具有不同属性的越狱，例如上下文连贯性和情绪/风格变化，因此研究可控越狱是有益的，即如何加强对LLM攻击的控制。在本文中，我们形式化地描述了可控攻击生成问题，并将该问题与自然语言处理的一个热门话题--可控文本生成建立了一种新的联系。基于此，我们采用了基于能量的朗之万动力学约束解码算法(COLD)，这是一种最新的、高效的可控文本生成算法，并引入了冷攻击框架，该框架可以在流畅性、隐蔽性、情感和左右一致性等各种控制要求下统一和自动化搜索敌意LLM攻击。冷攻击的可控性导致了新的越狱场景的多样化，这些场景不仅覆盖了生成具有连续约束的流畅(后缀)攻击的标准设置，而且允许我们应对新的可控攻击设置，如使用释义约束对用户查询进行恶意修改，以及在具有位置约束的上下文中插入隐蔽攻击。我们在不同的LLMS(大骆驼-2、米斯特拉尔、维库纳、瓜纳科、GPT-3.5和GPT-4)上的广泛实验表明，冷攻击具有广泛的适用性、很强的可控性、高成功率和攻击可转移性。我们的代码可以在https://github.com/Yu-Fangxu/COLD-Attack.上找到



## **44. Safety Alignment in NLP Tasks: Weakly Aligned Summarization as an In-Context Attack**

NLP任务中的安全一致：弱一致摘要作为上下文内攻击 cs.CL

Accepted to ACL2024 main

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2312.06924v2) [paper-pdf](http://arxiv.org/pdf/2312.06924v2)

**Authors**: Yu Fu, Yufei Li, Wen Xiao, Cong Liu, Yue Dong

**Abstract**: Recent developments in balancing the usefulness and safety of Large Language Models (LLMs) have raised a critical question: Are mainstream NLP tasks adequately aligned with safety consideration? Our study, focusing on safety-sensitive documents obtained through adversarial attacks, reveals significant disparities in the safety alignment of various NLP tasks. For instance, LLMs can effectively summarize malicious long documents but often refuse to translate them. This discrepancy highlights a previously unidentified vulnerability: attacks exploiting tasks with weaker safety alignment, like summarization, can potentially compromise the integrity of tasks traditionally deemed more robust, such as translation and question-answering (QA). Moreover, the concurrent use of multiple NLP tasks with lesser safety alignment increases the risk of LLMs inadvertently processing harmful content. We demonstrate these vulnerabilities in various safety-aligned LLMs, particularly Llama2 models, Gemini and GPT-4, indicating an urgent need for strengthening safety alignments across a broad spectrum of NLP tasks.

摘要: 最近在平衡大型语言模型(LLM)的有用性和安全性方面的发展提出了一个关键问题：主流NLP任务是否与安全考虑充分一致？我们的研究集中在通过对抗性攻击获得的安全敏感文件上，揭示了各种NLP任务在安全匹配方面的显著差异。例如，LLMS可以有效地汇总恶意的长文档，但通常拒绝翻译它们。这一差异突显了一个以前未知的漏洞：攻击利用安全一致性较弱的任务(如摘要)，可能会潜在地损害传统上被认为更健壮的任务的完整性，如翻译和问答(QA)。此外，同时使用安全性较低的多个NLP任务会增加LLMS无意中处理有害内容的风险。我们在各种安全对齐的LLM中展示了这些漏洞，特别是Llama2型号、Gemini和GPT-4，这表明迫切需要在广泛的NLP任务中加强安全对齐。



## **45. PromptFix: Few-shot Backdoor Removal via Adversarial Prompt Tuning**

AttributionFix：通过对抗性提示调整删除少量后门 cs.CL

NAACL 2024

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04478v1) [paper-pdf](http://arxiv.org/pdf/2406.04478v1)

**Authors**: Tianrong Zhang, Zhaohan Xi, Ting Wang, Prasenjit Mitra, Jinghui Chen

**Abstract**: Pre-trained language models (PLMs) have attracted enormous attention over the past few years with their unparalleled performances. Meanwhile, the soaring cost to train PLMs as well as their amazing generalizability have jointly contributed to few-shot fine-tuning and prompting as the most popular training paradigms for natural language processing (NLP) models. Nevertheless, existing studies have shown that these NLP models can be backdoored such that model behavior is manipulated when trigger tokens are presented. In this paper, we propose PromptFix, a novel backdoor mitigation strategy for NLP models via adversarial prompt-tuning in few-shot settings. Unlike existing NLP backdoor removal methods, which rely on accurate trigger inversion and subsequent model fine-tuning, PromptFix keeps the model parameters intact and only utilizes two extra sets of soft tokens which approximate the trigger and counteract it respectively. The use of soft tokens and adversarial optimization eliminates the need to enumerate possible backdoor configurations and enables an adaptive balance between trigger finding and preservation of performance. Experiments with various backdoor attacks validate the effectiveness of the proposed method and the performances when domain shift is present further shows PromptFix's applicability to models pretrained on unknown data source which is the common case in prompt tuning scenarios.

摘要: 在过去的几年里，预训练的语言模型(PLM)以其无与伦比的表现吸引了人们的极大关注。与此同时，训练PLM的成本飙升以及它们惊人的泛化能力共同导致了微调和提示成为自然语言处理(NLP)模型最受欢迎的训练范例。然而，现有的研究表明，这些NLP模型可以被倒退，使得当提供触发令牌时，模型行为被操纵。在本文中，我们提出了一种新的后门缓解策略PromptFix，该策略通过对抗性的快速调整在少镜头环境下对NLP模型进行缓解。与现有的NLP后门去除方法不同，PromptFix保持模型参数不变，只使用两组额外的软令牌来分别逼近和抵消触发，而不是依赖于精确的触发反转和后续的模型微调。软令牌和对抗性优化的使用消除了列举可能的后门配置的需要，并实现了触发查找和性能保持之间的自适应平衡。通过对各种后门攻击的实验，验证了该方法的有效性和当域转移时的性能，进一步表明了PromptFix对未知数据源上的预训练模型的适用性，这是快速调优场景中的常见情况。



## **46. BadRAG: Identifying Vulnerabilities in Retrieval Augmented Generation of Large Language Models**

BadRAG：识别大型语言模型检索增强生成中的漏洞 cs.CR

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.00083v2) [paper-pdf](http://arxiv.org/pdf/2406.00083v2)

**Authors**: Jiaqi Xue, Mengxin Zheng, Yebowen Hu, Fei Liu, Xun Chen, Qian Lou

**Abstract**: Large Language Models (LLMs) are constrained by outdated information and a tendency to generate incorrect data, commonly referred to as "hallucinations." Retrieval-Augmented Generation (RAG) addresses these limitations by combining the strengths of retrieval-based methods and generative models. This approach involves retrieving relevant information from a large, up-to-date dataset and using it to enhance the generation process, leading to more accurate and contextually appropriate responses. Despite its benefits, RAG introduces a new attack surface for LLMs, particularly because RAG databases are often sourced from public data, such as the web. In this paper, we propose \TrojRAG{} to identify the vulnerabilities and attacks on retrieval parts (RAG database) and their indirect attacks on generative parts (LLMs). Specifically, we identify that poisoning several customized content passages could achieve a retrieval backdoor, where the retrieval works well for clean queries but always returns customized poisoned adversarial queries. Triggers and poisoned passages can be highly customized to implement various attacks. For example, a trigger could be a semantic group like "The Republican Party, Donald Trump, etc." Adversarial passages can be tailored to different contents, not only linked to the triggers but also used to indirectly attack generative LLMs without modifying them. These attacks can include denial-of-service attacks on RAG and semantic steering attacks on LLM generations conditioned by the triggers. Our experiments demonstrate that by just poisoning 10 adversarial passages can induce 98.2\% success rate to retrieve the adversarial passages. Then, these passages can increase the reject ratio of RAG-based GPT-4 from 0.01\% to 74.6\% or increase the rate of negative responses from 0.22\% to 72\% for targeted queries.

摘要: 大型语言模型(LLM)受到过时信息和生成错误数据的倾向的限制，这通常被称为“幻觉”。检索-增强生成(RAG)结合了基于检索的方法和生成模型的优点，解决了这些局限性。这种方法涉及从大型最新数据集中检索相关信息，并使用它来改进生成过程，从而产生更准确和符合上下文的响应。尽管有好处，但RAG为LLMS带来了新的攻击面，特别是因为RAG数据库通常来自公共数据，如Web。本文提出用TrojRAG{}来识别检索零件(RAG数据库)上的漏洞和攻击，以及它们对生成零件(LLM)的间接攻击。具体地说，我们发现毒化几个定制的内容段落可以实现检索后门，其中检索对于干净的查询工作得很好，但总是返回定制的有毒对抗性查询。触发器和有毒段落可以高度定制，以实施各种攻击。例如，触发点可能是一个语义组，比如“共和党、唐纳德·特朗普等。”对抗性段落可以针对不同的内容量身定做，不仅与触发因素有关，还可以用来间接攻击生成性LLM而不修改它们。这些攻击可以包括针对RAG的拒绝服务攻击和针对受触发器限制的LLM生成的语义引导攻击。我们的实验表明，只要毒化10篇对抗性文章，就可以诱导98.2%的成功率来检索对抗性文章。然后，这些文章可以将基于RAG的GPT-4的拒绝率从0.01%提高到74.6%，或者将目标查询的否定回复率从0.22%提高到72%。



## **47. Batch-in-Batch: a new adversarial training framework for initial perturbation and sample selection**

批中批：用于初始扰动和样本选择的新对抗训练框架 cs.LG

29 pages, 11 figures

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04070v1) [paper-pdf](http://arxiv.org/pdf/2406.04070v1)

**Authors**: Yinting Wu, Pai Peng, Bo Cai, Le Li, .

**Abstract**: Adversarial training methods commonly generate independent initial perturbation for adversarial samples from a simple uniform distribution, and obtain the training batch for the classifier without selection. In this work, we propose a simple yet effective training framework called Batch-in-Batch (BB) to enhance models robustness. It involves specifically a joint construction of initial values that could simultaneously generates $m$ sets of perturbations from the original batch set to provide more diversity for adversarial samples; and also includes various sample selection strategies that enable the trained models to have smoother losses and avoid overconfident outputs. Through extensive experiments on three benchmark datasets (CIFAR-10, SVHN, CIFAR-100) with two networks (PreActResNet18 and WideResNet28-10) that are used in both the single-step (Noise-Fast Gradient Sign Method, N-FGSM) and multi-step (Projected Gradient Descent, PGD-10) adversarial training, we show that models trained within the BB framework consistently have higher adversarial accuracy across various adversarial settings, notably achieving over a 13% improvement on the SVHN dataset with an attack radius of 8/255 compared to the N-FGSM baseline model. Furthermore, experimental analysis of the efficiency of both the proposed initial perturbation method and sample selection strategies validates our insights. Finally, we show that our framework is cost-effective in terms of computational resources, even with a relatively large value of $m$.

摘要: 对抗性训练方法通常从简单的均匀分布对对抗性样本产生独立的初始扰动，不需要选择就可以获得分类器的训练批次。在这项工作中，我们提出了一种简单而有效的训练框架，称为Batch-in-Batch(BB)，以增强模型的稳健性。具体而言，它涉及联合构造初值，该初值可以同时从原始批次集合产生$m$集合的扰动，从而为对抗性样本提供更多的多样性；还包括各种样本选择策略，使训练的模型具有更平滑的损失并避免过度自信的输出。通过在三个基准数据集(CIFAR-10，SVHN，CIFAR-100)和两个网络(PreActResNet18和WideResNet28-10)上的广泛实验，这两个网络同时用于单步(噪声-快速梯度符号方法，N-FGSM)和多步(投影梯度下降，PGD-10)对抗训练，我们表明，在BB框架内训练的模型在各种对抗环境下一致具有更高的对抗准确率，与N-FGSM基线模型相比，攻击半径为8/255的SVHN数据集获得了13%以上的改进。此外，对初始摄动法和样本选择策略的效率进行了实验分析，验证了本文的观点。最后，我们证明了我们的框架在计算资源方面是有成本效益的，即使有相对较大的百万美元的价值。



## **48. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2406.04031v1) [paper-pdf](http://arxiv.org/pdf/2406.04031v1)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **49. Competition Report: Finding Universal Jailbreak Backdoors in Aligned LLMs**

竞争报告：在一致的LLC中寻找通用越狱后门 cs.CL

Competition Report

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2404.14461v2) [paper-pdf](http://arxiv.org/pdf/2404.14461v2)

**Authors**: Javier Rando, Francesco Croce, Kryštof Mitka, Stepan Shabalin, Maksym Andriushchenko, Nicolas Flammarion, Florian Tramèr

**Abstract**: Large language models are aligned to be safe, preventing users from generating harmful content like misinformation or instructions for illegal activities. However, previous work has shown that the alignment process is vulnerable to poisoning attacks. Adversaries can manipulate the safety training data to inject backdoors that act like a universal sudo command: adding the backdoor string to any prompt enables harmful responses from models that, otherwise, behave safely. Our competition, co-located at IEEE SaTML 2024, challenged participants to find universal backdoors in several large language models. This report summarizes the key findings and promising ideas for future research.

摘要: 大型语言模型经过调整以确保安全，防止用户生成错误信息或非法活动指令等有害内容。然而，之前的工作表明，对齐过程很容易受到中毒攻击。对手可以操纵安全训练数据来注入类似于通用sudo命令的后门：将后门字符串添加到任何提示中都会导致模型做出有害响应，否则这些模型会安全地运行。我们的竞赛在IEEE SaTML 2024上举行，挑战参与者在几个大型语言模型中找到通用后门。本报告总结了关键发现和未来研究的有希望的想法。



## **50. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

**SubmitDate**: 2024-06-06    [abs](http://arxiv.org/abs/2405.18942v2) [paper-pdf](http://arxiv.org/pdf/2405.18942v2)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces VRCP (Verifiably Robust Conformal Prediction), a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。本文介绍了一种新的框架VRCP，它利用最新的神经网络验证方法来恢复对抗性攻击下的覆盖保证。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



