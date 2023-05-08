# Latest Adversarial Attack Papers
**update at 2023-05-08 09:40:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. White-Box Multi-Objective Adversarial Attack on Dialogue Generation**

对话生成的白盒多目标对抗性攻击 cs.CL

ACL 2023 main conference long paper

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03655v1) [paper-pdf](http://arxiv.org/pdf/2305.03655v1)

**Authors**: Yufei Li, Zexin Li, Yingfan Gao, Cong Liu

**Abstract**: Pre-trained transformers are popular in state-of-the-art dialogue generation (DG) systems. Such language models are, however, vulnerable to various adversarial samples as studied in traditional tasks such as text classification, which inspires our curiosity about their robustness in DG systems. One main challenge of attacking DG models is that perturbations on the current sentence can hardly degrade the response accuracy because the unchanged chat histories are also considered for decision-making. Instead of merely pursuing pitfalls of performance metrics such as BLEU, ROUGE, we observe that crafting adversarial samples to force longer generation outputs benefits attack effectiveness -- the generated responses are typically irrelevant, lengthy, and repetitive. To this end, we propose a white-box multi-objective attack method called DGSlow. Specifically, DGSlow balances two objectives -- generation accuracy and length, via a gradient-based multi-objective optimizer and applies an adaptive searching mechanism to iteratively craft adversarial samples with only a few modifications. Comprehensive experiments on four benchmark datasets demonstrate that DGSlow could significantly degrade state-of-the-art DG models with a higher success rate than traditional accuracy-based methods. Besides, our crafted sentences also exhibit strong transferability in attacking other models.

摘要: 预先培训的变压器在最先进的对话生成(DG)系统中很受欢迎。然而，这类语言模型容易受到文本分类等传统任务中研究的各种对抗性样本的影响，这引发了我们对它们在DG系统中的健壮性的好奇。攻击DG模型的一个主要挑战是，当前句子的扰动几乎不会降低响应精度，因为没有变化的聊天历史也被考虑用于决策。而不是仅仅追求性能指标的陷阱，如BLEU，Rouge，我们观察到精心制作敌意样本来迫使更长的世代输出有利于攻击效率-生成的响应通常是无关的、冗长的和重复的。为此，我们提出了一种白盒多目标攻击方法DGSlow。具体地说，DGSlow通过基于梯度的多目标优化器来平衡生成精度和长度这两个目标，并应用自适应搜索机制来迭代地创建只需少量修改的对抗性样本。在四个基准数据集上的综合实验表明，DGSlow可以显著降低最新的DG模型，并且比传统的基于精度的方法具有更高的成功率。此外，我们制作的句子在攻击其他模型时也表现出很强的可转移性。



## **2. Verifiable Learning for Robust Tree Ensembles**

用于稳健树集成的可验证学习 cs.LG

17 pages, 3 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.03626v1) [paper-pdf](http://arxiv.org/pdf/2305.03626v1)

**Authors**: Stefano Calzavara, Lorenzo Cazzaro, Giulio Ermanno Pibiri, Nicola Prezza

**Abstract**: Verifying the robustness of machine learning models against evasion attacks at test time is an important research problem. Unfortunately, prior work established that this problem is NP-hard for decision tree ensembles, hence bound to be intractable for specific inputs. In this paper, we identify a restricted class of decision tree ensembles, called large-spread ensembles, which admit a security verification algorithm running in polynomial time. We then propose a new approach called verifiable learning, which advocates the training of such restricted model classes which are amenable for efficient verification. We show the benefits of this idea by designing a new training algorithm that automatically learns a large-spread decision tree ensemble from labelled data, thus enabling its security verification in polynomial time. Experimental results on publicly available datasets confirm that large-spread ensembles trained using our algorithm can be verified in a matter of seconds, using standard commercial hardware. Moreover, large-spread ensembles are more robust than traditional ensembles against evasion attacks, while incurring in just a relatively small loss of accuracy in the non-adversarial setting.

摘要: 验证机器学习模型在测试时对逃避攻击的稳健性是一个重要的研究问题。不幸的是，以前的工作确定了这个问题对于决策树集成来说是NP-Hard的，因此对于特定的输入必然是棘手的。在本文中，我们识别了一类受限的决策树集成，称为大分布集成，它允许安全验证算法在多项式时间内运行。然后，我们提出了一种新的方法，称为可验证学习，它主张训练这样的受限模型类，这些模型类适合于有效的验证。我们通过设计一种新的训练算法，从标记数据中自动学习大规模决策树集成，从而在多项式时间内实现其安全性验证，从而展示了这种思想的好处。在公开可用的数据集上的实验结果证实，使用我们的算法训练的大范围集成可以在几秒钟内使用标准商业硬件进行验证。此外，大范围的合奏比传统的合奏更能抵御逃避攻击，而在非对抗性的设置中，只会造成相对较小的准确性损失。



## **3. Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection**

与您签约的目标不同：使用间接提示注入损害真实世界的LLM集成应用程序 cs.CR

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.12173v2) [paper-pdf](http://arxiv.org/pdf/2302.12173v2)

**Authors**: Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, Mario Fritz

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into various applications. The functionalities of recent LLMs can be flexibly modulated via natural language prompts. This renders them susceptible to targeted adversarial prompting, e.g., Prompt Injection (PI) attacks enable attackers to override original instructions and employed controls. So far, it was assumed that the user is directly prompting the LLM. But, what if it is not the user prompting? We argue that LLM-Integrated Applications blur the line between data and instructions. We reveal new attack vectors, using Indirect Prompt Injection, that enable adversaries to remotely (without a direct interface) exploit LLM-integrated applications by strategically injecting prompts into data likely to be retrieved. We derive a comprehensive taxonomy from a computer security perspective to systematically investigate impacts and vulnerabilities, including data theft, worming, information ecosystem contamination, and other novel security risks. We demonstrate our attacks' practical viability against both real-world systems, such as Bing's GPT-4 powered Chat and code-completion engines, and synthetic applications built on GPT-4. We show how processing retrieved prompts can act as arbitrary code execution, manipulate the application's functionality, and control how and if other APIs are called. Despite the increasing integration and reliance on LLMs, effective mitigations of these emerging threats are currently lacking. By raising awareness of these vulnerabilities and providing key insights into their implications, we aim to promote the safe and responsible deployment of these powerful models and the development of robust defenses that protect users and systems from potential attacks.

摘要: 大型语言模型(LLM)越来越多地被集成到各种应用程序中。最近的LLMS的功能可以通过自然语言提示灵活地调节。这使得它们容易受到有针对性的对抗性提示，例如，提示注入(PI)攻击使攻击者能够覆盖原始指令和采用的控制。到目前为止，假设用户是在直接提示LLM。但是，如果不是用户提示呢？我们认为，LLM集成应用程序模糊了数据和指令之间的界限。我们使用间接提示注入揭示了新的攻击载体，使攻击者能够通过战略性地向可能被检索的数据注入提示来远程(无需直接接口)利用LLM集成的应用程序。我们从计算机安全的角度得出了一个全面的分类，以系统地调查影响和漏洞，包括数据窃取、蠕虫、信息生态系统污染和其他新的安全风险。我们展示了我们的攻击在现实世界系统上的实际可行性，例如Bing的GPT-4聊天和代码完成引擎，以及基于GPT-4的合成应用程序。我们展示了处理检索到的提示如何充当任意代码执行、操纵应用程序的功能以及控制调用其他API的方式和是否调用。尽管对小岛屿发展中国家的整合和依赖日益增加，但目前缺乏对这些新出现的威胁的有效缓解。通过提高对这些漏洞的认识并提供对其影响的关键见解，我们的目标是促进安全和负责任地部署这些强大的模型，并开发强大的防御措施，以保护用户和系统免受潜在攻击。



## **4. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2304.04033v2) [paper-pdf](http://arxiv.org/pdf/2304.04033v2)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **5. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2305.01361v2) [paper-pdf](http://arxiv.org/pdf/2305.01361v2)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at \textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解后的Top-1奇异值关联特征来计算输出逻辑，然后将其与原始逻辑相结合来优化对抗性实例，从而进行攻击。大量的实验结果验证了该方法的有效性，该方法可以很容易地集成到不同的基线中，显著提高对手样本干扰正常训练的CNN和高级防御策略的可转移性。这项研究的源代码可在\textcolor{blue}{\href{https://anonymous.4open.science/r/SVD-SSA-13BF/README.md}{Link}}.上获得



## **6. Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks**

具有自动复制/粘贴攻击的深度神经网络的诊断 cs.LG

Best paper award at the NeurIPS 2022 ML Safety Workshop --  https://neurips2022.mlsafety.org/

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2211.10024v3) [paper-pdf](http://arxiv.org/pdf/2211.10024v3)

**Authors**: Stephen Casper, Kaivalya Hariharan, Dylan Hadfield-Menell

**Abstract**: This paper considers the problem of helping humans exercise scalable oversight over deep neural networks (DNNs). Adversarial examples can be useful by helping to reveal weaknesses in DNNs, but they can be difficult to interpret or draw actionable conclusions from. Some previous works have proposed using human-interpretable adversarial attacks including copy/paste attacks in which one natural image pasted into another causes an unexpected misclassification. We build on these with two contributions. First, we introduce Search for Natural Adversarial Features Using Embeddings (SNAFUE) which offers a fully automated method for finding copy/paste attacks. Second, we use SNAFUE to red team an ImageNet classifier. We reproduce copy/paste attacks from previous works and find hundreds of other easily-describable vulnerabilities, all without a human in the loop. Code is available at https://github.com/thestephencasper/snafue

摘要: 本文研究了帮助人类对深度神经网络进行可扩展监督的问题。对抗性的例子可以通过帮助揭示DNN中的弱点而有用，但它们可能很难解释或从中得出可操作的结论。以前的一些工作已经提出使用人类可解释的对抗性攻击，包括复制/粘贴攻击，在这种攻击中，一幅自然图像粘贴到另一幅图像中会导致意外的错误分类。我们在这些基础上做出了两项贡献。首先，我们介绍了使用嵌入搜索自然对抗性特征(SNAFUE)，它提供了一种全自动的方法来发现复制/粘贴攻击。其次，我们使用SNAFUE对ImageNet分类器进行分组。我们复制了以前的作品中的复制/粘贴攻击，并发现了数百个其他容易描述的漏洞，所有这些都没有人参与。代码可在https://github.com/thestephencasper/snafue上找到



## **7. Efficient Adversarial Contrastive Learning via Robustness-Aware Coreset Selection**

基于鲁棒性感知CoReset选择的高效对抗性对比学习 cs.LG

**SubmitDate**: 2023-05-05    [abs](http://arxiv.org/abs/2302.03857v3) [paper-pdf](http://arxiv.org/pdf/2302.03857v3)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL) does not require expensive data annotations but outputs a robust representation that withstands adversarial attacks and also generalizes to a wide range of downstream tasks. However, ACL needs tremendous running time to generate the adversarial variants of all training data, which limits its scalability to large datasets. To speed up ACL, this paper proposes a robustness-aware coreset selection (RCS) method. RCS does not require label information and searches for an informative subset that minimizes a representational divergence, which is the distance of the representation between natural data and their virtual adversarial variants. The vanilla solution of RCS via traversing all possible subsets is computationally prohibitive. Therefore, we theoretically transform RCS into a surrogate problem of submodular maximization, of which the greedy search is an efficient solution with an optimality guarantee for the original problem. Empirically, our comprehensive results corroborate that RCS can speed up ACL by a large margin without significantly hurting the robustness transferability. Notably, to the best of our knowledge, we are the first to conduct ACL efficiently on the large-scale ImageNet-1K dataset to obtain an effective robust representation via RCS.

摘要: 对抗性对比学习(ACL)不需要昂贵的数据标注，但输出了一种稳健的表示，可以抵抗对抗性攻击，并适用于广泛的下游任务。然而，ACL需要大量的运行时间来生成所有训练数据的对抗性变体，这限制了其在大数据集上的可扩展性。为了提高访问控制列表的速度，提出了一种健壮性感知的核心重置选择(RCS)方法。RCS不需要标签信息，并且搜索最小化表示分歧的信息子集，表示分歧是自然数据和它们的虚拟对抗性变体之间的表示距离。通过遍历所有可能子集的RCS的香草解在计算上是令人望而却步的。因此，我们从理论上将RCS问题转化为子模最大化的代理问题，其中贪婪搜索是原问题的最优性保证的有效解。实验结果表明，RCS在不影响健壮性和可转移性的前提下，可以大幅度地提高ACL的速度。值得注意的是，据我们所知，我们是第一个在大规模ImageNet-1K数据集上高效地进行ACL的人，通过RCS获得了有效的健壮表示。



## **8. Single Node Injection Label Specificity Attack on Graph Neural Networks via Reinforcement Learning**

基于强化学习的图神经网络单节点注入标签专用性攻击 cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02901v1) [paper-pdf](http://arxiv.org/pdf/2305.02901v1)

**Authors**: Dayuan Chen, Jian Zhang, Yuqian Lv, Jinhuan Wang, Hongjie Ni, Shanqing Yu, Zhen Wang, Qi Xuan

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various real-world applications. However, recent studies highlight the vulnerability of GNNs to malicious perturbations. Previous adversaries primarily focus on graph modifications or node injections to existing graphs, yielding promising results but with notable limitations. Graph modification attack~(GMA) requires manipulation of the original graph, which is often impractical, while graph injection attack~(GIA) necessitates training a surrogate model in the black-box setting, leading to significant performance degradation due to divergence between the surrogate architecture and the actual victim model. Furthermore, most methods concentrate on a single attack goal and lack a generalizable adversary to develop distinct attack strategies for diverse goals, thus limiting precise control over victim model behavior in real-world scenarios. To address these issues, we present a gradient-free generalizable adversary that injects a single malicious node to manipulate the classification result of a target node in the black-box evasion setting. We propose Gradient-free Generalizable Single Node Injection Attack, namely G$^2$-SNIA, a reinforcement learning framework employing Proximal Policy Optimization. By directly querying the victim model, G$^2$-SNIA learns patterns from exploration to achieve diverse attack goals with extremely limited attack budgets. Through comprehensive experiments over three acknowledged benchmark datasets and four prominent GNNs in the most challenging and realistic scenario, we demonstrate the superior performance of our proposed G$^2$-SNIA over the existing state-of-the-art baselines. Moreover, by comparing G$^2$-SNIA with multiple white-box evasion baselines, we confirm its capacity to generate solutions comparable to those of the best adversaries.

摘要: 图神经网络(GNN)在各种实际应用中取得了显著的成功。然而，最近的研究强调了GNN对恶意扰动的脆弱性。以前的对手主要集中在修改图或向现有图注入节点，产生了有希望的结果，但具有显著的局限性。图修改攻击~(GMA)需要对原始图进行操作，这往往是不切实际的，而图注入攻击~(GIA)需要在黑盒环境下训练代理模型，由于代理体系结构与实际受害者模型之间的差异，导致性能显著下降。此外，大多数方法集中在单个攻击目标上，缺乏一个可概括的对手来针对不同的目标制定不同的攻击策略，从而限制了对现实场景中受害者模型行为的精确控制。为了解决这些问题，我们提出了一种无梯度泛化攻击，在黑盒规避环境下注入单个恶意节点来操纵目标节点的分类结果。本文提出了一种无梯度泛化单节点注入攻击，即G$^2$-SNIA，这是一种基于近邻策略优化的强化学习框架。通过直接查询受害者模型，G$^2$-SNIA从探索中学习模式，以极其有限的攻击预算实现不同的攻击目标。通过在最具挑战性和最现实的场景中对三个公认的基准数据集和四个重要的GNN进行全面的实验，我们证明了我们提出的G$^2$-SNIA比现有的最先进的基线具有更好的性能。此外，通过将G$^2$-SNIA与多个白盒规避基线进行比较，我们证实了它产生与最好的对手相当的解的能力。



## **9. IMAP: Intrinsically Motivated Adversarial Policy**

IMAP：内在动机的对抗政策 cs.LG

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02605v1) [paper-pdf](http://arxiv.org/pdf/2305.02605v1)

**Authors**: Xiang Zheng, Xingjun Ma, Shengjie Wang, Xinyu Wang, Chao Shen, Cong Wang

**Abstract**: Reinforcement learning (RL) agents are known to be vulnerable to evasion attacks during deployment. In single-agent environments, attackers can inject imperceptible perturbations on the policy or value network's inputs or outputs; in multi-agent environments, attackers can control an adversarial opponent to indirectly influence the victim's observation. Adversarial policies offer a promising solution to craft such attacks. Still, current approaches either require perfect or partial knowledge of the victim policy or suffer from sample inefficiency due to the sparsity of task-related rewards. To overcome these limitations, we propose the Intrinsically Motivated Adversarial Policy (IMAP) for efficient black-box evasion attacks in single- and multi-agent environments without any knowledge of the victim policy. IMAP uses four intrinsic objectives based on state coverage, policy coverage, risk, and policy divergence to encourage exploration and discover stronger attacking skills. We also design a novel Bias-Reduction (BR) method to boost IMAP further. Our experiments demonstrate the effectiveness of these intrinsic objectives and BR in improving adversarial policy learning in the black-box setting against multiple types of victim agents in various single- and multi-agent MuJoCo environments. Notably, our IMAP reduces the performance of the state-of-the-art robust WocaR-PPO agents by 34\%-54\% and achieves a SOTA attacking success rate of 83.91\% in the two-player zero-sum game YouShallNotPass.

摘要: 强化学习(RL)代理在部署过程中容易受到逃避攻击。在单智能体环境中，攻击者可以在策略或价值网络的输入或输出上注入不可察觉的扰动；在多智能体环境中，攻击者可以控制对手来间接影响受害者的观察。对抗性政策为精心策划此类攻击提供了一个前景看好的解决方案。尽管如此，目前的方法要么需要完全或部分了解受害者政策，要么由于与任务相关的奖励稀少而导致样本效率低下。为了克服这些局限性，我们提出了本质动机对抗策略(IMAP)，用于在不知道受害者策略的情况下，在单代理和多代理环境中进行有效的黑盒逃避攻击。IMAP使用基于州覆盖、政策覆盖、风险和政策分歧的四个内在目标来鼓励探索和发现更强大的攻击技能。我们还设计了一种新的偏置减少(BR)方法来进一步提高IMAP。我们的实验证明了这些内在目标和BR在黑箱环境中针对各种单代理和多代理MuJoCo环境中多种受害者代理的对抗策略学习的有效性。值得注意的是，我们的IMAP使最先进的健壮WocaR-PPO代理的性能降低了34-54，并在两人零和游戏YouShallNotPass中实现了83.91\%的SOTA攻击成功率。



## **10. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

MAdvex：机器学习恶意软件检测中基于工具的敌意攻击 cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2305.02559v1) [paper-pdf](http://arxiv.org/pdf/2305.02559v1)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.

摘要: WebAssembly(WASM)是一种用于Web应用程序的低级二进制格式，由于其改进的性能和与现有软件的兼容性而被广泛采用。然而，WASM的流行也导致了对其进行恶意攻击，例如加密劫持，即恶意行为者在未经受害者同意的情况下使用受害者的计算资源来挖掘加密货币。为了应对这种威胁，出现了基于机器学习的检测方法，旨在识别WASM代码中的加密劫持活动。众所周知，神经网络容易受到敌意攻击，在这种攻击中，分类器的输入会受到干扰，只需进行极小的更改，就会导致粗略的错误分类。虽然在图像分类中应用更改很容易，但在不更改功能的情况下以自动方式操作二进制文件来规避恶意软件分类并不是一件容易的事情。在这项工作中，我们提出了一种新的方法，通过插装将对抗性示例包括在二进制文件的代码部分中。引入的小工具允许包含任意字节，从而实现了高效的对抗性攻击，可靠地绕过了最先进的机器学习分类器，例如最近在NDSS 2021上提出的基于CNN的Minos。我们分析了基于插桩的对抗性实例生成的代价和可靠性，结果表明该方法在最小规模和最小性能开销的情况下能够可靠地工作。



## **11. Detecting Adversarial Faces Using Only Real Face Self-Perturbations**

仅利用真实人脸自扰动检测敌方人脸 cs.CV

IJCAI2023

**SubmitDate**: 2023-05-04    [abs](http://arxiv.org/abs/2304.11359v2) [paper-pdf](http://arxiv.org/pdf/2304.11359v2)

**Authors**: Qian Wang, Yongqin Xian, Hefei Ling, Jinyuan Zhang, Xiaorui Lin, Ping Li, Jiazhong Chen, Ning Yu

**Abstract**: Adversarial attacks aim to disturb the functionality of a target system by adding specific noise to the input samples, bringing potential threats to security and robustness when applied to facial recognition systems. Although existing defense techniques achieve high accuracy in detecting some specific adversarial faces (adv-faces), new attack methods especially GAN-based attacks with completely different noise patterns circumvent them and reach a higher attack success rate. Even worse, existing techniques require attack data before implementing the defense, making it impractical to defend newly emerging attacks that are unseen to defenders. In this paper, we investigate the intrinsic generality of adv-faces and propose to generate pseudo adv-faces by perturbing real faces with three heuristically designed noise patterns. We are the first to train an adv-face detector using only real faces and their self-perturbations, agnostic to victim facial recognition systems, and agnostic to unseen attacks. By regarding adv-faces as out-of-distribution data, we then naturally introduce a novel cascaded system for adv-face detection, which consists of training data self-perturbations, decision boundary regularization, and a max-pooling-based binary classifier focusing on abnormal local color aberrations. Experiments conducted on LFW and CelebA-HQ datasets with eight gradient-based and two GAN-based attacks validate that our method generalizes to a variety of unseen adversarial attacks.

摘要: 敌意攻击的目的是通过在输入样本中添加特定的噪声来扰乱目标系统的功能，从而在应用于面部识别系统时给安全性和健壮性带来潜在威胁。虽然现有的防御技术对某些特定的敌方人脸(adv-Faces)的检测准确率很高，但新的攻击方法，特别是具有完全不同噪声模式的基于GAN的攻击，可以绕过它们，达到更高的攻击成功率。更糟糕的是，现有技术在实施防御之前需要攻击数据，因此防御防御者看不到的新出现的攻击是不切实际的。在本文中，我们研究了广告人脸的内在共性，并提出了通过用三种启发式设计的噪声模式来扰动真实人脸来生成伪广告人脸的方法。我们是第一个只使用真实人脸及其自身扰动来训练Adv-Face检测器的人，对受害者面部识别系统不可知，对看不见的攻击不可知。通过将广告人脸看作非分布数据，我们自然地提出了一种新的级联广告人脸检测系统，该系统包括训练数据自扰动、决策边界正则化和基于最大池的二进制分类器，该分类器针对异常局部颜色偏差。在LFW和CelebA-HQ数据集上对8个基于梯度的攻击和2个基于GAN的攻击进行了实验，验证了我们的方法适用于各种看不见的对抗性攻击。



## **12. On the Security Risks of Knowledge Graph Reasoning**

论知识图推理的安全风险 cs.CR

In proceedings of USENIX Security'23. Codes:  https://github.com/HarrialX/security-risk-KG-reasoning

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.02383v1) [paper-pdf](http://arxiv.org/pdf/2305.02383v1)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Xiapu Luo, Xusheng Xiao, Fenglong Ma, Ting Wang

**Abstract**: Knowledge graph reasoning (KGR) -- answering complex logical queries over large knowledge graphs -- represents an important artificial intelligence task, entailing a range of applications (e.g., cyber threat hunting). However, despite its surging popularity, the potential security risks of KGR are largely unexplored, which is concerning, given the increasing use of such capability in security-critical domains.   This work represents a solid initial step towards bridging the striking gap. We systematize the security threats to KGR according to the adversary's objectives, knowledge, and attack vectors. Further, we present ROAR, a new class of attacks that instantiate a variety of such threats. Through empirical evaluation in representative use cases (e.g., medical decision support, cyber threat hunting, and commonsense reasoning), we demonstrate that ROAR is highly effective to mislead KGR to suggest pre-defined answers for target queries, yet with negligible impact on non-target ones. Finally, we explore potential countermeasures against ROAR, including filtering of potentially poisoning knowledge and training with adversarially augmented queries, which leads to several promising research directions.

摘要: 知识图推理(KGR)--在大型知识图上回答复杂的逻辑查询--代表着一项重要的人工智能任务，需要一系列应用程序(例如，网络威胁搜索)。然而，尽管KGR越来越受欢迎，但其潜在的安全风险在很大程度上还没有被探索出来，这是令人担忧的，因为这种能力在安全关键领域中的使用越来越多。这项工作是朝着弥合这一显著差距迈出的坚实的第一步。我们根据对手的目标、知识和攻击载体，对KGR面临的安全威胁进行系统化。此外，我们还介绍了咆哮，这是一种新的攻击类型，它实例化了各种此类威胁。通过对典型用例(如医疗决策支持、网络威胁搜索和常识推理)的实证评估，我们证明了Roar对于误导KGR为目标查询建议预定义答案是非常有效的，而对非目标查询的影响可以忽略不计。最后，我们探索了针对Roar的潜在对策，包括过滤潜在的中毒知识和使用恶意增强的查询进行训练，这导致了几个有前途的研究方向。



## **13. Privacy-Preserving Federated Recurrent Neural Networks**

隐私保护的联邦递归神经网络 cs.CR

Accepted for publication at the 23rd Privacy Enhancing Technologies  Symposium (PETS 2023)

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2207.13947v2) [paper-pdf](http://arxiv.org/pdf/2207.13947v2)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstract**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a cross-silo federated learning setting by relying on multiparty homomorphic encryption. RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates federated learning attacks that target the gradients under a passive-adversary threat model. We propose a packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, RHODE provides several clipping approximations for performing gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.

摘要: 我们提出了一种新的系统Rhode，它依靠多方同态加密在跨竖井的联合学习环境中实现对递归神经网络(RNN)的隐私保护训练和预测。Rhode保留了训练数据、模型和预测数据的机密性；它缓解了被动对手威胁模型下针对梯度的联合学习攻击。为了更好地利用加密环境下的单指令多数据(SIMD)运算，我们提出了一种多维打包方案。通过多维包装，Rhode能够并行高效地处理一批样品。为了避免爆炸渐变问题，Rhode提供了几种用于在加密情况下执行渐变裁剪的裁剪近似。我们的实验表明，对于数据持有者之间的同质和异质数据分布，Rhode模型的性能与非安全解决方案相似。我们的实验评估表明，Rhode与数据持有者数量和时间步数成线性关系，分别与RNN的特征数和隐含单元数成亚线性和次二次关系。据我们所知，Rhode是第一个在联合学习环境中加密的、为RNN及其变体的训练提供构建块的系统。



## **14. New Adversarial Image Detection Based on Sentiment Analysis**

一种新的基于情感分析的对抗性图像检测 cs.CR

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.03173v1) [paper-pdf](http://arxiv.org/pdf/2305.03173v1)

**Authors**: Yulong Wang, Tianxiang Li, Shenghong Li, Xin Yuan, Wei Ni

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, while adversarial attack models, e.g., DeepFool, are on the rise and outrunning adversarial example detection techniques. This paper presents a new adversarial example detector that outperforms state-of-the-art detectors in identifying the latest adversarial attacks on image datasets. Specifically, we propose to use sentiment analysis for adversarial example detection, qualified by the progressively manifesting impact of an adversarial perturbation on the hidden-layer feature maps of a DNN under attack. Accordingly, we design a modularized embedding layer with the minimum learnable parameters to embed the hidden-layer feature maps into word vectors and assemble sentences ready for sentiment analysis. Extensive experiments demonstrate that the new detector consistently surpasses the state-of-the-art detection algorithms in detecting the latest attacks launched against ResNet and Inception neutral networks on the CIFAR-10, CIFAR-100 and SVHN datasets. The detector only has about 2 million parameters, and takes shorter than 4.6 milliseconds to detect an adversarial example generated by the latest attack models using a Tesla K80 GPU card.

摘要: 深度神经网络(DNN)容易受到敌意实例的影响，而DeepFool等敌意攻击模型正在兴起，并且超越了敌意实例检测技术。本文提出了一种新的敌意实例检测器，它在识别图像数据集上的最新敌意攻击方面优于最新的检测器。具体地说，我们建议使用情感分析来检测敌意示例，以敌意扰动对DNN在攻击下的隐含层特征映射的逐渐显现的影响为条件。相应地，我们设计了一个具有最小学习参数的模块化嵌入层，将隐含层特征映射嵌入到词向量中，并组装句子，为情感分析做好准备。大量的实验表明，新的检测器在检测针对CIFAR-10、CIFAR-100和SVHN数据集上的ResNet和Inception神经网络的最新攻击时，始终优于最新的检测算法。该检测器只有大约200万个参数，只需不到4.6毫秒就可以检测到使用特斯拉K80 GPU卡的最新攻击模型生成的对抗性示例。



## **15. Text Adversarial Purification as Defense against Adversarial Attacks**

文本对抗性净化作为对抗攻击的防御 cs.CL

Accepted by ACL2023 main conference

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2203.14207v2) [paper-pdf](http://arxiv.org/pdf/2203.14207v2)

**Authors**: Linyang Li, Demin Song, Xipeng Qiu

**Abstract**: Adversarial purification is a successful defense mechanism against adversarial attacks without requiring knowledge of the form of the incoming attack. Generally, adversarial purification aims to remove the adversarial perturbations therefore can make correct predictions based on the recovered clean samples. Despite the success of adversarial purification in the computer vision field that incorporates generative models such as energy-based models and diffusion models, using purification as a defense strategy against textual adversarial attacks is rarely explored. In this work, we introduce a novel adversarial purification method that focuses on defending against textual adversarial attacks. With the help of language models, we can inject noise by masking input texts and reconstructing the masked texts based on the masked language models. In this way, we construct an adversarial purification process for textual models against the most widely used word-substitution adversarial attacks. We test our proposed adversarial purification method on several strong adversarial attack methods including Textfooler and BERT-Attack and experimental results indicate that the purification algorithm can successfully defend against strong word-substitution attacks.

摘要: 对抗性净化是一种成功的防御对抗性攻击的机制，不需要知道即将到来的攻击的形式。通常，对抗性净化的目的是消除对抗性扰动，因此可以基于恢复的干净样本做出正确的预测。尽管对抗性净化在计算机视觉领域取得了成功，它结合了生成模型，如基于能量的模型和扩散模型，但将净化作为一种防御文本对抗攻击的策略却很少被探索。在这项工作中，我们介绍了一种新的对抗净化方法，该方法专注于防御文本对抗攻击。在语言模型的帮助下，我们可以通过对输入文本进行掩蔽并基于掩蔽的语言模型重建被掩蔽的文本来注入噪声。通过这种方式，我们为文本模型构建了对抗最广泛使用的词替换对抗攻击的对抗净化过程。我们在几种强对抗性攻击方法上对我们提出的对抗性净化方法进行了测试，实验结果表明，该净化算法能够成功地防御强词替换攻击。



## **16. Adversarial Neon Beam: Robust Physical-World Adversarial Attack to DNNs**

对抗性霓虹灯：对DNN的强大物理世界对抗性攻击 cs.CV

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2204.00853v2) [paper-pdf](http://arxiv.org/pdf/2204.00853v2)

**Authors**: Chengyin Hu, Kalibinuer Tiliwalidi

**Abstract**: In the physical world, light affects the performance of deep neural networks. Nowadays, many products based on deep neural network have been put into daily life. There are few researches on the effect of light on the performance of deep neural network models. However, the adversarial perturbations generated by light may have extremely dangerous effects on these systems. In this work, we propose an attack method called adversarial neon beam (AdvNB), which can execute the physical attack by obtaining the physical parameters of adversarial neon beams with very few queries. Experiments show that our algorithm can achieve advanced attack effect in both digital test and physical test. In the digital environment, 99.3% attack success rate was achieved, and in the physical environment, 100% attack success rate was achieved. Compared with the most advanced physical attack methods, our method can achieve better physical perturbation concealment. In addition, by analyzing the experimental data, we reveal some new phenomena brought about by the adversarial neon beam attack.

摘要: 在物理世界中，光线会影响深度神经网络的性能。如今，许多基于深度神经网络的产品已经进入日常生活。关于光照对深度神经网络模型性能影响的研究很少。然而，光产生的对抗性扰动可能会对这些系统产生极其危险的影响。在这项工作中，我们提出了一种称为对抗性霓虹束的攻击方法(AdvNB)，该方法只需很少的查询就可以获得对抗性霓虹束的物理参数来执行物理攻击。实验表明，该算法在数字测试和物理测试中均能达到较好的攻击效果。在数字环境下，攻击成功率达到99.3%，在物理环境下，攻击成功率达到100%。与最先进的物理攻击方法相比，我们的方法可以实现更好的物理扰动隐藏。此外，通过对实验数据的分析，揭示了对抗性霓虹束攻击带来的一些新现象。



## **17. How Far Are We from Real Synonym Substitution Attacks?**

我们离真正的同义词替换攻击还有多远？ cs.CL

Findings in ACL 2023

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2210.02844v2) [paper-pdf](http://arxiv.org/pdf/2210.02844v2)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: In this paper, we explore the following question: how far are we from real synonym substitution attacks (SSAs). We approach this question by examining how SSAs replace words in the original sentence and show that there are still unresolved obstacles that make current SSAs generate invalid adversarial samples. We reveal that four widely used word substitution methods generate a large fraction of invalid substitution words that are ungrammatical or do not preserve the original sentence's semantics. Next, we show that the semantic and grammatical constraints used in SSAs for detecting invalid word replacements are highly insufficient in detecting invalid adversarial samples. Our work is an important stepping stone to constructing better SSAs in the future.

摘要: 在本文中，我们探讨了以下问题：我们距离真正的同义词替换攻击(SSA)还有多远。我们通过审查SSA如何替换原始句子中的单词来处理这个问题，并表明仍然存在尚未解决的障碍，使当前的SSA生成无效的对抗性样本。我们发现，四种广泛使用的词替换方法产生了很大一部分无效替换词，这些词不符合语法或没有保留原始句子的语义。其次，我们证明了SSA中用于检测无效单词替换的语义和语法约束在检测无效对抗性样本方面严重不足。我们的工作是未来构建更好的SSA的重要垫脚石。



## **18. Can Large Language Models Be an Alternative to Human Evaluations?**

大型语言模型能替代人类评估吗？ cs.CL

ACL 2023 main conference paper. Main content: 10 pages (including  limitations). Appendix: 13 pages

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01937v1) [paper-pdf](http://arxiv.org/pdf/2305.01937v1)

**Authors**: Cheng-Han Chiang, Hung-yi Lee

**Abstract**: Human evaluation is indispensable and inevitable for assessing the quality of texts generated by machine learning models or written by humans. However, human evaluation is very difficult to reproduce and its quality is notoriously unstable, hindering fair comparisons among different natural language processing (NLP) models and algorithms. Recently, large language models (LLMs) have demonstrated exceptional performance on unseen tasks when only the task instructions are provided. In this paper, we explore if such an ability of the LLMs can be used as an alternative to human evaluation. We present the LLMs with the exact same instructions, samples to be evaluated, and questions used to conduct human evaluation, and then ask the LLMs to generate responses to those questions; we dub this LLM evaluation. We use human evaluation and LLM evaluation to evaluate the texts in two NLP tasks: open-ended story generation and adversarial attacks. We show that the result of LLM evaluation is consistent with the results obtained by expert human evaluation: the texts rated higher by human experts are also rated higher by the LLMs. We also find that the results of LLM evaluation are stable over different formatting of the task instructions and the sampling algorithm used to generate the answer. We are the first to show the potential of using LLMs to assess the quality of texts and discuss the limitations and ethical considerations of LLM evaluation.

摘要: 对于机器学习模型生成的文本或由人类编写的文本的质量进行评估时，人工评估是必不可少的，也是不可避免的。然而，人类的评价很难重现，而且其质量是出了名的不稳定，阻碍了不同自然语言处理(NLP)模型和算法之间的公平比较。最近，大型语言模型(LLM)在只提供任务指令的情况下，在看不见的任务上表现出了出色的性能。在本文中，我们探索LLMS的这种能力是否可以用作人类评估的替代方案。我们向LLMS提供完全相同的指令、要评估的样本和用于进行人工评估的问题，然后要求LLMS对这些问题做出回应；我们将这种LLM评估称为LLM评估。我们使用人类评价和LLM评价来评价两个自然语言处理任务中的文本：开放式故事生成和对抗性攻击。结果表明，LLM评价的结果与专家人类评价的结果是一致的：人类专家评得越高的文本，LLMS也会给出更高的评价。我们还发现，在不同的任务指令格式和用于生成答案的采样算法下，LLM评估的结果是稳定的。我们首先展示了使用LLMS评估文本质量的潜力，并讨论了LLM评估的局限性和伦理考虑。



## **19. Towards Imperceptible Document Manipulations against Neural Ranking Models**

针对神经排序模型的不可感知文档操作 cs.IR

Accepted to Findings of ACL 2023

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2305.01860v1) [paper-pdf](http://arxiv.org/pdf/2305.01860v1)

**Authors**: Xuanang Chen, Ben He, Zheng Ye, Le Sun, Yingfei Sun

**Abstract**: Adversarial attacks have gained traction in order to identify potential vulnerabilities in neural ranking models (NRMs), but current attack methods often introduce grammatical errors, nonsensical expressions, or incoherent text fragments, which can be easily detected. Additionally, current methods rely heavily on the use of a well-imitated surrogate NRM to guarantee the attack effect, which makes them difficult to use in practice. To address these issues, we propose a framework called Imperceptible DocumEnt Manipulation (IDEM) to produce adversarial documents that are less noticeable to both algorithms and humans. IDEM instructs a well-established generative language model, such as BART, to generate connection sentences without introducing easy-to-detect errors, and employs a separate position-wise merging strategy to balance relevance and coherence of the perturbed text. Experimental results on the popular MS MARCO benchmark demonstrate that IDEM can outperform strong baselines while preserving fluency and correctness of the target documents as evidenced by automatic and human evaluations. Furthermore, the separation of adversarial text generation from the surrogate NRM makes IDEM more robust and less affected by the quality of the surrogate NRM.

摘要: 为了识别神经排名模型(NRM)中的潜在漏洞，对抗性攻击已经获得了吸引力，但目前的攻击方法通常会引入语法错误、无意义的表达或不连贯的文本片段，这些都很容易被检测到。此外，目前的方法严重依赖于使用一个模仿得很好的代理NRM来保证攻击效果，这使得它们在实践中很难使用。为了解决这些问题，我们提出了一个称为不可感知文档操作(IDEM)的框架，以生成算法和人类都不太注意的敌意文档。Idem指导一个成熟的生成语言模型，如BART，在不引入容易检测的错误的情况下生成连接句子，并采用单独的位置合并策略来平衡扰动文本的关联性和连贯性。在流行的MS Marco基准上的实验结果表明，IDEM的性能优于强基线，同时保持了目标文档的流畅性和正确性，这一点得到了自动和人工评估的证明。此外，将敌意文本生成从代理NRM中分离出来，使得IDEM更健壮，受代理NRM质量的影响更小。



## **20. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

高分：使用生成模型对对抗性扰动进行全局稳健性评估 cs.LG

**SubmitDate**: 2023-05-03    [abs](http://arxiv.org/abs/2304.09875v2) [paper-pdf](http://arxiv.org/pdf/2304.09875v2)

**Authors**: Zaitang Li, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **21. Sentiment Perception Adversarial Attacks on Neural Machine Translation Systems**

神经机器翻译系统中情感感知的敌意攻击 cs.CL

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01437v1) [paper-pdf](http://arxiv.org/pdf/2305.01437v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: With the advent of deep learning methods, Neural Machine Translation (NMT) systems have become increasingly powerful. However, deep learning based systems are susceptible to adversarial attacks, where imperceptible changes to the input can cause undesirable changes at the output of the system. To date there has been little work investigating adversarial attacks on sequence-to-sequence systems, such as NMT models. Previous work in NMT has examined attacks with the aim of introducing target phrases in the output sequence. In this work, adversarial attacks for NMT systems are explored from an output perception perspective. Thus the aim of an attack is to change the perception of the output sequence, without altering the perception of the input sequence. For example, an adversary may distort the sentiment of translated reviews to have an exaggerated positive sentiment. In practice it is challenging to run extensive human perception experiments, so a proxy deep-learning classifier applied to the NMT output is used to measure perception changes. Experiments demonstrate that the sentiment perception of NMT systems' output sequences can be changed significantly.

摘要: 随着深度学习方法的出现，神经机器翻译(NMT)系统变得越来越强大。然而，基于深度学习的系统很容易受到对抗性攻击，在这种攻击中，输入的不可察觉的变化可能会导致系统输出的不希望看到的变化。到目前为止，很少有人研究针对序列到序列系统的对抗性攻击，例如NMT模型。NMT之前的工作已经检查了攻击，目的是在输出序列中引入目标短语。本文从输出感知的角度研究了NMT系统的敌意攻击问题。因此，攻击的目的是改变对输出序列的感知，而不改变对输入序列的感知。例如，对手可能会扭曲翻译后的评论的情绪，使其具有夸大的积极情绪。在实践中，进行广泛的人类感知实验是具有挑战性的，因此将代理深度学习分类器应用于NMT输出来测量感知变化。实验表明，NMT系统输出序列的情感感知可以发生显著变化。



## **22. Improving adversarial robustness by putting more regularizations on less robust samples**

通过对健壮性较差的样本进行更多的正则化来提高对手的稳健性 stat.ML

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2206.03353v3) [paper-pdf](http://arxiv.org/pdf/2206.03353v3)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.

摘要: 对抗性训练是为了提高对抗攻击的稳健性，因为它很容易产生人类无法察觉的数据扰动来欺骗给定的深度神经网络。在本文中，我们提出了一种新的对抗性训练算法，该算法在理论上动机良好，在经验上优于其他现有的算法。与现有的正则化算法相比，该算法的一个新特点是对易受敌意攻击的数据进行了更多的正则化。理论上，我们的算法可以理解为最小化正则化经验风险的算法，该正则化经验风险是由新导出的稳健风险上界引起的。数值实验表明，我们提出的算法同时提高了泛化(例题准确率)和稳健性(对抗性攻击准确率)，达到了最好的性能。



## **23. StyleFool: Fooling Video Classification Systems via Style Transfer**

StyleFool：通过样式转换愚弄视频分类系统 cs.CV

18 pages, 9 figures. Accepted to S&P 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2203.16000v3) [paper-pdf](http://arxiv.org/pdf/2203.16000v3)

**Authors**: Yuxin Cao, Xi Xiao, Ruoxi Sun, Derui Wang, Minhui Xue, Sheng Wen

**Abstract**: Video classification systems are vulnerable to adversarial attacks, which can create severe security problems in video verification. Current black-box attacks need a large number of queries to succeed, resulting in high computational overhead in the process of attack. On the other hand, attacks with restricted perturbations are ineffective against defenses such as denoising or adversarial training. In this paper, we focus on unrestricted perturbations and propose StyleFool, a black-box video adversarial attack via style transfer to fool the video classification system. StyleFool first utilizes color theme proximity to select the best style image, which helps avoid unnatural details in the stylized videos. Meanwhile, the target class confidence is additionally considered in targeted attacks to influence the output distribution of the classifier by moving the stylized video closer to or even across the decision boundary. A gradient-free method is then employed to further optimize the adversarial perturbations. We carry out extensive experiments to evaluate StyleFool on two standard datasets, UCF-101 and HMDB-51. The experimental results demonstrate that StyleFool outperforms the state-of-the-art adversarial attacks in terms of both the number of queries and the robustness against existing defenses. Moreover, 50% of the stylized videos in untargeted attacks do not need any query since they can already fool the video classification model. Furthermore, we evaluate the indistinguishability through a user study to show that the adversarial samples of StyleFool look imperceptible to human eyes, despite unrestricted perturbations.

摘要: 视频分类系统容易受到敌意攻击，这会给视频验证带来严重的安全问题。当前的黑盒攻击需要大量的查询才能成功，导致攻击过程中的计算开销很高。另一方面，受限扰动的攻击对诸如去噪或对抗性训练等防御措施无效。本文针对无限制扰动，提出了StyleFool，一种通过风格转移来欺骗视频分类系统的黑盒视频对抗性攻击。StyleFool首先利用颜色主题贴近度来选择最佳风格的图像，这有助于避免风格化视频中不自然的细节。同时，在有针对性的攻击中，还考虑了目标类置信度，通过将风格化视频移动到更接近甚至跨越决策边界的位置来影响分类器的输出分布。然后使用无梯度方法进一步优化对抗性扰动。我们在两个标准数据集UCF-101和HMDB-51上进行了大量的实验来评估StyleFool。实验结果表明，StyleFool在查询次数和对现有防御的健壮性方面都优于最先进的对抗性攻击。此外，在非定向攻击中，50%的风格化视频不需要任何查询，因为它们已经可以愚弄视频分类模型。此外，我们通过用户研究对StyleFool的不可区分性进行了评估，以表明StyleFool的敌意样本在人眼看来是不可察觉的，尽管存在无限的扰动。



## **24. Exposing Fine-Grained Adversarial Vulnerability of Face Anti-Spoofing Models**

暴露Face反欺骗模型的细粒度攻击漏洞 cs.CV

Accepted by IEEE/CVF Conference on Computer Vision and Pattern  Recognition (CVPR) Workshop, 2023

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2205.14851v3) [paper-pdf](http://arxiv.org/pdf/2205.14851v3)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Ziwen He, Bo Peng, Jing Dong

**Abstract**: Face anti-spoofing aims to discriminate the spoofing face images (e.g., printed photos) from live ones. However, adversarial examples greatly challenge its credibility, where adding some perturbation noise can easily change the predictions. Previous works conducted adversarial attack methods to evaluate the face anti-spoofing performance without any fine-grained analysis that which model architecture or auxiliary feature is vulnerable to the adversary. To handle this problem, we propose a novel framework to expose the fine-grained adversarial vulnerability of the face anti-spoofing models, which consists of a multitask module and a semantic feature augmentation (SFA) module. The multitask module can obtain different semantic features for further evaluation, but only attacking these semantic features fails to reflect the discrimination-related vulnerability. We then design the SFA module to introduce the data distribution prior for more discrimination-related gradient directions for generating adversarial examples. Comprehensive experiments show that SFA module increases the attack success rate by nearly 40$\%$ on average. We conduct this fine-grained adversarial analysis on different annotations, geometric maps, and backbone networks (e.g., Resnet network). These fine-grained adversarial examples can be used for selecting robust backbone networks and auxiliary features. They also can be used for adversarial training, which makes it practical to further improve the accuracy and robustness of the face anti-spoofing models.

摘要: 人脸反欺骗的目的是区分伪造的人脸图像(如打印的照片)和活的人脸图像。然而，对抗性的例子极大地挑战了它的可信度，在那里添加一些扰动噪声很容易改变预测。以往的工作采用对抗性攻击的方法来评估人脸的反欺骗性能，没有任何细粒度的分析来确定哪个模型、架构或辅助特征容易受到对手的攻击。为了解决这个问题，我们提出了一种新的框架来暴露人脸反欺骗模型的细粒度攻击漏洞，该框架由多任务模块和语义特征增强(SFA)模块组成。多任务模块可以获得不同的语义特征用于进一步的评估，但仅攻击这些语义特征并不能反映与歧视相关的脆弱性。然后，我们设计了SFA模块来引入数据分布，以获得更多与区分相关的梯度方向，以生成对抗性示例。综合实验表明，SFA模块的攻击成功率平均提高了近40美元。我们在不同的注释、几何地图和骨干网络(例如RESNET网络)上进行了这种细粒度的对抗性分析。这些细粒度的对抗性实例可用于选择健壮的主干网络和辅助特征。它们还可以用于对抗性训练，从而进一步提高人脸反欺骗模型的准确性和稳健性。



## **25. Stratified Adversarial Robustness with Rejection**

具有拒绝的分层对抗健壮性 cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01139v1) [paper-pdf](http://arxiv.org/pdf/2305.01139v1)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

摘要: 最近，对抗性地训练具有拒绝选项的分类器(也称为选择性分类器)以增强对抗性健壮性是一种新的兴趣。虽然拒绝在许多应用中可能会导致成本，但现有研究通常将零成本与拒绝扰动输入联系在一起，这可能导致拒绝许多可以正确分类的轻微扰动输入。在这项工作中，我们研究了分层拒绝环境下的具有拒绝的对抗性鲁棒分类，其中拒绝代价由拒绝损失函数来建模，拒绝损失函数在扰动幅度上单调地不增加。我们从理论上分析了分层拒绝的设置，并提出了一种新的防御方法--基于一致预测拒绝的对抗训练(CPR)--来构建一个健壮的选择性分类器。在图像数据集上的实验表明，该方法在强自适应攻击下的性能明显优于已有方法。例如，在CIFAR-10上，CPR在看得见和看不见的攻击下都将总的稳健损失(针对不同的拒绝损失)减少了至少7.3%。



## **26. Randomized Reversible Gate-Based Obfuscation for Secured Compilation of Quantum Circuit**

基于随机可逆门的量子电路安全编译混淆算法 quant-ph

11 pages, 12 figures, conference

**SubmitDate**: 2023-05-02    [abs](http://arxiv.org/abs/2305.01133v1) [paper-pdf](http://arxiv.org/pdf/2305.01133v1)

**Authors**: Subrata Das, Swaroop Ghosh

**Abstract**: The success of quantum circuits in providing reliable outcomes for a given problem depends on the gate count and depth in near-term noisy quantum computers. Quantum circuit compilers that decompose high-level gates to native gates of the hardware and optimize the circuit play a key role in quantum computing. However, the quality and time complexity of the optimization process can vary significantly especially for practically relevant large-scale quantum circuits. As a result, third-party (often less-trusted/untrusted) compilers have emerged, claiming to provide better and faster optimization of complex quantum circuits than so-called trusted compilers. However, untrusted compilers can pose severe security risks, such as the theft of sensitive intellectual property (IP) embedded within the quantum circuit. We propose an obfuscation technique for quantum circuits using randomized reversible gates to protect them from such attacks during compilation. The idea is to insert a small random circuit into the original circuit and send it to the untrusted compiler. Since the circuit function is corrupted, the adversary may get incorrect IP. However, the user may also get incorrect output post-compilation. To circumvent this issue, we concatenate the inverse of the random circuit in the compiled circuit to recover the original functionality. We demonstrate the practicality of our method by conducting exhaustive experiments on a set of benchmark circuits and measuring the quality of obfuscation by calculating the Total Variation Distance (TVD) metric. Our method achieves TVD of up to 1.92 and performs at least 2X better than a previously reported obfuscation method. We also propose a novel adversarial reverse engineering (RE) approach and show that the proposed obfuscation is resilient against RE attacks. The proposed technique introduces minimal degradation in fidelity (~1% to ~3% on average).

摘要: 量子电路在为给定问题提供可靠结果方面的成功取决于近期嘈杂的量子计算机中的门数量和深度。量子电路编译器将高层门分解为硬件的本机门，并对电路进行优化，在量子计算中发挥着关键作用。然而，优化过程的质量和时间复杂性可能会有很大的变化，特别是对于实际相关的大规模量子电路。因此，第三方(通常不太可信/不可信)编译器应运而生，声称比所谓的可信编译器提供更好、更快的复杂量子电路优化。然而，不可信的编译器可能会带来严重的安全风险，例如嵌入量子电路中的敏感知识产权(IP)被窃取。我们提出了一种量子电路的混淆技术，该技术使用随机化的可逆门来保护它们在编译过程中免受此类攻击。其想法是在原始电路中插入一个小的随机电路，并将其发送给不可信的编译器。由于电路功能被破坏，对手可能获得错误的IP。但是，用户也可能在编译后得到不正确的输出。为了避免这个问题，我们将编译电路中的随机电路的逆连接起来，以恢复原来的功能。我们在一组基准电路上进行了详尽的实验，并通过计算总变化距离(TVD)度量来衡量混淆质量，从而证明了该方法的实用性。我们的方法获得了高达1.92的TVD，并且比先前报道的混淆方法的性能至少提高了2倍。我们还提出了一种新的对抗性逆向工程(RE)方法，并证明了该方法对逆向工程攻击具有较强的抵抗力。提出的技术在保真度方面引入了最小的降级(平均~1%到~3%)。



## **27. Evaluating Adversarial Robustness on Document Image Classification**

文档图像分类中的对抗健壮性评价 cs.CV

The 17th International Conference on Document Analysis and  Recognition

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2304.12486v2) [paper-pdf](http://arxiv.org/pdf/2304.12486v2)

**Authors**: Timothée Fronteau, Arnaud Paran, Aymen Shabou

**Abstract**: Adversarial attacks and defenses have gained increasing interest on computer vision systems in recent years, but as of today, most investigations are limited to images. However, many artificial intelligence models actually handle documentary data, which is very different from real world images. Hence, in this work, we try to apply the adversarial attack philosophy on documentary and natural data and to protect models against such attacks. We focus our work on untargeted gradient-based, transfer-based and score-based attacks and evaluate the impact of adversarial training, JPEG input compression and grey-scale input transformation on the robustness of ResNet50 and EfficientNetB0 model architectures. To the best of our knowledge, no such work has been conducted by the community in order to study the impact of these attacks on the document image classification task.

摘要: 近年来，对抗性攻击和防御对计算机视觉系统产生了越来越大的兴趣，但截至目前，大多数调查仅限于图像。然而，许多人工智能模型实际上处理的是纪实数据，这与现实世界的图像有很大不同。因此，在这项工作中，我们试图将对抗性攻击的理念应用于文献和自然数据，并保护模型免受此类攻击。我们的工作集中在基于非目标梯度、基于转移和基于分数的攻击上，并评估了对抗性训练、JPEG输入压缩和灰度输入变换对ResNet50和EfficientNetB0模型架构的健壮性的影响。据我们所知，社区还没有进行过这样的工作，以研究这些攻击对文档图像分类任务的影响。



## **28. Physical Adversarial Attacks for Surveillance: A Survey**

用于监视的物理对抗性攻击：综述 cs.CV

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.01074v1) [paper-pdf](http://arxiv.org/pdf/2305.01074v1)

**Authors**: Kien Nguyen, Tharindu Fernando, Clinton Fookes, Sridha Sridharan

**Abstract**: Modern automated surveillance techniques are heavily reliant on deep learning methods. Despite the superior performance, these learning systems are inherently vulnerable to adversarial attacks - maliciously crafted inputs that are designed to mislead, or trick, models into making incorrect predictions. An adversary can physically change their appearance by wearing adversarial t-shirts, glasses, or hats or by specific behavior, to potentially avoid various forms of detection, tracking and recognition of surveillance systems; and obtain unauthorized access to secure properties and assets. This poses a severe threat to the security and safety of modern surveillance systems. This paper reviews recent attempts and findings in learning and designing physical adversarial attacks for surveillance applications. In particular, we propose a framework to analyze physical adversarial attacks and provide a comprehensive survey of physical adversarial attacks on four key surveillance tasks: detection, identification, tracking, and action recognition under this framework. Furthermore, we review and analyze strategies to defend against the physical adversarial attacks and the methods for evaluating the strengths of the defense. The insights in this paper present an important step in building resilience within surveillance systems to physical adversarial attacks.

摘要: 现代自动监控技术严重依赖深度学习方法。尽管性能优越，但这些学习系统天生就容易受到敌意攻击--恶意设计的输入旨在误导或欺骗模型做出错误的预测。敌手可以通过穿着敌意的t恤、眼镜或帽子或通过特定的行为来改变自己的外表，以潜在地避免监视系统的各种形式的检测、跟踪和识别；并获得对安全财产和资产的未经授权的访问。这对现代监控系统的安全保障构成了严重威胁。本文回顾了最近在学习和设计用于监视应用的物理对抗性攻击方面的尝试和发现。特别是，我们提出了一个分析物理对抗攻击的框架，并在该框架下对物理对抗攻击的四个关键监视任务：检测、识别、跟踪和动作识别进行了全面的调查。此外，我们还回顾和分析了防御物理对抗性攻击的策略和评估防御强度的方法。本文的见解代表了在监视系统中建立对物理对手攻击的复原力的重要一步。



## **29. IoTFlowGenerator: Crafting Synthetic IoT Device Traffic Flows for Cyber Deception**

IoTFlowGenerator：为网络欺骗精心制作合成物联网设备流量 cs.CR

FLAIRS-36

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00925v1) [paper-pdf](http://arxiv.org/pdf/2305.00925v1)

**Authors**: Joseph Bao, Murat Kantarcioglu, Yevgeniy Vorobeychik, Charles Kamhoua

**Abstract**: Over the years, honeypots emerged as an important security tool to understand attacker intent and deceive attackers to spend time and resources. Recently, honeypots are being deployed for Internet of things (IoT) devices to lure attackers, and learn their behavior. However, most of the existing IoT honeypots, even the high interaction ones, are easily detected by an attacker who can observe honeypot traffic due to lack of real network traffic originating from the honeypot. This implies that, to build better honeypots and enhance cyber deception capabilities, IoT honeypots need to generate realistic network traffic flows. To achieve this goal, we propose a novel deep learning based approach for generating traffic flows that mimic real network traffic due to user and IoT device interactions. A key technical challenge that our approach overcomes is scarcity of device-specific IoT traffic data to effectively train a generator. We address this challenge by leveraging a core generative adversarial learning algorithm for sequences along with domain specific knowledge common to IoT devices. Through an extensive experimental evaluation with 18 IoT devices, we demonstrate that the proposed synthetic IoT traffic generation tool significantly outperforms state of the art sequence and packet generators in remaining indistinguishable from real traffic even to an adaptive attacker.

摘要: 多年来，蜜罐成为一种重要的安全工具，用于了解攻击者的意图并欺骗攻击者花费时间和资源。最近，物联网(IoT)设备正在部署蜜罐，以引诱攻击者，并了解他们的行为。然而，由于缺乏源自蜜罐的真实网络流量，大多数现有的物联网蜜罐，即使是高交互的蜜罐，也很容易被攻击者检测到，攻击者可以观察到蜜罐流量。这意味着，为了构建更好的蜜罐并增强网络欺骗能力，物联网蜜罐需要生成真实的网络流量。为了实现这一目标，我们提出了一种新的基于深度学习的方法来生成模拟真实网络流量的用户和物联网设备交互流量。我们的方法克服的一个关键技术挑战是缺乏特定于设备的物联网流量数据来有效培训发电机。我们通过利用针对序列的核心生成性对抗性学习算法以及物联网设备常见的特定领域知识来应对这一挑战。通过对18个物联网设备的广泛实验评估，我们证明了所提出的合成物联网流量生成工具的性能显著优于最新的序列和数据包生成器，即使对于自适应攻击者也是如此。



## **30. Attack-SAM: Towards Evaluating Adversarial Robustness of Segment Anything Model**

攻击-SAM：评估分段Anything模型的对抗健壮性 cs.CV

The first work to evaluate the adversarial robustness of Segment  Anything Model (ongoing)

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2305.00866v1) [paper-pdf](http://arxiv.org/pdf/2305.00866v1)

**Authors**: Chenshuang Zhang, Chaoning Zhang, Taegoo Kang, Donghun Kim, Sung-Ho Bae, In So Kweon

**Abstract**: Segment Anything Model (SAM) has attracted significant attention recently, due to its impressive performance on various downstream tasks in a zero-short manner. Computer vision (CV) area might follow the natural language processing (NLP) area to embark on a path from task-specific vision models toward foundation models. However, previous task-specific models are widely recognized as vulnerable to adversarial examples, which fool the model to make wrong predictions with imperceptible perturbation. Such vulnerability to adversarial attacks causes serious concerns when applying deep models to security-sensitive applications. Therefore, it is critical to know whether the vision foundation model SAM can also be easily fooled by adversarial attacks. To the best of our knowledge, our work is the first of its kind to conduct a comprehensive investigation on how to attack SAM with adversarial examples. Specifically, we find that SAM is vulnerable to white-box attacks while maintaining robustness to some extent in the black-box setting. This is an ongoing project and more results and findings will be updated soon through https://github.com/chenshuang-zhang/attack-sam.

摘要: 分段任意模型(SAM)最近受到了极大的关注，因为它在各种下游任务上以零-短的方式表现出令人印象深刻的性能。计算机视觉(CV)领域可能会跟随自然语言处理(NLP)领域，走上一条从特定于任务的视觉模型到基础模型的道路。然而，以前的特定于任务的模型被广泛认为容易受到对抗性例子的影响，这些例子愚弄了模型，使其在不知不觉中做出了错误的预测。在将深度模型应用于安全敏感应用程序时，此类易受敌意攻击的漏洞会引起严重关注。因此，了解VISION基础模型SAM是否也容易被对手攻击愚弄是至关重要的。据我们所知，我们的工作是第一次对如何用对抗性例子攻击SAM进行全面调查。具体地说，我们发现SAM在黑盒环境下很容易受到白盒攻击，同时在一定程度上保持了健壮性。这是一个正在进行的项目，更多的结果和发现将很快通过https://github.com/chenshuang-zhang/attack-sam.更新



## **31. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

ICASSP 2023

**SubmitDate**: 2023-05-01    [abs](http://arxiv.org/abs/2210.06284v4) [paper-pdf](http://arxiv.org/pdf/2210.06284v4)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **32. Robustness of Graph Neural Networks at Scale**

图神经网络的尺度稳健性 cs.LG

39 pages, 22 figures, 17 tables NeurIPS 2021

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2110.14038v4) [paper-pdf](http://arxiv.org/pdf/2110.14038v4)

**Authors**: Simon Geisler, Tobias Schmidt, Hakan Şirin, Daniel Zügner, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Graph Neural Networks (GNNs) are increasingly important given their popularity and the diversity of applications. Yet, existing studies of their vulnerability to adversarial attacks rely on relatively small graphs. We address this gap and study how to attack and defend GNNs at scale. We propose two sparsity-aware first-order optimization attacks that maintain an efficient representation despite optimizing over a number of parameters which is quadratic in the number of nodes. We show that common surrogate losses are not well-suited for global attacks on GNNs. Our alternatives can double the attack strength. Moreover, to improve GNNs' reliability we design a robust aggregation function, Soft Median, resulting in an effective defense at all scales. We evaluate our attacks and defense with standard GNNs on graphs more than 100 times larger compared to previous work. We even scale one order of magnitude further by extending our techniques to a scalable GNN.

摘要: 图神经网络(GNN)因其普及性和应用的多样性而变得越来越重要。然而，现有的关于它们对对手攻击的脆弱性的研究依赖于相对较小的图表。我们解决了这一差距，并研究了如何大规模攻击和防御GNN。我们提出了两种稀疏性感知的一阶优化攻击，它们在对节点数目为二次的多个参数进行优化的情况下仍能保持有效的表示。我们证明了常见的代理损失并不适用于针对GNN的全球攻击。我们的替代品可以使攻击强度加倍。此外，为了提高GNN的可靠性，我们设计了一个稳健的聚集函数--软中值，从而在所有尺度上都能得到有效的防御。与以前的工作相比，我们在大于100倍的图上使用标准GNN来评估我们的攻击和防御。我们甚至通过将我们的技术扩展到可扩展的GNN来进一步扩展一个数量级。



## **33. Assessing Vulnerabilities of Adversarial Learning Algorithm through Poisoning Attacks**

利用中毒攻击评估对抗性学习算法的脆弱性 cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00399v1) [paper-pdf](http://arxiv.org/pdf/2305.00399v1)

**Authors**: Jingfeng Zhang, Bo Song, Bo Han, Lei Liu, Gang Niu, Masashi Sugiyama

**Abstract**: Adversarial training (AT) is a robust learning algorithm that can defend against adversarial attacks in the inference phase and mitigate the side effects of corrupted data in the training phase. As such, it has become an indispensable component of many artificial intelligence (AI) systems. However, in high-stake AI applications, it is crucial to understand AT's vulnerabilities to ensure reliable deployment. In this paper, we investigate AT's susceptibility to poisoning attacks, a type of malicious attack that manipulates training data to compromise the performance of the trained model. Previous work has focused on poisoning attacks against standard training, but little research has been done on their effectiveness against AT. To fill this gap, we design and test effective poisoning attacks against AT. Specifically, we investigate and design clean-label poisoning attacks, allowing attackers to imperceptibly modify a small fraction of training data to control the algorithm's behavior on a specific target data point. Additionally, we propose the clean-label untargeted attack, enabling attackers can attach tiny stickers on training data to degrade the algorithm's performance on all test data, where the stickers could serve as a signal against unauthorized data collection. Our experiments demonstrate that AT can still be poisoned, highlighting the need for caution when using vanilla AT algorithms in security-related applications. The code is at https://github.com/zjfheart/Poison-adv-training.git.

摘要: 对抗性训练(AT)是一种稳健的学习算法，它能在推理阶段抵抗敌意攻击，在训练阶段减轻数据损坏的副作用。因此，它已经成为许多人工智能(AI)系统不可或缺的组成部分。然而，在高风险的人工智能应用中，了解AT的漏洞以确保可靠的部署至关重要。在本文中，我们调查了AT对中毒攻击的敏感性，这是一种操纵训练数据以损害训练模型性能的恶意攻击。以前的工作主要集中在针对标准训练的投毒攻击，但关于它们对抗AT的有效性的研究很少。为了填补这一空白，我们设计并测试了针对AT的有效中毒攻击。具体地说，我们调查和设计干净标签中毒攻击，允许攻击者在不知不觉中修改一小部分训练数据，以控制算法在特定目标数据点上的行为。此外，我们提出了干净标签无目标攻击，使攻击者能够在训练数据上贴上微小的贴纸，以降低算法在所有测试数据上的性能，其中贴纸可以作为反对未经授权的数据收集的信号。我们的实验证明AT仍然可能中毒，这突出了在与安全相关的应用程序中使用普通AT算法时需要谨慎的必要性。代码在https://github.com/zjfheart/Poison-adv-training.git.



## **34. Enhancing Adversarial Contrastive Learning via Adversarial Invariant Regularization**

对抗性不变正则化增强对抗性对比学习 cs.LG

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00374v1) [paper-pdf](http://arxiv.org/pdf/2305.00374v1)

**Authors**: Xilie Xu, Jingfeng Zhang, Feng Liu, Masashi Sugiyama, Mohan Kankanhalli

**Abstract**: Adversarial contrastive learning (ACL), without requiring labels, incorporates adversarial data with standard contrastive learning (SCL) and outputs a robust representation which is generalizable and resistant to adversarial attacks and common corruptions. The style-independence property of representations has been validated to be beneficial in improving robustness transferability. Standard invariant regularization (SIR) has been proposed to make the learned representations via SCL to be independent of the style factors. However, how to equip robust representations learned via ACL with the style-independence property is still unclear so far. To this end, we leverage the technique of causal reasoning to propose an adversarial invariant regularization (AIR) that enforces robust representations learned via ACL to be style-independent. Then, we enhance ACL using invariant regularization (IR), which is a weighted sum of SIR and AIR. Theoretically, we show that AIR implicitly encourages the prediction of adversarial data and consistency between adversarial and natural data to be independent of data augmentations. We also theoretically demonstrate that the style-independence property of robust representation learned via ACL still holds in downstream tasks, providing generalization guarantees. Empirically, our comprehensive experimental results corroborate that IR can significantly improve the performance of ACL and its variants on various datasets.

摘要: 对抗性对比学习(ACL)不需要标签，将对抗性数据与标准对比学习(SCL)相结合，输出具有泛化能力和抵抗对抗性攻击和常见腐败的稳健表示。事实证明，表示的风格无关性对于提高健壮性和可转移性是有益的。标准不变量正则化(SIR)被提出，以使通过SCL学习的表示独立于风格因素。然而，到目前为止，如何用样式无关的属性来装备通过ACL学习的健壮表示仍然不清楚。为此，我们利用因果推理技术提出了一种对抗不变正则化(AIR)，它强制通过ACL学习的稳健表示独立于样式。然后，我们使用不变正则化(IR)来增强ACL，IR是SIR和AIR的加权和。理论上，我们证明了AIR隐含地鼓励对抗性数据的预测以及对抗性数据和自然数据之间的一致性独立于数据扩充。我们还从理论上证明了通过ACL学习的健壮表示的风格无关性在下游任务中仍然成立，提供了泛化保证。实验结果表明，IR能够显著提高ACL及其变体在不同数据集上的性能。



## **35. MetaShard: A Novel Sharding Blockchain Platform for Metaverse Applications**

MetaShard：一种适用于Metverse应用的新型分片区块链平台 cs.CR

**SubmitDate**: 2023-04-30    [abs](http://arxiv.org/abs/2305.00367v1) [paper-pdf](http://arxiv.org/pdf/2305.00367v1)

**Authors**: Cong T. Nguyen, Dinh Thai Hoang, Diep N. Nguyen, Yong Xiao, Dusit Niyato, Eryk Dutkiewicz

**Abstract**: Due to its security, transparency, and flexibility in verifying virtual assets, blockchain has been identified as one of the key technologies for Metaverse. Unfortunately, blockchain-based Metaverse faces serious challenges such as massive resource demands, scalability, and security concerns. To address these issues, this paper proposes a novel sharding-based blockchain framework, namely MetaShard, for Metaverse applications. Particularly, we first develop an effective consensus mechanism, namely Proof-of-Engagement, that can incentivize MUs' data and computing resource contribution. Moreover, to improve the scalability of MetaShard, we propose an innovative sharding management scheme to maximize the network's throughput while protecting the shards from 51% attacks. Since the optimization problem is NP-complete, we develop a hybrid approach that decomposes the problem (using the binary search method) into sub-problems that can be solved effectively by the Lagrangian method. As a result, the proposed approach can obtain solutions in polynomial time, thereby enabling flexible shard reconfiguration and reducing the risk of corruption from the adversary. Extensive numerical experiments show that, compared to the state-of-the-art commercial solvers, our proposed approach can achieve up to 66.6% higher throughput in less than 1/30 running time. Moreover, the proposed approach can achieve global optimal solutions in most experiments.

摘要: 区块链因其在验证虚拟资产方面的安全性、透明度和灵活性，已被确定为Metverse的关键技术之一。不幸的是，基于区块链的Metverse面临着巨大的资源需求、可扩展性和安全问题等严峻挑战。针对这些问题，本文提出了一种新的基于分片的区块链框架MetaShard。特别是，我们首先开发了一个有效的共识机制，即参与度证明，可以激励MU的数据和计算资源贡献。此外，为了提高MetaShard的可扩展性，我们提出了一种创新的分片管理方案，在最大化网络吞吐量的同时保护分片免受51%的攻击。由于优化问题是NP完全的，我们提出了一种混合方法，将问题分解成可以用拉格朗日方法有效求解的子问题。因此，所提出的方法可以在多项式时间内获得解，从而实现灵活的分片重新配置，并降低来自对手的破坏风险。大量的数值实验表明，与最先进的商业求解器相比，我们提出的方法可以在不到1/30的运行时间内获得高达66.6%的吞吐量。此外，该方法在大多数实验中都能获得全局最优解。



## **36. FedGrad: Mitigating Backdoor Attacks in Federated Learning Through Local Ultimate Gradients Inspection**

FedGrad：通过局部最终梯度检测缓解联合学习中的后门攻击 cs.CV

Accepted for presentation at the International Joint Conference on  Neural Networks (IJCNN 2023)

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2305.00328v1) [paper-pdf](http://arxiv.org/pdf/2305.00328v1)

**Authors**: Thuy Dung Nguyen, Anh Duy Nguyen, Kok-Seng Wong, Huy Hieu Pham, Thanh Hung Nguyen, Phi Le Nguyen, Truong Thao Nguyen

**Abstract**: Federated learning (FL) enables multiple clients to train a model without compromising sensitive data. The decentralized nature of FL makes it susceptible to adversarial attacks, especially backdoor insertion during training. Recently, the edge-case backdoor attack employing the tail of the data distribution has been proposed as a powerful one, raising questions about the shortfall in current defenses' robustness guarantees. Specifically, most existing defenses cannot eliminate edge-case backdoor attacks or suffer from a trade-off between backdoor-defending effectiveness and overall performance on the primary task. To tackle this challenge, we propose FedGrad, a novel backdoor-resistant defense for FL that is resistant to cutting-edge backdoor attacks, including the edge-case attack, and performs effectively under heterogeneous client data and a large number of compromised clients. FedGrad is designed as a two-layer filtering mechanism that thoroughly analyzes the ultimate layer's gradient to identify suspicious local updates and remove them from the aggregation process. We evaluate FedGrad under different attack scenarios and show that it significantly outperforms state-of-the-art defense mechanisms. Notably, FedGrad can almost 100% correctly detect the malicious participants, thus providing a significant reduction in the backdoor effect (e.g., backdoor accuracy is less than 8%) while not reducing the main accuracy on the primary task.

摘要: 联合学习(FL)使多个客户能够在不损害敏感数据的情况下训练模型。FL的分散性使其容易受到对手的攻击，特别是训练期间的后门插入。最近，利用数据分布尾部的边缘情况后门攻击被提出为一种强有力的攻击，这引发了人们对当前防御系统健壮性保证不足的质疑。具体地说，大多数现有的防御系统无法消除边缘情况下的后门攻击，或者在后门防御效率和主要任务的整体性能之间进行权衡。为了应对这一挑战，我们提出了FedGrad，这是一种针对FL的新型后门防御机制，它能够抵抗包括Edge-Case攻击在内的尖端后门攻击，并在异类客户端数据和大量受攻击的客户端下有效执行。FedGrad被设计为一种两层过滤机制，它彻底分析最终层的梯度，以识别可疑的本地更新并将它们从聚合过程中删除。我们在不同的攻击场景下对FedGrad进行了评估，结果表明它的性能明显优于最先进的防御机制。值得注意的是，FedGrad几乎可以100%正确地检测恶意参与者，从而显著降低后门效应(例如，后门准确率低于8%)，同时不降低主要任务的主要准确率。



## **37. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

组合对抗性机器学习的博弈论混合专家 cs.LG

17pages, 10 figures

**SubmitDate**: 2023-04-29    [abs](http://arxiv.org/abs/2211.14669v2) [paper-pdf](http://arxiv.org/pdf/2211.14669v2)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically customized to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). We first conduct a transferability analysis, to demonstrate the adversarial examples generated by customized attacks on one defense, are not often misclassified by another defense.   This finding leads to two important questions. First, how can the low transferability between defenses be utilized in a game theoretic framework to improve the robustness? Second, how can an adversary within this framework develop effective multi-model attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses. Our framework is called Game theoretic Mixed Experts (GaME). It is designed to find the Mixed-Nash strategy for both a detector based and standard defender, when facing an attacker employing compositional adversarial attacks. We further propose three new attack algorithms, specifically designed to target defenses with randomized transformations, multi-model voting schemes, and adversarial detector architectures. These attacks serve to both strengthen defenses generated by the GaME framework and verify their robustness against unforeseen attacks. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.

摘要: 对抗性机器学习的最新进展表明，被认为是健壮的防御实际上容易受到针对其弱点而专门定制的对抗性攻击。这些防御包括随机变换弹幕(BART)、友好对手训练(FAT)、垃圾就是宝藏(TIT)以及由视觉变形金刚(VITS)、大转移模型和尖峰神经网络(SNN)组成的集成模型。我们首先进行可转移性分析，以证明定制攻击对一个防御系统生成的对抗性示例不会经常被另一个防御系统错误分类。这一发现引出了两个重要问题。首先，如何在博弈论框架中利用防守之间的低可转换性来提高健壮性？第二，在这个框架内的对手如何开发有效的多模式攻击？在这篇文章中，我们为集成对抗性攻击和防御提供了一个博弈论框架。我们的框架称为博弈论混合专家(GAME)。它的设计是为了在面对使用成分对抗攻击的攻击者时，为基于检测器的和标准的防守者找到混合纳什策略。我们进一步提出了三种新的攻击算法，分别针对随机变换、多模型投票方案和对抗性检测器体系结构的目标防御而设计。这些攻击既加强了游戏框架产生的防御，又验证了它们对不可预见的攻击的健壮性。总体而言，我们的框架和分析通过对组合攻击和防御公式产生新的见解，促进了对抗性机器学习领域的发展。



## **38. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

提高多重攻击下的高光谱对抗健壮性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2210.16346v3) [paper-pdf](http://arxiv.org/pdf/2210.16346v3)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.

摘要: 对高光谱图像进行分类的语义分割模型容易受到敌意例子的影响。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法的性能会下降。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。



## **39. The Power of Typed Affine Decision Structures: A Case Study**

类型化仿射决策结构的威力：案例研究 cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14888v1) [paper-pdf](http://arxiv.org/pdf/2304.14888v1)

**Authors**: Gerrit Nolte, Maximilian Schlüter, Alnis Murtovi, Bernhard Steffen

**Abstract**: TADS are a novel, concise white-box representation of neural networks. In this paper, we apply TADS to the problem of neural network verification, using them to generate either proofs or concise error characterizations for desirable neural network properties. In a case study, we consider the robustness of neural networks to adversarial attacks, i.e., small changes to an input that drastically change a neural networks perception, and show that TADS can be used to provide precise diagnostics on how and where robustness errors a occur. We achieve these results by introducing Precondition Projection, a technique that yields a TADS describing network behavior precisely on a given subset of its input space, and combining it with PCA, a traditional, well-understood dimensionality reduction technique. We show that PCA is easily compatible with TADS. All analyses can be implemented in a straightforward fashion using the rich algebraic properties of TADS, demonstrating the utility of the TADS framework for neural network explainability and verification. While TADS do not yet scale as efficiently as state-of-the-art neural network verifiers, we show that, using PCA-based simplifications, they can still scale to mediumsized problems and yield concise explanations for potential errors that can be used for other purposes such as debugging a network or generating new training samples.

摘要: TADS是神经网络的一种新颖、简洁的白盒表示。在本文中，我们将TADS应用于神经网络验证问题，使用它们来生成期望的神经网络性质的证明或简洁的误差特征。在一个案例研究中，我们考虑了神经网络对对抗性攻击的稳健性，即输入的微小变化极大地改变了神经网络的感知，并表明TADS可以用来提供关于健壮性错误如何以及在哪里发生的精确诊断。我们通过引入预条件投影来获得这些结果，这是一种在输入空间的给定子集上产生精确描述网络行为的TADS的技术，并将其与传统的、众所周知的降维技术PCA相结合。实验结果表明，主元分析算法与TADS算法具有很好的兼容性。所有的分析都可以使用TADS丰富的代数特性以一种简单的方式实现，展示了TADS框架在神经网络可解释性和验证方面的实用性。虽然TADS还没有像最先进的神经网络验证器那样有效地进行扩展，但我们表明，使用基于PCA的简化，它们仍然可以扩展到中等规模的问题，并为潜在错误提供简明的解释，这些解释可以用于其他目的，如调试网络或生成新的训练样本。



## **40. Topic-oriented Adversarial Attacks against Black-box Neural Ranking Models**

针对黑盒神经网络排序模型的主题对抗性攻击 cs.IR

Accepted by SIGIR 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14867v1) [paper-pdf](http://arxiv.org/pdf/2304.14867v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Wei Chen, Yixing Fan, Xueqi Cheng

**Abstract**: Neural ranking models (NRMs) have attracted considerable attention in information retrieval. Unfortunately, NRMs may inherit the adversarial vulnerabilities of general neural networks, which might be leveraged by black-hat search engine optimization practitioners. Recently, adversarial attacks against NRMs have been explored in the paired attack setting, generating an adversarial perturbation to a target document for a specific query. In this paper, we focus on a more general type of perturbation and introduce the topic-oriented adversarial ranking attack task against NRMs, which aims to find an imperceptible perturbation that can promote a target document in ranking for a group of queries with the same topic. We define both static and dynamic settings for the task and focus on decision-based black-box attacks. We propose a novel framework to improve topic-oriented attack performance based on a surrogate ranking model. The attack problem is formalized as a Markov decision process (MDP) and addressed using reinforcement learning. Specifically, a topic-oriented reward function guides the policy to find a successful adversarial example that can be promoted in rankings to as many queries as possible in a group. Experimental results demonstrate that the proposed framework can significantly outperform existing attack strategies, and we conclude by re-iterating that there exist potential risks for applying NRMs in the real world.

摘要: 神经排序模型(NRM)在信息检索领域引起了广泛的关注。不幸的是，NRM可能会继承一般神经网络的对抗性漏洞，这可能会被黑帽搜索引擎优化从业者利用。最近，针对NRM的对抗性攻击已经在配对攻击设置中被探索，为特定查询生成对目标文档的对抗性扰动。在本文中，我们着眼于一种更一般的扰动类型，并引入了针对NRMS的面向主题的对抗性排名攻击任务，其目的是找到一种可以促进目标文档对同一主题的一组查询进行排名的不可察觉的扰动。我们定义了任务的静态和动态设置，并专注于基于决策的黑盒攻击。提出了一种新的基于代理排名模型的面向主题攻击性能改进框架。攻击问题被形式化化为马尔可夫决策过程(MDP)，并使用强化学习来解决。具体地说，面向主题的奖励功能引导策略找到一个成功的对抗性例子，该例子可以在排名中提升到一个组中尽可能多的查询。实验结果表明，该框架的性能明显优于已有的攻击策略，并通过反复验证得出结论：在现实世界中应用NRM存在潜在的风险。



## **41. False Claims against Model Ownership Resolution**

针对所有权解决方案范本的虚假索赔 cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.06607v2) [paper-pdf](http://arxiv.org/pdf/2304.06607v2)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的经验评估，我们证明了我们的虚假声明攻击在所有具有现实配置的著名MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **42. Certified Robustness of Quantum Classifiers against Adversarial Examples through Quantum Noise**

通过量子噪声验证量子分类器对敌意例子的鲁棒性 quant-ph

Accepted to IEEE ICASSP 2023

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2211.00887v2) [paper-pdf](http://arxiv.org/pdf/2211.00887v2)

**Authors**: Jhih-Cing Huang, Yu-Lin Tsai, Chao-Han Huck Yang, Cheng-Fang Su, Chia-Mu Yu, Pin-Yu Chen, Sy-Yen Kuo

**Abstract**: Recently, quantum classifiers have been found to be vulnerable to adversarial attacks, in which quantum classifiers are deceived by imperceptible noises, leading to misclassification. In this paper, we propose the first theoretical study demonstrating that adding quantum random rotation noise can improve robustness in quantum classifiers against adversarial attacks. We link the definition of differential privacy and show that the quantum classifier trained with the natural presence of additive noise is differentially private. Finally, we derive a certified robustness bound to enable quantum classifiers to defend against adversarial examples, supported by experimental results simulated with noises from IBM's 7-qubits device.

摘要: 最近，量子分类器被发现容易受到敌意攻击，其中量子分类器被不可感知的噪声欺骗，导致错误分类。在本文中，我们提出了第一个理论研究，证明了加入量子随机旋转噪声可以提高量子分类器对敌意攻击的稳健性。我们将差分隐私的定义联系起来，证明了在自然存在加性噪声的情况下训练的量子分类器是差分隐私的。最后，我们得到了一个证明的稳健性界限，使量子分类器能够防御敌对的例子，支持用来自IBM的7量子比特设备的噪声模拟的实验结果。



## **43. Fusion is Not Enough: Single-Modal Attacks to Compromise Fusion Models in Autonomous Driving**

融合是不够的：自动驾驶中破坏融合模型的单模式攻击 cs.CV

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2304.14614v1) [paper-pdf](http://arxiv.org/pdf/2304.14614v1)

**Authors**: Zhiyuan Cheng, Hongjun Choi, James Liang, Shiwei Feng, Guanhong Tao, Dongfang Liu, Michael Zuzak, Xiangyu Zhang

**Abstract**: Multi-sensor fusion (MSF) is widely adopted for perception in autonomous vehicles (AVs), particularly for the task of 3D object detection with camera and LiDAR sensors. The rationale behind fusion is to capitalize on the strengths of each modality while mitigating their limitations. The exceptional and leading performance of fusion models has been demonstrated by advanced deep neural network (DNN)-based fusion techniques. Fusion models are also perceived as more robust to attacks compared to single-modal ones due to the redundant information in multiple modalities. In this work, we challenge this perspective with single-modal attacks that targets the camera modality, which is considered less significant in fusion but more affordable for attackers. We argue that the weakest link of fusion models depends on their most vulnerable modality, and propose an attack framework that targets advanced camera-LiDAR fusion models with adversarial patches. Our approach employs a two-stage optimization-based strategy that first comprehensively assesses vulnerable image areas under adversarial attacks, and then applies customized attack strategies to different fusion models, generating deployable patches. Evaluations with five state-of-the-art camera-LiDAR fusion models on a real-world dataset show that our attacks successfully compromise all models. Our approach can either reduce the mean average precision (mAP) of detection performance from 0.824 to 0.353 or degrade the detection score of the target object from 0.727 to 0.151 on average, demonstrating the effectiveness and practicality of our proposed attack framework.

摘要: 多传感器融合(MSF)被广泛地应用于自主车辆的感知，特别是在具有摄像机和激光雷达传感器的三维目标检测任务中。融合背后的基本原理是利用每种方式的优势，同时减轻它们的局限性。先进的基于深度神经网络(DNN)的融合技术证明了融合模型的卓越和领先的性能。由于多模式中的冗余信息，融合模型也被认为比单模式模型更具稳健性。在这项工作中，我们通过以相机通道为目标的单模式攻击来挑战这一观点，相机通道在融合中被认为不那么重要，但对于攻击者来说更负担得起。我们认为融合模型的最薄弱环节取决于它们最脆弱的通道，并提出了一种针对带有对抗性补丁的高级Camera-LiDAR融合模型的攻击框架。该方法采用两阶段优化策略，首先综合评估图像在敌方攻击下的易受攻击区域，然后将定制的攻击策略应用于不同的融合模型，生成可部署的补丁。在真实数据集上对五种最先进的相机-LiDAR融合模型进行的评估表明，我们的攻击成功地折衷了所有模型。我们的方法可以将检测性能的平均精度(MAP)从0.824降低到0.353，或者将目标对象的检测得分从平均0.727降低到0.151，从而证明了我们所提出的攻击框架的有效性和实用性。



## **44. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

基于在线深度强化学习的高效奖赏中毒攻击 cs.LG

**SubmitDate**: 2023-04-28    [abs](http://arxiv.org/abs/2205.14842v2) [paper-pdf](http://arxiv.org/pdf/2205.14842v2)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstract**: We study reward poisoning attacks on online deep reinforcement learning (DRL), where the attacker is oblivious to the learning algorithm used by the agent and the dynamics of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general, black-box reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct two new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. We provide a theoretical analysis of the efficiency of our attack and perform an extensive empirical evaluation. Our results show that our attacks efficiently poison agents learning in several popular classical control and MuJoCo environments with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc.

摘要: 研究了在线深度强化学习(DRL)中的奖赏中毒攻击，攻击者对智能体使用的学习算法和环境的动态特性视而不见。我们通过设计一个称为对抗性MDP攻击的通用黑盒奖励中毒框架来展示最新的DRL算法的内在脆弱性。我们实例化了我们的框架，构造了两个新的攻击，这两个攻击只破坏了总训练时间步骤的一小部分奖励，并使代理学习一个低性能的策略。我们对我们的攻击效率进行了理论分析，并进行了广泛的经验评估。我们的结果表明，我们的攻击有效地毒化了在几个流行的经典控制和MuJoCo环境中学习的代理，并使用了各种先进的DRL算法，如DQN，PPO，SAC等。



## **45. Adversary Aware Continual Learning**

对手意识到的持续学习 cs.LG

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14483v1) [paper-pdf](http://arxiv.org/pdf/2304.14483v1)

**Authors**: Muhammad Umer, Robi Polikar

**Abstract**: Class incremental learning approaches are useful as they help the model to learn new information (classes) sequentially, while also retaining the previously acquired information (classes). However, it has been shown that such approaches are extremely vulnerable to the adversarial backdoor attacks, where an intelligent adversary can introduce small amount of misinformation to the model in the form of imperceptible backdoor pattern during training to cause deliberate forgetting of a specific task or class at test time. In this work, we propose a novel defensive framework to counter such an insidious attack where, we use the attacker's primary strength-hiding the backdoor pattern by making it imperceptible to humans-against it, and propose to learn a perceptible (stronger) pattern (also during the training) that can overpower the attacker's imperceptible (weaker) pattern. We demonstrate the effectiveness of the proposed defensive mechanism through various commonly used Replay-based (both generative and exact replay-based) class incremental learning algorithms using continual learning benchmark variants of CIFAR-10, CIFAR-100, and MNIST datasets. Most noteworthy, our proposed defensive framework does not assume that the attacker's target task and target class is known to the defender. The defender is also unaware of the shape, size, and location of the attacker's pattern. We show that our proposed defensive framework considerably improves the performance of class incremental learning algorithms with no knowledge of the attacker's target task, attacker's target class, and attacker's imperceptible pattern. We term our defensive framework as Adversary Aware Continual Learning (AACL).

摘要: 类增量学习方法是有用的，因为它们帮助模型顺序地学习新的信息(类)，同时还保留了以前获得的信息(类)。然而，已有研究表明，这种方法极易受到对抗性后门攻击，在这种攻击中，聪明的对手可以在训练期间以不可察觉的后门模式的形式向模型引入少量错误信息，从而导致在测试时故意忘记特定的任务或类。在这项工作中，我们提出了一个新的防御框架来对抗这样的潜伏攻击，其中我们使用攻击者的主要优势-通过使其对人类不可察觉来隐藏后门模式-来对抗它，并建议学习一种可感知(更强)的模式(也是在训练期间)，该模式可以压倒攻击者不可察觉(更弱)的模式。我们使用CIFAR-10、CIFAR-100和MNIST数据集的连续学习基准变量，通过各种常用的基于重放(包括生成和精确重放)的类增量学习算法，验证了所提出的防御机制的有效性。最值得注意的是，我们提出的防御框架并不假设攻击者的目标任务和目标类对防御者是已知的。防御者也不知道攻击者图案的形状、大小和位置。我们的结果表明，在不知道攻击者的目标任务、攻击者的目标类和攻击者的不可察觉的模式的情况下，我们提出的防御框架显著提高了类增量学习算法的性能。我们将我们的防御框架称为对手感知持续学习(AACL)。



## **46. Attacking Fake News Detectors via Manipulating News Social Engagement**

通过操纵新闻社会参与打击假新闻检测器 cs.SI

ACM Web Conference 2023 (WWW'23)

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2302.07363v3) [paper-pdf](http://arxiv.org/pdf/2302.07363v3)

**Authors**: Haoran Wang, Yingtong Dou, Canyu Chen, Lichao Sun, Philip S. Yu, Kai Shu

**Abstract**: Social media is one of the main sources for news consumption, especially among the younger generation. With the increasing popularity of news consumption on various social media platforms, there has been a surge of misinformation which includes false information or unfounded claims. As various text- and social context-based fake news detectors are proposed to detect misinformation on social media, recent works start to focus on the vulnerabilities of fake news detectors. In this paper, we present the first adversarial attack framework against Graph Neural Network (GNN)-based fake news detectors to probe their robustness. Specifically, we leverage a multi-agent reinforcement learning (MARL) framework to simulate the adversarial behavior of fraudsters on social media. Research has shown that in real-world settings, fraudsters coordinate with each other to share different news in order to evade the detection of fake news detectors. Therefore, we modeled our MARL framework as a Markov Game with bot, cyborg, and crowd worker agents, which have their own distinctive cost, budget, and influence. We then use deep Q-learning to search for the optimal policy that maximizes the rewards. Extensive experimental results on two real-world fake news propagation datasets demonstrate that our proposed framework can effectively sabotage the GNN-based fake news detector performance. We hope this paper can provide insights for future research on fake news detection.

摘要: 社交媒体是新闻消费的主要来源之一，尤其是在年轻一代中。随着新闻消费在各种社交媒体平台上的日益流行，错误信息激增，其中包括虚假信息或毫无根据的说法。随着各种基于文本和社会语境的假新闻检测器被提出来检测社交媒体上的错误信息，最近的研究开始关注假新闻检测器的脆弱性。本文提出了第一个针对基于图神经网络(GNN)的假新闻检测器的对抗性攻击框架，以探讨其健壮性。具体地说，我们利用多智能体强化学习(MAIL)框架来模拟社交媒体上欺诈者的对抗行为。研究表明，在现实世界中，欺诈者相互协调，分享不同的新闻，以躲避假新闻检测器的检测。因此，我们将我们的Marl框架建模为一个包含BOT、半机械人和群工代理的马尔可夫博弈，这些代理都有自己独特的成本、预算和影响。然后，我们使用深度Q-学习来搜索最大化回报的最优策略。在两个真实假新闻传播数据集上的大量实验结果表明，我们提出的框架可以有效地破坏基于GNN的假新闻检测器的性能。希望本文能为今后的假新闻检测研究提供一些启示。



## **47. On the (In)security of Peer-to-Peer Decentralized Machine Learning**

点对点分散机器学习的安全性研究 cs.CR

IEEE S&P'23 (Previous title: "On the Privacy of Decentralized Machine  Learning")

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.08443v2) [paper-pdf](http://arxiv.org/pdf/2205.08443v2)

**Authors**: Dario Pasquini, Mathilde Raynal, Carmela Troncoso

**Abstract**: In this work, we carry out the first, in-depth, privacy analysis of Decentralized Learning -- a collaborative machine learning framework aimed at addressing the main limitations of federated learning. We introduce a suite of novel attacks for both passive and active decentralized adversaries. We demonstrate that, contrary to what is claimed by decentralized learning proposers, decentralized learning does not offer any security advantage over federated learning. Rather, it increases the attack surface enabling any user in the system to perform privacy attacks such as gradient inversion, and even gain full control over honest users' local model. We also show that, given the state of the art in protections, privacy-preserving configurations of decentralized learning require fully connected networks, losing any practical advantage over the federated setup and therefore completely defeating the objective of the decentralized approach.

摘要: 在这项工作中，我们进行了第一次，深入的，隐私分析的分散学习--一个合作的机器学习框架，旨在解决联合学习的主要限制。我们针对被动的和主动的分散攻击引入了一套新的攻击。我们证明，与去中心化学习提出者所声称的相反，去中心化学习并不比联邦学习提供任何安全优势。相反，它增加了攻击面，使系统中的任何用户都可以执行诸如梯度反转等隐私攻击，甚至获得对诚实用户的本地模型的完全控制。我们还表明，考虑到保护技术的最新水平，去中心化学习的隐私保护配置需要完全连接的网络，失去了与联邦设置相比的任何实际优势，因此完全违背了去中心化方法的目标。



## **48. Robust Resilient Signal Reconstruction under Adversarial Attacks**

对抗性攻击下的稳健恢复信号重构 math.OC

7 pages

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/1807.08004v2) [paper-pdf](http://arxiv.org/pdf/1807.08004v2)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Lalit Mestha, Hema Achanta

**Abstract**: We consider the problem of signal reconstruction for a system under sparse signal corruption by a malicious agent. The reconstruction problem follows the standard error coding problem that has been studied extensively in the literature. We include a new challenge of robust estimation of the attack support. The problem is then cast as a constrained optimization problem merging promising techniques in the area of deep learning and estimation theory. A pruning algorithm is developed to reduce the ``false positive" uncertainty of data-driven attack localization results, thereby improving the probability of correct signal reconstruction. Sufficient conditions for the correct reconstruction and the associated reconstruction error bounds are obtained for both exact and inexact attack support estimation. Moreover, a simulation of a water distribution system is presented to validate the proposed techniques.

摘要: 我们考虑了在稀疏信号被恶意代理破坏的情况下系统的信号重构问题。重建问题遵循在文献中已被广泛研究的标准误差编码问题。我们包括了一个新的挑战，即稳健地估计攻击支持。然后将该问题归结为一个约束优化问题，融合了深度学习和估计理论领域中有前途的技术。为了减少数据驱动攻击定位结果的“假阳性”不确定性，从而提高正确重构信号的概率，提出了一种剪枝算法.对于准确和不精确的攻击支持度估计，得到了正确重构的充分条件和相应的重构误差界.此外，通过对供水系统的仿真，验证了所提方法的有效性.



## **49. QEVSEC: Quick Electric Vehicle SEcure Charging via Dynamic Wireless Power Transfer**

QEVSEC：通过动态无线电能传输实现电动汽车快速安全充电 cs.CR

6 pages, conference

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2205.10292v2) [paper-pdf](http://arxiv.org/pdf/2205.10292v2)

**Authors**: Tommaso Bianchi, Surudhi Asokraj, Alessandro Brighente, Mauro Conti, Radha Poovendran

**Abstract**: Dynamic Wireless Power Transfer (DWPT) can be used for on-demand recharging of Electric Vehicles (EV) while driving. However, DWPT raises numerous security and privacy concerns. Recently, researchers demonstrated that DWPT systems are vulnerable to adversarial attacks. In an EV charging scenario, an attacker can prevent the authorized customer from charging, obtain a free charge by billing a victim user and track a target vehicle. State-of-the-art authentication schemes relying on centralized solutions are either vulnerable to various attacks or have high computational complexity, making them unsuitable for a dynamic scenario. In this paper, we propose Quick Electric Vehicle SEcure Charging (QEVSEC), a novel, secure, and efficient authentication protocol for the dynamic charging of EVs. Our idea for QEVSEC originates from multiple vulnerabilities we found in the state-of-the-art protocol that allows tracking of user activity and is susceptible to replay attacks. Based on these observations, the proposed protocol solves these issues and achieves lower computational complexity by using only primitive cryptographic operations in a very short message exchange. QEVSEC provides scalability and a reduced cost in each iteration, thus lowering the impact on the power needed from the grid.

摘要: 动态无线电能传输(DWPT)可用于电动汽车(EV)行驶时的按需充电。然而，DWPT带来了许多安全和隐私方面的问题。最近，研究人员证明了DWPT系统容易受到敌意攻击。在电动汽车充电场景中，攻击者可以阻止授权客户充电，通过向受害用户收费来获得免费费用，并跟踪目标车辆。依赖于集中式解决方案的最先进的身份验证方案要么容易受到各种攻击，要么具有很高的计算复杂性，不适合动态场景。本文提出了一种新颖、安全、高效的电动汽车动态充电认证协议--快速电动汽车安全充电协议。我们对QEVSEC的想法源于我们在最先进的协议中发现的多个漏洞，该协议允许跟踪用户活动，并且容易受到重播攻击。基于这些观察，提出的协议解决了这些问题，并通过在很短的消息交换中仅使用原始密码操作来实现较低的计算复杂度。QEVSEC在每次迭代中提供了可扩展性和更低的成本，从而降低了对电网所需电力的影响。



## **50. Boosting Big Brother: Attacking Search Engines with Encodings**

助推老大哥：用编码攻击搜索引擎 cs.CR

**SubmitDate**: 2023-04-27    [abs](http://arxiv.org/abs/2304.14031v1) [paper-pdf](http://arxiv.org/pdf/2304.14031v1)

**Authors**: Nicholas Boucher, Luca Pajola, Ilia Shumailov, Ross Anderson, Mauro Conti

**Abstract**: Search engines are vulnerable to attacks against indexing and searching via text encoding manipulation. By imperceptibly perturbing text using uncommon encoded representations, adversaries can control results across search engines for specific search queries. We demonstrate that this attack is successful against two major commercial search engines - Google and Bing - and one open source search engine - Elasticsearch. We further demonstrate that this attack is successful against LLM chat search including Bing's GPT-4 chatbot and Google's Bard chatbot. We also present a variant of the attack targeting text summarization and plagiarism detection models, two ML tasks closely tied to search. We provide a set of defenses against these techniques and warn that adversaries can leverage these attacks to launch disinformation campaigns against unsuspecting users, motivating the need for search engine maintainers to patch deployed systems.

摘要: 搜索引擎容易受到针对索引和搜索的攻击，这些攻击是通过文本编码操作进行的。通过使用不常见的编码表示在不知不觉中扰乱文本，攻击者可以控制特定搜索查询的搜索引擎结果。我们展示了对两个主要的商业搜索引擎--Google和Bing--和一个开源搜索引擎--Elasticearch的攻击是成功的。我们进一步证明了该攻击对LLM聊天搜索是成功的，包括Bing的GPT-4聊天机器人和Google的Bard聊天机器人。我们还提出了针对文本摘要和抄袭检测模型的攻击的一个变体，这两个ML任务与搜索密切相关。我们提供了一套针对这些技术的防御措施，并警告说，攻击者可以利用这些攻击对毫无戒心的用户发起虚假信息运动，从而刺激搜索引擎维护人员修补已部署的系统的需求。



