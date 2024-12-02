# Latest Adversarial Attack Papers
**update at 2024-12-02 09:45:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. GSE: Group-wise Sparse and Explainable Adversarial Attacks**

GSE：分组稀疏和可解释的对抗性攻击 cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2311.17434v4) [paper-pdf](http://arxiv.org/pdf/2311.17434v4)

**Authors**: Shpresim Sadiku, Moritz Wagner, Sebastian Pokutta

**Abstract**: Sparse adversarial attacks fool deep neural networks (DNNs) through minimal pixel perturbations, often regularized by the $\ell_0$ norm. Recent efforts have replaced this norm with a structural sparsity regularizer, such as the nuclear group norm, to craft group-wise sparse adversarial attacks. The resulting perturbations are thus explainable and hold significant practical relevance, shedding light on an even greater vulnerability of DNNs. However, crafting such attacks poses an optimization challenge, as it involves computing norms for groups of pixels within a non-convex objective. We address this by presenting a two-phase algorithm that generates group-wise sparse attacks within semantically meaningful areas of an image. Initially, we optimize a quasinorm adversarial loss using the $1/2-$quasinorm proximal operator tailored for non-convex programming. Subsequently, the algorithm transitions to a projected Nesterov's accelerated gradient descent with $2-$norm regularization applied to perturbation magnitudes. Rigorous evaluations on CIFAR-10 and ImageNet datasets demonstrate a remarkable increase in group-wise sparsity, e.g., $50.9\%$ on CIFAR-10 and $38.4\%$ on ImageNet (average case, targeted attack). This performance improvement is accompanied by significantly faster computation times, improved explainability, and a $100\%$ attack success rate.

摘要: 稀疏敌意攻击通过最小的像素扰动来欺骗深度神经网络(DNN)，这种扰动通常由$\ell_0$范数来正则化。最近的努力已经用结构稀疏性正则化规则取代了这一规范，例如核集团规范，以制定群组稀疏对抗性攻击。因此，由此产生的扰动是可以解释的，并具有重要的实际意义，揭示了DNN更大的脆弱性。然而，精心设计这样的攻击构成了一个优化挑战，因为它涉及到计算非凸目标内的像素组的规范。我们通过提出一个两阶段算法来解决这个问题，该算法在图像的语义有意义的区域内生成分组稀疏攻击。首先，我们使用为非凸规划量身定做的$1/2-$拟正态近似算子来优化拟正态对抗性损失。随后，算法过渡到投影的内斯特罗夫加速梯度下降，并对摄动幅度应用$2-$范数正则化。在CIFAR-10和ImageNet数据集上的严格评估表明，组内稀疏性显著增加，例如，CIFAR-10上的稀疏度为50.9美元，ImageNet上的稀疏度为38.4美元(平均案例，有针对性的攻击)。伴随着这种性能改进的是显著更快的计算时间、更好的可解释性以及$100\$攻击成功率。



## **2. Don't Command, Cultivate: An Exploratory Study of System-2 Alignment**

不命令，培养：System-2对齐的探索性研究 cs.CL

Preprint version, more results will be updated

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.17075v3) [paper-pdf](http://arxiv.org/pdf/2411.17075v3)

**Authors**: Yuhang Wang, Jitao Sang

**Abstract**: The o1 system card identifies the o1 models as the most robust within OpenAI, with their defining characteristic being the progression from rapid, intuitive thinking to slower, more deliberate reasoning. This observation motivated us to investigate the influence of System-2 thinking patterns on model safety. In our preliminary research, we conducted safety evaluations of the o1 model, including complex jailbreak attack scenarios using adversarial natural language prompts and mathematical encoding prompts. Our findings indicate that the o1 model demonstrates relatively improved safety performance; however, it still exhibits vulnerabilities, particularly against jailbreak attacks employing mathematical encoding. Through detailed case analysis, we identified specific patterns in the o1 model's responses. We also explored the alignment of System-2 safety in open-source models using prompt engineering and supervised fine-tuning techniques. Experimental results show that some simple methods to encourage the model to carefully scrutinize user requests are beneficial for model safety. Additionally, we proposed a implementation plan for process supervision to enhance safety alignment. The implementation details and experimental results will be provided in future versions.

摘要: O1系统卡将o1模型确定为OpenAI中最健壮的模型，它们的决定性特征是从快速、直观的思考到更慢、更深思熟虑的推理的过程。这一观察结果促使我们调查System-2思维模式对模型安全性的影响。在我们的初步研究中，我们对o1模型进行了安全性评估，包括使用对抗性自然语言提示和数学编码提示的复杂越狱攻击场景。我们的发现表明，o1模型显示出相对更好的安全性能；但是，它仍然存在漏洞，特别是对使用数学编码的越狱攻击。通过详细的案例分析，我们确定了o1模型反应的具体模式。我们还使用即时工程和有监督的微调技术探索了开源模型中System-2安全性的一致性。实验结果表明，一些简单的方法鼓励模型仔细审查用户请求，有利于模型的安全。此外，我们还提出了加强安全对接的过程监管实施方案。实现细节和实验结果将在未来的版本中提供。



## **3. Visual Adversarial Attack on Vision-Language Models for Autonomous Driving**

自动驾驶视觉语言模型的视觉对抗攻击 cs.CV

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18275v1) [paper-pdf](http://arxiv.org/pdf/2411.18275v1)

**Authors**: Tianyuan Zhang, Lu Wang, Xinwei Zhang, Yitong Zhang, Boyi Jia, Siyuan Liang, Shengshan Hu, Qiang Fu, Aishan Liu, Xianglong Liu

**Abstract**: Vision-language models (VLMs) have significantly advanced autonomous driving (AD) by enhancing reasoning capabilities. However, these models remain highly vulnerable to adversarial attacks. While existing research has primarily focused on general VLM attacks, the development of attacks tailored to the safety-critical AD context has been largely overlooked. In this paper, we take the first step toward designing adversarial attacks specifically targeting VLMs in AD, exposing the substantial risks these attacks pose within this critical domain. We identify two unique challenges for effective adversarial attacks on AD VLMs: the variability of textual instructions and the time-series nature of visual scenarios. To this end, we propose ADvLM, the first visual adversarial attack framework specifically designed for VLMs in AD. Our framework introduces Semantic-Invariant Induction, which uses a large language model to create a diverse prompt library of textual instructions with consistent semantic content, guided by semantic entropy. Building on this, we introduce Scenario-Associated Enhancement, an approach where attention mechanisms select key frames and perspectives within driving scenarios to optimize adversarial perturbations that generalize across the entire scenario. Extensive experiments on several AD VLMs over multiple benchmarks show that ADvLM achieves state-of-the-art attack effectiveness. Moreover, real-world attack studies further validate its applicability and potential in practice.

摘要: 视觉语言模型通过增强推理能力极大地促进了自动驾驶(AD)。然而，这些模型仍然非常容易受到对手的攻击。虽然现有的研究主要集中在一般的VLM攻击上，但针对安全关键型AD环境而定制的攻击的发展在很大程度上被忽视了。在本文中，我们向设计专门针对AD中的VLM的对抗性攻击迈出了第一步，暴露了这些攻击在这一关键领域中构成的实质性风险。我们确定了对AD VLMS进行有效的对抗性攻击的两个独特的挑战：文本指令的可变性和视觉场景的时间序列性质。为此，我们提出了ADvLM，这是第一个专门为AD中的VLM设计的可视化对抗性攻击框架。我们的框架引入了语义不变归纳法，它使用一个大型语言模型来创建一个具有一致语义内容的多样化提示库，并以语义熵为指导。在此基础上，我们引入了与场景相关的增强，这是一种注意机制在驾驶场景中选择关键帧和视角以优化整个场景中概括的对抗性扰动的方法。在多个基准上对多个AD VLM进行的大量实验表明，ADvLM达到了最先进的攻击效率。此外，真实世界的攻击研究进一步验证了其在实践中的适用性和潜力。



## **4. G-Designer: Architecting Multi-agent Communication Topologies via Graph Neural Networks**

G-Designer：通过图神经网络构建多智能体通信布局 cs.MA

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2410.11782v2) [paper-pdf](http://arxiv.org/pdf/2410.11782v2)

**Authors**: Guibin Zhang, Yanwei Yue, Xiangguo Sun, Guancheng Wan, Miao Yu, Junfeng Fang, Kun Wang, Dawei Cheng

**Abstract**: Recent advancements in large language model (LLM)-based agents have demonstrated that collective intelligence can significantly surpass the capabilities of individual agents, primarily due to well-crafted inter-agent communication topologies. Despite the diverse and high-performing designs available, practitioners often face confusion when selecting the most effective pipeline for their specific task: \textit{Which topology is the best choice for my task, avoiding unnecessary communication token overhead while ensuring high-quality solution?} In response to this dilemma, we introduce G-Designer, an adaptive, efficient, and robust solution for multi-agent deployment, which dynamically designs task-aware, customized communication topologies. Specifically, G-Designer models the multi-agent system as a multi-agent network, leveraging a variational graph auto-encoder to encode both the nodes (agents) and a task-specific virtual node, and decodes a task-adaptive and high-performing communication topology. Extensive experiments on six benchmarks showcase that G-Designer is: \textbf{(1) high-performing}, achieving superior results on MMLU with accuracy at $84.50\%$ and on HumanEval with pass@1 at $89.90\%$; \textbf{(2) task-adaptive}, architecting communication protocols tailored to task difficulty, reducing token consumption by up to $95.33\%$ on HumanEval; and \textbf{(3) adversarially robust}, defending against agent adversarial attacks with merely $0.3\%$ accuracy drop.

摘要: 基于大型语言模型(LLM)的代理的最新进展表明，集体智能可以显著超过单个代理的能力，这主要是由于精心设计的代理间通信拓扑。尽管有多样化和高性能的设计，但实践者在为他们的特定任务选择最有效的流水线时经常面临困惑：\textit{哪个拓扑是我的任务的最佳选择，在确保高质量解决方案的同时避免不必要的通信令牌开销？}针对这种困境，我们引入了G-Designer，这是一个自适应的、高效的、健壮的多代理部署解决方案，它动态地设计任务感知的、定制的通信拓扑。具体地说，G-Designer将多代理系统建模为多代理网络，利用变化图自动编码器对节点(代理)和特定于任务的虚拟节点进行编码，并解码任务自适应的高性能通信拓扑。在六个基准测试上的广泛实验表明，G-Designer是：\extbf{(1)高性能}，在MMLU上获得了更好的结果，准确率为84.50\$，在HumanEval上，PASS@1的准确率为89.90\$；\extbf{(2)任务自适应}，构建了针对任务难度的通信协议，在HumanEval上减少了高达95.33\$的令牌消耗；以及\extbf{(3)对手健壮性}，防御代理对手攻击，精确度仅下降了$0.3\%$。



## **5. R-MTLLMF: Resilient Multi-Task Large Language Model Fusion at the Wireless Edge**

R-MTLLMF：无线边缘的弹性多任务大型语言模型融合 eess.SP

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18220v1) [paper-pdf](http://arxiv.org/pdf/2411.18220v1)

**Authors**: Aladin Djuhera, Vlad C. Andrei, Mohsen Pourghasemian, Haris Gacanin, Holger Boche, Walid Saad

**Abstract**: Multi-task large language models (MTLLMs) are important for many applications at the wireless edge, where users demand specialized models to handle multiple tasks efficiently. However, training MTLLMs is complex and exhaustive, particularly when tasks are subject to change. Recently, the concept of model fusion via task vectors has emerged as an efficient approach for combining fine-tuning parameters to produce an MTLLM. In this paper, the problem of enabling edge users to collaboratively craft such MTTLMs via tasks vectors is studied, under the assumption of worst-case adversarial attacks. To this end, first the influence of adversarial noise to multi-task model fusion is investigated and a relationship between the so-called weight disentanglement error and the mean squared error (MSE) is derived. Using hypothesis testing, it is directly shown that the MSE increases interference between task vectors, thereby rendering model fusion ineffective. Then, a novel resilient MTLLM fusion (R-MTLLMF) is proposed, which leverages insights about the LLM architecture and fine-tuning process to safeguard task vector aggregation under adversarial noise by realigning the MTLLM. The proposed R-MTLLMF is then compared for both worst-case and ideal transmission scenarios to study the impact of the wireless channel. Extensive model fusion experiments with vision LLMs demonstrate R-MTLLMF's effectiveness, achieving close-to-baseline performance across eight different tasks in ideal noise scenarios and significantly outperforming unprotected model fusion in worst-case scenarios. The results further advocate for additional physical layer protection for a holistic approach to resilience, from both a wireless and LLM perspective.

摘要: 多任务大型语言模型(MTLLM)对于无线边缘的许多应用非常重要，因为用户需要专门的模型来高效地处理多个任务。然而，培训MTLLM是复杂和详尽的，特别是在任务可能发生变化的情况下。最近，基于任务向量的模型融合的概念已经成为一种结合微调参数以产生MTLLM的有效方法。本文在假设最坏情况下的对抗性攻击的前提下，研究了边缘用户通过任务向量协作生成MTTLM的问题。为此，首先研究了对抗性噪声对多任务模型融合的影响，推导了加权解缠误差与均方误差之间的关系。通过假设检验，直接表明MSE增加了任务向量之间的干扰，从而使模型融合无效。然后，提出了一种新的弹性MTLLM融合算法(R-MTLLMF)，该算法利用对LLM体系结构和微调过程的深入了解，通过重新排列MTLLM来保护对抗噪声下的任务向量聚合。然后将所提出的R-MTLLMF在最坏情况和理想传输场景下进行比较，以研究无线信道的影响。用VISION LLMS进行的大量模型融合实验证明了R-MTLLMF的有效性，在理想噪声场景中，R-MTLLMF在八个不同任务上的性能接近基线，而在最坏情况下，R-MTLLMF的性能明显优于无保护的模型融合。从无线和LLM的角度来看，研究结果进一步倡导为整体恢复方法提供额外的物理层保护。



## **6. Privacy-preserving Robotic-based Multi-factor Authentication Scheme for Secure Automated Delivery System**

安全自动交付系统中保护隐私的基于机器人的多因素认证方案 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18027v1) [paper-pdf](http://arxiv.org/pdf/2411.18027v1)

**Authors**: Yang Yang, Aryan Mohammadi Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Package delivery is a critical aspect of various industries, but it often incurs high financial costs and inefficiencies when relying solely on human resources. The last-mile transport problem, in particular, contributes significantly to the expenditure of human resources in major companies. Robot-based delivery systems have emerged as a potential solution for last-mile delivery to address this challenge. However, robotic delivery systems still face security and privacy issues, like impersonation, replay, man-in-the-middle attacks (MITM), unlinkability, and identity theft. In this context, we propose a privacy-preserving multi-factor authentication scheme specifically designed for robot delivery systems. Additionally, AI-assisted robotic delivery systems are susceptible to machine learning-based attacks (e.g. FGSM, PGD, etc.). We introduce the \emph{first} transformer-based audio-visual fusion defender to tackle this issue, which effectively provides resilience against adversarial samples. Furthermore, we provide a rigorous formal analysis of the proposed protocol and also analyse the protocol security using a popular symbolic proof tool called ProVerif and Scyther. Finally, we present a real-world implementation of the proposed robotic system with the computation cost and energy consumption analysis. Code and pre-trained models are available at: https://drive.google.com/drive/folders/18B2YbxtV0Pyj5RSFX-ZzCGtFOyorBHil

摘要: 包裹递送是各个行业的一个关键方面，但仅依靠人力资源往往会招致高昂的财务成本和效率低下。特别是最后一英里的运输问题，大大增加了大公司的人力资源支出。基于机器人的递送系统已经成为应对这一挑战的最后一英里递送的潜在解决方案。然而，机器人递送系统仍然面临安全和隐私问题，如模仿、重播、中间人攻击(MITM)、不可链接和身份盗窃。在此背景下，我们提出了一种专为机器人送货系统设计的隐私保护多因素认证方案。此外，人工智能辅助的机器人送货系统容易受到基于机器学习的攻击(例如FGSM、PGD等)。我们引入了基于变压器的视听融合防御者来解决这一问题，它有效地提供了对对手样本的弹性。此外，我们对协议进行了严格的形式化分析，并使用流行的符号证明工具ProVerif和Scyther对协议的安全性进行了分析。最后，我们给出了所提出的机器人系统的真实实现，并对计算成本和能量消耗进行了分析。代码和预先培训的模型可在以下网站获得：https://drive.google.com/drive/folders/18B2YbxtV0Pyj5RSFX-ZzCGtFOyorBHil



## **7. Leveraging A New GAN-based Transformer with ECDH Crypto-system for Enhancing Energy Theft Detection in Smart Grid**

利用新的基于GAN的Transformer和ECDH加密系统来增强智能电网中的能源盗窃检测 cs.CR

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.18023v1) [paper-pdf](http://arxiv.org/pdf/2411.18023v1)

**Authors**: Yang Yang, Xun Yuan, Arwa Alromih, Aryan Mohammadi Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Detecting energy theft is vital for effectively managing power grids, as it ensures precise billing and prevents financial losses. Split-learning emerges as a promising decentralized machine learning technique for identifying energy theft while preserving user data confidentiality. Nevertheless, traditional split learning approaches are vulnerable to privacy leakage attacks, which significantly threaten data confidentiality. To address this challenge, we propose a novel GAN-Transformer-based split learning framework in this paper. This framework leverages the strengths of the transformer architecture, which is known for its capability to process long-range dependencies in energy consumption data. Thus, it enhances the accuracy of energy theft detection without compromising user privacy. A distinctive feature of our approach is the deployment of a novel mask-based method, marking a first in its field to effectively combat privacy leakage in split learning scenarios targeted at AI-enabled adversaries. This method protects sensitive information during the model's training phase. Our experimental evaluations indicate that the proposed framework not only achieves accuracy levels comparable to conventional methods but also significantly enhances privacy protection. The results underscore the potential of the GAN-Transformer split learning framework as an effective and secure tool in the domain of energy theft detection.

摘要: 检测能源盗窃对于有效管理电网至关重要，因为它确保了准确的计费并防止了经济损失。分裂学习是一种很有前途的去中心化机器学习技术，用于识别能源盗窃，同时保护用户数据的机密性。然而，传统的分裂学习方法很容易受到隐私泄露攻击，这严重威胁到数据的保密性。为了应对这一挑战，本文提出了一种新的基于GaN-Transformer的分裂学习框架。该框架利用了转换器体系结构的优势，该体系结构以处理能源消耗数据中的长期依赖关系的能力而闻名。因此，它在不损害用户隐私的情况下提高了能源盗窃检测的准确性。我们方法的一个显著特点是部署了一种新的基于掩码的方法，标志着该领域首次在针对支持人工智能的对手的分裂学习场景中有效打击隐私泄露。该方法在模型的训练阶段保护敏感信息。我们的实验评估表明，该框架不仅达到了与传统方法相当的准确率水平，而且显著地增强了隐私保护。这些结果强调了GaN-Transformer分离学习框架作为一种有效和安全的工具在能源盗窃检测领域的潜力。



## **8. Exploring Visual Vulnerabilities via Multi-Loss Adversarial Search for Jailbreaking Vision-Language Models**

通过越狱视觉语言模型的多重损失对抗搜索探索视觉漏洞 cs.CV

**SubmitDate**: 2024-11-28    [abs](http://arxiv.org/abs/2411.18000v2) [paper-pdf](http://arxiv.org/pdf/2411.18000v2)

**Authors**: Shuyang Hao, Bryan Hooi, Jun Liu, Kai-Wei Chang, Zi Huang, Yujun Cai

**Abstract**: Despite inheriting security measures from underlying language models, Vision-Language Models (VLMs) may still be vulnerable to safety alignment issues. Through empirical analysis, we uncover two critical findings: scenario-matched images can significantly amplify harmful outputs, and contrary to common assumptions in gradient-based attacks, minimal loss values do not guarantee optimal attack effectiveness. Building on these insights, we introduce MLAI (Multi-Loss Adversarial Images), a novel jailbreak framework that leverages scenario-aware image generation for semantic alignment, exploits flat minima theory for robust adversarial image selection, and employs multi-image collaborative attacks for enhanced effectiveness. Extensive experiments demonstrate MLAI's significant impact, achieving attack success rates of 77.75% on MiniGPT-4 and 82.80% on LLaVA-2, substantially outperforming existing methods by margins of 34.37% and 12.77% respectively. Furthermore, MLAI shows considerable transferability to commercial black-box VLMs, achieving up to 60.11% success rate. Our work reveals fundamental visual vulnerabilities in current VLMs safety mechanisms and underscores the need for stronger defenses. Warning: This paper contains potentially harmful example text.

摘要: 尽管继承了底层语言模型的安全措施，但Vision-Language模型(VLM)可能仍然容易受到安全对齐问题的影响。通过实证分析，我们发现了两个关键发现：场景匹配的图像可以显著放大有害输出，与基于梯度的攻击中的常见假设相反，最小的损失值并不能保证最佳的攻击效果。基于这些见解，我们引入了MLAI(多损失对抗图像)，这是一个新的越狱框架，它利用场景感知图像生成来进行语义对齐，利用平坦极小理论来稳健地选择对抗性图像，并使用多图像协同攻击来增强有效性。大量的实验证明了MLAI的显著影响，在MiniGPT-4和LLaVA-2上分别获得了77.75%和82.80%的攻击成功率，分别比现有方法高出34.37%和12.77%。此外，MLAI表现出相当大的可移植到商业黑盒VLM的能力，实现了高达60.11%的成功率。我们的工作揭示了当前VLMS安全机制中的基本视觉漏洞，并强调了加强防御的必要性。警告：本文包含可能有害的示例文本。



## **9. Adversarial Training in Low-Label Regimes with Margin-Based Interpolation**

基于边际的内插的低标签方案中的对抗训练 cs.LG

**SubmitDate**: 2024-11-27    [abs](http://arxiv.org/abs/2411.17959v1) [paper-pdf](http://arxiv.org/pdf/2411.17959v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna

**Abstract**: Adversarial training has emerged as an effective approach to train robust neural network models that are resistant to adversarial attacks, even in low-label regimes where labeled data is scarce. In this paper, we introduce a novel semi-supervised adversarial training approach that enhances both robustness and natural accuracy by generating effective adversarial examples. Our method begins by applying linear interpolation between clean and adversarial examples to create interpolated adversarial examples that cross decision boundaries by a controlled margin. This sample-aware strategy tailors adversarial examples to the characteristics of each data point, enabling the model to learn from the most informative perturbations. Additionally, we propose a global epsilon scheduling strategy that progressively adjusts the upper bound of perturbation strengths during training. The combination of these strategies allows the model to develop increasingly complex decision boundaries with better robustness and natural accuracy. Empirical evaluations show that our approach effectively enhances performance against various adversarial attacks, such as PGD and AutoAttack.

摘要: 对抗性训练已经成为一种有效的方法来训练健壮的神经网络模型，这些模型能够抵抗对抗性攻击，即使在标签数据稀缺的低标签制度中也是如此。在本文中，我们介绍了一种新的半监督对抗性训练方法，通过生成有效的对抗性实例来提高稳健性和自然准确性。我们的方法首先在干净的和对抗性的例子之间应用线性内插，以创建内插的对抗性的例子，这些例子以受控的幅度跨越决策边界。这种样本感知策略根据每个数据点的特征定制对抗性示例，使模型能够从最具信息量的扰动中学习。此外，我们还提出了一种全局epsilon调度策略，该策略在训练过程中逐步调整扰动强度的上限。这些策略的组合使模型能够以更好的健壮性和自然准确性开发出日益复杂的决策边界。实验结果表明，该方法有效地提高了对PGD、AutoAttack等各种敌意攻击的性能。



## **10. Stealthy Multi-Task Adversarial Attacks**

隐形多任务对抗攻击 cs.CR

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17936v1) [paper-pdf](http://arxiv.org/pdf/2411.17936v1)

**Authors**: Jiacheng Guo, Tianyun Zhang, Lei Li, Haochen Yang, Hongkai Yu, Minghai Qin

**Abstract**: Deep Neural Networks exhibit inherent vulnerabilities to adversarial attacks, which can significantly compromise their outputs and reliability. While existing research primarily focuses on attacking single-task scenarios or indiscriminately targeting all tasks in multi-task environments, we investigate selectively targeting one task while preserving performance in others within a multi-task framework. This approach is motivated by varying security priorities among tasks in real-world applications, such as autonomous driving, where misinterpreting critical objects (e.g., signs, traffic lights) poses a greater security risk than minor depth miscalculations. Consequently, attackers may hope to target security-sensitive tasks while avoiding non-critical tasks from being compromised, thus evading being detected before compromising crucial functions. In this paper, we propose a method for the stealthy multi-task attack framework that utilizes multiple algorithms to inject imperceptible noise into the input. This novel method demonstrates remarkable efficacy in compromising the target task while simultaneously maintaining or even enhancing performance across non-targeted tasks - a criterion hitherto unexplored in the field. Additionally, we introduce an automated approach for searching the weighting factors in the loss function, further enhancing attack efficiency. Experimental results validate our framework's ability to successfully attack the target task while preserving the performance of non-targeted tasks. The automated loss function weight searching method demonstrates comparable efficacy to manual tuning, establishing a state-of-the-art multi-task attack framework.

摘要: 深度神经网络在敌意攻击中表现出固有的脆弱性，这可能会显著影响其输出和可靠性。虽然现有的研究主要集中在攻击单任务场景或不分青红皂白地针对多任务环境中的所有任务，但我们研究的是在多任务框架内选择性地针对一个任务，同时保持其他任务的性能。这种方法的动机是现实世界应用程序中不同任务之间的安全优先级不同，例如自动驾驶，其中误解关键对象(例如，标志、红绿灯)会比微小的深度错误计算带来更大的安全风险。因此，攻击者可能希望瞄准安全敏感任务，同时避免非关键任务受到危害，从而在危害关键功能之前避免被检测到。本文提出了一种隐身多任务攻击框架的方法，该框架利用多种算法向输入注入不可察觉的噪声。这一新的方法在折衷目标任务的同时保持甚至提高非目标任务的性能方面表现出显著的效果--这是该领域迄今尚未探索的标准。此外，我们还引入了一种自动搜索损失函数中加权因子的方法，进一步提高了攻击效率。实验结果验证了该框架在保持非目标任务性能的同时能够成功攻击目标任务的能力。自动损失函数权重搜索方法展示了与手动调整相当的效率，建立了最先进的多任务攻击框架。



## **11. Enhancing Robustness in Deep Reinforcement Learning: A Lyapunov Exponent Approach**

增强深度强化学习的鲁棒性：一种李雅普诺夫指数方法 cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.10674v2) [paper-pdf](http://arxiv.org/pdf/2410.10674v2)

**Authors**: Rory Young, Nicolas Pugeault

**Abstract**: Deep reinforcement learning agents achieve state-of-the-art performance in a wide range of simulated control tasks. However, successful applications to real-world problems remain limited. One reason for this dichotomy is because the learnt policies are not robust to observation noise or adversarial attacks. In this paper, we investigate the robustness of deep RL policies to a single small state perturbation in deterministic continuous control tasks. We demonstrate that RL policies can be deterministically chaotic, as small perturbations to the system state have a large impact on subsequent state and reward trajectories. This unstable non-linear behaviour has two consequences: first, inaccuracies in sensor readings, or adversarial attacks, can cause significant performance degradation; second, even policies that show robust performance in terms of rewards may have unpredictable behaviour in practice. These two facets of chaos in RL policies drastically restrict the application of deep RL to real-world problems. To address this issue, we propose an improvement on the successful Dreamer V3 architecture, implementing Maximal Lyapunov Exponent regularisation. This new approach reduces the chaotic state dynamics, rendering the learnt policies more resilient to sensor noise or adversarial attacks and thereby improving the suitability of deep reinforcement learning for real-world applications.

摘要: 深度强化学习代理在广泛的模拟控制任务中实现最先进的性能。然而，对现实世界问题的成功应用仍然有限。这种二分法的一个原因是，学习到的策略对观察噪音或对抗性攻击不是很健壮。本文研究了在确定性连续控制任务中，深度RL策略对单个小状态扰动的鲁棒性。我们证明了RL策略可以是确定性混沌的，因为系统状态的微小扰动对随后的状态和奖励轨迹有很大的影响。这种不稳定的非线性行为有两个后果：第一，传感器读数的不准确或敌意攻击可能导致性能显著下降；第二，即使是在奖励方面表现强劲的策略，在实践中也可能有不可预测的行为。RL政策中的这两个方面的混乱极大地限制了深度RL在现实世界问题中的应用。为了解决这个问题，我们提出了对成功的Dreamer V3架构的改进，实现了最大Lyapunov指数正则化。这种新方法降低了混沌状态动态，使学习到的策略对传感器噪声或敌意攻击更具弹性，从而提高了深度强化学习在实际应用中的适用性。



## **12. TrackPGD: Efficient Adversarial Attack using Object Binary Masks against Robust Transformer Trackers**

TrackPVD：使用对象二进制掩蔽针对稳健的Transformer跟踪器的高效对抗攻击 cs.CV

Accepted in The 3rd New Frontiers in Adversarial Machine Learning  (AdvML Frontiers @NeurIPS2024)

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2407.03946v2) [paper-pdf](http://arxiv.org/pdf/2407.03946v2)

**Authors**: Fatemeh Nourilenjan Nokabadi, Yann Batiste Pequignot, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Adversarial perturbations can deceive neural networks by adding small, imperceptible noise to the input. Recent object trackers with transformer backbones have shown strong performance on tracking datasets, but their adversarial robustness has not been thoroughly evaluated. While transformer trackers are resilient to black-box attacks, existing white-box adversarial attacks are not universally applicable against these new transformer trackers due to differences in backbone architecture. In this work, we introduce TrackPGD, a novel white-box attack that utilizes predicted object binary masks to target robust transformer trackers. Built upon the powerful segmentation attack SegPGD, our proposed TrackPGD effectively influences the decisions of transformer-based trackers. Our method addresses two primary challenges in adapting a segmentation attack for trackers: limited class numbers and extreme pixel class imbalance. TrackPGD uses the same number of iterations as other attack methods for tracker networks and produces competitive adversarial examples that mislead transformer and non-transformer trackers such as MixFormerM, OSTrackSTS, TransT-SEG, and RTS on datasets including VOT2022STS, DAVIS2016, UAV123, and GOT-10k.

摘要: 对抗性扰动可以通过在输入中添加微小的、不可察觉的噪声来欺骗神经网络。最近的带有变压器主干的对象跟踪器在跟踪数据集方面表现出了很强的性能，但它们的对抗性健壮性还没有得到彻底的评估。虽然变压器跟踪器对黑盒攻击具有弹性，但由于骨干架构的差异，现有的白盒对抗性攻击并不适用于这些新的变压器跟踪器。在这项工作中，我们介绍了TrackPGD，一种新的白盒攻击，它利用预测的对象二进制掩码来攻击稳健的变压器跟踪器。基于强大的分段攻击SegPGD，我们提出的TrackPGD有效地影响了基于变压器的跟踪器的决策。我们的方法解决了针对跟踪器的分割攻击的两个主要挑战：有限的类别数量和极端的像素类别失衡。TrackPGD使用与其他跟踪网络攻击方法相同的迭代次数，并在VOT2022STS、DAVIS2016、UAV123和GET-10k等数据集上生成误导转换器和非转换器跟踪器(如MixFormerM、OSTrackSTS、TransT-SEG和RTS)的竞争性对手示例。



## **13. Mitigating the Impact of Noisy Edges on Graph-Based Algorithms via Adversarial Robustness Evaluation**

通过对抗鲁棒性评估减轻噪音边缘对基于图的算法的影响 cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2401.15615v2) [paper-pdf](http://arxiv.org/pdf/2401.15615v2)

**Authors**: Yongyu Wang, Xiaotian Zhuang

**Abstract**: Given that no existing graph construction method can generate a perfect graph for a given dataset, graph-based algorithms are often affected by redundant and erroneous edges present within the constructed graphs. In this paper, we view these noisy edges as adversarial attack and propose to use a spectral adversarial robustness evaluation method to mitigate the impact of noisy edges on the performance of graph-based algorithms. Our method identifies the points that are less vulnerable to noisy edges and leverages only these robust points to perform graph-based algorithms. Our experiments demonstrate that our methodology is highly effective and outperforms state-of-the-art denoising methods by a large margin.

摘要: 鉴于现有的图构建方法无法为给定数据集生成完美的图，因此基于图的算法通常会受到所构建的图中存在的冗余和错误边的影响。本文将这些有噪边缘视为对抗性攻击，并建议使用谱对抗鲁棒性评估方法来减轻有噪边缘对基于图的算法性能的影响。我们的方法识别不太容易受到噪音边缘影响的点，并仅利用这些稳健的点来执行基于图的算法。我们的实验表明，我们的方法非常有效，并且大大优于最先进的去噪方法。



## **14. Adversarial Bounding Boxes Generation (ABBG) Attack against Visual Object Trackers**

针对视觉对象跟踪器的敌对边界盒生成（ABBG）攻击 cs.CV

Accepted in The 3rd New Frontiers in Adversarial Machine Learning  (AdvML Frontiers @NeurIPS2024)

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17468v1) [paper-pdf](http://arxiv.org/pdf/2411.17468v1)

**Authors**: Fatemeh Nourilenjan Nokabadi, Jean-Francois Lalonde, Christian Gagné

**Abstract**: Adversarial perturbations aim to deceive neural networks into predicting inaccurate results. For visual object trackers, adversarial attacks have been developed to generate perturbations by manipulating the outputs. However, transformer trackers predict a specific bounding box instead of an object candidate list, which limits the applicability of many existing attack scenarios. To address this issue, we present a novel white-box approach to attack visual object trackers with transformer backbones using only one bounding box. From the tracker predicted bounding box, we generate a list of adversarial bounding boxes and compute the adversarial loss for those bounding boxes. Experimental results demonstrate that our simple yet effective attack outperforms existing attacks against several robust transformer trackers, including TransT-M, ROMTrack, and MixFormer, on popular benchmark tracking datasets such as GOT-10k, UAV123, and VOT2022STS.

摘要: 对抗性扰动旨在欺骗神经网络预测不准确的结果。对于视觉对象跟踪器来说，对抗攻击已经被开发出来，通过操纵输出来产生扰动。然而，Transformer跟踪器预测特定的边界框而不是对象候选列表，这限制了许多现有攻击场景的适用性。为了解决这个问题，我们提出了一种新颖的白盒方法，仅使用一个边界框来攻击具有Transformer主干的视觉对象跟踪器。从跟踪器预测的边界框中，我们生成对抗边界框列表，并计算这些边界框的对抗损失。实验结果表明，在GOT-10 k、UAV 123和VOT 2022 STS等流行基准跟踪数据集上，我们简单而有效的攻击优于针对多种强大的Transformer跟踪器（包括TransT-M、ROMTrack和MixFormer）的现有攻击。



## **15. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

PEFTGuard：检测针对参数高效微调的后门攻击 cs.CR

20 pages, 8 figures

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17453v1) [paper-pdf](http://arxiv.org/pdf/2411.17453v1)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few of the current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters.   To fill this gap, we first construct (and will release) PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision our benchmark and method can shed light on future LLM backdoor detection research.

摘要: 微调是提高大型语言模型(LLM)在特定领域中性能的关键过程，参数高效微调(PEFT)因其能够通过集成低阶适配器来减少计算需求而广受欢迎。这些轻量级适配器，如Lora，可以在开源平台上共享和使用。然而，攻击者可以利用这一机制将后门注入这些适配器，导致不正确或有害的输出等恶意行为，这会给社区带来严重的安全风险。不幸的是，目前很少有人专注于分析后门模式或检测适配器中的后门。为了填补这一空白，我们首先构建(并将发布)PADB边，这是一个全面的基准测试，包含13,300个良性和反向适配器，通过各种数据集、攻击策略、PEFT方法和LLM进行了微调。此外，我们还提出了第一个针对基于PEFT的适配器的后门检测框架PEFTGuard。对PADBENCH的广泛评估表明，PEFTGuard的性能优于现有的检测方法，在大多数情况下实现了近乎完美的检测准确率(100%)。值得注意的是，PEFTGuard在三个方面表现出零命中率，包括不同的攻击、PEFT方法和适配器级别。此外，我们还考虑了各种自适应攻击，以展示PEFTGuard的高健壮性。我们进一步探索了几种可能的后门缓解防御措施，发现精细混合是最有效的方法。我们希望我们的基准和方法可以为未来的LLM后门检测研究提供参考。



## **16. Breaking the Illusion: Real-world Challenges for Adversarial Patches in Object Detection**

打破幻觉：对象检测中对抗性补丁的现实挑战 cs.CV

This paper has been accepted by the 1st Workshop on Enabling Machine  Learning Operations for next-Gen Embedded Wireless Networked Devices  (EMERGE), 2024

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2410.19863v2) [paper-pdf](http://arxiv.org/pdf/2410.19863v2)

**Authors**: Jakob Shack, Katarina Petrovic, Olga Saukh

**Abstract**: Adversarial attacks pose a significant threat to the robustness and reliability of machine learning systems, particularly in computer vision applications. This study investigates the performance of adversarial patches for the YOLO object detection network in the physical world. Two attacks were tested: a patch designed to be placed anywhere within the scene - global patch, and another patch intended to partially overlap with specific object targeted for removal from detection - local patch. Various factors such as patch size, position, rotation, brightness, and hue were analyzed to understand their impact on the effectiveness of the adversarial patches. The results reveal a notable dependency on these parameters, highlighting the challenges in maintaining attack efficacy in real-world conditions. Learning to align digitally applied transformation parameters with those measured in the real world still results in up to a 64\% discrepancy in patch performance. These findings underscore the importance of understanding environmental influences on adversarial attacks, which can inform the development of more robust defenses for practical machine learning applications.

摘要: 对抗性攻击对机器学习系统的健壮性和可靠性构成了严重威胁，尤其是在计算机视觉应用中。本文研究了YOLO目标检测网络中的敌意补丁在现实世界中的性能。测试了两种攻击：一种是被设计放置在场景全局面片内的任何位置的补丁，另一种是打算与要从检测局部面片中移除的特定对象部分重叠的补丁。分析了各种因素，如斑块大小、位置、旋转、亮度和色调，以了解它们对对抗性斑块有效性的影响。结果显示了对这些参数的显著依赖，突显了在现实世界条件下保持攻击效率的挑战。学习将数字应用的变换参数与现实世界中测量的参数对齐仍然会导致修补程序性能上存在高达的差异。这些发现强调了了解环境对对抗性攻击的影响的重要性，这可以为为实际的机器学习应用开发更强大的防御提供信息。



## **17. Enhancing generalization in high energy physics using white-box adversarial attacks**

使用白盒对抗攻击增强高能物理学的概括性 hep-ph

10 pages, 4 figures, 8 tables, 3 algorithms, to be published in  Physical Review D (PRD), presented at the ML4Jets 2024 conference

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.09296v2) [paper-pdf](http://arxiv.org/pdf/2411.09296v2)

**Authors**: Franck Rothen, Samuel Klein, Matthew Leigh, Tobias Golling

**Abstract**: Machine learning is becoming increasingly popular in the context of particle physics. Supervised learning, which uses labeled Monte Carlo (MC) simulations, remains one of the most widely used methods for discriminating signals beyond the Standard Model. However, this paper suggests that supervised models may depend excessively on artifacts and approximations from Monte Carlo simulations, potentially limiting their ability to generalize well to real data. This study aims to enhance the generalization properties of supervised models by reducing the sharpness of local minima. It reviews the application of four distinct white-box adversarial attacks in the context of classifying Higgs boson decay signals. The attacks are divided into weight space attacks, and feature space attacks. To study and quantify the sharpness of different local minima this paper presents two analysis methods: gradient ascent and reduced Hessian eigenvalue analysis. The results show that white-box adversarial attacks significantly improve generalization performance, albeit with increased computational complexity.

摘要: 在粒子物理的背景下，机器学习正变得越来越流行。监督学习使用标记的蒙特卡罗(MC)模拟，仍然是用于区分标准模型以外的信号的最广泛使用的方法之一。然而，本文认为，监督模型可能过度依赖于来自蒙特卡罗模拟的伪影和近似，潜在地限制了它们对真实数据的推广能力。该研究旨在通过降低局部极小值的锐度来增强监督模型的泛化性能。它回顾了四种不同的白盒对抗性攻击在对希格斯玻色子衰变信号进行分类的背景下的应用。攻击分为权重空间攻击和特征空间攻击。为了研究和量化不同局部极小值的锐度，本文提出了两种分析方法：梯度上升法和约化Hesse特征值分析法。结果表明，白盒对抗性攻击显著提高了泛化性能，但增加了计算复杂度。



## **18. BadScan: An Architectural Backdoor Attack on Visual State Space Models**

BadScan：对视觉状态空间模型的建筑后门攻击 cs.CV

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17283v1) [paper-pdf](http://arxiv.org/pdf/2411.17283v1)

**Authors**: Om Suhas Deshmukh, Sankalp Nagaonkar, Achyut Mani Tripathi, Ashish Mishra

**Abstract**: The newly introduced Visual State Space Model (VMamba), which employs \textit{State Space Mechanisms} (SSM) to interpret images as sequences of patches, has shown exceptional performance compared to Vision Transformers (ViT) across various computer vision tasks. However, recent studies have highlighted that deep models are susceptible to adversarial attacks. One common approach is to embed a trigger in the training data to retrain the model, causing it to misclassify data samples into a target class, a phenomenon known as a backdoor attack. In this paper, we first evaluate the robustness of the VMamba model against existing backdoor attacks. Based on this evaluation, we introduce a novel architectural backdoor attack, termed BadScan, designed to deceive the VMamba model. This attack utilizes bit plane slicing to create visually imperceptible backdoored images. During testing, if a trigger is detected by performing XOR operations between the $k^{th}$ bit planes of the modified triggered patches, the traditional 2D selective scan (SS2D) mechanism in the visual state space (VSS) block of VMamba is replaced with our newly designed BadScan block, which incorporates four newly developed scanning patterns. We demonstrate that the BadScan backdoor attack represents a significant threat to visual state space models and remains effective even after complete retraining from scratch. Experimental results on two widely used image classification datasets, CIFAR-10, and ImageNet-1K, reveal that while visual state space models generally exhibit robustness against current backdoor attacks, the BadScan attack is particularly effective, achieving a higher Triggered Accuracy Ratio (TAR) in misleading the VMamba model and its variants.

摘要: 新引入的视觉状态空间模型(VMamba)使用状态空间机制(SSM)将图像解释为面片序列，与视觉转换器(VIT)相比，它在各种计算机视觉任务中表现出了优异的性能。然而，最近的研究强调，深度模型容易受到对抗性攻击。一种常见的方法是在训练数据中嵌入触发器来重新训练模型，导致它将数据样本错误地分类到目标类，这种现象被称为后门攻击。在本文中，我们首先评估了VMamba模型对现有后门攻击的健壮性。基于这种评估，我们引入了一种新的架构后门攻击，称为BadScan，旨在欺骗VMamba模型。该攻击利用位平面切片来创建视觉上不可察觉的回溯图像。在测试过程中，如果通过在修改的触发补丁的$k^{th}$位平面之间执行异或操作来检测到触发，则将VMamba视觉状态空间(VSS)块中的传统2D选择性扫描(SS2D)机制替换为我们新设计的BadScan块，该块包含四个新开发的扫描模式。我们证明了BadScan后门攻击对视觉状态空间模型构成了重大威胁，并且即使在从头开始完全重新训练之后仍然有效。在两个广泛使用的图像分类数据集CIFAR-10和ImageNet-1K上的实验结果表明，虽然视觉状态空间模型对当前的后门攻击表现出了鲁棒性，但BadScan攻击尤其有效，在误导VMamba模型及其变体方面获得了更高的触发准确率(TAR)。



## **19. BadSFL: Backdoor Attack against Scaffold Federated Learning**

BadSFL：针对脚手架联邦学习的后门攻击 cs.LG

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.16167v2) [paper-pdf](http://arxiv.org/pdf/2411.16167v2)

**Authors**: Xingshuo Han, Xuanye Zhang, Xiang Lan, Haozhao Wang, Shengmin Xu, Shen Ren, Jason Zeng, Ming Wu, Michael Heinrich, Tianwei Zhang

**Abstract**: Federated learning (FL) enables the training of deep learning models on distributed clients to preserve data privacy. However, this learning paradigm is vulnerable to backdoor attacks, where malicious clients can upload poisoned local models to embed backdoors into the global model, leading to attacker-desired predictions. Existing backdoor attacks mainly focus on FL with independently and identically distributed (IID) scenarios, while real-world FL training data are typically non-IID. Current strategies for non-IID backdoor attacks suffer from limitations in maintaining effectiveness and durability. To address these challenges, we propose a novel backdoor attack method, BadSFL, specifically designed for the FL framework using the scaffold aggregation algorithm in non-IID settings. BadSFL leverages a Generative Adversarial Network (GAN) based on the global model to complement the training set, achieving high accuracy on both backdoor and benign samples. It utilizes a specific feature as the backdoor trigger to ensure stealthiness, and exploits the Scaffold's control variate to predict the global model's convergence direction, ensuring the backdoor's persistence. Extensive experiments on three benchmark datasets demonstrate the high effectiveness, stealthiness, and durability of BadSFL. Notably, our attack remains effective over 60 rounds in the global model and up to 3 times longer than existing baseline attacks after stopping the injection of malicious updates.

摘要: 联合学习(FL)支持在分布式客户端上训练深度学习模型以保护数据隐私。然而，这种学习模式容易受到后门攻击，恶意客户端可以上传有毒的本地模型，将后门嵌入到全局模型中，从而导致攻击者想要的预测。现有的后门攻击主要针对具有独立同分布(IID)场景的FL，而现实世界中的FL训练数据通常是非IID的。目前针对非IID后门攻击的战略在维持有效性和持久性方面存在局限性。为了应对这些挑战，我们提出了一种新的后门攻击方法BadSFL，该方法专门为FL框架设计，在非IID环境下使用脚手架聚集算法。BadSFL利用基于全球模型的生成性对抗网络(GAN)来补充训练集，在后门和良性样本上都实现了高精度。它利用特定的特征作为后门触发器来确保隐蔽性，并利用脚手架的控制变量来预测全局模型的收敛方向，确保后门的持久性。在三个基准数据集上的大量实验证明了BadSFL的高效性、隐蔽性和持久性。值得注意的是，在全球模型中，我们的攻击在60轮以上仍然有效，在停止注入恶意更新后，攻击时间最长可达现有基线攻击的3倍。



## **20. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

通过自我提示校准对微调大型语言模型的实用成员推断攻击 cs.CL

Repo: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2311.06062v4) [paper-pdf](http://arxiv.org/pdf/2311.06062v4)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Existing MIAs designed for large language models (LLMs) can be bifurcated into two types: reference-free and reference-based attacks. Although reference-based attacks appear promising performance by calibrating the probability measured on the target model with reference models, this illusion of privacy risk heavily depends on a reference dataset that closely resembles the training set. Both two types of attacks are predicated on the hypothesis that training records consistently maintain a higher probability of being sampled. However, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. Thus, these reasons lead to high false-positive rates of MIAs in practical scenarios. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs. Furthermore, we introduce probabilistic variation, a more reliable membership signal based on LLM memorization rather than overfitting, from which we rediscover the neighbour attack with theoretical grounding. Comprehensive evaluation conducted on three datasets and four exemplary LLMs shows that SPV-MIA raises the AUC of MIAs from 0.7 to a significantly high level of 0.9. Our code and dataset are available at: https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。现有的针对大型语言模型的MIA可以分为两种类型：无引用攻击和基于引用的攻击。虽然基于参考模型的攻击通过使用参考模型校准在目标模型上测量的概率而显示出良好的性能，但这种隐私风险的错觉严重依赖于与训练集非常相似的参考数据集。这两种类型的攻击都是基于这样的假设，即训练记录始终保持更高的被抽样概率。然而，这一假设严重依赖于目标模型的过拟合，而多种正则化方法和LLMS的推广将缓解这一问题。因此，这些原因导致在实际场景中MIA的假阳性率很高。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，我们引入了一种自我提示的方法，它构造数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。此外，我们引入了概率变异，这是一种基于LLM记忆而不是过拟合的更可靠的成员信号，从而重新发现了具有理论基础的邻居攻击。在三个数据集和四个示范性LLM上进行的综合评估表明，SPV-MIA将MIA的AUC从0.7提高到0.9的显著高水平。我们的代码和数据集可在以下网址获得：https://github.com/tsinghua-fib-lab/NeurIPS2024_SPV-MIA



## **21. RED: Robust Environmental Design**

RED：稳健的环境设计 cs.CV

**SubmitDate**: 2024-11-26    [abs](http://arxiv.org/abs/2411.17026v1) [paper-pdf](http://arxiv.org/pdf/2411.17026v1)

**Authors**: Jinghan Yang

**Abstract**: The classification of road signs by autonomous systems, especially those reliant on visual inputs, is highly susceptible to adversarial attacks. Traditional approaches to mitigating such vulnerabilities have focused on enhancing the robustness of classification models. In contrast, this paper adopts a fundamentally different strategy aimed at increasing robustness through the redesign of road signs themselves. We propose an attacker-agnostic learning scheme to automatically design road signs that are robust to a wide array of patch-based attacks. Empirical tests conducted in both digital and physical environments demonstrate that our approach significantly reduces vulnerability to patch attacks, outperforming existing techniques.

摘要: 自治系统（尤其是依赖视觉输入的系统）对路标进行分类，极易受到对抗性攻击。缓解此类漏洞的传统方法侧重于增强分类模型的稳健性。相比之下，本文采用了一种根本不同的策略，旨在通过重新设计路标本身来提高稳健性。我们提出了一种攻击者不可知的学习方案，以自动设计对各种基于补丁的攻击具有鲁棒性的路标。在数字和物理环境中进行的经验测试表明，我们的方法显着降低了补丁攻击的脆弱性，优于现有技术。



## **22. Unlocking The Potential of Adaptive Attacks on Diffusion-Based Purification**

释放基于扩散的净化自适应攻击的潜力 cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16598v1) [paper-pdf](http://arxiv.org/pdf/2411.16598v1)

**Authors**: Andre Kassis, Urs Hengartner, Yaoliang Yu

**Abstract**: Diffusion-based purification (DBP) is a defense against adversarial examples (AEs), amassing popularity for its ability to protect classifiers in an attack-oblivious manner and resistance to strong adversaries with access to the defense. Its robustness has been claimed to ensue from the reliance on diffusion models (DMs) that project the AEs onto the natural distribution. We revisit this claim, focusing on gradient-based strategies that back-propagate the loss gradients through the defense, commonly referred to as ``adaptive attacks". Analytically, we show that such an optimization method invalidates DBP's core foundations, effectively targeting the DM rather than the classifier and restricting the purified outputs to a distribution over malicious samples instead. Thus, we reassess the reported empirical robustness, uncovering implementation flaws in the gradient back-propagation techniques used thus far for DBP. We fix these issues, providing the first reliable gradient library for DBP and demonstrating how adaptive attacks drastically degrade its robustness. We then study a less efficient yet stricter majority-vote setting where the classifier evaluates multiple purified copies of the input to make its decision. Here, DBP's stochasticity enables it to remain partially robust against traditional norm-bounded AEs. We propose a novel adaptation of a recent optimization method against deepfake watermarking that crafts systemic malicious perturbations while ensuring imperceptibility. When integrated with the adaptive attack, it completely defeats DBP, even in the majority-vote setup. Our findings prove that DBP, in its current state, is not a viable defense against AEs.

摘要: 基于扩散的净化(DBP)是对敌意实例(AEs)的一种防御，它能够以攻击无关的方式保护分类器，并通过访问防御来抵抗强大的对手，因此越来越受欢迎。它的稳健性声称源于对扩散模型(DM)的依赖，这些扩散模型将AE投影到自然分布上。我们重新审视了这一说法，重点是基于梯度的策略，这种策略通过防御反向传播损失梯度，通常被称为“自适应攻击”。分析表明，这样的优化方法使DBP的核心基础失效，有效地针对DM而不是分类器，并将纯化的输出限制为对恶意样本的分布。因此，我们重新评估了已报道的经验稳健性，发现了迄今用于DBP的梯度反向传播技术中的实现缺陷。我们修复了这些问题，为DBP提供了第一个可靠的梯度库，并展示了自适应攻击如何显著降低其健壮性。然后我们研究一种效率较低但更严格的多数投票设置，在该设置中，分类器评估输入的多个净化副本以做出决定。这里，DBP的随机性使其能够对传统范数有界的AE保持部分健壮性。我们提出了一种新的针对深度伪水印的优化方法，该方法在确保不可感知性的同时，巧妙地制造了系统性恶意扰动。当与自适应攻击相结合时，它完全击败了DBP，即使在多数票设置中也是如此。我们的发现证明，在目前的状态下，DBP不是对抗AEs的可行防御措施。



## **23. Adversarial Attacks for Drift Detection**

漂移检测的对抗攻击 cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16591v1) [paper-pdf](http://arxiv.org/pdf/2411.16591v1)

**Authors**: Fabian Hinder, Valerie Vaquet, Barbara Hammer

**Abstract**: Concept drift refers to the change of data distributions over time. While drift poses a challenge for learning models, requiring their continual adaption, it is also relevant in system monitoring to detect malfunctions, system failures, and unexpected behavior. In the latter case, the robust and reliable detection of drifts is imperative. This work studies the shortcomings of commonly used drift detection schemes. We show how to construct data streams that are drifting without being detected. We refer to those as drift adversarials. In particular, we compute all possible adversairals for common detection schemes and underpin our theoretical findings with empirical evaluations.

摘要: 概念漂移是指数据分布随时间的变化。虽然漂移对学习模型构成了挑战，需要它们的持续适应，但它在系统监控中也与检测故障、系统故障和意外行为相关。在后一种情况下，对漂移进行稳健且可靠的检测至关重要。这项工作研究了常用漂移检测方案的缺点。我们展示了如何构建漂移而不被检测到的数据流。我们将这些称为漂移对手。特别是，我们计算了常见检测方案的所有可能的不利因素，并通过经验评估来支持我们的理论发现。



## **24. XAI and Android Malware Models**

XAI和Android恶意软件模型 cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16817v1) [paper-pdf](http://arxiv.org/pdf/2411.16817v1)

**Authors**: Maithili Kulkarni, Mark Stamp

**Abstract**: Android malware detection based on machine learning (ML) and deep learning (DL) models is widely used for mobile device security. Such models offer benefits in terms of detection accuracy and efficiency, but it is often difficult to understand how such learning models make decisions. As a result, these popular malware detection strategies are generally treated as black boxes, which can result in a lack of trust in the decisions made, as well as making adversarial attacks more difficult to detect. The field of eXplainable Artificial Intelligence (XAI) attempts to shed light on such black box models. In this paper, we apply XAI techniques to ML and DL models that have been trained on a challenging Android malware classification problem. Specifically, the classic ML models considered are Support Vector Machines (SVM), Random Forest, and $k$-Nearest Neighbors ($k$-NN), while the DL models we consider are Multi-Layer Perceptrons (MLP) and Convolutional Neural Networks (CNN). The state-of-the-art XAI techniques that we apply to these trained models are Local Interpretable Model-agnostic Explanations (LIME), Shapley Additive exPlanations (SHAP), PDP plots, ELI5, and Class Activation Mapping (CAM). We obtain global and local explanation results, and we discuss the utility of XAI techniques in this problem domain. We also provide a literature review of XAI work related to Android malware.

摘要: 基于机器学习(ML)和深度学习(DL)模型的Android恶意软件检测被广泛应用于移动设备安全。这类模型在检测准确性和效率方面提供了好处，但通常很难理解这种学习模型如何做出决策。因此，这些流行的恶意软件检测策略通常被视为黑盒，这可能导致对所做决策缺乏信任，并使敌意攻击更难检测。可解释人工智能(XAI)领域试图阐明这种黑盒模型。在本文中，我们将XAI技术应用于ML和DL模型，这些模型已经针对一个具有挑战性的Android恶意软件分类问题进行了训练。具体来说，经典的最大似然模型有支持向量机、随机森林和$k$-近邻($k$-NN)，而我们考虑的动态学习模型是多层感知器和卷积神经网络。我们应用于这些训练模型的最先进的XAI技术是局部可解释模型不可知性解释(LIME)、Shapley附加解释(Shap)、PDP图、ELI5和类激活映射(CAM)。我们得到了全局和局部的解释结果，并讨论了XAI技术在这个问题域中的应用。我们还提供了与Android恶意软件相关的XAI工作的文献回顾。



## **25. Privacy Protection in Personalized Diffusion Models via Targeted Cross-Attention Adversarial Attack**

通过有针对性的交叉注意对抗攻击实现个性化扩散模型中的隐私保护 cs.CV

Accepted at Safe Generative AI Workshop (NeurIPS 2024)

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16437v1) [paper-pdf](http://arxiv.org/pdf/2411.16437v1)

**Authors**: Xide Xu, Muhammad Atif Butt, Sandesh Kamath, Bogdan Raducanu

**Abstract**: The growing demand for customized visual content has led to the rise of personalized text-to-image (T2I) diffusion models. Despite their remarkable potential, they pose significant privacy risk when misused for malicious purposes. In this paper, we propose a novel and efficient adversarial attack method, Concept Protection by Selective Attention Manipulation (CoPSAM) which targets only the cross-attention layers of a T2I diffusion model. For this purpose, we carefully construct an imperceptible noise to be added to clean samples to get their adversarial counterparts. This is obtained during the fine-tuning process by maximizing the discrepancy between the corresponding cross-attention maps of the user-specific token and the class-specific token, respectively. Experimental validation on a subset of CelebA-HQ face images dataset demonstrates that our approach outperforms existing methods. Besides this, our method presents two important advantages derived from the qualitative evaluation: (i) we obtain better protection results for lower noise levels than our competitors; and (ii) we protect the content from unauthorized use thereby protecting the individual's identity from potential misuse.

摘要: 对定制视觉内容日益增长的需求导致了个性化文本到图像(T2I)扩散模式的兴起。尽管它们具有非凡的潜力，但如果被滥用于恶意目的，它们会带来重大的隐私风险。在本文中，我们提出了一种新颖而有效的敌意攻击方法--选择性注意操纵概念保护(CoPSAM)，它只针对T2I扩散模型的交叉注意层。为此，我们精心构造了一种不可察觉的噪声，将其添加到干净的样本中，以获得它们的对手。这是在微调过程中通过分别最大化特定于用户的令牌和特定于类的令牌的对应交叉注意图之间的差异来获得的。在CelebA-HQ人脸图像的一个子集上的实验验证表明，该方法的性能优于现有的方法。此外，我们的方法从定性评估中获得了两个重要的优势：(I)我们在较低噪声水平下获得了比竞争对手更好的保护结果；(Ii)我们保护内容不被未经授权使用，从而保护个人身份不被潜在的滥用。



## **26. Dark Miner: Defend against undesired generation for text-to-image diffusion models**

Dark Miner：防止文本到图像扩散模型的不必要生成 cs.CV

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2409.17682v2) [paper-pdf](http://arxiv.org/pdf/2409.17682v2)

**Authors**: Zheling Meng, Bo Peng, Xiaochuan Jin, Yue Jiang, Jing Dong, Wei Wang

**Abstract**: Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available at https://github.com/RichardSunnyMeng/DarkMiner-offical-codes.

摘要: 文本到图像的扩散模型已经被证明，由于未过滤的大规模训练数据，如性图像和版权，需要擦除不希望看到的概念，从而产生了不希望看到的结果。现有的方法大多着眼于修改以包含目标概念的文本为条件的生成概率。然而，它们不能保证在训练阶段看不到的期望文本的生成，特别是对于来自恶意攻击的对抗性文本。本文对擦除任务进行了分析，指出现有方法不能保证最小化不期望产生的总概率。为了解决这个问题，我们提出了Dark Miner。它需要一个重复的三个阶段的过程，包括挖掘、验证和规避。该方法贪婪地挖掘具有最大目标概念生成概率的嵌入，并更有效地减少目标概念的生成。在实验中，我们评估了它在不恰当概念、对象概念和风格概念上的表现。与以往的方法相比，我们的方法在保持模型的原生生成能力的同时，取得了更好的擦除和防御效果，特别是在多个对手攻击的情况下。我们的代码将在https://github.com/RichardSunnyMeng/DarkMiner-offical-codes.上提供



## **27. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16782v1) [paper-pdf](http://arxiv.org/pdf/2411.16782v1)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: A longstanding problem of deep learning models is their vulnerability to adversarial examples, which are often generated by applying imperceptible perturbations to natural examples. Adversarial examples exhibit cross-model transferability, enabling to attack black-box models with limited information about their architectures and parameters. Model ensembling is an effective strategy to improve the transferability by attacking multiple surrogate models simultaneously. However, as prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the findings in large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. By analyzing the relationship between the number of surrogate models and transferability of adversarial examples, we conclude with clear scaling laws, emphasizing the potential of using more surrogate models to enhance adversarial transferability. Extensive experiments verify the claims on standard image classifiers, multimodal large language models, and even proprietary models like GPT-4o, demonstrating consistent scaling effects and impressive attack success rates with more surrogate models. Further studies by visualization indicate that scaled attacks bring better interpretability in semantics, indicating that the common features of models are captured.

摘要: 深度学习模型的一个长期存在的问题是它们对对抗性示例的脆弱性，这些示例通常是通过对自然示例应用不可察觉的扰动而产生的。对抗性的例子表现出跨模型的可转移性，使得能够攻击具有关于其体系结构和参数的有限信息的黑盒模型。模型集成是一种通过同时攻击多个代理模型来提高可转移性的有效策略。然而，由于先前的研究通常采用的模型很少，所以增加模型的数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基础模型的启发，我们研究了黑盒对抗攻击的标度规律。通过分析代理模型的数量与对抗性实例的可转移性之间的关系，我们得出了明确的尺度律，强调了使用更多的代理模型来提高对抗性实例可转移性的潜力。广泛的实验验证了标准图像分类器、多模式大型语言模型，甚至像GPT-4o这样的专有模型的说法，展示了一致的缩放效果和令人印象深刻的攻击成功率，以及更多的代理模型。进一步的可视化研究表明，规模化攻击在语义上具有更好的可解释性，能够捕捉到模型的共性特征。



## **28. Sparse patches adversarial attacks via extrapolating point-wise information**

稀疏通过推断逐点信息来修复对抗攻击 cs.CV

AdvML-Frontiers 24: The 3nd Workshop on New Frontiers in Adversarial  Machine Learning, NeurIPS 24

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.16162v1) [paper-pdf](http://arxiv.org/pdf/2411.16162v1)

**Authors**: Yaniv Nemcovsky, Avi Mendelson, Chaim Baskin

**Abstract**: Sparse and patch adversarial attacks were previously shown to be applicable in realistic settings and are considered a security risk to autonomous systems. Sparse adversarial perturbations constitute a setting in which the adversarial perturbations are limited to affecting a relatively small number of points in the input. Patch adversarial attacks denote the setting where the sparse attacks are limited to a given structure, i.e., sparse patches with a given shape and number. However, previous patch adversarial attacks do not simultaneously optimize multiple patches' locations and perturbations. This work suggests a novel approach for sparse patches adversarial attacks via point-wise trimming dense adversarial perturbations. Our approach enables simultaneous optimization of multiple sparse patches' locations and perturbations for any given number and shape. Moreover, our approach is also applicable for standard sparse adversarial attacks, where we show that it significantly improves the state-of-the-art over multiple extensive settings. A reference implementation of the proposed method and the reported experiments is provided at \url{https://github.com/yanemcovsky/SparsePatches.git}

摘要: 稀疏和补丁对抗性攻击以前被证明在现实环境中适用，并被认为是自治系统的安全风险。稀疏对抗性扰动构成这样一种设置，在该设置中，对抗性扰动被限制为影响输入中相对较少数量的点。补丁对抗性攻击是指将稀疏攻击限制在给定的结构中，即具有给定形状和数量的稀疏补丁的设置。然而，以往的补丁对抗性攻击并不能同时优化多个补丁的位置和扰动。这项工作提出了一种新的方法，通过逐点裁剪密集的敌意扰动来进行稀疏补丁的对抗性攻击。我们的方法可以同时优化任意给定数量和形状的多个稀疏块的位置和扰动。此外，我们的方法也适用于标准稀疏对抗性攻击，在这些攻击中，我们表明它在多个广泛环境下显著改善了最新技术。在\url{https://github.com/yanemcovsky/SparsePatches.git}上提供了所提出的方法和报告的实验的参考实现



## **29. Unlearn to Relearn Backdoors: Deferred Backdoor Functionality Attacks on Deep Learning Models**

放弃学习重新学习后门：对深度学习模型的延迟后门功能攻击 cs.CR

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2411.14449v2) [paper-pdf](http://arxiv.org/pdf/2411.14449v2)

**Authors**: Jeongjin Shin, Sangdon Park

**Abstract**: Deep learning models are vulnerable to backdoor attacks, where adversaries inject malicious functionality during training that activates on trigger inputs at inference time. Extensive research has focused on developing stealthy backdoor attacks to evade detection and defense mechanisms. However, these approaches still have limitations that leave the door open for detection and mitigation due to their inherent design to cause malicious behavior in the presence of a trigger. To address this limitation, we introduce Deferred Activated Backdoor Functionality (DABF), a new paradigm in backdoor attacks. Unlike conventional attacks, DABF initially conceals its backdoor, producing benign outputs even when triggered. This stealthy behavior allows DABF to bypass multiple detection and defense methods, remaining undetected during initial inspections. The backdoor functionality is strategically activated only after the model undergoes subsequent updates, such as retraining on benign data. DABF attacks exploit the common practice in the life cycle of machine learning models to perform model updates and fine-tuning after initial deployment. To implement DABF attacks, we approach the problem by making the unlearning of the backdoor fragile, allowing it to be easily cancelled and subsequently reactivate the backdoor functionality. To achieve this, we propose a novel two-stage training scheme, called DeferBad. Our extensive experiments across various fine-tuning scenarios, backdoor attack types, datasets, and model architectures demonstrate the effectiveness and stealthiness of DeferBad.

摘要: 深度学习模型容易受到后门攻击，在后门攻击中，对手在训练期间注入恶意功能，在推理时激活触发器输入。广泛的研究集中在开发隐蔽的后门攻击，以逃避检测和防御机制。然而，这些方法仍然具有局限性，为检测和缓解敞开了大门，因为它们固有的设计是在存在触发器的情况下导致恶意行为。为了解决这一局限性，我们引入了延迟激活后门功能(DABF)，这是后门攻击中的一种新范例。与传统攻击不同，DABF最初隐藏其后门，即使被触发也会产生良性输出。这种隐身行为允许DABF绕过多种检测和防御方法，在初始检查期间保持不被检测到。只有在模型经历后续更新(如对良性数据进行再培训)后，才会战略性地激活后门功能。DABF攻击利用机器学习模型生命周期中的常见做法，在初始部署后执行模型更新和微调。为了实施DABF攻击，我们通过使后门的遗忘变得脆弱来解决问题，允许它很容易被取消并随后重新激活后门功能。为了实现这一点，我们提出了一种新的两阶段训练方案，称为DeferBad。我们在各种微调场景、后门攻击类型、数据集和模型架构上的广泛实验证明了DeferBad的有效性和隐蔽性。



## **30. Flexible Physical Camouflage Generation Based on a Differential Approach**

基于差异方法的灵活物理服装生成 cs.CV

**SubmitDate**: 2024-11-25    [abs](http://arxiv.org/abs/2402.13575v2) [paper-pdf](http://arxiv.org/pdf/2402.13575v2)

**Authors**: Yang Li, Wenyi Tan, Chenxing Zhao, Shuangju Zhou, Xinkai Liang, Quan Pan

**Abstract**: This study introduces a novel approach to neural rendering, specifically tailored for adversarial camouflage, within an extensive 3D rendering framework. Our method, named FPA, goes beyond traditional techniques by faithfully simulating lighting conditions and material variations, ensuring a nuanced and realistic representation of textures on a 3D target. To achieve this, we employ a generative approach that learns adversarial patterns from a diffusion model. This involves incorporating a specially designed adversarial loss and covert constraint loss to guarantee the adversarial and covert nature of the camouflage in the physical world. Furthermore, we showcase the effectiveness of the proposed camouflage in sticker mode, demonstrating its ability to cover the target without compromising adversarial information. Through empirical and physical experiments, FPA exhibits strong performance in terms of attack success rate and transferability. Additionally, the designed sticker-mode camouflage, coupled with a concealment constraint, adapts to the environment, yielding diverse styles of texture. Our findings highlight the versatility and efficacy of the FPA approach in adversarial camouflage applications.

摘要: 这项研究介绍了一种新的神经渲染方法，专门为对抗性伪装而定制，在一个广泛的3D渲染框架内。我们的方法名为FPA，超越了传统技术，忠实地模拟了照明条件和材质变化，确保了3D目标上纹理的细微差别和逼真表示。为了实现这一点，我们采用了一种生成性方法，从扩散模型中学习对抗性模式。这涉及到包括特别设计的对抗性损失和隐蔽约束损失，以保证物理世界中伪装的对抗性和隐蔽性。此外，我们在贴纸模式下展示了所提出的伪装的有效性，展示了其在不损害敌方信息的情况下覆盖目标的能力。通过实验和物理实验，FPA在攻击成功率和可转移性方面表现出很强的性能。此外，设计的贴纸模式伪装，加上隐藏限制，适应环境，产生不同风格的纹理。我们的发现突出了FPA方法在对抗性伪装应用中的多功能性和有效性。



## **31. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents**

AgentDojo：评估LLM代理的即时注入攻击和防御的动态环境 cs.CR

Updated version after fixing a bug in the Llama implementation and  updating the travel suite

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2406.13352v3) [paper-pdf](http://arxiv.org/pdf/2406.13352v3)

**Authors**: Edoardo Debenedetti, Jie Zhang, Mislav Balunović, Luca Beurer-Kellner, Marc Fischer, Florian Tramèr

**Abstract**: AI agents aim to solve complex tasks by combining text-based reasoning with external tool calls. Unfortunately, AI agents are vulnerable to prompt injection attacks where data returned by external tools hijacks the agent to execute malicious tasks. To measure the adversarial robustness of AI agents, we introduce AgentDojo, an evaluation framework for agents that execute tools over untrusted data. To capture the evolving nature of attacks and defenses, AgentDojo is not a static test suite, but rather an extensible environment for designing and evaluating new agent tasks, defenses, and adaptive attacks. We populate the environment with 97 realistic tasks (e.g., managing an email client, navigating an e-banking website, or making travel bookings), 629 security test cases, and various attack and defense paradigms from the literature. We find that AgentDojo poses a challenge for both attacks and defenses: state-of-the-art LLMs fail at many tasks (even in the absence of attacks), and existing prompt injection attacks break some security properties but not all. We hope that AgentDojo can foster research on new design principles for AI agents that solve common tasks in a reliable and robust manner.. We release the code for AgentDojo at https://github.com/ethz-spylab/agentdojo.

摘要: 人工智能代理旨在通过将基于文本的推理与外部工具调用相结合来解决复杂任务。不幸的是，人工智能代理容易受到提示注入攻击，外部工具返回的数据劫持代理执行恶意任务。为了衡量AI代理的对抗健壮性，我们引入了AgentDojo，一个针对在不可信数据上执行工具的代理的评估框架。为了捕捉攻击和防御不断演变的本质，AgentDojo不是一个静态测试套件，而是一个可扩展的环境，用于设计和评估新的代理任务、防御和适应性攻击。我们在环境中填充了97项现实任务(例如，管理电子邮件客户端、浏览电子银行网站或预订旅行)、629个安全测试用例以及文献中的各种攻击和防御范例。我们发现AgentDojo对攻击和防御都构成了挑战：最先进的LLM在许多任务中失败(即使在没有攻击的情况下也是如此)，并且现有的即时注入攻击破坏了一些安全属性，但不是全部。我们希望AgentDojo能够促进对AI代理新设计原则的研究，以可靠和健壮的方式解决常见任务。我们在https://github.com/ethz-spylab/agentdojo.上发布了AgentDojo的代码



## **32. AmpleGCG: Learning a Universal and Transferable Generative Model of Adversarial Suffixes for Jailbreaking Both Open and Closed LLMs**

AmpleGCG：学习通用且可转移的对抗性后缀生成模型，用于越狱开放和封闭LLM cs.CL

Published as a conference paper at COLM 2024  (https://colmweb.org/index.html)

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.07921v3) [paper-pdf](http://arxiv.org/pdf/2404.07921v3)

**Authors**: Zeyi Liao, Huan Sun

**Abstract**: As large language models (LLMs) become increasingly prevalent and integrated into autonomous systems, ensuring their safety is imperative. Despite significant strides toward safety alignment, recent work GCG~\citep{zou2023universal} proposes a discrete token optimization algorithm and selects the single suffix with the lowest loss to successfully jailbreak aligned LLMs. In this work, we first discuss the drawbacks of solely picking the suffix with the lowest loss during GCG optimization for jailbreaking and uncover the missed successful suffixes during the intermediate steps. Moreover, we utilize those successful suffixes as training data to learn a generative model, named AmpleGCG, which captures the distribution of adversarial suffixes given a harmful query and enables the rapid generation of hundreds of suffixes for any harmful queries in seconds. AmpleGCG achieves near 100\% attack success rate (ASR) on two aligned LLMs (Llama-2-7B-chat and Vicuna-7B), surpassing two strongest attack baselines. More interestingly, AmpleGCG also transfers seamlessly to attack different models, including closed-source LLMs, achieving a 99\% ASR on the latest GPT-3.5. To summarize, our work amplifies the impact of GCG by training a generative model of adversarial suffixes that is universal to any harmful queries and transferable from attacking open-source LLMs to closed-source LLMs. In addition, it can generate 200 adversarial suffixes for one harmful query in only 4 seconds, rendering it more challenging to defend.

摘要: 随着大型语言模型(LLM)变得越来越普遍并集成到自治系统中，确保它们的安全性是当务之急。尽管在安全对齐方面取得了长足的进步，但最近的工作GCG~\Citep{zou2023Universal}提出了一种离散令牌优化算法，并选择损失最小的单个后缀来成功越狱对齐LLM。在这项工作中，我们首先讨论了在GCG优化越狱过程中只选择损失最小的后缀的缺点，并在中间步骤中发现了遗漏的成功后缀。此外，我们利用这些成功的后缀作为训练数据来学习一种名为AmpleGCG的生成模型，该模型捕获给定有害查询的对抗性后缀的分布，并在几秒钟内为任何有害查询快速生成数百个后缀。AmpleGCG在两个对齐的LLM(Llama-2-7B-Chat和Vicuna-7B)上达到了近100%的攻击成功率(ASR)，超过了两个最强的攻击基线。更有趣的是，AmpleGCG还无缝传输以攻击不同的型号，包括闭源LLMS，在最新的GPT-3.5上实现了99\%的ASR。总之，我们的工作通过训练对抗性后缀的生成模型来放大GCG的影响，该模型对任何有害的查询都是通用的，并且可以从攻击开源LLM转移到闭源LLM。此外，它可以在短短4秒内为一个有害的查询生成200个敌意后缀，使其更具挑战性。



## **33. A Framework for Differential Privacy Against Timing Attacks**

针对时间攻击的差异隐私框架 cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2409.05623v2) [paper-pdf](http://arxiv.org/pdf/2409.05623v2)

**Authors**: Zachary Ratliff, Salil Vadhan

**Abstract**: The standard definition of differential privacy (DP) ensures that a mechanism's output distribution on adjacent datasets is indistinguishable. However, real-world implementations of DP can, and often do, reveal information through their runtime distributions, making them susceptible to timing attacks. In this work, we establish a general framework for ensuring differential privacy in the presence of timing side channels. We define a new notion of timing privacy, which captures programs that remain differentially private to an adversary that observes the program's runtime in addition to the output. Our framework enables chaining together component programs that are timing-stable followed by a random delay to obtain DP programs that achieve timing privacy. Importantly, our definitions allow for measuring timing privacy and output privacy using different privacy measures. We illustrate how to instantiate our framework by giving programs for standard DP computations in the RAM and Word RAM models of computation. Furthermore, we show how our framework can be realized in code through a natural extension of the OpenDP Programming Framework.

摘要: 差分隐私(DP)的标准定义确保了机制在相邻数据集上的输出分布是不可区分的。然而，DP的真实实现可以而且经常通过它们的运行时分发泄露信息，从而使它们容易受到计时攻击。在这项工作中，我们建立了一个通用的框架，以确保在存在定时侧信道的情况下的差异隐私。我们定义了一个新的时间隐私的概念，它捕获了对对手保持不同隐私的程序，除了输出之外，还观察程序的运行时。我们的框架允许将定时稳定的组件程序链接在一起，然后是随机延迟，以获得实现定时隐私的DP程序。重要的是，我们的定义允许使用不同的隐私度量来测量定时隐私和输出隐私。我们通过给出计算的RAM和Word RAM模型中的标准DP计算程序来说明如何实例化我们的框架。此外，我们还展示了如何通过OpenDP编程框架的自然扩展在代码中实现我们的框架。



## **34. A Tunable Despeckling Neural Network Stabilized via Diffusion Equation**

通过扩散方程稳定的可调谐降斑神经网络 cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15921v1) [paper-pdf](http://arxiv.org/pdf/2411.15921v1)

**Authors**: Yi Ran, Zhichang Guo, Jia Li, Yao Li, Martin Burger, Boying Wu

**Abstract**: Multiplicative Gamma noise remove is a critical research area in the application of synthetic aperture radar (SAR) imaging, where neural networks serve as a potent tool. However, real-world data often diverges from theoretical models, exhibiting various disturbances, which makes the neural network less effective. Adversarial attacks work by finding perturbations that significantly disrupt functionality of neural networks, as the inherent instability of neural networks makes them highly susceptible. A network designed to withstand such extreme cases can more effectively mitigate general disturbances in real SAR data. In this work, the dissipative nature of diffusion equations is employed to underpin a novel approach for countering adversarial attacks and improve the resistance of real noise disturbance. We propose a tunable, regularized neural network that unrolls a denoising unit and a regularization unit into a single network for end-to-end training. In the network, the denoising unit and the regularization unit are composed of the denoising network and the simplest linear diffusion equation respectively. The regularization unit enhances network stability, allowing post-training time step adjustments to effectively mitigate the adverse impacts of adversarial attacks. The stability and convergence of our model are theoretically proven, and in the experiments, we compare our model with several state-of-the-art denoising methods on simulated images, adversarial samples, and real SAR images, yielding superior results in both quantitative and visual evaluations.

摘要: 乘性伽马噪声去除是合成孔径雷达(SAR)成像应用中的一个关键研究领域，而神经网络是其中一个强有力的工具。然而，现实世界的数据经常与理论模型背道而驰，表现出各种干扰，这使得神经网络的效率较低。对抗性攻击的工作原理是找到显著扰乱神经网络功能的扰动，因为神经网络的固有不稳定性使它们非常容易受到影响。一个能够承受这种极端情况的网络可以更有效地缓解真实SAR数据中的一般干扰。在这项工作中，扩散方程的耗散性质被用来支持一种新的方法来对抗对手攻击，并提高对真实噪声干扰的抵抗力。我们提出了一种可调的正则化神经网络，它将去噪单元和正则化单元展开为单个网络，用于端到端的训练。在该网络中，去噪单元和正则化单元分别由去噪网络和最简线性扩散方程组成。正规化股增强了网络稳定性，允许在训练后调整时间步长，以有效减轻对抗性攻击的不利影响。从理论上证明了该模型的稳定性和收敛性，并在实验中将该模型与几种最新的去噪方法在模拟图像、对抗性样本和真实SAR图像上进行了比较，在定量和视觉评价方面都取得了较好的结果。



## **35. ExAL: An Exploration Enhanced Adversarial Learning Algorithm**

ExAL：一种探索增强的对抗学习算法 cs.LG

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15878v1) [paper-pdf](http://arxiv.org/pdf/2411.15878v1)

**Authors**: A Vinil, Aneesh Sreevallabh Chivukula, Pranav Chintareddy

**Abstract**: Adversarial learning is critical for enhancing model robustness, aiming to defend against adversarial attacks that jeopardize machine learning systems. Traditional methods often lack efficient mechanisms to explore diverse adversarial perturbations, leading to limited model resilience. Inspired by game-theoretic principles, where adversarial dynamics are analyzed through frameworks like Nash equilibrium, exploration mechanisms in such setups allow for the discovery of diverse strategies, enhancing system robustness. However, existing adversarial learning methods often fail to incorporate structured exploration effectively, reducing their ability to improve model defense comprehensively. To address these challenges, we propose a novel Exploration-enhanced Adversarial Learning Algorithm (ExAL), leveraging the Exponentially Weighted Momentum Particle Swarm Optimizer (EMPSO) to generate optimized adversarial perturbations. ExAL integrates exploration-driven mechanisms to discover perturbations that maximize impact on the model's decision boundary while preserving structural coherence in the data. We evaluate the performance of ExAL on the MNIST Handwritten Digits and Blended Malware datasets. Experimental results demonstrate that ExAL significantly enhances model resilience to adversarial attacks by improving robustness through adversarial learning.

摘要: 对抗性学习是增强模型稳健性的关键，旨在防御危及机器学习系统的对抗性攻击。传统的方法往往缺乏有效的机制来探索不同的对抗性扰动，导致模型的弹性有限。在博弈论原理的启发下，通过像纳什均衡这样的框架来分析对手的动态，这种设置中的探索机制允许发现不同的策略，增强了系统的健壮性。然而，现有的对抗性学习方法往往不能有效地结合结构化探索，从而降低了它们全面提高模型防御的能力。为了应对这些挑战，我们提出了一种新的探索增强的对抗性学习算法(ExAL)，该算法利用指数加权动量粒子群优化算法(EMPSO)来生成优化的对抗性扰动。ExAL集成了探索驱动的机制，以发现最大限度地影响模型决策边界的扰动，同时保持数据的结构一致性。我们评估了ExAL在MNIST手写数字和混合恶意软件数据集上的性能。实验结果表明，ExAL通过对抗性学习提高了模型对对抗性攻击的稳健性，从而显著提高了模型对对抗性攻击的恢复能力。



## **36. Data Lineage Inference: Uncovering Privacy Vulnerabilities of Dataset Pruning**

数据谱系推理：揭露数据集修剪的隐私漏洞 cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15796v1) [paper-pdf](http://arxiv.org/pdf/2411.15796v1)

**Authors**: Qi Li, Cheng-Long Wang, Yinzhi Cao, Di Wang

**Abstract**: In this work, we systematically explore the data privacy issues of dataset pruning in machine learning systems. Our findings reveal, for the first time, that even if data in the redundant set is solely used before model training, its pruning-phase membership status can still be detected through attacks. Since this is a fully upstream process before model training, traditional model output-based privacy inference methods are completely unsuitable. To address this, we introduce a new task called Data-Centric Membership Inference and propose the first ever data-centric privacy inference paradigm named Data Lineage Inference (DaLI). Under this paradigm, four threshold-based attacks are proposed, named WhoDis, CumDis, ArraDis and SpiDis. We show that even without access to downstream models, adversaries can accurately identify the redundant set with only limited prior knowledge. Furthermore, we find that different pruning methods involve varying levels of privacy leakage, and even the same pruning method can present different privacy risks at different pruning fractions. We conducted an in-depth analysis of these phenomena and introduced a metric called the Brimming score to offer guidance for selecting pruning methods with privacy protection in mind.

摘要: 在这项工作中，我们系统地研究了机器学习系统中数据集剪枝的数据隐私问题。我们的发现首次表明，即使在模型训练之前只使用冗余集合中的数据，其剪枝阶段的成员状态仍然可以通过攻击检测到。由于这是模型训练前的一个完全上游过程，传统的基于模型输出的隐私推理方法完全不适用。为了解决这个问题，我们引入了一个新的任务--以数据为中心的成员关系推理，并提出了第一个以数据为中心的隐私推理范例--数据谱系推理(DALI)。在这一范式下，提出了四种基于门限的攻击方法，分别为：Who Dis、CumDis、ArraDis和SpiDis。我们表明，即使没有下游模型的访问，攻击者也可以利用有限的先验知识准确地识别冗余集。此外，我们发现，不同的剪枝方法涉及不同程度的隐私泄露，即使是相同的剪枝方法，也会在不同的剪枝部分呈现不同的隐私风险。我们对这些现象进行了深入的分析，并引入了一种名为Inmming Score的指标，以提供在考虑隐私保护的情况下选择修剪方法的指导。



## **37. JailBreakV: A Benchmark for Assessing the Robustness of MultiModal Large Language Models against Jailbreak Attacks**

JailBreakV：评估多模式大型语言模型针对越狱攻击的稳健性的基准 cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2404.03027v4) [paper-pdf](http://arxiv.org/pdf/2404.03027v4)

**Authors**: Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

**Abstract**: With the rapid advancements in Multimodal Large Language Models (MLLMs), securing these models against malicious inputs while aligning them with human values has emerged as a critical challenge. In this paper, we investigate an important and unexplored question of whether techniques that successfully jailbreak Large Language Models (LLMs) can be equally effective in jailbreaking MLLMs. To explore this issue, we introduce JailBreakV-28K, a pioneering benchmark designed to assess the transferability of LLM jailbreak techniques to MLLMs, thereby evaluating the robustness of MLLMs against diverse jailbreak attacks. Utilizing a dataset of 2, 000 malicious queries that is also proposed in this paper, we generate 20, 000 text-based jailbreak prompts using advanced jailbreak attacks on LLMs, alongside 8, 000 image-based jailbreak inputs from recent MLLMs jailbreak attacks, our comprehensive dataset includes 28, 000 test cases across a spectrum of adversarial scenarios. Our evaluation of 10 open-source MLLMs reveals a notably high Attack Success Rate (ASR) for attacks transferred from LLMs, highlighting a critical vulnerability in MLLMs that stems from their text-processing capabilities. Our findings underscore the urgent need for future research to address alignment vulnerabilities in MLLMs from both textual and visual inputs.

摘要: 随着多模式大型语言模型(MLLMS)的快速发展，保护这些模型不受恶意输入的影响，同时使它们与人类的价值观保持一致，已经成为一项关键的挑战。在本文中，我们研究了一个重要而未被探索的问题，即成功越狱大语言模型(LLMS)的技术是否可以在越狱MLLM中同样有效。为了探讨这一问题，我们引入了JailBreakV-28K，这是一个开创性的基准测试，旨在评估LLM越狱技术到MLLM的可转移性，从而评估MLLMS对各种越狱攻击的健壮性。利用本文提出的包含2,000个恶意查询的数据集，我们使用针对LLMS的高级越狱攻击生成了20,000个基于文本的越狱提示，以及来自最近MLLMS越狱攻击的8,000个基于图像的越狱输入，我们的综合数据集包括来自各种对抗场景的28,000个测试用例。我们对10个开源MLLMS的评估显示，对于从LLMS转移的攻击，攻击成功率(ASR)非常高，这突显了MLLMS中源于其文本处理能力的一个严重漏洞。我们的发现强调了未来研究的迫切需要，以解决MLLMS中从文本和视觉输入的对齐漏洞。



## **38. Chain of Attack: On the Robustness of Vision-Language Models Against Transfer-Based Adversarial Attacks**

攻击链：视觉语言模型对抗基于传输的对抗性攻击的鲁棒性 cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.15720v1) [paper-pdf](http://arxiv.org/pdf/2411.15720v1)

**Authors**: Peng Xie, Yequan Bie, Jianda Mao, Yangqiu Song, Yang Wang, Hao Chen, Kani Chen

**Abstract**: Pre-trained vision-language models (VLMs) have showcased remarkable performance in image and natural language understanding, such as image captioning and response generation. As the practical applications of vision-language models become increasingly widespread, their potential safety and robustness issues raise concerns that adversaries may evade the system and cause these models to generate toxic content through malicious attacks. Therefore, evaluating the robustness of open-source VLMs against adversarial attacks has garnered growing attention, with transfer-based attacks as a representative black-box attacking strategy. However, most existing transfer-based attacks neglect the importance of the semantic correlations between vision and text modalities, leading to sub-optimal adversarial example generation and attack performance. To address this issue, we present Chain of Attack (CoA), which iteratively enhances the generation of adversarial examples based on the multi-modal semantic update using a series of intermediate attacking steps, achieving superior adversarial transferability and efficiency. A unified attack success rate computing method is further proposed for automatic evasion evaluation. Extensive experiments conducted under the most realistic and high-stakes scenario, demonstrate that our attacking strategy can effectively mislead models to generate targeted responses using only black-box attacks without any knowledge of the victim models. The comprehensive robustness evaluation in our paper provides insight into the vulnerabilities of VLMs and offers a reference for the safety considerations of future model developments.

摘要: 预先训练的视觉语言模型(VLM)在图像和自然语言理解方面表现出显著的性能，例如图像字幕和响应生成。随着视觉语言模型的实际应用日益广泛，其潜在的安全性和健壮性问题引发了人们的担忧，即攻击者可能会逃避系统，并通过恶意攻击导致这些模型生成有毒内容。因此，评估开源VLMS对敌意攻击的健壮性受到了越来越多的关注，其中基于传输的攻击是一种典型的黑盒攻击策略。然而，现有的大多数基于迁移的攻击忽略了视觉和文本模态之间的语义关联的重要性，导致了次优的对抗性实例生成和攻击性能。为了解决这一问题，我们提出了攻击链(CoA)，它通过一系列中间攻击步骤，迭代地增强基于多模式语义更新的对抗实例的生成，实现了优越的对抗可转移性和效率。在此基础上，提出了一种统一的攻击成功率计算方法，用于自动规避评估。在最真实和高风险的场景下进行的大量实验表明，我们的攻击策略可以有效地误导模型，使其在不了解受害者模型的情况下仅使用黑盒攻击来生成有针对性的响应。本文的综合健壮性评估为深入了解VLMS的脆弱性提供了依据，并为未来模型开发的安全考虑提供了参考。



## **39. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

通过虚构面部身份数据集对视觉语言模型取消学习进行基准测试 cs.CV

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2411.03554v2) [paper-pdf](http://arxiv.org/pdf/2411.03554v2)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.

摘要: 机器遗忘已经成为一种遗忘训练数据中特定信息的有效策略。然而，随着视觉数据的日益集成，视觉语言模型(VLM)中的隐私问题仍然没有得到充分的研究。为了解决这个问题，我们引入了面部身份遗忘基准(FIUB边)，这是一个新的VLM遗忘基准，设计用于在被遗忘的权利设置下稳健地评估遗忘算法的有效性。具体地说，我们通过构建虚拟面部身份VQA数据集来制定VLM遗忘任务，并应用旨在精确控制信息源及其暴露水平的两阶段评估管道。在评估方面，由于VLM支持多种形式的具有相同语义的问题，我们还提供了健壮的评估指标，包括成员关系推理攻击和精心设计的对抗性隐私攻击来评估算法的性能。通过对FIUBuch四种基线VLM遗忘算法的评估，我们发现所有方法的遗忘性能都是有限的，模型效用和遗忘质量之间存在着显著的权衡。此外，我们的发现还强调了隐私攻击对于稳健评估的重要性。我们希望FIUB边将推动在开发更有效的VLM遗忘算法方面取得进展。



## **40. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

游戏理论的内曼-皮尔森检测对抗战略规避 cs.CR

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2206.05276v4) [paper-pdf](http://arxiv.org/pdf/2206.05276v4)

**Authors**: Yinan Hu, Juntao Chen, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.

摘要: 网络系统的安全性在很大程度上取决于对敌方行为的识别和识别。传统的检测方法侧重于特定类别的攻击，已不适用于日益隐蔽和欺骗性的攻击，这些攻击旨在从战略上绕过检测。这项工作旨在开发一种整体理论来对抗这种规避攻击。基于Neyman-Pearson(NP)假设检验公式，我们重点扩展了一类基本的基于统计的检测方法。我们提出了博弈论框架来捕捉战略规避攻击者和规避感知NP检测器之间的冲突关系。通过分析攻击者和NP检测器的均衡行为，我们用均衡接收-操作-特征(EROC)曲线来表征它们的性能。我们证明了逃避感知NP检测器的性能优于被动NP检测器，前者可以针对攻击者的行为采取策略性行动，并根据收到的消息自适应地修改其决策规则。此外，我们将我们的框架扩展到顺序设置，在该设置中，用户发送相同分布的消息。我们通过一个异常检测的案例验证了分析结果。



## **41. Constructing Semantics-Aware Adversarial Examples with a Probabilistic Perspective**

从概率角度构建语义感知的对抗示例 stat.ML

21 pages, 9 figures

**SubmitDate**: 2024-11-24    [abs](http://arxiv.org/abs/2306.00353v3) [paper-pdf](http://arxiv.org/pdf/2306.00353v3)

**Authors**: Andi Zhang, Mingtian Zhang, Damon Wischik

**Abstract**: We propose a probabilistic perspective on adversarial examples, allowing us to embed subjective understanding of semantics as a distribution into the process of generating adversarial examples, in a principled manner. Despite significant pixel-level modifications compared to traditional adversarial attacks, our method preserves the overall semantics of the image, making the changes difficult for humans to detect. This extensive pixel-level modification enhances our method's ability to deceive classifiers designed to defend against adversarial attacks. Our empirical findings indicate that the proposed methods achieve higher success rates in circumventing adversarial defense mechanisms, while remaining difficult for human observers to detect.

摘要: 我们提出了对抗性示例的概率视角，使我们能够以原则性的方式将对语义的主观理解作为一种分布嵌入到生成对抗性示例的过程中。尽管与传统的对抗攻击相比进行了重大的像素级修改，但我们的方法保留了图像的整体语义，使人类难以检测到这些变化。这种广泛的像素级修改增强了我们的方法欺骗旨在防御对抗攻击的分类器的能力。我们的经验研究结果表明，所提出的方法在规避对抗性防御机制方面取得了更高的成功率，同时人类观察者仍然难以发现。



## **42. Enhancing the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation**

通过不同参数增强增强人脸识别对抗攻击的可转移性 cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15555v1) [paper-pdf](http://arxiv.org/pdf/2411.15555v1)

**Authors**: Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang

**Abstract**: Face Recognition (FR) models are vulnerable to adversarial examples that subtly manipulate benign face images, underscoring the urgent need to improve the transferability of adversarial attacks in order to expose the blind spots of these systems. Existing adversarial attack methods often overlook the potential benefits of augmenting the surrogate model with diverse initializations, which limits the transferability of the generated adversarial examples. To address this gap, we propose a novel method called Diverse Parameters Augmentation (DPA) attack method, which enhances surrogate models by incorporating diverse parameter initializations, resulting in a broader and more diverse set of surrogate models. Specifically, DPA consists of two key stages: Diverse Parameters Optimization (DPO) and Hard Model Aggregation (HMA). In the DPO stage, we initialize the parameters of the surrogate model using both pre-trained and random parameters. Subsequently, we save the models in the intermediate training process to obtain a diverse set of surrogate models. During the HMA stage, we enhance the feature maps of the diversified surrogate models by incorporating beneficial perturbations, thereby further improving the transferability. Experimental results demonstrate that our proposed attack method can effectively enhance the transferability of the crafted adversarial face examples.

摘要: 人脸识别(FR)模型很容易受到敌意例子的攻击，这些例子巧妙地操纵了良性的人脸图像，这突显了迫切需要提高对抗性攻击的可转移性，以暴露这些系统的盲点。现有的对抗性攻击方法往往忽略了用不同的初始化来扩充代理模型的潜在好处，这限制了生成的对抗性实例的可转移性。为了弥补这一缺陷，我们提出了一种新的方法，称为不同参数增强(DPA)攻击方法，它通过结合不同的参数初始化来增强代理模型，从而产生更广泛和更多样化的代理模型集。具体地说，DPA包括两个关键阶段：多参数优化(DPO)和硬模型聚合(HMA)。在DPO阶段，我们使用预先训练的参数和随机参数来初始化代理模型的参数。随后，我们在中间训练过程中保存模型，以获得多样化的代理模型集。在HMA阶段，我们通过加入有益的扰动来增强多样化代理模型的特征映射，从而进一步提高了可转移性。实验结果表明，本文提出的攻击方法可以有效地提高特制的对抗性人脸样本的可转移性。



## **43. Improving Transferable Targeted Attacks with Feature Tuning Mixup**

通过功能调整Mixup改进可转移有针对性的攻击 cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15553v1) [paper-pdf](http://arxiv.org/pdf/2411.15553v1)

**Authors**: Kaisheng Liang, Xuelong Dai, Yanjie Li, Dong Wang, Bin Xiao

**Abstract**: Deep neural networks exhibit vulnerability to adversarial examples that can transfer across different models. A particularly challenging problem is developing transferable targeted attacks that can mislead models into predicting specific target classes. While various methods have been proposed to enhance attack transferability, they often incur substantial computational costs while yielding limited improvements. Recent clean feature mixup methods use random clean features to perturb the feature space but lack optimization for disrupting adversarial examples, overlooking the advantages of attack-specific perturbations. In this paper, we propose Feature Tuning Mixup (FTM), a novel method that enhances targeted attack transferability by combining both random and optimized noises in the feature space. FTM introduces learnable feature perturbations and employs an efficient stochastic update strategy for optimization. These learnable perturbations facilitate the generation of more robust adversarial examples with improved transferability. We further demonstrate that attack performance can be enhanced through an ensemble of multiple FTM-perturbed surrogate models. Extensive experiments on the ImageNet-compatible dataset across various models demonstrate that our method achieves significant improvements over state-of-the-art methods while maintaining low computational cost.

摘要: 深度神经网络对可以跨不同模型传输的对抗性示例表现出脆弱性。一个特别具有挑战性的问题是开发可转移的有针对性的攻击，这些攻击可能会误导模型预测特定的目标类别。虽然已经提出了各种方法来增强攻击的可转移性，但它们往往会产生大量的计算成本，同时产生的改进有限。目前的清洁特征混合方法使用随机的清洁特征来扰动特征空间，但缺乏对破坏敌意示例的优化，忽略了攻击特定扰动的优势。在本文中，我们提出了一种新的方法--特征调优混合(FTM)，它通过在特征空间中结合随机和优化噪声来增强目标攻击的可转移性。FTM引入了可学习的特征扰动，并采用了一种有效的随机更新策略进行优化。这些可学习的扰动有助于生成更健壮的对抗性例子，具有更好的可转移性。我们进一步证明，攻击性能可以通过多个FTM扰动代理模型的集成来增强。在不同模型的ImageNet兼容数据集上的大量实验表明，我们的方法在保持较低计算成本的同时，比最先进的方法取得了显著的改进。



## **44. MUNBa: Machine Unlearning via Nash Bargaining**

MUNba：通过纳什讨价还价的机器学习 cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.15537v1) [paper-pdf](http://arxiv.org/pdf/2411.15537v1)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto front, effectively avoiding the gradient conflicts. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving superior performance on several benchmarks. For example, in the challenging scenario of sample-wise forgetting, our algorithm approaches the gold standard retrain baseline. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.

摘要: 机器遗忘旨在选择性地消除模型中的有害行为，同时保持模型的整体效用。作为一个多任务学习问题，MU涉及到平衡与忘记特定概念/数据相关的目标和保持总体性能。这些遗忘和保留目标的天真集成可能会导致梯度冲突，阻碍MU算法获得最优解。为了解决梯度冲突问题，我们将MU重新描述为一个两人合作博弈，其中两个参与者，即遗忘者和保留者，通过他们的梯度方案做出贡献，以最大化他们的整体收益。为此，受纳什讨价还价理论的启发，我们推导出了一个闭合形式的解，将模型引导到帕累托前沿，有效地避免了梯度冲突。我们的MU公式保证了一个均衡的解决方案，其中任何与最终状态的偏离都将导致两个球员的总体目标的减少，确保每个目标的最优化。我们评估了我们的算法在不同的任务集上的有效性，包括图像分类和图像生成。在ResNet、视觉语言模型CLIP和文本到图像扩散模型上的大量实验表明，我们的方法的性能优于最先进的MU算法，在几个基准测试上都取得了优异的性能。例如，在具有挑战性的样本遗忘场景中，我们的算法接近黄金标准的重新训练基线。我们的结果还强调了在忘记精确度、保持泛化和对对手攻击的健壮性方面的改进。



## **45. Harnessing LLM to Attack LLM-Guarded Text-to-Image Models**

利用LLM攻击LLM保护的文本到图像模型 cs.AI

10 pages, 7 figures, under review

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2312.07130v4) [paper-pdf](http://arxiv.org/pdf/2312.07130v4)

**Authors**: Yimo Deng, Huangxun Chen

**Abstract**: To prevent Text-to-Image (T2I) models from generating unethical images, people deploy safety filters to block inappropriate drawing prompts. Previous works have employed token replacement to search adversarial prompts that attempt to bypass these filters, but they have become ineffective as nonsensical tokens fail semantic logic checks. In this paper, we approach adversarial prompts from a different perspective. We demonstrate that rephrasing a drawing intent into multiple benign descriptions of individual visual components can obtain an effective adversarial prompt. We propose a LLM-piloted multi-agent method named DACA to automatically complete intended rephrasing. Our method successfully bypasses the safety filters of DALL-E 3 and Midjourney to generate the intended images, achieving success rates of up to 76.7% and 64% in the one-time attack, and 98% and 84% in the re-use attack, respectively. We open-source our code and dataset on [this link](https://github.com/researchcode003/DACA).

摘要: 为了防止文本到图像(T2I)模型生成不道德的图像，人们部署了安全过滤器来阻止不适当的绘图提示。以前的工作使用令牌替换来搜索试图绕过这些过滤器的敌意提示，但由于无意义的令牌未通过语义逻辑检查，这些提示变得无效。在这篇文章中，我们从一个不同的角度来研究对抗性提示。我们证明，将一个绘图意图重新表述为对单个视觉组件的多个良性描述可以获得有效的对抗性提示。我们提出了一种LLM引导的多智能体方法DACA来自动完成意图重述。我们的方法成功地绕过了DAL-E 3和中途的安全过滤器，生成了预期的图像，在一次性攻击和重用攻击中分别达到了76.7%和%的成功率和98%和84%的成功率。我们在[This link](https://github.com/researchcode003/DACA).]上开源了我们的代码和数据集



## **46. Steering Away from Harm: An Adaptive Approach to Defending Vision Language Model Against Jailbreaks**

远离伤害：保护视觉语言模型免受越狱的自适应方法 cs.CV

**SubmitDate**: 2024-11-23    [abs](http://arxiv.org/abs/2411.16721v1) [paper-pdf](http://arxiv.org/pdf/2411.16721v1)

**Authors**: Han Wang, Gang Wang, Huan Zhang

**Abstract**: Vision Language Models (VLMs) can produce unintended and harmful content when exposed to adversarial attacks, particularly because their vision capabilities create new vulnerabilities. Existing defenses, such as input preprocessing, adversarial training, and response evaluation-based methods, are often impractical for real-world deployment due to their high costs. To address this challenge, we propose ASTRA, an efficient and effective defense by adaptively steering models away from adversarial feature directions to resist VLM attacks. Our key procedures involve finding transferable steering vectors representing the direction of harmful response and applying adaptive activation steering to remove these directions at inference time. To create effective steering vectors, we randomly ablate the visual tokens from the adversarial images and identify those most strongly associated with jailbreaks. These tokens are then used to construct steering vectors. During inference, we perform the adaptive steering method that involves the projection between the steering vectors and calibrated activation, resulting in little performance drops on benign inputs while strongly avoiding harmful outputs under adversarial inputs. Extensive experiments across multiple models and baselines demonstrate our state-of-the-art performance and high efficiency in mitigating jailbreak risks. Additionally, ASTRA exhibits good transferability, defending against both unseen attacks at design time (i.e., structured-based attacks) and adversarial images from diverse distributions.

摘要: 视觉语言模型(VLM)在受到敌意攻击时可能会产生意想不到的有害内容，特别是因为它们的视觉能力会产生新的漏洞。现有的防御措施，如输入预处理、对抗性训练和基于响应评估的方法，由于成本较高，对于现实世界的部署往往是不切实际的。为了应对这一挑战，我们提出了ASTRA，这是一种通过自适应地引导模型远离敌对特征方向来抵御VLM攻击的高效和有效的防御方法。我们的关键步骤包括找到代表有害反应方向的可转移导向向量，并在推理时应用自适应激活导向来移除这些方向。为了创建有效的引导向量，我们随机地从敌对图像中去除视觉标记，并识别那些与越狱最相关的视觉标记。然后，这些令牌被用来构造导向矢量。在推理过程中，我们执行了自适应转向方法，该方法涉及引导向量和校准激活之间的投影，使得良性输入的性能下降很小，而在敌对输入下，我们强烈避免了有害输出。在多个型号和基线上的广泛实验证明，我们在降低越狱风险方面具有最先进的性能和高效率。此外，ASTRA表现出良好的可转移性，既可以防御设计时看不见的攻击(即基于结构化的攻击)，也可以防御来自不同分发版本的敌意图像。



## **47. Scalable and Optimal Security Allocation in Networks against Stealthy Injection Attacks**

针对隐形注入攻击的网络中可扩展且最优的安全分配 eess.SY

8 pages, 5 figures, journal submission

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15319v1) [paper-pdf](http://arxiv.org/pdf/2411.15319v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper addresses the security allocation problem in a networked control system under stealthy injection attacks. The networked system is comprised of interconnected subsystems which are represented by nodes in a digraph. An adversary compromises the system by injecting false data into several nodes with the aim of maximally disrupting the performance of the network while remaining stealthy to a defender. To minimize the impact of such stealthy attacks, the defender, with limited knowledge about attack policies and attack resources, allocates several sensors on nodes to impose the stealthiness constraint governing the attack policy. We provide an optimal security allocation algorithm to minimize the expected attack impact on the entire network. Furthermore, under a suitable local control design, the proposed security allocation algorithm can be executed in a scalable way. Finally, the obtained results are validated through several numerical examples.

摘要: 本文研究了隐形注入攻击下网络控制系统的安全分配问题。网络化系统由互连的子系统组成，这些子系统由有向图中的节点表示。对手通过将虚假数据注入多个节点来损害系统，目的是最大限度地破坏网络的性能，同时对防御者保持秘密。为了最大限度地减少此类隐形攻击的影响，防御者对攻击策略和攻击资源的了解有限，会在节点上分配多个传感器来施加管理攻击策略的隐形约束。我们提供了最佳的安全分配算法，以最大限度地减少预期攻击对整个网络的影响。此外，在合适的本地控制设计下，所提出的安全分配算法可以以可扩展的方式执行。最后，通过几个算例验证了所得结果。



## **48. UnMarker: A Universal Attack on Defensive Image Watermarking**

UnMarker：对防御性图像水印的普遍攻击 cs.CR

To appear at IEEE S&P 2025

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2405.08363v2) [paper-pdf](http://arxiv.org/pdf/2405.08363v2)

**Authors**: Andre Kassis, Urs Hengartner

**Abstract**: Reports regarding the misuse of Generative AI (GenAI) to create deepfakes are frequent. Defensive watermarking enables GenAI providers to hide fingerprints in their images and use them later for deepfake detection. Yet, its potential has not been fully explored. We present UnMarker -- the first practical universal attack on defensive watermarking. Unlike existing attacks, UnMarker requires no detector feedback, no unrealistic knowledge of the watermarking scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, UnMarker employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against SOTA schemes prove UnMarker's effectiveness. It not only defeats traditional schemes while retaining superior quality compared to existing attacks but also breaks semantic watermarks that alter an image's structure, reducing the best detection rate to $43\%$ and rendering them useless. To our knowledge, UnMarker is the first practical attack on semantic watermarks, which have been deemed the future of defensive watermarking. Our findings show that defensive watermarking is not a viable defense against deepfakes, and we urge the community to explore alternatives.

摘要: 关于滥用生成性人工智能(GenAI)来创建深度假冒的报道频繁。防御性水印使GenAI提供商能够在他们的图像中隐藏指纹，并在以后用于深度假冒检测。然而，它的潜力还没有得到充分的开发。我们提出了UnMarker--第一个实用的对防御性水印的通用攻击。与现有的攻击不同，UnMarker不需要检测器反馈，不需要不切实际的水印方案或类似模型的知识，也不需要可能无法获得的高级去噪管道。相反，UnMarker是对水印范例的深入分析的产物，揭示了健壮的方案必须在频谱幅度上构造水印，UnMarker采用了两种新的对抗性优化来扰乱水印图像的频谱，从而消除水印。对SOTA计划的评估证明了UnMarker的有效性。它不仅在保持现有攻击质量的同时击败了传统方案，而且还打破了改变图像结构的语义水印，使最佳检测率降至43美元，使它们变得毫无用处。据我们所知，UnMarker是对语义水印的第一次实用攻击，语义水印被认为是防御性水印的未来。我们的发现表明，防御性水印不是一种可行的防御深度假冒的方法，我们敦促社区探索替代方案。



## **49. Benchmarking the Robustness of Optical Flow Estimation to Corruptions**

对光流量估计对腐蚀的鲁棒性进行基准测试 eess.IV

The benchmarks and source code will be released at  https://github.com/ZhonghuaYi/optical_flow_robustness_benchmark

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.14865v1) [paper-pdf](http://arxiv.org/pdf/2411.14865v1)

**Authors**: Zhonghua Yi, Hao Shi, Qi Jiang, Yao Gao, Ze Wang, Yufan Zhang, Kailun Yang, Kaiwei Wang

**Abstract**: Optical flow estimation is extensively used in autonomous driving and video editing. While existing models demonstrate state-of-the-art performance across various benchmarks, the robustness of these methods has been infrequently investigated. Despite some research focusing on the robustness of optical flow models against adversarial attacks, there has been a lack of studies investigating their robustness to common corruptions. Taking into account the unique temporal characteristics of optical flow, we introduce 7 temporal corruptions specifically designed for benchmarking the robustness of optical flow models, in addition to 17 classical single-image corruptions, in which advanced PSF Blur simulation method is performed. Two robustness benchmarks, KITTI-FC and GoPro-FC, are subsequently established as the first corruption robustness benchmark for optical flow estimation, with Out-Of-Domain (OOD) and In-Domain (ID) settings to facilitate comprehensive studies. Robustness metrics, Corruption Robustness Error (CRE), Corruption Robustness Error ratio (CREr), and Relative Corruption Robustness Error (RCRE) are further introduced to quantify the optical flow estimation robustness. 29 model variants from 15 optical flow methods are evaluated, yielding 10 intriguing observations, such as 1) the absolute robustness of the model is heavily dependent on the estimation performance; 2) the corruptions that diminish local information are more serious than that reduce visual effects. We also give suggestions for the design and application of optical flow models. We anticipate that our benchmark will serve as a foundational resource for advancing research in robust optical flow estimation. The benchmarks and source code will be released at https://github.com/ZhonghuaYi/optical_flow_robustness_benchmark.

摘要: 光流估计在自动驾驶和视频编辑中有着广泛的应用。虽然现有的模型在各种基准测试中展示了最先进的性能，但这些方法的稳健性很少被研究。尽管有一些研究集中在光流模型对敌意攻击的稳健性上，但缺乏对其对常见腐败的稳健性的研究。考虑到光流独特的时间特性，我们引入了7个专门为检验光流模型的稳健性而设计的时间腐败，以及17个经典的单幅图像腐败，其中使用了先进的PSF模糊仿真方法。随后建立了两个稳健性基准，Kitti-FC和GoPro-FC，作为光流估计的第一个腐败稳健性基准，具有域外(OOD)和域内(ID)设置，以促进全面研究。进一步引入稳健性度量、腐败稳健误差(CRE)、腐败稳健误差率(CRER)和相对腐败稳健误差(RCRE)来量化光流估计的稳健性。对15种光流方法的29个模型变量进行了评估，得到了10个有趣的观察结果，如1)模型的绝对稳健性在很大程度上取决于估计性能；2)减少局部信息的破坏比减少视觉效果的破坏更严重。并对光流模型的设计和应用提出了建议。我们预计，我们的基准将成为推进稳健光流估计研究的基础资源。基准测试和源代码将在https://github.com/ZhonghuaYi/optical_flow_robustness_benchmark.上发布



## **50. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

统一XAI的无导扩散总管约束梯度 cs.CV

19 pages, 5 figures

**SubmitDate**: 2024-11-22    [abs](http://arxiv.org/abs/2411.15265v1) [paper-pdf](http://arxiv.org/pdf/2411.15265v1)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.

摘要: 基于梯度的方法是一类典型的可解释技术，特别是对于基于图像的模型。尽管如此，它们有几个缺点：(1)需要对模型进行白盒访问，(2)容易受到对手攻击，(3)产生偏离图像多维的属性，导致解释实际上不忠于模型，也不能很好地与人类感知保持一致。为了克服这些挑战，我们引入了无导数扩散流形约束梯度法(FreeMCG)，这是一种新的方法，它比传统的梯度法更好地改善了给定神经网络的解释能力。具体地说，通过利用集合卡尔曼滤波和扩散模型，我们得到了投影到数据流形上的模型梯度的无导数近似，只需要访问模型的输出。我们通过将其应用于反事实生成和特征属性这两个传统上被视为不同任务的任务来展示其有效性。通过对两个任务、反事实解释和特征属性的综合评估，我们表明我们的方法在保持XAI工具预期的本质属性的同时产生了最先进的结果。



