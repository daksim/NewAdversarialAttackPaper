# Latest Adversarial Attack Papers
**update at 2024-07-06 15:45:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Correlated Privacy Mechanisms for Differentially Private Distributed Mean Estimation**

用于差异私有分布均值估计的相关隐私机制 cs.IT

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03289v1) [paper-pdf](http://arxiv.org/pdf/2407.03289v1)

**Authors**: Sajani Vithana, Viveck R. Cadambe, Flavio P. Calmon, Haewon Jeong

**Abstract**: Differentially private distributed mean estimation (DP-DME) is a fundamental building block in privacy-preserving federated learning, where a central server estimates the mean of $d$-dimensional vectors held by $n$ users while ensuring $(\epsilon,\delta)$-DP. Local differential privacy (LDP) and distributed DP with secure aggregation (SecAgg) are the most common notions of DP used in DP-DME settings with an untrusted server. LDP provides strong resilience to dropouts, colluding users, and malicious server attacks, but suffers from poor utility. In contrast, SecAgg-based DP-DME achieves an $O(n)$ utility gain over LDP in DME, but requires increased communication and computation overheads and complex multi-round protocols to handle dropouts and malicious attacks. In this work, we propose CorDP-DME, a novel DP-DME mechanism that spans the gap between DME with LDP and distributed DP, offering a favorable balance between utility and resilience to dropout and collusion. CorDP-DME is based on correlated Gaussian noise, ensuring DP without the perfect conditional privacy guarantees of SecAgg-based approaches. We provide an information-theoretic analysis of CorDP-DME, and derive theoretical guarantees for utility under any given privacy parameters and dropout/colluding user thresholds. Our results demonstrate that (anti) correlated Gaussian DP mechanisms can significantly improve utility in mean estimation tasks compared to LDP -- even in adversarial settings -- while maintaining better resilience to dropouts and attacks compared to distributed DP.

摘要: 差分私有分布平均估计(DP-DME)是保护隐私的联合学习的基本构件，其中中央服务器估计$n$用户所持有的$d$维向量的平均值，同时确保$(？，？)$-DP。本地差异隐私(LDP)和带安全聚合的分布式DP(SecAgg)是DP-DME设置中使用不受信任服务器的最常见概念。LDP对辍学、串通用户和恶意服务器攻击具有很强的弹性，但实用性较差。相比之下，基于SecAgg的DP-DME在DME中比LDP获得$O(N)$效用收益，但需要更多的通信和计算开销以及复杂的多轮协议来处理丢弃和恶意攻击。在这项工作中，我们提出了一种新的DP-DME机制CorDP-DME，它跨越了DME与LDP和分布式DP之间的差距，在实用性和抗丢弃和共谋能力之间提供了良好的平衡。CorDP-DME基于相关的高斯噪声，在没有基于SecAgg的方法的完美条件隐私保证的情况下确保DP。我们对CorDP-DME进行了信息论分析，并推导出在任何给定的隐私参数和丢弃/合谋用户阈值下的效用的理论保证。我们的结果表明，与LDP相比，(反)相关的高斯DP机制可以显著提高均值估计任务的实用性--即使在对抗环境中--同时与分布式DP相比，保持更好的对丢弃和攻击的弹性。



## **2. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

自我评估作为对LLM的对抗攻击的防御 cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03234v1) [paper-pdf](http://arxiv.org/pdf/2407.03234v1)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/Adversarial-Alignments.

摘要: 当LLM部署在敏感的、面向人的环境中时，至关重要的是它们不输出不安全、有偏见或违反隐私的输出。出于这个原因，模特们既接受了培训，又被指示拒绝回答不安全的提示，比如“告诉我如何制造炸弹。”我们发现，尽管有这些保障措施，但只需在模型输入的末尾添加一个空格，就可以打破模型的防御。在对八个开源模型的研究中，我们证明了这是一种足够强大的攻击，足以导致大多数模型产生非常高的成功率的有害输出。我们研究了这种行为的原因，发现在标记化的训练数据中出现单个空格的上下文鼓励模型在得到提示时生成列表，从而覆盖拒绝回答不安全请求的训练信号。我们的发现强调了当前模型比对的脆弱状态，并促进了开发更稳健的比对方法的重要性。代码和数据将在https://github.com/Linlt-leon/Adversarial-Alignments.上提供



## **3. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

毒液杀手：联邦学习中的不可感知和按需定向后门攻击 cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03144v1) [paper-pdf](http://arxiv.org/pdf/2407.03144v1)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, and Multi-Krum. The source code is available at https://anonymous.4open.science/r/Venomancer-3426.

摘要: 联合学习(FL)是一种分布式机器学习方法，通过对分散的数据源进行训练来维护数据隐私。与集中式机器学习类似，FL也容易受到后门攻击。FL中的大多数后门攻击假设一个预定义的目标类，并需要控制大量客户端或了解良性客户端的信息。此外，由于毒物数据上留下了明显的伪影，它们并不是不可察觉的，并且很容易被人类检查发现。为了克服这些挑战，我们提出了毒液杀手，这是一种有效的后门攻击，可以潜移默化，并允许按需锁定目标。具体地说，不可感知性是通过使用视觉损失函数来实现的，以使有毒数据在视觉上与原始数据不可区分。按需目标属性允许攻击者通过有条件的对抗性训练选择任意目标类。此外，实验表明，该方法对诸如Norm裁剪、弱DP、Krum和多Krum等最先进的防御方法具有很强的鲁棒性。源代码可在https://anonymous.4open.science/r/Venomancer-3426.上找到



## **4. $L_p$-norm Distortion-Efficient Adversarial Attack**

$L_p$-规范失真高效对抗攻击 cs.CV

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03115v1) [paper-pdf](http://arxiv.org/pdf/2407.03115v1)

**Authors**: Chao Zhou, Yuan-Gen Wang, Zi-jia Wang, Xiangui Kang

**Abstract**: Adversarial examples have shown a powerful ability to make a well-trained model misclassified. Current mainstream adversarial attack methods only consider one of the distortions among $L_0$-norm, $L_2$-norm, and $L_\infty$-norm. $L_0$-norm based methods cause large modification on a single pixel, resulting in naked-eye visible detection, while $L_2$-norm and $L_\infty$-norm based methods suffer from weak robustness against adversarial defense since they always diffuse tiny perturbations to all pixels. A more realistic adversarial perturbation should be sparse and imperceptible. In this paper, we propose a novel $L_p$-norm distortion-efficient adversarial attack, which not only owns the least $L_2$-norm loss but also significantly reduces the $L_0$-norm distortion. To this aim, we design a new optimization scheme, which first optimizes an initial adversarial perturbation under $L_2$-norm constraint, and then constructs a dimension unimportance matrix for the initial perturbation. Such a dimension unimportance matrix can indicate the adversarial unimportance of each dimension of the initial perturbation. Furthermore, we introduce a new concept of adversarial threshold for the dimension unimportance matrix. The dimensions of the initial perturbation whose unimportance is higher than the threshold will be all set to zero, greatly decreasing the $L_0$-norm distortion. Experimental results on three benchmark datasets show that under the same query budget, the adversarial examples generated by our method have lower $L_0$-norm and $L_2$-norm distortion than the state-of-the-art. Especially for the MNIST dataset, our attack reduces 8.1$\%$ $L_2$-norm distortion meanwhile remaining 47$\%$ pixels unattacked. This demonstrates the superiority of the proposed method over its competitors in terms of adversarial robustness and visual imperceptibility.

摘要: 对抗性的例子已经显示出强大的能力，可以让训练有素的模型被错误分类。目前主流的对抗性攻击方法只考虑了$L_0$-范数、$L_2$-范数和$L_\INFTY$-范数之间的一种扭曲。基于$L_0$-范数的方法会对单个像素进行较大的修改，从而导致肉眼可见检测，而基于$L_2$-范数和$L_\inty$-范数的方法由于总是将微小的扰动扩散到所有像素，对敌意防御的鲁棒性较弱。更现实的对抗性扰动应该是稀疏和不可察觉的。在本文中，我们提出了一种新颖的$L_p$范数失真高效的对抗性攻击，它不仅具有最小的$L_2$范数损失，而且显著降低了$L_0$范数失真。为此，我们设计了一种新的优化方案，它首先在$L_2$-范数约束下优化初始对抗性扰动，然后构造初始扰动的维无关性矩阵。这样的维度无关性矩阵可以指示初始扰动的每个维度的对抗性无关性。此外，我们还引入了维度不重要矩阵的对抗性阈值的新概念。不重要程度高于阈值的初始扰动的维度都将被设置为零，从而大大降低了$L_0$范数的失真。在三个基准数据集上的实验结果表明，在相同的查询预算下，该方法生成的对抗性实例具有较低的$L_0$范数和$L_2$范数失真。特别是对于MNIST数据集，我们的攻击减少了8.1%$\$$L_2$-范数失真，同时保持了47$\$像素未被攻击。这表明了该方法在对抗鲁棒性和视觉不可感知性方面优于其竞争对手。



## **5. JailbreakHunter: A Visual Analytics Approach for Jailbreak Prompts Discovery from Large-Scale Human-LLM Conversational Datasets**

越狱猎人：越狱的视觉分析方法从大规模人类LLM对话数据集中进行发现 cs.HC

18 pages, 9 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.03045v1) [paper-pdf](http://arxiv.org/pdf/2407.03045v1)

**Authors**: Zhihua Jin, Shiyi Liu, Haotian Li, Xun Zhao, Huamin Qu

**Abstract**: Large Language Models (LLMs) have gained significant attention but also raised concerns due to the risk of misuse. Jailbreak prompts, a popular type of adversarial attack towards LLMs, have appeared and constantly evolved to breach the safety protocols of LLMs. To address this issue, LLMs are regularly updated with safety patches based on reported jailbreak prompts. However, malicious users often keep their successful jailbreak prompts private to exploit LLMs. To uncover these private jailbreak prompts, extensive analysis of large-scale conversational datasets is necessary to identify prompts that still manage to bypass the system's defenses. This task is highly challenging due to the immense volume of conversation data, diverse characteristics of jailbreak prompts, and their presence in complex multi-turn conversations. To tackle these challenges, we introduce JailbreakHunter, a visual analytics approach for identifying jailbreak prompts in large-scale human-LLM conversational datasets. We have designed a workflow with three analysis levels: group-level, conversation-level, and turn-level. Group-level analysis enables users to grasp the distribution of conversations and identify suspicious conversations using multiple criteria, such as similarity with reported jailbreak prompts in previous research and attack success rates. Conversation-level analysis facilitates the understanding of the progress of conversations and helps discover jailbreak prompts within their conversation contexts. Turn-level analysis allows users to explore the semantic similarity and token overlap between a singleturn prompt and the reported jailbreak prompts, aiding in the identification of new jailbreak strategies. The effectiveness and usability of the system were verified through multiple case studies and expert interviews.

摘要: 大型语言模型(LLM)获得了极大的关注，但也因误用的风险而引起了关注。越狱提示是一种流行的针对LLMS的对抗性攻击类型，已经出现并不断演变为违反LLMS的安全协议。为了解决这个问题，LLM会根据报告的越狱提示定期更新安全补丁。然而，恶意用户通常会将他们成功的越狱提示保密，以利用LLMS。为了发现这些私密的越狱提示，有必要对大规模对话数据集进行广泛分析，以确定仍然设法绕过系统防御的提示。由于对话数据量巨大，越狱提示的特点多种多样，而且它们存在于复杂的多轮对话中，这项任务具有极大的挑战性。为了应对这些挑战，我们引入了JailBreakHunter，这是一种视觉分析方法，用于在大规模的人-LLM对话数据集中识别越狱提示。我们设计了一个具有三个分析级别的工作流：小组级别、会话级别和话轮级别。组级分析使用户能够掌握对话的分布，并使用多种标准识别可疑对话，例如与之前研究中报告的越狱提示相似，以及攻击成功率。会话级别的分析有助于了解会话的进度，并帮助发现会话上下文中的越狱提示。话轮水平分析允许用户探索单一URN提示和报告的越狱提示之间的语义相似性和标记重叠，有助于识别新的越狱策略。通过多个案例研究和专家访谈，验证了该系统的有效性和可用性。



## **6. Expressivity of Graph Neural Networks Through the Lens of Adversarial Robustness**

从对抗鲁棒性角度看图神经网络的表现性 cs.LG

Published in ${2}^{nd}$ AdvML Frontiers workshop at ${40}^{th}$  International Conference on Machine Learning (ICML)

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2308.08173v2) [paper-pdf](http://arxiv.org/pdf/2308.08173v2)

**Authors**: Francesco Campi, Lukas Gosch, Tom Wollschläger, Yan Scholten, Stephan Günnemann

**Abstract**: We perform the first adversarial robustness study into Graph Neural Networks (GNNs) that are provably more powerful than traditional Message Passing Neural Networks (MPNNs). In particular, we use adversarial robustness as a tool to uncover a significant gap between their theoretically possible and empirically achieved expressive power. To do so, we focus on the ability of GNNs to count specific subgraph patterns, which is an established measure of expressivity, and extend the concept of adversarial robustness to this task. Based on this, we develop efficient adversarial attacks for subgraph counting and show that more powerful GNNs fail to generalize even to small perturbations to the graph's structure. Expanding on this, we show that such architectures also fail to count substructures on out-of-distribution graphs.

摘要: 我们对图形神经网络（GNN）进行了首次对抗鲁棒性研究，该网络被证明比传统的消息传递神经网络（MPNN）更强大。特别是，我们使用对抗鲁棒性作为工具来揭示它们理论上可能的表达能力和经验上实现的表达能力之间的显着差距。为此，我们重点关注GNN计算特定子图模式的能力（这是表达力的既定指标），并将对抗稳健性的概念扩展到这项任务。基于此，我们开发了用于子图计数的高效对抗攻击，并表明更强大的GNN即使是对图结构的微小扰动也无法推广。在此基础上进一步扩展，我们表明此类架构也无法计算出分布外图上的子结构。



## **7. A Wolf in Sheep's Clothing: Practical Black-box Adversarial Attacks for Evading Learning-based Windows Malware Detection in the Wild**

披着羊皮的狼：实用的黑匣子对抗攻击，用于逃避野外基于学习的Windows恶意软件检测 cs.CR

This paper has been accepted by 33rd USENIX Security Symposium 2024

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2407.02886v1) [paper-pdf](http://arxiv.org/pdf/2407.02886v1)

**Authors**: Xiang Ling, Zhiyu Wu, Bin Wang, Wei Deng, Jingzheng Wu, Shouling Ji, Tianyue Luo, Yanjun Wu

**Abstract**: Given the remarkable achievements of existing learning-based malware detection in both academia and industry, this paper presents MalGuise, a practical black-box adversarial attack framework that evaluates the security risks of existing learning-based Windows malware detection systems under the black-box setting. MalGuise first employs a novel semantics-preserving transformation of call-based redividing to concurrently manipulate both nodes and edges of malware's control-flow graph, making it less noticeable. By employing a Monte-Carlo-tree-search-based optimization, MalGuise then searches for an optimized sequence of call-based redividing transformations to apply to the input Windows malware for evasions. Finally, it reconstructs the adversarial malware file based on the optimized transformation sequence while adhering to Windows executable format constraints, thereby maintaining the same semantics as the original. MalGuise is systematically evaluated against three state-of-the-art learning-based Windows malware detection systems under the black-box setting. Evaluation results demonstrate that MalGuise achieves a remarkably high attack success rate, mostly exceeding 95%, with over 91% of the generated adversarial malware files maintaining the same semantics. Furthermore, MalGuise achieves up to a 74.97% attack success rate against five anti-virus products, highlighting potential tangible security concerns to real-world users.

摘要: 鉴于现有的基于学习的恶意软件检测在学术界和工业界都取得了显著的成果，本文提出了一个实用的黑盒对抗攻击框架MalGuise，该框架评估了现有基于学习的Windows恶意软件检测系统在黑盒环境下的安全风险。MalGuise首先采用了一种新颖的基于调用的重新划分的语义保留变换来同时操作恶意软件控制流图的节点和边，从而使其不那么引人注目。通过采用基于蒙特卡洛树搜索的优化，MalGuise然后搜索基于调用的重新划分转换的优化序列，以应用于输入Windows恶意软件的规避。最后，在遵循Windows可执行文件格式约束的同时，根据优化后的转换顺序重构恶意软件文件，从而保持与原始文件相同的语义。在黑盒设置下，针对三个最先进的基于学习的Windows恶意软件检测系统对MalGuise进行了系统评估。评估结果表明，MalGuise的攻击成功率非常高，大多超过95%，生成的恶意软件文件中91%以上保持相同的语义。此外，MalGuise对五种反病毒产品的攻击成功率高达74.97%，向现实世界的用户突出了潜在的有形安全问题。



## **8. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

14 pages, 4 figures

**SubmitDate**: 2024-07-03    [abs](http://arxiv.org/abs/2406.19692v2) [paper-pdf](http://arxiv.org/pdf/2406.19692v2)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **9. Light-weight Fine-tuning Method for Defending Adversarial Noise in Pre-trained Medical Vision-Language Models**

在预训练的医学视觉语言模型中防御对抗性噪音的轻量级微调方法 cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02716v1) [paper-pdf](http://arxiv.org/pdf/2407.02716v1)

**Authors**: Xu Han, Linghao Jin, Xuezhe Ma, Xiaofeng Liu

**Abstract**: Fine-tuning pre-trained Vision-Language Models (VLMs) has shown remarkable capabilities in medical image and textual depiction synergy. Nevertheless, many pre-training datasets are restricted by patient privacy concerns, potentially containing noise that can adversely affect downstream performance. Moreover, the growing reliance on multi-modal generation exacerbates this issue because of its susceptibility to adversarial attacks. To investigate how VLMs trained on adversarial noisy data perform on downstream medical tasks, we first craft noisy upstream datasets using multi-modal adversarial attacks. Through our comprehensive analysis, we unveil that moderate noise enhances model robustness and transferability, but increasing noise levels negatively impact downstream task performance. To mitigate this issue, we propose rectify adversarial noise (RAN) framework, a recipe designed to effectively defend adversarial attacks and rectify the influence of upstream noise during fine-tuning.

摘要: 微调预训练的视觉语言模型（VLM）在医学图像和文本描述协同方面表现出了非凡的能力。然而，许多预训练数据集受到患者隐私问题的限制，可能包含可能对下游性能产生不利影响的噪音。此外，对多模式发电的日益依赖加剧了这个问题，因为它容易受到对抗攻击。为了研究在对抗性有噪数据上训练的VLM如何执行下游医疗任务，我们首先使用多模式对抗攻击来制作有噪的上游数据集。通过我们的全面分析，我们发现适度的噪音增强了模型的稳健性和可移植性，但增加噪音水平会对下游任务性能产生负面影响。为了缓解这个问题，我们提出了纠正对抗性噪音（RAN）框架，该框架旨在有效防御对抗性攻击并纠正微调期间上游噪音的影响。



## **10. Adversarial Magnification to Deceive Deepfake Detection through Super Resolution**

对抗放大通过超分辨率欺骗Deepfake检测 cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02670v1) [paper-pdf](http://arxiv.org/pdf/2407.02670v1)

**Authors**: Davide Alessandro Coccomini, Roberto Caldelli, Giuseppe Amato, Fabrizio Falchi, Claudio Gennaro

**Abstract**: Deepfake technology is rapidly advancing, posing significant challenges to the detection of manipulated media content. Parallel to that, some adversarial attack techniques have been developed to fool the deepfake detectors and make deepfakes even more difficult to be detected. This paper explores the application of super resolution techniques as a possible adversarial attack in deepfake detection. Through our experiments, we demonstrate that minimal changes made by these methods in the visual appearance of images can have a profound impact on the performance of deepfake detection systems. We propose a novel attack using super resolution as a quick, black-box and effective method to camouflage fake images and/or generate false alarms on pristine images. Our results indicate that the usage of super resolution can significantly impair the accuracy of deepfake detectors, thereby highlighting the vulnerability of such systems to adversarial attacks. The code to reproduce our experiments is available at: https://github.com/davide-coccomini/Adversarial-Magnification-to-Deceive-Deepfake-Detection-through-Super-Resolution

摘要: 深伪技术正在迅速发展，给检测被篡改的媒体内容带来了巨大的挑战。与此同时，一些对抗性攻击技术已经开发出来，以愚弄深度假冒检测器，使深度假冒更难被检测到。本文探讨了超分辨技术作为一种可能的对抗性攻击在深度伪检测中的应用。通过实验，我们证明了这些方法对图像视觉外观的微小改变可以对深度伪检测系统的性能产生深远的影响。我们提出了一种新的攻击方法，利用超分辨率作为一种快速、黑箱和有效的方法来伪装虚假图像和/或在原始图像上产生虚假警报。我们的结果表明，超分辨率的使用会显著降低深度伪检测器的准确性，从而突出了此类系统对对手攻击的脆弱性。重现我们实验的代码可在以下网址获得：https://github.com/davide-coccomini/Adversarial-Magnification-to-Deceive-Deepfake-Detection-through-Super-Resolution



## **11. Towards More Realistic Extraction Attacks: An Adversarial Perspective**

走向更真实的提取攻击：对抗的角度 cs.CR

To be presented at PrivateNLP@ACL2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02596v1) [paper-pdf](http://arxiv.org/pdf/2407.02596v1)

**Authors**: Yash More, Prakhar Ganesh, Golnoosh Farnadi

**Abstract**: Language models are prone to memorizing large parts of their training data, making them vulnerable to extraction attacks. Existing research on these attacks remains limited in scope, often studying isolated trends rather than the real-world interactions with these models. In this paper, we revisit extraction attacks from an adversarial perspective, exploiting the brittleness of language models. We find significant churn in extraction attack trends, i.e., even minor, unintuitive changes to the prompt, or targeting smaller models and older checkpoints, can exacerbate the risks of extraction by up to $2-4 \times$. Moreover, relying solely on the widely accepted verbatim match underestimates the extent of extracted information, and we provide various alternatives to more accurately capture the true risks of extraction. We conclude our discussion with data deduplication, a commonly suggested mitigation strategy, and find that while it addresses some memorization concerns, it remains vulnerable to the same escalation of extraction risks against a real-world adversary. Our findings highlight the necessity of acknowledging an adversary's true capabilities to avoid underestimating extraction risks.

摘要: 语言模型容易记住它们的大部分训练数据，这使得它们很容易受到提取攻击。现有对这些攻击的研究范围仍然有限，往往研究孤立的趋势，而不是与这些模型的现实世界互动。在本文中，我们从敌意的角度重新审视提取攻击，利用语言模型的脆弱性。我们发现提取攻击趋势中的显著波动，即即使对提示进行微小的、不直观的更改，或者针对较小的模型和较旧的检查点，都可能使提取的风险增加高达2-4倍$。此外，仅依靠被广泛接受的逐字匹配低估了提取信息的程度，我们提供了各种替代方案来更准确地捕获提取的真实风险。我们以重复数据删除结束我们的讨论，这是一种通常建议的缓解策略，并发现虽然它解决了一些记忆问题，但它仍然容易受到针对现实世界对手的提取风险的相同升级的影响。我们的发现突显了承认对手真实能力的必要性，以避免低估开采风险。



## **12. A False Sense of Safety: Unsafe Information Leakage in 'Safe' AI Responses**

错误的安全感：“安全”人工智能响应中不安全的信息泄露 cs.CR

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02551v1) [paper-pdf](http://arxiv.org/pdf/2407.02551v1)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreaks$\unicode{x2013}$methods to elicit harmful or generally impermissible outputs. Safety measures are developed and assessed on their effectiveness at defending against jailbreak attacks, indicating a belief that safety is equivalent to robustness. We assert that current defense mechanisms, such as output filters and alignment fine-tuning, are, and will remain, fundamentally insufficient for ensuring model safety. These defenses fail to address risks arising from dual-intent queries and the ability to composite innocuous outputs to achieve harmful goals. To address this critical gap, we introduce an information-theoretic threat model called inferential adversaries who exploit impermissible information leakage from model outputs to achieve malicious goals. We distinguish these from commonly studied security adversaries who only seek to force victim models to generate specific impermissible outputs. We demonstrate the feasibility of automating inferential adversaries through question decomposition and response aggregation. To provide safety guarantees, we define an information censorship criterion for censorship mechanisms, bounding the leakage of impermissible information. We propose a defense mechanism which ensures this bound and reveal an intrinsic safety-utility trade-off. Our work provides the first theoretically grounded understanding of the requirements for releasing safe LLMs and the utility costs involved.

摘要: 大型语言模型(LLM)容易受到越狱$\Unicode{x2013}$方法的攻击，从而导致有害或通常不允许的输出。制定了安全措施，并对其在防御越狱攻击方面的有效性进行了评估，表明了一种信念，即安全等同于健壮性。我们断言，目前的防御机制，如输出过滤器和对齐微调，对于确保模型安全来说，无论是现在还是将来，都是根本不够的。这些防御措施未能解决双重意图询问产生的风险，以及将无害的产出综合起来以实现有害目标的能力。为了解决这一关键差距，我们引入了一个名为推理对手的信息论威胁模型，该模型利用模型输出中不允许的信息泄漏来实现恶意目标。我们将这些区别于通常研究的安全对手，后者只寻求迫使受害者模型生成特定的不允许的输出。我们论证了通过问题分解和响应聚合自动化推理对手的可行性。为了提供安全保障，我们为审查机制定义了信息审查标准，限制了不允许信息的泄露。我们提出了一种防御机制，确保了这一界限，并揭示了内在的安全和效用之间的权衡。我们的工作首次提供了对发布安全LLM的要求和涉及的公用事业成本的理论上的理解。



## **13. Greedy-DiM: Greedy Algorithms for Unreasonably Effective Face Morphs**

贪婪-缩小：不合理有效的面部形态的贪婪算法 cs.CV

Accepted as a conference paper at IJCB 2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2404.06025v2) [paper-pdf](http://arxiv.org/pdf/2404.06025v2)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: Morphing attacks are an emerging threat to state-of-the-art Face Recognition (FR) systems, which aim to create a single image that contains the biometric information of multiple identities. Diffusion Morphs (DiM) are a recently proposed morphing attack that has achieved state-of-the-art performance for representation-based morphing attacks. However, none of the existing research on DiMs have leveraged the iterative nature of DiMs and left the DiM model as a black box, treating it no differently than one would a Generative Adversarial Network (GAN) or Varational AutoEncoder (VAE). We propose a greedy strategy on the iterative sampling process of DiM models which searches for an optimal step guided by an identity-based heuristic function. We compare our proposed algorithm against ten other state-of-the-art morphing algorithms using the open-source SYN-MAD 2022 competition dataset. We find that our proposed algorithm is unreasonably effective, fooling all of the tested FR systems with an MMPMR of 100%, outperforming all other morphing algorithms compared.

摘要: 变形攻击是对最先进的人脸识别(FR)系统的新威胁，该系统旨在创建包含多个身份的生物识别信息的单一图像。扩散变形(Dim)是最近提出的一种变形攻击，它已经在基于表示的变形攻击中获得了最先进的性能。然而，现有的关于DIMS的研究都没有利用DIMS的迭代性质，将DIM模型视为一个黑盒，将其视为与生成性对抗性网络(GAN)或变分自动编码器(VAE)没有区别的模型。针对DIM模型的迭代采样过程，我们提出了一种贪婪策略，在基于身份的启发式函数的指导下寻找最优步长。我们使用开源的SYN-MAD 2022竞赛数据集将我们提出的算法与其他十种最先进的变形算法进行了比较。我们发现我们提出的算法是不合理的有效的，愚弄了所有测试的FR系统，MMPMR为100%，比所有其他变形算法都要好。



## **14. Steerable Pyramid Transform Enables Robust Left Ventricle Quantification**

可操纵金字塔变换实现稳健的左锥体量化 eess.IV

Code is available at https://github.com/yangyangyang127/RobustLV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2201.08388v2) [paper-pdf](http://arxiv.org/pdf/2201.08388v2)

**Authors**: Xiangyang Zhu, Kede Ma, Wufeng Xue

**Abstract**: Predicting cardiac indices has long been a focal point in the medical imaging community. While various deep learning models have demonstrated success in quantifying cardiac indices, they remain susceptible to mild input perturbations, e.g., spatial transformations, image distortions, and adversarial attacks. This vulnerability undermines confidence in using learning-based automated systems for diagnosing cardiovascular diseases. In this work, we describe a simple yet effective method to learn robust models for left ventricle (LV) quantification, encompassing cavity and myocardium areas, directional dimensions, and regional wall thicknesses. Our success hinges on employing the biologically inspired steerable pyramid transform (SPT) for fixed front-end processing, which offers three main benefits. First, the basis functions of SPT align with the anatomical structure of LV and the geometric features of the measured indices. Second, SPT facilitates weight sharing across different orientations as a form of parameter regularization and naturally captures the scale variations of LV. Third, the residual highpass subband can be conveniently discarded, promoting robust feature learning. Extensive experiments on the Cardiac-Dig benchmark show that our SPT-augmented model not only achieves reasonable prediction accuracy compared to state-of-the-art methods, but also exhibits significantly improved robustness against input perturbations.

摘要: 心脏指数的预测长期以来一直是医学影像学界关注的焦点。虽然各种深度学习模型在量化心脏指数方面取得了成功，但它们仍然容易受到轻微输入扰动的影响，例如空间变换、图像失真和对抗性攻击。这一脆弱性削弱了人们对使用基于学习的自动化系统诊断心血管疾病的信心。在这项工作中，我们描述了一种简单而有效的方法来学习用于左心室(LV)量化的稳健模型，包括腔和心肌面积、方向尺寸和局部室壁厚度。我们的成功取决于使用生物启发的可引导金字塔变换(SPT)进行固定的前端处理，这提供了三个主要好处。首先，SPT的基本功能与LV的解剖结构和测量指标的几何特征相一致。其次，作为参数正则化的一种形式，SPT促进了不同方向上的权重分担，并自然地捕捉到了LV的尺度变化。第三，可以方便地丢弃剩余的高通子带，促进稳健的特征学习。在心脏挖掘基准上的大量实验表明，与最先进的方法相比，我们的SPT增强模型不仅获得了合理的预测精度，而且对输入扰动具有显著改善的鲁棒性。



## **15. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

EvolBA：硬标签黑匣子条件下的进化边界攻击 cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02248v1) [paper-pdf](http://arxiv.org/pdf/2407.02248v1)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.

摘要: 研究表明，深度神经网络(DNN)存在漏洞，可能会导致对经过特殊设计的扰动的对抗性示例(AE)的错误识别。针对硬标签黑盒(HL-BB)环境下不存在损失梯度和置信度的漏洞检测问题，提出了多种对抗性攻击方法，但这些方法只搜索搜索空间的局部区域，容易陷入局部解.因此，本文提出了一种基于协方差矩阵自适应进化策略(CMA-ES)的对抗性攻击方法EvolBA，用于在目标DNN模型预测的类别标签不可用的HL-BB条件下生成AEs。受公式驱动的监督学习的启发，该方法在初始化过程中引入了领域无关的算子，并引入了一个跳跃来增强搜索探索。实验结果表明，该方法能够以较小的扰动确定图像中的声学效应，克服了以往方法的不足。



## **16. MALT Powers Up Adversarial Attacks**

MALT增强对抗性攻击 cs.LG

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02240v1) [paper-pdf](http://arxiv.org/pdf/2407.02240v1)

**Authors**: Odelia Melamed, Gilad Yehudai, Adi Shamir

**Abstract**: Current adversarial attacks for multi-class classifiers choose the target class for a given input naively, based on the classifier's confidence levels for various target classes. We present a novel adversarial targeting method, \textit{MALT - Mesoscopic Almost Linearity Targeting}, based on medium-scale almost linearity assumptions. Our attack wins over the current state of the art AutoAttack on the standard benchmark datasets CIFAR-100 and ImageNet and for a variety of robust models. In particular, our attack is \emph{five times faster} than AutoAttack, while successfully matching all of AutoAttack's successes and attacking additional samples that were previously out of reach. We then prove formally and demonstrate empirically that our targeting method, although inspired by linear predictors, also applies to standard non-linear models.

摘要: 当前针对多类分类器的对抗攻击天真地根据分类器对各种目标类的置信度水平为给定输入选择目标类。我们基于中等规模几乎线性假设，提出了一种新型的对抗性瞄准方法，\textit{MALT - Mesoscopic Almost Linearity Tagting}。我们的攻击战胜了对标准基准数据集CIFAR-100和ImageNet以及各种稳健模型的当前最先进的AutoAttack。特别是，我们的攻击比AutoAttack快\{五倍}，同时成功匹配了AutoAttack的所有成功并攻击了以前遥不可及的其他样本。然后，我们正式证明并以经验证明，我们的目标方法虽然受到线性预测器的启发，但也适用于标准非线性模型。



## **17. Secure Semantic Communication via Paired Adversarial Residual Networks**

通过配对对抗剩余网络的安全语义通信 cs.IT

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.02053v1) [paper-pdf](http://arxiv.org/pdf/2407.02053v1)

**Authors**: Boxiang He, Fanggang Wang, Tony Q. S. Quek

**Abstract**: This letter explores the positive side of the adversarial attack for the security-aware semantic communication system. Specifically, a pair of matching pluggable modules is installed: one after the semantic transmitter and the other before the semantic receiver. The module at transmitter uses a trainable adversarial residual network (ARN) to generate adversarial examples, while the module at receiver employs another trainable ARN to remove the adversarial attacks and the channel noise. To mitigate the threat of semantic eavesdropping, the trainable ARNs are jointly optimized to minimize the weighted sum of the power of adversarial attack, the mean squared error of semantic communication, and the confidence of eavesdropper correctly retrieving private information. Numerical results show that the proposed scheme is capable of fooling the eavesdropper while maintaining the high-quality semantic communication.

摘要: 这封信探讨了对抗性攻击对安全感知语义通信系统的积极一面。具体来说，安装了一对匹配的可插入模块：一个在语义发送器之后，另一个在语义接收器之前。发射机处的模块使用可训练的对抗剩余网络（ARN）来生成对抗示例，而接收机处的模块使用另一个可训练的ARN来消除对抗攻击和通道噪音。为了减轻语义窃听的威胁，对可训练的ARN进行了联合优化，以最小化对抗攻击的力量、语义通信的均方误差以及窃听者正确检索私人信息的信心的加权和。数值结果表明，该方案能够欺骗窃听者，同时保持高质量的语义通信。



## **18. TTSlow: Slow Down Text-to-Speech with Efficiency Robustness Evaluations**

TTSlow：通过效率稳健性评估减缓文本转语音 eess.AS

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01927v1) [paper-pdf](http://arxiv.org/pdf/2407.01927v1)

**Authors**: Xiaoxue Gao, Yiming Chen, Xianghu Yue, Yu Tsao, Nancy F. Chen

**Abstract**: Text-to-speech (TTS) has been extensively studied for generating high-quality speech with textual inputs, playing a crucial role in various real-time applications. For real-world deployment, ensuring stable and timely generation in TTS models against minor input perturbations is of paramount importance. Therefore, evaluating the robustness of TTS models against such perturbations, commonly known as adversarial attacks, is highly desirable. In this paper, we propose TTSlow, a novel adversarial approach specifically tailored to slow down the speech generation process in TTS systems. To induce long TTS waiting time, we design novel efficiency-oriented adversarial loss to encourage endless generation process. TTSlow encompasses two attack strategies targeting both text inputs and speaker embedding. Specifically, we propose TTSlow-text, which utilizes a combination of homoglyphs-based and swap-based perturbations, along with TTSlow-spk, which employs a gradient optimization attack approach for speaker embedding. TTSlow serves as the first attack approach targeting a wide range of TTS models, including autoregressive and non-autoregressive TTS ones, thereby advancing exploration in audio security. Extensive experiments are conducted to evaluate the inference efficiency of TTS models, and in-depth analysis of generated speech intelligibility is performed using Gemini. The results demonstrate that TTSlow can effectively slow down two TTS models across three publicly available datasets. We are committed to releasing the source code upon acceptance, facilitating further research and benchmarking in this domain.

摘要: 文本到语音(TTS，Text-to-Speech)技术在各种实时应用中扮演着至关重要的角色，能够生成高质量的文本输入语音。对于现实世界的部署，确保在TTS模型中针对微小的输入扰动稳定和及时地生成是至关重要的。因此，评估TTS模型对此类扰动(通常称为对抗性攻击)的稳健性是非常必要的。在本文中，我们提出了TTSlow，这是一种新的对抗性方法，专门用于减缓TTS系统中的语音生成过程。为了延长TTS的等待时间，我们设计了一种新颖的效率导向的对抗性损失来鼓励无休止的生成过程。TTSlow包含两种针对文本输入和说话人嵌入的攻击策略。具体地说，我们提出了TTSlow-Text和TTSlow-Spk，TTSlow-Text结合了基于同形文字和基于交换的扰动，TTSlow-spk使用了梯度优化攻击方法来嵌入说话人。TTSlow是第一个针对多种TTS模型的攻击方法，包括自回归和非自回归TTS模型，从而推动了音频安全方面的探索。通过大量的实验来评估TTS模型的推理效率，并使用Gemini对生成的语音清晰度进行了深入的分析。结果表明，TTSlow可以有效地降低三个公开可用的数据集上的两个TTS模型的速度。我们致力于在接受后发布源代码，促进该领域的进一步研究和基准测试。



## **19. Looking From the Future: Multi-order Iterations Can Enhance Adversarial Attack Transferability**

展望未来：多阶迭代可以增强对抗性攻击的可转移性 cs.CV

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01925v1) [paper-pdf](http://arxiv.org/pdf/2407.01925v1)

**Authors**: Zijian Ying, Qianmu Li, Tao Wang, Zhichao Lian, Shunmei Meng, Xuyun Zhang

**Abstract**: Various methods try to enhance adversarial transferability by improving the generalization from different perspectives. In this paper, we rethink the optimization process and propose a novel sequence optimization concept, which is named Looking From the Future (LFF). LFF makes use of the original optimization process to refine the very first local optimization choice. Adapting the LFF concept to the adversarial attack task, we further propose an LFF attack as well as an MLFF attack with better generalization ability. Furthermore, guiding with the LFF concept, we propose an $LLF^{\mathcal{N}}$ attack which entends the LFF attack to a multi-order attack, further enhancing the transfer attack ability. All our proposed methods can be directly applied to the iteration-based attack methods. We evaluate our proposed method on the ImageNet1k dataset by applying several SOTA adversarial attack methods under four kinds of tasks. Experimental results show that our proposed method can greatly enhance the attack transferability. Ablation experiments are also applied to verify the effectiveness of each component. The source code will be released after this paper is accepted.

摘要: 不同的方法试图通过从不同的角度改进泛化来增强对手的可转移性。在本文中，我们对优化过程进行了重新思考，并提出了一种新的序列优化概念--展望未来(LFF)。LFF利用原有的优化过程来提炼出第一个局部优化选择。将LFF的概念应用到对抗性攻击任务中，提出了一种具有更好泛化能力的LFF攻击和MLFF攻击。此外，在LFF概念的指导下，我们提出了一种$LLF^{\Mathcal{N}}$攻击，将LLF攻击扩展为多阶攻击，进一步增强了传输攻击的能力。我们提出的所有方法都可以直接应用于基于迭代的攻击方法。我们在ImageNet1k数据集上应用了四种任务下的几种SOTA对抗性攻击方法对我们的方法进行了评估。实验结果表明，该方法可以大大提高攻击的可转移性。还进行了烧蚀实验，以验证各部件的有效性。源代码将在这篇论文被接受后发布。



## **20. A Method to Facilitate Membership Inference Attacks in Deep Learning Models**

一种促进深度学习模型中成员推断攻击的方法 cs.CR

NDSS'25 (a shorter version of this paper will appear in the  conference proceeding)

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01919v1) [paper-pdf](http://arxiv.org/pdf/2407.01919v1)

**Authors**: Zitao Chen, Karthik Pattabiraman

**Abstract**: Modern machine learning (ML) ecosystems offer a surging number of ML frameworks and code repositories that can greatly facilitate the development of ML models. Today, even ordinary data holders who are not ML experts can apply off-the-shelf codebase to build high-performance ML models on their data, many of which are sensitive in nature (e.g., clinical records).   In this work, we consider a malicious ML provider who supplies model-training code to the data holders, does not have access to the training process, and has only black-box query access to the resulting model. In this setting, we demonstrate a new form of membership inference attack that is strictly more powerful than prior art. Our attack empowers the adversary to reliably de-identify all the training samples (average >99% attack TPR@0.1% FPR), and the compromised models still maintain competitive performance as their uncorrupted counterparts (average <1% accuracy drop). Moreover, we show that the poisoned models can effectively disguise the amplified membership leakage under common membership privacy auditing, which can only be revealed by a set of secret samples known by the adversary.   Overall, our study not only points to the worst-case membership privacy leakage, but also unveils a common pitfall underlying existing privacy auditing methods, which calls for future efforts to rethink the current practice of auditing membership privacy in machine learning models.

摘要: 现代机器学习(ML)生态系统提供了数量激增的ML框架和代码库，可以极大地促进ML模型的开发。今天，即使不是ML专家的普通数据持有者也可以应用现成的代码库来针对他们的数据构建高性能的ML模型，其中许多数据本质上是敏感的(例如，临床记录)。在这项工作中，我们考虑一个恶意的ML提供者，他向数据持有者提供模型训练代码，无权访问训练过程，并且只能对结果模型进行黑盒查询访问。在这种情况下，我们演示了一种新形式的成员关系推理攻击，它比现有技术更强大。我们的攻击使对手能够可靠地识别所有训练样本(平均>99%的攻击TPR@0.1%的FPR)，而受损的模型仍然保持与未被破坏的同行的竞争性能(平均<1%的准确率下降)。此外，我们还证明了有毒模型能够有效地掩盖在普通成员身份隐私审计下被放大的成员身份泄露，而这种被放大的成员身份泄露只能通过一组被敌手知道的秘密样本来揭示。总体而言，我们的研究不仅指出了最坏情况下的成员隐私泄露，还揭示了现有隐私审计方法背后的一个常见陷阱，这要求未来努力重新思考当前在机器学习模型中审计成员隐私的做法。



## **21. Sequential Manipulation Against Rank Aggregation: Theory and Algorithm**

针对排名聚集的顺序操纵：理论与算法 cs.AI

Accepted by IEEE TPAMI URL:  https://ieeexplore.ieee.org/document/10564181

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2407.01916v1) [paper-pdf](http://arxiv.org/pdf/2407.01916v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Wei Liu, Xiaochun Cao, Yingfei Sun, Qingming Huang

**Abstract**: Rank aggregation with pairwise comparisons is widely encountered in sociology, politics, economics, psychology, sports, etc . Given the enormous social impact and the consequent incentives, the potential adversary has a strong motivation to manipulate the ranking list. However, the ideal attack opportunity and the excessive adversarial capability cause the existing methods to be impractical. To fully explore the potential risks, we leverage an online attack on the vulnerable data collection process. Since it is independent of rank aggregation and lacks effective protection mechanisms, we disrupt the data collection process by fabricating pairwise comparisons without knowledge of the future data or the true distribution. From the game-theoretic perspective, the confrontation scenario between the online manipulator and the ranker who takes control of the original data source is formulated as a distributionally robust game that deals with the uncertainty of knowledge. Then we demonstrate that the equilibrium in the above game is potentially favorable to the adversary by analyzing the vulnerability of the sampling algorithms such as Bernoulli and reservoir methods. According to the above theoretical analysis, different sequential manipulation policies are proposed under a Bayesian decision framework and a large class of parametric pairwise comparison models. For attackers with complete knowledge, we establish the asymptotic optimality of the proposed policies. To increase the success rate of the sequential manipulation with incomplete knowledge, a distributionally robust estimator, which replaces the maximum likelihood estimation in a saddle point problem, provides a conservative data generation solution. Finally, the corroborating empirical evidence shows that the proposed method manipulates the results of rank aggregation methods in a sequential manner.

摘要: 两两比较的秩聚合法在社会学、政治学、经济学、心理学、体育学等领域都有广泛的应用。考虑到巨大的社会影响和随之而来的激励，潜在的对手有强烈的动机操纵排行榜。然而，理想的进攻机会和过多的对抗能力导致现有的方法不切实际。为了充分探索潜在风险，我们利用对易受攻击的数据收集过程的在线攻击。由于它独立于等级聚集，缺乏有效的保护机制，我们在不知道未来数据或真实分布的情况下，通过伪造成对比较来扰乱数据收集过程。从博弈论的角度来看，在线操纵者和控制原始数据源的排名者之间的对抗场景被描述为一个处理知识不确定性的分布式稳健博弈。然后，通过分析伯努利和水库等抽样算法的脆弱性，证明了上述博弈中的均衡对对手是潜在有利的。基于上述理论分析，在贝叶斯决策框架和一大类参数两两比较模型下，提出了不同的序贯操作策略。对于具有完全知识的攻击者，我们建立了所提出策略的渐近最优性。为了提高不完全知识顺序操作的成功率，用分布稳健估计器代替鞍点问题中的最大似然估计，提供了一种保守的数据生成方案。最后，实证结果表明，该方法可以连续地操纵等级聚集方法的结果。



## **22. A Curious Case of Searching for the Correlation between Training Data and Adversarial Robustness of Transformer Textual Models**

寻找训练数据与Transformer文本模型对抗鲁棒性之间相关性的奇怪案例 cs.LG

Accepted to ACL Findings 2024

**SubmitDate**: 2024-07-02    [abs](http://arxiv.org/abs/2402.11469v2) [paper-pdf](http://arxiv.org/pdf/2402.11469v2)

**Authors**: Cuong Dang, Dung D. Le, Thai Le

**Abstract**: Existing works have shown that fine-tuned textual transformer models achieve state-of-the-art prediction performances but are also vulnerable to adversarial text perturbations. Traditional adversarial evaluation is often done \textit{only after} fine-tuning the models and ignoring the training data. In this paper, we want to prove that there is also a strong correlation between training data and model robustness. To this end, we extract 13 different features representing a wide range of input fine-tuning corpora properties and use them to predict the adversarial robustness of the fine-tuned models. Focusing mostly on encoder-only transformer models BERT and RoBERTa with additional results for BART, ELECTRA, and GPT2, we provide diverse evidence to support our argument. First, empirical analyses show that (a) extracted features can be used with a lightweight classifier such as Random Forest to predict the attack success rate effectively, and (b) features with the most influence on the model robustness have a clear correlation with the robustness. Second, our framework can be used as a fast and effective additional tool for robustness evaluation since it (a) saves 30x-193x runtime compared to the traditional technique, (b) is transferable across models, (c) can be used under adversarial training, and (d) robust to statistical randomness. Our code is publicly available at \url{https://github.com/CaptainCuong/RobustText_ACL2024}.

摘要: 已有的工作表明，微调的文本变换模型取得了最先进的预测性能，但也容易受到对抗性文本扰动的影响。传统的对抗性评估往往是在对模型进行微调而忽略训练数据之后才进行的。在本文中，我们想要证明训练数据和模型稳健性之间也存在很强的相关性。为此，我们提取了13个不同的特征，代表了广泛的输入微调语料库属性，并使用它们来预测微调模型的对抗性健壮性。我们主要关注仅编码器的转换器模型BART和Roberta，以及BART、ELECTRA和GPT2的其他结果，我们提供了各种证据来支持我们的论点。首先，实证分析表明：(A)提取的特征可以与随机森林等轻量级分类器一起有效地预测攻击成功率；(B)对模型稳健性影响最大的特征与模型的稳健性有明显的相关性。其次，我们的框架可以作为快速有效的额外工具用于健壮性评估，因为它(A)比传统技术节省30-193倍的运行时间，(B)可以跨模型转移，(C)可以在对抗性训练下使用，以及(D)对统计随机性具有健壮性。我们的代码在\url{https://github.com/CaptainCuong/RobustText_ACL2024}.上公开提供



## **23. Purple-teaming LLMs with Adversarial Defender Training**

紫色团队LLM与对抗性辩护人培训 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01850v1) [paper-pdf](http://arxiv.org/pdf/2407.01850v1)

**Authors**: Jingyan Zhou, Kun Li, Junan Li, Jiawen Kang, Minda Hu, Xixin Wu, Helen Meng

**Abstract**: Existing efforts in safeguarding LLMs are limited in actively exposing the vulnerabilities of the target LLM and readily adapting to newly emerging safety risks. To address this, we present Purple-teaming LLMs with Adversarial Defender training (PAD), a pipeline designed to safeguard LLMs by novelly incorporating the red-teaming (attack) and blue-teaming (safety training) techniques. In PAD, we automatically collect conversational data that cover the vulnerabilities of an LLM around specific safety risks in a self-play manner, where the attacker aims to elicit unsafe responses and the defender generates safe responses to these attacks. We then update both modules in a generative adversarial network style by training the attacker to elicit more unsafe responses and updating the defender to identify them and explain the unsafe reason. Experimental results demonstrate that PAD significantly outperforms existing baselines in both finding effective attacks and establishing a robust safe guardrail. Furthermore, our findings indicate that PAD excels in striking a balance between safety and overall model quality. We also reveal key challenges in safeguarding LLMs, including defending multi-turn attacks and the need for more delicate strategies to identify specific risks.

摘要: 现有的保护低土地管理系统的努力在主动暴露目标低土地管理系统的脆弱性和随时适应新出现的安全风险方面是有限的。为了解决这个问题，我们提出了带有对抗性防守训练(PAD)的紫色团队LLMS，这是一种旨在通过新颖地结合红色团队(攻击)和蓝色团队(安全训练)技术来保护LLMS的管道。在PAD中，我们以自我发挥的方式自动收集涵盖LLM围绕特定安全风险的漏洞的对话数据，其中攻击者旨在引发不安全的响应，而防御者生成对这些攻击的安全响应。然后，我们以生成式对抗性网络风格更新这两个模块，方法是训练攻击者获得更多不安全的响应，并更新防御者以识别它们并解释不安全的原因。实验结果表明，PAD在发现有效攻击和建立坚固的安全护栏方面都明显优于现有的基线。此外，我们的研究结果表明，PAD在安全性和整体模型质量之间取得平衡方面表现出色。我们还揭示了在保护LLM方面的关键挑战，包括防御多回合攻击，以及需要更微妙的战略来识别特定的风险。



## **24. Adversarial Attacks on Reinforcement Learning Agents for Command and Control**

对指挥与控制强化学习代理的对抗攻击 cs.CR

Accepted to appear in the Journal Of Defense Modeling and Simulation  (JDMS)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.01693v2) [paper-pdf](http://arxiv.org/pdf/2405.01693v2)

**Authors**: Ahaan Dabholkar, James Z. Hare, Mark Mittrick, John Richardson, Nicholas Waytowich, Priya Narayanan, Saurabh Bagchi

**Abstract**: Given the recent impact of Deep Reinforcement Learning in training agents to win complex games like StarCraft and DoTA(Defense Of The Ancients) - there has been a surge in research for exploiting learning based techniques for professional wargaming, battlefield simulation and modeling. Real time strategy games and simulators have become a valuable resource for operational planning and military research. However, recent work has shown that such learning based approaches are highly susceptible to adversarial perturbations. In this paper, we investigate the robustness of an agent trained for a Command and Control task in an environment that is controlled by an active adversary. The C2 agent is trained on custom StarCraft II maps using the state of the art RL algorithms - A3C and PPO. We empirically show that an agent trained using these algorithms is highly susceptible to noise injected by the adversary and investigate the effects these perturbations have on the performance of the trained agent. Our work highlights the urgent need to develop more robust training algorithms especially for critical arenas like the battlefield.

摘要: 鉴于最近深度强化学习在训练代理以赢得星际争霸和DOTA(古人防御)等复杂游戏中的影响，将基于学习的技术用于专业战争游戏、战场模拟和建模的研究激增。实时战略游戏和模拟器已经成为作战规划和军事研究的宝贵资源。然而，最近的工作表明，这种基于学习的方法非常容易受到对抗性扰动的影响。在本文中，我们研究了在由活跃的对手控制的环境中为指挥与控制任务训练的代理的稳健性。C2特工使用最先进的RL算法-A3C和PPO-在定制的星际争霸II地图上进行训练。我们的经验表明，使用这些算法训练的代理对对手注入的噪声非常敏感，并调查了这些扰动对训练的代理性能的影响。我们的工作突出了开发更健壮的训练算法的迫切需要，特别是对于战场这样的关键领域。



## **25. On the Abuse and Detection of Polyglot Files**

多语言文件的滥用与检测 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01529v1) [paper-pdf](http://arxiv.org/pdf/2407.01529v1)

**Authors**: Luke Koch, Sean Oesch, Amul Chaulagain, Jared Dixon, Matthew Dixon, Mike Huettal, Amir Sadovnik, Cory Watson, Brian Weber, Jacob Hartman, Richard Patulski

**Abstract**: A polyglot is a file that is valid in two or more formats. Polyglot files pose a problem for malware detection systems that route files to format-specific detectors/signatures, as well as file upload and sanitization tools. In this work we found that existing file-format and embedded-file detection tools, even those developed specifically for polyglot files, fail to reliably detect polyglot files used in the wild, leaving organizations vulnerable to attack. To address this issue, we studied the use of polyglot files by malicious actors in the wild, finding $30$ polyglot samples and $15$ attack chains that leveraged polyglot files. In this report, we highlight two well-known APTs whose cyber attack chains relied on polyglot files to bypass detection mechanisms. Using knowledge from our survey of polyglot usage in the wild -- the first of its kind -- we created a novel data set based on adversary techniques. We then trained a machine learning detection solution, PolyConv, using this data set. PolyConv achieves a precision-recall area-under-curve score of $0.999$ with an F1 score of $99.20$% for polyglot detection and $99.47$% for file-format identification, significantly outperforming all other tools tested. We developed a content disarmament and reconstruction tool, ImSan, that successfully sanitized $100$% of the tested image-based polyglots, which were the most common type found via the survey. Our work provides concrete tools and suggestions to enable defenders to better defend themselves against polyglot files, as well as directions for future work to create more robust file specifications and methods of disarmament.

摘要: 多语种是指以两种或多种格式有效的文件。多语言文件给恶意软件检测系统带来了问题，恶意软件检测系统将文件路由到特定格式的检测器/签名，以及文件上传和清理工具。在这项工作中，我们发现现有的文件格式和嵌入式文件检测工具，即使是那些专门为多语言文件开发的工具，也无法可靠地检测在野外使用的多语言文件，从而使组织容易受到攻击。为了解决这个问题，我们研究了恶意攻击者在野外使用多语言文件的情况，发现了$30$多语言样本和$15$利用多语言文件的攻击链。在这份报告中，我们重点介绍了两个著名的APT，它们的网络攻击链依赖于多语言文件来绕过检测机制。利用我们对野外多语种使用情况的调查--这是此类调查中的第一次--我们创建了一个基于对手技术的新数据集。然后，我们使用这个数据集训练了一个机器学习检测解决方案PolyConv。PolyConv获得了0.999美元的曲线下区域精度召回分数，而F1多语言检测的分数为99.20$%，文件格式识别的F1分数为99.47$%，大大超过了所有其他测试工具。我们开发了一个内容解除和重建工具ImSAN，它成功地清理了100美元%的基于图像的测试多语种，这是通过调查发现的最常见的类型。我们的工作提供了具体的工具和建议，使捍卫者能够更好地保护自己免受多国语言文件的伤害，并为今后制定更可靠的文件规格和裁军方法指明了方向。



## **26. Image-to-Text Logic Jailbreak: Your Imagination can Help You Do Anything**

图像到文本逻辑越狱：你的想象力可以帮助你做任何事情 cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.02534v1) [paper-pdf](http://arxiv.org/pdf/2407.02534v1)

**Authors**: Xiaotian Zou, Yongkang Chen

**Abstract**: Large Visual Language Models (VLMs) such as GPT-4 have achieved remarkable success in generating comprehensive and nuanced responses, surpassing the capabilities of large language models. However, with the integration of visual inputs, new security concerns emerge, as malicious attackers can exploit multiple modalities to achieve their objectives. This has led to increasing attention on the vulnerabilities of VLMs to jailbreak. Most existing research focuses on generating adversarial images or nonsensical image collections to compromise these models. However, the challenge of leveraging meaningful images to produce targeted textual content using the VLMs' logical comprehension of images remains unexplored. In this paper, we explore the problem of logical jailbreak from meaningful images to text. To investigate this issue, we introduce a novel dataset designed to evaluate flowchart image jailbreak. Furthermore, we develop a framework for text-to-text jailbreak using VLMs. Finally, we conduct an extensive evaluation of the framework on GPT-4o and GPT-4-vision-preview, with jailbreak rates of 92.8% and 70.0%, respectively. Our research reveals significant vulnerabilities in current VLMs concerning image-to-text jailbreak. These findings underscore the need for a deeper examination of the security flaws in VLMs before their practical deployment.

摘要: 像GPT-4这样的大型视觉语言模型(VLM)在生成全面和细微差别的响应方面取得了显著的成功，超过了大型语言模型的能力。然而，随着视觉输入的集成，新的安全问题出现了，因为恶意攻击者可以利用多种模式来实现他们的目标。这引起了人们对越狱漏洞的越来越多的关注。现有的大多数研究都集中在生成敌意图像或无意义的图像集合来危害这些模型。然而，利用VLMS对图像的逻辑理解来利用有意义的图像来产生有针对性的文本内容的挑战仍然没有被探索。在本文中，我们探讨了从有意义的图像到文本的逻辑越狱问题。为了研究这个问题，我们引入了一个新的数据集，用于评估流程图图像越狱。此外，我们使用VLMS开发了一个文本到文本越狱的框架。最后，我们在GPT-4O和GPT-4-VISION-PREVIEW上对该框架进行了广泛的评估，越狱率分别为92.8%和70.0%。我们的研究揭示了当前VLM在图像到文本越狱方面的重大漏洞。这些调查结果突出表明，在实际部署VLM之前，需要更深入地审查VLM的安全缺陷。



## **27. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **28. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

突破错误的对抗性示例防御：修复1行代码破解Sabre cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.03672v3) [paper-pdf](http://arxiv.org/pdf/2405.03672v3)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.

摘要: Sabre是对IEEE S & P 2024上接受的敌对例子的辩护。我们首先揭示了评估中的重大缺陷，这些缺陷表明了梯度掩蔽的明显迹象。然后我们展示这种梯度掩蔽的原因：原始评估代码中的一个错误。通过在原始存储库中修复一行代码，我们将Sabre的稳健准确性降低到0%。作为回应，作者修改了辩护并引入了原始论文中未描述的新辩护组件。但此修复包含第二个错误;修改多一行代码会将稳健准确性降低到基线水平以下。在我们在线发布论文的第一版后，作者对防御进行了另一项更改;通过在攻击期间注释掉一行代码，我们将稳健准确性再次降低到0%。



## **29. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **30. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01295v1) [paper-pdf](http://arxiv.org/pdf/2407.01295v1)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **31. DeepiSign-G: Generic Watermark to Stamp Hidden DNN Parameters for Self-contained Tracking**

DeepiSign-G：通用水印，用于标记隐藏的DNN参数以进行自包含跟踪 cs.CR

13 pages

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01260v1) [paper-pdf](http://arxiv.org/pdf/2407.01260v1)

**Authors**: Alsharif Abuadbba, Nicholas Rhodes, Kristen Moore, Bushra Sabir, Shuo Wang, Yansong Gao

**Abstract**: Deep learning solutions in critical domains like autonomous vehicles, facial recognition, and sentiment analysis require caution due to the severe consequences of errors. Research shows these models are vulnerable to adversarial attacks, such as data poisoning and neural trojaning, which can covertly manipulate model behavior, compromising reliability and safety. Current defense strategies like watermarking have limitations: they fail to detect all model modifications and primarily focus on attacks on CNNs in the image domain, neglecting other critical architectures like RNNs.   To address these gaps, we introduce DeepiSign-G, a versatile watermarking approach designed for comprehensive verification of leading DNN architectures, including CNNs and RNNs. DeepiSign-G enhances model security by embedding an invisible watermark within the Walsh-Hadamard transform coefficients of the model's parameters. This watermark is highly sensitive and fragile, ensuring prompt detection of any modifications. Unlike traditional hashing techniques, DeepiSign-G allows substantial metadata incorporation directly within the model, enabling detailed, self-contained tracking and verification.   We demonstrate DeepiSign-G's applicability across various architectures, including CNN models (VGG, ResNets, DenseNet) and RNNs (Text sentiment classifier). We experiment with four popular datasets: VGG Face, CIFAR10, GTSRB Traffic Sign, and Large Movie Review. We also evaluate DeepiSign-G under five potential attacks. Our comprehensive evaluation confirms that DeepiSign-G effectively detects these attacks without compromising CNN and RNN model performance, highlighting its efficacy as a robust security measure for deep learning applications. Detection of integrity breaches is nearly perfect, while hiding only a bit in approximately 1% of the Walsh-Hadamard coefficients.

摘要: 由于错误的严重后果，自动驾驶汽车、面部识别和情绪分析等关键领域的深度学习解决方案需要谨慎。研究表明，这些模型容易受到数据中毒和神经木马等敌意攻击，这些攻击可能会秘密操纵模型行为，损害可靠性和安全性。当前的防御策略，如水印，都有局限性：它们无法检测到所有模型的修改，主要集中在对图像域中的CNN的攻击，而忽略了其他关键的体系结构，如RNN。为了弥补这些差距，我们引入了DeepSign-G，这是一种多功能水印方法，旨在全面验证领先的DNN架构，包括CNN和RNN。DeepSign-G通过在模型参数的Walsh-Hadamard变换系数中嵌入一个不可见的水印来增强模型的安全性。该水印高度敏感和脆弱，确保了对任何修改的及时检测。与传统的散列技术不同，DeepSign-G允许将大量元数据直接合并到模型中，从而实现详细、独立的跟踪和验证。我们演示了DeepSign-G在各种体系结构上的适用性，包括CNN模型(VGG、ResNet、DenseNet)和RNNS(文本情感分类器)。我们使用了四个流行的数据集：VGG Face、CIFAR10、GTSRB交通标志和大电影评论。我们还评估了DeepSign-G在五种潜在攻击下的性能。我们的全面评估证实，DeepSign-G在不影响CNN和RNN模型性能的情况下有效地检测到这些攻击，突出了其作为深度学习应用程序的强大安全措施的有效性。完整性破坏的检测近乎完美，同时只隐藏了大约1%的沃尔什-哈达玛系数中的一位。



## **32. QUEEN: Query Unlearning against Model Extraction**

QUEEN：针对模型提取的查询取消学习 cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01251v1) [paper-pdf](http://arxiv.org/pdf/2407.01251v1)

**Authors**: Huajie Chen, Tianqing Zhu, Lefeng Zhang, Bo Liu, Derui Wang, Wanlei Zhou, Minhui Xue

**Abstract**: Model extraction attacks currently pose a non-negligible threat to the security and privacy of deep learning models. By querying the model with a small dataset and usingthe query results as the ground-truth labels, an adversary can steal a piracy model with performance comparable to the original model. Two key issues that cause the threat are, on the one hand, accurate and unlimited queries can be obtained by the adversary; on the other hand, the adversary can aggregate the query results to train the model step by step. The existing defenses usually employ model watermarking or fingerprinting to protect the ownership. However, these methods cannot proactively prevent the violation from happening. To mitigate the threat, we propose QUEEN (QUEry unlEarNing) that proactively launches counterattacks on potential model extraction attacks from the very beginning. To limit the potential threat, QUEEN has sensitivity measurement and outputs perturbation that prevents the adversary from training a piracy model with high performance. In sensitivity measurement, QUEEN measures the single query sensitivity by its distance from the center of its cluster in the feature space. To reduce the learning accuracy of attacks, for the highly sensitive query batch, QUEEN applies query unlearning, which is implemented by gradient reverse to perturb the softmax output such that the piracy model will generate reverse gradients to worsen its performance unconsciously. Experiments show that QUEEN outperforms the state-of-the-art defenses against various model extraction attacks with a relatively low cost to the model accuracy. The artifact is publicly available at https://anonymous.4open.science/r/queen implementation-5408/.

摘要: 模型提取攻击目前对深度学习模型的安全和隐私构成了不可忽视的威胁。通过使用较小的数据集对模型进行查询，并将查询结果作为地面事实标签，攻击者可以窃取性能与原始模型相当的盗版模型。造成威胁的两个关键问题是，一方面，对手可以获得准确和无限的查询；另一方面，对手可以聚合查询结果，逐步训练模型。现有的防御工事通常采用模型水印或指纹来保护所有权。然而，这些方法不能主动阻止违规行为的发生。为了缓解这种威胁，我们提出了Queue(查询遗忘)算法，它从一开始就主动地对潜在的模型提取攻击发起反击。为了限制潜在威胁，皇后拥有敏感度测量和输出扰动，以防止对手训练高性能的盗版模型。在敏感度度量中，Queue通过距离特征空间中聚类中心的距离来衡量单个查询的敏感度。为了降低攻击的学习精度，对于高度敏感的查询批次，Queue应用查询遗忘，通过梯度反转来实现对Softmax输出的扰动，使得盗版模型会产生反向梯度，从而在不知不觉中恶化其性能。实验表明，Queue在抵抗各种模型提取攻击时的性能优于最先进的防御系统，而模型精确度的代价相对较低。该构件可在https://anonymous.4open.science/r/queen Implementation-5408/上公开获得。



## **33. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

使用对抗红外网格对红外行人探测器进行多视图黑匣子物理攻击 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01168v1) [paper-pdf](http://arxiv.org/pdf/2407.01168v1)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.

摘要: 虽然在可见光光谱内对物理对抗攻击已有广泛的研究，但在红外光谱中对这类技术的研究有限。红外目标探测器在现代技术应用中至关重要，但容易受到对抗性攻击，构成重大安全威胁。以前的研究证明，使用物理扰动，如灯泡阵列和气凝胶进行白盒攻击，或使用冷热补丁进行黑盒攻击，都被证明是不切实际的，或者在多视角支持方面受到限制。为了解决这些问题，我们提出了对抗性红外网格(AdvGrid)，它以网格的形式对扰动进行建模，并使用遗传算法进行黑盒优化。这些扰动被循环应用于行人衣服的不同部分，以促进对红外行人探测器的多视角黑匣子物理攻击。大量实验验证了AdvGrid的有效性、隐蔽性和健壮性。该方法在数字环境下的攻击成功率为80.00%，在物理环境下的攻击成功率为91.86%，优于基准攻击方法。此外，对主流检测器的平均攻击成功率超过50%，显示了AdvGrid的健壮性。我们的分析包括烧蚀研究、转移攻击和对抗性防御，证实了该方法的优越性。



## **34. Unaligning Everything: Or Aligning Any Text to Any Image in Multimodal Models**

将所有内容分开：或将任何文本与多模式模型中的任何图像对齐 cs.CV

14 pages, 14 figures. arXiv admin note: substantial text overlap with  arXiv:2401.15568, arXiv:2402.08473

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01157v1) [paper-pdf](http://arxiv.org/pdf/2407.01157v1)

**Authors**: Shaeke Salman, Md Montasir Bin Shams, Xiuwen Liu

**Abstract**: Utilizing a shared embedding space, emerging multimodal models exhibit unprecedented zero-shot capabilities. However, the shared embedding space could lead to new vulnerabilities if different modalities can be misaligned. In this paper, we extend and utilize a recently developed effective gradient-based procedure that allows us to match the embedding of a given text by minimally modifying an image. Using the procedure, we show that we can align the embeddings of distinguishable texts to any image through unnoticeable adversarial attacks in joint image-text models, revealing that semantically unrelated images can have embeddings of identical texts and at the same time visually indistinguishable images can be matched to the embeddings of very different texts. Our technique achieves 100\% success rate when it is applied to text datasets and images from multiple sources. Without overcoming the vulnerability, multimodal models cannot robustly align inputs from different modalities in a semantically meaningful way. \textbf{Warning: the text data used in this paper are toxic in nature and may be offensive to some readers.}

摘要: 利用共享的嵌入空间，新兴的多模式显示出前所未有的零射击能力。然而，如果不同的模式可能会错位，共享嵌入空间可能会导致新的漏洞。在本文中，我们扩展和利用了最近开发的一种有效的基于梯度的方法，该方法允许我们通过对图像进行最小限度的修改来匹配给定文本的嵌入。利用该过程，我们证明了在联合图文模型中，通过不可察觉的对抗性攻击，可以将可区分文本的嵌入与任何图像对齐，从而揭示了语义无关的图像可以具有相同文本的嵌入，同时视觉上不可区分的图像可以与非常不同的文本的嵌入相匹配。将该方法应用于多个来源的文本数据集和图像，取得了100%的准确率。如果不克服这一弱点，多通道模型就不能以语义有意义的方式稳健地对齐来自不同通道的输入。\textbf{警告：本文中使用的文本数据具有毒性，可能会冒犯某些读者。}



## **35. SecGenAI: Enhancing Security of Cloud-based Generative AI Applications within Australian Critical Technologies of National Interest**

SecGenAI：增强澳大利亚国家利益关键技术中基于云的生成性人工智能应用的安全性 cs.CR

10 pages, 4 figures, 9 tables, submitted to the 2024 11th  International Conference on Soft Computing & Machine Intelligence (ISCMI  2024)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01110v1) [paper-pdf](http://arxiv.org/pdf/2407.01110v1)

**Authors**: Christoforus Yoga Haryanto, Minh Hieu Vu, Trung Duc Nguyen, Emily Lomempow, Yulia Nurliana, Sona Taheri

**Abstract**: The rapid advancement of Generative AI (GenAI) technologies offers transformative opportunities within Australia's critical technologies of national interest while introducing unique security challenges. This paper presents SecGenAI, a comprehensive security framework for cloud-based GenAI applications, with a focus on Retrieval-Augmented Generation (RAG) systems. SecGenAI addresses functional, infrastructure, and governance requirements, integrating end-to-end security analysis to generate specifications emphasizing data privacy, secure deployment, and shared responsibility models. Aligned with Australian Privacy Principles, AI Ethics Principles, and guidelines from the Australian Cyber Security Centre and Digital Transformation Agency, SecGenAI mitigates threats such as data leakage, adversarial attacks, and model inversion. The framework's novel approach combines advanced machine learning techniques with robust security measures, ensuring compliance with Australian regulations while enhancing the reliability and trustworthiness of GenAI systems. This research contributes to the field of intelligent systems by providing actionable strategies for secure GenAI implementation in industry, fostering innovation in AI applications, and safeguarding national interests.

摘要: 产生式人工智能(GenAI)技术的快速发展为澳大利亚涉及国家利益的关键技术提供了变革性的机会，同时带来了独特的安全挑战。提出了一种基于云的GenAI应用安全框架SecGenAI，重点研究了检索-增强生成(RAG)系统。SecGenAI解决了功能、基础设施和治理需求，集成了端到端安全分析，以生成强调数据隐私、安全部署和分担责任模型的规范。SecGenAI与澳大利亚隐私原则、人工智能道德原则以及澳大利亚网络安全中心和数字转型机构的指导方针保持一致，可以缓解数据泄露、对抗性攻击和模型反转等威胁。该框架的新方法将先进的机器学习技术与强大的安全措施相结合，在确保符合澳大利亚法规的同时，增强了GenAI系统的可靠性和可信度。这项研究为工业中安全实施GenAI提供了可行的策略，促进了人工智能应用的创新，并维护了国家利益，从而为智能系统领域做出了贡献。



## **36. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross-Domain**

DifAttack++：跨域中通过分层解纠缠特征空间进行查询高效黑匣子对抗攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585 An  extension of the AAAI24 paper "DifAttack: Query-Efficient Black-Box Attack  via Disentangled Feature Space."

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.03017v3) [paper-pdf](http://arxiv.org/pdf/2406.03017v3)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian, Zheng Li

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (\textbf{ASR}) and good generalizability. We design a novel attack method based on a hierarchical DIsentangled Feature space, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an Adversarial Feature (\textbf{AF}) and a Visual Feature (\textbf{VF}) via an autoencoder equipped with our specially designed Hierarchical Decouple-Fusion (\textbf{HDF}) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such two autoencoders for the clean and adversarial image domains (i.e., cross-domain) respectively to achieve image reconstructions and feature disentanglement, by using pairs of clean images and their Adversarial Examples (\textbf{AE}s) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our DifAttack++ leads to superior ASR and query efficiency than state-of-the-art methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(Textbf{asr})和良好的泛化能力。我们设计了一种新的基于分层解缠特征空间的攻击方法，称为Textbf{DifAttack++}，它与现有的操作在整个特征空间上的攻击方法有很大的不同。具体地说，DifAttack++首先通过配备了我们特别设计的分层去耦合融合(\extbf{hdf})模块的自动编码器，将图像的潜在特征分解为对抗特征(\extbf{AF})和视觉特征(\extbf{Vf})，其中，AF主导图像的对抗能力，而VF在很大程度上决定其视觉外观。通过白盒攻击方法，利用已有代理模型生成的干净图像对和对抗性图像对(S)，分别对干净图像和对抗性图像域(即跨域)进行训练，以实现图像重建和特征解缠。最终，在黑盒攻击阶段，DifAttack++根据受害者模型的查询反馈迭代地优化AF，直到生成成功的AE，同时保持VF不变。大量的实验结果表明，我们的DifAttack++比现有的方法具有更高的ASR和查询效率，同时表现出更好的AEs视觉质量。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **37. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

深度学习模型对脑纹识别的时频联合不可感知的对抗攻击 cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2403.10021v3) [paper-pdf](http://arxiv.org/pdf/2403.10021v3)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.

摘要: 基于深度学习模型的脑电脑纹识别在生物特征识别中得到了广泛的关注。然而，研究表明，在有脑电输入的深度学习模型中，容易受到对抗性攻击。本文提出了一种利用小波变换联合攻击时频域脑电信号的对抗性攻击方法。不同于现有的大多数只针对时域脑电信号的方法，我们的方法不仅利用了时域攻击的强大对抗能力，而且得益于频域攻击固有的不可感知性，在攻击性能和不可感知性之间取得了更好的平衡。在白盒和灰盒场景下进行了大量的实验，结果表明，我们的攻击方法在三个数据集和三个深度学习模型上获得了最先进的攻击性能。同时，我们的方法攻击的信号中的扰动几乎不被人类视觉系统察觉到。



## **38. Learning Robust 3D Representation from CLIP via Dual Denoising**

通过双重去噪从CLIP学习稳健的3D表示 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00905v1) [paper-pdf](http://arxiv.org/pdf/2407.00905v1)

**Authors**: Shuqing Luo, Bowen Qu, Wei Gao

**Abstract**: In this paper, we explore a critical yet under-investigated issue: how to learn robust and well-generalized 3D representation from pre-trained vision language models such as CLIP. Previous works have demonstrated that cross-modal distillation can provide rich and useful knowledge for 3D data. However, like most deep learning models, the resultant 3D learning network is still vulnerable to adversarial attacks especially the iterative attack. In this work, we propose Dual Denoising, a novel framework for learning robust and well-generalized 3D representations from CLIP. It combines a denoising-based proxy task with a novel feature denoising network for 3D pre-training. Additionally, we propose utilizing parallel noise inference to enhance the generalization of point cloud features under cross domain settings. Experiments show that our model can effectively improve the representation learning performance and adversarial robustness of the 3D learning network under zero-shot settings without adversarial training. Our code is available at https://github.com/luoshuqing2001/Dual_Denoising.

摘要: 在本文中，我们探索了一个关键但未被研究的问题：如何从预先训练的视觉语言模型(如CLIP)中学习健壮和良好通用的3D表示。前人的工作已经证明，跨峰蒸馏可以为三维数据提供丰富而有用的知识。然而，与大多数深度学习模型一样，生成的3D学习网络仍然容易受到对抗性攻击，特别是迭代攻击。在这项工作中，我们提出了双重去噪，一个新的框架，学习稳健和良好的通用3D表示从CLIP。它将基于去噪的代理任务与一种新颖的特征去噪网络相结合，用于3D预训练。此外，我们还提出利用并行噪声推理来增强跨域环境下点云特征的泛化能力。实验表明，该模型可以有效地提高3D学习网络在零射击环境下的表征学习性能和对抗健壮性。我们的代码可以在https://github.com/luoshuqing2001/Dual_Denoising.上找到



## **39. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

GRACE：图形正规化注意卷积纠缠与拉普拉斯平滑，用于鲁棒的DeepFake视频检测 cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.19941v2) [paper-pdf](http://arxiv.org/pdf/2406.19941v2)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.

摘要: 随着DeepFake视频操纵技术的升级，构成了深刻的威胁，迫切需要开发有效的检测策略。然而，一个特别的问题是面部图像被误检，通常是由于视频降级或对手攻击，导致意外的时间伪影，这可能会破坏DeepFake视频检测技术的效率。提出了一种新的基于图拉普拉斯卷积网络的图正则化注意力卷积纠缠(GRACE)算法，用于检测DeepFake视频。首先，使用传统的卷积神经网络来执行整个视频的时空特征。然后通过构造具有稀疏约束的图将空间特征和时间特征相互纠缠在一起，在剩余的噪声人脸序列中强化有效人脸图像的本质特征，从而增强了DeepFake视频检测的稳定性和性能。此外，在图卷积网络中提出了图拉普拉斯先验，去除了特征空间中的噪声模式，进一步提高了性能。实验结果表明，本文提出的方法在含噪人脸序列下的DeepFake视频检测中具有较好的性能。源代码可在https://github.com/ming053l/GRACE.上找到



## **40. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

利用安全性和活力来增强性能的两层区块链碎片协议 cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2310.11373v4) [paper-pdf](http://arxiv.org/pdf/2310.11373v4)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.

摘要: 分片对于提高区块链可伸缩性至关重要。现有的协议忽略了不同的对抗性攻击，限制了交易吞吐量。本文提出了一种突破性的分片协议Reetum，解决了这个问题，提高了区块链的可扩展性。RENETUM采用两阶段方法，根据运行时敌意攻击调整事务吞吐量。它包括两层的“控制”和“流程”分片。进程碎片包含至少一个可信节点，而控制碎片包含大多数可信节点。在第一阶段，事务被写入块，并由流程碎片中的节点投票表决。一致接受的障碍得到确认。在第二阶段，未获得一致接受的块由控制碎片投票表决。如果多数人投赞成票，就会接受阻止，从而消除第一阶段的反对者和沉默的选民。第一阶段使用一致投票，涉及的节点更少，支持更多的并行进程碎片。控制碎片最终确定决策并解决纠纷。实验证实了ReNetum的创新设计，提供了高交易吞吐量和对各种网络攻击的稳健性，性能优于现有的区块链网络分片协议。



## **41. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

强化守护者，而不是宝藏：弹性对抗探测器 cs.CV

**SubmitDate**: 2024-06-30    [abs](http://arxiv.org/abs/2404.12120v2) [paper-pdf](http://arxiv.org/pdf/2404.12120v2)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.

摘要: 本文提出了RADART--通过对抗重训练的鲁棒对抗检测--一种旨在增强对抗检测器对抗自适应攻击的鲁棒性的方法，同时保持分类器性能。自适应攻击是攻击者意识到防御并相应调整策略的攻击。我们提出的方法利用对抗性训练来加强检测攻击的能力，而不会损害准确性。在训练阶段，我们将对抗性示例集成到数据集中，这些示例经过优化以愚弄分类器和对抗性检测器，使对抗性检测器能够学习和适应潜在的攻击场景。对CIFAR-10和SVHN数据集的实验评估表明，我们提出的算法显着提高了检测器准确识别自适应对抗攻击的能力，而不会牺牲清晰的准确性。



## **42. Query-Efficient Hard-Label Black-Box Attack against Vision Transformers**

针对Vision Transformers的查询高效硬标签黑匣子攻击 cs.CV

**SubmitDate**: 2024-06-29    [abs](http://arxiv.org/abs/2407.00389v1) [paper-pdf](http://arxiv.org/pdf/2407.00389v1)

**Authors**: Chao Zhou, Xiaowen Shi, Yuan-Gen Wang

**Abstract**: Recent studies have revealed that vision transformers (ViTs) face similar security risks from adversarial attacks as deep convolutional neural networks (CNNs). However, directly applying attack methodology on CNNs to ViTs has been demonstrated to be ineffective since the ViTs typically work on patch-wise encoding. This article explores the vulnerability of ViTs against adversarial attacks under a black-box scenario, and proposes a novel query-efficient hard-label adversarial attack method called AdvViT. Specifically, considering that ViTs are highly sensitive to patch modification, we propose to optimize the adversarial perturbation on the individual patches. To reduce the dimension of perturbation search space, we modify only a handful of low-frequency components of each patch. Moreover, we design a weight mask matrix for all patches to further optimize the perturbation on different regions of a whole image. We test six mainstream ViT backbones on the ImageNet-1k dataset. Experimental results show that compared with the state-of-the-art attacks on CNNs, our AdvViT achieves much lower $L_2$-norm distortion under the same query budget, sufficiently validating the vulnerability of ViTs against adversarial attacks.

摘要: 最近的研究表明，视觉转换器(VITS)面临着与深层卷积神经网络(CNN)相似的对抗攻击的安全风险。然而，直接将针对CNN的攻击方法应用于VITS已被证明是无效的，因为VITS通常工作在补丁式编码上。本文研究了VITS在黑盒场景下抵抗敌意攻击的脆弱性，提出了一种新的查询高效的硬标签敌意攻击方法AdvViT。具体地说，考虑到VITS对补丁修改高度敏感，我们提出了优化单个补丁上的对抗性扰动。为了降低扰动搜索空间的维度，我们只对每个块的少数低频分量进行了修改。此外，为了进一步优化整个图像不同区域的扰动，我们为所有的块设计了一个加权掩模矩阵。我们在ImageNet-1k数据集上测试了六个主流VIT主干。实验结果表明，与已有的针对CNN的攻击相比，在相同的查询预算下，我们的AdvViT获得了更低的$L_2$范数失真，充分验证了VITS对对手攻击的脆弱性。



## **43. DiffuseDef: Improved Robustness to Adversarial Attacks**

diffuseDef：增强对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2407.00248v1) [paper-pdf](http://arxiv.org/pdf/2407.00248v1)

**Authors**: Zhenhao Li, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to system built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over different existing adversarial defense methods and achieves state-of-the-art performance against common adversarial attacks.

摘要: 预训练的语言模型在各种自然语言处理任务中显着提高了性能。然而，对抗性攻击继续对使用这些模型构建的系统构成严峻挑战，因为它们可以被精心设计的对抗性文本利用。受到扩散模型预测和减少计算机视觉中噪音的能力的启发，我们提出了一种新颖且灵活的语言分类任务对抗防御方法：DistuseDef，它在编码器和分类器之间引入了扩散层作为降噪器。在推理过程中，对抗性隐藏状态首先与采样噪音相结合，然后迭代去噪，最后集成以产生稳健的文本表示。通过集成对抗性训练、去噪和集成技术，我们证明了DistuseDef比不同的现有对抗性防御方法进行了改进，并在对抗常见对抗性攻击时实现了最先进的性能。



## **44. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

破译事后OOD检测器对抗鲁棒性的定义 cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.15104v3) [paper-pdf](http://arxiv.org/pdf/2406.15104v3)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.

摘要: 检测非分布（OOD）输入对于在现实世界场景中安全部署深度学习模型至关重要。近年来，开发了很多OOD检测器，甚至基准测试也已经标准化，即OpenOOD。事后检测器的数量正在快速增长，并显示出一种可以保护预训练的分类器免受自然分布变化的影响的选择，声称已经为现实世界的场景做好了准备。然而，它在处理敌对例子方面的功效在大多数研究中被忽视了。本文研究了16个事后检测器对多种规避攻击的对抗鲁棒性，并讨论了OOD检测器对抗防御的路线图。



## **45. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

分布风险接受性和鲁棒性下$k$-次模函数的Stackelberg博弈 math.OC

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.13023v3) [paper-pdf](http://arxiv.org/pdf/2406.13023v3)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.

摘要: 我们研究了对抗性环境下的子模优化，适用于机器学习问题，例如使用对不确定性和攻击敏感的数据进行特征选择。我们主要研究攻击者(或中断者)和防御者之间的Stackelberg博弈，其中攻击者的目标是最小化防御者最大化$k$-子模函数的目标。我们允许攻击成功和固有数据噪声带来的不确定性，并解决由于不完全了解随机参数的概率分布而带来的挑战。具体地，我们引入了分布式风险厌恶$k$-子模阻断问题(DRA$k$-SIP)和分布式风险厌恶$k$-子模阻断问题(DRR$k$-SIP)，并给出了有限收敛的精确算法。DRA$k$-SIP解决方案允许风险厌恶中断者针对现实世界的不确定性制定稳健的策略。相反，DRR$k$-SIP解决方案建议攻击者采用攻击性策略，愿意承担(分布式)风险以造成最大损害，识别关键易受攻击的组件，可用于防御者的防御策略。从DRA$k$-SIP和DRR$k$-SIP导出的最佳值为防御者的目标函数的期望值提供了类似于置信度的范围，从而捕获了分布模糊性。我们分别使用特征选择和传感器放置问题的实例以及威斯康星州的乳腺癌数据和合成数据进行了计算实验。



## **46. Emotion Loss Attacking: Adversarial Attack Perception for Skeleton based on Multi-dimensional Features**

情感损失攻击：基于多维特征的骨架对抗攻击感知 cs.CV

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19815v1) [paper-pdf](http://arxiv.org/pdf/2406.19815v1)

**Authors**: Feng Liu, Qing Xu, Qijian Zheng

**Abstract**: Adversarial attack on skeletal motion is a hot topic. However, existing researches only consider part of dynamic features when measuring distance between skeleton graph sequences, which results in poor imperceptibility. To this end, we propose a novel adversarial attack method to attack action recognizers for skeletal motions. Firstly, our method systematically proposes a dynamic distance function to measure the difference between skeletal motions. Meanwhile, we innovatively introduce emotional features for complementary information. In addition, we use Alternating Direction Method of Multipliers(ADMM) to solve the constrained optimization problem, which generates adversarial samples with better imperceptibility to deceive the classifiers. Experiments show that our method is effective on multiple action classifiers and datasets. When the perturbation magnitude measured by l norms is the same, the dynamic perturbations generated by our method are much lower than that of other methods. What's more, we are the first to prove the effectiveness of emotional features, and provide a new idea for measuring the distance between skeletal motions.

摘要: 骨骼运动的对抗性攻击是一个热门话题。然而，现有的研究在度量骨架图序列之间的距离时只考虑了部分动态特征，导致隐蔽性较差。为此，我们提出了一种新的对抗性攻击方法来攻击骨骼运动的动作识别器。首先，我们的方法系统地提出了一个动态距离函数来衡量骨骼运动之间的差异。同时，我们创新性地引入了情感特征来补充信息。此外，我们使用乘子交替方向法(ADMM)来求解约束优化问题，生成的对抗性样本具有更好的隐蔽性来欺骗分类器。实验表明，该方法在多个动作分类器和数据集上是有效的。在L范数测得的摄动强度相同的情况下，我们的方法产生的动力摄动比其他方法产生的小得多。更重要的是，我们首次证明了情感特征的有效性，并为测量骨骼运动之间的距离提供了新的思路。



## **47. Deceptive Diffusion: Generating Synthetic Adversarial Examples**

欺骗性扩散：生成合成对抗示例 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19807v1) [paper-pdf](http://arxiv.org/pdf/2406.19807v1)

**Authors**: Lucas Beerens, Catherine F. Higham, Desmond J. Higham

**Abstract**: We introduce the concept of deceptive diffusion -- training a generative AI model to produce adversarial images. Whereas a traditional adversarial attack algorithm aims to perturb an existing image to induce a misclassificaton, the deceptive diffusion model can create an arbitrary number of new, misclassified images that are not directly associated with training or test images. Deceptive diffusion offers the possibility of strengthening defence algorithms by providing adversarial training data at scale, including types of misclassification that are otherwise difficult to find. In our experiments, we also investigate the effect of training on a partially attacked data set. This highlights a new type of vulnerability for generative diffusion models: if an attacker is able to stealthily poison a portion of the training data, then the resulting diffusion model will generate a similar proportion of misleading outputs.

摘要: 我们引入了欺骗性扩散的概念--训练生成式人工智能模型来产生对抗性图像。传统的对抗攻击算法旨在扰乱现有图像以引发错误分类，而欺骗性扩散模型可以创建任意数量的新的、错误分类的图像，这些图像与训练或测试图像不直接相关。欺骗性扩散通过大规模提供对抗训练数据（包括其他方式难以发现的错误分类类型）来加强防御算法。在我们的实验中，我们还研究了训练对部分攻击的数据集的影响。这凸显了生成性扩散模型的一种新型漏洞：如果攻击者能够悄悄毒害一部分训练数据，那么产生的扩散模型将生成类似比例的误导性输出。



## **48. Backdoor Attack in Prompt-Based Continual Learning**

基于预算的持续学习中的后门攻击 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19753v1) [paper-pdf](http://arxiv.org/pdf/2406.19753v1)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.

摘要: 基于提示的方法为持续学习中的数据隐私问题提供了一种尖端解决方案，特别是在涉及多个数据供应商的场景中，禁止长期存储私人用户数据。尽管提供了最先进的性能，但其令人印象深刻的记忆能力可能会成为一把双刃剑，这引发了安全问题，因为它可能会无意中保留在从私人用户数据学习过程中注入的有毒知识。根据这一见解，在本文中，我们将持续学习暴露于一个潜在的威胁：后门攻击，它驱动模型在出现特定触发时跟踪期望的对手目标，同时仍然在干净的样本上正常运行。我们强调了对增量学习者执行后门攻击的三个关键挑战并提出了相应的解决方案：(1)\emph{可传递性}：我们使用代理数据集并操纵提示选择来将后门知识传输到其他供应商的数据；(2)\emph{弹性}：我们模拟受害者的静态和动态，以确保后门触发在激烈的增量学习过程中保持健壮；以及(3)\emph{真实性}：我们应用二进制交叉熵损失作为反作弊因子，以防止后门触发演变为对抗性噪声。在各种基准数据集和不断学习的人中进行的广泛实验验证了我们的持续后门框架，实现了高达100美元的攻击成功率，进一步的消融研究证实了我们的贡献的有效性。



## **49. IDT: Dual-Task Adversarial Attacks for Privacy Protection**

IDT：隐私保护的双重任务对抗攻击 cs.CL

28 pages, 1 figure

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19642v1) [paper-pdf](http://arxiv.org/pdf/2406.19642v1)

**Authors**: Pedro Faustini, Shakila Mahjabin Tonni, Annabelle McIver, Qiongkai Xu, Mark Dras

**Abstract**: Natural language processing (NLP) models may leak private information in different ways, including membership inference, reconstruction or attribute inference attacks. Sensitive information may not be explicit in the text, but hidden in underlying writing characteristics. Methods to protect privacy can involve using representations inside models that are demonstrated not to detect sensitive attributes or -- for instance, in cases where users might not trust a model, the sort of scenario of interest here -- changing the raw text before models can have access to it. The goal is to rewrite text to prevent someone from inferring a sensitive attribute (e.g. the gender of the author, or their location by the writing style) whilst keeping the text useful for its original intention (e.g. the sentiment of a product review). The few works tackling this have focused on generative techniques. However, these often create extensively different texts from the original ones or face problems such as mode collapse. This paper explores a novel adaptation of adversarial attack techniques to manipulate a text to deceive a classifier w.r.t one task (privacy) whilst keeping the predictions of another classifier trained for another task (utility) unchanged. We propose IDT, a method that analyses predictions made by auxiliary and interpretable models to identify which tokens are important to change for the privacy task, and which ones should be kept for the utility task. We evaluate different datasets for NLP suitable for different tasks. Automatic and human evaluations show that IDT retains the utility of text, while also outperforming existing methods when deceiving a classifier w.r.t privacy task.

摘要: 自然语言处理(NLP)模型可能以不同的方式泄露隐私信息，包括成员关系推理、重构或属性推理攻击。敏感信息可能在文本中不是显性的，而是隐藏在潜在的写作特征中。保护隐私的方法可以涉及在模型中使用表示法，这些表示法被演示为不检测敏感属性，或者--例如，在用户可能不信任模型的情况下，这里涉及的场景--在模型可以访问它之前更改原始文本。目标是重写文本，以防止有人推断敏感属性(例如，作者的性别或他们的写作风格)，同时保持文本对其原始意图(例如，产品评论的情绪)的有用。解决这一问题的少数作品都集中在生成技术上。然而，这些通常会产生与原始文本截然不同的文本，或者面临模式崩溃等问题。本文探索了一种新颖的对抗性攻击技术，以操纵文本来欺骗一个任务(隐私)的分类器，同时保持为另一个任务(效用)训练的另一个分类器的预测不变。我们提出了IDT，这是一种分析辅助模型和可解释模型所做预测的方法，以确定哪些令牌对于隐私任务是重要的，哪些应该为实用任务保留。我们评估了适合不同任务的自然语言处理的不同数据集。自动和人工评估表明，IDT保留了文本的效用，同时在欺骗分类器w.r.t隐私任务时也优于现有方法。



## **50. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

数据驱动的Lipschitz连续性：提高对抗稳健性的经济有效方法 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19622v1) [paper-pdf](http://arxiv.org/pdf/2406.19622v1)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: The security and robustness of deep neural networks (DNNs) have become increasingly concerning. This paper aims to provide both a theoretical foundation and a practical solution to ensure the reliability of DNNs. We explore the concept of Lipschitz continuity to certify the robustness of DNNs against adversarial attacks, which aim to mislead the network with adding imperceptible perturbations into inputs. We propose a novel algorithm that remaps the input domain into a constrained range, reducing the Lipschitz constant and potentially enhancing robustness. Unlike existing adversarially trained models, where robustness is enhanced by introducing additional examples from other datasets or generative models, our method is almost cost-free as it can be integrated with existing models without requiring re-training. Experimental results demonstrate the generalizability of our method, as it can be combined with various models and achieve enhancements in robustness. Furthermore, our method achieves the best robust accuracy for CIFAR10, CIFAR100, and ImageNet datasets on the RobustBench leaderboard.

摘要: 深度神经网络(DNN)的安全性和健壮性越来越受到人们的关注。本文旨在为保证DNN的可靠性提供理论基础和实际解决方案。我们利用Lipschitz连续性的概念来证明DNN对敌意攻击的健壮性，目的是通过在输入中添加不可察觉的扰动来误导网络。我们提出了一种新的算法，将输入域重新映射到一个受约束的范围，降低了Lipschitz常数，并潜在地增强了鲁棒性。与现有的对抗性训练模型不同，我们的方法通过引入来自其他数据集或生成性模型的额外样本来增强稳健性，因为它可以与现有模型集成在一起，而不需要重新训练。实验结果表明，该方法具有较好的泛化能力，可以与多种模型相结合，提高了算法的鲁棒性。此外，我们的方法对于罗布斯本奇排行榜上的CIFAR10、CIFAR100和ImageNet数据集获得了最好的稳健精度。



