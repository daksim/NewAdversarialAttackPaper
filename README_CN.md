# Latest Adversarial Attack Papers
**update at 2024-11-04 11:09:37**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们对来自不同家族(Gema，Phi3，Mistral，Zephy，Llama2)和不同尺度(2B，3.8B，7B)的五个模型的经验评估表明，这两种算法在保持实用性的同时，显著增强了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **2. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

对抗非中心扰动：用$l_1$-模估计精确恢复线性系统 math.OC

Theorem 1 turned out to be incorrect

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.03218v2) [paper-pdf](http://arxiv.org/pdf/2410.03218v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.

摘要: 本文研究一般情况下的线性系统辨识问题，其中扰动是亚高斯的，相关的，可能是对抗性的。首先，我们考虑了具有非中心(非零均值)扰动的情况，对于这种情况，普通的最小二乘(OLS)方法不能正确地辨识系统。我们证明了在每个扰动具有相等的正负概率的条件下，$L_1$-范数估计量能够准确地辨识系统。这一条件限制了每个扰动的符号，但允许其大小任意。其次，在攻击次数偶尔发生但攻击值的分布完全任意的情况下，我们考虑了每次扰动是对抗性的情况。我们证明了当给定时刻发生攻击的概率小于0.5时，$L_1$-范数估计对任何对抗性非中心扰动都是有效的，并且在有限时间内实现了精确的恢复。这些结果为在安全关键系统中有效防御任意规模的非中心攻击铺平了道路。



## **3. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **4. Improved Generation of Adversarial Examples Against Safety-aligned LLMs**

针对安全一致的LLM改进对抗示例的生成 cs.CR

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.20778v2) [paper-pdf](http://arxiv.org/pdf/2405.20778v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing automatic jailbreak attacks against safety-aligned LLMs. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, into gradient-based adversarial prompt generation and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that 87% of the query-specific adversarial suffixes generated by the developed combination can induce Llama-2-7B-Chat to produce the output that exactly matches the target string on AdvBench. This match rate is 33% higher than that of a very strong baseline known as GCG, demonstrating advanced discrete optimization for adversarial prompt generation against LLMs. In addition, without introducing obvious cost, the combination achieves >30% absolute increase in attack success rates compared with GCG when generating both query-specific (38% -> 68%) and universal adversarial prompts (26.68% -> 60.32%) for attacking the Llama-2-7B-Chat model on AdvBench. Code at: https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.

摘要: 使用基于梯度的方法生成的对抗性提示在执行针对安全对齐的LLM的自动越狱攻击方面表现出出色的性能。然而，由于文本的离散性，LLMS的输入梯度难以准确反映提示中令牌替换导致的损失变化的大小，导致即使在白盒设置下，对安全对齐的LLM的攻击成功率也是有限的。在这篇文章中，我们探索了一个新的视角来解决这个问题，建议通过利用最初被提出用于攻击黑盒图像分类模型的基于传输的攻击的创新来缓解这个问题。我们首次将这些基于转移的攻击中有效方法的思想，即跳过梯度法和中级攻击，应用到基于梯度的对抗性提示生成中，并且在不引入明显计算代价的情况下获得了显著的性能提升。同时，通过讨论收益背后的机制，得出了新的见解，并开发了这些方法的适当组合。我们的实验结果表明，该组合生成的特定于查询的敌意后缀中，87%可以诱导Llama-2-7B-Chat生成与AdvBch上的目标字符串完全匹配的输出。这一匹配率比非常强的基线GCG的匹配率高出33%，展示了针对LLMS的对抗性提示生成的高级离散优化。此外，在不引入明显成本的情况下，与GCG相比，在生成针对特定查询(38%->68%)和通用对手提示(26.68%->60.32%)的攻击提示时，该组合的攻击成功率绝对值提高了30%以上。代码：https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.



## **5. Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions**

基于不确定性的离线变分Bayesian强化学习在不同数据损坏下的鲁棒性 cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00465v1) [paper-pdf](http://arxiv.org/pdf/2411.00465v1)

**Authors**: Rui Yang, Jie Wang, Guoping Wu, Bin Li

**Abstract**: Real-world offline datasets are often subject to data corruptions (such as noise or adversarial attacks) due to sensor failures or malicious attacks. Despite advances in robust offline reinforcement learning (RL), existing methods struggle to learn robust agents under high uncertainty caused by the diverse corrupted data (i.e., corrupted states, actions, rewards, and dynamics), leading to performance degradation in clean environments. To tackle this problem, we propose a novel robust variational Bayesian inference for offline RL (TRACER). It introduces Bayesian inference for the first time to capture the uncertainty via offline data for robustness against all types of data corruptions. Specifically, TRACER first models all corruptions as the uncertainty in the action-value function. Then, to capture such uncertainty, it uses all offline data as the observations to approximate the posterior distribution of the action-value function under a Bayesian inference framework. An appealing feature of TRACER is that it can distinguish corrupted data from clean data using an entropy-based uncertainty measure, since corrupted data often induces higher uncertainty and entropy. Based on the aforementioned measure, TRACER can regulate the loss associated with corrupted data to reduce its influence, thereby enhancing robustness and performance in clean environments. Experiments demonstrate that TRACER significantly outperforms several state-of-the-art approaches across both individual and simultaneous data corruptions.

摘要: 由于传感器故障或恶意攻击，现实世界中的离线数据集经常受到数据损坏(如噪声或敌意攻击)的影响。尽管在稳健的离线强化学习(RL)方面取得了进展，但现有的方法难以在由不同的被破坏的数据(即，被破坏的状态、动作、奖励和动态)造成的高度不确定性下学习稳健的主体，从而导致在清洁环境中的性能下降。针对这一问题，我们提出了一种新的用于离线跟踪的稳健变分贝叶斯推理方法。它首次引入贝叶斯推理，通过离线数据捕捉不确定性，从而对所有类型的数据损坏具有健壮性。具体地说，Tracer首先将所有腐败建模为动作值函数中的不确定性。然后，为了捕捉这种不确定性，它使用所有离线数据作为观测值，在贝叶斯推理框架下近似动作值函数的后验分布。Tracer的一个吸引人的特点是，它可以使用基于熵的不确定性度量来区分损坏的数据和干净的数据，因为损坏的数据通常会导致更高的不确定性和熵。基于上述措施，Tracer可以控制与损坏数据相关的损失，以减少其影响，从而增强在清洁环境中的健壮性和性能。实验表明，Tracer在单个和同时数据损坏方面的性能明显优于几种最先进的方法。



## **6. Adversarial Purification and Fine-tuning for Robust UDC Image Restoration**

鲁棒UDC图像恢复的对抗净化和微调 eess.IV

Failure to meet expectations

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2402.13629v3) [paper-pdf](http://arxiv.org/pdf/2402.13629v3)

**Authors**: Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Zhaoxin Fan, Jianfeng Lu

**Abstract**: This study delves into the enhancement of Under-Display Camera (UDC) image restoration models, focusing on their robustness against adversarial attacks. Despite its innovative approach to seamless display integration, UDC technology faces unique image degradation challenges exacerbated by the susceptibility to adversarial perturbations. Our research initially conducts an in-depth robustness evaluation of deep-learning-based UDC image restoration models by employing several white-box and black-box attacking methods. This evaluation is pivotal in understanding the vulnerabilities of current UDC image restoration techniques. Following the assessment, we introduce a defense framework integrating adversarial purification with subsequent fine-tuning processes. First, our approach employs diffusion-based adversarial purification, effectively neutralizing adversarial perturbations. Then, we apply the fine-tuning methodologies to refine the image restoration models further, ensuring that the quality and fidelity of the restored images are maintained. The effectiveness of our proposed approach is validated through extensive experiments, showing marked improvements in resilience against typical adversarial attacks.

摘要: 该研究深入研究了显示下摄像机(UDC)图像恢复模型的增强，重点研究其对对手攻击的健壮性。尽管UDC技术以创新的方式实现了无缝显示集成，但它面临着独特的图像降级挑战，这一挑战因易受对抗性干扰而加剧。我们的研究首先采用了几种白盒和黑盒攻击方法，对基于深度学习的UDC图像恢复模型进行了深入的稳健性评估。这一评估对于理解当前UDC图像恢复技术的脆弱性至关重要。在评估之后，我们介绍了一个集成了对抗性净化和后续微调过程的防御框架。首先，我们的方法采用了基于扩散的对抗性净化，有效地中和了对抗性扰动。然后，我们应用微调方法进一步改进图像恢复模型，确保恢复图像的质量和保真度得到保持。通过大量的实验验证了该方法的有效性，表明该方法在抵抗典型的对抗性攻击方面有了显著的提高。



## **7. Towards Building Secure UAV Navigation with FHE-aware Knowledge Distillation**

利用FHE感知的知识提炼构建安全的无人机导航 cs.CR

arXiv admin note: text overlap with arXiv:2404.17225

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00403v1) [paper-pdf](http://arxiv.org/pdf/2411.00403v1)

**Authors**: Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: In safeguarding mission-critical systems, such as Unmanned Aerial Vehicles (UAVs), preserving the privacy of path trajectories during navigation is paramount. While the combination of Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) holds promise, the computational overhead of FHE presents a significant challenge. This paper proposes an innovative approach that leverages Knowledge Distillation to enhance the practicality of secure UAV navigation. By integrating RL and FHE, our framework addresses vulnerabilities to adversarial attacks while enabling real-time processing of encrypted UAV camera feeds, ensuring data security. To mitigate FHE's latency, Knowledge Distillation is employed to compress the network, resulting in an impressive 18x speedup without compromising performance, as evidenced by an R-squared score of 0.9499 compared to the original model's score of 0.9631. Our methodology underscores the feasibility of processing encrypted data for UAV navigation tasks, emphasizing security alongside performance efficiency and timely processing. These findings pave the way for deploying autonomous UAVs in sensitive environments, bolstering their resilience against potential security threats.

摘要: 在保护任务关键系统，如无人机(UAV)中，保护导航过程中路径轨迹的隐私是至关重要的。虽然强化学习(RL)和完全同态加密(FHE)的结合有希望，但FHE的计算开销是一个巨大的挑战。本文提出了一种利用知识蒸馏来增强无人机安全导航实用性的创新方法。通过集成RL和FHE，我们的框架解决了对抗攻击的漏洞，同时允许实时处理加密的无人机摄像头馈送，确保数据安全。为了减少FHE的延迟，使用知识蒸馏来压缩网络，在不影响性能的情况下获得令人印象深刻的18倍加速，R平方分数为0.9499，而原始模型的分数为0.9631。我们的方法强调了为无人机导航任务处理加密数据的可行性，强调安全性以及性能效率和及时处理。这些发现为在敏感环境中部署自动无人机铺平了道路，增强了它们对潜在安全威胁的韧性。



## **8. OSLO: One-Shot Label-Only Membership Inference Attacks**

Oslo：一次性标签会员推断攻击 cs.LG

To appear at NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16978v3) [paper-pdf](http://arxiv.org/pdf/2405.16978v3)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is at least 7$\times$ higher under a 1\% FPR and at least 22$\times$ higher under a 0.1\% FPR on CIFAR100 for a ResNet18 model. We evaluated multiple defense mechanisms against OSLO.

摘要: 我们引入了一次仅标签(Oslo)成员关系推理攻击(MIA)，该攻击仅使用目标模型返回预测的硬标签，即可高精度地推断给定样本在目标模型训练集中的成员资格。这与最先进的纯标签攻击形成对比，后者需要$\sim6000$查询，但获得的攻击精度低于奥斯陆的攻击精度。奥斯陆利用基于传输的黑盒对抗攻击。其核心思想是成员样本比非成员样本对对抗性扰动表现出更强的抵抗力。我们将Oslo与最先进的纯标签攻击进行了比较，并证明了尽管只需要一次查询，但在相同的误检率(FPR)下，我们的方法在准确率和真阳性率(TPR)方面明显优于以前的攻击。例如，与以前的纯标签MIA相比，对于ResNet18型号，Oslo在CIFAR100上实现的TPR在1 FP R下至少高出7$\x$，在0.1 FP R下至少高出22$\x$。我们评估了针对奥斯陆的多种防御机制。



## **9. Quantum Entanglement Path Selection and Qubit Allocation via Adversarial Group Neural Bandits**

对抗群神经盗贼的量子纠缠路径选择和量子位分配 quant-ph

Accepted by IEEE/ACM Transactions on Networking

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00316v1) [paper-pdf](http://arxiv.org/pdf/2411.00316v1)

**Authors**: Yin Huang, Lei Wang, Jie Xu

**Abstract**: Quantum Data Networks (QDNs) have emerged as a promising framework in the field of information processing and transmission, harnessing the principles of quantum mechanics. QDNs utilize a quantum teleportation technique through long-distance entanglement connections, encoding data information in quantum bits (qubits). Despite being a cornerstone in various quantum applications, quantum entanglement encounters challenges in establishing connections over extended distances due to probabilistic processes influenced by factors like optical fiber losses. The creation of long-distance entanglement connections between quantum computers involves multiple entanglement links and entanglement swapping techniques through successive quantum nodes, including quantum computers and quantum repeaters, necessitating optimal path selection and qubit allocation. Current research predominantly assumes known success rates of entanglement links between neighboring quantum nodes and overlooks potential network attackers. This paper addresses the online challenge of optimal path selection and qubit allocation, aiming to learn the best strategy for achieving the highest success rate of entanglement connections between two chosen quantum computers without prior knowledge of the success rate and in the presence of a QDN attacker. The proposed approach is based on multi-armed bandits, specifically adversarial group neural bandits, which treat each path as a group and view qubit allocation as arm selection. Our contributions encompass formulating an online adversarial optimization problem, introducing the EXPNeuralUCB bandits algorithm with theoretical performance guarantees, and conducting comprehensive simulations to showcase its superiority over established advanced algorithms.

摘要: 利用量子力学的原理，量子数据网络(QDNS)已经成为信息处理和传输领域的一个很有前途的框架。QDNS利用一种通过长距离纠缠连接的量子隐形传态技术，将数据信息编码为量子比特(Qbit)。尽管量子纠缠是各种量子应用的基石，但由于受光纤损耗等因素影响的概率过程，量子纠缠在建立远距离连接方面遇到了挑战。在量子计算机之间建立远距离纠缠连接涉及多个纠缠链路和通过包括量子计算机和量子中继器在内的连续量子节点的纠缠交换技术，这就需要最优路径选择和量子比特分配。目前的研究主要假设相邻量子节点之间纠缠链路的已知成功率，而忽略了潜在的网络攻击者。针对最优路径选择和量子比特分配的在线挑战，研究如何在不知道纠缠成功率的情况下，在量子网络攻击者的存在下，实现两台量子计算机间纠缠连接的最高成功率.该方法基于多臂强盗，特别是对抗性群体神经强盗，将每条路径视为一组，并将量子比特分配视为手臂选择。我们的贡献包括构建一个在线对抗性优化问题，引入具有理论性能保证的EXPNeuralUCB Bandits算法，并进行全面的模拟以展示其相对于已有的高级算法的优势。



## **10. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

10 pages, 6 figures, 2 tables. Version Update: Neurips Camera Ready

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2406.18451v3) [paper-pdf](http://arxiv.org/pdf/2406.18451v3)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate high margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively and confidently use the logit margin to detect brittle decisions with such models. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to assess adversarial vulnerability in deployment scenarios efficiently.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过在CIFAR10和CIFAR100数据集上对各种稳健训练模型的综合实证分析，我们表明它们表明了高边际一致性，并且它们的输入空间边际与Logit边际之间具有很强的相关性。然后，我们证明了我们可以有效和自信地使用Logit边际来检测这样的模型的脆性决策。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **11. Efficient Model Compression for Bayesian Neural Networks**

Bayesian神经网络的高效模型压缩 cs.LG

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00273v1) [paper-pdf](http://arxiv.org/pdf/2411.00273v1)

**Authors**: Diptarka Saha, Zihe Liu, Feng Liang

**Abstract**: Model Compression has drawn much attention within the deep learning community recently. Compressing a dense neural network offers many advantages including lower computation cost, deployability to devices of limited storage and memories, and resistance to adversarial attacks. This may be achieved via weight pruning or fully discarding certain input features. Here we demonstrate a novel strategy to emulate principles of Bayesian model selection in a deep learning setup. Given a fully connected Bayesian neural network with spike-and-slab priors trained via a variational algorithm, we obtain the posterior inclusion probability for every node that typically gets lost. We employ these probabilities for pruning and feature selection on a host of simulated and real-world benchmark data and find evidence of better generalizability of the pruned model in all our experiments.

摘要: 模型压缩最近引起了深度学习社区的广泛关注。压缩密集神经网络具有许多优势，包括较低的计算成本、可部署到有限存储和内存的设备以及抵抗对抗性攻击。这可以通过权重修剪或完全丢弃某些输入特征来实现。在这里，我们展示了一种新颖的策略，可以在深度学习设置中模拟Bayesian模型选择的原则。给定一个完全连接的Bayesian神经网络，其具有通过变分算法训练的尖峰和板先验，我们获得每个通常丢失的节点的后验包含概率。我们在大量模拟和现实世界的基准数据上使用这些概率进行修剪和特征选择，并在所有实验中找到修剪模型更好概括性的证据。



## **12. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **13. Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding**

使用预测编码保护前向网络免受对抗攻击 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00222v1) [paper-pdf](http://arxiv.org/pdf/2411.00222v1)

**Authors**: Ehsan Ganjidoost, Jeff Orchard

**Abstract**: An adversarial example is a modified input image designed to cause a Machine Learning (ML) model to make a mistake; these perturbations are often invisible or subtle to human observers and highlight vulnerabilities in a model's ability to generalize from its training data. Several adversarial attacks can create such examples, each with a different perspective, effectiveness, and perceptibility of changes. Conversely, defending against such adversarial attacks improves the robustness of ML models in image processing and other domains of deep learning. Most defence mechanisms require either a level of model awareness, changes to the model, or access to a comprehensive set of adversarial examples during training, which is impractical. Another option is to use an auxiliary model in a preprocessing manner without changing the primary model. This study presents a practical and effective solution -- using predictive coding networks (PCnets) as an auxiliary step for adversarial defence. By seamlessly integrating PCnets into feed-forward networks as a preprocessing step, we substantially bolster resilience to adversarial perturbations. Our experiments on MNIST and CIFAR10 demonstrate the remarkable effectiveness of PCnets in mitigating adversarial examples with about 82% and 65% improvements in robustness, respectively. The PCnet, trained on a small subset of the dataset, leverages its generative nature to effectively counter adversarial efforts, reverting perturbed images closer to their original forms. This innovative approach holds promise for enhancing the security and reliability of neural network classifiers in the face of the escalating threat of adversarial attacks.

摘要: 一个对抗性的例子是修改的输入图像，旨在导致机器学习(ML)模型出错；这些扰动对于人类观察者来说通常是不可见的或微妙的，并突显了模型从其训练数据进行泛化的能力中的弱点。几个对抗性攻击可以创建这样的例子，每个例子都具有不同的视角、有效性和对变化的感知能力。相反，防御这种敌意攻击提高了ML模型在图像处理和其他深度学习领域的稳健性。大多数防御机制要么需要一定程度的模型意识，要么需要更改模型，或者需要在训练期间获得一套全面的对抗性例子，这是不切实际的。另一种选择是在不改变主模型的情况下以预处理方式使用辅助模型。本研究提出了一种实用有效的解决方案--使用预测编码网络(PCnet)作为对抗防御的辅助步骤。通过将PCnet无缝地整合到前馈网络中作为一个预处理步骤，我们大大增强了对对手扰动的复原力。我们在MNIST和CIFAR10上的实验表明，PCnet在减少敌意例子方面具有显著的效果，健壮性分别提高了82%和65%。PCnet在数据集的一小部分上进行训练，利用其生成性来有效地对抗对手的努力，使受干扰的图像恢复到更接近其原始形式。这种创新的方法有望在面对不断升级的对抗性攻击威胁时增强神经网络分类器的安全性和可靠性。



## **14. I Can Hear You: Selective Robust Training for Deepfake Audio Detection**

我能听到你：Deepfake音频检测的选择性稳健训练 cs.SD

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00121v1) [paper-pdf](http://arxiv.org/pdf/2411.00121v1)

**Authors**: Zirui Zhang, Wei Hao, Aroon Sankoh, William Lin, Emanuel Mendiola-Ortiz, Junfeng Yang, Chengzhi Mao

**Abstract**: Recent advances in AI-generated voices have intensified the challenge of detecting deepfake audio, posing risks for scams and the spread of disinformation. To tackle this issue, we establish the largest public voice dataset to date, named DeepFakeVox-HQ, comprising 1.3 million samples, including 270,000 high-quality deepfake samples from 14 diverse sources. Despite previously reported high accuracy, existing deepfake voice detectors struggle with our diversely collected dataset, and their detection success rates drop even further under realistic corruptions and adversarial attacks. We conduct a holistic investigation into factors that enhance model robustness and show that incorporating a diversified set of voice augmentations is beneficial. Moreover, we find that the best detection models often rely on high-frequency features, which are imperceptible to humans and can be easily manipulated by an attacker. To address this, we propose the F-SAT: Frequency-Selective Adversarial Training method focusing on high-frequency components. Empirical results demonstrate that using our training dataset boosts baseline model performance (without robust training) by 33%, and our robust training further improves accuracy by 7.7% on clean samples and by 29.3% on corrupted and attacked samples, over the state-of-the-art RawNet3 model.

摘要: 人工智能产生的声音的最新进展加剧了检测深度虚假音频的挑战，给诈骗和虚假信息的传播带来了风险。为了解决这个问题，我们建立了迄今为止最大的公共语音数据集DeepFakeVox-HQ，包含130万个样本，其中包括来自14个不同来源的27万个高质量深伪样本。尽管先前报道的准确率很高，但现有的深度假语音检测器难以处理我们多样化收集的数据集，在现实的腐败和对手攻击下，它们的检测成功率甚至会进一步下降。我们对增强模型稳健性的因素进行了全面的调查，并表明结合不同的语音增强集是有益的。此外，我们发现，最好的检测模型往往依赖于高频特征，这些特征对人类来说是不可察觉的，很容易被攻击者操纵。为了解决这个问题，我们提出了F-SAT：以高频成分为重点的频率选择性对抗性训练方法。实验结果表明，与最先进的RawNet3模型相比，使用我们的训练数据集可以将基线模型的性能提高33%(没有健壮的训练)，并且我们的健壮的训练在干净样本上的准确率进一步提高了7.7%，在被破坏和攻击的样本上提高了29.3%。



## **15. Untelegraphable Encryption and its Applications**

不可电报加密及其应用 quant-ph

55 pages

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24189v1) [paper-pdf](http://arxiv.org/pdf/2410.24189v1)

**Authors**: Jeffrey Champion, Fuyuki Kitagawa, Ryo Nishimaki, Takashi Yamakawa

**Abstract**: We initiate the study of untelegraphable encryption (UTE), founded on the no-telegraphing principle, which allows an encryptor to encrypt a message such that a binary string representation of the ciphertext cannot be decrypted by a user with the secret key, a task that is classically impossible. This is a natural relaxation of unclonable encryption, inspired by the recent work of Nehoran and Zhandry (ITCS 2024), who showed a computational separation between the no-cloning and no-telegraphing principles. In this work, we define and construct UTE information-theoretically in the plain model. Building off this, we give several applications of UTE and study the interplay of UTE with UE and well-studied tasks in quantum state learning, yielding the following contributions:   - A construction of collusion-resistant UTE from standard secret-key encryption. We additionally show that hyper-efficient shadow tomography (HEST) is impossible assuming collusion-resistant UTE exists. By considering a relaxation of collusion-resistant UTE, we are able to show the impossibility of HEST assuming only pseudorandom state generators (which may not imply one-way functions). This almost completely answers an open inquiry of Aaronson (STOC 2019).   - A construction of UTE from a one-shot message authentication code in the classical oracle model, such that there is an explicit attack that breaks UE security for an unbounded polynomial number of decryptors.   - A construction of everlasting secure collusion-resistant UTE, where the decryptor adversary can run in unbounded time, in the quantum random oracle model (QROM), and formal evidence that a construction in the plain model is a challenging task. We additionally show that HEST with unbounded post-processing time is impossible in the QROM.   - Constructions (and definitions) of untelegraphable secret sharing and untelegraphable functional encryption.

摘要: 我们启动了基于无电报原理的不可远程传送加密(UTE)的研究，该原理允许加密者对消息进行加密，使得密文的二进制字符串表示不能被用户用秘密密钥解密，这是传统上不可能完成的任务。这是不可克隆加密的自然放松，灵感来自Nehoran和Zhandry最近的工作(ITCS 2024)，他们展示了无克隆和无电报原则之间的计算分离。在这项工作中，我们定义并构造了UTE信息--理论上是在平面模型中。在此基础上，我们给出了UTE的几个应用，并研究了UE与UE的相互作用以及量子态学习中已有的研究任务，取得了以下贡献：-从标准密钥加密构造了抗合谋的UTE。此外，我们还表明，假设存在抗合谋的UTE，超高效阴影层析成像(HEST)是不可能的。通过考虑抗合谋UTE的松弛，我们能够证明仅假设伪随机状态生成器(这可能不包含单向函数)的HEST是不可能的。这几乎完全回答了对Aaronson(STOC 2019)的公开调查。-从经典Oracle模型中的一次性消息认证码构造UE，从而存在破坏无限多项式解密器的UE安全的显式攻击。-在量子随机预言模型(QROM)中构造永久安全的抗合谋UTE，其中解密者对手可以在无限的时间内运行，并且形式证据表明在普通模型中的构造是一项具有挑战性的任务。此外，我们还证明了后处理时间无界的HEST在QROM中是不可能的。-不可远程传送秘密共享和不可远程传送功能加密的构造(和定义)。



## **16. Unveiling Synthetic Faces: How Synthetic Datasets Can Expose Real Identities**

揭开合成面孔：合成数据集如何暴露真实身份 cs.CV

Accepted in NeurIPS 2024 Workshop on New Frontiers in Adversarial  Machine Learning

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24015v1) [paper-pdf](http://arxiv.org/pdf/2410.24015v1)

**Authors**: Hatef Otroshi Shahreza, Sébastien Marcel

**Abstract**: Synthetic data generation is gaining increasing popularity in different computer vision applications. Existing state-of-the-art face recognition models are trained using large-scale face datasets, which are crawled from the Internet and raise privacy and ethical concerns. To address such concerns, several works have proposed generating synthetic face datasets to train face recognition models. However, these methods depend on generative models, which are trained on real face images. In this work, we design a simple yet effective membership inference attack to systematically study if any of the existing synthetic face recognition datasets leak any information from the real data used to train the generator model. We provide an extensive study on 6 state-of-the-art synthetic face recognition datasets, and show that in all these synthetic datasets, several samples from the original real dataset are leaked. To our knowledge, this paper is the first work which shows the leakage from training data of generator models into the generated synthetic face recognition datasets. Our study demonstrates privacy pitfalls in synthetic face recognition datasets and paves the way for future studies on generating responsible synthetic face datasets.

摘要: 合成数据生成在不同的计算机视觉应用中越来越受欢迎。现有的最先进的人脸识别模型是使用大规模的人脸数据集进行训练的，这些数据集是从互联网上爬行的，会引起隐私和伦理问题。为了解决这些问题，一些工作已经提出生成合成人脸数据集来训练人脸识别模型。然而，这些方法依赖于生成模型，而生成模型是在真实人脸图像上训练的。在这项工作中，我们设计了一个简单而有效的隶属度推理攻击来系统地研究现有的合成人脸识别数据集是否会从用于训练生成器模型的真实数据中泄漏任何信息。我们在6个最先进的合成人脸识别数据集上进行了广泛的研究，结果表明，在所有这些合成数据集中，有几个来自原始真实数据集的样本是泄漏的。据我们所知，本文是第一次将生成器模型的训练数据泄漏到生成的合成人脸识别数据集。我们的研究展示了合成人脸识别数据集的隐私陷阱，并为未来生成负责任的合成人脸数据集的研究铺平了道路。



## **17. DiffPAD: Denoising Diffusion-based Adversarial Patch Decontamination**

迪夫pad：消除基于扩散的对抗性补丁净化 cs.CV

Accepted to 2025 IEEE/CVF Winter Conference on Applications of  Computer Vision (WACV)

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.24006v1) [paper-pdf](http://arxiv.org/pdf/2410.24006v1)

**Authors**: Jia Fu, Xiao Zhang, Sepideh Pashami, Fatemeh Rahimian, Anders Holst

**Abstract**: In the ever-evolving adversarial machine learning landscape, developing effective defenses against patch attacks has become a critical challenge, necessitating reliable solutions to safeguard real-world AI systems. Although diffusion models have shown remarkable capacity in image synthesis and have been recently utilized to counter $\ell_p$-norm bounded attacks, their potential in mitigating localized patch attacks remains largely underexplored. In this work, we propose DiffPAD, a novel framework that harnesses the power of diffusion models for adversarial patch decontamination. DiffPAD first performs super-resolution restoration on downsampled input images, then adopts binarization, dynamic thresholding scheme and sliding window for effective localization of adversarial patches. Such a design is inspired by the theoretically derived correlation between patch size and diffusion restoration error that is generalized across diverse patch attack scenarios. Finally, DiffPAD applies inpainting techniques to the original input images with the estimated patch region being masked. By integrating closed-form solutions for super-resolution restoration and image inpainting into the conditional reverse sampling process of a pre-trained diffusion model, DiffPAD obviates the need for text guidance or fine-tuning. Through comprehensive experiments, we demonstrate that DiffPAD not only achieves state-of-the-art adversarial robustness against patch attacks but also excels in recovering naturalistic images without patch remnants.

摘要: 在不断发展的对抗性机器学习环境中，开发针对补丁攻击的有效防御已成为一项关键挑战，需要可靠的解决方案来保护真实世界的AI系统。虽然扩散模型在图像合成方面表现出了显著的能力，并且最近已被用于对抗$\ell_p$-范数有界攻击，但它们在缓解局部补丁攻击方面的潜力仍未被充分挖掘。在这项工作中，我们提出了DiffPAD，一个新的框架，它利用扩散模型的力量来进行对抗性补丁去污。DiffPAD首先对下采样的输入图像进行超分辨率恢复，然后采用二值化、动态阈值和滑动窗口等方法对对抗性斑块进行有效定位。这种设计的灵感来自于理论上推导出的补丁大小和扩散恢复误差之间的相关性，该相关性在不同的补丁攻击场景中得到推广。最后，DiffPAD将修复技术应用于原始输入图像，并对估计的补丁区域进行掩蔽。通过将用于超分辨率恢复和图像修复的闭合形式解决方案集成到预先训练的扩散模型的条件反向采样过程中，DiffPAD消除了对文本指导或微调的需要。通过综合实验，我们证明了DiffPAD算法不仅对补丁攻击具有最好的对抗健壮性，而且在恢复没有补丁残留的自然图像方面具有很好的性能。



## **18. Meta-Learning Approaches for Improving Detection of Unseen Speech Deepfakes**

用于改进不可见语音Deepfakes检测的元学习方法 eess.AS

6 pages, accepted to the IEEE Spoken Language Technology Workshop  (SLT) 2024

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.20578v2) [paper-pdf](http://arxiv.org/pdf/2410.20578v2)

**Authors**: Ivan Kukanov, Janne Laakkonen, Tomi Kinnunen, Ville Hautamäki

**Abstract**: Current speech deepfake detection approaches perform satisfactorily against known adversaries; however, generalization to unseen attacks remains an open challenge. The proliferation of speech deepfakes on social media underscores the need for systems that can generalize to unseen attacks not observed during training. We address this problem from the perspective of meta-learning, aiming to learn attack-invariant features to adapt to unseen attacks with very few samples available. This approach is promising since generating of a high-scale training dataset is often expensive or infeasible. Our experiments demonstrated an improvement in the Equal Error Rate (EER) from 21.67% to 10.42% on the InTheWild dataset, using just 96 samples from the unseen dataset. Continuous few-shot adaptation ensures that the system remains up-to-date.

摘要: 当前的语音深度伪造检测方法对已知对手的表现令人满意;然而，对不可见攻击的概括仍然是一个悬而未决的挑战。社交媒体上语音深度造假的激增凸显了对能够概括训练期间未观察到的不可见攻击的系统的需求。我们从元学习的角度解决这个问题，旨在学习攻击不变的特征，以适应使用很少的样本的不可见的攻击。这种方法很有希望，因为生成大规模训练数据集通常昂贵或不可行。我们的实验表明，仅使用未见过数据集的96个样本，InTheWild数据集的等错误率（EER）从21.67%提高到10.42%。连续的少量镜头调整确保系统保持最新状态。



## **19. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2402.06255v4) [paper-pdf](http://arxiv.org/pdf/2402.06255v4)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreaking attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly focusing on model fine-tuning or heuristical defense designs. However, how to achieve intrinsic robustness through prompt optimization remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0%, while maintaining the model's utility on the benign task and incurring only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/PKU-ML/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在模型微调或启发式防御设计上。然而，如何通过快速优化来获得内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对灰盒和黑盒攻击都是有效的，将高级攻击的成功率降低到近0%，同时保持了模型在良性任务上的效用，并且只产生了可以忽略的计算开销，为LLM安全的进一步研究开辟了新的视角。我们的代码可以在https://github.com/PKU-ML/PAT.上找到



## **20. Noise as a Double-Edged Sword: Reinforcement Learning Exploits Randomized Defenses in Neural Networks**

噪音作为双刃剑：强化学习利用神经网络中的随机防御 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23870v1) [paper-pdf](http://arxiv.org/pdf/2410.23870v1)

**Authors**: Steve Bakos, Pooria Madani, Heidar Davoudi

**Abstract**: This study investigates a counterintuitive phenomenon in adversarial machine learning: the potential for noise-based defenses to inadvertently aid evasion attacks in certain scenarios. While randomness is often employed as a defensive strategy against adversarial examples, our research reveals that this approach can sometimes backfire, particularly when facing adaptive attackers using reinforcement learning (RL). Our findings show that in specific cases, especially with visually noisy classes, the introduction of noise in the classifier's confidence values can be exploited by the RL attacker, leading to a significant increase in evasion success rates. In some instances, the noise-based defense scenario outperformed other strategies by up to 20\% on a subset of classes. However, this effect was not consistent across all classifiers tested, highlighting the complexity of the interaction between noise-based defenses and different models. These results suggest that in some cases, noise-based defenses can inadvertently create an adversarial training loop beneficial to the RL attacker. Our study emphasizes the need for a more nuanced approach to defensive strategies in adversarial machine learning, particularly in safety-critical applications. It challenges the assumption that randomness universally enhances defense against evasion attacks and highlights the importance of considering adaptive, RL-based attackers when designing robust defense mechanisms.

摘要: 这项研究调查了对抗性机器学习中的一个违反直觉的现象：在某些情况下，基于噪声的防御可能会无意中帮助逃避攻击。虽然随机性经常被用作对抗对手示例的防御策略，但我们的研究表明，这种方法有时会适得其反，特别是在面对使用强化学习(RL)的自适应攻击者时。我们的研究结果表明，在特定情况下，特别是在视觉噪声类的情况下，在分类器的置信度中引入噪声可以被RL攻击者利用，从而显著提高规避成功率。在某些情况下，基于噪声的防御方案在类的子集上的性能比其他策略高出20%。然而，这种效果在所有测试的分类器中并不一致，这突显了基于噪声的防御和不同模型之间相互作用的复杂性。这些结果表明，在某些情况下，基于噪声的防御可能会无意中产生对RL攻击者有利的对抗性训练循环。我们的研究强调了在对抗性机器学习中，特别是在安全关键应用中，需要对防御策略采取更细微的方法。它挑战了随机性普遍增强对逃避攻击的防御的假设，并强调了在设计稳健的防御机制时考虑自适应的、基于RL的攻击者的重要性。



## **21. Rapid Plug-in Defenders**

快速插入式防守者 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2306.01762v4) [paper-pdf](http://arxiv.org/pdf/2306.01762v4)

**Authors**: Kai Wu, Yujian Betterest Li, Jian Lou, Xiaoyu Zhang, Handing Wang, Jing Liu

**Abstract**: In the realm of daily services, the deployment of deep neural networks underscores the paramount importance of their reliability. However, the vulnerability of these networks to adversarial attacks, primarily evasion-based, poses a concerning threat to their functionality. Common methods for enhancing robustness involve heavy adversarial training or leveraging learned knowledge from clean data, both necessitating substantial computational resources. This inherent time-intensive nature severely limits the agility of large foundational models to swiftly counter adversarial perturbations. To address this challenge, this paper focuses on the Rapid Plug-in Defender (RaPiD) problem, aiming to rapidly counter adversarial perturbations without altering the deployed model. Drawing inspiration from the generalization and the universal computation ability of pre-trained transformer models, we propose a novel method termed CeTaD (Considering Pre-trained Transformers as Defenders) for RaPiD, optimized for efficient computation. CeTaD strategically fine-tunes the normalization layer parameters within the defender using a limited set of clean and adversarial examples. Our evaluation centers on assessing CeTaD's effectiveness, transferability, and the impact of different components in scenarios involving one-shot adversarial examples. The proposed method is capable of rapidly adapting to various attacks and different application scenarios without altering the target model and clean training data. We also explore the influence of varying training data conditions on CeTaD's performance. Notably, CeTaD exhibits adaptability across differentiable service models and proves the potential of continuous learning.

摘要: 在日常服务领域，深度神经网络的部署突显了其可靠性的至高无上的重要性。然而，这些网络对对手攻击的脆弱性，主要是基于逃避的攻击，对其功能构成了令人担忧的威胁。增强健壮性的常见方法包括大量对抗性训练或利用从干净数据中学到的知识，这两种方法都需要大量的计算资源。这种固有的时间密集性严重限制了大型基础模型快速对抗对抗性扰动的敏捷性。为了应对这一挑战，本文重点研究了快速插件防御(Rapid Plug-in Defender，简称RAPID)问题，目的是在不改变已部署模型的情况下，快速应对对抗性扰动。借鉴预先训练的变压器模型的泛化能力和通用计算能力，提出了一种新的快速、优化的计算方法CeTaD(Serving Pre-Traded Transformers as Defders)。CeTaD战略性地微调了防御者内部的规格化层参数，使用了有限的一组干净的和对抗性的例子。我们的评估重点是评估CeTaD的有效性、可转移性，以及在包含单发对抗性例子的场景中不同组件的影响。该方法能够在不改变目标模型和干净训练数据的情况下，快速适应各种攻击和不同的应用场景。我们还探讨了不同的训练数据条件对CeTaD性能的影响。值得注意的是，CeTaD展示了跨不同服务模型的适应性，并证明了持续学习的潜力。



## **22. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Dataset & Benchmarking Track

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23746v1) [paper-pdf](http://arxiv.org/pdf/2410.23746v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating advanced prompt usages, human revisions like word substitutions, and writing errors. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟高级提示用法、人工修改(如单词替换)和书写错误。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **23. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

验证模型的一个提示：通过不可传输对抗性攻击进行黑匣子文本到图像模型验证 cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.22725v2) [paper-pdf](http://arxiv.org/pdf/2410.22725v2)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li

**Abstract**: Recently, the success of Text-to-Image (T2I) models has led to the rise of numerous third-party platforms, which claim to provide cheaper API services and more flexibility in model options. However, this also raises a new security concern: Are these third-party services truly offering the models they claim? To address this problem, we propose the first T2I model verification method named Text-to-Image Model Verification via Non-Transferable Adversarial Attacks (TVN). The non-transferability of adversarial examples means that these examples are only effective on a target model and ineffective on other models, thereby allowing for the verification of the target model. TVN utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating non-transferable adversarial prompts. By calculating the CLIP-text scores between the non-transferable adversarial prompts without perturbations and the images, we can verify if the model matches the claimed target model, based on a 3-sigma threshold. The experiments showed that TVN performed well in both closed-set and open-set scenarios, achieving a verification accuracy of over 90\%. Moreover, the adversarial prompts generated by TVN significantly reduced the CLIP-text scores of the target model, while having little effect on other models.

摘要: 最近，文本到图像(T2I)模式的成功导致了无数第三方平台的崛起，这些平台声称提供更便宜的API服务和更灵活的模式选择。然而，这也引发了一个新的安全问题：这些第三方服务是否真的提供了它们声称的模式？针对这一问题，我们提出了第一种T2I模型验证方法--基于不可转移攻击的文本到图像模型验证方法(TVN)。对抗性实例的不可转移性意味着这些实例仅对目标模型有效，而对其他模型无效，从而允许对目标模型进行验证。TVN使用非支配排序遗传算法II(NSGA-II)来优化提示文本编码的余弦相似度，生成不可转移的对抗性提示。通过计算不可转移的敌意提示与图像之间的剪贴文本分数，我们可以基于3-sigma阈值来验证该模型是否与所声称的目标模型匹配。实验表明，TVN在闭集和开集场景下都表现良好，验证准确率达到90%以上。此外，TVN生成的对抗性提示显著降低了目标模型的片段文本分数，而对其他模型影响不大。



## **24. The Influence of Ridership Weighting on Targeting and Recovery Strategies for Urban Rail Rapid Transit Systems**

乘客权重对城市轨道快速交通系统目标和恢复策略的影响 physics.soc-ph

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23688v1) [paper-pdf](http://arxiv.org/pdf/2410.23688v1)

**Authors**: Aran Chakraborty, Yushi Tsukimoto, August Posch, Jack Watson, Auroop Ganguly

**Abstract**: The resilience of urban rapid transit systems (URTs) to a rapidly evolving threat space is of much concern. Extreme rainfall events are both intensifying and growing more frequent under continuing climate change, exposing transit systems to flooding, while cyber threats and emerging technologies such as unmanned aerial vehicles are exposing such systems to targeted disruptions. An imperative has emerged to model how networked infrastructure systems fail and devise strategies to efficiently recover from disruptions. Passenger flow approaches can quantify more dimensions of resilience than network science-based approaches, but the former typically requires granular data from automatic fare collection and suffers from large runtime complexities. Some attempts have been made to include accessible low-resolution ridership data in topological frameworks. However, there is yet to be a systematic investigation of the effects of incorporating low-dimensional, coarsely-averaged ridership volume into topological network science methodologies. We simulate targeted attack and recovery sequences using station-level ridership, four centrality measures, and weighted combinations thereof. Resilience is quantified using two topological measures of performance: the node count of a network's giant connected component (GCC), and a new measure termed the "highest ridership connected component" (HRCC). Three transit systems are used as case studies: the subways of Boston, New York, and Osaka. Results show that centrality-based strategies are most effective when measuring performance via GCC, while centrality-ridership hybrid strategies perform strongest by HRCC. We show that the most effective strategies vary by network characteristics and the mission goals of emergency managers, highlighting the need to plan for strategic adversaries and rapid recovery according to each city's unique needs.

摘要: 城市快速交通系统(URT)对快速发展的威胁空间的适应能力是非常令人担忧的。在持续的气候变化下，极端降雨事件正在加剧，并变得越来越频繁，使交通系统面临洪水，而网络威胁和无人机等新兴技术正使此类系统面临有针对性的中断。建立联网基础设施系统故障的模型，并设计有效地从中断中恢复的战略，已成为当务之急。与基于网络科学的方法相比，客流方法可以量化更多的弹性维度，但前者通常需要来自自动票价收集的细粒度数据，并且存在较大的运行复杂性。已经进行了一些尝试，以将可访问的低分辨率乘客数据纳入拓扑框架。然而，将低维、粗略平均的客运量纳入到拓扑网络科学方法中的影响还没有进行系统的调查。我们使用站级乘员、四个中心性度量及其加权组合来模拟有针对性的攻击和恢复序列。弹性是使用两个拓扑性能度量来量化的：网络的巨型连通部件的节点数(GCC)和一个新的被称为“最高载客量连通部件”(HRCC)的度量。三个交通系统被用作案例研究：波士顿、纽约和大阪的地铁。结果表明，基于中心性的策略在通过GCC衡量绩效时最有效，而中心性-乘车混合策略在HRCC的绩效衡量中表现最好。我们表明，最有效的战略因网络特征和应急管理者的任务目标而异，强调需要根据每个城市的独特需求规划战略对手和快速恢复。



## **25. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

过去10年视觉任务的对抗性攻击：一项调查 cs.CV

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23687v1) [paper-pdf](http://arxiv.org/pdf/2410.23687v1)

**Authors**: Chiyu Zhang, Xiaogang Xu, Jiafei Wu, Zhe Liu, Lu Zhou

**Abstract**: Adversarial attacks, which manipulate input data to undermine model availability and integrity, pose significant security threats during machine learning inference. With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreak techniques, have emerged. Understanding these attacks is crucial for developing more robust systems and demystifying the inner workings of neural networks. However, existing reviews often focus on attack classifications and lack comprehensive, in-depth analysis. The research community currently needs: 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations of existing methods; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.

摘要: 对抗性攻击通过操纵输入数据来破坏模型的可用性和完整性，在机器学习推理过程中会造成严重的安全威胁。随着大型视觉语言模型的出现，新的攻击载体出现了，如认知偏差、快速注入和越狱技术。了解这些攻击对于开发更强大的系统和揭开神经网络内部工作的神秘面纱至关重要。然而，现有的审查往往侧重于攻击分类，缺乏全面、深入的分析。研究界目前需要：1)对对抗性、可转移性和泛化的统一见解；2)对现有方法的详细评估；3)动机驱动的攻击分类；以及4)对传统攻击和LVLM攻击的综合视角。本文对传统攻击和LVLM攻击进行了全面的总结，强调了它们之间的联系和区别，并为未来的研究提供了可操作的见解，从而解决了这些差距。



## **26. Pseudo-Conversation Injection for LLM Goal Hijacking**

LLM目标劫持的伪对话注入 cs.CL

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2410.23678v1) [paper-pdf](http://arxiv.org/pdf/2410.23678v1)

**Authors**: Zheng Chen, Buhui Yao

**Abstract**: Goal hijacking is a type of adversarial attack on Large Language Models (LLMs) where the objective is to manipulate the model into producing a specific, predetermined output, regardless of the user's original input. In goal hijacking, an attacker typically appends a carefully crafted malicious suffix to the user's prompt, which coerces the model into ignoring the user's original input and generating the target response. In this paper, we introduce a novel goal hijacking attack method called Pseudo-Conversation Injection, which leverages the weaknesses of LLMs in role identification within conversation contexts. Specifically, we construct the suffix by fabricating responses from the LLM to the user's initial prompt, followed by a prompt for a malicious new task. This leads the model to perceive the initial prompt and fabricated response as a completed conversation, thereby executing the new, falsified prompt. Following this approach, we propose three Pseudo-Conversation construction strategies: Targeted Pseudo-Conversation, Universal Pseudo-Conversation, and Robust Pseudo-Conversation. These strategies are designed to achieve effective goal hijacking across various scenarios. Our experiments, conducted on two mainstream LLM platforms including ChatGPT and Qwen, demonstrate that our proposed method significantly outperforms existing approaches in terms of attack effectiveness.

摘要: 目标劫持是一种针对大型语言模型(LLM)的对抗性攻击，其目标是操纵模型生成特定的、预定的输出，而不考虑用户的原始输入。在目标劫持中，攻击者通常会在用户提示后附加精心编制的恶意后缀，这会迫使模型忽略用户的原始输入并生成目标响应。本文提出了一种新的目标劫持攻击方法--伪会话注入，该方法利用了LLMS在会话上下文中角色识别方面的弱点。具体地说，我们构造后缀的方法是从LLM构造对用户初始提示的响应，然后是恶意新任务的提示。这导致模型将初始提示和捏造的响应视为完成的对话，从而执行新的、伪造的提示。在此基础上，我们提出了三种伪会话构建策略：目标伪会话、通用伪会话和健壮伪会话。这些策略旨在实现跨各种场景的有效目标劫持。我们在ChatGPT和Qwen两个主流LLM平台上进行的实验表明，我们提出的方法在攻击效率方面明显优于现有方法。



## **27. Adversarial Attacks on Code Models with Discriminative Graph Patterns**

对具有区分图模式的代码模型的对抗攻击 cs.SE

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2308.11161v2) [paper-pdf](http://arxiv.org/pdf/2308.11161v2)

**Authors**: Thanh-Dat Nguyen, Yang Zhou, Xuan Bach D. Le, Patanamon Thongtanunam, David Lo

**Abstract**: Pre-trained language models of code are now widely used in various software engineering tasks such as code generation, code completion, vulnerability detection, etc. This, in turn, poses security and reliability risks to these models. One of the important threats is \textit{adversarial attacks}, which can lead to erroneous predictions and largely affect model performance on downstream tasks. Current adversarial attacks on code models usually adopt fixed sets of program transformations, such as variable renaming and dead code insertion, leading to limited attack effectiveness. To address the aforementioned challenges, we propose a novel adversarial attack framework, GraphCodeAttack, to better evaluate the robustness of code models. Given a target code model, GraphCodeAttack automatically mines important code patterns, which can influence the model's decisions, to perturb the structure of input code to the model. To do so, GraphCodeAttack uses a set of input source codes to probe the model's outputs and identifies the \textit{discriminative} ASTs patterns that can influence the model decisions. GraphCodeAttack then selects appropriate AST patterns, concretizes the selected patterns as attacks, and inserts them as dead code into the model's input program. To effectively synthesize attacks from AST patterns, GraphCodeAttack uses a separate pre-trained code model to fill in the ASTs with concrete code snippets. We evaluate the robustness of two popular code models (e.g., CodeBERT and GraphCodeBERT) against our proposed approach on three tasks: Authorship Attribution, Vulnerability Prediction, and Clone Detection. The experimental results suggest that our proposed approach significantly outperforms state-of-the-art approaches in attacking code models such as CARROT and ALERT.

摘要: 预先训练的代码语言模型现在被广泛用于各种软件工程任务，如代码生成、代码完成、漏洞检测等。这反过来又给这些模型带来了安全和可靠性风险。其中一个重要的威胁是对抗性攻击，它会导致错误的预测，并在很大程度上影响模型在下游任务上的性能。当前针对代码模型的对抗性攻击通常采用固定的程序转换集，如变量重命名和死代码插入，导致攻击效果有限。为了应对上述挑战，我们提出了一种新的对抗性攻击框架GraphCodeAttack，以更好地评估代码模型的健壮性。在给定目标代码模型的情况下，GraphCodeAttack自动挖掘可能影响模型决策的重要代码模式，以扰乱模型的输入代码结构。为此，GraphCodeAttack使用一组输入源代码来探测模型的输出，并识别可能影响模型决策的\textit{鉴别性}ASTS模式。然后，GraphCodeAttack选择适当的AST模式，将所选模式具体化为攻击，并将它们作为死代码插入到模型的输入程序中。为了有效地从AST模式合成攻击，GraphCodeAttack使用单独的预先训练的代码模型来用具体的代码片段填充AST。我们评估了两个流行的代码模型(例如，CodeBERT和GraphCodeBERT)在作者属性、漏洞预测和克隆检测三个任务上的健壮性。实验结果表明，我们提出的方法在攻击胡萝卜和ALERT等代码模型方面明显优于最先进的方法。



## **28. Keep on Swimming: Real Attackers Only Need Partial Knowledge of a Multi-Model System**

继续游泳：真正的攻击者只需要对多模型系统的部分了解 cs.LG

11 pages, 2 figures

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23483v1) [paper-pdf](http://arxiv.org/pdf/2410.23483v1)

**Authors**: Julian Collado, Kevin Stangl

**Abstract**: Recent approaches in machine learning often solve a task using a composition of multiple models or agentic architectures. When targeting a composed system with adversarial attacks, it might not be computationally or informationally feasible to train an end-to-end proxy model or a proxy model for every component of the system. We introduce a method to craft an adversarial attack against the overall multi-model system when we only have a proxy model for the final black-box model, and when the transformation applied by the initial models can make the adversarial perturbations ineffective. Current methods handle this by applying many copies of the first model/transformation to an input and then re-use a standard adversarial attack by averaging gradients, or learning a proxy model for both stages. To our knowledge, this is the first attack specifically designed for this threat model and our method has a substantially higher attack success rate (80% vs 25%) and contains 9.4% smaller perturbations (MSE) compared to prior state-of-the-art methods. Our experiments focus on a supervised image pipeline, but we are confident the attack will generalize to other multi-model settings [e.g. a mix of open/closed source foundation models], or agentic systems

摘要: 最近机器学习中的方法通常使用多个模型或代理体系结构的组合来解决任务。当以具有对抗性攻击的组合系统为目标时，为系统的每个组件训练端到端代理模型或代理模型在计算或信息上可能是不可行的。我们介绍了一种方法，当我们只有最终黑盒模型的代理模型时，并且初始模型所应用的变换会使对抗性扰动无效时，对整个多模型系统进行对抗性攻击。当前的方法通过将第一模型/变换的许多副本应用于输入，然后通过平均梯度或学习用于两个阶段的代理模型来重复使用标准的对抗性攻击来处理这一问题。据我们所知，这是专门为该威胁模型设计的第一次攻击，与现有最先进的方法相比，我们的方法具有显著更高的攻击成功率(80%比25%)，并且包含9.4%的较小扰动(MSE)。我们的实验集中在有监督的图像管道上，但我们相信攻击将推广到其他多模型设置[例如，开放/封闭源代码基础模型的混合]或代理系统



## **29. Breach By A Thousand Leaks: Unsafe Information Leakage in `Safe' AI Responses**

千次泄密：“安全”人工智能响应中不安全的信息泄露 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2407.02551v2) [paper-pdf](http://arxiv.org/pdf/2407.02551v2)

**Authors**: David Glukhov, Ziwen Han, Ilia Shumailov, Vardan Papyan, Nicolas Papernot

**Abstract**: Vulnerability of Frontier language models to misuse and jailbreaks has prompted the development of safety measures like filters and alignment training in an effort to ensure safety through robustness to adversarially crafted prompts. We assert that robustness is fundamentally insufficient for ensuring safety goals, and current defenses and evaluation methods fail to account for risks of dual-intent queries and their composition for malicious goals. To quantify these risks, we introduce a new safety evaluation framework based on impermissible information leakage of model outputs and demonstrate how our proposed question-decomposition attack can extract dangerous knowledge from a censored LLM more effectively than traditional jailbreaking. Underlying our proposed evaluation method is a novel information-theoretic threat model of inferential adversaries, distinguished from security adversaries, such as jailbreaks, in that success is measured by inferring impermissible knowledge from victim outputs as opposed to forcing explicitly impermissible outputs from the victim. Through our information-theoretic framework, we show that to ensure safety against inferential adversaries, defense mechanisms must ensure information censorship, bounding the leakage of impermissible information. However, we prove that such defenses inevitably incur a safety-utility trade-off.

摘要: Frontier Language模型对误用和越狱的脆弱性促使了过滤器和对齐训练等安全措施的开发，以努力通过对恶意创建的提示的健壮性来确保安全。我们断言，健壮性从根本上不足以确保安全目标，并且当前的防御和评估方法未能考虑到双重意图查询及其恶意目标的组合的风险。为了量化这些风险，我们引入了一种新的安全评估框架，该框架基于模型输出的不允许信息泄漏，并展示了我们提出的问题分解攻击如何比传统越狱更有效地从删失的LLM中提取危险知识。我们提出的评估方法的基础是一个新的推论对手的信息论威胁模型，该模型不同于安全对手，例如越狱，因为成功的衡量标准是从受害者的输出推断不允许的知识，而不是强迫受害者提供明确不允许的输出。通过我们的信息论框架，我们表明，为了确保针对推理对手的安全，防御机制必须确保信息审查，限制不允许信息的泄露。然而，我们证明，这种防御不可避免地会招致安全和公用事业之间的权衡。



## **30. FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training**

FAIR-STAT：使用有针对性的对抗训练提高模型公平性 cs.LG

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23142v1) [paper-pdf](http://arxiv.org/pdf/2410.23142v1)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Deep neural networks are susceptible to adversarial attacks and common corruptions, which undermine their robustness. In order to enhance model resilience against such challenges, Adversarial Training (AT) has emerged as a prominent solution. Nevertheless, adversarial robustness is often attained at the expense of model fairness during AT, i.e., disparity in class-wise robustness of the model. While distinctive classes become more robust towards such adversaries, hard to detect classes suffer. Recently, research has focused on improving model fairness specifically for perturbed images, overlooking the accuracy of the most likely non-perturbed data. Additionally, despite their robustness against the adversaries encountered during model training, state-of-the-art adversarial trained models have difficulty maintaining robustness and fairness when confronted with diverse adversarial threats or common corruptions. In this work, we address the above concerns by introducing a novel approach called Fair Targeted Adversarial Training (FAIR-TAT). We show that using targeted adversarial attacks for adversarial training (instead of untargeted attacks) can allow for more favorable trade-offs with respect to adversarial fairness. Empirical results validate the efficacy of our approach.

摘要: 深度神经网络容易受到敌意攻击和常见的腐败，这削弱了它们的健壮性。为了提高模型对这些挑战的韧性，对抗性训练(AT)已成为一个突出的解决方案。然而，在AT过程中，对抗的稳健性往往是以牺牲模型的公平性为代价的，即模型的类稳健性存在差异。虽然独特的职业对这样的对手变得更加强大，但很难检测到的职业会受到影响。最近，研究集中于改善模型的公平性，特别是对于扰动的图像，忽略了最可能的非扰动数据的准确性。此外，尽管它们对模型训练过程中遇到的对手具有健壮性，但是最新的对抗性训练模型在面对不同的对手威胁或常见的腐败时很难保持健壮性和公平性。在这项工作中，我们通过引入一种名为公平目标对抗性训练(Fair-TAT)的新方法来解决上述问题。我们表明，使用有针对性的对抗性攻击进行对抗性训练(而不是无针对性的攻击)可以允许在对抗性公平方面进行更有利的权衡。实证结果验证了该方法的有效性。



## **31. Reassessing Noise Augmentation Methods in the Context of Adversarial Speech**

对抗性言语背景下重新评估噪音增强方法 eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2409.01813v2) [paper-pdf](http://arxiv.org/pdf/2409.01813v2)

**Authors**: Karla Pizzi, Matías Pizarro, Asja Fischer

**Abstract**: In this study, we investigate if noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different state-of-the-art ASR architectures, where each of the ASR architectures is trained under three different augmentation conditions: one subject to background noise, speed variations, and reverberations, another subject to speed variations only, and a third without any form of data augmentation. The results demonstrate that noise augmentation not only improves model performance on noisy speech but also the model's robustness to adversarial attacks.

摘要: 在这项研究中，我们研究了噪音增强训练是否可以同时提高自动语音识别（ASB）系统中的对抗鲁棒性。我们对四种不同最先进的ASB架构的对抗鲁棒性进行了比较分析，其中每个ASB架构都在三种不同的增强条件下训练：一种受到背景噪音、速度变化和回响的影响，另一种仅受到速度变化的影响，第三种没有任何形式的数据增强。结果表明，噪音增强不仅提高了模型在含噪语音上的性能，还提高了模型对对抗攻击的鲁棒性。



## **32. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2305.17000v7) [paper-pdf](http://arxiv.org/pdf/2305.17000v7)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **33. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

卡西姆·分歧：通过对抗性防御的扩散模型来启发性解纠缠 cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.23091v1) [paper-pdf](http://arxiv.org/pdf/2410.23091v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark).

摘要: 尽管不断努力保护神经分类器免受对手攻击，但它们仍然很脆弱，特别是面对看不见的攻击。相比之下，人类很难被微妙的操纵所欺骗，因为我们只根据基本因素做出判断。受到这一观察的启发，我们试图用基本的标签原因因素来建模标签生成，并结合标签非原因因素来辅助数据生成。对于一个对抗性的例子，我们的目标是将扰动区分为非致因因素，并仅基于标签致因因素进行预测。具体地说，我们提出了一个偶然扩散模型(CausalDiff)，该模型使扩散模型适用于条件数据生成，并通过向一个新的偶然信息瓶颈目标学习来区分这两种类型的偶然因素。经验上，CausalDiff在各种隐形攻击上的表现明显优于最先进的防御方法，在CIFAR-10上获得了86.39%(+4.01%)的平均健壮性，在CIFAR-100上获得了56.25%(+3.13%)的健壮性，在GTSRB(德国交通标志识别基准)上实现了82.62%(+4.93%)的平均健壮性。



## **34. Robustifying automatic speech recognition by extracting slowly varying features**

通过提取缓慢变化的特征来增强自动语音识别 eess.AS

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2112.07400v2) [paper-pdf](http://arxiv.org/pdf/2112.07400v2)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: In the past few years, it has been shown that deep learning systems are highly vulnerable under attacks with adversarial examples. Neural-network-based automatic speech recognition (ASR) systems are no exception. Targeted and untargeted attacks can modify an audio input signal in such a way that humans still recognise the same words, while ASR systems are steered to predict a different transcription. In this paper, we propose a defense mechanism against targeted adversarial attacks consisting in removing fast-changing features from the audio signals, either by applying slow feature analysis, a low-pass filter, or both, before feeding the input to the ASR system. We perform an empirical analysis of hybrid ASR models trained on data pre-processed in such a way. While the resulting models perform quite well on benign data, they are significantly more robust against targeted adversarial attacks: Our final, proposed model shows a performance on clean data similar to the baseline model, while being more than four times more robust.

摘要: 在过去的几年里，深度学习系统被证明在敌意攻击下是非常脆弱的。基于神经网络的自动语音识别(ASR)系统也不例外。定向和非定向攻击可以修改音频输入信号，使人类仍能识别相同的单词，而ASR系统则被引导预测不同的转录。在本文中，我们提出了一种针对目标攻击的防御机制，即在将输入输入到ASR系统之前，通过慢速特征分析、低通滤波或两者结合的方法，从音频信号中去除快速变化的特征。我们对以这种方式处理的数据训练的混合ASR模型进行了实证分析。虽然得到的模型在良性数据上表现得相当好，但它们对目标对手攻击的健壮性要强得多：我们最终提出的模型在干净数据上的性能与基准模型相似，但健壮性要高四倍以上。



## **35. Are Your Models Still Fair? Fairness Attacks on Graph Neural Networks via Node Injections**

你的模型仍然公平吗？通过节点注入对图神经网络的公平性攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2406.03052v2) [paper-pdf](http://arxiv.org/pdf/2406.03052v2)

**Authors**: Zihan Luo, Hong Huang, Yongkang Zhou, Jiping Zhang, Nuo Chen, Hai Jin

**Abstract**: Despite the remarkable capabilities demonstrated by Graph Neural Networks (GNNs) in graph-related tasks, recent research has revealed the fairness vulnerabilities in GNNs when facing malicious adversarial attacks. However, all existing fairness attacks require manipulating the connectivity between existing nodes, which may be prohibited in reality. To this end, we introduce a Node Injection-based Fairness Attack (NIFA), exploring the vulnerabilities of GNN fairness in such a more realistic setting. In detail, NIFA first designs two insightful principles for node injection operations, namely the uncertainty-maximization principle and homophily-increase principle, and then optimizes injected nodes' feature matrix to further ensure the effectiveness of fairness attacks. Comprehensive experiments on three real-world datasets consistently demonstrate that NIFA can significantly undermine the fairness of mainstream GNNs, even including fairness-aware GNNs, by injecting merely 1% of nodes. We sincerely hope that our work can stimulate increasing attention from researchers on the vulnerability of GNN fairness, and encourage the development of corresponding defense mechanisms. Our code and data are released at: https://github.com/CGCL-codes/NIFA.

摘要: 尽管图神经网络(GNN)在与图相关的任务中表现出了卓越的能力，但最近的研究揭示了GNN在面对恶意攻击时的公平性漏洞。然而，所有现有的公平攻击都需要操纵现有节点之间的连通性，这在现实中可能是被禁止的。为此，我们引入了一种基于节点注入的公平攻击(NIFA)，探讨了GNN公平性在这样一个更现实的环境下的脆弱性。具体而言，NIFA首先为节点注入操作设计了两个有洞察力的原则，即不确定性最大化原则和同质性增加原则，然后对注入节点的特征矩阵进行优化，进一步保证了公平攻击的有效性。在三个真实数据集上的综合实验一致表明，NIFA只注入1%的节点，就可以显著破坏主流GNN的公平性，即使包括公平感知的GNN。我们真诚地希望我们的工作能够引起研究人员对GNN公平性脆弱性的越来越多的关注，并鼓励开发相应的防御机制。我们的代码和数据发布在：https://github.com/CGCL-codes/NIFA.



## **36. Effective and Efficient Adversarial Detection for Vision-Language Models via A Single Vector**

通过单个载体对视觉语言模型进行有效且高效的对抗检测 cs.CV

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22888v1) [paper-pdf](http://arxiv.org/pdf/2410.22888v1)

**Authors**: Youcheng Huang, Fengbin Zhu, Jingkun Tang, Pan Zhou, Wenqiang Lei, Jiancheng Lv, Tat-Seng Chua

**Abstract**: Visual Language Models (VLMs) are vulnerable to adversarial attacks, especially those from adversarial images, which is however under-explored in literature. To facilitate research on this critical safety problem, we first construct a new laRge-scale Adervsarial images dataset with Diverse hArmful Responses (RADAR), given that existing datasets are either small-scale or only contain limited types of harmful responses. With the new RADAR dataset, we further develop a novel and effective iN-time Embedding-based AdveRSarial Image DEtection (NEARSIDE) method, which exploits a single vector that distilled from the hidden states of VLMs, which we call the attacking direction, to achieve the detection of adversarial images against benign ones in the input. Extensive experiments with two victim VLMs, LLaVA and MiniGPT-4, well demonstrate the effectiveness, efficiency, and cross-model transferrability of our proposed method. Our code is available at https://github.com/mob-scu/RADAR-NEARSIDE

摘要: 视觉语言模型（VLM）很容易受到对抗性攻击，尤其是来自对抗性图像的攻击，但文献中对此尚未充分探讨。为了促进对这个关键安全问题的研究，我们首先构建一个具有多样性干扰响应（RADART）的新的大规模Adervsarial图像数据集，因为现有数据集要么小规模，要么仅包含有限类型的有害反应。利用新的雷达数据集，我们进一步开发了一种新颖且有效的基于iN时间嵌入的AdveRSarial Image Detect（NEARSIDE）方法，该方法利用从VLM的隐藏状态（我们称之为攻击方向）中提取的单个载体，以实现针对输入中良性图像的对抗图像的检测。对两个受害VLM（LLaVA和MiniGPT-4）进行的大量实验很好地证明了我们提出的方法的有效性、效率和跨模型可移植性。我们的代码可在https://github.com/mob-scu/RADAR-NEARSIDE上获取



## **37. Stealing User Prompts from Mixture of Experts**

从专家混合处窃取用户预算 cs.CR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22884v1) [paper-pdf](http://arxiv.org/pdf/2410.22884v1)

**Authors**: Itay Yona, Ilia Shumailov, Jamie Hayes, Nicholas Carlini

**Abstract**: Mixture-of-Experts (MoE) models improve the efficiency and scalability of dense language models by routing each token to a small number of experts in each layer. In this paper, we show how an adversary that can arrange for their queries to appear in the same batch of examples as a victim's queries can exploit Expert-Choice-Routing to fully disclose a victim's prompt. We successfully demonstrate the effectiveness of this attack on a two-layer Mixtral model, exploiting the tie-handling behavior of the torch.topk CUDA implementation. Our results show that we can extract the entire prompt using $O({VM}^2)$ queries (with vocabulary size $V$ and prompt length $M$) or 100 queries on average per token in the setting we consider. This is the first attack to exploit architectural flaws for the purpose of extracting user prompts, introducing a new class of LLM vulnerabilities.

摘要: 专家混合（MoE）模型通过将每个令牌路由到每层中的少数专家来提高密集语言模型的效率和可扩展性。在本文中，我们展示了可以安排其查询与受害者查询出现在同一批示例中的对手如何利用Expert-Choice-Routing来完全披露受害者的提示。我们利用torch.topk CUDA实现的领带处理行为，在两层Mixtral模型上成功证明了这种攻击的有效性。我们的结果表明，我们可以使用$O（{VM}'#39;#39;#39; s）$查询（词汇量大小为$V$，提示长度为$M$）或在我们考虑的设置中每个令牌平均提取100个查询来提取整个提示。这是第一次利用架构缺陷来提取用户提示的攻击，从而引入了一类新的LLM漏洞。



## **38. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

了解和改进对抗性协作过滤以实现稳健推荐 cs.IR

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22844v1) [paper-pdf](http://arxiv.org/pdf/2410.22844v1)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.

摘要: 对抗性协同过滤(ACF)通过对抗性训练将对抗性扰动应用于用户和项目嵌入，被广泛认为是提高协同过滤推荐系统对中毒攻击的稳健性的有效策略。此外，大量研究表明，与传统的推荐算法相比，自适应过滤算法也能提高推荐性能。尽管取得了这些经验上的成功，但关于ACF在性能和稳健性方面的有效性的理论理解仍然不清楚。为了弥补这一差距，在本文中，我们首先从理论上证明，在干净和有毒的数据环境下，与相同训练周期的传统CF相比，ACF可以获得更低的推荐误差。此外，通过建立ACF优化过程中推荐误差减少的界限，我们发现根据不同用户的嵌入尺度对不同用户应用个性化的扰动幅度可以进一步提高ACF的有效性。在这些理论理解的基础上，我们提出了个性化幅度对抗协同过滤(PamaCF)。大量实验表明，PamaCF在有效防御各种类型的中毒攻击的同时，显著提高了推荐性能。



## **39. Geometry Cloak: Preventing TGS-based 3D Reconstruction from Copyrighted Images**

几何斗篷：防止受版权保护的图像进行基于TG的3D重建 cs.CV

Accepted by NeurIPS 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22705v1) [paper-pdf](http://arxiv.org/pdf/2410.22705v1)

**Authors**: Qi Song, Ziyuan Luo, Ka Chun Cheung, Simon See, Renjie Wan

**Abstract**: Single-view 3D reconstruction methods like Triplane Gaussian Splatting (TGS) have enabled high-quality 3D model generation from just a single image input within seconds. However, this capability raises concerns about potential misuse, where malicious users could exploit TGS to create unauthorized 3D models from copyrighted images. To prevent such infringement, we propose a novel image protection approach that embeds invisible geometry perturbations, termed "geometry cloaks", into images before supplying them to TGS. These carefully crafted perturbations encode a customized message that is revealed when TGS attempts 3D reconstructions of the cloaked image. Unlike conventional adversarial attacks that simply degrade output quality, our method forces TGS to fail the 3D reconstruction in a specific way - by generating an identifiable customized pattern that acts as a watermark. This watermark allows copyright holders to assert ownership over any attempted 3D reconstructions made from their protected images. Extensive experiments have verified the effectiveness of our geometry cloak. Our project is available at https://qsong2001.github.io/geometry_cloak.

摘要: 像三平面高斯飞溅(TGS)这样的单视图3D重建方法能够在几秒钟内从单一图像输入生成高质量的3D模型。然而，这一功能引发了人们对潜在滥用的担忧，恶意用户可能会利用TGS从受版权保护的图像创建未经授权的3D模型。为了防止这种侵权行为，我们提出了一种新的图像保护方法，该方法在将不可见的几何扰动(称为几何斗篷)嵌入到图像中，然后将其提供给TGS。这些精心制作的扰动编码了一条定制的消息，当TGS试图对被遮盖的图像进行3D重建时，该消息会被揭示出来。与简单地降低输出质量的传统对抗性攻击不同，我们的方法迫使TGS以特定的方式失败3D重建-通过生成可识别的定制图案作为水印。该水印允许版权所有者主张对从其受保护的图像进行的任何3D重建尝试的所有权。广泛的实验已经验证了我们几何斗篷的有效性。我们的项目可在https://qsong2001.github.io/geometry_cloak.上查看



## **40. Backdoor Attack Against Vision Transformers via Attention Gradient-Based Image Erosion**

通过基于注意力的图像侵蚀对视觉变形者进行后门攻击 cs.CV

Accepted by IEEE GLOBECOM 2024

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22678v1) [paper-pdf](http://arxiv.org/pdf/2410.22678v1)

**Authors**: Ji Guo, Hongwei Li, Wenbo Jiang, Guoming Lu

**Abstract**: Vision Transformers (ViTs) have outperformed traditional Convolutional Neural Networks (CNN) across various computer vision tasks. However, akin to CNN, ViTs are vulnerable to backdoor attacks, where the adversary embeds the backdoor into the victim model, causing it to make wrong predictions about testing samples containing a specific trigger. Existing backdoor attacks against ViTs have the limitation of failing to strike an optimal balance between attack stealthiness and attack effectiveness.   In this work, we propose an Attention Gradient-based Erosion Backdoor (AGEB) targeted at ViTs. Considering the attention mechanism of ViTs, AGEB selectively erodes pixels in areas of maximal attention gradient, embedding a covert backdoor trigger. Unlike previous backdoor attacks against ViTs, AGEB achieves an optimal balance between attack stealthiness and attack effectiveness, ensuring the trigger remains invisible to human detection while preserving the model's accuracy on clean samples. Extensive experimental evaluations across various ViT architectures and datasets confirm the effectiveness of AGEB, achieving a remarkable Attack Success Rate (ASR) without diminishing Clean Data Accuracy (CDA). Furthermore, the stealthiness of AGEB is rigorously validated, demonstrating minimal visual discrepancies between the clean and the triggered images.

摘要: 视觉转换器(VITS)在各种计算机视觉任务中的表现优于传统的卷积神经网络(CNN)。然而，与CNN类似，VITS容易受到后门攻击，即对手将后门嵌入受害者模型，导致其对包含特定触发因素的测试样本做出错误预测。现有的针对VITS的后门攻击存在未能在攻击隐蔽性和攻击有效性之间取得最佳平衡的局限性。在这项工作中，我们提出了一个针对VITS的基于注意力梯度的侵蚀后门(AGEB)。考虑到VITS的注意机制，AGEB通过嵌入一个隐蔽的后门触发器，选择性地侵蚀注意力梯度最大的区域的像素。与以前针对VITS的后门攻击不同，AGEB在攻击隐蔽性和攻击有效性之间实现了最佳平衡，确保触发器保持人类检测不到，同时保持模型对干净样本的准确性。对各种VIT架构和数据集进行的广泛实验评估证实了AGEB的有效性，在不降低清洁数据准确性(CDA)的情况下实现了显著的攻击成功率(ASR)。此外，AGEB的隐蔽性得到了严格的验证，显示了干净图像和触发图像之间最小的视觉差异。



## **41. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

用于机器学习文本分类器的自动可信度Oracle生成 cs.SE

**SubmitDate**: 2024-10-30    [abs](http://arxiv.org/abs/2410.22663v1) [paper-pdf](http://arxiv.org/pdf/2410.22663v1)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains, such as toxicity detection, chatbot consulting, and review analysis. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Several studies indicate that traditional metrics, such as model confidence and accuracy, are insufficient to build human trust in ML models. These models often learn spurious correlations during training and predict based on them during inference. In the real world, where such correlations are absent, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods, which is time-consuming and not scalable. We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers, which automatically checks whether the prediction-contributing words are related to the predicted class using explanation methods and word embeddings. To demonstrate its practical usefulness, we introduce a novel adversarial attack method targeting trustworthiness issues identified by TOKI. We compare TOKI with a naive baseline based solely on model confidence using human-created ground truths of 6,000 predictions. We also compare TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided adversarial attack method is more effective with fewer perturbations than A2T.

摘要: 机器学习用于文本分类已被广泛应用于毒性检测、聊天机器人咨询、评论分析等领域。这些应用程序可能会对伦理、经济和人类行为产生重大影响，从而引发对信任ML决策的严重担忧。一些研究表明，传统的度量标准，如模型的可信度和准确性，不足以建立人类对ML模型的信任。这些模型经常在训练过程中学习伪相关性，并在推理过程中基于它们进行预测。在缺乏这种相关性的现实世界中，它们的表现可能会显著恶化。为了避免这种情况，一种常见的做法是测试预测是否合理。随之而来的是一个被称为可信性先知问题的挑战。由于缺乏自动化的可信性先知，评估需要对解释方法披露的决策过程进行人工验证，这既耗时又不可扩展。本文提出了第一种自动生成文本分类器可信性预言的方法TOKI，它通过解释方法和词嵌入的方法自动检查预测贡献词是否与预测类相关。为了证明其实用性，我们引入了一种新的针对TOKI确定的可信性问题的对抗性攻击方法。我们将TOKI与单纯基于模型置信度的天真基线进行比较，该基线使用了6,000个预测的人为事实。我们还比较了TOKI引导的对抗性攻击方法和SOTA对抗性攻击方法A2T。结果表明，依靠预测的不确定性不能区分可信和不可信的预测，TOKI的准确率比朴素基线高142%，TOKI引导的对抗性攻击方法比A2T更有效，扰动更少。



## **42. AdvWeb: Controllable Black-box Attacks on VLM-powered Web Agents**

AdvWeb：对TLR驱动的Web代理的可控黑匣子攻击 cs.CR

15 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.17401v2) [paper-pdf](http://arxiv.org/pdf/2410.17401v2)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Vision Language Models (VLMs) have revolutionized the creation of generalist web agents, empowering them to autonomously complete diverse tasks on real-world websites, thereby boosting human efficiency and productivity. However, despite their remarkable capabilities, the safety and security of these agents against malicious attacks remain critically underexplored, raising significant concerns about their safe deployment. To uncover and exploit such vulnerabilities in web agents, we provide AdvWeb, a novel black-box attack framework designed against web agents. AdvWeb trains an adversarial prompter model that generates and injects adversarial prompts into web pages, misleading web agents into executing targeted adversarial actions such as inappropriate stock purchases or incorrect bank transactions, actions that could lead to severe real-world consequences. With only black-box access to the web agent, we train and optimize the adversarial prompter model using DPO, leveraging both successful and failed attack strings against the target agent. Unlike prior approaches, our adversarial string injection maintains stealth and control: (1) the appearance of the website remains unchanged before and after the attack, making it nearly impossible for users to detect tampering, and (2) attackers can modify specific substrings within the generated adversarial string to seamlessly change the attack objective (e.g., purchasing stocks from a different company), enhancing attack flexibility and efficiency. We conduct extensive evaluations, demonstrating that AdvWeb achieves high success rates in attacking SOTA GPT-4V-based VLM agent across various web tasks. Our findings expose critical vulnerabilities in current LLM/VLM-based agents, emphasizing the urgent need for developing more reliable web agents and effective defenses. Our code and data are available at https://ai-secure.github.io/AdvWeb/ .

摘要: 视觉语言模型(VLM)彻底改变了多面手Web代理的创建，使其能够在现实世界的网站上自主完成各种任务，从而提高了人类的效率和生产力。然而，尽管这些代理具有非凡的能力，但其抵御恶意攻击的安全性和安全性仍然严重不足，这引发了人们对其安全部署的严重担忧。为了发现和利用Web代理中的此类漏洞，我们提供了AdvWeb，这是一个针对Web代理设计的新型黑盒攻击框架。AdvWeb训练一种对抗性提示器模型，该模型生成对抗性提示并将其注入网页，误导网络代理执行有针对性的对抗性行动，如不适当的股票购买或不正确的银行交易，这些行动可能会导致严重的现实世界后果。在只有黑盒访问Web代理的情况下，我们使用DPO训练和优化对抗性提示器模型，利用针对目标代理的成功和失败的攻击字符串。与以前的方法不同，我们的敌意字符串注入保持了隐蔽性和可控性：(1)攻击前后网站的外观保持不变，使得用户几乎不可能检测到篡改；(2)攻击者可以修改生成的敌意字符串中的特定子字符串，以无缝更改攻击目标(例如，从不同公司购买股票)，从而增强攻击的灵活性和效率。我们进行了广泛的评估，表明AdvWeb在各种Web任务中攻击基于Sota GPT-4V的VLM代理取得了很高的成功率。我们的发现暴露了当前基于LLM/VLM的代理的严重漏洞，强调了开发更可靠的网络代理和有效防御的迫切需要。我们的代码和数据可在https://ai-secure.github.io/AdvWeb/上获得。



## **43. LookHere: Vision Transformers with Directed Attention Generalize and Extrapolate**

LookHere：具有定向注意力的视觉变形者概括和推断 cs.CV

NeurIPS 2024 Camera Ready

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2405.13985v2) [paper-pdf](http://arxiv.org/pdf/2405.13985v2)

**Authors**: Anthony Fuller, Daniel G. Kyrollos, Yousef Yassin, James R. Green

**Abstract**: High-resolution images offer more information about scenes that can improve model accuracy. However, the dominant model architecture in computer vision, the vision transformer (ViT), cannot effectively leverage larger images without finetuning -- ViTs poorly extrapolate to more patches at test time, although transformers offer sequence length flexibility. We attribute this shortcoming to the current patch position encoding methods, which create a distribution shift when extrapolating.   We propose a drop-in replacement for the position encoding of plain ViTs that restricts attention heads to fixed fields of view, pointed in different directions, using 2D attention masks. Our novel method, called LookHere, provides translation-equivariance, ensures attention head diversity, and limits the distribution shift that attention heads face when extrapolating. We demonstrate that LookHere improves performance on classification (avg. 1.6%), against adversarial attack (avg. 5.4%), and decreases calibration error (avg. 1.5%) -- on ImageNet without extrapolation. With extrapolation, LookHere outperforms the current SoTA position encoding method, 2D-RoPE, by 21.7% on ImageNet when trained at $224^2$ px and tested at $1024^2$ px. Additionally, we release a high-resolution test set to improve the evaluation of high-resolution image classifiers, called ImageNet-HR.

摘要: 高分辨率图像提供了有关场景的更多信息，可以提高模型精度。然而，计算机视觉中占主导地位的模型体系结构视觉转换器(VIT)在没有精细调整的情况下无法有效地利用更大的图像-VIT在测试时很难外推到更多的补丁，尽管转换器提供了序列长度的灵活性。我们将这一缺陷归因于目前的补丁位置编码方法，这些方法在外推时会产生分布偏移。我们提出了一种替代普通VITS的位置编码的方法，它使用2D注意掩码将注意力头部限制在指向不同方向的固定视野中。我们的新方法LookHere提供了平移等差性，确保了注意力头部的多样性，并限制了注意力头部在外推时面临的分布变化。我们证明LookHere提高了分类性能(平均1.6%)，抗对手攻击(Avg.5.4%)，降低了校准误差(平均1.5%)--在ImageNet上，没有外推。通过外推，LookHere在ImageNet上以$224^2$px进行训练并以$1024^2$px进行测试时，在ImageNet上的性能比当前的SOTA位置编码方法2D-ROPE高21.7%。此外，我们还发布了一个高分辨率测试集来改进高分辨率图像分类器的评估，称为ImageNet-HR。



## **44. Power side-channel leakage localization through adversarial training of deep neural networks**

通过深度神经网络的对抗训练进行电源侧通道泄漏定位 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22425v1) [paper-pdf](http://arxiv.org/pdf/2410.22425v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Kaushik Roy

**Abstract**: Supervised deep learning has emerged as an effective tool for carrying out power side-channel attacks on cryptographic implementations. While increasingly-powerful deep learning-based attacks are regularly published, comparatively-little work has gone into using deep learning to defend against these attacks. In this work we propose a technique for identifying which timesteps in a power trace are responsible for leaking a cryptographic key, through an adversarial game between a deep learning-based side-channel attacker which seeks to classify a sensitive variable from the power traces recorded during encryption, and a trainable noise generator which seeks to thwart this attack by introducing a minimal amount of noise into the power traces. We demonstrate on synthetic datasets that our method can outperform existing techniques in the presence of common countermeasures such as Boolean masking and trace desynchronization. Results on real datasets are weak because the technique is highly sensitive to hyperparameters and early-stop point, and we lack a holdout dataset with ground truth knowledge of leaking points for model selection. Nonetheless, we believe our work represents an important first step towards deep side-channel leakage localization without relying on strong assumptions about the implementation or the nature of its leakage. An open-source PyTorch implementation of our experiments is provided.

摘要: 有监督的深度学习已经成为对密码实现进行功率侧通道攻击的有效工具。虽然越来越强大的基于深度学习的攻击定期发布，但相对较少的工作是使用深度学习来防御这些攻击。在这项工作中，我们提出了一种技术，通过基于深度学习的侧通道攻击者和可训练噪声生成器之间的对抗性博弈来识别功率跟踪中的哪些时间步骤负责泄漏密钥，侧通道攻击者试图从加密过程中记录的功率跟踪中将敏感变量分类，而可训练噪声生成器试图通过在功率跟踪中引入最少量的噪声来阻止这种攻击。我们在合成数据集上演示了我们的方法在存在布尔掩蔽和跟踪去同步等常见对策的情况下可以优于现有的技术。在真实数据集上的结果很弱，因为该技术对超参数和提前停止点高度敏感，并且我们缺乏一个具有泄漏点基本事实知识的数据集来选择模型。尽管如此，我们相信我们的工作代表着迈向深侧沟道泄漏定位的重要的第一步，而不依赖于对其实施或其泄漏的性质的强烈假设。给出了我们实验的一个开源的PyTorch实现。



## **45. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

20 pages

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22307v1) [paper-pdf](http://arxiv.org/pdf/2410.22307v1)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: Open-source Large Language Models (LLMs) have recently demonstrated remarkable capabilities in natural language understanding and generation, leading to widespread adoption across various domains. However, their increasing model sizes render local deployment impractical for individual users, pushing many to rely on computing service providers for inference through a blackbox API. This reliance introduces a new risk: a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby delivering inferior outputs while benefiting from cost savings. In this paper, we formalize the problem of verifiable inference for LLMs. Existing verifiable computing solutions based on cryptographic or game-theoretic techniques are either computationally uneconomical or rest on strong assumptions. We introduce SVIP, a secret-based verifiable LLM inference protocol that leverages intermediate outputs from LLM as unique model identifiers. By training a proxy task on these outputs and requiring the computing provider to return both the generated text and the processed intermediate outputs, users can reliably verify whether the computing provider is acting honestly. In addition, the integration of a secret mechanism further enhances the security of our protocol. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per query for verification.

摘要: 开源的大型语言模型(LLM)最近在自然语言理解和生成方面表现出了非凡的能力，导致了在各个领域的广泛采用。然而，它们不断增长的模型规模使得本地部署对个人用户来说是不现实的，促使许多人依赖计算服务提供商通过黑盒API进行推理。这种依赖带来了新的风险：计算提供商可能会在未经用户同意的情况下，悄悄地用较小、功能较差的模型替换所请求的LLM，从而在提供劣质产出的同时受益于成本节约。本文对LLMS的可验证推理问题进行了形式化描述。现有的基于密码学或博弈论技术的可验证计算解决方案要么在计算上不经济，要么依赖于强有力的假设。我们引入了SVIP，这是一个基于秘密的可验证LLM推理协议，它利用LLM的中间输出作为唯一的模型标识符。通过对这些输出训练代理任务并要求计算提供商返回生成的文本和处理的中间输出，用户可以可靠地验证计算提供商是否诚实行事。此外，秘密机制的集成进一步增强了协议的安全性。我们深入分析了我们的协议在多种强和自适应对抗场景下的性能。大量实验表明，SVIP算法具有较高的准确性、通用性、计算效率和抵抗各种攻击的能力。值得注意的是，SVIP实现了5%以下的假阴性率和3%以下的假阳性率，而每次查询验证所需的时间不到0.01秒。



## **46. Embedding-based classifiers can detect prompt injection attacks**

基于嵌入的分类器可以检测提示注入攻击 cs.CR

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22284v1) [paper-pdf](http://arxiv.org/pdf/2410.22284v1)

**Authors**: Md. Ahsan Ayub, Subhabrata Majumdar

**Abstract**: Large Language Models (LLMs) are seeing significant adoption in every type of organization due to their exceptional generative capabilities. However, LLMs are found to be vulnerable to various adversarial attacks, particularly prompt injection attacks, which trick them into producing harmful or inappropriate content. Adversaries execute such attacks by crafting malicious prompts to deceive the LLMs. In this paper, we propose a novel approach based on embedding-based Machine Learning (ML) classifiers to protect LLM-based applications against this severe threat. We leverage three commonly used embedding models to generate embeddings of malicious and benign prompts and utilize ML classifiers to predict whether an input prompt is malicious. Out of several traditional ML methods, we achieve the best performance with classifiers built using Random Forest and XGBoost. Our classifiers outperform state-of-the-art prompt injection classifiers available in open-source implementations, which use encoder-only neural networks.

摘要: 大型语言模型(LLM)由于其非凡的生成能力，在每种类型的组织中都得到了大量采用。然而，LLM被发现容易受到各种对抗性攻击，特别是即时注入攻击，这些攻击会诱使它们产生有害或不适当的内容。攻击者通过精心编制恶意提示来欺骗LLM，从而执行此类攻击。在本文中，我们提出了一种基于嵌入的机器学习(ML)分类器的新方法来保护基于LLM的应用程序免受这种严重威胁。我们利用三种常用的嵌入模型来生成恶意提示和良性提示的嵌入，并利用ML分类器来预测输入提示是否为恶意提示。在几种传统的最大似然分类方法中，我们使用随机森林和XGBoost构建的分类器取得了最好的性能。我们的分类器比开源实现中可用的最先进的提示注入分类器性能更好，后者使用仅限编码器的神经网络。



## **47. AmpleGCG-Plus: A Strong Generative Model of Adversarial Suffixes to Jailbreak LLMs with Higher Success Rates in Fewer Attempts**

AmpleGCG-Plus：越狱LLC的对抗性后缀的强生成模型，以更少的尝试获得更高的成功率 cs.CL

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.22143v1) [paper-pdf](http://arxiv.org/pdf/2410.22143v1)

**Authors**: Vishal Kumar, Zeyi Liao, Jaylen Jones, Huan Sun

**Abstract**: Although large language models (LLMs) are typically aligned, they remain vulnerable to jailbreaking through either carefully crafted prompts in natural language or, interestingly, gibberish adversarial suffixes. However, gibberish tokens have received relatively less attention despite their success in attacking aligned LLMs. Recent work, AmpleGCG~\citep{liao2024amplegcg}, demonstrates that a generative model can quickly produce numerous customizable gibberish adversarial suffixes for any harmful query, exposing a range of alignment gaps in out-of-distribution (OOD) language spaces. To bring more attention to this area, we introduce AmpleGCG-Plus, an enhanced version that achieves better performance in fewer attempts. Through a series of exploratory experiments, we identify several training strategies to improve the learning of gibberish suffixes. Our results, verified under a strict evaluation setting, show that it outperforms AmpleGCG on both open-weight and closed-source models, achieving increases in attack success rate (ASR) of up to 17\% in the white-box setting against Llama-2-7B-chat, and more than tripling ASR in the black-box setting against GPT-4. Notably, AmpleGCG-Plus jailbreaks the newer GPT-4o series of models at similar rates to GPT-4, and, uncovers vulnerabilities against the recently proposed circuit breakers defense. We publicly release AmpleGCG-Plus along with our collected training datasets.

摘要: 尽管大型语言模型(LLM)通常是一致的，但它们仍然很容易通过精心设计的自然语言提示或有趣的胡言乱语对抗性后缀越狱。然而，令人费解的令牌尽管成功地攻击了对齐的LLM，但受到的关注相对较少。最近的工作，AmpleGCG~\Citep{Lio2024Amplegcg}，证明了生成模型可以为任何有害的查询快速生成大量可定制的胡言乱语对抗性后缀，从而暴露出分布外(OOD)语言空间中的一系列对齐差距。为了引起人们对这一领域的更多关注，我们推出了AmpleGCG-Plus，这是一个增强版本，在较少的尝试中获得了更好的性能。通过一系列的探索性实验，我们确定了几种训练策略来提高乱码后缀的学习效果。在严格的评估设置下验证的结果表明，它在开源和闭源模型上都优于AmpleGCG，在白盒环境下相对于Llama-2-7B-Chat的攻击成功率(ASR)提高了17%，在黑盒环境下相对于GPT-4的攻击成功率(ASR)提高了两倍以上。值得注意的是，AmpleGCG-Plus以类似于GPT-4的速度监禁了较新的GPT-4o系列型号，并揭示了针对最近提出的断路器防御的漏洞。我们公开发布AmpleGCG-Plus以及我们收集的训练数据集。



## **48. Iterative Window Mean Filter: Thwarting Diffusion-based Adversarial Purification**

迭代窗口均值过滤器：阻止基于扩散的对抗净化 cs.CR

Accepted in IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2408.10673v3) [paper-pdf](http://arxiv.org/pdf/2408.10673v3)

**Authors**: Hanrui Wang, Ruoxi Sun, Cunjian Chen, Minhui Xue, Lay-Ki Soon, Shuo Wang, Zhe Jin

**Abstract**: Face authentication systems have brought significant convenience and advanced developments, yet they have become unreliable due to their sensitivity to inconspicuous perturbations, such as adversarial attacks. Existing defenses often exhibit weaknesses when facing various attack algorithms and adaptive attacks or compromise accuracy for enhanced security. To address these challenges, we have developed a novel and highly efficient non-deep-learning-based image filter called the Iterative Window Mean Filter (IWMF) and proposed a new framework for adversarial purification, named IWMF-Diff, which integrates IWMF and denoising diffusion models. These methods can function as pre-processing modules to eliminate adversarial perturbations without necessitating further modifications or retraining of the target system. We demonstrate that our proposed methodologies fulfill four critical requirements: preserved accuracy, improved security, generalizability to various threats in different settings, and better resistance to adaptive attacks. This performance surpasses that of the state-of-the-art adversarial purification method, DiffPure.

摘要: 人脸认证系统带来了极大的便利和先进的发展，但由于它们对诸如敌意攻击等不起眼的扰动非常敏感，因此变得不可靠。现有的防御在面对各种攻击算法和自适应攻击时往往表现出弱点，或者为了增强安全性而损害准确性。为了应对这些挑战，我们开发了一种新颖高效的基于非深度学习的图像过滤器，称为迭代窗口均值过滤器(IWMF)，并提出了一种结合IWMF和去噪扩散模型的新的对抗性净化框架IWMF-DIFF。这些方法可以作为前处理模块来消除对抗性干扰，而不需要对目标系统进行进一步的修改或重新培训。我们证明了我们提出的方法满足了四个关键要求：保持准确性，提高安全性，对不同环境下的各种威胁具有通用性，以及更好地抵抗自适应攻击。这一性能超过了最先进的对抗性净化方法DiffPure。



## **49. Forging the Forger: An Attempt to Improve Authorship Verification via Data Augmentation**

伪造伪造者：通过数据增强改进作者身份验证的尝试 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2403.11265v2) [paper-pdf](http://arxiv.org/pdf/2403.11265v2)

**Authors**: Silvia Corbara, Alejandro Moreo

**Abstract**: Authorship Verification (AV) is a text classification task concerned with inferring whether a candidate text has been written by one specific author or by someone else. It has been shown that many AV systems are vulnerable to adversarial attacks, where a malicious author actively tries to fool the classifier by either concealing their writing style, or by imitating the style of another author. In this paper, we investigate the potential benefits of augmenting the classifier training set with (negative) synthetic examples. These synthetic examples are generated to imitate the style of the author of interest. We analyze the improvements in classifier prediction that this augmentation brings to bear in the task of AV in an adversarial setting. In particular, we experiment with three different generator architectures (one based on Recurrent Neural Networks, another based on small-scale transformers, and another based on the popular GPT model) and with two training strategies (one inspired by standard Language Models, and another inspired by Wasserstein Generative Adversarial Networks). We evaluate our hypothesis on five datasets (three of which have been specifically collected to represent an adversarial setting) and using two learning algorithms for the AV classifier (Support Vector Machines and Convolutional Neural Networks). This experimentation has yielded negative results, revealing that, although our methodology proves effective in many adversarial settings, its benefits are too sporadic for a pragmatical application.

摘要: 作者身份验证是一项文本分类任务，涉及推断候选文本是由某个特定作者还是其他人撰写的。已经证明，许多反病毒系统容易受到敌意攻击，恶意作者通过隐藏他们的写作风格或模仿另一位作者的风格来主动试图愚弄分类器。在本文中，我们研究了用(负的)合成例子来扩大分类器训练集的潜在好处。这些合成的例子是为了模仿感兴趣的作者的风格而产生的。我们分析了这种增强在对抗性环境下对AV任务带来的分类器预测方面的改进。特别是，我们实验了三种不同的生成器体系结构(一种基于递归神经网络，另一种基于小型变压器，另一种基于流行的GPT模型)和两种训练策略(一种受到标准语言模型的启发，另一种受到Wasserstein生成性对手网络的启发)。我们在五个数据集(其中三个已经被专门收集来表示对抗性环境)上评估了我们的假设，并使用了两种用于AV分类器的学习算法(支持向量机和卷积神经网络)。这种实验产生了负面的结果，表明尽管我们的方法在许多对抗性环境中被证明是有效的，但它的好处对于实用应用来说太零星了。



## **50. On the Robustness of Adversarial Training Against Uncertainty Attacks**

论对抗训练对不确定性攻击的鲁棒性 cs.LG

**SubmitDate**: 2024-10-29    [abs](http://arxiv.org/abs/2410.21952v1) [paper-pdf](http://arxiv.org/pdf/2410.21952v1)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.

摘要: 在学习问题中，手头任务固有的噪音阻碍了在没有一定程度的不确定性的情况下进行推断的可能性。量化这种不确定性，不管它是否被广泛使用，都假定它与安全敏感的应用程序高度相关。在这些场景中，保证良好的(即值得信赖的)不确定性度量变得至关重要，下游模块可以安全地使用这些度量来驱动最终的决策过程。然而，攻击者可能有兴趣强迫系统产生(I)危及系统可用性的高度不确定的输出，或(Ii)低不确定性估计，使系统接受不确定的样本，而不是需要仔细检查(例如，人工干预)。因此，了解如何针对这类攻击获得稳健的不确定性估计变得至关重要。在这项工作中，我们从经验和理论上揭示了对敌意示例的防御，即仔细扰动导致错误分类的样本，额外地保证了在常见攻击场景下更安全、更可信的不确定性估计，而不需要特别的防御策略。为了支持我们的主张，我们在CIFAR-10和ImageNet数据集上评估了来自公开可用的基准RobustBch的多个对抗性稳健模型。



