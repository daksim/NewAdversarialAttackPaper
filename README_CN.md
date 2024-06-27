# Latest Adversarial Attack Papers
**update at 2024-06-27 09:46:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models**

大规模的野生合作：从野外越狱到（相反）更安全的语言模型 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18510v1) [paper-pdf](http://arxiv.org/pdf/2406.18510v1)

**Authors**: Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildTeaming, an automatic LLM safety red-teaming framework that mines in-the-wild user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes multiple tactics for systematic exploration of novel jailbreaks. Compared to prior work that performed red-teaming via recruited human workers, gradient-based optimization, or iterative revision with LLMs, our work investigates jailbreaks from chatbot users who were not specifically instructed to break the system. WildTeaming reveals previously unidentified vulnerabilities of frontier LLMs, resulting in up to 4.6x more diverse and successful adversarial attacks compared to state-of-the-art jailbreak methods.   While many datasets exist for jailbreak evaluation, very few open-source datasets exist for jailbreak training, as safety training data has been closed even when model weights are open. With WildTeaming we create WildJailbreak, a large-scale open-source synthetic safety dataset with 262K vanilla (direct request) and adversarial (complex jailbreak) prompt-response pairs. To mitigate exaggerated safety behaviors, WildJailbreak provides two contrastive types of queries: 1) harmful queries (vanilla & adversarial) and 2) benign queries that resemble harmful queries in form but contain no harm. As WildJailbreak considerably upgrades the quality and scale of existing safety resources, it uniquely enables us to examine the scaling effects of data and the interplay of data properties and model capabilities during safety training. Through extensive experiments, we identify the training properties that enable an ideal balance of safety behaviors: appropriate safeguarding without over-refusal, effective handling of vanilla and adversarial queries, and minimal, if any, decrease in general capabilities. All components of WildJailbeak contribute to achieving balanced safety behaviors of models.

摘要: 我们介绍了WildTeaming，一个自动的LLM安全红色团队框架，它在野外挖掘用户和聊天机器人的交互来发现5.7k独特的新越狱战术簇，然后组成多个策略来系统地探索新的越狱战术。与之前通过招募人类工人进行红团队合作、基于梯度的优化或使用LLMS进行迭代修订相比，我们的工作调查了聊天机器人用户的越狱行为，这些用户没有得到明确的指示来破坏系统。WildTeaming揭示了FronTier LLMS以前未知的漏洞，导致与最先进的越狱方法相比，多样化和成功的对抗性攻击高达4.6倍。虽然存在许多用于越狱评估的数据集，但用于越狱培训的开源数据集很少，因为即使在模型重量打开的情况下，安全培训数据也已关闭。使用WildTeaming，我们创建了WildJailBreak，这是一个大规模的开源合成安全数据集，具有262K的普通(直接请求)和对抗性(复杂的越狱)提示-响应对。为了减少夸张的安全行为，WildJailBreak提供了两种对比类型的查询：1)有害查询(普通查询和对抗性查询)和2)在形式上类似于有害查询但不包含危害的良性查询。由于WildJailBreak极大地提升了现有安全资源的质量和规模，它独特地使我们能够在安全培训期间检查数据的缩放效应以及数据属性和模型功能的相互作用。通过广泛的实验，我们确定了能够实现安全行为的理想平衡的训练属性：适当的保护而不过度拒绝，有效地处理普通和敌意的查询，以及最小程度地降低一般能力(如果有的话)。WildJailbeak的所有组件都有助于实现模型的安全行为平衡。



## **2. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18495v1) [paper-pdf](http://arxiv.org/pdf/2406.18495v1)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **3. Enhancing Federated Learning with Adaptive Differential Privacy and Priority-Based Aggregation**

利用自适应差异隐私和基于优先级的聚合增强联邦学习 cs.LG

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18491v1) [paper-pdf](http://arxiv.org/pdf/2406.18491v1)

**Authors**: Mahtab Talaei, Iman Izadi

**Abstract**: Federated learning (FL), a novel branch of distributed machine learning (ML), develops global models through a private procedure without direct access to local datasets. However, it is still possible to access the model updates (gradient updates of deep neural networks) transferred between clients and servers, potentially revealing sensitive local information to adversaries using model inversion attacks. Differential privacy (DP) offers a promising approach to addressing this issue by adding noise to the parameters. On the other hand, heterogeneities in data structure, storage, communication, and computational capabilities of devices can cause convergence problems and delays in developing the global model. A personalized weighted averaging of local parameters based on the resources of each device can yield a better aggregated model in each round. In this paper, to efficiently preserve privacy, we propose a personalized DP framework that injects noise based on clients' relative impact factors and aggregates parameters while considering heterogeneities and adjusting properties. To fulfill the DP requirements, we first analyze the convergence boundary of the FL algorithm when impact factors are personalized and fixed throughout the learning process. We then further study the convergence property considering time-varying (adaptive) impact factors.

摘要: 联合学习(FL)是分布式机器学习(ML)的一个新分支，它通过私有过程建立全局模型，而不需要直接访问局部数据集。然而，仍有可能访问在客户端和服务器之间传输的模型更新(深度神经网络的梯度更新)，从而潜在地向使用模型反转攻击的攻击者泄露敏感的本地信息。通过在参数中添加噪声，差分隐私(DP)为解决这一问题提供了一种很有前途的方法。另一方面，设备的数据结构、存储、通信和计算能力的异构性可能会导致全局模型开发的收敛问题和延迟。基于每个设备的资源的本地参数的个性化加权平均可以在每一轮中产生更好的聚合模型。为了有效地保护隐私，我们提出了一种个性化的DP框架，该框架根据客户的相对影响因子和聚集参数来注入噪声，同时考虑异构性和调整属性。为了满足动态规划的要求，我们首先分析了影响因素在学习过程中被个性化和固定时FL算法的收敛边界。然后，我们进一步研究了考虑时变(自适应)影响因子时的收敛性质。



## **4. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **5. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18451v1) [paper-pdf](http://arxiv.org/pdf/2406.18451v1)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过对CIFAR10和CIFAR100数据集上各种稳健训练模型的综合实证分析，我们发现它们表明了很强的边际一致性，并且它们的输入空间边际和Logit边际之间存在很强的相关性。然后，我们证明了我们可以有效地使用Logit裕度来自信地检测此类模型的脆性决策，并通过仅在较小的子集上估计输入裕度来准确地估计任意大测试集上的稳健精度。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **6. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



## **7. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

SUB-SYS：针对部分观察的多智能体强化学习系统的对抗策略 cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2402.03741v3) [paper-pdf](http://arxiv.org/pdf/2402.03741v3)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.

摘要: 多智能体强化学习(MAIL)的最新进展为无人机群体控制、机械臂协同操纵、多目标包围等开辟了广阔的应用前景。然而，MAIL部署过程中的潜在安全威胁需要更多的关注和彻底的调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞，生成导致特定任务失败的对抗性策略。例如，将超人级别围棋人工智能的胜率降低到20%左右。现有的研究主要集中在两人竞争环境中，假设攻击者拥有完整的全局状态观测。在这项研究中，我们首次揭示了攻击者即使限于在多智能体竞争环境中对受害者的部分观察也能够生成对抗策略的能力。具体地说，我们提出了一种新的黑盒攻击(子游戏)，它结合了构造多个子博弈的概念来减轻部分可观测性的影响，并建议在子策略之间共享转移以提高攻击者的利用能力。广泛的评估证明了子游戏在三个典型的部分可观测性限制下的有效性。可视化结果表明，对抗性政策导致受害者的政策网络激活显著不同。此外，我们评估了三种潜在的防御措施，旨在探索减轻对抗性政策构成的安全威胁的方法，为在竞争环境中部署Marl提供建设性的建议。



## **8. Artificial Immune System of Secure Face Recognition Against Adversarial Attacks**

对抗攻击的安全人脸识别人工免疫系统 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18144v1) [paper-pdf](http://arxiv.org/pdf/2406.18144v1)

**Authors**: Min Ren, Yunlong Wang, Yuhao Zhu, Yongzhen Huang, Zhenan Sun, Qi Li, Tieniu Tan

**Abstract**: Insect production for food and feed presents a promising supplement to ensure food safety and address the adverse impacts of agriculture on climate and environment in the future. However, optimisation is required for insect production to realise its full potential. This can be by targeted improvement of traits of interest through selective breeding, an approach which has so far been underexplored and underutilised in insect farming. Here we present a comprehensive review of the selective breeding framework in the context of insect production. We systematically evaluate adjustments of selective breeding techniques to the realm of insects and highlight the essential components integral to the breeding process. The discussion covers every step of a conventional breeding scheme, such as formulation of breeding objectives, phenotyping, estimation of genetic parameters and breeding values, selection of appropriate breeding strategies, and mitigation of issues associated with genetic diversity depletion and inbreeding. This review combines knowledge from diverse disciplines, bridging the gap between animal breeding, quantitative genetics, evolutionary biology, and entomology, offering an integrated view of the insect breeding research area and uniting knowledge which has previously remained scattered across diverse fields of expertise.

摘要: 用于食品和饲料的昆虫生产为确保食品安全和解决未来农业对气候和环境的不利影响提供了一种有希望的补充。然而，昆虫生产需要优化才能充分发挥其潜力。这可以通过选择性育种对感兴趣的特征进行有针对性的改进，这种方法到目前为止在昆虫养殖业中还没有得到充分的探索和利用。在此，我们对昆虫生产中的选择性育种框架进行了全面的回顾。我们系统地评估了选择性育种技术对昆虫领域的调整，并强调了育种过程中不可或缺的基本组成部分。讨论涵盖传统育种计划的每一个步骤，如制定育种目标、表型鉴定、估计遗传参数和育种价值、选择适当的育种策略以及缓解与遗传多样性枯竭和近亲繁殖有关的问题。这篇综述结合了不同学科的知识，弥合了动物育种、数量遗传学、进化生物学和昆虫学之间的差距，提供了昆虫育种研究领域的综合视角，并统一了以前分散在不同专业领域的知识。



## **9. Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents**

打破障碍：平滑DRL代理的增强实用性和稳健性 cs.LG

Published in ICML 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18062v1) [paper-pdf](http://arxiv.org/pdf/2406.18062v1)

**Authors**: Chung-En Sun, Sicun Gao, Tsui-Wei Weng

**Abstract**: Robustness remains a paramount concern in deep reinforcement learning (DRL), with randomized smoothing emerging as a key technique for enhancing this attribute. However, a notable gap exists in the performance of current smoothed DRL agents, often characterized by significantly low clean rewards and weak robustness. In response to this challenge, our study introduces innovative algorithms aimed at training effective smoothed robust DRL agents. We propose S-DQN and S-PPO, novel approaches that demonstrate remarkable improvements in clean rewards, empirical robustness, and robustness guarantee across standard RL benchmarks. Notably, our S-DQN and S-PPO agents not only significantly outperform existing smoothed agents by an average factor of $2.16\times$ under the strongest attack, but also surpass previous robustly-trained agents by an average factor of $2.13\times$. This represents a significant leap forward in the field. Furthermore, we introduce Smoothed Attack, which is $1.89\times$ more effective in decreasing the rewards of smoothed agents than existing adversarial attacks.

摘要: 稳健性仍然是深度强化学习(DRL)中最重要的问题，随机化平滑成为增强这一属性的关键技术。然而，当前平滑的DRL代理的性能存在着显著的差距，通常具有明显低的清洁回报和较弱的稳健性。为了应对这一挑战，我们的研究引入了创新的算法，旨在训练有效的平滑稳健的DRL代理。我们提出了S-DQN和S-PPO这两种新颖的方法，它们在干净的回报、经验稳健性和跨标准RL基准的稳健性保证方面都有显著的改善。值得注意的是，我们的S-DQN和S-PPO代理不仅在最强攻击下的平均性能比现有的平滑代理高出2.16倍，而且比以前训练有素的代理高出2.13倍。这代表着该领域的一次重大飞跃。此外，我们引入了平滑攻击，它比现有的对抗性攻击在减少平滑代理的报酬方面更有效，其效率为1.89倍。



## **10. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

DirectTA：针对大型视觉语言模型的指令调整有针对性的攻击 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。在该文中，我们提出了一种新颖而实用的定向攻击场景，攻击者只能知道受害者LVLM的视觉编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(称为\Textsc{InstructTA})，以提供对具有高可转移性的LVLM的定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高指令调优的可转移性，我们用GPT-4中转译的指令扩充了指令$\boldsign{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。代码可在https://github.com/xunguangwang/InstructTA.上获得



## **11. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

破译事后OOD检测器对抗鲁棒性的定义 cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.15104v2) [paper-pdf](http://arxiv.org/pdf/2406.15104v2)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.

摘要: 检测非分布（OOD）输入对于在现实世界场景中安全部署深度学习模型至关重要。近年来，开发了很多OOD检测器，甚至基准测试也已经标准化，即OpenOOD。事后检测器的数量正在快速增长，并显示出一种可以保护预训练的分类器免受自然分布变化的影响的选择，声称已经为现实世界的场景做好了准备。然而，它在处理敌对例子方面的功效在大多数研究中被忽视了。本文研究了16个事后检测器对多种规避攻击的对抗鲁棒性，并讨论了OOD检测器对抗防御的路线图。



## **12. Diffusion-based Adversarial Purification for Intrusion Detection**

基于扩散的入侵检测对抗净化 cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17606v1) [paper-pdf](http://arxiv.org/pdf/2406.17606v1)

**Authors**: Mohamed Amine Merzouk, Erwan Beurier, Reda Yaich, Nora Boulahia-Cuppens, Frédéric Cuppens

**Abstract**: The escalating sophistication of cyberattacks has encouraged the integration of machine learning techniques in intrusion detection systems, but the rise of adversarial examples presents a significant challenge. These crafted perturbations mislead ML models, enabling attackers to evade detection or trigger false alerts. As a reaction, adversarial purification has emerged as a compelling solution, particularly with diffusion models showing promising results. However, their purification potential remains unexplored in the context of intrusion detection. This paper demonstrates the effectiveness of diffusion models in purifying adversarial examples in network intrusion detection. Through a comprehensive analysis of the diffusion parameters, we identify optimal configurations maximizing adversarial robustness with minimal impact on normal performance. Importantly, this study reveals insights into the relationship between diffusion noise and diffusion steps, representing a novel contribution to the field. Our experiments are carried out on two datasets and against 5 adversarial attacks. The implementation code is publicly available.

摘要: 网络攻击的日益复杂鼓励了将机器学习技术整合到入侵检测系统中，但敌意例子的兴起构成了一个重大挑战。这些精心设计的扰动误导了ML模型，使攻击者能够逃避检测或触发错误警报。作为一种反应，对抗性净化已成为一种引人注目的解决方案，特别是在扩散模型显示了有希望的结果的情况下。然而，在入侵检测的背景下，它们的净化潜力仍然没有被发掘。本文论证了扩散模型在网络入侵检测中净化恶意实例的有效性。通过对扩散参数的综合分析，我们确定了在对正常性能影响最小的情况下最大化对手健壮性的最优配置。重要的是，这项研究揭示了扩散噪声和扩散步骤之间的关系，代表了对该领域的新贡献。我们的实验是在两个数据集上进行的，并针对5个对手攻击进行了测试。实现代码是公开的。



## **13. Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness**

对抗鲁棒性随机平滑中统计估计问题的处理 stat.ML

comments are welcome

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17830v1) [paper-pdf](http://arxiv.org/pdf/2406.17830v1)

**Authors**: Vaclav Voracek

**Abstract**: Randomized smoothing is a popular certified defense against adversarial attacks. In its essence, we need to solve a problem of statistical estimation which is usually very time-consuming since we need to perform numerous (usually $10^5$) forward passes of the classifier for every point to be certified. In this paper, we review the statistical estimation problems for randomized smoothing to find out if the computational burden is necessary. In particular, we consider the (standard) task of adversarial robustness where we need to decide if a point is robust at a certain radius or not using as few samples as possible while maintaining statistical guarantees. We present estimation procedures employing confidence sequences enjoying the same statistical guarantees as the standard methods, with the optimal sample complexities for the estimation task and empirically demonstrate their good performance. Additionally, we provide a randomized version of Clopper-Pearson confidence intervals resulting in strictly stronger certificates.

摘要: 随机平滑是一种流行的对抗对手攻击的认证防御方法。本质上，我们需要解决通常非常耗时的统计估计问题，因为我们需要为要认证的每个点执行许多(通常是$10^5$)分类器的前向传递。在本文中，我们回顾了随机平滑的统计估计问题，以确定是否有必要增加计算负担。特别是，我们考虑了对抗健壮性的(标准)任务，其中我们需要确定一个点在特定半径处是否健壮，或者在保持统计保证的同时不使用尽可能少的样本。我们提出了使用具有与标准方法相同的统计保证的置信度序列的估计方法，对于估计任务具有最优的样本复杂性，并通过经验证明了其良好的性能。此外，我们还提供了Clopper-Pearson可信区间的随机化版本，从而产生严格更强的证书。



## **14. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

合成人脸图像检测：准确性、鲁棒性、概括性 cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17547v1) [paper-pdf](http://arxiv.org/pdf/2406.17547v1)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.

摘要: 进行了合成人脸图像检测的实验研究。我们收集了一个名为FF 5的数据集，包含五个假面部图像生成器，包括最近的扩散模型。我们发现，在特定图像生成器上训练的简单模型可以在分离合成图像和真实图像方面实现近乎完美的准确性。该模型通过使用数据增强来处理常见的图像失真（分辨率降低、压缩）。此外，还可以识别部分操纵（通过修补将合成图像混合到真实图像中），并通过YOLO架构的简单模型来本地化操纵区域。然而，事实证明，该模型很容易受到对抗攻击，并且不能推广到看不见的生成器。最近的最先进方法也会出现无法概括检测由较新生成器产生的图像的情况，我们在Realistic Vision上进行了测试，Realistic Vision是StabilityAI的Stable Dispatch图像生成器的微调版本。



## **15. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

通过自我提示校准对微调大型语言模型的实用成员推断攻击 cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **16. TSynD: Targeted Synthetic Data Generation for Enhanced Medical Image Classification**

TSynD：用于增强医学图像分类的有针对性的合成数据生成 cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17473v1) [paper-pdf](http://arxiv.org/pdf/2406.17473v1)

**Authors**: Joshua Niemeijer, Jan Ehrhardt, Hristina Uzunova, Heinz Handels

**Abstract**: The usage of medical image data for the training of large-scale machine learning approaches is particularly challenging due to its scarce availability and the costly generation of data annotations, typically requiring the engagement of medical professionals. The rapid development of generative models allows towards tackling this problem by leveraging large amounts of realistic synthetically generated data for the training process. However, randomly choosing synthetic samples, might not be an optimal strategy.   In this work, we investigate the targeted generation of synthetic training data, in order to improve the accuracy and robustness of image classification. Therefore, our approach aims to guide the generative model to synthesize data with high epistemic uncertainty, since large measures of epistemic uncertainty indicate underrepresented data points in the training set. During the image generation we feed images reconstructed by an auto encoder into the classifier and compute the mutual information over the class-probability distribution as a measure for uncertainty.We alter the feature space of the autoencoder through an optimization process with the objective of maximizing the classifier uncertainty on the decoded image. By training on such data we improve the performance and robustness against test time data augmentations and adversarial attacks on several classifications tasks.

摘要: 将医学图像数据用于大规模机器学习方法的培训尤其具有挑战性，因为它的可用性很少，而且生成数据注释的成本很高，通常需要医疗专业人员参与。生成性模型的快速发展允许通过利用大量真实的综合生成的数据来解决这一问题。然而，随机选择合成样本，可能不是最优策略。为了提高图像分类的准确性和稳健性，本文对合成训练数据的定向生成进行了研究。因此，我们的方法旨在指导生成模型合成具有高认知不确定性的数据，因为认知不确定性的大量测量表明训练集中的数据点代表不足。在图像生成过程中，我们将由自动编码器重建的图像送入分类器，计算类别概率分布上的互信息作为不确定性的度量，并通过优化过程改变自动编码器的特征空间，以最大化解码图像上的分类器不确定性为目标。通过对这些数据的训练，我们提高了对测试时间数据增加和对几个分类任务的敌意攻击的性能和稳健性。



## **17. Low-Cost Privacy-Aware Decentralized Learning**

低成本隐私意识的去中心化学习 cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2403.11795v2) [paper-pdf](http://arxiv.org/pdf/2403.11795v2)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that exploits correlated noise to provide strong privacy protection against a local adversary while yielding efficient convergence guarantees for a low communication cost. The progressive neutralization of the added noise during the distributed aggregation process results in ZIP-DL fostering a high model accuracy under privacy guarantees. ZIP-DL further uses a single communication round between each gradient descent, thus minimizing communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL significantly outperforms the state-of-the-art in terms of vulnerability/accuracy trade-off. In particular, ZIP-DL (i) reduces the efficacy of linkability attacks by up to 52 percentage points compared to baseline DL, (ii) improves accuracy by up to 37 percent w.r.t. the state-of-the-art privacy-preserving mechanism operating under the same threat model as ours, when configured to provide the same protection against membership inference attacks, and (iii) reduces communication by up to 10.5x against the same competitor for the same level of protection.

摘要: 介绍了一种新的隐私感知分散学习算法ZIP-DL，该算法利用相关噪声来提供对本地攻击者的强隐私保护，同时以较低的通信代价产生高效的收敛保证。分布式聚合过程中添加的噪声的渐进式中和导致ZIP-DL在隐私保证下培养高模型精度。Zip-DL还在每个梯度下降之间使用单个通信轮次，从而将通信开销降至最低。我们为收敛速度和隐私保证提供了理论上的保证，从而使ZIP-DL适用于实际场景。我们广泛的实验研究表明，ZIP-DL在脆弱性和准确性之间的权衡方面显著优于最先进的ZIP-DL。特别是，与基准DL相比，ZIP-DL(I)将链接性攻击的有效性降低了高达52个百分点，(Ii)将准确率提高了高达37%。最先进的隐私保护机制在与我们相同的威胁模型下运行，当配置为提供相同的成员身份推理攻击保护时，并且(Iii)在相同保护级别的情况下，针对相同竞争对手的通信最多减少10.5倍。



## **18. CuDA2: An approach for Incorporating Traitor Agents into Cooperative Multi-Agent Systems**

CuDA 2：一种将叛徒代理融入合作多代理系统的方法 cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17425v1) [paper-pdf](http://arxiv.org/pdf/2406.17425v1)

**Authors**: Zhen Chen, Yong Liao, Youpeng Zhao, Zipeng Dai, Jian Zhao

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (CMARL) strategies are well known to be vulnerable to adversarial perturbations. Previous works on adversarial attacks have primarily focused on white-box attacks that directly perturb the states or actions of victim agents, often in scenarios with a limited number of attacks. However, gaining complete access to victim agents in real-world environments is exceedingly difficult. To create more realistic adversarial attacks, we introduce a novel method that involves injecting traitor agents into the CMARL system. We model this problem as a Traitor Markov Decision Process (TMDP), where traitors cannot directly attack the victim agents but can influence their formation or positioning through collisions. In TMDP, traitors are trained using the same MARL algorithm as the victim agents, with their reward function set as the negative of the victim agents' reward. Despite this, the training efficiency for traitors remains low because it is challenging for them to directly associate their actions with the victim agents' rewards. To address this issue, we propose the Curiosity-Driven Adversarial Attack (CuDA2) framework. CuDA2 enhances the efficiency and aggressiveness of attacks on the specified victim agents' policies while maintaining the optimal policy invariance of the traitors. Specifically, we employ a pre-trained Random Network Distillation (RND) module, where the extra reward generated by the RND module encourages traitors to explore states unencountered by the victim agents. Extensive experiments on various scenarios from SMAC demonstrate that our CuDA2 framework offers comparable or superior adversarial attack capabilities compared to other baselines.

摘要: 众所周知，协作多智能体强化学习(CMARL)策略容易受到对手扰动的影响。以前关于对抗性攻击的研究主要集中在白盒攻击上，白盒攻击直接扰乱受害者代理的状态或行动，通常在攻击次数有限的情况下。然而，在现实环境中完全接触受害者代理是极其困难的。为了创建更真实的对抗性攻击，我们引入了一种新的方法，涉及到向CMARL系统中注入叛徒代理。我们将这个问题建模为叛徒马尔可夫决策过程(TMDP)，其中叛徒不能直接攻击受害者代理，但可以通过碰撞影响他们的队形或定位。在TMDP中，叛逆者使用与受害者代理相同的Marl算法进行训练，其奖励函数设置为受害者代理奖励的负值。尽管如此，叛徒的培训效率仍然很低，因为他们很难将自己的行动与受害者特工的奖励直接联系起来。为了解决这个问题，我们提出了好奇心驱动的对抗性攻击(CuDA2)框架。CuDA2在保持叛徒最优策略不变性的同时，提高了对指定受害代理策略的攻击效率和攻击性。具体地说，我们采用了一个预先训练的随机网络蒸馏(RND)模块，其中RND模块产生的额外奖励鼓励叛徒探索受害者代理未遇到的状态。来自SMAC的各种场景的广泛实验表明，与其他基线相比，我们的CuDA2框架提供了类似或更好的对抗性攻击能力。



## **19. Nakamoto Consensus under Bounded Processing Capacity**

有限处理能力下的中本共识 cs.CR

ACM Conference on Computer and Communications Security (CCS) 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2303.09113v4) [paper-pdf](http://arxiv.org/pdf/2303.09113v4)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with finite communication- and computation-resources, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of NC fail to answer this question, because their bounded-delay model does not capture the rate limits to nodes' processing of blocks, which cause congestion when blocks are released in quick succession. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW NC in a bounded-capacity model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack we call the teasing strategy, that exploits congestion, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making traditional PoS NC insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of PoS NC we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.

摘要: 对于Nakamoto的最长链共识协议，其工作证明(PoW)和风险证明(Pos)变体为比特币和Cardano等主要区块链提供支持，我们重温了安全与性能权衡的经典问题：给定一个通信和计算资源有限的节点网络，对于给定的块生产率，相对于对手能力的哪一部分，Nakamoto共识(NC)是安全的？最新的NC分析未能回答这个问题，因为它们的有界延迟模型没有考虑到节点对块处理的速率限制，当块被快速连续释放时，会导致拥塞。我们开发了一种新的分析技术来证明在有限容量模型下POW NC的改进的安全和性能权衡。在这个模型中，我们证明了与经典的有界延迟模型相比，Nakamoto的私人攻击不再是最糟糕的攻击，而我们称之为利用拥塞的一种新的攻击策略是严格更差的。在POS中，模棱两可的块会加剧拥塞，使得传统的POS NC不安全，除非在非常低的块生产率下。为了应对这种模棱两可的垃圾邮件，我们提出了一种POS NC的变体，我们称之为BLANC(BLANC)，它具有与POW NC相同的弹性。



## **20. I Don't Know You, But I Can Catch You: Real-Time Defense against Diverse Adversarial Patches for Object Detectors**

我不认识你，但我能抓住你：对象检测器的各种对抗补丁的实时防御 cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.10285v2) [paper-pdf](http://arxiv.org/pdf/2406.10285v2)

**Authors**: Zijin Lin, Yue Zhao, Kai Chen, Jinwen He

**Abstract**: Deep neural networks (DNNs) have revolutionized the field of computer vision like object detection with their unparalleled performance. However, existing research has shown that DNNs are vulnerable to adversarial attacks. In the physical world, an adversary could exploit adversarial patches to implement a Hiding Attack (HA) which patches the target object to make it disappear from the detector, and an Appearing Attack (AA) which fools the detector into misclassifying the patch as a specific object. Recently, many defense methods for detectors have been proposed to mitigate the potential threats of adversarial patches. However, such methods still have limitations in generalization, robustness and efficiency. Most defenses are only effective against the HA, leaving the detector vulnerable to the AA.   In this paper, we propose \textit{NutNet}, an innovative model for detecting adversarial patches, with high generalization, robustness and efficiency. With experiments for six detectors including YOLOv2-v4, SSD, Faster RCNN and DETR on both digital and physical domains, the results show that our proposed method can effectively defend against both the HA and AA, with only 0.4\% sacrifice of the clean performance. We compare NutNet with four baseline defense methods for detectors, and our method exhibits an average defense performance that is over 2.4 times and 4.7 times higher than existing approaches for HA and AA, respectively. In addition, NutNet only increases the inference time by 8\%, which can meet the real-time requirements of the detection systems. Demos of NutNet are available at: \url{https://sites.google.com/view/nutnet}.

摘要: 深度神经网络(DNN)以其无可比拟的性能给计算机视觉领域带来了革命性的变化，就像目标检测一样。然而，现有的研究表明，DNN很容易受到对抗性攻击。在现实世界中，敌手可以利用敌意补丁来实施隐藏攻击(HA)和外观攻击(AA)，前者对目标对象进行补丁以使其从检测器中消失，后者欺骗检测器将该补丁错误分类为特定对象。最近，许多检测器的防御方法被提出以缓解敌意补丁的潜在威胁。然而，这些方法在泛化、稳健性和效率方面仍有局限性。大多数防御措施只对医管局有效，使探测器容易受到机管局的攻击。本文提出了一种具有较高泛化能力、健壮性和效率的敌意补丁检测模型--Texttit{NutNet}。对YOLOv2-v4、SSD、较快的RCNN和DETR等6种检测器在数字域和物理域上的实验结果表明，我们提出的方法可以有效地防御HA和AA，而清洁性能只有0.4%的损失。我们将NutNet与四种检测器基线防御方法进行了比较，我们的方法对HA和AA的平均防御性能分别比现有方法高2.4倍和4.7倍以上。此外，NutNet只增加了8%的推理时间，能够满足检测系统的实时性要求。有关NutNet的演示，请访问：\url{https://sites.google.com/view/nutnet}.



## **21. ECLIPSE: Expunging Clean-label Indiscriminate Poisons via Sparse Diffusion Purification**

ECLIPSE：通过稀疏扩散纯化消除清洁标签不加区别的毒药 cs.CR

Accepted by ESORICS 2024

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.15093v2) [paper-pdf](http://arxiv.org/pdf/2406.15093v2)

**Authors**: Xianlong Wang, Shengshan Hu, Yechao Zhang, Ziqi Zhou, Leo Yu Zhang, Peng Xu, Wei Wan, Hai Jin

**Abstract**: Clean-label indiscriminate poisoning attacks add invisible perturbations to correctly labeled training images, thus dramatically reducing the generalization capability of the victim models. Recently, some defense mechanisms have been proposed such as adversarial training, image transformation techniques, and image purification. However, these schemes are either susceptible to adaptive attacks, built on unrealistic assumptions, or only effective against specific poison types, limiting their universal applicability. In this research, we propose a more universally effective, practical, and robust defense scheme called ECLIPSE. We first investigate the impact of Gaussian noise on the poisons and theoretically prove that any kind of poison will be largely assimilated when imposing sufficient random noise. In light of this, we assume the victim has access to an extremely limited number of clean images (a more practical scene) and subsequently enlarge this sparse set for training a denoising probabilistic model (a universal denoising tool). We then begin by introducing Gaussian noise to absorb the poisons and then apply the model for denoising, resulting in a roughly purified dataset. Finally, to address the trade-off of the inconsistency in the assimilation sensitivity of different poisons by Gaussian noise, we propose a lightweight corruption compensation module to effectively eliminate residual poisons, providing a more universal defense approach. Extensive experiments demonstrate that our defense approach outperforms 10 state-of-the-art defenses. We also propose an adaptive attack against ECLIPSE and verify the robustness of our defense scheme. Our code is available at https://github.com/CGCL-codes/ECLIPSE.

摘要: 干净标签的不分青红皂白的中毒攻击给正确标记的训练图像添加了不可见的扰动，从而大大降低了受害者模型的泛化能力。近年来，一些防御机制被提出，如对抗性训练、图像变换技术、图像净化等。然而，这些方案要么容易受到适应性攻击，要么建立在不切实际的假设之上，要么只对特定的毒物类型有效，限制了它们的普遍适用性。在这项研究中，我们提出了一种更普遍有效、实用和健壮的防御方案，称为ECLIPSE。我们首先研究了高斯噪声对毒物的影响，并从理论上证明了当施加足够的随机噪声时，任何一种毒物都会被很大程度上同化。有鉴于此，我们假设受害者可以访问数量极其有限的干净图像(更实际的场景)，并随后扩大该稀疏集合以训练去噪概率模型(通用去噪工具)。然后，我们首先引入高斯噪声来吸收毒物，然后应用该模型进行去噪，得到一个大致纯化的数据集。最后，针对高斯噪声对不同毒物同化敏感度不一致的问题，提出了一种轻量级的腐败补偿模块来有效消除残留毒物，提供了一种更通用的防御方法。广泛的实验表明，我们的防御方法超过了10种最先进的防御方法。我们还提出了一种针对ECLIPSE的自适应攻击，并验证了该防御方案的健壮性。我们的代码可以在https://github.com/CGCL-codes/ECLIPSE.上找到



## **22. Automated Adversarial Discovery for Safety Classifiers**

安全分类器的自动对抗发现 cs.CL

Published at Fourth Workshop on TrustworthyNLP (TrustNLP) at NAACL  2024

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.17104v1) [paper-pdf](http://arxiv.org/pdf/2406.17104v1)

**Authors**: Yash Kumar Lal, Preethi Lahoti, Aradhana Sinha, Yao Qin, Ananth Balashankar

**Abstract**: Safety classifiers are critical in mitigating toxicity on online forums such as social media and in chatbots. Still, they continue to be vulnerable to emergent, and often innumerable, adversarial attacks. Traditional automated adversarial data generation methods, however, tend to produce attacks that are not diverse, but variations of previously observed harm types. We formalize the task of automated adversarial discovery for safety classifiers - to find new attacks along previously unseen harm dimensions that expose new weaknesses in the classifier. We measure progress on this task along two key axes (1) adversarial success: does the attack fool the classifier? and (2) dimensional diversity: does the attack represent a previously unseen harm type? Our evaluation of existing attack generation methods on the CivilComments toxicity task reveals their limitations: Word perturbation attacks fail to fool classifiers, while prompt-based LLM attacks have more adversarial success, but lack dimensional diversity. Even our best-performing prompt-based method finds new successful attacks on unseen harm dimensions of attacks only 5\% of the time. Automatically finding new harmful dimensions of attack is crucial and there is substantial headroom for future research on our new task.

摘要: 在社交媒体和聊天机器人等在线论坛上，安全分类器对于减轻毒性至关重要。尽管如此，他们仍然很容易受到突然出现的、往往是无数的敌意攻击。然而，传统的自动对抗性数据生成方法往往产生的攻击不是多样化的，而是先前观察到的危害类型的变化。我们将安全分类器的自动敌意发现任务正式化--沿着以前未见过的危害维度发现新的攻击，从而暴露出分类器中的新弱点。我们沿着两个关键轴线衡量这项任务的进展：(1)对手的成功：攻击愚弄了分类器吗？以及(2)维度多样性：攻击是否代表了一种以前未见过的危害类型？我们在CivilComments毒性任务上对现有攻击生成方法的评估揭示了它们的局限性：单词扰动攻击无法愚弄分类器，而基于提示的LLM攻击具有更强的对抗性成功，但缺乏维度多样性。即使是我们性能最好的基于提示的方法，也只能在5%的时间内发现新的成功攻击，攻击的不可见危害维度。自动发现新的有害攻击维度是至关重要的，未来对我们的新任务的研究有很大的余地。



## **23. Robust Distribution Learning with Local and Global Adversarial Corruptions**

具有本地和全球对抗性腐蚀的稳健分布学习 cs.LG

Accepted for presentation at the Conference on Learning Theory (COLT)  2024

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.06509v2) [paper-pdf](http://arxiv.org/pdf/2406.06509v2)

**Authors**: Sloan Nietert, Ziv Goldfeld, Soroosh Shafiee

**Abstract**: We consider learning in an adversarial environment, where an $\varepsilon$-fraction of samples from a distribution $P$ are arbitrarily modified (global corruptions) and the remaining perturbations have average magnitude bounded by $\rho$ (local corruptions). Given access to $n$ such corrupted samples, we seek a computationally efficient estimator $\hat{P}_n$ that minimizes the Wasserstein distance $\mathsf{W}_1(\hat{P}_n,P)$. In fact, we attack the fine-grained task of minimizing $\mathsf{W}_1(\Pi_\# \hat{P}_n, \Pi_\# P)$ for all orthogonal projections $\Pi \in \mathbb{R}^{d \times d}$, with performance scaling with $\mathrm{rank}(\Pi) = k$. This allows us to account simultaneously for mean estimation ($k=1$), distribution estimation ($k=d$), as well as the settings interpolating between these two extremes. We characterize the optimal population-limit risk for this task and then develop an efficient finite-sample algorithm with error bounded by $\sqrt{\varepsilon k} + \rho + \tilde{O}(d\sqrt{k}n^{-1/(k \lor 2)})$ when $P$ has bounded covariance. This guarantee holds uniformly in $k$ and is minimax optimal up to the sub-optimality of the plug-in estimator when $\rho = \varepsilon = 0$. Our efficient procedure relies on a novel trace norm approximation of an ideal yet intractable 2-Wasserstein projection estimator. We apply this algorithm to robust stochastic optimization, and, in the process, uncover a new method for overcoming the curse of dimensionality in Wasserstein distributionally robust optimization.

摘要: 我们考虑在对抗性环境中学习，其中来自分布$P$的$\varepsilon$-分数样本被任意修改(全局破坏)，而其余扰动的平均幅度由$\rho$(局部破坏)限定。在给定$n$这样的破坏样本的情况下，我们寻找一个计算上有效的估计量$\hat{P}_n$以最小化Wasserstein距离$\mathsf{W}_1(\hat{P}_n，P)$。事实上，我们对所有的正交投影$\pI\in\mathbb{R}^{d\time d}$发起了最小化$\mathsf{W}_1(\pI_#\hat{P}_n，\pI_#P)$的细粒度任务，并且性能伸缩为$\mathm{RANK}(\pI)=k$。这允许我们同时考虑平均值估计($k=1$)、分布估计($k=d$)以及在这两个极值之间插入的设置。我们刻画了该任务的最优总体极限风险，并在此基础上提出了一个有效的有限样本算法，当$P的协方差有界时，误差有界于$Sqrt{varepsilon k}+Rho+tide{O}(dSqrt{k}n^-1/(Kor2)})$。这个保证在$k$中一致成立，并且当$\rho=\varepsilon=0$时，它是最小极大最优的，直到插件估计器的次最优。我们的有效过程依赖于一个理想但难以处理的2-Wasserstein投影估计量的一个新的迹范数近似。我们将该算法应用于稳健随机优化中，并在此过程中发现了一种克服Wasserstein分布稳健优化中的维度灾难的新方法。



## **24. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2402.17012v3) [paper-pdf](http://arxiv.org/pdf/2402.17012v3)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度预测训练数据的成员资格，以及这种攻击的一个变体，它只需要通过利用LLMS上最近的模型窃取工作来对模型进行Logit访问。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。我们的代码可以在githorb.com/Safr-ai-lab/pandora-llm上找到。



## **25. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZER：Red将大型语言模型与自动生成的越狱脚本结合起来 cs.AI

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2309.10253v3) [paper-pdf](http://arxiv.org/pdf/2309.10253v3)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **26. Security of Partially Corrupted Repeater Chains**

部分损坏的转发链的安全性 quant-ph

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16651v1) [paper-pdf](http://arxiv.org/pdf/2406.16651v1)

**Authors**: Adrian Harkness, Walter O. Krawec, Bing Wang

**Abstract**: Quantum Key Distribution allows two parties to establish a secret key that is secure against computationally unbounded adversaries. To extend the distance between parties, quantum networks, and in particular repeater chains, are vital. Typically, security in such scenarios assumes the absolute worst case: namely, an adversary has complete control over all repeaters and fiber links in a network and is able to replace them with perfect devices, thus allowing her to hide her attack within the expected natural noise. In a large-scale network, however, such a powerful attack may be infeasible. In this paper, we analyze the case where the adversary can only corrupt a contiguous subset of a repeater chain connecting Alice and Bob, while some portion of the network near Alice and Bob may be considered safe from attack (though still noisy). We derive a rigorous finite key proof of security assuming this attack model and show that improved performance and noise tolerances are possible.

摘要: 量子密钥分发允许双方建立针对计算无界对手安全的秘密密钥。为了扩大各方之间的距离，量子网络，特别是转发器链至关重要。通常，此类场景中的安全性假设绝对最坏的情况：即，对手完全控制网络中的所有转发器和光纤链路，并能够用完美的设备替换它们，从而使她能够将攻击隐藏在预期的自然噪音中。然而，在大规模网络中，如此强大的攻击可能不可行。在本文中，我们分析了对手只能破坏连接爱丽丝和鲍勃的转发器链的连续子集的情况，而靠近爱丽丝和鲍勃的网络的某个部分可能被认为是安全的，免受攻击（尽管仍然有噪音）。假设这种攻击模型，我们推导出严格的有限密钥安全性证明，并表明改进的性能和抗噪能力是可能的。



## **27. UNICAD: A Unified Approach for Attack Detection, Noise Reduction and Novel Class Identification**

UNICAD：攻击检测、降噪和新型类别识别的统一方法 cs.CV

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16501v1) [paper-pdf](http://arxiv.org/pdf/2406.16501v1)

**Authors**: Alvaro Lopez Pellicer, Kittipos Giatgong, Yi Li, Neeraj Suri, Plamen Angelov

**Abstract**: As the use of Deep Neural Networks (DNNs) becomes pervasive, their vulnerability to adversarial attacks and limitations in handling unseen classes poses significant challenges. The state-of-the-art offers discrete solutions aimed to tackle individual issues covering specific adversarial attack scenarios, classification or evolving learning. However, real-world systems need to be able to detect and recover from a wide range of adversarial attacks without sacrificing classification accuracy and to flexibly act in {\bf unseen} scenarios. In this paper, UNICAD, is proposed as a novel framework that integrates a variety of techniques to provide an adaptive solution.   For the targeted image classification, UNICAD achieves accurate image classification, detects unseen classes, and recovers from adversarial attacks using Prototype and Similarity-based DNNs with denoising autoencoders. Our experiments performed on the CIFAR-10 dataset highlight UNICAD's effectiveness in adversarial mitigation and unseen class classification, outperforming traditional models.

摘要: 随着深度神经网络(DNN)的广泛使用，它们对敌意攻击的脆弱性以及在处理未知类方面的局限性构成了巨大的挑战。最先进的解决方案提供离散的解决方案，旨在解决个别问题，涵盖特定的对抗性攻击场景、分类或不断演变的学习。然而，现实世界的系统需要能够在不牺牲分类准确性的情况下检测并从广泛的对手攻击中恢复，并在{\bf不可见}场景中灵活地操作。在这篇文章中，UNICAD被提出作为一个新的框架，它集成了各种技术来提供一个自适应的解决方案。对于目标图像分类，UNICAD实现了准确的图像分类，检测不可见的类别，并使用带有去噪自动编码器的原型和基于相似性的DNN从对手攻击中恢复。我们在CIFAR-10数据集上进行的实验突出了UNICAD在对抗性缓解和看不见的类别分类方面的有效性，优于传统模型。



## **28. The Economic Limits of Permissionless Consensus**

无许可共识的经济局限 cs.DC

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2405.09173v2) [paper-pdf](http://arxiv.org/pdf/2405.09173v2)

**Authors**: Eric Budish, Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The purpose of a consensus protocol is to keep a distributed network of nodes "in sync," even in the presence of an unpredictable communication network and adversarial behavior by some of the participating nodes. In the permissionless setting, these nodes may be operated by unknown players, with each player free to use multiple identifiers and to start or stop running the protocol at any time. Establishing that a permissionless consensus protocol is "secure" thus requires both a distributed computing argument (that the protocol guarantees consistency and liveness unless the fraction of adversarial participation is sufficiently large) and an economic argument (that carrying out an attack would be prohibitively expensive for an attacker). There is a mature toolbox for assembling arguments of the former type; the goal of this paper is to lay the foundations for arguments of the latter type.   An ideal permissionless consensus protocol would, in addition to satisfying standard consistency and liveness guarantees, render consistency violations prohibitively expensive for the attacker without collateral damage to honest participants. We make this idea precise with our notion of the EAAC (expensive to attack in the absence of collapse) property, and prove the following results:   1. In the synchronous and dynamically available setting, with an adversary that controls at least one-half of the overall resources, no protocol can be EAAC.   2. In the partially synchronous and quasi-permissionless setting, with an adversary that controls at least one-third of the overall resources, no protocol can be EAAC.   3. In the synchronous and quasi-permissionless setting, there is a proof-of-stake protocol that, provided the adversary controls less than two-thirds of the overall stake, satisfies the EAAC property.   All three results are optimal with respect to the size of the adversary.

摘要: 共识协议的目的是保持分布式节点网络的同步，即使在一些参与节点存在不可预测的通信网络和敌对行为的情况下也是如此。在未经许可的设置中，这些节点可以由未知玩家操作，每个玩家可以自由使用多个标识符，并且可以随时开始或停止运行协议。因此，要确定未经许可的共识协议是“安全的”，既需要分布式计算论证(该协议保证一致性和活性，除非敌方参与的比例足够大)，也需要经济论证(对攻击者来说，实施攻击的代价高得令人望而却步)。有一个成熟的工具箱用于组合前一种类型的参数；本文的目标是为后一种类型的参数奠定基础。一个理想的未经许可的协商一致协议，除了满足标准的一致性和活跃性保证外，还将使违反一致性的行为对攻击者来说代价高昂，而不会对诚实的参与者造成附带损害。我们用我们的EAAC(在没有崩溃的情况下攻击代价很高)属性的概念精确地表达了这个想法，并证明了以下结果：1.在同步和动态可用的环境中，如果对手控制了至少一半的总资源，则没有协议可以成为EAAC。2.在部分同步和准无许可的设置中，在对手控制至少三分之一的总资源的情况下，没有协议可以是EAAC。3.在同步和准无许可的设置中，存在一种风险证明协议，如果对手控制的总风险少于三分之二，则满足EAAC属性。就对手的规模而言，这三个结果都是最优的。



## **29. Investigating the Influence of Prompt-Specific Shortcuts in AI Generated Text Detection**

调查预算特定快捷方式对人工智能生成文本检测的影响 cs.CL

19 pages, 3 figures, 13 tables, under review

**SubmitDate**: 2024-06-24    [abs](http://arxiv.org/abs/2406.16275v1) [paper-pdf](http://arxiv.org/pdf/2406.16275v1)

**Authors**: Choonghyun Park, Hyuhng Joon Kim, Junyeob Kim, Youna Kim, Taeuk Kim, Hyunsoo Cho, Hwiyeol Jo, Sang-goo Lee, Kang Min Yoo

**Abstract**: AI Generated Text (AIGT) detectors are developed with texts from humans and LLMs of common tasks. Despite the diversity of plausible prompt choices, these datasets are generally constructed with a limited number of prompts. The lack of prompt variation can introduce prompt-specific shortcut features that exist in data collected with the chosen prompt, but do not generalize to others. In this paper, we analyze the impact of such shortcuts in AIGT detection. We propose Feedback-based Adversarial Instruction List Optimization (FAILOpt), an attack that searches for instructions deceptive to AIGT detectors exploiting prompt-specific shortcuts. FAILOpt effectively drops the detection performance of the target detector, comparable to other attacks based on adversarial in-context examples. We also utilize our method to enhance the robustness of the detector by mitigating the shortcuts. Based on the findings, we further train the classifier with the dataset augmented by FAILOpt prompt. The augmented classifier exhibits improvements across generation models, tasks, and attacks. Our code will be available at https://github.com/zxcvvxcz/FAILOpt.

摘要: AI生成文本(AIGT)检测器是用来自人类和常见任务的LLM的文本开发的。尽管有各种貌似合理的提示选项，但这些数据集通常是用有限数量的提示构建的。缺少提示变化可能会引入特定于提示的快捷方式功能，这些功能存在于与所选提示一起收集的数据中，但不适用于其他提示。本文分析了这些捷径对AIGT检测的影响。我们提出了基于反馈的对抗性指令列表优化(FAILOpt)，这是一种利用提示特定的快捷方式搜索对AIGT检测器具有欺骗性的指令的攻击。FAILOpt有效地降低了目标检测器的检测性能，与基于对抗性上下文中示例的其他攻击相当。我们还利用我们的方法通过减少捷径来增强检测器的健壮性。在此基础上，进一步用FAILOpt提示扩充的数据集训练分类器。增强的分类器在生成模型、任务和攻击方面都有改进。我们的代码将在https://github.com/zxcvvxcz/FAILOpt.上提供



## **30. Pareto Adversarial Robustness: Balancing Spatial Robustness and Sensitivity-based Robustness**

帕累托对抗稳健性：平衡空间稳健性和基于敏感性的稳健性 cs.LG

Published in SCIENCE CHINA Information Sciences (SCIS) in 2023.  Please also refer to the published version in the Journal reference  https://www.sciengine.com/SCIS/doi/10.1007/s11432-022-3861-8

**SubmitDate**: 2024-06-23    [abs](http://arxiv.org/abs/2111.01996v3) [paper-pdf](http://arxiv.org/pdf/2111.01996v3)

**Authors**: Ke Sun, Mingjie Li, Zhouchen Lin

**Abstract**: Adversarial robustness, which primarily comprises sensitivity-based robustness and spatial robustness, plays an integral part in achieving robust generalization. In this paper, we endeavor to design strategies to achieve universal adversarial robustness. To achieve this, we first investigate the relatively less-explored realm of spatial robustness. Then, we integrate the existing spatial robustness methods by incorporating both local and global spatial vulnerability into a unified spatial attack and adversarial training approach. Furthermore, we present a comprehensive relationship between natural accuracy, sensitivity-based robustness, and spatial robustness, supported by strong evidence from the perspective of robust representation. Crucially, to reconcile the interplay between the mutual impacts of various robustness components into one unified framework, we incorporate the \textit{Pareto criterion} into the adversarial robustness analysis, yielding a novel strategy called Pareto Adversarial Training for achieving universal robustness. The resulting Pareto front, which delineates the set of optimal solutions, provides an optimal balance between natural accuracy and various adversarial robustness. This sheds light on solutions for achieving universal robustness in the future. To the best of our knowledge, we are the first to consider universal adversarial robustness via multi-objective optimization.

摘要: 对抗性稳健性主要包括基于敏感度的稳健性和空间稳健性，是实现健壮性泛化的重要组成部分。在这篇文章中，我们努力设计策略来实现普遍的对抗健壮性。为了实现这一点，我们首先研究相对较少被探索的空间稳健性领域。然后，通过将局部和全局空间脆弱性结合到一个统一的空间攻击和对抗性训练方法中，整合了现有的空间稳健性方法。此外，我们从稳健表示的角度提出了自然准确性、基于敏感度的稳健性和空间稳健性之间的综合关系，并得到了强有力的证据的支持。重要的是，为了将不同健壮性成分之间的相互影响协调到一个统一的框架中，我们将文本{Pareto准则}引入到对抗健壮性分析中，产生了一种新的策略，称为Pareto对抗训练，以实现普遍的健壮性。由此产生的帕累托前沿描述了一组最优解，在自然准确性和各种对手稳健性之间提供了最佳平衡。这为未来实现普遍的健壮性提供了解决方案。据我们所知，我们是第一个通过多目标优化来考虑普遍对抗健壮性的人。



## **31. Federated Adversarial Learning for Robust Autonomous Landing Runway Detection**

用于鲁棒自主着陆跑道检测的联邦对抗学习 cs.CV

ICANN2024

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.15925v1) [paper-pdf](http://arxiv.org/pdf/2406.15925v1)

**Authors**: Yi Li, Plamen Angelov, Zhengxin Yu, Alvaro Lopez Pellicer, Neeraj Suri

**Abstract**: As the development of deep learning techniques in autonomous landing systems continues to grow, one of the major challenges is trust and security in the face of possible adversarial attacks. In this paper, we propose a federated adversarial learning-based framework to detect landing runways using paired data comprising of clean local data and its adversarial version. Firstly, the local model is pre-trained on a large-scale lane detection dataset. Then, instead of exploiting large instance-adaptive models, we resort to a parameter-efficient fine-tuning method known as scale and shift deep features (SSF), upon the pre-trained model. Secondly, in each SSF layer, distributions of clean local data and its adversarial version are disentangled for accurate statistics estimation. To the best of our knowledge, this marks the first instance of federated learning work that address the adversarial sample problem in landing runway detection. Our experimental evaluations over both synthesis and real images of Landing Approach Runway Detection (LARD) dataset consistently demonstrate good performance of the proposed federated adversarial learning and robust to adversarial attacks.

摘要: 随着自主着陆系统中深度学习技术的发展，主要挑战之一是面对可能的对手攻击时的信任和安全。在本文中，我们提出了一种基于联合对抗性学习的框架，使用由干净的本地数据及其对抗性版本组成的配对数据来检测着陆跑道。首先，在大规模车道检测数据集上对局部模型进行预训练。然后，我们没有使用大型的实例自适应模型，而是在预先训练的模型上求助于一种参数高效的微调方法，称为尺度和移位深度特征(SSF)。其次，在每个SSF层中，将干净的本地数据及其敌对版本的分布分开，以便进行准确的统计估计。据我们所知，这标志着联邦学习工作的第一个实例，解决了着陆跑道检测中的对抗性样本问题。我们在着陆进场跑道检测(LARD)合成图像和真实图像上的实验评估一致地表明，所提出的联合对抗性学习算法具有良好的性能和对对抗性攻击的健壮性。



## **32. A Recipe for Improved Certifiable Robustness**

提高可认证稳健性的配方 cs.LG

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2310.02513v2) [paper-pdf](http://arxiv.org/pdf/2310.02513v2)

**Authors**: Kai Hu, Klas Leino, Zifan Wang, Matt Fredrikson

**Abstract**: Recent studies have highlighted the potential of Lipschitz-based methods for training certifiably robust neural networks against adversarial attacks. A key challenge, supported both theoretically and empirically, is that robustness demands greater network capacity and more data than standard training. However, effectively adding capacity under stringent Lipschitz constraints has proven more difficult than it may seem, evident by the fact that state-of-the-art approach tend more towards \emph{underfitting} than overfitting. Moreover, we posit that a lack of careful exploration of the design space for Lipshitz-based approaches has left potential performance gains on the table. In this work, we provide a more comprehensive evaluation to better uncover the potential of Lipschitz-based certification methods. Using a combination of novel techniques, design optimizations, and synthesis of prior work, we are able to significantly improve the state-of-the-art VRA for deterministic certification on a variety of benchmark datasets, and over a range of perturbation sizes. Of particular note, we discover that the addition of large ``Cholesky-orthogonalized residual dense'' layers to the end of existing state-of-the-art Lipschitz-controlled ResNet architectures is especially effective for increasing network capacity and performance. Combined with filtered generative data augmentation, our final results further the state of the art deterministic VRA by up to 8.5 percentage points\footnote{Code is available at \url{https://github.com/hukkai/liresnet}}.

摘要: 最近的研究强调了基于Lipschitz的方法在训练具有可证明的健壮性的神经网络对抗对手攻击方面的潜力。理论上和经验上都支持的一个关键挑战是，与标准培训相比，健壮性需要更大的网络容量和更多的数据。然而，事实证明，在严格的Lipschitz约束下有效地增加产能比看起来要困难得多，事实证明，最先进的方法更倾向于{不足}而不是过度匹配。此外，我们假设，缺乏对基于Lipshitz的方法的设计空间的仔细探索，已经留下了潜在的性能收益。在这项工作中，我们提供了一个更全面的评估，以更好地发现基于Lipschitz的认证方法的潜力。使用新技术、设计优化和综合以前的工作，我们能够显著改进最先进的VRA，以在各种基准数据集和一系列扰动大小上进行确定性认证。特别值得注意的是，我们发现在现有最先进的由Lipschitz控制的ResNet体系结构的末尾添加大的“Cholesky-正交化剩余致密”层对于提高网络容量和性能特别有效。结合过滤的生成性数据增强，我们的最终结果将最高可达8.5%的确定性VRA进一步提高\脚注{代码可在\url{https://github.com/hukkai/liresnet}}.



## **33. The Effect of Similarity Measures on Accurate Stability Estimates for Local Surrogate Models in Text-based Explainable AI**

相似性度量对基于文本的可解释人工智能中局部代理模型准确稳定性估计的影响 cs.LG

11 pages, 8 Tables

**SubmitDate**: 2024-06-22    [abs](http://arxiv.org/abs/2406.15839v1) [paper-pdf](http://arxiv.org/pdf/2406.15839v1)

**Authors**: Christopher Burger, Charles Walter, Thai Le

**Abstract**: Recent work has investigated the vulnerability of local surrogate methods to adversarial perturbations on a machine learning (ML) model's inputs, where the explanation is manipulated while the meaning and structure of the original input remains similar under the complex model. While weaknesses across many methods have been shown to exist, the reasons behind why still remain little explored. Central to the concept of adversarial attacks on explainable AI (XAI) is the similarity measure used to calculate how one explanation differs from another A poor choice of similarity measure can result in erroneous conclusions on the efficacy of an XAI method. Too sensitive a measure results in exaggerated vulnerability, while too coarse understates its weakness. We investigate a variety of similarity measures designed for text-based ranked lists including Kendall's Tau, Spearman's Footrule and Rank-biased Overlap to determine how substantial changes in the type of measure or threshold of success affect the conclusions generated from common adversarial attack processes. Certain measures are found to be overly sensitive, resulting in erroneous estimates of stability.

摘要: 最近的工作研究了局部代理方法对机器学习(ML)模型输入的对抗性扰动的脆弱性，其中解释被操纵，而原始输入的含义和结构在复杂模型下保持相似。虽然许多方法的弱点已经被证明存在，但为什么背后的原因仍然很少被探索。对抗性攻击可解释人工智能(XAI)概念的核心是用于计算一种解释与另一种解释的不同之处的相似性度量。如果相似性度量选择不当，可能会导致对XAI方法有效性的错误结论。过于敏感的衡量标准会夸大脆弱性，而过于粗略的衡量标准则会低估其弱点。我们研究了为基于文本的排名列表设计的各种相似性度量，包括Kendall‘s Tau、Spearman’s Footrule和Rank-Biased Overlance，以确定度量类型或成功阈值的实质性变化如何影响从常见的对抗性攻击过程生成的结论。某些措施被发现过于敏感，导致对稳定性的错误估计。



## **34. DataFreeShield: Defending Adversarial Attacks without Training Data**

DataFreeShield：在没有训练数据的情况下防御对抗攻击 cs.LG

ICML 2024

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15635v1) [paper-pdf](http://arxiv.org/pdf/2406.15635v1)

**Authors**: Hyeyoon Lee, Kanghyun Choi, Dain Kwon, Sunjong Park, Mayoore Selvarasa Jaiswal, Noseong Park, Jonghyun Choi, Jinho Lee

**Abstract**: Recent advances in adversarial robustness rely on an abundant set of training data, where using external or additional datasets has become a common setting. However, in real life, the training data is often kept private for security and privacy issues, while only the pretrained weight is available to the public. In such scenarios, existing methods that assume accessibility to the original data become inapplicable. Thus we investigate the pivotal problem of data-free adversarial robustness, where we try to achieve adversarial robustness without accessing any real data. Through a preliminary study, we highlight the severity of the problem by showing that robustness without the original dataset is difficult to achieve, even with similar domain datasets. To address this issue, we propose DataFreeShield, which tackles the problem from two perspectives: surrogate dataset generation and adversarial training using the generated data. Through extensive validation, we show that DataFreeShield outperforms baselines, demonstrating that the proposed method sets the first entirely data-free solution for the adversarial robustness problem.

摘要: 最近在对抗健壮性方面的进展依赖于丰富的训练数据集，其中使用外部或额外的数据集已经成为一种常见的设置。然而，在现实生活中，出于安全和隐私问题，训练数据通常是保密的，而只有预先训练的体重对公众可用。在这种情况下，假定可访问原始数据的现有方法变得不适用。因此，我们研究了无数据对抗稳健性的关键问题，我们试图在不访问任何真实数据的情况下实现对抗稳健性。通过初步研究，我们强调了问题的严重性，因为我们表明，即使使用类似的领域数据集，没有原始数据集的健壮性也很难实现。为了解决这个问题，我们提出了DataFree Shield，它从两个角度来解决这个问题：代理数据集的生成和使用生成的数据进行对抗性训练。通过广泛的验证，我们证明了DataFree Shield的性能优于Baseline，从而证明了该方法为对抗健壮性问题设置了第一个完全无数据的解决方案。



## **35. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2405.15589v2) [paper-pdf](http://arxiv.org/pdf/2405.15589v2)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on four models from different families (Gemma, Phi3, Mistral, Zephyr) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们在不同家族(Gema，Phi3，Mistral，Zephy)和不同尺度(2B，3.8B，7B)的四个模型上的实验评估表明，这两种算法在保持实用性的同时，显著提高了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **36. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

分布风险接受性和鲁棒性下$k$-次模函数的Stackelberg博弈 math.OC

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.13023v2) [paper-pdf](http://arxiv.org/pdf/2406.13023v2)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.

摘要: 我们研究了对抗性环境下的子模优化，适用于机器学习问题，例如使用对不确定性和攻击敏感的数据进行特征选择。我们主要研究攻击者(或中断者)和防御者之间的Stackelberg博弈，其中攻击者的目标是最小化防御者最大化$k$-子模函数的目标。我们允许攻击成功和固有数据噪声带来的不确定性，并解决由于不完全了解随机参数的概率分布而带来的挑战。具体地，我们引入了分布式风险厌恶$k$-子模阻断问题(DRA$k$-SIP)和分布式风险厌恶$k$-子模阻断问题(DRR$k$-SIP)，并给出了有限收敛的精确算法。DRA$k$-SIP解决方案允许风险厌恶中断者针对现实世界的不确定性制定稳健的策略。相反，DRR$k$-SIP解决方案建议攻击者采用攻击性策略，愿意承担(分布式)风险以造成最大损害，识别关键易受攻击的组件，可用于防御者的防御策略。从DRA$k$-SIP和DRR$k$-SIP导出的最佳值为防御者的目标函数的期望值提供了类似于置信度的范围，从而捕获了分布模糊性。我们分别使用特征选择和传感器放置问题的实例以及威斯康星州的乳腺癌数据和合成数据进行了计算实验。



## **37. AdvQuNN: A Methodology for Analyzing the Adversarial Robustness of Quanvolutional Neural Networks**

AdvQuNN：一种分析量子卷积神经网络对抗鲁棒性的方法 quant-ph

7 pages, 6 figures

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2403.05596v2) [paper-pdf](http://arxiv.org/pdf/2403.05596v2)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Mohamed Bennai, Muhammad Shafique

**Abstract**: Recent advancements in quantum computing have led to the development of hybrid quantum neural networks (HQNNs) that employ a mixed set of quantum layers and classical layers, such as Quanvolutional Neural Networks (QuNNs). While several works have shown security threats of classical neural networks, such as adversarial attacks, their impact on QuNNs is still relatively unexplored. This work tackles this problem by designing AdvQuNN, a specialized methodology to investigate the robustness of HQNNs like QuNNs against adversarial attacks. It employs different types of Ansatzes as parametrized quantum circuits and different types of adversarial attacks. This study aims to rigorously assess the influence of quantum circuit architecture on the resilience of QuNN models, which opens up new pathways for enhancing the robustness of QuNNs and advancing the field of quantum cybersecurity. Our results show that, compared to classical convolutional networks, QuNNs achieve up to 60\% higher robustness for the MNIST and 40\% for FMNIST datasets.

摘要: 量子计算的最新进展导致了混合量子神经网络(HQNN)的发展，该混合量子神经网络使用了量子层和经典层的混合集合，例如量子卷积神经网络(QNNS)。虽然一些研究已经显示了经典神经网络的安全威胁，如对抗性攻击，但它们对量子神经网络的影响仍然相对未被探索。这项工作通过设计AdvQuNN来解决这个问题，AdvQuNN是一种专门的方法来研究像QuNN一样的HQNN对对手攻击的健壮性。它使用不同类型的Ansat作为参数化量子电路和不同类型的对抗性攻击。本研究旨在严格评估量子电路体系结构对量子网络模型弹性的影响，为提高量子网络的健壮性和推进量子网络安全领域开辟新的途径。结果表明，与经典卷积网络相比，量子神经网络对MNIST数据集的鲁棒性提高了60%，对FMNIST数据集的鲁棒性提高了40%。



## **38. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

交易魔鬼：通过随机投资模型和Bayesian方法进行强有力的后门攻击 cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.10719v3) [paper-pdf](http://arxiv.org/pdf/2406.10719v3)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.

摘要: 随着语音激活系统和语音识别技术的日益广泛使用，对音频数据进行后门攻击的危险显着增加。这项研究着眼于一种特定类型的攻击，称为基于随机投资的后门攻击（MarketBack），其中对手战略性地操纵音频的风格属性来愚弄语音识别系统。机器学习模型的安全性和完整性受到后门攻击的严重威胁，为了维护音频应用和系统的可靠性，识别此类攻击在音频数据环境中变得至关重要。实验结果表明，当毒害少于1%的训练数据时，MarketBack可以在7个受害者模型中实现接近100%的平均攻击成功率。



## **39. Fingerprint Membership and Identity Inference Against Generative Adversarial Networks**

针对生成性对抗网络的指纹成员资格和身份推断 cs.CV

Paper submitted at "Pattern Recognition Letters", 9 pages, 6 images

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15253v1) [paper-pdf](http://arxiv.org/pdf/2406.15253v1)

**Authors**: Saverio Cavasin, Daniele Mari, Simone Milani, Mauro Conti

**Abstract**: Generative models are gaining significant attention as potential catalysts for a novel industrial revolution. Since automated sample generation can be useful to solve privacy and data scarcity issues that usually affect learned biometric models, such technologies became widely spread in this field. In this paper, we assess the vulnerabilities of generative machine learning models concerning identity protection by designing and testing an identity inference attack on fingerprint datasets created by means of a generative adversarial network. Experimental results show that the proposed solution proves to be effective under different configurations and easily extendable to other biometric measurements.

摘要: 生成模型作为新型工业革命的潜在催化剂正在受到广泛关注。由于自动样本生成对于解决通常影响习得的生物识别模型的隐私和数据稀缺问题很有用，因此此类技术在该领域得到广泛传播。在本文中，我们通过设计和测试对生成式对抗网络创建的指纹数据集的身份推断攻击，评估生成式机器学习模型在身份保护方面的漏洞。实验结果表明，所提出的解决方案在不同配置下有效，并且可以轻松扩展到其他生物识别测量。



## **40. Injecting Bias in Text-To-Image Models via Composite-Trigger Backdoors**

通过复合触发后门在文本到图像模型中注入偏差 cs.LG

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15213v1) [paper-pdf](http://arxiv.org/pdf/2406.15213v1)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasaryan, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional image generative models such as Stable Diffusion, Midjourney, and DALL-E 3 have revolutionized the field of image generation, allowing users to produce high-quality, realistic images from textual prompts. While these developments have enhanced artistic creation and visual communication, they also present an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence society and spread propaganda. In this paper, we demonstrate the possibility of such a bias injection threat by an adversary who backdoors such models with a small number of malicious data samples; the implemented backdoor is activated when special triggers exist in the input prompt of the backdoored models. On the other hand, the model's utility is preserved in the absence of the triggers, making the attack highly undetectable. We present a novel framework that enables efficient generation of poisoning samples with composite (multi-word) triggers for such an attack. Our extensive experiments using over 1 million generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases can bypass conventional detection mechanisms, highlighting the challenges in proving the existence of biases within operational constraints. Our cost analysis confirms the low financial barrier to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in text-to-image generation models.

摘要: 大型文本条件图像生成模型的最新进展，如稳定扩散、中途旅行和Dall-E 3，使图像生成领域发生了革命性的变化，使用户能够从文本提示生成高质量、逼真的图像。虽然这些发展促进了艺术创作和视觉交流，但它们也提供了一个未被开发的攻击机会：可能会导致对手对生成的图像产生偏见，以达到恶意目的，例如影响社会和传播宣传。在本文中，我们证明了这样一种偏见注入威胁的可能性，即攻击者通过少量恶意数据样本对此类模型进行后门；当后门模型的输入提示中存在特殊触发器时，实现的后门被激活。另一方面，在没有触发器的情况下，该模型的实用性被保留下来，使得攻击高度不可检测。我们提出了一种新的框架，它能够高效地生成具有复合(多字)触发的中毒样本来进行此类攻击。我们使用100多万张生成的图像和数百个微调模型进行了广泛的实验，证明了所提出的后门攻击的可行性。我们说明了这些偏差如何绕过传统的检测机制，强调了在操作限制内证明存在偏差的挑战。我们的成本分析证实了执行此类攻击的低财务障碍，强调了针对文本到图像生成模型中此类漏洞的强大防御策略的必要性。



## **41. From LLMs to MLLMs: Exploring the Landscape of Multimodal Jailbreaking**

从LLM到MLLM：探索多模式越狱的格局 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.14859v1) [paper-pdf](http://arxiv.org/pdf/2406.14859v1)

**Authors**: Siyuan Wang, Zhuohan Long, Zhihao Fan, Zhongyu Wei

**Abstract**: The rapid development of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) has exposed vulnerabilities to various adversarial attacks. This paper provides a comprehensive overview of jailbreaking research targeting both LLMs and MLLMs, highlighting recent advancements in evaluation benchmarks, attack techniques and defense strategies. Compared to the more advanced state of unimodal jailbreaking, multimodal domain remains underexplored. We summarize the limitations and potential research directions of multimodal jailbreaking, aiming to inspire future research and further enhance the robustness and security of MLLMs.

摘要: 大型语言模型（LLM）和多模式大型语言模型（MLLM）的快速发展暴露了各种对抗攻击的脆弱性。本文全面概述了针对LLM和MLLM的越狱研究，重点介绍了评估基准、攻击技术和防御策略方面的最新进展。与更先进的单模式越狱相比，多模式领域仍然被探索不足。我们总结了多模式越狱的局限性和潜在研究方向，旨在启发未来的研究并进一步增强MLLM的稳健性和安全性。



## **42. Steering Without Side Effects: Improving Post-Deployment Control of Language Models**

无副作用的转向：改善语言模型的部署后控制 cs.CL

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2406.15518v1) [paper-pdf](http://arxiv.org/pdf/2406.15518v1)

**Authors**: Asa Cooper Stickland, Alexander Lyzhov, Jacob Pfau, Salsabila Mahdi, Samuel R. Bowman

**Abstract**: Language models (LMs) have been shown to behave unexpectedly post-deployment. For example, new jailbreaks continually arise, allowing model misuse, despite extensive red-teaming and adversarial training from developers. Given most model queries are unproblematic and frequent retraining results in unstable user experience, methods for mitigation of worst-case behavior should be targeted. One such method is classifying inputs as potentially problematic, then selectively applying steering vectors on these problematic inputs, i.e. adding particular vectors to model hidden states. However, steering vectors can also negatively affect model performance, which will be an issue on cases where the classifier was incorrect. We present KL-then-steer (KTS), a technique that decreases the side effects of steering while retaining its benefits, by first training a model to minimize Kullback-Leibler (KL) divergence between a steered and unsteered model on benign inputs, then steering the model that has undergone this training. Our best method prevents 44% of jailbreak attacks compared to the original Llama-2-chat-7B model while maintaining helpfulness (as measured by MT-Bench) on benign requests almost on par with the original LM. To demonstrate the generality and transferability of our method beyond jailbreaks, we show that our KTS model can be steered to reduce bias towards user-suggested answers on TruthfulQA. Code is available: https://github.com/AsaCooperStickland/kl-then-steer.

摘要: 语言模型(LMS)在部署后表现出出乎意料的行为。例如，新的越狱事件不断出现，允许滥用模型，尽管开发人员进行了大量的红团队和对抗性培训。考虑到大多数模型查询是没有问题的，并且频繁的重新训练会导致不稳定的用户体验，因此应该有针对性地减少最坏情况下的行为。一种这样的方法是将输入归类为潜在有问题的，然后选择性地对这些有问题的输入应用导向向量，即添加特定向量来对隐藏状态进行建模。然而，方向向量也会对模型性能产生负面影响，这在分类器不正确的情况下将是一个问题。我们提出了KL-Then-Steer(KTS)，这是一种在保留其好处的同时减少转向副作用的技术，首先训练一个模型，以最小化良性输入上转向和非转向模型之间的Kullback-Leibler(KL)发散，然后转向经过这种训练的模型。与原始的Llama-2-Chat-7B模型相比，我们最好的方法防止了44%的越狱攻击，同时保持了对良性请求的帮助(通过MT-BENCH衡量)，几乎与原始的LM相同。为了证明我们的方法在越狱之外的通用性和可转移性，我们证明了我们的KTS模型可以减少对用户建议的关于TruthfulQA的答案的偏见。代码可用：https://github.com/AsaCooperStickland/kl-then-steer.



## **43. FedSecurity: Benchmarking Attacks and Defenses in Federated Learning and Federated LLMs**

FedSecurity：联邦学习和联邦LLM中的攻击和防御基准 cs.CR

**SubmitDate**: 2024-06-21    [abs](http://arxiv.org/abs/2306.04959v5) [paper-pdf](http://arxiv.org/pdf/2306.04959v5)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Carlee Joe-Wong, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedSecurity, an end-to-end benchmark that serves as a supplementary component of the FedML library for simulating adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). FedSecurity eliminates the need for implementing the fundamental FL procedures, e.g., FL training and data loading, from scratch, thus enables users to focus on developing their own attack and defense strategies. It contains two key components, including FedAttacker that conducts a variety of attacks during FL training, and FedDefender that implements defensive mechanisms to counteract these attacks. FedSecurity has the following features: i) It offers extensive customization options to accommodate a broad range of machine learning models (e.g., Logistic Regression, ResNet, and GAN) and FL optimizers (e.g., FedAVG, FedOPT, and FedNOVA); ii) it enables exploring the effectiveness of attacks and defenses across different datasets and models; and iii) it supports flexible configuration and customization through a configuration file and some APIs. We further demonstrate FedSecurity's utility and adaptability through federated training of Large Language Models (LLMs) to showcase its potential on a wide range of complex applications.

摘要: 本文介绍了FedSecurity，这是一个端到端的基准测试，作为FedML库的补充组件，用于模拟联邦学习中的对抗性攻击和相应的防御机制。FedSecurity不需要从头开始实施基本的FL程序，例如FL训练和数据加载，从而使用户能够专注于开发他们自己的攻击和防御策略。它包含两个关键组件，包括在FL训练期间进行各种攻击的FedAttacker和实现防御机制以对抗这些攻击的FedDefender。FedSecurity具有以下功能：i)它提供广泛的定制选项，以适应广泛的机器学习模型(例如Logistic回归、ResNet和GAN)和FL优化器(例如FedAVG、FedOPT和FedNOVA)；ii)它能够跨不同的数据集和模型探索攻击和防御的有效性；iii)它通过一个配置文件和一些API支持灵活的配置和定制。通过对大型语言模型(LLM)的联合训练，我们进一步展示了FedSecurity的实用性和适应性，以展示其在广泛的复杂应用中的潜力。



## **44. AdaNCA: Neural Cellular Automata As Adaptors For More Robust Vision Transformer**

AdaNCA：神经元胞自动机作为更稳健的视觉Transformer的适配器 cs.CV

26 pages, 11 figures

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.08298v3) [paper-pdf](http://arxiv.org/pdf/2406.08298v3)

**Authors**: Yitao Xu, Tong Zhang, Sabine Süsstrunk

**Abstract**: Vision Transformers (ViTs) have demonstrated remarkable performance in image classification tasks, particularly when equipped with local information via region attention or convolutions. While such architectures improve the feature aggregation from different granularities, they often fail to contribute to the robustness of the networks. Neural Cellular Automata (NCA) enables the modeling of global cell representations through local interactions, with its training strategies and architecture design conferring strong generalization ability and robustness against noisy inputs. In this paper, we propose Adaptor Neural Cellular Automata (AdaNCA) for Vision Transformer that uses NCA as plug-in-play adaptors between ViT layers, enhancing ViT's performance and robustness against adversarial samples as well as out-of-distribution inputs. To overcome the large computational overhead of standard NCAs, we propose Dynamic Interaction for more efficient interaction learning. Furthermore, we develop an algorithm for identifying the most effective insertion points for AdaNCA based on our analysis of AdaNCA placement and robustness improvement. With less than a 3% increase in parameters, AdaNCA contributes to more than 10% absolute improvement in accuracy under adversarial attacks on the ImageNet1K benchmark. Moreover, we demonstrate with extensive evaluations across 8 robustness benchmarks and 4 ViT architectures that AdaNCA, as a plug-in-play module, consistently improves the robustness of ViTs.

摘要: 视觉变形器(VITS)在图像分类任务中表现出了显著的性能，特别是当通过区域注意或卷积来配备局部信息时。虽然这样的体系结构从不同的粒度改善了特征聚合，但它们往往无法提高网络的健壮性。神经元胞自动机(NCA)能够通过局部交互对全局细胞表示进行建模，其训练策略和结构设计具有很强的泛化能力和对噪声输入的鲁棒性。在本文中，我们提出了用于视觉转换器的适配器神经元胞自动机(AdaNCA)，它使用NCA作为VIT层之间的即插即用适配器，增强了VIT的性能和对敌意样本和分布外输入的鲁棒性。为了克服标准NCA计算开销大的缺点，我们提出了动态交互来实现更有效的交互学习。此外，基于对AdaNCA布局和健壮性改进的分析，我们提出了一种识别AdaNCA最有效插入点的算法。在参数增加不到3%的情况下，AdaNCA有助于在对ImageNet1K基准的敌意攻击下将准确率绝对提高10%以上。此外，我们通过对8个健壮性基准和4个VIT体系结构的广泛评估，证明了AdaNCA作为一个即插即用模块，持续提高了VIT的健壮性。



## **45. Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems**

遵循我的指示并说出豆子：从检索增强生成系统中提取可扩展数据 cs.CL

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2402.17840v2) [paper-pdf](http://arxiv.org/pdf/2402.17840v2)

**Authors**: Zhenting Qi, Hanlin Zhang, Eric Xing, Sham Kakade, Himabindu Lakkaraju

**Abstract**: Retrieval-Augmented Generation (RAG) improves pre-trained models by incorporating external knowledge at test time to enable customized adaptation. We study the risk of datastore leakage in Retrieval-In-Context RAG Language Models (LMs). We show that an adversary can exploit LMs' instruction-following capabilities to easily extract text data verbatim from the datastore of RAG systems built with instruction-tuned LMs via prompt injection. The vulnerability exists for a wide range of modern LMs that span Llama2, Mistral/Mixtral, Vicuna, SOLAR, WizardLM, Qwen1.5, and Platypus2, and the exploitability exacerbates as the model size scales up. Extending our study to production RAG models GPTs, we design an attack that can cause datastore leakage with a 100% success rate on 25 randomly selected customized GPTs with at most 2 queries, and we extract text data verbatim at a rate of 41% from a book of 77,000 words and 3% from a corpus of 1,569,000 words by prompting the GPTs with only 100 queries generated by themselves.

摘要: 检索-增强生成(RAG)通过在测试时纳入外部知识来改进预先训练的模型，以实现定制适应。研究了上下文检索RAG语言模型(LMS)中数据存储泄漏的风险。我们表明，攻击者可以利用LMS的指令跟随能力，通过提示注入从使用指令调整的LMS构建的RAG系统的数据存储中轻松地逐字提取文本数据。该漏洞存在于跨越Llama2、Mistral/Mixtral、Vicuna、Solar、WizardLM、Qwen1.5和Platypus2的各种现代LMS中，并且随着模型大小的增加，可利用性会加剧。将我们的研究扩展到生产RAG模型GPTS，我们设计了一个可以导致数据存储泄漏的攻击，对于随机选择的最多2个查询的25个定制GPTS，成功率为100%；在一本77,000字的书中，我们以41%的成功率逐字提取文本数据，在1,569,000字的语料库中，我们通过提示GPT只生成100个查询，以3%的速度逐字提取文本数据。



## **46. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

重新思考图表后门攻击：保留分布的角度 cs.LG

Accepted in KDD 2024

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2405.10757v2) [paper-pdf](http://arxiv.org/pdf/2405.10757v2)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.

摘要: 图形神经网络(GNN)在各种任务中表现出了显著的性能。然而，最近的研究表明，GNN很容易受到后门攻击。通常，后门攻击通过将后门触发器和目标类标签附加到训练图中的一组节点来毒化图。然后，在有毒图上训练的GNN将被误导，以预测与目标类的触发器附加的测试节点。尽管它们是有效的，但我们的实证分析表明，现有方法生成的触发因素往往是分布外(OOD)，这与干净的数据有很大不同。因此，这些注入的触发器可以很容易地被现实世界应用中广泛使用的离群点检测方法检测和修剪。因此，在本文中，我们研究了一种新的具有分布内(ID)触发器的不可察觉图后门攻击问题。为了生成ID触发器，我们引入了一个OOD检测器，并结合对抗性学习策略来生成分布内触发器的属性。为了确保ID触发器的高攻击成功率，我们引入了新的模块，通过在中毒图上训练受害者模型来增强对触发器的记忆。在真实数据集上的大量实验表明，该方法在生成分布触发器方面是有效的，可以绕过各种防御策略，同时保持较高的攻击成功率。



## **47. Multi-Robot Target Tracking with Sensing and Communication Danger Zones**

具有传感和通信危险区的多机器人目标跟踪 cs.RO

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2404.07880v2) [paper-pdf](http://arxiv.org/pdf/2404.07880v2)

**Authors**: Jiazhen Liu, Peihan Li, Yuwei Wu, Gaurav S. Sukhatme, Vijay Kumar, Lifeng Zhou

**Abstract**: Multi-robot target tracking finds extensive applications in different scenarios, such as environmental surveillance and wildfire management, which require the robustness of the practical deployment of multi-robot systems in uncertain and dangerous environments. Traditional approaches often focus on the performance of tracking accuracy with no modeling and assumption of the environments, neglecting potential environmental hazards which result in system failures in real-world deployments. To address this challenge, we investigate multi-robot target tracking in the adversarial environment considering sensing and communication attacks with uncertainty. We design specific strategies to avoid different danger zones and proposed a multi-agent tracking framework under the perilous environment. We approximate the probabilistic constraints and formulate practical optimization strategies to address computational challenges efficiently. We evaluate the performance of our proposed methods in simulations to demonstrate the ability of robots to adjust their risk-aware behaviors under different levels of environmental uncertainty and risk confidence. The proposed method is further validated via real-world robot experiments where a team of drones successfully track dynamic ground robots while being risk-aware of the sensing and/or communication danger zones.

摘要: 多机器人目标跟踪在环境监测、野火管理等不同场景中有着广泛的应用，这就要求多机器人系统在不确定和危险环境中的实际部署具有很强的鲁棒性。传统的方法往往只关注跟踪精度的性能，没有对环境进行建模和假设，而忽略了实际部署中可能导致系统故障的环境危害。为了应对这一挑战，我们研究了在具有不确定性的感知和通信攻击的对抗性环境中的多机器人目标跟踪。设计了避开不同危险区域的具体策略，提出了危险环境下的多智能体跟踪框架。我们对概率约束进行近似，并制定实用的优化策略来有效地应对计算挑战。我们在仿真中评估了我们提出的方法的性能，以展示机器人在不同的环境不确定性和风险置信度下调整其风险意识行为的能力。通过真实世界的机器人实验进一步验证了所提出的方法，其中一组无人机成功地跟踪了动态的地面机器人，同时意识到了传感和/或通信危险区域的风险。



## **48. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14393v1) [paper-pdf](http://arxiv.org/pdf/2406.14393v1)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **49. On countering adversarial perturbations in graphs using error correcting codes**

关于使用错误纠正码对抗图中的对抗扰动 cs.CR

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2406.14245v1) [paper-pdf](http://arxiv.org/pdf/2406.14245v1)

**Authors**: Saif Eddin Jabari

**Abstract**: We consider the problem of a graph subjected to adversarial perturbations, such as those arising from cyber-attacks, where edges are covertly added or removed. The adversarial perturbations occur during the transmission of the graph between a sender and a receiver. To counteract potential perturbations, we explore a repetition coding scheme with sender-assigned binary noise and majority voting on the receiver's end to rectify the graph's structure. Our approach operates without prior knowledge of the attack's characteristics. We provide an analytical derivation of a bound on the number of repetitions needed to satisfy probabilistic constraints on the quality of the reconstructed graph. We show that the method can accurately decode graphs that were subjected to non-random edge removal, namely, those connected to vertices with the highest eigenvector centrality, in addition to random addition and removal of edges by the attacker.

摘要: 我们考虑了遭受敌对扰动的图的问题，例如网络攻击引起的扰动，其中边被秘密添加或删除。对抗性扰动发生在发送者和接收者之间的图传输期间。为了抵消潜在的干扰，我们探索了一种重复编码方案，该方案具有发送者分配的二进制噪音和接收者端的多数投票，以纠正图的结构。我们的方法在不了解攻击特征的情况下运行。我们提供了满足重建图质量的概率约束所需的重复次数的界限的分析推导。我们表明，除了攻击者随机添加和删除边之外，该方法还可以准确地解码经过非随机边去除的图，即那些连接到特征向中心度最高的点的图。



## **50. Contractive Systems Improve Graph Neural Networks Against Adversarial Attacks**

收缩系统改进图神经网络对抗对抗攻击 cs.LG

**SubmitDate**: 2024-06-20    [abs](http://arxiv.org/abs/2311.06942v2) [paper-pdf](http://arxiv.org/pdf/2311.06942v2)

**Authors**: Moshe Eliasof, Davide Murari, Ferdia Sherry, Carola-Bibiane Schönlieb

**Abstract**: Graph Neural Networks (GNNs) have established themselves as a key component in addressing diverse graph-based tasks. Despite their notable successes, GNNs remain susceptible to input perturbations in the form of adversarial attacks. This paper introduces an innovative approach to fortify GNNs against adversarial perturbations through the lens of contractive dynamical systems. Our method introduces graph neural layers based on differential equations with contractive properties, which, as we show, improve the robustness of GNNs. A distinctive feature of the proposed approach is the simultaneous learned evolution of both the node features and the adjacency matrix, yielding an intrinsic enhancement of model robustness to perturbations in the input features and the connectivity of the graph. We mathematically derive the underpinnings of our novel architecture and provide theoretical insights to reason about its expected behavior. We demonstrate the efficacy of our method through numerous real-world benchmarks, reading on par or improved performance compared to existing methods.

摘要: 图形神经网络(GNN)已经成为解决各种基于图形的任务的关键组件。尽管GNN取得了显著的成功，但它们仍然容易受到对抗性攻击形式的投入扰动的影响。本文介绍了一种通过压缩动力系统的透镜来增强GNN抵抗敌意扰动的创新方法。我们的方法引入了基于具有压缩性质的微分方程的图神经层，从而提高了GNN的稳健性。该方法的一个显著特点是节点特征和邻接矩阵的同时学习进化，从而内在地增强了模型对输入特征扰动和图的连通性的稳健性。我们从数学上推导出我们的新体系结构的基础，并提供理论见解来推理其预期行为。我们通过许多真实世界的基准测试来证明我们的方法的有效性，与现有的方法相比，我们的阅读是平分的，或者是性能有所提高。



