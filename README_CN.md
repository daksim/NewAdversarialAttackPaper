# Latest Adversarial Attack Papers
**update at 2023-05-27 16:12:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Robustness of Segment Anything**

关于Segment Anything的健壮性 cs.CV

22 pages

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16220v1) [paper-pdf](http://arxiv.org/pdf/2305.16220v1)

**Authors**: Yihao Huang, Yue Cao, Tianlin Li, Felix Juefei-Xu, Di Lin, Ivor W. Tsang, Yang Liu, Qing Guo

**Abstract**: Segment anything model (SAM) has presented impressive objectness identification capability with the idea of prompt learning and a new collected large-scale dataset. Given a prompt (e.g., points, bounding boxes, or masks) and an input image, SAM is able to generate valid segment masks for all objects indicated by the prompts, presenting high generalization across diverse scenarios and being a general method for zero-shot transfer to downstream vision tasks. Nevertheless, it remains unclear whether SAM may introduce errors in certain threatening scenarios. Clarifying this is of significant importance for applications that require robustness, such as autonomous vehicles. In this paper, we aim to study the testing-time robustness of SAM under adversarial scenarios and common corruptions. To this end, we first build a testing-time robustness evaluation benchmark for SAM by integrating existing public datasets. Second, we extend representative adversarial attacks against SAM and study the influence of different prompts on robustness. Third, we study the robustness of SAM under diverse corruption types by evaluating SAM on corrupted datasets with different prompts. With experiments conducted on SA-1B and KITTI datasets, we find that SAM exhibits remarkable robustness against various corruptions, except for blur-related corruption. Furthermore, SAM remains susceptible to adversarial attacks, particularly when subjected to PGD and BIM attacks. We think such a comprehensive study could highlight the importance of the robustness issues of SAM and trigger a series of new tasks for SAM as well as downstream vision tasks.

摘要: 分段任意模型(SAM)以快速学习的思想和新收集的大规模数据集显示了令人印象深刻的客观性识别能力。在给定提示(例如，点、边界框或遮罩)和输入图像的情况下，SAM能够为提示所指示的所有对象生成有效的分段遮罩，呈现跨不同场景的高度概括性，并且是向下游视觉任务进行零镜头转移的通用方法。然而，目前尚不清楚SAM是否会在某些威胁场景中引入错误。澄清这一点对于需要健壮性的应用(如自动驾驶汽车)具有重要意义。在本文中，我们旨在研究SAM在对抗场景和常见腐败下的测试时间稳健性。为此，我们首先通过整合现有的公共数据集，为SAM构建了一个测试时健壮性评估基准。其次，扩展了针对SAM的典型对抗性攻击，并研究了不同提示对健壮性的影响。第三，通过对具有不同提示的受损数据集上的SAM进行评估，研究了SAM在不同破坏类型下的健壮性。通过在SA-1B和KITTI数据集上进行的实验，我们发现SAM对除模糊相关的腐败之外的各种腐败表现出了显著的健壮性。此外，SAM仍然容易受到对抗性攻击，特别是在受到PGD和BIM攻击时。我们认为，这样的综合研究可以突出SAM稳健性问题的重要性，并引发SAM以及下游视觉任务的一系列新任务。



## **2. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

ByzSecAgg：一种基于编码计算和向量承诺的联合学习抗拜占庭安全聚合方案 cs.CR

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2302.09913v2) [paper-pdf](http://arxiv.org/pdf/2302.09913v2)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, the proposed scheme also uses a vector commitment method, in which the commitment size remains constant (i.e. does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process.

摘要: 在本文中，我们提出了一种有效的联合学习安全聚合方案，该方案可以防止拜占庭攻击和隐私泄露。处理个人更新以管理敌对行为，同时针对串通节点保护数据隐私，需要某种类型的安全秘密共享。然而，更新的长矢量的秘密共享的通信负荷可能非常高。为了解决这一问题，在所提出的方案中，局部更新被划分为更小的子向量，并使用斜坡秘密共享来共享。然而，这种共享方法不允许离群点检测算法所需的双线性计算，例如成对距离计算。为了解决这个问题，每个用户运行另一轮坡道共享，在共享多项式中嵌入不同的数据。这项技术受编码计算思想的启发，实现了两两距离的安全计算。此外，为了保持本地更新的完整性和私密性，该方案还使用了向量承诺方法，承诺大小保持不变(即不随本地更新的长度增加)，同时允许对秘密共享过程进行验证。



## **3. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

对抗性训练对语言模型稳健性和泛化能力的影响 cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2211.05523v2) [paper-pdf](http://arxiv.org/pdf/2211.05523v2)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of transformer-based language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveals that the improved generalization is due to 'more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.

摘要: 对抗性训练被广泛认为是对抗对抗性攻击的最有效的防御方法。然而，众所周知，在对抗性训练的模型中实现稳健性和泛化都需要权衡。这项工作的目标是深入比较语言模型中对抗性训练的不同方法。具体地说，我们研究了训练前数据扩充以及训练时间输入扰动与嵌入空间扰动对基于变压器的语言模型的稳健性和泛化的影响。我们的发现表明，通过预训练数据增强或通过输入空间扰动训练可以获得更好的稳健性。然而，嵌入空间扰动的训练显著提高了泛化能力。对学习模型的神经元进行的语言相关性分析表明，改进的泛化是由于更专业的神经元。据我们所知，这是第一次对语言模型对抗性训练中生成对抗性实例的不同方法进行深入的定性分析。



## **4. IDEA: Invariant Causal Defense for Graph Adversarial Robustness**

IDEA：图对抗健壮性的不变因果防御 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15792v1) [paper-pdf](http://arxiv.org/pdf/2305.15792v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Bingbing Xu, Xueqi Cheng

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various tasks, however, their vulnerability to adversarial attacks raises concerns for the real-world applications. Existing defense methods can resist some attacks, but suffer unbearable performance degradation under other unknown attacks. This is due to their reliance on either limited observed adversarial examples to optimize (adversarial training) or specific heuristics to alter graph or model structures (graph purification or robust aggregation). In this paper, we propose an Invariant causal DEfense method against adversarial Attacks (IDEA), providing a new perspective to address this issue. The method aims to learn causal features that possess strong predictability for labels and invariant predictability across attacks, to achieve graph adversarial robustness. Through modeling and analyzing the causal relationships in graph adversarial attacks, we design two invariance objectives to learn the causal features. Extensive experiments demonstrate that our IDEA significantly outperforms all the baselines under both poisoning and evasion attacks on five benchmark datasets, highlighting the strong and invariant predictability of IDEA. The implementation of IDEA is available at https://anonymous.4open.science/r/IDEA_repo-666B.

摘要: 图神经网络(GNN)在各种任务中取得了显著的成功，但其对敌意攻击的脆弱性引起了人们对现实世界应用的担忧。现有的防御方法可以抵抗一些攻击，但在其他未知攻击下，性能会出现无法承受的下降。这是因为他们依赖于有限的观察到的对抗性例子来优化(对抗性训练)，或者依赖特定的启发式方法来改变图形或模型结构(图形净化或健壮聚合)。本文提出了一种对抗攻击的不变因果防御方法(IDEA)，为解决这一问题提供了一个新的视角。该方法旨在学习对标签具有很强可预测性和对攻击具有不变可预测性的因果特征，以实现图对抗的健壮性。通过对图对抗攻击中因果关系的建模和分析，设计了两个不变目标来学习因果关系的特征。大量实验表明，在五个基准数据集上，无论是中毒攻击还是逃避攻击，我们的算法都显著优于所有的基线算法，突出了IDEA算法强大且不变的可预测性。IDEA的实施可在https://anonymous.4open.science/r/IDEA_repo-666B.上获得



## **5. Healing Unsafe Dialogue Responses with Weak Supervision Signals**

用微弱的监督信号修复不安全的对话反应 cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15757v1) [paper-pdf](http://arxiv.org/pdf/2305.15757v1)

**Authors**: Zi Liang, Pinghui Wang, Ruofei Zhang, Shuo Zhang, Xiaofan Ye Yi Huang, Junlan Feng

**Abstract**: Recent years have seen increasing concerns about the unsafe response generation of large-scale dialogue systems, where agents will learn offensive or biased behaviors from the real-world corpus. Some methods are proposed to address the above issue by detecting and replacing unsafe training examples in a pipeline style. Though effective, they suffer from a high annotation cost and adapt poorly to unseen scenarios as well as adversarial attacks. Besides, the neglect of providing safe responses (e.g. simply replacing with templates) will cause the information-missing problem of dialogues. To address these issues, we propose an unsupervised pseudo-label sampling method, TEMP, that can automatically assign potential safe responses. Specifically, our TEMP method groups responses into several clusters and samples multiple labels with an adaptively sharpened sampling strategy, inspired by the observation that unsafe samples in the clusters are usually few and distribute in the tail. Extensive experiments in chitchat and task-oriented dialogues show that our TEMP outperforms state-of-the-art models with weak supervision signals and obtains comparable results under unsupervised learning settings.

摘要: 近年来，人们越来越担心大规模对话系统的不安全响应生成，在这种系统中，代理将从现实世界的语料库中学习攻击性或偏见行为。提出了通过检测和替换流水线形式的不安全训练实例来解决上述问题的一些方法。虽然它们很有效，但它们存在着较高的注释成本，并且对未知场景以及对抗性攻击的适应性很差。此外，忽视提供安全的响应(例如，简单地用模板替换)将导致对话的信息缺失问题。为了解决这些问题，我们提出了一种无监督的伪标签抽样方法TEMP，该方法可以自动分配潜在的安全响应。具体地说，我们的TEMP方法将响应分组到几个簇中，并使用自适应锐化的采样策略对多个标签进行采样，灵感来自于观察到簇中不安全的样本通常很少，并且分布在尾部。在聊天和面向任务的对话中的大量实验表明，我们的TEMP优于具有弱监督信号的最新模型，并且在无监督学习环境下获得了类似的结果。



## **6. PEARL: Preprocessing Enhanced Adversarial Robust Learning of Image Deraining for Semantic Segmentation**

PEAR：用于语义分割的图像降维的预处理增强的对抗性稳健学习 cs.CV

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15709v1) [paper-pdf](http://arxiv.org/pdf/2305.15709v1)

**Authors**: Xianghao Jiao, Yaohua Liu, Jiaxin Gao, Xinyuan Chu, Risheng Liu, Xin Fan

**Abstract**: In light of the significant progress made in the development and application of semantic segmentation tasks, there has been increasing attention towards improving the robustness of segmentation models against natural degradation factors (e.g., rain streaks) or artificially attack factors (e.g., adversarial attack). Whereas, most existing methods are designed to address a single degradation factor and are tailored to specific application scenarios. In this work, we present the first attempt to improve the robustness of semantic segmentation tasks by simultaneously handling different types of degradation factors. Specifically, we introduce the Preprocessing Enhanced Adversarial Robust Learning (PEARL) framework based on the analysis of our proposed Naive Adversarial Training (NAT) framework. Our approach effectively handles both rain streaks and adversarial perturbation by transferring the robustness of the segmentation model to the image derain model. Furthermore, as opposed to the commonly used Negative Adversarial Attack (NAA), we design the Auxiliary Mirror Attack (AMA) to introduce positive information prior to the training of the PEARL framework, which improves defense capability and segmentation performance. Our extensive experiments and ablation studies based on different derain methods and segmentation models have demonstrated the significant performance improvement of PEARL with AMA in defense against various adversarial attacks and rain streaks while maintaining high generalization performance across different datasets.

摘要: 鉴于在语义分割任务的开发和应用方面取得的重大进展，人们越来越关注提高分割模型对自然退化因素(例如，雨带)或人为攻击因素(例如，对抗性攻击)的稳健性。然而，大多数现有的方法都是为解决单一退化因素而设计的，并且是针对特定的应用场景量身定做的。在这项工作中，我们首次尝试通过同时处理不同类型的退化因素来提高语义分割任务的稳健性。具体地说，我们在分析了我们提出的朴素对抗性训练(NAT)框架的基础上，引入了预处理增强的对抗性稳健学习(PEAR)框架。通过将分割模型的稳健性转移到图像的DERAIN模型，我们的方法有效地处理了雨滴和对抗性扰动。此外，与常用的消极对抗攻击(NAA)不同，我们设计了辅助镜像攻击(AMA)，在PEARL框架的训练之前引入积极信息，从而提高了防御能力和分割性能。我们基于不同DERAIN方法和分割模型的大量实验和烧蚀研究表明，在保持对不同数据集的高泛化性能的同时，使用AMA的PEARE算法在防御各种对手攻击和雨带方面的性能有了显著的提高。



## **7. Rethink Diversity in Deep Learning Testing**

深度学习测试中的多样性再思考 cs.SE

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15698v1) [paper-pdf](http://arxiv.org/pdf/2305.15698v1)

**Authors**: Zi Wang, Jihye Choi, Somesh Jha

**Abstract**: Deep neural networks (DNNs) have demonstrated extraordinary capabilities and are an integral part of modern software systems. However, they also suffer from various vulnerabilities such as adversarial attacks and unfairness. Testing deep learning (DL) systems is therefore an important task, to detect and mitigate those vulnerabilities. Motivated by the success of traditional software testing, which often employs diversity heuristics, various diversity measures on DNNs have been proposed to help efficiently expose the buggy behavior of DNNs. In this work, we argue that many DNN testing tasks should be treated as directed testing problems rather than general-purpose testing tasks, because these tasks are specific and well-defined. Hence, the diversity-based approach is less effective.   Following our argument based on the semantics of DNNs and the testing goal, we derive $6$ metrics that can be used for DNN testing and carefully analyze their application scopes. We empirically show their efficacy in exposing bugs in DNNs compared to recent diversity-based metrics. Moreover, we also notice discrepancies between the practices of the software engineering (SE) community and the DL community. We point out some of these gaps, and hopefully, this can lead to bridging the SE practice and DL findings.

摘要: 深度神经网络(DNN)已经显示出非凡的能力，是现代软件系统不可或缺的一部分。然而，它们也存在各种脆弱性，如对抗性攻击和不公平。因此，测试深度学习(DL)系统是检测和缓解这些漏洞的一项重要任务。传统的软件测试通常采用多样性启发式方法，受此启发，人们提出了各种针对DNN的多样性措施，以帮助有效地暴露DNN的错误行为。在这项工作中，我们认为许多DNN测试任务应该被视为有指导的测试问题，而不是通用的测试任务，因为这些任务是特定的和定义良好的。因此，基于多样性的方法不太有效。根据DNN的语义和测试目标，我们推导出可用于DNN测试的$6$度量，并仔细分析了它们的应用范围。与最近的基于多样性的度量相比，我们经验地展示了它们在暴露DNN中的错误方面的有效性。此外，我们还注意到软件工程(SE)社区和DL社区的实践之间的差异。我们指出了其中的一些差距，希望这能导致SE实践和DL发现之间的桥梁。



## **8. AdvFunMatch: When Consistent Teaching Meets Adversarial Robustness**

AdvFunMatch：当一致的教学遇到对手的健壮性 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.14700v2) [paper-pdf](http://arxiv.org/pdf/2305.14700v2)

**Authors**: Zihui Wu, Haichang Gao, Bingqian Zhou, Ping Wang

**Abstract**: \emph{Consistent teaching} is an effective paradigm for implementing knowledge distillation (KD), where both student and teacher models receive identical inputs, and KD is treated as a function matching task (FunMatch). However, one limitation of FunMatch is that it does not account for the transfer of adversarial robustness, a model's resistance to adversarial attacks. To tackle this problem, we propose a simple but effective strategy called Adversarial Function Matching (AdvFunMatch), which aims to match distributions for all data points within the $\ell_p$-norm ball of the training data, in accordance with consistent teaching. Formulated as a min-max optimization problem, AdvFunMatch identifies the worst-case instances that maximizes the KL-divergence between teacher and student model outputs, which we refer to as "mismatched examples," and then matches the outputs on these mismatched examples. Our experimental results show that AdvFunMatch effectively produces student models with both high clean accuracy and robustness. Furthermore, we reveal that strong data augmentations (\emph{e.g.}, AutoAugment) are beneficial in AdvFunMatch, whereas prior works have found them less effective in adversarial training. Code is available at \url{https://gitee.com/zihui998/adv-fun-match}.

摘要: 一致性教学是实现知识提炼的一种有效范式，其中学生模型和教师模型接受相同的输入，而一致性教学被视为一项功能匹配任务。然而，FunMatch的一个局限性是它没有考虑到对抗健壮性的转移，即模型对对抗攻击的抵抗力。为了解决这一问题，我们提出了一种简单而有效的策略，称为对抗函数匹配(AdvFunMatch)，该策略旨在根据一致的教学匹配训练数据的$\ell_p$-范数球内所有数据点的分布。AdvFunMatch被描述为一个最小-最大优化问题，它识别最大化教师和学生模型输出之间的KL-分歧的最坏情况实例，我们将其称为“不匹配示例”，然后将输出与这些不匹配示例进行匹配。我们的实验结果表明，AdvFunMatch有效地生成了具有较高清洁准确率和鲁棒性的学生模型。此外，我们发现强数据扩充(例如，AutoAugment)在AdvFunMatch中是有益的，而先前的研究发现它们在对抗性训练中效果较差。代码可在\url{https://gitee.com/zihui998/adv-fun-match}.上找到



## **9. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2008.09312v4) [paper-pdf](http://arxiv.org/pdf/2008.09312v4)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.

摘要: 我研究了一个随机多臂强盗问题，其中报酬受到对抗性腐败的影响。提出了一种新的攻击策略，该策略利用UCB算法操纵学习者拉出一些非最优目标臂$T-o(T)$次，累积代价为$\widehat{O}(\Sqrt{\log T})$，其中$T$是轮数。我还证明了累积攻击成本的第一个下限。下界与最高可达$O(\LOG\LOG T)$因子的上界匹配，表明所提出的攻击策略接近最优。



## **10. How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks**

人类是如何感知敌意文本的？基于词语的对抗性攻击有效性和自然性的真实性检验 cs.CL

ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15587v1) [paper-pdf](http://arxiv.org/pdf/2305.15587v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy

**Abstract**: Natural Language Processing (NLP) models based on Machine Learning (ML) are susceptible to adversarial attacks -- malicious algorithms that imperceptibly modify input text to force models into making incorrect predictions. However, evaluations of these attacks ignore the property of imperceptibility or study it under limited settings. This entails that adversarial perturbations would not pass any human quality gate and do not represent real threats to human-checked NLP systems. To bypass this limitation and enable proper assessment (and later, improvement) of NLP model robustness, we have surveyed 378 human participants about the perceptibility of text adversarial examples produced by state-of-the-art methods. Our results underline that existing text attacks are impractical in real-world scenarios where humans are involved. This contrasts with previous smaller-scale human studies, which reported overly optimistic conclusions regarding attack success. Through our work, we hope to position human perceptibility as a first-class success criterion for text attacks, and provide guidance for research to build effective attack algorithms and, in turn, design appropriate defence mechanisms.

摘要: 基于机器学习(ML)的自然语言处理(NLP)模型容易受到敌意攻击--恶意算法潜移默化地修改输入文本，迫使模型做出错误的预测。然而，对这些攻击的评估忽略了不可感知性的属性，或者在有限的设置下研究它。这意味着对抗性扰动不会通过任何人类素质的关口，也不会对人类检查的NLP系统构成真正的威胁。为了绕过这一限制，并使适当的评估(以及后来的改进)自然语言处理模型的稳健性，我们调查了378名人类参与者关于由最先进的方法产生的文本对抗性例子的感知能力。我们的结果强调，现有的文本攻击在涉及人类的真实世界场景中是不切实际的。这与之前规模较小的人体研究形成了鲜明对比，后者报告了关于攻击成功的过于乐观的结论。通过我们的工作，我们希望将人类感知能力定位为文本攻击的一流成功标准，并为构建有效的攻击算法，进而设计适当的防御机制的研究提供指导。



## **11. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

训练数据重构的非渐近下界 cs.LG

Additional experiments and minor bug fixes

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2303.16372v4) [paper-pdf](http://arxiv.org/pdf/2303.16372v4)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: Mathematical notions of privacy, such as differential privacy, are often stated as probabilistic guarantees that are difficult to interpret. It is imperative, however, that the implications of data sharing be effectively communicated to the data principal to ensure informed decision-making and offer full transparency with regards to the associated privacy risks. To this end, our work presents a rigorous quantitative evaluation of the protection conferred by private learners by investigating their resilience to training data reconstruction attacks. We accomplish this by deriving non-asymptotic lower bounds on the reconstruction error incurred by any adversary against $(\epsilon, \delta)$ differentially private learners for target samples that belong to any compact metric space. Working with a generalization of differential privacy, termed metric privacy, we remove boundedness assumptions on the input space prevalent in prior work, and prove that our results hold for general locally compact metric spaces. We extend the analysis to cover the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget, and demonstrate that our bounds are minimax optimal under certain regimes.

摘要: 隐私的数学概念，如差异隐私，通常被声明为难以解释的概率保证。然而，必须将数据共享的影响有效地传达给数据负责人，以确保做出明智的决策，并就相关的隐私风险提供充分的透明度。为此，我们的工作通过调查私人学习者对训练数据重建攻击的弹性来对他们提供的保护进行严格的定量评估。我们通过对属于任何紧致度量空间的目标样本的任何对手对$(\epsilon，\Delta)$差分私人学习者所引起的重构误差的非渐近下界来实现这一点。利用差分度量隐私的推广，我们去掉了以前工作中普遍存在的输入空间的有界性假设，并证明了我们的结果对一般的局部紧度量空间成立。我们将分析扩展到高维体制，其中输入数据的维度可能大于对手的查询预算，并证明了在某些体制下我们的界是极小极大最优的。



## **12. Fast Adversarial CNN-based Perturbation Attack on No-Reference Image- and Video-Quality Metrics**

基于CNN的无参考图像和视频质量指标的快速对抗性扰动攻击 cs.CV

ICLR 2023 TinyPapers

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15544v1) [paper-pdf](http://arxiv.org/pdf/2305.15544v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Modern neural-network-based no-reference image- and video-quality metrics exhibit performance as high as full-reference metrics. These metrics are widely used to improve visual quality in computer vision methods and compare video processing methods. However, these metrics are not stable to traditional adversarial attacks, which can cause incorrect results. Our goal is to investigate the boundaries of no-reference metrics applicability, and in this paper, we propose a fast adversarial perturbation attack on no-reference quality metrics. The proposed attack (FACPA) can be exploited as a preprocessing step in real-time video processing and compression algorithms. This research can yield insights to further aid in designing of stable neural-network-based no-reference quality metrics.

摘要: 现代基于神经网络的无参考图像和视频质量指标表现出与全参考指标一样高的性能。这些度量被广泛用于改善计算机视觉方法中的视觉质量和比较视频处理方法。然而，这些指标对传统的对抗性攻击并不稳定，这可能会导致错误的结果。我们的目标是研究无参考度量的适用范围，在本文中，我们提出了一种针对无参考质量度量的快速对抗性扰动攻击。所提出的攻击(FACPA)可以作为实时视频处理和压缩算法的预处理步骤。这项研究可以为设计稳定的基于神经网络的无参考质量度量提供进一步的帮助。



## **13. Robust Classification via a Single Diffusion Model**

基于单扩散模型的稳健分类 cs.CV

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15241v1) [paper-pdf](http://arxiv.org/pdf/2305.15241v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Recently, diffusion models have been successfully applied to improving adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, the diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, in this paper we propose Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. Our method first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood of the diffusion model through Bayes' theorem. Since our method does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $73.24\%$ robust accuracy against $\ell_\infty$ norm-bounded perturbations with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+2.34\%$. The findings highlight the potential of generative classifiers by employing diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers.

摘要: 近年来，扩散模型已被成功地应用于提高图像分类器的对抗性鲁棒性，方法是净化对抗性噪声或生成用于对抗性训练的真实数据。然而，基于扩散的净化方法可以通过更强的自适应攻击来规避，而对抗性训练在看不见的威胁下表现不佳，显示出这些方法不可避免的局限性。为了更好地利用扩散模型的表达能力，本文提出了稳健扩散分类器(RDC)，这是一种由预先训练的扩散模型构造的反之稳健的生成式分类器。我们的方法首先最大化给定输入的数据似然，然后通过贝叶斯定理利用扩散模型的条件似然来预测优化输入的类别概率。由于我们的方法不需要在特定的对抗性攻击上进行培训，因此我们证明了它更具一般性，可以防御多个看不见的威胁。特别是，RDC在CIFAR-10上对$epsilon_INFTY=8/255$的范数有界摄动获得了$73.24$的稳健精度，比以前最先进的对抗性训练模型高出$+2.34$。这些发现突出了生成性分类器的潜力，与通常研究的判别性分类器相比，它使用扩散模型来实现对抗稳健性。



## **14. Adaptive Data Analysis in a Balanced Adversarial Model**

均衡对抗性模型中的自适应数据分析 cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15452v1) [paper-pdf](http://arxiv.org/pdf/2305.15452v1)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but does not have a prior knowledge of the underlying distribution. We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.

摘要: 在适应性数据分析中，一个机制得到$n$I.I.D.来自未知分布$D$的样本，并且需要对关于$D$的适应性选择的统计查询序列提供准确的估计。Hardt和Ullman(FOCS 2014)和Steinke和Ullman(COLT 2015)表明，假设存在单向函数，通常很难回答超过$\theta(n^2)$自适应查询。然而，这些负面结果强烈依赖于对抗性模型，该模型显著地使对抗性分析师相对于该机制具有优势，因为选择自适应查询的分析师也选择基础分布$D$。这种不平衡对所获得的硬度结果的适用性提出了问题--完全了解基本分布$D$的分析员将几乎不需要向一个仅保存来自$D$的有限数量样本的机制发出统计查询。我们考虑更受限制的对手，称为\emph{平衡}，其中每个这样的对手由两个独立的算法组成：\emph{Sampler}是选择分布并将样本提供给机制的实体，以及\emph{Analyst}选择自适应查询，但不事先知道底层分布。在标准的公钥密码学假设下，我们通过使用一个有效的、平衡的对手来重新访问以前的下界，从而提高了它们的质量。我们证明了这些更强的难度假设是不可避免的，因为任何具有所有已知攻击结构的计算有界的对手都意味着公钥密码学的存在。



## **15. Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension**

内隐偏见与对抗性攻击的内在维度关联 cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15203v1) [paper-pdf](http://arxiv.org/pdf/2305.15203v1)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementioned frequencies. By exploiting the entanglement between intrinsic dimension and correlation, we provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are closely tied.

摘要: 尽管神经网络在分类方面的表现令人印象深刻，但众所周知，它很容易受到对手的攻击。这些攻击是对输入数据的微小干扰，旨在愚弄模型。自然，模型的体系结构、设置或属性与攻击性质之间的潜在联系就会出现问题。在这项工作中，我们的目标是通过关注神经网络的隐含偏差来阐明这个问题，隐含偏差指的是它固有的偏爱特定模式或结果的倾向。具体地说，我们研究了隐式偏差的一个方面，它涉及准确图像分类所需的基本傅立叶频率。我们进行测试，以评估这些频率与成功攻击所必需的频率之间的统计关系。为了深入研究这种关系，我们提出了一种新的方法，可以发现坐标集合之间的非线性关联，在我们的例子中，这些集合就是前面提到的频率。通过利用内在维度和相关性之间的纠缠，我们提供了经验证据，证明了傅立叶空间中的网络偏差与对抗性攻击的目标频率密切相关。



## **16. IoT Threat Detection Testbed Using Generative Adversarial Networks**

基于产生式对抗网络的物联网威胁检测测试平台 cs.CR

8 pages, 5 figures

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15191v1) [paper-pdf](http://arxiv.org/pdf/2305.15191v1)

**Authors**: Farooq Shaikh, Elias Bou-Harb, Aldin Vehabovic, Jorge Crichigno, Aysegul Yayimli, Nasir Ghani

**Abstract**: The Internet of Things(IoT) paradigm provides persistent sensing and data collection capabilities and is becoming increasingly prevalent across many market sectors. However, most IoT devices emphasize usability and function over security, making them very vulnerable to malicious exploits. This concern is evidenced by the increased use of compromised IoT devices in large scale bot networks (botnets) to launch distributed denial of service(DDoS) attacks against high value targets. Unsecured IoT systems can also provide entry points to private networks, allowing adversaries relatively easy access to valuable resources and services. Indeed, these evolving IoT threat vectors (ranging from brute force attacks to remote code execution exploits) are posing key challenges. Moreover, many traditional security mechanisms are not amenable for deployment on smaller resource-constrained IoT platforms. As a result, researchers have been developing a range of methods for IoT security, with many strategies using advanced machine learning(ML) techniques. Along these lines, this paper presents a novel generative adversarial network(GAN) solution to detect threats from malicious IoT devices both inside and outside a network. This model is trained using both benign IoT traffic and global darknet data and further evaluated in a testbed with real IoT devices and malware threats.

摘要: 物联网(IoT)模式提供持久的传感和数据收集能力，并在许多市场领域变得越来越普遍。然而，大多数物联网设备强调可用性和功能，而不是安全性，这使得它们非常容易受到恶意攻击。大规模僵尸网络(僵尸网络)中越来越多地使用受攻击的物联网设备来对高价值目标发动分布式拒绝服务(DDoS)攻击，这证明了这一担忧。不安全的物联网系统还可以提供专用网络的入口点，使对手能够相对轻松地访问有价值的资源和服务。事实上，这些不断演变的物联网威胁向量(从暴力攻击到远程代码执行漏洞)构成了关键挑战。此外，许多传统的安全机制不适合在较小的资源受限的物联网平台上部署。因此，研究人员一直在开发一系列物联网安全方法，其中许多策略使用高级机器学习(ML)技术。为此，本文提出了一种新的生成性对抗网络(GAN)解决方案，用于检测来自网络内外恶意物联网设备的威胁。该模型使用良性物联网流量和全球暗网数据进行训练，并在试验台上使用真实物联网设备和恶意软件威胁进行进一步评估。



## **17. Another Dead End for Morphological Tags? Perturbed Inputs and Parsing**

形态标签的又一个死胡同？受干扰的输入和解析 cs.CL

Accepted at Findings of ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15119v1) [paper-pdf](http://arxiv.org/pdf/2305.15119v1)

**Authors**: Alberto Muñoz-Ortiz, David Vilares

**Abstract**: The usefulness of part-of-speech tags for parsing has been heavily questioned due to the success of word-contextualized parsers. Yet, most studies are limited to coarse-grained tags and high quality written content; while we know little about their influence when it comes to models in production that face lexical errors. We expand these setups and design an adversarial attack to verify if the use of morphological information by parsers: (i) contributes to error propagation or (ii) if on the other hand it can play a role to correct mistakes that word-only neural parsers make. The results on 14 diverse UD treebanks show that under such attacks, for transition- and graph-based models their use contributes to degrade the performance even faster, while for the (lower-performing) sequence labeling parsers they are helpful. We also show that if morphological tags were utopically robust against lexical perturbations, they would be able to correct parsing mistakes.

摘要: 由于单词上下文解析器的成功，词性标签对语法分析的有用性受到了严重质疑。然而，大多数研究仅限于粗粒度标签和高质量的书面内容；而当涉及到生产中面临词汇错误的模型时，我们对它们的影响知之甚少。我们扩展了这些设置，并设计了一个对抗性攻击来验证解析器对形态信息的使用：(I)有助于错误传播，或者(Ii)另一方面，它可以起到纠正只使用单词的神经解析器所犯错误的作用。在14个不同的UD树库上的结果表明，在这种攻击下，对于基于转换和基于图的模型，它们的使用有助于更快地降低性能，而对于(性能较差的)序列标记解析器，它们是有帮助的。我们还表明，如果形态标签对词汇扰动具有超乎寻常的健壮性，它们将能够纠正语法分析错误。



## **18. Adversarial Demonstration Attacks on Large Language Models**

针对大型语言模型的对抗性演示攻击 cs.CL

Work in Progress

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14950v1) [paper-pdf](http://arxiv.org/pdf/2305.14950v1)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose an ICL attack based on TextAttack, which aims to only manipulate the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decreases. Furthermore, we also observe that adversarially attacked demonstrations exhibit transferability to diverse input examples. These findings emphasize the critical security risks associated with ICL and underscore the necessity for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.

摘要: 随着更强大的大型语言模型(LLM)的出现，如ChatGPT和GPT-4，情境学习(ICL)通过将数据-标签对作为前提提示来利用这些模型来执行特定任务，从而获得了显著的突出地位。虽然合并演示可以极大地提高LLMS在各种任务中的性能，但它可能会引入一个新的安全问题：攻击者只能操作演示，而不会更改输入来执行攻击。在本文中，我们从对抗的角度研究了ICL的安全问题，重点关注了示威活动的影响。我们提出了一种基于TextAttack的ICL攻击，其目的是只操纵演示，而不改变输入来误导模型。我们的结果表明，随着演示数量的增加，情境学习的稳健性会降低。此外，我们还观察到，被敌意攻击的演示表现出对不同输入示例的迁移。这些研究结果强调了与大规模杀伤性武器相关的重大安全风险，并强调了对大规模杀伤性武器的稳健性进行广泛研究的必要性，特别是考虑到其在推进LLMS方面日益重要。



## **19. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

MAdvex：机器学习恶意软件检测中基于工具的敌意攻击 cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.02559v2) [paper-pdf](http://arxiv.org/pdf/2305.02559v2)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.

摘要: WebAssembly(WASM)是一种用于Web应用程序的低级二进制格式，由于其改进的性能和与现有软件的兼容性而被广泛采用。然而，WASM的流行也导致了对其进行恶意攻击，例如加密劫持，即恶意行为者在未经受害者同意的情况下使用受害者的计算资源来挖掘加密货币。为了应对这种威胁，出现了基于机器学习的检测方法，旨在识别WASM代码中的加密劫持活动。众所周知，神经网络容易受到敌意攻击，在这种攻击中，分类器的输入会受到干扰，只需进行极小的更改，就会导致粗略的错误分类。虽然在图像分类中应用更改很容易，但在不更改功能的情况下以自动方式操作二进制文件来规避恶意软件分类并不是一件容易的事情。在这项工作中，我们提出了一种新的方法，通过插装将对抗性示例包括在二进制文件的代码部分中。引入的小工具允许包含任意字节，从而实现了高效的对抗性攻击，可靠地绕过了最先进的机器学习分类器，例如最近在NDSS 2021上提出的基于CNN的Minos。我们分析了基于插桩的对抗性实例生成的代价和可靠性，结果表明该方法在最小规模和最小性能开销的情况下能够可靠地工作。



## **20. Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup**

引入竞争，通过干净的特征混合提高目标对抗性例子的可转移性 cs.CV

CVPR 2023 camera-ready

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14846v1) [paper-pdf](http://arxiv.org/pdf/2305.14846v1)

**Authors**: Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, Changick Kim

**Abstract**: Deep neural networks are widely known to be susceptible to adversarial examples, which can cause incorrect predictions through subtle input modifications. These adversarial examples tend to be transferable between models, but targeted attacks still have lower attack success rates due to significant variations in decision boundaries. To enhance the transferability of targeted adversarial examples, we propose introducing competition into the optimization process. Our idea is to craft adversarial perturbations in the presence of two new types of competitor noises: adversarial perturbations towards different target classes and friendly perturbations towards the correct class. With these competitors, even if an adversarial example deceives a network to extract specific features leading to the target class, this disturbance can be suppressed by other competitors. Therefore, within this competition, adversarial examples should take different attack strategies by leveraging more diverse features to overwhelm their interference, leading to improving their transferability to different models. Considering the computational complexity, we efficiently simulate various interference from these two types of competitors in feature space by randomly mixing up stored clean features in the model inference and named this method Clean Feature Mixup (CFM). Our extensive experimental results on the ImageNet-Compatible and CIFAR-10 datasets show that the proposed method outperforms the existing baselines with a clear margin. Our code is available at https://github.com/dreamflake/CFM.

摘要: 众所周知，深度神经网络容易受到对抗性例子的影响，这可能会通过微妙的输入修改导致错误的预测。这些对抗性的例子往往可以在模型之间转换，但由于决策边界的显著差异，定向攻击的攻击成功率仍然较低。为了提高目标对抗性实例的可转移性，我们建议在优化过程中引入竞争。我们的想法是在两种新的竞争对手噪声存在的情况下制造对抗性扰动：针对不同目标类别的对抗性扰动和针对正确类别的友好扰动。对于这些竞争对手，即使敌对的例子欺骗网络以提取导致目标类的特定特征，这种干扰也可以被其他竞争对手抑制。因此，在这场比赛中，对抗性榜样应该采取不同的攻击策略，利用更多样化的特征来压倒他们的干扰，从而提高他们对不同模型的可转移性。考虑到计算的复杂性，我们通过在模型推理中随机混合存储的干净特征来有效地模拟这两种竞争对手在特征空间中的各种干扰，将该方法命名为清洁特征混合(CFM)方法。我们在ImageNet兼容数据集和CIFAR-10数据集上的大量实验结果表明，该方法的性能明显优于现有的基线方法。我们的代码可以在https://github.com/dreamflake/CFM.上找到



## **21. Block Coordinate Descent on Smooth Manifolds**

光滑流形上的块坐标下降 math.OC

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14744v1) [paper-pdf](http://arxiv.org/pdf/2305.14744v1)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems (e.g., generalized PCA, optimal transport, matrix factorization, Burer-Monteiro factorization, outlier-robust estimation, alternating projection, maximal coding rate reduction, neural collapse, adversarial attacks, homomorphic sensing), either yielding novel corollaries or recovering previously known results.

摘要: 块坐标下降是一种每次迭代更新一个变量块的优化范例，由于其可扩展性和性能，使其非常适合大数据应用。它的收敛行为在(分块)凸的情况下已经被广泛地研究，但在非凸的情况下的研究要少得多。本文分析了块坐标方法在非凸集上的收敛，并在自然或弱于已有工作的假设下，得到了光滑流形上的收敛速度。我们的分析适用于许多非凸问题(例如，广义主成分分析、最优传输、矩阵因式分解、布里-蒙泰罗因式分解、离群点稳健估计、交替投影、最大码率降低、神经崩溃、敌意攻击、同态检测)，要么产生新的推论，要么恢复已有的已知结果。



## **22. Adversarial Machine Learning and Cybersecurity: Risks, Challenges, and Legal Implications**

对抗性机器学习和网络安全：风险、挑战和法律含义 cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14553v1) [paper-pdf](http://arxiv.org/pdf/2305.14553v1)

**Authors**: Micah Musser, Andrew Lohn, James X. Dempsey, Jonathan Spring, Ram Shankar Siva Kumar, Brenda Leong, Christina Liaghati, Cindy Martinez, Crystal D. Grant, Daniel Rohrer, Heather Frase, Jonathan Elliott, John Bansemer, Mikel Rodriguez, Mitt Regan, Rumman Chowdhury, Stefan Hermanek

**Abstract**: In July 2022, the Center for Security and Emerging Technology (CSET) at Georgetown University and the Program on Geopolitics, Technology, and Governance at the Stanford Cyber Policy Center convened a workshop of experts to examine the relationship between vulnerabilities in artificial intelligence systems and more traditional types of software vulnerabilities. Topics discussed included the extent to which AI vulnerabilities can be handled under standard cybersecurity processes, the barriers currently preventing the accurate sharing of information about AI vulnerabilities, legal issues associated with adversarial attacks on AI systems, and potential areas where government support could improve AI vulnerability management and mitigation.   This report is meant to accomplish two things. First, it provides a high-level discussion of AI vulnerabilities, including the ways in which they are disanalogous to other types of vulnerabilities, and the current state of affairs regarding information sharing and legal oversight of AI vulnerabilities. Second, it attempts to articulate broad recommendations as endorsed by the majority of participants at the workshop.

摘要: 2022年7月，乔治城大学安全与新兴技术中心(CSET)和斯坦福网络政策中心的地缘政治、技术和治理项目召开了一次专家研讨会，研究人工智能系统中的漏洞与更传统类型的软件漏洞之间的关系。讨论的主题包括在标准网络安全流程下可以在多大程度上处理人工智能漏洞，目前阻碍准确共享人工智能漏洞信息的障碍，与对人工智能系统的对抗性攻击相关的法律问题，以及政府支持可以改善人工智能漏洞管理和缓解的潜在领域。这份报告意在完成两件事。首先，它提供了对人工智能漏洞的高级别讨论，包括它们与其他类型的漏洞的不同之处，以及有关信息共享和对人工智能漏洞的法律监督的现状。第二，它试图阐明研讨会上大多数与会者赞同的广泛建议。



## **23. Translate your gibberish: black-box adversarial attack on machine translation systems**

翻译你的胡言乱语：对机器翻译系统的黑箱对抗性攻击 cs.CL

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2303.10974v2) [paper-pdf](http://arxiv.org/pdf/2303.10974v2)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.

摘要: 神经网络在工业规模的自然语言处理任务中被广泛部署，也许最常被用作自动机器翻译系统的复合体。在这项工作中，我们提出了一种简单的方法，在从俄语到英语的翻译任务中愚弄最先进的机器翻译工具，反之亦然。使用一种新的黑盒无梯度张量优化器，我们证明了许多在线翻译工具，如Google，DeepL和Yandex，都可能对无意义的对抗性输入查询产生错误或攻击性的翻译，并拒绝翻译看似良性的输入短语。此漏洞可能会干扰对新语言的理解，并只会恶化用户在使用机器翻译系统时的体验，因此，需要对这些工具进行额外的改进才能建立更好的翻译。



## **24. The Best Defense is a Good Offense: Adversarial Augmentation against Adversarial Attacks**

最好的防御就是好的进攻：对抗对手攻击的对抗性增强 cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14188v1) [paper-pdf](http://arxiv.org/pdf/2305.14188v1)

**Authors**: Iuri Frosio, Jan Kautz

**Abstract**: Many defenses against adversarial attacks (\eg robust classifiers, randomization, or image purification) use countermeasures put to work only after the attack has been crafted. We adopt a different perspective to introduce $A^5$ (Adversarial Augmentation Against Adversarial Attacks), a novel framework including the first certified preemptive defense against adversarial attacks. The main idea is to craft a defensive perturbation to guarantee that any attack (up to a given magnitude) towards the input in hand will fail. To this aim, we leverage existing automatic perturbation analysis tools for neural networks. We study the conditions to apply $A^5$ effectively, analyze the importance of the robustness of the to-be-defended classifier, and inspect the appearance of the robustified images. We show effective on-the-fly defensive augmentation with a robustifier network that ignores the ground truth label, and demonstrate the benefits of robustifier and classifier co-training. In our tests, $A^5$ consistently beats state of the art certified defenses on MNIST, CIFAR10, FashionMNIST and Tinyimagenet. We also show how to apply $A^5$ to create certifiably robust physical objects. Our code at https://github.com/NVlabs/A5 allows experimenting on a wide range of scenarios beyond the man-in-the-middle attack tested here, including the case of physical attacks.

摘要: 许多对抗攻击的防御(例如，稳健的分类器、随机化或图像净化)使用的对策只有在攻击被精心设计之后才起作用。我们采用了不同的视角来引入$A^5$(对抗性增强对抗攻击)，这是一个新的框架，包括第一个认证的针对对抗性攻击的先发制人防御。主要的想法是制造一个防御性的扰动，以保证对手头输入的任何攻击(直到给定的幅度)都会失败。为此，我们利用现有的神经网络自动扰动分析工具。我们研究了有效应用$A^5的条件，分析了要防御的分类器的稳健性的重要性，并检查了粗暴图像的外观。我们展示了使用忽略地面事实标签的强化器网络进行有效的动态防御增强，并展示了强化器和分类器联合训练的好处。在我们的测试中，$A^5$始终超过MNIST、CIFAR10、FashionMNIST和TinyImagenet上最先进的认证防御。我们还展示了如何应用$A^5$来创建可证明健壮的物理对象。我们在https://github.com/NVlabs/A5的代码允许在这里测试的中间人攻击之外的广泛场景中进行实验，包括物理攻击的情况。



## **25. Improving the Accuracy-Robustness Trade-Off of Classifiers via Adaptive Smoothing**

用自适应平滑提高分类器的精度和稳健性 cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2301.12554v2) [paper-pdf](http://arxiv.org/pdf/2301.12554v2)

**Authors**: Yatong Bai, Brendon G. Anderson, Aerin Kim, Somayeh Sojoudi

**Abstract**: While prior research has proposed a plethora of methods that enhance the adversarial robustness of neural classifiers, practitioners are still reluctant to adopt these techniques due to their unacceptably severe penalties in clean accuracy. This paper shows that by mixing the output probabilities of a standard classifier and a robust model, where the standard network is optimized for clean accuracy and is not robust in general, this accuracy-robustness trade-off can be significantly alleviated. We show that the robust base classifier's confidence difference for correct and incorrect examples is the key ingredient of this improvement. In addition to providing intuitive and empirical evidence, we also theoretically certify the robustness of the mixed classifier under realistic assumptions. Furthermore, we adapt an adversarial input detector into a mixing network that adaptively adjusts the mixture of the two base models, further reducing the accuracy penalty of achieving robustness. The proposed flexible method, termed "adaptive smoothing", can work in conjunction with existing or even future methods that improve clean accuracy, robustness, or adversary detection. Our empirical evaluation considers strong attack methods, including AutoAttack and adaptive attack. On the CIFAR-100 dataset, our method achieves an 85.21% clean accuracy while maintaining a 38.72% $\ell_\infty$-AutoAttacked ($\epsilon$=8/255) accuracy, becoming the second most robust method on the RobustBench CIFAR-100 benchmark as of submission, while improving the clean accuracy by ten percentage points compared with all listed models. The code that implements our method is available at https://github.com/Bai-YT/AdaptiveSmoothing.

摘要: 虽然先前的研究已经提出了过多的方法来增强神经分类器的对抗性稳健性，但实践者仍然不愿采用这些技术，因为它们在干净的准确性上受到了不可接受的严厉惩罚。本文表明，通过混合标准分类器和稳健模型的输出概率，其中标准网络针对干净的精度进行了优化，而通常不是稳健的，这种精度与稳健性的权衡可以得到显著缓解。结果表明，稳健基分类器对正确样本和错误样本的置信度差异是这种改进的关键因素。除了提供直观的经验证据外，我们还从理论上证明了混合分类器在现实假设下的稳健性。此外，我们将对抗性输入检测器引入混合网络，该混合网络自适应地调整两个基本模型的混合，从而进一步降低了实现稳健性的精度损失。这一灵活的方法被称为“自适应平滑”，可以与现有甚至未来的方法结合使用，以提高干净的准确性、健壮性或敌手检测。我们的经验评估考虑了强攻击方法，包括AutoAttack和自适应攻击。在CIFAR-100数据集上，我们的方法达到了85.21%的清洁准确率，同时保持了38.72%的$\ELL_\INFTY$-AutoAttack($\epsilon$=8/255)精度，成为截至提交时在RobustBuchCIFAR-100基准上第二健壮的方法，同时与所有列出的模型相比，清洁准确率提高了10个百分点。实现我们方法的代码可以在https://github.com/Bai-YT/AdaptiveSmoothing.上找到



## **26. Impact of Scaled Image on Robustness of Deep Neural Networks**

尺度图像对深度神经网络稳健性的影响 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02132v2) [paper-pdf](http://arxiv.org/pdf/2209.02132v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have been widely used in computer vision tasks like image classification, object detection and segmentation. Whereas recent studies have shown their vulnerability to manual digital perturbations or distortion in the input images. The accuracy of the networks is remarkably influenced by the data distribution of their training dataset. Scaling the raw images creates out-of-distribution data, which makes it a possible adversarial attack to fool the networks. In this work, we propose a Scaling-distortion dataset ImageNet-CS by Scaling a subset of the ImageNet Challenge dataset by different multiples. The aim of our work is to study the impact of scaled images on the performance of advanced DNNs. We perform experiments on several state-of-the-art deep neural network architectures on the proposed ImageNet-CS, and the results show a significant positive correlation between scaling size and accuracy decline. Moreover, based on ResNet50 architecture, we demonstrate some tests on the performance of recent proposed robust training techniques and strategies like Augmix, Revisiting and Normalizer Free on our proposed ImageNet-CS. Experiment results have shown that these robust training techniques can improve networks' robustness to scaling transformation.

摘要: 深度神经网络在图像分类、目标检测和分割等计算机视觉任务中有着广泛的应用。然而，最近的研究表明，它们在输入图像中容易受到人工数字干扰或失真的影响。训练数据集的数据分布对网络的精度有很大影响。对原始图像进行缩放会产生分布不均的数据，这使得它可能成为愚弄网络的敌意攻击。在这项工作中，我们通过对ImageNet挑战数据集的子集进行不同倍数的缩放，提出了一个缩放失真数据集ImageNet-CS。我们工作的目的是研究缩放图像对高级DNN性能的影响。我们在提出的ImageNet-CS上对几种最先进的深度神经网络结构进行了实验，结果表明，尺度大小与准确率下降呈显著正相关。此外，基于ResNet50体系结构，我们在我们提出的ImageNet-CS上对最近提出的健壮训练技术和策略，如AugMix、Revising和Normal izer Free的性能进行了测试。实验结果表明，这些稳健的训练技术可以提高网络对尺度变换的鲁棒性。



## **27. Impact of Colour Variation on Robustness of Deep Neural Networks**

颜色变化对深度神经网络稳健性的影响 cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02132

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02832v2) [paper-pdf](http://arxiv.org/pdf/2209.02832v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have have shown state-of-the-art performance for computer vision applications like image classification, segmentation and object detection. Whereas recent advances have shown their vulnerability to manual digital perturbations in the input data, namely adversarial attacks. The accuracy of the networks is significantly affected by the data distribution of their training dataset. Distortions or perturbations on color space of input images generates out-of-distribution data, which make networks more likely to misclassify them. In this work, we propose a color-variation dataset by distorting their RGB color on a subset of the ImageNet with 27 different combinations. The aim of our work is to study the impact of color variation on the performance of DNNs. We perform experiments on several state-of-the-art DNN architectures on the proposed dataset, and the result shows a significant correlation between color variation and loss of accuracy. Furthermore, based on the ResNet50 architecture, we demonstrate some experiments of the performance of recently proposed robust training techniques and strategies, such as Augmix, revisit, and free normalizer, on our proposed dataset. Experimental results indicate that these robust training techniques can improve the robustness of deep networks to color variation.

摘要: 深度神经网络(DNN)在图像分类、分割和目标检测等计算机视觉应用中表现出了最先进的性能。然而，最近的进展表明，它们容易受到输入数据中的人工数字扰动，即对抗性攻击。训练数据集的数据分布对网络的精度有很大的影响。输入图像颜色空间的失真或扰动会产生不分布的数据，这使得网络更有可能对它们进行错误分类。在这项工作中，我们提出了一个颜色变化数据集，通过在ImageNet的一个子集上使用27种不同的组合来扭曲它们的RGB颜色。我们工作的目的是研究颜色变化对DNN性能的影响。我们在提出的数据集上对几种最先进的DNN结构进行了实验，结果表明颜色变化与准确率损失之间存在显著的相关性。此外，基于ResNet50体系结构，我们展示了最近提出的稳健训练技术和策略的一些实验，例如AugMix、Revises和Free Normizer，在我们提出的数据集上。实验结果表明，这些稳健的训练技术可以提高深度网络对颜色变化的鲁棒性。



## **28. Adversarial Zoom Lens: A Novel Physical-World Attack to DNNs**

对抗性变焦镜头：一种新的物理世界对DNN的攻击 cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.12251v2) [paper-pdf](http://arxiv.org/pdf/2206.12251v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Although deep neural networks (DNNs) are known to be fragile, no one has studied the effects of zooming-in and zooming-out of images in the physical world on DNNs performance. In this paper, we demonstrate a novel physical adversarial attack technique called Adversarial Zoom Lens (AdvZL), which uses a zoom lens to zoom in and out of pictures of the physical world, fooling DNNs without changing the characteristics of the target object. The proposed method is so far the only adversarial attack technique that does not add physical adversarial perturbation attack DNNs. In a digital environment, we construct a data set based on AdvZL to verify the antagonism of equal-scale enlarged images to DNNs. In the physical environment, we manipulate the zoom lens to zoom in and out of the target object, and generate adversarial samples. The experimental results demonstrate the effectiveness of AdvZL in both digital and physical environments. We further analyze the antagonism of the proposed data set to the improved DNNs. On the other hand, we provide a guideline for defense against AdvZL by means of adversarial training. Finally, we look into the threat possibilities of the proposed approach to future autonomous driving and variant attack ideas similar to the proposed attack.

摘要: 尽管深度神经网络(DNN)被认为是脆弱的，但还没有人研究物理世界中图像的放大和缩小对DNN性能的影响。在本文中，我们展示了一种新的物理对抗性攻击技术，称为对抗性变焦镜头(AdvZL)，它使用变焦镜头来放大和缩小物理世界的图像，在不改变目标对象特征的情况下愚弄DNN。该方法是迄今为止唯一一种不添加物理对抗性扰动攻击DNN的对抗性攻击技术。在数字环境下，我们构建了一个基于AdvZL的数据集，以验证等比例放大图像对DNN的对抗。在物理环境中，我们操纵变焦镜头来放大和缩小目标对象，并生成对抗性样本。实验结果证明了AdvZL在数字和物理环境中的有效性。我们进一步分析了所提出的数据集对改进的DNN的对抗性。另一方面，我们通过对抗性训练的方式提供了防御AdvZL的指导方针。最后，我们展望了所提出的方法对未来自动驾驶的威胁可能性，以及类似于所提出的攻击的不同攻击思想。



## **29. Impact of Light and Shadow on Robustness of Deep Neural Networks**

光影对深度神经网络稳健性的影响 cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02832,  arXiv:2209.02132

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14165v1) [paper-pdf](http://arxiv.org/pdf/2305.14165v1)

**Authors**: Chengyin Hu, Weiwen Shi, Chao Li, Jialiang Sun, Donghua Wang, Junqi Wu, Guijian Tang

**Abstract**: Deep neural networks (DNNs) have made remarkable strides in various computer vision tasks, including image classification, segmentation, and object detection. However, recent research has revealed a vulnerability in advanced DNNs when faced with deliberate manipulations of input data, known as adversarial attacks. Moreover, the accuracy of DNNs is heavily influenced by the distribution of the training dataset. Distortions or perturbations in the color space of input images can introduce out-of-distribution data, resulting in misclassification. In this work, we propose a brightness-variation dataset, which incorporates 24 distinct brightness levels for each image within a subset of ImageNet. This dataset enables us to simulate the effects of light and shadow on the images, so as is to investigate the impact of light and shadow on the performance of DNNs. In our study, we conduct experiments using several state-of-the-art DNN architectures on the aforementioned dataset. Through our analysis, we discover a noteworthy positive correlation between the brightness levels and the loss of accuracy in DNNs. Furthermore, we assess the effectiveness of recently proposed robust training techniques and strategies, including AugMix, Revisit, and Free Normalizer, using the ResNet50 architecture on our brightness-variation dataset. Our experimental results demonstrate that these techniques can enhance the robustness of DNNs against brightness variation, leading to improved performance when dealing with images exhibiting varying brightness levels.

摘要: 深度神经网络(DNN)在图像分类、分割和目标检测等各种计算机视觉任务中取得了显著进展。然而，最近的研究揭示了高级DNN在面对输入数据的故意操纵时存在的漏洞，即所谓的对抗性攻击。此外，DNN的准确率受训练数据集分布的影响很大。输入图像的颜色空间中的失真或扰动可能会引入不分布的数据，从而导致错误分类。在这项工作中，我们提出了一个亮度变化数据集，它包含了ImageNet子集内每幅图像的24个不同的亮度级别。该数据集使我们能够模拟光照和阴影对图像的影响，从而研究光照和阴影对DNN性能的影响。在我们的研究中，我们使用了几种最先进的DNN体系结构在上述数据集上进行了实验。通过我们的分析，我们发现在DNN中亮度级别和精度损失之间存在值得注意的正相关关系。此外，我们使用ResNet50架构在我们的亮度变化数据集上评估了最近提出的健壮训练技术和策略的有效性，包括AugMix、Revise和Free Normal izer。我们的实验结果表明，这些技术可以增强DNN对亮度变化的稳健性，从而在处理显示出不同亮度级别的图像时提高性能。



## **30. QFA2SR: Query-Free Adversarial Transfer Attacks to Speaker Recognition Systems**

QFA2SR：对说话人识别系统的无查询对抗性转移攻击 cs.CR

Accepted by the 32nd USENIX Security Symposium (2023 USENIX  Security); Full Version

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14097v1) [paper-pdf](http://arxiv.org/pdf/2305.14097v1)

**Authors**: Guangke Chen, Yedi Zhang, Zhe Zhao, Fu Song

**Abstract**: Current adversarial attacks against speaker recognition systems (SRSs) require either white-box access or heavy black-box queries to the target SRS, thus still falling behind practical attacks against proprietary commercial APIs and voice-controlled devices. To fill this gap, we propose QFA2SR, an effective and imperceptible query-free black-box attack, by leveraging the transferability of adversarial voices. To improve transferability, we present three novel methods, tailored loss functions, SRS ensemble, and time-freq corrosion. The first one tailors loss functions to different attack scenarios. The latter two augment surrogate SRSs in two different ways. SRS ensemble combines diverse surrogate SRSs with new strategies, amenable to the unique scoring characteristics of SRSs. Time-freq corrosion augments surrogate SRSs by incorporating well-designed time-/frequency-domain modification functions, which simulate and approximate the decision boundary of the target SRS and distortions introduced during over-the-air attacks. QFA2SR boosts the targeted transferability by 20.9%-70.7% on four popular commercial APIs (Microsoft Azure, iFlytek, Jingdong, and TalentedSoft), significantly outperforming existing attacks in query-free setting, with negligible effect on the imperceptibility. QFA2SR is also highly effective when launched over the air against three wide-spread voice assistants (Google Assistant, Apple Siri, and TMall Genie) with 60%, 46%, and 70% targeted transferability, respectively.

摘要: 当前针对说话人识别系统的敌意攻击需要对目标说话人识别系统进行白盒访问或繁重的黑盒查询，因此仍然落后于针对专有商业API和语音控制设备的实际攻击。为了填补这一空白，我们提出了QFA2SR，一种有效的、不可察觉的、无查询的黑盒攻击，它利用了对抗性声音的可传递性。为了提高可转移性，我们提出了三种新的方法，定制损失函数、SRS集成和时频腐蚀。第一种是根据不同的攻击场景定制损失函数。后两者以两种不同的方式增加代理SRS。SRS合奏将不同的代理SRS与新的策略结合在一起，符合SRS的独特得分特征。时频腐蚀通过引入精心设计的时/频域修改函数来增强代理SRS，该函数模拟和近似目标SRS的决策边界和在空中攻击期间引入的失真。QFA2SR在微软Azure、iFLYTEK、京东、TalentedSoft四个热门商业API上，目标可转移性提升20.9%-70.7%，在免查询环境下显著优于现有攻击，对不可感知性影响微乎其微。QFA2SR在与三种广泛使用的语音助手(Google Assistant、Apple Siri和TMall Genie)进行空中传输时也非常有效，目标可转移率分别为60%、46%和70%。



## **31. Adversarial Catoptric Light: An Effective, Stealthy and Robust Physical-World Attack to DNNs**

对抗性反射光：对DNN的一种有效、隐身和健壮的物理世界攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.09652,  arXiv:2209.02430

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.11739v2) [paper-pdf](http://arxiv.org/pdf/2209.11739v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Deep neural networks (DNNs) have demonstrated exceptional success across various tasks, underscoring the need to evaluate the robustness of advanced DNNs. However, traditional methods using stickers as physical perturbations to deceive classifiers present challenges in achieving stealthiness and suffer from printing loss. Recent advancements in physical attacks have utilized light beams such as lasers and projectors to perform attacks, where the optical patterns generated are artificial rather than natural. In this study, we introduce a novel physical attack, adversarial catoptric light (AdvCL), where adversarial perturbations are generated using a common natural phenomenon, catoptric light, to achieve stealthy and naturalistic adversarial attacks against advanced DNNs in a black-box setting. We evaluate the proposed method in three aspects: effectiveness, stealthiness, and robustness. Quantitative results obtained in simulated environments demonstrate the effectiveness of the proposed method, and in physical scenarios, we achieve an attack success rate of 83.5%, surpassing the baseline. We use common catoptric light as a perturbation to enhance the stealthiness of the method and make physical samples appear more natural. Robustness is validated by successfully attacking advanced and robust DNNs with a success rate over 80% in all cases. Additionally, we discuss defense strategy against AdvCL and put forward some light-based physical attacks.

摘要: 深度神经网络(DNN)已经在各种任务中表现出了非凡的成功，这突显了评估高级DNN的健壮性的必要性。然而，传统的使用贴纸作为物理扰动来欺骗分类器的方法在实现隐蔽性方面提出了挑战，并且遭受了印刷损失。物理攻击的最新进展是利用激光和投影仪等光束进行攻击，其中产生的光学图案是人造的，而不是自然的。在这项研究中，我们引入了一种新的物理攻击，对抗反射光(AdvCL)，其中对抗扰动是利用一种常见的自然现象-反射光来产生的，以在黑盒环境下实现对高级DNN的隐蔽和自然主义的对抗攻击。我们从有效性、隐蔽性和稳健性三个方面对所提出的方法进行了评估。在模拟环境中获得的定量结果证明了该方法的有效性，在物理场景中，我们达到了83.5%的攻击成功率，超过了基线。我们使用普通的反射光作为扰动，增强了方法的隐蔽性，使物理样品看起来更自然。通过成功攻击高级和健壮的DNN验证了健壮性，在所有情况下成功率都在80%以上。此外，我们还讨论了针对AdvCL的防御策略，并提出了一些基于光的物理攻击。



## **32. MultiRobustBench: Benchmarking Robustness Against Multiple Attacks**

MultiRobustBch：针对多个攻击的健壮性基准 cs.LG

ICML 2023

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2302.10980v2) [paper-pdf](http://arxiv.org/pdf/2302.10980v2)

**Authors**: Sihui Dai, Saeed Mahloujifar, Chong Xiang, Vikash Sehwag, Pin-Yu Chen, Prateek Mittal

**Abstract**: The bulk of existing research in defending against adversarial examples focuses on defending against a single (typically bounded Lp-norm) attack, but for a practical setting, machine learning (ML) models should be robust to a wide variety of attacks. In this paper, we present the first unified framework for considering multiple attacks against ML models. Our framework is able to model different levels of learner's knowledge about the test-time adversary, allowing us to model robustness against unforeseen attacks and robustness against unions of attacks. Using our framework, we present the first leaderboard, MultiRobustBench, for benchmarking multiattack evaluation which captures performance across attack types and attack strengths. We evaluate the performance of 16 defended models for robustness against a set of 9 different attack types, including Lp-based threat models, spatial transformations, and color changes, at 20 different attack strengths (180 attacks total). Additionally, we analyze the state of current defenses against multiple attacks. Our analysis shows that while existing defenses have made progress in terms of average robustness across the set of attacks used, robustness against the worst-case attack is still a big open problem as all existing models perform worse than random guessing.

摘要: 现有的大量研究集中于防御单一(通常有界的Lp范数)攻击，但对于实际环境，机器学习(ML)模型应该对各种攻击具有健壮性。在这篇文章中，我们提出了第一个考虑针对ML模型的多重攻击的统一框架。我们的框架能够对学习者关于测试时间对手的不同级别的知识进行建模，使我们能够建模对意外攻击的健壮性和对攻击组合的健壮性。使用我们的框架，我们提出了第一个排行榜，MultiRobustBch，用于对多攻击进行基准评估，该评估捕获了攻击类型和攻击强度的性能。我们评估了16种防御模型在20种不同攻击强度(总共180次攻击)下对9种不同攻击类型的健壮性，包括基于LP的威胁模型、空间变换和颜色变化。此外，我们还分析了当前对多种攻击的防御状态。我们的分析表明，尽管现有防御在使用的一组攻击的平均健壮性方面取得了进展，但对最坏情况攻击的健壮性仍然是一个巨大的开放问题，因为所有现有模型的表现都不如随机猜测。



## **33. Adversarial Color Film: Effective Physical-World Attack to DNNs**

对抗性彩色电影：对DNN的有效物理世界攻击 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.02430v2) [paper-pdf](http://arxiv.org/pdf/2209.02430v2)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: It is well known that the performance of deep neural networks (DNNs) is susceptible to subtle interference. So far, camera-based physical adversarial attacks haven't gotten much attention, but it is the vacancy of physical attack. In this paper, we propose a simple and efficient camera-based physical attack called Adversarial Color Film (AdvCF), which manipulates the physical parameters of color film to perform attacks. Carefully designed experiments show the effectiveness of the proposed method in both digital and physical environments. In addition, experimental results show that the adversarial samples generated by AdvCF have excellent performance in attack transferability, which enables AdvCF effective black-box attacks. At the same time, we give the guidance of defense against AdvCF by means of adversarial training. Finally, we look into AdvCF's threat to future vision-based systems and propose some promising mentality for camera-based physical attacks.

摘要: 众所周知，深度神经网络(DNN)的性能容易受到细微干扰的影响。到目前为止，基于摄像机的身体对抗攻击还没有得到太多的关注，但它是身体攻击的空白。本文提出了一种简单而有效的基于摄像机的物理攻击方法，称为对抗性彩色胶片攻击(AdvCF)，它通过操纵彩色胶片的物理参数来进行攻击。精心设计的实验表明，该方法在数字和物理环境中都是有效的。此外，实验结果表明，由AdvCF生成的对抗性样本具有良好的攻击可转移性，使得AdvCF能够有效地进行黑盒攻击。同时，通过对抗性训练的方式，指导对Advcf的防御。最后，我们展望了AdvCF对未来基于视觉的系统的威胁，并对基于摄像机的物理攻击提出了一些有前景的思路。



## **34. Expressive Losses for Verified Robustness via Convex Combinations**

通过凸组合验证稳健性的表达损失 cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13991v1) [paper-pdf](http://arxiv.org/pdf/2305.13991v1)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth, Alessio Lomuscio

**Abstract**: In order to train networks for verified adversarial robustness, previous work typically over-approximates the worst-case loss over (subsets of) perturbation regions or induces verifiability on top of adversarial training. The key to state-of-the-art performance lies in the expressivity of the employed loss function, which should be able to match the tightness of the verifiers to be employed post-training. We formalize a definition of expressivity, and show that it can be satisfied via simple convex combinations between adversarial attacks and IBP bounds. We then show that the resulting algorithms, named CC-IBP and MTL-IBP, yield state-of-the-art results across a variety of settings in spite of their conceptual simplicity. In particular, for $\ell_\infty$ perturbations of radius $\frac{1}{255}$ on TinyImageNet and downscaled ImageNet, MTL-IBP improves on the best standard and verified accuracies from the literature by from $1.98\%$ to $3.92\%$ points while only relying on single-step adversarial attacks.

摘要: 为了训练网络以获得经过验证的对抗健壮性，以前的工作通常过度逼近扰动区域(子集)上的最坏情况损失，或者在对抗训练的基础上引入可验证性。最先进的表现的关键在于受雇损失函数的表现力，它应该能够与受雇培训后受雇的验证员的严密性相匹配。我们形式化了可表现性的定义，并证明了它可以通过对抗性攻击和IBP界之间的简单凸组合来满足。然后，我们展示了最终的算法，命名为CC-IBP和MTL-IBP，尽管它们在概念上很简单，但在各种环境下都能产生最先进的结果。特别是，对于TinyImageNet和缩小ImageNet上半径为$FRAC{1}{255}$的$\ell_\inty$摄动，MTL-IBP在仅依赖单步对手攻击的情况下，将文献中的最佳标准和验证精度从$1.98\$提高到$3.92\$。



## **35. Adversarial Color Projection: A Projector-based Physical Attack to DNNs**

对抗性颜色投影：一种基于投影器的对DNN的物理攻击 cs.CR

arXiv admin note: substantial text overlap with arXiv:2209.02430

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2209.09652v2) [paper-pdf](http://arxiv.org/pdf/2209.09652v2)

**Authors**: Chengyin Hu, Weiwen Shi, Ling Tian

**Abstract**: Recent research has demonstrated that deep neural networks (DNNs) are vulnerable to adversarial perturbations. Therefore, it is imperative to evaluate the resilience of advanced DNNs to adversarial attacks. However, traditional methods that use stickers as physical perturbations to deceive classifiers face challenges in achieving stealthiness and are susceptible to printing loss. Recently, advancements in physical attacks have utilized light beams, such as lasers, to perform attacks, where the optical patterns generated are artificial rather than natural. In this work, we propose a black-box projector-based physical attack, referred to as adversarial color projection (AdvCP), which manipulates the physical parameters of color projection to perform an adversarial attack. We evaluate our approach on three crucial criteria: effectiveness, stealthiness, and robustness. In the digital environment, we achieve an attack success rate of 97.60% on a subset of ImageNet, while in the physical environment, we attain an attack success rate of 100% in the indoor test and 82.14% in the outdoor test. The adversarial samples generated by AdvCP are compared with baseline samples to demonstrate the stealthiness of our approach. When attacking advanced DNNs, experimental results show that our method can achieve more than 85% attack success rate in all cases, which verifies the robustness of AdvCP. Finally, we consider the potential threats posed by AdvCP to future vision-based systems and applications and suggest some ideas for light-based physical attacks.

摘要: 最近的研究表明，深度神经网络(DNN)很容易受到对手的干扰。因此，评估高级DNN对敌意攻击的恢复能力势在必行。然而，使用贴纸作为物理扰动来欺骗分类器的传统方法在实现隐蔽性方面面临挑战，并且容易受到印刷损失的影响。最近，物理攻击的进步利用了光束，如激光，来执行攻击，其中产生的光学图案是人造的，而不是自然的。在这项工作中，我们提出了一种基于黑盒投影仪的物理攻击，称为对抗性颜色投影(AdvCP)，它通过操纵颜色投影的物理参数来执行对抗性攻击。我们根据三个关键标准来评估我们的方法：有效性、隐蔽性和健壮性。在数字环境下，我们对ImageNet的一个子集的攻击成功率达到了97.60%，而在物理环境中，我们在室内测试中达到了100%的攻击成功率，在室外测试中达到了82.14%的攻击成功率。将AdvCP生成的敌意样本与基线样本进行比较，证明了该方法的隐蔽性。在攻击高级DNN时，实验结果表明，该方法在所有情况下都能达到85%以上的攻击成功率，验证了AdvCP的健壮性。最后，我们考虑了AdvCP对未来基于视觉的系统和应用构成的潜在威胁，并对基于光的物理攻击提出了一些想法。



## **36. Unsafe Diffusion: On the Generation of Unsafe Images and Hateful Memes From Text-To-Image Models**

不安全的传播：从文本到图像模型中不安全图像和仇恨模因的产生 cs.CV

To Appear in the ACM Conference on Computer and Communications  Security, November 26, 2023

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13873v1) [paper-pdf](http://arxiv.org/pdf/2305.13873v1)

**Authors**: Yiting Qu, Xinyue Shen, Xinlei He, Michael Backes, Savvas Zannettou, Yang Zhang

**Abstract**: State-of-the-art Text-to-Image models like Stable Diffusion and DALLE$\cdot$2 are revolutionizing how people generate visual content. At the same time, society has serious concerns about how adversaries can exploit such models to generate unsafe images. In this work, we focus on demystifying the generation of unsafe images and hateful memes from Text-to-Image models. We first construct a typology of unsafe images consisting of five categories (sexually explicit, violent, disturbing, hateful, and political). Then, we assess the proportion of unsafe images generated by four advanced Text-to-Image models using four prompt datasets. We find that these models can generate a substantial percentage of unsafe images; across four models and four prompt datasets, 14.56% of all generated images are unsafe. When comparing the four models, we find different risk levels, with Stable Diffusion being the most prone to generating unsafe content (18.92% of all generated images are unsafe). Given Stable Diffusion's tendency to generate more unsafe content, we evaluate its potential to generate hateful meme variants if exploited by an adversary to attack a specific individual or community. We employ three image editing methods, DreamBooth, Textual Inversion, and SDEdit, which are supported by Stable Diffusion. Our evaluation result shows that 24% of the generated images using DreamBooth are hateful meme variants that present the features of the original hateful meme and the target individual/community; these generated images are comparable to hateful meme variants collected from the real world. Overall, our results demonstrate that the danger of large-scale generation of unsafe images is imminent. We discuss several mitigating measures, such as curating training data, regulating prompts, and implementing safety filters, and encourage better safeguard tools to be developed to prevent unsafe generation.

摘要: 最先进的文本到图像模型，如稳定扩散和DALE$\CDOT$2，正在彻底改变人们生成视觉内容的方式。与此同时，社会对对手如何利用这种模式来生成不安全的图像感到严重关切。在这项工作中，我们专注于揭开文本到图像模型中不安全图像和可恨迷因的生成的神秘面纱。我们首先构建了一个由五个类别(性暴露、暴力、令人不安、可恨和政治)组成的不安全形象的类型学。然后，我们使用四个Prompt数据集评估了四个高级文本到图像模型生成的不安全图像的比例。我们发现这些模型可以生成相当大比例的不安全图像；在四个模型和四个提示数据集上，14.56%的生成图像是不安全的。当比较这四种模型时，我们发现不同的风险级别，其中稳定扩散最容易产生不安全的内容(18.92%的生成图像是不安全的)。鉴于稳定扩散倾向于生成更多不安全的内容，我们评估了如果被对手利用来攻击特定个人或社区，它可能会生成令人憎恨的模因变体。我们使用了三种图像编辑方法：DreamBooth、Text Inversion和SDEdit，它们都得到了稳定扩散的支持。我们的评估结果表明，使用DreamBooth生成的图像中有24%是仇恨模因变体，呈现了原始仇恨模因和目标个人/社区的特征；这些生成的图像可以与从现实世界收集的仇恨模因变体相媲美。总体而言，我们的结果表明，大规模生成不安全图像的危险迫在眉睫。我们讨论了几种缓解措施，如整理训练数据、规范提示和实施安全过滤器，并鼓励开发更好的保障工具来防止不安全的产生。



## **37. A Study on the Efficiency and Generalization of Light Hybrid Retrievers**

轻型混合取样器的效率与推广研究 cs.IR

accepted to ACL23

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2210.01371v2) [paper-pdf](http://arxiv.org/pdf/2210.01371v2)

**Authors**: Man Luo, Shashank Jain, Anchit Gupta, Arash Einolghozati, Barlas Oguz, Debojeet Chatterjee, Xilun Chen, Chitta Baral, Peyman Heidari

**Abstract**: Hybrid retrievers can take advantage of both sparse and dense retrievers. Previous hybrid retrievers leverage indexing-heavy dense retrievers. In this work, we study "Is it possible to reduce the indexing memory of hybrid retrievers without sacrificing performance"? Driven by this question, we leverage an indexing-efficient dense retriever (i.e. DrBoost) and introduce a LITE retriever that further reduces the memory of DrBoost. LITE is jointly trained on contrastive learning and knowledge distillation from DrBoost. Then, we integrate BM25, a sparse retriever, with either LITE or DrBoost to form light hybrid retrievers. Our Hybrid-LITE retriever saves 13X memory while maintaining 98.0% performance of the hybrid retriever of BM25 and DPR. In addition, we study the generalization capacity of our light hybrid retrievers on out-of-domain dataset and a set of adversarial attacks datasets. Experiments showcase that light hybrid retrievers achieve better generalization performance than individual sparse and dense retrievers. Nevertheless, our analysis shows that there is a large room to improve the robustness of retrievers, suggesting a new research direction.

摘要: 混合寻回犬可以同时利用稀疏和密集寻回犬的优点。以前的混合取回器利用索引繁重的密集取回器。在这项工作中，我们研究了“是否有可能在不牺牲性能的情况下减少混合检索器的索引内存”？在这个问题的驱动下，我们利用了索引效率高的密集检索器(即DrBoost)，并引入了Lite检索器，进一步减少了DrBoost的内存。Lite是DrBoost在对比学习和知识提炼方面的联合培训。然后，我们将稀疏取回器BM25与Lite或DrBoost相结合，形成轻型混合取回器。我们的混合轻量级检索器节省了13倍的内存，同时保持了BM25和DPR混合检索器98.0%的性能。此外，我们还研究了轻型混合检索器在域外数据集和一组对抗性攻击数据集上的泛化能力。实验表明，轻型混合检索器比单个稀疏和密集检索器具有更好的泛化性能。然而，我们的分析表明，检索器的稳健性还有很大的提高空间，这为检索器的研究提供了一个新的方向。



## **38. Adversarial Laser Spot: Robust and Covert Physical-World Attack to DNNs**

敌意激光光斑：对DNN的强大而隐蔽的物理世界攻击 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.01034v2) [paper-pdf](http://arxiv.org/pdf/2206.01034v2)

**Authors**: Chengyin Hu, Yilong Wang, Kalibinuer Tiliwalidi, Wen Li

**Abstract**: Most existing deep neural networks (DNNs) are easily disturbed by slight noise. However, there are few researches on physical attacks by deploying lighting equipment. The light-based physical attacks has excellent covertness, which brings great security risks to many vision-based applications (such as self-driving). Therefore, we propose a light-based physical attack, called adversarial laser spot (AdvLS), which optimizes the physical parameters of laser spots through genetic algorithm to perform physical attacks. It realizes robust and covert physical attack by using low-cost laser equipment. As far as we know, AdvLS is the first light-based physical attack that perform physical attacks in the daytime. A large number of experiments in the digital and physical environments show that AdvLS has excellent robustness and covertness. In addition, through in-depth analysis of the experimental data, we find that the adversarial perturbations generated by AdvLS have superior adversarial attack migration. The experimental results show that AdvLS impose serious interference to advanced DNNs, we call for the attention of the proposed AdvLS. The code of AdvLS is available at: https://github.com/ChengYinHu/AdvLS

摘要: 现有的大部分深度神经网络(DNN)都容易受到微弱噪声的干扰。然而，通过部署照明设备进行物理攻击的研究很少。基于光的物理攻击具有极好的隐蔽性，这给许多基于视觉的应用(如自动驾驶)带来了很大的安全风险。因此，我们提出了一种基于光的物理攻击，称为对抗性激光光斑(AdvLS)，它通过遗传算法优化激光光斑的物理参数来执行物理攻击。它利用低成本的激光设备实现了健壮隐蔽的物理攻击。据我们所知，AdvLS是第一个在白天执行物理攻击的基于光的物理攻击。在数字和物理环境中的大量实验表明，AdvLS具有良好的健壮性和隐蔽性。此外，通过对实验数据的深入分析，我们发现AdvLS产生的对抗性扰动具有更好的对抗性攻击迁移能力。实验结果表明，AdvLS对高级DNN造成了严重的干扰，我们呼吁注意所提出的AdvLS。有关AdvLS的代码，请访问：https://github.com/ChengYinHu/AdvLS



## **39. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.09241v2) [paper-pdf](http://arxiv.org/pdf/2305.09241v2)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。



## **40. Adversarial Neon Beam: A Light-based Physical Attack to DNNs**

对抗性霓虹灯：对DNN的一种基于光的物理攻击 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2204.00853v3) [paper-pdf](http://arxiv.org/pdf/2204.00853v3)

**Authors**: Chengyin Hu, Weiwen Shi, Wen Li

**Abstract**: In the physical world, deep neural networks (DNNs) are impacted by light and shadow, which can have a significant effect on their performance. While stickers have traditionally been used as perturbations in most physical attacks, their perturbations can often be easily detected. To address this, some studies have explored the use of light-based perturbations, such as lasers or projectors, to generate more subtle perturbations, which are artificial rather than natural. In this study, we introduce a novel light-based attack called the adversarial neon beam (AdvNB), which utilizes common neon beams to create a natural black-box physical attack. Our approach is evaluated on three key criteria: effectiveness, stealthiness, and robustness. Quantitative results obtained in simulated environments demonstrate the effectiveness of the proposed method, and in physical scenarios, we achieve an attack success rate of 81.82%, surpassing the baseline. By using common neon beams as perturbations, we enhance the stealthiness of the proposed attack, enabling physical samples to appear more natural. Moreover, we validate the robustness of our approach by successfully attacking advanced DNNs with a success rate of over 75% in all cases. We also discuss defense strategies against the AdvNB attack and put forward other light-based physical attacks.

摘要: 在物理世界中，深度神经网络(DNN)会受到光和阴影的影响，这会对其性能产生重大影响。虽然传统上，贴纸在大多数物理攻击中都被用作干扰，但它们的干扰通常很容易被检测到。为了解决这个问题，一些研究探索了使用基于光的微扰，如激光或投影仪，来产生更微妙的微扰，这是人为的，而不是自然的。在这项研究中，我们介绍了一种新的基于光的攻击，称为对抗性霓虹灯(AdvNB)，它利用普通霓虹灯来创建一个自然的黑盒物理攻击。我们的方法根据三个关键标准进行评估：有效性、隐蔽性和健壮性。在模拟环境中获得的定量结果证明了该方法的有效性，在物理场景中，我们达到了81.82%的攻击成功率，超过了基线。通过使用普通的霓虹束作为微扰，我们增强了所提出的攻击的隐蔽性，使物理样本看起来更自然。此外，我们通过成功攻击高级DNN来验证我们的方法的健壮性，在所有情况下，成功率都在75%以上。我们还讨论了针对AdvNB攻击的防御策略，并提出了其他基于光的物理攻击。



## **41. Enhancing Accuracy and Robustness through Adversarial Training in Class Incremental Continual Learning**

通过对抗性训练提高课堂增量持续学习的准确性和稳健性 cs.LG

9 pages, 6 figures

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13678v1) [paper-pdf](http://arxiv.org/pdf/2305.13678v1)

**Authors**: Minchan Kwon, Kangil Kim

**Abstract**: In real life, adversarial attack to deep learning models is a fatal security issue. However, the issue has been rarely discussed in a widely used class-incremental continual learning (CICL). In this paper, we address problems of applying adversarial training to CICL, which is well-known defense method against adversarial attack. A well-known problem of CICL is class-imbalance that biases a model to the current task by a few samples of previous tasks. Meeting with the adversarial training, the imbalance causes another imbalance of attack trials over tasks. Lacking clean data of a minority class by the class-imbalance and increasing of attack trials from a majority class by the secondary imbalance, adversarial training distorts optimal decision boundaries. The distortion eventually decreases both accuracy and robustness than adversarial training. To exclude the effects, we propose a straightforward but significantly effective method, External Adversarial Training (EAT) which can be applied to methods using experience replay. This method conduct adversarial training to an auxiliary external model for the current task data at each time step, and applies generated adversarial examples to train the target model. We verify the effects on a toy problem and show significance on CICL benchmarks of image classification. We expect that the results will be used as the first baseline for robustness research of CICL.

摘要: 在现实生活中，对深度学习模型的敌意攻击是一个致命的安全问题。然而，在一个被广泛使用的课堂--增量式持续学习(CICL)中，这个问题很少被讨论。在本文中，我们讨论了将对抗性训练应用到CICL这一著名的对抗攻击防御方法中的问题。CICL的一个众所周知的问题是类不平衡，它通过以前任务的一些样本使模型偏向当前任务。与对抗性训练相遇，这种不平衡导致了另一种攻击试验与任务的不平衡。对抗性训练由于类不平衡而缺乏少数类的干净数据，而次要类不平衡导致多数类攻击试验的增加，扭曲了最优决策边界。与对抗性训练相比，这种失真最终会降低准确性和稳健性。为了排除这些影响，我们提出了一种简单但显著有效的方法-外部对手训练(EAT)，该方法可以应用于使用经验回放的方法。该方法在每个时间步对当前任务数据的辅助外部模型进行对抗性训练，并应用生成的对抗性实例对目标模型进行训练。我们在一个玩具问题上验证了该方法的效果，并在CICL图像分类基准上显示了其意义。我们期望这一结果将作为CICL稳健性研究的第一个基线。



## **42. Adversarial Defenses via Vector Quantization**

基于矢量量化的对抗性防御 cs.LG

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13651v1) [paper-pdf](http://arxiv.org/pdf/2305.13651v1)

**Authors**: Zhiyi Dong, Yongyi Mao

**Abstract**: Building upon Randomized Discretization, we develop two novel adversarial defenses against white-box PGD attacks, utilizing vector quantization in higher dimensional spaces. These methods, termed pRD and swRD, not only offer a theoretical guarantee in terms of certified accuracy, they are also shown, via abundant experiments, to perform comparably or even superior to the current art of adversarial defenses. These methods can be extended to a version that allows further training of the target classifier and demonstrates further improved performance.

摘要: 在随机离散化的基础上，利用高维空间中的矢量量化，提出了两种新的对抗白盒PGD攻击的方法。这些方法被称为PRD和SWRD，不仅在认证的准确性方面提供了理论保证，而且通过大量的实验表明，它们的性能与当前的对抗性防御技术相当，甚至更好。这些方法可以扩展到允许进一步训练目标分类器并表现出进一步改进的性能的版本。



## **43. Hardware Trojans in Power Conversion Circuits**

电源转换电路中的硬件木马 eess.SP

4 pages, 6 figures, will not be submitted to any journals

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13643v1) [paper-pdf](http://arxiv.org/pdf/2305.13643v1)

**Authors**: Jacob Sillman, Ajay Suresh

**Abstract**: This report investigates the potential impact of a Trojan attack on power conversion circuits, specifically a switching signal attack designed to trigger a locking of the pulse width modulation (PWM) signal that goes to a power field-effect transistor (FET). The first simulation shows that this type of attack can cause severe overvoltage, potentially leading to functional failure. The report proposes a solution using a large bypass capacitor to force signal parity, effectively negating the Trojan circuit. The simulation results demonstrate that the proposed solution can effectively thwart the Trojan attack. However, several caveats must be considered, such as the size of the capacitor, possible current leakage, and the possibility that the solution can be circumvented by an adversary with knowledge of the protection strategy. Overall, the findings suggest that proper protection mechanisms, such as the proposed signal-parity solution, must be considered when designing power conversion circuits to mitigate the risk of Trojan attacks.

摘要: 这份报告调查了特洛伊木马攻击对电源转换电路的潜在影响，特别是旨在触发锁定进入功率场效应晶体管(FET)的脉宽调制(PWM)信号的开关信号攻击。第一个模拟表明，这种类型的攻击可能导致严重的过电压，可能导致功能故障。该报告提出了一种解决方案，使用一个大的旁路电容器来强制信号奇偶，有效地否定了特洛伊木马电路。仿真结果表明，该方案能够有效地抵御木马攻击。但是，必须考虑几个注意事项，例如电容器的大小、可能的电流泄漏以及了解保护策略的对手可能绕过解决方案的可能性。总体而言，研究结果表明，在设计电源转换电路以降低特洛伊木马攻击风险时，必须考虑适当的保护机制，例如建议的信号奇偶校验解决方案。



## **44. Adversarial Ensemble Training by Jointly Learning Label Dependencies and Member Models**

联合学习标签依赖关系和成员模型的对抗性集成训练 cs.LG

This paper has been accepted by 19th Inter. Conf. on Intelligent  Computing (ICIC 2023)

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2206.14477v3) [paper-pdf](http://arxiv.org/pdf/2206.14477v3)

**Authors**: Lele Wang, Bin Liu

**Abstract**: Training an ensemble of diverse sub-models has been empirically demonstrated as an effective strategy for improving the adversarial robustness of deep neural networks. However, current ensemble training methods for image recognition typically encode image labels using one-hot vectors, which overlook dependency relationships between the labels. In this paper, we propose a novel adversarial en-semble training approach that jointly learns the label dependencies and member models. Our approach adaptively exploits the learned label dependencies to pro-mote diversity among the member models. We evaluate our approach on widely used datasets including MNIST, FashionMNIST, and CIFAR-10, and show that it achieves superior robustness against black-box attacks compared to state-of-the-art methods. Our code is available at https://github.com/ZJLAB-AMMI/LSD.

摘要: 实验证明，训练不同的子模型集合是提高深层神经网络对抗健壮性的有效策略。然而，当前用于图像识别的集成训练方法通常使用单热点向量来编码图像标签，这忽略了标签之间的依赖关系。在本文中，我们提出了一种新的对抗性集成训练方法，该方法联合学习标签依赖和成员模型。我们的方法自适应地利用学习到的标签依赖关系来促进成员模型之间的多样性。我们在MNIST、FashionMNIST和CIFAR-10等广泛使用的数据集上对我们的方法进行了评估，结果表明，与最先进的方法相比，该方法对黑盒攻击具有更好的鲁棒性。我们的代码可以在https://github.com/ZJLAB-AMMI/LSD.上找到



## **45. Adversarial Infrared Blocks: A Black-box Attack to Thermal Infrared Detectors at Multiple Angles in Physical World**

对抗性红外阻挡：物理世界中对热红外探测器的多角度黑匣子攻击 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2304.10712v2) [paper-pdf](http://arxiv.org/pdf/2304.10712v2)

**Authors**: Chengyin Hu, Weiwen Shi, Tingsong Jiang, Wen Yao, Ling Tian, Xiaoqian Chen

**Abstract**: Infrared imaging systems have a vast array of potential applications in pedestrian detection and autonomous driving, and their safety performance is of great concern. However, few studies have explored the safety of infrared imaging systems in real-world settings. Previous research has used physical perturbations such as small bulbs and thermal "QR codes" to attack infrared imaging detectors, but such methods are highly visible and lack stealthiness. Other researchers have used hot and cold blocks to deceive infrared imaging detectors, but this method is limited in its ability to execute attacks from various angles. To address these shortcomings, we propose a novel physical attack called adversarial infrared blocks (AdvIB). By optimizing the physical parameters of the adversarial infrared blocks, this method can execute a stealthy black-box attack on thermal imaging system from various angles. We evaluate the proposed method based on its effectiveness, stealthiness, and robustness. Our physical tests show that the proposed method achieves a success rate of over 80% under most distance and angle conditions, validating its effectiveness. For stealthiness, our method involves attaching the adversarial infrared block to the inside of clothing, enhancing its stealthiness. Additionally, we test the proposed method on advanced detectors, and experimental results demonstrate an average attack success rate of 51.2%, proving its robustness. Overall, our proposed AdvIB method offers a promising avenue for conducting stealthy, effective and robust black-box attacks on thermal imaging system, with potential implications for real-world safety and security applications.

摘要: 红外成像系统在行人检测和自动驾驶方面有着广泛的潜在应用，其安全性能备受关注。然而，很少有研究探讨红外成像系统在现实世界中的安全性。之前的研究曾使用小灯泡和热二维码等物理扰动来攻击红外成像探测器，但这种方法具有高度可见性和隐蔽性。其他研究人员也曾使用冷热块来欺骗红外成像探测器，但这种方法在从不同角度执行攻击的能力方面受到限制。为了克服这些缺陷，我们提出了一种新的物理攻击，称为对抗性红外线拦截(AdvIB)。该方法通过优化敌方红外块的物理参数，可以从多个角度对热成像系统进行隐身黑匣子攻击。我们从有效性、隐蔽性和稳健性三个方面对提出的方法进行了评估。我们的物理测试表明，该方法在大多数距离和角度条件下都达到了80%以上的成功率，验证了其有效性。对于隐蔽性，我们的方法是将敌对的红外线块安装在衣服的内部，增强其隐蔽性。此外，我们在高级检测器上测试了该方法，实验结果表明该方法的平均攻击成功率为51.2%，证明了该方法的健壮性。总体而言，我们提出的AdvIB方法为对热成像系统进行隐蔽、有效和健壮的黑盒攻击提供了一条很有前途的途径，对现实世界的安全和安保应用具有潜在的影响。



## **46. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

DiffProtect：使用扩散模型生成用于面部隐私保护的敌意示例 cs.CV

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.13625v1) [paper-pdf](http://arxiv.org/pdf/2305.13625v1)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.

摘要: 日益普及的面部识别(FR)系统引发了对个人隐私的严重担忧，特别是对数十亿在社交媒体上公开分享照片的用户来说。已经进行了几次尝试，以保护个人不被未经授权的FR系统利用敌意攻击来生成加密的面部图像来识别。然而，现有的方法存在视觉质量差或攻击成功率低的问题，这限制了它们的实用性。近年来，扩散模型在图像生成方面取得了巨大的成功。在这项工作中，我们问：扩散模型能否被用来生成对抗性例子，以提高视觉质量和攻击性能？我们提出了DiffProtect，它利用扩散自动编码器在FR系统上产生语义上有意义的扰动。大量实验表明，与最先进的方法相比，DiffProtect生成的加密图像看起来更自然，同时实现了显著更高的攻击成功率，例如，在CelebA-HQ和FFHQ数据集上的绝对改进了24.5%和25.1%。



## **47. Attribute-Guided Encryption with Facial Texture Masking**

使用面部纹理掩蔽的属性制导加密 cs.CV

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13548v1) [paper-pdf](http://arxiv.org/pdf/2305.13548v1)

**Authors**: Chun Pong Lau, Jiang Liu, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from unauthorized FR systems utilizing adversarial attacks to generate encrypted face images to protect users from being identified by FR systems. However, existing methods suffer from poor visual quality or low attack success rates, which limit their usability in practice. In this paper, we propose Attribute Guided Encryption with Facial Texture Masking (AGE-FTM) that performs a dual manifold adversarial attack on FR systems to achieve both good visual quality and high black box attack success rates. In particular, AGE-FTM utilizes a high fidelity generative adversarial network (GAN) to generate natural on-manifold adversarial samples by modifying facial attributes, and performs the facial texture masking attack to generate imperceptible off-manifold adversarial samples. Extensive experiments on the CelebA-HQ dataset demonstrate that our proposed method produces more natural-looking encrypted images than state-of-the-art methods while achieving competitive attack performance. We further evaluate the effectiveness of AGE-FTM in the real world using a commercial FR API and validate its usefulness in practice through an user study.

摘要: 日益普及的面部识别(FR)系统引发了对个人隐私的严重担忧，特别是对数十亿在社交媒体上公开分享照片的用户来说。已经进行了几次尝试来保护个人免受未经授权的FR系统的攻击，利用敌意攻击来生成加密的面部图像，以保护用户不被FR系统识别。然而，现有的方法存在视觉质量差或攻击成功率低的问题，这限制了它们在实践中的可用性。在本文中，我们提出了具有面部纹理掩蔽的属性引导加密(AGE-FTM)，它对FR系统进行双重流形对抗攻击，以获得良好的视觉质量和高的黑盒攻击成功率。特别地，AGE-FTM利用高保真的生成对抗网络(GAN)通过修改人脸属性来生成自然的流形上的对抗样本，并执行面部纹理掩蔽攻击来生成不可感知的流形外的对抗样本。在CelebA-HQ数据集上的大量实验表明，与现有方法相比，我们提出的方法产生了更自然的加密图像，同时获得了具有竞争力的攻击性能。我们使用商业FR API进一步评估了AGE-FTM在现实世界中的有效性，并通过用户研究验证了其在实践中的有效性。



## **48. And/or trade-off in artificial neurons: impact on adversarial robustness**

和/或人工神经元的权衡：对抗性稳健性的影响 cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2102.07389v3) [paper-pdf](http://arxiv.org/pdf/2102.07389v3)

**Authors**: Alessandro Fontana

**Abstract**: Despite the success of neural networks, the issue of classification robustness remains, particularly highlighted by adversarial examples. In this paper, we address this challenge by focusing on the continuum of functions implemented in artificial neurons, ranging from pure AND gates to pure OR gates. Our hypothesis is that the presence of a sufficient number of OR-like neurons in a network can lead to classification brittleness and increased vulnerability to adversarial attacks. We define AND-like neurons and propose measures to increase their proportion in the network. These measures involve rescaling inputs to the [-1,1] interval and reducing the number of points in the steepest section of the sigmoidal activation function. A crucial component of our method is the comparison between a neuron's output distribution when fed with the actual dataset and a randomised version called the "scrambled dataset." Experimental results on the MNIST dataset suggest that our approach holds promise as a direction for further exploration.

摘要: 尽管神经网络取得了成功，但分类稳健性的问题仍然存在，尤其是对抗性的例子。在本文中，我们通过关注在人工神经元中实现的功能的连续体来解决这一挑战，从纯AND门到纯OR门。我们的假设是，网络中存在足够数量的类OR神经元会导致分类脆性，并增加对对手攻击的脆弱性。我们定义了AND-Like神经元，并提出了提高其在网络中所占比例的措施。这些措施包括将输入重新调整到[-1，1]区间，并减少S型激活函数最陡峭部分中的点数。我们方法的一个关键组成部分是比较神经元在输入实际数据集时的输出分布与被称为“扰乱数据集”的随机版本之间的比较。在MNIST数据集上的实验结果表明，我们的方法有望成为进一步探索的方向。



## **49. Adversarial Nibbler: A Data-Centric Challenge for Improving the Safety of Text-to-Image Models**

对抗性Nibbler：提高文本到图像模型安全性的以数据为中心的挑战 cs.LG

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.14384v1) [paper-pdf](http://arxiv.org/pdf/2305.14384v1)

**Authors**: Alicia Parrish, Hannah Rose Kirk, Jessica Quaye, Charvi Rastogi, Max Bartolo, Oana Inel, Juan Ciro, Rafael Mosquera, Addison Howard, Will Cukierski, D. Sculley, Vijay Janapa Reddi, Lora Aroyo

**Abstract**: The generative AI revolution in recent years has been spurred by an expansion in compute power and data quantity, which together enable extensive pre-training of powerful text-to-image (T2I) models. With their greater capabilities to generate realistic and creative content, these T2I models like DALL-E, MidJourney, Imagen or Stable Diffusion are reaching ever wider audiences. Any unsafe behaviors inherited from pretraining on uncurated internet-scraped datasets thus have the potential to cause wide-reaching harm, for example, through generated images which are violent, sexually explicit, or contain biased and derogatory stereotypes. Despite this risk of harm, we lack systematic and structured evaluation datasets to scrutinize model behavior, especially adversarial attacks that bypass existing safety filters. A typical bottleneck in safety evaluation is achieving a wide coverage of different types of challenging examples in the evaluation set, i.e., identifying 'unknown unknowns' or long-tail problems. To address this need, we introduce the Adversarial Nibbler challenge. The goal of this challenge is to crowdsource a diverse set of failure modes and reward challenge participants for successfully finding safety vulnerabilities in current state-of-the-art T2I models. Ultimately, we aim to provide greater awareness of these issues and assist developers in improving the future safety and reliability of generative AI models. Adversarial Nibbler is a data-centric challenge, part of the DataPerf challenge suite, organized and supported by Kaggle and MLCommons.

摘要: 近年来发生的人工智能革命是由计算能力和数据量的扩张推动的，这两者一起使强大的文本到图像(T2I)模型的广泛预训练成为可能。凭借其更强大的生成现实和创意内容的能力，这些T2I模式，如Dall-E、MidTrik、Imagen或稳定扩散，正在接触到越来越多的受众。因此，在未经管理的互联网抓取的数据集上进行预培训所继承的任何不安全行为都有可能造成广泛的伤害，例如，通过生成的图像具有暴力、露骨的性或包含偏见和贬损的刻板印象。尽管存在这种危害风险，但我们缺乏系统和结构化的评估数据集来仔细检查模型行为，特别是绕过现有安全过滤器的对抗性攻击。安全评价的一个典型瓶颈是在评价集合中实现不同类型的挑战性实例的广泛覆盖，即识别未知未知数或长尾问题。为了满足这一需求，我们引入了对抗性Nibbler挑战。这项挑战的目标是众包一系列不同的故障模式，并奖励成功发现当前最先进T2I模型中的安全漏洞的挑战赛参与者。最终，我们的目标是提高对这些问题的认识，并帮助开发人员提高生成性AI模型未来的安全性和可靠性。对抗性Nibbler是一个以数据为中心的挑战，是由Kaggle和MLCommons组织和支持的DataPerf挑战套件的一部分。



## **50. Analyzing the Shuffle Model through the Lens of Quantitative Information Flow**

从定量信息流的视角分析洗牌模型 cs.CR

**SubmitDate**: 2023-05-22    [abs](http://arxiv.org/abs/2305.13075v1) [paper-pdf](http://arxiv.org/pdf/2305.13075v1)

**Authors**: Mireya Jurado, Ramon G. Gonze, Mário S. Alvim, Catuscia Palamidessi

**Abstract**: Local differential privacy (LDP) is a variant of differential privacy (DP) that avoids the need for a trusted central curator, at the cost of a worse trade-off between privacy and utility. The shuffle model is a way to provide greater anonymity to users by randomly permuting their messages, so that the link between users and their reported values is lost to the data collector. By combining an LDP mechanism with a shuffler, privacy can be improved at no cost for the accuracy of operations insensitive to permutations, thereby improving utility in many tasks. However, the privacy implications of shuffling are not always immediately evident, and derivations of privacy bounds are made on a case-by-case basis.   In this paper, we analyze the combination of LDP with shuffling in the rigorous framework of quantitative information flow (QIF), and reason about the resulting resilience to inference attacks. QIF naturally captures randomization mechanisms as information-theoretic channels, thus allowing for precise modeling of a variety of inference attacks in a natural way and for measuring the leakage of private information under these attacks. We exploit symmetries of the particular combination of k-RR mechanisms with the shuffle model to achieve closed formulas that express leakage exactly. In particular, we provide formulas that show how shuffling improves protection against leaks in the local model, and study how leakage behaves for various values of the privacy parameter of the LDP mechanism.   In contrast to the strong adversary from differential privacy, we focus on an uninformed adversary, who does not know the value of any individual in the dataset. This adversary is often more realistic as a consumer of statistical datasets, and we show that in some situations mechanisms that are equivalent w.r.t. the strong adversary can provide different privacy guarantees under the uninformed one.

摘要: 本地差异隐私(LDP)是差异隐私(DP)的变体，它避免了对受信任的中央管理员的需要，但代价是在隐私和效用之间进行了更糟糕的权衡。混洗模型是一种通过随机排列用户的消息为用户提供更大匿名性的方法，从而使数据收集器失去用户与其报告的值之间的联系。通过将LDP机制与洗牌器相结合，可以在不花费任何代价的情况下提高私密性，因为操作对排列不敏感，从而提高了在许多任务中的实用性。然而，洗牌对隐私的影响并不总是立竿见影的，隐私界限的推导是基于个案的。在严格的量化信息流(QIF)框架下，我们分析了LDP和置乱的结合，并对其对推理攻击的适应性进行了分析。QIF自然将随机化机制捕获为信息论通道，从而允许以自然的方式对各种推理攻击进行精确建模，并测量这些攻击下私人信息的泄漏。我们利用k-RR机制与Shuffle模型的特殊组合的对称性来获得准确表达泄漏的封闭公式。特别地，我们提供了公式，展示了置乱如何改善对本地模型中的泄漏的保护，并研究了LDP机制的隐私参数的不同值下的泄漏行为。与来自差异隐私的强大对手不同，我们关注的是不知情的对手，他不知道数据集中任何个人的价值。作为统计数据集的消费者，这个对手通常更现实，我们证明了在某些情况下，机制是等价的w.r.t.强大的对手可以在不知情的情况下提供不同的隐私保障。



