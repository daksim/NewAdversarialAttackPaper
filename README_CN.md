# Latest Adversarial Attack Papers
**update at 2024-05-30 19:10:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ConceptPrune: Concept Editing in Diffusion Models via Skilled Neuron Pruning**

ConceptPrune：通过熟练的神经元修剪在扩散模型中进行概念编辑 cs.CV

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19237v1) [paper-pdf](http://arxiv.org/pdf/2405.19237v1)

**Authors**: Ruchika Chavhan, Da Li, Timothy Hospedales

**Abstract**: While large-scale text-to-image diffusion models have demonstrated impressive image-generation capabilities, there are significant concerns about their potential misuse for generating unsafe content, violating copyright, and perpetuating societal biases. Recently, the text-to-image generation community has begun addressing these concerns by editing or unlearning undesired concepts from pre-trained models. However, these methods often involve data-intensive and inefficient fine-tuning or utilize various forms of token remapping, rendering them susceptible to adversarial jailbreaks. In this paper, we present a simple and effective training-free approach, ConceptPrune, wherein we first identify critical regions within pre-trained models responsible for generating undesirable concepts, thereby facilitating straightforward concept unlearning via weight pruning. Experiments across a range of concepts including artistic styles, nudity, object erasure, and gender debiasing demonstrate that target concepts can be efficiently erased by pruning a tiny fraction, approximately 0.12% of total weights, enabling multi-concept erasure and robustness against various white-box and black-box adversarial attacks.

摘要: 虽然大规模的文本到图像传播模型显示了令人印象深刻的图像生成能力，但人们非常担心它们可能被滥用来生成不安全的内容、侵犯版权和永久存在社会偏见。最近，文本到图像生成社区已经开始通过编辑或不学习预先训练的模型中不需要的概念来解决这些问题。然而，这些方法往往涉及数据密集型和低效的微调或利用各种形式的令牌重新映射，使得它们容易受到对抗性越狱的影响。在本文中，我们提出了一种简单而有效的免训练方法ConceptPrune，其中我们首先在预先训练的模型中识别负责产生不想要的概念的关键区域，从而通过权重剪枝来促进直接的概念遗忘。在艺术风格、裸体、对象擦除和性别去偏向等一系列概念上的实验表明，目标概念可以通过修剪极小的部分(约占总权重的0.12%)来有效擦除，从而实现多概念擦除和对各种白盒和黑盒对抗攻击的健壮性。



## **2. Gone but Not Forgotten: Improved Benchmarks for Machine Unlearning**

消失但未被遗忘：机器取消学习的改进基准 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19211v1) [paper-pdf](http://arxiv.org/pdf/2405.19211v1)

**Authors**: Keltin Grimes, Collin Abidi, Cole Frank, Shannon Gallagher

**Abstract**: Machine learning models are vulnerable to adversarial attacks, including attacks that leak information about the model's training data. There has recently been an increase in interest about how to best address privacy concerns, especially in the presence of data-removal requests. Machine unlearning algorithms aim to efficiently update trained models to comply with data deletion requests while maintaining performance and without having to resort to retraining the model from scratch, a costly endeavor. Several algorithms in the machine unlearning literature demonstrate some level of privacy gains, but they are often evaluated only on rudimentary membership inference attacks, which do not represent realistic threats. In this paper we describe and propose alternative evaluation methods for three key shortcomings in the current evaluation of unlearning algorithms. We show the utility of our alternative evaluations via a series of experiments of state-of-the-art unlearning algorithms on different computer vision datasets, presenting a more detailed picture of the state of the field.

摘要: 机器学习模型容易受到敌意攻击，包括泄露模型训练数据信息的攻击。最近，人们对如何最好地解决隐私问题的兴趣有所增加，特别是在存在数据删除请求的情况下。机器遗忘算法的目标是高效地更新训练好的模型，以符合数据删除请求，同时保持性能，而不必求助于从头开始重新训练模型，这是一项代价高昂的工作。机器遗忘文献中的几个算法展示了一定程度的隐私收益，但它们通常只在基本的成员关系推理攻击上进行评估，而这些攻击并不代表现实的威胁。本文针对当前遗忘算法评价中的三个关键缺陷，描述并提出了可供选择的评价方法。我们通过在不同的计算机视觉数据集上进行一系列最先进的遗忘算法的实验，展示了我们的替代评估的实用性，呈现了该领域状态的更详细的图景。



## **3. Model Agnostic Defense against Adversarial Patch Attacks on Object Detection in Unmanned Aerial Vehicles**

无人机目标检测对抗补丁攻击的模型不可知防御 cs.CV

submitted to IROS 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19179v1) [paper-pdf](http://arxiv.org/pdf/2405.19179v1)

**Authors**: Saurabh Pathak, Samridha Shrestha, Abdelrahman AlMahmoud

**Abstract**: Object detection forms a key component in Unmanned Aerial Vehicles (UAVs) for completing high-level tasks that depend on the awareness of objects on the ground from an aerial perspective. In that scenario, adversarial patch attacks on an onboard object detector can severely impair the performance of upstream tasks. This paper proposes a novel model-agnostic defense mechanism against the threat of adversarial patch attacks in the context of UAV-based object detection. We formulate adversarial patch defense as an occlusion removal task. The proposed defense method can neutralize adversarial patches located on objects of interest, without exposure to adversarial patches during training. Our lightweight single-stage defense approach allows us to maintain a model-agnostic nature, that once deployed does not require to be updated in response to changes in the object detection pipeline. The evaluations in digital and physical domains show the feasibility of our method for deployment in UAV object detection pipelines, by significantly decreasing the Attack Success Ratio without incurring significant processing costs. As a result, the proposed defense solution can improve the reliability of object detection for UAVs.

摘要: 目标检测是无人机(UAV)完成高层次任务的关键组成部分，它依赖于从空中角度对地面目标的感知。在这种情况下，对机载对象探测器的敌意补丁攻击可能会严重影响上游任务的性能。针对无人机目标检测中的敌意补丁攻击威胁，提出了一种新的模型不可知防御机制。我们将对抗性补丁防御定义为一种遮挡消除任务。所提出的防御方法可以中和位于感兴趣对象上的对抗性补丁，而不会在训练期间暴露于对抗性补丁。我们的轻量级单级防御方法允许我们保持与模型无关的性质，一旦部署，就不需要更新以响应对象检测管道的变化。数字和物理领域的评估表明，该方法在不产生显著处理成本的情况下，显著降低了攻击成功率，从而在无人机目标检测管道中部署的可行性。结果表明，所提出的防御方案能够提高无人机目标检测的可靠性。



## **4. Introducing Adaptive Continuous Adversarial Training (ACAT) to Enhance ML Robustness**

引入自适应持续对抗训练（ACAT）以增强ML稳健性 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2403.10461v2) [paper-pdf](http://arxiv.org/pdf/2403.10461v2)

**Authors**: Mohamed elShehaby, Aditya Kotha, Ashraf Matrawy

**Abstract**: Adversarial training enhances the robustness of Machine Learning (ML) models against adversarial attacks. However, obtaining labeled training and adversarial training data in network/cybersecurity domains is challenging and costly. Therefore, this letter introduces Adaptive Continuous Adversarial Training (ACAT), a method that integrates adversarial training samples into the model during continuous learning sessions using real-world detected adversarial data. Experimental results with a SPAM detection dataset demonstrate that ACAT reduces the time required for adversarial sample detection compared to traditional processes. Moreover, the accuracy of the under-attack ML-based SPAM filter increased from 69% to over 88% after just three retraining sessions.

摘要: 对抗性训练增强了机器学习（ML）模型对抗对抗性攻击的鲁棒性。然而，在网络/网络安全领域中获得标记训练和对抗训练数据具有挑战性且成本高昂。因此，这封信引入了自适应连续对抗训练（ACAT），这是一种使用现实世界检测到的对抗数据在连续学习会话期间将对抗训练样本集成到模型中的方法。SPAM检测数据集的实验结果表明，与传统过程相比，ACAT减少了对抗性样本检测所需的时间。此外，仅经过三次再培训后，受攻击的基于ML的垃圾邮件过滤器的准确性就从69%提高到了88%以上。



## **5. Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior**

通过函数先验引导的Bayesian优化进行高效的黑匣子对抗攻击 cs.LG

ICML 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19098v1) [paper-pdf](http://arxiv.org/pdf/2405.19098v1)

**Authors**: Shuyu Cheng, Yibo Miao, Yinpeng Dong, Xiao Yang, Xiao-Shan Gao, Jun Zhu

**Abstract**: This paper studies the challenging black-box adversarial attack that aims to generate adversarial examples against a black-box model by only using output feedback of the model to input queries. Some previous methods improve the query efficiency by incorporating the gradient of a surrogate white-box model into query-based attacks due to the adversarial transferability. However, the localized gradient is not informative enough, making these methods still query-intensive. In this paper, we propose a Prior-guided Bayesian Optimization (P-BO) algorithm that leverages the surrogate model as a global function prior in black-box adversarial attacks. As the surrogate model contains rich prior information of the black-box one, P-BO models the attack objective with a Gaussian process whose mean function is initialized as the surrogate model's loss. Our theoretical analysis on the regret bound indicates that the performance of P-BO may be affected by a bad prior. Therefore, we further propose an adaptive integration strategy to automatically adjust a coefficient on the function prior by minimizing the regret bound. Extensive experiments on image classifiers and large vision-language models demonstrate the superiority of the proposed algorithm in reducing queries and improving attack success rates compared with the state-of-the-art black-box attacks. Code is available at https://github.com/yibo-miao/PBO-Attack.

摘要: 研究了一种具有挑战性的黑盒对抗性攻击，其目的是通过只使用模型的输出反馈来输入查询来生成针对黑盒模型的对抗性实例。以前的一些方法通过将代理白盒模型的梯度融入到基于查询的攻击中来提高查询效率，这是因为攻击具有对抗性。然而，局部化的梯度信息不足，使得这些方法仍然是查询密集型的。本文提出了一种先验引导的贝叶斯优化算法(P-BO)，该算法利用代理模型作为黑盒对抗攻击的全局先验函数。由于代理模型包含了丰富的黑盒模型的先验信息，P-BO用一个高斯过程对攻击目标进行建模，其均值函数被初始化为代理模型的损失。我们对遗憾界的理论分析表明，坏的先验可能会影响P-BO的性能。因此，我们进一步提出了一种自适应积分策略，通过最小化遗憾界来自动调整函数先验上的系数。在图像分类器和大型视觉语言模型上的大量实验表明，与最先进的黑盒攻击相比，该算法在减少查询和提高攻击成功率方面具有优势。代码可在https://github.com/yibo-miao/PBO-Attack.上找到



## **6. New perspectives on the optimal placement of detectors for suicide bombers using metaheuristics**

使用元启发法研究自杀式炸弹袭击者探测器最佳放置的新观点 cs.NE

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.19060v1) [paper-pdf](http://arxiv.org/pdf/2405.19060v1)

**Authors**: Carlos Cotta, José E. Gallardo

**Abstract**: We consider an operational model of suicide bombing attacks -- an increasingly prevalent form of terrorism -- against specific targets, and the use of protective countermeasures based on the deployment of detectors over the area under threat. These detectors have to be carefully located in order to minimize the expected number of casualties or the economic damage suffered, resulting in a hard optimization problem for which different metaheuristics have been proposed. Rather than assuming random decisions by the attacker, the problem is approached by considering different models of the latter, whereby he takes informed decisions on which objective must be targeted and through which path it has to be reached based on knowledge on the importance or value of the objectives or on the defensive strategy of the defender (a scenario that can be regarded as an adversarial game). We consider four different algorithms, namely a greedy heuristic, a hill climber, tabu search and an evolutionary algorithm, and study their performance on a broad collection of problem instances trying to resemble different realistic settings such as a coastal area, a modern urban area, and the historic core of an old town. It is shown that the adversarial scenario is harder for all techniques, and that the evolutionary algorithm seems to adapt better to the complexity of the resulting search landscape.

摘要: 我们考虑一种针对具体目标的自杀式爆炸袭击--一种日益普遍的恐怖主义形式--的行动模式，以及在受威胁地区部署探测器的基础上使用保护性对策。这些探测器必须小心地放置，以使预期的伤亡人数或遭受的经济损失最小化，这导致了一个困难的优化问题，对于这个问题，已经提出了不同的元启发式算法。不是假定攻击者的随机决定，而是通过考虑后者的不同模型来处理问题，根据对目标的重要性或价值的了解或防守者的防守策略(可以被视为对抗性游戏的场景)，他根据对目标的重要性或价值的了解，做出关于哪个目标必须成为目标以及必须通过哪条路径到达的明智决定。我们考虑了四种不同的算法，即贪婪启发式算法、爬山者算法、禁忌搜索算法和进化算法，并研究了它们在大量问题实例上的性能，这些问题实例试图类似于不同的现实环境，如沿海地区、现代城市地区和古镇的历史核心。结果表明，对于所有技术来说，对抗性场景都更加困难，而进化算法似乎更好地适应了结果搜索环境的复杂性。



## **7. Verifiably Robust Conformal Prediction**

可验证鲁棒性保形预测 cs.LO

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18942v1) [paper-pdf](http://arxiv.org/pdf/2405.18942v1)

**Authors**: Linus Jeary, Tom Kuipers, Mehran Hosseini, Nicola Paoletti

**Abstract**: Conformal Prediction (CP) is a popular uncertainty quantification method that provides distribution-free, statistically valid prediction sets, assuming that training and test data are exchangeable. In such a case, CP's prediction sets are guaranteed to cover the (unknown) true test output with a user-specified probability. Nevertheless, this guarantee is violated when the data is subjected to adversarial attacks, which often result in a significant loss of coverage. Recently, several approaches have been put forward to recover CP guarantees in this setting. These approaches leverage variations of randomised smoothing to produce conservative sets which account for the effect of the adversarial perturbations. They are, however, limited in that they only support $\ell^2$-bounded perturbations and classification tasks. This paper introduces \emph{VRCP (Verifiably Robust Conformal Prediction)}, a new framework that leverages recent neural network verification methods to recover coverage guarantees under adversarial attacks. Our VRCP method is the first to support perturbations bounded by arbitrary norms including $\ell^1$, $\ell^2$, and $\ell^\infty$, as well as regression tasks. We evaluate and compare our approach on image classification tasks (CIFAR10, CIFAR100, and TinyImageNet) and regression tasks for deep reinforcement learning environments. In every case, VRCP achieves above nominal coverage and yields significantly more efficient and informative prediction regions than the SotA.

摘要: 保角预测是一种流行的不确定性量化方法，它假设训练和测试数据是可交换的，提供了无分布的、统计上有效的预测集。在这种情况下，CP的预测集保证以用户指定的概率覆盖(未知)真实测试输出。然而，当数据受到对抗性攻击时，这一保证就会被违反，这往往会导致覆盖范围的重大损失。最近，已经提出了几种在这种情况下恢复CP担保的方法。这些方法利用随机平滑的变化来产生保守集合，这些保守集合考虑了对抗性扰动的影响。然而，它们的局限性在于它们只支持$^2$有界的扰动和分类任务。介绍了一种新的基于神经网络验证方法的抗攻击覆盖恢复框架--可验证稳健共形预测(VRCP)。我们的VRCP方法是第一个支持以任意范数为界的扰动，包括$^1$，$^2$，$^inty$，以及回归任务。我们在深度强化学习环境下的图像分类任务(CIFAR10、CIFAR100和TinyImageNet)和回归任务上对我们的方法进行了评估和比较。在任何情况下，VRCP都达到了名义覆盖率以上，并产生了比SOTA更有效和更有信息量的预测区域。



## **8. Proactive Load-Shaping Strategies with Privacy-Cost Trade-offs in Residential Households based on Deep Reinforcement Learning**

基于深度强化学习的住宅家庭具有隐私成本权衡的主动负载塑造策略 eess.SY

7 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18888v1) [paper-pdf](http://arxiv.org/pdf/2405.18888v1)

**Authors**: Ruichang Zhang, Youcheng Sun, Mustafa A. Mustafa

**Abstract**: Smart meters play a crucial role in enhancing energy management and efficiency, but they raise significant privacy concerns by potentially revealing detailed user behaviors through energy consumption patterns. Recent scholarly efforts have focused on developing battery-aided load-shaping techniques to protect user privacy while balancing costs. This paper proposes a novel deep reinforcement learning-based load-shaping algorithm (PLS-DQN) designed to protect user privacy by proactively creating artificial load signatures that mislead potential attackers. We evaluate our proposed algorithm against a non-intrusive load monitoring (NILM) adversary. The results demonstrate that our approach not only effectively conceals real energy usage patterns but also outperforms state-of-the-art methods in enhancing user privacy while maintaining cost efficiency.

摘要: 智能电表在提高能源管理和效率方面发挥着至关重要的作用，但它们可能通过能源消耗模式揭示详细的用户行为，从而引发了严重的隐私问题。最近的学术工作重点是开发电池辅助负载整形技术，以保护用户隐私，同时平衡成本。本文提出了一种新型的基于深度强化学习的负载整形算法（PLS-DQN），旨在通过主动创建误导潜在攻击者的人工负载签名来保护用户隐私。我们针对非侵入性负载监控（NILM）对手评估了我们提出的算法。结果表明，我们的方法不仅有效地隐藏了真实的能源使用模式，而且在增强用户隐私的同时保持成本效率方面优于最先进的方法。



## **9. Enhancing Security and Privacy in Federated Learning using Update Digests and Voting-Based Defense**

使用更新摘要和基于投票的防御增强联邦学习中的安全性和隐私 cs.CR

14 pages

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18802v1) [paper-pdf](http://arxiv.org/pdf/2405.18802v1)

**Authors**: Wenjie Li, Kai Fan, Jingyuan Zhang, Hui Li, Wei Yang Bryan Lim, Qiang Yang

**Abstract**: Federated Learning (FL) is a promising privacy-preserving machine learning paradigm that allows data owners to collaboratively train models while keeping their data localized. Despite its potential, FL faces challenges related to the trustworthiness of both clients and servers, especially in the presence of curious or malicious adversaries. In this paper, we introduce a novel framework named \underline{\textbf{F}}ederated \underline{\textbf{L}}earning with \underline{\textbf{U}}pdate \underline{\textbf{D}}igest (FLUD), which addresses the critical issues of privacy preservation and resistance to Byzantine attacks within distributed learning environments. FLUD utilizes an innovative approach, the $\mathsf{LinfSample}$ method, allowing clients to compute the $l_{\infty}$ norm across sliding windows of updates as an update digest. This digest enables the server to calculate a shared distance matrix, significantly reducing the overhead associated with Secure Multi-Party Computation (SMPC) by three orders of magnitude while effectively distinguishing between benign and malicious updates. Additionally, FLUD integrates a privacy-preserving, voting-based defense mechanism that employs optimized SMPC protocols to minimize communication rounds. Our comprehensive experiments demonstrate FLUD's effectiveness in countering Byzantine adversaries while incurring low communication and runtime overhead. FLUD offers a scalable framework for secure and reliable FL in distributed environments, facilitating its application in scenarios requiring robust data management and security.

摘要: 联合学习(FL)是一种很有前途的隐私保护机器学习范例，允许数据所有者在保持数据本地化的同时协作训练模型。尽管有潜力，FL仍面临着与客户端和服务器的可信性相关的挑战，特别是在存在好奇或恶意对手的情况下。针对分布式学习环境中隐私保护和抵抗拜占庭攻击的关键问题，本文提出了一种新的框架-.Flud使用了一种创新的方法，即$\mathsf{LinfSample}$方法，允许客户端跨滑动更新窗口计算$L_{\infty}$范数作为更新摘要。此摘要使服务器能够计算共享距离矩阵，从而将与安全多方计算(SMPC)相关的开销显著降低三个数量级，同时有效区分良性更新和恶意更新。此外，Flud集成了隐私保护、基于投票的防御机制，该机制采用优化的SMPC协议来最大限度地减少通信轮次。我们的综合实验证明了Flud在对抗拜占庭对手方面的有效性，同时产生了较低的通信和运行时间开销。Flud为分布式环境中的安全可靠FL提供了一个可扩展的框架，促进了其在需要强大的数据管理和安全的场景中的应用。



## **10. MIST: Defending Against Membership Inference Attacks Through Membership-Invariant Subspace Training**

MIST：通过授权不变子空间训练防御成员推断攻击 cs.CR

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2311.00919v2) [paper-pdf](http://arxiv.org/pdf/2311.00919v2)

**Authors**: Jiacheng Li, Ninghui Li, Bruno Ribeiro

**Abstract**: In Member Inference (MI) attacks, the adversary try to determine whether an instance is used to train a machine learning (ML) model. MI attacks are a major privacy concern when using private data to train ML models. Most MI attacks in the literature take advantage of the fact that ML models are trained to fit the training data well, and thus have very low loss on training instances. Most defenses against MI attacks therefore try to make the model fit the training data less well. Doing so, however, generally results in lower accuracy. We observe that training instances have different degrees of vulnerability to MI attacks. Most instances will have low loss even when not included in training. For these instances, the model can fit them well without concerns of MI attacks. An effective defense only needs to (possibly implicitly) identify instances that are vulnerable to MI attacks and avoids overfitting them. A major challenge is how to achieve such an effect in an efficient training process. Leveraging two distinct recent advancements in representation learning: counterfactually-invariant representations and subspace learning methods, we introduce a novel Membership-Invariant Subspace Training (MIST) method to defend against MI attacks. MIST avoids overfitting the vulnerable instances without significant impact on other instances. We have conducted extensive experimental studies, comparing MIST with various other state-of-the-art (SOTA) MI defenses against several SOTA MI attacks. We find that MIST outperforms other defenses while resulting in minimal reduction in testing accuracy.

摘要: 在成员推理(MI)攻击中，对手试图确定是否使用实例来训练机器学习(ML)模型。在使用私有数据训练ML模型时，MI攻击是一个主要的隐私问题。文献中的大多数MI攻击都利用了ML模型经过训练以很好地拟合训练数据的事实，因此在训练实例上的损失非常低。因此，大多数针对MI攻击的防御措施都试图使模型不太适合训练数据。然而，这样做通常会导致精度降低。我们观察到，训练实例对MI攻击具有不同程度的脆弱性。即使不包括在培训中，大多数实例的损失也很低。对于这些实例，该模型可以很好地对它们进行拟合，而无需担心MI攻击。有效的防御只需要(可能是隐式地)识别易受MI攻击的实例，并避免过度匹配它们。一个主要的挑战是如何在有效的培训过程中达到这样的效果。利用表示学习的两个不同的最新进展：反事实不变表示和子空间学习方法，我们引入了一种新的成员不变子空间训练(MIST)方法来防御MI攻击。MIST避免对易受攻击的实例过度拟合，而不会对其他实例产生重大影响。我们进行了广泛的实验研究，将MIST与其他各种最先进的(SOTA)MI防御系统进行了比较，以抵御几种SOTA MI攻击。我们发现，MIST的性能优于其他防御系统，同时对测试精度的影响也很小。



## **11. Leveraging Many-To-Many Relationships for Defending Against Visual-Language Adversarial Attacks**

利用多对多关系防御视觉语言对抗攻击 cs.CV

Under review

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18770v1) [paper-pdf](http://arxiv.org/pdf/2405.18770v1)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos

**Abstract**: Recent studies have revealed that vision-language (VL) models are vulnerable to adversarial attacks for image-text retrieval (ITR). However, existing defense strategies for VL models primarily focus on zero-shot image classification, which do not consider the simultaneous manipulation of image and text, as well as the inherent many-to-many (N:N) nature of ITR, where a single image can be described in numerous ways, and vice versa. To this end, this paper studies defense strategies against adversarial attacks on VL models for ITR for the first time. Particularly, we focus on how to leverage the N:N relationship in ITR to enhance adversarial robustness. We found that, although adversarial training easily overfits to specific one-to-one (1:1) image-text pairs in the train data, diverse augmentation techniques to create one-to-many (1:N) / many-to-one (N:1) image-text pairs can significantly improve adversarial robustness in VL models. Additionally, we show that the alignment of the augmented image-text pairs is crucial for the effectiveness of the defense strategy, and that inappropriate augmentations can even degrade the model's performance. Based on these findings, we propose a novel defense strategy that leverages the N:N relationship in ITR, which effectively generates diverse yet highly-aligned N:N pairs using basic augmentations and generative model-based augmentations. This work provides a novel perspective on defending against adversarial attacks in VL tasks and opens up new research directions for future work.

摘要: 最近的研究表明，视觉语言(VL)模型容易受到图像文本检索(ITR)的敌意攻击。然而，现有的VL模型防御策略主要集中在零镜头图像分类，没有考虑图像和文本的同时操作，以及ITR固有的多对多(N：N)性质，其中单个图像可以以多种方式描述，反之亦然。为此，本文首次研究了针对ITR VL模型的对抗攻击防御策略。特别是，我们关注如何利用ITR中的N：N关系来增强对手的健壮性。我们发现，尽管对抗性训练很容易超过训练数据中特定的一对一(1：1)图文对，但创建一对多(1：N)/多对一(N：1)图文对的各种增强技术可以显著提高VL模型中的对抗性健壮性。此外，我们还证明了增强图文对的对齐对于防御策略的有效性是至关重要的，并且不适当的增强甚至会降低模型的性能。基于这些发现，我们提出了一种新的防御策略，该策略利用ITR中的N：N关系，使用基本增强和基于生成性模型的增强有效地生成各种但高度一致的N：N对。这项工作为虚拟学习任务中对抗攻击的防御提供了一个新的视角，并为未来的工作开辟了新的研究方向。



## **12. Genshin: General Shield for Natural Language Processing with Large Language Models**

Genshin：具有大型语言模型的自然语言处理的通用盾牌 cs.CL

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18741v1) [paper-pdf](http://arxiv.org/pdf/2405.18741v1)

**Authors**: Xiao Peng, Tao Liu, Ying Wang

**Abstract**: Large language models (LLMs) like ChatGPT, Gemini, or LLaMA have been trending recently, demonstrating considerable advancement and generalizability power in countless domains. However, LLMs create an even bigger black box exacerbating opacity, with interpretability limited to few approaches. The uncertainty and opacity embedded in LLMs' nature restrict their application in high-stakes domains like financial fraud, phishing, etc. Current approaches mainly rely on traditional textual classification with posterior interpretable algorithms, suffering from attackers who may create versatile adversarial samples to break the system's defense, forcing users to make trade-offs between efficiency and robustness. To address this issue, we propose a novel cascading framework called Genshin (General Shield for Natural Language Processing with Large Language Models), utilizing LLMs as defensive one-time plug-ins. Unlike most applications of LLMs that try to transform text into something new or structural, Genshin uses LLMs to recover text to its original state. Genshin aims to combine the generalizability of the LLM, the discrimination of the median model, and the interpretability of the simple model. Our experiments on the task of sentimental analysis and spam detection have shown fatal flaws of the current median models and exhilarating results on LLMs' recovery ability, demonstrating that Genshin is both effective and efficient. In our ablation study, we unearth several intriguing observations. Utilizing the LLM defender, a tool derived from the 4th paradigm, we have reproduced BERT's 15% optimal mask rate results in the 3rd paradigm of NLP. Additionally, when employing the LLM as a potential adversarial tool, attackers are capable of executing effective attacks that are nearly semantically lossless.

摘要: 像ChatGPT、Gemini或Llama这样的大型语言模型(LLM)最近已经成为趋势，在无数领域展示了相当大的先进性和泛化能力。然而，LLM创建了一个更大的黑匣子，加剧了不透明度，可解释性仅限于几种方法。LLMS本质上的不确定性和不透明性限制了它们在高风险领域的应用，如金融欺诈、网络钓鱼等。目前的方法主要依赖于传统的文本分类和后验可解释算法，攻击者可能会创建通用的对抗性样本来破坏系统的防御，迫使用户在效率和健壮性之间做出权衡。为了解决这个问题，我们提出了一种新颖的级联框架Genshin(General Shield For Natural Language Processing With Large Language Models)，利用LLMS作为防御性的一次性插件。与大多数试图将文本转换为新的或结构化的文本的LLMS应用程序不同，Genshin使用LLMS将文本恢复到其原始状态。Genshin的目标是将LLM的泛化能力、中值模型的区分性和简单模型的可解释性结合起来。我们在情感分析和垃圾邮件检测任务上的实验表明，现有的中值模型存在致命缺陷，并且在LLMS的恢复能力上取得了令人振奋的结果，证明了Genshin是有效的和高效的。在我们的消融研究中，我们发现了几个有趣的观察结果。利用LLM Defender，一个源自第四范式的工具，我们在NLP的第三范式中复制了Bert的15%最优掩蔽率结果。此外，当使用LLM作为潜在的敌意工具时，攻击者能够执行几乎在语义上无损的有效攻击。



## **13. Security--Throughput Tradeoff of Nakamoto Consensus under Bandwidth Constraints**

安全性--带宽限制下中本共识的短期权衡 cs.CR

ACM Conference on Computer and Communications Security 2024

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2303.09113v3) [paper-pdf](http://arxiv.org/pdf/2303.09113v3)

**Authors**: Lucianna Kiffer, Joachim Neu, Srivatsan Sridhar, Aviv Zohar, David Tse

**Abstract**: For Nakamoto's longest-chain consensus protocol, whose proof-of-work (PoW) and proof-of-stake (PoS) variants power major blockchains such as Bitcoin and Cardano, we revisit the classic problem of the security-performance tradeoff: Given a network of nodes with limited capacities, against what fraction of adversary power is Nakamoto consensus (NC) secure for a given block production rate? State-of-the-art analyses of Nakamoto's protocol fail to answer this question because their bounded-delay model does not capture realistic constraints such as limited communication- and computation-resources. We develop a new analysis technique to prove a refined security-performance tradeoff for PoW Nakamoto consensus in a bounded-bandwidth model. In this model, we show that, in contrast to the classic bounded-delay model, Nakamoto's private attack is no longer the worst attack, and a new attack strategy we call the teasing strategy, that exploits the network congestion caused by limited bandwidth, is strictly worse. In PoS, equivocating blocks can exacerbate congestion, making the traditional PoS Nakamoto consensus protocol insecure except at very low block production rates. To counter such equivocation spamming, we present a variant of the PoS NC protocol we call Blanking NC (BlaNC), which achieves the same resilience as PoW NC.

摘要: 对于Nakamoto的最长链共识协议，其工作证明(PoW)和风险证明(Pos)变体为比特币和Cardano等主要区块链提供支持，我们重温了安全与性能权衡的经典问题：给定一个容量有限的节点网络，对于给定的块生产率，相对于对手力量的多少部分，Nakamoto共识(NC)是安全的？对Nakamoto协议的最新分析未能回答这个问题，因为他们的有限延迟模型没有捕捉到现实的约束，如有限的通信和计算资源。我们开发了一种新的分析技术来证明在有限带宽模型中PoW Nakamoto共识的改进的安全与性能权衡。在该模型中，我们证明了与经典的有限延迟模型相比，Nakamoto的私有攻击不再是最糟糕的攻击，而一种新的攻击策略--利用有限带宽造成的网络拥塞--严格地更差。在PoS中，模棱两可的块会加剧拥塞，使得传统的PoS Nakamoto共识协议不安全，除非在非常低的块生产率下。为了应对这种模棱两可的垃圾邮件，我们提出了一种POS NC协议的变体，我们称之为BLANC(BLANC)，它实现了与POW NC相同的弹性。



## **14. Watermarking Counterfactual Explanations**

水印反事实解释 cs.LG

**SubmitDate**: 2024-05-29    [abs](http://arxiv.org/abs/2405.18671v1) [paper-pdf](http://arxiv.org/pdf/2405.18671v1)

**Authors**: Hangzhi Guo, Amulya Yadav

**Abstract**: The field of Explainable Artificial Intelligence (XAI) focuses on techniques for providing explanations to end-users about the decision-making processes that underlie modern-day machine learning (ML) models. Within the vast universe of XAI techniques, counterfactual (CF) explanations are often preferred by end-users as they help explain the predictions of ML models by providing an easy-to-understand & actionable recourse (or contrastive) case to individual end-users who are adversely impacted by predicted outcomes. However, recent studies have shown significant security concerns with using CF explanations in real-world applications; in particular, malicious adversaries can exploit CF explanations to perform query-efficient model extraction attacks on proprietary ML models. In this paper, we propose a model-agnostic watermarking framework (for adding watermarks to CF explanations) that can be leveraged to detect unauthorized model extraction attacks (which rely on the watermarked CF explanations). Our novel framework solves a bi-level optimization problem to embed an indistinguishable watermark into the generated CF explanation such that any future model extraction attacks that rely on these watermarked CF explanations can be detected using a null hypothesis significance testing (NHST) scheme, while ensuring that these embedded watermarks do not compromise the quality of the generated CF explanations. We evaluate this framework's performance across a diverse set of real-world datasets, CF explanation methods, and model extraction techniques, and show that our watermarking detection system can be used to accurately identify extracted ML models that are trained using the watermarked CF explanations. Our work paves the way for the secure adoption of CF explanations in real-world applications.

摘要: 可解释人工智能(XAI)领域的重点是向最终用户提供关于现代机器学习(ML)模型基础的决策过程的解释的技术。在XAI技术的浩瀚宇宙中，反事实(CF)解释往往受到最终用户的青睐，因为它们通过向受到预测结果不利影响的个人最终用户提供易于理解和可操作的资源(或对比)案例来帮助解释ML模型的预测。然而，最近的研究表明，在现实世界的应用程序中使用CF解释存在严重的安全问题；特别是，恶意攻击者可以利用CF解释对专有ML模型执行查询高效的模型提取攻击。在本文中，我们提出了一个模型无关的水印框架(用于在CF解释中添加水印)，该框架可用于检测未经授权的模型提取攻击(依赖于带水印的CF解释)。我们的新框架解决了一个双层优化问题，将不可区分的水印嵌入到生成的CF解释中，使得依赖于这些带水印的CF解释的任何未来模型提取攻击可以使用零假设显著性检验(NHST)方案来检测，同时确保这些嵌入的水印不会损害生成的CF解释的质量。我们在一组不同的真实数据集、CF解释方法和模型提取技术上对该框架的性能进行了评估，并表明我们的水印检测系统可以用于准确识别提取的ML模型，这些模型是使用带水印的CF解释训练的。我们的工作为在实际应用中安全地采用CF解释铺平了道路。



## **15. PureGen: Universal Data Purification for Train-Time Poison Defense via Generative Model Dynamics**

PureGen：通过生成模型动力学进行训练时毒物防御的通用数据净化 cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18627v1) [paper-pdf](http://arxiv.org/pdf/2405.18627v1)

**Authors**: Sunay Bhat, Jeffrey Jiang, Omead Pooladzandi, Alexander Branch, Gregory Pottie

**Abstract**: Train-time data poisoning attacks threaten machine learning models by introducing adversarial examples during training, leading to misclassification. Current defense methods often reduce generalization performance, are attack-specific, and impose significant training overhead. To address this, we introduce a set of universal data purification methods using a stochastic transform, $\Psi(x)$, realized via iterative Langevin dynamics of Energy-Based Models (EBMs), Denoising Diffusion Probabilistic Models (DDPMs), or both. These approaches purify poisoned data with minimal impact on classifier generalization. Our specially trained EBMs and DDPMs provide state-of-the-art defense against various attacks (including Narcissus, Bullseye Polytope, Gradient Matching) on CIFAR-10, Tiny-ImageNet, and CINIC-10, without needing attack or classifier-specific information. We discuss performance trade-offs and show that our methods remain highly effective even with poisoned or distributionally shifted generative model training data.

摘要: 训练时间数据中毒攻击通过在训练过程中引入对抗性示例来威胁机器学习模型，导致错误分类。当前的防御方法通常会降低泛化性能，针对特定攻击，并且会带来显著的训练开销。为了解决这个问题，我们介绍了一套通用的数据净化方法，它使用随机变换$\Psi(X)$，通过基于能量的模型的迭代朗之万动力学(EBMS)或去噪扩散概率模型(DDPM)实现，或者两者兼而有之。这些方法净化有毒数据，对分类器泛化的影响最小。我们经过专门训练的EBM和DDPM提供针对CIFAR-10、Tiny-ImageNet和CINIC-10的各种攻击(包括水仙攻击、Bullseye多面体攻击、梯度匹配攻击)的最先进防御，而不需要攻击或特定于分类器的信息。我们讨论了性能权衡，并表明我们的方法仍然非常有效，即使在有毒或分布转移的生成性模型训练数据的情况下。



## **16. Wavelet-Based Image Tokenizer for Vision Transformers**

用于视觉变形者的基于微波的图像代币化器 cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18616v1) [paper-pdf](http://arxiv.org/pdf/2405.18616v1)

**Authors**: Zhenhai Zhu, Radu Soricut

**Abstract**: Non-overlapping patch-wise convolution is the default image tokenizer for all state-of-the-art vision Transformer (ViT) models. Even though many ViT variants have been proposed to improve its efficiency and accuracy, little research on improving the image tokenizer itself has been reported in the literature. In this paper, we propose a new image tokenizer based on wavelet transformation. We show that ViT models with the new tokenizer achieve both higher training throughput and better top-1 precision for the ImageNet validation set. We present a theoretical analysis on why the proposed tokenizer improves the training throughput without any change to ViT model architecture. Our analysis suggests that the new tokenizer can effectively handle high-resolution images and is naturally resistant to adversarial attack. Furthermore, the proposed image tokenizer offers a fresh perspective on important new research directions for ViT-based model design, such as image tokens on a non-uniform grid for image understanding.

摘要: 非重叠的面片卷积是所有最先进的视觉转换器(VIT)模型的默认图像标记器。尽管已经提出了许多VIT变体来提高其效率和准确性，但文献中关于改进图像标记器本身的研究很少。提出了一种新的基于小波变换的图像标记器。我们表明，使用新的标记器的VIT模型在ImageNet验证集上实现了更高的训练吞吐量和更好的TOP-1精度。我们从理论上分析了为什么在不改变VIT模型结构的情况下，所提出的记号器提高了训练吞吐量。我们的分析表明，新的标记器可以有效地处理高分辨率图像，并且具有天然的抵抗对手攻击的能力。此外，所提出的图像标记器为基于VIT的模型设计提供了新的研究方向，例如用于图像理解的非均匀网格上的图像标记物。



## **17. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2402.17012v2) [paper-pdf](http://arxiv.org/pdf/2402.17012v2)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model which leverages recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Taken together, these results represent the strongest existing privacy attacks against both pretrained and fine-tuned LLMs for MIAs and training data extraction, which are of independent scientific interest and have important practical implications for LLM security, privacy, and copyright issues.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度来预测训练数据的成员资格，以及这种攻击的一个变体，它只需要Logit访问模型，利用了最近在LLMS上的模型窃取工作。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。综上所述，这些结果代表了针对用于MIA和训练数据提取的预先训练和微调的LLM的现有最强隐私攻击，这些攻击具有独立的科学意义，并对LLM的安全、隐私和版权问题具有重要的实践意义。



## **18. Unleashing the potential of prompt engineering: a comprehensive review**

释放即时工程的潜力：全面回顾 cs.CL

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2310.14735v3) [paper-pdf](http://arxiv.org/pdf/2310.14735v3)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review explores the transformative potential of prompt engineering within the realm of large language models (LLMs) and multimodal language models (MMLMs). The development of AI, from its inception in the 1950s to the emergence of neural networks and deep learning architectures, has culminated in sophisticated LLMs like GPT-4 and BERT, as well as MMLMs like DALL-E and CLIP. These models have revolutionized tasks in diverse fields such as workplace automation, healthcare, and education. Prompt engineering emerges as a crucial technique to maximize the utility and accuracy of these models. This paper delves into both foundational and advanced methodologies of prompt engineering, including techniques like Chain of Thought, Self-consistency, and Generated Knowledge, which significantly enhance model performance. Additionally, it examines the integration of multimodal data through innovative approaches such as Multi-modal Prompt Learning (MaPLe), Conditional Prompt Learning, and Context Optimization. Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review underscores the pivotal role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的综述探索了快速工程在大型语言模型(LLM)和多模式语言模型(MMLM)领域中的变革潜力。人工智能从20世纪50年代开始发展到神经网络和深度学习体系结构的出现，最终出现了GPT-4和BERT等复杂的LLM，以及Dall-E和CLIP等MMLM。这些模式使工作场所自动化、医疗保健和教育等不同领域的任务发生了革命性变化。为了最大限度地提高这些模型的实用性和准确性，快速工程技术应运而生。本文深入研究了即时工程的基础和高级方法，包括思想链、自我一致性和生成知识等技术，这些技术显著提高了模型的性能。此外，它还通过多模式快速学习(Maple)、条件性快速学习和上下文优化等创新方法研究了多模式数据的集成。对这一讨论至关重要的是人工智能安全方面，特别是利用即时工程中的漏洞进行的对抗性攻击。对缓解这些风险和增强模型稳健性的策略进行了彻底的回顾。对快速方法的评估通过主观和客观两个指标进行，确保对其有效性进行稳健的分析。这篇综述强调了快速工程在推进人工智能能力方面的关键作用，为未来的研究和应用提供了一个结构化的框架。



## **19. Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing**

通过特定层的编辑保护大型语言模型免受越狱攻击 cs.AI

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18166v1) [paper-pdf](http://arxiv.org/pdf/2405.18166v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Ye Zhang, Jun Sun

**Abstract**: Large language models (LLMs) are increasingly being adopted in a wide range of real-world applications. Despite their impressive performance, recent studies have shown that LLMs are vulnerable to deliberately crafted adversarial prompts even when aligned via Reinforcement Learning from Human Feedback or supervised fine-tuning. While existing defense methods focus on either detecting harmful prompts or reducing the likelihood of harmful responses through various means, defending LLMs against jailbreak attacks based on the inner mechanisms of LLMs remains largely unexplored. In this work, we investigate how LLMs response to harmful prompts and propose a novel defense method termed \textbf{L}ayer-specific \textbf{Ed}iting (LED) to enhance the resilience of LLMs against jailbreak attacks. Through LED, we reveal that several critical \textit{safety layers} exist among the early layers of LLMs. We then show that realigning these safety layers (and some selected additional layers) with the decoded safe response from selected target layers can significantly improve the alignment of LLMs against jailbreak attacks. Extensive experiments across various LLMs (e.g., Llama2, Mistral) show the effectiveness of LED, which effectively defends against jailbreak attacks while maintaining performance on benign prompts. Our code is available at \url{https://github.com/ledllm/ledllm}.

摘要: 大型语言模型(LLM)正越来越多地被广泛地应用于现实世界中。尽管它们的表现令人印象深刻，但最近的研究表明，即使在通过从人类反馈的强化学习或监督微调进行调整时，LLM仍容易受到故意设计的敌意提示的攻击。虽然现有的防御方法侧重于检测有害提示或通过各种手段减少有害响应的可能性，但基于LLMS的内部机制来防御LLMS的越狱攻击在很大程度上仍未被探索。在这项工作中，我们研究了LLMS对有害提示的响应，并提出了一种新的防御方法-.通过LED，我们揭示了LLMS的早期层之间存在着几个关键的安全层。然后，我们展示了将这些安全层(以及一些选定的附加层)与选定目标层的解码安全响应重新对准可以显著提高LLM对抗越狱攻击的对准。在各种LLM(如Llama2、Mistral)上的广泛实验表明，LED是有效的，它可以有效防御越狱攻击，同时保持对良性提示的性能。我们的代码可在\url{https://github.com/ledllm/ledllm}.



## **20. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.18137v1) [paper-pdf](http://arxiv.org/pdf/2405.18137v1)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **21. S-Eval: Automatic and Adaptive Test Generation for Benchmarking Safety Evaluation of Large Language Models**

S-Eval：用于大型语言模型基准安全评估的自动和自适应测试生成 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.14191v3) [paper-pdf](http://arxiv.org/pdf/2405.14191v3)

**Authors**: Xiaohan Yuan, Jinfeng Li, Dongxia Wang, Yuefeng Chen, Xiaofeng Mao, Longtao Huang, Hui Xue, Wenhai Wang, Kui Ren, Jingyi Wang

**Abstract**: Large Language Models have gained considerable attention for their revolutionary capabilities. However, there is also growing concern on their safety implications, making a comprehensive safety evaluation for LLMs urgently needed before model deployment. In this work, we propose S-Eval, a new comprehensive, multi-dimensional and open-ended safety evaluation benchmark. At the core of S-Eval is a novel LLM-based automatic test prompt generation and selection framework, which trains an expert testing LLM Mt combined with a range of test selection strategies to automatically construct a high-quality test suite for the safety evaluation. The key to the automation of this process is a novel expert safety-critique LLM Mc able to quantify the riskiness score of an LLM's response, and additionally produce risk tags and explanations. Besides, the generation process is also guided by a carefully designed risk taxonomy with four different levels, covering comprehensive and multi-dimensional safety risks of concern. Based on these, we systematically construct a new and large-scale safety evaluation benchmark for LLMs consisting of 220,000 evaluation prompts, including 20,000 base risk prompts (10,000 in Chinese and 10,000 in English) and 200,000 corresponding attack prompts derived from 10 popular adversarial instruction attacks against LLMs. Moreover, considering the rapid evolution of LLMs and accompanied safety threats, S-Eval can be flexibly configured and adapted to include new risks, attacks and models. S-Eval is extensively evaluated on 20 popular and representative LLMs. The results confirm that S-Eval can better reflect and inform the safety risks of LLMs compared to existing benchmarks. We also explore the impacts of parameter scales, language environments, and decoding parameters on the evaluation, providing a systematic methodology for evaluating the safety of LLMs.

摘要: 大型语言模型因其革命性的能力而获得了相当大的关注。然而，人们也越来越担心它们的安全影响，这使得在模型部署之前迫切需要对LLMS进行全面的安全评估。在这项工作中，我们提出了一种新的全面、多维、开放式的安全评价基准S-EVAL。S-EVAL的核心是一种新颖的基于LLM的测试提示自动生成和选择框架，该框架训练一名测试专家，结合一系列测试选择策略，自动构建用于安全评估的高质量测试用例集。这一过程自动化的关键是一种新颖的专家安全评论LLm Mc，它能够量化LLm响应的风险分数，并另外产生风险标签和解释。此外，生成过程还遵循了精心设计的四个不同级别的风险分类，涵盖了令人关注的全面和多维度的安全风险。在此基础上，我们系统地构建了一个新的大规模的低层管理系统安全评估基准，该基准由22万条评估提示组成，其中包括2万条基本风险提示(中文10000条，英文10000条)和来自10种流行的对抗性指令攻击的20万条相应的攻击提示。此外，考虑到LLM的快速演化和伴随的安全威胁，S-EVAL可以灵活配置和调整，以包括新的风险、攻击和模型。S-EVAL在20个流行和有代表性的低成本模型上进行了广泛的评估。结果证实，与现有基准相比，S-EVAL能够更好地反映和告知低成本机械的安全风险。我们还探讨了参数尺度、语言环境和解码参数对评估的影响，为评估LLMS的安全性提供了一种系统的方法。



## **22. Towards Unified Robustness Against Both Backdoor and Adversarial Attacks**

迈向针对后门和对抗攻击的统一稳健性 cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17929v1) [paper-pdf](http://arxiv.org/pdf/2405.17929v1)

**Authors**: Zhenxing Niu, Yuyao Sun, Qiguang Miao, Rong Jin, Gang Hua

**Abstract**: Deep Neural Networks (DNNs) are known to be vulnerable to both backdoor and adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct robustness problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, this paper revealed that there is an intriguing connection between them: (1) planting a backdoor into a model will significantly affect the model's adversarial examples; (2) for an infected model, its adversarial examples have similar features as the triggered images. Based on these observations, a novel Progressive Unified Defense (PUD) algorithm is proposed to defend against backdoor and adversarial attacks simultaneously. Specifically, our PUD has a progressive model purification scheme to jointly erase backdoors and enhance the model's adversarial robustness. At the early stage, the adversarial examples of infected models are utilized to erase backdoors. With the backdoor gradually erased, our model purification can naturally turn into a stage to boost the model's robustness against adversarial attacks. Besides, our PUD algorithm can effectively identify poisoned images, which allows the initial extra dataset not to be completely clean. Extensive experimental results show that, our discovered connection between backdoor and adversarial attacks is ubiquitous, no matter what type of backdoor attack. The proposed PUD outperforms the state-of-the-art backdoor defense, including the model repairing-based and data filtering-based methods. Besides, it also has the ability to compete with the most advanced adversarial defense methods.

摘要: 深度神经网络(DNN)很容易受到后门攻击和敌意攻击。在文献中，这两类攻击由于分别属于训练时间攻击和推理时间攻击，通常被视为不同的健壮性问题而分别解决。然而，本文揭示了它们之间有一个有趣的联系：(1)在模型中植入后门将显著影响模型的对抗性示例；(2)对于受感染的模型，其对抗性示例具有与触发图像相似的特征。基于这些观察结果，提出了一种新的渐进统一防御算法(PUD)，以同时防御后门攻击和对手攻击。具体地说，我们的PUD具有渐进式模型净化方案，以联合擦除后门并增强模型的对抗健壮性。在早期阶段，被感染模型的对抗性例子被用来擦除后门。随着后门的逐渐消除，我们的模型净化自然可以变成一个阶段，以增强模型对对手攻击的健壮性。此外，我们的PUD算法可以有效地识别有毒图像，这使得初始额外的数据集不是完全干净的。广泛的实验结果表明，我们发现的后门攻击和敌意攻击之间的联系是普遍存在的，无论是哪种类型的后门攻击。提出的PUD的性能优于最先进的后门防御，包括基于模型修复和基于数据过滤的方法。此外，它还具有与最先进的对抗性防御手段竞争的能力。



## **23. White-box Multimodal Jailbreaks Against Large Vision-Language Models**

针对大型视觉语言模型的白盒多模式越狱 cs.CV

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17894v1) [paper-pdf](http://arxiv.org/pdf/2405.17894v1)

**Authors**: Ruofan Wang, Xingjun Ma, Hanxu Zhou, Chuanjun Ji, Guangnan Ye, Yu-Gang Jiang

**Abstract**: Recent advancements in Large Vision-Language Models (VLMs) have underscored their superiority in various multimodal tasks. However, the adversarial robustness of VLMs has not been fully explored. Existing methods mainly assess robustness through unimodal adversarial attacks that perturb images, while assuming inherent resilience against text-based attacks. Different from existing attacks, in this work we propose a more comprehensive strategy that jointly attacks both text and image modalities to exploit a broader spectrum of vulnerability within VLMs. Specifically, we propose a dual optimization objective aimed at guiding the model to generate affirmative responses with high toxicity. Our attack method begins by optimizing an adversarial image prefix from random noise to generate diverse harmful responses in the absence of text input, thus imbuing the image with toxic semantics. Subsequently, an adversarial text suffix is integrated and co-optimized with the adversarial image prefix to maximize the probability of eliciting affirmative responses to various harmful instructions. The discovered adversarial image prefix and text suffix are collectively denoted as a Universal Master Key (UMK). When integrated into various malicious queries, UMK can circumvent the alignment defenses of VLMs and lead to the generation of objectionable content, known as jailbreaks. The experimental results demonstrate that our universal attack strategy can effectively jailbreak MiniGPT-4 with a 96% success rate, highlighting the vulnerability of VLMs and the urgent need for new alignment strategies.

摘要: 大型视觉语言模型(VLM)的最新进展凸显了它们在各种多通道任务中的优越性。然而，VLMS的对抗健壮性还没有得到充分的研究。现有的方法主要通过扰乱图像的单峰对抗性攻击来评估稳健性，同时假设对基于文本的攻击具有内在的弹性。与已有的攻击不同，我们提出了一种更全面的策略，联合攻击文本和图像模式，以利用VLM中更广泛的漏洞。具体地说，我们提出了一个双重优化目标，旨在引导模型产生高毒性的肯定反应。我们的攻击方法首先从随机噪声中优化一个敌意图像前缀，在没有文本输入的情况下产生不同的有害响应，从而使图像充满有毒语义。随后，对抗性文本后缀与对抗性图像前缀集成并共同优化，以最大限度地引起对各种有害指令的肯定响应的概率。所发现的敌意图像前缀和文本后缀统称为通用主密钥(UMK)。当集成到各种恶意查询中时，UMK可以绕过VLM的对齐防御，并导致生成令人反感的内容，即所谓的越狱。实验结果表明，我们的通用攻击策略能够有效地越狱MiniGPT-4，成功率为96%，凸显了VLMS的脆弱性和对新的对齐策略的迫切需求。



## **24. ALA: Naturalness-aware Adversarial Lightness Attack**

ALA：自然意识对抗性轻量级攻击 cs.CV

9 pages

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2201.06070v3) [paper-pdf](http://arxiv.org/pdf/2201.06070v3)

**Authors**: Yihao Huang, Liangru Sun, Qing Guo, Felix Juefei-Xu, Jiayi Zhu, Jincao Feng, Yang Liu, Geguang Pu

**Abstract**: Most researchers have tried to enhance the robustness of DNNs by revealing and repairing the vulnerability of DNNs with specialized adversarial examples. Parts of the attack examples have imperceptible perturbations restricted by Lp norm. However, due to their high-frequency property, the adversarial examples can be defended by denoising methods and are hard to realize in the physical world. To avoid the defects, some works have proposed unrestricted attacks to gain better robustness and practicality. It is disappointing that these examples usually look unnatural and can alert the guards. In this paper, we propose Adversarial Lightness Attack (ALA), a white-box unrestricted adversarial attack that focuses on modifying the lightness of the images. The shape and color of the samples, which are crucial to human perception, are barely influenced. To obtain adversarial examples with a high attack success rate, we propose unconstrained enhancement in terms of the light and shade relationship in images. To enhance the naturalness of images, we craft the naturalness-aware regularization according to the range and distribution of light. The effectiveness of ALA is verified on two popular datasets for different tasks (i.e., ImageNet for image classification and Places-365 for scene recognition).

摘要: 大多数研究人员试图通过专门的对抗性例子来揭示和修复DNN的脆弱性，从而增强DNN的健壮性。部分攻击实例具有Lp范数约束下的不可察觉扰动。然而，由于对抗性例子的高频特性，可以通过去噪的方法来防御，并且很难在物理世界中实现。为了避免这些缺陷，一些工作提出了无限制攻击，以获得更好的健壮性和实用性。令人失望的是，这些例子通常看起来不自然，可以提醒警卫。在本文中，我们提出了对抗性亮度攻击(ALA)，这是一种白盒无限制的对抗性攻击，其重点是改变图像的亮度。样本的形状和颜色对人类的感知至关重要，几乎不受影响。为了获得攻击成功率较高的对抗性例子，我们提出了基于图像明暗关系的无约束增强。为了增强图像的自然度，我们根据光线的范围和分布进行了自然度感知正则化。在两个用于不同任务的流行数据集(即用于图像分类的ImageNet和用于场景识别的Places-365)上验证了ALA的有效性。



## **25. Unmasking Vulnerabilities: Cardinality Sketches under Adaptive Inputs**

揭露漏洞：自适应输入下的基数草图 cs.DS

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.17780v1) [paper-pdf](http://arxiv.org/pdf/2405.17780v1)

**Authors**: Sara Ahmadian, Edith Cohen

**Abstract**: Cardinality sketches are popular data structures that enhance the efficiency of working with large data sets. The sketches are randomized representations of sets that are only of logarithmic size but can support set merges and approximate cardinality (i.e., distinct count) queries. When queries are not adaptive, that is, they do not depend on preceding query responses, the design provides strong guarantees of correctly answering a number of queries exponential in the sketch size $k$.   In this work, we investigate the performance of cardinality sketches in adaptive settings and unveil inherent vulnerabilities. We design an attack against the ``standard'' estimators that constructs an adversarial input by post-processing responses to a set of simple non-adaptive queries of size linear in the sketch size $k$. Empirically, our attack used only $4k$ queries with the widely used HyperLogLog (HLL++)~\citep{hyperloglog:2007,hyperloglogpractice:EDBT2013} sketch. The simple attack technique suggests it can be effective with post-processed natural workloads. Finally and importantly, we demonstrate that the vulnerability is inherent as \emph{any} estimator applied to known sketch structures can be attacked using a number of queries that is quadratic in $k$, matching a generic upper bound.

摘要: 基数草图是一种流行的数据结构，可以提高处理大型数据集的效率。草图是仅具有对数大小但可以支持集合合并和近似基数(即，DISTINCT计数)查询的集合的随机表示。当查询不是自适应的，即它们不依赖于先前的查询响应时，该设计提供了正确回答以草图大小$k$为指数的数量的查询的强有力的保证。在这项工作中，我们调查了基数草图在自适应设置下的性能，并揭示了固有的漏洞。我们设计了一个针对“标准”估计器的攻击，它通过对一组简单的、大小在草图大小$k$中线性的非自适应查询的响应进行后处理来构造对抗性输入。根据经验，我们的攻击仅使用了$4k$查询和广泛使用的HyperLogLog(HLL++)~\CITEP{HYPERLOG：2007，HYLOGLOG实践：EDBT2013}草图。这种简单的攻击技术表明，它可以有效地处理后处理的自然工作负载。最后也是重要的是，我们证明了该漏洞是固有的，因为应用于已知草图结构的任何估计器都可以使用在$k$中二次的查询来攻击，并且匹配一般的上界。



## **26. Adversarial Attacks on Hidden Tasks in Multi-Task Learning**

多任务学习中隐藏任务的对抗攻击 cs.LG

14 pages, 6 figures

**SubmitDate**: 2024-05-28    [abs](http://arxiv.org/abs/2405.15244v2) [paper-pdf](http://arxiv.org/pdf/2405.15244v2)

**Authors**: Yu Zhe, Rei Nagaike, Daiki Nishiyama, Kazuto Fukuchi, Jun Sakuma

**Abstract**: Deep learning models are susceptible to adversarial attacks, where slight perturbations to input data lead to misclassification. Adversarial attacks become increasingly effective with access to information about the targeted classifier. In the context of multi-task learning, where a single model learns multiple tasks simultaneously, attackers may aim to exploit vulnerabilities in specific tasks with limited information. This paper investigates the feasibility of attacking hidden tasks within multi-task classifiers, where model access regarding the hidden target task and labeled data for the hidden target task are not available, but model access regarding the non-target tasks is available. We propose a novel adversarial attack method that leverages knowledge from non-target tasks and the shared backbone network of the multi-task model to force the model to forget knowledge related to the target task. Experimental results on CelebA and DeepFashion datasets demonstrate the effectiveness of our method in degrading the accuracy of hidden tasks while preserving the performance of visible tasks, contributing to the understanding of adversarial vulnerabilities in multi-task classifiers.

摘要: 深度学习模型容易受到敌意攻击，在这种攻击中，输入数据的轻微扰动会导致错误分类。通过访问有关目标分类器的信息，对抗性攻击变得越来越有效。在多任务学习的情况下，单个模型同时学习多个任务，攻击者可能会利用有限信息的特定任务中的漏洞进行攻击。研究了在多任务分类器中攻击隐藏任务的可行性，其中隐藏目标任务的模型访问和隐藏目标任务的标记数据不可用，而非目标任务的模型访问可用。我们提出了一种新的对抗性攻击方法，该方法利用来自非目标任务的知识和多任务模型的共享骨干网络来迫使模型忘记与目标任务相关的知识。在CelebA和DeepFashion数据集上的实验结果表明，该方法在保持可见任务性能的同时，降低了隐藏任务的准确率，有助于理解多任务分类器中的对抗性弱点。



## **27. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

突破错误的对抗性示例防御：修复1行代码破解Sabre cs.CR

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.03672v2) [paper-pdf](http://arxiv.org/pdf/2405.03672v2)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.

摘要: Sabre是对IEEE S & P 2024上接受的敌对例子的辩护。我们首先揭示了评估中的重大缺陷，这些缺陷表明了梯度掩蔽的明显迹象。然后我们展示这种梯度掩蔽的原因：原始评估代码中的一个错误。通过在原始存储库中修复一行代码，我们将Sabre的稳健准确性降低到0%。作为回应，作者修改了辩护并引入了原始论文中未描述的新辩护组件。但此修复包含第二个错误;修改多一行代码会将稳健准确性降低到基线水平以下。在我们在线发布论文的第一版后，作者对防御进行了另一项更改;通过在攻击期间注释掉一行代码，我们将稳健准确性再次降低到0%。



## **28. Universal Adversarial Defense in Remote Sensing Based on Pre-trained Denoising Diffusion Models**

基于预训练去噪扩散模型的遥感通用对抗防御 cs.CV

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2307.16865v3) [paper-pdf](http://arxiv.org/pdf/2307.16865v3)

**Authors**: Weikang Yu, Yonghao Xu, Pedram Ghamisi

**Abstract**: Deep neural networks (DNNs) have risen to prominence as key solutions in numerous AI applications for earth observation (AI4EO). However, their susceptibility to adversarial examples poses a critical challenge, compromising the reliability of AI4EO algorithms. This paper presents a novel Universal Adversarial Defense approach in Remote Sensing Imagery (UAD-RS), leveraging pre-trained diffusion models to protect DNNs against universal adversarial examples exhibiting heterogeneous patterns. Specifically, a universal adversarial purification framework is developed utilizing pre-trained diffusion models to mitigate adversarial perturbations through the introduction of Gaussian noise and subsequent purification of the perturbations from adversarial examples. Additionally, an Adaptive Noise Level Selection (ANLS) mechanism is introduced to determine the optimal noise level for the purification framework with a task-guided Frechet Inception Distance (FID) ranking strategy, thereby enhancing purification performance. Consequently, only a single pre-trained diffusion model is required for purifying universal adversarial samples with heterogeneous patterns across each dataset, significantly reducing training efforts for multiple attack settings while maintaining high performance without prior knowledge of adversarial perturbations. Experimental results on four heterogeneous RS datasets, focusing on scene classification and semantic segmentation, demonstrate that UAD-RS outperforms state-of-the-art adversarial purification approaches, providing universal defense against seven commonly encountered adversarial perturbations. Codes and the pre-trained models are available online (https://github.com/EricYu97/UAD-RS).

摘要: 深度神经网络(DNN)已成为众多人工智能对地观测应用(AI4EO)中的关键解决方案。然而，它们对敌意例子的敏感性构成了一个关键挑战，损害了AI4EO算法的可靠性。提出了一种新的遥感图像通用对抗防御方法(UAD-RS)，该方法利用预先训练的扩散模型来保护DNN免受表现出异质模式的通用对抗实例的影响。具体地说，利用预先训练的扩散模型，通过引入高斯噪声和随后从对抗性样本中净化扰动来缓解对抗性扰动，开发了一个通用的对抗性净化框架。此外，引入了自适应噪声水平选择(ANLS)机制，通过任务导向的Frechet初始距离(FID)排序策略来确定净化框架的最佳噪声水平，从而提高了净化性能。因此，只需要一个预先训练的扩散模型来提纯具有跨每个数据集的异质模式的通用对抗性样本，大大减少了针对多个攻击设置的训练工作量，同时在不预先知道对抗性扰动的情况下保持高性能。在四个不同类型的RS数据集上的实验结果表明，UAD-RS在场景分类和语义分割方面的性能优于最新的对抗性净化方法，对七种常见的对抗性扰动提供了普遍的防御。代码和预先培训的模型可在线获取(https://github.com/EricYu97/UAD-RS).



## **29. Spectral regularization for adversarially-robust representation learning**

用于对抗稳健表示学习的谱正规化 cs.LG

15 + 15 pages, 8 + 11 figures

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.17181v1) [paper-pdf](http://arxiv.org/pdf/2405.17181v1)

**Authors**: Sheng Yang, Jacob A. Zavatone-Veth, Cengiz Pehlevan

**Abstract**: The vulnerability of neural network classifiers to adversarial attacks is a major obstacle to their deployment in safety-critical applications. Regularization of network parameters during training can be used to improve adversarial robustness and generalization performance. Usually, the network is regularized end-to-end, with parameters at all layers affected by regularization. However, in settings where learning representations is key, such as self-supervised learning (SSL), layers after the feature representation will be discarded when performing inference. For these models, regularizing up to the feature space is more suitable. To this end, we propose a new spectral regularizer for representation learning that encourages black-box adversarial robustness in downstream classification tasks. In supervised classification settings, we show empirically that this method is more effective in boosting test accuracy and robustness than previously-proposed methods that regularize all layers of the network. We then show that this method improves the adversarial robustness of classifiers using representations learned with self-supervised training or transferred from another classification task. In all, our work begins to unveil how representational structure affects adversarial robustness.

摘要: 神经网络分类器对敌意攻击的脆弱性是其在安全关键应用中部署的主要障碍。训练过程中网络参数的正则化可用于提高对手的稳健性和泛化性能。通常，网络是端到端的正则化的，所有层的参数都受正则化的影响。然而，在学习表示是关键的设置中，例如自我监督学习(SSL)，当执行推理时，特征表示之后的层将被丢弃。对于这些模型，最大限度地正则化到特征空间更为合适。为此，我们提出了一种新的用于表示学习的谱正则化方法，以鼓励下游分类任务中的黑盒对抗健壮性。在有监督的分类环境下，我们的经验表明，该方法在提高测试精度和稳健性方面比先前提出的正则化网络所有层的方法更有效。然后，我们证明了这种方法提高了分类器的对抗稳健性，使用了通过自我监督训练学习的表示或从另一个分类任务转移的表示。总而言之，我们的工作开始揭示表征结构如何影响对手的稳健性。



## **30. DD-RobustBench: An Adversarial Robustness Benchmark for Dataset Distillation**

DD-RobustBench：数据集蒸馏的对抗稳健性基准 cs.CV

* denotes equal contributions; ^ denotes corresponding author. In  this updated version, we have expanded our research to include more  experiments on various adversarial attack methods and latest dataset  distillation studies. All new results have been incorporated into the  document

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.13322v2) [paper-pdf](http://arxiv.org/pdf/2403.13322v2)

**Authors**: Yifan Wu, Jiawei Du, Ping Liu, Yuewei Lin, Wenqing Cheng, Wei Xu

**Abstract**: Dataset distillation is an advanced technique aimed at compressing datasets into significantly smaller counterparts, while preserving formidable training performance. Significant efforts have been devoted to promote evaluation accuracy under limited compression ratio while overlooked the robustness of distilled dataset. In this work, we introduce a comprehensive benchmark that, to the best of our knowledge, is the most extensive to date for evaluating the adversarial robustness of distilled datasets in a unified way. Our benchmark significantly expands upon prior efforts by incorporating a wider range of dataset distillation methods, including the latest advancements such as TESLA and SRe2L, a diverse array of adversarial attack methods, and evaluations across a broader and more extensive collection of datasets such as ImageNet-1K. Moreover, we assessed the robustness of these distilled datasets against representative adversarial attack algorithms like PGD and AutoAttack, while exploring their resilience from a frequency perspective. We also discovered that incorporating distilled data into the training batches of the original dataset can yield to improvement of robustness.

摘要: 数据集精馏是一种高级技术，旨在将数据集压缩成小得多的对应物，同时保持强大的训练性能。人们一直致力于提高有限压缩比下的评估精度，而忽略了提取数据集的稳健性。在这项工作中，我们引入了一个全面的基准，据我们所知，这是到目前为止最广泛的评估提取数据集的对抗稳健性的统一方式。我们的基准显著扩展了之前的工作，纳入了更广泛的数据集蒸馏方法，包括最新的进步，如特斯拉和SRe2L，多种对抗性攻击方法，以及对更广泛的数据集集合(如ImageNet-1K)的评估。此外，我们评估了这些提取的数据集对PGD和AutoAttack等典型对抗性攻击算法的健壮性，同时从频率的角度探讨了它们的弹性。我们还发现，将提取的数据结合到原始数据集的训练批次中可以提高稳健性。



## **31. Local Model Reconstruction Attacks in Federated Learning and their Uses**

联邦学习中的局部模型重建攻击及其用途 cs.LG

we discover bugs in experiments

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2210.16205v3) [paper-pdf](http://arxiv.org/pdf/2210.16205v3)

**Authors**: Ilias Driouich, Chuan Xu, Giovanni Neglia, Frederic Giroire, Eoin Thomas

**Abstract**: In this paper, we initiate the study of local model reconstruction attacks for federated learning, where a honest-but-curious adversary eavesdrops the messages exchanged between a targeted client and the server, and then reconstructs the local/personalized model of the victim. The local model reconstruction attack allows the adversary to trigger other classical attacks in a more effective way, since the local model only depends on the client's data and can leak more private information than the global model learned by the server. Additionally, we propose a novel model-based attribute inference attack in federated learning leveraging the local model reconstruction attack. We provide an analytical lower-bound for this attribute inference attack. Empirical results using real world datasets confirm that our local reconstruction attack works well for both regression and classification tasks. Moreover, we benchmark our novel attribute inference attack against the state-of-the-art attacks in federated learning. Our attack results in higher reconstruction accuracy especially when the clients' datasets are heterogeneous. Our work provides a new angle for designing powerful and explainable attacks to effectively quantify the privacy risk in FL.

摘要: 在本文中，我们发起了联合学习的本地模型重建攻击的研究，其中诚实但好奇的攻击者窃听目标客户端和服务器之间交换的消息，然后重建受害者的本地/个性化模型。本地模型重构攻击允许攻击者以更有效的方式触发其他经典攻击，因为本地模型仅依赖于客户端的数据，并且可以比服务器学习的全局模型泄露更多的私人信息。此外，利用局部模型重构攻击，提出了一种新的联邦学习中基于模型的属性推理攻击。我们给出了这种属性推理攻击的一个分析下界。使用真实世界数据集的实验结果证实，我们的局部重建攻击对于回归和分类任务都很好地工作。此外，我们还对联邦学习中最新的属性推理攻击进行了基准测试。我们的攻击导致了更高的重建精度，特别是当客户的数据集是异质的时候。我们的工作为设计强大的、可解释的攻击以有效量化FL中的隐私风险提供了一个新的角度。



## **32. Verifying Properties of Binary Neural Networks Using Sparse Polynomial Optimization**

使用稀疏多元优化的二元神经网络特性 cs.LG

22 pages, 2 figures, 7 tables

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.17049v1) [paper-pdf](http://arxiv.org/pdf/2405.17049v1)

**Authors**: Jianting Yang, Srećko Ðurašinović, Jean-Bernard Lasserre, Victor Magron, Jun Zhao

**Abstract**: This paper explores methods for verifying the properties of Binary Neural Networks (BNNs), focusing on robustness against adversarial attacks. Despite their lower computational and memory needs, BNNs, like their full-precision counterparts, are also sensitive to input perturbations. Established methods for solving this problem are predominantly based on Satisfiability Modulo Theories and Mixed-Integer Linear Programming techniques, which are characterized by NP complexity and often face scalability issues.   We introduce an alternative approach using Semidefinite Programming relaxations derived from sparse Polynomial Optimization. Our approach, compatible with continuous input space, not only mitigates numerical issues associated with floating-point calculations but also enhances verification scalability through the strategic use of tighter first-order semidefinite relaxations. We demonstrate the effectiveness of our method in verifying robustness against both $\|.\|_\infty$ and $\|.\|_2$-based adversarial attacks.

摘要: 本文探讨了二进制神经网络(BNN)的性能验证方法，重点研究了BNN对敌意攻击的鲁棒性。尽管BNN的计算和内存需求较低，但与全精度网络一样，BNN对输入扰动也很敏感。已有的解决这一问题的方法主要基于可满足性模理论和混合整数线性规划技术，这些方法具有NP复杂性并且经常面临可伸缩性问题。我们介绍了一种利用稀疏多项式优化的半定规划松弛的替代方法。我们的方法与连续输入空间兼容，不仅缓解了与浮点计算相关的数值问题，而且通过策略性地使用更紧的一阶半定松弛来增强验证的可伸缩性。我们证明了我们的方法在对基于$.INFTY$和基于$.2$的敌意攻击的健壮性方面是有效的。



## **33. OSLO: One-Shot Label-Only Membership Inference Attacks**

Oslo：一次性标签会员推断攻击 cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16978v1) [paper-pdf](http://arxiv.org/pdf/2405.16978v1)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is 7$\times$ to 28$\times$ stronger under a 0.1\% FPR on CIFAR10 for a ResNet model. We evaluated multiple defense mechanisms against OSLO.

摘要: 我们引入了一次仅标签(Oslo)成员关系推理攻击(MIA)，该攻击仅使用目标模型返回预测的硬标签，即可高精度地推断给定样本在目标模型训练集中的成员资格。这与最先进的纯标签攻击形成对比，后者需要$\sim6000$查询，但获得的攻击精度低于奥斯陆的攻击精度。奥斯陆利用基于传输的黑盒对抗攻击。其核心思想是成员样本比非成员样本对对抗性扰动表现出更强的抵抗力。我们将Oslo与最先进的纯标签攻击进行了比较，并证明了尽管只需要一次查询，但在相同的误检率(FPR)下，我们的方法在准确率和真阳性率(TPR)方面明显优于以前的攻击。例如，与以前的纯标签MIA相比，对于ResNet模型，Oslo在CIFAR10上实现了7$\x$到28$\x$的TPR。我们评估了针对奥斯陆的多种防御机制。



## **34. Adversarial Attacks on Both Face Recognition and Face Anti-spoofing Models**

对人脸识别和人脸反欺骗模型的对抗攻击 cs.CV

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16940v1) [paper-pdf](http://arxiv.org/pdf/2405.16940v1)

**Authors**: Fengfan Zhou, Qianyu Zhou, Xiangtai Li, Xuequan Lu, Lizhuang Ma, Hefei Ling

**Abstract**: Adversarial attacks on Face Recognition (FR) systems have proven highly effective in compromising pure FR models, yet adversarial examples may be ineffective to the complete FR systems as Face Anti-Spoofing (FAS) models are often incorporated and can detect a significant number of them. To address this under-explored and essential problem, we propose a novel setting of adversarially attacking both FR and FAS models simultaneously, aiming to enhance the practicability of adversarial attacks on FR systems. In particular, we introduce a new attack method, namely Style-aligned Distribution Biasing (SDB), to improve the capacity of black-box attacks on both FR and FAS models. Specifically, our SDB framework consists of three key components. Firstly, to enhance the transferability of FAS models, we design a Distribution-aware Score Biasing module to optimize adversarial face examples away from the distribution of spoof images utilizing scores. Secondly, to mitigate the substantial style differences between live images and adversarial examples initialized with spoof images, we introduce an Instance Style Alignment module that aligns the style of adversarial examples with live images. In addition, to alleviate the conflicts between the gradients of FR and FAS models, we propose a Gradient Consistency Maintenance module to minimize disparities between the gradients using Hessian approximation. Extensive experiments showcase the superiority of our proposed attack method to state-of-the-art adversarial attacks.

摘要: 针对人脸识别(FR)系统的对抗性攻击已被证明在危害纯FR模型方面非常有效，然而，对抗性例子对于完整的FR系统可能是无效的，因为人脸反欺骗(Fas)模型经常被结合并且可以检测到大量这样的模型。为了解决这一未被充分开发的基本问题，我们提出了一种同时对FR和FAS模型进行敌意攻击的新设置，旨在增强对FR系统的敌意攻击的实用性。特别是，我们引入了一种新的攻击方法，即样式对齐分布偏差(SDB)，以提高对FR和FAS模型的黑盒攻击能力。具体地说，我们的SDB框架由三个关键组成部分组成。首先，为了提高FAS模型的可转移性，我们设计了一个基于分布感知的分数偏置模块，以优化敌方人脸样本，使其远离利用分数进行恶搞图像的分布。其次，为了缓解实况图像和用恶搞图像初始化的对抗性实例之间的本质差异，我们引入了实例样式对齐模块，将对抗性实例的样式与实况图像进行对齐。此外，为了缓解FR和FAS模型中梯度之间的冲突，我们提出了一个梯度一致性维护模块，利用Hessian近似来最小化梯度之间的差异。大量的实验表明，我们提出的攻击方法比最先进的对抗性攻击方法更具优越性。



## **35. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.16918v1) [paper-pdf](http://arxiv.org/pdf/2405.16918v1)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization but is also related to adversarial robustness, since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running it runs into a flat uncanny valley where the label remains flipped. We find this phenomenon across various model architectures and datasets. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, the adversarial examples rarely reach a truly flat region. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗性稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦度首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的诡异山谷，在那里标签仍然被翻转。我们在各种模型体系结构和数据集中发现了这种现象。我们的结果也推广到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健模型的低全局Lipschitz常数相结合的必要性。



## **36. Accelerating Greedy Coordinate Gradient via Probe Sampling**

通过探针采样加速贪婪坐标梯度 cs.CL

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2403.01251v2) [paper-pdf](http://arxiv.org/pdf/2403.01251v2)

**Authors**: Yiran Zhao, Wenyue Zheng, Tianle Cai, Xuan Long Do, Kenji Kawaguchi, Anirudh Goyal, Michael Shieh

**Abstract**: Safety of Large Language Models (LLMs) has become a critical issue given their rapid progresses. Greedy Coordinate Gradient (GCG) is shown to be effective in constructing adversarial prompts to break the aligned LLMs, but optimization of GCG is time-consuming. To reduce the time cost of GCG and enable more comprehensive studies of LLM safety, in this work, we study a new algorithm called $\texttt{Probe sampling}$. At the core of the algorithm is a mechanism that dynamically determines how similar a smaller draft model's predictions are to the target model's predictions for prompt candidates. When the target model is similar to the draft model, we rely heavily on the draft model to filter out a large number of potential prompt candidates. Probe sampling achieves up to $5.6$ times speedup using Llama2-7b-chat and leads to equal or improved attack success rate (ASR) on the AdvBench. Furthermore, probe sampling is also able to accelerate other prompt optimization techniques and adversarial methods, leading to acceleration of $1.8\times$ for AutoPrompt, $2.4\times$ for APE and $2.4\times$ for AutoDAN.

摘要: 随着大型语言模型的快速发展，其安全性已成为一个关键问题。贪婪坐标梯度(GCG)在构造敌意提示以打破排列的LLM方面是有效的，但GCG的优化是耗时的。为了减少GCG的时间开销，更全面地研究LLM的安全性，本文研究了一种新的算法--$\exttt{Probe Samples}$。该算法的核心是一种机制，该机制动态地确定较小的草稿模型的预测与目标模型对提示候选人的预测的相似性程度。当目标模型与选秀模型相似时，我们严重依赖选秀模型来筛选出大量潜在的提示候选者。使用Llama2-7b-Chat，探测采样可获得高达5.6美元的加速比，并可带来同等或更高的AdvBch攻击成功率(ASR)。此外，探针采样还能够加速其他即时优化技术和对抗方法，导致AutoPrompt、APE和AutoDAN的加速分别为1.8倍$、2.4倍$和2.4倍$。



## **37. Rethinking Independent Cross-Entropy Loss For Graph-Structured Data**

重新思考图结构数据的独立交叉熵损失 cs.LG

20 pages, 4 figures

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.15564v2) [paper-pdf](http://arxiv.org/pdf/2405.15564v2)

**Authors**: Rui Miao, Kaixiong Zhou, Yili Wang, Ninghao Liu, Ying Wang, Xin Wang

**Abstract**: Graph neural networks (GNNs) have exhibited prominent performance in learning graph-structured data. Considering node classification task, based on the i.i.d assumption among node labels, the traditional supervised learning simply sums up cross-entropy losses of the independent training nodes and applies the average loss to optimize GNNs' weights. But different from other data formats, the nodes are naturally connected. It is found that the independent distribution modeling of node labels restricts GNNs' capability to generalize over the entire graph and defend adversarial attacks. In this work, we propose a new framework, termed joint-cluster supervised learning, to model the joint distribution of each node with its corresponding cluster. We learn the joint distribution of node and cluster labels conditioned on their representations, and train GNNs with the obtained joint loss. In this way, the data-label reference signals extracted from the local cluster explicitly strengthen the discrimination ability on the target node. The extensive experiments demonstrate that our joint-cluster supervised learning can effectively bolster GNNs' node classification accuracy. Furthermore, being benefited from the reference signals which may be free from spiteful interference, our learning paradigm significantly protects the node classification from being affected by the adversarial attack.

摘要: 图神经网络(GNN)在学习图结构数据方面表现出了突出的性能。考虑到节点分类任务，传统的监督学习基于节点标签间的I.I.D假设，简单地总结独立训练节点的交叉熵损失，并利用平均损失来优化GNN的权值。但与其他数据格式不同的是，节点是自然相连的。研究发现，节点标签的独立分布建模限制了GNN在整个图上的泛化能力和防御敌意攻击的能力。在这项工作中，我们提出了一种新的框架，称为联合聚类监督学习，以建模每个节点与其对应的簇的联合分布。我们学习节点和簇标签在表示条件下的联合分布，并用获得的联合损失来训练GNN。这样，从本地簇中提取的数据标签参考信号明确地增强了对目标节点的区分能力。大量实验表明，我们的联合聚类监督学习能够有效地提高GNN的节点分类精度。此外，得益于参考信号可能不会受到恶意干扰，我们的学习范例显著地保护了节点分类不受恶意攻击的影响。



## **38. Adaptive Batch Normalization Networks for Adversarial Robustness**

对抗鲁棒性的自适应批量正规化网络 cs.LG

Accepted at IEEE International Conference on Advanced Video and  Signal-based Surveillance (AVSS) 2024

**SubmitDate**: 2024-05-27    [abs](http://arxiv.org/abs/2405.11708v2) [paper-pdf](http://arxiv.org/pdf/2405.11708v2)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstract**: Deep networks are vulnerable to adversarial examples. Adversarial Training (AT) has been a standard foundation of modern adversarial defense approaches due to its remarkable effectiveness. However, AT is extremely time-consuming, refraining it from wide deployment in practical applications. In this paper, we aim at a non-AT defense: How to design a defense method that gets rid of AT but is still robust against strong adversarial attacks? To answer this question, we resort to adaptive Batch Normalization (BN), inspired by the recent advances in test-time domain adaptation. We propose a novel defense accordingly, referred to as the Adaptive Batch Normalization Network (ABNN). ABNN employs a pre-trained substitute model to generate clean BN statistics and sends them to the target model. The target model is exclusively trained on clean data and learns to align the substitute model's BN statistics. Experimental results show that ABNN consistently improves adversarial robustness against both digital and physically realizable attacks on both image and video datasets. Furthermore, ABNN can achieve higher clean data performance and significantly lower training time complexity compared to AT-based approaches.

摘要: 深层网络很容易受到敌意例子的攻击。对抗训练(AT)因其显著的有效性而成为现代对抗防御方法的标准基础。然而，AT非常耗时，阻碍了它在实际应用中的广泛应用。在本文中，我们针对的是一种非AT防御：如何设计一种既能去除AT，又能对强对手攻击保持健壮性的防御方法？为了回答这个问题，我们求助于自适应批处理归一化(BN)，灵感来自于测试-时间域自适应的最新进展。因此，我们提出了一种新的防御方法，称为自适应批处理归一化网络(ABNN)。ABNN使用预先训练的替代模型来生成干净的BN统计数据，并将其发送到目标模型。目标模型专门接受关于干净数据的培训，并学习如何调整替代模型的BN统计数据。实验结果表明，ABNN在抵抗图像和视频数据集上的数字攻击和物理可实现攻击时，都一致地提高了对手的健壮性。此外，与基于AT的方法相比，ABNN可以获得更高的清洁数据性能和更低的训练时间复杂度。



## **39. Tradeoffs Between Alignment and Helpfulness in Language Models with Representation Engineering**

表示工程语言模型中的一致性和帮助性之间的权衡 cs.CL

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2401.16332v3) [paper-pdf](http://arxiv.org/pdf/2401.16332v3)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

摘要: 语言模型对齐已经成为人工智能安全的重要组成部分，通过增强期望的行为和抑制不期望的行为，允许人类和语言模型之间的安全交互。这通常通过调整模型或插入预设对齐提示来完成。最近，表征工程，一种通过在训练后改变模型表征来改变模型行为的方法，被证明在对齐LLM方面是有效的(Zou等人，2023a)。表征工程在对抗对抗性攻击和减少社会偏见等面向对齐的任务中产生收益，但也被证明导致模型执行基本任务的能力下降。在这篇文章中，我们研究了模型的一致性增加和有助性降低之间的权衡。我们提出了一个理论框架，提供了这两个量的界限，并从经验上证明了它们之间的相关性。首先，我们发现，在我们的框架条件下，可以用表示工程来保证对齐，但同时在这个过程中有助性受到了损害。其次，我们证明了有助性与表示工程向量的范数成二次曲线关系，而对齐则随其线性增加，这表明使用表示工程是有效的。我们通过实证验证了我们的发现，并绘制了表征工程对比对的有用性的界限。



## **40. Pruning for Robust Concept Erasing in Diffusion Models**

扩散模型中鲁棒概念擦除的修剪 cs.CV

Under review

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16534v1) [paper-pdf](http://arxiv.org/pdf/2405.16534v1)

**Authors**: Tianyun Yang, Juan Cao, Chang Xu

**Abstract**: Despite the impressive capabilities of generating images, text-to-image diffusion models are susceptible to producing undesirable outputs such as NSFW content and copyrighted artworks. To address this issue, recent studies have focused on fine-tuning model parameters to erase problematic concepts. However, existing methods exhibit a major flaw in robustness, as fine-tuned models often reproduce the undesirable outputs when faced with cleverly crafted prompts. This reveals a fundamental limitation in the current approaches and may raise risks for the deployment of diffusion models in the open world. To address this gap, we locate the concept-correlated neurons and find that these neurons show high sensitivity to adversarial prompts, thus could be deactivated when erasing and reactivated again under attacks. To improve the robustness, we introduce a new pruning-based strategy for concept erasing. Our method selectively prunes critical parameters associated with the concepts targeted for removal, thereby reducing the sensitivity of concept-related neurons. Our method can be easily integrated with existing concept-erasing techniques, offering a robust improvement against adversarial inputs. Experimental results show a significant enhancement in our model's ability to resist adversarial inputs, achieving nearly a 40% improvement in erasing the NSFW content and a 30% improvement in erasing artwork style.

摘要: 尽管生成图像的能力令人印象深刻，但文本到图像的扩散模型很容易产生不受欢迎的输出，如NSFW内容和受版权保护的艺术品。为了解决这个问题，最近的研究集中在微调模型参数以消除有问题的概念上。然而，现有的方法在稳健性方面存在重大缺陷，因为当面对巧妙地制作的提示时，微调的模型往往会重现不希望看到的输出。这揭示了当前方法的根本局限性，并可能增加在开放世界中部署扩散模型的风险。为了弥补这一差距，我们定位了与概念相关的神经元，发现这些神经元对对抗性提示表现出高度的敏感性，因此可以在擦除时被去激活，并在攻击下再次被激活。为了提高概念删除的健壮性，我们引入了一种新的基于剪枝的概念删除策略。我们的方法选择性地修剪与目标移除概念相关的关键参数，从而降低概念相关神经元的敏感度。我们的方法可以很容易地与现有的概念擦除技术相结合，提供了针对敌意输入的稳健改进。实验结果表明，我们的模型在抵抗敌意输入的能力方面有了显著的增强，在擦除NSFW内容方面提高了近40%，在擦除艺术风格方面提高了30%。



## **41. FLTrojan: Privacy Leakage Attacks against Federated Language Models Through Selective Weight Tampering**

FLTrojan：通过选择性权重篡改对联邦语言模型进行隐私泄露攻击 cs.CR

20 pages (including bibliography and Appendix), Submitted to ACM CCS  '24

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2310.16152v2) [paper-pdf](http://arxiv.org/pdf/2310.16152v2)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, determining the extent of privacy leakage in federated language models is challenging and not straightforward. Moreover, existing attacks aim to extract data regardless of how sensitive or naive it is. To fill this research gap, we introduce two novel findings with regard to leaking privacy-sensitive user data from federated large language models. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that privacy leakage can be aggravated by tampering with a model's selective weights that are specifically responsible for memorizing the sensitive training data. We show how a malicious client can leak the privacy-sensitive data of some other users in FL even without any cooperation from the server. Our best-performing method improves the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks with stronger assumptions of adversary capabilities.

摘要: 联合学习(FL)已经成为机器翻译、下一词预测和病历分析等各种语言建模应用中的关键组件。这些应用程序是在来自许多FL参与者的数据集上进行训练的，这些数据集通常包括隐私敏感数据，如医疗记录、电话/信用卡号码、登录凭据等。尽管FL可以在不需要客户共享其原始数据的情况下进行计算，但在联合语言模型中确定隐私泄漏的程度是具有挑战性的，而且不是直接的。此外，现有的攻击旨在提取数据，无论它是多么敏感或幼稚。为了填补这一研究空白，我们介绍了关于从联合大型语言模型泄露隐私敏感用户数据的两个新发现。首先，我们做了一个关键的观察，在FL的中间轮中的模型快照比最终训练的模型会导致更大的隐私泄露。其次，我们发现，通过篡改模型的选择性权重可能会加剧隐私泄露，这些选择性权重专门负责记忆敏感的训练数据。我们展示了恶意客户端如何在没有任何服务器合作的情况下泄露FL中其他用户的隐私敏感数据。该方法的成员关系推理召回率提高了29%，私有数据重构效率高达71%，明显优于对敌方能力假设更强的现有攻击。



## **42. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

29 pages

**SubmitDate**: 2024-05-26    [abs](http://arxiv.org/abs/2405.16405v1) [paper-pdf](http://arxiv.org/pdf/2405.16405v1)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **43. R.A.C.E.: Robust Adversarial Concept Erasure for Secure Text-to-Image Diffusion Model**

皇家海关：安全文本到图像扩散模型的鲁棒对抗概念擦除 cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16341v1) [paper-pdf](http://arxiv.org/pdf/2405.16341v1)

**Authors**: Changhoon Kim, Kyle Min, Yezhou Yang

**Abstract**: In the evolving landscape of text-to-image (T2I) diffusion models, the remarkable capability to generate high-quality images from textual descriptions faces challenges with the potential misuse of reproducing sensitive content. To address this critical issue, we introduce Robust Adversarial Concept Erase (RACE), a novel approach designed to mitigate these risks by enhancing the robustness of concept erasure method for T2I models. RACE utilizes a sophisticated adversarial training framework to identify and mitigate adversarial text embeddings, significantly reducing the Attack Success Rate (ASR). Impressively, RACE achieves a 30 percentage point reduction in ASR for the ``nudity'' concept against the leading white-box attack method. Our extensive evaluations demonstrate RACE's effectiveness in defending against both white-box and black-box attacks, marking a significant advancement in protecting T2I diffusion models from generating inappropriate or misleading imagery. This work underlines the essential need for proactive defense measures in adapting to the rapidly advancing field of adversarial challenges.

摘要: 在不断发展的文本到图像(T2I)扩散模型中，从文本描述生成高质量图像的非凡能力面临着潜在的误用，即复制敏感内容。为了解决这一关键问题，我们引入了稳健对抗性概念擦除(RACE)，这是一种新的方法，旨在通过增强T2I模型概念擦除方法的健壮性来降低这些风险。RACE利用复杂的对抗性训练框架来识别和缓解对抗性文本嵌入，显著降低攻击成功率(ASR)。令人印象深刻的是，与领先的白盒攻击方法相比，RACE实现了“裸体”概念的ASR降低30个百分点。我们广泛的评估证明了RACE在防御白盒和黑盒攻击方面的有效性，标志着在保护T2I扩散模型免受不适当或误导图像方面取得了重大进展。这项工作突出表明，必须采取积极主动的防御措施，以适应迅速发展的对抗性挑战领域。



## **44. Secure Hierarchical Federated Learning in Vehicular Networks Using Dynamic Client Selection and Anomaly Detection**

使用动态客户端选择和异常检测的车辆网络中的安全分层联邦学习 cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.17497v1) [paper-pdf](http://arxiv.org/pdf/2405.17497v1)

**Authors**: M. Saeid HaghighiFard, Sinem Coleri

**Abstract**: Hierarchical Federated Learning (HFL) faces the significant challenge of adversarial or unreliable vehicles in vehicular networks, which can compromise the model's integrity through misleading updates. Addressing this, our study introduces a novel framework that integrates dynamic vehicle selection and robust anomaly detection mechanisms, aiming to optimize participant selection and mitigate risks associated with malicious contributions. Our approach involves a comprehensive vehicle reliability assessment, considering historical accuracy, contribution frequency, and anomaly records. An anomaly detection algorithm is utilized to identify anomalous behavior by analyzing the cosine similarity of local or model parameters during the federated learning (FL) process. These anomaly records are then registered and combined with past performance for accuracy and contribution frequency to identify the most suitable vehicles for each learning round. Dynamic client selection and anomaly detection algorithms are deployed at different levels, including cluster heads (CHs), cluster members (CMs), and the Evolving Packet Core (EPC), to detect and filter out spurious updates. Through simulation-based performance evaluation, our proposed algorithm demonstrates remarkable resilience even under intense attack conditions. Even in the worst-case scenarios, it achieves convergence times at $63$\% as effective as those in scenarios without any attacks. Conversely, in scenarios without utilizing our proposed algorithm, there is a high likelihood of non-convergence in the FL process.

摘要: 分层联邦学习(HFL)面临着车载网络中敌意或不可靠车辆的巨大挑战，这可能会通过误导性的更新而损害模型的完整性。针对这一问题，我们的研究引入了一种新的框架，该框架集成了动态车辆选择和稳健的异常检测机制，旨在优化参与者选择并降低与恶意贡献相关的风险。我们的方法涉及全面的车辆可靠性评估，考虑了历史准确性、贡献频率和异常记录。通过分析联邦学习(FL)过程中局部或模型参数的余弦相似度，利用异常检测算法识别异常行为。这些异常记录然后被登记，并结合过去的准确性和贡献频率的表现，以确定最适合每一轮学习的工具。动态客户端选择和异常检测算法部署在不同的级别，包括簇头(CHS)、簇成员(CMS)和演进分组核心(EPC)，以检测和过滤虚假更新。通过基于仿真的性能评估，我们提出的算法即使在激烈的攻击条件下也表现出了显著的抗攻击能力。即使在最糟糕的情况下，它的收敛时间也只需63美元，与在没有任何攻击的情况下一样有效。相反，在没有使用我们提出的算法的情况下，FL过程中有很高的不收敛可能性。



## **45. Layer-Aware Analysis of Catastrophic Overfitting: Revealing the Pseudo-Robust Shortcut Dependency**

灾难性过度匹配的分层感知分析：揭示伪稳健的预设依赖 cs.LG

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16262v1) [paper-pdf](http://arxiv.org/pdf/2405.16262v1)

**Authors**: Runqi Lin, Chaojian Yu, Bo Han, Hang Su, Tongliang Liu

**Abstract**: Catastrophic overfitting (CO) presents a significant challenge in single-step adversarial training (AT), manifesting as highly distorted deep neural networks (DNNs) that are vulnerable to multi-step adversarial attacks. However, the underlying factors that lead to the distortion of decision boundaries remain unclear. In this work, we delve into the specific changes within different DNN layers and discover that during CO, the former layers are more susceptible, experiencing earlier and greater distortion, while the latter layers show relative insensitivity. Our analysis further reveals that this increased sensitivity in former layers stems from the formation of pseudo-robust shortcuts, which alone can impeccably defend against single-step adversarial attacks but bypass genuine-robust learning, resulting in distorted decision boundaries. Eliminating these shortcuts can partially restore robustness in DNNs from the CO state, thereby verifying that dependence on them triggers the occurrence of CO. This understanding motivates us to implement adaptive weight perturbations across different layers to hinder the generation of pseudo-robust shortcuts, consequently mitigating CO. Extensive experiments demonstrate that our proposed method, Layer-Aware Adversarial Weight Perturbation (LAP), can effectively prevent CO and further enhance robustness.

摘要: 灾难性过拟合(CO)是单步对抗训练(AT)中的一个重大挑战，表现为高度扭曲的深度神经网络(DNN)，容易受到多步对抗攻击。然而，导致决策边界扭曲的潜在因素仍然不清楚。在这项工作中，我们深入研究了不同DNN层内的具体变化，发现在CO过程中，前一层更容易受到影响，经历更早和更大的失真，而后一层表现出相对不敏感。我们的分析进一步表明，前几层敏感度的增加源于伪稳健捷径的形成，这些捷径可以无懈可击地防御单步对手攻击，但绕过了真正的稳健学习，导致决策边界扭曲。消除这些捷径可以从CO状态部分恢复DNN的稳健性，从而验证对它们的依赖是否触发了CO的发生。这种理解促使我们在不同的层上实现自适应的权重扰动，以阻止伪稳健捷径的生成，从而减少CO。大量实验表明，本文提出的层感知对抗性权重扰动(LAP)方法能够有效地防止CO，并进一步增强了鲁棒性。



## **46. Detecting Adversarial Data via Perturbation Forgery**

通过微扰伪造检测对抗数据 cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16226v1) [paper-pdf](http://arxiv.org/pdf/2405.16226v1)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Ping Li, Jiazhong Chen, Shijuan Huang, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, existing techniques either necessitate access to attack data before deploying a defense or incur a significant time cost for inference, rendering them impractical for defending against newly emerging attacks that are unseen by defenders. In this paper, we explore the proximity relationship between adversarial noise distributions and demonstrate the existence of an open covering for them. By learning to distinguish this open covering from the distribution of natural data, we can develop a detector with strong generalization capabilities against all types of adversarial attacks. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting unseen gradient-based, generative-model-based, and physical adversarial attacks, while remaining agnostic to any specific models. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.

摘要: 敌意检测是针对敌意攻击的一种防御策略，其目的是根据自然数据和敌意数据之间的分布和噪声模式的差异，从数据流中识别和过滤敌意数据。虽然以前的检测方法在检测基于梯度的敌意攻击方面取得了较高的性能，但基于非平衡和各向异性噪声模式的生成模型的新攻击可以逃避检测。更糟糕的是，现有技术要么需要在部署防御之前访问攻击数据，要么需要花费大量时间进行推断，这使得它们在防御防御者看不到的新出现的攻击方面不切实际。本文研究了对抗性噪声分布之间的邻近关系，并证明了它们的开覆盖的存在性。通过学习将这种开放覆盖与自然数据的分布区分开来，我们可以开发出一个具有强大的泛化能力的检测器，以抵御所有类型的对抗性攻击。基于这一观点，我们启发式地提出了扰动伪造，它包括噪声分布扰动、稀疏掩码生成和伪对抗数据生成，以训练一个能够检测基于不可见的基于梯度、基于生成模型的和物理的对抗攻击的对抗检测器，同时保持对任何特定模型的不可知性。在多个普通数据集和人脸数据集上进行的综合实验表明，该方法具有较强的泛化能力。



## **47. Enhancing Adversarial Transferability Through Neighborhood Conditional Sampling**

通过邻居条件抽样增强对抗性可转移性 cs.CV

Under review

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16181v1) [paper-pdf](http://arxiv.org/pdf/2405.16181v1)

**Authors**: Chunlin Qiu, Yiheng Duan, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples utilizing a white-box surrogate model to compromise various black-box target models, posing significant threats to many real-world applications. However, existing transfer attacks suffer from either weak transferability or expensive computation. To bridge the gap, we propose a novel sample-based attack, named neighborhood conditional sampling (NCS), which enjoys high transferability with lightweight computation. Inspired by the observation that flat maxima result in better transferability, NCS is formulated as a max-min bi-level optimization problem to seek adversarial regions with high expected adversarial loss and small standard deviations. Specifically, due to the inner minimization problem being computationally intensive to resolve, and affecting the overall transferability, we propose a momentum-based previous gradient inversion approximation (PGIA) method to effectively solve the inner problem without any computation cost. In addition, we prove that two newly proposed attacks, which achieve flat maxima for better transferability, are actually specific cases of NCS under particular conditions. Extensive experiments demonstrate that NCS efficiently generates highly transferable adversarial examples, surpassing the current best method in transferability while requiring only 50% of the computational cost. Additionally, NCS can be seamlessly integrated with other methods to further enhance transferability.

摘要: 基于传输的攻击利用白盒代理模型来创建敌意示例，以危害各种黑盒目标模型，对许多现实世界的应用程序构成重大威胁。然而，现有的传输攻击要么传输能力弱，要么计算量大。为了弥补这一差距，我们提出了一种新的基于样本的攻击，称为邻域条件抽样(NCS)，该攻击具有高可移植性和轻量级计算的优点。受平坦极大值具有更好的可转移性这一观察结果的启发，NCS被描述为一个最大-最小双层优化问题，以寻找具有较高预期对手损失和较小标准差的对抗性区域。具体地说，针对内极小化问题计算量大且影响整体可转移性的问题，提出了一种基于动量的先验梯度求逆近似(PGIA)方法，在不增加任何计算代价的情况下有效地解决了内极小化问题。此外，我们还证明了两个新提出的攻击在特定条件下实际上是NCS的特例，它们实现了平坦的极大值以获得更好的可转移性。大量的实验表明，NCS能有效地生成高度可移植的对抗性实例，在可移植性方面超过目前最好的方法，而所需的计算代价仅为50%。此外，NCS还可以与其他方法无缝集成，进一步增强可转移性。



## **48. Breaking the False Sense of Security in Backdoor Defense through Re-Activation Attack**

通过重新激活攻击打破后门防御中的虚假安全感 cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16134v1) [paper-pdf](http://arxiv.org/pdf/2405.16134v1)

**Authors**: Mingli Zhu, Siyuan Liang, Baoyuan Wu

**Abstract**: Deep neural networks face persistent challenges in defending against backdoor attacks, leading to an ongoing battle between attacks and defenses. While existing backdoor defense strategies have shown promising performance on reducing attack success rates, can we confidently claim that the backdoor threat has truly been eliminated from the model? To address it, we re-investigate the characteristics of the backdoored models after defense (denoted as defense models). Surprisingly, we find that the original backdoors still exist in defense models derived from existing post-training defense strategies, and the backdoor existence is measured by a novel metric called backdoor existence coefficient. It implies that the backdoors just lie dormant rather than being eliminated. To further verify this finding, we empirically show that these dormant backdoors can be easily re-activated during inference, by manipulating the original trigger with well-designed tiny perturbation using universal adversarial attack. More practically, we extend our backdoor reactivation to black-box scenario, where the defense model can only be queried by the adversary during inference, and develop two effective methods, i.e., query-based and transfer-based backdoor re-activation attacks. The effectiveness of the proposed methods are verified on both image classification and multimodal contrastive learning (i.e., CLIP) tasks. In conclusion, this work uncovers a critical vulnerability that has never been explored in existing defense strategies, emphasizing the urgency of designing more robust and advanced backdoor defense mechanisms in the future.

摘要: 深度神经网络在防御后门攻击方面面临持续的挑战，导致攻击和防御之间的持续战斗。虽然现有的后门防御策略在降低攻击成功率方面表现出了令人振奋的表现，但我们是否可以自信地声称，后门威胁已经真正从模型中消除了？为了解决这个问题，我们重新研究了防御后的回溯模型(记为防御模型)的特征。令人惊讶的是，我们发现在现有的训练后防御策略的防御模型中仍然存在原始的后门，并且后门的存在被称为后门存在系数来衡量。这意味着后门只是处于休眠状态，而不是被消除。为了进一步验证这一发现，我们的经验表明，这些休眠的后门可以很容易地在推理过程中重新激活，方法是使用通用的对抗性攻击，在精心设计的微小扰动下操纵原始触发器。在更实际的情况下，我们将后门重激活扩展到黑盒场景，其中防御模型在推理过程中只能被对手查询，并提出了两种有效的方法，即基于查询的后门重激活攻击和基于传输的后门重激活攻击。在图像分类和多通道对比学习(CLIP)任务上验证了所提方法的有效性。总而言之，这项工作揭示了一个在现有防御策略中从未探索过的关键漏洞，强调了未来设计更强大和更先进的后门防御机制的紧迫性。



## **49. Mitigating Backdoor Attack by Injecting Proactive Defensive Backdoor**

通过注入主动防御后门来缓解后门攻击 cs.CR

13 pages, 5 figures and 5 tables

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2405.16112v1) [paper-pdf](http://arxiv.org/pdf/2405.16112v1)

**Authors**: Shaokui Wei, Hongyuan Zha, Baoyuan Wu

**Abstract**: Data-poisoning backdoor attacks are serious security threats to machine learning models, where an adversary can manipulate the training dataset to inject backdoors into models. In this paper, we focus on in-training backdoor defense, aiming to train a clean model even when the dataset may be potentially poisoned. Unlike most existing methods that primarily detect and remove/unlearn suspicious samples to mitigate malicious backdoor attacks, we propose a novel defense approach called PDB (Proactive Defensive Backdoor). Specifically, PDB leverages the "home field" advantage of defenders by proactively injecting a defensive backdoor into the model during training. Taking advantage of controlling the training process, the defensive backdoor is designed to suppress the malicious backdoor effectively while remaining secret to attackers. In addition, we introduce a reversible mapping to determine the defensive target label. During inference, PDB embeds a defensive trigger in the inputs and reverses the model's prediction, suppressing malicious backdoor and ensuring the model's utility on the original task. Experimental results across various datasets and models demonstrate that our approach achieves state-of-the-art defense performance against a wide range of backdoor attacks.

摘要: 数据中毒后门攻击是对机器学习模型的严重安全威胁，攻击者可以操纵训练数据集向模型注入后门。在本文中，我们将重点放在训练中的后门防御上，目的是训练一个干净的模型，即使数据集可能被毒化。不同于现有的大多数方法主要是检测和删除/取消学习可疑样本来缓解恶意后门攻击，我们提出了一种称为主动防御后门的新防御方法。具体地说，PDB通过在训练期间主动向模型中注入防御性后门来利用后卫的“主场”优势。利用控制训练过程的优势，防御性后门被设计成在对攻击者保密的同时有效地抑制恶意后门。此外，我们还引入了一种可逆映射来确定防御目标标签。在推理过程中，PDB在输入中嵌入一个防御触发器，逆转模型的预测，抑制恶意后门，确保模型在原始任务上的实用性。在不同的数据集和模型上的实验结果表明，我们的方法在抵抗广泛的后门攻击时获得了最先进的防御性能。



## **50. Mitigating Dialogue Hallucination for Large Vision Language Models via Adversarial Instruction Tuning**

通过对抗性指令调优缓解大视野语言模型的对话幻觉 cs.CV

**SubmitDate**: 2024-05-25    [abs](http://arxiv.org/abs/2403.10492v2) [paper-pdf](http://arxiv.org/pdf/2403.10492v2)

**Authors**: Dongmin Park, Zhaofang Qian, Guangxing Han, Ser-Nam Lim

**Abstract**: Mitigating hallucinations of Large Vision Language Models,(LVLMs) is crucial to enhance their reliability for general-purpose assistants. This paper shows that such hallucinations of LVLMs can be significantly exacerbated by preceding user-system dialogues. To precisely measure this, we first present an evaluation benchmark by extending popular multi-modal benchmark datasets with prepended hallucinatory dialogues powered by our novel Adversarial Question Generator (AQG), which can automatically generate image-related yet adversarial dialogues by adopting adversarial attacks on LVLMs. On our benchmark, the zero-shot performance of state-of-the-art LVLMs drops significantly for both the VQA and Captioning tasks. Next, we further reveal this hallucination is mainly due to the prediction bias toward preceding dialogues rather than visual content. To reduce this bias, we propose Adversarial Instruction Tuning (AIT) that robustly fine-tunes LVLMs against hallucinatory dialogues. Extensive experiments show our proposed approach successfully reduces dialogue hallucination while maintaining performance.

摘要: 减轻大型视觉语言模型(LVLMS)的幻觉对于提高其对通用助理的可靠性至关重要。这篇论文表明，之前的用户-系统对话可以显著加剧LVLMS的这种幻觉。为了准确地衡量这一点，我们首先提出了一个评估基准，通过扩展流行的多模式基准数据集，在我们的新型对抗性问题生成器(AQG)的支持下，使用预先设定的幻觉对话，该生成器可以通过对LVLM进行对抗性攻击来自动生成与图像相关的对抗性对话。在我们的基准测试中，最先进的LVLMS在VQA和字幕任务中的零镜头性能都显著下降。接下来，我们进一步揭示这种幻觉主要是由于预测偏向于之前的对话而不是视觉内容。为了减少这种偏差，我们提出了对抗性指令调整(AIT)，它针对幻觉对话对LVLM进行强有力的微调。大量的实验表明，我们提出的方法在保持性能的同时成功地减少了对话幻觉。



