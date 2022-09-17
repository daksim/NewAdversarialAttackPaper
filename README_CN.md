# Latest Adversarial Attack Papers
**update at 2022-09-18 06:38:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

31 pages, 6 figures, fixed incorrect citation

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2205.01663v3)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用一个语言生成任务作为测试平台，通过对抗性训练来实现高可靠性。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们简单的“避免受伤”任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。在我们选择的阈值下，使用我们的基准分类器进行过滤可以将分发内数据的不安全完成率从大约2.4%降低到0.003%，这接近我们的测量能力极限。我们发现，对抗性训练显著提高了对我们训练的对抗性攻击的健壮性，而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **2. How to Attack and Defend NextG Radio Access Network Slicing with Reinforcement Learning**

基于强化学习的下一代无线接入网络分片攻防 cs.NI

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2101.05768v2)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek, M. Cenk Gursoy

**Abstracts**: In this paper, reinforcement learning (RL) for network slicing is considered in NextG radio access networks, where the base station (gNodeB) allocates resource blocks (RBs) to the requests of user equipments and aims to maximize the total reward of accepted requests over time. Based on adversarial machine learning, a novel over-the-air attack is introduced to manipulate the RL algorithm and disrupt NextG network slicing. The adversary observes the spectrum and builds its own RL based surrogate model that selects which RBs to jam subject to an energy budget with the objective of maximizing the number of failed requests due to jammed RBs. By jamming the RBs, the adversary reduces the RL algorithm's reward. As this reward is used as the input to update the RL algorithm, the performance does not recover even after the adversary stops jamming. This attack is evaluated in terms of both the recovery time and the (maximum and total) reward loss, and it is shown to be much more effective than benchmark (random and myopic) jamming attacks. Different reactive and proactive defense schemes (protecting the RL algorithm's updates or misleading the adversary's learning process) are introduced to show that it is viable to defend NextG network slicing against this attack.

摘要: 本文研究了下一代无线接入网络中网络切片的强化学习方法，其中基站(GNodeB)为用户设备的请求分配资源块(RB)，并以最大化接受请求的总回报为目标。在对抗性机器学习的基础上，引入了一种新的空中攻击来操纵RL算法并破坏NextG网络切片。敌手观察频谱并构建其自己的基于RL的代理模型，该代理模型选择在能量预算下阻塞哪个RB，目标是最大化由于阻塞的RB而失败的请求的数量。通过干扰RBS，对手降低了RL算法的奖励。由于该奖励被用作更新RL算法的输入，因此即使在对手停止干扰之后，性能也不会恢复。该攻击从恢复时间和(最大和总的)报酬损失两个方面进行评估，并且被证明比基准(随机和近视)干扰攻击要有效得多。不同的反应性和主动性防御方案(保护RL算法的更新或误导对手的学习过程)被引入，以表明防御NextG网络切片攻击是可行的。



## **3. A Light Recipe to Train Robust Vision Transformers**

培养健壮的视觉变形器的光明秘诀 cs.CV

Code available at https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07399v1)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstracts**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.

摘要: 在这篇文章中，我们问视觉转换器(VITS)是否可以作为一个底层架构来提高机器学习模型对逃避攻击的对抗性健壮性。虽然早期的工作集中在改进卷积神经网络上，但我们也表明VITS非常适合于对抗性训练，以获得竞争性的性能。我们通过对ImageNet数据集的子集进行严格的消融研究，发现了一种定制的对抗性训练配方，从而实现了这一目标。VITS的规范训练配方建议进行强大的数据增强，部分原因是为了弥补与卷曲相比，注意力模块缺乏视觉诱导偏差。我们表明，当用于对抗性训练时，这个配方达到了次优的性能。相反，我们发现，省略所有繁重的数据增强，并添加一些额外的技巧($varepsilon$-热身和更大的权重衰减)，显著提高了健壮VITS的性能。我们表明，我们的配方适用于不同类别的VIT体系结构和完整的ImageNet-1k上的大型模型。此外，研究了我们的模型健壮性的原因，我们表明，当使用我们的配方时，在训练期间更容易产生强攻击，这导致在测试时更好的健壮性。最后，我们进一步研究了对抗性训练的一个后果，提出了一种量化对抗性扰动的语义性质的方法，并强调了它与模型的稳健性的相关性。总体而言，我们建议社会应避免将VITS中的规范训练食谱转化为稳健的训练，并在对抗性训练的背景下重新考虑常见的训练选择。



## **4. Continuous Patrolling Games**

连续巡逻小游戏 cs.DM

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2008.07369v2)

**Authors**: Steve Alpern, Thuy Bui, Thomas Lidbetter, Katerina Papadaki

**Abstracts**: We study a patrolling game played on a network $Q$, considered as a metric space. The Attacker chooses a point of $Q$ (not necessarily a node) to attack during a chosen time interval of fixed duration. The Patroller chooses a unit speed path on $Q$ and intercepts the attack (and wins) if she visits the attacked point during the attack time interval. This zero-sum game models the problem of protecting roads or pipelines from an adversarial attack. The payoff to the maximizing Patroller is the probability that the attack is intercepted. Our results include the following: (i) a solution to the game for any network $Q$, as long as the time required to carry out the attack is sufficiently short, (ii) a solution to the game for all tree networks that satisfy a certain condition on their extremities, and (iii) a solution to the game for any attack duration for stars with one long arc and the remaining arcs equal in length. We present a conjecture on the solution of the game for arbitrary trees and establish it in certain cases.

摘要: 我们研究了在被认为是度量空间的网络$Q$上进行的巡逻对策。攻击者在选定的固定持续时间间隔内选择一个$Q$点(不一定是节点)进行攻击。巡护员在$Q$上选择一条单位速度路径，如果她在攻击时间间隔内访问被攻击点，则拦截攻击(并获胜)。这个零和博弈模拟了保护道路或管道免受对手攻击的问题。最大化巡逻的回报是攻击被拦截的概率。我们的结果包括：(I)任意网络$q$的对策解，只要进行攻击所需的时间足够短；(Ii)对于所有在其末端满足一定条件的树网络的对策解；(Iii)对于具有一条长弧线且其余弧长相等的恒星的任意攻击持续时间的对策解。我们对任意树的对策的解提出了一个猜想，并在某些情况下建立了它。



## **5. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

通过内部过度激活分析防御物理上可实现的敌意攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2203.07341v2)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.

摘要: 这项工作提出了一种健壮而有效的Z-MASK策略来提高卷积网络对物理可实现的敌意攻击的健壮性。所提出的防御依赖于对内部网络特征执行的特定Z-Score分析来检测和掩蔽输入图像中与敌对对象对应的像素。为此，在浅层和深层检查了空间上连续的激活，以暗示潜在的对抗性区域。然后，通过多门槛机制汇总这些建议。通过在语义分割和目标检测模型上进行的大量实验，对Z-MASK的有效性进行了评估。使用添加到输入图像的数字补丁和位于真实世界中的打印补丁来执行评估。实验结果表明，Z-MASK在检测准确率和网络整体性能方面均优于现有的方法。其他实验表明，Z-MASK对可能的防御感知攻击也具有很强的健壮性。



## **6. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

基于决策的基于补丁对抗性去除的视觉变形金刚黑盒攻击 cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2112.03492v2)

**Authors**: Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.

摘要: 与卷积神经网络(CNN)相比，视觉转换器(VITS)表现出了令人印象深刻的性能和更强的对抗鲁棒性。一方面，VITS对单个斑块之间全局交互的关注降低了图像的局部噪声敏感度。另一方面，现有的基于决策的攻击忽略了图像区域之间的噪声敏感度差异，进一步影响了噪声压缩的效率，特别是对VITS。因此，当目标模型只能被查询时，验证VITS的黑箱对抗健壮性仍然是一个具有挑战性的问题。本文从图像区域噪声敏感度差异的角度，从理论上分析了现有的基于决策的攻击方法的局限性，提出了一种新的基于决策的VITS黑盒攻击方法，称为Patch-Wise Aversarial Removal(PAR)。PAR通过从粗到精的搜索过程将图像分成多个块，并分别压缩每个块上的噪声。PAR记录每个面片的噪声大小和噪声敏感度，并选择查询值最高的面片进行噪声压缩。此外，PAR可以用作其他基于决策的攻击的噪声初始化方法，从而在不引入额外计算的情况下提高VITS和CNN的噪声压缩效率。在三个数据集上的大量实验表明，在相同的查询次数下，PAR获得了低得多的噪声幅度。



## **7. PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack**

PointACL：对抗性攻击下鲁棒点云表示的对抗性对比学习 cs.CV

arXiv admin note: text overlap with arXiv:2109.00179 by other authors

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06971v1)

**Authors**: Junxuan Huang, Yatong An, Lu cheng, Bai Chen, Junsong Yuan, Chunming Qiao

**Abstracts**: Despite recent success of self-supervised based contrastive learning model for 3D point clouds representation, the adversarial robustness of such pre-trained models raised concerns. Adversarial contrastive learning (ACL) is considered an effective way to improve the robustness of pre-trained models. In contrastive learning, the projector is considered an effective component for removing unnecessary feature information during contrastive pretraining and most ACL works also use contrastive loss with projected feature representations to generate adversarial examples in pretraining, while "unprojected " feature representations are used in generating adversarial inputs during inference.Because of the distribution gap between projected and "unprojected" features, their models are constrained of obtaining robust feature representations for downstream tasks. We introduce a new method to generate high-quality 3D adversarial examples for adversarial training by utilizing virtual adversarial loss with "unprojected" feature representations in contrastive learning framework. We present our robust aware loss function to train self-supervised contrastive learning framework adversarially. Furthermore, we find selecting high difference points with the Difference of Normal (DoN) operator as additional input for adversarial self-supervised contrastive learning can significantly improve the adversarial robustness of the pre-trained model. We validate our method, PointACL on downstream tasks, including 3D classification and 3D segmentation with multiple datasets. It obtains comparable robust accuracy over state-of-the-art contrastive adversarial learning methods.

摘要: 尽管最近基于自监督的对比学习模型在三维点云表示中取得了成功，但这种预训练模型的对抗性健壮性引起了人们的关注。对抗性对比学习被认为是提高预训练模型稳健性的有效方法。在对比学习中，投影器被认为是对比预训练中去除不必要特征信息的有效部件，大多数ACL工作在预训练中也使用对比损失和投影特征表征来生成对抗性样本，而在推理过程中则使用“非投影”特征表征来生成对抗性输入，由于投影和非投影特征之间的分布差距，它们的模型在获得下游任务的稳健特征表征方面受到限制。我们介绍了一种新的方法，通过在对比学习框架中利用虚拟对抗性损失和非投影的特征表示来生成用于对抗性训练的高质量3D对抗性实例。我们提出了稳健的意识损失函数来对抗性地训练自监督对比学习框架。此外，我们发现选择高差点和正态(DON)算子的差作为对抗性自监督对比学习的附加输入可以显著提高预训练模型的对抗性健壮性。我们在下游任务上验证了我们的方法，PointACL，包括3D分类和多个数据集的3D分割。与最先进的对比对抗性学习方法相比，它获得了相当的鲁棒性精度。



## **8. Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering**

基于关联信息瓶颈的视觉问答精调算法 cs.CV

20 pages, 4 figures, 13 tables

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06954v1)

**Authors**: Jingjing Jiang, Ziyi Liu, Nanning Zheng

**Abstracts**: Benefiting from large-scale Pretrained Vision-Language Models (VL-PMs), the performance of Visual Question Answering (VQA) has started to approach human oracle performance. However, finetuning large-scale VL-PMs with limited data for VQA usually faces overfitting and poor generalization issues, leading to a lack of robustness. In this paper, we aim to improve the robustness of VQA systems (ie, the ability of the systems to defend against input variations and human-adversarial attacks) from the perspective of Information Bottleneck when finetuning VL-PMs for VQA. Generally, internal representations obtained by VL-PMs inevitably contain irrelevant and redundant information for the downstream VQA task, resulting in statistically spurious correlations and insensitivity to input variations. To encourage representations to converge to a minimal sufficient statistic in vision-language learning, we propose the Correlation Information Bottleneck (CIB) principle, which seeks a tradeoff between representation compression and redundancy by minimizing the mutual information (MI) between the inputs and internal representations while maximizing the MI between the outputs and the representations. Meanwhile, CIB measures the internal correlations among visual and linguistic inputs and representations by a symmetrized joint MI estimation. Extensive experiments on five VQA benchmarks of input robustness and two VQA benchmarks of human-adversarial robustness demonstrate the effectiveness and superiority of the proposed CIB in improving the robustness of VQA systems.

摘要: 得益于大规模预训练视觉语言模型(VL-PM)，视觉问答(VQA)的性能已经开始接近人类的预言性能。然而，在VQA数据有限的情况下，对大规模的VL-PM进行精调通常会面临过拟合和泛化能力差的问题，从而导致缺乏健壮性。本文从信息瓶颈的角度对VQA的VL-PM进行优化，旨在提高VQA系统的健壮性(即系统抵抗输入变化和人类攻击的能力)。通常，VL-PM得到的内部表示不可避免地包含与下游VQA任务无关和冗余的信息，导致统计上虚假的相关性和对输入变化的不敏感。为了在视觉语言学习中鼓励表征收敛到最小的充分统计量，我们提出了关联信息瓶颈(CIB)原则，该原则通过最小化输入和内部表征之间的互信息(MI)同时最大化输出和表征之间的MI来寻求表征压缩和冗余之间的折衷。同时，CIB通过对称化的联合MI估计来衡量视觉输入和语言输入与表征之间的内在关联。在5个输入健壮性VQA基准和2个人-对手健壮性VQA基准上的大量实验证明了所提出的CIB在提高VQA系统稳健性方面的有效性和优越性。



## **9. On the interplay of adversarial robustness and architecture components: patches, convolution and attention**

关于对抗性健壮性和体系结构组件的相互作用：补丁、卷积和注意力 cs.CV

Presented at the "New Frontiers in Adversarial Machine Learning"  Workshop at ICML 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06953v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. An ablation from ResNet to ConvNeXt reveals key architectural changes leading to almost $10\%$ higher $\ell_\infty$-robustness.

摘要: 近年来，从变压器中使用的注意和补丁开始，开发了用于图像分类的新的体系结构组件。虽然以前的工作已经分析了体系结构组件的某些方面对对抗攻击的健壮性的影响，特别是对于视觉转换器，但对主要因素的理解仍然有限。我们比较了几种不同结构的(非)稳健分类器，并研究了它们的性质，包括对抗性训练对学习特征的可解释性的影响以及对不可见威胁模型的稳健性。从ResNet到ConvNeXt的消融揭示了导致$10\$更高的$\ell_\inty$-健壮性的关键架构变化。



## **10. Robust Constrained Reinforcement Learning**

稳健的约束强化学习 cs.LG

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06866v1)

**Authors**: Yue Wang, Fei Miao, Shaofeng Zou

**Abstracts**: Constrained reinforcement learning is to maximize the expected reward subject to constraints on utilities/costs. However, the training environment may not be the same as the test one, due to, e.g., modeling error, adversarial attack, non-stationarity, resulting in severe performance degradation and more importantly constraint violation. We propose a framework of robust constrained reinforcement learning under model uncertainty, where the MDP is not fixed but lies in some uncertainty set, the goal is to guarantee that constraints on utilities/costs are satisfied for all MDPs in the uncertainty set, and to maximize the worst-case reward performance over the uncertainty set. We design a robust primal-dual approach, and further theoretically develop guarantee on its convergence, complexity and robust feasibility. We then investigate a concrete example of $\delta$-contamination uncertainty set, design an online and model-free algorithm and theoretically characterize its sample complexity.

摘要: 约束强化学习是在效用/成本约束下最大化期望收益的学习方法。然而，由于例如建模错误、对抗性攻击、非平稳性等原因，训练环境可能与测试环境不同，从而导致严重的性能降级，更重要的是违反约束。我们提出了一种模型不确定性下的鲁棒约束强化学习框架，其中MDP不是固定的，而是位于某个不确定集合中，目标是保证不确定集合中的所有MDP都满足效用/成本约束，并且最大化不确定集合上的最坏情况下的回报性能。我们设计了一种稳健的原始-对偶方法，并从理论上对其收敛、复杂性和稳健可行性进行了保证。然后，我们研究了一个具体的例子--污染不确定集，设计了一个在线的、无模型的算法，并从理论上刻画了其样本复杂性。



## **11. Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models**

神经排序模型对单词替换排序攻击的验证稳健性 cs.IR

Accepted by CIKM2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06691v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yixing Fan, Maarten de Rijke, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have achieved promising results in information retrieval. NRMs have also been shown to be vulnerable to adversarial examples. A typical Word Substitution Ranking Attack (WSRA) against NRMs was proposed recently, in which an attacker promotes a target document in rankings by adding human-imperceptible perturbations to its text. This raises concerns when deploying NRMs in real-world applications. Therefore, it is important to develop techniques that defend against such attacks for NRMs. In empirical defenses adversarial examples are found during training and used to augment the training set. However, such methods offer no theoretical guarantee on the models' robustness and may eventually be broken by other sophisticated WSRAs. To escape this arms race, rigorous and provable certified defense methods for NRMs are needed.   To this end, we first define the \textit{Certified Top-$K$ Robustness} for ranking models since users mainly care about the top ranked results in real-world scenarios. A ranking model is said to be Certified Top-$K$ Robust on a ranked list when it is guaranteed to keep documents that are out of the top $K$ away from the top $K$ under any attack. Then, we introduce a Certified Defense method, named CertDR, to achieve certified top-$K$ robustness against WSRA, based on the idea of randomized smoothing. Specifically, we first construct a smoothed ranker by applying random word substitutions on the documents, and then leverage the ranking property jointly with the statistical property of the ensemble to provably certify top-$K$ robustness. Extensive experiments on two representative web search datasets demonstrate that CertDR can significantly outperform state-of-the-art empirical defense methods for ranking models.

摘要: 神经排序模型(NRM)在信息检索中取得了良好的效果。NRM也被证明容易受到敌意例子的影响。最近提出了一种针对NRMS的典型单词替换排名攻击(WSRA)，攻击者通过在文本中添加人类无法察觉的扰动来提升目标文档的排名。在实际应用程序中部署NRM时，这会引起人们的担忧。因此，开发针对NRM的防御此类攻击的技术非常重要。在经验防御中，对抗性的例子是在训练期间发现的，并被用来增加训练集。然而，这些方法对模型的稳健性没有提供理论上的保证，最终可能会被其他复杂的WSRA所打破。为了避免这场军备竞赛，需要针对NRM的严格和可证明的认证防御方法。为此，由于用户主要关心真实场景中排名靠前的结果，因此我们首先定义了模型排名的\textit{认证排名-$K$健壮性}。当保证在任何攻击下使排名前$K$之外的文档远离排名前$K$的文档时，排名模型在排名列表上被称为认证的Top-$K$健壮。然后，基于随机平滑的思想，提出了一种认证防御方法CertDR，以实现对WSRA的认证TOP-K$健壮性。具体地说，我们首先通过对文档进行随机单词替换来构造平滑的排序器，然后利用排序属性和集成的统计属性来证明top-$K$的稳健性。在两个具有代表性的网络搜索数据集上的大量实验表明，CertDR在排序模型方面的性能明显优于最先进的经验防御方法。



## **12. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

有序-无序：对黑盒神经网络排序模型的模仿敌意攻击 cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06506v1)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstracts**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.

摘要: 神经文本排序模型已经取得了显著的进步，并越来越多地应用于实践中。不幸的是，它们也继承了一般神经模型的对抗性漏洞，这些漏洞已经被检测到，但仍未被先前的研究充分探索。此外，BlackHat SEO可能会利用继承的敌意漏洞来击败保护更好的搜索引擎。在这项研究中，我们提出了一种对黑盒神经通路排序模型的模仿对抗性攻击。我们首先证明了目标文章排序模型可以通过列举关键查询/候选来透明化和模仿，然后训练一个排序模仿模型。利用排序模拟模型，可以对排序结果进行精细的操作，并将操纵攻击转移到目标排序模型。为此，我们提出了一种创新的基于梯度的攻击方法，该方法在两两目标函数的授权下，生成对抗性触发器，以极少的令牌造成有预谋的无序。为了装备触发伪装，我们在目标函数中加入了下一句预测损失和语言模型流畅度约束。文章排序的实验结果证明了该排序模仿攻击模型和敌意触发器对各种SOTA神经排序模型的有效性。此外，各种缓解分析和人类评估表明，当面临潜在的缓解方法时，伪装是有效的。为了激励其他学者进一步研究这一新颖而重要的问题，我们公开了实验数据和代码。



## **13. Targeting interventions for displacement minimization in opinion dynamics**

意见动力学中位移最小化的定向干预 cs.SI

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06481v1)

**Authors**: Luca Damonte, Giacomo Como, Fabio Fagnani

**Abstracts**: Social influence is largely recognized as a key factor in opinion formation processes. Recently, the role of external forces in inducing opinion displacement and polarization in social networks has attracted significant attention. This is in particular motivated by the necessity to understand and possibly prevent interference phenomena during political campaigns and elections. In this paper, we formulate and solve a targeted intervention problem for opinion displacement minimization on a social network. Specifically, we consider a min-max problem whereby a social planner (the defender) aims at selecting the optimal network intervention within her given budget constraint in order to minimize the opinion displacement in the system that an adversary (the attacker) is instead trying to maximize. Our results show that the optimal intervention of the defender has two regimes. For large enough budget, the optimal intervention of the social planner acts on all nodes proportionally to a new notion of network centrality. For lower budget values, such optimal intervention has a more delicate structure and is rather concentrated on a few target individuals.

摘要: 社会影响力在很大程度上被认为是舆论形成过程中的一个关键因素。最近，外力在社交网络中引发意见转移和两极分化的作用引起了人们的极大关注。这尤其是因为有必要了解并可能防止政治竞选和选举期间的干扰现象。在这篇文章中，我们制定并解决了一个针对社会网络上的意见移位最小化的定向干预问题。具体地说，我们考虑了一个最小-最大问题，在这个问题中，社会规划者(防御者)的目标是在她给定的预算限制内选择最优的网络干预，以便最小化系统中对手(攻击者)试图最大化的意见位移。我们的结果表明，防御者的最优干预有两种机制。对于足够大的预算，社会规划者的最优干预按照网络中心性的新概念按比例作用于所有节点。对于较低的预算价值，这种最佳干预具有更微妙的结构，并且相当集中在少数目标个人身上。



## **14. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

私人眼睛：视频会议中通过眼镜反射窥视文本屏幕的限度 cs.CR

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2205.03971v2)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.

摘要: 通过数学建模和人体实验，这项研究探索了新兴的网络摄像头可能在多大程度上泄露从网络摄像头捕获的眼镜反射中闪烁的可识别的文本和图形信息。我们工作的主要目标是测量、计算和预测随着未来网络摄像头技术的发展而产生的可识别性的因素、限制和阈值。我们的工作利用视频帧序列上的多帧超分辨率技术，探索和表征了基于光学攻击的可行威胁模型。我们的模型和在受控实验室环境下的实验结果表明，使用720p网络摄像头可以重建和识别高度低至10 mm的屏幕文本，准确率超过75%。我们进一步将该威胁模型应用于具有不同攻击者能力的Web文本内容，以找出文本变得可识别的阈值。我们对20名参与者的用户研究表明，目前的720p网络摄像头足以让对手在大字体网站上重建文本内容。我们的模型进一步表明，向4K摄像头的演变将使文本泄漏的门槛倾斜到重建流行网站上的大多数标题文本。除了文本目标，在使用720P网络摄像头识别Alexa前100名网站的封闭世界数据集上的案例研究显示，即使不使用机器学习模型，在10个参与者的情况下，最高识别准确率也达到94%。我们的研究提出了近期的缓解措施，包括一个软件原型，用户可以使用它来模糊他们视频流的眼镜区域。对于可能的长期防御，我们主张使用个人反射测试程序来评估各种环境下的威胁，并证明在隐私敏感场景中遵循最小特权原则的重要性。



## **15. TSFool: Crafting High-quality Adversarial Time Series through Multi-objective Optimization to Fool Recurrent Neural Network Classifiers**

TSFool：基于多目标优化的递归神经网络分类器生成高质量的对抗性时间序列 cs.LG

9 pages, 5 figures

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06388v1)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstracts**: Deep neural network (DNN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks have achieved good performance in feed-forward model and image recognition tasks, the extension for time series classification in the recurrent neural network (RNN) remains a dilemma, because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity to perturbations of time series data challenges the traditional local optimization objective to minimize perturbation. In this paper, an efficient and widely applicable approach called TSFool for crafting high-quality adversarial time series for the RNN classifier is proposed. We propose a novel global optimization objective named Camouflage Coefficient to consider how well the adversarial samples hide in class clusters, and accordingly redefine the high-quality adversarial attack as a multi-objective optimization problem. We also propose a new idea to use intervalized weighted finite automata (IWFA) to capture deeply embedded vulnerable samples having otherness between features and latent manifold to guide the approximation to the optimization solution. Experiments on 22 UCR datasets are conducted to confirm that TSFool is a widely effective, efficient and high-quality approach with 93.22% less local perturbation, 32.33% better global camouflage, and 1.12 times speedup to existing methods.

摘要: 深度神经网络(DNN)分类器容易受到敌意攻击。尽管现有的基于梯度的攻击在前馈模型和图像识别任务中取得了良好的性能，但由于递归神经网络的周期性结构阻止了直接的模型区分，以及对时间序列数据扰动的视觉敏感性挑战了传统的局部优化目标以最小化扰动，因此递归神经网络对时间序列分类的扩展仍然是一个两难问题。本文提出了一种用于为RNN分类器生成高质量对抗性时间序列的方法TSFool，该方法具有较高的效率和广泛的适用性。我们提出了一种新的全局优化目标伪装系数来考虑敌方样本在类簇中的隐藏程度，从而将高质量的敌方攻击重新定义为一个多目标优化问题。我们还提出了一种新的思想，使用区间加权有限自动机(IWFA)来捕获特征和潜在流形之间存在差异的深度嵌入的易受攻击样本，以指导逼近到最优解。在22个UCR数据集上的实验表明，TSFool是一种广泛有效、高效和高质量的方法，局部扰动减少了93.22%，全局伪装效果提高了32.33%，是现有方法的1.12倍。



## **16. PINCH: An Adversarial Extraction Attack Framework for Deep Learning Models**

PINCH：一种面向深度学习模型的对抗性抽取攻击框架 cs.CR

15 pages, 11 figures, 2 tables

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.06300v1)

**Authors**: William Hackett, Stefan Trawicki, Zhengxin Yu, Neeraj Suri, Peter Garraghan

**Abstracts**: Deep Learning (DL) models increasingly power a diversity of applications. Unfortunately, this pervasiveness also makes them attractive targets for extraction attacks which can steal the architecture, parameters, and hyper-parameters of a targeted DL model. Existing extraction attack studies have observed varying levels of attack success for different DL models and datasets, yet the underlying cause(s) behind their susceptibility often remain unclear. Ascertaining such root-cause weaknesses would help facilitate secure DL systems, though this requires studying extraction attacks in a wide variety of scenarios to identify commonalities across attack success and DL characteristics. The overwhelmingly high technical effort and time required to understand, implement, and evaluate even a single attack makes it infeasible to explore the large number of unique extraction attack scenarios in existence, with current frameworks typically designed to only operate for specific attack types, datasets and hardware platforms. In this paper we present PINCH: an efficient and automated extraction attack framework capable of deploying and evaluating multiple DL models and attacks across heterogeneous hardware platforms. We demonstrate the effectiveness of PINCH by empirically evaluating a large number of previously unexplored extraction attack scenarios, as well as secondary attack staging. Our key findings show that 1) multiple characteristics affect extraction attack success spanning DL model architecture, dataset complexity, hardware, attack type, and 2) partially successful extraction attacks significantly enhance the success of further adversarial attack staging.

摘要: 深度学习(DL)模型越来越支持各种应用程序。不幸的是，这种普及性也使它们成为提取攻击的有吸引力的目标，这些攻击可以窃取目标DL模型的体系结构、参数和超参数。现有的提取攻击研究已经观察到不同的DL模型和数据集的攻击成功程度不同，但其易感性背后的潜在原因往往尚不清楚。查明此类根本原因弱点将有助于促进数字图书馆系统的安全，尽管这需要研究各种情况下的提取攻击，以确定攻击成功和数字图书馆特征之间的共性。理解、实施和评估单个攻击所需的极高的技术工作量和时间，使得探索现有的大量独特的提取攻击场景变得不可行，目前的框架通常设计为仅针对特定攻击类型、数据集和硬件平台运行。在本文中，我们提出了PINCH：一个高效的自动提取攻击框架，能够部署和评估跨不同硬件平台的多个DL模型和攻击。我们通过对大量以前未探索的提取攻击场景以及二次攻击阶段的经验评估，证明了PIPCH的有效性。我们的主要发现表明，1)多个特征影响跨DL模型架构、数据集复杂性、硬件、攻击类型的提取攻击成功；2)部分成功的提取攻击显著提高了进一步对抗性攻击的成功率。



## **17. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

基于语义分割的对抗性补丁攻击认证防御 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.

摘要: 对抗性补丁攻击是现实世界深度学习应用面临的一种新的安全威胁。我们提出了去任务平滑，这是第一种(据我们所知)来证明语义分割模型对这种威胁模型的稳健性的方法。以前关于可证明防御补丁攻击的工作主要集中在图像分类任务上，并且经常需要改变模型体系结构和额外的训练，这是不受欢迎的，并且计算代价高昂。在去任务平滑中，任何分割模型都可以在没有特定训练、微调或体系结构限制的情况下应用。使用不同的掩码策略，去掩码平滑可以应用于认证检测和认证恢复。在ADE20K数据集上的大量实验中，对于检测任务中1%的块，去任务平滑平均可以保证64%的像素预测，对于恢复任务，对于0.5%的块，去任务平滑平均可以保证48%的像素预测。



## **18. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

对抗性组间链路注入降低了图神经网络的公平性 cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.

摘要: 我们提出了针对图神经网络(GNN)的对抗性攻击的存在和有效性的证据，这些攻击旨在降低公平性。这些攻击可能使基于GNN的节点分类中的特定节点子组处于不利地位，其中底层网络的节点具有敏感属性，如种族或性别。我们进行了定性和实验分析，解释了敌意链接注入如何损害GNN预测的公平性。例如，攻击者可以通过在属于相反子组和相反类标签的节点之间注入敌对链接来损害基于GNN的节点分类的公平性。我们在经验数据集上的实验表明，对抗性公平攻击能够以较低的扰动率(攻击是有效的)显著降低GNN预测的公平性(攻击是有效的)，并且不会显著降低准确率(攻击是欺骗性的)。这项工作证明了GNN模型对敌意公平攻击的脆弱性。我们希望我们的发现提高我们社区对这个问题的认识，并为未来开发更稳健地抵御此类攻击的GNN模型奠定基础。



## **19. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

一种进化、无梯度、查询高效的深层网络对抗性实例生成黑盒算法 cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.

摘要: 深度神经网络(DNN)对各种场景中的敌意数据很敏感，包括黑盒场景，在这种场景中，攻击者只被允许查询训练的模型并接收输出。现有的创建对抗性实例的黑盒方法成本很高，通常使用梯度估计或训练替换网络。本文介绍了一种非常有效的无目标、基于分数的黑盒攻击查询攻击基于一种新的目标函数，可用于无梯度优化问题。攻击只需要访问分类器的输出逻辑，因此不受梯度掩蔽的影响。不需要额外的信息，使我们的方法更适合实际情况。我们使用三个不同的最先进的模型--先启-v3、ResNet-50和VGG-16-BN--针对三个基准数据集：MNIST、CIFAR10和ImageNet测试其性能。此外，我们还评估了查询攻击在非差分变换防御和现有健壮性模型上的性能。实验结果表明，查询攻击在准确率和查询效率方面都具有较好的性能。



## **20. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **21. Adversarial Coreset Selection for Efficient Robust Training**

用于高效稳健训练的对抗性同位重置选择 cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.

摘要: 神经网络很容易受到敌意攻击：在它们的输入中添加精心设计的、不可察觉的扰动可以修改它们的输出。对抗性训练是训练抵抗此类攻击的稳健模型的最有效方法之一。遗憾的是，这种方法比普通的神经网络训练要慢得多，因为它需要在每次迭代中为整个训练数据构造对抗性样本。通过利用核心选择理论，我们展示了如何选择一小部分训练数据提供了一种原则性的方法来降低稳健训练的时间复杂性。为此，我们首先为对抗性核心重置选择提供收敛保证。特别是，我们证明了收敛界与我们的核集在整个训练数据上计算的梯度的逼近程度直接相关。在理论分析的基础上，我们提出了利用这一梯度逼近误差作为对抗性核心集选择的目标，以有效地减少训练集的规模。一旦构建，我们就对训练数据的这个子集进行对抗性训练。与现有方法不同，我们的方法可以适应广泛的训练目标，包括交易、$\ell_p$-PGD和感知对手训练。我们进行了大量的实验，证明了我们的方法将对手训练速度提高了2-3倍，同时经历了干净和健壮的准确性的轻微下降。



## **22. Adaptive Perturbation Generation for Multiple Backdoors Detection**

多后门检测的自适应扰动生成 cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).

摘要: 大量证据表明，深度神经网络(DNN)很容易受到后门攻击，这促使了后门检测方法的发展。现有的后门检测方法通常是为个别特定类型的后门攻击量身定做的(例如，基于补丁或基于扰动)。然而，攻击者在实践中可能会产生多种类型的后门攻击，这对现有的检测策略提出了挑战。基于敌意扰动与触发模式高度相关的事实，提出了自适应扰动生成(APG)框架，通过自适应注入敌意扰动来检测多种类型的后门攻击。由于不同的触发模式在相同的对抗性扰动下表现出高度不同的行为，我们首先设计了全局到局部的策略，通过调整攻击的区域和预算来适应多种类型的后门触发。为了进一步提高扰动注入的效率，我们引入了一种梯度引导的掩码生成策略来搜索对抗性攻击的最优区域。在多个数据集(CIFAR-10，GTSRB，Tiny-ImageNet)上进行的大量实验表明，我们的方法比最先进的基线方法有很大的优势(+12%)。



## **23. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

HodgeRank和谱方法的故事：针对等级聚集的目标攻击是对抗性游戏的固定点 cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.

摘要: 在选举、体育竞赛、推荐和信息检索等领域，采用配对比较的排名聚合方法已显示出良好的效果。然而，与大量关于算法的计算和统计特性的研究工作相比，对这类算法的安全问题关注较少。在巨额利润的驱动下，潜在对手操纵排行榜的动机和动机很强。同时，文献中对秩聚类方法的内在脆弱性的研究还不够深入。为了充分理解可能的风险，我们将重点放在有目的的对手身上，他们希望通过修改成对数据来指定聚合结果。从动力系统的角度来看，具有目标排行榜的攻击行为是属于对手和受害者组成的固定点。为了进行有针对性的攻击，我们将对手和受害者之间的相互作用描述为一个由两个连续算子组成的博弈论框架，同时建立了纳什均衡。然后构造了针对HodgeRank和RankCentrality的两个过程来产生对原始数据的修改。此外，我们证明了一旦对手掌握了完整的信息，受害者就会产生目标排名表。值得注意的是，所提出的方法只允许对手持有不完全信息或不完全反馈，并执行有目的的攻击。通过一系列玩具仿真和几个真实世界的数据实验，验证了所提出的目标攻击策略的有效性。实验结果表明，所提出的方法能够达到攻击者的目的，即扰动排序列表的领先候选者就是对手指定的候选者。



## **24. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

基于UCB的最佳武器识别策略下对抗性攻击的样本复杂性 cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.

摘要: 在这项工作中，我研究了多臂强盗(MAB)环境下的对抗性报酬摄动问题。具体地说，我将重点放在对应用于随机MAB的UCB类型的最佳ARM识别策略的对抗性攻击上。文[1]中提出的UCB攻击导致经常拉动目标手臂K。我使用了[1]的攻击模型来推导出选择目标手臂K作为最佳手臂所需的样本复杂度。证明了文[2]中给出的基于UCB的最佳手臂识别算法的停止条件可以由目标手臂K在T轮中实现，其中T仅取决于手臂的总数和手臂的$\sigma^2-$亚高斯随机奖励的$\sigma参数。



## **25. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

基于重放的自主机器人对传感器欺骗攻击的恢复 cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.

摘要: 传感器对于机器人车辆(RV)的自主操作至关重要。对传感器的物理攻击，如传感器篡改或欺骗，可能会通过物理通道向房车提供错误的值，从而导致任务失败。在本文中，我们提出了DeLorean，一个全面的诊断和恢复框架，用于保护自主房车免受物理攻击。我们考虑了一种称为传感器欺骗攻击(SDA)的强物理攻击形式，在这种攻击中，对手同时针对不同类型的多个传感器(甚至包括所有传感器)。在SDAS下，DeLorean检查攻击导致的错误，识别目标传感器，并防止错误的传感器输入用于房车的反馈控制回路。DeLorean在反馈控制环路中重放历史状态信息，并恢复RV免受攻击。我们对四辆真实房车和两辆模拟房车的评估表明，DeLorean可以从不同的攻击中恢复房车，并确保94%的任务成功(平均而言)，而不会发生任何崩溃。DeLorean的性能、内存和电池开销都很低。



## **26. Boosting Robustness Verification of Semantic Feature Neighborhoods**

增强语义特征邻域的健壮性验证 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.

摘要: 深度神经网络已被证明容易受到基于语义特征的输入扰乱的对抗性攻击。现有的健壮性分析器可以对语义特征邻域进行推理，以增加网络的可靠性。然而，尽管这些技术取得了重大进展，它们仍难以扩展到深度网络和大型社区。在这项工作中，我们引入了VeEP，这是一种主动学习方法，它将验证过程划分为一系列较小的验证步骤，每个步骤都提交给现有的健壮性分析器。其关键思想是建立在先前步骤的基础上，以预测下一个最佳步骤。通过参数回归估计认证速度和灵敏度，预测最优步骤。我们在MNIST、Fashion-MNIST、CIFAR-10和ImageNet上对Veep进行了评估，结果表明，它可以分析各种特征的社区：亮度、对比度、色调、饱和度和亮度。我们发现，在平均90分钟的超时时间内，Veep在29分钟内验证了96%的最大可证明邻域，而现有的分割方法平均在58分钟内验证了73%的最大可证明邻域。



## **27. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

低水平收缩的两层优化：无热启动的最优样本复杂性 stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates

摘要: 我们分析了一类一般的两层问题，其中上层问题在于光滑目标函数的最小化，下层问题是寻找光滑压缩映射的不动点。这类问题包括元学习、均衡模型、超参数优化和数据中毒攻击等实例。最近的一些工作提出了暖启动下层问题的算法，即使用先前的下层近似解作为下层求解器的起始点。这种热启动过程允许人们在随机和确定性设置下改善样本复杂性，在某些情况下实现顺序最优的样本复杂性。然而，在一些情况下，例如元学习和平衡模型，热启动程序不是很适合或无效的。在这项工作中，我们证明了在没有热启动的情况下，仍然有可能获得按顺序最优或接近最优的样本复杂度。特别地，我们提出了一种简单的方法，它在低层使用随机不动点迭代，在上层使用投影的不精确梯度下降，分别在随机和确定环境下使用$O(epsilon^{-2})$和$tide{O}(epsilon^{-1})$样本达到$epsilon$-驻点。最后，与使用热启动的方法相比，我们的方法产生了更简单的分析，不需要研究上层和下层迭代之间的耦合作用



## **28. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

菲亚特-沙米尔的证据缺乏证据，即使在存在共同纠缠的情况下 quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.

摘要: 我们探索任意共享物理资源的加密能力。最常见的这类资源是在每个协议执行开始时访问新的纠缠量子态。我们称之为公共参考量子态(CRQS)模型，类似于众所周知的公共参考弦(CRS)。CRQS模型是CRS模型的自然推广，但似乎更强大：在两方设置中，CRQS有时可以通过测量许多相互无偏的碱基之一中的最大纠缠态来展示与查询一次的随机Oracle相关联的属性。我们将这个概念形式化为弱一次性随机Oracle(WOTRO)，其中我们只要求$m$位的输出在以$n$位输入为条件时具有一定的随机性。我们证明了当$n-m\in\omega(\lg n)$时，CRQS模型中用于WOTRO的任何协议都可以被(低效的)攻击者攻击。此外，我们的对手是高效可模拟的，这排除了通过将黑盒简化为密码博弈假设来证明方案的计算安全性的可能性。另一方面，我们引入了散列函数的非博弈量子假设，在CRQ模型(其中CRQS只由EPR对组成)中隐含了WOTRO。我们首先构建一个统计安全的WOTRO协议，其中$m=n$，然后对输出进行散列。WOTRO的不可能性会产生以下后果。首先，我们证明了量子Fiat-Shamir变换的黑盒不可能性，推广了Bitansky等人的不可能结果。其次，我们给出了一个加强版量子闪电(Zhandry，Eurocrypt‘19)的黑箱不可能结果，其中量子闪电有一个额外的参数，如果不产生新的闪电，这个参数就不能改变。



## **29. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 几十年来，计算机系统保存着大量的个人数据。一方面，这样的数据丰富使人工智能(AI)，特别是机器学习(ML)模型取得了突破。另一方面，它会威胁用户的隐私，削弱人类与AI之间的信任。最近的法规要求，一般情况下，可以从计算机系统中删除关于用户的私人信息，特别是在请求时可以从ML模型中删除用户的私人信息(例如，“被遗忘权”)。虽然从后端数据库中删除数据应该很简单，但在人工智能环境中这是不够的，因为ML模型经常“记住”旧数据。现有的对抗性攻击证明，我们可以从训练好的模型中学习训练数据的私人成员或属性。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。在这篇调查论文中，我们试图对机器遗忘的定义、场景、机制和应用进行全面的调查。具体地说，作为最新研究的分类集合，我们希望为那些寻求机器遗忘及其各种公式、设计要求、移除请求、算法和在各种ML应用中使用的入门知识的人提供广泛的参考。此外，我们希望概述该范式中的主要发现和趋势，并强调尚未看到机器遗忘应用的新研究领域，但仍可能受益匪浅。我们希望这项调查为ML研究人员以及那些寻求创新隐私技术的人提供有价值的参考。我们的资源在https://github.com/tamlhp/awesome-machine-unlearning.



## **30. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

GRNN：生成回归神经网络--一种面向联邦学习的数据泄漏攻击 cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.

摘要: 数据隐私已经成为机器学习(ML)中一个日益重要的问题，人们已经开发了许多方法来应对这一挑战，例如密码学(同态加密(HE)、差分隐私(DP)等)。和协作培训(安全多方计算(MPC)、分布式学习和联合学习(FL))。这些技术特别关注数据加密或安全本地计算。他们将中间信息传递给第三方来计算最终结果。梯度交换通常被认为是深度学习中协作训练健壮模型的一种安全方式。然而，最近的研究表明，敏感信息可以从共享梯度中恢复出来。尤其是生成性对抗网络(GAN)在恢复这类信息方面是有效的。然而，基于GaN的技术需要额外的信息，例如类别标签，这些信息通常不能用于隐私保护学习。在本文中，我们证明，在FL系统中，仅通过我们提出的生成回归神经网络(GRNN)就可以很容易地从共享梯度中完全恢复基于图像的隐私数据。我们将攻击描述为一个回归问题，并通过最小化梯度之间的距离来优化生成模型的两个分支。我们在几个图像分类任务上对我们的方法进行了评估。实验结果表明，本文提出的GRNN方法在稳定性、鲁棒性和准确率等方面均优于现有的方法。它对全局FL模型也没有收敛要求。此外，我们还使用人脸重新识别来演示信息泄漏。文中还讨论了一些防御策略。



## **31. Semantic-Preserving Adversarial Code Comprehension**

保留语义的对抗性代码理解 cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.

摘要: 基于预先训练的语言模型在源代码理解任务中的巨大成功，目前的文献研究要么是进一步提高预先训练的语言模型的性能(泛化)，要么是研究它们对对手攻击的健壮性。然而，他们不得不在这两个方面的权衡上妥协，没有一个人考虑以有效和实际的方式改善双方。为了填补这一空白，我们提出了保持语义的对抗性代码嵌入(SPACE)，以发现最坏情况下保持语义的攻击，同时迫使模型在这些最坏情况下预测正确的标签。实验和分析表明，SPACE在提高PrLMS代码性能的同时，可以保持对最先进攻击的健壮性。



## **32. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **33. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

注意：通过变分推理进行推理的可证明稳健学习 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.

摘要: 尽管深度神经网络(DNN)最近取得了很大的进展，但它们往往容易受到对手的攻击。人们已经进行了大量的研究来提高DNN的稳健性，然而，大多数经验防御都可以再次自适应攻击，理论上证明的健壮性是有限的，特别是在大规模数据集上。DNN这种漏洞的一个潜在根本原因是，尽管它们表现出强大的表现力，但它们缺乏做出稳健和可靠预测的推理能力。在本文中，我们的目标是将领域知识集成到推理范式中，以实现稳健的学习。特别地，我们提出了一种带推理的可证明稳健学习流水线(CARE)，该流水线由学习组件和推理组件组成。具体地说，我们使用一组标准的DNN作为学习组件进行语义预测，并利用马尔可夫逻辑网络(MLN)等概率图形模型作为推理组件来实现知识/逻辑推理。然而，众所周知，MLN(推理)的精确推理是#P-完全的，这限制了流水线的可扩展性。为此，我们提出了基于一种有效的期望最大化算法的变分推理来逼近最大似然推理。特别是，我们利用图卷积网络(GCNS)对变分推理过程中的后验分布进行编码，并迭代地更新GCNS的参数(E步)和MLN中知识规则的权值(M步)。我们在不同的数据集上进行了广泛的实验，并表明与最先进的基线相比，CARE实现了显著更高的认证稳健性。此外，我们还进行了不同的消融研究，以证明CARE的经验稳健性和不同知识整合的有效性。



## **34. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

GFCL：一种基于GRU的联合持续学习框架对抗IoV中的数据中毒攻击 cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.

摘要: 将机器学习(ML)集成到基于5G的车联网(IoV)网络中，实现了智能交通和智能交通管理。尽管如此，防御对抗性中毒攻击的安全也日益成为一项具有挑战性的任务。其中，深度强化学习(DRL)是IoV应用中广泛使用的ML设计之一。标准的ML安全技术在DRL中并不有效，在DRL中，算法通过与环境的持续交互来学习解决顺序决策，并且环境是时变的、动态的和移动的。针对物联网中基于Sybil的数据中毒攻击，提出了一种基于门控递归单元(GRU)的联合连续学习(GFCL)异常检测框架。其目的是提供一个轻量级和可扩展的框架，在没有包含攻击样本的先验训练数据集的情况下学习和检测非法行为。我们使用GRU来预测未来的数据序列，以基于联合学习的分布式方式来分析和检测车辆的非法行为。我们使用真实世界的车辆移动轨迹来研究我们的框架的性能。结果表明，在不同的性能指标下，我们提出的解决方案是有效的。



## **35. Generate novel and robust samples from data: accessible sharing without privacy concerns**

从数据中生成新颖且可靠的样本：无隐私问题的可访问共享 cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.

摘要: 从数据集生成新样本可以减少额外昂贵的操作、增加侵入性程序，并缓解隐私问题。当隐私受到关注时，这些在统计上稳健的新样本可以用作临时和中间替代。这种方法可以实现更好的数据共享实践，而不会出现与识别问题或作为对抗性攻击缺陷的偏见有关的问题。



## **36. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

基于特征随机化的抗敌意攻击传递的深度学习模型 cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.

摘要: 在过去的几十年里，人工智能的崛起给了我们解决日常生活中最具挑战性的问题的能力，比如癌症预测和自主导航。但是，如果不确保这些应用程序不能抵御敌意攻击，则这些应用程序可能不可靠。此外，最近的工作表明，一些对抗性例子可以在不同的模型之间转移。因此，通过稳健的模型抵抗敌意操纵来避免这种可转移性是至关重要的。在本文中，我们提出了一种基于特征随机化的方法，该方法在测试阶段抵抗了八种针对深度学习模型的对抗性攻击。我们的新方法包括改变目标网络分类器中的训练策略和随机选择特征样本。我们认为攻击者在有限知识和半知识条件下可以进行最常见的对抗性攻击类型。我们使用著名的UNSW-NB15数据集评估了我们方法的健壮性，其中包括现实攻击和合成攻击。之后，我们证明了我们的策略优于现有的最先进的方法，例如最强大的攻击，它包括针对特定的对手攻击对网络模型进行微调。实验结果表明，该方法能够保证目标网络的安全，并能抵抗60%以上的敌意攻击可转移性。



## **37. Detecting Adversarial Perturbations in Multi-Task Perception**

多任务感知中的对抗性扰动检测 cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.

摘要: 虽然深度神经网络(DNN)在环境感知任务中取得了令人印象深刻的性能，但它们对对抗性扰动的敏感性限制了它们在实际应用中的应用。本文(I)提出了一种基于复杂视觉任务多任务感知(深度估计和语义分割)的对抗性扰动检测方法。具体地，通过所提取的输入图像的边缘、深度输出和分割输出之间的不一致来检测对抗性扰动。为了进一步改进这一技术，我们(Ii)在所有三种模式之间开发了一种新的边缘一致性损失，从而改善了它们的初始一致性，这反过来又支持我们的检测方案。通过使用各种已知攻击和图像噪声来验证我们的检测方案的有效性。此外，我们(Iii)开发了一种多任务对抗性攻击，旨在愚弄两个任务和我们的检测方案。在CITYSCAPES和KITTI数据集上的实验评估表明，在假阳性率为5%的假设下，高达100%的图像被正确检测为恶意扰动，这取决于扰动的强度。代码可在https://github.com/ifnspaml/AdvAttackDet.上找到Https://youtu.be/KKa6gOyWmH4上的一段简短视频提供了定性的结果。



## **38. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

对比性对抗训练中认知分离缓解的稳健性 cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.

摘要: 本文提出了一种新的神经网络训练框架，通过将对比学习(CL)和对抗训练(AT)相结合，在保持较高精度的同时，提高了模型对对手攻击的鲁棒性。我们提出通过学习在数据扩充和对抗性扰动下都是一致的特征表示来提高模型对对抗性攻击的稳健性。我们利用对比学习来提高对抗性样本的稳健性，将一个对抗性样本作为另一个正例，目标是最大化随机增加的数据样本与其对抗性样本之间的相似度，同时不断更新分类头，以避免分类头与嵌入空间之间的认知分离。这种分离是由于CL将网络更新到嵌入空间，同时冻结用于生成新的正面对抗性实例的分类头。我们在CIFAR-10数据集上验证了我们的方法，即带有对抗性特征的对比学习(CLAF)，在CIFAR-10数据集上，它的性能优于其他监督和自我监督对抗性学习方法的稳健准确率和干净准确率。



## **39. Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense**

散射模型制导的SAR目标识别对抗实例：攻防 cs.CV

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04779v1)

**Authors**: Bowen Peng, Bo Peng, Jie Zhou, Jianyue Xie, Li Liu

**Abstracts**: Deep Neural Networks (DNNs) based Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems have shown to be highly vulnerable to adversarial perturbations that are deliberately designed yet almost imperceptible but can bias DNN inference when added to targeted objects. This leads to serious safety concerns when applying DNNs to high-stake SAR ATR applications. Therefore, enhancing the adversarial robustness of DNNs is essential for implementing DNNs to modern real-world SAR ATR systems. Toward building more robust DNN-based SAR ATR models, this article explores the domain knowledge of SAR imaging process and proposes a novel Scattering Model Guided Adversarial Attack (SMGAA) algorithm which can generate adversarial perturbations in the form of electromagnetic scattering response (called adversarial scatterers). The proposed SMGAA consists of two parts: 1) a parametric scattering model and corresponding imaging method and 2) a customized gradient-based optimization algorithm. First, we introduce the effective Attributed Scattering Center Model (ASCM) and a general imaging method to describe the scattering behavior of typical geometric structures in the SAR imaging process. By further devising several strategies to take the domain knowledge of SAR target images into account and relax the greedy search procedure, the proposed method does not need to be prudentially finetuned, but can efficiently to find the effective ASCM parameters to fool the SAR classifiers and facilitate the robust model training. Comprehensive evaluations on the MSTAR dataset show that the adversarial scatterers generated by SMGAA are more robust to perturbations and transformations in the SAR processing chain than the currently studied attacks, and are effective to construct a defensive model against the malicious scatterers.

摘要: 基于深度神经网络(DNN)的合成孔径雷达(SAR)自动目标识别(ATR)系统被证明是非常容易受到敌意扰动的，这些扰动是故意设计的，但几乎不可察觉，但当添加到目标对象时，会使DNN推理产生偏差。这导致在将DNN应用于高风险的SAR ATR应用时存在严重的安全问题。因此，增强DNN的对抗健壮性对于将DNN应用于现代真实的SARATR系统是至关重要的。为了建立更稳健的基于离散神经网络的合成孔径雷达ATR模型，本文深入研究了合成孔径雷达成像过程中的领域知识，提出了一种新的散射模型制导对抗攻击算法(SMGAA)，该算法可以产生电磁散射响应形式的对抗扰动(称为对抗散射者)。SMGAA由两部分组成：1)参数散射模型和相应的成像方法；2)定制的基于梯度的优化算法。首先，我们引入了有效属性散射中心模型(ASCM)和一种通用的成像方法来描述合成孔径雷达成像过程中典型几何结构的散射行为。通过进一步设计几种策略来考虑SAR目标图像的领域知识，并放松贪婪搜索过程，该方法不需要谨慎地精调，但可以有效地找到有效的ASCM参数来愚弄SAR分类器，便于稳健的模型训练。在MStar数据集上的综合评估表明，SMGAA生成的敌意散射体对SAR处理链中的扰动和变换具有更强的鲁棒性，并且能够有效地构建针对恶意散射体的防御模型。



## **40. Atomic cross-chain exchanges of shared assets**

共享资产的原子跨链交换 cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2202.12855v3)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.

摘要: 区块链或DLT互操作性的核心推动因素是能够自动交换不同分类账上相互不信任的所有者持有的资产。这个原子交换问题已经得到了很好的研究，哈希时间锁定合同(HTLC)成为一种规范的解决方案。HTLC确保了交换的原子性，但对节点故障和索赔的及时性提出了警告。但HTLC的一个更大限制是，它只适用于由两个对立方单独拥有每个分类账中的一项资产的模型。资产可能由多方共同拥有的模型的现实扩展，其中交易需要所有各方的同意，或者必须用多个资产交换一个资产，这容易受到共谋攻击，因此无法由HTLC处理。本文对跨DLT网络的资产交换模型进行了推广，给出了用例的分类，描述了威胁模型，并提出了一种用于原子多所有者和资产交换的扩展HTLC协议MPHTLC。分析了MPHTLC的正确性、安全性和适用范围。作为概念验证，我们展示了如何在建立在Hyperledger Fabric和Corda上的网络中实现MPHTLC原语，以及如何通过增强Hyperledger Labs Weaver框架的现有HTLC协议来实现MPHTLC。



## **41. Phantom Sponges: Exploiting Non-Maximum Suppression to Attack Deep Object Detectors**

幻影海绵：利用非最大抑制攻击深度对象探测器 cs.CV

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.13618v2)

**Authors**: Avishag Shapira, Alon Zolfi, Luca Demetrio, Battista Biggio, Asaf Shabtai

**Abstracts**: Adversarial attacks against deep learning-based object detectors have been studied extensively in the past few years. Most of the attacks proposed have targeted the model's integrity (i.e., caused the model to make incorrect predictions), while adversarial attacks targeting the model's availability, a critical aspect in safety-critical domains such as autonomous driving, have not yet been explored by the machine learning research community. In this paper, we propose a novel attack that negatively affects the decision latency of an end-to-end object detection pipeline. We craft a universal adversarial perturbation (UAP) that targets a widely used technique integrated in many object detector pipelines -- non-maximum suppression (NMS). Our experiments demonstrate the proposed UAP's ability to increase the processing time of individual frames by adding "phantom" objects that overload the NMS algorithm while preserving the detection of the original objects (which allows the attack to go undetected for a longer period of time).

摘要: 针对基于深度学习的目标检测器的对抗性攻击在过去的几年中得到了广泛的研究。大多数提出的攻击都是针对模型的完整性(即导致模型做出错误的预测)，而针对模型可用性的对抗性攻击(自动驾驶等安全关键领域的一个关键方面)尚未被机器学习研究社区探索。在本文中，我们提出了一种新的攻击，它对端到端对象检测流水线的决策延迟产生负面影响。我们设计了一种通用对抗摄动(UAP)，目标是集成在许多对象探测器流水线中的一种广泛使用的技术--非最大抑制(NMS)。我们的实验证明了所提出的UAP能够通过添加超载NMS算法的“幻影”对象来增加单个帧的处理时间，同时保持对原始对象的检测(这使得攻击在更长的时间内不被检测到)。



## **42. Bankrupting DoS Attackers Despite Uncertainty**

尽管存在不确定性，但仍使DoS攻击者破产 cs.CR

**SubmitDate**: 2022-09-10    [paper-pdf](http://arxiv.org/pdf/2205.08287v2)

**Authors**: Trisha Chakraborty, Abir Islam, Valerie King, Daniel Rayborn, Jared Saia, Maxwell Young

**Abstracts**: On-demand provisioning in the cloud allows for services to remain available despite massive denial-of-service (DoS) attacks. Unfortunately, on-demand provisioning is expensive and must be weighed against the costs incurred by an adversary. This leads to a recent threat known as {\it economic denial-of-sustainability (EDoS)}, where the cost for defending a service is higher than that of attacking.   A natural tool for combating EDoS is to impose costs via resource burning (RB). Here, a client must verifiably consume resources -- for example, by solving a computational challenge -- before service is rendered. However, prior RB-based defenses with security guarantees do not account for the cost of on-demand provisioning.   Another common approach is the use of heuristics -- such as a client's reputation score or the geographical location -- to identify and discard spurious job requests. However, these heuristics may err and existing approaches do not provide security guarantees when this occurs.   Here, we propose an EDoS defense, LCharge, that uses resource burning while accounting for on-demand provisioning. LCharge leverages an estimate of the number of job requests from honest clients (i.e., good jobs) in any set $S$ of requests to within an $O(\alpha)$-factor, for any unknown $\alpha>0$, but retains a strong security guarantee despite the uncertainty of this estimate. Specifically, against an adversary that expends $B$ resources to attack, the total cost for defending is $O( \alpha^{5/2}\sqrt{B\,(g+1)} + \alpha^3(g+\alpha))$ where $g$ is the number of good jobs. Notably, for large $B$ relative to $g$ and $\alpha$, the adversary has higher cost, implying that the algorithm has an economic advantage. Finally, we prove a lower bound for our problem of $\Omega(\sqrt{\alpha B g})$, showing that the cost of LCharge is asymptotically tight for $\alpha=\Theta(1)$.

摘要: 云中的按需配置允许服务在遭受大规模拒绝服务(DoS)攻击时保持可用。不幸的是，按需配置的成本很高，必须权衡对手所产生的成本。这导致了最近的一种称为{\it经济拒绝可持续性(EDOS)}的威胁，在这种威胁下，防御服务的成本高于攻击。对抗EDO的一个自然工具是通过资源燃烧(RB)来施加成本。在这里，在提供服务之前，客户端必须可验证地消耗资源--例如，通过解决计算挑战。然而，以前的基于RB的安全保证防御不会考虑按需配置的成本。另一种常见的方法是使用试探法--例如客户的声誉分数或地理位置--来识别和丢弃虚假的工作请求。然而，这些启发式方法可能会出错，并且现有方法在发生这种情况时不提供安全保证。在这里，我们提出了EDOS防御LCharge，它在考虑按需配置的同时使用资源消耗。LCharge利用任何$S$请求集合中来自诚实客户(即好工作)的工作请求数量的估计，对于任何未知的$\α>0$，在$O(\Alpha)$因子内，但尽管该估计存在不确定性，LCharge仍保持强大的安全保证。具体地说，针对花费$B$资源进行攻击的对手，防御的总成本为$O(\alpha^{5/2}\sqrt{B\，(g+1)}+\alpha^3(g+\pha))$，其中$g$是好工作的数量。值得注意的是，对于较大的$B$相对于$g$和$\α$，对手具有更高的成本，这意味着该算法具有经济优势。最后，我们证明了问题$Omega(\Sqrt{\Alpha Bg})$的一个下界，证明了LCharge的代价对于$\α=\Theta(1)$是渐近紧的。



## **43. The Space of Adversarial Strategies**

对抗性策略的空间 cs.CR

Accepted to the 32nd USENIX Security Symposium

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04521v1)

**Authors**: Ryan Sheatsley, Blaine Hoak, Eric Pauley, Patrick McDaniel

**Abstracts**: Adversarial examples, inputs designed to induce worst-case behavior in machine learning models, have been extensively studied over the past decade. Yet, our understanding of this phenomenon stems from a rather fragmented pool of knowledge; at present, there are a handful of attacks, each with disparate assumptions in threat models and incomparable definitions of optimality. In this paper, we propose a systematic approach to characterize worst-case (i.e., optimal) adversaries. We first introduce an extensible decomposition of attacks in adversarial machine learning by atomizing attack components into surfaces and travelers. With our decomposition, we enumerate over components to create 576 attacks (568 of which were previously unexplored). Next, we propose the Pareto Ensemble Attack (PEA): a theoretical attack that upper-bounds attack performance. With our new attacks, we measure performance relative to the PEA on: both robust and non-robust models, seven datasets, and three extended lp-based threat models incorporating compute costs, formalizing the Space of Adversarial Strategies. From our evaluation we find that attack performance to be highly contextual: the domain, model robustness, and threat model can have a profound influence on attack efficacy. Our investigation suggests that future studies measuring the security of machine learning should: (1) be contextualized to the domain & threat models, and (2) go beyond the handful of known attacks used today.

摘要: 对抗性例子是机器学习模型中旨在诱导最坏情况行为的输入，在过去的十年里得到了广泛的研究。然而，我们对这一现象的理解源于相当零散的知识池；目前，有几种攻击，每一种攻击在威胁模型中都有不同的假设，对最优的定义也是无与伦比的。在本文中，我们提出了一种系统的方法来刻画最坏情况(即最佳)对手的特征。我们首先介绍了对抗性机器学习中攻击的一种可扩展分解，将攻击组件原子化到表面和旅行者中。通过我们的分解，我们列举了组件以创建576次攻击(其中568次是以前未曾探索过的)。接下来，我们提出了Pareto系综攻击(PEA)：一种上界攻击性能的理论攻击。在我们的新攻击中，我们在以下方面衡量相对于PEA的性能：稳健和非稳健模型、七个数据集和三个包含计算成本的扩展的基于LP的威胁模型，正式确定了对手战略空间。从我们的评估中，我们发现攻击性能与上下文高度相关：域、模型健壮性和威胁模型可以对攻击效率产生深远的影响。我们的调查表明，未来衡量机器学习安全性的研究应该：(1)从域和威胁模型出发，(2)超越目前使用的少数已知攻击。



## **44. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络认证的健壮性 cs.LG

To appear at 2023 IEEE Symposium on Security and Privacy (SP); 14  pages for the main text; benchmark & tool website:  http://sokcertifiedrobustness.github.io/

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2009.04131v8)

**Authors**: Linyi Li, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which can usually be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches, which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate 20+ representative certifiably robust approaches.

摘要: 深度神经网络(DNN)的巨大进步导致了在各种任务中最先进的性能。然而，最近的研究表明，DNN很容易受到对手攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常可以在不提供健壮性证明的情况下自适应地再次攻击；b)可证明的健壮性方法，包括在一定条件下提供对任何攻击的健壮性精度下界的健壮性验证和相应的健壮训练方法。在这篇文章中，我们系统化了可证明的稳健方法以及相关的实践和理论意义和发现。我们还提供了关于不同数据集上现有稳健性验证和训练方法的第一个全面基准。具体地说，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优势、局限性和基本联系；3)讨论了当前的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多个具有代表性的可证健壮方法。



## **45. Adversarial Examples in Constrained Domains**

受限领域中的对抗性例子 cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2011.01183v3)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.

摘要: 已经证明，机器学习算法通过系统地修改诸如图像识别等领域中的输入(例如，对抗性例子)而容易受到对抗性操纵。在默认威胁模型下，对手利用图像的不受限制的性质；每个特征(像素)都完全在对手的控制之下。然而，目前尚不清楚这些攻击如何转化为受限的域，从而限制攻击者可以修改哪些功能以及如何修改(例如，网络入侵检测)。在本文中，我们探讨了受限域是否比非约束域更不容易受到敌意示例生成算法的影响。我们创建了一种生成对抗性草图的算法：目标通用扰动向量，它在领域约束的包络内编码特征显著。为了评估这些算法的性能，我们在受限(例如，网络入侵检测)和非受限(例如，图像识别)域中对它们进行评估。结果表明，我们的方法在受限领域产生的错误分类率与非约束领域相当(大于95%)。我们的调查表明，受约束域暴露的狭窄攻击面仍然足够大，足以制作成功的敌意示例；因此，约束似乎不会使域变得健壮。事实上，只需随机选择五个特征，就仍然可以生成对抗性的例子。



## **46. Robust-by-Design Classification via Unitary-Gradient Neural Networks**

基于么正梯度神经网络的稳健设计分类 cs.LG

Under review

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04293v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of a SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.

摘要: 由于存在对抗性攻击，在安全关键系统中使用神经网络需要安全和健壮的模型。知道任何输入x的最小对抗性扰动，或者，等价地，知道x到分类边界的距离，允许评估分类稳健性，提供可证明的预测。不幸的是，用于计算这种距离的最先进的技术计算成本很高，因此不适合在线应用。这项工作提出了一类新的分类器，即符号距离分类器(SDCS)，从理论上讲，它直接输出x到分类边界的准确距离，而不是概率分数(例如SoftMax)。SDC代表了一系列稳健的按设计分类的分类器。为了满足SDC的理论要求，提出了一种新的网络体系结构--酉梯度神经网络。实验结果表明，该体系结构接近于符号距离分类器，从而允许以单一推理为代价对x进行在线可证明分类。



## **47. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

通过认知不确定性对抗性训练改进失配检测 cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.03148v2)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.

摘要: 不确定性的量化对于机器学习的采用非常重要，特别是对于拒绝将分布外(OOD)数据返回给人类专家进行审查。然而，进展缓慢，因为必须在计算效率和不确定性估计的质量之间取得平衡。出于这个原因，许多人使用神经网络的深度集成或蒙特卡罗退学来在相对最小的计算和内存下进行合理的不确定性估计。令人惊讶的是，当我们关注现实世界中可应用的假阳性率(FPR)约束时，现有方法无法可靠地检测出OOD样本。值得注意的是，即使是高斯随机噪声也无法触发这些流行的OOD技术。我们通过设计一个简单的对抗性训练方案来帮助缓解这个问题，该方案结合了对辍学生群体预测的认知不确定性的攻击。我们证明了该方法提高了对标准数据的OOD检测性能(即，不是恶意定制的)，并将标准化的部分AUC从近乎随机猜测的性能提高到0.75美元。



## **48. Harnessing Perceptual Adversarial Patches for Crowd Counting**

利用感知对抗性斑块进行人群计数 cs.CV

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2109.07986v2)

**Authors**: Shunchang Liu, Jiakai Wang, Aishan Liu, Yingwei Li, Yijie Gao, Xianglong Liu, Dacheng Tao

**Abstracts**: Crowd counting, which has been widely adopted for estimating the number of people in safety-critical scenes, is shown to be vulnerable to adversarial examples in the physical world (e.g., adversarial patches). Though harmful, adversarial examples are also valuable for evaluating and better understanding model robustness. However, existing adversarial example generation methods for crowd counting lack strong transferability among different black-box models, which limits their practicability for real-world systems. Motivated by the fact that attacking transferability is positively correlated to the model-invariant characteristics, this paper proposes the Perceptual Adversarial Patch (PAP) generation framework to tailor the adversarial perturbations for crowd counting scenes using the model-shared perceptual features. Specifically, we handcraft an adaptive crowd density weighting approach to capture the invariant scale perception features across various models and utilize the density guided attention to capture the model-shared position perception. Both of them are demonstrated to improve the attacking transferability of our adversarial patches. Extensive experiments show that our PAP could achieve state-of-the-art attacking performance in both the digital and physical world, and outperform previous proposals by large margins (at most +685.7 MAE and +699.5 MSE). Besides, we empirically demonstrate that adversarial training with our PAP can benefit the performance of vanilla models in alleviating several practical challenges in crowd counting scenarios, including generalization across datasets (up to -376.0 MAE and -354.9 MSE) and robustness towards complex backgrounds (up to -10.3 MAE and -16.4 MSE).

摘要: 人群计数被广泛用于估计安全关键场景中的人数，但在现实世界中，它很容易受到对抗性例子的影响(例如，对抗性补丁)。对抗性例子虽然有害，但对于评估和更好地理解模型的健壮性也是有价值的。然而，现有的人群计数对抗性实例生成方法在不同的黑盒模型之间缺乏很强的可移植性，这限制了它们在现实系统中的实用性。基于攻击的可转移性与模型不变特性正相关这一事实，提出了感知对抗性补丁(PAP)生成框架，利用模型共享的感知特征来定制人群计数场景中的对抗性扰动。具体地说，我们手工设计了一种自适应的人群密度加权方法来捕捉各种模型上的不变尺度感知特征，并利用密度引导注意力来捕捉模型共享的位置感知。它们都被证明可以提高我们对手补丁的攻击可转移性。大量的实验表明，我们的PAP在数字和物理世界都可以达到最先进的攻击性能，并且比以前的方案有很大的优势(最多+685.7 MAE和+699.5 MSE)。此外，我们的经验证明，使用我们的PAP进行对抗性训练可以帮助Vanilla模型在缓解人群计数场景中的几个实际挑战方面的性能，包括跨数据集的泛化(高达-376.0 MAE和-354.9 MSE)以及对复杂背景的稳健性(高达-10.3MAE和-16.4MSE)。



## **49. Uncovering the Connection Between Differential Privacy and Certified Robustness of Federated Learning against Poisoning Attacks**

揭示差分隐私与联合学习对中毒攻击的认证健壮性之间的联系 cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04030v1)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Bo Li

**Abstracts**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As the local training data come from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is always trained in a differentially private way (DPFL). Thus, in this paper, we ask: Can we leverage the innate privacy property of DPFL to provide certified robustness against poisoning attacks? Can we further improve the privacy of FL to improve such certification? We first investigate both user-level and instance-level privacy of FL and propose novel mechanisms to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack cost for DPFL on both levels. Theoretically, we prove the certified robustness of DPFL under a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of attacks on different datasets. We show that DPFL with a tighter privacy guarantee always provides stronger robustness certification in terms of certified attack cost, but the optimal certified prediction is achieved under a proper balance between privacy protection and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不可信的不同用户，多项研究表明FL很容易受到中毒攻击。同时，为了保护本地用户的隐私，FL总是以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：我们能否利用DPFL固有的隐私属性来提供经过认证的针对中毒攻击的健壮性？我们能否进一步改善FL的隐私，以提高此类认证？我们首先研究了FL的用户级隐私和实例级隐私，并提出了新的机制来实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在两个级别上的认证预测和认证攻击成本。理论上，我们证明了DPFL在有限数量的敌意用户或实例下的证明的健壮性。在经验上，我们在不同数据集的一系列攻击下进行了广泛的实验来验证我们的理论。我们发现，在认证攻击代价方面，具有更严格隐私保障的DPFL总是提供更强的健壮性认证，但最优认证预测是在隐私保护和效用损失之间取得适当平衡的情况下实现的。



## **50. Evaluating the Security of Aircraft Systems**

评估飞机系统的安全性 cs.CR

38 pages,

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04028v1)

**Authors**: Edan Habler, Ron Bitton, Asaf Shabtai

**Abstracts**: The sophistication and complexity of cyber attacks and the variety of targeted platforms have been growing in recent years. Various adversaries are abusing an increasing range of platforms, e.g., enterprise platforms, mobile phones, PCs, transportation systems, and industrial control systems. In recent years, we have witnessed various cyber attacks on transportation systems, including attacks on ports, airports, and trains. It is only a matter of time before transportation systems become a more common target of cyber attackers. Due to the enormous potential damage inherent in attacking vehicles carrying many passengers and the lack of security measures applied in traditional airborne systems, the vulnerability of aircraft systems is one of the most concerning topics in the vehicle security domain. This paper provides a comprehensive review of aircraft systems and components and their various networks, emphasizing the cyber threats they are exposed to and the impact of a cyber attack on these components and networks and the essential capabilities of the aircraft. In addition, we present a comprehensive and in-depth taxonomy that standardizes the knowledge and understanding of cyber security in the avionics field from an adversary's perspective. The taxonomy divides techniques into relevant categories (tactics) reflecting the various phases of the adversarial attack lifecycle and maps existing attacks according to the MITRE ATT&CK methodology. Furthermore, we analyze the security risks among the various systems according to the potential threat actors and categorize the threats based on STRIDE threat model. Future work directions are presented as guidelines for industry and academia.

摘要: 近年来，网络攻击的复杂性和复杂性以及目标平台的多样性一直在增长。各种对手正在滥用越来越多的平台，例如企业平台、移动电话、PC、交通系统和工业控制系统。近年来，我们目睹了针对交通系统的各种网络攻击，包括对港口、机场和火车的攻击。交通系统成为网络攻击者更常见的目标只是个时间问题。由于攻击载客车辆固有的巨大潜在危害，以及传统机载系统缺乏安全措施，飞机系统的脆弱性是车辆安全领域最受关注的话题之一。本文对飞机系统和部件及其各种网络进行了全面的回顾，强调了它们所面临的网络威胁，以及网络攻击对这些部件和网络以及飞机的基本能力的影响。此外，我们提出了一个全面和深入的分类，从对手的角度标准化了对航空电子领域网络安全的知识和理解。该分类将技术划分为相关类别(战术)，反映对抗性攻击生命周期的不同阶段，并根据MITRE ATT&CK方法映射现有攻击。在此基础上，根据潜在威胁主体分析了各个系统之间的安全风险，并基于STRIDE威胁模型对威胁进行了分类。提出了未来的工作方向，作为产业界和学术界的指导方针。



