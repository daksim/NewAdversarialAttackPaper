# Latest Adversarial Attack Papers
**update at 2022-03-18 06:31:57**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Attacking deep networks with surrogate-based adversarial black-box methods is easy**

用基于代理的对抗性黑盒方法攻击深层网络很容易 cs.LG

ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08725v1)

**Authors**: Nicholas A. Lord, Romain Mueller, Luca Bertinetto

**Abstracts**: A recent line of work on black-box adversarial attacks has revived the use of transfer from surrogate models by integrating it into query-based search. However, we find that existing approaches of this type underperform their potential, and can be overly complicated besides. Here, we provide a short and simple algorithm which achieves state-of-the-art results through a search which uses the surrogate network's class-score gradients, with no need for other priors or heuristics. The guiding assumption of the algorithm is that the studied networks are in a fundamental sense learning similar functions, and that a transfer attack from one to the other should thus be fairly "easy". This assumption is validated by the extremely low query counts and failure rates achieved: e.g. an untargeted attack on a VGG-16 ImageNet network using a ResNet-152 as the surrogate yields a median query count of 6 at a success rate of 99.9%. Code is available at https://github.com/fiveai/GFCS.

摘要: 最近关于黑盒对抗性攻击的一系列工作通过将从代理模型转移集成到基于查询的搜索中，重新唤醒了它的使用。然而，我们发现现有的这类方法没有发挥其潜力，而且可能过于复杂。在这里，我们提供了一个简短而简单的算法，它通过使用代理网络的类分数梯度的搜索来获得最先进的结果，而不需要其他先验或启发式方法。该算法的指导性假设是，所研究的网络在基本意义上学习相似的函数，因此从一个网络到另一个网络的转移攻击应该相当“容易”。实现的极低的查询计数和失败率证实了这一假设：例如，使用ResNet-152作为代理对VGG-16 ImageNet网络进行的无目标攻击产生的查询计数中值为6，成功率为99.9%。代码可在https://github.com/fiveai/GFCS.上获得



## **2. On the Security & Privacy in Federated Learning**

论联合学习中的安全性和隐私性 cs.CR

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2112.05423v2)

**Authors**: Gorka Abad, Stjepan Picek, Víctor Julio Ramírez-Durán, Aitor Urbieta

**Abstracts**: Recent privacy awareness initiatives such as the EU General Data Protection Regulation subdued Machine Learning (ML) to privacy and security assessments. Federated Learning (FL) grants a privacy-driven, decentralized training scheme that improves ML models' security. The industry's fast-growing adaptation and security evaluations of FL technology exposed various vulnerabilities that threaten FL's confidentiality, integrity, or availability (CIA). This work assesses the CIA of FL by reviewing the state-of-the-art (SoTA) and creating a threat model that embraces the attack's surface, adversarial actors, capabilities, and goals. We propose the first unifying taxonomy for attacks and defenses and provide promising future research directions.

摘要: 最近的隐私意识倡议，如欧盟通用数据保护条例，使机器学习(ML)受制于隐私和安全评估。联邦学习(FL)提供了一种隐私驱动的、分散的训练方案，提高了ML模型的安全性。业界对FL技术的快速适应和安全评估暴露了威胁FL的机密性、完整性或可用性(CIA)的各种漏洞。这项工作评估中央情报局的FL通过审查最先进的(SOTA)和创建一个威胁模型，包括攻击的表面，敌对行为者，能力和目标。我们提出了第一个统一的攻击和防御分类法，并提供了有前景的未来研究方向。



## **3. Towards Practical Certifiable Patch Defense with Vision Transformer**

用视觉变送器实现实用的可认证补丁防御 cs.CV

Accepted by CVPR2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08519v1)

**Authors**: Zhaoyu Chen, Bo Li, Jianghe Xu, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstracts**: Patch attacks, one of the most threatening forms of physical attack in adversarial examples, can lead networks to induce misclassification by modifying pixels arbitrarily in a continuous region. Certifiable patch defense can guarantee robustness that the classifier is not affected by patch attacks. Existing certifiable patch defenses sacrifice the clean accuracy of classifiers and only obtain a low certified accuracy on toy datasets. Furthermore, the clean and certified accuracy of these methods is still significantly lower than the accuracy of normal classification networks, which limits their application in practice. To move towards a practical certifiable patch defense, we introduce Vision Transformer (ViT) into the framework of Derandomized Smoothing (DS). Specifically, we propose a progressive smoothed image modeling task to train Vision Transformer, which can capture the more discriminable local context of an image while preserving the global semantic information. For efficient inference and deployment in the real world, we innovatively reconstruct the global self-attention structure of the original ViT into isolated band unit self-attention. On ImageNet, under 2% area patch attacks our method achieves 41.70% certified accuracy, a nearly 1-fold increase over the previous best method (26.00%). Simultaneously, our method achieves 78.58% clean accuracy, which is quite close to the normal ResNet-101 accuracy. Extensive experiments show that our method obtains state-of-the-art clean and certified accuracy with inferring efficiently on CIFAR-10 and ImageNet.

摘要: 补丁攻击是敌方实例中最具威胁性的物理攻击形式之一，它可以通过在连续区域内任意修改像素来导致网络误分类。可认证补丁防御可以保证分类器不受补丁攻击影响的健壮性。现有的可认证补丁防御牺牲了分类器的干净准确性，并且仅在玩具数据集上获得了较低的认证准确性。此外，这些方法的清洁准确率和认证准确率仍然明显低于普通分类网络的准确率，这限制了它们在实践中的应用。为了向实用的可认证补丁防御迈进，我们将视觉变换器(VIT)引入随机平滑(DS)的框架中。具体地说，我们提出了一种渐进式平滑图像建模任务来训练视觉转换器，它可以在保持全局语义信息的同时捕捉图像更具区分性的局部上下文。为了在现实世界中有效地推理和部署，我们创新性地将原始VIT的全局自我注意结构重构为孤立的频带单元自我注意。在ImageNet上，在2%的面积补丁攻击下，我们的方法达到了41.70%的认证准确率，比以前最好的方法(26.00%)提高了近1倍。同时，我们的方法达到了78.58%的清洁准确率，相当接近正常的ResNet-101的准确率。大量的实验表明，我们的方法在CIFAR-10和ImageNet上进行有效的推理，获得了最先进的、干净的和经过验证的准确性。



## **4. SHIELD: Defending Textual Neural Networks against Multiple Black-Box Adversarial Attacks with Stochastic Multi-Expert Patcher**

Shield：用随机多专家补丁防御文本神经网络的多黑盒攻击 cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22)

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2011.08908v2)

**Authors**: Thai Le, Noseong Park, Dongwon Lee

**Abstracts**: Even though several methods have proposed to defend textual neural network (NN) models against black-box adversarial attacks, they often defend against a specific text perturbation strategy and/or require re-training the models from scratch. This leads to a lack of generalization in practice and redundant computation. In particular, the state-of-the-art transformer models (e.g., BERT, RoBERTa) require great time and computation resources. By borrowing an idea from software engineering, in order to address these limitations, we propose a novel algorithm, SHIELD, which modifies and re-trains only the last layer of a textual NN, and thus it "patches" and "transforms" the NN into a stochastic weighted ensemble of multi-expert prediction heads. Considering that most of current black-box attacks rely on iterative search mechanisms to optimize their adversarial perturbations, SHIELD confuses the attackers by automatically utilizing different weighted ensembles of predictors depending on the input. In other words, SHIELD breaks a fundamental assumption of the attack, which is a victim NN model remains constant during an attack. By conducting comprehensive experiments, we demonstrate that all of CNN, RNN, BERT, and RoBERTa-based textual NNs, once patched by SHIELD, exhibit a relative enhancement of 15%--70% in accuracy on average against 14 different black-box attacks, outperforming 6 defensive baselines across 3 public datasets. All codes are to be released.

摘要: 尽管已经有几种方法提出了保护文本神经网络(NN)模型免受黑盒攻击的方法，但它们通常防御特定的文本扰动策略和/或需要从头开始重新训练模型。这导致了在实践中缺乏泛化和冗余计算。特别地，最先进的变压器模型(例如，Bert、Roberta)需要大量的时间和计算资源。借鉴软件工程的思想，为了解决这些局限性，我们提出了一种新的算法Shield，它只修改和重新训练文本NN的最后一层，从而将NN“补丁”并“变换”成一个由多专家预测头组成的随机加权集成。“Shield”算法只修改和重新训练文本NN的最后一层，从而将NN“补丁”并“变换”成一个由多专家预测头组成的随机加权集成。考虑到当前大多数黑盒攻击依赖于迭代搜索机制来优化其对抗性扰动，Shield通过根据输入自动利用不同的加权预测器集成来迷惑攻击者。换句话说，盾牌打破了攻击的一个基本假设，即受害者神经网络模型在攻击期间保持不变。通过全面的实验，我们证明了所有基于CNN、RNN、BERT和Roberta的文本神经网络，一旦被Shield修补，对于14种不同的黑盒攻击，准确率平均提高了15%-70%，超过了3个公共数据集的6条防御基线。所有密码都要公布。



## **5. CROP: Certifying Robust Policies for Reinforcement Learning through Functional Smoothing**

CROP：通过函数平滑验证强化学习的健壮策略 cs.LG

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2106.09292v2)

**Authors**: Fan Wu, Linyi Li, Zijian Huang, Yevgeniy Vorobeychik, Ding Zhao, Bo Li

**Abstracts**: As reinforcement learning (RL) has achieved great success and been even adopted in safety-critical domains such as autonomous vehicles, a range of empirical studies have been conducted to improve its robustness against adversarial attacks. However, how to certify its robustness with theoretical guarantees still remains challenging. In this paper, we present the first unified framework CROP (Certifying Robust Policies for RL) to provide robustness certification on both action and reward levels. In particular, we propose two robustness certification criteria: robustness of per-state actions and lower bound of cumulative rewards. We then develop a local smoothing algorithm for policies derived from Q-functions to guarantee the robustness of actions taken along the trajectory; we also develop a global smoothing algorithm for certifying the lower bound of a finite-horizon cumulative reward, as well as a novel local smoothing algorithm to perform adaptive search in order to obtain tighter reward certification. Empirically, we apply CROP to evaluate several existing empirically robust RL algorithms, including adversarial training and different robust regularization, in four environments (two representative Atari games, Highway, and CartPole). Furthermore, by evaluating these algorithms against adversarial attacks, we demonstrate that our certification are often tight. All experiment results are available at website https://crop-leaderboard.github.io.

摘要: 由于强化学习(RL)已经取得了很大的成功，甚至在自动驾驶汽车等安全关键领域得到了应用，人们进行了一系列的实证研究，以提高其对对手攻击的鲁棒性。然而，如何在理论上保证其鲁棒性仍然具有挑战性。在这篇文章中，我们提出了第一个统一的框架CROP(认证RL的健壮策略)，以提供动作和奖励级别的健壮性认证。特别地，我们提出了两个健壮性判定标准：每个状态动作的健壮性和累积奖励的下界。在此基础上，我们提出了基于Q函数的策略的局部平滑算法，以保证沿轨迹采取的动作的鲁棒性；我们还提出了证明有限范围累积奖励下界的全局平滑算法，以及一种新的局部平滑算法，以执行自适应搜索以获得更紧密的奖励证明。经验上，我们应用CROP在四个环境(两个典型的Atari游戏、高速公路和CartPole)中评估了几种现有的经验健壮的RL算法，包括对抗性训练和不同的鲁棒正则化。此外，通过评估这些算法对敌意攻击的抵抗力，我们证明了我们的认证通常是严格的。所有实验结果均可在网站https://crop-leaderboard.github.io.上查阅



## **6. Patch-Fool: Are Vision Transformers Always Robust Against Adversarial Perturbations?**

补丁-愚人：视觉变形金刚在对抗敌方干扰时总是健壮吗？ cs.CV

Accepted at ICLR 2022

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08392v1)

**Authors**: Yonggan Fu, Shunyao Zhang, Shang Wu, Cheng Wan, Yingyan Lin

**Abstracts**: Vision transformers (ViTs) have recently set off a new wave in neural architecture design thanks to their record-breaking performance in various vision tasks. In parallel, to fulfill the goal of deploying ViTs into real-world vision applications, their robustness against potential malicious attacks has gained increasing attention. In particular, recent works show that ViTs are more robust against adversarial attacks as compared with convolutional neural networks (CNNs), and conjecture that this is because ViTs focus more on capturing global interactions among different input/feature patches, leading to their improved robustness to local perturbations imposed by adversarial attacks. In this work, we ask an intriguing question: "Under what kinds of perturbations do ViTs become more vulnerable learners compared to CNNs?" Driven by this question, we first conduct a comprehensive experiment regarding the robustness of both ViTs and CNNs under various existing adversarial attacks to understand the underlying reason favoring their robustness. Based on the drawn insights, we then propose a dedicated attack framework, dubbed Patch-Fool, that fools the self-attention mechanism by attacking its basic component (i.e., a single patch) with a series of attention-aware optimization techniques. Interestingly, our Patch-Fool framework shows for the first time that ViTs are not necessarily more robust than CNNs against adversarial perturbations. In particular, we find that ViTs are more vulnerable learners compared with CNNs against our Patch-Fool attack which is consistent across extensive experiments, and the observations from Sparse/Mild Patch-Fool, two variants of Patch-Fool, indicate an intriguing insight that the perturbation density and strength on each patch seem to be the key factors that influence the robustness ranking between ViTs and CNNs.

摘要: 视觉转换器(VITS)最近掀起了神经结构设计的新浪潮，这要归功于它们在各种视觉任务中创纪录的表现。同时，为了实现将VITS部署到现实世界视觉应用中的目标，VITS对潜在恶意攻击的健壮性也越来越受到关注。特别是最近的研究表明，与卷积神经网络(CNNs)相比，VITS对敌意攻击具有更强的鲁棒性，推测这是因为VITS更注重捕捉不同输入/特征块之间的全局交互，从而提高了它们对敌意攻击造成的局部扰动的鲁棒性。在这项工作中，我们提出了一个耐人寻味的问题：“在什么样的扰动下，VITS比CNN更容易成为学习者？”在这个问题的驱动下，我们首先对VITS和CNN在现有的各种敌意攻击下的健壮性进行了全面的实验，以了解有利于其健壮性的潜在原因。在此基础上，我们提出了一个专门的攻击框架，称为Patch-Fool，它通过使用一系列注意力感知优化技术攻击自我注意机制的基本组成部分(即单个补丁)来愚弄自我注意机制。有趣的是，我们的Patch-Fool框架首次表明，VITS在对抗对手扰动时不一定比CNN更健壮。特别是，我们发现VITS比CNN更容易学习，这在广泛的实验中是一致的，并且来自Patch-Fool的两个变体稀疏/温和Patch-Fool的观察表明，每个补丁上的扰动密度和强度似乎是影响VITS和CNN之间鲁棒性排名的关键因素。



## **7. Synthesis of the Supremal Covert Attacker Against Unknown Supervisors by Using Observations**

利用观测值综合上位隐蔽攻击者对抗未知监督者 eess.SY

**SubmitDate**: 2022-03-16    [paper-pdf](http://arxiv.org/pdf/2203.08360v1)

**Authors**: Ruochen Tai, Liyong Lin, Yuting Zhu, Rong Su

**Abstracts**: In this paper, we consider the problem of synthesizing the supremal covert damage-reachable attacker under the normality assumption, in the setup where the model of the supervisor is unknown to the adversary but the adversary has recorded a (prefix-closed) finite set of observations of the runs of the closed-loop system. The synthesized attacker needs to ensure both the damage-reachability and the covertness against all the supervisors which are consistent with the given set of observations. There is a gap between the de facto supremality, assuming the model of the supervisor is known, and the supremality that can be attained with a limited knowledge of the model of the supervisor, from the adversary's point of view. We consider the setup where the attacker can exercise sensor replacement/deletion attacks and actuator enablement/disablement attacks. The solution methodology proposed in this work is to reduce the synthesis of the supremal covert damage-reachable attacker, given the model of the plant and the finite set of observations, to the synthesis of the supremal safe supervisor for certain transformed plant, which shows the decidability of the observation-assisted covert attacker synthesis problem. The effectiveness of our approach is illustrated on a water tank example adapted from the literature.

摘要: 本文考虑了在正态假设下，在敌方未知监督者模型但敌方记录了闭环系统运行的(前缀闭合的)有限观测集的情况下，综合最高隐蔽损伤可达攻击者的问题，提出了一种基于前缀闭合的闭环系统运行状态综合方法，该问题是在正态分布的假设下，考虑了攻击者未知的监控器模型，但敌手记录了闭环系统运行的(前缀闭合的)有限观测集的情形。合成攻击者需要确保针对所有监督者的损害可达性和隐蔽性，这与给定的观察集合是一致的。在假设监督者的模型已知的情况下，事实上的至高无上与从对手的角度看，通过对监督者的模型的有限了解可以获得的至高无上之间存在差距。我们考虑攻击者可以执行传感器替换/删除攻击和致动器启用/禁用攻击的设置。本文提出的解决方法是，在给定对象模型和有限观测集的情况下，将隐蔽可达攻击者的综合归结为某一变换对象的最高安全监督器的综合，从而证明了观测辅助的隐蔽攻击者综合问题的可判性。从文献中改编的一个水箱例子说明了我们方法的有效性。



## **8. Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation**

利用合成对抗性数据生成提高问答模型稳健性 cs.CL

EMNLP 2021

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2104.08678v3)

**Authors**: Max Bartolo, Tristan Thrush, Robin Jia, Sebastian Riedel, Pontus Stenetorp, Douwe Kiela

**Abstracts**: Despite recent progress, state-of-the-art question answering models remain vulnerable to a variety of adversarial attacks. While dynamic adversarial data collection, in which a human annotator tries to write examples that fool a model-in-the-loop, can improve model robustness, this process is expensive which limits the scale of the collected data. In this work, we are the first to use synthetic adversarial data generation to make question answering models more robust to human adversaries. We develop a data generation pipeline that selects source passages, identifies candidate answers, generates questions, then finally filters or re-labels them to improve quality. Using this approach, we amplify a smaller human-written adversarial dataset to a much larger set of synthetic question-answer pairs. By incorporating our synthetic data, we improve the state-of-the-art on the AdversarialQA dataset by 3.7F1 and improve model generalisation on nine of the twelve MRQA datasets. We further conduct a novel human-in-the-loop evaluation to show that our models are considerably more robust to new human-written adversarial examples: crowdworkers can fool our model only 8.8% of the time on average, compared to 17.6% for a model trained without synthetic data.

摘要: 尽管最近取得了进展，但最先进的问答模型仍然容易受到各种对抗性攻击。虽然动态对抗性数据收集(其中人工注释员试图编写愚弄循环中模型的示例)可以提高模型的健壮性，但此过程代价高昂，从而限制了收集的数据的规模。在这项工作中，我们首次使用合成敌意数据生成来使问答模型对人类对手更健壮。我们开发了一个数据生成管道，它选择源段落，识别候选答案，生成问题，最后对它们进行过滤或重新标记，以提高质量。使用这种方法，我们将较小的人类书写的对抗性数据集放大到更大的合成问答对集。通过整合我们的合成数据，我们将AdversarialQA数据集的最新水平提高了3.7F1，并提高了12个MRQA数据集中9个的模型通用性。我们进一步进行了一项新的人在回路中的评估，以表明我们的模型对新的人类编写的对抗性例子更加健壮：众筹人员平均只有8.8%的时间可以愚弄我们的模型，而没有合成数据训练的模型的这一比例为17.6%。



## **9. Knowledge Enhanced Machine Learning Pipeline against Diverse Adversarial Attacks**

针对不同对手攻击的知识增强型机器学习流水线 cs.LG

International Conference on Machine Learning 2021, 37 pages, 8  figures, 9 tables

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2106.06235v2)

**Authors**: Nezihe Merve Gürel, Xiangyu Qi, Luka Rimanic, Ce Zhang, Bo Li

**Abstracts**: Despite the great successes achieved by deep neural networks (DNNs), recent studies show that they are vulnerable against adversarial examples, which aim to mislead DNNs by adding small adversarial perturbations. Several defenses have been proposed against such attacks, while many of them have been adaptively attacked. In this work, we aim to enhance the ML robustness from a different perspective by leveraging domain knowledge: We propose a Knowledge Enhanced Machine Learning Pipeline (KEMLP) to integrate domain knowledge (i.e., logic relationships among different predictions) into a probabilistic graphical model via first-order logic rules. In particular, we develop KEMLP by integrating a diverse set of weak auxiliary models based on their logical relationships to the main DNN model that performs the target task. Theoretically, we provide convergence results and prove that, under mild conditions, the prediction of KEMLP is more robust than that of the main DNN model. Empirically, we take road sign recognition as an example and leverage the relationships between road signs and their shapes and contents as domain knowledge. We show that compared with adversarial training and other baselines, KEMLP achieves higher robustness against physical attacks, $\mathcal{L}_p$ bounded attacks, unforeseen attacks, and natural corruptions under both whitebox and blackbox settings, while still maintaining high clean accuracy.

摘要: 尽管深度神经网络(DNNs)取得了巨大的成功，但最近的研究表明，它们很容易受到敌意例子的攻击，这些例子的目的是通过添加小的对抗性扰动来误导DNN。针对这类攻击提出了几种防御措施，其中许多都受到了适应性攻击。在这项工作中，我们从不同的角度利用领域知识来增强ML的健壮性：我们提出了一种知识增强型机器学习流水线(KEMLP)，通过一阶逻辑规则将领域知识(即不同预测之间的逻辑关系)集成到概率图形模型中。特别地，我们通过基于它们与执行目标任务的主DNN模型的逻辑关系集成一组不同的弱辅助模型来开发KEMLP。理论上，我们给出了收敛结果，并证明了在温和的条件下，KEMLP的预测比主要的DNN模型更稳健。在经验上，我们以路牌识别为例，利用路牌与其形状和内容之间的关系作为领域知识。结果表明，与对抗性训练和其他基线相比，KEMLP在白盒和黑盒设置下，对物理攻击、数学{L}_p$有界攻击、不可预见攻击和自然破坏都具有更高的鲁棒性，同时仍然保持了较高的清洁准确率。



## **10. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

基于频率驱动的语义相似度潜伏攻击 cs.CV

10 pages, 7 figure, CVPR 2022 conference

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2203.05151v2)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.

摘要: 目前的对抗性攻击研究揭示了基于学习的分类器对精心设计的扰动的脆弱性。然而，现有的大多数攻击方法在跨数据集泛化方面都有固有的局限性，因为它们依赖于具有封闭类别集的分类层。此外，由这些方法产生的扰动可能出现在人类视觉系统(HVS)容易察觉的区域。针对上述问题，我们提出了一种攻击特征表示语义相似度的新算法。通过这种方式，我们能够愚弄分类器，而不会将攻击限制在特定的数据集。对于不可感知性，我们引入了低频约束来限制高频分量内的扰动，以确保对抗性示例与原始示例之间的感知相似性。在三个数据集(CIFAR-10、CIFAR-100和ImageNet-1K)和三个公共在线平台上的广泛实验表明，我们的攻击可以产生跨体系结构和数据集的误导性和可转移的敌意示例。此外，可视化结果和量化性能(根据四个不同的度量)表明，该算法比现有的方法产生更多的不可察觉的扰动。代码可在以下位置获得：。



## **11. Towards Adversarial Control Loops in Sensor Attacks: A Case Study to Control the Kinematics and Actuation of Embedded Systems**

传感器攻击中的对抗性控制回路--以嵌入式系统运动学和执行控制为例 cs.CR

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2203.07670v1)

**Authors**: Yazhou Tu, Sara Rampazzi, Xiali Hei

**Abstracts**: Recent works investigated attacks on sensors by influencing analog sensor components with acoustic, light, and electromagnetic signals. Such attacks can have extensive security, reliability, and safety implications since many types of the targeted sensors are also widely used in critical process control, robotics, automation, and industrial control systems. While existing works advanced our understanding of the physical-level risks that are hidden from a digital-domain perspective, gaps exist in how the attack can be guided to achieve system-level control in real-time, continuous processes. This paper proposes an adversarial control loop-based approach for real-time attacks on control systems relying on sensors. We study how to utilize the system feedback extracted from physical-domain signals to guide the attacks. In the attack process, injection signals are adjusted in real time based on the extracted feedback to exert targeted influence on a victim control system that is continuously affected by the injected perturbations and applying changes to the physical environment. In our case study, we investigate how an external adversarial control system can be constructed over sensor-actuator systems and demonstrate the attacks with program-controlled processes to manipulate the victim system without accessing its internal statuses.

摘要: 最近的工作通过用声、光和电磁信号影响模拟传感器组件来研究对传感器的攻击。这类攻击可能具有广泛的安全性、可靠性和安全性，因为许多类型的目标传感器也广泛用于关键过程控制、机器人、自动化和工业控制系统。虽然现有的工作促进了我们对从数字域角度隐藏的物理级风险的理解，但在如何引导攻击以实现实时、连续过程的系统级控制方面存在差距。针对基于传感器的控制系统实时攻击问题，提出了一种基于对抗性控制回路的方法。我们研究了如何利用从物理域信号中提取的系统反馈来指导攻击。在攻击过程中，基于提取的反馈实时调整注入信号，以对受害者控制系统施加目标影响，该受害者控制系统持续受到注入扰动的影响，并对物理环境施加改变。在我们的案例研究中，我们研究了如何在传感器-执行器系统上构建外部对抗控制系统，并通过程序控制的过程演示了如何在不访问受害者系统内部状态的情况下操纵受害者系统。



## **12. A Regularization Method to Improve Adversarial Robustness of Neural Networks for ECG Signal Classification**

一种提高心电信号分类神经网络对抗鲁棒性的正则化方法 cs.LG

This paper has been published by Computers in Biology and Medicine

**SubmitDate**: 2022-03-15    [paper-pdf](http://arxiv.org/pdf/2110.09759v2)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Electrocardiogram (ECG) is the most widely used diagnostic tool to monitor the condition of the human heart. By using deep neural networks (DNNs), interpretation of ECG signals can be fully automated for the identification of potential abnormalities in a patient's heart in a fraction of a second. Studies have shown that given a sufficiently large amount of training data, DNN accuracy for ECG classification could reach human-expert cardiologist level. However, despite of the excellent performance in classification accuracy, DNNs are highly vulnerable to adversarial noises that are subtle changes in the input of a DNN and may lead to a wrong class-label prediction. It is challenging and essential to improve robustness of DNNs against adversarial noises, which are a threat to life-critical applications. In this work, we proposed a regularization method to improve DNN robustness from the perspective of noise-to-signal ratio (NSR) for the application of ECG signal classification. We evaluated our method on PhysioNet MIT-BIH dataset and CPSC2018 ECG dataset, and the results show that our method can substantially enhance DNN robustness against adversarial noises generated from adversarial attacks, with a minimal change in accuracy on clean data.

摘要: 心电图(ECG)是目前应用最广泛的监测心脏状况的诊断工具。通过使用深度神经网络(DNNs)，心电信号的解释可以完全自动化，以便在几分之一秒内识别患者心脏的潜在异常。研究表明，在训练数据量足够大的情况下，DNN对ECG分类的准确率可以达到人类心脏病专家的水平。然而，尽管DNN在分类精度方面表现优异，但它很容易受到敌意噪声的影响，这些噪声是DNN输入中的细微变化，可能导致错误的类别标签预测。提高DNN对威胁生命的应用程序的敌意噪声的健壮性是具有挑战性的，也是至关重要的。在这项工作中，我们提出了一种正则化方法，以提高DNN的鲁棒性从信噪比(NSR)的角度应用于心电信号分类。我们在PhysioNet MIT-BIH数据集和CPSC2018心电数据集上测试了我们的方法，结果表明，我们的方法可以显著提高DNN对敌方攻击产生的对抗性噪声的鲁棒性，而在干净数据上的准确率变化很小。



## **13. Semantically Distributed Robust Optimization for Vision-and-Language Inference**

面向视觉和语言推理的语义分布式鲁棒优化 cs.CV

Findings of ACL 2022; code available at  https://github.com/ASU-APG/VLI_SDRO

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2110.07165v2)

**Authors**: Tejas Gokhale, Abhishek Chaudhary, Pratyay Banerjee, Chitta Baral, Yezhou Yang

**Abstracts**: Analysis of vision-and-language models has revealed their brittleness under linguistic phenomena such as paraphrasing, negation, textual entailment, and word substitutions with synonyms or antonyms. While data augmentation techniques have been designed to mitigate against these failure modes, methods that can integrate this knowledge into the training pipeline remain under-explored. In this paper, we present \textbf{SDRO}, a model-agnostic method that utilizes a set linguistic transformations in a distributed robust optimization setting, along with an ensembling technique to leverage these transformations during inference. Experiments on benchmark datasets with images (NLVR$^2$) and video (VIOLIN) demonstrate performance improvements as well as robustness to adversarial attacks. Experiments on binary VQA explore the generalizability of this method to other V\&L tasks.

摘要: 通过对视觉与语言模型的分析，揭示了视觉与语言模型在释义、否定、文本蕴涵、同义词或反义词替换等语言现象下的脆性。虽然已经设计了数据增强技术来缓解这些故障模式，但将这些知识集成到训练管道中的方法仍未得到充分探索。在本文中，我们提出了一种与模型无关的方法--textbf{SDRO}，它在分布式鲁棒优化环境中利用一组语言变换，并使用集成技术在推理过程中利用这些变换。在包含图像(NLVR$^2$)和视频(小提琴)的基准数据集上的实验表明，性能有所提高，并且对敌意攻击具有较强的鲁棒性。在二进制VQA上的实验验证了该方法对其他V&L任务的通用性。



## **14. RES-HD: Resilient Intelligent Fault Diagnosis Against Adversarial Attacks Using Hyper-Dimensional Computing**

RES-HD：基于超维计算的抗敌意攻击弹性智能故障诊断 cs.CR

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.08148v1)

**Authors**: Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstracts**: Industrial Internet of Things (I-IoT) enables fully automated production systems by continuously monitoring devices and analyzing collected data. Machine learning methods are commonly utilized for data analytics in such systems. Cyber-attacks are a grave threat to I-IoT as they can manipulate legitimate inputs, corrupting ML predictions and causing disruptions in the production systems. Hyper-dimensional computing (HDC) is a brain-inspired machine learning method that has been shown to be sufficiently accurate while being extremely robust, fast, and energy-efficient. In this work, we use HDC for intelligent fault diagnosis against different adversarial attacks. Our black-box adversarial attacks first train a substitute model and create perturbed test instances using this trained model. These examples are then transferred to the target models. The change in the classification accuracy is measured as the difference before and after the attacks. This change measures the resiliency of a learning method. Our experiments show that HDC leads to a more resilient and lightweight learning solution than the state-of-the-art deep learning methods. HDC has up to 67.5% higher resiliency compared to the state-of-the-art methods while being up to 25.1% faster to train.

摘要: 工业物联网(I-IoT)通过持续监控设备和分析收集的数据，实现了全自动化生产系统。机器学习方法通常用于此类系统中的数据分析。网络攻击是对物联网的严重威胁，因为它们可以操纵合法的输入，破坏ML预测，并导致生产系统中断。超维计算(HDC)是一种受大脑启发的机器学习方法，已被证明具有足够的准确性，同时非常健壮、快速和节能。在这项工作中，我们使用HDC对不同的对手攻击进行智能故障诊断。我们的黑盒对抗性攻击首先训练一个替身模型，并使用这个训练好的模型创建扰动测试实例。然后将这些示例传输到目标模型。分类精度的变化以攻击前和攻击后的差值来衡量。这一变化衡量的是一种学习方法的弹性。我们的实验表明，HDC比现有的深度学习方法具有更强的弹性和轻量级的学习解决方案。与最先进的方法相比，HDC具有高达67.5%的恢复能力，同时培训速度提高了25.1%。



## **15. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

通过内部过度激活分析防御物理可实现的敌意攻击 cs.CV

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07341v1)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.

摘要: 本文提出了一种健壮有效的Z-Mask策略，以提高卷积网络对物理可实现的敌意攻击的鲁棒性。所提出的防御方法依赖于对内部网络特征执行的特定Z得分分析来检测和掩蔽输入图像中与对抗性目标相对应的像素。为此，在浅层和深层检查空间上连续的激活，以暗示潜在的对抗性区域。然后，这些建议通过多阈值机制进行汇总。通过在语义分割和目标检测模型上进行的大量实验，对Z-Mask的有效性进行了评估。使用添加到输入图像的数字补丁和位于真实世界中的打印补丁来执行评估。实验结果表明，Z-Mask在检测准确率和网络整体性能方面均优于目前最先进的方法。额外的实验表明，Z-Mask对可能的防御感知攻击也是健壮的。



## **16. MACER: Attack-free and Scalable Robust Training via Maximizing Certified Radius**

Macer：通过最大化认证半径实现无攻击、可扩展的强健培训 cs.LG

Published in ICLR 2020. 20 Pages

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2001.02378v4)

**Authors**: Runtian Zhai, Chen Dan, Di He, Huan Zhang, Boqing Gong, Pradeep Ravikumar, Cho-Jui Hsieh, Liwei Wang

**Abstracts**: Adversarial training is one of the most popular ways to learn robust models but is usually attack-dependent and time costly. In this paper, we propose the MACER algorithm, which learns robust models without using adversarial training but performs better than all existing provable l2-defenses. Recent work shows that randomized smoothing can be used to provide a certified l2 radius to smoothed classifiers, and our algorithm trains provably robust smoothed classifiers via MAximizing the CErtified Radius (MACER). The attack-free characteristic makes MACER faster to train and easier to optimize. In our experiments, we show that our method can be applied to modern deep neural networks on a wide range of datasets, including Cifar-10, ImageNet, MNIST, and SVHN. For all tasks, MACER spends less training time than state-of-the-art adversarial training algorithms, and the learned models achieve larger average certified radius.

摘要: 对抗性训练是学习健壮模型最流行的方法之一，但通常依赖于攻击且耗时。在本文中，我们提出了Macer算法，该算法无需使用对抗性训练即可学习鲁棒模型，但性能优于所有现有的可证明的L2-防御。最近的工作表明，随机平滑可以用来为平滑分类器提供认证的L2半径，并且我们的算法通过最大化认证半径(Macer)来训练可证明鲁棒的平滑分类器。无攻击特性使Macer训练更快，更容易优化。在我们的实验中，我们的方法可以应用于包括CIFAR-10、ImageNet、MNIST和SVHN在内的各种数据集上的现代深度神经网络。对于所有任务，Macer比最先进的对抗性训练算法花费更少的训练时间，学习的模型获得更大的平均认证半径。



## **17. On the benefits of knowledge distillation for adversarial robustness**

论知识提炼对对手健壮性的益处 cs.LG

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07159v1)

**Authors**: Javier Maroto, Guillermo Ortiz-Jiménez, Pascal Frossard

**Abstracts**: Knowledge distillation is normally used to compress a big network, or teacher, onto a smaller one, the student, by training it to match its outputs. Recently, some works have shown that robustness against adversarial attacks can also be distilled effectively to achieve good rates of robustness on mobile-friendly models. In this work, however, we take a different point of view, and show that knowledge distillation can be used directly to boost the performance of state-of-the-art models in adversarial robustness. In this sense, we present a thorough analysis and provide general guidelines to distill knowledge from a robust teacher and boost the clean and adversarial performance of a student model even further. To that end, we present Adversarial Knowledge Distillation (AKD), a new framework to improve a model's robust performance, consisting on adversarially training a student on a mixture of the original labels and the teacher outputs. Through carefully controlled ablation studies, we show that using early-stopping, model ensembles and weak adversarial training are key techniques to maximize performance of the student, and show that these insights generalize across different robust distillation techniques. Finally, we provide insights on the effect of robust knowledge distillation on the dynamics of the student network, and show that AKD mostly improves the calibration of the network and modify its training dynamics on samples that the model finds difficult to learn, or even memorize.

摘要: 知识提炼通常被用来将一个大的网络或教师压缩成一个较小的网络或教师，通过训练它来与其输出相匹配的一个较小的网络(即学生)。最近的一些工作表明，在移动友好模型上也可以有效地提取对敌意攻击的鲁棒性，以获得良好的鲁棒性。然而，在这项工作中，我们采取了不同的观点，并表明知识蒸馏可以直接用于提高最新模型在对抗鲁棒性方面的性能。在这个意义上，我们提出了一个彻底的分析，并提供了一般性的指导方针，以从一个健壮的教师那里提取知识，并进一步提高学生模型的干净和对抗性表现。为此，我们提出了对抗性知识蒸馏(AKD)，这是一个新的框架来提高模型的鲁棒性能，包括对抗性地训练学生对原始标签和教师输出的混合。通过仔细控制的烧蚀研究，我们表明，使用提前停止、模型集成和弱对抗性训练是最大化学生成绩的关键技术，并表明这些见解可以推广到不同的鲁棒蒸馏技术。最后，我们对鲁棒知识提炼对学生网络动力学的影响进行了深入的研究，结果表明，AKD主要是在模型难以学习甚至记忆的样本上改进网络的校准和修改网络的训练动力学。



## **18. Detection of Electromagnetic Signal Injection Attacks on Actuator Systems**

执行机构系统电磁信号注入攻击的检测 cs.CR

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.07102v1)

**Authors**: Youqian Zhang, Kasper Rasmussen

**Abstracts**: An actuator is a device that converts electricity into another form of energy, typically physical movement. They are absolutely essential for any system that needs to impact or modify the physical world, and are used in millions of systems of all sizes, all over the world, from cars and spacecraft to factory control systems and critical infrastructure. An actuator is a "dumb device" that is entirely controlled by the surrounding electronics, e.g., a microcontroller, and thus cannot authenticate its control signals or do any other form of processing. The problem we look at in this paper is how the wires that connect an actuator to its control electronics can act like antennas, picking up electromagnetic signals from the environment. This makes it possible for a remote attacker to wirelessly inject signals (energy) into these wires to bypass the controller and directly control the actuator.   To detect such attacks, we propose a novel detection method that allows the microcontroller to monitor the control signal and detect attacks as a deviation from the intended value. We have managed to do this without requiring the microcontroller to sample the signal at a high rate or run any signal processing. That makes our defense mechanism practical and easy to integrate into existing systems. Our method is general and applies to any type of actuator (provided a few basic assumptions are met), and can deal with adversaries with arbitrarily high transmission power. We implement our detection method on two different practical systems to show its generality, effectiveness, and robustness.

摘要: 致动器是一种将电能转化为另一种形式的能量的装置，通常是物理运动。它们对于任何需要影响或改变物理世界的系统来说都是绝对必要的，并被用于世界各地数以百万计的各种规模的系统中，从汽车和宇宙飞船到工厂控制系统和关键基础设施。致动器是完全由周围的电子设备(例如微控制器)控制的“哑巴设备”，因此不能验证其控制信号或进行任何其他形式的处理。我们在这篇文章中研究的问题是，连接致动器和其控制电子设备的导线如何像天线一样工作，从环境中接收电磁信号。这使得远程攻击者可以向这些线路无线注入信号(能量)，从而绕过控制器，直接控制执行器。为了检测这类攻击，我们提出了一种新的检测方法，允许微控制器监控控制信号，并将攻击检测为偏离预设值。我们已经设法做到了这一点，而不需要微控制器以高速率采样信号或运行任何信号处理。这使得我们的防御机制切实可行，并且很容易集成到现有系统中。我们的方法是通用的，适用于任何类型的执行器(只要满足一些基本假设)，并且可以对付具有任意高发射功率的对手。我们在两个不同的实际系统上实现了我们的检测方法，以显示其通用性、有效性和鲁棒性。



## **19. Beyond ImageNet Attack: Towards Crafting Adversarial Examples for Black-box Domains**

超越ImageNet攻击：为黑盒领域精心制作敌意示例 cs.CV

Accepted by ICLR 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2201.11528v4)

**Authors**: Qilong Zhang, Xiaodan Li, Yuefeng Chen, Jingkuan Song, Lianli Gao, Yuan He, Hui Xue

**Abstracts**: Adversarial examples have posed a severe threat to deep neural networks due to their transferable nature. Currently, various works have paid great efforts to enhance the cross-model transferability, which mostly assume the substitute model is trained in the same domain as the target model. However, in reality, the relevant information of the deployed model is unlikely to leak. Hence, it is vital to build a more practical black-box threat model to overcome this limitation and evaluate the vulnerability of deployed models. In this paper, with only the knowledge of the ImageNet domain, we propose a Beyond ImageNet Attack (BIA) to investigate the transferability towards black-box domains (unknown classification tasks). Specifically, we leverage a generative model to learn the adversarial function for disrupting low-level features of input images. Based on this framework, we further propose two variants to narrow the gap between the source and target domains from the data and model perspectives, respectively. Extensive experiments on coarse-grained and fine-grained domains demonstrate the effectiveness of our proposed methods. Notably, our methods outperform state-of-the-art approaches by up to 7.71\% (towards coarse-grained domains) and 25.91\% (towards fine-grained domains) on average. Our code is available at \url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.

摘要: 对抗性例子由于其可转移性，对深度神经网络构成了严重的威胁。目前，各种研究都在努力提高模型间的可移植性，大多假设替身模型与目标模型在同一领域进行训练。然而，在现实中，部署的模型的相关信息不太可能泄露。因此，构建一个更实用的黑盒威胁模型来克服这一限制并评估已部署模型的脆弱性是至关重要的。本文在仅知道ImageNet域的情况下，提出了一种超越ImageNet攻击(BIA)来研究向黑盒域(未知分类任务)的可传递性。具体地说，我们利用生成模型来学习破坏输入图像的低层特征的对抗性函数。基于这一框架，我们进一步提出了两种变体，分别从数据和模型的角度来缩小源域和目标域之间的差距。在粗粒域和细粒域上的大量实验证明了我们提出的方法的有效性。值得注意的是，我们的方法平均比最先进的方法高出7.71%(对于粗粒度领域)和25.91%(对于细粒度领域)。我们的代码可在\url{https://github.com/qilong-zhang/Beyond-ImageNet-Attack}.获得



## **20. Data Poisoning Won't Save You From Facial Recognition**

数据中毒不会将你从面部识别中解救出来 cs.LG

ICLR 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2106.14851v2)

**Authors**: Evani Radiya-Dixit, Sanghyun Hong, Nicholas Carlini, Florian Tramèr

**Abstracts**: Data poisoning has been proposed as a compelling defense against facial recognition models trained on Web-scraped pictures. Users can perturb images they post online, so that models will misclassify future (unperturbed) pictures. We demonstrate that this strategy provides a false sense of security, as it ignores an inherent asymmetry between the parties: users' pictures are perturbed once and for all before being published (at which point they are scraped) and must thereafter fool all future models -- including models trained adaptively against the users' past attacks, or models that use technologies discovered after the attack. We evaluate two systems for poisoning attacks against large-scale facial recognition, Fawkes (500'000+ downloads) and LowKey. We demonstrate how an "oblivious" model trainer can simply wait for future developments in computer vision to nullify the protection of pictures collected in the past. We further show that an adversary with black-box access to the attack can (i) train a robust model that resists the perturbations of collected pictures and (ii) detect poisoned pictures uploaded online. We caution that facial recognition poisoning will not admit an "arms race" between attackers and defenders. Once perturbed pictures are scraped, the attack cannot be changed so any future successful defense irrevocably undermines users' privacy.

摘要: 数据中毒已经被提出作为对抗在网络抓取的图片上训练的面部识别模型的一种令人信服的防御措施。用户可以篡改他们在网上发布的图片，这样模特们就会错误地将未来的(未受干扰的)图片分类。我们证明，这种策略提供了一种错误的安全感，因为它忽略了各方之间固有的不对称性：用户的图片在发布之前会一劳永逸地受到干扰(此时它们会被刮掉)，然后必须欺骗所有未来的模型--包括针对用户过去的攻击进行自适应训练的模型，或者使用攻击后发现的技术的模型。我们评估了两个针对大规模面部识别的中毒攻击系统，Fawkes(500‘000+下载量)和LowKey。我们演示了“迟钝”模型训练员如何简单地等待计算机视觉的未来发展，以取消对过去收集的图片的保护。我们进一步表明，拥有黑盒访问权限的攻击者可以(I)训练一个稳健的模型来抵抗收集到的图片的干扰，以及(Ii)检测上传到网上的有毒图片。我们警告说，面部识别中毒不会允许攻击者和防御者之间的“军备竞赛”。一旦被扰乱的图片被刮掉，攻击就无法改变，因此未来任何成功的防御都将不可挽回地破坏用户的隐私。



## **21. Efficient universal shuffle attack for visual object tracking**

一种高效的视觉目标跟踪通用混洗攻击 cs.CV

accepted for ICASSP 2022

**SubmitDate**: 2022-03-14    [paper-pdf](http://arxiv.org/pdf/2203.06898v1)

**Authors**: Siao Liu, Zhaoyu Chen, Wei Li, Jiwei Zhu, Jiafeng Wang, Wenqiang Zhang, Zhongxue Gan

**Abstracts**: Recently, adversarial attacks have been applied in visual object tracking to deceive deep trackers by injecting imperceptible perturbations into video frames. However, previous work only generates the video-specific perturbations, which restricts its application scenarios. In addition, existing attacks are difficult to implement in reality due to the real-time of tracking and the re-initialization mechanism. To address these issues, we propose an offline universal adversarial attack called Efficient Universal Shuffle Attack. It takes only one perturbation to cause the tracker malfunction on all videos. To improve the computational efficiency and attack performance, we propose a greedy gradient strategy and a triple loss to efficiently capture and attack model-specific feature representations through the gradients. Experimental results show that EUSA can significantly reduce the performance of state-of-the-art trackers on OTB2015 and VOT2018.

摘要: 近年来，对抗性攻击被应用于视觉目标跟踪中，通过在视频帧中注入不可察觉的扰动来欺骗深度跟踪者。然而，以往的工作只产生特定于视频的扰动，这限制了它的应用场景。此外，由于跟踪的实时性和重新初始化机制，现有的攻击在现实中很难实现。为了解决这些问题，我们提出了一种称为高效通用洗牌攻击的离线通用对抗性攻击。只需一次扰动即可导致所有视频的跟踪器故障。为了提高计算效率和攻击性能，我们提出了贪婪梯度策略和三重损失策略，通过梯度有效地捕获和攻击特定于模型的特征表示。实验结果表明，在OTB2015和VOT2018上，EUSA可以显著降低最新跟踪器的性能。



## **22. Generating Practical Adversarial Network Traffic Flows Using NIDSGAN**

利用NIDSGAN生成实用的对抗性网络流量 cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06694v1)

**Authors**: Bolor-Erdene Zolbayar, Ryan Sheatsley, Patrick McDaniel, Michael J. Weisman, Sencun Zhu, Shitong Zhu, Srikanth Krishnamurthy

**Abstracts**: Network intrusion detection systems (NIDS) are an essential defense for computer networks and the hosts within them. Machine learning (ML) nowadays predominantly serves as the basis for NIDS decision making, where models are tuned to reduce false alarms, increase detection rates, and detect known and unknown attacks. At the same time, ML models have been found to be vulnerable to adversarial examples that undermine the downstream task. In this work, we ask the practical question of whether real-world ML-based NIDS can be circumvented by crafted adversarial flows, and if so, how can they be created. We develop the generative adversarial network (GAN)-based attack algorithm NIDSGAN and evaluate its effectiveness against realistic ML-based NIDS. Two main challenges arise for generating adversarial network traffic flows: (1) the network features must obey the constraints of the domain (i.e., represent realistic network behavior), and (2) the adversary must learn the decision behavior of the target NIDS without knowing its model internals (e.g., architecture and meta-parameters) and training data. Despite these challenges, the NIDSGAN algorithm generates highly realistic adversarial traffic flows that evade ML-based NIDS. We evaluate our attack algorithm against two state-of-the-art DNN-based NIDS in whitebox, blackbox, and restricted-blackbox threat models and achieve success rates which are on average 99%, 85%, and 70%, respectively. We also show that our attack algorithm can evade NIDS based on classical ML models including logistic regression, SVM, decision trees and KNNs, with a success rate of 70% on average. Our results demonstrate that deploying ML-based NIDS without careful defensive strategies against adversarial flows may (and arguably likely will) lead to future compromises.

摘要: 网络入侵检测系统(NIDS)是对计算机网络及其内部主机进行必要的防御。如今，机器学习(ML)主要作为NIDS决策的基础，调整模型以减少错误警报，提高检测率，并检测已知和未知的攻击。同时，ML模型被发现容易受到破坏下游任务的对抗性例子的影响。在这项工作中，我们提出了一个实际问题，即基于ML的真实世界的NIDS是否可以通过精心设计的敌意流来规避，如果可以，如何创建它们。我们开发了基于产生式对抗网络(GAN)的攻击算法NIDSGAN，并对其对现实的基于ML的网络入侵检测系统进行了有效性评估。产生敌意网络流量面临两个主要挑战：(1)网络特征必须服从域的约束(即，表示真实的网络行为)；(2)敌手必须在不知道目标NIDS的模型内部(例如，体系结构和元参数)和训练数据的情况下学习目标NIDS的决策行为。尽管存在这些挑战，NIDSGAN算法生成的高度逼真的敌意流量可以避开基于ML的NIDS。我们在白盒、黑盒和受限黑盒威胁模型中对两种最新的基于DNN的网络入侵检测系统进行了测试，平均成功率分别为99%、85%和70%。实验还表明，该攻击算法可以规避基于Logistic回归、支持向量机、决策树和KNNs等经典ML模型的网络入侵检测系统，平均成功率为70%。我们的结果表明，部署基于ML的NIDS而没有针对敌意流量的谨慎防御策略可能(而且很可能会)导致未来的妥协。



## **23. LAS-AT: Adversarial Training with Learnable Attack Strategy**

LAS-AT：采用可学习攻击策略的对抗性训练 cs.CV

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06616v1)

**Authors**: Xiaojun Jia, Yong Zhang, Baoyuan Wu, Ke Ma, Jue Wang, Xiaochun Cao

**Abstracts**: Adversarial training (AT) is always formulated as a minimax problem, of which the performance depends on the inner optimization that involves the generation of adversarial examples (AEs). Most previous methods adopt Projected Gradient Decent (PGD) with manually specifying attack parameters for AE generation. A combination of the attack parameters can be referred to as an attack strategy. Several works have revealed that using a fixed attack strategy to generate AEs during the whole training phase limits the model robustness and propose to exploit different attack strategies at different training stages to improve robustness. But those multi-stage hand-crafted attack strategies need much domain expertise, and the robustness improvement is limited. In this paper, we propose a novel framework for adversarial training by introducing the concept of "learnable attack strategy", dubbed LAS-AT, which learns to automatically produce attack strategies to improve the model robustness. Our framework is composed of a target network that uses AEs for training to improve robustness and a strategy network that produces attack strategies to control the AE generation. Experimental evaluations on three benchmark databases demonstrate the superiority of the proposed method. The code is released at https://github.com/jiaxiaojunQAQ/LAS-AT.

摘要: 对抗性训练(AT)通常被描述为一个极小极大问题，其性能取决于对抗性实例(AES)生成过程中的内部优化。以往的方法大多采用人工指定攻击参数的投影梯度下降(PGD)方法来生成AE。攻击参数的组合可以称为攻击策略。一些工作表明，在整个训练阶段使用固定的攻击策略来生成AES限制了模型的健壮性，并提出在不同的训练阶段采用不同的攻击策略来提高健壮性。但这些多阶段手工设计的攻击策略需要较多的领域专业知识，健壮性提高有限。本文通过引入“可学习攻击策略”(LAS-AT)的概念，提出了一种新的对抗性训练框架，该框架通过学习自动生成攻击策略来提高模型的健壮性。我们的框架由使用AE进行训练以提高健壮性的目标网络和产生攻击策略以控制AE生成的策略网络组成。在三个基准数据库上的实验评估表明了该方法的优越性。该代码在https://github.com/jiaxiaojunQAQ/LAS-AT.上发布



## **24. One Parameter Defense -- Defending against Data Inference Attacks via Differential Privacy**

单参数防御--基于差分隐私的数据推理攻击防御 cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06580v1)

**Authors**: Dayong Ye, Sheng Shen, Tianqing Zhu, Bo Liu, Wanlei Zhou

**Abstracts**: Machine learning models are vulnerable to data inference attacks, such as membership inference and model inversion attacks. In these types of breaches, an adversary attempts to infer a data record's membership in a dataset or even reconstruct this data record using a confidence score vector predicted by the target model. However, most existing defense methods only protect against membership inference attacks. Methods that can combat both types of attacks require a new model to be trained, which may not be time-efficient. In this paper, we propose a differentially private defense method that handles both types of attacks in a time-efficient manner by tuning only one parameter, the privacy budget. The central idea is to modify and normalize the confidence score vectors with a differential privacy mechanism which preserves privacy and obscures membership and reconstructed data. Moreover, this method can guarantee the order of scores in the vector to avoid any loss in classification accuracy. The experimental results show the method to be an effective and timely defense against both membership inference and model inversion attacks with no reduction in accuracy.

摘要: 机器学习模型容易受到数据推理攻击，如成员关系推理和模型反转攻击。在这些类型的入侵中，对手试图推断数据记录在数据集中的成员资格，或者甚至使用由目标模型预测的置信度分数向量来重建该数据记录。然而，现有的防御方法大多只针对成员关系推理攻击。可以对抗这两种类型的攻击的方法需要训练新的模型，这可能不是很省时。在这篇文章中，我们提出了一种差分隐私防御方法，通过只调整一个参数，即隐私预算，以高效的方式处理这两种类型的攻击。其核心思想是利用差分隐私机制对置信度向量进行修改和归一化，该机制保护隐私，模糊成员资格和重建数据。此外，该方法还可以保证向量中分数的排序，避免了分类精度的损失。实验结果表明，该方法在不降低准确率的前提下，对隶属度推理和模型反转攻击都具有较好的防御效果。



## **25. Model Inversion Attack against Transfer Learning: Inverting a Model without Accessing It**

针对迁移学习的模型倒置攻击：在不访问模型的情况下倒置模型 cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06570v1)

**Authors**: Dayong Ye, Huiqiang Chen, Shuai Zhou, Tianqing Zhu, Wanlei Zhou, Shouling Ji

**Abstracts**: Transfer learning is an important approach that produces pre-trained teacher models which can be used to quickly build specialized student models. However, recent research on transfer learning has found that it is vulnerable to various attacks, e.g., misclassification and backdoor attacks. However, it is still not clear whether transfer learning is vulnerable to model inversion attacks. Launching a model inversion attack against transfer learning scheme is challenging. Not only does the student model hide its structural parameters, but it is also inaccessible to the adversary. Hence, when targeting a student model, both the white-box and black-box versions of existing model inversion attacks fail. White-box attacks fail as they need the target model's parameters. Black-box attacks fail as they depend on making repeated queries of the target model. However, they may not mean that transfer learning models are impervious to model inversion attacks. Hence, with this paper, we initiate research into model inversion attacks against transfer learning schemes with two novel attack methods. Both are black-box attacks, suiting different situations, that do not rely on queries to the target student model. In the first method, the adversary has the data samples that share the same distribution as the training set of the teacher model. In the second method, the adversary does not have any such samples. Experiments show that highly recognizable data records can be recovered with both of these methods. This means that even if a model is an inaccessible black-box, it can still be inverted.

摘要: 迁移学习是产生预训教师模型的重要途径，可用于快速构建专业学生模型。然而，最近关于迁移学习的研究发现，迁移学习很容易受到各种攻击，例如错误分类和后门攻击。然而，目前还不清楚转移学习是否容易受到模型反转攻击。针对迁移学习方案发起模型反转攻击具有挑战性。学生模型不仅隐藏了其结构参数，而且对手也无法访问它。因此，当以学生模型为目标时，现有模型反转攻击的白盒和黑盒版本都会失败。白盒攻击失败，因为它们需要目标模型的参数。黑盒攻击失败，因为它们依赖于重复查询目标模型。然而，它们可能并不意味着迁移学习模型不受模型反转攻击的影响。因此，本文利用两种新的攻击方法对迁移学习方案进行了模型反转攻击的研究。这两种攻击都是黑箱攻击，适合不同的情况，不依赖于对目标学生模型的查询。在第一种方法中，对手的数据样本与教师模型的训练集具有相同的分布。在第二种方法中，对手没有任何这样的样本。实验表明，这两种方法都可以恢复高度可识别性的数据记录。这意味着，即使一个模型是一个无法接近的黑匣子，它仍然可以倒置。



## **26. Query-Efficient Black-box Adversarial Attacks Guided by a Transfer-based Prior**

基于转移的先验引导的查询高效黑盒对抗攻击 cs.LG

Accepted by IEEE Transactions on Pattern Recognition and Machine  Intelligence (TPAMI). The official version is at  https://ieeexplore.ieee.org/document/9609659

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06560v1)

**Authors**: Yinpeng Dong, Shuyu Cheng, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Adversarial attacks have been extensively studied in recent years since they can identify the vulnerability of deep learning models before deployed. In this paper, we consider the black-box adversarial setting, where the adversary needs to craft adversarial examples without access to the gradients of a target model. Previous methods attempted to approximate the true gradient either by using the transfer gradient of a surrogate white-box model or based on the feedback of model queries. However, the existing methods inevitably suffer from low attack success rates or poor query efficiency since it is difficult to estimate the gradient in a high-dimensional input space with limited information. To address these problems and improve black-box attacks, we propose two prior-guided random gradient-free (PRGF) algorithms based on biased sampling and gradient averaging, respectively. Our methods can take the advantage of a transfer-based prior given by the gradient of a surrogate model and the query information simultaneously. Through theoretical analyses, the transfer-based prior is appropriately integrated with model queries by an optimal coefficient in each method. Extensive experiments demonstrate that, in comparison with the alternative state-of-the-arts, both of our methods require much fewer queries to attack black-box models with higher success rates.

摘要: 近年来，对抗性攻击得到了广泛的研究，因为它们可以在部署前识别深度学习模型的脆弱性。在本文中，我们考虑了黑箱对抗性设置，其中对手需要制作对抗性示例，而不需要访问目标模型的梯度。以前的方法试图通过使用代理白盒模型的传递梯度或基于模型查询的反馈来近似真实梯度。然而，现有的方法不可避免地存在攻击成功率低或查询效率低的问题，因为很难在信息有限的高维输入空间中估计梯度。为了解决这些问题并改进黑盒攻击，我们提出了两种先验引导的随机无梯度(PRGF)算法，分别基于有偏采样和梯度平均。我们的方法可以同时利用代理模型的梯度和查询信息给出的基于转移的先验。通过理论分析，每种方法都通过一个最优系数将基于转移的先验与模型查询适当地结合在一起。大量的实验表明，与其他最先进的方法相比，我们的两种方法都需要更少的查询来攻击黑盒模型，成功率更高。



## **27. Label-only Model Inversion Attack: The Attack that Requires the Least Information**

仅标签模型反转攻击：需要最少信息的攻击 cs.CR

**SubmitDate**: 2022-03-13    [paper-pdf](http://arxiv.org/pdf/2203.06555v1)

**Authors**: Dayong Ye, Tianqing Zhu, Shuai Zhou, Bo Liu, Wanlei Zhou

**Abstracts**: In a model inversion attack, an adversary attempts to reconstruct the data records, used to train a target model, using only the model's output. In launching a contemporary model inversion attack, the strategies discussed are generally based on either predicted confidence score vectors, i.e., black-box attacks, or the parameters of a target model, i.e., white-box attacks. However, in the real world, model owners usually only give out the predicted labels; the confidence score vectors and model parameters are hidden as a defense mechanism to prevent such attacks. Unfortunately, we have found a model inversion method that can reconstruct the input data records based only on the output labels. We believe this is the attack that requires the least information to succeed and, therefore, has the best applicability. The key idea is to exploit the error rate of the target model to compute the median distance from a set of data records to the decision boundary of the target model. The distance, then, is used to generate confidence score vectors which are adopted to train an attack model to reconstruct the data records. The experimental results show that highly recognizable data records can be reconstructed with far less information than existing methods.

摘要: 在模型反转攻击中，攻击者试图仅使用模型的输出来重建用于训练目标模型的数据记录。在发起当代模型反转攻击时，所讨论的策略通常要么基于预测的置信度向量，即黑盒攻击，要么基于目标模型的参数，即白盒攻击。然而，在现实世界中，模型所有者通常只给出预测的标签，置信度得分向量和模型参数被隐藏起来，作为一种防御机制来防止此类攻击。不幸的是，我们已经找到了一种模型反演方法，它可以仅基于输出标签来重建输入数据记录。我们认为，这是需要最少信息才能成功的攻击，因此具有最好的适用性。其核心思想是利用目标模型的错误率来计算一组数据记录到目标模型决策边界的中值距离。然后，该距离被用来生成置信度得分向量，该置信得分向量被用来训练攻击模型以重构数据记录。实验结果表明，与现有方法相比，该方法可以用更少的信息来重建高度可识别的数据记录。



## **28. Mal2GCN: A Robust Malware Detection Approach Using Deep Graph Convolutional Networks With Non-Negative Weights**

Mal2GCN：一种基于非负权深图卷积网络的鲁棒恶意软件检测方法 cs.CR

13 pages, 12 figures, 5 tables

**SubmitDate**: 2022-03-12    [paper-pdf](http://arxiv.org/pdf/2108.12473v2)

**Authors**: Omid Kargarnovin, Amir Mahdi Sadeghzadeh, Rasool Jalili

**Abstracts**: With the growing pace of using Deep Learning (DL) to solve various problems, securing these models against adversaries has become one of the main concerns of researchers. Recent studies have shown that DL-based malware detectors are vulnerable to adversarial examples. An adversary can create carefully crafted adversarial examples to evade DL-based malware detectors. In this paper, we propose Mal2GCN, a robust malware detection model that uses Function Call Graph (FCG) representation of executable files combined with Graph Convolution Network (GCN) to detect Windows malware. Since FCG representation of executable files is more robust than raw byte sequence representation, numerous proposed adversarial example generating methods are ineffective in evading Mal2GCN. Moreover, we use the non-negative training method to transform Mal2GCN to a monotonically non-decreasing function; thereby, it becomes theoretically robust against appending attacks. We then present a black-box source code-based adversarial malware generation approach that can be used to evaluate the robustness of malware detection models against real-world adversaries. The proposed approach injects adversarial codes into the various locations of malware source codes to evade malware detection models. The experiments demonstrate that Mal2GCN with non-negative weights has high accuracy in detecting Windows malware, and it is also robust against adversarial attacks that add benign features to the Malware source code.

摘要: 随着利用深度学习(DL)来解决各种问题的速度越来越快，保护这些模型不受攻击已成为研究人员主要关注的问题之一。最近的研究表明，基于DL的恶意软件检测器容易受到敌意示例的攻击。敌手可以创建精心设计的敌意示例，以躲避基于DL的恶意软件检测器。本文提出了一种健壮的恶意软件检测模型Mal2GCN，该模型利用可执行文件的函数调用图(FCG)表示法和图卷积网络(GCN)相结合的方法检测Windows恶意软件。由于可执行文件的FCG表示比原始字节序列表示更健壮，许多已提出的敌意示例生成方法在规避Mal2GCN方面是无效的。此外，我们使用非负训练方法将Mal2GCN转化为单调非减函数，从而使其在理论上对附加攻击具有较强的鲁棒性。然后，我们提出了一种基于黑盒源代码的恶意软件生成方法，该方法可以用来评估恶意软件检测模型对真实攻击的健壮性。该方法将恶意代码注入恶意软件源代码的各个位置，以逃避恶意软件检测模型。实验表明，具有非负权重的Mal2GCN对检测Windows恶意软件具有较高的准确率，并且对恶意软件源代码中添加良性特征的恶意攻击具有较强的鲁棒性。



## **29. A Survey in Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御与健壮性研究综述 cs.CL

**SubmitDate**: 2022-03-12    [paper-pdf](http://arxiv.org/pdf/2203.06414v1)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstracts**: In recent years, it has been seen that deep neural networks are lacking robustness and are likely to break in case of adversarial perturbations in input data. Strong adversarial attacks are proposed by various authors for computer vision and Natural Language Processing (NLP). As a counter-effort, several defense mechanisms are also proposed to save these networks from failing. In contrast with image data, generating adversarial attacks and defending these models is not easy in NLP because of the discrete nature of the text data. However, numerous methods for adversarial defense are proposed of late, for different NLP tasks such as text classification, named entity recognition, natural language inferencing, etc. These methods are not just used for defending neural networks from adversarial attacks, but also used as a regularization mechanism during training, saving the model from overfitting. The proposed survey is an attempt to review different methods proposed for adversarial defenses in NLP in the recent past by proposing a novel taxonomy. This survey also highlights the fragility of the advanced deep neural networks in NLP and the challenges in defending them.

摘要: 近年来，人们已经看到，深度神经网络缺乏健壮性，在输入数据受到对抗性扰动的情况下很可能会崩溃。强对抗性攻击是针对计算机视觉和自然语言处理(NLP)提出的一种新的攻击方法。作为反击，还提出了几种防御机制，以避免这些网络出现故障。与图像数据相比，由于文本数据的离散性，在自然语言处理中生成敌意攻击并对这些模型进行防御并非易事。然而，最近针对不同的NLP任务，如文本分类、命名实体识别、自然语言推理等，提出了大量的对抗性防御方法，这些方法不仅用于保护神经网络免受对抗性攻击，而且在训练过程中还作为一种正则化机制，避免了模型的过度拟合。这项拟议的调查试图通过提出一种新的分类法来回顾最近在NLP中提出的不同的对抗性防御方法。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及在防御它们方面的挑战。



## **30. Detecting CAN Masquerade Attacks with Signal Clustering Similarity**

利用信号聚类相似度检测CAN伪装攻击 cs.CR

8 pages, 5 figures, 3 tables

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2201.02665v2)

**Authors**: Pablo Moriano, Robert A. Bridges, Michael D. Iannacone

**Abstracts**: Vehicular Controller Area Networks (CANs) are susceptible to cyber attacks of different levels of sophistication. Fabrication attacks are the easiest to administer -- an adversary simply sends (extra) frames on a CAN -- but also the easiest to detect because they disrupt frame frequency. To overcome time-based detection methods, adversaries must administer masquerade attacks by sending frames in lieu of (and therefore at the expected time of) benign frames but with malicious payloads. Research efforts have proven that CAN attacks, and masquerade attacks in particular, can affect vehicle functionality. Examples include causing unintended acceleration, deactivation of vehicle's brakes, as well as steering the vehicle. We hypothesize that masquerade attacks modify the nuanced correlations of CAN signal time series and how they cluster together. Therefore, changes in cluster assignments should indicate anomalous behavior. We confirm this hypothesis by leveraging our previously developed capability for reverse engineering CAN signals (i.e., CAN-D [Controller Area Network Decoder]) and focus on advancing the state of the art for detecting masquerade attacks by analyzing time series extracted from raw CAN frames. Specifically, we demonstrate that masquerade attacks can be detected by computing time series clustering similarity using hierarchical clustering on the vehicle's CAN signals (time series) and comparing the clustering similarity across CAN captures with and without attacks. We test our approach in a previously collected CAN dataset with masquerade attacks (i.e., the ROAD dataset) and develop a forensic tool as a proof of concept to demonstrate the potential of the proposed approach for detecting CAN masquerade attacks.

摘要: 车辆控制器局域网(CAN)容易受到不同复杂程度的网络攻击。伪造攻击是最容易管理的--对手只是在CAN上发送(额外的)帧--但也是最容易检测到的，因为它们扰乱了帧频率。要克服基于时间的检测方法，攻击者必须通过发送帧来管理伪装攻击，而不是(因此在预期时间发送)良性帧，但带有恶意有效负载。研究工作已经证明，CAN攻击，特别是伪装攻击，会影响车辆的功能。例如，造成意外加速，车辆刹车失灵，以及驾驶车辆。我们假设伪装攻击修改了CAN信号时间序列的细微差别相关性，以及它们是如何聚集在一起的。因此，群集分配中的更改应该表示异常行为。我们利用我们之前开发的CAN信号逆向工程能力(即CAN-D[Controller Area Network Decoder])确认了这一假设，并通过分析从原始CAN帧中提取的时间序列，专注于提高检测伪装攻击的技术水平。具体地说，我们通过对车辆CAN信号(时间序列)进行层次聚类来计算时间序列聚类相似度，并比较有攻击和无攻击的CAN捕获的聚类相似度，从而实现伪装攻击的检测。我们在以前收集的带有伪装攻击的CAN数据集(即道路数据集)上测试了我们的方法，并开发了一个取证工具作为概念证明，以展示所提出的方法检测CAN伪装攻击的潜力。



## **31. On Adversarial Robustness of Trajectory Prediction for Autonomous Vehicles**

自主车辆轨迹预测的对抗鲁棒性研究 cs.CV

13 pages, 13 figures, accepted by CVPR 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2201.05057v2)

**Authors**: Qingzhao Zhang, Shengtuo Hu, Jiachen Sun, Qi Alfred Chen, Z. Morley Mao

**Abstracts**: Trajectory prediction is a critical component for autonomous vehicles (AVs) to perform safe planning and navigation. However, few studies have analyzed the adversarial robustness of trajectory prediction or investigated whether the worst-case prediction can still lead to safe planning. To bridge this gap, we study the adversarial robustness of trajectory prediction models by proposing a new adversarial attack that perturbs normal vehicle trajectories to maximize the prediction error. Our experiments on three models and three datasets show that the adversarial prediction increases the prediction error by more than 150%. Our case studies show that if an adversary drives a vehicle close to the target AV following the adversarial trajectory, the AV may make an inaccurate prediction and even make unsafe driving decisions. We also explore possible mitigation techniques via data augmentation and trajectory smoothing. The implementation is open source at https://github.com/zqzqz/AdvTrajectoryPrediction.

摘要: 轨迹预测是自动驾驶车辆进行安全规划和导航的重要组成部分。然而，很少有研究分析弹道预测的对抗稳健性，或研究最坏情况的预测是否仍能导致安全规划。为了弥补这一差距，我们研究了轨迹预测模型的对抗性，提出了一种新的对抗性攻击，通过扰动正常的车辆轨迹来最大化预测误差。在三个模型和三个数据集上的实验表明，对抗性预测使预测误差增加了150%以上。我们的案例研究表明，如果对手沿着敌对的轨迹驾驶车辆接近目标AV，AV可能会做出不准确的预测，甚至做出不安全的驾驶决策。我们还通过数据增强和轨迹平滑来探索可能的缓解技术。该实现在https://github.com/zqzqz/AdvTrajectoryPrediction.上是开源的



## **32. Sparse Black-box Video Attack with Reinforcement Learning**

基于强化学习的稀疏黑盒视频攻击 cs.CV

Accepted at IJCV 2022

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2001.03754v3)

**Authors**: Xingxing Wei, Huanqian Yan, Bo Li

**Abstracts**: Adversarial attacks on video recognition models have been explored recently. However, most existing works treat each video frame equally and ignore their temporal interactions. To overcome this drawback, a few methods try to select some key frames and then perform attacks based on them. Unfortunately, their selection strategy is independent of the attacking step, therefore the resulting performance is limited. Instead, we argue the frame selection phase is closely relevant with the attacking phase. The key frames should be adjusted according to the attacking results. For that, we formulate the black-box video attacks into a Reinforcement Learning (RL) framework. Specifically, the environment in RL is set as the recognition model, and the agent in RL plays the role of frame selecting. By continuously querying the recognition models and receiving the attacking feedback, the agent gradually adjusts its frame selection strategy and adversarial perturbations become smaller and smaller. We conduct a series of experiments with two mainstream video recognition models: C3D and LRCN on the public UCF-101 and HMDB-51 datasets. The results demonstrate that the proposed method can significantly reduce the adversarial perturbations with efficient query times.

摘要: 最近，针对视频识别模型的对抗性攻击已经被探索出来。然而，现有的大多数工作都将每个视频帧一视同仁地对待，而忽略了它们之间的时间交互。为了克服这一缺点，有几种方法试图选择一些关键帧，然后根据这些关键帧进行攻击。不幸的是，它们的选择策略与攻击步骤无关，因此所产生的性能是有限的。相反，我们认为帧选择阶段与攻击阶段密切相关。应根据攻击结果调整关键帧。为此，我们将黑盒视频攻击描述为强化学习(RL)框架。具体地说，RL中的环境被设置为识别模型，RL中的Agent起到框架选择的作用。通过不断查询识别模型和接收攻击反馈，Agent逐渐调整其帧选择策略，敌方扰动变得越来越小。我们在公开的UCF-101和HMDB-51数据集上用两种主流的视频识别模型C3D和LRCN进行了一系列的实验。实验结果表明，该方法可以有效地减少对抗性扰动，提高查询效率。



## **33. Block-Sparse Adversarial Attack to Fool Transformer-Based Text Classifiers**

对基于愚人转换器的文本分类器的挡路稀疏敌意攻击 cs.CL

ICASSP 2022, Code available at:  https://github.com/sssadrizadeh/transformer-text-classifier-attack

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05948v1)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstracts**: Recently, it has been shown that, in spite of the significant performance of deep neural networks in different fields, those are vulnerable to adversarial examples. In this paper, we propose a gradient-based adversarial attack against transformer-based text classifiers. The adversarial perturbation in our method is imposed to be block-sparse so that the resultant adversarial example differs from the original sentence in only a few words. Due to the discrete nature of textual data, we perform gradient projection to find the minimizer of our proposed optimization problem. Experimental results demonstrate that, while our adversarial attack maintains the semantics of the sentence, it can reduce the accuracy of GPT-2 to less than 5% on different datasets (AG News, MNLI, and Yelp Reviews). Furthermore, the block-sparsity constraint of the proposed optimization problem results in small perturbations in the adversarial example.

摘要: 最近的研究表明，尽管深度神经网络在不同的领域有着显著的性能，但它们很容易受到敌意例子的影响。本文针对基于变换的文本分类器提出了一种基于梯度的对抗性攻击方法。我们方法中的对抗性扰动被强加为挡路稀疏的，这样得到的对抗性示例与原始句子只有几个字的不同。由于文本数据的离散性，我们使用梯度投影来寻找我们所提出的优化问题的最小值。实验结果表明，我们的对抗性攻击在保持句子语义的同时，可以将GPT-2在不同数据集(AG News、MNLI和Yelp Reviews)上的准确率降低到5%以下。此外，所提出的优化问题的挡路稀疏性约束导致了对抗性例子中的微小扰动。



## **34. Learning from Attacks: Attacking Variational Autoencoder for Improving Image Classification**

从攻击中学习：攻击改进图像分类的变分自动编码器 cs.LG

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.07027v1)

**Authors**: Jianzhang Zheng, Fan Yang, Hao Shen, Xuan Tang, Mingsong Chen, Liang Song, Xian Wei

**Abstracts**: Adversarial attacks are often considered as threats to the robustness of Deep Neural Networks (DNNs). Various defending techniques have been developed to mitigate the potential negative impact of adversarial attacks against task predictions. This work analyzes adversarial attacks from a different perspective. Namely, adversarial examples contain implicit information that is useful to the predictions i.e., image classification, and treat the adversarial attacks against DNNs for data self-expression as extracted abstract representations that are capable of facilitating specific learning tasks. We propose an algorithmic framework that leverages the advantages of the DNNs for data self-expression and task-specific predictions, to improve image classification. The framework jointly learns a DNN for attacking Variational Autoencoder (VAE) networks and a DNN for classification, coined as Attacking VAE for Improve Classification (AVIC). The experiment results show that AVIC can achieve higher accuracy on standard datasets compared to the training with clean examples and the traditional adversarial training.

摘要: 敌意攻击通常被认为是对深度神经网络(DNNs)健壮性的威胁。已经开发了各种防御技术来减轻针对任务预测的对抗性攻击的潜在负面影响。这项工作从不同的角度分析了对抗性攻击。也就是说，对抗性示例包含对预测(即图像分类)有用的隐含信息，并且将针对数据自我表达的对DNN的对抗性攻击视为能够促进特定学习任务的提取的抽象表示。我们提出了一个算法框架，利用DNNs在数据自我表达和特定任务预测方面的优势，来改进图像分类。该框架联合学习用于攻击变分自动编码器(VAE)网络的DNN和用于分类的DNN，称为用于改进分类的攻击VAE(AVIC)。实验结果表明，AVIC算法在标准数据集上的分类准确率高于干净样本训练和传统的对抗性训练。



## **35. Reinforcement Learning for Linear Quadratic Control is Vulnerable Under Cost Manipulation**

线性二次控制的强化学习在成本操纵下易受攻击 eess.SY

This paper is yet to be peer-reviewed

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05774v1)

**Authors**: Yunhan Huang, Quanyan Zhu

**Abstracts**: In this work, we study the deception of a Linear-Quadratic-Gaussian (LQG) agent by manipulating the cost signals. We show that a small falsification on the cost parameters will only lead to a bounded change in the optimal policy and the bound is linear on the amount of falsification the attacker can apply on the cost parameters. We propose an attack model where the goal of the attacker is to mislead the agent into learning a `nefarious' policy with intended falsification on the cost parameters. We formulate the attack's problem as an optimization problem, which is proved to be convex, and developed necessary and sufficient conditions to check the achievability of the attacker's goal.   We showcase the adversarial manipulation on two types of LQG learners: the batch RL learner and the other is the adaptive dynamic programming (ADP) learner. Our results demonstrate that with only 2.296% of falsification on the cost data, the attacker misleads the batch RL into learning the 'nefarious' policy that leads the vehicle to a dangerous position. The attacker can also gradually trick the ADP learner into learning the same `nefarious' policy by consistently feeding the learner a falsified cost signal that stays close to the true cost signal. The aim of the paper is to raise people's awareness of the security threats faced by RL-enabled control systems.

摘要: 在这项工作中，我们通过操纵费用信号来研究线性二次高斯(LQG)代理的欺骗行为。我们证明了对代价参数的微小篡改只会导致最优策略有界的改变，并且攻击者可以对代价参数应用的伪造量是线性的。我们提出了一个攻击模型，其中攻击者的目标是误导代理学习具有故意篡改成本参数的“邪恶”策略。我们将攻击问题描述为一个优化问题，证明了该优化问题是凸的，并给出了检验攻击者目标可达性的充要条件。我们展示了在两种类型的LQG学习器上的对抗操作：批量RL学习器和自适应动态规划(ADP)学习器。我们的结果表明，由于只有2.296%的成本数据被篡改，攻击者误导批次RL学习将车辆引向危险位置的“邪恶”策略。攻击者还可以通过始终如一地向学习者提供接近真实成本信号的伪造成本信号，逐渐欺骗ADP学习者学习相同的“邪恶”策略。本文的目的是提高人们对启用RL的控制系统所面临的安全威胁的认识。



## **36. Single Loop Gaussian Homotopy Method for Non-convex Optimization**

求解非凸优化问题的单圈高斯同伦方法 math.OC

45 pages

**SubmitDate**: 2022-03-11    [paper-pdf](http://arxiv.org/pdf/2203.05717v1)

**Authors**: Hidenori Iwakiri, Yuhang Wang, Shinji Ito, Akiko Takeda

**Abstracts**: The Gaussian homotopy (GH) method is a popular approach to finding better local minima for non-convex optimization problems by gradually changing the problem to be solved from a simple one to the original target one. Existing GH-based methods consisting of a double loop structure incur high computational costs, which may limit their potential for practical application. We propose a novel single loop framework for GH methods (SLGH) for both deterministic and stochastic settings. For those applications in which the convolution calculation required to build a GH function is difficult, we present zeroth-order SLGH algorithms with gradient-free oracles. The convergence rate of (zeroth-order) SLGH depends on the decreasing speed of a smoothing hyperparameter, and when the hyperparameter is chosen appropriately, it becomes consistent with the convergence rate of (zeroth-order) gradient descent. In experiments that included artificial highly non-convex examples and black-box adversarial attacks, we have demonstrated that our algorithms converge much faster than an existing double loop GH method while outperforming gradient descent-based methods in terms of finding a better solution.

摘要: 高斯同伦(GH)方法是一种流行的寻找非凸优化问题局部极小值的方法，它将待求解问题从简单问题逐步转化为原始目标问题。现有的基于双环结构的GH方法计算量大，限制了其实际应用的潜力。我们提出了一种新的单环GH方法框架(SLGH)，既适用于确定性环境，也适用于随机环境。对于构造GH函数所需的卷积计算困难的应用，我们提出了具有无梯度预言的零阶SLGH算法。(零阶)SLGH的收敛速度取决于平滑超参数的下降速度，当超参数选择适当时，收敛速度与(零阶)梯度下降的收敛速度一致。在包含人工高度非凸集和黑盒攻击的实验中，我们证明了我们的算法比现有的双环GH方法收敛速度快得多，并且在找到更好的解方面优于基于梯度下降的方法。



## **37. Formalizing and Estimating Distribution Inference Risks**

配电推理风险的形式化与估计 cs.LG

Update: New version with more theoretical results and a deeper  exploration of results. We noted some discrepancies in our experiments on the  CelebA dataset and re-ran all of our experiments for this dataset, updating  Table 1 and Figures 2c, 3b, 4, 7a, and 8a in the process. These did not  substantially impact our results, and our conclusions and observations in  trends remain unchanged

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2109.06024v5)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型基于私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即，生成捕获有关分布的统计属性的模型。在Yeom等人的成员关系推理框架的启发下，我们提出了分布推理攻击的形式化定义，该定义足够通用，可以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均节点度或聚类系数。为了了解分布推理风险，我们引入了一个度量，通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，对观察到的泄漏进行量化。我们报告了使用新颖的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。



## **38. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

TraSw：针对多目标跟踪的Tracklet-Switch敌意攻击 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2111.08954v2)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Multi-Object Tracking (MOT) has achieved aggressive progress and derives many excellent deep learning models. However, the robustness of the trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during the tracking. In this work, we analyze the vulnerability of popular pedestrian MOT trackers and propose a novel adversarial attack method called Tracklet-Switch (TraSw) against the complete tracking pipeline of MOT. TraSw can fool the advanced deep trackers (i.e., FairMOT and ByteTrack) to fail to track the targets in the subsequent frames by attacking very few frames. Experiments on the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20) show that TraSw can achieve an extraordinarily high success rate of over 95% by attacking only four frames on average. To our knowledge, this is the first work on the adversarial attack against pedestrian MOT trackers. The code is available at https://github.com/DerryHub/FairMOT-attack .

摘要: 多目标跟踪(MOT)取得了突破性的进展，衍生出了许多优秀的深度学习模型。然而，很少有人研究跟踪器的鲁棒性，而且由于其成熟的关联算法被设计成对跟踪过程中的错误具有鲁棒性，因此对MOT系统的攻击是具有挑战性的。在这项工作中，我们分析了流行的行人MOT跟踪器的脆弱性，并提出了一种新的针对MOT完整跟踪管道的对抗性攻击方法Tracklet-Switch(TraSw)。TraSw可以通过攻击很少的帧来欺骗高级深度跟踪器(即FairMOT和ByteTrack)，使其无法跟踪后续帧中的目标。在MOT-Challenger数据集(2DMOT15、MOT17和MOT20)上的实验表明，TraSw平均只攻击4帧，可以达到95%以上的超高成功率。据我们所知，这是针对行人MOT跟踪器的首次对抗性攻击。代码可在https://github.com/DerryHub/FairMOT-attack上获得。



## **39. SoK: On the Semantic AI Security in Autonomous Driving**

SOK：关于自动驾驶中的语义人工智能安全 cs.CR

Project website: https://sites.google.com/view/cav-sec/pass

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05314v1)

**Authors**: Junjie Shen, Ningfei Wang, Ziwen Wan, Yunpeng Luo, Takami Sato, Zhisheng Hu, Xinyang Zhang, Shengjian Guo, Zhenyu Zhong, Kang Li, Ziming Zhao, Chunming Qiao, Qi Alfred Chen

**Abstracts**: Autonomous Driving (AD) systems rely on AI components to make safety and correct driving decisions. Unfortunately, today's AI algorithms are known to be generally vulnerable to adversarial attacks. However, for such AI component-level vulnerabilities to be semantically impactful at the system level, it needs to address non-trivial semantic gaps both (1) from the system-level attack input spaces to those at AI component level, and (2) from AI component-level attack impacts to those at the system level. In this paper, we define such research space as semantic AI security as opposed to generic AI security. Over the past 5 years, increasingly more research works are performed to tackle such semantic AI security challenges in AD context, which has started to show an exponential growth trend.   In this paper, we perform the first systematization of knowledge of such growing semantic AD AI security research space. In total, we collect and analyze 53 such papers, and systematically taxonomize them based on research aspects critical for the security field. We summarize 6 most substantial scientific gaps observed based on quantitative comparisons both vertically among existing AD AI security works and horizontally with security works from closely-related domains. With these, we are able to provide insights and potential future directions not only at the design level, but also at the research goal, methodology, and community levels. To address the most critical scientific methodology-level gap, we take the initiative to develop an open-source, uniform, and extensible system-driven evaluation platform, named PASS, for the semantic AD AI security research community. We also use our implemented platform prototype to showcase the capabilities and benefits of such a platform using representative semantic AD AI attacks.

摘要: 自动驾驶(AD)系统依赖人工智能组件来做出安全和正确的驾驶决策。不幸的是，众所周知，今天的人工智能算法通常容易受到对手的攻击。然而，要使这种AI组件级别的漏洞在系统级别产生语义影响，它需要解决以下两个方面的重要语义差距：(1)从系统级别的攻击输入空间到AI组件级别的输入空间，以及(2)从AI组件级别的攻击影响到系统级别的影响。在本文中，我们将这样的研究空间定义为语义人工智能安全，而不是一般的人工智能安全。在过去的5年里，越来越多的研究工作对撞击这样的语义AI在AD环境下的安全挑战进行了研究，并开始呈现指数增长的趋势。在本文中，我们首次对这种不断增长的语义AD AI安全研究空间的知识进行了系统化。我们总共收集和分析了53篇这样的论文，并根据对安全领域至关重要的研究方面对它们进行了系统的分类。我们总结了基于定量比较观察到的6个最实质性的科学差距，这6个差距既包括现有AD AI安全作品之间的纵向比较，也包括与密切相关领域的安全作品的横向比较。有了这些，我们不仅能够在设计层面上，而且在研究目标、方法和社区层面上提供洞察力和潜在的未来方向。为了解决最关键的科学方法论层面的差距，我们主动开发了一个开源的、统一的、可扩展的系统驱动的评估平台，名为PASS，用于语义AD AI安全研究社区。我们还使用我们实现的平台原型来展示这样一个使用典型语义AD AI攻击的平台的能力和好处。



## **40. Adversarial Attacks on Machinery Fault Diagnosis**

机械故障诊断中的对抗性攻击 cs.CR

5 pages, 5 figures. Submitted to Interspeech 2022

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2110.02498v2)

**Authors**: Jiahao Chen, Diqun Yan

**Abstracts**: Despite the great progress of neural network-based (NN-based) machinery fault diagnosis methods, their robustness has been largely neglected, for they can be easily fooled through adding imperceptible perturbation to the input. For fault diagnosis problems, in this paper, we reformulate various adversarial attacks and intensively investigate them under untargeted and targeted conditions. Experimental results on six typical NN-based models show that accuracies of the models are greatly reduced by adding small perturbations. We further propose a simple, efficient and universal scheme to protect the victim models. This work provides an in-depth look at adversarial examples of machinery vibration signals for developing protection methods against adversarial attack and improving the robustness of NN-based models.

摘要: 尽管基于神经网络(NN)的机械故障诊断方法有了很大的进步，但它们的鲁棒性很大程度上被忽略了，因为它们很容易通过在输入中添加不可察觉的扰动而被愚弄。针对故障诊断问题，本文对各种对抗性攻击进行了重新定义，并在无目标和有目标的情况下对其进行了深入的研究。在6个典型的神经网络模型上的实验结果表明，加入小扰动会大大降低模型的精度。在此基础上，提出了一种简单、高效、通用的受害者模型保护方案。这项工作对机械振动信号的对抗性实例进行了深入的研究，以开发针对对抗性攻击的保护方法，并提高基于神经网络的模型的鲁棒性。



## **41. Clustering Label Inference Attack against Practical Split Learning**

针对实用分裂学习的聚类标签推理攻击 cs.LG

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05222v1)

**Authors**: Junlin Liu, Xinchen Lyu

**Abstracts**: Split learning is deemed as a promising paradigm for privacy-preserving distributed learning, where the learning model can be cut into multiple portions to be trained at the participants collaboratively. The participants only exchange the intermediate learning results at the cut layer, including smashed data via forward-pass (i.e., features extracted from the raw data) and gradients during backward-propagation.Understanding the security performance of split learning is critical for various privacy-sensitive applications.With the emphasis on private labels, this paper proposes a passive clustering label inference attack for practical split learning. The adversary (either clients or servers) can accurately retrieve the private labels by collecting the exchanged gradients and smashed data.We mathematically analyse potential label leakages in split learning and propose the cosine and Euclidean similarity measurements for clustering attack. Experimental results validate that the proposed approach is scalable and robust under different settings (e.g., cut layer positions, epochs, and batch sizes) for practical split learning.The adversary can still achieve accurate predictions, even when differential privacy and gradient compression are adopted for label protections.

摘要: 分裂学习被认为是一种很有前途的隐私保护分布式学习范例，它可以将学习模型分割成多个部分，在参与者处进行协作训练。参与者只在切割层交换中间学习结果，包括前向传递的粉碎数据(即从原始数据中提取的特征)和后向传播过程中的梯度，了解分裂学习的安全性能对于各种隐私敏感应用至关重要，该文以私有标签为重点，提出了一种用于实际分裂学习的被动聚类标签推理攻击。通过收集交换的梯度和粉碎的数据，攻击者(无论是客户端还是服务器)都可以准确地恢复私有标签，对分裂学习中潜在的标签泄漏进行了数学分析，并提出了基于余弦和欧几里德相似度量的聚类攻击方法。实验结果表明，该方法在不同环境(如切割层位置、历元、批次大小等)下具有较好的扩展性和鲁棒性，即使在采用差分隐私和梯度压缩进行标签保护的情况下，对手仍能获得准确的预测。



## **42. Membership Privacy Protection for Image Translation Models via Adversarial Knowledge Distillation**

基于对抗性知识提取的图像翻译模型成员隐私保护 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05212v1)

**Authors**: Saeed Ranjbar Alvar, Lanjun Wang, Jian Pei, Yong Zhang

**Abstracts**: Image-to-image translation models are shown to be vulnerable to the Membership Inference Attack (MIA), in which the adversary's goal is to identify whether a sample is used to train the model or not. With daily increasing applications based on image-to-image translation models, it is crucial to protect the privacy of these models against MIAs.   We propose adversarial knowledge distillation (AKD) as a defense method against MIAs for image-to-image translation models. The proposed method protects the privacy of the training samples by improving the generalizability of the model. We conduct experiments on the image-to-image translation models and show that AKD achieves the state-of-the-art utility-privacy tradeoff by reducing the attack performance up to 38.9% compared with the regular training model at the cost of a slight drop in the quality of the generated output images. The experimental results also indicate that the models trained by AKD generalize better than the regular training models. Furthermore, compared with existing defense methods, the results show that at the same privacy protection level, image translation models trained by AKD generate outputs with higher quality; while at the same quality of outputs, AKD enhances the privacy protection over 30%.

摘要: 图像到图像的翻译模型容易受到成员关系推理攻击(MIA)的攻击，在MIA攻击中，对手的目标是识别是否使用样本来训练模型。随着基于图像到图像翻译模型的应用日益增多，保护这些模型的隐私免受MIA攻击变得至关重要。针对图像到图像翻译模型，我们提出了对抗性知识蒸馏(AKD)作为一种防御MIA的方法。该方法通过提高模型的泛化能力来保护训练样本的私密性。我们对图像到图像的转换模型进行了实验，结果表明，AKD在生成图像质量略有下降的情况下，与常规训练模型相比，攻击性能降低了38.9%，达到了最先进的效用-隐私折衷。实验结果还表明，AKD训练的模型比常规训练模型具有更好的泛化能力。此外，与现有的防御方法相比，实验结果表明，在相同的隐私保护水平下，AKD训练的图像翻译模型生成的输出质量更高，而在相同的输出质量下，AKD对隐私的保护提高了30%以上。



## **43. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

基于自适应自动攻击的对手健壮性实用评估 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.05154v1)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$

摘要: 对抗对手攻击的防御模型已经显著增长，但缺乏实用的评估方法阻碍了进展。评估可以定义为在给定预算迭代次数和测试数据集的情况下寻找防御模型的健壮性下限。一种实用的评估方法应该是方便(即无参数)、高效(即迭代次数较少)和可靠(即接近鲁棒性的下界)。针对这一目标，我们提出了一种无参数的自适应自动攻击(A$^3$)评估方法，该方法以测试时间训练的方式来解决效率和可靠性问题。具体地说，通过观察特定防御模型的对抗性示例在起始点遵循一定的规律，我们设计了一种自适应方向初始化策略来加快评估速度。此外，为了在预算迭代次数下逼近鲁棒性的下界，我们提出了一种基于在线统计的丢弃策略，自动识别和丢弃不易攻击的图像。广泛的实验证明了我们的澳元^3元的有效性。特别是，我们将澳元^3美元应用于近50种广泛使用的防御模型。通过比现有方法消耗更少的迭代次数，即平均$1/10$(10$\倍$加速)，我们在所有情况下都获得了较低的鲁棒精度。值得注意的是，我们用这种方法赢得了CVPR 2021年白盒对抗性攻击防御模型比赛1681支队伍中的$\textbf{第一名}$。代码可在以下网址获得：$\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **44. Controllable Evaluation and Generation of Physical Adversarial Patch on Face Recognition**

人脸识别中物理对抗性补丁的可控评价与生成 cs.CV

**SubmitDate**: 2022-03-10    [paper-pdf](http://arxiv.org/pdf/2203.04623v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Zihao Xiao, Hang Su, Jun Zhu

**Abstracts**: Recent studies have revealed the vulnerability of face recognition models against physical adversarial patches, which raises security concerns about the deployed face recognition systems. However, it is still challenging to ensure the reproducibility for most attack algorithms under complex physical conditions, which leads to the lack of a systematic evaluation of the existing methods. It is therefore imperative to develop a framework that can enable a comprehensive evaluation of the vulnerability of face recognition in the physical world. To this end, we propose to simulate the complex transformations of faces in the physical world via 3D-face modeling, which serves as a digital counterpart of physical faces. The generic framework allows us to control different face variations and physical conditions to conduct reproducible evaluations comprehensively. With this digital simulator, we further propose a Face3DAdv method considering the 3D face transformations and realistic physical variations. Extensive experiments validate that Face3DAdv can significantly improve the effectiveness of diverse physically realizable adversarial patches in both simulated and physical environments, against various white-box and black-box face recognition models.

摘要: 最近的研究揭示了人脸识别模型对物理对手补丁的脆弱性，这引发了人们对部署的人脸识别系统的安全担忧。然而，大多数攻击算法在复杂物理条件下的可重复性仍然是具有挑战性的，这导致对现有方法缺乏系统的评估。因此，当务之急是制定一个框架，使之能够全面评估现实世界中人脸识别的脆弱性。为此，我们建议通过3D人脸建模来模拟人脸在物理世界中的复杂变换，3D人脸建模是物理人脸的数字对应。通用框架允许我们控制不同的脸部变化和身体条件，以进行全面的可重复性评估。在此数字模拟器的基础上，我们进一步提出了一种考虑3D人脸变换和真实物理变化的Face3DAdv方法。大量实验证明，Face3DAdv能够显著提高各种物理可实现的对抗性补丁在模拟和物理环境中对抗各种白盒和黑盒人脸识别模型的有效性。



## **45. Security of quantum key distribution from generalised entropy accumulation**

广义熵积累下量子密钥分配的安全性 quant-ph

32 pages

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04993v1)

**Authors**: Tony Metger, Renato Renner

**Abstracts**: The goal of quantum key distribution (QKD) is to establish a secure key between two parties connected by an insecure quantum channel. To use a QKD protocol in practice, one has to prove that it is secure against general attacks: even if an adversary performs a complicated attack involving all of the rounds of the protocol, they cannot gain useful information about the key. A much simpler task is to prove security against collective attacks, where the adversary is assumed to behave the same in each round. Using a recently developed information-theoretic tool called generalised entropy accumulation, we show that for a very broad class of QKD protocols, security against collective attacks implies security against general attacks. Compared to existing techniques such as the quantum de Finetti theorem or a previous version of entropy accumulation, our result can be applied much more broadly and easily: it does not require special assumptions on the protocol such as symmetry or a Markov property between rounds, its bounds are independent of the dimension of the underlying Hilbert space, and it can be applied to prepare-and-measure protocols directly without switching to an entanglement-based version.

摘要: 量子密钥分发(QKD)的目标是在通过不安全的量子信道连接的双方之间建立安全密钥。要在实践中使用QKD协议，必须证明它对一般攻击是安全的：即使对手执行了涉及协议所有轮的复杂攻击，他们也无法获得有关密钥的有用信息。一个简单得多的任务是证明针对集体攻击的安全性，假设对手在每一轮中的行为都是一样的。使用最近开发的称为广义熵积累的信息论工具，我们证明了对于非常广泛的一类QKD协议，针对集体攻击的安全性意味着针对一般攻击的安全性。与量子de Finetti定理或前一版本的熵积累等现有技术相比，我们的结果可以更广泛和更容易地应用：它不需要对协议进行特殊的假设，例如轮间的对称性或马尔可夫性质，它的界限与底层Hilbert空间的维数无关，并且它可以直接应用于制备和测量协议，而不需要切换到基于纠缠的版本。



## **46. Physics-aware Complex-valued Adversarial Machine Learning in Reconfigurable Diffractive All-optical Neural Network**

可重构衍射全光神经网络中的物理感知复值对抗性机器学习 cs.ET

34 pages, 4 figures

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.06055v1)

**Authors**: Ruiyang Chen, Yingjie Li, Minhan Lou, Jichao Fan, Yingheng Tang, Berardi Sensale-Rodriguez, Cunxi Yu, Weilu Gao

**Abstracts**: Diffractive optical neural networks have shown promising advantages over electronic circuits for accelerating modern machine learning (ML) algorithms. However, it is challenging to achieve fully programmable all-optical implementation and rapid hardware deployment. Furthermore, understanding the threat of adversarial ML in such system becomes crucial for real-world applications, which remains unexplored. Here, we demonstrate a large-scale, cost-effective, complex-valued, and reconfigurable diffractive all-optical neural networks system in the visible range based on cascaded transmissive twisted nematic liquid crystal spatial light modulators. With the assist of categorical reparameterization, we create a physics-aware training framework for the fast and accurate deployment of computer-trained models onto optical hardware. Furthermore, we theoretically analyze and experimentally demonstrate physics-aware adversarial attacks onto the system, which are generated from a complex-valued gradient-based algorithm. The detailed adversarial robustness comparison with conventional multiple layer perceptrons and convolutional neural networks features a distinct statistical adversarial property in diffractive optical neural networks. Our full stack of software and hardware provides new opportunities of employing diffractive optics in a variety of ML tasks and enabling the research on optical adversarial ML.

摘要: 与电子电路相比，衍射光学神经网络在加速现代机器学习(ML)算法方面显示出了巨大的优势。然而，要实现完全可编程的全光实现和快速的硬件部署是具有挑战性的。此外，了解敌意ML在这样的系统中的威胁对于现实世界的应用来说是至关重要的，这一点仍然有待探索。在这里，我们展示了一种基于级联透射式扭曲向列相液晶空间光调制器的大规模、高性价比、复值和可重构的可见光衍射全光神经网络系统。在分类重参数化的帮助下，我们创建了一个物理感知训练框架，用于快速准确地将计算机训练的模型部署到光学硬件上。此外，我们对基于复值梯度算法产生的物理感知的敌意攻击进行了理论分析和实验演示。与传统的多层感知器和卷积神经网络相比，衍射光学神经网络具有明显的统计对抗性。我们的全套软件和硬件提供了在各种ML任务中使用衍射光学的新机会，并使光学对抗性ML的研究成为可能。



## **47. Reverse Engineering $\ell_p$ attacks: A block-sparse optimization approach with recovery guarantees**

逆向工程$\ell_p$攻击：一种具有恢复保证的挡路稀疏优化方法 cs.LG

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04886v1)

**Authors**: Darshan Thaker, Paris Giampouras, René Vidal

**Abstracts**: Deep neural network-based classifiers have been shown to be vulnerable to imperceptible perturbations to their input, such as $\ell_p$-bounded norm adversarial attacks. This has motivated the development of many defense methods, which are then broken by new attacks, and so on. This paper focuses on a different but related problem of reverse engineering adversarial attacks. Specifically, given an attacked signal, we study conditions under which one can determine the type of attack ($\ell_1$, $\ell_2$ or $\ell_\infty$) and recover the clean signal. We pose this problem as a block-sparse recovery problem, where both the signal and the attack are assumed to lie in a union of subspaces that includes one subspace per class and one subspace per attack type. We derive geometric conditions on the subspaces under which any attacked signal can be decomposed as the sum of a clean signal plus an attack. In addition, by determining the subspaces that contain the signal and the attack, we can also classify the signal and determine the attack type. Experiments on digit and face classification demonstrate the effectiveness of the proposed approach.

摘要: 基于深度神经网络的分类器很容易受到不可察觉的输入扰动，例如$\ellp$-有界范数的对抗性攻击。这推动了许多防御方法的发展，然后这些方法被新的攻击所打破，等等。本文关注的是一个不同但又相关的逆向工程对抗性攻击问题。具体地说，在给定攻击信号的情况下，我们研究了确定攻击类型($\ell_1$、$\ell_2$或$\ell_\infty$)并恢复干净信号的条件。我们把这个问题归结为一个挡路稀疏恢复问题，这里假设信号和攻击都位于一个子空间的并中，每个类包含一个子空间，每个攻击类型包含一个子空间。我们导出了子空间上的几何条件，在这些条件下，任何被攻击的信号都可以分解为一个干净的信号加一个攻击的和。此外，通过确定包含信号和攻击的子空间，还可以对信号进行分类，确定攻击类型。数字和人脸分类实验证明了该方法的有效性。



## **48. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04713v1)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Deep learning has been regarded as the `go to' solution for many tasks today, but its intrinsic vulnerability to malicious attacks has become a major concern. The vulnerability is affected by a variety of factors including models, tasks, data, and attackers. Consequently, methods such as Adversarial Training and Randomized Smoothing have been proposed to tackle the problem in a wide range of applications. In this paper, we investigate skeleton-based Human Activity Recognition, which is an important type of time-series data but under-explored in defense against attacks. Our method is featured by (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new parameterization of the adversarial sample manifold of actions, and (3) a new post-train Bayesian treatment on both the adversarial samples and the classifier. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of action classifiers and datasets, under various attacks.

摘要: 深度学习已被认为是当今许多任务的“去”解决方案，但其固有的易受恶意攻击的脆弱性已成为人们主要关注的问题。该漏洞受多种因素影响，包括模型、任务、数据和攻击者。因此，对抗性训练和随机化平滑等方法被提出，以解决撞击的这一问题，并得到了广泛的应用。在本文中，我们研究了基于骨架的人类活动识别，这是一种重要的时间序列数据类型，但在防御攻击方面还没有得到充分的探索。我们的方法的特点是(1)新的基于贝叶斯能量的鲁棒判别分类器公式，(2)新的对抗性样本动作流形的参数化，(3)对对抗性样本和分类器的新的训练后贝叶斯处理。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是直截了当但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的动作分类器和数据集上展示了令人惊讶的和普遍的有效性。



## **49. Robust Federated Learning Against Adversarial Attacks for Speech Emotion Recognition**

语音情感识别中抗敌意攻击的鲁棒联合学习 cs.SD

11 pages, 6 figures, 3 tables

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.04696v1)

**Authors**: Yi Chang, Sofiane Laridi, Zhao Ren, Gregory Palmer, Björn W. Schuller, Marco Fisichella

**Abstracts**: Due to the development of machine learning and speech processing, speech emotion recognition has been a popular research topic in recent years. However, the speech data cannot be protected when it is uploaded and processed on servers in the internet-of-things applications of speech emotion recognition. Furthermore, deep neural networks have proven to be vulnerable to human-indistinguishable adversarial perturbations. The adversarial attacks generated from the perturbations may result in deep neural networks wrongly predicting the emotional states. We propose a novel federated adversarial learning framework for protecting both data and deep neural networks. The proposed framework consists of i) federated learning for data privacy, and ii) adversarial training at the training stage and randomisation at the testing stage for model robustness. The experiments show that our proposed framework can effectively protect the speech data locally and improve the model robustness against a series of adversarial attacks.

摘要: 近年来，随着机器学习和语音处理技术的发展，语音情感识别成为一个热门的研究课题。然而，在语音情感识别的物联网应用中，当语音数据被上传并在服务器上处理时，语音数据不能得到保护。此外，深层神经网络已被证明容易受到人类无法区分的敌意干扰。由扰动产生的对抗性攻击可能会导致深层神经网络错误地预测情绪状态。我们提出了一种新的联合对抗性学习框架，用于保护数据和深度神经网络。该框架包括：i)数据隐私的联合学习；ii)训练阶段的对抗性训练和模型鲁棒性测试阶段的随机化。实验表明，该框架能有效地保护语音数据的局部安全，提高了模型对一系列对抗性攻击的鲁棒性。



## **50. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

阴影可能是危险的：自然现象对物理世界的隐秘而有效的对抗性攻击 cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-09    [paper-pdf](http://arxiv.org/pdf/2203.03818v2)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.

摘要: 估计对抗性示例的风险水平对于在现实世界中安全地部署机器学习模型是至关重要的。物理世界攻击的一种流行方法是采用“粘贴”策略，但该策略受到一些限制，包括难以接近目标或以有效颜色打印。最近出现了一种新型的非侵入性攻击，它试图通过激光束和投影仪等基于光学的工具对目标进行摄动。然而，添加的光学图案是人造的，但不是自然的。因此，它们仍然是引人注目和引人注目的，很容易被人类注意到。本文研究了一种新的光学对抗实例，其中的扰动是由一种非常常见的自然现象--阴影产生的，从而在黑盒环境下实现了自然主义的、隐身的物理世界对抗攻击。我们广泛评估了这种新攻击在模拟和真实环境中的有效性。在交通标志识别上的实验结果表明，该算法能够有效地生成对抗性样本，在LISA和GTSRB测试集上的成功率分别达到98.23%和90.47%，而在真实场景中，95%以上的时间都能连续误导移动的摄像机。我们还讨论了这种攻击的局限性和防御机制。



