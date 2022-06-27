# Latest Adversarial Attack Papers
**update at 2022-06-28 06:31:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Debiasing Learning for Membership Inference Attacks Against Recommender Systems**

推荐系统成员关系推理攻击的去偏学习 cs.IR

Accepted by KDD 2022

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12401v1)

**Authors**: Zihan Wang, Na Huang, Fei Sun, Pengjie Ren, Zhumin Chen, Hengliang Luo, Maarten de Rijke, Zhaochun Ren

**Abstracts**: Learned recommender systems may inadvertently leak information about their training data, leading to privacy violations. We investigate privacy threats faced by recommender systems through the lens of membership inference. In such attacks, an adversary aims to infer whether a user's data is used to train the target recommender. To achieve this, previous work has used a shadow recommender to derive training data for the attack model, and then predicts the membership by calculating difference vectors between users' historical interactions and recommended items. State-of-the-art methods face two challenging problems: (1) training data for the attack model is biased due to the gap between shadow and target recommenders, and (2) hidden states in recommenders are not observational, resulting in inaccurate estimations of difference vectors. To address the above limitations, we propose a Debiasing Learning for Membership Inference Attacks against recommender systems (DL-MIA) framework that has four main components: (1) a difference vector generator, (2) a disentangled encoder, (3) a weight estimator, and (4) an attack model. To mitigate the gap between recommenders, a variational auto-encoder (VAE) based disentangled encoder is devised to identify recommender invariant and specific features. To reduce the estimation bias, we design a weight estimator, assigning a truth-level score for each difference vector to indicate estimation accuracy. We evaluate DL-MIA against both general recommenders and sequential recommenders on three real-world datasets. Experimental results show that DL-MIA effectively alleviates training and estimation biases simultaneously, and achieves state-of-the-art attack performance.

摘要: 学习推荐系统可能会无意中泄露有关其训练数据的信息，导致侵犯隐私。我们通过成员关系推理的视角来研究推荐系统所面临的隐私威胁。在这类攻击中，对手的目标是推断用户的数据是否被用来训练目标推荐者。为此，以前的工作使用影子推荐器来获取攻击模型的训练数据，然后通过计算用户历史交互与推荐项目之间的差异向量来预测成员资格。最新的方法面临两个具有挑战性的问题：(1)由于阴影和目标推荐器之间的差距，攻击模型的训练数据存在偏差；(2)推荐器中的隐藏状态不是可观测的，导致对差异向量的估计不准确。针对上述局限性，我们提出了一个针对推荐系统成员推理攻击的去偏学习框架(DL-MIA)，该框架包括四个主要部分：(1)差分向量生成器，(2)解缠编码器，(3)权重估计器，(4)攻击模型。为了缩小推荐者之间的差距，设计了一种基于变分自动编码器(VAE)的解缠编码器来识别推荐者的不变性和特定特征。为了减少估计偏差，我们设计了一个权重估计器，为每个差异向量分配一个真实度分数来表示估计的准确性。在三个真实数据集上，我们对比了一般推荐器和顺序推荐器对DL-MIA进行了评估。实验结果表明，DL-MIA有效地同时缓解了训练偏差和估计偏差，达到了最好的攻击性能。



## **2. Defending Backdoor Attacks on Vision Transformer via Patch Processing**

利用补丁处理防御视觉转换器的后门攻击 cs.CV

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12381v1)

**Authors**: Khoa D. Doan, Yingjie Lao, Peng Yang, Ping Li

**Abstracts**: Vision Transformers (ViTs) have a radically different architecture with significantly less inductive bias than Convolutional Neural Networks. Along with the improvement in performance, security and robustness of ViTs are also of great importance to study. In contrast to many recent works that exploit the robustness of ViTs against adversarial examples, this paper investigates a representative causative attack, i.e., backdoor. We first examine the vulnerability of ViTs against various backdoor attacks and find that ViTs are also quite vulnerable to existing attacks. However, we observe that the clean-data accuracy and backdoor attack success rate of ViTs respond distinctively to patch transformations before the positional encoding. Then, based on this finding, we propose an effective method for ViTs to defend both patch-based and blending-based trigger backdoor attacks via patch processing. The performances are evaluated on several benchmark datasets, including CIFAR10, GTSRB, and TinyImageNet, which show the proposed novel defense is very successful in mitigating backdoor attacks for ViTs. To the best of our knowledge, this paper presents the first defensive strategy that utilizes a unique characteristic of ViTs against backdoor attacks.

摘要: 与卷积神经网络相比，视觉转换器(VITS)具有完全不同的体系结构，具有明显更少的感应偏差。随着性能的提高，VITS的安全性和健壮性也具有重要的研究意义。与最近许多利用VITS对敌意例子的健壮性的工作不同，本文研究了一种典型的致因攻击，即后门攻击。我们首先检查VITS对各种后门攻击的脆弱性，发现VITS也很容易受到现有攻击的攻击。然而，我们观察到VITS的干净数据准确性和后门攻击成功率对位置编码之前的补丁变换有明显的响应。然后，基于这一发现，我们提出了一种VITS通过补丁处理来防御基于补丁和基于混合的触发后门攻击的有效方法。在包括CIFAR10、GTSRB和TinyImageNet在内的几个基准数据集上进行了性能评估，表明所提出的新型防御在缓解VITS后门攻击方面是非常成功的。据我们所知，本文提出了第一种利用VITS的独特特性来抵御后门攻击的防御策略。



## **3. Robustness of Explanation Methods for NLP Models**

NLP模型解释方法的稳健性 cs.CL

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12284v1)

**Authors**: Shriya Atmakuri, Tejas Chheda, Dinesh Kandula, Nishant Yadav, Taesung Lee, Hessel Tuinhof

**Abstracts**: Explanation methods have emerged as an important tool to highlight the features responsible for the predictions of neural networks. There is mounting evidence that many explanation methods are rather unreliable and susceptible to malicious manipulations. In this paper, we particularly aim to understand the robustness of explanation methods in the context of text modality. We provide initial insights and results towards devising a successful adversarial attack against text explanations. To our knowledge, this is the first attempt to evaluate the adversarial robustness of an explanation method. Our experiments show the explanation method can be largely disturbed for up to 86% of the tested samples with small changes in the input sentence and its semantics.

摘要: 解释方法已成为突出神经网络预测特征的重要工具。越来越多的证据表明，许多解释方法相当不可靠，容易受到恶意操纵。在这篇文章中，我们特别致力于理解解释方法在语篇情态语境中的稳健性。我们为设计对文本解释的成功的对抗性攻击提供了初步的见解和结果。据我们所知，这是第一次尝试评估解释方法的对抗性稳健性。我们的实验表明，在输入句子及其语义稍有变化的情况下，该解释方法可以对高达86%的测试样本产生很大的干扰。



## **4. Property Unlearning: A Defense Strategy Against Property Inference Attacks**

属性遗忘：一种防御属性推理攻击的策略 cs.CR

Please note: As of June 24, 2022, we have discovered some flaws in  our experimental setup. The defense mechanism property unlearning is not as  strong as the experimental results in the current version of the paper  suggest. We will provide an updated version soon

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2205.08821v2)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstracts**: During the training of machine learning models, they may store or "learn" more information about the training data than what is actually needed for the prediction or classification task. This is exploited by property inference attacks which aim at extracting statistical properties from the training data of a given model without having access to the training data itself. These properties may include the quality of pictures to identify the camera model, the age distribution to reveal the target audience of a product, or the included host types to refine a malware attack in computer networks. This attack is especially accurate when the attacker has access to all model parameters, i.e., in a white-box scenario. By defending against such attacks, model owners are able to ensure that their training data, associated properties, and thus their intellectual property stays private, even if they deliberately share their models, e.g., to train collaboratively, or if models are leaked. In this paper, we introduce property unlearning, an effective defense mechanism against white-box property inference attacks, independent of the training data type, model task, or number of properties. Property unlearning mitigates property inference attacks by systematically changing the trained weights and biases of a target model such that an adversary cannot extract chosen properties. We empirically evaluate property unlearning on three different data sets, including tabular and image data, and two types of artificial neural networks. Our results show that property unlearning is both efficient and reliable to protect machine learning models against property inference attacks, with a good privacy-utility trade-off. Furthermore, our approach indicates that this mechanism is also effective to unlearn multiple properties.

摘要: 在机器学习模型的训练过程中，它们可能存储或“学习”比预测或分类任务实际需要的更多关于训练数据的信息。这被属性推理攻击所利用，该属性推理攻击的目的是从给定模型的训练数据中提取统计属性，而不访问训练数据本身。这些属性可以包括用于识别相机型号的图片质量、用于揭示产品目标受众的年龄分布、或用于改进计算机网络中的恶意软件攻击的所包括的主机类型。当攻击者有权访问所有模型参数时，即在白盒情况下，此攻击尤其准确。通过防御此类攻击，模型所有者能够确保他们的训练数据、相关属性以及他们的知识产权是保密的，即使他们故意共享他们的模型，例如协作训练，或者如果模型被泄露。在本文中，我们引入了属性遗忘，这是一种有效的防御白盒属性推理攻击的机制，独立于训练数据类型、模型任务或属性数量。属性遗忘通过系统地改变目标模型的训练权重和偏差来减轻属性推断攻击，使得对手无法提取所选的属性。我们在三个不同的数据集上经验地评估了属性遗忘，包括表格和图像数据，以及两种类型的人工神经网络。我们的结果表明，属性忘却在保护机器学习模型免受属性推理攻击方面是有效和可靠的，并且具有良好的隐私效用权衡。此外，我们的方法表明，该机制也有效地忘却了多个属性。



## **5. Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective**

深度神经网络的对抗健壮性：从形式验证的角度综述 cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12227v1)

**Authors**: Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong

**Abstracts**: Neural networks have been widely applied in security applications such as spam and phishing detection, intrusion prevention, and malware detection. This black-box method, however, often has uncertainty and poor explainability in applications. Furthermore, neural networks themselves are often vulnerable to adversarial attacks. For those reasons, there is a high demand for trustworthy and rigorous methods to verify the robustness of neural network models. Adversarial robustness, which concerns the reliability of a neural network when dealing with maliciously manipulated inputs, is one of the hottest topics in security and machine learning. In this work, we survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies. We also demonstrate representative techniques that have been applied in existing studies with a sample model. Finally, we discuss open questions for future research.

摘要: 神经网络已广泛应用于垃圾邮件和网络钓鱼检测、入侵防御和恶意软件检测等安全应用中。然而，这种黑箱方法在应用中往往具有不确定性和较差的可解释性。此外，神经网络本身往往容易受到敌意攻击。因此，对神经网络模型的稳健性验证方法提出了更高的要求。对抗健壮性是安全和机器学习领域中最热门的话题之一，它涉及到神经网络在处理恶意操作的输入时的可靠性。在这项工作中，我们综述了现有的神经网络对抗健壮性验证的文献，并收集了39个不同的研究工作，涉及机器学习、安全和软件工程领域。我们系统地分析了他们的方法，包括健壮性是如何形成的，使用了什么验证技术，以及每种技术的优点和局限性。为了全面理解这一主题，我们从正式验证的角度提供了一个分类法。我们根据属性规范、问题约简和推理策略对现有技术进行分类。我们还用一个样本模型演示了已在现有研究中应用的代表性技术。最后，我们讨论了未来研究的有待解决的问题。



## **6. Cluster Attack: Query-based Adversarial Attacks on Graphs with Graph-Dependent Priors**

簇攻击：图依赖先验图上基于查询的敌意攻击 cs.LG

IJCAI 2022 (Long Presentation)

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2109.13069v2)

**Authors**: Zhengyi Wang, Zhongkai Hao, Ziqiao Wang, Hang Su, Jun Zhu

**Abstracts**: While deep neural networks have achieved great success in graph analysis, recent work has shown that they are vulnerable to adversarial attacks. Compared with adversarial attacks on image classification, performing adversarial attacks on graphs is more challenging because of the discrete and non-differential nature of the adjacent matrix for a graph. In this work, we propose Cluster Attack -- a Graph Injection Attack (GIA) on node classification, which injects fake nodes into the original graph to degenerate the performance of graph neural networks (GNNs) on certain victim nodes while affecting the other nodes as little as possible. We demonstrate that a GIA problem can be equivalently formulated as a graph clustering problem; thus, the discrete optimization problem of the adjacency matrix can be solved in the context of graph clustering. In particular, we propose to measure the similarity between victim nodes by a metric of Adversarial Vulnerability, which is related to how the victim nodes will be affected by the injected fake node, and to cluster the victim nodes accordingly. Our attack is performed in a practical and unnoticeable query-based black-box manner with only a few nodes on the graphs that can be accessed. Theoretical analysis and extensive experiments demonstrate the effectiveness of our method by fooling the node classifiers with only a small number of queries.

摘要: 虽然深度神经网络在图分析方面取得了巨大的成功，但最近的研究表明，它们很容易受到对手的攻击。与图像分类中的对抗性攻击相比，由于图的邻接矩阵的离散和非可微性质，对图进行对抗性攻击具有更大的挑战性。在这项工作中，我们提出了一种针对节点分类的图注入攻击(GIA)--图注入攻击(GIA)，它将伪节点注入到原始图中，以降低图神经网络(GNN)在某些受害节点上的性能，同时尽可能地减少对其他节点的影响。我们证明了GIA问题可以等价地表示为图聚类问题，从而可以在图聚类的背景下解决邻接矩阵的离散优化问题。特别是，我们提出了通过敌意脆弱性来衡量受害节点之间的相似性，该度量与注入的伪节点将如何影响受害节点有关，并据此对受害节点进行聚类。我们的攻击是以一种实用的、不可察觉的基于查询的黑盒方式进行的，图上只有几个节点可以访问。理论分析和大量实验证明了该方法的有效性，仅用少量的查询就可以愚弄节点分类器。



## **7. An Improved Lattice-Based Ring Signature with Unclaimable Anonymity in the Standard Model**

一种改进的标准模型下不可否认匿名性的格环签名 cs.CR

**SubmitDate**: 2022-06-24    [paper-pdf](http://arxiv.org/pdf/2206.12093v1)

**Authors**: Mingxing Hu, Weijiong Zhang, Zhen Liu

**Abstracts**: Ring signatures enable a user to sign messages on behalf of an arbitrary set of users, called the ring, without revealing exactly which member of that ring actually generated the signature. The signer-anonymity property makes ring signatures have been an active research topic. Recently, Park and Sealfon (CRYPTO 19) presented an important anonymity notion named signer-unclaimability and constructed a lattice-based ring signature scheme with unclaimable anonymity in the standard model, however, it did not consider the unforgeable w.r.t. adversarially-chosen-key attack (the public key ring of a signature may contain keys created by an adversary) and the signature size grows quadratically in the size of ring and message. In this work, we propose a new lattice-based ring signature scheme with unclaimable anonymity in the standard model. In particular, our work improves the security and efficiency of Park and Sealfons work, which is unforgeable w.r.t. adversarially-chosen-key attack, and the ring signature size grows linearly in the ring size.

摘要: 环签名使用户能够代表称为环的任意一组用户对消息进行签名，而不会确切地揭示该环中的哪个成员实际生成了签名。签名者匿名性使得环签名成为一个活跃的研究课题。最近，Park和Sealfon(Crypto 19)提出了一个重要的匿名性概念：签名者不可否认性，并在标准模型下构造了一个不可否认匿名性的格型环签名方案，但它没有考虑不可伪造性。恶意选择密钥攻击(签名的公钥环可能包含对手创建的密钥)，签名大小随着环和消息的大小呈二次曲线增长。在这项工作中，我们提出了一个新的基于格的环签名方案，在标准模型下具有不可否认的匿名性。特别是，我们的工作提高了公园和Sealfons工作的安全性和效率，这是无法伪造的W.r.t.恶意选择密钥攻击，且环签名大小随环大小线性增长。



## **8. Keep Your Transactions On Short Leashes**

在短时间内控制你的交易 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11974v1)

**Authors**: Bennet Yee

**Abstracts**: The adversary's goal in mounting Long Range Attacks (LRAs) is to fool potential victims into using and relying on a side chain, i.e., a false, alternate history of transactions, and into proposing transactions that end up harming themselves or others. Previous research work on LRAs on blockchain systems have used, at a high level, one of two approaches. They either try to (1) prevent the creation of a bogus side chain or (2) make it possible to distinguish such a side chain from the main consensus chain.   In this paper, we take a different approach. We start with the indistinguishability of side chains from the consensus chain -- for the eclipsed victim -- as a given and assume the potential victim will be fooled. Instead, we protect the victim via harm reduction applying "short leashes" to transactions. The leashes prevent transactions from being used in the wrong context.   The primary contribution of this paper is the design and analysis of leashes. A secondary contribution is the careful explication of the LRA threat model in the context of BAR fault tolerance, and using it to analyze related work to identify their limitations.

摘要: 对手发起远程攻击(LRA)的目的是欺骗潜在受害者使用和依赖侧链，即虚假的、替代的交易历史，并提出最终伤害自己或他人的交易。以前关于区块链系统上的LRA的研究工作在高水平上使用了两种方法之一。他们要么试图(1)防止虚假侧链的产生，要么(2)使这种侧链与主要共识链区分开来成为可能。在本文中，我们采取了一种不同的方法。我们从侧链和共识链的不可区分开始--对于黯然失色的受害者--作为给定的假设，并假设潜在的受害者将被愚弄。取而代之的是，我们通过减少伤害来保护受害者，对交易施加“短皮带”。捆绑可以防止交易在错误的上下文中使用。本文的主要贡献是对牵引带的设计和分析。第二个贡献是在BAR容错的背景下仔细解释了LRA威胁模型，并使用它来分析相关工作以确定它们的局限性。



## **9. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

将你的力量转向你：检测和减轻健壮的和通用的对抗性补丁攻击 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2108.05075v3)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks that inject arbitrary distortions within a bounded region of an image, can trigger misclassification in deep neural networks (DNNs). These attacks are robust (i.e., physically realizable) and universally malicious, and hence represent a severe security threat to real-world DNN-based systems.   This work proposes Jujutsu, a two-stage technique to detect and mitigate robust and universal adversarial patch attacks. We first observe that patch attacks often yield large influence on the prediction output in order to dominate the prediction on any input, and Jujutsu is built to expose this behavior for effective attack detection. For mitigation, we observe that patch attacks corrupt only a localized region while the remaining contents are unperturbed, based on which Jujutsu leverages GAN-based image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image for correct prediction.   We evaluate Jujutsu on four diverse datasets and show that it achieves superior performance and significantly outperforms four leading defenses. Jujutsu can further defend against physical-world attacks, attacks that target diverse classes, and adaptive attacks. Our code is available at https://github.com/DependableSystemsLab/Jujutsu.

摘要: 对抗性补丁攻击在图像的有界区域内注入任意扭曲，可在深度神经网络(DNN)中引发错误分类。这些攻击是健壮的(即，物理上可实现的)并且普遍是恶意的，因此对现实世界中基于DNN的系统构成了严重的安全威胁。这项工作提出了Jujutsu，这是一种两阶段技术，用于检测和缓解健壮且通用的敌意补丁攻击。我们首先观察到补丁攻击通常会对预测输出产生很大的影响，以便在任何输入上控制预测，Jujutsu被构建来暴露这种行为以进行有效的攻击检测。为了缓解攻击，我们观察到补丁攻击只破坏了局部区域，而其余内容不受干扰，基于此，Jujutsu利用基于GaN的图像修复来合成被攻击破坏的像素中的语义内容，并重建出正确的预测。我们在四个不同的数据集上对Jujutsu进行了评估，结果表明它取得了优越的性能，并显著超过了四个领先的防御系统。魔术可以进一步防御物理世界的攻击，针对不同职业的攻击，以及适应性攻击。我们的代码可以在https://github.com/DependableSystemsLab/Jujutsu.上找到



## **10. Probabilistically Resilient Multi-Robot Informative Path Planning**

概率弹性多机器人信息路径规划 cs.RO

9 pages, 6 figures, submitted to IEEE Robotics and Automation Letters  (RA-L)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11789v1)

**Authors**: Remy Wehbe, Ryan K. Williams

**Abstracts**: In this paper, we solve a multi-robot informative path planning (MIPP) task under the influence of uncertain communication and adversarial attackers. The goal is to create a multi-robot system that can learn and unify its knowledge of an unknown environment despite the presence of corrupted robots sharing malicious information. We use a Gaussian Process (GP) to model our unknown environment and define informativeness using the metric of mutual information. The objectives of our MIPP task is to maximize the amount of information collected by the team while maximizing the probability of resilience to attack. Unfortunately, these objectives are at odds especially when exploring large environments which necessitates disconnections between robots. As a result, we impose a probabilistic communication constraint that allows robots to meet intermittently and resiliently share information, and then act to maximize collected information during all other times. To solve our problem, we select meeting locations with the highest probability of resilience and use a sequential greedy algorithm to optimize paths for robots to explore. Finally, we show the validity of our results by comparing the learning ability of well-behaving robots applying resilient vs. non-resilient MIPP algorithms.

摘要: 本文解决了通信不确定和敌方攻击者影响下的多机器人信息路径规划问题。其目标是创建一种多机器人系统，即使存在共享恶意信息的被破坏的机器人，也可以学习和统一其对未知环境的知识。我们使用高斯过程(GP)对未知环境进行建模，并使用互信息度量来定义信息量。我们的MIPP任务的目标是最大化团队收集的信息量，同时最大化抵抗攻击的可能性。不幸的是，这些目标是不一致的，特别是在探索需要断开机器人之间连接的大型环境时。因此，我们施加了一个概率通信约束，允许机器人间歇性地会面并弹性地共享信息，然后在所有其他时间采取行动最大限度地收集信息。为了解决我们的问题，我们选择具有最高弹性的会议地点，并使用顺序贪婪算法来优化供机器人探索的路径。最后，我们通过比较弹性和非弹性MIPP算法对行为良好的机器人的学习能力，证明了我们的结果的有效性。



## **11. Towards End-to-End Private Automatic Speaker Recognition**

走向端到端的私人自动说话人识别 eess.AS

Accepted for publication at Interspeech 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11750v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstracts**: The development of privacy-preserving automatic speaker verification systems has been the focus of a number of studies with the intent of allowing users to authenticate themselves without risking the privacy of their voice. However, current privacy-preserving methods assume that the template voice representations (or speaker embeddings) used for authentication are extracted locally by the user. This poses two important issues: first, knowledge of the speaker embedding extraction model may create security and robustness liabilities for the authentication system, as this knowledge might help attackers in crafting adversarial examples able to mislead the system; second, from the point of view of a service provider the speaker embedding extraction model is arguably one of the most valuable components in the system and, as such, disclosing it would be highly undesirable. In this work, we show how speaker embeddings can be extracted while keeping both the speaker's voice and the service provider's model private, using Secure Multiparty Computation. Further, we show that it is possible to obtain reasonable trade-offs between security and computational cost. This work is complementary to those showing how authentication may be performed privately, and thus can be considered as another step towards fully private automatic speaker recognition.

摘要: 保护隐私的自动说话人验证系统的开发一直是许多研究的重点，目的是允许用户在不危及其语音隐私的情况下验证自己。然而，当前的隐私保护方法假设用于认证的模板语音表示(或说话人嵌入)是由用户本地提取的。这提出了两个重要问题：第一，知道说话人嵌入提取模型可能会给认证系统带来安全和健壮性方面的风险，因为这种知识可能帮助攻击者制作能够误导系统的敌意例子；第二，从服务提供商的角度来看，说话人嵌入提取模型可能是系统中最有价值的组件之一，因此，公开它将是非常不可取的。在这项工作中，我们展示了如何使用安全多方计算在保持说话人的语音和服务提供商的模型隐私的同时提取说话人嵌入。此外，我们还证明了在安全性和计算成本之间取得合理的折衷是可能的。这项工作是对那些展示如何私下执行身份验证的工作的补充，因此可以被视为迈向完全私密自动说话人识别的又一步。



## **12. BERT Rankers are Brittle: a Study using Adversarial Document Perturbations**

Bert Rankers是脆弱的：一项使用对抗性文件扰动的研究 cs.IR

To appear in ICTIR 2022

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11724v1)

**Authors**: Yumeng Wang, Lijun Lyu, Avishek Anand

**Abstracts**: Contextual ranking models based on BERT are now well established for a wide range of passage and document ranking tasks. However, the robustness of BERT-based ranking models under adversarial inputs is under-explored. In this paper, we argue that BERT-rankers are not immune to adversarial attacks targeting retrieved documents given a query. Firstly, we propose algorithms for adversarial perturbation of both highly relevant and non-relevant documents using gradient-based optimization methods. The aim of our algorithms is to add/replace a small number of tokens to a highly relevant or non-relevant document to cause a large rank demotion or promotion. Our experiments show that a small number of tokens can already result in a large change in the rank of a document. Moreover, we find that BERT-rankers heavily rely on the document start/head for relevance prediction, making the initial part of the document more susceptible to adversarial attacks. More interestingly, we find a small set of recurring adversarial words that when added to documents result in successful rank demotion/promotion of any relevant/non-relevant document respectively. Finally, our adversarial tokens also show particular topic preferences within and across datasets, exposing potential biases from BERT pre-training or downstream datasets.

摘要: 基于BERT的上下文排名模型现在已经很好地建立了用于广泛的段落和文档排名任务。然而，基于BERT的排序模型在对抗性输入下的稳健性还没有得到充分的研究。在这篇文章中，我们认为BERT排名者也不能幸免于针对给定查询的检索文档的对抗性攻击。首先，我们使用基于梯度的优化方法提出了针对高度相关和无关文档的对抗性扰动的算法。我们的算法的目的是在高度相关或不相关的文档中添加/替换少量令牌，从而导致较大的排名降级或提升。我们的实验表明，少量的标记已经可以导致文档的排名发生很大的变化。此外，我们发现BERT排名者严重依赖文档START/HEAD进行相关性预测，使得文档的开头部分更容易受到对手攻击。更有趣的是，我们发现了一小部分重复出现的敌意单词，当添加到文档中时，这些单词会分别成功地对任何相关/不相关的文档进行排名降级/提升。最后，我们的对抗性令牌还显示了数据集内和数据集之间的特定主题偏好，暴露了来自BERT预训练或下游数据集的潜在偏差。



## **13. Adversarial Zoom Lens: A Novel Physical-World Attack to DNNs**

对抗性变焦镜头：一种新的物理世界对DNN的攻击 cs.CR

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12251v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Although deep neural networks (DNNs) are known to be fragile, no one has studied the effects of zooming-in and zooming-out of images in the physical world on DNNs performance. In this paper, we demonstrate a novel physical adversarial attack technique called Adversarial Zoom Lens (AdvZL), which uses a zoom lens to zoom in and out of pictures of the physical world, fooling DNNs without changing the characteristics of the target object. The proposed method is so far the only adversarial attack technique that does not add physical adversarial perturbation attack DNNs. In a digital environment, we construct a data set based on AdvZL to verify the antagonism of equal-scale enlarged images to DNNs. In the physical environment, we manipulate the zoom lens to zoom in and out of the target object, and generate adversarial samples. The experimental results demonstrate the effectiveness of AdvZL in both digital and physical environments. We further analyze the antagonism of the proposed data set to the improved DNNs. On the other hand, we provide a guideline for defense against AdvZL by means of adversarial training. Finally, we look into the threat possibilities of the proposed approach to future autonomous driving and variant attack ideas similar to the proposed attack.

摘要: 尽管深度神经网络(DNN)被认为是脆弱的，但还没有人研究物理世界中图像的放大和缩小对DNN性能的影响。在本文中，我们展示了一种新的物理对抗性攻击技术，称为对抗性变焦镜头(AdvZL)，它使用变焦镜头来放大和缩小物理世界的图像，在不改变目标对象特征的情况下愚弄DNN。该方法是迄今为止唯一一种不添加物理对抗性扰动攻击DNN的对抗性攻击技术。在数字环境下，我们构建了一个基于AdvZL的数据集，以验证等比例放大图像对DNN的对抗。在物理环境中，我们操纵变焦镜头来放大和缩小目标对象，并生成对抗性样本。实验结果证明了AdvZL在数字和物理环境中的有效性。我们进一步分析了所提出的数据集对改进的DNN的对抗性。另一方面，我们通过对抗性训练的方式提供了防御AdvZL的指导方针。最后，我们展望了所提出的方法对未来自动驾驶的威胁可能性，以及类似于所提出的攻击的不同攻击思想。



## **14. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

6 pages workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.06761v2)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **15. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的边界训练数据重构 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2201.12383v4)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 在ML中，差异隐私被广泛接受为防止数据泄露的事实上的方法，传统观点认为，它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。本文首先在形式化威胁模型下给出了DP机制抵抗训练数据重构攻击的语义保证。我们发现，两种不同的隐私记账方法--Renyi Differential Privacy和Fisher信息泄漏--都提供了对数据重构攻击的强大语义保护。



## **16. A Framework for Understanding Model Extraction Attack and Defense**

一种理解模型提取攻击与防御的框架 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11480v1)

**Authors**: Xun Xian, Mingyi Hong, Jie Ding

**Abstracts**: The privacy of machine learning models has become a significant concern in many emerging Machine-Learning-as-a-Service applications, where prediction services based on well-trained models are offered to users via pay-per-query. The lack of a defense mechanism can impose a high risk on the privacy of the server's model since an adversary could efficiently steal the model by querying only a few `good' data points. The interplay between a server's defense and an adversary's attack inevitably leads to an arms race dilemma, as commonly seen in Adversarial Machine Learning. To study the fundamental tradeoffs between model utility from a benign user's view and privacy from an adversary's view, we develop new metrics to quantify such tradeoffs, analyze their theoretical properties, and develop an optimization problem to understand the optimal adversarial attack and defense strategies. The developed concepts and theory match the empirical findings on the `equilibrium' between privacy and utility. In terms of optimization, the key ingredient that enables our results is a unified representation of the attack-defense problem as a min-max bi-level problem. The developed results will be demonstrated by examples and experiments.

摘要: 在许多新兴的机器学习即服务应用中，机器学习模型的隐私已经成为一个重要的问题，其中基于训练有素的模型的预测服务通过按查询付费的方式提供给用户。缺乏防御机制可能会给服务器模型的隐私带来很高的风险，因为攻击者只需查询几个“好”的数据点就可以有效地窃取模型。服务器的防御和对手的攻击之间的相互作用不可避免地导致了军备竞赛的两难境地，就像在对抗性机器学习中常见的那样。为了从良性用户的角度研究模型效用和从对手的角度研究隐私之间的基本权衡，我们开发了新的度量来量化这种权衡，分析了它们的理论性质，并开发了一个优化问题来理解最优的对抗性攻击和防御策略。所发展的概念和理论与关于隐私和效用之间的“平衡”的经验研究结果相吻合。在优化方面，使我们的结果得以实现的关键因素是将攻防问题统一表示为最小-最大双层问题。所开发的结果将通过实例和实验进行验证。



## **17. InfoAT: Improving Adversarial Training Using the Information Bottleneck Principle**

InfoAT：利用信息瓶颈原理改进对抗性训练 cs.LG

Published in: IEEE Transactions on Neural Networks and Learning  Systems ( Early Access )

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.12292v1)

**Authors**: Mengting Xu, Tao Zhang, Zhongnian Li, Daoqiang Zhang

**Abstracts**: Adversarial training (AT) has shown excellent high performance in defending against adversarial examples. Recent studies demonstrate that examples are not equally important to the final robustness of models during AT, that is, the so-called hard examples that can be attacked easily exhibit more influence than robust examples on the final robustness. Therefore, guaranteeing the robustness of hard examples is crucial for improving the final robustness of the model. However, defining effective heuristics to search for hard examples is still difficult. In this article, inspired by the information bottleneck (IB) principle, we uncover that an example with high mutual information of the input and its associated latent representation is more likely to be attacked. Based on this observation, we propose a novel and effective adversarial training method (InfoAT). InfoAT is encouraged to find examples with high mutual information and exploit them efficiently to improve the final robustness of models. Experimental results show that InfoAT achieves the best robustness among different datasets and models in comparison with several state-of-the-art methods.

摘要: 对抗性训练(AT)在防御对抗性例子方面表现出了出色的高性能。最近的研究表明，在AT过程中，样本对模型的最终稳健性并不是同样重要，即所谓的易受攻击的硬样本对最终稳健性的影响比健壮样本更大。因此，保证硬样本的稳健性是提高模型最终稳健性的关键。然而，定义有效的启发式算法来搜索困难的例子仍然是困难的。在本文中，受信息瓶颈(IB)原理的启发，我们发现输入及其关联的潜在表示具有高互信息的示例更容易受到攻击。基于此，我们提出了一种新颖而有效的对抗性训练方法(InfoAT)。InfoAT被鼓励寻找具有高互信息的例子，并有效地利用它们来提高模型的最终稳健性。实验结果表明，与几种最先进的方法相比，InfoAT在不同的数据集和模型中获得了最好的稳健性。



## **18. Incorporating Hidden Layer representation into Adversarial Attacks and Defences**

在对抗性攻击和防御中引入隐藏层表示 cs.LG

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2011.14045v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: In this paper, we propose a defence strategy to improve adversarial robustness by incorporating hidden layer representation. The key of this defence strategy aims to compress or filter input information including adversarial perturbation. And this defence strategy can be regarded as an activation function which can be applied to any kind of neural network. We also prove theoretically the effectiveness of this defense strategy under certain conditions. Besides, incorporating hidden layer representation we propose three types of adversarial attacks to generate three types of adversarial examples, respectively. The experiments show that our defence method can significantly improve the adversarial robustness of deep neural networks which achieves the state-of-the-art performance even though we do not adopt adversarial training.

摘要: 在本文中，我们提出了一种防御策略，通过引入隐含层表示来提高对手的稳健性。这种防御策略的关键是压缩或过滤输入信息，包括对抗性扰动。这种防御策略可以看作是一种激活函数，可以应用于任何类型的神经网络。在一定条件下，我们还从理论上证明了该防御策略的有效性。此外，结合隐含层表示，我们提出了三种类型的对抗性攻击，分别生成三种类型的对抗性实例。实验表明，在不采用对抗性训练的情况下，我们的防御方法能够显着提高深层神经网络的对抗性，达到了最好的性能。



## **19. Adversarial Learning with Cost-Sensitive Classes**

成本敏感类的对抗性学习 cs.LG

12 pages

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2101.12372v2)

**Authors**: Haojing Shen, Sihong Chen, Ran Wang, Xizhao Wang

**Abstracts**: It is necessary to improve the performance of some special classes or to particularly protect them from attacks in adversarial learning. This paper proposes a framework combining cost-sensitive classification and adversarial learning together to train a model that can distinguish between protected and unprotected classes, such that the protected classes are less vulnerable to adversarial examples. We find in this framework an interesting phenomenon during the training of deep neural networks, called Min-Max property, that is, the absolute values of most parameters in the convolutional layer approach zero while the absolute values of a few parameters are significantly larger becoming bigger. Based on this Min-Max property which is formulated and analyzed in a view of random distribution, we further build a new defense model against adversarial examples for adversarial robustness improvement. An advantage of the built model is that it performs better than the standard one and can combine with adversarial training to achieve an improved performance. It is experimentally confirmed that, regarding the average accuracy of all classes, our model is almost as same as the existing models when an attack does not occur and is better than the existing models when an attack occurs. Specifically, regarding the accuracy of protected classes, the proposed model is much better than the existing models when an attack occurs.

摘要: 在对抗性学习中，有必要提高某些特殊类的表现，或特别保护它们免受攻击。本文提出了一种结合代价敏感分类和对抗性学习的框架，以训练一个能够区分保护类和非保护类的模型，从而使受保护类不太容易受到对抗性例子的影响。在该框架中，我们发现了深层神经网络训练过程中一个有趣的现象，称为Min-Max性质，即卷积层中大部分参数的绝对值趋于零，而少数参数的绝对值明显变大。基于这种从随机分布的角度来描述和分析的Min-Max性质，我们进一步构建了一种新的对抗实例的防御模型，以提高对抗的健壮性。建立的模型的一个优点是它的性能比标准模型更好，并且可以与对抗性训练相结合，以实现更好的性能。实验证实，在所有类别的平均准确率方面，当攻击不发生时，我们的模型与现有模型几乎相同，而当攻击发生时，我们的模型优于现有模型。具体地说，在保护类的准确性方面，当攻击发生时，所提出的模型比现有的模型要好得多。



## **20. Shilling Black-box Recommender Systems by Learning to Generate Fake User Profiles**

通过学习生成虚假用户配置文件来攻击黑盒推荐系统 cs.IR

Accepted by TNNLS. 15 pages, 8 figures

**SubmitDate**: 2022-06-23    [paper-pdf](http://arxiv.org/pdf/2206.11433v1)

**Authors**: Chen Lin, Si Chen, Meifang Zeng, Sheng Zhang, Min Gao, Hui Li

**Abstracts**: Due to the pivotal role of Recommender Systems (RS) in guiding customers towards the purchase, there is a natural motivation for unscrupulous parties to spoof RS for profits. In this paper, we study Shilling Attack where an adversarial party injects a number of fake user profiles for improper purposes. Conventional Shilling Attack approaches lack attack transferability (i.e., attacks are not effective on some victim RS models) and/or attack invisibility (i.e., injected profiles can be easily detected). To overcome these issues, we present Leg-UP, a novel attack model based on the Generative Adversarial Network. Leg-UP learns user behavior patterns from real users in the sampled ``templates'' and constructs fake user profiles. To simulate real users, the generator in Leg-UP directly outputs discrete ratings. To enhance attack transferability, the parameters of the generator are optimized by maximizing the attack performance on a surrogate RS model. To improve attack invisibility, Leg-UP adopts a discriminator to guide the generator to generate undetectable fake user profiles. Experiments on benchmarks have shown that Leg-UP exceeds state-of-the-art Shilling Attack methods on a wide range of victim RS models. The source code of our work is available at: https://github.com/XMUDM/ShillingAttack.

摘要: 由于推荐系统在引导消费者购买方面起着举足轻重的作用，不道德的人有一个自然的动机来欺骗推荐系统以获取利润。在这篇文章中，我们研究了恶意用户出于不正当目的注入大量虚假用户配置文件的先令攻击。传统的先令攻击方法缺乏攻击的可转移性(即，攻击在某些受害者RS模型上无效)和/或攻击的不可见性(即，可以很容易地检测到注入的配置文件)。为了克服这些问题，我们提出了一种基于产生式对抗网络的新型攻击模型--LEG-UP。Leg-Up从采样的“模板”中的真实用户那里学习用户行为模式，并构建虚假的用户配置文件。为了模拟真实用户，立式发电机直接输出离散的额定值。为了增强攻击的可转移性，在代理RS模型上通过最大化攻击性能来优化生成器的参数。为了提高攻击的不可见性，Leg-Up采用了一个鉴别器来引导生成器生成无法检测的虚假用户配置文件。基准测试实验表明，在广泛的受害者RS模型上，Leg-Up攻击方法超过了最先进的先令攻击方法。我们工作的源代码可以在https://github.com/XMUDM/ShillingAttack.上找到



## **21. Making Generated Images Hard To Spot: A Transferable Attack On Synthetic Image Detectors**

使生成的图像难以识别：对合成图像检测器的可转移攻击 cs.CV

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2104.12069v2)

**Authors**: Xinwei Zhao, Matthew C. Stamm

**Abstracts**: Visually realistic GAN-generated images have recently emerged as an important misinformation threat. Research has shown that these synthetic images contain forensic traces that are readily identifiable by forensic detectors. Unfortunately, these detectors are built upon neural networks, which are vulnerable to recently developed adversarial attacks. In this paper, we propose a new anti-forensic attack capable of fooling GAN-generated image detectors. Our attack uses an adversarially trained generator to synthesize traces that these detectors associate with real images. Furthermore, we propose a technique to train our attack so that it can achieve transferability, i.e. it can fool unknown CNNs that it was not explicitly trained against. We evaluate our attack through an extensive set of experiments, where we show that our attack can fool eight state-of-the-art detection CNNs with synthetic images created using seven different GANs, and outperform other alternative attacks.

摘要: 视觉逼真的GaN生成的图像最近已经成为一种重要的错误信息威胁。研究表明，这些合成图像包含法医探测器可以很容易识别的法医痕迹。不幸的是，这些探测器是建立在神经网络的基础上的，而神经网络很容易受到最近发展起来的对抗性攻击。在本文中，我们提出了一种新的反取证攻击，能够欺骗GaN生成的图像检测器。我们的攻击使用一个经过敌意训练的生成器来合成这些检测器与真实图像相关联的痕迹。此外，我们还提出了一种技术来训练我们的攻击，以便它能够实现可转移性，即它可以欺骗它没有明确训练针对的未知CNN。我们通过一组广泛的实验来评估我们的攻击，其中我们表明我们的攻击可以通过使用7个不同的GAN创建的合成图像来欺骗8个最先进的检测CNN，并且性能优于其他替代攻击。



## **22. AdvSmo: Black-box Adversarial Attack by Smoothing Linear Structure of Texture**

AdvSmo：平滑纹理线性结构的黑盒对抗性攻击 cs.CV

6 pages,3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10988v1)

**Authors**: Hui Xia, Rui Zhang, Shuliang Jiang, Zi Kang

**Abstracts**: Black-box attacks usually face two problems: poor transferability and the inability to evade the adversarial defense. To overcome these shortcomings, we create an original approach to generate adversarial examples by smoothing the linear structure of the texture in the benign image, called AdvSmo. We construct the adversarial examples without relying on any internal information to the target model and design the imperceptible-high attack success rate constraint to guide the Gabor filter to select appropriate angles and scales to smooth the linear texture from the input images to generate adversarial examples. Benefiting from the above design concept, AdvSmo will generate adversarial examples with strong transferability and solid evasiveness. Finally, compared to the four advanced black-box adversarial attack methods, for the eight target models, the results show that AdvSmo improves the average attack success rate by 9% on the CIFAR-10 and 16% on the Tiny-ImageNet dataset compared to the best of these attack methods.

摘要: 黑盒攻击通常面临两个问题：可转移性差和无法躲避对手的防御。为了克服这些缺点，我们创建了一种新颖的方法，通过平滑良性图像中纹理的线性结构来生成对抗性示例，称为AdvSmo。我们在不依赖目标模型任何内部信息的情况下构造敌意样本，并设计了不可察觉的高攻击成功率约束来指导Gabor滤波器选择合适的角度和尺度来平滑输入图像中的线性纹理来生成敌意样本。受益于上述设计理念，AdvSmo将生成具有很强的可转移性和坚实的规避能力的对抗性范例。最后，与四种先进的黑盒对抗攻击方法相比，对于8个目标模型，结果表明，AdvSmo在CIFAR-10上的平均攻击成功率比这些攻击方法中最好的方法提高了9%，在Tiny-ImageNet数据集上的平均攻击成功率提高了16%。



## **23. Adversarial Reconfigurable Intelligent Surface Against Physical Layer Key Generation**

对抗物理层密钥生成的对抗性可重构智能表面 eess.SP

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10955v1)

**Authors**: Zhuangkun Wei, Bin Li, Weisi Guo

**Abstracts**: The development of reconfigurable intelligent surface (RIS) has recently advanced the research of physical layer security (PLS). Beneficial impact of RIS includes but is not limited to offering a new domain of freedom (DoF) for key-less PLS optimization, and increasing channel randomness for physical layer secret key generation (PL-SKG). However, there is a lack of research studying how adversarial RIS can be used to damage the communication confidentiality. In this work, we show how a Eve controlled adversarial RIS (Eve-RIS) can be used to reconstruct the shared PLS secret key between legitimate users (Alice and Bob). This is achieved by Eve-RIS overlaying the legitimate channel with an artificial random and reciprocal channel. The resulting Eve-RIS corrupted channel enable Eve to successfully attack the PL-SKG process. To operationalize this novel concept, we design Eve-RIS schemes against two PL-SKG techniques used: (i) the channel estimation based PL-SKG, and (ii) the two-way cross multiplication based PL-SKG. Our results show a high key match rate between the designed Eve-RIS and the legitimate users. We also present theoretical key match rate between Eve-RIS and legitimate users. Our novel scheme is different from the existing spoofing-Eve, in that the latter can be easily detected by comparing the channel estimation results of the legitimate users. Indeed, our proposed Eve-RIS can maintain the legitimate channel reciprocity, which makes detection challenging. This means the novel Eve-RIS provides a new eavesdropping threat on PL-SKG, which can spur new research areas to counter adversarial RIS attacks.

摘要: 近年来，可重构智能表面(RIS)的发展推动了物理层安全(PLS)的研究。RIS的有益影响包括但不限于提供用于无密钥的PLS优化的新的自由域(DoF)，以及增加用于物理层秘密密钥生成(PL-SKG)的信道随机性。然而，缺乏研究如何利用敌意RIS来破坏通信机密性。在这项工作中，我们展示了如何使用Eve控制的对抗RIS(Eve-RIS)来重构合法用户(Alice和Bob)之间共享的PLS秘密密钥。这是通过EVE-RIS用人工随机和互惠的信道覆盖合法信道来实现的。由此产生的Eve-RIS损坏的通道使Eve能够成功攻击PL-SKG进程。为了实现这一新概念，我们针对使用的两种PL-SKG技术设计了Eve-RIS方案：(I)基于信道估计的PL-SKG和(Ii)基于双向交叉乘法的PL-SKG。结果表明，所设计的Eve-RIS与合法用户之间具有较高的密钥匹配率。我们还给出了Eve-RIS与合法用户之间的理论密钥匹配率。我们的新方案不同于现有的欺骗-EVE方案，后者可以通过比较合法用户的信道估计结果来容易地检测到。事实上，我们提出的Eve-RIS能够保持合法的信道互惠，这使得检测具有挑战性。这意味着新的Eve-RIS为PL-SKG提供了一种新的窃听威胁，可以刺激新的研究领域来对抗敌意RIS攻击。



## **24. Introduction to Machine Learning for the Sciences**

面向科学的机器学习导论 physics.comp-ph

84 pages, 37 figures. The content of these lecture notes together  with exercises is available under http://www.ml-lectures.org. A shorter  German version of the lecture notes is published in the Springer essential  series, ISBN 978-3-658-32268-7, doi:10.1007/978-3-658-32268-7

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2102.04883v2)

**Authors**: Titus Neupert, Mark H Fischer, Eliska Greplova, Kenny Choo, M. Michael Denner

**Abstracts**: This is an introductory machine-learning course specifically developed with STEM students in mind. Our goal is to provide the interested reader with the basics to employ machine learning in their own projects and to familiarize themself with the terminology as a foundation for further reading of the relevant literature. In these lecture notes, we discuss supervised, unsupervised, and reinforcement learning. The notes start with an exposition of machine learning methods without neural networks, such as principle component analysis, t-SNE, clustering, as well as linear regression and linear classifiers. We continue with an introduction to both basic and advanced neural-network structures such as dense feed-forward and conventional neural networks, recurrent neural networks, restricted Boltzmann machines, (variational) autoencoders, generative adversarial networks. Questions of interpretability are discussed for latent-space representations and using the examples of dreaming and adversarial attacks. The final section is dedicated to reinforcement learning, where we introduce basic notions of value functions and policy learning.

摘要: 这是一门专门为STEM学生开发的机器学习入门课程。我们的目标是为感兴趣的读者提供在他们自己的项目中使用机器学习的基础知识，并熟悉这些术语作为进一步阅读相关文献的基础。在这些课堂讲稿中，我们讨论有监督、无监督和强化学习。这些笔记首先阐述了没有神经网络的机器学习方法，如主成分分析、t-SNE、聚类以及线性回归和线性分类器。我们继续介绍基本和高级神经网络结构，如密集前馈和常规神经网络、递归神经网络、受限Boltzmann机器、(变分)自动编码器、生成性对手网络。讨论了潜在空间表示的可解释性问题，并使用了梦和对抗性攻击的例子。最后一节致力于强化学习，在那里我们介绍价值函数和策略学习的基本概念。



## **25. Guided Diffusion Model for Adversarial Purification from Random Noise**

随机噪声中对抗性净化的引导扩散模型 cs.LG

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10875v1)

**Authors**: Quanlin Wu, Hang Ye, Yuntian Gu

**Abstracts**: In this paper, we propose a novel guided diffusion purification approach to provide a strong defense against adversarial attacks. Our model achieves 89.62% robust accuracy under PGD-L_inf attack (eps = 8/255) on the CIFAR-10 dataset. We first explore the essential correlations between unguided diffusion models and randomized smoothing, enabling us to apply the models to certified robustness. The empirical results show that our models outperform randomized smoothing by 5% when the certified L2 radius r is larger than 0.5.

摘要: 在本文中，我们提出了一种新的引导扩散净化方法，以提供对对手攻击的强大防御。在Pgd-L_inf攻击(EPS=8/255)下，我们的模型在CIFAR-10数据集上达到了89.62%的稳健准确率。我们首先探讨了无引导扩散模型和随机平滑之间的本质关联，使我们能够将这些模型应用于已证明的稳健性。实证结果表明，当认证的L2半径r大于0.5时，我们的模型比随机平滑的性能高出5%。



## **26. Robust Universal Adversarial Perturbations**

稳健的普遍对抗性摄动 cs.LG

16 pages, 3 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10858v1)

**Authors**: Changming Xu, Gagandeep Singh

**Abstracts**: Universal Adversarial Perturbations (UAPs) are imperceptible, image-agnostic vectors that cause deep neural networks (DNNs) to misclassify inputs from a data distribution with high probability. Existing methods do not create UAPs robust to transformations, thereby limiting their applicability as a real-world attacks. In this work, we introduce a new concept and formulation of robust universal adversarial perturbations. Based on our formulation, we build a novel, iterative algorithm that leverages probabilistic robustness bounds for generating UAPs robust against transformations generated by composing arbitrary sub-differentiable transformation functions. We perform an extensive evaluation on the popular CIFAR-10 and ILSVRC 2012 datasets measuring robustness under human-interpretable semantic transformations, such as rotation, contrast changes, etc, that are common in the real-world. Our results show that our generated UAPs are significantly more robust than those from baselines.

摘要: 通用对抗性扰动(UAP)是一种不可察觉的、与图像无关的向量，它会导致深度神经网络(DNN)高概率地对来自数据分布的输入进行错误分类。现有方法不能创建对变换具有健壮性的UAP，从而限制了它们作为现实世界攻击的适用性。在这项工作中，我们引入了一个新的概念和形式，稳健的泛对抗摄动。基于我们的公式，我们构建了一种新颖的迭代算法，该算法利用概率鲁棒界来生成对通过合成任意次可微变换函数而产生的变换具有鲁棒性的UAP。我们在流行的CIFAR-10和ILSVRC 2012数据集上进行了广泛的评估，测量了在人类可解释的语义转换下的健壮性，例如旋转、对比度变化等，这些转换在现实世界中很常见。我们的结果表明，我们生成的UAP明显比基线生成的UAP更健壮。



## **27. Adaptive Adversarial Training to Improve Adversarial Robustness of DNNs for Medical Image Segmentation and Detection**

自适应对抗训练提高DNN在医学图像分割和检测中的对抗鲁棒性 eess.IV

17 pages

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.01736v2)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: It is known that Deep Neural networks (DNNs) are vulnerable to adversarial attacks, and the adversarial robustness of DNNs could be improved by adding adversarial noises to training data (e.g., the standard adversarial training (SAT)). However, inappropriate noises added to training data may reduce a model's performance, which is termed the trade-off between accuracy and robustness. This problem has been sufficiently studied for the classification of whole images but has rarely been explored for image analysis tasks in the medical application domain, including image segmentation, landmark detection, and object detection tasks. In this study, we show that, for those medical image analysis tasks, the SAT method has a severe issue that limits its practical use: it generates a fixed and unified level of noise for all training samples for robust DNN training. A high noise level may lead to a large reduction in model performance and a low noise level may not be effective in improving robustness. To resolve this issue, we design an adaptive-margin adversarial training (AMAT) method that generates sample-wise adaptive adversarial noises for robust DNN training. In contrast to the existing, classification-oriented adversarial training methods, our AMAT method uses a loss-defined-margin strategy so that it can be applied to different tasks as long as the loss functions are well-defined. We successfully apply our AMAT method to state-of-the-art DNNs, using five publicly available datasets. The experimental results demonstrate that: (1) our AMAT method can be applied to the three seemingly different tasks in the medical image application domain; (2) AMAT outperforms the SAT method in adversarial robustness; (3) AMAT has a minimal reduction in prediction accuracy on clean data, compared with the SAT method; and (4) AMAT has almost the same training time cost as SAT.

摘要: 众所周知，深度神经网络(DNN)容易受到对抗性攻击，通过在训练数据(例如标准对抗性训练(SAT))中添加对抗性噪声可以提高DNN的对抗性健壮性。然而，在训练数据中添加不适当的噪声可能会降低模型的性能，这被称为精度和稳健性之间的权衡。对于整个图像的分类，这个问题已经得到了充分的研究，但在医学应用领域的图像分析任务中，包括图像分割、地标检测和目标检测任务，却很少被探索。在这项研究中，我们发现，对于这些医学图像分析任务，SAT方法有一个严重的问题限制了它的实际应用：它为所有训练样本生成固定和统一的噪声水平，以便进行稳健的DNN训练。较高的噪声水平可能会导致模型性能的大幅降低，而较低的噪声水平可能不能有效地提高鲁棒性。为了解决这一问题，我们设计了一种自适应差值对抗性训练(AMAT)方法，该方法产生样本级自适应对抗性噪声，用于稳健的DNN训练。与现有的面向分类的对抗性训练方法相比，我们的AMAT方法使用了损失定义边际策略，因此只要损失函数定义得很好，它就可以应用于不同的任务。我们成功地将我们的AMAT方法应用于最先进的DNN，使用了五个公开可用的数据集。实验结果表明：(1)我们的AMAT方法可以应用于医学图像应用领域中三个看似不同的任务；(2)AMAT方法在对抗健壮性方面优于SAT方法；(3)与SAT方法相比，AMAT方法对干净数据的预测精度有很小的降低；(4)AMAT方法的训练时间开销与SAT方法几乎相同。



## **28. SSMI: How to Make Objects of Interest Disappear without Accessing Object Detectors?**

SSMI：如何在不访问对象探测器的情况下使感兴趣的对象消失？ cs.CV

6 pages, 2 figures

**SubmitDate**: 2022-06-22    [paper-pdf](http://arxiv.org/pdf/2206.10809v1)

**Authors**: Hui Xia, Rui Zhang, Zi Kang, Shuliang Jiang

**Abstracts**: Most black-box adversarial attack schemes for object detectors mainly face two shortcomings: requiring access to the target model and generating inefficient adversarial examples (failing to make objects disappear in large numbers). To overcome these shortcomings, we propose a black-box adversarial attack scheme based on semantic segmentation and model inversion (SSMI). We first locate the position of the target object using semantic segmentation techniques. Next, we design a neighborhood background pixel replacement to replace the target region pixels with background pixels to ensure that the pixel modifications are not easily detected by human vision. Finally, we reconstruct a machine-recognizable example and use the mask matrix to select pixels in the reconstructed example to modify the benign image to generate an adversarial example. Detailed experimental results show that SSMI can generate efficient adversarial examples to evade human-eye perception and make objects of interest disappear. And more importantly, SSMI outperforms existing same kinds of attacks. The maximum increase in new and disappearing labels is 16%, and the maximum decrease in mAP metrics for object detection is 36%.

摘要: 大多数针对目标探测器的黑盒对抗性攻击方案主要面临两个缺点：需要访问目标模型和生成低效的对抗性实例(无法使对象大量消失)。为了克服这些不足，我们提出了一种基于语义分割和模型反转的黑盒对抗攻击方案。我们首先使用语义分割技术定位目标对象的位置。接下来，我们设计了一种邻域背景像素替换算法，将目标区域的像素替换为背景像素，以保证像素的变化不易被人的视觉检测到。最后，我们重建一个机器可识别的样本，并使用掩码矩阵来选择重建样本中的像素来修正良性图像以生成对抗性样本。详细的实验结果表明，SSMI能够生成有效的对抗性实例，避开人眼的感知，使感兴趣的对象消失。更重要的是，SSMI的性能优于现有的同类攻击。新的和消失的标签的最大增幅为16%，用于目标检测的MAP指标的最大降幅为36%。



## **29. Secure and Efficient Query Processing in Outsourced Databases**

外包数据库中安全高效的查询处理 cs.CR

Ph.D. thesis

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10753v1)

**Authors**: Dmytro Bogatov

**Abstracts**: Various cryptographic techniques are used in outsourced database systems to ensure data privacy while allowing for efficient querying. This work proposes a definition and components of a new secure and efficient outsourced database system, which answers various types of queries, with different privacy guarantees in different security models. This work starts with the survey of five order-revealing encryption schemes that can be used directly in many database indices and five range query protocols with various security / efficiency tradeoffs. The survey systematizes the state-of-the-art range query solutions in a snapshot adversary setting and offers some non-obvious observations regarding the efficiency of the constructions. In $\mathcal{E}\text{psolute}$, a secure range query engine, security is achieved in a setting with a much stronger adversary where she can continuously observe everything on the server, and leaking even the result size can enable a reconstruction attack. $\mathcal{E}\text{psolute}$ proposes a definition, construction, analysis, and experimental evaluation of a system that provably hides both access pattern and communication volume while remaining efficient. The work concludes with $k\text{-a}n\text{o}n$ -- a secure similarity search engine in a snapshot adversary model. The work presents a construction in which the security of $k\text{NN}$ queries is achieved similarly to OPE / ORE solutions -- encrypting the input with an approximate Distance Comparison Preserving Encryption scheme so that the inputs, the points in a hyperspace, are perturbed, but the query algorithm still produces accurate results. We use TREC datasets and queries for the search, and track the rank quality metrics such as MRR and nDCG. For the attacks, we build an LSTM model that trains on the correlation between a sentence and its embedding and then predicts words from the embedding.

摘要: 在外包数据库系统中使用了各种加密技术，以确保数据隐私，同时允许高效查询。提出了一种新的安全高效的外包数据库系统的定义和组成，该系统可以回答不同类型的查询，在不同的安全模型下具有不同的隐私保障。这项工作首先调查了五种可以直接用于许多数据库索引的顺序揭示加密方案和五种具有各种安全/效率权衡的范围查询协议。该调查将快照对手环境中最先进的范围查询解决方案系统化，并提供了一些关于构造效率的不明显的观察。在安全范围查询引擎$\Mathcal{E}\Text{psolte}$中，安全是在一个更强大的对手的设置下实现的，在这种设置中，她可以连续观察服务器上的一切，即使泄漏结果大小也可能导致重建攻击。$\Mathcal{E}\Text{psolte}$提出了一种系统的定义、构造、分析和实验评估，该系统可以证明在保持效率的同时隐藏了访问模式和通信量。这项工作以$k\Text{-a}n\Text{o}n$结束--快照对手模型中的安全相似性搜索引擎。该工作提出了一种结构，其中$k\Text{NN}$查询的安全性类似于OPE/ORE解决方案--使用一种保持近似距离比较的加密方案对输入进行加密，使得输入，即超空间中的点，被扰动，但查询算法仍然产生准确的结果。我们使用TREC数据集和查询进行搜索，并跟踪排名质量度量，如MRR和nDCG。对于攻击，我们建立了一个LSTM模型，该模型根据句子与其嵌入之间的相关性进行训练，然后从嵌入中预测单词。



## **30. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

FlashSyn：基于反例驱动近似的闪贷攻击合成 cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10708v1)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate DeFi protocol functional behaviors using numerical methods. Then, we propose a novel algorithm to find an adversarial attack which constitutes of a sequence of invocations of functions in a DeFi protocol with the optimized parameters for profits. We implemented our framework in a tool called FlashSyn. We run FlashSyn on 5 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.

摘要: 在去中心化金融(Defi)生态系统中，贷款人可以向借款人提供闪存贷款，即仅在区块链交易中有效且必须在该交易结束前支付一定费用的贷款。与正常贷款不同，闪付贷款允许借款人借入大量资产，而无需预付抵押金。恶意攻击者可以使用闪存贷款来收集大量资产，以发起针对Defi协议的代价高昂的攻击。在这篇文章中，我们介绍了一个新的框架，用于自动合成利用闪存贷款的Defi协议的对抗性合同。为了绕过DEFI协议的复杂性，我们提出了一种利用数值方法来近似DEFI协议功能行为的新技术。然后，我们提出了一种新的算法来发现敌意攻击，该攻击由Defi协议中的一系列函数调用组成，并通过优化参数来获利。我们在一个名为FlashSyn的工具中实现了我们的框架。我们在5个Defi协议上运行FlashSyn，这些协议是闪电贷款攻击的受害者，并且是来自该死的脆弱Defi挑战的Defi协议。FlashSyn会自动为它们中的每一个合成一次对抗性攻击。



## **31. Using EBGAN for Anomaly Intrusion Detection**

利用EBGAN进行异常入侵检测 cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10400v1)

**Authors**: Yi Cui, Wenfeng Shen, Jian Zhang, Weijia Lu, Chuang Liu, Lin Sun, Si Chen

**Abstracts**: As an active network security protection scheme, intrusion detection system (IDS) undertakes the important responsibility of detecting network attacks in the form of malicious network traffic. Intrusion detection technology is an important part of IDS. At present, many scholars have carried out extensive research on intrusion detection technology. However, developing an efficient intrusion detection method for massive network traffic data is still difficult. Since Generative Adversarial Networks (GANs) have powerful modeling capabilities for complex high-dimensional data, they provide new ideas for addressing this problem. In this paper, we put forward an EBGAN-based intrusion detection method, IDS-EBGAN, that classifies network records as normal traffic or malicious traffic. The generator in IDS-EBGAN is responsible for converting the original malicious network traffic in the training set into adversarial malicious examples. This is because we want to use adversarial learning to improve the ability of discriminator to detect malicious traffic. At the same time, the discriminator adopts Autoencoder model. During testing, IDS-EBGAN uses reconstruction error of discriminator to classify traffic records.

摘要: 入侵检测系统作为一种主动的网络安全防护方案，担负着检测以恶意网络流量为形式的网络攻击的重要责任。入侵检测技术是入侵检测系统的重要组成部分。目前，许多学者对入侵检测技术进行了广泛的研究。然而，开发一种高效的针对海量网络流量数据的入侵检测方法仍然是一个难点。由于生成性对抗网络(GAN)对复杂的高维数据具有强大的建模能力，它们为解决这一问题提供了新的思路。本文提出了一种基于EBGAN的入侵检测方法--IDS-EBGAN，将网络记录分为正常流量和恶意流量。入侵检测系统EBGAN中的生成器负责将训练集中的原始恶意网络流量转换为对抗性恶意实例。这是因为我们希望利用对抗性学习来提高鉴别器检测恶意流量的能力。同时，该鉴别器采用自动编码器模型。在测试过程中，IDS-EBGAN利用识别器的重构误差对流量记录进行分类。



## **32. Problem-Space Evasion Attacks in the Android OS: a Survey**

Android操作系统中的问题空间逃避攻击：综述 cs.CR

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2205.14576v2)

**Authors**: Harel Berger, Chen Hajaj, Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.

摘要: 安卓是全球最受欢迎的操作系统。因此，它是各种恶意软件的目标。作为对策，安全社区夜以继日地开发合适的Android恶意软件检测系统，基于ML或基于DL的系统被认为是一些最常见的类型。针对这些检测系统，智能攻击者开发了一系列广泛的逃避攻击，攻击者略微修改恶意软件样本以逃避其目标检测系统。在这篇调查中，我们讨论了Android操作系统中的问题空间逃避攻击，即攻击者操纵实际的APK，而不是他们提取的特征向量。我们的目标是探索这类攻击，由于缺乏Android领域的知识，或者由于专注于一般的数学逃避攻击，即特征空间逃避攻击，经常被研究界忽视。我们讨论了问题空间逃避攻击的不同方面，使用了一种新的分类方法，重点讨论了每种问题空间攻击的关键要素，如攻击者的模型、攻击者的操作模式以及攻击后应用程序的功能评估。



## **33. Certifiably Robust Policy Learning against Adversarial Communication in Multi-agent Systems**

多智能体系统中对抗敌意通信的可证明稳健策略学习 cs.LG

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10158v1)

**Authors**: Yanchao Sun, Ruijie Zheng, Parisa Hassanzadeh, Yongyuan Liang, Soheil Feizi, Sumitra Ganesh, Furong Huang

**Abstracts**: Communication is important in many multi-agent reinforcement learning (MARL) problems for agents to share information and make good decisions. However, when deploying trained communicative agents in a real-world application where noise and potential attackers exist, the safety of communication-based policies becomes a severe issue that is underexplored. Specifically, if communication messages are manipulated by malicious attackers, agents relying on untrustworthy communication may take unsafe actions that lead to catastrophic consequences. Therefore, it is crucial to ensure that agents will not be misled by corrupted communication, while still benefiting from benign communication. In this work, we consider an environment with $N$ agents, where the attacker may arbitrarily change the communication from any $C<\frac{N-1}{2}$ agents to a victim agent. For this strong threat model, we propose a certifiable defense by constructing a message-ensemble policy that aggregates multiple randomly ablated message sets. Theoretical analysis shows that this message-ensemble policy can utilize benign communication while being certifiably robust to adversarial communication, regardless of the attacking algorithm. Experiments in multiple environments verify that our defense significantly improves the robustness of trained policies against various types of attacks.

摘要: 在许多多智能体强化学习(MAIL)问题中，通信对于智能体共享信息和做出正确的决策是非常重要的。然而，当在存在噪声和潜在攻击者的真实世界应用中部署训练有素的通信代理时，基于通信的策略的安全性成为一个未被探索的严重问题。具体地说，如果通信消息被恶意攻击者操纵，依赖于不可信通信的代理可能会采取不安全的行为，导致灾难性的后果。因此，确保代理不会被损坏的通信误导，同时仍受益于良性通信是至关重要的。在这项工作中，我们考虑了一个具有$N$代理的环境，在该环境中，攻击者可以任意地将通信从任何$C<\frac{N-1}{2}$代理更改为受害者代理。对于这种强威胁模型，我们通过构造聚合多个随机消融消息集的消息集成策略来提出一种可证明的防御。理论分析表明，无论采用何种攻击算法，该消息集成策略都能充分利用良性通信，同时对敌意通信具有较强的鲁棒性。在多个环境中的实验证明，我们的防御显著提高了经过训练的策略对各种类型攻击的健壮性。



## **34. ProML: A Decentralised Platform for Provenance Management of Machine Learning Software Systems**

ProML：一种用于机器学习软件系统来源管理的分布式平台 cs.SE

Accepted as full paper in ECSA 2022 conference. To be presented

**SubmitDate**: 2022-06-21    [paper-pdf](http://arxiv.org/pdf/2206.10110v1)

**Authors**: Nguyen Khoi Tran, Bushra Sabir, M. Ali Babar, Nini Cui, Mehran Abolhasan, Justin Lipman

**Abstracts**: Large-scale Machine Learning (ML) based Software Systems are increasingly developed by distributed teams situated in different trust domains. Insider threats can launch attacks from any domain to compromise ML assets (models and datasets). Therefore, practitioners require information about how and by whom ML assets were developed to assess their quality attributes such as security, safety, and fairness. Unfortunately, it is challenging for ML teams to access and reconstruct such historical information of ML assets (ML provenance) because it is generally fragmented across distributed ML teams and threatened by the same adversaries that attack ML assets. This paper proposes ProML, a decentralised platform that leverages blockchain and smart contracts to empower distributed ML teams to jointly manage a single source of truth about circulated ML assets' provenance without relying on a third party, which is vulnerable to insider threats and presents a single point of failure. We propose a novel architectural approach called Artefact-as-a-State-Machine to leverage blockchain transactions and smart contracts for managing ML provenance information and introduce a user-driven provenance capturing mechanism to integrate existing scripts and tools to ProML without compromising participants' control over their assets and toolchains. We evaluate the performance and overheads of ProML by benchmarking a proof-of-concept system on a global blockchain. Furthermore, we assessed ProML's security against a threat model of a distributed ML workflow.

摘要: 基于大规模机器学习(ML)的软件系统越来越多地由分布在不同信任域的团队开发。内部威胁可以从任何域发起攻击，以危害ML资产(模型和数据集)。因此，实践者需要有关ML资产是如何以及由谁开发的信息，以评估其质量属性，如安全性、安全性和公平性。不幸的是，ML团队访问和重建这种ML资产的历史信息(ML起源)是具有挑战性的，因为这些信息通常分散在分散的ML团队中，并且受到攻击ML资产的相同对手的威胁。本文提出了ProML，这是一个分散的平台，利用区块链和智能合同来授权分布式ML团队联合管理关于流通的ML资产来源的单一真相来源，而不依赖于第三方，这容易受到内部威胁，并提供单点故障。我们提出了一种称为Arteact-as-State-Machine的新颖架构方法来利用区块链事务和智能合约来管理ML起源信息，并引入了用户驱动的起源捕获机制来将现有脚本和工具集成到ProML中，而不会损害参与者对其资产和工具链的控制。我们通过在全球区块链上对概念验证系统进行基准测试来评估ProML的性能和开销。此外，我们根据分布式ML工作流的威胁模型评估了ProML的安全性。



## **35. Make Some Noise: Reliable and Efficient Single-Step Adversarial Training**

制造一些噪音：可靠而高效的单步对抗性训练 cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2202.01181v2)

**Authors**: Pau de Jorge, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip H. S. Torr, Grégory Rogez, Puneet K. Dokania

**Abstracts**: Recently, Wong et al. showed that adversarial training with single-step FGSM leads to a characteristic failure mode named catastrophic overfitting (CO), in which a model becomes suddenly vulnerable to multi-step attacks. They showed that adding a random perturbation prior to FGSM (RS-FGSM) seemed to be sufficient to prevent CO. However, Andriushchenko and Flammarion observed that RS-FGSM still leads to CO for larger perturbations, and proposed an expensive regularizer (GradAlign) to avoid CO. In this work, we methodically revisit the role of noise and clipping in single-step adversarial training. Contrary to previous intuitions, we find that using a stronger noise around the clean sample combined with not clipping is highly effective in avoiding CO for large perturbation radii. Based on these observations, we then propose Noise-FGSM (N-FGSM) that, while providing the benefits of single-step adversarial training, does not suffer from CO. Empirical analyses on a large suite of experiments show that N-FGSM is able to match or surpass the performance of previous single-step methods while achieving a 3$\times$ speed-up. Code can be found in https://github.com/pdejorge/N-FGSM

摘要: 最近，Wong et al.研究表明，采用单步FGSM的对抗性训练会导致一种称为灾难性过匹配(CO)的特征故障模式，在这种模式下，模型突然变得容易受到多步攻击。他们表明，在FGSM(RS-FGSM)之前增加随机扰动似乎足以防止CO。然而，Andriushchenko和Flammarion观察到，对于较大的扰动，RS-FGSM仍然会导致CO，并提出了一种昂贵的正则化(GradAlign)来避免CO。在这项工作中，我们有条不紊地重新审视噪声和剪辑在单步对抗性训练中的作用。与以前的直觉相反，我们发现在清洁样本周围使用更强的噪声结合不削波在大扰动半径下避免CO是非常有效的。基于这些观察，我们提出了Noise-FGSM(N-FGSM)，它在提供单步对抗性训练的好处的同时，不会受到CO的影响。大量实验的实验结果表明，N-FGSM在性能上达到或超过了以往单步算法的性能，同时获得了3倍于3倍的加速。代码可在https://github.com/pdejorge/N-FGSM中找到



## **36. Diversified Adversarial Attacks based on Conjugate Gradient Method**

基于共轭梯度法的多样化对抗性攻击 cs.LG

**SubmitDate**: 2022-06-20    [paper-pdf](http://arxiv.org/pdf/2206.09628v1)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.

摘要: 深度学习模型容易受到对抗性实例的影响，而用于生成此类实例的对抗性攻击已经引起了相当大的研究兴趣。虽然现有的基于最陡下降的方法已经取得了很高的攻击成功率，但条件恶劣的问题有时会降低它们的性能。针对这一局限性，我们利用对这类问题有效的共轭梯度(CG)方法，并在CG方法的启发下提出了一种新的攻击算法，称为自动共轭梯度(ACG)攻击。在最新的稳健模型上进行的大规模评估实验结果表明，对于大多数模型，ACG能够以更少的迭代发现更多的对抗性实例，而不是现有的SOTA算法Auto-PGD(APGD)。我们研究了ACG和APGD在多样化和集约化方面的搜索性能差异，并定义了一个称为多样性指数(DI)的度量来量化多样性程度。从该指标的多样性分析可以看出，该方法搜索的多样性显著提高了其攻击成功率。



## **37. On the Limitations of Stochastic Pre-processing Defenses**

论随机前处理防御的局限性 cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09491v1)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstracts**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. Our code is available in the supplementary material.

摘要: 抵御敌意的例子仍然是一个悬而未决的问题。一种普遍的看法是，推理的随机性增加了寻找敌对输入的成本。这种防御的一个例子是在将输入提供给模型之前对它们应用随机转换。在本文中，我们从经验和理论上研究了这种随机预处理防御机制，并证明了它们是有缺陷的。首先，我们证明了大多数随机防御比之前认为的要弱；它们缺乏足够的随机性，即使是像投影梯度下降这样的标准攻击也是如此。这让人对一个长期持有的假设产生了怀疑，即随机防御使旨在逃避确定性防御的攻击无效，并迫使攻击者整合期望过转换(EOT)概念。其次，我们证明了随机防御面临着对抗稳健性和模型不变性之间的权衡；随着被防御模型对其随机化获得更多的不变性，它们变得不那么有效。未来的工作将需要将这两种影响脱钩。我们的代码可以在补充材料中找到。



## **38. A Universal Adversarial Policy for Text Classifiers**

一种适用于文本分类器的通用对抗策略 cs.LG

Accepted for publication in Neural Networks (2022), see  https://doi.org/10.1016/j.neunet.2022.06.018

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09458v1)

**Authors**: Gallil Maimon, Lior Rokach

**Abstracts**: Discovering the existence of universal adversarial perturbations had large theoretical and practical impacts on the field of adversarial learning. In the text domain, most universal studies focused on adversarial prefixes which are added to all texts. However, unlike the vision domain, adding the same perturbation to different inputs results in noticeably unnatural inputs. Therefore, we introduce a new universal adversarial setup - a universal adversarial policy, which has many advantages of other universal attacks but also results in valid texts - thus making it relevant in practice. We achieve this by learning a single search policy over a predefined set of semantics preserving text alterations, on many texts. This formulation is universal in that the policy is successful in finding adversarial examples on new texts efficiently. Our approach uses text perturbations which were extensively shown to produce natural attacks in the non-universal setup (specific synonym replacements). We suggest a strong baseline approach for this formulation which uses reinforcement learning. It's ability to generalise (from as few as 500 training texts) shows that universal adversarial patterns exist in the text domain as well.

摘要: 发现普遍存在的对抗性扰动对对抗性学习领域有很大的理论和实践影响。在语篇领域，大多数普遍的研究集中于添加到所有语篇中的对抗性前缀。然而，与视觉领域不同的是，将相同的扰动添加到不同的输入会导致明显不自然的输入。因此，我们引入了一种新的通用对抗设置-通用对抗策略，它具有其他通用攻击的许多优点，但也产生了有效的文本-从而使其在实践中具有相关性。我们通过在预定义的一组语义上学习单个搜索策略来实现这一点，该语义集保留了对许多文本的文本更改。这一提法具有普遍性，因为该政策成功地在新文本上有效地找到了对抗性的例子。我们的方法使用文本扰动，这被广泛显示为在非通用设置(特定同义词替换)中产生自然攻击。我们为这种使用强化学习的公式建议了一个强大的基线方法。它的泛化能力(从短短500个训练文本)表明，普遍的对抗性模式也存在于文本领域。



## **39. JPEG Compression-Resistant Low-Mid Adversarial Perturbation against Unauthorized Face Recognition System**

抗JPEG压缩的非授权人脸识别系统的中低端对抗性扰动 cs.CV

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09410v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: It has been observed that the unauthorized use of face recognition system raises privacy problems. Using adversarial perturbations provides one possible solution to address this issue. A critical issue to exploit adversarial perturbation against unauthorized face recognition system is that: The images uploaded to the web need to be processed by JPEG compression, which weakens the effectiveness of adversarial perturbation. Existing JPEG compression-resistant methods fails to achieve a balance among compression resistance, transferability, and attack effectiveness. To this end, we propose a more natural solution called low frequency adversarial perturbation (LFAP). Instead of restricting the adversarial perturbations, we turn to regularize the source model to employing more low-frequency features by adversarial training. Moreover, to better influence model in different frequency components, we proposed the refined low-mid frequency adversarial perturbation (LMFAP) considering the mid frequency components as the productive complement. We designed a variety of settings in this study to simulate the real-world application scenario, including cross backbones, supervisory heads, training datasets and testing datasets. Quantitative and qualitative experimental results validate the effectivenss of proposed solutions.

摘要: 据观察，未经授权使用人脸识别系统会带来隐私问题。使用对抗性扰动为解决这一问题提供了一种可能的解决方案。利用敌意扰动对抗未经授权的人脸识别系统的一个关键问题是：上传到网络的图像需要进行JPEG压缩处理，这削弱了对抗扰动的有效性。现有的JPEG抗压缩方法不能在抗压缩、可转移性和攻击有效性之间取得平衡。为此，我们提出了一种更自然的解决方案，称为低频对抗性扰动(LFAP)。我们没有限制对抗性扰动，而是通过对抗性训练来规则化信源模型以使用更多的低频特征。此外，为了更好地影响模型对不同频率成分的影响，我们提出了以中频成分作为产生性补充的精化中低频对抗扰动(LMFAP)。在本研究中，我们设计了多种设置来模拟真实世界的应用场景，包括交叉骨干、主管、训练数据集和测试数据集。定量和定性实验结果验证了所提出的解决方案的有效性。



## **40. Towards Adversarial Attack on Vision-Language Pre-training Models**

视觉语言预训练模型的对抗性攻击 cs.LG

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.09391v1)

**Authors**: Jiaming Zhang, Qi Yi, Jitao Sang

**Abstracts**: While vision-language pre-training model (VLP) has shown revolutionary improvements on various vision-language (V+L) tasks, the studies regarding its adversarial robustness remain largely unexplored. This paper studied the adversarial attack on popular VLP models and V+L tasks. First, we analyzed the performance of adversarial attacks under different settings. By examining the influence of different perturbed objects and attack targets, we concluded some key observations as guidance on both designing strong multimodal adversarial attack and constructing robust VLP models. Second, we proposed a novel multimodal attack method on the VLP models called Collaborative Multimodal Adversarial Attack (Co-Attack), which collectively carries out the attacks on the image modality and the text modality. Experimental results demonstrated that the proposed method achieves improved attack performances on different V+L downstream tasks and VLP models. The analysis observations and novel attack method hopefully provide new understanding into the adversarial robustness of VLP models, so as to contribute their safe and reliable deployment in more real-world scenarios.

摘要: 虽然视觉-语言预训练模型(VLP)在各种视觉-语言(V+L)任务上有了革命性的改进，但关于其对抗健壮性的研究仍然很少。研究了对流行的VLP模型和V+L任务的对抗性攻击。首先，我们分析了不同环境下对抗性攻击的性能。通过考察不同扰动对象和攻击目标的影响，我们总结了一些关键的观察结果，作为设计强多通道对抗性攻击和构建稳健VLP模型的指导。其次，我们提出了一种新的针对VLP模型的多模式攻击方法，称为协作式多模式对抗攻击(Co-Attack)，它共同对图像通道和文本通道进行攻击。实验结果表明，该方法在不同的V+L下游任务和VLP模型下均能获得较好的攻击性能。分析观察和新颖的攻击方法有望对VLP模型的对抗健壮性提供新的理解，从而有助于在更真实的场景中安全可靠地部署VLP模型。



## **41. Adversarially trained neural representations may already be as robust as corresponding biological neural representations**

反向训练的神经表征可能已经和相应的生物神经表征一样健壮 q-bio.NC

10 pages, 6 figures, ICML2022

**SubmitDate**: 2022-06-19    [paper-pdf](http://arxiv.org/pdf/2206.11228v1)

**Authors**: Chong Guo, Michael J. Lee, Guillaume Leclerc, Joel Dapello, Yug Rao, Aleksander Madry, James J. DiCarlo

**Abstracts**: Visual systems of primates are the gold standard of robust perception. There is thus a general belief that mimicking the neural representations that underlie those systems will yield artificial visual systems that are adversarially robust. In this work, we develop a method for performing adversarial visual attacks directly on primate brain activity. We then leverage this method to demonstrate that the above-mentioned belief might not be well founded. Specifically, we report that the biological neurons that make up visual systems of primates exhibit susceptibility to adversarial perturbations that is comparable in magnitude to existing (robustly trained) artificial neural networks.

摘要: 灵长类动物的视觉系统是强健感知的黄金标准。因此，人们普遍认为，模仿构成这些系统的神经表示将产生相反的健壮的人工视觉系统。在这项工作中，我们开发了一种直接对灵长类大脑活动进行对抗性视觉攻击的方法。然后，我们利用这种方法来证明上述信念可能没有很好的依据。具体地说，我们报告了组成灵长类视觉系统的生物神经元对对抗性扰动的敏感性，其大小与现有的(稳健训练的)人工神经网络相当。



## **42. Efficient and Transferable Adversarial Examples from Bayesian Neural Networks**

贝叶斯神经网络中高效且可移植的对抗性实例 cs.LG

Accepted at UAI 2022

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2011.05074v4)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: An established way to improve the transferability of black-box evasion attacks is to craft the adversarial examples on an ensemble-based surrogate to increase diversity. We argue that transferability is fundamentally related to uncertainty. Based on a state-of-the-art Bayesian Deep Learning technique, we propose a new method to efficiently build a surrogate by sampling approximately from the posterior distribution of neural network weights, which represents the belief about the value of each parameter. Our extensive experiments on ImageNet, CIFAR-10 and MNIST show that our approach improves the success rates of four state-of-the-art attacks significantly (up to 83.2 percentage points), in both intra-architecture and inter-architecture transferability. On ImageNet, our approach can reach 94% of success rate while reducing training computations from 11.6 to 2.4 exaflops, compared to an ensemble of independently trained DNNs. Our vanilla surrogate achieves 87.5% of the time higher transferability than three test-time techniques designed for this purpose. Our work demonstrates that the way to train a surrogate has been overlooked, although it is an important element of transfer-based attacks. We are, therefore, the first to review the effectiveness of several training methods in increasing transferability. We provide new directions to better understand the transferability phenomenon and offer a simple but strong baseline for future work.

摘要: 提高黑盒逃避攻击可转移性的一种既定方法是在基于集成的代理上精心制作对抗性示例，以增加多样性。我们认为，可转让性从根本上与不确定性有关。基于最新的贝叶斯深度学习技术，我们提出了一种新的方法，通过对神经网络权值的后验分布进行近似采样来有效地构建代理，该后验分布代表了对每个参数的值的信念。我们在ImageNet、CIFAR-10和MNIST上的广泛实验表明，我们的方法显著提高了四种最先进攻击的成功率(高达83.2个百分点)，在架构内和架构间的可转移性方面都是如此。在ImageNet上，与独立训练的DNN集成相比，我们的方法可以达到94%的成功率，同时将训练计算量从11.6exaflop减少到2.4exaflop。我们的香草代理在87.5%的时间内实现了比为此目的而设计的三种测试时间技术更高的可转移性。我们的工作表明，训练代理的方法被忽视了，尽管它是基于传输的攻击的一个重要元素。因此，我们第一次审查了几种培训方法在提高可转移性方面的有效性。我们为更好地理解可转移性现象提供了新的方向，并为未来的工作提供了一个简单但强有力的基线。



## **43. DECK: Model Hardening for Defending Pervasive Backdoors**

甲板：用于防御无处不在的后门的模型强化 cs.CR

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09272v1)

**Authors**: Guanhong Tao, Yingqi Liu, Siyuan Cheng, Shengwei An, Zhuo Zhang, Qiuling Xu, Guangyu Shen, Xiangyu Zhang

**Abstracts**: Pervasive backdoors are triggered by dynamic and pervasive input perturbations. They can be intentionally injected by attackers or naturally exist in normally trained models. They have a different nature from the traditional static and localized backdoors that can be triggered by perturbing a small input area with some fixed pattern, e.g., a patch with solid color. Existing defense techniques are highly effective for traditional backdoors. However, they may not work well for pervasive backdoors, especially regarding backdoor removal and model hardening. In this paper, we propose a novel model hardening technique against pervasive backdoors, including both natural and injected backdoors. We develop a general pervasive attack based on an encoder-decoder architecture enhanced with a special transformation layer. The attack can model a wide range of existing pervasive backdoor attacks and quantify them by class distances. As such, using the samples derived from our attack in adversarial training can harden a model against these backdoor vulnerabilities. Our evaluation on 9 datasets with 15 model structures shows that our technique can enlarge class distances by 59.65% on average with less than 1% accuracy degradation and no robustness loss, outperforming five hardening techniques such as adversarial training, universal adversarial training, MOTH, etc. It can reduce the attack success rate of six pervasive backdoor attacks from 99.06% to 1.94%, surpassing seven state-of-the-art backdoor removal techniques.

摘要: 无处不在的后门由动态和无处不在的输入扰动触发。它们可能是攻击者故意注入的，也可能是正常训练的模型中自然存在的。它们与传统的静态和本地化后门不同，后者可以通过以某种固定图案干扰小输入区域来触发，例如，具有纯色的补丁。现有的防御技术对传统的后门非常有效。然而，它们可能不适用于普遍存在的后门，特别是在后门删除和模型强化方面。在这篇文章中，我们提出了一种新的模型硬化技术，针对普遍存在的后门，包括自然后门和注入后门。我们开发了一种通用的普适攻击，该攻击基于一种编解码器体系结构，并通过特殊的转换层进行了增强。该攻击可以模拟一系列现有的普遍存在的后门攻击，并根据类别距离对它们进行量化。因此，在对抗性训练中使用从我们的攻击中获得的样本可以加强针对这些后门漏洞的模型。在15个模型结构的9个数据集上的测试结果表明，该技术在准确率和健壮性没有损失的情况下，类距离平均扩展59.65%，优于对抗性训练、万能对抗性训练、MOST等5种强化技术，将6种普遍存在的后门攻击的攻击成功率从99.06%降低到1.94%，超过了7种最先进的后门移除技术。



## **44. On the Role of Generalization in Transferability of Adversarial Examples**

论概括在对抗性例句可转移性中的作用 cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09238v1)

**Authors**: Yilin Wang, Farzan Farnia

**Abstracts**: Black-box adversarial attacks designing adversarial examples for unseen neural networks (NNs) have received great attention over the past years. While several successful black-box attack schemes have been proposed in the literature, the underlying factors driving the transferability of black-box adversarial examples still lack a thorough understanding. In this paper, we aim to demonstrate the role of the generalization properties of the substitute classifier used for generating adversarial examples in the transferability of the attack scheme to unobserved NN classifiers. To do this, we apply the max-min adversarial example game framework and show the importance of the generalization properties of the substitute NN in the success of the black-box attack scheme in application to different NN classifiers. We prove theoretical generalization bounds on the difference between the attack transferability rates on training and test samples. Our bounds suggest that a substitute NN with better generalization behavior could result in more transferable adversarial examples. In addition, we show that standard operator norm-based regularization methods could improve the transferability of the designed adversarial examples. We support our theoretical results by performing several numerical experiments showing the role of the substitute network's generalization in generating transferable adversarial examples. Our empirical results indicate the power of Lipschitz regularization methods in improving the transferability of adversarial examples.

摘要: 针对看不见的神经网络设计对抗性实例的黑盒对抗性攻击在过去的几年里受到了极大的关注。虽然文献中已经提出了几个成功的黑盒攻击方案，但驱动黑盒对抗性例子可转换性的潜在因素仍然缺乏深入的了解。在本文中，我们的目的是证明用于生成对抗性示例的替换分类器的泛化性质在攻击方案向不可观测的NN分类器的可转移性中所起的作用。为此，我们应用了最大-最小对抗性例子博弈框架，并展示了替代神经网络的泛化性质在黑盒攻击方案应用于不同的神经网络分类器中的重要性。我们证明了训练样本和测试样本上的攻击可转移率之间的差异的理论概括界。我们的界限表明，具有更好泛化行为的替代NN可以产生更多可转移的对抗性实例。此外，我们还证明了基于标准算子范数的正则化方法可以提高所设计的对抗性实例的可转移性。我们通过几个数值实验来支持我们的理论结果，这些实验显示了替代网络的泛化在生成可转移的对抗性例子中的作用。我们的实证结果表明，Lipschitz正则化方法在提高对抗性例子的可转移性方面是有效的。



## **45. Measuring Lower Bounds of Local Differential Privacy via Adversary Instantiations in Federated Learning**

联合学习中通过敌意实例化测量局部差分隐私的下界 cs.CR

15 pages, 7 figures

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09122v1)

**Authors**: Marin Matsumoto, Tsubasa Takahashi, Seng Pei Liew, Masato Oguchi

**Abstracts**: Local differential privacy (LDP) gives a strong privacy guarantee to be used in a distributed setting like federated learning (FL). LDP mechanisms in FL protect a client's gradient by randomizing it on the client; however, how can we interpret the privacy level given by the randomization? Moreover, what types of attacks can we mitigate in practice? To answer these questions, we introduce an empirical privacy test by measuring the lower bounds of LDP. The privacy test estimates how an adversary predicts if a reported randomized gradient was crafted from a raw gradient $g_1$ or $g_2$. We then instantiate six adversaries in FL under LDP to measure empirical LDP at various attack surfaces, including a worst-case attack that reaches the theoretical upper bound of LDP. The empirical privacy test with the adversary instantiations enables us to interpret LDP more intuitively and discuss relaxation of the privacy parameter until a particular instantiated attack surfaces. We also demonstrate numerical observations of the measured privacy in these adversarial settings, and the worst-case attack is not realistic in FL. In the end, we also discuss the possible relaxation of privacy levels in FL under LDP.

摘要: 局部差异隐私(LDP)为联邦学习(FL)等分布式环境下的应用提供了强有力的隐私保障。FL中的LDP机制通过将客户端上的梯度随机化来保护客户端的梯度；然而，我们如何解释随机化所提供的隐私级别？此外，我们在实践中可以减轻哪些类型的攻击？为了回答这些问题，我们引入了一项经验隐私测试，通过测量自民党的下限。隐私测试估计对手如何预测所报告的随机梯度是从原始梯度$g_1$还是$g_2$创建的。然后，我们在LDP下实例化了FL中的六个对手，以测量不同攻击面上的经验LDP，包括达到LDP理论上限的最坏情况攻击。使用对手实例的经验隐私测试使我们能够更直观地解释LDP，并讨论隐私参数的放松，直到特定的实例化攻击浮出水面。我们还展示了在这些对抗性设置下测量的隐私的数值观测，并且最坏情况下的攻击在FL中是不现实的。最后，我们还讨论了LDP下FL隐私级别放宽的可能性。



## **46. Existence and Minimax Theorems for Adversarial Surrogate Risks in Binary Classification**

二元分类中对抗性代理风险的存在性和极大极小定理 cs.LG

37 pages, 1 Figure

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09098v1)

**Authors**: Natalie S. Frank

**Abstracts**: Adversarial training is one of the most popular methods for training methods robust to adversarial attacks, however, it is not well-understood from a theoretical perspective. We prove and existence, regularity, and minimax theorems for adversarial surrogate risks. Our results explain some empirical observations on adversarial robustness from prior work and suggest new directions in algorithm development. Furthermore, our results extend previously known existence and minimax theorems for the adversarial classification risk to surrogate risks.

摘要: 对抗性训练是对抗攻击能力最强的训练方法之一，但从理论上对它的理解还不够深入。我们证明了对抗性代理风险的存在性、正则性和极大极小定理。我们的结果解释了以前工作中关于对手稳健性的一些经验观察，并为算法开发提供了新的方向。此外，我们的结果推广了已知的对抗性分类风险到代理风险的存在性和极大极小定理。



## **47. Comment on Transferability and Input Transformation with Additive Noise**

加性噪声条件下的可转移性与输入变换 cs.LG

**SubmitDate**: 2022-06-18    [paper-pdf](http://arxiv.org/pdf/2206.09075v1)

**Authors**: Hoki Kim, Jinseong Park, Jaewook Lee

**Abstracts**: Adversarial attacks have verified the existence of the vulnerability of neural networks. By adding small perturbations to a benign example, adversarial attacks successfully generate adversarial examples that lead misclassification of deep learning models. More importantly, an adversarial example generated from a specific model can also deceive other models without modification. We call this phenomenon ``transferability". Here, we analyze the relationship between transferability and input transformation with additive noise by mathematically proving that the modified optimization can produce more transferable adversarial examples.

摘要: 对抗性攻击验证了神经网络脆弱性的存在。通过在良性示例中添加小的扰动，对抗性攻击成功地生成了导致深度学习模型错误分类的对抗性示例。更重要的是，从特定模型生成的对抗性示例也可以在不修改的情况下欺骗其他模型。我们称这种现象为可转移性。在这里，我们通过数学证明改进的最优化方法可以产生更多可转移性的对抗性例子，分析了可转移性与加性噪声的输入变换之间的关系。



## **48. Learning Generative Deception Strategies in Combinatorial Masking Games**

在组合掩饰博弈中学习生成性欺骗策略 cs.GT

GameSec 2021

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2109.11637v2)

**Authors**: Junlin Wu, Charles Kamhoua, Murat Kantarcioglu, Yevgeniy Vorobeychik

**Abstracts**: Deception is a crucial tool in the cyberdefence repertoire, enabling defenders to leverage their informational advantage to reduce the likelihood of successful attacks. One way deception can be employed is through obscuring, or masking, some of the information about how systems are configured, increasing attacker's uncertainty about their targets. We present a novel game-theoretic model of the resulting defender-attacker interaction, where the defender chooses a subset of attributes to mask, while the attacker responds by choosing an exploit to execute. The strategies of both players have combinatorial structure with complex informational dependencies, and therefore even representing these strategies is not trivial. First, we show that the problem of computing an equilibrium of the resulting zero-sum defender-attacker game can be represented as a linear program with a combinatorial number of system configuration variables and constraints, and develop a constraint generation approach for solving this problem. Next, we present a novel highly scalable approach for approximately solving such games by representing the strategies of both players as neural networks. The key idea is to represent the defender's mixed strategy using a deep neural network generator, and then using alternating gradient-descent-ascent algorithm, analogous to the training of Generative Adversarial Networks. Our experiments, as well as a case study, demonstrate the efficacy of the proposed approach.

摘要: 欺骗是网络防御体系中的一个重要工具，使防御者能够利用他们的信息优势来降低攻击成功的可能性。欺骗的一种方式是通过模糊或掩盖有关系统配置的一些信息，增加攻击者对目标的不确定性。我们提出了一个新的博弈论模型来描述由此产生的防御者和攻击者的相互作用，其中防御者选择掩蔽属性的子集，而攻击者通过选择一个利用漏洞来执行。两个参与者的策略都具有组合结构，具有复杂的信息依赖关系，因此即使是表示这些策略也不是微不足道的。首先，我们证明了由此产生的零和防御者-攻击者博弈的均衡计算问题可以表示为具有多个系统配置变量和约束的组合数量的线性规划，并给出了一种求解该问题的约束生成方法。接下来，我们提出了一种新的高度可扩展的方法，通过将双方的策略表示为神经网络来近似求解这类对策。其核心思想是用一个深度神经网络生成器来表示防御者的混合策略，然后使用交替的梯度-下降-上升算法，类似于生成式对手网络的训练。我们的实验和一个案例研究证明了该方法的有效性。



## **49. RetrievalGuard: Provably Robust 1-Nearest Neighbor Image Retrieval**

RetrievalGuard：可证明健壮的1-近邻图像检索 cs.IR

accepted by ICML 2022

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.11225v1)

**Authors**: Yihan Wu, Hongyang Zhang, Heng Huang

**Abstracts**: Recent research works have shown that image retrieval models are vulnerable to adversarial attacks, where slightly modified test inputs could lead to problematic retrieval results. In this paper, we aim to design a provably robust image retrieval model which keeps the most important evaluation metric Recall@1 invariant to adversarial perturbation. We propose the first 1-nearest neighbor (NN) image retrieval algorithm, RetrievalGuard, which is provably robust against adversarial perturbations within an $\ell_2$ ball of calculable radius. The challenge is to design a provably robust algorithm that takes into consideration the 1-NN search and the high-dimensional nature of the embedding space. Algorithmically, given a base retrieval model and a query sample, we build a smoothed retrieval model by carefully analyzing the 1-NN search procedure in the high-dimensional embedding space. We show that the smoothed retrieval model has bounded Lipschitz constant and thus the retrieval score is invariant to $\ell_2$ adversarial perturbations. Experiments on image retrieval tasks validate the robustness of our RetrievalGuard method.

摘要: 最近的研究工作表明，图像检索模型容易受到对抗性攻击，其中稍加修改的测试输入可能会导致有问题的检索结果。在本文中，我们的目标是设计一个可证明稳健的图像检索模型，使最重要的评价指标recall@1对对手的扰动保持不变。我们提出了第一个1-近邻(NN)图像检索算法RetrivalGuard，该算法在半径可计算的$ell2$球内对敌意扰动具有证明的健壮性。挑战在于设计一种可证明稳健的算法，该算法考虑到1-NN搜索和嵌入空间的高维性质。在算法上，给出了一个基本的检索模型和一个查询样本，通过仔细分析高维嵌入空间中的1-NN搜索过程，建立了一个平滑的检索模型。我们证明了平滑的检索模型具有有界的Lipschitz常数，因此检索分数对于$\ell_2$对抗性扰动是不变的。在图像检索任务上的实验验证了该方法的健壮性。



## **50. Is Multi-Modal Necessarily Better? Robustness Evaluation of Multi-modal Fake News Detection**

多式联运一定会更好吗？多模式假新闻检测的稳健性评价 cs.AI

**SubmitDate**: 2022-06-17    [paper-pdf](http://arxiv.org/pdf/2206.08788v1)

**Authors**: Jinyin Chen, Chengyu Jia, Haibin Zheng, Ruoxi Chen, Chenbo Fu

**Abstracts**: The proliferation of fake news and its serious negative social influence push fake news detection methods to become necessary tools for web managers. Meanwhile, the multi-media nature of social media makes multi-modal fake news detection popular for its ability to capture more modal features than uni-modal detection methods. However, current literature on multi-modal detection is more likely to pursue the detection accuracy but ignore the robustness of the detector. To address this problem, we propose a comprehensive robustness evaluation of multi-modal fake news detectors. In this work, we simulate the attack methods of malicious users and developers, i.e., posting fake news and injecting backdoors. Specifically, we evaluate multi-modal detectors with five adversarial and two backdoor attack methods. Experiment results imply that: (1) The detection performance of the state-of-the-art detectors degrades significantly under adversarial attacks, even worse than general detectors; (2) Most multi-modal detectors are more vulnerable when subjected to attacks on visual modality than textual modality; (3) Popular events' images will cause significant degradation to the detectors when they are subjected to backdoor attacks; (4) The performance of these detectors under multi-modal attacks is worse than under uni-modal attacks; (5) Defensive methods will improve the robustness of the multi-modal detectors.

摘要: 假新闻的泛滥及其严重的负面社会影响，促使假新闻检测方法成为网络管理者的必备工具。同时，社交媒体的多媒体特性使得多模式假新闻检测因其能够捕捉到比单模式检测方法更多的模式特征而广受欢迎。然而，目前关于多模式检测的文献更倾向于追求检测精度，而忽略了检测器的稳健性。针对这一问题，我们提出了一种多模式假新闻检测器的综合稳健性评估方法。在这项工作中，我们模拟了恶意用户和开发者的攻击方法，即发布假新闻和注入后门。具体地说，我们使用五种对抗性攻击方法和两种后门攻击方法来评估多模式检测器。实验结果表明：(1)现有检测器在对抗性攻击下检测性能显著下降，甚至比一般检测器更差；(2)大多数多模式检测器在受到视觉通道攻击时比文本通道更容易受到攻击；(3)热门事件的图像在受到后门攻击时会导致检测器性能显著下降；(4)多模式检测器在多模式攻击下的性能比单模式攻击下的性能差；(5)防御方法将提高多模式检测器的健壮性。



