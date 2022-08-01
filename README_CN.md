# Latest Adversarial Attack Papers
**update at 2022-08-02 06:31:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can We Mitigate Backdoor Attack Using Adversarial Detection Methods?**

我们可以使用对抗性检测方法来减少后门攻击吗？ cs.LG

Accepted by IEEE TDSC

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2006.14871v2)

**Authors**: Kaidi Jin, Tianwei Zhang, Chao Shen, Yufei Chen, Ming Fan, Chenhao Lin, Ting Liu

**Abstracts**: Deep Neural Networks are well known to be vulnerable to adversarial attacks and backdoor attacks, where minor modifications on the input are able to mislead the models to give wrong results. Although defenses against adversarial attacks have been widely studied, investigation on mitigating backdoor attacks is still at an early stage. It is unknown whether there are any connections and common characteristics between the defenses against these two attacks. We conduct comprehensive studies on the connections between adversarial examples and backdoor examples of Deep Neural Networks to seek to answer the question: can we detect backdoor using adversarial detection methods. Our insights are based on the observation that both adversarial examples and backdoor examples have anomalies during the inference process, highly distinguishable from benign samples. As a result, we revise four existing adversarial defense methods for detecting backdoor examples. Extensive evaluations indicate that these approaches provide reliable protection against backdoor attacks, with a higher accuracy than detecting adversarial examples. These solutions also reveal the relations of adversarial examples, backdoor examples and normal samples in model sensitivity, activation space and feature space. This is able to enhance our understanding about the inherent features of these two attacks and the defense opportunities.

摘要: 众所周知，深度神经网络容易受到对抗性攻击和后门攻击，在这些攻击中，对输入的微小修改能够误导模型给出错误的结果。尽管针对敌意攻击的防御已经被广泛研究，但关于减轻后门攻击的调查仍处于早期阶段。目前尚不清楚针对这两种攻击的防御之间是否有任何联系和共同特征。我们对深度神经网络的对抗性实例和后门实例之间的联系进行了全面的研究，试图回答这样一个问题：我们是否可以使用对抗性检测方法来检测后门。我们的洞察是基于这样的观察，即对抗性例子和后门例子在推理过程中都有异常，与良性样本具有高度的区分性。因此，我们对现有的四种检测后门实例的对抗性防御方法进行了修改。广泛的评估表明，这些方法提供了可靠的后门攻击保护，比检测敌意示例具有更高的准确性。这些解还揭示了对抗性样本、后门样本和正常样本在模型敏感度、激活空间和特征空间中的关系。这能够增进我们对这两起袭击的内在特征和防御机会的了解。



## **2. Look Closer to Your Enemy: Learning to Attack via Teacher-student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

13 pages, 8 figures, NDSS

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13381v2)

**Authors**: Mingejie Wang, Zhiqing Tang, Sirui Li, Dingwen Xiao

**Abstracts**: This paper aims to generate realistic attack samples of person re-identification, ReID, by reading the enemy's mind (VM). In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE, to generate adversarial query images. Concretely, LCYE first distills VM's knowledge via teacher-student memory mimicking in the proxy task. Then this knowledge prior acts as an explicit cipher conveying what is essential and realistic, believed by VM, for accurate adversarial misleading. Besides, benefiting from the multiple opposing task framework of LCYE, we further investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. Our code is now available at https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.

摘要: 本文旨在通过读取敌人的心理(Vm)来生成真实的人重新识别的攻击样本，Reid。本文提出了一种新的隐蔽可控的Reid攻击基线--LCYE，用于生成敌意查询图像。具体来说，LCYE首先通过模仿代理任务中的师生记忆来提取VM的知识。然后，这种先验知识就像一个明确的密码，传达了被VM认为是必要和现实的东西，以实现准确的对抗性误导。此外，得益于LCYE的多重对立任务框架，我们从对抗性攻击的角度进一步考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。我们的代码现已在https://gitfront.io/r/user-3704489/mKXusqDT4ffr/LCYE/.上提供



## **3. Privacy-Preserving Federated Recurrent Neural Networks**

隐私保护的联邦递归神经网络 cs.CR

**SubmitDate**: 2022-07-28    [paper-pdf](http://arxiv.org/pdf/2207.13947v1)

**Authors**: Sinem Sav, Abdulrahman Diaa, Apostolos Pyrgelis, Jean-Philippe Bossuat, Jean-Pierre Hubaux

**Abstracts**: We present RHODE, a novel system that enables privacy-preserving training of and prediction on Recurrent Neural Networks (RNNs) in a federated learning setting by relying on multiparty homomorphic encryption (MHE). RHODE preserves the confidentiality of the training data, the model, and the prediction data; and it mitigates the federated learning attacks that target the gradients under a passive-adversary threat model. We propose a novel packing scheme, multi-dimensional packing, for a better utilization of Single Instruction, Multiple Data (SIMD) operations under encryption. With multi-dimensional packing, RHODE enables the efficient processing, in parallel, of a batch of samples. To avoid the exploding gradients problem, we also provide several clip-by-value approximations for enabling gradient clipping under encryption. We experimentally show that the model performance with RHODE remains similar to non-secure solutions both for homogeneous and heterogeneous data distribution among the data holders. Our experimental evaluation shows that RHODE scales linearly with the number of data holders and the number of timesteps, sub-linearly and sub-quadratically with the number of features and the number of hidden units of RNNs, respectively. To the best of our knowledge, RHODE is the first system that provides the building blocks for the training of RNNs and its variants, under encryption in a federated learning setting.

摘要: 我们提出了一种新的系统Rhode，它依靠多方同态加密(MHE)在联邦学习环境中实现对递归神经网络(RNN)的隐私保护训练和预测。Rhode保留了训练数据、模型和预测数据的机密性；它缓解了被动对手威胁模型下针对梯度的联合学习攻击。为了更好地利用加密环境下的单指令、多数据(SIMD)运算，提出了一种新的打包方案--多维打包。通过多维包装，Rhode能够并行高效地处理一批样品。为了避免爆炸的梯度问题，我们还提供了几种逐值近似的方法来实现加密下的梯度裁剪。我们的实验表明，对于数据持有者之间的同质和异质数据分布，Rhode模型的性能与非安全解决方案相似。我们的实验评估表明，Rhode与数据持有者数量和时间步数成线性关系，分别与RNN的特征数和隐含单元数成亚线性和次二次关系。据我们所知，Rhode是第一个在联合学习环境中加密的、为RNN及其变体的训练提供构建块的系统。



## **4. Label-Only Membership Inference Attack against Node-Level Graph Neural Networks**

针对节点级图神经网络的仅标签隶属度推理攻击 cs.CR

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13766v1)

**Authors**: Mauro Conti, Jiaxin Li, Stjepan Picek, Jing Xu

**Abstracts**: Graph Neural Networks (GNNs), inspired by Convolutional Neural Networks (CNNs), aggregate the message of nodes' neighbors and structure information to acquire expressive representations of nodes for node classification, graph classification, and link prediction. Previous studies have indicated that GNNs are vulnerable to Membership Inference Attacks (MIAs), which infer whether a node is in the training data of GNNs and leak the node's private information, like the patient's disease history. The implementation of previous MIAs takes advantage of the models' probability output, which is infeasible if GNNs only provide the prediction label (label-only) for the input.   In this paper, we propose a label-only MIA against GNNs for node classification with the help of GNNs' flexible prediction mechanism, e.g., obtaining the prediction label of one node even when neighbors' information is unavailable. Our attacking method achieves around 60\% accuracy, precision, and Area Under the Curve (AUC) for most datasets and GNN models, some of which are competitive or even better than state-of-the-art probability-based MIAs implemented under our environment and settings. Additionally, we analyze the influence of the sampling method, model selection approach, and overfitting level on the attack performance of our label-only MIA. Both of those factors have an impact on the attack performance. Then, we consider scenarios where assumptions about the adversary's additional dataset (shadow dataset) and extra information about the target model are relaxed. Even in those scenarios, our label-only MIA achieves a better attack performance in most cases. Finally, we explore the effectiveness of possible defenses, including Dropout, Regularization, Normalization, and Jumping knowledge. None of those four defenses prevent our attack completely.

摘要: 图神经网络(GNN)受卷积神经网络(CNN)的启发，将节点的邻居信息和结构信息聚合在一起，得到节点的表达形式，用于节点分类、图分类和链接预测。以往的研究表明，GNN容易受到成员关系推断攻击(MIA)，MIA可以推断节点是否在GNN的训练数据中，并泄露节点的私有信息，如患者的病史。以前的MIA的实现利用了模型的概率输出，如果GNN只为输入提供预测标签(仅标签)，这是不可行的。本文利用GNN灵活的预测机制，提出了一种针对GNN的只有标签的MIA用于节点分类，例如，即使在邻居信息不可用的情况下也能获得一个节点的预测标签。对于大多数数据集和GNN模型，我们的攻击方法达到了大约60%的准确率、精确度和曲线下面积(AUC)，其中一些可以与在我们的环境和设置下实现的最先进的基于概率的MIA相媲美，甚至更好。此外，我们还分析了采样方法、模型选择方法和过拟合程度对仅标签MIA攻击性能的影响。这两个因素都会对攻击性能产生影响。然后，我们考虑放松对对手的额外数据集(阴影数据集)和关于目标模型的额外信息的假设。即使在这些情况下，我们的仅标签MIA在大多数情况下也可以实现更好的攻击性能。最后，我们探讨了可能的防御措施的有效性，包括丢弃、正则化、正规化和跳跃知识。这四种防御手段都不能完全阻止我们的进攻。



## **5. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

SAC-AP：基于软参与者批评者的深度强化学习告警优先级 cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13666v1)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.

摘要: 入侵检测系统(入侵检测系统)产生大量的错误警报，使得对真实阳性的检测变得困难。因此，警报优先级在决定从由入侵检测系统生成的大量警报中调查哪些警报时起着至关重要的作用。近年来，与其他方法相比，基于深度强化学习(DRL)的深度确定性策略梯度(DDPG)非策略方法能够获得更好的告警优先级排序结果。然而，DDPG容易出现过度匹配的问题。此外，它的探测能力也很差，因此不适合于具有随机环境的问题。针对这些局限性，我们提出了一种基于软参与者-批评者的DRL警报优先排序算法(SAC-AP)，这是一种基于最大熵强化学习框架的非策略方法，旨在最大化期望回报的同时最大化熵。在此基础上，将对手和防御者之间的相互作用建模为零和博弈，并利用双预言框架得到近似的混合策略纳什均衡。SAC-AP发现稳健的警戒调查策略，并针对对手的混合策略计算纯策略的最佳响应。我们介绍了SAC-AP的总体设计，并与其他最先进的警报优先排序方法进行了比较，评估了其性能。我们将防御者的损失，即防御者无法调查由于攻击而触发的警报作为性能指标。结果表明，与基于DDPG的告警优先级排序方法相比，SAC-AP可以减少高达30%的防御者损失，从而提供更好的防御入侵保护。此外，当SAC-AP与其他传统的警报优先排序方法(包括Uniform、Gain、Rio和Suricata)相比时，好处甚至更高。



## **6. Membership Inference Attacks via Adversarial Examples**

基于对抗性例子的成员关系推理攻击 cs.LG

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13572v1)

**Authors**: Hamid Jalalzai, Elie Kadoche, Rémi Leluc, Vincent Plassier

**Abstracts**: The raise of machine learning and deep learning led to significant improvement in several domains. This change is supported by both the dramatic rise in computation power and the collection of large datasets. Such massive datasets often include personal data which can represent a threat to privacy. Membership inference attacks are a novel direction of research which aims at recovering training data used by a learning algorithm. In this paper, we develop a mean to measure the leakage of training data leveraging a quantity appearing as a proxy of the total variation of a trained model near its training samples. We extend our work by providing a novel defense mechanism. Our contributions are supported by empirical evidence through convincing numerical experiments.

摘要: 机器学习和深度学习的兴起导致了几个领域的显著改善。计算能力的戏剧性增长和大型数据集的收集都支持这种变化。如此庞大的数据集通常包括可能对隐私构成威胁的个人数据。隶属度推理攻击是一个新的研究方向，其目的是恢复学习算法所使用的训练数据。在本文中，我们开发了一种方法来衡量训练数据的泄漏，该方法利用一个量来衡量训练模型在其训练样本附近的总变异。我们通过提供一种新颖的防御机制来扩展我们的工作。通过令人信服的数值实验，我们的贡献得到了经验证据的支持。



## **7. Robust Textual Embedding against Word-level Adversarial Attacks**

抵抗词级敌意攻击的稳健文本嵌入 cs.CL

Accepted by UAI 2022, code is available at  https://github.com/JHL-HUST/FTML

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2202.13817v2)

**Authors**: Yichen Yang, Xiaosen Wang, Kun He

**Abstracts**: We attribute the vulnerability of natural language processing models to the fact that similar inputs are converted to dissimilar representations in the embedding space, leading to inconsistent outputs, and we propose a novel robust training method, termed Fast Triplet Metric Learning (FTML). Specifically, we argue that the original sample should have similar representation with its adversarial counterparts and distinguish its representation from other samples for better robustness. To this end, we adopt the triplet metric learning into the standard training to pull words closer to their positive samples (i.e., synonyms) and push away their negative samples (i.e., non-synonyms) in the embedding space. Extensive experiments demonstrate that FTML can significantly promote the model robustness against various advanced adversarial attacks while keeping competitive classification accuracy on original samples. Besides, our method is efficient as it only needs to adjust the embedding and introduces very little overhead on the standard training. Our work shows great potential of improving the textual robustness through robust word embedding.

摘要: 我们将自然语言处理模型的脆弱性归因于相似的输入在嵌入空间中被转换为不相似的表示，从而导致输出不一致，并提出了一种新的稳健训练方法，称为快速三重度量学习(Fast Triplet Metric Learning，FTML)。具体地说，我们认为原始样本应该具有与对手样本相似的表示，并将其表示与其他样本区分开来，以获得更好的稳健性。为此，我们将三元组度量学习引入标准训练中，将单词拉近其正样本(即同义词)，并在嵌入空间中推开其负样本(即非同义词)。大量实验表明，FTML能够显著提高模型对各种高级对抗性攻击的稳健性，同时保持对原始样本的竞争性分类精度。此外，我们的方法是有效的，因为它只需要调整嵌入，并且对标准训练的开销很小。我们的工作显示了通过稳健的词嵌入来提高文本稳健性的巨大潜力。



## **8. Improved and Interpretable Defense to Transferred Adversarial Examples by Jacobian Norm with Selective Input Gradient Regularization**

基于选择输入梯度正则化的雅可比范数对转移对抗性实例的改进和可解释防御 cs.LG

Under review

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13036v2)

**Authors**: Deyin Liu, Lin Wu, Farid Boussaid, Mohammed Bennamoun

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial examples that are crafted with imperceptible perturbations, i.e., a small change in an input image can induce a mis-classification, and thus threatens the reliability of deep learning based deployment systems. Adversarial training (AT) is often adopted to improve the robustness of DNNs through training a mixture of corrupted and clean data. However, most of AT based methods are ineffective in dealing with \textit{transferred adversarial examples} which are generated to fool a wide spectrum of defense models, and thus cannot satisfy the generalization requirement raised in real-world scenarios. Moreover, adversarially training a defense model in general cannot produce interpretable predictions towards the inputs with perturbations, whilst a highly interpretable robust model is required by different domain experts to understand the behaviour of a DNN. In this work, we propose an approach based on Jacobian norm and Selective Input Gradient Regularization (J-SIGR), which suggests the linearized robustness through Jacobian normalization and also regularizes the perturbation-based saliency maps to imitate the model's interpretable predictions. As such, we achieve both the improved defense and high interpretability of DNNs. Finally, we evaluate our method across different architectures against powerful adversarial attacks. Experiments demonstrate that the proposed J-SIGR confers improved robustness against transferred adversarial attacks, and we also show that the predictions from the neural network are easy to interpret.

摘要: 众所周知，深度神经网络(DNN)容易受到带有不可察觉扰动的敌意示例的影响，即输入图像的微小变化就会导致误分类，从而威胁到基于深度学习的部署系统的可靠性。为了提高DNN的鲁棒性，经常采用对抗性训练(AT)，方法是训练一组受破坏的和一组干净的数据。然而，大多数基于AT的方法都不能有效地处理为愚弄各种防御模型而生成的文本传递的对抗性实例，从而不能满足现实场景中提出的泛化要求。此外，对抗性地训练防御模型一般不能产生对带有扰动的输入的可解释预测，而不同领域的专家需要高度可解释的稳健模型来理解DNN的行为。在这项工作中，我们提出了一种基于雅可比范数和选择性输入梯度正则化(J-SIGR)的方法，该方法通过雅可比归一化来证明线性化的稳健性，并对基于扰动的显著图进行正则化来模拟模型的可解释预测。因此，我们实现了DNN的改进的防御性和高度的可解释性。最后，我们在不同的体系结构上对我们的方法进行了评估，以对抗强大的对手攻击。实验表明，所提出的J-SIGR算法对转移攻击具有较好的稳健性，并且神经网络的预测结果易于解释。



## **9. Point Cloud Attacks in Graph Spectral Domain: When 3D Geometry Meets Graph Signal Processing**

图谱域中的点云攻击：当3D几何遇到图信号处理时 cs.CV

arXiv admin note: substantial text overlap with arXiv:2202.07261

**SubmitDate**: 2022-07-27    [paper-pdf](http://arxiv.org/pdf/2207.13326v1)

**Authors**: Daizong Liu, Wei Hu, Xin Li

**Abstracts**: With the increasing attention in various 3D safety-critical applications, point cloud learning models have been shown to be vulnerable to adversarial attacks. Although existing 3D attack methods achieve high success rates, they delve into the data space with point-wise perturbation, which may neglect the geometric characteristics. Instead, we propose point cloud attacks from a new perspective -- the graph spectral domain attack, aiming to perturb graph transform coefficients in the spectral domain that corresponds to varying certain geometric structure. Specifically, leveraging on graph signal processing, we first adaptively transform the coordinates of points onto the spectral domain via graph Fourier transform (GFT) for compact representation. Then, we analyze the influence of different spectral bands on the geometric structure, based on which we propose to perturb the GFT coefficients via a learnable graph spectral filter. Considering the low-frequency components mainly contribute to the rough shape of the 3D object, we further introduce a low-frequency constraint to limit perturbations within imperceptible high-frequency components. Finally, the adversarial point cloud is generated by transforming the perturbed spectral representation back to the data domain via the inverse GFT. Experimental results demonstrate the effectiveness of the proposed attack in terms of both the imperceptibility and attack success rates.

摘要: 随着各种3D安全关键应用的日益关注，点云学习模型已被证明容易受到敌意攻击。现有的3D攻击方法虽然成功率很高，但都是以逐点扰动的方式深入数据空间，可能忽略了数据的几何特征。相反，我们从一个新的角度提出了点云攻击--图谱域攻击，目的是扰动对应于改变某些几何结构的谱域中的图变换系数。具体地说，利用图形信号处理，我们首先通过图形傅里叶变换(GFT)将点的坐标自适应地变换到谱域上以进行紧凑表示。然后，我们分析了不同谱带对几何结构的影响，并在此基础上提出了通过可学习的图谱滤波器来扰动GFT系数。考虑到低频分量主要影响三维物体的粗略形状，我们进一步引入了低频约束来限制不可察觉的高频分量内的扰动。最后，通过逆GFT将扰动后的谱表示变换回数据域，生成对抗性点云。实验结果表明，该攻击在不可感知性和攻击成功率方面都是有效的。



## **10. Perception-Aware Attack: Creating Adversarial Music via Reverse-Engineering Human Perception**

感知攻击：通过逆向工程人类感知创造对抗性音乐 cs.SD

ACM CCS 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13192v1)

**Authors**: Rui Duan, Zhe Qu, Shangqing Zhao, Leah Ding, Yao Liu, Zhuo Lu

**Abstracts**: Recently, adversarial machine learning attacks have posed serious security threats against practical audio signal classification systems, including speech recognition, speaker recognition, and music copyright detection. Previous studies have mainly focused on ensuring the effectiveness of attacking an audio signal classifier via creating a small noise-like perturbation on the original signal. It is still unclear if an attacker is able to create audio signal perturbations that can be well perceived by human beings in addition to its attack effectiveness. This is particularly important for music signals as they are carefully crafted with human-enjoyable audio characteristics.   In this work, we formulate the adversarial attack against music signals as a new perception-aware attack framework, which integrates human study into adversarial attack design. Specifically, we conduct a human study to quantify the human perception with respect to a change of a music signal. We invite human participants to rate their perceived deviation based on pairs of original and perturbed music signals, and reverse-engineer the human perception process by regression analysis to predict the human-perceived deviation given a perturbed signal. The perception-aware attack is then formulated as an optimization problem that finds an optimal perturbation signal to minimize the prediction of perceived deviation from the regressed human perception model. We use the perception-aware framework to design a realistic adversarial music attack against YouTube's copyright detector. Experiments show that the perception-aware attack produces adversarial music with significantly better perceptual quality than prior work.

摘要: 近年来，对抗性机器学习攻击对语音识别、说话人识别、音乐版权检测等实用音频信号分类系统构成了严重的安全威胁。以前的研究主要集中在通过在原始信号上产生类似噪声的小扰动来确保攻击音频信号分类器的有效性。目前还不清楚攻击者是否能够制造出人类能够很好地感知的音频信号扰动，以及它的攻击效率。这对于音乐信号尤其重要，因为它们是精心制作的，具有人类享受的音频特征。在这项工作中，我们将对音乐信号的对抗性攻击描述为一种新的感知感知攻击框架，将人类学习融入到对抗性攻击设计中。具体地说，我们进行了一项人类研究，以量化人类对音乐信号变化的感知。我们邀请人类参与者根据原始和扰动音乐信号对他们的感知偏差进行评级，并通过回归分析反向工程人类感知过程，以预测给定扰动信号的人感知偏差。然后，感知攻击被描述为一个优化问题，该优化问题找到最优扰动信号，以最小化对回归的人类感知模型的感知偏差的预测。我们使用感知感知框架设计了一个针对YouTube版权检测器的现实对抗性音乐攻击。实验表明，感知攻击产生的对抗性音乐的感知质量明显好于以往的工作。



## **11. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

FlashSyn：基于反例驱动近似的闪贷攻击合成 cs.PL

29 pages, 8 figures, technical report

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2206.10708v2)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstracts**: In decentralized finance (DeFi) ecosystem, lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with some fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow a large amount of assets without upfront collaterals deposits. Malicious adversaries can use flash loans to gather large amount of assets to launch costly exploitations targeting DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial contracts that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a new counterexamples-driven approximation refinement technique. We implement our framework in a tool called FlashSyn. We evaluate FlashSyn on 12 DeFi protocols that were victims to flash loan attacks and DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for each one of them.

摘要: 在去中心化金融(Defi)生态系统中，贷款人可以向借款人提供闪存贷款，即仅在区块链交易中有效且必须在该交易结束前支付一定费用的贷款。与正常贷款不同，闪付贷款允许借款人借入大量资产，而无需预付抵押金。恶意攻击者可以使用闪存贷款来收集大量资产，以发起针对Defi协议的代价高昂的攻击。在这篇文章中，我们介绍了一个新的框架，用于自动合成利用闪存贷款的Defi协议的对抗性合同。为了绕过DEFI协议的复杂性，我们提出了一种利用数值方法(多项式线性回归和最近邻内插)来逼近DEFI协议功能行为的新技术。然后，我们使用DEFI协议的近似函数构造一个优化查询，以找到由一系列具有最优参数的函数调用组成的对抗性攻击，从而给出最大利润。为了提高逼近的精度，我们提出了一种新的反例驱动的逼近求精技术。我们在一个名为FlashSyn的工具中实现我们的框架。我们对FlashSyn的12个Defi协议进行了评估，这些协议是闪电贷款攻击的受害者，并且是来自Damn Vulnerable Defi Challenges的Defi协议。FlashSyn会自动为它们中的每一个合成一次对抗性攻击。



## **12. Exploring the Unprecedented Privacy Risks of the Metaverse**

探索Metverse前所未有的隐私风险 cs.CR

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13176v1)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstracts**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.

摘要: 30名研究参与者在虚拟现实(VR)中玩了一个看起来很无辜的“逃生室”游戏。在幕后，一个对抗性的程序在玩游戏的短短几分钟内就准确地推断出了25个人的数据属性，从身高和翼展等人体测量数据到年龄和性别等人口统计数据。随着以渴望数据著称的公司越来越多地参与到VR开发中来，这种实验场景可能很快就会代表一种典型的VR用户体验。虽然虚拟远程呈现应用(以及所谓的虚拟现实)最近得到了主要科技公司越来越多的关注和投资，但从安全和隐私的角度来看，这些环境的研究仍然相对较少。在这项工作中，我们展示了VR攻击者如何从VRChat等流行虚拟世界应用程序的看似匿名的用户那里秘密确定数十个个人数据属性。这些攻击者可以像其他没有特殊权限的VR用户一样简单，而且这种数据收集的潜在规模和范围远远超出了传统移动和网络应用程序中的可行范围。我们的目标是阐明虚拟世界独特的隐私风险，并提供第一个整体框架，以了解这些新兴的虚拟现实生态系统中的侵入性数据收集攻击。



## **13. LGV: Boosting Adversarial Example Transferability from Large Geometric Vicinity**

LGV：增强来自大几何范围的对抗性范例的可转移性 cs.LG

Accepted at ECCV 2022

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2207.13129v1)

**Authors**: Martin Gubri, Maxime Cordy, Mike Papadakis, Yves Le Traon, Koushik Sen

**Abstracts**: We propose transferability from Large Geometric Vicinity (LGV), a new technique to increase the transferability of black-box adversarial attacks. LGV starts from a pretrained surrogate model and collects multiple weight sets from a few additional training epochs with a constant and high learning rate. LGV exploits two geometric properties that we relate to transferability. First, models that belong to a wider weight optimum are better surrogates. Second, we identify a subspace able to generate an effective surrogate ensemble among this wider optimum. Through extensive experiments, we show that LGV alone outperforms all (combinations of) four established test-time transformations by 1.8 to 59.9 percentage points. Our findings shed new light on the importance of the geometry of the weight space to explain the transferability of adversarial examples.

摘要: 为了提高黑盒对抗攻击的可转移性，我们提出了大几何邻域可转移性(LGV)的新技术。LGV从一个预先训练的代理模型开始，从几个额外的训练时期收集多个权值集，具有恒定和高的学习率。LGV利用了我们与可转移性相关的两个几何属性。首先，属于更广泛的重量最优的模型是更好的替代品。其次，我们在这个更广泛的最优解中识别出一个能够产生有效代理集成的子空间。通过广泛的实验，我们表明LGV本身就比所有四个已建立的测试时间转换(组合)高出1.8到59.9个百分点。我们的发现揭示了权重空间的几何对于解释对抗性例子的可转移性的重要性。



## **14. Making Corgis Important for Honeycomb Classification: Adversarial Attacks on Concept-based Explainability Tools**

让柯基对蜂巢分类变得重要：对基于概念的可解释性工具的对抗性攻击 cs.LG

AdvML Frontiers 2022 @ ICML 2022 workshop

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2110.07120v2)

**Authors**: Davis Brown, Henry Kvinge

**Abstracts**: Methods for model explainability have become increasingly critical for testing the fairness and soundness of deep learning. Concept-based interpretability techniques, which use a small set of human-interpretable concept exemplars in order to measure the influence of a concept on a model's internal representation of input, are an important thread in this line of research. In this work we show that these explainability methods can suffer the same vulnerability to adversarial attacks as the models they are meant to analyze. We demonstrate this phenomenon on two well-known concept-based interpretability methods: TCAV and faceted feature visualization. We show that by carefully perturbing the examples of the concept that is being investigated, we can radically change the output of the interpretability method. The attacks that we propose can either induce positive interpretations (polka dots are an important concept for a model when classifying zebras) or negative interpretations (stripes are not an important factor in identifying images of a zebra). Our work highlights the fact that in safety-critical applications, there is need for security around not only the machine learning pipeline but also the model interpretation process.

摘要: 对于测试深度学习的公平性和稳健性，模型可解释性的方法变得越来越重要。基于概念的可解释性技术是这一研究领域的一条重要线索，它使用一小部分人类可解释的概念样本来衡量概念对模型输入的内部表示的影响。在这项工作中，我们表明这些可解释性方法可以遭受与它们要分析的模型相同的对抗性攻击漏洞。我们在两种著名的基于概念的可解释性方法上演示了这一现象：TCAV和刻面特征可视化。我们证明，通过仔细地扰动正在研究的概念的例子，我们可以从根本上改变可解释性方法的输出。我们提出的攻击既可以引起积极的解释(斑马分类时，圆点是模型的重要概念)，也可以引起消极的解释(斑马的条纹不是识别斑马图像的重要因素)。我们的工作突出了这样一个事实，即在安全关键型应用程序中，不仅需要围绕机器学习管道，而且需要围绕模型解释过程进行安全保护。



## **15. TnT Attacks! Universal Naturalistic Adversarial Patches Against Deep Neural Network Systems**

TNT攻击！针对深度神经网络系统的普遍自然主义对抗性补丁 cs.CV

Accepted for publication in the IEEE Transactions on Information  Forensics & Security (TIFS)

**SubmitDate**: 2022-07-26    [paper-pdf](http://arxiv.org/pdf/2111.09999v2)

**Authors**: Bao Gia Doan, Minhui Xue, Shiqing Ma, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Deep neural networks are vulnerable to attacks from adversarial inputs and, more recently, Trojans to misguide or hijack the model's decision. We expose the existence of an intriguing class of spatially bounded, physically realizable, adversarial examples -- Universal NaTuralistic adversarial paTches -- we call TnTs, by exploring the superset of the spatially bounded adversarial example space and the natural input space within generative adversarial networks. Now, an adversary can arm themselves with a patch that is naturalistic, less malicious-looking, physically realizable, highly effective achieving high attack success rates, and universal. A TnT is universal because any input image captured with a TnT in the scene will: i) misguide a network (untargeted attack); or ii) force the network to make a malicious decision (targeted attack). Interestingly, now, an adversarial patch attacker has the potential to exert a greater level of control -- the ability to choose a location-independent, natural-looking patch as a trigger in contrast to being constrained to noisy perturbations -- an ability is thus far shown to be only possible with Trojan attack methods needing to interfere with the model building processes to embed a backdoor at the risk discovery; but, still realize a patch deployable in the physical world. Through extensive experiments on the large-scale visual classification task, ImageNet with evaluations across its entire validation set of 50,000 images, we demonstrate the realistic threat from TnTs and the robustness of the attack. We show a generalization of the attack to create patches achieving higher attack success rates than existing state-of-the-art methods. Our results show the generalizability of the attack to different visual classification tasks (CIFAR-10, GTSRB, PubFig) and multiple state-of-the-art deep neural networks such as WideResnet50, Inception-V3 and VGG-16.

摘要: 深度神经网络很容易受到敌意输入的攻击，最近还受到特洛伊木马的攻击，以误导或劫持模型的决策。我们通过探索空间受限的对抗性实例空间和生成性对抗性网络中的自然输入空间的超集，揭示了一类有趣的空间有界的、物理上可实现的对抗性例子的存在--通用的自然主义对抗性斑块--我们称之为TNTs。现在，对手可以用一个自然主义的、看起来不那么恶毒的、物理上可实现的、高效的、实现高攻击成功率和通用性的补丁来武装自己。TNT是通用的，因为在场景中使用TNT捕获的任何输入图像将：i)误导网络(非定向攻击)；或ii)迫使网络做出恶意决策(定向攻击)。有趣的是，现在，敌意补丁攻击者有可能施加更高级别的控制--选择与位置无关的、看起来自然的补丁作为触发器的能力，而不是受限于嘈杂的干扰--到目前为止，这种能力被证明只有在需要干扰模型构建过程以在风险发现时嵌入后门的特洛伊木马攻击方法中才是可能的；但是，仍然实现了可在物理世界中部署的补丁。通过对大规模视觉分类任务ImageNet的大量实验，以及对其50,000张图像的整个验证集的评估，我们展示了TNT的现实威胁和攻击的健壮性。我们展示了创建补丁的攻击的泛化，实现了比现有最先进的方法更高的攻击成功率。实验结果表明，该攻击对不同的视觉分类任务(CIFAR-10、GTSRB、PubFig)和多种最新的深度神经网络如WideResnet50、Inception-V3和VGG-16具有较强的泛化能力。



## **16. Verification-Aided Deep Ensemble Selection**

辅助验证的深度集成选择 cs.LG

To appear in FMCAD 2022

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2202.03898v2)

**Authors**: Guy Amir, Tom Zelazny, Guy Katz, Michael Schapira

**Abstracts**: Deep neural networks (DNNs) have become the technology of choice for realizing a variety of complex tasks. However, as highlighted by many recent studies, even an imperceptible perturbation to a correctly classified input can lead to misclassification by a DNN. This renders DNNs vulnerable to strategic input manipulations by attackers, and also oversensitive to environmental noise.   To mitigate this phenomenon, practitioners apply joint classification by an *ensemble* of DNNs. By aggregating the classification outputs of different individual DNNs for the same input, ensemble-based classification reduces the risk of misclassifications due to the specific realization of the stochastic training process of any single DNN. However, the effectiveness of a DNN ensemble is highly dependent on its members *not simultaneously erring* on many different inputs.   In this case study, we harness recent advances in DNN verification to devise a methodology for identifying ensemble compositions that are less prone to simultaneous errors, even when the input is adversarially perturbed -- resulting in more robustly-accurate ensemble-based classification.   Our proposed framework uses a DNN verifier as a backend, and includes heuristics that help reduce the high complexity of directly verifying ensembles. More broadly, our work puts forth a novel universal objective for formal verification that can potentially improve the robustness of real-world, deep-learning-based systems across a variety of application domains.

摘要: 深度神经网络(DNN)已经成为实现各种复杂任务的首选技术。然而，正如最近的许多研究所强调的那样，即使是对正确分类的输入进行了不可察觉的扰动，也可能导致DNN的错误分类。这使得DNN容易受到攻击者的战略性输入操纵，并且对环境噪声过于敏感。为了缓解这一现象，从业者根据DNN的“集合”进行联合分类。通过聚合不同个体DNN对同一输入的分类输出，基于集成的分类降低了由于任意单个DNN的随机训练过程的具体实现而导致的误分类风险。然而，DNN合奏的有效性高度依赖于其成员，而不是同时在许多不同的输入上出错。在这个案例研究中，我们利用DNN验证方面的最新进展来设计一种方法，用于识别不太容易同时出错的集成成分，即使在输入受到相反的扰动时也是如此--从而产生更健壮的基于集成的分类。我们提出的框架使用DNN验证器作为后端，并包括有助于降低直接验证集成的高复杂性的启发式算法。更广泛地说，我们的工作为形式验证提出了一个新的通用目标，可以潜在地提高现实世界中基于深度学习的系统在各种应用领域的健壮性。



## **17. $p$-DkNN: Out-of-Distribution Detection Through Statistical Testing of Deep Representations**

$p$-DkNN：基于深度表示统计测试的失配检测 cs.LG

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12545v1)

**Authors**: Adam Dziedzic, Stephan Rabanser, Mohammad Yaghini, Armin Ale, Murat A. Erdogdu, Nicolas Papernot

**Abstracts**: The lack of well-calibrated confidence estimates makes neural networks inadequate in safety-critical domains such as autonomous driving or healthcare. In these settings, having the ability to abstain from making a prediction on out-of-distribution (OOD) data can be as important as correctly classifying in-distribution data. We introduce $p$-DkNN, a novel inference procedure that takes a trained deep neural network and analyzes the similarity structures of its intermediate hidden representations to compute $p$-values associated with the end-to-end model prediction. The intuition is that statistical tests performed on latent representations can serve not only as a classifier, but also offer a statistically well-founded estimation of uncertainty. $p$-DkNN is scalable and leverages the composition of representations learned by hidden layers, which makes deep representation learning successful. Our theoretical analysis builds on Neyman-Pearson classification and connects it to recent advances in selective classification (reject option). We demonstrate advantageous trade-offs between abstaining from predicting on OOD inputs and maintaining high accuracy on in-distribution inputs. We find that $p$-DkNN forces adaptive attackers crafting adversarial examples, a form of worst-case OOD inputs, to introduce semantically meaningful changes to the inputs.

摘要: 缺乏经过良好校准的置信度估计，使得神经网络在自动驾驶或医疗保健等安全关键领域不够充分。在这些设置中，能够避免对分布外(OOD)数据进行预测与正确地对分布内数据进行分类一样重要。我们介绍了一种新的推理过程$p$-DkNN，它利用训练好的深度神经网络并分析其中间隐含表示的相似结构来计算与端到端模型预测相关的$p$值。人们的直觉是，对潜在表征进行的统计测试不仅可以作为分类器，还可以提供对不确定性的统计上有充分依据的估计。$p$-DkNN是可伸缩的，并利用隐藏层学习的表示的组合，这使得深度表示学习成功。我们的理论分析建立在Neyman-Pearson分类的基础上，并将其与选择性分类(拒绝选项)的最新进展联系起来。我们展示了在避免预测OOD输入和保持分布内输入的高准确性之间的有利权衡。我们发现，$p$-DkNN迫使自适应攻击者精心制作敌意示例，这是最坏情况下OOD输入的一种形式，以对输入进行语义上有意义的更改。



## **18. TAFIM: Targeted Adversarial Attacks against Facial Image Manipulations**

TAFIM：针对面部图像处理的有针对性的对抗性攻击 cs.CV

(ECCV 2022 Paper) Video: https://youtu.be/11VMOJI7tKg Project Page:  https://shivangi-aneja.github.io/projects/tafim/

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2112.09151v2)

**Authors**: Shivangi Aneja, Lev Markhasin, Matthias Niessner

**Abstracts**: Face manipulation methods can be misused to affect an individual's privacy or to spread disinformation. To this end, we introduce a novel data-driven approach that produces image-specific perturbations which are embedded in the original images. The key idea is that these protected images prevent face manipulation by causing the manipulation model to produce a predefined manipulation target (uniformly colored output image in our case) instead of the actual manipulation. In addition, we propose to leverage differentiable compression approximation, hence making generated perturbations robust to common image compression. In order to prevent against multiple manipulation methods simultaneously, we further propose a novel attention-based fusion of manipulation-specific perturbations. Compared to traditional adversarial attacks that optimize noise patterns for each image individually, our generalized model only needs a single forward pass, thus running orders of magnitude faster and allowing for easy integration in image processing stacks, even on resource-constrained devices like smartphones.

摘要: 面部处理方法可能被滥用来影响个人隐私或传播虚假信息。为此，我们引入了一种新的数据驱动方法，该方法产生嵌入在原始图像中的特定于图像的扰动。其关键思想是，这些受保护的图像通过使操纵模型产生预定义的操纵目标(在我们的例子中为均匀着色的输出图像)而不是实际的操纵来防止面部操纵。此外，我们建议利用可微压缩近似，从而使所产生的扰动对普通图像压缩具有健壮性。为了防止多种操作方法同时出现，我们进一步提出了一种新的基于注意力的操作特定扰动融合方法。与分别优化每个图像的噪声模式的传统对抗性攻击相比，我们的通用模型只需要一次前向传递，因此运行速度快几个数量级，并允许轻松集成到图像处理堆栈中，即使在智能手机等资源受限的设备上也是如此。



## **19. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

SegPGD：一种评估和提高分割健壮性的高效对抗性攻击 cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12391v1)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstracts**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.

摘要: 基于深度神经网络的图像分类容易受到对抗性扰动的影响。通过在输入图像中添加人为的微小和不可察觉的扰动，可以很容易地欺骗图像分类。对抗性训练作为最有效的防御策略之一，被提出用来解决分类模型的脆弱性，即在训练过程中创建对抗性实例并注入训练数据。分类模型的攻防问题在过去的几年里得到了广泛的研究。语义切分作为分类的延伸，近年来也受到了极大的关注。最近的工作表明，需要大量的攻击迭代来创建有效的对抗性示例来愚弄分段模型。这种观察结果使得分割模型的健壮性评估和对抗性训练都具有挑战性。在这项工作中，我们提出了一种有效且高效的分段攻击方法，称为SegPGD。此外，我们还进行了收敛分析，结果表明，在相同的攻击迭代次数下，所提出的SegPGD算法能够生成比PGD算法更有效的攻击实例。此外，我们建议将我们的SegPGD作为分割对手训练的底层攻击方法。由于SegPGD可以生成更有效的对抗性实例，因此使用我们的SegPGD进行对抗性训练可以提高分割模型的稳健性。在流行的分割模型体系结构和标准分割数据集上的实验也验证了我们的建议。



## **20. Adversarial Attack across Datasets**

跨数据集的对抗性攻击 cs.CV

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2110.07718v2)

**Authors**: Yunxiao Qin, Yuanhao Xiong, Jinfeng Yi, Lihong Cao, Cho-Jui Hsieh

**Abstracts**: Existing transfer attack methods commonly assume that the attacker knows the training set (e.g., the label set, the input size) of the black-box victim models, which is usually unrealistic because in some cases the attacker cannot know this information. In this paper, we define a Generalized Transferable Attack (GTA) problem where the attacker doesn't know this information and is acquired to attack any randomly encountered images that may come from unknown datasets. To solve the GTA problem, we propose a novel Image Classification Eraser (ICE) that trains a particular attacker to erase classification information of any images from arbitrary datasets. Experiments on several datasets demonstrate that ICE greatly outperforms existing transfer attacks on GTA, and show that ICE uses similar texture-like noises to perturb different images from different datasets. Moreover, fast fourier transformation analysis indicates that the main components in each ICE noise are three sine waves for the R, G, and B image channels. Inspired by this interesting finding, we then design a novel Sine Attack (SA) method to optimize the three sine waves. Experiments show that SA performs comparably to ICE, indicating that the three sine waves are effective and enough to break DNNs under the GTA setting.

摘要: 现有的传输攻击方法通常假设攻击者知道黑盒受害者模型的训练集(例如，标签集、输入大小)，这通常是不现实的，因为在某些情况下攻击者无法知道该信息。在本文中，我们定义了一个广义可转移攻击(GTA)问题，其中攻击者不知道这些信息，并且被获取来攻击任何可能来自未知数据集的随机遇到的图像。为了解决GTA问题，我们提出了一种新的图像分类橡皮擦(ICE)，它训练特定的攻击者从任意数据集中擦除任何图像的分类信息。在几个数据集上的实验表明，ICE的性能大大优于现有的GTA传输攻击，并表明ICE使用类似纹理的噪声来扰动来自不同数据集的不同图像。此外，快速傅立叶变换分析表明，每个ICE噪声的主要分量是R、G和B图像通道的三个正弦波。受这一有趣发现的启发，我们设计了一种新的正弦攻击(SA)方法来优化三个正弦波。实验表明，SA的性能与ICE相当，说明在GTA设置下，这三个正弦波都是有效的，足以破解DNN。



## **21. Improving Adversarial Robustness via Mutual Information Estimation**

利用互信息估计提高对手的稳健性 cs.LG

This version has modified Eq.2 and its proof in the published version

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12203v1)

**Authors**: Dawei Zhou, Nannan Wang, Xinbo Gao, Bo Han, Xiaoyu Wang, Yibing Zhan, Tongliang Liu

**Abstracts**: Deep neural networks (DNNs) are found to be vulnerable to adversarial noise. They are typically misled by adversarial samples to make wrong predictions. To alleviate this negative effect, in this paper, we investigate the dependence between outputs of the target model and input adversarial samples from the perspective of information theory, and propose an adversarial defense method. Specifically, we first measure the dependence by estimating the mutual information (MI) between outputs and the natural patterns of inputs (called natural MI) and MI between outputs and the adversarial patterns of inputs (called adversarial MI), respectively. We find that adversarial samples usually have larger adversarial MI and smaller natural MI compared with those w.r.t. natural samples. Motivated by this observation, we propose to enhance the adversarial robustness by maximizing the natural MI and minimizing the adversarial MI during the training process. In this way, the target model is expected to pay more attention to the natural pattern that contains objective semantics. Empirical evaluations demonstrate that our method could effectively improve the adversarial accuracy against multiple attacks.

摘要: 深度神经网络(DNN)被发现容易受到对抗性噪声的影响。他们通常会被对抗性样本误导，做出错误的预测。为了缓解这种负面影响，本文从信息论的角度研究了目标模型的输出与输入敌方样本之间的依赖关系，并提出了一种对抗性防御方法。具体地说，我们首先通过估计输出与输入的自然模式之间的互信息(称为自然MI)和输出与输入的对抗性模式之间的互信息(称为对抗性MI)来度量依赖。我们发现，与W.r.t.相比，对抗性样本通常具有较大的对抗性MI和较小的自然MI。天然样品。基于这一观察结果，我们提出在训练过程中通过最大化自然MI和最小化对手MI来增强对手的稳健性。这样，目标模型将更多地关注包含客观语义的自然模式。实验结果表明，该方法能够有效地提高对抗多重攻击的准确率。



## **22. Versatile Weight Attack via Flipping Limited Bits**

通过翻转有限比特进行多功能重量攻击 cs.CR

Extension of our ICLR 2021 work: arXiv:2102.10496

**SubmitDate**: 2022-07-25    [paper-pdf](http://arxiv.org/pdf/2207.12405v1)

**Authors**: Jiawang Bai, Baoyuan Wu, Zhifeng Li, Shu-tao Xia

**Abstracts**: To explore the vulnerability of deep neural networks (DNNs), many attack paradigms have been well studied, such as the poisoning-based backdoor attack in the training stage and the adversarial attack in the inference stage. In this paper, we study a novel attack paradigm, which modifies model parameters in the deployment stage. Considering the effectiveness and stealthiness goals, we provide a general formulation to perform the bit-flip based weight attack, where the effectiveness term could be customized depending on the attacker's purpose. Furthermore, we present two cases of the general formulation with different malicious purposes, i.e., single sample attack (SSA) and triggered samples attack (TSA). To this end, we formulate this problem as a mixed integer programming (MIP) to jointly determine the state of the binary bits (0 or 1) in the memory and learn the sample modification. Utilizing the latest technique in integer programming, we equivalently reformulate this MIP problem as a continuous optimization problem, which can be effectively and efficiently solved using the alternating direction method of multipliers (ADMM) method. Consequently, the flipped critical bits can be easily determined through optimization, rather than using a heuristic strategy. Extensive experiments demonstrate the superiority of SSA and TSA in attacking DNNs.

摘要: 为了探索深层神经网络的脆弱性，人们研究了许多攻击范例，如训练阶段的基于中毒的后门攻击和推理阶段的对抗性攻击。本文研究了一种在部署阶段对模型参数进行修改的新型攻击范式。考虑到攻击的有效性和隐蔽性目标，我们给出了基于比特翻转的权重攻击的一般公式，其中有效项可以根据攻击者的目的进行定制。此外，我们还给出了两种具有不同恶意目的的通用公式，即单样本攻击(SSA)和触发样本攻击(TSA)。为此，我们将该问题描述为混合整数规划(MIP)，以共同确定存储器中二进制位(0或1)的状态并学习样本修改。利用整数规划的最新技术，我们将MIP问题等价地转化为一个连续优化问题，并利用乘子交替方向法(ADMM)对其进行了有效求解。因此，可以通过优化而不是使用启发式策略来容易地确定翻转的关键比特。大量的实验证明了SSA和TSA在攻击DNN方面的优势。



## **23. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11788v1)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, two privacy-preserving schemes are proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving schemes.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了两种隐私保护方案，这两种方案在保留VFL给主动方带来的全部利益的同时，恶化了对手的重构攻击。最后，实验结果证明了所提出的攻击和隐私保护方案的有效性。



## **24. Can we achieve robustness from data alone?**

我们能仅从数据中获得稳健性吗？ cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11727v1)

**Authors**: Nikolaos Tsilivis, Jingtong Su, Julia Kempe

**Abstracts**: Adversarial training and its variants have come to be the prevailing methods to achieve adversarially robust classification using neural networks. However, its increased computational cost together with the significant gap between standard and robust performance hinder progress and beg the question of whether we can do better. In this work, we take a step back and ask: Can models achieve robustness via standard training on a suitably optimized set? To this end, we devise a meta-learning method for robust classification, that optimizes the dataset prior to its deployment in a principled way, and aims to effectively remove the non-robust parts of the data. We cast our optimization method as a multi-step PGD procedure on kernel regression, with a class of kernels that describe infinitely wide neural nets (Neural Tangent Kernels - NTKs). Experiments on MNIST and CIFAR-10 demonstrate that the datasets we produce enjoy very high robustness against PGD attacks, when deployed in both kernel regression classifiers and neural networks. However, this robustness is somewhat fallacious, as alternative attacks manage to fool the models, which we find to be the case for previous similar works in the literature as well. We discuss potential reasons for this and outline further avenues of research.

摘要: 对抗性训练及其变体已经成为使用神经网络实现对抗性稳健分类的主流方法。然而，它增加的计算成本以及标准和健壮性能之间的巨大差距阻碍了进步，并提出了我们是否可以做得更好的问题。在这项工作中，我们退一步问：模型能否通过在适当优化的集合上进行标准训练来实现健壮性？为此，我们设计了一种用于稳健分类的元学习方法，该方法在数据集部署之前有原则地对其进行优化，旨在有效地去除数据中不稳健的部分。我们把我们的优化方法归结为一个关于核回归的多步PGD过程，用一类描述无限宽神经网络的核--神经切核-NTK来描述。在MNIST和CIFAR-10上的实验表明，当我们生成的数据集部署在核回归分类器和神经网络中时，对PGD攻击具有非常高的鲁棒性。然而，这种健壮性在某种程度上是错误的，因为替代攻击设法愚弄了模型，我们发现之前文献中的类似作品也是如此。我们讨论了造成这种情况的潜在原因，并概述了进一步的研究途径。



## **25. Proving Common Mechanisms Shared by Twelve Methods of Boosting Adversarial Transferability**

证明提高对抗转移能力的十二种方法所共有的共同机制 cs.LG

**SubmitDate**: 2022-07-24    [paper-pdf](http://arxiv.org/pdf/2207.11694v1)

**Authors**: Quanshi Zhang, Xin Wang, Jie Ren, Xu Cheng, Shuyun Lin, Yisen Wang, Xiangming Zhu

**Abstracts**: Although many methods have been proposed to enhance the transferability of adversarial perturbations, these methods are designed in a heuristic manner, and the essential mechanism for improving adversarial transferability is still unclear. This paper summarizes the common mechanism shared by twelve previous transferability-boosting methods in a unified view, i.e., these methods all reduce game-theoretic interactions between regional adversarial perturbations. To this end, we focus on the attacking utility of all interactions between regional adversarial perturbations, and we first discover and prove the negative correlation between the adversarial transferability and the attacking utility of interactions. Based on this discovery, we theoretically prove and empirically verify that twelve previous transferability-boosting methods all reduce interactions between regional adversarial perturbations. More crucially, we consider the reduction of interactions as the essential reason for the enhancement of adversarial transferability. Furthermore, we design the interaction loss to directly penalize interactions between regional adversarial perturbations during attacking. Experimental results show that the interaction loss significantly improves the transferability of adversarial perturbations.

摘要: 虽然已经提出了许多方法来增强对抗性扰动的可转移性，但这些方法都是以启发式的方式设计的，提高对抗性可转移性的基本机制仍然不清楚。本文从一个统一的角度总结了12种可转移性增强方法所共有的共同机制，即这些方法都减少了区域对抗性扰动之间的博弈论互动。为此，我们重点研究了区域对抗性扰动之间所有相互作用的攻击效用，并首次发现并证明了对抗性转移与相互作用的攻击效用之间的负相关关系。基于这一发现，我们从理论上证明并实证验证了以前的12种转移增强方法都减少了区域对抗性扰动之间的交互作用。更重要的是，我们认为减少互动是加强对抗性可转移性的根本原因。此外，我们设计了交互损失来直接惩罚攻击过程中区域对抗性扰动之间的交互。实验结果表明，交互损失显著提高了对抗性扰动的可转移性。



## **26. Testing the Robustness of Learned Index Structures**

测试学习索引结构的稳健性 cs.DB

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2207.11575v1)

**Authors**: Matthias Bachfischer, Renata Borovica-Gajic, Benjamin I. P. Rubinstein

**Abstracts**: While early empirical evidence has supported the case for learned index structures as having favourable average-case performance, little is known about their worst-case performance. By contrast, classical structures are known to achieve optimal worst-case behaviour. This work evaluates the robustness of learned index structures in the presence of adversarial workloads. To simulate adversarial workloads, we carry out a data poisoning attack on linear regression models that manipulates the cumulative distribution function (CDF) on which the learned index model is trained. The attack deteriorates the fit of the underlying ML model by injecting a set of poisoning keys into the training dataset, which leads to an increase in the prediction error of the model and thus deteriorates the overall performance of the learned index structure. We assess the performance of various regression methods and the learned index implementations ALEX and PGM-Index. We show that learned index structures can suffer from a significant performance deterioration of up to 20% when evaluated on poisoned vs. non-poisoned datasets.

摘要: 尽管早期的经验证据支持习得的指数结构具有有利的平均表现，但人们对其最坏情况的表现知之甚少。相比之下，众所周知，经典结构可以实现最坏情况下的最佳行为。这项工作评估了学习的索引结构在存在对抗性工作负载的情况下的稳健性。为了模拟敌意工作负载，我们对线性回归模型执行数据中毒攻击，该攻击操纵学习的指数模型在其上训练的累积分布函数(CDF)。该攻击通过向训练数据集注入一组中毒关键字来降低底层ML模型的适配度，从而导致模型预测误差的增加，从而降低学习到的索引结构的整体性能。我们评估了各种回归方法和学习到的索引实现Alex和PGM-Index的性能。我们发现，在有毒数据集和非有毒数据集上进行评估时，学习的索引结构可能会遭受高达20%的显著性能下降。



## **27. How does Heterophily Impact the Robustness of Graph Neural Networks? Theoretical Connections and Practical Implications**

异构性如何影响图神经网络的健壮性？理论联系和实践意义 cs.LG

KDD 2022 camera ready version + full appendix; 20 pages, 2 figures

**SubmitDate**: 2022-07-23    [paper-pdf](http://arxiv.org/pdf/2106.07767v4)

**Authors**: Jiong Zhu, Junchen Jin, Donald Loveland, Michael T. Schaub, Danai Koutra

**Abstracts**: We bridge two research directions on graph neural networks (GNNs), by formalizing the relation between heterophily of node labels (i.e., connected nodes tend to have dissimilar labels) and the robustness of GNNs to adversarial attacks. Our theoretical and empirical analyses show that for homophilous graph data, impactful structural attacks always lead to reduced homophily, while for heterophilous graph data the change in the homophily level depends on the node degrees. These insights have practical implications for defending against attacks on real-world graphs: we deduce that separate aggregators for ego- and neighbor-embeddings, a design principle which has been identified to significantly improve prediction for heterophilous graph data, can also offer increased robustness to GNNs. Our comprehensive experiments show that GNNs merely adopting this design achieve improved empirical and certifiable robustness compared to the best-performing unvaccinated model. Additionally, combining this design with explicit defense mechanisms against adversarial attacks leads to an improved robustness with up to 18.33% performance increase under attacks compared to the best-performing vaccinated model.

摘要: 通过形式化节点标签的异质性(即连接的节点往往具有不同的标签)与图神经网络对对手攻击的健壮性之间的关系，我们在图神经网络(GNN)的两个研究方向之间架起了桥梁。我们的理论和实证分析表明，对于同嗜性的图数据，有效的结构攻击总是导致同质性的降低，而对于异嗜性的图数据，同质性水平的变化取决于节点度。这些见解对防御真实世界图上的攻击具有实际意义：我们推断，针对自我和邻居嵌入的单独聚集器，这一设计原则已被确定为显著改善对异嗜图数据的预测，也可以提高GNN的健壮性。我们的综合实验表明，与性能最好的未接种疫苗模型相比，仅采用这种设计的GNN获得了更好的经验和可证明的稳健性。此外，将此设计与针对对手攻击的显式防御机制相结合，可以提高健壮性，与性能最好的疫苗模型相比，在攻击下的性能最高可提高18.33%。



## **28. Do Perceptually Aligned Gradients Imply Adversarial Robustness?**

感知上对齐的梯度是否意味着对抗的健壮性？ cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.11378v1)

**Authors**: Roy Ganz, Bahjat Kawar, Michael Elad

**Abstracts**: In the past decade, deep learning-based networks have achieved unprecedented success in numerous tasks, including image classification. Despite this remarkable achievement, recent studies have demonstrated that such networks are easily fooled by small malicious perturbations, also known as adversarial examples. This security weakness led to extensive research aimed at obtaining robust models. Beyond the clear robustness benefits of such models, it was also observed that their gradients with respect to the input align with human perception. Several works have identified Perceptually Aligned Gradients (PAG) as a byproduct of robust training, but none have considered it as a standalone phenomenon nor studied its own implications. In this work, we focus on this trait and test whether Perceptually Aligned Gradients imply Robustness. To this end, we develop a novel objective to directly promote PAG in training classifiers and examine whether models with such gradients are more robust to adversarial attacks. Extensive experiments on CIFAR-10 and STL validate that such models have improved robust performance, exposing the surprising bidirectional connection between PAG and robustness.

摘要: 在过去的十年中，基于深度学习的网络在包括图像分类在内的众多任务中取得了前所未有的成功。尽管取得了如此显著的成就，但最近的研究表明，这种网络很容易被微小的恶意扰动所愚弄，也被称为对抗性例子。这一安全弱点导致了旨在获得健壮模型的广泛研究。除了这类模型的明显稳健性好处外，还观察到它们相对于输入的梯度与人的感知一致。一些研究已经发现知觉对齐梯度(PAG)是稳健训练的副产品，但没有人将其视为一种独立的现象，也没有研究其本身的含义。在这项工作中，我们关注这一特征，并测试感知对齐的梯度是否意味着稳健性。为此，我们提出了一个新的目标，即在训练分类器时直接推广PAG，并检验具有这种梯度的模型是否对对手攻击更健壮。在CIFAR-10和STL上的广泛实验验证了这些模型提高了稳健性能，揭示了PAG和稳健性之间令人惊讶的双向联系。



## **29. Practical Privacy Attacks on Vertical Federated Learning**

针对垂直联合学习的实用隐私攻击 cs.CR

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2011.09290v3)

**Authors**: Haiqin Weng, Juntao Zhang, Xingjun Ma, Feng Xue, Tao Wei, Shouling Ji, Zhiyuan Zong

**Abstracts**: Federated learning (FL) is a privacy-preserving learning paradigm that allows multiple parities to jointly train a powerful machine learning model without sharing their private data. According to the form of collaboration, FL can be further divided into horizontal federated learning (HFL) and vertical federated learning (VFL). In HFL, participants share the same feature space and collaborate on data samples, while in VFL, participants share the same sample IDs and collaborate on features. VFL has a broader scope of applications and is arguably more suitable for joint model training between large enterprises.   In this paper, we focus on VFL and investigate potential privacy leakage in real-world VFL frameworks. We design and implement two practical privacy attacks: reverse multiplication attack for the logistic regression VFL protocol; and reverse sum attack for the XGBoost VFL protocol. We empirically show that the two attacks are (1) effective - the adversary can successfully steal the private training data, even when the intermediate outputs are encrypted to protect data privacy; (2) evasive - the attacks do not deviate from the protocol specification nor deteriorate the accuracy of the target model; and (3) easy - the adversary needs little prior knowledge about the data distribution of the target participant. We also show the leaked information is as effective as the raw training data in training an alternative classifier. We further discuss potential countermeasures and their challenges, which we hope can lead to several promising research directions.

摘要: 联合学习(FL)是一种隐私保护的学习范式，允许多个奇偶校验在不共享其私人数据的情况下联合训练一个强大的机器学习模型。根据协作形式的不同，外语学习又可分为水平联合学习和垂直联合学习。在HFL中，参与者共享相同的特征空间并就数据样本进行协作，而在VFL中，参与者共享相同的样本ID并就特征进行协作。VFL的应用范围更广，可以说更适合大型企业之间的联合模式培训。在本文中，我们聚焦于虚拟现实语言，并调查现实世界虚拟现实语言框架中潜在的隐私泄漏。设计并实现了两种实用的隐私攻击：针对Logistic回归VFL协议的反向乘法攻击和针对XGBoost VFL协议的反向求和攻击。我们的经验表明，这两种攻击是有效的--攻击者可以成功窃取私人训练数据，即使中间输出被加密以保护数据隐私；(2)规避-攻击不偏离协议规范，也不会恶化目标模型的准确性；以及(3)易用性--攻击者几乎不需要关于目标参与者的数据分布的先验知识。我们还表明，泄漏的信息在训练另一种分类器时与原始训练数据一样有效。我们进一步讨论了潜在的对策及其挑战，我们希望这可以引导出几个有前途的研究方向。



## **30. On Higher Adversarial Susceptibility of Contrastive Self-Supervised Learning**

关于对比性自我监督学习的高对抗敏感性 cs.CV

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.10862v1)

**Authors**: Rohit Gupta, Naveed Akhtar, Ajmal Mian, Mubarak Shah

**Abstracts**: Contrastive self-supervised learning (CSL) has managed to match or surpass the performance of supervised learning in image and video classification. However, it is still largely unknown if the nature of the representation induced by the two learning paradigms is similar. We investigate this under the lens of adversarial robustness. Our analytical treatment of the problem reveals intrinsic higher sensitivity of CSL over supervised learning. It identifies the uniform distribution of data representation over a unit hypersphere in the CSL representation space as the key contributor to this phenomenon. We establish that this increases model sensitivity to input perturbations in the presence of false negatives in the training data. Our finding is supported by extensive experiments for image and video classification using adversarial perturbations and other input corruptions. Building on the insights, we devise strategies that are simple, yet effective in improving model robustness with CSL training. We demonstrate up to 68% reduction in the performance gap between adversarially attacked CSL and its supervised counterpart. Finally, we contribute to robust CSL paradigm by incorporating our findings in adversarial self-supervised learning. We demonstrate an average gain of about 5% over two different state-of-the-art methods in this domain.

摘要: 对比自监督学习(CSL)在图像和视频分类中的性能已经达到或超过了监督学习。然而，这两种学习范式诱导的表征的性质是否相似仍在很大程度上是未知的。我们在对抗稳健性的视角下对此进行了研究。我们对问题的分析处理揭示了CSL对监督学习的内在更高的敏感性。它认为CSL表示空间中单位超球面上数据表示的均匀分布是造成这一现象的关键因素。我们证明，在训练数据中存在假阴性的情况下，这增加了模型对输入扰动的敏感性。我们的发现得到了使用对抗性扰动和其他输入损坏的图像和视频分类的广泛实验的支持。在这些见解的基础上，我们制定了简单但有效的策略，通过CSL培训提高模型的健壮性。我们展示了被对手攻击的CSL与其受监督的对手之间的性能差距降低了高达68%。最后，我们通过将我们的发现纳入对抗性自我监督学习中，为稳健的CSL范式做出了贡献。在这一领域，我们通过两种不同的最先进的方法展示了大约5%的平均增益。



## **31. Adversarially-Aware Robust Object Detector**

对抗性感知的鲁棒目标检测器 cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2207.06202v3)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.

摘要: 随着深度神经网络的出现，目标检测作为一项基本的计算机视觉任务已经取得了显著的进展。然而，很少有研究探讨对象检测器在各种真实场景中的实际应用中抵抗对手攻击的对抗性健壮性。检测器受到了不可察觉的扰动的极大挑战，在干净图像上的性能急剧下降，在对抗性图像上的性能极差。在这项工作中，我们经验性地探索了目标检测中对抗鲁棒性的模型训练，这在很大程度上归因于学习干净图像和对抗图像之间的冲突。为了缓解这一问题，我们提出了一种基于对抗性感知卷积的稳健检测器(RobustDet)，用于在干净图像和对抗性图像上进行模型学习。RobustDet还采用了对抗性图像鉴别器(AID)和重建一致特征(CFR)，以确保可靠的健壮性。在PASCAL、VOC和MS-COCO上的大量实验表明，该模型在保持对干净图像的检测能力的同时，有效地解开了梯度的纠缠，显著提高了检测的鲁棒性。



## **32. Boosting Transferability of Targeted Adversarial Examples via Hierarchical Generative Networks**

通过层次化生成网络提高目标对抗性实例的可转移性 cs.LG

**SubmitDate**: 2022-07-22    [paper-pdf](http://arxiv.org/pdf/2107.01809v2)

**Authors**: Xiao Yang, Yinpeng Dong, Tianyu Pang, Hang Su, Jun Zhu

**Abstracts**: Transfer-based adversarial attacks can evaluate model robustness in the black-box setting. Several methods have demonstrated impressive untargeted transferability, however, it is still challenging to efficiently produce targeted transferability. To this end, we develop a simple yet effective framework to craft targeted transfer-based adversarial examples, applying a hierarchical generative network. In particular, we contribute to amortized designs that well adapt to multi-class targeted attacks. Extensive experiments on ImageNet show that our method improves the success rates of targeted black-box attacks by a significant margin over the existing methods -- it reaches an average success rate of 29.1\% against six diverse models based only on one substitute white-box model, which significantly outperforms the state-of-the-art gradient-based attack methods. Moreover, the proposed method is also more efficient beyond an order of magnitude than gradient-based methods.

摘要: 基于传输的对抗性攻击可以在黑盒环境下评估模型的稳健性。几种方法已经证明了令人印象深刻的非定向可转移性，然而，有效地产生定向可转移性仍然具有挑战性。为此，我们开发了一个简单而有效的框架，应用分层生成网络来制作有针对性的基于迁移的对抗性例子。特别是，我们为能够很好地适应多类别目标攻击的分期设计做出了贡献。在ImageNet上的大量实验表明，与现有方法相比，该方法显著提高了目标黑盒攻击的成功率--仅基于一个替代白盒模型，对6种不同模型的平均成功率达到29.1，显著优于最先进的基于梯度的攻击方法。此外，该方法的效率也比基于梯度的方法高出一个数量级。



## **33. Synthetic Dataset Generation for Adversarial Machine Learning Research**

用于对抗性机器学习研究的合成数据集生成 cs.CV

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10719v1)

**Authors**: Xiruo Liu, Shibani Singh, Cory Cornelius, Colin Busho, Mike Tan, Anindya Paul, Jason Martin

**Abstracts**: Existing adversarial example research focuses on digitally inserted perturbations on top of existing natural image datasets. This construction of adversarial examples is not realistic because it may be difficult, or even impossible, for an attacker to deploy such an attack in the real-world due to sensing and environmental effects. To better understand adversarial examples against cyber-physical systems, we propose approximating the real-world through simulation. In this paper we describe our synthetic dataset generation tool that enables scalable collection of such a synthetic dataset with realistic adversarial examples. We use the CARLA simulator to collect such a dataset and demonstrate simulated attacks that undergo the same environmental transforms and processing as real-world images. Our tools have been used to collect datasets to help evaluate the efficacy of adversarial examples, and can be found at https://github.com/carla-simulator/carla/pull/4992.

摘要: 现有的对抗性实例研究集中在现有自然图像数据集上的数字插入扰动。这种对抗性示例的构建是不现实的，因为由于传感和环境影响，攻击者在现实世界中部署这样的攻击可能很困难，甚至不可能。为了更好地理解针对网络物理系统的敌意例子，我们建议通过模拟来近似真实世界。在这篇文章中，我们描述了我们的合成数据集生成工具，它能够通过现实的对抗性例子来实现对这样的合成数据集的可伸缩收集。我们使用CALA模拟器来收集这样的数据集，并演示模拟攻击，这些攻击经历了与真实世界图像相同的环境转换和处理。我们的工具已被用于收集数据集，以帮助评估对抗性例子的效果，可在https://github.com/carla-simulator/carla/pull/4992.上找到



## **34. Careful What You Wish For: on the Extraction of Adversarially Trained Models**

小心你想要的：关于敌对训练模型的提取 cs.LG

To be published in the proceedings of the 19th Annual International  Conference on Privacy, Security & Trust (PST 2022). The conference  proceedings will be included in IEEE Xplore as in previous editions of the  conference

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10561v1)

**Authors**: Kacem Khaled, Gabriela Nicolescu, Felipe Gohring de Magalhães

**Abstracts**: Recent attacks on Machine Learning (ML) models such as evasion attacks with adversarial examples and models stealing through extraction attacks pose several security and privacy threats. Prior work proposes to use adversarial training to secure models from adversarial examples that can evade the classification of a model and deteriorate its performance. However, this protection technique affects the model's decision boundary and its prediction probabilities, hence it might raise model privacy risks. In fact, a malicious user using only a query access to the prediction output of a model can extract it and obtain a high-accuracy and high-fidelity surrogate model. To have a greater extraction, these attacks leverage the prediction probabilities of the victim model. Indeed, all previous work on extraction attacks do not take into consideration the changes in the training process for security purposes. In this paper, we propose a framework to assess extraction attacks on adversarially trained models with vision datasets. To the best of our knowledge, our work is the first to perform such evaluation. Through an extensive empirical study, we demonstrate that adversarially trained models are more vulnerable to extraction attacks than models obtained under natural training circumstances. They can achieve up to $\times1.2$ higher accuracy and agreement with a fraction lower than $\times0.75$ of the queries. We additionally find that the adversarial robustness capability is transferable through extraction attacks, i.e., extracted Deep Neural Networks (DNNs) from robust models show an enhanced accuracy to adversarial examples compared to extracted DNNs from naturally trained (i.e. standard) models.

摘要: 最近对机器学习(ML)模型的攻击，如利用对抗性示例的逃避攻击和通过提取攻击窃取模型，构成了几种安全和隐私威胁。以前的工作建议使用对抗性训练来从对抗性示例中保护模型，这些示例可能会逃避模型的分类并降低其性能。然而，这种保护技术影响了模型的决策边界及其预测概率，因此可能会增加模型的隐私风险。事实上，恶意用户只使用对模型的预测输出的查询访问就可以提取它，并获得高精度和高保真的代理模型。为了进行更大的提取，这些攻击利用了受害者模型的预测概率。事实上，以前关于提取攻击的所有工作都没有考虑到出于安全目的在训练过程中的变化。在本文中，我们提出了一种评估提取攻击的框架，该框架使用视觉数据集来评估对反向训练模型的提取攻击。据我们所知，我们的工作是第一次进行这样的评估。通过大量的实证研究，我们证明了逆向训练的模型比自然训练环境下的模型更容易受到抽取攻击。它们可以实现高达$\x 1.2$的准确率和与低于$\x 0.75$的小部分查询的一致性。此外，我们还发现，对抗的稳健性能力可以通过抽取攻击来传递，即从健壮模型中提取的深度神经网络(DNN)与从自然训练(即标准)模型中提取的DNN相比，对对抗性实例的准确率更高。



## **35. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

三角攻击：一种查询高效的基于决策的对抗性攻击 cs.CV

Accepted by ECCV 2022, code is available at  https://github.com/xiaosen-wang/TA

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2112.06569v3)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples can naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on ImageNet dataset show that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further validate the applicability of TA on real-world API, i.e., Tencent Cloud API.

摘要: 基于决策的攻击将目标模型视为黑匣子，只访问硬预测标签，对现实世界的应用构成了严重威胁。最近已经做出了很大的努力来减少查询的数量；然而，现有的基于决策的攻击仍然需要数千个查询才能生成高质量的对抗性例子。在这项工作中，我们发现一个良性样本、当前和下一个对抗性样本可以自然地在子空间中为任何迭代攻击构造一个三角形。基于正弦定律，提出了一种新的三角形攻击算法(TA)，该算法利用任意三角形中长边总是与较大角相对的几何信息来优化扰动。然而，直接将这些信息应用于输入图像是无效的，因为它不能在高维空间中彻底探索输入样本的邻域。为了解决这个问题，由于这种几何性质的普遍性，TA优化了低频空间中的扰动，以实现有效的降维。在ImageNet数据集上的广泛评估表明，与现有的基于决策的攻击相比，TA在1000个查询中实现了更高的攻击成功率，并且在各种扰动预算下需要更少的查询来达到相同的攻击成功率。在如此高的效率下，我们进一步验证了TA在现实世界的API上的适用性，即腾讯云API。



## **36. Knowledge-enhanced Black-box Attacks for Recommendations**

用于推荐的知识增强型黑盒攻击 cs.LG

Accepted in the KDD'22

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10307v1)

**Authors**: Jingfan Chen, Wenqi Fan, Guanghui Zhu, Xiangyu Zhao, Chunfeng Yuan, Qing Li, Yihua Huang

**Abstracts**: Recent studies have shown that deep neural networks-based recommender systems are vulnerable to adversarial attacks, where attackers can inject carefully crafted fake user profiles (i.e., a set of items that fake users have interacted with) into a target recommender system to achieve malicious purposes, such as promote or demote a set of target items. Due to the security and privacy concerns, it is more practical to perform adversarial attacks under the black-box setting, where the architecture/parameters and training data of target systems cannot be easily accessed by attackers. However, generating high-quality fake user profiles under black-box setting is rather challenging with limited resources to target systems. To address this challenge, in this work, we introduce a novel strategy by leveraging items' attribute information (i.e., items' knowledge graph), which can be publicly accessible and provide rich auxiliary knowledge to enhance the generation of fake user profiles. More specifically, we propose a knowledge graph-enhanced black-box attacking framework (KGAttack) to effectively learn attacking policies through deep reinforcement learning techniques, in which knowledge graph is seamlessly integrated into hierarchical policy networks to generate fake user profiles for performing adversarial black-box attacks. Comprehensive experiments on various real-world datasets demonstrate the effectiveness of the proposed attacking framework under the black-box setting.

摘要: 最近的研究表明，基于深度神经网络的推荐系统容易受到敌意攻击，攻击者可以向目标推荐系统注入精心制作的虚假用户配置文件(即，虚假用户与之交互的一组项目)，以达到恶意目的，如升级或降级一组目标项目。由于安全和隐私方面的考虑，在目标系统的体系结构/参数和训练数据不容易被攻击者访问的黑盒环境下执行对抗性攻击更实用。然而，在目标系统资源有限的情况下，在黑盒环境下生成高质量的虚假用户配置文件是相当具有挑战性的。为了应对这一挑战，在本工作中，我们引入了一种新的策略，利用项目的属性信息(即项目的知识图)，这些信息可以公开访问，并提供丰富的辅助知识来增强虚假用户配置文件的生成。更具体地说，我们提出了一种知识图增强的黑盒攻击框架(KGAttack)，通过深度强化学习技术有效地学习攻击策略，该框架将知识图无缝地集成到分层策略网络中，生成用于执行对抗性黑盒攻击的虚假用户配置文件。在各种真实数据集上的综合实验证明了该攻击框架在黑盒环境下的有效性。



## **37. Image Generation Network for Covert Transmission in Online Social Network**

在线社交网络中用于隐蔽传输的图像生成网络 cs.CV

ACMMM2022 Poster

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10292v1)

**Authors**: Zhengxin You, Qichao Ying, Sheng Li, Zhenxing Qian, Xinpeng Zhang

**Abstracts**: Online social networks have stimulated communications over the Internet more than ever, making it possible for secret message transmission over such noisy channels. In this paper, we propose a Coverless Image Steganography Network, called CIS-Net, that synthesizes a high-quality image directly conditioned on the secret message to transfer. CIS-Net is composed of four modules, namely, the Generation, Adversarial, Extraction, and Noise Module. The receiver can extract the hidden message without any loss even the images have been distorted by JPEG compression attacks. To disguise the behaviour of steganography, we collected images in the context of profile photos and stickers and train our network accordingly. As such, the generated images are more inclined to escape from malicious detection and attack. The distinctions from previous image steganography methods are majorly the robustness and losslessness against diverse attacks. Experiments over diverse public datasets have manifested the superior ability of anti-steganalysis.

摘要: 在线社交网络比以往任何时候都更多地刺激了互联网上的交流，使得在这种嘈杂的渠道上传输秘密信息成为可能。在本文中，我们提出了一种无覆盖的图像隐写网络，称为CIS-Net，它直接根据秘密信息合成高质量的图像进行传输。该网络由四个模块组成，即生成模块、对抗性模块、抽取模块和噪声模块。即使图像被JPEG压缩攻击篡改了，接收者也可以无损地提取隐藏信息。为了掩盖隐写术的行为，我们收集了头像照片和贴纸背景下的图像，并对我们的网络进行了相应的培训。因此，生成的图像更容易逃脱恶意检测和攻击。与以往的图像隐写方法不同的是，该方法对各种攻击具有较强的稳健性和无损性能。在不同的公开数据集上的实验表明，该算法具有较好的抗隐写分析能力。



## **38. Switching One-Versus-the-Rest Loss to Increase the Margin of Logits for Adversarial Robustness**

切换一对一损失以增加对战健壮性的Logit裕度 cs.LG

20 pages, 16 figures

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10283v1)

**Authors**: Sekitoshi Kanai, Shin'ya Yamaguchi, Masanori Yamada, Hiroshi Takahashi, Yasutoshi Ida

**Abstracts**: Defending deep neural networks against adversarial examples is a key challenge for AI safety. To improve the robustness effectively, recent methods focus on important data points near the decision boundary in adversarial training. However, these methods are vulnerable to Auto-Attack, which is an ensemble of parameter-free attacks for reliable evaluation. In this paper, we experimentally investigate the causes of their vulnerability and find that existing methods reduce margins between logits for the true label and the other labels while keeping their gradient norms non-small values. Reduced margins and non-small gradient norms cause their vulnerability since the largest logit can be easily flipped by the perturbation. Our experiments also show that the histogram of the logit margins has two peaks, i.e., small and large logit margins. From the observations, we propose switching one-versus-the-rest loss (SOVR), which uses one-versus-the-rest loss when data have small logit margins so that it increases the margins. We find that SOVR increases logit margins more than existing methods while keeping gradient norms small and outperforms them in terms of the robustness against Auto-Attack.

摘要: 防御深层神经网络以抵御敌意示例是人工智能安全的关键挑战。在对抗性训练中，为了有效地提高鲁棒性，目前的方法主要集中在决策边界附近的重要数据点。然而，这些方法容易受到自动攻击的攻击，自动攻击是用于可靠评估的无参数攻击的集合。在实验中，我们调查了它们易受攻击的原因，发现现有的方法在保持它们的梯度范数非小值的同时，减少了真实标签和其他标签的对数之间的差值。边际减小和非小梯度范数导致了它们的脆弱性，因为最大的logit很容易被扰动翻转。我们的实验还表明，Logit边缘的直方图有两个峰值，即小的和大的Logit边缘。根据观察，我们建议转换一对其余损失(SOVR)，即当数据具有较小的Logit边际时使用一对其余损失，从而增加边际。我们发现，SOVR在保持小的梯度范数的同时，比现有的方法更能提高Logit裕度，并且在抵抗自动攻击方面优于现有的方法。



## **39. FOCUS: Fairness via Agent-Awareness for Federated Learning on Heterogeneous Data**

焦点：异类数据联合学习中基于代理感知的公平性 cs.LG

**SubmitDate**: 2022-07-21    [paper-pdf](http://arxiv.org/pdf/2207.10265v1)

**Authors**: Wenda Chu, Chulin Xie, Boxin Wang, Linyi Li, Lang Yin, Han Zhao, Bo Li

**Abstracts**: Federated learning (FL) provides an effective paradigm to train machine learning models over distributed data with privacy protection. However, recent studies show that FL is subject to various security, privacy, and fairness threats due to the potentially malicious and heterogeneous local agents. For instance, it is vulnerable to local adversarial agents who only contribute low-quality data, with the goal of harming the performance of those with high-quality data. This kind of attack hence breaks existing definitions of fairness in FL that mainly focus on a certain notion of performance parity. In this work, we aim to address this limitation and propose a formal definition of fairness via agent-awareness for FL (FAA), which takes the heterogeneous data contributions of local agents into account. In addition, we propose a fair FL training algorithm based on agent clustering (FOCUS) to achieve FAA. Theoretically, we prove the convergence and optimality of FOCUS under mild conditions for linear models and general convex loss functions with bounded smoothness. We also prove that FOCUS always achieves higher fairness measured by FAA compared with standard FedAvg protocol under both linear models and general convex loss functions. Empirically, we evaluate FOCUS on four datasets, including synthetic data, images, and texts under different settings, and we show that FOCUS achieves significantly higher fairness based on FAA while maintaining similar or even higher prediction accuracy compared with FedAvg.

摘要: 联合学习(FL)提供了一种有效的范例来训练具有隐私保护的分布式数据上的机器学习模型。然而，最近的研究表明，由于潜在的恶意和异构性的本地代理，FL受到各种安全、隐私和公平的威胁。例如，它很容易受到当地对手代理人的攻击，这些代理人只提供低质量的数据，目的是损害那些拥有高质量数据的人的表现。因此，这种攻击打破了外语教学中对公平的现有定义，这些定义主要集中在某个绩效平等的概念上。在这项工作中，我们旨在解决这一局限性，并提出了一种基于代理感知的公平的形式化定义(FAA)，该定义考虑了本地代理的异质数据贡献。此外，我们还提出了一种基于主体聚类的公平FL训练算法(FOCUS)来实现FAA。从理论上证明了线性模型和具有有界光滑性的一般凸损失函数在较温和的条件下焦点的收敛和最优性。证明了无论是在线性模型下还是在一般的凸损失函数下，Focus协议都比标准FedAvg协议获得了更高的公平性。实验结果表明，基于FAA的Focus算法在保持与FedAvg相似甚至更高的预测精度的同时，获得了显著更高的公平性。



## **40. Illusionary Attacks on Sequential Decision Makers and Countermeasures**

对序贯决策者的幻觉攻击及其对策 cs.AI

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.10170v1)

**Authors**: Tim Franzmeyer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstracts**: Autonomous intelligent agents deployed to the real-world need to be robust against adversarial attacks on sensory inputs. Existing work in reinforcement learning focuses on minimum-norm perturbation attacks, which were originally introduced to mimic a notion of perceptual invariance in computer vision. In this paper, we note that such minimum-norm perturbation attacks can be trivially detected by victim agents, as these result in observation sequences that are not consistent with the victim agent's actions. Furthermore, many real-world agents, such as physical robots, commonly operate under human supervisors, which are not susceptible to such perturbation attacks. As a result, we propose to instead focus on illusionary attacks, a novel form of attack that is consistent with the world model of the victim agent. We provide a formal definition of this novel attack framework, explore its characteristics under a variety of conditions, and conclude that agents must seek realism feedback to be robust to illusionary attacks.

摘要: 部署在现实世界中的自主智能代理需要对感官输入的敌意攻击具有健壮性。强化学习的现有工作集中在最小范数扰动攻击上，最初引入最小范数扰动攻击是为了模仿计算机视觉中的感知不变性的概念。在本文中，我们注意到这种最小范数扰动攻击可以被受害者代理检测到，因为这些攻击导致的观察序列与受害者代理的行为不一致。此外，许多真实世界的代理，如物理机器人，通常在人类监督下操作，而人类监督不容易受到此类扰动攻击。因此，我们建议转而专注于幻觉攻击，这是一种与受害者代理的世界模型一致的新型攻击形式。我们给出了这种新的攻击框架的形式化定义，探讨了它在各种条件下的特征，并得出结论：代理必须寻求现实主义反馈才能对虚幻攻击具有健壮性。



## **41. PFMC: a parallel symbolic model checker for security protocol verification**

PFMC：一种用于安全协议验证的并行符号模型检查器 cs.LO

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09895v1)

**Authors**: Alex James, Alwen Tiu, Nisansala Yatapanage

**Abstracts**: We present an investigation into the design and implementation of a parallel model checker for security protocol verification that is based on a symbolic model of the adversary, where instantiations of concrete terms and messages are avoided until needed to resolve a particular assertion. We propose to build on this naturally lazy approach to parallelise this symbolic state exploration and evaluation. We utilise the concept of strategies in Haskell, which abstracts away from the low-level details of thread management and modularly adds parallel evaluation strategies (encapsulated as a monad in Haskell). We build on an existing symbolic model checker, OFMC, which is already implemented in Haskell. We show that there is a very significant speed up of around 3-5 times improvement when moving from the original single-threaded implementation of OFMC to our multi-threaded version, for both the Dolev-Yao attacker model and more general algebraic attacker models. We identify several issues in parallelising the model checker: among others, controlling growth of memory consumption, balancing lazy vs strict evaluation, and achieving an optimal granularity of parallelism.

摘要: 本文研究了一种用于安全协议验证的并行模型检查器的设计和实现，该模型检查器基于敌手的符号模型，其中避免实例化具体的术语和消息，直到需要解决特定断言。我们建议基于这种天生懒惰的方法来并行化这种象征性的状态探索和评估。我们利用了Haskell中的策略概念，它从线程管理的底层细节中抽象出来，并模块化地添加了并行计算策略(在Haskell中封装为Monad)。我们构建在现有的符号模型检查器OFMC上，该检查器已经在Haskell中实现。我们表明，从最初的单线程OFMC实现转移到我们的多线程版本时，无论是对于Dolev-姚攻击者模型还是更一般的代数攻击者模型，都有非常显著的速度提升约3-5倍。我们确定了模型检查器并行化中的几个问题：控制内存消耗的增长，平衡懒惰和严格计算，以及实现最优的并行粒度。



## **42. Adaptive Image Transformations for Transfer-based Adversarial Attack**

基于传输的对抗性攻击中的自适应图像变换 cs.CV

34 pages, 7 figures, 11 tables. Accepted by ECCV2022

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2111.13844v3)

**Authors**: Zheng Yuan, Jie Zhang, Shiguang Shan

**Abstracts**: Adversarial attacks provide a good way to study the robustness of deep learning models. One category of methods in transfer-based black-box attack utilizes several image transformation operations to improve the transferability of adversarial examples, which is effective, but fails to take the specific characteristic of the input image into consideration. In this work, we propose a novel architecture, called Adaptive Image Transformation Learner (AITL), which incorporates different image transformation operations into a unified framework to further improve the transferability of adversarial examples. Unlike the fixed combinational transformations used in existing works, our elaborately designed transformation learner adaptively selects the most effective combination of image transformations specific to the input image. Extensive experiments on ImageNet demonstrate that our method significantly improves the attack success rates on both normally trained models and defense models under various settings.

摘要: 对抗性攻击为研究深度学习模型的稳健性提供了一种很好的方法。一类基于转移的黑盒攻击方法利用多幅图像变换操作来提高对抗性样本的可转移性，这种方法是有效的，但没有考虑到输入图像的具体特征。在这项工作中，我们提出了一种新的体系结构，称为自适应图像变换学习器(AITL)，它将不同的图像变换操作整合到一个统一的框架中，以进一步提高对抗性例子的可转移性。与现有工作中使用的固定组合变换不同，我们精心设计的变换学习器自适应地选择特定于输入图像的最有效的图像变换组合。在ImageNet上的大量实验表明，该方法在正常训练模型和防御模型上的攻击成功率在各种设置下都有显著提高。



## **43. On the Robustness of Quality Measures for GANs**

论GAN质量度量的稳健性 cs.LG

Accepted at the European Conference in Computer Vision (ECCV 2022)

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2201.13019v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Anna Frühstück, Philip H. S. Torr, Peter Wonka, Bernard Ghanem

**Abstracts**: This work evaluates the robustness of quality measures of generative models such as Inception Score (IS) and Fr\'echet Inception Distance (FID). Analogous to the vulnerability of deep models against a variety of adversarial attacks, we show that such metrics can also be manipulated by additive pixel perturbations. Our experiments indicate that one can generate a distribution of images with very high scores but low perceptual quality. Conversely, one can optimize for small imperceptible perturbations that, when added to real world images, deteriorate their scores. We further extend our evaluation to generative models themselves, including the state of the art network StyleGANv2. We show the vulnerability of both the generative model and the FID against additive perturbations in the latent space. Finally, we show that the FID can be robustified by simply replacing the standard Inception with a robust Inception. We validate the effectiveness of the robustified metric through extensive experiments, showing it is more robust against manipulation.

摘要: 该工作评估了产生式模型的质量度量的稳健性，如初始得分(IS)和Fr回声初始距离(FID)。类似于深度模型对各种对抗性攻击的脆弱性，我们证明了这样的度量也可以被加性像素扰动所操纵。我们的实验表明，我们可以生成得分很高但感知质量较低的图像分布。相反，人们可以针对微小的不可察觉的扰动进行优化，当这些扰动添加到现实世界的图像中时，会降低他们的得分。我们进一步将我们的评估扩展到生成性模型本身，包括最先进的网络StyleGANv2。我们证明了生成模型和FID在潜在空间中对加性扰动的脆弱性。最后，我们展示了FID可以通过简单地用健壮的初始替换标准的初始来增强。我们通过大量的实验验证了鲁棒性度量的有效性，表明它对操纵具有更强的健壮性。



## **44. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

偏距离相关在深度学习中的广泛应用 cs.CV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2207.09684v1)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstracts**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **45. Detecting Textual Adversarial Examples through Randomized Substitution and Vote**

基于随机化替换和投票的文本对抗性实例检测 cs.CL

Accepted by UAI 2022, code is avaliable at  https://github.com/JHL-HUST/RSV

**SubmitDate**: 2022-07-20    [paper-pdf](http://arxiv.org/pdf/2109.05698v2)

**Authors**: Xiaosen Wang, Yifeng Xiong, Kun He

**Abstracts**: A line of work has shown that natural text processing models are vulnerable to adversarial examples. Correspondingly, various defense methods are proposed to mitigate the threat of textual adversarial examples, eg, adversarial training, input transformations, detection, etc. In this work, we treat the optimization process for synonym substitution based textual adversarial attacks as a specific sequence of word replacement, in which each word mutually influences other words. We identify that we could destroy such mutual interaction and eliminate the adversarial perturbation by randomly substituting a word with its synonyms. Based on this observation, we propose a novel textual adversarial example detection method, termed Randomized Substitution and Vote (RS&V), which votes the prediction label by accumulating the logits of k samples generated by randomly substituting the words in the input text with synonyms. The proposed RS&V is generally applicable to any existing neural networks without modification on the architecture or extra training, and it is orthogonal to prior work on making the classification network itself more robust. Empirical evaluations on three benchmark datasets demonstrate that our RS&V could detect the textual adversarial examples more successfully than the existing detection methods while maintaining the high classification accuracy on benign samples.

摘要: 一系列研究表明，自然文本处理模型很容易受到敌意例子的影响。相应地，人们提出了各种防御方法来缓解文本对抗性实例的威胁，如对抗性训练、输入转换、检测等。在本文中，我们将基于同义词替换的文本对抗性攻击的优化过程视为一个特定的单词替换序列，其中每个单词都会影响其他单词。我们发现，通过随机地用一个词的同义词替换一个词，我们可以破坏这种相互作用，并消除对抗性扰动。基于这一观察结果，我们提出了一种新的文本对抗性实例检测方法，称为随机替换和投票(RS&V)，该方法通过累加k个样本的逻辑来投票预测标签，所述k个样本是通过随机地将输入文本中的单词替换为同义词而产生的。所提出的RS&V算法一般适用于任何现有的神经网络，而不需要修改结构或进行额外的训练，并且它与先前的使分类网络本身更健壮的工作是正交的。在三个基准数据集上的实验结果表明，与现有的检测方法相比，本文的RS&V方法能够更成功地检测出文本中的敌意实例，同时保持了对良性样本的高分类准确率。



## **46. Diversified Adversarial Attacks based on Conjugate Gradient Method**

基于共轭梯度法的多样化对抗性攻击 cs.LG

Proceedings of the 39th International Conference on Machine Learning  (ICML 2022)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2206.09628v2)

**Authors**: Keiichiro Yamamura, Haruki Sato, Nariaki Tateiwa, Nozomi Hata, Toru Mitsutake, Issa Oe, Hiroki Ishikura, Katsuki Fujisawa

**Abstracts**: Deep learning models are vulnerable to adversarial examples, and adversarial attacks used to generate such examples have attracted considerable research interest. Although existing methods based on the steepest descent have achieved high attack success rates, ill-conditioned problems occasionally reduce their performance. To address this limitation, we utilize the conjugate gradient (CG) method, which is effective for this type of problem, and propose a novel attack algorithm inspired by the CG method, named the Auto Conjugate Gradient (ACG) attack. The results of large-scale evaluation experiments conducted on the latest robust models show that, for most models, ACG was able to find more adversarial examples with fewer iterations than the existing SOTA algorithm Auto-PGD (APGD). We investigated the difference in search performance between ACG and APGD in terms of diversification and intensification, and define a measure called Diversity Index (DI) to quantify the degree of diversity. From the analysis of the diversity using this index, we show that the more diverse search of the proposed method remarkably improves its attack success rate.

摘要: 深度学习模型容易受到对抗性实例的影响，而用于生成此类实例的对抗性攻击已经引起了相当大的研究兴趣。虽然现有的基于最陡下降的方法已经取得了很高的攻击成功率，但条件恶劣的问题有时会降低它们的性能。针对这一局限性，我们利用对这类问题有效的共轭梯度(CG)方法，并在CG方法的启发下提出了一种新的攻击算法，称为自动共轭梯度(ACG)攻击。在最新的稳健模型上进行的大规模评估实验结果表明，对于大多数模型，ACG能够以更少的迭代发现更多的对抗性实例，而不是现有的SOTA算法Auto-PGD(APGD)。我们研究了ACG和APGD在多样化和集约化方面的搜索性能差异，并定义了一个称为多样性指数(DI)的度量来量化多样性程度。从该指标的多样性分析可以看出，该方法搜索的多样性显著提高了其攻击成功率。



## **47. Towards Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

走向稳健的多变量时间序列预测：对抗性攻击和防御机制 cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09572v1)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstracts**: As deep learning models have gradually become the main workhorse of time series forecasting, the potential vulnerability under adversarial attacks to forecasting and decision system accordingly has emerged as a main issue in recent years. Albeit such behaviors and defense mechanisms started to be investigated for the univariate time series forecasting, there are still few studies regarding the multivariate forecasting which is often preferred due to its capacity to encode correlations between different time series. In this work, we study and design adversarial attack on multivariate probabilistic forecasting models, taking into consideration attack budget constraints and the correlation architecture between multiple time series. Specifically, we investigate a sparse indirect attack that hurts the prediction of an item (time series) by only attacking the history of a small number of other items to save attacking cost. In order to combat these attacks, we also develop two defense strategies. First, we adopt randomized smoothing to multivariate time series scenario and verify its effectiveness via empirical experiments. Second, we leverage a sparse attacker to enable end-to-end adversarial training that delivers robust probabilistic forecasters. Extensive experiments on real dataset confirm that our attack schemes are powerful and our defend algorithms are more effective compared with other baseline defense mechanisms.

摘要: 随着深度学习模型逐渐成为时间序列预测的主要工具，预测和决策系统在敌意攻击下的潜在脆弱性也成为近年来的主要问题。虽然单变量时间序列预测的这种行为和防御机制已经开始被研究，但关于多变量预测的研究仍然很少，因为多变量预测往往因为能够编码不同时间序列之间的相关性而受到青睐。在这项工作中，我们研究和设计了基于多变量概率预测模型的对抗性攻击，考虑了攻击预算约束和多个时间序列之间的关联结构。具体地说，我们调查了一种稀疏的间接攻击，该攻击通过仅攻击少量其他项的历史来节省攻击成本，从而损害了一项(时间序列)的预测。为了对抗这些攻击，我们还制定了两种防御策略。首先，将随机平滑方法应用于多变量时间序列情景，并通过实证实验验证其有效性。其次，我们利用稀疏攻击者实现端到端的对抗性训练，从而提供强大的概率预测者。在真实数据集上的大量实验证实了我们的攻击方案是强大的，与其他基线防御机制相比，我们的防御算法更有效。



## **48. Increasing the Cost of Model Extraction with Calibrated Proof of Work**

使用校准的工作证明增加模型提取的成本 cs.CR

Published as a conference paper at ICLR 2022 (Spotlight - 5% of  submitted papers)

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2201.09243v2)

**Authors**: Adam Dziedzic, Muhammad Ahmad Kaleem, Yu Shen Lu, Nicolas Papernot

**Abstracts**: In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen.

摘要: 在模型提取攻击中，攻击者可以通过反复查询通过公共API暴露的机器学习模型，并根据获得的预测调整自己的模型，从而窃取该模型。为了防止模型窃取，现有的防御措施侧重于检测恶意查询、截断或扭曲输出，因此必然会在健壮性和模型实用程序之间为合法用户带来折衷。相反，我们建议通过要求用户在阅读模型预测之前完成工作证明来阻碍模型提取。这大大增加了(甚至高达100倍)利用查询访问进行模型提取所需的计算工作量，从而阻止了攻击者。由于我们对完成每个查询的工作证明所需的工作量进行了校准，因此这只会给普通用户带来很小的开销(最高可达2倍)。为了实现这一点，我们的校准应用了来自差异隐私的工具来衡量查询所揭示的信息。我们的方法不需要修改受害者模型，并且可以被机器学习从业者应用，以保护他们公开曝光的模型不会轻易被窃取。



## **49. Assaying Out-Of-Distribution Generalization in Transfer Learning**

迁移学习中的分布外泛化分析 cs.LG

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09239v1)

**Authors**: Florian Wenzel, Andrea Dittadi, Peter Vincent Gehler, Carl-Johann Simon-Gabriel, Max Horn, Dominik Zietlow, David Kernert, Chris Russell, Thomas Brox, Bernt Schiele, Bernhard Schölkopf, Francesco Locatello

**Abstracts**: Since out-of-distribution generalization is a generally ill-posed problem, various proxy targets (e.g., calibration, adversarial robustness, algorithmic corruptions, invariance across shifts) were studied across different research programs resulting in different recommendations. While sharing the same aspirational goal, these approaches have never been tested under the same experimental conditions on real data. In this paper, we take a unified view of previous work, highlighting message discrepancies that we address empirically, and providing recommendations on how to measure the robustness of a model and how to improve it. To this end, we collect 172 publicly available dataset pairs for training and out-of-distribution evaluation of accuracy, calibration error, adversarial attacks, environment invariance, and synthetic corruptions. We fine-tune over 31k networks, from nine different architectures in the many- and few-shot setting. Our findings confirm that in- and out-of-distribution accuracies tend to increase jointly, but show that their relation is largely dataset-dependent, and in general more nuanced and more complex than posited by previous, smaller scale studies.

摘要: 由于分布外泛化是一个通常不适定的问题，因此对不同的代理目标(例如，校准、对手健壮性、算法腐败、跨班次不变性)进行了研究，得出了不同的建议。虽然有着相同的理想目标，但这些方法从未在相同的实验条件下对真实数据进行过测试。在本文中，我们对以前的工作进行了统一的审查，强调了我们通过经验解决的信息差异，并就如何衡量模型的稳健性以及如何改进模型提供了建议。为此，我们收集了172个公开可用的数据集对，用于训练和分布外评估准确性、校准误差、对抗性攻击、环境不变性和合成腐败。我们微调了超过31k的网络，这些网络来自9种不同的架构，在多发和少发的情况下。我们的发现证实，分布内和分布外的精度往往会共同增加，但表明它们的关系在很大程度上依赖于数据集，总体上比之前的较小规模的研究假设的更细微和更复杂。



## **50. MUD-PQFed: Towards Malicious User Detection in Privacy-Preserving Quantized Federated Learning**

MUD-PQFed：隐私保护量化联合学习中的恶意用户检测 cs.CR

13 pages,13 figures

**SubmitDate**: 2022-07-19    [paper-pdf](http://arxiv.org/pdf/2207.09080v1)

**Authors**: Hua Ma, Qun Li, Yifeng Zheng, Zhi Zhang, Xiaoning Liu, Yansong Gao, Said F. Al-Sarawi, Derek Abbott

**Abstracts**: Federated Learning (FL), a distributed machine learning paradigm, has been adapted to mitigate privacy concerns for customers. Despite their appeal, there are various inference attacks that can exploit shared-plaintext model updates to embed traces of customer private information, leading to serious privacy concerns. To alleviate this privacy issue, cryptographic techniques such as Secure Multi-Party Computation and Homomorphic Encryption have been used for privacy-preserving FL. However, such security issues in privacy-preserving FL are poorly elucidated and underexplored. This work is the first attempt to elucidate the triviality of performing model corruption attacks on privacy-preserving FL based on lightweight secret sharing. We consider scenarios in which model updates are quantized to reduce communication overhead in this case, where an adversary can simply provide local parameters outside the legal range to corrupt the model. We then propose the MUD-PQFed protocol, which can precisely detect malicious clients performing attacks and enforce fair penalties. By removing the contributions of detected malicious clients, the global model utility is preserved to be comparable to the baseline global model without the attack. Extensive experiments validate effectiveness in maintaining baseline accuracy and detecting malicious clients in a fine-grained manner

摘要: 联邦学习(FL)是一种分布式机器学习范式，已被用于缓解客户的隐私担忧。尽管有吸引力，但仍有各种推理攻击可以利用共享明文模型更新来嵌入客户私人信息的痕迹，从而导致严重的隐私问题。为了缓解这一隐私问题，安全多方计算和同态加密等密码技术被用于隐私保护FL。然而，在保护隐私的FL中，这样的安全问题还没有得到很好的阐述和探讨。这项工作是首次尝试阐明基于轻量级秘密共享对隐私保护FL执行模型腐败攻击的琐碎之处。在这种情况下，我们考虑对模型更新进行量化以减少通信开销的场景，其中对手只需提供合法范围之外的本地参数即可破坏模型。然后，我们提出了MUD-PQFed协议，该协议能够准确地检测执行攻击的恶意客户端，并执行公平的惩罚。通过去除检测到的恶意客户端的贡献，全局模型实用程序被保留为与没有攻击的基准全局模型相当。大量实验验证了在保持基线准确性和细粒度检测恶意客户端方面的有效性



