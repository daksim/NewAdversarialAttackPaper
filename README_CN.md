# Latest Adversarial Attack Papers
**update at 2023-04-24 11:02:46**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Individual Fairness in Bayesian Neural Networks**

贝叶斯神经网络中的个体公平性 cs.LG

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10828v1) [paper-pdf](http://arxiv.org/pdf/2304.10828v1)

**Authors**: Alice Doherty, Matthew Wicker, Luca Laurenti, Andrea Patane

**Abstract**: We study Individual Fairness (IF) for Bayesian neural networks (BNNs). Specifically, we consider the $\epsilon$-$\delta$-individual fairness notion, which requires that, for any pair of input points that are $\epsilon$-similar according to a given similarity metrics, the output of the BNN is within a given tolerance $\delta>0.$ We leverage bounds on statistical sampling over the input space and the relationship between adversarial robustness and individual fairness to derive a framework for the systematic estimation of $\epsilon$-$\delta$-IF, designing Fair-FGSM and Fair-PGD as global,fairness-aware extensions to gradient-based attacks for BNNs. We empirically study IF of a variety of approximately inferred BNNs with different architectures on fairness benchmarks, and compare against deterministic models learnt using frequentist techniques. Interestingly, we find that BNNs trained by means of approximate Bayesian inference consistently tend to be markedly more individually fair than their deterministic counterparts.

摘要: 研究了贝叶斯神经网络的个体公平性。具体地说，我们考虑了个体公平性的概念，它要求对于根据给定的相似性度量是$\epon$-相似的任何一对输入点，BNN的输出都在给定的容差$\Delta>0的范围内。$我们利用输入空间上的统计抽样的界以及对手健壮性和个体公平性之间的关系来推导出系统估计$-$\Delta$-IF的框架，并将公平-FGSM和公平-PGD设计为基于梯度攻击的全局公平感知扩展。我们对各种不同结构的近似推断BNN的公平性基准进行了实证研究，并与使用频域技术学习的确定性模型进行了比较。有趣的是，我们发现通过近似贝叶斯推理训练的BNN一致倾向于比它们的确定性对应物更个别地更公平。



## **2. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

可靠的表示使防御者更强大：健壮GNN的无监督结构求精 cs.LG

Accepted in KDD2022

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2207.00012v4) [paper-pdf](http://arxiv.org/pdf/2207.00012v4)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstract**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.

摘要: 得益于消息传递机制，图神经网络(GNN)已经成功地处理了大量的图数据任务。然而，最近的研究表明，攻击者可以通过恶意修改图结构来灾难性地降低GNN的性能。解决这一问题的一个直接解决方案是通过学习两个末端节点的成对表示之间的度量函数来对边权重进行建模，该度量函数试图为对抗性边分配较低的权重。现有的方法要么使用原始特征，要么使用由监督GNN学习的表示来对边权重进行建模。然而，这两种策略都面临着一些迫在眉睫的问题：原始特征不能表示节点的各种属性(例如结构信息)，而有监督GNN学习的表示可能会受到有毒图上分类器性能较差的影响。我们需要既携带特征信息又尽可能正确的结构信息并对结构扰动不敏感的表示法。为此，我们提出了一种名为STRATE的无监督流水线来优化图的结构。最后，我们将精化后的图输入到下游分类器中。对于这一部分，我们设计了一种改进的GCN，它在不增加时间复杂度的情况下显著增强了普通GCN的健壮性。在四个真实图形基准上的大量实验表明，STRATE的性能优于最先进的方法，并成功地防御了各种攻击。



## **3. Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning**

拒绝服务或细粒度控制：对联邦学习的灵活模型中毒攻击 cs.LG

This paper has been accepted by the 32st International Joint  Conference on Artificial Intelligence (IJCAI-23, Main Track)

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10783v1) [paper-pdf](http://arxiv.org/pdf/2304.10783v1)

**Authors**: Hangtao Zhang, Zeming Yao, Leo Yu Zhang, Shengshan Hu, Chao Chen, Alan Liew, Zhetao Li

**Abstract**: Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed with precise control, malicious FL service providers can gain advantages over their competitors without getting noticed, hence opening a new attack surface in FL other than DoS. Even for the purpose of DoS, experiments show that FMPA significantly decreases the global accuracy, outperforming six state-of-the-art attacks.

摘要: 联合学习(FL)容易受到中毒攻击，攻击者破坏全局聚合结果并导致拒绝服务(DoS)。与目前的模型中毒攻击不同的是，我们提出了一种灵活的模型中毒攻击(FMPA)，它可以实现多种攻击目标。我们考虑了一个实际的威胁场景，其中没有关于FL系统的额外知识(例如，聚合规则或良性设备上的更新)可供攻击者使用。FMPA利用全球历史信息来构建一个估计器，该估计器预测下一轮全球模型作为良性参考。然后，它微调参考模型，以获得所需的低精度和小扰动的中毒模型。除了造成DoS的目标外，FMPA还可以自然地扩展到发起细粒度的可控攻击，从而有可能精准地降低全局精度。拥有精确控制的恶意FL服务提供商可以在不被注意的情况下获得相对于竞争对手的优势，从而打开了FL除DoS之外的新攻击面。实验表明，即使是在DoS攻击的情况下，FMPA也显著降低了全局的准确率，性能超过了六种最先进的攻击。



## **4. Interpretable and Robust AI in EEG Systems: A Survey**

可解释和稳健的人工智能在脑电系统中的研究进展 eess.SP

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10755v1) [paper-pdf](http://arxiv.org/pdf/2304.10755v1)

**Authors**: Xinliang Zhou, Chenyu Liu, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.

摘要: 人工智能(AI)和脑电(EEG)的紧密结合在AI时代极大地推动了人机交互(HCI)技术的进步。与传统的脑电系统不同，基于人工智能的脑电系统的可解释性和稳健性变得尤为关键。这种可解释性阐明了人工智能模型的内部工作机制，从而可以赢得用户的信任。健壮性反映了人工智能对攻击和扰动的可靠性，这对于敏感和脆弱的脑电信号是必不可少的。因此，人工智能在脑电系统中的可解释性和稳健性受到越来越多的关注，近年来其研究取得了很大进展。然而，目前还没有关于这一领域最新进展的调查。在本文中，我们介绍了第一个全面的综述，并总结了可解释的和健壮的脑电系统人工智能技术。具体地说，我们首先提出了一种可解释性的分类，将其描述为三种类型：反向传播方法、扰动方法和内在可解释方法。然后，我们将健壮性机制分为四类：噪声和伪影、人类可变性、数据获取不稳定性和对抗性攻击。最后，我们确定了脑电系统中可解释的和健壮的人工智能面临的几个关键和尚未解决的挑战，并进一步讨论了它们的未来发展方向。



## **5. Fooling Thermal Infrared Detectors in Physical World**

在现实世界中愚弄热红外探测器 cs.CV

**SubmitDate**: 2023-04-21    [abs](http://arxiv.org/abs/2304.10712v1) [paper-pdf](http://arxiv.org/pdf/2304.10712v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Infrared imaging systems have a vast array of potential applications in pedestrian detection and autonomous driving, and their safety performance is of great concern. However, few studies have explored the safety of infrared imaging systems in real-world settings. Previous research has used physical perturbations such as small bulbs and thermal "QR codes" to attack infrared imaging detectors, but such methods are highly visible and lack stealthiness. Other researchers have used hot and cold blocks to deceive infrared imaging detectors, but this method is limited in its ability to execute attacks from various angles. To address these shortcomings, we propose a novel physical attack called adversarial infrared blocks (AdvIB). By optimizing the physical parameters of the adversarial infrared blocks, this method can execute a stealthy black-box attack on thermal imaging system from various angles. We evaluate the proposed method based on its effectiveness, stealthiness, and robustness. Our physical tests show that the proposed method achieves a success rate of over 80% under most distance and angle conditions, validating its effectiveness. For stealthiness, our method involves attaching the adversarial infrared block to the inside of clothing, enhancing its stealthiness. Additionally, we test the proposed method on advanced detectors, and experimental results demonstrate an average attack success rate of 51.2%, proving its robustness. Overall, our proposed AdvIB method offers a promising avenue for conducting stealthy, effective and robust black-box attacks on thermal imaging system, with potential implications for real-world safety and security applications.

摘要: 红外成像系统在行人检测和自动驾驶方面有着广泛的潜在应用，其安全性能备受关注。然而，很少有研究探讨红外成像系统在现实世界中的安全性。之前的研究曾使用小灯泡和热二维码等物理扰动来攻击红外成像探测器，但这种方法具有高度可见性和隐蔽性。其他研究人员也曾使用冷热块来欺骗红外成像探测器，但这种方法在从不同角度执行攻击的能力方面受到限制。为了克服这些缺陷，我们提出了一种新的物理攻击，称为对抗性红外线拦截(AdvIB)。该方法通过优化敌方红外块的物理参数，可以从多个角度对热成像系统进行隐身黑匣子攻击。我们从有效性、隐蔽性和稳健性三个方面对提出的方法进行了评估。我们的物理测试表明，该方法在大多数距离和角度条件下都达到了80%以上的成功率，验证了其有效性。对于隐蔽性，我们的方法是将敌对的红外线块安装在衣服的内部，增强其隐蔽性。此外，我们在高级检测器上测试了该方法，实验结果表明该方法的平均攻击成功率为51.2%，证明了该方法的健壮性。总体而言，我们提出的AdvIB方法为对热成像系统进行隐蔽、有效和健壮的黑盒攻击提供了一条很有前途的途径，对现实世界的安全和安保应用具有潜在的影响。



## **6. VenoMave: Targeted Poisoning Against Speech Recognition**

VenoMave：针对语音识别的定向中毒 cs.SD

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2010.10682v3) [paper-pdf](http://arxiv.org/pdf/2010.10682v3)

**Authors**: Hojjat Aghakhani, Lea Schönherr, Thorsten Eisenhofer, Dorothea Kolossa, Thorsten Holz, Christopher Kruegel, Giovanni Vigna

**Abstract**: Despite remarkable improvements, automatic speech recognition is susceptible to adversarial perturbations. Compared to standard machine learning architectures, these attacks are significantly more challenging, especially since the inputs to a speech recognition system are time series that contain both acoustic and linguistic properties of speech. Extracting all recognition-relevant information requires more complex pipelines and an ensemble of specialized components. Consequently, an attacker needs to consider the entire pipeline. In this paper, we present VENOMAVE, the first training-time poisoning attack against speech recognition. Similar to the predominantly studied evasion attacks, we pursue the same goal: leading the system to an incorrect and attacker-chosen transcription of a target audio waveform. In contrast to evasion attacks, however, we assume that the attacker can only manipulate a small part of the training data without altering the target audio waveform at runtime. We evaluate our attack on two datasets: TIDIGITS and Speech Commands. When poisoning less than 0.17% of the dataset, VENOMAVE achieves attack success rates of more than 80.0%, without access to the victim's network architecture or hyperparameters. In a more realistic scenario, when the target audio waveform is played over the air in different rooms, VENOMAVE maintains a success rate of up to 73.3%. Finally, VENOMAVE achieves an attack transferability rate of 36.4% between two different model architectures.

摘要: 尽管有了显著的改进，自动语音识别仍然容易受到对抗性干扰的影响。与标准的机器学习体系结构相比，这些攻击更具挑战性，特别是因为语音识别系统的输入是同时包含语音的声学和语言属性的时间序列。提取所有与识别相关的信息需要更复杂的管道和一系列专门的组件。因此，攻击者需要考虑整个管道。本文提出了第一个针对语音识别的训练时间中毒攻击方法VENOMAVE。与主要研究的规避攻击类似，我们追求相同的目标：将系统引导到攻击者选择的目标音频波形的错误转录。然而，与逃避攻击不同的是，我们假设攻击者只能操纵一小部分训练数据，而不会在运行时改变目标音频波形。我们在两个数据集：TIDIGITS和语音命令上评估了我们的攻击。当毒化少于0.17%的数据集时，VENOMAVE实现了80.0%以上的攻击成功率，而无需访问受害者的网络架构或超级参数。在更真实的场景中，当目标音频波形在不同房间的空中播放时，VENOMAVE保持高达73.3%的成功率。最后，VENOMAVE在两种不同的模型架构之间达到了36.4%的攻击可转移率。



## **7. Get Rid Of Your Trail: Remotely Erasing Backdoors in Federated Learning**

摆脱你的痕迹：远程清除联合学习中的后门 cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10638v1) [paper-pdf](http://arxiv.org/pdf/2304.10638v1)

**Authors**: Manaar Alam, Hithem Lamri, Michail Maniatakos

**Abstract**: Federated Learning (FL) enables collaborative deep learning training across multiple participants without exposing sensitive personal data. However, the distributed nature of FL and the unvetted participants' data makes it vulnerable to backdoor attacks. In these attacks, adversaries inject malicious functionality into the centralized model during training, leading to intentional misclassifications for specific adversary-chosen inputs. While previous research has demonstrated successful injections of persistent backdoors in FL, the persistence also poses a challenge, as their existence in the centralized model can prompt the central aggregation server to take preventive measures to penalize the adversaries. Therefore, this paper proposes a methodology that enables adversaries to effectively remove backdoors from the centralized model upon achieving their objectives or upon suspicion of possible detection. The proposed approach extends the concept of machine unlearning and presents strategies to preserve the performance of the centralized model and simultaneously prevent over-unlearning of information unrelated to backdoor patterns, making the adversaries stealthy while removing backdoors. To the best of our knowledge, this is the first work that explores machine unlearning in FL to remove backdoors to the benefit of adversaries. Exhaustive evaluation considering image classification scenarios demonstrates the efficacy of the proposed method in efficient backdoor removal from the centralized model, injected by state-of-the-art attacks across multiple configurations.

摘要: 联合学习(FL)支持多个参与者之间的协作深度学习培训，而不会暴露敏感的个人数据。然而，FL的分布式性质和未经审查的参与者的数据使其容易受到后门攻击。在这些攻击中，攻击者在训练期间向集中式模型注入恶意功能，导致故意对特定对手选择的输入进行错误分类。虽然之前的研究已经证明在FL中成功地注入了持久性后门，但持久性也构成了一个挑战，因为它们存在于集中模式中可以促使中央聚合服务器采取预防措施来惩罚对手。因此，本文提出了一种方法，使攻击者能够在达到他们的目标或怀疑可能被检测到时，有效地从集中式模型中删除后门。该方法扩展了机器遗忘的概念，并提出了保持集中式模型性能的策略，同时防止了与后门模式无关的信息的过度遗忘，使对手在移除后门的同时具有隐蔽性。据我们所知，这是第一个探索机器遗忘在FL中的工作，以消除后门，使对手受益。考虑图像分类场景的详尽评估证明了所提出的方法在从集中式模型中高效移除后门的有效性，该模型通过跨多个配置的最先进的攻击注入。



## **8. SoK: Let the Privacy Games Begin! A Unified Treatment of Data Inference Privacy in Machine Learning**

索克：让隐私游戏开始吧！机器学习中数据推理隐私的统一处理 cs.LG

20 pages, to appear in 2023 IEEE Symposium on Security and Privacy

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2212.10986v2) [paper-pdf](http://arxiv.org/pdf/2212.10986v2)

**Authors**: Ahmed Salem, Giovanni Cherubin, David Evans, Boris Köpf, Andrew Paverd, Anshuman Suri, Shruti Tople, Santiago Zanella-Béguelin

**Abstract**: Deploying machine learning models in production may allow adversaries to infer sensitive information about training data. There is a vast literature analyzing different types of inference risks, ranging from membership inference to reconstruction attacks. Inspired by the success of games (i.e., probabilistic experiments) to study security properties in cryptography, some authors describe privacy inference risks in machine learning using a similar game-based style. However, adversary capabilities and goals are often stated in subtly different ways from one presentation to the other, which makes it hard to relate and compose results. In this paper, we present a game-based framework to systematize the body of knowledge on privacy inference risks in machine learning. We use this framework to (1) provide a unifying structure for definitions of inference risks, (2) formally establish known relations among definitions, and (3) to uncover hitherto unknown relations that would have been difficult to spot otherwise.

摘要: 在生产中部署机器学习模型可能会允许攻击者推断有关训练数据的敏感信息。有大量的文献分析了不同类型的推理风险，从成员关系推理到重构攻击。受游戏(即概率实验)在密码学中研究安全属性的成功启发，一些作者使用类似的基于游戏的风格描述了机器学习中的隐私推理风险。然而，对手的能力和目标往往在不同的演示文稿中以微妙的不同方式表达，这使得很难联系和撰写结果。在本文中，我们提出了一个基于博弈的框架来系统化机器学习中隐私推理风险的知识体系。我们使用这个框架来(1)为推理风险的定义提供一个统一的结构，(2)正式地建立定义之间的已知关系，以及(3)揭示否则很难发现的迄今未知的关系。



## **9. More is Better (Mostly): On the Backdoor Attacks in Federated Graph Neural Networks**

越多越好(多数)：联邦图神经网络中的后门攻击 cs.CR

15 pages, 13 figures

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2202.03195v5) [paper-pdf](http://arxiv.org/pdf/2202.03195v5)

**Authors**: Jing Xu, Rui Wang, Stefanos Koffas, Kaitai Liang, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) are a class of deep learning-based methods for processing graph domain information. GNNs have recently become a widely used graph analysis method due to their superior ability to learn representations for complex graph data. However, due to privacy concerns and regulation restrictions, centralized GNNs can be difficult to apply to data-sensitive scenarios. Federated learning (FL) is an emerging technology developed for privacy-preserving settings when several parties need to train a shared global model collaboratively. Although several research works have applied FL to train GNNs (Federated GNNs), there is no research on their robustness to backdoor attacks.   This paper bridges this gap by conducting two types of backdoor attacks in Federated GNNs: centralized backdoor attacks (CBA) and distributed backdoor attacks (DBA). Our experiments show that the DBA attack success rate is higher than CBA in almost all evaluated cases. For CBA, the attack success rate of all local triggers is similar to the global trigger even if the training set of the adversarial party is embedded with the global trigger. To further explore the properties of two backdoor attacks in Federated GNNs, we evaluate the attack performance for a different number of clients, trigger sizes, poisoning intensities, and trigger densities. Moreover, we explore the robustness of DBA and CBA against one defense. We find that both attacks are robust against the investigated defense, necessitating the need to consider backdoor attacks in Federated GNNs as a novel threat that requires custom defenses.

摘要: 图神经网络是一类基于深度学习的图域信息处理方法。由于其优越的学习复杂图形数据表示的能力，GNN最近已成为一种广泛使用的图形分析方法。然而，由于隐私问题和监管限制，集中式GNN可能很难适用于数据敏感的情况。联合学习(FL)是一种新兴的技术，是为保护隐私而开发的，当多个参与方需要协作训练共享的全球模型时。虽然一些研究工作已经将FL用于训练GNN(Federated GNN)，但还没有关于其对后门攻击的健壮性的研究。本文通过在联邦GNN中实施两种类型的后门攻击来弥合这一差距：集中式后门攻击(CBA)和分布式后门攻击(DBA)。我们的实验表明，DBA攻击的成功率几乎在所有评估案例中都高于CBA。对于CBA，即使敌方的训练集嵌入了全局触发器，所有局部触发器的攻击成功率也与全局触发器相似。为了进一步探索联邦GNN中两种后门攻击的特性，我们评估了不同客户端数量、触发器大小、中毒强度和触发器密度下的攻击性能。此外，我们还探讨了DBA和CBA对一种防御的健壮性。我们发现，这两种攻击对所调查的防御都是健壮的，因此有必要将联邦GNN中的后门攻击视为一种需要自定义防御的新威胁。



## **10. Certified Adversarial Robustness Within Multiple Perturbation Bounds**

多扰动界下的认证对抗稳健性 cs.LG

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10446v1) [paper-pdf](http://arxiv.org/pdf/2304.10446v1)

**Authors**: Soumalya Nandi, Sravanti Addepalli, Harsh Rangwani, R. Venkatesh Babu

**Abstract**: Randomized smoothing (RS) is a well known certified defense against adversarial attacks, which creates a smoothed classifier by predicting the most likely class under random noise perturbations of inputs during inference. While initial work focused on robustness to $\ell_2$ norm perturbations using noise sampled from a Gaussian distribution, subsequent works have shown that different noise distributions can result in robustness to other $\ell_p$ norm bounds as well. In general, a specific noise distribution is optimal for defending against a given $\ell_p$ norm based attack. In this work, we aim to improve the certified adversarial robustness against multiple perturbation bounds simultaneously. Towards this, we firstly present a novel \textit{certification scheme}, that effectively combines the certificates obtained using different noise distributions to obtain optimal results against multiple perturbation bounds. We further propose a novel \textit{training noise distribution} along with a \textit{regularized training scheme} to improve the certification within both $\ell_1$ and $\ell_2$ perturbation norms simultaneously. Contrary to prior works, we compare the certified robustness of different training algorithms across the same natural (clean) accuracy, rather than across fixed noise levels used for training and certification. We also empirically invalidate the argument that training and certifying the classifier with the same amount of noise gives the best results. The proposed approach achieves improvements on the ACR (Average Certified Radius) metric across both $\ell_1$ and $\ell_2$ perturbation bounds.

摘要: 随机平滑(RS)是一种著名的对抗攻击的认证防御方法，它通过在推理过程中输入的随机噪声扰动下预测最可能的类别来创建平滑的分类器。虽然最初的工作集中于使用从高斯分布采样的噪声对$\ell_2$范数扰动的鲁棒性，但随后的工作表明，不同的噪声分布也可以导致对其他$\ell_p$范数界的鲁棒性。一般来说，特定的噪声分布对于防御给定的基于$\ell_p$范数的攻击是最优的。在这项工作中，我们的目标是同时提高认证对手对多个扰动界的稳健性。为此，我们首先提出了一种新的认证方案，该方案有效地结合了使用不同噪声分布获得的证书，从而在多个扰动界下获得最优结果。我们进一步提出了一种新的训练噪声分布和一种正则化训练方案，以同时改进在$1和$2扰动范数下的证明。与以前的工作相反，我们在相同的自然(干净)精度范围内比较不同训练算法的认证稳健性，而不是在用于训练和认证的固定噪声水平上进行比较。我们还从经验上证明了训练和认证具有相同噪声量的分类器可以得到最好的结果的论点。所提出的方法在$\ell_1$和$\ell_2$摄动界上实现了对ACR(平均认证半径)度量的改进。



## **11. An Analysis of the Completion Time of the BB84 Protocol**

对BB84议定书完成时间的分析 cs.PF

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10218v1) [paper-pdf](http://arxiv.org/pdf/2304.10218v1)

**Authors**: Sounak Kar, Jean-Yves Le Boudec

**Abstract**: The BB84 QKD protocol is based on the idea that the sender and the receiver can reconcile a certain fraction of the teleported qubits to detect eavesdropping or noise and decode the rest to use as a private key. Under the present hardware infrastructure, decoherence of quantum states poses a significant challenge to performing perfect or efficient teleportation, meaning that a teleportation-based protocol must be run multiple times to observe success. Thus, performance analyses of such protocols usually consider the completion time, i.e., the time until success, rather than the duration of a single attempt. Moreover, due to decoherence, the success of an attempt is in general dependent on the duration of individual phases of that attempt, as quantum states must wait in memory while the success or failure of a generation phase is communicated to the relevant parties. In this work, we do a performance analysis of the completion time of the BB84 protocol in a setting where the sender and the receiver are connected via a single quantum repeater and the only quantum channel between them does not see any adversarial attack. Assuming certain distributional forms for the generation and communication phases of teleportation, we provide a method to compute the MGF of the completion time and subsequently derive an estimate of the CDF and a bound on the tail probability. This result helps us gauge the (tail) behaviour of the completion time in terms of the parameters characterising the elementary phases of teleportation, without having to run the protocol multiple times. We also provide an efficient simulation scheme to generate the completion time, which relies on expressing the completion time in terms of aggregated teleportation times. We numerically compare our approach with a full-scale simulation and observe good agreement between them.

摘要: BB84量子密钥分发协议基于这样的思想，即发送者和接收者可以协调传送的量子比特的特定部分，以检测窃听或噪声，并解码其余的量子比特作为私钥。在目前的硬件基础设施下，量子态的退相干对执行完美或有效的隐形传态构成了巨大的挑战，这意味着基于隐形传态的协议必须多次运行才能观察到成功。因此，此类协议的性能分析通常考虑完成时间，即成功之前的时间，而不是单次尝试的持续时间。此外，由于退相干，尝试的成功通常取决于尝试的各个阶段的持续时间，因为当生成阶段的成功或失败被传递给相关方时，量子态必须在存储器中等待。在这项工作中，我们对BB84协议的完成时间进行了性能分析，在发送方和接收方通过单个量子中继器连接并且它们之间唯一的量子信道没有任何敌意攻击的情况下。假设隐形传态的产生和通信阶段有一定的分布形式，我们提供了一种计算完成时间的MGF的方法，并由此得到了CDF的估计和尾概率的界。这一结果帮助我们根据表征隐形传态的基本阶段的参数来衡量完成时间的(尾部)行为，而不必多次运行协议。我们还提供了一种高效的仿真方案来生成完成时间，该方案依赖于将完成时间表示为聚合隐形传态时间。我们将我们的方法与全尺寸模拟进行了数值比较，并观察到它们之间有很好的一致性。



## **12. Quantum-secure message authentication via blind-unforgeability**

基于盲不可伪造性的量子安全消息认证 quant-ph

37 pages, v4: Erratum added. We removed a result that had an error in  its proof

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/1803.03761v4) [paper-pdf](http://arxiv.org/pdf/1803.03761v4)

**Authors**: Gorjan Alagic, Christian Majenz, Alexander Russell, Fang Song

**Abstract**: Formulating and designing authentication of classical messages in the presence of adversaries with quantum query access has been a longstanding challenge, as the familiar classical notions of unforgeability do not directly translate into meaningful notions in the quantum setting. A particular difficulty is how to fairly capture the notion of "predicting an unqueried value" when the adversary can query in quantum superposition.   We propose a natural definition of unforgeability against quantum adversaries called blind unforgeability. This notion defines a function to be predictable if there exists an adversary who can use "partially blinded" oracle access to predict values in the blinded region. We support the proposal with a number of technical results. We begin by establishing that the notion coincides with EUF-CMA in the classical setting and go on to demonstrate that the notion is satisfied by a number of simple guiding examples, such as random functions and quantum-query-secure pseudorandom functions. We then show the suitability of blind unforgeability for supporting canonical constructions and reductions. We prove that the "hash-and-MAC" paradigm and the Lamport one-time digital signature scheme are indeed unforgeable according to the definition. To support our analysis, we additionally define and study a new variety of quantum-secure hash functions called Bernoulli-preserving.   Finally, we demonstrate that blind unforgeability is stronger than a previous definition of Boneh and Zhandry [EUROCRYPT '13, CRYPTO '13] in the sense that we can construct an explicit function family which is forgeable by an attack that is recognized by blind-unforgeability, yet satisfies the definition by Boneh and Zhandry.

摘要: 在具有量子查询访问的攻击者在场的情况下，制定和设计经典消息的认证一直是一个长期存在的挑战，因为熟悉的不可伪造性的经典概念不能直接转化为在量子环境中有意义的概念。一个特别的困难是，当对手可以在量子叠加中进行查询时，如何公平地捕捉到“预测一个未被质疑的值”的概念。针对量子对手，我们提出了不可伪造性的自然定义，称为盲不可伪造性。这个概念定义了一个函数是可预测的，如果存在一个对手，该对手可以使用“部分盲目的”先知访问来预测盲区中的值。我们用一系列技术成果支持这项提议。我们首先建立了这个概念在经典设置下与EUF-CMA重合，然后通过一些简单的指导性例子，例如随机函数和量子查询安全伪随机函数，证明了这个概念是满足的。然后，我们证明了盲不可伪造性对于支持规范构造和约简的适用性。根据定义，我们证明了Hash-and-MAC范例和Lamport一次性数字签名方案确实是不可伪造的。为了支持我们的分析，我们还定义和研究了一种新的量子安全散列函数，称为伯努利保持。最后，我们证明了盲不可伪造性比Boneh和Zhandry[Eurocrypt‘13，Crypto’13]的定义更强，因为我们可以构造一个显式函数族，该函数族可以被盲不可伪造性识别的攻击伪造，但满足Boneh和Zhandry的定义。



## **13. Diversifying the High-level Features for better Adversarial Transferability**

使高级功能多样化，以实现更好的对手可转换性 cs.CV

15 pages

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10136v1) [paper-pdf](http://arxiv.org/pdf/2304.10136v1)

**Authors**: Zhiyuan Wang, Zeliang Zhang, Siyuan Liang, Xiaosen Wang

**Abstract**: Given the great threat of adversarial attacks against Deep Neural Networks (DNNs), numerous works have been proposed to boost transferability to attack real-world applications. However, existing attacks often utilize advanced gradient calculation or input transformation but ignore the white-box model. Inspired by the fact that DNNs are over-parameterized for superior performance, we propose diversifying the high-level features (DHF) for more transferable adversarial examples. In particular, DHF perturbs the high-level features by randomly transforming the high-level features and mixing them with the feature of benign samples when calculating the gradient at each iteration. Due to the redundancy of parameters, such transformation does not affect the classification performance but helps identify the invariant features across different models, leading to much better transferability. Empirical evaluations on ImageNet dataset show that DHF could effectively improve the transferability of existing momentum-based attacks. Incorporated into the input transformation-based attacks, DHF generates more transferable adversarial examples and outperforms the baselines with a clear margin when attacking several defense models, showing its generalization to various attacks and high effectiveness for boosting transferability.

摘要: 鉴于针对深度神经网络的敌意攻击的巨大威胁，人们已经提出了许多工作来提高可转移性以攻击真实世界的应用。然而，现有的攻击往往利用先进的梯度计算或输入变换，而忽略了白盒模型。受DNN过度参数化以获得卓越性能这一事实的启发，我们建议将高级特征(DHF)多样化，以获得更多可转移的对抗性示例。特别是，DHF在每次迭代计算梯度时，通过随机变换高层特征并将其与良性样本的特征混合来扰动高层特征。由于参数的冗余性，这种变换不会影响分类性能，但有助于识别不同模型之间的不变特征，从而产生更好的可移植性。在ImageNet数据集上的实验评估表明，DHF能够有效地提高现有动量攻击的可转移性。DHF结合到基于输入变换的攻击中，生成了更多可转移的对抗性实例，在攻击多种防御模型时以明显的优势超过基线，显示了其对各种攻击的通用性和提高可转移性的高效性。



## **14. Towards the Universal Defense for Query-Based Audio Adversarial Attacks**

面向基于查询的音频攻击的通用防御 eess.AS

Submitted to Cybersecurity journal

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10088v1) [paper-pdf](http://arxiv.org/pdf/2304.10088v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: Recently, studies show that deep learning-based automatic speech recognition (ASR) systems are vulnerable to adversarial examples (AEs), which add a small amount of noise to the original audio examples. These AE attacks pose new challenges to deep learning security and have raised significant concerns about deploying ASR systems and devices. The existing defense methods are either limited in application or only defend on results, but not on process. In this work, we propose a novel method to infer the adversary intent and discover audio adversarial examples based on the AEs generation process. The insight of this method is based on the observation: many existing audio AE attacks utilize query-based methods, which means the adversary must send continuous and similar queries to target ASR models during the audio AE generation process. Inspired by this observation, We propose a memory mechanism by adopting audio fingerprint technology to analyze the similarity of the current query with a certain length of memory query. Thus, we can identify when a sequence of queries appears to be suspectable to generate audio AEs. Through extensive evaluation on four state-of-the-art audio AE attacks, we demonstrate that on average our defense identify the adversary intent with over 90% accuracy. With careful regard for robustness evaluations, we also analyze our proposed defense and its strength to withstand two adaptive attacks. Finally, our scheme is available out-of-the-box and directly compatible with any ensemble of ASR defense models to uncover audio AE attacks effectively without model retraining.

摘要: 最近的研究表明，基于深度学习的自动语音识别(ASR)系统容易受到对抗性样本(AES)的攻击，这些样本会在原始音频样本中添加少量噪声。这些AE攻击对深度学习安全提出了新的挑战，并引发了对部署ASR系统和设备的重大担忧。现有的防御方法要么局限于应用，要么只针对结果进行防御，而不是针对过程进行防御。在这项工作中，我们提出了一种新的方法来推断对手的意图，并发现音频对抗性实例的基础上，AES的生成过程。这种方法的洞察力是基于观察到的：现有的许多音频AE攻击都使用基于查询的方法，这意味着在音频AE生成过程中，攻击者必须向目标ASR模型发送连续的相似查询。受此启发，我们提出了一种采用音频指纹技术的记忆机制来分析当前查询与一定长度的记忆查询的相似度。因此，我们可以识别查询序列何时看起来可疑以生成音频AE。通过对四种最先进的音频AE攻击的广泛评估，我们证明了我们的防御平均识别对手意图的准确率超过90%。在仔细考虑健壮性评估的同时，我们还分析了我们提出的防御方案及其抵抗两种自适应攻击的能力。最后，我们的方案是开箱即用的，并且直接与任何ASR防御模型集成兼容，以有效地发现音频AE攻击，而不需要重新训练模型。



## **15. A Search-Based Testing Approach for Deep Reinforcement Learning Agents**

一种基于搜索的深度强化学习代理测试方法 cs.SE

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2206.07813v3) [paper-pdf](http://arxiv.org/pdf/2206.07813v3)

**Authors**: Amirhossein Zolfagharian, Manel Abdellatif, Lionel Briand, Mojtaba Bagherzadeh, Ramesh S

**Abstract**: Deep Reinforcement Learning (DRL) algorithms have been increasingly employed during the last decade to solve various decision-making problems such as autonomous driving and robotics. However, these algorithms have faced great challenges when deployed in safety-critical environments since they often exhibit erroneous behaviors that can lead to potentially critical errors. One way to assess the safety of DRL agents is to test them to detect possible faults leading to critical failures during their execution. This raises the question of how we can efficiently test DRL policies to ensure their correctness and adherence to safety requirements. Most existing works on testing DRL agents use adversarial attacks that perturb states or actions of the agent. However, such attacks often lead to unrealistic states of the environment. Their main goal is to test the robustness of DRL agents rather than testing the compliance of agents' policies with respect to requirements. Due to the huge state space of DRL environments, the high cost of test execution, and the black-box nature of DRL algorithms, the exhaustive testing of DRL agents is impossible. In this paper, we propose a Search-based Testing Approach of Reinforcement Learning Agents (STARLA) to test the policy of a DRL agent by effectively searching for failing executions of the agent within a limited testing budget. We use machine learning models and a dedicated genetic algorithm to narrow the search towards faulty episodes. We apply STARLA on Deep-Q-Learning agents which are widely used as benchmarks and show that it significantly outperforms Random Testing by detecting more faults related to the agent's policy. We also investigate how to extract rules that characterize faulty episodes of the DRL agent using our search results. Such rules can be used to understand the conditions under which the agent fails and thus assess its deployment risks.

摘要: 在过去的十年中，深度强化学习(DRL)算法被越来越多地用于解决各种决策问题，如自动驾驶和机器人技术。然而，当这些算法部署在安全关键环境中时，它们面临着巨大的挑战，因为它们经常表现出可能导致潜在关键错误的错误行为。评估DRL代理安全性的一种方法是对其进行测试，以检测在其执行期间可能导致严重故障的故障。这提出了一个问题，即我们如何有效地测试DRL政策，以确保它们的正确性和对安全要求的遵守。大多数现有的测试DRL代理的工作使用对抗性攻击，扰乱代理的状态或动作。然而，这样的攻击往往会导致不切实际的环境状况。他们的主要目标是测试DRL代理的健壮性，而不是测试代理策略与需求的符合性。由于DRL环境的状态空间巨大、测试执行成本高以及DRL算法的黑箱性质，对DRL代理进行穷举测试是不可能的。在本文中，我们提出了一种基于搜索的强化学习代理测试方法(STARLA)，通过在有限的测试预算内有效地搜索代理的失败执行来测试DRL代理的策略。我们使用机器学习模型和专门的遗传算法将搜索范围缩小到故障剧集。我们将Starla应用于被广泛用作基准测试的Deep-Q-Learning代理上，结果表明它在检测到更多与代理策略相关的错误方面明显优于随机测试。我们还研究了如何使用搜索结果提取表征DRL代理故障情节的规则。此类规则可用于了解代理发生故障的条件，从而评估其部署风险。



## **16. Quantifying the Preferential Direction of the Model Gradient in Adversarial Training With Projected Gradient Descent**

用投影梯度下降法量化对抗性训练中模型梯度的优先方向 stat.ML

This paper was published in Pattern Recognition

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2009.04709v5) [paper-pdf](http://arxiv.org/pdf/2009.04709v5)

**Authors**: Ricardo Bigolin Lanfredi, Joyce D. Schroeder, Tolga Tasdizen

**Abstract**: Adversarial training, especially projected gradient descent (PGD), has proven to be a successful approach for improving robustness against adversarial attacks. After adversarial training, gradients of models with respect to their inputs have a preferential direction. However, the direction of alignment is not mathematically well established, making it difficult to evaluate quantitatively. We propose a novel definition of this direction as the direction of the vector pointing toward the closest point of the support of the closest inaccurate class in decision space. To evaluate the alignment with this direction after adversarial training, we apply a metric that uses generative adversarial networks to produce the smallest residual needed to change the class present in the image. We show that PGD-trained models have a higher alignment than the baseline according to our definition, that our metric presents higher alignment values than a competing metric formulation, and that enforcing this alignment increases the robustness of models.

摘要: 对抗性训练，特别是投影梯度下降(PGD)，已被证明是提高对抗攻击的稳健性的一种成功方法。经过对抗性训练后，模型相对于其输入的梯度具有优先的方向。然而，对齐的方向在数学上没有很好的确定，因此很难进行定量评估。我们提出了一种新的方向定义，即向量指向决策空间中最接近的不准确类的支持度的最近点的方向。为了在对抗性训练后评估与这一方向的一致性，我们应用了一种度量，该度量使用生成性对抗性网络来产生改变图像中存在的类别所需的最小残差。我们表明，根据我们的定义，PGD训练的模型具有比基线更高的比对，我们的指标比竞争指标公式提供了更高的比对值，并且强制执行这种比对提高了模型的稳健性。



## **17. Jedi: Entropy-based Localization and Removal of Adversarial Patches**

绝地：基于熵的敌方补丁定位与移除 cs.CR

9 pages, 11 figures. To appear in CVPR 2023

**SubmitDate**: 2023-04-20    [abs](http://arxiv.org/abs/2304.10029v1) [paper-pdf](http://arxiv.org/pdf/2304.10029v1)

**Authors**: Bilel Tarchoun, Anouar Ben Khalifa, Mohamed Ali Mahjoub, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Real-world adversarial physical patches were shown to be successful in compromising state-of-the-art models in a variety of computer vision applications. Existing defenses that are based on either input gradient or features analysis have been compromised by recent GAN-based attacks that generate naturalistic patches. In this paper, we propose Jedi, a new defense against adversarial patches that is resilient to realistic patch attacks. Jedi tackles the patch localization problem from an information theory perspective; leverages two new ideas: (1) it improves the identification of potential patch regions using entropy analysis: we show that the entropy of adversarial patches is high, even in naturalistic patches; and (2) it improves the localization of adversarial patches, using an autoencoder that is able to complete patch regions from high entropy kernels. Jedi achieves high-precision adversarial patch localization, which we show is critical to successfully repair the images. Since Jedi relies on an input entropy analysis, it is model-agnostic, and can be applied on pre-trained off-the-shelf models without changes to the training or inference of the protected models. Jedi detects on average 90% of adversarial patches across different benchmarks and recovers up to 94% of successful patch attacks (Compared to 75% and 65% for LGS and Jujutsu, respectively).

摘要: 现实世界中的对抗性物理补丁被证明在各种计算机视觉应用中成功地折衷了最先进的模型。现有的基于输入梯度或特征分析的防御已经被最近基于GaN的攻击所破坏，这些攻击产生了自然主义的补丁。在这篇文章中，我们提出了一种新的防御对手补丁的绝地，它对现实的补丁攻击具有弹性。绝地从信息论的角度解决了补丁定位问题；利用了两个新的想法：(1)它利用熵分析改进了潜在补丁区域的识别：我们证明了对抗性补丁的熵很高，即使在自然补丁中也是如此；(2)它改进了对抗性补丁的定位，使用了能够从高熵内核完成补丁区域的自动编码器。绝地实现了高精度的对抗性补丁定位，我们证明这是成功修复图像的关键。由于绝地依赖于输入熵分析，它与模型无关，可以应用于预先训练的现成模型，而不需要改变受保护模型的训练或推断。绝地平均可以通过不同的基准检测90%的敌方补丁，并恢复高达94%的成功补丁攻击(相比之下，LGS和Jujutsu的这一比例分别为75%和65%)。



## **18. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.11082v1) [paper-pdf](http://arxiv.org/pdf/2304.11082v1)

**Authors**: Yotam Wolf, Noam Wies, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了对于模型表现出的有限概率的任何行为，都存在可以触发模型输出该行为的提示，其概率随着提示的长度增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，增加了LLM被提示进入不希望看到的行为的倾向。此外，我们在我们的BEB框架中包括了人物角色的概念，并发现通过促使模型表现为特定的人物角色，通常不太可能在模型中表现的行为可以被带到前面。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **19. GREAT Score: Global Robustness Evaluation of Adversarial Perturbation using Generative Models**

高分：使用生成模型对对抗性扰动进行全局稳健性评估 cs.LG

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09875v1) [paper-pdf](http://arxiv.org/pdf/2304.09875v1)

**Authors**: Li Zaitang, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Current studies on adversarial robustness mainly focus on aggregating local robustness results from a set of data samples to evaluate and rank different models. However, the local statistics may not well represent the true global robustness of the underlying unknown data distribution. To address this challenge, this paper makes the first attempt to present a new framework, called GREAT Score , for global robustness evaluation of adversarial perturbation using generative models. Formally, GREAT Score carries the physical meaning of a global statistic capturing a mean certified attack-proof perturbation level over all samples drawn from a generative model. For finite-sample evaluation, we also derive a probabilistic guarantee on the sample complexity and the difference between the sample mean and the true mean. GREAT Score has several advantages: (1) Robustness evaluations using GREAT Score are efficient and scalable to large models, by sparing the need of running adversarial attacks. In particular, we show high correlation and significantly reduced computation cost of GREAT Score when compared to the attack-based model ranking on RobustBench (Croce,et. al. 2021). (2) The use of generative models facilitates the approximation of the unknown data distribution. In our ablation study with different generative adversarial networks (GANs), we observe consistency between global robustness evaluation and the quality of GANs. (3) GREAT Score can be used for remote auditing of privacy-sensitive black-box models, as demonstrated by our robustness evaluation on several online facial recognition services.

摘要: 目前关于对抗稳健性的研究主要集中在从一组数据样本中聚集局部稳健性结果来评估和排序不同的模型。然而，局部统计可能不能很好地代表潜在未知数据分布的真实全局稳健性。为了应对这一挑战，本文首次尝试提出了一种新的框架，称为Great Score，用于利用产生式模型评估对抗扰动的全局稳健性。在形式上，高分具有全球统计的物理意义，该统计捕获来自生成模型的所有样本的平均经认证的防攻击扰动水平。对于有限样本评价，我们还得到了样本复杂度和样本均值与真均值之差的概率保证。Great Score有几个优点：(1)使用Great Score进行健壮性评估是高效的，并且可以扩展到大型模型，因为它避免了运行对抗性攻击的需要。特别是，与基于攻击的模型排名相比，我们表现出了高度的相关性和显著的降低了计算开销。艾尔2021年)。(2)生成模型的使用有利于未知数据分布的近似。在我们对不同生成对抗网络(GANS)的消融研究中，我们观察到全局健壮性评估与GANS质量之间的一致性。(3)Great Score可以用于隐私敏感的黑盒模型的远程审计，我们在几种在线人脸识别服务上的健壮性评估证明了这一点。



## **20. Experimental Certification of Quantum Transmission via Bell's Theorem**

用贝尔定理进行量子传输的实验验证 quant-ph

34 pages, 14 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09605v1) [paper-pdf](http://arxiv.org/pdf/2304.09605v1)

**Authors**: Simon Neves, Laura dos Santos Martins, Verena Yacoub, Pascal Lefebvre, Ivan Supic, Damian Markham, Eleni Diamanti

**Abstract**: Quantum transmission links are central elements in essentially all implementations of quantum information protocols. Emerging progress in quantum technologies involving such links needs to be accompanied by appropriate certification tools. In adversarial scenarios, a certification method can be vulnerable to attacks if too much trust is placed on the underlying system. Here, we propose a protocol in a device independent framework, which allows for the certification of practical quantum transmission links in scenarios where minimal assumptions are made about the functioning of the certification setup. In particular, we take unavoidable transmission losses into account by modeling the link as a completely-positive trace-decreasing map. We also, crucially, remove the assumption of independent and identically distributed samples, which is known to be incompatible with adversarial settings. Finally, in view of the use of the certified transmitted states for follow-up applications, our protocol moves beyond certification of the channel to allow us to estimate the quality of the transmitted state itself. To illustrate the practical relevance and the feasibility of our protocol with currently available technology we provide an experimental implementation based on a state-of-the-art polarization entangled photon pair source in a Sagnac configuration and analyze its robustness for realistic losses and errors.

摘要: 量子传输链路是几乎所有量子信息协议实现的核心要素。涉及这种联系的量子技术的新进展需要伴随着适当的认证工具。在对抗性场景中，如果对底层系统的信任过高，则认证方法可能容易受到攻击。在这里，我们提出了一个独立于设备的框架中的协议，该协议允许在对认证设置的功能做出最小假设的情况下对实际量子传输链路进行认证。特别地，我们通过将链路建模为完全正迹递减映射来考虑不可避免的传输损耗。至关重要的是，我们还取消了独立和同分布样本的假设，这是已知与对抗性设置不兼容的。最后，考虑到后续应用对认证的传输状态的使用，我们的协议超越了对信道的认证，允许我们估计传输状态本身的质量。为了说明我们的协议与现有技术的实际相关性和可行性，我们提供了一个基于Sagnac结构中最先进的偏振纠缠光子对源的实验实现，并分析了其对现实损失和错误的稳健性。



## **21. Masked Language Model Based Textual Adversarial Example Detection**

基于掩蔽语言模型的文本对抗性实例检测 cs.CR

13 pages,3 figures

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.08767v2) [paper-pdf](http://arxiv.org/pdf/2304.08767v2)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.

摘要: 对抗性攻击对机器学习模型在安全关键型应用中的可靠部署构成严重威胁。他们可以通过稍微修改输入来误导当前的模型进行错误的预测。最近的大量工作表明，对抗性例子往往偏离正常例子的底层数据流形，而预先训练的掩蔽语言模型可以适应正常NLP数据的流形。为了探索掩蔽语言模型在对抗性检测中的应用，我们提出了一种新的文本对抗性实例检测方法，即基于掩蔽语言模型的检测方法(MLMD)，该方法通过研究掩蔽语言模型引起的流形变化来产生能够清晰区分正常例子和对抗性例子的信号。MLMD具有即插即用的特点(即，不需要重新训练受害者模型)用于对抗防御，并且它与分类任务、受害者模型的体系结构和要防御的攻击方法无关。我们在各种基准文本数据集、广泛研究的机器学习模型和最先进的(SOTA)对手攻击(总计$3*4*4=48$设置)上评估MLMD。实验结果表明，该算法在AG-NEWS、IMDB和SST-2数据集上的检测准确率分别达到0.984、0.967和0.901。此外，MLMD在检测精度和F1得分方面优于SOTA检测防御，或至少与SOTA检测防御相当。在许多基于对抗性例子的非流形假设的防御中，这项工作为捕捉流形变化提供了一个新的角度。这项工作的代码可以在\url{https://github.com/mlmddetection/MLMDdetection}.上公开访问



## **22. Understanding Overfitting in Adversarial Training via Kernel Regression**

用核回归方法理解对抗性训练中的过度适应 stat.ML

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.06326v2) [paper-pdf](http://arxiv.org/pdf/2304.06326v2)

**Authors**: Teng Zhang, Kang Li

**Abstract**: Adversarial training and data augmentation with noise are widely adopted techniques to enhance the performance of neural networks. This paper investigates adversarial training and data augmentation with noise in the context of regularized regression in a reproducing kernel Hilbert space (RKHS). We establish the limiting formula for these techniques as the attack and noise size, as well as the regularization parameter, tend to zero. Based on this limiting formula, we analyze specific scenarios and demonstrate that, without appropriate regularization, these two methods may have larger generalization error and Lipschitz constant than standard kernel regression. However, by selecting the appropriate regularization parameter, these two methods can outperform standard kernel regression and achieve smaller generalization error and Lipschitz constant. These findings support the empirical observations that adversarial training can lead to overfitting, and appropriate regularization methods, such as early stopping, can alleviate this issue.

摘要: 对抗性训练和带噪声的数据增强是提高神经网络性能的广泛采用的技术。在再生核Hilbert空间(RKHS)的正则化回归背景下，研究了带噪声的对抗性训练和数据增强问题。当攻击和噪声大小以及正则化参数趋于零时，我们建立了这些技术的极限公式。基于这一极限公式，我们分析了具体的情形，并证明了在没有适当的正则化的情况下，这两种方法可能具有比标准核回归更大的泛化误差和Lipschitz常数。然而，通过选择合适的正则化参数，这两种方法都可以获得比标准核回归更好的性能，并获得更小的泛化误差和Lipschitz常数。这些发现支持了经验观察，即对抗性训练会导致过度适应，而适当的正规化方法，如提前停止，可以缓解这一问题。



## **23. Secure Split Learning against Property Inference, Data Reconstruction, and Feature Space Hijacking Attacks**

抗属性推理、数据重构和特征空间劫持攻击的安全分裂学习 cs.LG

23 pages

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09515v1) [paper-pdf](http://arxiv.org/pdf/2304.09515v1)

**Authors**: Yunlong Mao, Zexi Xin, Zhenyu Li, Jue Hong, Qingyou Yang, Sheng Zhong

**Abstract**: Split learning of deep neural networks (SplitNN) has provided a promising solution to learning jointly for the mutual interest of a guest and a host, which may come from different backgrounds, holding features partitioned vertically. However, SplitNN creates a new attack surface for the adversarial participant, holding back its practical use in the real world. By investigating the adversarial effects of highly threatening attacks, including property inference, data reconstruction, and feature hijacking attacks, we identify the underlying vulnerability of SplitNN and propose a countermeasure. To prevent potential threats and ensure the learning guarantees of SplitNN, we design a privacy-preserving tunnel for information exchange between the guest and the host. The intuition is to perturb the propagation of knowledge in each direction with a controllable unified solution. To this end, we propose a new activation function named R3eLU, transferring private smashed data and partial loss into randomized responses in forward and backward propagations, respectively. We give the first attempt to secure split learning against three threatening attacks and present a fine-grained privacy budget allocation scheme. The analysis proves that our privacy-preserving SplitNN solution provides a tight privacy budget, while the experimental results show that our solution performs better than existing solutions in most cases and achieves a good tradeoff between defense and model usability.

摘要: 深度神经网络的分裂学习(SplitNN)为来自不同背景、拥有垂直分割特征的宾主共同兴趣的联合学习提供了一种很有前途的解决方案。然而，SplitNN为对手参与者创造了一个新的攻击面，阻碍了其在现实世界中的实际应用。通过研究高威胁性攻击的对抗性，包括属性推断、数据重构和特征劫持攻击，我们识别了SplitNN的潜在脆弱性，并提出了对策。为了防止潜在的威胁，并保证SplitNN的学习保证，我们设计了一条隐私保护隧道，用于客户和主机之间的信息交换。直觉是用一种可控的统一解决方案扰乱知识在各个方向的传播。为此，我们提出了一种新的激活函数R3eLU，在前向传播和后向传播中分别将私有粉碎数据和部分丢失转移到随机响应中。我们首次尝试保护分裂学习不受三种威胁攻击，并提出了一种细粒度的隐私预算分配方案。分析证明，我们的隐私保护SplitNN方案提供了较少的隐私预算，而实验结果表明，我们的方案在大多数情况下都比现有的方案性能更好，并在防御性和模型可用性之间取得了良好的折衷。



## **24. Maybenot: A Framework for Traffic Analysis Defenses**

Maybenot：一种流量分析防御框架 cs.CR

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09510v1) [paper-pdf](http://arxiv.org/pdf/2304.09510v1)

**Authors**: Tobias Pulls

**Abstract**: End-to-end encryption is a powerful tool for protecting the privacy of Internet users. Together with the increasing use of technologies such as Tor, VPNs, and encrypted messaging, it is becoming increasingly difficult for network adversaries to monitor and censor Internet traffic. One remaining avenue for adversaries is traffic analysis: the analysis of patterns in encrypted traffic to infer information about the users and their activities. Recent improvements using deep learning have made traffic analysis attacks more effective than ever before.   We present Maybenot, a framework for traffic analysis defenses. Maybenot is designed to be easy to use and integrate into existing end-to-end encrypted protocols. It is implemented in the Rust programming language as a crate (library), together with a simulator to further the development of defenses. Defenses in Maybenot are expressed as probabilistic state machines that schedule actions to inject padding or block outgoing traffic. Maybenot is an evolution from the Tor Circuit Padding Framework by Perry and Kadianakis, designed to support a wide range of protocols and use cases.

摘要: 端到端加密是保护互联网用户隐私的有力工具。随着ToR、VPN和加密消息等技术的使用越来越多，网络对手监控和审查互联网流量变得越来越困难。攻击者剩下的另一个途径是流量分析：分析加密流量中的模式，以推断有关用户及其活动的信息。最近使用深度学习的改进使流量分析攻击比以往任何时候都更有效。我们提出了一个用于流量分析防御的框架--Maybenot。Maybenot被设计为易于使用并集成到现有的端到端加密协议中。它在Rust编程语言中被实现为一个板条箱(库)，以及一个模拟器来进一步开发防御。Maybenot中的防御被表示为概率状态机，该状态机调度注入填充或阻止传出流量的动作。Maybenot是由Perry和Kadianakis提出的Tor电路填充框架的演变，旨在支持广泛的协议和用例。



## **25. Wavelets Beat Monkeys at Adversarial Robustness**

小波在对抗健壮性上击败猴子 cs.LG

Machine Learning and the Physical Sciences Workshop, NeurIPS 2022

**SubmitDate**: 2023-04-19    [abs](http://arxiv.org/abs/2304.09403v1) [paper-pdf](http://arxiv.org/pdf/2304.09403v1)

**Authors**: Jingtong Su, Julia Kempe

**Abstract**: Research on improving the robustness of neural networks to adversarial noise - imperceptible malicious perturbations of the data - has received significant attention. The currently uncontested state-of-the-art defense to obtain robust deep neural networks is Adversarial Training (AT), but it consumes significantly more resources compared to standard training and trades off accuracy for robustness. An inspiring recent work [Dapello et al.] aims to bring neurobiological tools to the question: How can we develop Neural Nets that robustly generalize like human vision? [Dapello et al.] design a network structure with a neural hidden first layer that mimics the primate primary visual cortex (V1), followed by a back-end structure adapted from current CNN vision models. It seems to achieve non-trivial adversarial robustness on standard vision benchmarks when tested on small perturbations. Here we revisit this biologically inspired work, and ask whether a principled parameter-free representation with inspiration from physics is able to achieve the same goal. We discover that the wavelet scattering transform can replace the complex V1-cortex and simple uniform Gaussian noise can take the role of neural stochasticity, to achieve adversarial robustness. In extensive experiments on the CIFAR-10 benchmark with adaptive adversarial attacks we show that: 1) Robustness of VOneBlock architectures is relatively weak (though non-zero) when the strength of the adversarial attack radius is set to commonly used benchmarks. 2) Replacing the front-end VOneBlock by an off-the-shelf parameter-free Scatternet followed by simple uniform Gaussian noise can achieve much more substantial adversarial robustness without adversarial training. Our work shows how physically inspired structures yield new insights into robustness that were previously only thought possible by meticulously mimicking the human cortex.

摘要: 提高神经网络对敌意噪声--数据的不可察觉的恶意扰动--的稳健性的研究受到了极大的关注。目前无可争议的获得稳健深度神经网络的最先进的防御是对抗性训练(AT)，但与标准训练相比，它消耗的资源明显更多，并以精度换取健壮性。最近一部鼓舞人心的作品[Dapello等人]旨在将神经生物学工具引入这个问题：我们如何才能开发出像人类视觉一样具有强大泛化能力的神经网络？[Dapello等人]设计一个网络结构，该网络结构具有一个模仿灵长类初级视觉皮质(V1)的神经隐藏第一层，然后是一个改编自当前CNN视觉模型的后端结构。当在小扰动上测试时，它似乎在标准视觉基准上实现了非平凡的对抗性健壮性。在这里，我们重新审视这项受生物启发的工作，并询问受物理学启发的原则性无参数表示法是否能够实现同样的目标。我们发现，小波散射变换可以代替复杂的V1皮层，而简单的均匀高斯噪声可以起到神经随机性的作用，达到对抗的稳健性。在CIFAR-10基准上进行的大量自适应攻击实验表明：1)当敌方攻击半径设置为常用基准时，VOneBlock架构的健壮性相对较弱(尽管非零值)。2)用无参数散射网和简单的均匀高斯噪声代替前端的VOneBlock，无需对抗性训练即可获得更强的对抗性健壮性。我们的工作展示了受物理启发的结构如何产生对健壮性的新见解，而这在以前只被认为是通过精心模仿人类皮质才可能实现的。



## **26. CodeAttack: Code-Based Adversarial Attacks for Pre-trained Programming Language Models**

CodeAttack：针对预先训练的编程语言模型的基于代码的对抗性攻击 cs.CL

AAAI Conference on Artificial Intelligence (AAAI) 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2206.00052v3) [paper-pdf](http://arxiv.org/pdf/2206.00052v3)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstract**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models operate in the natural channel of code, i.e., they are primarily concerned with the human understanding of the code. They are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks in the natural channel. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate effective, efficient, and imperceptible adversarial code samples and demonstrates the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall drop in performance while being more efficient, imperceptible, consistent, and fluent. The code can be found at https://github.com/reddy-lab-code-research/CodeAttack.

摘要: 预先训练的编程语言(PL)模型(如CodeT5、CodeBERT、GraphCodeBERT等)有可能自动化涉及代码理解和代码生成的软件工程任务。然而，这些模型在代码的自然通道中运行，即它们主要关注人类对代码的理解。它们对输入的变化不是很健壮，因此，在自然通道中可能容易受到对抗性攻击。我们提出了一个简单而有效的黑盒攻击模型CodeAttack，它使用代码结构来生成有效、高效和不可察觉的对抗性代码样本，并展示了最新的PL模型对代码特定的对抗性攻击的脆弱性。我们评估了CodeAttack在几个代码-代码(翻译和修复)和代码-NL(摘要)任务上跨不同编程语言的可移植性。CodeAttack超越了最先进的对抗性NLP攻击模型，在更高效、更隐蔽、更一致和更流畅的同时，实现了最佳的整体性能下降。代码可在https://github.com/reddy-lab-code-research/CodeAttack.上找到



## **27. Analyzing Activity and Suspension Patterns of Twitter Bots Attacking Turkish Twitter Trends by a Longitudinal Dataset**

利用纵向数据集分析Twitter机器人攻击土耳其Twitter趋势的活跃度和暂停模式 cs.SI

Accepted to Cyber Social Threats (CySoc) 2023 colocated with  WebConf23

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.07907v2) [paper-pdf](http://arxiv.org/pdf/2304.07907v2)

**Authors**: Tuğrulcan Elmas

**Abstract**: Twitter bots amplify target content in a coordinated manner to make them appear popular, which is an astroturfing attack. Such attacks promote certain keywords to push them to Twitter trends to make them visible to a broader audience. Past work on such fake trends revealed a new astroturfing attack named ephemeral astroturfing that employs a very unique bot behavior in which bots post and delete generated tweets in a coordinated manner. As such, it is easy to mass-annotate such bots reliably, making them a convenient source of ground truth for bot research. In this paper, we detect and disclose over 212,000 such bots targeting Turkish trends, which we name astrobots. We also analyze their activity and suspension patterns. We found that Twitter purged those bots en-masse 6 times since June 2018. However, the adversaries reacted quickly and deployed new bots that were created years ago. We also found that many such bots do not post tweets apart from promoting fake trends, which makes it challenging for bot detection methods to detect them. Our work provides insights into platforms' content moderation practices and bot detection research. The dataset is publicly available at https://github.com/tugrulz/EphemeralAstroturfing.

摘要: Twitter机器人以一种协调的方式放大目标内容，使它们看起来很受欢迎，这是一种占星术的攻击。这类攻击推广某些关键字，将它们推向Twitter趋势，让更多的受众看到它们。过去对这种虚假趋势的研究揭示了一种名为短暂占星术的新的占星术攻击，它采用了一种非常独特的机器人行为，机器人以协调的方式发布和删除生成的推文。因此，很容易对这类机器人进行可靠的大规模注释，使它们成为机器人研究的一个方便的基本事实来源。在本文中，我们检测并披露了超过21.2万个针对土耳其趋势的此类机器人，我们将其命名为机器人。我们还分析了它们的活动和悬浮模式。我们发现，自2018年6月以来，Twitter对这些机器人进行了6次集体清除。然而，对手们反应迅速，部署了几年前创造的新机器人。我们还发现，许多这类机器人除了宣传虚假趋势外，不会发布推文，这使得机器人检测方法对检测它们具有挑战性。我们的工作为平台的内容审核实践和机器人检测研究提供了见解。该数据集在https://github.com/tugrulz/EphemeralAstroturfing.上公开提供



## **28. An Analysis of Robustness of Non-Lipschitz Networks**

非Lipschitz网络的稳健性分析 cs.LG

To appear in Journal of Machine Learning Research (JMLR)

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2010.06154v4) [paper-pdf](http://arxiv.org/pdf/2010.06154v4)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.

摘要: 尽管取得了重大进展，深度网络仍然极易受到对手的攻击。一个基本的挑战是，小的输入扰动通常会在网络的最后一层特征空间中产生大的运动。在本文中，我们定义了一个抽象这一挑战的攻击模型，以帮助理解其内在属性。在我们的模型中，敌手可以将数据在特征空间中移动任意距离，但只能在随机低维子空间中移动。我们证明了这样的对手可以是相当强大的：击败任何必须对给定的任何输入进行分类的算法。然而，通过允许算法在不寻常的输入上弃权，我们证明了当类在特征空间中合理地分离时，这样的对手可以被克服。我们进一步为使用数据驱动方法设置算法参数以优化过度精确度权衡提供了有力的理论保证。我们的结果为最近邻式算法提供了新的稳健性保证，并在对比学习中也得到了应用，我们的经验证明了这种算法能够在较低的弃权率下获得较高的鲁棒性精度。我们的模型也受到战略分类的推动，在战略分类中，被分类的实体旨在操纵它们的可观察特征来产生首选的分类，我们也提供了对该领域的新见解。



## **29. BadVFL: Backdoor Attacks in Vertical Federated Learning**

BadVFL：垂直联合学习中的后门攻击 cs.LG

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08847v1) [paper-pdf](http://arxiv.org/pdf/2304.08847v1)

**Authors**: Mohammad Naseri, Yufei Han, Emiliano De Cristofaro

**Abstract**: Federated learning (FL) enables multiple parties to collaboratively train a machine learning model without sharing their data; rather, they train their own model locally and send updates to a central server for aggregation. Depending on how the data is distributed among the participants, FL can be classified into Horizontal (HFL) and Vertical (VFL). In VFL, the participants share the same set of training instances but only host a different and non-overlapping subset of the whole feature space. Whereas in HFL, each participant shares the same set of features while the training set is split into locally owned training data subsets.   VFL is increasingly used in applications like financial fraud detection; nonetheless, very little work has analyzed its security. In this paper, we focus on robustness in VFL, in particular, on backdoor attacks, whereby an adversary attempts to manipulate the aggregate model during the training process to trigger misclassifications. Performing backdoor attacks in VFL is more challenging than in HFL, as the adversary i) does not have access to the labels during training and ii) cannot change the labels as she only has access to the feature embeddings. We present a first-of-its-kind clean-label backdoor attack in VFL, which consists of two phases: a label inference and a backdoor phase. We demonstrate the effectiveness of the attack on three different datasets, investigate the factors involved in its success, and discuss countermeasures to mitigate its impact.

摘要: 联合学习(FL)使多方能够协作地训练机器学习模型，而不共享他们的数据；相反，他们在本地训练他们自己的模型，并将更新发送到中央服务器以进行聚合。根据数据在参与者之间的分布情况，外语可分为水平(HFL)和垂直(VFL)。在VFL中，参与者共享相同的训练实例集，但仅托管整个特征空间的不同且不重叠的子集。而在HFL中，每个参与者共享相同的特征集，而训练集被分成本地拥有的训练数据子集。VFL越来越多地被用于金融欺诈检测等应用中；然而，很少有工作分析它的安全性。在本文中，我们关注VFL的稳健性，特别是后门攻击，即对手试图在训练过程中操纵聚合模型以触发错误分类。在VFL中执行后门攻击比在HFL中更具挑战性，因为对手i)在训练期间无法访问标签，ii)无法更改标签，因为她只能访问特征嵌入。在VFL中，我们提出了一种首次的干净标签后门攻击，它包括两个阶段：标签推理和后门阶段。我们在三个不同的数据集上演示了攻击的有效性，调查了其成功的相关因素，并讨论了减轻其影响的对策。



## **30. Towards the Transferable Audio Adversarial Attack via Ensemble Methods**

基于集成方法的可转移音频对抗攻击 cs.CR

Submitted to Cybersecurity journal 2023

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.08811v1) [paper-pdf](http://arxiv.org/pdf/2304.08811v1)

**Authors**: Feng Guo, Zheng Sun, Yuxuan Chen, Lei Ju

**Abstract**: In recent years, deep learning (DL) models have achieved significant progress in many domains, such as autonomous driving, facial recognition, and speech recognition. However, the vulnerability of deep learning models to adversarial attacks has raised serious concerns in the community because of their insufficient robustness and generalization. Also, transferable attacks have become a prominent method for black-box attacks. In this work, we explore the potential factors that impact adversarial examples (AEs) transferability in DL-based speech recognition. We also discuss the vulnerability of different DL systems and the irregular nature of decision boundaries. Our results show a remarkable difference in the transferability of AEs between speech and images, with the data relevance being low in images but opposite in speech recognition. Motivated by dropout-based ensemble approaches, we propose random gradient ensembles and dynamic gradient-weighted ensembles, and we evaluate the impact of ensembles on the transferability of AEs. The results show that the AEs created by both approaches are valid for transfer to the black box API.

摘要: 近年来，深度学习模型在自动驾驶、人脸识别、语音识别等领域取得了长足的进步。然而，深度学习模型对敌意攻击的脆弱性已经引起了社会各界的严重关注，因为它们的健壮性和泛化程度不够。此外，可转移攻击已经成为黑盒攻击的一种重要方法。在这项工作中，我们探讨了在基于数字图书馆的语音识别中影响对抗性例子(AES)可转移性的潜在因素。我们还讨论了不同DL系统的脆弱性和决策边界的不规则性。我们的结果表明，语音和图像之间的声学效应的可转移性有显著差异，图像中的数据相关性较低，而语音识别中的数据相关性则相反。在基于辍学的集成方法的启发下，我们提出了随机梯度集成和动态梯度加权集成，并评估了集成对高级进化的可转移性的影响。结果表明，这两种方法生成的动态平衡函数都可以有效地转移到黑盒API中。



## **31. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

有序-无序：对黑盒神经网络排序模型的模仿敌意攻击 cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022, Best Paper Nomination

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2209.06506v2) [paper-pdf](http://arxiv.org/pdf/2209.06506v2)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstract**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.

摘要: 神经文本排序模型已经取得了显著的进步，并越来越多地应用于实践中。不幸的是，它们也继承了一般神经模型的对抗性漏洞，这些漏洞已经被检测到，但仍未被先前的研究充分探索。此外，BlackHat SEO可能会利用继承的敌意漏洞来击败保护更好的搜索引擎。在这项研究中，我们提出了一种对黑盒神经通路排序模型的模仿对抗性攻击。我们首先证明了目标文章排序模型可以通过列举关键查询/候选来透明化和模仿，然后训练一个排序模仿模型。利用排序模拟模型，可以对排序结果进行精细的操作，并将操纵攻击转移到目标排序模型。为此，我们提出了一种创新的基于梯度的攻击方法，该方法在两两目标函数的授权下，生成对抗性触发器，以极少的令牌造成有预谋的无序。为了装备触发伪装，我们在目标函数中加入了下一句预测损失和语言模型流畅度约束。文章排序的实验结果证明了该排序模仿攻击模型和敌意触发器对各种SOTA神经排序模型的有效性。此外，各种缓解分析和人类评估表明，当面临潜在的缓解方法时，伪装是有效的。为了激励其他学者进一步研究这一新颖而重要的问题，我们公开了实验数据和代码。



## **32. AI Product Security: A Primer for Developers**

AI产品安全：开发者入门 cs.CR

10 pages, 1 figure

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2304.11087v1) [paper-pdf](http://arxiv.org/pdf/2304.11087v1)

**Authors**: Ebenezer R. H. P. Isaac, Jim Reno

**Abstract**: Not too long ago, AI security used to mean the research and practice of how AI can empower cybersecurity, that is, AI for security. Ever since Ian Goodfellow and his team popularized adversarial attacks on machine learning, security for AI became an important concern and also part of AI security. It is imperative to understand the threats to machine learning products and avoid common pitfalls in AI product development. This article is addressed to developers, designers, managers and researchers of AI software products.

摘要: 不久前，AI安全曾经意味着AI如何赋能网络安全的研究和实践，也就是AI for Security。自从伊恩·古德费罗和他的团队普及了对机器学习的对抗性攻击以来，人工智能的安全成为一个重要的问题，也是人工智能安全的一部分。了解机器学习产品面临的威胁，避免人工智能产品开发中的常见陷阱是当务之急。本文面向AI软件产品的开发人员、设计人员、管理人员和研究人员。



## **33. A Survey of Adversarial Defences and Robustness in NLP**

自然语言处理中的对抗性防御和健壮性研究综述 cs.CL

Accepted for publication at ACM Computing Surveys

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2203.06414v4) [paper-pdf](http://arxiv.org/pdf/2203.06414v4)

**Authors**: Shreya Goyal, Sumanth Doddapaneni, Mitesh M. Khapra, Balaraman Ravindran

**Abstract**: In the past few years, it has become increasingly evident that deep neural networks are not resilient enough to withstand adversarial perturbations in input data, leaving them vulnerable to attack. Various authors have proposed strong adversarial attacks for computer vision and Natural Language Processing (NLP) tasks. As a response, many defense mechanisms have also been proposed to prevent these networks from failing. The significance of defending neural networks against adversarial attacks lies in ensuring that the model's predictions remain unchanged even if the input data is perturbed. Several methods for adversarial defense in NLP have been proposed, catering to different NLP tasks such as text classification, named entity recognition, and natural language inference. Some of these methods not only defend neural networks against adversarial attacks but also act as a regularization mechanism during training, saving the model from overfitting. This survey aims to review the various methods proposed for adversarial defenses in NLP over the past few years by introducing a novel taxonomy. The survey also highlights the fragility of advanced deep neural networks in NLP and the challenges involved in defending them.

摘要: 在过去的几年里，越来越明显的是，深度神经网络没有足够的弹性来抵御输入数据中的对抗性扰动，从而使它们容易受到攻击。许多作者提出了针对计算机视觉和自然语言处理(NLP)任务的强对抗性攻击。作为回应，许多防御机制也被提出，以防止这些网络崩溃。保护神经网络免受敌意攻击的意义在于确保即使输入数据受到扰动，模型的预测也保持不变。针对自然语言处理中文本分类、命名实体识别、自然语言推理等不同的自然语言处理任务，提出了几种自然语言处理对抗性防御方法。这些方法中的一些不仅保护神经网络免受对手攻击，而且在训练过程中充当正则化机制，避免了模型的过拟合。本综述旨在通过引入一种新的分类方法来回顾过去几年中提出的在NLP中进行对抗防御的各种方法。这项调查还突显了NLP中先进的深度神经网络的脆弱性以及捍卫它们所涉及的挑战。



## **34. Binarized ResNet: Enabling Robust Automatic Modulation Classification at the resource-constrained Edge**

二值化ResNet：在资源受限的边缘实现稳健的自动调制分类 cs.IT

This version has a total of 8 figures and 3 tables. It has extra  content on the adversarial robustness of the proposed method that was not  present in the previous submission. Also one more ensemble method called  RBLResNet-MCK is proposed to improve the performance further

**SubmitDate**: 2023-04-18    [abs](http://arxiv.org/abs/2110.14357v2) [paper-pdf](http://arxiv.org/pdf/2110.14357v2)

**Authors**: Deepsayan Sadhukhan, Nitin Priyadarshini Shankar, Nancy Nayak, Thulasi Tholeti, Sheetal Kalyani

**Abstract**: Recently, deep neural networks (DNNs) have been used extensively for automatic modulation classification (AMC), and the results have been quite promising. However, DNNs have high memory and computation requirements making them impractical for edge networks where the devices are resource-constrained. They are also vulnerable to adversarial attacks, which is a significant security concern. This work proposes a rotated binary large ResNet (RBLResNet) for AMC that can be deployed at the edge network because of low memory and computational complexity. The performance gap between the RBLResNet and existing architectures with floating-point weights and activations can be closed by two proposed ensemble methods: (i) multilevel classification (MC), and (ii) bagging multiple RBLResNets while retaining low memory and computational power. The MC method achieves an accuracy of $93.39\%$ at $10$dB over all the $24$ modulation classes of the Deepsig dataset. This performance is comparable to state-of-the-art (SOTA) performances, with $4.75$ times lower memory and $1214$ times lower computation. Furthermore, RBLResNet also has high adversarial robustness compared to existing DNN models. The proposed MC method with RBLResNets has an adversarial accuracy of $87.25\%$ over a wide range of SNRs, surpassing the robustness of all existing SOTA methods to the best of our knowledge. Properties such as low memory, low computation, and the highest adversarial robustness make it a better choice for robust AMC in low-power edge devices.

摘要: 近年来，深度神经网络(DNN)已被广泛应用于自动调制分类(AMC)，并取得了良好的效果。然而，DNN对内存和计算的要求很高，这使得它们不适用于设备资源有限的边缘网络。它们还容易受到敌意攻击，这是一个重大的安全问题。本文提出了一种适用于AMC的轮转二进制大型ResNet(RBLResNet)，该网络具有较低的存储容量和计算复杂度，可以部署在边缘网络中。RBLResNet与现有的具有浮点权重和激活的体系结构之间的性能差距可以通过两种提出的集成方法来缩小：(I)多级分类(MC)和(Ii)在保持低存储和计算能力的情况下打包多个RBLResNet。MC方法在Deepsig数据集的所有$24$调制类别上，在$10$分贝下实现了$93.39\$的精度。这一性能与最先进的(SOTA)性能相当，内存减少了4.75美元，计算减少了1214美元。此外，与现有的DNN模型相比，RBLResNet还具有较高的对抗健壮性。提出的基于RBLResNets的MC方法在较宽的信噪比范围内具有87.25美元的对抗精度，超过了现有SOTA方法的稳健性。低内存、低计算量和最高的对抗健壮性等特性使其成为低功率边缘设备中健壮AMC的更好选择。



## **35. An Unbiased Transformer Source Code Learning with Semantic Vulnerability Graph**

基于语义脆弱图的无偏转换器源代码学习 cs.CR

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2304.11072v1) [paper-pdf](http://arxiv.org/pdf/2304.11072v1)

**Authors**: Nafis Tanveer Islam, Gonzalo De La Torre Parra, Dylan Manuel, Elias Bou-Harb, Peyman Najafirad

**Abstract**: Over the years, open-source software systems have become prey to threat actors. Even as open-source communities act quickly to patch the breach, code vulnerability screening should be an integral part of agile software development from the beginning. Unfortunately, current vulnerability screening techniques are ineffective at identifying novel vulnerabilities or providing developers with code vulnerability and classification. Furthermore, the datasets used for vulnerability learning often exhibit distribution shifts from the real-world testing distribution due to novel attack strategies deployed by adversaries and as a result, the machine learning model's performance may be hindered or biased. To address these issues, we propose a joint interpolated multitasked unbiased vulnerability classifier comprising a transformer "RoBERTa" and graph convolution neural network (GCN). We present a training process utilizing a semantic vulnerability graph (SVG) representation from source code, created by integrating edges from a sequential flow, control flow, and data flow, as well as a novel flow dubbed Poacher Flow (PF). Poacher flow edges reduce the gap between dynamic and static program analysis and handle complex long-range dependencies. Moreover, our approach reduces biases of classifiers regarding unbalanced datasets by integrating Focal Loss objective function along with SVG. Remarkably, experimental results show that our classifier outperforms state-of-the-art results on vulnerability detection with fewer false negatives and false positives. After testing our model across multiple datasets, it shows an improvement of at least 2.41% and 18.75% in the best-case scenario. Evaluations using N-day program samples demonstrate that our proposed approach achieves a 93% accuracy and was able to detect 4, zero-day vulnerabilities from popular GitHub repositories.

摘要: 多年来，开源软件系统已成为威胁参与者的牺牲品。即使开源社区迅速采取行动修补漏洞，代码漏洞筛选也应该从一开始就是敏捷软件开发不可或缺的一部分。遗憾的是，当前的漏洞筛选技术在识别新的漏洞或向开发人员提供代码漏洞和分类方面无效。此外，由于对手采用了新的攻击策略，用于漏洞学习的数据集经常表现出与真实世界测试分布的分布偏移，因此，机器学习模型的性能可能会受到阻碍或偏差。为了解决这些问题，我们提出了一种联合内插多任务无偏漏洞分类器，该分类器由一个转换器“Roberta”和图卷积神经网络(GCN)组成。我们提出了一种训练过程，利用源代码中的语义脆弱图(SVG)表示，通过整合来自顺序流、控制流和数据流的边来创建，以及一种被称为Poacher流(PF)的新流。偷猎者流边缘缩小了动态和静态程序分析之间的差距，并处理复杂的远程依赖关系。此外，我们的方法通过集成焦点损失目标函数和SVG来减少分类器对不平衡数据集的偏差。值得注意的是，实验结果表明，我们的分类器在漏洞检测方面的性能优于最先进的结果，具有较少的漏报和误报。在多个数据集上测试我们的模型后，在最好的情况下，它显示出至少2.41%和18.75%的改进。使用N天程序样本进行的评估表明，我们提出的方法达到了93%的准确率，并能够从流行的GitHub存储库中检测到4个零天漏洞。



## **36. Employing Deep Ensemble Learning for Improving the Security of Computer Networks against Adversarial Attacks**

利用深度集成学习提高计算机网络抗敌意攻击的安全性 cs.CR

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2209.12195v2) [paper-pdf](http://arxiv.org/pdf/2209.12195v2)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Erkay Savas, Mauro Conti, Yassine Mekdad

**Abstract**: In the past few years, Convolutional Neural Networks (CNN) have demonstrated promising performance in various real-world cybersecurity applications, such as network and multimedia security. However, the underlying fragility of CNN structures poses major security problems, making them inappropriate for use in security-oriented applications including such computer networks. Protecting these architectures from adversarial attacks necessitates using security-wise architectures that are challenging to attack.   In this study, we present a novel architecture based on an ensemble classifier that combines the enhanced security of 1-Class classification (known as 1C) with the high performance of conventional 2-Class classification (known as 2C) in the absence of attacks.Our architecture is referred to as the 1.5-Class (SPRITZ-1.5C) classifier and constructed using a final dense classifier, one 2C classifier (i.e., CNNs), and two parallel 1C classifiers (i.e., auto-encoders). In our experiments, we evaluated the robustness of our proposed architecture by considering eight possible adversarial attacks in various scenarios. We performed these attacks on the 2C and SPRITZ-1.5C architectures separately. The experimental results of our study showed that the Attack Success Rate (ASR) of the I-FGSM attack against a 2C classifier trained with the N-BaIoT dataset is 0.9900. In contrast, the ASR is 0.0000 for the SPRITZ-1.5C classifier.

摘要: 在过去的几年里，卷积神经网络(CNN)在各种现实世界的网络安全应用中表现出了良好的性能，如网络和多媒体安全。然而，CNN结构的潜在脆弱性造成了重大的安全问题，使其不适合用于包括此类计算机网络在内的面向安全的应用程序。要保护这些架构免受敌意攻击，就必须使用具有安全性的架构，这些架构很难受到攻击。在这项研究中，我们提出了一种新的基于集成分类器的体系结构，它结合了1类分类(称为1C)的增强安全性和传统2类分类(称为2C)的高性能，在没有攻击的情况下被称为1.5类(SPRITZ-1.5C)分类器，并使用最终的密集分类器、一个2C分类器(即CNN)和两个并行的1C分类器(即自动编码器)来构建。在我们的实验中，我们通过考虑不同场景中八种可能的对抗性攻击来评估我们所提出的体系结构的健壮性。我们分别在2C和SPRITZ-1.5C架构上执行了这些攻击。实验结果表明，利用N-BaIoT数据集训练的2C分类器对I-FGSM的攻击成功率为0.9900。相比之下，SPRITZ-1.5C分级机的ASR为0.0000。



## **37. RNN-Guard: Certified Robustness Against Multi-frame Attacks for Recurrent Neural Networks**

RNN-Guard：递归神经网络对多帧攻击的验证鲁棒性 cs.LG

13 pages, 7 figures, 6 tables

**SubmitDate**: 2023-04-17    [abs](http://arxiv.org/abs/2304.07980v1) [paper-pdf](http://arxiv.org/pdf/2304.07980v1)

**Authors**: Yunruo Zhang, Tianyu Du, Shouling Ji, Peng Tang, Shanqing Guo

**Abstract**: It is well-known that recurrent neural networks (RNNs), although widely used, are vulnerable to adversarial attacks including one-frame attacks and multi-frame attacks. Though a few certified defenses exist to provide guaranteed robustness against one-frame attacks, we prove that defending against multi-frame attacks remains a challenging problem due to their enormous perturbation space. In this paper, we propose the first certified defense against multi-frame attacks for RNNs called RNN-Guard. To address the above challenge, we adopt the perturb-all-frame strategy to construct perturbation spaces consistent with those in multi-frame attacks. However, the perturb-all-frame strategy causes a precision issue in linear relaxations. To address this issue, we introduce a novel abstract domain called InterZono and design tighter relaxations. We prove that InterZono is more precise than Zonotope yet carries the same time complexity. Experimental evaluations across various datasets and model structures show that the certified robust accuracy calculated by RNN-Guard with InterZono is up to 2.18 times higher than that with Zonotope. In addition, we extend RNN-Guard as the first certified training method against multi-frame attacks to directly enhance RNNs' robustness. The results show that the certified robust accuracy of models trained with RNN-Guard against multi-frame attacks is 15.47 to 67.65 percentage points higher than those with other training methods.

摘要: 众所周知，递归神经网络(RNN)虽然被广泛使用，但很容易受到包括一帧攻击和多帧攻击在内的对抗性攻击。尽管存在一些经过认证的防御措施可以提供对单帧攻击的保证健壮性，但我们证明，由于多帧攻击的巨大扰动空间，防御多帧攻击仍然是一个具有挑战性的问题。在本文中，我们提出了第一个经过认证的针对RNN的多帧攻击防御机制RNN-Guard。为了应对上述挑战，我们采用扰动全帧策略来构造与多帧攻击中一致的扰动空间。然而，扰动全框架策略会导致线性松弛中的精度问题。为了解决这个问题，我们引入了一个新的抽象域InterZono，并设计了更紧密的松弛。我们证明了InterZono比Zonotope更精确，但具有相同的时间复杂度。在不同的数据集和模型结构上的实验评估表明，RNN-Guard使用InterZono计算的健壮性精度比使用Zonotope计算的健壮性高2.18倍。此外，我们扩展了RNN-Guard作为第一种经过认证的针对多帧攻击的训练方法，以直接增强RNN的健壮性。结果表明，用RNN-Guard训练的模型对多帧攻击的鲁棒性比其他训练方法高15.47~67.65个百分点。



## **38. A Review of Speech-centric Trustworthy Machine Learning: Privacy, Safety, and Fairness**

以语音为中心的可信机器学习：隐私、安全和公平 cs.SD

**SubmitDate**: 2023-04-16    [abs](http://arxiv.org/abs/2212.09006v2) [paper-pdf](http://arxiv.org/pdf/2212.09006v2)

**Authors**: Tiantian Feng, Rajat Hebbar, Nicholas Mehlman, Xuan Shi, Aditya Kommineni, and Shrikanth Narayanan

**Abstract**: Speech-centric machine learning systems have revolutionized many leading domains ranging from transportation and healthcare to education and defense, profoundly changing how people live, work, and interact with each other. However, recent studies have demonstrated that many speech-centric ML systems may need to be considered more trustworthy for broader deployment. Specifically, concerns over privacy breaches, discriminating performance, and vulnerability to adversarial attacks have all been discovered in ML research fields. In order to address the above challenges and risks, a significant number of efforts have been made to ensure these ML systems are trustworthy, especially private, safe, and fair. In this paper, we conduct the first comprehensive survey on speech-centric trustworthy ML topics related to privacy, safety, and fairness. In addition to serving as a summary report for the research community, we point out several promising future research directions to inspire the researchers who wish to explore further in this area.

摘要: 以语音为中心的机器学习系统已经彻底改变了从交通和医疗到教育和国防等许多领先领域，深刻地改变了人们的生活、工作和相互互动的方式。然而，最近的研究表明，许多以语音为中心的ML系统可能需要被认为更值得信赖，以便更广泛地部署。具体地说，在ML研究领域中发现了对隐私泄露、区分性能和对对手攻击的脆弱性的担忧。为了应对上述挑战和风险，已经做出了大量努力，以确保这些ML系统是值得信任的，特别是私密、安全和公平。在本文中，我们首次对以语音为中心的可信ML话题进行了全面的调查，这些话题涉及隐私、安全和公平。除了作为研究界的总结报告外，我们还指出了几个有前途的未来研究方向，以激励希望在这一领域进一步探索的研究人员。



## **39. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

ICASSP 2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2210.06284v3) [paper-pdf](http://arxiv.org/pdf/2210.06284v3)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **40. XploreNAS: Explore Adversarially Robust & Hardware-efficient Neural Architectures for Non-ideal Xbars**

XploreNAS：探索针对非理想Xbar的强大且硬件高效的神经体系结构 cs.LG

Accepted to ACM Transactions on Embedded Computing Systems in April  2023

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2302.07769v2) [paper-pdf](http://arxiv.org/pdf/2302.07769v2)

**Authors**: Abhiroop Bhattacharjee, Abhishek Moitra, Priyadarshini Panda

**Abstract**: Compute In-Memory platforms such as memristive crossbars are gaining focus as they facilitate acceleration of Deep Neural Networks (DNNs) with high area and compute-efficiencies. However, the intrinsic non-idealities associated with the analog nature of computing in crossbars limits the performance of the deployed DNNs. Furthermore, DNNs are shown to be vulnerable to adversarial attacks leading to severe security threats in their large-scale deployment. Thus, finding adversarially robust DNN architectures for non-ideal crossbars is critical to the safe and secure deployment of DNNs on the edge. This work proposes a two-phase algorithm-hardware co-optimization approach called XploreNAS that searches for hardware-efficient & adversarially robust neural architectures for non-ideal crossbar platforms. We use the one-shot Neural Architecture Search (NAS) approach to train a large Supernet with crossbar-awareness and sample adversarially robust Subnets therefrom, maintaining competitive hardware-efficiency. Our experiments on crossbars with benchmark datasets (SVHN, CIFAR10 & CIFAR100) show upto ~8-16% improvement in the adversarial robustness of the searched Subnets against a baseline ResNet-18 model subjected to crossbar-aware adversarial training. We benchmark our robust Subnets for Energy-Delay-Area-Products (EDAPs) using the Neurosim tool and find that with additional hardware-efficiency driven optimizations, the Subnets attain ~1.5-1.6x lower EDAPs than ResNet-18 baseline.

摘要: 内存交叉开关等计算内存平台因其高面积和高计算效率促进深度神经网络(DNN)的加速而备受关注。然而，与交叉开关中计算的模拟性质相关联的固有非理想性限制了所部署的DNN的性能。此外，DNN在大规模部署时容易受到对抗性攻击，从而造成严重的安全威胁。因此，为非理想的交叉开关找到相对健壮的DNN架构对于在边缘安全地部署DNN至关重要。该工作提出了一种称为XploreNAS的两阶段算法-硬件联合优化方法，该方法为非理想的纵横制平台寻找硬件高效且相对健壮的神经结构。我们使用一次神经体系结构搜索(NAS)方法来训练一个具有交叉开关感知的大型超网，并从中采样具有敌意健壮性的子网，从而保持具有竞争力的硬件效率。我们使用基准数据集(SVHN、CIFAR10和CIFAR100)在Crosbar上的实验表明，相对于接受Crosbar感知对抗性训练的基线ResNet-18模型，搜索到的子网的对抗健壮性提高了约8-16%。我们使用Neurosim工具对我们的健壮的能量延迟面积产品(EDAP)子网进行了基准测试，发现在额外的硬件效率驱动的优化下，这些子网获得的EDAP比ResNet-18基准低约1.5-1.6倍。



## **41. Visually Adversarial Attacks and Defenses in the Physical World: A Survey**

物理世界中的视觉对抗性攻击和防御：综述 cs.CV

**SubmitDate**: 2023-04-15    [abs](http://arxiv.org/abs/2211.01671v4) [paper-pdf](http://arxiv.org/pdf/2211.01671v4)

**Authors**: Xingxing Wei, Bangzheng Pu, Jiefan Lu, Baoyuan Wu

**Abstract**: Although Deep Neural Networks (DNNs) have been widely applied in various real-world scenarios, they are vulnerable to adversarial examples. The current adversarial attacks in computer vision can be divided into digital attacks and physical attacks according to their different attack forms. Compared with digital attacks, which generate perturbations in the digital pixels, physical attacks are more practical in the real world. Owing to the serious security problem caused by physically adversarial examples, many works have been proposed to evaluate the physically adversarial robustness of DNNs in the past years. In this paper, we summarize a survey versus the current physically adversarial attacks and physically adversarial defenses in computer vision. To establish a taxonomy, we organize the current physical attacks from attack tasks, attack forms, and attack methods, respectively. Thus, readers can have a systematic knowledge of this topic from different aspects. For the physical defenses, we establish the taxonomy from pre-processing, in-processing, and post-processing for the DNN models to achieve full coverage of the adversarial defenses. Based on the above survey, we finally discuss the challenges of this research field and further outlook on the future direction.

摘要: 尽管深度神经网络(DNN)已被广泛应用于各种现实场景中，但它们很容易受到对手例子的影响。根据攻击形式的不同，目前计算机视觉中的对抗性攻击可分为数字攻击和物理攻击。与在数字像素中产生扰动的数字攻击相比，物理攻击在现实世界中更实用。由于物理对抗实例带来了严重的安全问题，在过去的几年里，人们已经提出了许多工作来评估DNN的物理对抗健壮性。本文对当前计算机视觉中的身体对抗攻击和身体对抗防御进行了综述。为了建立分类，我们分别从攻击任务、攻击形式和攻击方法三个方面对当前的物理攻击进行了组织。因此，读者可以从不同的方面对这一主题有一个系统的了解。对于物理防御，我们从DNN模型的前处理、内处理和后处理三个方面建立了分类，以实现对抗性防御的全覆盖。在上述调查的基础上，我们最后讨论了该研究领域面临的挑战和对未来方向的进一步展望。



## **42. Combining Generators of Adversarial Malware Examples to Increase Evasion Rate**

组合敌意恶意软件示例的生成器以提高逃避率 cs.CR

9 pages, 5 figures, 2 tables. Under review

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07360v1) [paper-pdf](http://arxiv.org/pdf/2304.07360v1)

**Authors**: Matouš Kozák, Martin Jureček

**Abstract**: Antivirus developers are increasingly embracing machine learning as a key component of malware defense. While machine learning achieves cutting-edge outcomes in many fields, it also has weaknesses that are exploited by several adversarial attack techniques. Many authors have presented both white-box and black-box generators of adversarial malware examples capable of bypassing malware detectors with varying success. We propose to combine contemporary generators in order to increase their potential. Combining different generators can create more sophisticated adversarial examples that are more likely to evade anti-malware tools. We demonstrated this technique on five well-known generators and recorded promising results. The best-performing combination of AMG-random and MAB-Malware generators achieved an average evasion rate of 15.9% against top-tier antivirus products. This represents an average improvement of more than 36% and 627% over using only the AMG-random and MAB-Malware generators, respectively. The generator that benefited the most from having another generator follow its procedure was the FGSM injection attack, which improved the evasion rate on average between 91.97% and 1,304.73%, depending on the second generator used. These results demonstrate that combining different generators can significantly improve their effectiveness against leading antivirus programs.

摘要: 反病毒开发人员越来越多地将机器学习作为恶意软件防御的关键组件。虽然机器学习在许多领域取得了尖端成果，但它也有被几种对抗性攻击技术利用的弱点。许多作者都提出了敌意恶意软件的白盒和黑盒生成器的例子，这些例子能够绕过恶意软件检测器，取得了不同的成功。我们建议将当代发电机结合起来，以增加它们的潜力。组合不同的生成器可以创建更复杂的敌意示例，这些示例更有可能逃避反恶意软件工具。我们在五个著名的发电机上演示了这项技术，并记录了有希望的结果。AMG-RANDOM和MAB-恶意软件生成器的最佳组合对顶级反病毒产品的平均逃避率为15.9%。这意味着分别比只使用AMG-RANDOM和MAB-恶意软件生成器的平均性能提高了36%和627%以上。从让另一个生成器遵循其过程中受益最大的生成器是FGSM注入攻击，它根据使用的第二个生成器的不同，将逃避率平均提高了91.97%到1304.73%。这些结果表明，组合不同的生成器可以显著提高它们对抗领先反病毒程序的有效性。



## **43. Pool Inference Attacks on Local Differential Privacy: Quantifying the Privacy Guarantees of Apple's Count Mean Sketch in Practice**

局部差分隐私的Pool推理攻击：Apple计数均值素描的隐私保证量化实践 cs.CR

Published at USENIX Security 2022. This is the full version, please  cite the USENIX version (see journal reference field)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.07134v1) [paper-pdf](http://arxiv.org/pdf/2304.07134v1)

**Authors**: Andrea Gadotti, Florimond Houssiau, Meenatchi Sundaram Muthu Selva Annamalai, Yves-Alexandre de Montjoye

**Abstract**: Behavioral data generated by users' devices, ranging from emoji use to pages visited, are collected at scale to improve apps and services. These data, however, contain fine-grained records and can reveal sensitive information about individual users. Local differential privacy has been used by companies as a solution to collect data from users while preserving privacy. We here first introduce pool inference attacks, where an adversary has access to a user's obfuscated data, defines pools of objects, and exploits the user's polarized behavior in multiple data collections to infer the user's preferred pool. Second, we instantiate this attack against Count Mean Sketch, a local differential privacy mechanism proposed by Apple and deployed in iOS and Mac OS devices, using a Bayesian model. Using Apple's parameters for the privacy loss $\varepsilon$, we then consider two specific attacks: one in the emojis setting -- where an adversary aims at inferring a user's preferred skin tone for emojis -- and one against visited websites -- where an adversary wants to learn the political orientation of a user from the news websites they visit. In both cases, we show the attack to be much more effective than a random guess when the adversary collects enough data. We find that users with high polarization and relevant interest are significantly more vulnerable, and we show that our attack is well-calibrated, allowing the adversary to target such vulnerable users. We finally validate our results for the emojis setting using user data from Twitter. Taken together, our results show that pool inference attacks are a concern for data protected by local differential privacy mechanisms with a large $\varepsilon$, emphasizing the need for additional technical safeguards and the need for more research on how to apply local differential privacy for multiple collections.

摘要: 用户设备产生的行为数据，从表情符号的使用到访问的页面，都会大规模收集，以改进应用程序和服务。然而，这些数据包含细粒度的记录，可能会泄露有关个人用户的敏感信息。本地差异隐私已被公司用作在保护隐私的同时收集用户数据的解决方案。这里我们首先引入池推理攻击，在这种攻击中，敌手可以访问用户的混淆数据，定义对象池，并利用用户在多个数据集合中的两极分化行为来推断用户的首选池。其次，我们使用贝叶斯模型实例化了针对Count Mean Sketch的攻击，Count Mean Sketch是Apple提出的一种本地差异隐私机制，部署在iOS和Mac OS设备上。使用苹果的隐私损失参数，然后我们考虑了两种具体的攻击：一种是在表情符号设置中--对手的目标是推断用户喜欢的表情符号的肤色--另一种是针对访问的网站--对手想要从用户访问的新闻网站了解用户的政治倾向。在这两种情况下，我们都表明，当对手收集到足够的数据时，攻击比随机猜测要有效得多。我们发现，具有高极化和相关兴趣的用户明显更容易受到攻击，并且我们表明我们的攻击经过了很好的校准，允许对手针对这些易受攻击的用户。最后，我们使用Twitter上的用户数据验证了表情符号设置的结果。总而言之，我们的结果表明，池推理攻击是对受本地差异隐私机制保护的数据的关注，需要花费很大的$\varepsilon$，这强调了需要额外的技术保障，以及需要更多的研究如何将本地差异隐私应用于多个集合。



## **44. Interpretability is a Kind of Safety: An Interpreter-based Ensemble for Adversary Defense**

可解释性是一种安全：一种基于口译员的对手防御合奏 cs.LG

10 pages, accepted to KDD'20

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06919v1) [paper-pdf](http://arxiv.org/pdf/2304.06919v1)

**Authors**: Jingyuan Wang, Yufan Wu, Mingxuan Li, Xin Lin, Junjie Wu, Chao Li

**Abstract**: While having achieved great success in rich real-life applications, deep neural network (DNN) models have long been criticized for their vulnerability to adversarial attacks. Tremendous research efforts have been dedicated to mitigating the threats of adversarial attacks, but the essential trait of adversarial examples is not yet clear, and most existing methods are yet vulnerable to hybrid attacks and suffer from counterattacks. In light of this, in this paper, we first reveal a gradient-based correlation between sensitivity analysis-based DNN interpreters and the generation process of adversarial examples, which indicates the Achilles's heel of adversarial attacks and sheds light on linking together the two long-standing challenges of DNN: fragility and unexplainability. We then propose an interpreter-based ensemble framework called X-Ensemble for robust adversary defense. X-Ensemble adopts a novel detection-rectification process and features in building multiple sub-detectors and a rectifier upon various types of interpretation information toward target classifiers. Moreover, X-Ensemble employs the Random Forests (RF) model to combine sub-detectors into an ensemble detector for adversarial hybrid attacks defense. The non-differentiable property of RF further makes it a precious choice against the counterattack of adversaries. Extensive experiments under various types of state-of-the-art attacks and diverse attack scenarios demonstrate the advantages of X-Ensemble to competitive baseline methods.

摘要: 尽管深度神经网络(DNN)模型在丰富的现实应用中取得了巨大的成功，但长期以来一直因其易受对手攻击而受到批评。为了缓解对抗性攻击的威胁，人们进行了大量的研究工作，但对抗性例子的本质特征尚不清楚，而且现有的大多数方法仍然容易受到混合攻击和反击。有鉴于此，本文首先揭示了基于敏感度分析的DNN解释器与敌意实例生成过程之间的梯度关系，这表明了敌意攻击的致命弱点，并有助于将DNN的两个长期挑战：脆弱性和不可解释性联系在一起。然后，我们提出了一个基于解释器的集成框架，称为X-集成，用于强健的对手防御。X-Ensymble采用了一种新颖的检测-纠正过程，其特点是根据目标分类器的各种解释信息构建多个子检测器和一个校正器。此外，X-EnSemble采用随机森林(RF)模型将子检测器组合为集成检测器，用于对抗性混合攻击防御。射频的不可微性进一步使其成为对抗对手反击的宝贵选择。在各种最先进的攻击和不同的攻击场景下进行的广泛实验证明了X-Ensymble相对于竞争基线方法的优势。



## **45. Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms**

稳健的多变量时间序列预测：对抗性攻击与防御机制 cs.LG

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2207.09572v3) [paper-pdf](http://arxiv.org/pdf/2207.09572v3)

**Authors**: Linbo Liu, Youngsuk Park, Trong Nghia Hoang, Hilaf Hasson, Jun Huan

**Abstract**: This work studies the threats of adversarial attack on multivariate probabilistic forecasting models and viable defense mechanisms. Our studies discover a new attack pattern that negatively impact the forecasting of a target time series via making strategic, sparse (imperceptible) modifications to the past observations of a small number of other time series. To mitigate the impact of such attack, we have developed two defense strategies. First, we extend a previously developed randomized smoothing technique in classification to multivariate forecasting scenarios. Second, we develop an adversarial training algorithm that learns to create adversarial examples and at the same time optimizes the forecasting model to improve its robustness against such adversarial simulation. Extensive experiments on real-world datasets confirm that our attack schemes are powerful and our defense algorithms are more effective compared with baseline defense mechanisms.

摘要: 本文研究了对抗性攻击对多变量概率预测模型的威胁和可行的防御机制。我们的研究发现了一种新的攻击模式，它通过对过去对少量其他时间序列的观察进行战略性的稀疏(不可察觉的)修改，对目标时间序列的预测产生负面影响。为了减轻此类攻击的影响，我们制定了两种防御策略。首先，我们将以前开发的随机平滑分类技术扩展到多变量预测场景。其次，我们开发了一种对抗性训练算法，该算法学习创建对抗性示例，同时优化预测模型，以提高其对此类对抗性模拟的健壮性。在真实数据集上的广泛实验证实了我们的攻击方案是强大的，我们的防御算法比基线防御机制更有效。



## **46. Generating Adversarial Examples with Better Transferability via Masking Unimportant Parameters of Surrogate Model**

通过屏蔽代理模型的不重要参数生成可移植性更好的对抗性实例 cs.LG

Accepted at 2023 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2023-04-14    [abs](http://arxiv.org/abs/2304.06908v1) [paper-pdf](http://arxiv.org/pdf/2304.06908v1)

**Authors**: Dingcheng Yang, Wenjian Yu, Zihao Xiao, Jiaqi Luo

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples. Moreover, the transferability of the adversarial examples has received broad attention in recent years, which means that adversarial examples crafted by a surrogate model can also attack unknown models. This phenomenon gave birth to the transfer-based adversarial attacks, which aim to improve the transferability of the generated adversarial examples. In this paper, we propose to improve the transferability of adversarial examples in the transfer-based attack via masking unimportant parameters (MUP). The key idea in MUP is to refine the pretrained surrogate models to boost the transfer-based attack. Based on this idea, a Taylor expansion-based metric is used to evaluate the parameter importance score and the unimportant parameters are masked during the generation of adversarial examples. This process is simple, yet can be naturally combined with various existing gradient-based optimizers for generating adversarial examples, thus further improving the transferability of the generated adversarial examples. Extensive experiments are conducted to validate the effectiveness of the proposed MUP-based methods.

摘要: 深度神经网络(DNN)已被证明容易受到敌意例子的攻击。此外，对抗性实例的可转移性近年来受到了广泛的关注，这意味着由代理模型构造的对抗性实例也可以攻击未知模型。这一现象催生了基于迁移的对抗性攻击，其目的是提高生成的对抗性实例的可转移性。在本文中，我们提出了通过掩蔽不重要参数(MUP)来提高基于传输的攻击中敌意实例的可转移性。MUP的关键思想是改进预先训练的代理模型，以增强基于传输的攻击。基于这一思想，使用基于泰勒展开的度量来评估参数重要性得分，并在对抗性实例的生成过程中屏蔽不重要的参数。这个过程简单，但可以自然地与现有的各种基于梯度的优化器相结合来生成对抗性实例，从而进一步提高生成的对抗性实例的可转移性。大量的实验验证了所提出的基于MUP的方法的有效性。



## **47. Don't Knock! Rowhammer at the Backdoor of DNN Models**

别敲门！DNN模型的后门Rowhammer cs.LG

2023 53rd Annual IEEE/IFIP International Conference on Dependable  Systems and Networks (DSN)

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2110.07683v3) [paper-pdf](http://arxiv.org/pdf/2110.07683v3)

**Authors**: M. Caner Tol, Saad Islam, Andrew J. Adiletta, Berk Sunar, Ziming Zhang

**Abstract**: State-of-the-art deep neural networks (DNNs) have been proven to be vulnerable to adversarial manipulation and backdoor attacks. Backdoored models deviate from expected behavior on inputs with predefined triggers while retaining performance on clean data. Recent works focus on software simulation of backdoor injection during the inference phase by modifying network weights, which we find often unrealistic in practice due to restrictions in hardware.   In contrast, in this work for the first time, we present an end-to-end backdoor injection attack realized on actual hardware on a classifier model using Rowhammer as the fault injection method. To this end, we first investigate the viability of backdoor injection attacks in real-life deployments of DNNs on hardware and address such practical issues in hardware implementation from a novel optimization perspective. We are motivated by the fact that vulnerable memory locations are very rare, device-specific, and sparsely distributed. Consequently, we propose a novel network training algorithm based on constrained optimization to achieve a realistic backdoor injection attack in hardware. By modifying parameters uniformly across the convolutional and fully-connected layers as well as optimizing the trigger pattern together, we achieve state-of-the-art attack performance with fewer bit flips. For instance, our method on a hardware-deployed ResNet-20 model trained on CIFAR-10 achieves over 89% test accuracy and 92% attack success rate by flipping only 10 out of 2.2 million bits.

摘要: 最先进的深度神经网络(DNN)已被证明容易受到对手操纵和后门攻击。回溯模型偏离了具有预定义触发器的输入的预期行为，同时保持了对干净数据的性能。最近的工作集中在通过修改网络权重来模拟推理阶段的后门注入，但由于硬件的限制，这在实践中往往是不现实的。相反，在这项工作中，我们首次提出了一种端到端的后门注入攻击，在以Rowhammer作为故障注入方法的分类器模型上在实际硬件上实现。为此，我们首先研究了DNN在硬件上的实际部署中后门注入攻击的可行性，并从一个新的优化角度解决了这些硬件实现中的实际问题。我们的动机是，易受攻击的内存位置非常罕见，特定于设备，并且分布稀疏。因此，我们提出了一种新的基于约束优化的网络训练算法，在硬件上实现了逼真的后门注入攻击。通过统一修改卷积层和全连通层的参数，以及一起优化触发模式，我们以更少的比特翻转实现了最先进的攻击性能。例如，我们的方法在硬件部署的ResNet-20模型上训练CIFAR-10，通过在220万比特中仅翻转10比特，获得了超过89%的测试准确率和92%的攻击成功率。



## **48. PowerGAN: A Machine Learning Approach for Power Side-Channel Attack on Compute-in-Memory Accelerators**

PowerGAN：一种针对内存计算加速器电源侧通道攻击的机器学习方法 cs.CR

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.11056v1) [paper-pdf](http://arxiv.org/pdf/2304.11056v1)

**Authors**: Ziyu Wang, Yuting Wu, Yongmo Park, Sangmin Yoo, Xinxin Wang, Jason K. Eshraghian, Wei D. Lu

**Abstract**: Analog compute-in-memory (CIM) accelerators are becoming increasingly popular for deep neural network (DNN) inference due to their energy efficiency and in-situ vector-matrix multiplication (VMM) capabilities. However, as the use of DNNs expands, protecting user input privacy has become increasingly important. In this paper, we identify a security vulnerability wherein an adversary can reconstruct the user's private input data from a power side-channel attack, under proper data acquisition and pre-processing, even without knowledge of the DNN model. We further demonstrate a machine learning-based attack approach using a generative adversarial network (GAN) to enhance the reconstruction. Our results show that the attack methodology is effective in reconstructing user inputs from analog CIM accelerator power leakage, even when at large noise levels and countermeasures are applied. Specifically, we demonstrate the efficacy of our approach on the U-Net for brain tumor detection in magnetic resonance imaging (MRI) medical images, with a noise-level of 20% standard deviation of the maximum power signal value. Our study highlights a significant security vulnerability in analog CIM accelerators and proposes an effective attack methodology using a GAN to breach user privacy.

摘要: 模拟内存中计算(CIM)加速器因其能量效率和原位向量矩阵乘法(VMM)能力而越来越多地用于深度神经网络(DNN)推理。然而，随着DNN的使用范围扩大，保护用户输入隐私变得越来越重要。在本文中，我们发现了一个安全漏洞，在该漏洞中，攻击者可以在不了解DNN模型的情况下，在适当的数据获取和预处理下，从功率侧通道攻击中重构用户的私人输入数据。我们进一步展示了一种基于机器学习的攻击方法，使用生成性对抗网络(GAN)来增强重建。我们的结果表明，即使在高噪声水平和对抗措施的情况下，该攻击方法也能有效地从模拟CIM加速器功率泄漏中恢复用户输入。具体地说，我们展示了我们的方法在U网络上的有效性，用于磁共振成像(MRI)医学图像中的脑肿瘤检测，噪声水平为最大功率信号值的20%标准差。我们的研究突出了模拟CIM加速器中的一个严重的安全漏洞，并提出了一种使用GAN来破坏用户隐私的有效攻击方法。



## **49. False Claims against Model Ownership Resolution**

针对所有权解决方案范本的虚假索赔 cs.CR

13pages,3 figures

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.06607v1) [paper-pdf](http://arxiv.org/pdf/2304.06607v1)

**Authors**: Jian Liu, Rui Zhang, Sebastian Szyller, Kui Ren, N. Asokan

**Abstract**: Deep neural network (DNN) models are valuable intellectual property of model owners, constituting a competitive advantage. Therefore, it is crucial to develop techniques to protect against model theft. Model ownership resolution (MOR) is a class of techniques that can deter model theft. A MOR scheme enables an accuser to assert an ownership claim for a suspect model by presenting evidence, such as a watermark or fingerprint, to show that the suspect model was stolen or derived from a source model owned by the accuser. Most of the existing MOR schemes prioritize robustness against malicious suspects, ensuring that the accuser will win if the suspect model is indeed a stolen model.   In this paper, we show that common MOR schemes in the literature are vulnerable to a different, equally important but insufficiently explored, robustness concern: a malicious accuser. We show how malicious accusers can successfully make false claims against independent suspect models that were not stolen. Our core idea is that a malicious accuser can deviate (without detection) from the specified MOR process by finding (transferable) adversarial examples that successfully serve as evidence against independent suspect models. To this end, we first generalize the procedures of common MOR schemes and show that, under this generalization, defending against false claims is as challenging as preventing (transferable) adversarial examples. Via systematic empirical evaluation we demonstrate that our false claim attacks always succeed in all prominent MOR schemes with realistic configurations, including against a real-world model: Amazon's Rekognition API.

摘要: 深度神经网络(DNN)模型是模型所有者宝贵的知识产权，构成了竞争优势。因此，开发防止模型盗窃的技术至关重要。模型所有权解析(MOR)是一类能够阻止模型被盗的技术。MOR方案使原告能够通过出示证据(例如水印或指纹)来断言可疑模型的所有权主张，以显示可疑模型是被盗的或从原告拥有的源模型派生的。现有的大多数MOR方案都优先考虑针对恶意嫌疑人的健壮性，确保在可疑模型确实是被盗模型的情况下原告获胜。在这篇文章中，我们证明了文献中常见的MOR方案容易受到另一个同样重要但未被充分研究的健壮性问题的影响：恶意指控者。我们展示了恶意原告如何成功地对未被窃取的独立可疑模型做出虚假声明。我们的核心思想是，恶意指控者可以通过找到(可转移的)对抗性例子来偏离指定的MOR过程(而不被检测到)，这些例子成功地充当了针对独立嫌疑人模型的证据。为此，我们首先推广了常见MOR方案的步骤，并证明在这种推广下，对虚假声明的防御与防止(可转移)对抗性例子一样具有挑战性。通过系统的经验评估，我们证明了我们的虚假声明攻击在所有具有现实配置的著名MOR方案中总是成功的，包括针对真实世界的模型：亚马逊的Rekognition API。



## **50. EGC: Image Generation and Classification via a Diffusion Energy-Based Model**

EGC：基于扩散能量模型的图像生成和分类 cs.CV

**SubmitDate**: 2023-04-13    [abs](http://arxiv.org/abs/2304.02012v3) [paper-pdf](http://arxiv.org/pdf/2304.02012v3)

**Authors**: Qiushan Guo, Chuofan Ma, Yi Jiang, Zehuan Yuan, Yizhou Yu, Ping Luo

**Abstract**: Learning image classification and image generation using the same set of network parameters is a challenging problem. Recent advanced approaches perform well in one task often exhibit poor performance in the other. This work introduces an energy-based classifier and generator, namely EGC, which can achieve superior performance in both tasks using a single neural network. Unlike a conventional classifier that outputs a label given an image (i.e., a conditional distribution $p(y|\mathbf{x})$), the forward pass in EGC is a classifier that outputs a joint distribution $p(\mathbf{x},y)$, enabling an image generator in its backward pass by marginalizing out the label $y$. This is done by estimating the energy and classification probability given a noisy image in the forward pass, while denoising it using the score function estimated in the backward pass. EGC achieves competitive generation results compared with state-of-the-art approaches on ImageNet-1k, CelebA-HQ and LSUN Church, while achieving superior classification accuracy and robustness against adversarial attacks on CIFAR-10. This work represents the first successful attempt to simultaneously excel in both tasks using a single set of network parameters. We believe that EGC bridges the gap between discriminative and generative learning.

摘要: 使用相同的网络参数学习图像分类和图像生成是一个具有挑战性的问题。最近的高级方法在一项任务中表现良好，但在另一项任务中往往表现不佳。这项工作介绍了一种基于能量的分类器和生成器，即EGC，它可以使用单个神经网络在这两个任务中获得优越的性能。与输出给定图像的标签(即，条件分布$p(y|\mathbf{x})$)的传统分类器不同，EGC中的前向通道是输出联合分布$p(\mathbf{x}，y)$的分类器，从而使图像生成器在其后向通道中通过边缘化标签$y$来实现。这是通过在前传中给出噪声图像的情况下估计能量和分类概率来完成的，同时使用在后传中估计的得分函数来对其进行去噪。EGC在ImageNet-1k、CelebA-HQ和LSUN Church上实现了与最先进的方法相比具有竞争力的生成结果，同时在CIFAR-10上获得了卓越的分类准确性和对对手攻击的稳健性。这项工作代表了首次成功尝试使用一组网络参数同时在两项任务中脱颖而出。我们认为，EGC弥合了歧视性学习和生成性学习之间的差距。



