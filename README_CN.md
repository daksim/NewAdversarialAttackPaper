# Latest Adversarial Attack Papers
**update at 2022-08-12 21:24:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

MulVAL扩展及其攻击场景覆盖研究综述 cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.

摘要: 组织使用各种对手模型来评估攻击对其网络的风险和潜在影响。攻击图表示攻击者可以采取的漏洞和行动，以识别和危害组织的资产。攻击图便于以攻击路径的形式对攻击场景进行可视化呈现和算法分析。MulVAL是一个用于构建逻辑攻击图的通用开源框架，已经被研究人员和实践者广泛使用，并被他们用额外的攻击场景进行扩展。本文综述了现有的所有MulVAL扩展，并将所有的MulVAL交互规则映射到MITRE ATT&CK技术，以估计它们的攻击场景覆盖率。这项调查将当前的MulVAL扩展与统一的本体概念保持一致，并强调了存在的差距。它为MulVAL的系统改进和对MITRE ATT&CK捕获的敌对行为的整个场景的全面建模铺平了道路。



## **2. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

注意空间上可转移对抗性攻击的不同生成性对抗性扰动 cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.

摘要: 具有改进的可转移性的对抗性攻击--在已知模型上制作的对抗性例子也能够愚弄未知模型的能力--由于其实用性最近受到了极大的关注。然而，现有的可转移攻击以确定性的方式制造扰动，往往不能充分探索损失曲面，从而陷入较差的局部最优，且可转移性较低。为了解决这一问题，我们提出了注意力多样性攻击(ADA)，它以随机的方式破坏不同的显著特征，以提高可转移性。首先，我们扰乱图像注意力，以扰乱不同模型共享的通用特征。然后，为了有效地避免局部最优，我们以随机的方式破坏了这些特征，并更详尽地探索了可转移扰动的搜索空间。更具体地说，我们使用生成器来产生对抗性扰动，每个扰动都以不同的方式干扰特征，具体取决于输入的潜在代码。广泛的实验评估表明，我们的方法是有效的，超过了最先进的方法的可转移性。有关代码，请访问https://github.com/wkim97/ADA.



## **3. Controlled Quantum Teleportation in the Presence of an Adversary**

对手在场时的受控量子隐形传态 quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.

摘要: 在接收者不可信任的情况下，我们提出了受控量子隐形传态的设备无关分析。我们证明了真正的三方非局部性的概念允许我们在这种情况下证明控制权。通过考虑具有去极化噪声特征的设备上的特定对抗攻击策略，我们发现控制功率是真三方非定域性的单调递增函数。这些结果对构建实用的量子通信网络具有重要意义，也有助于揭示非定域性在多体量子信息处理中的作用。



## **4. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

Pikachu：通过使用Taproot检查点进入比特币PoW来保护PoS区块链免受远程攻击 cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.

摘要: 基于可重用资源的区块链系统，如风险证明(POS)，提供的安全保证比基于工作证明的系统更弱。具体地说，它们容易受到远程攻击，在远程攻击中，对手可以破坏之前的参与者，以便重写链的完整历史。为了防止这种对PoS链的攻击，我们提出了一种协议，将PoS链的状态检查到工作证明区块链，如比特币。因此，我们的检查点协议不依赖于任何中央机构。我们的工作使用Schnorr签名并利用比特币最近的Taproot升级，使我们能够创建恒定大小的检查点交易。我们为协议的安全性进行了论证，并给出了一个在比特币测试网上进行测试的开源实现。



## **5. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2202.07568v2)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了StratDef，这是一个针对恶意软件检测领域定制的基于运动目标防御方法的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御措施来看，只有少数经过对抗性训练的模型提供了比只使用普通模型更好的保护，但仍然优于StratDef。



## **6. Reducing Exploitability with Population Based Training**

通过基于人口的培训减少可利用性 cs.LG

Presented at New Frontiers in Adversarial Machine Learning Workshop,  ICML 2022

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05083v1)

**Authors**: Pavel Czempin, Adam Gleave

**Abstracts**: Self-play reinforcement learning has achieved state-of-the-art, and often superhuman, performance in a variety of zero-sum games. Yet prior work has found that policies that are highly capable against regular opponents can fail catastrophically against adversarial policies: an opponent trained explicitly against the victim. Prior defenses using adversarial training were able to make the victim robust to a specific adversary, but the victim remained vulnerable to new ones. We conjecture this limitation was due to insufficient diversity of adversaries seen during training. We propose a defense using population based training to pit the victim against a diverse set of opponents. We evaluate this defense's robustness against new adversaries in two low-dimensional environments. Our defense increases robustness against adversaries, as measured by number of attacker training timesteps to exploit the victim. Furthermore, we show that robustness is correlated with the size of the opponent population.

摘要: 自我发挥强化学习在各种零和游戏中实现了最先进的，往往是超人的表现。然而，先前的工作已经发现，对常规对手具有高度能力的政策，可能会在对抗对手的政策上灾难性地失败：一个明确针对受害者的对手。使用对抗性训练的先前防御能够使受害者对特定的对手变得健壮，但受害者仍然容易受到新对手的攻击。我们推测，这一限制是由于训练过程中看到的对手多样性不足所致。我们建议使用基于人口的训练来防御，让受害者与不同的对手对抗。我们在两个低维环境中评估了该防御对新对手的健壮性。我们的防御提高了对抗对手的健壮性，这是通过攻击者训练时间步数来衡量的，以利用受害者。此外，我们还证明了健壮性与对手种群的大小相关。



## **7. Adversarial Machine Learning-Based Anticipation of Threats Against Vehicle-to-Microgrid Services**

基于对抗性机器学习的车辆到微电网服务威胁预测 cs.CR

IEEE Global Communications Conference (Globecom), 2022, 6 pages, 2  Figures, 4 Tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2208.05073v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstracts**: In this paper, we study the expanding attack surface of Adversarial Machine Learning (AML) and the potential attacks against Vehicle-to-Microgrid (V2M) services. We present an anticipatory study of a multi-stage gray-box attack that can achieve a comparable result to a white-box attack. Adversaries aim to deceive the targeted Machine Learning (ML) classifier at the network edge to misclassify the incoming energy requests from microgrids. With an inference attack, an adversary can collect real-time data from the communication between smart microgrids and a 5G gNodeB to train a surrogate (i.e., shadow) model of the targeted classifier at the edge. To anticipate the associated impact of an adversary's capability to collect real-time data instances, we study five different cases, each representing different amounts of real-time data instances collected by an adversary. Out of six ML models trained on the complete dataset, K-Nearest Neighbour (K-NN) is selected as the surrogate model, and through simulations, we demonstrate that the multi-stage gray-box attack is able to mislead the ML classifier and cause an Evasion Increase Rate (EIR) up to 73.2% using 40% less data than what a white-box attack needs to achieve a similar EIR.

摘要: 本文研究了对抗性机器学习(AML)不断扩大的攻击面和针对车辆到微电网(V2M)服务的潜在攻击。我们提出了一种多阶段灰盒攻击的预期研究，它可以获得与白盒攻击相当的结果。攻击者的目标是在网络边缘欺骗目标机器学习(ML)分类器，以对来自微电网的传入能源请求进行错误分类。利用推理攻击，攻击者可以从智能微网和5G gNodeB之间的通信中收集实时数据，以在边缘训练目标分类器的代理(即，影子)模型。为了预测对手收集实时数据实例的能力的相关影响，我们研究了五个不同的案例，每个案例代表了对手收集的不同数量的实时数据实例。在完整数据集上训练的6个ML模型中，选择K-近邻(K-NN)作为代理模型，通过仿真，我们证明了多级灰盒攻击能够误导ML分类器，并导致高达73.2%的逃避增加率(EIR)，而白盒攻击所需的数据比白盒攻击所需的数据少40%。



## **8. Get your Foes Fooled: Proximal Gradient Split Learning for Defense against Model Inversion Attacks on IoMT data**

愚弄你的敌人：用于防御IoMT数据模型反转攻击的近距离梯度分裂学习 cs.CR

10 pages, 5 figures, 2 tables

**SubmitDate**: 2022-08-09    [paper-pdf](http://arxiv.org/pdf/2201.04569v3)

**Authors**: Sunder Ali Khowaja, Ik Hyun Lee, Kapal Dev, Muhammad Aslam Jarwar, Nawab Muhammad Faseeh Qureshi

**Abstracts**: The past decade has seen a rapid adoption of Artificial Intelligence (AI), specifically the deep learning networks, in Internet of Medical Things (IoMT) ecosystem. However, it has been shown recently that the deep learning networks can be exploited by adversarial attacks that not only make IoMT vulnerable to the data theft but also to the manipulation of medical diagnosis. The existing studies consider adding noise to the raw IoMT data or model parameters which not only reduces the overall performance concerning medical inferences but also is ineffective to the likes of deep leakage from gradients method. In this work, we propose proximal gradient split learning (PSGL) method for defense against the model inversion attacks. The proposed method intentionally attacks the IoMT data when undergoing the deep neural network training process at client side. We propose the use of proximal gradient method to recover gradient maps and a decision-level fusion strategy to improve the recognition performance. Extensive analysis show that the PGSL not only provides effective defense mechanism against the model inversion attacks but also helps in improving the recognition performance on publicly available datasets. We report 14.0$\%$, 17.9$\%$, and 36.9$\%$ gains in accuracy over reconstructed and adversarial attacked images, respectively.

摘要: 在过去的十年中，人工智能(AI)，特别是深度学习网络，在医疗物联网(IoMT)生态系统中得到了迅速的采用。然而，最近的研究表明，深度学习网络可以被敌意攻击所利用，这些攻击不仅使物联网容易受到数据窃取的攻击，而且还容易受到医疗诊断的篡改。现有的研究认为在原始IoMT数据或模型参数中加入噪声，不仅降低了医学推断的整体性能，而且对梯度法等深度泄漏方法无效。在这项工作中，我们提出了近邻梯度分裂学习(PSGL)方法来防御模型反转攻击。该方法在客户端进行深度神经网络训练时，对IoMT数据进行故意攻击。提出了利用近邻梯度法恢复梯度图，并采用决策层融合策略来提高识别性能。广泛的分析表明，PGSL不仅提供了对模型反转攻击的有效防御机制，而且有助于提高对公开可用的数据集的识别性能。与重建图像和对抗性攻击图像相比，准确率分别提高了14.0、17.9和36.9美元。



## **9. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.04435v1)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **10. Can collaborative learning be private, robust and scalable?**

协作学习能否做到私密性、健壮性和可扩展性？ cs.LG

Accepted at MICCAI DeCaF 2022

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.02652v2)

**Authors**: Dmitrii Usynin, Helena Klause, Johannes C. Paetzold, Daniel Rueckert, Georgios Kaissis

**Abstracts**: In federated learning for medical image analysis, the safety of the learning protocol is paramount. Such settings can often be compromised by adversaries that target either the private data used by the federation or the integrity of the model itself. This requires the medical imaging community to develop mechanisms to train collaborative models that are private and robust against adversarial data. In response to these challenges, we propose a practical open-source framework to study the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples under train- and inference-time attacks. Using our framework, we achieve competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation, critical in medical image analysis.

摘要: 在医学图像分析的联合学习中，学习协议的安全性至关重要。此类设置通常会被针对联盟使用的私有数据或模型本身的完整性的攻击者所攻破。这需要医学成像界开发机制，以训练针对敌对数据的私有和健壮的协作模型。针对这些挑战，我们提出了一个实用的开源框架来研究差异隐私、模型压缩和对抗性训练相结合的有效性，以提高模型在训练和推理时间攻击下对对抗性样本的健壮性。使用我们的框架，我们获得了具有竞争力的模型性能，显著减少了模型的规模，并在不严重性能下降的情况下改善了经验对抗鲁棒性，这在医学图像分析中至关重要。



## **11. Sparse Adversarial Attack in Multi-agent Reinforcement Learning**

多智能体强化学习中的稀疏对抗性攻击 cs.AI

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2205.09362v2)

**Authors**: Yizheng Hu, Zhihua Zhang

**Abstracts**: Cooperative multi-agent reinforcement learning (cMARL) has many real applications, but the policy trained by existing cMARL algorithms is not robust enough when deployed. There exist also many methods about adversarial attacks on the RL system, which implies that the RL system can suffer from adversarial attacks, but most of them focused on single agent RL. In this paper, we propose a \textit{sparse adversarial attack} on cMARL systems. We use (MA)RL with regularization to train the attack policy. Our experiments show that the policy trained by the current cMARL algorithm can obtain poor performance when only one or a few agents in the team (e.g., 1 of 8 or 5 of 25) were attacked at a few timesteps (e.g., attack 3 of total 40 timesteps).

摘要: 协作多智能体强化学习(CMARL)有很多实际应用，但已有的cMARL算法训练的策略在实际应用中不够健壮。针对RL系统的对抗性攻击也有很多方法，这意味着RL系统可能会遭受对抗性攻击，但大多数方法都集中在单个代理RL上。本文提出了一种针对cMARL系统的稀疏对抗攻击。我们使用带正则化的(MA)RL来训练攻击策略。我们的实验表明，当团队中只有一个或几个代理(例如，8个代理中的1个或25个代理中的5个)在几个时间步骤(例如，总共40个时间步骤中的攻击3个)受到攻击时，由当前cMARL算法训练的策略会获得较差的性能。



## **12. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

对抗性像素恢复作为可转移扰动的借口任务 cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2207.08803v2)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max scheme which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to the adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our training approach improves the transferability of the baseline unsupervised training method by 16.4% on ImageNet val. set. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR

摘要: 可转移对抗性攻击从预先训练的代理模型和已知标签空间中优化对手，以愚弄未知的黑盒模型。因此，这些攻击受到有效代理模型可用性的限制。在这项工作中，我们放松了这一假设，提出了对抗性像素复原作为一种自我监督的替代方案，在没有标签和数据样本的情况下，从零开始训练一个有效的代理模型。我们的训练方法基于最小-最大方案，该方案减少了通过对抗性目标的过度拟合，从而优化了更具普适性的代理模型。我们提出的攻击是对抗性像素恢复的补充，并且独立于任何特定于任务的目标，因为它可以以自我监督的方式发起。我们成功地展示了我们的视觉变形方法以及卷积神经网络方法在分类、目标检测和视频分割任务中的对抗性可转移性。我们的训练方法将基线无监督训练方法在ImageNet Val上的可转移性提高了16.4%。准备好了。我们的代码和预先培训的代孕模型可在以下网址获得：https://github.com/HashmatShadab/APR



## **13. Adversarial robustness of $β-$VAE through the lens of local geometry**

通过局部几何透镜分析$β-$VAE的对抗健壮性 cs.LG

The 2022 ICML Workshop on New Frontiers in Adversarial Machine  Learning

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2208.03923v1)

**Authors**: Asif Khan, Amos Storkey

**Abstracts**: Variational autoencoders (VAEs) are susceptible to adversarial attacks. An adversary can find a small perturbation in the input sample to change its latent encoding non-smoothly, thereby compromising the reconstruction. A known reason for such vulnerability is the latent space distortions arising from a mismatch between approximated latent posterior and a prior distribution. Consequently, a slight change in the inputs leads to a significant change in the latent space encodings. This paper demonstrates that the sensitivity around a data point is due to a directional bias of a stochastic pullback metric tensor induced by the encoder network. The pullback metric tensor measures the infinitesimal volume change from input to latent space. Thus, it can be viewed as a lens to analyse the effect of small changes in the input leading to distortions in the latent space. We propose robustness evaluation scores using the eigenspectrum of a pullback metric. Moreover, we empirically show that the scores correlate with the robustness parameter $\beta$ of the $\beta-$VAE.

摘要: 可变自动编码器(VAE)容易受到敌意攻击。攻击者可以在输入样本中发现微小的扰动，从而非平稳地改变其潜在编码，从而危及重建。这种脆弱性的一个已知原因是由于近似的潜在后验分布和先验分布之间的失配而引起的潜在空间扭曲。因此，输入的微小变化会导致潜在空间编码的显著变化。本文证明了数据点附近的敏感性是由编码器网络引起的随机拉回度量张量的方向偏差造成的。拉回度量张量测量从输入到潜在空间的无限小体积变化。因此，它可以看作是一个透镜，用来分析输入的微小变化导致潜在空间扭曲的影响。我们使用拉回度量的特征谱来提出稳健性评估分数。此外，我们的经验表明，得分与$\beta-$VAE的稳健性参数$\beta$相关。



## **14. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

对抗性后门防御微调：将后门攻击与对抗性攻击联系起来 cs.CV

**SubmitDate**: 2022-08-08    [paper-pdf](http://arxiv.org/pdf/2202.06312v3)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.

摘要: 众所周知，深度神经网络(DNN)既容易受到后门攻击，也容易受到对手攻击。在文献中，这两类攻击通常被视为不同的问题，分别属于训练时间攻击和推理时间攻击。然而，在本文中，我们发现了它们之间的一个有趣的联系：对于一个植入后门的模型，我们观察到其敌对示例与其触发样本具有相似的行为，即两者都激活了相同的DNN神经元子集。它表明，在模型中植入后门将显著影响模型的对抗性示例。基于这些观察结果，我们设计了一种新的对抗精调(AFT)算法来防御后门攻击。我们的实验表明，对于5种最先进的后门攻击，我们的AFT可以有效地清除后门触发，而在干净的样本上没有明显的性能下降，并且显著优于现有的防御方法。



## **15. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v2)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了一种隐私保护方案，该方案在保留VFL给主动方带来的全部好处的同时，恶化了对手的重构攻击。最后，实验结果验证了所提出的攻击和隐私保护方案的有效性。



## **16. Garbled EDA: Privacy Preserving Electronic Design Automation**

乱码EDA：保护隐私的电子设计自动化 cs.CR

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03822v1)

**Authors**: Mohammad Hashemi, Steffi Roy, Fatemeh Ganji, Domenic Forte

**Abstracts**: The complexity of modern integrated circuits (ICs) necessitates collaboration between multiple distrusting parties, including thirdparty intellectual property (3PIP) vendors, design houses, CAD/EDA tool vendors, and foundries, which jeopardizes confidentiality and integrity of each party's IP. IP protection standards and the existing techniques proposed by researchers are ad hoc and vulnerable to numerous structural, functional, and/or side-channel attacks. Our framework, Garbled EDA, proposes an alternative direction through formulating the problem in a secure multi-party computation setting, where the privacy of IPs, CAD tools, and process design kits (PDKs) is maintained. As a proof-of-concept, Garbled EDA is evaluated in the context of simulation, where multiple IP description formats (Verilog, C, S) are supported. Our results demonstrate a reasonable logical-resource cost and negligible memory overhead. To further reduce the overhead, we present another efficient implementation methodology, feasible when the resource utilization is a bottleneck, but the communication between two parties is not restricted. Interestingly, this implementation is private and secure even in the presence of malicious adversaries attempting to, e.g., gain access to PDKs or in-house IPs of the CAD tool providers.

摘要: 现代集成电路(IC)的复杂性需要多方合作，包括第三方知识产权(3PIP)供应商、设计公司、CAD/EDA工具供应商和铸造厂，这危及每一方知识产权的机密性和完整性。研究人员提出的IP保护标准和现有技术是特别的，容易受到许多结构性、功能性和/或旁路攻击。我们的框架，乱码EDA，通过在保护IP、CAD工具和工艺设计工具包(PDK)隐私的安全多方计算环境中描述问题，提出了另一种方向。作为概念验证，乱码EDA在支持多种IP描述格式(Verilog、C、S)的模拟环境中进行评估。我们的结果证明了合理的逻辑资源开销和可以忽略的内存开销。为了进一步减少开销，我们提出了另一种高效的实现方法，当资源利用率成为瓶颈时，该方法是可行的，但双方之间的通信不受限制。有趣的是，即使在恶意攻击者试图例如访问CAD工具提供商的PDK或内部IP的情况下，该实现也是私有和安全的。



## **17. Federated Adversarial Learning: A Framework with Convergence Analysis**

联合对抗性学习：一个收敛分析框架 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03635v1)

**Authors**: Xiaoxiao Li, Zhao Song, Jiaming Yang

**Abstracts**: Federated learning (FL) is a trending training paradigm to utilize decentralized training data. FL allows clients to update model parameters locally for several epochs, then share them to a global model for aggregation. This training paradigm with multi-local step updating before aggregation exposes unique vulnerabilities to adversarial attacks. Adversarial training is a popular and effective method to improve the robustness of networks against adversaries. In this work, we formulate a general form of federated adversarial learning (FAL) that is adapted from adversarial learning in the centralized setting. On the client side of FL training, FAL has an inner loop to generate adversarial samples for adversarial training and an outer loop to update local model parameters. On the server side, FAL aggregates local model updates and broadcast the aggregated model. We design a global robust training loss and formulate FAL training as a min-max optimization problem. Unlike the convergence analysis in classical centralized training that relies on the gradient direction, it is significantly harder to analyze the convergence in FAL for three reasons: 1) the complexity of min-max optimization, 2) model not updating in the gradient direction due to the multi-local updates on the client-side before aggregation and 3) inter-client heterogeneity. We address these challenges by using appropriate gradient approximation and coupling techniques and present the convergence analysis in the over-parameterized regime. Our main result theoretically shows that the minimum loss under our algorithm can converge to $\epsilon$ small with chosen learning rate and communication rounds. It is noteworthy that our analysis is feasible for non-IID clients.

摘要: 联合学习(FL)是一种利用分散训练数据的训练范型。FL允许客户本地更新几个纪元的模型参数，然后将它们共享到全局模型以进行聚合。这种在聚集之前进行多局部步骤更新的训练范例暴露了独特的易受敌意攻击的弱点。对抗性训练是提高网络对抗敌手健壮性的一种流行而有效的方法。在这项工作中，我们提出了一种联邦对抗性学习(FAL)的一般形式，它是从集中式对抗性学习改编而来的。在FL训练的客户端，FAL有一个用于生成对抗性训练的对抗性样本的内环和一个用于更新局部模型参数的外环。在服务器端，FAL聚合本地模型更新并广播聚合的模型。我们设计了一个全局稳健的训练损失，并将FAL训练描述为一个最小-最大优化问题。与传统集中式训练中依赖于梯度方向的收敛分析不同，FAL的收敛分析明显困难，原因有三：1)最小-最大优化的复杂性；2)模型不能在梯度方向上更新；3)客户端在聚集前的多局部更新；3)客户端之间的异构性。我们通过使用适当的梯度近似和耦合技术来解决这些挑战，并给出了在过参数区域的收敛分析。理论上，我们的主要结果表明，在选择学习速率和通信轮数的情况下，我们的算法可以使最小损失收敛到较小的值。值得注意的是，我们的分析对非IID客户是可行的。



## **18. Blackbox Attacks via Surrogate Ensemble Search**

通过代理集成搜索进行黑盒攻击 cs.LG

**SubmitDate**: 2022-08-07    [paper-pdf](http://arxiv.org/pdf/2208.03610v1)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for blackbox attacks via surrogate ensemble search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet (including VGG-19, DenseNet-121, and ResNext-50). In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for non-targeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% non-targeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks.

摘要: 黑盒对抗性攻击可分为基于传输的攻击和基于查询的攻击。传输方法不需要来自受害者模型的任何反馈，但与基于查询的方法相比，提供了更低的成功率。查询攻击通常需要大量查询才能成功。为了达到这两种方法的最佳效果，最近的努力试图将它们结合起来，但仍然需要数百次查询才能获得高成功率(特别是针对有针对性的攻击)。在本文中，我们提出了一种新的方法，通过代理集成搜索(基)可以用极少的查询生成高度成功的黑盒攻击。我们首先定义了一种微扰机，它通过最小化一组固定代理模型上的加权损失函数来生成扰动图像。为了针对给定的受害者模型生成攻击，我们使用由扰动机器生成的查询来搜索损失函数中的权重。由于搜索空间的维度很小(与代理模型的数量相同)，因此搜索需要少量的查询。我们的实验结果表明，在使用ImageNet(包括VGG-19、DenseNet-121和ResNext-50)训练的不同图像分类器上，我们提出的方法获得了更好的成功率，查询次数至少减少了30倍。特别是，我们的方法只需每幅图像3个查询(平均)就可以实现90%以上的定向攻击成功率和1-2个查询的非定向攻击成功率99%以上。我们的方法在Google Cloud Vision API上也是有效的，在每张图片2.9个查询的情况下，实现了91%的非定向攻击成功率。我们还证明了我们提出的方法产生的扰动具有很高的可转移性，可以用于硬标签黑盒攻击。



## **19. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

未知聚类个数在线聚类的重访高斯神经元 cs.LG

Reviewed at  https://openreview.net/forum?id=h05RLBNweX&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR)

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.00920v2)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. Although these weaknesses are not specifically addressed, a novel local learning rule is presented that performs online clustering with an upper limit on the number of clusters to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.

摘要: 尽管人工神经网络最近取得了成功，但可能需要更多生物学上可信的学习方法来解决反向传播训练模型的弱点，如灾难性遗忘和对抗性攻击。虽然没有具体解决这些缺点，但提出了一种新的局部学习规则，该规则执行在线聚类时，对要发现的簇的数量设置上限，而不是固定的簇计数。不使用正交权重或输出激活约束，而是通过侧向高斯神经元的相互排斥来获得激活稀疏性，以确保多个神经元中心不会占据输入域中的相同位置。在数据样本可以用均值和方差表示的情况下，提出了一种调整高斯神经元宽度的更新方法。这些算法被应用于MNIST和CIFAR-10数据集，以创建捕捉不同大小的像素斑块的输入模式的过滤器。实验结果表明，在大量的训练样本中，学习的参数是稳定的。



## **20. On the Fundamental Limits of Formally (Dis)Proving Robustness in Proof-of-Learning**

学习证明中形式(Dis)证明稳健性的基本极限 cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03567v1)

**Authors**: Congyu Fang, Hengrui Jia, Anvith Thudi, Mohammad Yaghini, Christopher A. Choquette-Choo, Natalie Dullerud, Varun Chandrasekaran, Nicolas Papernot

**Abstracts**: Proof-of-learning (PoL) proposes a model owner use machine learning training checkpoints to establish a proof of having expended the necessary compute for training. The authors of PoL forego cryptographic approaches and trade rigorous security guarantees for scalability to deep learning by being applicable to stochastic gradient descent and adaptive variants. This lack of formal analysis leaves the possibility that an attacker may be able to spoof a proof for a model they did not train.   We contribute a formal analysis of why the PoL protocol cannot be formally (dis)proven to be robust against spoofing adversaries. To do so, we disentangle the two roles of proof verification in PoL: (a) efficiently determining if a proof is a valid gradient descent trajectory, and (b) establishing precedence by making it more expensive to craft a proof after training completes (i.e., spoofing). We show that efficient verification results in a tradeoff between accepting legitimate proofs and rejecting invalid proofs because deep learning necessarily involves noise. Without a precise analytical model for how this noise affects training, we cannot formally guarantee if a PoL verification algorithm is robust. Then, we demonstrate that establishing precedence robustly also reduces to an open problem in learning theory: spoofing a PoL post hoc training is akin to finding different trajectories with the same endpoint in non-convex learning. Yet, we do not rigorously know if priori knowledge of the final model weights helps discover such trajectories.   We conclude that, until the aforementioned open problems are addressed, relying more heavily on cryptography is likely needed to formulate a new class of PoL protocols with formal robustness guarantees. In particular, this will help with establishing precedence. As a by-product of insights from our analysis, we also demonstrate two novel attacks against PoL.

摘要: 学习证明(POL)建议模型所有者使用机器学习训练检查点来建立已经为训练花费了必要的计算机的证明。POL FOREO密码方法的作者通过适用于随机梯度下降和自适应变体，在可伸缩性到深度学习的严格安全保证之间进行权衡。由于缺乏正式的分析，攻击者有可能伪造他们没有训练过的模型的证据。我们对POL协议为什么不能被形式化(DIS)证明对欺骗对手是健壮的进行了正式的分析。为此，我们将POL中证明验证的两个角色分开：(A)有效地确定证明是否为有效的梯度下降轨迹，以及(B)通过使训练完成后制作证明的成本更高(即，欺骗)来建立优先级。我们证明了有效的验证在接受合法证明和拒绝无效证明之间产生了折衷，因为深度学习必然涉及噪声。如果没有准确的分析模型来分析噪声如何影响训练，我们就不能正式保证POL验证算法是否健壮。然后，我们证明了稳健地建立优先权也归结为学习理论中的一个开放问题：欺骗POL后自组织训练类似于在非凸学习中找到具有相同终点的不同轨迹。然而，我们并不确切地知道最终模型权重的先验知识是否有助于发现这样的轨迹。我们的结论是，在上述公开问题得到解决之前，很可能需要更多地依赖密码学来制定具有形式健壮性保证的新型POL协议。特别是，这将有助于确立优先地位。作为我们分析的副产品，我们还演示了针对POL的两种新攻击。



## **21. Preventing or Mitigating Adversarial Supply Chain Attacks; a legal analysis**

预防或减轻对抗性供应链攻击；法律分析 cs.CY

23 pages

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2208.03466v1)

**Authors**: Kaspar Rosager Ludvigsen, Shishir Nagaraja, Angela Daly

**Abstracts**: The world is currently strongly connected through both the internet at large, but also the very supply chains which provide everything from food to infrastructure and technology. The supply chains are themselves vulnerable to adversarial attacks, both in a digital and physical sense, which can disrupt or at worst destroy them. In this paper, we take a look at two examples of such successful attacks and consider what their consequences may be going forward, and analyse how EU and national law can prevent these attacks or otherwise punish companies which do not try to mitigate them at all possible costs. We find that the current types of national regulation are not technology specific enough, and cannot force or otherwise mandate the correct parties who could play the biggest role in preventing supply chain attacks to do everything in their power to mitigate them. But, current EU law is on the right path, and further vigilance may be what is necessary to consider these large threats, as national law tends to fail at properly regulating companies when it comes to cybersecurity.

摘要: 目前，世界通过互联网和供应链紧密相连，供应链提供从食品到基础设施和技术的一切东西。供应链本身在数字和物理意义上都很容易受到敌意攻击，这些攻击可能会扰乱供应链，甚至在最坏的情况下摧毁它们。在这篇文章中，我们看了两个此类成功攻击的例子，并考虑它们未来可能产生的后果，并分析欧盟和国家法律如何防止这些攻击或以其他方式惩罚那些不试图不惜一切代价减轻攻击的公司。我们发现，目前的国家监管类型不够具体，不能强迫或以其他方式强制正确的各方尽其所能缓解供应链攻击，这些各方可以在防止供应链攻击方面发挥最大作用。但是，当前的欧盟法律走在正确的道路上，进一步的警惕可能是考虑这些重大威胁所必需的，因为在网络安全方面，各国法律往往无法对公司进行适当的监管。



## **22. Searching for the Essence of Adversarial Perturbations**

寻找对抗性扰动的本质 cs.LG

**SubmitDate**: 2022-08-06    [paper-pdf](http://arxiv.org/pdf/2205.15357v2)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance in various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is shown to fool neural networks' predictions. This would lead to potential risks for real-world applications such as endangering autonomous driving and messing up text identification. To mitigate such risks, an understanding of how adversarial examples operate is critical, which however remains unresolved. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction, in contrast to a widely discussed argument that human-imperceptible information plays the critical role in fooling a network. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, including the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.

摘要: 神经网络在不同的机器学习领域取得了最先进的性能，然而在输入数据中加入恶意扰动(对抗性的例子)被证明愚弄了神经网络的预测。这将给现实世界的应用带来潜在风险，如危及自动驾驶和扰乱文本识别。为了减轻这种风险，了解对抗性案例如何运作是至关重要的，但这一问题仍未得到解决。在这里，我们证明了对抗性扰动包含人类可识别的信息，这是导致神经网络错误预测的关键阴谋者，而不是广泛讨论的人类不可感知的信息在愚弄网络方面发挥关键作用的论点。这一人类可识别信息的概念允许我们解释与对抗性扰动相关的关键特征，包括对抗性例子的存在，不同神经网络之间的可转换性，以及用于对抗性训练的神经网络更高的可解释性。揭示了欺骗神经网络的对抗性扰动中的两个独特性质：掩蔽和生成。当神经网络对输入图像进行分类时，识别出一种特殊的类，即互补类。敌意干扰中包含的人类可识别的信息使研究人员能够深入了解神经网络的工作原理，并可能导致开发检测/防御敌意攻击的技术。



## **23. Success of Uncertainty-Aware Deep Models Depends on Data Manifold Geometry**

不确定性感知深度模型的成功依赖于数据流形几何 cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.01705v2)

**Authors**: Mark Penrod, Harrison Termotto, Varshini Reddy, Jiayu Yao, Finale Doshi-Velez, Weiwei Pan

**Abstracts**: For responsible decision making in safety-critical settings, machine learning models must effectively detect and process edge-case data. Although existing works show that predictive uncertainty is useful for these tasks, it is not evident from literature which uncertainty-aware models are best suited for a given dataset. Thus, we compare six uncertainty-aware deep learning models on a set of edge-case tasks: robustness to adversarial attacks as well as out-of-distribution and adversarial detection. We find that the geometry of the data sub-manifold is an important factor in determining the success of various models. Our finding suggests an interesting direction in the study of uncertainty-aware deep learning models.

摘要: 为了在安全关键环境中做出负责任的决策，机器学习模型必须有效地检测和处理边缘案例数据。虽然现有的工作表明，预测不确定性对这些任务是有用的，但从文献中并不明显地看到，哪些不确定性感知模型最适合给定的数据集。因此，我们在一组边缘情况任务上比较了六种不确定性感知的深度学习模型：对对手攻击的健壮性以及分布外和对抗性检测。我们发现，数据子流形的几何形状是决定各种模型成功与否的重要因素。我们的发现为不确定性感知深度学习模型的研究提供了一个有趣的方向。



## **24. Attacking Adversarial Defences by Smoothing the Loss Landscape**

通过平滑损失图景来攻击对抗性防御 cs.LG

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.00862v2)

**Authors**: Panagiotis Eustratiadis, Henry Gouk, Da Li, Timothy Hospedales

**Abstracts**: This paper investigates a family of methods for defending against adversarial attacks that owe part of their success to creating a noisy, discontinuous, or otherwise rugged loss landscape that adversaries find difficult to navigate. A common, but not universal, way to achieve this effect is via the use of stochastic neural networks. We show that this is a form of gradient obfuscation, and propose a general extension to gradient-based adversaries based on the Weierstrass transform, which smooths the surface of the loss function and provides more reliable gradient estimates. We further show that the same principle can strengthen gradient-free adversaries. We demonstrate the efficacy of our loss-smoothing method against both stochastic and non-stochastic adversarial defences that exhibit robustness due to this type of obfuscation. Furthermore, we provide analysis of how it interacts with Expectation over Transformation; a popular gradient-sampling method currently used to attack stochastic defences.

摘要: 本文研究了一系列防御对手攻击的方法，这些攻击的成功部分归因于创建了一个嘈杂的、不连续的或以其他方式崎岖的损失场景，对手发现很难导航。实现这一效果的一种常见但并不普遍的方法是通过使用随机神经网络。我们证明了这是一种梯度混淆的形式，并提出了一种基于魏尔斯特拉斯变换的对基于梯度的攻击的一般扩展，它平滑了损失函数的表面，并提供了更可靠的梯度估计。我们进一步证明，同样的原理可以加强无梯度的对手。我们证明了我们的损失平滑方法对随机和非随机对抗防御的有效性，这些防御由于这种类型的混淆而表现出稳健性。此外，我们还分析了它如何与变换上的期望相互作用，变换上的期望是目前用于攻击随机防御的一种流行的梯度抽样方法。



## **25. Adversarial Robustness of MR Image Reconstruction under Realistic Perturbations**

现实摄动下MR图像重建的对抗稳健性 eess.IV

Accepted at the MICCAI-2022 workshop: Machine Learning for Medical  Image Reconstruction

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2208.03161v1)

**Authors**: Jan Nikolas Morshuis, Sergios Gatidis, Matthias Hein, Christian F. Baumgartner

**Abstracts**: Deep Learning (DL) methods have shown promising results for solving ill-posed inverse problems such as MR image reconstruction from undersampled $k$-space data. However, these approaches currently have no guarantees for reconstruction quality and the reliability of such algorithms is only poorly understood. Adversarial attacks offer a valuable tool to understand possible failure modes and worst case performance of DL-based reconstruction algorithms. In this paper we describe adversarial attacks on multi-coil $k$-space measurements and evaluate them on the recently proposed E2E-VarNet and a simpler UNet-based model. In contrast to prior work, the attacks are targeted to specifically alter diagnostically relevant regions. Using two realistic attack models (adversarial $k$-space noise and adversarial rotations) we are able to show that current state-of-the-art DL-based reconstruction algorithms are indeed sensitive to such perturbations to a degree where relevant diagnostic information may be lost. Surprisingly, in our experiments the UNet and the more sophisticated E2E-VarNet were similarly sensitive to such attacks. Our findings add further to the evidence that caution must be exercised as DL-based methods move closer to clinical practice.

摘要: 深度学习方法在解决不适定反问题，如从欠采样的$k$空间数据重建MR图像方面显示出良好的结果。然而，这些方法目前并不能保证重建质量，而且人们对这些算法的可靠性知之甚少。对抗性攻击为理解基于DL的重构算法可能的失效模式和最坏情况下的性能提供了有价值的工具。在这篇文章中，我们描述了对多线圈$k$空间测量的对抗性攻击，并在最近提出的E2E-Varnet和一个更简单的基于UNT的模型上对它们进行了评估。与以前的工作不同，这些攻击的目标是专门改变诊断相关的区域。使用两个真实的攻击模型(对抗性空间噪声和对抗性旋转)，我们能够证明当前最先进的基于DL的重建算法确实对此类扰动敏感到可能丢失相关诊断信息的程度。令人惊讶的是，在我们的实验中，UNT和更复杂的E2E-Varnet对此类攻击同样敏感。我们的发现进一步证明，随着基于DL的方法越来越接近临床实践，必须谨慎行事。



## **26. A Systematic Survey of Attack Detection and Prevention in Connected and Autonomous Vehicles**

互联无人驾驶汽车攻击检测与防御的系统研究 cs.CR

This article is published in the Vehicular Communications journal

**SubmitDate**: 2022-08-05    [paper-pdf](http://arxiv.org/pdf/2203.14965v2)

**Authors**: Trupil Limbasiya, Ko Zheng Teng, Sudipta Chattopadhyay, Jianying Zhou

**Abstracts**: The number of Connected and Autonomous Vehicles (CAVs) is increasing rapidly in various smart transportation services and applications, considering many benefits to society, people, and the environment. Several research surveys for CAVs were conducted by primarily focusing on various security threats and vulnerabilities in the domain of CAVs to classify different types of attacks, impacts of attacks, attack features, cyber-risk, defense methodologies against attacks, and safety standards. However, the importance of attack detection and prevention approaches for CAVs has not been discussed extensively in the state-of-the-art surveys, and there is a clear gap in the existing literature on such methodologies to detect new and conventional threats and protect the CAV systems from unexpected hazards on the road. Some surveys have a limited discussion on Attacks Detection and Prevention Systems (ADPS), but such surveys provide only partial coverage of different types of ADPS for CAVs. Furthermore, there is a scope for discussing security, privacy, and efficiency challenges in ADPS that can give an overview of important security and performance attributes.   This survey paper, therefore, presents the significance of CAVs in the market, potential challenges in CAVs, key requirements of essential security and privacy properties, various capabilities of adversaries, possible attacks in CAVs, and performance evaluation parameters for ADPS. An extensive analysis is discussed of different ADPS categories for CAVs and state-of-the-art research works based on each ADPS category that gives the latest findings in this research domain. This survey also discusses crucial and open security research problems that are required to be focused on the secure deployment of CAVs in the market.

摘要: 考虑到对社会、人民和环境的诸多好处，在各种智能交通服务和应用中，互联和自动驾驶车辆(CAV)的数量正在迅速增加。通过主要关注CAV领域中的各种安全威胁和漏洞，对CAV进行了几项研究调查，以对不同类型的攻击、攻击的影响、攻击特征、网络风险、攻击防御方法和安全标准进行分类。然而，在最新的调查中，攻击检测和预防方法对CAV的重要性没有得到广泛的讨论，并且在现有文献中，对于检测新的和常规的威胁并保护CAV系统免受道路上的意外危险的方法，存在着明显的空白。一些调查对攻击检测和预防系统(ADPS)进行了有限的讨论，但这些调查只提供了针对骑士队不同类型的ADP的部分覆盖。此外，ADPS中还有讨论安全、隐私和效率挑战的范围，可以概述重要的安全和性能属性。因此，这份调查报告介绍了CAV在市场上的重要性、CAV中的潜在挑战、基本安全和隐私属性的关键要求、对手的各种能力、CAV中可能的攻击以及ADPS的性能评估参数。对CAV的不同ADPS类别进行了广泛的分析，并根据每个ADPS类别进行了最新的研究工作，给出了该研究领域的最新发现。本次调查还讨论了需要重点关注Cavs在市场上的安全部署的关键和开放的安全研究问题。



## **27. Differentially Private Counterfactuals via Functional Mechanism**

从作用机制看区分私法反事实 cs.LG

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02878v1)

**Authors**: Fan Yang, Qizhang Feng, Kaixiong Zhou, Jiahao Chen, Xia Hu

**Abstracts**: Counterfactual, serving as one emerging type of model explanation, has attracted tons of attentions recently from both industry and academia. Different from the conventional feature-based explanations (e.g., attributions), counterfactuals are a series of hypothetical samples which can flip model decisions with minimal perturbations on queries. Given valid counterfactuals, humans are capable of reasoning under ``what-if'' circumstances, so as to better understand the model decision boundaries. However, releasing counterfactuals could be detrimental, since it may unintentionally leak sensitive information to adversaries, which brings about higher risks on both model security and data privacy. To bridge the gap, in this paper, we propose a novel framework to generate differentially private counterfactual (DPC) without touching the deployed model or explanation set, where noises are injected for protection while maintaining the explanation roles of counterfactual. In particular, we train an autoencoder with the functional mechanism to construct noisy class prototypes, and then derive the DPC from the latent prototypes based on the post-processing immunity of differential privacy. Further evaluations demonstrate the effectiveness of the proposed framework, showing that DPC can successfully relieve the risks on both extraction and inference attacks.

摘要: 反事实作为一种新兴的模型解释，近年来引起了产业界和学术界的广泛关注。与传统的基于特征的解释(例如，属性)不同，反事实是一系列假设样本，可以在对查询的扰动最小的情况下反转模型决策。在有了有效的反事实的情况下，人类能够在“假设”的情况下进行推理，从而更好地理解模型决策的边界。然而，发布反事实可能是有害的，因为它可能会无意中将敏感信息泄露给对手，这会给模型安全和数据隐私带来更高的风险。为了弥补这一差距，在本文中，我们提出了一种新的框架，在不接触部署的模型或解释集的情况下，生成差异私有反事实(DPC)，其中注入噪声以保护，同时保持反事实的解释角色。特别是，我们训练了一个具有构造噪声类原型的功能机制的自动编码器，然后基于差分隐私的后处理免疫力从潜在原型中推导出DPC。进一步的测试证明了该框架的有效性，表明DPC能够成功地缓解抽取攻击和推理攻击的风险。



## **28. Self-Ensembling Vision Transformer (SEViT) for Robust Medical Image Classification**

自集成视觉转换器(SEViT)用于稳健的医学图像分类 cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02851v1)

**Authors**: Faris Almalik, Mohammad Yaqub, Karthik Nandakumar

**Abstracts**: Vision Transformers (ViT) are competing to replace Convolutional Neural Networks (CNN) for various computer vision tasks in medical imaging such as classification and segmentation. While the vulnerability of CNNs to adversarial attacks is a well-known problem, recent works have shown that ViTs are also susceptible to such attacks and suffer significant performance degradation under attack. The vulnerability of ViTs to carefully engineered adversarial samples raises serious concerns about their safety in clinical settings. In this paper, we propose a novel self-ensembling method to enhance the robustness of ViT in the presence of adversarial attacks. The proposed Self-Ensembling Vision Transformer (SEViT) leverages the fact that feature representations learned by initial blocks of a ViT are relatively unaffected by adversarial perturbations. Learning multiple classifiers based on these intermediate feature representations and combining these predictions with that of the final ViT classifier can provide robustness against adversarial attacks. Measuring the consistency between the various predictions can also help detect adversarial samples. Experiments on two modalities (chest X-ray and fundoscopy) demonstrate the efficacy of SEViT architecture to defend against various adversarial attacks in the gray-box (attacker has full knowledge of the target model, but not the defense mechanism) setting. Code: https://github.com/faresmalik/SEViT

摘要: 视觉转换器(VIT)正竞相取代卷积神经网络(CNN)用于医学成像中的各种计算机视觉任务，如分类和分割。虽然CNN对敌意攻击的脆弱性是一个众所周知的问题，但最近的研究表明，VITS也容易受到此类攻击，并且在攻击下性能显著下降。VITS对精心设计的对抗性样本的脆弱性引起了人们对其临床安全性的严重担忧。在本文中，我们提出了一种新的自集成方法来增强VIT在存在对手攻击时的健壮性。提出的自集成视觉转换器(SEViT)利用了这样一个事实，即VIT的初始块学习的特征表示相对不受对抗性扰动的影响。基于这些中间特征表示学习多个分类器，并将这些预测与最终的VIT分类器的预测相结合，可以提供对对手攻击的稳健性。衡量各种预测之间的一致性也有助于检测敌意样本。在两种模式(胸部X光和眼底镜)上的实验证明了SEViT架构在灰盒(攻击者完全知道目标模型，但不知道防御机制)环境下防御各种对抗性攻击的有效性。代码：https://github.com/faresmalik/SEViT



## **29. Adversarial Attacks on Image Generation With Made-Up Words**

对虚构词语图像生成的对抗性攻击 cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.04135v1)

**Authors**: Raphaël Millière

**Abstracts**: Text-guided image generation models can be prompted to generate images using nonce words adversarially designed to robustly evoke specific visual concepts. Two approaches for such generation are introduced: macaronic prompting, which involves designing cryptic hybrid words by concatenating subword units from different languages; and evocative prompting, which involves designing nonce words whose broad morphological features are similar enough to that of existing words to trigger robust visual associations. The two methods can also be combined to generate images associated with more specific visual concepts. The implications of these techniques for the circumvention of existing approaches to content moderation, and particularly the generation of offensive or harmful images, are discussed.

摘要: 文本引导的图像生成模型可以被提示使用恶意设计的随机词来生成图像，以强健地唤起特定的视觉概念。介绍了两种生成方法：Macaronic提示，它涉及通过连接来自不同语言的子词单元来设计神秘的混合词；以及唤起提示，它涉及设计其广泛的形态特征与现有单词的广泛形态特征足够相似的随机词，以触发稳健的视觉联想。这两种方法还可以组合以生成与更具体的视觉概念相关联的图像。这些技术对规避现有的内容审核方法，特别是产生攻击性或有害图像的影响进行了讨论。



## **30. Mass Exit Attacks on the Lightning Network**

闪电网络上的大规模出口攻击 cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.01908v2)

**Authors**: Cosimo Sguanci, Anastasios Sidiropoulos

**Abstracts**: The Lightning Network (LN) has enjoyed rapid growth over recent years, and has become the most popular scaling solution for the Bitcoin blockchain. The security of the LN hinges on the ability of the nodes to close a channel by settling their balances, which requires confirming a transaction on the Bitcoin blockchain within a pre-agreed time period. This inherent timing restriction that the LN must satisfy, make it susceptible to attacks that seek to increase the congestion on the Bitcoin blockchain, thus preventing correct protocol execution. We study the susceptibility of the LN to \emph{mass exit} attacks, in the presence of a small coalition of adversarial nodes. This is a scenario where an adversary forces a large set of honest protocol participants to interact with the blockchain. We focus on two types of attacks: (i) The first is a \emph{zombie} attack, where a set of $k$ nodes become unresponsive with the goal to lock the funds of many channels for a period of time longer than what the LN protocol dictates. (ii) The second is a \emph{mass double-spend} attack, where a set of $k$ nodes attempt to steal funds by submitting many closing transactions that settle channels using expired protocol states; this causes many honest nodes to have to quickly respond by submitting invalidating transactions. We show via simulations that, under historically-plausible congestion conditions, with mild statistical assumptions on channel balances, both of the attacks can be performed by a very small coalition. To perform our simulations, we formulate the problem of finding a worst-case coalition of $k$ adversarial nodes as a graph cut problem. Our experimental findings are supported by a theoretical justification based on the scale-free topology of the LN.

摘要: 闪电网络(Lightning Network，LN)近年来增长迅速，已成为比特币区块链最受欢迎的扩展解决方案。LN的安全性取决于节点通过结算余额关闭通道的能力，这需要在预先商定的时间段内确认比特币区块链上的交易。LN必须满足的这一固有时间限制使其容易受到攻击，这些攻击试图增加比特币区块链上的拥塞，从而阻止正确的协议执行。我们研究了在存在一个小的敌方节点联盟的情况下，LN对EMPH{MASS EXIT}攻击的敏感性。这是一种对手迫使大量诚实的协议参与者与区块链交互的场景。我们主要关注两种类型的攻击：(I)第一种是僵尸攻击，其中一组$k$节点变得无响应，目标是锁定多个频道的资金长于LN协议规定的时间段。(Ii)第二种攻击是大规模双重花费攻击，其中一组$k$节点试图通过提交许多关闭的事务来窃取资金，这些事务使用过期的协议状态来结算通道；这导致许多诚实的节点不得不通过提交无效事务来快速响应。我们通过模拟表明，在历史上看似合理的拥塞条件下，在对信道平衡的温和统计假设下，这两种攻击都可以由非常小的联盟来执行。为了执行我们的模拟，我们将寻找$k$个敌对节点的最坏情况联盟的问题描述为一个图割问题。我们的实验结果得到了基于LN的无标度拓扑的理论证明。



## **31. Design Considerations and Architecture for a Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

基于风险的弹性自适应身份验证和授权(RAD-AA)框架的设计注意事项和体系结构 cs.CR

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02592v1)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstracts**: A strong cyber attack is capable of degrading the performance of any Information Technology (IT) or Operational Technology (OT) system. In recent cyber attacks, credential theft emerged as one of the primary vectors of gaining entry into the system. Once, an attacker has a foothold in the system, they use token manipulation techniques to elevate the privileges and access protected resources. This makes authentication and authorization a critical component for a secure and resilient cyber system. In this paper we consider the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework Resilient Risk-based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch any cyber attack and provides much-needed strength to critical infrastructure.

摘要: 强大的网络攻击能够降低任何信息技术(IT)或操作技术(OT)系统的性能。在最近的网络攻击中，凭据盗窃成为进入系统的主要载体之一。一旦攻击者在系统中站稳脚跟，他们就会使用令牌操作技术来提升权限并访问受保护的资源。这使得身份验证和授权成为安全和有弹性的网络系统的关键组件。在本文中，我们考虑了这样一个安全的、具有弹性的认证和授权框架的设计考虑因素，该框架能够基于风险分数和信任配置文件自适应。我们将该设计与OAuth 2.0、OpenID Connect和SAML 2.0等现有标准进行了比较。然后，我们研究了流行的威胁模型，如STRIDE和PASA，并总结了所提出的体系结构对常见和相关威胁向量的恢复能力。我们将此框架称为弹性基于风险的自适应身份验证和授权(RAD-AA)。拟议的框架过度增加了对手发动任何网络攻击的成本，并为关键基础设施提供了亟需的力量。



## **32. Prompt Tuning for Generative Multimodal Pretrained Models**

产生式多模式预训练模型的快速调整 cs.CL

Work in progress

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02532v1)

**Authors**: Hao Yang, Junyang Lin, An Yang, Peng Wang, Chang Zhou, Hongxia Yang

**Abstracts**: Prompt tuning has become a new paradigm for model tuning and it has demonstrated success in natural language pretraining and even vision pretraining. In this work, we explore the transfer of prompt tuning to multimodal pretraining, with a focus on generative multimodal pretrained models, instead of contrastive ones. Specifically, we implement prompt tuning on the unified sequence-to-sequence pretrained model adaptive to both understanding and generation tasks. Experimental results demonstrate that the light-weight prompt tuning can achieve comparable performance with finetuning and surpass other light-weight tuning methods. Besides, in comparison with finetuned models, the prompt-tuned models demonstrate improved robustness against adversarial attacks. We further figure out that experimental factors, including the prompt length, prompt depth, and reparameteratization, have great impacts on the model performance, and thus we empirically provide a recommendation for the setups of prompt tuning. Despite the observed advantages, we still find some limitations in prompt tuning, and we correspondingly point out the directions for future studies. Codes are available at \url{https://github.com/OFA-Sys/OFA}

摘要: 即时调优已经成为一种新的模型调优范式，在自然语言预训练甚至视觉预训练中都取得了成功。在这项工作中，我们探索了从即时调整到多模式预训练的转换，重点是生成性的多模式预训练模型，而不是对比模型。具体地说，我们实现了对统一的序列到序列的预训练模型的即时调整，该模型同时适用于理解和生成任务。实验结果表明，轻量级快速调谐可以达到与精调相当的性能，并超过其他轻量级调谐方法。此外，与精调模型相比，快速调谐模型具有更好的抗敌意攻击能力。我们进一步发现，实验因素，包括提示长度、提示深度和再参数化，对模型的性能有很大的影响，因此，我们实证地为提示调整的设置提供了建议。尽管观察到了这些优点，但我们仍然发现了快速调谐的一些局限性，并相应地为未来的研究指明了方向。代码可在\url{https://github.com/OFA-Sys/OFA}



## **33. NoiLIn: Improving Adversarial Training and Correcting Stereotype of Noisy Labels**

NoiLin：改进对抗性训练，纠正对嘈杂标签的刻板印象 cs.LG

Accepted at Transactions on Machine Learning Research (TMLR) at June  2022

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2105.14676v2)

**Authors**: Jingfeng Zhang, Xilie Xu, Bo Han, Tongliang Liu, Gang Niu, Lizhen Cui, Masashi Sugiyama

**Abstracts**: Adversarial training (AT) formulated as the minimax optimization problem can effectively enhance the model's robustness against adversarial attacks. The existing AT methods mainly focused on manipulating the inner maximization for generating quality adversarial variants or manipulating the outer minimization for designing effective learning objectives. However, empirical results of AT always exhibit the robustness at odds with accuracy and the existence of the cross-over mixture problem, which motivates us to study some label randomness for benefiting the AT. First, we thoroughly investigate noisy labels (NLs) injection into AT's inner maximization and outer minimization, respectively and obtain the observations on when NL injection benefits AT. Second, based on the observations, we propose a simple but effective method -- NoiLIn that randomly injects NLs into training data at each training epoch and dynamically increases the NL injection rate once robust overfitting occurs. Empirically, NoiLIn can significantly mitigate the AT's undesirable issue of robust overfitting and even further improve the generalization of the state-of-the-art AT methods. Philosophically, NoiLIn sheds light on a new perspective of learning with NLs: NLs should not always be deemed detrimental, and even in the absence of NLs in the training set, we may consider injecting them deliberately. Codes are available in https://github.com/zjfheart/NoiLIn.

摘要: 对抗性训练(AT)被描述为极小极大优化问题，可以有效地增强模型对对手攻击的鲁棒性。现有的AT方法主要集中在操纵内部最大化来生成高质量的对抗性变体，或者操纵外部最小化来设计有效的学习目标。然而，AT的实验结果总是表现出与准确性不一致的稳健性，以及交叉混合问题的存在，这促使我们研究一些标签随机性，以利于AT。首先，我们深入研究了噪声标签(NLS)注入到AT的内极大化和外极小化中，并得到了NL注入何时有利于AT的观察。其次，在此基础上，提出了一种简单而有效的方法--NoiLIn方法，该方法在每个训练时段将NLS随机注入到训练数据中，并在出现稳健过拟合时动态增加NL的注入率。经验证明，NoiLIn可以显著缓解AT的健壮性过拟合的不良问题，甚至进一步改进最先进的AT方法的普适性。从哲学上讲，NoiLin揭示了使用NLS学习的新视角：NLS不应该总是被认为是有害的，即使训练集中没有NLS，我们也可以考虑故意注入它们。代码在https://github.com/zjfheart/NoiLIn.中可用



## **34. A Robust graph attention network with dynamic adjusted Graph**

一种具有动态调整图的健壮图注意网络 cs.LG

21 pages,13 figures

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2009.13038v3)

**Authors**: Xianchen Zhou, Yaoyun Zeng, Hongxia Wang

**Abstracts**: Graph Attention Networks(GATs) are useful deep learning models to deal with the graph data. However, recent works show that the classical GAT is vulnerable to adversarial attacks. It degrades dramatically with slight perturbations. Therefore, how to enhance the robustness of GAT is a critical problem. Robust GAT(RoGAT) is proposed in this paper to improve the robustness of GAT based on the revision of the attention mechanism. Different from the original GAT, which uses the attention mechanism for different edges but is still sensitive to the perturbation, RoGAT adds an extra dynamic attention score progressively and improves the robustness. Firstly, RoGAT revises the edges weight based on the smoothness assumption which is quite common for ordinary graphs. Secondly, RoGAT further revises the features to suppress features' noise. Then, an extra attention score is generated by the dynamic edge's weight and can be used to reduce the impact of adversarial attacks. Different experiments against targeted and untargeted attacks on citation data on citation data demonstrate that RoGAT outperforms most of the recent defensive methods.

摘要: 图注意网络是处理图数据的一种有用的深度学习模型。然而，最近的研究表明，经典的GAT很容易受到对抗性攻击。在轻微的扰动下，它会急剧退化。因此，如何增强GAT的健壮性是一个关键问题。在对注意机制进行修改的基础上，提出了健壮性GAT(ROGAT)，以提高GAT的健壮性。与原有的GAT算法对不同的边缘使用注意机制但对扰动仍然敏感不同，该算法渐进地增加了额外的动态注意分数，提高了算法的鲁棒性。首先，基于光滑性假设对边的权值进行修正，这在普通图中是很常见的。其次，ROAT进一步修正特征以抑制特征的噪声。然后，通过动态边缘的权重产生额外的注意力分数，并可用于减少对抗性攻击的影响。针对引文数据上的定向攻击和非定向攻击的不同实验表明，蟑螂的表现优于大多数最近的防御方法。



## **35. Privacy Safe Representation Learning via Frequency Filtering Encoder**

基于频率滤波编码器的隐私安全表征学习 cs.CV

The IJCAI-ECAI-22 Workshop on Artificial Intelligence Safety  (AISafety 2022)

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02482v1)

**Authors**: Jonghu Jeong, Minyong Cho, Philipp Benz, Jinwoo Hwang, Jeewook Kim, Seungkwan Lee, Tae-hoon Kim

**Abstracts**: Deep learning models are increasingly deployed in real-world applications. These models are often deployed on the server-side and receive user data in an information-rich representation to solve a specific task, such as image classification. Since images can contain sensitive information, which users might not be willing to share, privacy protection becomes increasingly important. Adversarial Representation Learning (ARL) is a common approach to train an encoder that runs on the client-side and obfuscates an image. It is assumed, that the obfuscated image can safely be transmitted and used for the task on the server without privacy concerns. However, in this work, we find that training a reconstruction attacker can successfully recover the original image of existing ARL methods. To this end, we introduce a novel ARL method enhanced through low-pass filtering, limiting the available information amount to be encoded in the frequency domain. Our experimental results reveal that our approach withstands reconstruction attacks while outperforming previous state-of-the-art methods regarding the privacy-utility trade-off. We further conduct a user study to qualitatively assess our defense of the reconstruction attack.

摘要: 深度学习模型越来越多地部署在现实世界的应用中。这些模型通常部署在服务器端，并以信息丰富的表示形式接收用户数据，以解决特定任务，如图像分类。由于图像可能包含用户可能不愿意分享的敏感信息，因此隐私保护变得越来越重要。对抗表示学习(ARL)是一种常见的方法，用于训练运行在客户端并对图像进行混淆的编码器。假设模糊后的图像可以安全地传输并用于服务器上的任务，而不会引起隐私问题。然而，在这项工作中，我们发现训练一个重建攻击者可以成功地恢复现有ARL方法的原始图像。为此，我们引入了一种新的ARL方法，通过低通滤波来增强，限制了可在频域中编码的信息量。我们的实验结果表明，我们的方法经受住了重建攻击，同时优于之前关于隐私效用权衡的最新方法。我们进一步进行了一项用户研究，以定性评估我们对重建攻击的防御。



## **36. Node Copying: A Random Graph Model for Effective Graph Sampling**

节点复制：一种有效图采样的随机图模型 stat.ML

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02435v1)

**Authors**: Florence Regol, Soumyasundar Pal, Jianing Sun, Yingxue Zhang, Yanhui Geng, Mark Coates

**Abstracts**: There has been an increased interest in applying machine learning techniques on relational structured-data based on an observed graph. Often, this graph is not fully representative of the true relationship amongst nodes. In these settings, building a generative model conditioned on the observed graph allows to take the graph uncertainty into account. Various existing techniques either rely on restrictive assumptions, fail to preserve topological properties within the samples or are prohibitively expensive for larger graphs. In this work, we introduce the node copying model for constructing a distribution over graphs. Sampling of a random graph is carried out by replacing each node's neighbors by those of a randomly sampled similar node. The sampled graphs preserve key characteristics of the graph structure without explicitly targeting them. Additionally, sampling from this model is extremely simple and scales linearly with the nodes. We show the usefulness of the copying model in three tasks. First, in node classification, a Bayesian formulation based on node copying achieves higher accuracy in sparse data settings. Second, we employ our proposed model to mitigate the effect of adversarial attacks on the graph topology. Last, incorporation of the model in a recommendation system setting improves recall over state-of-the-art methods.

摘要: 在基于观察到的图的关系结构数据上应用机器学习技术已经引起了越来越多的兴趣。通常，此图不能完全代表节点之间的真实关系。在这些设置中，建立以观察到的图为条件的生成模型允许将图的不确定性考虑在内。现有的各种技术要么依赖于限制性假设，要么不能保持样本中的拓扑属性，要么对于更大的图来说昂贵得令人望而却步。在这项工作中，我们引入了节点复制模型来构造图上的分布。随机图的采样是通过用随机采样的相似节点的邻居替换每个节点的邻居来执行的。采样的图形保留了图形结构的关键特征，而没有明确地以它们为目标。此外，该模型的采样非常简单，并且随着节点的增加而线性扩展。我们在三个任务中展示了复制模型的有效性。首先，在节点分类中，基于节点复制的贝叶斯公式在稀疏数据环境下实现了更高的精度。其次，我们使用我们提出的模型来缓解对抗性攻击对图拓扑的影响。最后，将该模型结合到推荐系统设置中，提高了对最先进方法的召回率。



## **37. Is current research on adversarial robustness addressing the right problem?**

目前关于对手稳健性的研究解决了正确的问题吗？ cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.00539v2)

**Authors**: Ali Borji

**Abstracts**: Short answer: Yes, Long answer: No! Indeed, research on adversarial robustness has led to invaluable insights helping us understand and explore different aspects of the problem. Many attacks and defenses have been proposed over the last couple of years. The problem, however, remains largely unsolved and poorly understood. Here, I argue that the current formulation of the problem serves short term goals, and needs to be revised for us to achieve bigger gains. Specifically, the bound on perturbation has created a somewhat contrived setting and needs to be relaxed. This has misled us to focus on model classes that are not expressive enough to begin with. Instead, inspired by human vision and the fact that we rely more on robust features such as shape, vertices, and foreground objects than non-robust features such as texture, efforts should be steered towards looking for significantly different classes of models. Maybe instead of narrowing down on imperceptible adversarial perturbations, we should attack a more general problem which is finding architectures that are simultaneously robust to perceptible perturbations, geometric transformations (e.g. rotation, scaling), image distortions (lighting, blur), and more (e.g. occlusion, shadow). Only then we may be able to solve the problem of adversarial vulnerability.

摘要: 简短的答案是：是的，长期的答案是：不！事实上，对对手健壮性的研究已经带来了宝贵的见解，帮助我们理解和探索问题的不同方面。在过去的几年里，已经提出了许多攻击和防御措施。然而，这个问题在很大程度上仍然没有得到解决，人们对此知之甚少。在这里，我认为，目前对问题的表述是为短期目标服务的，需要进行修改，以便我们实现更大的收益。具体地说，微扰的界限创造了一种有点做作的设置，需要放松。这误导了我们将注意力集中在一开始就不够有表现力的模型类上。取而代之的是，受人类视觉的启发，以及我们更依赖于形状、顶点和前景对象等稳健特征而不是纹理等非稳健特征的事实，应该努力寻找显著不同类别的模型。也许我们不应该缩小到不可感知的对抗性扰动，而应该解决一个更一般的问题，即寻找同时对可感知扰动、几何变换(例如旋转、缩放)、图像失真(照明、模糊)以及更多(例如遮挡、阴影)具有健壮性的体系结构。只有到那时，我们才可能解决对手脆弱性的问题。



## **38. A New Kind of Adversarial Example**

一种新的对抗性例证 cs.CV

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02430v1)

**Authors**: Ali Borji

**Abstracts**: Almost all adversarial attacks are formulated to add an imperceptible perturbation to an image in order to fool a model. Here, we consider the opposite which is adversarial examples that can fool a human but not a model. A large enough and perceptible perturbation is added to an image such that a model maintains its original decision, whereas a human will most likely make a mistake if forced to decide (or opt not to decide at all). Existing targeted attacks can be reformulated to synthesize such adversarial examples. Our proposed attack, dubbed NKE, is similar in essence to the fooling images, but is more efficient since it uses gradient descent instead of evolutionary algorithms. It also offers a new and unified perspective into the problem of adversarial vulnerability. Experimental results over MNIST and CIFAR-10 datasets show that our attack is quite efficient in fooling deep neural networks. Code is available at https://github.com/aliborji/NKE.

摘要: 几乎所有的对抗性攻击都是为了给图像添加一个难以察觉的扰动，以愚弄模型。在这里，我们考虑的是相反的情况，即可以愚弄人类但不能愚弄模型的对抗性例子。一个足够大和可感知的扰动被添加到图像中，使得模型保持其原始决定，而如果被迫做出决定(或者选择根本不决定)，人类很可能会犯错误。现有的有针对性的攻击可以重新制定，以合成这种对抗性的例子。我们提出的名为NKE的攻击在本质上类似于愚弄图像，但由于它使用了梯度下降而不是进化算法，因此效率更高。它还为敌方脆弱性问题提供了一个新的统一视角。在MNIST和CIFAR-10数据集上的实验结果表明，我们的攻击在欺骗深度神经网络方面是相当有效的。代码可在https://github.com/aliborji/NKE.上找到



## **39. MOVE: Effective and Harmless Ownership Verification via Embedded External Features**

Move：通过嵌入式外部功能进行有效、无害的所有权验证 cs.CR

15 pages. The journal extension of our conference paper in AAAI 2022  (https://ojs.aaai.org/index.php/AAAI/article/view/20036). arXiv admin note:  substantial text overlap with arXiv:2112.03476

**SubmitDate**: 2022-08-04    [paper-pdf](http://arxiv.org/pdf/2208.02820v1)

**Authors**: Yiming Li, Linghui Zhu, Xiaojun Jia, Yang Bai, Yong Jiang, Shu-Tao Xia, Xiaochun Cao

**Abstracts**: Currently, deep neural networks (DNNs) are widely adopted in different applications. Despite its commercial values, training a well-performed DNN is resource-consuming. Accordingly, the well-trained model is valuable intellectual property for its owner. However, recent studies revealed the threats of model stealing, where the adversaries can obtain a function-similar copy of the victim model, even when they can only query the model. In this paper, we propose an effective and harmless model ownership verification (MOVE) to defend against different types of model stealing simultaneously, without introducing new security risks. In general, we conduct the ownership verification by verifying whether a suspicious model contains the knowledge of defender-specified external features. Specifically, we embed the external features by tempering a few training samples with style transfer. We then train a meta-classifier to determine whether a model is stolen from the victim. This approach is inspired by the understanding that the stolen models should contain the knowledge of features learned by the victim model. In particular, we develop our MOVE method under both white-box and black-box settings to provide comprehensive model protection. Extensive experiments on benchmark datasets verify the effectiveness of our method and its resistance to potential adaptive attacks. The codes for reproducing the main experiments of our method are available at \url{https://github.com/THUYimingLi/MOVE}.

摘要: 目前，深度神经网络(DNN)被广泛应用于不同的领域。尽管它具有商业价值，但培训一名表现良好的DNN是耗费资源的。因此，训练有素的车型对其所有者来说是宝贵的知识产权。然而，最近的研究揭示了模型窃取的威胁，其中攻击者可以获得与受害者模型功能相似的副本，即使他们只能查询模型。在本文中，我们提出了一种有效且无害的模型所有权验证(MOVE)来同时防御不同类型的模型窃取，而不会引入新的安全风险。一般来说，我们通过验证可疑模型是否包含防御者指定的外部特征的知识来进行所有权验证。具体地说，我们通过使用风格转移来回火一些训练样本来嵌入外部特征。然后，我们训练元分类器来确定模型是否从受害者那里被盗。这种方法的灵感来自于这样一种理解，即被盗模型应该包含受害者模型学习的特征知识。特别是，我们在白盒和黑盒设置下开发了移动方法，以提供全面的模型保护。在基准数据集上的大量实验验证了该方法的有效性和对潜在自适应攻击的抵抗力。复制我们方法的主要实验的代码可在URL{https://github.com/THUYimingLi/MOVE}.



## **40. Deep VULMAN: A Deep Reinforcement Learning-Enabled Cyber Vulnerability Management Framework**

Deep VULMAN：一种深度强化学习的网络漏洞管理框架 cs.AI

12 pages, 3 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02369v1)

**Authors**: Soumyadeep Hore, Ankit Shah, Nathaniel D. Bastian

**Abstracts**: Cyber vulnerability management is a critical function of a cybersecurity operations center (CSOC) that helps protect organizations against cyber-attacks on their computer and network systems. Adversaries hold an asymmetric advantage over the CSOC, as the number of deficiencies in these systems is increasing at a significantly higher rate compared to the expansion rate of the security teams to mitigate them in a resource-constrained environment. The current approaches are deterministic and one-time decision-making methods, which do not consider future uncertainties when prioritizing and selecting vulnerabilities for mitigation. These approaches are also constrained by the sub-optimal distribution of resources, providing no flexibility to adjust their response to fluctuations in vulnerability arrivals. We propose a novel framework, Deep VULMAN, consisting of a deep reinforcement learning agent and an integer programming method to fill this gap in the cyber vulnerability management process. Our sequential decision-making framework, first, determines the near-optimal amount of resources to be allocated for mitigation under uncertainty for a given system state and then determines the optimal set of prioritized vulnerability instances for mitigation. Our proposed framework outperforms the current methods in prioritizing the selection of important organization-specific vulnerabilities, on both simulated and real-world vulnerability data, observed over a one-year period.

摘要: 网络漏洞管理是网络安全运营中心(CSOC)的一项重要职能，有助于保护组织免受对其计算机和网络系统的网络攻击。与CSOC相比，对手拥有不对称的优势，因为与安全团队的扩张率相比，这些系统中的缺陷数量正在以显著更高的速度增加，以在资源受限的环境中缓解这些缺陷。目前的方法是确定性和一次性决策方法，在确定和选择要缓解的脆弱性时，不考虑未来的不确定性。这些办法还受到资源分配次优的限制，无法灵活地调整其对脆弱抵达人数波动的反应。我们提出了一种新的框架--Deep VULMAN，它由深度强化学习代理和整数规划方法组成，以填补网络漏洞管理过程中的这一空白。我们的顺序决策框架首先确定在给定系统状态下的不确定性情况下为缓解而分配的接近最优的资源量，然后确定用于缓解的最优优先级漏洞实例集。我们提出的框架在优先选择重要的特定于组织的漏洞方面优于目前的方法，该方法基于模拟和真实世界的漏洞数据，在一年的时间内观察到。



## **41. Membership Inference Attacks and Defenses in Neural Network Pruning**

神经网络修剪中的隶属度推理攻击与防御 cs.CR

This paper has been accepted to USENIX Security Symposium 2022. This  is an extended version with more experimental results

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.03335v2)

**Authors**: Xiaoyong Yuan, Lan Zhang

**Abstracts**: Neural network pruning has been an essential technique to reduce the computation and memory requirements for using deep neural networks for resource-constrained devices. Most existing research focuses primarily on balancing the sparsity and accuracy of a pruned neural network by strategically removing insignificant parameters and retraining the pruned model. Such efforts on reusing training samples pose serious privacy risks due to increased memorization, which, however, has not been investigated yet.   In this paper, we conduct the first analysis of privacy risks in neural network pruning. Specifically, we investigate the impacts of neural network pruning on training data privacy, i.e., membership inference attacks. We first explore the impact of neural network pruning on prediction divergence, where the pruning process disproportionately affects the pruned model's behavior for members and non-members. Meanwhile, the influence of divergence even varies among different classes in a fine-grained manner. Enlighten by such divergence, we proposed a self-attention membership inference attack against the pruned neural networks. Extensive experiments are conducted to rigorously evaluate the privacy impacts of different pruning approaches, sparsity levels, and adversary knowledge. The proposed attack shows the higher attack performance on the pruned models when compared with eight existing membership inference attacks. In addition, we propose a new defense mechanism to protect the pruning process by mitigating the prediction divergence based on KL-divergence distance, whose effectiveness has been experimentally demonstrated to effectively mitigate the privacy risks while maintaining the sparsity and accuracy of the pruned models.

摘要: 对于资源受限的设备，为了减少对深层神经网络的计算和存储需求，神经网络修剪已经成为一项基本技术。现有的大多数研究主要集中在通过有策略地去除无关紧要的参数和重新训练修剪的模型来平衡修剪神经网络的稀疏性和准确性。这种重复使用训练样本的努力由于增加了记忆而带来了严重的隐私风险，然而，这一点尚未得到调查。本文首先对神经网络修剪中的隐私风险进行了分析。具体地说，我们研究了神经网络剪枝对训练数据隐私的影响，即成员推理攻击。我们首先探讨了神经网络修剪对预测发散的影响，其中修剪过程不成比例地影响修剪后的模型对成员和非成员的行为。同时，分歧的影响甚至在不同的阶层之间以一种细粒度的方式存在差异。受这种分歧的启发，我们提出了一种针对修剪后的神经网络的自注意成员推理攻击。进行了大量的实验，以严格评估不同的剪枝方法、稀疏程度和敌意知识对隐私的影响。与现有的8种成员关系推理攻击相比，该攻击在剪枝模型上表现出更高的攻击性能。此外，我们提出了一种新的防御机制来保护剪枝过程，通过减少基于KL-发散距离的预测发散来保护剪枝过程，实验证明该机制在保持剪枝模型的稀疏性和准确性的同时有效地缓解了隐私风险。



## **42. Design of secure and robust cognitive system for malware detection**

一种安全健壮的恶意软件检测认知系统设计 cs.CR

arXiv admin note: substantial text overlap with arXiv:2104.06652

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02310v1)

**Authors**: Sanket Shukla

**Abstracts**: Machine learning based malware detection techniques rely on grayscale images of malware and tends to classify malware based on the distribution of textures in graycale images. Albeit the advancement and promising results shown by machine learning techniques, attackers can exploit the vulnerabilities by generating adversarial samples. Adversarial samples are generated by intelligently crafting and adding perturbations to the input samples. There exists majority of the software based adversarial attacks and defenses. To defend against the adversaries, the existing malware detection based on machine learning and grayscale images needs a preprocessing for the adversarial data. This can cause an additional overhead and can prolong the real-time malware detection. So, as an alternative to this, we explore RRAM (Resistive Random Access Memory) based defense against adversaries. Therefore, the aim of this thesis is to address the above mentioned critical system security issues. The above mentioned challenges are addressed by demonstrating proposed techniques to design a secure and robust cognitive system. First, a novel technique to detect stealthy malware is proposed. The technique uses malware binary images and then extract different features from the same and then employ different ML-classifiers on the dataset thus obtained. Results demonstrate that this technique is successful in differentiating classes of malware based on the features extracted. Secondly, I demonstrate the effects of adversarial attacks on a reconfigurable RRAM-neuromorphic architecture with different learning algorithms and device characteristics. I also propose an integrated solution for mitigating the effects of the adversarial attack using the reconfigurable RRAM architecture.

摘要: 基于机器学习的恶意软件检测技术依赖于恶意软件的灰度图像，并倾向于根据灰度图像中纹理的分布对恶意软件进行分类。尽管机器学习技术具有先进性和可喜的结果，但攻击者可以通过生成敌意样本来利用这些漏洞。敌意样本是通过智能地制作并向输入样本添加扰动来生成的。存在大多数基于软件的对抗性攻击和防御。为了防御恶意软件攻击，现有的基于机器学习和灰度图像的恶意软件检测方法需要对恶意数据进行预处理。这可能会导致额外的开销，并会延长实时恶意软件检测的时间。因此，作为一种替代方案，我们探索了基于RRAM(电阻随机存取存储器)的攻击防御。因此，本文的研究目的就是解决上述关键的系统安全问题。上述挑战通过演示设计安全和健壮的认知系统的拟议技术来解决。首先，提出了一种检测隐形恶意软件的新技术。该技术使用恶意软件二值图像，然后从相同的二值图像中提取不同的特征，然后对得到的数据集使用不同的ML分类器。实验结果表明，该方法能够很好地根据提取的特征区分恶意软件的类别。其次，论证了对抗性攻击对具有不同学习算法和设备特征的可重构RRAM-神经形态结构的影响。我还提出了一种使用可重构RRAM体系结构来缓解敌意攻击影响的集成解决方案。



## **43. Generating Image Adversarial Examples by Embedding Digital Watermarks**

嵌入数字水印生成图像对抗性实例 cs.CV

10 pages, 4 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2009.05107v2)

**Authors**: Yuexin Xiang, Tiantian Li, Wei Ren, Tianqing Zhu, Kim-Kwang Raymond Choo

**Abstracts**: With the increasing attention to deep neural network (DNN) models, attacks are also upcoming for such models. For example, an attacker may carefully construct images in specific ways (also referred to as adversarial examples) aiming to mislead the DNN models to output incorrect classification results. Similarly, many efforts are proposed to detect and mitigate adversarial examples, usually for certain dedicated attacks. In this paper, we propose a novel digital watermark-based method to generate image adversarial examples to fool DNN models. Specifically, partial main features of the watermark image are embedded into the host image almost invisibly, aiming to tamper with and damage the recognition capabilities of the DNN models. We devise an efficient mechanism to select host images and watermark images and utilize the improved discrete wavelet transform (DWT) based Patchwork watermarking algorithm with a set of valid hyperparameters to embed digital watermarks from the watermark image dataset into original images for generating image adversarial examples. The experimental results illustrate that the attack success rate on common DNN models can reach an average of 95.47% on the CIFAR-10 dataset and the highest at 98.71%. Besides, our scheme is able to generate a large number of adversarial examples efficiently, concretely, an average of 1.17 seconds for completing the attacks on each image on the CIFAR-10 dataset. In addition, we design a baseline experiment using the watermark images generated by Gaussian noise as the watermark image dataset that also displays the effectiveness of our scheme. Similarly, we also propose the modified discrete cosine transform (DCT) based Patchwork watermarking algorithm. To ensure repeatability and reproducibility, the source code is available on GitHub.

摘要: 随着深度神经网络(DNN)模型受到越来越多的关注，针对这类模型的攻击也随之而来。例如，攻击者可能会以特定的方式仔细构建图像(也称为对抗性示例)，目的是误导DNN模型输出错误的分类结果。同样，提出了许多努力来检测和减轻敌意示例，通常是针对某些特定的专用攻击。本文提出了一种新的基于数字水印的生成图像对抗性实例的方法来欺骗DNN模型。具体地说，水印图像的部分主要特征被嵌入到宿主图像中，几乎是不可见的，目的是篡改和破坏DNN模型的识别能力。我们设计了一种有效的选择宿主图像和水印图像的机制，并利用基于改进的离散小波变换(DWT)的拼接水印算法和一组有效的超参数来将水印图像数据集中的数字水印嵌入到原始图像中，以生成图像对抗性示例。实验结果表明，在CIFAR-10数据集上，常用DNN模型的攻击成功率平均可达95.47%，最高可达98.71%。此外，我们的方案能够高效地生成大量的对抗性实例，具体而言，完成对CIFAR-10数据集上的每幅图像的攻击平均需要1.17秒。此外，利用高斯噪声产生的水印图像作为水印图像数据集，设计了一个基线实验，验证了该算法的有效性。同样，我们还提出了基于修正离散余弦变换(DCT)的补丁水印算法。为了确保可重复性和再现性，GitHub上提供了源代码。



## **44. Abusing Commodity DRAMs in IoT Devices to Remotely Spy on Temperature**

在物联网设备中滥用商品DRAM远程监视温度 cs.CR

Submitted to IEEE TIFS and currently under review

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02125v1)

**Authors**: Florian Frank, Wenjie Xiong, Nikolaos Athanasios Anagnostopoulos, André Schaller, Tolga Arul, Farinaz Koushanfar, Stefan Katzenbeisser, Ulrich Ruhrmair, Jakub Szefer

**Abstracts**: The ubiquity and pervasiveness of modern Internet of Things (IoT) devices opens up vast possibilities for novel applications, but simultaneously also allows spying on, and collecting data from, unsuspecting users to a previously unseen extent. This paper details a new attack form in this vein, in which the decay properties of widespread, off-the-shelf DRAM modules are exploited to accurately sense the temperature in the vicinity of the DRAM-carrying device. Among others, this enables adversaries to remotely and purely digitally spy on personal behavior in users' private homes, or to collect security-critical data in server farms, cloud storage centers, or commercial production lines. We demonstrate that our attack can be performed by merely compromising the software of an IoT device and does not require hardware modifications or physical access at attack time. It can achieve temperature resolutions of up to 0.5{\deg}C over a range of 0{\deg}C to 70{\deg}C in practice. Perhaps most interestingly, it even works in devices that do not have a dedicated temperature sensor on board. To complete our work, we discuss practical attack scenarios as well as possible countermeasures against our temperature espionage attacks.

摘要: 现代物联网(IoT)设备的无处不在和无处不在，为新的应用打开了巨大的可能性，但同时也允许对毫无戒心的用户进行间谍活动，并从他们那里收集数据，达到前所未有的程度。本文详细介绍了一种新的攻击形式，利用广泛存在的现成DRAM模块的衰减特性来准确检测DRAM携带设备附近的温度。其中，这使攻击者能够远程、纯数字地监视用户私人住宅中的个人行为，或者收集服务器群、云存储中心或商业生产线中的安全关键数据。我们证明，我们的攻击可以仅通过危害物联网设备的软件来执行，并且在攻击时不需要修改硬件或进行物理访问。实际应用表明，在0~70℃的温度范围内，温度分辨率最高可达0.5℃。也许最有趣的是，它甚至可以在没有专用温度传感器的设备上工作。为了完成我们的工作，我们讨论了实际的攻击方案以及针对我们的温度间谍攻击的可能对策。



## **45. Local Differential Privacy for Federated Learning**

联合学习中的局部差分隐私 cs.CR

17 pages

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2202.06053v2)

**Authors**: M. A. P. Chamikara, Dongxi Liu, Seyit Camtepe, Surya Nepal, Marthie Grobler, Peter Bertok, Ibrahim Khalil

**Abstracts**: Advanced adversarial attacks such as membership inference and model memorization can make federated learning (FL) vulnerable and potentially leak sensitive private data. Local differentially private (LDP) approaches are gaining more popularity due to stronger privacy notions and native support for data distribution compared to other differentially private (DP) solutions. However, DP approaches assume that the FL server (that aggregates the models) is honest (run the FL protocol honestly) or semi-honest (run the FL protocol honestly while also trying to learn as much information as possible). These assumptions make such approaches unrealistic and unreliable for real-world settings. Besides, in real-world industrial environments (e.g., healthcare), the distributed entities (e.g., hospitals) are already composed of locally running machine learning models (this setting is also referred to as the cross-silo setting). Existing approaches do not provide a scalable mechanism for privacy-preserving FL to be utilized under such settings, potentially with untrusted parties. This paper proposes a new local differentially private FL (named LDPFL) protocol for industrial settings. LDPFL can run in industrial settings with untrusted entities while enforcing stronger privacy guarantees than existing approaches. LDPFL shows high FL model performance (up to 98%) under small privacy budgets (e.g., epsilon = 0.5) in comparison to existing methods.

摘要: 高级对抗性攻击，如成员推理和模型记忆，会使联邦学习(FL)容易受到攻击，并可能泄露敏感的私人数据。与其他差异私有(DP)解决方案相比，本地差异私有(LDP)方法由于更强的隐私概念和对数据分发的本地支持而越来越受欢迎。然而，DP方法假设FL服务器(聚集模型)是诚实的(诚实地运行FL协议)或半诚实的(诚实地运行FL协议，同时还试图了解尽可能多的信息)。这些假设使得这种方法对于现实世界的设置来说是不现实和不可靠的。此外，在真实世界的工业环境(例如，医疗保健)中，分布式实体(例如，医院)已经由本地运行的机器学习模型组成(该设置也被称为跨竖井设置)。现有方法没有提供用于保护隐私的FL的可扩展机制以在这样的设置下使用，可能与不可信方一起使用。提出了一种适用于工业环境的局部差分私有FL协议(简称LDPFL)。LDPFL可以在具有不可信实体的工业环境中运行，同时执行比现有方法更强大的隐私保障。与现有方法相比，LDPFL在较小的隐私预算(例如，epsilon=0.5)下表现出高的FL模型性能(高达98%)。



## **46. SAC-AP: Soft Actor Critic based Deep Reinforcement Learning for Alert Prioritization**

SAC-AP：基于软参与者批评者的深度强化学习告警优先级 cs.CR

8 pages, 8 figures, IEEE WORLD CONGRESS ON COMPUTATIONAL INTELLIGENCE  2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2207.13666v3)

**Authors**: Lalitha Chavali, Tanay Gupta, Paresh Saxena

**Abstracts**: Intrusion detection systems (IDS) generate a large number of false alerts which makes it difficult to inspect true positives. Hence, alert prioritization plays a crucial role in deciding which alerts to investigate from an enormous number of alerts that are generated by IDS. Recently, deep reinforcement learning (DRL) based deep deterministic policy gradient (DDPG) off-policy method has shown to achieve better results for alert prioritization as compared to other state-of-the-art methods. However, DDPG is prone to the problem of overfitting. Additionally, it also has a poor exploration capability and hence it is not suitable for problems with a stochastic environment. To address these limitations, we present a soft actor-critic based DRL algorithm for alert prioritization (SAC-AP), an off-policy method, based on the maximum entropy reinforcement learning framework that aims to maximize the expected reward while also maximizing the entropy. Further, the interaction between an adversary and a defender is modeled as a zero-sum game and a double oracle framework is utilized to obtain the approximate mixed strategy Nash equilibrium (MSNE). SAC-AP finds robust alert investigation policies and computes pure strategy best response against opponent's mixed strategy. We present the overall design of SAC-AP and evaluate its performance as compared to other state-of-the art alert prioritization methods. We consider defender's loss, i.e., the defender's inability to investigate the alerts that are triggered due to attacks, as the performance metric. Our results show that SAC-AP achieves up to 30% decrease in defender's loss as compared to the DDPG based alert prioritization method and hence provides better protection against intrusions. Moreover, the benefits are even higher when SAC-AP is compared to other traditional alert prioritization methods including Uniform, GAIN, RIO and Suricata.

摘要: 入侵检测系统(入侵检测系统)产生大量的错误警报，使得对真实阳性的检测变得困难。因此，警报优先级在决定从由入侵检测系统生成的大量警报中调查哪些警报时起着至关重要的作用。近年来，与其他方法相比，基于深度强化学习(DRL)的深度确定性策略梯度(DDPG)非策略方法能够获得更好的告警优先级排序结果。然而，DDPG容易出现过度匹配的问题。此外，它的探测能力也很差，因此不适合于具有随机环境的问题。针对这些局限性，我们提出了一种基于软参与者-批评者的DRL警报优先排序算法(SAC-AP)，这是一种基于最大熵强化学习框架的非策略方法，旨在最大化期望回报的同时最大化熵。在此基础上，将对手和防御者之间的相互作用建模为零和博弈，并利用双预言框架得到近似的混合策略纳什均衡。SAC-AP发现稳健的警戒调查策略，并针对对手的混合策略计算纯策略的最佳响应。我们介绍了SAC-AP的总体设计，并与其他最先进的警报优先排序方法进行了比较，评估了其性能。我们将防御者的损失，即防御者无法调查由于攻击而触发的警报作为性能指标。结果表明，与基于DDPG的告警优先级排序方法相比，SAC-AP可以减少高达30%的防御者损失，从而提供更好的防御入侵保护。此外，当SAC-AP与其他传统的警报优先排序方法(包括Uniform、Gain、Rio和Suricata)相比时，好处甚至更高。



## **47. Spectrum Focused Frequency Adversarial Attacks for Automatic Modulation Classification**

用于自动调制分类的频谱聚焦频率对抗攻击 cs.CR

6 pages, 9 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01919v1)

**Authors**: Sicheng Zhang, Jiarun Yu, Zhida Bao, Shiwen Mao, Yun Lin

**Abstracts**: Artificial intelligence (AI) technology has provided a potential solution for automatic modulation recognition (AMC). Unfortunately, AI-based AMC models are vulnerable to adversarial examples, which seriously threatens the efficient, secure and trusted application of AI in AMC. This issue has attracted the attention of researchers. Various studies on adversarial attacks and defenses evolve in a spiral. However, the existing adversarial attack methods are all designed in the time domain. They introduce more high-frequency components in the frequency domain, due to abrupt updates in the time domain. For this issue, from the perspective of frequency domain, we propose a spectrum focused frequency adversarial attacks (SFFAA) for AMC model, and further draw on the idea of meta-learning, propose a Meta-SFFAA algorithm to improve the transferability in the black-box attacks. Extensive experiments, qualitative and quantitative metrics demonstrate that the proposed algorithm can concentrate the adversarial energy on the spectrum where the signal is located, significantly improve the adversarial attack performance while maintaining the concealment in the frequency domain.

摘要: 人工智能(AI)技术为自动调制识别(AMC)提供了一种潜在的解决方案。不幸的是，基于人工智能的AMC模型容易受到敌意例子的攻击，这严重威胁了人工智能在AMC中的高效、安全和可信的应用。这个问题已经引起了研究人员的关注。关于对抗性攻击和防御的各种研究呈螺旋式发展。然而，现有的对抗性攻击方法都是在时间域设计的。由于时间域中的突然更新，它们在频域中引入了更多的高频分量。针对这一问题，从频域的角度出发，提出了一种针对AMC模型的频谱聚焦频率对抗攻击算法(SFFAA)，并进一步借鉴元学习的思想，提出了一种Meta-SFFAA算法来提高黑盒攻击的可转移性。大量的实验、定性和定量指标表明，该算法可以将对抗能量集中在信号所在的频谱上，在保持频域隐蔽性的同时，显著提高了对抗攻击的性能。



## **48. On the Evaluation of User Privacy in Deep Neural Networks using Timing Side Channel**

基于时序侧通道的深度神经网络用户隐私评估研究 cs.CR

15 pages, 20 figures

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01113v2)

**Authors**: Shubhi Shukla, Manaar Alam, Sarani Bhattacharya, Debdeep Mukhopadhyay, Pabitra Mitra

**Abstracts**: Recent Deep Learning (DL) advancements in solving complex real-world tasks have led to its widespread adoption in practical applications. However, this opportunity comes with significant underlying risks, as many of these models rely on privacy-sensitive data for training in a variety of applications, making them an overly-exposed threat surface for privacy violations. Furthermore, the widespread use of cloud-based Machine-Learning-as-a-Service (MLaaS) for its robust infrastructure support has broadened the threat surface to include a variety of remote side-channel attacks. In this paper, we first identify and report a novel data-dependent timing side-channel leakage (termed Class Leakage) in DL implementations originating from non-constant time branching operation in a widely used DL framework PyTorch. We further demonstrate a practical inference-time attack where an adversary with user privilege and hard-label black-box access to an MLaaS can exploit Class Leakage to compromise the privacy of MLaaS users. DL models are vulnerable to Membership Inference Attack (MIA), where an adversary's objective is to deduce whether any particular data has been used while training the model. In this paper, as a separate case study, we demonstrate that a DL model secured with differential privacy (a popular countermeasure against MIA) is still vulnerable to MIA against an adversary exploiting Class Leakage. We develop an easy-to-implement countermeasure by making a constant-time branching operation that alleviates the Class Leakage and also aids in mitigating MIA. We have chosen two standard benchmarking image classification datasets, CIFAR-10 and CIFAR-100 to train five state-of-the-art pre-trained DL models, over two different computing environments having Intel Xeon and Intel i7 processors to validate our approach.

摘要: 最近深度学习(DL)在解决复杂现实世界任务方面的进步导致了它在实际应用中的广泛采用。然而，这种机会伴随着巨大的潜在风险，因为这些模型中的许多依赖于隐私敏感数据来进行各种应用程序的培训，使它们成为侵犯隐私的过度暴露的威胁表面。此外，基于云的机器学习即服务(MLaaS)因其强大的基础设施支持而广泛使用，扩大了威胁面，包括各种远程侧通道攻击。在这篇文章中，我们首先识别和报告了一种新的数据相关的定时侧通道泄漏(称为类泄漏)，该泄漏是由广泛使用的动态链接库框架中的非常数时间分支操作引起的。我们进一步展示了一个实用的推理时间攻击，其中具有用户权限和硬标签黑盒访问MLaaS的攻击者可以利用类泄漏来危害MLaaS用户的隐私。DL模型容易受到成员推理攻击(MIA)，对手的目标是推断在训练模型时是否使用了特定的数据。在本文中，作为一个单独的案例研究，我们证明了在差异隐私保护下的DL模型(一种流行的针对MIA的对策)仍然容易受到MIA对利用类泄漏的攻击者的攻击。我们开发了一种易于实现的对策，通过进行恒定时间分支操作来缓解类泄漏，并帮助缓解MIA。我们选择了两个标准的基准图像分类数据集，CIFAR-10和CIFAR-100来训练五个最先进的预训练的DL模型，在两种不同的计算环境中使用Intel Xeon和Intel i7处理器来验证我们的方法。



## **49. Adversarial Attacks on ASR Systems: An Overview**

对ASR系统的敌意攻击：综述 cs.SD

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.02250v1)

**Authors**: Xiao Zhang, Hao Tan, Xuan Huang, Denghui Zhang, Keke Tang, Zhaoquan Gu

**Abstracts**: With the development of hardware and algorithms, ASR(Automatic Speech Recognition) systems evolve a lot. As The models get simpler, the difficulty of development and deployment become easier, ASR systems are getting closer to our life. On the one hand, we often use APPs or APIs of ASR to generate subtitles and record meetings. On the other hand, smart speaker and self-driving car rely on ASR systems to control AIoT devices. In past few years, there are a lot of works on adversarial examples attacks against ASR systems. By adding a small perturbation to the waveforms, the recognition results make a big difference. In this paper, we describe the development of ASR system, different assumptions of attacks, and how to evaluate these attacks. Next, we introduce the current works on adversarial examples attacks from two attack assumptions: white-box attack and black-box attack. Different from other surveys, we pay more attention to which layer they perturb waveforms in ASR system, the relationship between these attacks, and their implementation methods. We focus on the effect of their works.

摘要: 随着硬件和算法的发展，ASR(Automatic Speech Recognition，自动语音识别)系统也在不断发展。随着模型变得更简单，开发和部署的难度变得更容易，ASR系统越来越接近我们的生活。一方面，我们经常使用ASR的APP或API来生成字幕和录制会议。另一方面，智能音箱和自动驾驶汽车依靠ASR系统来控制AIoT设备。在过去的几年里，已经有很多关于针对ASR系统的对抗性例子攻击的工作。通过对波形添加小的扰动，识别结果有很大的不同。在本文中，我们描述了ASR系统的发展，不同的攻击假设，以及如何评估这些攻击。接下来，我们从白盒攻击和黑盒攻击两个攻击假设出发，介绍了目前对抗性例子攻击的研究成果。与其他研究不同的是，我们更关注它们对ASR系统中的哪一层波形的扰动，这些攻击之间的关系，以及它们的实现方法。我们关注的是他们作品的效果。



## **50. Robust Graph Neural Networks using Weighted Graph Laplacian**

基于加权图拉普拉斯的稳健图神经网络 cs.LG

Accepted at IEEE International Conference on Signal Processing and  Communications (SPCOM), 2022

**SubmitDate**: 2022-08-03    [paper-pdf](http://arxiv.org/pdf/2208.01853v1)

**Authors**: Bharat Runwal, Vivek, Sandeep Kumar

**Abstracts**: Graph neural network (GNN) is achieving remarkable performances in a variety of application domains. However, GNN is vulnerable to noise and adversarial attacks in input data. Making GNN robust against noises and adversarial attacks is an important problem. The existing defense methods for GNNs are computationally demanding and are not scalable. In this paper, we propose a generic framework for robustifying GNN known as Weighted Laplacian GNN (RWL-GNN). The method combines Weighted Graph Laplacian learning with the GNN implementation. The proposed method benefits from the positive semi-definiteness property of Laplacian matrix, feature smoothness, and latent features via formulating a unified optimization framework, which ensures the adversarial/noisy edges are discarded and connections in the graph are appropriately weighted. For demonstration, the experiments are conducted with Graph convolutional neural network(GCNN) architecture, however, the proposed framework is easily amenable to any existing GNN architecture. The simulation results with benchmark dataset establish the efficacy of the proposed method, both in accuracy and computational efficiency. Code can be accessed at https://github.com/Bharat-Runwal/RWL-GNN.

摘要: 图形神经网络(GNN)在各种应用领域都取得了令人瞩目的成绩。然而，GNN很容易受到输入数据中的噪声和对抗性攻击。如何使GNN对噪声和敌意攻击具有健壮性是一个重要的问题。现有的GNN防御方法计算量大且不可扩展。在本文中，我们提出了一种称为加权拉普拉斯GNN(RWL-GNN)的通用GNN框架。该方法将加权图拉普拉斯学习与GNN实现相结合。该方法充分利用了拉普拉斯矩阵的正半定性、特征的光滑性和潜在特征，建立了统一的优化框架，确保了对敌边/噪声边的丢弃和图中连接的适当加权。为了进行演示，实验使用了图卷积神经网络(GCNN)结构，然而，所提出的框架可以很容易地服从于任何现有的GNN结构。利用基准数据集的仿真结果验证了该方法在精度和计算效率上的有效性。代码可在https://github.com/Bharat-Runwal/RWL-GNN.上访问



