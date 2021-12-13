# Latest Adversarial Attack Papers
**update at 2021-12-13 16:17:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Preemptive Image Robustification for Protecting Users against Man-in-the-Middle Adversarial Attacks**

保护用户免受中间人攻击的抢占式图像盗用 cs.LG

Accepted and to appear at AAAI 2022

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05634v1)

**Authors**: Seungyong Moon, Gaon An, Hyun Oh Song

**Abstracts**: Deep neural networks have become the driving force of modern image recognition systems. However, the vulnerability of neural networks against adversarial attacks poses a serious threat to the people affected by these systems. In this paper, we focus on a real-world threat model where a Man-in-the-Middle adversary maliciously intercepts and perturbs images web users upload online. This type of attack can raise severe ethical concerns on top of simple performance degradation. To prevent this attack, we devise a novel bi-level optimization algorithm that finds points in the vicinity of natural images that are robust to adversarial perturbations. Experiments on CIFAR-10 and ImageNet show our method can effectively robustify natural images within the given modification budget. We also show the proposed method can improve robustness when jointly used with randomized smoothing.

摘要: 深度神经网络已经成为现代图像识别系统的驱动力。然而，神经网络对敌意攻击的脆弱性对受这些系统影响的人们构成了严重威胁。在这篇文章中，我们关注的是一个真实世界的威胁模型，在这个模型中，中间人对手恶意截取和干扰网络用户在线上传的图像。除了简单的性能降级之外，此类攻击还会引发严重的道德问题。为了防止这种攻击，我们设计了一种新的双层优化算法，该算法在自然图像附近寻找对对手扰动具有鲁棒性的点。在CIFAR-10和ImageNet上的实验表明，我们的方法可以在给定的修改预算内有效地增强自然图像的鲁棒性。我们还表明，当与随机平滑联合使用时，所提出的方法可以提高鲁棒性。



## **2. How Private Is Your RL Policy? An Inverse RL Based Analysis Framework**

您的RL政策有多私密？一种基于逆向RL的分析框架 cs.LG

15 pages, 7 figures, 5 tables, version accepted at AAAI 2022

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05495v1)

**Authors**: Kritika Prakash, Fiza Husain, Praveen Paruchuri, Sujit P. Gujar

**Abstracts**: Reinforcement Learning (RL) enables agents to learn how to perform various tasks from scratch. In domains like autonomous driving, recommendation systems, and more, optimal RL policies learned could cause a privacy breach if the policies memorize any part of the private reward. We study the set of existing differentially-private RL policies derived from various RL algorithms such as Value Iteration, Deep Q Networks, and Vanilla Proximal Policy Optimization. We propose a new Privacy-Aware Inverse RL (PRIL) analysis framework, that performs reward reconstruction as an adversarial attack on private policies that the agents may deploy. For this, we introduce the reward reconstruction attack, wherein we seek to reconstruct the original reward from a privacy-preserving policy using an Inverse RL algorithm. An adversary must do poorly at reconstructing the original reward function if the agent uses a tightly private policy. Using this framework, we empirically test the effectiveness of the privacy guarantee offered by the private algorithms on multiple instances of the FrozenLake domain of varying complexities. Based on the analysis performed, we infer a gap between the current standard of privacy offered and the standard of privacy needed to protect reward functions in RL. We do so by quantifying the extent to which each private policy protects the reward function by measuring distances between the original and reconstructed rewards.

摘要: 强化学习(RL)使座席能够学习如何从头开始执行各种任务。在自动驾驶、推荐系统等领域，如果策略记住了私人奖励的任何部分，那么学习到的最优RL策略可能会导致隐私泄露。我们研究了现有的差分私有RL策略集，这些策略来自各种RL算法，如值迭代、深度Q网络和Vanilla近似值策略优化。我们提出了一种新的隐私感知逆向RL(Pril)分析框架，该框架将报酬重构作为对代理可能部署的私有策略的对抗性攻击来执行。为此，我们引入了奖赏重构攻击，利用逆RL算法从隐私保护策略中重构原始奖赏。如果代理人使用严格的私人策略，则对手在重建原始奖励函数方面肯定做得很差。使用该框架，我们在不同复杂度的FrozenLake域的多个实例上对私有算法提供的隐私保证的有效性进行了实证测试。在分析的基础上，我们推断现行的隐私标准与RL中保护奖励功能所需的隐私标准之间存在差距。为此，我们通过测量原始奖励和重建奖励之间的距离来量化每个私人政策保护奖励功能的程度。



## **3. SoK: On the Security & Privacy in Federated Learning**

SOK：论联合学习中的安全与隐私 cs.CR

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05423v1)

**Authors**: Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstracts**: Advances in Machine Learning (ML) and its wide range of applications boosted its popularity. Recent privacy awareness initiatives as the EU General Data Protection Regulation (GDPR) - European Parliament and Council Regulation No 2016/679, subdued ML to privacy and security assessments. Federated Learning (FL) grants a privacy-driven, decentralized training scheme that improves ML models' security. The industry's fast-growing adaptation and security evaluations of FL technology exposed various vulnerabilities. Depending on the FL phase, i.e., training or inference, the adversarial actor capabilities, and the attack type threaten FL's confidentiality, integrity, or availability (CIA). Therefore, the researchers apply the knowledge from distinct domains as countermeasures, like cryptography and statistics.   This work assesses the CIA of FL by reviewing the state-of-the-art (SoTA) for creating a threat model that embraces the attack's surface, adversarial actors, capabilities, and goals. We propose the first unifying taxonomy for attacks and defenses by applying this model. Additionally, we provide critical insights extracted by applying the suggested novel taxonomies to the SoTA, yielding promising future research directions.

摘要: 机器学习(ML)的发展及其广泛的应用推动了它的普及。最近的隐私意识举措，如欧盟一般数据保护条例(GDPR)-欧洲议会和理事会第2016/679号条例，降低了ML对隐私和安全评估的要求。联邦学习(FL)提供了一种隐私驱动的、分散的训练方案，提高了ML模型的安全性。业界对FL技术的快速适应和安全评估暴露了各种漏洞。根据FL阶段，即训练或推理，敌方参与者的能力和攻击类型威胁FL的机密性、完整性或可用性(CIA)。因此，研究人员将来自不同领域的知识作为对策，如密码学和统计学。这项工作评估中央情报局的FL通过审查国家的艺术(SOTA)创建了一个威胁模型，涵盖了攻击的表面，敌对行为者，能力和目标。通过应用该模型，我们提出了第一个统一的攻击和防御分类法。此外，我们提供了通过将建议的新分类法应用于SOTA而提取的批判性见解，产生了有前途的未来研究方向。



## **4. Cross-Modal Transferable Adversarial Attacks from Images to Videos**

从图像到视频的跨模态可转移敌意攻击 cs.CV

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05379v1)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Recent studies have shown that adversarial examples hand-crafted on one white-box model can be used to attack other black-box models. Such cross-model transferability makes it feasible to perform black-box attacks, which has raised security concerns for real-world DNNs applications. Nevertheless, existing works mostly focus on investigating the adversarial transferability across different deep models that share the same modality of input data. The cross-modal transferability of adversarial perturbation has never been explored. This paper investigates the transferability of adversarial perturbation across different modalities, i.e., leveraging adversarial perturbation generated on white-box image models to attack black-box video models. Specifically, motivated by the observation that the low-level feature space between images and video frames are similar, we propose a simple yet effective cross-modal attack method, named as Image To Video (I2V) attack. I2V generates adversarial frames by minimizing the cosine similarity between features of pre-trained image models from adversarial and benign examples, then combines the generated adversarial frames to perform black-box attacks on video recognition models. Extensive experiments demonstrate that I2V can achieve high attack success rates on different black-box video recognition models. On Kinetics-400 and UCF-101, I2V achieves an average attack success rate of 77.88% and 65.68%, respectively, which sheds light on the feasibility of cross-modal adversarial attacks.

摘要: 最近的研究表明，在一个白盒模型上手工制作的对抗性例子可以用来攻击其他黑盒模型。这种跨模型的可移植性使得执行黑盒攻击成为可能，这给现实世界中的DNNs应用带来了安全隐患。然而，现有的工作大多集中在研究具有相同输入数据模态的不同深度模型之间的对抗性转移。对抗性扰动的跨模态可转移性从未被探讨过。研究了对抗性扰动在不同模态之间的可传递性，即利用白盒图像模型产生的对抗性扰动攻击黑盒视频模型。具体地说，基于图像和视频帧之间的低层特征空间相似的观察，我们提出了一种简单而有效的跨模式攻击方法，称为图像到视频(I2V)攻击。I2V通过最小化来自对抗性和良性示例的预训练图像模型的特征之间的余弦相似度来生成对抗性帧，然后将生成的对抗性帧组合起来对视频识别模型进行黑盒攻击。大量实验表明，I2V在不同的黑盒视频识别模型上都能获得较高的攻击成功率。在Kinetics-400和UCF-101上，I2V的平均攻击成功率分别为77.88%和65.68%，说明了跨模式对抗攻击的可行性。



## **5. Efficient Action Poisoning Attacks on Linear Contextual Bandits**

线性上下文环上的有效动作毒化攻击 cs.LG

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05367v1)

**Authors**: Guanlin Liu, Lifeng Lai

**Abstracts**: Contextual bandit algorithms have many applicants in a variety of scenarios. In order to develop trustworthy contextual bandit systems, understanding the impacts of various adversarial attacks on contextual bandit algorithms is essential. In this paper, we propose a new class of attacks: action poisoning attacks, where an adversary can change the action signal selected by the agent. We design action poisoning attack schemes against linear contextual bandit algorithms in both white-box and black-box settings. We further analyze the cost of the proposed attack strategies for a very popular and widely used bandit algorithm: LinUCB. We show that, in both white-box and black-box settings, the proposed attack schemes can force the LinUCB agent to pull a target arm very frequently by spending only logarithm cost.

摘要: 上下文强盗算法在各种情况下都有很多应用程序。为了开发可信的上下文盗贼系统，了解各种对抗性攻击对上下文盗贼算法的影响是至关重要的。在本文中，我们提出了一类新的攻击：动作中毒攻击，在这种攻击中，对手可以改变Agent选择的动作信号。我们设计了白盒和黑盒环境下针对线性上下文盗贼算法的动作中毒攻击方案。我们进一步分析了一种非常流行和广泛使用的盗版算法LinUCB所提出的攻击策略的代价。我们证明了在白盒和黑盒环境下，所提出的攻击方案仅需花费对数代价就可以迫使LinUCB代理非常频繁地拉出目标手臂。



## **6. RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit**

RamBoAttack：一种鲁棒查询高效的深度神经网络决策开发 cs.LG

To appear in NDSS 2022

**SubmitDate**: 2021-12-10    [paper-pdf](http://arxiv.org/pdf/2112.05282v1)

**Authors**: Viet Quoc Vo, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstracts**: Machine learning models are critically susceptible to evasion attacks from adversarial examples. Generally, adversarial examples, modified inputs deceptively similar to the original input, are constructed under whitebox settings by adversaries with full access to the model. However, recent attacks have shown a remarkable reduction in query numbers to craft adversarial examples using blackbox attacks. Particularly, alarming is the ability to exploit the classification decision from the access interface of a trained model provided by a growing number of Machine Learning as a Service providers including Google, Microsoft, IBM and used by a plethora of applications incorporating these models. The ability of an adversary to exploit only the predicted label from a model to craft adversarial examples is distinguished as a decision-based attack. In our study, we first deep dive into recent state-of-the-art decision-based attacks in ICLR and SP to highlight the costly nature of discovering low distortion adversarial employing gradient estimation methods. We develop a robust query efficient attack capable of avoiding entrapment in a local minimum and misdirection from noisy gradients seen in gradient estimation methods. The attack method we propose, RamBoAttack, exploits the notion of Randomized Block Coordinate Descent to explore the hidden classifier manifold, targeting perturbations to manipulate only localized input features to address the issues of gradient estimation methods. Importantly, the RamBoAttack is more robust to the different sample inputs available to an adversary and the targeted class. Overall, for a given target class, RamBoAttack is demonstrated to be more robust at achieving a lower distortion within a given query budget. We curate our extensive results using the large-scale high-resolution ImageNet dataset and open-source our attack, test samples and artifacts on GitHub.

摘要: 机器学习模型极易受到敌意例子的逃避攻击。通常，敌意示例(修改后的输入欺骗性地类似于原始输入)是由具有完全访问模型的敌手在白盒设置下构建的。然而，最近的攻击显示，使用黑盒攻击伪造敌意示例的查询数量显著减少。具体地说，警报是利用由越来越多的机器学习即服务提供商(包括Google、Microsoft、IBM)提供的训练模型的访问接口中的分类决策的能力，并且被结合这些模型的大量应用程序使用。对手仅利用模型中预测的标签来制作敌意示例的能力被区分为基于决策的攻击。在我们的研究中，我们首先深入研究了ICLR和SP中最新的基于决策的攻击，以强调使用梯度估计方法发现低失真攻击的代价。我们开发了一种健壮的查询高效攻击，能够避免陷入局部最小值和从梯度估计方法中看到的噪声梯度的误导。我们提出的攻击方法RamBoAttack利用随机化挡路坐标下降的概念来探索隐藏的分类器流形，针对扰动只操纵局部输入特征来解决梯度估计方法的问题。重要的是，RamBoAttack对于对手和目标类可用的不同样本输入更加健壮。总体而言，对于给定的目标类，RamBoAttack在给定的查询预算内实现较低的失真方面表现得更加健壮。我们使用大规模高分辨率ImageNet数据集和在GitHub上开源的攻击、测试样本和人工制品来管理我们广泛的结果。



## **7. The Dilemma Between Data Transformations and Adversarial Robustness for Time Series Application Systems**

时序应用系统中数据转换与对抗鲁棒性的两难选择 cs.LG

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2006.10885v2)

**Authors**: Sheila Alemany, Niki Pissinou

**Abstracts**: Adversarial examples, or nearly indistinguishable inputs created by an attacker, significantly reduce machine learning accuracy. Theoretical evidence has shown that the high intrinsic dimensionality of datasets facilitates an adversary's ability to develop effective adversarial examples in classification models. Adjacently, the presentation of data to a learning model impacts its performance. For example, we have seen this through dimensionality reduction techniques used to aid with the generalization of features in machine learning applications. Thus, data transformation techniques go hand-in-hand with state-of-the-art learning models in decision-making applications such as intelligent medical or military systems. With this work, we explore how data transformations techniques such as feature selection, dimensionality reduction, or trend extraction techniques may impact an adversary's ability to create effective adversarial samples on a recurrent neural network. Specifically, we analyze it from the perspective of the data manifold and the presentation of its intrinsic features. Our evaluation empirically shows that feature selection and trend extraction techniques may increase the RNN's vulnerability. A data transformation technique reduces the vulnerability to adversarial examples only if it approximates the dataset's intrinsic dimension, minimizes codimension, and maintains higher manifold coverage.

摘要: 敌意的例子，或者攻击者创建的几乎无法区分的输入，都会显著降低机器学习的准确性。理论证据表明，数据集的高内在维数促进了对手在分类模型中开发有效的对抗性实例的能力。此外，将数据呈现给学习模型会影响其性能。例如，我们通过用于帮助机器学习应用程序中的特征泛化的降维技术看到了这一点。因此，在智能医疗或军事系统等决策应用程序中，数据转换技术与最先进的学习模型齐头并进。通过这项工作，我们探索了数据转换技术(如特征选择、降维或趋势提取技术)如何影响对手在递归神经网络上创建有效对抗样本的能力。具体地说，我们从数据流形和其内在特征的呈现的角度对其进行分析。我们的评估经验表明，特征选择和趋势提取技术可能会增加RNN的脆弱性。数据转换技术只有在接近数据集的内在维度、最小化余维并保持较高的流形覆盖时才能降低对敌意示例的易损性。



## **8. Spinning Language Models for Propaganda-As-A-Service**

宣传即服务的旋转语言模型 cs.CR

arXiv admin note: text overlap with arXiv:2107.10443

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.05224v1)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstracts**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their outputs so as to support an adversary-chosen sentiment or point of view, but only when the input contains adversary-chosen trigger words. For example, a spinned summarization model would output positive summaries of any text that mentions the name of some individual or organization.   Model spinning enables propaganda-as-a-service. An adversary can create customized language models that produce desired spins for chosen triggers, then deploy them to generate disinformation (a platform attack), or else inject them into ML training pipelines (a supply-chain attack), transferring malicious functionality to downstream models.   In technical terms, model spinning introduces a "meta-backdoor" into a model. Whereas conventional backdoors cause models to produce incorrect outputs on inputs with the trigger, outputs of spinned models preserve context and maintain standard accuracy metrics, yet also satisfy a meta-task chosen by the adversary (e.g., positive sentiment).   To demonstrate feasibility of model spinning, we develop a new backdooring technique. It stacks the adversarial meta-task onto a seq2seq model, backpropagates the desired meta-task output to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. We evaluate this attack on language generation, summarization, and translation models with different triggers and meta-tasks such as sentiment, toxicity, and entailment. Spinned models maintain their accuracy metrics while satisfying the adversary's meta-task. In supply chain attack the spin transfers to downstream models.   Finally, we propose a black-box, meta-task-independent defense to detect models that selectively apply spin to inputs with a certain trigger.

摘要: 我们调查了神经序列到序列(Seq2seq)模型的一种新威胁：训练时间攻击，它导致模型的输出“旋转”，以支持对手选择的情绪或观点，但只有当输入包含对手选择的触发词时。例如，旋转的摘要模型将输出提及某个个人或组织名称的任何文本的正面摘要。模型旋转实现了宣传即服务。对手可以创建自定义语言模型，为选定的触发器生成所需的旋转，然后部署它们以生成虚假信息(平台攻击)，或者将它们注入ML训练管道(供应链攻击)，将恶意功能转移到下游模型。在技术术语中，模型旋转在模型中引入了“元后门”。传统的后门导致模型在具有触发器的输入上产生不正确的输出，而旋转模型的输出保留上下文并保持标准的准确性度量，但也满足对手选择的元任务(例如，积极情绪)。为了论证模型旋转的可行性，我们开发了一种新的回溯技术。它将敌意元任务堆叠到seq2seq模型上，将所需的元任务输出反向传播到我们称为“伪词”的单词嵌入空间中的点，并使用伪词来移动seq2seq模型的整个输出分布。我们用不同的触发因素和元任务(如情感、毒性和蕴涵)来评估这种对语言生成、摘要和翻译模型的攻击。旋转模型在满足对手元任务的同时保持其准确性度量。在供应链攻击中，旋转转移到下游模型。最后，我们提出了一种黑盒、元任务无关的防御方法来检测具有特定触发器的有选择地对输入施加自旋的模型。



## **9. Towards Understanding Adversarial Robustness of Optical Flow Networks**

理解光流网络的对抗健壮性 cs.CV

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2103.16255v2)

**Authors**: Simon Schrodi, Tonmoy Saikia, Thomas Brox

**Abstracts**: Recent work demonstrated the lack of robustness of optical flow networks to physical, patch-based adversarial attacks. The possibility to physically attack a basic component of automotive systems is a reason for serious concerns. In this paper, we analyze the cause of the problem and show that the lack of robustness is rooted in the classical aperture problem of optical flow estimation in combination with bad choices in the details of the network architecture. We show how these mistakes can be rectified in order to make optical flow networks robust to physical, patch-based attacks. Additionally, we take a look at global white-box attacks in the scope of optical flow. We find that targeted white-box attacks can be crafted to bias flow estimation models towards any desired output, but this requires access to the input images and model weights. Our results indicate that optical flow networks are robust to universal attacks.

摘要: 最近的研究表明，光流网络对物理的、基于补丁的敌意攻击缺乏健壮性。对汽车系统的基本部件进行物理攻击的可能性是一个令人严重担忧的原因。本文分析了问题产生的原因，指出健壮性不足的根源在于经典的光流估计孔径问题和网络结构细节选择不当。我们展示了如何纠正这些错误，以便使光流网络对物理的、基于补丁的攻击具有健壮性。此外，我们还在光流的范围内研究了全局白盒攻击。我们发现，可以精心设计有针对性的白盒攻击来使流量估计模型偏向任何期望的输出，但这需要访问输入图像和模型权重。结果表明，光流网络对普遍攻击具有较强的鲁棒性。



## **10. Mutual Adversarial Training: Learning together is better than going alone**

相互对抗训练，一起学习总比单独去要好 cs.LG

Under submission

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.05005v1)

**Authors**: Jiang Liu, Chun Pong Lau, Hossein Souri, Soheil Feizi, Rama Chellappa

**Abstracts**: Recent studies have shown that robustness to adversarial attacks can be transferred across networks. In other words, we can make a weak model more robust with the help of a strong teacher model. We ask if instead of learning from a static teacher, can models "learn together" and "teach each other" to achieve better robustness? In this paper, we study how interactions among models affect robustness via knowledge distillation. We propose mutual adversarial training (MAT), in which multiple models are trained together and share the knowledge of adversarial examples to achieve improved robustness. MAT allows robust models to explore a larger space of adversarial samples, and find more robust feature spaces and decision boundaries. Through extensive experiments on CIFAR-10 and CIFAR-100, we demonstrate that MAT can effectively improve model robustness and outperform state-of-the-art methods under white-box attacks, bringing $\sim$8% accuracy gain to vanilla adversarial training (AT) under PGD-100 attacks. In addition, we show that MAT can also mitigate the robustness trade-off among different perturbation types, bringing as much as 13.1% accuracy gain to AT baselines against the union of $l_\infty$, $l_2$ and $l_1$ attacks. These results show the superiority of the proposed method and demonstrate that collaborative learning is an effective strategy for designing robust models.

摘要: 最近的研究表明，对敌意攻击的健壮性可以通过网络传递。换句话说，我们可以借助一个强教师模型使一个弱模型变得更健壮。我们问，是否与其向静电老师学习，模特们能不能“一起学习”、“互相传授”，以达到更好的健壮性？在本文中，我们研究了模型之间的交互如何通过知识提取来影响健壮性。我们提出了相互对抗性训练(MAT)，将多个模型一起训练，共享对抗性实例的知识，以达到提高鲁棒性的目的。MAT允许鲁棒模型探索更大的对抗性样本空间，并找到更稳健的特征空间和决策边界。通过在CIFAR-10和CIFAR-100上的大量实验，我们证明MAT可以有效地提高模型的鲁棒性，在白盒攻击下的性能优于最先进的方法，在PGD-100攻击下给普通的对抗性训练(AT)带来了8%的准确率提升。此外，我们还表明，MAT还可以缓解不同扰动类型之间的鲁棒性权衡，针对$l_\infty$、$l_2$和$l_1$攻击的联合，给AT基线带来高达13.1%的精度提升。这些结果表明了该方法的优越性，证明了协作学习是设计鲁棒模型的一种有效策略。



## **11. FCA: Learning a 3D Full-coverage Vehicle Camouflage for Multi-view Physical Adversarial Attack**

FCA：学习用于多视点物理对抗攻击的3D全覆盖车辆伪装 cs.CV

9 pages, 5 figures

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2109.07193v2)

**Authors**: Donghua Wang, Tingsong Jiang, Jialiang Sun, Weien Zhou, Xiaoya Zhang, Zhiqiang Gong, Wen Yao, Xiaoqian Chen

**Abstracts**: Physical adversarial attacks in object detection have attracted increasing attention. However, most previous works focus on hiding the objects from the detector by generating an individual adversarial patch, which only covers the planar part of the vehicle's surface and fails to attack the detector in physical scenarios for multi-view, long-distance and partially occluded objects. To bridge the gap between digital attacks and physical attacks, we exploit the full 3D vehicle surface to propose a robust Full-coverage Camouflage Attack (FCA) to fool detectors. Specifically, we first try rendering the nonplanar camouflage texture over the full vehicle surface. To mimic the real-world environment conditions, we then introduce a transformation function to transfer the rendered camouflaged vehicle into a photo realistic scenario. Finally, we design an efficient loss function to optimize the camouflage texture. Experiments show that the full-coverage camouflage attack can not only outperform state-of-the-art methods under various test cases but also generalize to different environments, vehicles, and object detectors. The code of FCA will be available at: https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.

摘要: 物理对抗性攻击在目标检测中受到越来越多的关注。然而，以往的工作大多集中在通过生成一个单独的对抗性面片来隐藏检测器，该面片只覆盖了车辆表面的平面部分，不能在多视角、远距离和部分遮挡的物理场景中攻击检测器。为了弥合数字攻击和物理攻击之间的差距，我们利用全3D车辆表面提出了一种健壮的全覆盖伪装攻击(FCA)来欺骗检测器。具体地说，我们首先尝试在整个车辆表面上渲染非平面伪装纹理。为了模拟真实世界的环境条件，我们引入了一个变换函数来将渲染的伪装车辆转换为照片逼真的场景。最后，我们设计了一个有效的损失函数来优化伪装纹理。实验表明，全覆盖伪装攻击不仅在各种测试用例下的性能优于最新的伪装攻击方法，而且可以推广到不同的环境、车辆和目标探测器。FCA的代码可在以下网址获得：https://idrl-lab.github.io/Full-coverage-camouflage-adversarial-attack/.



## **12. Adversarial Attacks on Neural Networks for Graph Data**

图数据对神经网络的敌意攻击 stat.ML

Accepted as a full paper at KDD 2018 on May 6, 2018

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/1805.07984v4)

**Authors**: Daniel Zügner, Amir Akbarnejad, Stephan Günnemann

**Abstracts**: Deep learning models for graphs have achieved strong performance for the task of node classification. Despite their proliferation, currently there is no study of their robustness to adversarial attacks. Yet, in domains where they are likely to be used, e.g. the web, adversaries are common. Can deep learning models for graphs be easily fooled? In this work, we introduce the first study of adversarial attacks on attributed graphs, specifically focusing on models exploiting ideas of graph convolutions. In addition to attacks at test time, we tackle the more challenging class of poisoning/causative attacks, which focus on the training phase of a machine learning model. We generate adversarial perturbations targeting the node's features and the graph structure, thus, taking the dependencies between instances in account. Moreover, we ensure that the perturbations remain unnoticeable by preserving important data characteristics. To cope with the underlying discrete domain we propose an efficient algorithm Nettack exploiting incremental computations. Our experimental study shows that accuracy of node classification significantly drops even when performing only few perturbations. Even more, our attacks are transferable: the learned attacks generalize to other state-of-the-art node classification models and unsupervised approaches, and likewise are successful even when only limited knowledge about the graph is given.

摘要: 图的深度学习模型在节点分类任务中取得了很好的性能。尽管它们大量繁殖，但目前还没有关于它们对敌意攻击的健壮性的研究。然而，在它们可能被使用的领域中，例如网络，对手是常见的。图的深度学习模型很容易被愚弄吗？在这项工作中，我们首先介绍了对属性图的敌意攻击的研究，特别是集中在利用图卷积的思想的模型上。除了测试时的攻击外，我们撞击还推出了更具挑战性的中毒/致因攻击类别，这些攻击侧重于机器学习模型的训练阶段。我们针对节点的特征和图结构产生对抗性扰动，从而考虑了实例之间的依赖关系。此外，我们通过保留重要的数据特征来确保扰动保持不可察觉。为了处理底层的离散域，我们提出了一种利用增量计算的高效算法Nettack。我们的实验研究表明，即使只进行很少的扰动，节点分类的准确率也会显着下降。更重要的是，我们的攻击是可移植的：学习的攻击推广到其他最先进的节点分类模型和无监督方法，即使只给出关于图的有限知识也同样成功。



## **13. PARL: Enhancing Diversity of Ensemble Networks to Resist Adversarial Attacks via Pairwise Adversarially Robust Loss Function**

PARL：通过两两对抗性鲁棒损失函数增强集成网络的多样性以抵抗对抗性攻击 cs.LG

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.04948v1)

**Authors**: Manaar Alam, Shubhajit Datta, Debdeep Mukhopadhyay, Arijit Mondal, Partha Pratim Chakrabarti

**Abstracts**: The security of Deep Learning classifiers is a critical field of study because of the existence of adversarial attacks. Such attacks usually rely on the principle of transferability, where an adversarial example crafted on a surrogate classifier tends to mislead the target classifier trained on the same dataset even if both classifiers have quite different architecture. Ensemble methods against adversarial attacks demonstrate that an adversarial example is less likely to mislead multiple classifiers in an ensemble having diverse decision boundaries. However, recent ensemble methods have either been shown to be vulnerable to stronger adversaries or shown to lack an end-to-end evaluation. This paper attempts to develop a new ensemble methodology that constructs multiple diverse classifiers using a Pairwise Adversarially Robust Loss (PARL) function during the training procedure. PARL utilizes gradients of each layer with respect to input in every classifier within the ensemble simultaneously. The proposed training procedure enables PARL to achieve higher robustness against black-box transfer attacks compared to previous ensemble methods without adversely affecting the accuracy of clean examples. We also evaluate the robustness in the presence of white-box attacks, where adversarial examples are crafted using parameters of the target classifier. We present extensive experiments using standard image classification datasets like CIFAR-10 and CIFAR-100 trained using standard ResNet20 classifier against state-of-the-art adversarial attacks to demonstrate the robustness of the proposed ensemble methodology.

摘要: 由于敌意攻击的存在，深度学习分类器的安全性一直是一个重要的研究领域。这类攻击通常依赖于可转移性原则，在代理分类器上制作的敌意示例往往会误导在同一数据集上训练的目标分类器，即使两个分类器具有完全不同的体系结构。针对对抗性攻击的集成方法表明，对抗性示例不太可能误导具有不同决策边界的集成中的多个分类器。然而，最近的集成方法要么被证明容易受到更强大的对手的攻击，要么被证明缺乏端到端的评估。本文试图开发一种新的集成方法，在训练过程中使用成对的对抗性鲁棒损失(PAL)函数来构造多个不同的分类器。PARL同时利用相对于集成内的每个分类器中的输入的每一层的梯度。与以往的集成方法相比，所提出的训练过程使得PAL能够在不影响干净示例的准确性的情况下获得更高的抗黑盒转移攻击的鲁棒性。我们还评估了在白盒攻击存在的情况下的鲁棒性，在白盒攻击中，敌意示例是使用目标分类器的参数来制作的。我们使用标准图像分类数据集(如CIFAR-10和CIFAR-100)进行了大量的实验，这些数据集使用标准的ResNet20分类器训练来抵御最先进的对手攻击，以证明所提出的集成方法的鲁棒性。



## **14. Detecting Adversaries, yet Faltering to Noise? Leveraging Conditional Variational AutoEncoders for Adversary Detection in the Presence of Noisy Images**

侦测到对手，却对噪音犹豫不决？利用条件变分自动编码器在噪声图像中进行敌意检测 cs.LG

Accepted at Adversarial Machine Learning (AdvML) workshop, AAAI 2022

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2111.15518v2)

**Authors**: Dvij Kalaria, Aritra Hazra, Partha Pratim Chakrabarti

**Abstracts**: With the rapid advancement and increased use of deep learning models in image identification, security becomes a major concern to their deployment in safety-critical systems. Since the accuracy and robustness of deep learning models are primarily attributed from the purity of the training samples, therefore the deep learning architectures are often susceptible to adversarial attacks. Adversarial attacks are often obtained by making subtle perturbations to normal images, which are mostly imperceptible to humans, but can seriously confuse the state-of-the-art machine learning models. What is so special in the slightest intelligent perturbations or noise additions over normal images that it leads to catastrophic classifications by the deep neural networks? Using statistical hypothesis testing, we find that Conditional Variational AutoEncoders (CVAE) are surprisingly good at detecting imperceptible image perturbations. In this paper, we show how CVAEs can be effectively used to detect adversarial attacks on image classification networks. We demonstrate our results over MNIST, CIFAR-10 dataset and show how our method gives comparable performance to the state-of-the-art methods in detecting adversaries while not getting confused with noisy images, where most of the existing methods falter.

摘要: 随着深度学习模型在图像识别中的快速发展和越来越多的使用，安全性成为它们在安全关键系统中部署的主要考虑因素。由于深度学习模型的准确性和鲁棒性主要取决于训练样本的纯度，因此深度学习结构往往容易受到敌意攻击。对抗性攻击通常是通过对正常图像进行微妙的扰动来获得的，这对人类来说大多是不可察觉的，但会严重混淆最先进的机器学习模型。在正常图像上，最轻微的智能扰动或噪声添加有什么特别之处，以至于导致深层神经网络进行灾难性的分类？利用统计假设检验，我们发现条件变分自动编码器(CVAE)在检测不可察觉的图像扰动方面表现出惊人的优势。在本文中，我们展示了如何有效地使用CVAE来检测对图像分类网络的敌意攻击。我们在MNIST，CIFAR-10数据集上演示了我们的结果，并展示了我们的方法如何在检测对手方面提供与最先进的方法相当的性能，同时又不会与大多数现有方法步履蹒跚的噪声图像混淆。



## **15. On the privacy-utility trade-off in differentially private hierarchical text classification**

差异隐私层次文本分类中的隐私效用权衡研究 cs.CR

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2103.02895v2)

**Authors**: Dominik Wunderlich, Daniel Bernau, Francesco Aldà, Javier Parra-Arnau, Thorsten Strufe

**Abstracts**: Hierarchical text classification consists in classifying text documents into a hierarchy of classes and sub-classes. Although artificial neural networks have proved useful to perform this task, unfortunately they can leak training data information to adversaries due to training data memorization. Using differential privacy during model training can mitigate leakage attacks against trained models, enabling the models to be shared safely at the cost of reduced model accuracy. This work investigates the privacy-utility trade-off in hierarchical text classification with differential privacy guarantees, and identifies neural network architectures that offer superior trade-offs. To this end, we use a white-box membership inference attack to empirically assess the information leakage of three widely used neural network architectures. We show that large differential privacy parameters already suffice to completely mitigate membership inference attacks, thus resulting only in a moderate decrease in model utility. More specifically, for large datasets with long texts we observed Transformer-based models to achieve an overall favorable privacy-utility trade-off, while for smaller datasets with shorter texts convolutional neural networks are preferable.

摘要: 分层文本分类在于将文本文档分类成类和子类的分层结构。虽然人工神经网络已被证明对执行这一任务很有用，但不幸的是，由于训练数据的记忆，它们可能会将训练数据信息泄露给对手。在模型训练期间使用差异隐私可以减少对训练模型的泄漏攻击，从而以降低模型精度为代价实现模型的安全共享。这项工作研究了具有不同隐私保证的分层文本分类中的隐私效用权衡，并确定了提供优越权衡的神经网络结构。为此，我们使用白盒隶属度推理攻击对三种广泛使用的神经网络结构的信息泄漏进行了实证评估。我们表明，较大的差分隐私参数已经足以完全缓解成员关系推理攻击，因此只会导致模型效用的适度降低。更具体地说，对于具有长文本的大型数据集，我们观察到基于Transformer的模型可以实现总体上有利的隐私-效用权衡，而对于具有较短文本的较小数据集，卷积神经网络是首选的。



## **16. Amicable Aid: Turning Adversarial Attack to Benefit Classification**

友好援助：变对抗性攻击为利益分类 cs.CV

16 pages (3 pages for appendix)

**SubmitDate**: 2021-12-09    [paper-pdf](http://arxiv.org/pdf/2112.04720v1)

**Authors**: Juyeop Kim, Jun-Ho Choi, Soobeom Jang, Jong-Seok Lee

**Abstracts**: While adversarial attacks on deep image classification models pose serious security concerns in practice, this paper suggests a novel paradigm where the concept of adversarial attacks can benefit classification performance, which we call amicable aid. We show that by taking the opposite search direction of perturbation, an image can be converted to another yielding higher confidence by the classification model and even a wrongly classified image can be made to be correctly classified. Furthermore, with a large amount of perturbation, an image can be made unrecognizable by human eyes, while it is correctly recognized by the model. The mechanism of the amicable aid is explained in the viewpoint of the underlying natural image manifold. We also consider universal amicable perturbations, i.e., a fixed perturbation can be applied to multiple images to improve their classification results. While it is challenging to find such perturbations, we show that making the decision boundary as perpendicular to the image manifold as possible via training with modified data is effective to obtain a model for which universal amicable perturbations are more easily found. Finally, we discuss several application scenarios where the amicable aid can be useful, including secure image communication, privacy-preserving image communication, and protection against adversarial attacks.

摘要: 虽然针对深度图像分类模型的对抗性攻击在实践中会带来严重的安全问题，但本文提出了一种新的范式，其中对抗性攻击的概念有助于提高分类性能，我们称之为友好辅助。我们表明，通过采取相反的扰动搜索方向，分类模型可以将一幅图像转换成另一幅可信度更高的图像，甚至可以使错误分类的图像被正确分类。此外，在大量扰动的情况下，可以使图像在被模型正确识别的情况下无法被人眼识别。从底层自然意象流形的角度解释了友好相助的机制。我们还考虑了普遍的友好扰动，即一个固定的扰动可以应用于多幅图像，以改善它们的分类结果。虽然很难找到这样的扰动，但我们表明，通过用修改后的数据进行训练，使决策边界尽可能垂直于图像流形，可以有效地获得一个更容易找到普遍友好扰动的模型。最后，我们讨论了友好辅助可能有用的几个应用场景，包括安全的图像通信、隐私保护的图像通信以及对敌意攻击的保护。



## **17. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

分段和完全：利用鲁棒补丁检测保护对象检测器免受敌意补丁攻击 cs.CV

Under submission

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04532v1)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detecting and removing adversarial patches. We first train a patch segmenter that outputs patch masks that provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images given the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no performance drop on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks.

摘要: 目标检测在许多安全关键系统中起着关键作用。对抗性补丁攻击很容易在物理世界中实现，对最先进的对象检测器构成严重威胁。开发可靠的物体探测器防御补丁攻击是至关重要的，但研究严重不足。本文提出了分段完全防御(SAC)，这是一种通过检测和删除敌意补丁来防御对象检测器免受补丁攻击的通用框架。我们首先训练一个补丁分割器，该补丁分割器输出提供对抗性补丁像素级定位的补丁掩码。然后，我们提出了一种自对抗训练算法来增强补丁分割器的鲁棒性。此外，我们还设计了一种鲁棒的形状补全算法，该算法可以保证在补丁分割器的输出与地面真实的补丁掩模保持一定的汉明距离的情况下，将整个补丁从图像中去除。我们在CoCo和xView数据集上的实验表明，SAC算法即使在强自适应攻击下也能获得优异的鲁棒性，在干净的图像上不会有性能下降，并且对看不见的补丁形状、攻击预算和看不见的攻击方法都具有很好的通用性。此外，我们还给出了杏树掩码数据集，它用对抗性补丁的像素级标注来扩充杏树数据集。结果表明，SAC可以显著降低物理补丁攻击的定向攻击成功率。



## **18. On anti-stochastic properties of unlabeled graphs**

关于无标号图的反随机性 cs.DM

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04395v1)

**Authors**: Sergei Kiselev, Andrey Kupavskii, Oleg Verbitsky, Maksim Zhukovskii

**Abstracts**: We study vulnerability of a uniformly distributed random graph to an attack by an adversary who aims for a global change of the distribution while being able to make only a local change in the graph. We call a graph property $A$ anti-stochastic if the probability that a random graph $G$ satisfies $A$ is small but, with high probability, there is a small perturbation transforming $G$ into a graph satisfying $A$. While for labeled graphs such properties are easy to obtain from binary covering codes, the existence of anti-stochastic properties for unlabeled graphs is not so evident. If an admissible perturbation is either the addition or the deletion of one edge, we exhibit an anti-stochastic property that is satisfied by a random unlabeled graph of order $n$ with probability $(2+o(1))/n^2$, which is as small as possible. We also express another anti-stochastic property in terms of the degree sequence of a graph. This property has probability $(2+o(1))/(n\ln n)$, which is optimal up to factor of 2.

摘要: 我们研究了均匀分布随机图在遭受敌手攻击时的脆弱性，该敌手的目标是改变分布的全局，但只能对图进行局部改变。如果一个随机图$G$满足$A$的概率很小，但在很高的概率下，存在一个将$G$转换成满足$A$的图的小扰动，我们称图性质$A$是反随机的。而对于有标号的图，这样的性质很容易从二元覆盖码中获得，而无标号图的反随机性的存在就不那么明显了。如果一个允许的扰动是增加或删除一条边，我们表现出一个反随机性质，它由一个概率为$(2+o(1))/n^2$的n阶随机无标号图所满足，它是尽可能小的。我们还用图的度序列来表示另一个反随机性质。该属性的概率为$(2+o(1))/(n\ln)$，最优为因子2。



## **19. SNEAK: Synonymous Sentences-Aware Adversarial Attack on Natural Language Video Localization**

Screak：自然语言视频本地化的同义句感知对抗性攻击 cs.CV

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2112.04154v1)

**Authors**: Wenbo Gou, Wen Shi, Jian Lou, Lijie Huang, Pan Zhou, Ruixuan Li

**Abstracts**: Natural language video localization (NLVL) is an important task in the vision-language understanding area, which calls for an in-depth understanding of not only computer vision and natural language side alone, but more importantly the interplay between both sides. Adversarial vulnerability has been well-recognized as a critical security issue of deep neural network models, which requires prudent investigation. Despite its extensive yet separated studies in video and language tasks, current understanding of the adversarial robustness in vision-language joint tasks like NLVL is less developed. This paper therefore aims to comprehensively investigate the adversarial robustness of NLVL models by examining three facets of vulnerabilities from both attack and defense aspects. To achieve the attack goal, we propose a new adversarial attack paradigm called synonymous sentences-aware adversarial attack on NLVL (SNEAK), which captures the cross-modality interplay between the vision and language sides.

摘要: 自然语言视频定位(NLVL)是视觉-语言理解领域的一项重要任务，不仅需要深入理解计算机视觉和自然语言两个方面，更重要的是要深入理解两者之间的相互作用。对抗性漏洞已被公认为深度神经网络模型中的一个关键安全问题，需要进行仔细的研究。尽管它对视频和语言任务进行了广泛而独立的研究，但目前对NLVL等视觉-语言联合任务中的对抗性健壮性的理解还不够深入。因此，本文旨在通过从攻击和防御两个方面检查漏洞的三个方面来全面研究NLVL模型的对抗健壮性。为了达到攻击目标，我们提出了一种新的对抗性攻击范式，称为NLVL同义句感知对抗性攻击(SINVAK)，它捕捉了视觉和语言双方之间的跨通道交互作用。



## **20. Adversarial Prefetch: New Cross-Core Cache Side Channel Attacks**

对抗性预取：新的跨核心缓存侧通道攻击 cs.CR

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2110.12340v2)

**Authors**: Yanan Guo, Andrew Zigerelli, Youtao Zhang, Jun Yang

**Abstracts**: Modern x86 processors have many prefetch instructions that can be used by programmers to boost performance. However, these instructions may also cause security problems. In particular, we found that on Intel processors, there are two security flaws in the implementation of PREFETCHW, an instruction for accelerating future writes. First, this instruction can execute on data with read-only permission. Second, the execution time of this instruction leaks the current coherence state of the target data.   Based on these two design issues, we build two cross-core private cache attacks that work with both inclusive and non-inclusive LLCs, named Prefetch+Reload and Prefetch+Prefetch. We demonstrate the significance of our attacks in different scenarios. First, in the covert channel case, Prefetch+Reload and Prefetch+Prefetch achieve 782 KB/s and 822 KB/s channel capacities, when using only one shared cache line between the sender and receiver, the largest-to-date single-line capacities for CPU cache covert channels. Further, in the side channel case, our attacks can monitor the access pattern of the victim on the same processor, with almost zero error rate. We show that they can be used to leak private information of real-world applications such as cryptographic keys. Finally, our attacks can be used in transient execution attacks in order to leak more secrets within the transient window than prior work. From the experimental results, our attacks allow leaking about 2 times as many secret bytes, compared to Flush+Reload, which is widely used in transient execution attacks.

摘要: 现代x86处理器有许多预取指令，程序员可以使用这些指令来提高性能。但是，这些说明也可能导致安全问题。特别是，我们发现在Intel处理器上，PREFETCHW的实现存在两个安全缺陷，PREFETCHW是一条用于加速未来写入的指令。首先，此指令可以在具有只读权限的数据上执行。其次，此指令的执行时间会泄漏目标数据的当前一致性状态。基于这两个设计问题，我们构建了两种同时适用于包含性和非包含性LLC的跨核私有缓存攻击，分别称为预取+重新加载和预取+预取。我们在不同的情况下展示了我们的攻击的重要性。首先，在隐蔽通道的情况下，当在发送器和接收器之间仅使用一个共享高速缓存线时，预取+重新加载和预取+预取分别达到782KB/s和822KB/s的通道容量，这是迄今为止CPU高速缓存隐蔽通道的最大单线容量。此外，在旁信道情况下，我们的攻击可以监视受害者在同一处理器上的访问模式，误码率几乎为零。我们证明了它们可以被用来泄露真实世界应用程序的私有信息，例如密钥。最后，我们的攻击可以用于瞬态执行攻击，以便在瞬态窗口内泄露比以往工作更多的秘密。从实验结果看，与瞬时执行攻击中广泛使用的刷新+重新加载相比，我们的攻击允许泄漏大约2倍的秘密字节。



## **21. Two Coupled Rejection Metrics Can Tell Adversarial Examples Apart**

两个耦合的拒绝度量可以区分敌意的例子 cs.LG

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2105.14785v3)

**Authors**: Tianyu Pang, Huishuai Zhang, Di He, Yinpeng Dong, Hang Su, Wei Chen, Jun Zhu, Tie-Yan Liu

**Abstracts**: Correctly classifying adversarial examples is an essential but challenging requirement for safely deploying machine learning models. As reported in RobustBench, even the state-of-the-art adversarially trained models struggle to exceed 67% robust test accuracy on CIFAR-10, which is far from practical. A complementary way towards robustness is to introduce a rejection option, allowing the model to not return predictions on uncertain inputs, where confidence is a commonly used certainty proxy. Along with this routine, we find that confidence and a rectified confidence (R-Con) can form two coupled rejection metrics, which could provably distinguish wrongly classified inputs from correctly classified ones. This intriguing property sheds light on using coupling strategies to better detect and reject adversarial examples. We evaluate our rectified rejection (RR) module on CIFAR-10, CIFAR-10-C, and CIFAR-100 under several attacks including adaptive ones, and demonstrate that the RR module is compatible with different adversarial training frameworks on improving robustness, with little extra computation. The code is available at https://github.com/P2333/Rectified-Rejection.

摘要: 正确分类敌意示例是安全部署机器学习模型的基本要求，但也是具有挑战性的要求。正如RobustBench报道的那样，即使是经过对抗性训练的最先进的模型也难以在CIFAR-10上超过67%的稳健测试准确率，这是远远不现实的。稳健性的一种补充方式是引入拒绝选项，允许模型不返回对不确定输入的预测，其中置信度是常用的确定性代理。伴随着这个例程，我们发现置信度和校正置信度(R-CON)可以形成两个耦合的拒绝度量，它们可以很好地区分错误分类的输入和正确分类的输入。这一耐人寻味的性质有助于使用耦合策略更好地检测和拒绝敌意示例。我们在CIFAR-10、CIFAR-10-C和CIFAR-100上测试了我们的纠偏拒绝(RR)模块在包括自适应攻击在内的几种攻击下的性能，并证明了RR模块在提高鲁棒性方面与不同的对手训练框架兼容，并且几乎不需要额外的计算量。代码可在https://github.com/P2333/Rectified-Rejection.上获得



## **22. Local Convolutions Cause an Implicit Bias towards High Frequency Adversarial Examples**

局部卷积导致对高频对抗性例子的隐性偏向 stat.ML

20 pages, 11 figures, 12 Tables

**SubmitDate**: 2021-12-08    [paper-pdf](http://arxiv.org/pdf/2006.11440v4)

**Authors**: Josue Ortega Caro, Yilong Ju, Ryan Pyle, Sourav Dey, Wieland Brendel, Fabio Anselmi, Ankit Patel

**Abstracts**: Adversarial Attacks are still a significant challenge for neural networks. Recent work has shown that adversarial perturbations typically contain high-frequency features, but the root cause of this phenomenon remains unknown. Inspired by theoretical work on linear full-width convolutional models, we hypothesize that the local (i.e. bounded-width) convolutional operations commonly used in current neural networks are implicitly biased to learn high frequency features, and that this is one of the root causes of high frequency adversarial examples. To test this hypothesis, we analyzed the impact of different choices of linear and nonlinear architectures on the implicit bias of the learned features and the adversarial perturbations, in both spatial and frequency domains. We find that the high-frequency adversarial perturbations are critically dependent on the convolution operation because the spatially-limited nature of local convolutions induces an implicit bias towards high frequency features. The explanation for the latter involves the Fourier Uncertainty Principle: a spatially-limited (local in the space domain) filter cannot also be frequency-limited (local in the frequency domain). Furthermore, using larger convolution kernel sizes or avoiding convolutions (e.g. by using Vision Transformers architecture) significantly reduces this high frequency bias, but not the overall susceptibility to attacks. Looking forward, our work strongly suggests that understanding and controlling the implicit bias of architectures will be essential for achieving adversarial robustness.

摘要: 对抗性攻击仍然是神经网络面临的重大挑战。最近的研究表明，对抗性扰动通常包含高频特征，但这种现象的根本原因尚不清楚。受线性全宽度卷积模型理论工作的启发，我们假设当前神经网络中常用的局部(即有界宽度)卷积运算隐含地偏向于学习高频特征，这是造成高频对抗性例子的根本原因之一。为了验证这一假设，我们在空间域和频域分析了线性和非线性结构的不同选择对学习特征的内隐偏差和对抗性扰动的影响。我们发现，高频对抗性扰动严重依赖于卷积运算，因为局部卷积的空间有限性质导致了对高频特征的隐式偏差。对后者的解释涉及到傅立叶测不准原理：空间受限(空间域中的局部)过滤不能也是频率受限的(频域中的局部)。此外，使用更大的卷积核大小或避免卷积(例如，通过使用Vision Transformers架构)可以显著降低这种高频偏差，但不会显著降低对攻击的总体易感性。展望未来，我们的工作强烈表明，理解和控制体系结构的隐含偏差将是实现对抗性健壮性的关键。



## **23. SoK: Certified Robustness for Deep Neural Networks**

SOK：深度神经网络的认证鲁棒性 cs.LG

14 pages for the main text

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2009.04131v4)

**Authors**: Linyi Li, Xiangyu Qi, Tao Xie, Bo Li

**Abstracts**: Great advances in deep neural networks (DNNs) have led to state-of-the-art performance on a wide range of tasks. However, recent studies have shown that DNNs are vulnerable to adversarial attacks, which have brought great concerns when deploying these models to safety-critical applications such as autonomous driving. Different defense approaches have been proposed against adversarial attacks, including: a) empirical defenses, which usually can be adaptively attacked again without providing robustness certification; and b) certifiably robust approaches which consist of robustness verification providing the lower bound of robust accuracy against any attacks under certain conditions and corresponding robust training approaches. In this paper, we systematize the certifiably robust approaches and related practical and theoretical implications and findings. We also provide the first comprehensive benchmark on existing robustness verification and training approaches on different datasets. In particular, we 1) provide a taxonomy for the robustness verification and training approaches, as well as summarize the methodologies for representative algorithms, 2) reveal the characteristics, strengths, limitations, and fundamental connections among these approaches, 3) discuss current research progresses, theoretical barriers, main challenges, and future directions for certifiably robust approaches for DNNs, and 4) provide an open-sourced unified platform to evaluate over 20 representative certifiably robust approaches for a wide range of DNNs.

摘要: 深度神经网络(DNNs)的巨大进步导致了在广泛任务上的最先进的性能。然而，最近的研究表明，DNN很容易受到敌意攻击，这在将这些模型部署到自动驾驶等安全关键型应用时带来了极大的担忧。针对敌意攻击已经提出了不同的防御方法，包括：a)经验防御，通常无需提供健壮性证明即可自适应地再次攻击；b)可证明健壮性方法，包括在一定条件下提供对任何攻击的鲁棒精度下界的健壮性验证和相应的健壮性训练方法。在这篇文章中，我们系统化的证明稳健的方法和相关的实际和理论意义和发现。我们还提供了关于不同数据集上现有健壮性验证和训练方法的第一个全面基准。特别地，我们1)提供了健壮性验证和训练方法的分类，并总结了典型算法的方法论；2)揭示了这些方法的特点、优点、局限性和基本联系；3)讨论了当前DNNs的研究进展、理论障碍、主要挑战和未来的发展方向；4)提供了一个开源的统一平台来评估20多种具有代表性的DNNs的可证健壮性方法。



## **24. Saliency Diversified Deep Ensemble for Robustness to Adversaries**

显著多样化的深度集成，增强了对对手的健壮性 cs.CV

Accepted to AAAI Workshop on Adversarial Machine Learning and Beyond  2022

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03615v1)

**Authors**: Alex Bogun, Dimche Kostadinov, Damian Borth

**Abstracts**: Deep learning models have shown incredible performance on numerous image recognition, classification, and reconstruction tasks. Although very appealing and valuable due to their predictive capabilities, one common threat remains challenging to resolve. A specifically trained attacker can introduce malicious input perturbations to fool the network, thus causing potentially harmful mispredictions. Moreover, these attacks can succeed when the adversary has full access to the target model (white-box) and even when such access is limited (black-box setting). The ensemble of models can protect against such attacks but might be brittle under shared vulnerabilities in its members (attack transferability). To that end, this work proposes a novel diversity-promoting learning approach for the deep ensembles. The idea is to promote saliency map diversity (SMD) on ensemble members to prevent the attacker from targeting all ensemble members at once by introducing an additional term in our learning objective. During training, this helps us minimize the alignment between model saliencies to reduce shared member vulnerabilities and, thus, increase ensemble robustness to adversaries. We empirically show a reduced transferability between ensemble members and improved performance compared to the state-of-the-art ensemble defense against medium and high strength white-box attacks. In addition, we demonstrate that our approach combined with existing methods outperforms state-of-the-art ensemble algorithms for defense under white-box and black-box attacks.

摘要: 深度学习模型在众多的图像识别、分类和重建任务中表现出了令人难以置信的性能。尽管它们的预测能力非常有吸引力和价值，但一个共同的威胁仍然难以解决。经过特殊训练的攻击者可以引入恶意输入扰动来欺骗网络，从而导致潜在的有害误判。此外，当对手拥有对目标模型的完全访问权限(白盒)，甚至当此类访问受限(黑盒设置)时，这些攻击也可能成功。模型集合可以防止此类攻击，但在其成员的共享漏洞(攻击可转移性)下可能会变得脆弱。为此，本工作提出了一种新的促进深度集成的多样性学习方法。我们的想法是通过在我们的学习目标中引入一个额外的术语来促进集合成员的显著地图多样性(SMD)，以防止攻击者一次针对所有集合成员。在训练过程中，这有助于我们最大限度地减少模型显著性之间的对齐，以减少共享的成员漏洞，从而提高对对手的整体健壮性。我们的经验表明，与针对中高强度白盒攻击的最先进的组合防御相比，组合成员之间的可传递性降低了，性能得到了提高。此外，我们还证明了我们的方法与现有方法相结合，在白盒和黑盒攻击下的防御性能优于最先进的集成算法。



## **25. Membership Inference Attacks From First Principles**

基于第一性原理的隶属度推理攻击 cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03570v1)

**Authors**: Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, Florian Tramer

**Abstracts**: A membership inference attack allows an adversary to query a trained machine learning model to predict whether or not a particular example was contained in the model's training dataset. These attacks are currently evaluated using average-case "accuracy" metrics that fail to characterize whether the attack can confidently identify any members of the training set. We argue that attacks should instead be evaluated by computing their true-positive rate at low (e.g., <0.1%) false-positive rates, and find most prior attacks perform poorly when evaluated in this way. To address this we develop a Likelihood Ratio Attack (LiRA) that carefully combines multiple ideas from the literature. Our attack is 10x more powerful at low false-positive rates, and also strictly dominates prior attacks on existing metrics.

摘要: 成员关系推断攻击允许对手查询经过训练的机器学习模型，以预测特定示例是否包含在该模型的训练数据集中。目前使用平均案例“准确性”度量来评估这些攻击，这些度量无法表征攻击是否可以自信地识别训练集的任何成员。我们认为，应该通过计算低(例如<0.1%)假阳性率下的真阳性率来评估攻击，并且发现大多数以前的攻击在这样评估时表现很差。为了解决这个问题，我们开发了一种似然比攻击(LIRA)，它仔细地结合了文献中的多种想法。我们的攻击以较低的误报率提高了10倍的威力，并且严格控制了先前对现有指标的攻击。



## **26. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

基于决策的基于补丁对抗性去除的视觉变形金刚黑盒攻击 cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03492v1)

**Authors**: Yucheng Shi, Yahong Han

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Deep Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the existing decision-based attacks for CNNs ignore the difference in noise sensitivity between different regions of the image, which affects the efficiency of noise compression. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we propose a new decision-based black-box attack against ViTs termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on ImageNet-21k, ILSVRC-2012, and Tiny-Imagenet datasets demonstrate that PAR achieves a much lower magnitude of perturbation on average with the same number of queries.

摘要: 与深度卷积神经网络(CNNs)相比，视觉变换器(VITS)表现出令人印象深刻的性能和更强的对抗鲁棒性。一方面，VITS对单个斑块之间全局交互的关注降低了图像的局部噪声敏感度。另一方面，现有的基于决策的CNN攻击忽略了图像不同区域之间噪声敏感性的差异，影响了噪声压缩的效率。因此，当目标模型只能查询时，验证VITS的黑箱对抗鲁棒性仍然是一个具有挑战性的问题。本文提出了一种新的基于决策的针对VITS的黑盒攻击，称为补丁对抗性删除(PAR)。PAR通过从粗到细的搜索过程将图像分成多个块，并分别压缩每个块上的噪声。PAR记录每个面片的噪声大小和噪声敏感度，并选择查询值最高的面片进行噪声压缩。此外，PAR可以作为其他基于判决的攻击的噪声初始化方法，在不引入额外计算的情况下提高VITS和CNN的噪声压缩效率。在ImageNet-21k、ILSVRC-2012和Tiny-Imagenet数据集上的大量实验表明，在相同的查询数量下，PAR的平均扰动幅度要小得多。



## **27. BDFA: A Blind Data Adversarial Bit-flip Attack on Deep Neural Networks**

BDFA：一种基于深度神经网络的盲数据对抗性比特翻转攻击 cs.CR

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2112.03477v1)

**Authors**: Behnam Ghavami, Mani Sadati, Mohammad Shahidzadeh, Zhenman Fang, Lesley Shannon

**Abstracts**: Adversarial bit-flip attack (BFA) on Neural Network weights can result in catastrophic accuracy degradation by flipping a very small number of bits. A major drawback of prior bit flip attack techniques is their reliance on test data. This is frequently not possible for applications that contain sensitive or proprietary data. In this paper, we propose Blind Data Adversarial Bit-flip Attack (BDFA), a novel technique to enable BFA without any access to the training or testing data. This is achieved by optimizing for a synthetic dataset, which is engineered to match the statistics of batch normalization across different layers of the network and the targeted label. Experimental results show that BDFA could decrease the accuracy of ResNet50 significantly from 75.96\% to 13.94\% with only 4 bits flips.

摘要: 对神经网络权重的对抗性比特翻转攻击(BFA)可以通过翻转非常少量的比特来导致灾难性的精度降低。现有比特翻转攻击技术的主要缺点是它们对测试数据的依赖。对于包含敏感或专有数据的应用程序而言，这通常是不可能的。在本文中，我们提出了盲数据对抗比特翻转攻击(BDFA)，这是一种新的技术，可以在不访问任何训练或测试数据的情况下实现BFA。这是通过对合成数据集进行优化来实现的，该合成数据集被设计为匹配跨网络的不同层和目标标签的批归一化的统计数据。实验结果表明，只需4位翻转，BDFA就能将ResNet50的精度从75.96降到13.94。



## **28. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

GasHis-Transformer：一种用于胃组织病理图像分类的多尺度视觉变换方法 cs.CV

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2104.14528v5)

**Authors**: Haoyuan Chen, Chen Li, Xiaoyan Li, Ge Wang, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Yudong Yao, Yueyang Teng, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.

摘要: 现有的胃癌诊断深度学习方法普遍采用卷积神经网络。近年来，视觉变压器因其高性能和高效率而备受关注，但其应用大多集中在计算机视觉领域。本文提出了一种用于胃组织病理图像分类(GHIC)的多尺度视觉转换器模型(简称GasHis-Transformer)，该模型能够自动将胃显微图像分类为异常和正常病例。GasHis-Transformer模型由两个关键模块组成：全局信息模块和局部信息模块，有效地提取组织病理学特征。在我们的实验中，一个公共的苏木精伊红(H&E)染色的胃组织病理学数据集以1：1：2的比例分为训练集、验证集和测试集，训练集、验证集和测试集的比例为1：1：2。应用GasHis-Transformer模型估计胃组织病理学数据集的准确率、召回率、F1得分和准确率分别为98.0%、100.0%、96.0%和98.0%。此外，还对GasHis-Transformer的稳健性进行了关键研究，添加了10种不同的噪声，包括4种对抗性攻击和6种常规图像噪声。另外，利用620幅异常图像对GasHis-Transformer的胃肠道肿瘤识别性能进行了有临床意义的测试，准确率达到96.8%。最后，在淋巴瘤图像数据集和乳腺癌数据集上对H&E和免疫组织化学染色图像的泛化能力进行了比较研究，得到了可比的F1得分(85.6%和82.8%)和准确率(83.9%和89.4%)。总之，GasHisTransformer表现出很高的分类性能，并在GHIC任务中显示出巨大的潜力。



## **29. Introducing the DOME Activation Functions**

介绍穹顶激活功能 cs.LG

16 pages, 9 figures

**SubmitDate**: 2021-12-07    [paper-pdf](http://arxiv.org/pdf/2109.14798v2)

**Authors**: Mohamed E. Hussein, Wael AbdAlmageed

**Abstracts**: In this paper, we introduce a novel non-linear activation function that spontaneously induces class-compactness and regularization in the embedding space of neural networks. The function is dubbed DOME for Difference Of Mirrored Exponential terms. The basic form of the function can replace the sigmoid or the hyperbolic tangent functions as an output activation function for binary classification problems. The function can also be extended to the case of multi-class classification, and used as an alternative to the standard softmax function. It can also be further generalized to take more flexible shapes suitable for intermediate layers of a network. We empirically demonstrate the properties of the function. We also show that models using the function exhibit extra robustness against adversarial attacks.

摘要: 本文介绍了一种新的非线性激活函数，它在神经网络的嵌入空间中自发地诱导类紧性和正则化。由于镜像指数项的差异，该函数被称为穹顶。该函数的基本形式可以代替Sigmoid或双曲正切函数作为二进制分类问题的输出激活函数。该功能还可以扩展到多类分类的情况，并用作标准Softmax功能的替代。它还可以被进一步推广，以采取适合于网络中间层的更灵活的形状。我们实证地证明了该函数的性质。我们还表明，使用该函数的模型在抵抗敌意攻击时表现出额外的鲁棒性。



## **30. Adversarial Attacks in Cooperative AI**

协作式人工智能中的对抗性攻击 cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.14833v2)

**Authors**: Ted Fujimoto, Arthur Paul Pedersen

**Abstracts**: Single-agent reinforcement learning algorithms in a multi-agent environment are inadequate for fostering cooperation. If intelligent agents are to interact and work together to solve complex problems, methods that counter non-cooperative behavior are needed to facilitate the training of multiple agents. This is the goal of cooperative AI. Recent work in adversarial machine learning, however, shows that models (e.g., image classifiers) can be easily deceived into making incorrect decisions. In addition, some past research in cooperative AI has relied on new notions of representations, like public beliefs, to accelerate the learning of optimally cooperative behavior. Hence, cooperative AI might introduce new weaknesses not investigated in previous machine learning research. In this paper, our contributions include: (1) arguing that three algorithms inspired by human-like social intelligence introduce new vulnerabilities, unique to cooperative AI, that adversaries can exploit, and (2) an experiment showing that simple, adversarial perturbations on the agents' beliefs can negatively impact performance. This evidence points to the possibility that formal representations of social behavior are vulnerable to adversarial attacks.

摘要: 多智能体环境中的单智能体强化学习算法不能很好地促进协作。如果智能Agent要交互并共同工作来解决复杂问题，就需要针对不合作行为的方法，以便于多个Agent的训练。这是合作AI的目标。然而，最近在对抗性机器学习方面的工作表明，模型(例如，图像分类器)很容易被欺骗，从而做出不正确的决定。此外，过去对合作人工智能的一些研究依赖于新的表征概念，如公众信仰，以加速最佳合作行为的学习。因此，合作人工智能可能会引入以前的机器学习研究中没有研究的新弱点。在本文中，我们的贡献包括：(1)论证了三种受类人类社会智能启发的算法引入了新的漏洞，这些漏洞是合作人工智能所特有的，攻击者可以利用这些漏洞；(2)实验表明，对Agent信念的简单对抗性扰动可能会对性能产生负面影响。这一证据表明，社交行为的正式表述很容易受到敌意攻击。



## **31. Shape Defense Against Adversarial Attacks**

塑造对敌方攻击的防御 cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2008.13336v3)

**Authors**: Ali Borji

**Abstracts**: Humans rely heavily on shape information to recognize objects. Conversely, convolutional neural networks (CNNs) are biased more towards texture. This is perhaps the main reason why CNNs are vulnerable to adversarial examples. Here, we explore how shape bias can be incorporated into CNNs to improve their robustness. Two algorithms are proposed, based on the observation that edges are invariant to moderate imperceptible perturbations. In the first one, a classifier is adversarially trained on images with the edge map as an additional channel. At inference time, the edge map is recomputed and concatenated to the image. In the second algorithm, a conditional GAN is trained to translate the edge maps, from clean and/or perturbed images, into clean images. Inference is done over the generated image corresponding to the input's edge map. Extensive experiments over 10 datasets demonstrate the effectiveness of the proposed algorithms against FGSM and $\ell_\infty$ PGD-40 attacks. Further, we show that a) edge information can also benefit other adversarial training methods, and b) CNNs trained on edge-augmented inputs are more robust against natural image corruptions such as motion blur, impulse noise and JPEG compression, than CNNs trained solely on RGB images. From a broader perspective, our study suggests that CNNs do not adequately account for image structures that are crucial for robustness. Code is available at:~\url{https://github.com/aliborji/Shapedefense.git}.

摘要: 人类在很大程度上依赖于形状信息来识别物体。相反，卷积神经网络(CNN)更偏向于纹理。这可能是CNN容易受到敌意例子攻击的主要原因。在这里，我们将探索如何将形状偏差融入到CNN中，以提高其鲁棒性。基于边缘不变到适度的不可察觉扰动这一观察结果，提出了两种算法。在第一种方法中，分类器以边缘图作为附加通道对图像进行对抗性训练。在推断时，重新计算边缘映射并将其连接到图像。在第二种算法中，训练条件GAN以将边缘图从干净和/或扰动的图像转换成干净的图像。对与输入的边缘映射相对应的生成图像进行推断。在10个数据集上的大量实验证明了所提算法对FGSM和$\ELL_\INFTY$PGD-40攻击的有效性。此外，我们还表明：a)边缘信息也可以用于其他对抗性训练方法；b)与仅训练在RGB图像上的CNN相比，基于边缘增强输入训练的CNN对运动模糊、脉冲噪声和JPEG压缩等自然图像的破坏具有更强的鲁棒性。从更广泛的角度来看，我们的研究表明，CNN没有充分考虑对鲁棒性至关重要的图像结构。代码可用at：~\url{https://github.com/aliborji/Shapedefense.git}.



## **32. Adversarial Machine Learning In Network Intrusion Detection Domain: A Systematic Review**

网络入侵检测领域的对抗性机器学习研究综述 cs.CR

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03315v1)

**Authors**: Huda Ali Alatwi, Charles Morisset

**Abstracts**: Due to their massive success in various domains, deep learning techniques are increasingly used to design network intrusion detection solutions that detect and mitigate unknown and known attacks with high accuracy detection rates and minimal feature engineering. However, it has been found that deep learning models are vulnerable to data instances that can mislead the model to make incorrect classification decisions so-called (adversarial examples). Such vulnerability allows attackers to target NIDSs by adding small crafty perturbations to the malicious traffic to evade detection and disrupt the system's critical functionalities. The problem of deep adversarial learning has been extensively studied in the computer vision domain; however, it is still an area of open research in network security applications. Therefore, this survey explores the researches that employ different aspects of adversarial machine learning in the area of network intrusion detection in order to provide directions for potential solutions. First, the surveyed studies are categorized based on their contribution to generating adversarial examples, evaluating the robustness of ML-based NIDs towards adversarial examples, and defending these models against such attacks. Second, we highlight the characteristics identified in the surveyed research. Furthermore, we discuss the applicability of the existing generic adversarial attacks for the NIDS domain, the feasibility of launching the proposed attacks in real-world scenarios, and the limitations of the existing mitigation solutions.

摘要: 由于深度学习技术在各个领域取得了巨大的成功，越来越多的人将深度学习技术用于设计网络入侵检测解决方案，这些解决方案能够以较高的准确率和最小的特征工程来检测和缓解未知和已知的攻击。然而，人们发现深度学习模型容易受到数据实例的影响，这些数据实例可能会误导模型做出不正确的分类决策，即所谓的(对抗性示例)。这样的漏洞使得攻击者能够通过向恶意通信量添加小而狡猾的干扰来瞄准NIDS，以逃避检测并破坏系统的关键功能。深度对抗性学习问题在计算机视觉领域得到了广泛的研究，但在网络安全应用中仍是一个开放的研究领域。因此，本综述探讨了在网络入侵检测领域应用对抗性机器学习的不同方面的研究，以期为潜在的解决方案提供方向。首先，调查的研究根据它们在生成对抗性实例、评估基于ML的NID对对抗性实例的健壮性以及保护这些模型免受此类攻击方面的贡献进行分类。其次，我们突出了调查研究中确定的特点。此外，我们还讨论了现有针对NIDS域的通用对抗性攻击的适用性、在现实场景中发起攻击的可行性以及现有缓解方案的局限性。



## **33. Context-Aware Transfer Attacks for Object Detection**

面向对象检测的上下文感知传输攻击 cs.CV

accepted to AAAI 2022

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.03223v1)

**Authors**: Zikui Cai, Xinxin Xie, Shasha Li, Mingjun Yin, Chengyu Song, Srikanth V. Krishnamurthy, Amit K. Roy-Chowdhury, M. Salman Asif

**Abstracts**: Blackbox transfer attacks for image classifiers have been extensively studied in recent years. In contrast, little progress has been made on transfer attacks for object detectors. Object detectors take a holistic view of the image and the detection of one object (or lack thereof) often depends on other objects in the scene. This makes such detectors inherently context-aware and adversarial attacks in this space are more challenging than those targeting image classifiers. In this paper, we present a new approach to generate context-aware attacks for object detectors. We show that by using co-occurrence of objects and their relative locations and sizes as context information, we can successfully generate targeted mis-categorization attacks that achieve higher transfer success rates on blackbox object detectors than the state-of-the-art. We test our approach on a variety of object detectors with images from PASCAL VOC and MS COCO datasets and demonstrate up to $20$ percentage points improvement in performance compared to the other state-of-the-art methods.

摘要: 近年来，针对图像分类器的黑盒传输攻击得到了广泛的研究。相比之下，针对目标检测器的传输攻击研究进展甚微。对象检测器对图像进行整体观察，并且一个对象(或其缺失)的检测通常取决于场景中的其他对象。这使得这类检测器固有的上下文感知和敌意攻击比那些针对图像分类器的攻击更具挑战性。本文提出了一种生成对象检测器上下文感知攻击的新方法。通过使用对象的共现及其相对位置和大小作为上下文信息，我们可以成功地生成具有针对性的误分类攻击，从而在黑盒对象检测器上获得比现有技术更高的传输成功率。我们使用Pascal VOC和MS Coco数据集的图像在各种对象探测器上测试了我们的方法，与其他最先进的方法相比，性能提高了多达20个百分点。



## **34. Improving the Adversarial Robustness for Speaker Verification by Self-Supervised Learning**

利用自监督学习提高说话人确认的对抗性 cs.SD

Accepted by TASLP

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2106.00273v3)

**Authors**: Haibin Wu, Xu Li, Andy T. Liu, Zhiyong Wu, Helen Meng, Hung-yi Lee

**Abstracts**: Previous works have shown that automatic speaker verification (ASV) is seriously vulnerable to malicious spoofing attacks, such as replay, synthetic speech, and recently emerged adversarial attacks. Great efforts have been dedicated to defending ASV against replay and synthetic speech; however, only a few approaches have been explored to deal with adversarial attacks. All the existing approaches to tackle adversarial attacks for ASV require the knowledge for adversarial samples generation, but it is impractical for defenders to know the exact attack algorithms that are applied by the in-the-wild attackers. This work is among the first to perform adversarial defense for ASV without knowing the specific attack algorithms. Inspired by self-supervised learning models (SSLMs) that possess the merits of alleviating the superficial noise in the inputs and reconstructing clean samples from the interrupted ones, this work regards adversarial perturbations as one kind of noise and conducts adversarial defense for ASV by SSLMs. Specifically, we propose to perform adversarial defense from two perspectives: 1) adversarial perturbation purification and 2) adversarial perturbation detection. Experimental results show that our detection module effectively shields the ASV by detecting adversarial samples with an accuracy of around 80%. Moreover, since there is no common metric for evaluating the adversarial defense performance for ASV, this work also formalizes evaluation metrics for adversarial defense considering both purification and detection based approaches into account. We sincerely encourage future works to benchmark their approaches based on the proposed evaluation framework.

摘要: 以往的研究表明，自动说话人验证(ASV)很容易受到恶意欺骗攻击，如重放、合成语音以及最近出现的敌意攻击。人们一直致力于保护ASV免受重播和合成语音的攻击，然而，只探索了几种方法来应对对抗性攻击。现有的撞击对抗性攻击方法都需要生成对抗性样本的知识，但是防御者要知道野外攻击者使用的确切攻击算法是不切实际的。这项工作是第一批在不知道具体攻击算法的情况下对ASV进行对抗性防御的工作之一。受自监督学习模型(SSLMs)减少输入表面噪声和从中断样本中重构干净样本等优点的启发，本文将对抗性扰动视为一种噪声，利用SSLMs对ASV进行对抗性防御。具体地说，我们提出从两个角度进行对抗性防御：1)对抗性扰动净化和2)对抗性扰动检测。实验结果表明，我们的检测模块通过检测敌意样本，有效地屏蔽了ASV，准确率在80%左右。此外，由于ASV的对抗防御性能没有统一的评价指标，本文还考虑了基于净化和基于检测的方法，形式化了对抗防御的评价指标。我们真诚地鼓励今后的工作在拟议的评价框架基础上对其方法进行基准。



## **35. Adversarial Example Detection for DNN Models: A Review and Experimental Comparison**

DNN模型的对抗性范例检测：综述与实验比较 cs.CV

To be published on Artificial Intelligence Review journal (after  minor revision)

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2105.00203v3)

**Authors**: Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Deforges

**Abstracts**: Deep learning (DL) has shown great success in many human-related tasks, which has led to its adoption in many computer vision based applications, such as security surveillance systems, autonomous vehicles and healthcare. Such safety-critical applications have to draw their path to success deployment once they have the capability to overcome safety-critical challenges. Among these challenges are the defense against or/and the detection of the adversarial examples (AEs). Adversaries can carefully craft small, often imperceptible, noise called perturbations to be added to the clean image to generate the AE. The aim of AE is to fool the DL model which makes it a potential risk for DL applications. Many test-time evasion attacks and countermeasures,i.e., defense or detection methods, are proposed in the literature. Moreover, few reviews and surveys were published and theoretically showed the taxonomy of the threats and the countermeasure methods with little focus in AE detection methods. In this paper, we focus on image classification task and attempt to provide a survey for detection methods of test-time evasion attacks on neural network classifiers. A detailed discussion for such methods is provided with experimental results for eight state-of-the-art detectors under different scenarios on four datasets. We also provide potential challenges and future perspectives for this research direction.

摘要: 深度学习(DL)在许多与人类相关的任务中取得了巨大的成功，这使得它被许多基于计算机视觉的应用所采用，如安全监控系统、自动驾驶汽车和医疗保健。此类安全关键型应用程序一旦具备了克服安全关键型挑战的能力，就必须为成功部署画上句号。在这些挑战中，包括防御或/和检测对抗性示例(AEs)。攻击者可以小心翼翼地制造称为扰动的小噪音，通常是难以察觉的，并将其添加到干净的图像中，以生成AE。AE的目的是愚弄DL模型，使其成为DL应用程序的潜在风险。文献中提出了许多测试时间逃避攻击和对策，即防御或检测方法。此外，很少有综述和调查发表，从理论上给出了威胁的分类和对策方法，而对声发射检测方法的关注较少。本文以图像分类任务为研究对象，对神经网络分类器测试时间逃避攻击的检测方法进行了综述。对这些方法进行了详细的讨论，并给出了在四个数据集上的不同场景下八个最先进检测器的实验结果。我们还对这一研究方向提出了潜在的挑战和未来的展望。



## **36. Robust Person Re-identification with Multi-Modal Joint Defence**

基于多模态联合防御的鲁棒人物再识别 cs.CV

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2111.09571v2)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.

摘要: 基于度量学习的人物识别(ReID)系统继承了深层神经网络(DNNs)易被恶意度量攻击欺骗的弱点。现有的工作主要依靠对抗性训练进行度量防御，更多的方法还没有得到充分的研究。通过研究攻击对底层特征的影响，提出了有针对性的度量攻击方法和防御方法。在度量攻击方面，我们利用局部颜色偏差来构造输入的类内变异来攻击颜色特征。在度量防御方面，我们提出了一种包括主动防御和被动防御两部分的联合防御方法。主动防御通过从多模态图像构造不同的输入来增强模型对颜色变化的鲁棒性和跨多模态的结构关系的学习，而被动防御通过迂回缩放利用结构特征在变化的像素空间中的不变性来保留结构特征，同时消除一些对抗性噪声。大量实验表明，与现有的对抗性度量防御方法相比，本文提出的联合防御方法不仅可以同时防御多个攻击，而且没有显着降低模型的泛化能力。代码可在https://github.com/finger-monkey/multi-modal_joint_defence.上获得



## **37. ML Attack Models: Adversarial Attacks and Data Poisoning Attacks**

ML攻击模型：对抗性攻击和数据中毒攻击 cs.LG

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02797v1)

**Authors**: Jing Lin, Long Dang, Mohamed Rahouti, Kaiqi Xiong

**Abstracts**: Many state-of-the-art ML models have outperformed humans in various tasks such as image classification. With such outstanding performance, ML models are widely used today. However, the existence of adversarial attacks and data poisoning attacks really questions the robustness of ML models. For instance, Engstrom et al. demonstrated that state-of-the-art image classifiers could be easily fooled by a small rotation on an arbitrary image. As ML systems are being increasingly integrated into safety and security-sensitive applications, adversarial attacks and data poisoning attacks pose a considerable threat. This chapter focuses on the two broad and important areas of ML security: adversarial attacks and data poisoning attacks.

摘要: 许多最先进的ML模型在图像分类等各种任务中的表现都超过了人类。ML模型以其出色的性能在今天得到了广泛的应用。然而，敌意攻击和数据中毒攻击的存在确实对ML模型的稳健性提出了质疑。例如，Engstrom等人。展示了最先进的图像分类器可以很容易地被任意图像上的小旋转所愚弄。随着ML系统越来越多地集成到安全和安全敏感的应用程序中，对抗性攻击和数据中毒攻击构成了相当大的威胁。本章重点介绍ML安全的两个广泛而重要的领域：对抗性攻击和数据中毒攻击。



## **38. An Improved Genetic Algorithm and Its Application in Neural Network Adversarial Attack**

一种改进的遗传算法及其在神经网络攻击中的应用 cs.NE

14 pages, 7 figures, 4 tables and 20 References

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2110.01818v4)

**Authors**: Dingming Yang, Zeyu Yu, Hongqiang Yuan, Yanrong Cui

**Abstracts**: The choice of crossover and mutation strategies plays a crucial role in the search ability, convergence efficiency and precision of genetic algorithms. In this paper, a novel improved genetic algorithm is proposed by improving the crossover and mutation operation of the simple genetic algorithm, and it is verified by four test functions. Simulation results show that, comparing with three other mainstream swarm intelligence optimization algorithms, the algorithm can not only improve the global search ability, convergence efficiency and precision, but also increase the success rate of convergence to the optimal value under the same experimental conditions. Finally, the algorithm is applied to neural networks adversarial attacks. The applied results show that the method does not need the structure and parameter information inside the neural network model, and it can obtain the adversarial samples with high confidence in a brief time just by the classification and confidence information output from the neural network.

摘要: 交叉和变异策略的选择对遗传算法的搜索能力、收敛效率和精度起着至关重要的作用。通过对简单遗传算法交叉和变异操作的改进，提出了一种新的改进遗传算法，并通过四个测试函数进行了验证。仿真结果表明，与其他三种主流群体智能优化算法相比，该算法在相同的实验条件下，不仅提高了全局搜索能力、收敛效率和精度，而且提高了收敛到最优值的成功率。最后，将该算法应用于神经网络的对抗性攻击。应用结果表明，该方法不需要神经网络模型内部的结构和参数信息，仅根据神经网络输出的分类和置信度信息，即可在短时间内获得高置信度的对抗性样本。



## **39. Staring Down the Digital Fulda Gap Path Dependency as a Cyber Defense Vulnerability**

向下看数字富尔达缺口路径依赖是一个网络防御漏洞 cs.CY

**SubmitDate**: 2021-12-06    [paper-pdf](http://arxiv.org/pdf/2112.02773v1)

**Authors**: Jan Kallberg

**Abstracts**: Academia, homeland security, defense, and media have accepted the perception that critical infrastructure in a future cyber war cyber conflict is the main gateway for a massive cyber assault on the U.S. The question is not if the assumption is correct or not, the question is instead of how did we arrive at that assumption. The cyber paradigm considers critical infrastructure the primary attack vector for future cyber conflicts. The national vulnerability embedded in critical infrastructure is given a position in the cyber discourse as close to an unquestionable truth as a natural law.   The American reaction to Sept. 11, and any attack on U.S. soil, hint to an adversary that attacking critical infrastructure to create hardship for the population could work contrary to the intended softening of the will to resist foreign influence. It is more likely that attacks that affect the general population instead strengthen the will to resist and fight, similar to the British reaction to the German bombing campaign Blitzen in 1940. We cannot rule out attacks that affect the general population, but there are not enough adversarial offensive capabilities to attack all 16 critical infrastructure sectors and gain strategic momentum. An adversary has limited cyberattack capabilities and needs to prioritize cyber targets that are aligned with the overall strategy. Logically, an adversary will focus their OCO on operations that has national security implications and support their military operations by denying, degrading, and confusing the U.S. information environment and U.S. cyber assets.

摘要: 学术界、国土安全、国防和媒体已经接受了这样的看法，即未来网络战中的关键基础设施网络冲突是针对美国的大规模网络攻击的主要门户。问题不是假设是否正确，而是我们如何得出这个假设。网络范式认为关键基础设施是未来网络冲突的主要攻击载体。关键基础设施中嵌入的国家脆弱性在网络话语中被赋予了与自然法一样接近毋庸置疑的真理的地位。美国对9·11事件的反应。11，以及对美国领土的任何袭击，都暗示着对手，攻击关键基础设施给人民带来困难，可能与抵制外国影响的意愿软化的意图背道而驰。更有可能的是，影响到普通民众的袭击反而增强了抵抗和战斗的意志，类似于英国对1940年德国轰炸行动Blitzen的反应。我们不能排除影响到普通民众的袭击，但没有足够的对抗性进攻能力来攻击所有16个关键基础设施部门，并获得战略势头。对手的网络攻击能力有限，需要优先考虑与整体战略一致的网络目标。从逻辑上讲，对手将把他们的OCO集中在影响国家安全的行动上，并通过否认、贬低和混淆美国信息环境和美国网络资产来支持他们的军事行动。



## **40. Label-Only Membership Inference Attacks**

仅标签成员关系推理攻击 cs.CR

16 pages, 11 figures, 2 tables Revision 2: 19 pages, 12 figures, 3  tables. Improved text and additional experiments. Final ICML paper

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2007.14321v3)

**Authors**: Christopher A. Choquette-Choo, Florian Tramer, Nicholas Carlini, Nicolas Papernot

**Abstracts**: Membership inference attacks are one of the simplest forms of privacy leakage for machine learning models: given a data point and model, determine whether the point was used to train the model. Existing membership inference attacks exploit models' abnormal confidence when queried on their training data. These attacks do not apply if the adversary only gets access to models' predicted labels, without a confidence measure. In this paper, we introduce label-only membership inference attacks. Instead of relying on confidence scores, our attacks evaluate the robustness of a model's predicted labels under perturbations to obtain a fine-grained membership signal. These perturbations include common data augmentations or adversarial examples. We empirically show that our label-only membership inference attacks perform on par with prior attacks that required access to model confidences. We further demonstrate that label-only attacks break multiple defenses against membership inference attacks that (implicitly or explicitly) rely on a phenomenon we call confidence masking. These defenses modify a model's confidence scores in order to thwart attacks, but leave the model's predicted labels unchanged. Our label-only attacks demonstrate that confidence-masking is not a viable defense strategy against membership inference. Finally, we investigate worst-case label-only attacks, that infer membership for a small number of outlier data points. We show that label-only attacks also match confidence-based attacks in this setting. We find that training models with differential privacy and (strong) L2 regularization are the only known defense strategies that successfully prevents all attacks. This remains true even when the differential privacy budget is too high to offer meaningful provable guarantees.

摘要: 成员关系推理攻击是机器学习模型隐私泄露的最简单形式之一：给定一个数据点和模型，确定该点是否被用来训练该模型。现有的隶属度推理攻击利用模型在查询训练数据时的异常置信度。如果对手只能访问模型的预测标签，而没有置信度度量，则这些攻击不适用。在本文中，我们引入了仅标签成员关系推理攻击。我们的攻击不依赖于置信度分数，而是评估模型的预测标签在扰动下的鲁棒性，以获得细粒度的成员资格信号。这些扰动包括常见的数据扩充或对抗性示例。我们的经验表明，我们的仅标签成员关系推理攻击的性能与之前需要访问模型可信度的攻击相当。我们进一步证明，仅标签攻击打破了对(隐式或显式)依赖于我们称为置信度掩蔽现象的成员关系推断攻击的多个防御。这些防御措施修改模型的置信度分数以阻止攻击，但保持模型的预测标签不变。我们的仅标签攻击表明，置信度掩蔽不是一种可行的针对成员关系推断的防御策略。最后，我们研究了最坏情况下的仅标签攻击，即推断少量离群点的成员资格。我们表明，在此设置下，仅标签攻击也与基于置信度的攻击相匹配。我们发现，具有差异隐私和(强)L2正则化的训练模型是唯一已知的成功阻止所有攻击的防御策略。即使差别隐私预算太高，无法提供有意义的、可证明的保证，这一点仍然成立。



## **41. Learning Swarm Interaction Dynamics from Density Evolution**

从密度演化中学习群体相互作用动力学 eess.SY

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02675v1)

**Authors**: Christos Mavridis, Amoolya Tirumalai, John Baras

**Abstracts**: We consider the problem of understanding the coordinated movements of biological or artificial swarms. In this regard, we propose a learning scheme to estimate the coordination laws of the interacting agents from observations of the swarm's density over time. We describe the dynamics of the swarm based on pairwise interactions according to a Cucker-Smale flocking model, and express the swarm's density evolution as the solution to a system of mean-field hydrodynamic equations. We propose a new family of parametric functions to model the pairwise interactions, which allows for the mean-field macroscopic system of integro-differential equations to be efficiently solved as an augmented system of PDEs. Finally, we incorporate the augmented system in an iterative optimization scheme to learn the dynamics of the interacting agents from observations of the swarm's density evolution over time. The results of this work can offer an alternative approach to study how animal flocks coordinate, create new control schemes for large networked systems, and serve as a central part of defense mechanisms against adversarial drone attacks.

摘要: 我们考虑理解生物或人造蜂群的协调运动的问题。在这方面，我们提出了一种学习方案，通过观察种群密度随时间的变化来估计相互作用Agent的协调规律。我们根据Cucker-Smer群集模型描述了基于成对相互作用的群体动力学，并将群体密度演化表示为平均场流体动力学方程组的解。我们提出了一族新的参数函数族来模拟两两相互作用，使得平均场宏观积分微分方程组可以作为一个增广的偏微分方程组有效地求解。最后，我们将增广系统结合到迭代优化方案中，通过观察种群密度随时间的演变来学习交互Agent的动态。这项工作的结果可以提供另一种方法来研究动物群是如何协调的，为大型网络系统创造新的控制方案，并作为对抗无人机攻击的防御机制的核心部分。



## **42. Stochastic Local Winner-Takes-All Networks Enable Profound Adversarial Robustness**

随机本地赢家通吃网络实现深刻的对手鲁棒性 cs.LG

Bayesian Deep Learning Workshop, NeurIPS 2021

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2112.02671v1)

**Authors**: Konstantinos P. Panousis, Sotirios Chatzis, Sergios Theodoridis

**Abstracts**: This work explores the potency of stochastic competition-based activations, namely Stochastic Local Winner-Takes-All (LWTA), against powerful (gradient-based) white-box and black-box adversarial attacks; we especially focus on Adversarial Training settings. In our work, we replace the conventional ReLU-based nonlinearities with blocks comprising locally and stochastically competing linear units. The output of each network layer now yields a sparse output, depending on the outcome of winner sampling in each block. We rely on the Variational Bayesian framework for training and inference; we incorporate conventional PGD-based adversarial training arguments to increase the overall adversarial robustness. As we experimentally show, the arising networks yield state-of-the-art robustness against powerful adversarial attacks while retaining very high classification rate in the benign case.

摘要: 这项工作探索了基于随机竞争的激活，即随机局部赢家通吃(LWTA)，对抗强大的(基于梯度的)白盒和黑盒对抗性攻击的有效性；我们特别关注对抗性训练环境。在我们的工作中，我们用由局部和随机竞争的线性单元组成的块来代替传统的基于REU的非线性。现在，每个网络层的输出都会产生稀疏输出，具体取决于每个挡路中获胜者采样的结果。我们依靠变分贝叶斯框架进行训练和推理；我们结合了传统的基于PGD的对抗性训练论据，以增加对抗性的整体健壮性。正如我们的实验所表明的那样，出现的网络对强大的对手攻击产生了最先进的健壮性，同时在良性情况下保持了非常高的分类率。



## **43. Formalizing and Estimating Distribution Inference Risks**

配电推理风险的形式化与估计 cs.LG

Shorter version of work available at arXiv:2106.03699 Update: New  version with more theoretical results and a deeper exploration of results

**SubmitDate**: 2021-12-05    [paper-pdf](http://arxiv.org/pdf/2109.06024v4)

**Authors**: Anshuman Suri, David Evans

**Abstracts**: Distribution inference, sometimes called property inference, infers statistical properties about a training set from access to a model trained on that data. Distribution inference attacks can pose serious risks when models are trained on private data, but are difficult to distinguish from the intrinsic purpose of statistical machine learning -- namely, to produce models that capture statistical properties about a distribution. Motivated by Yeom et al.'s membership inference framework, we propose a formal definition of distribution inference attacks that is general enough to describe a broad class of attacks distinguishing between possible training distributions. We show how our definition captures previous ratio-based property inference attacks as well as new kinds of attack including revealing the average node degree or clustering coefficient of a training graph. To understand distribution inference risks, we introduce a metric that quantifies observed leakage by relating it to the leakage that would occur if samples from the training distribution were provided directly to the adversary. We report on a series of experiments across a range of different distributions using both novel black-box attacks and improved versions of the state-of-the-art white-box attacks. Our results show that inexpensive attacks are often as effective as expensive meta-classifier attacks, and that there are surprising asymmetries in the effectiveness of attacks.

摘要: 分布推理，有时被称为属性推理，从对基于该数据训练的模型的访问中推断出关于训练集的统计属性。当模型基于私有数据进行训练时，分布推断攻击可能会带来严重的风险，但很难与统计机器学习的内在目的区分开来--即，生成捕获有关分布的统计属性的模型。在Yeom等人的成员关系推理框架的启发下，我们提出了分布推理攻击的形式化定义，该定义足够通用，可以描述区分可能的训练分布的广泛的攻击类别。我们展示了我们的定义如何捕获以前的基于比率的属性推理攻击，以及新的攻击类型，包括揭示训练图的平均节点度或聚类系数。为了了解分布推理风险，我们引入了一个度量，通过将观察到的泄漏与训练分布的样本直接提供给对手时将发生的泄漏联系起来，对观察到的泄漏进行量化。我们报告了使用新颖的黑盒攻击和最先进的白盒攻击的改进版本在一系列不同的发行版上进行的一系列实验。我们的结果表明，廉价的攻击通常与昂贵的元分类器攻击一样有效，并且攻击的有效性存在惊人的不对称性。



## **44. Adv-4-Adv: Thwarting Changing Adversarial Perturbations via Adversarial Domain Adaptation**

ADV-4-ADV：通过对抗性领域适应挫败不断变化的对抗性扰动 cs.CV

9 pages

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.00428v2)

**Authors**: Tianyue Zheng, Zhe Chen, Shuya Ding, Chao Cai, Jun Luo

**Abstracts**: Whereas adversarial training can be useful against specific adversarial perturbations, they have also proven ineffective in generalizing towards attacks deviating from those used for training. However, we observe that this ineffectiveness is intrinsically connected to domain adaptability, another crucial issue in deep learning for which adversarial domain adaptation appears to be a promising solution. Consequently, we proposed Adv-4-Adv as a novel adversarial training method that aims to retain robustness against unseen adversarial perturbations. Essentially, Adv-4-Adv treats attacks incurring different perturbations as distinct domains, and by leveraging the power of adversarial domain adaptation, it aims to remove the domain/attack-specific features. This forces a trained model to learn a robust domain-invariant representation, which in turn enhances its generalization ability. Extensive evaluations on Fashion-MNIST, SVHN, CIFAR-10, and CIFAR-100 demonstrate that a model trained by Adv-4-Adv based on samples crafted by simple attacks (e.g., FGSM) can be generalized to more advanced attacks (e.g., PGD), and the performance exceeds state-of-the-art proposals on these datasets.

摘要: 虽然对抗性训练对对抗特定的对抗性干扰是有用的，但事实证明，它们也不能有效地概括出与用于训练的攻击不同的攻击。然而，我们观察到这种低效与领域适应性有内在的联系，这是深度学习中的另一个关键问题，对抗性领域适应似乎是一个有希望的解决方案。因此，我们提出了ADV-4-ADV作为一种新的对抗性训练方法，旨在保持对不可见的对抗性扰动的鲁棒性。从本质上讲，ADV-4-ADV将遭受不同扰动的攻击视为不同的域，并利用敌对域自适应的能力，旨在去除域/攻击特定的特征。这迫使训练后的模型学习健壮的领域不变表示，进而增强其泛化能力。在Fashion-MNIST、SVHN、CIFAR-10和CIFAR-100上的广泛评估表明，由ADV-4-ADV基于简单攻击(例如FGSM)构造的样本训练的模型可以推广到更高级的攻击(例如PGD)，并且性能超过了在这些数据集上的最新建议。



## **45. Statically Detecting Adversarial Malware through Randomised Chaining**

通过随机链静态检测敌意恶意软件 cs.CR

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2111.14037v2)

**Authors**: Matthew Crawford, Wei Wang, Ruoxi Sun, Minhui Xue

**Abstracts**: With the rapid growth of malware attacks, more antivirus developers consider deploying machine learning technologies into their productions. Researchers and developers published various machine learning-based detectors with high precision on malware detection in recent years. Although numerous machine learning-based malware detectors are available, they face various machine learning-targeted attacks, including evasion and adversarial attacks. This project explores how and why adversarial examples evade malware detectors, then proposes a randomised chaining method to defend against adversarial malware statically. This research is crucial for working towards combating the pertinent malware cybercrime.

摘要: 随着恶意软件攻击的快速增长，越来越多的反病毒开发人员考虑将机器学习技术部署到他们的产品中。近年来，研究人员和开发人员发布了各种基于机器学习的恶意软件检测高精度检测器。虽然有许多基于机器学习的恶意软件检测器可用，但它们面临着各种机器学习目标攻击，包括逃避和敌意攻击。该项目探讨了敌意实例如何以及为什么躲避恶意软件检测器，然后提出了一种随机链接的方法来静态防御敌意恶意软件。这项研究对于打击相关的恶意软件网络犯罪至关重要。



## **46. Generalized Likelihood Ratio Test for Adversarially Robust Hypothesis Testing**

逆稳健假设检验的广义似然比检验 stat.ML

Submitted to the IEEE Transactions on Signal Processing

**SubmitDate**: 2021-12-04    [paper-pdf](http://arxiv.org/pdf/2112.02209v1)

**Authors**: Bhagyashree Puranik, Upamanyu Madhow, Ramtin Pedarsani

**Abstracts**: Machine learning models are known to be susceptible to adversarial attacks which can cause misclassification by introducing small but well designed perturbations. In this paper, we consider a classical hypothesis testing problem in order to develop fundamental insight into defending against such adversarial perturbations. We interpret an adversarial perturbation as a nuisance parameter, and propose a defense based on applying the generalized likelihood ratio test (GLRT) to the resulting composite hypothesis testing problem, jointly estimating the class of interest and the adversarial perturbation. While the GLRT approach is applicable to general multi-class hypothesis testing, we first evaluate it for binary hypothesis testing in white Gaussian noise under $\ell_{\infty}$ norm-bounded adversarial perturbations, for which a known minimax defense optimizing for the worst-case attack provides a benchmark. We derive the worst-case attack for the GLRT defense, and show that its asymptotic performance (as the dimension of the data increases) approaches that of the minimax defense. For non-asymptotic regimes, we show via simulations that the GLRT defense is competitive with the minimax approach under the worst-case attack, while yielding a better robustness-accuracy tradeoff under weaker attacks. We also illustrate the GLRT approach for a multi-class hypothesis testing problem, for which a minimax strategy is not known, evaluating its performance under both noise-agnostic and noise-aware adversarial settings, by providing a method to find optimal noise-aware attacks, and heuristics to find noise-agnostic attacks that are close to optimal in the high SNR regime.

摘要: 众所周知，机器学习模型容易受到敌意攻击，这种攻击可能会通过引入小但设计良好的扰动而导致误分类。在这篇文章中，我们考虑了一个经典的假设检验问题，以发展对这种敌对扰动的防御的基本见解。我们将敌意扰动解释为干扰参数，并提出了一种基于广义似然比检验(GLRT)的防御方法，将广义似然比检验应用于由此产生的复合假设检验问题，联合估计感兴趣的类别和对抗性扰动。虽然GLRT方法适用于一般的多类假设检验，但我们首先评估了它在高斯白噪声中范数有界的对抗扰动下的二元假设检验，一个已知的针对最坏情况攻击的极小极大防御优化提供了一个基准。我们推导了GLRT防御的最坏情况攻击，并证明了它的渐近性能(随着数据维数的增加)接近极小极大防御的渐近性能。对于非渐近体制，我们通过仿真表明，在最坏情况下，广义似然比防御是基于极小极大方法的好胜防御，而在较弱攻击下获得了较好的稳健性和准确性折衷。我们还举例说明了GLRT方法用于多类假设检验问题，对于未知的极小极大策略，通过提供一种寻找最优噪声感知攻击的方法和寻找在高信噪比条件下接近最优的噪声不可知攻击的启发式方法，来评估其在噪声不可知性和噪声感知对抗环境下的性能。



## **47. IRShield: A Countermeasure Against Adversarial Physical-Layer Wireless Sensing**

IRShield：对抗敌意物理层无线传感的对策 cs.CR

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01967v1)

**Authors**: Paul Staat, Simon Mulzer, Stefan Roth, Veelasha Moonsamy, Aydin Sezgin, Christof Paar

**Abstracts**: Wireless radio channels are known to contain information about the surrounding propagation environment, which can be extracted using established wireless sensing methods. Thus, today's ubiquitous wireless devices are attractive targets for passive eavesdroppers to launch reconnaissance attacks. In particular, by overhearing standard communication signals, eavesdroppers obtain estimations of wireless channels which can give away sensitive information about indoor environments. For instance, by applying simple statistical methods, adversaries can infer human motion from wireless channel observations, allowing to remotely monitor premises of victims. In this work, building on the advent of intelligent reflecting surfaces (IRSs), we propose IRShield as a novel countermeasure against adversarial wireless sensing. IRShield is designed as a plug-and-play privacy-preserving extension to existing wireless networks. At the core of IRShield, we design an IRS configuration algorithm to obfuscate wireless channels. We validate the effectiveness with extensive experimental evaluations. In a state-of-the-art human motion detection attack using off-the-shelf Wi-Fi devices, IRShield lowered detection rates to 5% or less.

摘要: 众所周知，无线无线信道包含有关周围传播环境的信息，可以使用已建立的无线侦听方法提取这些信息。因此，今天无处不在的无线设备是被动窃听者发动侦察攻击的诱人目标。特别是，通过偷听标准通信信号，窃听者可以获得对无线信道的估计，这可能会泄露有关室内环境的敏感信息。例如，通过应用简单的统计方法，攻击者可以从无线信道观测中推断人体运动，从而允许远程监控受害者的办公场所。在这项工作中，基于智能反射面(IRS)的出现，我们提出了IRShield作为对抗敌意无线传感的一种新的对策。IRShield被设计为现有无线网络的即插即用隐私保护扩展。在IRShield的核心部分，我们设计了一种IRS配置算法来对无线信道进行模糊处理。我们通过大量的实验评估验证了该方法的有效性。在一次使用现成Wi-Fi设备的最先进的人体运动检测攻击中，IRShield将检测率降至5%或更低。



## **48. Mind the box: $l_1$-APGD for sparse adversarial attacks on image classifiers**

注意方框：$l_1$-针对图像分类器的稀疏对抗性攻击的APGD cs.LG

In ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.01208v2)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: We show that when taking into account also the image domain $[0,1]^d$, established $l_1$-projected gradient descent (PGD) attacks are suboptimal as they do not consider that the effective threat model is the intersection of the $l_1$-ball and $[0,1]^d$. We study the expected sparsity of the steepest descent step for this effective threat model and show that the exact projection onto this set is computationally feasible and yields better performance. Moreover, we propose an adaptive form of PGD which is highly effective even with a small budget of iterations. Our resulting $l_1$-APGD is a strong white-box attack showing that prior works overestimated their $l_1$-robustness. Using $l_1$-APGD for adversarial training we get a robust classifier with SOTA $l_1$-robustness. Finally, we combine $l_1$-APGD and an adaptation of the Square Attack to $l_1$ into $l_1$-AutoAttack, an ensemble of attacks which reliably assesses adversarial robustness for the threat model of $l_1$-ball intersected with $[0,1]^d$.

摘要: 我们证明了当同时考虑象域$[0，1]^d$时，所建立的$l_1$投影梯度下降(PGD)攻击是次优的，因为它们没有考虑到有效的威胁模型是$l_1$球和$[0，1]^d$的交集。我们研究了该有效威胁模型的最陡下降步长的期望稀疏性，并证明了在该集合上的精确投影在计算上是可行的，并且产生了更好的性能。此外，我们还提出了一种自适应形式的PGD，即使在很小的迭代预算下也是非常有效的。我们得到的$l_1$-APGD是一个强白盒攻击，表明以前的工作高估了它们的$l_1$-稳健性。利用$l_1$-APGD进行对抗性训练，得到一个具有SOTA$l_1$-鲁棒性的鲁棒分类器。最后，我们将$l_1$-APGD和对$l_1$的Square攻击的改进结合成$l_1$-AutoAttack，这是一个攻击集合，它可靠地评估了$l_1$-ball与$[0，1]^d$相交的威胁模型的对手健壮性。



## **49. Graph Neural Networks Inspired by Classical Iterative Algorithms**

受经典迭代算法启发的图神经网络 cs.LG

accepted as long oral for ICML 2021

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2103.06064v4)

**Authors**: Yongyi Yang, Tang Liu, Yangkun Wang, Jinjing Zhou, Quan Gan, Zhewei Wei, Zheng Zhang, Zengfeng Huang, David Wipf

**Abstracts**: Despite the recent success of graph neural networks (GNN), common architectures often exhibit significant limitations, including sensitivity to oversmoothing, long-range dependencies, and spurious edges, e.g., as can occur as a result of graph heterophily or adversarial attacks. To at least partially address these issues within a simple transparent framework, we consider a new family of GNN layers designed to mimic and integrate the update rules of two classical iterative algorithms, namely, proximal gradient descent and iterative reweighted least squares (IRLS). The former defines an extensible base GNN architecture that is immune to oversmoothing while nonetheless capturing long-range dependencies by allowing arbitrary propagation steps. In contrast, the latter produces a novel attention mechanism that is explicitly anchored to an underlying end-to-end energy function, contributing stability with respect to edge uncertainty. When combined we obtain an extremely simple yet robust model that we evaluate across disparate scenarios including standardized benchmarks, adversarially-perturbated graphs, graphs with heterophily, and graphs involving long-range dependencies. In doing so, we compare against SOTA GNN approaches that have been explicitly designed for the respective task, achieving competitive or superior node classification accuracy. Our code is available at https://github.com/FFTYYY/TWIRLS.

摘要: 尽管图神经网络(GNN)最近取得了成功，但常见的体系结构通常表现出显著的局限性，包括对过度平滑、长范围依赖和伪边的敏感性，例如，由于图的异嗜性或敌意攻击而可能发生的情况。为了在一个简单透明的框架内至少部分解决这些问题，我们考虑了一族新的GNN层，它们被设计成模仿和集成两种经典迭代算法的更新规则，即最近梯度下降和迭代重加权最小二乘(IRLS)。前者定义了一个可扩展的基本GNN体系结构，该体系结构不受过度平滑的影响，同时通过允许任意传播步骤来捕获远程依赖关系。相反，后者产生了一种新的注意机制，该机制显式地锚定在潜在的端到端能量函数上，有助于相对于边缘不确定性的稳定性。当组合在一起时，我们得到了一个极其简单但健壮的模型，我们可以跨不同的场景进行评估，包括标准化的基准测试、对抗性干扰图、具有异质性的图以及涉及长范围依赖的图。在此过程中，我们将其与明确为各自任务设计的Sota GNN方法进行比较，以达到好胜或更高的节点分类精度。我们的代码可在https://github.com/FFTYYY/TWIRLS.获得



## **50. Blackbox Untargeted Adversarial Testing of Automatic Speech Recognition Systems**

自动语音识别系统的黑盒非目标对抗性测试 cs.SD

10 pages, 6 figures and 7 tables

**SubmitDate**: 2021-12-03    [paper-pdf](http://arxiv.org/pdf/2112.01821v1)

**Authors**: Xiaoliang Wu, Ajitha Rajan

**Abstracts**: Automatic speech recognition (ASR) systems are prevalent, particularly in applications for voice navigation and voice control of domestic appliances. The computational core of ASRs are deep neural networks (DNNs) that have been shown to be susceptible to adversarial perturbations; easily misused by attackers to generate malicious outputs. To help test the correctness of ASRS, we propose techniques that automatically generate blackbox (agnostic to the DNN), untargeted adversarial attacks that are portable across ASRs. Much of the existing work on adversarial ASR testing focuses on targeted attacks, i.e generating audio samples given an output text. Targeted techniques are not portable, customised to the structure of DNNs (whitebox) within a specific ASR. In contrast, our method attacks the signal processing stage of the ASR pipeline that is shared across most ASRs. Additionally, we ensure the generated adversarial audio samples have no human audible difference by manipulating the acoustic signal using a psychoacoustic model that maintains the signal below the thresholds of human perception. We evaluate portability and effectiveness of our techniques using three popular ASRs and three input audio datasets using the metrics - WER of output text, Similarity to original audio and attack Success Rate on different ASRs. We found our testing techniques were portable across ASRs, with the adversarial audio samples producing high Success Rates, WERs and Similarities to the original audio.

摘要: 自动语音识别(ASR)系统很普遍，特别是在用于语音导航和家用电器的语音控制的应用中。ASR的计算核心是深度神经网络(DNNs)，已经证明它们容易受到对手的干扰；很容易被攻击者误用来生成恶意输出。为了帮助测试ASR的正确性，我们提出了自动生成黑盒(与DNN无关)的技术，这是一种可跨ASR移植的无目标对抗性攻击。对抗性ASR测试的大部分现有工作都集中在有针对性的攻击上，即在给定输出文本的情况下生成音频样本。目标技术不是便携的，不能根据特定ASR内的DNN(白盒)结构进行定制。相反，我们的方法攻击大多数ASR共享的ASR流水线的信号处理阶段。此外，我们通过使用将信号保持在人类感知阈值以下的心理声学模型来处理声音信号，以确保生成的敌意音频样本没有人耳可闻的差异。我们使用三个流行的ASR和三个输入音频数据集，使用输出文本的WER、与原始音频的相似度和对不同ASR的攻击成功率来评估我们的技术的可移植性和有效性。我们发现我们的测试技术在ASR之间是可移植的，敌意音频样本产生了很高的成功率，与原始音频有很大的相似之处。



