# Latest Adversarial Attack Papers
**update at 2022-10-14 17:17:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

Pikachu：通过使用Taproot检查点进入比特币PoW来保护PoS区块链免受远程攻击 cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2208.05408v2) [paper-pdf](http://arxiv.org/pdf/2208.05408v2)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstract**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.

摘要: 基于可重用资源的区块链系统，如风险证明(POS)，提供的安全保证比基于工作证明的系统更弱。具体地说，它们容易受到远程攻击，在远程攻击中，对手可以破坏之前的参与者，以便重写链的完整历史。为了防止这种对PoS链的攻击，我们提出了一种协议，将PoS链的状态检查到工作证明区块链，如比特币。因此，我们的检查点协议不依赖于任何中央机构。我们的工作使用Schnorr签名并利用比特币最近的Taproot升级，使我们能够创建恒定大小的检查点交易。我们为协议的安全性进行了论证，并给出了一个在比特币测试网上进行测试的开源实现。



## **2. AccelAT: A Framework for Accelerating the Adversarial Training of Deep Neural Networks through Accuracy Gradient**

AccelAT：一种通过精度梯度加速深度神经网络对抗性训练的框架 cs.LG

12 pages

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06888v1) [paper-pdf](http://arxiv.org/pdf/2210.06888v1)

**Authors**: Farzad Nikfam, Alberto Marchisio, Maurizio Martina, Muhammad Shafique

**Abstract**: Adversarial training is exploited to develop a robust Deep Neural Network (DNN) model against the malicious altered data. These attacks may have catastrophic effects on DNN models but are indistinguishable for a human being. For example, an external attack can modify an image adding noises invisible for a human eye, but a DNN model misclassified the image. A key objective for developing robust DNN models is to use a learning algorithm that is fast but can also give model that is robust against different types of adversarial attacks. Especially for adversarial training, enormously long training times are needed for obtaining high accuracy under many different types of adversarial samples generated using different adversarial attack techniques.   This paper aims at accelerating the adversarial training to enable fast development of robust DNN models against adversarial attacks. The general method for improving the training performance is the hyperparameters fine-tuning, where the learning rate is one of the most crucial hyperparameters. By modifying its shape (the value over time) and value during the training, we can obtain a model robust to adversarial attacks faster than standard training.   First, we conduct experiments on two different datasets (CIFAR10, CIFAR100), exploring various techniques. Then, this analysis is leveraged to develop a novel fast training methodology, AccelAT, which automatically adjusts the learning rate for different epochs based on the accuracy gradient. The experiments show comparable results with the related works, and in several experiments, the adversarial training of DNNs using our AccelAT framework is conducted up to 2 times faster than the existing techniques. Thus, our findings boost the speed of adversarial training in an era in which security and performance are fundamental optimization objectives in DNN-based applications.

摘要: 利用对抗性训练建立了一种针对恶意篡改数据的稳健深度神经网络(DNN)模型。这些攻击可能会对DNN模型产生灾难性影响，但对人类来说是无法区分的。例如，外部攻击可以修改图像，添加人眼看不见的噪声，但DNN模型错误地分类了图像。开发健壮DNN模型的一个关键目标是使用一种快速的学习算法，并且能够给出对不同类型的对手攻击具有健壮性的模型。特别是对于对抗性训练，在使用不同的对抗性攻击技术生成的许多不同类型的对抗性样本下，需要非常长的训练时间才能获得高的准确率。本文的目的是加速对抗性训练，以便快速开发出抵抗对抗性攻击的稳健DNN模型。提高训练性能的一般方法是超参数微调，其中学习率是最关键的超参数之一。通过在训练过程中修改其形状(随时间变化的值)和值，我们可以得到一个比标准训练更快地抗击对手攻击的模型。首先，我们在两个不同的数据集(CIFAR10，CIFAR100)上进行了实验，探索了各种技术。然后，利用这一分析来开发一种新的快速训练方法AccelAT，该方法根据精度梯度自动调整不同历元的学习率。实验结果与相关工作具有可比性，在多个实验中，使用AccelAT框架对DNN进行对抗性训练的速度比现有技术快2倍。因此，在安全性和性能是基于DNN的应用程序的基本优化目标的时代，我们的发现提高了对抗性训练的速度。



## **3. Adv-Attribute: Inconspicuous and Transferable Adversarial Attack on Face Recognition**

ADV-ATTRIBUTE：对人脸识别的隐蔽且可转移的敌意攻击 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06871v1) [paper-pdf](http://arxiv.org/pdf/2210.06871v1)

**Authors**: Shuai Jia, Bangjie Yin, Taiping Yao, Shouhong Ding, Chunhua Shen, Xiaokang Yang, Chao Ma

**Abstract**: Deep learning models have shown their vulnerability when dealing with adversarial attacks. Existing attacks almost perform on low-level instances, such as pixels and super-pixels, and rarely exploit semantic clues. For face recognition attacks, existing methods typically generate the l_p-norm perturbations on pixels, however, resulting in low attack transferability and high vulnerability to denoising defense models. In this work, instead of performing perturbations on the low-level pixels, we propose to generate attacks through perturbing on the high-level semantics to improve attack transferability. Specifically, a unified flexible framework, Adversarial Attributes (Adv-Attribute), is designed to generate inconspicuous and transferable attacks on face recognition, which crafts the adversarial noise and adds it into different attributes based on the guidance of the difference in face recognition features from the target. Moreover, the importance-aware attribute selection and the multi-objective optimization strategy are introduced to further ensure the balance of stealthiness and attacking strength. Extensive experiments on the FFHQ and CelebA-HQ datasets show that the proposed Adv-Attribute method achieves the state-of-the-art attacking success rates while maintaining better visual effects against recent attack methods.

摘要: 深度学习模型在处理对抗性攻击时显示出了它们的脆弱性。现有的攻击几乎是在低层实例上执行的，例如像素和超像素，很少利用语义线索。对于人脸识别攻击，现有的方法通常会产生像素上的l_p范数扰动，导致攻击可传递性低，对去噪防御模型的脆弱性高。在这项工作中，我们不是对低层像素进行扰动，而是通过对高层语义的扰动来产生攻击，以提高攻击的可转移性。具体地说，设计了一个统一的灵活框架--对抗性属性(ADV-ATTRIBUTE)，用于产生对人脸识别的隐蔽性和可转移性攻击，该框架根据人脸识别特征与目标的差异指导生成对抗性噪声并将其添加到不同的属性中。此外，引入了重要性感知的属性选择和多目标优化策略，进一步保证了隐蔽性和攻击力的平衡。在FFHQ和CelebA-HQ数据集上的大量实验表明，所提出的ADV属性方法达到了最先进的攻击成功率，同时对最近的攻击方法保持了更好的视觉效果。



## **4. Federated Learning for Tabular Data: Exploring Potential Risk to Privacy**

表格数据的联合学习：探索隐私的潜在风险 cs.CR

In the proceedings of The 33rd IEEE International Symposium on  Software Reliability Engineering (ISSRE), November 2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06856v1) [paper-pdf](http://arxiv.org/pdf/2210.06856v1)

**Authors**: Han Wu, Zilong Zhao, Lydia Y. Chen, Aad van Moorsel

**Abstract**: Federated Learning (FL) has emerged as a potentially powerful privacy-preserving machine learning methodology, since it avoids exchanging data between participants, but instead exchanges model parameters. FL has traditionally been applied to image, voice and similar data, but recently it has started to draw attention from domains including financial services where the data is predominantly tabular. However, the work on tabular data has not yet considered potential attacks, in particular attacks using Generative Adversarial Networks (GANs), which have been successfully applied to FL for non-tabular data. This paper is the first to explore leakage of private data in Federated Learning systems that process tabular data. We design a Generative Adversarial Networks (GANs)-based attack model which can be deployed on a malicious client to reconstruct data and its properties from other participants. As a side-effect of considering tabular data, we are able to statistically assess the efficacy of the attack (without relying on human observation such as done for FL for images). We implement our attack model in a recently developed generic FL software framework for tabular data processing. The experimental results demonstrate the effectiveness of the proposed attack model, thus suggesting that further research is required to counter GAN-based privacy attacks.

摘要: 联合学习(FL)已经成为一种潜在的强大的隐私保护机器学习方法，因为它避免了参与者之间交换数据，而是交换模型参数。传统上，FL被应用于图像、语音和类似数据，但最近它开始吸引包括金融服务在内的领域的注意，这些领域的数据主要是表格。然而，关于表格数据的工作还没有考虑到潜在的攻击，特别是使用生成性对抗网络(GANS)的攻击，这些攻击已经成功地应用于非表格数据的FL。本文首次探讨了联邦学习系统中处理表格数据的私有数据泄漏问题。我们设计了一个基于生成性对抗网络(GANS)的攻击模型，该模型可以部署在恶意客户端上，以重构来自其他参与者的数据及其属性。考虑表格数据的一个副作用是，我们能够在统计上评估攻击的效果(不依赖于人的观察，如对图像的FL所做的)。我们在最近开发的用于表格数据处理的通用FL软件框架中实现了我们的攻击模型。实验结果证明了该攻击模型的有效性，表明需要对基于GAN的隐私攻击进行进一步的研究。



## **5. Observed Adversaries in Deep Reinforcement Learning**

深度强化学习中观察到的对手 cs.LG

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06787v1) [paper-pdf](http://arxiv.org/pdf/2210.06787v1)

**Authors**: Eugene Lim, Harold Soh

**Abstract**: In this work, we point out the problem of observed adversaries for deep policies. Specifically, recent work has shown that deep reinforcement learning is susceptible to adversarial attacks where an observed adversary acts under environmental constraints to invoke natural but adversarial observations. This setting is particularly relevant for HRI since HRI-related robots are expected to perform their tasks around and with other agents. In this work, we demonstrate that this effect persists even with low-dimensional observations. We further show that these adversarial attacks transfer across victims, which potentially allows malicious attackers to train an adversary without access to the target victim.

摘要: 在这项工作中，我们指出了深度政策的观察对手的问题。具体地说，最近的工作表明，深度强化学习容易受到对抗性攻击，即被观察到的对手在环境约束下采取行动，援引自然但对抗性的观察。这一设置与HRI特别相关，因为与HRI相关的机器人应该在其他代理周围和与其他代理一起执行任务。在这项工作中，我们证明了即使在低维观测中，这种效应仍然存在。我们进一步表明，这些对抗性攻击在受害者之间转移，这可能允许恶意攻击者在没有访问目标受害者的情况下训练对手。



## **6. COLLIDER: A Robust Training Framework for Backdoor Data**

Collider：一个健壮的后门数据训练框架 cs.LG

Accepted to the 16th Asian Conference on Computer Vision (ACCV 2022)

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06704v1) [paper-pdf](http://arxiv.org/pdf/2210.06704v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Deep neural network (DNN) classifiers are vulnerable to backdoor attacks. An adversary poisons some of the training data in such attacks by installing a trigger. The goal is to make the trained DNN output the attacker's desired class whenever the trigger is activated while performing as usual for clean data. Various approaches have recently been proposed to detect malicious backdoored DNNs. However, a robust, end-to-end training approach, like adversarial training, is yet to be discovered for backdoor poisoned data. In this paper, we take the first step toward such methods by developing a robust training framework, COLLIDER, that selects the most prominent samples by exploiting the underlying geometric structures of the data. Specifically, we effectively filter out candidate poisoned data at each training epoch by solving a geometrical coreset selection objective. We first argue how clean data samples exhibit (1) gradients similar to the clean majority of data and (2) low local intrinsic dimensionality (LID). Based on these criteria, we define a novel coreset selection objective to find such samples, which are used for training a DNN. We show the effectiveness of the proposed method for robust training of DNNs on various poisoned datasets, reducing the backdoor success rate significantly.

摘要: 深度神经网络(DNN)分类器容易受到后门攻击。对手通过安装触发器来毒化此类攻击中的一些训练数据。这样做的目的是让经过训练的DNN在触发器被激活时输出攻击者想要的类，同时像往常一样执行干净的数据。最近提出了各种方法来检测恶意回溯的DNN。然而，一种强大的端到端培训方法，如对抗性培训，尚未发现针对后门有毒数据的方法。在本文中，我们通过开发一个健壮的训练框架Collider来朝着这种方法迈出第一步，该框架通过利用数据的基本几何结构来选择最突出的样本。具体地说，我们通过求解几何核心重置选择目标，有效地过滤出每个训练时期的候选有毒数据。我们首先讨论干净的数据样本如何表现出(1)类似于干净的大多数数据的梯度和(2)低的局部固有维度(LID)。基于这些准则，我们定义了一种新的核心选择目标来寻找用于训练DNN的样本。我们在不同的有毒数据集上展示了所提出的方法用于DNN稳健训练的有效性，显著降低了后门成功率。



## **7. A Game Theoretical vulnerability analysis of Adversarial Attack**

对抗性攻击的博弈论脆弱性分析 cs.GT

Accepted in 17th International Symposium on Visual Computing,2022

**SubmitDate**: 2022-10-13    [abs](http://arxiv.org/abs/2210.06670v1) [paper-pdf](http://arxiv.org/pdf/2210.06670v1)

**Authors**: Khondker Fariha Hossain, Alireza Tavakkoli, Shamik Sengupta

**Abstract**: In recent times deep learning has been widely used for automating various security tasks in Cyber Domains. However, adversaries manipulate data in many situations and diminish the deployed deep learning model's accuracy. One notable example is fooling CAPTCHA data to access the CAPTCHA-based Classifier leading to the critical system being vulnerable to cybersecurity attacks. To alleviate this, we propose a computational framework of game theory to analyze the CAPTCHA-based Classifier's vulnerability, strategy, and outcomes by forming a simultaneous two-player game. We apply the Fast Gradient Symbol Method (FGSM) and One Pixel Attack on CAPTCHA Data to imitate real-life scenarios of possible cyber-attack. Subsequently, to interpret this scenario from a Game theoretical perspective, we represent the interaction in the Stackelberg Game in Kuhn tree to study players' possible behaviors and actions by applying our Classifier's actual predicted values. Thus, we interpret potential attacks in deep learning applications while representing viable defense strategies in the game theory prospect.

摘要: 近年来，深度学习被广泛用于自动化网络领域中的各种安全任务。然而，敌手在许多情况下操纵数据，降低了部署的深度学习模型的准确性。一个值得注意的例子是欺骗验证码数据访问基于验证码的分类器，导致关键系统容易受到网络安全攻击。为了缓解这一问题，我们提出了一个博弈论的计算框架，通过形成一个同时的两人博弈来分析基于验证码的分类器的脆弱性、策略和结果。我们对验证码数据应用快速梯度符号方法(FGSM)和单像素攻击来模拟可能的网络攻击的真实场景。随后，为了从博弈论的角度解释这一场景，我们将Stackelberg博弈中的交互表示在Kuhn树中，通过应用我们的分类器的实际预测值来研究玩家可能的行为和行动。因此，我们解释了深度学习应用中的潜在攻击，同时表示了博弈论前景中可行的防御策略。



## **8. Understanding Impacts of Task Similarity on Backdoor Attack and Detection**

了解任务相似度对后门攻击和检测的影响 cs.CR

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06509v1) [paper-pdf](http://arxiv.org/pdf/2210.06509v1)

**Authors**: Di Tang, Rui Zhu, XiaoFeng Wang, Haixu Tang, Yi Chen

**Abstract**: With extensive studies on backdoor attack and detection, still fundamental questions are left unanswered regarding the limits in the adversary's capability to attack and the defender's capability to detect. We believe that answers to these questions can be found through an in-depth understanding of the relations between the primary task that a benign model is supposed to accomplish and the backdoor task that a backdoored model actually performs. For this purpose, we leverage similarity metrics in multi-task learning to formally define the backdoor distance (similarity) between the primary task and the backdoor task, and analyze existing stealthy backdoor attacks, revealing that most of them fail to effectively reduce the backdoor distance and even for those that do, still much room is left to further improve their stealthiness. So we further design a new method, called TSA attack, to automatically generate a backdoor model under a given distance constraint, and demonstrate that our new attack indeed outperforms existing attacks, making a step closer to understanding the attacker's limits. Most importantly, we provide both theoretic results and experimental evidence on various datasets for the positive correlation between the backdoor distance and backdoor detectability, demonstrating that indeed our task similarity analysis help us better understand backdoor risks and has the potential to identify more effective mitigations.

摘要: 随着对后门攻击和检测的广泛研究，仍然没有回答关于对手攻击能力和防御者检测能力的限制的根本问题。我们相信，通过深入理解良性模型应该完成的主要任务和后门模型实际执行的后门任务之间的关系，可以找到这些问题的答案。为此，我们利用多任务学习中的相似性度量来形式化地定义主任务和后门任务之间的后门距离(相似性)，并分析了现有的隐身后门攻击，发现它们中的大多数都无法有效地减少后门距离，即使对于那些能够有效降低后门距离的攻击，仍然有很大的空间来进一步提高它们的隐蔽性。因此，我们进一步设计了一种新的方法，称为TSA攻击，在给定的距离约束下自动生成一个后门模型，并证明了我们的新攻击确实比现有的攻击性能更好，使我们更接近了解攻击者的限制。最重要的是，我们提供了理论结果和在各种数据集上的实验证据，证明了后门距离和后门可检测性之间的正相关关系，表明我们的任务相似性分析确实有助于我们更好地理解后门风险，并有可能识别更有效的缓解措施。



## **9. On Attacking Out-Domain Uncertainty Estimation in Deep Neural Networks**

关于深度神经网络攻击域外不确定性估计问题 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02191v2) [paper-pdf](http://arxiv.org/pdf/2210.02191v2)

**Authors**: Huimin Zeng, Zhenrui Yue, Yang Zhang, Ziyi Kou, Lanyu Shang, Dong Wang

**Abstract**: In many applications with real-world consequences, it is crucial to develop reliable uncertainty estimation for the predictions made by the AI decision systems. Targeting at the goal of estimating uncertainty, various deep neural network (DNN) based uncertainty estimation algorithms have been proposed. However, the robustness of the uncertainty returned by these algorithms has not been systematically explored. In this work, to raise the awareness of the research community on robust uncertainty estimation, we show that state-of-the-art uncertainty estimation algorithms could fail catastrophically under our proposed adversarial attack despite their impressive performance on uncertainty estimation. In particular, we aim at attacking the out-domain uncertainty estimation: under our attack, the uncertainty model would be fooled to make high-confident predictions for the out-domain data, which they originally would have rejected. Extensive experimental results on various benchmark image datasets show that the uncertainty estimated by state-of-the-art methods could be easily corrupted by our attack.

摘要: 在许多具有真实世界后果的应用中，为人工智能决策系统做出的预测开发可靠的不确定性估计是至关重要的。针对不确定性估计的目标，人们提出了各种基于深度神经网络(DNN)的不确定性估计算法。然而，这些算法返回的不确定性的稳健性还没有得到系统的探讨。在这项工作中，为了提高研究界对稳健不确定性估计的认识，我们证明了最新的不确定性估计算法在我们提出的对抗性攻击下可能会灾难性地失败，尽管它们在不确定性估计方面的表现令人印象深刻。特别是，我们的目标是攻击域外的不确定性估计：在我们的攻击下，不确定性模型将被愚弄，以对域外数据做出高度自信的预测，而他们最初会拒绝这些预测。在不同基准图像数据集上的大量实验结果表明，最新方法估计的不确定性很容易被我们的攻击所破坏。



## **10. On Optimal Learning Under Targeted Data Poisoning**

目标数据中毒下的最优学习问题研究 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.02713v2) [paper-pdf](http://arxiv.org/pdf/2210.02713v2)

**Authors**: Steve Hanneke, Amin Karbasi, Mohammad Mahmoody, Idan Mehalel, Shay Moran

**Abstract**: Consider the task of learning a hypothesis class $\mathcal{H}$ in the presence of an adversary that can replace up to an $\eta$ fraction of the examples in the training set with arbitrary adversarial examples. The adversary aims to fail the learner on a particular target test point $x$ which is known to the adversary but not to the learner. In this work we aim to characterize the smallest achievable error $\epsilon=\epsilon(\eta)$ by the learner in the presence of such an adversary in both realizable and agnostic settings. We fully achieve this in the realizable setting, proving that $\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $\mathtt{VC}(\mathcal{H})$ is the VC dimension of $\mathcal{H}$. Remarkably, we show that the upper bound can be attained by a deterministic learner. In the agnostic setting we reveal a more elaborate landscape: we devise a deterministic learner with a multiplicative regret guarantee of $\epsilon \leq C\cdot\mathtt{OPT} + O(\mathtt{VC}(\mathcal{H})\cdot \eta)$, where $C > 1$ is a universal numerical constant. We complement this by showing that for any deterministic learner there is an attack which worsens its error to at least $2\cdot \mathtt{OPT}$. This implies that a multiplicative deterioration in the regret is unavoidable in this case. Finally, the algorithms we develop for achieving the optimal rates are inherently improper. Nevertheless, we show that for a variety of natural concept classes, such as linear classifiers, it is possible to retain the dependence $\epsilon=\Theta_{\mathcal{H}}(\eta)$ by a proper algorithm in the realizable setting. Here $\Theta_{\mathcal{H}}$ conceals a polynomial dependence on $\mathtt{VC}(\mathcal{H})$.

摘要: 考虑在对手在场的情况下学习假设类$\mathcal{H}$的任务，该对手可以用任意的对抗性例子替换训练集中的$\eta$分数的例子。对手的目标是在对手知道但学习者不知道的特定目标测试点$x$上让学习者不及格。在这项工作中，我们的目标是刻画在可实现和不可知的情况下，学习者在这样的对手存在的情况下所能达到的最小误差。我们在可实现的设置下完全实现了这一点，证明了$\epsilon=\Theta(\mathtt{VC}(\mathcal{H})\cdot\eta)$，其中$\mathtt{VC}(\mathcal{H})$是$\mathcal{H}$的VC维。值得注意的是，我们证明了这一上界可以由确定性学习者获得。在不可知论的背景下，我们展示了一个更精细的场景：我们设计了一个确定性学习者，其乘性后悔保证为$\epsilon\leq C\cdot\mathtt{opt}+O(\mathtt{VC}(\mathcal{H})\cdot\eta)$，其中$C>1$是通用数值常量。我们的补充是，对于任何确定性学习者，都存在将其错误恶化到至少$2\cdot\mathtt{opt}$的攻击。这意味着，在这种情况下，遗憾的成倍恶化是不可避免的。最后，我们开发的用于实现最优速率的算法本质上是不正确的。然而，我们证明了对于各种自然概念类，例如线性分类器，在可实现的设置下，通过适当的算法可以保持依赖关系$\epsilon=\tha_{\mathcal{H}}(\eta)$。这里，$theta_{\mathcal{H}}$隐藏了对$\mathtt{VC}(\mathcal{H})$的多项式依赖关系。



## **11. Alleviating Adversarial Attacks on Variational Autoencoders with MCMC**

利用MCMC减轻对变分自动编码器的敌意攻击 cs.LG

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2203.09940v2) [paper-pdf](http://arxiv.org/pdf/2203.09940v2)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstract**: Variational autoencoders (VAEs) are latent variable models that can generate complex objects and provide meaningful latent representations. Moreover, they could be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attack construction proposed previously and present a solution to alleviate the effect of these attacks. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step that we motivate with a theoretical analysis. Thus, we do not incorporate any extra costs during training, and the performance on non-attacked inputs is not decreased. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, $\beta$-TCVAE), and show that our approach consistently improves the model robustness to adversarial attacks.

摘要: 变分自动编码器(VAE)是一种潜在变量模型，可以生成复杂的对象并提供有意义的潜在表示。此外，它们还可以进一步用于分类等下游任务。正如以前的工作所表明的，人们可以很容易地愚弄VAE，为视觉上稍有修改的输入产生意想不到的潜在表示和重建。在这里，我们检查了几个以前提出的对抗性攻击构造的目标函数，并提出了一个解决方案来减轻这些攻击的影响。我们的方法在推理步骤中使用了马尔科夫链蒙特卡罗(MCMC)技术，并进行了理论分析。因此，我们在训练期间不会纳入任何额外的成本，并且非攻击输入的性能不会降低。我们在各种数据集(MNIST、Fashion MNIST、Color MNIST、CelebA)和VAE配置($\beta$-VAE、NVAE、$\beta$-TCVAE)上验证了我们的方法，并表明我们的方法持续提高了模型对对手攻击的稳健性。



## **12. Visual Prompting for Adversarial Robustness**

对抗健壮性的视觉提示 cs.CV

6 pages, 4 figures, 3 tables

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06284v1) [paper-pdf](http://arxiv.org/pdf/2210.06284v1)

**Authors**: Aochuan Chen, Peter Lorenz, Yuguang Yao, Pin-Yu Chen, Sijia Liu

**Abstract**: In this work, we leverage visual prompting (VP) to improve adversarial robustness of a fixed, pre-trained model at testing time. Compared to conventional adversarial defenses, VP allows us to design universal (i.e., data-agnostic) input prompting templates, which have plug-and-play capabilities at testing time to achieve desired model performance without introducing much computation overhead. Although VP has been successfully applied to improving model generalization, it remains elusive whether and how it can be used to defend against adversarial attacks. We investigate this problem and show that the vanilla VP approach is not effective in adversarial defense since a universal input prompt lacks the capacity for robust learning against sample-specific adversarial perturbations. To circumvent it, we propose a new VP method, termed Class-wise Adversarial Visual Prompting (C-AVP), to generate class-wise visual prompts so as to not only leverage the strengths of ensemble prompts but also optimize their interrelations to improve model robustness. Our experiments show that C-AVP outperforms the conventional VP method, with 2.1X standard accuracy gain and 2X robust accuracy gain. Compared to classical test-time defenses, C-AVP also yields a 42X inference time speedup.

摘要: 在这项工作中，我们利用视觉提示(VP)来提高测试时固定的、预先训练的模型的对抗健壮性。与传统的对抗性防御相比，VP允许我们设计通用的(即数据不可知的)输入提示模板，该模板在测试时具有即插即用的能力，在不引入太多计算开销的情况下达到期望的模型性能。虽然VP已经被成功地应用于改进模型泛化，但它是否以及如何被用来防御对手攻击仍然是一个难以捉摸的问题。我们研究了这个问题，并证明了普通VP方法在对抗防御中不是有效的，因为通用的输入提示缺乏针对特定样本的对抗扰动的稳健学习能力。针对这一问题，我们提出了一种新的分类对抗性视觉提示生成方法--分类对抗性视觉提示(C-AVP)，该方法不仅可以利用集成提示的优点，而且可以优化它们之间的相互关系，从而提高模型的稳健性。实验表明，C-AVP比传统的VP方法有2.1倍的标准精度增益和2倍的稳健精度增益。与经典的测试时间防御相比，C-AVP的推理时间加速比也提高了42倍。



## **13. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

半监督对抗性鲁棒PAC学习性的一个刻画 cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2202.05420v2) [paper-pdf](http://arxiv.org/pdf/2202.05420v2)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.

摘要: 我们研究了在半监督PAC模型中学习对抗性稳健预测器来测试时间攻击的问题。我们解决了需要多少已标记和未标记的示例才能确保学习的问题。我们证明，有足够的未标记数据(完全监督方法所需的标记样本的大小)，标记样本的复杂度可以比以前的工作任意小，并且明显地被不同的复杂性度量所刻画。我们证明了这一样本复杂性的上下界几乎一致。这表明，即使在最坏情况下无分布的模型中，半监督稳健学习也有显著的好处，并在监督和半监督标签复杂性之间建立了差距，这在标准的非稳健PAC学习中是不存在的。



## **14. Double Bubble, Toil and Trouble: Enhancing Certified Robustness through Transitivity**

双重泡沫、辛劳和麻烦：通过传递性增强认证的健壮性 cs.LG

Accepted for Neurips`22, 19 pages, 14 figures, for associated code  see https://github.com/andrew-cullen/DoubleBubble

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.06077v1) [paper-pdf](http://arxiv.org/pdf/2210.06077v1)

**Authors**: Andrew C. Cullen, Paul Montague, Shijie Liu, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In response to subtle adversarial examples flipping classifications of neural network models, recent research has promoted certified robustness as a solution. There, invariance of predictions to all norm-bounded attacks is achieved through randomised smoothing of network inputs. Today's state-of-the-art certifications make optimal use of the class output scores at the input instance under test: no better radius of certification (under the $L_2$ norm) is possible given only these score. However, it is an open question as to whether such lower bounds can be improved using local information around the instance under test. In this work, we demonstrate how today's "optimal" certificates can be improved by exploiting both the transitivity of certifications, and the geometry of the input space, giving rise to what we term Geometrically-Informed Certified Robustness. By considering the smallest distance to points on the boundary of a set of certifications this approach improves certifications for more than $80\%$ of Tiny-Imagenet instances, yielding an on average $5 \%$ increase in the associated certification. When incorporating training time processes that enhance the certified radius, our technique shows even more promising results, with a uniform $4$ percentage point increase in the achieved certified radius.

摘要: 为了应对微妙的敌意例子颠覆神经网络模型的分类，最近的研究已经将经过认证的稳健性作为解决方案。通过对网络输入的随机平滑，实现了对所有范数有界攻击的预测不变性。当今最先进的认证最好地利用了测试中输入实例的类输出分数：如果只给出这些分数，就没有更好的认证半径(在$L_2$范数下)。然而，是否可以使用测试实例周围的本地信息来提高这种下限，这是一个悬而未决的问题。在这项工作中，我们演示了如何通过利用证书的传递性和输入空间的几何来改进当今的“最佳”证书，从而产生我们所称的几何信息的认证健壮性。通过考虑到一组证书的边界上的点的最小距离，该方法改进了超过$80$的微型Imagenet实例的证书，导致相关证书的平均增加$5\$。当加入了提高认证半径的训练时间过程时，我们的技术显示出更有希望的结果，所获得的认证半径统一增加了$4$百分点。



## **15. SA: Sliding attack for synthetic speech detection with resistance to clipping and self-splicing**

SA：抗剪裁和自拼接的合成语音检测滑动攻击 cs.SD

Updated description and formula

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2208.13066v2) [paper-pdf](http://arxiv.org/pdf/2208.13066v2)

**Authors**: Deng JiaCheng, Dong Li, Yan Diqun, Wang Rangding, Zeng Jiaming

**Abstract**: Deep neural networks are vulnerable to adversarial examples that mislead models with imperceptible perturbations. In audio, although adversarial examples have achieved incredible attack success rates on white-box settings and black-box settings, most existing adversarial attacks are constrained by the input length. A More practical scenario is that the adversarial examples must be clipped or self-spliced and input into the black-box model. Therefore, it is necessary to explore how to improve transferability in different input length settings. In this paper, we take the synthetic speech detection task as an example and consider two representative SOTA models. We observe that the gradients of fragments with the same sample value are similar in different models via analyzing the gradients obtained by feeding samples into the model after cropping or self-splicing. Inspired by the above observation, we propose a new adversarial attack method termed sliding attack. Specifically, we make each sampling point aware of gradients at different locations, which can simulate the situation where adversarial examples are input to black-box models with varying input lengths. Therefore, instead of using the current gradient directly in each iteration of the gradient calculation, we go through the following three steps. First, we extract subsegments of different lengths using sliding windows. We then augment the subsegments with data from the adjacent domains. Finally, we feed the sub-segments into different models to obtain aggregate gradients to update adversarial examples. Empirical results demonstrate that our method could significantly improve the transferability of adversarial examples after clipping or self-splicing. Besides, our method could also enhance the transferability between models based on different features.

摘要: 深度神经网络很容易受到敌意例子的影响，这些例子用无法察觉的扰动误导模型。在音频方面，虽然对抗性例子在白盒和黑盒设置上取得了令人难以置信的攻击成功率，但大多数现有的对抗性攻击都受到输入长度的限制。一个更实际的场景是，对抗性的例子必须被剪裁或自我拼接，并输入到黑盒模型中。因此，有必要探索如何在不同的输入长度设置下提高可转移性。在本文中，我们以合成语音检测任务为例，考虑了两个具有代表性的SOTA模型。通过分析剪裁或自剪接后将样本送入模型获得的梯度，我们观察到相同样本值的片段在不同模型中的梯度是相似的。受此启发，我们提出了一种新的对抗性攻击方法--滑动攻击。具体地说，我们使每个采样点知道不同位置的梯度，这可以模拟对抗性例子被输入到具有不同输入长度的黑盒模型的情况。因此，我们不是在梯度计算的每次迭代中直接使用当前梯度，而是经历以下三个步骤。首先，我们使用滑动窗口提取不同长度的子段。然后，我们使用来自相邻域的数据来增强子分段。最后，我们将子片段输入到不同的模型中，以获得聚合梯度来更新对抗性实例。实验结果表明，该方法能够显著提高截取或自拼接后的对抗性样本的可转移性。此外，我们的方法还可以增强基于不同特征的模型之间的可移植性。



## **16. Boosting the Transferability of Adversarial Attacks with Reverse Adversarial Perturbation**

利用反向对抗性扰动提高对抗性攻击的可转移性 cs.CV

NeurIPS 2022 conference paper

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05968v1) [paper-pdf](http://arxiv.org/pdf/2210.05968v1)

**Authors**: Zeyu Qin, Yanbo Fan, Yi Liu, Li Shen, Yong Zhang, Jue Wang, Baoyuan Wu

**Abstract**: Deep neural networks (DNNs) have been shown to be vulnerable to adversarial examples, which can produce erroneous predictions by injecting imperceptible perturbations. In this work, we study the transferability of adversarial examples, which is significant due to its threat to real-world applications where model architecture or parameters are usually unknown. Many existing works reveal that the adversarial examples are likely to overfit the surrogate model that they are generated from, limiting its transfer attack performance against different target models. To mitigate the overfitting of the surrogate model, we propose a novel attack method, dubbed reverse adversarial perturbation (RAP). Specifically, instead of minimizing the loss of a single adversarial point, we advocate seeking adversarial example located at a region with unified low loss value, by injecting the worst-case perturbation (the reverse adversarial perturbation) for each step of the optimization procedure. The adversarial attack with RAP is formulated as a min-max bi-level optimization problem. By integrating RAP into the iterative process for attacks, our method can find more stable adversarial examples which are less sensitive to the changes of decision boundary, mitigating the overfitting of the surrogate model. Comprehensive experimental comparisons demonstrate that RAP can significantly boost adversarial transferability. Furthermore, RAP can be naturally combined with many existing black-box attack techniques, to further boost the transferability. When attacking a real-world image recognition system, Google Cloud Vision API, we obtain 22% performance improvement of targeted attacks over the compared method. Our codes are available at https://github.com/SCLBD/Transfer_attack_RAP.

摘要: 深度神经网络(DNN)已被证明容易受到敌意例子的攻击，这些例子通过注入不可察觉的扰动而产生错误的预测。在这项工作中，我们研究了对抗性例子的可转移性，这一点很重要，因为它对模型结构或参数通常未知的现实世界应用程序构成了威胁。许多已有的工作表明，敌意示例可能会过度匹配生成它们的代理模型，从而限制了其对不同目标模型的传输攻击性能。为了缓解代理模型的过度拟合，我们提出了一种新的攻击方法，称为反向对抗扰动(RAP)。具体地说，我们主张通过为优化过程的每一步注入最坏情况的扰动(反向对抗性扰动)来寻找位于具有统一低损失值的区域的对抗性实例，而不是最小化单个对抗点的损失。基于RAP的对抗性攻击被描述为一个最小-最大双层优化问题。通过将RAP集成到攻击的迭代过程中，我们的方法可以找到更稳定的对抗性实例，这些实例对决策边界的变化不那么敏感，从而缓解了代理模型的过度拟合问题。综合实验比较表明，RAP能够显著提高对抗性转移能力。此外，RAP可以自然地与许多现有的黑盒攻击技术相结合，进一步提高可转移性。在攻击真实世界的图像识别系统Google Cloud Vision API时，与比较的方法相比，我们获得了22%的定向攻击性能提升。我们的代码可在https://github.com/SCLBD/Transfer_attack_RAP.上获得



## **17. Robust Models are less Over-Confident**

稳健的模型不那么过度自信 cs.CV

accepted at NeuRips 2022

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05938v1) [paper-pdf](http://arxiv.org/pdf/2210.05938v1)

**Authors**: Julia Grabinski, Paul Gavrikov, Janis Keuper, Margret Keuper

**Abstract**: Despite the success of convolutional neural networks (CNNs) in many academic benchmarks for computer vision tasks, their application in the real-world is still facing fundamental challenges. One of these open problems is the inherent lack of robustness, unveiled by the striking effectiveness of adversarial attacks. Current attack methods are able to manipulate the network's prediction by adding specific but small amounts of noise to the input. In turn, adversarial training (AT) aims to achieve robustness against such attacks and ideally a better model generalization ability by including adversarial samples in the trainingset. However, an in-depth analysis of the resulting robust models beyond adversarial robustness is still pending. In this paper, we empirically analyze a variety of adversarially trained models that achieve high robust accuracies when facing state-of-the-art attacks and we show that AT has an interesting side-effect: it leads to models that are significantly less overconfident with their decisions, even on clean data than non-robust models. Further, our analysis of robust models shows that not only AT but also the model's building blocks (like activation functions and pooling) have a strong influence on the models' prediction confidences. Data & Project website: https://github.com/GeJulia/robustness_confidences_evaluation

摘要: 尽管卷积神经网络(CNN)在许多计算机视觉任务的学术基准中取得了成功，但它们在现实世界中的应用仍然面临着根本性的挑战。这些悬而未决的问题之一是固有的健壮性不足，这一点从对抗性攻击的惊人有效性中可见一斑。目前的攻击方法能够通过向输入添加特定但少量的噪声来操纵网络的预测。反过来，对抗性训练(AT)的目的是通过将对抗性样本包括在训练集中来实现对此类攻击的健壮性，并且理想地实现更好的模型泛化能力。然而，对由此产生的超越对抗性稳健性的稳健性模型的深入分析仍然悬而未决。在这篇文章中，我们实证分析了各种对抗训练的模型，这些模型在面对最先进的攻击时获得了很高的稳健精度，我们发现AT有一个有趣的副作用：它导致模型对他们的决策不那么过度自信，即使是在干净的数据上也是如此。此外，我们对稳健模型的分析表明，不仅AT而且模型的构件(如激活函数和池化)对模型的预测置信度有很大的影响。数据与项目网站：https://github.com/GeJulia/robustness_confidences_evaluation



## **18. Efficient Adversarial Training without Attacking: Worst-Case-Aware Robust Reinforcement Learning**

无攻击的高效对抗性训练：最坏情况感知的稳健强化学习 cs.LG

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-12    [abs](http://arxiv.org/abs/2210.05927v1) [paper-pdf](http://arxiv.org/pdf/2210.05927v1)

**Authors**: Yongyuan Liang, Yanchao Sun, Ruijie Zheng, Furong Huang

**Abstract**: Recent studies reveal that a well-trained deep reinforcement learning (RL) policy can be particularly vulnerable to adversarial perturbations on input observations. Therefore, it is crucial to train RL agents that are robust against any attacks with a bounded budget. Existing robust training methods in deep RL either treat correlated steps separately, ignoring the robustness of long-term rewards, or train the agents and RL-based attacker together, doubling the computational burden and sample complexity of the training process. In this work, we propose a strong and efficient robust training framework for RL, named Worst-case-aware Robust RL (WocaR-RL) that directly estimates and optimizes the worst-case reward of a policy under bounded l_p attacks without requiring extra samples for learning an attacker. Experiments on multiple environments show that WocaR-RL achieves state-of-the-art performance under various strong attacks, and obtains significantly higher training efficiency than prior state-of-the-art robust training methods. The code of this work is available at https://github.com/umd-huang-lab/WocaR-RL.

摘要: 最近的研究表明，训练有素的深度强化学习(RL)策略特别容易受到输入观测的对抗性扰动。因此，在有限的预算内培训对任何攻击具有健壮性的RL代理是至关重要的。现有的深度RL稳健训练方法要么单独处理相关步骤，忽略长期回报的稳健性，要么将代理和基于RL的攻击者一起训练，使训练过程的计算负担和样本复杂度翻了一番。在这项工作中，我们提出了一种强而有效的RL稳健训练框架，称为最坏情况感知稳健RL(WocaR-RL)，它可以直接估计和优化策略在有界l_p攻击下的最坏情况奖励，而不需要额外的样本来学习攻击者。在多种环境下的实验表明，WocaR-RL在各种强攻击下具有最好的性能，训练效率明显高于现有的健壮训练方法。这项工作的代码可以在https://github.com/umd-huang-lab/WocaR-RL.上找到



## **19. On the Limitations of Stochastic Pre-processing Defenses**

论随机前处理防御的局限性 cs.LG

Accepted by Proceedings of the 36th Conference on Neural Information  Processing Systems

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.09491v3) [paper-pdf](http://arxiv.org/pdf/2206.09491v3)

**Authors**: Yue Gao, Ilia Shumailov, Kassem Fawaz, Nicolas Papernot

**Abstract**: Defending against adversarial examples remains an open problem. A common belief is that randomness at inference increases the cost of finding adversarial inputs. An example of such a defense is to apply a random transformation to inputs prior to feeding them to the model. In this paper, we empirically and theoretically investigate such stochastic pre-processing defenses and demonstrate that they are flawed. First, we show that most stochastic defenses are weaker than previously thought; they lack sufficient randomness to withstand even standard attacks like projected gradient descent. This casts doubt on a long-held assumption that stochastic defenses invalidate attacks designed to evade deterministic defenses and force attackers to integrate the Expectation over Transformation (EOT) concept. Second, we show that stochastic defenses confront a trade-off between adversarial robustness and model invariance; they become less effective as the defended model acquires more invariance to their randomization. Future work will need to decouple these two effects. We also discuss implications and guidance for future research.

摘要: 抵御敌意的例子仍然是一个悬而未决的问题。一种普遍的看法是，推理的随机性增加了寻找敌对输入的成本。这种防御的一个例子是在将输入提供给模型之前对它们应用随机转换。在本文中，我们从经验和理论上研究了这种随机预处理防御机制，并证明了它们是有缺陷的。首先，我们证明了大多数随机防御比之前认为的要弱；它们缺乏足够的随机性，即使是像投影梯度下降这样的标准攻击也是如此。这让人对一个长期持有的假设产生了怀疑，即随机防御使旨在逃避确定性防御的攻击无效，并迫使攻击者整合期望过转换(EOT)概念。其次，我们证明了随机防御面临着对抗稳健性和模型不变性之间的权衡；随着被防御模型对其随机化获得更多的不变性，它们变得不那么有效。未来的工作将需要将这两种影响脱钩。我们还讨论了对未来研究的启示和指导。



## **20. Adversarial Attack Against Image-Based Localization Neural Networks**

基于图像的定位神经网络的对抗性攻击 cs.CV

13 pages, 10 figures

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.06589v1) [paper-pdf](http://arxiv.org/pdf/2210.06589v1)

**Authors**: Meir Brand, Itay Naeh, Daniel Teitelman

**Abstract**: In this paper, we present a proof of concept for adversarially attacking the image-based localization module of an autonomous vehicle. This attack aims to cause the vehicle to perform a wrong navigational decisions and prevent it from reaching a desired predefined destination in a simulated urban environment. A database of rendered images allowed us to train a deep neural network that performs a localization task and implement, develop and assess the adversarial pattern. Our tests show that using this adversarial attack we can prevent the vehicle from turning at a given intersection. This is done by manipulating the vehicle's navigational module to falsely estimate its current position and thus fail to initialize the turning procedure until the vehicle misses the last opportunity to perform a safe turn in a given intersection.

摘要: 在这篇文章中，我们提出了一种概念证明，用于恶意攻击自主车辆的基于图像的定位模块。这种攻击旨在导致车辆执行错误的导航决策，并阻止其在模拟城市环境中到达预期的预定目的地。渲染图像的数据库使我们能够训练执行定位任务的深度神经网络，并实施、开发和评估对抗性模式。我们的测试表明，使用这种对抗性攻击，我们可以防止车辆在给定的十字路口转弯。这是通过操纵车辆的导航模块错误地估计其当前位置，从而无法初始化转弯程序，直到车辆错过在给定十字路口执行安全转弯的最后机会来实现的。



## **21. Indicators of Attack Failure: Debugging and Improving Optimization of Adversarial Examples**

攻击失败的指标：对抗性实例的调试和改进优化 cs.LG

Accepted at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2106.09947v3) [paper-pdf](http://arxiv.org/pdf/2106.09947v3)

**Authors**: Maura Pintor, Luca Demetrio, Angelo Sotgiu, Ambra Demontis, Nicholas Carlini, Battista Biggio, Fabio Roli

**Abstract**: Evaluating robustness of machine-learning models to adversarial examples is a challenging problem. Many defenses have been shown to provide a false sense of robustness by causing gradient-based attacks to fail, and they have been broken under more rigorous evaluations. Although guidelines and best practices have been suggested to improve current adversarial robustness evaluations, the lack of automatic testing and debugging tools makes it difficult to apply these recommendations in a systematic manner. In this work, we overcome these limitations by: (i) categorizing attack failures based on how they affect the optimization of gradient-based attacks, while also unveiling two novel failures affecting many popular attack implementations and past evaluations; (ii) proposing six novel indicators of failure, to automatically detect the presence of such failures in the attack optimization process; and (iii) suggesting a systematic protocol to apply the corresponding fixes. Our extensive experimental analysis, involving more than 15 models in 3 distinct application domains, shows that our indicators of failure can be used to debug and improve current adversarial robustness evaluations, thereby providing a first concrete step towards automatizing and systematizing them. Our open-source code is available at: https://github.com/pralab/IndicatorsOfAttackFailure.

摘要: 评估机器学习模型对对抗性样本的稳健性是一个具有挑战性的问题。事实证明，许多防御措施通过导致基于梯度的攻击失败来提供一种错误的健壮感，这些防御措施已经在更严格的评估下被打破。虽然有人建议采用准则和最佳做法来改进目前的对抗性评估，但由于缺乏自动测试和调试工具，很难系统地适用这些建议。在这项工作中，我们克服了这些局限性：(I)根据攻击失败如何影响基于梯度的攻击的优化进行分类，同时也揭示了影响许多流行攻击实现和过去评估的两个新失败；(Ii)提出了六个新的失败指示器，以自动检测攻击优化过程中此类失败的存在；以及(Iii)提出了一个系统的协议来应用相应的修复。我们广泛的实验分析，涉及3个不同应用领域的15个模型，表明我们的失败指示器可以用于调试和改进当前的对手健壮性评估，从而为实现自动化和系统化迈出了具体的第一步。我们的开源代码可以在https://github.com/pralab/IndicatorsOfAttackFailure.上找到



## **22. Stable and Efficient Adversarial Training through Local Linearization**

基于局部线性化的稳定高效的对战训练 cs.LG

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05373v1) [paper-pdf](http://arxiv.org/pdf/2210.05373v1)

**Authors**: Zhuorong Li, Daiwei Yu

**Abstract**: There has been a recent surge in single-step adversarial training as it shows robustness and efficiency. However, a phenomenon referred to as ``catastrophic overfitting" has been observed, which is prevalent in single-step defenses and may frustrate attempts to use FGSM adversarial training. To address this issue, we propose a novel method, Stable and Efficient Adversarial Training (SEAT), which mitigates catastrophic overfitting by harnessing on local properties that distinguish a robust model from that of a catastrophic overfitted model. The proposed SEAT has strong theoretical justifications, in that minimizing the SEAT loss can be shown to favour smooth empirical risk, thereby leading to robustness. Experimental results demonstrate that the proposed method successfully mitigates catastrophic overfitting, yielding superior performance amongst efficient defenses. Our single-step method can reach 51% robust accuracy for CIFAR-10 with $l_\infty$ perturbations of radius $8/255$ under a strong PGD-50 attack, matching the performance of a 10-step iterative adversarial training at merely 3% computational cost.

摘要: 最近，单步对抗性训练出现了激增，因为它显示出健壮性和效率。然而，已经观察到一种被称为“灾难性过配”的现象，这种现象在单步防御中很普遍，可能会挫败使用FGSM对抗性训练的尝试。为了解决这个问题，我们提出了一种新的方法，稳定有效的对抗性训练(SEAT)，它通过利用局部性质来区分稳健模型和灾难性过拟合模型，从而减轻灾难性过拟合。拟议的席位有很强的理论依据，因为最大限度地减少席位损失可以证明有利于平稳的经验风险，从而导致稳健性。实验结果表明，提出的方法成功地缓解了灾难性的过拟合，在有效的防御中产生了优越的性能。在强PGD-50攻击下，对于半径为$8/255$摄动的CIFAR-10，我们的单步方法可以达到51%的稳健准确率，与10步迭代对抗性训练的性能相当，而计算代价仅为3%。



## **23. Adversarial Robustness of Deep Neural Networks: A Survey from a Formal Verification Perspective**

深度神经网络的对抗健壮性：从形式验证的角度综述 cs.CR

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2206.12227v2) [paper-pdf](http://arxiv.org/pdf/2206.12227v2)

**Authors**: Mark Huasong Meng, Guangdong Bai, Sin Gee Teo, Zhe Hou, Yan Xiao, Yun Lin, Jin Song Dong

**Abstract**: Neural networks have been widely applied in security applications such as spam and phishing detection, intrusion prevention, and malware detection. This black-box method, however, often has uncertainty and poor explainability in applications. Furthermore, neural networks themselves are often vulnerable to adversarial attacks. For those reasons, there is a high demand for trustworthy and rigorous methods to verify the robustness of neural network models. Adversarial robustness, which concerns the reliability of a neural network when dealing with maliciously manipulated inputs, is one of the hottest topics in security and machine learning. In this work, we survey existing literature in adversarial robustness verification for neural networks and collect 39 diversified research works across machine learning, security, and software engineering domains. We systematically analyze their approaches, including how robustness is formulated, what verification techniques are used, and the strengths and limitations of each technique. We provide a taxonomy from a formal verification perspective for a comprehensive understanding of this topic. We classify the existing techniques based on property specification, problem reduction, and reasoning strategies. We also demonstrate representative techniques that have been applied in existing studies with a sample model. Finally, we discuss open questions for future research.

摘要: 神经网络已广泛应用于垃圾邮件和网络钓鱼检测、入侵防御和恶意软件检测等安全应用中。然而，这种黑箱方法在应用中往往具有不确定性和较差的可解释性。此外，神经网络本身往往容易受到敌意攻击。因此，对神经网络模型的稳健性验证方法提出了更高的要求。对抗健壮性是安全和机器学习领域中最热门的话题之一，它涉及到神经网络在处理恶意操作的输入时的可靠性。在这项工作中，我们综述了现有的神经网络对抗健壮性验证的文献，并收集了39个不同的研究工作，涉及机器学习、安全和软件工程领域。我们系统地分析了他们的方法，包括健壮性是如何形成的，使用了什么验证技术，以及每种技术的优点和局限性。为了全面理解这一主题，我们从正式验证的角度提供了一个分类法。我们根据属性规范、问题约简和推理策略对现有技术进行分类。我们还用一个样本模型演示了已在现有研究中应用的代表性技术。最后，我们讨论了未来研究的有待解决的问题。



## **24. Zeroth-Order Hard-Thresholding: Gradient Error vs. Expansivity**

零阶硬阈值：梯度误差与扩张性 cs.LG

Accepted for publication at NeurIPS 2022

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05279v1) [paper-pdf](http://arxiv.org/pdf/2210.05279v1)

**Authors**: William de Vazelhes, Hualin Zhang, Huimin Wu, Xiao-Tong Yuan, Bin Gu

**Abstract**: $\ell_0$ constrained optimization is prevalent in machine learning, particularly for high-dimensional problems, because it is a fundamental approach to achieve sparse learning. Hard-thresholding gradient descent is a dominant technique to solve this problem. However, first-order gradients of the objective function may be either unavailable or expensive to calculate in a lot of real-world problems, where zeroth-order (ZO) gradients could be a good surrogate. Unfortunately, whether ZO gradients can work with the hard-thresholding operator is still an unsolved problem. To solve this puzzle, in this paper, we focus on the $\ell_0$ constrained black-box stochastic optimization problems, and propose a new stochastic zeroth-order gradient hard-thresholding (SZOHT) algorithm with a general ZO gradient estimator powered by a novel random support sampling. We provide the convergence analysis of SZOHT under standard assumptions. Importantly, we reveal a conflict between the deviation of ZO estimators and the expansivity of the hard-thresholding operator, and provide a theoretical minimal value of the number of random directions in ZO gradients. In addition, we find that the query complexity of SZOHT is independent or weakly dependent on the dimensionality under different settings. Finally, we illustrate the utility of our method on a portfolio optimization problem as well as black-box adversarial attacks.

摘要: 约束优化是实现稀疏学习的基本途径，在机器学习中得到了广泛的应用，特别是对于高维问题。硬阈值梯度下降是解决这一问题的主流技术。然而，在许多实际问题中，目标函数的一阶梯度可能无法获得或计算成本很高，其中零阶(ZO)梯度可能是一个很好的替代。遗憾的是，ZO梯度是否能与硬阈值算子一起工作，仍然是一个悬而未决的问题。为解决这一难题，本文以0元约束黑箱随机优化问题为研究对象，提出了一种新的随机零阶梯度硬阈值算法(SZOHT)，该算法采用了一种新的随机支持抽样的广义ZO梯度估值器。在标准假设下，给出了SZOHT算法的收敛分析。重要的是，我们揭示了ZO估计器的偏差与硬阈值算子的可扩性之间的冲突，并给出了ZO梯度中随机方向数的理论最小值。此外，我们还发现，在不同的设置下，SZOHT的查询复杂度与维度无关或弱依赖。最后，我们说明了我们的方法在一个投资组合优化问题和黑箱对抗攻击中的实用性。



## **25. RoHNAS: A Neural Architecture Search Framework with Conjoint Optimization for Adversarial Robustness and Hardware Efficiency of Convolutional and Capsule Networks**

RoHNAS：一种联合优化卷积网络和胶囊网络对抗健壮性和硬件效率的神经结构搜索框架 cs.LG

Accepted for publication at IEEE Access

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2210.05276v1) [paper-pdf](http://arxiv.org/pdf/2210.05276v1)

**Authors**: Alberto Marchisio, Vojtech Mrazek, Andrea Massa, Beatrice Bussolino, Maurizio Martina, Muhammad Shafique

**Abstract**: Neural Architecture Search (NAS) algorithms aim at finding efficient Deep Neural Network (DNN) architectures for a given application under given system constraints. DNNs are computationally-complex as well as vulnerable to adversarial attacks. In order to address multiple design objectives, we propose RoHNAS, a novel NAS framework that jointly optimizes for adversarial-robustness and hardware-efficiency of DNNs executed on specialized hardware accelerators. Besides the traditional convolutional DNNs, RoHNAS additionally accounts for complex types of DNNs such as Capsule Networks. For reducing the exploration time, RoHNAS analyzes and selects appropriate values of adversarial perturbation for each dataset to employ in the NAS flow. Extensive evaluations on multi - Graphics Processing Unit (GPU) - High Performance Computing (HPC) nodes provide a set of Pareto-optimal solutions, leveraging the tradeoff between the above-discussed design objectives. For example, a Pareto-optimal DNN for the CIFAR-10 dataset exhibits 86.07% accuracy, while having an energy of 38.63 mJ, a memory footprint of 11.85 MiB, and a latency of 4.47 ms.

摘要: 神经结构搜索(NAS)算法的目标是在给定的系统约束下为给定的应用找到有效的深度神经网络(DNN)结构。DNN在计算上很复杂，而且容易受到对手攻击。为了解决多个设计目标，我们提出了RoHNAS，这是一种新的NAS框架，它联合优化了在专用硬件加速器上执行的DNN的对抗性健壮性和硬件效率。除了传统的卷积DNN，RoHNAS还考虑了复杂类型的DNN，如胶囊网络。为了减少探测时间，RoHNAS为每个数据集分析并选择适当的对抗性扰动值以用于NAS流。对多图形处理器(GPU)-高性能计算(HPC)节点的广泛评估提供了一组帕累托最优解决方案，充分利用了上述设计目标之间的权衡。例如，对于CIFAR-10数据集，Pareto最优DNN的准确率为86.07%，而能量为38.63 MJ，内存占用量为11.85 MiB，延迟为4.47 ms。



## **26. Towards Lightweight Black-Box Attacks against Deep Neural Networks**

面向深度神经网络的轻量级黑盒攻击 cs.LG

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2209.14826v3) [paper-pdf](http://arxiv.org/pdf/2209.14826v3)

**Authors**: Chenghao Sun, Yonggang Zhang, Wan Chaoqun, Qizhou Wang, Ya Li, Tongliang Liu, Bo Han, Xinmei Tian

**Abstract**: Black-box attacks can generate adversarial examples without accessing the parameters of target model, largely exacerbating the threats of deployed deep neural networks (DNNs). However, previous works state that black-box attacks fail to mislead target models when their training data and outputs are inaccessible. In this work, we argue that black-box attacks can pose practical attacks in this extremely restrictive scenario where only several test samples are available. Specifically, we find that attacking the shallow layers of DNNs trained on a few test samples can generate powerful adversarial examples. As only a few samples are required, we refer to these attacks as lightweight black-box attacks. The main challenge to promoting lightweight attacks is to mitigate the adverse impact caused by the approximation error of shallow layers. As it is hard to mitigate the approximation error with few available samples, we propose Error TransFormer (ETF) for lightweight attacks. Namely, ETF transforms the approximation error in the parameter space into a perturbation in the feature space and alleviates the error by disturbing features. In experiments, lightweight black-box attacks with the proposed ETF achieve surprising results. For example, even if only 1 sample per category available, the attack success rate in lightweight black-box attacks is only about 3% lower than that of the black-box attacks with complete training data.

摘要: 黑盒攻击可以在不访问目标模型参数的情况下生成敌意示例，从而在很大程度上加剧了已部署的深度神经网络(DNN)的威胁。然而，以前的工作指出，当目标模型的训练数据和输出不可访问时，黑盒攻击无法误导目标模型。在这项工作中，我们认为黑盒攻击可以在这种极端限制性的场景中构成实际攻击，其中只有几个测试样本可用。具体地说，我们发现，攻击在几个测试样本上训练的DNN的浅层可以产生强大的对抗性例子。由于只需要几个样本，我们将这些攻击称为轻量级黑盒攻击。推广轻量级攻击的主要挑战是缓解浅层近似误差造成的不利影响。针对现有样本较少难以消除近似误差的问题，提出了一种用于轻量级攻击的误差转换器(ETF)。也就是说，ETF将参数空间中的逼近误差转化为特征空间中的扰动，并通过扰动特征来减轻误差。在实验中，使用提出的ETF进行的轻量级黑盒攻击取得了令人惊讶的结果。例如，即使每个类别只有1个样本，轻量级黑盒攻击的攻击成功率也只比拥有完整训练数据的黑盒攻击低3%左右。



## **27. Content-Adaptive Pixel Discretization to Improve Model Robustness**

改进模型稳健性的内容自适应像素离散化 cs.CV

**SubmitDate**: 2022-10-11    [abs](http://arxiv.org/abs/2012.01699v4) [paper-pdf](http://arxiv.org/pdf/2012.01699v4)

**Authors**: Ryan Feng, Wu-chi Feng, Atul Prakash

**Abstract**: Preprocessing defenses such as pixel discretization are appealing to remove adversarial attacks due to their simplicity. However, they have been shown to be ineffective except on simple datasets like MNIST. We hypothesize that existing discretization approaches failed because using a fixed codebook for the entire dataset limits their ability to balance image representation and codeword separability. We first formally prove that adaptive codebooks can provide stronger robustness guarantees than fixed codebooks as a preprocessing defense on some datasets. Based on that insight, we propose a content-adaptive pixel discretization defense called Essential Features, which discretizes the image to a per-image adaptive codebook to reduce the color space. We then find that Essential Features can be further optimized by applying adaptive blurring before the discretization to push perturbed pixel values back to their original value before determining the codebook. Against adaptive attacks, we show that content-adaptive pixel discretization extends the range of datasets that benefit in terms of both L_2 and L_infinity robustness where previously fixed codebooks were found to have failed. Our findings suggest that content-adaptive pixel discretization should be part of the repertoire for making models robust.

摘要: 像像素离散化这样的预处理防御由于其简单性而被用来消除对抗性攻击。然而，它们已经被证明是无效的，除非在MNIST这样的简单数据集上。我们假设现有的离散化方法失败了，因为对整个数据集使用固定的码本限制了它们平衡图像表示和码字可分性的能力。我们首先形式化地证明了自适应码本可以提供比固定码本更强的稳健性保证，作为对某些数据集的预处理防御。基于这一观点，我们提出了一种称为基本特征的内容自适应像素离散化防御方法，它将图像离散化为每幅图像的自适应码本，以减少颜色空间。然后我们发现，通过在离散化之前应用自适应模糊，在确定码本之前将扰动的像素值推回到其原始值，可以进一步优化基本特征。对于自适应攻击，我们证明了内容自适应像素离散化扩展了数据集的范围，在先前固定的码本被发现失败的情况下，数据集在L2和L_infinity健壮性方面都受益。我们的发现表明，内容自适应像素离散化应该是使模型健壮的一部分。



## **28. Spinning Sequence-to-Sequence Models with Meta-Backdoors**

带元后门的旋转序列到序列模型 cs.CR

Outdated. Superseded by arXiv:2112.05224 and published at IEEE S&P'22  with title: "Spinning Language Models: Risks of Propaganda-As-A-Service and  Countermeasures"

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2107.10443v2) [paper-pdf](http://arxiv.org/pdf/2107.10443v2)

**Authors**: Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: We investigate a new threat to neural sequence-to-sequence (seq2seq) models: training-time attacks that cause models to "spin" their output and support a certain sentiment when the input contains adversary-chosen trigger words. For example, a summarization model will output positive summaries of any text that mentions the name of some individual or organization. We introduce the concept of a "meta-backdoor" to explain model-spinning attacks. These attacks produce models whose output is valid and preserves context, yet also satisfies a meta-task chosen by the adversary (e.g., positive sentiment). Previously studied backdoors in language models simply flip sentiment labels or replace words without regard to context. Their outputs are incorrect on inputs with the trigger. Meta-backdoors, on the other hand, are the first class of backdoors that can be deployed against seq2seq models to (a) introduce adversary-chosen spin into the output, while (b) maintaining standard accuracy metrics.   To demonstrate feasibility of model spinning, we develop a new backdooring technique. It stacks the adversarial meta-task (e.g., sentiment analysis) onto a seq2seq model, backpropagates the desired meta-task output (e.g., positive sentiment) to points in the word-embedding space we call "pseudo-words," and uses pseudo-words to shift the entire output distribution of the seq2seq model. Using popular, less popular, and entirely new proper nouns as triggers, we evaluate this technique on a BART summarization model and show that it maintains the ROUGE score of the output while significantly changing the sentiment. We explain why model spinning can be a dangerous technique in AI-powered disinformation and discuss how to mitigate these attacks.

摘要: 我们研究了神经序列到序列(Seq2seq)模型的一种新威胁：训练时间攻击，当输入包含对手选择的触发词时，这种攻击会导致模型的输出“旋转”，并支持特定的情感。例如，摘要模型将输出提及某个个人或组织名称的任何文本的正面摘要。我们引入了“元后门”的概念来解释模型旋转攻击。这些攻击产生的模型的输出是有效的，保留了上下文，但也满足了对手选择的元任务(例如，积极的情绪)。以前在语言模型中研究的后门只是简单地翻转情感标签或替换单词，而不考虑上下文。它们的输出在带有触发器的输入上不正确。另一方面，元后门是第一类可以针对seq2seq模型部署的后门，以(A)在输出中引入对手选择的旋转，同时(B)维护标准精度度量。为了论证模型旋转的可行性，我们开发了一种新的回溯技术。它将敌意元任务(例如，情绪分析)堆叠到seq2seq模型上，将期望的元任务输出(例如，积极的情绪)反向传播到我们称为“伪词”的词嵌入空间中的点，并使用伪词来移位seq2seq模型的整个输出分布。使用流行的、不太流行的和全新的专有名词作为触发器，我们在BART摘要模型上评估了这一技术，结果表明它保持了输出的Rouge分数，同时显著改变了情绪。我们解释了为什么模型旋转在人工智能支持的虚假信息中可能是一种危险的技术，并讨论了如何减轻这些攻击。



## **29. Towards Out-of-Distribution Adversarial Robustness**

向分布外对手稳健性迈进 cs.LG

Under review ICLR 2023

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2210.03150v2) [paper-pdf](http://arxiv.org/pdf/2210.03150v2)

**Authors**: Adam Ibrahim, Charles Guille-Escuret, Ioannis Mitliagkas, Irina Rish, David Krueger, Pouya Bashivan

**Abstract**: Adversarial robustness continues to be a major challenge for deep learning. A core issue is that robustness to one type of attack often fails to transfer to other attacks. While prior work establishes a theoretical trade-off in robustness against different $L_p$ norms, we show that there is potential for improvement against many commonly used attacks by adopting a domain generalisation approach. Concretely, we treat each type of attack as a domain, and apply the Risk Extrapolation method (REx), which promotes similar levels of robustness against all training attacks. Compared to existing methods, we obtain similar or superior worst-case adversarial robustness on attacks seen during training. Moreover, we achieve superior performance on families or tunings of attacks only encountered at test time. On ensembles of attacks, our approach improves the accuracy from 3.4% the best existing baseline to 25.9% on MNIST, and from 16.9% to 23.5% on CIFAR10.

摘要: 对抗的稳健性仍然是深度学习的主要挑战。一个核心问题是，对一种攻击的稳健性往往无法转移到其他攻击上。虽然以前的工作建立了对不同$L_p$范数的稳健性的理论权衡，但我们证明了通过采用域泛化方法来改进对许多常用攻击的潜力。具体地说，我们将每种类型的攻击视为一个域，并应用风险外推方法(REX)，该方法提高了对所有训练攻击的类似健壮性。与现有方法相比，对于训练过程中看到的攻击，我们获得了类似或更好的最坏情况下的对抗鲁棒性。此外，我们在仅在测试时遇到的攻击的家庭或调谐上实现了卓越的性能。在攻击集合上，我们的方法在MNIST上将准确率从3.4%提高到25.9%，在CIFAR10上从16.9%提高到23.5%。



## **30. Sampling without Replacement Leads to Faster Rates in Finite-Sum Minimax Optimization**

有限和极小极大优化中无替换抽样的快速算法 math.OC

36th Conference on Neural Information Processing Systems (NeurIPS  2022)

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2206.02953v2) [paper-pdf](http://arxiv.org/pdf/2206.02953v2)

**Authors**: Aniket Das, Bernhard Schölkopf, Michael Muehlebach

**Abstract**: We analyze the convergence rates of stochastic gradient algorithms for smooth finite-sum minimax optimization and show that, for many such algorithms, sampling the data points without replacement leads to faster convergence compared to sampling with replacement. For the smooth and strongly convex-strongly concave setting, we consider gradient descent ascent and the proximal point method, and present a unified analysis of two popular without-replacement sampling strategies, namely Random Reshuffling (RR), which shuffles the data every epoch, and Single Shuffling or Shuffle Once (SO), which shuffles only at the beginning. We obtain tight convergence rates for RR and SO and demonstrate that these strategies lead to faster convergence than uniform sampling. Moving beyond convexity, we obtain similar results for smooth nonconvex-nonconcave objectives satisfying a two-sided Polyak-{\L}ojasiewicz inequality. Finally, we demonstrate that our techniques are general enough to analyze the effect of data-ordering attacks, where an adversary manipulates the order in which data points are supplied to the optimizer. Our analysis also recovers tight rates for the incremental gradient method, where the data points are not shuffled at all.

摘要: 我们分析了光滑有限和极大极小优化问题的随机梯度算法的收敛速度，并证明了对于许多这类算法，对数据点进行不替换采样比用替换采样可以更快地收敛。对于光滑和强凸-强凹的情况，我们考虑了梯度下降上升和近似点方法，并对两种流行的无替换抽样策略进行了统一的分析，即随机重洗(RR)和单次洗牌(SO)。随机重洗(RR)是每一个时期都要洗牌的抽样策略，而单次洗牌(SO)是只在开始洗牌的抽样策略。我们得到了RR和SO的紧收敛速度，并证明了这两种策略比均匀抽样的收敛速度更快。超越凸性，我们得到了满足双边Polyak-L ojasiewicz不等式的光滑非凸-非凹目标的类似结果。最后，我们演示了我们的技术足够通用，可以分析数据排序攻击的影响，在这种攻击中，对手操纵向优化器提供数据点的顺序。我们的分析还恢复了增量梯度法的紧缩率，在这种方法中，数据点根本没有被洗牌。



## **31. On The Robustness of Channel Allocation in Joint Radar And Communication Systems: An Auction Approach**

雷达与通信联合系统中信道分配的稳健性：拍卖方法 cs.GT

**SubmitDate**: 2022-10-10    [abs](http://arxiv.org/abs/2208.09821v2) [paper-pdf](http://arxiv.org/pdf/2208.09821v2)

**Authors**: Ismail Lotfi, Hongyang Du, Dusit Niyato, Sumei Sun, Dong In Kim

**Abstract**: Joint radar and communication (JRC) is a promising technique for spectrum re-utilization, which enables radar sensing and data transmission to operate on the same frequencies and the same devices. However, due to the multi-objective property of JRC systems, channel allocation to JRC nodes should be carefully designed to maximize system performance. Additionally, because of the broadcast nature of wireless signals, a watchful adversary, i.e., a warden, can detect ongoing transmissions and attack the system. Thus, we develop a covert JRC system that minimizes the detection probability by wardens, in which friendly jammers are deployed to improve the covertness of the JRC nodes during radar sensing and data transmission operations. Furthermore, we propose a robust multi-item auction design for channel allocation for such a JRC system that considers the uncertainty in bids. The proposed auction mechanism achieves the properties of truthfulness, individual rationality, budget feasibility, and computational efficiency. The simulations clearly show the benefits of our design to support covert JRC systems and to provide incentive to the JRC nodes in obtaining spectrum, in which the auction-based channel allocation mechanism is robust against perturbations in the bids, which is highly effective for JRC nodes working in uncertain environments.

摘要: 联合雷达与通信(JRC)是一种很有前途的频谱再利用技术，它使雷达感知和数据传输能够在相同的频率和相同的设备上运行。然而，由于JRC系统的多目标特性，必须仔细设计JRC节点的信道分配以最大化系统性能。此外，由于无线信号的广播性质，警惕的对手，即典狱长，可以检测到正在进行的传输并攻击系统。因此，我们开发了一种隐蔽的JRC系统，该系统可以最小化管理员的发现概率，在该系统中部署友好的干扰器来提高JRC节点在雷达侦听和数据传输操作中的隐蔽性。此外，我们还提出了一种稳健的多物品拍卖设计，用于考虑出价不确定性的JRC系统的信道分配。该拍卖机制具有真实性、个体合理性、预算可行性和计算效率等特点。仿真结果表明，本文设计的JRC系统支持隐蔽JRC系统，并激励JRC节点获得频谱，其中基于拍卖的信道分配机制对投标中的扰动具有较强的鲁棒性，这对于工作在不确定环境中的JRC节点是非常有效的。



## **32. Pruning Adversarially Robust Neural Networks without Adversarial Examples**

无对抗性样本的对抗性稳健神经网络剪枝 cs.LG

Published at ICDM 2022 as a conference paper

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2210.04311v1) [paper-pdf](http://arxiv.org/pdf/2210.04311v1)

**Authors**: Tong Jian, Zifeng Wang, Yanzhi Wang, Jennifer Dy, Stratis Ioannidis

**Abstract**: Adversarial pruning compresses models while preserving robustness. Current methods require access to adversarial examples during pruning. This significantly hampers training efficiency. Moreover, as new adversarial attacks and training methods develop at a rapid rate, adversarial pruning methods need to be modified accordingly to keep up. In this work, we propose a novel framework to prune a previously trained robust neural network while maintaining adversarial robustness, without further generating adversarial examples. We leverage concurrent self-distillation and pruning to preserve knowledge in the original model as well as regularizing the pruned model via the Hilbert-Schmidt Information Bottleneck. We comprehensively evaluate our proposed framework and show its superior performance in terms of both adversarial robustness and efficiency when pruning architectures trained on the MNIST, CIFAR-10, and CIFAR-100 datasets against five state-of-the-art attacks. Code is available at https://github.com/neu-spiral/PwoA/.

摘要: 对抗性剪枝在保持健壮性的同时压缩模型。目前的方法需要在修剪过程中访问对抗性的例子。这严重影响了培训效率。此外，随着新的对抗性攻击和训练方法的快速发展，对抗性剪枝方法需要进行相应的修改以跟上。在这项工作中，我们提出了一种新的框架来修剪先前训练的稳健神经网络，同时保持对抗性的健壮性，而不会进一步产生对抗性的例子。我们利用并发的自我蒸馏和剪枝来保存原始模型中的知识，并通过希尔伯特-施密特信息瓶颈来规则化剪枝后的模型。我们对我们提出的框架进行了全面的评估，并在剪枝在MNIST、CIFAR-10和CIFAR-100数据集上训练的体系结构对抗五种最先进的攻击时，显示了其在对抗攻击鲁棒性和效率方面的优越性能。代码可在https://github.com/neu-spiral/PwoA/.上找到



## **33. Towards Understanding and Boosting Adversarial Transferability from a Distribution Perspective**

从分配角度理解和提高对抗性转移能力 cs.CV

\copyright 20XX IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2210.04213v1) [paper-pdf](http://arxiv.org/pdf/2210.04213v1)

**Authors**: Yao Zhu, Yuefeng Chen, Xiaodan Li, Kejiang Chen, Yuan He, Xiang Tian, Bolun Zheng, Yaowu Chen, Qingming Huang

**Abstract**: Transferable adversarial attacks against Deep neural networks (DNNs) have received broad attention in recent years. An adversarial example can be crafted by a surrogate model and then attack the unknown target model successfully, which brings a severe threat to DNNs. The exact underlying reasons for the transferability are still not completely understood. Previous work mostly explores the causes from the model perspective, e.g., decision boundary, model architecture, and model capacity. adversarial attacks against Deep neural networks (DNNs) have received broad attention in recent years. An adversarial example can be crafted by a surrogate model and then attack the unknown target model successfully, which brings a severe threat to DNNs. The exact underlying reasons for the transferability are still not completely understood. Previous work mostly explores the causes from the model perspective. Here, we investigate the transferability from the data distribution perspective and hypothesize that pushing the image away from its original distribution can enhance the adversarial transferability. To be specific, moving the image out of its original distribution makes different models hardly classify the image correctly, which benefits the untargeted attack, and dragging the image into the target distribution misleads the models to classify the image as the target class, which benefits the targeted attack. Towards this end, we propose a novel method that crafts adversarial examples by manipulating the distribution of the image. We conduct comprehensive transferable attacks against multiple DNNs to demonstrate the effectiveness of the proposed method. Our method can significantly improve the transferability of the crafted attacks and achieves state-of-the-art performance in both untargeted and targeted scenarios, surpassing the previous best method by up to 40$\%$ in some cases.

摘要: 针对深度神经网络(DNN)的可转移敌意攻击近年来受到广泛关注。利用代理模型可以构造出敌意实例，然后成功攻击未知目标模型，给DNN带来了严重的威胁。这种可转让性的确切潜在原因仍不完全清楚。以前的工作大多从模型的角度来探讨原因，例如决策边界、模型体系结构和模型容量。近年来，针对深度神经网络的敌意攻击受到了广泛的关注。利用代理模型可以构造出敌意实例，然后成功攻击未知目标模型，给DNN带来了严重的威胁。这种可转让性的确切潜在原因仍不完全清楚。以往的工作大多是从模型的角度来探讨原因。这里，我们从数据分布的角度研究图像的可转移性，并假设将图像从其原始分布推开可以增强对抗的可转移性。具体地说，将图像移出其原始分布会使不同的模型很难正确地对图像进行分类，这有利于非目标攻击，而将图像拖入目标分布会误导模型将图像分类为目标类，从而有利于目标攻击。为此，我们提出了一种新的方法，通过操纵图像的分布来制作对抗性例子。为了验证该方法的有效性，我们对多个DNN进行了综合的可转移攻击。我们的方法可以显著提高特制攻击的可转移性，在非目标和目标场景中都能达到最好的性能，在某些情况下比以前最好的方法高出40美元。



## **34. Adversarial Attacks against Windows PE Malware Detection: A Survey of the State-of-the-Art**

针对Windows PE恶意软件检测的对抗性攻击：现状综述 cs.CR

**SubmitDate**: 2022-10-09    [abs](http://arxiv.org/abs/2112.12310v2) [paper-pdf](http://arxiv.org/pdf/2112.12310v2)

**Authors**: Xiang Ling, Lingfei Wu, Jiangyu Zhang, Zhenqing Qu, Wei Deng, Xiang Chen, Yaguan Qian, Chunming Wu, Shouling Ji, Tianyue Luo, Jingzheng Wu, Yanjun Wu

**Abstract**: Malware has been one of the most damaging threats to computers that span across multiple operating systems and various file formats. To defend against ever-increasing and ever-evolving malware, tremendous efforts have been made to propose a variety of malware detection that attempt to effectively and efficiently detect malware so as to mitigate possible damages as early as possible. Recent studies have shown that, on the one hand, existing ML and DL techniques enable superior solutions in detecting newly emerging and previously unseen malware. However, on the other hand, ML and DL models are inherently vulnerable to adversarial attacks in the form of adversarial examples. In this paper, we focus on malware with the file format of portable executable (PE) in the family of Windows operating systems, namely Windows PE malware, as a representative case to study the adversarial attack methods in such adversarial settings. To be specific, we start by first outlining the general learning framework of Windows PE malware detection based on ML/DL and subsequently highlighting three unique challenges of performing adversarial attacks in the context of Windows PE malware. Then, we conduct a comprehensive and systematic review to categorize the state-of-the-art adversarial attacks against PE malware detection, as well as corresponding defenses to increase the robustness of Windows PE malware detection. Finally, we conclude the paper by first presenting other related attacks against Windows PE malware detection beyond the adversarial attacks and then shedding light on future research directions and opportunities. In addition, a curated resource list of adversarial attacks and defenses for Windows PE malware detection is also available at https://github.com/ryderling/ adversarial-attacks-and-defenses-for-windows-pe-malware-detection.

摘要: 恶意软件一直是计算机面临的最具破坏性的威胁之一，这些威胁跨越多个操作系统和各种文件格式。为了防御不断增长和不断演变的恶意软件，人们做出了巨大的努力，提出了各种恶意软件检测方法，试图有效和高效地检测恶意软件，以便尽早减轻可能的损害。最近的研究表明，一方面，现有的ML和DL技术能够在检测新出现的和以前未见过的恶意软件方面提供更好的解决方案。然而，另一方面，ML和DL模型天生就容易受到对抗性例子形式的对抗性攻击。本文以Windows操作系统家族中具有可移植可执行文件(PE)文件格式的恶意软件，即Windows PE恶意软件为典型案例，研究这种对抗性环境下的对抗性攻击方法。具体地说，我们首先概述了基于ML/DL的Windows PE恶意软件检测的一般学习框架，然后重点介绍了在Windows PE恶意软件环境中执行对抗性攻击的三个独特挑战。然后，我们对针对PE恶意软件检测的对抗性攻击进行了全面系统的回顾，并对相应的防御措施进行了分类，以增加Windows PE恶意软件检测的健壮性。最后，我们首先介绍了Windows PE恶意软件检测中除了对抗性攻击之外的其他相关攻击，并对未来的研究方向和机会进行了展望。此外，https://github.com/ryderling/adversarial-attacks-and-defenses-for-windows-pe-malware-detection.上还提供了针对Windows PE恶意软件检测的对抗性攻击和防御的精选资源列表



## **35. A Zero-Sum Game Framework for Optimal Sensor Placement in Uncertain Networked Control Systems under Cyber-Attacks**

网络攻击下不确定网络控制系统传感器最优配置的零和博弈框架 eess.SY

8 pages, 3 figues, Accepted to the 61st Conference on Decision and  Control, Cancun, December 2022

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04091v1) [paper-pdf](http://arxiv.org/pdf/2210.04091v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper proposes a game-theoretic approach to address the problem of optimal sensor placement against an adversary in uncertain networked control systems. The problem is formulated as a zero-sum game with two players, namely a malicious adversary and a detector. Given a protected performance vertex, we consider a detector, with uncertain system knowledge, that selects another vertex on which to place a sensor and monitors its output with the aim of detecting the presence of the adversary. On the other hand, the adversary, also with uncertain system knowledge, chooses a single vertex and conducts a cyber-attack on its input. The purpose of the adversary is to drive the attack vertex as to maximally disrupt the protected performance vertex while remaining undetected by the detector. As our first contribution, the game payoff of the above-defined zero-sum game is formulated in terms of the Value-at-Risk of the adversary's impact. However, this game payoff corresponds to an intractable optimization problem. To tackle the problem, we adopt the scenario approach to approximately compute the game payoff. Then, the optimal monitor selection is determined by analyzing the equilibrium of the zero-sum game. The proposed approach is illustrated via a numerical example of a 10-vertex networked control system.

摘要: 本文提出了一种博弈论方法来解决不确定网络控制系统中对抗对手的传感器最优配置问题。该问题被描述为一个有两个参与者的零和博弈，即一个恶意对手和一个检测器。在给定一个受保护的性能顶点的情况下，我们考虑了一个检测器，在系统知识不确定的情况下，它选择另一个顶点放置传感器并监视其输出，目的是检测对手的存在。另一方面，敌手也具有不确定的系统知识，选择单个顶点并对其输入进行网络攻击。敌手的目的是驱动攻击顶点，以最大限度地破坏受保护的性能顶点，同时保持不被检测器检测到。作为我们的第一个贡献，上面定义的零和博弈的博弈收益是根据对手影响的风险值来表示的。然而，这种游戏收益对应于一个棘手的优化问题。为了解决这个问题，我们采用情景方法来近似计算博弈收益。然后，通过对零和博弈均衡的分析，确定最优监控器选择。通过一个10点网络控制系统的数值算例，说明了该方法的有效性。



## **36. Symmetry Subgroup Defense Against Adversarial Attacks**

对抗攻击的对称性子群防御 cs.LG

14 pages

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04087v1) [paper-pdf](http://arxiv.org/pdf/2210.04087v1)

**Authors**: Blerta Lindqvist

**Abstract**: Adversarial attacks and defenses disregard the lack of invariance of convolutional neural networks (CNNs), that is, the inability of CNNs to classify samples and their symmetric transformations the same. The lack of invariance of CNNs with respect to symmetry transformations is detrimental when classifying transformed original samples but not necessarily detrimental when classifying transformed adversarial samples. For original images, the lack of invariance means that symmetrically transformed original samples are classified differently from their correct labels. However, for adversarial images, the lack of invariance means that symmetrically transformed adversarial images are classified differently from their incorrect adversarial labels. Might the CNN lack of invariance revert symmetrically transformed adversarial samples to the correct classification? This paper answers this question affirmatively for a threat model that ranges from zero-knowledge adversaries to perfect-knowledge adversaries. We base our defense against perfect-knowledge adversaries on devising a Klein four symmetry subgroup that incorporates an additional artificial symmetry of pixel intensity inversion. The closure property of the subgroup not only provides a framework for the accuracy evaluation but also confines the transformations that an adaptive, perfect-knowledge adversary can apply. We find that by using only symmetry defense, no adversarial samples, and by changing nothing in the model architecture and parameters, we can defend against white-box PGD adversarial attacks, surpassing the PGD adversarial training defense by up to ~50% even against a perfect-knowledge adversary for ImageNet. The proposed defense also maintains and surpasses the classification accuracy for non-adversarial samples.

摘要: 对抗性攻击和防御忽略了卷积神经网络(CNN)缺乏不变性，即CNN无法对样本及其对称变换进行相同的分类。当对变换的原始样本进行分类时，CNN关于对称变换的不变性的缺乏是有害的，但当对变换的对抗性样本进行分类时，CNN不一定是有害的。对于原始图像，缺乏不变性意味着对称变换的原始样本的分类不同于其正确的标签。然而，对于对抗性图像，缺乏不变性意味着对称变换的对抗性图像的分类与其错误的对抗性标签不同。缺乏不变性的CNN能否将对称变换的对抗性样本还原为正确的分类？对于一个从零知识对手到完全知识对手的威胁模型，本文肯定地回答了这个问题。我们基于对完美知识对手的防御，设计了一个Klein4对称子群，它包含了一个额外的像素强度反转的人工对称性。子群的封闭性不仅为准确度评估提供了一个框架，而且限制了自适应的、完全知识的对手可以应用的变换。我们发现，在不改变模型结构和参数的情况下，只使用对称防御，不使用对抗性样本，我们可以防御白盒PGD对抗性攻击，即使是针对ImageNet的完全知识对手，也可以超过PGD对抗性训练防御高达50%。该方法还保持并超过了非对抗性样本的分类精度。



## **37. Robustness of Unsupervised Representation Learning without Labels**

无标签非监督表示学习的鲁棒性 cs.LG

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04076v1) [paper-pdf](http://arxiv.org/pdf/2210.04076v1)

**Authors**: Aleksandar Petrov, Marta Kwiatkowska

**Abstract**: Unsupervised representation learning leverages large unlabeled datasets and is competitive with supervised learning. But non-robust encoders may affect downstream task robustness. Recently, robust representation encoders have become of interest. Still, all prior work evaluates robustness using a downstream classification task. Instead, we propose a family of unsupervised robustness measures, which are model- and task-agnostic and label-free. We benchmark state-of-the-art representation encoders and show that none dominates the rest. We offer unsupervised extensions to the FGSM and PGD attacks. When used in adversarial training, they improve most unsupervised robustness measures, including certified robustness. We validate our results against a linear probe and show that, for MOCOv2, adversarial training results in 3 times higher certified accuracy, a 2-fold decrease in impersonation attack success rate and considerable improvements in certified robustness.

摘要: 无监督表示学习利用了大量未标记的数据集，与监督学习相比具有竞争性。但是，非健壮的编码器可能会影响下游任务的健壮性。最近，稳健的表示编码器已经成为人们感兴趣的对象。尽管如此，所有先前的工作都是使用下游分类任务来评估稳健性。相反，我们提出了一类无监督的健壮性度量，它们与模型和任务无关，也与标签无关。我们对最先进的表示编码器进行了基准测试，并表明没有一个是主导其余的。我们提供对FGSM和PGD攻击的无监督扩展。当用于对抗性训练时，它们改善了大多数无监督的健壮性度量，包括认证的健壮性。我们用线性探头验证了我们的结果，结果表明，对于MOCOv2，对抗性训练导致认证准确率提高3倍，模拟攻击成功率降低2倍，认证稳健性显著提高。



## **38. FedDef: Robust Federated Learning-based Network Intrusion Detection Systems Against Gradient Leakage**

FedDef：基于联邦学习的抗梯度泄漏的健壮网络入侵检测系统 cs.CR

14 pages, 9 figures

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2210.04052v1) [paper-pdf](http://arxiv.org/pdf/2210.04052v1)

**Authors**: Jiahui Chen, Yi Zhao, Qi Li, Ke Xu

**Abstract**: Deep learning methods have been widely applied to anomaly-based network intrusion detection systems (NIDS) to detect malicious traffic. To expand the usage scenarios of DL-based methods, the federated learning (FL) framework allows intelligent techniques to jointly train a model by multiple individuals on the basis of respecting individual data privacy. However, it has not yet been systematically evaluated how robust FL-based NIDSs are against existing privacy attacks under existing defenses. To address this issue, in this paper we propose two privacy evaluation metrics designed for FL-based NIDSs, including leveraging two reconstruction attacks to recover the training data to obtain the privacy score for traffic features, followed by Generative Adversarial Network (GAN) based attack that generates adversarial examples with the reconstructed benign traffic to evaluate evasion rate against other NIDSs. We conduct experiments to show that existing defenses provide little protection that the corresponding adversarial traffic can even evade the SOTA NIDS Kitsune. To build a more robust FL-based NIDS, we further propose a novel optimization-based input perturbation defense strategy with theoretical guarantee that achieves both high utility by minimizing the gradient distance and strong privacy protection by maximizing the input distance. We experimentally evaluate four existing defenses on four datasets and show that our defense outperforms all the baselines with strong privacy guarantee while maintaining model accuracy loss within 3% under optimal parameter combination.

摘要: 深度学习方法已被广泛应用于基于异常的网络入侵检测系统中以检测恶意流量。为了扩展基于DL的方法的使用场景，联邦学习(FL)框架允许智能技术在尊重个人数据隐私的基础上由多个个体联合训练模型。然而，还没有系统地评估基于FL的NIDS在现有防御系统下对现有隐私攻击的健壮性。针对这一问题，本文提出了两个针对基于FL的网络入侵检测系统的隐私评估指标，包括利用两次重构攻击恢复训练数据以获得流量特征的隐私得分，以及基于生成性对抗网络(GAN)的攻击，利用重构的良性流量生成对抗性样本以评估对其他NIDS的逃避率。我们进行的实验表明，现有的防御措施提供的保护很少，相应的敌意流量甚至可以避开Sota NIDS Kitsune。为了构建一个更健壮的基于FL的网络入侵检测系统，我们进一步提出了一种新的基于优化的输入扰动防御策略，该策略通过最小化输入距离来实现高效用，并通过最大化输入距离来实现强隐私保护。我们在四个数据集上对四个已有的防御方案进行了实验评估，结果表明，在最优参数组合下，我们的防御方案在保持模型精度损失在3%以内的同时，性能优于所有具有较强隐私保障的基线。



## **39. Can Adversarial Training Be Manipulated By Non-Robust Features?**

对抗性训练能被非强健特征操纵吗？ cs.LG

NeurIPS 2022

**SubmitDate**: 2022-10-08    [abs](http://arxiv.org/abs/2201.13329v4) [paper-pdf](http://arxiv.org/pdf/2201.13329v4)

**Authors**: Lue Tao, Lei Feng, Hongxin Wei, Jinfeng Yi, Sheng-Jun Huang, Songcan Chen

**Abstract**: Adversarial training, originally designed to resist test-time adversarial examples, has shown to be promising in mitigating training-time availability attacks. This defense ability, however, is challenged in this paper. We identify a novel threat model named stability attack, which aims to hinder robust availability by slightly manipulating the training data. Under this threat, we show that adversarial training using a conventional defense budget $\epsilon$ provably fails to provide test robustness in a simple statistical setting, where the non-robust features of the training data can be reinforced by $\epsilon$-bounded perturbation. Further, we analyze the necessity of enlarging the defense budget to counter stability attacks. Finally, comprehensive experiments demonstrate that stability attacks are harmful on benchmark datasets, and thus the adaptive defense is necessary to maintain robustness. Our code is available at https://github.com/TLMichael/Hypocritical-Perturbation.

摘要: 对抗性训练最初是为了抵抗测试时间对抗性的例子，已经被证明在减轻训练时间可用性攻击方面很有希望。然而，这种防御能力在本文中受到了挑战。我们提出了一种名为稳定性攻击的新威胁模型，该模型旨在通过对训练数据的细微操作来阻碍健壮性可用性。在这种威胁下，我们证明了在简单的统计设置下，使用常规国防预算的对抗性训练不能提供测试稳健性，其中训练数据的非稳健性特征可以通过有界扰动来加强。进一步，我们分析了增加国防预算以对抗稳定性攻击的必要性。最后，综合实验表明，稳定性攻击对基准数据集是有害的，因此需要自适应防御来保持稳健性。我们的代码可以在https://github.com/TLMichael/Hypocritical-Perturbation.上找到



## **40. BayesImposter: Bayesian Estimation Based .bss Imposter Attack on Industrial Control Systems**

BayesImposter：基于贝叶斯估计的.BSS Imposter对工控系统的攻击 cs.CR

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03719v1) [paper-pdf](http://arxiv.org/pdf/2210.03719v1)

**Authors**: Anomadarshi Barua, Lelin Pan, Mohammad Abdullah Al Faruque

**Abstract**: Over the last six years, several papers used memory deduplication to trigger various security issues, such as leaking heap-address and causing bit-flip in the physical memory. The most essential requirement for successful memory deduplication is to provide identical copies of a physical page. Recent works use a brute-force approach to create identical copies of a physical page that is an inaccurate and time-consuming primitive from the attacker's perspective.   Our work begins to fill this gap by providing a domain-specific structured way to duplicate a physical page in cloud settings in the context of industrial control systems (ICSs). Here, we show a new attack primitive - \textit{BayesImposter}, which points out that the attacker can duplicate the .bss section of the target control DLL file of cloud protocols using the \textit{Bayesian estimation} technique. Our approach results in less memory (i.e., 4 KB compared to GB) and time (i.e., 13 minutes compared to hours) compared to the brute-force approach used in recent works. We point out that ICSs can be expressed as state-space models; hence, the \textit{Bayesian estimation} is an ideal choice to be combined with memory deduplication for a successful attack in cloud settings. To demonstrate the strength of \textit{BayesImposter}, we create a real-world automation platform using a scaled-down automated high-bay warehouse and industrial-grade SIMATIC S7-1500 PLC from Siemens as a target ICS. We demonstrate that \textit{BayesImposter} can predictively inject false commands into the PLC that can cause possible equipment damage with machine failure in the target ICS. Moreover, we show that \textit{BayesImposter} is capable of adversarial control over the target ICS resulting in severe consequences, such as killing a person but making it looks like an accident. Therefore, we also provide countermeasures to prevent the attack.

摘要: 在过去的六年里，几篇论文使用内存重复数据删除来触发各种安全问题，例如堆地址泄漏和导致物理内存中的位翻转。成功的内存重复数据删除最基本的要求是提供物理页面的完全相同的副本。最近的作品使用暴力方法来创建物理页面的相同副本，从攻击者的角度来看，这是一个不准确且耗时的原语。我们的工作开始填补这一空白，提供了一种特定于领域的结构化方法，在工业控制系统(ICSS)的环境中复制云环境中的物理页面。这里，我们展示了一个新的攻击原语--\textit{BayesImposter}，它指出攻击者可以使用\textit{贝叶斯估计}技术复制云协议的目标控制DLL文件的.bss部分。与最近工作中使用的暴力方法相比，我们的方法产生了更少的内存(即，4KB与GB相比)和时间(即，与数小时相比，13分钟)。我们指出，ICSS可以表示为状态空间模型，因此，在云环境下，将文本{贝叶斯估计}与内存重复数据删除相结合是成功攻击的理想选择。为了展示\textit{BayesImposter}的优势，我们使用一个缩小的自动化高架仓库和西门子的工业级SIMATIC S7-1500 PLC作为目标IC，创建了一个真实的自动化平台。我们证明了\textit{BayesImposter}可以预测性地向PLC注入错误命令，这些错误命令可能会在目标IC中出现机器故障时导致设备损坏。此外，我们证明了\textit{BayesImposter}能够对目标ICS进行对抗性控制，从而导致严重的后果，例如杀死一个人，但使其看起来像是一场事故。因此，我们也提供了防范攻击的对策。



## **41. A Wolf in Sheep's Clothing: Spreading Deadly Pathogens Under the Disguise of Popular Music**

披着羊皮的狼：在流行音乐的伪装下传播致命的病原体 cs.CR

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03688v1) [paper-pdf](http://arxiv.org/pdf/2210.03688v1)

**Authors**: Anomadarshi Barua, Yonatan Gizachew Achamyeleh, Mohammad Abdullah Al Faruque

**Abstract**: A Negative Pressure Room (NPR) is an essential requirement by the Bio-Safety Levels (BSLs) in biolabs or infectious-control hospitals to prevent deadly pathogens from being leaked from the facility. An NPR maintains a negative pressure inside with respect to the outside reference space so that microbes are contained inside of an NPR. Nowadays, differential pressure sensors (DPSs) are utilized by the Building Management Systems (BMSs) to control and monitor the negative pressure in an NPR. This paper demonstrates a non-invasive and stealthy attack on NPRs by spoofing a DPS at its resonant frequency. Our contributions are: (1) We show that DPSs used in NPRs typically have resonant frequencies in the audible range. (2) We use this finding to design malicious music to create resonance in DPSs, resulting in an overshooting in the DPS's normal pressure readings. (3) We show how the resonance in DPSs can fool the BMSs so that the NPR turns its negative pressure to a positive one, causing a potential \textit{leak} of deadly microbes from NPRs. We do experiments on 8 DPSs from 5 different manufacturers to evaluate their resonant frequencies considering the sampling tube length and find resonance in 6 DPSs. We can achieve a 2.5 Pa change in negative pressure from a $\sim$7 cm distance when a sampling tube is not present and from a $\sim$2.5 cm distance for a 1 m sampling tube length. We also introduce an interval-time variation approach for an adversarial control over the negative pressure and show that the \textit{forged} pressure can be varied within 12 - 33 Pa. Our attack is also capable of attacking multiple NPRs simultaneously. Moreover, we demonstrate our attack at a real-world NPR located in an anonymous bioresearch facility, which is FDA approved and follows CDC guidelines. We also provide countermeasures to prevent the attack.

摘要: 负压室(NPR)是Biolab或传染病控制医院的生物安全级别(BSL)的基本要求，以防止致命病原体从设施中泄漏。NPR内部相对于外部参考空间保持负压力，因此微生物被包含在NPR内部。目前，建筑管理系统(BMS)使用差压传感器(DPS)来控制和监测核电站的负压。通过对DPS的共振频率进行欺骗，实现了对NPR的非侵入性和隐蔽性攻击。我们的贡献是：(1)我们发现，NPR中使用的DPS通常具有在可听范围内的共振频率。(2)我们利用这一发现来设计恶意音乐来在DPSS中产生共鸣，导致DPS的正常压力读数超调。(3)我们展示了DPSS中的共振如何欺骗BMS，从而使NPR将其负压变为正压，从而导致来自NPR的潜在致命微生物的泄漏。我们对来自5个不同厂家的8个DPS进行了实验，在考虑采样管长度的情况下评估了它们的谐振频率，并在6个DPS中发现了共振。当没有采样管时，我们可以从7厘米的距离获得2.5帕的负压变化，而对于1米的采样管长度，我们可以从2.5厘米的距离实现负压的变化。我们还介绍了一种对负压进行对抗性控制的区间-时间变化法，并证明了在12-33Pa.我们的攻击也能够同时攻击多个NP。此外，我们演示了我们对位于匿名生物研究机构中的真实世界NPR的攻击，该机构是FDA批准的，并遵循CDC的指导方针。我们还提供了防止攻击的对策。



## **42. Empowering Graph Representation Learning with Test-Time Graph Transformation**

利用测试时间图变换增强图表示学习能力 cs.LG

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03561v1) [paper-pdf](http://arxiv.org/pdf/2210.03561v1)

**Authors**: Wei Jin, Tong Zhao, Jiayuan Ding, Yozen Liu, Jiliang Tang, Neil Shah

**Abstract**: As powerful tools for representation learning on graphs, graph neural networks (GNNs) have facilitated various applications from drug discovery to recommender systems. Nevertheless, the effectiveness of GNNs is immensely challenged by issues related to data quality, such as distribution shift, abnormal features and adversarial attacks. Recent efforts have been made on tackling these issues from a modeling perspective which requires additional cost of changing model architectures or re-training model parameters. In this work, we provide a data-centric view to tackle these issues and propose a graph transformation framework named GTrans which adapts and refines graph data at test time to achieve better performance. We provide theoretical analysis on the design of the framework and discuss why adapting graph data works better than adapting the model. Extensive experiments have demonstrated the effectiveness of GTrans on three distinct scenarios for eight benchmark datasets where suboptimal data is presented. Remarkably, GTrans performs the best in most cases with improvements up to 2.8%, 8.2% and 3.8% over the best baselines on three experimental settings.

摘要: 作为图上表示学习的强大工具，图神经网络(GNN)促进了从药物发现到推荐系统的各种应用。然而，GNN的有效性受到与数据质量有关的问题的巨大挑战，例如分布偏移、异常特征和对抗性攻击。最近在从建模的角度解决这些问题方面做出了努力，这需要更改模型体系结构或重新培训模型参数的额外成本。在这项工作中，我们提供了一个以数据为中心的视图来解决这些问题，并提出了一个名为GTrans的图转换框架，它在测试时对图数据进行调整和提炼，以获得更好的性能。我们对框架的设计进行了理论分析，并讨论了为什么采用图形数据比采用模型效果更好。广泛的实验已经证明了GTrans在三种不同的场景中的有效性，这些场景针对八个基准数据集，其中呈现的数据不是最优的。值得注意的是，在大多数情况下，GTrans的性能最好，在三个实验设置中，GTrans的性能比最佳基线分别提高了2.8%、8.2%和3.8%。



## **43. A2: Efficient Automated Attacker for Boosting Adversarial Training**

A2：用于加强对抗性训练的高效自动攻击者 cs.CV

Accepted by NeurIPS2022

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03543v1) [paper-pdf](http://arxiv.org/pdf/2210.03543v1)

**Authors**: Zhuoer Xu, Guanghui Zhu, Changhua Meng, Shiwen Cui, Zhenzhe Ying, Weiqiang Wang, Ming GU, Yihua Huang

**Abstract**: Based on the significant improvement of model robustness by AT (Adversarial Training), various variants have been proposed to further boost the performance. Well-recognized methods have focused on different components of AT (e.g., designing loss functions and leveraging additional unlabeled data). It is generally accepted that stronger perturbations yield more robust models. However, how to generate stronger perturbations efficiently is still missed. In this paper, we propose an efficient automated attacker called A2 to boost AT by generating the optimal perturbations on-the-fly during training. A2 is a parameterized automated attacker to search in the attacker space for the best attacker against the defense model and examples. Extensive experiments across different datasets demonstrate that A2 generates stronger perturbations with low extra cost and reliably improves the robustness of various AT methods against different attacks.

摘要: 在对抗训练显著提高模型稳健性的基础上，各种变种被提出以进一步提高性能。公认的方法侧重于AT的不同组成部分(例如，设计损失函数和利用额外的未标记数据)。人们普遍认为，更强的扰动会产生更稳健的模型。然而，如何有效地产生更强的扰动仍然是一个未解决的问题。在本文中，我们提出了一种称为A2的高效自动攻击者，通过在训练过程中生成最优的动态扰动来增强AT。A2是一个参数化的自动攻击者，可以在攻击者空间中搜索最好的攻击者，并针对防御模型和实例进行攻击。在不同数据集上的大量实验表明，A2以较低的额外代价产生更强的扰动，并可靠地提高了各种AT方法对不同攻击的健壮性。



## **44. Adversarially Robust Prototypical Few-shot Segmentation with Neural-ODEs**

基于神经网络的逆鲁棒原型少镜头分割 cs.CV

MICCAI 2022. arXiv admin note: substantial text overlap with  arXiv:2208.12428

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03429v1) [paper-pdf](http://arxiv.org/pdf/2210.03429v1)

**Authors**: Prashant Pandey, Aleti Vardhan, Mustafa Chasmai, Tanuj Sur, Brejesh Lall

**Abstract**: Few-shot Learning (FSL) methods are being adopted in settings where data is not abundantly available. This is especially seen in medical domains where the annotations are expensive to obtain. Deep Neural Networks have been shown to be vulnerable to adversarial attacks. This is even more severe in the case of FSL due to the lack of a large number of training examples. In this paper, we provide a framework to make few-shot segmentation models adversarially robust in the medical domain where such attacks can severely impact the decisions made by clinicians who use them. We propose a novel robust few-shot segmentation framework, Prototypical Neural Ordinary Differential Equation (PNODE), that provides defense against gradient-based adversarial attacks. We show that our framework is more robust compared to traditional adversarial defense mechanisms such as adversarial training. Adversarial training involves increased training time and shows robustness to limited types of attacks depending on the type of adversarial examples seen during training. Our proposed framework generalises well to common adversarial attacks like FGSM, PGD and SMIA while having the model parameters comparable to the existing few-shot segmentation models. We show the effectiveness of our proposed approach on three publicly available multi-organ segmentation datasets in both in-domain and cross-domain settings by attacking the support and query sets without the need for ad-hoc adversarial training.

摘要: 在数据不充足的情况下，正在采用少发式学习(FSL)方法。这在医学领域中尤其常见，在那里获得注释的成本很高。深度神经网络已被证明容易受到敌意攻击。在FSL的情况下，由于缺乏大量的训练实例，这一点更加严重。在这篇文章中，我们提供了一个框架，使少数镜头分割模型在医学领域具有相反的健壮性，其中此类攻击可能严重影响使用它们的临床医生的决策。我们提出了一种新的健壮的少镜头分割框架，原型神经常微分方程(PNODE)，它提供了对基于梯度的敌意攻击的防御。实验结果表明，与传统的对抗性训练等对抗性防御机制相比，该框架具有更强的健壮性。对抗性训练需要增加训练时间，并根据训练期间看到的对抗性例子的类型，对有限类型的攻击表现出健壮性。该框架对常见的对抗性攻击如FGSM、PGD和SMIA具有很好的通用性，同时具有与现有的少镜头分割模型相当的模型参数。我们在三个公开可用的多器官分割数据集上，通过攻击支持集和查询集，在域内和跨域环境下都证明了我们所提出的方法的有效性，而不需要特别的对抗性训练。



## **45. Semi-quantum private comparison and its generalization to the key agreement, summation, and anonymous ranking**

半量子私密比较及其对密钥协商、求和和匿名排序的推广 quant-ph

19 pages 5 tables

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03421v1) [paper-pdf](http://arxiv.org/pdf/2210.03421v1)

**Authors**: Chong-Qiang Ye, Jian Li, Xiu-Bo Chen, Yanyan Hou, Zhou Wang

**Abstract**: Semi-quantum protocols construct connections between quantum users and ``classical'' users who can only perform certain ``classical'' operations. In this paper, we present a new semi-quantum private comparison protocol based on entangled states and single particles, which does not require pre-shared keys between the ``classical'' users to guarantee the security of their private data. By utilizing multi-particle entangled states and single particles, our protocol can be easily extended to multi-party scenarios to meet the requirements of multiple ``classical'' users who want to compare their private data. The security analysis shows that the protocol can effectively prevent attacks from outside eavesdroppers and adversarial participants. Besides, we generalize the proposed protocol to other semi-quantum protocols such as semi-quantum key agreement, semi-quantum summation, and semi-quantum anonymous ranking protocols. We compare and discuss the proposed protocols with previous similar protocols. The results show that our protocols satisfy the demands of their respective counterparts separately. Therefore, our protocols have a wide range of application scenarios.

摘要: 半量子协议在量子用户和只能执行某些“经典”操作的“经典”用户之间建立连接。本文提出了一种新的基于纠缠态和单粒子的半量子私有比较协议，该协议不需要经典用户之间的预共享密钥来保证其私有数据的安全性。通过利用多粒子纠缠态和单粒子，我们的协议可以很容易地扩展到多方场景，以满足多个想要比较他们的私有数据的经典用户的需求。安全性分析表明，该协议能够有效地防御来自外部窃听者和敌对参与者的攻击。此外，我们将该协议推广到其他半量子协议，如半量子密钥协商协议、半量子求和协议和半量子匿名排序协议。我们将所提出的协议与以前的类似协议进行了比较和讨论。结果表明，我们的协议分别满足了相应协议的要求。因此，我们的协议具有广泛的应用场景。



## **46. Pre-trained Adversarial Perturbations**

预先训练的对抗性扰动 cs.CV

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03372v1) [paper-pdf](http://arxiv.org/pdf/2210.03372v1)

**Authors**: Yuanhao Ban, Yinpeng Dong

**Abstract**: Self-supervised pre-training has drawn increasing attention in recent years due to its superior performance on numerous downstream tasks after fine-tuning. However, it is well-known that deep learning models lack the robustness to adversarial examples, which can also invoke security issues to pre-trained models, despite being less explored. In this paper, we delve into the robustness of pre-trained models by introducing Pre-trained Adversarial Perturbations (PAPs), which are universal perturbations crafted for the pre-trained models to maintain the effectiveness when attacking fine-tuned ones without any knowledge of the downstream tasks. To this end, we propose a Low-Level Layer Lifting Attack (L4A) method to generate effective PAPs by lifting the neuron activations of low-level layers of the pre-trained models. Equipped with an enhanced noise augmentation strategy, L4A is effective at generating more transferable PAPs against fine-tuned models. Extensive experiments on typical pre-trained vision models and ten downstream tasks demonstrate that our method improves the attack success rate by a large margin compared with state-of-the-art methods.

摘要: 近年来，自监督预训练因其在经过微调后在众多下游任务中的优异表现而受到越来越多的关注。然而，众所周知，深度学习模型缺乏对敌意示例的健壮性，这也可能会引发预先训练的模型的安全问题，尽管研究较少。在本文中，我们通过引入预训练对抗扰动(PAP)来深入研究预训练模型的稳健性，PAP是为预训练模型设计的通用扰动，用于在攻击精调模型时保持有效性，而不需要了解下游任务。为此，我们提出了一种低层提升攻击(L4A)的方法，通过提升预训练模型低层神经元的激活来生成有效的PAP。配备了增强的噪音增强策略，L4A在针对微调模型生成更多可转移的PAP方面是有效的。在典型的预训练视觉模型和十个下游任务上的大量实验表明，该方法与现有方法相比，攻击成功率有较大幅度的提高。



## **47. Preprocessors Matter! Realistic Decision-Based Attacks on Machine Learning Systems**

预处理器很重要！基于现实决策的机器学习系统攻击 cs.CR

Code can be found at  https://github.com/google-research/preprocessor-aware-black-box-attack

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2210.03297v1) [paper-pdf](http://arxiv.org/pdf/2210.03297v1)

**Authors**: Chawin Sitawarin, Florian Tramèr, Nicholas Carlini

**Abstract**: Decision-based adversarial attacks construct inputs that fool a machine-learning model into making targeted mispredictions by making only hard-label queries. For the most part, these attacks have been applied directly to isolated neural network models. However, in practice, machine learning models are just a component of a much larger system. By adding just a single preprocessor in front of a classifier, we find that state-of-the-art query-based attacks are as much as seven times less effective at attacking a prediction pipeline than attacking the machine learning model alone. Hence, attacks that are unaware of this invariance inevitably waste a large number of queries to re-discover or overcome it. We, therefore, develop techniques to first reverse-engineer the preprocessor and then use this extracted information to attack the end-to-end system. Our extraction method requires only a few hundred queries to learn the preprocessors used by most publicly available model pipelines, and our preprocessor-aware attacks recover the same efficacy as just attacking the model alone. The code can be found at https://github.com/google-research/preprocessor-aware-black-box-attack.

摘要: 基于决策的对抗性攻击构建的输入通过只进行硬标签查询来欺骗机器学习模型进行有针对性的错误预测。在很大程度上，这些攻击直接应用于孤立的神经网络模型。然而，在实践中，机器学习模型只是一个大得多的系统的一个组成部分。通过在分类器前面添加单个预处理器，我们发现最先进的基于查询的攻击在攻击预测管道时的效率比单独攻击机器学习模型低七倍之多。因此，没有意识到这种不变性的攻击不可避免地会浪费大量查询来重新发现或克服它。因此，我们开发了一些技术，首先对预处理器进行逆向工程，然后使用这些提取的信息来攻击端到端系统。我们的提取方法只需要几百个查询就可以了解大多数公开可用的模型管道使用的预处理器，而我们的预处理器感知攻击与仅攻击模型具有相同的效果。代码可在https://github.com/google-research/preprocessor-aware-black-box-attack.上找到



## **48. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

30 pages, 7 figures, draft NeurIPS version

**SubmitDate**: 2022-10-07    [abs](http://arxiv.org/abs/2205.01663v4) [paper-pdf](http://arxiv.org/pdf/2205.01663v4)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstract**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a safe language generation task (``avoid injuries'') as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. We found that adversarial training increased robustness to the adversarial attacks that we trained on -- doubling the time for our contractors to find adversarial examples both with our tool (from 13 to 26 minutes) and without (from 20 to 44 minutes) -- without affecting in-distribution performance.   We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用了一个安全的语言生成任务(`避免受伤‘)作为通过对抗性训练获得高可靠性的试验床。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们的任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。我们发现对抗性训练增加了对我们训练的对抗性攻击的健壮性--使我们的承包商在使用我们的工具(从13分钟到26分钟)和不使用我们的工具(从20分钟到44分钟)的情况下找到对抗性例子的时间翻了一番--而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **49. Truth Serum: Poisoning Machine Learning Models to Reveal Their Secrets**

真相血清：毒化机器学习模型以揭示其秘密 cs.CR

ACM CCS 2022

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2204.00032v2) [paper-pdf](http://arxiv.org/pdf/2204.00032v2)

**Authors**: Florian Tramèr, Reza Shokri, Ayrton San Joaquin, Hoang Le, Matthew Jagielski, Sanghyun Hong, Nicholas Carlini

**Abstract**: We introduce a new class of attacks on machine learning models. We show that an adversary who can poison a training dataset can cause models trained on this dataset to leak significant private details of training points belonging to other parties. Our active inference attacks connect two independent lines of work targeting the integrity and privacy of machine learning training data.   Our attacks are effective across membership inference, attribute inference, and data extraction. For example, our targeted attacks can poison <0.1% of the training dataset to boost the performance of inference attacks by 1 to 2 orders of magnitude. Further, an adversary who controls a significant fraction of the training data (e.g., 50%) can launch untargeted attacks that enable 8x more precise inference on all other users' otherwise-private data points.   Our results cast doubts on the relevance of cryptographic privacy guarantees in multiparty computation protocols for machine learning, if parties can arbitrarily select their share of training data.

摘要: 我们在机器学习模型上引入了一类新的攻击。我们表明，可以毒化训练数据集的对手可以导致在该数据集上训练的模型泄露属于其他方的训练点的重要私人细节。我们的主动推理攻击将两个独立的工作线连接在一起，目标是机器学习训练数据的完整性和隐私。我们的攻击在成员关系推理、属性推理和数据提取方面都是有效的。例如，我们的有针对性的攻击可以毒化<0.1%的训练数据集，将推理攻击的性能提高1到2个数量级。此外，控制很大一部分训练数据(例如50%)的对手可以发起无目标攻击，从而能够对所有其他用户的其他私有数据点进行8倍的精确推断。我们的结果对用于机器学习的多方计算协议中的密码隐私保证的相关性提出了怀疑，如果各方可以任意选择他们在训练数据中的份额。



## **50. EvilScreen Attack: Smart TV Hijacking via Multi-channel Remote Control Mimicry**

EvilScreen攻击：模仿多频道遥控器劫持智能电视 cs.CR

**SubmitDate**: 2022-10-06    [abs](http://arxiv.org/abs/2210.03014v1) [paper-pdf](http://arxiv.org/pdf/2210.03014v1)

**Authors**: Yiwei Zhang, Siqi Ma, Tiancheng Chen, Juanru Li, Robert H. Deng, Elisa Bertino

**Abstract**: Modern smart TVs often communicate with their remote controls (including those smart phone simulated ones) using multiple wireless channels (e.g., Infrared, Bluetooth, and Wi-Fi). However, this multi-channel remote control communication introduces a new attack surface. An inherent security flaw is that remote controls of most smart TVs are designed to work in a benign environment rather than an adversarial one, and thus wireless communications between a smart TV and its remote controls are not strongly protected. Attackers could leverage such flaw to abuse the remote control communication and compromise smart TV systems. In this paper, we propose EvilScreen, a novel attack that exploits ill-protected remote control communications to access protected resources of a smart TV or even control the screen. EvilScreen exploits a multi-channel remote control mimicry vulnerability present in today smart TVs. Unlike other attacks, which compromise the TV system by exploiting code vulnerabilities or malicious third-party apps, EvilScreen directly reuses commands of different remote controls, combines them together to circumvent deployed authentication and isolation policies, and finally accesses or controls TV resources remotely. We evaluated eight mainstream smart TVs and found that they are all vulnerable to EvilScreen attacks, including a Samsung product adopting the ISO/IEC security specification.

摘要: 现代智能电视通常使用多种无线通道(如红外线、蓝牙和Wi-Fi)与遥控器(包括那些模拟智能手机的遥控器)进行通信。然而，这种多通道远程控制通信引入了一个新的攻击面。一个固有的安全缺陷是，大多数智能电视的遥控器被设计为在良性环境中工作，而不是在对抗性环境中工作，因此智能电视与遥控器之间的无线通信没有得到强有力的保护。攻击者可以利用这样的漏洞来滥用遥控器通信并危害智能电视系统。在本文中，我们提出了一种新的攻击EvilScreen，它利用不受保护的远程控制通信来访问受保护的智能电视资源，甚至控制屏幕。EvilScreen利用了当今智能电视中存在的多频道遥控器模仿漏洞。与利用代码漏洞或恶意第三方应用程序危害电视系统的其他攻击不同，EvilScreen直接重用不同遥控器的命令，将它们组合在一起以规避部署的身份验证和隔离策略，最终远程访问或控制电视资源。我们评估了8款主流智能电视，发现它们都容易受到EvilScreen攻击，其中包括一款采用ISO/IEC安全规范的三星产品。



