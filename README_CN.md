# New Adversarial Attack Papers
**update at 2021-11-20 23:56:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Resilient Consensus-based Multi-agent Reinforcement Learning with Function Approximation**

基于弹性共识的函数逼近多智能体强化学习 cs.LG

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.06776v2)

**Authors**: Martin Figura, Yixuan Lin, Ji Liu, Vijay Gupta

**Abstracts**: Adversarial attacks during training can strongly influence the performance of multi-agent reinforcement learning algorithms. It is, thus, highly desirable to augment existing algorithms such that the impact of adversarial attacks on cooperative networks is eliminated, or at least bounded. In this work, we consider a fully decentralized network, where each agent receives a local reward and observes the global state and action. We propose a resilient consensus-based actor-critic algorithm, whereby each agent estimates the team-average reward and value function, and communicates the associated parameter vectors to its immediate neighbors. We show that in the presence of Byzantine agents, whose estimation and communication strategies are completely arbitrary, the estimates of the cooperative agents converge to a bounded consensus value with probability one, provided that there are at most $H$ Byzantine agents in the neighborhood of each cooperative agent and the network is $(2H+1)$-robust. Furthermore, we prove that the policy of the cooperative agents converges with probability one to a bounded neighborhood around a local maximizer of their team-average objective function under the assumption that the policies of the adversarial agents asymptotically become stationary.

摘要: 训练过程中的对抗性攻击会严重影响多智能体强化学习算法的性能。因此，非常需要对现有算法进行扩充，以便消除或至少有界地消除对抗性攻击对协作网络的影响。在这项工作中，我们考虑了一个完全分散的网络，在这个网络中，每个代理都会获得局部奖励，并观察全局状态和行动。我们提出了一种弹性的基于共识的行动者-批评者算法，其中每个Agent估计团队平均奖励和价值函数，并将相关的参数向量传达给它的直接邻居。我们证明了当拜占庭代理的估计和通信策略完全任意时，假设每个合作代理的邻域中至多有$H$拜占庭代理，并且网络是$(2H+1)$-鲁棒的，则合作代理的估计以概率1收敛到有界的合意值。在假设对抗性Agent的策略渐近平稳的前提下，证明了合作Agent的策略以概率1收敛到其团队平均目标函数的局部极大值附近的有界邻域。



## **2. Robust Person Re-identification with Multi-Modal Joint Defence**

基于多模态联合防御的鲁棒人物再识别 cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09571v1)

**Authors**: Yunpeng Gong, Lifei Chen

**Abstracts**: The Person Re-identification (ReID) system based on metric learning has been proved to inherit the vulnerability of deep neural networks (DNNs), which are easy to be fooled by adversarail metric attacks. Existing work mainly relies on adversarial training for metric defense, and more methods have not been fully studied. By exploring the impact of attacks on the underlying features, we propose targeted methods for metric attacks and defence methods. In terms of metric attack, we use the local color deviation to construct the intra-class variation of the input to attack color features. In terms of metric defenses, we propose a joint defense method which includes two parts of proactive defense and passive defense. Proactive defense helps to enhance the robustness of the model to color variations and the learning of structure relations across multiple modalities by constructing different inputs from multimodal images, and passive defense exploits the invariance of structural features in a changing pixel space by circuitous scaling to preserve structural features while eliminating some of the adversarial noise. Extensive experiments demonstrate that the proposed joint defense compared with the existing adversarial metric defense methods which not only against multiple attacks at the same time but also has not significantly reduced the generalization capacity of the model. The code is available at https://github.com/finger-monkey/multi-modal_joint_defence.

摘要: 基于度量学习的人物识别(ReID)系统继承了深层神经网络(DNNs)易被恶意度量攻击欺骗的弱点。现有的工作主要依靠对抗性训练进行度量防御，更多的方法还没有得到充分的研究。通过研究攻击对底层特征的影响，提出了有针对性的度量攻击方法和防御方法。在度量攻击方面，我们利用局部颜色偏差来构造输入的类内变异来攻击颜色特征。在度量防御方面，我们提出了一种包括主动防御和被动防御两部分的联合防御方法。主动防御通过从多模态图像构造不同的输入来增强模型对颜色变化的鲁棒性和跨多模态的结构关系的学习，而被动防御通过迂回缩放利用结构特征在变化的像素空间中的不变性来保留结构特征，同时消除一些对抗性噪声。大量实验表明，与现有的对抗性度量防御方法相比，本文提出的联合防御方法不仅可以同时防御多个攻击，而且没有显着降低模型的泛化能力。代码可在https://github.com/finger-monkey/multi-modal_joint_defence.上获得



## **3. DPA: Learning Robust Physical Adversarial Camouflages for Object Detectors**

DPA：学习对象检测器的健壮物理对抗伪装 cs.CV

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2109.00124v2)

**Authors**: Yexin Duan, Jialin Chen, Xingyu Zhou, Junhua Zou, Zhengyun He, Wu Zhang, Jin Zhang, Zhisong Pan

**Abstracts**: Adversarial attacks are feasible in the real world for object detection. However, most of the previous works have tried to learn local "patches" applied to an object to fool detectors, which become less effective in squint view angles. To address this issue, we propose the Dense Proposals Attack (DPA) to learn one-piece, physical, and targeted adversarial camouflages for detectors. The camouflages are one-piece because they are generated as a whole for an object, physical because they remain adversarial when filmed under arbitrary viewpoints and different illumination conditions, and targeted because they can cause detectors to misidentify an object as a specific target class. In order to make the generated camouflages robust in the physical world, we introduce a combination of transformations to model the physical phenomena. In addition, to improve the attacks, DPA simultaneously attacks all the classifications in the fixed proposals. Moreover, we build a virtual 3D scene using the Unity simulation engine to fairly and reproducibly evaluate different physical attacks. Extensive experiments demonstrate that DPA outperforms the state-of-the-art methods, and it is generic for any object and generalized well to the real world, posing a potential threat to the security-critical computer vision systems.

摘要: 对抗性攻击在现实世界中用于目标检测是可行的。然而，以前的大多数工作都试图学习应用于对象的局部“补丁”来愚弄检测器，这在斜视视角下变得不那么有效。为了解决这个问题，我们提出了密集建议攻击(DPA)，以学习检测器的整体、物理和有针对性的对抗伪装。伪装是一体式的，因为它们是为对象整体生成的；物理伪装是因为当在任意视点和不同的照明条件下拍摄时它们仍然是对抗性的；以及目标伪装是因为它们可能导致检测器将对象错误地识别为特定的目标类别。为了使生成的伪装在物理世界中具有鲁棒性，我们引入了一种组合变换来模拟物理现象。此外，为了改进攻击，DPA同时攻击固定方案中的所有分类。此外，我们使用Unity仿真引擎构建了一个虚拟的3D场景，以公平、可重复性地评估不同的物理攻击。大量的实验表明，DPA的性能优于目前最先进的方法，它对任何对象都是通用的，并且可以很好地推广到现实世界，这对安全关键的计算机视觉系统构成了潜在的威胁。



## **4. Adversarial attacks on voter model dynamics in complex networks**

复杂网络中选民模型动态的对抗性攻击 physics.soc-ph

6 pages, 4 figures

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09561v1)

**Authors**: Katsumi Chiyomaru, Kazuhiro Takemoto

**Abstracts**: This study investigates adversarial attacks conducted to distort the voter model dynamics in complex networks. Specifically, a simple adversarial attack method is proposed for holding the state of an individual's opinions closer to the target state in the voter model dynamics; the method shows that even when one opinion is the majority, the vote outcome can be inverted (i.e., the outcome can lean toward the other opinion) by adding extremely small (hard-to-detect) perturbations strategically generated in social networks. Adversarial attacks are relatively more effective for complex (large and dense) networks. The results indicate that opinion dynamics can be unknowingly distorted.

摘要: 这项研究调查了复杂网络中为扭曲选民模型动态而进行的对抗性攻击。具体地说，提出了一种简单的对抗性攻击方法，用于在选民模型动态中保持个人意见的状态更接近目标状态；该方法表明，即使当一种意见占多数时，投票结果也可以通过添加在社会网络中策略性地产生的极小(难以检测)的扰动来反转(即，结果可以倾向于另一种意见)。对抗性攻击对于复杂(大型和密集)网络相对更有效。结果表明，意见动态可能在不知不觉中被扭曲。



## **5. ZeBRA: Precisely Destroying Neural Networks with Zero-Data Based Repeated Bit Flip Attack**

Zebra：基于零数据重复位翻转攻击精确摧毁神经网络 cs.LG

14 pages, 3 figures, 5 tables, Accepted at British Machine Vision  Conference (BMVC) 2021

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.01080v2)

**Authors**: Dahoon Park, Kon-Woo Kwon, Sunghoon Im, Jaeha Kung

**Abstracts**: In this paper, we present Zero-data Based Repeated bit flip Attack (ZeBRA) that precisely destroys deep neural networks (DNNs) by synthesizing its own attack datasets. Many prior works on adversarial weight attack require not only the weight parameters, but also the training or test dataset in searching vulnerable bits to be attacked. We propose to synthesize the attack dataset, named distilled target data, by utilizing the statistics of batch normalization layers in the victim DNN model. Equipped with the distilled target data, our ZeBRA algorithm can search vulnerable bits in the model without accessing training or test dataset. Thus, our approach makes the adversarial weight attack more fatal to the security of DNNs. Our experimental results show that 2.0x (CIFAR-10) and 1.6x (ImageNet) less number of bit flips are required on average to destroy DNNs compared to the previous attack method. Our code is available at https://github. com/pdh930105/ZeBRA.

摘要: 本文提出了一种基于零数据的重复位翻转攻击(Zebra)，它通过合成自己的攻击数据集来精确地破坏深度神经网络(DNNs)。以往许多关于对抗性权重攻击的工作不仅需要权重参数，还需要训练或测试数据集来搜索易受攻击的部位。我们提出利用受害者DNN模型中的批归一化层的统计信息来合成攻击数据集，称为提取的目标数据。有了提取的目标数据，我们的斑马算法可以在不访问训练或测试数据集的情况下搜索模型中的易受攻击的位。因此，我们的方法使得敌意加权攻击对DNNs的安全性更加致命。我们的实验结果表明，与以前的攻击方法相比，破坏DNN平均需要减少2.0倍(CIFAR-10)和1.6倍(ImageNet)的比特翻转次数。我们的代码可在https://github.获得com/pdh930105/zebra。



## **6. Finding Optimal Tangent Points for Reducing Distortions of Hard-label Attacks**

寻找最优切点以减少硬标签攻击的失真 cs.CV

accepted at NeurIPS 2021, including the appendix

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.07492v2)

**Authors**: Chen Ma, Xiangyu Guo, Li Chen, Jun-Hai Yong, Yisen Wang

**Abstracts**: One major problem in black-box adversarial attacks is the high query complexity in the hard-label attack setting, where only the top-1 predicted label is available. In this paper, we propose a novel geometric-based approach called Tangent Attack (TA), which identifies an optimal tangent point of a virtual hemisphere located on the decision boundary to reduce the distortion of the attack. Assuming the decision boundary is locally flat, we theoretically prove that the minimum $\ell_2$ distortion can be obtained by reaching the decision boundary along the tangent line passing through such tangent point in each iteration. To improve the robustness of our method, we further propose a generalized method which replaces the hemisphere with a semi-ellipsoid to adapt to curved decision boundaries. Our approach is free of hyperparameters and pre-training. Extensive experiments conducted on the ImageNet and CIFAR-10 datasets demonstrate that our approach can consume only a small number of queries to achieve the low-magnitude distortion. The implementation source code is released online at https://github.com/machanic/TangentAttack.

摘要: 黑盒对抗性攻击的一个主要问题是硬标签攻击设置中的高查询复杂度，在硬标签攻击设置中，只有前1个预测标签可用。本文提出了一种新的基于几何的切线攻击方法(TA)，该方法识别位于决策边界上的虚拟半球的最佳切点，以减少攻击的失真。假设决策边界是局部平坦的，我们从理论上证明了在每一次迭代中，沿着通过该切点的切线到达决策边界可以获得最小的$\\ell2$失真。为了提高方法的鲁棒性，我们进一步提出了一种广义方法，用半椭球代替半球，以适应弯曲的决策边界。我们的方法没有超参数和预训练。在ImageNet和CIFAR-10数据集上进行的大量实验表明，我们的方法可以只消耗少量的查询来实现低幅度的失真。实现源代码在https://github.com/machanic/TangentAttack.上在线发布



## **7. Attacking Deep Learning AI Hardware with Universal Adversarial Perturbation**

利用普遍对抗性扰动攻击深度学习人工智能硬件 cs.CR

**SubmitDate**: 2021-11-18    [paper-pdf](http://arxiv.org/pdf/2111.09488v1)

**Authors**: Mehdi Sadi, B. M. S. Bahar Talukder, Kaniz Mishty, Md Tauhidur Rahman

**Abstracts**: Universal Adversarial Perturbations are image-agnostic and model-independent noise that when added with any image can mislead the trained Deep Convolutional Neural Networks into the wrong prediction. Since these Universal Adversarial Perturbations can seriously jeopardize the security and integrity of practical Deep Learning applications, existing techniques use additional neural networks to detect the existence of these noises at the input image source. In this paper, we demonstrate an attack strategy that when activated by rogue means (e.g., malware, trojan) can bypass these existing countermeasures by augmenting the adversarial noise at the AI hardware accelerator stage. We demonstrate the accelerator-level universal adversarial noise attack on several deep Learning models using co-simulation of the software kernel of Conv2D function and the Verilog RTL model of the hardware under the FuseSoC environment.

摘要: 普遍的对抗性扰动是图像不可知和模型无关的噪声，当加入任何图像时，都会将训练好的深卷积神经网络误导到错误的预测中。由于这些普遍的对抗性扰动会严重危害实际深度学习应用的安全性和完整性，现有技术使用附加的神经网络来检测输入图像源处是否存在这些噪声。在本文中，我们展示了一种攻击策略，当被流氓手段(如恶意软件、特洛伊木马)激活时，可以通过在人工智能硬件加速器阶段增加对抗性噪声来绕过这些现有的对策。在FuseSoC环境下，通过Conv2D函数的软件内核和硬件的Verilog RTL模型的联合仿真，演示了几种深度学习模型上的加速级通用对抗噪声攻击。



## **8. Cortical Features for Defense Against Adversarial Audio Attacks**

防御敌意音频攻击的皮层特征 cs.SD

Co-author legal name changed

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2102.00313v2)

**Authors**: Ilya Kavalerov, Ruijie Zheng, Wojciech Czaja, Rama Chellappa

**Abstracts**: We propose using a computational model of the auditory cortex as a defense against adversarial attacks on audio. We apply several white-box iterative optimization-based adversarial attacks to an implementation of Amazon Alexa's HW network, and a modified version of this network with an integrated cortical representation, and show that the cortical features help defend against universal adversarial examples. At the same level of distortion, the adversarial noises found for the cortical network are always less effective for universal audio attacks. We make our code publicly available at https://github.com/ilyakava/py3fst.

摘要: 我们建议使用听觉皮层的计算模型来防御对音频的敌意攻击。我们将几个基于白盒迭代优化的敌意攻击应用到Amazon Alexa的硬件网络的一个实现中，以及该网络的一个带有集成皮层表示的修改版本，并显示了皮层特征有助于防御通用的敌意示例。在相同的失真水平下，皮层网络中发现的对抗性噪声对于通用音频攻击总是不太有效。我们在https://github.com/ilyakava/py3fst.上公开了我们的代码



## **9. Address Behaviour Vulnerabilities in the Next Generation of Autonomous Robots**

解决下一代自主机器人的行为漏洞 cs.RO

preprint and extended version of Nature Machine Intelligence, Vol 3,  November 2021, Pag 927-928

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2103.13268v2)

**Authors**: Michele Colledanchise

**Abstracts**: Robots applications in our daily life increase at an unprecedented pace. As robots will soon operate "out in the wild", we must identify the safety and security vulnerabilities they will face. Robotics researchers and manufacturers focus their attention on new, cheaper, and more reliable applications. Still, they often disregard the operability in adversarial environments where a trusted or untrusted user can jeopardize or even alter the robot's task.   In this paper, we identify a new paradigm of security threats in the next generation of robots. These threats fall beyond the known hardware or network-based ones, and we must find new solutions to address them. These new threats include malicious use of the robot's privileged access, tampering with the robot sensors system, and tricking the robot's deliberation into harmful behaviors. We provide a taxonomy of attacks that exploit these vulnerabilities with realistic examples, and we outline effective countermeasures to prevent better, detect, and mitigate them.

摘要: 机器人在我们日常生活中的应用正以前所未有的速度增长。由于机器人即将“在野外”作业，我们必须确定它们将面临的安全和安保漏洞。机器人研究人员和制造商将他们的注意力集中在新的、更便宜的和更可靠的应用上。尽管如此，他们经常忽视在敌对环境中的可操作性，在这种环境中，可信或不可信的用户可能会危及甚至改变机器人的任务。在这篇文章中，我们确定了下一代机器人安全威胁的新范例。这些威胁超出了已知的硬件或基于网络的威胁，我们必须找到新的解决方案来应对它们。这些新的威胁包括恶意使用机器人的特权访问，篡改机器人传感器系统，以及欺骗机器人的蓄意做出有害行为。我们通过实际示例提供了利用这些漏洞的攻击分类，并概述了有效的对策以更好地预防、检测和缓解它们。



## **10. Do Not Trust Prediction Scores for Membership Inference Attacks**

不信任成员身份推断攻击的预测分数 cs.LG

15 pages, 9 figures, 9 tables

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.09076v1)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Arguably, most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures, e.g., ReLU type neural networks produce almost always high prediction scores far away from the training data. Consequently, MIAs will miserably fail since this behavior leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of classifiers and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions far away from the training data, the more they reveal the training data.

摘要: 成员关系推断攻击(MIA)的目的是确定特定样本是否用于训练预测模型。知道这一点确实可能会导致隐私被侵犯。然而，可以说，大多数MIA都利用了模型的预测分数-在给定一些输入的情况下，每个输出的概率-遵循这样的直觉，即训练的模型在其训练数据上往往表现不同。我们认为这对于许多现代深层网络结构来说是一种谬误，例如，RELU类型的神经网络在远离训练数据的地方几乎总是产生高的预测分数。因此，MIA将悲惨地失败，因为这种行为不仅在已知域上，而且在分布外的数据上都会导致高的假阳性率，并且隐含地起到了防御MIA的作用。具体地说，使用生成性对抗性网络，我们能够产生潜在的无限数量的样本，这些样本被错误地分类为训练数据的一部分。换句话说，MIA的威胁被高估了，泄露的信息比之前假设的要少。此外，分类器的过度自信和他们对MIA的敏感性之间实际上存在着权衡：分类器知道的越多，他们不知道的时候，做出远离训练数据的低置信度预测的人就越多，他们透露的训练数据就越多。



## **11. Generating Unrestricted 3D Adversarial Point Clouds**

生成不受限制的3D对抗性点云 cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08973v1)

**Authors**: Xuelong Dai, Yanjie Li, Hua Dai, Bin Xiao

**Abstracts**: Utilizing 3D point cloud data has become an urgent need for the deployment of artificial intelligence in many areas like facial recognition and self-driving. However, deep learning for 3D point clouds is still vulnerable to adversarial attacks, e.g., iterative attacks, point transformation attacks, and generative attacks. These attacks need to restrict perturbations of adversarial examples within a strict bound, leading to the unrealistic adversarial 3D point clouds. In this paper, we propose an Adversarial Graph-Convolutional Generative Adversarial Network (AdvGCGAN) to generate visually realistic adversarial 3D point clouds from scratch. Specifically, we use a graph convolutional generator and a discriminator with an auxiliary classifier to generate realistic point clouds, which learn the latent distribution from the real 3D data. The unrestricted adversarial attack loss is incorporated in the special adversarial training of GAN, which enables the generator to generate the adversarial examples to spoof the target network. Compared with the existing state-of-art attack methods, the experiment results demonstrate the effectiveness of our unrestricted adversarial attack methods with a higher attack success rate and visual quality. Additionally, the proposed AdvGCGAN can achieve better performance against defense models and better transferability than existing attack methods with strong camouflage.

摘要: 利用三维点云数据已经成为人脸识别、自动驾驶等多个领域人工智能部署的迫切需要。然而，三维点云的深度学习仍然容易受到对抗性攻击，如迭代攻击、点变换攻击和生成性攻击。这些攻击需要将对抗性示例的扰动限制在一个严格的范围内，从而导致不真实的对抗性三维点云。本文提出了一种对抗性图形-卷积生成对抗性网络(AdvGCGAN)，用于从头开始生成视觉逼真的对抗性三维点云。具体地说，我们使用一个图形卷积生成器和一个带有辅助分类器的鉴别器来生成逼真的点云，从真实的3D数据中学习潜在的分布。将不受限制的对抗性攻击损失纳入到GAN的特殊对抗性训练中，使生成器能够生成欺骗目标网络的对抗性示例。实验结果表明，与现有的现有攻击方法相比，本文提出的无限制对抗性攻击方法具有更高的攻击成功率和视觉质量。此外，与现有的伪装性强的攻击方法相比，提出的AdvGCGAN能够获得更好的防御模型性能和更好的可移植性。



## **12. TraSw: Tracklet-Switch Adversarial Attacks against Multi-Object Tracking**

TraSw：针对多目标跟踪的Tracklet-Switch敌意攻击 cs.CV

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2111.08954v1)

**Authors**: Delv Lin, Qi Chen, Chengyu Zhou, Kun He

**Abstracts**: Benefiting from the development of Deep Neural Networks, Multi-Object Tracking (MOT) has achieved aggressive progress. Currently, the real-time Joint-Detection-Tracking (JDT) based MOT trackers gain increasing attention and derive many excellent models. However, the robustness of JDT trackers is rarely studied, and it is challenging to attack the MOT system since its mature association algorithms are designed to be robust against errors during tracking. In this work, we analyze the weakness of JDT trackers and propose a novel adversarial attack method, called Tracklet-Switch (TraSw), against the complete tracking pipeline of MOT. Specifically, a push-pull loss and a center leaping optimization are designed to generate adversarial examples for both re-ID feature and object detection. TraSw can fool the tracker to fail to track the targets in the subsequent frames by attacking very few frames. We evaluate our method on the advanced deep trackers (i.e., FairMOT, JDE, ByteTrack) using the MOT-Challenge datasets (i.e., 2DMOT15, MOT17, and MOT20). Experiments show that TraSw can achieve a high success rate of over 95% by attacking only five frames on average for the single-target attack and a reasonably high success rate of over 80% for the multiple-target attack. The code is available at https://github.com/DerryHub/FairMOT-attack .

摘要: 得益于深度神经网络的发展，多目标跟踪(MOT)取得了突飞猛进的发展。目前，基于实时联合检测跟踪(JDT)的MOT跟踪器受到越来越多的关注，并衍生出许多优秀的模型。然而，JDT跟踪器的鲁棒性研究很少，而且由于其成熟的关联算法被设计成对跟踪过程中的错误具有鲁棒性，因此对MOT系统的攻击是具有挑战性的。在这项工作中，我们分析了JDT跟踪器的弱点，并针对MOT的完整跟踪流水线提出了一种新的对抗性攻击方法，称为Tracklet-Switch(TraSw)。具体地说，推拉损失和中心跳跃优化被设计为生成Re-ID特征和目标检测的对抗性示例。TraSw可以通过攻击极少的帧来欺骗跟踪器，使其无法跟踪后续帧中的目标。我们在先进的深度跟踪器(即FairMOT、JDE、ByteTrack)上使用MOT-Challenge2DMOT15、MOT17和MOT20数据集对我们的方法进行了评估。实验表明，TraSw对于单目标攻击平均只攻击5帧，对多目标攻击具有相当高的成功率，成功率在95%以上，而对于多目标攻击，成功率在80%以上。代码可在https://github.com/DerryHub/FairMOT-attack上获得。



## **13. Turning Your Strength against You: Detecting and Mitigating Robust and Universal Adversarial Patch Attacks**

把你的力量转向你：检测和减轻健壮的和通用的敌意补丁攻击 cs.CR

**SubmitDate**: 2021-11-17    [paper-pdf](http://arxiv.org/pdf/2108.05075v2)

**Authors**: Zitao Chen, Pritam Dash, Karthik Pattabiraman

**Abstracts**: Adversarial patch attacks against image classification deep neural networks (DNNs), which inject arbitrary distortions within a bounded region of an image, can generate adversarial perturbations that are robust (i.e., remain adversarial in physical world) and universal (i.e., remain adversarial on any input). Such attacks can lead to severe consequences in real-world DNN-based systems.   This work proposes Jujutsu, a technique to detect and mitigate robust and universal adversarial patch attacks. For detection, Jujutsu exploits the attacks' universal property - Jujutsu first locates the region of the potential adversarial patch, and then strategically transfers it to a dedicated region in a new image to determine whether it is truly malicious. For attack mitigation, Jujutsu leverages the attacks' localized nature via image inpainting to synthesize the semantic contents in the pixels that are corrupted by the attacks, and reconstruct the ``clean'' image.   We evaluate Jujutsu on four diverse datasets (ImageNet, ImageNette, CelebA and Place365), and show that Jujutsu achieves superior performance and significantly outperforms existing techniques. We find that Jujutsu can further defend against different variants of the basic attack, including 1) physical-world attack; 2) attacks that target diverse classes; 3) attacks that construct patches in different shapes and 4) adaptive attacks.

摘要: 针对图像分类深度神经网络(DNNs)的对抗性补丁攻击在图像的有界区域内注入任意失真，可以产生鲁棒的(即，在物理世界中保持对抗性)和普遍的(即，在任何输入上保持对抗性)的对抗性扰动。这类攻击可能会在现实世界中基于DNN的系统中导致严重后果。这项工作提出了Jujutsu，这是一种检测和减轻健壮的、通用的敌意补丁攻击的技术。为了进行检测，Jujutsu利用攻击的通用属性-Jujutsu首先定位潜在对手补丁的区域，然后战略性地将其传输到新图像中的专用区域，以确定它是否真的是恶意的。为了缓解攻击，Jujutsu利用攻击的局部性，通过图像修复来合成被攻击破坏的像素中的语义内容，并重建“干净”的图像。我们在四个不同的数据集(ImageNet，ImageNette，CelebA和Place365)上对Jujutsu进行了评估，结果表明Jujutsu取得了优越的性能，并且远远超过了现有的技术。我们发现Jujutsu可以进一步防御基本攻击的不同变体，包括1)物理世界攻击；2)针对不同类别的攻击；3)构造不同形状补丁的攻击；4)自适应攻击。



## **14. Detecting AutoAttack Perturbations in the Frequency Domain**

在频域中检测AutoAttack扰动 cs.CV

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08785v1)

**Authors**: Peter Lorenz, Paula Harder, Dominik Strassel, Margret Keuper, Janis Keuper

**Abstracts**: Recently, adversarial attacks on image classification networks by the AutoAttack (Croce and Hein, 2020b) framework have drawn a lot of attention. While AutoAttack has shown a very high attack success rate, most defense approaches are focusing on network hardening and robustness enhancements, like adversarial training. This way, the currently best-reported method can withstand about 66% of adversarial examples on CIFAR10. In this paper, we investigate the spatial and frequency domain properties of AutoAttack and propose an alternative defense. Instead of hardening a network, we detect adversarial attacks during inference, rejecting manipulated inputs. Based on a rather simple and fast analysis in the frequency domain, we introduce two different detection algorithms. First, a black box detector that only operates on the input images and achieves a detection accuracy of 100% on the AutoAttack CIFAR10 benchmark and 99.3% on ImageNet, for epsilon = 8/255 in both cases. Second, a whitebox detector using an analysis of CNN feature maps, leading to a detection rate of also 100% and 98.7% on the same benchmarks.

摘要: 最近，AutoAttack(Croce and Hein，2020b)框架对图像分类网络的敌意攻击引起了广泛关注。虽然AutoAttack显示了非常高的攻击成功率，但大多数防御方法都集中在网络强化和健壮性增强上，如对抗性训练。这样，目前报道最好的方法可以承受CIFAR10上约66%的对抗性例子。在本文中，我们研究了AutoAttack的空域和频域特性，并提出了一种替代的防御方案。我们不是强化网络，而是在推理过程中检测敌意攻击，拒绝被操纵的输入。在对频域进行较为简单快速的分析的基础上，介绍了两种不同的检测算法。首先，黑盒检测器仅对输入图像进行操作，在AutoAttack CIFAR10基准上实现100%的检测准确率，在ImageNet上达到99.3%的检测准确率，对于epsilon=8/255这两种情况都是如此。其次，白盒检测器使用CNN特征地图分析，在相同基准上的检测率也分别为100%和98.7%。



## **15. Robustness of Bayesian Neural Networks to White-Box Adversarial Attacks**

贝叶斯神经网络对白盒攻击的鲁棒性 cs.LG

Accepted at the fourth IEEE International Conference on Artificial  Intelligence and Knowledge Engineering (AIKE 2021)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08591v1)

**Authors**: Adaku Uchendu, Daniel Campoy, Christopher Menart, Alexandra Hildenbrandt

**Abstracts**: Bayesian Neural Networks (BNNs), unlike Traditional Neural Networks (TNNs) are robust and adept at handling adversarial attacks by incorporating randomness. This randomness improves the estimation of uncertainty, a feature lacking in TNNs. Thus, we investigate the robustness of BNNs to white-box attacks using multiple Bayesian neural architectures. Furthermore, we create our BNN model, called BNN-DenseNet, by fusing Bayesian inference (i.e., variational Bayes) to the DenseNet architecture, and BDAV, by combining this intervention with adversarial training. Experiments are conducted on the CIFAR-10 and FGVC-Aircraft datasets. We attack our models with strong white-box attacks ($l_\infty$-FGSM, $l_\infty$-PGD, $l_2$-PGD, EOT $l_\infty$-FGSM, and EOT $l_\infty$-PGD). In all experiments, at least one BNN outperforms traditional neural networks during adversarial attack scenarios. An adversarially-trained BNN outperforms its non-Bayesian, adversarially-trained counterpart in most experiments, and often by significant margins. Lastly, we investigate network calibration and find that BNNs do not make overconfident predictions, providing evidence that BNNs are also better at measuring uncertainty.

摘要: 贝叶斯神经网络(BNNs)不同于传统的神经网络(TNNs)，它通过结合随机性，具有较强的鲁棒性和处理敌意攻击的能力。这种随机性改善了对不确定性的估计，这是TNN所缺乏的一个特征。因此，我们使用多种贝叶斯神经结构来研究BNN对白盒攻击的鲁棒性。此外，我们通过将贝叶斯推理(即变分贝叶斯)融合到DenseNet体系结构中，创建了我们的BNN模型，称为BNN-DenseNet，并将这种干预与对抗性训练相结合，创建了BDAV。实验是在CIFAR-10和FGVC-Aircraft数据集上进行的。我们用强白盒攻击($l_\infty$-FGSM、$l_\infty$-pgd、$l_2$-pgd、EOT$l_\infty$-FGSM和EOT$l_\infty$-pgd)攻击我们的模型。在所有实验中，在对抗性攻击场景中，至少有一个BNN的性能优于传统神经网络。在大多数实验中，经过对抗性训练的BNN比非贝叶斯的对应物性能要好，而且往往有很大的差距。最后，我们对网络校准进行了研究，发现BNN不会做出过于自信的预测，这为BNN在测量不确定性方面也做得更好提供了证据。



## **16. Improving the robustness and accuracy of biomedical language models through adversarial training**

通过对抗性训练提高生物医学语言模型的稳健性和准确性 cs.CL

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08529v1)

**Authors**: Milad Moradi, Matthias Samwald

**Abstracts**: Deep transformer neural network models have improved the predictive accuracy of intelligent text processing systems in the biomedical domain. They have obtained state-of-the-art performance scores on a wide variety of biomedical and clinical Natural Language Processing (NLP) benchmarks. However, the robustness and reliability of these models has been less explored so far. Neural NLP models can be easily fooled by adversarial samples, i.e. minor changes to input that preserve the meaning and understandability of the text but force the NLP system to make erroneous decisions. This raises serious concerns about the security and trust-worthiness of biomedical NLP systems, especially when they are intended to be deployed in real-world use cases. We investigated the robustness of several transformer neural language models, i.e. BioBERT, SciBERT, BioMed-RoBERTa, and Bio-ClinicalBERT, on a wide range of biomedical and clinical text processing tasks. We implemented various adversarial attack methods to test the NLP systems in different attack scenarios. Experimental results showed that the biomedical NLP models are sensitive to adversarial samples; their performance dropped in average by 21 and 18.9 absolute percent on character-level and word-level adversarial noise, respectively. Conducting extensive adversarial training experiments, we fine-tuned the NLP models on a mixture of clean samples and adversarial inputs. Results showed that adversarial training is an effective defense mechanism against adversarial noise; the models robustness improved in average by 11.3 absolute percent. In addition, the models performance on clean data increased in average by 2.4 absolute present, demonstrating that adversarial training can boost generalization abilities of biomedical NLP systems.

摘要: 深层变压器神经网络模型提高了生物医学领域智能文本处理系统的预测精度。他们在各种各样的生物医学和临床自然语言处理(NLP)基准上获得了最先进的性能分数。然而，到目前为止，人们对这些模型的稳健性和可靠性的探讨较少。神经NLP模型很容易被对抗性样本愚弄，即对输入进行微小的改变，以保持文本的含义和可理解性，但迫使NLP系统做出错误的决定。这引起了人们对生物医学NLP系统的安全性和可信性的严重担忧，特别是当它们打算部署在现实世界的用例中时。我们研究了几种变压器神经语言模型，即BioBERT、SciBERT、BioMed-Roberta和Bio-ClinicalBERT在广泛的生物医学和临床文本处理任务中的鲁棒性。我们实现了各种对抗性攻击方法，在不同的攻击场景下对NLP系统进行了测试。实验结果表明，生物医学自然语言处理模型对对抗性样本比较敏感，在字符级和词级对抗性噪声方面的性能分别平均下降了21%和18.9%。通过进行广泛的对抗性训练实验，我们在干净样本和对抗性输入的混合上对NLP模型进行了微调。结果表明，对抗性训练是抵抗对抗性噪声的一种有效防御机制，模型的鲁棒性平均提高了11.3个绝对百分比。此外，模型在干净数据上的性能目前平均提高了2.4绝对值，表明对抗性训练可以提高生物医学自然语言处理系统的泛化能力。



## **17. Consistent Semantic Attacks on Optical Flow**

对光流的一致语义攻击 cs.CV

Paper and supplementary material

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08485v1)

**Authors**: Tom Koren, Lior Talker, Michael Dinerstein, Roy J Jevnisek

**Abstracts**: We present a novel approach for semantically targeted adversarial attacks on Optical Flow. In such attacks the goal is to corrupt the flow predictions of a specific object category or instance. Usually, an attacker seeks to hide the adversarial perturbations in the input. However, a quick scan of the output reveals the attack. In contrast, our method helps to hide the attackers intent in the output as well. We achieve this thanks to a regularization term that encourages off-target consistency. We perform extensive tests on leading optical flow models to demonstrate the benefits of our approach in both white-box and black-box settings. Also, we demonstrate the effectiveness of our attack on subsequent tasks that depend on the optical flow.

摘要: 提出了一种新的针对光流的语义定向对抗性攻击方法。在这类攻击中，目标是破坏特定对象、类别或实例的流量预测。通常，攻击者试图隐藏输入中的对抗性扰动。但是，快速扫描输出会发现攻击。相反，我们的方法还有助于在输出中隐藏攻击者的意图。我们实现了这一点，这要归功于鼓励脱离目标的一致性的正规化条件。我们对主要的光流模型进行了广泛的测试，以演示我们的方法在白盒和黑盒设置中的优势。此外，我们还展示了我们对依赖于光流的后续任务的攻击的有效性。



## **18. Meta-Learning the Search Distribution of Black-Box Random Search Based Adversarial Attacks**

基于黑盒随机搜索的对抗性攻击搜索分布元学习 cs.LG

accepted at NeurIPS 2021; updated the numbers in Table 5 and added  references

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.01714v2)

**Authors**: Maksym Yatsura, Jan Hendrik Metzen, Matthias Hein

**Abstracts**: Adversarial attacks based on randomized search schemes have obtained state-of-the-art results in black-box robustness evaluation recently. However, as we demonstrate in this work, their efficiency in different query budget regimes depends on manual design and heuristic tuning of the underlying proposal distributions. We study how this issue can be addressed by adapting the proposal distribution online based on the information obtained during the attack. We consider Square Attack, which is a state-of-the-art score-based black-box attack, and demonstrate how its performance can be improved by a learned controller that adjusts the parameters of the proposal distribution online during the attack. We train the controller using gradient-based end-to-end training on a CIFAR10 model with white box access. We demonstrate that plugging the learned controller into the attack consistently improves its black-box robustness estimate in different query regimes by up to 20% for a wide range of different models with black-box access. We further show that the learned adaptation principle transfers well to the other data distributions such as CIFAR100 or ImageNet and to the targeted attack setting.

摘要: 近年来，基于随机搜索方案的对抗性攻击在黑盒健壮性评估方面取得了最新的研究成果。然而，正如我们在这项工作中演示的那样，它们在不同查询预算机制中的效率取决于对底层提案分布的手动设计和启发式调优。我们研究了如何根据攻击期间获得的信息在线调整建议分发来解决这个问题。我们考虑Square攻击，这是一种最先进的基于分数的黑盒攻击，并展示了如何通过学习控制器在攻击期间在线调整建议分布的参数来提高其性能。我们在带有白盒访问的CIFAR10模型上使用基于梯度的端到端训练来训练控制器。我们证明，对于具有黑盒访问的大范围不同模型，在不同的查询机制下，将学习控制器插入攻击可持续提高其黑盒健壮性估计高达20%。我们进一步表明，学习的适应原则很好地移植到其他数据分布，如CIFAR100或ImageNet，以及目标攻击设置。



## **19. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

弥合简历和NLP之间的鸿沟！一种基于梯度的文本对抗性攻击框架 cs.CL

Work on progress

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2110.15317v2)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstracts**: Despite great success on many machine learning tasks, deep neural networks are still vulnerable to adversarial samples. While gradient-based adversarial attack methods are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of text. To bridge this gap, we propose a general framework to adapt existing gradient-based methods to craft textual adversarial samples. In this framework, gradient-based continuous perturbations are added to the embedding layer and are amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a mask language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with \textbf{T}extual \textbf{P}rojected \textbf{G}radient \textbf{D}escent (\textbf{TPGD}). We conduct comprehensive experiments to evaluate our framework by performing transfer black-box attacks on BERT, RoBERTa and ALBERT on three benchmark datasets. Experimental results demonstrate our method achieves an overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. All the code and data will be made public.

摘要: 尽管深度神经网络在许多机器学习任务中取得了巨大的成功，但它仍然容易受到敌意样本的影响。虽然基于梯度的对抗性攻击方法在计算机视觉领域得到了很好的探索，但由于文本的离散性，将其直接应用于自然语言处理是不切实际的。为了弥补这一差距，我们提出了一个通用框架，以适应现有的基于梯度的方法来制作文本对抗性样本。在该框架中，基于梯度的连续扰动被添加到嵌入层，并在前向传播过程中被放大。然后用掩码语言模型头部对最终扰动的潜在表示进行解码，得到潜在的对抗性样本。在本文中，我们用\textbf{T}extual\textbf{P}rojected\textbf{G}Radient\textbf{D}light(\textbf{tpgd})实例化我们的框架。我们通过在三个基准数据集上对Bert、Roberta和Albert进行传输黑盒攻击，对我们的框架进行了全面的测试。实验结果表明，与强基线方法相比，我们的方法取得了总体上更好的性能，生成了更流畅、更具语法意义的对抗性样本。所有的代码和数据都将公之于众。



## **20. InFlow: Robust outlier detection utilizing Normalizing Flows**

流入：利用归一化流进行稳健的离群值检测 cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2106.12894v2)

**Authors**: Nishant Kumar, Pia Hanfeld, Michael Hecht, Michael Bussmann, Stefan Gumhold, Nico Hoffmann

**Abstracts**: Normalizing flows are prominent deep generative models that provide tractable probability distributions and efficient density estimation. However, they are well known to fail while detecting Out-of-Distribution (OOD) inputs as they directly encode the local features of the input representations in their latent space. In this paper, we solve this overconfidence issue of normalizing flows by demonstrating that flows, if extended by an attention mechanism, can reliably detect outliers including adversarial attacks. Our approach does not require outlier data for training and we showcase the efficiency of our method for OOD detection by reporting state-of-the-art performance in diverse experimental settings. Code available at https://github.com/ComputationalRadiationPhysics/InFlow .

摘要: 归一化流是重要的深度生成模型，它提供易于处理的概率分布和有效的密度估计。然而，众所周知，它们在检测非分布(OOD)输入时会失败，因为它们直接在其潜在空间中编码输入表示的局部特征。在本文中，我们通过证明流，如果通过注意机制扩展，可以可靠地检测包括对抗性攻击在内的离群值，来解决流归一化的过度自信问题。我们的方法不需要用于训练的离群值数据，我们通过报告在不同实验设置中的最新性能来展示我们的方法用于OOD检测的效率。代码可在https://github.com/ComputationalRadiationPhysics/InFlow上找到。



## **21. TSS: Transformation-Specific Smoothing for Robustness Certification**

TSS：用于鲁棒性认证的变换特定平滑 cs.LG

2021 ACM SIGSAC Conference on Computer and Communications Security  (CCS '21)

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2002.12398v5)

**Authors**: Linyi Li, Maurice Weber, Xiaojun Xu, Luka Rimanic, Bhavya Kailkhura, Tao Xie, Ce Zhang, Bo Li

**Abstracts**: As machine learning (ML) systems become pervasive, safeguarding their security is critical. However, recently it has been demonstrated that motivated adversaries are able to mislead ML systems by perturbing test data using semantic transformations. While there exists a rich body of research providing provable robustness guarantees for ML models against $\ell_p$ norm bounded adversarial perturbations, guarantees against semantic perturbations remain largely underexplored. In this paper, we provide TSS -- a unified framework for certifying ML robustness against general adversarial semantic transformations. First, depending on the properties of each transformation, we divide common transformations into two categories, namely resolvable (e.g., Gaussian blur) and differentially resolvable (e.g., rotation) transformations. For the former, we propose transformation-specific randomized smoothing strategies and obtain strong robustness certification. The latter category covers transformations that involve interpolation errors, and we propose a novel approach based on stratified sampling to certify the robustness. Our framework TSS leverages these certification strategies and combines with consistency-enhanced training to provide rigorous certification of robustness. We conduct extensive experiments on over ten types of challenging semantic transformations and show that TSS significantly outperforms the state of the art. Moreover, to the best of our knowledge, TSS is the first approach that achieves nontrivial certified robustness on the large-scale ImageNet dataset. For instance, our framework achieves 30.4% certified robust accuracy against rotation attack (within $\pm 30^\circ$) on ImageNet. Moreover, to consider a broader range of transformations, we show TSS is also robust against adaptive attacks and unforeseen image corruptions such as CIFAR-10-C and ImageNet-C.

摘要: 随着机器学习(ML)系统的普及，保护其安全性至关重要。然而，最近已经证明，有动机的攻击者能够通过使用语义转换扰乱测试数据来误导ML系统。虽然已经有大量的研究为ML模型提供了针对$\ellp$范数有界的对抗性扰动的可证明的健壮性保证，但针对语义扰动的保证在很大程度上还没有被探索。在这篇文章中，我们提供了TSS--一个统一的框架，用于证明ML对一般敌意语义转换的健壮性。首先，根据每个变换的性质，我们将常见变换分为两类，即可分辨变换(例如，高斯模糊)和可微分分辨变换(例如，旋转)。对于前者，我们提出了特定于变换的随机化平滑策略，并获得了强鲁棒性证明。后一类包括涉及插值误差的变换，我们提出了一种新的基于分层抽样的方法来证明鲁棒性。我们的框架TSS利用这些认证策略，并与一致性增强培训相结合，以提供严格的健壮性认证。我们在十多种具有挑战性的语义转换上进行了广泛的实验，结果表明TSS的性能明显优于现有技术。此外，据我们所知，TSS是在大规模ImageNet数据集上实现非平凡认证健壮性的第一种方法。例如，我们的框架在ImageNet上对旋转攻击(在$\pm 30^\cic$内)达到了30.4%的认证鲁棒准确率。此外，为了考虑更广泛的变换，我们还证明了TSS对诸如CIFAR-10-C和ImageNet-C这样的自适应攻击和不可预见的图像损坏也是健壮的。



## **22. Android HIV: A Study of Repackaging Malware for Evading Machine-Learning Detection**

Android HIV：逃避机器学习检测的恶意软件重新打包研究 cs.CR

14 pages, 11 figures

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/1808.04218v4)

**Authors**: Xiao Chen, Chaoran Li, Derui Wang, Sheng Wen, Jun Zhang, Surya Nepal, Yang Xiang, Kui Ren

**Abstracts**: Machine learning based solutions have been successfully employed for automatic detection of malware on Android. However, machine learning models lack robustness to adversarial examples, which are crafted by adding carefully chosen perturbations to the normal inputs. So far, the adversarial examples can only deceive detectors that rely on syntactic features (e.g., requested permissions, API calls, etc), and the perturbations can only be implemented by simply modifying application's manifest. While recent Android malware detectors rely more on semantic features from Dalvik bytecode rather than manifest, existing attacking/defending methods are no longer effective. In this paper, we introduce a new attacking method that generates adversarial examples of Android malware and evades being detected by the current models. To this end, we propose a method of applying optimal perturbations onto Android APK that can successfully deceive the machine learning detectors. We develop an automated tool to generate the adversarial examples without human intervention. In contrast to existing works, the adversarial examples crafted by our method can also deceive recent machine learning based detectors that rely on semantic features such as control-flow-graph. The perturbations can also be implemented directly onto APK's Dalvik bytecode rather than Android manifest to evade from recent detectors. We demonstrate our attack on two state-of-the-art Android malware detection schemes, MaMaDroid and Drebin. Our results show that the malware detection rates decreased from 96% to 0% in MaMaDroid, and from 97% to 0% in Drebin, with just a small number of codes to be inserted into the APK.

摘要: 基于机器学习的解决方案已经成功地应用于Android上的恶意软件自动检测。然而，机器学习模型对敌意示例缺乏鲁棒性，这些示例是通过在正常输入中添加精心选择的扰动来制作的。到目前为止，敌意示例只能欺骗依赖于句法特征(例如，请求的权限、API调用等)的检测器，并且只能通过简单地修改应用程序的清单来实现扰动。虽然最近的Android恶意软件检测器更多地依赖于Dalvik字节码的语义特征，而不是表现出来，但现有的攻击/防御方法已经不再有效。本文介绍了一种新的攻击方法，该方法生成Android恶意软件的恶意示例，从而逃避被现有模型检测到的攻击。为此，我们提出了一种将最优扰动应用于Android APK的方法，成功地欺骗了机器学习检测器。我们开发了一个自动生成对抗性实例的工具，无需人工干预。与已有工作相比，该方法制作的对抗性实例还可以欺骗目前依赖于控制流图等语义特征的基于机器学习的检测器。这些干扰也可以直接实现到APK的Dalvik字节码上，而不是Android清单上，以躲避最近的检测器。我们演示了我们对两个最先进的Android恶意软件检测方案MaMaDroid和Drebin的攻击。结果表明，在APK中插入少量代码的情况下，MaMaDroid的恶意软件检测率从96%下降到0%，Drebin的恶意软件检测率从97%下降到0%。



## **23. A Survey on Adversarial Attacks for Malware Analysis**

面向恶意软件分析的敌意攻击研究综述 cs.CR

42 Pages, 31 Figures, 11 Tables

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08223v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam

**Abstracts**: Machine learning has witnessed tremendous growth in its adoption and advancement in the last decade. The evolution of machine learning from traditional algorithms to modern deep learning architectures has shaped the way today's technology functions. Its unprecedented ability to discover knowledge/patterns from unstructured data and automate the decision-making process led to its application in wide domains. High flying machine learning arena has been recently pegged back by the introduction of adversarial attacks. Adversaries are able to modify data, maximizing the classification error of the models. The discovery of blind spots in machine learning models has been exploited by adversarial attackers by generating subtle intentional perturbations in test samples. Increasing dependency on data has paved the blueprint for ever-high incentives to camouflage machine learning models. To cope with probable catastrophic consequences in the future, continuous research is required to find vulnerabilities in form of adversarial and design remedies in systems. This survey aims at providing the encyclopedic introduction to adversarial attacks that are carried out against malware detection systems. The paper will introduce various machine learning techniques used to generate adversarial and explain the structure of target files. The survey will also model the threat posed by the adversary and followed by brief descriptions of widely accepted adversarial algorithms. Work will provide a taxonomy of adversarial evasion attacks on the basis of attack domain and adversarial generation techniques. Adversarial evasion attacks carried out against malware detectors will be discussed briefly under each taxonomical headings and compared with concomitant researches. Analyzing the current research challenges in an adversarial generation, the survey will conclude by pinpointing the open future research directions.

摘要: 在过去的十年里，机器学习在其采用和发展方面取得了巨大的增长。机器学习从传统算法到现代深度学习体系结构的演变塑造了当今技术的运作方式。它前所未有的从非结构化数据中发现知识/模式并使决策过程自动化的能力使其在广泛的领域得到了应用。最近，由于对抗性攻击的引入，高飞行机器学习领域受到了阻碍。攻击者能够修改数据，最大化模型的分类错误。机器学习模型中盲点的发现已经被敌意攻击者通过在测试样本中产生微妙的故意扰动来利用。对数据的日益依赖已经为伪装机器学习模型的持续高额激励铺平了蓝图。为了应对未来可能出现的灾难性后果，需要不断进行研究，以发现系统中对抗性形式的漏洞和设计补救措施。本调查旨在提供针对恶意软件检测系统进行的敌意攻击的百科全书介绍。本文将介绍用于生成对抗性的各种机器学习技术，并解释目标文件的结构。调查还将模拟对手构成的威胁，随后简要描述被广泛接受的对抗性算法。工作将提供基于攻击域和敌意生成技术的对抗性逃避攻击的分类。针对恶意软件检测器进行的敌意规避攻击将在每个分类标题下简要讨论，并与相应的研究进行比较。通过分析当前对抗性世代的研究挑战，本调查将通过指出开放的未来研究方向来得出结论。



## **24. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

FedCG：利用条件GAN保护隐私并保持联合学习中的好胜性能 cs.LG

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v1)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to collaboratively build machine learning models without sharing their private data. However, recent works demonstrate that FL is vulnerable to gradient-based data recovery attacks. Varieties of privacy-preserving technologies have been leveraged to further enhance the privacy of FL. Nonetheless, they either are computational or communication expensive (e.g., homomorphic encryption) or suffer from precision loss (e.g., differential privacy). In this work, we propose \textsc{FedCG}, a novel \underline{fed}erated learning method that leverages \underline{c}onditional \underline{g}enerative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. More specifically, \textsc{FedCG} decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors which is the culprit of privacy leakage, \textsc{FedCG} shares clients' generators with the server for aggregating common knowledge aiming to enhance the performance of clients' local networks. Extensive experiments demonstrate that \textsc{FedCG} can achieve competitive model performance compared with baseline FL methods, and numerical privacy analysis shows that \textsc{FedCG} has high-level privacy-preserving capability.

摘要: 联合学习(FL)旨在通过使客户能够在不共享其私有数据的情况下协作地构建机器学习模型来保护数据隐私。然而，最近的研究表明FL很容易受到基于梯度的数据恢复攻击。各种隐私保护技术已经被利用来进一步增强FL的隐私。然而，它们或者计算或通信昂贵(例如，同态加密)，或者遭受精度损失(例如，差分隐私)。在这项工作中，我们提出了一种新颖的学习方法--Textsc{FedCG}，它利用条件{c}生成的敌意网络来实现高级别的隐私保护，同时保持好胜模型的性能。在此基础上，我们提出了一种新的学习方法--下划线{fedCG}，它利用{c}条件{g}生成的敌意网络来实现高水平的隐私保护。更具体地说，\textsc{FedCG}将每个客户端的本地网络分解为私有提取器和公共分类器，并将提取器保留在本地以保护隐私。与暴露隐私泄露的罪魁祸首提取器不同，\textsc{FedCG}将客户端的生成器与服务器共享，用于聚合共同知识，旨在提高客户端本地网络的性能。大量实验表明，与基线FL方法相比，Textsc{FedCG}可以达到好胜模型的性能，数值隐私分析表明，Textsc{FedCG}具有较高的隐私保护能力。



## **25. 3D Adversarial Attacks Beyond Point Cloud**

超越点云的3D对抗性攻击 cs.CV

8 pages, 6 figs

**SubmitDate**: 2021-11-16    [paper-pdf](http://arxiv.org/pdf/2104.12146v3)

**Authors**: Jinlai Zhang, Lyujie Chen, Binbin Liu, Bo Ouyang, Qizhi Xie, Jihong Zhu, Weiming Li, Yanmei Meng

**Abstracts**: Recently, 3D deep learning models have been shown to be susceptible to adversarial attacks like their 2D counterparts. Most of the state-of-the-art (SOTA) 3D adversarial attacks perform perturbation to 3D point clouds. To reproduce these attacks in the physical scenario, a generated adversarial 3D point cloud need to be reconstructed to mesh, which leads to a significant drop in its adversarial effect. In this paper, we propose a strong 3D adversarial attack named Mesh Attack to address this problem by directly performing perturbation on mesh of a 3D object. In order to take advantage of the most effective gradient-based attack, a differentiable sample module that back-propagate the gradient of point cloud to mesh is introduced. To further ensure the adversarial mesh examples without outlier and 3D printable, three mesh losses are adopted. Extensive experiments demonstrate that the proposed scheme outperforms SOTA 3D attacks by a significant margin. We also achieved SOTA performance under various defenses. Our code is available at: https://github.com/cuge1995/Mesh-Attack.

摘要: 最近，3D深度学习模型被证明像2D模型一样容易受到敌意攻击。大多数最先进的三维对抗性攻击(SOTA)都是对三维点云进行扰动。为了在物理场景中再现这些攻击，需要将生成的对抗性三维点云重建为网格，这会导致其对抗性效果显著下降。为了解决这一问题，我们提出了一种强3D对抗性攻击，称为网格攻击，通过直接对3D对象的网格进行扰动来解决这一问题。为了利用最有效的基于梯度的攻击，引入了一种将点云梯度反向传播到网格的可微样本模块。为了进一步保证对抗性网格实例的无离群点和3D可打印，采用了三种网格损失。大量实验表明，该方案的性能明显优于SOTA3D攻击。我们还在不同的防守下取得了SOTA的表现。我们的代码可从以下网址获得：https://github.com/cuge1995/Mesh-Attack.



## **26. Augmenting Zero Trust Architecture to Endpoints Using Blockchain: A State-of-The-Art Review**

使用区块链将零信任架构扩展到端点：最新综述 cs.CR

(1) Fixed the reference numbering (2) Fixed syntax errors,  improvements (3) document re-structured

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2104.00460v4)

**Authors**: Lampis Alevizos, Vinh Thong Ta, Max Hashem Eiza

**Abstracts**: With the purpose of defending against lateral movement in today's borderless networks, Zero Trust Architecture (ZTA) adoption is gaining momentum. With a full scale ZTA implementation, it is unlikely that adversaries will be able to spread through the network starting from a compromised endpoint. However, the already authenticated and authorised session of a compromised endpoint can be leveraged to perform limited, though malicious, activities ultimately rendering the endpoints the Achilles heel of ZTA. To effectively detect such attacks, distributed collaborative intrusion detection systems with an attack scenario-based approach have been developed. Nonetheless, Advanced Persistent Threats (APTs) have demonstrated their ability to bypass this approach with a high success ratio. As a result, adversaries can pass undetected or potentially alter the detection logging mechanisms to achieve a stealthy presence. Recently, blockchain technology has demonstrated solid use cases in the cyber security domain. In this paper, motivated by the convergence of ZTA and blockchain-based intrusion detection and prevention, we examine how ZTA can be augmented onto endpoints. Namely, we perform a state-of-the-art review of ZTA models, real-world architectures with a focus on endpoints, and blockchain-based intrusion detection systems. We discuss the potential of blockchain's immutability fortifying the detection process and identify open challenges as well as potential solutions and future directions.

摘要: 为了防御当今无边界网络中的横向移动，零信任架构(ZTA)的采用势头正在增强。在全面实施ZTA的情况下，攻击者不太可能从受危害的端点开始通过网络传播。但是，可以利用受危害端点的已验证和授权会话来执行有限但恶意的活动，最终使端点成为ZTA的致命弱点。为了有效地检测此类攻击，基于攻击场景的分布式协同入侵检测系统应运而生。尽管如此，高级持续性威胁(APT)已证明它们有能力绕过此方法，成功率很高。因此，攻击者可以在未被检测到的情况下通过或潜在地更改检测日志记录机制，以实现隐蔽存在。最近，区块链技术在网络安全领域展示了坚实的使用案例。受ZTA和基于区块链的入侵检测与防御融合的激励，我们研究了如何将ZTA扩展到端点。也就是说，我们对ZTA模型、以端点为重点的现实世界架构和基于区块链的入侵检测系统进行了最先进的审查。我们讨论了区块链的不变性加强检测过程的潜力，并确定了开放的挑战以及潜在的解决方案和未来方向。



## **27. NNoculation: Catching BadNets in the Wild**

NNoculation：野外抓恶网 cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2002.08313v2)

**Authors**: Akshaj Kumar Veldanda, Kang Liu, Benjamin Tan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Brendan Dolan-Gavitt, Siddharth Garg

**Abstracts**: This paper proposes a novel two-stage defense (NNoculation) against backdoored neural networks (BadNets) that, repairs a BadNet both pre-deployment and online in response to backdoored test inputs encountered in the field. In the pre-deployment stage, NNoculation retrains the BadNet with random perturbations of clean validation inputs to partially reduce the adversarial impact of a backdoor. Post-deployment, NNoculation detects and quarantines backdoored test inputs by recording disagreements between the original and pre-deployment patched networks. A CycleGAN is then trained to learn transformations between clean validation and quarantined inputs; i.e., it learns to add triggers to clean validation images. Backdoored validation images along with their correct labels are used to further retrain the pre-deployment patched network, yielding our final defense. Empirical evaluation on a comprehensive suite of backdoor attacks show that NNoculation outperforms all state-of-the-art defenses that make restrictive assumptions and only work on specific backdoor attacks, or fail on adaptive attacks. In contrast, NNoculation makes minimal assumptions and provides an effective defense, even under settings where existing defenses are ineffective due to attackers circumventing their restrictive assumptions.

摘要: 提出了一种针对回溯神经网络(BadNets)的新的两阶段防御(NNoculation)，即对BadNet进行预部署和在线修复，以响应现场遇到的回溯测试输入。在部署前阶段，NNoculation使用干净验证输入的随机扰动重新训练BadNet，以部分降低后门的敌对影响。部署后，NNoculation通过记录原始修补网络和部署前修补网络之间的不一致来检测和隔离反向测试输入。然后，训练CycleGAN学习干净验证和隔离输入之间的转换；即，它学习向干净验证图像添加触发器。后置的验证映像及其正确的标签用于进一步重新训练部署前修补的网络，从而实现我们的最终防御。对一套全面的后门攻击的经验评估表明，NNoculation的性能优于所有做出限制性假设并仅在特定后门攻击上有效，或在自适应攻击上失败的最先进的防御措施。相比之下，NNoculation只做最少的假设并提供有效的防御，即使在现有防御因攻击者绕过其限制性假设而无效的情况下也是如此。



## **28. Generative Dynamic Patch Attack**

生成式动态补丁攻击 cs.CV

Published as a conference paper at BMVC 2021

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.04266v2)

**Authors**: Xiang Li, Shihao Ji

**Abstracts**: Adversarial patch attack is a family of attack algorithms that perturb a part of image to fool a deep neural network model. Existing patch attacks mostly consider injecting adversarial patches at input-agnostic locations: either a predefined location or a random location. This attack setup may be sufficient for attack but has considerable limitations when using it for adversarial training. Thus, robust models trained with existing patch attacks cannot effectively defend other adversarial attacks. In this paper, we first propose an end-to-end patch attack algorithm, Generative Dynamic Patch Attack (GDPA), which generates both patch pattern and patch location adversarially for each input image. We show that GDPA is a generic attack framework that can produce dynamic/static and visible/invisible patches with a few configuration changes. Secondly, GDPA can be readily integrated for adversarial training to improve model robustness to various adversarial attacks. Extensive experiments on VGGFace, Traffic Sign and ImageNet show that GDPA achieves higher attack success rates than state-of-the-art patch attacks, while adversarially trained model with GDPA demonstrates superior robustness to adversarial patch attacks than competing methods. Our source code can be found at https://github.com/lxuniverse/gdpa.

摘要: 对抗性补丁攻击是一系列攻击算法，通过扰动图像的一部分来欺骗深层神经网络模型。现有的补丁攻击大多考虑在与输入无关的位置(预定义位置或随机位置)注入敌意补丁。这种攻击设置对于攻击来说可能是足够的，但在用于对抗性训练时有相当大的限制。因此，用现有补丁攻击训练的鲁棒模型不能有效防御其他对抗性攻击。本文首先提出了一种端到端的补丁攻击算法--生成性动态补丁攻击(GDPA)，该算法对每幅输入图像分别生成补丁模式和补丁位置。我们证明了GDPA是一个通用的攻击框架，只需少量的配置更改，就可以生成动态/静电和可见/不可见的补丁。其次，GDPA可以很容易地集成到对抗性训练中，以提高模型对各种对抗性攻击的鲁棒性。在VGGFace、交通标志和ImageNet上的大量实验表明，GDPA比最新的补丁攻击具有更高的攻击成功率，而带有GDPA的对抗性训练模型对敌意补丁攻击表现出比竞争方法更好的鲁棒性。我们的源代码可以在https://github.com/lxuniverse/gdpa.上找到



## **29. Website fingerprinting on early QUIC traffic**

早期Quic流量的网站指纹分析 cs.CR

This work has been accepted by Elsevier Computer Networks for  publication

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2101.11871v2)

**Authors**: Pengwei Zhan, Liming Wang, Yi Tang

**Abstracts**: Cryptographic protocols have been widely used to protect the user's privacy and avoid exposing private information. QUIC (Quick UDP Internet Connections), including the version originally designed by Google (GQUIC) and the version standardized by IETF (IQUIC), as alternatives to the traditional HTTP, demonstrate their unique transmission characteristics: based on UDP for encrypted resource transmitting, accelerating web page rendering. However, existing encrypted transmission schemes based on TCP are vulnerable to website fingerprinting (WFP) attacks, allowing adversaries to infer the users' visited websites by eavesdropping on the transmission channel. Whether GQUIC and IQUIC can effectively resist such attacks is worth investigating. In this paper, we study the vulnerabilities of GQUIC, IQUIC, and HTTPS to WFP attacks from the perspective of traffic analysis. Extensive experiments show that, in the early traffic scenario, GQUIC is the most vulnerable to WFP attacks among GQUIC, IQUIC, and HTTPS, while IQUIC is more vulnerable than HTTPS, but the vulnerability of the three protocols is similar in the normal full traffic scenario. Features transferring analysis shows that most features are transferable between protocols when on normal full traffic scenario. However, combining with the qualitative analysis of latent feature representation, we find that the transferring is inefficient when on early traffic, as GQUIC, IQUIC, and HTTPS show the significantly different magnitude of variation in the traffic distribution on early traffic. By upgrading the one-time WFP attacks to multiple WFP Top-a attacks, we find that the attack accuracy on GQUIC and IQUIC reach 95.4% and 95.5%, respectively, with only 40 packets and just using simple features, whereas reach only 60.7% when on HTTPS. We also demonstrate that the vulnerability of IQUIC is only slightly dependent on the network environment.

摘要: 密码协议已被广泛用于保护用户隐私和避免泄露私人信息。Quic(Quick UDP Internet Connections，快速UDP Internet连接)，包括Google(GQUIC)最初设计的版本和IETF(IQUIC)标准化的版本，作为传统HTTP的替代品，展示了它们独特的传输特性：基于UDP进行加密资源传输，加速网页渲染。然而，现有的基于TCP的加密传输方案容易受到网站指纹识别(WFP)攻击，使得攻击者能够通过窃听传输通道来推断用户访问的网站。GQUIC和IQUIC能否有效抵御此类攻击值得研究。本文从流量分析的角度研究了GQUIC、IQUIC和HTTPS对WFP攻击的脆弱性。大量的实验表明，在早期流量场景中，GQUIC是GQUIC、IQUIC和HTTPS中最容易受到WFP攻击的协议，而IQUIC比HTTPS更容易受到攻击，但在正常的全流量场景下，这三种协议的漏洞是相似的。特征转移分析表明，在正常全流量场景下，协议间的大部分特征是可以转移的。然而，结合潜在特征表示的定性分析，我们发现在早期流量上的传输效率较低，因为GQUIC、IQUIC和HTTPS在早期流量上的流量分布表现出明显不同的变化幅度。通过将一次性的WFP攻击升级为多个WFP Top-a攻击，我们发现GQUIC和IQUIC的攻击准确率分别达到了95.4%和95.5%，只有40个数据包，而且只使用了简单的特征，而在HTTPS上的攻击准确率只有60.7%。我们还证明了IQUIC的漏洞对网络环境的依赖性很小。



## **30. Adversarial Detection Avoidance Attacks: Evaluating the robustness of perceptual hashing-based client-side scanning**

敌意检测规避攻击：评估基于感知散列的客户端扫描的健壮性 cs.CR

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2106.09820v2)

**Authors**: Shubham Jain, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstracts**: End-to-end encryption (E2EE) by messaging platforms enable people to securely and privately communicate with one another. Its widespread adoption however raised concerns that illegal content might now be shared undetected. Following the global pushback against key escrow systems, client-side scanning based on perceptual hashing has been recently proposed by tech companies, governments and researchers to detect illegal content in E2EE communications. We here propose the first framework to evaluate the robustness of perceptual hashing-based client-side scanning to detection avoidance attacks and show current systems to not be robust. More specifically, we propose three adversarial attacks--a general black-box attack and two white-box attacks for discrete cosine transform-based algorithms--against perceptual hashing algorithms. In a large-scale evaluation, we show perceptual hashing-based client-side scanning mechanisms to be highly vulnerable to detection avoidance attacks in a black-box setting, with more than 99.9\% of images successfully attacked while preserving the content of the image. We furthermore show our attack to generate diverse perturbations, strongly suggesting that straightforward mitigation strategies would be ineffective. Finally, we show that the larger thresholds necessary to make the attack harder would probably require more than one billion images to be flagged and decrypted daily, raising strong privacy concerns. Taken together, our results shed serious doubts on the robustness of perceptual hashing-based client-side scanning mechanisms currently proposed by governments, organizations, and researchers around the world.

摘要: 消息传递平台提供的端到端加密(E2EE)使人们能够安全、私密地相互通信。然而，它的广泛采用引发了人们的担忧，即非法内容现在可能会被分享而不被发现。继全球对密钥托管系统的抵制之后，科技公司、政府和研究人员最近提出了基于感知散列的客户端扫描，以检测E2EE通信中的非法内容。我们在这里提出了第一个框架来评估基于感知散列的客户端扫描对检测规避攻击的健壮性，并表明当前的系统是不健壮的。更具体地说，我们提出了三种针对感知散列算法的对抗性攻击--一种通用的黑盒攻击和两种基于离散余弦变换的算法的白盒攻击。在大规模的评估中，我们发现基于感知散列的客户端扫描机制在黑盒环境下非常容易受到检测回避攻击，在保护图像内容的同时，99.9%以上的图像被成功攻击。此外，我们还展示了我们的攻击会产生不同的扰动，这强烈地表明直接的缓解策略将是无效的。最后，我们指出，增加攻击难度所需的更大门槛可能需要每天标记和解密超过10亿张图像，这引发了强烈的隐私问题。综上所述，我们的结果对目前世界各地的政府、组织和研究人员提出的基于感知散列的客户端扫描机制的健壮性提出了严重的质疑。



## **31. Property Inference Attacks Against GANs**

针对GAN的属性推理攻击 cs.CR

To Appear in NDSS 2022

**SubmitDate**: 2021-11-15    [paper-pdf](http://arxiv.org/pdf/2111.07608v1)

**Authors**: Junhao Zhou, Yufei Chen, Chao Shen, Yang Zhang

**Abstracts**: While machine learning (ML) has made tremendous progress during the past decade, recent research has shown that ML models are vulnerable to various security and privacy attacks. So far, most of the attacks in this field focus on discriminative models, represented by classifiers. Meanwhile, little attention has been paid to the security and privacy risks of generative models, such as generative adversarial networks (GANs). In this paper, we propose the first set of training dataset property inference attacks against GANs. Concretely, the adversary aims to infer the macro-level training dataset property, i.e., the proportion of samples used to train a target GAN with respect to a certain attribute. A successful property inference attack can allow the adversary to gain extra knowledge of the target GAN's training dataset, thereby directly violating the intellectual property of the target model owner. Also, it can be used as a fairness auditor to check whether the target GAN is trained with a biased dataset. Besides, property inference can serve as a building block for other advanced attacks, such as membership inference. We propose a general attack pipeline that can be tailored to two attack scenarios, including the full black-box setting and partial black-box setting. For the latter, we introduce a novel optimization framework to increase the attack efficacy. Extensive experiments over four representative GAN models on five property inference tasks show that our attacks achieve strong performance. In addition, we show that our attacks can be used to enhance the performance of membership inference against GANs.

摘要: 虽然机器学习(ML)在过去的十年中取得了巨大的进步，但最近的研究表明，ML模型容易受到各种安全和隐私攻击。到目前为止，该领域的攻击大多集中在以分类器为代表的区分模型上。同时，生成性模型，如生成性对抗性网络(GANS)的安全和隐私风险也很少受到关注。在本文中，我们提出了第一组针对GANS的训练数据集属性推理攻击。具体地说，对手的目标是推断宏观级别的训练数据集属性，即用于训练目标GAN的样本相对于特定属性的比例。成功的属性推理攻击可以让攻击者获得目标GAN训练数据集的额外知识，从而直接侵犯目标模型所有者的知识产权。此外，它还可以用作公平性审核器，以检查目标GAN是否使用有偏差的数据集进行训练。此外，属性推理还可以作为构建挡路的平台，用于其他高级攻击，如成员身份推理。我们提出了一种通用攻击流水线，该流水线可以针对两种攻击场景进行定制，包括完全黑盒设置和部分黑盒设置。对于后者，我们引入了一种新的优化框架来提高攻击效率。在4个典型的GAN模型上对5个属性推理任务进行的大量实验表明，我们的攻击取得了很好的性能。此外，我们还证明了我们的攻击可以用来提高针对GANS的成员关系推理的性能。



## **32. Towards Interpretability of Speech Pause in Dementia Detection using Adversarial Learning**

对抗性学习在痴呆检测中言语停顿的可解释性研究 cs.CL

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07454v1)

**Authors**: Youxiang Zhu, Bang Tran, Xiaohui Liang, John A. Batsis, Robert M. Roth

**Abstracts**: Speech pause is an effective biomarker in dementia detection. Recent deep learning models have exploited speech pauses to achieve highly accurate dementia detection, but have not exploited the interpretability of speech pauses, i.e., what and how positions and lengths of speech pauses affect the result of dementia detection. In this paper, we will study the positions and lengths of dementia-sensitive pauses using adversarial learning approaches. Specifically, we first utilize an adversarial attack approach by adding the perturbation to the speech pauses of the testing samples, aiming to reduce the confidence levels of the detection model. Then, we apply an adversarial training approach to evaluate the impact of the perturbation in training samples on the detection model. We examine the interpretability from the perspectives of model accuracy, pause context, and pause length. We found that some pauses are more sensitive to dementia than other pauses from the model's perspective, e.g., speech pauses near to the verb "is". Increasing lengths of sensitive pauses or adding sensitive pauses leads the model inference to Alzheimer's Disease, while decreasing the lengths of sensitive pauses or deleting sensitive pauses leads to non-AD.

摘要: 言语停顿是检测痴呆的有效生物标志物。最近的深度学习模型利用语音停顿来实现高精度的痴呆症检测，但没有利用语音停顿的可解释性，即语音停顿的位置和长度如何以及如何影响痴呆症检测的结果。在本文中，我们将使用对抗性学习方法来研究痴呆症敏感停顿的位置和长度。具体地说，我们首先利用对抗性攻击方法，通过在测试样本的语音停顿中添加扰动来降低检测模型的置信度。然后，我们应用对抗性训练方法来评估训练样本中的扰动对检测模型的影响。我们从模型精度、暂停上下文和暂停长度的角度来检查可解释性。我们发现，从模型的角度来看，一些停顿比其他停顿对痴呆症更敏感，例如，动词“is”附近的言语停顿。增加敏感停顿的长度或增加敏感停顿会导致模型对阿尔茨海默病的推断，而减少敏感停顿的长度或删除敏感停顿会导致非AD。



## **33. Generating Band-Limited Adversarial Surfaces Using Neural Networks**

用神经网络生成带限对抗性曲面 cs.CV

**SubmitDate**: 2021-11-14    [paper-pdf](http://arxiv.org/pdf/2111.07424v1)

**Authors**: Roee Ben Shlomo, Yevgeniy Men, Ido Imanuel

**Abstracts**: Generating adversarial examples is the art of creating a noise that is added to an input signal of a classifying neural network, and thus changing the network's classification, while keeping the noise as tenuous as possible. While the subject is well-researched in the 2D regime, it is lagging behind in the 3D regime, i.e. attacking a classifying network that works on 3D point-clouds or meshes and, for example, classifies the pose of people's 3D scans. As of now, the vast majority of papers that describe adversarial attacks in this regime work by methods of optimization. In this technical report we suggest a neural network that generates the attacks. This network utilizes PointNet's architecture with some alterations. While the previous articles on which we based our work on have to optimize each shape separately, i.e. tailor an attack from scratch for each individual input without any learning, we attempt to create a unified model that can deduce the needed adversarial example with a single forward run.

摘要: 生成对抗性示例是创建噪声的艺术，该噪声被添加到分类神经网络的输入信号，从而改变网络的分类，同时保持噪声尽可能微弱。虽然这个主题在2D模式下研究得很好，但在3D模式下却落后了，即攻击工作在3D点云或网格上的分类网络，例如，对人们3D扫描的姿势进行分类。到目前为止，绝大多数描述该制度下的对抗性攻击的论文都是通过优化的方法来工作的。在这份技术报告中，我们建议使用神经网络来生成攻击。这个网络采用了PointNet的架构，但做了一些改动。虽然我们工作所基于的前面的文章必须分别优化每个形状，即在没有任何学习的情况下为每个单独的输入从头开始定制攻击，但我们试图创建一个统一的模型，它可以通过一次向前运行来推导出所需的对抗性示例。



## **34. Measuring the Contribution of Multiple Model Representations in Detecting Adversarial Instances**

测量多个模型表示在检测对抗性实例中的贡献 cs.LG

**SubmitDate**: 2021-11-13    [paper-pdf](http://arxiv.org/pdf/2111.07035v1)

**Authors**: Daniel Steinberg, Paul Munro

**Abstracts**: Deep learning models have been used for a wide variety of tasks. They are prevalent in computer vision, natural language processing, speech recognition, and other areas. While these models have worked well under many scenarios, it has been shown that they are vulnerable to adversarial attacks. This has led to a proliferation of research into ways that such attacks could be identified and/or defended against. Our goal is to explore the contribution that can be attributed to using multiple underlying models for the purpose of adversarial instance detection. Our paper describes two approaches that incorporate representations from multiple models for detecting adversarial examples. We devise controlled experiments for measuring the detection impact of incrementally utilizing additional models. For many of the scenarios we consider, the results show that performance increases with the number of underlying models used for extracting representations.

摘要: 深度学习模型已被广泛用于各种任务。它们广泛应用于计算机视觉、自然语言处理、语音识别等领域。虽然这些模型在许多情况下都工作得很好，但已经表明它们很容易受到对手的攻击。这导致了对如何识别和/或防御此类攻击的研究激增。我们的目标是探索可以归因于使用多个底层模型进行对抗性实例检测的贡献。我们的论文描述了两种方法，它们融合了来自多个模型的表示，用于检测对抗性示例。我们设计了对照实验来衡量增量利用额外模型的检测影响。对于我们考虑的许多场景，结果显示性能随着用于提取表示的底层模型数量的增加而提高。



## **35. Adversarially Robust Learning for Security-Constrained Optimal Power Flow**

安全约束最优潮流的对抗性鲁棒学习 math.OC

Accepted at Neural Information Processing Systems (NeurIPS) 2021

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06961v1)

**Authors**: Priya L. Donti, Aayushya Agarwal, Neeraj Vijay Bedmutha, Larry Pileggi, J. Zico Kolter

**Abstracts**: In recent years, the ML community has seen surges of interest in both adversarially robust learning and implicit layers, but connections between these two areas have seldom been explored. In this work, we combine innovations from these areas to tackle the problem of N-k security-constrained optimal power flow (SCOPF). N-k SCOPF is a core problem for the operation of electrical grids, and aims to schedule power generation in a manner that is robust to potentially k simultaneous equipment outages. Inspired by methods in adversarially robust training, we frame N-k SCOPF as a minimax optimization problem - viewing power generation settings as adjustable parameters and equipment outages as (adversarial) attacks - and solve this problem via gradient-based techniques. The loss function of this minimax problem involves resolving implicit equations representing grid physics and operational decisions, which we differentiate through via the implicit function theorem. We demonstrate the efficacy of our framework in solving N-3 SCOPF, which has traditionally been considered as prohibitively expensive to solve given that the problem size depends combinatorially on the number of potential outages.

摘要: 近些年来，ML社区看到了对相反的健壮学习和隐含层的兴趣激增，但这两个领域之间的联系很少被探索。在这项工作中，我们结合这些领域的创新成果，提出了撞击求解N-k安全约束最优潮流问题。n-k SCOPF是电网运行的核心问题，其目标是以一种对潜在的k个同时设备故障具有鲁棒性的方式来调度发电。受对抗性鲁棒训练方法的启发，我们将N-k SCOPF定义为一个极小极大优化问题--将发电设置视为可调参数，将设备故障视为(对抗性)攻击--并通过基于梯度的技术解决该问题。这个极小极大问题的损失函数涉及到求解代表网格物理和操作决策的隐式方程，我们通过隐函数定理来区分这些方程。我们证明了我们的框架在解决N-3 SCOPF方面的有效性，传统上认为解决N-3 SCOPF的成本高得令人望而却步，因为问题的大小组合地取决于潜在的中断次数。



## **36. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知散列：用例NeuralHash cs.LG

22 pages, 15 figures, 5 tables

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2111.06628v1)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstracts**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知散列系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知散列的综合实证分析。具体地说，我们表明当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的改变来操纵散列值，这些改变要么是由基于梯度的方法引起的，要么是简单地通过执行标准图像转换来强制或防止散列冲突。这样的攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **37. Characterizing and Improving the Robustness of Self-Supervised Learning through Background Augmentations**

通过背景增强来表征和提高自监督学习的鲁棒性 cs.CV

Technical Report; Additional Results

**SubmitDate**: 2021-11-12    [paper-pdf](http://arxiv.org/pdf/2103.12719v2)

**Authors**: Chaitanya K. Ryali, David J. Schwab, Ari S. Morcos

**Abstracts**: Recent progress in self-supervised learning has demonstrated promising results in multiple visual tasks. An important ingredient in high-performing self-supervised methods is the use of data augmentation by training models to place different augmented views of the same image nearby in embedding space. However, commonly used augmentation pipelines treat images holistically, ignoring the semantic relevance of parts of an image-e.g. a subject vs. a background-which can lead to the learning of spurious correlations. Our work addresses this problem by investigating a class of simple, yet highly effective "background augmentations", which encourage models to focus on semantically-relevant content by discouraging them from focusing on image backgrounds. Through a systematic investigation, we show that background augmentations lead to substantial improvements in performance across a spectrum of state-of-the-art self-supervised methods (MoCo-v2, BYOL, SwAV) on a variety of tasks, e.g. $\sim$+1-2% gains on ImageNet, enabling performance on par with the supervised baseline. Further, we find the improvement in limited-labels settings is even larger (up to 4.2%). Background augmentations also improve robustness to a number of distribution shifts, including natural adversarial examples, ImageNet-9, adversarial attacks, ImageNet-Renditions. We also make progress in completely unsupervised saliency detection, in the process of generating saliency masks used for background augmentations.

摘要: 自我监督学习的最新进展在多视觉任务中显示出良好的结果。高性能自监督方法的一个重要组成部分是通过训练模型使用数据增强来将同一图像的不同增强视图放置在嵌入空间的附近。然而，常用的增强流水线从整体上对待图像，忽略了图像各部分的语义相关性。主题与背景--这可能导致学习虚假的相关性。我们的工作通过调查一类简单但高效的“背景增强”来解决这个问题，这种“背景增强”通过阻止模型关注图像背景来鼓励模型专注于语义相关的内容。通过系统调查，我们发现背景增强在各种任务(例如$\sim$+ImageNet$\sim$+1-2%)的一系列最先进的自我监督方法(MoCo-v2、BYOL、SwAV)上显著提高了性能，使性能与监督基线持平。此外，我们发现限制标签设置的改进更大(高达4.2%)。背景增强还提高了对许多分布变化的健壮性，包括自然对抗性示例、ImageNet-9、对抗性攻击、ImageNet-Renditions。在生成用于背景增强的显著掩码的过程中，我们还在完全无监督的显著性检测方面取得了进展。



## **38. Distributionally Robust Trajectory Optimization Under Uncertain Dynamics via Relative Entropy Trust-Regions**

基于相对熵信赖域的不确定动态分布鲁棒轨迹优化 eess.SY

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2103.15388v3)

**Authors**: Hany Abdulsamad, Tim Dorau, Boris Belousov, Jia-Jie Zhu, Jan Peters

**Abstracts**: Trajectory optimization and model predictive control are essential techniques underpinning advanced robotic applications, ranging from autonomous driving to full-body humanoid control. State-of-the-art algorithms have focused on data-driven approaches that infer the system dynamics online and incorporate posterior uncertainty during planning and control. Despite their success, such approaches are still susceptible to catastrophic errors that may arise due to statistical learning biases, unmodeled disturbances, or even directed adversarial attacks. In this paper, we tackle the problem of dynamics mismatch and propose a distributionally robust optimal control formulation that alternates between two relative entropy trust-region optimization problems. Our method finds the worst-case maximum entropy Gaussian posterior over the dynamics parameters and the corresponding robust policy. Furthermore, we show that our approach admits a closed-form backward-pass for a certain class of systems. Finally, we demonstrate the resulting robustness on linear and nonlinear numerical examples.

摘要: 轨迹优化和模型预测控制是支撑先进机器人应用的关键技术，从自动驾驶到全身仿人控制。最先进的算法专注于数据驱动的方法，这些方法在线推断系统动态，并在计划和控制过程中纳入后验不确定性。尽管这些方法取得了成功，但它们仍然容易受到灾难性错误的影响，这些错误可能是由于统计学习偏差、未建模的干扰，甚至是定向的对抗性攻击而产生的。本文对动态失配问题进行了撞击研究，提出了一种在两个相对熵信赖域优化问题之间交替的分布鲁棒最优控制公式。我们的方法求出了动力学参数的最坏情况下的最大熵高斯后验分布，并给出了相应的鲁棒策略。此外，我们还证明了我们的方法对于某一类系统允许闭合形式的后向传递(Closed-Form Back-Pass)。最后，我们在线性和非线性数值算例上证明了所得结果的鲁棒性。



## **39. Poisoning Knowledge Graph Embeddings via Relation Inference Patterns**

通过关系推理模式毒化知识图嵌入 cs.LG

Joint Conference of the 59th Annual Meeting of the Association for  Computational Linguistics and the 11th International Joint Conference on  Natural Language Processing (ACL-IJCNLP 2021)

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2111.06345v1)

**Authors**: Peru Bhardwaj, John Kelleher, Luca Costabello, Declan O'Sullivan

**Abstracts**: We study the problem of generating data poisoning attacks against Knowledge Graph Embedding (KGE) models for the task of link prediction in knowledge graphs. To poison KGE models, we propose to exploit their inductive abilities which are captured through the relationship patterns like symmetry, inversion and composition in the knowledge graph. Specifically, to degrade the model's prediction confidence on target facts, we propose to improve the model's prediction confidence on a set of decoy facts. Thus, we craft adversarial additions that can improve the model's prediction confidence on decoy facts through different inference patterns. Our experiments demonstrate that the proposed poisoning attacks outperform state-of-art baselines on four KGE models for two publicly available datasets. We also find that the symmetry pattern based attacks generalize across all model-dataset combinations which indicates the sensitivity of KGE models to this pattern.

摘要: 研究了针对知识图中链接预测任务的知识图嵌入(KGE)模型产生数据中毒攻击的问题。为了毒化KGE模型，我们提出开发KGE模型的归纳能力，这些能力是通过知识图中的对称性、反转和合成等关系模式捕捉到的。具体地说，为了降低模型对目标事实的预测置信度，我们提出了提高模型对一组诱饵事实的预测置信度的方法。因此，我们设计了对抗性的加法，通过不同的推理模式来提高模型对诱饵事实的预测置信度。我们的实验表明，提出的中毒攻击在两个公开可用的数据集的四个KGE模型上的性能优于最新的基线。我们还发现，基于对称模式的攻击在所有模型-数据集组合中都是通用的，这表明了KGE模型对该模式的敏感性。



## **40. Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes**

Qu反化：利用量化伪像实现对抗性结果 cs.LG

Accepted to NeurIPS 2021 [Poster]

**SubmitDate**: 2021-11-11    [paper-pdf](http://arxiv.org/pdf/2110.13541v2)

**Authors**: Sanghyun Hong, Michael-Andrei Panaitescu-Liess, Yiğitcan Kaya, Tudor Dumitraş

**Abstracts**: Quantization is a popular technique that $transforms$ the parameter representation of a neural network from floating-point numbers into lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint and the computational cost at inference, facilitating the deployment of resource-hungry models. However, the parameter perturbations caused by this transformation result in $behavioral$ $disparities$ between the model before and after quantization. For example, a quantized model can misclassify some test-time samples that are otherwise classified correctly. It is not known whether such differences lead to a new security vulnerability. We hypothesize that an adversary may control this disparity to introduce specific behaviors that activate upon quantization. To study this hypothesis, we weaponize quantization-aware training and propose a new training framework to implement adversarial quantization outcomes. Following this framework, we present three attacks we carry out with quantization: (i) an indiscriminate attack for significant accuracy loss; (ii) a targeted attack against specific samples; and (iii) a backdoor attack for controlling the model with an input trigger. We further show that a single compromised model defeats multiple quantization schemes, including robust quantization techniques. Moreover, in a federated learning scenario, we demonstrate that a set of malicious participants who conspire can inject our quantization-activated backdoor. Lastly, we discuss potential counter-measures and show that only re-training consistently removes the attack artifacts. Our code is available at https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation

摘要: 量化是一种流行的技术，它将神经网络的参数表示从浮点数转换为低精度数字(例如$，8位整数)。它减少了推理时的内存占用和计算成本，便于部署资源匮乏的模型。然而，这种变换引起的参数扰动导致了量化前后模型之间的$行为$$差异$。例如，量化模型可能会错误分类一些本来可以正确分类的测试时间样本。目前尚不清楚这种差异是否会导致新的安全漏洞。我们假设对手可以控制这种差异，以引入量化后激活的特定行为。为了研究这一假设，我们将量化意识训练武器化，并提出了一个新的训练框架来实现对抗性量化结果。在此框架下，我们提出了三种量化的攻击：(I)不加区别的攻击，造成显著的精度损失；(Ii)针对特定样本的有针对性的攻击；以及(Iii)用输入触发器控制模型的后门攻击。我们进一步表明，单一的折衷模型击败了包括鲁棒量化技术在内的多个量化方案。此外，在联合学习场景中，我们演示了一组合谋的恶意参与者可以注入我们的量化激活后门。最后，我们讨论了潜在的对策，并表明只有重新训练才能始终如一地移除攻击伪影。我们的代码可在https://github.com/Secure-AI-Systems-Group/Qu-ANTI-zation获得



## **41. Robust Deep Reinforcement Learning through Adversarial Loss**

对抗性损失下的鲁棒深度强化学习 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2008.01976v2)

**Authors**: Tuomas Oikarinen, Wang Zhang, Alexandre Megretski, Luca Daniel, Tsui-Wei Weng

**Abstracts**: Recent studies have shown that deep reinforcement learning agents are vulnerable to small adversarial perturbations on the agent's inputs, which raises concerns about deploying such agents in the real world. To address this issue, we propose RADIAL-RL, a principled framework to train reinforcement learning agents with improved robustness against $l_p$-norm bounded adversarial attacks. Our framework is compatible with popular deep reinforcement learning algorithms and we demonstrate its performance with deep Q-learning, A3C and PPO. We experiment on three deep RL benchmarks (Atari, MuJoCo and ProcGen) to show the effectiveness of our robust training algorithm. Our RADIAL-RL agents consistently outperform prior methods when tested against attacks of varying strength and are more computationally efficient to train. In addition, we propose a new evaluation method called Greedy Worst-Case Reward (GWC) to measure attack agnostic robustness of deep RL agents. We show that GWC can be evaluated efficiently and is a good estimate of the reward under the worst possible sequence of adversarial attacks. All code used for our experiments is available at https://github.com/tuomaso/radial_rl_v2.

摘要: 最近的研究表明，深度强化学习Agent容易受到Agent输入的小的对抗性扰动，这引起了人们对在现实世界中部署此类Agent的担忧。为了解决这个问题，我们提出了RADIUS-RL，一个原则性的框架来训练强化学习Agent，使其对$l_p$-范数有界的攻击具有更好的鲁棒性。我们的框架与流行的深度强化学习算法兼容，并通过深度Q-学习、A3C和PPO验证了其性能。我们在三个深度RL基准(Atari、MuJoCo和ProcGen)上进行了实验，以验证我们的鲁棒训练算法的有效性。我们的Radius-RL代理在针对不同强度的攻击进行测试时，性能一直优于以前的方法，并且在训练时的计算效率更高。此外，我们还提出了一种新的评估方法--贪婪最坏情况奖励(GWC)来度量深度RL代理的攻击不可知性。我们表明，GWC可以被有效地评估，并且是在可能的最坏的对抗性攻击序列下的一个很好的奖励估计。我们实验使用的所有代码都可以在https://github.com/tuomaso/radial_rl_v2.上找到



## **42. Trustworthy Medical Segmentation with Uncertainty Estimation**

基于不确定性估计的可信医学分割 eess.IV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05978v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Nidhal C. Bouaynaya, Rasool Ghulam, Hassan M. Fathallah-Shaykh

**Abstracts**: Deep Learning (DL) holds great promise in reshaping the healthcare systems given its precision, efficiency, and objectivity. However, the brittleness of DL models to noisy and out-of-distribution inputs is ailing their deployment in the clinic. Most systems produce point estimates without further information about model uncertainty or confidence. This paper introduces a new Bayesian deep learning framework for uncertainty quantification in segmentation neural networks, specifically encoder-decoder architectures. The proposed framework uses the first-order Taylor series approximation to propagate and learn the first two moments (mean and covariance) of the distribution of the model parameters given the training data by maximizing the evidence lower bound. The output consists of two maps: the segmented image and the uncertainty map of the segmentation. The uncertainty in the segmentation decisions is captured by the covariance matrix of the predictive distribution. We evaluate the proposed framework on medical image segmentation data from Magnetic Resonances Imaging and Computed Tomography scans. Our experiments on multiple benchmark datasets demonstrate that the proposed framework is more robust to noise and adversarial attacks as compared to state-of-the-art segmentation models. Moreover, the uncertainty map of the proposed framework associates low confidence (or equivalently high uncertainty) to patches in the test input images that are corrupted with noise, artifacts or adversarial attacks. Thus, the model can self-assess its segmentation decisions when it makes an erroneous prediction or misses part of the segmentation structures, e.g., tumor, by presenting higher values in the uncertainty map.

摘要: 深度学习(DL)由于其精确性、效率和客观性，在重塑医疗系统方面有着巨大的希望。然而，DL模型对噪声和非分布输入的脆性阻碍了它们在临床上的部署。大多数系统在没有关于模型不确定性或置信度的进一步信息的情况下产生点估计。本文介绍了一种新的贝叶斯深度学习框架，用于分段神经网络中的不确定性量化，特别是编解码器的体系结构。该框架使用一阶泰勒级数近似，通过最大化证据下界来传播和学习给定训练数据的模型参数分布的前两个矩(均值和协方差)。输出由两幅图组成：分割后的图像和分割的不确定性图。通过预测分布的协方差矩阵来捕捉分割决策中的不确定性。我们在磁共振成像和计算机断层扫描的医学图像分割数据上对所提出的框架进行了评估。我们在多个基准数据集上的实验表明，与现有的分割模型相比，该框架对噪声和敌意攻击具有更强的鲁棒性。此外，该框架的不确定性图将低置信度(或相当于高不确定性)与测试输入图像中被噪声、伪影或敌意攻击破坏的补丁相关联。因此，当模型做出错误的预测或通过在不确定性图中呈现更高的值来错过部分分割结构(例如，肿瘤)时，该模型可以自我评估其分割决策。



## **43. Robust Learning via Ensemble Density Propagation in Deep Neural Networks**

基于集成密度传播的深度神经网络鲁棒学习 cs.LG

submitted to 2020 IEEE International Workshop on Machine Learning for  Signal Processing

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05953v1)

**Authors**: Giuseppina Carannante, Dimah Dera, Ghulam Rasool, Nidhal C. Bouaynaya, Lyudmila Mihaylova

**Abstracts**: Learning in uncertain, noisy, or adversarial environments is a challenging task for deep neural networks (DNNs). We propose a new theoretically grounded and efficient approach for robust learning that builds upon Bayesian estimation and Variational Inference. We formulate the problem of density propagation through layers of a DNN and solve it using an Ensemble Density Propagation (EnDP) scheme. The EnDP approach allows us to propagate moments of the variational probability distribution across the layers of a Bayesian DNN, enabling the estimation of the mean and covariance of the predictive distribution at the output of the model. Our experiments using MNIST and CIFAR-10 datasets show a significant improvement in the robustness of the trained models to random noise and adversarial attacks.

摘要: 对于深度神经网络(DNNs)来说，在不确定、噪声或敌对环境中学习是一项具有挑战性的任务。在贝叶斯估计和变分推理的基础上，提出了一种新的具有理论基础的、高效的鲁棒学习方法。我们用集合密度传播(ENDP)方案描述了DNN各层间的密度传播问题，并对其进行了求解。ENDP方法允许我们在贝叶斯DNN的各层之间传播变分概率分布的矩，从而能够在模型的输出处估计预测分布的均值和协方差。我们使用MNIST和CIFAR-10数据集进行的实验表明，训练后的模型对随机噪声和敌意攻击的鲁棒性有了显着的提高。



## **44. Audio Attacks and Defenses against AED Systems -- A Practical Study**

AED系统的音频攻击与防御--一项实用研究 cs.SD

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2106.07428v4)

**Authors**: Rodrigo dos Santos, Shirin Nilizadeh

**Abstracts**: In this paper, we evaluate deep learning-enabled AED systems against evasion attacks based on adversarial examples. We test the robustness of multiple security critical AED tasks, implemented as CNNs classifiers, as well as existing third-party Nest devices, manufactured by Google, which run their own black-box deep learning models. Our adversarial examples use audio perturbations made of white and background noises. Such disturbances are easy to create, to perform and to reproduce, and can be accessible to a large number of potential attackers, even non-technically savvy ones.   We show that an adversary can focus on audio adversarial inputs to cause AED systems to misclassify, achieving high success rates, even when we use small levels of a given type of noisy disturbance. For instance, on the case of the gunshot sound class, we achieve nearly 100% success rate when employing as little as 0.05 white noise level. Similarly to what has been previously done by works focusing on adversarial examples from the image domain as well as on the speech recognition domain. We then, seek to improve classifiers' robustness through countermeasures. We employ adversarial training and audio denoising. We show that these countermeasures, when applied to audio input, can be successful, either in isolation or in combination, generating relevant increases of nearly fifty percent in the performance of the classifiers when these are under attack.

摘要: 在这篇文章中，我们评估了深度学习使能的AED系统对逃避攻击的抵抗能力，这是基于敌意的例子。我们测试了多个安全关键AED任务(实现为CNNS分类器)以及由Google制造的现有第三方Nest设备的健壮性，这些设备运行自己的黑盒深度学习模型。我们的对抗性例子使用由白噪声和背景噪声构成的音频扰动。这样的干扰很容易制造、执行和复制，而且可以被大量潜在的攻击者接触到，即使是不懂技术的人也可以接触到。我们表明，即使当我们使用少量的给定类型的噪声干扰时，对手也可以专注于音频对手输入，导致AED系统错误分类，从而实现高成功率。例如，在枪声类的情况下，当使用低至0.05的白噪声水平时，我们获得了近100%的成功率。类似于先前通过关注来自图像域以及语音识别领域的对抗性示例的作品所做的工作。然后，通过对策寻求提高分类器的鲁棒性。我们采用对抗性训练和音频去噪。我们表明，当这些对策应用于音频输入时，无论是单独应用还是组合应用，都可以取得成功，当分类器受到攻击时，分类器的性能会相应提高近50%。



## **45. A black-box adversarial attack for poisoning clustering**

一种针对中毒聚类的黑盒对抗性攻击 cs.LG

18 pages, Pattern Recognition 2022

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2009.05474v4)

**Authors**: Antonio Emanuele Cinà, Alessandro Torcinovich, Marcello Pelillo

**Abstracts**: Clustering algorithms play a fundamental role as tools in decision-making and sensible automation processes. Due to the widespread use of these applications, a robustness analysis of this family of algorithms against adversarial noise has become imperative. To the best of our knowledge, however, only a few works have currently addressed this problem. In an attempt to fill this gap, in this work, we propose a black-box adversarial attack for crafting adversarial samples to test the robustness of clustering algorithms. We formulate the problem as a constrained minimization program, general in its structure and customizable by the attacker according to her capability constraints. We do not assume any information about the internal structure of the victim clustering algorithm, and we allow the attacker to query it as a service only. In the absence of any derivative information, we perform the optimization with a custom approach inspired by the Abstract Genetic Algorithm (AGA). In the experimental part, we demonstrate the sensibility of different single and ensemble clustering algorithms against our crafted adversarial samples on different scenarios. Furthermore, we perform a comparison of our algorithm with a state-of-the-art approach showing that we are able to reach or even outperform its performance. Finally, to highlight the general nature of the generated noise, we show that our attacks are transferable even against supervised algorithms such as SVMs, random forests, and neural networks.

摘要: 聚类算法在决策和明智的自动化过程中起着基础性的作用。由于这些应用的广泛使用，对这类算法进行抗对抗噪声的鲁棒性分析已成为当务之急。然而，就我们所知，目前只有几部著作解决了这个问题。为了填补这一空白，在这项工作中，我们提出了一种用于制作敌意样本的黑盒对抗性攻击，以测试聚类算法的健壮性。我们将问题描述为一个有约束的最小化规划，其结构一般，攻击者可以根据她的能力约束进行定制。我们不假定有关受害者群集算法的内部结构的任何信息，并且我们仅允许攻击者将其作为服务进行查询。在没有任何导数信息的情况下，受抽象遗传算法(AGA)的启发，采用定制的方法进行优化。在实验部分，我们展示了不同的单一聚类算法和集成聚类算法在不同场景下对我们制作的敌意样本的敏感度。此外，我们将我们的算法与最先进的方法进行了比较，结果表明我们能够达到甚至超过它的性能。最后，为了突出生成噪声的一般性质，我们证明了我们的攻击即使是针对支持向量机、随机森林和神经网络等有监督算法也是可以转移的。



## **46. Universal Multi-Party Poisoning Attacks**

普遍存在的多方中毒攻击 cs.LG

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/1809.03474v3)

**Authors**: Saeed Mahloujifar, Mohammad Mahmoody, Ameer Mohammed

**Abstracts**: In this work, we demonstrate universal multi-party poisoning attacks that adapt and apply to any multi-party learning process with arbitrary interaction pattern between the parties. More generally, we introduce and study $(k,p)$-poisoning attacks in which an adversary controls $k\in[m]$ of the parties, and for each corrupted party $P_i$, the adversary submits some poisoned data $\mathcal{T}'_i$ on behalf of $P_i$ that is still ``$(1-p)$-close'' to the correct data $\mathcal{T}_i$ (e.g., $1-p$ fraction of $\mathcal{T}'_i$ is still honestly generated). We prove that for any ``bad'' property $B$ of the final trained hypothesis $h$ (e.g., $h$ failing on a particular test example or having ``large'' risk) that has an arbitrarily small constant probability of happening without the attack, there always is a $(k,p)$-poisoning attack that increases the probability of $B$ from $\mu$ to by $\mu^{1-p \cdot k/m} = \mu + \Omega(p \cdot k/m)$. Our attack only uses clean labels, and it is online.   More generally, we prove that for any bounded function $f(x_1,\dots,x_n) \in [0,1]$ defined over an $n$-step random process $\mathbf{X} = (x_1,\dots,x_n)$, an adversary who can override each of the $n$ blocks with even dependent probability $p$ can increase the expected output by at least $\Omega(p \cdot \mathrm{Var}[f(\mathbf{x})])$.

摘要: 在这项工作中，我们展示了通用的多方中毒攻击，适用于任何具有任意交互模式的多方学习过程。更一般地，我们引入和研究了$(k，p)$中毒攻击，在这种攻击中，敌手控制着当事人的$k\in[m]$，并且对于每个被破坏的一方$P_i$，对手代表$P_i$提交一些有毒数据$\数学{T}‘_i$，该数据仍然是’‘$(1-p)$-接近’‘正确数据$\数学{T}_i$(例如，$1-p$分数$\我们证明了对于最终训练假设$h$的任何“坏”性质$B$(例如，$h$在特定的测试用例上失败或具有“大的”风险)，在没有攻击的情况下发生的概率任意小的恒定概率，总是存在$(k，p)$中毒攻击，它将$B$的概率从$\µ$增加到$\µ^{1-p\CDOT k/m}=\Mu+\Omega(p\CDOT k/m)$。我们的攻击只使用干净的标签，而且是在线的。更一般地，我们证明了对于定义在$n$步随机过程$\mathbf{X}=(x_1，\点，x_n)$上的[0，1]$中的任何有界函数$f(x_1，\dots，x_n)\，能够以偶数依赖概率$p$覆盖$n$块中的每个块的对手可以使期望输出至少增加$\Omega(p\cdot\mathm{var}[f(\mam



## **47. Sparse Adversarial Video Attacks with Spatial Transformations**

基于空间变换的稀疏对抗性视频攻击 cs.CV

The short version of this work will appear in the BMVC 2021  conference

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05468v1)

**Authors**: Ronghui Mu, Wenjie Ruan, Leandro Soriano Marcolino, Qiang Ni

**Abstracts**: In recent years, a significant amount of research efforts concentrated on adversarial attacks on images, while adversarial video attacks have seldom been explored. We propose an adversarial attack strategy on videos, called DeepSAVA. Our model includes both additive perturbation and spatial transformation by a unified optimisation framework, where the structural similarity index (SSIM) measure is adopted to measure the adversarial distance. We design an effective and novel optimisation scheme which alternatively utilizes Bayesian optimisation to identify the most influential frame in a video and Stochastic gradient descent (SGD) based optimisation to produce both additive and spatial-transformed perturbations. Doing so enables DeepSAVA to perform a very sparse attack on videos for maintaining human imperceptibility while still achieving state-of-the-art performance in terms of both attack success rate and adversarial transferability. Our intensive experiments on various types of deep neural networks and video datasets confirm the superiority of DeepSAVA.

摘要: 近年来，大量的研究工作集中在图像的对抗性攻击上，而对抗性视频攻击的研究很少。我们提出了一种针对视频的对抗性攻击策略，称为DeepSAVA。我们的模型通过一个统一的优化框架同时包括加性扰动和空间变换，其中采用结构相似指数(SSIM)度量对抗距离。我们设计了一种有效和新颖的优化方案，它交替使用贝叶斯优化来识别视频中最有影响力的帧，以及基于随机梯度下降(SGD)的优化来产生加性和空间变换的扰动。这样做使DeepSAVA能够对视频执行非常稀疏的攻击，以保持人的不可感知性，同时在攻击成功率和对手可转移性方面仍获得最先进的性能。我们在不同类型的深度神经网络和视频数据集上的密集实验证实了DeepSAVA的优越性。



## **48. Are Transformers More Robust Than CNNs?**

变形金刚比CNN更健壮吗？ cs.CV

**SubmitDate**: 2021-11-10    [paper-pdf](http://arxiv.org/pdf/2111.05464v1)

**Authors**: Yutong Bai, Jieru Mei, Alan Yuille, Cihang Xie

**Abstracts**: Transformer emerges as a powerful tool for visual recognition. In addition to demonstrating competitive performance on a broad range of visual benchmarks, recent works also argue that Transformers are much more robust than Convolutions Neural Networks (CNNs). Nonetheless, surprisingly, we find these conclusions are drawn from unfair experimental settings, where Transformers and CNNs are compared at different scales and are applied with distinct training frameworks. In this paper, we aim to provide the first fair & in-depth comparisons between Transformers and CNNs, focusing on robustness evaluations.   With our unified training setup, we first challenge the previous belief that Transformers outshine CNNs when measuring adversarial robustness. More surprisingly, we find CNNs can easily be as robust as Transformers on defending against adversarial attacks, if they properly adopt Transformers' training recipes. While regarding generalization on out-of-distribution samples, we show pre-training on (external) large-scale datasets is not a fundamental request for enabling Transformers to achieve better performance than CNNs. Moreover, our ablations suggest such stronger generalization is largely benefited by the Transformer's self-attention-like architectures per se, rather than by other training setups. We hope this work can help the community better understand and benchmark the robustness of Transformers and CNNs. The code and models are publicly available at https://github.com/ytongbai/ViTs-vs-CNNs.

摘要: 变压器作为一种强有力的视觉识别工具应运而生。除了展示好胜在广泛的可视基准上的性能外，最近的研究还认为，变形金刚比卷积神经网络(CNN)更健壮。然而，令人惊讶的是，我们发现这些结论是从不公平的实验环境中得出的，在这些实验环境中，变形金刚和CNN在不同的尺度上进行了比较，并应用了不同的训练框架。在本文中，我们的目标是提供变压器和CNN之间的第一次公平和深入的比较，重点是鲁棒性评估。有了我们的统一训练设置，我们首先挑战了以前的信念，即在衡量对手的健壮性时，变形金刚优于CNN。更令人惊讶的是，我们发现，如果CNN恰当地采用了变形金刚的训练食谱，它们在防御对手攻击方面可以很容易地像变形金刚一样健壮。虽然关于分布外样本的泛化，我们表明(外部)大规模数据集的预训练并不是使Transformers获得比CNN更好的性能的基本要求。此外，我们的消融表明，这种更强的概括性在很大程度上得益于变形金刚的自我关注式架构本身，而不是其他培训设置。我们希望这项工作可以帮助社区更好地理解Transformers和CNN的健壮性，并对其进行基准测试。代码和模型可在https://github.com/ytongbai/ViTs-vs-CNNs.上公开获得



## **49. Statistical Perspectives on Reliability of Artificial Intelligence Systems**

人工智能系统可靠性的统计透视 cs.SE

40 pages

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2111.05391v1)

**Authors**: Yili Hong, Jiayi Lian, Li Xu, Jie Min, Yueyao Wang, Laura J. Freeman, Xinwei Deng

**Abstracts**: Artificial intelligence (AI) systems have become increasingly popular in many areas. Nevertheless, AI technologies are still in their developing stages, and many issues need to be addressed. Among those, the reliability of AI systems needs to be demonstrated so that the AI systems can be used with confidence by the general public. In this paper, we provide statistical perspectives on the reliability of AI systems. Different from other considerations, the reliability of AI systems focuses on the time dimension. That is, the system can perform its designed functionality for the intended period. We introduce a so-called SMART statistical framework for AI reliability research, which includes five components: Structure of the system, Metrics of reliability, Analysis of failure causes, Reliability assessment, and Test planning. We review traditional methods in reliability data analysis and software reliability, and discuss how those existing methods can be transformed for reliability modeling and assessment of AI systems. We also describe recent developments in modeling and analysis of AI reliability and outline statistical research challenges in this area, including out-of-distribution detection, the effect of the training set, adversarial attacks, model accuracy, and uncertainty quantification, and discuss how those topics can be related to AI reliability, with illustrative examples. Finally, we discuss data collection and test planning for AI reliability assessment and how to improve system designs for higher AI reliability. The paper closes with some concluding remarks.

摘要: 人工智能(AI)系统在许多领域变得越来越受欢迎。然而，人工智能技术仍处于发展阶段，许多问题需要解决。其中，需要证明人工智能系统的可靠性，以便普通公众可以放心地使用人工智能系统。在这篇文章中，我们提供了关于人工智能系统可靠性的统计观点。与其他考虑因素不同，人工智能系统的可靠性侧重于时间维度。也就是说，系统可以在预期的时间段内执行其设计的功能。介绍了一种用于人工智能可靠性研究的智能统计框架，该框架包括五个组成部分：系统结构、可靠性度量、故障原因分析、可靠性评估和测试规划。我们回顾了可靠性数据分析和软件可靠性的传统方法，并讨论了如何将这些现有方法转化为人工智能系统的可靠性建模和评估。我们还描述了人工智能可靠性建模和分析的最新进展，并概述了该领域的统计研究挑战，包括分布失调检测、训练集的影响、对抗性攻击、模型精度和不确定性量化，并用说明性例子讨论了这些主题如何与人工智能可靠性相关。最后，我们讨论了人工智能可靠性评估的数据收集和测试规划，以及如何改进系统设计以提高人工智能可靠性。论文最后以一些结束语结束。



## **50. TDGIA:Effective Injection Attacks on Graph Neural Networks**

TDGIA：对图神经网络的有效注入攻击 cs.LG

KDD 2021 research track paper

**SubmitDate**: 2021-11-09    [paper-pdf](http://arxiv.org/pdf/2106.06663v2)

**Authors**: Xu Zou, Qinkai Zheng, Yuxiao Dong, Xinyu Guan, Evgeny Kharlamov, Jialiang Lu, Jie Tang

**Abstracts**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. However, recent studies have shown that GNNs are vulnerable to adversarial attacks. In this paper, we study a recently-introduced realistic attack scenario on graphs -- graph injection attack (GIA). In the GIA scenario, the adversary is not able to modify the existing link structure and node attributes of the input graph, instead the attack is performed by injecting adversarial nodes into it. We present an analysis on the topological vulnerability of GNNs under GIA setting, based on which we propose the Topological Defective Graph Injection Attack (TDGIA) for effective injection attacks. TDGIA first introduces the topological defective edge selection strategy to choose the original nodes for connecting with the injected ones. It then designs the smooth feature optimization objective to generate the features for the injected nodes. Extensive experiments on large-scale datasets show that TDGIA can consistently and significantly outperform various attack baselines in attacking dozens of defense GNN models. Notably, the performance drop on target GNNs resultant from TDGIA is more than double the damage brought by the best attack solution among hundreds of submissions on KDD-CUP 2020.

摘要: 图神经网络(GNNs)在各种实际应用中取得了良好的性能。然而，最近的研究表明，GNN很容易受到敌意攻击。本文研究了最近引入的一种图的现实攻击场景--图注入攻击(GIA)。在GIA场景中，敌手不能修改输入图的现有链接结构和节点属性，而是通过向其中注入敌方节点来执行攻击。分析了GIA环境下GNNs的拓扑脆弱性，在此基础上提出了针对有效注入攻击的拓扑缺陷图注入攻击(TDGIA)。TDGIA首先引入拓扑缺陷边选择策略，选择原始节点与注入节点连接。然后设计平滑特征优化目标，为注入节点生成特征。在大规模数据集上的广泛实验表明，TDGIA在攻击数十个防御GNN模型时，可以一致且显着地优于各种攻击基线。值得注意的是，TDGIA对目标GNN造成的性能下降是KDD-Cup 2020上百份提交的最佳攻击解决方案带来的损害的两倍多。



