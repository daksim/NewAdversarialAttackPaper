# Latest Adversarial Attack Papers
**update at 2022-02-23 09:57:36**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Examples in Constrained Domains**

受限领域中的对抗性例子 cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2011.01183v2)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.

摘要: 已经证明机器学习算法通过对诸如图像识别等领域中的输入(例如，对抗性示例)进行系统修改而容易受到对抗性操纵。在默认威胁模型下，敌方利用图像的不受约束的性质；每个功能(像素)都完全在敌方的控制之下。然而，目前尚不清楚这些攻击如何转化为限制哪些特征以及如何被攻击者修改的约束域(例如，网络入侵检测)。在这篇文章中，我们探讨了约束域是否比非约束域更不容易受到敌意示例生成算法的影响。我们创建了一种生成对抗性草图的算法：目标通用扰动向量，它在域约束的包络内编码特征显著性。为了评估这些算法的性能，我们在受限(例如，网络入侵检测)和非受限(例如，图像识别)域中对它们进行评估。结果表明，我们的方法在受限领域产生的错误分类率与非约束领域相当(大于95%)。我们的调查表明，受约束域暴露的狭窄攻击面仍然足够大，足以伪造成功的敌意示例；因此，约束似乎不会使域变得健壮。事实上，只需随机选择5个特征，就仍然可以生成对抗性的例子。



## **2. A Tutorial on Adversarial Learning Attacks and Countermeasures**

对抗性学习攻击与对策教程 cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10377v1)

**Authors**: Cato Pauling, Michael Gimson, Muhammed Qaid, Ahmad Kida, Basel Halak

**Abstracts**: Machine learning algorithms are used to construct a mathematical model for a system based on training data. Such a model is capable of making highly accurate predictions without being explicitly programmed to do so. These techniques have a great many applications in all areas of the modern digital economy and artificial intelligence. More importantly, these methods are essential for a rapidly increasing number of safety-critical applications such as autonomous vehicles and intelligent defense systems. However, emerging adversarial learning attacks pose a serious security threat that greatly undermines further such systems. The latter are classified into four types, evasion (manipulating data to avoid detection), poisoning (injection malicious training samples to disrupt retraining), model stealing (extraction), and inference (leveraging over-generalization on training data). Understanding this type of attacks is a crucial first step for the development of effective countermeasures. The paper provides a detailed tutorial on the principles of adversarial machining learning, explains the different attack scenarios, and gives an in-depth insight into the state-of-art defense mechanisms against this rising threat .

摘要: 机器学习算法用于基于训练数据构建系统的数学模型。这样的模型能够做出高度精确的预测，而不需要明确地编程来这样做。这些技术在现代数字经济和人工智能的各个领域都有大量的应用。更重要的是，这些方法对于迅速增加的安全关键型应用(如自动驾驶汽车和智能防御系统)至关重要。然而，新出现的对抗性学习攻击构成了严重的安全威胁，极大地破坏了这样的系统。后者分为四种类型：逃避(操纵数据以避免检测)、中毒(注入恶意训练样本以中断再训练)、模型窃取(提取)和推理(利用训练数据的过度泛化)。了解这类攻击是制定有效对策的关键第一步。本文详细介绍了对抗性机器学习的原理，解释了不同的攻击场景，并深入了解了针对这一不断上升的威胁的最新防御机制。



## **3. Cyber-Physical Defense in the Quantum Era**

量子时代的网络物理防御 cs.CR

14 pages, 7 figures, 1 table, 4 boxes

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10354v1)

**Authors**: Michel Barbeau, Joaquin Garcia-Alfaro

**Abstracts**: Networked-Control Systems (NCSs), a type of cyber-physical systems, consist of tightly integrated computing, communication and control technologies. While being very flexible environments, they are vulnerable to computing and networking attacks. Recent NCSs hacking incidents had major impact. They call for more research on cyber-physical security. Fears about the use of quantum computing to break current cryptosystems make matters worse. While the quantum threat motivated the creation of new disciplines to handle the issue, such as post-quantum cryptography, other fields have overlooked the existence of quantum-enabled adversaries. This is the case of cyber-physical defense research, a distinct but complementary discipline to cyber-physical protection. Cyber-physical defense refers to the capability to detect and react in response to cyber-physical attacks. Concretely, it involves the integration of mechanisms to identify adverse events and prepare response plans, during and after incidents occur. In this paper, we make the assumption that the eventually available quantum computer will provide an advantage to adversaries against defenders, unless they also adopt this technology. We envision the necessity for a paradigm shift, where an increase of adversarial resources because of quantum supremacy does not translate into higher likelihood of disruptions. Consistently with current system design practices in other areas, such as the use of artificial intelligence for the reinforcement of attack detection tools, we outline a vision for next generation cyber-physical defense layers leveraging ideas from quantum computing and machine learning. Through an example, we show that defenders of NCSs can learn and improve their strategies to anticipate and recover from attacks.

摘要: 网络控制系统(NCSs)是一种集计算、通信和控制技术于一体的网络物理系统。虽然它们是非常灵活的环境，但很容易受到计算和网络攻击。最近NCS的黑客事件产生了重大影响。他们呼吁对网络物理安全进行更多研究。对使用量子计算来破解现有密码系统的担忧使情况变得更糟。虽然量子威胁促使创建新的学科来处理这个问题，如后量子密码学，但其他领域忽略了量子对手的存在。这就是网络物理防御研究的情况，这是一门与网络物理保护截然不同但相辅相成的学科。网络物理防御是指检测并响应网络物理攻击的能力。具体地说，它涉及整合各种机制，以便在事件发生期间和之后识别不良事件并准备应对计划。在这篇文章中，我们假设最终可用的量子计算机将为对手对抗防御者提供优势，除非他们也采用这种技术。我们预见了范式转变的必要性，在这种情况下，由于量子优势而增加的对抗性资源并不会转化为更高的破坏可能性。与其他领域目前的系统设计实践一致，例如使用人工智能来加强攻击检测工具，我们利用量子计算和机器学习的想法勾勒出下一代网络物理防御层的愿景。通过一个实例，我们表明NCS的防御者可以学习和改进他们的策略，以预测攻击并从攻击中恢复。



## **4. Measurement-Device-Independent Quantum Secure Direct Communication with User Authentication**

具有用户认证的独立于测量设备的量子安全直接通信 quant-ph

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10316v1)

**Authors**: Nayana Das, Goutam Paul

**Abstracts**: Quantum secure direct communication (QSDC) and deterministic secure quantum communication (DSQC) are two important branches of quantum cryptography, where one can transmit a secret message securely without encrypting it by a prior key. In the practical scenario, an adversary can apply detector-side-channel attacks to get some non-negligible amount of information about the secret message. Measurement-device-independent (MDI) quantum protocols can remove this kind of detector-side-channel attack, by introducing an untrusted third party (UTP), who performs all the measurements during the protocol with imperfect measurement devices. In this paper, we put forward the first MDI-QSDC protocol with user identity authentication, where both the sender and the receiver first check the authenticity of the other party and then exchange the secret message. Then we extend this to an MDI quantum dialogue (QD) protocol, where both the parties can send their respective secret messages after verifying the identity of the other party. Along with this, we also report the first MDI-DSQC protocol with user identity authentication. Theoretical analyses prove the security of our proposed protocols against common attacks.

摘要: 量子安全直接通信(QSDC)和确定性安全量子通信(DSQC)是量子密码学的两个重要分支。在实际场景中，攻击者可以应用检测器端信道攻击来获取有关秘密消息的一些不可忽略的信息。测量设备无关(MDI)量子协议可以通过引入一个不可信的第三方(UTP)来消除这种探测器侧信道攻击，该第三方使用不完善的测量设备执行协议中的所有测量。本文提出了第一个具有用户身份认证的MDI-QSDC协议，其中发送方和接收方都先检查对方的真实性，然后交换秘密消息。然后我们将其扩展到MDI量子对话(QD)协议，在该协议中，双方可以在验证对方的身份后发送各自的秘密消息。同时，我们还报道了第一个支持用户身份认证的MDI-DSQC协议。理论分析证明了我们提出的协议具有抗常见攻击的安全性。



## **5. HoneyModels: Machine Learning Honeypots**

HoneyModels：机器学习的蜜罐 cs.CR

Published in: MILCOM 2021 - 2021 IEEE Military Communications  Conference (MILCOM)

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10309v1)

**Authors**: Ahmed Abdou, Ryan Sheatsley, Yohan Beugin, Tyler Shipp, Patrick McDaniel

**Abstracts**: Machine Learning is becoming a pivotal aspect of many systems today, offering newfound performance on classification and prediction tasks, but this rapid integration also comes with new unforeseen vulnerabilities. To harden these systems the ever-growing field of Adversarial Machine Learning has proposed new attack and defense mechanisms. However, a great asymmetry exists as these defensive methods can only provide security to certain models and lack scalability, computational efficiency, and practicality due to overly restrictive constraints. Moreover, newly introduced attacks can easily bypass defensive strategies by making subtle alterations. In this paper, we study an alternate approach inspired by honeypots to detect adversaries. Our approach yields learned models with an embedded watermark. When an adversary initiates an interaction with our model, attacks are encouraged to add this predetermined watermark stimulating detection of adversarial examples. We show that HoneyModels can reveal 69.5% of adversaries attempting to attack a Neural Network while preserving the original functionality of the model. HoneyModels offer an alternate direction to secure Machine Learning that slightly affects the accuracy while encouraging the creation of watermarked adversarial samples detectable by the HoneyModel but indistinguishable from others for the adversary.

摘要: 机器学习正在成为当今许多系统的一个关键方面，它在分类和预测任务上提供了新的性能，但这种快速集成也伴随着新的不可预见的漏洞。为了强化这些系统，不断发展的对抗性机器学习领域提出了新的攻防机制。然而，由于这些防御方法只能为某些模型提供安全性，并且由于过于严格的约束而缺乏可扩展性、计算效率和实用性，因此存在很大的不对称性。此外，新引入的攻击可以通过微妙的更改轻松绕过防御策略。在本文中，我们研究了一种受蜜罐启发的另一种检测对手的方法。我们的方法产生带有嵌入水印的学习模型。当敌方发起与我们的模型的交互时，鼓励攻击添加该预定水印来刺激对敌方示例的检测。结果表明，HoneyModels在保持模型原有功能的同时，可以发现69.5%的攻击者试图攻击神经网络。HoneyModel为确保机器学习的安全提供了另一种方向，这对准确性略有影响，同时鼓励创建HoneyModel可以检测到的带水印的敌意样本，但对于敌手来说无法与其他样本区分开来。



## **6. Hardware Obfuscation of Digital FIR Filters**

数字FIR滤波器的硬件混淆 cs.CR

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.10022v1)

**Authors**: Levent Aksoy, Alexander Hepp, Johanna Baehr, Samuel Pagliarini

**Abstracts**: A finite impulse response (FIR) filter is a ubiquitous block in digital signal processing applications. Its characteristics are determined by its coefficients, which are the intellectual property (IP) for its designer. However, in a hardware efficient realization, its coefficients become vulnerable to reverse engineering. This paper presents a filter design technique that can protect this IP, taking into account hardware complexity and ensuring that the filter behaves as specified only when a secret key is provided. To do so, coefficients are hidden among decoys, which are selected beyond possible values of coefficients using three alternative methods. As an attack scenario, an adversary at an untrusted foundry is considered. A reverse engineering technique is developed to find the chosen decoy selection method and explore the potential leakage of coefficients through decoys. An oracle-less attack is also used to find the secret key. Experimental results show that the proposed technique can lead to filter designs with competitive hardware complexity and higher resiliency to attacks with respect to previously proposed methods.

摘要: 有限脉冲响应(FIR)过滤是数字信号处理应用中普遍存在的一种挡路。它的特性是由它的系数决定的，这些系数是它的设计者的知识产权(IP)。然而，在硬件高效实现中，其系数容易受到逆向工程的影响。本文提出了一种过滤的设计技术，它可以保护这个IP，考虑到硬件的复杂性，并确保只有在提供密钥的情况下，过滤才能按照规定的方式运行。为此，系数隐藏在诱饵中，使用三种替代方法选择超出系数可能值的诱饵。作为攻击场景，考虑不可信铸造厂的对手。开发了一种逆向工程技术来寻找所选择的诱饵选择方法，并通过诱饵探测系数的潜在泄漏。也可以使用无预言机攻击来查找密钥。实验结果表明，与以往的过滤设计方法相比，该方法可以设计出硬件复杂度更高、抗攻击能力更强的好胜设计。



## **7. Learning to Attack with Fewer Pixels: A Probabilistic Post-hoc Framework for Refining Arbitrary Dense Adversarial Attacks**

学习用更少的像素进行攻击：一种精化任意密集对手攻击的概率后自组织框架 cs.CV

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2010.06131v2)

**Authors**: He Zhao, Thanh Nguyen, Trung Le, Paul Montague, Olivier De Vel, Tamas Abraham, Dinh Phung

**Abstracts**: Deep neural network image classifiers are reported to be susceptible to adversarial evasion attacks, which use carefully crafted images created to mislead a classifier. Many adversarial attacks belong to the category of dense attacks, which generate adversarial examples by perturbing all the pixels of a natural image. To generate sparse perturbations, sparse attacks have been recently developed, which are usually independent attacks derived by modifying a dense attack's algorithm with sparsity regularisations, resulting in reduced attack efficiency. In this paper, we aim to tackle this task from a different perspective. We select the most effective perturbations from the ones generated from a dense attack, based on the fact we find that a considerable amount of the perturbations on an image generated by dense attacks may contribute little to attacking a classifier. Accordingly, we propose a probabilistic post-hoc framework that refines given dense attacks by significantly reducing the number of perturbed pixels but keeping their attack power, trained with mutual information maximisation. Given an arbitrary dense attack, the proposed model enjoys appealing compatibility for making its adversarial images more realistic and less detectable with fewer perturbations. Moreover, our framework performs adversarial attacks much faster than existing sparse attacks.

摘要: 据报道，深度神经网络图像分类器容易受到敌意规避攻击，这些攻击使用精心制作的图像来误导分类器。许多对抗性攻击属于密集攻击的范畴，通过扰乱自然图像的所有像素来生成对抗性示例。为了产生稀疏扰动，最近发展了稀疏攻击，这些攻击通常是通过用稀疏正则化修改稠密攻击算法而得到的独立攻击，从而降低了攻击效率。在本文中，我们旨在从不同的角度来看待撞击这一任务。我们从密集攻击产生的扰动中选择最有效的扰动，因为我们发现密集攻击对图像产生的相当大的扰动对攻击分类器的贡献很小。因此，我们提出了一种概率后自组织框架，它通过显著减少扰动像素的数量，但保持它们的攻击能力，并用互信息最大化来训练，从而优化给定的密集攻击。在给定任意密集攻击的情况下，所提出的模型具有良好的兼容性，使其对抗性图像更逼真，且在较少扰动的情况下不易被检测到。此外，我们的框架执行对抗性攻击的速度比现有的稀疏攻击要快得多。



## **8. Transferring Adversarial Robustness Through Robust Representation Matching**

通过鲁棒表示匹配传递对抗鲁棒性 cs.LG

To appear at USENIX'22

**SubmitDate**: 2022-02-21    [paper-pdf](http://arxiv.org/pdf/2202.09994v1)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.

摘要: 随着机器学习的广泛应用，人们普遍关注机器学习的安全性和可靠性。因此，许多人已经开发出防御措施，以加强神经网络对敌意例子的抵挡，这些例子是潜移默化的，输入被可靠地错误分类。对抗性训练，即在训练期间生成并使用对抗性例子，是为数不多的能够可靠地抵御针对神经网络的此类攻击的已知防御措施之一。然而，对抗性训练带来了巨大的训练开销，并且与模型复杂度和输入维度的比例关系不佳。在本文中，我们提出了鲁棒表示匹配(RRM)，这是一种低成本的方法，可以将敌对训练模型的鲁棒性转移到为同一任务训练的新模型，而不考虑体系结构的差异。受师生学习的启发，我们的方法引入了一种新颖的训练损失，鼓励学生学习教师的健壮表征。与前人的工作相比，RRM在模型性能和对抗性训练时间方面都具有优势。在CIFAR-10上，RRM训练的健壮模型$\sim\比最先进的模型快1.8倍。此外，RRM在高维数据集上仍然有效。在受限的ImageNet上，RRM训练的ResNet50型号$\sim比标准对手训练快18倍。



## **9. Overparametrization improves robustness against adversarial attacks: A replication study**

过度参数化提高对抗对手攻击的稳健性：一项重复研究 cs.LG

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09735v1)

**Authors**: Ali Borji

**Abstracts**: Overparametrization has become a de facto standard in machine learning. Despite numerous efforts, our understanding of how and where overparametrization helps model accuracy and robustness is still limited. To this end, here we conduct an empirical investigation to systemically study and replicate previous findings in this area, in particular the study by Madry et al. Together with this study, our findings support the "universal law of robustness" recently proposed by Bubeck et al. We argue that while critical for robust perception, overparametrization may not be enough to achieve full robustness and smarter architectures e.g. the ones implemented by the human visual cortex) seem inevitable.

摘要: 过度参数化已经成为机器学习中事实上的标准。尽管做了很多努力，我们对过度参数化如何以及在哪里有助于模型的准确性和健壮性的理解仍然有限。为此，我们在这里进行了实证调查，以系统地研究和复制这一领域的前人研究成果，特别是Madry等人的研究。结合这项研究，我们的发现支持了Bubeck等人最近提出的“稳健性普遍定律”。我们认为，虽然过度参数化对于鲁棒感知至关重要，但过度参数化可能不足以实现完全的鲁棒性和更智能的架构(例如，由人眼视皮层实现的架构)似乎是不可避免的。



## **10. Runtime-Assured, Real-Time Neural Control of Microgrids**

保证运行时间的微电网实时神经控制 eess.SY

**SubmitDate**: 2022-02-20    [paper-pdf](http://arxiv.org/pdf/2202.09710v1)

**Authors**: Amol Damare, Shouvik Roy, Scott A. Smolka, Scott D. Stoller

**Abstracts**: We present SimpleMG, a new, provably correct design methodology for runtime assurance of microgrids (MGs) with neural controllers. Our approach is centered around the Neural Simplex Architecture, which in turn is based on Sha et al.'s Simplex Control Architecture. Reinforcement Learning is used to synthesize high-performance neural controllers for MGs. Barrier Certificates are used to establish SimpleMG's runtime-assurance guarantees. We present a novel method to derive the condition for switching from the unverified neural controller to the verified-safe baseline controller, and we prove that the method is correct. We conduct an extensive experimental evaluation of SimpleMG using RTDS, a high-fidelity, real-time simulation environment for power systems, on a realistic model of a microgrid comprising three distributed energy resources (battery, photovoltaic, and diesel generator). Our experiments confirm that SimpleMG can be used to develop high-performance neural controllers for complex microgrids while assuring runtime safety, even in the presence of adversarial input attacks on the neural controller. Our experiments also demonstrate the benefits of online retraining of the neural controller while the baseline controller is in control

摘要: 我们提出了SimpleMG，这是一种新的、可以证明是正确的设计方法，用于带神经控制器的微电网(MG)的运行时保证。我们的方法是以神经单纯形体系结构为中心的，而神经单纯形体系结构又基于沙等人的单纯形控制体系结构。强化学习被用来综合高性能的磁控系统的神经控制器。屏障证书用于建立SimpleMG的运行时保证。我们提出了一种新的方法来推导从未经验证的神经控制器切换到验证安全的基线控制器的条件，并证明了该方法的正确性。在一个由三种分布式能源(电池、光伏和柴油发电机)组成的真实微电网模型上，我们使用一个高保真、实时的电力系统仿真环境RTDS对SimpleMG进行了广泛的实验评估。我们的实验证实，SimpleMG可以用来开发复杂微网格的高性能神经控制器，同时保证运行时的安全性，即使在神经控制器受到敌意输入攻击的情况下也是如此。我们的实验也证明了在基线控制器处于控制状态时在线重新训练神经控制器的好处。



## **11. Detection of Stealthy Adversaries for Networked Unmanned Aerial Vehicles**

网络化无人机隐身对手的检测 eess.SY

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09661v1)

**Authors**: Mohammad Bahrami, Hamidreza Jafarnejadsani

**Abstracts**: A network of unmanned aerial vehicles (UAVs) provides distributed coverage, reconfigurability, and maneuverability in performing complex cooperative tasks. However, it relies on wireless communications that can be susceptible to cyber adversaries and intrusions, disrupting the entire network's operation. This paper develops model-based centralized and decentralized observer techniques for detecting a class of stealthy intrusions, namely zero-dynamics and covert attacks, on networked UAVs in formation control settings. The centralized observer that runs in a control center leverages switching in the UAVs' communication topology for attack detection, and the decentralized observers, implemented onboard each UAV in the network, use the model of networked UAVs and locally available measurements. Experimental results are provided to show the effectiveness of the proposed detection schemes in different case studies.

摘要: 无人驾驶飞行器(UAV)网络在执行复杂的协作任务时提供了分布式覆盖、可重构性和机动性。然而，它依赖于无线通信，这可能会受到网络对手和入侵的影响，扰乱整个网络的运行。提出了一种基于模型的集中式和分散式观测器技术，用于检测编队控制环境下网络化无人机的一类隐身入侵，即零动态攻击和隐蔽攻击。在控制中心运行的集中式观察器利用无人机通信拓扑中的切换来进行攻击检测，而在网络中的每架无人机上实现的分散式观察器使用联网的无人机模型和本地可用的测量。实验结果表明，所提出的检测方案在不同的案例研究中是有效的。



## **12. Stochastic sparse adversarial attacks**

随机稀疏对抗性攻击 cs.LG

Final version published at the ICTAI 2021 conference with a best  student paper award. Codes are available through the link:  https://github.com/hhajri/stochastic-sparse-adv-attacks

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2011.12423v4)

**Authors**: Manon Césaire, Lucas Schott, Hatem Hajri, Sylvain Lamprier, Patrick Gallinari

**Abstracts**: This paper introduces stochastic sparse adversarial attacks (SSAA), standing as simple, fast and purely noise-based targeted and untargeted attacks of neural network classifiers (NNC). SSAA offer new examples of sparse (or $L_0$) attacks for which only few methods have been proposed previously. These attacks are devised by exploiting a small-time expansion idea widely used for Markov processes. Experiments on small and large datasets (CIFAR-10 and ImageNet) illustrate several advantages of SSAA in comparison with the-state-of-the-art methods. For instance, in the untargeted case, our method called Voting Folded Gaussian Attack (VFGA) scales efficiently to ImageNet and achieves a significantly lower $L_0$ score than SparseFool (up to $\frac{2}{5}$) while being faster. Moreover, VFGA achieves better $L_0$ scores on ImageNet than Sparse-RS when both attacks are fully successful on a large number of samples.

摘要: 介绍了随机稀疏对抗攻击(SSAA)，即简单、快速、纯基于噪声的神经网络分类器(NNC)目标攻击和非目标攻击(NNC)。SSAA为稀疏(或$L_0$)攻击提供了新的例子，以前只有很少的方法被提出。这些攻击是通过利用马尔可夫过程中广泛使用的小时间扩展思想来设计的。在小型和大型数据集(CIFAR-10和ImageNet)上的实验表明，与最先进的方法相比，SSAA具有一些优势。例如，在无目标的情况下，我们的方法称为投票折叠高斯攻击(VFGA)，可以有效地扩展到ImageNet，并且获得比SparseFool(最高可达$\frac{2}{5}$)低得多的$L_0$分数，同时速度更快。此外，当两种攻击在大量样本上都完全成功时，VFGA在ImageNet上获得了比稀疏RS更好的$L_0$得分。



## **13. Internal Wasserstein Distance for Adversarial Attack and Defense**

对抗性攻防的瓦瑟斯坦内部距离 cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2103.07598v2)

**Authors**: Mingkui Tan, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Yanwu Xu

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been important ways to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.

摘要: 深度神经网络(DNNs)很容易受到敌意攻击，这些攻击可能会导致DNN的错误分类，但可能无法被人类感知到。对抗性防御已经成为提高DNNs健壮性的重要途径。现有的攻击方法通常依赖于$\ell_p$距离等度量来构建敌意示例来扰动样本。然而，由于其有限的扰动，这些度量可能不足以进行对抗性攻击。本文提出了一种新的内部Wasserstein距离(IWD)来刻画两个样本之间的语义相似性，从而有助于获得比目前使用的$\\ell_p$距离等度量更大的扰动。然后利用内部Wasserstein距离进行对抗性攻击和防御。特别地，我们开发了一种新的攻击方法，该方法依赖于IWD来计算图像与其对手示例之间的相似度。这样，我们就可以生成不同的、语义相似的对抗性例子，而这些例子是现有防御方法更难防御的。此外，我们设计了一种新的防御方法，依靠IWD学习鲁棒模型来抵御看不见的对手例子。我们提供了充分的理论和经验证据来支持我们的方法。



## **14. Robust Reinforcement Learning as a Stackelberg Game via Adaptively-Regularized Adversarial Training**

基于自适应正则化对抗性训练的Stackelberg博弈鲁棒强化学习 cs.LG

**SubmitDate**: 2022-02-19    [paper-pdf](http://arxiv.org/pdf/2202.09514v1)

**Authors**: Peide Huang, Mengdi Xu, Fei Fang, Ding Zhao

**Abstracts**: Robust Reinforcement Learning (RL) focuses on improving performances under model errors or adversarial attacks, which facilitates the real-life deployment of RL agents. Robust Adversarial Reinforcement Learning (RARL) is one of the most popular frameworks for robust RL. However, most of the existing literature models RARL as a zero-sum simultaneous game with Nash equilibrium as the solution concept, which could overlook the sequential nature of RL deployments, produce overly conservative agents, and induce training instability. In this paper, we introduce a novel hierarchical formulation of robust RL - a general-sum Stackelberg game model called RRL-Stack - to formalize the sequential nature and provide extra flexibility for robust training. We develop the Stackelberg Policy Gradient algorithm to solve RRL-Stack, leveraging the Stackelberg learning dynamics by considering the adversary's response. Our method generates challenging yet solvable adversarial environments which benefit RL agents' robust learning. Our algorithm demonstrates better training stability and robustness against different testing conditions in the single-agent robotics control and multi-agent highway merging tasks.

摘要: 鲁棒强化学习(RL)侧重于提高在模型错误或敌意攻击下的性能，有利于RL Agent的实际部署。鲁棒对抗强化学习(RARL)是目前最流行的鲁棒对抗强化学习框架之一。然而，现有的文献大多将RARL建模为以纳什均衡为解概念的零和同时博弈，这可能会忽略RL部署的序贯性质，产生过于保守的代理，并导致训练不稳定。在本文中，我们引入了一种新的鲁棒RL的分层表示-称为RRL-Stack的一般和Stackelberg博弈模型-以形式化顺序性质，并为鲁棒训练提供额外的灵活性。我们开发了Stackelberg策略梯度算法来求解RRL-Stack，通过考虑对手的响应来利用Stackelberg学习动态。我们的方法产生了具有挑战性但可解决的对抗环境，这有利于RL Agent的鲁棒学习。在单智能体机器人控制和多智能体公路合并任务中，我们的算法对不同的测试条件表现出较好的训练稳定性和鲁棒性。



## **15. Attacks, Defenses, And Tools: A Framework To Facilitate Robust AI/ML Systems**

攻击、防御和工具：促进健壮AI/ML系统的框架 cs.CR

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09465v1)

**Authors**: Mohamad Fazelnia, Igor Khokhlov, Mehdi Mirakhorli

**Abstracts**: Software systems are increasingly relying on Artificial Intelligence (AI) and Machine Learning (ML) components. The emerging popularity of AI techniques in various application domains attracts malicious actors and adversaries. Therefore, the developers of AI-enabled software systems need to take into account various novel cyber-attacks and vulnerabilities that these systems may be susceptible to. This paper presents a framework to characterize attacks and weaknesses associated with AI-enabled systems and provide mitigation techniques and defense strategies. This framework aims to support software designers in taking proactive measures in developing AI-enabled software, understanding the attack surface of such systems, and developing products that are resilient to various emerging attacks associated with ML. The developed framework covers a broad spectrum of attacks, mitigation techniques, and defensive and offensive tools. In this paper, we demonstrate the framework architecture and its major components, describe their attributes, and discuss the long-term goals of this research.

摘要: 软件系统越来越依赖人工智能(AI)和机器学习(ML)组件。人工智能技术在各个应用领域的新兴普及吸引了恶意行为者和对手。因此，人工智能软件系统的开发人员需要考虑到这些系统可能容易受到的各种新型网络攻击和漏洞。本文提出了一个框架来描述与人工智能系统相关的攻击和弱点，并提供缓解技术和防御策略。该框架旨在支持软件设计人员在开发支持人工智能的软件时采取主动措施，了解此类系统的攻击面，并开发对与ML相关的各种新兴攻击具有弹性的产品。开发的框架涵盖了广泛的攻击、缓解技术以及防御和进攻工具。在本文中，我们展示了框架体系结构及其主要组件，描述了它们的属性，并讨论了本研究的长期目标。



## **16. Black-box Node Injection Attack for Graph Neural Networks**

图神经网络的黑盒节点注入攻击 cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09389v1)

**Authors**: Mingxuan Ju, Yujie Fan, Yanfang Ye, Liang Zhao

**Abstracts**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to vital fields that require high security standard such as product recommendation and traffic forecasting. Under such scenarios, exploiting GNN's vulnerabilities and further downgrade its classification performance become highly incentive for adversaries. Previous attackers mainly focus on structural perturbations of existing graphs. Although they deliver promising results, the actual implementation needs capability of manipulating the graph connectivity, which is impractical in some circumstances. In this work, we study the possibility of injecting nodes to evade the victim GNN model, and unlike previous related works with white-box setting, we significantly restrict the amount of accessible knowledge and explore the black-box setting. Specifically, we model the node injection attack as a Markov decision process and propose GA2C, a graph reinforcement learning framework in the fashion of advantage actor critic, to generate realistic features for injected nodes and seamlessly merge them into the original graph following the same topology characteristics. Through our extensive experiments on multiple acknowledged benchmark datasets, we demonstrate the superior performance of our proposed GA2C over existing state-of-the-art methods. The data and source code are publicly accessible at: https://github.com/jumxglhf/GA2C.

摘要: 多年来，图神经网络(GNNs)引起了人们的广泛关注，并被广泛应用于产品推荐、流量预测等对安全性要求较高的重要领域。在这种情况下，利用GNN的漏洞并进一步降低其分类性能成为对手的极大诱因。以往的攻击者主要集中在现有图的结构扰动上。虽然它们提供了有希望的结果，但实际实现需要能够操作图形连接，这在某些情况下是不切实际的。在这项工作中，我们研究了注入节点来逃避受害者GNN模型的可能性，与以往白盒设置的相关工作不同，我们显著限制了可访问的知识量，并探索了黑盒设置。具体地说，我们将节点注入攻击建模为马尔可夫决策过程，并提出了一种优势角色批判式的图强化学习框架GA2C，用于生成注入节点的真实特征，并按照相同的拓扑特征将其无缝合并到原始图中。通过我们在多个公认的基准数据集上的广泛实验，我们证明了我们提出的GA2C比现有最先进的方法具有更好的性能。数据和源代码可在以下网址公开访问：https://github.com/jumxglhf/GA2C.



## **17. Synthetic Disinformation Attacks on Automated Fact Verification Systems**

对自动事实验证系统的合成虚假信息攻击 cs.CL

AAAI 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09381v1)

**Authors**: Yibing Du, Antoine Bosselut, Christopher D. Manning

**Abstracts**: Automated fact-checking is a needed technology to curtail the spread of online misinformation. One current framework for such solutions proposes to verify claims by retrieving supporting or refuting evidence from related textual sources. However, the realistic use cases for fact-checkers will require verifying claims against evidence sources that could be affected by the same misinformation. Furthermore, the development of modern NLP tools that can produce coherent, fabricated content would allow malicious actors to systematically generate adversarial disinformation for fact-checkers.   In this work, we explore the sensitivity of automated fact-checkers to synthetic adversarial evidence in two simulated settings: AdversarialAddition, where we fabricate documents and add them to the evidence repository available to the fact-checking system, and AdversarialModification, where existing evidence source documents in the repository are automatically altered. Our study across multiple models on three benchmarks demonstrates that these systems suffer significant performance drops against these attacks. Finally, we discuss the growing threat of modern NLG systems as generators of disinformation in the context of the challenges they pose to automated fact-checkers.

摘要: 自动事实核查是遏制在线错误信息传播所必需的技术。目前这类解决方案的一个框架建议通过从相关文本来源检索、支持或驳斥证据来核实主张。然而，事实核查人员的现实用例将需要对照可能受到相同错误信息影响的证据来源来验证声明。此外，能够产生连贯的、捏造的内容的现代NLP工具的开发将允许恶意行为者系统地为事实核查人员生成对抗性的虚假信息。在这项工作中，我们在两个模拟设置中探索了自动事实检查器对合成敌对证据的敏感性：AdversarialAddition，我们伪造文档并将它们添加到事实检查系统可用的证据储存库；AdversarialMoentation，其中储存库中的现有证据源文档被自动更改。我们在三个基准测试的多个模型上的研究表明，这些系统在抵御这些攻击时性能显著下降。最后，我们讨论了现代NLG系统作为虚假信息生成器的日益增长的威胁，在它们对自动事实核查人员构成挑战的背景下。



## **18. Exploring Adversarially Robust Training for Unsupervised Domain Adaptation**

无监督领域自适应的对抗性鲁棒训练探索 cs.CV

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09300v1)

**Authors**: Shao-Yuan Lo, Vishal M. Patel

**Abstracts**: Unsupervised Domain Adaptation (UDA) methods aim to transfer knowledge from a labeled source domain to an unlabeled target domain. UDA has been extensively studied in the computer vision literature. Deep networks have been shown to be vulnerable to adversarial attacks. However, very little focus is devoted to improving the adversarial robustness of deep UDA models, causing serious concerns about model reliability. Adversarial Training (AT) has been considered to be the most successful adversarial defense approach. Nevertheless, conventional AT requires ground-truth labels to generate adversarial examples and train models, which limits its effectiveness in the unlabeled target domain. In this paper, we aim to explore AT to robustify UDA models: How to enhance the unlabeled data robustness via AT while learning domain-invariant features for UDA? To answer this, we provide a systematic study into multiple AT variants that potentially apply to UDA. Moreover, we propose a novel Adversarially Robust Training method for UDA accordingly, referred to as ARTUDA. Extensive experiments on multiple attacks and benchmarks show that ARTUDA consistently improves the adversarial robustness of UDA models.

摘要: 无监督域自适应(UDA)方法旨在将知识从有标签的源域转移到无标签的目标域。UDA在计算机视觉文献中得到了广泛的研究。深层网络已被证明容易受到敌意攻击。然而，很少有人致力于提高深度UDA模型的对抗健壮性，这引起了人们对模型可靠性的严重关注。对抗性训练(AT)被认为是最成功的对抗性防御方法。然而，传统的自动测试需要地面事实标签来生成对抗性示例和训练模型，这限制了其在未标记的目标领域的有效性。在本文中，我们的目标是探索AT对UDA模型的鲁棒性：如何在学习UDA的域不变性特征的同时，通过AT增强未标记数据的健壮性？为了回答这个问题，我们对可能适用于UDA的多个AT变体进行了系统研究。此外，我们还针对UDA提出了一种新颖的对抗性鲁棒训练方法，称为ARTUDA。在多个攻击和基准测试上的大量实验表明，ARTUDA一致地提高了UDA模型的对抗健壮性。



## **19. Resurrecting Trust in Facial Recognition: Mitigating Backdoor Attacks in Face Recognition to Prevent Potential Privacy Breaches**

恢复面部识别中的信任：减轻面部识别中的后门攻击，以防止潜在的隐私泄露 cs.CV

15 pages

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.10320v1)

**Authors**: Reena Zelenkova, Jack Swallow, M. A. P. Chamikara, Dongxi Liu, Mohan Baruwal Chhetri, Seyit Camtepe, Marthie Grobler, Mahathir Almashor

**Abstracts**: Biometric data, such as face images, are often associated with sensitive information (e.g medical, financial, personal government records). Hence, a data breach in a system storing such information can have devastating consequences. Deep learning is widely utilized for face recognition (FR); however, such models are vulnerable to backdoor attacks executed by malicious parties. Backdoor attacks cause a model to misclassify a particular class as a target class during recognition. This vulnerability can allow adversaries to gain access to highly sensitive data protected by biometric authentication measures or allow the malicious party to masquerade as an individual with higher system permissions. Such breaches pose a serious privacy threat. Previous methods integrate noise addition mechanisms into face recognition models to mitigate this issue and improve the robustness of classification against backdoor attacks. However, this can drastically affect model accuracy. We propose a novel and generalizable approach (named BA-BAM: Biometric Authentication - Backdoor Attack Mitigation), that aims to prevent backdoor attacks on face authentication deep learning models through transfer learning and selective image perturbation. The empirical evidence shows that BA-BAM is highly robust and incurs a maximal accuracy drop of 2.4%, while reducing the attack success rate to a maximum of 20%. Comparisons with existing approaches show that BA-BAM provides a more practical backdoor mitigation approach for face recognition.

摘要: 生物特征数据(如面部图像)通常与敏感信息(如医疗、金融、个人政府记录)相关联。因此，存储此类信息的系统中的数据泄露可能会造成毁灭性的后果。深度学习被广泛用于人脸识别(FR)；然而，此类模型容易受到恶意方执行的后门攻击。后门攻击会导致模型在识别过程中将特定类错误分类为目标类。此漏洞可让攻击者访问受生物特征验证措施保护的高度敏感数据，或允许恶意方伪装成具有更高系统权限的个人。这类侵犯隐私的行为构成了严重的隐私威胁。以往的方法将噪声添加机制集成到人脸识别模型中，以缓解这一问题，并提高分类对后门攻击的鲁棒性。但是，这可能会极大地影响模型精度。我们提出了一种新颖的、可推广的方法(BA-BAM：Biometry Authentication-Backdoor Attack Mitigation)，旨在通过迁移学习和选择性图像扰动来防止对人脸认证深度学习模型的后门攻击。实验结果表明，BA-BAM算法具有很强的鲁棒性，最大准确率下降2.4%，而攻击成功率最高可达20%。与现有方法的比较表明，BA-BAM为人脸识别提供了一种更实用的后门缓解方法。



## **20. Critical Checkpoints for Evaluating Defence Models Against Adversarial Attack and Robustness**

评估防御模型对抗攻击和健壮性的关键检查点 cs.CR

16 pages, 8 figures

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2202.09039v1)

**Authors**: Kanak Tekwani, Manojkumar Parmar

**Abstracts**: From past couple of years there is a cycle of researchers proposing a defence model for adversaries in machine learning which is arguably defensible to most of the existing attacks in restricted condition (they evaluate on some bounded inputs or datasets). And then shortly another set of researcher finding the vulnerabilities in that defence model and breaking it by proposing a stronger attack model. Some common flaws are been noticed in the past defence models that were broken in very short time. Defence models being broken so easily is a point of concern as decision of many crucial activities are taken with the help of machine learning models. So there is an utter need of some defence checkpoints that any researcher should keep in mind while evaluating the soundness of technique and declaring it to be decent defence technique. In this paper, we have suggested few checkpoints that should be taken into consideration while building and evaluating the soundness of defence models. All these points are recommended after observing why some past defence models failed and how some model remained adamant and proved their soundness against some of the very strong attacks.

摘要: 在过去的几年里，研究人员在机器学习中提出了一个针对对手的防御模型，该模型可以在有限条件下防御大多数现有的攻击(他们在一些有界的输入或数据集上进行评估)。不久，另一组研究人员发现了该防御模型中的漏洞，并提出了一种更强大的攻击模型来打破它。过去的防御模式在很短的时间内就被打破了，人们注意到了一些常见的缺陷。防御模型如此容易被打破是一个令人担忧的问题，因为许多关键活动的决策都是在机器学习模型的帮助下做出的。因此，非常需要一些防御检查点，任何研究者在评估技术的可靠性并宣布它是一种像样的防御技术时，都应该牢记这一点。在本文中，我们提出了在建立和评估防御模型的可靠性时应考虑的几个检查点。所有这些观点都是在观察了过去的一些防御模型失败的原因，以及一些模型是如何保持顽固的，并证明了它们在一些非常强大的攻击下是健全的之后推荐的。



## **21. Explaining Adversarial Vulnerability with a Data Sparsity Hypothesis**

用数据稀疏性假说解释对抗性脆弱性 cs.AI

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2103.00778v3)

**Authors**: Mahsa Paknezhad, Cuong Phuc Ngo, Amadeus Aristo Winarto, Alistair Cheong, Chuen Yang Beh, Jiayang Wu, Hwee Kuan Lee

**Abstracts**: Despite many proposed algorithms to provide robustness to deep learning (DL) models, DL models remain susceptible to adversarial attacks. We hypothesize that the adversarial vulnerability of DL models stems from two factors. The first factor is data sparsity which is that in the high dimensional input data space, there exist large regions outside the support of the data distribution. The second factor is the existence of many redundant parameters in the DL models. Owing to these factors, different models are able to come up with different decision boundaries with comparably high prediction accuracy. The appearance of the decision boundaries in the space outside the support of the data distribution does not affect the prediction accuracy of the model. However, it makes an important difference in the adversarial robustness of the model. We hypothesize that the ideal decision boundary is as far as possible from the support of the data distribution. In this paper, we develop a training framework to observe if DL models are able to learn such a decision boundary spanning the space around the class distributions further from the data points themselves. Semi-supervised learning was deployed during training by leveraging unlabeled data generated in the space outside the support of the data distribution. We measured adversarial robustness of the models trained using this training framework against well-known adversarial attacks and by using robustness metrics. We found that models trained using our framework, as well as other regularization methods and adversarial training support our hypothesis of data sparsity and that models trained with these methods learn to have decision boundaries more similar to the aforementioned ideal decision boundary. The code for our training framework is available at https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.

摘要: 尽管提出了许多算法来提供深度学习(DL)模型的鲁棒性，但是DL模型仍然容易受到敌意攻击。我们假设DL模型的对抗脆弱性源于两个因素。第一个因素是数据稀疏性，即在高维输入数据空间中，存在数据分布支持之外的大区域。第二个因素是DL模型中存在许多冗余参数。由于这些因素的影响，不同的模型能够给出不同的决策边界，具有较高的预测精度。在数据分布支持度之外的空间出现决策边界并不影响模型的预测精度。然而，它在模型的对抗性鲁棒性方面有很大的不同。我们假设理想的决策边界尽可能远离数据分布的支持。在本文中，我们开发了一个训练框架来观察DL模型是否能够从数据点本身进一步学习跨越类分布周围空间的决策边界。通过利用在数据分布支持之外的空间中生成的未标记数据，在训练期间部署半监督学习。我们通过使用健壮性度量来衡量使用该训练框架训练的模型对众所周知的敌意攻击的敌意稳健性。我们发现，使用我们的框架训练的模型，以及其他正则化方法和对抗性训练，都支持我们的数据稀疏性假设，并且用这些方法训练的模型学习的决策边界更类似于前面提到的理想决策边界。我们培训框架的代码可以在https://github.com/MahsaPaknezhad/AdversariallyRobustTraining.上找到



## **22. Amicable examples for informed source separation**

知情信源分离的友好示例 cs.SD

Accepted to ICASSP 2022

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2110.05059v2)

**Authors**: Naoya Takahashi, Yuki Mitsufuji

**Abstracts**: This paper deals with the problem of informed source separation (ISS), where the sources are accessible during the so-called \textit{encoding} stage. Previous works computed side-information during the encoding stage and source separation models were designed to utilize the side-information to improve the separation performance. In contrast, in this work, we improve the performance of a pretrained separation model that does not use any side-information. To this end, we propose to adopt an adversarial attack for the opposite purpose, i.e., rather than computing the perturbation to degrade the separation, we compute an imperceptible perturbation called amicable noise to improve the separation. Experimental results show that the proposed approach selectively improves the performance of the targeted separation model by 2.23 dB on average and is robust to signal compression. Moreover, we propose multi-model multi-purpose learning that control the effect of the perturbation on different models individually.

摘要: 本文研究信息源分离(ISS)问题，即信息源在所谓的\textit{编码}阶段是可访问的。以前的工作是在编码阶段计算边信息，并设计了源分离模型来利用边信息来提高分离性能。相反，在这项工作中，我们改进了不使用任何边信息的预训练分离模型的性能。为此，我们建议采取相反目的的对抗性攻击，即，我们不计算扰动来降低分离度，而是计算一种称为友好噪声的不可察觉的扰动来改善分离度。实验结果表明，该方法选择性地将目标分离模型的性能平均提高了2.23dB，并且对信号压缩具有较强的鲁棒性。此外，我们还提出了多模型多目标学习，分别控制扰动对不同模型的影响。



## **23. Morphence: Moving Target Defense Against Adversarial Examples**

Morphence：针对敌方的移动目标防御示例 cs.LG

**SubmitDate**: 2022-02-18    [paper-pdf](http://arxiv.org/pdf/2108.13952v4)

**Authors**: Abderrahmen Amich, Birhanu Eshete

**Abstracts**: Robustness to adversarial examples of machine learning models remains an open topic of research. Attacks often succeed by repeatedly probing a fixed target model with adversarial examples purposely crafted to fool it. In this paper, we introduce Morphence, an approach that shifts the defense landscape by making a model a moving target against adversarial examples. By regularly moving the decision function of a model, Morphence makes it significantly challenging for repeated or correlated attacks to succeed. Morphence deploys a pool of models generated from a base model in a manner that introduces sufficient randomness when it responds to prediction queries. To ensure repeated or correlated attacks fail, the deployed pool of models automatically expires after a query budget is reached and the model pool is seamlessly replaced by a new model pool generated in advance. We evaluate Morphence on two benchmark image classification datasets (MNIST and CIFAR10) against five reference attacks (2 white-box and 3 black-box). In all cases, Morphence consistently outperforms the thus-far effective defense, adversarial training, even in the face of strong white-box attacks, while preserving accuracy on clean data.

摘要: 对机器学习模型的对抗性示例的鲁棒性仍然是一个开放的研究课题。攻击通常通过反复探测固定的目标模型而得逞，其中带有故意设计的敌意示例来愚弄它。在本文中，我们介绍了Morphence，一种通过使模型成为移动目标来对抗对手示例来改变防御格局的方法。通过定期移动模型的决策函数，Morphence使重复或相关攻击的成功变得极具挑战性。Morphence以在响应预测查询时引入足够的随机性的方式部署从基础模型生成的模型池。为确保重复或相关攻击失败，部署的模型池在达到查询预算后自动过期，并由预先生成的新模型池无缝替换。我们在两个基准图像分类数据集(MNIST和CIFAR10)上测试了Morphence在5个参考攻击(2个白盒和3个黑盒)下的性能。在所有情况下，Morphence的表现都始终如一地优于迄今有效的防御、对抗性训练，即使面对强大的白盒攻击，也能保持干净数据的准确性。



## **24. What Doesn't Kill You Makes You Robust(er): How to Adversarially Train against Data Poisoning**

什么不会杀死你，让你变得健壮(呃)：如何对抗数据中毒 cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.13624v2)

**Authors**: Jonas Geiping, Liam Fowl, Gowthami Somepalli, Micah Goldblum, Michael Moeller, Tom Goldstein

**Abstracts**: Data poisoning is a threat model in which a malicious actor tampers with training data to manipulate outcomes at inference time. A variety of defenses against this threat model have been proposed, but each suffers from at least one of the following flaws: they are easily overcome by adaptive attacks, they severely reduce testing performance, or they cannot generalize to diverse data poisoning threat models. Adversarial training, and its variants, are currently considered the only empirically strong defense against (inference-time) adversarial attacks. In this work, we extend the adversarial training framework to defend against (training-time) data poisoning, including targeted and backdoor attacks. Our method desensitizes networks to the effects of such attacks by creating poisons during training and injecting them into training batches. We show that this defense withstands adaptive attacks, generalizes to diverse threat models, and incurs a better performance trade-off than previous defenses such as DP-SGD or (evasion) adversarial training.

摘要: 数据中毒是一种威胁模型，在该模型中，恶意行为者篡改训练数据以在推断时操纵结果。针对此威胁模型提出了多种防御方案，但每种方案都至少存在以下缺陷之一：它们很容易被自适应攻击克服，严重降低了测试性能，或者不能推广到不同的数据中毒威胁模型。对抗性训练及其变体，目前被认为是对抗(推理时间)对抗性攻击的唯一经验性强防御。在这项工作中，我们扩展了对抗性训练框架来防御(训练时)数据中毒，包括有针对性的攻击和后门攻击。我们的方法通过在训练期间制造毒药并将它们注入训练批次来使网络对此类攻击的影响变得不敏感。我们表明，这种防御可以抵抗适应性攻击，适用于不同的威胁模型，并且比以前的防御(如DP-SGD或(回避)对抗性训练)具有更好的性能权衡。



## **25. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

利用计算机视觉技术开发隐形对抗性补丁来伪装军事资产 cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08892v1)

**Authors**: Christopher Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.

摘要: 卷积神经网络(CNNs)在目标检测方面取得了快速的进展和很高的成功率。然而，最近的证据突显了它们在对抗性攻击中的脆弱性。这些攻击是经过计算的图像扰动或导致对象误分类或检测抑制的对抗性补丁。传统的伪装方法用于伪装飞机和其他大型机动资产，使其免受情报、监视和侦察技术以及第五代导弹的自主探测，是不切实际的。在这篇文章中，我们提出了一种独特的方法，它可以从计算机视觉技术中产生能够伪装大型军事资产的隐形补丁。我们开发了这些补丁，通过最大化目标检测损失，同时限制补丁的颜色敏感度。这项工作还旨在进一步理解对抗性例子及其对目标检测算法的影响。



## **26. Alexa versus Alexa: Controlling Smart Speakers by Self-Issuing Voice Commands**

Alexa与Alexa：通过自行发出语音命令控制智能扬声器 cs.CR

15 pages, 5 figures, published in Proceedings of the 2022 ACM Asia  Conference on Computer and Communications Security (ASIA CCS '22)

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08619v1)

**Authors**: Sergio Esposito, Daniele Sgandurra, Giampaolo Bella

**Abstracts**: We present Alexa versus Alexa (AvA), a novel attack that leverages audio files containing voice commands and audio reproduction methods in an offensive fashion, to gain control of Amazon Echo devices for a prolonged amount of time. AvA leverages the fact that Alexa running on an Echo device correctly interprets voice commands originated from audio files even when they are played by the device itself -- i.e., it leverages a command self-issue vulnerability. Hence, AvA removes the necessity of having a rogue speaker in proximity of the victim's Echo, a constraint that many attacks share. With AvA, an attacker can self-issue any permissible command to Echo, controlling it on behalf of the legitimate user. We have verified that, via AvA, attackers can control smart appliances within the household, buy unwanted items, tamper linked calendars and eavesdrop on the user. We also discovered two additional Echo vulnerabilities, which we call Full Volume and Break Tag Chain. The Full Volume increases the self-issue command recognition rate, by doubling it on average, hence allowing attackers to perform additional self-issue commands. Break Tag Chain increases the time a skill can run without user interaction, from eight seconds to more than one hour, hence enabling attackers to setup realistic social engineering scenarios. By exploiting these vulnerabilities, the adversary can self-issue commands that are correctly executed 99% of the times and can keep control of the device for a prolonged amount of time. We reported these vulnerabilities to Amazon via their vulnerability research program, who rated them with a Medium severity score. Finally, to assess limitations of AvA on a larger scale, we provide the results of a survey performed on a study group of 18 users, and we show that most of the limitations against AvA are hardly used in practice.

摘要: 我们提出了Alexa vs.Alexa(AVA)，这是一种新颖的攻击，它以攻击性的方式利用包含语音命令和音频复制方法的音频文件来获得对Amazon Echo设备的长时间控制。AVA利用在Echo设备上运行的Alexa能够正确解释源自音频文件的语音命令这一事实，即使音频文件是由设备本身播放的，即它利用了命令自发布漏洞。因此，AVA消除了在受害者的回声附近设置流氓扬声器的必要性，这是许多攻击都有的限制。使用AVA，攻击者可以自行向Echo发出任何允许的命令，并代表合法用户控制它。我们已经证实，通过AVA，攻击者可以控制家庭内的智能家电，购买不需要的物品，篡改链接的日历，并窃听用户。我们还发现了另外两个Echo漏洞，我们称之为Full Volume和Break Tag Chain。全音量通过将其平均翻倍来提高自发布命令识别率，从而允许攻击者执行额外的自发布命令。中断标签链增加了技能在没有用户交互的情况下可以运行的时间，从8秒增加到1小时以上，从而使攻击者能够设置现实的社会工程场景。通过利用这些漏洞，攻击者可以自行发出99%的正确执行次数的命令，并可以在较长时间内保持对设备的控制。我们通过他们的漏洞研究计划向亚马逊报告了这些漏洞，亚马逊对它们进行了中等严重程度的评级。最后，为了在更大的范围内评估AVA的局限性，我们提供了对18名用户进行的一项调查的结果，我们发现大多数针对AVA的限制在实践中几乎没有使用。



## **27. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

基于普遍对抗性扰动的深度神经网络全局指纹识别 cs.CR

**SubmitDate**: 2022-02-22    [paper-pdf](http://arxiv.org/pdf/2202.08602v2)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its \textit{Universal Adversarial Perturbations (UAPs)}. UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via \textit{contrastive learning} that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence $> 99.99 \%$ within only $20$ fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.

摘要: 本文提出了一种新颖而实用的机制，使服务提供商能够通过模型提取攻击来验证受害者模型中的可疑模型是否被窃取。我们的主要见解是DNN模型的决策边界的轮廓可以由它的\textit(通用对抗性扰动(UAP))来唯一地刻画。UAP属于低维子空间，与非盗版模型相比，盗版模型的子空间与受害者模型的子空间更加一致。在此基础上，提出了一种DNN模型的UAP指纹识别方法，并以指纹为输入，通过对比学习训练编码器，输出相似度得分。大量的研究表明，我们的框架可以在可疑模型的$2 0$指纹范围内以>99.99$的置信度检测到模型IP泄露。它具有良好的跨不同模型体系结构的通用性，并且对窃取模型的后期修改具有健壮性。



## **28. Improving Robustness of Deep Reinforcement Learning Agents: Environment Attack based on the Critic Network**

提高深度强化学习代理的健壮性：基于批判网络的环境攻击 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.03154v2)

**Authors**: Lucas Schott, Hatem Hajri, Sylvain Lamprier

**Abstracts**: To improve policy robustness of deep reinforcement learning agents, a line of recent works focus on producing disturbances of the environment. Existing approaches of the literature to generate meaningful disturbances of the environment are adversarial reinforcement learning methods. These methods set the problem as a two-player game between the protagonist agent, which learns to perform a task in an environment, and the adversary agent, which learns to disturb the protagonist via modifications of the considered environment. Both protagonist and adversary are trained with deep reinforcement learning algorithms. Alternatively, we propose in this paper to build on gradient-based adversarial attacks, usually used for classification tasks for instance, that we apply on the critic network of the protagonist to identify efficient disturbances of the environment. Rather than learning an attacker policy, which usually reveals as very complex and unstable, we leverage the knowledge of the critic network of the protagonist, to dynamically complexify the task at each step of the learning process. We show that our method, while being faster and lighter, leads to significantly better improvements in policy robustness than existing methods of the literature.

摘要: 为了提高深度强化学习代理的策略鲁棒性，最近的一系列工作集中在产生环境扰动上。现有文献中产生有意义的环境扰动的方法是对抗性强化学习方法。这些方法将问题设置为主角Agent和对手Agent之间的两人博弈，前者学习在环境中执行任务，后者学习通过修改所考虑的环境来干扰主角。利用深度强化学习算法对主角和对手进行训练。或者，我们在本文中建议建立基于梯度的对抗性攻击，通常用于分类任务，例如，我们应用于主人公的批评网络来识别环境的有效干扰。我们不学习通常表现为非常复杂和不稳定的攻击者策略，而是利用主人公批评网络的知识，在学习过程的每一步动态地使任务复杂化。我们表明，虽然我们的方法更快、更轻，但与现有的文献方法相比，我们的方法在政策稳健性方面的改善要明显更好。



## **29. GasHis-Transformer: A Multi-scale Visual Transformer Approach for Gastric Histopathology Image Classification**

GasHis-Transformer：一种用于胃组织病理图像分类的多尺度视觉变换方法 cs.CV

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2104.14528v6)

**Authors**: Haoyuan Chen, Chen Li, Ge Wang, Xiaoyan Li, Md Rahaman, Hongzan Sun, Weiming Hu, Yixin Li, Wanli Liu, Changhao Sun, Shiliang Ai, Marcin Grzegorzek

**Abstracts**: Existing deep learning methods for diagnosis of gastric cancer commonly use convolutional neural network. Recently, the Visual Transformer has attracted great attention because of its performance and efficiency, but its applications are mostly in the field of computer vision. In this paper, a multi-scale visual transformer model, referred to as GasHis-Transformer, is proposed for Gastric Histopathological Image Classification (GHIC), which enables the automatic classification of microscopic gastric images into abnormal and normal cases. The GasHis-Transformer model consists of two key modules: A global information module and a local information module to extract histopathological features effectively. In our experiments, a public hematoxylin and eosin (H&E) stained gastric histopathological dataset with 280 abnormal and normal images are divided into training, validation and test sets by a ratio of 1 : 1 : 2. The GasHis-Transformer model is applied to estimate precision, recall, F1-score and accuracy on the test set of gastric histopathological dataset as 98.0%, 100.0%, 96.0% and 98.0%, respectively. Furthermore, a critical study is conducted to evaluate the robustness of GasHis-Transformer, where ten different noises including four adversarial attack and six conventional image noises are added. In addition, a clinically meaningful study is executed to test the gastrointestinal cancer identification performance of GasHis-Transformer with 620 abnormal images and achieves 96.8% accuracy. Finally, a comparative study is performed to test the generalizability with both H&E and immunohistochemical stained images on a lymphoma image dataset and a breast cancer dataset, producing comparable F1-scores (85.6% and 82.8%) and accuracies (83.9% and 89.4%), respectively. In conclusion, GasHisTransformer demonstrates high classification performance and shows its significant potential in the GHIC task.

摘要: 现有的胃癌诊断深度学习方法普遍采用卷积神经网络。近年来，视觉变压器因其高性能和高效率而备受关注，但其应用大多集中在计算机视觉领域。本文提出了一种用于胃组织病理图像分类(GHIC)的多尺度视觉转换器模型(简称GasHis-Transformer)，该模型能够自动将胃显微图像分类为异常和正常病例。GasHis-Transformer模型由两个关键模块组成：全局信息模块和局部信息模块，有效地提取组织病理学特征。在我们的实验中，一个公共的苏木精伊红(H&E)染色的胃组织病理学数据集以1：1：2的比例分为训练集、验证集和测试集，训练集、验证集和测试集的比例为1：1：2。应用GasHis-Transformer模型估计胃组织病理学数据集的准确率、召回率、F1得分和准确率分别为98.0%、100.0%、96.0%和98.0%。此外，还对GasHis-Transformer的稳健性进行了关键研究，添加了10种不同的噪声，包括4种对抗性攻击和6种常规图像噪声。另外，利用620幅异常图像对GasHis-Transformer的胃肠道肿瘤识别性能进行了有临床意义的测试，准确率达到96.8%。最后，在淋巴瘤图像数据集和乳腺癌数据集上对H&E和免疫组织化学染色图像的泛化能力进行了比较研究，得到了可比的F1得分(85.6%和82.8%)和准确率(83.9%和89.4%)。总之，GasHisTransformer表现出很高的分类性能，并在GHIC任务中显示出巨大的潜力。



## **30. Measuring the Transferability of $\ell_\infty$ Attacks by the $\ell_2$ Norm**

用$\ELL_2$范数度量$\ELL_\INFTY$攻击的可转移性 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2102.10343v3)

**Authors**: Sizhe Chen, Qinghua Tao, Zhixing Ye, Xiaolin Huang

**Abstracts**: Deep neural networks could be fooled by adversarial examples with trivial differences to original samples. To keep the difference imperceptible in human eyes, researchers bound the adversarial perturbations by the $\ell_\infty$ norm, which is now commonly served as the standard to align the strength of different attacks for a fair comparison. However, we propose that using the $\ell_\infty$ norm alone is not sufficient in measuring the attack strength, because even with a fixed $\ell_\infty$ distance, the $\ell_2$ distance also greatly affects the attack transferability between models. Through the discovery, we reach more in-depth understandings towards the attack mechanism, i.e., several existing methods attack black-box models better partly because they craft perturbations with 70\% to 130\% larger $\ell_2$ distances. Since larger perturbations naturally lead to better transferability, we thereby advocate that the strength of attacks should be simultaneously measured by both the $\ell_\infty$ and $\ell_2$ norm. Our proposal is firmly supported by extensive experiments on ImageNet dataset from 7 attacks, 4 white-box models, and 9 black-box models.

摘要: 深层神经网络可能会被与原始样本有微小差异的对抗性例子所欺骗。为了保持这种差异在人眼中不可察觉，研究人员用$\ell_\infty$范数来约束对抗性扰动，这现在通常被用作对不同攻击强度进行公平比较的标准。然而，我们认为单独使用$\ell_\infty$范数来度量攻击强度是不够的，因为即使在固定的$\ell_\infty$距离的情况下，$\ell_2$距离也会极大地影响攻击在模型之间的可转移性。通过这一发现，我们对攻击机制有了更深入的理解，即现有的几种方法对黑盒模型的攻击效果较好，部分原因是它们设计的扰动具有较大的$ell_2$70~130\{##**$}${##**$}}。由于更大的扰动自然会导致更好的可转移性，因此我们主张攻击的强度应该同时用$\ell_inty$和$\ell_2$范数来度量。我们的建议得到了来自7个攻击、4个白盒模型和9个黑盒模型的ImageNet数据集的广泛实验的坚定支持。



## **31. Towards Evaluating the Robustness of Neural Networks Learned by Transduction**

基于转导学习的神经网络鲁棒性评价方法研究 cs.LG

Paper published at ICLR 2022. arXiv admin note: text overlap with  arXiv:2106.08387

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2110.14735v2)

**Authors**: Jiefeng Chen, Xi Wu, Yang Guo, Yingyu Liang, Somesh Jha

**Abstracts**: There has been emerging interest in using transductive learning for adversarial robustness (Goldwasser et al., NeurIPS 2020; Wu et al., ICML 2020; Wang et al., ArXiv 2021). Compared to traditional defenses, these defense mechanisms "dynamically learn" the model based on test-time input; and theoretically, attacking these defenses reduces to solving a bilevel optimization problem, which poses difficulty in crafting adaptive attacks. In this paper, we examine these defense mechanisms from a principled threat analysis perspective. We formulate and analyze threat models for transductive-learning based defenses, and point out important subtleties. We propose the principle of attacking model space for solving bilevel attack objectives, and present Greedy Model Space Attack (GMSA), an attack framework that can serve as a new baseline for evaluating transductive-learning based defenses. Through systematic evaluation, we show that GMSA, even with weak instantiations, can break previous transductive-learning based defenses, which were resilient to previous attacks, such as AutoAttack. On the positive side, we report a somewhat surprising empirical result of "transductive adversarial training": Adversarially retraining the model using fresh randomness at the test time gives a significant increase in robustness against attacks we consider.

摘要: 人们对使用转导学习来实现对抗鲁棒性产生了新的兴趣(Goldwasser等人，NeurIPS 2020；Wu等人，ICML 2020；Wang等人，Arxiv 2021)。与传统防御相比，这些防御机制“动态学习”基于测试时间输入的模型；从理论上讲，攻击这些防御归结为求解一个双层优化问题，这给自适应攻击的设计带来了困难。在本文中，我们从原则性威胁分析的角度来检查这些防御机制。建立并分析了基于传导式学习防御的威胁模型，指出了重要的细微之处。我们提出了攻击模型空间求解双层攻击目标的原理，并提出了贪婪模型空间攻击(GMSA)这一攻击框架，可作为评估基于转导学习的防御的新基线。通过系统的评估，我们证明了GMSA即使在弱实例化的情况下，也可以打破以往基于传导式学习的防御机制，这些防御机制对AutoAttack等先前的攻击是有弹性的。从积极的一面来看，我们报告了一个有点令人惊讶的“转导对抗训练”的经验结果：在测试时使用新的随机性对模型进行对抗性重新训练，可以显著提高对我们所考虑的攻击的鲁棒性。



## **32. Generalizable Information Theoretic Causal Representation**

广义信息论因果表示 cs.LG

**SubmitDate**: 2022-02-17    [paper-pdf](http://arxiv.org/pdf/2202.08388v1)

**Authors**: Mengyue Yang, Xinyu Cai, Furui Liu, Xu Chen, Zhitang Chen, Jianye Hao, Jun Wang

**Abstracts**: It is evidence that representation learning can improve model's performance over multiple downstream tasks in many real-world scenarios, such as image classification and recommender systems. Existing learning approaches rely on establishing the correlation (or its proxy) between features and the downstream task (labels), which typically results in a representation containing cause, effect and spurious correlated variables of the label. Its generalizability may deteriorate because of the unstability of the non-causal parts. In this paper, we propose to learn causal representation from observational data by regularizing the learning procedure with mutual information measures according to our hypothetical causal graph. The optimization involves a counterfactual loss, based on which we deduce a theoretical guarantee that the causality-inspired learning is with reduced sample complexity and better generalization ability. Extensive experiments show that the models trained on causal representations learned by our approach is robust under adversarial attacks and distribution shift.

摘要: 事实证明，在图像分类和推荐系统等实际场景中，表征学习可以提高模型在多个下游任务上的性能。现有的学习方法依赖于在特征和下游任务(标签)之间建立相关性(或其代理)，这通常导致包含标签的原因、结果和虚假相关变量的表示。由于非因果部分的不稳定性，其泛化能力可能会恶化。在本文中，我们建议根据假设的因果图，通过互信息度量来规范学习过程，从而从观测数据中学习因果表示。这种优化涉及到反事实的损失，在此基础上，我们推导出因果启发学习具有更低的样本复杂度和更好的泛化能力的理论保证。大量实验表明，该方法训练的因果表示模型在对抗攻击和分布偏移情况下具有较好的鲁棒性。



## **33. Characterizing Attacks on Deep Reinforcement Learning**

深度强化学习攻击的特征描述 cs.LG

AAMAS 2022, 13 pages, 6 figures

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1907.09470v3)

**Authors**: Xinlei Pan, Chaowei Xiao, Warren He, Shuang Yang, Jian Peng, Mingjie Sun, Jinfeng Yi, Zijiang Yang, Mingyan Liu, Bo Li, Dawn Song

**Abstracts**: Recent studies show that Deep Reinforcement Learning (DRL) models are vulnerable to adversarial attacks, which attack DRL models by adding small perturbations to the observations. However, some attacks assume full availability of the victim model, and some require a huge amount of computation, making them less feasible for real world applications. In this work, we make further explorations of the vulnerabilities of DRL by studying other aspects of attacks on DRL using realistic and efficient attacks. First, we adapt and propose efficient black-box attacks when we do not have access to DRL model parameters. Second, to address the high computational demands of existing attacks, we introduce efficient online sequential attacks that exploit temporal consistency across consecutive steps. Third, we explore the possibility of an attacker perturbing other aspects in the DRL setting, such as the environment dynamics. Finally, to account for imperfections in how an attacker would inject perturbations in the physical world, we devise a method for generating a robust physical perturbations to be printed. The attack is evaluated on a real-world robot under various conditions. We conduct extensive experiments both in simulation such as Atari games, robotics and autonomous driving, and on real-world robotics, to compare the effectiveness of the proposed attacks with baseline approaches. To the best of our knowledge, we are the first to apply adversarial attacks on DRL systems to physical robots.

摘要: 最近的研究表明，深度强化学习(DRL)模型容易受到敌意攻击，这种攻击是通过在观测值中添加小扰动来攻击DRL模型的。然而，一些攻击假设受害者模型完全可用，而另一些攻击需要大量的计算，这使得它们在现实世界的应用程序中不太可行。在这项工作中，我们通过研究对DRL的其他方面的攻击，使用真实而有效的攻击，进一步探讨了DRL的漏洞。首先，当我们无法获得DRL模型参数时，我们采用并提出了有效的黑盒攻击。其次，为了解决现有攻击对计算的高要求，我们引入了高效的在线顺序攻击，该攻击利用了连续步骤之间的时间一致性。第三，我们探讨攻击者干扰DRL设置中其他方面的可能性，例如环境动态。最后，为了说明攻击者如何在物理世界中注入扰动的不完善之处，我们设计了一种生成要打印的健壮物理扰动的方法。在不同条件下对真实机器人进行了攻击评估。我们在Atari游戏、机器人和自动驾驶等模拟游戏中，以及在真实机器人上进行了广泛的实验，以比较所提出的攻击和基线方法的有效性。据我们所知，我们是第一个将针对DRL系统的对抗性攻击应用于物理机器人的公司。



## **34. Real-Time Neural Voice Camouflage**

实时神经语音伪装 cs.SD

14 pages

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2112.07076v2)

**Authors**: Mia Chiquier, Chengzhi Mao, Carl Vondrick

**Abstracts**: Automatic speech recognition systems have created exciting possibilities for applications, however they also enable opportunities for systematic eavesdropping. We propose a method to camouflage a person's voice over-the-air from these systems without inconveniencing the conversation between people in the room. Standard adversarial attacks are not effective in real-time streaming situations because the characteristics of the signal will have changed by the time the attack is executed. We introduce predictive attacks, which achieve real-time performance by forecasting the attack that will be the most effective in the future. Under real-time constraints, our method jams the established speech recognition system DeepSpeech 3.9x more than baselines as measured through word error rate, and 6.6x more as measured through character error rate. We furthermore demonstrate our approach is practically effective in realistic environments over physical distances.

摘要: 自动语音识别系统为应用创造了令人兴奋的可能性，然而它们也为系统窃听提供了机会。我们提出了一种方法，从这些系统中伪装出人的空中语音，而不会给房间里的人之间的对话带来不便。标准对抗性攻击在实时流情况下无效，因为在执行攻击时信号的特性将发生变化。我们引入预测性攻击，通过预测未来最有效的攻击来实现实时性能。在实时约束条件下，我们的方法对已建立的语音识别系统DeepSpeech的拥塞程度是基线的3.9倍，通过字符错误率的衡量是基线的6.6倍。我们进一步证明了我们的方法在物理距离上的现实环境中是实际有效的。



## **35. Ideal Tightly Couple (t,m,n) Secret Sharing**

理想紧耦合(t，m，n)秘密共享 cs.CR

few errors in the articles within the proposed scheme, and also  grammatical errors, so its our request pls withdraw our articles as soon as  possible

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/1905.02004v2)

**Authors**: Fuyou Miao, Keju Meng, Wenchao Huang, Yan Xiong, Xingfu Wang

**Abstracts**: As a fundamental cryptographic tool, (t,n)-threshold secret sharing ((t,n)-SS) divides a secret among n shareholders and requires at least t, (t<=n), of them to reconstruct the secret. Ideal (t,n)-SSs are most desirable in security and efficiency among basic (t,n)-SSs. However, an adversary, even without any valid share, may mount Illegal Participant (IP) attack or t/2-Private Channel Cracking (t/2-PCC) attack to obtain the secret in most (t,n)-SSs.To secure ideal (t,n)-SSs against the 2 attacks, 1) the paper introduces the notion of Ideal Tightly cOupled (t,m,n) Secret Sharing (or (t,m,n)-ITOSS ) to thwart IP attack without Verifiable SS; (t,m,n)-ITOSS binds all m, (m>=t), participants into a tightly coupled group and requires all participants to be legal shareholders before recovering the secret. 2) As an example, the paper presents a polynomial-based (t,m,n)-ITOSS scheme, in which the proposed k-round Random Number Selection (RNS) guarantees that adversaries have to crack at least symmetrical private channels among participants before obtaining the secret. Therefore, k-round RNS enhances the robustness of (t,m,n)-ITOSS against t/2-PCC attack to the utmost. 3) The paper finally presents a generalized method of converting an ideal (t,n)-SS into a (t,m,n)-ITOSS, which helps an ideal (t,n)-SS substantially improve the robustness against the above 2 attacks.

摘要: 作为一种基本的密码工具，(t，n)-门限秘密共享((t，n)-SS)将一个秘密分配给n个股东，并要求其中至少t个(t<=n)个股东重构秘密。在基本(t，n)-SS中，理想(t，n)-SS在安全性和效率方面是最理想的。为了保证理想(t，n)-SS不受这两种攻击的攻击，1)引入理想紧耦合(t，m，n)秘密共享(或(t，m，n)-ITOSS)的概念，在没有可验证SS的情况下阻止IP攻击；(t，m，n)-ITOSS将所有m，(m>=t)个参与者绑定到一个紧密耦合的组中，并要求所有参与者在恢复秘密之前都是合法股东。2)作为例子，提出了一个基于多项式的(t，m，n)-ITOSS方案，其中所提出的k轮随机数选择(RNS)方案保证攻击者在获得秘密之前必须至少破解参与者之间的对称私有信道。因此，k轮RNS最大限度地增强了(t，m，n)-ITOSS对t/2-PCC攻击的鲁棒性。3)最后给出了将理想(t，n)-SS转换为(t，m，n)-ITOSS的一般方法，从而大大提高了理想(t，n)-SS对上述两种攻击的鲁棒性。



## **36. Deduplicating Training Data Mitigates Privacy Risks in Language Models**

对训练数据进行重复数据消除可降低语言模型中的隐私风险 cs.CR

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06539v2)

**Authors**: Nikhil Kandpal, Eric Wallace, Colin Raffel

**Abstracts**: Past work has shown that large language models are susceptible to privacy attacks, where adversaries generate sequences from a trained model and detect which sequences are memorized from the training set. In this work, we show that the success of these attacks is largely due to duplication in commonly used web-scraped training sets. We first show that the rate at which language models regenerate training sequences is superlinearly related to a sequence's count in the training set. For instance, a sequence that is present 10 times in the training data is on average generated ~1000 times more often than a sequence that is present only once. We next show that existing methods for detecting memorized sequences have near-chance accuracy on non-duplicated training sequences. Finally, we find that after applying methods to deduplicate training data, language models are considerably more secure against these types of privacy attacks. Taken together, our results motivate an increased focus on deduplication in privacy-sensitive applications and a reevaluation of the practicality of existing privacy attacks.

摘要: 过去的工作表明，大型语言模型容易受到隐私攻击，攻击者从训练的模型生成序列，并从训练集中检测哪些序列被记忆。在这项工作中，我们表明这些攻击的成功在很大程度上是由于常用的Web抓取训练集的重复。我们首先证明了语言模型重新生成训练序列的速度与训练集中序列的计数呈超线性关系。例如，在训练数据中出现10次的序列的平均生成频率是只出现一次的序列的~1000倍。接下来，我们展示了现有的检测记忆序列的方法在非重复训练序列上具有近乎概率的准确性。最后，我们发现，在应用方法对训练数据进行去重之后，语言模型对这些类型的隐私攻击的安全性要高得多。综上所述，我们的结果促使人们更加关注隐私敏感应用程序中的重复数据删除，并重新评估现有隐私攻击的实用性。



## **37. The Adversarial Security Mitigations of mmWave Beamforming Prediction Models using Defensive Distillation and Adversarial Retraining**

基于防御蒸馏和对抗性再训练的毫米波波束形成预测模型的对抗性安全缓解 cs.CR

26 pages, under review

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08185v1)

**Authors**: Murat Kuzlu, Ferhat Ozgur Catak, Umit Cali, Evren Catak, Ozgur Guler

**Abstracts**: The design of a security scheme for beamforming prediction is critical for next-generation wireless networks (5G, 6G, and beyond). However, there is no consensus about protecting the beamforming prediction using deep learning algorithms in these networks. This paper presents the security vulnerabilities in deep learning for beamforming prediction using deep neural networks (DNNs) in 6G wireless networks, which treats the beamforming prediction as a multi-output regression problem. It is indicated that the initial DNN model is vulnerable against adversarial attacks, such as Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Projected Gradient Descent (PGD), and Momentum Iterative Method (MIM), because the initial DNN model is sensitive to the perturbations of the adversarial samples of the training data. This study also offers two mitigation methods, such as adversarial training and defensive distillation, for adversarial attacks against artificial intelligence (AI)-based models used in the millimeter-wave (mmWave) beamforming prediction. Furthermore, the proposed scheme can be used in situations where the data are corrupted due to the adversarial examples in the training data. Experimental results show that the proposed methods effectively defend the DNN models against adversarial attacks in next-generation wireless networks.

摘要: 波束成形预测安全方案的设计对下一代无线网络(5G、6G等)至关重要。然而，在这些网络中使用深度学习算法来保护波束形成预测并没有达成共识。针对6G无线网络中使用深度神经网络(DNNs)进行波束形成预测的深度学习中存在的安全漏洞，将波束形成预测处理为多输出回归问题。研究表明，由于初始DNN模型对训练数据对抗性样本的扰动比较敏感，因此容易受到敌意攻击，如快速梯度符号法(FGSM)、基本迭代法(BIM)、投影梯度下降法(PGD)和动量迭代法(MIM)。本研究还针对毫米波波束形成预测中使用的基于人工智能(AI)模型的对抗性攻击，提供了两种缓解方法，如对抗性训练和防御蒸馏。此外，所提出的方案还可以用于由于训练数据中的对抗性示例而导致数据被破坏的情况。实验结果表明，在下一代无线网络中，本文提出的方法有效地防御了DNN模型的攻击。



## **38. Finding Dynamics Preserving Adversarial Winning Tickets**

寻找动态保存的对抗性中奖彩票 cs.LG

Accepted by AISTATS2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06488v2)

**Authors**: Xupeng Shi, Pengfei Zheng, A. Adam Ding, Yuan Gao, Weizhong Zhang

**Abstracts**: Modern deep neural networks (DNNs) are vulnerable to adversarial attacks and adversarial training has been shown to be a promising method for improving the adversarial robustness of DNNs. Pruning methods have been considered in adversarial context to reduce model capacity and improve adversarial robustness simultaneously in training. Existing adversarial pruning methods generally mimic the classical pruning methods for natural training, which follow the three-stage 'training-pruning-fine-tuning' pipelines. We observe that such pruning methods do not necessarily preserve the dynamics of dense networks, making it potentially hard to be fine-tuned to compensate the accuracy degradation in pruning. Based on recent works of \textit{Neural Tangent Kernel} (NTK), we systematically study the dynamics of adversarial training and prove the existence of trainable sparse sub-network at initialization which can be trained to be adversarial robust from scratch. This theoretically verifies the \textit{lottery ticket hypothesis} in adversarial context and we refer such sub-network structure as \textit{Adversarial Winning Ticket} (AWT). We also show empirical evidences that AWT preserves the dynamics of adversarial training and achieve equal performance as dense adversarial training.

摘要: 现代深层神经网络(DNNs)容易受到敌意攻击，对抗性训练已被证明是提高DNN对抗性鲁棒性的一种很有前途的方法。在训练过程中，考虑了对抗性环境下的剪枝方法，在减少模型容量的同时提高对抗性鲁棒性。现有的对抗性剪枝方法一般是模仿经典的自然训练剪枝方法，遵循“训练-剪枝-微调”三阶段的流水线。我们观察到，这样的剪枝方法并不一定保持密集网络的动态，使得它可能很难被微调来补偿剪枝过程中的精度下降。基于神经切核(NTK)的最新工作，系统地研究了对抗性训练的动力学，证明了在初始化时存在可训练的稀疏子网络，它可以从头开始训练为对抗性健壮性网络。这从理论上验证了对抗性环境下的\text{彩票假设}，我们将这种子网络结构称为\text{对抗性中票}(AWT)。我们还展示了经验证据，AWT保持了对抗性训练的动态性，并获得了与密集对抗性训练相同的性能。



## **39. Neural Network Trojans Analysis and Mitigation from the Input Domain**

基于输入域的神经网络木马分析与消除 cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.06382v2)

**Authors**: Zhenting Wang, Hailun Ding, Juan Zhai, Shiqing Ma

**Abstracts**: Deep Neural Networks (DNNs) can learn Trojans (or backdoors) from benign or poisoned data, which raises security concerns of using them. By exploiting such Trojans, the adversary can add a fixed input space perturbation to any given input to mislead the model predicting certain outputs (i.e., target labels). In this paper, we analyze such input space Trojans in DNNs, and propose a theory to explain the relationship of a model's decision regions and Trojans: a complete and accurate Trojan corresponds to a hyperplane decision region in the input domain. We provide a formal proof of this theory, and provide empirical evidence to support the theory and its relaxations. Based on our analysis, we design a novel training method that removes Trojans during training even on poisoned datasets, and evaluate our prototype on five datasets and five different attacks. Results show that our method outperforms existing solutions. Code: \url{https://anonymous.4open.science/r/NOLE-84C3}.

摘要: 深度神经网络(DNNs)可以从良性或有毒的数据中学习特洛伊木马程序(或后门程序)，这增加了使用它们的安全问题。通过利用这种特洛伊木马，攻击者可以向任何给定的输入添加固定的输入空间扰动，以误导预测特定输出(即目标标签)的模型。本文分析了DNNs中的这类输入空间木马，提出了一种解释模型决策域与木马关系的理论：一个完整准确的木马对应于输入域中的一个超平面决策域。我们给出了这一理论的形式证明，并提供了支持该理论及其松弛的经验证据。基于我们的分析，我们设计了一种新的训练方法，即使在有毒的数据集上也能在训练过程中清除木马程序，并在五个数据集和五个不同的攻击上对我们的原型进行了评估。结果表明，我们的方法比已有的方法具有更好的性能。编码：\url{https://anonymous.4open.science/r/NOLE-84C3}.



## **40. Understanding and Improving Graph Injection Attack by Promoting Unnoticeability**

通过提高不可察觉来理解和改进图注入攻击 cs.LG

ICLR2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.08057v1)

**Authors**: Yongqiang Chen, Han Yang, Yonggang Zhang, Kaili Ma, Tongliang Liu, Bo Han, James Cheng

**Abstracts**: Recently Graph Injection Attack (GIA) emerges as a practical attack scenario on Graph Neural Networks (GNNs), where the adversary can merely inject few malicious nodes instead of modifying existing nodes or edges, i.e., Graph Modification Attack (GMA). Although GIA has achieved promising results, little is known about why it is successful and whether there is any pitfall behind the success. To understand the power of GIA, we compare it with GMA and find that GIA can be provably more harmful than GMA due to its relatively high flexibility. However, the high flexibility will also lead to great damage to the homophily distribution of the original graph, i.e., similarity among neighbors. Consequently, the threats of GIA can be easily alleviated or even prevented by homophily-based defenses designed to recover the original homophily. To mitigate the issue, we introduce a novel constraint -- homophily unnoticeability that enforces GIA to preserve the homophily, and propose Harmonious Adversarial Objective (HAO) to instantiate it. Extensive experiments verify that GIA with HAO can break homophily-based defenses and outperform previous GIA attacks by a significant margin. We believe our methods can serve for a more reliable evaluation of the robustness of GNNs.

摘要: 图注入攻击(GIA)是近年来在图神经网络(GNNs)上出现的一种实用攻击方案，即攻击者只能注入少量的恶意节点，而不需要修改已有的节点或边，即图修改攻击(GMA)。尽管GIA取得了令人振奋的成果，但人们对其成功的原因以及成功背后是否存在陷阱知之甚少。为了理解GIA的力量，我们将其与GMA进行比较，发现由于其相对较高的灵活性，GIA显然比GMA更具危害性。但是，较高的灵活性也会对原图的同源分布造成很大的破坏，即邻域间的相似性。因此，GIA的威胁可以很容易地减轻，甚至可以通过基于同源的防御措施来恢复原始的同源。为了缓解这一问题，我们引入了一种新的约束--同形不可察觉，强制GIA保持同形，并提出了和谐对抗目标(HAO)来实例化它。广泛的实验证明，带有HAO的GIA可以打破基于同源的防御，并显著超过以前的GIA攻击。我们相信我们的方法可以更可靠地评估GNNs的健壮性。



## **41. Increasing-Margin Adversarial (IMA) Training to Improve Adversarial Robustness of Neural Networks**

提高神经网络对抗鲁棒性的增量对抗性(IMA)训练 cs.CV

45 pages, 15 figures, 31 tables

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2005.09147v7)

**Authors**: Linhai Ma, Liang Liang

**Abstracts**: Convolutional neural network (CNN) has surpassed traditional methods for medical image classification. However, CNN is vulnerable to adversarial attacks which may lead to disastrous consequences in medical applications. Although adversarial noises are usually generated by attack algorithms, white-noise-induced adversarial samples can exist, and therefore the threats are real. In this study, we propose a novel training method, named IMA, to improve the robust-ness of CNN against adversarial noises. During training, the IMA method increases the margins of training samples in the input space, i.e., moving CNN decision boundaries far away from the training samples to improve robustness. The IMA method is evaluated on publicly available datasets under strong 100-PGD white-box adversarial attacks, and the results show that the proposed method significantly improved CNN classification and segmentation accuracy on noisy data while keeping a high accuracy on clean data. We hope our approach may facilitate the development of robust applications in medical field.

摘要: 卷积神经网络(CNN)已经超越了传统的医学图像分类方法。然而，CNN很容易受到对抗性攻击，这可能会导致医疗应用中的灾难性后果。虽然攻击算法通常会产生对抗性噪声，但白噪声诱导的对抗性样本可能存在，因此威胁是真实存在的。在这项研究中，我们提出了一种新的训练方法，称为IMA，以提高CNN对对抗性噪声的鲁棒性。在训练过程中，IMA方法增加了输入空间中训练样本的边际，即使CNN决策边界远离训练样本，以提高鲁棒性。在100-PGD强白盒攻击下，在公开数据集上对IMA方法进行了评估，结果表明，该方法在保持对干净数据较高精度的同时，显著提高了对含噪声数据的CNN分类和分割的准确率。我们希望我们的方法可以促进医学领域健壮应用的发展。



## **42. Backdoor Learning: A Survey**

借壳学习：一项调查 cs.CR

17 pages. A curated list of backdoor learning resources in this paper  is presented in the Github Repo  (https://github.com/THUYimingLi/backdoor-learning-resources). We will try our  best to continuously maintain this Github Repo

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2007.08745v5)

**Authors**: Yiming Li, Yong Jiang, Zhifeng Li, Shu-Tao Xia

**Abstracts**: Backdoor attack intends to embed hidden backdoor into deep neural networks (DNNs), so that the attacked models perform well on benign samples, whereas their predictions will be maliciously changed if the hidden backdoor is activated by attacker-specified triggers. This threat could happen when the training process is not fully controlled, such as training on third-party datasets or adopting third-party models, which poses a new and realistic threat. Although backdoor learning is an emerging and rapidly growing research area, its systematic review, however, remains blank. In this paper, we present the first comprehensive survey of this realm. We summarize and categorize existing backdoor attacks and defenses based on their characteristics, and provide a unified framework for analyzing poisoning-based backdoor attacks. Besides, we also analyze the relation between backdoor attacks and relevant fields ($i.e.,$ adversarial attacks and data poisoning), and summarize widely adopted benchmark datasets. Finally, we briefly outline certain future research directions relying upon reviewed works. A curated list of backdoor-related resources is also available at \url{https://github.com/THUYimingLi/backdoor-learning-resources}.

摘要: 后门攻击的目的是将隐藏的后门嵌入到深度神经网络(DNNs)中，使得被攻击的模型在良性样本上表现良好，而如果隐藏的后门被攻击者指定的触发器激活，则其预测将被恶意改变。这种威胁可能发生在培训过程没有得到完全控制时，例如在第三方数据集上进行培训或采用第三方模型，这会构成新的现实威胁。虽然借壳学习是一个新兴的、发展迅速的研究领域，但其系统评价仍然是空白。在这篇文章中，我们首次对这一领域进行了全面的调查。根据后门攻击和防御的特点，对现有的后门攻击和防御进行了总结和分类，为分析基于中毒的后门攻击提供了一个统一的框架。此外，我们还分析了后门攻击与相关领域($对抗性攻击和数据中毒)之间的关系，并总结了广泛采用的基准数据集。最后，在回顾工作的基础上，简要概述了未来的研究方向。\url{https://github.com/THUYimingLi/backdoor-learning-resources}.上还提供了与后门相关的资源的精选列表



## **43. FedCG: Leverage Conditional GAN for Protecting Privacy and Maintaining Competitive Performance in Federated Learning**

FedCG：利用条件GAN保护隐私并保持联合学习中的好胜性能 cs.LG

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2111.08211v2)

**Authors**: Yuezhou Wu, Yan Kang, Jiahuan Luo, Yuanqin He, Qiang Yang

**Abstracts**: Federated learning (FL) aims to protect data privacy by enabling clients to build machine learning models collaboratively without sharing their private data. Recent works demonstrate that information exchanged during FL is subject to gradient-based privacy attacks and, consequently, a variety of privacy-preserving methods have been adopted to thwart such attacks. However, these defensive methods either introduce orders of magnitudes more computational and communication overheads (e.g., with homomorphic encryption) or incur substantial model performance losses in terms of prediction accuracy (e.g., with differential privacy). In this work, we propose $\textsc{FedCG}$, a novel federated learning method that leverages conditional generative adversarial networks to achieve high-level privacy protection while still maintaining competitive model performance. $\textsc{FedCG}$ decomposes each client's local network into a private extractor and a public classifier and keeps the extractor local to protect privacy. Instead of exposing extractors, $\textsc{FedCG}$ shares clients' generators with the server for aggregating clients' shared knowledge aiming to enhance the performance of each client's local networks. Extensive experiments demonstrate that $\textsc{FedCG}$ can achieve competitive model performance compared with FL baselines, and privacy analysis shows that $\textsc{FedCG}$ has a high-level privacy-preserving capability.

摘要: 联合学习(FL)旨在通过使客户能够在不共享其私有数据的情况下协作地构建机器学习模型来保护数据隐私。最近的研究表明，在外语学习过程中交换的信息会受到基于梯度的隐私攻击，因此，已经采取了各种隐私保护方法来阻止这种攻击。然而，这些防御方法要么引入更多数量级的计算和通信开销(例如，利用同态加密)，要么在预测精度方面招致大量的模型性能损失(例如，利用差分保密)。在这项工作中，我们提出了一种新的联邦学习方法$\textsc{fedcg}$，它利用条件生成性对抗网络来实现高级别的隐私保护，同时保持好胜模型的性能。$\textsc{fedcg}$将每个客户端的本地网络分解为私有提取器和公共分类器，并将提取器保留在本地以保护隐私。$\textsc{FedCG}$不公开提取器，而是与服务器共享客户端生成器，用于聚合客户端共享的知识，旨在增强每个客户端的本地网络的性能。大量实验表明，与FL基线相比，$\textsc{fedcg}$能够达到好胜模型的性能，隐私分析表明$\textsc{fedcg}$具有较高的隐私保护能力。



## **44. Generative Adversarial Network-Driven Detection of Adversarial Tasks in Mobile Crowdsensing**

生成式对抗性网络驱动的移动树冠感知对抗性任务检测 cs.CR

This paper contains pages, 4 figures which is accepted by IEEE ICC  2022

**SubmitDate**: 2022-02-16    [paper-pdf](http://arxiv.org/pdf/2202.07802v1)

**Authors**: Zhiyan Chen, Burak Kantarci

**Abstracts**: Mobile Crowdsensing systems are vulnerable to various attacks as they build on non-dedicated and ubiquitous properties. Machine learning (ML)-based approaches are widely investigated to build attack detection systems and ensure MCS systems security. However, adversaries that aim to clog the sensing front-end and MCS back-end leverage intelligent techniques, which are challenging for MCS platform and service providers to develop appropriate detection frameworks against these attacks. Generative Adversarial Networks (GANs) have been applied to generate synthetic samples, that are extremely similar to the real ones, deceiving classifiers such that the synthetic samples are indistinguishable from the originals. Previous works suggest that GAN-based attacks exhibit more crucial devastation than empirically designed attack samples, and result in low detection rate at the MCS platform. With this in mind, this paper aims to detect intelligently designed illegitimate sensing service requests by integrating a GAN-based model. To this end, we propose a two-level cascading classifier that combines the GAN discriminator with a binary classifier to prevent adversarial fake tasks. Through simulations, we compare our results to a single-level binary classifier, and the numeric results show that proposed approach raises Adversarial Attack Detection Rate (AADR), from $0\%$ to $97.5\%$ by KNN/NB, from $45.9\%$ to $100\%$ by Decision Tree. Meanwhile, with two-levels classifiers, Original Attack Detection Rate (OADR) improves for the three binary classifiers, with comparison, such as NB from $26.1\%$ to $61.5\%$.

摘要: 由于移动树冠传感系统建立在非专用和无处不在的特性之上，因此容易受到各种攻击。基于机器学习(ML)的方法在构建攻击检测系统和保证MCS系统安全方面得到了广泛的研究。然而，旨在阻塞传感前端和MCS后端的攻击者利用智能技术，这对MCS平台和服务提供商开发针对这些攻击的适当检测框架是具有挑战性的。生成性对抗网络(GANS)被用来生成与真实样本极其相似的合成样本，欺骗分类器，使得合成样本与原始样本无法区分。以往的工作表明，基于GAN的攻击比经验设计的攻击样本表现出更严重的破坏性，导致MCS平台的检测率较低。考虑到这一点，本文旨在通过集成一个基于GAN的模型来检测智能设计的非法传感服务请求。为此，我们提出了一种将GAN鉴别器和二进制分类器相结合的两级级联分类器，以防止敌意虚假任务。通过仿真，我们将我们的结果与单级二进制分类器进行了比较，数值结果表明，该方法将对手攻击检测率(AADR)从0美元提高到97.5美元(KNN/NB从0美元提高到97.5美元)，通过决策树将AADR从4 5.9美元提高到1 0 0美元。同时，在使用两级分类器的情况下，三种二值分类器的原始攻击检测率(OADR)都有不同程度的提高，例如NB从26.1美元提高到61.5美元。



## **45. Vulnerability-Aware Poisoning Mechanism for Online RL with Unknown Dynamics**

动态未知的在线RL漏洞感知中毒机制 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2009.00774v5)

**Authors**: Yanchao Sun, Da Huo, Furong Huang

**Abstracts**: Poisoning attacks on Reinforcement Learning (RL) systems could take advantage of RL algorithm's vulnerabilities and cause failure of the learning. However, prior works on poisoning RL usually either unrealistically assume the attacker knows the underlying Markov Decision Process (MDP), or directly apply the poisoning methods in supervised learning to RL. In this work, we build a generic poisoning framework for online RL via a comprehensive investigation of heterogeneous poisoning models in RL. Without any prior knowledge of the MDP, we propose a strategic poisoning algorithm called Vulnerability-Aware Adversarial Critic Poison (VA2C-P), which works for most policy-based deep RL agents, closing the gap that no poisoning method exists for policy-based RL agents. VA2C-P uses a novel metric, stability radius in RL, that measures the vulnerability of RL algorithms. Experiments on multiple deep RL agents and multiple environments show that our poisoning algorithm successfully prevents agents from learning a good policy or teaches the agents to converge to a target policy, with a limited attacking budget.

摘要: 对强化学习(RL)系统的毒化攻击可以利用RL算法的脆弱性，导致学习失败。然而，以往的毒化RL的工作通常要么不切实际地假设攻击者知道潜在的马尔可夫决策过程(MDP)，要么直接将有监督学习中的毒化方法应用于RL。在这项工作中，我们通过对RL中异构中毒模型的全面研究，构建了一个适用于在线RL的通用中毒框架。在对MDP没有任何先验知识的情况下，我们提出了一种策略毒化算法VA2C-P(VA2C-P)，该算法适用于大多数基于策略的深度RL代理，弥补了基于策略的RL代理没有毒化方法的空白。VA2C-P使用了一种新的度量，即RL中的稳定半径，该度量度量了RL算法的脆弱性。在多个深度RL代理和多个环境上的实验表明，我们的中毒算法在有限的攻击预算下，成功地阻止了代理学习好的策略或教导代理收敛到目标策略。



## **46. Defending against Reconstruction Attacks with Rényi Differential Privacy**

利用Rényi差分私密性防御重构攻击 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07623v1)

**Authors**: Pierre Stock, Igor Shilov, Ilya Mironov, Alexandre Sablayrolles

**Abstracts**: Reconstruction attacks allow an adversary to regenerate data samples of the training set using access to only a trained model. It has been recently shown that simple heuristics can reconstruct data samples from language models, making this threat scenario an important aspect of model release. Differential privacy is a known solution to such attacks, but is often used with a relatively large privacy budget (epsilon > 8) which does not translate to meaningful guarantees. In this paper we show that, for a same mechanism, we can derive privacy guarantees for reconstruction attacks that are better than the traditional ones from the literature. In particular, we show that larger privacy budgets do not protect against membership inference, but can still protect extraction of rare secrets. We show experimentally that our guarantees hold against various language models, including GPT-2 finetuned on Wikitext-103.

摘要: 重构攻击允许对手仅使用对训练模型的访问来重新生成训练集的数据样本。最近的研究表明，简单的启发式算法可以从语言模型中重建数据样本，从而使这种威胁场景成为模型发布的一个重要方面。差异隐私是此类攻击的已知解决方案，但通常使用相对较大的隐私预算(epsilon>8)，这并不能转化为有意义的保证。在本文中，我们表明，对于相同的机制，我们可以从文献中推导出比传统的重构攻击更好的隐私保证。特别是，我们表明，较大的隐私预算不能防止成员关系推断，但仍然可以保护罕见秘密的提取。我们的实验表明，我们的保证适用于各种语言模型，包括在Wikitext-103上微调的GPT-2。



## **47. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07568v1)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image processing domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring defenses focuses on feature-based, gradient-based or randomized methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a Moving Target Defense and Game Theory approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型抗敌意攻击的研究大多集中在图像处理领域。恶意软件检测域尽管很重要，但受到的关注较少。而且，大多数探索防御的工作都集中在基于特征的、基于梯度的或随机的方法上，而在应用这些方法时没有策略。本文介绍了StratDef，这是一个基于移动目标防御和博弈论的针对恶意软件检测领域定制的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的健壮性。StratDef动态地和战略性地选择最佳模型，以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们在恶意软件检测的机器学习中首次全面评估了防御敌意攻击的能力，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最严重的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御来看，只有少数经过对抗性训练的模型提供了比仅仅使用普通模型更好的保护，但仍然优于StratDef。



## **48. Random Walks for Adversarial Meshes**

对抗性网格的随机游动 cs.CV

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07453v1)

**Authors**: Amir Belder, Gal Yefet, Ran Ben Izhak, Ayellet Tal

**Abstracts**: A polygonal mesh is the most-commonly used representation of surfaces in computer graphics; thus, a variety of classification networks have been recently proposed. However, while adversarial attacks are wildly researched in 2D, almost no works on adversarial meshes exist. This paper proposes a novel, unified, and general adversarial attack, which leads to misclassification of numerous state-of-the-art mesh classification neural networks. Our attack approach is black-box, i.e. it has access only to the network's predictions, but not to the network's full architecture or gradients. The key idea is to train a network to imitate a given classification network. This is done by utilizing random walks along the mesh surface, which gather geometric information. These walks provide insight onto the regions of the mesh that are important for the correct prediction of the given classification network. These mesh regions are then modified more than other regions in order to attack the network in a manner that is barely visible to the naked eye.

摘要: 多边形网格是计算机图形学中最常用的曲面表示，因此，最近提出了各种分类网络。然而，尽管对抗性攻击在2D方面得到了广泛的研究，但几乎没有关于对抗性网络的工作。本文提出了一种新颖的、统一的、通用的对抗性攻击，该攻击导致了众多最新的网格分类神经网络的误分类。我们的攻击方法是黑匣子，即它只能访问网络的预测，而不能访问网络的完整架构或梯度。其核心思想是训练一个网络来模仿给定的分类网络。这是通过利用沿网格曲面的随机漫游来完成的，该漫游收集几何信息。这些遍历提供了对网格区域的洞察，这些区域对于给定分类网络的正确预测非常重要。然后，这些网格区域被修改得比其他区域更多，以便以肉眼几乎看不见的方式攻击网络。



## **49. Unreasonable Effectiveness of Last Hidden Layer Activations**

最后一次隐藏层激活的不合理效果 cs.LG

22 pages, Under review

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2202.07342v1)

**Authors**: Omer Faruk Tuna, Ferhat Ozgur Catak, M. Taner Eskil

**Abstracts**: In standard Deep Neural Network (DNN) based classifiers, the general convention is to omit the activation function in the last (output) layer and directly apply the softmax function on the logits to get the probability scores of each class. In this type of architectures, the loss value of the classifier against any output class is directly proportional to the difference between the final probability score and the label value of the associated class. Standard White-box adversarial evasion attacks, whether targeted or untargeted, mainly try to exploit the gradient of the model loss function to craft adversarial samples and fool the model. In this study, we show both mathematically and experimentally that using some widely known activation functions in the output layer of the model with high temperature values has the effect of zeroing out the gradients for both targeted and untargeted attack cases, preventing attackers from exploiting the model's loss function to craft adversarial samples. We've experimentally verified the efficacy of our approach on MNIST (Digit), CIFAR10 datasets. Detailed experiments confirmed that our approach substantially improves robustness against gradient-based targeted and untargeted attack threats. And, we showed that the increased non-linearity at the output layer has some additional benefits against some other attack methods like Deepfool attack.

摘要: 在标准的基于深度神经网络(DNN)的分类器中，一般的做法是省略最后一层(输出层)的激活函数，直接对Logit应用Softmax函数来得到每一类的概率得分。在这种类型的体系结构中，分类器相对于任何输出类别的损失值与最终概率得分和相关类别的标签值之间的差值成正比。标准的白盒对抗性规避攻击，无论是有针对性的还是无针对性的，主要是利用模型损失函数的梯度来伪造对抗性样本，愚弄模型。在这项研究中，我们从数学和实验两个方面证明了在具有高温值的模型输出层使用一些广为人知的激活函数具有将目标攻击和非目标攻击的梯度归零的效果，防止攻击者利用模型的损失函数来伪造敌意样本。我们已经在MNIST(数字)、CIFAR10数据集上实验验证了我们的方法的有效性。详细的实验证实，我们的方法大大提高了对基于梯度的目标攻击和非目标攻击威胁的鲁棒性。并且，我们还表明，在输出层增加的非线性比其他一些攻击方法(如DeepfoOff攻击)有一些额外的好处。



## **50. Unity is strength: Improving the Detection of Adversarial Examples with Ensemble Approaches**

团结就是力量：用集成方法改进对抗性实例的检测 cs.CV

Code is available at https://github.com/BIMIB-DISCo/ENAD-experiments

**SubmitDate**: 2022-02-15    [paper-pdf](http://arxiv.org/pdf/2111.12631v3)

**Authors**: Francesco Craighero, Fabrizio Angaroni, Fabio Stella, Chiara Damiani, Marco Antoniotti, Alex Graudenzi

**Abstracts**: A key challenge in computer vision and deep learning is the definition of robust strategies for the detection of adversarial examples. Here, we propose the adoption of ensemble approaches to leverage the effectiveness of multiple detectors in exploiting distinct properties of the input data. To this end, the ENsemble Adversarial Detector (ENAD) framework integrates scoring functions from state-of-the-art detectors based on Mahalanobis distance, Local Intrinsic Dimensionality, and One-Class Support Vector Machines, which process the hidden features of deep neural networks. ENAD is designed to ensure high standardization and reproducibility to the computational workflow. Importantly, extensive tests on benchmark datasets, models and adversarial attacks show that ENAD outperforms all competing methods in the large majority of settings. The improvement over the state-of-the-art and the intrinsic generality of the framework, which allows one to easily extend ENAD to include any set of detectors, set the foundations for the new area of ensemble adversarial detection.

摘要: 计算机视觉和深度学习中的一个关键挑战是定义用于检测对抗性示例的鲁棒策略。在这里，我们建议采用集成方法来利用多个检测器的有效性来利用输入数据的不同属性。为此，集成敌意检测器(ENAD)框架集成了基于马氏距离、局部本征维数和一类支持向量机的最新检测器的评分函数，这些功能处理了深层神经网络的隐藏特征。ENAD旨在确保计算工作流的高度标准化和重复性。重要的是，对基准数据集、模型和对抗性攻击的广泛测试表明，ENAD在绝大多数情况下都优于所有竞争方法。对现有技术的改进和框架固有的通用性，使得人们可以很容易地将ENAD扩展到包括任何一组检测器，为集成对手检测的新领域奠定了基础。



