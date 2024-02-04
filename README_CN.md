# Latest Adversarial Attack Papers
**update at 2024-02-04 13:14:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. ALISON: Fast and Effective Stylometric Authorship Obfuscation**

ALISON：快速有效的风格作者混淆 cs.CL

10 pages, 6 figures, 4 tables. To be published in the Proceedings of  the 38th Annual AAAI Conference on Artificial Intelligence (AAAI-24)

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00835v1) [paper-pdf](http://arxiv.org/pdf/2402.00835v1)

**Authors**: Eric Xing, Saranya Venkatraman, Thai Le, Dongwon Lee

**Abstract**: Authorship Attribution (AA) and Authorship Obfuscation (AO) are two competing tasks of increasing importance in privacy research. Modern AA leverages an author's consistent writing style to match a text to its author using an AA classifier. AO is the corresponding adversarial task, aiming to modify a text in such a way that its semantics are preserved, yet an AA model cannot correctly infer its authorship. To address privacy concerns raised by state-of-the-art (SOTA) AA methods, new AO methods have been proposed but remain largely impractical to use due to their prohibitively slow training and obfuscation speed, often taking hours. To this challenge, we propose a practical AO method, ALISON, that (1) dramatically reduces training/obfuscation time, demonstrating more than 10x faster obfuscation than SOTA AO methods, (2) achieves better obfuscation success through attacking three transformer-based AA methods on two benchmark datasets, typically performing 15% better than competing methods, (3) does not require direct signals from a target AA classifier during obfuscation, and (4) utilizes unique stylometric features, allowing sound model interpretation for explainable obfuscation. We also demonstrate that ALISON can effectively prevent four SOTA AA methods from accurately determining the authorship of ChatGPT-generated texts, all while minimally changing the original text semantics. To ensure the reproducibility of our findings, our code and data are available at: https://github.com/EricX003/ALISON.

摘要: 作者身份归属(AA)和作者身份混淆(AO)是隐私研究中日益重要的两个相互竞争的任务。现代AA利用作者一贯的写作风格，使用AA分类器将文本与其作者进行匹配。人工智能是相应的对抗性任务，旨在修改文本，使其语义保持不变，但AA模型不能正确推断其作者。为了解决最先进的(SOTA)AA方法引起的隐私问题，已经提出了新的AO方法，但由于其训练和混淆速度太慢，通常需要几个小时，因此在很大程度上仍然不实用。为了应对这一挑战，我们提出了一种实用的AO方法Alison，它(1)显著减少了训练/混淆时间，表现出比Sota AO方法快10倍以上；(2)通过攻击两个基准数据集上的三个基于变压器的AA方法获得了更好的混淆成功，通常比竞争方法的性能高15%；(3)在混淆过程中不需要来自目标AA分类器的直接信号；(4)利用独特的风格特征，允许对可解释的混淆进行声音模型解释。我们还证明了Alison可以有效地防止四种Sota AA方法在最小限度地改变原始文本语义的情况下准确地确定ChatGPT生成的文本的作者。为了确保我们研究结果的重现性，我们的代码和数据可在以下网址获得：https://github.com/EricX003/ALISON.



## **2. Tropical Decision Boundaries for Neural Networks Are Robust Against Adversarial Attacks**

神经网络热带决策边界对敌方攻击的健壮性 cs.LG

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00576v1) [paper-pdf](http://arxiv.org/pdf/2402.00576v1)

**Authors**: Kurt Pasque, Christopher Teska, Ruriko Yoshida, Keiji Miura, Jefferson Huang

**Abstract**: We introduce a simple, easy to implement, and computationally efficient tropical convolutional neural network architecture that is robust against adversarial attacks. We exploit the tropical nature of piece-wise linear neural networks by embedding the data in the tropical projective torus in a single hidden layer which can be added to any model. We study the geometry of its decision boundary theoretically and show its robustness against adversarial attacks on image datasets using computational experiments.

摘要: 我们介绍了一种简单、易于实现、计算高效的热带卷积神经网络结构，该结构对对手攻击具有健壮性。我们利用分段线性神经网络的热带性质，将数据嵌入到热带投影环面中的单个隐层中，该隐层可以添加到任何模型中。我们从理论上研究了它的决策边界的几何性质，并通过计算实验证明了它对图像数据集上的敌意攻击的稳健性。



## **3. Short: Benchmarking transferable adversarial attacks**

简短：对可转移的对抗性攻击进行基准测试 cs.CV

Accepted by NDSS 2024 Workshop

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00418v1) [paper-pdf](http://arxiv.org/pdf/2402.00418v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Huaming Chen

**Abstract**: The robustness of deep learning models against adversarial attacks remains a pivotal concern. This study presents, for the first time, an exhaustive review of the transferability aspect of adversarial attacks. It systematically categorizes and critically evaluates various methodologies developed to augment the transferability of adversarial attacks. This study encompasses a spectrum of techniques, including Generative Structure, Semantic Similarity, Gradient Editing, Target Modification, and Ensemble Approach. Concurrently, this paper introduces a benchmark framework \textit{TAA-Bench}, integrating ten leading methodologies for adversarial attack transferability, thereby providing a standardized and systematic platform for comparative analysis across diverse model architectures. Through comprehensive scrutiny, we delineate the efficacy and constraints of each method, shedding light on their underlying operational principles and practical utility. This review endeavors to be a quintessential resource for both scholars and practitioners in the field, charting the complex terrain of adversarial transferability and setting a foundation for future explorations in this vital sector. The associated codebase is accessible at: https://github.com/KxPlaug/TAA-Bench

摘要: 深度学习模型对敌意攻击的稳健性仍然是一个关键问题。这项研究首次对对抗性攻击的可转移性进行了详尽的回顾。它系统地分类和批判性地评价了为加强对抗性攻击的可转移性而开发的各种方法。这项研究涵盖了一系列技术，包括生成结构、语义相似性、梯度编辑、目标修改和集成方法。同时，本文引入了一个基准框架\TAA-BENCH，集成了十种主流的对抗性攻击可转移性方法，从而为跨不同模型体系结构的比较分析提供了一个标准化和系统化的平台。通过全面的审查，我们描述了每种方法的有效性和制约因素，揭示了它们潜在的操作原理和实用价值。本综述努力成为该领域学者和实践者的典型资源，描绘了对抗性可转移性的复杂地形，并为未来在这一重要领域的探索奠定了基础。相关的代码库可在以下网址访问：https://github.com/KxPlaug/TAA-Bench



## **4. Hidding the Ghostwriters: An Adversarial Evaluation of AI-Generated Student Essay Detection**

隐藏代写人：人工智能生成的学生作文检测的对抗性评估 cs.CL

Accepted by EMNLP 2023 Main conference, Oral Presentation

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00412v1) [paper-pdf](http://arxiv.org/pdf/2402.00412v1)

**Authors**: Xinlin Peng, Ying Zhou, Ben He, Le Sun, Yingfei Sun

**Abstract**: Large language models (LLMs) have exhibited remarkable capabilities in text generation tasks. However, the utilization of these models carries inherent risks, including but not limited to plagiarism, the dissemination of fake news, and issues in educational exercises. Although several detectors have been proposed to address these concerns, their effectiveness against adversarial perturbations, specifically in the context of student essay writing, remains largely unexplored. This paper aims to bridge this gap by constructing AIG-ASAP, an AI-generated student essay dataset, employing a range of text perturbation methods that are expected to generate high-quality essays while evading detection. Through empirical experiments, we assess the performance of current AIGC detectors on the AIG-ASAP dataset. The results reveal that the existing detectors can be easily circumvented using straightforward automatic adversarial attacks. Specifically, we explore word substitution and sentence substitution perturbation methods that effectively evade detection while maintaining the quality of the generated essays. This highlights the urgent need for more accurate and robust methods to detect AI-generated student essays in the education domain.

摘要: 大型语言模型(LLM)在文本生成任务中表现出了非凡的能力。然而，利用这些模式存在固有的风险，包括但不限于抄袭、传播假新闻和教育练习中的问题。虽然已经提出了几种检测器来解决这些问题，但它们对抗对抗性干扰的有效性，特别是在学生作文中的有效性，在很大程度上还没有被探索。本文旨在通过构建人工智能生成的学生作文数据集AIG-ASAP来弥合这一差距，该数据库使用了一系列文本扰动方法，有望在避免检测的同时生成高质量的作文。通过实验，我们评估了现有的AIGC检测器在AIG-ASAP数据集上的性能。结果表明，现有的检测器可以很容易地通过直接的自动对抗性攻击来绕过。具体地说，我们探索了单词替换和句子替换扰动方法，这些方法在保持生成的论文质量的同时有效地躲避了检测。这突显出迫切需要更准确和更强大的方法来检测教育领域中人工智能生成的学生作文。



## **5. Comparing Spectral Bias and Robustness For Two-Layer Neural Networks: SGD vs Adaptive Random Fourier Features**

两层神经网络频谱偏差和稳健性的比较：SGD与自适应随机傅立叶特征 cs.LG

6 Pages, 4 Figures; Accepted in the International Conference on  Scientific Computing and Machine Learning

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00332v1) [paper-pdf](http://arxiv.org/pdf/2402.00332v1)

**Authors**: Aku Kammonen, Lisi Liang, Anamika Pandey, Raúl Tempone

**Abstract**: We present experimental results highlighting two key differences resulting from the choice of training algorithm for two-layer neural networks. The spectral bias of neural networks is well known, while the spectral bias dependence on the choice of training algorithm is less studied. Our experiments demonstrate that an adaptive random Fourier features algorithm (ARFF) can yield a spectral bias closer to zero compared to the stochastic gradient descent optimizer (SGD). Additionally, we train two identically structured classifiers, employing SGD and ARFF, to the same accuracy levels and empirically assess their robustness against adversarial noise attacks.

摘要: 我们给出了实验结果，突出了两个关键的差异，这两个差异是由于两层神经网络的训练算法的选择造成的。神经网络的谱偏差是众所周知的，而谱偏差对训练算法选择的依赖关系研究较少。我们的实验表明，与随机梯度下降优化器(SGD)相比，自适应随机傅立叶特征算法(ARFF)可以产生更接近于零的频谱偏差。此外，我们训练了两个相同结构的分类器，使用SGD和ARFF，使其达到相同的精度水平，并经验地评估了它们对对抗噪声攻击的稳健性。



## **6. Invariance-powered Trustworthy Defense via Remove Then Restore**

通过删除然后恢复的不变性驱动的可信防御 cs.CV

**SubmitDate**: 2024-02-01    [abs](http://arxiv.org/abs/2402.00304v1) [paper-pdf](http://arxiv.org/pdf/2402.00304v1)

**Authors**: Xiaowei Fu, Yuhang Zhou, Lina Ma, Lei Zhang

**Abstract**: Adversarial attacks pose a challenge to the deployment of deep neural networks (DNNs), while previous defense models overlook the generalization to various attacks. Inspired by targeted therapies for cancer, we view adversarial samples as local lesions of natural benign samples, because a key finding is that salient attack in an adversarial sample dominates the attacking process, while trivial attack unexpectedly provides trustworthy evidence for obtaining generalizable robustness. Based on this finding, a Pixel Surgery and Semantic Regeneration (PSSR) model following the targeted therapy mechanism is developed, which has three merits: 1) To remove the salient attack, a score-based Pixel Surgery module is proposed, which retains the trivial attack as a kind of invariance information. 2) To restore the discriminative content, a Semantic Regeneration module based on a conditional alignment extrapolator is proposed, which achieves pixel and semantic consistency. 3) To further harmonize robustness and accuracy, an intractable problem, a self-augmentation regularizer with adversarial R-drop is designed. Experiments on numerous benchmarks show the superiority of PSSR.

摘要: 对抗性攻击对深度神经网络(DNN)的部署提出了挑战，而以往的防御模型忽略了对各种攻击的泛化。受癌症靶向治疗的启发，我们将对抗性样本视为天然良性样本的局部病变，因为一个关键发现是，对抗性样本中的显著攻击主导了攻击过程，而平凡的攻击意外地为获得泛化的稳健性提供了可靠的证据。在此基础上，提出了一种遵循靶向治疗机制的像素手术和语义再生(PSSR)模型，该模型有三个优点：1)为了去除显著攻击，提出了一种基于分数的像素手术模型，将琐碎攻击保留为一种不变性信息。2)提出了一种基于条件对齐外推器的语义再生模块，实现了像素和语义的一致性。3)为了进一步协调鲁棒性和准确性这一棘手问题，设计了一种对抗性R-Drop自增强正则化算法。在多个基准上的实验表明了PSSR的优越性。



## **7. Adversarial Quantum Machine Learning: An Information-Theoretic Generalization Analysis**

对抗性量子机器学习：一个信息论的推广分析 quant-ph

10 pages, 2 figures

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2402.00176v1) [paper-pdf](http://arxiv.org/pdf/2402.00176v1)

**Authors**: Petros Georgiou, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: In a manner analogous to their classical counterparts, quantum classifiers are vulnerable to adversarial attacks that perturb their inputs. A promising countermeasure is to train the quantum classifier by adopting an attack-aware, or adversarial, loss function. This paper studies the generalization properties of quantum classifiers that are adversarially trained against bounded-norm white-box attacks. Specifically, a quantum adversary maximizes the classifier's loss by transforming an input state $\rho(x)$ into a state $\lambda$ that is $\epsilon$-close to the original state $\rho(x)$ in $p$-Schatten distance. Under suitable assumptions on the quantum embedding $\rho(x)$, we derive novel information-theoretic upper bounds on the generalization error of adversarially trained quantum classifiers for $p = 1$ and $p = \infty$. The derived upper bounds consist of two terms: the first is an exponential function of the 2-R\'enyi mutual information between classical data and quantum embedding, while the second term scales linearly with the adversarial perturbation size $\epsilon$. Both terms are shown to decrease as $1/\sqrt{T}$ over the training set size $T$ . An extension is also considered in which the adversary assumed during training has different parameters $p$ and $\epsilon$ as compared to the adversary affecting the test inputs. Finally, we validate our theoretical findings with numerical experiments for a synthetic setting.

摘要: 与经典分类器类似，量子分类器很容易受到对手的攻击，扰乱它们的输入。一种有希望的对策是通过采用攻击感知或对抗性损失函数来训练量子分类器。研究了反向训练的量子分类器抵抗有界范数白盒攻击的泛化性质。具体地说，量子对手通过将输入状态$\rho(X)$转换为$\epsilon$-接近$p$-Schatten距离中的原始状态$\rho(X)$来最大化分类器的损失。在适当的量子嵌入假设下，我们得到了对抗性训练的量子分类器对$p=1$和$p=inty$的泛化误差的新的信息论上界。得到的上界由两项组成：第一项是经典数据和量子嵌入之间的2-R‘Enyi互信息的指数函数，第二项与对抗性扰动的大小成线性关系。这两项在训练集大小为$T$的情况下都减小了$1/\Sqrt{T}$。我们还考虑了一种扩展，其中假设的对手方在训练过程中与影响测试输入的对手方相比具有不同的参数$p$和$\epsilon$。最后，我们用数值实验验证了我们的理论结果。



## **8. Privacy Risks Analysis and Mitigation in Federated Learning for Medical Images**

医学图像联合学习中的隐私风险分析与缓解 cs.LG

V1

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2311.06643v2) [paper-pdf](http://arxiv.org/pdf/2311.06643v2)

**Authors**: Badhan Chandra Das, M. Hadi Amini, Yanzhao Wu

**Abstract**: Federated learning (FL) is gaining increasing popularity in the medical domain for analyzing medical images, which is considered an effective technique to safeguard sensitive patient data and comply with privacy regulations. However, several recent studies have revealed that the default settings of FL may leak private training data under privacy attacks. Thus, it is still unclear whether and to what extent such privacy risks of FL exist in the medical domain, and if so, "how to mitigate such risks?". In this paper, first, we propose a holistic framework for Medical data Privacy risk analysis and mitigation in Federated Learning (MedPFL) to analyze privacy risks and develop effective mitigation strategies in FL for protecting private medical data. Second, we demonstrate the substantial privacy risks of using FL to process medical images, where adversaries can easily perform privacy attacks to reconstruct private medical images accurately. Third, we show that the defense approach of adding random noises may not always work effectively to protect medical images against privacy attacks in FL, which poses unique and pressing challenges associated with medical data for privacy protection.

摘要: 联合学习(FL)在医学领域的医学图像分析中越来越受欢迎，它被认为是保护敏感患者数据和遵守隐私法规的有效技术。然而，最近的一些研究表明，在隐私攻击下，FL的默认设置可能会泄露私人训练数据。因此，目前仍不清楚FL在医疗领域是否存在这种隐私风险，以及在多大程度上存在这种风险，如果存在的话，“如何减轻这种风险？”本文首先提出了一种联邦学习中医疗数据隐私风险分析和缓解的整体框架(MedPFL)，以分析联邦学习中的隐私风险，并在FL中制定有效的缓解策略来保护私人医疗数据。其次，我们展示了使用FL处理医学图像的巨大隐私风险，在这种情况下，攻击者可以很容易地执行隐私攻击来准确重建私人医学图像。第三，我们发现，在FL中添加随机噪声的防御方法并不总是有效地保护医学图像免受隐私攻击，这对隐私保护提出了与医疗数据相关的独特而紧迫的挑战。



## **9. Elephants Do Not Forget: Differential Privacy with State Continuity for Privacy Budget**

大象不会忘记：具有国家连续性的差别隐私预算 cs.CR

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17628v1) [paper-pdf](http://arxiv.org/pdf/2401.17628v1)

**Authors**: Jiankai Jin, Chitchanok Chuengsatiansup, Toby Murray, Benjamin I. P. Rubinstein, Yuval Yarom, Olga Ohrimenko

**Abstract**: Current implementations of differentially-private (DP) systems either lack support to track the global privacy budget consumed on a dataset, or fail to faithfully maintain the state continuity of this budget. We show that failure to maintain a privacy budget enables an adversary to mount replay, rollback and fork attacks - obtaining answers to many more queries than what a secure system would allow. As a result the attacker can reconstruct secret data that DP aims to protect - even if DP code runs in a Trusted Execution Environment (TEE). We propose ElephantDP, a system that aims to provide the same guarantees as a trusted curator in the global DP model would, albeit set in an untrusted environment. Our system relies on a state continuity module to provide protection for the privacy budget and a TEE to faithfully execute DP code and update the budget. To provide security, our protocol makes several design choices including the content of the persistent state and the order between budget updates and query answers. We prove that ElephantDP provides liveness (i.e., the protocol can restart from a correct state and respond to queries as long as the budget is not exceeded) and DP confidentiality (i.e., an attacker learns about a dataset as much as it would from interacting with a trusted curator). Our implementation and evaluation of the protocol use Intel SGX as a TEE to run the DP code and a network of TEEs to maintain state continuity. Compared to an insecure baseline, we observe only 1.1-2$\times$ overheads and lower relative overheads for larger datasets and complex DP queries.

摘要: 当前的差分隐私（DP）系统的实现要么缺乏对跟踪数据集上消耗的全局隐私预算的支持，要么无法忠实地保持该预算的状态连续性。我们表明，未能保持隐私预算使对手能够安装重放，回滚和分叉攻击-获得更多的查询的答案比一个安全的系统将允许。因此，攻击者可以重建DP旨在保护的秘密数据-即使DP代码在可信执行环境（TEE）中运行。我们提出了ElephantDP，一个系统，旨在提供相同的保证作为一个可信的馆长在全球DP模型，虽然设置在一个不可信的环境。我们的系统依赖于一个状态连续性模块，以提供保护的隐私预算和TEE忠实地执行DP代码和更新的预算。为了提供安全性，我们的协议做出了几个设计选择，包括持久状态的内容和预算更新和查询答案之间的顺序。我们证明了ElephantDP提供了活性（即，协议可以从正确的状态重新开始并且只要不超过预算就对查询作出响应）和DP机密性（即，攻击者从与可信管理者的交互中尽可能多地了解数据集）。我们对该协议的实施和评估使用英特尔SGX作为TEE来运行DP代码，并使用TEE网络来保持状态连续性。与不安全的基线相比，我们观察到只有1.1-2$\times$开销和较低的相对开销较大的数据集和复杂的DP查询。



## **10. Game-Theoretic Unlearnable Example Generator**

博弈论的不可学习示例生成器 cs.LG

**SubmitDate**: 2024-01-31    [abs](http://arxiv.org/abs/2401.17523v1) [paper-pdf](http://arxiv.org/pdf/2401.17523v1)

**Authors**: Shuang Liu, Yihan Wang, Xiao-Shan Gao

**Abstract**: Unlearnable example attacks are data poisoning attacks aiming to degrade the clean test accuracy of deep learning by adding imperceptible perturbations to the training samples, which can be formulated as a bi-level optimization problem. However, directly solving this optimization problem is intractable for deep neural networks. In this paper, we investigate unlearnable example attacks from a game-theoretic perspective, by formulating the attack as a nonzero sum Stackelberg game. First, the existence of game equilibria is proved under the normal setting and the adversarial training setting. It is shown that the game equilibrium gives the most powerful poison attack in that the victim has the lowest test accuracy among all networks within the same hypothesis space, when certain loss functions are used. Second, we propose a novel attack method, called the Game Unlearnable Example (GUE), which has three main gradients. (1) The poisons are obtained by directly solving the equilibrium of the Stackelberg game with a first-order algorithm. (2) We employ an autoencoder-like generative network model as the poison attacker. (3) A novel payoff function is introduced to evaluate the performance of the poison. Comprehensive experiments demonstrate that GUE can effectively poison the model in various scenarios. Furthermore, the GUE still works by using a relatively small percentage of the training data to train the generator, and the poison generator can generalize to unseen data well. Our implementation code can be found at https://github.com/hong-xian/gue.

摘要: 不可学示例攻击是一种数据中毒攻击，其目的是通过在训练样本中添加不可察觉的扰动来降低深度学习的清洁测试精度，这可以被描述为一个双层优化问题。然而，对于深度神经网络来说，直接求解这一优化问题是一件棘手的事情。本文从博弈论的角度研究了不可学实例攻击，将其描述为一个非零和Stackelberg博弈。首先，证明了在正常情况下和对抗性训练情况下博弈均衡的存在性。结果表明，当使用一定的损失函数时，博弈均衡给出了最强的毒物攻击，在相同假设空间内的所有网络中，受害者的测试精度最低。其次，我们提出了一种新的攻击方法，称为游戏不可学习示例(GUE)，它有三个主要梯度。(1)用一阶算法直接求解Stackelberg博弈的均衡，得到毒物。(2)采用一种类似自动编码器的产生式网络模型作为有毒攻击者。(3)引入了一种新的支付函数来评价毒药的性能。综合实验表明，GUE在各种场景下都能有效地毒化模型。此外，GUE仍然通过使用相对较小百分比的训练数据来训练生成器来工作，并且有毒生成器可以很好地推广到看不见的数据。我们的实现代码可以在https://github.com/hong-xian/gue.上找到



## **11. AdvGPS: Adversarial GPS for Multi-Agent Perception Attack**

AdvGPS：多智能体感知攻击的对抗性GPS cs.CV

Accepted by the 2024 IEEE International Conference on Robotics and  Automation (ICRA)

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17499v1) [paper-pdf](http://arxiv.org/pdf/2401.17499v1)

**Authors**: Jinlong Li, Baolu Li, Xinyu Liu, Jianwu Fang, Felix Juefei-Xu, Qing Guo, Hongkai Yu

**Abstract**: The multi-agent perception system collects visual data from sensors located on various agents and leverages their relative poses determined by GPS signals to effectively fuse information, mitigating the limitations of single-agent sensing, such as occlusion. However, the precision of GPS signals can be influenced by a range of factors, including wireless transmission and obstructions like buildings. Given the pivotal role of GPS signals in perception fusion and the potential for various interference, it becomes imperative to investigate whether specific GPS signals can easily mislead the multi-agent perception system. To address this concern, we frame the task as an adversarial attack challenge and introduce \textsc{AdvGPS}, a method capable of generating adversarial GPS signals which are also stealthy for individual agents within the system, significantly reducing object detection accuracy. To enhance the success rates of these attacks in a black-box scenario, we introduce three types of statistically sensitive natural discrepancies: appearance-based discrepancy, distribution-based discrepancy, and task-aware discrepancy. Our extensive experiments on the OPV2V dataset demonstrate that these attacks substantially undermine the performance of state-of-the-art methods, showcasing remarkable transferability across different point cloud based 3D detection systems. This alarming revelation underscores the pressing need to address security implications within multi-agent perception systems, thereby underscoring a critical area of research.

摘要: 多智能体感知系统从位于各种智能体上的传感器收集视觉数据，并利用由GPS信号确定的它们的相对姿态来有效地融合信息，减轻单智能体感测的限制，例如遮挡。然而，GPS信号的精度可能会受到一系列因素的影响，包括无线传输和建筑物等障碍物。鉴于GPS信号在感知融合中的关键作用和各种干扰的可能性，研究特定的GPS信号是否容易误导多智能体感知系统变得势在必行。为了解决这一问题，我们将该任务定义为对抗性攻击挑战，并引入了\textsc{AdvGPS}，这是一种能够生成对抗性GPS信号的方法，该信号对于系统内的单个代理也是隐身的，显著降低了对象检测精度。为了提高这些攻击在黑盒场景中的成功率，我们引入了三种类型的统计敏感的自然差异：基于外观的差异，基于分布的差异，和任务感知的差异。我们在OPV 2 V数据集上进行的大量实验表明，这些攻击大大破坏了最先进方法的性能，在不同的基于点云的3D检测系统之间显示出显著的可移植性。这一令人震惊的启示强调了迫切需要解决多智能体感知系统中的安全问题，从而强调了一个关键的研究领域。



## **12. Camouflage Adversarial Attacks on Multiple Agent Systems**

伪装对多智能体系统的敌意攻击 cs.MA

arXiv admin note: text overlap with arXiv:2311.00859

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17405v1) [paper-pdf](http://arxiv.org/pdf/2401.17405v1)

**Authors**: Ziqing Lu, Guanlin Liu, Lifeng Lai, Weiyu Xu

**Abstract**: The multi-agent reinforcement learning systems (MARL) based on the Markov decision process (MDP) have emerged in many critical applications. To improve the robustness/defense of MARL systems against adversarial attacks, the study of various adversarial attacks on reinforcement learning systems is very important. Previous works on adversarial attacks considered some possible features to attack in MDP, such as the action poisoning attacks, the reward poisoning attacks, and the state perception attacks. In this paper, we propose a brand-new form of attack called the camouflage attack in the MARL systems. In the camouflage attack, the attackers change the appearances of some objects without changing the actual objects themselves; and the camouflaged appearances may look the same to all the targeted recipient (victim) agents. The camouflaged appearances can mislead the recipient agents to misguided actions. We design algorithms that give the optimal camouflage attacks minimizing the rewards of recipient agents. Our numerical and theoretical results show that camouflage attacks can rival the more conventional, but likely more difficult state perception attacks. We also investigate cost-constrained camouflage attacks and showed numerically how cost budgets affect the attack performance.

摘要: 基于马尔可夫决策过程(MDP)的多智能体强化学习系统(MAIL)已经出现在许多关键应用中。为了提高MAIL系统对敌意攻击的稳健性/防御性，对强化学习系统进行各种对抗性攻击的研究是非常重要的。以往关于对抗性攻击的工作考虑了MDP中可能的一些攻击特征，如动作中毒攻击、奖励中毒攻击和状态感知攻击。在本文中，我们提出了一种全新的攻击形式，称为MAIL系统中的伪装攻击。在伪装攻击中，攻击者改变一些对象的外观，而不改变实际对象本身；伪装的外观对所有目标接收者(受害者)代理来说可能是相同的。伪装的外表可能会误导收件人代理采取误导的行动。我们设计的算法可以给出最优伪装攻击，最小化接收方代理的回报。我们的数值和理论结果表明，伪装攻击可以与更传统但可能更困难的状态感知攻击相媲美。我们还研究了成本受限的伪装攻击，并用数字说明了成本预算如何影响攻击性能。



## **13. Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks**

防御越狱攻击的语言模型的稳健提示优化 cs.LG

code available at https://github.com/andyz245/rpo

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17263v1) [paper-pdf](http://arxiv.org/pdf/2401.17263v1)

**Authors**: Andy Zhou, Bo Li, Haohan Wang

**Abstract**: Despite advances in AI alignment, language models (LM) remain vulnerable to adversarial attacks or jailbreaking, in which adversaries modify input prompts to induce harmful behavior. While some defenses have been proposed, they focus on narrow threat models and fall short of a strong defense, which we posit should be effective, universal, and practical. To achieve this, we propose the first adversarial objective for defending LMs against jailbreaking attacks and an algorithm, robust prompt optimization (RPO), that uses gradient-based token optimization to enforce harmless outputs. This results in an easily accessible suffix that significantly improves robustness to both jailbreaks seen during optimization and unknown, held-out jailbreaks, reducing the attack success rate on Starling-7B from 84% to 8.66% across 20 jailbreaks. In addition, we find that RPO has a minor effect on normal LM use, is successful under adaptive attacks, and can transfer to black-box models, reducing the success rate of the strongest attack on GPT-4 from 92% to 6%.

摘要: 尽管在人工智能对齐方面取得了进展，但语言模型(LM)仍然容易受到对抗性攻击或越狱，在这些攻击或越狱中，对手修改输入提示以诱导有害行为。虽然已经提出了一些防御措施，但它们侧重于狭隘的威胁模型，缺乏强大的防御措施，我们认为这些防御措施应该是有效的、通用的和实用的。为了实现这一点，我们提出了防御LMS免受越狱攻击的第一个对抗性目标和一个算法，即稳健提示优化(RPO)，它使用基于梯度的令牌优化来强制执行无害的输出。这导致了一个易于访问的后缀，显著提高了对优化过程中看到的越狱和未知的、坚持的越狱的稳健性，将Starling-7B在20次越狱中的攻击成功率从84%降低到8.66%。此外，我们发现RPO对正常的LM使用的影响很小，在自适应攻击下是成功的，并且可以转换到黑盒模型，将对GPT-4的最强攻击的成功率从92%降低到6%。



## **14. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的从弱到强的越狱 cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17256v1) [paper-pdf](http://arxiv.org/pdf/2401.17256v1)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs. The efficacy of this attack is demonstrated through experiments conducted on five models from three different organizations. Our study reveals a previously unnoticed yet efficient way of jailbreaking, exposing an urgent safety issue that needs to be considered when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 尽管已经致力于调整大型语言模型(LLM)，但红队报告表明，这些精心调整的LLM仍然可能通过敌意提示、调整或解码而越狱。在考察了对齐LLM的越狱脆弱性后，我们观察到，越狱模型和对齐模型的解码分布仅在最初几代中有所不同。这一观察结果促使我们提出了从弱到强的越狱攻击，其中对手可以利用较小的不安全/对齐的LLM(例如，7B)来指导对较大的对齐的LLM(例如，70B)的越狱。要越狱，只需额外解码两个较小的LLM一次，与解码较大的LLM相比，这涉及的计算量和延迟最小。通过对来自三个不同组织的五个模型进行的实验，证明了该攻击的有效性。我们的研究揭示了一种以前未被注意但有效的越狱方式，暴露了一个紧急的安全问题，在调整LLM时需要考虑这个问题。作为最初的尝试，我们提出了一种防御战略来防御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上找到



## **15. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

只需更改一个单词即可：为文本分类器设计攻击和防御 cs.CL

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17196v1) [paper-pdf](http://arxiv.org/pdf/2401.17196v1)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.

摘要: 在文本分类中，创建对抗性示例意味着在不改变句子含义的情况下巧妙地干扰句子中的几个单词，从而导致分类器错误分类。一个令人担忧的观察是，现有方法生成的对抗性示例中有很大一部分只改变了一个单词。这个单字扰动漏洞代表了分类器的一个显著弱点，恶意用户可以利用它来有效地创建大量对抗性示例。本文研究了这一问题，并做出了以下主要贡献：（1）我们引入了一个新的度量\r{ho}来定量评估分类器对单字扰动的鲁棒性。(2)我们提出了SP-Attack，旨在利用单字扰动漏洞，实现更高的攻击成功率，更好地保留句子含义，同时与最先进的对抗方法相比降低了计算成本。(3)我们提出了SP-Defense，旨在通过在学习中应用数据增强来改善\r{ho}。在4个数据集和BERT、distilBERT分类器上的实验结果表明，SP-Defense在两个分类器上分别将\r{ho}提高了14.6%和13.9%，将SP-Attack的攻击成功率降低了30.4%和21.2%，并降低了现有多词扰动攻击方法的攻击成功率.



## **16. Adversarial Machine Learning in Latent Representations of Neural Networks**

神经网络潜在表示中的对抗性机器学习 cs.LG

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2309.17401v3) [paper-pdf](http://arxiv.org/pdf/2309.17401v3)

**Authors**: Milin Zhang, Mohammad Abdi, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge the resilience of distributed DNNs to adversarial action still remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and introduce two new measurements for distortion and robustness. Our theoretical findings indicate that (i) assuming the same level of information distortion, latent features are always more robust than input representations; (ii) the adversarial robustness is jointly determined by the feature dimension and the generalization capability of the DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks to the ImageNet-1K dataset. Our experimental results support our theoretical findings by showing that the compressed latent representations can reduce the success rate of adversarial attacks by 88% in the best case and by 57% on the average compared to attacks to the input space.

摘要: 分布式深度神经网络可以减轻移动设备的计算负担，减少边缘计算场景中的端到端推理延迟。虽然已经对分布式DNN进行了研究，但就我们所知，分布式DNN对敌意行为的恢复能力仍然是一个悬而未决的问题。在本文中，我们通过严格分析分布式DNN对攻击行为的健壮性来填补现有的研究空白。我们把这个问题放在信息论的背景下，并引入了两个新的失真和稳健性度量。我们的理论结果表明：(I)假设信息失真程度相同，潜在特征总是比输入表示更健壮；(Ii)DNN的对抗健壮性由特征维度和泛化能力共同决定。为了验证我们的理论发现，我们通过考虑6种不同的DNN体系结构、6种不同的分布式DNN方法和10种不同的针对ImageNet-1K数据集的对手攻击进行了广泛的实验分析。我们的实验结果支持我们的理论发现，与对输入空间的攻击相比，压缩的潜在表示在最好的情况下可以使对抗性攻击的成功率降低88%，平均降低57%。



## **17. Systematically Assessing the Security Risks of AI/ML-enabled Connected Healthcare Systems**

系统评估支持AI/ML的互联医疗系统的安全风险 cs.CR

13 pages, 5 figures, 3 tables

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17136v1) [paper-pdf](http://arxiv.org/pdf/2401.17136v1)

**Authors**: Mohammed Elnawawy, Mohammadreza Hallajiyan, Gargi Mitra, Shahrear Iqbal, Karthik Pattabiraman

**Abstract**: The adoption of machine-learning-enabled systems in the healthcare domain is on the rise. While the use of ML in healthcare has several benefits, it also expands the threat surface of medical systems. We show that the use of ML in medical systems, particularly connected systems that involve interfacing the ML engine with multiple peripheral devices, has security risks that might cause life-threatening damage to a patient's health in case of adversarial interventions. These new risks arise due to security vulnerabilities in the peripheral devices and communication channels. We present a case study where we demonstrate an attack on an ML-enabled blood glucose monitoring system by introducing adversarial data points during inference. We show that an adversary can achieve this by exploiting a known vulnerability in the Bluetooth communication channel connecting the glucose meter with the ML-enabled app. We further show that state-of-the-art risk assessment techniques are not adequate for identifying and assessing these new risks. Our study highlights the need for novel risk analysis methods for analyzing the security of AI-enabled connected health devices.

摘要: 在医疗保健领域，采用机器学习系统的情况正在上升。虽然ML在医疗保健中的使用有几个好处，但它也扩大了医疗系统的威胁表面。我们表明，在医疗系统中使用ML，特别是涉及将ML引擎与多个外围设备接口的互联系统，具有安全风险，如果进行对抗性干预，可能会对患者的健康造成危及生命的损害。这些新的风险是由于外围设备和通信通道中的安全漏洞造成的。我们提供了一个案例研究，通过在推理过程中引入敌对数据点来演示对启用ML的血糖监测系统的攻击。我们表明，攻击者可以通过利用连接血糖仪和支持ML的应用程序的蓝牙通信通道中的已知漏洞来实现这一点。我们进一步表明，最先进的风险评估技术不足以识别和评估这些新风险。我们的研究突显了需要新的风险分析方法来分析支持人工智能的联网医疗设备的安全性。



## **18. What can Information Guess? Guessing Advantage vs. Rényi Entropy for Small Leakages**

信息能猜到什么？小泄漏的猜测优势与Rényi熵 cs.IT

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17057v1) [paper-pdf](http://arxiv.org/pdf/2401.17057v1)

**Authors**: Julien Béguinot, Olivier Rioul

**Abstract**: We leverage the Gibbs inequality and its natural generalization to R\'enyi entropies to derive closed-form parametric expressions of the optimal lower bounds of $\rho$th-order guessing entropy (guessing moment) of a secret taking values on a finite set, in terms of the R\'enyi-Arimoto $\alpha$-entropy. This is carried out in an non-asymptotic regime when side information may be available. The resulting bounds yield a theoretical solution to a fundamental problem in side-channel analysis: Ensure that an adversary will not gain much guessing advantage when the leakage information is sufficiently weakened by proper countermeasures in a given cryptographic implementation. Practical evaluation for classical leakage models show that the proposed bounds greatly improve previous ones for analyzing the capability of an adversary to perform side-channel attacks.

摘要: 我们利用Gibbs不等式及其对R‘Enyi熵的自然推广，以R’Enyi-Arimoto$α$-熵的形式，导出了有限集上取值的秘密的最优下界$‘Enyi-Arimoto$\α$-熵的闭式参数表达式.当可以获得辅助信息时，这是在非渐近的制度下进行的。由此得到的边界为边信道分析中的一个基本问题提供了一个理论解决方案：在给定的密码实现中，当泄漏的信息被适当的对策充分削弱时，确保对手不会获得太多的猜测优势。对经典泄漏模型的实际评估表明，所提出的边界极大地改进了以前用于分析对手执行侧信道攻击的能力的边界。



## **19. Towards Assessing the Synthetic-to-Measured Adversarial Vulnerability of SAR ATR**

评估合成孔径雷达ATR的合成-测量对抗脆弱性 cs.CV

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17038v1) [paper-pdf](http://arxiv.org/pdf/2401.17038v1)

**Authors**: Bowen Peng, Bo Peng, Jingyuan Xia, Tianpeng Liu, Yongxiang Liu, Li Liu

**Abstract**: Recently, there has been increasing concern about the vulnerability of deep neural network (DNN)-based synthetic aperture radar (SAR) automatic target recognition (ATR) to adversarial attacks, where a DNN could be easily deceived by clean input with imperceptible but aggressive perturbations. This paper studies the synthetic-to-measured (S2M) transfer setting, where an attacker generates adversarial perturbation based solely on synthetic data and transfers it against victim models trained with measured data. Compared with the current measured-to-measured (M2M) transfer setting, our approach does not need direct access to the victim model or the measured SAR data. We also propose the transferability estimation attack (TEA) to uncover the adversarial risks in this more challenging and practical scenario. The TEA makes full use of the limited similarity between the synthetic and measured data pairs for blind estimation and optimization of S2M transferability, leading to feasible surrogate model enhancement without mastering the victim model and data. Comprehensive evaluations based on the publicly available synthetic and measured paired labeled experiment (SAMPLE) dataset demonstrate that the TEA outperforms state-of-the-art methods and can significantly enhance various attack algorithms in computer vision and remote sensing applications. Codes and data are available at https://github.com/scenarri/S2M-TEA.

摘要: 近年来，基于深度神经网络(DNN)的合成孔径雷达(SAR)自动目标识别(ATR)在敌方攻击中的脆弱性受到越来越多的关注，在这种攻击中，DNN很容易被干净的输入和不可感知的攻击性扰动所欺骗。本文研究了合成到测量(S2M)传输设置，其中攻击者仅基于合成数据产生对抗性扰动，并将其与使用测量数据训练的受害者模型进行传输。与当前的测量到测量(M2M)传输设置相比，我们的方法不需要直接访问受害者模型或测量的SAR数据。我们还提出了可转移性估计攻击(TEA)，以揭示这种更具挑战性和实用性的场景中的对抗性风险。TEA充分利用合成数据对和测量数据对之间有限的相似性进行S2M转移能力的盲估计和优化，从而在不掌握受害者模型和数据的情况下实现可行的代理模型增强。基于公开可用的合成和测量的成对标记实验(样本)数据集的综合评估表明，TEA的性能优于最先进的方法，并可以显著增强计算机视觉和遥感应用中的各种攻击算法。有关代码和数据，请访问https://github.com/scenarri/S2M-TEA.



## **20. Quantum Transfer Learning with Adversarial Robustness for Classification of High-Resolution Image Datasets**

具有对抗鲁棒性的量子迁移学习用于高分辨率图像数据集分类 quant-ph

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.17009v1) [paper-pdf](http://arxiv.org/pdf/2401.17009v1)

**Authors**: Amena Khatun, Muhammad Usman

**Abstract**: The application of quantum machine learning to large-scale high-resolution image datasets is not yet possible due to the limited number of qubits and relatively high level of noise in the current generation of quantum devices. In this work, we address this challenge by proposing a quantum transfer learning (QTL) architecture that integrates quantum variational circuits with a classical machine learning network pre-trained on ImageNet dataset. Through a systematic set of simulations over a variety of image datasets such as Ants & Bees, CIFAR-10, and Road Sign Detection, we demonstrate the superior performance of our QTL approach over classical and quantum machine learning without involving transfer learning. Furthermore, we evaluate the adversarial robustness of QTL architecture with and without adversarial training, confirming that our QTL method is adversarially robust against data manipulation attacks and outperforms classical methods.

摘要: 由于当前一代量子器件中量子比特数量有限，噪声水平相对较高，将量子机器学习应用于大规模高分辨率图像数据集还不可能。在这项工作中，我们通过提出一种量子转移学习(QTL)体系结构来解决这一挑战，该体系结构将量子变分电路与基于ImageNet数据集预训练的经典机器学习网络相结合。通过对蚂蚁和蜜蜂、CIFAR-10和道路标志检测等各种图像数据集的系统模拟，我们证明了我们的QTL方法在不涉及转移学习的情况下比经典和量子机器学习具有更好的性能。此外，我们评估了QTL体系结构在对抗训练和未训练的情况下的对抗健壮性，证实了我们的QTL方法对数据操纵攻击具有对抗健壮性，并且优于经典方法。



## **21. Provably Robust Multi-bit Watermarking for AI-generated Text via Error Correction Code**

基于纠错码的人工智能文本可证明稳健多比特水印算法 cs.CR

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.16820v1) [paper-pdf](http://arxiv.org/pdf/2401.16820v1)

**Authors**: Wenjie Qu, Dong Yin, Zixin He, Wei Zou, Tianyang Tao, Jinyuan Jia, Jiaheng Zhang

**Abstract**: Large Language Models (LLMs) have been widely deployed for their remarkable capability to generate texts resembling human language. However, they could be misused by criminals to create deceptive content, such as fake news and phishing emails, which raises ethical concerns. Watermarking is a key technique to mitigate the misuse of LLMs, which embeds a watermark (e.g., a bit string) into a text generated by a LLM. Consequently, this enables the detection of texts generated by a LLM as well as the tracing of generated texts to a specific user. The major limitation of existing watermark techniques is that they cannot accurately or efficiently extract the watermark from a text, especially when the watermark is a long bit string. This key limitation impedes their deployment for real-world applications, e.g., tracing generated texts to a specific user.   This work introduces a novel watermarking method for LLM-generated text grounded in \textbf{error-correction codes} to address this challenge. We provide strong theoretical analysis, demonstrating that under bounded adversarial word/token edits (insertion, deletion, and substitution), our method can correctly extract watermarks, offering a provable robustness guarantee. This breakthrough is also evidenced by our extensive experimental results. The experiments show that our method substantially outperforms existing baselines in both accuracy and robustness on benchmark datasets. For instance, when embedding a bit string of length 12 into a 200-token generated text, our approach attains an impressive match rate of $98.4\%$, surpassing the performance of Yoo et al. (state-of-the-art baseline) at $85.6\%$. When subjected to a copy-paste attack involving the injection of 50 tokens to generated texts with 200 words, our method maintains a substantial match rate of $90.8\%$, while the match rate of Yoo et al. diminishes to below $65\%$.

摘要: 大型语言模型(LLM)因其生成类似人类语言的文本的非凡能力而被广泛使用。然而，它们可能被犯罪分子滥用来创造欺骗性内容，如假新闻和钓鱼电子邮件，这引发了伦理问题。水印是缓解LLMS误用的一项关键技术，它将水印(如比特串)嵌入到LLM生成的文本中。因此，这使得能够检测由LLM生成的文本以及将生成的文本跟踪到特定用户。现有水印技术的主要局限性是不能准确或高效地从文本中提取水印，特别是当水印是长比特串的时候。这一关键限制阻碍了它们在现实世界应用程序中的部署，例如，跟踪生成的文本到特定用户。为了解决这一问题，提出了一种新的基于文本纠错码的LLM文本水印方法。我们提供了强有力的理论分析，证明了在有界的敌意单词/令牌编辑(插入、删除和替换)下，我们的方法可以正确地提取水印，提供了可证明的健壮性保证。这一突破也被我们广泛的实验结果所证明。实验表明，在基准数据集上，我们的方法在准确率和稳健性方面都大大优于现有的基线。例如，当将长度为12的比特串嵌入到200个标记生成的文本中时，我们的方法获得了令人印象深刻的匹配率$98.4\$，超过了Yoo等人的性能。(最新基线)为85.6美元。在对200个单词的文本进行50个标记的复制粘贴攻击时，我们的方法保持了相当高的匹配率为90.8美元，而Yoo等人的匹配率是90.8美元。降至65美元以下。



## **22. Machine-learned Adversarial Attacks against Fault Prediction Systems in Smart Electrical Grids**

智能电网故障预测系统的机器学习对抗攻击 cs.CR

Accepted in AdvML@KDD'22

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2303.18136v2) [paper-pdf](http://arxiv.org/pdf/2303.18136v2)

**Authors**: Carmelo Ardito, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio, Fatemeh Nazary, Giovanni Servedio

**Abstract**: In smart electrical grids, fault detection tasks may have a high impact on society due to their economic and critical implications. In the recent years, numerous smart grid applications, such as defect detection and load forecasting, have embraced data-driven methodologies. The purpose of this study is to investigate the challenges associated with the security of machine learning (ML) applications in the smart grid scenario. Indeed, the robustness and security of these data-driven algorithms have not been extensively studied in relation to all power grid applications. We demonstrate first that the deep neural network method used in the smart grid is susceptible to adversarial perturbation. Then, we highlight how studies on fault localization and type classification illustrate the weaknesses of present ML algorithms in smart grids to various adversarial attacks

摘要: 在智能电网中，故障检测任务可能会对社会产生很大的影响，因为它们具有经济和关键意义。近年来，许多智能电网应用，如缺陷检测和负荷预测，都采用了数据驱动的方法。这项研究的目的是调查与智能电网场景中的机器学习(ML)应用程序的安全相关的挑战。事实上，这些数据驱动算法的健壮性和安全性并没有在所有的电网应用中得到广泛的研究。我们首先证明了智能电网中使用的深度神经网络方法容易受到对抗性扰动的影响。然后重点介绍了故障定位和类型分类的研究如何说明智能电网中现有最大似然算法在各种敌意攻击下的弱点



## **23. GE-AdvGAN: Improving the transferability of adversarial samples by gradient editing-based adversarial generative model**

GE-AdvGAN：基于梯度编辑的对抗性生成模型提高对抗性样本的可转移性 cs.CV

Accepted by SIAM International Conference on Data Mining (SDM24)

**SubmitDate**: 2024-01-30    [abs](http://arxiv.org/abs/2401.06031v2) [paper-pdf](http://arxiv.org/pdf/2401.06031v2)

**Authors**: Zhiyu Zhu, Huaming Chen, Xinyi Wang, Jiayu Zhang, Zhibo Jin, Kim-Kwang Raymond Choo, Jun Shen, Dong Yuan

**Abstract**: Adversarial generative models, such as Generative Adversarial Networks (GANs), are widely applied for generating various types of data, i.e., images, text, and audio. Accordingly, its promising performance has led to the GAN-based adversarial attack methods in the white-box and black-box attack scenarios. The importance of transferable black-box attacks lies in their ability to be effective across different models and settings, more closely aligning with real-world applications. However, it remains challenging to retain the performance in terms of transferable adversarial examples for such methods. Meanwhile, we observe that some enhanced gradient-based transferable adversarial attack algorithms require prolonged time for adversarial sample generation. Thus, in this work, we propose a novel algorithm named GE-AdvGAN to enhance the transferability of adversarial samples whilst improving the algorithm's efficiency. The main approach is via optimising the training process of the generator parameters. With the functional and characteristic similarity analysis, we introduce a novel gradient editing (GE) mechanism and verify its feasibility in generating transferable samples on various models. Moreover, by exploring the frequency domain information to determine the gradient editing direction, GE-AdvGAN can generate highly transferable adversarial samples while minimizing the execution time in comparison to the state-of-the-art transferable adversarial attack algorithms. The performance of GE-AdvGAN is comprehensively evaluated by large-scale experiments on different datasets, which results demonstrate the superiority of our algorithm. The code for our algorithm is available at: https://github.com/LMBTough/GE-advGAN

摘要: 诸如生成性对抗性网络(GANS)之类的对抗性生成模型被广泛应用于生成各种类型的数据，即图像、文本和音频。相应地，其良好的性能导致了白盒和黑盒攻击场景下基于GAN的对抗性攻击方法。可转移黑盒攻击的重要性在于它们能够在不同的模型和环境中有效，更紧密地与现实世界的应用程序保持一致。然而，就这类方法的可转让对抗性例子而言，保持业绩仍然具有挑战性。同时，我们观察到一些增强的基于梯度的可转移对抗性攻击算法需要较长的对抗性样本生成时间。因此，在这项工作中，我们提出了一种新的算法GE-AdvGAN，在提高算法效率的同时，增强了对抗性样本的可传递性。主要方法是通过优化发电机参数的训练过程。通过功能和特征的相似性分析，我们引入了一种新的梯度编辑机制，并验证了其在各种模型上生成可移植样本的可行性。此外，通过利用频域信息来确定梯度编辑方向，GE-AdvGAN可以生成高度可转移的对抗性样本，同时与最新的可转移对抗性攻击算法相比，可以最大限度地减少执行时间。通过在不同数据集上的大规模实验，对GE-AdvGAN算法的性能进行了综合评估，结果表明了该算法的优越性。我们算法的代码可以在https://github.com/LMBTough/GE-advGAN上找到



## **24. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

对抗性净化训练(TOOP)：提高健壮性和泛化能力 cs.CV

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16352v1) [paper-pdf](http://arxiv.org/pdf/2401.16352v1)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel framework called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。最成功的基于对抗性训练(AT)的防御技术可以达到对特定攻击的最佳健壮性，但不能很好地推广到看不见的攻击。另一种基于对抗性净化(AP)的有效防御技术可以增强泛化能力，但不能达到最优的健壮性。同时，这两种方法都有一个共同的缺陷，那就是标准精度下降。为了缓解这些问题，我们提出了一种新的框架，称为对抗性净化训练(TOOP)，该框架由两部分组成：通过随机变换的扰动破坏(RT)和通过对抗性损失微调(FT)的净化器模型。RT对于避免对已知攻击的过度学习导致对未知攻击的健壮性泛化至关重要，而FT对于提高健壮性是必不可少的。为了有效和可扩展地评估我们的方法，我们在CIFAR-10、CIFAR-100和ImageNette上进行了大量的实验，证明了我们的方法取得了最先进的结果，并表现出对不可见攻击的泛化能力。



## **25. Tradeoffs Between Alignment and Helpfulness in Language Models**

语言模型中对齐和有用之间的权衡 cs.CL

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16332v1) [paper-pdf](http://arxiv.org/pdf/2401.16332v1)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. Interestingly, we find that while the helpfulness generally decreases, it does so quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

摘要: 语言模型对齐已经成为人工智能安全的重要组成部分，通过增强期望的行为和抑制不期望的行为，允许人类和语言模型之间的安全交互。这通常通过调整模型或插入预设对齐提示来完成。最近，表征工程，一种通过在训练后改变模型表征来改变模型行为的方法，被证明在对齐LLM方面是有效的(Zou等人，2023a)。表征工程在对抗对抗性攻击和减少社会偏见等面向对齐的任务中产生收益，但也被证明导致模型执行基本任务的能力下降。在这篇文章中，我们研究了模型的一致性增加和有助性降低之间的权衡。我们提出了一个理论框架，提供了这两个量的界限，并从经验上证明了它们之间的相关性。有趣的是，我们发现，虽然总体上有益性降低，但它与表示工程向量的范数呈二次曲线关系，而比对则随其线性增加，这表明使用表示工程是有效的。我们通过实证验证了我们的发现，并绘制了表征工程对比对的有用性的界限。



## **26. Understanding Adversarial Robustness from Feature Maps of Convolutional Layers**

从卷积层特征图理解对手的稳健性 cs.CV

14pages

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2202.12435v2) [paper-pdf](http://arxiv.org/pdf/2202.12435v2)

**Authors**: Cong Xu, Wei Zhang, Jun Wang, Min Yang

**Abstract**: The adversarial robustness of a neural network mainly relies on two factors: model capacity and anti-perturbation ability. In this paper, we study the anti-perturbation ability of the network from the feature maps of convolutional layers. Our theoretical analysis discovers that larger convolutional feature maps before average pooling can contribute to better resistance to perturbations, but the conclusion is not true for max pooling. It brings new inspiration to the design of robust neural networks and urges us to apply these findings to improve existing architectures. The proposed modifications are very simple and only require upsampling the inputs or slightly modifying the stride configurations of downsampling operators. We verify our approaches on several benchmark neural network architectures, including AlexNet, VGG, RestNet18, and PreActResNet18. Non-trivial improvements in terms of both natural accuracy and adversarial robustness can be achieved under various attack and defense mechanisms. The code is available at \url{https://github.com/MTandHJ/rcm}.

摘要: 神经网络的对抗健壮性主要取决于两个因素：模型容量和抗扰动能力。本文从卷积层的特征映射出发，研究了该网络的抗扰动能力。我们的理论分析发现，在平均池化之前，较大的卷积特征映射有助于更好地抵抗扰动，但对于最大池化，这一结论不成立。这给健壮神经网络的设计带来了新的启发，并促使我们应用这些发现来改进现有的体系结构。所提出的修改非常简单，只需要对输入进行上采样或略微修改下采样算子的步长配置。我们在几个基准神经网络结构上验证了我们的方法，包括AlexNet、VGG、RestNet18和PreActResNet18。在各种攻击和防御机制下，在自然精确度和对手健壮性方面都可以实现不平凡的改进。代码可在\url{https://github.com/MTandHJ/rcm}.



## **27. LESSON: Multi-Label Adversarial False Data Injection Attack for Deep Learning Locational Detection**

经验：用于深度学习定位检测的多标签对抗性虚假数据注入攻击 cs.CR

Accepted by TDSC

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.16001v1) [paper-pdf](http://arxiv.org/pdf/2401.16001v1)

**Authors**: Jiwei Tian, Chao Shen, Buhong Wang, Xiaofang Xia, Meng Zhang, Chenhao Lin, Qian Li

**Abstract**: Deep learning methods can not only detect false data injection attacks (FDIA) but also locate attacks of FDIA. Although adversarial false data injection attacks (AFDIA) based on deep learning vulnerabilities have been studied in the field of single-label FDIA detection, the adversarial attack and defense against multi-label FDIA locational detection are still not involved. To bridge this gap, this paper first explores the multi-label adversarial example attacks against multi-label FDIA locational detectors and proposes a general multi-label adversarial attack framework, namely muLti-labEl adverSarial falSe data injectiON attack (LESSON). The proposed LESSON attack framework includes three key designs, namely Perturbing State Variables, Tailored Loss Function Design, and Change of Variables, which can help find suitable multi-label adversarial perturbations within the physical constraints to circumvent both Bad Data Detection (BDD) and Neural Attack Location (NAL). Four typical LESSON attacks based on the proposed framework and two dimensions of attack objectives are examined, and the experimental results demonstrate the effectiveness of the proposed attack framework, posing serious and pressing security concerns in smart grids.

摘要: 深度学习方法不仅可以检测虚假数据注入攻击(FDIA)，还可以定位FDIA的攻击。尽管基于深度学习漏洞的对抗虚假数据注入攻击(AFDIA)已经在单标签FDIA检测领域得到了研究，但针对多标签FDIA定位检测的对抗性攻击和防御仍未涉及。为了弥补这一差距，本文首先研究了针对多标签FDIA位置检测器的多标签对抗实例攻击，并提出了一个通用的多标签对抗攻击框架，即多标签对抗虚假数据注入攻击(Lesson)。Lesson攻击框架包括三个关键设计，即扰动状态变量、定制损失函数设计和变量变化，它们可以帮助在物理约束内找到合适的多标签对抗扰动，从而同时规避不良数据检测(BDD)和神经攻击定位(NAL)。对基于该框架和两个攻击目标维度的四种典型的Lesson攻击进行了测试，实验结果表明了该攻击框架的有效性，给智能电网带来了严重而紧迫的安全隐患。



## **28. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

一种利用安全性和活跃性增强性能的双层区块链分片协议 cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2310.11373v2) [paper-pdf](http://arxiv.org/pdf/2310.11373v2)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.

摘要: 分片对于提高区块链可伸缩性至关重要。现有的协议忽略了不同的对抗性攻击，限制了交易吞吐量。本文提出了一种突破性的分片协议Reetum，解决了这个问题，提高了区块链的可扩展性。RENETUM采用两阶段方法，根据运行时敌意攻击调整事务吞吐量。它包括两层的“控制”和“流程”分片。进程碎片包含至少一个可信节点，而控制碎片包含大多数可信节点。在第一阶段，事务被写入块，并由流程碎片中的节点投票表决。一致接受的障碍得到确认。在第二阶段，未获得一致接受的块由控制碎片投票表决。如果多数人投赞成票，就会接受阻止，从而消除第一阶段的反对者和沉默的选民。第一阶段使用一致投票，涉及的节点更少，支持更多的并行进程碎片。控制碎片最终确定决策并解决纠纷。实验证实了ReNetum的创新设计，提供了高交易吞吐量和对各种网络攻击的稳健性，性能优于现有的区块链网络分片协议。



## **29. Mitigation of Channel Tampering Attacks in Continuous-Variable Quantum Key Distribution**

连续变量量子密钥分配中信道篡改攻击的缓解 quant-ph

10 pages, 5 figures

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15898v1) [paper-pdf](http://arxiv.org/pdf/2401.15898v1)

**Authors**: Sebastian P. Kish, Chandra Thapa, Mikhael Sayat, Hajime Suzuki, Josef Pieprzyk, Seyit Camtepe

**Abstract**: Despite significant advancements in continuous-variable quantum key distribution (CV-QKD), practical CV-QKD systems can be compromised by various attacks. Consequently, identifying new attack vectors and countermeasures for CV-QKD implementations is important for the continued robustness of CV-QKD. In particular, as CV-QKD relies on a public quantum channel, vulnerability to communication disruption persists from potential adversaries employing Denial-of-Service (DoS) attacks. Inspired by DoS attacks, this paper introduces a novel threat in CV-QKD called the Channel Amplification (CA) attack, wherein Eve manipulates the communication channel through amplification. We specifically model this attack in a CV-QKD optical fiber setup. To counter this threat, we propose a detection and mitigation strategy. Detection involves a machine learning (ML) model based on a decision tree classifier, classifying various channel tampering attacks, including CA and DoS attacks. For mitigation, Bob, post-selects quadrature data by classifying the attack type and frequency. Our ML model exhibits high accuracy in distinguishing and categorizing these attacks. The CA attack's impact on the secret key rate (SKR) is explored concerning Eve's location and the relative intensity noise of the local oscillator (LO). The proposed mitigation strategy improves the attacked SKR for CA attacks and, in some cases, for hybrid CA-DoS attacks. Our study marks a novel application of both ML classification and post-selection in this context. These findings are important for enhancing the robustness of CV-QKD systems against emerging threats on the channel.

摘要: 尽管连续变量量子密钥分发（CV-QKD）取得了重大进展，但实际的CV-QKD系统可能会受到各种攻击。因此，为CV-QKD实现识别新的攻击向量和对策对于CV-QKD的持续鲁棒性非常重要。特别是，由于CV-QKD依赖于公共量子信道，因此潜在对手采用拒绝服务（DoS）攻击仍然存在通信中断的脆弱性。受拒绝服务攻击的启发，本文介绍了CV-QKD中的一种新威胁，称为信道放大（CA）攻击，其中Eve通过放大来操纵通信信道。我们专门在CV-QKD光纤设置中对这种攻击进行建模。为了应对这种威胁，我们提出了一个检测和缓解策略。检测涉及基于决策树分类器的机器学习（ML）模型，对各种信道篡改攻击（包括CA和DoS攻击）进行分类。为了缓解，Bob通过对攻击类型和频率进行分类来后选择正交数据。我们的ML模型在区分和分类这些攻击方面具有很高的准确性。CA攻击的秘密密钥速率（SKR）的影响进行了探讨有关夏娃的位置和本地振荡器（LO）的相对强度噪声。所提出的缓解策略提高了攻击的SKR CA攻击，在某些情况下，混合CA-DoS攻击。我们的研究标志着ML分类和后选择在这种情况下的新应用。这些发现对于增强CV-QKD系统对信道上出现的威胁的鲁棒性非常重要。



## **30. TransTroj: Transferable Backdoor Attacks to Pre-trained Models via Embedding Indistinguishability**

TransTroj：通过嵌入不可分辨将后门攻击转移到预先训练的模型 cs.CR

13 pages, 16 figures, 5 tables

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15883v1) [paper-pdf](http://arxiv.org/pdf/2401.15883v1)

**Authors**: Hao Wang, Tao Xiang, Shangwei Guo, Jialing He, Hangcheng Liu, Tianwei Zhang

**Abstract**: Pre-trained models (PTMs) are extensively utilized in various downstream tasks. Adopting untrusted PTMs may suffer from backdoor attacks, where the adversary can compromise the downstream models by injecting backdoors into the PTM. However, existing backdoor attacks to PTMs can only achieve partially task-agnostic and the embedded backdoors are easily erased during the fine-tuning process. In this paper, we propose a novel transferable backdoor attack, TransTroj, to simultaneously meet functionality-preserving, durable, and task-agnostic. In particular, we first formalize transferable backdoor attacks as the indistinguishability problem between poisoned and clean samples in the embedding space. We decompose the embedding indistinguishability into pre- and post-indistinguishability, representing the similarity of the poisoned and reference embeddings before and after the attack. Then, we propose a two-stage optimization that separately optimizes triggers and victim PTMs to achieve embedding indistinguishability. We evaluate TransTroj on four PTMs and six downstream tasks. Experimental results show that TransTroj significantly outperforms SOTA task-agnostic backdoor attacks (18%$\sim$99%, 68% on average) and exhibits superior performance under various system settings. The code is available at https://github.com/haowang-cqu/TransTroj .

摘要: 预训练模型(PTM)被广泛应用于各种下游任务。采用不可信的PTM可能会遭受后门攻击，攻击者可以通过向PTM注入后门来危害下游模型。然而，现有的对PTMS的后门攻击只能实现部分任务无关，并且嵌入的后门在微调过程中很容易被擦除。在本文中，我们提出了一种新的可转移后门攻击TransTroj，以同时满足功能保护、持久和任务无关的要求。特别地，我们首先将可转移的后门攻击形式化为嵌入空间中有毒样本和干净样本之间的不可区分问题。我们将嵌入不可区分性分解为攻击前后的不可区分性，表示攻击前后中毒嵌入和参考嵌入的相似性。然后，我们提出了一种两阶段优化方法，分别对触发者和受害者PTM进行优化，以达到嵌入不可区分的目的。我们在四个PTM和六个下游任务上对TransTroj进行了评估。实验结果表明，TransTroj的性能明显优于SOTA任务不可知的后门攻击(平均为18%、99%、68%)，并在不同的系统设置下表现出优异的性能。代码可在https://github.com/haowang-cqu/TransTroj上获得。



## **31. Adversarial Attacks and Defenses in 6G Network-Assisted IoT Systems**

6G网络辅助物联网系统中的对抗性攻击与防御 cs.IT

17 pages, 5 figures, and 4 tables. Submitted for publications

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.14780v2) [paper-pdf](http://arxiv.org/pdf/2401.14780v2)

**Authors**: Bui Duc Son, Nguyen Tien Hoa, Trinh Van Chien, Waqas Khalid, Mohamed Amine Ferrag, Wan Choi, Merouane Debbah

**Abstract**: The Internet of Things (IoT) and massive IoT systems are key to sixth-generation (6G) networks due to dense connectivity, ultra-reliability, low latency, and high throughput. Artificial intelligence, including deep learning and machine learning, offers solutions for optimizing and deploying cutting-edge technologies for future radio communications. However, these techniques are vulnerable to adversarial attacks, leading to degraded performance and erroneous predictions, outcomes unacceptable for ubiquitous networks. This survey extensively addresses adversarial attacks and defense methods in 6G network-assisted IoT systems. The theoretical background and up-to-date research on adversarial attacks and defenses are discussed. Furthermore, we provide Monte Carlo simulations to validate the effectiveness of adversarial attacks compared to jamming attacks. Additionally, we examine the vulnerability of 6G IoT systems by demonstrating attack strategies applicable to key technologies, including reconfigurable intelligent surfaces, massive multiple-input multiple-output (MIMO)/cell-free massive MIMO, satellites, the metaverse, and semantic communications. Finally, we outline the challenges and future developments associated with adversarial attacks and defenses in 6G IoT systems.

摘要: 物联网(IoT)和大规模物联网系统是第六代(6G)网络的关键，因为它们具有密集连接、超可靠、低延迟和高吞吐量。人工智能，包括深度学习和机器学习，为优化和部署未来无线电通信的尖端技术提供了解决方案。然而，这些技术容易受到敌意攻击，导致性能下降和错误预测，这是泛在网络无法接受的结果。这项调查广泛讨论了6G网络辅助物联网系统中的对抗性攻击和防御方法。讨论了对抗性攻防的理论背景和最新研究进展。此外，我们还提供了蒙特卡罗仿真来验证对抗攻击相对于干扰攻击的有效性。此外，我们通过演示适用于关键技术的攻击策略来检查6G IoT系统的脆弱性，这些关键技术包括可重构智能表面、大规模多输入多输出(MIMO)/无信元大规模MIMO、卫星、虚拟现实和语义通信。最后，我们概述了与6G物联网系统中的对抗性攻击和防御相关的挑战和未来发展。



## **32. Transparency Attacks: How Imperceptible Image Layers Can Fool AI Perception**

透明攻击：难以察觉的图像层如何愚弄AI感知 cs.CV

**SubmitDate**: 2024-01-29    [abs](http://arxiv.org/abs/2401.15817v1) [paper-pdf](http://arxiv.org/pdf/2401.15817v1)

**Authors**: Forrest McKee, David Noever

**Abstract**: This paper investigates a novel algorithmic vulnerability when imperceptible image layers confound multiple vision models into arbitrary label assignments and captions. We explore image preprocessing methods to introduce stealth transparency, which triggers AI misinterpretation of what the human eye perceives. The research compiles a broad attack surface to investigate the consequences ranging from traditional watermarking, steganography, and background-foreground miscues. We demonstrate dataset poisoning using the attack to mislabel a collection of grayscale landscapes and logos using either a single attack layer or randomly selected poisoning classes. For example, a military tank to the human eye is a mislabeled bridge to object classifiers based on convolutional networks (YOLO, etc.) and vision transformers (ViT, GPT-Vision, etc.). A notable attack limitation stems from its dependency on the background (hidden) layer in grayscale as a rough match to the transparent foreground image that the human eye perceives. This dependency limits the practical success rate without manual tuning and exposes the hidden layers when placed on the opposite display theme (e.g., light background, light transparent foreground visible, works best against a light theme image viewer or browser). The stealth transparency confounds established vision systems, including evading facial recognition and surveillance, digital watermarking, content filtering, dataset curating, automotive and drone autonomy, forensic evidence tampering, and retail product misclassifying. This method stands in contrast to traditional adversarial attacks that typically focus on modifying pixel values in ways that are either slightly perceptible or entirely imperceptible for both humans and machines.

摘要: 本文研究了一种新的算法漏洞时，不可感知的图像层混淆多个视觉模型到任意的标签分配和字幕。我们探索图像预处理方法来引入隐形透明度，这会引发AI对人眼感知的误解。该研究编制了一个广泛的攻击面调查的后果，从传统的水印，隐写术，和背景前景的失误。我们演示了数据集中毒使用攻击错误标记的灰度景观和徽标的集合使用单个攻击层或随机选择的中毒类。例如，对于人眼来说，军用坦克是基于卷积网络（YOLO等）的对象分类器的错误标记桥梁。和视觉转换器（ViT，GPT-Vision等）。一个值得注意的攻击限制源于它对灰度背景（隐藏）层的依赖性，作为人眼感知的透明前景图像的粗略匹配。这种依赖性限制了在没有手动调整的情况下的实际成功率，并且当被放置在相反的显示主题上时暴露隐藏层（例如，浅色背景，浅色透明前景可见，最适合浅色主题图像查看器或浏览器）。隐形透明混淆了现有的视觉系统，包括逃避面部识别和监视、数字水印、内容过滤、数据集管理、汽车和无人机自主性、法医证据篡改以及零售产品错误分类。这种方法与传统的对抗性攻击形成鲜明对比，传统的对抗性攻击通常专注于以人类和机器略微感知或完全感知不到的方式修改像素值。



## **33. Improving Transformation-based Defenses against Adversarial Examples with First-order Perturbations**

改进基于变换的一阶摄动对抗性实例防御 cs.CV

This paper has technical errors

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2103.04565v3) [paper-pdf](http://arxiv.org/pdf/2103.04565v3)

**Authors**: Haimin Zhang, Min Xu

**Abstract**: Deep neural networks have been successfully applied in various machine learning tasks. However, studies show that neural networks are susceptible to adversarial attacks. This exposes a potential threat to neural network-based intelligent systems. We observe that the probability of the correct result outputted by the neural network increases by applying small first-order perturbations generated for non-predicted class labels to adversarial examples. Based on this observation, we propose a method for counteracting adversarial perturbations to improve adversarial robustness. In the proposed method, we randomly select a number of class labels and generate small first-order perturbations for these selected labels. The generated perturbations are added together and then clamped onto a specified space. The obtained perturbation is finally added to the adversarial example to counteract the adversarial perturbation contained in the example. The proposed method is applied at inference time and does not require retraining or finetuning the model. We experimentally validate the proposed method on CIFAR-10 and CIFAR-100. The results demonstrate that our method effectively improves the defense performance of several transformation-based defense methods, especially against strong adversarial examples generated using more iterations.

摘要: 深度神经网络已成功地应用于各种机器学习任务。然而，研究表明，神经网络容易受到对抗性攻击。这给基于神经网络的智能系统带来了潜在的威胁。我们观察到，通过将为非预测类标签产生的一阶小扰动应用于对抗性例子，神经网络输出正确结果的概率增加。基于这一观察结果，我们提出了一种对抗对抗性扰动的方法来提高对抗性稳健性。在所提出的方法中，我们随机选择多个类别标签，并对这些选定的标签产生小的一阶扰动。产生的扰动被加在一起，然后被夹在指定的空间上。最后将所获得的扰动添加到对抗性示例，以抵消该示例中包含的对抗性扰动。所提出的方法是在推理时应用的，不需要对模型进行重新训练或微调。我们在CIFAR-10和CIFAR-100上进行了实验验证。实验结果表明，该方法有效地提高了几种基于变换的防御方法的防御性能，特别是对迭代次数较多的强敌意实例的防御效果更佳。



## **34. Adversarial Attacks on Graph Neural Networks via Meta Learning**

基于元学习的图神经网络敌意攻击 cs.LG

ICLR submission

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/1902.08412v2) [paper-pdf](http://arxiv.org/pdf/1902.08412v2)

**Authors**: Daniel Zügner, Stephan Günnemann

**Abstract**: Deep learning models for graphs have advanced the state of the art on many tasks. Despite their recent success, little is known about their robustness. We investigate training time attacks on graph neural networks for node classification that perturb the discrete graph structure. Our core principle is to use meta-gradients to solve the bilevel problem underlying training-time attacks, essentially treating the graph as a hyperparameter to optimize. Our experiments show that small graph perturbations consistently lead to a strong decrease in performance for graph convolutional networks, and even transfer to unsupervised embeddings. Remarkably, the perturbations created by our algorithm can misguide the graph neural networks such that they perform worse than a simple baseline that ignores all relational information. Our attacks do not assume any knowledge about or access to the target classifiers.

摘要: 图的深度学习模型提高了许多任务的技术水平。尽管它们最近取得了成功，但人们对它们的健壮性知之甚少。研究了扰动离散图结构的用于节点分类的图神经网络的训练时间攻击。我们的核心原则是使用亚梯度来解决训练时间攻击背后的双层问题，本质上将图作为一个超参数进行优化。我们的实验表明，小的图扰动一直导致图卷积网络的性能显著下降，甚至转移到无监督嵌入。值得注意的是，我们的算法产生的扰动可能会误导图神经网络，导致它们的性能比忽略所有关系信息的简单基线更差。我们的攻击不假设任何关于目标分类器的知识或访问权限。



## **35. Passive Inference Attacks on Split Learning via Adversarial Regularization**

基于对抗性正则化的分裂学习被动推理攻击 cs.CR

19 pages, 20 figures

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2310.10483v4) [paper-pdf](http://arxiv.org/pdf/2310.10483v4)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.

摘要: 分裂学习(Split Learning，SL)已成为传统联合学习的一种实用有效的替代方案。虽然以前攻击SL的尝试通常依赖于过于强烈的假设或目标明确、易于利用的模型，但我们寻求开发更实际的攻击。我们介绍了SDAR，这是一种针对SL的新型攻击框架，具有诚实但好奇的服务器。SDAR利用辅助数据和对抗性正则化学习客户私有模型的可解码模拟器，该模拟器可以有效地推断客户在香草SL下的私有特征，以及U形SL下的特征和标签。我们在两种配置下都进行了大量的实验，以验证我们提出的攻击的有效性。值得注意的是，在具有挑战性但实用的场景中，现有的被动攻击难以有效地重建客户端的私有数据，SDAR始终实现与主动攻击相当的攻击性能。在CIFAR-10上，在7的深度分裂水平上，SDAR实现了私有特征重建，在普通SL和U形SL上的均方误差都小于0.025，在U形背景下获得了98%以上的标签推理准确率，而现有的攻击无法产生非平凡的结果。



## **36. Masked Language Model Based Textual Adversarial Example Detection**

基于掩蔽语言模型的文本对抗性实例检测 cs.CR

13 pages,3 figures

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2304.08767v3) [paper-pdf](http://arxiv.org/pdf/2304.08767v3)

**Authors**: Xiaomei Zhang, Zhaoxi Zhang, Qi Zhong, Xufei Zheng, Yanjun Zhang, Shengshan Hu, Leo Yu Zhang

**Abstract**: Adversarial attacks are a serious threat to the reliable deployment of machine learning models in safety-critical applications. They can misguide current models to predict incorrectly by slightly modifying the inputs. Recently, substantial work has shown that adversarial examples tend to deviate from the underlying data manifold of normal examples, whereas pre-trained masked language models can fit the manifold of normal NLP data. To explore how to use the masked language model in adversarial detection, we propose a novel textual adversarial example detection method, namely Masked Language Model-based Detection (MLMD), which can produce clearly distinguishable signals between normal examples and adversarial examples by exploring the changes in manifolds induced by the masked language model. MLMD features a plug and play usage (i.e., no need to retrain the victim model) for adversarial defense and it is agnostic to classification tasks, victim model's architectures, and to-be-defended attack methods. We evaluate MLMD on various benchmark textual datasets, widely studied machine learning models, and state-of-the-art (SOTA) adversarial attacks (in total $3*4*4 = 48$ settings). Experimental results show that MLMD can achieve strong performance, with detection accuracy up to 0.984, 0.967, and 0.901 on AG-NEWS, IMDB, and SST-2 datasets, respectively. Additionally, MLMD is superior, or at least comparable to, the SOTA detection defenses in detection accuracy and F1 score. Among many defenses based on the off-manifold assumption of adversarial examples, this work offers a new angle for capturing the manifold change. The code for this work is openly accessible at \url{https://github.com/mlmddetection/MLMDdetection}.

摘要: 对抗性攻击对机器学习模型在安全关键型应用中的可靠部署构成严重威胁。他们可以通过稍微修改输入来误导当前的模型进行错误的预测。最近的大量工作表明，对抗性例子往往偏离正常例子的底层数据流形，而预先训练的掩蔽语言模型可以适应正常NLP数据的流形。为了探索掩蔽语言模型在对抗性检测中的应用，我们提出了一种新的文本对抗性实例检测方法，即基于掩蔽语言模型的检测方法(MLMD)，该方法通过研究掩蔽语言模型引起的流形变化来产生能够清晰区分正常例子和对抗性例子的信号。MLMD具有即插即用的特点(即，不需要重新训练受害者模型)用于对抗防御，并且它与分类任务、受害者模型的体系结构和要防御的攻击方法无关。我们在各种基准文本数据集、广泛研究的机器学习模型和最先进的(SOTA)对手攻击(总计$3*4*4=48$设置)上评估MLMD。实验结果表明，该算法在AG-NEWS、IMDB和SST-2数据集上的检测准确率分别达到0.984、0.967和0.901。此外，MLMD在检测精度和F1得分方面优于SOTA检测防御，或至少与SOTA检测防御相当。在许多基于对抗性例子的非流形假设的防御中，这项工作为捕捉流形变化提供了一个新的角度。这项工作的代码可以在\url{https://github.com/mlmddetection/MLMDdetection}.上公开访问



## **37. Addressing Noise and Efficiency Issues in Graph-Based Machine Learning Models From the Perspective of Adversarial Attack**

从对抗攻击的角度解决基于图的机器学习模型中的噪声和效率问题 cs.LG

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2401.15615v1) [paper-pdf](http://arxiv.org/pdf/2401.15615v1)

**Authors**: Yongyu Wang

**Abstract**: Given that no existing graph construction method can generate a perfect graph for a given dataset, graph-based algorithms are invariably affected by the plethora of redundant and erroneous edges present within the constructed graphs. In this paper, we propose treating these noisy edges as adversarial attack and use a spectral adversarial robustness evaluation method to diminish the impact of noisy edges on the performance of graph algorithms. Our method identifies those points that are less vulnerable to noisy edges and leverages only these robust points to perform graph-based algorithms. Our experiments with spectral clustering, one of the most representative and widely utilized graph algorithms, reveal that our methodology not only substantially elevates the precision of the algorithm but also greatly accelerates its computational efficiency by leveraging only a select number of robust data points.

摘要: 由于没有一种现有的图构造方法能够为给定的数据集生成完美的图，基于图的算法总是受到所构造图中存在的过多的冗余和错误边的影响。在本文中，我们建议将这些噪声边视为对抗性攻击，并使用谱对抗健壮性评估方法来减小噪声边对图算法性能的影响。我们的方法识别那些不太容易受到噪声边缘影响的点，并只利用这些健壮点来执行基于图的算法。我们对最具代表性和使用最广泛的图算法之一的谱聚类进行的实验表明，我们的方法不仅大大提高了算法的精度，而且通过只利用选定数量的稳健数据点来极大地提高其计算效率。



## **38. Generalizing Speaker Verification for Spoof Awareness in the Embedding Space**

嵌入空间中基于欺骗感知的说话人确认泛化 cs.CR

Published in IEEE/ACM Transactions on Audio, Speech, and Language  Processing (doi updated)

**SubmitDate**: 2024-01-28    [abs](http://arxiv.org/abs/2401.11156v2) [paper-pdf](http://arxiv.org/pdf/2401.11156v2)

**Authors**: Xuechen Liu, Md Sahidullah, Kong Aik Lee, Tomi Kinnunen

**Abstract**: It is now well-known that automatic speaker verification (ASV) systems can be spoofed using various types of adversaries. The usual approach to counteract ASV systems against such attacks is to develop a separate spoofing countermeasure (CM) module to classify speech input either as a bonafide, or a spoofed utterance. Nevertheless, such a design requires additional computation and utilization efforts at the authentication stage. An alternative strategy involves a single monolithic ASV system designed to handle both zero-effort imposter (non-targets) and spoofing attacks. Such spoof-aware ASV systems have the potential to provide stronger protections and more economic computations. To this end, we propose to generalize the standalone ASV (G-SASV) against spoofing attacks, where we leverage limited training data from CM to enhance a simple backend in the embedding space, without the involvement of a separate CM module during the test (authentication) phase. We propose a novel yet simple backend classifier based on deep neural networks and conduct the study via domain adaptation and multi-task integration of spoof embeddings at the training stage. Experiments are conducted on the ASVspoof 2019 logical access dataset, where we improve the performance of statistical ASV backends on the joint (bonafide and spoofed) and spoofed conditions by a maximum of 36.2% and 49.8% in terms of equal error rates, respectively.

摘要: 现在众所周知，自动说话人验证(ASV)系统可以使用各种类型的对手进行欺骗。对抗ASV系统抵御此类攻击的通常方法是开发单独的欺骗对策(CM)模块，以将语音输入分类为真正的或欺骗的话语。然而，这样的设计在身份验证阶段需要额外的计算和利用工作。另一种策略包括一个单一的单片ASV系统，旨在同时处理零努力冒名顶替者(非目标)和欺骗攻击。这种感知欺骗的ASV系统有可能提供更强大的保护和更多的经济计算。为此，我们建议推广抗欺骗攻击的独立ASV(G-SASV)，其中我们利用来自CM的有限训练数据来增强嵌入空间中的简单后端，而不需要在测试(身份验证)阶段涉及单独的CM模块。我们提出了一种新颖而简单的基于深度神经网络的后端分类器，并在训练阶段通过域自适应和欺骗嵌入的多任务集成进行了研究。在ASVspoof 2019逻辑访问数据集上进行了实验，在相同错误率的情况下，我们将联合(真实和欺骗)和欺骗条件下的统计ASV后端的性能分别提高了36.2%和49.8%。



## **39. Style-News: Incorporating Stylized News Generation and Adversarial Verification for Neural Fake News Detection**

风格新闻：结合风格化新闻生成和对抗性验证的神经网络假新闻检测 cs.CL

EACL 2024 Main Track

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15509v1) [paper-pdf](http://arxiv.org/pdf/2401.15509v1)

**Authors**: Wei-Yao Wang, Yu-Chieh Chang, Wen-Chih Peng

**Abstract**: With the improvements in generative models, the issues of producing hallucinations in various domains (e.g., law, writing) have been brought to people's attention due to concerns about misinformation. In this paper, we focus on neural fake news, which refers to content generated by neural networks aiming to mimic the style of real news to deceive people. To prevent harmful disinformation spreading fallaciously from malicious social media (e.g., content farms), we propose a novel verification framework, Style-News, using publisher metadata to imply a publisher's template with the corresponding text types, political stance, and credibility. Based on threat modeling aspects, a style-aware neural news generator is introduced as an adversary for generating news content conditioning for a specific publisher, and style and source discriminators are trained to defend against this attack by identifying which publisher the style corresponds with, and discriminating whether the source of the given news is human-written or machine-generated. To evaluate the quality of the generated content, we integrate various dimensional metrics (language fluency, content preservation, and style adherence) and demonstrate that Style-News significantly outperforms the previous approaches by a margin of 0.35 for fluency, 15.24 for content, and 0.38 for style at most. Moreover, our discriminative model outperforms state-of-the-art baselines in terms of publisher prediction (up to 4.64%) and neural fake news detection (+6.94% $\sim$ 31.72%).

摘要: 随着生成模型的改进，由于对错误信息的担忧，在不同领域(如法律、写作)产生幻觉的问题引起了人们的注意。在本文中，我们关注的是神经假新闻，它是指通过神经网络生成的内容，目的是模仿真实新闻的风格来欺骗人们。为了防止有害的虚假信息从恶意的社交媒体(如内容农场)错误地传播，我们提出了一个新的验证框架Style-News，它使用发布者的元数据来暗示发布者的模板，以及相应的文本类型、政治立场和可信度。基于威胁建模方面，引入样式感知神经新闻生成器作为为特定发布者生成新闻内容条件的对手，并训练样式和来源鉴别器以通过识别样式对应于哪个发布者来防御这种攻击，并区分给定新闻的来源是人写的还是机器生成的。为了评估生成的内容的质量，我们综合了各种维度的度量(语言流畅度、内容保持和风格坚持)，并表明Style-News在流畅度方面显著优于以前的方法，流畅度为0.35，内容为15.24，风格至多为0.38。此外，我们的判别模型在出版商预测(高达4.64%)和神经假新闻检测(+6.94%$\sim$31.72%)方面优于最先进的基线。



## **40. No-Box Attacks on 3D Point Cloud Classification**

三维点云分类中的No-Box攻击 cs.CV

10 pages, 6 figures

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2210.14164v3) [paper-pdf](http://arxiv.org/pdf/2210.14164v3)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the access to the DNN model itself to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, where adversarial points can be predicted without access to the target DNN model, which is referred to as a ``no-box'' attack. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for adversarial point prediction, and which combination of features is best suited for this purpose. Experiments show that a suitable combination of features is able to predict adversarial points of four different networks -- PointNet, PointNet++, DGCNN, and PointConv -- significantly better than a random guess and comparable to white-box attacks. Additionally, we show that no-box attack is transferable to unseen models. The results also provide further insight into DNNs for point cloud classification, by showing which features play key roles in their decision-making process.

摘要: 对抗性攻击对基于深度神经网络(DNN)的各种输入信号分析提出了严峻的挑战。在3D点云的情况下，已经开发出方法来识别在网络决策中起关键作用的点，并且这些点在产生现有的对抗性攻击时变得至关重要。例如，显著图方法是一种流行的识别对抗性丢弃点的方法，其移除将显著影响网络决策。通常，识别敌对点的方法依赖于对DNN模型本身的访问来确定哪些点对模型的决策至关重要。本文旨在为这一问题提供一种新的观点，即可以在不访问目标DNN模型的情况下预测敌对点，这被称为“无盒”攻击。为此，我们定义了14个点云特征，并使用多元线性回归来检验这些特征是否可以用于对抗点预测，以及哪种特征组合最适合于此目的。实验表明，适当的特征组合能够预测四种不同网络--PointNet、PointNet++、DGCNN和PointConv的敌对点--显著优于随机猜测，与白盒攻击相当。此外，我们还证明了无盒攻击可以转移到不可见模型上。通过显示哪些特征在其决策过程中起关键作用，该结果还为点云分类的DNN提供了进一步的洞察。



## **41. L-AutoDA: Leveraging Large Language Models for Automated Decision-based Adversarial Attacks**

L-AutoDA：利用大型语言模型进行基于决策的自动对抗性攻击 cs.CR

Under Review of IJCNN 2024

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15335v1) [paper-pdf](http://arxiv.org/pdf/2401.15335v1)

**Authors**: Ping Guo, Fei Liu, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: In the rapidly evolving field of machine learning, adversarial attacks present a significant challenge to model robustness and security. Decision-based attacks, which only require feedback on the decision of a model rather than detailed probabilities or scores, are particularly insidious and difficult to defend against. This work introduces L-AutoDA (Large Language Model-based Automated Decision-based Adversarial Attacks), a novel approach leveraging the generative capabilities of Large Language Models (LLMs) to automate the design of these attacks. By iteratively interacting with LLMs in an evolutionary framework, L-AutoDA automatically designs competitive attack algorithms efficiently without much human effort. We demonstrate the efficacy of L-AutoDA on CIFAR-10 dataset, showing significant improvements over baseline methods in both success rate and computational efficiency. Our findings underscore the potential of language models as tools for adversarial attack generation and highlight new avenues for the development of robust AI systems.

摘要: 在快速发展的机器学习领域，敌意攻击对模型的健壮性和安全性提出了重大挑战。基于决策的攻击，只需要对模型的决策进行反馈，而不需要详细的概率或分数，特别隐蔽，很难防御。本文介绍了L-AUTODA(基于大语言模型的自动决策对抗性攻击)，它是一种利用大语言模型的生成能力来自动化攻击设计的新方法。通过在进化框架中迭代地与LLM交互，L自动DA无需太多人力即可自动高效地设计竞争攻击算法。我们在CIFAR-10数据集上验证了L-AutoDA的有效性，在成功率和计算效率上都比基线方法有了显著的提高。我们的发现强调了语言模型作为对抗性攻击生成工具的潜力，并强调了开发健壮的人工智能系统的新途径。



## **42. Multi-Trigger Backdoor Attacks: More Triggers, More Threats**

多触发后门攻击：更多触发因素，更多威胁 cs.LG

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15295v1) [paper-pdf](http://arxiv.org/pdf/2401.15295v1)

**Authors**: Yige Li, Xingjun Ma, Jiabo He, Hanxun Huang, Yu-Gang Jiang

**Abstract**: Backdoor attacks have emerged as a primary threat to (pre-)training and deployment of deep neural networks (DNNs). While backdoor attacks have been extensively studied in a body of works, most of them were focused on single-trigger attacks that poison a dataset using a single type of trigger. Arguably, real-world backdoor attacks can be much more complex, e.g., the existence of multiple adversaries for the same dataset if it is of high value. In this work, we investigate the practical threat of backdoor attacks under the setting of \textbf{multi-trigger attacks} where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks, including parallel, sequential, and hybrid attacks, we provide a set of important understandings of the coexisting, overwriting, and cross-activating effects between different triggers on the same dataset. Moreover, we show that single-trigger attacks tend to cause overly optimistic views of the security of current defense techniques, as all examined defense methods struggle to defend against multi-trigger attacks. Finally, we create a multi-trigger backdoor poisoning dataset to help future evaluation of backdoor attacks and defenses. Although our work is purely empirical, we hope it can help steer backdoor research toward more realistic settings.

摘要: 后门攻击已成为深度神经网络(DNN)(预)训练和部署的主要威胁。虽然后门攻击已经在一系列著作中得到了广泛的研究，但大多数都集中在使用单一类型的触发器毒化数据集的单触发器攻击上。可以说，真实世界的后门攻击可能要复杂得多，例如，如果同一数据集具有很高的价值，则存在多个对手。在这项工作中，我们研究了在多个对手利用不同类型的触发器来毒化同一数据集的情况下，后门攻击的实际威胁。通过提出和研究三种类型的多触发攻击，包括并行攻击、顺序攻击和混合攻击，我们对同一数据集上不同触发之间的共存、覆盖和交叉激活效应提供了一套重要的理解。此外，我们发现，单触发攻击往往会导致对当前防御技术的安全性过于乐观的看法，因为所有被检查的防御方法都难以防御多触发攻击。最后，我们创建了一个多触发后门中毒数据集，以帮助未来评估后门攻击和防御。尽管我们的工作纯粹是经验性的，但我们希望它能帮助将后门研究引向更现实的环境。



## **43. Unraveling Attacks in Machine Learning-based IoT Ecosystems: A Survey and the Open Libraries Behind Them**

破解基于机器学习的物联网生态系统中的攻击：综述及其背后的开放图书馆 cs.CR

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.11723v2) [paper-pdf](http://arxiv.org/pdf/2401.11723v2)

**Authors**: Chao Liu, Boxi Chen, Wei Shao, Chris Zhang, Kelvin Wong, Yi Zhang

**Abstract**: The advent of the Internet of Things (IoT) has brought forth an era of unprecedented connectivity, with an estimated 80 billion smart devices expected to be in operation by the end of 2025. These devices facilitate a multitude of smart applications, enhancing the quality of life and efficiency across various domains. Machine Learning (ML) serves as a crucial technology, not only for analyzing IoT-generated data but also for diverse applications within the IoT ecosystem. For instance, ML finds utility in IoT device recognition, anomaly detection, and even in uncovering malicious activities. This paper embarks on a comprehensive exploration of the security threats arising from ML's integration into various facets of IoT, spanning various attack types including membership inference, adversarial evasion, reconstruction, property inference, model extraction, and poisoning attacks. Unlike previous studies, our work offers a holistic perspective, categorizing threats based on criteria such as adversary models, attack targets, and key security attributes (confidentiality, availability, and integrity). We delve into the underlying techniques of ML attacks in IoT environment, providing a critical evaluation of their mechanisms and impacts. Furthermore, our research thoroughly assesses 65 libraries, both author-contributed and third-party, evaluating their role in safeguarding model and data privacy. We emphasize the availability and usability of these libraries, aiming to arm the community with the necessary tools to bolster their defenses against the evolving threat landscape. Through our comprehensive review and analysis, this paper seeks to contribute to the ongoing discourse on ML-based IoT security, offering valuable insights and practical solutions to secure ML models and data in the rapidly expanding field of artificial intelligence in IoT.

摘要: 物联网(IoT)的到来带来了一个前所未有的互联时代，预计到2025年底，将有800亿台智能设备投入运营。这些设备促进了大量智能应用，提高了各个领域的生活质量和效率。机器学习(ML)是一项关键技术，不仅用于分析物联网生成的数据，还用于分析物联网生态系统中的各种应用。例如，ML在物联网设备识别、异常检测，甚至在发现恶意活动方面都有用武之地。本文对ML融入物联网的各个方面所带来的安全威胁进行了全面的探讨，包括成员身份推断、对抗性逃避、重构、属性推理、模型提取和中毒攻击等各种攻击类型。与以前的研究不同，我们的工作提供了一个整体的视角，根据对手模型、攻击目标和关键安全属性(机密性、可用性和完整性)等标准对威胁进行分类。我们深入研究了物联网环境下ML攻击的基本技术，并对其机制和影响进行了关键评估。此外，我们的研究全面评估了65个图书馆，包括作者贡献的图书馆和第三方图书馆，评估它们在保护模型和数据隐私方面的作用。我们强调这些库的可用性和可用性，旨在为社区提供必要的工具，以加强他们对不断变化的威胁环境的防御。通过我们的全面回顾和分析，本文试图为正在进行的基于ML的物联网安全讨论做出贡献，为保护物联网快速扩张的人工智能领域中的ML模型和数据提供有价值的见解和实用解决方案。



## **44. Asymptotic Behavior of Adversarial Training Estimator under $\ell_\infty$-Perturbation**

对抗性训练估计量在摄动下的渐近行为 math.ST

**SubmitDate**: 2024-01-27    [abs](http://arxiv.org/abs/2401.15262v1) [paper-pdf](http://arxiv.org/pdf/2401.15262v1)

**Authors**: Yiling Xie, Xiaoming Huo

**Abstract**: Adversarial training has been proposed to hedge against adversarial attacks in machine learning and statistical models. This paper focuses on adversarial training under $\ell_\infty$-perturbation, which has recently attracted much research attention. The asymptotic behavior of the adversarial training estimator is investigated in the generalized linear model. The results imply that the limiting distribution of the adversarial training estimator under $\ell_\infty$-perturbation could put a positive probability mass at $0$ when the true parameter is $0$, providing a theoretical guarantee of the associated sparsity-recovery ability. Alternatively, a two-step procedure is proposed -- adaptive adversarial training, which could further improve the performance of adversarial training under $\ell_\infty$-perturbation. Specifically, the proposed procedure could achieve asymptotic unbiasedness and variable-selection consistency. Numerical experiments are conducted to show the sparsity-recovery ability of adversarial training under $\ell_\infty$-perturbation and to compare the empirical performance between classic adversarial training and adaptive adversarial training.

摘要: 对抗性训练已经被提议用来对冲机器学习和统计模型中的对抗性攻击。本文主要研究近年来备受关注的对抗性训练问题。研究了广义线性模型中对抗性训练估计器的渐近行为。结果表明，当真参数为$0$时，当真参数为$0$时，对抗性训练估计量的极限分布可以使正概率质量为$0$，从而为相应的稀疏恢复能力提供了理论保证。或者，提出了一种分两步进行的训练方法--自适应对抗性训练，它可以进一步提高对抗性训练在干扰下的性能。具体地说，该方法可以实现渐近无偏和变量选择的一致性。通过数值实验验证了对抗性训练在扰动下的稀疏性恢复能力，并比较了经典对抗性训练和自适应对抗性训练的经验性能。



## **45. Better Representations via Adversarial Training in Pre-Training: A Theoretical Perspective**

前训练中通过对抗性训练获得更好的表征：一个理论视角 cs.LG

To appear in AISTATS2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.15248v1) [paper-pdf](http://arxiv.org/pdf/2401.15248v1)

**Authors**: Yue Xing, Xiaofeng Lin, Qifan Song, Yi Xu, Belinda Zeng, Guang Cheng

**Abstract**: Pre-training is known to generate universal representations for downstream tasks in large-scale deep learning such as large language models. Existing literature, e.g., \cite{kim2020adversarial}, empirically observe that the downstream tasks can inherit the adversarial robustness of the pre-trained model. We provide theoretical justifications for this robustness inheritance phenomenon. Our theoretical results reveal that feature purification plays an important role in connecting the adversarial robustness of the pre-trained model and the downstream tasks in two-layer neural networks. Specifically, we show that (i) with adversarial training, each hidden node tends to pick only one (or a few) feature; (ii) without adversarial training, the hidden nodes can be vulnerable to attacks. This observation is valid for both supervised pre-training and contrastive learning. With purified nodes, it turns out that clean training is enough to achieve adversarial robustness in downstream tasks.

摘要: 众所周知，预训练可以为大规模深度学习(如大型语言模型)中的下游任务生成通用表示。现有文献，如{kim2020对抗)，从经验上观察到下游任务可以继承预训练模型的对抗健壮性。我们为这种健壮性继承现象提供了理论依据。我们的理论结果表明，在两层神经网络中，特征提纯在连接预训练模型的对抗健壮性和下游任务方面起着重要作用。具体地说，我们证明了(I)在对抗性训练下，每个隐藏节点往往只选择一个(或几个)特征；(Ii)在没有对抗性训练的情况下，隐藏节点可能容易受到攻击。这一观察结果对有监督的预训练和对比学习都是有效的。事实证明，在净化节点的情况下，干净的训练足以在下游任务中实现对抗健壮性。



## **46. End-To-End Set-Based Training for Neural Network Verification**

用于神经网络验证的端到端集合训练 cs.LG

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14961v1) [paper-pdf](http://arxiv.org/pdf/2401.14961v1)

**Authors**: Lukas Koller, Tobias Ladner, Matthias Althoff

**Abstract**: Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can result in substantially different outputs of a neural network. Safety-critical environments require neural networks that are robust against input perturbations. However, training and formally verifying robust neural networks is challenging. We address this challenge by employing, for the first time, a end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure drastically simplifies the subsequent formal robustness verification of the trained neural network. While previous research has predominantly focused on augmenting neural network training with adversarial attacks, our approach leverages set-based computing to train neural networks with entire sets of perturbed inputs. Moreover, we demonstrate that our set-based training procedure effectively trains robust neural networks, which are easier to verify. In many cases, set-based trained neural networks outperform neural networks trained with state-of-the-art adversarial attacks.

摘要: 神经网络容易受到对抗性攻击，即，微小的输入扰动可能导致神经网络的输出本质上不同。安全关键型环境要求神经网络对输入扰动具有健壮性。然而，训练和正式验证稳健的神经网络是具有挑战性的。我们通过首次采用端到端基于集合的训练过程来解决这一挑战，该过程训练用于正式验证的稳健神经网络。我们的训练过程极大地简化了随后训练的神经网络的形式稳健性验证。虽然以前的研究主要集中在通过对抗性攻击来增强神经网络的训练，但我们的方法利用基于集合的计算来训练具有整个扰动输入集的神经网络。此外，我们证明了我们的基于集合的训练过程有效地训练了健壮的神经网络，更容易验证。在许多情况下，基于集合的训练神经网络的性能优于使用最先进的对抗性攻击训练的神经网络。



## **47. Conserve-Update-Revise to Cure Generalization and Robustness Trade-off in Adversarial Training**

保存-更新-修订以解决对抗性训练中泛化和健壮性的权衡 cs.LG

Accepted as a conference paper at ICLR 2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14948v1) [paper-pdf](http://arxiv.org/pdf/2401.14948v1)

**Authors**: Shruthi Gowda, Bahram Zonooz, Elahe Arani

**Abstract**: Adversarial training improves the robustness of neural networks against adversarial attacks, albeit at the expense of the trade-off between standard and robust generalization. To unveil the underlying factors driving this phenomenon, we examine the layer-wise learning capabilities of neural networks during the transition from a standard to an adversarial setting. Our empirical findings demonstrate that selectively updating specific layers while preserving others can substantially enhance the network's learning capacity. We therefore propose CURE, a novel training framework that leverages a gradient prominence criterion to perform selective conservation, updating, and revision of weights. Importantly, CURE is designed to be dataset- and architecture-agnostic, ensuring its applicability across various scenarios. It effectively tackles both memorization and overfitting issues, thus enhancing the trade-off between robustness and generalization and additionally, this training approach also aids in mitigating "robust overfitting". Furthermore, our study provides valuable insights into the mechanisms of selective adversarial training and offers a promising avenue for future research.

摘要: 对抗性训练提高了神经网络对对抗性攻击的鲁棒性，尽管代价是标准和鲁棒泛化之间的权衡。为了揭示驱动这一现象的潜在因素，我们研究了神经网络在从标准到对抗环境的过渡过程中的分层学习能力。我们的实证研究结果表明，选择性地更新特定层，同时保留其他层，可以大大提高网络的学习能力。因此，我们提出了CURE，一个新的训练框架，利用梯度突出标准来执行选择性的保护，更新和修订的权重。重要的是，CURE被设计为与数据集和架构无关，确保其适用于各种场景。它有效地解决了记忆和过拟合问题，从而增强了鲁棒性和泛化之间的权衡，此外，这种训练方法还有助于减轻“鲁棒过拟合”。此外，我们的研究为选择性对抗训练的机制提供了有价值的见解，并为未来的研究提供了一个有希望的途径。



## **48. Where and How to Attack? A Causality-Inspired Recipe for Generating Counterfactual Adversarial Examples**

向哪里进攻，如何进攻？由因果关系启发生成反事实对抗性例子的秘诀 cs.LG

Accepted by AAAI-2024

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2312.13628v2) [paper-pdf](http://arxiv.org/pdf/2312.13628v2)

**Authors**: Ruichu Cai, Yuxuan Zhu, Jie Qiao, Zefeng Liang, Furui Liu, Zhifeng Hao

**Abstract**: Deep neural networks (DNNs) have been demonstrated to be vulnerable to well-crafted \emph{adversarial examples}, which are generated through either well-conceived $\mathcal{L}_p$-norm restricted or unrestricted attacks. Nevertheless, the majority of those approaches assume that adversaries can modify any features as they wish, and neglect the causal generating process of the data, which is unreasonable and unpractical. For instance, a modification in income would inevitably impact features like the debt-to-income ratio within a banking system. By considering the underappreciated causal generating process, first, we pinpoint the source of the vulnerability of DNNs via the lens of causality, then give theoretical results to answer \emph{where to attack}. Second, considering the consequences of the attack interventions on the current state of the examples to generate more realistic adversarial examples, we propose CADE, a framework that can generate \textbf{C}ounterfactual \textbf{AD}versarial \textbf{E}xamples to answer \emph{how to attack}. The empirical results demonstrate CADE's effectiveness, as evidenced by its competitive performance across diverse attack scenarios, including white-box, transfer-based, and random intervention attacks.

摘要: 深度神经网络(DNN)已经被证明容易受到精心设计的对手例子的攻击，这些例子是通过精心设计的数学{L}_p$-范数受限或非受限攻击而产生的。然而，这些方法中的大多数都假设对手可以随意修改任何特征，而忽略了数据的因果生成过程，这是不合理和不切实际的。例如，收入的调整将不可避免地影响银行体系内的债务收入比等特征。通过考虑被低估的因果生成过程，我们首先通过因果镜头找出DNN脆弱性的来源，然后给出理论结果来回答{攻击在哪里}。其次，考虑到攻击干预对实例的当前状态的影响，为了生成更真实的对抗性实例，我们提出了CADE框架，它可以生成\extbf{C}非事实\extbf{AD}versariative\extbf{E}样例来回答\emph{如何攻击}。实验结果证明了CADE的有效性，它在各种攻击场景中的竞争性能证明了这一点，包括白盒攻击、基于传输的攻击和随机干预攻击。



## **49. A General Framework for Robust G-Invariance in G-Equivariant Networks**

G-等变网络鲁棒G-不变性的一般框架 cs.LG

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2310.18564v2) [paper-pdf](http://arxiv.org/pdf/2310.18564v2)

**Authors**: Sophia Sanborn, Nina Miolane

**Abstract**: We introduce a general method for achieving robust group-invariance in group-equivariant convolutional neural networks ($G$-CNNs), which we call the $G$-triple-correlation ($G$-TC) layer. The approach leverages the theory of the triple-correlation on groups, which is the unique, lowest-degree polynomial invariant map that is also complete. Many commonly used invariant maps--such as the max--are incomplete: they remove both group and signal structure. A complete invariant, by contrast, removes only the variation due to the actions of the group, while preserving all information about the structure of the signal. The completeness of the triple correlation endows the $G$-TC layer with strong robustness, which can be observed in its resistance to invariance-based adversarial attacks. In addition, we observe that it yields measurable improvements in classification accuracy over standard Max $G$-Pooling in $G$-CNN architectures. We provide a general and efficient implementation of the method for any discretized group, which requires only a table defining the group's product structure. We demonstrate the benefits of this method for $G$-CNNs defined on both commutative and non-commutative groups--$SO(2)$, $O(2)$, $SO(3)$, and $O(3)$ (discretized as the cyclic $C8$, dihedral $D16$, chiral octahedral $O$ and full octahedral $O_h$ groups)--acting on $\mathbb{R}^2$ and $\mathbb{R}^3$ on both $G$-MNIST and $G$-ModelNet10 datasets.

摘要: 介绍了一种在群等变卷积神经网络($G$-CNN)中实现稳健群不变性的一般方法，我们称之为$G$-三相关($G$-TC)层。该方法利用了群上的三重相关理论，这是唯一的、也是完全的最低次多项式不变映射。许多常用的不变量映射--例如max--是不完整的：它们既去掉了组结构又去掉了信号结构。相比之下，完全不变量只删除由于群的作用而引起的变化，同时保留关于信号结构的所有信息。三重相关性的完备性赋予$G$-TC层很强的健壮性，这可以从它对基于不变性的对手攻击的抵抗中观察到。此外，我们观察到它在分类准确率方面比标准的MAX$G$-在$G$-CNN架构中合并产生了显著的改进。我们为任何离散化的组提供了该方法的通用和有效的实现，它只需要一个定义组的产品结构的表。我们证明了这种方法对于定义在交换群和非交换群上的$G$-CNN的好处--$SO(2)$、$O(2)$、$SO(3)$和$O(3)$(离散为循环$C8$、二面体$D16$、手性八面体$O$和全八面体$O_h$群)--作用于$G$-MNIST和$G$-ModelNetData10集上的$\mathbb{R}^2$和$\mathbb{R}^3$。



## **50. Physical Trajectory Inference Attack and Defense in Decentralized POI Recommendation**

分散式POI推荐中的物理轨迹推理攻击与防御 cs.IR

**SubmitDate**: 2024-01-26    [abs](http://arxiv.org/abs/2401.14583v1) [paper-pdf](http://arxiv.org/pdf/2401.14583v1)

**Authors**: Jing Long, Tong Chen, Guanhua Ye, Kai Zheng, Nguyen Quoc Viet Hung, Hongzhi Yin

**Abstract**: As an indispensable personalized service within Location-Based Social Networks (LBSNs), the Point-of-Interest (POI) recommendation aims to assist individuals in discovering attractive and engaging places. However, the accurate recommendation capability relies on the powerful server collecting a vast amount of users' historical check-in data, posing significant risks of privacy breaches. Although several collaborative learning (CL) frameworks for POI recommendation enhance recommendation resilience and allow users to keep personal data on-device, they still share personal knowledge to improve recommendation performance, thus leaving vulnerabilities for potential attackers. Given this, we design a new Physical Trajectory Inference Attack (PTIA) to expose users' historical trajectories. Specifically, for each user, we identify the set of interacted POIs by analyzing the aggregated information from the target POIs and their correlated POIs. We evaluate the effectiveness of PTIA on two real-world datasets across two types of decentralized CL frameworks for POI recommendation. Empirical results demonstrate that PTIA poses a significant threat to users' historical trajectories. Furthermore, Local Differential Privacy (LDP), the traditional privacy-preserving method for CL frameworks, has also been proven ineffective against PTIA. In light of this, we propose a novel defense mechanism (AGD) against PTIA based on an adversarial game to eliminate sensitive POIs and their information in correlated POIs. After conducting intensive experiments, AGD has been proven precise and practical, with minimal impact on recommendation performance.

摘要: 作为基于位置的社交网络（LBSNs）中不可或缺的个性化服务，兴趣点（POI）推荐旨在帮助个人发现有吸引力和吸引人的地方。然而，准确的推荐能力依赖于强大的服务器收集大量用户的历史入住数据，存在很大的隐私泄露风险。尽管一些用于POI推荐的协作学习（CL）框架增强了推荐弹性，并允许用户将个人数据保存在设备上，但它们仍然共享个人知识以提高推荐性能，从而为潜在的攻击者留下漏洞。鉴于此，我们设计了一个新的物理轨迹推理攻击（PTIA）暴露用户的历史轨迹。具体而言，对于每个用户，我们通过分析来自目标POI及其相关POI的聚合信息来识别交互POI的集合。我们评估了PTIA在两个真实世界的数据集上的有效性，这些数据集跨越两种类型的分散CL框架用于POI推荐。实证结果表明，PTIA对用户的历史轨迹构成了重大威胁。此外，局部差分隐私（LDP），CL框架的传统隐私保护方法，也已被证明对PTIA无效。有鉴于此，我们提出了一种新的防御机制（AGD）对PTIA的基础上的对抗性游戏，以消除敏感的兴趣点和相关的兴趣点的信息。经过大量的实验，AGD被证明是精确和实用的，对推荐性能的影响最小。



