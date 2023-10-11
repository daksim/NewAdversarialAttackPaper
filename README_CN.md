# Latest Adversarial Attack Papers
**update at 2023-10-11 17:11:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

基于图的结合特定分布距离的攻击检测方法 cs.LG

published in Neural Networks

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2306.00042v2) [paper-pdf](http://arxiv.org/pdf/2306.00042v2)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an input image, we compute an associated sparse graph using the layer-wise relevance propagation algorithm \cite{bach15}. Specifically, we only keep edges of the neural network with the highest relevance values. Three quantities are then computed from the graph which are then compared against those computed from the training set. The result of the comparison is a classification of the image as benign or adversarial. To make the comparison, two classification methods are introduced: 1) an explicit formula based on Wasserstein distance applied to the degree of node and 2) a logistic regression. Both classification methods produce strong results which lead us to believe that a graph-based interpretation of adversarial attacks is valuable.

摘要: 人工神经网络很容易被精心扰动的输入所愚弄，这会导致严重的错误分类。这些对抗性攻击一直是广泛研究的焦点。同样，在检测和防御它们的方法方面也进行了大量的研究。我们从图的角度介绍了一种新的检测和解释对抗性攻击的方法。对于输入图像，我们使用分层相关传播算法来计算关联稀疏图。具体地说，我们只保留具有最高相关值的神经网络的边缘。然后从图中计算三个量，然后将其与从训练集计算的量进行比较。比较的结果是将图像分类为良性的或敌对的。为了进行比较，引入了两种分类方法：1)基于Wasserstein距离的节点程度的显式公式；2)Logistic回归。这两种分类方法都产生了很强的结果，这使得我们相信基于图的对抗性攻击的解释是有价值的。



## **2. Privacy-oriented manipulation of speaker representations**

面向隐私的说话人表征操作 eess.AS

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06652v1) [paper-pdf](http://arxiv.org/pdf/2310.06652v1)

**Authors**: Francisco Teixeira, Alberto Abad, Bhiksha Raj, Isabel Trancoso

**Abstract**: Speaker embeddings are ubiquitous, with applications ranging from speaker recognition and diarization to speech synthesis and voice anonymisation. The amount of information held by these embeddings lends them versatility, but also raises privacy concerns. Speaker embeddings have been shown to contain information on age, sex, health and more, which speakers may want to keep private, especially when this information is not required for the target task. In this work, we propose a method for removing and manipulating private attributes from speaker embeddings that leverages a Vector-Quantized Variational Autoencoder architecture, combined with an adversarial classifier and a novel mutual information loss. We validate our model on two attributes, sex and age, and perform experiments with ignorant and fully-informed attackers, and with in-domain and out-of-domain data.

摘要: 说话人嵌入是无处不在的，应用范围从说话人识别和二值化到语音合成和语音匿名化。这些嵌入的信息量使它们具有多功能性，但也引发了隐私问题。说话人嵌入已被证明包含年龄、性别、健康等信息，说话人可能希望对这些信息保密，特别是当目标任务不需要这些信息时。在这项工作中，我们提出了一种从说话人嵌入中去除和处理私人属性的方法，该方法利用矢量量化变分自动编码器结构，结合对抗性分类器和新的互信息损失。我们在性别和年龄两个属性上验证了我们的模型，并使用无知和完全知情的攻击者以及域内和域外的数据进行了实验。



## **3. A Geometrical Approach to Evaluate the Adversarial Robustness of Deep Neural Networks**

一种评估深度神经网络对抗健壮性的几何方法 cs.CV

ACM Transactions on Multimedia Computing, Communications, and  Applications (ACM TOMM)

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06468v1) [paper-pdf](http://arxiv.org/pdf/2310.06468v1)

**Authors**: Yang Wang, Bo Dong, Ke Xu, Haiyin Piao, Yufei Ding, Baocai Yin, Xin Yang

**Abstract**: Deep Neural Networks (DNNs) are widely used for computer vision tasks. However, it has been shown that deep models are vulnerable to adversarial attacks, i.e., their performances drop when imperceptible perturbations are made to the original inputs, which may further degrade the following visual tasks or introduce new problems such as data and privacy security. Hence, metrics for evaluating the robustness of deep models against adversarial attacks are desired. However, previous metrics are mainly proposed for evaluating the adversarial robustness of shallow networks on the small-scale datasets. Although the Cross Lipschitz Extreme Value for nEtwork Robustness (CLEVER) metric has been proposed for large-scale datasets (e.g., the ImageNet dataset), it is computationally expensive and its performance relies on a tractable number of samples. In this paper, we propose the Adversarial Converging Time Score (ACTS), an attack-dependent metric that quantifies the adversarial robustness of a DNN on a specific input. Our key observation is that local neighborhoods on a DNN's output surface would have different shapes given different inputs. Hence, given different inputs, it requires different time for converging to an adversarial sample. Based on this geometry meaning, ACTS measures the converging time as an adversarial robustness metric. We validate the effectiveness and generalization of the proposed ACTS metric against different adversarial attacks on the large-scale ImageNet dataset using state-of-the-art deep networks. Extensive experiments show that our ACTS metric is an efficient and effective adversarial metric over the previous CLEVER metric.

摘要: 深度神经网络被广泛应用于计算机视觉任务中。然而，已有研究表明，深层模型容易受到敌意攻击，即当原始输入受到不知不觉的扰动时，其性能会下降，这可能会进一步降低后续的视觉任务或带来新的问题，如数据和隐私安全。因此，需要评估深度模型对敌方攻击的稳健性的度量。然而，以前的度量主要是针对小规模数据集上的浅层网络的对抗健壮性提出的。虽然已经为大规模数据集(例如，ImageNet数据集)提出了网络健壮性的交叉Lipschitz极值(Clear)度量，但它的计算代价很高，并且其性能依赖于可处理的样本数量。在本文中，我们提出了对抗收敛时间得分(ACTS)，这是一种依赖于攻击的度量，它量化了DNN在特定输入上的对抗健壮性。我们的主要观察是，在给定不同输入的情况下，DNN输出表面上的局部邻域将具有不同的形状。因此，给定不同的输入，收敛到对抗性样本所需的时间也不同。基于这一几何意义，ACTS将收敛时间作为对抗性健壮性度量。我们使用最先进的深度网络在大规模ImageNet数据集上验证了所提出的ACTS度量针对不同对手攻击的有效性和泛化能力。大量的实验表明，我们的ACTS度量是一种比以前的聪明度量更有效的对抗性度量。



## **4. Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models**

红色团队博弈：红色团队语言模型的博弈论框架 cs.CL

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.00322v2) [paper-pdf](http://arxiv.org/pdf/2310.00322v2)

**Authors**: Chengdong Ma, Ziran Yang, Minquan Gao, Hai Ci, Jun Gao, Xuehai Pan, Yaodong Yang

**Abstract**: Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

摘要: 可部署的大型语言模型(LLMS)必须符合有益和无害的标准，从而实现LLMS的输出与人的价值之间的一致性。红团队技术构成了实现这一标准的关键途径。现有的工作完全依赖于手动红色团队设计和启发式对抗性提示来进行漏洞检测和优化。这些方法缺乏严格的数学描述，从而限制了在可量化的度量范围内探索多样化的攻击策略，以及在收敛保证下对LLMS进行优化。在本文中，我们提出了一种不需要人工注释的通用博弈论框架--Red-Teaming Game(RTG)。RTG用于分析红队语言模型(RLMS)和蓝队语言模型(BLM)之间的多回合攻防交互。在RTG中，我们提出了一种具有语义空间多样性度量的Gamalized Red-Teaming Solver(GRTS)。GRTS是一种自动红队技术，通过元博弈分析解决RTG向纳什均衡的方向，这对应于理论上保证的RLMS和BLM的优化方向。在RLMS多回合攻击中的实验结果表明，GRTS自主发现多样化的攻击策略，有效地提高了LLMS的安全性，优于已有的启发式红队设计。总体而言，RTG为红色团队任务建立了一个基本框架，并构建了一种新的可扩展的协调监督技术。



## **5. Adversarial Robustness in Graph Neural Networks: A Hamiltonian Approach**

图神经网络的对抗性稳健性：哈密顿方法 cs.LG

Accepted by Advances in Neural Information Processing Systems  (NeurIPS), New Orleans, USA, Dec. 2023, spotlight

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06396v1) [paper-pdf](http://arxiv.org/pdf/2310.06396v1)

**Authors**: Kai Zhao, Qiyu Kang, Yang Song, Rui She, Sijie Wang, Wee Peng Tay

**Abstract**: Graph neural networks (GNNs) are vulnerable to adversarial perturbations, including those that affect both node features and graph topology. This paper investigates GNNs derived from diverse neural flows, concentrating on their connection to various stability notions such as BIBO stability, Lyapunov stability, structural stability, and conservative stability. We argue that Lyapunov stability, despite its common use, does not necessarily ensure adversarial robustness. Inspired by physics principles, we advocate for the use of conservative Hamiltonian neural flows to construct GNNs that are robust to adversarial attacks. The adversarial robustness of different neural flow GNNs is empirically compared on several benchmark datasets under a variety of adversarial attacks. Extensive numerical experiments demonstrate that GNNs leveraging conservative Hamiltonian flows with Lyapunov stability substantially improve robustness against adversarial perturbations. The implementation code of experiments is available at https://github.com/zknus/NeurIPS-2023-HANG-Robustness.

摘要: 图神经网络(GNN)容易受到敌意扰动的影响，包括影响节点特征和图拓扑的扰动。本文研究了来自不同神经流的GNN，集中讨论了它们与各种稳定性概念的联系，如Bibo稳定性、Lyapunov稳定性、结构稳定性和保守稳定性。我们认为，李亚普诺夫稳定性，尽管它的普遍使用，并不一定确保对手的稳健性。受物理学原理的启发，我们提倡使用保守的哈密顿神经流来构造对对手攻击具有健壮性的GNN。在几个基准数据集上对不同神经流GNN在各种对抗攻击下的对抗健壮性进行了经验比较。大量的数值实验表明，GNN利用具有Lyapunov稳定性的保守哈密顿流大大提高了对对手扰动的鲁棒性。有关实验的实现代码，请访问https://github.com/zknus/NeurIPS-2023-HANG-Robustness.



## **6. Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations**

越狱和警卫对齐的语言模型，只有很少的上下文演示 cs.LG

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06387v1) [paper-pdf](http://arxiv.org/pdf/2310.06387v1)

**Authors**: Zeming Wei, Yifei Wang, Yisen Wang

**Abstract**: Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL to influence LLM behavior and provide a new perspective for enhancing the safety and alignment of LLMs.

摘要: 大型语言模型(LLM)在各种任务中取得了显著的成功，但也出现了对其安全性和生成恶意内容的可能性的担忧。在这篇文章中，我们探索了情境中学习(ICL)在操纵LLMS对齐能力方面的力量。我们发现，通过提供很少的上下文演示而不进行微调，LLMS可以被操纵以增加或降低越狱的可能性，即回答恶意提示。基于这些观察，我们提出了上下文中攻击(ICA)和上下文中防御(ICD)方法，用于越狱和保护对齐语言模型。ICA制作恶意上下文来引导模型生成有害输出，而ICD通过演示拒绝回答有害提示来增强模型的稳健性。实验结果表明，ICA和ICD能够有效地提高或降低对抗性越狱攻击的成功率。总体而言，我们阐明了ICL影响LLM行为的潜力，并为提高LLM的安全性和一致性提供了一个新的视角。



## **7. Double Public Key Signing Function Oracle Attack on EdDSA Software Implementations**

双重公钥签名函数Oracle对EdDSA软件实现的攻击 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2308.15009v2) [paper-pdf](http://arxiv.org/pdf/2308.15009v2)

**Authors**: Sam Grierson, Konstantinos Chalkias, William J Buchanan, Leandros Maglaras

**Abstract**: EdDSA is a standardised elliptic curve digital signature scheme introduced to overcome some of the issues prevalent in the more established ECDSA standard. Due to the EdDSA standard specifying that the EdDSA signature be deterministic, if the signing function were to be used as a public key signing oracle for the attacker, the unforgeability notion of security of the scheme can be broken. This paper describes an attack against some of the most popular EdDSA implementations, which results in an adversary recovering the private key used during signing. With this recovered secret key, an adversary can sign arbitrary messages that would be seen as valid by the EdDSA verification function. A list of libraries with vulnerable APIs at the time of publication is provided. Furthermore, this paper provides two suggestions for securing EdDSA signing APIs against this vulnerability while it additionally discusses failed attempts to solve the issue.

摘要: EdDSA是一种标准化的椭圆曲线数字签名方案，引入该方案是为了克服在更成熟的ECDSA标准中普遍存在的一些问题。由于EdDSA标准规定EdDSA签名是确定性的，如果签名函数被用作攻击者的公钥签名预言，则方案的安全性的不可伪造性概念可能被打破。本文描述了对一些最流行的EdDSA实现的攻击，该攻击导致攻击者恢复签名期间使用的私钥。利用恢复的密钥，攻击者可以对EdDSA验证功能认为有效的任意消息进行签名。提供了在发布时具有易受攻击的API的库列表。此外，本文还提供了两条保护EdDSA签名API免受该漏洞攻击的建议，同时还讨论了解决该问题的失败尝试。



## **8. Exploring adversarial attacks in federated learning for medical imaging**

医学影像联合学习中的对抗性攻击研究 cs.CR

**SubmitDate**: 2023-10-10    [abs](http://arxiv.org/abs/2310.06227v1) [paper-pdf](http://arxiv.org/pdf/2310.06227v1)

**Authors**: Erfan Darzi, Florian Dubost, N. M. Sijtsema, P. M. A van Ooijen

**Abstract**: Federated learning offers a privacy-preserving framework for medical image analysis but exposes the system to adversarial attacks. This paper aims to evaluate the vulnerabilities of federated learning networks in medical image analysis against such attacks. Employing domain-specific MRI tumor and pathology imaging datasets, we assess the effectiveness of known threat scenarios in a federated learning environment. Our tests reveal that domain-specific configurations can increase the attacker's success rate significantly. The findings emphasize the urgent need for effective defense mechanisms and suggest a critical re-evaluation of current security protocols in federated medical image analysis systems.

摘要: 联合学习为医学图像分析提供了隐私保护框架，但会使系统面临敌意攻击。本文旨在评估联合学习网络在医学图像分析中抵抗此类攻击的脆弱性。利用特定领域的MRI肿瘤和病理成像数据集，我们评估了联合学习环境中已知威胁场景的有效性。我们的测试表明，特定于域的配置可以显著提高攻击者的成功率。这些发现强调了对有效防御机制的迫切需要，并建议对联合医学图像分析系统中的当前安全协议进行关键的重新评估。



## **9. PAC-Bayesian Spectrally-Normalized Bounds for Adversarially Robust Generalization**

对抗性泛化的PAC-贝叶斯谱归一化界 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.06182v1) [paper-pdf](http://arxiv.org/pdf/2310.06182v1)

**Authors**: Jiancong Xiao, Ruoyu Sun, Zhi-quan Luo

**Abstract**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. It is found empirically that adversarially robust generalization is crucial in establishing defense algorithms against adversarial attacks. Therefore, it is interesting to study the theoretical guarantee of robust generalization. This paper focuses on norm-based complexity, based on a PAC-Bayes approach (Neyshabur et al., 2017). The main challenge lies in extending the key ingredient, which is a weight perturbation bound in standard settings, to the robust settings. Existing attempts heavily rely on additional strong assumptions, leading to loose bounds. In this paper, we address this issue and provide a spectrally-normalized robust generalization bound for DNNs. Compared to existing bounds, our bound offers two significant advantages: Firstly, it does not depend on additional assumptions. Secondly, it is considerably tighter, aligning with the bounds of standard generalization. Therefore, our result provides a different perspective on understanding robust generalization: The mismatch terms between standard and robust generalization bounds shown in previous studies do not contribute to the poor robust generalization. Instead, these disparities solely due to mathematical issues. Finally, we extend the main result to adversarial robustness against general non-$\ell_p$ attacks and other neural network architectures.

摘要: 深度神经网络(DNN)很容易受到敌意攻击。实验发现，对抗性健壮性泛化对于建立抵抗对抗性攻击的防御算法至关重要。因此，研究健壮性泛化的理论保障是很有意义的。本文基于PAC-Bayes方法(Neyshabur等人，2017年)，重点研究基于范数的复杂性。主要的挑战在于将关键成分(标准设置中的权重扰动范围)扩展到稳健设置。现有的尝试严重依赖于额外的强有力的假设，导致了宽松的界限。在这篇文章中，我们解决了这个问题，并为DNN提供了一个谱归一化的鲁棒泛化上界。与现有的边界相比，我们的边界有两个显著的优点：第一，它不依赖于额外的假设。其次，它相当紧凑，与标准泛化的界限一致。因此，我们的结果为理解健壮性概括提供了一个不同的视角：以前的研究中显示的标准和健壮性概化界限之间的不匹配项并不是导致健壮性概括较差的原因。相反，这些差异完全是由于数学问题。最后，我们将主要结果推广到对抗一般非EELL_p$攻击和其他神经网络结构的健壮性。



## **10. Lessons Learned: Defending Against Property Inference Attacks**

经验教训：防御属性推理攻击 cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2205.08821v4) [paper-pdf](http://arxiv.org/pdf/2205.08821v4)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.

摘要: 本文研究和评估了针对机器学习模型的隐私攻击--属性推理攻击(PIA)的多种防御策略。给定一个经过训练的机器学习模型，PIA的目标是提取其基本训练数据的统计属性，例如，揭示医学训练数据集中的男性和女性的比例。虽然对于其他隐私攻击，如成员身份推断，已经发表了大量关于防御机制的研究，但这是第一个专注于防御PIA的工作。以开发一种针对白盒PIA的通用缓解策略为主要目标，我们提出了一种新的方法-属性遗忘。大量的属性遗忘实验表明，虽然它在防御特定对手的目标模型时非常有效，但属性遗忘不能泛化，即保护一整类PIA。为了调查这种限制背后的原因，我们给出了使用可解释的人工智能工具LIME的实验结果。它们展示了具有相同目标的最先进的属性推理对手如何专注于目标模型的不同部分。我们通过后续实验进一步阐述了这一点，在该实验中，我们使用可视化技术t-SNE来展示统计训练数据属性在机器学习模型中的表现是多么严重。在此基础上，我们提出了这样的猜测，即训练后的技术，如属性遗忘，可能不足以提供理想的通用保护，以防止PIA。作为另一种选择，我们研究了更简单的训练数据预处理方法，如向训练数据集的图像添加高斯噪声对PIA成功率的影响。最后，我们讨论了不同的防御方法，总结了经验教训，并为未来的工作提供了方向。



## **11. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

量子分类器多分类任务的普遍对抗性扰动 quant-ph

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2306.11974v2) [paper-pdf](http://arxiv.org/pdf/2306.11974v2)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.

摘要: 量子对抗机器学习是一个新兴的研究领域，它研究量子学习系统对对抗扰动的脆弱性，并开发可能的防御策略。量子通用对抗性扰动是一种微小的扰动，它可以使不同的输入样本变成可能欺骗给定量子分类器的对抗性例子。这是一个很少被研究但值得研究的领域，因为普遍的扰动可能会在很大程度上简化恶意攻击，给量子机器学习模型造成意想不到的破坏。在这篇文章中，我们向前迈进了一步，探索了异质分类任务背景下的量子普适微扰。特别是，我们发现，在两个不同的分类任务上获得几乎最先进的精度的量子分类器都可能最终被一个精心设计的普遍扰动所欺骗。这一结果通过设计良好的弹性权重巩固方法的量子连续学习模型来避免灾难性遗忘，以及来自手写数字和医学MRI图像的现实生活中的异质数据集得到了明确的证明。我们的结果提供了一种简单而有效的方法来产生对异类分类任务的普遍扰动，从而为未来的量子学习技术提供了有价值的指导。



## **12. RECESS Vaccine for Federated Learning: Proactive Defense Against Model Poisoning Attacks**

联邦学习的休会疫苗：对模型中毒攻击的主动防御 cs.CR

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05431v1) [paper-pdf](http://arxiv.org/pdf/2310.05431v1)

**Authors**: Haonan Yan, Wenjing Zhang, Qian Chen, Xiaoguang Li, Wenhai Sun, Hui Li, Xiaodong Lin

**Abstract**: Model poisoning attacks greatly jeopardize the application of federated learning (FL). The effectiveness of existing defenses is susceptible to the latest model poisoning attacks, leading to a decrease in prediction accuracy. Besides, these defenses are intractable to distinguish benign outliers from malicious gradients, which further compromises the model generalization. In this work, we propose a novel proactive defense named RECESS against model poisoning attacks. Different from the passive analysis in previous defenses, RECESS proactively queries each participating client with a delicately constructed aggregation gradient, accompanied by the detection of malicious clients according to their responses with higher accuracy. Furthermore, RECESS uses a new trust scoring mechanism to robustly aggregate gradients. Unlike previous methods that score each iteration, RECESS considers clients' performance correlation across multiple iterations to estimate the trust score, substantially increasing fault tolerance. Finally, we extensively evaluate RECESS on typical model architectures and four datasets under various settings. We also evaluated the defensive effectiveness against other types of poisoning attacks, the sensitivity of hyperparameters, and adaptive adversarial attacks. Experimental results show the superiority of RECESS in terms of reducing accuracy loss caused by the latest model poisoning attacks over five classic and two state-of-the-art defenses.

摘要: 模型中毒攻击极大地危害了联邦学习的应用。现有防御的有效性容易受到最新模型中毒攻击的影响，导致预测准确性下降。此外，这些防御措施很难区分良性异常值和恶意梯度，这进一步损害了模型的泛化。在这项工作中，我们提出了一种新的针对模型中毒攻击的主动防御机制--休息。与以往防御中的被动分析不同，Recess通过精心构建的聚合梯度主动查询每个参与的客户端，并根据恶意客户端的响应检测恶意客户端，具有更高的准确率。此外，SESESS使用了一种新的信任评分机制来稳健地聚合梯度。与以前对每次迭代进行评分的方法不同，Recess考虑客户在多次迭代中的性能相关性来估计信任分数，从而大大提高了容错能力。最后，我们在典型的模型结构和四个不同设置下的数据集上对RESECT进行了广泛的评估。我们还评估了对其他类型的中毒攻击的防御效果、超参数的敏感度和自适应对抗性攻击。实验结果表明，在减少最新模型中毒攻击造成的精度损失方面，SESECT优于五个经典防御系统和两个最新防御系统。



## **13. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

AdvSV：一种用于说话人确认的空中对抗攻击数据集 cs.SD

Submitted to ICASSP2024

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05369v1) [paper-pdf](http://arxiv.org/pdf/2310.05369v1)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC-BY license. In addition, we also provide a detection baseline for reproducible research.

摘要: 众所周知，深度神经网络很容易受到敌意攻击。尽管建立在深度神经网络之上的自动说话人确认(ASV)在受控场景下表现出较强的性能，但许多研究证实ASV容易受到对手攻击。缺乏标准数据集是进一步研究的瓶颈，特别是可重复性研究。在这项研究中，我们开发了一个开源的对抗性攻击数据集，用于说话人验证研究。作为第一步，我们把重点放在空中攻击上。空中对抗性攻击涉及扰动生成算法、扬声器、麦克风和声学环境。记录配置的变化使得复制先前的研究非常具有挑战性。AdvSV数据集是使用Voxeleb1验证测试集作为其基础来构建的。该数据集采用了典型的受到对抗性攻击的ASV模型，并记录了对抗性样本以模拟空中攻击设置。数据集的范围可以很容易地扩展到包括更多类型的对抗性攻击。数据集将根据CC-BY许可证向公众发布。此外，我们还为可重复性研究提供了检测基线。



## **14. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

用于自动说话人确认的空中对抗扰动神经重放模拟器的初步研究 cs.SD

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05354v1) [paper-pdf](http://arxiv.org/pdf/2310.05354v1)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.

摘要: 在过去的几年里，深度学习发展了自动说话人确认(ASV)。虽然众所周知，基于深度学习的ASV系统在数字访问中容易受到敌意攻击，但在涉及重播过程(即空中重播)的物理访问环境中，很少有关于对抗性攻击的研究。空中攻击包括扬声器、麦克风和影响声波移动的重放环境。我们的初步实验证实，重放过程会影响空中攻击性能的有效性。本研究对利用神经重放模拟器来提高空中对抗攻击的稳健性进行了初步的研究。这是通过使用神经波形合成器来模拟在估计对抗性扰动时的重播过程来实现的。在ASVspoof2019数据集上进行的实验证实，神经重放模拟器可以显著提高空中对抗性攻击的成功率。这引起了人们对物理访问应用中说话人验证的对抗性攻击的关注。



## **15. GReAT: A Graph Regularized Adversarial Training Method**

GREAT：一种图规化的对抗性训练方法 cs.LG

25 pages including references. 7 figures and 4 tables

**SubmitDate**: 2023-10-09    [abs](http://arxiv.org/abs/2310.05336v1) [paper-pdf](http://arxiv.org/pdf/2310.05336v1)

**Authors**: Samet Bayram, Kenneth Barner

**Abstract**: This paper proposes a regularization method called GReAT, Graph Regularized Adversarial Training, to improve deep learning models' classification performance. Adversarial examples are a well-known challenge in machine learning, where small, purposeful perturbations to input data can mislead models. Adversarial training, a powerful and one of the most effective defense strategies, involves training models with both regular and adversarial examples. However, it often neglects the underlying structure of the data. In response, we propose GReAT, a method that leverages data graph structure to enhance model robustness. GReAT deploys the graph structure of the data into the adversarial training process, resulting in more robust models that better generalize its testing performance and defend against adversarial attacks. Through extensive evaluation on benchmark datasets, we demonstrate GReAT's effectiveness compared to state-of-the-art classification methods, highlighting its potential in improving deep learning models' classification performance.

摘要: 为了提高深度学习模型的分类性能，提出了一种称为大图正则化对抗性训练的正则化方法。对抗性的例子是机器学习中众所周知的挑战，在机器学习中，对输入数据的微小、有目的的扰动可能会误导模型。对抗性训练是一种强有力的、最有效的防御策略之一，它包括常规训练模式和对抗性训练模式。然而，它经常忽略数据的底层结构。对此，我们提出了一种利用数据图结构来增强模型稳健性的方法--GREAT。Great将数据的图形结构部署到对抗性训练过程中，从而产生更健壮的模型，更好地概括其测试性能并防御对抗性攻击。通过在基准数据集上的广泛评估，我们展示了相对于最先进的分类方法的有效性，突出了其在提高深度学习模型的分类性能方面的潜力。



## **16. On the Query Complexity of Training Data Reconstruction in Private Learning**

私学中训练数据重构的查询复杂性研究 cs.LG

Matching upper bounds, new corollaries for DP variants

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2303.16372v5) [paper-pdf](http://arxiv.org/pdf/2303.16372v5)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.

摘要: 我们分析了白盒攻击者为了重建其训练数据而需要向私人学习者进行的查询数量。对于具有来自任意紧致度量空间的训练数据的$(\epsilon，\Delta)$DP学习者，我们提供了作为学习者隐私参数的函数的\emph(对手查询复杂性的第一个已知下界)。{我们的结果对[0，1]$中的每个$\epsilon\geq0，\Delta\都是极小极大最优的，推论包括$\epsilon$-dp和$(0，\Delta)$dp}。在此基础上，我们得到了$(\α，\epsilon)$R‘Enyi DP学习者的查询复杂性下界，这些下界对任何$\α>1，\epsion\0$都是有效的。最后，我们通过度量DP框架分析了局部紧度量空间上的数据重构攻击。度量DP是DP的推广，它解释了数据的基本度量结构。在这个背景下，我们首次对无界高维空间中的数据重构进行了分析，得到了几乎紧模对数因子的查询复杂度下界。



## **17. Adversarial Attacks on Combinatorial Multi-Armed Bandits**

组合式多臂土匪的对抗性攻击 cs.LG

28 pages

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05308v1) [paper-pdf](http://arxiv.org/pdf/2310.05308v1)

**Authors**: Rishab Balasubramanian, Jiawei Li, Prasad Tadepalli, Huazheng Wang, Qingyun Wu, Haoyu Zhao

**Abstract**: We study reward poisoning attacks on Combinatorial Multi-armed Bandits (CMAB). We first provide a sufficient and necessary condition for the attackability of CMAB, which depends on the intrinsic properties of the corresponding CMAB instance such as the reward distributions of super arms and outcome distributions of base arms. Additionally, we devise an attack algorithm for attackable CMAB instances. Contrary to prior understanding of multi-armed bandits, our work reveals a surprising fact that the attackability of a specific CMAB instance also depends on whether the bandit instance is known or unknown to the adversary. This finding indicates that adversarial attacks on CMAB are difficult in practice and a general attack strategy for any CMAB instance does not exist since the environment is mostly unknown to the adversary. We validate our theoretical findings via extensive experiments on real-world CMAB applications including probabilistic maximum covering problem, online minimum spanning tree, cascading bandits for online ranking, and online shortest path.

摘要: 研究了组合多臂土匪(CMAB)的悬赏中毒攻击。我们首先给出了CMAB可攻击性的充要条件，该条件依赖于相应CMAB实例的内在性质，如超臂的奖赏分布和基臂的结果分布。此外，我们还设计了一个针对可攻击CMAB实例的攻击算法。与之前对多武装强盗的理解相反，我们的工作揭示了一个令人惊讶的事实，即特定CMAB实例的可攻击性还取决于对手是否知道该强盗实例。这一发现表明，对CMAB的对抗性攻击在实践中是困难的，并且不存在针对任何CMAB实例的通用攻击策略，因为对手基本上不知道环境。通过在概率最大覆盖问题、在线最小生成树、在线排名的级联强盗和在线最短路径等实际CMAB应用上的大量实验，我们验证了我们的理论发现。



## **18. Robust Lipschitz Bandits to Adversarial Corruptions**

对抗腐败的强健Lipschitz Bandits cs.LG

Thirty-seventh Conference on Neural Information Processing Systems  (NeurIPS 2023)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2305.18543v2) [paper-pdf](http://arxiv.org/pdf/2305.18543v2)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.

摘要: Lipschitz Bandit是随机强盗的变种，它处理定义在度量空间上的连续臂集，其中奖励函数受Lipschitz约束。在这篇文章中，我们引入了一个新的存在对抗性腐败的Lipschitz强盗问题，其中一个自适应的敌对者破坏了总预算为$C$的随机报酬。预算是通过整个时间范围内腐败程度的总和新台币来衡量的。我们既考虑弱对手，也考虑强对手，其中弱对手在攻击前不知道当前的行动，而强对手可以观察到。我们的工作提出了第一行稳健的Lipschitz强盗算法，即使在代理未透露腐败$C$的总预算时，该算法也可以在这两种类型的对手下实现次线性遗憾。在每种类型的对手下，我们给出了一个下界，并证明了我们的算法在强情形下是最优的。最后，通过实验验证了该算法对两种经典攻击的有效性。



## **19. Susceptibility of Continual Learning Against Adversarial Attacks**

持续学习对敌意攻击的敏感性 cs.LG

18 pages, 13 figures

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2207.05225v5) [paper-pdf](http://arxiv.org/pdf/2207.05225v5)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam, Qasim Zia

**Abstract**: Recent continual learning approaches have primarily focused on mitigating catastrophic forgetting. Nevertheless, two critical areas have remained relatively unexplored: 1) evaluating the robustness of proposed methods and 2) ensuring the security of learned tasks. This paper investigates the susceptibility of continually learned tasks, including current and previously acquired tasks, to adversarial attacks. Specifically, we have observed that any class belonging to any task can be easily targeted and misclassified as the desired target class of any other task. Such susceptibility or vulnerability of learned tasks to adversarial attacks raises profound concerns regarding data integrity and privacy. To assess the robustness of continual learning approaches, we consider continual learning approaches in all three scenarios, i.e., task-incremental learning, domain-incremental learning, and class-incremental learning. In this regard, we explore the robustness of three regularization-based methods, three replay-based approaches, and one hybrid technique that combines replay and exemplar approaches. We empirically demonstrated that in any setting of continual learning, any class, whether belonging to the current or previously learned tasks, is susceptible to misclassification. Our observations identify potential limitations of continual learning approaches against adversarial attacks and highlight that current continual learning algorithms could not be suitable for deployment in real-world settings.

摘要: 最近的持续学习方法主要集中在减轻灾难性遗忘上。然而，有两个关键领域仍然相对未被探索：1)评估所提出方法的稳健性；2)确保学习任务的安全性。本文研究了持续学习任务，包括当前任务和以前获得的任务，对敌意攻击的敏感性。具体地说，我们观察到，属于任何任务的任何类都很容易成为目标，并被错误地归类为任何其他任务所需的目标类。学习任务对敌意攻击的这种易感性或脆弱性引起了人们对数据完整性和隐私的深切关注。为了评估持续学习方法的稳健性，我们考虑了三种情况下的持续学习方法，即任务增量学习、领域增量学习和班级增量学习。在这方面，我们探索了三种基于正则化的方法、三种基于回放的方法以及一种结合了回放和样本方法的混合技术的稳健性。我们的经验表明，在任何持续学习的环境中，任何班级，无论是属于当前学习的还是以前学习的任务，都容易发生错误分类。我们的观察发现了持续学习方法在对抗对手攻击时的潜在局限性，并强调了当前的持续学习算法不适合在现实世界中部署。



## **20. Transferable Availability Poisoning Attacks**

可转让可用性中毒攻击 cs.CR

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05141v1) [paper-pdf](http://arxiv.org/pdf/2310.05141v1)

**Authors**: Yiyong Liu, Michael Backes, Xiao Zhang

**Abstract**: We consider availability data poisoning attacks, where an adversary aims to degrade the overall test accuracy of a machine learning model by crafting small perturbations to its training data. Existing poisoning strategies can achieve the attack goal but assume the victim to employ the same learning method as what the adversary uses to mount the attack. In this paper, we argue that this assumption is strong, since the victim may choose any learning algorithm to train the model as long as it can achieve some targeted performance on clean data. Empirically, we observe a large decrease in the effectiveness of prior poisoning attacks if the victim uses a different learning paradigm to train the model and show marked differences in frequency-level characteristics between perturbations generated with respect to different learners and attack methods. To enhance the attack transferability, we propose Transferable Poisoning, which generates high-frequency poisoning perturbations by alternately leveraging the gradient information with two specific algorithms selected from supervised and unsupervised contrastive learning paradigms. Through extensive experiments on benchmark image datasets, we show that our transferable poisoning attack can produce poisoned samples with significantly improved transferability, not only applicable to the two learners used to devise the attack but also for learning algorithms and even paradigms beyond.

摘要: 我们考虑可用性数据中毒攻击，其中对手的目标是通过对机器学习模型的训练数据进行小的扰动来降低其总体测试精度。现有的中毒策略可以达到攻击目标，但假设受害者使用与对手发动攻击相同的学习方法。在本文中，我们认为这一假设是强有力的，因为受害者可以选择任何学习算法来训练模型，只要它能够在干净的数据上达到一些目标性能。从经验上看，如果受害者使用不同的学习范式来训练模型，并显示出针对不同学习者和攻击方法产生的扰动之间的频率水平特征显著差异，我们观察到先前中毒攻击的有效性大幅下降。为了增强攻击的可转移性，我们提出了可传递中毒，通过交替利用梯度信息和从监督和非监督对比学习范例中选择的两种特定算法来产生高频中毒扰动。通过在基准图像数据集上的大量实验表明，我们的可转移中毒攻击可以产生中毒样本，并显著提高了可转移性，不仅适用于设计攻击的两个学习器，而且适用于学习算法甚至更多的范例。



## **21. Model Extraction Attack against Self-supervised Speech Models**

针对自监督语音模型的模型提取攻击 cs.SD

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2211.16044v2) [paper-pdf](http://arxiv.org/pdf/2211.16044v2)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.

摘要: 自监督学习(SSL)语音模型生成给定片段的有意义的表示，并在各种下游任务中获得令人难以置信的性能。模型提取攻击(MEA)通常是指攻击者仅通过查询访问来窃取受害者模型的功能。在这项工作中，我们研究了带有少量查询的SSL语音模型的MEA问题。我们提出了一个两阶段的模型提取框架。在第一阶段，对大规模的未标注语料库进行SSL，以预先训练一个小的语音模型。其次，我们从未标注的语料库中主动采样一小部分片段，并用这些片段查询目标模型，以获得它们的表示作为小模型第二阶段训练的标签。实验结果表明，我们的采样方法可以在不知道目标模型结构的情况下有效地提取目标模型。



## **22. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

一种保障车辆自主感知的异常行为分析框架 cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2023-10-08    [abs](http://arxiv.org/abs/2310.05041v1) [paper-pdf](http://arxiv.org/pdf/2310.05041v1)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.

摘要: 作为一个快速发展的网络物理平台，自动驾驶汽车(AVs)随着其能力的不断扩大，面临着更多的安全挑战。近年来，对手积极瞄准自动驾驶车辆的感知传感器，进行复杂的攻击，而这些攻击不容易被车辆的控制系统检测到。本文提出了一种异常行为分析方法来检测感知传感器对自动驾驶车辆的攻击。该框架依赖于从基于物理的自动驾驶车辆行为模型中提取的时间特征来捕捉自动驾驶中车辆感知的正常行为。通过结合基于模型的技术和机器学习算法，该框架区分了正常和异常的车辆感知行为。为了验证该框架在实践中的应用，我们在自主车辆试验台上进行了深度相机攻击实验，并生成了大量的数据集。我们使用这些真实世界的数据验证了提出的框架的有效性，并发布了数据集以供公众访问。据我们所知，该数据集是此类数据的第一个，将为研究界提供宝贵的资源，以有效地评估他们的入侵检测技术。



## **23. Robust Network Pruning With Sparse Entropic Wasserstein Regression**

基于稀疏熵Wasserstein回归的稳健网络剪枝 cs.AI

submitted to ICLR 2024

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04918v1) [paper-pdf](http://arxiv.org/pdf/2310.04918v1)

**Authors**: Lei You, Hei Victor Cheng

**Abstract**: This study unveils a cutting-edge technique for neural network pruning that judiciously addresses noisy gradients during the computation of the empirical Fisher Information Matrix (FIM). We introduce an entropic Wasserstein regression (EWR) formulation, capitalizing on the geometric attributes of the optimal transport (OT) problem. This is analytically showcased to excel in noise mitigation by adopting neighborhood interpolation across data points. The unique strength of the Wasserstein distance is its intrinsic ability to strike a balance between noise reduction and covariance information preservation. Extensive experiments performed on various networks show comparable performance of the proposed method with state-of-the-art (SoTA) network pruning algorithms. Our proposed method outperforms the SoTA when the network size or the target sparsity is large, the gain is even larger with the existence of noisy gradients, possibly from noisy data, analog memory, or adversarial attacks. Notably, our proposed method achieves a gain of 6% improvement in accuracy and 8% improvement in testing loss for MobileNetV1 with less than one-fourth of the network parameters remaining.

摘要: 这项研究揭示了一种用于神经网络修剪的尖端技术，它明智地解决了经验Fisher信息矩阵(FIM)计算过程中的噪声梯度问题。利用最优运输(OT)问题的几何属性，我们引入了一个熵Wasserstein回归(EWR)公式。分析表明，通过采用跨数据点的邻域内插，这在噪声缓解方面表现出色。沃瑟斯坦距离的独特优势在于它在降噪和保留协方差信息之间取得平衡的内在能力。在不同网络上进行的大量实验表明，该方法的性能与最新的网络剪枝算法(SOTA)相当。当网络规模或目标稀疏度较大时，我们提出的方法的性能优于SOTA，当存在噪声梯度时，增益甚至更大，可能来自噪声数据、模拟记忆或敌对攻击。值得注意的是，我们提出的方法在剩余不到四分之一的网络参数的情况下，对MobileNetV1实现了6%的准确率提高和8%的测试损失改善。



## **24. A Survey of Graph Unlearning**

图忘却学习研究综述 cs.LG

22 page review paper on graph unlearning

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.02164v2) [paper-pdf](http://arxiv.org/pdf/2310.02164v2)

**Authors**: Anwar Said, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the right to be forgotten. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. Additionally, we establish the vital connections between graph unlearning and differential privacy, augmenting our understanding of the relevance of privacy-preserving techniques in this context. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, and resource-constrained environments like the Internet of Things (IoT), illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.

摘要: 在追求负责任的人工智能方面，图形遗忘成为一个关键的进步，提供了从训练的模型中移除敏感数据痕迹的手段，从而维护了被遗忘的权利。显然，图机器学习表现出对数据隐私和敌意攻击的敏感性，因此有必要应用图遗忘技术来有效地解决这些问题。在这篇全面的调查论文中，我们提出了第一次系统地回顾图形遗忘方法，包括一系列不同的方法，并提供了详细的分类和最新的文献综述，以促进新进入该领域的研究人员的理解。此外，我们建立了图遗忘和差异隐私之间的重要联系，增强了我们对隐私保护技术在这一背景下的相关性的理解。为了确保清晰，我们对图形遗忘中使用的基本概念和评估措施进行了清晰的解释，以迎合具有不同专业水平的更广泛的受众。深入挖掘潜在的应用，我们探索了图遗忘在不同领域的多功能性，包括但不限于社交网络、对抗性环境和物联网(IoT)等资源受限环境，说明了它在保护数据隐私和增强AI系统健壮性方面的潜在影响。最后，我们阐明了有前途的研究方向，鼓励在图忘却学习领域内的进一步进步和创新。通过奠定坚实的基础和促进持续进步，这项调查旨在激励研究人员进一步推进图形遗忘领域，从而灌输对人工智能系统伦理增长的信心，并加强机器学习技术在各个领域的负责任应用。



## **25. Untargeted White-box Adversarial Attack with Heuristic Defence Methods in Real-time Deep Learning based Network Intrusion Detection System**

基于实时深度学习的网络入侵检测系统中基于启发式防御方法的非目标白盒攻击 cs.LG

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.03334v2) [paper-pdf](http://arxiv.org/pdf/2310.03334v2)

**Authors**: Khushnaseeb Roshan, Aasim Zafar, Sheikh Burhan Ul Haque

**Abstract**: Network Intrusion Detection System (NIDS) is a key component in securing the computer network from various cyber security threats and network attacks. However, consider an unfortunate situation where the NIDS is itself attacked and vulnerable more specifically, we can say, How to defend the defender?. In Adversarial Machine Learning (AML), the malicious actors aim to fool the Machine Learning (ML) and Deep Learning (DL) models to produce incorrect predictions with intentionally crafted adversarial examples. These adversarial perturbed examples have become the biggest vulnerability of ML and DL based systems and are major obstacles to their adoption in real-time and mission-critical applications such as NIDS. AML is an emerging research domain, and it has become a necessity for the in-depth study of adversarial attacks and their defence strategies to safeguard the computer network from various cyber security threads. In this research work, we aim to cover important aspects related to NIDS, adversarial attacks and its defence mechanism to increase the robustness of the ML and DL based NIDS. We implemented four powerful adversarial attack techniques, namely, Fast Gradient Sign Method (FGSM), Jacobian Saliency Map Attack (JSMA), Projected Gradient Descent (PGD) and Carlini & Wagner (C&W) in NIDS. We analyzed its performance in terms of various performance metrics in detail. Furthermore, the three heuristics defence strategies, i.e., Adversarial Training (AT), Gaussian Data Augmentation (GDA) and High Confidence (HC), are implemented to improve the NIDS robustness under adversarial attack situations. The complete workflow is demonstrated in real-time network with data packet flow. This research work provides the overall background for the researchers interested in AML and its implementation from a computer network security point of view.

摘要: 网络入侵检测系统是保障计算机网络免受各种网络安全威胁和网络攻击的重要组成部分。然而，考虑到一个不幸的情况，NIDS本身也受到攻击和攻击，更具体地说，我们可以说，如何保护防御者？在对抗性机器学习(AML)中，恶意行为者旨在欺骗机器学习(ML)和深度学习(DL)模型，通过故意制作的对抗性示例来产生错误的预测。这些对抗性扰动的例子已经成为基于ML和DL的系统的最大漏洞，并成为它们在实时和任务关键型应用(如网络入侵检测系统)中采用的主要障碍。反洗钱是一个新兴的研究领域，深入研究敌方攻击及其防御策略已成为保障计算机网络免受各种网络安全威胁的必要手段。在这项研究工作中，我们的目标是涵盖与网络入侵检测系统相关的重要方面，对手攻击及其防御机制，以增加基于ML和DL的网络入侵检测系统的健壮性。在网络入侵检测系统中实现了四种强大的对抗性攻击技术，即快速梯度符号方法(FGSM)、雅可比显著图攻击(JSMA)、投影梯度下降(PGD)和Carlini&Wagner(C&W)。我们从各种性能度量的角度详细分析了它的性能。在此基础上，提出了三种启发式防御策略，即对抗性训练(AT)、高斯数据增强(GDA)和高置信度(HC)，以提高网络入侵检测系统在对抗性攻击情况下的鲁棒性。整个工作流程在实时网络中以数据包流的形式进行演示。这项研究工作为从计算机网络安全的角度对AML及其实现感兴趣的研究人员提供了总体背景。



## **26. Understanding and Improving Adversarial Attacks on Latent Diffusion Model**

对潜在扩散模型的敌意攻击的理解与改进 cs.CV

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04687v1) [paper-pdf](http://arxiv.org/pdf/2310.04687v1)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Latent Diffusion Model (LDM) has emerged as a leading tool in image generation, particularly with its capability in few-shot generation. This capability also presents risks, notably in unauthorized artwork replication and misinformation generation. In response, adversarial attacks have been designed to safeguard personal images from being used as reference data. However, existing adversarial attacks are predominantly empirical, lacking a solid theoretical foundation. In this paper, we introduce a comprehensive theoretical framework for understanding adversarial attacks on LDM. Based on the framework, we propose a novel adversarial attack that exploits a unified target to guide the adversarial attack both in the forward and the reverse process of LDM. We provide empirical evidences that our method overcomes the offset problem of the optimization of adversarial attacks in existing methods. Through rigorous experiments, our findings demonstrate that our method outperforms current attacks and is able to generalize over different state-of-the-art few-shot generation pipelines based on LDM. Our method can serve as a stronger and efficient tool for people exposed to the risk of data privacy and security to protect themselves in the new era of powerful generative models. The code is available on GitHub: https://github.com/CaradryanLiang/ImprovedAdvDM.git.

摘要: 潜在扩散模型(LDM)已经成为图像生成的主要工具，特别是它在少镜头生成方面的能力。此功能也存在风险，特别是在未经授权的图稿复制和错误信息生成方面。对此，敌意攻击的目的是保护个人形象不被用作参考数据。然而，现有的对抗性攻击主要是经验性的，缺乏坚实的理论基础。在这篇文章中，我们介绍了一个全面的理论框架来理解对LDM的对抗性攻击。基于该框架，我们提出了一种新颖的对抗性攻击，它利用一个统一的目标来引导LDM的正反向过程中的对抗性攻击。实验结果表明，该方法克服了现有方法中对抗性攻击优化的抵消问题。通过严格的实验，我们的发现表明，我们的方法优于现有的攻击，并且能够在不同的基于LDM的最先进的少镜头生成流水线上推广。我们的方法可以作为一个更强大和高效的工具，让暴露在数据隐私和安全风险中的人在强大的生成模型的新时代保护自己。代码可在giHub上找到：https://github.com/CaradryanLiang/ImprovedAdvDM.git.



## **27. VLAttack: Multimodal Adversarial Attacks on Vision-Language Tasks via Pre-trained Models**

VLAttack：通过预先训练的模型对视觉语言任务进行多模式对抗性攻击 cs.CR

Accepted by NeurIPS 2023

**SubmitDate**: 2023-10-07    [abs](http://arxiv.org/abs/2310.04655v1) [paper-pdf](http://arxiv.org/pdf/2310.04655v1)

**Authors**: Ziyi Yin, Muchao Ye, Tianrong Zhang, Tianyu Du, Jinguo Zhu, Han Liu, Jinghui Chen, Ting Wang, Fenglong Ma

**Abstract**: Vision-Language (VL) pre-trained models have shown their superiority on many multimodal tasks. However, the adversarial robustness of such models has not been fully explored. Existing approaches mainly focus on exploring the adversarial robustness under the white-box setting, which is unrealistic. In this paper, we aim to investigate a new yet practical task to craft image and text perturbations using pre-trained VL models to attack black-box fine-tuned models on different downstream tasks. Towards this end, we propose VLAttack to generate adversarial samples by fusing perturbations of images and texts from both single-modal and multimodal levels. At the single-modal level, we propose a new block-wise similarity attack (BSA) strategy to learn image perturbations for disrupting universal representations. Besides, we adopt an existing text attack strategy to generate text perturbations independent of the image-modal attack. At the multimodal level, we design a novel iterative cross-search attack (ICSA) method to update adversarial image-text pairs periodically, starting with the outputs from the single-modal level. We conduct extensive experiments to attack three widely-used VL pretrained models for six tasks on eight datasets. Experimental results show that the proposed VLAttack framework achieves the highest attack success rates on all tasks compared with state-of-the-art baselines, which reveals a significant blind spot in the deployment of pre-trained VL models. Codes will be released soon.

摘要: 视觉-语言(VL)预训练模型在许多多通道任务中显示了其优越性。然而，这类模型的对抗性健壮性还没有得到充分的研究。现有的研究方法主要集中在研究白盒环境下的对抗稳健性，这是不现实的。在本文中，我们的目标是研究一种新的但实用的任务，使用预先训练的VL模型来攻击不同下游任务的黑盒微调模型来制造图像和文本扰动。为此，我们提出了VLAttack，通过从单模和多模两个层次融合图像和文本的扰动来生成对抗性样本。在单模式层面，我们提出了一种新的分块相似性攻击(BSA)策略来学习图像扰动以破坏通用表示。此外，我们采用了一种现有的文本攻击策略来产生独立于图像模式攻击的文本扰动。在多模态级，我们设计了一种新的迭代交叉搜索攻击(ICSA)方法，从单模式级的输出开始，周期性地更新敌方图文对。我们进行了广泛的实验来攻击三个广泛使用的VL预训练模型，在八个数据集上执行六个任务。实验结果表明，VLAttack框架在所有任务上获得了最高的攻击成功率，这揭示了预先训练的VL模型的部署存在明显的盲点。代码很快就会发布。



## **28. RETVec: Resilient and Efficient Text Vectorizer**

RETVec：弹性高效的文本向量器 cs.CL

Accepted at NeurIPS 2023

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2302.09207v2) [paper-pdf](http://arxiv.org/pdf/2302.09207v2)

**Authors**: Elie Bursztein, Marina Zhang, Owen Vallis, Xinyu Jia, Alexey Kurakin

**Abstract**: This paper describes RETVec, an efficient, resilient, and multilingual text vectorizer designed for neural-based text processing. RETVec combines a novel character encoding with an optional small embedding model to embed words into a 256-dimensional vector space. The RETVec embedding model is pre-trained using pair-wise metric learning to be robust against typos and character-level adversarial attacks. In this paper, we evaluate and compare RETVec to state-of-the-art vectorizers and word embeddings on popular model architectures and datasets. These comparisons demonstrate that RETVec leads to competitive, multilingual models that are significantly more resilient to typos and adversarial text attacks. RETVec is available under the Apache 2 license at https://github.com/google-research/retvec.

摘要: 本文描述了RETVec，一个高效的、有弹性的、多语言的文本向量器，是为基于神经的文本处理而设计的。RETVec将一种新颖的字符编码与可选的小嵌入模型相结合，将单词嵌入到256维矢量空间中。RETVec嵌入模型使用成对度量学习进行预训练，以对打字错误和字符级对手攻击具有健壮性。在这篇文章中，我们评估和比较了RETVEC与最先进的向量化器和单词嵌入在流行的模型体系结构和数据集上的性能。这些比较表明，RETVec导致了具有竞争力的多语言模型，这些模型对打字错误和敌意文本攻击的弹性要强得多。RETVEC在Apache2许可下可在https://github.com/google-research/retvec.上获得



## **29. Adjustable Robust Reinforcement Learning for Online 3D Bin Packing**

在线3D装箱的可调鲁棒强化学习 cs.LG

Accepted to NeurIPS2023

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04323v1) [paper-pdf](http://arxiv.org/pdf/2310.04323v1)

**Authors**: Yuxin Pan, Yize Chen, Fangzhen Lin

**Abstract**: Designing effective policies for the online 3D bin packing problem (3D-BPP) has been a long-standing challenge, primarily due to the unpredictable nature of incoming box sequences and stringent physical constraints. While current deep reinforcement learning (DRL) methods for online 3D-BPP have shown promising results in optimizing average performance over an underlying box sequence distribution, they often fail in real-world settings where some worst-case scenarios can materialize. Standard robust DRL algorithms tend to overly prioritize optimizing the worst-case performance at the expense of performance under normal problem instance distribution. To address these issues, we first introduce a permutation-based attacker to investigate the practical robustness of both DRL-based and heuristic methods proposed for solving online 3D-BPP. Then, we propose an adjustable robust reinforcement learning (AR2L) framework that allows efficient adjustment of robustness weights to achieve the desired balance of the policy's performance in average and worst-case environments. Specifically, we formulate the objective function as a weighted sum of expected and worst-case returns, and derive the lower performance bound by relating to the return under a mixture dynamics. To realize this lower bound, we adopt an iterative procedure that searches for the associated mixture dynamics and improves the corresponding policy. We integrate this procedure into two popular robust adversarial algorithms to develop the exact and approximate AR2L algorithms. Experiments demonstrate that AR2L is versatile in the sense that it improves policy robustness while maintaining an acceptable level of performance for the nominal case.

摘要: 为在线3D装箱问题(3D-BPP)设计有效的策略一直是一个长期存在的挑战，主要是由于进入箱子序列的不可预测性质和严格的物理约束。虽然目前用于在线3D-BPP的深度强化学习(DRL)方法在优化底层盒序列分布的平均性能方面显示出良好的结果，但它们在现实世界中经常失败，其中一些最坏的情况可能成为现实。标准的稳健DRL算法倾向于以牺牲正态问题实例分布下的性能为代价来优化最坏情况下的性能。为了解决这些问题，我们首先引入了一个基于排列的攻击者来研究基于DRL和启发式方法的在线3D-BPP求解方法的实用稳健性。然后，我们提出了一种可调鲁棒强化学习(AR2L)框架，该框架允许有效地调整健壮性权重以在平均和最坏情况下实现策略性能的期望平衡。具体地说，我们将目标函数表示为预期收益和最坏情况收益的加权和，并通过与混合动态下的收益相关来推导出性能的下界。为了实现这一下界，我们采用了一种迭代过程来搜索相关的混合动力学并改进相应的策略。我们将这一过程集成到两种流行的健壮对抗算法中，以开发精确和近似的AR2L算法。实验表明，AR2L在提高策略健壮性的同时，在名义情况下保持了可接受的性能水平，因此具有很强的通用性。



## **30. Assessing Robustness via Score-Based Adversarial Image Generation**

基于分数的对抗性图像生成方法评估健壮性 cs.CV

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04285v1) [paper-pdf](http://arxiv.org/pdf/2310.04285v1)

**Authors**: Marcel Kollovieh, Lukas Gosch, Yan Scholten, Marten Lienen, Stephan Günnemann

**Abstract**: Most adversarial attacks and defenses focus on perturbations within small $\ell_p$-norm constraints. However, $\ell_p$ threat models cannot capture all relevant semantic-preserving perturbations, and hence, the scope of robustness evaluations is limited. In this work, we introduce Score-Based Adversarial Generation (ScoreAG), a novel framework that leverages the advancements in score-based generative models to generate adversarial examples beyond $\ell_p$-norm constraints, so-called unrestricted adversarial examples, overcoming their limitations. Unlike traditional methods, ScoreAG maintains the core semantics of images while generating realistic adversarial examples, either by transforming existing images or synthesizing new ones entirely from scratch. We further exploit the generative capability of ScoreAG to purify images, empirically enhancing the robustness of classifiers. Our extensive empirical evaluation demonstrates that ScoreAG matches the performance of state-of-the-art attacks and defenses across multiple benchmarks. This work highlights the importance of investigating adversarial examples bounded by semantics rather than $\ell_p$-norm constraints. ScoreAG represents an important step towards more encompassing robustness assessments.

摘要: 大多数对抗性攻击和防御都集中在较小的$\ell_p$-范数约束内的扰动。然而，$\ell_p$威胁模型不能捕获所有相关的语义保持扰动，因此健壮性评估的范围是有限的。在这项工作中，我们引入了基于分数的对抗性生成(ScoreAG)，这是一种新的框架，它利用基于分数的生成模型的进步来生成超越$\ell_p$-范数约束的对抗性实例，即所谓的无限制对抗性实例，克服了它们的局限性。与传统方法不同，ScoreAG在生成真实对抗性示例的同时保持了图像的核心语义，要么转换现有图像，要么完全从头开始合成新的图像。我们进一步利用ScoreAG的生成能力来净化图像，经验上增强了分类器的稳健性。我们广泛的经验评估表明，ScoreAG在多个基准上与最先进的攻击和防御性能相匹配。这项工作强调了研究受语义约束的对抗性例子的重要性，而不是$\ell_p$-范数约束。ScoreAG代表着朝着更全面的稳健性评估迈出的重要一步。



## **31. Threat Trekker: An Approach to Cyber Threat Hunting**

Threat Trekker：一种网络威胁追捕方法 cs.CR

I am disseminating this outcome to all of you, despite the fact that  the results may appear somewhat idealistic, given that certain datasets  utilized for the training of the machine learning model comprise simulated  data

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04197v1) [paper-pdf](http://arxiv.org/pdf/2310.04197v1)

**Authors**: Ángel Casanova Bienzobas, Alfonso Sánchez-Macián

**Abstract**: Threat hunting is a proactive methodology for exploring, detecting and mitigating cyberattacks within complex environments. As opposed to conventional detection systems, threat hunting strategies assume adversaries have infiltrated the system; as a result they proactively search out any unusual patterns or activities which might indicate intrusion attempts.   Historically, this endeavour has been pursued using three investigation methodologies: (1) Hypothesis-Driven Investigations; (2) Indicator of Compromise (IOC); and (3) High-level machine learning analysis-based approaches. Therefore, this paper introduces a novel machine learning paradigm known as Threat Trekker. This proposal utilizes connectors to feed data directly into an event streaming channel for processing by the algorithm and provide feedback back into its host network.   Conclusions drawn from these experiments clearly establish the efficacy of employing machine learning for classifying more subtle attacks.

摘要: 威胁追捕是一种在复杂环境中探索、检测和缓解网络攻击的主动方法。与传统的检测系统不同，威胁追捕策略假定对手已经渗透到系统中；因此，它们主动搜索任何可能表明入侵企图的不寻常模式或活动。从历史上看，这项工作采用了三种调查方法：(1)假设驱动的调查；(2)折衷指标；(3)基于高级机器学习分析的方法。因此，本文引入了一种新的机器学习范式--威胁探险者。该方案利用连接器将数据直接馈送到事件流通道以供算法处理，并将反馈反馈回其主机网络。从这些实验中得出的结论清楚地证明了利用机器学习对更微妙的攻击进行分类的有效性。



## **32. Adversarial Illusions in Multi-Modal Embeddings**

多通道嵌入中的对抗性错觉 cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2308.11804v2) [paper-pdf](http://arxiv.org/pdf/2308.11804v2)

**Authors**: Eugene Bagdasaryan, Rishi Jha, Tingwei Zhang, Vitaly Shmatikov

**Abstract**: Multi-modal embeddings encode images, sounds, texts, videos, etc. into a single embedding space, aligning representations across modalities (e.g., associate an image of a dog with a barking sound). We show that multi-modal embeddings can be vulnerable to an attack we call "adversarial illusions." Given an image or a sound, an adversary can perturb it so as to make its embedding close to an arbitrary, adversary-chosen input in another modality. This enables the adversary to align any image and any sound with any text.   Adversarial illusions exploit proximity in the embedding space and are thus agnostic to downstream tasks. Using ImageBind embeddings, we demonstrate how adversarially aligned inputs, generated without knowledge of specific downstream tasks, mislead image generation, text generation, and zero-shot classification.

摘要: 多模式嵌入将图像、声音、文本、视频等编码到单个嵌入空间中，使跨模式的表示对齐(例如，将狗的图像与犬吠声相关联)。我们表明，多模式嵌入可能容易受到一种我们称为“对抗性错觉”的攻击。在给定图像或声音的情况下，敌手可以对其进行干扰，以便将其嵌入到另一种形式中，接近对手选择的任意输入。这使对手能够将任何图像和任何声音与任何文本对齐。对抗性错觉利用嵌入空间中的接近，因此对下游任务是不可知的。使用ImageBind嵌入，我们演示了在不知道特定下游任务的情况下生成的恶意对齐的输入如何误导图像生成、文本生成和零镜头分类。



## **33. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and Federated LLMs**

FedMLSecurity：联邦学习和联邦LLM中攻击和防御的基准 cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2306.04959v3) [paper-pdf](http://arxiv.org/pdf/2306.04959v3)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Wenxuan Wu, Chulin Xie, Yuhang Yao, Kai Zhang, Qifan Zhang, Yuhui Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: This paper introduces FedMLSecurity, a benchmark designed to simulate adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances FedML's capabilities to evaluate security issues and potential remedies in FL. FedMLSecurity comprises two major components: FedMLAttacker that simulates attacks injected during FL training, and FedMLDefender that simulates defensive mechanisms to mitigate the impacts of the attacks. FedMLSecurity is open-sourced and can be customized to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). FedMLSecurity can also be applied to Large Language Models (LLMs) easily, demonstrating its adaptability and applicability in various scenarios.

摘要: 本文介绍了联邦学习中用于模拟对抗性攻击和相应防御机制的基准测试程序FedMLSecurity。作为促进FL算法开发和性能比较的开源库FedML的一个不可或缺的模块，FedMLSecurity增强了FedML评估FL中的安全问题和潜在补救措施的能力。FedMLSecurity由两个主要组件组成：模拟在FL训练期间注入的攻击的FedMLAttracker和模拟防御机制以减轻攻击影响的FedMLDefender。FedMLSecurity是开源的，可以针对多种机器学习模型(例如Logistic回归、ResNet、GAN等)进行定制。以及联合优化器(例如，FedAVG、FedOPT、FedNOVA等)。FedMLSecurity也可以很容易地应用到大型语言模型(LLM)中，展示了它在各种场景中的适应性和适用性。



## **34. Kick Bad Guys Out! Zero-Knowledge-Proof-Based Anomaly Detection in Federated Learning**

把坏人踢出去！联邦学习中基于零知识证明的异常检测 cs.CR

**SubmitDate**: 2023-10-06    [abs](http://arxiv.org/abs/2310.04055v1) [paper-pdf](http://arxiv.org/pdf/2310.04055v1)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Yuhang Yao, Qifan Zhang, Salman Avestimehr, Chaoyang He

**Abstract**: Federated learning (FL) systems are vulnerable to malicious clients that submit poisoned local models to achieve their adversarial goals, such as preventing the convergence of the global model or inducing the global model to misclassify some data. Many existing defense mechanisms are impractical in real-world FL systems, as they require prior knowledge of the number of malicious clients or rely on re-weighting or modifying submissions. This is because adversaries typically do not announce their intentions before attacking, and re-weighting might change aggregation results even in the absence of attacks. To address these challenges in real FL systems, this paper introduces a cutting-edge anomaly detection approach with the following features: i) Detecting the occurrence of attacks and performing defense operations only when attacks happen; ii) Upon the occurrence of an attack, further detecting the malicious client models and eliminating them without harming the benign ones; iii) Ensuring honest execution of defense mechanisms at the server by leveraging a zero-knowledge proof mechanism. We validate the superior performance of the proposed approach with extensive experiments.

摘要: 联合学习(FL)系统很容易受到恶意客户端的攻击，这些恶意客户端提交有毒的局部模型来实现其敌对目标，如阻止全局模型的收敛或诱导全局模型对某些数据进行错误分类。许多现有的防御机制在现实世界的FL系统中是不切实际的，因为它们需要事先知道恶意客户端的数量，或者依赖于重新加权或修改提交的内容。这是因为对手通常不会在攻击前宣布他们的意图，即使在没有攻击的情况下，重新加权也可能会改变聚合结果。为了应对实际FL系统中的这些挑战，本文提出了一种前沿的异常检测方法，该方法具有以下特点：i)检测攻击的发生，并仅在攻击发生时执行防御操作；ii)攻击发生时，进一步检测恶意客户端模型并在不损害良性客户端模型的情况下将其清除；iii)利用零知识证明机制确保服务器端防御机制的诚实执行。通过大量的实验验证了该方法的优越性能。



## **35. Improving classifier decision boundaries using nearest neighbors**

利用最近邻法改进分类器决策边界 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03927v1) [paper-pdf](http://arxiv.org/pdf/2310.03927v1)

**Authors**: Johannes Schneider

**Abstract**: Neural networks are not learning optimal decision boundaries. We show that decision boundaries are situated in areas of low training data density. They are impacted by few training samples which can easily lead to overfitting. We provide a simple algorithm performing a weighted average of the prediction of a sample and its nearest neighbors' (computed in latent space) leading to a minor favorable outcomes for a variety of important measures for neural networks. In our evaluation, we employ various self-trained and pre-trained convolutional neural networks to show that our approach improves (i) resistance to label noise, (ii) robustness against adversarial attacks, (iii) classification accuracy, and to some degree even (iv) interpretability. While improvements are not necessarily large in all four areas, our approach is conceptually simple, i.e., improvements come without any modification to network architecture, training procedure or dataset. Furthermore, they are in stark contrast to prior works that often require trade-offs among the four objectives or provide valuable, but non-actionable insights.

摘要: 神经网络没有学习最优决策边界。我们表明，决策边界位于训练数据密度较低的区域。它们受训练样本较少的影响，容易导致过拟合。我们提供了一种简单的算法，对样本及其最近邻居的预测进行加权平均(在潜在空间中计算)，从而对神经网络的各种重要指标产生次要的有利结果。在我们的评估中，我们使用了各种自训练和预训练的卷积神经网络来表明我们的方法提高了(I)对标签噪声的抵抗力，(Ii)对对手攻击的健壮性，(Iii)分类精度，甚至在一定程度上(Iv)可解释性。虽然在所有四个领域的改进不一定都很大，但我们的方法在概念上很简单，即在不对网络体系结构、训练过程或数据集进行任何修改的情况下进行改进。此外，它们与以前的工作形成了鲜明的对比，以前的工作往往需要在四个目标之间进行权衡，或者提供有价值但不可操作的见解。



## **36. Preserving Semantics in Textual Adversarial Attacks**

文本对抗性攻击中的语义保护 cs.CL

8 pages, 4 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2211.04205v2) [paper-pdf](http://arxiv.org/pdf/2211.04205v2)

**Authors**: David Herel, Hugo Cisneros, Tomas Mikolov

**Abstract**: The growth of hateful online content, or hate speech, has been associated with a global increase in violent crimes against minorities [23]. Harmful online content can be produced easily, automatically and anonymously. Even though, some form of auto-detection is already achieved through text classifiers in NLP, they can be fooled by adversarial attacks. To strengthen existing systems and stay ahead of attackers, we need better adversarial attacks. In this paper, we show that up to 70% of adversarial examples generated by adversarial attacks should be discarded because they do not preserve semantics. We address this core weakness and propose a new, fully supervised sentence embedding technique called Semantics-Preserving-Encoder (SPE). Our method outperforms existing sentence encoders used in adversarial attacks by achieving 1.2x - 5.1x better real attack success rate. We release our code as a plugin that can be used in any existing adversarial attack to improve its quality and speed up its execution.

摘要: 仇恨在线内容或仇恨言论的增长与全球针对少数群体的暴力犯罪增加有关[23]。有害的在线内容可以很容易地、自动地和匿名地产生。尽管在NLP中已经通过文本分类器实现了某种形式的自动检测，但它们可能会被对手攻击愚弄。为了加强现有系统并保持领先于攻击者，我们需要更好的对抗性攻击。在这篇文章中，我们证明了高达70%的由对抗性攻击产生的对抗性实例应该被丢弃，因为它们不保留语义。针对这一核心缺陷，我们提出了一种新的、完全有监督的句子嵌入技术，称为语义保留编码器(SPE)。我们的方法比现有的对抗性攻击中使用的句子编码器获得了1.2倍-5.1倍的真实攻击成功率。我们将代码作为插件发布，该插件可用于任何现有的恶意攻击，以提高其质量并加快其执行速度。



## **37. OMG-ATTACK: Self-Supervised On-Manifold Generation of Transferable Evasion Attacks**

OMG-Attack：自我监督的流形上的可转移规避攻击 cs.LG

ICCV 2023, AROW Workshop

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03707v1) [paper-pdf](http://arxiv.org/pdf/2310.03707v1)

**Authors**: Ofir Bar Tal, Adi Haviv, Amit H. Bermano

**Abstract**: Evasion Attacks (EA) are used to test the robustness of trained neural networks by distorting input data to misguide the model into incorrect classifications. Creating these attacks is a challenging task, especially with the ever-increasing complexity of models and datasets. In this work, we introduce a self-supervised, computationally economical method for generating adversarial examples, designed for the unseen black-box setting. Adapting techniques from representation learning, our method generates on-manifold EAs that are encouraged to resemble the data distribution. These attacks are comparable in effectiveness compared to the state-of-the-art when attacking the model trained on, but are significantly more effective when attacking unseen models, as the attacks are more related to the data rather than the model itself. Our experiments consistently demonstrate the method is effective across various models, unseen data categories, and even defended models, suggesting a significant role for on-manifold EAs when targeting unseen models.

摘要: 回避攻击(EA)被用来测试训练神经网络的稳健性，方法是扭曲输入数据以将模型误导到错误的分类。创建这些攻击是一项具有挑战性的任务，特别是在模型和数据集的日益复杂的情况下。在这项工作中，我们介绍了一种自监督的、计算上经济的方法来生成对抗性实例，该方法是针对不可见的黑箱设置而设计的。我们的方法采用了表示学习的技术，生成了流形上的EA，鼓励它们类似于数据分布。在攻击受过训练的模型时，这些攻击在有效性上与最先进的攻击相当，但在攻击看不见的模型时，这些攻击的效率要高得多，因为攻击更多地与数据相关，而不是模型本身。我们的实验一直表明，该方法对各种模型、不可见数据类别，甚至是防御模型都是有效的，这表明在针对不可见模型时，流形上的EA具有重要的作用。



## **38. SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks**

SmoothLLM：保护大型语言模型免受越狱攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03684v1) [paper-pdf](http://arxiv.org/pdf/2310.03684v1)

**Authors**: Alexander Robey, Eric Wong, Hamed Hassani, George J. Pappas

**Abstract**: Despite efforts to align large language models (LLMs) with human values, widely-used LLMs such as GPT, Llama, Claude, and PaLM are susceptible to jailbreaking attacks, wherein an adversary fools a targeted LLM into generating objectionable content. To address this vulnerability, we propose SmoothLLM, the first algorithm designed to mitigate jailbreaking attacks on LLMs. Based on our finding that adversarially-generated prompts are brittle to character-level changes, our defense first randomly perturbs multiple copies of a given input prompt, and then aggregates the corresponding predictions to detect adversarial inputs. SmoothLLM reduces the attack success rate on numerous popular LLMs to below one percentage point, avoids unnecessary conservatism, and admits provable guarantees on attack mitigation. Moreover, our defense uses exponentially fewer queries than existing attacks and is compatible with any LLM.

摘要: 尽管努力使大型语言模型(LLM)与人类价值观保持一致，但GPT、Llama、Claude和Palm等广泛使用的LLM容易受到越狱攻击，即对手欺骗目标LLM生成令人反感的内容。为了解决这一漏洞，我们提出了SmoothLLM，这是第一个旨在缓解对LLM的越狱攻击的算法。基于我们的发现，对抗性生成的提示对字符级别的变化很脆弱，我们的防御首先随机扰动给定输入提示的多个副本，然后聚合相应的预测来检测对抗性输入。SmoothLLM将许多流行的LLM的攻击成功率降低到1个百分点以下，避免了不必要的保守主义，并承认了对攻击缓解的可证明保证。此外，我们的防御使用的查询比现有攻击少得多，并且与任何LLM兼容。



## **39. Certification of Deep Learning Models for Medical Image Segmentation**

医学图像分割中深度学习模型的验证 eess.IV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03664v1) [paper-pdf](http://arxiv.org/pdf/2310.03664v1)

**Authors**: Othmane Laousy, Alexandre Araujo, Guillaume Chassagnon, Nikos Paragios, Marie-Pierre Revel, Maria Vakalopoulou

**Abstract**: In medical imaging, segmentation models have known a significant improvement in the past decade and are now used daily in clinical practice. However, similar to classification models, segmentation models are affected by adversarial attacks. In a safety-critical field like healthcare, certifying model predictions is of the utmost importance. Randomized smoothing has been introduced lately and provides a framework to certify models and obtain theoretical guarantees. In this paper, we present for the first time a certified segmentation baseline for medical imaging based on randomized smoothing and diffusion models. Our results show that leveraging the power of denoising diffusion probabilistic models helps us overcome the limits of randomized smoothing. We conduct extensive experiments on five public datasets of chest X-rays, skin lesions, and colonoscopies, and empirically show that we are able to maintain high certified Dice scores even for highly perturbed images. Our work represents the first attempt to certify medical image segmentation models, and we aspire for it to set a foundation for future benchmarks in this crucial and largely uncharted area.

摘要: 在医学成像中，分割模型在过去十年中有了显著的改进，现在每天都在临床实践中使用。然而，与分类模型类似，分割模型也会受到对抗性攻击的影响。在像医疗保健这样的安全关键领域，验证模型预测是至关重要的。随机化平滑是最近引入的，它为验证模型和获得理论保证提供了一个框架。在这篇文章中，我们首次提出了一种基于随机平滑和扩散模型的医学图像分割基线。我们的结果表明，利用扩散概率模型的去噪能力可以帮助我们克服随机平滑的局限性。我们在胸部X光、皮肤损伤和结肠镜检查的五个公共数据集上进行了广泛的实验，经验表明，即使是高度扰动的图像，我们也能够保持高认证Dice分数。我们的工作代表着认证医学图像分割模型的第一次尝试，我们渴望它为这一关键且基本上未知的领域的未来基准奠定基础。



## **40. Targeted Adversarial Attacks on Generalizable Neural Radiance Fields**

可推广神经辐射场上的定向对抗性攻击 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03578v1) [paper-pdf](http://arxiv.org/pdf/2310.03578v1)

**Authors**: Andras Horvath, Csaba M. Jozsa

**Abstract**: Neural Radiance Fields (NeRFs) have recently emerged as a powerful tool for 3D scene representation and rendering. These data-driven models can learn to synthesize high-quality images from sparse 2D observations, enabling realistic and interactive scene reconstructions. However, the growing usage of NeRFs in critical applications such as augmented reality, robotics, and virtual environments could be threatened by adversarial attacks.   In this paper we present how generalizable NeRFs can be attacked by both low-intensity adversarial attacks and adversarial patches, where the later could be robust enough to be used in real world applications. We also demonstrate targeted attacks, where a specific, predefined output scene is generated by these attack with success.

摘要: 神经辐射场(NERF)是最近出现的一种用于3D场景表示和渲染的强大工具。这些数据驱动的模型可以学习从稀疏的2D观测合成高质量的图像，从而实现逼真和交互的场景重建。然而，在增强现实、机器人和虚拟环境等关键应用中越来越多地使用nerf可能会受到对手攻击的威胁。在这篇文章中，我们介绍了可推广的NERF如何同时被低强度的对抗性攻击和对抗性补丁攻击，其中后者可以足够健壮，以用于现实世界的应用。我们还演示了目标攻击，这些攻击成功地生成了特定的、预定义的输出场景。



## **41. Enhancing Adversarial Robustness via Score-Based Optimization**

通过基于分数的优化增强对手的健壮性 cs.LG

NeurIPS 2023

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2307.04333v2) [paper-pdf](http://arxiv.org/pdf/2307.04333v2)

**Authors**: Boya Zhang, Weijian Luo, Zhihua Zhang

**Abstract**: Adversarial attacks have the potential to mislead deep neural network classifiers by introducing slight perturbations. Developing algorithms that can mitigate the effects of these attacks is crucial for ensuring the safe use of artificial intelligence. Recent studies have suggested that score-based diffusion models are effective in adversarial defenses. However, existing diffusion-based defenses rely on the sequential simulation of the reversed stochastic differential equations of diffusion models, which are computationally inefficient and yield suboptimal results. In this paper, we introduce a novel adversarial defense scheme named ScoreOpt, which optimizes adversarial samples at test-time, towards original clean data in the direction guided by score-based priors. We conduct comprehensive experiments on multiple datasets, including CIFAR10, CIFAR100 and ImageNet. Our experimental results demonstrate that our approach outperforms existing adversarial defenses in terms of both robustness performance and inference speed.

摘要: 对抗性攻击有可能通过引入轻微的扰动来误导深度神经网络分类器。开发能够缓解这些攻击影响的算法，对于确保人工智能的安全使用至关重要。最近的研究表明，基于分数的扩散模型在对抗防御中是有效的。然而，现有的基于扩散的防御依赖于对扩散模型的逆随机微分方程的顺序模拟，这在计算上效率低下，并且产生次优结果。在本文中，我们提出了一种新的对抗防御方案ScoreOpt，该方案在测试时优化对手样本，在基于分数的先验的指导下，朝着原始干净数据的方向进行优化。我们在包括CIFAR10、CIFAR100和ImageNet在内的多个数据集上进行了全面的实验。实验结果表明，该方法在稳健性和推理速度上均优于现有的对抗性防御方法。



## **42. AdvRain: Adversarial Raindrops to Attack Camera-based Smart Vision Systems**

AdvRain：敌意雨滴攻击基于摄像头的智能视觉系统 cs.CV

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2303.01338v2) [paper-pdf](http://arxiv.org/pdf/2303.01338v2)

**Authors**: Amira Guesmi, Muhammad Abdullah Hanif, Muhammad Shafique

**Abstract**: Vision-based perception modules are increasingly deployed in many applications, especially autonomous vehicles and intelligent robots. These modules are being used to acquire information about the surroundings and identify obstacles. Hence, accurate detection and classification are essential to reach appropriate decisions and take appropriate and safe actions at all times. Current studies have demonstrated that "printed adversarial attacks", known as physical adversarial attacks, can successfully mislead perception models such as object detectors and image classifiers. However, most of these physical attacks are based on noticeable and eye-catching patterns for generated perturbations making them identifiable/detectable by human eye or in test drives. In this paper, we propose a camera-based inconspicuous adversarial attack (\textbf{AdvRain}) capable of fooling camera-based perception systems over all objects of the same class. Unlike mask based fake-weather attacks that require access to the underlying computing hardware or image memory, our attack is based on emulating the effects of a natural weather condition (i.e., Raindrops) that can be printed on a translucent sticker, which is externally placed over the lens of a camera. To accomplish this, we provide an iterative process based on performing a random search aiming to identify critical positions to make sure that the performed transformation is adversarial for a target classifier. Our transformation is based on blurring predefined parts of the captured image corresponding to the areas covered by the raindrop. We achieve a drop in average model accuracy of more than $45\%$ and $40\%$ on VGG19 for ImageNet and Resnet34 for Caltech-101, respectively, using only $20$ raindrops.

摘要: 基于视觉的感知模块越来越多地部署在许多应用中，特别是自动驾驶汽车和智能机器人。这些模块被用来获取关于周围环境的信息和识别障碍物。因此，准确的检测和分类对于作出适当的决定并始终采取适当和安全的行动至关重要。目前的研究表明，被称为物理对抗性攻击的“印刷对抗性攻击”可以成功地误导对象检测器和图像分类器等感知模型。然而，大多数这些物理攻击都是基于所产生的扰动的明显和醒目的模式，使得它们可以被人眼或试驾识别/检测。在这篇文章中，我们提出了一种基于摄像机的隐蔽敌意攻击(Textbf{AdvRain})，能够在同一类对象上欺骗基于摄像机的感知系统。与需要访问底层计算硬件或图像内存的基于面具的假天气攻击不同，我们的攻击基于模拟自然天气条件(即雨滴)的影响，可以将其打印在半透明贴纸上，该贴纸外部放置在相机的镜头上。为了实现这一点，我们提供了一种迭代过程，该过程基于执行旨在识别关键位置的随机搜索，以确保所执行的变换对目标分类器是对抗性的。我们的变换是基于模糊与雨滴覆盖的区域相对应的捕获图像的预定义部分。在VGG19(ImageNet)和Resnet34(Caltech-101)上，仅使用$20$雨滴，我们的平均模型精度分别下降了$45$和$40$。



## **43. Efficient Biologically Plausible Adversarial Training**

有效的生物学上看似合理的对抗性训练 cs.LG

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2309.17348v3) [paper-pdf](http://arxiv.org/pdf/2309.17348v3)

**Authors**: Matilde Tristany Farinha, Thomas Ortner, Giorgia Dellaferrera, Benjamin Grewe, Angeliki Pantazi

**Abstract**: Artificial Neural Networks (ANNs) trained with Backpropagation (BP) show astounding performance and are increasingly often used in performing our daily life tasks. However, ANNs are highly vulnerable to adversarial attacks, which alter inputs with small targeted perturbations that drastically disrupt the models' performance. The most effective method to make ANNs robust against these attacks is adversarial training, in which the training dataset is augmented with exemplary adversarial samples. Unfortunately, this approach has the drawback of increased training complexity since generating adversarial samples is very computationally demanding. In contrast to ANNs, humans are not susceptible to adversarial attacks. Therefore, in this work, we investigate whether biologically-plausible learning algorithms are more robust against adversarial attacks than BP. In particular, we present an extensive comparative analysis of the adversarial robustness of BP and Present the Error to Perturb the Input To modulate Activity (PEPITA), a recently proposed biologically-plausible learning algorithm, on various computer vision tasks. We observe that PEPITA has higher intrinsic adversarial robustness and, with adversarial training, has a more favourable natural-vs-adversarial performance trade-off as, for the same natural accuracies, PEPITA's adversarial accuracies decrease in average by 0.26% and BP's by 8.05%.

摘要: 用反向传播(BP)训练的人工神经网络(ANN)表现出惊人的性能，越来越多地被用于执行我们的日常生活任务。然而，人工神经网络非常容易受到对抗性攻击，这些攻击通过小的有针对性的扰动改变输入，从而极大地破坏模型的性能。使神经网络对这些攻击具有健壮性的最有效的方法是对抗性训练，其中训练数据集被示例性对抗性样本扩充。不幸的是，这种方法的缺点是增加了训练的复杂性，因为生成对抗性样本的计算要求非常高。与人工神经网络不同，人类不容易受到敌意攻击。因此，在这项工作中，我们调查了生物上可信的学习算法是否比BP算法对对手攻击更健壮。特别是，我们对BP的对抗健壮性进行了广泛的比较分析，并提出了在各种计算机视觉任务中扰动输入以调节活动的误差(PEITA)，这是一种最近提出的生物学上看似合理的学习算法。我们观察到，Pepita具有更高的内在对抗健壮性，并且经过对抗训练后，具有更有利的自然与对抗性能权衡，对于相同的自然精度，Pepita的对抗精度平均下降0.26%，BP的平均下降8.05%。



## **44. An Integrated Algorithm for Robust and Imperceptible Audio Adversarial Examples**

一种稳健不可察觉的音频对抗性实例综合算法 cs.SD

Proc. 3rd Symposium on Security and Privacy in Speech Communication

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03349v1) [paper-pdf](http://arxiv.org/pdf/2310.03349v1)

**Authors**: Armin Ettenhofer, Jan-Philipp Schulze, Karla Pizzi

**Abstract**: Audio adversarial examples are audio files that have been manipulated to fool an automatic speech recognition (ASR) system, while still sounding benign to a human listener. Most methods to generate such samples are based on a two-step algorithm: first, a viable adversarial audio file is produced, then, this is fine-tuned with respect to perceptibility and robustness. In this work, we present an integrated algorithm that uses psychoacoustic models and room impulse responses (RIR) in the generation step. The RIRs are dynamically created by a neural network during the generation process to simulate a physical environment to harden our examples against transformations experienced in over-the-air attacks. We compare the different approaches in three experiments: in a simulated environment and in a realistic over-the-air scenario to evaluate the robustness, and in a human study to evaluate the perceptibility. Our algorithms considering psychoacoustics only or in addition to the robustness show an improvement in the signal-to-noise ratio (SNR) as well as in the human perception study, at the cost of an increased word error rate (WER).

摘要: 音频敌意的例子是被操纵以愚弄自动语音识别(ASR)系统的音频文件，同时对人类收听者来说听起来仍然是良性的。大多数生成此类样本的方法都是基于两步算法：首先生成一个可行的敌意音频文件，然后在可感知性和稳健性方面进行微调。在这项工作中，我们提出了一种在生成步骤中使用心理声学模型和房间脉冲响应(RIR)的综合算法。RIR是由神经网络在生成过程中动态创建的，以模拟物理环境，以强化我们的示例，防止在空中攻击中经历的转换。我们在三个实验中比较了不同的方法：在模拟环境中和在现实的空中场景中评估稳健性，以及在人体研究中评估感知能力。我们的算法只考虑了心理声学，或者除了稳健性之外，在信噪比(SNR)和人类感知研究中都得到了改善，但代价是字错误率(WER)增加。



## **45. Certifiably Robust Graph Contrastive Learning**

可证明稳健的图对比学习 cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03312v1) [paper-pdf](http://arxiv.org/pdf/2310.03312v1)

**Authors**: Minhua Lin, Teng Xiao, Enyan Dai, Xiang Zhang, Suhang Wang

**Abstract**: Graph Contrastive Learning (GCL) has emerged as a popular unsupervised graph representation learning method. However, it has been shown that GCL is vulnerable to adversarial attacks on both the graph structure and node attributes. Although empirical approaches have been proposed to enhance the robustness of GCL, the certifiable robustness of GCL is still remain unexplored. In this paper, we develop the first certifiably robust framework in GCL. Specifically, we first propose a unified criteria to evaluate and certify the robustness of GCL. We then introduce a novel technique, RES (Randomized Edgedrop Smoothing), to ensure certifiable robustness for any GCL model, and this certified robustness can be provably preserved in downstream tasks. Furthermore, an effective training method is proposed for robust GCL. Extensive experiments on real-world datasets demonstrate the effectiveness of our proposed method in providing effective certifiable robustness and enhancing the robustness of any GCL model. The source code of RES is available at https://github.com/ventr1c/RES-GCL.

摘要: 图对比学习(GCL)是一种流行的无监督图表示学习方法。然而，已有研究表明，GCL在图结构和节点属性上都容易受到敌意攻击。虽然已经提出了一些经验方法来增强GCL的稳健性，但GCL的可证明稳健性仍未得到探索。在本文中，我们在GCL中开发了第一个可证明的健壮性框架。具体地说，我们首先提出了一个统一的标准来评价和证明GCL的健壮性。然后，我们引入了一种新的技术，随机Edgedrop平滑技术，以确保对任何GCL模型都具有可证明的稳健性，并且这种经证明的稳健性可以在下游任务中被证明保持。在此基础上，提出了一种有效的鲁棒GCL训练方法。在真实数据集上的大量实验表明，我们提出的方法在提供有效的可证明的稳健性和增强任何GCL模型的稳健性方面是有效的。RES的源代码可在https://github.com/ventr1c/RES-GCL.上找到



## **46. BaDExpert: Extracting Backdoor Functionality for Accurate Backdoor Input Detection**

BaDExpert：提取后门功能以进行准确的后门输入检测 cs.CR

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2308.12439v2) [paper-pdf](http://arxiv.org/pdf/2308.12439v2)

**Authors**: Tinghao Xie, Xiangyu Qi, Ping He, Yiming Li, Jiachen T. Wang, Prateek Mittal

**Abstract**: We present a novel defense, against backdoor attacks on Deep Neural Networks (DNNs), wherein adversaries covertly implant malicious behaviors (backdoors) into DNNs. Our defense falls within the category of post-development defenses that operate independently of how the model was generated. The proposed defense is built upon a novel reverse engineering approach that can directly extract backdoor functionality of a given backdoored model to a backdoor expert model. The approach is straightforward -- finetuning the backdoored model over a small set of intentionally mislabeled clean samples, such that it unlearns the normal functionality while still preserving the backdoor functionality, and thus resulting in a model (dubbed a backdoor expert model) that can only recognize backdoor inputs. Based on the extracted backdoor expert model, we show the feasibility of devising highly accurate backdoor input detectors that filter out the backdoor inputs during model inference. Further augmented by an ensemble strategy with a finetuned auxiliary model, our defense, BaDExpert (Backdoor Input Detection with Backdoor Expert), effectively mitigates 17 SOTA backdoor attacks while minimally impacting clean utility. The effectiveness of BaDExpert has been verified on multiple datasets (CIFAR10, GTSRB and ImageNet) across various model architectures (ResNet, VGG, MobileNetV2 and Vision Transformer).

摘要: 针对深度神经网络(DNNS)的后门攻击，提出了一种新的防御方法，即攻击者秘密地在DNN中植入恶意行为(后门)。我们的防御属于开发后防御的范畴，其运作独立于模型是如何生成的。建议的防御建立在一种新的逆向工程方法之上，该方法可以直接将给定后门模型的后门功能提取到后门专家模型中。这种方法很简单--在一小部分故意错误标记的干净样本上优化后门模型，以便它在保留后门功能的同时取消学习正常功能，从而产生只能识别后门输入的模型(称为后门专家模型)。基于提取的后门专家模型，我们证明了设计高精度的后门输入检测器的可行性，该检测器在模型推理过程中过滤掉后门输入。我们的防御系统BaDExpert(带有后门专家的后门输入检测)通过精细的辅助模型进一步增强了整体策略，有效地减少了17次SOTA后门攻击，同时将对清洁实用程序的影响降至最低。BaDExpert的有效性已经在各种模型架构(ResNet、VGG、MobileNetV2和Vision Transformer)的多个数据集(CIFAR10、GTSRB和ImageNet)上得到了验证。



## **47. Burning the Adversarial Bridges: Robust Windows Malware Detection Against Binary-level Mutations**

烧毁敌意桥梁：针对二进制级突变的强大Windows恶意软件检测 cs.LG

12 pages

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2310.03285v1) [paper-pdf](http://arxiv.org/pdf/2310.03285v1)

**Authors**: Ahmed Abusnaina, Yizhen Wang, Sunpreet Arora, Ke Wang, Mihai Christodorescu, David Mohaisen

**Abstract**: Toward robust malware detection, we explore the attack surface of existing malware detection systems. We conduct root-cause analyses of the practical binary-level black-box adversarial malware examples. Additionally, we uncover the sensitivity of volatile features within the detection engines and exhibit their exploitability. Highlighting volatile information channels within the software, we introduce three software pre-processing steps to eliminate the attack surface, namely, padding removal, software stripping, and inter-section information resetting. Further, to counter the emerging section injection attacks, we propose a graph-based section-dependent information extraction scheme for software representation. The proposed scheme leverages aggregated information within various sections in the software to enable robust malware detection and mitigate adversarial settings. Our experimental results show that traditional malware detection models are ineffective against adversarial threats. However, the attack surface can be largely reduced by eliminating the volatile information. Therefore, we propose simple-yet-effective methods to mitigate the impacts of binary manipulation attacks. Overall, our graph-based malware detection scheme can accurately detect malware with an area under the curve score of 88.32\% and a score of 88.19% under a combination of binary manipulation attacks, exhibiting the efficiency of our proposed scheme.

摘要: 对于健壮的恶意软件检测，我们探索了现有恶意软件检测系统的攻击面。我们对实际的二进制级黑盒恶意软件实例进行了根本原因分析。此外，我们还揭示了检测引擎中易失性特征的敏感性，并展示了它们的可利用性。突出软件内部易变的信息通道，介绍了消除攻击面的三个软件预处理步骤，即填充去除、软件剥离和区间信息重置。此外，针对目前出现的段注入攻击，我们提出了一种基于图的段依赖信息提取方法。建议的方案利用软件中各个部分的聚合信息来实现健壮的恶意软件检测和缓解敌意设置。我们的实验结果表明，传统的恶意软件检测模型不能有效地对抗恶意威胁。然而，通过消除不稳定的信息，可以大大减少攻击面。因此，我们提出了简单而有效的方法来缓解二进制操纵攻击的影响。总体而言，基于图的恶意软件检测方案能够准确地检测出恶意软件，在二进制操纵攻击的组合下，曲线下面积得分为88.32%，得分为88.19%，表明了该方案的有效性。



## **48. Network Cascade Vulnerability using Constrained Bayesian Optimization**

基于约束贝叶斯优化的网络级联漏洞 cs.SI

13 pages, 5 figures

**SubmitDate**: 2023-10-05    [abs](http://arxiv.org/abs/2304.14420v2) [paper-pdf](http://arxiv.org/pdf/2304.14420v2)

**Authors**: Albert Lam, Mihai Anitescu, Anirudh Subramanyam

**Abstract**: Measures of power grid vulnerability are often assessed by the amount of damage an adversary can exact on the network. However, the cascading impact of such attacks is often overlooked, even though cascades are one of the primary causes of large-scale blackouts. This paper explores modifications of transmission line protection settings as candidates for adversarial attacks, which can remain undetectable as long as the network equilibrium state remains unaltered. This forms the basis of a black-box function in a Bayesian optimization procedure, where the objective is to find protection settings that maximize network degradation due to cascading. Notably, our proposed method is agnostic to the choice of the cascade simulator and its underlying assumptions. Numerical experiments reveal that, against conventional wisdom, maximally misconfiguring the protection settings of all network lines does not cause the most cascading. More surprisingly, even when the degree of misconfiguration is limited due to resource constraints, it is still possible to find settings that produce cascades comparable in severity to instances where there are no resource constraints.

摘要: 电网脆弱性的衡量标准通常是根据对手对网络造成的破坏程度来评估的。然而，这类攻击的连锁影响往往被忽视，尽管连锁是大规模停电的主要原因之一。本文探讨了输电线路保护设置的修改作为对抗性攻击的候选对象，只要网络平衡状态保持不变，这种攻击就可以保持不可检测。这形成了贝叶斯优化过程中的黑盒函数的基础，其中的目标是找到使由于级联而导致的网络降级最大化的保护设置。值得注意的是，我们提出的方法与叶栅模拟器的选择及其基本假设无关。数值实验表明，与传统观点相反，最大限度地错误配置所有网络线路的保护设置并不会导致最大程度的级联。更令人惊讶的是，即使错误配置的程度因资源限制而受到限制，仍有可能找到在严重程度上与没有资源限制的情况相媲美的设置。



## **49. LinGCN: Structural Linearized Graph Convolutional Network for Homomorphically Encrypted Inference**

LinGCN：同态加密推理的结构线性化图卷积网络 cs.LG

NeurIPS 2023 accepted publication

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2309.14331v3) [paper-pdf](http://arxiv.org/pdf/2309.14331v3)

**Authors**: Hongwu Peng, Ran Ran, Yukui Luo, Jiahui Zhao, Shaoyi Huang, Kiran Thorat, Tong Geng, Chenghong Wang, Xiaolin Xu, Wujie Wen, Caiwen Ding

**Abstract**: The growth of Graph Convolution Network (GCN) model sizes has revolutionized numerous applications, surpassing human performance in areas such as personal healthcare and financial systems. The deployment of GCNs in the cloud raises privacy concerns due to potential adversarial attacks on client data. To address security concerns, Privacy-Preserving Machine Learning (PPML) using Homomorphic Encryption (HE) secures sensitive client data. However, it introduces substantial computational overhead in practical applications. To tackle those challenges, we present LinGCN, a framework designed to reduce multiplication depth and optimize the performance of HE based GCN inference. LinGCN is structured around three key elements: (1) A differentiable structural linearization algorithm, complemented by a parameterized discrete indicator function, co-trained with model weights to meet the optimization goal. This strategy promotes fine-grained node-level non-linear location selection, resulting in a model with minimized multiplication depth. (2) A compact node-wise polynomial replacement policy with a second-order trainable activation function, steered towards superior convergence by a two-level distillation approach from an all-ReLU based teacher model. (3) an enhanced HE solution that enables finer-grained operator fusion for node-wise activation functions, further reducing multiplication level consumption in HE-based inference. Our experiments on the NTU-XVIEW skeleton joint dataset reveal that LinGCN excels in latency, accuracy, and scalability for homomorphically encrypted inference, outperforming solutions such as CryptoGCN. Remarkably, LinGCN achieves a 14.2x latency speedup relative to CryptoGCN, while preserving an inference accuracy of 75% and notably reducing multiplication depth.

摘要: 图形卷积网络(GCN)模型大小的增长彻底改变了许多应用程序，在个人医疗保健和金融系统等领域超过了人类的表现。由于对客户数据的潜在敌意攻击，GCNS在云中的部署引发了隐私问题。为了解决安全问题，使用同态加密(HE)的隐私保护机器学习(PPML)保护敏感的客户端数据。然而，它在实际应用中引入了大量的计算开销。为了应对这些挑战，我们提出了LinGCN框架，该框架旨在减少乘法深度并优化基于HE的GCN推理的性能。LinGCN围绕三个关键元素构建：(1)可微结构线性化算法，辅以参数化的离散指标函数，与模型权重共同训练以满足优化目标。该策略促进了细粒度的节点级非线性位置选择，使得模型的乘法深度最小化。(2)一种具有二阶可训练激活函数的紧凑节点多项式替换策略，该策略通过基于全REU的教师模型的两级蒸馏方法引导到优越的收敛。(3)增强的HE解决方案，支持针对节点激活函数的更细粒度的算子融合，进一步减少基于HE的推理中的乘法级别消耗。我们在NTU-XVIEW骨骼关节数据集上的实验表明，LinGCN在同态加密推理的延迟、准确性和可扩展性方面都优于CryptoGCN等解决方案。值得注意的是，与CryptoGCN相比，LinGCN实现了14.2倍的延迟加速，同时保持了75%的推理准确率，并显著减少了乘法深度。



## **50. Misusing Tools in Large Language Models With Visual Adversarial Examples**

大型语言模型中的误用工具与视觉对抗性例子 cs.CR

**SubmitDate**: 2023-10-04    [abs](http://arxiv.org/abs/2310.03185v1) [paper-pdf](http://arxiv.org/pdf/2310.03185v1)

**Authors**: Xiaohan Fu, Zihan Wang, Shuheng Li, Rajesh K. Gupta, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Earlence Fernandes

**Abstract**: Large Language Models (LLMs) are being enhanced with the ability to use tools and to process multiple modalities. These new capabilities bring new benefits and also new security risks. In this work, we show that an attacker can use visual adversarial examples to cause attacker-desired tool usage. For example, the attacker could cause a victim LLM to delete calendar events, leak private conversations and book hotels. Different from prior work, our attacks can affect the confidentiality and integrity of user resources connected to the LLM while being stealthy and generalizable to multiple input prompts. We construct these attacks using gradient-based adversarial training and characterize performance along multiple dimensions. We find that our adversarial images can manipulate the LLM to invoke tools following real-world syntax almost always (~98%) while maintaining high similarity to clean images (~0.9 SSIM). Furthermore, using human scoring and automated metrics, we find that the attacks do not noticeably affect the conversation (and its semantics) between the user and the LLM.

摘要: 大型语言模型(LLM)正在得到增强，具有使用工具和处理多种模式的能力。这些新功能带来了新的好处，但也带来了新的安全风险。在这项工作中，我们展示了攻击者可以使用可视化的对抗性示例来导致攻击者所需的工具使用。例如，攻击者可能会导致受害者LLM删除日历事件、泄露私人对话并预订酒店。与以前的工作不同，我们的攻击可以影响连接到LLM的用户资源的机密性和完整性，同时具有隐蔽性和对多个输入提示的通用性。我们使用基于梯度的对抗性训练来构建这些攻击，并在多个维度上表征性能。我们发现，我们的敌意图像可以操纵LLM调用遵循真实语法的工具(~98%)，同时保持与干净图像的高度相似(~0.9SSIM)。此外，使用人工评分和自动度量，我们发现攻击没有显著影响用户和LLM之间的对话(及其语义)。



