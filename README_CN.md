# Latest Adversarial Attack Papers
**update at 2022-06-03 06:31:44**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Attacks on Gaussian Process Bandits**

对高斯过程环的对抗性攻击 stat.ML

Accepted to ICML 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2110.08449v2)

**Authors**: Eric Han, Jonathan Scarlett

**Abstracts**: Gaussian processes (GP) are a widely-adopted tool used to sequentially optimize black-box functions, where evaluations are costly and potentially noisy. Recent works on GP bandits have proposed to move beyond random noise and devise algorithms robust to adversarial attacks. This paper studies this problem from the attacker's perspective, proposing various adversarial attack methods with differing assumptions on the attacker's strength and prior information. Our goal is to understand adversarial attacks on GP bandits from theoretical and practical perspectives. We focus primarily on targeted attacks on the popular GP-UCB algorithm and a related elimination-based algorithm, based on adversarially perturbing the function $f$ to produce another function $\tilde{f}$ whose optima are in some target region $\mathcal{R}_{\rm target}$. Based on our theoretical analysis, we devise both white-box attacks (known $f$) and black-box attacks (unknown $f$), with the former including a Subtraction attack and Clipping attack, and the latter including an Aggressive subtraction attack. We demonstrate that adversarial attacks on GP bandits can succeed in forcing the algorithm towards $\mathcal{R}_{\rm target}$ even with a low attack budget, and we test our attacks' effectiveness on a diverse range of objective functions.

摘要: 高斯过程(GP)是一种广泛采用的工具，用于顺序优化黑盒函数，其中评估成本较高，并且可能存在噪声。最近关于GP盗贼的研究已经提出超越随机噪声，设计出对对手攻击强大的算法。本文从攻击者的角度研究这一问题，提出了各种对抗性攻击方法，并对攻击者的强度和先验信息进行了不同的假设。我们的目标是从理论和实践的角度来理解对GP土匪的敌意攻击。我们主要关注对流行的GP-UCB算法和相关的基于消元的算法的定向攻击，该算法基于对函数$f$的恶意扰动来产生另一个函数$\tide{f}$，其最优值位于某个目标区域$\数学{R}_{\rm目标}$。在理论分析的基础上，我们设计了白盒攻击(已知$f$)和黑盒攻击(未知$f$)，前者包括减法攻击和剪裁攻击，后者包括侵略性减法攻击。我们证明了对GP盗贼的敌意攻击即使在较低的攻击预算下也能成功地迫使算法向数学上的{R}_{\Rm目标}$逼近，并在不同的目标函数上测试了我们的攻击的有效性。



## **2. The robust way to stack and bag: the local Lipschitz way**

稳健的堆叠和打包方式：当地的利普希茨方式 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00513v1)

**Authors**: Thulasi Tholeti, Sheetal Kalyani

**Abstracts**: Recent research has established that the local Lipschitz constant of a neural network directly influences its adversarial robustness. We exploit this relationship to construct an ensemble of neural networks which not only improves the accuracy, but also provides increased adversarial robustness. The local Lipschitz constants for two different ensemble methods - bagging and stacking - are derived and the architectures best suited for ensuring adversarial robustness are deduced. The proposed ensemble architectures are tested on MNIST and CIFAR-10 datasets in the presence of white-box attacks, FGSM and PGD. The proposed architecture is found to be more robust than a) a single network and b) traditional ensemble methods.

摘要: 最近的研究表明，神经网络的局部Lipschitz常数直接影响其对抗鲁棒性。我们利用这种关系来构造神经网络集成，这不仅提高了精度，而且增加了对手的稳健性。推导了两种不同的集成方法--袋装和堆叠--的局部Lipschitz常数，并推导出最适合于确保对抗性稳健性的结构。在MNIST和CIFAR-10数据集上，在白盒攻击、FGSM和PGD的情况下对所提出的集成架构进行了测试。研究发现，该体系结构比a)单一网络和b)传统集成方法更健壮。



## **3. Attack-Agnostic Adversarial Detection**

攻击不可知的敌意检测 cs.CV

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00489v1)

**Authors**: Jiaxin Cheng, Mohamed Hussein, Jay Billa, Wael AbdAlmageed

**Abstracts**: The growing number of adversarial attacks in recent years gives attackers an advantage over defenders, as defenders must train detectors after knowing the types of attacks, and many models need to be maintained to ensure good performance in detecting any upcoming attacks. We propose a way to end the tug-of-war between attackers and defenders by treating adversarial attack detection as an anomaly detection problem so that the detector is agnostic to the attack. We quantify the statistical deviation caused by adversarial perturbations in two aspects. The Least Significant Component Feature (LSCF) quantifies the deviation of adversarial examples from the statistics of benign samples and Hessian Feature (HF) reflects how adversarial examples distort the landscape of the model's optima by measuring the local loss curvature. Empirical results show that our method can achieve an overall ROC AUC of 94.9%, 89.7%, and 94.6% on CIFAR10, CIFAR100, and SVHN, respectively, and has comparable performance to adversarial detectors trained with adversarial examples on most of the attacks.

摘要: 近年来，越来越多的对抗性攻击使攻击者相对于防御者具有优势，因为防御者必须在知道攻击类型后培训检测器，并且需要维护许多模型，以确保在检测任何即将到来的攻击时具有良好的性能。我们提出了一种结束攻击者和防御者之间的拉锯战的方法，将对抗性攻击检测视为一个异常检测问题，使得检测器对攻击是不可知的。我们从两个方面对对抗性扰动造成的统计偏差进行了量化。最低有效成分特征(LSCF)量化了对抗性样本与良性样本统计的偏差，而海森特征(HF)则通过测量局部损失曲率来反映对抗性样本如何扭曲模型的最优解。实验结果表明，我们的方法在CIFAR10、CIFAR100和SVHN上的总体ROC AUC分别达到94.9%、89.7%和94.6%，并且在大多数攻击上具有与使用对抗性实例训练的对抗性检测器相当的性能。



## **4. Generating End-to-End Adversarial Examples for Malware Classifiers Using Explainability**

使用可解释性为恶意软件分类器生成端到端对抗性示例 cs.CR

Accepted as a conference paper at IJCNN 2020

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2009.13243v2)

**Authors**: Ishai Rosenberg, Shai Meir, Jonathan Berrebi, Ilay Gordon, Guillaume Sicard, Eli David

**Abstracts**: In recent years, the topic of explainable machine learning (ML) has been extensively researched. Up until now, this research focused on regular ML users use-cases such as debugging a ML model. This paper takes a different posture and show that adversaries can leverage explainable ML to bypass multi-feature types malware classifiers. Previous adversarial attacks against such classifiers only add new features and not modify existing ones to avoid harming the modified malware executable's functionality. Current attacks use a single algorithm that both selects which features to modify and modifies them blindly, treating all features the same. In this paper, we present a different approach. We split the adversarial example generation task into two parts: First we find the importance of all features for a specific sample using explainability algorithms, and then we conduct a feature-specific modification, feature-by-feature. In order to apply our attack in black-box scenarios, we introduce the concept of transferability of explainability, that is, applying explainability algorithms to different classifiers using different features subsets and trained on different datasets still result in a similar subset of important features. We conclude that explainability algorithms can be leveraged by adversaries and thus the advocates of training more interpretable classifiers should consider the trade-off of higher vulnerability of those classifiers to adversarial attacks.

摘要: 近年来，可解释机器学习得到了广泛的研究。到目前为止，这项研究主要针对常规的ML用户用例，比如调试一个ML模型。本文采取了一种不同的姿态，并展示了攻击者可以利用可解释的ML绕过多特征类型的恶意软件分类器。以前针对此类分类器的敌意攻击只添加新功能，而不修改现有功能，以避免损害修改后的恶意软件可执行文件的功能。当前的攻击使用单一的算法，既选择要修改的特征，又盲目地修改它们，对所有特征一视同仁。在本文中，我们提出了一种不同的方法。我们将对抗性示例生成任务分为两部分：首先使用可解释性算法找出特定样本中所有特征的重要性，然后逐个特征地进行特定特征的修改。为了将我们的攻击应用到黑盒场景中，我们引入了可解释性的概念，即使用不同的特征子集对不同的分类器应用可解释性算法，并在不同的数据集上进行训练，仍然会产生相似的重要特征子集。我们的结论是，可解释性算法可以被对手利用，因此训练更多可解释分类器的倡导者应该考虑这些分类器对对手攻击的更高脆弱性的权衡。



## **5. Anti-Forgery: Towards a Stealthy and Robust DeepFake Disruption Attack via Adversarial Perceptual-aware Perturbations**

防伪：通过对抗性感知扰动实现隐形且强大的DeepFake中断攻击 cs.CR

Accepted by IJCAI 2022

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00477v1)

**Authors**: Run Wang, Ziheng Huang, Zhikai Chen, Li Liu, Jing Chen, Lina Wang

**Abstracts**: DeepFake is becoming a real risk to society and brings potential threats to both individual privacy and political security due to the DeepFaked multimedia are realistic and convincing. However, the popular DeepFake passive detection is an ex-post forensics countermeasure and failed in blocking the disinformation spreading in advance. To address this limitation, researchers study the proactive defense techniques by adding adversarial noises into the source data to disrupt the DeepFake manipulation. However, the existing studies on proactive DeepFake defense via injecting adversarial noises are not robust, which could be easily bypassed by employing simple image reconstruction revealed in a recent study MagDR.   In this paper, we investigate the vulnerability of the existing forgery techniques and propose a novel \emph{anti-forgery} technique that helps users protect the shared facial images from attackers who are capable of applying the popular forgery techniques. Our proposed method generates perceptual-aware perturbations in an incessant manner which is vastly different from the prior studies by adding adversarial noises that is sparse. Experimental results reveal that our perceptual-aware perturbations are robust to diverse image transformations, especially the competitive evasion technique, MagDR via image reconstruction. Our findings potentially open up a new research direction towards thorough understanding and investigation of perceptual-aware adversarial attack for protecting facial images against DeepFakes in a proactive and robust manner. We open-source our tool to foster future research. Code is available at https://github.com/AbstractTeen/AntiForgery/.

摘要: DeepFake正在成为一个真正的社会风险，并给个人隐私和政治安全带来潜在的威胁，因为DeepFak的多媒体是真实和令人信服的。然而，流行的DeepFake被动探测是一种事后取证对策，未能提前阻止虚假信息的传播。为了解决这一局限性，研究人员研究了主动防御技术，通过在源数据中添加对抗性噪声来破坏DeepFake的操纵。然而，现有的通过注入对抗性噪声的主动DeepFake防御的研究并不稳健，这可以通过最近的一项研究MagDR揭示的简单图像重建来容易地绕过。本文研究了现有伪造技术的脆弱性，并提出了一种新的防伪技术，帮助用户保护共享的人脸图像免受攻击者的攻击，攻击者能够应用流行的伪造技术。我们提出的方法以一种不间断的方式产生感知扰动，这与以往的研究通过添加稀疏的对抗性噪声而有很大不同。实验结果表明，我们的感知扰动对不同的图像变换，特别是竞争规避技术，即通过图像重建的MagDR具有很强的鲁棒性。我们的发现可能为深入理解和研究感知感知的敌意攻击以主动和稳健的方式保护面部图像免受DeepFake攻击开辟了新的研究方向。我们将我们的工具开源，以促进未来的研究。代码可在https://github.com/AbstractTeen/AntiForgery/.上找到



## **6. PerDoor: Persistent Non-Uniform Backdoors in Federated Learning using Adversarial Perturbations**

PerDoor：使用对抗性扰动的联合学习中持久的非一致后门 cs.CR

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2205.13523v2)

**Authors**: Manaar Alam, Esha Sarkar, Michail Maniatakos

**Abstracts**: Federated Learning (FL) enables numerous participants to train deep learning models collaboratively without exposing their personal, potentially sensitive data, making it a promising solution for data privacy in collaborative training. The distributed nature of FL and unvetted data, however, makes it inherently vulnerable to backdoor attacks: In this scenario, an adversary injects backdoor functionality into the centralized model during training, which can be triggered to cause the desired misclassification for a specific adversary-chosen input. A range of prior work establishes successful backdoor injection in an FL system; however, these backdoors are not demonstrated to be long-lasting. The backdoor functionality does not remain in the system if the adversary is removed from the training process since the centralized model parameters continuously mutate during successive FL training rounds. Therefore, in this work, we propose PerDoor, a persistent-by-construction backdoor injection technique for FL, driven by adversarial perturbation and targeting parameters of the centralized model that deviate less in successive FL rounds and contribute the least to the main task accuracy. An exhaustive evaluation considering an image classification scenario portrays on average $10.5\times$ persistence over multiple FL rounds compared to traditional backdoor attacks. Through experiments, we further exhibit the potency of PerDoor in the presence of state-of-the-art backdoor prevention techniques in an FL system. Additionally, the operation of adversarial perturbation also assists PerDoor in developing non-uniform trigger patterns for backdoor inputs compared to uniform triggers (with fixed patterns and locations) of existing backdoor techniques, which are prone to be easily mitigated.

摘要: 联合学习(FL)使众多参与者能够协作地训练深度学习模型，而不会暴露他们的个人、潜在敏感数据，使其成为协作培训中数据隐私的一种有前途的解决方案。然而，FL和未经审查的数据的分布式性质使其天生就容易受到后门攻击：在这种情况下，对手在训练期间向集中式模型注入后门功能，这可能会被触发，导致对特定对手选择的输入造成所需的错误分类。先前的一系列工作在FL系统中建立了成功的后门注入；然而，这些后门并没有被证明是持久的。如果将对手从训练过程中移除，则后门功能不会保留在系统中，因为集中式模型参数在连续的FL训练轮期间不断变化。因此，在这项工作中，我们提出了PerDoor，这是一种持久的构造后门注入技术，受对手扰动和集中式模型的目标参数的驱动，这些参数在连续的FL轮中偏离较小，对主任务精度的贡献最小。与传统的后门攻击相比，考虑图像分类场景的详尽评估描绘了在多个FL轮上平均花费10.5\x$持久性。通过实验，我们进一步展示了PerDoor在FL系统中存在最先进的后门预防技术时的有效性。此外，对抗性扰动的操作还有助于PerDoor为后门输入开发非统一的触发模式，而不是现有后门技术的统一触发(具有固定的模式和位置)，后者容易被缓解。



## **7. NeuroUnlock: Unlocking the Architecture of Obfuscated Deep Neural Networks**

NeuroUnlock：解锁模糊深度神经网络的体系结构 cs.CR

The definitive Version of Record will be Published in the 2022  International Joint Conference on Neural Networks (IJCNN)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00402v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Alessio Colucci, Ozgur Sinanoglu, Muhammad Shafique

**Abstracts**: The advancements of deep neural networks (DNNs) have led to their deployment in diverse settings, including safety and security-critical applications. As a result, the characteristics of these models have become sensitive intellectual properties that require protection from malicious users. Extracting the architecture of a DNN through leaky side-channels (e.g., memory access) allows adversaries to (i) clone the model, and (ii) craft adversarial attacks. DNN obfuscation thwarts side-channel-based architecture stealing (SCAS) attacks by altering the run-time traces of a given DNN while preserving its functionality. In this work, we expose the vulnerability of state-of-the-art DNN obfuscation methods to these attacks. We present NeuroUnlock, a novel SCAS attack against obfuscated DNNs. Our NeuroUnlock employs a sequence-to-sequence model that learns the obfuscation procedure and automatically reverts it, thereby recovering the original DNN architecture. We demonstrate the effectiveness of NeuroUnlock by recovering the architecture of 200 randomly generated and obfuscated DNNs running on the Nvidia RTX 2080 TI graphics processing unit (GPU). Moreover, NeuroUnlock recovers the architecture of various other obfuscated DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. After recovering the architecture, NeuroUnlock automatically builds a near-equivalent DNN with only a 1.4% drop in the testing accuracy. We further show that launching a subsequent adversarial attack on the recovered DNNs boosts the success rate of the adversarial attack by 51.7% in average compared to launching it on the obfuscated versions. Additionally, we propose a novel methodology for DNN obfuscation, ReDLock, which eradicates the deterministic nature of the obfuscation and achieves 2.16X more resilience to the NeuroUnlock attack. We release the NeuroUnlock and the ReDLock as open-source frameworks.

摘要: 深度神经网络(DNN)的进步导致它们在不同的环境中部署，包括安全和安全关键应用。因此，这些模型的特征已成为敏感的知识产权，需要保护其免受恶意用户的攻击。通过泄漏的旁路(例如，存储器访问)提取DNN的体系结构允许攻击者(I)克隆模型和(Ii)精心设计敌意攻击。DNN混淆通过改变给定DNN的运行时踪迹，同时保持其功能，从而阻止基于侧通道的体系结构窃取(SCAS)攻击。在这项工作中，我们暴露了最先进的DNN混淆方法对这些攻击的脆弱性。提出了一种新的针对混淆DNN的SCAS攻击--NeuroUnlock。我们的NeuroUnlock采用了序列到序列的模型，该模型学习混淆过程并自动恢复它，从而恢复原始的DNN架构。我们通过恢复在NVIDIA RTX 2080 TI图形处理单元(GPU)上运行的200个随机生成和混淆的DNN的架构来演示NeuroUnlock的有效性。此外，NeuroUnlock恢复了各种其他模糊DNN的架构，如VGG-11、VGG-13、ResNet-20和ResNet-32网络。在恢复架构后，NeuroUnlock会自动构建一个近乎相同的DNN，而测试精度只会下降1.4%。我们进一步表明，与在混淆版本上发起攻击相比，对恢复的DNN发起后续敌意攻击的成功率平均提高了51.7%。此外，我们还提出了一种新的DNN混淆方法ReDLock，它消除了混淆的确定性，并获得了2.16倍的抗NeuroUnlock攻击的能力。我们将NeuroUnlock和ReDLock作为开源框架发布。



## **8. Support Vector Machines under Adversarial Label Contamination**

对抗性标签污染下的支持向量机 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2206.00352v1)

**Authors**: Huang Xiao, Battista Biggio, Blaine Nelson, Han Xiao, Claudia Eckert, Fabio Roli

**Abstracts**: Machine learning algorithms are increasingly being applied in security-related tasks such as spam and malware detection, although their security properties against deliberate attacks have not yet been widely understood. Intelligent and adaptive attackers may indeed exploit specific vulnerabilities exposed by machine learning techniques to violate system security. Being robust to adversarial data manipulation is thus an important, additional requirement for machine learning algorithms to successfully operate in adversarial settings. In this work, we evaluate the security of Support Vector Machines (SVMs) to well-crafted, adversarial label noise attacks. In particular, we consider an attacker that aims to maximize the SVM's classification error by flipping a number of labels in the training data. We formalize a corresponding optimal attack strategy, and solve it by means of heuristic approaches to keep the computational complexity tractable. We report an extensive experimental analysis on the effectiveness of the considered attacks against linear and non-linear SVMs, both on synthetic and real-world datasets. We finally argue that our approach can also provide useful insights for developing more secure SVM learning algorithms, and also novel techniques in a number of related research areas, such as semi-supervised and active learning.

摘要: 机器学习算法正越来越多地应用于垃圾邮件和恶意软件检测等与安全相关的任务中，尽管它们针对故意攻击的安全特性尚未得到广泛了解。智能和适应性攻击者确实可能利用机器学习技术暴露的特定漏洞来破坏系统安全。因此，对对抗性数据操纵具有健壮性是机器学习算法在对抗性环境中成功运行的一个重要的额外要求。在这项工作中，我们评估了支持向量机(SVMs)对精心设计的对抗性标签噪声攻击的安全性。特别是，我们考虑了一个攻击者，他的目标是通过翻转训练数据中的多个标签来最大化支持向量机的分类错误。我们形式化了相应的最优攻击策略，并利用启发式方法进行求解，以保持计算的复杂性。我们对所考虑的针对线性和非线性支持向量机的攻击的有效性进行了广泛的实验分析，包括在合成数据集和真实数据集上的攻击。最后，我们认为，我们的方法还可以为开发更安全的支持向量机学习算法提供有用的见解，并为半监督和主动学习等相关研究领域提供新的技术。



## **9. A Simple Structure For Building A Robust Model**

一种用于建立稳健模型的简单结构 cs.CV

Accepted by Fifth International Conference on Intelligence Science  (ICIS2022); 10 pages, 3 figures, 4 tables

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.11596v2)

**Authors**: Xiao Tan, Jingbo Gao, Ruolin Li

**Abstracts**: As deep learning applications, especially programs of computer vision, are increasingly deployed in our lives, we have to think more urgently about the security of these applications.One effective way to improve the security of deep learning models is to perform adversarial training, which allows the model to be compatible with samples that are deliberately created for use in attacking the model.Based on this, we propose a simple architecture to build a model with a certain degree of robustness, which improves the robustness of the trained network by adding an adversarial sample detection network for cooperative training. At the same time, we design a new data sampling strategy that incorporates multiple existing attacks, allowing the model to adapt to many different adversarial attacks with a single training.We conducted some experiments to test the effectiveness of this design based on Cifar10 dataset, and the results indicate that it has some degree of positive effect on the robustness of the model.Our code could be found at https://github.com/dowdyboy/simple_structure_for_robust_model .

摘要: 随着深度学习应用，特别是计算机视觉应用的日益广泛，我们不得不更加迫切地考虑这些应用的安全性。对抗性训练是提高深度学习模型安全性的有效方法之一，它可以使模型与特意用于攻击模型的样本相兼容。在此基础上，我们提出了一种简单的架构来构建具有一定鲁棒性的模型，通过增加对抗性样本检测网络来进行协作训练，从而提高了训练网络的健壮性。同时，我们设计了一种新的数据采样策略，融合了多种已有的攻击，使得该模型能够通过一次训练来适应多种不同的对手攻击，并基于Cifar10数据集进行了一些实验，结果表明该设计对模型的健壮性有一定的积极作用。我们的代码可以在https://github.com/dowdyboy/simple_structure_for_robust_model上找到。



## **10. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2202.12232v2)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。在本文中，我们给出了当训练算法提供$\epsilon$-DP或$(\epsilon，\Delta)$-DP时，MI对手的正确率(即攻击精度)的一个更严格的界。我们的界提供了一种新的隐私放大方案的设计，其中有效的训练集在训练开始之前从较大的集合中被亚采样，以极大地降低对MI准确率的界。因此，我们的方案允许DP用户在训练他们的模型时采用更宽松的DP保证来限制任何MI对手的成功；这确保了模型的准确性较少地受到隐私保证的影响。最后，我们讨论了我们的MI界在机器遗忘领域的意义。



## **11. Metamorphic Testing-based Adversarial Attack to Fool Deepfake Detectors**

基于变形测试的对愚人深伪检测器的攻击 cs.CV

paper accepted at 26TH International Conference on Pattern  Recognition (ICPR2022)

**SubmitDate**: 2022-06-01    [paper-pdf](http://arxiv.org/pdf/2204.08612v2)

**Authors**: Nyee Thoang Lim, Meng Yi Kuan, Muxin Pu, Mei Kuan Lim, Chun Yong Chong

**Abstracts**: Deepfakes utilise Artificial Intelligence (AI) techniques to create synthetic media where the likeness of one person is replaced with another. There are growing concerns that deepfakes can be maliciously used to create misleading and harmful digital contents. As deepfakes become more common, there is a dire need for deepfake detection technology to help spot deepfake media. Present deepfake detection models are able to achieve outstanding accuracy (>90%). However, most of them are limited to within-dataset scenario, where the same dataset is used for training and testing. Most models do not generalise well enough in cross-dataset scenario, where models are tested on unseen datasets from another source. Furthermore, state-of-the-art deepfake detection models rely on neural network-based classification models that are known to be vulnerable to adversarial attacks. Motivated by the need for a robust deepfake detection model, this study adapts metamorphic testing (MT) principles to help identify potential factors that could influence the robustness of the examined model, while overcoming the test oracle problem in this domain. Metamorphic testing is specifically chosen as the testing technique as it fits our demand to address learning-based system testing with probabilistic outcomes from largely black-box components, based on potentially large input domains. We performed our evaluations on MesoInception-4 and TwoStreamNet models, which are the state-of-the-art deepfake detection models. This study identified makeup application as an adversarial attack that could fool deepfake detectors. Our experimental results demonstrate that both the MesoInception-4 and TwoStreamNet models degrade in their performance by up to 30\% when the input data is perturbed with makeup.

摘要: Deepfakes利用人工智能(AI)技术来创建合成媒体，其中一个人的肖像被另一个人取代。越来越多的人担心，深度假货可能被恶意用于创建误导性和有害的数字内容。随着深度假变得越来越普遍，迫切需要深度假检测技术来帮助识别深度假媒体。现有的深度伪检测模型能够达到显著的准确率(>90%)。然而，它们中的大多数仅限于数据集内的场景，其中相同的数据集用于训练和测试。大多数模型在跨数据集情况下不能很好地泛化，在这种情况下，模型是在来自另一个来源的不可见的数据集上进行测试的。此外，最先进的深度伪检测模型依赖于基于神经网络的分类模型，这些模型已知容易受到对手攻击。出于对稳健深度伪检测模型的需求，本研究采用变形测试(MT)原理来帮助识别可能影响被检查模型的稳健性的潜在因素，同时克服了该领域的测试预言问题。变形测试被特别选为测试技术，因为它符合我们的需求，以解决基于学习的系统测试，其结果主要来自黑盒组件，基于潜在的大输入域。我们对目前最先进的深度伪检测模型MesoInception-4和TwoStreamNet模型进行了评估。这项研究发现，化妆应用是一种对抗性攻击，可以愚弄深度假货检测器。实验结果表明，当输入数据受到置乱干扰时，两种模型的性能都下降了30%。



## **12. FoveaTer: Foveated Transformer for Image Classification**

FoveaTer：用于图像分类的凹槽转换器 cs.CV

17 pages, 7 figures

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2105.14173v2)

**Authors**: Aditya Jonnalagadda, William Yang Wang, B. S. Manjunath, Miguel P. Eckstein

**Abstracts**: Many animals and humans process the visual field with a varying spatial resolution (foveated vision) and use peripheral processing to make eye movements and point the fovea to acquire high-resolution information about objects of interest. This architecture results in computationally efficient rapid scene exploration. Recent progress in self-attention-based vision Transformers allow global interactions between feature locations and result in increased robustness to adversarial attacks. However, the Transformer models do not explicitly model the foveated properties of the visual system nor the interaction between eye movements and the classification task. We propose foveated Transformer (FoveaTer) model, which uses pooling regions and eye movements to perform object classification tasks. Our proposed model pools the image features using squared pooling regions, an approximation to the biologically-inspired foveated architecture. It decides on subsequent fixation locations based on the attention assigned by the Transformer to various locations from past and present fixations. It dynamically allocates more fixation/computational resources to more challenging images before making the final object category decision. We compare FoveaTer against a Full-resolution baseline model, which does not contain any pooling. On the ImageNet dataset, the Foveated model with Dynamic-stop achieves an accuracy of $1.9\%$ below the full-resolution model with a throughput gain of $51\%$. Using a Foveated model with Dynamic-stop and the Full-resolution model, the ensemble outperforms the baseline Full-resolution by $0.2\%$ with a throughput gain of $7.7\%$. We also demonstrate our model's robustness against adversarial attacks. Finally, we compare the Foveated model to human performance in a scene categorization task and show similar dependence of accuracy with number of exploratory fixations.

摘要: 许多动物和人类以不同的空间分辨率处理视野(中心凹视觉)，并使用外围处理来进行眼睛运动和指向中心凹以获取有关感兴趣对象的高分辨率信息。这种架构带来了计算效率高的快速场景探索。基于自我注意的视觉转换器的最新进展允许功能位置之间的全球交互，并提高了对对手攻击的健壮性。然而，变形金刚模型没有明确地模拟视觉系统的凹陷属性，也没有明确地模拟眼球运动和分类任务之间的相互作用。我们提出了凹陷变形器(FoveatedTransformer，FoveaTer)模型，该模型利用汇聚区域和眼动来执行目标分类任务。我们提出的模型使用平方池区域来集合图像特征，这近似于受生物启发的凹陷建筑。它根据变形金刚分配给过去和现在注视的不同位置的注意力来决定后续的注视位置。它在做出最终对象类别决定之前，动态地将更多的注视/计算资源分配给更具挑战性的图像。我们将FoveaTer与不包含任何池的全分辨率基线模型进行比较。在ImageNet数据集上，动态停止的Foveated模型的精度比全分辨率模型低1.9美元，吞吐量收益为51美元。使用动态停止模式和全分辨率模式，集合的性能比基线全分辨率高0.2美元，吞吐量增加7.7美元。我们还证明了我们的模型对对手攻击的健壮性。最后，我们将Foveated模型与人类在场景分类任务中的表现进行了比较，并显示出与探索性注视次数类似的准确率依赖关系。



## **13. Generative Models with Information-Theoretic Protection Against Membership Inference Attacks**

具有信息论保护的产生式模型抵抗成员推理攻击 cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00071v1)

**Authors**: Parisa Hassanzadeh, Robert E. Tillman

**Abstracts**: Deep generative models, such as Generative Adversarial Networks (GANs), synthesize diverse high-fidelity data samples by estimating the underlying distribution of high dimensional data. Despite their success, GANs may disclose private information from the data they are trained on, making them susceptible to adversarial attacks such as membership inference attacks, in which an adversary aims to determine if a record was part of the training set. We propose an information theoretically motivated regularization term that prevents the generative model from overfitting to training data and encourages generalizability. We show that this penalty minimizes the JensenShannon divergence between components of the generator trained on data with different membership, and that it can be implemented at low cost using an additional classifier. Our experiments on image datasets demonstrate that with the proposed regularization, which comes at only a small added computational cost, GANs are able to preserve privacy and generate high-quality samples that achieve better downstream classification performance compared to non-private and differentially private generative models.

摘要: 深度生成模型，如生成性对抗网络(GANS)，通过估计高维数据的潜在分布来合成各种高保真数据样本。尽管Gans取得了成功，但他们可能会从他们接受训练的数据中泄露私人信息，使他们容易受到对抗性攻击，如成员推理攻击，在这种攻击中，对手的目标是确定记录是否属于训练集的一部分。我们提出了一种信息理论激励的正则化条件，防止了生成模型对训练数据的过度拟合，并鼓励了泛化。我们表明，这种惩罚最小化了在不同成员资格的数据上训练的生成器组件之间的JensenShannon偏差，并且可以使用额外的分类器以低成本实现。我们在图像数据集上的实验表明，与非私有和差分私有的生成模型相比，GANS能够保护隐私并生成高质量的样本，从而获得更好的下游分类性能。



## **14. CodeAttack: Code-based Adversarial Attacks for Pre-Trained Programming Language Models**

CodeAttack：针对预先训练的编程语言模型的基于代码的对抗性攻击 cs.CL

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2206.00052v1)

**Authors**: Akshita Jha, Chandan K. Reddy

**Abstracts**: Pre-trained programming language (PL) models (such as CodeT5, CodeBERT, GraphCodeBERT, etc.,) have the potential to automate software engineering tasks involving code understanding and code generation. However, these models are not robust to changes in the input and thus, are potentially susceptible to adversarial attacks. We propose, CodeAttack, a simple yet effective black-box attack model that uses code structure to generate imperceptible, effective, and minimally perturbed adversarial code samples. We demonstrate the vulnerabilities of the state-of-the-art PL models to code-specific adversarial attacks. We evaluate the transferability of CodeAttack on several code-code (translation and repair) and code-NL (summarization) tasks across different programming languages. CodeAttack outperforms state-of-the-art adversarial NLP attack models to achieve the best overall performance while being more efficient and imperceptible.

摘要: 预先训练的编程语言(PL)模型(如CodeT5、CodeBERT、GraphCodeBERT等)有可能自动化涉及代码理解和代码生成的软件工程任务。然而，这些模型对输入的变化不是很健壮，因此可能容易受到对抗性攻击。我们提出了一个简单而有效的黑盒攻击模型CodeAttack，它使用代码结构来生成不可察觉的、有效的和最小扰动的对抗性代码样本。我们演示了最新的PL模型对代码特定的对抗性攻击的脆弱性。我们评估了CodeAttack在几个代码-代码(翻译和修复)和代码-NL(摘要)任务上跨不同编程语言的可移植性。CodeAttack超越了最先进的对抗性NLP攻击模型，在更高效和更隐蔽的同时实现了最佳的整体性能。



## **15. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

捉迷藏：关于深度学习系统攻击的隐蔽性 cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15944v1)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.

摘要: 随着人工智能和机器学习的日益普及，文献中提出了针对深度学习模型的广泛攻击。逃避攻击和投毒攻击都试图利用敌意更改的样本来愚弄受害者模型来错误分类敌意样本。虽然这种攻击声称是或预计是隐蔽的，即人眼看不见，但这种说法很少得到评估。本文首次对深度学习攻击中使用的敌意样本的隐蔽性进行了大规模研究。我们已经在六个流行的基准数据集上实施了20个具有代表性的对抗性ML攻击。我们使用两种互补的方法来评估攻击样本的隐蔽性：(1)采用24个度量来评估图像相似性或质量的数值研究；(2)用户研究3组问卷，从1000多个回复中收集了20,000多个注释。我们的结果表明，现有的大多数攻击都引入了不可忽略的扰动，这些扰动对人眼来说是不隐形的。进一步分析了影响攻击隐蔽性的因素。我们进一步检验了数值分析和用户研究之间的相关性，并证明了一些图像质量度量可以为攻击设计提供有用的指导，而评估的图像质量和攻击的视觉隐蔽性之间仍然存在着显著的差距。



## **16. Atomic cross-chain exchanges of shared assets**

共享资产的原子跨链交换 cs.CR

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2202.12855v2)

**Authors**: Krishnasuri Narayanam, Venkatraman Ramakrishna, Dhinakaran Vinayagamurthy, Sandeep Nishad

**Abstracts**: A core enabler for blockchain or DLT interoperability is the ability to atomically exchange assets held by mutually untrusting owners on different ledgers. This atomic swap problem has been well-studied, with the Hash Time Locked Contract (HTLC) emerging as a canonical solution. HTLC ensures atomicity of exchange, albeit with caveats for node failure and timeliness of claims. But a bigger limitation of HTLC is that it only applies to a model consisting of two adversarial parties having sole ownership of a single asset in each ledger. Realistic extensions of the model in which assets may be jointly owned by multiple parties, all of whose consents are required for exchanges, or where multiple assets must be exchanged for one, are susceptible to collusion attacks and hence cannot be handled by HTLC. In this paper, we generalize the model of asset exchanges across DLT networks and present a taxonomy of use cases, describe the threat model, and propose MPHTLC, an augmented HTLC protocol for atomic multi-owner-and-asset exchanges. We analyze the correctness, safety, and application scope of MPHTLC. As proof-of-concept, we show how MPHTLC primitives can be implemented in networks built on Hyperledger Fabric and Corda, and how MPHTLC can be implemented in the Hyperledger Labs Weaver framework by augmenting its existing HTLC protocol.

摘要: 区块链或DLT互操作性的核心推动因素是能够自动交换不同分类账上相互不信任的所有者持有的资产。这个原子交换问题已经得到了很好的研究，哈希时间锁定合同(HTLC)成为一种规范的解决方案。HTLC确保了交换的原子性，但对节点故障和索赔的及时性提出了警告。但HTLC的一个更大限制是，它只适用于由两个对立方单独拥有每个分类账中的一项资产的模型。资产可能由多方共同拥有的模型的现实扩展，其中交易需要所有各方的同意，或者必须用多个资产交换一个资产，这容易受到共谋攻击，因此无法由HTLC处理。本文对跨DLT网络的资产交换模型进行了推广，给出了用例的分类，描述了威胁模型，并提出了一种用于原子多所有者和资产交换的扩展HTLC协议MPHTLC。分析了MPHTLC的正确性、安全性和适用范围。作为概念验证，我们展示了如何在建立在Hyperledger Fabric和Corda上的网络中实现MPHTLC原语，以及如何通过增强Hyperledger Labs Weaver框架的现有HTLC协议来实现MPHTLC。



## **17. Semantic Autoencoder and Its Potential Usage for Adversarial Attack**

语义自动编码器及其在敌意攻击中的潜在应用 cs.LG

**SubmitDate**: 2022-05-31    [paper-pdf](http://arxiv.org/pdf/2205.15592v1)

**Authors**: Yurui Ming, Cuihuan Du, Chin-Teng Lin

**Abstracts**: Autoencoder can give rise to an appropriate latent representation of the input data, however, the representation which is solely based on the intrinsic property of the input data, is usually inferior to express some semantic information. A typical case is the potential incapability of forming a clear boundary upon clustering of these representations. By encoding the latent representation that not only depends on the content of the input data, but also the semantic of the input data, such as label information, we propose an enhanced autoencoder architecture named semantic autoencoder. Experiments of representation distribution via t-SNE shows a clear distinction between these two types of encoders and confirm the supremacy of the semantic one, whilst the decoded samples of these two types of autoencoders exhibit faint dissimilarity either objectively or subjectively. Based on this observation, we consider adversarial attacks to learning algorithms that rely on the latent representation obtained via autoencoders. It turns out that latent contents of adversarial samples constructed from semantic encoder with deliberate wrong label information exhibit different distribution compared with that of the original input data, while both of these samples manifest very marginal difference. This new way of attack set up by our work is worthy of attention due to the necessity to secure the widespread deep learning applications.

摘要: 自动编码器可以对输入数据进行适当的隐含表示，但单纯基于输入数据固有属性的表示往往不能很好地表达某些语义信息。一个典型的情况是，在对这些表示进行聚集时，可能无法形成清晰的边界。通过对潜在表示不仅依赖于输入数据的内容，还依赖于输入数据的语义(如标签信息)进行编码，提出了一种改进的自动编码器体系结构--语义自动编码器。通过t-SNE的表征分布实验表明，这两种类型的编码器之间有明显的区别，并证实了语义编码器的优越性，而这两种类型的自动编码器的解码样本无论在客观上还是主观上都表现出微弱的差异。基于这一观察结果，我们考虑了对依赖于通过自动编码器获得的潜在表示的学习算法的对抗性攻击。结果表明，含有故意错误标签信息的语义编码器构建的敌意样本的潜在内容与原始输入数据的潜在内容呈现不同的分布，但两者表现出非常微小的差异。这种由我们的工作建立的新的攻击方式值得关注，因为需要确保广泛的深度学习应用。



## **18. Connecting adversarial attacks and optimal transport for domain adaptation**

连接对抗性攻击和最优传输以实现域自适应 cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15424v1)

**Authors**: Arip Asadulaev, Vitaly Shutov, Alexander Korotin, Alexander Panfilov, Andrey Filchenkov

**Abstracts**: We present a novel algorithm for domain adaptation using optimal transport. In domain adaptation, the goal is to adapt a classifier trained on the source domain samples to the target domain. In our method, we use optimal transport to map target samples to the domain named source fiction. This domain differs from the source but is accurately classified by the source domain classifier. Our main idea is to generate a source fiction by c-cyclically monotone transformation over the target domain. If samples with the same labels in two domains are c-cyclically monotone, the optimal transport map between these domains preserves the class-wise structure, which is the main goal of domain adaptation. To generate a source fiction domain, we propose an algorithm that is based on our finding that adversarial attacks are a c-cyclically monotone transformation of the dataset. We conduct experiments on Digits and Modern Office-31 datasets and achieve improvement in performance for simple discrete optimal transport solvers for all adaptation tasks.

摘要: 我们提出了一种新的基于最优传输的域自适应算法。在领域自适应中，目标是使在源域样本上训练的分类器适应于目标域。在我们的方法中，我们使用最优传输将目标样本映射到名为源虚构的域。此域与源不同，但源域分类器会对其进行准确分类。我们的主要思想是通过目标域上的c-循环单调变换来生成源小说。如果两个结构域中具有相同标记的样本是c-循环单调的，那么这些结构域之间的最优传输映射保持了类结构，这是结构域适应的主要目标。为了生成源虚构领域，我们提出了一种算法，该算法基于我们的发现，即对抗性攻击是数据集的c循环单调变换。我们在Digits和现代Office-31数据集上进行了实验，并在所有适应任务的简单离散最优传输求解器的性能上取得了改进。



## **19. Fooling SHAP with Stealthily Biased Sampling**

用偷偷的有偏抽样愚弄Shap cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15419v1)

**Authors**: Gabriel Laberge, Ulrich Aïvodji, Satoshi Hara

**Abstracts**: SHAP explanations aim at identifying which features contribute the most to the difference in model prediction at a specific input versus a background distribution. Recent studies have shown that they can be manipulated by malicious adversaries to produce arbitrary desired explanations. However, existing attacks focus solely on altering the black-box model itself. In this paper, we propose a complementary family of attacks that leave the model intact and manipulate SHAP explanations using stealthily biased sampling of the data points used to approximate expectations w.r.t the background distribution. In the context of fairness audit, we show that our attack can reduce the importance of a sensitive feature when explaining the difference in outcomes between groups, while remaining undetected. These results highlight the manipulability of SHAP explanations and encourage auditors to treat post-hoc explanations with skepticism.

摘要: Shap解释旨在确定在特定输入与背景分布下，哪些特征对模型预测的差异贡献最大。最近的研究表明，它们可以被恶意攻击者操纵，以产生任意想要的解释。然而，现有的攻击仅仅集中在改变黑盒模型本身。在本文中，我们提出了一类互补的攻击，这些攻击保持模型不变，并通过对用于近似预期的背景分布的数据点的秘密有偏采样来操纵Shap解释。在公平审计的背景下，我们证明了我们的攻击可以在解释组之间结果差异时降低敏感特征的重要性，同时保持未被检测到。这些结果突显了Shap解释的可操纵性，并鼓励审计师对事后的解释持怀疑态度。



## **20. Searching for the Essence of Adversarial Perturbations**

寻找对抗性扰动的本质 cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15357v1)

**Authors**: Dennis Y. Menn, Hung-yi Lee

**Abstracts**: Neural networks have achieved the state-of-the-art performance on various machine learning fields, yet the incorporation of malicious perturbations with input data (adversarial example) is able to fool neural networks' predictions. This would lead to potential risks in real-world applications, for example, auto piloting and facial recognition. However, the reason for the existence of adversarial examples remains controversial. Here we demonstrate that adversarial perturbations contain human-recognizable information, which is the key conspirator responsible for a neural network's erroneous prediction. This concept of human-recognizable information allows us to explain key features related to adversarial perturbations, which include the existence of adversarial examples, the transferability among different neural networks, and the increased neural network interpretability for adversarial training. Two unique properties in adversarial perturbations that fool neural networks are uncovered: masking and generation. A special class, the complementary class, is identified when neural networks classify input images. The human-recognizable information contained in adversarial perturbations allows researchers to gain insight on the working principles of neural networks and may lead to develop techniques that detect/defense adversarial attacks.

摘要: 神经网络在不同的机器学习领域取得了最先进的性能，然而在输入数据中加入恶意扰动(对抗性的例子)能够愚弄神经网络的预测。这将导致现实世界应用中的潜在风险，例如自动驾驶和面部识别。然而，对抗性例子存在的原因仍然存在争议。在这里，我们证明了对抗性扰动包含人类可识别的信息，这是导致神经网络错误预测的关键阴谋者。这一人类可识别信息的概念使我们能够解释与对抗性扰动相关的关键特征，其中包括对抗性示例的存在、不同神经网络之间的可转换性以及用于对抗性训练的更高的神经网络可解释性。揭示了欺骗神经网络的对抗性扰动中的两个独特性质：掩蔽和生成。当神经网络对输入图像进行分类时，识别出一种特殊的类，即互补类。敌意干扰中包含的人类可识别的信息使研究人员能够深入了解神经网络的工作原理，并可能导致开发检测/防御敌意攻击的技术。



## **21. GAN-based Medical Image Small Region Forgery Detection via a Two-Stage Cascade Framework**

基于两级级联框架的GaN医学图像小区域篡改检测 eess.IV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15170v1)

**Authors**: Jianyi Zhang, Xuanxi Huang, Yaqi Liu, Yuyang Han, Zixiao Xiang

**Abstracts**: Using generative adversarial network (GAN)\cite{RN90} for data enhancement of medical images is significantly helpful for many computer-aided diagnosis (CAD) tasks. A new attack called CT-GAN has emerged. It can inject or remove lung cancer lesions to CT scans. Because the tampering region may even account for less than 1\% of the original image, even state-of-the-art methods are challenging to detect the traces of such tampering.   This paper proposes a cascade framework to detect GAN-based medical image small region forgery like CT-GAN. In the local detection stage, we train the detector network with small sub-images so that interference information in authentic regions will not affect the detector. We use depthwise separable convolution and residual to prevent the detector from over-fitting and enhance the ability to find forged regions through the attention mechanism. The detection results of all sub-images in the same image will be combined into a heatmap. In the global classification stage, using gray level co-occurrence matrix (GLCM) can better extract features of the heatmap. Because the shape and size of the tampered area are uncertain, we train PCA and SVM methods for classification. Our method can classify whether a CT image has been tampered and locate the tampered position. Sufficient experiments show that our method can achieve excellent performance.

摘要: 利用产生式对抗性网络(GAN)对医学图像进行数据增强，对许多计算机辅助诊断(CAD)任务有重要的帮助。一种名为CT-GAN的新攻击已经出现。它可以将肺癌病变注入或移除到CT扫描中。由于篡改区域甚至可能只占原始图像的不到1%，即使是最先进的方法也很难检测到这种篡改的痕迹。针对CT-GaN等基于GaN的医学图像小区域伪造问题，提出了一种级联检测框架。在局部检测阶段，我们用较小的子图像来训练检测器网络，使得真实区域中的干扰信息不会影响检测器。我们使用深度可分离的卷积和残差来防止检测器过拟合，并通过注意机制增强发现伪造区域的能力。将同一图像中所有子图像的检测结果合并成热图。在全局分类阶段，使用灰度共生矩阵(GLCM)可以更好地提取热图的特征。由于篡改区域的形状和大小是不确定的，我们训练了主成分分析和支持向量机方法进行分类。我们的方法可以对CT图像是否被篡改进行分类，并定位被篡改的位置。充分的实验表明，我们的方法可以达到很好的性能。



## **22. Why Adversarial Training of ReLU Networks Is Difficult?**

为什么RELU网络的对抗性训练很难？ cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15130v1)

**Authors**: Xu Cheng, Hao Zhang, Yue Xin, Wen Shen, Jie Ren, Quanshi Zhang

**Abstracts**: This paper mathematically derives an analytic solution of the adversarial perturbation on a ReLU network, and theoretically explains the difficulty of adversarial training. Specifically, we formulate the dynamics of the adversarial perturbation generated by the multi-step attack, which shows that the adversarial perturbation tends to strengthen eigenvectors corresponding to a few top-ranked eigenvalues of the Hessian matrix of the loss w.r.t. the input. We also prove that adversarial training tends to strengthen the influence of unconfident input samples with large gradient norms in an exponential manner. Besides, we find that adversarial training strengthens the influence of the Hessian matrix of the loss w.r.t. network parameters, which makes the adversarial training more likely to oscillate along directions of a few samples, and boosts the difficulty of adversarial training. Crucially, our proofs provide a unified explanation for previous findings in understanding adversarial training.

摘要: 本文从数学上推导了RELU网络上对抗性扰动的解析解，并从理论上解释了对抗性训练的困难。具体地说，我们描述了由多步攻击产生的对抗扰动的动力学，这表明对抗扰动倾向于增强对应于损失的Hessian矩阵的几个顶层特征值的特征向量。输入。我们还证明了对抗性训练倾向于以指数的方式增强具有大梯度范数的不自信输入样本的影响。此外，我们还发现，对抗性训练增强了损失的黑森矩阵的影响。网络参数，使得对抗性训练更容易沿着少数样本的方向振荡，增加了对抗性训练的难度。至关重要的是，我们的证据为理解对抗性训练提供了一个统一的解释。



## **23. Domain Constraints in Feature Space: Strengthening Robustness of Android Malware Detection against Realizable Adversarial Examples**

特征空间中的域约束：增强Android恶意软件检测对可实现的恶意示例的健壮性 cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.15128v1)

**Authors**: Hamid Bostani, Zhuoran Liu, Zhengyu Zhao, Veelasha Moonsamy

**Abstracts**: Strengthening the robustness of machine learning-based malware detectors against realistic evasion attacks remains one of the major obstacles for Android malware detection. To that end, existing work has focused on interpreting domain constraints of Android malware in the problem space, where problem-space realizable adversarial examples are generated. In this paper, we provide another promising way to achieve the same goal but based on interpreting the domain constraints in the feature space, where feature-space realizable adversarial examples are generated. Specifically, we present a novel approach to extracting feature-space domain constraints by learning meaningful feature dependencies from data, and applying them based on a novel robust feature space. Experimental results successfully demonstrate the effectiveness of our novel robust feature space in providing adversarial robustness for DREBIN, a state-of-the-art Android malware detector. For example, it can decrease the evasion rate of a realistic gradient-based attack by $96.4\%$ in a limited-knowledge (transfer) setting and by $13.8\%$ in a more challenging, perfect-knowledge setting. In addition, we show that directly using our learned domain constraints in the adversarial retraining framework leads to about $84\%$ improvement in a limited-knowledge setting, with up to $377\times$ faster implementation than using problem-space adversarial examples.

摘要: 增强基于机器学习的恶意软件检测器对现实规避攻击的健壮性仍然是Android恶意软件检测的主要障碍之一。为此，现有的工作集中于在问题空间中解释Android恶意软件的域约束，在问题空间中生成可实现的敌意示例。在本文中，我们提供了另一种有希望的方法来实现相同的目标，但基于对特征空间中的域约束的解释，在特征空间中生成可实现的对抗性实例。具体地说，我们提出了一种新的方法，通过从数据中学习有意义的特征依赖关系，并基于新的稳健特征空间来应用它们来提取特征空间域约束。实验结果成功地证明了新的稳健特征空间在为最先进的Android恶意软件检测器Drebin提供对抗鲁棒性方面的有效性。例如，它可以使基于现实梯度的攻击的逃避率在有限知识(迁移)环境下降低96.4美元，在更具挑战性的完美知识环境下降低13.8美元。此外，我们还表明，在对抗性再训练框架中直接使用我们学习到的领域约束可以在知识有限的情况下带来约84美元的改进，与使用问题空间对抗性示例相比，执行速度最高可快377倍。



## **24. Guided Diffusion Model for Adversarial Purification**

对抗性净化中的引导扩散模型 cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14969v1)

**Authors**: Jinyi Wang, Zhaoyang Lyu, Dahua Lin, Bo Dai, Hongfei Fu

**Abstracts**: With wider application of deep neural networks (DNNs) in various algorithms and frameworks, security threats have become one of the concerns. Adversarial attacks disturb DNN-based image classifiers, in which attackers can intentionally add imperceptible adversarial perturbations on input images to fool the classifiers. In this paper, we propose a novel purification approach, referred to as guided diffusion model for purification (GDMP), to help protect classifiers from adversarial attacks. The core of our approach is to embed purification into the diffusion denoising process of a Denoised Diffusion Probabilistic Model (DDPM), so that its diffusion process could submerge the adversarial perturbations with gradually added Gaussian noises, and both of these noises can be simultaneously removed following a guided denoising process. On our comprehensive experiments across various datasets, the proposed GDMP is shown to reduce the perturbations raised by adversarial attacks to a shallow range, thereby significantly improving the correctness of classification. GDMP improves the robust accuracy by 5%, obtaining 90.1% under PGD attack on the CIFAR10 dataset. Moreover, GDMP achieves 70.94% robustness on the challenging ImageNet dataset.

摘要: 随着深度神经网络(DNN)在各种算法和框架中的广泛应用，安全威胁已成为人们关注的问题之一。对抗性攻击干扰了基于DNN的图像分类器，攻击者可以故意在输入图像上添加不可察觉的对抗性扰动来愚弄分类器。在本文中，我们提出了一种新的净化方法，称为引导扩散净化模型(GDMP)，以帮助保护分类器免受对手攻击。该方法的核心是将净化嵌入到去噪扩散概率模型(DDPM)的扩散去噪过程中，使其扩散过程能够淹没带有逐渐增加的高斯噪声的对抗性扰动，并在引导去噪过程后同时去除这两种噪声。在不同数据集上的综合实验表明，所提出的GDMP将对抗性攻击引起的扰动减少到较小的范围，从而显著提高了分类的正确性。GDMP在CIFAR10数据集上的稳健准确率提高了5%，在PGD攻击下达到了90.1%。此外，GDMP在具有挑战性的ImageNet数据集上获得了70.94%的健壮性。



## **25. CalFAT: Calibrated Federated Adversarial Training with Label Skewness**

卡尔法特：带有标签偏斜度的校准联合对抗性训练 cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14926v1)

**Authors**: Chen Chen, Yuchen Liu, Xingjun Ma, Lingjuan Lyu

**Abstracts**: Recent studies have shown that, like traditional machine learning, federated learning (FL) is also vulnerable to adversarial attacks. To improve the adversarial robustness of FL, few federated adversarial training (FAT) methods have been proposed to apply adversarial training locally before global aggregation. Although these methods demonstrate promising results on independent identically distributed (IID) data, they suffer from training instability issues on non-IID data with label skewness, resulting in much degraded natural accuracy. This tends to hinder the application of FAT in real-world applications where the label distribution across the clients is often skewed. In this paper, we study the problem of FAT under label skewness, and firstly reveal one root cause of the training instability and natural accuracy degradation issues: skewed labels lead to non-identical class probabilities and heterogeneous local models. We then propose a Calibrated FAT (CalFAT) approach to tackle the instability issue by calibrating the logits adaptively to balance the classes. We show both theoretically and empirically that the optimization of CalFAT leads to homogeneous local models across the clients and much improved convergence rate and final performance.

摘要: 最近的研究表明，与传统的机器学习一样，联邦学习(FL)也容易受到对手攻击。为了提高FL的对抗健壮性，已有几种联邦对抗训练(FAT)方法被提出在全局聚集之前局部应用对抗训练。虽然这些方法在独立同分布(IID)数据上显示了良好的结果，但它们在具有标签偏斜的非IID数据上存在训练不稳定性问题，导致自然精度大大降低。这往往会阻碍FAT在实际应用中的应用，在现实应用中，跨客户端的标签分布通常是不对称的。本文研究了标签倾斜下的FAT问题，首次揭示了训练不稳定性和自然准确率下降问题的一个根本原因：标签倾斜会导致类别概率不一致和局部模型的异构性。然后，我们提出了一种校准FAT(CALFAT)方法来解决不稳定性问题，方法是自适应地校准逻辑以平衡类别。我们从理论和实验两个方面证明了CALFAT算法的优化可以得到跨客户的同质局部模型，并且大大提高了收敛速度和最终性能。



## **26. CausalAdv: Adversarial Robustness through the Lens of Causality**

CausalAdv：通过因果关系的透镜进行对抗的健壮性 cs.LG

ICLR2022, 20 pages, 3 figures

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2106.06196v2)

**Authors**: Yonggang Zhang, Mingming Gong, Tongliang Liu, Gang Niu, Xinmei Tian, Bo Han, Bernhard Schölkopf, Kun Zhang

**Abstracts**: The adversarial vulnerability of deep neural networks has attracted significant attention in machine learning. As causal reasoning has an instinct for modelling distribution change, it is essential to incorporate causality into analyzing this specific type of distribution change induced by adversarial attacks. However, causal formulations of the intuition of adversarial attacks and the development of robust DNNs are still lacking in the literature. To bridge this gap, we construct a causal graph to model the generation process of adversarial examples and define the adversarial distribution to formalize the intuition of adversarial attacks. From the causal perspective, we study the distinction between the natural and adversarial distribution and conclude that the origin of adversarial vulnerability is the focus of models on spurious correlations. Inspired by the causal understanding, we propose the Causal inspired Adversarial distribution alignment method, CausalAdv, to eliminate the difference between natural and adversarial distributions by considering spurious correlations. Extensive experiments demonstrate the efficacy of the proposed method. Our work is the first attempt towards using causality to understand and mitigate the adversarial vulnerability.

摘要: 深度神经网络的对抗性脆弱性在机器学习中引起了广泛的关注。由于因果推理具有模拟分布变化的本能，因此将因果关系纳入到分析由对抗性攻击引起的这种特定类型的分布变化中是至关重要的。然而，对抗性攻击的直觉和健壮DNN的发展的因果公式在文献中仍然缺乏。为了弥补这一差距，我们构建了一个因果图来建模对抗性实例的生成过程，并定义了对抗性分布来形式化对抗性攻击的直觉。从因果关系的角度，我们研究了自然分布和对抗性分布之间的区别，并得出结论：对抗性脆弱性的来源是伪相关性模型的重点。受因果理解的启发，我们提出了因果启发的对抗性分布对齐方法CausalAdv，通过考虑伪相关性来消除自然分布和对抗性分布之间的差异。大量实验证明了该方法的有效性。我们的工作是首次尝试使用因果关系来理解和缓解对抗性脆弱性。



## **27. Exposing Fine-grained Adversarial Vulnerability of Face Anti-spoofing Models**

暴露Face反欺骗模型的细粒度攻击漏洞 cs.CV

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14851v1)

**Authors**: Songlin Yang, Wei Wang, Chenye Xu, Bo Peng, Jing Dong

**Abstracts**: Adversarial attacks seriously threaten the high accuracy of face anti-spoofing models. Little adversarial noise can perturb their classification of live and spoofing. The existing adversarial attacks fail to figure out which part of the target face anti-spoofing model is vulnerable, making adversarial analysis tricky. So we propose fine-grained attacks for exposing adversarial vulnerability of face anti-spoofing models. Firstly, we propose Semantic Feature Augmentation (SFA) module, which makes adversarial noise semantic-aware to live and spoofing features. SFA considers the contrastive classes of data and texture bias of models in the context of face anti-spoofing, increasing the attack success rate by nearly 40% on average. Secondly, we generate fine-grained adversarial examples based on SFA and the multitask network with auxiliary information. We evaluate three annotations (facial attributes, spoofing types and illumination) and two geometric maps (depth and reflection), on four backbone networks (VGG, Resnet, Densenet and Swin Transformer). We find that facial attributes annotation and state-of-art networks fail to guarantee that models are robust to adversarial attacks. Such adversarial attacks can be generalized to more auxiliary information and backbone networks, to help our community handle the trade-off between accuracy and adversarial robustness.

摘要: 对抗性攻击严重威胁着人脸反欺骗模型的高精度。一点敌意的噪音就会扰乱他们对现场直播和恶搞的分类。现有的对抗性攻击无法计算出目标人脸的哪一部分是易受攻击的反欺骗模型，这使得对抗性分析变得棘手。因此，我们提出了细粒度攻击，以暴露人脸反欺骗模型的攻击漏洞。首先，我们提出了语义特征增强(SFA)模块，使得对抗噪声能够感知活特征和欺骗特征。SFA在人脸反欺骗的背景下考虑了模型的对比类数据和纹理偏向，使攻击成功率平均提高了近40%。其次，基于SFA和带辅助信息的多任务网络生成细粒度的对抗性实例。我们在四个骨干网络(VGG、RESNET、Densenet和Swin Transformer)上评估了三种标注(面部属性、欺骗类型和光照)和两种几何映射(深度和反射)。我们发现，人脸属性标注和最新的网络无法保证模型对对手攻击是健壮的。这种对抗性攻击可以推广到更多的辅助信息和主干网络，以帮助我们的社区处理准确性和对抗性健壮性之间的权衡。



## **28. Efficient Reward Poisoning Attacks on Online Deep Reinforcement Learning**

基于在线深度强化学习的高效奖赏中毒攻击 cs.LG

**SubmitDate**: 2022-05-30    [paper-pdf](http://arxiv.org/pdf/2205.14842v1)

**Authors**: Yinglun Xu, Qi Zeng, Gagandeep Singh

**Abstracts**: We study data poisoning attacks on online deep reinforcement learning (DRL) where the attacker is oblivious to the learning algorithm used by the agent and does not necessarily have full knowledge of the environment. We demonstrate the intrinsic vulnerability of state-of-the-art DRL algorithms by designing a general reward poisoning framework called adversarial MDP attacks. We instantiate our framework to construct several new attacks which only corrupt the rewards for a small fraction of the total training timesteps and make the agent learn a low-performing policy. Our key insight is that the state-of-the-art DRL algorithms strategically explore the environment to find a high-performing policy. Our attacks leverage this insight to construct a corrupted environment for misleading the agent towards learning low-performing policies with a limited attack budget. We provide a theoretical analysis of the efficiency of our attack and perform an extensive evaluation. Our results show that our attacks efficiently poison agents learning with a variety of state-of-the-art DRL algorithms, such as DQN, PPO, SAC, etc. under several popular classical control and MuJoCo environments.

摘要: 我们研究了针对在线深度强化学习(DRL)的数据中毒攻击，其中攻击者不知道代理使用的学习算法，并且不一定完全了解环境。我们通过设计一个称为对抗性MDP攻击的通用奖励中毒框架来展示最新的DRL算法的内在脆弱性。我们实例化了我们的框架来构造几个新的攻击，这些攻击只破坏了总训练时间步骤的一小部分回报，并使代理学习一个低性能的策略。我们的关键见解是，最先进的DRL算法从战略上探索环境，以找到高性能的策略。我们的攻击利用这一洞察力来构建一个腐败的环境，以误导代理在有限的攻击预算下学习低性能策略。我们对我们的攻击效率进行了理论分析，并进行了广泛的评估。我们的结果表明，在几种流行的经典控制和MuJoCo环境下，我们的攻击有效地毒化了使用各种先进的DRL算法学习的代理，如DQN、PPO、SAC等。



## **29. Mixture GAN For Modulation Classification Resiliency Against Adversarial Attacks**

混合遗传算法在调制分类抗攻击中的应用 cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.15743v1)

**Authors**: Eyad Shtaiwi, Ahmed El Ouadrhiri, Majid Moradikia, Salma Sultana, Ahmed Abdelhadi, Zhu Han

**Abstracts**: Automatic modulation classification (AMC) using the Deep Neural Network (DNN) approach outperforms the traditional classification techniques, even in the presence of challenging wireless channel environments. However, the adversarial attacks cause the loss of accuracy for the DNN-based AMC by injecting a well-designed perturbation to the wireless channels. In this paper, we propose a novel generative adversarial network (GAN)-based countermeasure approach to safeguard the DNN-based AMC systems against adversarial attack examples. GAN-based aims to eliminate the adversarial attack examples before feeding to the DNN-based classifier. Specifically, we have shown the resiliency of our proposed defense GAN against the Fast-Gradient Sign method (FGSM) algorithm as one of the most potent kinds of attack algorithms to craft the perturbed signals. The existing defense-GAN has been designed for image classification and does not work in our case where the above-mentioned communication system is considered. Thus, our proposed countermeasure approach deploys GANs with a mixture of generators to overcome the mode collapsing problem in a typical GAN facing radio signal classification problem. Simulation results show the effectiveness of our proposed defense GAN so that it could enhance the accuracy of the DNN-based AMC under adversarial attacks to 81%, approximately.

摘要: 使用深度神经网络(DNN)方法的自动调制分类(AMC)性能优于传统的分类技术，即使在具有挑战性的无线信道环境中也是如此。然而，敌意攻击通过向无线信道注入精心设计的扰动而导致基于DNN的AMC的准确性损失。本文提出了一种新的基于产生式对抗网络(GAN)的对抗方法来保护基于DNN的AMC系统免受敌意攻击。基于GAN的分类器的目的是在输入到基于DNN的分类器之前消除对抗性攻击实例。具体地说，我们已经展示了我们提出的防御GAN对快速梯度符号方法(FGSM)算法的弹性，作为制造扰动信号的最有效的攻击算法之一。现有的防御-GAN是为图像分类而设计的，并且在我们考虑上述通信系统的情况下不起作用。因此，我们提出的对策方法在典型的面向无线电信号分类问题的GaN中部署混合生成器来克服模式崩溃问题。仿真结果表明，本文提出的防御GAN算法是有效的，可以将基于DNN的AMC在对抗攻击下的准确率提高到大约81%。



## **30. Unfooling Perturbation-Based Post Hoc Explainers**

基于非愚弄扰动的帖子随机解说器 cs.AI

10 pages (not including references and supplemental)

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14772v1)

**Authors**: Zachariah Carmichael, Walter J Scheirer

**Abstracts**: Monumental advancements in artificial intelligence (AI) have lured the interest of doctors, lenders, judges, and other professionals. While these high-stakes decision-makers are optimistic about the technology, those familiar with AI systems are wary about the lack of transparency of its decision-making processes. Perturbation-based post hoc explainers offer a model agnostic means of interpreting these systems while only requiring query-level access. However, recent work demonstrates that these explainers can be fooled adversarially. This discovery has adverse implications for auditors, regulators, and other sentinels. With this in mind, several natural questions arise - how can we audit these black box systems? And how can we ascertain that the auditee is complying with the audit in good faith? In this work, we rigorously formalize this problem and devise a defense against adversarial attacks on perturbation-based explainers. We propose algorithms for the detection (CAD-Detect) and defense (CAD-Defend) of these attacks, which are aided by our novel conditional anomaly detection approach, KNN-CAD. We demonstrate that our approach successfully detects whether a black box system adversarially conceals its decision-making process and mitigates the adversarial attack on real-world data for the prevalent explainers, LIME and SHAP.

摘要: 人工智能(AI)的巨大进步吸引了医生、贷款人、法官和其他专业人士的兴趣。尽管这些事关重大的决策者对这项技术持乐观态度，但那些熟悉人工智能系统的人对其决策过程缺乏透明度持谨慎态度。基于扰动的后自组织解释器提供了一种模型不可知的方法来解释这些系统，而只需要查询级别的访问。然而，最近的研究表明，这些解释程序可能会被相反的人愚弄。这一发现对审计师、监管者和其他哨兵产生了不利影响。考虑到这一点，几个自然的问题就产生了--我们如何审计这些黑匣子系统？我们如何确定被审计人是真诚地遵守审计的？在这项工作中，我们严格地形式化了这个问题，并设计了一个防御对基于扰动的解释器的敌意攻击。在新的条件异常检测方法KNN-CAD的辅助下，我们提出了针对这些攻击的检测(CAD-检测)和防御(CAD-防御)算法。我们证明，我们的方法成功地检测到黑盒系统是否恶意地隐藏了其决策过程，并缓解了流行的解释程序LIME和Shap对真实数据的恶意攻击。



## **31. On the Robustness of Safe Reinforcement Learning under Observational Perturbations**

安全强化学习在观测摄动下的稳健性 cs.LG

27 pages, 3 figures, 3 tables

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14691v1)

**Authors**: Zuxin Liu, Zijian Guo, Zhepeng Cen, Huan Zhang, Jie Tan, Bo Li, Ding Zhao

**Abstracts**: Safe reinforcement learning (RL) trains a policy to maximize the task reward while satisfying safety constraints. While prior works focus on the performance optimality, we find that the optimal solutions of many safe RL problems are not robust and safe against carefully designed observational perturbations. We formally analyze the unique properties of designing effective state adversarial attackers in the safe RL setting. We show that baseline adversarial attack techniques for standard RL tasks are not always effective for safe RL and proposed two new approaches - one maximizes the cost and the other maximizes the reward. One interesting and counter-intuitive finding is that the maximum reward attack is strong, as it can both induce unsafe behaviors and make the attack stealthy by maintaining the reward. We further propose a more effective adversarial training framework for safe RL and evaluate it via comprehensive experiments. This work sheds light on the inherited connection between observational robustness and safety in RL and provides a pioneer work for future safe RL studies.

摘要: 安全强化学习(RL)训练一种策略，在满足安全约束的同时最大化任务奖励。虽然以前的工作主要集中在性能最优性上，但我们发现许多安全RL问题的最优解对于精心设计的观测扰动并不是健壮的和安全的。我们形式化地分析了在安全RL环境下设计有效的状态对抗攻击者的独特性质。我们证明了标准RL任务的基线对抗性攻击技术对于安全RL并不总是有效的，并提出了两种新的方法-一种最大化成本，另一种最大化回报。一个有趣和违反直觉的发现是，最大奖励攻击是强大的，因为它既可以诱导不安全的行为，又可以通过保持奖励来使攻击隐形。我们进一步提出了一种更有效的安全RL对抗训练框架，并通过综合实验对其进行了评估。这项工作揭示了RL中观测稳健性和安全性之间的遗传联系，并为未来的安全RL研究提供了开创性的工作。



## **32. Superclass Adversarial Attack**

超类对抗性攻击 cs.CV

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14629v1)

**Authors**: Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**Abstracts**: Adversarial attacks have only focused on changing the predictions of the classifier, but their danger greatly depends on how the class is mistaken. For example, when an automatic driving system mistakes a Persian cat for a Siamese cat, it is hardly a problem. However, if it mistakes a cat for a 120km/h minimum speed sign, serious problems can arise. As a stepping stone to more threatening adversarial attacks, we consider the superclass adversarial attack, which causes misclassification of not only fine classes, but also superclasses. We conducted the first comprehensive analysis of superclass adversarial attacks (an existing and 19 new methods) in terms of accuracy, speed, and stability, and identified several strategies to achieve better performance. Although this study is aimed at superclass misclassification, the findings can be applied to other problem settings involving multiple classes, such as top-k and multi-label classification attacks.

摘要: 对抗性攻击只专注于改变分类器的预测，但它们的危险在很大程度上取决于类的错误程度。例如，当自动驾驶系统将波斯猫误认为暹罗猫时，这几乎不是问题。然而，如果它把猫错当成120公里/小时的最低速度标志，可能会出现严重的问题。作为更具威胁性的对抗性攻击的垫脚石，我们认为超类对抗性攻击不仅会导致细类的错误分类，而且会导致超类的错误分类。我们首次对超类对抗性攻击(现有方法和19种新方法)在准确性、速度和稳定性方面进行了全面分析，并确定了几种实现更好性能的策略。虽然这项研究是针对超类错误分类的，但研究结果也适用于其他涉及多类的问题，如top-k和多标签分类攻击。



## **33. Graph Structure Based Data Augmentation Method**

一种基于图结构的数据增强方法 cs.LG

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14619v1)

**Authors**: Kyung Geun Kim, Byeong Tak Lee

**Abstracts**: In this paper, we propose a novel graph-based data augmentation method that can generally be applied to medical waveform data with graph structures. In the process of recording medical waveform data, such as electrocardiogram (ECG) or electroencephalogram (EEG), angular perturbations between the measurement leads exist due to discrepancies in lead positions. The data samples with large angular perturbations often cause inaccuracy in algorithmic prediction tasks. We design a graph-based data augmentation technique that exploits the inherent graph structures within the medical waveform data to improve both performance and robustness. In addition, we show that the performance gain from graph augmentation results from robustness by testing against adversarial attacks. Since the bases of performance gain are orthogonal, the graph augmentation can be used in conjunction with existing data augmentation techniques to further improve the final performance. We believe that our graph augmentation method opens up new possibilities to explore in data augmentation.

摘要: 在本文中，我们提出了一种新的基于图的数据增强方法，该方法普遍适用于具有图结构的医学波形数据。在记录心电或脑电等医学波形数据的过程中，由于导联位置的差异，测量导联之间存在角度摄动。在算法预测任务中，具有较大角度摄动的数据样本经常导致不准确。我们设计了一种基于图形的数据增强技术，该技术利用医学波形数据中固有的图形结构来提高性能和稳健性。此外，我们还通过对敌意攻击的测试，证明了图增强带来的性能提升来自于健壮性。由于性能增益的基础是正交性的，图增强可以与现有的数据增强技术相结合来进一步提高最终的性能。我们相信，我们的图形增强方法为探索数据增强开辟了新的可能性。



## **34. Problem-Space Evasion Attacks in the Android OS: a Survey**

Android操作系统中的问题空间逃避攻击：综述 cs.CR

**SubmitDate**: 2022-05-29    [paper-pdf](http://arxiv.org/pdf/2205.14576v1)

**Authors**: Harel Berger, Dr. Chen Hajaj, Dr. Amit Dvir

**Abstracts**: Android is the most popular OS worldwide. Therefore, it is a target for various kinds of malware. As a countermeasure, the security community works day and night to develop appropriate Android malware detection systems, with ML-based or DL-based systems considered as some of the most common types. Against these detection systems, intelligent adversaries develop a wide set of evasion attacks, in which an attacker slightly modifies a malware sample to evade its target detection system. In this survey, we address problem-space evasion attacks in the Android OS, where attackers manipulate actual APKs, rather than their extracted feature vector. We aim to explore this kind of attacks, frequently overlooked by the research community due to a lack of knowledge of the Android domain, or due to focusing on general mathematical evasion attacks - i.e., feature-space evasion attacks. We discuss the different aspects of problem-space evasion attacks, using a new taxonomy, which focuses on key ingredients of each problem-space attack, such as the attacker model, the attacker's mode of operation, and the functional assessment of post-attack applications.

摘要: 安卓是全球最受欢迎的操作系统。因此，它是各种恶意软件的目标。作为对策，安全社区夜以继日地开发合适的Android恶意软件检测系统，基于ML或基于DL的系统被认为是一些最常见的类型。针对这些检测系统，智能攻击者开发了一系列广泛的逃避攻击，攻击者略微修改恶意软件样本以逃避其目标检测系统。在这篇调查中，我们讨论了Android操作系统中的问题空间逃避攻击，即攻击者操纵实际的APK，而不是他们提取的特征向量。我们的目标是探索这类攻击，由于缺乏Android领域的知识，或者由于专注于一般的数学逃避攻击，即特征空间逃避攻击，经常被研究界忽视。我们讨论了问题空间逃避攻击的不同方面，使用了一种新的分类方法，重点讨论了每种问题空间攻击的关键要素，如攻击者的模型、攻击者的操作模式以及攻击后应用程序的功能评估。



## **35. BadDet: Backdoor Attacks on Object Detection**

BadDet：针对对象检测的后门攻击 cs.CV

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14497v1)

**Authors**: Shih-Han Chan, Yinpeng Dong, Jun Zhu, Xiaolu Zhang, Jun Zhou

**Abstracts**: Deep learning models have been deployed in numerous real-world applications such as autonomous driving and surveillance. However, these models are vulnerable in adversarial environments. Backdoor attack is emerging as a severe security threat which injects a backdoor trigger into a small portion of training data such that the trained model behaves normally on benign inputs but gives incorrect predictions when the specific trigger appears. While most research in backdoor attacks focuses on image classification, backdoor attacks on object detection have not been explored but are of equal importance. Object detection has been adopted as an important module in various security-sensitive applications such as autonomous driving. Therefore, backdoor attacks on object detection could pose severe threats to human lives and properties. We propose four kinds of backdoor attacks for object detection task: 1) Object Generation Attack: a trigger can falsely generate an object of the target class; 2) Regional Misclassification Attack: a trigger can change the prediction of a surrounding object to the target class; 3) Global Misclassification Attack: a single trigger can change the predictions of all objects in an image to the target class; and 4) Object Disappearance Attack: a trigger can make the detector fail to detect the object of the target class. We develop appropriate metrics to evaluate the four backdoor attacks on object detection. We perform experiments using two typical object detection models -- Faster-RCNN and YOLOv3 on different datasets. More crucially, we demonstrate that even fine-tuning on another benign dataset cannot remove the backdoor hidden in the object detection model. To defend against these backdoor attacks, we propose Detector Cleanse, an entropy-based run-time detection framework to identify poisoned testing samples for any deployed object detector.

摘要: 深度学习模型已被部署在许多现实世界的应用中，如自动驾驶和监控。然而，这些模型在对抗性环境中很容易受到攻击。后门攻击正在成为一种严重的安全威胁，它向一小部分训练数据注入后门触发器，使训练后的模型在良性输入下正常运行，但在特定触发器出现时给出错误的预测。虽然大多数关于后门攻击的研究都集中在图像分类上，但对目标检测的后门攻击还没有被探索过，但同样重要。目标检测已经成为自动驾驶等各种安全敏感应用中的一个重要模块。因此，对目标检测的后门攻击可能会对人类的生命和财产构成严重威胁。针对目标检测任务，我们提出了四种后门攻击：1)对象生成攻击：1)对象生成攻击：1)对象生成攻击；2)区域误分类攻击：1)局部误分类攻击：1)全局误分类攻击；3)全局误分类攻击：单个触发器可以将图像中所有对象的预测更改为目标类；4)对象消失攻击：1)触发器可以使检测器无法检测到目标类对象。我们开发了适当的度量来评估对象检测中的四种后门攻击。我们使用两个典型的目标检测模型--FASTER-RCNN和YOLOv3在不同的数据集上进行了实验。更关键的是，我们证明了即使对另一个良性数据集进行微调也不能消除隐藏在目标检测模型中的后门。为了防御这些后门攻击，我们提出了检测器Cleanse，这是一个基于熵的运行时检测框架，可以为任何部署的对象检测器识别有毒测试样本。



## **36. Policy Smoothing for Provably Robust Reinforcement Learning**

可证明稳健强化学习的策略平滑 cs.LG

Published as a conference paper at ICLR 2022

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2106.11420v3)

**Authors**: Aounon Kumar, Alexander Levine, Soheil Feizi

**Abstracts**: The study of provable adversarial robustness for deep neural networks (DNNs) has mainly focused on static supervised learning tasks such as image classification. However, DNNs have been used extensively in real-world adaptive tasks such as reinforcement learning (RL), making such systems vulnerable to adversarial attacks as well. Prior works in provable robustness in RL seek to certify the behaviour of the victim policy at every time-step against a non-adaptive adversary using methods developed for the static setting. But in the real world, an RL adversary can infer the defense strategy used by the victim agent by observing the states, actions, etc., from previous time-steps and adapt itself to produce stronger attacks in future steps. We present an efficient procedure, designed specifically to defend against an adaptive RL adversary, that can directly certify the total reward without requiring the policy to be robust at each time-step. Our main theoretical contribution is to prove an adaptive version of the Neyman-Pearson Lemma -- a key lemma for smoothing-based certificates -- where the adversarial perturbation at a particular time can be a stochastic function of current and previous observations and states as well as previous actions. Building on this result, we propose policy smoothing where the agent adds a Gaussian noise to its observation at each time-step before passing it through the policy function. Our robustness certificates guarantee that the final total reward obtained by policy smoothing remains above a certain threshold, even though the actions at intermediate time-steps may change under the attack. Our experiments on various environments like Cartpole, Pong, Freeway and Mountain Car show that our method can yield meaningful robustness guarantees in practice.

摘要: 深度神经网络(DNN)的可证对抗鲁棒性研究主要集中在图像分类等静态监督学习任务上。然而，DNN已被广泛应用于现实世界中的自适应任务，如强化学习(RL)，这使得此类系统也容易受到对手攻击。RL中的可证明健壮性方面的先前工作试图使用为静态设置开发的方法来证明受害者策略在针对非自适应对手的每个时间步的行为。但在现实世界中，RL对手可以通过观察以前时间步长的状态、动作等来推断受害者代理所使用的防御策略，并在未来的步骤中进行调整以产生更强的攻击。我们提出了一个有效的程序，专门设计来防御自适应的RL对手，它可以直接证明总的奖励，而不需要策略在每个时间步都是健壮的。我们的主要理论贡献是证明了Neyman-Pearson引理的一个自适应版本--这是一个用于平滑基于证书的关键引理--其中在特定时间的对抗性扰动可以是当前和先前的观察和状态以及先前操作的随机函数。基于这一结果，我们提出了策略平滑，即在通过策略函数之前，代理在每个时间步向其观测结果添加高斯噪声。我们的稳健性证书保证了策略平滑获得的最终总回报保持在一定的阈值以上，即使中间时间步长的动作在攻击下可能发生变化。我们在Cartpoll、Pong、Freeway和Mountain Car等环境下的实验表明，该方法在实践中可以产生有意义的健壮性保证。



## **37. Certifying Model Accuracy under Distribution Shifts**

分布漂移下的模型精度验证 cs.LG

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2201.12440v2)

**Authors**: Aounon Kumar, Alexander Levine, Tom Goldstein, Soheil Feizi

**Abstracts**: Certified robustness in machine learning has primarily focused on adversarial perturbations of the input with a fixed attack budget for each point in the data distribution. In this work, we present provable robustness guarantees on the accuracy of a model under bounded Wasserstein shifts of the data distribution. We show that a simple procedure that randomizes the input of the model within a transformation space is provably robust to distributional shifts under the transformation. Our framework allows the datum-specific perturbation size to vary across different points in the input distribution and is general enough to include fixed-sized perturbations as well. Our certificates produce guaranteed lower bounds on the performance of the model for any (natural or adversarial) shift of the input distribution within a Wasserstein ball around the original distribution. We apply our technique to: (i) certify robustness against natural (non-adversarial) transformations of images such as color shifts, hue shifts and changes in brightness and saturation, (ii) certify robustness against adversarial shifts of the input distribution, and (iii) show provable lower bounds (hardness results) on the performance of models trained on so-called "unlearnable" datasets that have been poisoned to interfere with model training.

摘要: 机器学习中已证明的稳健性主要集中在输入的对抗性扰动上，对数据分布中的每个点都有固定的攻击预算。在这项工作中，我们给出了在数据分布的有界Wasserstein位移下模型精度的可证明的稳健性保证。我们证明了在变换空间内随机化模型输入的简单过程对变换下的分布位移是被证明是健壮的。我们的框架允许特定于基准的扰动大小在输入分布的不同点上变化，并且足够普遍以包括固定大小的扰动。我们的证书为输入分布在Wasserstein球中围绕原始分布的任何(自然或对抗性)移动产生了模型性能的保证下限。我们将我们的技术应用于：(I)证明对图像的自然(非对抗性)变换的稳健性，例如颜色漂移、色调漂移以及亮度和饱和度的变化，(Ii)验证对输入分布的对抗性漂移的稳健性，以及(Iii)显示在已被毒害到干扰模型训练的所谓的“不可学习”数据集上训练的模型的性能的可证明的下界(困难结果)。



## **38. SHORTSTACK: Distributed, Fault-tolerant, Oblivious Data Access**

ShortStack：分布式、容错、不经意的数据访问 cs.CR

Full version of USENIX OSDI'22 paper

**SubmitDate**: 2022-05-28    [paper-pdf](http://arxiv.org/pdf/2205.14281v1)

**Authors**: Midhul Vuppalapati, Kushal Babel, Anurag Khandelwal, Rachit Agarwal

**Abstracts**: Many applications that benefit from data offload to cloud services operate on private data. A now-long line of work has shown that, even when data is offloaded in an encrypted form, an adversary can learn sensitive information by analyzing data access patterns. Existing techniques for oblivious data access--that protect against access pattern attacks--require a centralized, stateful and trusted, proxy to orchestrate data accesses from applications to cloud services. We show that, in failure-prone deployments, such a centralized and stateful proxy results in violation of oblivious data access security guarantees and/or system unavailability. Thus, we initiate the study of distributed, fault-tolerant, oblivious data access.   We present SHORTSTACK, a distributed proxy architecture for oblivious data access in failure-prone deployments. SHORTSTACK achieves the classical obliviousness guarantee--access patterns observed by the adversary being independent of the input--even under a powerful passive persistent adversary that can force failure of arbitrary (bounded-sized) subset of proxy servers at arbitrary times. We also introduce a security model that enables studying oblivious data access with distributed, failure-prone, servers. We provide a formal proof that SHORTSTACK enables oblivious data access under this model, and show empirically that SHORTSTACK performance scales near-linearly with number of distributed proxy servers.

摘要: 许多受益于数据分流到云服务的应用程序都在私有数据上运行。目前的一系列工作表明，即使以加密的形式卸载数据，对手也可以通过分析数据访问模式来获取敏感信息。现有的不经意数据访问技术--防止访问模式攻击--需要一个集中的、有状态的、可信的代理来协调从应用程序到云服务的数据访问。我们表明，在容易出现故障的部署中，这种集中式和有状态的代理会导致违反不经意的数据访问安全保证和/或系统不可用。因此，我们开始了对分布式、容错、不经意的数据访问的研究。我们提出了ShortStack，这是一种分布式代理体系结构，用于在容易出现故障的部署中进行不经意的数据访问。ShortStack实现了经典的遗忘保证--攻击者观察到的访问模式独立于输入--即使在强大的被动持久对手下也是如此，该对手可以在任意时间强制任意(有限大小的)代理服务器子集发生故障。我们还介绍了一个安全模型，该模型能够研究使用分布式的、容易发生故障的服务器的不经意的数据访问。我们给出了一个形式化的证明，证明了在该模型下，ShortStack能够实现不经意的数据访问，并通过实验证明了ShortStack的性能随着分布式代理服务器的数量近似线性地扩展。



## **39. Semi-supervised Semantics-guided Adversarial Training for Trajectory Prediction**

用于弹道预测的半监督语义制导对抗性训练 cs.LG

11 pages, adversarial training for trajectory prediction

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14230v1)

**Authors**: Ruochen Jiao, Xiangguo Liu, Takami Sato, Qi Alfred Chen, Qi Zhu

**Abstracts**: Predicting the trajectories of surrounding objects is a critical task in self-driving and many other autonomous systems. Recent works demonstrate that adversarial attacks on trajectory prediction, where small crafted perturbations are introduced to history trajectories, may significantly mislead the prediction of future trajectories and ultimately induce unsafe planning. However, few works have addressed enhancing the robustness of this important safety-critical task. In this paper, we present the first adversarial training method for trajectory prediction. Compared with typical adversarial training on image tasks, our work is challenged by more random inputs with rich context, and a lack of class labels. To address these challenges, we propose a method based on a semi-supervised adversarial autoencoder that models disentangled semantic features with domain knowledge and provides additional latent labels for the adversarial training. Extensive experiments with different types of attacks demonstrate that our semi-supervised semantics-guided adversarial training method can effectively mitigate the impact of adversarial attacks and generally improve the system's adversarial robustness to a variety of attacks, including unseen ones. We believe that such semantics-guided architecture and advancement in robust generalization is an important step for developing robust prediction models and enabling safe decision making.

摘要: 在自动驾驶和许多其他自主系统中，预测周围物体的轨迹是一项关键任务。最近的工作表明，对轨迹预测的敌意攻击，即在历史轨迹中引入微小的精心设计的扰动，可能会严重误导对未来轨迹的预测，并最终导致不安全的规划。然而，很少有工作涉及增强这一重要的安全关键任务的健壮性。在本文中，我们提出了第一种用于轨迹预测的对抗性训练方法。与典型的对抗性图像训练相比，我们的工作面临着更多的随机输入和丰富的上下文，以及缺乏类别标签的挑战。为了应对这些挑战，我们提出了一种基于半监督对抗性自动编码器的方法，该方法利用领域知识对解开的语义特征进行建模，并为对抗性训练提供额外的潜在标签。对不同类型的攻击进行的大量实验表明，本文提出的半监督语义制导的对抗性训练方法能够有效地缓解对抗性攻击的影响，并总体上提高了系统对包括不可见攻击在内的各种攻击的鲁棒性。我们认为，这种语义引导的体系结构和在健壮泛化方面的进步是开发健壮预测模型和实现安全决策的重要一步。



## **40. A Single-Adversary-Single-Detector Zero-Sum Game in Networked Control Systems**

网络控制系统中的单对手-单检测器零和博弈 math.OC

6 pages, 6 figures, 1 table, accepted to the 9th IFAC Conference on  Networked Systems, Zurich, July 2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.14001v1)

**Authors**: Anh Tung Nguyen, André M. H. Teixeira, Alexander Medvedev

**Abstracts**: This paper proposes a game-theoretic approach to address the problem of optimal sensor placement for detecting cyber-attacks in networked control systems. The problem is formulated as a zero-sum game with two players, namely a malicious adversary and a detector. Given a protected target vertex, the detector places a sensor at a single vertex to monitor the system and detect the presence of the adversary. On the other hand, the adversary selects a single vertex through which to conduct a cyber-attack that maximally disrupts the target vertex while remaining undetected by the detector. As our first contribution, for a given pair of attack and monitor vertices and a known target vertex, the game payoff function is defined as the output-to-output gain of the respective system. Then, the paper characterizes the set of feasible actions by the detector that ensures bounded values of the game payoff. Finally, an algebraic sufficient condition is proposed to examine whether a given vertex belongs to the set of feasible monitor vertices. The optimal sensor placement is then determined by computing the mixed-strategy Nash equilibrium of the zero-sum game through linear programming. The approach is illustrated via a numerical example of a 10-vertex networked control system with a given target vertex.

摘要: 本文提出了一种基于博弈论的方法来解决网络控制系统中检测网络攻击的传感器最优配置问题。该问题被描述为一个有两个参与者的零和博弈，即一个恶意对手和一个检测器。在给定一个受保护的目标顶点的情况下，检测器在单个顶点放置一个传感器来监视系统并检测对手的存在。另一方面，敌手选择单个顶点进行网络攻击，最大限度地破坏目标顶点，同时保持不被检测器检测。作为我们的第一个贡献，对于给定的攻击和监视顶点对和已知的目标顶点，博弈收益函数被定义为各自系统的输出到输出的增益。然后，利用检测器刻画了保证博弈收益有界值的可行行为集。最后，给出了一个判定给定顶点是否属于可行监视顶点集的代数充分条件。然后通过线性规划计算零和博弈的混合策略纳什均衡来确定传感器的最优配置。给出了一个具有给定目标节点的10点网络控制系统的算例。



## **41. Standalone Neural ODEs with Sensitivity Analysis**

带敏感度分析的独立神经网络模型 cs.LG

25 pages, 15 figures

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13933v1)

**Authors**: Rym Jaroudi, Lukáš Malý, Gabriel Eilertsen, Tomas B. Johansson, Jonas Unger, George Baravdish

**Abstracts**: This paper presents the Standalone Neural ODE (sNODE), a continuous-depth neural ODE model capable of describing a full deep neural network. This uses a novel nonlinear conjugate gradient (NCG) descent optimization scheme for training, where the Sobolev gradient can be incorporated to improve smoothness of model weights. We also present a general formulation of the neural sensitivity problem and show how it is used in the NCG training. The sensitivity analysis provides a reliable measure of uncertainty propagation throughout a network, and can be used to study model robustness and to generate adversarial attacks. Our evaluations demonstrate that our novel formulations lead to increased robustness and performance as compared to ResNet models, and that it opens up for new opportunities for designing and developing machine learning with improved explainability.

摘要: 提出了一种能够描述完整深度神经网络的连续深度神经网络模型--独立神经网络模型(SNODE)。该算法采用一种新的非线性共轭梯度(NCG)下降优化方案进行训练，其中可以引入Soblev梯度来改善模型权重的平滑程度。我们还给出了神经敏感度问题的一般公式，并展示了如何将其用于NCG训练。敏感度分析为不确定性在整个网络中的传播提供了可靠的度量，并可用于研究模型的健壮性和生成对抗性攻击。我们的评估表明，与ResNet模型相比，我们的新公式导致了更高的稳健性和性能，并为设计和开发具有更好解释性的机器学习开辟了新的机会。



## **42. Evaluating the Robustness of Deep Reinforcement Learning for Autonomous and Adversarial Policies in a Multi-agent Urban Driving Environment**

多智能体城市驾驶环境下自主对抗性策略的深度强化学习稳健性评价 cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11947v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Deep reinforcement learning is actively used for training autonomous and adversarial car policies in a simulated driving environment. Due to the large availability of various reinforcement learning algorithms and the lack of their systematic comparison across different driving scenarios, we are unsure of which ones are more effective for training and testing autonomous car software in single-agent as well as multi-agent driving environments. A benchmarking framework for the comparison of deep reinforcement learning in a vision-based autonomous driving will open up the possibilities for training better autonomous car driving policies. Furthermore, autonomous cars trained on deep reinforcement learning-based algorithms are known for being vulnerable to adversarial attacks. To guard against adversarial attacks, we can train autonomous cars on adversarial driving policies. However, we lack the knowledge of which deep reinforcement learning algorithms would act as good adversarial agents able to effectively test autonomous cars. To address these challenges, we provide an open and reusable benchmarking framework for systematic evaluation and comparative analysis of deep reinforcement learning algorithms for autonomous and adversarial driving in a single- and multi-agent environment. Using the framework, we perform a comparative study of five discrete and two continuous action space deep reinforcement learning algorithms. We run the experiments in a vision-only high fidelity urban driving simulated environments. The results indicate that only some of the deep reinforcement learning algorithms perform consistently better across single and multi-agent scenarios when trained in a multi-agent-only setting.

摘要: 深度强化学习被用于在模拟驾驶环境中训练自主的和对抗性的汽车策略。由于各种强化学习算法的可用性很高，而且缺乏对不同驾驶场景的系统比较，我们不确定哪种算法在单代理和多代理驾驶环境下训练和测试自动驾驶汽车软件更有效。在基于视觉的自动驾驶中比较深度强化学习的基准框架将为培训更好的自动汽车驾驶政策打开可能性。此外，经过深度强化学习算法训练的自动驾驶汽车，众所周知容易受到对手的攻击。为了防范对抗性攻击，我们可以对自动驾驶汽车进行对抗性驾驶策略培训。然而，我们缺乏关于哪些深度强化学习算法可以作为能够有效测试自动驾驶汽车的好的对抗性代理的知识。为了应对这些挑战，我们提供了一个开放和可重用的基准测试框架，用于在单代理和多代理环境中对自主和对抗性驾驶的深度强化学习算法进行系统评估和比较分析。利用该框架，我们对五种离散动作空间和两种连续动作空间深度强化学习算法进行了比较研究。我们在视觉高保真的城市驾驶模拟环境中进行了实验。结果表明，只有一些深度强化学习算法在仅有多智能体的情况下训练时，在单智能体和多智能体场景中的表现一致较好。



## **43. Adversarial Deep Reinforcement Learning for Improving the Robustness of Multi-agent Autonomous Driving Policies**

对抗性深度强化学习提高多智能体自主驾驶策略的稳健性 cs.AI

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2112.11937v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstracts**: Autonomous cars are well known for being vulnerable to adversarial attacks that can compromise the safety of the car and pose danger to other road users. To effectively defend against adversaries, it is required to not only test autonomous cars for finding driving errors, but to improve the robustness of the cars to these errors. To this end, in this paper, we propose a two-step methodology for autonomous cars that consists of (i) finding failure states in autonomous cars by training the adversarial driving agent, and (ii) improving the robustness of autonomous cars by retraining them with effective adversarial inputs. Our methodology supports testing ACs in a multi-agent environment, where we train and compare adversarial car policy on two custom reward functions to test the driving control decision of autonomous cars. We run experiments in a vision-based high fidelity urban driving simulated environment. Our results show that adversarial testing can be used for finding erroneous autonomous driving behavior, followed by adversarial training for improving the robustness of deep reinforcement learning based autonomous driving policies. We demonstrate that the autonomous cars retrained using the effective adversarial inputs noticeably increase the performance of their driving policies in terms of reduced collision and offroad steering errors.

摘要: 众所周知，自动驾驶汽车容易受到对抗性攻击，这些攻击可能会危及汽车的安全，并对其他道路使用者构成危险。为了有效地防御对手，不仅需要测试自动驾驶汽车是否发现驾驶错误，还需要提高汽车对这些错误的稳健性。为此，在本文中，我们提出了一种自动驾驶汽车的两步方法，包括(I)通过训练对抗性驾驶主体来发现自动驾驶汽车中的故障状态，(Ii)通过对自动驾驶汽车进行有效的对抗性输入来重新训练它们来提高自动驾驶汽车的稳健性。我们的方法支持在多智能体环境中测试自动驾驶控制系统，在这个环境中，我们训练并比较两个定制奖励函数上的对抗性汽车策略，以测试自动驾驶汽车的驾驶控制决策。我们在基于视觉的高保真城市驾驶模拟环境中进行了实验。结果表明，对抗性测试可以用来发现错误的自主驾驶行为，然后通过对抗性训练来提高基于深度强化学习的自主驾驶策略的稳健性。我们证明，使用有效的对抗性输入进行再培训的自动驾驶汽车在减少碰撞和越野转向错误方面显著提高了其驾驶策略的性能。



## **44. fakeWeather: Adversarial Attacks for Deep Neural Networks Emulating Weather Conditions on the Camera Lens of Autonomous Systems**

虚假天气：对自主系统摄像机镜头上模拟天气条件的深度神经网络的敌意攻击 cs.LG

To appear at the 2022 International Joint Conference on Neural  Networks (IJCNN), at the 2022 IEEE World Congress on Computational  Intelligence (WCCI), July 2022, Padua, Italy

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13807v1)

**Authors**: Alberto Marchisio, Giovanni Caramia, Maurizio Martina, Muhammad Shafique

**Abstracts**: Recently, Deep Neural Networks (DNNs) have achieved remarkable performances in many applications, while several studies have enhanced their vulnerabilities to malicious attacks. In this paper, we emulate the effects of natural weather conditions to introduce plausible perturbations that mislead the DNNs. By observing the effects of such atmospheric perturbations on the camera lenses, we model the patterns to create different masks that fake the effects of rain, snow, and hail. Even though the perturbations introduced by our attacks are visible, their presence remains unnoticed due to their association with natural events, which can be especially catastrophic for fully-autonomous and unmanned vehicles. We test our proposed fakeWeather attacks on multiple Convolutional Neural Network and Capsule Network models, and report noticeable accuracy drops in the presence of such adversarial perturbations. Our work introduces a new security threat for DNNs, which is especially severe for safety-critical applications and autonomous systems.

摘要: 近年来，深度神经网络(DNN)在许多应用中取得了令人瞩目的性能，同时一些研究也增强了它们对恶意攻击的脆弱性。在本文中，我们模拟自然天气条件的影响来引入看似合理的扰动来误导DNN。通过观察这种大气扰动对相机镜头的影响，我们对图案进行建模，以创建不同的面具来模拟雨、雪和冰雹的影响。尽管我们的攻击带来的干扰是可见的，但由于它们与自然事件有关，它们的存在仍然没有被注意到，这对全自动驾驶和无人驾驶车辆来说可能是特别灾难性的。我们在多个卷积神经网络和胶囊网络模型上测试了我们提出的虚假天气攻击，并且报告了在存在这种对抗性扰动的情况下显著的准确率下降。我们的工作给DNN带来了新的安全威胁，这对安全关键型应用和自治系统尤其严重。



## **45. Face Morphing: Fooling a Face Recognition System Is Simple!**

人脸变形：愚弄人脸识别系统很简单！ cs.CV

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13796v1)

**Authors**: Stefan Hörmann, Tianlin Kong, Torben Teepe, Fabian Herzog, Martin Knoche, Gerhard Rigoll

**Abstracts**: State-of-the-art face recognition (FR) approaches have shown remarkable results in predicting whether two faces belong to the same identity, yielding accuracies between 92% and 100% depending on the difficulty of the protocol. However, the accuracy drops substantially when exposed to morphed faces, specifically generated to look similar to two identities. To generate morphed faces, we integrate a simple pretrained FR model into a generative adversarial network (GAN) and modify several loss functions for face morphing. In contrast to previous works, our approach and analyses are not limited to pairs of frontal faces with the same ethnicity and gender. Our qualitative and quantitative results affirm that our approach achieves a seamless change between two faces even in unconstrained scenarios. Despite using features from a simpler FR model for face morphing, we demonstrate that even recent FR systems struggle to distinguish the morphed face from both identities obtaining an accuracy of only 55-70%. Besides, we provide further insights into how knowing the FR system makes it particularly vulnerable to face morphing attacks.

摘要: 最新的人脸识别(FR)方法在预测两个人脸是否属于同一身份方面表现出了显著的效果，根据协议的难度，准确率在92%到100%之间。然而，当暴露在变形的面部时，准确度会大幅下降，这些变形的面部是专门生成的，看起来类似于两个身份。为了生成变形人脸，我们将一个简单的预先训练的FR模型集成到一个生成性对抗网络(GAN)中，并修改了几个用于人脸变形的损失函数。与前人的工作不同，我们的方法和分析并不局限于相同种族和性别的正面脸对。我们的定性和定量结果证实了我们的方法实现了两个人脸之间的无缝变化，即使在不受约束的场景中也是如此。尽管使用了更简单的FR模型的特征进行人脸变形，但我们证明了即使是最新的FR系统也很难区分变形后的人脸和两种身份，获得的准确率仅为55-70%。此外，我们还提供了关于了解FR系统如何使其特别容易受到变形攻击的进一步见解。



## **46. Adversarial attacks and defenses in Speaker Recognition Systems: A survey**

说话人识别系统中的对抗性攻击与防御 cs.CR

38pages, 2 figures, 2 tables. Journal of Systems Architecture,2022

**SubmitDate**: 2022-05-27    [paper-pdf](http://arxiv.org/pdf/2205.13685v1)

**Authors**: Jiahe Lan, Rui Zhang, Zheng Yan, Jie Wang, Yu Chen, Ronghui Hou

**Abstracts**: Speaker recognition has become very popular in many application scenarios, such as smart homes and smart assistants, due to ease of use for remote control and economic-friendly features. The rapid development of SRSs is inseparable from the advancement of machine learning, especially neural networks. However, previous work has shown that machine learning models are vulnerable to adversarial attacks in the image domain, which inspired researchers to explore adversarial attacks and defenses in Speaker Recognition Systems (SRS). Unfortunately, existing literature lacks a thorough review of this topic. In this paper, we fill this gap by performing a comprehensive survey on adversarial attacks and defenses in SRSs. We first introduce the basics of SRSs and concepts related to adversarial attacks. Then, we propose two sets of criteria to evaluate the performance of attack methods and defense methods in SRSs, respectively. After that, we provide taxonomies of existing attack methods and defense methods, and further review them by employing our proposed criteria. Finally, based on our review, we find some open issues and further specify a number of future directions to motivate the research of SRSs security.

摘要: 说话人识别由于易于远程控制和经济友好的特点，在智能家居和智能助理等许多应用场景中变得非常流行。支持向量机的快速发展离不开机器学习特别是神经网络的发展。然而，以往的工作表明，机器学习模型在图像领域容易受到对抗性攻击，这启发了研究人员探索说话人识别系统中的对抗性攻击和防御。遗憾的是，现有文献缺乏对这一主题的全面回顾。在本文中，我们通过对SRSS中的对抗性攻击和防御进行全面的调查来填补这一空白。我们首先介绍了SRSS的基本知识和与对抗性攻击相关的概念。然后，我们提出了两套标准来分别评价SRSS中攻击方法和防御方法的性能。之后，我们提供了现有的攻击方法和防御方法的分类，并使用我们提出的标准对它们进行了进一步的审查。最后，基于我们的回顾，我们发现了一些尚待解决的问题，并进一步指出了一些未来的方向，以推动SRSS安全的研究。



## **47. Sequential Nature of Recommender Systems Disrupts the Evaluation Process**

推荐系统的顺序性扰乱了评价过程 cs.IR

To Appear in Third International Workshop on Algorithmic Bias in  Search and Recommendation (Bias 2022)

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13681v1)

**Authors**: Ali Shirali

**Abstracts**: Datasets are often generated in a sequential manner, where the previous samples and intermediate decisions or interventions affect subsequent samples. This is especially prominent in cases where there are significant human-AI interactions, such as in recommender systems. To characterize the importance of this relationship across samples, we propose to use adversarial attacks on popular evaluation processes. We present sequence-aware boosting attacks and provide a lower bound on the amount of extra information that can be exploited from a confidential test set solely based on the order of the observed data. We use real and synthetic data to test our methods and show that the evaluation process on the MovieLense-100k dataset can be affected by $\sim1\%$ which is important when considering the close competition. Codes are publicly available.

摘要: 数据集通常是以顺序方式生成的，其中先前样本和中间决策或干预会影响后续样本。这在存在重大人类与人工智能交互的情况下尤其突出，例如在推荐系统中。为了表征这种关系在样本中的重要性，我们建议在流行的评估过程中使用对抗性攻击。我们提出了顺序感知的Boost攻击，并提供了仅基于观察到的数据的顺序可以从机密测试集中利用的额外信息量的下限。我们使用真实数据和合成数据来测试我们的方法，并表明在MovieLense-100k数据集上的评估过程会受到$Sim1$的影响，这在考虑到激烈的竞争时是很重要的。代码是公开提供的。



## **48. Membership Inference Attack Using Self Influence Functions**

基于自影响函数的隶属度推理攻击 cs.LG

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2205.13680v1)

**Authors**: Gilad Cohen, Raja Giryes

**Abstracts**: Member inference (MI) attacks aim to determine if a specific data sample was used to train a machine learning model. Thus, MI is a major privacy threat to models trained on private sensitive data, such as medical records. In MI attacks one may consider the black-box settings, where the model's parameters and activations are hidden from the adversary, or the white-box case where they are available to the attacker. In this work, we focus on the latter and present a novel MI attack for it that employs influence functions, or more specifically the samples' self-influence scores, to perform the MI prediction. We evaluate our attack on CIFAR-10, CIFAR-100, and Tiny ImageNet datasets, using versatile architectures such as AlexNet, ResNet, and DenseNet. Our attack method achieves new state-of-the-art results for both training with and without data augmentations. Code is available at https://github.com/giladcohen/sif_mi_attack.

摘要: 成员推理(MI)攻击旨在确定是否使用特定数据样本来训练机器学习模型。因此，MI是对针对私人敏感数据(如医疗记录)进行培训的模型的主要隐私威胁。在MI攻击中，可以考虑黑盒设置，在黑盒设置中，模型的参数和激活对对手隐藏，或者在白盒情况下，攻击者可以使用它们。在这项工作中，我们关注后者，并提出了一种新的MI攻击，该攻击使用影响函数，或者更具体地说，样本的自我影响分数来执行MI预测。我们使用多种架构，如AlexNet、ResNet和DenseNet，评估我们对CIFAR-10、CIFAR-100和微小ImageNet数据集的攻击。我们的攻击方法在有数据增强和没有数据增强的训练中都获得了新的最先进的结果。代码可在https://github.com/giladcohen/sif_mi_attack.上找到



## **49. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

加密货币使用的对等网络匿名方案的匿名性研究 cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2201.11860v2)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstracts**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.

摘要: 通过利用其对等网络上的网络级通信，加密货币系统可能会受到反匿名化攻击。在对等网络中控制一组串通节点的敌手可以观察正在交换的交易并推断所涉及的各方。因此，各种网络匿名方案被提出来缓解这一问题，一些解决方案提供了理论上的匿名性保证。在这项工作中，我们对这种对等网络匿名解决方案进行建模，并评估它们的匿名性保证。为此，我们提出了一个新的框架，它使用贝叶斯推理来获得将事务链接到可能的发起者的概率分布。我们使用这些分布来表征交易匿名性，使用熵作为对发起者身份的敌意不确定性的度量。特别是，我们对蒲公英、蒲公英++和闪电网络进行了建模。我们研究了不同的配置，并证明它们都不能为用户提供可接受的匿名性。例如，我们的分析表明，在广泛部署的闪电网络中，通过1%的策略选择合谋节点，对手可以唯一地确定网络中约50%的总交易的发起者。在蒲公英中，一个控制了15%节点的对手平均只有8个可能的发起者中存在不确定性。此外，我们观察到，由于蒲公英和蒲公英++的设计方式，增加网络规模并不对应于潜在发起者匿名性集合的增加。令人担忧的是，我们对Lightning Network的纵向分析揭示了一个相反的趋势--随着网络的增长，总体匿名性下降。



## **50. Towards Practical Deployment-Stage Backdoor Attack on Deep Neural Networks**

走向实用化部署--深度神经网络的阶段后门攻击 cs.CR

**SubmitDate**: 2022-05-26    [paper-pdf](http://arxiv.org/pdf/2111.12965v2)

**Authors**: Xiangyu Qi, Tinghao Xie, Ruizhe Pan, Jifeng Zhu, Yong Yang, Kai Bu

**Abstracts**: One major goal of the AI security community is to securely and reliably produce and deploy deep learning models for real-world applications. To this end, data poisoning based backdoor attacks on deep neural networks (DNNs) in the production stage (or training stage) and corresponding defenses are extensively explored in recent years. Ironically, backdoor attacks in the deployment stage, which can often happen in unprofessional users' devices and are thus arguably far more threatening in real-world scenarios, draw much less attention of the community. We attribute this imbalance of vigilance to the weak practicality of existing deployment-stage backdoor attack algorithms and the insufficiency of real-world attack demonstrations. To fill the blank, in this work, we study the realistic threat of deployment-stage backdoor attacks on DNNs. We base our study on a commonly used deployment-stage attack paradigm -- adversarial weight attack, where adversaries selectively modify model weights to embed backdoor into deployed DNNs. To approach realistic practicality, we propose the first gray-box and physically realizable weights attack algorithm for backdoor injection, namely subnet replacement attack (SRA), which only requires architecture information of the victim model and can support physical triggers in the real world. Extensive experimental simulations and system-level real-world attack demonstrations are conducted. Our results not only suggest the effectiveness and practicality of the proposed attack algorithm, but also reveal the practical risk of a novel type of computer virus that may widely spread and stealthily inject backdoor into DNN models in user devices. By our study, we call for more attention to the vulnerability of DNNs in the deployment stage.

摘要: AI安全社区的一大目标是安全可靠地为现实世界的应用程序生成和部署深度学习模型。为此，基于数据中毒的深度神经网络(DNN)在生产阶段(或训练阶段)的后门攻击以及相应的防御措施近年来得到了广泛的研究。具有讽刺意味的是，部署阶段的后门攻击通常会发生在非专业用户的设备上，因此在现实世界中可以说威胁要大得多，但社区对此的关注要少得多。我们将这种不平衡的警觉性归因于现有部署阶段后门攻击算法的实用性较弱，以及现实世界攻击演示的不足。为了填补这一空白，在这项工作中，我们研究了部署阶段后门攻击对DNN的现实威胁。我们的研究基于一种常用的部署阶段攻击范式--对抗性权重攻击，在这种攻击中，攻击者选择性地修改模型权重，将后门嵌入到部署的DNN中。为了更接近实际应用，我们提出了第一种灰盒和物理可实现权重的后门注入攻击算法，即子网替换攻击算法(SRA)，该算法只需要受害者模型的体系结构信息，并且能够支持现实世界中的物理触发。进行了广泛的实验模拟和系统级真实世界攻击演示。我们的结果不仅表明了所提出的攻击算法的有效性和实用性，还揭示了一种新型计算机病毒的实际风险，这种病毒可能会广泛传播并悄悄地向用户设备中的DNN模型注入后门。通过我们的研究，我们呼吁更多地关注DNN在部署阶段的脆弱性。



