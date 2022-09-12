# Latest Adversarial Attack Papers
**update at 2022-09-13 06:31:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Examples in Constrained Domains**

受限领域中的对抗性例子 cs.CR

Accepted to IOS Press Journal of Computer Security

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2011.01183v3)

**Authors**: Ryan Sheatsley, Nicolas Papernot, Michael Weisman, Gunjan Verma, Patrick McDaniel

**Abstracts**: Machine learning algorithms have been shown to be vulnerable to adversarial manipulation through systematic modification of inputs (e.g., adversarial examples) in domains such as image recognition. Under the default threat model, the adversary exploits the unconstrained nature of images; each feature (pixel) is fully under control of the adversary. However, it is not clear how these attacks translate to constrained domains that limit which and how features can be modified by the adversary (e.g., network intrusion detection). In this paper, we explore whether constrained domains are less vulnerable than unconstrained domains to adversarial example generation algorithms. We create an algorithm for generating adversarial sketches: targeted universal perturbation vectors which encode feature saliency within the envelope of domain constraints. To assess how these algorithms perform, we evaluate them in constrained (e.g., network intrusion detection) and unconstrained (e.g., image recognition) domains. The results demonstrate that our approaches generate misclassification rates in constrained domains that were comparable to those of unconstrained domains (greater than 95%). Our investigation shows that the narrow attack surface exposed by constrained domains is still sufficiently large to craft successful adversarial examples; and thus, constraints do not appear to make a domain robust. Indeed, with as little as five randomly selected features, one can still generate adversarial examples.

摘要: 已经证明，机器学习算法通过系统地修改诸如图像识别等领域中的输入(例如，对抗性例子)而容易受到对抗性操纵。在默认威胁模型下，对手利用图像的不受限制的性质；每个特征(像素)都完全在对手的控制之下。然而，目前尚不清楚这些攻击如何转化为受限的域，从而限制攻击者可以修改哪些功能以及如何修改(例如，网络入侵检测)。在本文中，我们探讨了受限域是否比非约束域更不容易受到敌意示例生成算法的影响。我们创建了一种生成对抗性草图的算法：目标通用扰动向量，它在领域约束的包络内编码特征显著。为了评估这些算法的性能，我们在受限(例如，网络入侵检测)和非受限(例如，图像识别)域中对它们进行评估。结果表明，我们的方法在受限领域产生的错误分类率与非约束领域相当(大于95%)。我们的调查表明，受约束域暴露的狭窄攻击面仍然足够大，足以制作成功的敌意示例；因此，约束似乎不会使域变得健壮。事实上，只需随机选择五个特征，就仍然可以生成对抗性的例子。



## **2. Robust-by-Design Classification via Unitary-Gradient Neural Networks**

基于么正梯度神经网络的稳健设计分类 cs.LG

Under review

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.04293v1)

**Authors**: Fabio Brau, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: The use of neural networks in safety-critical systems requires safe and robust models, due to the existence of adversarial attacks. Knowing the minimal adversarial perturbation of any input x, or, equivalently, knowing the distance of x from the classification boundary, allows evaluating the classification robustness, providing certifiable predictions. Unfortunately, state-of-the-art techniques for computing such a distance are computationally expensive and hence not suited for online applications. This work proposes a novel family of classifiers, namely Signed Distance Classifiers (SDCs), that, from a theoretical perspective, directly output the exact distance of x from the classification boundary, rather than a probability score (e.g., SoftMax). SDCs represent a family of robust-by-design classifiers. To practically address the theoretical requirements of a SDC, a novel network architecture named Unitary-Gradient Neural Network is presented. Experimental results show that the proposed architecture approximates a signed distance classifier, hence allowing an online certifiable classification of x at the cost of a single inference.

摘要: 由于存在对抗性攻击，在安全关键系统中使用神经网络需要安全和健壮的模型。知道任何输入x的最小对抗性扰动，或者，等价地，知道x到分类边界的距离，允许评估分类稳健性，提供可证明的预测。不幸的是，用于计算这种距离的最先进的技术计算成本很高，因此不适合在线应用。这项工作提出了一类新的分类器，即符号距离分类器(SDCS)，从理论上讲，它直接输出x到分类边界的准确距离，而不是概率分数(例如SoftMax)。SDC代表了一系列稳健的按设计分类的分类器。为了满足SDC的理论要求，提出了一种新的网络体系结构--酉梯度神经网络。实验结果表明，该体系结构接近于符号距离分类器，从而允许以单一推理为代价对x进行在线可证明分类。



## **3. Improving Out-of-Distribution Detection via Epistemic Uncertainty Adversarial Training**

通过认知不确定性对抗性训练改进失配检测 cs.LG

8 pages, 5 figures

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2209.03148v2)

**Authors**: Derek Everett, Andre T. Nguyen, Luke E. Richards, Edward Raff

**Abstracts**: The quantification of uncertainty is important for the adoption of machine learning, especially to reject out-of-distribution (OOD) data back to human experts for review. Yet progress has been slow, as a balance must be struck between computational efficiency and the quality of uncertainty estimates. For this reason many use deep ensembles of neural networks or Monte Carlo dropout for reasonable uncertainty estimates at relatively minimal compute and memory. Surprisingly, when we focus on the real-world applicable constraint of $\leq 1\%$ false positive rate (FPR), prior methods fail to reliably detect OOD samples as such. Notably, even Gaussian random noise fails to trigger these popular OOD techniques. We help to alleviate this problem by devising a simple adversarial training scheme that incorporates an attack of the epistemic uncertainty predicted by the dropout ensemble. We demonstrate this method improves OOD detection performance on standard data (i.e., not adversarially crafted), and improves the standardized partial AUC from near-random guessing performance to $\geq 0.75$.

摘要: 不确定性的量化对于机器学习的采用非常重要，特别是对于拒绝将分布外(OOD)数据返回给人类专家进行审查。然而，进展缓慢，因为必须在计算效率和不确定性估计的质量之间取得平衡。出于这个原因，许多人使用神经网络的深度集成或蒙特卡罗退学来在相对最小的计算和内存下进行合理的不确定性估计。令人惊讶的是，当我们关注现实世界中可应用的假阳性率(FPR)约束时，现有方法无法可靠地检测出OOD样本。值得注意的是，即使是高斯随机噪声也无法触发这些流行的OOD技术。我们通过设计一个简单的对抗性训练方案来帮助缓解这个问题，该方案结合了对辍学生群体预测的认知不确定性的攻击。我们证明了该方法提高了对标准数据的OOD检测性能(即，不是恶意定制的)，并将标准化的部分AUC从近乎随机猜测的性能提高到0.75美元。



## **4. Harnessing Perceptual Adversarial Patches for Crowd Counting**

利用感知对抗性斑块进行人群计数 cs.CV

**SubmitDate**: 2022-09-09    [paper-pdf](http://arxiv.org/pdf/2109.07986v2)

**Authors**: Shunchang Liu, Jiakai Wang, Aishan Liu, Yingwei Li, Yijie Gao, Xianglong Liu, Dacheng Tao

**Abstracts**: Crowd counting, which has been widely adopted for estimating the number of people in safety-critical scenes, is shown to be vulnerable to adversarial examples in the physical world (e.g., adversarial patches). Though harmful, adversarial examples are also valuable for evaluating and better understanding model robustness. However, existing adversarial example generation methods for crowd counting lack strong transferability among different black-box models, which limits their practicability for real-world systems. Motivated by the fact that attacking transferability is positively correlated to the model-invariant characteristics, this paper proposes the Perceptual Adversarial Patch (PAP) generation framework to tailor the adversarial perturbations for crowd counting scenes using the model-shared perceptual features. Specifically, we handcraft an adaptive crowd density weighting approach to capture the invariant scale perception features across various models and utilize the density guided attention to capture the model-shared position perception. Both of them are demonstrated to improve the attacking transferability of our adversarial patches. Extensive experiments show that our PAP could achieve state-of-the-art attacking performance in both the digital and physical world, and outperform previous proposals by large margins (at most +685.7 MAE and +699.5 MSE). Besides, we empirically demonstrate that adversarial training with our PAP can benefit the performance of vanilla models in alleviating several practical challenges in crowd counting scenarios, including generalization across datasets (up to -376.0 MAE and -354.9 MSE) and robustness towards complex backgrounds (up to -10.3 MAE and -16.4 MSE).

摘要: 人群计数被广泛用于估计安全关键场景中的人数，但在现实世界中，它很容易受到对抗性例子的影响(例如，对抗性补丁)。对抗性例子虽然有害，但对于评估和更好地理解模型的健壮性也是有价值的。然而，现有的人群计数对抗性实例生成方法在不同的黑盒模型之间缺乏很强的可移植性，这限制了它们在现实系统中的实用性。基于攻击的可转移性与模型不变特性正相关这一事实，提出了感知对抗性补丁(PAP)生成框架，利用模型共享的感知特征来定制人群计数场景中的对抗性扰动。具体地说，我们手工设计了一种自适应的人群密度加权方法来捕捉各种模型上的不变尺度感知特征，并利用密度引导注意力来捕捉模型共享的位置感知。它们都被证明可以提高我们对手补丁的攻击可转移性。大量的实验表明，我们的PAP在数字和物理世界都可以达到最先进的攻击性能，并且比以前的方案有很大的优势(最多+685.7 MAE和+699.5 MSE)。此外，我们的经验证明，使用我们的PAP进行对抗性训练可以帮助Vanilla模型在缓解人群计数场景中的几个实际挑战方面的性能，包括跨数据集的泛化(高达-376.0 MAE和-354.9 MSE)以及对复杂背景的稳健性(高达-10.3MAE和-16.4MSE)。



## **5. Uncovering the Connection Between Differential Privacy and Certified Robustness of Federated Learning against Poisoning Attacks**

揭示差分隐私与联合学习对中毒攻击的认证健壮性之间的联系 cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04030v1)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Bo Li

**Abstracts**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As the local training data come from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is always trained in a differentially private way (DPFL). Thus, in this paper, we ask: Can we leverage the innate privacy property of DPFL to provide certified robustness against poisoning attacks? Can we further improve the privacy of FL to improve such certification? We first investigate both user-level and instance-level privacy of FL and propose novel mechanisms to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack cost for DPFL on both levels. Theoretically, we prove the certified robustness of DPFL under a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of attacks on different datasets. We show that DPFL with a tighter privacy guarantee always provides stronger robustness certification in terms of certified attack cost, but the optimal certified prediction is achieved under a proper balance between privacy protection and utility loss.

摘要: 联合学习(FL)提供了一种有效的范例来联合训练利用来自分布式用户的数据的全局模型。由于本地训练数据来自可能不可信的不同用户，多项研究表明FL很容易受到中毒攻击。同时，为了保护本地用户的隐私，FL总是以一种不同的私人方式进行培训(DPFL)。因此，在这篇文章中，我们问：我们能否利用DPFL固有的隐私属性来提供经过认证的针对中毒攻击的健壮性？我们能否进一步改善FL的隐私，以提高此类认证？我们首先研究了FL的用户级隐私和实例级隐私，并提出了新的机制来实现改进的实例级隐私。然后，我们提供了两个健壮性认证标准：DPFL在两个级别上的认证预测和认证攻击成本。理论上，我们证明了DPFL在有限数量的敌意用户或实例下的证明的健壮性。在经验上，我们在不同数据集的一系列攻击下进行了广泛的实验来验证我们的理论。我们发现，在认证攻击代价方面，具有更严格隐私保障的DPFL总是提供更强的健壮性认证，但最优认证预测是在隐私保护和效用损失之间取得适当平衡的情况下实现的。



## **6. Evaluating the Security of Aircraft Systems**

评估飞机系统的安全性 cs.CR

38 pages,

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.04028v1)

**Authors**: Edan Habler, Ron Bitton, Asaf Shabtai

**Abstracts**: The sophistication and complexity of cyber attacks and the variety of targeted platforms have been growing in recent years. Various adversaries are abusing an increasing range of platforms, e.g., enterprise platforms, mobile phones, PCs, transportation systems, and industrial control systems. In recent years, we have witnessed various cyber attacks on transportation systems, including attacks on ports, airports, and trains. It is only a matter of time before transportation systems become a more common target of cyber attackers. Due to the enormous potential damage inherent in attacking vehicles carrying many passengers and the lack of security measures applied in traditional airborne systems, the vulnerability of aircraft systems is one of the most concerning topics in the vehicle security domain. This paper provides a comprehensive review of aircraft systems and components and their various networks, emphasizing the cyber threats they are exposed to and the impact of a cyber attack on these components and networks and the essential capabilities of the aircraft. In addition, we present a comprehensive and in-depth taxonomy that standardizes the knowledge and understanding of cyber security in the avionics field from an adversary's perspective. The taxonomy divides techniques into relevant categories (tactics) reflecting the various phases of the adversarial attack lifecycle and maps existing attacks according to the MITRE ATT&CK methodology. Furthermore, we analyze the security risks among the various systems according to the potential threat actors and categorize the threats based on STRIDE threat model. Future work directions are presented as guidelines for industry and academia.

摘要: 近年来，网络攻击的复杂性和复杂性以及目标平台的多样性一直在增长。各种对手正在滥用越来越多的平台，例如企业平台、移动电话、PC、交通系统和工业控制系统。近年来，我们目睹了针对交通系统的各种网络攻击，包括对港口、机场和火车的攻击。交通系统成为网络攻击者更常见的目标只是个时间问题。由于攻击载客车辆固有的巨大潜在危害，以及传统机载系统缺乏安全措施，飞机系统的脆弱性是车辆安全领域最受关注的话题之一。本文对飞机系统和部件及其各种网络进行了全面的回顾，强调了它们所面临的网络威胁，以及网络攻击对这些部件和网络以及飞机的基本能力的影响。此外，我们提出了一个全面和深入的分类，从对手的角度标准化了对航空电子领域网络安全的知识和理解。该分类将技术划分为相关类别(战术)，反映对抗性攻击生命周期的不同阶段，并根据MITRE ATT&CK方法映射现有攻击。在此基础上，根据潜在威胁主体分析了各个系统之间的安全风险，并基于STRIDE威胁模型对威胁进行了分类。提出了未来的工作方向，作为产业界和学术界的指导方针。



## **7. A Survey of Machine Unlearning**

机器遗忘研究综述 cs.LG

arXiv admin note: text overlap with arXiv:2109.13398,  arXiv:2109.08266 by other authors. author note: fixed some overlaps

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.02299v3)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.

摘要: 几十年来，计算机系统保存着大量的个人数据。一方面，这样的数据丰富使人工智能(AI)，特别是机器学习(ML)模型取得了突破。另一方面，它会威胁用户的隐私，削弱人类与AI之间的信任。最近的法规要求，一般情况下，可以从计算机系统中删除关于用户的私人信息，特别是在请求时可以从ML模型中删除用户的私人信息(例如，“被遗忘权”)。虽然从后端数据库中删除数据应该很简单，但在人工智能环境中这是不够的，因为ML模型经常“记住”旧数据。现有的对抗性攻击证明，我们可以从训练好的模型中学习训练数据的私人成员或属性。这种现象呼唤一种新的范式，即机器遗忘，以使ML模型忘记特定的数据。事实证明，由于缺乏通用的框架和资源，最近关于机器遗忘的研究并不能完全解决这个问题。在这篇调查论文中，我们试图对机器遗忘的定义、场景、机制和应用进行全面的调查。具体地说，作为最新研究的分类集合，我们希望为那些寻求机器遗忘及其各种公式、设计要求、移除请求、算法和在各种ML应用中使用的入门知识的人提供广泛的参考。此外，我们希望概述该范式中的主要发现和趋势，并强调尚未看到机器遗忘应用的新研究领域，但仍可能受益匪浅。我们希望这项调查为ML研究人员以及那些寻求创新隐私技术的人提供有价值的参考。我们的资源在https://github.com/tamlhp/awesome-machine-unlearning.



## **8. SafeNet: The Unreasonable Effectiveness of Ensembles in Private Collaborative Learning**

SafeNet：私人合作学习中合奏的不合理有效性 cs.CR

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2205.09986v2)

**Authors**: Harsh Chaudhari, Matthew Jagielski, Alina Oprea

**Abstracts**: Secure multiparty computation (MPC) has been proposed to allow multiple mutually distrustful data owners to jointly train machine learning (ML) models on their combined data. However, by design, MPC protocols faithfully compute the training functionality, which the adversarial ML community has shown to leak private information and can be tampered with in poisoning attacks. In this work, we argue that model ensembles, implemented in our framework called SafeNet, are a highly MPC-amenable way to avoid many adversarial ML attacks. The natural partitioning of data amongst owners in MPC training allows this approach to be highly scalable at training time, provide provable protection from poisoning attacks, and provably defense against a number of privacy attacks. We demonstrate SafeNet's efficiency, accuracy, and resilience to poisoning on several machine learning datasets and models trained in end-to-end and transfer learning scenarios. For instance, SafeNet reduces backdoor attack success significantly, while achieving $39\times$ faster training and $36 \times$ less communication than the four-party MPC framework of Dalskov et al. Our experiments show that ensembling retains these benefits even in many non-iid settings. The simplicity, cheap setup, and robustness properties of ensembling make it a strong first choice for training ML models privately in MPC.

摘要: 安全多方计算(MPC)已被提出，以允许多个相互不信任的数据所有者联合训练机器学习(ML)模型。然而，通过设计，MPC协议忠实地计算训练功能，敌对的ML社区已经证明这些功能会泄露私人信息，并且可以在中毒攻击中篡改。在这项工作中，我们认为在我们的框架SafeNet中实现的模型集成是一种高度兼容MPC的方法，可以避免许多对抗性的ML攻击。在MPC培训中，所有者之间的数据自然分区允许此方法在培训期间高度可扩展，提供针对中毒攻击的可证明保护，并可证明针对多个隐私攻击的防御。我们在几个在端到端和转移学习场景中训练的机器学习数据集和模型上展示了SafeNet的效率、准确性和对中毒的弹性。例如，SafeNet显著降低了后门攻击的成功率，同时实现了比Dalskov等人的四方MPC框架快39倍的培训和36倍的通信。我们的实验表明，即使在许多非IID环境中，集合也保留了这些好处。集成的简单性、设置成本和健壮性使其成为在MPC中私下训练ML模型的首选方法。



## **9. Incorporating Locality of Images to Generate Targeted Transferable Adversarial Examples**

结合图像的局部性生成目标可转移的对抗性实例 cs.CV

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2209.03716v1)

**Authors**: Zhipeng Wei, Jingjing Chen, Zuxuan Wu, Yu-Gang Jiang

**Abstracts**: Despite that leveraging the transferability of adversarial examples can attain a fairly high attack success rate for non-targeted attacks, it does not work well in targeted attacks since the gradient directions from a source image to a targeted class are usually different in different DNNs. To increase the transferability of target attacks, recent studies make efforts in aligning the feature of the generated adversarial example with the feature distributions of the targeted class learned from an auxiliary network or a generative adversarial network. However, these works assume that the training dataset is available and require a lot of time to train networks, which makes it hard to apply to real-world scenarios. In this paper, we revisit adversarial examples with targeted transferability from the perspective of universality and find that highly universal adversarial perturbations tend to be more transferable. Based on this observation, we propose the Locality of Images (LI) attack to improve targeted transferability. Specifically, instead of using the classification loss only, LI introduces a feature similarity loss between intermediate features from adversarial perturbed original images and randomly cropped images, which makes the features from adversarial perturbations to be more dominant than that of benign images, hence improving targeted transferability. Through incorporating locality of images into optimizing perturbations, the LI attack emphasizes that targeted perturbations should be universal to diverse input patterns, even local image patches. Extensive experiments demonstrate that LI can achieve high success rates for transfer-based targeted attacks. On attacking the ImageNet-compatible dataset, LI yields an improvement of 12\% compared with existing state-of-the-art methods.

摘要: 尽管利用对抗性例子的可转移性可以在非目标攻击中获得相当高的攻击成功率，但它在目标攻击中不能很好地工作，因为在不同的DNN中，从源图像到目标类别的梯度方向通常是不同的。为了提高目标攻击的可转移性，最近的研究致力于将生成的对抗性实例的特征与从辅助网络或生成性对抗性网络学习的目标类的特征分布相匹配。然而，这些工作假设训练数据集是可用的，并且需要大量的时间来训练网络，这使得很难将其应用于现实世界的场景。在本文中，我们从普遍性的角度重新考察了具有针对性可转移性的对抗性例子，发现具有高度普遍性的对抗性扰动往往更具可转移性。基于这一观察结果，我们提出了图像局部性(LI)攻击来提高目标可转移性。具体地说，Li不是只使用分类损失，而是在来自对抗性扰动的原始图像和随机裁剪图像的中间特征之间引入了特征相似性损失，使得来自对抗性扰动的特征比良性图像的特征更具优势，从而提高了目标可转移性。通过将图像的局部性引入优化扰动，LI攻击强调目标扰动对于不同的输入模式应该是通用的，甚至对局部图像块也是如此。广泛的实验证明，李灿对基于转会的靶向攻击取得了很高的成功率。在攻击与ImageNet兼容的数据集方面，与现有的最先进方法相比，LI的性能提高了12%。



## **10. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

ICML 2022 Workshop paper accepted at AdvML Frontiers

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2206.06761v4)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **11. Feature Importance Guided Attack: A Model Agnostic Adversarial Attack**

特征重要性制导攻击：一种不可知的对抗性攻击模型 cs.LG

**SubmitDate**: 2022-09-08    [paper-pdf](http://arxiv.org/pdf/2106.14815v2)

**Authors**: Gilad Gressel, Niranjan Hegde, Archana Sreekumar, Rishikumar Radhakrishnan, Kalyani Harikumar, Anjali S., Michael Darling

**Abstracts**: Research in adversarial learning has primarily focused on homogeneous unstructured datasets, which often map into the problem space naturally. Inverting a feature space attack on heterogeneous datasets into the problem space is much more challenging, particularly the task of finding the perturbation to perform. This work presents a formal search strategy: the `Feature Importance Guided Attack' (FIGA), which finds perturbations in the feature space of heterogeneous tabular datasets to produce evasion attacks. We first demonstrate FIGA in the feature space and then in the problem space. FIGA assumes no prior knowledge of the defending model's learning algorithm and does not require any gradient information. FIGA assumes knowledge of the feature representation and the mean feature values of defending model's dataset. FIGA leverages feature importance rankings by perturbing the most important features of the input in the direction of the target class. While FIGA is conceptually similar to other work which uses feature selection processes (e.g., mimicry attacks), we formalize an attack algorithm with three tunable parameters and investigate the strength of FIGA on tabular datasets. We demonstrate the effectiveness of FIGA by evading phishing detection models trained on four different tabular phishing datasets and one financial dataset with an average success rate of 94%. We extend FIGA to the phishing problem space by limiting the possible perturbations to be valid and feasible in the phishing domain. We generate valid adversarial phishing sites that are visually identical to their unperturbed counterpart and use them to attack six tabular ML models achieving a 13.05% average success rate.

摘要: 对抗性学习的研究主要集中在同质的非结构化数据集上，这些数据集往往自然地映射到问题空间。将异类数据集上的特征空间攻击转化到问题空间中的挑战要大得多，特别是找到要执行的扰动的任务。该工作提出了一种形式化的搜索策略：特征重要性制导攻击(FIGA)，它在异类表格数据集的特征空间中发现扰动，从而产生规避攻击。我们首先在特征空间中证明FIGA，然后在问题空间中证明FIGA。Figa不假定防御模型的学习算法的先验知识，也不需要任何梯度信息。FigA假设已知防御模型数据集的特征表示和平均特征值。FigA通过在目标类的方向上干扰输入的最重要特征来利用特征重要性排名。虽然FIGA在概念上类似于其他使用特征选择过程的工作(例如，模仿攻击)，但我们使用三个可调参数来形式化攻击算法，并研究了FIGA在表格数据集上的优势。我们通过在四个不同的表格钓鱼数据集和一个金融数据集上训练的钓鱼检测模型，证明了FIGA的有效性，平均成功率为94%。通过限制可能的扰动在钓鱼领域是有效和可行的，我们将FIGA扩展到钓鱼问题空间。我们生成了有效的敌意钓鱼网站，这些网站在视觉上与未受干扰的网站相同，并使用它们攻击六个表格ML模型，平均成功率为13.05%。



## **12. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

AdaptOver：蜂窝网络中的自适应遮蔽攻击 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2106.05039v3)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Roeschlin, Srdjan Capkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver - a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.

摘要: 在蜂窝网络中，对移动设备和核心网络之间的通信链路的攻击会严重影响隐私和可用性。到目前为止，伪基站已经被要求执行这样的攻击。由于它们需要持续高的输出功率来吸引受害者，因此它们的射程有限，运营商和用户智能手机上的专用应用程序都很容易检测到它们。介绍了一种专为LTE和5G-NSA蜂窝网络设计的MITM攻击系统--AdaptOver。AdaptOver允许对手在网络和移动设备之间的任一方向上通过空中解码、掩盖(替换)和注入任意消息。使用遮蔽，AdaptOver可以触发UE以纯文本形式传输其永久标识符(IMSI)，从而导致持续($\geq$12h)DoS或隐私泄露。这些攻击可以针对一个小区内的所有用户，也可以根据受害者的电话号码专门针对受害者。我们使用软件定义的无线电和低成本的放大设置来实现AdaptOver。我们使用各种智能手机演示了这些攻击对实时运行的LTE和5G-NSA网络的影响和实用性。我们的实验表明，AdaptOver可以对距离攻击者3.8公里以上的受害者发动攻击。考虑到其实用性和效率，AdaptOver表明，专注于伪基站的现有对策不再足够，标志着蜂窝网络安全机制设计的范式转变。



## **13. Combing for Credentials: Active Pattern Extraction from Smart Reply**

梳理凭据：从智能回复中提取活动模式 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.10802v2)

**Authors**: Bargav Jayaraman, Esha Ghosh, Melissa Chase, Sambuddha Roy, Huseyin Inan, Wei Dai, David Evans

**Abstracts**: With the wide availability of large pre-trained language models such as GPT-2 and BERT, the recent trend has been to fine-tune a pre-trained model to achieve state-of-the-art performance on a downstream task. One natural example is the "Smart Reply" application where a pre-trained model is tuned to provide suggested responses for a given query message. Since these models are often tuned using sensitive data such as emails or chat transcripts, it is important to understand and mitigate the risk that the model leaks its tuning data. We investigate potential information leakage vulnerabilities in a typical Smart Reply pipeline and introduce a new type of active extraction attack that exploits canonical patterns in text containing sensitive data. We show experimentally that it is possible for an adversary to extract sensitive user information present in the training data. We explore potential mitigation strategies and demonstrate empirically how differential privacy appears to be an effective defense mechanism to such pattern extraction attacks.

摘要: 随着GPT-2和BERT等大型预训练语言模型的广泛使用，最近的趋势是微调预训练模型，以在下游任务中实现最先进的性能。一个自然的例子是“智能回复”应用程序，其中预先训练的模型被调优以提供对给定查询消息的建议响应。由于这些模型通常使用电子邮件或聊天记录等敏感数据进行调整，因此了解并降低模型泄露其调整数据的风险非常重要。我们调查了一个典型的智能回复管道中潜在的信息泄漏漏洞，并引入了一种新型的主动提取攻击，该攻击利用了包含敏感数据的文本中的规范模式。我们的实验表明，对手有可能提取训练数据中存在的敏感用户信息。我们探索了潜在的缓解策略，并经验地证明了差异隐私似乎是应对此类模式提取攻击的一种有效防御机制。



## **14. Inferring Sensitive Attributes from Model Explanations**

从模型解释中推断敏感属性 cs.CR

ACM CIKM 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2208.09967v2)

**Authors**: Vasisht Duddu, Antoine Boutet

**Abstracts**: Model explanations provide transparency into a trained machine learning model's blackbox behavior to a model builder. They indicate the influence of different input attributes to its corresponding model prediction. The dependency of explanations on input raises privacy concerns for sensitive user data. However, current literature has limited discussion on privacy risks of model explanations.   We focus on the specific privacy risk of attribute inference attack wherein an adversary infers sensitive attributes of an input (e.g., race and sex) given its model explanations. We design the first attribute inference attack against model explanations in two threat models where model builder either (a) includes the sensitive attributes in training data and input or (b) censors the sensitive attributes by not including them in the training data and input.   We evaluate our proposed attack on four benchmark datasets and four state-of-the-art algorithms. We show that an adversary can successfully infer the value of sensitive attributes from explanations in both the threat models accurately. Moreover, the attack is successful even by exploiting only the explanations corresponding to sensitive attributes. These suggest that our attack is effective against explanations and poses a practical threat to data privacy.   On combining the model predictions (an attack surface exploited by prior attacks) with explanations, we note that the attack success does not improve. Additionally, the attack success on exploiting model explanations is better compared to exploiting only model predictions. These suggest that model explanations are a strong attack surface to exploit for an adversary.

摘要: 模型解释为模型构建者提供了对经过训练的机器学习模型的黑箱行为的透明性。它们表明了不同的输入属性对其相应模型预测的影响。解释对输入的依赖引发了对敏感用户数据的隐私问题。然而，目前的文献对模型解释的隐私风险的讨论有限。我们专注于属性推理攻击的特定隐私风险，其中对手根据输入的模型解释推断输入的敏感属性(例如，种族和性别)。我们针对两个威胁模型中的模型解释设计了第一个属性推理攻击，在这两个模型中，建模者或者(A)在训练数据和输入中包括敏感属性，或者(B)通过在训练数据和输入中不包括敏感属性来审查敏感属性。我们在四个基准数据集和四个最先进的算法上评估了我们提出的攻击。我们表明，攻击者可以从两种威胁模型中的解释中准确地推断出敏感属性的值。此外，即使只利用与敏感属性对应的解释，攻击也是成功的。这些都表明，我们的攻击针对解释是有效的，并对数据隐私构成了实际威胁。在将模型预测(先前攻击所利用的攻击面)与解释相结合时，我们注意到攻击成功率并没有提高。此外，与仅利用模型预测相比，利用模型解释的攻击成功更好。这些都表明，模型解释是对手可以利用的强大攻击面。



## **15. Securing the Spike: On the Transferabilty and Security of Spiking Neural Networks to Adversarial Examples**

保护尖峰：尖峰神经网络对对抗性例子的可传递性和安全性 cs.NE

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03358v1)

**Authors**: Nuo Xu, Kaleel Mahmood, Haowen Fang, Ethan Rathbun, Caiwen Ding, Wujie Wen

**Abstracts**: Spiking neural networks (SNNs) have attracted much attention for their high energy efficiency and for recent advances in their classification performance. However, unlike traditional deep learning approaches, the analysis and study of the robustness of SNNs to adversarial examples remains relatively underdeveloped. In this work we advance the field of adversarial machine learning through experimentation and analyses of three important SNN security attributes. First, we show that successful white-box adversarial attacks on SNNs are highly dependent on the underlying surrogate gradient technique. Second, we analyze the transferability of adversarial examples generated by SNNs and other state-of-the-art architectures like Vision Transformers and Big Transfer CNNs. We demonstrate that SNNs are not often deceived by adversarial examples generated by Vision Transformers and certain types of CNNs. Lastly, we develop a novel white-box attack that generates adversarial examples capable of fooling both SNN models and non-SNN models simultaneously. Our experiments and analyses are broad and rigorous covering two datasets (CIFAR-10 and CIFAR-100), five different white-box attacks and twelve different classifier models.

摘要: 尖峰神经网络(SNN)因其高能量效率和分类性能的最新进展而备受关注。然而，与传统的深度学习方法不同的是，对SNN对敌意例子的稳健性的分析和研究还相对较不发达。在这项工作中，我们通过实验和分析三个重要的SNN安全属性来推进对抗性机器学习领域。首先，我们证明了针对SNN的成功的白盒对抗攻击高度依赖于潜在的代理梯度技术。其次，我们分析了SNN和其他最先进的架构，如Vision Transformers和Big Transfer CNN生成的对抗性例子的可转移性。我们证明了SNN不会经常被Vision Transformers和某些类型的CNN生成的敌意例子所欺骗。最后，我们开发了一种新的白盒攻击，它可以生成能够同时愚弄SNN模型和非SNN模型的对抗性示例。我们的实验和分析涵盖了两个数据集(CIFAR-10和CIFAR-100)、五种不同的白盒攻击和12种不同的分类器模型。



## **16. Minotaur: Multi-Resource Blockchain Consensus**

牛头人：多资源区块链共识 cs.CR

To appear in ACM CCS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2201.11780v2)

**Authors**: Matthias Fitzi, Xuechao Wang, Sreeram Kannan, Aggelos Kiayias, Nikos Leonardos, Pramod Viswanath, Gerui Wang

**Abstracts**: Resource-based consensus is the backbone of permissionless distributed ledger systems. The security of such protocols relies fundamentally on the level of resources actively engaged in the system. The variety of different resources (and related proof protocols, some times referred to as PoX in the literature) raises the fundamental question whether it is possible to utilize many of them in tandem and build multi-resource consensus protocols. The challenge in combining different resources is to achieve fungibility between them, in the sense that security would hold as long as the cumulative adversarial power across all resources is bounded.   In this work, we put forth Minotaur, a multi-resource blockchain consensus protocol that combines proof-of-work (PoW) and proof-of-stake (PoS), and we prove it optimally fungible. At the core of our design, Minotaur operates in epochs while continuously sampling the active computational power to provide a fair exchange between the two resources, work and stake. Further, we demonstrate the ability of Minotaur to handle a higher degree of work fluctuation as compared to the Bitcoin blockchain; we also generalize Minotaur to any number of resources.   We demonstrate the simplicity of Minotaur via implementing a full stack client in Rust (available open source). We use the client to test the robustness of Minotaur to variable mining power and combined work/stake attacks and demonstrate concrete empirical evidence towards the suitability of Minotaur to serve as the consensus layer of a real-world blockchain.

摘要: 基于资源的共识是未经许可的分布式分类账系统的支柱。这类协议的安全性从根本上取决于系统中活跃的资源水平。不同资源的多样性(以及相关的证明协议，在文献中有时被称为POX)提出了一个基本的问题，即是否有可能同时利用其中的许多资源并建立多资源共识协议。组合不同资源的挑战是实现它们之间的互换性，从这个意义上说，只要所有资源的累积对抗能力是有限度的，安全就会保持。在这项工作中，我们提出了Minotaur，一个结合了工作证明(PoW)和风险证明(POS)的多资源区块链共识协议，并证明了它的最优可替换性。在我们设计的核心，Minotaur在不断采样活跃的计算能力的同时，在工作和赌注这两种资源之间提供公平的交换。此外，我们还展示了与比特币区块链相比，Minotaur能够处理更高程度的工作波动；我们还将Minotaur推广到任何数量的资源。我们通过在Rust(开放源码可用)中实现一个完整的堆栈客户端来演示Minotaur的简单性。我们使用客户端来测试Minotaur对可变挖掘功率和组合工作/桩攻击的健壮性，并展示了具体的经验证据，证明Minotaur适合作为现实世界区块链的共识层。



## **17. Distributed Adversarial Training to Robustify Deep Neural Networks at Scale**

深度神经网络规模化的分布式对抗性训练 cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2206.06257v2)

**Authors**: Gaoyuan Zhang, Songtao Lu, Yihua Zhang, Xiangyi Chen, Pin-Yu Chen, Quanfu Fan, Lee Martie, Lior Horesh, Mingyi Hong, Sijia Liu

**Abstracts**: Current deep neural networks (DNNs) are vulnerable to adversarial attacks, where adversarial perturbations to the inputs can change or manipulate classification. To defend against such attacks, an effective and popular approach, known as adversarial training (AT), has been shown to mitigate the negative impact of adversarial attacks by virtue of a min-max robust training method. While effective, it remains unclear whether it can successfully be adapted to the distributed learning context. The power of distributed optimization over multiple machines enables us to scale up robust training over large models and datasets. Spurred by that, we propose distributed adversarial training (DAT), a large-batch adversarial training framework implemented over multiple machines. We show that DAT is general, which supports training over labeled and unlabeled data, multiple types of attack generation methods, and gradient compression operations favored for distributed optimization. Theoretically, we provide, under standard conditions in the optimization theory, the convergence rate of DAT to the first-order stationary points in general non-convex settings. Empirically, we demonstrate that DAT either matches or outperforms state-of-the-art robust accuracies and achieves a graceful training speedup (e.g., on ResNet-50 under ImageNet). Codes are available at https://github.com/dat-2022/dat.

摘要: 当前的深度神经网络(DNN)很容易受到敌意攻击，对输入的敌意扰动可以改变或操纵分类。为了防御这种攻击，一种被称为对抗性训练(AT)的有效和流行的方法已经被证明通过最小-最大稳健训练方法来减轻对抗性攻击的负面影响。虽然有效，但它是否能成功地适应分布式学习环境仍不清楚。在多台机器上进行分布式优化的能力使我们能够在大型模型和数据集上扩大健壮的训练。受此启发，我们提出了分布式对抗训练(DAT)，这是一种在多台机器上实现的大批量对抗训练框架。我们证明DAT是通用的，它支持对有标签和无标签数据的训练，支持多种类型的攻击生成方法，以及有利于分布式优化的梯度压缩操作。理论上，在最优化理论的标准条件下，我们给出了一般非凸集上DAT收敛到一阶驻点的收敛速度。在实验上，我们证明了DAT匹配或超过了最先进的稳健精度，并实现了优雅的训练加速比(例如，在ImageNet下的ResNet-50上)。有关代码，请访问https://github.com/dat-2022/dat.



## **18. Privacy Against Inference Attacks in Vertical Federated Learning**

垂直联合学习中抵抗推理攻击的隐私保护 cs.LG

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2207.11788v3)

**Authors**: Borzoo Rassouli, Morteza Varasteh, Deniz Gunduz

**Abstracts**: Vertical federated learning is considered, where an active party, having access to true class labels, wishes to build a classification model by utilizing more features from a passive party, which has no access to the labels, to improve the model accuracy. In the prediction phase, with logistic regression as the classification model, several inference attack techniques are proposed that the adversary, i.e., the active party, can employ to reconstruct the passive party's features, regarded as sensitive information. These attacks, which are mainly based on a classical notion of the center of a set, i.e., the Chebyshev center, are shown to be superior to those proposed in the literature. Moreover, several theoretical performance guarantees are provided for the aforementioned attacks. Subsequently, we consider the minimum amount of information that the adversary needs to fully reconstruct the passive party's features. In particular, it is shown that when the passive party holds one feature, and the adversary is only aware of the signs of the parameters involved, it can perfectly reconstruct that feature when the number of predictions is large enough. Next, as a defense mechanism, a privacy-preserving scheme is proposed that worsen the adversary's reconstruction attacks, while preserving the full benefits that VFL brings to the active party. Finally, experimental results demonstrate the effectiveness of the proposed attacks and the privacy-preserving scheme.

摘要: 考虑垂直联合学习，其中可以访问真实类别标签的主动方希望通过利用来自被动方的更多特征来构建分类模型，而被动方不能访问标签，以提高模型的精度。在预测阶段，以Logistic回归为分类模型，提出了几种推理攻击技术，对手即主动方可以用来重构被动方的特征，并将其视为敏感信息。这些攻击主要基于经典的集合中心概念，即切比雪夫中心，被证明优于文献中提出的攻击。此外，还为上述攻击提供了几个理论上的性能保证。随后，我们考虑了对手完全重建被动方特征所需的最小信息量。特别地，当被动方持有一个特征，并且对手只知道所涉及的参数的符号时，当预测次数足够大时，它可以完美地重构该特征。接下来，作为一种防御机制，提出了一种隐私保护方案，该方案在保留VFL给主动方带来的全部好处的同时，恶化了对手的重构攻击。最后，实验结果验证了所提出的攻击和隐私保护方案的有效性。



## **19. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03755v1)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstracts**: Mis- and disinformation are now a substantial global threat to our security and safety. To cope with the scale of online misinformation, one viable solution is to automate the fact-checking of claims by retrieving and verifying against relevant evidence. While major recent advances have been achieved in pushing forward the automatic fact-verification, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence, or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence, in addition to generating diverse and claim-aligned evidence. As a result, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息现在是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，一个可行的解决方案是通过检索和核实相关证据来自动化索赔的事实核查。虽然最近在推动自动事实核查方面取得了重大进展，但仍然缺乏对针对这类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，除了生成多样化的和与索赔一致的证据外，还可以微妙地修改证据中突出索赔的片段。因此，在分类维度的许多不同排列下，我们会极大地降低事实检查性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，我们最后讨论了未来防御的挑战和方向。



## **20. State of Security Awareness in the AM Industry: 2020 Survey**

AM行业的安全意识状况：2020年调查 cs.CR

The material was presented at ASTM ICAM 2021 and a publication was  accepted for publication as a Selected Technical Papers (STP)

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.03073v1)

**Authors**: Mark Yampolskiy, Paul Bates, Mohsen Seifi, Nima Shamsaei

**Abstracts**: Security of Additive Manufacturing (AM) gets increased attention due to the growing proliferation and adoption of AM in a variety of applications and business models. However, there is a significant disconnect between AM community focused on manufacturing and AM Security community focused on securing this highly computerized manufacturing technology. To bridge this gap, we surveyed the America Makes AM community, asking in total eleven AM security-related questions aiming to discover the existing concerns, posture, and expectations. The first set of questions aimed to discover how many of these organizations use AM, outsource AM, or provide AM as a service. Then we asked about biggest security concerns as well as about assessment of who the potential adversaries might be and their motivation for attack. We then proceeded with questions on any experienced security incidents, if any security risk assessment was conducted, and if the participants' organizations were partnering with external experts to secure AM. Lastly, we asked whether security measures are implemented at all and, if yes, whether they fall under the general cyber-security category. Out of 69 participants affiliated with commercial industry, agencies, and academia, 53 have completed the entire survey. This paper presents the results of this survey, as well as provides our assessment of the AM Security posture. The answers are a mixture of what we could label as expected, "shocking but not surprising," and completely unexpected. Assuming that the provided answers are somewhat representative to the current state of the AM industry, we conclude that the industry is not ready to prevent or detect AM-specific attacks that have been demonstrated in the research literature.

摘要: 由于添加制造(AM)在各种应用和商业模式中的普及和采用，AM的安全性受到越来越多的关注。然而，在专注于制造的AM社区和专注于保护这种高度计算机化的制造技术的AM Security社区之间存在着严重的脱节。为了弥合这一差距，我们对美国制造AM社区进行了调查，总共询问了11个与AM安全相关的问题，旨在发现现有的担忧、状况和期望。第一组问题旨在发现这些组织中有多少使用AM、外包AM或将AM作为服务提供。然后，我们询问了最大的安全担忧，以及对潜在对手可能是谁及其攻击动机的评估。然后，我们继续就任何有经验的安全事件、是否进行了任何安全风险评估以及参与者的组织是否与外部专家合作确保AM的安全进行了提问最后，我们询问是否实施了安全措施，如果是，这些措施是否属于一般网络安全类别。在69名与商业、机构和学术界有关联的参与者中，有53人完成了整个调查。本文介绍了本次调查的结果，并提供了我们对AM安全态势的评估。答案既有我们所期待的，也有完全出乎意料的，既令人震惊，又不令人惊讶。假设所提供的答案在某种程度上代表了AM行业的当前状态，我们得出的结论是，该行业还没有准备好预防或检测研究文献中已经证明的AM特定攻击。



## **21. On the Transferability of Adversarial Examples between Encrypted Models**

关于对抗性例子在加密模型之间的可转移性 cs.CV

to be appear in ISPACS 2022

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02997v1)

**Authors**: Miki Tanaka, Isao Echizen, Hitoshi Kiya

**Abstracts**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, namely, AEs generated for a source model fool other (target) models. In this paper, we investigate the transferability of models encrypted for adversarially robust defense for the first time. To objectively verify the property of transferability, the robustness of models is evaluated by using a benchmark attack method, called AutoAttack. In an image-classification experiment, the use of encrypted models is confirmed not only to be robust against AEs but to also reduce the influence of AEs in terms of the transferability of models.

摘要: 众所周知，深度神经网络(DNN)很容易受到敌意例子(AEs)的攻击。此外，AEs具有对抗性可转移性，即，为源模型生成的AEs欺骗其他(目标)模型。在这篇文章中，我们首次研究了用于对抗健壮防御的加密模型的可转移性。为了客观地验证模型的可转移性，使用一种称为AutoAttack的基准攻击方法对模型的稳健性进行了评估。在图像分类实验中，加密模型的使用被证实不仅对AEs具有健壮性，而且在模型的可转移性方面也减少了AEs的影响。



## **22. Adversarial Mask: Real-World Universal Adversarial Attack on Face Recognition Model**

对抗性面具：现实世界中人脸识别模型的通用对抗性攻击 cs.CV

16 pages, 9 figures

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2111.10759v3)

**Authors**: Alon Zolfi, Shai Avidan, Yuval Elovici, Asaf Shabtai

**Abstracts**: Deep learning-based facial recognition (FR) models have demonstrated state-of-the-art performance in the past few years, even when wearing protective medical face masks became commonplace during the COVID-19 pandemic. Given the outstanding performance of these models, the machine learning research community has shown increasing interest in challenging their robustness. Initially, researchers presented adversarial attacks in the digital domain, and later the attacks were transferred to the physical domain. However, in many cases, attacks in the physical domain are conspicuous, and thus may raise suspicion in real-world environments (e.g., airports). In this paper, we propose Adversarial Mask, a physical universal adversarial perturbation (UAP) against state-of-the-art FR models that is applied on face masks in the form of a carefully crafted pattern. In our experiments, we examined the transferability of our adversarial mask to a wide range of FR model architectures and datasets. In addition, we validated our adversarial mask's effectiveness in real-world experiments (CCTV use case) by printing the adversarial pattern on a fabric face mask. In these experiments, the FR system was only able to identify 3.34% of the participants wearing the mask (compared to a minimum of 83.34% with other evaluated masks). A demo of our experiments can be found at: https://youtu.be/_TXkDO5z11w.

摘要: 基于深度学习的面部识别(FR)模型在过去几年展示了最先进的性能，即使在新冠肺炎大流行期间戴防护性医用口罩变得司空见惯。鉴于这些模型的出色性能，机器学习研究界对挑战它们的稳健性表现出越来越大的兴趣。最初，研究人员在数字领域提出了对抗性攻击，后来攻击被转移到物理领域。然而，在许多情况下，物理域中的攻击很明显，因此可能会在现实环境(例如机场)中引起怀疑。在本文中，我们提出了对抗面具，一种针对最先进的FR模型的物理通用对抗扰动(UAP)，它以精心制作的模式的形式应用于人脸面具上。在我们的实验中，我们检查了我们的对手面具在广泛的FR模型体系结构和数据集上的可转移性。此外，我们在真实世界的实验中(CCTV用例)验证了我们的对抗面具的有效性，通过将对抗图案打印在织物面膜上。在这些实验中，FR系统只能识别3.34%的戴口罩的参与者(相比之下，其他评估的口罩的最低识别率为83.34%)。我们的实验演示可在以下网址找到：https://youtu.be/_TXkDO5z11w.



## **23. Facial De-morphing: Extracting Component Faces from a Single Morph**

面部去变形：从单个变形中提取组件人脸 cs.CV

**SubmitDate**: 2022-09-07    [paper-pdf](http://arxiv.org/pdf/2209.02933v1)

**Authors**: Sudipta Banerjee, Prateek Jaiswal, Arun Ross

**Abstracts**: A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.

摘要: 面部变形是通过策略性地组合对应于多个身份的两个或多个面部图像来创建的。其目的是使变形后的图像与多个身份匹配。当前的变形攻击检测策略可以检测变形，但无法恢复创建变形时使用的图像或身份。从变形后的人脸图像中推断出单个人脸图像的任务称为纹理{去变形}。现有的去变形工作假定存在与一个身份有关的参考图像，以便恢复共犯--即另一个身份--的图像。在这项工作中，我们提出了一种新的去变形方法，它可以从一幅变形后的人脸图像中同时恢复出两种身份的图像，而不需要参考图像或关于变形过程的先验信息。我们提出了一种产生式对抗性网络，它实现了基于单一图像的去变形，具有与原始人脸图像惊人的高度视觉真实感和生物特征相似性。我们展示了我们的方法在基于里程碑的变形和基于生成性模型的变形上的性能，并取得了令人满意的结果。



## **24. Annealing Optimization for Progressive Learning with Stochastic Approximation**

随机逼近渐进式学习的退火法优化 eess.SY

arXiv admin note: text overlap with arXiv:2102.05836

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02826v1)

**Authors**: Christos Mavridis, John Baras

**Abstracts**: In this work, we introduce a learning model designed to meet the needs of applications in which computational resources are limited, and robustness and interpretability are prioritized. Learning problems can be formulated as constrained stochastic optimization problems, with the constraints originating mainly from model assumptions that define a trade-off between complexity and performance. This trade-off is closely related to over-fitting, generalization capacity, and robustness to noise and adversarial attacks, and depends on both the structure and complexity of the model, as well as the properties of the optimization methods used. We develop an online prototype-based learning algorithm based on annealing optimization that is formulated as an online gradient-free stochastic approximation algorithm. The learning model can be viewed as an interpretable and progressively growing competitive-learning neural network model to be used for supervised, unsupervised, and reinforcement learning. The annealing nature of the algorithm contributes to minimal hyper-parameter tuning requirements, poor local minima prevention, and robustness with respect to the initial conditions. At the same time, it provides online control over the performance-complexity trade-off by progressively increasing the complexity of the learning model as needed, through an intuitive bifurcation phenomenon. Finally, the use of stochastic approximation enables the study of the convergence of the learning algorithm through mathematical tools from dynamical systems and control, and allows for its integration with reinforcement learning algorithms, constructing an adaptive state-action aggregation scheme.

摘要: 在这项工作中，我们引入了一个学习模型，旨在满足计算资源有限、稳健性和可解释性优先的应用程序的需求。学习问题可以表示为受约束的随机优化问题，约束主要来自定义复杂性和性能之间权衡的模型假设。这种权衡与过拟合、泛化能力以及对噪声和敌意攻击的稳健性密切相关，并且取决于模型的结构和复杂性以及所使用的优化方法的性质。提出了一种基于退火法的在线原型学习算法，该算法是一种在线无梯度随机逼近算法。该学习模型可以看作是一个可解释的、渐进式增长的竞争学习神经网络模型，可用于有监督、无监督和强化学习。该算法的退火性有助于最小的超参数调整要求、较差的局部极小值防止以及相对于初始条件的稳健性。同时，它通过直观的分叉现象，根据需要逐步增加学习模型的复杂性，从而提供对性能和复杂性权衡的在线控制。最后，随机逼近的使用使得能够通过动态系统和控制的数学工具来研究学习算法的收敛，并允许其与强化学习算法相结合，构造自适应状态-动作聚集方案。



## **25. Bag of Tricks for FGSM Adversarial Training**

用于FGSM对抗性训练的技巧包 cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02684v1)

**Authors**: Zichao Li, Li Liu, Zeyu Wang, Yuyin Zhou, Cihang Xie

**Abstracts**: Adversarial training (AT) with samples generated by Fast Gradient Sign Method (FGSM), also known as FGSM-AT, is a computationally simple method to train robust networks. However, during its training procedure, an unstable mode of "catastrophic overfitting" has been identified in arXiv:2001.03994 [cs.LG], where the robust accuracy abruptly drops to zero within a single training step. Existing methods use gradient regularizers or random initialization tricks to attenuate this issue, whereas they either take high computational cost or lead to lower robust accuracy. In this work, we provide the first study, which thoroughly examines a collection of tricks from three perspectives: Data Initialization, Network Structure, and Optimization, to overcome the catastrophic overfitting in FGSM-AT.   Surprisingly, we find that simple tricks, i.e., a) masking partial pixels (even without randomness), b) setting a large convolution stride and smooth activation functions, or c) regularizing the weights of the first convolutional layer, can effectively tackle the overfitting issue. Extensive results on a range of network architectures validate the effectiveness of each proposed trick, and the combinations of tricks are also investigated. For example, trained with PreActResNet-18 on CIFAR-10, our method attains 49.8% accuracy against PGD-50 attacker and 46.4% accuracy against AutoAttack, demonstrating that pure FGSM-AT is capable of enabling robust learners. The code and models are publicly available at https://github.com/UCSC-VLAA/Bag-of-Tricks-for-FGSM-AT.

摘要: 用快速梯度符号法(FGSM)生成样本的对抗性训练(AT)，也称为FGSM-AT，是一种计算简单的稳健网络训练方法。然而，在其训练过程中，在ARXIV：2001.03994[cs.LG]中发现了一种不稳定的“灾难性过拟合”模式，其中鲁棒精度在单一训练步骤内突然降至零。现有的方法使用梯度正则化或随机初始化技巧来减弱这一问题，但它们要么计算量大，要么导致鲁棒精度较低。在这项工作中，我们提供了第一项研究，从数据初始化、网络结构和优化三个角度深入研究了一系列技巧，以克服FGSM-AT中灾难性的过拟合。令人惊讶的是，我们发现，简单的技巧，即a)掩蔽部分像素(即使没有随机性)，b)设置大的卷积步长和平滑的激活函数，或c)正则化第一卷积层的权重，可以有效地解决过拟合问题。在一系列网络体系结构上的广泛结果验证了所提出的每种技巧的有效性，并对各种技巧的组合进行了研究。例如，在CIFAR-10上用PreActResNet-18进行训练，我们的方法对PGD-50攻击者的准确率达到49.8%，对AutoAttack的准确率达到46.4%，表明纯FGSM-AT能够支持健壮的学习者。代码和模型可在https://github.com/UCSC-VLAA/Bag-of-Tricks-for-FGSM-AT.上公开获取



## **26. Improving the Accuracy and Robustness of CNNs Using a Deep CCA Neural Data Regularizer**

利用深度CCA神经数据规则器提高CNN的精度和鲁棒性 cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02582v1)

**Authors**: Cassidy Pirlot, Richard C. Gerum, Cory Efird, Joel Zylberberg, Alona Fyshe

**Abstracts**: As convolutional neural networks (CNNs) become more accurate at object recognition, their representations become more similar to the primate visual system. This finding has inspired us and other researchers to ask if the implication also runs the other way: If CNN representations become more brain-like, does the network become more accurate? Previous attempts to address this question showed very modest gains in accuracy, owing in part to limitations of the regularization method. To overcome these limitations, we developed a new neural data regularizer for CNNs that uses Deep Canonical Correlation Analysis (DCCA) to optimize the resemblance of the CNN's image representations to that of the monkey visual cortex. Using this new neural data regularizer, we see much larger performance gains in both classification accuracy and within-super-class accuracy, as compared to the previous state-of-the-art neural data regularizers. These networks are also more robust to adversarial attacks than their unregularized counterparts. Together, these results confirm that neural data regularization can push CNN performance higher, and introduces a new method that obtains a larger performance boost.

摘要: 随着卷积神经网络(CNN)在物体识别方面变得更加准确，它们的表示变得更加类似于灵长类视觉系统。这一发现激发了我们和其他研究人员的疑问：如果CNN的表现变得更像大脑，网络是否会变得更准确？以前解决这一问题的尝试显示，由于正规化方法的局限性，在准确性方面取得了很小的进步。为了克服这些限制，我们开发了一种新的CNN神经数据正则化方法，它使用深度典型相关分析(DCCA)来优化CNN图像表示与猴子视觉皮质的相似性。使用这种新的神经数据正则化器，我们看到与以前最先进的神经数据正则化器相比，在分类精度和超类内精度方面都有更大的性能收益。与非正规网络相比，这些网络对对手攻击的抵抗力也更强。综上所述，这些结果证实了神经数据正则化可以提高CNN的性能，并介绍了一种新的方法，获得了更大的性能提升。



## **27. Instance Attack:An Explanation-based Vulnerability Analysis Framework Against DNNs for Malware Detection**

实例攻击：一种基于解释的DNN漏洞分析框架 cs.CR

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02453v1)

**Authors**: Sun RuiJin, Guo ShiZe, Guo JinHong, Xing ChangYou, Yang LuMing, Guo Xi, Pan ZhiSong

**Abstracts**: Deep neural networks (DNNs) are increasingly being applied in malware detection and their robustness has been widely debated. Traditionally an adversarial example generation scheme relies on either detailed model information (gradient-based methods) or lots of samples to train a surrogate model, neither of which are available in most scenarios.   We propose the notion of the instance-based attack. Our scheme is interpretable and can work in a black-box environment. Given a specific binary example and a malware classifier, we use the data augmentation strategies to produce enough data from which we can train a simple interpretable model. We explain the detection model by displaying the weight of different parts of the specific binary. By analyzing the explanations, we found that the data subsections play an important role in Windows PE malware detection. We proposed a new function preserving transformation algorithm that can be applied to data subsections. By employing the binary-diversification techniques that we proposed, we eliminated the influence of the most weighted part to generate adversarial examples. Our algorithm can fool the DNNs in certain cases with a success rate of nearly 100\%. Our method outperforms the state-of-the-art method . The most important aspect is that our method operates in black-box settings and the results can be validated with domain knowledge. Our analysis model can assist people in improving the robustness of malware detectors.

摘要: 深度神经网络(DNN)在恶意软件检测中的应用越来越广泛，其健壮性一直备受争议。传统的对抗性示例生成方案依赖于详细的模型信息(基于梯度的方法)或大量样本来训练代理模型，这两种方法在大多数情况下都不可用。我们提出了基于实例攻击的概念。我们的方案是可解释的，可以在黑盒环境中工作。给出一个特定的二进制例子和一个恶意软件分类器，我们使用数据扩充策略来产生足够的数据，从中我们可以训练一个简单的可解释模型。我们通过显示特定二进制的不同部分的权重来解释检测模型。通过分析解释，我们发现数据子部分在Windows PE恶意软件检测中起着重要作用。提出了一种新的适用于数据细分的保函数变换算法。通过采用我们提出的二元多样化技术，我们消除了权重最大的部分对生成对抗性例子的影响。在某些情况下，我们的算法可以欺骗DNN，成功率接近100%。我们的方法比最先进的方法性能更好。最重要的是，我们的方法是在黑盒环境下运行的，并且结果可以用领域知识来验证。我们的分析模型可以帮助人们提高恶意软件检测器的健壮性。



## **28. MACAB: Model-Agnostic Clean-Annotation Backdoor to Object Detection with Natural Trigger in Real-World**

MACAB：真实世界中自然触发目标检测的模型不可知性Clean-Annotation后门 cs.CV

**SubmitDate**: 2022-09-06    [paper-pdf](http://arxiv.org/pdf/2209.02339v1)

**Authors**: Hua Ma, Yinshan Li, Yansong Gao, Zhi Zhang, Alsharif Abuadbba, Anmin Fu, Said F. Al-Sarawi, Nepal Surya, Derek Abbott

**Abstracts**: Object detection is the foundation of various critical computer-vision tasks such as segmentation, object tracking, and event detection. To train an object detector with satisfactory accuracy, a large amount of data is required. However, due to the intensive workforce involved with annotating large datasets, such a data curation task is often outsourced to a third party or relied on volunteers. This work reveals severe vulnerabilities of such data curation pipeline. We propose MACAB that crafts clean-annotated images to stealthily implant the backdoor into the object detectors trained on them even when the data curator can manually audit the images. We observe that the backdoor effect of both misclassification and the cloaking are robustly achieved in the wild when the backdoor is activated with inconspicuously natural physical triggers. Backdooring non-classification object detection with clean-annotation is challenging compared to backdooring existing image classification tasks with clean-label, owing to the complexity of having multiple objects within each frame, including victim and non-victim objects. The efficacy of the MACAB is ensured by constructively i abusing the image-scaling function used by the deep learning framework, ii incorporating the proposed adversarial clean image replica technique, and iii combining poison data selection criteria given constrained attacking budget. Extensive experiments demonstrate that MACAB exhibits more than 90% attack success rate under various real-world scenes. This includes both cloaking and misclassification backdoor effect even restricted with a small attack budget. The poisoned samples cannot be effectively identified by state-of-the-art detection techniques.The comprehensive video demo is at https://youtu.be/MA7L_LpXkp4, which is based on a poison rate of 0.14% for YOLOv4 cloaking backdoor and Faster R-CNN misclassification backdoor.

摘要: 目标检测是各种关键的计算机视觉任务的基础，如分割、目标跟踪和事件检测。为了以令人满意的精度训练目标检测器，需要大量的数据。然而，由于注释大型数据集所涉及的密集劳动力，这样的数据管理任务通常被外包给第三方或依赖于志愿者。这项工作揭示了这种数据管理管道的严重漏洞。我们建议MACAB制作经过干净注释的图像，以便在数据管理员可以手动审计图像的情况下，将后门秘密植入对其训练的对象检测器。我们观察到，当后门被不明显的自然物理触发激活时，错误分类和伪装的后门效应在野外都得到了很好的实现。由于每帧中包含多个对象(包括受害者和非受害者对象)的复杂性，与使用干净标签回溯现有图像分类任务相比，使用干净注释来回溯非分类对象检测是具有挑战性的。MACAB的有效性是通过建设性地I滥用深度学习框架使用的图像缩放函数，II结合所提出的对抗性干净图像复制技术，III在给定有限攻击预算的情况下结合毒物数据选择标准来确保的。大量实验表明，MACAB在各种真实场景下的攻击成功率均在90%以上。这包括隐形和错误分类的后门效应，即使在攻击预算很小的情况下也是如此。最先进的检测技术无法有效地识别有毒样本。全面的视频演示在https://youtu.be/MA7L_LpXkp4，上进行，这是基于YOLOv4伪装后门的毒用率为0.14%和更快的R-CNN错误分类后门。



## **29. White-Box Adversarial Policies in Deep Reinforcement Learning**

深度强化学习中的白盒对抗策略 cs.AI

Code is available at  https://github.com/thestephencasper/white_box_rarl

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02167v1)

**Authors**: Stephen Casper, Dylan Hadfield-Menell, Gabriel Kreiman

**Abstracts**: Adversarial examples against AI systems pose both risks via malicious attacks and opportunities for improving robustness via adversarial training. In multiagent settings, adversarial policies can be developed by training an adversarial agent to minimize a victim agent's rewards. Prior work has studied black-box attacks where the adversary only sees the state observations and effectively treats the victim as any other part of the environment. In this work, we experiment with white-box adversarial policies to study whether an agent's internal state can offer useful information for other agents. We make three contributions. First, we introduce white-box adversarial policies in which an attacker can observe a victim's internal state at each timestep. Second, we demonstrate that white-box access to a victim makes for better attacks in two-agent environments, resulting in both faster initial learning and higher asymptotic performance against the victim. Third, we show that training against white-box adversarial policies can be used to make learners in single-agent environments more robust to domain shifts.

摘要: 针对人工智能系统的对抗性例子既有通过恶意攻击带来的风险，也有通过对抗性培训提高稳健性的机会。在多代理设置中，可以通过训练对抗代理来制定对抗策略，以将受害者代理的回报降至最低。以前的工作已经研究了黑盒攻击，在这种攻击中，对手只看到状态观察，并有效地将受害者视为环境的任何其他部分。在这项工作中，我们实验了白盒对抗策略，以研究一个代理的内部状态是否能为其他代理提供有用的信息。我们有三点贡献。首先，我们引入了白盒对抗性策略，其中攻击者可以在每个时间步观察受害者的内部状态。其次，我们证明了在双代理环境中，对受害者的白盒访问有助于更好的攻击，从而导致更快的初始学习和更高的针对受害者的渐近性能。第三，我们证明了针对白盒对抗策略的训练可以用来使单代理环境中的学习者对域转移更健壮。



## **30. Reinforcement learning-based optimised control for tracking of nonlinear systems with adversarial attacks**

基于强化学习的对抗性非线性系统跟踪优化控制 eess.SY

Submitted for The 10th RSI International Conference on Robotics and  Mechatronics (ICRoM 2022)

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02165v1)

**Authors**: Farshad Rahimi, Sepideh Ziaei

**Abstracts**: This paper introduces a reinforcement learning-based tracking control approach for a class of nonlinear systems using neural networks. In this approach, adversarial attacks were considered both in the actuator and on the outputs. This approach incorporates a simultaneous tracking and optimization process. It is necessary to be able to solve the Hamilton-Jacobi-Bellman equation (HJB) in order to obtain optimal control input, but this is difficult due to the strong nonlinearity terms in the equation. In order to find the solution to the HJB equation, we used a reinforcement learning approach. In this online adaptive learning approach, three neural networks are simultaneously adapted: the critic neural network, the actor neural network, and the adversary neural network. Ultimately, simulation results are presented to demonstrate the effectiveness of the introduced method on a manipulator.

摘要: 针对一类神经网络非线性系统，提出了一种基于强化学习的跟踪控制方法。在这种方法中，在执行器和输出端都考虑了对抗性攻击。这种方法结合了同步跟踪和优化过程。为了获得最优控制输入，必须能解Hamilton-Jacobi-Bellman方程(HJB)，但由于方程中的强非线性项，这是很困难的。为了找到HJB方程的解，我们使用了强化学习方法。在这种在线自适应学习方法中，同时自适应了三个神经网络：批评者神经网络、行动者神经网络和对手神经网络。最后，以机械手为例，给出了仿真结果，验证了该方法的有效性。



## **31. On the Anonymity of Peer-To-Peer Network Anonymity Schemes Used by Cryptocurrencies**

加密货币使用的对等网络匿名方案的匿名性研究 cs.CR

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2201.11860v3)

**Authors**: Piyush Kumar Sharma, Devashish Gosain, Claudia Diaz

**Abstracts**: Cryptocurrency systems can be subject to deanonimization attacks by exploiting the network-level communication on their peer-to-peer network. Adversaries who control a set of colluding node(s) within the peer-to-peer network can observe transactions being exchanged and infer the parties involved. Thus, various network anonymity schemes have been proposed to mitigate this problem, with some solutions providing theoretical anonymity guarantees.   In this work, we model such peer-to-peer network anonymity solutions and evaluate their anonymity guarantees. To do so, we propose a novel framework that uses Bayesian inference to obtain the probability distributions linking transactions to their possible originators. We characterize transaction anonymity with those distributions, using entropy as metric of adversarial uncertainty on the originator's identity. In particular, we model Dandelion, Dandelion++ and Lightning Network. We study different configurations and demonstrate that none of them offers acceptable anonymity to their users. For instance, our analysis reveals that in the widely deployed Lightning Network, with 1% strategically chosen colluding nodes the adversary can uniquely determine the originator for about 50% of the total transactions in the network. In Dandelion, an adversary that controls 15% of the nodes has on average uncertainty among only 8 possible originators. Moreover, we observe that due to the way Dandelion and Dandelion++ are designed, increasing the network size does not correspond to an increase in the anonymity set of potential originators. Alarmingly, our longitudinal analysis of Lightning Network reveals rather an inverse trend -- with the growth of the network the overall anonymity decreases.

摘要: 通过利用其对等网络上的网络级通信，加密货币系统可能会受到反匿名化攻击。在对等网络中控制一组串通节点的敌手可以观察正在交换的交易并推断所涉及的各方。因此，各种网络匿名方案被提出来缓解这一问题，一些解决方案提供了理论上的匿名性保证。在这项工作中，我们对这种对等网络匿名解决方案进行建模，并评估它们的匿名性保证。为此，我们提出了一个新的框架，它使用贝叶斯推理来获得将事务链接到可能的发起者的概率分布。我们使用这些分布来表征交易匿名性，使用熵作为对发起者身份的敌意不确定性的度量。特别是，我们对蒲公英、蒲公英++和闪电网络进行了建模。我们研究了不同的配置，并证明它们都不能为用户提供可接受的匿名性。例如，我们的分析表明，在广泛部署的闪电网络中，通过1%的策略选择合谋节点，对手可以唯一地确定网络中约50%的总交易的发起者。在蒲公英中，一个控制了15%节点的对手平均只有8个可能的发起者中存在不确定性。此外，我们观察到，由于蒲公英和蒲公英++的设计方式，增加网络规模并不对应于潜在发起者匿名性集合的增加。令人担忧的是，我们对Lightning Network的纵向分析揭示了一个相反的趋势--随着网络的增长，总体匿名性下降。



## **32. Evaluating the Susceptibility of Pre-Trained Language Models via Handcrafted Adversarial Examples**

通过手工制作的对抗性例子评估预先训练的语言模型的敏感性 cs.CL

10 pages, 1 figure, 3 tables

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.02128v1)

**Authors**: Hezekiah J. Branch, Jonathan Rodriguez Cefalu, Jeremy McHugh, Leyla Hujer, Aditya Bahl, Daniel del Castillo Iglesias, Ron Heichman, Ramesh Darwishi

**Abstracts**: Recent advances in the development of large language models have resulted in public access to state-of-the-art pre-trained language models (PLMs), including Generative Pre-trained Transformer 3 (GPT-3) and Bidirectional Encoder Representations from Transformers (BERT). However, evaluations of PLMs, in practice, have shown their susceptibility to adversarial attacks during the training and fine-tuning stages of development. Such attacks can result in erroneous outputs, model-generated hate speech, and the exposure of users' sensitive information. While existing research has focused on adversarial attacks during either the training or the fine-tuning of PLMs, there is a deficit of information on attacks made between these two development phases. In this work, we highlight a major security vulnerability in the public release of GPT-3 and further investigate this vulnerability in other state-of-the-art PLMs. We restrict our work to pre-trained models that have not undergone fine-tuning. Further, we underscore token distance-minimized perturbations as an effective adversarial approach, bypassing both supervised and unsupervised quality measures. Following this approach, we observe a significant decrease in text classification quality when evaluating for semantic similarity.

摘要: 在开发大型语言模型方面的最新进展导致公众能够访问最先进的预训练语言模型(PLM)，包括生成性预训练转换器3(GPT-3)和来自转换器的双向编码器表示(BERT)。然而，在实践中，对PLM的评估表明，它们在训练和发展的微调阶段容易受到对抗性攻击。此类攻击可能导致错误输出、模型生成的仇恨言论以及用户敏感信息的暴露。虽然现有的研究集中在PLM的训练或微调期间的对抗性攻击，但关于这两个开发阶段之间的攻击的信息不足。在这项工作中，我们突出了公开发布的GPT-3中的一个主要安全漏洞，并进一步调查了其他最先进的PLM中的此漏洞。我们将我们的工作限制在未经微调的预先训练的模型上。此外，我们强调令牌距离最小化扰动是一种有效的对抗性方法，绕过了监督和非监督质量度量。按照这种方法，我们观察到在评估语义相似性时文本分类质量显著下降。



## **33. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

PatchZero：通过检测和归零补丁来防御敌意补丁攻击 cs.CV

Accepted to WACV 2023

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2207.01795v3)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.

摘要: 对抗性补丁攻击通过在局部区域内注入对抗性像素来误导神经网络。补丁攻击可以在各种任务中非常有效，并且可以通过附着(例如贴纸)到真实世界的对象来物理实现。尽管攻击模式多种多样，但敌方补丁往往纹理丰富，外观与自然图像不同。我们利用这一特性，提出了PatchZero，一种针对白盒恶意补丁的通用防御管道，而不需要重新训练下游的分类器或检测器。具体地说，我们的防御在像素级检测对手，并通过使用平均像素值重新绘制来对补丁区域进行“清零”。我们进一步设计了一种两阶段对抗性训练方案，以抵御更强的适应性攻击。PatchZero在图像分类(ImageNet，RESISC45)、目标检测(Pascal VOC)和视频分类(UCF101)任务上实现了SOTA防御性能，性能良好，性能几乎没有下降。此外，PatchZero还可以转换为不同的补丁形状和攻击类型。



## **34. Adversarial Detection: Attacking Object Detection in Real Time**

对抗性检测：攻击目标的实时检测 cs.AI

7 pages, 10 figures

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01962v1)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: Intelligent robots hinge on accurate object detection models to perceive the environment. Advances in deep learning security unveil that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. It is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. There is still a gap between theoretical discoveries and real-world applications. We bridge the gap by proposing the first real-time online attack against object detection models. We devised three attacks that fabricate bounding boxes for nonexistent objects at desired locations.

摘要: 智能机器人依赖于准确的目标检测模型来感知环境。深度学习安全方面的进展揭示了目标检测模型容易受到对手攻击。然而，以往的研究主要集中在攻击静态图像或离线视频上。目前尚不清楚此类攻击是否会危及动态环境中真实世界的机器人应用。在理论发现和现实应用之间仍有差距。我们通过提出第一个针对目标检测的实时在线攻击模型来弥补这一差距。我们设计了三种攻击，在所需位置为不存在的对象制造边界框。



## **35. Jamming Modulation: An Active Anti-Jamming Scheme**

干扰调制：一种主动抗干扰性方案 cs.IT

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01943v1)

**Authors**: Jianhui Ma, Qiang Li, Zilong Liu, Linsong Du, Hongyang Chen, Nirwan Ansari

**Abstracts**: Providing quality communications under adversarial electronic attacks, e.g., broadband jamming attacks, is a challenging task. Unlike state-of-the-art approaches which treat jamming signals as destructive interference, this paper presents a novel active anti-jamming (AAJ) scheme for a jammed channel to enhance the communication quality between a transmitter node (TN) and receiver node (RN), where the TN actively exploits the jamming signal as a carrier to send messages. Specifically, the TN is equipped with a programmable-gain amplifier, which is capable of re-modulating the jamming signals for jamming modulation. Considering four typical jamming types, we derive both the bit error rates (BER) and the corresponding optimal detection thresholds of the AAJ scheme. The asymptotic performances of the AAJ scheme are discussed under the high jamming-to-noise ratio (JNR) and sampling rate cases. Our analysis shows that there exists a BER floor for sufficiently large JNR. Simulation results indicate that the proposed AAJ scheme allows the TN to communicate with the RN reliably even under extremely strong and/or broadband jamming. Additionally, we investigate the channel capacity of the proposed AAJ scheme and show that the channel capacity of the AAJ scheme outperforms that of the direct transmission when the JNR is relatively high.

摘要: 在敌意电子攻击(如宽带干扰攻击)下提供高质量的通信是一项具有挑战性的任务。不同于现有的将干扰信号视为破坏性干扰的方法，提出了一种新的针对干扰信道的有源抗扰(AAJ)方案，以提高发送节点(TN)和接收节点(RN)之间的通信质量，其中TN主动利用干扰信号作为发送消息的载波。具体地说，TN配备了可编程增益放大器，能够对干扰信号进行重新调制以进行干扰调制。考虑到四种典型的干扰类型，我们推导了AAJ方案的误比特率和相应的最优检测门限。讨论了AAJ方案在高信噪比和高采样率情况下的渐近性能。我们的分析表明，对于足够大的Jnr，存在一个误码率下限。仿真结果表明，所提出的AAJ方案允许TN在极强和/或宽带干扰下与RN可靠地通信。此外，我们还对所提出的AAJ方案的信道容量进行了研究，结果表明，当JNR相对较高时，AAJ方案的信道容量优于直接传输。



## **36. Identifying a Training-Set Attack's Target Using Renormalized Influence Estimation**

基于重整化影响估计的训练集攻击目标识别 cs.LG

Accepted at CCS'2022 -- Extended version including the supplementary  material

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2201.10055v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstracts**: Targeted training-set attacks inject malicious instances into the training set to cause a trained model to mislabel one or more specific test instances. This work proposes the task of target identification, which determines whether a specific test instance is the target of a training-set attack. Target identification can be combined with adversarial-instance identification to find (and remove) the attack instances, mitigating the attack with minimal impact on other predictions. Rather than focusing on a single attack method or data modality, we build on influence estimation, which quantifies each training instance's contribution to a model's prediction. We show that existing influence estimators' poor practical performance often derives from their over-reliance on training instances and iterations with large losses. Our renormalized influence estimators fix this weakness; they far outperform the original estimators at identifying influential groups of training examples in both adversarial and non-adversarial settings, even finding up to 100% of adversarial training instances with no clean-data false positives. Target identification then simplifies to detecting test instances with anomalous influence values. We demonstrate our method's effectiveness on backdoor and poisoning attacks across various data domains, including text, vision, and speech, as well as against a gray-box, adaptive attacker that specifically optimizes the adversarial instances to evade our method. Our source code is available at https://github.com/ZaydH/target_identification.

摘要: 有针对性的训练集攻击将恶意实例注入训练集，以导致训练模型错误标记一个或多个特定测试实例。本文提出了目标识别的任务，即确定特定的测试实例是否是训练集攻击的目标。目标识别可以与对抗性实例识别相结合来发现(和删除)攻击实例，从而在对其他预测影响最小的情况下减轻攻击。我们不是专注于单一的攻击方法或数据模式，而是建立在影响估计的基础上，该估计量化了每个训练实例对模型预测的贡献。我们指出，现有影响估值器的实际性能较差往往源于它们过度依赖训练实例和具有较大损失的迭代。我们的重整化影响估计器修复了这一弱点；在识别对抗性和非对抗性设置中有影响力的训练样本组方面，它们远远优于原始估计器，甚至可以发现高达100%的对抗性训练实例，而没有干净的数据误报。然后，目标识别简化为检测具有异常影响值的测试实例。我们展示了我们的方法在各种数据领域的后门和中毒攻击中的有效性，包括文本、视觉和语音，以及针对灰盒、自适应攻击者的攻击，该攻击者专门优化敌对实例来逃避我们的方法。我们的源代码可以在https://github.com/ZaydH/target_identification.上找到



## **37. "Is your explanation stable?": A Robustness Evaluation Framework for Feature Attribution**

“你的解释稳定吗？”：特征归因的稳健性评估框架 cs.AI

Accepted by ACM CCS 2022

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01782v1)

**Authors**: Yuyou Gan, Yuhao Mao, Xuhong Zhang, Shouling Ji, Yuwen Pu, Meng Han, Jianwei Yin, Ting Wang

**Abstracts**: Understanding the decision process of neural networks is hard. One vital method for explanation is to attribute its decision to pivotal features. Although many algorithms are proposed, most of them solely improve the faithfulness to the model. However, the real environment contains many random noises, which may leads to great fluctuations in the explanations. More seriously, recent works show that explanation algorithms are vulnerable to adversarial attacks. All of these make the explanation hard to trust in real scenarios.   To bridge this gap, we propose a model-agnostic method \emph{Median Test for Feature Attribution} (MeTFA) to quantify the uncertainty and increase the stability of explanation algorithms with theoretical guarantees. MeTFA has the following two functions: (1) examine whether one feature is significantly important or unimportant and generate a MeTFA-significant map to visualize the results; (2) compute the confidence interval of a feature attribution score and generate a MeTFA-smoothed map to increase the stability of the explanation. Experiments show that MeTFA improves the visual quality of explanations and significantly reduces the instability while maintaining the faithfulness. To quantitatively evaluate the faithfulness of an explanation under different noise settings, we further propose several robust faithfulness metrics. Experiment results show that the MeTFA-smoothed explanation can significantly increase the robust faithfulness. In addition, we use two scenarios to show MeTFA's potential in the applications. First, when applied to the SOTA explanation method to locate context bias for semantic segmentation models, MeTFA-significant explanations use far smaller regions to maintain 99\%+ faithfulness. Second, when tested with different explanation-oriented attacks, MeTFA can help defend vanilla, as well as adaptive, adversarial attacks against explanations.

摘要: 理解神经网络的决策过程是很困难的。解释的一个重要方法是将其决定归因于关键特征。虽然有很多算法被提出，但大多数算法都只是提高了对模型的忠诚度。然而，现实环境中包含了许多随机噪声，这可能会导致解释中的巨大波动。更严重的是，最近的研究表明，解释算法容易受到敌意攻击。所有这些都使得这种解释在真实情况下很难令人信任。为了弥补这一差距，我们提出了一种模型不可知的方法\emph(特征属性的中位数测试)来量化不确定性并在理论上保证解释算法的稳定性。MeTFA具有以下两个功能：(1)检查一个特征是否显著重要，并生成MeTFA显著图以可视化结果；(2)计算特征属性得分的置信度，并生成MeTFA平滑的图，以增加解释的稳定性。实验表明，MeTFA在保持解释真实性的同时，提高了解释的视觉质量，显著降低了解释的不稳定性。为了定量评估不同噪声环境下解释的真实性，我们进一步提出了几种稳健的忠诚度度量。实验结果表明，经MeTFA平滑后的解释能够显著提高系统的稳健忠实性。此外，我们还通过两个场景展示了MeTFA在应用中的潜力。首先，当应用于SOTA解释方法来定位语义分割模型的语境偏差时，MeTFA显著解释使用小得多的区域来保持99+的忠实性。其次，当使用不同的面向解释的攻击进行测试时，MeTFA可以帮助防御普通攻击，以及针对解释的适应性对抗性攻击。



## **38. An Adaptive Black-box Defense against Trojan Attacks (TrojDef)**

木马攻击的自适应黑盒防御(TrojDef) cs.CR

**SubmitDate**: 2022-09-05    [paper-pdf](http://arxiv.org/pdf/2209.01721v1)

**Authors**: Guanxiong Liu, Abdallah Khreishah, Fatima Sharadgah, Issa Khalil

**Abstracts**: Trojan backdoor is a poisoning attack against Neural Network (NN) classifiers in which adversaries try to exploit the (highly desirable) model reuse property to implant Trojans into model parameters for backdoor breaches through a poisoned training process. Most of the proposed defenses against Trojan attacks assume a white-box setup, in which the defender either has access to the inner state of NN or is able to run back-propagation through it. In this work, we propose a more practical black-box defense, dubbed TrojDef, which can only run forward-pass of the NN. TrojDef tries to identify and filter out Trojan inputs (i.e., inputs augmented with the Trojan trigger) by monitoring the changes in the prediction confidence when the input is repeatedly perturbed by random noise. We derive a function based on the prediction outputs which is called the prediction confidence bound to decide whether the input example is Trojan or not. The intuition is that Trojan inputs are more stable as the misclassification only depends on the trigger, while benign inputs will suffer when augmented with noise due to the perturbation of the classification features.   Through mathematical analysis, we show that if the attacker is perfect in injecting the backdoor, the Trojan infected model will be trained to learn the appropriate prediction confidence bound, which is used to distinguish Trojan and benign inputs under arbitrary perturbations. However, because the attacker might not be perfect in injecting the backdoor, we introduce a nonlinear transform to the prediction confidence bound to improve the detection accuracy in practical settings. Extensive empirical evaluations show that TrojDef significantly outperforms the-state-of-the-art defenses and is highly stable under different settings, even when the classifier architecture, the training process, or the hyper-parameters change.

摘要: 特洛伊木马后门是一种针对神经网络(NN)分类器的中毒攻击，攻击者试图利用(非常理想的)模型重用属性，通过有毒的训练过程将特洛伊木马植入到后门入侵的模型参数中。大多数针对特洛伊木马攻击的防御建议采用白盒设置，在白盒设置中，防御者要么可以访问NN的内部状态，要么能够通过它进行反向传播。在这项工作中，我们提出了一种更实用的黑盒防御方法，称为TrojDef，它只能运行神经网络的前传。当输入被随机噪声反复扰动时，TrojDef试图通过监视预测置信度的变化来识别和过滤特洛伊木马输入(即，使用特洛伊木马触发器增强的输入)。我们根据预测输出推导出一个函数，称为预测置信限，用来判断输入示例是否为木马。直觉是，木马输入更稳定，因为误分类只取决于触发器，而良性输入将由于分类特征的扰动而在噪声中增强时受到影响。通过数学分析，我们证明了如果攻击者是完美的后门注入，木马感染模型将被训练来学习适当的预测置信界，用于区分任意扰动下的木马和良性输入。然而，由于攻击者在注入后门时可能并不完美，我们引入了对预测置信限的非线性变换，以提高实际环境中的检测精度。大量的实验评估表明，TrojDef的性能显著优于最先进的防御措施，并且在不同的设置下具有高度的稳定性，即使在分类器结构、训练过程或超参数发生变化的情况下也是如此。



## **39. Autonomous Cross Domain Adaptation under Extreme Label Scarcity**

极端标签稀缺下的自主跨域自适应 cs.LG

**SubmitDate**: 2022-09-04    [paper-pdf](http://arxiv.org/pdf/2209.01548v1)

**Authors**: Weiwei Weng, Mahardhika Pratama, Choiru Za'in, Marcus De Carvalho, Rakaraddi Appan, Andri Ashfahani, Edward Yapp Kien Yee

**Abstracts**: A cross domain multistream classification is a challenging problem calling for fast domain adaptations to handle different but related streams in never-ending and rapidly changing environments. Notwithstanding that existing multistream classifiers assume no labelled samples in the target stream, they still incur expensive labelling cost since they require fully labelled samples of the source stream. This paper aims to attack the problem of extreme label shortage in the cross domain multistream classification problems where only very few labelled samples of the source stream are provided before process runs. Our solution, namely Learning Streaming Process from Partial Ground Truth (LEOPARD), is built upon a flexible deep clustering network where its hidden nodes, layers and clusters are added and removed dynamically in respect to varying data distributions. A deep clustering strategy is underpinned by a simultaneous feature learning and clustering technique leading to clustering-friendly latent spaces. A domain adaptation strategy relies on the adversarial domain adaptation technique where a feature extractor is trained to fool a domain classifier classifying source and target streams. Our numerical study demonstrates the efficacy of LEOPARD where it delivers improved performances compared to prominent algorithms in 15 of 24 cases. Source codes of LEOPARD are shared in \url{https://github.com/wengweng001/LEOPARD.git} to enable further study.

摘要: 跨域多流分类是一个具有挑战性的问题，需要快速的域自适应来处理永无止境和快速变化的环境中不同但相关的流。尽管现有的多数据流分类器假定目标流中没有标记的样本，但它们仍然招致昂贵的标记成本，因为它们需要源流的完全标记的样本。本文旨在解决跨域多数据流分类问题中的极端标签短缺问题，即在处理运行之前只提供极少的源流的标签样本。我们的解决方案，即从部分地面真相学习流媒体过程(LEOPARD)，建立在一个灵活的深度聚类网络之上，其中隐藏的节点、层和簇根据不同的数据分布动态地添加和删除。深度聚类策略由同时的特征学习和聚类技术支持，从而产生对聚类友好的潜在空间。域自适应策略依赖于对抗性域自适应技术，其中特征提取者被训练来愚弄对源和目标流进行分类的域分类器。我们的数值研究证明了Leopard的有效性，与24个案例中的15个案例相比，它提供了比著名算法更好的性能。Leopard的源代码在\url{https://github.com/wengweng001/LEOPARD.git}上共享，以便进一步研究。



## **40. Are Attribute Inference Attacks Just Imputation?**

属性推理攻击仅仅是归罪吗？ cs.CR

13 (main body) + 4 (references and appendix) pages. To appear in  CCS'22

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.01292v1)

**Authors**: Bargav Jayaraman, David Evans

**Abstracts**: Models can expose sensitive information about their training data. In an attribute inference attack, an adversary has partial knowledge of some training records and access to a model trained on those records, and infers the unknown values of a sensitive feature of those records. We study a fine-grained variant of attribute inference we call \emph{sensitive value inference}, where the adversary's goal is to identify with high confidence some records from a candidate set where the unknown attribute has a particular sensitive value. We explicitly compare attribute inference with data imputation that captures the training distribution statistics, under various assumptions about the training data available to the adversary. Our main conclusions are: (1) previous attribute inference methods do not reveal more about the training data from the model than can be inferred by an adversary without access to the trained model, but with the same knowledge of the underlying distribution as needed to train the attribute inference attack; (2) black-box attribute inference attacks rarely learn anything that cannot be learned without the model; but (3) white-box attacks, which we introduce and evaluate in the paper, can reliably identify some records with the sensitive value attribute that would not be predicted without having access to the model. Furthermore, we show that proposed defenses such as differentially private training and removing vulnerable records from training do not mitigate this privacy risk. The code for our experiments is available at \url{https://github.com/bargavj/EvaluatingDPML}.

摘要: 模型可能会暴露有关其训练数据的敏感信息。在属性推理攻击中，敌手对一些训练记录具有部分知识，并访问在这些记录上训练的模型，并推断这些记录的敏感特征的未知值。我们研究了属性推理的一种细粒度变体，称为\emph(敏感值推理)，其中对手的目标是高置信度地从未知属性具有特定敏感值的候选集合中识别一些记录。在关于对手可用的训练数据的各种假设下，我们显式地将属性推理与捕获训练分布统计的数据补偿进行比较。我们的主要结论是：(1)以前的属性推理方法并不能揭示更多关于模型训练数据的信息，而不是对手无法访问训练的模型所能推断的数据，而是具有与训练属性推理攻击所需的相同的底层分布知识；(2)黑盒属性推理攻击很少学习没有模型无法学习的任何东西；但是(3)我们在论文中引入和评估的白盒攻击可以可靠地识别一些具有敏感值属性的记录，这些记录在没有访问模型的情况下是无法预测的。此外，我们表明，建议的防御措施，如区分隐私培训和从培训中删除易受攻击的记录，并不能减轻这种隐私风险。我们实验的代码可以在\url{https://github.com/bargavj/EvaluatingDPML}.上找到



## **41. Semi-supervised Conditional GAN for Simultaneous Generation and Detection of Phishing URLs: A Game theoretic Perspective**

基于博弈论的半监督条件遗传算法同时生成和检测钓鱼URL cs.CR

Accepted to ICMLA 2022

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2108.01852v2)

**Authors**: Sharif Amit Kamran, Shamik Sengupta, Alireza Tavakkoli

**Abstracts**: Spear Phishing is a type of cyber-attack where the attacker sends hyperlinks through email on well-researched targets. The objective is to obtain sensitive information by imitating oneself as a trustworthy website. In recent times, deep learning has become the standard for defending against such attacks. However, these architectures were designed with only defense in mind. Moreover, the attacker's perspective and motivation are absent while creating such models. To address this, we need a game-theoretic approach to understand the perspective of the attacker (Hacker) and the defender (Phishing URL detector). We propose a Conditional Generative Adversarial Network with novel training strategy for real-time phishing URL detection. Additionally, we train our architecture in a semi-supervised manner to distinguish between adversarial and real examples, along with detecting malicious and benign URLs. We also design two games between the attacker and defender in training and deployment settings by utilizing the game-theoretic perspective. Our experiments confirm that the proposed architecture surpasses recent state-of-the-art architectures for phishing URLs detection.

摘要: 鱼叉式网络钓鱼是一种网络攻击，攻击者通过电子邮件向经过充分研究的目标发送超链接。其目标是通过将自己伪装成一个值得信赖的网站来获取敏感信息。最近，深度学习已成为防御此类攻击的标准。然而，这些架构在设计时只考虑到了防御。此外，在创建这样的模型时，攻击者的视角和动机是缺失的。为了解决这个问题，我们需要一个博弈论的方法来理解攻击者(黑客)和防御者(网络钓鱼URL检测器)的角度。提出了一种具有新颖训练策略的条件生成对抗网络，用于实时网络钓鱼URL检测。此外，我们以半监督的方式训练我们的体系结构，以区分敌意和真实的示例，以及检测恶意和良性URL。我们还利用博弈论的观点设计了攻防双方在训练和部署环境下的两场比赛。我们的实验证实，该体系结构在网络钓鱼URL检测方面优于目前最先进的体系结构。



## **42. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

贝叶斯伪标签：稳健有效的半监督分割的期望最大化 cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2208.04435v2)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.

摘要: 本文研究的是分割中的伪标注问题。我们的贡献是四倍的。首先，我们提出了一种新的伪标记公式，作为一种用于清晰统计解释的期望最大化(EM)算法。其次，提出了一种完全基于原始伪标记的半监督医学图像分割方法--SegPL。在2D多类MRI脑肿瘤分割任务和3D二值CT肺血管分割任务中，我们证明了SegPL是一种与最先进的基于一致性正则化的半监督分割方法相竞争的方法。与以前的方法相比，SegPL的简单性允许更少的计算成本。第三，我们证明了SegPL的有效性可能源于它对分布外噪声和对手攻击的健壮性。最后，在EM框架下，我们通过变分推理对SegPL进行概率推广，在训练过程中学习伪标签的动态阈值。我们证明了带变分推理的SegPL方法可以与金标准方法深层集成一样进行不确定度估计。



## **43. Subject Membership Inference Attacks in Federated Learning**

联合学习中的主体成员推理攻击 cs.LG

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2206.03317v2)

**Authors**: Anshuman Suri, Pallika Kanani, Virendra J. Marathe, Daniel W. Peterson

**Abstracts**: Privacy attacks on Machine Learning (ML) models often focus on inferring the existence of particular data points in the training data. However, what the adversary really wants to know is if a particular \emph{individual}'s (\emph{subject}'s) data was included during training. In such scenarios, the adversary is more likely to have access to the distribution of a particular subject, than actual records. Furthermore, in settings like cross-silo Federated Learning (FL), a subject's data can be embodied by multiple data records that are spread across multiple organizations. Nearly all of the existing private FL literature is dedicated to studying privacy at two granularities -- item-level (individual data records), and user-level (participating user in the federation), neither of which apply to data subjects in cross-silo FL. This insight motivates us to shift our attention from the privacy of data records to the privacy of \emph{data subjects}, also known as subject-level privacy. We propose two black-box attacks for \emph{subject membership inference}, of which one assumes access to a model after each training round. Using these attacks, we estimate subject membership inference risk on real-world data for single-party models as well as FL scenarios. We find our attacks to be extremely potent, even without access to exact training records, and using the knowledge of membership for a handful of subjects. To better understand the various factors that may influence subject privacy risk in cross-silo FL settings, we systematically generate several hundred synthetic federation configurations, varying properties of the data, model design and training, and the federation itself. Finally, we investigate the effectiveness of Differential Privacy in mitigating this threat.

摘要: 针对机器学习(ML)模型的隐私攻击通常集中在推断训练数据中特定数据点的存在。然而，对手真正想知道的是，在训练过程中是否包括了特定的\emph{个人}的数据。在这种情况下，对手更有可能访问特定主题的分布，而不是实际记录。此外，在像跨竖井联合学习(FL)这样的环境中，受试者的数据可以通过分布在多个组织中的多个数据记录来体现。几乎所有现有的私有FL文献都致力于在两个粒度上研究隐私--项级(个人数据记录)和用户级(参与联盟的用户)，这两者都不适用于跨竖井FL中的数据主体。这种洞察力促使我们将注意力从数据记录的隐私转移到数据主题的隐私，也称为主题级别隐私。我们提出了两种针对主题成员推理的黑盒攻击，其中一种假设在每一轮训练后都可以访问模型。使用这些攻击，我们在单方模型和FL场景的真实世界数据上估计了主体成员关系推断风险。我们发现我们的攻击非常强大，即使没有获得确切的训练记录，并使用少数科目的成员知识。为了更好地了解在跨竖井FL设置中可能影响主体隐私风险的各种因素，我们系统地生成了数百个合成联邦配置、数据的不同属性、模型设计和训练以及联邦本身。最后，我们研究了差分隐私在缓解这一威胁方面的有效性。



## **44. Group Property Inference Attacks Against Graph Neural Networks**

针对图神经网络的群属性推理攻击 cs.LG

Full version of the ACM CCS'22 paper

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.01100v1)

**Authors**: Xiuling Wang, Wendy Hui Wang

**Abstracts**: With the fast adoption of machine learning (ML) techniques, sharing of ML models is becoming popular. However, ML models are vulnerable to privacy attacks that leak information about the training data. In this work, we focus on a particular type of privacy attacks named property inference attack (PIA) which infers the sensitive properties of the training data through the access to the target ML model. In particular, we consider Graph Neural Networks (GNNs) as the target model, and distribution of particular groups of nodes and links in the training graph as the target property. While the existing work has investigated PIAs that target at graph-level properties, no prior works have studied the inference of node and link properties at group level yet.   In this work, we perform the first systematic study of group property inference attacks (GPIA) against GNNs. First, we consider a taxonomy of threat models under both black-box and white-box settings with various types of adversary knowledge, and design six different attacks for these settings. We evaluate the effectiveness of these attacks through extensive experiments on three representative GNN models and three real-world graphs. Our results demonstrate the effectiveness of these attacks whose accuracy outperforms the baseline approaches. Second, we analyze the underlying factors that contribute to GPIA's success, and show that the target model trained on the graphs with or without the target property represents some dissimilarity in model parameters and/or model outputs, which enables the adversary to infer the existence of the property. Further, we design a set of defense mechanisms against the GPIA attacks, and demonstrate that these mechanisms can reduce attack accuracy effectively with small loss on GNN model accuracy.

摘要: 随着机器学习(ML)技术的快速采用，ML模型的共享变得越来越流行。然而，ML模型容易受到隐私攻击，从而泄露有关训练数据的信息。在这项工作中，我们重点研究了一种特殊类型的隐私攻击，称为属性推理攻击(PIA)，它通过访问目标ML模型来推断训练数据的敏感属性。特别地，我们将图神经网络(GNN)作为目标模型，将训练图中特定节点组和链路组的分布作为目标属性。虽然现有的工作已经研究了针对图级属性的PIA，但还没有先前的工作研究在组级别上的节点和链接属性的推理。在本工作中，我们首次系统地研究了针对GNN的群属性推理攻击(GPIA)。首先，我们考虑了黑盒和白盒两种不同类型对手知识的威胁模型的分类，并针对这些设置设计了六种不同的攻击。我们通过在三个具有代表性的GNN模型和三个真实世界图上的大量实验来评估这些攻击的有效性。我们的结果证明了这些攻击的有效性，它们的准确率超过了基线方法。其次，我们分析了导致GPIA成功的潜在因素，并证明了在有或没有目标属性的图上训练的目标模型表示了模型参数和/或模型输出的一些不同，这使得对手能够推断该属性的存在。进一步，我们设计了一套针对GPIA攻击的防御机制，并证明了这些机制可以在较小的GNN模型精度损失的情况下有效地降低攻击精度。



## **45. Scalable Adversarial Attack Algorithms on Influence Maximization**

基于影响力最大化的可扩展敌意攻击算法 cs.SI

11 pages, 2 figures

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.00892v1)

**Authors**: Lichao Sun, Xiaobin Rui, Wei Chen

**Abstracts**: In this paper, we study the adversarial attacks on influence maximization under dynamic influence propagation models in social networks. In particular, given a known seed set S, the problem is to minimize the influence spread from S by deleting a limited number of nodes and edges. This problem reflects many application scenarios, such as blocking virus (e.g. COVID-19) propagation in social networks by quarantine and vaccination, blocking rumor spread by freezing fake accounts, or attacking competitor's influence by incentivizing some users to ignore the information from the competitor. In this paper, under the linear threshold model, we adapt the reverse influence sampling approach and provide efficient algorithms of sampling valid reverse reachable paths to solve the problem.

摘要: 本文研究了社会网络中动态影响传播模型下影响最大化的对抗性攻击。特别地，给定一个已知的种子集S，问题是通过删除有限数量的节点和边来最小化从S传播的影响。这个问题反映了很多应用场景，比如通过隔离和接种疫苗来阻止病毒(例如新冠肺炎)在社交网络中的传播，通过冻结虚假账号来阻止谣言传播，或者通过激励一些用户忽略竞争对手的信息来攻击竞争对手的影响力。本文在线性门限模型下，采用反向影响抽样方法，给出了有效反向可达路径抽样的有效算法。



## **46. Adversarial Color Film: Effective Physical-World Attack to DNNs**

对抗性彩色电影：对DNN的有效物理世界攻击 cs.CV

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02430v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: It is well known that the performance of deep neural networks (DNNs) is susceptible to subtle interference. So far, camera-based physical adversarial attacks haven't gotten much attention, but it is the vacancy of physical attack. In this paper, we propose a simple and efficient camera-based physical attack called Adversarial Color Film (AdvCF), which manipulates the physical parameters of color film to perform attacks. Carefully designed experiments show the effectiveness of the proposed method in both digital and physical environments. In addition, experimental results show that the adversarial samples generated by AdvCF have excellent performance in attack transferability, which enables AdvCF effective black-box attacks. At the same time, we give the guidance of defense against AdvCF by means of adversarial training. Finally, we look into AdvCF's threat to future vision-based systems and propose some promising mentality for camera-based physical attacks.

摘要: 众所周知，深度神经网络(DNN)的性能容易受到细微干扰的影响。到目前为止，基于摄像机的身体对抗攻击还没有得到太多的关注，但它是身体攻击的空白。本文提出了一种简单而有效的基于摄像机的物理攻击方法，称为对抗性彩色胶片攻击(AdvCF)，它通过操纵彩色胶片的物理参数来进行攻击。精心设计的实验表明，该方法在数字和物理环境中都是有效的。此外，实验结果表明，由AdvCF生成的对抗性样本具有良好的攻击可转移性，使得AdvCF能够有效地进行黑盒攻击。同时，通过对抗性训练的方式，指导对Advcf的防御。最后，我们展望了AdvCF对未来基于视觉的系统的威胁，并对基于摄像机的物理攻击提出了一些有前景的思路。



## **47. Impact of Colour Variation on Robustness of Deep Neural Networks**

颜色变化对深度神经网络稳健性的影响 cs.CV

arXiv admin note: substantial text overlap with arXiv:2209.02132

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02832v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have have shown state-of-the-art performance for computer vision applications like image classification, segmentation and object detection. Whereas recent advances have shown their vulnerability to manual digital perturbations in the input data, namely adversarial attacks. The accuracy of the networks is significantly affected by the data distribution of their training dataset. Distortions or perturbations on color space of input images generates out-of-distribution data, which make networks more likely to misclassify them. In this work, we propose a color-variation dataset by distorting their RGB color on a subset of the ImageNet with 27 different combinations. The aim of our work is to study the impact of color variation on the performance of DNNs. We perform experiments on several state-of-the-art DNN architectures on the proposed dataset, and the result shows a significant correlation between color variation and loss of accuracy. Furthermore, based on the ResNet50 architecture, we demonstrate some experiments of the performance of recently proposed robust training techniques and strategies, such as Augmix, revisit, and free normalizer, on our proposed dataset. Experimental results indicate that these robust training techniques can improve the robustness of deep networks to color variation.

摘要: 深度神经网络(DNN)在图像分类、分割和目标检测等计算机视觉应用中表现出了最先进的性能。然而，最近的进展表明，它们容易受到输入数据中的人工数字扰动，即对抗性攻击。训练数据集的数据分布对网络的精度有很大的影响。输入图像颜色空间的失真或扰动会产生不分布的数据，这使得网络更有可能对它们进行错误分类。在这项工作中，我们提出了一个颜色变化数据集，通过在ImageNet的一个子集上使用27种不同的组合来扭曲它们的RGB颜色。我们工作的目的是研究颜色变化对DNN性能的影响。我们在提出的数据集上对几种最先进的DNN结构进行了实验，结果表明颜色变化与准确率损失之间存在显著的相关性。此外，基于ResNet50体系结构，我们展示了最近提出的稳健训练技术和策略的一些实验，例如AugMix、Revises和Free Normizer，在我们提出的数据集上。实验结果表明，这些稳健的训练技术可以提高深度网络对颜色变化的鲁棒性。



## **48. Impact of Scaled Image on Robustness of Deep Neural Networks**

尺度图像对深度神经网络稳健性的影响 cs.CV

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.02132v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstracts**: Deep neural networks (DNNs) have been widely used in computer vision tasks like image classification, object detection and segmentation. Whereas recent studies have shown their vulnerability to manual digital perturbations or distortion in the input images. The accuracy of the networks is remarkably influenced by the data distribution of their training dataset. Scaling the raw images creates out-of-distribution data, which makes it a possible adversarial attack to fool the networks. In this work, we propose a Scaling-distortion dataset ImageNet-CS by Scaling a subset of the ImageNet Challenge dataset by different multiples. The aim of our work is to study the impact of scaled images on the performance of advanced DNNs. We perform experiments on several state-of-the-art deep neural network architectures on the proposed ImageNet-CS, and the results show a significant positive correlation between scaling size and accuracy decline. Moreover, based on ResNet50 architecture, we demonstrate some tests on the performance of recent proposed robust training techniques and strategies like Augmix, Revisiting and Normalizer Free on our proposed ImageNet-CS. Experiment results have shown that these robust training techniques can improve networks' robustness to scaling transformation.

摘要: 深度神经网络在图像分类、目标检测和分割等计算机视觉任务中有着广泛的应用。然而，最近的研究表明，它们在输入图像中容易受到人工数字干扰或失真的影响。训练数据集的数据分布对网络的精度有很大影响。对原始图像进行缩放会产生分布不均的数据，这使得它可能成为愚弄网络的敌意攻击。在这项工作中，我们通过对ImageNet挑战数据集的子集进行不同倍数的缩放，提出了一个缩放失真数据集ImageNet-CS。我们工作的目的是研究缩放图像对高级DNN性能的影响。我们在提出的ImageNet-CS上对几种最先进的深度神经网络结构进行了实验，结果表明，尺度大小与准确率下降呈显著正相关。此外，基于ResNet50体系结构，我们在我们提出的ImageNet-CS上对最近提出的健壮训练技术和策略，如AugMix、Revising和Normal izer Free的性能进行了测试。实验结果表明，这些稳健的训练技术可以提高网络对尺度变换的鲁棒性。



## **49. Reliable Representations Make A Stronger Defender: Unsupervised Structure Refinement for Robust GNN**

可靠的表示使防御者更强大：健壮GNN的无监督结构求精 cs.LG

Accepted in KDD2022

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2207.00012v2)

**Authors**: Kuan Li, Yang Liu, Xiang Ao, Jianfeng Chi, Jinghua Feng, Hao Yang, Qing He

**Abstracts**: Benefiting from the message passing mechanism, Graph Neural Networks (GNNs) have been successful on flourish tasks over graph data. However, recent studies have shown that attackers can catastrophically degrade the performance of GNNs by maliciously modifying the graph structure. A straightforward solution to remedy this issue is to model the edge weights by learning a metric function between pairwise representations of two end nodes, which attempts to assign low weights to adversarial edges. The existing methods use either raw features or representations learned by supervised GNNs to model the edge weights. However, both strategies are faced with some immediate problems: raw features cannot represent various properties of nodes (e.g., structure information), and representations learned by supervised GNN may suffer from the poor performance of the classifier on the poisoned graph. We need representations that carry both feature information and as mush correct structure information as possible and are insensitive to structural perturbations. To this end, we propose an unsupervised pipeline, named STABLE, to optimize the graph structure. Finally, we input the well-refined graph into a downstream classifier. For this part, we design an advanced GCN that significantly enhances the robustness of vanilla GCN without increasing the time complexity. Extensive experiments on four real-world graph benchmarks demonstrate that STABLE outperforms the state-of-the-art methods and successfully defends against various attacks.

摘要: 得益于消息传递机制，图神经网络(GNN)已经成功地处理了大量的图数据任务。然而，最近的研究表明，攻击者可以通过恶意修改图结构来灾难性地降低GNN的性能。解决这一问题的一个直接解决方案是通过学习两个末端节点的成对表示之间的度量函数来对边权重进行建模，该度量函数试图为对抗性边分配较低的权重。现有的方法要么使用原始特征，要么使用由监督GNN学习的表示来对边权重进行建模。然而，这两种策略都面临着一些迫在眉睫的问题：原始特征不能表示节点的各种属性(例如结构信息)，而有监督GNN学习的表示可能会受到有毒图上分类器性能较差的影响。我们需要既携带特征信息又尽可能正确的结构信息并对结构扰动不敏感的表示法。为此，我们提出了一种名为STRATE的无监督流水线来优化图的结构。最后，我们将精化后的图输入到下游分类器中。对于这一部分，我们设计了一种改进的GCN，它在不增加时间复杂度的情况下显著增强了普通GCN的健壮性。在四个真实图形基准上的大量实验表明，STRATE的性能优于最先进的方法，并成功地防御了各种攻击。



## **50. Universal Fourier Attack for Time Series**

时间序列的通用傅里叶攻击 cs.CR

**SubmitDate**: 2022-09-02    [paper-pdf](http://arxiv.org/pdf/2209.00757v1)

**Authors**: Elizabeth Coda, Brad Clymer, Chance DeSmet, Yijing Watkins, Michael Girard

**Abstracts**: A wide variety of adversarial attacks have been proposed and explored using image and audio data. These attacks are notoriously easy to generate digitally when the attacker can directly manipulate the input to a model, but are much more difficult to implement in the real-world. In this paper we present a universal, time invariant attack for general time series data such that the attack has a frequency spectrum primarily composed of the frequencies present in the original data. The universality of the attack makes it fast and easy to implement as no computation is required to add it to an input, while time invariance is useful for real-world deployment. Additionally, the frequency constraint ensures the attack can withstand filtering. We demonstrate the effectiveness of the attack in two different domains, speech recognition and unintended radiated emission, and show that the attack is robust against common transform-and-compare defense pipelines.

摘要: 已经提出并探索了使用图像和音频数据的各种对抗性攻击。当攻击者可以直接操作模型的输入时，这些攻击以数字方式生成是出了名的容易，但在现实世界中实现起来要困难得多。在本文中，我们提出了一种针对一般时间序列数据的通用、时不变攻击，使得该攻击具有主要由原始数据中存在的频率组成的频谱。该攻击的普遍性使其易于快速实现，因为不需要计算即可将其添加到输入，而时间不变性对于真实世界的部署很有用。此外，频率限制确保了攻击能够经受住过滤。我们在语音识别和意外辐射两个不同的领域证明了该攻击的有效性，并证明了该攻击对常见的变换和比较防御流水线具有健壮性。



