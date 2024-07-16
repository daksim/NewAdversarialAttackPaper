# Latest Adversarial Attack Papers
**update at 2024-07-16 09:53:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

17 pages

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2404.09005v6) [paper-pdf](http://arxiv.org/pdf/2404.09005v6)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **2. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



## **3. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

通过沿着对抗轨迹交叉区域的多样化来提高视觉语言攻击的可移植性 cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2403.12445v3) [paper-pdf](http://arxiv.org/pdf/2403.12445v3)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.

摘要: 视觉-语言预训练(VLP)模型在理解图像和文本方面表现出显著的能力，但它们仍然容易受到多通道对抗性例子(AEs)的影响。加强攻击和发现漏洞，特别是VLP模型中的常见问题(例如，高可转移的AE)，可以促进可靠和实用的VLP模型。最近的一项工作(即集合级制导攻击)表明，增加图文对以增加优化路径上的声发射多样性显著地提高了对抗性例子的可转移性。然而，这种方法主要强调围绕在线对抗性例子的多样性(即，处于优化期的AEs)，导致过度匹配受害者模型并影响可转移性的风险。在这项研究中，我们假设，针对干净输入和在线AEs的对抗性例子的多样性对于提高VLP模型之间的可转移性都是关键。因此，我们建议沿着对抗性轨迹的交叉点区域进行多样化，以扩大AEs的多样性。为了充分利用通道之间的交互作用，我们在优化过程中引入了文本引导的对抗性实例选择。此外，为了进一步缓解潜在的过拟合，我们沿着优化路径引导偏离最后一个交集区域的对抗性文本，而不是现有方法中的对抗性图像。广泛的实验证实了我们的方法在提高各种VLP模型和下游视觉和语言任务的可转移性方面的有效性。



## **4. CLIP-Guided Networks for Transferable Targeted Attacks**

CLIP引导的可转移定向攻击网络 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10179v1) [paper-pdf](http://arxiv.org/pdf/2407.10179v1)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.

摘要: 可转移的目标对抗性攻击旨在误导模型，使其在黑盒场景中输出对手指定的预测。最近的研究引入了生成性攻击，这种攻击为每个目标类训练一个生成器来生成高度可传递的扰动，导致在处理多个类时产生大量的计算开销。\textit{多目标}攻击通过仅训练多个类的一个类条件生成器来解决此问题。然而，生成器简单地使用类标签作为条件，没有利用目标类的丰富语义信息。为此，我们设计了一个唇形引导的生成模块(CGNC)，通过在生成器中加入剪辑文本知识来增强多目标攻击。大量的实验表明，CGNC比以前的多目标生成性攻击有显著的改进，例如，成功率从ResNet-152提高到DenseNet-121，提高了21.46%.此外，我们还提出了一种屏蔽微调机制，进一步加强了我们的攻击单一类的方法，超越了现有的单目标攻击方法。



## **5. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

可以解析对抗性例子来揭示受害者模型信息吗？ cs.CV

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2303.07474v3) [paper-pdf](http://arxiv.org/pdf/2303.07474v3)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.

摘要: 已经开发了许多对抗性攻击方法来产生不可察觉的图像扰动，这可能导致对最先进的机器学习(ML)模型的错误预测，特别是深度神经网络(DNN)。尽管对对抗性攻击进行了密集的研究，但几乎没有努力去发现对抗性攻击中携带的“奥秘”。在这项工作中，我们问是否有可能从特定于数据的对抗性实例中推断出与数据无关的受害者模型(VM)信息(即，用于生成对抗性攻击的ML模型或DNN的特征)。我们称之为“对抗性攻击的模型解析”--根据攻击中隐藏的VM信息来发现“奥秘”的任务。我们通过有监督的学习来实现模型解析，它正确地将VM的模型属性的类别(根据体系结构类型、核大小、激活函数和权重稀疏性)分配给从该VM生成的攻击实例。我们收集了从135个受害者模型(由5个体系结构类型、3个核大小设置、3个激活函数类型和3个权重稀疏率配置)生成的7种攻击类型的对抗性攻击的数据集。我们证明了一个简单的有监督的模型解析网络(MPN)能够从未知的敌意攻击中推断出VM属性，如果它们的攻击设置与训练设置一致(即分布内泛化评估)。我们还提供了大量的实验，以验证在敌意攻击下进行VM解析的可行性，以及训练和评估因素(例如，在分布外评估中提出的泛化挑战)对解析性能的影响。我们进一步演示了如何使用所提出的MPN来发现来自传输攻击的源VM属性，并阐明了模型解析和攻击可转移性之间的潜在联系。



## **6. Transferable 3D Adversarial Shape Completion using Diffusion Models**

使用扩散模型的可转移3D对抗形状完成 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10077v1) [paper-pdf](http://arxiv.org/pdf/2407.10077v1)

**Authors**: Xuelong Dai, Bin Xiao

**Abstract**: Recent studies that incorporate geometric features and transformers into 3D point cloud feature learning have significantly improved the performance of 3D deep-learning models. However, their robustness against adversarial attacks has not been thoroughly explored. Existing attack methods primarily focus on white-box scenarios and struggle to transfer to recently proposed 3D deep-learning models. Even worse, these attacks introduce perturbations to 3D coordinates, generating unrealistic adversarial examples and resulting in poor performance against 3D adversarial defenses. In this paper, we generate high-quality adversarial point clouds using diffusion models. By using partial points as prior knowledge, we generate realistic adversarial examples through shape completion with adversarial guidance. The proposed adversarial shape completion allows for a more reliable generation of adversarial point clouds. To enhance attack transferability, we delve into the characteristics of 3D point clouds and employ model uncertainty for better inference of model classification through random down-sampling of point clouds. We adopt ensemble adversarial guidance for improved transferability across different network architectures. To maintain the generation quality, we limit our adversarial guidance solely to the critical points of the point clouds by calculating saliency scores. Extensive experiments demonstrate that our proposed attacks outperform state-of-the-art adversarial attack methods against both black-box models and defenses. Our black-box attack establishes a new baseline for evaluating the robustness of various 3D point cloud classification models.

摘要: 最近的研究将几何特征和变换融入到三维点云特征学习中，显著提高了三维深度学习模型的性能。然而，它们对敌意攻击的健壮性还没有得到彻底的研究。现有的攻击方法主要集中在白盒场景，很难转移到最近提出的3D深度学习模型。更糟糕的是，这些攻击对3D坐标引入了扰动，生成了不现实的对抗性示例，并导致对3D对抗性防御的性能不佳。在本文中，我们使用扩散模型来生成高质量的对抗性点云。利用局部点作为先验知识，通过对抗性指导下的形状补全生成真实的对抗性实例。所提出的对抗性形状补全允许更可靠地生成对抗性点云。为了增强攻击的可转移性，我们深入研究了三维点云的特点，并利用模型不确定性通过对点云进行随机下采样来更好地推断模型分类。我们采用集成对抗性指导，以提高跨不同网络架构的可传输性。为了保持生成质量，我们通过计算显著分数，将我们的对抗性指导仅限于点云的关键点。大量的实验表明，我们提出的攻击方法优于最新的对抗性攻击方法，无论是针对黑盒模型还是针对防御。我们的黑盒攻击为评估各种三维点云分类模型的稳健性建立了一个新的基线。



## **7. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

AdvDiff：使用扩散模型生成无限制的对抗示例 cs.LG

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2307.12499v4) [paper-pdf](http://arxiv.org/pdf/2307.12499v4)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often directly inject Projected Gradient Descent (PGD) gradients into the sampling of generative models, which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for GAN-based methods on large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable in generating high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective in generating unrestricted adversarial examples, which outperforms state-of-the-art unrestricted adversarial attack methods in terms of attack performance and generation quality.

摘要: 不受限制的对抗性攻击对深度学习模型和对抗性防御技术构成了严重威胁。它们会给深度学习应用程序带来严重的安全问题，因为它们可以有效地绕过防御机制。然而，以往的攻击方法往往直接将投影梯度下降(PGD)梯度注入生成模型的样本中，这在理论上是不可证明的，因此通过结合对抗性目标来生成不现实的例子，特别是对于基于GAN的方法在像ImageNet这样的大规模数据集上。在这篇文章中，我们提出了一种新的方法，称为AdvDiff，用来生成带有扩散模型的无限制对抗实例。我们设计了两种新的对抗性制导技术，用于在扩散模型的逆向生成过程中进行对抗性采样。这两种技术通过可解释地集成目标分类器的梯度，在生成高质量、真实的对抗性实例方面是有效和稳定的。在MNIST和ImageNet数据集上的实验结果表明，AdvDiff在生成无限制对抗性实例方面是有效的，在攻击性能和生成质量方面都优于现有的无限制对抗性攻击方法。



## **8. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

增强神经微调以实现高效后门净化 cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10052v1) [paper-pdf](http://arxiv.org/pdf/2407.10052v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too- sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.

摘要: 最近的研究揭示了深度神经网络(DNN)对各种后门攻击的脆弱性，其中DNN的行为可以通过利用某些类型的触发或中毒机制来危害。最先进的(SOTA)防御使用了过于复杂的机制，需要计算昂贵的对抗性搜索模块来对触发分布进行反向工程，或者需要过于敏感的超参数选择模块。此外，它们在具有挑战性的场景中提供了低于平均水平的性能，例如有限的验证数据和强大的攻击。在本文中，我们提出了神经掩码微调(NFT)，目的是以一种消除后门影响的方式来优化重组神经元的活动。利用简单的数据增强，如混合，NFT放松了触发器合成过程，并消除了对敌方搜索模块的要求。我们的研究进一步表明，在有限的验证数据下直接权重微调会导致净化后清洁测试的准确性较差，这主要是由于过度拟合问题。为了克服这一点，我们建议微调神经掩模而不是模型权重。此外，还设计了一种掩膜正则化算法，以进一步缓解纯化过程中的模型漂移。NFT的独特特性使得它在运行时和样本使用方面都非常高效，因为即使每个类只有一个样本可用，它也可以删除后门。通过在图像分类、目标检测、视频动作识别、三维点云和自然语言处理等方面的大量实验，验证了NFT的有效性。我们针对14种不同的攻击(Lira、WaNet等)对我们的方法进行了评估。基于ImageNet、UCF101、Pascal VOC、ModelNet、OpenSubtitles2012等11个基准数据集。



## **9. Harvesting Private Medical Images in Federated Learning Systems with Crafted Models**

使用精心设计的模型在联邦学习系统中收集私人医疗图像 cs.LG

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2407.09972v1) [paper-pdf](http://arxiv.org/pdf/2407.09972v1)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows a set of clients to collaboratively train a machine-learning model without exposing local training samples. In this context, it is considered to be privacy-preserving and hence has been adopted by medical centers to train machine-learning models over private data. However, in this paper, we propose a novel attack named MediLeak that enables a malicious parameter server to recover high-fidelity patient images from the model updates uploaded by the clients. MediLeak requires the server to generate an adversarial model by adding a crafted module in front of the original model architecture. It is published to the clients in the regular FL training process and each client conducts local training on it to generate corresponding model updates. Then, based on the FL protocol, the model updates are sent back to the server and our proposed analytical method recovers private data from the parameter updates of the crafted module. We provide a comprehensive analysis for MediLeak and show that it can successfully break the state-of-the-art cryptographic secure aggregation protocols, designed to protect the FL systems from privacy inference attacks. We implement MediLeak on the MedMNIST and COVIDx CXR-4 datasets. The results show that MediLeak can nearly perfectly recover private images with high recovery rates and quantitative scores. We further perform downstream tasks such as disease classification with the recovered data, where our results show no significant performance degradation compared to using the original training samples.

摘要: 联合学习(FL)允许一组客户在不暴露本地训练样本的情况下协作训练机器学习模型。在这种情况下，它被认为是隐私保护的，因此被医学中心采用来训练机器学习模型，而不是私人数据。然而，在本文中，我们提出了一种名为MediLeak的新型攻击，该攻击使恶意参数服务器能够从客户端上传的模型更新中恢复高保真的患者图像。MediLeak要求服务器通过在原始模型架构前面添加一个特制的模块来生成对抗性模型。它在定期的FL训练过程中发布给客户端，每个客户端对其进行本地训练，生成相应的模型更新。然后，基于FL协议，模型更新被发送回服务器，并且我们提出的分析方法从定制模块的参数更新中恢复私有数据。我们对MediLeak进行了全面的分析，表明它可以成功地破解最先进的密码安全聚合协议，这些协议旨在保护FL系统免受隐私推理攻击。我们在MedMNIST和COVIDx CXR-4数据集上实现了MediLeak。结果表明，MediLeak能够近乎完美地恢复私密图像，恢复率和量化分数都很高。我们使用恢复的数据进一步执行下游任务，例如疾病分类，其中我们的结果显示与使用原始训练样本相比，性能没有显著下降。



## **10. Black-Box Detection of Language Model Watermarks**

语言模型水印的黑匣子检测 cs.CR

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2405.20777v2) [paper-pdf](http://arxiv.org/pdf/2405.20777v2)

**Authors**: Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev

**Abstract**: Watermarking has emerged as a promising way to detect LLM-generated text. To apply a watermark an LLM provider, given a secret key, augments generations with a signal that is later detectable by any party with the same key. Recent work has proposed three main families of watermarking schemes, two of which focus on the property of preserving the LLM distribution. This is motivated by it being a tractable proxy for maintaining LLM capabilities, but also by the idea that concealing a watermark deployment makes it harder for malicious actors to hide misuse by avoiding a certain LLM or attacking its watermark. Yet, despite much discourse around detectability, no prior work has investigated if any of these scheme families are detectable in a realistic black-box setting. We tackle this for the first time, developing rigorous statistical tests to detect the presence of all three most popular watermarking scheme families using only a limited number of black-box queries. We experimentally confirm the effectiveness of our methods on a range of schemes and a diverse set of open-source models. Our findings indicate that current watermarking schemes are more detectable than previously believed, and that obscuring the fact that a watermark was deployed may not be a viable way for providers to protect against adversaries. We further apply our methods to test for watermark presence behind the most popular public APIs: GPT4, Claude 3, Gemini 1.0 Pro, finding no strong evidence of a watermark at this point in time.

摘要: 水印技术已经成为检测LLM生成文本的一种很有前途的方法。为了应用水印，LLM提供商在给定秘密密钥的情况下，使用稍后可被具有相同密钥的任何一方检测的信号来增加生成。最近的工作已经提出了三类主要的数字水印方案，其中两类侧重于保持LLM分布的性质。这是因为它是维护LLM功能的易于处理的代理，但也是因为隐藏水印部署会使恶意攻击者更难通过避免特定LLM或攻击其水印来隐藏误用。然而，尽管有很多关于可检测性的讨论，但之前的工作还没有调查过这些方案家族中是否有任何一个在现实的黑盒环境中是可检测的。我们首次解决了这一问题，开发了严格的统计测试，仅使用有限数量的黑盒查询来检测所有三个最受欢迎的水印方案家族的存在。我们在一系列方案和一组不同的开源模型上实验证实了我们的方法的有效性。我们的发现表明，目前的水印方案比之前认为的更容易检测到，并且掩盖水印被部署的事实可能不是提供商保护免受对手攻击的可行方法。我们进一步应用我们的方法来测试最流行的公共API：GPT4、Claude 3、Gemini 1.0 Pro背后的水印存在，目前没有找到水印的有力证据。



## **11. SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization**

SpecFormer：通过最大奇异值惩罚保护Vision Transformer的稳健性 cs.CV

Accepted by ECCV 2024; 27 pages; code is at:  https://github.com/microsoft/robustlearn

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2402.03317v2) [paper-pdf](http://arxiv.org/pdf/2402.03317v2)

**Authors**: Xixu Hu, Runkai Zheng, Jindong Wang, Cheuk Hang Leung, Qi Wu, Xing Xie

**Abstract**: Vision Transformers (ViTs) are increasingly used in computer vision due to their high performance, but their vulnerability to adversarial attacks is a concern. Existing methods lack a solid theoretical basis, focusing mainly on empirical training adjustments. This study introduces SpecFormer, tailored to fortify ViTs against adversarial attacks, with theoretical underpinnings. We establish local Lipschitz bounds for the self-attention layer and propose the Maximum Singular Value Penalization (MSVP) to precisely manage these bounds By incorporating MSVP into ViTs' attention layers, we enhance the model's robustness without compromising training efficiency. SpecFormer, the resulting model, outperforms other state-of-the-art models in defending against adversarial attacks, as proven by experiments on CIFAR and ImageNet datasets. Code is released at https://github.com/microsoft/robustlearn.

摘要: Vision Transformers（ViT）因其高性能而越来越多地应用于计算机视觉，但其对对抗攻击的脆弱性令人担忧。现有方法缺乏坚实的理论基础，主要侧重于经验性的训练调整。这项研究引入了SpecFormer，旨在增强ViT抵御对抗攻击，并提供了理论基础。我们为自我注意力层建立了局部Lipschitz界限，并提出最大奇异值惩罚（MSVP）来精确管理这些界限。通过将MSVP整合到ViT的注意力层中，我们增强了模型的鲁棒性，而不会影响训练效率。CIFAR和ImageNet数据集的实验证明，由此产生的模型SpecFormer在防御对抗攻击方面优于其他最先进的模型。代码发布于https://github.com/microsoft/robustlearn。



## **12. Enhancing Tracking Robustness with Auxiliary Adversarial Defense Networks**

利用辅助对抗防御网络增强跟踪稳健性 cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2402.17976v2) [paper-pdf](http://arxiv.org/pdf/2402.17976v2)

**Authors**: Zhewei Wu, Ruilong Yu, Qihe Liu, Shuying Cheng, Shilin Qiu, Shijie Zhou

**Abstract**: Adversarial attacks in visual object tracking have significantly degraded the performance of advanced trackers by introducing imperceptible perturbations into images. However, there is still a lack of research on designing adversarial defense methods for object tracking. To address these issues, we propose an effective auxiliary pre-processing defense network, AADN, which performs defensive transformations on the input images before feeding them into the tracker. Moreover, it can be seamlessly integrated with other visual trackers as a plug-and-play module without parameter adjustments. We train AADN using adversarial training, specifically employing Dua-Loss to generate adversarial samples that simultaneously attack the classification and regression branches of the tracker. Extensive experiments conducted on the OTB100, LaSOT, and VOT2018 benchmarks demonstrate that AADN maintains excellent defense robustness against adversarial attack methods in both adaptive and non-adaptive attack scenarios. Moreover, when transferring the defense network to heterogeneous trackers, it exhibits reliable transferability. Finally, AADN achieves a processing time of up to 5ms/frame, allowing seamless integration with existing high-speed trackers without introducing significant computational overhead.

摘要: 视觉目标跟踪中的对抗性攻击通过在图像中引入不可感知的扰动而显著降低了高级跟踪器的性能。然而，目前还缺乏针对目标跟踪设计对抗性防御方法的研究。为了解决这些问题，我们提出了一种有效的辅助预处理防御网络AADN，它在将输入图像送入跟踪器之前对其进行防御性转换。此外，它还可以作为即插即用模块与其他视觉追踪器无缝集成，无需调整参数。我们使用对抗性训练来训练AADN，特别是使用Dua-Loss来生成同时攻击跟踪器的分类和回归分支的对抗性样本。在OTB100、LaSOT和VOT2018基准上进行的大量实验表明，AADN在自适应和非自适应攻击场景中都对对抗性攻击方法保持了良好的防御鲁棒性。此外，当将防御网络转移到不同的跟踪器时，它表现出可靠的转移能力。最后，AADN实现了高达5ms/帧的处理时间，允许与现有的高速跟踪器无缝集成，而不会带来显著的计算开销。



## **13. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

利用安全性和活力来增强性能的两层区块链碎片协议 cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2310.11373v5) [paper-pdf](http://arxiv.org/pdf/2310.11373v5)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.

摘要: 分片对于提高区块链可伸缩性至关重要。现有的协议忽略了不同的对抗性攻击，限制了交易吞吐量。本文提出了一种突破性的分片协议Reetum，解决了这个问题，提高了区块链的可扩展性。RENETUM采用两阶段方法，根据运行时敌意攻击调整事务吞吐量。它包括两层的“控制”和“流程”分片。进程碎片包含至少一个可信节点，而控制碎片包含大多数可信节点。在第一阶段，事务被写入块，并由流程碎片中的节点投票表决。一致接受的障碍得到确认。在第二阶段，未获得一致接受的块由控制碎片投票表决。如果多数人投赞成票，就会接受阻止，从而消除第一阶段的反对者和沉默的选民。第一阶段使用一致投票，涉及的节点更少，支持更多的并行进程碎片。控制碎片最终确定决策并解决纠纷。实验证实了ReNetum的创新设计，提供了高交易吞吐量和对各种网络攻击的稳健性，性能优于现有的区块链网络分片协议。



## **14. Improving Alignment and Robustness with Circuit Breakers**

改善断路器的对准和稳健性 cs.LG

Code and models are available at  https://github.com/GraySwanAI/circuit-breakers

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.04313v4) [paper-pdf](http://arxiv.org/pdf/2406.04313v4)

**Authors**: Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, Rowan Wang, Zico Kolter, Matt Fredrikson, Dan Hendrycks

**Abstract**: AI systems can take harmful actions and are highly vulnerable to adversarial attacks. We present an approach, inspired by recent advances in representation engineering, that interrupts the models as they respond with harmful outputs with "circuit breakers." Existing techniques aimed at improving alignment, such as refusal training, are often bypassed. Techniques such as adversarial training try to plug these holes by countering specific attacks. As an alternative to refusal training and adversarial training, circuit-breaking directly controls the representations that are responsible for harmful outputs in the first place. Our technique can be applied to both text-only and multimodal language models to prevent the generation of harmful outputs without sacrificing utility -- even in the presence of powerful unseen attacks. Notably, while adversarial robustness in standalone image recognition remains an open challenge, circuit breakers allow the larger multimodal system to reliably withstand image "hijacks" that aim to produce harmful content. Finally, we extend our approach to AI agents, demonstrating considerable reductions in the rate of harmful actions when they are under attack. Our approach represents a significant step forward in the development of reliable safeguards to harmful behavior and adversarial attacks.

摘要: 人工智能系统可能采取有害行动，并且非常容易受到对抗性攻击。我们提出了一种方法，灵感来自于最近在表示工程方面的进展，该方法中断了模型，因为它们用“断路器”来响应有害的输出。旨在改善一致性的现有技术，如拒绝训练，经常被绕过。对抗性训练等技术试图通过反击特定攻击来堵塞这些漏洞。作为拒绝训练和对抗性训练的另一种选择，断路直接控制首先要对有害输出负责的陈述。我们的技术可以应用于纯文本和多模式语言模型，在不牺牲效用的情况下防止产生有害输出-即使在存在强大的看不见的攻击的情况下也是如此。值得注意的是，虽然独立图像识别中的对抗性健壮性仍然是一个开放的挑战，但断路器允许更大的多模式系统可靠地经受住旨在产生有害内容的图像“劫持”。最后，我们将我们的方法扩展到人工智能代理，表明当他们受到攻击时，有害行动的比率大大降低。我们的方法代表着在发展对有害行为和敌对攻击的可靠保障方面向前迈出了重要的一步。



## **15. Deep Adversarial Defense Against Multilevel-Lp Attacks**

针对多层LP攻击的深度对抗防御 cs.LG

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09251v1) [paper-pdf](http://arxiv.org/pdf/2407.09251v1)

**Authors**: Ren Wang, Yuxuan Li, Alfred Hero

**Abstract**: Deep learning models have shown considerable vulnerability to adversarial attacks, particularly as attacker strategies become more sophisticated. While traditional adversarial training (AT) techniques offer some resilience, they often focus on defending against a single type of attack, e.g., the $\ell_\infty$-norm attack, which can fail for other types. This paper introduces a computationally efficient multilevel $\ell_p$ defense, called the Efficient Robust Mode Connectivity (EMRC) method, which aims to enhance a deep learning model's resilience against multiple $\ell_p$-norm attacks. Similar to analytical continuation approaches used in continuous optimization, the method blends two $p$-specific adversarially optimal models, the $\ell_1$- and $\ell_\infty$-norm AT solutions, to provide good adversarial robustness for a range of $p$. We present experiments demonstrating that our approach performs better on various attacks as compared to AT-$\ell_\infty$, E-AT, and MSD, for datasets/architectures including: CIFAR-10, CIFAR-100 / PreResNet110, WideResNet, ViT-Base.

摘要: 深度学习模型在敌意攻击中表现出相当大的脆弱性，特别是在攻击者策略变得更加复杂的情况下。虽然传统的对手训练(AT)技术提供了一定的弹性，但它们通常专注于防御单一类型的攻击，例如$\ell_\Inty$-Norm攻击，而这种攻击可能会在其他类型的攻击中失败。本文介绍了一种计算高效的多级$ell_p$防御方法，称为高效稳健模式连通性(EMRC)方法，旨在增强深度学习模型对多重$ell_p-范数攻击的弹性。类似于连续优化中使用的分析延拓方法，该方法融合了两个$p$特定的对抗性最优模型，即$\ell_1$-和$\ell_\inty$-范数AT解，以在$p$范围内提供良好的对抗性稳健性。我们的实验表明，对于包括CIFAR-10、CIFAR-100/PreResNet110、WideResNet、Vit-Base在内的数据集/体系结构，与AT-$\ELL_\INFTY$、E-AT和MSD相比，我们的方法对各种攻击具有更好的性能。



## **16. Robust Yet Efficient Conformal Prediction Sets**

稳健而高效的保形预测集 cs.LG

Proceedings of the 41st International Conference on Machine Learning

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09165v1) [paper-pdf](http://arxiv.org/pdf/2407.09165v1)

**Authors**: Soroush H. Zargarbashi, Mohammad Sadegh Akhondzadeh, Aleksandar Bojchevski

**Abstract**: Conformal prediction (CP) can convert any model's output into prediction sets guaranteed to include the true label with any user-specified probability. However, same as the model itself, CP is vulnerable to adversarial test examples (evasion) and perturbed calibration data (poisoning). We derive provably robust sets by bounding the worst-case change in conformity scores. Our tighter bounds lead to more efficient sets. We cover both continuous and discrete (sparse) data and our guarantees work both for evasion and poisoning attacks (on both features and labels).

摘要: 保形预测（CP）可以将任何模型的输出转换为预测集，确保以任何用户指定的概率包含真实标签。然而，与模型本身一样，CP也容易受到对抗性测试示例（规避）和扰动校准数据（中毒）的影响。我们通过限制一致性分数的最坏情况变化来获得可证明稳健的集。我们更严格的界限会带来更高效的集合。我们涵盖连续和离散（稀疏）数据，我们的保证适用于规避和中毒攻击（功能和标签）。



## **17. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很强的威胁性(攻击成功率提高高达89.3%)和隐蔽性(在触发设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **18. Evaluating the Adversarial Robustness of Semantic Segmentation: Trying Harder Pays Off**

评估语义分割的对抗鲁棒性：努力才有回报 cs.CV

Accepted for ECCV 2024. For the implementation, see  https://github.com/szegedai/Robust-Segmentation-Evaluation

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.09150v1) [paper-pdf](http://arxiv.org/pdf/2407.09150v1)

**Authors**: Levente Halmosi, Bálint Mohos, Márk Jelasity

**Abstract**: Machine learning models are vulnerable to tiny adversarial input perturbations optimized to cause a very large output error. To measure this vulnerability, we need reliable methods that can find such adversarial perturbations. For image classification models, evaluation methodologies have emerged that have stood the test of time. However, we argue that in the area of semantic segmentation, a good approximation of the sensitivity to adversarial perturbations requires significantly more effort than what is currently considered satisfactory. To support this claim, we re-evaluate a number of well-known robust segmentation models in an extensive empirical study. We propose new attacks and combine them with the strongest attacks available in the literature. We also analyze the sensitivity of the models in fine detail. The results indicate that most of the state-of-the-art models have a dramatically larger sensitivity to adversarial perturbations than previously reported. We also demonstrate a size-bias: small objects are often more easily attacked, even if the large objects are robust, a phenomenon not revealed by current evaluation metrics. Our results also demonstrate that a diverse set of strong attacks is necessary, because different models are often vulnerable to different attacks.

摘要: 机器学习模型容易受到微小的对抗性输入扰动的影响，优化后的输入会导致非常大的输出误差。为了衡量这种脆弱性，我们需要可靠的方法来发现这种对抗性的扰动。对于图像分类模型，已经出现了经得起时间考验的评估方法。然而，我们认为，在语义分割领域，对对抗性扰动的敏感度的良好近似需要比目前被认为令人满意的工作要多得多。为了支持这一主张，我们在一项广泛的实证研究中重新评估了一些著名的稳健分割模型。我们提出了新的攻击，并将它们与文献中可用的最强攻击相结合。我们还对模型的灵敏度进行了详细的分析。结果表明，大多数最先进的模型对对抗性扰动的敏感度比以前报道的要大得多。我们还展示了大小偏差：小对象通常更容易受到攻击，即使大对象是健壮的，这一现象在当前的评估指标中没有揭示出来。我们的结果还表明，一组不同的强攻击是必要的，因为不同的模型往往容易受到不同的攻击。



## **19. Jailbreaking as a Reward Misspecification Problem**

越狱是奖励错误指定问题 cs.LG

github url added

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2406.14393v2) [paper-pdf](http://arxiv.org/pdf/2406.14393v2)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts against various target aligned LLMs. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark while preserving the human readability of the generated prompts. Detailed analysis highlights the unique advantages brought by the proposed reward misspecification objective compared to previous methods.

摘要: 大型语言模型（LLM）的广泛采用引发了人们对其安全性和可靠性的担忧，特别是对其容易受到对抗攻击的影响。在本文中，我们提出了一种新颖的视角，将此漏洞归因于对齐过程中的奖励错误指定。我们引入了一个指标ReGap来量化奖励错误指定的程度，并展示其在检测有害后门提示方面的有效性和稳健性。在这些见解的基础上，我们介绍了ReMiss，这是一个用于自动化红色分组的系统，可以针对各种目标对齐的LLM生成对抗提示。ReMiss在AdvBench基准上实现了最先进的攻击成功率，同时保留了生成提示的人类可读性。与以前的方法相比，详细的分析强调了拟议的奖励错误指定目标所带来的独特优势。



## **20. A Survey of Attacks on Large Vision-Language Models: Resources, Advances, and Future Trends**

大型视觉语言模型攻击调查：资源、进展和未来趋势 cs.CV

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.07403v2) [paper-pdf](http://arxiv.org/pdf/2407.07403v2)

**Authors**: Daizong Liu, Mingyu Yang, Xiaoye Qu, Pan Zhou, Yu Cheng, Wei Hu

**Abstract**: With the significant development of large models in recent years, Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a wide range of multimodal understanding and reasoning tasks. Compared to traditional Large Language Models (LLMs), LVLMs present great potential and challenges due to its closer proximity to the multi-resource real-world applications and the complexity of multi-modal processing. However, the vulnerability of LVLMs is relatively underexplored, posing potential security risks in daily usage. In this paper, we provide a comprehensive review of the various forms of existing LVLM attacks. Specifically, we first introduce the background of attacks targeting LVLMs, including the attack preliminary, attack challenges, and attack resources. Then, we systematically review the development of LVLM attack methods, such as adversarial attacks that manipulate model outputs, jailbreak attacks that exploit model vulnerabilities for unauthorized actions, prompt injection attacks that engineer the prompt type and pattern, and data poisoning that affects model training. Finally, we discuss promising research directions in the future. We believe that our survey provides insights into the current landscape of LVLM vulnerabilities, inspiring more researchers to explore and mitigate potential safety issues in LVLM developments. The latest papers on LVLM attacks are continuously collected in https://github.com/liudaizong/Awesome-LVLM-Attack.

摘要: 近年来，随着大型模型的显著发展，大型视觉语言模型在广泛的多通道理解和推理任务中表现出了卓越的能力。与传统的大语言模型相比，大语言模型因其更接近多资源的实际应用和多模式处理的复杂性而显示出巨大的潜力和挑战。然而，LVLMS的脆弱性相对较少，在日常使用中存在潜在的安全风险。在本文中，我们对现有的各种形式的LVLM攻击进行了全面的回顾。具体地说，我们首先介绍了针对LVLMS的攻击背景，包括攻击准备、攻击挑战和攻击资源。然后，我们系统地回顾了LVLM攻击方法的发展，如操纵模型输出的对抗性攻击，利用模型漏洞进行未经授权操作的越狱攻击，设计提示类型和模式的提示注入攻击，以及影响模型训练的数据中毒。最后，我们讨论了未来的研究方向。我们相信，我们的调查提供了对LVLM漏洞现状的洞察，激励更多的研究人员探索和缓解LVLM开发中的潜在安全问题。有关LVLm攻击的最新论文在https://github.com/liudaizong/Awesome-LVLM-Attack.上不断收集



## **21. Soft Prompts Go Hard: Steering Visual Language Models with Hidden Meta-Instructions**

软指令走得更紧：用隐藏的元指令引导视觉语言模型 cs.CR

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2407.08970v1) [paper-pdf](http://arxiv.org/pdf/2407.08970v1)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasaryan, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect injection vulnerabilities in language models that operate on images: hidden "meta-instructions" that influence how the model interprets the image and steer the model's outputs to express an adversary-chosen style, sentiment, or point of view.   We explain how to create meta-instructions by generating images that act as soft prompts. Unlike jailbreaking attacks and adversarial examples, the outputs resulting from these images are plausible and based on the visual content of the image, yet follow the adversary's (meta-)instructions. We describe the risks of these attacks, including misinformation and spin, evaluate their efficacy for multiple visual language models and adversarial meta-objectives, and demonstrate how they can "unlock" the capabilities of the underlying language models that are unavailable via explicit text instructions. Finally, we discuss defenses against these attacks.

摘要: 我们在对图像进行操作的语言模型中引入了一种新型的间接注入漏洞：隐藏的“元指令”，影响模型如何解释图像并引导模型的输出来表达对手选择的风格、情感或观点。   我们解释了如何通过生成充当软提示的图像来创建元指令。与越狱攻击和对抗示例不同，这些图像产生的输出是合理的，并且基于图像的视觉内容，但遵循对手的（Meta）指令。我们描述了这些攻击的风险，包括错误信息和旋转，评估了它们对多个视觉语言模型和对抗性元目标的有效性，并演示了它们如何“解锁”通过显式文本指令不可用的底层语言模型的功能。最后，我们讨论针对这些攻击的防御。



## **22. Rethinking Graph Backdoor Attacks: A Distribution-Preserving Perspective**

重新思考图表后门攻击：保留分布的角度 cs.LG

Accepted by KDD 2024

**SubmitDate**: 2024-07-12    [abs](http://arxiv.org/abs/2405.10757v3) [paper-pdf](http://arxiv.org/pdf/2405.10757v3)

**Authors**: Zhiwei Zhang, Minhua Lin, Enyan Dai, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in various tasks. However, recent works reveal that GNNs are vulnerable to backdoor attacks. Generally, backdoor attack poisons the graph by attaching backdoor triggers and the target class label to a set of nodes in the training graph. A GNN trained on the poisoned graph will then be misled to predict test nodes attached with trigger to the target class. Despite their effectiveness, our empirical analysis shows that triggers generated by existing methods tend to be out-of-distribution (OOD), which significantly differ from the clean data. Hence, these injected triggers can be easily detected and pruned with widely used outlier detection methods in real-world applications. Therefore, in this paper, we study a novel problem of unnoticeable graph backdoor attacks with in-distribution (ID) triggers. To generate ID triggers, we introduce an OOD detector in conjunction with an adversarial learning strategy to generate the attributes of the triggers within distribution. To ensure a high attack success rate with ID triggers, we introduce novel modules designed to enhance trigger memorization by the victim model trained on poisoned graph. Extensive experiments on real-world datasets demonstrate the effectiveness of the proposed method in generating in distribution triggers that can by-pass various defense strategies while maintaining a high attack success rate.

摘要: 图形神经网络(GNN)在各种任务中表现出了显著的性能。然而，最近的研究表明，GNN很容易受到后门攻击。通常，后门攻击通过将后门触发器和目标类标签附加到训练图中的一组节点来毒化图。然后，在有毒图上训练的GNN将被误导，以预测与目标类的触发器附加的测试节点。尽管它们是有效的，但我们的实证分析表明，现有方法生成的触发因素往往是分布外(OOD)，这与干净的数据有很大不同。因此，这些注入的触发器可以很容易地被现实世界应用中广泛使用的离群点检测方法检测和修剪。因此，在本文中，我们研究了一种新的具有分布内(ID)触发器的不可察觉图后门攻击问题。为了生成ID触发器，我们引入了一个OOD检测器，并结合对抗性学习策略来生成分布内触发器的属性。为了确保ID触发器的高攻击成功率，我们引入了新的模块，通过在中毒图上训练受害者模型来增强对触发器的记忆。在真实数据集上的大量实验表明，该方法在生成分布触发器方面是有效的，可以绕过各种防御策略，同时保持较高的攻击成功率。



## **23. HO-FMN: Hyperparameter Optimization for Fast Minimum-Norm Attacks**

HO-FNN：针对快速最小规范攻击的超参数优化 cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08806v1) [paper-pdf](http://arxiv.org/pdf/2407.08806v1)

**Authors**: Raffaele Mura, Giuseppe Floris, Luca Scionis, Giorgio Piras, Maura Pintor, Ambra Demontis, Giorgio Giacinto, Battista Biggio, Fabio Roli

**Abstract**: Gradient-based attacks are a primary tool to evaluate robustness of machine-learning models. However, many attacks tend to provide overly-optimistic evaluations as they use fixed loss functions, optimizers, step-size schedulers, and default hyperparameters. In this work, we tackle these limitations by proposing a parametric variation of the well-known fast minimum-norm attack algorithm, whose loss, optimizer, step-size scheduler, and hyperparameters can be dynamically adjusted. We re-evaluate 12 robust models, showing that our attack finds smaller adversarial perturbations without requiring any additional tuning. This also enables reporting adversarial robustness as a function of the perturbation budget, providing a more complete evaluation than that offered by fixed-budget attacks, while remaining efficient. We release our open-source code at https://github.com/pralab/HO-FMN.

摘要: 基于对象的攻击是评估机器学习模型稳健性的主要工具。然而，许多攻击往往会提供过于乐观的评估，因为它们使用固定损失函数、优化器、步进大小排序器和默认超参数。在这项工作中，我们通过提出著名的快速最小模攻击算法的参数变体来解决这些限制，该算法的损失、优化器、步进大小调度器和超参数可以动态调整。我们重新评估了12个稳健模型，表明我们的攻击可以发现更小的对抗扰动，而不需要任何额外的调整。这还使得能够将对抗稳健性报告为扰动预算的函数，从而提供比固定预算攻击更完整的评估，同时保持高效。我们在https://github.com/pralab/HO-FMN上发布我们的开源代码。



## **24. How to beat a Bayesian adversary**

如何击败Bayesian对手 cs.LG

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08678v1) [paper-pdf](http://arxiv.org/pdf/2407.08678v1)

**Authors**: Zihan Ding, Kexin Jin, Jonas Latz, Chenguang Liu

**Abstract**: Deep neural networks and other modern machine learning models are often susceptible to adversarial attacks. Indeed, an adversary may often be able to change a model's prediction through a small, directed perturbation of the model's input - an issue in safety-critical applications. Adversarially robust machine learning is usually based on a minmax optimisation problem that minimises the machine learning loss under maximisation-based adversarial attacks.   In this work, we study adversaries that determine their attack using a Bayesian statistical approach rather than maximisation. The resulting Bayesian adversarial robustness problem is a relaxation of the usual minmax problem. To solve this problem, we propose Abram - a continuous-time particle system that shall approximate the gradient flow corresponding to the underlying learning problem. We show that Abram approximates a McKean-Vlasov process and justify the use of Abram by giving assumptions under which the McKean-Vlasov process finds the minimiser of the Bayesian adversarial robustness problem. We discuss two ways to discretise Abram and show its suitability in benchmark adversarial deep learning experiments.

摘要: 深度神经网络和其他现代机器学习模型往往容易受到敌意攻击。事实上，对手往往能够通过对模型的输入进行小的、直接的扰动来改变模型的预测--这在安全关键的应用程序中是一个问题。对抗性稳健机器学习通常基于最小化最大优化问题，该问题在基于最大化的对抗性攻击下最小化机器学习损失。在这项工作中，我们研究了使用贝叶斯统计方法而不是最大化来确定攻击的对手。由此产生的贝叶斯对抗健壮性问题是通常的极大极小问题的松弛。为了解决这个问题，我们提出了Abram-一个连续时间粒子系统，它应该近似于对应于底层学习问题的梯度流。我们证明了Abram逼近McKean-Vlasov过程，并通过给出McKean-Vlasov过程找到贝叶斯对抗健壮性问题的最小值的假设来证明Abram的使用。我们讨论了两种离散化Abram的方法，并在基准对抗性深度学习实验中展示了它的适用性。



## **25. Large-Scale Dataset Pruning in Adversarial Training through Data Importance Extrapolation**

通过数据重要性外推进行对抗训练中的大规模数据集修剪 cs.LG

8 pages, 5 figures, 3 tables, to be published in ICML: DMLR workshop

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2406.13283v2) [paper-pdf](http://arxiv.org/pdf/2406.13283v2)

**Authors**: Björn Nieth, Thomas Altstidl, Leo Schwinn, Björn Eskofier

**Abstract**: Their vulnerability to small, imperceptible attacks limits the adoption of deep learning models to real-world systems. Adversarial training has proven to be one of the most promising strategies against these attacks, at the expense of a substantial increase in training time. With the ongoing trend of integrating large-scale synthetic data this is only expected to increase even further. Thus, the need for data-centric approaches that reduce the number of training samples while maintaining accuracy and robustness arises. While data pruning and active learning are prominent research topics in deep learning, they are as of now largely unexplored in the adversarial training literature. We address this gap and propose a new data pruning strategy based on extrapolating data importance scores from a small set of data to a larger set. In an empirical evaluation, we demonstrate that extrapolation-based pruning can efficiently reduce dataset size while maintaining robustness.

摘要: 它们对小型、不可感知的攻击的脆弱性限制了深度学习模型在现实世界系统中的采用。事实证明，对抗训练是对抗这些攻击的最有希望的策略之一，但代价是训练时间的大幅增加。随着集成大规模合成数据的持续趋势，预计这一数字只会进一步增加。因此，需要以数据为中心的方法来减少训练样本数量，同时保持准确性和稳健性。虽然数据修剪和主动学习是深度学习中的重要研究主题，但迄今为止，对抗性训练文献中基本上尚未对其进行探讨。我们解决了这一差距，并提出了一种新的数据修剪策略，该策略基于将数据重要性分数从小数据集外推到大数据集。在经验评估中，我们证明基于外推的修剪可以有效地减少数据集大小，同时保持稳健性。



## **26. DART: A Solution for Decentralized Federated Learning Model Robustness Analysis**

DART：分散联邦学习模型鲁棒性分析的解决方案 cs.DC

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08652v1) [paper-pdf](http://arxiv.org/pdf/2407.08652v1)

**Authors**: Chao Feng, Alberto Huertas Celdrán, Jan von der Assen, Enrique Tomás Martínez Beltrán, Gérôme Bovet, Burkhard Stiller

**Abstract**: Federated Learning (FL) has emerged as a promising approach to address privacy concerns inherent in Machine Learning (ML) practices. However, conventional FL methods, particularly those following the Centralized FL (CFL) paradigm, utilize a central server for global aggregation, which exhibits limitations such as bottleneck and single point of failure. To address these issues, the Decentralized FL (DFL) paradigm has been proposed, which removes the client-server boundary and enables all participants to engage in model training and aggregation tasks. Nevertheless, as CFL, DFL remains vulnerable to adversarial attacks, notably poisoning attacks that undermine model performance. While existing research on model robustness has predominantly focused on CFL, there is a noteworthy gap in understanding the model robustness of the DFL paradigm. In this paper, a thorough review of poisoning attacks targeting the model robustness in DFL systems, as well as their corresponding countermeasures, are presented. Additionally, a solution called DART is proposed to evaluate the robustness of DFL models, which is implemented and integrated into a DFL platform. Through extensive experiments, this paper compares the behavior of CFL and DFL under diverse poisoning attacks, pinpointing key factors affecting attack spread and effectiveness within the DFL. It also evaluates the performance of different defense mechanisms and investigates whether defense mechanisms designed for CFL are compatible with DFL. The empirical results provide insights into research challenges and suggest ways to improve the robustness of DFL models for future research.

摘要: 联合学习(FL)已经成为解决机器学习(ML)实践中固有的隐私问题的一种有前途的方法。然而，传统的FL方法，特别是那些遵循集中式FL(CFL)范例的方法，使用中央服务器进行全局聚合，这表现出诸如瓶颈和单点故障等限制。为了解决这些问题，提出了分散式FL(DFL)范例，它消除了客户端-服务器的边界，使所有参与者都能够参与模型训练和聚合任务。然而，作为CFL，DFL仍然容易受到对手的攻击，特别是破坏模型性能的中毒攻击。虽然现有的关于模型稳健性的研究主要集中在CFL上，但在理解DFL范式的模型稳健性方面存在着明显的差距。本文对DFL系统中以模型稳健性为目标的中毒攻击及其相应的对策进行了深入的综述。此外，还提出了一种称为DART的解决方案来评估DFL模型的健壮性，并将其实现并集成到DFL平台中。通过大量的实验，比较了CFL和DFL在不同的中毒攻击下的行为，找出了影响攻击在DFL内传播和有效性的关键因素。评估了不同防御机制的性能，并研究了为CFL设计的防御机制是否与DFL兼容。实证结果提供了对研究挑战的洞察，并为未来的研究提出了改进DFL模型的稳健性的方法。



## **27. RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Adversarial Data Manipulation**

RAIFLE：对具有对抗性数据操纵的基于交互的联邦学习的重建攻击 cs.CR

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2310.19163v2) [paper-pdf](http://arxiv.org/pdf/2310.19163v2)

**Authors**: Dzung Pham, Shreyas Kulkarni, Amir Houmansadr

**Abstract**: Federated learning has emerged as a promising privacy-preserving solution for machine learning domains that rely on user interactions, particularly recommender systems and online learning to rank. While there has been substantial research on the privacy of traditional federated learning, little attention has been paid to the privacy properties of these interaction-based settings. In this work, we show that users face an elevated risk of having their private interactions reconstructed by the central server when the server can control the training features of the items that users interact with. We introduce RAIFLE, a novel optimization-based attack framework where the server actively manipulates the features of the items presented to users to increase the success rate of reconstruction. Our experiments with federated recommendation and online learning-to-rank scenarios demonstrate that RAIFLE is significantly more powerful than existing reconstruction attacks like gradient inversion, achieving high performance consistently in most settings. We discuss the pros and cons of several possible countermeasures to defend against RAIFLE in the context of interaction-based federated learning. Our code is open-sourced at https://github.com/dzungvpham/raifle.

摘要: 联合学习已经成为一种很有前途的隐私保护解决方案，适用于依赖用户交互的机器学习领域，特别是推荐系统和在线学习来排名。虽然已经有大量关于传统联合学习隐私的研究，但很少有人关注这些基于交互的设置的隐私属性。在这项工作中，我们表明，当中央服务器可以控制用户交互的项目的训练特征时，用户面临由中央服务器重建他们的私人交互的风险增加。我们引入了RAIFLE，一个新的基于优化的攻击框架，服务器主动操纵呈现给用户的项目的特征，以提高重建的成功率。我们在联邦推荐和在线学习排名场景下的实验表明，RAIFLE比现有的重建攻击(如梯度反转)要强大得多，在大多数情况下都能获得一致的高性能。我们讨论了在基于交互的联邦学习背景下防御RAIFLE的几种可能对策的利弊。我们的代码在https://github.com/dzungvpham/raifle.上是开源的



## **28. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：基于脑电波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2402.08656v5) [paper-pdf](http://arxiv.org/pdf/2402.08656v5)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **29. Boosting Adversarial Transferability for Skeleton-based Action Recognition via Exploring the Model Posterior Space**

通过探索模型后验空间增强基于线粒体的动作识别的对抗可移植性 cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08572v1) [paper-pdf](http://arxiv.org/pdf/2407.08572v1)

**Authors**: Yunfeng Diao, Baiqi Wu, Ruixuan Zhang, Xun Yang, Meng Wang, He Wang

**Abstract**: Skeletal motion plays a pivotal role in human activity recognition (HAR). Recently, attack methods have been proposed to identify the universal vulnerability of skeleton-based HAR(S-HAR). However, the research of adversarial transferability on S-HAR is largely missing. More importantly, existing attacks all struggle in transfer across unknown S-HAR models. We observed that the key reason is that the loss landscape of the action recognizers is rugged and sharp. Given the established correlation in prior studies~\cite{qin2022boosting,wu2020towards} between loss landscape and adversarial transferability, we assume and empirically validate that smoothing the loss landscape could potentially improve adversarial transferability on S-HAR. This is achieved by proposing a new post-train Dual Bayesian strategy, which can effectively explore the model posterior space for a collection of surrogates without the need for re-training. Furthermore, to craft adversarial examples along the motion manifold, we incorporate the attack gradient with information of the motion dynamics in a Bayesian manner. Evaluated on benchmark datasets, e.g. HDM05 and NTU 60, the average transfer success rate can reach as high as 35.9\% and 45.5\% respectively. In comparison, current state-of-the-art skeletal attacks achieve only 3.6\% and 9.8\%. The high adversarial transferability remains consistent across various surrogate, victim, and even defense models. Through a comprehensive analysis of the results, we provide insights on what surrogates are more likely to exhibit transferability, to shed light on future research.

摘要: 骨骼运动在人类活动识别(HAR)中起着至关重要的作用。近年来，为了识别基于骨架的HAR(S-HAR)的普遍脆弱性，人们提出了攻击方法。然而，关于S-哈尔对抗性转会的研究还很少。更重要的是，现有的进攻都在挣扎着跨越未知的S-哈尔模型进行转移。我们观察到，关键原因是动作识别器的损失情况是崎岖和尖锐的。鉴于先前的研究已经建立了损失情景与对手可转移性之间的相关性，我们假设并实证平滑损失情景可以潜在地提高S-HAR上的对手可转移性。这是通过提出一种新的训练后双重贝叶斯策略来实现的，该策略可以有效地探索代理集合的模型后验空间，而不需要重新训练。此外，为了制作沿运动流形的对抗性示例，我们以贝叶斯方式将攻击梯度与运动动力学信息相结合。在HDM05和NTU 60等基准数据集上进行评估，平均传输成功率分别高达35.9%和45.5%。相比之下，目前最先进的骨架攻击只实现了3.6%和9.8%。高度的对抗性可转移性在各种代理、受害者甚至防御模型中保持一致。通过对结果的综合分析，我们对哪些替代品更有可能表现出可转移性提供了见解，为未来的研究提供了启示。



## **30. BriDe Arbitrager: Enhancing Arbitrage in Ethereum 2.0 via Bribery-enabled Delayed Block Production**

BriDe Arbitrager：通过贿赂支持的延迟区块生产增强以太坊2.0中的套利 cs.NI

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08537v1) [paper-pdf](http://arxiv.org/pdf/2407.08537v1)

**Authors**: Hulin Yang, Mingzhe Li, Jin Zhang, Alia Asheralieva, Qingsong Wei, Siow Mong Rick Goh

**Abstract**: The advent of Ethereum 2.0 has introduced significant changes, particularly the shift to Proof-of-Stake consensus. This change presents new opportunities and challenges for arbitrage. Amidst these changes, we introduce BriDe Arbitrager, a novel tool designed for Ethereum 2.0 that leverages Bribery-driven attacks to Delay block production and increase arbitrage gains. The main idea is to allow malicious proposers to delay block production by bribing validators/proposers, thereby gaining more time to identify arbitrage opportunities. Through analysing the bribery process, we design an adaptive bribery strategy. Additionally, we propose a Delayed Transaction Ordering Algorithm to leverage the delayed time to amplify arbitrage profits for malicious proposers. To ensure fairness and automate the bribery process, we design and implement a bribery smart contract and a bribery client. As a result, BriDe Arbitrager enables adversaries controlling a limited (< 1/4) fraction of the voting powers to delay block production via bribery and arbitrage more profit. Extensive experimental results based on Ethereum historical transactions demonstrate that BriDe Arbitrager yields an average of 8.66 ETH (16,442.23 USD) daily profits. Furthermore, our approach does not trigger any slashing mechanisms and remains effective even under Proposer Builder Separation and other potential mechanisms will be adopted by Ethereum.

摘要: Etherum 2.0的问世带来了重大变化，特别是转向利害关系证明共识。这种变化给套利带来了新的机遇和挑战。在这些变化中，我们引入了新娘套利，这是为以太2.0设计的一个新工具，它利用贿赂驱动的攻击来延迟大宗生产并增加套利收益。其主要思想是允许恶意提议者通过贿赂验证者/提议者来延迟区块生产，从而获得更多时间来识别套利机会。通过对贿赂过程的分析，设计了一种自适应的贿赂策略。此外，我们还提出了延迟交易排序算法，以利用延迟时间为恶意提出者放大套利利润。为了确保贿赂过程的公平性和自动化，我们设计并实现了一个贿赂智能合同和一个贿赂客户端。因此，新娘套利者使控制有限(<1/4)投票权的对手能够通过贿赂推迟阻止生产，并套利更多利润。基于以太历史交易的广泛实验结果表明，新娘套利者平均每天获得8.66 ETH(16,442.23美元)的利润。此外，我们的方法不会触发任何削减机制，即使在Proposer Builder Separation和Etherum将采用其他潜在机制的情况下也仍然有效。



## **31. Rethinking the Threat and Accessibility of Adversarial Attacks against Face Recognition Systems**

重新思考针对人脸识别系统的对抗性攻击的威胁和可及性 cs.CV

19 pages, 12 figures

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.08514v1) [paper-pdf](http://arxiv.org/pdf/2407.08514v1)

**Authors**: Yuxin Cao, Yumeng Zhu, Derui Wang, Sheng Wen, Minhui Xue, Jin Lu, Hao Ge

**Abstract**: Face recognition pipelines have been widely deployed in various mission-critical systems in trust, equitable and responsible AI applications. However, the emergence of adversarial attacks has threatened the security of the entire recognition pipeline. Despite the sheer number of attack methods proposed for crafting adversarial examples in both digital and physical forms, it is never an easy task to assess the real threat level of different attacks and obtain useful insight into the key risks confronted by face recognition systems. Traditional attacks view imperceptibility as the most important measurement to keep perturbations stealthy, while we suspect that industry professionals may possess a different opinion. In this paper, we delve into measuring the threat brought about by adversarial attacks from the perspectives of the industry and the applications of face recognition. In contrast to widely studied sophisticated attacks in the field, we propose an effective yet easy-to-launch physical adversarial attack, named AdvColor, against black-box face recognition pipelines in the physical world. AdvColor fools models in the recognition pipeline via directly supplying printed photos of human faces to the system under adversarial illuminations. Experimental results show that physical AdvColor examples can achieve a fooling rate of more than 96% against the anti-spoofing model and an overall attack success rate of 88% against the face recognition pipeline. We also conduct a survey on the threats of prevailing adversarial attacks, including AdvColor, to understand the gap between the machine-measured and human-assessed threat levels of different forms of adversarial attacks. The survey results surprisingly indicate that, compared to deliberately launched imperceptible attacks, perceptible but accessible attacks pose more lethal threats to real-world commercial systems of face recognition.

摘要: 人脸识别管道已广泛部署在各种任务关键系统中，用于信任、公平和负责任的人工智能应用。然而，对抗性攻击的出现威胁到了整个识别管道的安全。尽管人们提出了大量的攻击方法来制作数字和物理形式的敌意例子，但评估不同攻击的真实威胁级别并对人脸识别系统面临的关键风险进行有用的洞察从来都不是一项容易的任务。传统攻击将不可感知性视为保持扰动隐蔽性的最重要衡量标准，而我们怀疑行业专业人士可能持有不同的观点。在本文中，我们从行业和人脸识别应用的角度，深入研究了对抗性攻击带来的威胁的度量。与该领域广泛研究的复杂攻击不同，我们提出了一种针对物理世界中的黑盒人脸识别管道的有效且易于启动的物理对手攻击，称为AdvColor。AdvColor通过在对抗性照明下直接向系统提供打印的人脸照片来愚弄识别管道中的模型。实验结果表明，物理AdvColor实例对反欺骗模型的欺骗率达到96%以上，对人脸识别管道的整体攻击成功率达到88%。我们还对包括AdvColor在内的主流对抗性攻击的威胁进行了调查，以了解不同形式的对抗性攻击的机器测量和人类评估的威胁水平之间的差距。调查结果令人惊讶地表明，与故意发起的潜伏攻击相比，可感知但可访问的攻击对现实世界中的人脸识别商业系统构成了更致命的威胁。



## **32. Resilience of Entropy Model in Distributed Neural Networks**

分布式神经网络中的熵模型的弹性 cs.LG

accepted at ECCV 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2403.00942v2) [paper-pdf](http://arxiv.org/pdf/2403.00942v2)

**Authors**: Milin Zhang, Mohammad Abdi, Shahriar Rifat, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have emerged as a key technique to reduce communication overhead without sacrificing performance in edge computing systems. Recently, entropy coding has been introduced to further reduce the communication overhead. The key idea is to train the distributed DNN jointly with an entropy model, which is used as side information during inference time to adaptively encode latent representations into bit streams with variable length. To the best of our knowledge, the resilience of entropy models is yet to be investigated. As such, in this paper we formulate and investigate the resilience of entropy models to intentional interference (e.g., adversarial attacks) and unintentional interference (e.g., weather changes and motion blur). Through an extensive experimental campaign with 3 different DNN architectures, 2 entropy models and 4 rate-distortion trade-off factors, we demonstrate that the entropy attacks can increase the communication overhead by up to 95%. By separating compression features in frequency and spatial domain, we propose a new defense mechanism that can reduce the transmission overhead of the attacked input by about 9% compared to unperturbed data, with only about 2% accuracy loss. Importantly, the proposed defense mechanism is a standalone approach which can be applied in conjunction with approaches such as adversarial training to further improve robustness. Code will be shared for reproducibility.

摘要: 分布式深度神经网络(DNN)已成为边缘计算系统中在不牺牲性能的前提下减少通信开销的关键技术。最近，引入了熵编码来进一步降低通信开销。该算法的核心思想是将分布的DNN与一个熵模型联合训练，作为推理时的辅助信息，自适应地将潜在的表示编码成可变长度的比特流。就我们所知，熵模型的弹性还有待研究。因此，在本文中，我们建立并研究了熵模型对有意干扰(例如，对抗性攻击)和无意干扰(例如，天气变化和运动模糊)的弹性。通过使用3种不同的DNN结构、2种熵模型和4种率失真权衡因子的广泛实验活动，我们证明了熵攻击可以使通信开销增加高达95%。通过在频域和空间域分离压缩特征，我们提出了一种新的防御机制，与未受干扰的数据相比，该机制可以使被攻击输入的传输开销减少约9%，而精确度损失仅约2%。重要的是，建议的防御机制是一种独立的方法，可以与对抗性训练等方法结合使用，以进一步提高健壮性。代码将被共享，以实现重现性。



## **33. Shedding More Light on Robust Classifiers under the lens of Energy-based Models**

在基于能量的模型的视角下更多地关注稳健分类器 cs.CV

Accepted at European Conference on Computer Vision (ECCV) 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.06315v2) [paper-pdf](http://arxiv.org/pdf/2407.06315v2)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Senad Beadini, Iacopo Masi

**Abstract**: By reinterpreting a robust discriminative classifier as Energy-based Model (EBM), we offer a new take on the dynamics of adversarial training (AT). Our analysis of the energy landscape during AT reveals that untargeted attacks generate adversarial images much more in-distribution (lower energy) than the original data from the point of view of the model. Conversely, we observe the opposite for targeted attacks. On the ground of our thorough analysis, we present new theoretical and practical results that show how interpreting AT energy dynamics unlocks a better understanding: (1) AT dynamic is governed by three phases and robust overfitting occurs in the third phase with a drastic divergence between natural and adversarial energies (2) by rewriting the loss of TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization (TRADES) in terms of energies, we show that TRADES implicitly alleviates overfitting by means of aligning the natural energy with the adversarial one (3) we empirically show that all recent state-of-the-art robust classifiers are smoothing the energy landscape and we reconcile a variety of studies about understanding AT and weighting the loss function under the umbrella of EBMs. Motivated by rigorous evidence, we propose Weighted Energy Adversarial Training (WEAT), a novel sample weighting scheme that yields robust accuracy matching the state-of-the-art on multiple benchmarks such as CIFAR-10 and SVHN and going beyond in CIFAR-100 and Tiny-ImageNet. We further show that robust classifiers vary in the intensity and quality of their generative capabilities, and offer a simple method to push this capability, reaching a remarkable Inception Score (IS) and FID using a robust classifier without training for generative modeling. The code to reproduce our results is available at http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/ .

摘要: 通过将稳健的判别分类器重新解释为基于能量的模型(EBM)，我们提供了一种新的方法来研究对手训练(AT)的动态。我们对AT过程中的能量格局的分析表明，从模型的角度来看，非目标攻击产生的敌意图像比原始数据更不均匀(能量更低)。相反，我们在有针对性的攻击中观察到相反的情况。在我们深入分析的基础上，我们提出了新的理论和实践结果，表明解释AT能量动力学如何揭示更好的理解：(1)AT动态由三个阶段控制，鲁棒过拟合发生在第三阶段，自然能量和对抗能量之间存在巨大差异(2)通过代理损失最小化(交易)在能量方面改写了权衡激发的对抗性防御的损失，我们表明，交易通过将自然能量与对手能量对齐的方式隐含地缓解了过度匹配。(3)我们的经验表明，所有最近最先进的稳健分类器都在平滑能量格局，我们协调了关于理解AT和在EBM保护伞下加权损失函数的各种研究。在严格证据的激励下，我们提出了加权能量对抗训练(Weat)，这是一种新的样本加权方案，其精度与CIFAR-10和SVHN等多个基准测试的最新水平相当，并超过CIFAR-100和Tiny-ImageNet。我们进一步证明了健壮分类器在其生成能力的强度和质量上存在差异，并提供了一种简单的方法来推动这一能力，使用健壮分类器而不需要为生成性建模进行训练就可以达到显著的初始得分(IS)和FID。复制我们结果的代码可以在http://github.com/OmnAI-Lab/Robust-Classifiers-under-the-lens-of-EBM/上找到。



## **34. A Human-in-the-Middle Attack against Object Detection Systems**

针对对象检测系统的中间人攻击 cs.RO

Accepted by IEEE Transactions on Artificial Intelligence, 2024

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2208.07174v4) [paper-pdf](http://arxiv.org/pdf/2208.07174v4)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Object detection systems using deep learning models have become increasingly popular in robotics thanks to the rising power of CPUs and GPUs in embedded systems. However, these models are susceptible to adversarial attacks. While some attacks are limited by strict assumptions on access to the detection system, we propose a novel hardware attack inspired by Man-in-the-Middle attacks in cryptography. This attack generates a Universal Adversarial Perturbations (UAP) and injects the perturbation between the USB camera and the detection system via a hardware attack. Besides, prior research is misled by an evaluation metric that measures the model accuracy rather than the attack performance. In combination with our proposed evaluation metrics, we significantly increased the strength of adversarial perturbations. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving.

摘要: 由于嵌入式系统中中央处理器和图形处理器的性能不断提高，使用深度学习模型的对象检测系统在机器人领域变得越来越受欢迎。然而，这些模型很容易受到对抗攻击。虽然有些攻击受到对检测系统访问权限的严格假设的限制，但我们提出了一种受密码学中中间人攻击启发的新型硬件攻击。该攻击会产生通用对抗性扰动（UAP），并通过硬件攻击在USB摄像头和检测系统之间注入扰动。此外，之前的研究被衡量模型准确性而不是攻击性能的评估指标所误导。结合我们提出的评估指标，我们显着增加了对抗性扰动的强度。这些发现引发了深度学习模型在自动驾驶等安全关键系统中的应用的严重担忧。



## **35. Venomancer: Towards Imperceptible and Target-on-Demand Backdoor Attacks in Federated Learning**

毒液杀手：联邦学习中的不可感知和按需定向后门攻击 cs.CV

**SubmitDate**: 2024-07-11    [abs](http://arxiv.org/abs/2407.03144v2) [paper-pdf](http://arxiv.org/pdf/2407.03144v2)

**Authors**: Son Nguyen, Thinh Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) is a distributed machine learning approach that maintains data privacy by training on decentralized data sources. Similar to centralized machine learning, FL is also susceptible to backdoor attacks, where an attacker can compromise some clients by injecting a backdoor trigger into local models of those clients, leading to the global model's behavior being manipulated as desired by the attacker. Most backdoor attacks in FL assume a predefined target class and require control over a large number of clients or knowledge of benign clients' information. Furthermore, they are not imperceptible and are easily detected by human inspection due to clear artifacts left on the poison data. To overcome these challenges, we propose Venomancer, an effective backdoor attack that is imperceptible and allows target-on-demand. Specifically, imperceptibility is achieved by using a visual loss function to make the poison data visually indistinguishable from the original data. Target-on-demand property allows the attacker to choose arbitrary target classes via conditional adversarial training. Additionally, experiments showed that the method is robust against state-of-the-art defenses such as Norm Clipping, Weak DP, Krum, Multi-Krum, RLR, FedRAD, Deepsight, and RFLBAT. The source code is available at https://github.com/nguyenhongson1902/Venomancer.

摘要: 联合学习(FL)是一种分布式机器学习方法，通过对分散的数据源进行训练来维护数据隐私。与集中式机器学习类似，FL也容易受到后门攻击，攻击者可以通过向某些客户端的本地模型注入后门触发器来危害这些客户端，从而导致全局模型的行为被攻击者想要的操纵。FL中的大多数后门攻击假设一个预定义的目标类，并需要控制大量客户端或了解良性客户端的信息。此外，由于毒物数据上留下了明显的伪影，它们并不是不可察觉的，并且很容易被人类检查发现。为了克服这些挑战，我们提出了毒液杀手，这是一种有效的后门攻击，可以潜移默化，并允许按需锁定目标。具体地说，不可感知性是通过使用视觉损失函数来实现的，以使有毒数据在视觉上与原始数据不可区分。按需目标属性允许攻击者通过有条件的对抗性训练选择任意目标类。此外，实验表明，该方法对Norm裁剪、弱DP、Krum、多Krum、RLR、FedRAD、Deepsight和RFLBAT等最先进的防御方法具有较强的鲁棒性。源代码可在https://github.com/nguyenhongson1902/Venomancer.上找到



## **36. A Comprehensive Survey on the Security of Smart Grid: Challenges, Mitigations, and Future Research Opportunities**

智能电网安全性全面调查：挑战、缓解措施和未来研究机会 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07966v1) [paper-pdf](http://arxiv.org/pdf/2407.07966v1)

**Authors**: Arastoo Zibaeirad, Farnoosh Koleini, Shengping Bi, Tao Hou, Tao Wang

**Abstract**: In this study, we conduct a comprehensive review of smart grid security, exploring system architectures, attack methodologies, defense strategies, and future research opportunities. We provide an in-depth analysis of various attack vectors, focusing on new attack surfaces introduced by advanced components in smart grids. The review particularly includes an extensive analysis of coordinated attacks that incorporate multiple attack strategies and exploit vulnerabilities across various smart grid components to increase their adverse impact, demonstrating the complexity and potential severity of these threats. Following this, we examine innovative detection and mitigation strategies, including game theory, graph theory, blockchain, and machine learning, discussing their advancements in counteracting evolving threats and associated research challenges. In particular, our review covers a thorough examination of widely used machine learning-based mitigation strategies, analyzing their applications and research challenges spanning across supervised, unsupervised, semi-supervised, ensemble, and reinforcement learning. Further, we outline future research directions and explore new techniques and concerns. We first discuss the research opportunities for existing and emerging strategies, and then explore the potential role of new techniques, such as large language models (LLMs), and the emerging threat of adversarial machine learning in the future of smart grid security.

摘要: 在这项研究中，我们对智能电网安全进行了全面的回顾，探索了系统架构、攻击方法、防御策略和未来的研究机会。我们深入分析了各种攻击载体，重点分析了智能电网中先进组件引入的新攻击面。审查特别包括对协调攻击的广泛分析，这些攻击整合了多种攻击策略，并利用各种智能电网组件的漏洞来增加其不利影响，从而展示了这些威胁的复杂性和潜在严重性。随后，我们研究了创新的检测和缓解策略，包括博弈论、图论、区块链和机器学习，讨论了它们在应对不断演变的威胁和相关研究挑战方面的进展。特别是，我们的综述涵盖了广泛使用的基于机器学习的缓解策略的彻底检查，分析了它们在监督、非监督、半监督、集成和强化学习中的应用和研究挑战。此外，我们概述了未来的研究方向，并探索了新的技术和关注的问题。我们首先讨论了现有和新兴策略的研究机会，然后探讨了新技术的潜在作用，如大型语言模型(LLMS)，以及未来智能电网安全中对抗性机器学习的新威胁。



## **37. Adversarial Robustness Limits via Scaling-Law and Human-Alignment Studies**

通过比例定律和人际关系研究的对抗稳健性限制 cs.LG

ICML 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2404.09349v2) [paper-pdf](http://arxiv.org/pdf/2404.09349v2)

**Authors**: Brian R. Bartoldson, James Diffenderfer, Konstantinos Parasyris, Bhavya Kailkhura

**Abstract**: This paper revisits the simple, long-studied, yet still unsolved problem of making image classifiers robust to imperceptible perturbations. Taking CIFAR10 as an example, SOTA clean accuracy is about $100$%, but SOTA robustness to $\ell_{\infty}$-norm bounded perturbations barely exceeds $70$%. To understand this gap, we analyze how model size, dataset size, and synthetic data quality affect robustness by developing the first scaling laws for adversarial training. Our scaling laws reveal inefficiencies in prior art and provide actionable feedback to advance the field. For instance, we discovered that SOTA methods diverge notably from compute-optimal setups, using excess compute for their level of robustness. Leveraging a compute-efficient setup, we surpass the prior SOTA with $20$% ($70$%) fewer training (inference) FLOPs. We trained various compute-efficient models, with our best achieving $74$% AutoAttack accuracy ($+3$% gain). However, our scaling laws also predict robustness slowly grows then plateaus at $90$%: dwarfing our new SOTA by scaling is impractical, and perfect robustness is impossible. To better understand this predicted limit, we carry out a small-scale human evaluation on the AutoAttack data that fools our top-performing model. Concerningly, we estimate that human performance also plateaus near $90$%, which we show to be attributable to $\ell_{\infty}$-constrained attacks' generation of invalid images not consistent with their original labels. Having characterized limiting roadblocks, we outline promising paths for future research.

摘要: 本文回顾了一个简单、研究已久但仍未解决的问题，即使图像分类器对不可察觉的扰动具有健壮性。以CIFAR10为例，SOTA的清洁精度约为$100$%，但对$\ell_{inty}$-范数有界摄动的鲁棒性仅略高于$70$%。为了理解这一差距，我们分析了模型大小、数据集大小和合成数据质量如何通过开发用于对抗性训练的第一个缩放规则来影响稳健性。我们的比例法则揭示了现有技术中的低效，并提供了可操作的反馈来推动该领域的发展。例如，我们发现SOTA方法与计算最优设置明显不同，使用过量计算作为其健壮性级别。利用高效计算的设置，我们比以前的SOTA少了20美元%(70美元%)的培训(推理)失败。我们训练了各种计算效率高的模型，最大限度地达到了$74$%的AutoAttack精度($+3$%的收益)。然而，我们的定标法则也预测稳健性在90美元时缓慢增长然后停滞不前：通过定标来使我们的新SOTA相形见绌是不切实际的，而且完美的稳健性是不可能的。为了更好地理解这一预测极限，我们对AutoAttack数据进行了小规模的人工评估，该评估愚弄了我们的最佳模型。令人担忧的是，我们估计人类的性能也停滞不前近90$%，我们表明这归因于$受限攻击生成的无效图像与其原始标签不一致。在描述了限制障碍的特征之后，我们概述了未来研究的有希望的道路。



## **38. Targeted Augmented Data for Audio Deepfake Detection**

用于音频Deepfake检测的定向增强数据 cs.SD

Accepted in EUSIPCO 2024

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07598v1) [paper-pdf](http://arxiv.org/pdf/2407.07598v1)

**Authors**: Marcella Astrid, Enjie Ghorbel, Djamila Aouada

**Abstract**: The availability of highly convincing audio deepfake generators highlights the need for designing robust audio deepfake detectors. Existing works often rely solely on real and fake data available in the training set, which may lead to overfitting, thereby reducing the robustness to unseen manipulations. To enhance the generalization capabilities of audio deepfake detectors, we propose a novel augmentation method for generating audio pseudo-fakes targeting the decision boundary of the model. Inspired by adversarial attacks, we perturb original real data to synthesize pseudo-fakes with ambiguous prediction probabilities. Comprehensive experiments on two well-known architectures demonstrate that the proposed augmentation contributes to improving the generalization capabilities of these architectures.

摘要: 高度令人信服的音频深度伪造生成器的可用性凸显了设计稳健的音频深度伪造检测器的必要性。现有的作品通常仅依赖于训练集中可用的真实和虚假数据，这可能会导致过度匹配，从而降低对不可见操纵的鲁棒性。为了增强音频深度伪造检测器的概括能力，我们提出了一种新颖的增强方法，用于生成针对模型决策边界的音频伪伪造。受对抗攻击的启发，我们扰乱原始真实数据以合成预测概率模糊的伪假货。对两种知名架构的综合实验表明，所提出的增强有助于提高这些架构的概括能力。



## **39. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别对抗性音频样本 cs.SD

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2305.17000v5) [paper-pdf](http://arxiv.org/pdf/2305.17000v5)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic curve for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，在干净和有噪声的数据下，接收器操作特征曲线下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **40. Evaluating the Adversarial Robustness of Retrieval-Based In-Context Learning for Large Language Models**

评估大型语言模型基于检索的上下文学习的对抗鲁棒性 cs.CL

COLM 2024, 29 pages, 6 figures

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2405.15984v2) [paper-pdf](http://arxiv.org/pdf/2405.15984v2)

**Authors**: Simon Chi Lok Yu, Jie He, Pasquale Minervini, Jeff Z. Pan

**Abstract**: With the emergence of large language models, such as LLaMA and OpenAI GPT-3, In-Context Learning (ICL) gained significant attention due to its effectiveness and efficiency. However, ICL is very sensitive to the choice, order, and verbaliser used to encode the demonstrations in the prompt. Retrieval-Augmented ICL methods try to address this problem by leveraging retrievers to extract semantically related examples as demonstrations. While this approach yields more accurate results, its robustness against various types of adversarial attacks, including perturbations on test samples, demonstrations, and retrieved data, remains under-explored. Our study reveals that retrieval-augmented models can enhance robustness against test sample attacks, outperforming vanilla ICL with a 4.87% reduction in Attack Success Rate (ASR); however, they exhibit overconfidence in the demonstrations, leading to a 2% increase in ASR for demonstration attacks. Adversarial training can help improve the robustness of ICL methods to adversarial attacks; however, such a training scheme can be too costly in the context of LLMs. As an alternative, we introduce an effective training-free adversarial defence method, DARD, which enriches the example pool with those attacked samples. We show that DARD yields improvements in performance and robustness, achieving a 15% reduction in ASR over the baselines. Code and data are released to encourage further research: https://github.com/simonucl/adv-retreival-icl

摘要: 随着大型语言模型的出现，如Llama和OpenAI GPT-3，情景中学习(ICL)因其有效性和高效性而受到广泛关注。但是，ICL对用于对提示符中的演示进行编码的选择、顺序和形容词非常敏感。检索增强的ICL方法试图通过利用检索器来提取语义相关的示例作为演示来解决这个问题。虽然这种方法可以产生更准确的结果，但它对各种类型的对抗性攻击的稳健性，包括对测试样本、演示和检索数据的扰动，仍然没有得到充分的研究。我们的研究表明，检索增强模型可以增强对测试样本攻击的健壮性，性能优于普通ICL，攻击成功率(ASR)降低4.87%；然而，它们在演示中表现出过度自信，导致演示攻击的ASR提高了2%。对抗性训练可以帮助提高ICL方法对对抗性攻击的稳健性；然而，在LLMS的背景下，这样的训练方案可能代价太高。作为另一种选择，我们引入了一种有效的无需训练的对抗防御方法DARD，它用被攻击的样本丰富了样本库。我们表明，DARD在性能和健壮性方面都有改进，ASR比基准降低了15%。发布代码和数据是为了鼓励进一步的研究：https://github.com/simonucl/adv-retreival-icl



## **41. Invisible Optical Adversarial Stripes on Traffic Sign against Autonomous Vehicles**

针对自动驾驶车辆的交通标志上的隐形光学对抗条纹 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07510v1) [paper-pdf](http://arxiv.org/pdf/2407.07510v1)

**Authors**: Dongfang Guo, Yuting Wu, Yimin Dai, Pengfei Zhou, Xin Lou, Rui Tan

**Abstract**: Camera-based computer vision is essential to autonomous vehicle's perception. This paper presents an attack that uses light-emitting diodes and exploits the camera's rolling shutter effect to create adversarial stripes in the captured images to mislead traffic sign recognition. The attack is stealthy because the stripes on the traffic sign are invisible to human. For the attack to be threatening, the recognition results need to be stable over consecutive image frames. To achieve this, we design and implement GhostStripe, an attack system that controls the timing of the modulated light emission to adapt to camera operations and victim vehicle movements. Evaluated on real testbeds, GhostStripe can stably spoof the traffic sign recognition results for up to 94\% of frames to a wrong class when the victim vehicle passes the road section. In reality, such attack effect may fool victim vehicles into life-threatening incidents. We discuss the countermeasures at the levels of camera sensor, perception model, and autonomous driving system.

摘要: 基于摄像头的计算机视觉对于自动驾驶汽车的感知是必不可少的。本文提出了一种利用发光二极管和利用摄像机的滚动快门效应在捕获的图像中产生对抗性条纹来误导交通标志识别的攻击方法。这次袭击是隐形的，因为交通标志上的条纹是人类看不见的。为了使攻击具有威胁性，识别结果需要在连续的图像帧上保持稳定。为了实现这一点，我们设计并实现了Ghost Strike，这是一个攻击系统，它控制调制光发射的时间，以适应相机操作和受害者车辆的移动。在真实的测试平台上进行了测试，当受害者车辆通过路段时，Ghost Strike可以稳定地将高达94%的帧的交通标志识别结果伪造到错误的类别。在现实中，这种攻击效果可能会欺骗受害者车辆发生危及生命的事件。分别从摄像机传感器、感知模型、自动驾驶系统三个层面探讨了对策。



## **42. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.01295v4) [paper-pdf](http://arxiv.org/pdf/2407.01295v4)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **43. Marlin: Knowledge-Driven Analysis of Provenance Graphs for Efficient and Robust Detection of Cyber Attacks**

马林：知识驱动的源源图分析，以高效、稳健地检测网络攻击 cs.CR

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2403.12541v2) [paper-pdf](http://arxiv.org/pdf/2403.12541v2)

**Authors**: Zhenyuan Li, Yangyang Wei, Xiangmin Shen, Lingzhi Wang, Yan Chen, Haitao Xu, Shouling Ji, Fan Zhang, Liang Hou, Wenmao Liu, Xuhong Zhang, Jianwei Ying

**Abstract**: Recent research in both academia and industry has validated the effectiveness of provenance graph-based detection for advanced cyber attack detection and investigation. However, analyzing large-scale provenance graphs often results in substantial overhead. To improve performance, existing detection systems implement various optimization strategies. Yet, as several recent studies suggest, these strategies could lose necessary context information and be vulnerable to evasions. Designing a detection system that is efficient and robust against adversarial attacks is an open problem. We introduce Marlin, which approaches cyber attack detection through real-time provenance graph alignment.By leveraging query graphs embedded with attack knowledge, Marlin can efficiently identify entities and events within provenance graphs, embedding targeted analysis and significantly narrowing the search space. Moreover, we incorporate our graph alignment algorithm into a tag propagation-based schema to eliminate the need for storing and reprocessing raw logs. This design significantly reduces in-memory storage requirements and minimizes data processing overhead. As a result, it enables real-time graph alignment while preserving essential context information, thereby enhancing the robustness of cyber attack detection. Moreover, Marlin allows analysts to customize attack query graphs flexibly to detect extended attacks and provide interpretable detection results. We conduct experimental evaluations on two large-scale public datasets containing 257.42 GB of logs and 12 query graphs of varying sizes, covering multiple attack techniques and scenarios. The results show that Marlin can process 137K events per second while accurately identifying 120 subgraphs with 31 confirmed attacks, along with only 1 false positive, demonstrating its efficiency and accuracy in handling massive data.

摘要: 最近学术界和工业界的研究都证实了基于起源图的检测对于高级网络攻击检测和调查的有效性。然而，分析大规模的种源图表往往会产生相当大的开销。为了提高性能，现有的检测系统采用了各种优化策略。然而，正如最近的几项研究表明的那样，这些策略可能会失去必要的背景信息，并容易受到规避。设计一个对敌方攻击高效且健壮的检测系统是一个悬而未决的问题。介绍了Marlin算法，该算法通过对源图进行实时比对来实现网络攻击的检测，利用嵌入攻击知识的查询图，能够有效地识别源图中的实体和事件，嵌入针对性的分析，大大缩小了搜索空间。此外，我们将我们的图对齐算法整合到基于标记传播的模式中，以消除存储和重新处理原始日志的需要。这种设计显著降低了内存存储需求，并最大限度地减少了数据处理开销。因此，它能够在保留基本上下文信息的同时实现实时图形对齐，从而增强网络攻击检测的健壮性。此外，Marlin允许分析人员灵活地定制攻击查询图，以检测扩展的攻击并提供可解释的检测结果。我们在两个包含257.42 GB日志和12个不同大小的查询图的大规模公共数据集上进行了实验评估，涵盖了多种攻击技术和场景。实验结果表明，Marlin能够在每秒处理137K事件的同时，准确识别出120个子图中31个已确认的攻击，并且只有1个误报，证明了其在处理海量数据时的效率和准确性。



## **44. Characterizing Encrypted Application Traffic through Cellular Radio Interface Protocol**

通过蜂窝无线电接口协议描述加密应用流量 cs.NI

9 pages, 8 figures, 2 tables. This paper has been accepted for  publication by the 21st IEEE International Conference on Mobile Ad-Hoc and  Smart Systems (MASS 2024)

**SubmitDate**: 2024-07-10    [abs](http://arxiv.org/abs/2407.07361v1) [paper-pdf](http://arxiv.org/pdf/2407.07361v1)

**Authors**: Md Ruman Islam, Raja Hasnain Anwar, Spyridon Mastorakis, Muhammad Taqi Raza

**Abstract**: Modern applications are end-to-end encrypted to prevent data from being read or secretly modified. 5G tech nology provides ubiquitous access to these applications without compromising the application-specific performance and latency goals. In this paper, we empirically demonstrate that 5G radio communication becomes the side channel to precisely infer the user's applications in real-time. The key idea lies in observing the 5G physical and MAC layer interactions over time that reveal the application's behavior. The MAC layer receives the data from the application and requests the network to assign the radio resource blocks. The network assigns the radio resources as per application requirements, such as priority, Quality of Service (QoS) needs, amount of data to be transmitted, and buffer size. The adversary can passively observe the radio resources to fingerprint the applications. We empirically demonstrate this attack by considering four different categories of applications: online shopping, voice/video conferencing, video streaming, and Over-The-Top (OTT) media platforms. Finally, we have also demonstrated that an attacker can differentiate various types of applications in real-time within each category.

摘要: 现代应用程序是端到端加密的，以防止数据被读取或秘密修改。5G技术提供了对这些应用的无处不在的访问，而不会影响特定于应用的性能和延迟目标。在本文中，我们实证地论证了5G无线通信成为实时准确推断用户应用的辅助通道。关键思想在于观察5G物理层和MAC层随时间的交互，以揭示应用的行为。MAC层从应用程序接收数据，并请求网络分配无线电资源块。网络根据诸如优先级、服务质量(Qos)需求、要传输的数据量和缓冲区大小等应用需求来分配无线电资源。敌手可以被动地观察无线电资源来识别应用程序。我们考虑了四种不同类别的应用程序：在线购物、语音/视频会议、视频流和Over-the-Top(OTT)媒体平台，对这一攻击进行了实证演示。最后，我们还演示了攻击者可以在每个类别中实时区分各种类型的应用程序。



## **45. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

量子模仿游戏：量子机器学习模型的反向工程 quant-ph

11 pages, 12 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.07237v2) [paper-pdf](http://arxiv.org/pdf/2407.07237v2)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.

摘要: 量子机器学习(QML)融合了量子计算范式和机器学习模型，为解决复杂问题提供了重要的前景。然而，在喧嚣的中间尺度量子计算(NISQ)时代，随着众多第三方供应商的扩张，QML模型的安全性至关重要，特别是在对抗逆向工程时，逆向工程可能会暴露模型的训练参数和算法。我们假设不可信的量子云提供商是一个对手，在推理过程中可以通过白盒访问用户设计的经过训练的QML模型。逆向工程(RE)提取预转换的QML电路将使模型能够重新转置并用于具有完全不同的本机门设置甚至不同的量子比特技术的各种硬件。这种灵活性可能不是从绑定到特定硬件和量子比特技术的分流电路获得的。关于参数数目和最佳值的信息可以允许进一步训练QML模型以改变QML模型、篡改水印、和/或出于其他目的嵌入它们自己的水印或改进模型。在第一次研究QML电路的RE时，我们进行了RE，并比较了不同大小的原始和反向工程量子神经网络(QNN)的训练精度。我们注意到，多量子比特分类器可以在特定条件下进行逆向工程，在合理的时间内，平均误差为1e-2阶。我们还建议在QML模型中增加虚拟固定参数门，以增加防御的RE开销。例如，对于具有2个量子比特和3个层的分类器，添加2个虚拟量子比特和2个层会使开销增加~1.76倍，而性能开销不到9%。我们注意到，RE是一种非常强大的攻击模式，需要在防御上进一步努力。



## **46. Robust Neural Information Retrieval: An Adversarial and Out-of-distribution Perspective**

稳健的神经信息检索：对抗性和非分布性的角度 cs.IR

Survey paper

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06992v1) [paper-pdf](http://arxiv.org/pdf/2407.06992v1)

**Authors**: Yu-An Liu, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, Xueqi Cheng

**Abstract**: Recent advances in neural information retrieval (IR) models have significantly enhanced their effectiveness over various IR tasks. The robustness of these models, essential for ensuring their reliability in practice, has also garnered significant attention. With a wide array of research on robust IR being proposed, we believe it is the opportune moment to consolidate the current status, glean insights from existing methodologies, and lay the groundwork for future development. We view the robustness of IR to be a multifaceted concept, emphasizing its necessity against adversarial attacks, out-of-distribution (OOD) scenarios and performance variance. With a focus on adversarial and OOD robustness, we dissect robustness solutions for dense retrieval models (DRMs) and neural ranking models (NRMs), respectively, recognizing them as pivotal components of the neural IR pipeline. We provide an in-depth discussion of existing methods, datasets, and evaluation metrics, shedding light on challenges and future directions in the era of large language models. To the best of our knowledge, this is the first comprehensive survey on the robustness of neural IR models, and we will also be giving our first tutorial presentation at SIGIR 2024 \url{https://sigir2024-robust-information-retrieval.github.io}. Along with the organization of existing work, we introduce a Benchmark for robust IR (BestIR), a heterogeneous evaluation benchmark for robust neural information retrieval, which is publicly available at \url{https://github.com/Davion-Liu/BestIR}. We hope that this study provides useful clues for future research on the robustness of IR models and helps to develop trustworthy search engines \url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.

摘要: 神经信息检索(IR)模型的最新进展显著提高了它们在各种IR任务中的有效性。这些模型的稳健性对于确保它们在实践中的可靠性至关重要，也引起了人们的极大关注。随着对稳健IR的广泛研究的提出，我们认为现在是巩固当前状况、从现有方法中收集见解并为未来发展奠定基础的好时机。我们认为信息检索的稳健性是一个多方面的概念，强调了它对对抗攻击、分布外(OOD)场景和性能差异的必要性。以对抗性和面向对象的稳健性为重点，我们分别剖析了密集检索模型(DRM)和神经排名模型(NRM)的稳健性解决方案，将它们识别为神经IR管道的关键组件。我们提供了对现有方法、数据集和评估度量的深入讨论，揭示了大型语言模型时代的挑战和未来方向。据我们所知，这是关于神经IR模型稳健性的第一次全面调查，我们还将在SIGIR2024\url{https://sigir2024-robust-information-retrieval.github.io}.上进行我们的第一次教程演示在组织现有工作的同时，我们还介绍了稳健IR基准(BSTIR)，这是一个用于稳健神经信息检索的异质评估基准，可在\url{https://github.com/Davion-Liu/BestIR}.希望本研究为今后研究信息检索模型的健壮性提供有用的线索，并为开发可信搜索引擎\url{https://github.com/Davion-Liu/Awesome-Robustness-in-Information-Retrieval}.提供帮助



## **47. Does CLIP Know My Face?**

CLIP认识我的脸吗？ cs.LG

Published in the Journal of Artificial Intelligence Research (JAIR)

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2209.07341v4) [paper-pdf](http://arxiv.org/pdf/2209.07341v4)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data have become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.

摘要: 随着深度学习在各种应用中的兴起，围绕训练数据保护的隐私问题已经成为一个关键的研究领域。鉴于以往的研究主要集中于单通道模型中的隐私风险，我们引入了一种新的方法来评估多通道模型的隐私，特别是像CLIP这样的视觉语言模型。提出的身份推断攻击(IDIA)通过用同一人的图像查询模型来揭示该人是否包括在训练数据中。让模型从各种各样的可能的文本标签中进行选择，该模型显示它是否识别出这个人，因此，它被用于训练。我们在CLIP上的大规模实验表明，用于训练的个体可以非常准确地识别。我们确认，该模型已经学会了将姓名与所描述的个人相关联，这意味着存在可被对手提取的敏感信息。我们的结果强调了在大规模模型中加强隐私保护的必要性，并建议可以使用IDIA来证明未经授权使用数据进行培训和执行隐私法。



## **48. Performance Evaluation of Knowledge Graph Embedding Approaches under Non-adversarial Attacks**

非对抗性攻击下知识图嵌入方法的性能评估 cs.LG

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.06855v1) [paper-pdf](http://arxiv.org/pdf/2407.06855v1)

**Authors**: Sourabh Kapoor, Arnab Sharma, Michael Röder, Caglar Demir, Axel-Cyrille Ngonga Ngomo

**Abstract**: Knowledge Graph Embedding (KGE) transforms a discrete Knowledge Graph (KG) into a continuous vector space facilitating its use in various AI-driven applications like Semantic Search, Question Answering, or Recommenders. While KGE approaches are effective in these applications, most existing approaches assume that all information in the given KG is correct. This enables attackers to influence the output of these approaches, e.g., by perturbing the input. Consequently, the robustness of such KGE approaches has to be addressed. Recent work focused on adversarial attacks. However, non-adversarial attacks on all attack surfaces of these approaches have not been thoroughly examined. We close this gap by evaluating the impact of non-adversarial attacks on the performance of 5 state-of-the-art KGE algorithms on 5 datasets with respect to attacks on 3 attack surfaces-graph, parameter, and label perturbation. Our evaluation results suggest that label perturbation has a strong effect on the KGE performance, followed by parameter perturbation with a moderate and graph with a low effect.

摘要: 知识图嵌入(KGE)将离散的知识图(KG)转换为连续的向量空间，便于其在语义搜索、问答或推荐器等各种人工智能驱动的应用中的使用。虽然KGE方法在这些应用中是有效的，但大多数现有方法都假设给定KG中的所有信息都是正确的。这使得攻击者能够影响这些方法的输出，例如，通过干扰输入。因此，必须解决这种KGE方法的稳健性问题。最近的工作集中在对抗性攻击上。然而，这些方法的所有攻击面上的非对抗性攻击还没有得到彻底的审查。我们通过评估非对抗性攻击对5种最先进的KGE算法在5个数据集上的性能的影响来缩小这一差距，这些影响涉及3个攻击面-图、参数和标签扰动。我们的评估结果表明，标签扰动对KGE性能的影响很大，其次是参数扰动，影响中等，图的影响较小。



## **49. EvolBA: Evolutionary Boundary Attack under Hard-label Black Box condition**

EvolBA：硬标签黑匣子条件下的进化边界攻击 cs.CV

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2407.02248v3) [paper-pdf](http://arxiv.org/pdf/2407.02248v3)

**Authors**: Ayane Tajima, Satoshi Ono

**Abstract**: Research has shown that deep neural networks (DNNs) have vulnerabilities that can lead to the misrecognition of Adversarial Examples (AEs) with specifically designed perturbations. Various adversarial attack methods have been proposed to detect vulnerabilities under hard-label black box (HL-BB) conditions in the absence of loss gradients and confidence scores.However, these methods fall into local solutions because they search only local regions of the search space. Therefore, this study proposes an adversarial attack method named EvolBA to generate AEs using Covariance Matrix Adaptation Evolution Strategy (CMA-ES) under the HL-BB condition, where only a class label predicted by the target DNN model is available. Inspired by formula-driven supervised learning, the proposed method introduces domain-independent operators for the initialization process and a jump that enhances search exploration. Experimental results confirmed that the proposed method could determine AEs with smaller perturbations than previous methods in images where the previous methods have difficulty.

摘要: 研究表明，深度神经网络(DNN)存在漏洞，可能会导致对经过特殊设计的扰动的对抗性示例(AE)的错误识别。针对硬标签黑盒(HL-BB)环境下不存在损失梯度和置信度的漏洞检测问题，提出了多种对抗性攻击方法，但这些方法只搜索搜索空间的局部区域，容易陷入局部解.因此，本文提出了一种基于协方差矩阵自适应进化策略(CMA-ES)的对抗性攻击方法EvolBA，用于在目标DNN模型预测的类别标签不可用的HL-BB条件下生成AEs。受公式驱动的监督学习的启发，该方法在初始化过程中引入了领域无关的算子，并引入了一个跳跃来增强搜索探索。实验结果表明，该方法能够以较小的扰动确定图像中的声学效应，克服了以往方法的不足。



## **50. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

Accepted to IEEE Euro S&P 2024

**SubmitDate**: 2024-07-09    [abs](http://arxiv.org/abs/2401.04929v3) [paper-pdf](http://arxiv.org/pdf/2401.04929v3)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



