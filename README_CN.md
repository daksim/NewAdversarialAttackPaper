# Latest Adversarial Attack Papers
**update at 2024-03-13 10:01:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robustifying Point Cloud Networks by Refocusing**

通过重新聚焦实现点云网络的规模化 cs.CV

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2308.05525v3) [paper-pdf](http://arxiv.org/pdf/2308.05525v3)

**Authors**: Meir Yossef Levi, Guy Gilboa

**Abstract**: The ability to cope with out-of-distribution (OOD) corruptions and adversarial attacks is crucial in real-world safety-demanding applications. In this study, we develop a general mechanism to increase neural network robustness based on focus analysis.   Recent studies have revealed the phenomenon of \textit{Overfocusing}, which leads to a performance drop. When the network is primarily influenced by small input regions, it becomes less robust and prone to misclassify under noise and corruptions.   However, quantifying overfocusing is still vague and lacks clear definitions. Here, we provide a mathematical definition of \textbf{focus}, \textbf{overfocusing} and \textbf{underfocusing}. The notions are general, but in this study, we specifically investigate the case of 3D point clouds.   We observe that corrupted sets result in a biased focus distribution compared to the clean training set.   We show that as focus distribution deviates from the one learned in the training phase - classification performance deteriorates.   We thus propose a parameter-free \textbf{refocusing} algorithm that aims to unify all corruptions under the same distribution.   We validate our findings on a 3D zero-shot classification task, achieving SOTA in robust 3D classification on ModelNet-C dataset, and in adversarial defense against Shape-Invariant attack. Code is available in: https://github.com/yossilevii100/refocusing.

摘要: 应对分发外(OOD)损坏和敌意攻击的能力在现实世界对安全要求苛刻的应用程序中至关重要。在这项研究中，我们提出了一种基于焦点分析的提高神经网络健壮性的通用机制。最近的研究发现了文本{过度聚焦}的现象，这会导致性能下降。当网络主要受到小输入区域的影响时，它变得不那么健壮，并且容易在噪声和损坏下被错误分类。然而，对过度关注的量化仍然含糊不清，缺乏明确的定义。在这里，我们给出了\extbf{焦点}、\extbf{过度聚焦}和\extbf{欠聚焦}的数学定义。这些概念是一般的，但在本研究中，我们专门研究3D点云的情况。我们观察到，与干净的训练集相比，损坏的集导致了偏向的焦点分布。我们表明，随着焦点分布与在训练阶段学习的焦点分布背离，分类性能会恶化。因此，我们提出了一种无参数的Textbf{重新聚焦}算法，旨在统一同一分布下的所有损坏。我们在3D零镜头分类任务上验证了我们的发现，在ModelNet-C数据集上实现了稳健的3D分类，并在对抗形状不变攻击中实现了SOTA。代码可在以下位置获得：https://github.com/yossilevii100/refocusing.



## **2. Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models**

序列到序列相关性模型上的敌意攻击分析 cs.IR

13 pages, 3 figures, Accepted at ECIR 2024 as a Full Paper

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07654v1) [paper-pdf](http://arxiv.org/pdf/2403.07654v1)

**Authors**: Andrew Parry, Maik Fröbe, Sean MacAvaney, Martin Potthast, Matthias Hagen

**Abstract**: Modern sequence-to-sequence relevance models like monoT5 can effectively capture complex textual interactions between queries and documents through cross-encoding. However, the use of natural language tokens in prompts, such as Query, Document, and Relevant for monoT5, opens an attack vector for malicious documents to manipulate their relevance score through prompt injection, e.g., by adding target words such as true. Since such possibilities have not yet been considered in retrieval evaluation, we analyze the impact of query-independent prompt injection via manually constructed templates and LLM-based rewriting of documents on several existing relevance models. Our experiments on the TREC Deep Learning track show that adversarial documents can easily manipulate different sequence-to-sequence relevance models, while BM25 (as a typical lexical model) is not affected. Remarkably, the attacks also affect encoder-only relevance models (which do not rely on natural language prompt tokens), albeit to a lesser extent.

摘要: 像monT5这样的现代序列到序列相关性模型可以通过交叉编码有效地捕获查询和文档之间的复杂文本交互。然而，在提示中使用自然语言令牌，诸如查询、文档和与monT5相关，打开了恶意文档的攻击向量，以通过提示注入来操纵它们的相关性分数，例如通过添加诸如真的目标词。由于在检索评价中还没有考虑到这种可能性，我们通过手动构建模板和基于LLM的文档重写来分析与查询无关的提示注入对现有几种相关性模型的影响。我们在TREC深度学习路径上的实验表明，敌意文档可以很容易地操纵不同的序列到序列相关性模型，而BM25(一个典型的词汇模型)不受影响。值得注意的是，这些攻击也影响了仅编码者的相关性模型(不依赖于自然语言提示令牌)，尽管影响程度较小。



## **3. Visual Privacy Auditing with Diffusion Models**

基于扩散模型的可视化隐私审计 cs.LG

**SubmitDate**: 2024-03-12    [abs](http://arxiv.org/abs/2403.07588v1) [paper-pdf](http://arxiv.org/pdf/2403.07588v1)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Daniel Rueckert, Georgios Kaissis, Alexander Ziller

**Abstract**: Image reconstruction attacks on machine learning models pose a significant risk to privacy by potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) has proven effective, determining appropriate DP parameters remains challenging. Current formal guarantees on data reconstruction success suffer from overly theoretical assumptions regarding adversary knowledge about the target data, particularly in the image domain. In this work, we empirically investigate this discrepancy and find that the practicality of these assumptions strongly depends on the domain shift between the data prior and the reconstruction target. We propose a reconstruction attack based on diffusion models (DMs) that assumes adversary access to real-world image priors and assess its implications on privacy leakage under DP-SGD. We show that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as effective auditing tools for visualizing privacy leakage.

摘要: 对机器学习模型的图像重建攻击可能会泄露敏感信息，从而对隐私构成重大风险。尽管使用差异隐私(DP)防御此类攻击已被证明是有效的，但确定适当的DP参数仍然具有挑战性。目前对数据重建成功的正式保证受到关于目标数据的敌对知识的过度理论假设的影响，特别是在图像领域。在这项工作中，我们对这种差异进行了实证研究，发现这些假设的实用性很大程度上依赖于先前数据和重建目标之间的域转移。提出了一种基于扩散模型(DMS)的重构攻击，该攻击假定攻击者可以访问真实世界的图像先验，并在DP-SGD下评估了其对隐私泄露的影响。我们发现：(1)真实世界的数据先验对重建成功有显著影响，(2)当前的重建边界没有很好地模拟数据先验带来的风险，(3)数据挖掘可以作为可视化隐私泄露的有效审计工具。



## **4. Improving deep learning with prior knowledge and cognitive models: A survey on enhancing explainability, adversarial robustness and zero-shot learning**

利用先验知识和认知模型改进深度学习：提高解释性、对手稳健性和零命中学习的研究综述 cs.LG

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.07078v1) [paper-pdf](http://arxiv.org/pdf/2403.07078v1)

**Authors**: Fuseinin Mumuni, Alhassan Mumuni

**Abstract**: We review current and emerging knowledge-informed and brain-inspired cognitive systems for realizing adversarial defenses, eXplainable Artificial Intelligence (XAI), and zero-shot or few-short learning. Data-driven deep learning models have achieved remarkable performance and demonstrated capabilities surpassing human experts in many applications. Yet, their inability to exploit domain knowledge leads to serious performance limitations in practical applications. In particular, deep learning systems are exposed to adversarial attacks, which can trick them into making glaringly incorrect decisions. Moreover, complex data-driven models typically lack interpretability or explainability, i.e., their decisions cannot be understood by human subjects. Furthermore, models are usually trained on standard datasets with a closed-world assumption. Hence, they struggle to generalize to unseen cases during inference in practical open-world environments, thus, raising the zero- or few-shot generalization problem. Although many conventional solutions exist, explicit domain knowledge, brain-inspired neural network and cognitive architectures offer powerful new dimensions towards alleviating these problems. Prior knowledge is represented in appropriate forms and incorporated in deep learning frameworks to improve performance. Brain-inspired cognition methods use computational models that mimic the human mind to enhance intelligent behavior in artificial agents and autonomous robots. Ultimately, these models achieve better explainability, higher adversarial robustness and data-efficient learning, and can, in turn, provide insights for cognitive science and neuroscience-that is, to deepen human understanding on how the brain works in general, and how it handles these problems.

摘要: 我们回顾了当前和新兴的知识知情和大脑启发的认知系统，用于实现对抗性防御、可解释人工智能(XAI)和零射击或少数短时间学习。数据驱动的深度学习模型已经取得了显著的性能，并在许多应用中展示了超越人类专家的能力。然而，它们不能利用领域知识导致实际应用中严重的性能限制。特别是，深度学习系统面临敌意攻击，这可能会诱使它们做出明显错误的决定。此外，复杂的数据驱动模型通常缺乏可解释性或可解释性，即它们的决策无法被人类主体理解。此外，模型通常在具有封闭世界假设的标准数据集上进行训练。因此，在实际的开放世界环境中，他们难以在推理过程中将其推广到未知的情况，从而提出了零或极少的推广问题。尽管存在许多传统的解决方案，但显性的领域知识、大脑启发的神经网络和认知架构为缓解这些问题提供了强大的新维度。先验知识以适当的形式表示，并纳入深度学习框架，以提高绩效。大脑启发的认知方法使用模拟人类思维的计算模型来增强人工代理和自主机器人的智能行为。归根结底，这些模型实现了更好的可解释性、更高的对手稳健性和数据效率学习，并反过来可以为认知科学和神经科学提供见解--也就是说，加深人类对大脑总体工作方式以及如何处理这些问题的理解。



## **5. Enhancing Adversarial Training with Prior Knowledge Distillation for Robust Image Compression**

基于先验知识提取的强健图像压缩对抗性训练 eess.IV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06700v1) [paper-pdf](http://arxiv.org/pdf/2403.06700v1)

**Authors**: Cao Zhi, Bao Youneng, Meng Fanyang, Li Chao, Tan Wen, Wang Genhong, Liang Yongsheng

**Abstract**: Deep neural network-based image compression (NIC) has achieved excellent performance, but NIC method models have been shown to be susceptible to backdoor attacks. Adversarial training has been validated in image compression models as a common method to enhance model robustness. However, the improvement effect of adversarial training on model robustness is limited. In this paper, we propose a prior knowledge-guided adversarial training framework for image compression models. Specifically, first, we propose a gradient regularization constraint for training robust teacher models. Subsequently, we design a knowledge distillation based strategy to generate a priori knowledge from the teacher model to the student model for guiding adversarial training. Experimental results show that our method improves the reconstruction quality by about 9dB when the Kodak dataset is elected as the backdoor attack object for psnr attack. Compared with Ma2023, our method has a 5dB higher PSNR output at high bitrate points.

摘要: 基于深度神经网络的图像压缩算法(NIC)取得了很好的效果，但NIC方法模型容易受到后门攻击。对抗性训练已在图像压缩模型中被验证为一种增强模型稳健性的常用方法。然而，对抗性训练对模型稳健性的改善效果有限。本文提出了一种基于先验知识的图像压缩模型对抗性训练框架。具体地说，首先，我们提出了一种用于训练稳健教师模型的梯度正则化约束。随后，我们设计了一种基于知识提炼的策略，从教师模型到学生模型生成先验知识，用于指导对抗性训练。实验结果表明，当选择柯达数据集作为PSNR攻击的后门攻击对象时，我们的方法将重建质量提高了约9DB。与MA2023相比，我们的方法在高码率下的输出峰值信噪比提高了5d B。



## **6. PCLD: Point Cloud Layerwise Diffusion for Adversarial Purification**

PCLD：对抗性净化的点云分层扩散算法 cs.CV

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06698v1) [paper-pdf](http://arxiv.org/pdf/2403.06698v1)

**Authors**: Mert Gulsen, Batuhan Cengiz, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds are extensively employed in a variety of real-world applications such as robotics, autonomous driving and augmented reality. Despite the recent success of point cloud neural networks, especially for safety-critical tasks, it is essential to also ensure the robustness of the model. A typical way to assess a model's robustness is through adversarial attacks, where test-time examples are generated based on gradients to deceive the model. While many different defense mechanisms are studied in 2D, studies on 3D point clouds have been relatively limited in the academic field. Inspired from PointDP, which denoises the network inputs by diffusion, we propose Point Cloud Layerwise Diffusion (PCLD), a layerwise diffusion based 3D point cloud defense strategy. Unlike PointDP, we propagated the diffusion denoising after each layer to incrementally enhance the results. We apply our defense method to different types of commonly used point cloud models and adversarial attacks to evaluate its robustness. Our experiments demonstrate that the proposed defense method achieved results that are comparable to or surpass those of existing methodologies, establishing robustness through a novel technique. Code is available at https://github.com/batuceng/diffusion-layer-robustness-pc.

摘要: 点云被广泛应用于机器人、自动驾驶和增强现实等各种现实应用中。尽管最近点云神经网络取得了成功，特别是对于安全关键任务，但也必须确保模型的健壮性。评估模型稳健性的一种典型方法是通过对抗性攻击，其中测试时间示例是基于梯度生成的，以欺骗模型。虽然在二维空间中研究了许多不同的防御机制，但学术界对三维点云的研究相对有限。受PointDP扩散去噪的启发，我们提出了点云分层扩散(PCLD)，这是一种基于分层扩散的三维点云防御策略。与PointDP不同的是，我们在每一层之后传播扩散去噪，以增量地增强结果。我们将我们的防御方法应用于不同类型的常用点云模型和对抗性攻击，以评估其健壮性。我们的实验表明，该防御方法取得了与现有方法相当或超过的结果，通过一种新的技术建立了稳健性。代码可在https://github.com/batuceng/diffusion-layer-robustness-pc.上找到



## **7. PeerAiD: Improving Adversarial Distillation from a Specialized Peer Tutor**

PeerAiD：从专业的同行导师那里改进对抗性蒸馏 cs.LG

Accepted to CVPR 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06668v1) [paper-pdf](http://arxiv.org/pdf/2403.06668v1)

**Authors**: Jaewon Jung, Hongsun Jang, Jaeyong Song, Jinho Lee

**Abstract**: Adversarial robustness of the neural network is a significant concern when it is applied to security-critical domains. In this situation, adversarial distillation is a promising option which aims to distill the robustness of the teacher network to improve the robustness of a small student network. Previous works pretrain the teacher network to make it robust to the adversarial examples aimed at itself. However, the adversarial examples are dependent on the parameters of the target network. The fixed teacher network inevitably degrades its robustness against the unseen transferred adversarial examples which targets the parameters of the student network in the adversarial distillation process. We propose PeerAiD to make a peer network learn the adversarial examples of the student network instead of adversarial examples aimed at itself. PeerAiD is an adversarial distillation that trains the peer network and the student network simultaneously in order to make the peer network specialized for defending the student network. We observe that such peer networks surpass the robustness of pretrained robust teacher network against student-attacked adversarial samples. With this peer network and adversarial distillation, PeerAiD achieves significantly higher robustness of the student network with AutoAttack (AA) accuracy up to 1.66%p and improves the natural accuracy of the student network up to 4.72%p with ResNet-18 and TinyImageNet dataset.

摘要: 当神经网络应用于安全关键领域时，它的对抗健壮性是一个重要的问题。在这种情况下，对抗性蒸馏是一种很有前途的选择，它旨在提取教师网络的健壮性，以提高小型学生网络的健壮性。以前的工作预先训练教师网络，使其对针对自己的对抗性例子具有健壮性。然而，对抗性的例子取决于目标网络的参数。在对抗性提取过程中，固定的教师网络不可避免地降低了其对看不见的转移的对抗性范例的鲁棒性，这些例子针对的是学生网络的参数。我们建议PeerAiD使对等网络学习学生网络的对抗性例子，而不是针对自己的对抗性例子。PeerAiD是一种对抗性的升华，它同时训练对等网络和学生网络，使对等网络专门用于防御学生网络。我们观察到这种对等网络超过了预先训练的稳健教师网络对学生攻击的对手样本的稳健性。通过这种对等网络和对抗性蒸馏，PeerAiD实现了显著更高的学生网络的健壮性，AutoAttack(AA)准确率高达1.66%p，并使用ResNet-18和TinyImageNet数据集将学生网络的自然准确率提高到4.72%p。



## **8. epsilon-Mesh Attack: A Surface-based Adversarial Point Cloud Attack for Facial Expression Recognition**

Epsilon-Mesh攻击：一种基于表面的人脸表情识别对抗性点云攻击 cs.CV

Accepted at 18th IEEE International Conference on Automatic Face &  Gesture Recognition (FG 2024)

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06661v1) [paper-pdf](http://arxiv.org/pdf/2403.06661v1)

**Authors**: Batuhan Cengiz, Mert Gulsen, Yusuf H. Sahin, Gozde Unal

**Abstract**: Point clouds and meshes are widely used 3D data structures for many computer vision applications. While the meshes represent the surfaces of an object, point cloud represents sampled points from the surface which is also the output of modern sensors such as LiDAR and RGB-D cameras. Due to the wide application area of point clouds and the recent advancements in deep neural networks, studies focusing on robust classification of the 3D point cloud data emerged. To evaluate the robustness of deep classifier networks, a common method is to use adversarial attacks where the gradient direction is followed to change the input slightly. The previous studies on adversarial attacks are generally evaluated on point clouds of daily objects. However, considering 3D faces, these adversarial attacks tend to affect the person's facial structure more than the desired amount and cause malformation. Specifically for facial expressions, even a small adversarial attack can have a significant effect on the face structure. In this paper, we suggest an adversarial attack called $\epsilon$-Mesh Attack, which operates on point cloud data via limiting perturbations to be on the mesh surface. We also parameterize our attack by $\epsilon$ to scale the perturbation mesh. Our surface-based attack has tighter perturbation bounds compared to $L_2$ and $L_\infty$ norm bounded attacks that operate on unit-ball. Even though our method has additional constraints, our experiments on CoMA, Bosphorus and FaceWarehouse datasets show that $\epsilon$-Mesh Attack (Perpendicular) successfully confuses trained DGCNN and PointNet models $99.72\%$ and $97.06\%$ of the time, with indistinguishable facial deformations. The code is available at https://github.com/batuceng/e-mesh-attack.

摘要: 点云和网格是许多计算机视觉应用中广泛使用的三维数据结构。网格表示对象的表面，点云表示来自表面的采样点，这也是现代传感器(如LiDAR和RGB-D相机)的输出。由于点云的广泛应用领域和深度神经网络的最新进展，三维点云数据的稳健分类研究应运而生。为了评估深度分类器网络的稳健性，常用的方法是使用对抗性攻击，其中遵循梯度方向略微改变输入。以往关于对抗性攻击的研究一般都是基于日常物体的点云进行评估。然而，考虑到3D人脸，这些对抗性攻击往往会对人的面部结构造成比预期更大的影响，并导致畸形。具体地说，对于面部表情，即使是一次小小的对抗性攻击也可能对面部结构产生重大影响。在本文中，我们提出了一种对抗性攻击，称为$\epsilon$-Mesh攻击，它通过将扰动限制在网格曲面上来操作点云数据。我们还用$\epsilon$对我们的攻击进行了参数化，以扩大扰动网格。与单位球上的$L_2$和$L_\inty$有界进攻相比，我们的地面进攻具有更严格的扰动界。尽管我们的方法有额外的限制，但我们在Coma、Bsporus和FaceWarehouse数据集上的实验表明，$epsilon$-Mesh攻击(垂直)成功地将训练过的DGCNN和PointNet模型$99.72\$和$97.06\$混淆为无法区分的面部变形。代码可在https://github.com/batuceng/e-mesh-attack.上获得



## **9. Real is not True: Backdoor Attacks Against Deepfake Detection**

真实不是真的：对深伪检测的后门攻击 cs.CR

BigDIA 2023

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06610v1) [paper-pdf](http://arxiv.org/pdf/2403.06610v1)

**Authors**: Hong Sun, Ziqiang Li, Lei Liu, Bin Li

**Abstract**: The proliferation of malicious deepfake applications has ignited substantial public apprehension, casting a shadow of doubt upon the integrity of digital media. Despite the development of proficient deepfake detection mechanisms, they persistently demonstrate pronounced vulnerability to an array of attacks. It is noteworthy that the pre-existing repertoire of attacks predominantly comprises adversarial example attack, predominantly manifesting during the testing phase. In the present study, we introduce a pioneering paradigm denominated as Bad-Deepfake, which represents a novel foray into the realm of backdoor attacks levied against deepfake detectors. Our approach hinges upon the strategic manipulation of a delimited subset of the training data, enabling us to wield disproportionate influence over the operational characteristics of a trained model. This manipulation leverages inherent frailties inherent to deepfake detectors, affording us the capacity to engineer triggers and judiciously select the most efficacious samples for the construction of the poisoned set. Through the synergistic amalgamation of these sophisticated techniques, we achieve an remarkable performance-a 100% attack success rate (ASR) against extensively employed deepfake detectors.

摘要: 恶意深度假冒应用程序的激增引发了公众的大量担忧，给数字媒体的完整性蒙上了一层疑云。尽管发展了熟练的深度假冒检测机制，但它们对一系列攻击始终表现出明显的脆弱性。值得注意的是，先前存在的攻击曲目主要包括对抗性示例攻击，主要表现在测试阶段。在目前的研究中，我们介绍了一个名为Bad-Deepfac的开创性范例，它代表了对深度假检测器的后门攻击领域的一种新的进军。我们的方法取决于对训练数据的定界子集的战略操纵，使我们能够对训练模型的操作特征施加不成比例的影响。这种操纵利用了深度假探测器固有的弱点，使我们能够设计触发器，并明智地选择最有效的样本来构建有毒的集合。通过这些复杂技术的协同融合，我们获得了显著的性能-对广泛使用的深度假冒检测器的攻击成功率(ASR)为100%。



## **10. DNNShield: Embedding Identifiers for Deep Neural Network Ownership Verification**

DNNShield：用于深度神经网络所有权验证的嵌入标识 cs.CR

18 pages, 11 figures, 6 tables

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06581v1) [paper-pdf](http://arxiv.org/pdf/2403.06581v1)

**Authors**: Jasper Stang, Torsten Krauß, Alexandra Dmitrienko

**Abstract**: The surge in popularity of machine learning (ML) has driven significant investments in training Deep Neural Networks (DNNs). However, these models that require resource-intensive training are vulnerable to theft and unauthorized use. This paper addresses this challenge by introducing DNNShield, a novel approach for DNN protection that integrates seamlessly before training. DNNShield embeds unique identifiers within the model architecture using specialized protection layers. These layers enable secure training and deployment while offering high resilience against various attacks, including fine-tuning, pruning, and adaptive adversarial attacks. Notably, our approach achieves this security with minimal performance and computational overhead (less than 5\% runtime increase). We validate the effectiveness and efficiency of DNNShield through extensive evaluations across three datasets and four model architectures. This practical solution empowers developers to protect their DNNs and intellectual property rights.

摘要: 机器学习(ML)的普及推动了在训练深度神经网络(DNN)方面的大量投资。然而，这些需要资源密集型培训的型号很容易被窃取和未经授权使用。本文通过引入DNNShield来解决这一挑战，DNNShield是一种在训练前无缝集成的DNN保护新方法。DNNShield使用专门的保护层在模型体系结构中嵌入唯一的标识符。这些层支持安全的培训和部署，同时提供对各种攻击的高弹性，包括微调、修剪和自适应对手攻击。值得注意的是，我们的方法以最小的性能和计算开销(不到5%的运行时增加)实现了这种安全性。我们通过对三个数据集和四个模型体系结构的广泛评估来验证DNNShield的有效性和效率。这一实用的解决方案使开发商能够保护他们的DNN和知识产权。



## **11. DistriBlock: Identifying adversarial audio samples by leveraging characteristics of the output distribution**

DistriBlock：通过利用输出分布的特征来识别敌意音频样本 cs.SD

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2305.17000v3) [paper-pdf](http://arxiv.org/pdf/2305.17000v3)

**Authors**: Matías Pizarro, Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks can mislead automatic speech recognition (ASR) systems into predicting an arbitrary target text, thus posing a clear security threat. To prevent such attacks, we propose DistriBlock, an efficient detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy of the distribution, as well as the Kullback-Leibler and the Jensen-Shannon divergence with respect to the distributions of the subsequent time step. Then, by leveraging the characteristics observed for both benign and adversarial data, we apply binary classifiers, including simple threshold-based classification, ensembles of such classifiers, and neural networks. Through extensive analysis across different state-of-the-art ASR systems and language data sets, we demonstrate the supreme performance of this approach, with a mean area under the receiver operating characteristic for distinguishing target adversarial examples against clean and noisy data of 99% and 97%, respectively. To assess the robustness of our method, we show that adaptive adversarial examples that can circumvent DistriBlock are much noisier, which makes them easier to detect through filtering and creates another avenue for preserving the system's robustness.

摘要: 敌意攻击可以误导自动语音识别(ASR)系统预测任意目标文本，从而构成明显的安全威胁。为了防止此类攻击，我们提出了DistriBlock，这是一种适用于任何ASR系统的有效检测策略，它预测每个时间步输出令牌上的概率分布。我们测量了该分布的一组特征：输出概率的中位数、最大值和最小值，分布的熵，以及关于后续时间步分布的Kullback-Leibler和Jensen-Shannon散度。然后，通过利用对良性数据和恶意数据观察到的特征，我们应用二进制分类器，包括简单的基于阈值的分类、这种分类器的集成和神经网络。通过对不同的ASR系统和语言数据集的广泛分析，我们证明了该方法的最高性能，对于干净和有噪声的数据，接收器操作特征下的平均面积分别为99%和97%。为了评估我们方法的健壮性，我们证明了可以绕过DistriBlock的自适应攻击示例的噪声要大得多，这使得它们更容易通过过滤来检测，并为保持系统的健壮性创造了另一种途径。



## **12. Fooling Neural Networks for Motion Forecasting via Adversarial Attacks**

利用对抗性攻击愚弄神经网络进行运动预测 cs.CV

11 pages, 8 figures, VISSAP 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.04954v2) [paper-pdf](http://arxiv.org/pdf/2403.04954v2)

**Authors**: Edgar Medina, Leyong Loh

**Abstract**: Human motion prediction is still an open problem, which is extremely important for autonomous driving and safety applications. Although there are great advances in this area, the widely studied topic of adversarial attacks has not been applied to multi-regression models such as GCNs and MLP-based architectures in human motion prediction. This work intends to reduce this gap using extensive quantitative and qualitative experiments in state-of-the-art architectures similar to the initial stages of adversarial attacks in image classification. The results suggest that models are susceptible to attacks even on low levels of perturbation. We also show experiments with 3D transformations that affect the model performance, in particular, we show that most models are sensitive to simple rotations and translations which do not alter joint distances. We conclude that similar to earlier CNN models, motion forecasting tasks are susceptible to small perturbations and simple 3D transformations.

摘要: 人体运动预测仍然是一个悬而未决的问题，对于自动驾驶和安全应用具有极其重要的意义。尽管这一领域已经取得了很大的进展，但被广泛研究的对抗性攻击主题还没有被应用到多元回归模型中，如GCNS和基于MLP的人体运动预测体系。这项工作旨在通过在最先进的体系结构中进行广泛的定量和定性实验来缩小这一差距，该体系结构类似于图像分类中对抗性攻击的初始阶段。结果表明，即使在低水平的扰动下，模型也容易受到攻击。我们还展示了影响模型性能的3D变换的实验，特别是，我们表明大多数模型对简单的旋转和平移都很敏感，这些旋转和平移不会改变关节距离。我们的结论是，与早期的CNN模型类似，运动预测任务容易受到小扰动和简单的3D变换的影响。



## **13. Intra-Section Code Cave Injection for Adversarial Evasion Attacks on Windows PE Malware File**

针对Windows PE恶意软件文件的对抗性逃避攻击的段内代码洞穴注入 cs.CR

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06428v1) [paper-pdf](http://arxiv.org/pdf/2403.06428v1)

**Authors**: Kshitiz Aryal, Maanak Gupta, Mahmoud Abdelsalam, Moustafa Saleh

**Abstract**: Windows malware is predominantly available in cyberspace and is a prime target for deliberate adversarial evasion attacks. Although researchers have investigated the adversarial malware attack problem, a multitude of important questions remain unanswered, including (a) Are the existing techniques to inject adversarial perturbations in Windows Portable Executable (PE) malware files effective enough for evasion purposes?; (b) Does the attack process preserve the original behavior of malware?; (c) Are there unexplored approaches/locations that can be used to carry out adversarial evasion attacks on Windows PE malware?; and (d) What are the optimal locations and sizes of adversarial perturbations required to evade an ML-based malware detector without significant structural change in the PE file? To answer some of these questions, this work proposes a novel approach that injects a code cave within the section (i.e., intra-section) of Windows PE malware files to make space for adversarial perturbations. In addition, a code loader is also injected inside the PE file, which reverts adversarial malware to its original form during the execution, preserving the malware's functionality and executability. To understand the effectiveness of our approach, we injected adversarial perturbations inside the .text, .data and .rdata sections, generated using the gradient descent and Fast Gradient Sign Method (FGSM), to target the two popular CNN-based malware detectors, MalConv and MalConv2. Our experiments yielded notable results, achieving a 92.31% evasion rate with gradient descent and 96.26% with FGSM against MalConv, compared to the 16.17% evasion rate for append attacks. Similarly, when targeting MalConv2, our approach achieved a remarkable maximum evasion rate of 97.93% with gradient descent and 94.34% with FGSM, significantly surpassing the 4.01% evasion rate observed with append attacks.

摘要: Windows恶意软件主要在网络空间可用，是蓄意对抗性逃避攻击的主要目标。为了回答这些问题，本工作提出了一种新的方法，在Windows PE恶意软件文件的段(即段内)内注入代码洞，为对抗性扰动腾出空间。此外，还在PE文件中注入了代码加载器，在执行过程中将恶意软件还原为其原始形式，保留了恶意软件的功能和可执行性。我们的实验取得了显著的效果，与Append攻击16.17%的逃避率相比，梯度下降和FGSM对MalConv的逃避率分别达到了92.31%和96.26%。同样，当目标为MalConv2时，我们的方法获得了97.93%的最大逃避率和94.34%的FGSM逃避率，显著超过了Append攻击的4.01%的逃避率。



## **14. A Zero Trust Framework for Realization and Defense Against Generative AI Attacks in Power Grid**

一种实现和防御电网生成性AI攻击的零信任框架 cs.CR

Accepted article by IEEE International Conference on Communications  (ICC 2024), Copyright 2024 IEEE

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2403.06388v1) [paper-pdf](http://arxiv.org/pdf/2403.06388v1)

**Authors**: Md. Shirajum Munir, Sravanthi Proddatoori, Manjushree Muralidhara, Walid Saad, Zhu Han, Sachin Shetty

**Abstract**: Understanding the potential of generative AI (GenAI)-based attacks on the power grid is a fundamental challenge that must be addressed in order to protect the power grid by realizing and validating risk in new attack vectors. In this paper, a novel zero trust framework for a power grid supply chain (PGSC) is proposed. This framework facilitates early detection of potential GenAI-driven attack vectors (e.g., replay and protocol-type attacks), assessment of tail risk-based stability measures, and mitigation of such threats. First, a new zero trust system model of PGSC is designed and formulated as a zero-trust problem that seeks to guarantee for a stable PGSC by realizing and defending against GenAI-driven cyber attacks. Second, in which a domain-specific generative adversarial networks (GAN)-based attack generation mechanism is developed to create a new vulnerability cyberspace for further understanding that threat. Third, tail-based risk realization metrics are developed and implemented for quantifying the extreme risk of a potential attack while leveraging a trust measurement approach for continuous validation. Fourth, an ensemble learning-based bootstrap aggregation scheme is devised to detect the attacks that are generating synthetic identities with convincing user and distributed energy resources device profiles. Experimental results show the efficacy of the proposed zero trust framework that achieves an accuracy of 95.7% on attack vector generation, a risk measure of 9.61% for a 95% stable PGSC, and a 99% confidence in defense against GenAI-driven attack.

摘要: 了解基于生成性人工智能(GenAI)的攻击对电网的潜在影响是一项必须解决的根本挑战，以便通过识别和验证新攻击载体中的风险来保护电网。提出了一种新的电网供应链零信任框架。该框架有助于及早检测潜在的GenAI驱动的攻击载体(例如，重放和协议型攻击)，评估基于尾部风险的稳定性措施，并缓解此类威胁。首先，设计并建立了一种新的PGSC零信任系统模型，将其描述为一个零信任问题，旨在通过实现和防御GenAI驱动的网络攻击来保证稳定的PGSC。其次，开发了一种基于特定领域生成性对抗网络(GAN)的攻击生成机制，以创建一个新的易受攻击的网络空间，以进一步了解该威胁。第三，开发和实现了基于尾部的风险实现度量，用于量化潜在攻击的极端风险，同时利用信任度量方法进行持续验证。第四，设计了一种基于集成学习的Bootstrap聚合方案，用于检测通过令人信服的用户和分布式能源设备配置文件生成合成身份的攻击。实验结果表明，提出的零信任框架对攻击向量生成的准确率为95.7%，对95%稳定的PGSC的风险度量为9.61%，对GenAI驱动的攻击的防御置信度为99%。



## **15. Towards Scalable and Robust Model Versioning**

走向可扩展和健壮的模型版本控制 cs.LG

Published in IEEE SaTML 2024

**SubmitDate**: 2024-03-11    [abs](http://arxiv.org/abs/2401.09574v2) [paper-pdf](http://arxiv.org/pdf/2401.09574v2)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.

摘要: 随着深度学习模型的部署继续跨行业扩展，旨在访问这些已部署模型的恶意入侵威胁正在上升。如果攻击者获得对已部署模型的访问权限，无论是通过服务器入侵、内部攻击或模型倒置技术，他们都可以构建白盒对抗性攻击来操纵模型的分类结果，从而给依赖这些模型执行关键任务的组织带来重大风险。模型所有者需要机制来保护自己免受此类损失，而不需要获取新的培训数据-这一过程通常需要在时间和资金上进行大量投资。在本文中，我们探索了在不获取新的训练数据或改变模型体系结构的情况下，生成具有不同攻击属性的模型的多个版本的可行性。模型所有者可以一次部署一个版本，并立即用新版本替换泄漏的版本。新部署的模型版本可以抵抗利用白盒访问一个或所有先前泄露的版本而产生的对抗性攻击。我们从理论上证明，这可以通过将参数化的隐藏分布结合到模型训练数据中来实现，迫使模型学习由所选数据唯一定义的与任务无关的特征。此外，隐藏分布的最佳选择可以产生一系列模型版本，能够随着时间的推移抵抗复合可转移性攻击。利用我们的分析洞察力，我们设计并实现了一种实用的DNN分类器模型版本控制方法，与现有方法相比，该方法具有显著的健壮性改进。我们相信，我们的工作为保护DNN服务提供了一个很有前途的方向，而不是最初的部署。



## **16. Fake or Compromised? Making Sense of Malicious Clients in Federated Learning**

是假的还是妥协的？联合学习中对恶意客户端的理解 cs.LG

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2403.06319v1) [paper-pdf](http://arxiv.org/pdf/2403.06319v1)

**Authors**: Hamid Mozaffari, Sunav Choudhary, Amir Houmansadr

**Abstract**: Federated learning (FL) is a distributed machine learning paradigm that enables training models on decentralized data. The field of FL security against poisoning attacks is plagued with confusion due to the proliferation of research that makes different assumptions about the capabilities of adversaries and the adversary models they operate under. Our work aims to clarify this confusion by presenting a comprehensive analysis of the various poisoning attacks and defensive aggregation rules (AGRs) proposed in the literature, and connecting them under a common framework. To connect existing adversary models, we present a hybrid adversary model, which lies in the middle of the spectrum of adversaries, where the adversary compromises a few clients, trains a generative (e.g., DDPM) model with their compromised samples, and generates new synthetic data to solve an optimization for a stronger (e.g., cheaper, more practical) attack against different robust aggregation rules. By presenting the spectrum of FL adversaries, we aim to provide practitioners and researchers with a clear understanding of the different types of threats they need to consider when designing FL systems, and identify areas where further research is needed.

摘要: 联合学习(FL)是一种分布式机器学习范例，支持对分散数据的训练模型。由于越来越多的研究对对手的能力和他们所在的对手模型做出了不同的假设，因此针对中毒攻击的FL安全领域充满了困惑。我们的工作旨在通过对文献中提出的各种中毒攻击和防御聚集规则(AGR)进行全面分析，并在一个共同的框架下将它们联系起来，来澄清这种混淆。为了连接现有的敌手模型，我们提出了一种混合敌手模型，该模型位于敌手光谱的中间，其中敌手妥协一些客户端，用他们妥协的样本训练产生式(例如，DDPM)模型，并生成新的合成数据来解决针对不同健壮聚集规则的更强(例如，更便宜、更实用)攻击的优化。通过介绍外语对手的范围，我们的目的是让从业者和研究人员清楚地了解他们在设计外语系统时需要考虑的不同类型的威胁，并确定需要进一步研究的领域。



## **17. Improving behavior based authentication against adversarial attack using XAI**

 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2402.16430v2) [paper-pdf](http://arxiv.org/pdf/2402.16430v2)

**Authors**: Dong Qin, George Amariucai, Daji Qiao, Yong Guan

**Abstract**: In recent years, machine learning models, especially deep neural networks, have been widely used for classification tasks in the security domain. However, these models have been shown to be vulnerable to adversarial manipulation: small changes learned by an adversarial attack model, when applied to the input, can cause significant changes in the output. Most research on adversarial attacks and corresponding defense methods focuses only on scenarios where adversarial samples are directly generated by the attack model. In this study, we explore a more practical scenario in behavior-based authentication, where adversarial samples are collected from the attacker. The generated adversarial samples from the model are replicated by attackers with a certain level of discrepancy. We propose an eXplainable AI (XAI) based defense strategy against adversarial attacks in such scenarios. A feature selector, trained with our method, can be used as a filter in front of the original authenticator. It filters out features that are more vulnerable to adversarial attacks or irrelevant to authentication, while retaining features that are more robust. Through comprehensive experiments, we demonstrate that our XAI based defense strategy is effective against adversarial attacks and outperforms other defense strategies, such as adversarial training and defensive distillation.

摘要: 近年来，机器学习模型，特别是深度神经网络被广泛应用于安全领域的分类任务。然而，这些模型已被证明容易受到对抗性操纵：对抗性攻击模型学习到的微小变化，当应用于输入时，可能会导致输出的重大变化。关于对抗性攻击及其防御方法的研究大多集中在攻击模型直接生成对抗性样本的场景中。在这项研究中，我们探索了一种更实用的基于行为的身份验证场景，其中从攻击者那里收集了敌意样本。从该模型生成的对抗性样本被具有一定差异的攻击者复制。我们提出了一种基于可解释人工智能(XAI)的防御策略，以抵御此类场景中的对抗性攻击。用我们的方法训练的特征选择器可以用作原始认证器前面的过滤器。它过滤掉更容易受到对手攻击或与身份验证无关的功能，同时保留更健壮的功能。通过综合实验，我们证明了我们的基于XAI的防御策略对对手攻击是有效的，并且优于其他防御策略，如对抗性训练和防御蒸馏。



## **18. Learn from the Past: A Proxy Guided Adversarial Defense Framework with Self Distillation Regularization**

借鉴过去：一种自蒸馏正规化的代理制导对抗防御框架 cs.LG

13 Pages

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.12713v2) [paper-pdf](http://arxiv.org/pdf/2310.12713v2)

**Authors**: Yaohua Liu, Jiaxin Gao, Xianghao Jiao, Zhu Liu, Xin Fan, Risheng Liu

**Abstract**: Adversarial Training (AT), pivotal in fortifying the robustness of deep learning models, is extensively adopted in practical applications. However, prevailing AT methods, relying on direct iterative updates for target model's defense, frequently encounter obstacles such as unstable training and catastrophic overfitting. In this context, our work illuminates the potential of leveraging the target model's historical states as a proxy to provide effective initialization and defense prior, which results in a general proxy guided defense framework, `LAST' ({\bf L}earn from the P{\bf ast}). Specifically, LAST derives response of the proxy model as dynamically learned fast weights, which continuously corrects the update direction of the target model. Besides, we introduce a self-distillation regularized defense objective, ingeniously designed to steer the proxy model's update trajectory without resorting to external teacher models, thereby ameliorating the impact of catastrophic overfitting on performance. Extensive experiments and ablation studies showcase the framework's efficacy in markedly improving model robustness (e.g., up to 9.2\% and 20.3\% enhancement in robust accuracy on CIFAR10 and CIFAR100 datasets, respectively) and training stability. These improvements are consistently observed across various model architectures, larger datasets, perturbation sizes, and attack modalities, affirming LAST's ability to consistently refine both single-step and multi-step AT strategies. The code will be available at~\url{https://github.com/callous-youth/LAST}.

摘要: 对抗性训练(AT)是增强深度学习模型鲁棒性的关键，在实际应用中得到了广泛的应用。然而，目前流行的AT方法依赖于直接迭代更新来防御目标模型，经常会遇到训练不稳定、灾难性过拟合等障碍。在此背景下，我们的工作阐明了利用目标模型的历史状态作为代理来提供有效的初始化和防御事前的潜力，从而产生了一个通用的代理制导防御框架`last‘(L从P{\bf ast}中赚取)。具体地说，LAST将代理模型的响应导出为动态学习的快速权值，从而不断修正目标模型的更新方向。此外，我们还引入了自蒸馏正则化防御目标，巧妙地设计了在不依赖外部教师模型的情况下引导代理模型的更新轨迹，从而改善了灾难性过拟合对性能的影响。广泛的实验和烧蚀研究表明，该框架在显著提高模型稳健性(例如，在CIFAR10和CIFAR100数据集上的稳健精度分别提高高达9.2和20.3%)和训练稳定性方面的有效性。这些改进在不同的模型架构、更大的数据集、扰动大小和攻击模式中都得到了一致的观察，证实了LAST一致地改进单步和多步AT策略的能力。该代码将在at~\url{https://github.com/callous-youth/LAST}.上提供



## **19. Deep Reinforcement Learning with Spiking Q-learning**

基于尖峰Q学习的深度强化学习 cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2201.09754v2) [paper-pdf](http://arxiv.org/pdf/2201.09754v2)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.

摘要: 在特殊的神经形态硬件的帮助下，脉冲神经网络(SNN)有望以更少的能量消耗实现人工智能(AI)。它将神经网络和深度强化学习相结合，为实际控制任务提供了一种很有前途的节能方法。目前已有的基于SNN的RL方法很少。大多数人要么缺乏泛化能力，要么在训练中使用人工神经网络(ANN)来估计价值函数。前者需要针对每个场景调整大量的超参数，而后者限制了不同类型RL算法的应用，忽略了训练过程中的巨大能量消耗。为了开发一种稳健的基于棘波的RL方法，我们从昆虫中发现的非尖峰中间神经元中吸取灵感，提出了深度尖峰Q-网络(DSQN)，它使用非尖峰神经元的膜电压作为Q值的表示，可以使用端到端RL直接从高维感觉输入中学习鲁棒策略。在17个Atari游戏上的实验表明，DSQN是有效的，甚至在大多数游戏中都优于基于神经网络的深度Q网络(DQN)。实验表明，DSQN具有良好的学习稳定性和对敌意攻击的稳健性。



## **20. Language-Driven Anchors for Zero-Shot Adversarial Robustness**

语言驱动的零射击对抗稳健性锚 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2301.13096v3) [paper-pdf](http://arxiv.org/pdf/2301.13096v3)

**Authors**: Xiao Li, Wei Zhang, Yining Liu, Zhanhao Hu, Bo Zhang, Xiaolin Hu

**Abstract**: Deep Neural Networks (DNNs) are known to be susceptible to adversarial attacks. Previous researches mainly focus on improving adversarial robustness in the fully supervised setting, leaving the challenging domain of zero-shot adversarial robustness an open question. In this work, we investigate this domain by leveraging the recent advances in large vision-language models, such as CLIP, to introduce zero-shot adversarial robustness to DNNs. We propose LAAT, a Language-driven, Anchor-based Adversarial Training strategy. LAAT utilizes the features of a text encoder for each category as fixed anchors (normalized feature embeddings) for each category, which are then employed for adversarial training. By leveraging the semantic consistency of the text encoders, LAAT aims to enhance the adversarial robustness of the image model on novel categories. However, naively using text encoders leads to poor results. Through analysis, we identified the issue to be the high cosine similarity between text encoders. We then design an expansion algorithm and an alignment cross-entropy loss to alleviate the problem. Our experimental results demonstrated that LAAT significantly improves zero-shot adversarial robustness over state-of-the-art methods. LAAT has the potential to enhance adversarial robustness by large-scale multimodal models, especially when labeled data is unavailable during training.

摘要: 深度神经网络(DNN)是公认的易受敌意攻击的网络。以往的研究主要集中在提高完全监督环境下的对抗稳健性，而对零射击对抗稳健性这一挑战领域的研究还是个未知数。在这项工作中，我们通过利用大型视觉语言模型(如CLIP)的最新进展来研究这一领域，为DNN引入零射击对抗性健壮性。我们提出了LAAT，一种语言驱动的、基于锚的对抗性训练策略。LAAT利用每个类别的文本编码器的特征作为每个类别的固定锚(归一化特征嵌入)，然后将其用于对抗性训练。通过利用文本编码者的语义一致性，LAAT旨在增强图像模型在新类别上的对抗性健壮性。然而，幼稚地使用文本编码器会导致较差的结果。通过分析，我们认为问题在于文本编码者之间存在很高的余弦相似度。然后，我们设计了扩展算法和对齐交叉熵损失来缓解该问题。我们的实验结果表明，与最先进的方法相比，LAAT显著提高了零命中对手的稳健性。LAAT有可能通过大规模多模式模型来增强对手的稳健性，特别是在训练过程中无法获得标记数据的情况下。



## **21. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

对抗性净化训练(TOOP)：提高健壮性和泛化能力 cs.CV

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2401.16352v2) [paper-pdf](http://arxiv.org/pdf/2401.16352v2)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline called Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks resulting in the robustness generalization to unseen attacks and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves state-of-the-art results and exhibits generalization ability against unseen attacks.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。最成功的基于对抗性训练(AT)的防御技术可以达到对特定攻击的最佳健壮性，但不能很好地推广到看不见的攻击。另一种基于对抗性净化(AP)的有效防御技术可以增强泛化能力，但不能达到最优的健壮性。同时，这两种方法都有一个共同的缺陷，那就是标准精度下降。为了缓解这些问题，我们提出了一种新的流水线，称为对抗性净化训练(TOOP)，该流水线由两部分组成：通过随机变换的扰动破坏(RT)和通过对抗性损失微调(FT)的净化器模型。RT对于避免对已知攻击的过度学习导致对未知攻击的健壮性泛化至关重要，而FT对于提高健壮性是必不可少的。为了有效和可扩展地评估我们的方法，我们在CIFAR-10、CIFAR-100和ImageNette上进行了大量的实验，证明了我们的方法取得了最先进的结果，并表现出对不可见攻击的泛化能力。



## **22. From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**

从聊天机器人到网络钓鱼机器人？--防止使用ChatGPT、Google Bard和Claude创建的网络钓鱼诈骗 cs.CR

**SubmitDate**: 2024-03-10    [abs](http://arxiv.org/abs/2310.19181v2) [paper-pdf](http://arxiv.org/pdf/2310.19181v2)

**Authors**: Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs, i.e., ChatGPT (GPT 3.5 Turbo), GPT 4, Claude, and Bard, to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing websites and emails that can convincingly imitate well-known brands and also deploy a range of evasive tactics that are used to elude detection mechanisms employed by anti-phishing systems. These attacks can be generated using unmodified or "vanilla" versions of these LLMs without requiring any prior adversarial exploits such as jailbreaking. We evaluate the performance of the LLMs towards generating these attacks and find that they can also be utilized to create malicious prompts that, in turn, can be fed back to the model to generate phishing scams - thus massively reducing the prompt-engineering effort required by attackers to scale these threats. As a countermeasure, we build a BERT-based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content. Our model is transferable across all four commercial LLMs, attaining an average accuracy of 96% for phishing website prompts and 94% for phishing email prompts. We also disclose the vulnerabilities to the concerned LLMs, with Google acknowledging it as a severe issue. Our detection model is available for use at Hugging Face, as well as a ChatGPT Actions plugin.

摘要: 大型语言模型(LLM)的高级功能使其在从会话代理和内容创建到数据分析、研究和创新的各种应用程序中具有无价的价值。然而，它们的有效性和可访问性也使它们容易被滥用来生成恶意内容，包括网络钓鱼攻击。这项研究探索了使用四种流行的商用LLM，即ChatGPT(GPT 3.5 Turbo)、GPT 4、Claude和Bard，通过一系列恶意提示来生成功能性网络钓鱼攻击的可能性。我们发现，这些LLM可以生成钓鱼网站和电子邮件，这些网站和电子邮件可以令人信服地模仿知名品牌，还可以部署一系列规避策略，用于逃避反钓鱼系统使用的检测机制。这些攻击可以使用这些LLM的未修改或“普通”版本来生成，而不需要任何先前的对抗性利用，例如越狱。我们评估了LLMS在生成这些攻击方面的性能，发现它们还可以被用来创建恶意提示，进而可以反馈到模型以生成网络钓鱼诈骗-从而极大地减少了攻击者扩展这些威胁所需的提示工程工作。作为对策，我们构建了一个基于ERT的自动化检测工具，可以用于早期检测恶意提示，以防止LLMS生成钓鱼内容。我们的模型可以在所有四个商业LLM上传输，对于钓鱼网站提示的平均准确率为96%，对于钓鱼电子邮件提示的平均准确率为94%。我们还向相关的LLMS披露了漏洞，谷歌承认这是一个严重的问题。我们的检测模型可用于拥抱脸部，以及ChatGPT Actions插件。



## **23. Hard-label based Small Query Black-box Adversarial Attack**

基于硬标签的小查询黑盒对抗攻击 cs.LG

11 pages, 3 figures

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.06014v1) [paper-pdf](http://arxiv.org/pdf/2403.06014v1)

**Authors**: Jeonghwan Park, Paul Miller, Niall McLaughlin

**Abstract**: We consider the hard label based black box adversarial attack setting which solely observes predicted classes from the target model. Most of the attack methods in this setting suffer from impractical number of queries required to achieve a successful attack. One approach to tackle this drawback is utilising the adversarial transferability between white box surrogate models and black box target model. However, the majority of the methods adopting this approach are soft label based to take the full advantage of zeroth order optimisation. Unlike mainstream methods, we propose a new practical setting of hard label based attack with an optimisation process guided by a pretrained surrogate model. Experiments show the proposed method significantly improves the query efficiency of the hard label based black-box attack across various target model architectures. We find the proposed method achieves approximately 5 times higher attack success rate compared to the benchmarks, especially at the small query budgets as 100 and 250.

摘要: 我们考虑了基于硬标签的黑盒对抗攻击设置，它只观察目标模型中的预测类。此设置中的大多数攻击方法都需要大量不切实际的查询才能实现成功的攻击。解决这一缺陷的一种方法是利用白盒代理模型和黑盒目标模型之间的对抗性转移。然而，大多数采用这种方法的方法都是基于软标签的，以充分利用零阶优化的优势。与主流方法不同，我们提出了一种新的实用的基于硬标签的攻击设置，并在预先训练的代理模型的指导下进行优化。实验表明，该方法显著提高了基于硬标签的黑盒攻击在不同目标模型体系结构下的查询效率。我们发现，与基准测试相比，该方法的攻击成功率大约提高了5倍，特别是在查询预算较小的情况下，如100和250。



## **24. IOI: Invisible One-Iteration Adversarial Attack on No-Reference Image- and Video-Quality Metrics**

IOI：对无参考图像和视频质量度量的不可见一次迭代敌意攻击 eess.IV

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2403.05955v1) [paper-pdf](http://arxiv.org/pdf/2403.05955v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: No-reference image- and video-quality metrics are widely used in video processing benchmarks. The robustness of learning-based metrics under video attacks has not been widely studied. In addition to having success, attacks that can be employed in video processing benchmarks must be fast and imperceptible. This paper introduces an Invisible One-Iteration (IOI) adversarial attack on no reference image and video quality metrics. We compared our method alongside eight prior approaches using image and video datasets via objective and subjective tests. Our method exhibited superior visual quality across various attacked metric architectures while maintaining comparable attack success and speed. We made the code available on GitHub.

摘要: 无参考图像和视频质量度量被广泛应用于视频处理基准测试。基于学习的度量在视频攻击下的稳健性还没有得到广泛的研究。除了取得成功外，可用于视频处理基准的攻击必须快速且不可察觉。提出了一种针对无参考图像和视频质量指标的隐形单次迭代攻击方法。我们通过客观和主观测试，使用图像和视频数据集，将我们的方法与之前的八种方法进行了比较。我们的方法在各种受攻击的指标体系结构上显示出卓越的视觉质量，同时保持了相当的攻击成功率和速度。我们在GitHub上提供了代码。



## **25. SoK: Secure Human-centered Wireless Sensing**

SOK：安全的以人为中心的无线传感 cs.CR

**SubmitDate**: 2024-03-09    [abs](http://arxiv.org/abs/2211.12087v2) [paper-pdf](http://arxiv.org/pdf/2211.12087v2)

**Authors**: Wei Sun, Tingjun Chen, Neil Gong

**Abstract**: Human-centered wireless sensing (HCWS) aims to understand the fine-grained environment and activities of a human using the diverse wireless signals around him/her. While the sensed information about a human can be used for many good purposes such as enhancing life quality, an adversary can also abuse it to steal private information about the human (e.g., location and person's identity). However, the literature lacks a systematic understanding of the privacy vulnerabilities of wireless sensing and the defenses against them, resulting in the privacy-compromising HCWS design.   In this work, we aim to bridge this gap to achieve the vision of secure human-centered wireless sensing. First, we propose a signal processing pipeline to identify private information leakage and further understand the benefits and tradeoffs of wireless sensing-based inference attacks and defenses. Based on this framework, we present the taxonomy of existing inference attacks and defenses. As a result, we can identify the open challenges and gaps in achieving privacy-preserving human-centered wireless sensing in the era of machine learning and further propose directions for future research in this field.

摘要: 以人为中心的无线传感(HCWS)旨在利用周围各种无线信号来了解人类的细粒度环境和活动。虽然感知到的关于人的信息可以用于许多良好的目的，如提高生活质量，但攻击者也可以滥用这些信息来窃取关于人的私人信息(例如，位置和个人身份)。然而，文献对无线传感的隐私漏洞及其防御缺乏系统的了解，导致了隐私泄露的HCWS设计。在这项工作中，我们的目标是弥合这一差距，以实现以人为中心的安全无线传感的愿景。首先，我们提出了一种信号处理流水线来识别隐私信息泄露，并进一步了解基于无线传感的推理攻击和防御的利弊。基于这一框架，我们给出了现有推理攻击和防御的分类。因此，我们可以识别在机器学习时代实现以人为中心的隐私保护无线传感方面的开放挑战和差距，并进一步提出该领域未来研究的方向。



## **26. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

为最坏情况做好准备：一种基于学习的对抗性攻击用于ICP算法的弹性分析 cs.RO

8 pages (7 content, 1 reference). 5 figures, submitted to the IEEE  Robotics and Automation Letters (RA-L)

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05666v1) [paper-pdf](http://arxiv.org/pdf/2403.05666v1)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method to assess the resilience of the Iterative Closest Point (ICP) algorithm via deep-learning-based attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms prior to deployments is of utmost importance. The ICP algorithm has become the standard for lidar-based localization. However, the pose estimate it produces can be greatly affected by corruption in the measurements. Corruption can arise from a variety of scenarios such as occlusions, adverse weather, or mechanical issues in the sensor. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP empirically, our method focuses on finding the maximum possible ICP pose error using perturbation-based adversarial attacks. The proposed attack induces significant pose errors on ICP and outperforms baselines more than 88% of the time across a wide range of scenarios. As an example application, we demonstrate that our attack can be used to identify areas on a map where ICP is particularly vulnerable to corruption in the measurements.

摘要: 提出了一种基于深度学习的激光雷达点云攻击评估迭代最近点算法抗攻击能力的新方法。对于自主导航等安全关键型应用，在部署之前确保算法的弹性是至关重要的。该算法已成为激光雷达定位的标准算法。然而，它产生的姿势估计可能会受到测量中的干扰的很大影响。损坏可能由多种情况引起，例如堵塞、恶劣天气或传感器中的机械问题。不幸的是，比较方案的复杂性和迭代性使评估其对腐败的复原力具有挑战性。虽然已经有人努力创建具有挑战性的数据集和开发仿真来经验地评估ICP的弹性，但我们的方法专注于使用基于扰动的对抗性攻击来寻找最大可能的ICP姿态误差。所提出的攻击在ICP上引起显著的姿势误差，并且在广泛的场景中超过基线的时间超过88%。作为一个示例应用程序，我们演示了我们的攻击可以用来识别地图上的那些区域，在测量中，ICP特别容易受到腐败的影响。



## **27. Invariant Aggregator for Defending against Federated Backdoor Attacks**

用于防御联合后门攻击的不变聚合器 cs.LG

AISTATS 2024 camera-ready

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2210.01834v4) [paper-pdf](http://arxiv.org/pdf/2210.01834v4)

**Authors**: Xiaoyang Wang, Dimitrios Dimitriadis, Sanmi Koyejo, Shruti Tople

**Abstract**: Federated learning enables training high-utility models across several clients without directly sharing their private data. As a downside, the federated setting makes the model vulnerable to various adversarial attacks in the presence of malicious clients. Despite the theoretical and empirical success in defending against attacks that aim to degrade models' utility, defense against backdoor attacks that increase model accuracy on backdoor samples exclusively without hurting the utility on other samples remains challenging. To this end, we first analyze the failure modes of existing defenses over a flat loss landscape, which is common for well-designed neural networks such as Resnet (He et al., 2015) but is often overlooked by previous works. Then, we propose an invariant aggregator that redirects the aggregated update to invariant directions that are generally useful via selectively masking out the update elements that favor few and possibly malicious clients. Theoretical results suggest that our approach provably mitigates backdoor attacks and remains effective over flat loss landscapes. Empirical results on three datasets with different modalities and varying numbers of clients further demonstrate that our approach mitigates a broad class of backdoor attacks with a negligible cost on the model utility.

摘要: 联合学习允许在多个客户之间培训高实用模型，而无需直接共享他们的私人数据。缺点是，联合设置使模型在存在恶意客户端的情况下容易受到各种敌意攻击。尽管在防御旨在降低模型效用的攻击方面取得了理论和经验上的成功，但针对后门攻击的防御仍然具有挑战性，这种攻击只能提高后门样本的模型精度，而不会损害其他样本的效用。为此，我们首先分析了平坦损失情况下现有防御的故障模式，这在设计良好的神经网络如RESNET(他等人，2015)中很常见，但经常被以前的工作忽视。然后，我们提出了一个不变聚集器，它通过有选择地屏蔽有利于少数甚至可能是恶意客户端的更新元素，将聚合的更新重定向到通常有用的不变方向。理论结果表明，我们的方法可以有效地减少后门攻击，并且在平价损失场景下仍然有效。在三个具有不同模式和不同客户端数量的数据集上的实验结果进一步表明，我们的方法以可以忽略不计的模型效用代价缓解了广泛类别的后门攻击。



## **28. Can LLMs Follow Simple Rules?**

低收入国家能遵循简单的规则吗？ cs.AI

Project website: https://eecs.berkeley.edu/~normanmu/llm_rules;  revised content

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2311.04235v3) [paper-pdf](http://arxiv.org/pdf/2311.04235v3)

**Authors**: Norman Mu, Sarah Chen, Zifan Wang, Sizhe Chen, David Karamardian, Lulwa Aljeraisy, Basel Alomair, Dan Hendrycks, David Wagner

**Abstract**: As Large Language Models (LLMs) are deployed with increasing real-world responsibilities, it is important to be able to specify and constrain the behavior of these systems in a reliable manner. Model developers may wish to set explicit rules for the model, such as "do not generate abusive content", but these may be circumvented by jailbreaking techniques. Existing evaluations of adversarial attacks and defenses on LLMs generally require either expensive manual review or unreliable heuristic checks. To address this issue, we propose Rule-following Language Evaluation Scenarios (RuLES), a programmatic framework for measuring rule-following ability in LLMs. RuLES consists of 14 simple text scenarios in which the model is instructed to obey various rules while interacting with the user. Each scenario has a programmatic evaluation function to determine whether the model has broken any rules in a conversation. Our evaluations of proprietary and open models show that almost all current models struggle to follow scenario rules, even on straightforward test cases. We also demonstrate that simple optimization attacks suffice to significantly increase failure rates on test cases. We conclude by exploring two potential avenues for improvement: test-time steering and supervised fine-tuning.

摘要: 随着大型语言模型(LLM)的部署承担着越来越多的现实责任，能够以可靠的方式指定和约束这些系统的行为是很重要的。模型开发人员可能希望为模型设置明确的规则，例如“不要生成滥用内容”，但可以通过越狱技术绕过这些规则。现有的对抗性攻击和防御评估通常需要昂贵的人工审查或不可靠的启发式检查。为了解决这一问题，我们提出了规则遵循语言评估场景(Rules)，这是一个衡量LLMS中规则遵循能力的程序性框架。规则由14个简单的文本场景组成，在这些场景中，模型被指示在与用户交互时遵守各种规则。每个场景都有一个程序化的评估功能，以确定模型是否违反了对话中的任何规则。我们对专有和开放模型的评估表明，几乎所有当前的模型都难以遵循场景规则，即使在简单的测试用例上也是如此。我们还证明了简单的优化攻击足以显著增加测试用例的失败率。最后，我们探索了两个潜在的改进途径：测试时间控制和有监督的微调。



## **29. On Practicality of Using ARM TrustZone Trusted Execution Environment for Securing Programmable Logic Controllers**

利用ARM TrustZone可信执行环境保护可编程逻辑控制器的实用性研究 cs.CR

To appear at ACM AsiaCCS 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05448v1) [paper-pdf](http://arxiv.org/pdf/2403.05448v1)

**Authors**: Zhiang Li, Daisuke Mashima, Wen Shei Ong, Ertem Esiner, Zbigniew Kalbarczyk, Ee-Chien Chang

**Abstract**: Programmable logic controllers (PLCs) are crucial devices for implementing automated control in various industrial control systems (ICS), such as smart power grids, water treatment systems, manufacturing, and transportation systems. Owing to their importance, PLCs are often the target of cyber attackers that are aiming at disrupting the operation of ICS, including the nation's critical infrastructure, by compromising the integrity of control logic execution. While a wide range of cybersecurity solutions for ICS have been proposed, they cannot counter strong adversaries with a foothold on the PLC devices, which could manipulate memory, I/O interface, or PLC logic itself. These days, many ICS devices in the market, including PLCs, run on ARM-based processors, and there is a promising security technology called ARM TrustZone, to offer a Trusted Execution Environment (TEE) on embedded devices. Envisioning that such a hardware-assisted security feature becomes available for ICS devices in the near future, this paper investigates the application of the ARM TrustZone TEE technology for enhancing the security of PLC. Our aim is to evaluate the feasibility and practicality of the TEE-based PLCs through the proof-of-concept design and implementation using open-source software such as OP-TEE and OpenPLC. Our evaluation assesses the performance and resource consumption in real-world ICS configurations, and based on the results, we discuss bottlenecks in the OP-TEE secure OS towards a large-scale ICS and desired changes for its application on ICS devices. Our implementation is made available to public for further study and research.

摘要: 可编程控制器(PLC)是智能电网、水处理系统、制造业和交通运输系统等工业控制系统中实现自动化控制的关键器件。由于PLC的重要性，PLC经常成为网络攻击者的目标，他们的目标是通过损害控制逻辑执行的完整性来扰乱IC的运行，包括国家的关键基础设施。虽然已经提出了广泛的ICS网络安全解决方案，但它们无法对抗立足于PLC设备的强大对手，这些设备可以操纵内存、I/O接口或PLC逻辑本身。如今，市场上的许多ICS设备，包括PLC，都运行在基于ARM的处理器上，有一种很有前途的安全技术ARM TrustZone，它可以在嵌入式设备上提供一个可信执行环境(TEE)。鉴于这种硬件辅助的安全特性将在不久的将来应用于ICS设备，本文研究了ARM TrustZone TEE技术在增强PLC安全性方面的应用。我们的目的是通过使用开源软件OP-TEE和OpenPLC进行概念验证设计和实现，来评估基于TEE的PLC的可行性和实用性。我们的评估评估了实际ICS配置中的性能和资源消耗，并基于评估结果，讨论了操作安全操作系统向大规模ICS发展的瓶颈以及它在ICS设备上应用的期望变化。我们的实现可供公众进一步学习和研究。



## **30. EVD4UAV: An Altitude-Sensitive Benchmark to Evade Vehicle Detection in UAV**

EVD4无人机：一种用于躲避无人机车辆检测的高度敏感基准 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05422v1) [paper-pdf](http://arxiv.org/pdf/2403.05422v1)

**Authors**: Huiming Sun, Jiacheng Guo, Zibo Meng, Tianyun Zhang, Jianwu Fang, Yuewei Lin, Hongkai Yu

**Abstract**: Vehicle detection in Unmanned Aerial Vehicle (UAV) captured images has wide applications in aerial photography and remote sensing. There are many public benchmark datasets proposed for the vehicle detection and tracking in UAV images. Recent studies show that adding an adversarial patch on objects can fool the well-trained deep neural networks based object detectors, posing security concerns to the downstream tasks. However, the current public UAV datasets might ignore the diverse altitudes, vehicle attributes, fine-grained instance-level annotation in mostly side view with blurred vehicle roof, so none of them is good to study the adversarial patch based vehicle detection attack problem. In this paper, we propose a new dataset named EVD4UAV as an altitude-sensitive benchmark to evade vehicle detection in UAV with 6,284 images and 90,886 fine-grained annotated vehicles. The EVD4UAV dataset has diverse altitudes (50m, 70m, 90m), vehicle attributes (color, type), fine-grained annotation (horizontal and rotated bounding boxes, instance-level mask) in top view with clear vehicle roof. One white-box and two black-box patch based attack methods are implemented to attack three classic deep neural networks based object detectors on EVD4UAV. The experimental results show that these representative attack methods could not achieve the robust altitude-insensitive attack performance.

摘要: 无人机拍摄的图像中的车辆检测在航空摄影和遥感中有着广泛的应用。针对无人机图像中的车辆检测和跟踪，已经提出了许多公开的基准数据集。最近的研究表明，在对象上添加对抗性补丁可以欺骗训练有素的基于深度神经网络的对象检测器，从而给下游任务带来安全隐患。然而，目前公开的无人机数据集可能忽略了车顶模糊的侧视图中不同的高度、车辆属性、细粒度的实例级标注，因此不利于研究基于对抗性补丁的车辆检测攻击问题。在本文中，我们提出了一个新的数据集EVD4UAV作为高度敏感基准来逃避无人机中的车辆检测，该数据集包含6,284张图像和90,886辆细粒度标注的车辆。EVD4UAV数据集在俯视图中具有不同的高度(50m、70m、90m)、车辆属性(颜色、类型)、细粒度注释(水平和旋转的边界框、实例级遮罩)，并具有清晰的车顶。采用一种基于白盒和两种基于黑盒补丁的攻击方法，对EVD4无人机上三种经典的基于深度神经网络的目标探测器进行攻击。实验结果表明，这些具有代表性的攻击方法不能达到稳健的高度不敏感攻击性能。



## **31. The Impact of Quantization on the Robustness of Transformer-based Text Classifiers**

量化对基于Transformer的文本分类器稳健性的影响 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05365v1) [paper-pdf](http://arxiv.org/pdf/2403.05365v1)

**Authors**: Seyed Parsa Neshaei, Yasaman Boreshban, Gholamreza Ghassem-Sani, Seyed Abolghasem Mirroshandel

**Abstract**: Transformer-based models have made remarkable advancements in various NLP areas. Nevertheless, these models often exhibit vulnerabilities when confronted with adversarial attacks. In this paper, we explore the effect of quantization on the robustness of Transformer-based models. Quantization usually involves mapping a high-precision real number to a lower-precision value, aiming at reducing the size of the model at hand. To the best of our knowledge, this work is the first application of quantization on the robustness of NLP models. In our experiments, we evaluate the impact of quantization on BERT and DistilBERT models in text classification using SST-2, Emotion, and MR datasets. We also evaluate the performance of these models against TextFooler, PWWS, and PSO adversarial attacks. Our findings show that quantization significantly improves (by an average of 18.68%) the adversarial accuracy of the models. Furthermore, we compare the effect of quantization versus that of the adversarial training approach on robustness. Our experiments indicate that quantization increases the robustness of the model by 18.80% on average compared to adversarial training without imposing any extra computational overhead during training. Therefore, our results highlight the effectiveness of quantization in improving the robustness of NLP models.

摘要: 基于变压器的模型在各个NLP领域取得了显著的进步。然而，这些模型在面对对手攻击时往往表现出脆弱性。在本文中，我们探讨了量化对基于变压器的模型的稳健性的影响。量化通常涉及将高精度的实数映射到低精度的值，目的是减小手头模型的大小。据我们所知，这项工作是量化在NLP模型稳健性方面的首次应用。在我们的实验中，我们使用SST-2、情感和MR数据集评估了量化对BERT和DistilBERT模型在文本分类中的影响。我们还评估了这些模型对TextFooler、PWWS和PSO对手攻击的性能。我们的结果表明，量化显著提高了模型的对抗准确率(平均提高了18.68%)。此外，我们比较了量化和对抗性训练方法在稳健性方面的效果。我们的实验表明，与对抗性训练相比，量化使模型的健壮性平均提高了18.80%，而在训练过程中不增加任何额外的计算开销。因此，我们的结果突出了量化在提高NLP模型的稳健性方面的有效性。



## **32. Hide in Thicket: Generating Imperceptible and Rational Adversarial Perturbations on 3D Point Clouds**

隐藏在诡计中：在3D点云上生成不可察觉的和理性的对抗性扰动 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05247v1) [paper-pdf](http://arxiv.org/pdf/2403.05247v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Jindong Gu, Li Liu, Siyuan Liang, Bangyan He, Xiaochun Cao

**Abstract**: Adversarial attack methods based on point manipulation for 3D point cloud classification have revealed the fragility of 3D models, yet the adversarial examples they produce are easily perceived or defended against. The trade-off between the imperceptibility and adversarial strength leads most point attack methods to inevitably introduce easily detectable outlier points upon a successful attack. Another promising strategy, shape-based attack, can effectively eliminate outliers, but existing methods often suffer significant reductions in imperceptibility due to irrational deformations. We find that concealing deformation perturbations in areas insensitive to human eyes can achieve a better trade-off between imperceptibility and adversarial strength, specifically in parts of the object surface that are complex and exhibit drastic curvature changes. Therefore, we propose a novel shape-based adversarial attack method, HiT-ADV, which initially conducts a two-stage search for attack regions based on saliency and imperceptibility scores, and then adds deformation perturbations in each attack region using Gaussian kernel functions. Additionally, HiT-ADV is extendable to physical attack. We propose that by employing benign resampling and benign rigid transformations, we can further enhance physical adversarial strength with little sacrifice to imperceptibility. Extensive experiments have validated the superiority of our method in terms of adversarial and imperceptible properties in both digital and physical spaces. Our code is avaliable at: https://github.com/TRLou/HiT-ADV.

摘要: 基于点操作的三维点云分类对抗性攻击方法暴露了三维模型的脆弱性，但它们产生的对抗性实例很容易被感知或防御。隐蔽性和对抗性之间的权衡导致大多数点攻击方法在攻击成功后不可避免地引入容易检测到的离群点。另一种很有希望的策略是基于形状的攻击，它可以有效地消除离群点，但现有的方法由于不合理的变形往往会显著降低不可感知性。我们发现，在对人眼不敏感的区域隐藏变形扰动可以在不可感知性和对抗强度之间实现更好的权衡，特别是在物体表面复杂和曲率变化剧烈的部分。因此，我们提出了一种新的基于形状的对抗性攻击方法HIT-ADV，该方法首先根据显著分数和不可感知性分数进行两阶段的攻击区域搜索，然后使用高斯核函数在每个攻击区域添加变形扰动。此外，HIT-ADV可以扩展到物理攻击。我们提出，通过使用良性重采样和良性刚性变换，我们可以在几乎不牺牲不可感知性的情况下进一步增强物理对抗强度。广泛的实验已经验证了我们的方法在数字空间和物理空间中的对抗性和不可感知性方面的优越性。我们的代码可在：https://github.com/TRLou/HiT-ADV.上获得



## **33. Adversarial Sparse Teacher: Defense Against Distillation-Based Model Stealing Attacks Using Adversarial Examples**

对抗性稀疏教师：使用对抗性实例防御基于蒸馏的模型窃取攻击 cs.LG

12 pages, 3 figures, 6 tables

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05181v1) [paper-pdf](http://arxiv.org/pdf/2403.05181v1)

**Authors**: Eda Yilmaz, Hacer Yalim Keles

**Abstract**: Knowledge Distillation (KD) facilitates the transfer of discriminative capabilities from an advanced teacher model to a simpler student model, ensuring performance enhancement without compromising accuracy. It is also exploited for model stealing attacks, where adversaries use KD to mimic the functionality of a teacher model. Recent developments in this domain have been influenced by the Stingy Teacher model, which provided empirical analysis showing that sparse outputs can significantly degrade the performance of student models. Addressing the risk of intellectual property leakage, our work introduces an approach to train a teacher model that inherently protects its logits, influenced by the Nasty Teacher concept. Differing from existing methods, we incorporate sparse outputs of adversarial examples with standard training data to strengthen the teacher's defense against student distillation. Our approach carefully reduces the relative entropy between the original and adversarially perturbed outputs, allowing the model to produce adversarial logits with minimal impact on overall performance. The source codes will be made publicly available soon.

摘要: 知识蒸馏(KD)有助于将区分能力从高级教师模型转移到更简单的学生模型，确保在不影响准确性的情况下提高成绩。它还被利用来进行模型窃取攻击，攻击者使用KD来模仿教师模型的功能。这一领域的最新发展受到吝啬教师模型的影响，该模型提供的实证分析表明，稀疏的输出会显著降低学生模型的表现。为了应对知识产权泄露的风险，我们的工作引入了一种方法，以培养一种受肮脏的教师概念影响而内在地保护其逻辑的教师模式。与现有方法不同的是，我们将对抗性样本的稀疏输出与标准训练数据相结合，以加强教师对学生蒸馏的防御。我们的方法仔细地减少了原始输出和对抗性扰动输出之间的相对熵，允许模型在对整体性能影响最小的情况下生成对抗性逻辑。源代码很快就会公开。



## **34. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2310.07726v3) [paper-pdf](http://arxiv.org/pdf/2310.07726v3)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容(AIGC)越来越受欢迎，出现了许多新兴的商业服务和应用程序。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像和流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一且不可察觉的水印，用于服务验证和归属。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后绕过服务提供商的监管自由使用。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出战争，一种统一的方法论，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型来进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们对不同的数据集和嵌入设置进行了战争评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，战争的速度要快5050~11000倍。



## **35. Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models**

针对大型语言模型的间接提示注入攻击的基准测试和防御 cs.CL

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2312.14197v3) [paper-pdf](http://arxiv.org/pdf/2312.14197v3)

**Authors**: Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, Fangzhao Wu

**Abstract**: The integration of large language models (LLMs) with external content has enabled more up-to-date and wide-ranging applications of LLMs, such as Microsoft Copilot. However, this integration has also exposed LLMs to the risk of indirect prompt injection attacks, where an attacker can embed malicious instructions within external content, compromising LLM output and causing responses to deviate from user expectations. To investigate this important but underexplored issue, we introduce the first benchmark for indirect prompt injection attacks, named BIPIA, to evaluate the risk of such attacks. Based on the evaluation, our work makes a key analysis of the underlying reason for the success of the attack, namely the inability of LLMs to distinguish between instructions and external content and the absence of LLMs' awareness to not execute instructions within external content. Building upon this analysis, we develop two black-box methods based on prompt learning and a white-box defense method based on fine-tuning with adversarial training accordingly. Experimental results demonstrate that black-box defenses are highly effective in mitigating these attacks, while the white-box defense reduces the attack success rate to near-zero levels. Overall, our work systematically investigates indirect prompt injection attacks by introducing a benchmark, analyzing the underlying reason for the success of the attack, and developing an initial set of defenses.

摘要: 大型语言模型(LLM)与外部内容的集成使LLM能够更新、更广泛地应用，如Microsoft Copilot。然而，这种集成也使LLMS面临间接提示注入攻击的风险，攻击者可以在外部内容中嵌入恶意指令，损害LLM输出并导致响应偏离用户预期。为了研究这一重要但未被探索的问题，我们引入了第一个间接即时注入攻击基准，称为BIPIA，以评估此类攻击的风险。在评估的基础上，我们的工作重点分析了攻击成功的根本原因，即LLMS无法区分指令和外部内容，以及LLMS缺乏不执行外部内容中的指令的意识。在此基础上，我们提出了两种基于快速学习的黑盒防御方法和一种基于微调对抗性训练的白盒防御方法。实验结果表明，黑盒防御对于缓解这些攻击是非常有效的，而白盒防御将攻击成功率降低到接近于零的水平。总体而言，我们的工作通过引入基准、分析攻击成功的根本原因以及开发一套初始防御措施来系统地调查间接即时注入攻击。



## **36. Exploring the Adversarial Frontier: Quantifying Robustness via Adversarial Hypervolume**

探索对抗性前沿：通过对抗性超卷量化稳健性 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05100v1) [paper-pdf](http://arxiv.org/pdf/2403.05100v1)

**Authors**: Ping Guo, Cheng Gong, Xi Lin, Zhiyuan Yang, Qingfu Zhang

**Abstract**: The escalating threat of adversarial attacks on deep learning models, particularly in security-critical fields, has underscored the need for robust deep learning systems. Conventional robustness evaluations have relied on adversarial accuracy, which measures a model's performance under a specific perturbation intensity. However, this singular metric does not fully encapsulate the overall resilience of a model against varying degrees of perturbation. To address this gap, we propose a new metric termed adversarial hypervolume, assessing the robustness of deep learning models comprehensively over a range of perturbation intensities from a multi-objective optimization standpoint. This metric allows for an in-depth comparison of defense mechanisms and recognizes the trivial improvements in robustness afforded by less potent defensive strategies. Additionally, we adopt a novel training algorithm that enhances adversarial robustness uniformly across various perturbation intensities, in contrast to methods narrowly focused on optimizing adversarial accuracy. Our extensive empirical studies validate the effectiveness of the adversarial hypervolume metric, demonstrating its ability to reveal subtle differences in robustness that adversarial accuracy overlooks. This research contributes a new measure of robustness and establishes a standard for assessing and benchmarking the resilience of current and future defensive models against adversarial threats.

摘要: 对深度学习模型的敌意攻击的威胁不断升级，特别是在安全关键领域，这突显了需要强大的深度学习系统。传统的稳健性评估依赖于对抗精度，该精度衡量模型在特定扰动强度下的性能。然而，这种单一的度量并不能完全概括模型对不同程度扰动的总体弹性。为了弥补这一差距，我们提出了一种新的度量标准，称为对抗性超体积，从多目标优化的角度全面评估深度学习模型在一系列扰动强度下的稳健性。这一指标允许对防御机制进行深入比较，并认识到较弱的防御策略在健壮性方面的微小改进。此外，我们采用了一种新的训练算法，该算法在不同的扰动强度下均匀地增强了对抗的稳健性，而不是狭隘地专注于优化对抗的准确性。我们广泛的实证研究验证了对抗性超卷度量的有效性，证明了它能够揭示对抗性准确性忽略的稳健性的细微差异。这项研究提供了一种新的稳健性衡量标准，并为评估和基准当前和未来防御模型对对手威胁的弹性建立了标准。



## **37. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

用潜在对手训练防御不可预见的失败模式 cs.CR

**SubmitDate**: 2024-03-08    [abs](http://arxiv.org/abs/2403.05030v1) [paper-pdf](http://arxiv.org/pdf/2403.05030v1)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: AI systems sometimes exhibit harmful unintended behaviors post-deployment. This is often despite extensive diagnostics and debugging by developers. Minimizing risks from models is challenging because the attack surface is so large. It is not tractable to exhaustively search for inputs that may cause a model to fail. Red-teaming and adversarial training (AT) are commonly used to make AI systems more robust. However, they have not been sufficient to avoid many real-world failure modes that differ from the ones adversarially trained on. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without generating inputs that elicit them. LAT leverages the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. We use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 人工智能系统有时会在部署后表现出有害的意外行为。这通常是尽管开发人员进行了广泛的诊断和调试。将模型的风险降至最低是具有挑战性的，因为攻击面如此之大。要详尽地搜索可能导致模型失败的输入是不容易的。红队和对抗训练(AT)通常被用来使AI系统更健壮。然而，它们还不足以避免许多现实世界中的失败模式，这些模式与对手训练的模式不同。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不会生成引发漏洞的输入。随后，利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。我们使用LAT来删除特洛伊木马程序，并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，与AT相比，LAT通常可以提高对干净数据的稳健性和性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **38. Optimal Denial-of-Service Attacks Against Status Updating**

针对状态更新的最优拒绝服务攻击 cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04489v1) [paper-pdf](http://arxiv.org/pdf/2403.04489v1)

**Authors**: Saad Kriouile, Mohamad Assaad, Deniz Gündüz, Touraj Soleymani

**Abstract**: In this paper, we investigate denial-of-service attacks against status updating. The target system is modeled by a Markov chain and an unreliable wireless channel, and the performance of status updating in the target system is measured based on two metrics: age of information and age of incorrect information. Our objective is to devise optimal attack policies that strike a balance between the deterioration of the system's performance and the adversary's energy. We model the optimal problem as a Markov decision process and prove rigorously that the optimal jamming policy is a threshold-based policy under both metrics. In addition, we provide a low-complexity algorithm to obtain the optimal threshold value of the jamming policy. Our numerical results show that the networked system with the age-of-incorrect-information metric is less sensitive to jamming attacks than with the age-of-information metric. Index Terms-age of incorrect information, age of information, cyber-physical systems, status updating, remote monitoring.

摘要: 在本文中，我们研究针对状态更新的拒绝服务攻击。将目标系统建模为马尔可夫链和不可靠无线信道，并基于信息年龄和错误信息年龄两个度量来衡量目标系统的状态更新性能。我们的目标是设计最优的攻击策略，在系统性能的恶化和对手的能量之间取得平衡。我们将最优问题建模为马尔可夫决策过程，并严格证明了在两种度量下，最优干扰策略都是基于门限的策略。此外，我们还提出了一种低复杂度的算法来获取干扰策略的最优阈值。数值结果表明，具有错误信息年龄度量的网络系统对干扰攻击的敏感度低于具有信息年龄度量的网络系统。索引术语-不正确信息的年龄、信息的年龄、网络物理系统、状态更新、远程监控。



## **39. AdvQuNN: A Methodology for Analyzing the Adversarial Robustness of Quanvolutional Neural Networks**

AdvQuNN：一种分析量子卷积神经网络对抗健壮性的方法 quant-ph

7 pages, 6 figures

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.05596v1) [paper-pdf](http://arxiv.org/pdf/2403.05596v1)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Mohamed Bennai, Muhammad Shafique

**Abstract**: Recent advancements in quantum computing have led to the development of hybrid quantum neural networks (HQNNs) that employ a mixed set of quantum layers and classical layers, such as Quanvolutional Neural Networks (QuNNs). While several works have shown security threats of classical neural networks, such as adversarial attacks, their impact on QuNNs is still relatively unexplored. This work tackles this problem by designing AdvQuNN, a specialized methodology to investigate the robustness of HQNNs like QuNNs against adversarial attacks. It employs different types of Ansatzes as parametrized quantum circuits and different types of adversarial attacks. This study aims to rigorously assess the influence of quantum circuit architecture on the resilience of QuNN models, which opens up new pathways for enhancing the robustness of QuNNs and advancing the field of quantum cybersecurity. Our results show that, compared to classical convolutional networks, QuNNs achieve up to 60\% higher robustness for the MNIST and 40\% for FMNIST datasets.

摘要: 量子计算的最新进展导致了混合量子神经网络(HQNN)的发展，该混合量子神经网络使用了量子层和经典层的混合集合，例如量子卷积神经网络(QNNS)。虽然一些研究已经显示了经典神经网络的安全威胁，如对抗性攻击，但它们对量子神经网络的影响仍然相对未被探索。这项工作通过设计AdvQuNN来解决这个问题，AdvQuNN是一种专门的方法来研究像QuNN一样的HQNN对对手攻击的健壮性。它使用不同类型的Ansat作为参数化量子电路和不同类型的对抗性攻击。本研究旨在严格评估量子电路体系结构对量子网络模型弹性的影响，为提高量子网络的健壮性和推进量子网络安全领域开辟新的途径。结果表明，与经典卷积网络相比，量子神经网络对MNIST数据集的鲁棒性提高了60%，对FMNIST数据集的鲁棒性提高了40%。



## **40. Pilot Spoofing Attack on the Downlink of Cell-Free Massive MIMO: From the Perspective of Adversaries**

无蜂窝海量MIMO下行链路的飞行员欺骗攻击：对手视角 cs.IT

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2403.04435v1) [paper-pdf](http://arxiv.org/pdf/2403.04435v1)

**Authors**: Weiyang Xu, Yuan Zhang, Ruiguang Wang, Hien Quoc Ngo, Wei Xiang

**Abstract**: The channel hardening effect is less pronounced in the cell-free massive multiple-input multiple-output (mMIMO) system compared to its cellular counterpart, making it necessary to estimate the downlink effective channel gains to ensure decent performance. However, the downlink training inadvertently creates an opportunity for adversarial nodes to launch pilot spoofing attacks (PSAs). First, we demonstrate that adversarial distributed access points (APs) can severely degrade the achievable downlink rate. They achieve this by estimating their channels to users in the uplink training phase and then precoding and sending the same pilot sequences as those used by legitimate APs during the downlink training phase. Then, the impact of the downlink PSA is investigated by rigorously deriving a closed-form expression of the per-user achievable downlink rate. By employing the min-max criterion to optimize the power allocation coefficients, the maximum per-user achievable rate of downlink transmission is minimized from the perspective of adversarial APs. As an alternative to the downlink PSA, adversarial APs may opt to precode random interference during the downlink data transmission phase in order to disrupt legitimate communications. In this scenario, the achievable downlink rate is derived, and then power optimization algorithms are also developed. We present numerical results to showcase the detrimental impact of the downlink PSA and compare the effects of these two types of attacks.

摘要: 与蜂窝系统相比，无小区大规模多输入多输出(MMIMO)系统中的信道硬化效应不那么明显，因此有必要估计下行链路的有效信道增益以确保良好的性能。然而，下行训练无意中为敌对节点创造了发起试点欺骗攻击(PSA)的机会。首先，我们证明了敌意分布式接入点(AP)会严重降低可实现的下行链路速率。它们通过在上行链路训练阶段估计其对用户的信道，然后预编码并发送与合法AP在下行链路训练阶段使用的导频序列相同的导频序列来实现这一点。然后，通过严格推导每个用户可实现的下行链路速率的闭合形式表达式来研究下行链路PSA的影响。通过使用最小-最大准则来优化功率分配系数，从对抗性AP的角度最小化每用户可实现的最大下行传输速率。作为下行链路PSA的替代方案，敌意AP可以选择在下行链路数据传输阶段对随机干扰进行预编码，以便中断合法通信。在这种情况下，推导了可实现的下行链路速率，并开发了功率优化算法。我们给出了数值结果来展示下行PSA的有害影响，并比较了这两种类型的攻击的影响。



## **41. Evaluating the security of CRYSTALS-Dilithium in the quantum random oracle model**

在量子随机预言模型中评估晶体双锂的安全性 cs.CR

23 pages; v2: added description of CRYSTALS-Dilithium, improved  analysis of concrete parameters

**SubmitDate**: 2024-03-07    [abs](http://arxiv.org/abs/2312.16619v2) [paper-pdf](http://arxiv.org/pdf/2312.16619v2)

**Authors**: Kelsey A. Jackson, Carl A. Miller, Daochen Wang

**Abstract**: In the wake of recent progress on quantum computing hardware, the National Institute of Standards and Technology (NIST) is standardizing cryptographic protocols that are resistant to attacks by quantum adversaries. The primary digital signature scheme that NIST has chosen is CRYSTALS-Dilithium. The hardness of this scheme is based on the hardness of three computational problems: Module Learning with Errors (MLWE), Module Short Integer Solution (MSIS), and SelfTargetMSIS. MLWE and MSIS have been well-studied and are widely believed to be secure. However, SelfTargetMSIS is novel and, though classically as hard as MSIS, its quantum hardness is unclear. In this paper, we provide the first proof of the hardness of SelfTargetMSIS via a reduction from MLWE in the Quantum Random Oracle Model (QROM). Our proof uses recently developed techniques in quantum reprogramming and rewinding. A central part of our approach is a proof that a certain hash function, derived from the MSIS problem, is collapsing. From this approach, we deduce a new security proof for Dilithium under appropriate parameter settings. Compared to the previous work by Kiltz, Lyubashevsky, and Schaffner (EUROCRYPT 2018) that gave the only other rigorous security proof for a variant of Dilithium, our proof has the advantage of being applicable under the condition q = 1 mod 2n, where q denotes the modulus and n the dimension of the underlying algebraic ring. This condition is part of the original Dilithium proposal and is crucial for the efficient implementation of the scheme. We provide new secure parameter sets for Dilithium under the condition q = 1 mod 2n, finding that our public key size and signature size are about 2.9 times and 1.3 times larger, respectively, than those proposed by Kiltz et al. at the same security level.

摘要: 随着量子计算硬件的最新进展，美国国家标准与技术研究所(NIST)正在对能够抵抗量子对手攻击的密码协议进行标准化。NIST选择的主要数字签名方案是Crystal-Dilithium。该方案的难易程度基于三个计算问题的难易程度：带错误的模块学习(MLWE)、模块短整数解(MSIS)和自目标短整数解。MLWE和MSIS已经得到了很好的研究，并被广泛认为是安全的。然而，SelfTargetMSIS是新颖的，尽管经典上和MSIS一样难，但它的量子硬度尚不清楚。本文通过对量子随机Oracle模型(QROM)中MLWE的简化，首次证明了自目标MSIS的硬度。我们的证明使用了最近发展起来的量子重编程和倒带技术。我们方法的一个核心部分是证明从MSIS问题派生的某个散列函数正在崩溃。通过这种方法，我们在适当的参数设置下，给出了Dilithium的一个新的安全证明。与Kiltz，Lyubashevsky和Schaffner(Eurocrypt 2018)之前的工作相比，我们的证明具有在q=1mod 2n的条件下适用的优点，其中q表示基础代数环的模，n表示基础代数环的维度。这一条件是最初的Dilithium提议的一部分，对该计划的有效实施至关重要。在Q=1 mod 2n的条件下，我们给出了Dilithium的新的安全参数集，发现我们的公钥长度和签名长度分别是Kiltz等人提出的安全参数集的2.9倍和1.3倍。在相同的安全级别。



## **42. Multi-Agent Reinforcement Learning for Assessing False-Data Injection Attacks on Transportation Networks**

基于多智能体强化学习的交通网络虚假数据注入攻击评估 cs.AI

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2312.14625v2) [paper-pdf](http://arxiv.org/pdf/2312.14625v2)

**Authors**: Taha Eghtesad, Sirui Li, Yevgeniy Vorobeychik, Aron Laszka

**Abstract**: The increasing reliance of drivers on navigation applications has made transportation networks more susceptible to data-manipulation attacks by malicious actors. Adversaries may exploit vulnerabilities in the data collection or processing of navigation services to inject false information, and to thus interfere with the drivers' route selection. Such attacks can significantly increase traffic congestions, resulting in substantial waste of time and resources, and may even disrupt essential services that rely on road networks. To assess the threat posed by such attacks, we introduce a computational framework to find worst-case data-injection attacks against transportation networks. First, we devise an adversarial model with a threat actor who can manipulate drivers by increasing the travel times that they perceive on certain roads. Then, we employ hierarchical multi-agent reinforcement learning to find an approximate optimal adversarial strategy for data manipulation. We demonstrate the applicability of our approach through simulating attacks on the Sioux Falls, ND network topology.

摘要: 司机越来越依赖导航应用程序，这使得交通网络更容易受到恶意行为者的数据操纵攻击。攻击者可能会利用导航服务的数据收集或处理中的漏洞来注入虚假信息，从而干扰司机的路线选择。此类攻击可能会显著加剧交通拥堵，导致大量时间和资源的浪费，甚至可能扰乱依赖道路网络的基本服务。为了评估这类攻击造成的威胁，我们引入了一个计算框架来发现针对交通网络的最坏情况下的数据注入攻击。首先，我们设计了一个带有威胁参与者的对抗性模型，该威胁参与者可以通过增加司机在某些道路上感知的旅行时间来操纵司机。然后，我们使用分层多智能体强化学习来寻找数据操作的近似最优对抗策略。通过模拟对苏福尔斯网络拓扑结构的攻击，验证了该方法的适用性。



## **43. Improving Adversarial Training using Vulnerability-Aware Perturbation Budget**

利用脆弱性感知扰动预算改进对抗性训练 cs.LG

19 pages, 2 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.04070v1) [paper-pdf](http://arxiv.org/pdf/2403.04070v1)

**Authors**: Olukorede Fakorede, Modeste Atsague, Jin Tian

**Abstract**: Adversarial Training (AT) effectively improves the robustness of Deep Neural Networks (DNNs) to adversarial attacks. Generally, AT involves training DNN models with adversarial examples obtained within a pre-defined, fixed perturbation bound. Notably, individual natural examples from which these adversarial examples are crafted exhibit varying degrees of intrinsic vulnerabilities, and as such, crafting adversarial examples with fixed perturbation radius for all instances may not sufficiently unleash the potency of AT. Motivated by this observation, we propose two simple, computationally cheap vulnerability-aware reweighting functions for assigning perturbation bounds to adversarial examples used for AT, named Margin-Weighted Perturbation Budget (MWPB) and Standard-Deviation-Weighted Perturbation Budget (SDWPB). The proposed methods assign perturbation radii to individual adversarial samples based on the vulnerability of their corresponding natural examples. Experimental results show that the proposed methods yield genuine improvements in the robustness of AT algorithms against various adversarial attacks.

摘要: 对抗训练(AT)有效地提高了深度神经网络(DNN)对对抗攻击的稳健性。通常，AT涉及用在预定义的固定扰动范围内获得的对抗性样本来训练DNN模型。值得注意的是，制作这些对抗性例子的个别自然例子表现出不同程度的内在脆弱性，因此，为所有实例制作具有固定扰动半径的对抗性例子可能不能充分释放AT的效力。基于这一观察结果，我们提出了两个简单的、计算上廉价的脆弱性感知重加权函数，用于为AT中的对抗性例子分配扰动界，分别称为差值加权扰动预算(MWPB)和标准差加权扰动预算(SDWPB)。所提出的方法根据单个对抗性样本的自然样本的脆弱性为其分配扰动半径。实验结果表明，该方法确实提高了AT算法对各种敌意攻击的稳健性。



## **44. Improving Adversarial Attacks on Latent Diffusion Model**

基于潜在扩散模型的对抗性攻击改进 cs.CV

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2310.04687v3) [paper-pdf](http://arxiv.org/pdf/2310.04687v3)

**Authors**: Boyang Zheng, Chumeng Liang, Xiaoyu Wu, Yan Liu

**Abstract**: Adversarial attacks on Latent Diffusion Model (LDM), the state-of-the-art image generative model, have been adopted as effective protection against malicious finetuning of LDM on unauthorized images. We show that these attacks add an extra error to the score function of adversarial examples predicted by LDM. LDM finetuned on these adversarial examples learns to lower the error by a bias, from which the model is attacked and predicts the score function with biases.   Based on the dynamics, we propose to improve the adversarial attack on LDM by Attacking with Consistent score-function Errors (ACE). ACE unifies the pattern of the extra error added to the predicted score function. This induces the finetuned LDM to learn the same pattern as a bias in predicting the score function. We then introduce a well-crafted pattern to improve the attack. Our method outperforms state-of-the-art methods in adversarial attacks on LDM.

摘要: 针对当前最先进的图像生成模型--潜在扩散模型(LDM)的敌意攻击已被用作对未经授权的图像进行恶意微调的有效保护。我们证明了这些攻击给LDM预测的对抗性例子的得分函数增加了额外的误差。在这些对抗性例子上精调的LDM学习通过偏差来降低误差，由此对模型进行攻击并预测带有偏差的得分函数。在此基础上，提出了利用一致得分函数错误(ACE)攻击来提高对LDM的对抗性攻击。ACE统一了添加到预测得分函数的额外误差的模式。这导致精调的LDM在预测得分函数时学习与偏差相同的模式。然后，我们引入一个精心设计的模式来改进攻击。在对LDM的对抗性攻击中，我们的方法优于最先进的方法。



## **45. A Survey on Adversarial Contention Resolution**

对抗性争议解决机制研究综述 cs.DC

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03876v1) [paper-pdf](http://arxiv.org/pdf/2403.03876v1)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.

摘要: 争用解决方案解决了协调多个进程对共享资源(如内存、磁盘存储或通信通道)的访问的挑战。争用解决最初是由数据库系统和总线网络中的挑战推动的，尽管经历了几十年的技术变革，但它作为资源共享的一个重要抽象概念一直存在。在这里，我们回顾了关于解决最坏情况争用的文献，在这种情况下，进程的数量和每个进程可能开始寻求访问资源的时间由对手决定。我们重点介绍争用解决方案的演变，其中新的关注点--如安全性、服务质量和能源效率--是由现代系统驱动的。这些努力使人们深入了解了随机化和确定性方法的局限性，以及不同模型假设的影响，如全球时钟同步、处理器数量的知识、访问尝试的反馈以及对共享资源可用性的攻击。



## **46. Effect of Ambient-Intrinsic Dimension Gap on Adversarial Vulnerability**

环境-本征维度差距对对手脆弱性的影响 cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03967v1) [paper-pdf](http://arxiv.org/pdf/2403.03967v1)

**Authors**: Rajdeep Haldar, Yue Xing, Qifan Song

**Abstract**: The existence of adversarial attacks on machine learning models imperceptible to a human is still quite a mystery from a theoretical perspective. In this work, we introduce two notions of adversarial attacks: natural or on-manifold attacks, which are perceptible by a human/oracle, and unnatural or off-manifold attacks, which are not. We argue that the existence of the off-manifold attacks is a natural consequence of the dimension gap between the intrinsic and ambient dimensions of the data. For 2-layer ReLU networks, we prove that even though the dimension gap does not affect generalization performance on samples drawn from the observed data space, it makes the clean-trained model more vulnerable to adversarial perturbations in the off-manifold direction of the data space. Our main results provide an explicit relationship between the $\ell_2,\ell_{\infty}$ attack strength of the on/off-manifold attack and the dimension gap.

摘要: 从理论上讲，对人类无法察觉的机器学习模型存在敌意攻击仍然是一个相当神秘的问题。在这项工作中，我们引入了两个对抗性攻击的概念：人类/先知可以感知的自然或流形上的攻击，以及不可察觉的非自然或非流形攻击。我们认为，非流形攻击的存在是数据的内在维度和环境维度之间存在维度差距的自然结果。对于两层RELU网络，我们证明了尽管维度间隙不影响对来自观测数据空间的样本的泛化性能，但它使得干净训练的模型更容易受到数据空间非流形方向上的对抗性扰动。我们的主要结果提供了On/Off流形攻击的攻击强度与维度间隙之间的显式关系。



## **47. Neural Exec: Learning (and Learning from) Execution Triggers for Prompt Injection Attacks**

NeuroExec：学习(和学习)快速注入攻击的执行触发器 cs.CR

v0.1

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03792v1) [paper-pdf](http://arxiv.org/pdf/2403.03792v1)

**Authors**: Dario Pasquini, Martin Strohmeier, Carmela Troncoso

**Abstract**: We introduce a new family of prompt injection attacks, termed Neural Exec. Unlike known attacks that rely on handcrafted strings (e.g., "Ignore previous instructions and..."), we show that it is possible to conceptualize the creation of execution triggers as a differentiable search problem and use learning-based methods to autonomously generate them.   Our results demonstrate that a motivated adversary can forge triggers that are not only drastically more effective than current handcrafted ones but also exhibit inherent flexibility in shape, properties, and functionality. In this direction, we show that an attacker can design and generate Neural Execs capable of persisting through multi-stage preprocessing pipelines, such as in the case of Retrieval-Augmented Generation (RAG)-based applications. More critically, our findings show that attackers can produce triggers that deviate markedly in form and shape from any known attack, sidestepping existing blacklist-based detection and sanitation approaches.

摘要: 我们介绍了一类新的快速注入攻击，称为神经执行攻击。与依赖手工创建的字符串(例如，“忽略先前的指令和...”)的已知攻击不同，我们展示了将创建执行触发器概念化为可区分的搜索问题并使用基于学习的方法自主生成它们是可能的。我们的结果表明，有动机的对手可以伪造触发器，不仅比目前手工制作的触发器有效得多，而且在形状、属性和功能上表现出固有的灵活性。在这个方向上，我们展示了攻击者可以设计和生成能够在多阶段预处理管道中持久存在的神经Execs，例如在基于检索-增强生成(RAG)的应用程序的情况下。更关键的是，我们的发现表明，攻击者可以产生在形式和形状上与任何已知攻击显著偏离的触发器，绕过现有的基于黑名单的检测和卫生方法。



## **48. PPTC-R benchmark: Towards Evaluating the Robustness of Large Language Models for PowerPoint Task Completion**

PPTC-R基准：评估用于PowerPoint任务完成的大型语言模型的健壮性 cs.CL

LLM evaluation, Multi-turn, Multi-language, Multi-modal benchmark

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03788v1) [paper-pdf](http://arxiv.org/pdf/2403.03788v1)

**Authors**: Zekai Zhang, Yiduo Guo, Yaobo Liang, Dongyan Zhao, Nan Duan

**Abstract**: The growing dependence on Large Language Models (LLMs) for finishing user instructions necessitates a comprehensive understanding of their robustness to complex task completion in real-world situations. To address this critical need, we propose the PowerPoint Task Completion Robustness benchmark (PPTC-R) to measure LLMs' robustness to the user PPT task instruction and software version. Specifically, we construct adversarial user instructions by attacking user instructions at sentence, semantic, and multi-language levels. To assess the robustness of Language Models to software versions, we vary the number of provided APIs to simulate both the newest version and earlier version settings. Subsequently, we test 3 closed-source and 4 open-source LLMs using a benchmark that incorporates these robustness settings, aiming to evaluate how deviations impact LLMs' API calls for task completion. We find that GPT-4 exhibits the highest performance and strong robustness in our benchmark, particularly in the version update and the multilingual settings. However, we find that all LLMs lose their robustness when confronted with multiple challenges (e.g., multi-turn) simultaneously, leading to significant performance drops. We further analyze the robustness behavior and error reasons of LLMs in our benchmark, which provide valuable insights for researchers to understand the LLM's robustness in task completion and develop more robust LLMs and agents. We release the code and data at \url{https://github.com/ZekaiGalaxy/PPTCR}.

摘要: 越来越多地依赖大型语言模型(LLM)来完成用户指令，这就需要全面了解它们在现实世界中完成复杂任务时的健壮性。为了解决这一关键需求，我们提出了PowerPoint任务完成健壮性基准(PPTC-R)来测量LLMS对用户PPT任务指令和软件版本的健壮性。具体地说，我们通过在句子、语义和多语言级别攻击用户指令来构建对抗性用户指令。为了评估语言模型对软件版本的稳健性，我们改变了提供的API的数量，以模拟最新版本和较早版本的设置。随后，我们使用结合了这些健壮性设置的基准测试了3个封闭源代码LLMS和4个开放源代码LLMS，旨在评估偏差如何影响LLMS完成任务的API调用。我们发现GPT-4在我们的基准测试中表现出了最高的性能和强大的健壮性，特别是在版本更新和多语言设置方面。然而，我们发现，当同时面对多个挑战(例如，多回合)时，所有的LLM都失去了它们的健壮性，导致性能显著下降。我们进一步分析了LLM在基准测试中的健壮性行为和错误原因，这为研究人员理解LLM在任务完成时的健壮性以及开发更健壮的LLM和代理提供了有价值的见解。我们将代码和数据发布到\url{https://github.com/ZekaiGalaxy/PPTCR}.



## **49. Verification of Neural Networks' Global Robustness**

神经网络的全局健壮性验证 cs.LG

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2402.19322v2) [paper-pdf](http://arxiv.org/pdf/2402.19322v2)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstract**: Neural networks are successful in various applications but are also susceptible to adversarial attacks. To show the safety of network classifiers, many verifiers have been introduced to reason about the local robustness of a given input to a given perturbation. While successful, local robustness cannot generalize to unseen inputs. Several works analyze global robustness properties, however, neither can provide a precise guarantee about the cases where a network classifier does not change its classification. In this work, we propose a new global robustness property for classifiers aiming at finding the minimal globally robust bound, which naturally extends the popular local robustness property for classifiers. We introduce VHAGaR, an anytime verifier for computing this bound. VHAGaR relies on three main ideas: encoding the problem as a mixed-integer programming and pruning the search space by identifying dependencies stemming from the perturbation or the network's computation and generalizing adversarial attacks to unknown inputs. We evaluate VHAGaR on several datasets and classifiers and show that, given a three hour timeout, the average gap between the lower and upper bound on the minimal globally robust bound computed by VHAGaR is 1.9, while the gap of an existing global robustness verifier is 154.7. Moreover, VHAGaR is 130.6x faster than this verifier. Our results further indicate that leveraging dependencies and adversarial attacks makes VHAGaR 78.6x faster.

摘要: 神经网络在各种应用中都很成功，但也容易受到对抗性攻击。为了表明网络分类器的安全性，已经引入了许多验证器来推理给定输入对给定扰动的局部稳健性。虽然取得了成功，但局部稳健性不能推广到看不见的输入。然而，一些工作分析了全局健壮性，但都不能提供关于网络分类器不改变其分类的情况的精确保证。在这项工作中，我们提出了一种新的分类器的全局稳健性，旨在寻找最小的全局稳健界，这自然地扩展了流行的分类器的局部稳健性。我们介绍了VHAGaR，一个计算这个界的随时验证器。VHAGaR依赖于三个主要思想：将问题编码为混合整数规划，通过识别源于扰动或网络计算的依赖来削减搜索空间，以及将敌意攻击推广到未知输入。我们在几个数据集和分类器上对VHAGaR进行了评估，结果表明，在超时3小时的情况下，VHAGaR计算的最小全局健壮界的上下界之间的平均差距为1.9%，而现有的全局健壮性验证器的差距为154.7。此外，VHAGaR比该验证器快130.6倍。我们的结果进一步表明，利用依赖关系和对抗性攻击使VHAGaR的速度提高了78.6倍。



## **50. Simplified PCNet with Robustness**

具有健壮性的简化PCNet cs.LG

10 pages, 3 figures

**SubmitDate**: 2024-03-06    [abs](http://arxiv.org/abs/2403.03676v1) [paper-pdf](http://arxiv.org/pdf/2403.03676v1)

**Authors**: Bingheng Li, Xuanting Xie, Haoxiang Lei, Ruiyi Fang, Zhao Kang

**Abstract**: Graph Neural Networks (GNNs) have garnered significant attention for their success in learning the representation of homophilic or heterophilic graphs. However, they cannot generalize well to real-world graphs with different levels of homophily. In response, the Possion-Charlier Network (PCNet) \cite{li2024pc}, the previous work, allows graph representation to be learned from heterophily to homophily. Although PCNet alleviates the heterophily issue, there remain some challenges in further improving the efficacy and efficiency. In this paper, we simplify PCNet and enhance its robustness. We first extend the filter order to continuous values and reduce its parameters. Two variants with adaptive neighborhood sizes are implemented. Theoretical analysis shows our model's robustness to graph structure perturbations or adversarial attacks. We validate our approach through semi-supervised learning tasks on various datasets representing both homophilic and heterophilic graphs.

摘要: 图神经网络(GNN)因其在学习同亲图或异亲图的表示方面的成功而受到极大的关注。然而，它们不能很好地推广到具有不同同质性水平的真实世界的图。作为回应，Possion-Charlier Network(PCNet)引用了以前的工作{li2024pc}，允许从异形到同形学习图表示。虽然PCNet缓解了异质性问题，但在进一步提高疗效和效率方面仍存在一些挑战。在本文中，我们简化了PCNet，增强了它的健壮性。我们首先将滤波阶扩展到连续值，并对其参数进行降阶。实现了两种具有自适应邻域大小的变体。理论分析表明，该模型对图结构扰动或敌意攻击具有较强的稳健性。我们通过在不同数据集上的半监督学习任务来验证我们的方法，这些数据集既代表同嗜图，也代表异嗜图。



