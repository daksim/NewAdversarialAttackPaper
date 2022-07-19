# Latest Adversarial Attack Papers
**update at 2022-07-20 06:31:27**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Adversarial Pixel Restoration as a Pretext Task for Transferable Perturbations**

对抗性像素恢复作为可转移扰动的借口任务 cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08803v1)

**Authors**: Hashmat Shadab Malik, Shahina K Kunhimon, Muzammal Naseer, Salman Khan, Fahad Shahbaz Khan

**Abstracts**: Transferable adversarial attacks optimize adversaries from a pretrained surrogate model and known label space to fool the unknown black-box models. Therefore, these attacks are restricted by the availability of an effective surrogate model. In this work, we relax this assumption and propose Adversarial Pixel Restoration as a self-supervised alternative to train an effective surrogate model from scratch under the condition of no labels and few data samples. Our training approach is based on a min-max objective which reduces overfitting via an adversarial objective and thus optimizes for a more generalizable surrogate model. Our proposed attack is complimentary to our adversarial pixel restoration and is independent of any task specific objective as it can be launched in a self-supervised manner. We successfully demonstrate the adversarial transferability of our approach to Vision Transformers as well as Convolutional Neural Networks for the tasks of classification, object detection, and video segmentation. Our codes & pre-trained surrogate models are available at: https://github.com/HashmatShadab/APR

摘要: 可转移对抗性攻击从预先训练的代理模型和已知标签空间中优化对手，以愚弄未知的黑盒模型。因此，这些攻击受到有效代理模型可用性的限制。在这项工作中，我们放松了这一假设，提出了对抗性像素复原作为一种自我监督的替代方案，在没有标签和数据样本的情况下，从零开始训练一个有效的代理模型。我们的训练方法基于最小-最大目标，该目标减少了通过对抗性目标的过度拟合，从而优化了更具通用性的代理模型。我们建议的攻击是对我们的对抗性像素恢复的补充，并且独立于任何特定任务的目标，因为它可以以自我监督的方式发起。我们成功地展示了我们的视觉变形方法以及卷积神经网络方法在分类、目标检测和视频分割任务中的对抗性可转移性。我们的代码和预先培训的代孕模型可在以下网址获得：https://github.com/HashmatShadab/APR



## **2. Are Vision Transformers Robust to Patch Perturbations?**

视觉变形器对补丁扰动有健壮性吗？ cs.CV

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2111.10659v2)

**Authors**: Jindong Gu, Volker Tresp, Yao Qin

**Abstracts**: Recent advances in Vision Transformer (ViT) have demonstrated its impressive performance in image classification, which makes it a promising alternative to Convolutional Neural Network (CNN). Unlike CNNs, ViT represents an input image as a sequence of image patches. The patch-based input image representation makes the following question interesting: How does ViT perform when individual input image patches are perturbed with natural corruptions or adversarial perturbations, compared to CNNs? In this work, we study the robustness of ViT to patch-wise perturbations. Surprisingly, we find that ViTs are more robust to naturally corrupted patches than CNNs, whereas they are more vulnerable to adversarial patches. Furthermore, we discover that the attention mechanism greatly affects the robustness of vision transformers. Specifically, the attention module can help improve the robustness of ViT by effectively ignoring natural corrupted patches. However, when ViTs are attacked by an adversary, the attention mechanism can be easily fooled to focus more on the adversarially perturbed patches and cause a mistake. Based on our analysis, we propose a simple temperature-scaling based method to improve the robustness of ViT against adversarial patches. Extensive qualitative and quantitative experiments are performed to support our findings, understanding, and improvement of ViT robustness to patch-wise perturbations across a set of transformer-based architectures.

摘要: 近年来，视觉转换器(VIT)在图像分类中表现出了令人印象深刻的性能，这使得它成为卷积神经网络(CNN)的一种有前途的替代方案。与CNN不同，VIT将输入图像表示为一系列图像补丁。基于块的输入图像表示使得以下问题变得有趣：与CNN相比，当单个输入图像块受到自然破坏或敌意扰动时，VIT的性能如何？在这项工作中，我们研究了VIT对面片扰动的稳健性。令人惊讶的是，我们发现VITS比CNN对自然损坏的补丁更健壮，而它们更容易受到对手补丁的攻击。此外，我们发现注意机制对视觉转换器的稳健性有很大的影响。具体地说，注意模块可以通过有效地忽略自然损坏的补丁来帮助提高VIT的健壮性。然而，当VITS受到对手的攻击时，注意力机制很容易被愚弄，使其更多地集中在对手扰乱的补丁上，从而导致错误。基于我们的分析，我们提出了一种简单的基于温度缩放的方法来提高VIT对恶意补丁的健壮性。进行了大量的定性和定量实验，以支持我们的发现、理解和提高VIT对一组基于变压器的体系结构的补丁扰动的稳健性。



## **3. Authentication Attacks on Projection-based Cancelable Biometric Schemes**

对基于投影的可取消生物特征识别方案的认证攻击 cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2110.15163v6)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **4. Detection of Poisoning Attacks with Anomaly Detection in Federated Learning for Healthcare Applications: A Machine Learning Approach**

医疗联合学习中异常检测的中毒攻击检测：一种机器学习方法 cs.LG

We will updated this article soon

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08486v1)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstracts**: The application of Federated Learning (FL) is steadily increasing, especially in privacy-aware applications, such as healthcare. However, its applications have been limited by security concerns due to various adversarial attacks, such as poisoning attacks (model and data poisoning). Such attacks attempt to poison the local models and data to manipulate the global models in order to obtain undue benefits and malicious use. Traditional methods of data auditing to mitigate poisoning attacks find their limited applications in FL because the edge devices never share their raw data directly due to privacy concerns, and are globally distributed with no insight into their training data. Thereafter, it is challenging to develop appropriate strategies to address such attacks and minimize their impact on the global model in federated learning. In order to address such challenges in FL, we proposed a novel framework to detect poisoning attacks using deep neural networks and support vector machines, in the form of anomaly without acquiring any direct access or information about the underlying training data of local edge devices. We illustrate and evaluate the proposed framework using different state of art poisoning attacks for two different healthcare applications: Electrocardiograph classification and human activity recognition. Our experimental analysis shows that the proposed method can efficiently detect poisoning attacks and can remove the identified poisoned updated from the global aggregation. Thereafter can increase the performance of the federated global.

摘要: 联合学习(FL)的应用正在稳步增加，特别是在隐私感知应用中，如医疗保健。然而，由于各种对抗性攻击，如中毒攻击(模型中毒和数据中毒)，其应用受到了安全方面的考虑。这类攻击试图毒化本地模型和数据以操纵全局模型，以获取不正当的利益和恶意使用。缓解中毒攻击的传统数据审计方法在FL中的应用有限，因为由于隐私问题，边缘设备从未直接共享它们的原始数据，并且是全球分布的，无法洞察它们的训练数据。此后，制定适当的策略来应对此类攻击并将其对联合学习中的全球模型的影响降至最低是具有挑战性的。为了解决FL中的这种挑战，我们提出了一种新的框架，使用深度神经网络和支持向量机来检测异常形式的中毒攻击，而不需要获取任何关于本地边缘设备的底层训练数据的直接访问或信息。我们使用针对两种不同医疗应用的不同技术水平的中毒攻击来说明和评估所提出的框架：心电图机分类和人类活动识别。实验分析表明，该方法能够有效地检测中毒攻击，并能从全局聚集中剔除已识别的中毒更新。之后可以提高联合全局的性能。



## **5. Towards Automated Classification of Attackers' TTPs by combining NLP with ML Techniques**

结合NLP和ML技术实现攻击者TTP的自动分类 cs.CR

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2207.08478v1)

**Authors**: Clemens Sauerwein, Alexander Pfohl

**Abstracts**: The increasingly sophisticated and growing number of threat actors along with the sheer speed at which cyber attacks unfold, make timely identification of attacks imperative to an organisations' security. Consequently, persons responsible for security employ a large variety of information sources concerning emerging attacks, attackers' course of actions or indicators of compromise. However, a vast amount of the needed security information is available in unstructured textual form, which complicates the automated and timely extraction of attackers' Tactics, Techniques and Procedures (TTPs). In order to address this problem we systematically evaluate and compare different Natural Language Processing (NLP) and machine learning techniques used for security information extraction in research. Based on our investigations we propose a data processing pipeline that automatically classifies unstructured text according to attackers' tactics and techniques derived from a knowledge base of adversary tactics, techniques and procedures.

摘要: 威胁因素的日益复杂和数量不断增加，以及网络攻击展开的速度之快，使得及时识别攻击对组织的安全至关重要。因此，负责安全的人员使用大量关于新出现的攻击、攻击者的行动过程或妥协迹象的信息来源。然而，大量所需的安全信息是以非结构化文本形式提供的，这使得自动、及时地提取攻击者的战术、技术和程序(TTP)变得复杂。为了解决这个问题，我们系统地评估和比较了不同的自然语言处理(NLP)和机器学习技术在安全信息提取中的应用研究。基于我们的研究，我们提出了一种数据处理流水线，它根据攻击者的战术和技术来自动对非结构化文本进行分类，这些策略和技术是从对手的战术、技术和过程知识库中得出的。



## **6. A Perturbation-Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

一种用于评估光流稳健性的扰动约束对抗性攻击 cs.CV

Accepted at the European Conference on Computer Vision (ECCV) 2022

**SubmitDate**: 2022-07-18    [paper-pdf](http://arxiv.org/pdf/2203.13214v2)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods focus on real-world attacking scenarios rather than a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation-Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments demonstrate PCFA's applicability in white- and black-box settings, and show it finds stronger adversarial samples than previous attacks. Based on these strong samples, we provide the first joint ranking of optical flow methods considering both prediction quality and adversarial robustness, which reveals state-of-the-art methods to be particularly vulnerable. Code is available at https://github.com/cv-stuttgart/PCFA.

摘要: 最近的光流方法几乎完全是根据准确性来判断的，而它们的稳健性往往被忽视。尽管对抗性攻击为执行这种分析提供了一个有用的工具，但目前对光流方法的攻击主要集中在真实世界的攻击场景上，而不是最坏情况下的健壮性评估。因此，在这项工作中，我们提出了一种新的对抗性攻击-扰动约束流攻击(PCFA)-强调破坏性而不是适用性作为现实世界的攻击。PCFA是一种全局攻击，它优化对抗性扰动，将预测流向指定的目标流移动，同时将扰动的L2范数保持在选定的界限以下。我们的实验证明了PCFA在白盒和黑盒环境下的适用性，并表明它比以前的攻击发现了更强的敌意样本。基于这些强样本，我们提供了第一个综合考虑预测质量和对抗稳健性的光流方法的联合排名，这揭示了最新的方法特别脆弱。代码可在https://github.com/cv-stuttgart/PCFA.上找到



## **7. Universal Adversarial Examples in Remote Sensing: Methodology and Benchmark**

遥感领域的普遍对抗性实例：方法论和基准 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2202.07054v3)

**Authors**: Yonghao Xu, Pedram Ghamisi

**Abstracts**: Deep neural networks have achieved great success in many important remote sensing tasks. Nevertheless, their vulnerability to adversarial examples should not be neglected. In this study, we systematically analyze the universal adversarial examples in remote sensing data for the first time, without any knowledge from the victim model. Specifically, we propose a novel black-box adversarial attack method, namely Mixup-Attack, and its simple variant Mixcut-Attack, for remote sensing data. The key idea of the proposed methods is to find common vulnerabilities among different networks by attacking the features in the shallow layer of a given surrogate model. Despite their simplicity, the proposed methods can generate transferable adversarial examples that deceive most of the state-of-the-art deep neural networks in both scene classification and semantic segmentation tasks with high success rates. We further provide the generated universal adversarial examples in the dataset named UAE-RS, which is the first dataset that provides black-box adversarial samples in the remote sensing field. We hope UAE-RS may serve as a benchmark that helps researchers to design deep neural networks with strong resistance toward adversarial attacks in the remote sensing field. Codes and the UAE-RS dataset are available online (https://github.com/YonghaoXu/UAE-RS).

摘要: 深度神经网络在许多重要的遥感任务中取得了巨大的成功。然而，不应忽视它们在对抗性例子面前的脆弱性。在本研究中，我们首次在没有任何受害者模型知识的情况下，系统地分析了遥感数据中的通用对抗性实例。具体来说，我们提出了一种新的针对遥感数据的黑盒对抗攻击方法，即Mixup攻击及其简单的变种MixCut攻击。提出的方法的核心思想是通过攻击给定代理模型浅层的特征来发现不同网络之间的共同漏洞。尽管方法简单，但在场景分类和语义分割任务中，所提出的方法可以生成可转移的对抗性样本，欺骗了大多数最新的深度神经网络，并且成功率很高。此外，我们还在名为UAE-RS的数据集上提供了生成的通用对抗性实例，这是遥感领域中第一个提供黑盒对抗性样本的数据集。我们希望UAE-RS可以作为一个基准，帮助研究人员设计出对遥感领域的敌意攻击具有很强抵抗能力的深度神经网络。代码和阿联酋-RS数据集可在网上获得(https://github.com/YonghaoXu/UAE-RS).



## **8. Watermark Vaccine: Adversarial Attacks to Prevent Watermark Removal**

水印疫苗：防止水印去除的对抗性攻击 cs.CV

ECCV 2022

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08178v1)

**Authors**: Xinwei Liu, Jian Liu, Yang Bai, Jindong Gu, Tao Chen, Xiaojun Jia, Xiaochun Cao

**Abstracts**: As a common security tool, visible watermarking has been widely applied to protect copyrights of digital images. However, recent works have shown that visible watermarks can be removed by DNNs without damaging their host images. Such watermark-removal techniques pose a great threat to the ownership of images. Inspired by the vulnerability of DNNs on adversarial perturbations, we propose a novel defence mechanism by adversarial machine learning for good. From the perspective of the adversary, blind watermark-removal networks can be posed as our target models; then we actually optimize an imperceptible adversarial perturbation on the host images to proactively attack against watermark-removal networks, dubbed Watermark Vaccine. Specifically, two types of vaccines are proposed. Disrupting Watermark Vaccine (DWV) induces to ruin the host image along with watermark after passing through watermark-removal networks. In contrast, Inerasable Watermark Vaccine (IWV) works in another fashion of trying to keep the watermark not removed and still noticeable. Extensive experiments demonstrate the effectiveness of our DWV/IWV in preventing watermark removal, especially on various watermark removal networks.

摘要: 可见水印作为一种常用的安全工具，已被广泛应用于数字图像的版权保护。然而，最近的研究表明，DNN可以在不损害宿主图像的情况下去除可见水印。这种水印去除技术对图像的所有权构成了极大的威胁。受DNN对对抗性扰动的脆弱性的启发，我们提出了一种基于对抗性机器学习的新型防御机制。从敌手的角度来看，盲水印去除网络可以作为我们的目标模型，然后我们实际上优化了宿主图像上的一种不可感知的对抗性扰动，以主动攻击水印去除网络，称为水印疫苗。具体地说，提出了两种疫苗。破坏水印疫苗(DWV)在通过水印去除网络后，会导致宿主图像与水印一起被破坏。相比之下，不可擦除水印疫苗(IWV)的另一种工作方式是试图保持水印不被移除并仍然可见。大量的实验证明了我们的DWV/IWV在防止水印去除方面的有效性，特别是在各种水印去除网络上。



## **9. Modeling Adversarial Noise for Adversarial Training**

对抗性训练中的对抗性噪声建模 cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2109.09901v5)

**Authors**: Dawei Zhou, Nannan Wang, Bo Han, Tongliang Liu

**Abstracts**: Deep neural networks have been demonstrated to be vulnerable to adversarial noise, promoting the development of defense against adversarial attacks. Motivated by the fact that adversarial noise contains well-generalizing features and that the relationship between adversarial data and natural data can help infer natural data and make reliable predictions, in this paper, we study to model adversarial noise by learning the transition relationship between adversarial labels (i.e. the flipped labels used to generate adversarial data) and natural labels (i.e. the ground truth labels of the natural data). Specifically, we introduce an instance-dependent transition matrix to relate adversarial labels and natural labels, which can be seamlessly embedded with the target model (enabling us to model stronger adaptive adversarial noise). Empirical evaluations demonstrate that our method could effectively improve adversarial accuracy.

摘要: 深度神经网络已被证明对对抗性噪声很敏感，这促进了防御对抗性攻击的发展。对抗性噪声包含了良好的泛化特征，并且对抗性数据与自然数据之间的关系可以帮助推断自然数据并做出可靠的预测，本文通过学习对抗性标签(即用于生成对抗性数据的翻转标签)和自然标签(即自然数据的地面真值标签)之间的转换关系来研究对抗性噪声的建模。具体地说，我们引入了依赖于实例的转移矩阵来关联对抗性标签和自然标签，该转移矩阵可以无缝地嵌入目标模型(使我们能够建模更强的自适应对抗性噪声)。实验结果表明，该方法能够有效地提高对手识别的准确率。



## **10. Automated Repair of Neural Networks**

神经网络的自动修复 cs.LG

Code and results are available at  https://github.com/dorcoh/NNSynthesizer

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08157v1)

**Authors**: Dor Cohen, Ofer Strichman

**Abstracts**: Over the last decade, Neural Networks (NNs) have been widely used in numerous applications including safety-critical ones such as autonomous systems. Despite their emerging adoption, it is well known that NNs are susceptible to Adversarial Attacks. Hence, it is highly important to provide guarantees that such systems work correctly. To remedy these issues we introduce a framework for repairing unsafe NNs w.r.t. safety specification, that is by utilizing satisfiability modulo theories (SMT) solvers. Our method is able to search for a new, safe NN representation, by modifying only a few of its weight values. In addition, our technique attempts to maximize the similarity to original network with regard to its decision boundaries. We perform extensive experiments which demonstrate the capability of our proposed framework to yield safe NNs w.r.t. the Adversarial Robustness property, with only a mild loss of accuracy (in terms of similarity). Moreover, we compare our method with a naive baseline to empirically prove its effectiveness. To conclude, we provide an algorithm to automatically repair NNs given safety properties, and suggest a few heuristics to improve its computational performance. Currently, by following this approach we are capable of producing small-sized (i.e., with up to few hundreds of parameters) correct NNs, composed of the piecewise linear ReLU activation function. Nevertheless, our framework is general in the sense that it can synthesize NNs w.r.t. any decidable fragment of first-order logic specification.

摘要: 在过去的十年里，神经网络(NNS)被广泛地应用于许多应用中，包括诸如自治系统等安全关键的应用。尽管它们正在被采用，但众所周知，NNS很容易受到对手的攻击。因此，提供此类系统正常工作的保证是非常重要的。为了解决这些问题，我们引入了一个修复不安全NNW.r.t.的框架。安全规范，即利用可满足性模理论(SMT)求解器。我们的方法能够搜索一个新的、安全的神经网络表示，只需修改它的几个权值。此外，我们的技术试图在决策边界方面最大化与原始网络的相似性。我们进行了大量的实验，证明了我们所提出的框架能够产生安全的NNW.r.t.对抗性稳健性，只有轻微的准确性损失(就相似性而言)。此外，我们将我们的方法与天真的基线进行了比较，以经验证明其有效性。综上所述，我们提出了一种在给定安全属性的情况下自动修复神经网络的算法，并提出了一些启发式算法来提高其计算性能。目前，通过采用这种方法，我们能够产生由分段线性REU激活函数组成的小尺寸(即，多达数百个参数)的正确神经网络。然而，我们的框架在某种意义上是通用的，它可以合成NNS w.r.t.一阶逻辑规范的任何可判定片段。



## **11. Achieve Optimal Adversarial Accuracy for Adversarial Deep Learning using Stackelberg Game**

利用Stackelberg博弈实现对抗性深度学习的最优对抗性准确率 cs.LG

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08137v1)

**Authors**: Xiao-Shan Gao, Shuang Liu, Lijia Yu

**Abstracts**: Adversarial deep learning is to train robust DNNs against adversarial attacks, which is one of the major research focuses of deep learning. Game theory has been used to answer some of the basic questions about adversarial deep learning such as the existence of a classifier with optimal robustness and the existence of optimal adversarial samples for a given class of classifiers. In most previous work, adversarial deep learning was formulated as a simultaneous game and the strategy spaces are assumed to be certain probability distributions in order for the Nash equilibrium to exist. But, this assumption is not applicable to the practical situation. In this paper, we give answers to these basic questions for the practical case where the classifiers are DNNs with a given structure, by formulating the adversarial deep learning as sequential games. The existence of Stackelberg equilibria for these games are proved. Furthermore, it is shown that the equilibrium DNN has the largest adversarial accuracy among all DNNs with the same structure, when Carlini-Wagner's margin loss is used. Trade-off between robustness and accuracy in adversarial deep learning is also studied from game theoretical aspect.

摘要: 对抗性深度学习是针对敌意攻击训练健壮的DNN，是深度学习的主要研究热点之一。博弈论已经被用来回答对抗性深度学习的一些基本问题，例如具有最优稳健性的分类器的存在以及给定类别的分类器的最优对抗性样本的存在。在以前的工作中，对抗性深度学习被描述为一个联立博弈，并且假设策略空间是一定的概率分布，以便纳什均衡的存在。但是，这一假设并不适用于实际情况。在本文中，我们通过将对抗性深度学习描述为序列博弈，针对分类器是具有给定结构的DNN的实际情况，回答了这些基本问题。证明了这些博弈的Stackelberg均衡的存在性。此外，当使用Carlini-Wagner边际损失时，均衡DNN在所有相同结构的DNN中具有最大的对抗准确率。从博弈论的角度研究了对抗性深度学习的稳健性和精确度之间的权衡。



## **12. Threat Model-Agnostic Adversarial Defense using Diffusion Models**

威胁模型--使用扩散模型的不可知式对抗防御 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08089v1)

**Authors**: Tsachi Blau, Roy Ganz, Bahjat Kawar, Alex Bronstein, Michael Elad

**Abstracts**: Deep Neural Networks (DNNs) are highly sensitive to imperceptible malicious perturbations, known as adversarial attacks. Following the discovery of this vulnerability in real-world imaging and vision applications, the associated safety concerns have attracted vast research attention, and many defense techniques have been developed. Most of these defense methods rely on adversarial training (AT) -- training the classification network on images perturbed according to a specific threat model, which defines the magnitude of the allowed modification. Although AT leads to promising results, training on a specific threat model fails to generalize to other types of perturbations. A different approach utilizes a preprocessing step to remove the adversarial perturbation from the attacked image. In this work, we follow the latter path and aim to develop a technique that leads to robust classifiers across various realizations of threat models. To this end, we harness the recent advances in stochastic generative modeling, and means to leverage these for sampling from conditional distributions. Our defense relies on an addition of Gaussian i.i.d noise to the attacked image, followed by a pretrained diffusion process -- an architecture that performs a stochastic iterative process over a denoising network, yielding a high perceptual quality denoised outcome. The obtained robustness with this stochastic preprocessing step is validated through extensive experiments on the CIFAR-10 dataset, showing that our method outperforms the leading defense methods under various threat models.

摘要: 深度神经网络(DNN)对不可察觉的恶意扰动高度敏感，这种恶意扰动称为对抗性攻击。随着在真实世界的成像和视觉应用中发现该漏洞，相关的安全问题引起了广泛的研究关注，许多防御技术也被开发出来。这些防御方法中的大多数依赖于对抗性训练(AT)-根据特定的威胁模型对受干扰的图像训练分类网络，该模型定义了允许修改的幅度。虽然AT带来了有希望的结果，但对特定威胁模型的训练无法推广到其他类型的扰动。一种不同的方法利用预处理步骤来从被攻击的图像中移除对抗性扰动。在这项工作中，我们遵循后一条道路，目标是开发一种技术，从而在威胁模型的各种实现中产生健壮的分类器。为此，我们利用了随机生成建模的最新进展，并利用这些方法从条件分布中进行抽样。我们的防御依赖于在受攻击的图像中添加高斯I.I.D噪声，然后是预先训练的扩散过程--一种在去噪网络上执行随机迭代过程的体系结构，产生高感知质量的去噪结果。通过在CIFAR-10数据集上的大量实验，验证了该随机预处理步骤所获得的稳健性，表明该方法在各种威胁模型下的性能优于领先的防御方法。



## **13. DIMBA: Discretely Masked Black-Box Attack in Single Object Tracking**

DIMBA：单目标跟踪中的离散掩蔽黑盒攻击 cs.CV

**SubmitDate**: 2022-07-17    [paper-pdf](http://arxiv.org/pdf/2207.08044v1)

**Authors**: Xiangyu Yin, Wenjie Ruan, Jonathan Fieldsend

**Abstracts**: The adversarial attack can force a CNN-based model to produce an incorrect output by craftily manipulating human-imperceptible input. Exploring such perturbations can help us gain a deeper understanding of the vulnerability of neural networks, and provide robustness to deep learning against miscellaneous adversaries. Despite extensive studies focusing on the robustness of image, audio, and NLP, works on adversarial examples of visual object tracking -- especially in a black-box manner -- are quite lacking. In this paper, we propose a novel adversarial attack method to generate noises for single object tracking under black-box settings, where perturbations are merely added on initial frames of tracking sequences, which is difficult to be noticed from the perspective of a whole video clip. Specifically, we divide our algorithm into three components and exploit reinforcement learning for localizing important frame patches precisely while reducing unnecessary computational queries overhead. Compared to existing techniques, our method requires fewer queries on initialized frames of a video to manipulate competitive or even better attack performance. We test our algorithm in both long-term and short-term datasets, including OTB100, VOT2018, UAV123, and LaSOT. Extensive experiments demonstrate the effectiveness of our method on three mainstream types of trackers: discrimination, Siamese-based, and reinforcement learning-based trackers.

摘要: 敌意攻击可以通过巧妙地操纵人类无法察觉的输入，迫使基于CNN的模型产生错误的输出。探索这种扰动可以帮助我们更深入地了解神经网络的脆弱性，并为针对各种对手的深度学习提供健壮性。尽管广泛的研究集中在图像、音频和NLP的健壮性上，但关于视觉对象跟踪的对抗性例子--尤其是以黑盒方式--的工作相当缺乏。本文提出了一种新的对抗性攻击方法，用于黑盒环境下的单目标跟踪，该方法只对跟踪序列的初始帧添加扰动，从整个视频片段的角度来看很难注意到这一点。具体地说，我们将算法分为三个部分，并利用强化学习来精确定位重要的帧补丁，同时减少不必要的计算查询开销。与现有技术相比，我们的方法需要对视频的初始化帧进行更少的查询来操纵竞争甚至更好的攻击性能。我们在长期和短期数据集上测试了我们的算法，包括OTB100、VOT2018、UAV123和LaSOT。大量的实验表明，我们的方法在三种主流类型的跟踪器上是有效的：区分跟踪器、基于暹罗的跟踪器和基于强化学习的跟踪器。



## **14. Optimal Strategic Mining Against Cryptographic Self-Selection in Proof-of-Stake**

基于密码学自我选择的最优策略挖掘 cs.CR

31 pages, ACM EC 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07996v1)

**Authors**: Matheus V. X. Ferreira, Ye Lin Sally Hahn, S. Matthew Weinberg, Catherine Yu

**Abstracts**: Cryptographic Self-Selection is a subroutine used to select a leader for modern proof-of-stake consensus protocols, such as Algorand. In cryptographic self-selection, each round $r$ has a seed $Q_r$. In round $r$, each account owner is asked to digitally sign $Q_r$, hash their digital signature to produce a credential, and then broadcast this credential to the entire network. A publicly-known function scores each credential in a manner so that the distribution of the lowest scoring credential is identical to the distribution of stake owned by each account. The user who broadcasts the lowest-scoring credential is the leader for round $r$, and their credential becomes the seed $Q_{r+1}$. Such protocols leave open the possibility of a selfish-mining style attack: a user who owns multiple accounts that each produce low-scoring credentials in round $r$ can selectively choose which ones to broadcast in order to influence the seed for round $r+1$. Indeed, the user can pre-compute their credentials for round $r+1$ for each potential seed, and broadcast only the credential (among those with a low enough score to be the leader) that produces the most favorable seed.   We consider an adversary who wishes to maximize the expected fraction of rounds in which an account they own is the leader. We show such an adversary always benefits from deviating from the intended protocol, regardless of the fraction of the stake controlled. We characterize the optimal strategy; first by proving the existence of optimal positive recurrent strategies whenever the adversary owns last than $38\%$ of the stake. Then, we provide a Markov Decision Process formulation to compute the optimal strategy.

摘要: 加密自我选择是用于为现代利害关系证明共识协议(如算法)选择领导者的子例程。在加密自我选择中，每轮$r$都有一个种子$q_r$。在$r$中，每个帐户所有者被要求对$q_r$进行数字签名，对他们的数字签名进行散列以生成凭据，然后将该凭据广播到整个网络。公知函数以一种方式对每个凭证进行评分，使得得分最低的凭证的分布与每个帐户拥有的赌注的分布相同。广播得分最低的凭据的用户是$r$的领先者，他们的凭据成为种子$q_{r+1}$。这样的协议为自私挖掘式攻击提供了可能性：拥有多个账户的用户，每个账户都在$r$中产生低得分凭据，可以有选择地选择广播哪些账户，以便在$r+1$中影响种子。事实上，用户可以为每个潜在种子预先计算$r+1$的凭据，并且只广播产生最有利种子的凭据(在那些得分足够低的人中)。我们考虑一个对手，他希望最大化他们拥有的账户是领导者的回合的预期比例。我们表明，这样的对手总是从偏离预期的协议中受益，无论控制的风险有多大。我们刻画了最优策略：首先，证明了当对手拥有最后$38的股份时，最优正回归策略的存在性。然后，我们给出了一个马尔可夫决策过程公式来计算最优策略。



## **15. BOSS: Bidirectional One-Shot Synthesis of Adversarial Examples**

BOSS：对抗性范例的双向一次合成 cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2108.02756v2)

**Authors**: Ismail R. Alkhouri, Alvaro Velasquez, George K. Atia

**Abstracts**: The design of additive imperceptible perturbations to the inputs of deep classifiers to maximize their misclassification rates is a central focus of adversarial machine learning. An alternative approach is to synthesize adversarial examples from scratch using GAN-like structures, albeit with the use of large amounts of training data. By contrast, this paper considers one-shot synthesis of adversarial examples; the inputs are synthesized from scratch to induce arbitrary soft predictions at the output of pre-trained models, while simultaneously maintaining high similarity to specified inputs. To this end, we present a problem that encodes objectives on the distance between the desired and output distributions of the trained model and the similarity between such inputs and the synthesized examples. We prove that the formulated problem is NP-complete. Then, we advance a generative approach to the solution in which the adversarial examples are obtained as the output of a generative network whose parameters are iteratively updated by optimizing surrogate loss functions for the dual-objective. We demonstrate the generality and versatility of the framework and approach proposed through applications to the design of targeted adversarial attacks, generation of decision boundary samples, and synthesis of low confidence classification inputs. The approach is further extended to an ensemble of models with different soft output specifications. The experimental results verify that the targeted and confidence reduction attack methods developed perform on par with state-of-the-art algorithms.

摘要: 设计对深层分类器的输入进行不可察觉的加性扰动以最大化其错误分类率是对抗性机器学习的中心问题。另一种方法是使用GaN类结构从头开始合成对抗性例子，尽管使用了大量的训练数据。相反，本文考虑了对抗性例子的一次合成；输入是从头开始合成的，在预先训练的模型的输出处诱导出任意的软预测，同时保持与指定输入的高度相似。为此，我们提出了一个问题，即根据训练模型的期望分布和输出分布之间的距离以及这些输入和合成样本之间的相似性对目标进行编码。我们证明了所提出的问题是NP完全的。然后，我们提出了一种求解该问题的产生式方法，即通过优化双目标的代理损失函数，将对抗性实例作为产生式网络的输出来迭代地更新其参数。我们通过在目标对抗性攻击的设计、决策边界样本的生成和低置信度分类输入的合成中的应用，展示了所提出的框架和方法的通用性和通用性。该方法进一步扩展到具有不同软输出规格的模型集成。实验结果表明，本文提出的目标攻击和置信度降低攻击方法的性能与目前最先进的算法相当。



## **16. Certified Neural Network Watermarks with Randomized Smoothing**

随机平滑认证神经网络水印 cs.LG

ICML 2022

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07972v1)

**Authors**: Arpit Bansal, Ping-yeh Chiang, Michael Curry, Rajiv Jain, Curtis Wigington, Varun Manjunatha, John P Dickerson, Tom Goldstein

**Abstracts**: Watermarking is a commonly used strategy to protect creators' rights to digital images, videos and audio. Recently, watermarking methods have been extended to deep learning models -- in principle, the watermark should be preserved when an adversary tries to copy the model. However, in practice, watermarks can often be removed by an intelligent adversary. Several papers have proposed watermarking methods that claim to be empirically resistant to different types of removal attacks, but these new techniques often fail in the face of new or better-tuned adversaries. In this paper, we propose a certifiable watermarking method. Using the randomized smoothing technique proposed in Chiang et al., we show that our watermark is guaranteed to be unremovable unless the model parameters are changed by more than a certain l2 threshold. In addition to being certifiable, our watermark is also empirically more robust compared to previous watermarking methods. Our experiments can be reproduced with code at https://github.com/arpitbansal297/Certified_Watermarks

摘要: 水印是保护创作者对数字图像、视频和音频的权利的一种常用策略。最近，水印方法已经扩展到深度学习模型--原则上，当对手试图复制模型时，水印应该被保留。然而，在实践中，水印往往可以被聪明的对手移除。有几篇论文提出了一些水印方法，这些方法声称在经验上可以抵抗不同类型的删除攻击，但这些新技术在面对新的或调整得更好的对手时往往会失败。在本文中，我们提出了一种可认证的水印方法。使用Chiang等人提出的随机平滑技术，我们证明了除非模型参数改变超过一定的L2阈值，否则我们的水印是不可移除的。除了是可认证的，我们的水印在经验上也比以前的水印方法更健壮。我们的实验可以通过https://github.com/arpitbansal297/Certified_Watermarks上的代码重现



## **17. MixTailor: Mixed Gradient Aggregation for Robust Learning Against Tailored Attacks**

MixTailor：针对定制攻击的稳健学习的混合梯度聚合 cs.LG

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07941v1)

**Authors**: Ali Ramezani-Kebrya, Iman Tabrizian, Fartash Faghri, Petar Popovski

**Abstracts**: Implementations of SGD on distributed and multi-GPU systems creates new vulnerabilities, which can be identified and misused by one or more adversarial agents. Recently, it has been shown that well-known Byzantine-resilient gradient aggregation schemes are indeed vulnerable to informed attackers that can tailor the attacks (Fang et al., 2020; Xie et al., 2020b). We introduce MixTailor, a scheme based on randomization of the aggregation strategies that makes it impossible for the attacker to be fully informed. Deterministic schemes can be integrated into MixTailor on the fly without introducing any additional hyperparameters. Randomization decreases the capability of a powerful adversary to tailor its attacks, while the resulting randomized aggregation scheme is still competitive in terms of performance. For both iid and non-iid settings, we establish almost sure convergence guarantees that are both stronger and more general than those available in the literature. Our empirical studies across various datasets, attacks, and settings, validate our hypothesis and show that MixTailor successfully defends when well-known Byzantine-tolerant schemes fail.

摘要: SGD在分布式和多GPU系统上的实现会产生新的漏洞，这些漏洞可能会被一个或多个对抗性代理识别和滥用。最近，有研究表明，众所周知的拜占庭弹性梯度聚合方案确实容易受到可以定制攻击的知情攻击者的攻击(方等人，2020；谢等人，2020b)。我们引入了MixTailor，这是一种基于聚合策略随机化的方案，使得攻击者不可能被完全告知。确定性方案可以动态地集成到MixTailor中，而不需要引入任何额外的超参数。随机化降低了强大对手定制其攻击的能力，而由此产生的随机化聚集方案在性能方面仍然具有竞争力。对于iID和非iID设置，我们几乎肯定建立了比文献中提供的更强大和更一般的收敛保证。我们对各种数据集、攻击和环境的经验研究验证了我们的假设，并表明当众所周知的拜占庭容忍方案失败时，MixTailor成功地进行了辩护。



## **18. Masked Spatial-Spectral Autoencoders Are Excellent Hyperspectral Defenders**

屏蔽式空间光谱自动编码器是优秀的高光谱防御者 cs.CV

14 pages, 9 figures

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07803v1)

**Authors**: Jiahao Qi, Zhiqiang Gong, Xingyue Liu, Kangcheng Bin, Chen Chen, Yongqian Li, Wei Xue, Yu Zhang, Ping Zhong

**Abstracts**: Deep learning methodology contributes a lot to the development of hyperspectral image (HSI) analysis community. However, it also makes HSI analysis systems vulnerable to adversarial attacks. To this end, we propose a masked spatial-spectral autoencoder (MSSA) in this paper under self-supervised learning theory, for enhancing the robustness of HSI analysis systems. First, a masked sequence attention learning module is conducted to promote the inherent robustness of HSI analysis systems along spectral channel. Then, we develop a graph convolutional network with learnable graph structure to establish global pixel-wise combinations.In this way, the attack effect would be dispersed by all the related pixels among each combination, and a better defense performance is achievable in spatial aspect.Finally, to improve the defense transferability and address the problem of limited labelled samples, MSSA employs spectra reconstruction as a pretext task and fits the datasets in a self-supervised manner.Comprehensive experiments over three benchmarks verify the effectiveness of MSSA in comparison with the state-of-the-art hyperspectral classification methods and representative adversarial defense strategies.

摘要: 深度学习方法对高光谱图像(HSI)分析社区的发展做出了重要贡献。然而，它也使HSI分析系统容易受到对手攻击。为此，本文提出了一种基于自监督学习理论的屏蔽空间谱自动编码器(MSSA)，以增强HSI分析系统的鲁棒性。首先，采用掩蔽序列注意学习模块来提高HSI分析系统在频谱信道上的固有健壮性。然后，提出了一种具有可学习图结构的图卷积网络来建立全局像素级组合，这样攻击效果将由每个组合之间的所有相关像素来分散，从而在空间方面获得更好的防御性能；最后，为了提高防御的可传递性并解决标记样本有限的问题，MSSA采用谱重建作为借口任务，并以自监督的方式对数据集进行拟合。



## **19. CARBEN: Composite Adversarial Robustness Benchmark**

Carben：复合对抗健壮性基准 cs.CV

IJCAI 2022 Demo Track; The demonstration is at  https://hsiung.cc/CARBEN/

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07797v1)

**Authors**: Lei Hsiung, Yun-Yun Tsai, Pin-Yu Chen, Tsung-Yi Ho

**Abstracts**: Prior literature on adversarial attack methods has mainly focused on attacking with and defending against a single threat model, e.g., perturbations bounded in Lp ball. However, multiple threat models can be combined into composite perturbations. One such approach, composite adversarial attack (CAA), not only expands the perturbable space of the image, but also may be overlooked by current modes of robustness evaluation. This paper demonstrates how CAA's attack order affects the resulting image, and provides real-time inferences of different models, which will facilitate users' configuration of the parameters of the attack level and their rapid evaluation of model prediction. A leaderboard to benchmark adversarial robustness against CAA is also introduced.

摘要: 以前关于对抗性攻击方法的文献主要集中在利用单一威胁模型进行攻击和防御，例如，在LP球中有界的扰动。但是，可以将多个威胁模型组合成复合扰动。一种这样的方法，复合对抗攻击(CAA)，不仅扩展了图像的可扰动空间，而且可能被现有的健壮性评估模式所忽视。文中演示了CAA的攻击顺序对结果图像的影响，并提供了不同模型的实时推理，这将方便用户配置攻击级别的参数，并快速评估模型预测。此外，还引入了一个排行榜来衡量对抗CAA的健壮性。



## **20. Towards the Desirable Decision Boundary by Moderate-Margin Adversarial Training**

通过适度的对抗性训练达到理想的决策边界 cs.CV

**SubmitDate**: 2022-07-16    [paper-pdf](http://arxiv.org/pdf/2207.07793v1)

**Authors**: Xiaoyu Liang, Yaguan Qian, Jianchang Huang, Xiang Ling, Bin Wang, Chunming Wu, Wassim Swaileh

**Abstracts**: Adversarial training, as one of the most effective defense methods against adversarial attacks, tends to learn an inclusive decision boundary to increase the robustness of deep learning models. However, due to the large and unnecessary increase in the margin along adversarial directions, adversarial training causes heavy cross-over between natural examples and adversarial examples, which is not conducive to balancing the trade-off between robustness and natural accuracy. In this paper, we propose a novel adversarial training scheme to achieve a better trade-off between robustness and natural accuracy. It aims to learn a moderate-inclusive decision boundary, which means that the margins of natural examples under the decision boundary are moderate. We call this scheme Moderate-Margin Adversarial Training (MMAT), which generates finer-grained adversarial examples to mitigate the cross-over problem. We also take advantage of logits from a teacher model that has been well-trained to guide the learning of our model. Finally, MMAT achieves high natural accuracy and robustness under both black-box and white-box attacks. On SVHN, for example, state-of-the-art robustness and natural accuracy are achieved.

摘要: 对抗性训练作为对抗对抗性攻击的最有效的防御方法之一，倾向于学习一个包容的决策边界，以增加深度学习模型的鲁棒性。然而，由于对抗性方向上的差值有很大且不必要的增加，对抗性训练导致自然例子和对抗性例子之间的严重交叉，这不利于在稳健性和自然准确性之间取得平衡。在本文中，我们提出了一种新的对抗性训练方案，以在稳健性和自然准确性之间实现更好的权衡。它的目的是学习一个适度包容的决策边界，这意味着决策边界下的自然例子的边际是适度的。我们称之为中等边际对抗性训练(MMAT)，它生成更细粒度的对抗性例子来缓解交叉问题。我们还利用训练有素的教师模型的Logit来指导我们模型的学习。最后，在黑盒和白盒攻击下，MMAT都达到了很高的自然准确率和稳健性。例如，在SVHN上，实现了最先进的健壮性和自然准确性。



## **21. Demystifying the Adversarial Robustness of Random Transformation Defenses**

揭开随机变换防御对抗健壮性的神秘面纱 cs.CR

ICML 2022 (short presentation), AAAI 2022 AdvML Workshop (best paper,  oral presentation)

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.03574v2)

**Authors**: Chawin Sitawarin, Zachary Golan-Strieb, David Wagner

**Abstracts**: Neural networks' lack of robustness against attacks raises concerns in security-sensitive settings such as autonomous vehicles. While many countermeasures may look promising, only a few withstand rigorous evaluation. Defenses using random transformations (RT) have shown impressive results, particularly BaRT (Raff et al., 2019) on ImageNet. However, this type of defense has not been rigorously evaluated, leaving its robustness properties poorly understood. Their stochastic properties make evaluation more challenging and render many proposed attacks on deterministic models inapplicable. First, we show that the BPDA attack (Athalye et al., 2018a) used in BaRT's evaluation is ineffective and likely overestimates its robustness. We then attempt to construct the strongest possible RT defense through the informed selection of transformations and Bayesian optimization for tuning their parameters. Furthermore, we create the strongest possible attack to evaluate our RT defense. Our new attack vastly outperforms the baseline, reducing the accuracy by 83% compared to the 19% reduction by the commonly used EoT attack ($4.3\times$ improvement). Our result indicates that the RT defense on the Imagenette dataset (a ten-class subset of ImageNet) is not robust against adversarial examples. Extending the study further, we use our new attack to adversarially train RT defense (called AdvRT), resulting in a large robustness gain. Code is available at https://github.com/wagner-group/demystify-random-transform.

摘要: 神经网络对攻击缺乏稳健性，这在自动驾驶汽车等安全敏感环境中引发了担忧。尽管许多对策看起来很有希望，但只有少数几项经得起严格的评估。使用随机变换(RT)的防御已经显示出令人印象深刻的结果，特别是Bart(Raff等人，2019年)在ImageNet上。然而，这种类型的防御没有得到严格的评估，使得人们对其健壮性属性知之甚少。它们的随机性使评估变得更具挑战性，并使许多已提出的对确定性模型的攻击不适用。首先，我们证明了BART评估中使用的BPDA攻击(Athalye等人，2018a)是无效的，并且可能高估了它的健壮性。然后，我们试图通过对变换的知情选择和贝叶斯优化来调整它们的参数，从而构建尽可能强的RT防御。此外，我们创建了尽可能强的攻击来评估我们的RT防御。我们的新攻击大大超过了基准，与常用的EoT攻击相比，准确率降低了83%($4.3倍$改进)。我们的结果表明，在Imagenette数据集(ImageNet的十类子集)上的RT防御对敌意示例不是健壮的。进一步扩展研究，我们使用我们的新攻击来恶意训练RT防御(称为AdvRT)，从而获得了很大的健壮性收益。代码可在https://github.com/wagner-group/demystify-random-transform.上找到



## **22. CC-Fuzz: Genetic algorithm-based fuzzing for stress testing congestion control algorithms**

CC-Fuzz：基于遗传算法的模糊压力测试拥塞控制算法 cs.NI

This version was submitted to Hotnets 2022

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07300v1)

**Authors**: Devdeep Ray, Srinivasan Seshan

**Abstracts**: Congestion control research has experienced a significant increase in interest in the past few years, with many purpose-built algorithms being designed with the needs of specific applications in mind. These algorithms undergo limited testing before being deployed on the Internet, where they interact with other congestion control algorithms and run across a variety of network conditions. This often results in unforeseen performance issues in the wild due to algorithmic inadequacies or implementation bugs, and these issues are often hard to identify since packet traces are not available.   In this paper, we present CC-Fuzz, an automated congestion control testing framework that uses a genetic search algorithm in order to stress test congestion control algorithms by generating adversarial network traces and traffic patterns. Initial results using this approach are promising - CC-Fuzz automatically found a bug in BBR that causes it to stall permanently, and is able to automatically discover the well-known low-rate TCP attack, among other things.

摘要: 在过去的几年中，拥塞控制研究的兴趣显著增加，许多专门设计的算法都是考虑到特定应用的需求而设计的。这些算法在部署到Internet之前经过有限的测试，在Internet上它们与其他拥塞控制算法交互，并在各种网络条件下运行。这通常会由于算法不足或实现错误而导致无法预见的性能问题，而且这些问题通常很难识别，因为数据包跟踪不可用。在本文中，我们提出了一个自动化拥塞控制测试框架CC-Fuzz，它使用遗传搜索算法，通过生成敌对的网络轨迹和流量模式来对拥塞控制算法进行压力测试。使用这种方法的初步结果是有希望的-CC-Fuzz自动发现BBR中的一个错误，导致它永久停止，并能够自动发现众所周知的低速率TCP攻击等。



## **23. PASS: Parameters Audit-based Secure and Fair Federated Learning Scheme against Free Rider**

PASS：基于参数审计的反搭便车安全公平联邦学习方案 cs.CR

8 pages, 5 figures, 3 tables

**SubmitDate**: 2022-07-15    [paper-pdf](http://arxiv.org/pdf/2207.07292v1)

**Authors**: Jianhua Wang

**Abstracts**: Federated Learning (FL) as a secure distributed learning frame gains interest in Internet of Things (IoT) due to its capability of protecting private data of participants. However, traditional FL systems are vulnerable to attacks such as Free-Rider (FR) attack, which causes not only unfairness but also privacy leakage and inferior performance to FL systems. The existing defense mechanisms against FR attacks only concern the scenarios where the adversaries declare less than 50% of the total amount of clients. Moreover, they lose effectiveness in resisting selfish FR (SFR) attacks. In this paper, we propose a Parameter Audit-based Secure and fair federated learning Scheme (PASS) against FR attacks. The PASS has the following key features: (a) works well in the scenario where adversaries are more than 50% of the total amount of clients; (b) is effective in countering anonymous FR attacks and SFR attacks; (c) prevents from privacy leakage without accuracy loss. Extensive experimental results verify the data protecting capability in mean square error against privacy leakage and reveal the effectiveness of PASS in terms of a higher defense success rate and lower false positive rate against anonymous SFR attacks. Note in addition, PASS produces no effect on FL accuracy when there is no FR adversary.

摘要: 联邦学习(FL)作为一种安全的分布式学习框架，因其能够保护参与者的隐私数据而受到物联网(IoT)的关注。然而，传统的FL系统容易受到搭便车(FR)攻击等攻击，这不仅会造成不公平，而且还会泄露隐私，降低FL系统的性能。现有的FR攻击防御机制只针对对手申报的客户端总数不到50%的场景。此外，它们在抵抗自私FR(SFR)攻击方面也失去了效力。针对FR攻击，提出了一种基于参数审计的安全公平的联邦学习方案(PASS)。PASS具有以下主要特点：(A)在对手占客户端总数50%以上的场景下工作良好；(B)有效对抗匿名FR攻击和SFR攻击；(C)在不损失准确性的情况下防止隐私泄露。大量的实验结果验证了PASS在均方误差下对隐私泄露的保护能力，揭示了PASS对匿名SFR攻击具有较高的防御成功率和较低的误检率。另外，在没有FR对手的情况下，传球不会影响FL的准确性。



## **24. Lipschitz Bound Analysis of Neural Networks**

神经网络的Lipschitz界分析 cs.LG

5 pages, 7 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07232v1)

**Authors**: Sarosij Bose

**Abstracts**: Lipschitz Bound Estimation is an effective method of regularizing deep neural networks to make them robust against adversarial attacks. This is useful in a variety of applications ranging from reinforcement learning to autonomous systems. In this paper, we highlight the significant gap in obtaining a non-trivial Lipschitz bound certificate for Convolutional Neural Networks (CNNs) and empirically support it with extensive graphical analysis. We also show that unrolling Convolutional layers or Toeplitz matrices can be employed to convert Convolutional Neural Networks (CNNs) to a Fully Connected Network. Further, we propose a simple algorithm to show the existing 20x-50x gap in a particular data distribution between the actual lipschitz constant and the obtained tight bound. We also ran sets of thorough experiments on various network architectures and benchmark them on datasets like MNIST and CIFAR-10. All these proposals are supported by extensive testing, graphs, histograms and comparative analysis.

摘要: Lipschitz界估计是一种有效的正则化深度神经网络的方法，使其对敌意攻击具有较强的鲁棒性。这在从强化学习到自主系统的各种应用中都很有用。在这篇文章中，我们强调了在获得卷积神经网络(CNN)的非平凡Lipschitz界证书方面的显著差距，并用广泛的图形分析对其进行了经验支持。我们还证明了卷积层展开或Toeplitz矩阵可用于将卷积神经网络(CNN)转换为完全连通网络。此外，我们提出了一个简单的算法来显示在特定数据分布中实际的Lipschitz常数和得到的紧界之间存在的20x-50x的差距。我们还在各种网络体系结构上运行了一组彻底的实验，并在MNIST和CIFAR-10等数据集上对它们进行了基准测试。所有这些建议都得到了广泛的测试、图表、直方图和比较分析的支持。



## **25. Multi-Agent Deep Reinforcement Learning-Driven Mitigation of Adverse Effects of Cyber-Attacks on Electric Vehicle Charging Station**

基于多智能体深度强化学习的电动汽车充电站网络攻击缓解 eess.SY

Submitted to IEEE Transactions on Smart Grids

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07041v1)

**Authors**: M. Basnet, MH Ali

**Abstracts**: An electric vehicle charging station (EVCS) infrastructure is the backbone of transportation electrification. However, the EVCS has myriads of exploitable vulnerabilities in software, hardware, supply chain, and incumbent legacy technologies such as network, communication, and control. These standalone or networked EVCS open up large attack surfaces for the local or state-funded adversaries. The state-of-the-art approaches are not agile and intelligent enough to defend against and mitigate advanced persistent threats (APT). We propose the data-driven model-free distributed intelligence based on multiagent Deep Reinforcement Learning (MADRL)-- Twin Delayed Deep Deterministic Policy Gradient (TD3) -- that efficiently learns the control policy to mitigate the cyberattacks on the controllers of EVCS. Also, we have proposed two additional mitigation methods: the manual/Bruteforce mitigation and the controller clone-based mitigation. The attack model considers the APT designed to malfunction the duty cycles of the EVCS controllers with Type-I low-frequency attack and Type-II constant attack. The proposed model restores the EVCS operation under threat incidence in any/all controllers by correcting the control signals generated by the legacy controllers. Also, the TD3 algorithm provides higher granularity by learning nonlinear control policies as compared to the other two mitigation methods. Index Terms: Cyberattack, Deep Reinforcement Learning(DRL), Electric Vehicle Charging Station, Mitigation.

摘要: 电动汽车充电站(EVCS)基础设施是交通电气化的支柱。然而，EVCS在软件、硬件、供应链和现有的遗留技术(如网络、通信和控制)中存在无数可利用的漏洞。这些独立或联网的EVCS为当地或国家资助的对手打开了巨大的攻击面。最先进的方法不够灵活和智能，无法防御和缓解高级持续威胁(APT)。提出了一种基于多智能体深度强化学习(MADRL)的无模型数据驱动的分布式智能模型--双延迟深度确定性策略梯度(TD3)，它能有效地学习控制策略以减轻对EVCS控制器的网络攻击。此外，我们还提出了两种额外的缓解方法：手动/暴力缓解和基于控制器克隆的缓解。该攻击模型考虑了APT，该APT被设计为在I型低频攻击和II型恒定攻击下使EVCS控制器的占空比发生故障。该模型通过校正传统控制器产生的控制信号，恢复了任意/所有控制器在威胁发生时的EVCS操作。此外，与其他两种缓解方法相比，TD3算法通过学习非线性控制策略提供了更高的粒度。索引词：网络攻击，深度强化学习，电动汽车充电站，缓解。



## **26. Adversarial Attacks on Monocular Pose Estimation**

针对单目位姿估计的对抗性攻击 cs.CV

Accepted at the 2022 IEEE/RSJ International Conference on Intelligent  Robots and Systems (IROS 2022)

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.07032v1)

**Authors**: Hemang Chawla, Arnav Varma, Elahe Arani, Bahram Zonooz

**Abstracts**: Advances in deep learning have resulted in steady progress in computer vision with improved accuracy on tasks such as object detection and semantic segmentation. Nevertheless, deep neural networks are vulnerable to adversarial attacks, thus presenting a challenge in reliable deployment. Two of the prominent tasks in 3D scene-understanding for robotics and advanced drive assistance systems are monocular depth and pose estimation, often learned together in an unsupervised manner. While studies evaluating the impact of adversarial attacks on monocular depth estimation exist, a systematic demonstration and analysis of adversarial perturbations against pose estimation are lacking. We show how additive imperceptible perturbations can not only change predictions to increase the trajectory drift but also catastrophically alter its geometry. We also study the relation between adversarial perturbations targeting monocular depth and pose estimation networks, as well as the transferability of perturbations to other networks with different architectures and losses. Our experiments show how the generated perturbations lead to notable errors in relative rotation and translation predictions and elucidate vulnerabilities of the networks.

摘要: 深度学习的进步导致了计算机视觉的稳步发展，提高了目标检测和语义分割等任务的准确性。然而，深度神经网络很容易受到敌意攻击，因此在可靠部署方面提出了挑战。在机器人和先进驾驶辅助系统的3D场景理解中，两项突出的任务是单目深度和姿势估计，它们通常是在无人监督的方式下一起学习的。虽然已有研究评估对抗性攻击对单目深度估计的影响，但缺乏针对姿态估计的对抗性扰动的系统论证和分析。我们展示了加性不可察觉的扰动不仅可以改变预测以增加轨迹漂移，而且还可以灾难性地改变其几何形状。我们还研究了针对单目深度的对抗扰动与姿态估计网络之间的关系，以及扰动在具有不同结构和损失的其他网络中的可传递性。我们的实验显示了产生的扰动如何导致相对旋转和平移预测的显著错误，并阐明了网络的脆弱性。



## **27. Susceptibility of Continual Learning Against Adversarial Attacks**

持续学习对敌意攻击的敏感性 cs.LG

18 pages, 13 figures

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.05225v3)

**Authors**: Hikmat Khan, Pir Masoom Shah, Syed Farhan Alam Zaidi, Saif ul Islam

**Abstracts**: The recent advances in continual (incremental or lifelong) learning have concentrated on the prevention of forgetting that can lead to catastrophic consequences, but there are two outstanding challenges that must be addressed. The first is the evaluation of the robustness of the proposed methods. The second is ensuring the security of learned tasks remains largely unexplored. This paper presents a comprehensive study of the susceptibility of the continually learned tasks (including both current and previously learned tasks) that are vulnerable to forgetting. Such vulnerability of tasks against adversarial attacks raises profound issues in data integrity and privacy. We consider all three scenarios (i.e, task-incremental leaning, domain-incremental learning and class-incremental learning) of continual learning and explore three regularization-based experiments, three replay-based experiments, and one hybrid technique based on the reply and exemplar approach. We examine the robustness of these methods. In particular, we consider cases where we demonstrate that any class belonging to the current or previously learned tasks is prone to misclassification. Our observations, we identify potential limitations in continual learning approaches against adversarial attacks. Our empirical study recommends that the research community consider the robustness of the proposed continual learning approaches and invest extensive efforts in mitigating catastrophic forgetting.

摘要: 持续(增量或终身)学习的最新进展集中在防止可能导致灾难性后果的遗忘上，但有两个突出的挑战必须解决。首先是对所提出方法的稳健性进行评估。第二，确保学习任务的安全性在很大程度上仍未得到探索。本文对易被遗忘的持续学习任务(包括当前学习任务和先前学习任务)的易感性进行了全面的研究。任务对对手攻击的这种脆弱性引发了数据完整性和隐私方面的严重问题。我们考虑了持续学习的三种情景(任务增量式学习、领域增量式学习和班级增量式学习)，探索了三种基于正则化的实验、三种基于回放的实验以及一种基于回复和样例的混合技术。我们检验了这些方法的稳健性。特别是，我们考虑了这样的情况，即我们证明属于当前或以前学习的任务的任何类都容易发生错误分类。根据我们的观察，我们确定了针对敌意攻击的持续学习方法的潜在局限性。我们的实证研究建议研究界考虑所提出的持续学习方法的稳健性，并投入广泛的努力来缓解灾难性遗忘。



## **28. Adversarial Examples for Model-Based Control: A Sensitivity Analysis**

基于模型控制的对抗性实例：敏感度分析 eess.SY

Submission to the 58th Annual Allerton Conference on Communication,  Control, and Computing

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06982v1)

**Authors**: Po-han Li, Ufuk Topcu, Sandeep P. Chinchali

**Abstracts**: We propose a method to attack controllers that rely on external timeseries forecasts as task parameters. An adversary can manipulate the costs, states, and actions of the controllers by forging the timeseries, in this case perturbing the real timeseries. Since the controllers often encode safety requirements or energy limits in their costs and constraints, we refer to such manipulation as an adversarial attack. We show that different attacks on model-based controllers can increase control costs, activate constraints, or even make the control optimization problem infeasible. We use the linear quadratic regulator and convex model predictive controllers as examples of how adversarial attacks succeed and demonstrate the impact of adversarial attacks on a battery storage control task for power grid operators. As a result, our method increases control cost by $8500\%$ and energy constraints by $13\%$ on real electricity demand timeseries.

摘要: 我们提出了一种攻击依赖外部时间序列预测作为任务参数的控制器的方法。对手可以通过伪造时间序列来操纵控制器的成本、状态和操作，在这种情况下，会扰乱真实的时间序列。由于控制器经常在其成本和约束中编码安全要求或能量限制，我们将这种操纵称为对抗性攻击。我们证明了对基于模型的控制器的不同攻击会增加控制成本，激活约束，甚至使控制优化问题变得不可行。我们使用线性二次型调节器和凸模型预测控制器作为对抗性攻击如何成功的例子，并展示了对抗性攻击对电网运营商电池存储控制任务的影响。结果表明，该方法使实际电力需求时间序列的控制成本增加了8500美元，能源约束增加了13美元。



## **29. RSD-GAN: Regularized Sobolev Defense GAN Against Speech-to-Text Adversarial Attacks**

RSD-GAN：正规化Sobolev防御GAN防止语音到文本的对抗性攻击 cs.SD

Paper submitted to IEEE Signal Processing Letters Journal

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06858v1)

**Authors**: Mohammad Esmaeilpour, Nourhene Chaalia, Patrick Cardinal

**Abstracts**: This paper introduces a new synthesis-based defense algorithm for counteracting with a varieties of adversarial attacks developed for challenging the performance of the cutting-edge speech-to-text transcription systems. Our algorithm implements a Sobolev-based GAN and proposes a novel regularizer for effectively controlling over the functionality of the entire generative model, particularly the discriminator network during training. Our achieved results upon carrying out numerous experiments on the victim DeepSpeech, Kaldi, and Lingvo speech transcription systems corroborate the remarkable performance of our defense approach against a comprehensive range of targeted and non-targeted adversarial attacks.

摘要: 本文介绍了一种新的基于合成的防御算法，用于对抗各种针对尖端语音到文本转录系统性能的挑战而开发的对抗性攻击。我们的算法实现了一种基于Sobolev的GAN，并提出了一种新的正则化算法来有效地控制整个生成模型的功能，特别是在训练过程中的鉴别器网络。我们在受害者DeepSpeech、Kaldi和Lingvo语音转录系统上进行的大量实验所取得的结果证实了我们的防御方法在应对全面的定向和非定向对手攻击方面的卓越表现。



## **30. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

AGIC：联邦学习中的近似梯度反转攻击 cs.LG

This paper is accepted at the 41st International Symposium on  Reliable Distributed Systems (SRDS 2022)

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2204.13784v3)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.

摘要: 联合学习是一种私人设计的分布式学习范例，其中客户端在中央服务器聚合其本地更新以计算全局模型之前，根据自己的数据训练本地模型。根据所使用的聚合方法，局部更新要么是局部学习模型的梯度，要么是局部学习模型的权重。最近的重建攻击将梯度倒置优化应用于单个小批量的梯度更新，以重建客户在训练期间使用的私有数据。由于最新的重建攻击只关注单个更新，因此忽略了现实的对抗性场景，例如跨多个更新的观察和从多个小批次训练的更新。一些研究考虑了一种更具挑战性的对抗性场景，其中只能观察到基于多个小批次的模型更新，并求助于计算代价高昂的模拟来解开每个局部步骤的潜在样本。在本文中，我们提出了AGIC，一种新的近似梯度反转攻击，它可以高效地从模型或梯度更新中重建图像，并跨越多个历元。简而言之，AGIC(I)根据模型更新近似使用的训练样本的梯度更新以避免昂贵的模拟过程，(Ii)利用从多个历元收集的梯度/模型更新，以及(Iii)为重建质量向层分配相对于神经网络结构的不断增加的权重。我们在三个数据集CIFAR-10、CIFAR-100和ImageNet上对AGIC进行了广泛的评估。实验结果表明，与两种典型的梯度反转攻击相比，AGIC的峰值信噪比(PSNR)提高了50%。此外，AGIC比最先进的基于模拟的攻击速度更快，例如，在模型更新之间有8个本地步骤的情况下，攻击FedAvg的速度要快5倍。



## **31. Superclass Adversarial Attack**

超类对抗性攻击 cs.CV

ICML Workshop 2022 on Adversarial Machine Learning Frontiers

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2205.14629v2)

**Authors**: Soichiro Kumano, Hiroshi Kera, Toshihiko Yamasaki

**Abstracts**: Adversarial attacks have only focused on changing the predictions of the classifier, but their danger greatly depends on how the class is mistaken. For example, when an automatic driving system mistakes a Persian cat for a Siamese cat, it is hardly a problem. However, if it mistakes a cat for a 120km/h minimum speed sign, serious problems can arise. As a stepping stone to more threatening adversarial attacks, we consider the superclass adversarial attack, which causes misclassification of not only fine classes, but also superclasses. We conducted the first comprehensive analysis of superclass adversarial attacks (an existing and 19 new methods) in terms of accuracy, speed, and stability, and identified several strategies to achieve better performance. Although this study is aimed at superclass misclassification, the findings can be applied to other problem settings involving multiple classes, such as top-k and multi-label classification attacks.

摘要: 对抗性攻击只专注于改变分类器的预测，但它们的危险在很大程度上取决于类的错误程度。例如，当自动驾驶系统将波斯猫误认为暹罗猫时，这几乎不是问题。然而，如果它把猫错当成120公里/小时的最低速度标志，可能会出现严重的问题。作为更具威胁性的对抗性攻击的垫脚石，我们认为超类对抗性攻击不仅会导致细类的错误分类，而且会导致超类的错误分类。我们首次对超类对抗性攻击(现有方法和19种新方法)在准确性、速度和稳定性方面进行了全面分析，并确定了几种实现更好性能的策略。虽然这项研究是针对超类错误分类的，但研究结果也适用于其他涉及多类的问题，如top-k和多标签分类攻击。



## **32. Adversarially-Aware Robust Object Detector**

对抗性感知的鲁棒目标检测器 cs.CV

ECCV2022 oral paper

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06202v2)

**Authors**: Ziyi Dong, Pengxu Wei, Liang Lin

**Abstracts**: Object detection, as a fundamental computer vision task, has achieved a remarkable progress with the emergence of deep neural networks. Nevertheless, few works explore the adversarial robustness of object detectors to resist adversarial attacks for practical applications in various real-world scenarios. Detectors have been greatly challenged by unnoticeable perturbation, with sharp performance drop on clean images and extremely poor performance on adversarial images. In this work, we empirically explore the model training for adversarial robustness in object detection, which greatly attributes to the conflict between learning clean images and adversarial images. To mitigate this issue, we propose a Robust Detector (RobustDet) based on adversarially-aware convolution to disentangle gradients for model learning on clean and adversarial images. RobustDet also employs the Adversarial Image Discriminator (AID) and Consistent Features with Reconstruction (CFR) to ensure a reliable robustness. Extensive experiments on PASCAL VOC and MS-COCO demonstrate that our model effectively disentangles gradients and significantly enhances the detection robustness with maintaining the detection ability on clean images.

摘要: 随着深度神经网络的出现，目标检测作为一项基本的计算机视觉任务已经取得了显著的进展。然而，很少有研究探讨对象检测器在各种真实场景中的实际应用中抵抗对手攻击的对抗性健壮性。检测器受到了不可察觉的扰动的极大挑战，在干净图像上的性能急剧下降，在对抗性图像上的性能极差。在这项工作中，我们经验性地探索了目标检测中对抗鲁棒性的模型训练，这在很大程度上归因于学习干净图像和对抗图像之间的冲突。为了缓解这一问题，我们提出了一种基于对抗性感知卷积的稳健检测器(RobustDet)，用于在干净图像和对抗性图像上进行模型学习。RobustDet还采用了对抗性图像鉴别器(AID)和重建一致特征(CFR)，以确保可靠的健壮性。在PASCAL、VOC和MS-COCO上的大量实验表明，该模型在保持对干净图像的检测能力的同时，有效地解开了梯度的纠缠，显著提高了检测的鲁棒性。



## **33. PIAT: Physics Informed Adversarial Training for Solving Partial Differential Equations**

PIAT：解偏微分方程解的物理对抗性训练 cs.LG

**SubmitDate**: 2022-07-14    [paper-pdf](http://arxiv.org/pdf/2207.06647v1)

**Authors**: Simin Shekarpaz, Mohammad Azizmalayeri, Mohammad Hossein Rohban

**Abstracts**: In this paper, we propose the physics informed adversarial training (PIAT) of neural networks for solving nonlinear differential equations (NDE). It is well-known that the standard training of neural networks results in non-smooth functions. Adversarial training (AT) is an established defense mechanism against adversarial attacks, which could also help in making the solution smooth. AT include augmenting the training mini-batch with a perturbation that makes the network output mismatch the desired output adversarially. Unlike formal AT, which relies only on the training data, here we encode the governing physical laws in the form of nonlinear differential equations using automatic differentiation in the adversarial network architecture. We compare PIAT with PINN to indicate the effectiveness of our method in solving NDEs for up to 10 dimensions. Moreover, we propose weight decay and Gaussian smoothing to demonstrate the PIAT advantages. The code repository is available at https://github.com/rohban-lab/PIAT.

摘要: 本文提出了求解非线性微分方程组的神经网络的物理知情对抗训练(PIAT)方法。众所周知，神经网络的标准训练会导致函数的非光滑。对抗性训练(AT)是一种针对对抗性攻击的既定防御机制，它也有助于使解决方案顺利进行。AT包括用使网络输出与期望输出相反地失配的扰动来扩充训练小批量。与仅依赖训练数据的形式AT不同，我们在对抗性网络结构中使用自动微分将支配物理定律编码为非线性微分方程组的形式。我们将PIAT和Pinn进行了比较，以表明我们的方法在求解高达10维的非定常方程方面的有效性。此外，我们提出了权重衰减和高斯平滑来展示PIAT的优势。代码存储库可在https://github.com/rohban-lab/PIAT.上找到



## **34. Exploring Adversarial Attacks and Defenses in Vision Transformers trained with DINO**

探索与恐龙一起训练的视觉变形金刚的对抗性攻击和防御 cs.CV

Workshop paper accepted at AdvML Frontiers (ICML 2022)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2206.06761v3)

**Authors**: Javier Rando, Nasib Naimi, Thomas Baumann, Max Mathys

**Abstracts**: This work conducts the first analysis on the robustness against adversarial attacks on self-supervised Vision Transformers trained using DINO. First, we evaluate whether features learned through self-supervision are more robust to adversarial attacks than those emerging from supervised learning. Then, we present properties arising for attacks in the latent space. Finally, we evaluate whether three well-known defense strategies can increase adversarial robustness in downstream tasks by only fine-tuning the classification head to provide robustness even in view of limited compute resources. These defense strategies are: Adversarial Training, Ensemble Adversarial Training and Ensemble of Specialized Networks.

摘要: 本文首次对使用Dino训练的自监督视觉转换器的抗敌意攻击能力进行了分析。首先，我们评估通过自我监督学习的特征是否比通过监督学习获得的特征对对手攻击更健壮。然后，我们给出了潜在空间中攻击产生的性质。最后，我们评估了三种著名的防御策略是否能够在下游任务中通过微调分类头来提高对手的健壮性，即使在计算资源有限的情况下也是如此。这些防御策略是：对抗性训练、系列性对抗性训练和专业网络系列化。



## **35. Interactive Machine Learning: A State of the Art Review**

交互式机器学习：最新进展 cs.LG

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06196v1)

**Authors**: Natnael A. Wondimu, Cédric Buche, Ubbo Visser

**Abstracts**: Machine learning has proved useful in many software disciplines, including computer vision, speech and audio processing, natural language processing, robotics and some other fields. However, its applicability has been significantly hampered due its black-box nature and significant resource consumption. Performance is achieved at the expense of enormous computational resource and usually compromising the robustness and trustworthiness of the model. Recent researches have been identifying a lack of interactivity as the prime source of these machine learning problems. Consequently, interactive machine learning (iML) has acquired increased attention of researchers on account of its human-in-the-loop modality and relatively efficient resource utilization. Thereby, a state-of-the-art review of interactive machine learning plays a vital role in easing the effort toward building human-centred models. In this paper, we provide a comprehensive analysis of the state-of-the-art of iML. We analyze salient research works using merit-oriented and application/task oriented mixed taxonomy. We use a bottom-up clustering approach to generate a taxonomy of iML research works. Research works on adversarial black-box attacks and corresponding iML based defense system, exploratory machine learning, resource constrained learning, and iML performance evaluation are analyzed under their corresponding theme in our merit-oriented taxonomy. We have further classified these research works into technical and sectoral categories. Finally, research opportunities that we believe are inspiring for future work in iML are discussed thoroughly.

摘要: 机器学习已被证明在许多软件学科中都很有用，包括计算机视觉、语音和音频处理、自然语言处理、机器人学和其他一些领域。然而，由于其黑箱性质和巨大的资源消耗，其适用性受到了极大的阻碍。性能的实现是以牺牲巨大的计算资源为代价的，并且通常会损害模型的健壮性和可信性。最近的研究已经确定缺乏互动性是这些机器学习问题的主要来源。因此，交互式机器学习(IML)以其人在环中的方式和相对高效的资源利用而受到越来越多的研究者的关注。因此，对交互式机器学习的最新回顾在减轻建立以人为中心的模型的努力方面发挥着至关重要的作用。在本文中，我们对iML的最新发展进行了全面的分析。我们使用面向价值的和面向应用/任务的混合分类来分析重要的研究作品。我们使用自下而上的聚类方法来生成iML研究作品的分类。在我们的价值导向分类法中，对抗性黑盒攻击和相应的基于iML的防御系统、探索性机器学习、资源受限学习和iML性能评估的研究工作都在相应的主题下进行了分析。我们进一步将这些研究工作分为技术和部门两个类别。最后，深入讨论了我们认为对iML未来工作有启发的研究机会。



## **36. On the Robustness of Bayesian Neural Networks to Adversarial Attacks**

贝叶斯神经网络对敌方攻击的稳健性研究 cs.LG

arXiv admin note: text overlap with arXiv:2002.04359

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06154v1)

**Authors**: Luca Bortolussi, Ginevra Carbone, Luca Laurenti, Andrea Patane, Guido Sanguinetti, Matthew Wicker

**Abstracts**: Vulnerability to adversarial attacks is one of the principal hurdles to the adoption of deep learning in safety-critical applications. Despite significant efforts, both practical and theoretical, training deep learning models robust to adversarial attacks is still an open problem. In this paper, we analyse the geometry of adversarial attacks in the large-data, overparameterized limit for Bayesian Neural Networks (BNNs). We show that, in the limit, vulnerability to gradient-based attacks arises as a result of degeneracy in the data distribution, i.e., when the data lies on a lower-dimensional submanifold of the ambient space. As a direct consequence, we demonstrate that in this limit BNN posteriors are robust to gradient-based adversarial attacks. Crucially, we prove that the expected gradient of the loss with respect to the BNN posterior distribution is vanishing, even when each neural network sampled from the posterior is vulnerable to gradient-based attacks. Experimental results on the MNIST, Fashion MNIST, and half moons datasets, representing the finite data regime, with BNNs trained with Hamiltonian Monte Carlo and Variational Inference, support this line of arguments, showing that BNNs can display both high accuracy on clean data and robustness to both gradient-based and gradient-free based adversarial attacks.

摘要: 对敌意攻击的脆弱性是在安全关键应用中采用深度学习的主要障碍之一。尽管在实践和理论上都做了大量的努力，但训练对对手攻击稳健的深度学习模型仍然是一个悬而未决的问题。本文分析了贝叶斯神经网络(BNN)在大数据、过参数限制下的攻击几何。我们证明，在极限情况下，由于数据分布的退化，即当数据位于环境空间的低维子流形上时，对基于梯度的攻击的脆弱性出现。作为一个直接的推论，我们证明了在这个极限下，BNN后验网络对基于梯度的敌意攻击是稳健的。重要的是，我们证明了损失相对于BNN后验分布的期望梯度是零的，即使从后验采样的每个神经网络都容易受到基于梯度的攻击。在代表有限数据区的MNIST、Fashion MNIST和半月数据集上的实验结果支持这一论点，BNN采用哈密顿蒙特卡罗和变分推理进行训练，表明BNN在干净数据上具有很高的准确率，并且对基于梯度和基于无梯度的敌意攻击都具有很好的鲁棒性。



## **37. Neural Network Robustness as a Verification Property: A Principled Case Study**

作为验证属性的神经网络健壮性：原则性案例研究 cs.LG

11 pages, CAV 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2104.01396v2)

**Authors**: Marco Casadio, Ekaterina Komendantskaya, Matthew L. Daggitt, Wen Kokke, Guy Katz, Guy Amir, Idan Refaeli

**Abstracts**: Neural networks are very successful at detecting patterns in noisy data, and have become the technology of choice in many fields. However, their usefulness is hampered by their susceptibility to adversarial attacks. Recently, many methods for measuring and improving a network's robustness to adversarial perturbations have been proposed, and this growing body of research has given rise to numerous explicit or implicit notions of robustness. Connections between these notions are often subtle, and a systematic comparison between them is missing in the literature. In this paper we begin addressing this gap, by setting up general principles for the empirical analysis and evaluation of a network's robustness as a mathematical property - during the network's training phase, its verification, and after its deployment. We then apply these principles and conduct a case study that showcases the practical benefits of our general approach.

摘要: 神经网络在检测噪声数据中的模式方面非常成功，已经成为许多领域的首选技术。然而，由于它们容易受到对抗性攻击，它们的有用性受到了阻碍。最近，已经提出了许多方法来衡量和提高网络对敌意干扰的稳健性，并且这一不断增长的研究已经产生了许多显式或隐式的健壮性概念。这些概念之间的联系往往很微妙，文献中也没有对它们进行系统的比较。在本文中，我们开始解决这一差距，通过建立一般原则，将网络的稳健性作为一种数学属性进行经验分析和评估--在网络的训练阶段、验证阶段和部署之后。然后，我们应用这些原则并进行案例研究，展示我们一般方法的实际好处。



## **38. Perturbation Inactivation Based Adversarial Defense for Face Recognition**

基于扰动失活的人脸识别对抗性防御 cs.CV

Accepted by IEEE Transactions on Information Forensics & Security  (T-IFS)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.06035v1)

**Authors**: Min Ren, Yuhao Zhu, Yunlong Wang, Zhenan Sun

**Abstracts**: Deep learning-based face recognition models are vulnerable to adversarial attacks. To curb these attacks, most defense methods aim to improve the robustness of recognition models against adversarial perturbations. However, the generalization capacities of these methods are quite limited. In practice, they are still vulnerable to unseen adversarial attacks. Deep learning models are fairly robust to general perturbations, such as Gaussian noises. A straightforward approach is to inactivate the adversarial perturbations so that they can be easily handled as general perturbations. In this paper, a plug-and-play adversarial defense method, named perturbation inactivation (PIN), is proposed to inactivate adversarial perturbations for adversarial defense. We discover that the perturbations in different subspaces have different influences on the recognition model. There should be a subspace, called the immune space, in which the perturbations have fewer adverse impacts on the recognition model than in other subspaces. Hence, our method estimates the immune space and inactivates the adversarial perturbations by restricting them to this subspace. The proposed method can be generalized to unseen adversarial perturbations since it does not rely on a specific kind of adversarial attack method. This approach not only outperforms several state-of-the-art adversarial defense methods but also demonstrates a superior generalization capacity through exhaustive experiments. Moreover, the proposed method can be successfully applied to four commercial APIs without additional training, indicating that it can be easily generalized to existing face recognition systems. The source code is available at https://github.com/RenMin1991/Perturbation-Inactivate

摘要: 基于深度学习的人脸识别模型容易受到敌意攻击。为了遏制这些攻击，大多数防御方法的目的是提高识别模型对对手扰动的稳健性。然而，这些方法的泛化能力相当有限。在实践中，他们仍然容易受到看不见的对手攻击。深度学习模型对一般扰动具有较强的鲁棒性，如高斯噪声。一种简单的方法是停用对抗性扰动，这样它们就可以很容易地作为一般扰动来处理。本文提出了一种即插即用的对抗防御方法，称为扰动失活(PIN)，用于灭活对抗防御中的对抗扰动。我们发现，不同子空间中的扰动对识别模型有不同的影响。应该有一个称为免疫空间的子空间，在这个子空间中，扰动对识别模型的不利影响比在其他子空间中要小。因此，我们的方法估计免疫空间，并通过将对抗性扰动限制在此子空间来使其失活。由于该方法不依赖于一种特定的对抗性攻击方法，因此可以推广到不可见的对抗性扰动。通过详尽的实验证明，该方法不仅比目前最先进的对抗性防御方法有更好的性能，而且具有更好的泛化能力。此外，该方法可以成功地应用于四个商业API，而无需额外的训练，这表明它可以很容易地推广到现有的人脸识别系统中。源代码可在https://github.com/RenMin1991/Perturbation-Inactivate上找到



## **39. BadHash: Invisible Backdoor Attacks against Deep Hashing with Clean Label**

BadHash：使用Clean Label对深度哈希进行隐形后门攻击 cs.CV

This paper has been accepted by the 30th ACM International Conference  on Multimedia (MM '22, October 10--14, 2022, Lisboa, Portugal)

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.00278v3)

**Authors**: Shengshan Hu, Ziqi Zhou, Yechao Zhang, Leo Yu Zhang, Yifeng Zheng, Yuanyuan HE, Hai Jin

**Abstracts**: Due to its powerful feature learning capability and high efficiency, deep hashing has achieved great success in large-scale image retrieval. Meanwhile, extensive works have demonstrated that deep neural networks (DNNs) are susceptible to adversarial examples, and exploring adversarial attack against deep hashing has attracted many research efforts. Nevertheless, backdoor attack, another famous threat to DNNs, has not been studied for deep hashing yet. Although various backdoor attacks have been proposed in the field of image classification, existing approaches failed to realize a truly imperceptive backdoor attack that enjoys invisible triggers and clean label setting simultaneously, and they also cannot meet the intrinsic demand of image retrieval backdoor. In this paper, we propose BadHash, the first generative-based imperceptible backdoor attack against deep hashing, which can effectively generate invisible and input-specific poisoned images with clean label. Specifically, we first propose a new conditional generative adversarial network (cGAN) pipeline to effectively generate poisoned samples. For any given benign image, it seeks to generate a natural-looking poisoned counterpart with a unique invisible trigger. In order to improve the attack effectiveness, we introduce a label-based contrastive learning network LabCLN to exploit the semantic characteristics of different labels, which are subsequently used for confusing and misleading the target model to learn the embedded trigger. We finally explore the mechanism of backdoor attacks on image retrieval in the hash space. Extensive experiments on multiple benchmark datasets verify that BadHash can generate imperceptible poisoned samples with strong attack ability and transferability over state-of-the-art deep hashing schemes.

摘要: 深度哈希法由于其强大的特征学习能力和高效的检索效率，在大规模图像检索中取得了巨大的成功。同时，大量的研究表明，深度神经网络(DNN)容易受到敌意例子的影响，探索针对深度散列的敌意攻击吸引了许多研究努力。然而，DNNS的另一个著名威胁--后门攻击，还没有被研究过深度散列。虽然在图像分类领域已经提出了各种各样的后门攻击，但现有的方法未能实现真正的隐蔽的、同时具有不可见触发器和干净标签设置的后门攻击，也不能满足图像检索的内在需求。本文提出了BadHash，这是第一个基于生成性的针对深度哈希的不可察觉的后门攻击，它可以有效地生成标签清晰的不可见和输入特定的有毒图像。具体地说，我们首先提出了一种新的条件生成对抗网络(CGAN)管道来有效地生成有毒样本。对于任何给定的良性形象，它都试图生成一个看起来自然、有毒的形象，并带有独特的无形触发器。为了提高攻击的有效性，我们引入了一个基于标签的对比学习网络LabCLN来利用不同标签的语义特征，这些语义特征被用来混淆和误导目标模型学习嵌入的触发器。最后，我们探讨了哈希空间中后门攻击对图像检索的影响机制。在多个基准数据集上的大量实验证明，BadHash可以生成不可察觉的有毒样本，具有很强的攻击能力和可转移性，优于最先进的深度哈希方案。



## **40. Physical Backdoor Attacks to Lane Detection Systems in Autonomous Driving**

自动驾驶中车道检测系统的物理后门攻击 cs.CV

Accepted by ACM MultiMedia 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2203.00858v2)

**Authors**: Xingshuo Han, Guowen Xu, Yuan Zhou, Xuehuan Yang, Jiwei Li, Tianwei Zhang

**Abstracts**: Modern autonomous vehicles adopt state-of-the-art DNN models to interpret the sensor data and perceive the environment. However, DNN models are vulnerable to different types of adversarial attacks, which pose significant risks to the security and safety of the vehicles and passengers. One prominent threat is the backdoor attack, where the adversary can compromise the DNN model by poisoning the training samples. Although lots of effort has been devoted to the investigation of the backdoor attack to conventional computer vision tasks, its practicality and applicability to the autonomous driving scenario is rarely explored, especially in the physical world.   In this paper, we target the lane detection system, which is an indispensable module for many autonomous driving tasks, e.g., navigation, lane switching. We design and realize the first physical backdoor attacks to such system. Our attacks are comprehensively effective against different types of lane detection algorithms. Specifically, we introduce two attack methodologies (poison-annotation and clean-annotation) to generate poisoned samples. With those samples, the trained lane detection model will be infected with the backdoor, and can be activated by common objects (e.g., traffic cones) to make wrong detections, leading the vehicle to drive off the road or onto the opposite lane. Extensive evaluations on public datasets and physical autonomous vehicles demonstrate that our backdoor attacks are effective, stealthy and robust against various defense solutions. Our codes and experimental videos can be found in https://sites.google.com/view/lane-detection-attack/lda.

摘要: 现代自动驾驶汽车采用最先进的DNN模型来解释传感器数据和感知环境。然而，DNN模型容易受到不同类型的对抗性攻击，这对车辆和乘客的安全和安全构成了重大风险。一个突出的威胁是后门攻击，在这种攻击中，对手可以通过毒化训练样本来危害DNN模型。虽然对传统计算机视觉任务的后门攻击已经投入了大量的精力来研究，但很少有人探索它在自动驾驶场景中的实用性和适用性，特别是在物理世界中。本文以车道检测系统为研究对象，车道检测系统是导航、车道切换等自动驾驶任务中不可缺少的模块。我们设计并实现了对此类系统的第一次物理后门攻击。我们的攻击对不同类型的车道检测算法都是全面有效的。具体地说，我们引入了两种攻击方法(毒注解和干净注解)来生成中毒样本。利用这些样本，训练后的车道检测模型将被后门感染，并可能被常见对象(如交通锥体)激活以进行错误检测，导致车辆驶离道路或进入对面车道。对公共数据集和物理自动驾驶车辆的广泛评估表明，我们的后门攻击针对各种防御解决方案是有效的、隐蔽的和健壮的。我们的代码和实验视频可在https://sites.google.com/view/lane-detection-attack/lda.中找到



## **41. PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch**

PatchZero：通过检测和归零补丁来防御敌意补丁攻击 cs.CV

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.01795v2)

**Authors**: Ke Xu, Yao Xiao, Zhaoheng Zheng, Kaijie Cai, Ram Nevatia

**Abstracts**: Adversarial patch attacks mislead neural networks by injecting adversarial pixels within a local region. Patch attacks can be highly effective in a variety of tasks and physically realizable via attachment (e.g. a sticker) to the real-world objects. Despite the diversity in attack patterns, adversarial patches tend to be highly textured and different in appearance from natural images. We exploit this property and present PatchZero, a general defense pipeline against white-box adversarial patches without retraining the downstream classifier or detector. Specifically, our defense detects adversaries at the pixel-level and "zeros out" the patch region by repainting with mean pixel values. We further design a two-stage adversarial training scheme to defend against the stronger adaptive attacks. PatchZero achieves SOTA defense performance on the image classification (ImageNet, RESISC45), object detection (PASCAL VOC), and video classification (UCF101) tasks with little degradation in benign performance. In addition, PatchZero transfers to different patch shapes and attack types.

摘要: 对抗性补丁攻击通过在局部区域内注入对抗性像素来误导神经网络。补丁攻击可以在各种任务中非常有效，并且可以通过附着(例如贴纸)到真实世界的对象来物理实现。尽管攻击模式多种多样，但敌方补丁往往纹理丰富，外观与自然图像不同。我们利用这一特性，提出了PatchZero，一种针对白盒恶意补丁的通用防御管道，而不需要重新训练下游的分类器或检测器。具体地说，我们的防御在像素级检测对手，并通过使用平均像素值重新绘制来对补丁区域进行“清零”。我们进一步设计了一种两阶段对抗性训练方案，以抵御更强的适应性攻击。PatchZero在图像分类(ImageNet，RESISC45)、目标检测(Pascal VOC)和视频分类(UCF101)任务上实现了SOTA防御性能，性能良好，性能几乎没有下降。此外，PatchZero还可以转换为不同的补丁形状和攻击类型。



## **42. Game of Trojans: A Submodular Byzantine Approach**

特洛伊木马游戏：一种拜占庭式的子模块方法 cs.LG

Submitted to GameSec 2022

**SubmitDate**: 2022-07-13    [paper-pdf](http://arxiv.org/pdf/2207.05937v1)

**Authors**: Dinuka Sahabandu, Arezoo Rajabi, Luyao Niu, Bo Li, Bhaskar Ramasubramanian, Radha Poovendran

**Abstracts**: Machine learning models in the wild have been shown to be vulnerable to Trojan attacks during training. Although many detection mechanisms have been proposed, strong adaptive attackers have been shown to be effective against them. In this paper, we aim to answer the questions considering an intelligent and adaptive adversary: (i) What is the minimal amount of instances required to be Trojaned by a strong attacker? and (ii) Is it possible for such an attacker to bypass strong detection mechanisms?   We provide an analytical characterization of adversarial capability and strategic interactions between the adversary and detection mechanism that take place in such models. We characterize adversary capability in terms of the fraction of the input dataset that can be embedded with a Trojan trigger. We show that the loss function has a submodular structure, which leads to the design of computationally efficient algorithms to determine this fraction with provable bounds on optimality. We propose a Submodular Trojan algorithm to determine the minimal fraction of samples to inject a Trojan trigger. To evade detection of the Trojaned model, we model strategic interactions between the adversary and Trojan detection mechanism as a two-player game. We show that the adversary wins the game with probability one, thus bypassing detection. We establish this by proving that output probability distributions of a Trojan model and a clean model are identical when following the Min-Max (MM) Trojan algorithm.   We perform extensive evaluations of our algorithms on MNIST, CIFAR-10, and EuroSAT datasets. The results show that (i) with Submodular Trojan algorithm, the adversary needs to embed a Trojan trigger into a very small fraction of samples to achieve high accuracy on both Trojan and clean samples, and (ii) the MM Trojan algorithm yields a trained Trojan model that evades detection with probability 1.

摘要: 野外的机器学习模型已被证明在训练期间容易受到特洛伊木马的攻击。虽然已经提出了许多检测机制，但强自适应攻击者被证明对它们是有效的。在本文中，我们的目标是考虑一个智能和自适应的对手来回答以下问题：(I)强攻击者需要木马的最小实例数量是多少？以及(Ii)这样的攻击者是否有可能绕过强大的检测机制？我们提供了发生在这样的模型中的对手能力和对手与检测机制之间的战略交互的分析特征。我们根据可以嵌入特洛伊木马触发器的输入数据集的比例来表征攻击者的能力。我们证明了损失函数具有子模结构，这导致设计了计算效率高的算法来确定具有可证明的最优界的分数。我们提出了一种子模块木马算法来确定注入木马触发器的最小样本比例。为了逃避木马模型的检测，我们将对手和木马检测机制之间的战略交互建模为两人博弈。我们证明了对手以概率1赢得比赛，从而绕过了检测。我们证明了当遵循Min-Max(MM)木马算法时，木马模型和CLEAN模型的输出概率分布是相同的。我们在MNIST、CIFAR-10和EuroSAT数据集上对我们的算法进行了广泛的评估。结果表明：(I)利用子模块木马算法，攻击者需要在很小一部分样本中嵌入木马触发器，以获得对木马和干净样本的高精度；(Ii)MM木马算法生成一个以1的概率逃避检测的训练有素的木马模型。



## **43. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Predictions**

一句话抵得上一千美元：对推特傻瓜股预测的敌意攻击 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2205.01094v3)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **44. Practical Attacks on Machine Learning: A Case Study on Adversarial Windows Malware**

对机器学习的实用攻击：恶意Windows恶意软件的案例研究 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05548v1)

**Authors**: Luca Demetrio, Battista Biggio, Fabio Roli

**Abstracts**: While machine learning is vulnerable to adversarial examples, it still lacks systematic procedures and tools for evaluating its security in different application contexts. In this article, we discuss how to develop automated and scalable security evaluations of machine learning using practical attacks, reporting a use case on Windows malware detection.

摘要: 虽然机器学习很容易受到敌意例子的影响，但它仍然缺乏系统的程序和工具来评估其在不同应用环境中的安全性。在本文中，我们讨论了如何使用实际攻击来开发自动化和可扩展的机器学习安全评估，并报告了一个Windows恶意软件检测的用例。



## **45. Improving the Robustness and Generalization of Deep Neural Network with Confidence Threshold Reduction**

降低置信度阈值提高深度神经网络的鲁棒性和泛化能力 cs.LG

Under review

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2206.00913v2)

**Authors**: Xiangyuan Yang, Jie Lin, Hanlin Zhang, Xinyu Yang, Peng Zhao

**Abstracts**: Deep neural networks are easily attacked by imperceptible perturbation. Presently, adversarial training (AT) is the most effective method to enhance the robustness of the model against adversarial examples. However, because adversarial training solved a min-max value problem, in comparison with natural training, the robustness and generalization are contradictory, i.e., the robustness improvement of the model will decrease the generalization of the model. To address this issue, in this paper, a new concept, namely confidence threshold (CT), is introduced and the reducing of the confidence threshold, known as confidence threshold reduction (CTR), is proven to improve both the generalization and robustness of the model. Specifically, to reduce the CT for natural training (i.e., for natural training with CTR), we propose a mask-guided divergence loss function (MDL) consisting of a cross-entropy loss term and an orthogonal term. The empirical and theoretical analysis demonstrates that the MDL loss improves the robustness and generalization of the model simultaneously for natural training. However, the model robustness improvement of natural training with CTR is not comparable to that of adversarial training. Therefore, for adversarial training, we propose a standard deviation loss function (STD), which minimizes the difference in the probabilities of the wrong categories, to reduce the CT by being integrated into the loss function of adversarial training. The empirical and theoretical analysis demonstrates that the STD based loss function can further improve the robustness of the adversarially trained model on basis of guaranteeing the changeless or slight improvement of the natural accuracy.

摘要: 深层神经网络很容易受到不可察觉的扰动的攻击。目前，对抗性训练(AT)是提高模型对对抗性例子的稳健性的最有效方法。然而，由于对抗性训练解决了最小-最大值问题，与自然训练相比，鲁棒性和泛化是矛盾的，即模型的稳健性提高会降低模型的泛化能力。针对这一问题，本文引入了置信度阈值的概念，并证明了置信度阈值的降低既能提高模型的泛化能力，又能提高模型的鲁棒性。具体地说，为了减少自然训练(即具有CTR的自然训练)的CT，我们提出了一种掩模引导的发散损失函数(MDL)，该函数由交叉熵损失项和正交项组成。实验和理论分析表明，对于自然训练，MDL损失同时提高了模型的鲁棒性和泛化能力。然而，CTR自然训练对模型稳健性的改善与对抗性训练不可同日而语。因此，对于对抗性训练，我们提出了一种标准偏差损失函数(STD)，它最小化了错误类别概率的差异，通过将其整合到对抗性训练的损失函数中来降低CT。实证和理论分析表明，基于STD的损失函数可以在保证自然精度不变或略有提高的基础上，进一步提高对抗性训练模型的稳健性。



## **46. Adversarial Robustness Assessment of NeuroEvolution Approaches**

神经进化方法的对抗性稳健性评估 cs.NE

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05451v1)

**Authors**: Inês Valentim, Nuno Lourenço, Nuno Antunes

**Abstracts**: NeuroEvolution automates the generation of Artificial Neural Networks through the application of techniques from Evolutionary Computation. The main goal of these approaches is to build models that maximize predictive performance, sometimes with an additional objective of minimizing computational complexity. Although the evolved models achieve competitive results performance-wise, their robustness to adversarial examples, which becomes a concern in security-critical scenarios, has received limited attention. In this paper, we evaluate the adversarial robustness of models found by two prominent NeuroEvolution approaches on the CIFAR-10 image classification task: DENSER and NSGA-Net. Since the models are publicly available, we consider white-box untargeted attacks, where the perturbations are bounded by either the L2 or the Linfinity-norm. Similarly to manually-designed networks, our results show that when the evolved models are attacked with iterative methods, their accuracy usually drops to, or close to, zero under both distance metrics. The DENSER model is an exception to this trend, showing some resistance under the L2 threat model, where its accuracy only drops from 93.70% to 18.10% even with iterative attacks. Additionally, we analyzed the impact of pre-processing applied to the data before the first layer of the network. Our observations suggest that some of these techniques can exacerbate the perturbations added to the original inputs, potentially harming robustness. Thus, this choice should not be neglected when automatically designing networks for applications where adversarial attacks are prone to occur.

摘要: 神经进化通过应用进化计算的技术自动生成人工神经网络。这些方法的主要目标是构建最大限度提高预测性能的模型，有时还有最小化计算复杂性的额外目标。虽然进化模型在性能方面达到了竞争的结果，但它们对敌意示例的健壮性受到了有限的关注，这在安全关键场景中成为一个令人担忧的问题。在本文中，我们评估了两种重要的神经进化方法在CIFAR-10图像分类任务中发现的模型的对抗性健壮性：Denser和NSGA-Net。由于模型是公开可用的，我们考虑白盒非目标攻击，其中扰动由L2或Linfinity范数有界。与人工设计的网络类似，我们的结果表明，当进化模型受到迭代方法的攻击时，在两种距离度量下，它们的精度通常下降到或接近于零。密度更高的模型是这一趋势的一个例外，在L2威胁模型下显示出一些阻力，即使在迭代攻击的情况下，其准确率也只从93.70%下降到18.10%。此外，我们还分析了在网络第一层之前应用预处理对数据的影响。我们的观察表明，其中一些技术可能会加剧添加到原始输入的扰动，潜在地损害稳健性。因此，在为容易发生对抗性攻击的应用程序自动设计网络时，这一选择不应被忽视。



## **47. A Security-aware and LUT-based CAD Flow for the Physical Synthesis of eASICs**

一种安全感知的基于查找表的eASIC物理综合CAD流程 cs.CR

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05413v1)

**Authors**: Zain UlAbideen, Tiago Diadami Perez, Mayler Martins, Samuel Pagliarini

**Abstracts**: Numerous threats are associated with the globalized integrated circuit (IC) supply chain, such as piracy, reverse engineering, overproduction, and malicious logic insertion. Many obfuscation approaches have been proposed to mitigate these threats by preventing an adversary from fully understanding the IC (or parts of it). The use of reconfigurable elements inside an IC is a known obfuscation technique, either as a coarse grain reconfigurable block (i.e., eFPGA) or as a fine grain element (i.e., FPGA-like look-up tables). This paper presents a security-aware CAD flow that is LUT-based yet still compatible with the standard cell based physical synthesis flow. More precisely, our CAD flow explores the FPGA-ASIC design space and produces heavily obfuscated designs where only small portions of the logic resemble an ASIC. Therefore, we term this specialized solution an "embedded ASIC" (eASIC). Nevertheless, even for heavily LUT-dominated designs, our proposed decomposition and pin swapping algorithms allow for performance gains that enable performance levels that only ASICs would otherwise achieve. On the security side, we have developed novel template-based attacks and also applied existing attacks, both oracle-free and oracle-based. Our security analysis revealed that the obfuscation rate for an SHA-256 study case should be at least 45% for withstanding traditional attacks and at least 80% for withstanding template-based attacks. When the 80\% obfuscated SHA-256 design is physically implemented, it achieves a remarkable frequency of 368MHz in a 65nm commercial technology, whereas its FPGA implementation (in a superior technology) achieves only 77MHz.

摘要: 与全球化集成电路(IC)供应链相关的许多威胁，如盗版、逆向工程、生产过剩和恶意逻辑插入。已经提出了许多模糊方法来通过阻止对手完全理解IC(或其部分)来缓解这些威胁。在IC内使用可重构元件是一种已知的混淆技术，或者作为粗粒度可重构块(即，eFPGA)，或者作为细粒度元件(即，类似于FPGA的查找表)。本文提出了一种安全感知的CAD流程，该流程是基于LUT的，但仍然与基于标准单元的物理综合流程兼容。更准确地说，我们的CAD流程探索了FPGA-ASIC设计空间，并产生了高度混淆的设计，其中只有一小部分逻辑类似于ASIC。因此，我们将这种专门的解决方案称为“嵌入式ASIC”(EASIC)。然而，即使对于以查找表为主的设计，我们建议的分解和管脚交换算法也可以实现性能提升，从而实现只有ASIC才能达到的性能水平。在安全方面，我们开发了新的基于模板的攻击，并应用了现有的攻击，包括无Oracle攻击和基于Oracle的攻击。我们的安全分析显示，对于SHA-256研究案例，对于抵抗传统攻击至少应该是45%，对于基于模板的攻击至少应该是80%。在实际实现80型混淆SHA-256设计时，它在65 nm的商用工艺下达到了368 MHz的显著频率，而它的FPGA实现(在更高的工艺下)只达到了77 MHz。



## **48. Frequency Domain Model Augmentation for Adversarial Attack**

对抗性攻击的频域模型增强 cs.CV

Accepted by ECCV 2022

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05382v1)

**Authors**: Yuyang Long, Qilong Zhang, Boheng Zeng, Lianli Gao, Xianglong Liu, Jian Zhang, Jingkuan Song

**Abstracts**: For black-box attacks, the gap between the substitute model and the victim model is usually large, which manifests as a weak attack performance. Motivated by the observation that the transferability of adversarial examples can be improved by attacking diverse models simultaneously, model augmentation methods which simulate different models by using transformed images are proposed. However, existing transformations for spatial domain do not translate to significantly diverse augmented models. To tackle this issue, we propose a novel spectrum simulation attack to craft more transferable adversarial examples against both normally trained and defense models. Specifically, we apply a spectrum transformation to the input and thus perform the model augmentation in the frequency domain. We theoretically prove that the transformation derived from frequency domain leads to a diverse spectrum saliency map, an indicator we proposed to reflect the diversity of substitute models. Notably, our method can be generally combined with existing attacks. Extensive experiments on the ImageNet dataset demonstrate the effectiveness of our method, \textit{e.g.}, attacking nine state-of-the-art defense models with an average success rate of \textbf{95.4\%}. Our code is available in \url{https://github.com/yuyang-long/SSA}.

摘要: 对于黑盒攻击，替换模型与受害者模型之间的差距通常较大，表现为攻击性能较弱。基于同时攻击不同模型可以提高对抗性实例的可转移性这一观察结果，提出了利用变换后的图像模拟不同模型的模型增强方法。然而，现有的空间域变换并不能转化为显著不同的增强模型。为了解决这个问题，我们提出了一种新颖的频谱模拟攻击，以针对正常训练的模型和防御模型创建更多可转移的对抗性示例。具体地说，我们对输入应用频谱变换，从而在频域中执行模型增强。我们从理论上证明了从频域得到的变换导致了不同的频谱显著图，这是我们提出的反映替代模型多样性的一个指标。值得注意的是，我们的方法通常可以与现有攻击相结合。在ImageNet数据集上的大量实验证明了该方法的有效性，该方法攻击了9个最先进的防御模型，平均成功率为\textbf{95.4\}。我们的代码位于\url{https://github.com/yuyang-long/SSA}.



## **49. Bi-fidelity Evolutionary Multiobjective Search for Adversarially Robust Deep Neural Architectures**

双保真进化多目标搜索逆鲁棒深神经网络结构 cs.LG

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05321v1)

**Authors**: Jia Liu, Ran Cheng, Yaochu Jin

**Abstracts**: Deep neural networks have been found vulnerable to adversarial attacks, thus raising potentially concerns in security-sensitive contexts. To address this problem, recent research has investigated the adversarial robustness of deep neural networks from the architectural point of view. However, searching for architectures of deep neural networks is computationally expensive, particularly when coupled with adversarial training process. To meet the above challenge, this paper proposes a bi-fidelity multiobjective neural architecture search approach. First, we formulate the NAS problem for enhancing adversarial robustness of deep neural networks into a multiobjective optimization problem. Specifically, in addition to a low-fidelity performance predictor as the first objective, we leverage an auxiliary-objective -- the value of which is the output of a surrogate model trained with high-fidelity evaluations. Secondly, we reduce the computational cost by combining three performance estimation methods, i.e., parameter sharing, low-fidelity evaluation, and surrogate-based predictor. The effectiveness of the proposed approach is confirmed by extensive experiments conducted on CIFAR-10, CIFAR-100 and SVHN datasets.

摘要: 深度神经网络被发现容易受到敌意攻击，因此在安全敏感的环境中引发了潜在的担忧。为了解决这个问题，最近的研究从体系结构的角度研究了深度神经网络的对抗健壮性。然而，寻找深度神经网络的结构在计算上是昂贵的，特别是当与对抗性训练过程相结合时。为了应对上述挑战，本文提出了一种双保真多目标神经结构搜索方法。首先，我们将增强深层神经网络对抗健壮性的NAS问题转化为一个多目标优化问题。具体地说，除了作为第一个目标的低保真性能预测器之外，我们还利用一个辅助目标--其值是用高保真评估训练的代理模型的输出。其次，通过结合参数共享、低保真评估和基于代理的预测器三种性能估计方法来降低计算代价。在CIFAR-10、CIFAR-100和SVHN数据集上进行的大量实验证实了该方法的有效性。



## **50. Multitask Learning from Augmented Auxiliary Data for Improving Speech Emotion Recognition**

基于增强辅助数据的多任务学习改进语音情感识别 cs.SD

Under review IEEE Transactions on Affective Computing

**SubmitDate**: 2022-07-12    [paper-pdf](http://arxiv.org/pdf/2207.05298v1)

**Authors**: Siddique Latif, Rajib Rana, Sara Khalifa, Raja Jurdak, Björn W. Schuller

**Abstracts**: Despite the recent progress in speech emotion recognition (SER), state-of-the-art systems lack generalisation across different conditions. A key underlying reason for poor generalisation is the scarcity of emotion datasets, which is a significant roadblock to designing robust machine learning (ML) models. Recent works in SER focus on utilising multitask learning (MTL) methods to improve generalisation by learning shared representations. However, most of these studies propose MTL solutions with the requirement of meta labels for auxiliary tasks, which limits the training of SER systems. This paper proposes an MTL framework (MTL-AUG) that learns generalised representations from augmented data. We utilise augmentation-type classification and unsupervised reconstruction as auxiliary tasks, which allow training SER systems on augmented data without requiring any meta labels for auxiliary tasks. The semi-supervised nature of MTL-AUG allows for the exploitation of the abundant unlabelled data to further boost the performance of SER. We comprehensively evaluate the proposed framework in the following settings: (1) within corpus, (2) cross-corpus and cross-language, (3) noisy speech, (4) and adversarial attacks. Our evaluations using the widely used IEMOCAP, MSP-IMPROV, and EMODB datasets show improved results compared to existing state-of-the-art methods.

摘要: 尽管最近在语音情感识别(SER)方面取得了进展，但最先进的系统缺乏对不同条件的通用性。泛化能力差的一个关键根本原因是情感数据集的稀缺，这是设计健壮的机器学习(ML)模型的一个重要障碍。SER最近的工作集中在利用多任务学习(MTL)方法通过学习共享表示来提高泛化能力。然而，这些研究大多提出了辅助任务需要元标签的MTL解决方案，这限制了SER系统的训练。提出了一个从扩充数据中学习泛化表示的MTL框架(MTL-AUG)。我们使用增强型分类和无监督重建作为辅助任务，允许在增强型数据上训练SER系统，而不需要任何辅助任务的元标签。MTL-AUG的半监督性质允许利用丰富的未标记数据来进一步提高SER的性能。我们在以下几个方面对该框架进行了综合评估：(1)在语料库内，(2)跨语料库和跨语言，(3)噪声语音，(4)和对抗性攻击。我们使用广泛使用的IEMOCAP、MSP-Improv和EMODB数据集进行的评估显示，与现有最先进的方法相比，结果有所改善。



