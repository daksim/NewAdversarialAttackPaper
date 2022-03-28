# Latest Adversarial Attack Papers
**update at 2022-03-28 09:45:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Origins of Low-dimensional Adversarial Perturbations**

低维对抗性扰动的起源 stat.ML

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13779v1)

**Authors**: Elvis Dohmatob, Chuan Guo, Morgane Goibert

**Abstracts**: In this note, we initiate a rigorous study of the phenomenon of low-dimensional adversarial perturbations in classification. These are adversarial perturbations wherein, unlike the classical setting, the attacker's search is limited to a low-dimensional subspace of the feature space. The goal is to fool the classifier into flipping its decision on a nonzero fraction of inputs from a designated class, upon the addition of perturbations from a subspace chosen by the attacker and fixed once and for all. It is desirable that the dimension $k$ of the subspace be much smaller than the dimension $d$ of the feature space, while the norm of the perturbations should be negligible compared to the norm of a typical data point. In this work, we consider binary classification models under very general regularity conditions, which are verified by certain feedforward neural networks (e.g., with sufficiently smooth, or else ReLU activation function), and compute analytical lower-bounds for the fooling rate of any subspace. These bounds explicitly highlight the dependence that the fooling rate has on the margin of the model (i.e., the ratio of the output to its $L_2$-norm of its gradient at a test point), and on the alignment of the given subspace with the gradients of the model w.r.t. inputs. Our results provide a theoretical explanation for the recent success of heuristic methods for efficiently generating low-dimensional adversarial perturbations. Moreover, our theoretical results are confirmed by experiments.

摘要: 在这篇笔记中，我们开始对分类中的低维对抗性扰动现象进行严格的研究。这些是对抗性扰动，其中，与经典设置不同，攻击者的搜索被限制在特征空间的低维子空间。目的是在攻击者选择的子空间中添加扰动并一劳永逸地修复后，愚弄分类器在指定类别的输入的非零分数上翻转其决定。希望子空间的维数$k$比特征空间的维数$d$小得多，而与典型数据点的范数相比，扰动的范数应该是可以忽略的。在这项工作中，我们考虑了在非常一般的正则性条件下的二分类模型，这些模型被某些前馈神经网络(例如，具有足够光滑的激活函数)所验证，并计算了任意子空间的愚弄率的解析下界。这些界明确地强调了愚弄率对模型的边缘(即，输出与其在测试点的梯度的$L_2$-范数的比值)以及给定子空间与模型的梯度的对齐的依赖关系(即，输出与其在测试点的梯度的$L_2$-范数的比值)，以及给定子空间与模型的梯度对齐的依赖性。输入。我们的结果为最近成功的启发式方法有效地产生低维对抗性扰动提供了理论上的解释。此外，我们的理论结果也得到了实验的证实。



## **2. Practical Evaluation of Adversarial Robustness via Adaptive Auto Attack**

基于自适应自动攻击的对手健壮性实用评估 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.05154v2)

**Authors**: Ye Liu, Yaya Cheng, Lianli Gao, Xianglong Liu, Qilong Zhang, Jingkuan Song

**Abstracts**: Defense models against adversarial attacks have grown significantly, but the lack of practical evaluation methods has hindered progress. Evaluation can be defined as looking for defense models' lower bound of robustness given a budget number of iterations and a test dataset. A practical evaluation method should be convenient (i.e., parameter-free), efficient (i.e., fewer iterations) and reliable (i.e., approaching the lower bound of robustness). Towards this target, we propose a parameter-free Adaptive Auto Attack (A$^3$) evaluation method which addresses the efficiency and reliability in a test-time-training fashion. Specifically, by observing that adversarial examples to a specific defense model follow some regularities in their starting points, we design an Adaptive Direction Initialization strategy to speed up the evaluation. Furthermore, to approach the lower bound of robustness under the budget number of iterations, we propose an online statistics-based discarding strategy that automatically identifies and abandons hard-to-attack images. Extensive experiments demonstrate the effectiveness of our A$^3$. Particularly, we apply A$^3$ to nearly 50 widely-used defense models. By consuming much fewer iterations than existing methods, i.e., $1/10$ on average (10$\times$ speed up), we achieve lower robust accuracy in all cases. Notably, we won $\textbf{first place}$ out of 1681 teams in CVPR 2021 White-box Adversarial Attacks on Defense Models competitions with this method. Code is available at: $\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$

摘要: 对抗对手攻击的防御模型已经显著增长，但缺乏实用的评估方法阻碍了进展。评估可以定义为在给定预算迭代次数和测试数据集的情况下寻找防御模型的健壮性下限。一种实用的评估方法应该是方便(即无参数)、高效(即迭代次数较少)和可靠(即接近鲁棒性的下界)。针对这一目标，我们提出了一种无参数的自适应自动攻击(A$^3$)评估方法，该方法以测试时间训练的方式来解决效率和可靠性问题。具体地说，通过观察特定防御模型的对抗性示例在起始点遵循一定的规律，我们设计了一种自适应方向初始化策略来加快评估速度。此外，为了在预算迭代次数下逼近鲁棒性的下界，我们提出了一种基于在线统计的丢弃策略，自动识别和丢弃不易攻击的图像。广泛的实验证明了我们的澳元^3元的有效性。特别是，我们将澳元^3美元应用于近50种广泛使用的防御模型。通过比现有方法消耗更少的迭代次数，即平均$1/10$(10$\倍$加速)，我们在所有情况下都获得了较低的鲁棒精度。值得注意的是，我们用这种方法赢得了CVPR 2021年白盒对抗性攻击防御模型比赛1681支队伍中的$\textbf{第一名}$。代码可在以下网址获得：$\href{https://github.com/liuye6666/adaptive_auto_attack}{https://github.com/liuye6666/adaptive\_auto\_attack}$



## **3. Understanding and Increasing Efficiency of Frank-Wolfe Adversarial Training**

理解和提高弗兰克-沃尔夫对抗性训练的效率 cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR  2022. Preliminary version ICML 2021 Adversarial Machine Learning Workshop.  Code: https://github.com/TheoT1/FW-AT-Adapt

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2012.12368v5)

**Authors**: Theodoros Tsiligkaridis, Jay Roberts

**Abstracts**: Deep neural networks are easily fooled by small perturbations known as adversarial attacks. Adversarial Training (AT) is a technique that approximately solves a robust optimization problem to minimize the worst-case loss and is widely regarded as the most effective defense. Due to the high computation time for generating strong adversarial examples in the AT process, single-step approaches have been proposed to reduce training time. However, these methods suffer from catastrophic overfitting where adversarial accuracy drops during training, and although improvements have been proposed, they increase training time and robustness is far from that of multi-step AT. We develop a theoretical framework for adversarial training with FW optimization (FW-AT) that reveals a geometric connection between the loss landscape and the $\ell_2$ distortion of $\ell_\infty$ FW attacks. We analytically show that high distortion of FW attacks is equivalent to small gradient variation along the attack path. It is then experimentally demonstrated on various deep neural network architectures that $\ell_\infty$ attacks against robust models achieve near maximal distortion, while standard networks have lower distortion. It is experimentally shown that catastrophic overfitting is strongly correlated with low distortion of FW attacks. This mathematical transparency differentiates FW from Projected Gradient Descent (PGD) optimization. To demonstrate the utility of our theoretical framework we develop FW-AT-Adapt, a novel adversarial training algorithm which uses a simple distortion measure to adapt the number of attack steps during training to increase efficiency without compromising robustness. FW-AT-Adapt provides training time on par with single-step fast AT methods and closes the gap between fast AT methods and multi-step PGD-AT with minimal loss in adversarial accuracy in white-box and black-box settings.

摘要: 深度神经网络很容易被被称为对抗性攻击的小扰动所愚弄。对抗性训练(AT)是一种近似地解决稳健优化问题以最小化最坏情况损失的技术，被广泛认为是最有效的防御方法。由于在AT过程中生成强对抗性样本需要很高的计算时间，因此提出了一种单步方法来减少训练时间。然而，这些方法在训练过程中存在灾难性的过拟合问题，其中对抗性精度会下降，虽然已经提出了改进，但它们增加了训练时间，并且鲁棒性远远低于多步AT。我们开发了一个基于FW优化的对抗性训练理论框架(FW-AT)，该框架揭示了损失情况与$\ell_inty$FW攻击的$\ell_2$失真之间的几何关系。我们分析表明，FW攻击的高失真等价于攻击路径上的小梯度变化。在不同深度神经网络结构上的实验结果表明，对健壮模型的攻击可以获得接近最大的失真，而标准网络的失真较小。实验表明，灾难性过拟合与FW攻击的低失真密切相关。这种数学透明度将FW与投影渐变下降(PGD)优化区分开来。为了证明我们的理论框架的有效性，我们开发了一种新的对抗性训练算法FW-AT-Adapt，它使用一个简单的失真度量来调整训练过程中的攻击步数，以在不影响健壮性的情况下提高效率。FW-AT-Adapt提供与单步快速AT方法相当的训练时间，并在白盒和黑盒设置下以最小的对手精度损失缩小了快速AT方法和多步PGD-AT方法之间的差距。



## **4. Give Me Your Attention: Dot-Product Attention Considered Harmful for Adversarial Patch Robustness**

给我注意：点积注意被认为对对手补丁的健壮性有害 cs.CV

to be published in IEEE/CVF Conference on Computer Vision and Pattern  Recognition 2022, CVPR22

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13639v1)

**Authors**: Giulio Lovisotto, Nicole Finnie, Mauricio Munoz, Chaithanya Kumar Mummadi, Jan Hendrik Metzen

**Abstracts**: Neural architectures based on attention such as vision transformers are revolutionizing image recognition. Their main benefit is that attention allows reasoning about all parts of a scene jointly. In this paper, we show how the global reasoning of (scaled) dot-product attention can be the source of a major vulnerability when confronted with adversarial patch attacks. We provide a theoretical understanding of this vulnerability and relate it to an adversary's ability to misdirect the attention of all queries to a single key token under the control of the adversarial patch. We propose novel adversarial objectives for crafting adversarial patches which target this vulnerability explicitly. We show the effectiveness of the proposed patch attacks on popular image classification (ViTs and DeiTs) and object detection models (DETR). We find that adversarial patches occupying 0.5% of the input can lead to robust accuracies as low as 0% for ViT on ImageNet, and reduce the mAP of DETR on MS COCO to less than 3%.

摘要: 基于注意力的神经结构，如视觉转换器，正在给图像识别带来革命性的变化。它们的主要好处是，注意力允许对场景的所有部分进行联合推理。在这篇文章中，我们展示了在面对敌意补丁攻击时，(缩放的)点积注意力的全局推理是如何成为主要漏洞的来源。我们提供了对此漏洞的理论理解，并将其与对手将所有查询的注意力误导到对手补丁控制下的单个密钥令牌的能力相关联。我们提出了新的对抗性目标，用于制作明确针对此漏洞的对抗性补丁。我们在流行的图像分类(VITS和DEITS)和目标检测模型(DETR)上展示了所提出的补丁攻击的有效性。我们发现，当敌意补丁占输入的0.5%时，VIT在ImageNet上的鲁棒准确率可以低至0%，而DETR在MS CoCo上的MAP可以降低到3%以下。



## **5. Adversarial Bone Length Attack on Action Recognition**

动作识别的对抗性骨长攻击 cs.CV

12 pages, 8 figures, accepted to AAAI2022

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2109.05830v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstracts**: Skeleton-based action recognition models have recently been shown to be vulnerable to adversarial attacks. Compared to adversarial attacks on images, perturbations to skeletons are typically bounded to a lower dimension of approximately 100 per frame. This lower-dimensional setting makes it more difficult to generate imperceptible perturbations. Existing attacks resolve this by exploiting the temporal structure of the skeleton motion so that the perturbation dimension increases to thousands. In this paper, we show that adversarial attacks can be performed on skeleton-based action recognition models, even in a significantly low-dimensional setting without any temporal manipulation. Specifically, we restrict the perturbations to the lengths of the skeleton's bones, which allows an adversary to manipulate only approximately 30 effective dimensions. We conducted experiments on the NTU RGB+D and HDM05 datasets and demonstrate that the proposed attack successfully deceived models with sometimes greater than 90% success rate by small perturbations. Furthermore, we discovered an interesting phenomenon: in our low-dimensional setting, the adversarial training with the bone length attack shares a similar property with data augmentation, and it not only improves the adversarial robustness but also improves the classification accuracy on the original data. This is an interesting counterexample of the trade-off between adversarial robustness and clean accuracy, which has been widely observed in studies on adversarial training in the high-dimensional regime.

摘要: 基于骨架的动作识别模型最近被证明容易受到对手攻击。与对图像的敌意攻击相比，对骨骼的扰动通常被限制在大约每帧100个维度的较低维度。这种较低维度的设置使产生难以察觉的微扰变得更加困难。现有的攻击通过利用骨骼运动的时间结构来解决这个问题，从而使扰动维度增加到数千。在本文中，我们证明了对抗性攻击可以在基于骨架的动作识别模型上执行，即使在显著低维的环境中也不需要任何时间处理。具体地说，我们将扰动限制在骨骼的长度上，这使得对手只能操纵大约30个有效维度。我们在NTU、RGB+D和HDM05数据集上进行了实验，结果表明，该攻击通过微小的扰动成功地欺骗了模型，有时成功率超过90%。此外，我们还发现了一个有趣的现象：在我们的低维环境下，带有骨长攻击的对抗性训练与数据增强具有相似的性质，它不仅提高了对抗性的健壮性，而且提高了对原始数据的分类精度。这是一个有趣的反例，说明了对抗性稳健性和清晰准确性之间的权衡，这在高维体制下对抗性训练的研究中得到了广泛的观察。



## **6. Improving Adversarial Transferability with Spatial Momentum**

利用空间动量提高对抗性转移能力 cs.CV

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2203.13479v1)

**Authors**: Guoqiu Wang, Xingxing Wei, Huanqian Yan

**Abstracts**: Deep Neural Networks (DNN) are vulnerable to adversarial examples. Although many adversarial attack methods achieve satisfactory attack success rates under the white-box setting, they usually show poor transferability when attacking other DNN models. Momentum-based attack (MI-FGSM) is one effective method to improve transferability. It integrates the momentum term into the iterative process, which can stabilize the update directions by adding the gradients' temporal correlation for each pixel. We argue that only this temporal momentum is not enough, the gradients from the spatial domain within an image, i.e. gradients from the context pixels centered on the target pixel are also important to the stabilization. For that, in this paper, we propose a novel method named Spatial Momentum Iterative FGSM Attack (SMI-FGSM), which introduces the mechanism of momentum accumulation from temporal domain to spatial domain by considering the context gradient information from different regions within the image. SMI-FGSM is then integrated with MI-FGSM to simultaneously stabilize the gradients' update direction from both the temporal and spatial domain. The final method is called SM$^2$I-FGSM. Extensive experiments are conducted on the ImageNet dataset and results show that SM$^2$I-FGSM indeed further enhances the transferability. It achieves the best transferability success rate for multiple mainstream undefended and defended models, which outperforms the state-of-the-art methods by a large margin.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响。虽然许多对抗性攻击方法在白盒设置下取得了令人满意的攻击成功率，但它们在攻击其他DNN模型时往往表现出较差的可移植性。基于动量的攻击(MI-FGSM)是提高可转移性的一种有效方法。该算法在迭代过程中引入动量项，通过对每个像素增加梯度的时间相关性来稳定更新方向。我们认为，仅有这种时间动量是不够的，图像中来自空间域的梯度，即来自以目标像素为中心的上下文像素的梯度，对于稳定也是重要的。为此，本文提出了一种新的方法--空间动量迭代FGSM攻击(SMI-FGSM)，该方法通过考虑图像内不同区域的上下文梯度信息，引入了从时间域到空间域的动量积累机制。然后将SMI-FGSM与MI-FGSM相结合，从时间域和空间域同时稳定梯度的更新方向。最后一种方法称为SM$^2$I-FGSM。在ImageNet数据集上进行了大量的实验，结果表明SM$^2$I-FGSM确实进一步提高了可转移性。它在多个主流的无防御和有防御的模型上实现了最好的可转移性成功率，远远超过了最先进的方法。



## **7. Bounding Training Data Reconstruction in Private (Deep) Learning**

私密(深度)学习中的定界训练数据重构 cs.LG

**SubmitDate**: 2022-03-25    [paper-pdf](http://arxiv.org/pdf/2201.12383v2)

**Authors**: Chuan Guo, Brian Karrer, Kamalika Chaudhuri, Laurens van der Maaten

**Abstracts**: Differential privacy is widely accepted as the de facto method for preventing data leakage in ML, and conventional wisdom suggests that it offers strong protection against privacy attacks. However, existing semantic guarantees for DP focus on membership inference, which may overestimate the adversary's capabilities and is not applicable when membership status itself is non-sensitive. In this paper, we derive the first semantic guarantees for DP mechanisms against training data reconstruction attacks under a formal threat model. We show that two distinct privacy accounting methods -- Renyi differential privacy and Fisher information leakage -- both offer strong semantic protection against data reconstruction attacks.

摘要: 差别隐私被广泛接受为ML中防止数据泄露的事实上的方法，传统观点认为它提供了针对隐私攻击的强大保护。然而，现有的DP语义保证侧重于成员关系推理，这可能会高估对手的能力，并且不适用于成员身份本身不敏感的情况。在形式化威胁模型下，我们推导了DP机制抵抗训练数据重构攻击的第一个语义保证。我们发现，两种不同的隐私计费方法--Renyi差分隐私和Fisher信息泄漏--都提供了针对数据重构攻击的强大语义保护。



## **8. A Perturbation Constrained Adversarial Attack for Evaluating the Robustness of Optical Flow**

一种用于评估光流稳健性的扰动约束对抗性攻击 cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.13214v1)

**Authors**: Jenny Schmalfuss, Philipp Scholze, Andrés Bruhn

**Abstracts**: Recent optical flow methods are almost exclusively judged in terms of accuracy, while analyzing their robustness is often neglected. Although adversarial attacks offer a useful tool to perform such an analysis, current attacks on optical flow methods rather focus on real-world attacking scenarios than on a worst case robustness assessment. Hence, in this work, we propose a novel adversarial attack - the Perturbation Constrained Flow Attack (PCFA) - that emphasizes destructivity over applicability as a real-world attack. More precisely, PCFA is a global attack that optimizes adversarial perturbations to shift the predicted flow towards a specified target flow, while keeping the L2 norm of the perturbation below a chosen bound. Our experiments not only demonstrate PCFA's applicability in white- and black-box settings, but also show that it finds stronger adversarial samples for optical flow than previous attacking frameworks. Moreover, based on these strong samples, we provide the first common ranking of optical flow methods in the literature considering both prediction quality and adversarial robustness, indicating that high quality methods are not necessarily robust. Our source code will be publicly available.

摘要: 目前的光流方法几乎完全是根据精度来判断的，而对它们的稳健性分析往往被忽视。虽然对抗性攻击提供了执行这种分析的有用工具，但是当前对光流方法的攻击更多地集中在真实世界的攻击场景上，而不是最坏情况下的健壮性评估。因此，在这项工作中，我们提出了一种新的对抗性攻击-扰动约束流攻击(PCFA)-作为一种现实世界的攻击，它强调破坏性而不是适用性。更准确地说，PCFA是一种全局攻击，它优化对抗性扰动，将预测流向指定的目标流移动，同时将扰动的L2范数保持在选定的界限以下。我们的实验不仅证明了PCFA在白盒和黑盒环境下的适用性，而且表明它发现的光流攻击样本比以前的攻击框架更具对抗性。此外，基于这些强样本，我们给出了文献中第一个同时考虑预测质量和对抗鲁棒性的光流方法的通用排名，表明高质量的方法不一定是健壮的。我们的源代码将向公众开放。



## **9. Frequency-driven Imperceptible Adversarial Attack on Semantic Similarity**

基于频率驱动的语义相似度潜伏攻击 cs.CV

CVPR 2022 conference (accepted), 18 pages, 17 figure

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2203.05151v4)

**Authors**: Cheng Luo, Qinliang Lin, Weicheng Xie, Bizhu Wu, Jinheng Xie, Linlin Shen

**Abstracts**: Current adversarial attack research reveals the vulnerability of learning-based classifiers against carefully crafted perturbations. However, most existing attack methods have inherent limitations in cross-dataset generalization as they rely on a classification layer with a closed set of categories. Furthermore, the perturbations generated by these methods may appear in regions easily perceptible to the human visual system (HVS). To circumvent the former problem, we propose a novel algorithm that attacks semantic similarity on feature representations. In this way, we are able to fool classifiers without limiting attacks to a specific dataset. For imperceptibility, we introduce the low-frequency constraint to limit perturbations within high-frequency components, ensuring perceptual similarity between adversarial examples and originals. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) and three public online platforms indicate that our attack can yield misleading and transferable adversarial examples across architectures and datasets. Additionally, visualization results and quantitative performance (in terms of four different metrics) show that the proposed algorithm generates more imperceptible perturbations than the state-of-the-art methods. Code is made available at.

摘要: 目前的对抗性攻击研究揭示了基于学习的分类器对精心设计的扰动的脆弱性。然而，现有的大多数攻击方法在跨数据集泛化方面都有固有的局限性，因为它们依赖于具有封闭类别集的分类层。此外，由这些方法产生的扰动可能出现在人类视觉系统(HVS)容易察觉的区域。针对上述问题，我们提出了一种攻击特征表示语义相似度的新算法。通过这种方式，我们能够愚弄分类器，而不会将攻击限制在特定的数据集。对于不可感知性，我们引入了低频约束来限制高频分量内的扰动，以确保对抗性示例与原始示例之间的感知相似性。在三个数据集(CIFAR-10、CIFAR-100和ImageNet-1K)和三个公共在线平台上的广泛实验表明，我们的攻击可以产生跨体系结构和数据集的误导性和可转移的敌意示例。此外，可视化结果和量化性能(根据四个不同的度量)表明，该算法比现有的方法产生更多的不可察觉的扰动。代码可在以下位置获得：。



## **10. Imperceptible Transfer Attack and Defense on 3D Point Cloud Classification**

三维点云分类的隐形转移攻防 cs.CV

**SubmitDate**: 2022-03-24    [paper-pdf](http://arxiv.org/pdf/2111.10990v2)

**Authors**: Daizong Liu, Wei Hu

**Abstracts**: Although many efforts have been made into attack and defense on the 2D image domain in recent years, few methods explore the vulnerability of 3D models. Existing 3D attackers generally perform point-wise perturbation over point clouds, resulting in deformed structures or outliers, which is easily perceivable by humans. Moreover, their adversarial examples are generated under the white-box setting, which frequently suffers from low success rates when transferred to attack remote black-box models. In this paper, we study 3D point cloud attacks from two new and challenging perspectives by proposing a novel Imperceptible Transfer Attack (ITA): 1) Imperceptibility: we constrain the perturbation direction of each point along its normal vector of the neighborhood surface, leading to generated examples with similar geometric properties and thus enhancing the imperceptibility. 2) Transferability: we develop an adversarial transformation model to generate the most harmful distortions and enforce the adversarial examples to resist it, improving their transferability to unknown black-box models. Further, we propose to train more robust black-box 3D models to defend against such ITA attacks by learning more discriminative point cloud representations. Extensive evaluations demonstrate that our ITA attack is more imperceptible and transferable than state-of-the-arts and validate the superiority of our defense strategy.

摘要: 虽然近年来人们在二维图像领域的攻防方面做了很多努力，但很少有方法研究三维模型的脆弱性。现有的3D攻击者一般对点云进行逐点摄动，产生变形的结构或离群点，这很容易被人察觉到。此外，它们的对抗性例子是在白盒环境下产生的，当转移到攻击远程黑盒模型时，白盒模型的成功率往往很低。本文从两个新的具有挑战性的角度对三维点云攻击进行了研究，提出了一种新的不可感知性转移攻击(ITA)：1)不可感知性：我们约束每个点沿其邻域曲面的法向量的扰动方向，从而生成具有相似几何性质的示例，从而增强了不可感知性。2)可转换性：我们建立了一个对抗性转换模型来产生最有害的扭曲，并加强了对抗性例子来抵抗它，提高了它们到未知黑盒模型的可转移性。此外，我们建议通过学习更具区别性的点云表示来训练更健壮的黑盒3D模型来防御此类ITA攻击。广泛的评估表明，我们的ITA攻击比最先进的攻击更具隐蔽性和可移动性，验证了我们防御战略的优越性。



## **11. Adversarial Training for Improving Model Robustness? Look at Both Prediction and Interpretation**

提高模型稳健性的对抗性训练？预测和解释都要看 cs.CL

AAAI 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12709v1)

**Authors**: Hanjie Chen, Yangfeng Ji

**Abstracts**: Neural language models show vulnerability to adversarial examples which are semantically similar to their original counterparts with a few words replaced by their synonyms. A common way to improve model robustness is adversarial training which follows two steps-collecting adversarial examples by attacking a target model, and fine-tuning the model on the augmented dataset with these adversarial examples. The objective of traditional adversarial training is to make a model produce the same correct predictions on an original/adversarial example pair. However, the consistency between model decision-makings on two similar texts is ignored. We argue that a robust model should behave consistently on original/adversarial example pairs, that is making the same predictions (what) based on the same reasons (how) which can be reflected by consistent interpretations. In this work, we propose a novel feature-level adversarial training method named FLAT. FLAT aims at improving model robustness in terms of both predictions and interpretations. FLAT incorporates variational word masks in neural networks to learn global word importance and play as a bottleneck teaching the model to make predictions based on important words. FLAT explicitly shoots at the vulnerability problem caused by the mismatch between model understandings on the replaced words and their synonyms in original/adversarial example pairs by regularizing the corresponding global word importance scores. Experiments show the effectiveness of FLAT in improving the robustness with respect to both predictions and interpretations of four neural network models (LSTM, CNN, BERT, and DeBERTa) to two adversarial attacks on four text classification tasks. The models trained via FLAT also show better robustness than baseline models on unforeseen adversarial examples across different attacks.

摘要: 神经语言模型显示出对敌意例子的脆弱性，这些例子在语义上与它们的原始对应物相似，但有几个单词被它们的同义词取代。提高模型鲁棒性的一种常见方法是对抗性训练，它遵循两个步骤-通过攻击目标模型来收集对抗性示例，并使用这些对抗性示例在扩充的数据集上对模型进行微调。传统对抗性训练的目标是使模型在原始/对抗性示例对上产生相同的正确预测。然而，两个相似文本上的模型决策之间的一致性被忽略了。我们认为一个健壮的模型应该在原始/对抗性示例对上表现一致，即基于相同的原因(如何)做出相同的预测，这些预测可以被一致的解释反映出来。在这项工作中，我们提出了一种新的特征级对抗性训练方法，称为Flat。Flat旨在提高模型在预测和解释方面的稳健性。Flat在神经网络中引入变量词掩码来学习全局词的重要性，并作为瓶颈，教导模型基于重要词进行预测。Flat通过正则化相应的全局单词重要性分数，显式地解决了替换单词与其在原始/对抗性示例对中的同义词之间的模型理解不匹配所造成的脆弱性问题。实验表明，对于4种文本分类任务上的两种敌意攻击，Flat能有效地提高4种神经网络模型(LSTM、CNN、BERT和DeBERTa)的预测和解释鲁棒性。通过Flat训练的模型在不同攻击的不可预见的对抗性实例上也表现出比基线模型更好的鲁棒性。



## **12. Enhancing Classifier Conservativeness and Robustness by Polynomiality**

利用多项式增强分类器的保守性和稳健性 cs.LG

IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12693v1)

**Authors**: Ziqi Wang, Marco Loog

**Abstracts**: We illustrate the detrimental effect, such as overconfident decisions, that exponential behavior can have in methods like classical LDA and logistic regression. We then show how polynomiality can remedy the situation. This, among others, leads purposefully to random-level performance in the tails, away from the bulk of the training data. A directly related, simple, yet important technical novelty we subsequently present is softRmax: a reasoned alternative to the standard softmax function employed in contemporary (deep) neural networks. It is derived through linking the standard softmax to Gaussian class-conditional models, as employed in LDA, and replacing those by a polynomial alternative. We show that two aspects of softRmax, conservativeness and inherent gradient regularization, lead to robustness against adversarial attacks without gradient obfuscation.

摘要: 我们举例说明了指数行为在经典的LDA和Logistic回归等方法中可能产生的不利影响，如过度自信的决策。然后，我们展示了多项式是如何弥补这种情况的。这在其他方面中，故意导致尾部的随机级别的性能，而不是大量的训练数据。我们随后提出的一个直接相关、简单但重要的技术创新是softRmax：当代(深层)神经网络中采用的标准Softmax函数的合理替代方案。它是通过将标准Softmax链接到LDA中使用的高斯类条件模型，并用多项式替代来推导出来的。我们证明了软Rmax的两个方面，保守性和固有的梯度正则化，使得在没有梯度混淆的情况下对敌意攻击具有鲁棒性。



## **13. Explainability-Aware One Point Attack for Point Cloud Neural Networks**

基于可解释性的点云神经网络单点攻击 cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2110.04158v3)

**Authors**: Hanxiao Tan, Helena Kotthaus

**Abstracts**: With the proposition of neural networks for point clouds, deep learning has started to shine in the field of 3D object recognition while researchers have shown an increased interest to investigate the reliability of point cloud networks by adversarial attacks. However, most of the existing studies aim to deceive humans or defense algorithms, while the few that address the operation principles of the models themselves remain flawed in terms of critical point selection. In this work, we propose two adversarial methods: One Point Attack (OPA) and Critical Traversal Attack (CTA), which incorporate the explainability technologies and aim to explore the intrinsic operating principle of point cloud networks and their sensitivity against critical points perturbations. Our results show that popular point cloud networks can be deceived with almost $100\%$ success rate by shifting only one point from the input instance. In addition, we show the interesting impact of different point attribution distributions on the adversarial robustness of point cloud networks. Finally, we discuss how our approaches facilitate the explainability study for point cloud networks. To the best of our knowledge, this is the first point-cloud-based adversarial approach concerning explainability. Our code is available at https://github.com/Explain3D/Exp-One-Point-Atk-PC.

摘要: 随着神经网络在点云领域的提出，深度学习开始在三维物体识别领域闪耀光芒，而研究人员对利用对抗性攻击来研究点云网络的可靠性也表现出越来越大的兴趣。然而，现有的大多数研究都是为了欺骗人类或防御算法，而少数针对模型本身操作原理的研究在临界点选择方面仍然存在缺陷。在这项工作中，我们提出了两种对抗方法：单点攻击(OPA)和临界遍历攻击(CTA)，它们融合了可解释性技术，旨在探索点云网络的内在工作原理及其对临界点扰动的敏感性。我们的结果表明，只要从输入实例中移动一个点，流行的点云网络就可以被欺骗，成功率接近100美元。此外，我们还展示了不同的点属性分布对点云网络对抗健壮性的影响。最后，我们讨论了我们的方法如何促进点云网络的可解释性研究。据我们所知，这是第一种基于点云的对抗性解释方法。我们的代码可在https://github.com/Explain3D/Exp-One-Point-Atk-PC.获得



## **14. Adversarial Fine-tuning for Backdoor Defense: Connecting Backdoor Attacks to Adversarial Attacks**

对抗性后门防御微调：将后门攻击与对抗性攻击联系起来 cs.CV

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2202.06312v2)

**Authors**: Bingxu Mu, Zhenxing Niu, Le Wang, Xue Wang, Rong Jin, Gang Hua

**Abstracts**: Deep neural networks (DNNs) are known to be vulnerable to both backdoor attacks as well as adversarial attacks. In the literature, these two types of attacks are commonly treated as distinct problems and solved separately, since they belong to training-time and inference-time attacks respectively. However, in this paper we find an intriguing connection between them: for a model planted with backdoors, we observe that its adversarial examples have similar behaviors as its triggered samples, i.e., both activate the same subset of DNN neurons. It indicates that planting a backdoor into a model will significantly affect the model's adversarial examples. Based on this observations, we design a new Adversarial Fine-Tuning (AFT) algorithm to defend against backdoor attacks. We empirically show that, against 5 state-of-the-art backdoor attacks, our AFT can effectively erase the backdoor triggers without obvious performance degradation on clean samples and significantly outperforms existing defense methods.

摘要: 众所周知，深度神经网络(DNN)既容易受到后门攻击，也容易受到对手攻击。在文献中，这两类攻击通常被视为不同的问题，分别属于训练时间攻击和推理时间攻击。然而，在本文中，我们发现了它们之间的一个有趣的联系：对于一个植入后门的模型，我们观察到其敌对示例与其触发样本具有相似的行为，即两者都激活了相同的DNN神经元子集。它表明，在模型中植入后门将显著影响模型的对抗性示例。基于这些观察结果，我们设计了一种新的对抗精调(AFT)算法来防御后门攻击。我们的实验表明，对于5种最先进的后门攻击，我们的AFT可以有效地清除后门触发，而在干净的样本上没有明显的性能下降，并且显著优于现有的防御方法。



## **15. Input-specific Attention Subnetworks for Adversarial Detection**

用于敌意检测的输入特定关注子网络 cs.CL

Accepted at Findings of ACL 2022, 14 pages, 6 Tables and 9 Figures

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.12298v1)

**Authors**: Emil Biju, Anirudh Sriram, Pratyush Kumar, Mitesh M Khapra

**Abstracts**: Self-attention heads are characteristic of Transformer models and have been well studied for interpretability and pruning. In this work, we demonstrate an altogether different utility of attention heads, namely for adversarial detection. Specifically, we propose a method to construct input-specific attention subnetworks (IAS) from which we extract three features to discriminate between authentic and adversarial inputs. The resultant detector significantly improves (by over 7.5%) the state-of-the-art adversarial detection accuracy for the BERT encoder on 10 NLU datasets with 11 different adversarial attack types. We also demonstrate that our method (a) is more accurate for larger models which are likely to have more spurious correlations and thus vulnerable to adversarial attack, and (b) performs well even with modest training sets of adversarial examples.

摘要: 自关注磁头是变压器模型的特征，在可解释性和剪枝方面已经得到了很好的研究。在这项工作中，我们展示了注意力头部的一种完全不同的用途，即用于敌意检测。具体地说，我们提出了一种构造输入特定关注子网络(IAS)的方法，从IAS中提取三个特征来区分真实输入和敌意输入。该检测器在具有11种不同攻击类型的10个NLU数据集上显著提高了BERT编码器的最新敌方检测精度(超过7.5%)。我们还证明了我们的方法(A)对于较大的模型更准确，这些模型可能具有更多的伪相关性，因此容易受到对抗性攻击，并且(B)即使在适度的对抗性示例训练集的情况下，我们的方法也表现得很好。



## **16. Integrity Fingerprinting of DNN with Double Black-box Design and Verification**

基于双黑盒设计和验证的DNN完整性指纹分析 cs.CR

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.10902v2)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Surya Nepal, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.

摘要: 支持云的机器学习即服务(MLaaS)在转变深度学习模型的开发和部署方式方面显示出巨大的潜力。尽管如此，使用此类服务仍存在潜在风险，因为恶意方可能会对其进行修改以达到不利的结果。因此，模型所有者、服务提供商和最终用户必须验证部署的模型是否未被篡改。这样的验证需要公开的可验证性(即，指纹模式对所有各方都可用，包括对手)，并且需要通过API对部署的模型进行黑盒访问。然而，现有的水印和指纹方法需要白盒知识(如梯度)来设计指纹，并且只支持私密可验证性，即由诚实的一方进行验证。在本文中，我们描述了一种实用的水印技术，它能够在指纹设计中提供黑盒知识，并在验证过程中提供黑盒查询。该服务通过公开验证(即包括对手在内的各方均可使用指纹模式)来确保云服务的完整性。如果对手操纵模型，这将导致决策边界的移动。因此，双黑水印的基本原理是模型的决策边界可以作为水印的固有指纹。我们的方法通过生成有限数量的包裹样本指纹来捕获决策边界，这些指纹是围绕模型决策边界的一组自然转换和扩充的输入，以捕获模型的固有指纹。我们针对各种模型完整性攻击和模型压缩攻击对我们的水印方法进行了评估。



## **17. Shadows can be Dangerous: Stealthy and Effective Physical-world Adversarial Attack by Natural Phenomenon**

阴影可能是危险的：自然现象对物理世界的隐秘而有效的对抗性攻击 cs.CV

This paper has been accepted by CVPR2022. Code:  https://github.com/hncszyq/ShadowAttack

**SubmitDate**: 2022-03-23    [paper-pdf](http://arxiv.org/pdf/2203.03818v3)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstracts**: Estimating the risk level of adversarial examples is essential for safely deploying machine learning models in the real world. One popular approach for physical-world attacks is to adopt the "sticker-pasting" strategy, which however suffers from some limitations, including difficulties in access to the target or printing by valid colors. A new type of non-invasive attacks emerged recently, which attempt to cast perturbation onto the target by optics based tools, such as laser beam and projector. However, the added optical patterns are artificial but not natural. Thus, they are still conspicuous and attention-grabbed, and can be easily noticed by humans. In this paper, we study a new type of optical adversarial examples, in which the perturbations are generated by a very common natural phenomenon, shadow, to achieve naturalistic and stealthy physical-world adversarial attack under the black-box setting. We extensively evaluate the effectiveness of this new attack on both simulated and real-world environments. Experimental results on traffic sign recognition demonstrate that our algorithm can generate adversarial examples effectively, reaching 98.23% and 90.47% success rates on LISA and GTSRB test sets respectively, while continuously misleading a moving camera over 95% of the time in real-world scenarios. We also offer discussions about the limitations and the defense mechanism of this attack.

摘要: 估计对抗性示例的风险水平对于在现实世界中安全地部署机器学习模型是至关重要的。物理世界攻击的一种流行方法是采用“粘贴”策略，但该策略受到一些限制，包括难以接近目标或以有效颜色打印。最近出现了一种新型的非侵入性攻击，它试图通过激光束和投影仪等基于光学的工具对目标进行摄动。然而，添加的光学图案是人造的，但不是自然的。因此，它们仍然是引人注目和引人注目的，很容易被人类注意到。本文研究了一种新的光学对抗实例，其中的扰动是由一种非常常见的自然现象--阴影产生的，从而在黑盒环境下实现了自然主义的、隐身的物理世界对抗攻击。我们广泛评估了这种新攻击在模拟和真实环境中的有效性。在交通标志识别上的实验结果表明，该算法能够有效地生成对抗性样本，在LISA和GTSRB测试集上的成功率分别达到98.23%和90.47%，而在真实场景中，95%以上的时间都能连续误导移动的摄像机。我们还讨论了这种攻击的局限性和防御机制。



## **18. Online Adversarial Attacks**

在线对抗性攻击 cs.LG

ICLR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2103.02014v4)

**Authors**: Andjela Mladenovic, Avishek Joey Bose, Hugo Berard, William L. Hamilton, Simon Lacoste-Julien, Pascal Vincent, Gauthier Gidel

**Abstracts**: Adversarial attacks expose important vulnerabilities of deep learning models, yet little attention has been paid to settings where data arrives as a stream. In this paper, we formalize the online adversarial attack problem, emphasizing two key elements found in real-world use-cases: attackers must operate under partial knowledge of the target model, and the decisions made by the attacker are irrevocable since they operate on a transient data stream. We first rigorously analyze a deterministic variant of the online threat model by drawing parallels to the well-studied $k$-secretary problem in theoretical computer science and propose Virtual+, a simple yet practical online algorithm. Our main theoretical result shows Virtual+ yields provably the best competitive ratio over all single-threshold algorithms for $k<5$ -- extending the previous analysis of the $k$-secretary problem. We also introduce the \textit{stochastic $k$-secretary} -- effectively reducing online blackbox transfer attacks to a $k$-secretary problem under noise -- and prove theoretical bounds on the performance of Virtual+ adapted to this setting. Finally, we complement our theoretical results by conducting experiments on MNIST, CIFAR-10, and Imagenet classifiers, revealing the necessity of online algorithms in achieving near-optimal performance and also the rich interplay between attack strategies and online attack selection, enabling simple strategies like FGSM to outperform stronger adversaries.

摘要: 对抗性攻击暴露了深度学习模型的重要漏洞，但很少有人关注数据以流的形式到达的设置。本文对在线敌意攻击问题进行了形式化描述，强调了在实际用例中发现的两个关键因素：攻击者必须在部分了解目标模型的情况下操作，并且攻击者的决策是不可撤销的，因为他们操作的是瞬态数据流。我们首先对在线威胁模型的一个确定性变体进行了严格的分析，将其与理论计算机科学中研究得很好的$k$秘书问题进行了比较，并提出了一种简单实用的在线算法Virtual+。我们的主要理论结果表明，对于$k<5$，Virtual+在所有单阈值算法中可以产生最优的好胜比--扩展了之前对$k$秘书问题的分析。我们还引入了\textit{随机$k$-秘书}--在噪声环境下有效地将在线黑盒传输攻击归结为一个$k$-秘书问题--并证明了适用于此设置的Virtual+性能的理论界限。最后，我们通过在MNIST、CIFAR-10和Imagenet分类器上进行实验来补充我们的理论结果，揭示了在线算法获得接近最优性能的必要性，以及攻击策略和在线攻击选择之间的丰富交互作用，使FGSM等简单策略的性能优于更强大的对手。



## **19. NNReArch: A Tensor Program Scheduling Framework Against Neural Network Architecture Reverse Engineering**

NNReArch：一种抗神经网络结构逆向工程的张量程序调度框架 cs.CR

Accepted by FCCM 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.12046v1)

**Authors**: Yukui Luo, Shijin Duan, Cheng Gongye, Yunsi Fei, Xiaolin Xu

**Abstracts**: Architecture reverse engineering has become an emerging attack against deep neural network (DNN) implementations. Several prior works have utilized side-channel leakage to recover the model architecture while the target is executing on a hardware acceleration platform. In this work, we target an open-source deep-learning accelerator, Versatile Tensor Accelerator (VTA), and utilize electromagnetic (EM) side-channel leakage to comprehensively learn the association between DNN architecture configurations and EM emanations. We also consider the holistic system -- including the low-level tensor program code of the VTA accelerator on a Xilinx FPGA and explore the effect of such low-level configurations on the EM leakage. Our study demonstrates that both the optimization and configuration of tensor programs will affect the EM side-channel leakage.   Gaining knowledge of the association between the low-level tensor program and the EM emanations, we propose NNReArch, a lightweight tensor program scheduling framework against side-channel-based DNN model architecture reverse engineering. Specifically, NNReArch targets reshaping the EM traces of different DNN operators, through scheduling the tensor program execution of the DNN model so as to confuse the adversary. NNReArch is a comprehensive protection framework supporting two modes, a balanced mode that strikes a balance between the DNN model confidentiality and execution performance, and a secure mode where the most secure setting is chosen. We implement and evaluate the proposed framework on the open-source VTA with state-of-the-art DNN architectures. The experimental results demonstrate that NNReArch can efficiently enhance the model architecture security with a small performance overhead. In addition, the proposed obfuscation technique makes reverse engineering of the DNN architecture significantly harder.

摘要: 体系结构逆向工程已经成为对深度神经网络(DNN)实现的一种新兴攻击。已有的几个工作已经利用侧信道泄漏来恢复目标在硬件加速平台上执行时的模型体系结构。在这项工作中，我们以开源的深度学习加速器-通用张量加速器(VTA)为目标，并利用电磁(EM)侧通道泄漏来全面了解DNN架构配置与EM发射之间的关联。我们还考虑了整个系统--包括Xilinx FPGA上VTA加速器的低电平张量程序代码，并探索了这种低电平配置对电磁泄漏的影响。我们的研究表明，张量程序的优化和配置都会影响电磁侧沟道泄漏。通过了解低级张量程序与EM发射之间的关联，我们提出了一种轻量级张量程序调度框架NNReArch，该框架针对基于侧通道的DNN模型体系结构逆向工程。具体地说，NNReArch的目标是通过调度DNN模型的张量程序执行来重塑不同DNN算子的EM轨迹，从而迷惑对手。NNReArch是一个全面的保护框架，支持两种模式，一种是在DNN模型机密性和执行性能之间取得平衡的平衡模式，另一种是选择最安全设置的安全模式。我们在采用最先进的DNN体系结构的开源VTA上实现了该框架，并对其进行了评估。实验结果表明，NNReArch能够以较小的性能开销有效地增强模型体系结构的安全性。此外，所提出的模糊技术使得DNN体系结构的逆向工程变得非常困难。



## **20. Shape-invariant 3D Adversarial Point Clouds**

形状不变的三维对抗性点云 cs.CV

Accepted at CVPR 2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.04041v2)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Nenghai Yu

**Abstracts**: Adversary and invisibility are two fundamental but conflict characters of adversarial perturbations. Previous adversarial attacks on 3D point cloud recognition have often been criticized for their noticeable point outliers, since they just involve an "implicit constrain" like global distance loss in the time-consuming optimization to limit the generated noise. While point cloud is a highly structured data format, it is hard to constrain its perturbation with a simple loss or metric properly. In this paper, we propose a novel Point-Cloud Sensitivity Map to boost both the efficiency and imperceptibility of point perturbations. This map reveals the vulnerability of point cloud recognition models when encountering shape-invariant adversarial noises. These noises are designed along the shape surface with an "explicit constrain" instead of extra distance loss. Specifically, we first apply a reversible coordinate transformation on each point of the point cloud input, to reduce one degree of point freedom and limit its movement on the tangent plane. Then we calculate the best attacking direction with the gradients of the transformed point cloud obtained on the white-box model. Finally we assign each point with a non-negative score to construct the sensitivity map, which benefits both white-box adversarial invisibility and black-box query-efficiency extended in our work. Extensive evaluations prove that our method can achieve the superior performance on various point cloud recognition models, with its satisfying adversarial imperceptibility and strong resistance to different point cloud defense settings. Our code is available at: https://github.com/shikiw/SI-Adv.

摘要: 对抗性和隐蔽性是对抗性扰动的两个基本但又相互冲突的特征。以前针对3D点云识别的敌意攻击经常因为其明显的点离群值而受到批评，因为它们只是在耗时的优化过程中涉及诸如全局距离损失这样的“隐式约束”，以限制生成的噪声。虽然点云是一种高度结构化的数据格式，但很难用简单的损失或度量来恰当地约束它的扰动。在本文中，我们提出了一种新的点云敏感度图，以提高点扰动的效率和隐蔽性。这张地图揭示了点云识别模型在遇到形状不变的对抗性噪声时的脆弱性。这些噪波是沿着形状表面设计的，带有“显式约束”，而不是额外的距离损失。具体地说，我们首先对点云输入的每个点应用可逆坐标变换，以减少一个点自由度并限制其在切面上的移动。然后利用白盒模型得到的变换后的点云梯度计算最佳攻击方向。最后，我们给每个点分配一个非负分数来构造敏感度图，这样既有利于白盒对抗不可见性，也有利于提高黑盒查询效率。广泛的评测表明，该方法在各种点云识别模型上都能取得较好的性能，具有令人满意的对抗性和对不同点云防御设置的较强抵抗力。我们的代码可从以下网址获得：https://github.com/shikiw/SI-Adv.



## **21. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

基于后向误差分析的联合学习半目标模型中毒攻击 cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11633v1)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.

摘要: 针对联邦学习(FL)的模型中毒攻击通过破坏边缘模型侵入整个系统，导致机器学习模型失效。这种被破坏的模型被篡改以执行对手期望的行为。特别地，我们考虑了一种半目标情况，其中源类是预先确定的，而目标类不是。目标是使全局分类器对源类的数据进行错误分类。虽然已经采用了标签翻转等方法向FL注入有毒参数，但研究表明，它们的性能通常是类敏感的，随着所使用的目标类的不同，它们的性能也会有所不同。通常，当转移到不同的目标类别时，攻击可能会变得不那么有效。为了克服这一挑战，我们提出了攻击距离感知攻击(ADA)，通过在特征空间中寻找优化的目标类来增强中毒攻击。此外，我们研究了一种更具挑战性的情况，在这种情况下，对手对客户数据的先验知识有限。针对撞击的这一问题，该算法基于后向误差分析，从共享的模型参数中推导出潜在特征空间中不同类别之间的成对距离。在三种不同的图像分类任务中，通过改变攻击频率因子，我们对ADA进行了广泛的经验评估。结果，在攻击频率为0.01的最具挑战性的情况下，ADA成功地将攻击性能提高了1.8倍。



## **22. Exploring High-Order Structure for Robust Graph Structure Learning**

用于鲁棒图结构学习的高阶结构探索 cs.LG

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11492v1)

**Authors**: Guangqian Yang, Yibing Zhan, Jinlong Li, Baosheng Yu, Liu Liu, Fengxiang He

**Abstracts**: Recent studies show that Graph Neural Networks (GNNs) are vulnerable to adversarial attack, i.e., an imperceptible structure perturbation can fool GNNs to make wrong predictions. Some researches explore specific properties of clean graphs such as the feature smoothness to defense the attack, but the analysis of it has not been well-studied. In this paper, we analyze the adversarial attack on graphs from the perspective of feature smoothness which further contributes to an efficient new adversarial defensive algorithm for GNNs. We discover that the effect of the high-order graph structure is a smoother filter for processing graph structures. Intuitively, the high-order graph structure denotes the path number between nodes, where larger number indicates closer connection, so it naturally contributes to defense the adversarial perturbation. Further, we propose a novel algorithm that incorporates the high-order structural information into the graph structure learning. We perform experiments on three popular benchmark datasets, Cora, Citeseer and Polblogs. Extensive experiments demonstrate the effectiveness of our method for defending against graph adversarial attacks.

摘要: 最近的研究表明，图神经网络(GNN)容易受到敌意攻击，即不可察觉的结构扰动可以欺骗GNN做出错误的预测。一些研究探索了干净图的一些特殊性质，如特征光滑性来防御攻击，但对它的分析还没有得到很好的研究。本文从特征光滑性的角度对图的对抗性攻击进行了分析，进而提出了一种新的高效的GNN对抗性防御算法。我们发现，高阶图结构的影响是处理图结构的更平滑的过滤器。直观地说，高阶图结构表示节点之间的路径数，其中越大表示连接越紧密，因此自然有助于防御对手的扰动。在此基础上，提出了一种将高阶结构信息融入到图结构学习中的新算法。我们在三个流行的基准数据集CORA、Citeseer和Polblog上进行了实验。大量的实验证明了该方法对图攻击的有效防御。



## **23. Making DeepFakes more spurious: evading deep face forgery detection via trace removal attack**

让DeepFake更具欺骗性：通过痕迹移除攻击逃避深度人脸伪造检测 cs.CV

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2203.11433v1)

**Authors**: Chi Liu, Huajie Chen, Tianqing Zhu, Jun Zhang, Wanlei Zhou

**Abstracts**: DeepFakes are raising significant social concerns. Although various DeepFake detectors have been developed as forensic countermeasures, these detectors are still vulnerable to attacks. Recently, a few attacks, principally adversarial attacks, have succeeded in cloaking DeepFake images to evade detection. However, these attacks have typical detector-specific designs, which require prior knowledge about the detector, leading to poor transferability. Moreover, these attacks only consider simple security scenarios. Less is known about how effective they are in high-level scenarios where either the detectors or the attacker's knowledge varies. In this paper, we solve the above challenges with presenting a novel detector-agnostic trace removal attack for DeepFake anti-forensics. Instead of investigating the detector side, our attack looks into the original DeepFake creation pipeline, attempting to remove all detectable natural DeepFake traces to render the fake images more "authentic". To implement this attack, first, we perform a DeepFake trace discovery, identifying three discernible traces. Then a trace removal network (TR-Net) is proposed based on an adversarial learning framework involving one generator and multiple discriminators. Each discriminator is responsible for one individual trace representation to avoid cross-trace interference. These discriminators are arranged in parallel, which prompts the generator to remove various traces simultaneously. To evaluate the attack efficacy, we crafted heterogeneous security scenarios where the detectors were embedded with different levels of defense and the attackers' background knowledge of data varies. The experimental results show that the proposed attack can significantly compromise the detection accuracy of six state-of-the-art DeepFake detectors while causing only a negligible loss in visual quality to the original DeepFake samples.

摘要: DeepFake引起了重大的社会关注。虽然已经开发了各种DeepFake检测器作为取证对策，但这些检测器仍然容易受到攻击。最近，一些攻击，主要是对抗性攻击，成功地伪装DeepFake图像以逃避检测。然而，这些攻击具有典型的特定于检测器的设计，需要事先了解检测器，导致可移植性较差。此外，这些攻击只考虑简单的安全场景。对于它们在检测器或攻击者的知识各不相同的高级场景中的有效性，我们知之甚少。在本文中，我们提出了一种新的针对DeepFake反取证的与检测器无关的痕迹移除攻击，从而解决了上述挑战。我们的攻击不是调查探测器端，而是查看原始的DeepFake创建管道，试图删除所有可检测到的自然DeepFake痕迹，以使虚假图像更“真实”。要实现此攻击，首先，我们执行DeepFake跟踪发现，识别三个可识别的跟踪。然后，基于一个生成器和多个鉴别器的对抗性学习框架，提出了一种痕迹去除网络(TR-Net)。每个鉴别器负责一个单独的轨迹表示，以避免交叉轨迹干扰。这些鉴别器是并行排列的，这会提示发生器同时移除各种痕迹。为了评估攻击效能，我们精心设计了不同的安全场景，其中检测器嵌入了不同级别的防御，攻击者的数据背景知识各不相同。实验结果表明，该攻击可以显著降低现有的6种DeepFake检测器的检测精度，而对原始DeepFake样本的视觉质量损失可以忽略不计。



## **24. Subspace Adversarial Training**

子空间对抗训练 cs.LG

CVPR2022

**SubmitDate**: 2022-03-22    [paper-pdf](http://arxiv.org/pdf/2111.12229v2)

**Authors**: Tao Li, Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: Single-step adversarial training (AT) has received wide attention as it proved to be both efficient and robust. However, a serious problem of catastrophic overfitting exists, i.e., the robust accuracy against projected gradient descent (PGD) attack suddenly drops to 0% during the training. In this paper, we approach this problem from a novel perspective of optimization and firstly reveal the close link between the fast-growing gradient of each sample and overfitting, which can also be applied to understand robust overfitting in multi-step AT. To control the growth of the gradient, we propose a new AT method, Subspace Adversarial Training (Sub-AT), which constrains AT in a carefully extracted subspace. It successfully resolves both kinds of overfitting and significantly boosts the robustness. In subspace, we also allow single-step AT with larger steps and larger radius, further improving the robustness performance. As a result, we achieve state-of-the-art single-step AT performance. Without any regularization term, our single-step AT can reach over 51% robust accuracy against strong PGD-50 attack of radius 8/255 on CIFAR-10, reaching a competitive performance against standard multi-step PGD-10 AT with huge computational advantages. The code is released at https://github.com/nblt/Sub-AT.

摘要: 单步对抗性训练(AT)因其高效、健壮而受到广泛关注。然而，存在一个严重的灾难性过拟合问题，即在训练过程中，对投影梯度下降(PGD)攻击的鲁棒精度突然下降到0%。本文从一个新的优化角度来研究这一问题，首次揭示了每个样本快速增长的梯度与过拟合之间的密切联系，这也可以用来理解多步AT中的稳健过拟合。为了控制梯度的增长，我们提出了一种新的AT方法--子空间对抗训练(Sub-Space Adversative Trading，Sub-AT)，它将AT约束在一个仔细提取的子空间中。它成功地解决了这两种过拟合问题，并显著提高了鲁棒性。在子空间中，我们还允许单步AT具有更大的步长和更大的半径，进一步提高了算法的鲁棒性。因此，我们实现了最先进的单步AT性能。在没有任何正则项的情况下，我们的单步AT在抵抗CIFAR-10上半径为8/255的强PGD-50攻击时可以达到51%以上的鲁棒准确率，达到了与标准多步PGD-10 AT相当的性能，同时具有巨大的计算优势。该代码在https://github.com/nblt/Sub-AT.上发布



## **25. On The Robustness of Offensive Language Classifiers**

论攻击性语言量词的稳健性 cs.CL

9 pages, 2 figures, Accepted at ACL 2022

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11331v1)

**Authors**: Jonathan Rusert, Zubair Shafiq, Padmini Srinivasan

**Abstracts**: Social media platforms are deploying machine learning based offensive language classification systems to combat hateful, racist, and other forms of offensive speech at scale. However, despite their real-world deployment, we do not yet comprehensively understand the extent to which offensive language classifiers are robust against adversarial attacks. Prior work in this space is limited to studying robustness of offensive language classifiers against primitive attacks such as misspellings and extraneous spaces. To address this gap, we systematically analyze the robustness of state-of-the-art offensive language classifiers against more crafty adversarial attacks that leverage greedy- and attention-based word selection and context-aware embeddings for word replacement. Our results on multiple datasets show that these crafty adversarial attacks can degrade the accuracy of offensive language classifiers by more than 50% while also being able to preserve the readability and meaning of the modified text.

摘要: 社交媒体平台正在部署基于机器学习的攻击性语言分类系统，以大规模打击仇恨、种族主义和其他形式的攻击性言论。然而，尽管在现实世界中部署了攻击性语言分类器，但我们还没有全面了解攻击性语言分类器在多大程度上对对手攻击具有健壮性。以前在这个领域的工作仅限于研究攻击性语言分类器对原始攻击(如拼写错误和无关空格)的健壮性。为了弥补这一差距，我们系统地分析了最先进的攻击性语言分类器对更狡猾的对手攻击的鲁棒性，这些攻击利用基于贪婪和注意力的单词选择和上下文感知嵌入进行单词替换。我们在多个数据集上的实验结果表明，这些狡猾的敌意攻击可以使攻击性语言分类器的准确率降低50%以上，同时还能够保持修改后的文本的可读性和意义。



## **26. FGAN: Federated Generative Adversarial Networks for Anomaly Detection in Network Traffic**

FGAN：用于网络流量异常检测的联合生成对抗网络 cs.CR

8 pages, 2 figures

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.11106v1)

**Authors**: Sankha Das

**Abstracts**: Over the last two decades, a lot of work has been done in improving network security, particularly in intrusion detection systems (IDS) and anomaly detection. Machine learning solutions have also been employed in IDSs to detect known and plausible attacks in incoming traffic. Parameters such as packet contents, sender IP and sender port, connection duration, etc. have been previously used to train these machine learning models to learn to differentiate genuine traffic from malicious ones. Generative Adversarial Networks (GANs) have been significantly successful in detecting such anomalies, mostly attributed to the adversarial training of the generator and discriminator in an attempt to bypass each other and in turn increase their own power and accuracy. However, in large networks having a wide variety of traffic at possibly different regions of the network and susceptible to a large number of potential attacks, training these GANs for a particular kind of anomaly may make it oblivious to other anomalies and attacks. In addition, the dataset required to train these models has to be made centrally available and publicly accessible, posing the obvious question of privacy of the communications of the respective participants of the network. The solution proposed in this work aims at tackling the above two issues by using GANs in a federated architecture in networks of such scale and capacity. In such a setting, different users of the network will be able to train and customize a centrally available adversarial model according to their own frequently faced conditions. Simultaneously, the member users of the network will also able to gain from the experiences of the other users in the network.

摘要: 在过去的二十年里，人们在提高网络安全方面做了大量的工作，特别是在入侵检测系统和异常检测方面。机器学习解决方案也被用于入侵检测系统中，以检测传入流量中的已知和可能的攻击。以前已经使用数据包内容、发送方IP和发送方端口、连接持续时间等参数来训练这些机器学习模型，以学习区分真实流量和恶意流量。生成性对抗网络(GANS)在检测这种异常方面取得了很大的成功，这主要归因于对生成器和鉴别器的对抗性训练，试图绕过彼此，进而提高自己的能力和准确性。然而，在可能在网络的不同区域具有各种流量并且容易受到大量潜在攻击的大型网络中，针对特定类型的异常对这些GAN进行训练可能会使其对其他异常和攻击视而不见。此外，训练这些模型所需的数据集必须集中提供并向公众开放，这就提出了网络各参与方通信隐私的明显问题。本工作提出的解决方案旨在通过在如此规模和容量的网络中的联合架构中使用GAN来解决上述两个问题。在这样的设置下，网络的不同用户将能够根据他们自己经常面临的情况来训练和定制中央可用的对抗模型。同时，网络的成员用户也将能够从网络中的其他用户的体验中获益。



## **27. An Intermediate-level Attack Framework on The Basis of Linear Regression**

一种基于线性回归的中级攻击框架 cs.CV

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10723v1)

**Authors**: Yiwen Guo, Qizhang Li, Wangmeng Zuo, Hao Chen

**Abstracts**: This paper substantially extends our work published at ECCV, in which an intermediate-level attack was proposed to improve the transferability of some baseline adversarial examples. We advocate to establish a direct linear mapping from the intermediate-level discrepancies (between adversarial features and benign features) to classification prediction loss of the adversarial example. In this paper, we delve deep into the core components of such a framework by performing comprehensive studies and extensive experiments. We show that 1) a variety of linear regression models can all be considered in order to establish the mapping, 2) the magnitude of the finally obtained intermediate-level discrepancy is linearly correlated with adversarial transferability, 3) further boost of the performance can be achieved by performing multiple runs of the baseline attack with random initialization. By leveraging these findings, we achieve new state-of-the-arts on transfer-based $\ell_\infty$ and $\ell_2$ attacks.

摘要: 本文大大扩展了我们在ECCV上发表的工作，其中提出了一种中级攻击来提高一些基线对抗性示例的可移植性。我们主张建立一个直接的线性映射，从对抗性实例的中间层差异(对抗性特征与良性特征之间)到分类预测损失之间建立一个直接的线性映射。在本文中，我们通过全面的研究和广泛的实验，深入研究了这一框架的核心组成部分。我们表明：1)为了建立映射，可以考虑多种线性回归模型；2)最终得到的中间层差异的大小与对抗可转移性线性相关；3)通过随机初始化执行多次基线攻击，可以进一步提高性能。通过利用这些发现，我们实现了针对基于传输的$\ell_\infty$和$\ell_2$攻击的新技术。



## **28. A Prompting-based Approach for Adversarial Example Generation and Robustness Enhancement**

一种基于提示的对抗性实例生成与健壮性增强方法 cs.CL

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10714v1)

**Authors**: Yuting Yang, Pei Huang, Juan Cao, Jintao Li, Yun Lin, Jin Song Dong, Feifei Ma, Jian Zhang

**Abstracts**: Recent years have seen the wide application of NLP models in crucial areas such as finance, medical treatment, and news media, raising concerns of the model robustness and vulnerabilities. In this paper, we propose a novel prompt-based adversarial attack to compromise NLP models and robustness enhancement technique. We first construct malicious prompts for each instance and generate adversarial examples via mask-and-filling under the effect of a malicious purpose. Our attack technique targets the inherent vulnerabilities of NLP models, allowing us to generate samples even without interacting with the victim NLP model, as long as it is based on pre-trained language models (PLMs). Furthermore, we design a prompt-based adversarial training method to improve the robustness of PLMs. As our training method does not actually generate adversarial samples, it can be applied to large-scale training sets efficiently. The experimental results show that our attack method can achieve a high attack success rate with more diverse, fluent and natural adversarial examples. In addition, our robustness enhancement method can significantly improve the robustness of models to resist adversarial attacks. Our work indicates that prompting paradigm has great potential in probing some fundamental flaws of PLMs and fine-tuning them for downstream tasks.

摘要: 近年来，NLP模型在金融、医疗和新闻媒体等关键领域得到了广泛应用，引起了人们对模型健壮性和脆弱性的担忧。本文提出了一种新的基于提示的敌意攻击方法来折衷NLP模型，并提出了健壮性增强技术。我们首先为每个实例构建恶意提示，并在恶意目的的影响下通过掩码和填充生成敌意实例。我们的攻击技术针对NLP模型的固有漏洞，允许我们在不与受害者NLP模型交互的情况下生成样本，只要它是基于预先训练的语言模型(PLM)。此外，我们还设计了一种基于提示的对抗性训练方法来提高PLM的鲁棒性。由于我们的训练方法并不实际生成对抗性样本，因此可以有效地应用于大规模训练集。实验结果表明，该攻击方法具有更丰富、更流畅、更自然的对抗性实例，攻击成功率较高。此外，我们的鲁棒性增强方法可以显著提高模型抵抗对手攻击的鲁棒性。我们的工作表明，提示范式在探测PLM的一些根本缺陷并针对下游任务进行微调方面具有很大的潜力。



## **29. Leveraging Expert Guided Adversarial Augmentation For Improving Generalization in Named Entity Recognition**

利用专家引导的对抗性增强改进命名实体识别中的泛化 cs.CL

ACL 2022 (Findings)

**SubmitDate**: 2022-03-21    [paper-pdf](http://arxiv.org/pdf/2203.10693v1)

**Authors**: Aaron Reich, Jiaao Chen, Aastha Agrawal, Yanzhe Zhang, Diyi Yang

**Abstracts**: Named Entity Recognition (NER) systems often demonstrate great performance on in-distribution data, but perform poorly on examples drawn from a shifted distribution. One way to evaluate the generalization ability of NER models is to use adversarial examples, on which the specific variations associated with named entities are rarely considered. To this end, we propose leveraging expert-guided heuristics to change the entity tokens and their surrounding contexts thereby altering their entity types as adversarial attacks. Using expert-guided heuristics, we augmented the CoNLL 2003 test set and manually annotated it to construct a high-quality challenging set. We found that state-of-the-art NER systems trained on CoNLL 2003 training data drop performance dramatically on our challenging set. By training on adversarial augmented training examples and using mixup for regularization, we were able to significantly improve the performance on the challenging set as well as improve out-of-domain generalization which we evaluated by using OntoNotes data. We have publicly released our dataset and code at https://github.com/GT-SALT/Guided-Adversarial-Augmentation.

摘要: 命名实体识别(NER)系统通常在分布内数据上表现出很好的性能，但在从平移分布中提取的示例上表现得很差。评估NER模型泛化能力的一种方法是使用对抗性例子，在这些例子上很少考虑与命名实体相关的具体变化。为此，我们建议利用专家指导的启发式方法来更改实体令牌及其周围上下文，从而将其实体类型更改为对抗性攻击。使用专家指导的启发式算法，我们扩充了CoNLL 2003测试集，并手动对其进行注释，以构建高质量的挑战性测试集。我们发现，在我们具有挑战性的集合中，使用CoNLL 2003训练数据训练的最先进的NER系统会显著降低性能。通过对抗性增强训练实例的训练和使用混合正则化，我们能够显著提高在具有挑战性的集合上的性能，以及改善我们使用OntoNotes数据评估的域外泛化。我们已经在https://github.com/GT-SALT/Guided-Adversarial-Augmentation.公开发布了我们的数据集和代码



## **30. RareGAN: Generating Samples for Rare Classes**

RareGan：为稀有类生成样本 cs.LG

Published in AAAI 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10674v1)

**Authors**: Zinan Lin, Hao Liang, Giulia Fanti, Vyas Sekar

**Abstracts**: We study the problem of learning generative adversarial networks (GANs) for a rare class of an unlabeled dataset subject to a labeling budget. This problem is motivated from practical applications in domains including security (e.g., synthesizing packets for DNS amplification attacks), systems and networking (e.g., synthesizing workloads that trigger high resource usage), and machine learning (e.g., generating images from a rare class). Existing approaches are unsuitable, either requiring fully-labeled datasets or sacrificing the fidelity of the rare class for that of the common classes. We propose RareGAN, a novel synthesis of three key ideas: (1) extending conditional GANs to use labelled and unlabelled data for better generalization; (2) an active learning approach that requests the most useful labels; and (3) a weighted loss function to favor learning the rare class. We show that RareGAN achieves a better fidelity-diversity tradeoff on the rare class than prior work across different applications, budgets, rare class fractions, GAN losses, and architectures.

摘要: 我们研究了一类稀有的未标记数据集的生成性对抗网络(GANS)的学习问题，这类数据集受标记预算的约束。该问题源于领域中的实际应用，包括安全(例如，合成用于DNS放大攻击的分组)、系统和联网(例如，合成触发高资源使用率的工作负载)以及机器学习(例如，从罕见类生成图像)。现有的方法是不合适的，要么需要完全标记的数据集，要么牺牲稀有类的保真度来换取普通类的保真度。我们提出了RareGAN，这是一种新的综合了三个关键思想的方法：(1)扩展条件Gans以使用有标签和无标签的数据进行更好的泛化；(2)要求最有用的标签的主动学习方法；以及(3)有利于学习稀有类的加权损失函数。我们表明，与以前的工作相比，RareGAN在不同的应用、预算、稀有类分数、GaN损耗和架构上在稀有类上实现了更好的保真度-分集折衷。



## **31. Does DQN really learn? Exploring adversarial training schemes in Pong**

DQN真的会学习吗？探索乒乓球对抗性训练方案 cs.LG

RLDM 2022

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10614v1)

**Authors**: Bowen He, Sreehari Rammohan, Jessica Forde, Michael Littman

**Abstracts**: In this work, we study two self-play training schemes, Chainer and Pool, and show they lead to improved agent performance in Atari Pong compared to a standard DQN agent -- trained against the built-in Atari opponent. To measure agent performance, we define a robustness metric that captures how difficult it is to learn a strategy that beats the agent's learned policy. Through playing past versions of themselves, Chainer and Pool are able to target weaknesses in their policies and improve their resistance to attack. Agents trained using these methods score well on our robustness metric and can easily defeat the standard DQN agent. We conclude by using linear probing to illuminate what internal structures the different agents develop to play the game. We show that training agents with Chainer or Pool leads to richer network activations with greater predictive power to estimate critical game-state features compared to the standard DQN agent.

摘要: 在这项工作中，我们研究了两种自我发挥训练方案，Chainer和Pool，并证明与标准的DQN代理相比，它们在Atari Pong中的代理性能有所提高--针对内置的Atari对手进行训练。为了衡量代理的性能，我们定义了一个健壮性度量，该度量捕获了学习超过代理的学习策略的策略有多难。通过扮演过去版本的自己，Chainer和Pool能够针对他们政策中的弱点，并提高他们对攻击的抵抗力。使用这些方法训练的代理在我们的健壮性度量上得分很好，可以很容易地击败标准的DQN代理。我们最后使用线性探测来说明不同的代理在玩游戏时发展了哪些内部结构。我们表明，与标准的DQN代理相比，用Chainer或Pool训练代理可以导致更丰富的网络激活，并具有更强的预测能力来估计关键的游戏状态特征。



## **32. Improved Semi-Quantum Key Distribution with Two Almost-Classical Users**

改进的两个准经典用户的半量子密钥分配 quant-ph

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10567v1)

**Authors**: Saachi Mutreja, Walter O. Krawec

**Abstracts**: Semi-quantum key distribution (SQKD) protocols attempt to establish a shared secret key between users, secure against computationally unbounded adversaries. Unlike standard quantum key distribution protocols, SQKD protocols contain at least one user who is limited in their quantum abilities and is almost "classical" in nature. In this paper, we revisit a mediated semi-quantum key distribution protocol, introduced by Massa et al., in 2019, where users need only the ability to detect a qubit, or reflect a qubit; they do not need to perform any other basis measurement; nor do they need to prepare quantum signals. Users require the services of a quantum server which may be controlled by the adversary. In this paper, we show how this protocol may be extended to improve its efficiency and also its noise tolerance. We discuss an extension which allows more communication rounds to be directly usable; we analyze the key-rate of this extension in the asymptotic scenario for a particular class of attacks and compare with prior work. Finally, we evaluate the protocol's performance in a variety of lossy and noisy channels.

摘要: 半量子密钥分发(SQKD)协议试图在用户之间建立共享密钥，以防止计算上的无限攻击。与标准量子密钥分发协议不同，SQKD协议至少包含一个用户，该用户的量子能力是有限的，并且本质上几乎是经典的。在本文中，我们回顾了由Massa等人于2019年提出的一种中介半量子密钥分发协议，其中用户只需要检测或反映量子比特的能力；他们不需要执行任何其他的基测量；也不需要准备量子信号。用户需要量子服务器的服务，该服务器可能由对手控制。在本文中，我们展示了如何对该协议进行扩展以提高其效率和抗噪能力。我们讨论了一种允许更多通信轮次直接使用的扩展；我们分析了该扩展在一类特定攻击的渐近场景下的密钥率，并与以前的工作进行了比较。最后，我们对该协议在各种有损和噪声信道中的性能进行了评估。



## **33. Adversarial Parameter Attack on Deep Neural Networks**

基于深度神经网络的对抗性参数攻击 cs.LG

**SubmitDate**: 2022-03-20    [paper-pdf](http://arxiv.org/pdf/2203.10502v1)

**Authors**: Lijia Yu, Yihan Wang, Xiao-Shan Gao

**Abstracts**: In this paper, a new parameter perturbation attack on DNNs, called adversarial parameter attack, is proposed, in which small perturbations to the parameters of the DNN are made such that the accuracy of the attacked DNN does not decrease much, but its robustness becomes much lower. The adversarial parameter attack is stronger than previous parameter perturbation attacks in that the attack is more difficult to be recognized by users and the attacked DNN gives a wrong label for any modified sample input with high probability. The existence of adversarial parameters is proved. For a DNN $F_{\Theta}$ with the parameter set $\Theta$ satisfying certain conditions, it is shown that if the depth of the DNN is sufficiently large, then there exists an adversarial parameter set $\Theta_a$ for $\Theta$ such that the accuracy of $F_{\Theta_a}$ is equal to that of $F_{\Theta}$, but the robustness measure of $F_{\Theta_a}$ is smaller than any given bound. An effective training algorithm is given to compute adversarial parameters and numerical experiments are used to demonstrate that the algorithms are effective to produce high quality adversarial parameters.

摘要: 本文提出了一种新的DNN参数扰动攻击，称为对抗性参数攻击，通过对DNN的参数进行微小的扰动，使得被攻击的DNN的准确率不会降低太多，但鲁棒性却大大降低。对抗性参数攻击比以往的参数扰动攻击更强，因为攻击更难被用户识别，并且被攻击的DNN对任何修改后的样本输入都给出了错误的标签，概率很高。证明了对抗性参数的存在性。对于参数集$\theta$满足一定条件的DNN$F_{\Theta}$，证明了如果DNN的深度足够大，则对于$\Theta$存在对抗参数集$\θ_a$，使得$F_{\theta}$的精度与$F_{\theta}$的精度相等，而$F_{\theta}$的稳健性测度等于$F_{\theta}$给出了一种计算对抗参数的有效训练算法，并通过数值实验证明了该算法能有效地产生高质量的对抗参数。



## **34. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

对比性对抗性训练中认知分离缓解的稳健性 cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08959v2)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.

摘要: 本文提出了一种新的神经网络训练框架，通过将对比学习(CL)和对抗训练(AT)相结合，在保持较高精度的同时，提高了模型对对手攻击的鲁棒性。我们提出通过学习在数据扩充和对抗性扰动下都是一致的特征表示来提高模型对对抗性攻击的稳健性。我们利用对比学习来提高对抗性样本的稳健性，将一个对抗性样本作为另一个正例，目标是最大化随机增加的数据样本与其对抗性样本之间的相似度，同时不断更新分类头，以避免分类头与嵌入空间之间的认知分离。这种分离是由于CL将网络更新到嵌入空间，同时冻结用于生成新的正面对抗性实例的分类头。我们在CIFAR-10数据集上验证了我们的方法，即带有对抗性特征的对比学习(CLAF)，在CIFAR-10数据集上，它的性能优于其他监督和自我监督对抗性学习方法的稳健准确率和干净准确率。



## **35. On Robust Prefix-Tuning for Text Classification**

面向文本分类的鲁棒前缀调优方法研究 cs.CL

Accepted in ICLR 2022. We release the code at  https://github.com/minicheshire/Robust-Prefix-Tuning

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10378v1)

**Authors**: Zonghan Yang, Yang Liu

**Abstracts**: Recently, prefix-tuning has gained increasing attention as a parameter-efficient finetuning method for large-scale pretrained language models. The method keeps the pretrained models fixed and only updates the prefix token parameters for each downstream task. Despite being lightweight and modular, prefix-tuning still lacks robustness to textual adversarial attacks. However, most currently developed defense techniques necessitate auxiliary model update and storage, which inevitably hamper the modularity and low storage of prefix-tuning. In this work, we propose a robust prefix-tuning framework that preserves the efficiency and modularity of prefix-tuning. The core idea of our framework is leveraging the layerwise activations of the language model by correctly-classified training data as the standard for additional prefix finetuning. During the test phase, an extra batch-level prefix is tuned for each batch and added to the original prefix for robustness enhancement. Extensive experiments on three text classification benchmarks show that our framework substantially improves robustness over several strong baselines against five textual attacks of different types while maintaining comparable accuracy on clean texts. We also interpret our robust prefix-tuning framework from the optimal control perspective and pose several directions for future research.

摘要: 近年来，前缀调优作为一种针对大规模预训练语言模型的参数高效精调方法受到越来越多的关注。该方法保持预先训练的模型不变，并且只更新每个下游任务的前缀令牌参数。尽管前缀调优是轻量级和模块化的，但仍然缺乏对文本对手攻击的健壮性。然而，目前开发的大多数防御技术需要辅助模型更新和存储，这不可避免地阻碍了前缀调整的模块化和低存储量。在这项工作中，我们提出了一个健壮的前缀调整框架，它保持了前缀调整的效率和模块性。我们框架的核心思想是通过正确分类训练数据来利用语言模型的LayerWise激活，作为额外前缀优化的标准。在测试阶段，为每个批处理调优一个额外的批级前缀，并将其添加到原始前缀以增强健壮性。在三个文本分类基准上的大量实验表明，我们的框架在保持对干净文本的相对准确率的同时，显著提高了对五种不同类型的文本攻击在几条强基线上的鲁棒性。我们还从最优控制的角度解释了我们的鲁棒前缀调节框架，并对未来的研究提出了几个方向。



## **36. Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense**

荒野中的扰动：利用人类书写的文本扰动进行现实的对抗性攻击和防御 cs.LG

Accepted to the 60th Annual Meeting of the Association for  Computational Linguistics (ACL'22), Findings

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10346v1)

**Authors**: Thai Le, Jooyoung Lee, Kevin Yen, Yifan Hu, Dongwon Lee

**Abstracts**: We proposes a novel algorithm, ANTHRO, that inductively extracts over 600K human-written text perturbations in the wild and leverages them for realistic adversarial attack. Unlike existing character-based attacks which often deductively hypothesize a set of manipulation strategies, our work is grounded on actual observations from real-world texts. We find that adversarial texts generated by ANTHRO achieve the best trade-off between (1) attack success rate, (2) semantic preservation of the original text, and (3) stealthiness--i.e. indistinguishable from human writings hence harder to be flagged as suspicious. Specifically, our attacks accomplished around 83% and 91% attack success rates on BERT and RoBERTa, respectively. Moreover, it outperformed the TextBugger baseline with an increase of 50% and 40% in terms of semantic preservation and stealthiness when evaluated by both layperson and professional human workers. ANTHRO can further enhance a BERT classifier's performance in understanding different variations of human-written toxic texts via adversarial training when compared to the Perspective API.

摘要: 我们提出了一种新的算法Anthro，该算法在野外归纳提取超过600K个人类书写的文本扰动，并利用它们进行现实的对抗性攻击。与现有的基于字符的攻击经常演绎地假设一组操纵策略不同，我们的工作是基于对真实世界文本的实际观察。我们发现，Anthro生成的敌意文本在(1)攻击成功率、(2)保留原文语义和(3)隐蔽性--即隐蔽性--之间达到了最佳的折衷。与人类的作品难以区分，因此更难被标记为可疑。具体地说，我们对伯特和罗伯塔的攻击成功率分别约为83%和91%。此外，在外行和专业人员的评估中，它在语义保留和隐蔽性方面的表现优于TextBugger基线，分别提高了50%和40%。与透视API相比，Anthro可以通过对抗性训练进一步提高BERT分类器在理解人类书写的有毒文本的不同变体方面的性能。



## **37. Efficient Neural Network Analysis with Sum-of-Infeasibilities**

不可行和条件下的高效神经网络分析 cs.LG

TACAS'22

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11201v1)

**Authors**: Haoze Wu, Aleksandar Zeljić, Guy Katz, Clark Barrett

**Abstracts**: Inspired by sum-of-infeasibilities methods in convex optimization, we propose a novel procedure for analyzing verification queries on neural networks with piecewise-linear activation functions. Given a convex relaxation which over-approximates the non-convex activation functions, we encode the violations of activation functions as a cost function and optimize it with respect to the convex relaxation. The cost function, referred to as the Sum-of-Infeasibilities (SoI), is designed so that its minimum is zero and achieved only if all the activation functions are satisfied. We propose a stochastic procedure, DeepSoI, to efficiently minimize the SoI. An extension to a canonical case-analysis-based complete search procedure can be achieved by replacing the convex procedure executed at each search state with DeepSoI. Extending the complete search with DeepSoI achieves multiple simultaneous goals: 1) it guides the search towards a counter-example; 2) it enables more informed branching decisions; and 3) it creates additional opportunities for bound derivation. An extensive evaluation across different benchmarks and solvers demonstrates the benefit of the proposed techniques. In particular, we demonstrate that SoI significantly improves the performance of an existing complete search procedure. Moreover, the SoI-based implementation outperforms other state-of-the-art complete verifiers. We also show that our technique can efficiently improve upon the perturbation bound derived by a recent adversarial attack algorithm.

摘要: 受凸优化中不可行和方法的启发，提出了一种分析分段线性激活函数神经网络验证查询的新方法。在给定过逼近非凸激活函数的凸松弛的情况下，将激活函数的违例编码为代价函数，并相对于凸松弛进行优化。成本函数被称为不可行和(SOI)，其最小值被设计为零，并且只有在满足所有激活函数的情况下才能实现。我们提出了一个随机过程，DeepSoI，以有效地最小化SOI。通过用DeepSoI替换在每个搜索状态执行的凸过程，可以实现对基于规范案例分析的完全搜索过程的扩展。使用DeepSoI扩展完整搜索可同时实现多个目标：1)它引导搜索指向反例；2)它支持更知情的分支决策；3)它为界限派生创造更多机会。对不同基准和求解器的广泛评估表明了所提出的技术的好处。特别地，我们证明SOI显著提高了现有完整搜索过程的性能。此外，基于SOI的实现优于其他最先进的完全验证器。我们还表明，我们的技术可以有效地改善最近的对抗性攻击算法得出的扰动界。



## **38. Distinguishing Non-natural from Natural Adversarial Samples for More Robust Pre-trained Language Model**

用于更健壮的预训练语言模型的非自然和自然对抗性样本的区分 cs.LG

Accepted by findings of ACL 2022

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.11199v1)

**Authors**: Jiayi Wang, Rongzhou Bao, Zhuosheng Zhang, Hai Zhao

**Abstracts**: Recently, the problem of robustness of pre-trained language models (PrLMs) has received increasing research interest. Latest studies on adversarial attacks achieve high attack success rates against PrLMs, claiming that PrLMs are not robust. However, we find that the adversarial samples that PrLMs fail are mostly non-natural and do not appear in reality. We question the validity of current evaluation of robustness of PrLMs based on these non-natural adversarial samples and propose an anomaly detector to evaluate the robustness of PrLMs with more natural adversarial samples. We also investigate two applications of the anomaly detector: (1) In data augmentation, we employ the anomaly detector to force generating augmented data that are distinguished as non-natural, which brings larger gains to the accuracy of PrLMs. (2) We apply the anomaly detector to a defense framework to enhance the robustness of PrLMs. It can be used to defend all types of attacks and achieves higher accuracy on both adversarial samples and compliant samples than other defense frameworks.

摘要: 近年来，预训练语言模型(PrLM)的鲁棒性问题引起了越来越多的研究兴趣。最近关于对抗性攻击的研究取得了对PrLM的高攻击成功率，声称PrLM并不健壮。然而，我们发现PrLM失败的对抗性样本大多是非自然的，在现实中并不出现。我们对目前基于这些非自然对抗性样本的PrLM鲁棒性评估的有效性提出了质疑，并提出了一种异常检测器来评估具有更多自然对抗性样本的PrLM的健壮性。我们还研究了异常检测器的两个应用：(1)在数据增强方面，我们使用异常检测器来强制生成被区分为非自然的扩展数据，从而使PrLM的准确率有了较大的提高。(2)将异常检测器应用到防御框架中，增强了PrLM的健壮性。它可以防御所有类型的攻击，在对抗性样本和顺从性样本上都比其他防御框架获得了更高的准确率。



## **39. Adversarial Defense via Image Denoising with Chaotic Encryption**

基于混沌加密图像去噪的对抗性防御 cs.LG

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.10290v1)

**Authors**: Shi Hu, Eric Nalisnick, Max Welling

**Abstracts**: In the literature on adversarial examples, white box and black box attacks have received the most attention. The adversary is assumed to have either full (white) or no (black) access to the defender's model. In this work, we focus on the equally practical gray box setting, assuming an attacker has partial information. We propose a novel defense that assumes everything but a private key will be made available to the attacker. Our framework uses an image denoising procedure coupled with encryption via a discretized Baker map. Extensive testing against adversarial images (e.g. FGSM, PGD) crafted using various gradients shows that our defense achieves significantly better results on CIFAR-10 and CIFAR-100 than the state-of-the-art gray box defenses in both natural and adversarial accuracy.

摘要: 在对抗性例子的文献中，白盒和黑盒攻击受到的关注最多。假设对手对防御者的模型具有完全(白色)或没有(黑色)访问权限。在这项工作中，我们将重点放在同样实用的灰盒设置上，假设攻击者拥有部分信息。我们提出了一种新的防御方案，该方案假定除了私钥之外，攻击者可以使用任何东西。我们的框架使用图像去噪过程，并通过离散化的贝克图进行加密。对使用各种梯度制作的对抗性图像(例如FGSM、PGD)的广泛测试表明，我们的防御在自然和对抗性准确性方面在CIFAR-10和CIFAR-100上都比最先进的灰盒防御取得了显著更好的结果。



## **40. Synthesis of the Supremal Covert Attacker Against Unknown Supervisors by Using Observations**

利用观测值合成对抗未知监督者的最高隐蔽性攻击者 eess.SY

arXiv admin note: text overlap with arXiv:2106.12268

**SubmitDate**: 2022-03-19    [paper-pdf](http://arxiv.org/pdf/2203.08360v2)

**Authors**: Ruochen Tai, Liyong Lin, Yuting Zhu, Rong Su

**Abstracts**: In this paper, we consider the problem of synthesizing the supremal covert damage-reachable attacker, in the setup where the model of the supervisor is unknown to the adversary but the adversary has recorded a (prefix-closed) finite set of observations of the runs of the closed-loop system. The synthesized attacker needs to ensure both the damage-reachability and the covertness against all the supervisors which are consistent with the given set of observations. There is a gap between the de facto supremality, assuming the model of the supervisor is known, and the supremality that can be attained with a limited knowledge of the model of the supervisor, from the adversary's point of view. We consider the setup where the attacker can exercise sensor replacement/deletion attacks and actuator enablement/disablement attacks. The solution methodology proposed in this work is to reduce the synthesis of the supremal covert damage-reachable attacker, given the model of the plant and the finite set of observations, to the synthesis of the supremal safe supervisor for certain transformed plant, which shows the decidability of the observation-assisted covert attacker synthesis problem. The effectiveness of our approach is illustrated on a water tank example adapted from the literature.

摘要: 在这篇文章中，我们考虑了在对手未知监督者模型但对手已经记录了闭环系统运行的(前缀闭合的)有限观测集合的情况下，合成最高隐蔽损害可达攻击者的问题。合成攻击者需要确保针对所有监督者的损害可达性和隐蔽性，这与给定的观察集合一致。在假设监督者的模式已知的情况下，事实上的至高无上与从对手的角度看，通过对监督者的模式的有限了解可以获得的至高无上之间存在差距。我们考虑攻击者可以执行传感器替换/删除攻击和致动器启用/禁用攻击的设置。本文提出的解决方法是，在给定对象模型和有限观测集的情况下，将隐蔽可达攻击者的综合归结为某一变换对象的最高安全监督器的综合，从而证明了观测辅助的隐蔽攻击者综合问题的可判性。从文献中改编的一个水箱例子说明了我们方法的有效性。



## **41. Adversarial Attacks on Deep Learning-based Video Compression and Classification Systems**

基于深度学习的视频压缩和分类系统的敌意攻击 cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.10183v1)

**Authors**: Jung-Woo Chang, Mojan Javaheripi, Seira Hidano, Farinaz Koushanfar

**Abstracts**: Video compression plays a crucial role in enabling video streaming and classification systems and maximizing the end-user quality of experience (QoE) at a given bandwidth budget. In this paper, we conduct the first systematic study for adversarial attacks on deep learning based video compression and downstream classification systems. We propose an adaptive adversarial attack that can manipulate the Rate-Distortion (R-D) relationship of a video compression model to achieve two adversarial goals: (1) increasing the network bandwidth or (2) degrading the video quality for end-users. We further devise novel objectives for targeted and untargeted attacks to a downstream video classification service. Finally, we design an input-invariant perturbation that universally disrupts video compression and classification systems in real time. Unlike previously proposed attacks on video classification, our adversarial perturbations are the first to withstand compression. We empirically show the resilience of our attacks against various defenses, i.e., adversarial training, video denoising, and JPEG compression. Our extensive experimental results on various video datasets demonstrate the effectiveness of our attacks. Our video quality and bandwidth attacks deteriorate peak signal-to-noise ratio by up to 5.4dB and the bit-rate by up to 2.4 times on the standard video compression datasets while achieving over 90% attack success rate on a downstream classifier.

摘要: 视频压缩在实现视频流和分类系统以及在给定带宽预算下最大化最终用户体验质量(QoE)方面起着至关重要的作用。本文首次对基于深度学习的视频压缩和下行分类系统的敌意攻击进行了系统的研究。本文提出了一种自适应对抗性攻击方法，可以通过操控视频压缩模型的率失真(R-D)关系来实现两个对抗性目标：(1)增加网络带宽；(2)降低终端用户的视频质量。我们进一步设计了针对下游视频分类服务的定向和非定向攻击的新目标。最后，我们设计了一种输入不变的扰动，该扰动普遍地扰乱了视频压缩和分类系统的实时运行。与之前提出的针对视频分类的攻击不同，我们的对抗性扰动最先经受住了压缩。我们经验性地展示了我们的攻击对各种防御的恢复能力，即对抗性训练、视频去噪和JPEG压缩。我们在各种视频数据集上的大量实验结果证明了我们攻击的有效性。我们的视频质量和带宽攻击在标准视频压缩数据集上使峰值信噪比降低了5.4dB，比特率降低了2.4倍，同时在下游分类器上实现了90%以上的攻击成功率。



## **42. Concept-based Adversarial Attacks: Tricking Humans and Classifiers Alike**

基于概念的对抗性攻击：欺骗人类和分类器 cs.LG

Accepted at IEEE Symposium on Security and Privacy (S&P) Workshop on  Deep Learning and Security, 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.10166v1)

**Authors**: Johannes Schneider, Giovanni Apruzzese

**Abstracts**: We propose to generate adversarial samples by modifying activations of upper layers encoding semantically meaningful concepts. The original sample is shifted towards a target sample, yielding an adversarial sample, by using the modified activations to reconstruct the original sample. A human might (and possibly should) notice differences between the original and the adversarial sample. Depending on the attacker-provided constraints, an adversarial sample can exhibit subtle differences or appear like a "forged" sample from another class. Our approach and goal are in stark contrast to common attacks involving perturbations of single pixels that are not recognizable by humans. Our approach is relevant in, e.g., multi-stage processing of inputs, where both humans and machines are involved in decision-making because invisible perturbations will not fool a human. Our evaluation focuses on deep neural networks. We also show the transferability of our adversarial examples among networks.

摘要: 我们建议通过修改编码有语义意义的概念的上层激活来生成对抗性样本。通过使用修改的激活来重构原始样本，原始样本被移向目标样本，从而产生对抗性样本。人类可能(也可能应该)注意到原始样本和对抗性样本之间的差异。根据攻击者提供的约束，敌意样本可能会显示出细微的差异，或者看起来像是来自另一个类的“伪造”样本。我们的方法和目标与常见的攻击形成鲜明对比，这些攻击涉及人类无法识别的单像素扰动。例如，我们的方法与输入的多阶段处理相关，其中人和机器都参与决策，因为看不见的扰动不会愚弄人。我们的评估集中在深度神经网络上。我们还展示了我们的对抗性例子在网络之间的可转移性。



## **43. All You Need is RAW: Defending Against Adversarial Attacks with Camera Image Pipelines**

您所需要的只是RAW：使用摄像机图像管道防御敌意攻击 cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2112.09219v2)

**Authors**: Yuxuan Zhang, Bo Dong, Felix Heide

**Abstracts**: Existing neural networks for computer vision tasks are vulnerable to adversarial attacks: adding imperceptible perturbations to the input images can fool these methods to make a false prediction on an image that was correctly predicted without the perturbation. Various defense methods have proposed image-to-image mapping methods, either including these perturbations in the training process or removing them in a preprocessing denoising step. In doing so, existing methods often ignore that the natural RGB images in today's datasets are not captured but, in fact, recovered from RAW color filter array captures that are subject to various degradations in the capture. In this work, we exploit this RAW data distribution as an empirical prior for adversarial defense. Specifically, we proposed a model-agnostic adversarial defensive method, which maps the input RGB images to Bayer RAW space and back to output RGB using a learned camera image signal processing (ISP) pipeline to eliminate potential adversarial patterns. The proposed method acts as an off-the-shelf preprocessing module and, unlike model-specific adversarial training methods, does not require adversarial images to train. As a result, the method generalizes to unseen tasks without additional retraining. Experiments on large-scale datasets (e.g., ImageNet, COCO) for different vision tasks (e.g., classification, semantic segmentation, object detection) validate that the method significantly outperforms existing methods across task domains.

摘要: 现有的用于计算机视觉任务的神经网络容易受到敌意攻击：向输入图像添加不可察觉的扰动可以欺骗这些方法在没有扰动的情况下对正确预测的图像进行错误预测。各种防御方法已经提出了图像到图像的映射方法，或者在训练过程中包括这些扰动，或者在预处理去噪步骤中去除它们。在这样做时，现有方法通常忽略没有捕获今天数据集中的自然的rgb图像，而实际上是从在捕获中遭受各种降级的原始颜色过滤阵列捕获中恢复的。在这项工作中，我们利用这个原始数据分布作为对抗防御的经验先验。具体地说，我们提出了一种模型不可知的对抗防御方法，该方法将输入的RGB图像映射到拜耳原始空间，然后使用学习的摄像机图像信号处理(ISP)流水线将输入的RGB图像映射回输出RGB，以消除潜在的敌对模式。该方法作为一个现成的预处理模块，与特定模型的对抗性训练方法不同，不需要对抗性图像进行训练。因此，该方法可以推广到看不见的任务，而不需要额外的再培训。在不同视觉任务(如分类、语义分割、目标检测)的大规模数据集(如ImageNet、CoCo)上的实验验证了该方法在跨任务域的性能上显著优于现有方法。



## **44. Graph-Fraudster: Adversarial Attacks on Graph Neural Network Based Vertical Federated Learning**

图欺诈者：基于图神经网络垂直联合学习的对抗性攻击 cs.LG

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2110.06468v2)

**Authors**: Jinyin Chen, Guohan Huang, Haibin Zheng, Shanqing Yu, Wenrong Jiang, Chen Cui

**Abstracts**: Graph neural network (GNN) has achieved great success on graph representation learning. Challenged by large scale private data collected from user-side, GNN may not be able to reflect the excellent performance, without rich features and complete adjacent relationships. Addressing the problem, vertical federated learning (VFL) is proposed to implement local data protection through training a global model collaboratively. Consequently, for graph-structured data, it is a natural idea to construct a GNN based VFL framework, denoted as GVFL. However, GNN has been proved vulnerable to adversarial attacks. Whether the vulnerability will be brought into the GVFL has not been studied. This is the first study of adversarial attacks on GVFL. A novel adversarial attack method is proposed, named Graph-Fraudster. It generates adversarial perturbations based on the noise-added global node embeddings via the privacy leakage and the gradient of pairwise node. Specifically, first, Graph-Fraudster steals the global node embeddings and sets up a shadow model of the server for the attack generator. Second, noise is added into node embeddings to confuse the shadow model. At last, the gradient of pairwise node is used to generate attacks with the guidance of noise-added node embeddings. Extensive experiments on five benchmark datasets demonstrate that Graph-Fraudster achieves the state-of-the-art attack performance compared with baselines in different GNN based GVFLs. Furthermore, Graph-Fraudster can remain a threat to GVFL even if two possible defense mechanisms are applied. Additionally, some suggestions are put forward for the future work to improve the robustness of GVFL. The code and datasets can be downloaded at https://github.com/hgh0545/Graph-Fraudster.

摘要: 图神经网络(GNN)在图表示学习方面取得了巨大的成功。由于受到来自用户端的大规模隐私数据的挑战，GNN如果没有丰富的功能和完整的邻接关系，可能无法反映出优秀的性能。针对这一问题，提出了垂直联合学习(VFL)，通过协作训练全局模型来实现局部数据保护。因此，对于图结构的数据，构建基于GNN的VFL框架(表示为GVFL)是一个自然的想法。然而，GNN已被证明容易受到敌意攻击。该漏洞是否会被带入GVFL还没有研究。这是首次对GVFL进行对抗性攻击的研究。提出了一种新的对抗性攻击方法--图欺诈器。该算法通过两个结点的隐私泄露和梯度，在加入噪声的全局结点嵌入的基础上产生对抗性扰动。具体地说，首先，Graph欺诈者窃取全局节点嵌入，并为攻击生成器建立服务器的影子模型。其次，在节点嵌入中加入噪声以混淆阴影模型。最后，在加入噪声的节点嵌入指导下，利用成对节点的梯度生成攻击。在五个基准数据集上的大量实验表明，与基线相比，Graph-Fraudster在不同的基于GNN的GVFL上达到了最先进的攻击性能。此外，即使应用了两种可能的防御机制，图形欺诈者仍可能对GVFL构成威胁。此外，对未来的工作提出了一些建议，以提高GVFL的鲁棒性。代码和数据集可从https://github.com/hgh0545/Graph-Fraudster.下载



## **45. Defending Variational Autoencoders from Adversarial Attacks with MCMC**

利用MCMC保护可变自动编码器免受敌意攻击 cs.LG

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09940v1)

**Authors**: Anna Kuzina, Max Welling, Jakub M. Tomczak

**Abstracts**: Variational autoencoders (VAEs) are deep generative models used in various domains. VAEs can generate complex objects and provide meaningful latent representations, which can be further used in downstream tasks such as classification. As previous work has shown, one can easily fool VAEs to produce unexpected latent representations and reconstructions for a visually slightly modified input. Here, we examine several objective functions for adversarial attacks construction, suggest metrics assess the model robustness, and propose a solution to alleviate the effect of an attack. Our method utilizes the Markov Chain Monte Carlo (MCMC) technique in the inference step and is motivated by our theoretical analysis. Thus, we do not incorporate any additional costs during training or we do not decrease the performance on non-attacked inputs. We validate our approach on a variety of datasets (MNIST, Fashion MNIST, Color MNIST, CelebA) and VAE configurations ($\beta$-VAE, NVAE, TC-VAE) and show that it consistently improves the model robustness to adversarial attacks.

摘要: 变分自动编码器(VAE)是广泛应用于各个领域的深度产生式模型。Vaes可以生成复杂的对象，并提供有意义的潜在表示，这可以进一步用于下游任务，如分类。正如以前的工作所表明的那样，人们可以很容易地欺骗VAE为视觉上稍有修改的输入产生意想不到的潜在表示和重建。在这里，我们检查了几个对抗性攻击构建的目标函数，提出了评估模型健壮性的度量标准，并提出了一种减轻攻击效果的解决方案。我们的方法在推理步骤中使用了马尔可夫链蒙特卡罗(MCMC)技术，并基于我们的理论分析。因此，我们在培训期间不会加入任何额外成本，或者我们不会降低非攻击输入的性能。我们在各种数据集(MNIST，Fashion MNIST，Color MNIST，CelebA)和VAE配置($\beta$-VAE，NVAE，TC-VAE)上验证了我们的方法，并表明它一致地提高了模型对对手攻击的鲁棒性。



## **46. Neural Predictor for Black-Box Adversarial Attacks on Speech Recognition**

语音识别黑盒对抗性攻击的神经预测器 cs.SD

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09849v1)

**Authors**: Marie Biolková, Bac Nguyen

**Abstracts**: Recent works have revealed the vulnerability of automatic speech recognition (ASR) models to adversarial examples (AEs), i.e., small perturbations that cause an error in the transcription of the audio signal. Studying audio adversarial attacks is therefore the first step towards robust ASR. Despite the significant progress made in attacking audio examples, the black-box attack remains challenging because only the hard-label information of transcriptions is provided. Due to this limited information, existing black-box methods often require an excessive number of queries to attack a single audio example. In this paper, we introduce NP-Attack, a neural predictor-based method, which progressively evolves the search towards a small adversarial perturbation. Given a perturbation direction, our neural predictor directly estimates the smallest perturbation that causes a mistranscription. In particular, it enables NP-Attack to accurately learn promising perturbation directions via gradient-based optimization. Experimental results show that NP-Attack achieves competitive results with other state-of-the-art black-box adversarial attacks while requiring a significantly smaller number of queries. The code of NP-Attack is available online.

摘要: 最近的工作揭示了自动语音识别(ASR)模型对对抗性示例(AE)的脆弱性，即导致音频信号转录错误的微小扰动。因此，研究音频对抗性攻击是迈向稳健ASR的第一步。尽管在攻击音频样本方面取得了重大进展，但黑盒攻击仍然具有挑战性，因为只提供了转录的硬标签信息。由于这些有限的信息，现有的黑盒方法通常需要过多的查询来攻击单个音频示例。在本文中，我们引入了NP-攻击，这是一种基于神经预测器的方法，它将搜索逐步演化为一个小的对抗性扰动。给定一个扰动方向，我们的神经预测器会直接估计导致误译的最小扰动。特别是，它使NP-Attack能够通过基于梯度的优化准确地学习有希望的扰动方向。实验结果表明，NP-Attack在查询次数明显减少的情况下，达到了与其他黑盒对抗性攻击相当的效果。NP-Attack的代码可以在网上找到。



## **47. DTA: Physical Camouflage Attacks using Differentiable Transformation Network**

DTA：基于微分变换网络的物理伪装攻击 cs.CV

Accepted for CVPR 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09831v1)

**Authors**: Naufal Suryanto, Yongsu Kim, Hyoeun Kang, Harashta Tatimma Larasati, Youngyeo Yun, Thi-Thu-Huong Le, Hunmin Yang, Se-Yoon Oh, Howon Kim

**Abstracts**: To perform adversarial attacks in the physical world, many studies have proposed adversarial camouflage, a method to hide a target object by applying camouflage patterns on 3D object surfaces. For obtaining optimal physical adversarial camouflage, previous studies have utilized the so-called neural renderer, as it supports differentiability. However, existing neural renderers cannot fully represent various real-world transformations due to a lack of control of scene parameters compared to the legacy photo-realistic renderers. In this paper, we propose the Differentiable Transformation Attack (DTA), a framework for generating a robust physical adversarial pattern on a target object to camouflage it against object detection models with a wide range of transformations. It utilizes our novel Differentiable Transformation Network (DTN), which learns the expected transformation of a rendered object when the texture is changed while preserving the original properties of the target object. Using our attack framework, an adversary can gain both the advantages of the legacy photo-realistic renderers including various physical-world transformations and the benefit of white-box access by offering differentiability. Our experiments show that our camouflaged 3D vehicles can successfully evade state-of-the-art object detection models in the photo-realistic environment (i.e., CARLA on Unreal Engine). Furthermore, our demonstration on a scaled Tesla Model 3 proves the applicability and transferability of our method to the real world.

摘要: 为了在物理世界中进行对抗性攻击，许多研究提出了对抗性伪装，即通过在三维物体表面应用伪装图案来隐藏目标对象的方法。为了获得最佳的物理对抗伪装，以前的研究已经利用了所谓的神经渲染器，因为它支持可分性。然而，与传统的照片真实感渲染器相比，由于缺乏对场景参数的控制，现有的神经渲染器不能完全表示各种真实世界的变换。本文提出了差分变换攻击(Differentiable Transform Attack，DTA)的框架，该框架用于在目标对象上生成一个健壮的物理对抗模式，以针对具有广泛变换的目标检测模型进行伪装。它利用了我们新颖的微分变换网络(DTN)，它在保持目标对象的原始属性的同时，学习绘制对象在纹理改变时的期望变换。使用我们的攻击框架，对手既可以获得传统照片真实感渲染器(包括各种物理世界变换)的优势，也可以通过提供差异化获得白盒访问的好处。我们的实验表明，我们的伪装3D车辆可以在真实感环境(即虚幻引擎上的Carla)中成功地躲避最先进的目标检测模型。此外，我们在一个缩放的特斯拉Model3上的演示证明了我们的方法在现实世界中的适用性和可移植性。



## **48. AdIoTack: Quantifying and Refining Resilience of Decision Tree Ensemble Inference Models against Adversarial Volumetric Attacks on IoT Networks**

AdIoTack：量化和提炼决策树集成推理模型对物联网敌意体积攻击的恢复能力 cs.LG

15 pages, 16 figures, 4 tables

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09792v1)

**Authors**: Arman Pashamokhtari, Gustavo Batista, Hassan Habibi Gharakheili

**Abstracts**: Machine Learning-based techniques have shown success in cyber intelligence. However, they are increasingly becoming targets of sophisticated data-driven adversarial attacks resulting in misprediction, eroding their ability to detect threats on network devices. In this paper, we present AdIoTack, a system that highlights vulnerabilities of decision trees against adversarial attacks, helping cybersecurity teams quantify and refine the resilience of their trained models for monitoring IoT networks. To assess the model for the worst-case scenario, AdIoTack performs white-box adversarial learning to launch successful volumetric attacks that decision tree ensemble models cannot flag. Our first contribution is to develop a white-box algorithm that takes a trained decision tree ensemble model and the profile of an intended network-based attack on a victim class as inputs. It then automatically generates recipes that specify certain packets on top of the indented attack packets (less than 15% overhead) that together can bypass the inference model unnoticed. We ensure that the generated attack instances are feasible for launching on IP networks and effective in their volumetric impact. Our second contribution develops a method to monitor the network behavior of connected devices actively, inject adversarial traffic (when feasible) on behalf of a victim IoT device, and successfully launch the intended attack. Our third contribution prototypes AdIoTack and validates its efficacy on a testbed consisting of a handful of real IoT devices monitored by a trained inference model. We demonstrate how the model detects all non-adversarial volumetric attacks on IoT devices while missing many adversarial ones. The fourth contribution develops systematic methods for applying patches to trained decision tree ensemble models, improving their resilience against adversarial volumetric attacks.

摘要: 基于机器学习的技术在网络智能领域取得了成功。然而，它们越来越多地成为复杂的数据驱动的敌意攻击的目标，导致预测错误，侵蚀了它们检测网络设备上的威胁的能力。在本文中，我们提出了AdIoTack，这是一个突出决策树针对对手攻击的漏洞的系统，帮助网络安全团队量化和改进他们训练的物联网网络监控模型的弹性。为了评估最坏情况下的模型，AdIoTack执行白盒对抗性学习，以发起决策树集成模型无法标记的成功体积攻击。我们的第一个贡献是开发了一个白盒算法，该算法将经过训练的决策树集成模型和对受害者类别的预期网络攻击的概况作为输入。然后，它自动生成配方，在缩进的攻击数据包(开销不到15%)之上指定某些数据包，这些数据包一起可以绕过推理模型而不会被注意到。我们确保生成的攻击实例可以在IP网络上启动，并有效地发挥其体积影响。我们的第二个贡献开发了一种方法，可以主动监控连接设备的网络行为，代表受害者物联网设备注入敌意流量(如果可行)，并成功发起预期攻击。我们的第三个贡献是AdIoTack的原型，并在由几个由训练有素的推理模型监控的真实物联网设备组成的试验台上验证了其有效性。我们演示了该模型如何检测物联网设备上的所有非对抗性体积攻击，而同时遗漏了许多对抗性攻击。第四个贡献开发了将补丁应用于训练的决策树集成模型的系统方法，提高了它们对对手体积攻击的弹性。



## **49. Adversarial Texture for Fooling Person Detectors in the Physical World**

物理世界中愚人探测器的对抗性纹理 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.03373v3)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Xiaolin Hu, Fuchun Sun, Bo Zhang

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.

摘要: 如今，配备人工智能系统的摄像头可以捕捉和分析图像，自动检测人。然而，当接收到现实世界中故意设计的模式时，人工智能系统可能会出错，即物理对抗性示例。以前的工作已经表明，可以在衣服上打印敌意补丁来躲避基于DNN的人检测器。然而，当视角(即相机朝向对象的角度)改变时，这些对抗性的例子可能会使攻击成功率灾难性地下降。为了进行多角度攻击，我们提出了对抗性纹理(AdvTexture)。AdvTexture可以覆盖任意形状的衣服，这样穿着这种衣服的人就可以从不同的视角躲避人的探测器。提出了一种基于环形裁剪的可扩展生成攻击方法(TC-EGA)来制作具有重复结构的AdvTexture。我们用AdvTexure打印了几块布，然后在现实世界中制作了T恤、裙子和连衣裙。实验表明，这些衣服可以愚弄物理世界中的人体探测器。



## **50. AutoAdversary: A Pixel Pruning Method for Sparse Adversarial Attack**

AutoAdversary：一种稀疏对抗性攻击的像素剪枝方法 cs.CV

**SubmitDate**: 2022-03-18    [paper-pdf](http://arxiv.org/pdf/2203.09756v1)

**Authors**: Jinqiao Li, Xiaotao Liu, Jian Zhao, Furao Shen

**Abstracts**: Deep neural networks (DNNs) have been proven to be vulnerable to adversarial examples. A special branch of adversarial examples, namely sparse adversarial examples, can fool the target DNNs by perturbing only a few pixels. However, many existing sparse adversarial attacks use heuristic methods to select the pixels to be perturbed, and regard the pixel selection and the adversarial attack as two separate steps. From the perspective of neural network pruning, we propose a novel end-to-end sparse adversarial attack method, namely AutoAdversary, which can find the most important pixels automatically by integrating the pixel selection into the adversarial attack. Specifically, our method utilizes a trainable neural network to generate a binary mask for the pixel selection. After jointly optimizing the adversarial perturbation and the neural network, only the pixels corresponding to the value 1 in the mask are perturbed. Experiments demonstrate the superiority of our proposed method over several state-of-the-art methods. Furthermore, since AutoAdversary does not require a heuristic pixel selection process, it does not slow down excessively as other methods when the image size increases.

摘要: 深度神经网络(DNNs)已被证明容易受到敌意例子的攻击。对抗性示例的一个特殊分支，即稀疏对抗性示例，可以通过仅扰动几个像素来欺骗目标DNN。然而，现有的许多稀疏对抗性攻击采用启发式方法选择要扰动的像素，将像素选择和对抗性攻击视为两个独立的步骤。从神经网络剪枝的角度出发，提出了一种新的端到端稀疏对抗攻击方法--AutoAdversary，该方法将像素选择融入到对抗攻击中，能够自动找到最重要的像素。具体地说，我们的方法利用一个可训练的神经网络来生成用于像素选择的二值掩码。在联合优化对抗性扰动和神经网络之后，仅对掩码中与值1对应的像素进行扰动。实验表明，我们提出的方法比几种最先进的方法具有更好的性能。此外，由于AutoAdversary不需要启发式像素选择过程，因此当图像大小增加时，它不会像其他方法那样过度减速。



