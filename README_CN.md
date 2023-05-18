# Latest Adversarial Attack Papers
**update at 2023-05-18 10:39:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Raising the Bar for Certified Adversarial Robustness with Diffusion Models**

利用扩散模型提高认证对抗性稳健性的标准 cs.LG

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2305.10388v1) [paper-pdf](http://arxiv.org/pdf/2305.10388v1)

**Authors**: Thomas Altstidl, David Dobre, Björn Eskofier, Gauthier Gidel, Leo Schwinn

**Abstract**: Certified defenses against adversarial attacks offer formal guarantees on the robustness of a model, making them more reliable than empirical methods such as adversarial training, whose effectiveness is often later reduced by unseen attacks. Still, the limited certified robustness that is currently achievable has been a bottleneck for their practical adoption. Gowal et al. and Wang et al. have shown that generating additional training data using state-of-the-art diffusion models can considerably improve the robustness of adversarial training. In this work, we demonstrate that a similar approach can substantially improve deterministic certified defenses. In addition, we provide a list of recommendations to scale the robustness of certified training approaches. One of our main insights is that the generalization gap, i.e., the difference between the training and test accuracy of the original model, is a good predictor of the magnitude of the robustness improvement when using additional generated data. Our approach achieves state-of-the-art deterministic robustness certificates on CIFAR-10 for the $\ell_2$ ($\epsilon = 36/255$) and $\ell_\infty$ ($\epsilon = 8/255$) threat models, outperforming the previous best results by $+3.95\%$ and $+1.39\%$, respectively. Furthermore, we report similar improvements for CIFAR-100.

摘要: 针对对抗性攻击的认证防御为模型的健壮性提供了正式保证，使它们比对抗性训练等经验方法更可靠，后者的有效性后来往往因看不见的攻击而降低。尽管如此，目前可以实现的有限的认证健壮性一直是它们实际采用的瓶颈。Gowal等人。和Wang等人。已经表明，使用最先进的扩散模型生成额外的训练数据可以显著提高对抗性训练的稳健性。在这项工作中，我们证明了类似的方法可以实质性地改进确定性认证防御。此外，我们还提供了一系列建议，以衡量认证培训方法的健壮性。我们的主要见解之一是，泛化差距，即原始模型的训练精度和测试精度之间的差异，是使用额外生成的数据时稳健性改善幅度的一个很好的预测指标。我们的方法在CIFAR-10上为$\ell_2$($\epsilon=36/255$)和$\ell_inty$($\epsilon=8/255$)威胁模型获得了最先进的确定性稳健性证书，分别比之前的最好结果高出$+3.95\$和$+1.39\$。此外，我们还报告了CIFAR-100的类似改进。



## **2. Certified Invertibility in Neural Networks via Mixed-Integer Programming**

基于混合整数规划的神经网络可逆性证明 cs.LG

22 pages, 7 figures

**SubmitDate**: 2023-05-17    [abs](http://arxiv.org/abs/2301.11783v2) [paper-pdf](http://arxiv.org/pdf/2301.11783v2)

**Authors**: Tianqi Cui, Thomas Bertalan, George J. Pappas, Manfred Morari, Ioannis G. Kevrekidis, Mahyar Fazlyab

**Abstract**: Neural networks are known to be vulnerable to adversarial attacks, which are small, imperceptible perturbations that can significantly alter the network's output. Conversely, there may exist large, meaningful perturbations that do not affect the network's decision (excessive invariance). In our research, we investigate this latter phenomenon in two contexts: (a) discrete-time dynamical system identification, and (b) the calibration of a neural network's output to that of another network. We examine noninvertibility through the lens of mathematical optimization, where the global solution measures the ``safety" of the network predictions by their distance from the non-invertibility boundary. We formulate mixed-integer programs (MIPs) for ReLU networks and $L_p$ norms ($p=1,2,\infty$) that apply to neural network approximators of dynamical systems. We also discuss how our findings can be useful for invertibility certification in transformations between neural networks, e.g. between different levels of network pruning.

摘要: 众所周知，神经网络容易受到对抗性攻击，这种攻击是微小的、不可察觉的扰动，可以显著改变网络的输出。相反，可能存在不影响网络决策的大的、有意义的扰动(过度的不变性)。在我们的研究中，我们在两个背景下研究后一种现象：(A)离散时间动态系统辨识，和(B)神经网络输出到另一网络的校准。我们通过数学最优化的视角研究不可逆性，其中全局解通过网络预测到不可逆边界的距离来衡量网络预测的“安全性”。我们为RELU网络和$L_p$范数($p=1，2，\inty$)建立了混合整数规划(MIP)。我们还讨论了我们的结果如何用于神经网络之间的变换，例如在不同级别的网络剪枝之间的可逆性证明。



## **3. Manipulating Visually-aware Federated Recommender Systems and Its Countermeasures**

操纵视觉感知的联邦推荐系统及其对策 cs.IR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.08183v2) [paper-pdf](http://arxiv.org/pdf/2305.08183v2)

**Authors**: Wei Yuan, Shilong Yuan, Chaoqun Yang, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Federated recommender systems (FedRecs) have been widely explored recently due to their ability to protect user data privacy. In FedRecs, a central server collaboratively learns recommendation models by sharing model public parameters with clients, thereby offering a privacy-preserving solution. Unfortunately, the exposure of model parameters leaves a backdoor for adversaries to manipulate FedRecs. Existing works about FedRec security already reveal that items can easily be promoted by malicious users via model poisoning attacks, but all of them mainly focus on FedRecs with only collaborative information (i.e., user-item interactions). We argue that these attacks are effective because of the data sparsity of collaborative signals. In practice, auxiliary information, such as products' visual descriptions, is used to alleviate collaborative filtering data's sparsity. Therefore, when incorporating visual information in FedRecs, all existing model poisoning attacks' effectiveness becomes questionable. In this paper, we conduct extensive experiments to verify that incorporating visual information can beat existing state-of-the-art attacks in reasonable settings. However, since visual information is usually provided by external sources, simply including it will create new security problems. Specifically, we propose a new kind of poisoning attack for visually-aware FedRecs, namely image poisoning attacks, where adversaries can gradually modify the uploaded image to manipulate item ranks during FedRecs' training process. Furthermore, we reveal that the potential collaboration between image poisoning attacks and model poisoning attacks will make visually-aware FedRecs more vulnerable to being manipulated. To safely use visual information, we employ a diffusion model in visually-aware FedRecs to purify each uploaded image and detect the adversarial images.

摘要: 联邦推荐系统(FedRecs)由于具有保护用户数据隐私的能力，近年来得到了广泛的研究。在FedRecs中，中央服务器通过与客户共享模型公共参数来协作学习推荐模型，从而提供隐私保护解决方案。不幸的是，模型参数的曝光为对手操纵FedRecs留下了后门。已有的关于FedRec安全的研究已经表明，恶意用户很容易通过模型中毒攻击来推销物品，但这些研究主要集中在只有协作信息的FedRecs上(即用户与物品的交互)。我们认为，由于协同信号的数据稀疏性，这些攻击是有效的。在实际应用中，产品的视觉描述等辅助信息被用来缓解协同过滤数据的稀疏性。因此，当在FedRecs中加入视觉信息时，所有现有的模型中毒攻击的有效性都会受到质疑。在本文中，我们进行了大量的实验，以验证在合理的设置下，结合视觉信息可以抵抗现有的最先进的攻击。然而，由于可视信息通常由外部来源提供，简单地将其包括在内将会产生新的安全问题。具体地说，我们提出了一种新的针对视觉感知FedRecs的中毒攻击，即图像中毒攻击，在FedRecs的训练过程中，攻击者可以逐渐修改上传的图像来操纵物品等级。此外，我们揭示了图像中毒攻击和模型中毒攻击之间的潜在合作将使视觉感知的FedRecs更容易被操纵。为了安全地使用视觉信息，我们在视觉感知的FedRecs中使用了扩散模型来净化每一张上传的图像并检测出恶意图像。



## **4. A theoretical basis for Blockchain Extractable Value**

区块链可提取价值的理论基础 cs.CR

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02154v3) [paper-pdf](http://arxiv.org/pdf/2302.02154v3)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Extractable Value refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream protocols, like e.g. decentralized exchanges, are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the growing impact of these attacks in the real world, theoretical foundations are still missing. We propose a formal theory of Extractable Value, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against Extractable Value attacks.

摘要: 可提取价值指的是对公共区块链的一大类经济攻击，在这些攻击中，有能力在区块中重新排序、丢弃或插入交易的对手可以从智能合约中“提取”价值。经验研究表明，主流协议，如分散交换，是这些攻击的大规模目标，对其用户和区块链网络造成有害影响。尽管这些袭击在现实世界中的影响越来越大，但理论基础仍然缺乏。基于区块链和智能合约的一般抽象模型，我们提出了可提取价值的形式理论。我们的理论是针对可提取值攻击的安全性证明的基础。



## **5. Exploring the Connection between Robust and Generative Models**

探索健壮性模型和生成性模型之间的联系 cs.LG

technical report, 6 pages, 6 figures

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2304.04033v3) [paper-pdf](http://arxiv.org/pdf/2304.04033v3)

**Authors**: Senad Beadini, Iacopo Masi

**Abstract**: We offer a study that connects robust discriminative classifiers trained with adversarial training (AT) with generative modeling in the form of Energy-based Models (EBM). We do so by decomposing the loss of a discriminative classifier and showing that the discriminative model is also aware of the input data density. Though a common assumption is that adversarial points leave the manifold of the input data, our study finds out that, surprisingly, untargeted adversarial points in the input space are very likely under the generative model hidden inside the discriminative classifier -- have low energy in the EBM. We present two evidence: untargeted attacks are even more likely than the natural data and their likelihood increases as the attack strength increases. This allows us to easily detect them and craft a novel attack called High-Energy PGD that fools the classifier yet has energy similar to the data set.

摘要: 我们提供了一项研究，将经过对抗性训练(AT)训练的稳健区分分类器与基于能量的模型(EBM)形式的生成性建模相结合。我们通过分解判别分类器的损失来做到这一点，并表明判别模型也知道输入数据的密度。虽然一个普遍的假设是敌对点离开了输入数据的流形，但我们的研究发现，令人惊讶的是，在隐藏在判别分类器中的生成模型下，输入空间中的非目标对抗性点很可能在EBM中具有低能量。我们提出了两个证据：非目标攻击的可能性甚至比自然数据更高，并且随着攻击强度的增加，它们的可能性也会增加。这使我们能够轻松地检测到它们，并创建一种名为高能PGD的新型攻击，它愚弄了分类器，但具有与数据集相似的能量。



## **6. Unlearnable Examples Give a False Sense of Security: Piercing through Unexploitable Data with Learnable Examples**

无法学习的例子给人一种错误的安全感：用可学习的例子穿透不可利用的数据 cs.LG

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09241v1) [paper-pdf](http://arxiv.org/pdf/2305.09241v1)

**Authors**: Wan Jiang, Yunfeng Diao, He Wang, Jianxin Sun, Meng Wang, Richang Hong

**Abstract**: Safeguarding data from unauthorized exploitation is vital for privacy and security, especially in recent rampant research in security breach such as adversarial/membership attacks. To this end, \textit{unlearnable examples} (UEs) have been recently proposed as a compelling protection, by adding imperceptible perturbation to data so that models trained on them cannot classify them accurately on original clean distribution. Unfortunately, we find UEs provide a false sense of security, because they cannot stop unauthorized users from utilizing other unprotected data to remove the protection, by turning unlearnable data into learnable again. Motivated by this observation, we formally define a new threat by introducing \textit{learnable unauthorized examples} (LEs) which are UEs with their protection removed. The core of this approach is a novel purification process that projects UEs onto the manifold of LEs. This is realized by a new joint-conditional diffusion model which denoises UEs conditioned on the pixel and perceptual similarity between UEs and LEs. Extensive experiments demonstrate that LE delivers state-of-the-art countering performance against both supervised UEs and unsupervised UEs in various scenarios, which is the first generalizable countermeasure to UEs across supervised learning and unsupervised learning.

摘要: 保护数据不被未经授权的利用对隐私和安全至关重要，特别是在最近对安全漏洞的猖獗研究中，例如对抗性/成员攻击。为此，最近提出了不可学习的例子(UE)作为一种强制保护，通过向数据添加不可察觉的扰动，使得训练在这些数据上的模型不能根据原始的干净分布对它们进行准确的分类。不幸的是，我们发现UE提供了一种错误的安全感，因为它们无法阻止未经授权的用户利用其他不受保护的数据来取消保护，方法是将无法学习的数据再次变为可学习的数据。受此观察的启发，我们正式定义了一种新的威胁，引入了去除了保护的可学习未经授权示例(LES)。这种方法的核心是一种新颖的净化过程，将UE投射到LES的流形上。这是通过一种新的联合条件扩散模型来实现的，该模型根据UE和LES之间的像素和感知相似性来对UE进行去噪。大量的实验表明，在不同的场景下，LE对监督UE和非监督UE都提供了最先进的对抗性能，这是针对监督学习和非监督学习的UE的第一个可推广的对策。



## **7. Ortho-ODE: Enhancing Robustness and of Neural ODEs against Adversarial Attacks**

正交法：增强神经网络对敌方攻击的稳健性和稳健性 cs.LG

Final project paper

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2305.09179v1) [paper-pdf](http://arxiv.org/pdf/2305.09179v1)

**Authors**: Vishal Purohit

**Abstract**: Neural Ordinary Differential Equations (NODEs) probed the usage of numerical solvers to solve the differential equation characterized by a Neural Network (NN), therefore initiating a new paradigm of deep learning models with infinite depth. NODEs were designed to tackle the irregular time series problem. However, NODEs have demonstrated robustness against various noises and adversarial attacks. This paper is about the natural robustness of NODEs and examines the cause behind such surprising behaviour. We show that by controlling the Lipschitz constant of the ODE dynamics the robustness can be significantly improved. We derive our approach from Grownwall's inequality. Further, we draw parallels between contractivity theory and Grownwall's inequality. Experimentally we corroborate the enhanced robustness on numerous datasets - MNIST, CIFAR-10, and CIFAR 100. We also present the impact of adaptive and non-adaptive solvers on the robustness of NODEs.

摘要: 神经常微分方程组(节点)探索了用数值求解器来求解以神经网络(NN)为特征的微分方程，从而开创了一种无限深度深度学习模型的新范式。节点的设计是为了处理不规则的时间序列问题。然而，节点对各种噪声和敌意攻击表现出了健壮性。这篇论文是关于节点的自然健壮性的，并研究了这种令人惊讶的行为背后的原因。结果表明，通过控制常微分方程组的Lipschitz常数，可以显着提高系统的鲁棒性。我们的方法源于Grownwall的不等式性。此外，我们还比较了可逆性理论和Grownwall不等式之间的异同。在实验上，我们在许多数据集--MNIST、CIFAR-10和CIFAR 100上证实了增强的稳健性。我们还给出了自适应和非自适应求解器对节点健壮性的影响。



## **8. Run-Off Election: Improved Provable Defense against Data Poisoning Attacks**

决选：改进了针对数据中毒攻击的可证明防御 cs.LG

Accepted to ICML 2023

**SubmitDate**: 2023-05-16    [abs](http://arxiv.org/abs/2302.02300v3) [paper-pdf](http://arxiv.org/pdf/2302.02300v3)

**Authors**: Keivan Rezaei, Kiarash Banihashem, Atoosa Chegini, Soheil Feizi

**Abstract**: In data poisoning attacks, an adversary tries to change a model's prediction by adding, modifying, or removing samples in the training data. Recently, ensemble-based approaches for obtaining provable defenses against data poisoning have been proposed where predictions are done by taking a majority vote across multiple base models. In this work, we show that merely considering the majority vote in ensemble defenses is wasteful as it does not effectively utilize available information in the logits layers of the base models. Instead, we propose Run-Off Election (ROE), a novel aggregation method based on a two-round election across the base models: In the first round, models vote for their preferred class and then a second, Run-Off election is held between the top two classes in the first round. Based on this approach, we propose DPA+ROE and FA+ROE defense methods based on Deep Partition Aggregation (DPA) and Finite Aggregation (FA) approaches from prior work. We evaluate our methods on MNIST, CIFAR-10, and GTSRB and obtain improvements in certified accuracy by up to 3%-4%. Also, by applying ROE on a boosted version of DPA, we gain improvements around 12%-27% comparing to the current state-of-the-art, establishing a new state-of-the-art in (pointwise) certified robustness against data poisoning. In many cases, our approach outperforms the state-of-the-art, even when using 32 times less computational power.

摘要: 在数据中毒攻击中，对手试图通过添加、修改或删除训练数据中的样本来更改模型的预测。最近，已经提出了基于集成的方法来获得针对数据中毒的可证明防御，其中预测是通过在多个基础模型上获得多数票来完成的。在这项工作中，我们表明，仅仅在集成防御中考虑多数投票是浪费的，因为它没有有效地利用基本模型的Logits层中的可用信息。相反，我们提出了决选选举(ROE)，这是一种基于基础模型之间的两轮选举的新型聚合方法：在第一轮中，模型投票选择他们喜欢的类，然后在第一轮中前两个类之间举行第二次决选。在此基础上，提出了基于深度划分聚集(DPA)和有限聚集(FA)的DPA+ROE和FA+ROE防御方法。我们在MNIST、CIFAR-10和GTSRB上对我们的方法进行了评估，并在认证的准确性方面获得了高达3%-4%的改进。此外，通过在增强版本的DPA上应用ROE，与当前最先进的版本相比，我们获得了约12%-27%的改进，从而建立了针对数据中毒的(按点)经认证的新的最先进的健壮性。在许多情况下，我们的方法优于最先进的方法，即使在使用32倍的计算能力时也是如此。



## **9. Training Neural Networks without Backpropagation: A Deeper Dive into the Likelihood Ratio Method**

无反向传播训练神经网络：对似然比方法的深入探讨 cs.LG

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08960v1) [paper-pdf](http://arxiv.org/pdf/2305.08960v1)

**Authors**: Jinyang Jiang, Zeliang Zhang, Chenliang Xu, Zhaofei Yu, Yijie Peng

**Abstract**: Backpropagation (BP) is the most important gradient estimation method for training neural networks in deep learning. However, the literature shows that neural networks trained by BP are vulnerable to adversarial attacks. We develop the likelihood ratio (LR) method, a new gradient estimation method, for training a broad range of neural network architectures, including convolutional neural networks, recurrent neural networks, graph neural networks, and spiking neural networks, without recursive gradient computation. We propose three methods to efficiently reduce the variance of the gradient estimation in the neural network training process. Our experiments yield numerical results for training different neural networks on several datasets. All results demonstrate that the LR method is effective for training various neural networks and significantly improves the robustness of the neural networks under adversarial attacks relative to the BP method.

摘要: 反向传播(BP)是深度学习中训练神经网络最重要的梯度估计方法。然而，文献表明，BP训练的神经网络很容易受到对手的攻击。我们发展了一种新的梯度估计方法-似然比方法，用于训练包括卷积神经网络、递归神经网络、图神经网络和尖峰神经网络在内的广泛的神经网络结构，而不需要递归梯度计算。在神经网络训练过程中，我们提出了三种有效降低梯度估计方差的方法。我们的实验给出了在几个数据集上训练不同神经网络的数值结果。结果表明，与BP方法相比，LR方法对训练各种神经网络是有效的，并且显著提高了神经网络在对抗攻击下的鲁棒性。



## **10. Attacking Perceptual Similarity Metrics**

攻击感知相似性度量 cs.CV

TMLR 2023 (Featured Certification). Code is available at  https://tinyurl.com/attackingpsm

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2305.08840v1) [paper-pdf](http://arxiv.org/pdf/2305.08840v1)

**Authors**: Abhijay Ghildyal, Feng Liu

**Abstract**: Perceptual similarity metrics have progressively become more correlated with human judgments on perceptual similarity; however, despite recent advances, the addition of an imperceptible distortion can still compromise these metrics. In our study, we systematically examine the robustness of these metrics to imperceptible adversarial perturbations. Following the two-alternative forced-choice experimental design with two distorted images and one reference image, we perturb the distorted image closer to the reference via an adversarial attack until the metric flips its judgment. We first show that all metrics in our study are susceptible to perturbations generated via common adversarial attacks such as FGSM, PGD, and the One-pixel attack. Next, we attack the widely adopted LPIPS metric using spatial-transformation-based adversarial perturbations (stAdv) in a white-box setting to craft adversarial examples that can effectively transfer to other similarity metrics in a black-box setting. We also combine the spatial attack stAdv with PGD ($\ell_\infty$-bounded) attack to increase transferability and use these adversarial examples to benchmark the robustness of both traditional and recently developed metrics. Our benchmark provides a good starting point for discussion and further research on the robustness of metrics to imperceptible adversarial perturbations.

摘要: 知觉相似性指标已逐渐与人类对知觉相似性的判断更加相关；然而，尽管最近取得了进展，添加了不可察觉的失真仍然可能损害这些指标。在我们的研究中，我们系统地检查了这些度量对不可察觉的对抗性扰动的稳健性。在两个失真图像和一个参考图像的两种选择强迫选择实验设计之后，我们通过对抗性攻击使失真图像更接近参考图像，直到度量颠倒其判断。我们首先表明，我们研究中的所有指标都容易受到常见的对抗性攻击(如FGSM、PGD和单像素攻击)产生的扰动的影响。接下来，我们在白盒环境中使用基于空间变换的对抗性扰动(StAdv)来攻击广泛采用的LPIPS度量，以创建可以有效地转换到黑盒环境中的其他相似性度量的对抗性示例。我们还将空间攻击stAdv与pgd($\ell_\inty$-bound)攻击相结合，以增加可转移性，并使用这些对抗性示例来对传统度量和最近开发的度量的健壮性进行基准测试。我们的基准为讨论和进一步研究度量对不可察觉的对抗性扰动的健壮性提供了一个很好的起点。



## **11. Defending Against Misinformation Attacks in Open-Domain Question Answering**

开放领域答疑中防误报攻击的研究 cs.CL

**SubmitDate**: 2023-05-15    [abs](http://arxiv.org/abs/2212.10002v2) [paper-pdf](http://arxiv.org/pdf/2212.10002v2)

**Authors**: Orion Weller, Aleem Khan, Nathaniel Weir, Dawn Lawrie, Benjamin Van Durme

**Abstract**: Recent work in open-domain question answering (ODQA) has shown that adversarial poisoning of the search collection can cause large drops in accuracy for production systems. However, little to no work has proposed methods to defend against these attacks. To do so, we rely on the intuition that redundant information often exists in large corpora. To find it, we introduce a method that uses query augmentation to search for a diverse set of passages that could answer the original question but are less likely to have been poisoned. We integrate these new passages into the model through the design of a novel confidence method, comparing the predicted answer to its appearance in the retrieved contexts (what we call \textit{Confidence from Answer Redundancy}, i.e. CAR). Together these methods allow for a simple but effective way to defend against poisoning attacks that provides gains of nearly 20\% exact match across varying levels of data poisoning/knowledge conflicts.

摘要: 最近在开放领域问答(ODQA)方面的研究表明，搜索集合的敌意中毒会导致产生式系统的准确率大幅下降。然而，几乎没有工作提出了防御这些攻击的方法。要做到这一点，我们依赖于这样一种直觉，即大型语料库中往往存在冗余信息。为了找到它，我们引入了一种方法，使用查询增强来搜索一组不同的段落，这些段落可以回答原始问题，但不太可能被毒化。我们通过设计一种新的置信度方法将这些新的段落集成到模型中，将预测的答案与其在检索到的上下文中的表现进行比较(我们称其为来自答案冗余的置信度)，即CAR。这些方法结合在一起，提供了一种简单但有效的方法来防御中毒攻击，在不同级别的数据中毒/知识冲突中提供了近20%的精确匹配。



## **12. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

不可察觉和可转移对抗性攻击的扩散模型 cs.CV

Code Page: https://github.com/WindVChen/DiffAttack

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08192v1) [paper-pdf](http://arxiv.org/pdf/2305.08192v1)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an additional recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into adversarial attack field. Extensive experiments on various model structures (including CNNs, Transformers, MLPs) and defense methods have demonstrated our superiority over other attack methods.

摘要: 许多现有的对抗性攻击在图像RGB空间上产生$L_p$-范数扰动。尽管在可转移性和攻击成功率方面取得了一些成就，但制作的对抗性例子很容易被人眼察觉。对于视觉不可感知性，最近的一些工作探索了没有$L_p$-范数约束的无限攻击，但缺乏攻击黑盒模型的可转移性。在这项工作中，我们提出了一种新的不可察觉和可转移的攻击，利用扩散模型的生成性和区分性。具体地说，我们不是在像素空间中直接操作，而是在扩散模型的潜在空间中制造扰动。与设计良好的内容保持结构相结合，我们可以生成嵌入语义线索的人类不敏感的扰动。为了获得更好的可转移性，我们通过将扩散模型的注意力从目标区域转移开，进一步欺骗了扩散模型，该模型可以被视为一个额外的识别代理。据我们所知，我们提出的DiffAttack方法是第一个将扩散模型引入对抗性攻击领域的方法。在各种模型结构(包括CNN、Transformers、MLP)和防御方法上的广泛实验证明了该攻击方法相对于其他攻击方法的优越性。



## **13. Watermarking Text Generated by Black-Box Language Models**

黑盒语言模型生成的文本水印 cs.CL

Code will be available at  https://github.com/Kiode/Text_Watermark_Language_Models

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08883v1) [paper-pdf](http://arxiv.org/pdf/2305.08883v1)

**Authors**: Xi Yang, Kejiang Chen, Weiming Zhang, Chang Liu, Yuang Qi, Jie Zhang, Han Fang, Nenghai Yu

**Abstract**: LLMs now exhibit human-like skills in various fields, leading to worries about misuse. Thus, detecting generated text is crucial. However, passive detection methods are stuck in domain specificity and limited adversarial robustness. To achieve reliable detection, a watermark-based method was proposed for white-box LLMs, allowing them to embed watermarks during text generation. The method involves randomly dividing the model vocabulary to obtain a special list and adjusting the probability distribution to promote the selection of words in the list. A detection algorithm aware of the list can identify the watermarked text. However, this method is not applicable in many real-world scenarios where only black-box language models are available. For instance, third-parties that develop API-based vertical applications cannot watermark text themselves because API providers only supply generated text and withhold probability distributions to shield their commercial interests. To allow third-parties to autonomously inject watermarks into generated text, we develop a watermarking framework for black-box language model usage scenarios. Specifically, we first define a binary encoding function to compute a random binary encoding corresponding to a word. The encodings computed for non-watermarked text conform to a Bernoulli distribution, wherein the probability of a word representing bit-1 being approximately 0.5. To inject a watermark, we alter the distribution by selectively replacing words representing bit-0 with context-based synonyms that represent bit-1. A statistical test is then used to identify the watermark. Experiments demonstrate the effectiveness of our method on both Chinese and English datasets. Furthermore, results under re-translation, polishing, word deletion, and synonym substitution attacks reveal that it is arduous to remove the watermark without compromising the original semantics.

摘要: LLM现在各个领域都展示了类似人类的技能，这导致了人们对滥用的担忧。因此，检测生成的文本至关重要。然而，被动检测方法停留在领域特异性和有限的对抗稳健性。为了实现可靠的检测，提出了一种基于水印的白盒LLMS检测方法，允许白盒LLMS在文本生成过程中嵌入水印。该方法对模型词汇进行随机划分，得到一个特殊的词汇表，并通过调整概率分布来促进词汇表中的词的选择。知道该列表的检测算法可以识别加水印的文本。然而，这种方法不适用于许多实际场景，因为只有黑盒语言模型可用。例如，开发基于API的垂直应用程序的第三方不能自己为文本添加水印，因为API提供商只提供生成的文本，并扣留概率分布以保护他们的商业利益。为了允许第三方自主地向生成的文本中注入水印，我们开发了一个适用于黑盒语言模型使用场景的水印框架。具体地说，我们首先定义一个二进制编码函数来计算对应于一个单词的随机二进制编码。为未加水印的文本计算的编码符合伯努利分布，其中表示位1的字的概率约为0.5。为了注入水印，我们通过有选择地将表示位0的单词替换为表示位1的基于上下文的同义词来改变分布。然后使用统计测试来识别水印。在中文和英文数据集上的实验证明了该方法的有效性。此外，在重译、润色、单词删除和同义词替换攻击下的结果表明，在不损害原始语义的情况下去除水印是困难的。



## **14. Improving Defensive Distillation using Teacher Assistant**

利用助教提高防守蒸馏能力 cs.CV

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08076v1) [paper-pdf](http://arxiv.org/pdf/2305.08076v1)

**Authors**: Maniratnam Mandal, Suna Gao

**Abstract**: Adversarial attacks pose a significant threat to the security and safety of deep neural networks being applied to modern applications. More specifically, in computer vision-based tasks, experts can use the knowledge of model architecture to create adversarial samples imperceptible to the human eye. These attacks can lead to security problems in popular applications such as self-driving cars, face recognition, etc. Hence, building networks which are robust to such attacks is highly desirable and essential. Among the various methods present in literature, defensive distillation has shown promise in recent years. Using knowledge distillation, researchers have been able to create models robust against some of those attacks. However, more attacks have been developed exposing weakness in defensive distillation. In this project, we derive inspiration from teacher assistant knowledge distillation and propose that introducing an assistant network can improve the robustness of the distilled model. Through a series of experiments, we evaluate the distilled models for different distillation temperatures in terms of accuracy, sensitivity, and robustness. Our experiments demonstrate that the proposed hypothesis can improve robustness in most cases. Additionally, we show that multi-step distillation can further improve robustness with very little impact on model accuracy.

摘要: 对抗性攻击对应用于现代应用的深度神经网络的安全性和安全性构成了严重威胁。更具体地说，在基于计算机视觉的任务中，专家可以利用模型体系结构的知识来创建人眼看不到的对抗性样本。这些攻击可能会导致自动驾驶汽车、人脸识别等热门应用中的安全问题。因此，构建对此类攻击具有健壮性的网络是非常必要的。在文献中出现的各种方法中，防御性蒸馏在最近几年显示出了希望。使用知识蒸馏，研究人员已经能够创建针对其中一些攻击的稳健模型。然而，更多的攻击暴露了防守蒸馏的弱点。在本项目中，我们从教师辅助知识提取中得到启发，并提出引入辅助网络可以提高提取模型的健壮性。通过一系列的实验，我们评估了不同蒸馏温度下的蒸馏模型的准确性、灵敏度和稳健性。我们的实验表明，该假设在大多数情况下都能提高稳健性。此外，我们还表明，多步精馏可以在对模型精度影响很小的情况下进一步提高稳健性。



## **15. DNN-Defender: An in-DRAM Deep Neural Network Defense Mechanism for Adversarial Weight Attack**

DNN-Defender：一种DRAM深度神经网络对抗性权重攻击防御机制 cs.CR

10 pages, 11 figures

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08034v1) [paper-pdf](http://arxiv.org/pdf/2305.08034v1)

**Authors**: Ranyang Zhou, Sabbir Ahmed, Adnan Siraj Rakin, Shaahin Angizi

**Abstract**: With deep learning deployed in many security-sensitive areas, machine learning security is becoming progressively important. Recent studies demonstrate attackers can exploit system-level techniques exploiting the RowHammer vulnerability of DRAM to deterministically and precisely flip bits in Deep Neural Networks (DNN) model weights to affect inference accuracy. The existing defense mechanisms are software-based, such as weight reconstruction requiring expensive training overhead or performance degradation. On the other hand, generic hardware-based victim-/aggressor-focused mechanisms impose expensive hardware overheads and preserve the spatial connection between victim and aggressor rows. In this paper, we present the first DRAM-based victim-focused defense mechanism tailored for quantized DNNs, named DNN-Defender that leverages the potential of in-DRAM swapping to withstand the targeted bit-flip attacks. Our results indicate that DNN-Defender can deliver a high level of protection downgrading the performance of targeted RowHammer attacks to a random attack level. In addition, the proposed defense has no accuracy drop on CIFAR-10 and ImageNet datasets without requiring any software training or incurring additional hardware overhead.

摘要: 随着深度学习在许多安全敏感领域的部署，机器学习的安全性正变得越来越重要。最近的研究表明，攻击者可以利用系统级技术，利用DRAM的RowHammer漏洞来确定并精确地翻转深度神经网络(DNN)模型中的位，以影响推理精度。现有的防御机制是基于软件的，例如需要昂贵的训练开销的权重重建或性能下降。另一方面，通用的基于硬件的以受害者/攻击者为中心的机制增加了昂贵的硬件开销，并保持了受害者和攻击者行之间的空间连接。在本文中，我们提出了第一个基于DRAM的针对量化DNN的以受害者为中心的防御机制，称为DNN-Defender，它利用DRAM内交换的潜力来抵御目标位翻转攻击。我们的结果表明，DNN-Defender可以提供高级别的保护，将目标RowHammer攻击的性能降低到随机攻击级别。此外，拟议的防御在CIFAR-10和ImageNet数据集上没有精度下降，不需要任何软件培训或产生额外的硬件开销。



## **16. On enhancing the robustness of Vision Transformers: Defensive Diffusion**

增强视觉变形金刚的稳健性：防御性扩散 cs.CV

Our code is publicly available at  https://github.com/Muhammad-Huzaifaa/Defensive_Diffusion

**SubmitDate**: 2023-05-14    [abs](http://arxiv.org/abs/2305.08031v1) [paper-pdf](http://arxiv.org/pdf/2305.08031v1)

**Authors**: Raza Imam, Muhammad Huzaifa, Mohammed El-Amine Azz

**Abstract**: Privacy and confidentiality of medical data are of utmost importance in healthcare settings. ViTs, the SOTA vision model, rely on large amounts of patient data for training, which raises concerns about data security and the potential for unauthorized access. Adversaries may exploit vulnerabilities in ViTs to extract sensitive patient information and compromising patient privacy. This work address these vulnerabilities to ensure the trustworthiness and reliability of ViTs in medical applications. In this work, we introduced a defensive diffusion technique as an adversarial purifier to eliminate adversarial noise introduced by attackers in the original image. By utilizing the denoising capabilities of the diffusion model, we employ a reverse diffusion process to effectively eliminate the adversarial noise from the attack sample, resulting in a cleaner image that is then fed into the ViT blocks. Our findings demonstrate the effectiveness of the diffusion model in eliminating attack-agnostic adversarial noise from images. Additionally, we propose combining knowledge distillation with our framework to obtain a lightweight student model that is both computationally efficient and robust against gray box attacks. Comparison of our method with a SOTA baseline method, SEViT, shows that our work is able to outperform the baseline. Extensive experiments conducted on a publicly available Tuberculosis X-ray dataset validate the computational efficiency and improved robustness achieved by our proposed architecture.

摘要: 医疗数据的隐私和机密性在医疗保健环境中至关重要。VITS是SOTA的视觉模型，它依赖于大量的患者数据进行培训，这引发了人们对数据安全和未经授权访问的可能性的担忧。攻击者可能会利用VITS中的漏洞来提取敏感的患者信息，从而危及患者隐私。这项工作解决了这些漏洞，以确保VITS在医疗应用中的可信性和可靠性。在这项工作中，我们引入了一种防御扩散技术作为对抗性净化器来消除攻击者在原始图像中引入的对抗性噪声。通过利用扩散模型的去噪能力，我们采用反向扩散过程来有效地消除攻击样本中的对抗性噪声，从而得到更干净的图像，然后将其送入VIT块。我们的发现证明了扩散模型在消除图像中与攻击无关的对抗性噪声方面的有效性。此外，我们建议将知识提炼与我们的框架相结合，以获得一个轻量级的学生模型，该模型在计算效率上是有效的，并且对灰盒攻击具有健壮性。我们的方法与SOTA基线方法SEViT的比较表明，我们的工作能够超过基线。在公开可用的结核病X光数据集上进行的大量实验验证了我们所提出的体系结构的计算效率和提高的稳健性。



## **17. Detection and Mitigation of Byzantine Attacks in Distributed Training**

分布式训练中拜占庭攻击的检测与缓解 cs.LG

21 pages, 17 figures, 6 tables. The material in this work appeared in  part at arXiv:2108.02416 which has been published at the 2022 IEEE  International Symposium on Information Theory

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2208.08085v4) [paper-pdf](http://arxiv.org/pdf/2208.08085v4)

**Authors**: Konstantinos Konstantinidis, Namrata Vaswani, Aditya Ramamoorthy

**Abstract**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients.   In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. We also show the convergence of our method to the optimal point under common assumptions and settings considered in literature. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.

摘要: 过多的现代机器学习任务需要利用大规模分布式集群作为培训管道的关键组成部分。然而，工作者节点的异常拜占庭行为会破坏训练，影响推理的质量。此类行为可归因于无意的系统故障或精心策划的攻击；因此，某些节点可能会向协调训练的参数服务器(PS)返回任意结果。最近的工作考虑了广泛的攻击模型，并探索了稳健的聚集和/或计算冗余来纠正扭曲的梯度。在这项工作中，我们考虑了从强到强的攻击模型：$q$全知的对手，完全了解防御协议，可以从一个迭代到另一个迭代变化；$q$随机选择的对手，合谋能力有限，一次只有几个迭代改变。我们的算法依赖于冗余的任务分配以及对敌对行为的检测。我们还证明了在文献中常见的假设和设置下，我们的方法收敛到最优点。对于强攻击，我们展示了与以前最先进的技术相比，扭曲梯度的比例降低了16%-99%。我们在CIFAR-10数据集上的TOP-1分类精度结果显示，在最复杂的攻击下，与最先进的方法相比，准确率(在强和弱场景下平均)提高了25%。



## **18. Decision-based iterative fragile watermarking for model integrity verification**

用于模型完整性验证的基于决策的迭代脆弱水印 cs.CR

**SubmitDate**: 2023-05-13    [abs](http://arxiv.org/abs/2305.09684v1) [paper-pdf](http://arxiv.org/pdf/2305.09684v1)

**Authors**: Zhaoxia Yin, Heng Yin, Hang Su, Xinpeng Zhang, Zhenzhe Gao

**Abstract**: Typically, foundation models are hosted on cloud servers to meet the high demand for their services. However, this exposes them to security risks, as attackers can modify them after uploading to the cloud or transferring from a local system. To address this issue, we propose an iterative decision-based fragile watermarking algorithm that transforms normal training samples into fragile samples that are sensitive to model changes. We then compare the output of sensitive samples from the original model to that of the compromised model during validation to assess the model's completeness.The proposed fragile watermarking algorithm is an optimization problem that aims to minimize the variance of the predicted probability distribution outputed by the target model when fed with the converted sample.We convert normal samples to fragile samples through multiple iterations. Our method has some advantages: (1) the iterative update of samples is done in a decision-based black-box manner, relying solely on the predicted probability distribution of the target model, which reduces the risk of exposure to adversarial attacks, (2) the small-amplitude multiple iterations approach allows the fragile samples to perform well visually, with a PSNR of 55 dB in TinyImageNet compared to the original samples, (3) even with changes in the overall parameters of the model of magnitude 1e-4, the fragile samples can detect such changes, and (4) the method is independent of the specific model structure and dataset. We demonstrate the effectiveness of our method on multiple models and datasets, and show that it outperforms the current state-of-the-art.

摘要: 通常，基础模型托管在云服务器上，以满足对其服务的高需求。然而，这会使它们面临安全风险，因为攻击者可以在上传到云或从本地系统传输后修改它们。为了解决这个问题，我们提出了一种基于迭代判决的脆弱水印算法，将正常的训练样本转换为对模型变化敏感的脆弱样本。然后在验证过程中比较原始模型和受损模型的敏感样本的输出，以评估模型的完备性。脆弱水印算法是一个优化问题，旨在最小化目标模型输入转换后的样本时预测概率分布的方差，通过多次迭代将正常样本转换为脆弱样本。我们的方法有一些优点：(1)样本的迭代更新是以基于决策的黑盒方式进行的，仅依赖于目标模型的预测概率分布，从而降低了暴露于对抗性攻击的风险；(2)小幅度多次迭代方法允许脆弱样本在视觉上表现良好，与原始样本相比，TinyImageNet的PSNR为55dB；(3)即使1e-4量级的模型的总体参数发生变化，脆弱样本也可以检测到这种变化；(4)该方法独立于特定的模型结构和数据集。我们在多个模型和数据集上验证了我们的方法的有效性，并表明它的性能优于目前最先进的方法。



## **19. Quantum Lock: A Provable Quantum Communication Advantage**

量子锁：一种可证明的量子通信优势 quant-ph

47 pages, 13 figures

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2110.09469v4) [paper-pdf](http://arxiv.org/pdf/2110.09469v4)

**Authors**: Kaushik Chakraborty, Mina Doosti, Yao Ma, Chirag Wadhwa, Myrto Arapinis, Elham Kashefi

**Abstract**: Physical unclonable functions(PUFs) provide a unique fingerprint to a physical entity by exploiting the inherent physical randomness. Gao et al. discussed the vulnerability of most current-day PUFs to sophisticated machine learning-based attacks. We address this problem by integrating classical PUFs and existing quantum communication technology. Specifically, this paper proposes a generic design of provably secure PUFs, called hybrid locked PUFs(HLPUFs), providing a practical solution for securing classical PUFs. An HLPUF uses a classical PUF(CPUF), and encodes the output into non-orthogonal quantum states to hide the outcomes of the underlying CPUF from any adversary. Here we introduce a quantum lock to protect the HLPUFs from any general adversaries. The indistinguishability property of the non-orthogonal quantum states, together with the quantum lockdown technique prevents the adversary from accessing the outcome of the CPUFs. Moreover, we show that by exploiting non-classical properties of quantum states, the HLPUF allows the server to reuse the challenge-response pairs for further client authentication. This result provides an efficient solution for running PUF-based client authentication for an extended period while maintaining a small-sized challenge-response pairs database on the server side. Later, we support our theoretical contributions by instantiating the HLPUFs design using accessible real-world CPUFs. We use the optimal classical machine-learning attacks to forge both the CPUFs and HLPUFs, and we certify the security gap in our numerical simulation for construction which is ready for implementation.

摘要: 物理不可克隆函数(PUF)通过利用固有的物理随机性为物理实体提供唯一指纹。高等人。讨论了当前大多数PUF对复杂的基于机器学习的攻击的脆弱性。我们通过将经典的PUF和现有的量子通信技术相结合来解决这个问题。具体地说，本文提出了一种可证明安全的PUF的通用设计，称为混合锁定PUF(HLPUF)，为保护经典PUF提供了一种实用的解决方案。HLPUF使用经典的PUF(CPUF)，并将输出编码为非正交的量子态，以向任何对手隐藏底层CPUF的结果。在这里，我们引入量子锁来保护HLPUF免受任何一般对手的攻击。非正交量子态的不可分辨特性，加上量子锁定技术，阻止了攻击者访问CPUF的结果。此外，我们证明了通过利用量子态的非经典属性，HLPUF允许服务器重用挑战-响应对来进行进一步的客户端认证。这一结果为长期运行基于PUF的客户端身份验证提供了一个有效的解决方案，同时在服务器端维护一个小型的挑战-响应对数据库。随后，我们通过使用可访问的真实CPUF来实例化HLPUF设计来支持我们的理论贡献。我们使用最优经典机器学习攻击来伪造CPUF和HLPUF，并证明了我们的构造数值模拟中的安全漏洞。



## **20. Two-in-One: A Model Hijacking Attack Against Text Generation Models**

二合一：一种针对文本生成模型的模型劫持攻击 cs.CR

To appear in the 32nd USENIX Security Symposium, August 2023,  Anaheim, CA, USA

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07406v1) [paper-pdf](http://arxiv.org/pdf/2305.07406v1)

**Authors**: Wai Man Si, Michael Backes, Yang Zhang, Ahmed Salem

**Abstract**: Machine learning has progressed significantly in various applications ranging from face recognition to text generation. However, its success has been accompanied by different attacks. Recently a new attack has been proposed which raises both accountability and parasitic computing risks, namely the model hijacking attack. Nevertheless, this attack has only focused on image classification tasks. In this work, we broaden the scope of this attack to include text generation and classification models, hence showing its broader applicability. More concretely, we propose a new model hijacking attack, Ditto, that can hijack different text classification tasks into multiple generation ones, e.g., language translation, text summarization, and language modeling. We use a range of text benchmark datasets such as SST-2, TweetEval, AGnews, QNLI, and IMDB to evaluate the performance of our attacks. Our results show that by using Ditto, an adversary can successfully hijack text generation models without jeopardizing their utility.

摘要: 机器学习在从人脸识别到文本生成的各种应用中都取得了显著的进展。然而，它的成功伴随着不同的攻击。最近提出了一种新的攻击，它同时增加了可追究性和寄生计算的风险，即模型劫持攻击。尽管如此，这次攻击只集中在图像分类任务上。在这项工作中，我们扩大了该攻击的范围，将文本生成和分类模型包括在内，从而显示了其更广泛的适用性。更具体地说，我们提出了一种新的劫持攻击模型Ditto，该模型可以将不同的文本分类任务劫持为多个世代任务，例如语言翻译、文本摘要和语言建模。我们使用一系列文本基准数据集，如SST-2、TweetEval、AgNews、QNLI和IMDB来评估我们的攻击性能。我们的结果表明，通过使用Ditto，攻击者可以在不损害其实用性的情况下成功劫持文本生成模型。



## **21. Novel bribery mining attacks in the bitcoin system and the bribery miner's dilemma**

比特币系统中的新型贿赂挖掘攻击与贿赂挖掘者的困境 cs.GT

26 pages, 16 figures, 3 tables

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07381v1) [paper-pdf](http://arxiv.org/pdf/2305.07381v1)

**Authors**: Junjie Hu, Chunxiang Xu, Zhe Jiang, Jiwu Cao

**Abstract**: Mining attacks allow adversaries to obtain a disproportionate share of the mining reward by deviating from the honest mining strategy in the Bitcoin system. Among them, the most well-known are selfish mining (SM), block withholding (BWH), fork after withholding (FAW) and bribery mining. In this paper, we propose two novel mining attacks: bribery semi-selfish mining (BSSM) and bribery stubborn mining (BSM). Both of them can increase the relative extra reward of the adversary and will make the target bribery miners suffer from the bribery miner dilemma. All targets earn less under the Nash equilibrium. For each target, their local optimal strategy is to accept the bribes. However, they will suffer losses, comparing with denying the bribes. Furthermore, for all targets, their global optimal strategy is to deny the bribes. Quantitative analysis and simulation have been verified our theoretical analysis. We propose practical measures to mitigate more advanced mining attack strategies based on bribery mining, and provide new ideas for addressing bribery mining attacks in the future. However, how to completely and effectively prevent these attacks is still needed on further research.

摘要: 挖矿攻击允许对手通过偏离比特币系统中诚实的挖矿策略，获得不成比例的挖矿回报。其中，最广为人知的是自私挖矿(SM)、集体扣留(BWH)、扣后分叉(FAW)和贿赂挖矿。本文提出了两种新的挖掘攻击：贿赂半自私挖掘(BSSM)和贿赂顽固挖掘(BSM)。两者都能增加对手的相对额外报酬，使受贿目标矿工陷入受贿矿工困境。在纳什均衡下，所有目标的收入都较低。对于每个目标，他们在当地的最优策略是收受贿赂。然而，与拒绝贿赂相比，他们将蒙受损失。此外，对于所有目标来说，他们的全球最佳策略是否认贿赂。定量分析和仿真验证了我们的理论分析。提出了缓解基于贿赂挖掘的更高级挖掘攻击策略的实用措施，为未来应对贿赂挖掘攻击提供了新的思路。然而，如何完全有效地防范这些攻击还需要进一步的研究。



## **22. Efficient Search of Comprehensively Robust Neural Architectures via Multi-fidelity Evaluation**

基于多保真度评价的综合稳健神经网络高效搜索 cs.CV

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07308v1) [paper-pdf](http://arxiv.org/pdf/2305.07308v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstract**: Neural architecture search (NAS) has emerged as one successful technique to find robust deep neural network (DNN) architectures. However, most existing robustness evaluations in NAS only consider $l_{\infty}$ norm-based adversarial noises. In order to improve the robustness of DNN models against multiple types of noises, it is necessary to consider a comprehensive evaluation in NAS for robust architectures. But with the increasing number of types of robustness evaluations, it also becomes more time-consuming to find comprehensively robust architectures. To alleviate this problem, we propose a novel efficient search of comprehensively robust neural architectures via multi-fidelity evaluation (ES-CRNA-ME). Specifically, we first search for comprehensively robust architectures under multiple types of evaluations using the weight-sharing-based NAS method, including different $l_{p}$ norm attacks, semantic adversarial attacks, and composite adversarial attacks. In addition, we reduce the number of robustness evaluations by the correlation analysis, which can incorporate similar evaluations and decrease the evaluation cost. Finally, we propose a multi-fidelity online surrogate during optimization to further decrease the search cost. On the basis of the surrogate constructed by low-fidelity data, the online high-fidelity data is utilized to finetune the surrogate. Experiments on CIFAR10 and CIFAR100 datasets show the effectiveness of our proposed method.

摘要: 神经体系结构搜索(NAS)已成为一种发现健壮的深度神经网络(DNN)体系结构的成功技术。然而，现有的NAS健壮性评估大多只考虑基于$L范数的对抗性噪声。为了提高DNN模型对多种类型噪声的鲁棒性，有必要考虑在NAS中对健壮体系结构进行综合评估。但随着健壮性评估类型的增加，寻找全面健壮的体系结构也变得更加耗时。为了缓解这一问题，我们提出了一种新的高效的通过多保真度评估(ES-CRNA-ME)来寻找全面稳健的神经结构的方法。具体地说，我们首先使用基于权重共享的NAS方法在多种评估类型下搜索全面的健壮性体系结构，包括不同的$L_{p}$范数攻击、语义对抗攻击和复合对抗攻击。另外，通过相关性分析，减少了健壮性评价的次数，可以融合相似评价，降低评价成本。最后，在优化过程中提出了一种多保真的在线代理，进一步降低了搜索成本。在低保真数据构建代理的基础上，利用在线高保真数据对代理进行微调。在CIFAR10和CIFAR100数据集上的实验表明了该方法的有效性。



## **23. Parameter identifiability of a deep feedforward ReLU neural network**

深度前馈RELU神经网络的参数可辨识性 math.ST

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2112.12982v2) [paper-pdf](http://arxiv.org/pdf/2112.12982v2)

**Authors**: Joachim Bona-Pellissier, François Bachoc, François Malgouyres

**Abstract**: The possibility for one to recover the parameters-weights and biases-of a neural network thanks to the knowledge of its function on a subset of the input space can be, depending on the situation, a curse or a blessing. On one hand, recovering the parameters allows for better adversarial attacks and could also disclose sensitive information from the dataset used to construct the network. On the other hand, if the parameters of a network can be recovered, it guarantees the user that the features in the latent spaces can be interpreted. It also provides foundations to obtain formal guarantees on the performances of the network. It is therefore important to characterize the networks whose parameters can be identified and those whose parameters cannot. In this article, we provide a set of conditions on a deep fully-connected feedforward ReLU neural network under which the parameters of the network are uniquely identified-modulo permutation and positive rescaling-from the function it implements on a subset of the input space.

摘要: 由于知道神经网络在输入空间的子集上的功能，人们能够恢复神经网络的参数--权重和偏差--的可能性可能是诅咒，也可能是祝福，具体取决于具体情况。一方面，恢复参数可以进行更好的对抗性攻击，还可能泄露用于构建网络的数据集的敏感信息。另一方面，如果可以恢复网络的参数，就可以保证用户可以解释潜在空间中的特征。它还为获得对网络性能的正式保证提供了基础。因此，重要的是要确定其参数可以识别和参数不能识别的网络的特征。本文给出了一个深度全连通的前馈RELU神经网络的一组条件，在该条件下，网络的参数可以从它在输入空间的一个子集上实现的函数中唯一地识别出来--模置换和正重标度。



## **24. Adversarial Security and Differential Privacy in mmWave Beam Prediction in 6G networks**

6G网络毫米波波束预测中的对抗安全和差分保密 cs.CR

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.09679v1) [paper-pdf](http://arxiv.org/pdf/2305.09679v1)

**Authors**: Ghanta Sai Krishna, Kundrapu Supriya, Sanskar Singh, Sabur Baidya

**Abstract**: In the forthcoming era of 6G, the mmWave communication is envisioned to be used in dense user scenarios with high bandwidth requirements, that necessitate efficient and accurate beam prediction. Machine learning (ML) based approaches are ushering as a critical solution for achieving such efficient beam prediction for 6G mmWave communications. However, most contemporary ML classifiers are quite susceptible to adversarial inputs. Attackers can easily perturb the methodology through noise addition in the model itself. To mitigate this, the current work presents a defensive mechanism for attenuating the adversarial attacks against projected ML-based models for mmWave beam anticipation by incorporating adversarial training. Furthermore, as training 6G mmWave beam prediction model necessitates the use of large and comprehensive datasets that could include sensitive information regarding the user's location, differential privacy (DP) has been introduced as a technique to preserve the confidentiality of the information by purposefully adding a low sensitivity controlled noise in the datasets. It ensures that even if the information about a user location could be retrieved, the attacker would have no means to determine whether the information is significant or meaningless. With ray-tracing simulations for various outdoor and indoor scenarios, we illustrate the advantage of our proposed novel framework in terms of beam prediction accuracy and effective achievable rate while ensuring the security and privacy in communications.

摘要: 在即将到来的6G时代，毫米波通信被设想用于高带宽要求的密集用户场景，这就需要高效和准确的波束预测。基于机器学习(ML)的方法正在成为实现6G毫米波通信如此高效的波束预测的关键解决方案。然而，大多数当代ML量词都很容易受到对抗性输入的影响。攻击者可以很容易地通过在模型本身中添加噪声来扰乱该方法。为了缓解这一问题，当前的工作提出了一种防御机制，通过结合对抗性训练来减弱针对基于ML的毫米波波束预测投影模型的对抗性攻击。此外，由于训练6G毫米波波束预测模型需要使用可能包括关于用户位置的敏感信息的大型和全面的数据集，因此引入了差分隐私(DP)作为一种技术，通过在数据集中有目的地添加低灵敏度受控噪声来保护信息的机密性。它确保即使可以检索到有关用户位置的信息，攻击者也无法确定该信息是重要的还是毫无意义的。通过对各种室外和室内场景的光线跟踪仿真，说明了该框架在保证通信安全性和保密性的同时，在波束预测精度和有效可达速率方面的优势。



## **25. Physical-layer Adversarial Robustness for Deep Learning-based Semantic Communications**

基于深度学习的语义通信物理层对抗健壮性 eess.SP

17 pages, 28 figures, accepted by IEEE jsac

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.07220v1) [paper-pdf](http://arxiv.org/pdf/2305.07220v1)

**Authors**: Guoshun Nan, Zhichun Li, Jinli Zhai, Qimei Cui, Gong Chen, Xin Du, Xuefei Zhang, Xiaofeng Tao, Zhu Han, Tony Q. S. Quek

**Abstract**: End-to-end semantic communications (ESC) rely on deep neural networks (DNN) to boost communication efficiency by only transmitting the semantics of data, showing great potential for high-demand mobile applications. We argue that central to the success of ESC is the robust interpretation of conveyed semantics at the receiver side, especially for security-critical applications such as automatic driving and smart healthcare. However, robustifying semantic interpretation is challenging as ESC is extremely vulnerable to physical-layer adversarial attacks due to the openness of wireless channels and the fragileness of neural models. Toward ESC robustness in practice, we ask the following two questions: Q1: For attacks, is it possible to generate semantic-oriented physical-layer adversarial attacks that are imperceptible, input-agnostic and controllable? Q2: Can we develop a defense strategy against such semantic distortions and previously proposed adversaries? To this end, we first present MobileSC, a novel semantic communication framework that considers the computation and memory efficiency in wireless environments. Equipped with this framework, we propose SemAdv, a physical-layer adversarial perturbation generator that aims to craft semantic adversaries over the air with the abovementioned criteria, thus answering the Q1. To better characterize the realworld effects for robust training and evaluation, we further introduce a novel adversarial training method SemMixed to harden the ESC against SemAdv attacks and existing strong threats, thus answering the Q2. Extensive experiments on three public benchmarks verify the effectiveness of our proposed methods against various physical adversarial attacks. We also show some interesting findings, e.g., our MobileSC can even be more robust than classical block-wise communication systems in the low SNR regime.

摘要: 端到端语义通信(ESC)依靠深度神经网络(DNN)来提高通信效率，只传输数据的语义，在高需求的移动应用中显示出巨大的潜力。我们认为，ESC成功的核心是在接收方对所传达的语义进行强有力的解释，特别是对于自动驾驶和智能医疗等安全关键型应用。然而，由于无线信道的开放性和神经模型的脆弱性，ESC极易受到物理层的敌意攻击，因此增强语义解释是具有挑战性的。对于ESC在实践中的健壮性，我们提出了以下两个问题：Q1：对于攻击，是否有可能产生不可察觉的、与输入无关的、可控的、面向语义的物理层对抗性攻击？问题2：我们能否针对这种语义扭曲和之前提出的对手制定防御策略？为此，我们首先提出了一种新的语义通信框架MobileSC，该框架考虑了无线环境下的计算和存储效率。在此框架的基础上，我们提出了一种物理层敌意扰动生成器SemAdv，旨在利用上述标准在空中构建语义对手，从而回答问题1。为了更好地表征稳健训练和评估的真实效果，我们进一步引入了一种新的对抗性训练方法SemMixed来强化ESC对SemAdv攻击和现有的强威胁的攻击，从而回答了Q2。在三个公共基准上的大量实验验证了我们提出的方法对各种物理攻击的有效性。我们还发现了一些有趣的发现，例如，在低信噪比条件下，我们的MobileSC甚至可以比经典的分组通信系统更健壮。



## **26. Stratified Adversarial Robustness with Rejection**

具有拒绝的分层对抗健壮性 cs.LG

Paper published at International Conference on Machine Learning  (ICML'23)

**SubmitDate**: 2023-05-12    [abs](http://arxiv.org/abs/2305.01139v2) [paper-pdf](http://arxiv.org/pdf/2305.01139v2)

**Authors**: Jiefeng Chen, Jayaram Raghuram, Jihye Choi, Xi Wu, Yingyu Liang, Somesh Jha

**Abstract**: Recently, there is an emerging interest in adversarially training a classifier with a rejection option (also known as a selective classifier) for boosting adversarial robustness. While rejection can incur a cost in many applications, existing studies typically associate zero cost with rejecting perturbed inputs, which can result in the rejection of numerous slightly-perturbed inputs that could be correctly classified. In this work, we study adversarially-robust classification with rejection in the stratified rejection setting, where the rejection cost is modeled by rejection loss functions monotonically non-increasing in the perturbation magnitude. We theoretically analyze the stratified rejection setting and propose a novel defense method -- Adversarial Training with Consistent Prediction-based Rejection (CPR) -- for building a robust selective classifier. Experiments on image datasets demonstrate that the proposed method significantly outperforms existing methods under strong adaptive attacks. For instance, on CIFAR-10, CPR reduces the total robust loss (for different rejection losses) by at least 7.3% under both seen and unseen attacks.

摘要: 最近，对抗性地训练具有拒绝选项的分类器(也称为选择性分类器)以增强对抗性健壮性是一种新的兴趣。虽然拒绝在许多应用中可能会导致成本，但现有研究通常将零成本与拒绝扰动输入联系在一起，这可能导致拒绝许多可以正确分类的轻微扰动输入。在这项工作中，我们研究了分层拒绝环境下的具有拒绝的对抗性鲁棒分类，其中拒绝代价由拒绝损失函数来建模，拒绝损失函数在扰动幅度上单调地不增加。我们从理论上分析了分层拒绝的设置，并提出了一种新的防御方法--基于一致预测拒绝的对抗训练(CPR)--来构建一个健壮的选择性分类器。在图像数据集上的实验表明，该方法在强自适应攻击下的性能明显优于已有方法。例如，在CIFAR-10上，CPR在看得见和看不见的攻击下都将总的稳健损失(针对不同的拒绝损失)减少了至少7.3%。



## **27. Improving Hyperspectral Adversarial Robustness Under Multiple Attacks**

提高多重攻击下的高光谱对抗健壮性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2210.16346v4) [paper-pdf](http://arxiv.org/pdf/2210.16346v4)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation models classifying hyperspectral images (HSI) are vulnerable to adversarial examples. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease in performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network.

摘要: 对高光谱图像进行分类的语义分割模型容易受到敌意例子的影响。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法的性能会下降。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。



## **28. Untargeted Near-collision Attacks in Biometric Recognition**

生物特征识别中的无目标近碰撞攻击 cs.CR

Addition of results and correction of typos

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2304.01580v2) [paper-pdf](http://arxiv.org/pdf/2304.01580v2)

**Authors**: Axel Durbet, Paul-Marie Grollemund, Kevin Thiry-Atighehchi

**Abstract**: A biometric recognition system can operate in two distinct modes, identification or verification. In the first mode, the system recognizes an individual by searching the enrolled templates of all the users for a match. In the second mode, the system validates a user's identity claim by comparing the fresh provided template with the enrolled template. The biometric transformation schemes usually produce binary templates that are better handled by cryptographic schemes, and the comparison is based on a distance that leaks information about the similarities between two biometric templates. Both the experimentally determined false match rate and false non-match rate through recognition threshold adjustment define the recognition accuracy, and hence the security of the system. To the best of our knowledge, few works provide a formal treatment of the security under minimum leakage of information, i.e., the binary outcome of a comparison with a threshold. In this paper, we rely on probabilistic modelling to quantify the security strength of binary templates. We investigate the influence of template size, database size and threshold on the probability of having a near-collision. We highlight several untargeted attacks on biometric systems considering naive and adaptive adversaries. Interestingly, these attacks can be launched both online and offline and, both in the identification mode and in the verification mode. We discuss the choice of parameters through the generic presented attacks.

摘要: 生物识别系统可以在两种截然不同的模式下工作，即识别或验证。在第一种模式中，系统通过在所有用户的注册模板中搜索匹配项来识别个人。在第二种模式中，系统通过将新提供的模板与注册的模板进行比较来验证用户的身份声明。生物特征转换方案通常产生由加密方案更好地处理的二进制模板，并且比较基于泄露关于两个生物特征模板之间的相似性的信息的距离。实验确定的误匹配率和通过调整识别阈值确定的误不匹配率都定义了识别精度，从而决定了系统的安全性。就我们所知，很少有文献在信息泄露最小的情况下提供安全的形式处理，即与阈值比较的二进制结果。在本文中，我们依赖于概率建模来量化二进制模板的安全强度。我们研究了模板大小、数据库大小和阈值对近碰撞概率的影响。我们重点介绍了几种针对生物识别系统的非定向攻击，考虑到了天真和自适应的对手。有趣的是，这些攻击既可以在线上也可以离线发起，也可以在识别模式和验证模式下发起。我们通过一般提出的攻击讨论参数的选择。



## **29. Distracting Downpour: Adversarial Weather Attacks for Motion Estimation**

分散注意力的倾盆大雨：运动估计的对抗性天气攻击 cs.CV

This work is a direct extension of our extended abstract from  arXiv:2210.11242

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06716v1) [paper-pdf](http://arxiv.org/pdf/2305.06716v1)

**Authors**: Jenny Schmalfuss, Lukas Mehl, Andrés Bruhn

**Abstract**: Current adversarial attacks on motion estimation, or optical flow, optimize small per-pixel perturbations, which are unlikely to appear in the real world. In contrast, adverse weather conditions constitute a much more realistic threat scenario. Hence, in this work, we present a novel attack on motion estimation that exploits adversarially optimized particles to mimic weather effects like snowflakes, rain streaks or fog clouds. At the core of our attack framework is a differentiable particle rendering system that integrates particles (i) consistently over multiple time steps (ii) into the 3D space (iii) with a photo-realistic appearance. Through optimization, we obtain adversarial weather that significantly impacts the motion estimation. Surprisingly, methods that previously showed good robustness towards small per-pixel perturbations are particularly vulnerable to adversarial weather. At the same time, augmenting the training with non-optimized weather increases a method's robustness towards weather effects and improves generalizability at almost no additional cost.

摘要: 目前对运动估计或光流的敌意攻击，优化了每像素的小扰动，这在现实世界中不太可能出现。相比之下，不利的天气条件构成了更现实的威胁情景。因此，在这项工作中，我们提出了一种新颖的攻击运动估计的方法，该方法利用反向优化的粒子来模拟雪花、雨带或雾云等天气效果。在我们的攻击框架的核心是一个可区分的粒子渲染系统，它以照片般的外观将粒子(I)在多个时间步骤(Ii)一致地集成到3D空间(Iii)中。通过优化，得到对运动估计有显著影响的对抗性天气。令人惊讶的是，以前对每像素微小扰动表现出良好稳健性的方法特别容易受到恶劣天气的影响。同时，在不增加额外代价的情况下，用非优化的天气来增加训练，增加了方法对天气影响的鲁棒性，并提高了泛化能力。



## **30. Beyond the Model: Data Pre-processing Attack to Deep Learning Models in Android Apps**

超越模型：Android应用程序中对深度学习模型的数据预处理攻击 cs.CR

Accepted to AsiaCCS WorkShop on Secure and Trustworthy Deep Learning  Systems (SecTL 2023)

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.03963v2) [paper-pdf](http://arxiv.org/pdf/2305.03963v2)

**Authors**: Ye Sang, Yujin Huang, Shuo Huang, Helei Cui

**Abstract**: The increasing popularity of deep learning (DL) models and the advantages of computing, including low latency and bandwidth savings on smartphones, have led to the emergence of intelligent mobile applications, also known as DL apps, in recent years. However, this technological development has also given rise to several security concerns, including adversarial examples, model stealing, and data poisoning issues. Existing works on attacks and countermeasures for on-device DL models have primarily focused on the models themselves. However, scant attention has been paid to the impact of data processing disturbance on the model inference. This knowledge disparity highlights the need for additional research to fully comprehend and address security issues related to data processing for on-device models. In this paper, we introduce a data processing-based attacks against real-world DL apps. In particular, our attack could influence the performance and latency of the model without affecting the operation of a DL app. To demonstrate the effectiveness of our attack, we carry out an empirical study on 517 real-world DL apps collected from Google Play. Among 320 apps utilizing MLkit, we find that 81.56\% of them can be successfully attacked.   The results emphasize the importance of DL app developers being aware of and taking actions to secure on-device models from the perspective of data processing.

摘要: 近年来，深度学习模型的日益流行以及计算的优势，包括智能手机上的低延迟和带宽节省，导致了智能移动应用程序的出现，也被称为深度学习应用程序。然而，这种技术的发展也引起了一些安全问题，包括对抗性例子、模型窃取和数据中毒问题。现有的针对设备上DL模型的攻击和对策的研究主要集中在模型本身。然而，数据处理干扰对模型推理的影响还没有引起足够的重视。这种知识差距突出表明，需要进行更多的研究，以充分理解和解决与设备上模型的数据处理相关的安全问题。在本文中，我们介绍了一种基于数据处理的针对现实世界数字图书馆应用程序的攻击。特别是，我们的攻击可能会影响模型的性能和延迟，而不会影响DL应用的操作。为了证明我们攻击的有效性，我们对从Google Play收集的517个真实数字图书馆应用程序进行了实证研究。在320个使用MLkit的应用中，我们发现81.56%的应用可以被成功攻击。这些结果强调了数字图书馆应用程序开发人员从数据处理的角度意识到并采取行动保护设备上模型的重要性。



## **31. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

关于图神经扩散对拓扑扰动的稳健性 cs.LG

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2209.07754v2) [paper-pdf](http://arxiv.org/pdf/2209.07754v2)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstract**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.

摘要: 图上的神经扩散是一类新的图神经网络，近年来受到越来越多的关注。图神经偏微分方程组(PDE)在解决图神经网络(GNN)的常见障碍(如过光滑和瓶颈问题)方面的能力已被研究，但其对对手攻击的稳健性尚未得到研究。在这项工作中，我们研究了图神经偏微分方程的稳健性。我们的经验证明，与其他GNN相比，图神经PDE在本质上对拓扑扰动具有更强的鲁棒性。通过利用图的拓扑扰动下热半群的稳定性，我们提供了对这一现象的见解。我们讨论了各种图扩散算子，并将它们与现有的图神经偏微分方程联系起来。此外，我们还提出了一个通用的图神经偏微分方程框架，基于该框架可以定义一类新的健壮GNN。我们在几个基准数据集上验证了新模型取得了相当于最先进的性能。



## **32. Prevention of shoulder-surfing attacks using shifting condition using digraph substitution rules**

基于有向图替换规则的移位条件防止冲浪攻击 cs.CR

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06549v1) [paper-pdf](http://arxiv.org/pdf/2305.06549v1)

**Authors**: Amanul Islam, Fazidah Othman, Nazmus Sakib, Hafiz Md. Hasan Babu

**Abstract**: Graphical passwords are implemented as an alternative scheme to replace alphanumeric passwords to help users to memorize their password. However, most of the graphical password systems are vulnerable to shoulder-surfing attack due to the usage of the visual interface. In this research, a method that uses shifting condition with digraph substitution rules is proposed to address shoulder-surfing attack problem. The proposed algorithm uses both password images and decoy images throughout the user authentication procedure to confuse adversaries from obtaining the password images via direct observation or watching from a recorded session. The pass-images generated by this suggested algorithm are random and can only be generated if the algorithm is fully understood. As a result, adversaries will have no clue to obtain the right password images to log in. A user study was undertaken to assess the proposed method's effectiveness to avoid shoulder-surfing attacks. The results of the user study indicate that the proposed approach can withstand shoulder-surfing attacks (both direct observation and video recording method).The proposed method was tested and the results showed that it is able to resist shoulder-surfing and frequency of occurrence analysis attacks. Moreover, the experience gained in this research can be pervaded the gap on the realm of knowledge of the graphical password.

摘要: 图形密码作为替代字母数字密码的替代方案来实施，以帮助用户记住他们的密码。然而，由于可视化界面的使用，大多数图形化密码系统都容易受到肩部冲浪攻击。针对肩部冲浪攻击问题，提出了一种基于有向图替换规则的移位条件攻击方法。该算法在用户认证过程中同时使用口令图像和诱骗图像，以迷惑攻击者通过直接观察或从记录的会话中观看来获得口令图像。该算法生成的通道图像是随机的，只有在充分理解该算法的情况下才能生成。因此，攻击者将没有任何线索来获取正确的密码图像来登录。进行了一项用户研究，以评估所提出的方法在避免肩部冲浪攻击方面的有效性。用户研究结果表明，该方法能够抵抗直接观察法和录像法的肩部冲浪攻击，并对该方法进行了测试，结果表明该方法能够抵抗肩部冲浪和频度分析攻击。此外，在本研究中获得的经验可以填补图形密码知识领域的空白。



## **33. Inter-frame Accelerate Attack against Video Interpolation Models**

针对视频插补模型的帧间加速攻击 cs.CV

**SubmitDate**: 2023-05-11    [abs](http://arxiv.org/abs/2305.06540v1) [paper-pdf](http://arxiv.org/pdf/2305.06540v1)

**Authors**: Junpei Liao, Zhikai Chen, Liang Yi, Wenyuan Yang, Baoyuan Wu, Xiaochun Cao

**Abstract**: Deep learning based video frame interpolation (VIF) method, aiming to synthesis the intermediate frames to enhance video quality, have been highly developed in the past few years. This paper investigates the adversarial robustness of VIF models. We apply adversarial attacks to VIF models and find that the VIF models are very vulnerable to adversarial examples. To improve attack efficiency, we suggest to make full use of the property of video frame interpolation task. The intuition is that the gap between adjacent frames would be small, leading to the corresponding adversarial perturbations being similar as well. Then we propose a novel attack method named Inter-frame Accelerate Attack (IAA) that initializes the perturbation as the perturbation for the previous adjacent frame and reduces the number of attack iterations. It is shown that our method can improve attack efficiency greatly while achieving comparable attack performance with traditional methods. Besides, we also extend our method to video recognition models which are higher level vision tasks and achieves great attack efficiency.

摘要: 基于深度学习的视频帧内插方法(VIF)旨在合成中间帧以提高视频质量，在过去的几年中得到了很大的发展。本文研究了VIF模型的对抗稳健性。我们将对抗性攻击应用于VIF模型，发现VIF模型非常容易受到对抗性例子的攻击。为了提高攻击效率，我们建议充分利用视频帧内插任务的特性。直觉是，相邻帧之间的间隙会很小，导致相应的对抗性扰动也是相似的。然后，我们提出了一种新的攻击方法--帧间加速攻击(IAA)，该方法将扰动初始化为对前一相邻帧的扰动，并减少了攻击迭代的次数。实验结果表明，该方法在取得与传统方法相当的攻击性能的同时，大大提高了攻击效率。此外，我们还将我们的方法扩展到视频识别模型，这些模型是较高级别的视觉任务，具有很高的攻击效率。



## **34. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

20 pages, 6 figures

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2210.14410v2) [paper-pdf](http://arxiv.org/pdf/2210.14410v2)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



## **35. Towards Adversarial-Resilient Deep Neural Networks for False Data Injection Attack Detection in Power Grids**

用于电网虚假数据注入攻击检测的对抗性深度神经网络 cs.CR

This paper has been accepted by the the 32nd International Conference  on Computer Communications and Networks (ICCCN 2023)

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2102.09057v2) [paper-pdf](http://arxiv.org/pdf/2102.09057v2)

**Authors**: Jiangnan Li, Yingyuan Yang, Jinyuan Stella Sun, Kevin Tomsovic, Hairong Qi

**Abstract**: False data injection attacks (FDIAs) pose a significant security threat to power system state estimation. To detect such attacks, recent studies have proposed machine learning (ML) techniques, particularly deep neural networks (DNNs). However, most of these methods fail to account for the risk posed by adversarial measurements, which can compromise the reliability of DNNs in various ML applications. In this paper, we present a DNN-based FDIA detection approach that is resilient to adversarial attacks. We first analyze several adversarial defense mechanisms used in computer vision and show their inherent limitations in FDIA detection. We then propose an adversarial-resilient DNN detection framework for FDIA that incorporates random input padding in both the training and inference phases. Our simulations, based on an IEEE standard power system, demonstrate that this framework significantly reduces the effectiveness of adversarial attacks while having a negligible impact on the DNNs' detection performance.

摘要: 虚假数据注入攻击(FDIA)对电力系统状态估计造成了严重的安全威胁。为了检测此类攻击，最近的研究提出了机器学习(ML)技术，特别是深度神经网络(DNN)。然而，这些方法中的大多数都没有考虑到对抗性测量所带来的风险，这可能会损害DNN在各种ML应用中的可靠性。在本文中，我们提出了一种基于DNN的对敌方攻击具有弹性的FDIA检测方法。我们首先分析了计算机视觉中使用的几种对抗性防御机制，并指出了它们在FDIA检测中的固有局限性。然后，我们提出了一种用于FDIA的对抗性DNN检测框架，该框架在训练和推理阶段都加入了随机输入填充。基于IEEE标准电力系统的仿真表明，该框架显著降低了对抗性攻击的有效性，而对DNN的检测性能影响可以忽略不计。



## **36. Invisible Backdoor Attack with Dynamic Triggers against Person Re-identification**

利用动态触发器对个人重新身份进行隐形后门攻击 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2211.10933v2) [paper-pdf](http://arxiv.org/pdf/2211.10933v2)

**Authors**: Wenli Sun, Xinyang Jiang, Shuguang Dou, Dongsheng Li, Duoqian Miao, Cheng Deng, Cairong Zhao

**Abstract**: In recent years, person Re-identification (ReID) has rapidly progressed with wide real-world applications, but also poses significant risks of adversarial attacks. In this paper, we focus on the backdoor attack on deep ReID models. Existing backdoor attack methods follow an all-to-one or all-to-all attack scenario, where all the target classes in the test set have already been seen in the training set. However, ReID is a much more complex fine-grained open-set recognition problem, where the identities in the test set are not contained in the training set. Thus, previous backdoor attack methods for classification are not applicable for ReID. To ameliorate this issue, we propose a novel backdoor attack on deep ReID under a new all-to-unknown scenario, called Dynamic Triggers Invisible Backdoor Attack (DT-IBA). Instead of learning fixed triggers for the target classes from the training set, DT-IBA can dynamically generate new triggers for any unknown identities. Specifically, an identity hashing network is proposed to first extract target identity information from a reference image, which is then injected into the benign images by image steganography. We extensively validate the effectiveness and stealthiness of the proposed attack on benchmark datasets, and evaluate the effectiveness of several defense methods against our attack.

摘要: 近年来，身份识别技术发展迅速，在实际应用中得到了广泛的应用，但同时也带来了巨大的对抗性攻击风险。本文主要研究对深度Reid模型的后门攻击。现有的后门攻击方法遵循All-to-One或All-to-All攻击方案，其中测试集中的所有目标类都已在训练集中看到。然而，REID是一个更复杂的细粒度开集识别问题，其中测试集中的身份不包含在训练集中。因此，以前用于分类的后门攻击方法不适用于REID。为了改善这一问题，我们提出了一种新的全未知场景下对深度Reid的后门攻击，称为动态触发器不可见后门攻击(DT-IBA)。DT-IBA不需要从训练集中学习目标类的固定触发器，而是可以为任何未知身份动态生成新的触发器。具体地说，提出了一种身份散列网络，首先从参考图像中提取目标身份信息，然后通过图像隐写将这些身份信息注入到良性图像中。我们在基准数据集上广泛验证了提出的攻击的有效性和隐蔽性，并评估了几种防御方法对我们的攻击的有效性。



## **37. The Robustness of Computer Vision Models against Common Corruptions: a Survey**

计算机视觉模型对常见腐败的稳健性研究综述 cs.CV

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.06024v1) [paper-pdf](http://arxiv.org/pdf/2305.06024v1)

**Authors**: Shunxin Wang, Raymond Veldhuis, Nicola Strisciuglio

**Abstract**: The performance of computer vision models is susceptible to unexpected changes in input images when deployed in real scenarios. These changes are referred to as common corruptions. While they can hinder the applicability of computer vision models in real-world scenarios, they are not always considered as a testbed for model generalization and robustness. In this survey, we present a comprehensive and systematic overview of methods that improve corruption robustness of computer vision models. Unlike existing surveys that focus on adversarial attacks and label noise, we cover extensively the study of robustness to common corruptions that can occur when deploying computer vision models to work in practical applications. We describe different types of image corruption and provide the definition of corruption robustness. We then introduce relevant evaluation metrics and benchmark datasets. We categorize methods into four groups. We also cover indirect methods that show improvements in generalization and may improve corruption robustness as a byproduct. We report benchmark results collected from the literature and find that they are not evaluated in a unified manner, making it difficult to compare and analyze. We thus built a unified benchmark framework to obtain directly comparable results on benchmark datasets. Furthermore, we evaluate relevant backbone networks pre-trained on ImageNet using our framework, providing an overview of the base corruption robustness of existing models to help choose appropriate backbones for computer vision tasks. We identify that developing methods to handle a wide range of corruptions and efficiently learn with limited data and computational resources is crucial for future development. Additionally, we highlight the need for further investigation into the relationship among corruption robustness, OOD generalization, and shortcut learning.

摘要: 当计算机视觉模型部署在真实场景中时，其性能很容易受到输入图像中意外变化的影响。这些变化被称为常见的腐败。虽然它们会阻碍计算机视觉模型在现实世界场景中的适用性，但它们并不总是被视为模型泛化和健壮性的试验台。在这次调查中，我们全面和系统地概述了提高计算机视觉模型的腐败稳健性的方法。与专注于对抗性攻击和标签噪声的现有调查不同，我们广泛涵盖了对在实际应用中部署计算机视觉模型时可能发生的常见腐败的稳健性研究。我们描述了不同类型的图像损坏，并给出了损坏稳健性的定义。然后我们介绍了相关的评估指标和基准数据集。我们将方法分为四类。我们还介绍了间接方法，这些方法显示了泛化方面的改进，并可能作为副产品提高腐败健壮性。我们报告了从文献中收集的基准结果，发现它们没有以统一的方式进行评估，这使得比较和分析变得困难。因此，我们建立了一个统一的基准框架，以获得基准数据集的直接可比结果。此外，我们使用我们的框架评估了在ImageNet上预先训练的相关骨干网络，提供了现有模型的基本腐败稳健性的概述，以帮助选择合适的骨干网络来执行计算机视觉任务。我们认识到，开发方法来处理广泛的腐败问题，并利用有限的数据和计算资源有效地学习，对未来的发展至关重要。此外，我们强调有必要进一步调查腐败稳健性、面向对象设计泛化和快捷学习之间的关系。



## **38. Robust multi-agent coordination via evolutionary generation of auxiliary adversarial attackers**

通过进化生成辅助对抗性攻击者实现健壮的多智能体协作 cs.MA

In: Proceedings of the 37th AAAI Conference on Artificial  Intelligence (AAAI'23), 2023

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05909v1) [paper-pdf](http://arxiv.org/pdf/2305.05909v1)

**Authors**: Lei Yuan, Zi-Qian Zhang, Ke Xue, Hao Yin, Feng Chen, Cong Guan, Li-He Li, Chao Qian, Yang Yu

**Abstract**: Cooperative multi-agent reinforcement learning (CMARL) has shown to be promising for many real-world applications. Previous works mainly focus on improving coordination ability via solving MARL-specific challenges (e.g., non-stationarity, credit assignment, scalability), but ignore the policy perturbation issue when testing in a different environment. This issue hasn't been considered in problem formulation or efficient algorithm design. To address this issue, we firstly model the problem as a limited policy adversary Dec-POMDP (LPA-Dec-POMDP), where some coordinators from a team might accidentally and unpredictably encounter a limited number of malicious action attacks, but the regular coordinators still strive for the intended goal. Then, we propose Robust Multi-Agent Coordination via Evolutionary Generation of Auxiliary Adversarial Attackers (ROMANCE), which enables the trained policy to encounter diversified and strong auxiliary adversarial attacks during training, thus achieving high robustness under various policy perturbations. Concretely, to avoid the ego-system overfitting to a specific attacker, we maintain a set of attackers, which is optimized to guarantee the attackers high attacking quality and behavior diversity. The goal of quality is to minimize the ego-system coordination effect, and a novel diversity regularizer based on sparse action is applied to diversify the behaviors among attackers. The ego-system is then paired with a population of attackers selected from the maintained attacker set, and alternately trained against the constantly evolving attackers. Extensive experiments on multiple scenarios from SMAC indicate our ROMANCE provides comparable or better robustness and generalization ability than other baselines.

摘要: 协作多智能体强化学习(CMARL)已被证明在许多实际应用中具有广阔的应用前景。以往的工作主要集中在通过解决MAIL特有的挑战(如非平稳性、信用分配、可扩展性)来提高协调能力，而忽略了在不同环境中测试时的策略扰动问题。这个问题在问题描述和有效的算法设计中都没有考虑到。为了解决这个问题，我们首先将问题建模为有限策略对手DEC-POMDP(LPA-DEC-POMDP)，其中团队中的一些协调者可能意外地和不可预测地遇到有限数量的恶意行为攻击，但常规协调者仍然努力实现预期的目标。在此基础上，提出了基于辅助对抗性攻击进化生成的稳健多智能体协调算法(ROMANCE)，使训练后的策略在训练过程中能够遇到多样化且强的辅助对抗性攻击，从而在各种策略扰动下具有较高的鲁棒性。具体地说，为了避免自我系统对特定攻击者的过度匹配，我们维护了一组攻击者，并对其进行了优化，以保证攻击者的高攻击质量和行为多样性。质量的目标是最小化自我-系统协调效应，并采用一种新的基于稀疏动作的多样性正则化算法来使攻击者的行为多样化。然后，自我系统与从维护的攻击者集合中选择的一群攻击者配对，并交替地针对不断演变的攻击者进行训练。在SMAC的多个场景上的大量实验表明，我们的Romance提供了与其他基线相当或更好的健壮性和泛化能力。



## **39. RNNS: Representation Nearest Neighbor Search Black-Box Attack on Code Models**

RNNS：代码模型上的表示最近邻搜索黑盒攻击 cs.CR

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05896v1) [paper-pdf](http://arxiv.org/pdf/2305.05896v1)

**Authors**: Jie Zhang, Wei Ma, Qiang Hu, Xiaofei Xie, Yves Le Traon, Yang Liu

**Abstract**: Pre-trained code models are mainly evaluated using the in-distribution test data. The robustness of models, i.e., the ability to handle hard unseen data, still lacks evaluation. In this paper, we propose a novel search-based black-box adversarial attack guided by model behaviours for pre-trained programming language models, named Representation Nearest Neighbor Search(RNNS), to evaluate the robustness of Pre-trained PL models. Unlike other black-box adversarial attacks, RNNS uses the model-change signal to guide the search in the space of the variable names collected from real-world projects. Specifically, RNNS contains two main steps, 1) indicate which variable (attack position location) we should attack based on model uncertainty, and 2) search which adversarial tokens we should use for variable renaming according to the model behaviour observations. We evaluate RNNS on 6 code tasks (e.g., clone detection), 3 programming languages (Java, Python, and C), and 3 pre-trained code models: CodeBERT, GraphCodeBERT, and CodeT5. The results demonstrate that RNNS outperforms the state-of-the-art black-box attacking methods (MHM and ALERT) in terms of attack success rate (ASR) and query times (QT). The perturbation of generated adversarial examples from RNNS is smaller than the baselines with respect to the number of replaced variables and the variable length change. Our experiments also show that RNNS is efficient in attacking the defended models and is useful for adversarial training.

摘要: 预先训练的代码模型主要使用分发内测试数据进行评估。模型的稳健性，即处理硬的看不见的数据的能力，仍然缺乏评估。针对预先训练的程序设计语言模型，提出了一种以模型行为为导向的基于搜索的黑盒对抗攻击方法--表示最近邻搜索算法(RNNS)，以评估预先训练的程序设计语言模型的健壮性。与其他黑盒对抗性攻击不同，RNNS使用模型更改信号来指导在从现实世界项目中收集的变量名称空间中的搜索。具体地说，RNNS包含两个主要步骤，1)根据模型的不确定性指示我们应该攻击哪个变量(攻击位置)，2)根据模型行为观察寻找应该使用哪些敌意标记进行变量重命名。我们在6个代码任务(例如克隆检测)、3种编程语言(Java、Python和C)以及3种预先训练的代码模型上对RNNS进行了评估：CodeBERT、GraphCodeBERT和CodeT5。结果表明，RNNS在攻击成功率(ASR)和查询次数(Qt)方面均优于目前最先进的黑盒攻击方法(MHM和ALERT)。从RNNS生成的对抗性样本在替换变量的数量和可变长度变化方面的扰动小于基线。我们的实验还表明，RNNS在攻击防御模型方面是有效的，并且对于对抗性训练是有用的。



## **40. Quantization Aware Attack: Enhancing the Transferability of Adversarial Attacks across Target Models with Different Quantization Bitwidths**

量化感知攻击：提高敌意攻击在不同量化位宽的目标模型上的可转移性 cs.CR

9 pages

**SubmitDate**: 2023-05-10    [abs](http://arxiv.org/abs/2305.05875v1) [paper-pdf](http://arxiv.org/pdf/2305.05875v1)

**Authors**: Yulong Yang, Chenhao Lin, Qian Li, Chao Shen, Dawei Zhou, Nannan Wang, Tongliang Liu

**Abstract**: Quantized Neural Networks (QNNs) receive increasing attention in resource-constrained scenarios because of their excellent generalization abilities, but their robustness under realistic black-box adversarial attacks has not been deeply studied, in which the adversary requires to improve the attack capability across target models with unknown quantization bitwidths. One major challenge is that adversarial examples transfer poorly against QNNs with unknown bitwidths because of the quantization shift and gradient misalignment issues. This paper proposes the Quantization Aware Attack to enhance the attack transferability by making the substitute model ``aware of'' the target of attacking models with multiple bitwidths. Specifically, we design a training objective with multiple bitwidths to align the gradient of the substitute model with the target model with different bitwidths and thus mitigate the negative effect of the above two issues. We conduct comprehensive evaluations by performing multiple transfer-based attacks on standard models and defense models with different architectures and quantization bitwidths. Experimental results show that QAA significantly improves the adversarial transferability of the state-of-the-art attacks by 3.4%-20.9% against normally trained models and 3.7%-13.4% against adversarially trained models on average.

摘要: 量化神经网络(QNN)以其良好的泛化能力在资源受限的场景中受到越来越多的关注，但其在现实黑盒攻击下的健壮性还没有得到深入的研究，在现实的黑盒攻击中，对手要求提高对未知量化比特目标模型的攻击能力。一个主要的挑战是，由于量化漂移和梯度对齐问题，对抗性例子在比特宽度未知的QNN上的传输效果很差。为了提高攻击的可转移性，提出了量化感知攻击，通过使替换模型“感知”到多个比特攻击模型的目标。具体地说，我们设计了一个多位宽的训练目标，将替换模型的梯度与不同位宽的目标模型对齐，从而缓解了上述两个问题的负面影响。通过对具有不同体系结构和量化位宽的标准模型和防御模型进行多次基于传输的攻击，进行综合评估。实验结果表明，QAA显著提高了最新攻击的对抗性，对正常训练的模型平均提高了3.4%-20.9%，对对抗性训练的模型平均提高了3.7%-13.4%。



## **41. VSMask: Defending Against Voice Synthesis Attack via Real-Time Predictive Perturbation**

VSMAsk：利用实时预测扰动防御语音合成攻击 cs.SD

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05736v1) [paper-pdf](http://arxiv.org/pdf/2305.05736v1)

**Authors**: Yuanda Wang, Hanqing Guo, Guangjing Wang, Bocheng Chen, Qiben Yan

**Abstract**: Deep learning based voice synthesis technology generates artificial human-like speeches, which has been used in deepfakes or identity theft attacks. Existing defense mechanisms inject subtle adversarial perturbations into the raw speech audios to mislead the voice synthesis models. However, optimizing the adversarial perturbation not only consumes substantial computation time, but it also requires the availability of entire speech. Therefore, they are not suitable for protecting live speech streams, such as voice messages or online meetings. In this paper, we propose VSMask, a real-time protection mechanism against voice synthesis attacks. Different from offline protection schemes, VSMask leverages a predictive neural network to forecast the most effective perturbation for the upcoming streaming speech. VSMask introduces a universal perturbation tailored for arbitrary speech input to shield a real-time speech in its entirety. To minimize the audio distortion within the protected speech, we implement a weight-based perturbation constraint to reduce the perceptibility of the added perturbation. We comprehensively evaluate VSMask protection performance under different scenarios. The experimental results indicate that VSMask can effectively defend against 3 popular voice synthesis models. None of the synthetic voice could deceive the speaker verification models or human ears with VSMask protection. In a physical world experiment, we demonstrate that VSMask successfully safeguards the real-time speech by injecting the perturbation over the air.

摘要: 基于深度学习的语音合成技术生成的人工语音已被用于深度假冒或身份盗窃攻击。现有的防御机制在原始语音音频中注入微妙的对抗性扰动，以误导语音合成模型。然而，优化对抗性扰动不仅需要消耗大量的计算时间，而且还需要整个语音的可用性。因此，它们不适合保护实时语音流，例如语音消息或在线会议。本文提出了一种针对语音合成攻击的实时防护机制VSMASK。与离线保护方案不同，VSMAsk利用预测神经网络来预测即将到来的流传输语音的最有效扰动。VSMask引入了一种为任意语音输入量身定做的通用扰动，以完整地屏蔽实时语音。为了最大限度地减少受保护语音中的音频失真，我们实现了基于权重的扰动约束来降低附加扰动的可感知性。我们综合评估了不同场景下的VSMASK保护性能。实验结果表明，VSMASK能够有效防御3种流行的语音合成模型。任何合成语音都无法欺骗说话人验证模型或具有VSMASK保护的人耳。在物理世界的实验中，我们演示了VSMAsk通过在空中注入扰动来成功地保护实时语音。



## **42. Using Anomaly Detection to Detect Poisoning Attacks in Federated Learning Applications**

在联合学习应用中使用异常检测来检测中毒攻击 cs.LG

We will updated this article soon

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2207.08486v2) [paper-pdf](http://arxiv.org/pdf/2207.08486v2)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl

**Abstract**: Adversarial attacks such as poisoning attacks have attracted the attention of many machine learning researchers. Traditionally, poisoning attacks attempt to inject adversarial training data in order to manipulate the trained model. In federated learning (FL), data poisoning attacks can be generalized to model poisoning attacks, which cannot be detected by simpler methods due to the lack of access to local training data by the detector. State-of-the-art poisoning attack detection methods for FL have various weaknesses, e.g., the number of attackers has to be known or not high enough, working with i.i.d. data only, and high computational complexity. To overcome above weaknesses, we propose a novel framework for detecting poisoning attacks in FL, which employs a reference model based on a public dataset and an auditor model to detect malicious updates. We implemented a detector based on the proposed framework and using a one-class support vector machine (OC-SVM), which reaches the lowest possible computational complexity O(K) where K is the number of clients. We evaluated our detector's performance against state-of-the-art (SOTA) poisoning attacks for two typical applications of FL: electrocardiograph (ECG) classification and human activity recognition (HAR). Our experimental results validated the performance of our detector over other SOTA detection methods.

摘要: 中毒攻击等对抗性攻击引起了许多机器学习研究人员的关注。传统上，中毒攻击试图注入对抗性的训练数据，以操纵训练的模型。在联邦学习中，数据中毒攻击可以被概括为模型中毒攻击，但由于检测器无法访问本地训练数据，因此无法用更简单的方法检测到中毒攻击。目前针对FL的中毒攻击检测方法有很多缺点，例如，攻击者的数量必须已知或不够高，与I.I.D.配合使用。仅限数据，且计算复杂性高。为了克服上述缺陷，我们提出了一种新的FL中毒攻击检测框架，该框架使用基于公共数据集的参考模型和审计者模型来检测恶意更新。我们基于提出的框架实现了一个检测器，并使用了单类支持向量机(OC-SVM)，它达到了最低的计算复杂度O(K)，其中K是客户端的数量。我们针对FL的两个典型应用：心电图分类和人类活动识别(HAR)，评估了我们的检测器对最先进的(SOTA)中毒攻击的性能。我们的实验结果验证了我们的检测器相对于其他SOTA检测方法的性能。



## **43. Improving Adversarial Transferability via Intermediate-level Perturbation Decay**

通过中层扰动衰减提高对手的可转换性 cs.LG

Revision of ICML '23 submission for better clarity

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2304.13410v2) [paper-pdf](http://arxiv.org/pdf/2304.13410v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Intermediate-level attacks that attempt to perturb feature representations following an adversarial direction drastically have shown favorable performance in crafting transferable adversarial examples. Existing methods in this category are normally formulated with two separate stages, where a directional guide is required to be determined at first and the scalar projection of the intermediate-level perturbation onto the directional guide is enlarged thereafter. The obtained perturbation deviates from the guide inevitably in the feature space, and it is revealed in this paper that such a deviation may lead to sub-optimal attack. To address this issue, we develop a novel intermediate-level method that crafts adversarial examples within a single stage of optimization. In particular, the proposed method, named intermediate-level perturbation decay (ILPD), encourages the intermediate-level perturbation to be in an effective adversarial direction and to possess a great magnitude simultaneously. In-depth discussion verifies the effectiveness of our method. Experimental results show that it outperforms state-of-the-arts by large margins in attacking various victim models on ImageNet (+10.07% on average) and CIFAR-10 (+3.88% on average). Our code is at https://github.com/qizhangli/ILPD-attack.

摘要: 中级攻击试图按照对抗性方向彻底扰乱特征表示，在制作可转移的对抗性示例方面表现出了良好的性能。现有的这类方法通常分为两个不同的阶段，首先需要确定一个方向导轨，然后放大中层摄动在该方向导轨上的标量投影。所得到的扰动在特征空间中不可避免地偏离了导引，本文揭示了这种偏离可能导致次优攻击。为了解决这个问题，我们开发了一种新的中级方法，该方法在单个优化阶段内创建对抗性示例。特别是，所提出的方法，称为中层扰动衰变(ILPD)，它鼓励中层扰动朝着有效的对抗性方向发展，同时具有较大的幅度。通过深入讨论，验证了该方法的有效性。实验结果表明，在ImageNet(平均+10.07%)和CIFAR-10(平均+3.88%)上攻击各种受害者模型时，该算法的性能明显优于最新的攻击模型。我们的代码在https://github.com/qizhangli/ILPD-attack.



## **44. Turning Privacy-preserving Mechanisms against Federated Learning**

将隐私保护机制转向联合学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05355v1) [paper-pdf](http://arxiv.org/pdf/2305.05355v1)

**Authors**: Marco Arazzi, Mauro Conti, Antonino Nocera, Stjepan Picek

**Abstract**: Recently, researchers have successfully employed Graph Neural Networks (GNNs) to build enhanced recommender systems due to their capability to learn patterns from the interaction between involved entities. In addition, previous studies have investigated federated learning as the main solution to enable a native privacy-preserving mechanism for the construction of global GNN models without collecting sensitive data into a single computation unit. Still, privacy issues may arise as the analysis of local model updates produced by the federated clients can return information related to sensitive local data. For this reason, experts proposed solutions that combine federated learning with Differential Privacy strategies and community-driven approaches, which involve combining data from neighbor clients to make the individual local updates less dependent on local sensitive data. In this paper, we identify a crucial security flaw in such a configuration, and we design an attack capable of deceiving state-of-the-art defenses for federated learning. The proposed attack includes two operating modes, the first one focusing on convergence inhibition (Adversarial Mode), and the second one aiming at building a deceptive rating injection on the global federated model (Backdoor Mode). The experimental results show the effectiveness of our attack in both its modes, returning on average 60% performance detriment in all the tests on Adversarial Mode and fully effective backdoors in 93% of cases for the tests performed on Backdoor Mode.

摘要: 最近，研究人员已经成功地使用图神经网络(GNN)来构建增强的推荐系统，这是因为它们能够从相关实体之间的交互中学习模式。此外，以前的研究已经将联合学习作为主要解决方案，以实现在不将敏感数据收集到单个计算单元的情况下构建全局GNN模型的本地隐私保护机制。尽管如此，隐私问题可能会出现，因为对联合客户端生成的本地模型更新的分析可能会返回与敏感本地数据相关的信息。为此，专家们提出了将联合学习与差异隐私策略和社区驱动方法相结合的解决方案，其中包括合并来自邻居客户端的数据，以减少个别本地更新对本地敏感数据的依赖。在本文中，我们确定了这种配置中的一个关键安全漏洞，并设计了一个能够欺骗联邦学习的最新防御的攻击。提出的攻击包括两种工作模式，第一种集中在收敛抑制(对抗性模式)，第二种旨在在全球联邦模型上建立欺骗性评级注入(后门模式)。实验结果表明，我们的攻击在两种模式下都是有效的，在对抗性模式下的所有测试中平均返回60%的性能损失，在后门模式上执行的测试中，93%的情况下完全有效的后门程序。



## **45. Data Protection and Security Issues With Network Error Logging**

网络错误记录的数据保护和安全问题 cs.CR

Accepted for SECRYPT'23

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05343v1) [paper-pdf](http://arxiv.org/pdf/2305.05343v1)

**Authors**: Libor Polčák, Kamil Jeřábek

**Abstract**: Network Error Logging helps web server operators detect operational problems in real-time to provide fast and reliable services. This paper analyses Network Error Logging from two angles. Firstly, this paper overviews Network Error Logging from the data protection view. The ePrivacy Directive requires consent for non-essential access to the end devices. Nevertheless, the Network Error Logging design does not allow limiting the tracking to consenting users. Other issues lay in GDPR requirements for transparency and the obligations in the contract between controllers and processors of personal data. Secondly, this paper explains Network Error Logging exploitations to deploy long-time trackers to the victim devices. Even though users should be able to disable Network Error Logging, it is not clear how to do so. Web server operators can mitigate the attack by configuring servers to preventively remove policies that adversaries might have added.

摘要: 网络错误记录帮助Web服务器操作员实时检测运行问题，以提供快速可靠的服务。本文从两个角度对网络错误记录进行了分析。首先，本文从数据保护的角度对网络错误记录进行了综述。电子隐私指令要求对终端设备进行非必要访问的同意。然而，网络错误记录设计不允许将跟踪限制到同意的用户。其他问题包括GDPR对透明度的要求以及个人数据管制员和处理者之间合同中的义务。其次，本文解释了利用网络错误记录漏洞将长期跟踪器部署到受攻击设备。尽管用户应该能够禁用网络错误记录，但不清楚如何这样做。Web服务器运营商可以通过配置服务器以预防性地删除攻击者可能添加的策略来缓解攻击。



## **46. Attack Named Entity Recognition by Entity Boundary Interference**

利用实体边界干扰攻击命名实体识别 cs.CL

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05253v1) [paper-pdf](http://arxiv.org/pdf/2305.05253v1)

**Authors**: Yifei Yang, Hongqiu Wu, Hai Zhao

**Abstract**: Named Entity Recognition (NER) is a cornerstone NLP task while its robustness has been given little attention. This paper rethinks the principles of NER attacks derived from sentence classification, as they can easily violate the label consistency between the original and adversarial NER examples. This is due to the fine-grained nature of NER, as even minor word changes in the sentence can result in the emergence or mutation of any entities, resulting in invalid adversarial examples. To this end, we propose a novel one-word modification NER attack based on a key insight, NER models are always vulnerable to the boundary position of an entity to make their decision. We thus strategically insert a new boundary into the sentence and trigger the Entity Boundary Interference that the victim model makes the wrong prediction either on this boundary word or on other words in the sentence. We call this attack Virtual Boundary Attack (ViBA), which is shown to be remarkably effective when attacking both English and Chinese models with a 70%-90% attack success rate on state-of-the-art language models (e.g. RoBERTa, DeBERTa) and also significantly faster than previous methods.

摘要: 命名实体识别(NER)是一项基础性的自然语言处理任务，但其健壮性却鲜有人关注。本文重新思考了基于句子分类的NER攻击的原理，因为它们很容易破坏原始例子和对抗性例子之间的标签一致性。这是由于NER的细粒度性质，因为即使句子中的微小单词变化也可能导致任何实体的出现或突变，从而导致无效的对抗性例子。为此，我们提出了一种新颖的基于关键洞察力的单字修正NER攻击，NER模型总是容易受到实体边界位置的影响而做出决策。因此，我们策略性地在句子中插入新的边界，并触发实体边界干扰，即受害者模型对句子中的该边界词或其他词做出错误预测。我们将这种攻击称为虚拟边界攻击(VIBA)，该攻击在对最先进的语言模型(如Roberta，DeBERTa)的攻击成功率为70%-90%的情况下，对英文和中文模型的攻击都非常有效，而且攻击速度也明显快于以前的方法。



## **47. Generating Phishing Attacks using ChatGPT**

使用ChatGPT生成网络钓鱼攻击 cs.CR

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05133v1) [paper-pdf](http://arxiv.org/pdf/2305.05133v1)

**Authors**: Sayak Saha Roy, Krishna Vamsi Naragam, Shirin Nilizadeh

**Abstract**: The ability of ChatGPT to generate human-like responses and understand context has made it a popular tool for conversational agents, content creation, data analysis, and research and innovation. However, its effectiveness and ease of accessibility makes it a prime target for generating malicious content, such as phishing attacks, that can put users at risk. In this work, we identify several malicious prompts that can be provided to ChatGPT to generate functional phishing websites. Through an iterative approach, we find that these phishing websites can be made to imitate popular brands and emulate several evasive tactics that have been known to avoid detection by anti-phishing entities. These attacks can be generated using vanilla ChatGPT without the need of any prior adversarial exploits (jailbreaking).

摘要: ChatGPT生成类似人类的响应并理解上下文的能力使其成为对话代理、内容创建、数据分析以及研究和创新的流行工具。然而，它的有效性和易访问性使其成为生成恶意内容的主要目标，例如网络钓鱼攻击，这些内容可能会将用户置于风险之中。在这项工作中，我们识别了几个可以提供给ChatGPT以生成功能性钓鱼网站的恶意提示。通过迭代的方法，我们发现可以让这些钓鱼网站模仿流行品牌，并模仿几种已知的规避策略，以避免被反钓鱼实体发现。这些攻击可以使用普通的ChatGPT生成，而不需要任何先前的对抗性攻击(越狱)。



## **48. Communication-Robust Multi-Agent Learning by Adaptable Auxiliary Multi-Agent Adversary Generation**

基于自适应辅助多智能体对手生成的通信健壮多智能体学习 cs.LG

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2305.05116v1) [paper-pdf](http://arxiv.org/pdf/2305.05116v1)

**Authors**: Lei Yuan, Feng Chen, Zhongzhang Zhang, Yang Yu

**Abstract**: Communication can promote coordination in cooperative Multi-Agent Reinforcement Learning (MARL). Nowadays, existing works mainly focus on improving the communication efficiency of agents, neglecting that real-world communication is much more challenging as there may exist noise or potential attackers. Thus the robustness of the communication-based policies becomes an emergent and severe issue that needs more exploration. In this paper, we posit that the ego system trained with auxiliary adversaries may handle this limitation and propose an adaptable method of Multi-Agent Auxiliary Adversaries Generation for robust Communication, dubbed MA3C, to obtain a robust communication-based policy. In specific, we introduce a novel message-attacking approach that models the learning of the auxiliary attacker as a cooperative problem under a shared goal to minimize the coordination ability of the ego system, with which every information channel may suffer from distinct message attacks. Furthermore, as naive adversarial training may impede the generalization ability of the ego system, we design an attacker population generation approach based on evolutionary learning. Finally, the ego system is paired with an attacker population and then alternatively trained against the continuously evolving attackers to improve its robustness, meaning that both the ego system and the attackers are adaptable. Extensive experiments on multiple benchmarks indicate that our proposed MA3C provides comparable or better robustness and generalization ability than other baselines.

摘要: 在协作多智能体强化学习(MAIL)中，通信可以促进协作。目前，已有的研究主要集中在提高智能体的通信效率上，而忽略了现实世界中可能存在噪声或潜在攻击者的情况下，通信更具挑战性。因此，基于通信的策略的健壮性成为一个迫切而严峻的问题，需要进一步探讨。在本文中，我们假设用辅助对手训练的EGO系统可以处理这一局限性，并提出了一种用于稳健通信的自适应多智能体辅助对手生成方法MA3C，以获得基于通信的健壮策略。具体地说，我们引入了一种新的消息攻击方法，将辅助攻击者的学习建模为一个共享目标下的合作问题，以最小化EGO系统的协调能力，在这种情况下，每个信息通道都可能遭受不同的消息攻击。此外，由于天真的对抗性训练可能会阻碍EGO系统的泛化能力，我们设计了一种基于进化学习的攻击种群生成方法。最后，EGO系统与攻击者群体配对，然后交替地针对不断进化的攻击者进行训练，以提高其健壮性，这意味着EGO系统和攻击者都是自适应的。在多个基准上的大量实验表明，我们提出的MA3C提供了与其他基准相当或更好的稳健性和泛化能力。



## **49. Escaping saddle points in zeroth-order optimization: the power of two-point estimators**

零阶最优化中的鞍点逃逸：两点估计的能力 math.OC

To appear at ICML 2023

**SubmitDate**: 2023-05-09    [abs](http://arxiv.org/abs/2209.13555v3) [paper-pdf](http://arxiv.org/pdf/2209.13555v3)

**Authors**: Zhaolin Ren, Yujie Tang, Na Li

**Abstract**: Two-point zeroth order methods are important in many applications of zeroth-order optimization, such as robotics, wind farms, power systems, online optimization, and adversarial robustness to black-box attacks in deep neural networks, where the problem may be high-dimensional and/or time-varying. Most problems in these applications are nonconvex and contain saddle points. While existing works have shown that zeroth-order methods utilizing $\Omega(d)$ function valuations per iteration (with $d$ denoting the problem dimension) can escape saddle points efficiently, it remains an open question if zeroth-order methods based on two-point estimators can escape saddle points. In this paper, we show that by adding an appropriate isotropic perturbation at each iteration, a zeroth-order algorithm based on $2m$ (for any $1 \leq m \leq d$) function evaluations per iteration can not only find $\epsilon$-second order stationary points polynomially fast, but do so using only $\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$ function evaluations, where $\bar{\psi} \geq \tilde{\Omega}\left(\sqrt{\epsilon}\right)$ is a parameter capturing the extent to which the function of interest exhibits the strict saddle property.

摘要: 两点零阶方法在零阶优化的许多应用中都是重要的，例如机器人、风电场、电力系统、在线优化以及深层神经网络中对黑盒攻击的对抗鲁棒性，这些问题可能是高维的和/或时变的。这些应用中的大多数问题都是非凸的，并且包含鞍点。虽然已有的工作表明，利用每次迭代的$\Omega(D)$函数赋值(其中$d$表示问题的维度)的零级方法可以有效地逃离鞍点，但基于两点估计的零级方法是否能够逃离鞍点仍然是一个悬而未决的问题。本文证明了，通过在每次迭代中加入适当的各向同性扰动，基于每一次迭代的$2m$(对于任意$1\leq m\leq d$)函数求值的零阶算法不仅可以多项式地快速地找到$-二阶驻点，而且只使用$\tilde{O}\left(\frac{d}{m\epsilon^{2}\bar{\psi}}\right)$函数求值，其中，$\bar{\psi}\geq\tilde{\omega}\Left(\Sqrt{\epsilon}\right)$是捕获感兴趣函数展现严格鞍形属性的程度的参数。



## **50. Less is More: Removing Text-regions Improves CLIP Training Efficiency and Robustness**

少即是多：去除文本区域可以提高剪辑训练的效率和健壮性 cs.CV

10 pages, 8 figures

**SubmitDate**: 2023-05-08    [abs](http://arxiv.org/abs/2305.05095v1) [paper-pdf](http://arxiv.org/pdf/2305.05095v1)

**Authors**: Liangliang Cao, Bowen Zhang, Chen Chen, Yinfei Yang, Xianzhi Du, Wencong Zhang, Zhiyun Lu, Yantao Zheng

**Abstract**: The CLIP (Contrastive Language-Image Pre-training) model and its variants are becoming the de facto backbone in many applications. However, training a CLIP model from hundreds of millions of image-text pairs can be prohibitively expensive. Furthermore, the conventional CLIP model doesn't differentiate between the visual semantics and meaning of text regions embedded in images. This can lead to non-robustness when the text in the embedded region doesn't match the image's visual appearance. In this paper, we discuss two effective approaches to improve the efficiency and robustness of CLIP training: (1) augmenting the training dataset while maintaining the same number of optimization steps, and (2) filtering out samples that contain text regions in the image. By doing so, we significantly improve the classification and retrieval accuracy on public benchmarks like ImageNet and CoCo. Filtering out images with text regions also protects the model from typographic attacks. To verify this, we build a new dataset named ImageNet with Adversarial Text Regions (ImageNet-Attr). Our filter-based CLIP model demonstrates a top-1 accuracy of 68.78\%, outperforming previous models whose accuracy was all below 50\%.

摘要: CLIP(对比语言-图像预训练)模型及其变体正在成为许多应用中事实上的支柱。然而，从数以亿计的图像-文本对中训练剪辑模型的成本可能高得令人望而却步。此外，传统的剪辑模型不区分嵌入在图像中的文本区域的视觉语义和含义。当嵌入区域中的文本与图像的视觉外观不匹配时，这可能会导致不稳定。在本文中，我们讨论了两种有效的方法来提高剪辑训练的效率和稳健性：(1)在保持相同的优化步数的情况下扩大训练数据集；(2)过滤掉图像中包含文本区域的样本。通过这样做，我们在ImageNet和CoCo等公共基准上显著提高了分类和检索的准确性。过滤掉带有文本区域的图像还可以保护模型免受排版攻击。为了验证这一点，我们构建了一个名为ImageNet的新数据集，其中包含敌对文本区域(ImageNet-Attr)。我们的基于过滤器的剪辑模型的TOP-1精度为68.78\%，超过了以前的精度都在50\%以下的模型。



