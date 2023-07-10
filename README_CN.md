# Latest Adversarial Attack Papers
**update at 2023-07-10 10:31:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When and How to Fool Explainable Models (and Humans) with Adversarial Examples**

什么时候以及如何用对抗性的例子愚弄可解释的模型(和人类) cs.LG

Updated version. 43 pages, 9 figures, 4 tables

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2107.01943v2) [paper-pdf](http://arxiv.org/pdf/2107.01943v2)

**Authors**: Jon Vadillo, Roberto Santana, Jose A. Lozano

**Abstract**: Reliable deployment of machine learning models such as neural networks continues to be challenging due to several limitations. Some of the main shortcomings are the lack of interpretability and the lack of robustness against adversarial examples or out-of-distribution inputs. In this exploratory review, we explore the possibilities and limits of adversarial attacks for explainable machine learning models. First, we extend the notion of adversarial examples to fit in explainable machine learning scenarios, in which the inputs, the output classifications and the explanations of the model's decisions are assessed by humans. Next, we propose a comprehensive framework to study whether (and how) adversarial examples can be generated for explainable models under human assessment, introducing and illustrating novel attack paradigms. In particular, our framework considers a wide range of relevant yet often ignored factors such as the type of problem, the user expertise or the objective of the explanations, in order to identify the attack strategies that should be adopted in each scenario to successfully deceive the model (and the human). The intention of these contributions is to serve as a basis for a more rigorous and realistic study of adversarial examples in the field of explainable machine learning.

摘要: 由于几个限制，机器学习模型(如神经网络)的可靠部署仍然具有挑战性。其中一些主要缺点是缺乏可解释性，对敌意例子或分配外的投入缺乏稳健性。在这篇探索性综述中，我们探索了针对可解释机器学习模型的对抗性攻击的可能性和局限性。首先，我们将对抗性例子的概念扩展到适合可解释的机器学习场景，在这种场景中，模型的输入、输出分类和解释都是由人类评估的。接下来，我们提出了一个全面的框架来研究是否(以及如何)在人类评估下可以为可解释的模型生成对抗性实例，引入并说明了新的攻击范例。特别是，我们的框架考虑了广泛的相关但经常被忽略的因素，如问题类型、用户专业知识或解释的目标，以便确定在每个场景中应该采用的攻击策略，以成功欺骗模型(和人)。这些贡献的目的是作为对可解释机器学习领域中的对抗性例子进行更严格和现实的研究的基础。



## **2. Enhancing Adversarial Training via Reweighting Optimization Trajectory**

通过重新加权优化轨迹加强对抗性训练 cs.LG

Accepted by ECML 2023

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.14275v3) [paper-pdf](http://arxiv.org/pdf/2306.14275v3)

**Authors**: Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlaod Menkovski, Lu Yin, Yulong Pei, Mykola Pechenizkiy

**Abstract**: Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-$L_{\infty}$ attack by 1.53\% $\sim$ 6.11\% and meanwhile increases the clean accuracy by 0.55\%$\sim$5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.

摘要: 尽管对抗性训练已经成为提高深度神经网络鲁棒性的事实上的方法，但众所周知，对抗性训练存在令人望而生畏的健壮性过拟合问题，导致不能令人满意的健壮泛化。在过去的几年里，已经提出了一些方法来解决这些缺点，例如额外的正则化、对抗性权重扰动和使用更多数据进行训练。然而，健壮的泛化改进还远远不能令人满意。在本文中，我们以一种全新的视角来应对这一挑战--提炼历史优化轨迹。我们提出了一种新的方法我们已经进行了大量的实验，以证明WOT在各种最先进的对抗性攻击下的有效性。实验结果表明，WOT算法与现有的对抗性训练方法无缝结合，始终克服了健壮性超调的问题，具有更好的对抗性。例如，WOT将AT-PGD在AA-L攻击下的稳健准确率提高了1.53$\sim$6.11\%，同时在SVHN、CIFAR-10、CIFAR-100和微型ImageNet数据集中将CLEAN准确率提高了0.55$\sim$5.47\%。



## **3. Evaluating Similitude and Robustness of Deep Image Denoising Models via Adversarial Attack**

对抗性攻击下深度图像去噪模型的相似性和稳健性评价 cs.CV

**SubmitDate**: 2023-07-07    [abs](http://arxiv.org/abs/2306.16050v2) [paper-pdf](http://arxiv.org/pdf/2306.16050v2)

**Authors**: Jie Ning, Jiebao Sun, Yao Li, Zhichang Guo, Wangmeng Zuo

**Abstract**: Deep neural networks (DNNs) have shown superior performance comparing to traditional image denoising algorithms. However, DNNs are inevitably vulnerable while facing adversarial attacks. In this paper, we propose an adversarial attack method named denoising-PGD which can successfully attack all the current deep denoising models while keep the noise distribution almost unchanged. We surprisingly find that the current mainstream non-blind denoising models (DnCNN, FFDNet, ECNDNet, BRDNet), blind denoising models (DnCNN-B, Noise2Noise, RDDCNN-B, FAN), plug-and-play (DPIR, CurvPnP) and unfolding denoising models (DeamNet) almost share the same adversarial sample set on both grayscale and color images, respectively. Shared adversarial sample set indicates that all these models are similar in term of local behaviors at the neighborhood of all the test samples. Thus, we further propose an indicator to measure the local similarity of models, called robustness similitude. Non-blind denoising models are found to have high robustness similitude across each other, while hybrid-driven models are also found to have high robustness similitude with pure data-driven non-blind denoising models. According to our robustness assessment, data-driven non-blind denoising models are the most robust. We use adversarial training to complement the vulnerability to adversarial attacks. Moreover, the model-driven image denoising BM3D shows resistance on adversarial attacks.

摘要: 与传统的图像去噪算法相比，深度神经网络(DNN)表现出了更好的性能。然而，DNN在面临敌意攻击时不可避免地容易受到攻击。在本文中，我们提出了一种对抗性攻击方法-去噪-PGD，它可以在保持噪声分布几乎不变的情况下成功地攻击所有现有的深度去噪模型。我们惊奇地发现，当前主流的非盲去噪模型(DnCNN、FFDNet、ECNDNet、BRDNet)、盲去噪模型(DnCNN-B、Noise2Noise、RDDCNN-B、FAN)、即插即用模型(DPIR、CurvPnP)和展开去噪模型(DeamNet)分别在灰度和彩色图像上几乎共享相同的对抗性样本集。共享对抗性样本集表明，所有这些模型在所有测试样本的邻域内的局部行为是相似的。因此，我们进一步提出了一个度量模型局部相似性的指标，称为稳健性相似度。非盲去噪模型之间具有较高的鲁棒性相似性，而混合驱动模型与纯数据驱动的非盲去噪模型也具有较高的鲁棒性相似性。根据我们的稳健性评估，数据驱动的非盲去噪模型是最健壮的。我们使用对抗性训练来弥补对对抗性攻击的脆弱性。此外，模型驱动的图像去噪算法BM3D表现出了对敌意攻击的抵抗能力。



## **4. A Vulnerability of Attribution Methods Using Pre-Softmax Scores**

使用Pre-Softmax分数的归因方法的漏洞 cs.LG

7 pages, 5 figures,

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03305v1) [paper-pdf](http://arxiv.org/pdf/2307.03305v1)

**Authors**: Miguel Lerma, Mirtha Lucas

**Abstract**: We discuss a vulnerability involving a category of attribution methods used to provide explanations for the outputs of convolutional neural networks working as classifiers. It is known that this type of networks are vulnerable to adversarial attacks, in which imperceptible perturbations of the input may alter the outputs of the model. In contrast, here we focus on effects that small modifications in the model may cause on the attribution method without altering the model outputs.

摘要: 我们讨论了一个漏洞，涉及一类属性方法，用于解释用作分类器的卷积神经网络的输出。众所周知，这种类型的网络容易受到敌意攻击，在这种攻击中，输入的不知不觉的扰动可能会改变模型的输出。相反，这里我们关注的是在不改变模型输出的情况下，模型中的微小修改可能会对归因方法造成的影响。



## **5. Quantum Solutions to the Privacy vs. Utility Tradeoff**

隐私与效用权衡的量子解 quant-ph

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03118v1) [paper-pdf](http://arxiv.org/pdf/2307.03118v1)

**Authors**: Sagnik Chatterjee, Vyacheslav Kungurtsev

**Abstract**: In this work, we propose a novel architecture (and several variants thereof) based on quantum cryptographic primitives with provable privacy and security guarantees regarding membership inference attacks on generative models. Our architecture can be used on top of any existing classical or quantum generative models. We argue that the use of quantum gates associated with unitary operators provides inherent advantages compared to standard Differential Privacy based techniques for establishing guaranteed security from all polynomial-time adversaries.

摘要: 在这项工作中，我们提出了一个新的体系结构(及其几个变种)，基于量子密码原语，具有可证明的私密性和对生成模型的成员推理攻击的安全性保证。我们的体系结构可以在任何现有的经典或量子生成模型上使用。我们认为，与基于标准差分隐私的技术相比，使用与酉运算符相关的量子门提供了固有的优势，以建立针对所有多项式时间对手的保证安全。



## **6. On Distribution-Preserving Mitigation Strategies for Communication under Cognitive Adversaries**

认知对手环境下通信的分布保持缓解策略研究 cs.IT

Presented at IEEE ISIT 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.03105v1) [paper-pdf](http://arxiv.org/pdf/2307.03105v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: In wireless security, cognitive adversaries are known to inject jamming energy on the victim's frequency band and monitor the same band for countermeasures thereby trapping the victim. Under the class of cognitive adversaries, we propose a new threat model wherein the adversary, upon executing the jamming attack, measures the long-term statistic of Kullback-Leibler Divergence (KLD) between its observations over each of the network frequencies before and after the jamming attack. To mitigate this adversary, we propose a new cooperative strategy wherein the victim takes the assistance for a helper node in the network to reliably communicate its message to the destination. The underlying idea is to appropriately split their energy and time resources such that their messages are reliably communicated without disturbing the statistical distribution of the samples in the network. We present rigorous analyses on the reliability and the covertness metrics at the destination and the adversary, respectively, and then synthesize tractable algorithms to obtain near-optimal division of resources between the victim and the helper. Finally, we show that the obtained near-optimal division of energy facilitates in deceiving the adversary with a KLD estimator.

摘要: 在无线安全中，已知认知对手在受害者的频段上注入干扰能量，并监控同一频段以采取对策，从而诱捕受害者。在认知对手的情况下，我们提出了一个新的威胁模型，在该模型中，对手在实施干扰攻击时，测量干扰攻击前后其在每个网络频率上的观测之间的Kullback-Leibler散度(KLD)的长期统计量。为了缓解这种敌意，我们提出了一种新的合作策略，在该策略中，受害者接受网络中帮助节点的帮助，以可靠地将其消息传递到目的地。其基本思想是适当地分割它们的能量和时间资源，以便在不干扰网络中样本的统计分布的情况下可靠地传递它们的消息。我们分别对目标和对手的可靠性和隐蔽性度量进行了严格的分析，然后综合易处理的算法来获得受害者和帮助者之间的近最优资源分配。最后，我们证明了所得到的接近最优的能量分配便于利用KLD估计来欺骗对手。



## **7. Probabilistic and Semantic Descriptions of Image Manifolds and Their Applications**

图像流形的概率和语义描述及其应用 cs.CV

23 pages, 17 figures, 1 table

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02881v1) [paper-pdf](http://arxiv.org/pdf/2307.02881v1)

**Authors**: Peter Tu, Zhaoyuan Yang, Richard Hartley, Zhiwei Xu, Jing Zhang, Dylan Campbell, Jaskirat Singh, Tianyu Wang

**Abstract**: This paper begins with a description of methods for estimating probability density functions for images that reflects the observation that such data is usually constrained to lie in restricted regions of the high-dimensional image space - not every pattern of pixels is an image. It is common to say that images lie on a lower-dimensional manifold in the high-dimensional space. However, although images may lie on such lower-dimensional manifolds, it is not the case that all points on the manifold have an equal probability of being images. Images are unevenly distributed on the manifold, and our task is to devise ways to model this distribution as a probability distribution. In pursuing this goal, we consider generative models that are popular in AI and computer vision community. For our purposes, generative/probabilistic models should have the properties of 1) sample generation: it should be possible to sample from this distribution according to the modelled density function, and 2) probability computation: given a previously unseen sample from the dataset of interest, one should be able to compute the probability of the sample, at least up to a normalising constant. To this end, we investigate the use of methods such as normalising flow and diffusion models. We then show that such probabilistic descriptions can be used to construct defences against adversarial attacks. In addition to describing the manifold in terms of density, we also consider how semantic interpretations can be used to describe points on the manifold. To this end, we consider an emergent language framework which makes use of variational encoders to produce a disentangled representation of points that reside on a given manifold. Trajectories between points on a manifold can then be described in terms of evolving semantic descriptions.

摘要: 本文首先描述了用于估计图像的概率密度函数的方法，该方法反映了这样的观察，即这种数据通常被限制在高维图像空间的受限区域--并不是每一种像素模式都是图像。人们常说，图像位于高维空间中的低维流形上。然而，尽管图像可能位于这样的低维流形上，但流形上的所有点成为图像的概率并不相等。图像在流形上是不均匀分布的，我们的任务是设计出将这种分布建模为概率分布的方法。在追求这一目标的过程中，我们考虑了人工智能和计算机视觉领域中流行的生成性模型。就我们的目的而言，生成/概率模型应该具有以下特性：1)样本生成：应该能够根据建模的密度函数从该分布中进行样本；以及2)概率计算：给定感兴趣的数据集中以前未见过的样本，应该能够计算该样本的概率，至少达到归一化常数。为此，我们研究了流和扩散模型等方法的使用。然后，我们证明了这种概率描述可以用来构建对抗攻击的防御。除了用密度来描述流形之外，我们还考虑了如何使用语义解释来描述流形上的点。为此，我们考虑了一种新的语言框架，它利用变分编码器来产生驻留在给定流形上的点的无纠缠表示。然后，流形上的点之间的轨迹可以通过不断演变的语义描述来描述。



## **8. NatLogAttack: A Framework for Attacking Natural Language Inference Models with Natural Logic**

NatLogAttack：一个用自然逻辑攻击自然语言推理模型的框架 cs.CL

Published as a conference paper at ACL 2023

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02849v1) [paper-pdf](http://arxiv.org/pdf/2307.02849v1)

**Authors**: Zi'ou Zheng, Xiaodan Zhu

**Abstract**: Reasoning has been a central topic in artificial intelligence from the beginning. The recent progress made on distributed representation and neural networks continues to improve the state-of-the-art performance of natural language inference. However, it remains an open question whether the models perform real reasoning to reach their conclusions or rely on spurious correlations. Adversarial attacks have proven to be an important tool to help evaluate the Achilles' heel of the victim models. In this study, we explore the fundamental problem of developing attack models based on logic formalism. We propose NatLogAttack to perform systematic attacks centring around natural logic, a classical logic formalism that is traceable back to Aristotle's syllogism and has been closely developed for natural language inference. The proposed framework renders both label-preserving and label-flipping attacks. We show that compared to the existing attack models, NatLogAttack generates better adversarial examples with fewer visits to the victim models. The victim models are found to be more vulnerable under the label-flipping setting. NatLogAttack provides a tool to probe the existing and future NLI models' capacity from a key viewpoint and we hope more logic-based attacks will be further explored for understanding the desired property of reasoning.

摘要: 从一开始，推理就是人工智能的中心话题。最近在分布式表示和神经网络方面取得的进展继续提高了自然语言推理的最新性能。然而，这些模型是进行真正的推理来得出结论，还是依赖于虚假的相关性，这仍然是一个悬而未决的问题。对抗性攻击已被证明是帮助评估受害者模型的致命弱点的重要工具。在本研究中，我们探讨了建立基于逻辑形式主义的攻击模型的基本问题。我们建议NatLogAttack以自然逻辑为中心执行系统攻击，自然逻辑是一种经典的逻辑形式主义，可以追溯到亚里士多德的三段论，并为自然语言推理而密切发展。该框架同时提供了标签保留攻击和标签翻转攻击。结果表明，与已有的攻击模型相比，NatLogAttack能够以较少的访问受害者模型生成更好的对抗性实例。受害者模特被发现在标签翻转的设置下更容易受到攻击。NatLogAttack提供了一个工具，可以从一个关键的角度来探索现有和未来的NLI模型的能力，我们希望进一步探索更多基于逻辑的攻击，以理解所需的推理属性。



## **9. Sampling-based Fast Gradient Rescaling Method for Highly Transferable Adversarial Attacks**

一种基于采样的高可转移对抗性攻击快速梯度重缩放方法 cs.CV

10 pages, 6 figures, 7 tables. arXiv admin note: substantial text  overlap with arXiv:2204.02887

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02828v1) [paper-pdf](http://arxiv.org/pdf/2307.02828v1)

**Authors**: Xu Han, Anmin Liu, Chenxuan Yao, Yanbo Fan, Kun He

**Abstract**: Deep neural networks are known to be vulnerable to adversarial examples crafted by adding human-imperceptible perturbations to the benign input. After achieving nearly 100% attack success rates in white-box setting, more focus is shifted to black-box attacks, of which the transferability of adversarial examples has gained significant attention. In either case, the common gradient-based methods generally use the sign function to generate perturbations on the gradient update, that offers a roughly correct direction and has gained great success. But little work pays attention to its possible limitation. In this work, we observe that the deviation between the original gradient and the generated noise may lead to inaccurate gradient update estimation and suboptimal solutions for adversarial transferability. To this end, we propose a Sampling-based Fast Gradient Rescaling Method (S-FGRM). Specifically, we use data rescaling to substitute the sign function without extra computational cost. We further propose a Depth First Sampling method to eliminate the fluctuation of rescaling and stabilize the gradient update. Our method could be used in any gradient-based attacks and is extensible to be integrated with various input transformation or ensemble methods to further improve the adversarial transferability. Extensive experiments on the standard ImageNet dataset show that our method could significantly boost the transferability of gradient-based attacks and outperform the state-of-the-art baselines.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。在白盒环境下达到近100%的攻击成功率后，更多的注意力转移到黑盒攻击上，其中对抗性例子的可转移性得到了显著的关注。在这两种情况下，常用的基于梯度的方法一般都使用符号函数对梯度更新产生扰动，这提供了一个大致正确的方向，并取得了很大的成功。但很少有人注意到它可能存在的局限性。在这项工作中，我们观察到原始梯度和产生的噪声之间的偏差可能导致不准确的梯度更新估计和对抗性可转移性的次最优解。为此，我们提出了一种基于采样的快速梯度重缩放方法(S-FGRM)。具体地说，我们使用数据重缩放来代替符号函数，而不需要额外的计算代价。在此基础上，提出了深度优先的采样方法，消除了重缩放的波动，稳定了梯度的更新。我们的方法可以用于任何基于梯度的攻击，并且可以扩展到与各种输入变换或集成方法相集成，以进一步提高对抗转移的能力。在标准的ImageNet数据集上的大量实验表明，我们的方法可以显著提高基于梯度的攻击的可转移性，并优于最新的基线。



## **10. A Testbed To Study Adversarial Cyber-Attack Strategies in Enterprise Networks**

企业网络中对抗性网络攻击策略研究的试验台 cs.CR

**SubmitDate**: 2023-07-06    [abs](http://arxiv.org/abs/2307.02794v1) [paper-pdf](http://arxiv.org/pdf/2307.02794v1)

**Authors**: Ayush Kumar, David K. Yau

**Abstract**: In this work, we propose a testbed environment to capture the attack strategies of an adversary carrying out a cyber-attack on an enterprise network. The testbed contains nodes with known security vulnerabilities which can be exploited by hackers. Participants can be invited to play the role of a hacker (e.g., black-hat, hacktivist) and attack the testbed. The testbed is designed such that there are multiple attack pathways available to hackers. We describe the working of the testbed components and discuss its implementation on a VMware ESXi server. Finally, we subject our testbed implementation to a few well-known cyber-attack strategies, collect data during the process and present our analysis of the data.

摘要: 在这项工作中，我们提出了一个试验台环境来捕获对企业网络进行网络攻击的对手的攻击策略。测试床包含具有已知安全漏洞的节点，黑客可以利用这些漏洞进行攻击。参与者可以被邀请扮演黑客(例如，黑帽、黑客活动家)的角色并攻击试验床。试验台的设计使得黑客可以使用多种攻击路径。我们将介绍试验床组件的工作原理，并讨论其在VMware ESXi服务器上的实施。最后，我们对我们的试验台实施进行了一些众所周知的网络攻击策略，收集了过程中的数据，并给出了我们对数据的分析。



## **11. Chaos Theory and Adversarial Robustness**

混沌理论与对抗稳健性 cs.LG

14 pages, 6 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2210.13235v2) [paper-pdf](http://arxiv.org/pdf/2210.13235v2)

**Authors**: Jonathan S. Kent

**Abstract**: Neural networks, being susceptible to adversarial attacks, should face a strict level of scrutiny before being deployed in critical or adversarial applications. This paper uses ideas from Chaos Theory to explain, analyze, and quantify the degree to which neural networks are susceptible to or robust against adversarial attacks. To this end, we present a new metric, the "susceptibility ratio," given by $\hat \Psi(h, \theta)$, which captures how greatly a model's output will be changed by perturbations to a given input.   Our results show that susceptibility to attack grows significantly with the depth of the model, which has safety implications for the design of neural networks for production environments. We provide experimental evidence of the relationship between $\hat \Psi$ and the post-attack accuracy of classification models, as well as a discussion of its application to tasks lacking hard decision boundaries. We also demonstrate how to quickly and easily approximate the certified robustness radii for extremely large models, which until now has been computationally infeasible to calculate directly.

摘要: 神经网络容易受到对抗性攻击，在部署到关键或对抗性应用程序之前，应该面临严格的审查。本文利用混沌理论的思想来解释、分析和量化神经网络对敌意攻击的敏感程度或稳健程度。为此，我们提出了一个新的度量，由$\HAT\Psi(h，\theta)$给出的“敏感度比”，它捕捉到模型的输出将因给定输入的扰动而发生多大变化。我们的结果表明，随着模型深度的增加，对攻击的敏感性显著增加，这对生产环境下的神经网络设计具有安全意义。我们提供了分类模型的攻击后精度与$HAT\PSI之间关系的实验证据，并讨论了它在缺乏硬决策边界的任务中的应用。我们还演示了如何快速、轻松地近似极大模型的认证稳健性半径，到目前为止，这在计算上是不可行的直接计算。



## **12. GIT: Detecting Uncertainty, Out-Of-Distribution and Adversarial Samples using Gradients and Invariance Transformations**

GIT：使用梯度和不变性变换检测不确定性、非分布和对抗性样本 cs.LG

Accepted at IJCNN 2023

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02672v1) [paper-pdf](http://arxiv.org/pdf/2307.02672v1)

**Authors**: Julia Lust, Alexandru P. Condurache

**Abstract**: Deep neural networks tend to make overconfident predictions and often require additional detectors for misclassifications, particularly for safety-critical applications. Existing detection methods usually only focus on adversarial attacks or out-of-distribution samples as reasons for false predictions. However, generalization errors occur due to diverse reasons often related to poorly learning relevant invariances. We therefore propose GIT, a holistic approach for the detection of generalization errors that combines the usage of gradient information and invariance transformations. The invariance transformations are designed to shift misclassified samples back into the generalization area of the neural network, while the gradient information measures the contradiction between the initial prediction and the corresponding inherent computations of the neural network using the transformed sample. Our experiments demonstrate the superior performance of GIT compared to the state-of-the-art on a variety of network architectures, problem setups and perturbation types.

摘要: 深度神经网络往往会做出过于自信的预测，并且经常需要额外的检测器来进行错误分类，特别是对于安全关键应用。现有的检测方法通常只关注对抗性攻击或分布不正确的样本作为错误预测的原因。然而，泛化错误是由于各种原因造成的，通常与学习不好的相关不变性有关。因此，我们提出了GIT，一种结合使用梯度信息和不变性变换来检测泛化误差的整体方法。不变性变换的目的是将错误分类的样本移回神经网络的泛化区域，而梯度信息则衡量了初始预测与使用变换后的样本进行相应的神经网络固有计算之间的矛盾。我们的实验表明，与最先进的网络架构、问题设置和扰动类型相比，GIT具有更高的性能。



## **13. Jailbroken: How Does LLM Safety Training Fail?**

越狱：LLM安全培训是如何失败的？ cs.LG

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02483v1) [paper-pdf](http://arxiv.org/pdf/2307.02483v1)

**Authors**: Alexander Wei, Nika Haghtalab, Jacob Steinhardt

**Abstract**: Large language models trained for safety and harmlessness remain susceptible to adversarial misuse, as evidenced by the prevalence of "jailbreak" attacks on early releases of ChatGPT that elicit undesired behavior. Going beyond recognition of the issue, we investigate why such attacks succeed and how they can be created. We hypothesize two failure modes of safety training: competing objectives and mismatched generalization. Competing objectives arise when a model's capabilities and safety goals conflict, while mismatched generalization occurs when safety training fails to generalize to a domain for which capabilities exist. We use these failure modes to guide jailbreak design and then evaluate state-of-the-art models, including OpenAI's GPT-4 and Anthropic's Claude v1.3, against both existing and newly designed attacks. We find that vulnerabilities persist despite the extensive red-teaming and safety-training efforts behind these models. Notably, new attacks utilizing our failure modes succeed on every prompt in a collection of unsafe requests from the models' red-teaming evaluation sets and outperform existing ad hoc jailbreaks. Our analysis emphasizes the need for safety-capability parity -- that safety mechanisms should be as sophisticated as the underlying model -- and argues against the idea that scaling alone can resolve these safety failure modes.

摘要: 经过安全和无害培训的大型语言模型仍然容易受到对手滥用的影响，对早期版本的ChatGPT进行“越狱”攻击的盛行就证明了这一点，这引发了不受欢迎的行为。除了认识到这个问题，我们还调查了此类攻击成功的原因以及如何创建这些攻击。我们假设了安全培训的两种失败模式：目标竞争和不匹配的概括。当模型的能力和安全目标冲突时，就会出现相互竞争的目标，而当安全培训未能概括到存在能力的领域时，就会出现不匹配的泛化。我们使用这些失败模式来指导越狱设计，然后评估最先进的模型，包括OpenAI的GPT-4和Anthropic的Claude v1.3，针对现有的和新设计的攻击。我们发现，尽管在这些模型背后进行了广泛的红色团队和安全培训努力，但漏洞仍然存在。值得注意的是，利用我们的失败模式的新攻击在模型的红团队评估集的不安全请求集合中的每一个提示下都会成功，并且表现优于现有的临时越狱。我们的分析强调了安全能力对等的必要性--安全机制应该与基础模型一样复杂--并反对仅靠扩展就能解决这些安全故障模式的想法。



## **14. Defense against Adversarial Cloud Attack on Remote Sensing Salient Object Detection**

遥感显著目标检测中对敌云攻击的防御 cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2306.17431v2) [paper-pdf](http://arxiv.org/pdf/2306.17431v2)

**Authors**: Huiming Sun, Lan Fu, Jinlong Li, Qing Guo, Zibo Meng, Tianyun Zhang, Yuewei Lin, Hongkai Yu

**Abstract**: Detecting the salient objects in a remote sensing image has wide applications for the interdisciplinary research. Many existing deep learning methods have been proposed for Salient Object Detection (SOD) in remote sensing images and get remarkable results. However, the recent adversarial attack examples, generated by changing a few pixel values on the original remote sensing image, could result in a collapse for the well-trained deep learning based SOD model. Different with existing methods adding perturbation to original images, we propose to jointly tune adversarial exposure and additive perturbation for attack and constrain image close to cloudy image as Adversarial Cloud. Cloud is natural and common in remote sensing images, however, camouflaging cloud based adversarial attack and defense for remote sensing images are not well studied before. Furthermore, we design DefenseNet as a learn-able pre-processing to the adversarial cloudy images so as to preserve the performance of the deep learning based remote sensing SOD model, without tuning the already deployed deep SOD model. By considering both regular and generalized adversarial examples, the proposed DefenseNet can defend the proposed Adversarial Cloud in white-box setting and other attack methods in black-box setting. Experimental results on a synthesized benchmark from the public remote sensing SOD dataset (EORSSD) show the promising defense against adversarial cloud attacks.

摘要: 遥感图像中显著目标的检测在多学科交叉研究中有着广泛的应用。已有的许多深度学习方法被提出用于遥感图像中的显著目标检测，并取得了显著的效果。然而，最近通过改变原始遥感图像上的几个像素值而生成的对抗性攻击实例，可能会导致基于深度学习的训练有素的SOD模型崩溃。与已有的在原始图像上添加扰动的方法不同，我们提出了联合调整攻击的对抗性曝光和加性扰动，并将接近云图的图像约束为对抗性云。云层是遥感图像中常见的自然现象，但基于云层伪装的遥感图像对抗攻防研究较少。此外，我们将DefenseNet设计为对敌意云图进行可学习的预处理，以保持基于深度学习的遥感SOD模型的性能，而不需要调整已经部署的深度SOD模型。通过考虑常规和广义的对抗性实例，所提出的防御网络可以在白盒环境下防御所提出的对抗性云，并在黑盒环境下防御其他攻击方法。在一个基于公共遥感数据集(EORSSD)的合成基准上的实验结果表明，该方法能够有效地防御敌意云攻击。



## **15. On the Adversarial Robustness of Generative Autoencoders in the Latent Space**

生成式自动编码器在潜在空间中的对抗健壮性 cs.LG

18 pages, 12 figures

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02202v1) [paper-pdf](http://arxiv.org/pdf/2307.02202v1)

**Authors**: Mingfei Lu, Badong Chen

**Abstract**: The generative autoencoders, such as the variational autoencoders or the adversarial autoencoders, have achieved great success in lots of real-world applications, including image generation, and signal communication.   However, little concern has been devoted to their robustness during practical deployment.   Due to the probabilistic latent structure, variational autoencoders (VAEs) may confront problems such as a mismatch between the posterior distribution of the latent and real data manifold, or discontinuity in the posterior distribution of the latent.   This leaves a back door for malicious attackers to collapse VAEs from the latent space, especially in scenarios where the encoder and decoder are used separately, such as communication and compressed sensing.   In this work, we provide the first study on the adversarial robustness of generative autoencoders in the latent space.   Specifically, we empirically demonstrate the latent vulnerability of popular generative autoencoders through attacks in the latent space.   We also evaluate the difference between variational autoencoders and their deterministic variants and observe that the latter performs better in latent robustness.   Meanwhile, we identify a potential trade-off between the adversarial robustness and the degree of the disentanglement of the latent codes.   Additionally, we also verify the feasibility of improvement for the latent robustness of VAEs through adversarial training.   In summary, we suggest concerning the adversarial latent robustness of the generative autoencoders, analyze several robustness-relative issues, and give some insights into a series of key challenges.

摘要: 产生式自动编码器，如变分自动编码器或对抗性自动编码器，已经在图像生成、信号通信等实际应用中取得了巨大的成功。然而，很少有人关注它们在实际部署过程中的健壮性。由于概率潜在结构，变分自动编码器可能会遇到潜在数据流形和真实数据流形的后验分布不匹配或后验分布不连续等问题。这为恶意攻击者从潜在空间崩溃VAE留下了后门，特别是在单独使用编码器和解码器的场景中，例如通信和压缩传感。在这项工作中，我们首次研究了生成式自动编码器在潜在空间中的对抗健壮性。具体地说，我们通过对潜在空间的攻击，经验地证明了流行的生成式自动编码器的潜在脆弱性。我们还评估了变分自动编码器和它们的确定性变体之间的差异，并观察到后者在潜在稳健性方面表现得更好。同时，我们确定了潜在代码的对抗健壮性和解缠程度之间的潜在权衡。此外，我们还验证了通过对抗性训练来提高VAE的潜在健壮性的可行性。综上所述，我们建议关注生成式自动编码器的对抗性潜在健壮性，分析了几个与健壮性相关的问题，并对一系列关键挑战给出了一些见解。



## **16. Boosting Adversarial Transferability via Fusing Logits of Top-1 Decomposed Feature**

融合Top-1分解特征的Logit提高对手的可转移性 cs.CV

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2305.01361v3) [paper-pdf](http://arxiv.org/pdf/2305.01361v3)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li, Zhun Zhong

**Abstract**: Recent research has shown that Deep Neural Networks (DNNs) are highly vulnerable to adversarial samples, which are highly transferable and can be used to attack other unknown black-box models. To improve the transferability of adversarial samples, several feature-based adversarial attack methods have been proposed to disrupt neuron activation in the middle layers. However, current state-of-the-art feature-based attack methods typically require additional computation costs for estimating the importance of neurons. To address this challenge, we propose a Singular Value Decomposition (SVD)-based feature-level attack method. Our approach is inspired by the discovery that eigenvectors associated with the larger singular values decomposed from the middle layer features exhibit superior generalization and attention properties. Specifically, we conduct the attack by retaining the decomposed Top-1 singular value-associated feature for computing the output logits, which are then combined with the original logits to optimize adversarial examples. Our extensive experimental results verify the effectiveness of our proposed method, which can be easily integrated into various baselines to significantly enhance the transferability of adversarial samples for disturbing normally trained CNNs and advanced defense strategies. The source code of this study is available at https://github.com/WJJLL/SVD-SSA

摘要: 最近的研究表明，深度神经网络非常容易受到敌意样本的攻击，这些样本具有很高的可传递性，可以用来攻击其他未知的黑盒模型。为了提高对抗性样本的可转移性，已经提出了几种基于特征的对抗性攻击方法来破坏中间层神经元的激活。然而，当前最先进的基于特征的攻击方法通常需要额外的计算成本来估计神经元的重要性。为了应对这一挑战，我们提出了一种基于奇异值分解(SVD)的特征级攻击方法。我们的方法是受到这样的发现的启发，即与从中间层特征分解的较大奇异值相关的特征向量具有更好的泛化和注意特性。具体地说，我们通过保留分解后的Top-1奇异值关联特征来计算输出逻辑，然后将其与原始逻辑相结合来优化对抗性实例，从而进行攻击。大量的实验结果验证了该方法的有效性，该方法可以很容易地集成到不同的基线中，显著提高对手样本干扰正常训练的CNN和高级防御策略的可转移性。这项研究的源代码可在https://github.com/WJJLL/SVD-SSA上获得



## **17. Adversarial Attacks on Image Classification Models: FGSM and Patch Attacks and their Impact**

对图像分类模型的敌意攻击：FGSM和Patch攻击及其影响 cs.CV

This is the preprint of the chapter titled "Adversarial Attacks on  Image Classification Models: FGSM and Patch Attacks and their Impact" which  will be published in the volume titled "Information Security and Privacy in  the Digital World - Some Selected Cases", edited by Jaydip Sen. The book will  be published by IntechOpen, London, UK, in 2023. This is not the final  version of the chapter

**SubmitDate**: 2023-07-05    [abs](http://arxiv.org/abs/2307.02055v1) [paper-pdf](http://arxiv.org/pdf/2307.02055v1)

**Authors**: Jaydip Sen, Subhasis Dasgupta

**Abstract**: This chapter introduces the concept of adversarial attacks on image classification models built on convolutional neural networks (CNN). CNNs are very popular deep-learning models which are used in image classification tasks. However, very powerful and pre-trained CNN models working very accurately on image datasets for image classification tasks may perform disastrously when the networks are under adversarial attacks. In this work, two very well-known adversarial attacks are discussed and their impact on the performance of image classifiers is analyzed. These two adversarial attacks are the fast gradient sign method (FGSM) and adversarial patch attack. These attacks are launched on three powerful pre-trained image classifier architectures, ResNet-34, GoogleNet, and DenseNet-161. The classification accuracy of the models in the absence and presence of the two attacks are computed on images from the publicly accessible ImageNet dataset. The results are analyzed to evaluate the impact of the attacks on the image classification task.

摘要: 本章介绍了基于卷积神经网络(CNN)的图像分类模型的对抗性攻击的概念。CNN是一种非常流行的深度学习模型，用于图像分类任务。然而，非常强大和预先训练的CNN模型在图像数据集上非常准确地工作以执行图像分类任务，当网络受到敌意攻击时，可能会灾难性地执行。在这项工作中，讨论了两个非常著名的对抗性攻击，并分析了它们对图像分类器性能的影响。这两种对抗性攻击是快速梯度符号方法(FGSM)和对抗性补丁攻击。这些攻击是在三个强大的预先训练的图像分类器架构上发起的，ResNet-34、GoogLeNet和DenseNet-161。对来自公众可访问的ImageNet数据集中的图像计算在两种攻击不存在的情况下模型的分类精度。分析结果以评估攻击对图像分类任务的影响。



## **18. Complex Graph Laplacian Regularizer for Inferencing Grid States**

用于网格状态推断的复图拉普拉斯正则化算法 eess.SP

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01906v1) [paper-pdf](http://arxiv.org/pdf/2307.01906v1)

**Authors**: Chinthaka Dinesh, Junfei Wang, Gene Cheung, Pirathayini Srikantha

**Abstract**: In order to maintain stable grid operations, system monitoring and control processes require the computation of grid states (e.g. voltage magnitude and angles) at high granularity. It is necessary to infer these grid states from measurements generated by a limited number of sensors like phasor measurement units (PMUs) that can be subjected to delays and losses due to channel artefacts, and/or adversarial attacks (e.g. denial of service, jamming, etc.). We propose a novel graph signal processing (GSP) based algorithm to interpolate states of the entire grid from observations of a small number of grid measurements. It is a two-stage process, where first an underlying Hermitian graph is learnt empirically from existing grid datasets. Then, the graph is used to interpolate missing grid signal samples in linear time. With our proposal, we can effectively reconstruct grid signals with significantly smaller number of observations when compared to existing traditional approaches (e.g. state estimation). In contrast to existing GSP approaches, we do not require knowledge of the underlying grid structure and parameters and are able to guarantee fast spectral optimization. We demonstrate the computational efficacy and accuracy of our proposal via practical studies conducted on the IEEE 118 bus system.

摘要: 为了维持稳定的电网运行，系统监测和控制过程需要计算高粒度的电网状态(如电压幅值和角度)。有必要从有限数量的传感器(如相量测量单元(PMU))生成的测量结果中推断这些网格状态，这些传感器可能由于信道伪影和/或对抗性攻击(例如拒绝服务、干扰等)而受到延迟和损失。我们提出了一种基于图信号处理(GSP)的新算法，该算法根据少量网格测量的观测值来内插整个网格的状态。这是一个分两个阶段的过程，首先从现有的网格数据集中经验地学习潜在的厄米特图。然后，使用该图在线性时间内对缺失的网格信号样本进行内插。与现有的传统方法(例如状态估计)相比，我们的建议可以用明显更少的观测值来有效地重建网格信号。与现有的GSP方法相比，我们不需要底层网格结构和参数的知识，并且能够保证快速的频谱优化。通过在IEEE118节点系统上进行的实际研究，证明了该方法的计算效率和准确性。



## **19. Physically Realizable Natural-Looking Clothing Textures Evade Person Detectors via 3D Modeling**

物理上可实现的自然外观服装纹理通过3D建模躲避人的检测 cs.CV

Accepted by CVPR 2023

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01778v1) [paper-pdf](http://arxiv.org/pdf/2307.01778v1)

**Authors**: Zhanhao Hu, Wenda Chu, Xiaopei Zhu, Hui Zhang, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works have proposed to craft adversarial clothes for evading person detectors, while they are either only effective at limited viewing angles or very conspicuous to humans. We aim to craft adversarial texture for clothes based on 3D modeling, an idea that has been used to craft rigid adversarial objects such as a 3D-printed turtle. Unlike rigid objects, humans and clothes are non-rigid, leading to difficulties in physical realization. In order to craft natural-looking adversarial clothes that can evade person detectors at multiple viewing angles, we propose adversarial camouflage textures (AdvCaT) that resemble one kind of the typical textures of daily clothes, camouflage textures. We leverage the Voronoi diagram and Gumbel-softmax trick to parameterize the camouflage textures and optimize the parameters via 3D modeling. Moreover, we propose an efficient augmentation pipeline on 3D meshes combining topologically plausible projection (TopoProj) and Thin Plate Spline (TPS) to narrow the gap between digital and real-world objects. We printed the developed 3D texture pieces on fabric materials and tailored them into T-shirts and trousers. Experiments show high attack success rates of these clothes against multiple detectors.

摘要: 最近的研究提出了为躲避人体探测器而制作对抗服装，而这些服装要么只在有限的视角下有效，要么对人类来说非常显眼。我们的目标是基于3D建模为衣服制作对抗性纹理，这一想法已被用于制作刚性对抗性对象，如3D打印的乌龟。与刚性物体不同，人和衣服是非刚性的，这导致了物理实现的困难。为了制作出看起来自然的、能够在多个视角下躲避人体探测的对抗性服装，我们提出了一种类似于日常服装的典型纹理--伪装纹理的对抗性伪装纹理(AdvCaT)。我们利用Voronoi图和Gumbel-Softmax技巧对伪装纹理进行参数化，并通过3D建模优化参数。此外，我们还提出了一种结合拓扑似然投影(TOPO Proj)和薄板样条线(TPS)的三维网格增强流水线，以缩小数字对象和真实对象之间的差距。我们将开发的3D纹理块打印在面料上，并将它们裁剪成T恤和裤子。实验表明，这些衣服对多个探测器的攻击成功率很高。



## **20. vWitness: Certifying Web Page Interactions with Computer Vision**

VWitness：使用计算机视觉验证网页交互 cs.CR

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2007.15805v2) [paper-pdf](http://arxiv.org/pdf/2007.15805v2)

**Authors**: He Shuang, Lianying Zhao, David Lie

**Abstract**: Web servers service client requests, some of which might cause the web server to perform security-sensitive operations (e.g. money transfer, voting). An attacker may thus forge or maliciously manipulate such requests by compromising a web client. Unfortunately, a web server has no way of knowing whether the client from which it receives a request has been compromised or not -- current "best practice" defenses such as user authentication or network encryption cannot aid a server as they all assume web client integrity. To address this shortcoming, we propose vWitness, which "witnesses" the interactions of a user with a web page and certifies whether they match a specification provided by the web server, enabling the web server to know that the web request is user-intended. The main challenge that vWitness overcomes is that even benign clients introduce unpredictable variations in the way they render web pages. vWitness differentiates between these benign variations and malicious manipulation using computer vision, allowing it to certify to the web server that 1) the web page user interface is properly displayed 2) observed user interactions are used to construct the web request. Our vWitness prototype achieves compatibility with modern web pages, is resilient to adversarial example attacks and is accurate and performant -- vWitness achieves 99.97% accuracy and adds 197ms of overhead to the entire interaction session in the average case.

摘要: Web服务器为客户端请求提供服务，其中一些请求可能会导致Web服务器执行安全敏感操作(例如，转账、投票)。因此，攻击者可以通过危害Web客户端来伪造或恶意操纵此类请求。不幸的是，Web服务器无法知道它从其接收请求的客户端是否已被破坏--当前的“最佳实践”防御，如用户身份验证或网络加密，无法帮助服务器，因为它们都假定Web客户端的完整性。为了解决这一缺点，我们提出了vWitness，它“见证”用户与网页的交互，并验证它们是否符合Web服务器提供的规范，使Web服务器知道Web请求是用户预期的。VWitness克服的主要挑战是，即使是良性的客户端，也会在呈现网页的方式上引入不可预测的变化。VWitness使用计算机视觉区分这些良性变化和恶意操纵，允许它向Web服务器证明1)网页用户界面被正确显示2)使用观察到的用户交互来构造Web请求。我们的vWitness原型实现了与现代网页的兼容性，对敌意示例攻击具有弹性，并且是准确和高性能的--vWitness达到99.97%的准确率，在平均情况下，整个交互会话增加了197ms的开销。



## **21. Interpretable Computer Vision Models through Adversarial Training: Unveiling the Robustness-Interpretability Connection**

对抗性训练中的可解释计算机视觉模型：揭示稳健性与可解释性之间的联系 cs.CV

13 pages, 19 figures, 6 tables

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.02500v1) [paper-pdf](http://arxiv.org/pdf/2307.02500v1)

**Authors**: Delyan Boychev

**Abstract**: With the perpetual increase of complexity of the state-of-the-art deep neural networks, it becomes a more and more challenging task to maintain their interpretability. Our work aims to evaluate the effects of adversarial training utilized to produce robust models - less vulnerable to adversarial attacks. It has been shown to make computer vision models more interpretable. Interpretability is as essential as robustness when we deploy the models to the real world. To prove the correlation between these two problems, we extensively examine the models using local feature-importance methods (SHAP, Integrated Gradients) and feature visualization techniques (Representation Inversion, Class Specific Image Generation). Standard models, compared to robust are more susceptible to adversarial attacks, and their learned representations are less meaningful to humans. Conversely, these models focus on distinctive regions of the images which support their predictions. Moreover, the features learned by the robust model are closer to the real ones.

摘要: 随着最新的深度神经网络的复杂性不断增加，保持其可解释性成为一项越来越具有挑战性的任务。我们的工作旨在评估用于产生健壮模型的对抗性训练的效果--较不容易受到对抗性攻击。它已被证明使计算机视觉模型更易于解释。当我们将模型部署到真实世界时，可解释性与健壮性同样重要。为了证明这两个问题之间的相关性，我们使用局部特征重要性方法(Shap，集成梯度)和特征可视化技术(表示反转，类特定图像生成)对模型进行了广泛的检验。与稳健模型相比，标准模型更容易受到对抗性攻击，其学习的表示对人类的意义较小。相反，这些模型关注的是图像中支持其预测的独特区域。此外，稳健模型所学习的特征更接近真实的特征。



## **22. LEAT: Towards Robust Deepfake Disruption in Real-World Scenarios via Latent Ensemble Attack**

Leat：通过潜在的集成攻击在现实世界场景中实现稳健的深伪破坏 cs.CV

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01520v1) [paper-pdf](http://arxiv.org/pdf/2307.01520v1)

**Authors**: Joonkyo Shim, Hyunsoo Yoon

**Abstract**: Deepfakes, malicious visual contents created by generative models, pose an increasingly harmful threat to society. To proactively mitigate deepfake damages, recent studies have employed adversarial perturbation to disrupt deepfake model outputs. However, previous approaches primarily focus on generating distorted outputs based on only predetermined target attributes, leading to a lack of robustness in real-world scenarios where target attributes are unknown. Additionally, the transferability of perturbations between two prominent generative models, Generative Adversarial Networks (GANs) and Diffusion Models, remains unexplored. In this paper, we emphasize the importance of target attribute-transferability and model-transferability for achieving robust deepfake disruption. To address this challenge, we propose a simple yet effective disruption method called Latent Ensemble ATtack (LEAT), which attacks the independent latent encoding process. By disrupting the latent encoding process, it generates distorted output images in subsequent generation processes, regardless of the given target attributes. This target attribute-agnostic attack ensures robust disruption even when the target attributes are unknown. Additionally, we introduce a Normalized Gradient Ensemble strategy that effectively aggregates gradients for iterative gradient attacks, enabling simultaneous attacks on various types of deepfake models, involving both GAN-based and Diffusion-based models. Moreover, we demonstrate the insufficiency of evaluating disruption quality solely based on pixel-level differences. As a result, we propose an alternative protocol for comprehensively evaluating the success of defense. Extensive experiments confirm the efficacy of our method in disrupting deepfakes in real-world scenarios, reporting a higher defense success rate compared to previous methods.

摘要: Deepfake是由生成性模型创造的恶意视觉内容，对社会构成越来越有害的威胁。为了主动减轻深度伪模型的损害，最近的研究采用对抗性扰动来干扰深度伪模型的输出。然而，以前的方法主要集中于仅基于预定的目标属性来生成失真的输出，导致在目标属性未知的现实世界场景中缺乏稳健性。此外，两个重要的生成性模型--生成性对抗网络(GANS)和扩散模型--之间的扰动的可转移性仍未被探索。在本文中，我们强调了目标属性可转移性和模型可转移性对于实现稳健的深度伪干扰的重要性。为了应对这一挑战，我们提出了一种简单而有效的破坏方法，称为潜在集成攻击(Leat)，该方法攻击独立的潜在编码过程。通过中断潜在编码过程，它在随后的生成过程中生成失真的输出图像，而与给定的目标属性无关。这种与目标属性无关的攻击确保了即使在目标属性未知的情况下也能进行稳健的破坏。此外，我们引入了归一化梯度集成策略，该策略有效地聚合了用于迭代梯度攻击的梯度，使得能够同时攻击各种类型的深度伪模型，包括基于GaN的模型和基于扩散的模型。此外，我们还证明了仅基于像素级差异来评估干扰质量的不足。因此，我们提出了一种全面评估防御成功的替代方案。广泛的实验证实了我们的方法在现实世界场景中破坏深度假冒的有效性，报告了比以前的方法更高的防御成功率。



## **23. SCAT: Robust Self-supervised Contrastive Learning via Adversarial Training for Text Classification**

SCAT：基于对抗性训练的文本分类稳健自监督对比学习 cs.CL

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01488v1) [paper-pdf](http://arxiv.org/pdf/2307.01488v1)

**Authors**: Junjie Wu, Dit-Yan Yeung

**Abstract**: Despite their promising performance across various natural language processing (NLP) tasks, current NLP systems are vulnerable to textual adversarial attacks. To defend against these attacks, most existing methods apply adversarial training by incorporating adversarial examples. However, these methods have to rely on ground-truth labels to generate adversarial examples, rendering it impractical for large-scale model pre-training which is commonly used nowadays for NLP and many other tasks. In this paper, we propose a novel learning framework called SCAT (Self-supervised Contrastive Learning via Adversarial Training), which can learn robust representations without requiring labeled data. Specifically, SCAT modifies random augmentations of the data in a fully labelfree manner to generate adversarial examples. Adversarial training is achieved by minimizing the contrastive loss between the augmentations and their adversarial counterparts. We evaluate SCAT on two text classification datasets using two state-of-the-art attack schemes proposed recently. Our results show that SCAT can not only train robust language models from scratch, but it can also significantly improve the robustness of existing pre-trained language models. Moreover, to demonstrate its flexibility, we show that SCAT can also be combined with supervised adversarial training to further enhance model robustness.

摘要: 尽管它们在各种自然语言处理(NLP)任务中具有良好的性能，但当前的自然语言处理系统容易受到文本对手攻击。为了防御这些攻击，现有的大多数方法都是通过结合对抗性例子进行对抗性训练。然而，这些方法必须依赖地面事实标签来生成对抗性实例，这使得目前用于自然语言处理和许多其他任务的大规模模型预训练是不现实的。在本文中，我们提出了一种新的学习框架，称为SCAT(Self-Supervised Contrastive Learning via Aversative Trading)，它可以在不需要标记数据的情况下学习稳健的表示。具体地说，SCAT以完全无标签的方式修改数据的随机增加，以生成对抗性示例。对抗性训练是通过最小化增强器和对抗器之间的对比损失来实现的。我们使用最近提出的两种最先进的攻击方案在两个文本分类数据集上对SCAT进行了评估。结果表明，SCAT不仅可以从头开始训练健壮的语言模型，而且可以显著提高已有的预先训练的语言模型的健壮性。此外，为了展示其灵活性，我们还展示了SCAT还可以与有监督的对抗性训练相结合，以进一步增强模型的稳健性。



## **24. Web3Recommend: Decentralised recommendations with trust and relevance**

Web3Recommend：具有信任和相关性的分散推荐 cs.DC

**SubmitDate**: 2023-07-04    [abs](http://arxiv.org/abs/2307.01411v1) [paper-pdf](http://arxiv.org/pdf/2307.01411v1)

**Authors**: Rohan Madhwal, Johan Pouwelse

**Abstract**: Web3Recommend is a decentralized Social Recommender System implementation that enables Web3 Platforms on Android to generate recommendations that balance trust and relevance. Generating recommendations in decentralized networks is a non-trivial problem because these networks lack a global perspective due to the absence of a central authority. Further, decentralized networks are prone to Sybil Attacks in which a single malicious user can generate multiple fake or Sybil identities. Web3Recommend relies on a novel graph-based content recommendation design inspired by GraphJet, a recommendation system used in Twitter enhanced with MeritRank, a decentralized reputation scheme that provides Sybil-resistance to the system. By adding MeritRank's decay parameters to the vanilla Social Recommender Systems' personalized SALSA graph algorithm, we can provide theoretical guarantees against Sybil Attacks in the generated recommendations. Similar to GraphJet, we focus on generating real-time recommendations by only acting on recent interactions in the social network, allowing us to cater temporally contextual recommendations while keeping a tight bound on the memory usage in resource-constrained devices, allowing for a seamless user experience. As a proof-of-concept, we integrate our system with MusicDAO, an open-source Web3 music-sharing platform, to generate personalized, real-time recommendations. Thus, we provide the first Sybil-resistant Social Recommender System, allowing real-time recommendations beyond classic user-based collaborative filtering. The system is also rigorously tested with extensive unit and integration tests. Further, our experiments demonstrate the trust-relevance balance of recommendations against multiple adversarial strategies in a test network generated using data from real music platforms.

摘要: Web3Recommend是一个去中心化的社交推荐系统实现，它使Android上的Web3平台能够生成平衡信任和相关性的推荐。在分散的网络中生成建议是一个不小的问题，因为这些网络由于没有中央当局而缺乏全球视角。此外，分散的网络容易受到Sybil攻击，在这种攻击中，单个恶意用户可以生成多个虚假或Sybil身份。Web3Recommend依赖于一种新颖的基于图形的内容推荐设计，灵感来自GraphJet，这是一种在Twitter中使用的推荐系统，使用了MeritRank增强的推荐系统，MeritRank是一种去中心化的声誉方案，为系统提供Sybil抵抗。通过将MeritRank的衰减参数添加到Vanilla Social推荐系统的个性化Salsa图算法中，我们可以在生成的推荐中提供抵抗Sybil攻击的理论保证。与GraphJet类似，我们专注于通过只对社交网络中最近的互动采取行动来生成实时推荐，从而允许我们在严格控制资源受限设备的内存使用的同时，满足临时上下文推荐，从而实现无缝的用户体验。作为概念验证，我们将我们的系统与开源Web3音乐共享平台MusicDAO集成，以生成个性化的实时推荐。因此，我们提供了第一个抗Sybil的社交推荐系统，允许在传统的基于用户的协作过滤之外提供实时推荐。该系统还通过广泛的单元和集成测试进行了严格测试。此外，在使用真实音乐平台的数据生成的测试网络中，我们的实验证明了推荐与多种对抗策略之间的信任-相关性平衡。



## **25. Adversarial Learning in Real-World Fraud Detection: Challenges and Perspectives**

现实世界欺诈检测中的对抗性学习：挑战和前景 cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01390v1) [paper-pdf](http://arxiv.org/pdf/2307.01390v1)

**Authors**: Danele Lunghi, Alkis Simitsis, Olivier Caelen, Gianluca Bontempi

**Abstract**: Data economy relies on data-driven systems and complex machine learning applications are fueled by them. Unfortunately, however, machine learning models are exposed to fraudulent activities and adversarial attacks, which threaten their security and trustworthiness. In the last decade or so, the research interest on adversarial machine learning has grown significantly, revealing how learning applications could be severely impacted by effective attacks. Although early results of adversarial machine learning indicate the huge potential of the approach to specific domains such as image processing, still there is a gap in both the research literature and practice regarding how to generalize adversarial techniques in other domains and applications. Fraud detection is a critical defense mechanism for data economy, as it is for other applications as well, which poses several challenges for machine learning. In this work, we describe how attacks against fraud detection systems differ from other applications of adversarial machine learning, and propose a number of interesting directions to bridge this gap.

摘要: 数据经济依赖于数据驱动系统，而复杂的机器学习应用正是由它们推动的。然而，不幸的是，机器学习模型暴露在欺诈性活动和对抗性攻击中，这威胁到它们的安全性和可信性。在过去的十年左右，对抗性机器学习的研究兴趣显著增长，揭示了学习应用如何受到有效攻击的严重影响。尽管对抗性机器学习的早期结果表明该方法在图像处理等特定领域具有巨大的潜力，但在如何将对抗性技术推广到其他领域和应用方面，研究文献和实践中仍存在差距。欺诈检测是数据经济的关键防御机制，对于其他应用也是如此，这给机器学习带来了一些挑战。在这项工作中，我们描述了针对欺诈检测系统的攻击与对抗性机器学习的其他应用程序的不同之处，并提出了一些有趣的方向来弥合这一差距。



## **26. When Can Linear Learners be Robust to Indiscriminate Poisoning Attacks?**

线性学习者何时才能对不分青红皂白的中毒攻击表现得很健壮？ cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01073v1) [paper-pdf](http://arxiv.org/pdf/2307.01073v1)

**Authors**: Fnu Suya, Xiao Zhang, Yuan Tian, David Evans

**Abstract**: We study indiscriminate poisoning for linear learners where an adversary injects a few crafted examples into the training data with the goal of forcing the induced model to incur higher test error. Inspired by the observation that linear learners on some datasets are able to resist the best known attacks even without any defenses, we further investigate whether datasets can be inherently robust to indiscriminate poisoning attacks for linear learners. For theoretical Gaussian distributions, we rigorously characterize the behavior of an optimal poisoning attack, defined as the poisoning strategy that attains the maximum risk of the induced model at a given poisoning budget. Our results prove that linear learners can indeed be robust to indiscriminate poisoning if the class-wise data distributions are well-separated with low variance and the size of the constraint set containing all permissible poisoning points is also small. These findings largely explain the drastic variation in empirical attack performance of the state-of-the-art poisoning attacks on linear learners across benchmark datasets, making an important initial step towards understanding the underlying reasons some learning tasks are vulnerable to data poisoning attacks.

摘要: 我们研究了线性学习者的不分青红皂白的中毒，其中对手向训练数据中注入一些精心制作的示例，目的是迫使诱导模型招致更高的测试错误。受一些数据集上的线性学习者即使在没有任何防御的情况下也能够抵抗最著名的攻击的观察的启发，我们进一步调查了数据集对于线性学习者的不分青红皂白的中毒攻击是否具有内在的健壮性。对于理论上的高斯分布，我们严格地刻画了最优中毒攻击的行为，定义为在给定的中毒预算下，达到诱导模型的最大风险的中毒策略。我们的结果证明，如果类的数据分布是分开的且方差很小，并且包含所有允许毒点的约束集的大小也很小，那么线性学习器确实可以对不分青红皂白的中毒具有健壮性。这些发现在很大程度上解释了在基准数据集上针对线性学习者的最新中毒攻击在经验攻击性能上的巨大差异，为理解一些学习任务容易受到数据中毒攻击的根本原因迈出了重要的第一步。



## **27. Enhancing the Robustness of QMIX against State-adversarial Attacks**

增强QMIX对状态对抗攻击的健壮性 cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.00907v1) [paper-pdf](http://arxiv.org/pdf/2307.00907v1)

**Authors**: Weiran Guo, Guanjun Liu, Ziyuan Zhou, Ling Wang, Jiacun Wang

**Abstract**: Deep reinforcement learning (DRL) performance is generally impacted by state-adversarial attacks, a perturbation applied to an agent's observation. Most recent research has concentrated on robust single-agent reinforcement learning (SARL) algorithms against state-adversarial attacks. Still, there has yet to be much work on robust multi-agent reinforcement learning. Using QMIX, one of the popular cooperative multi-agent reinforcement algorithms, as an example, we discuss four techniques to improve the robustness of SARL algorithms and extend them to multi-agent scenarios. To increase the robustness of multi-agent reinforcement learning (MARL) algorithms, we train models using a variety of attacks in this research. We then test the models taught using the other attacks by subjecting them to the corresponding attacks throughout the training phase. In this way, we organize and summarize techniques for enhancing robustness when used with MARL.

摘要: 深度强化学习(DRL)的性能通常受到状态对抗攻击的影响，状态对抗攻击是一种应用于代理观察的扰动。最近的研究集中在针对状态对抗攻击的健壮单智能体强化学习(SARL)算法。然而，在稳健的多智能体强化学习方面还没有太多的工作。以目前流行的协作多智能体增强算法QMIX为例，讨论了四种提高SARL算法健壮性的技术，并将其扩展到多智能体场景。为了提高多智能体强化学习(MAIL)算法的健壮性，在本研究中，我们使用各种攻击来训练模型。然后，我们测试使用其他攻击教授的模型，方法是在整个培训阶段对它们进行相应的攻击。通过这种方式，我们组织和总结了在与Marl一起使用时增强健壮性的技术。



## **28. Data Poisoning Attack Aiming the Vulnerability of Continual Learning**

针对持续学习漏洞的数据中毒攻击 cs.LG

ICIP 2023 (NeurIPS 2022 ML Safety Workshop accepted paper)

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2211.15875v2) [paper-pdf](http://arxiv.org/pdf/2211.15875v2)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world constraints related to memory and privacy. However, this introduces a problem in these models by not being able to track the performance on each task. In essence, current continual learning methods are susceptible to attacks on previous tasks. We demonstrate the vulnerability of regularization-based continual learning methods by presenting a simple task-specific data poisoning attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. We experiment with the attack on the two representative regularization-based continual learning methods, Elastic Weight Consolidation (EWC) and Synaptic Intelligence (SI), trained with variants of MNIST dataset. The experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attacks.

摘要: 一般而言，基于正则化的持续学习模型限制了对先前任务数据的访问，以模仿现实世界中与记忆和隐私相关的约束。然而，这在这些模型中引入了一个问题，因为无法跟踪每个任务的性能。从本质上讲，当前的持续学习方法很容易受到对以前任务的攻击。我们通过一个简单的任务特定的数据中毒攻击来证明基于正则化的持续学习方法的脆弱性，该攻击可以用于新任务的学习过程。建议的攻击生成的训练数据会导致攻击者针对的特定任务的性能下降。我们对两种典型的基于正则化的连续学习方法--弹性权值合并(EWC)和突触智能(SI)--进行了实验，并用MNIST数据集的变体进行了训练。实验结果验证了本文提出的脆弱性，并证明了开发对对手攻击具有健壮性的持续学习模型的重要性。



## **29. Evaluating the Adversarial Robustness of Convolution-based Human Motion Prediction**

基于卷积的人体运动预测的对抗稳健性评价 cs.CV

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2306.11990v2) [paper-pdf](http://arxiv.org/pdf/2306.11990v2)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of CNNs, which facilitates human-machine cooperation. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks, which may cause danger in real applications. The adversarial attack will face two problems against human motion prediction: 1. For naturalness, pose data is highly related to the physical dynamics of human skeletons where Lp norm constraints cannot constrain the adversarial example well; 2. Unlike the pixel value in images, pose data is diverse at scale because of the different acquisition equipment and the data processing, which makes it hard to set fixed parameters to perform attacks. To solve the problems above, we propose a new adversarial attack method that perturbs the input human motion sequence by maximizing the prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the imperceptibility of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models can be easily disturbed under the proposed attack. The quantitative analysis shows that prior knowledge and semantic information modeling can be the key to the adversarial robustness of human motion predictors. The qualitative results indicate that the adversarial sample is hard to be noticed when compared frame by frame but is relatively easy to be detected when the sample is animated.

摘要: 在人工神经网络的帮助下，人体运动预测取得了很好的效果，有利于人机协作。然而，目前还没有对人体运动预测中的潜在风险进行评估的工作，这在实际应用中可能会造成危险。对抗性攻击将面临两个针对人体运动预测的问题：1.对于自然度，姿势数据与人体骨骼的物理动力学高度相关，其中Lp范数约束不能很好地约束对抗性示例；2.与图像中的像素值不同，由于采集设备和数据处理的不同，姿势数据在尺度上是多样的，这使得很难设置固定的参数来执行攻击。为了解决上述问题，我们提出了一种新的对抗性攻击方法，该方法通过在物理约束下最大化预测误差来扰动输入的人体运动序列。具体地说，我们引入了一种新的自适应方案来促进攻击以适应目标姿态的规模和两个物理约束来增强对抗性例子的不可见性。在三个数据集上的评估实验表明，所有目标模型的预测误差都显著增大，这意味着现有的基于卷积的人体运动预测模型在所提出的攻击下很容易受到干扰。定量分析表明，先验知识和语义信息建模是人体运动预测器对抗性健壮性的关键。定性结果表明，对抗性样本在逐帧比较时很难被注意到，但在样本被动画时相对容易被检测到。



## **30. Sneaky Spikes: Uncovering Stealthy Backdoor Attacks in Spiking Neural Networks with Neuromorphic Data**

偷偷摸摸的尖峰：用神经形态数据揭示尖峰神经网络中的秘密后门攻击 cs.CR

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2302.06279v2) [paper-pdf](http://arxiv.org/pdf/2302.06279v2)

**Authors**: Gorka Abad, Oguzhan Ersoy, Stjepan Picek, Aitor Urbieta

**Abstract**: Deep neural networks (DNNs) have demonstrated remarkable performance across various tasks, including image and speech recognition. However, maximizing the effectiveness of DNNs requires meticulous optimization of numerous hyperparameters and network parameters through training. Moreover, high-performance DNNs entail many parameters, which consume significant energy during training. In order to overcome these challenges, researchers have turned to spiking neural networks (SNNs), which offer enhanced energy efficiency and biologically plausible data processing capabilities, rendering them highly suitable for sensory data tasks, particularly in neuromorphic data. Despite their advantages, SNNs, like DNNs, are susceptible to various threats, including adversarial examples and backdoor attacks. Yet, the field of SNNs still needs to be explored in terms of understanding and countering these attacks.   This paper delves into backdoor attacks in SNNs using neuromorphic datasets and diverse triggers. Specifically, we explore backdoor triggers within neuromorphic data that can manipulate their position and color, providing a broader scope of possibilities than conventional triggers in domains like images. We present various attack strategies, achieving an attack success rate of up to 100\% while maintaining a negligible impact on clean accuracy. Furthermore, we assess these attacks' stealthiness, revealing that our most potent attacks possess significant stealth capabilities. Lastly, we adapt several state-of-the-art defenses from the image domain, evaluating their efficacy on neuromorphic data and uncovering instances where they fall short, leading to compromised performance.

摘要: 深度神经网络(DNN)在包括图像和语音识别在内的各种任务中表现出了显著的性能。然而，要最大限度地发挥DNN的有效性，需要通过训练对大量的超参数和网络参数进行细致的优化。此外，高性能的DNN需要许多参数，这些参数在训练过程中消耗大量的能量。为了克服这些挑战，研究人员转向尖峰神经网络(SNN)，它提供了更高的能量效率和生物上可信的数据处理能力，使其非常适合于感觉数据任务，特别是在神经形态数据中。尽管SNN具有优势，但与DNN一样，SNN也容易受到各种威胁，包括敌意示例和后门攻击。然而，在理解和对抗这些攻击方面，SNN领域仍然需要探索。本文使用神经形态数据集和不同的触发器来深入研究SNN中的后门攻击。具体地说，我们探索神经形态数据中的后门触发器，这些数据可以操纵它们的位置和颜色，提供比图像等领域的传统触发器更广泛的可能性。我们提出了不同的攻击策略，实现了高达100%的攻击成功率，同时保持了对干净准确性的微小影响。此外，我们评估了这些攻击的隐蔽性，揭示了我们最强大的攻击具有显著的隐形能力。最后，我们从图像领域采用了几种最先进的防御措施，评估了它们在神经形态数据上的有效性，并发现了它们不足的地方，导致了性能下降。



## **31. Feature Partition Aggregation: A Fast Certified Defense Against a Union of $\ell_0$ Attacks**

功能分区聚合：针对联合$\ELL_0$攻击的快速认证防御 cs.LG

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2302.11628v2) [paper-pdf](http://arxiv.org/pdf/2302.11628v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Sparse or $\ell_0$ adversarial attacks arbitrarily perturb an unknown subset of the features. $\ell_0$ robustness analysis is particularly well-suited for heterogeneous (tabular) data where features have different types or scales. State-of-the-art $\ell_0$ certified defenses are based on randomized smoothing and apply to evasion attacks only. This paper proposes feature partition aggregation (FPA) -- a certified defense against the union of $\ell_0$ evasion, backdoor, and poisoning attacks. FPA generates its stronger robustness guarantees via an ensemble whose submodels are trained on disjoint feature sets. Compared to state-of-the-art $\ell_0$ defenses, FPA is up to 3,000${\times}$ faster and provides larger median robustness guarantees (e.g., median certificates of 13 pixels over 10 for CIFAR10, 12 pixels over 10 for MNIST, 4 features over 1 for Weather, and 3 features over 1 for Ames), meaning FPA provides the additional dimensions of robustness essentially for free.

摘要: 稀疏或$\ELL_0$对抗性攻击任意扰乱未知特征的子集。$\ELL_0$稳健性分析特别适合于要素具有不同类型或比例的异类(表格)数据。最先进的$\ELL_0$认证防御基于随机平滑，仅适用于躲避攻击。本文提出了特征分区聚合(FPA)--一种针对$\ELL_0$逃避、后门和中毒攻击的联合认证防御。FPA通过在不相交的特征集上训练子模型的集成来产生更强的稳健性保证。与最先进的$\ELL_0$防御相比，FPA的速度快达3,000美元，并提供更大的中位数稳健性保证(例如，CIFAR10的中位数证书为13像素超过10，MNIST的中位数证书为10以上的12像素，天气的中位数证书为4，Ames的中位数证书为1以上)，这意味着FPA基本上免费提供了额外的稳健性维度。



## **32. Interpretability and Transparency-Driven Detection and Transformation of Textual Adversarial Examples (IT-DT)**

可解释性和透明度驱动的文本对抗性实例的检测和转换(IT-DT) cs.CL

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.01225v1) [paper-pdf](http://arxiv.org/pdf/2307.01225v1)

**Authors**: Bushra Sabir, M. Ali Babar, Sharif Abuadbba

**Abstract**: Transformer-based text classifiers like BERT, Roberta, T5, and GPT-3 have shown impressive performance in NLP. However, their vulnerability to adversarial examples poses a security risk. Existing defense methods lack interpretability, making it hard to understand adversarial classifications and identify model vulnerabilities. To address this, we propose the Interpretability and Transparency-Driven Detection and Transformation (IT-DT) framework. It focuses on interpretability and transparency in detecting and transforming textual adversarial examples. IT-DT utilizes techniques like attention maps, integrated gradients, and model feedback for interpretability during detection. This helps identify salient features and perturbed words contributing to adversarial classifications. In the transformation phase, IT-DT uses pre-trained embeddings and model feedback to generate optimal replacements for perturbed words. By finding suitable substitutions, we aim to convert adversarial examples into non-adversarial counterparts that align with the model's intended behavior while preserving the text's meaning. Transparency is emphasized through human expert involvement. Experts review and provide feedback on detection and transformation results, enhancing decision-making, especially in complex scenarios. The framework generates insights and threat intelligence empowering analysts to identify vulnerabilities and improve model robustness. Comprehensive experiments demonstrate the effectiveness of IT-DT in detecting and transforming adversarial examples. The approach enhances interpretability, provides transparency, and enables accurate identification and successful transformation of adversarial inputs. By combining technical analysis and human expertise, IT-DT significantly improves the resilience and trustworthiness of transformer-based text classifiers against adversarial attacks.

摘要: 基于转换器的文本分类器，如Bert、Roberta、T5和GPT-3，在NLP中表现出了令人印象深刻的性能。然而，它们在敌意例子面前的脆弱性构成了安全风险。现有的防御方法缺乏可解释性，难以理解对抗性分类和识别模型漏洞。为了解决这一问题，我们提出了可解释性和透明度驱动的检测和转换(IT-DT)框架。它侧重于检测和转换文本对抗性例子的可解释性和透明度。IT-DT利用注意图、集成梯度和模型反馈等技术在检测过程中实现可解释性。这有助于识别有助于对抗性分类的显著特征和受干扰的词语。在转换阶段，IT-DT使用预先训练的嵌入和模型反馈来生成扰动单词的最优替换。通过找到合适的替换，我们的目标是将对抗性例子转换为非对抗性例子，使之与模型的预期行为保持一致，同时保持文本的意义。通过人类专家的参与来强调透明度。专家对检测和转换结果进行审查并提供反馈，从而提高决策能力，尤其是在复杂情况下。该框架生成洞察力和威胁情报，使分析人员能够识别漏洞并提高模型的稳健性。综合实验证明了IT-DT在检测和转换敌意实例方面的有效性。该方法提高了可解释性，提供了透明度，并能够准确识别和成功转换对抗性输入。通过将技术分析和人类专业知识相结合，IT-DT显著提高了基于转换器的文本分类器对对手攻击的弹性和可信度。



## **33. From ChatGPT to ThreatGPT: Impact of Generative AI in Cybersecurity and Privacy**

从ChatGPT到ThreatGPT：生成性人工智能对网络安全和隐私的影响 cs.CR

**SubmitDate**: 2023-07-03    [abs](http://arxiv.org/abs/2307.00691v1) [paper-pdf](http://arxiv.org/pdf/2307.00691v1)

**Authors**: Maanak Gupta, CharanKumar Akiri, Kshitiz Aryal, Eli Parker, Lopamudra Praharaj

**Abstract**: Undoubtedly, the evolution of Generative AI (GenAI) models has been the highlight of digital transformation in the year 2022. As the different GenAI models like ChatGPT and Google Bard continue to foster their complexity and capability, it's critical to understand its consequences from a cybersecurity perspective. Several instances recently have demonstrated the use of GenAI tools in both the defensive and offensive side of cybersecurity, and focusing on the social, ethical and privacy implications this technology possesses. This research paper highlights the limitations, challenges, potential risks, and opportunities of GenAI in the domain of cybersecurity and privacy. The work presents the vulnerabilities of ChatGPT, which can be exploited by malicious users to exfiltrate malicious information bypassing the ethical constraints on the model. This paper demonstrates successful example attacks like Jailbreaks, reverse psychology, and prompt injection attacks on the ChatGPT. The paper also investigates how cyber offenders can use the GenAI tools in developing cyber attacks, and explore the scenarios where ChatGPT can be used by adversaries to create social engineering attacks, phishing attacks, automated hacking, attack payload generation, malware creation, and polymorphic malware. This paper then examines defense techniques and uses GenAI tools to improve security measures, including cyber defense automation, reporting, threat intelligence, secure code generation and detection, attack identification, developing ethical guidelines, incidence response plans, and malware detection. We will also discuss the social, legal, and ethical implications of ChatGPT. In conclusion, the paper highlights open challenges and future directions to make this GenAI secure, safe, trustworthy, and ethical as the community understands its cybersecurity impacts.

摘要: 毫无疑问，产生式AI(GenAI)模式的演变一直是2022年数字化转型的亮点。随着ChatGPT和Google Bard等不同的GenAI模型继续培养其复杂性和能力，从网络安全的角度理解其后果至关重要。最近的几个例子表明，GenAI工具在网络安全的防御和进攻方面都有使用，并侧重于这项技术所具有的社会、伦理和隐私影响。这份研究报告强调了GenAI在网络安全和隐私领域的限制、挑战、潜在风险和机遇。这项工作揭示了ChatGPT的漏洞，恶意用户可以利用该漏洞绕过模型上的道德约束渗出恶意信息。本文演示了对ChatGPT的越狱、反向心理和快速注入攻击等成功的示例攻击。本文还研究了网络攻击者如何使用GenAI工具开发网络攻击，并探索了攻击者可以使用ChatGPT来创建社会工程攻击、网络钓鱼攻击、自动黑客攻击、攻击有效负载生成、恶意软件创建和多态恶意软件的场景。然后，本文研究了防御技术，并使用GenAI工具来改进安全措施，包括网络防御自动化、报告、威胁情报、安全代码生成和检测、攻击识别、制定道德准则、事件响应计划和恶意软件检测。我们还将讨论ChatGPT的社会、法律和伦理影响。最后，白皮书强调了开放的挑战和未来的方向，以使这一GenAI安全、安全、值得信赖和道德，因为社区了解其网络安全影响。



## **34. Soft Actor-Critic Algorithm with Truly-satisfied Inequality Constraint**

具有真正满足的不等式约束的软执行者-批评者算法 cs.LG

10 pages, 9 figures

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2303.04356v2) [paper-pdf](http://arxiv.org/pdf/2303.04356v2)

**Authors**: Taisuke Kobayashi

**Abstract**: Soft actor-critic (SAC) in reinforcement learning is expected to be one of the next-generation robot control schemes. Its ability to maximize policy entropy would make a robotic controller robust to noise and perturbation, which is useful for real-world robot applications. However, the priority of maximizing the policy entropy is automatically tuned in the current implementation, the rule of which can be interpreted as one for equality constraint, binding the policy entropy into its specified lower bound. The current SAC is therefore no longer maximize the policy entropy, contrary to our expectation. To resolve this issue in SAC, this paper improves its implementation with a learnable state-dependent slack variable for appropriately handling the inequality constraint to maximize the policy entropy by reformulating it as the corresponding equality constraint. The introduced slack variable is optimized by a switching-type loss function that takes into account the dual objectives of satisfying the equality constraint and checking the lower bound. In Mujoco and Pybullet simulators, the modified SAC statistically achieved the higher robustness for adversarial attacks than before while regularizing the norm of action. A real-robot variable impedance task was demonstrated for showing the applicability of the modified SAC to real-world robot control. In particular, the modified SAC maintained adaptive behaviors for physical human-robot interaction, which had no experience at all during training. https://youtu.be/EH3xVtlVaJw

摘要: 强化学习中的软行动者-批评者(SAC)控制方案有望成为下一代机器人控制方案之一。其最大化策略熵的能力将使机器人控制器对噪声和扰动具有健壮性，这对现实世界的机器人应用很有用。然而，在当前的实现中，最大化策略熵的优先级是自动调整的，其规则可以解释为等式约束的规则，将策略熵绑定到其指定的下界。因此，当前的国资委不再是政策熵的最大化，与我们的预期相反。为了解决SAC中的这一问题，本文改进了SAC的实现，引入了一个可学习的状态依赖松弛变量，以适当地处理不平等约束，通过将其重新表示为相应的等式约束来最大化政策熵。引入的松弛变量通过切换型损失函数进行优化，该函数考虑了满足等式约束和检查下界的双重目标。在Mujoco和PYBullet模拟器中，改进的SAC在正规化行为规范的同时，在统计上获得了比以前更高的对抗攻击的鲁棒性。为了说明改进的SAC在实际机器人控制中的适用性，给出了一个真实的机器人变阻抗任务。特别是，改进的SAC保持了人与机器人物理交互的适应行为，在训练过程中完全没有经验。Https://youtu.be/EH3xVtlVaJw



## **35. X-Detect: Explainable Adversarial Patch Detection for Object Detectors in Retail**

X-Detect：零售业目标检测器的可解释敌意补丁检测 cs.CV

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2306.08422v2) [paper-pdf](http://arxiv.org/pdf/2306.08422v2)

**Authors**: Omer Hofman, Amit Giloni, Yarin Hayun, Ikuya Morikawa, Toshiya Shimizu, Yuval Elovici, Asaf Shabtai

**Abstract**: Object detection models, which are widely used in various domains (such as retail), have been shown to be vulnerable to adversarial attacks. Existing methods for detecting adversarial attacks on object detectors have had difficulty detecting new real-life attacks. We present X-Detect, a novel adversarial patch detector that can: i) detect adversarial samples in real time, allowing the defender to take preventive action; ii) provide explanations for the alerts raised to support the defender's decision-making process, and iii) handle unfamiliar threats in the form of new attacks. Given a new scene, X-Detect uses an ensemble of explainable-by-design detectors that utilize object extraction, scene manipulation, and feature transformation techniques to determine whether an alert needs to be raised. X-Detect was evaluated in both the physical and digital space using five different attack scenarios (including adaptive attacks) and the COCO dataset and our new Superstore dataset. The physical evaluation was performed using a smart shopping cart setup in real-world settings and included 17 adversarial patch attacks recorded in 1,700 adversarial videos. The results showed that X-Detect outperforms the state-of-the-art methods in distinguishing between benign and adversarial scenes for all attack scenarios while maintaining a 0% FPR (no false alarms) and providing actionable explanations for the alerts raised. A demo is available.

摘要: 目标检测模型被广泛应用于各个领域(如零售)，已被证明容易受到对手攻击。现有的用于检测对象检测器上的敌意攻击的方法已经很难检测到新的现实生活中的攻击。我们提出了X-Detect，这是一种新型的对抗性补丁检测器，它可以：i)实时检测对手样本，允许防御者采取预防措施；ii)为支持防御者决策过程而发出的警报提供解释；iii)处理新攻击形式的陌生威胁。给定一个新场景，X-Detect使用一组可通过设计解释的检测器，这些检测器利用对象提取、场景操作和特征转换技术来确定是否需要发出警报。X-Detect在物理和数字空间中使用五种不同的攻击场景(包括自适应攻击)以及Coco数据集和我们新的Superstore数据集进行了评估。物理评估是使用真实世界设置中的智能购物车进行的，包括1700个对抗性视频中记录的17个对抗性补丁攻击。结果表明，X-Detect在区分所有攻击场景的良性和敌意场景方面优于最先进的方法，同时保持0%的FPR(无错误警报)，并为发出的警报提供可行的解释。现已提供演示。



## **36. Query-Efficient Decision-based Black-Box Patch Attack**

基于查询高效决策的黑盒补丁攻击 cs.CV

**SubmitDate**: 2023-07-02    [abs](http://arxiv.org/abs/2307.00477v1) [paper-pdf](http://arxiv.org/pdf/2307.00477v1)

**Authors**: Zhaoyu Chen, Bo Li, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstract**: Deep neural networks (DNNs) have been showed to be highly vulnerable to imperceptible adversarial perturbations. As a complementary type of adversary, patch attacks that introduce perceptible perturbations to the images have attracted the interest of researchers. Existing patch attacks rely on the architecture of the model or the probabilities of predictions and perform poorly in the decision-based setting, which can still construct a perturbation with the minimal information exposed -- the top-1 predicted label. In this work, we first explore the decision-based patch attack. To enhance the attack efficiency, we model the patches using paired key-points and use targeted images as the initialization of patches, and parameter optimizations are all performed on the integer domain. Then, we propose a differential evolutionary algorithm named DevoPatch for query-efficient decision-based patch attacks. Experiments demonstrate that DevoPatch outperforms the state-of-the-art black-box patch attacks in terms of patch area and attack success rate within a given query budget on image classification and face verification. Additionally, we conduct the vulnerability evaluation of ViT and MLP on image classification in the decision-based patch attack setting for the first time. Using DevoPatch, we can evaluate the robustness of models to black-box patch attacks. We believe this method could inspire the design and deployment of robust vision models based on various DNN architectures in the future.

摘要: 深度神经网络(DNN)已被证明对难以察觉的对抗性扰动具有很强的脆弱性。作为一种互补类型的攻击，补丁攻击给图像带来了可感知的扰动，引起了研究人员的兴趣。现有的补丁攻击依赖于模型的体系结构或预测概率，在基于决策的环境下性能较差，仍然可以利用暴露的最小信息-TOP-1预测标签来构造扰动。在本工作中，我们首先探讨了基于决策的补丁攻击。为了提高攻击效率，我们使用成对的关键点对补丁进行建模，并使用目标图像作为补丁的初始化，并且参数优化都是在整数域上进行的。在此基础上，提出了一种用于查询高效的基于决策的补丁攻击的差分进化算法DevoPatch。实验表明，在给定的查询预算下，DevoPatch在图像分类和人脸验证方面，在补丁面积和攻击成功率方面都优于最先进的黑盒补丁攻击。此外，我们还首次在基于决策的补丁攻击环境下对VIT和MLP在图像分类上的脆弱性进行了评估。使用DevoPatch，我们可以评估模型对黑盒补丁攻击的稳健性。我们相信，这种方法可以启发未来基于各种DNN体系结构的健壮视觉模型的设计和部署。



## **37. Brightness-Restricted Adversarial Attack Patch**

亮度受限的对抗性攻击补丁 cs.CV

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00421v1) [paper-pdf](http://arxiv.org/pdf/2307.00421v1)

**Authors**: Mingzhen Shao

**Abstract**: Adversarial attack patches have gained increasing attention due to their practical applicability in physical-world scenarios. However, the bright colors used in attack patches represent a significant drawback, as they can be easily identified by human observers. Moreover, even though these attacks have been highly successful in deceiving target networks, which specific features of the attack patch contribute to its success are still unknown. Our paper introduces a brightness-restricted patch (BrPatch) that uses optical characteristics to effectively reduce conspicuousness while preserving image independence. We also conducted an analysis of the impact of various image features (such as color, texture, noise, and size) on the effectiveness of an attack patch in physical-world deployment. Our experiments show that attack patches exhibit strong redundancy to brightness and are resistant to color transfer and noise. Based on our findings, we propose some additional methods to further reduce the conspicuousness of BrPatch. Our findings also explain the robustness of attack patches observed in physical-world scenarios.

摘要: 对抗性攻击补丁由于其在物理世界场景中的实用适用性而受到越来越多的关注。然而，攻击补丁中使用的明亮颜色代表着一个重大缺陷，因为它们很容易被人类观察者识别出来。此外，尽管这些攻击在欺骗目标网络方面已经非常成功，但攻击补丁的哪些特定功能有助于其成功仍是未知的。本文提出了一种亮度受限的补丁(BrPatch)，它利用图像的光学特性，在保持图像独立性的同时，有效地降低了图像的显着性。我们还分析了各种图像特征(如颜色、纹理、噪声和大小)对物理世界部署中攻击补丁有效性的影响。实验表明，攻击补丁对亮度具有很强的冗余性，并且对颜色传递和噪声具有较强的抵抗能力。基于我们的发现，我们提出了一些额外的方法来进一步降低BrPatch的显着性。我们的发现也解释了在物理世界场景中观察到的攻击补丁的健壮性。



## **38. CasTGAN: Cascaded Generative Adversarial Network for Realistic Tabular Data Synthesis**

CasTGAN：用于现实表格数据合成的级联生成性对抗网络 cs.LG

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00384v1) [paper-pdf](http://arxiv.org/pdf/2307.00384v1)

**Authors**: Abdallah Alshantti, Damiano Varagnolo, Adil Rasheed, Aria Rahmati, Frank Westad

**Abstract**: Generative adversarial networks (GANs) have drawn considerable attention in recent years for their proven capability in generating synthetic data which can be utilized for multiple purposes. While GANs have demonstrated tremendous successes in producing synthetic data samples that replicate the dynamics of the original datasets, the validity of the synthetic data and the underlying privacy concerns represent major challenges which are not sufficiently addressed. In this work, we design a cascaded tabular GAN framework (CasTGAN) for generating realistic tabular data with a specific focus on the validity of the output. In this context, validity refers to the the dependency between features that can be found in the real data, but is typically misrepresented by traditional generative models. Our key idea entails that employing a cascaded architecture in which a dedicated generator samples each feature, the synthetic output becomes more representative of the real data. Our experimental results demonstrate that our model well captures the constraints and the correlations between the features of the real data, especially the high dimensional datasets. Furthermore, we evaluate the risk of white-box privacy attacks on our model and subsequently show that applying some perturbations to the auxiliary learners in CasTGAN increases the overall robustness of our model against targeted attacks.

摘要: 近年来，生成性对抗网络(GAN)因其在生成可用于多种目的的合成数据方面的能力而引起了相当大的关注。尽管Gans在生产复制原始数据集动态的合成数据样本方面取得了巨大成功，但合成数据的有效性和潜在的隐私问题是没有得到充分解决的主要挑战。在这项工作中，我们设计了一个级联表格GAN框架(CasTGAN)，用于生成真实的表格数据，并特别关注输出的有效性。在这种情况下，有效性是指在真实数据中可以找到的特征之间的依赖关系，但传统的生成模型通常会错误地表示这些特征。我们的关键思想是采用级联结构，其中由专用生成器对每个特征进行采样，合成输出变得更能代表真实数据。实验结果表明，我们的模型很好地捕捉了真实数据，特别是高维数据集的特征之间的约束和相关性。此外，我们评估了我们的模型受到白盒隐私攻击的风险，并随后表明，对CasTGAN中的辅助学习器应用一些扰动可以提高模型对目标攻击的整体稳健性。



## **39. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

一种稳健联邦学习的一阶Meta Stackelberg方法(技术报告) cs.CR

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Workshop Proceedings version: arXiv:2306.13800

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2306.13273v2) [paper-pdf](http://arxiv.org/pdf/2306.13273v2)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.

摘要: 最近的研究表明，联邦学习(FL)系统容易受到各种安全漏洞的攻击。虽然已经提出了许多防御策略，但它们主要是针对特定的攻击模式而设计的，缺乏适应性，使得它们在面临不确定或适应性威胁时效率较低。该工作将对抗性FL建模为防御者和攻击者之间的贝叶斯Stackelberg马尔可夫博弈(BSMG)，以解决对不确定自适应攻击缺乏适应性的问题。我们进一步设计了一种有效的元学习技术来求解Stackelberg均衡，从而导致具有弹性和适应性的防御。实验结果表明，我们的Meta-Stackelberg学习方法在对抗强烈的模型中毒和不确定类型的后门攻击方面表现出色。



## **40. A First Order Meta Stackelberg Method for Robust Federated Learning**

一种用于鲁棒联邦学习的一阶Meta Stackelberg方法 cs.LG

Accepted to ICML 2023 Workshop on The 2nd New Frontiers In  Adversarial Machine Learning. Associated technical report arXiv:2306.13273

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2306.13800v2) [paper-pdf](http://arxiv.org/pdf/2306.13800v2)

**Authors**: Yunian Pan, Tao Li, Henger Li, Tianyi Xu, Zizhan Zheng, Quanyan Zhu

**Abstract**: Previous research has shown that federated learning (FL) systems are exposed to an array of security risks. Despite the proposal of several defensive strategies, they tend to be non-adaptive and specific to certain types of attacks, rendering them ineffective against unpredictable or adaptive threats. This work models adversarial federated learning as a Bayesian Stackelberg Markov game (BSMG) to capture the defender's incomplete information of various attack types. We propose meta-Stackelberg learning (meta-SL), a provably efficient meta-learning algorithm, to solve the equilibrium strategy in BSMG, leading to an adaptable FL defense. We demonstrate that meta-SL converges to the first-order $\varepsilon$-equilibrium point in $O(\varepsilon^{-2})$ gradient iterations, with $O(\varepsilon^{-4})$ samples needed per iteration, matching the state of the art. Empirical evidence indicates that our meta-Stackelberg framework performs exceptionally well against potent model poisoning and backdoor attacks of an uncertain nature.

摘要: 先前的研究表明，联合学习(FL)系统面临一系列安全风险。尽管提出了几种防御战略，但它们往往是非适应性的，并且特定于某些类型的攻击，使得它们对不可预测或适应性威胁无效。该工作将对抗性联邦学习建模为贝叶斯Stackelberg马尔可夫博弈(BSMG)，以捕获防御者各种攻击类型的不完全信息。我们提出了元Stackelberg学习算法(META-SL)来解决BSMG中的均衡策略，从而得到一种自适应的FL防御。我们证明了META-SL在$O(varepsilon^{-2})$梯度迭代中收敛到一阶$varepsilon$-均衡点，每次迭代需要$O(varepsilon^{-4})$样本，与现有技术相匹配。经验证据表明，我们的Meta-Stackelberg框架在对抗强大的模型中毒和不确定性质的后门攻击时表现得非常好。



## **41. Fedward: Flexible Federated Backdoor Defense Framework with Non-IID Data**

Fedward：支持非IID数据的灵活联邦后门防御框架 cs.LG

Accepted by IEEE ICME 2023

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00356v1) [paper-pdf](http://arxiv.org/pdf/2307.00356v1)

**Authors**: Zekai Chen, Fuyi Wang, Zhiwei Zheng, Ximeng Liu, Yujie Lin

**Abstract**: Federated learning (FL) enables multiple clients to collaboratively train deep learning models while considering sensitive local datasets' privacy. However, adversaries can manipulate datasets and upload models by injecting triggers for federated backdoor attacks (FBA). Existing defense strategies against FBA consider specific and limited attacker models, and a sufficient amount of noise to be injected only mitigates rather than eliminates FBA. To address these deficiencies, we introduce a Flexible Federated Backdoor Defense Framework (Fedward) to ensure the elimination of adversarial backdoors. We decompose FBA into various attacks, and design amplified magnitude sparsification (AmGrad) and adaptive OPTICS clustering (AutoOPTICS) to address each attack. Meanwhile, Fedward uses the adaptive clipping method by regarding the number of samples in the benign group as constraints on the boundary. This ensures that Fedward can maintain the performance for the Non-IID scenario. We conduct experimental evaluations over three benchmark datasets and thoroughly compare them to state-of-the-art studies. The results demonstrate the promising defense performance from Fedward, moderately improved by 33% $\sim$ 75 in clustering defense methods, and 96.98%, 90.74%, and 89.8% for Non-IID to the utmost extent for the average FBA success rate over MNIST, FMNIST, and CIFAR10, respectively.

摘要: 联合学习(FL)使多个客户端能够协作训练深度学习模型，同时考虑敏感局部数据集的隐私。然而，攻击者可以通过为联合后门攻击(FBA)注入触发器来操纵数据集和上传模型。现有的针对FBA的防御策略考虑了特定和有限的攻击者模型，注入足够数量的噪音只会缓解而不是消除FBA。为了解决这些缺陷，我们引入了灵活的联邦后门防御框架(Fedward)来确保消除对抗性后门。我们将FBA分解为各种攻击，并设计了放大幅度稀疏化(AmGrad)和自适应光学聚类(AutoOPTICS)来应对每种攻击。同时，Fedward使用自适应裁剪方法，将良性组中的样本数作为边界约束。这确保了Fedward可以在非IID场景中保持性能。我们对三个基准数据集进行了实验评估，并将它们与最先进的研究进行了彻底的比较。结果表明，Fedward具有良好的防御性能，与MNIST、FMNIST和CIFAR10相比，集群防御方法的平均FBA成功率分别适度提高了33%、96.98%、90.74%和89.8%。



## **42. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

三维点云分类的对抗性攻防综述 cs.CV

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00309v1) [paper-pdf](http://arxiv.org/pdf/2307.00309v1)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification. This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes the adversarial example generation methods in recent years. Besides, it classifies defense strategies as input transformation, data optimization, and deep model modification. Finally, it presents several challenging issues and future research directions in this domain.

摘要: 深度学习作为一种占主导地位的人工智能技术，已经成功地解决了2D视觉中的一系列任务。近年来，针对三维点云的深度学习成为解决该领域各种问题的热门方法。尽管深度学习算法取得了令人瞩目的成就，但它仍然容易受到对手的攻击。这些攻击是人眼看不见的，但在测试和部署阶段很容易就能愚弄深度神经网络。为了鼓励未来的研究，本综述总结了当前点云分类对抗性攻防技术的研究进展。本文首先介绍了对抗性攻击的原理和特点，并对近年来对抗性实例生成方法进行了总结和分析。此外，还将防御策略分为输入转换、数据优化和深度模型修改。最后提出了该领域的几个具有挑战性的问题和未来的研究方向。



## **43. Common Knowledge Learning for Generating Transferable Adversarial Examples**

用于生成可转移对抗性实例的常识学习 cs.LG

11 pages, 5 figures

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00274v1) [paper-pdf](http://arxiv.org/pdf/2307.00274v1)

**Authors**: Ruijie Yang, Yuanfang Guo, Junfu Wang, Jiantao Zhou, Yunhong Wang

**Abstract**: This paper focuses on an important type of black-box attacks, i.e., transfer-based adversarial attacks, where the adversary generates adversarial examples by a substitute (source) model and utilize them to attack an unseen target model, without knowing its information. Existing methods tend to give unsatisfactory adversarial transferability when the source and target models are from different types of DNN architectures (e.g. ResNet-18 and Swin Transformer). In this paper, we observe that the above phenomenon is induced by the output inconsistency problem. To alleviate this problem while effectively utilizing the existing DNN models, we propose a common knowledge learning (CKL) framework to learn better network weights to generate adversarial examples with better transferability, under fixed network architectures. Specifically, to reduce the model-specific features and obtain better output distributions, we construct a multi-teacher framework, where the knowledge is distilled from different teacher architectures into one student network. By considering that the gradient of input is usually utilized to generated adversarial examples, we impose constraints on the gradients between the student and teacher models, to further alleviate the output inconsistency problem and enhance the adversarial transferability. Extensive experiments demonstrate that our proposed work can significantly improve the adversarial transferability.

摘要: 本文研究了一种重要的黑盒攻击类型，即基于转移的对抗性攻击，对手通过替代(源)模型生成对抗性实例，并利用它们攻击一个看不见的目标模型，而不需要知道其信息。当源模型和目标模型来自不同类型的DNN结构(例如ResNet-18和Swin Transformer)时，现有方法往往给出不令人满意的对抗性可转移性。在本文中，我们观察到上述现象是由输出不一致问题引起的。为了在有效利用现有DNN模型的同时缓解这一问题，我们提出了一种公共知识学习(CKL)框架，在固定的网络结构下学习更好的网络权重来生成具有更好可移植性的对抗性示例。具体地说，为了减少模型的特定特征并获得更好的输出分布，我们构建了一个多教师框架，其中来自不同教师体系的知识提取到一个学生网络中。考虑到输入的梯度通常被用来生成对抗性示例，我们对学生模型和教师模型之间的梯度施加约束，以进一步缓解输出不一致问题，增强对抗性可转移性。大量实验表明，我们提出的工作可以显著提高对抗性可转移性。



## **44. Hiding in Plain Sight: Differential Privacy Noise Exploitation for Evasion-resilient Localized Poisoning Attacks in Multiagent Reinforcement Learning**

隐藏在明显的视线中：在多智能体强化学习中利用差分隐私噪声进行逃避弹性局部中毒攻击 cs.LG

Accepted for publication in the proceeding of ICMLC 2023, 9-11 July  2023, The University of Adelaide, Adelaide, Australia

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00268v1) [paper-pdf](http://arxiv.org/pdf/2307.00268v1)

**Authors**: Md Tamjid Hossain, Hung La

**Abstract**: Lately, differential privacy (DP) has been introduced in cooperative multiagent reinforcement learning (CMARL) to safeguard the agents' privacy against adversarial inference during knowledge sharing. Nevertheless, we argue that the noise introduced by DP mechanisms may inadvertently give rise to a novel poisoning threat, specifically in the context of private knowledge sharing during CMARL, which remains unexplored in the literature. To address this shortcoming, we present an adaptive, privacy-exploiting, and evasion-resilient localized poisoning attack (PeLPA) that capitalizes on the inherent DP-noise to circumvent anomaly detection systems and hinder the optimal convergence of the CMARL model. We rigorously evaluate our proposed PeLPA attack in diverse environments, encompassing both non-adversarial and multiple-adversarial contexts. Our findings reveal that, in a medium-scale environment, the PeLPA attack with attacker ratios of 20% and 40% can lead to an increase in average steps to goal by 50.69% and 64.41%, respectively. Furthermore, under similar conditions, PeLPA can result in a 1.4x and 1.6x computational time increase in optimal reward attainment and a 1.18x and 1.38x slower convergence for attacker ratios of 20% and 40%, respectively.

摘要: 最近，在协作多智能体强化学习(CMARL)中引入了差异隐私(DP)，以保护智能体在知识共享过程中的隐私不受对手推理的影响。然而，我们认为DP机制引入的噪声可能无意中引起一种新的中毒威胁，特别是在CMARL期间的私人知识共享的背景下，这在文献中仍未被探索。针对这一缺陷，我们提出了一种自适应的、利用隐私攻击和逃避弹性的局部中毒攻击(PeLPA)，该攻击利用固有的DP噪声来绕过异常检测系统并阻碍CMARL模型的最优收敛。我们在不同的环境中严格评估我们提出的PeLPA攻击，包括非对抗性和多对抗性环境。我们的研究结果表明，在中等规模的环境中，攻击者比率分别为20%和40%的PeLPA攻击可以使到达目标的平均步数分别增加50.69%和64.41%。此外，在类似的条件下，PeLPA可以使最优奖励获得的计算时间分别增加1.4倍和1.6倍，对于攻击比率分别为20%和40%的攻击者，收敛速度分别慢1.18倍和1.38倍。



## **45. A Black-box NLP Classifier Attacker**

黑盒NLP分类器攻击者 cs.LG

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2112.11660v3) [paper-pdf](http://arxiv.org/pdf/2112.11660v3)

**Authors**: Yueyang Liu, Hunmin Lee, Zhipeng Cai

**Abstract**: Deep neural networks have a wide range of applications in solving various real-world tasks and have achieved satisfactory results, in domains such as computer vision, image classification, and natural language processing. Meanwhile, the security and robustness of neural networks have become imperative, as diverse researches have shown the vulnerable aspects of neural networks. Case in point, in Natural language processing tasks, the neural network may be fooled by an attentively modified text, which has a high similarity to the original one. As per previous research, most of the studies are focused on the image domain; Different from image adversarial attacks, the text is represented in a discrete sequence, traditional image attack methods are not applicable in the NLP field. In this paper, we propose a word-level NLP sentiment classifier attack model, which includes a self-attention mechanism-based word selection method and a greedy search algorithm for word substitution. We experiment with our attack model by attacking GRU and 1D-CNN victim models on IMDB datasets. Experimental results demonstrate that our model achieves a higher attack success rate and more efficient than previous methods due to the efficient word selection algorithms are employed and minimized the word substitute number. Also, our model is transferable, which can be used in the image domain with several modifications.

摘要: 深度神经网络在计算机视觉、图像分类、自然语言处理等领域有着广泛的应用，并取得了令人满意的结果。与此同时，随着各种研究表明神经网络的脆弱方面，神经网络的安全性和健壮性变得势在必行。例如，在自然语言处理任务中，神经网络可能会被精心修改的文本所愚弄，因为它与原始文本具有很高的相似性。根据以往的研究，大多数研究都集中在图像领域；与图像对抗性攻击不同，文本是离散序列表示的，传统的图像攻击方法不适用于自然语言处理领域。本文提出了一个词级NLP情感分类器攻击模型，该模型包括一种基于自我注意机制的词选择方法和一种贪婪的词替换搜索算法。我们在IMDB数据集上通过攻击GRU和1D-CNN受害者模型来测试我们的攻击模型。实验结果表明，由于采用了高效的选词算法和最小化了替换词的数量，该模型获得了更高的攻击成功率和更高的效率。此外，我们的模型是可移植的，只需进行几次修改就可以在图像域使用。



## **46. SecBeam: Securing mmWave Beam Alignment against Beam-Stealing Attacks**

SecBeam：保护毫米波波束对准免受波束窃取攻击 cs.CR

**SubmitDate**: 2023-07-01    [abs](http://arxiv.org/abs/2307.00178v1) [paper-pdf](http://arxiv.org/pdf/2307.00178v1)

**Authors**: Jingcheng Li, Loukas Lazos, Ming Li

**Abstract**: Millimeter wave (mmWave) communications employ narrow-beam directional communications to compensate for the high path loss at mmWave frequencies. Compared to their omnidirectional counterparts, an additional step of aligning the transmitter's and receiver's antennas is required. In current standards such as 802.11ad, this beam alignment process is implemented via an exhaustive search through the horizontal plane known as beam sweeping. However, the beam sweeping process is unauthenticated. As a result, an adversary, Mallory, can launch an active beam-stealing attack by injecting forged beacons of high power, forcing the legitimate devices to beamform towards her direction. Mallory is now in control of the communication link between the two devices, thus breaking the false sense of security given by the directionality of mmWave transmissions.   Prior works have added integrity protection to beam alignment messages to prevent forgeries. In this paper, we demonstrate a new beam-stealing attack that does not require message forging. We show that Mallory can amplify and relay a beam sweeping frame from her direction without altering its contents. Intuitively, cryptographic primitives cannot verify physical properties such as the SNR used in beam selection. We propose a new beam sweeping protocol called SecBeam that utilizes power/sector randomization and coarse angle-of-arrival information to detect amplify-and-relay attacks. We demonstrate the security and performance of SecBeam using an experimental mmWave platform and via ray-tracing simulations.

摘要: 毫米波(毫米波)通信使用窄波束定向通信来补偿毫米波频率下的高路径损耗。与它们的全方位对应物相比，需要一个额外的步骤来对准发射器和接收器的天线。在当前的标准中，例如802.11ad，这种波束对准过程是通过称为波束扫描的水平面的穷举搜索来实现的。然而，波束扫描过程未经验证。因此，对手Mallory可以通过注入高功率的伪造信标来发动主动窃取波束攻击，迫使合法设备朝着她的方向波束成形。马洛里现在控制着这两个设备之间的通信链路，从而打破了毫米波传输的方向性带来的错误安全感。以前的工作已经为波束对准消息增加了完整性保护，以防止伪造。在本文中，我们证明了一种新的不需要消息伪造的波束窃取攻击。我们证明了Mallory可以在不改变其内容的情况下放大和中继从她的方向扫描的光束帧。直观地说，密码原语不能验证物理属性，例如在波束选择中使用的SNR。我们提出了一种新的波束扫描协议SecBeam，该协议利用功率/扇区随机化和粗到达角信息来检测放大和中继攻击。我们使用一个实验性的毫米波平台和光线跟踪模拟来演示SecBeam的安全性和性能。



## **47. Beyond Neural-on-Neural Approaches to Speaker Gender Protection**

超越神经对神经的说话人性别保护方法 eess.AS

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2306.17700v1) [paper-pdf](http://arxiv.org/pdf/2306.17700v1)

**Authors**: Loes van Bemmel, Zhuoran Liu, Nik Vaessen, Martha Larson

**Abstract**: Recent research has proposed approaches that modify speech to defend against gender inference attacks. The goal of these protection algorithms is to control the availability of information about a speaker's gender, a privacy-sensitive attribute. Currently, the common practice for developing and testing gender protection algorithms is "neural-on-neural", i.e., perturbations are generated and tested with a neural network. In this paper, we propose to go beyond this practice to strengthen the study of gender protection. First, we demonstrate the importance of testing gender inference attacks that are based on speech features historically developed by speech scientists, alongside the conventionally used neural classifiers. Next, we argue that researchers should use speech features to gain insight into how protective modifications change the speech signal. Finally, we point out that gender-protection algorithms should be compared with novel "vocal adversaries", human-executed voice adaptations, in order to improve interpretability and enable before-the-mic protection.

摘要: 最近的研究提出了修改语音以防御性别推断攻击的方法。这些保护算法的目标是控制有关说话人性别的信息的可用性，这是一种隐私敏感属性。目前，开发和测试性别保护算法的常见做法是“神经对神经”，即产生扰动并使用神经网络进行测试。在本文中，我们建议超越这一做法，加强对性别保护的研究。首先，我们证明了测试性别推断攻击的重要性，这些攻击基于语音科学家历史上开发的语音特征，以及常规使用的神经分类器。接下来，我们认为研究人员应该使用语音特征来洞察保护性修改如何改变语音信号。最后，我们指出，应该将性别保护算法与人类执行的语音改编的新颖的“声音对手”进行比较，以提高可解释性，并实现麦克风前的保护。



## **48. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

MalProtect：基于ML的恶意软件检测中对抗恶意查询攻击的状态防御 cs.LG

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2302.10739v3) [paper-pdf](http://arxiv.org/pdf/2302.10739v3)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.

摘要: 众所周知，ML模型容易受到敌意查询攻击。在这些攻击中，查询被迭代地扰动到特定的类，除了其输出之外，不知道目标模型。远程托管的ML分类模型和机器学习即服务平台的流行意味着查询攻击对这些系统的安全构成了真正的威胁。为了解决这个问题，已经提出了状态防御来检测查询攻击，并通过监控和分析系统接收到的查询序列来防止敌对实例的生成。近年来，有人提出了几项有状态的辩护。然而，这些防御完全依赖于可能在其他领域有效的相似性或分布外检测方法。在恶意软件检测领域，生成恶意示例的方法本质上是不同的，因此我们发现这种检测机制的有效性显著降低。因此，在本文中，我们提出了MalProtect，它是恶意软件检测领域中针对查询攻击的一种状态防御。MalProtect使用多个威胁指示器来检测攻击。我们的结果表明，在各种攻击场景下，该算法将Android和Windows恶意软件中恶意查询攻击的逃避率降低了80%+\%。在该类型的第一次评估中，我们表明MalProtect的性能优于先前的状态防御，特别是在峰值敌意威胁下。



## **49. Efficient Backdoor Removal Through Natural Gradient Fine-tuning**

通过自然梯度微调高效删除后门 cs.CV

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2306.17441v1) [paper-pdf](http://arxiv.org/pdf/2306.17441v1)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Naznin Rahnavard

**Abstract**: The success of a deep neural network (DNN) heavily relies on the details of the training scheme; e.g., training data, architectures, hyper-parameters, etc. Recent backdoor attacks suggest that an adversary can take advantage of such training details and compromise the integrity of a DNN. Our studies show that a backdoor model is usually optimized to a bad local minima, i.e. sharper minima as compared to a benign model. Intuitively, a backdoor model can be purified by reoptimizing the model to a smoother minima through fine-tuning with a few clean validation data. However, fine-tuning all DNN parameters often requires huge computational costs and often results in sub-par clean test performance. To address this concern, we propose a novel backdoor purification technique, Natural Gradient Fine-tuning (NGF), which focuses on removing the backdoor by fine-tuning only one layer. Specifically, NGF utilizes a loss surface geometry-aware optimizer that can successfully overcome the challenge of reaching a smooth minima under a one-layer optimization scenario. To enhance the generalization performance of our proposed method, we introduce a clean data distribution-aware regularizer based on the knowledge of loss surface curvature matrix, i.e., Fisher Information Matrix. Extensive experiments show that the proposed method achieves state-of-the-art performance on a wide range of backdoor defense benchmarks: four different datasets- CIFAR10, GTSRB, Tiny-ImageNet, and ImageNet; 13 recent backdoor attacks, e.g. Blend, Dynamic, WaNet, ISSBA, etc.

摘要: 深度神经网络(DNN)的成功在很大程度上依赖于训练方案的细节；例如，训练数据、体系结构、超参数等。最近的后门攻击表明，对手可以利用这些训练细节并损害DNN的完整性。我们的研究表明，与良性模型相比，后门模型通常会优化到较差的局部最小值，即更尖锐的最小值。直观地说，后门模型可以通过使用几个干净的验证数据进行微调，将模型重新优化到更平滑的最小值来进行净化。然而，微调所有DNN参数通常需要巨大的计算成本，并且通常会导致测试性能低于平均水平。为了解决这个问题，我们提出了一种新的后门净化技术，自然梯度微调(NGF)，它专注于通过微调一层来消除后门。具体地说，NGF利用了一种损失面几何感知优化器，它可以成功地克服在单层优化场景下达到平滑最小值的挑战。为了提高该方法的泛化性能，我们引入了一种干净的基于损失曲面曲率矩阵知识的数据分布感知正则化方法，即Fisher信息矩阵。大量实验表明，该方法在CIFAR10、GTSRB、Tiny-ImageNet和ImageNet四个不同的数据集上，以及最近的13个后门攻击，如Blend、Dynamic、WaNet、Issba等上，都达到了最先进的性能。



## **50. LTD: Low Temperature Distillation for Robust Adversarial Training**

LTD：低温蒸馏用于强大的对抗性训练 cs.CV

**SubmitDate**: 2023-06-30    [abs](http://arxiv.org/abs/2111.02331v3) [paper-pdf](http://arxiv.org/pdf/2111.02331v3)

**Authors**: Erh-Chung Chen, Che-Rung Lee

**Abstract**: Adversarial training has been widely used to enhance the robustness of neural network models against adversarial attacks. Despite the popularity of neural network models, a significant gap exists between the natural and robust accuracy of these models. In this paper, we identify one of the primary reasons for this gap is the common use of one-hot vectors as labels, which hinders the learning process for image recognition. Representing ambiguous images with one-hot vectors is imprecise and may lead the model to suboptimal solutions. To overcome this issue, we propose a novel method called Low Temperature Distillation (LTD) that generates soft labels using the modified knowledge distillation framework. Unlike previous approaches, LTD uses a relatively low temperature in the teacher model and fixed, but different temperatures for the teacher and student models. This modification boosts the model's robustness without encountering the gradient masking problem that has been addressed in defensive distillation. The experimental results demonstrate the effectiveness of the proposed LTD method combined with previous techniques, achieving robust accuracy rates of 58.19%, 31.13%, and 42.08% on CIFAR-10, CIFAR-100, and ImageNet data sets, respectively, without additional unlabeled data.

摘要: 对抗性训练已被广泛应用于增强神经网络模型对对抗性攻击的鲁棒性。尽管神经网络模型很受欢迎，但这些模型的自然精度和稳健精度之间存在着巨大的差距。在本文中，我们发现造成这一差距的主要原因之一是普遍使用单一热点向量作为标签，这阻碍了图像识别的学习过程。用一个热点向量表示模糊图像是不精确的，并且可能导致模型得到次优解。为了解决这个问题，我们提出了一种新的方法，称为低温蒸馏(LTD)，它使用改进的知识蒸馏框架来生成软标签。与以前的方法不同，LTD在教师模型中使用相对较低的温度，并为教师和学生模型使用固定但不同的温度。这种修改增强了模型的稳健性，而不会遇到在防御蒸馏中已经解决的梯度掩蔽问题。实验结果证明了LTD方法的有效性，在CIFAR-10、CIFAR-100和ImageNet数据集上，在不增加额外的未标记数据的情况下，分别获得了58.19%、31.13%和42.08%的稳健准确率。



