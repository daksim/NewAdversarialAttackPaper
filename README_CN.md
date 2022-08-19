# Latest Adversarial Attack Papers
**update at 2022-08-19 10:27:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. How many perturbations break this model? Evaluating robustness beyond adversarial accuracy**

有多少扰动打破了这个模型？评估超越对手准确性的稳健性 cs.LG

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2207.04129v2)

**Authors**: Raphael Olivier, Bhiksha Raj

**Abstracts**: Robustness to adversarial attack is typically evaluated with adversarial accuracy. This metric quantifies the number of points for which, given a threat model, successful adversarial perturbations cannot be found. While essential, this metric does not capture all aspects of robustness and in particular leaves out the question of how many perturbations can be found for each point. In this work we introduce an alternative approach, adversarial sparsity, which quantifies how difficult it is to find a successful perturbation given both an input point and a constraint on the direction of the perturbation. This constraint may be angular (L2 perturbations), or based on the number of pixels (Linf perturbations).   We show that sparsity provides valuable insight on neural networks in multiple ways. analyzing the sparsity of existing robust models illustrates important differences between them that accuracy analysis does not, and suggests approaches for improving their robustness. When applying broken defenses effective against weak attacks but not strong ones, sparsity can discriminate between the totally ineffective and the partially effective defenses. Finally, with sparsity we can measure increases in robustness that do not affect accuracy: we show for example that data augmentation can by itself increase adversarial robustness, without using adversarial training.

摘要: 对敌方攻击的稳健性通常用敌方准确度来评估。此度量量化了在给定威胁模型的情况下，无法找到成功的对抗性扰动的点数。虽然这一指标很重要，但它并没有涵盖健壮性的所有方面，尤其是忽略了每个点可以发现多少扰动的问题。在这项工作中，我们引入了另一种方法，逆稀疏性，它量化了在给定输入点和对扰动方向的约束的情况下找到成功的扰动的难度。该约束可以是角度的(L2扰动)，也可以基于像素数(LINF扰动)。我们表明，稀疏性以多种方式提供了对神经网络的有价值的见解。分析现有稳健模型的稀疏性，说明了它们之间的重要差异，而精度分析则没有，并提出了提高其稳健性的方法。当应用破碎的防御对弱攻击有效而对强攻击无效时，稀疏性可以区分完全无效的防御和部分有效的防御。最后，利用稀疏性，我们可以测量不影响准确性的稳健性增加：例如，我们证明了数据增强本身可以增加对抗性健壮性，而不使用对抗性训练。



## **2. Resisting Adversarial Attacks in Deep Neural Networks using Diverse Decision Boundaries**

利用不同决策边界的深度神经网络抵抗敌意攻击 cs.LG

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08697v1)

**Authors**: Manaar Alam, Shubhajit Datta, Debdeep Mukhopadhyay, Arijit Mondal, Partha Pratim Chakrabarti

**Abstracts**: The security of deep learning (DL) systems is an extremely important field of study as they are being deployed in several applications due to their ever-improving performance to solve challenging tasks. Despite overwhelming promises, the deep learning systems are vulnerable to crafted adversarial examples, which may be imperceptible to the human eye, but can lead the model to misclassify. Protections against adversarial perturbations on ensemble-based techniques have either been shown to be vulnerable to stronger adversaries or shown to lack an end-to-end evaluation. In this paper, we attempt to develop a new ensemble-based solution that constructs defender models with diverse decision boundaries with respect to the original model. The ensemble of classifiers constructed by (1) transformation of the input by a method called Split-and-Shuffle, and (2) restricting the significant features by a method called Contrast-Significant-Features are shown to result in diverse gradients with respect to adversarial attacks, which reduces the chance of transferring adversarial examples from the original to the defender model targeting the same class. We present extensive experimentations using standard image classification datasets, namely MNIST, CIFAR-10 and CIFAR-100 against state-of-the-art adversarial attacks to demonstrate the robustness of the proposed ensemble-based defense. We also evaluate the robustness in the presence of a stronger adversary targeting all the models within the ensemble simultaneously. Results for the overall false positives and false negatives have been furnished to estimate the overall performance of the proposed methodology.

摘要: 深度学习系统的安全性是一个非常重要的研究领域，因为它们被部署在几个应用程序中，因为它们的性能不断提高，以解决具有挑战性的任务。尽管有着压倒性的承诺，但深度学习系统很容易受到精心制作的对抗性例子的攻击，这些例子可能是人眼看不见的，但可能会导致模型错误分类。针对基于集合的技术的对抗性扰动的保护要么被证明容易受到更强大的对手的攻击，要么被证明缺乏端到端的评估。在本文中，我们试图开发一种新的基于集成的解决方案，该方案构建相对于原始模型具有不同决策边界的防御者模型。通过(1)通过称为Split-and-Shuffle的方法对输入进行变换，以及(2)通过称为对比度显著特征的方法限制显著特征来构建的分类器集成被证明导致相对于对抗性攻击的不同梯度，这降低了将对抗性样本从原始模型转移到针对同一类的防御者模型的机会。我们使用标准的图像分类数据集，即MNIST，CIFAR-10和CIFAR-100，针对最先进的对手攻击进行了广泛的实验，以证明所提出的基于集成的防御的健壮性。我们还评估了当一个更强大的对手同时针对集成内的所有模型时的稳健性。提供了总体假阳性和假阴性的结果，以评估所提出的方法的总体性能。



## **3. Reverse Engineering of Integrated Circuits: Tools and Techniques**

集成电路逆向工程：工具与技术 cs.CR

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08689v1)

**Authors**: Abhijitt Dhavlle

**Abstracts**: Consumer and defense systems demanded design and manufacturing of electronics with increased performance, compared to their predecessors. As such systems became ubiquitous in a plethora of domains, their application surface increased, thus making them a target for adversaries. Hence, with improved performance the aspect of security demanded even more attention of the designers. The research community is rife with extensive details of attacks that target the confidential design details by exploiting vulnerabilities. The adversary could target the physical design of a semiconductor chip or break a cryptographic algorithm by extracting the secret keys, using attacks that will be discussed in this thesis. This thesis focuses on presenting a brief overview of IC reverse engineering attack and attacks targeting cryptographic systems. Further, the thesis presents my contributions to the defenses for the discussed attacks. The globalization of the Integrated Circuit (IC) supply chain has rendered the advantage of low-cost and high-performance ICs in the market for the end users. But this has also made the design vulnerable to over production, IP Piracy, reverse engineering attacks and hardware malware during the manufacturing and post manufacturing process. Logic locking schemes have been proposed in the past to overcome the design trust issues but the new state-of-the-art attacks such as SAT has proven a larger threat. This work highlights the reverse engineering attack and a proposed hardened platform along with its framework.

摘要: 消费者和国防系统要求设计和制造的电子产品与它们的前身相比性能更高。随着这样的系统在太多的领域变得无处不在，它们的应用表面增加了，从而使它们成为对手的目标。因此，随着性能的提高，安全性方面对设计者提出了更高的要求。研究界充斥着大量攻击的细节，这些攻击通过利用漏洞来攻击机密的设计细节。攻击者可能以半导体芯片的物理设计为目标，或通过提取密钥来破解密码算法，使用本文将讨论的攻击。本文对IC逆向工程攻击和针对密码系统的攻击进行了简要的综述。此外，本文还提出了我对所讨论的攻击的防御的贡献。集成电路(IC)供应链的全球化为终端用户提供了低成本、高性能IC的市场优势。但这也使该设计在制造和后期制造过程中容易受到过度生产、知识产权盗版、反向工程攻击和硬件恶意软件的攻击。过去有人提出了逻辑锁定方案来克服设计信任问题，但事实证明，新的最先进的攻击，如SAT，是一个更大的威胁。这项工作突出了反向工程攻击和拟议的强化平台及其框架。



## **4. Enhancing Targeted Attack Transferability via Diversified Weight Pruning**

通过不同的权重剪枝提高目标攻击的可转移性 cs.CV

8 pages + 2 pages of references

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08677v1)

**Authors**: Hung-Jui Wang, Yu-Yu Wu, Shang-Tse Chen

**Abstracts**: Malicious attackers can generate targeted adversarial examples by imposing human-imperceptible noise on images, forcing neural network models to produce specific incorrect outputs. With cross-model transferable adversarial examples, the vulnerability of neural networks remains even if the model information is kept secret from the attacker. Recent studies have shown the effectiveness of ensemble-based methods in generating transferable adversarial examples. However, existing methods fall short under the more challenging scenario of creating targeted attacks transferable among distinct models. In this work, we propose Diversified Weight Pruning (DWP) to further enhance the ensemble-based methods by leveraging the weight pruning method commonly used in model compression. Specifically, we obtain multiple diverse models by a random weight pruning method. These models preserve similar accuracies and can serve as additional models for ensemble-based methods, yielding stronger transferable targeted attacks. Experiments on ImageNet-Compatible Dataset under the more challenging scenarios are provided: transferring to distinct architectures and to adversarially trained models. The results show that our proposed DWP improves the targeted attack success rates with up to 4.1% and 8.0% on the combination of state-of-the-art methods, respectively

摘要: 恶意攻击者可以通过在图像上施加人类无法察觉的噪声，迫使神经网络模型生成特定的错误输出，从而生成有针对性的对抗性示例。对于跨模型可传递的对抗性例子，即使模型信息对攻击者保密，神经网络的脆弱性仍然存在。最近的研究表明，基于集成的方法在生成可转移的对抗性例子方面是有效的。然而，现有的方法不能满足更具挑战性的场景，即创建可在不同模型之间转移的有针对性的攻击。在这项工作中，我们提出了多样性权重剪枝(DWP)，通过利用模型压缩中常用的权重剪枝方法，进一步增强了基于集成的方法。具体地说，我们通过随机权重剪枝的方法得到多个不同的模型。这些模型保持了类似的精度，并可以作为基于集成的方法的附加模型，产生更强的可转移的定向攻击。提供了在更具挑战性的场景下对ImageNet兼容的数据集进行的实验：转换到不同的体系结构和相反的训练模型。结果表明，与最新的攻击方法相结合，我们提出的DWP分别提高了4.1%和8.0%的目标攻击成功率



## **5. Enhancing Diffusion-Based Image Synthesis with Robust Classifier Guidance**

稳健分类器引导增强基于扩散的图像合成 cs.CV

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08664v1)

**Authors**: Bahjat Kawar, Roy Ganz, Michael Elad

**Abstracts**: Denoising diffusion probabilistic models (DDPMs) are a recent family of generative models that achieve state-of-the-art results. In order to obtain class-conditional generation, it was suggested to guide the diffusion process by gradients from a time-dependent classifier. While the idea is theoretically sound, deep learning-based classifiers are infamously susceptible to gradient-based adversarial attacks. Therefore, while traditional classifiers may achieve good accuracy scores, their gradients are possibly unreliable and might hinder the improvement of the generation results. Recent work discovered that adversarially robust classifiers exhibit gradients that are aligned with human perception, and these could better guide a generative process towards semantically meaningful images. We utilize this observation by defining and training a time-dependent adversarially robust classifier and use it as guidance for a generative diffusion model. In experiments on the highly challenging and diverse ImageNet dataset, our scheme introduces significantly more intelligible intermediate gradients, better alignment with theoretical findings, as well as improved generation results under several evaluation metrics. Furthermore, we conduct an opinion survey whose findings indicate that human raters prefer our method's results.

摘要: 去噪扩散概率模型(DDPM)是最近出现的一类产生式模型，可以得到最先进的结果。为了获得类条件生成，建议使用依赖于时间的分类器的梯度来指导扩散过程。虽然这个想法在理论上是合理的，但基于深度学习的分类器很容易受到基于梯度的对抗性攻击。因此，虽然传统的分类器可以获得很好的精度分数，但它们的梯度可能是不可靠的，并可能阻碍生成结果的改进。最近的研究发现，逆序稳健的分类器表现出与人类感知一致的梯度，这些梯度可以更好地引导生成过程走向有语义意义的图像。我们通过定义和训练一个时间相关的对抗性稳健分类器来利用这一观察结果，并将其用作生成性扩散模型的指导。在高度挑战性和多样性的ImageNet数据集上的实验中，我们的方案引入了明显更易理解的中间梯度，更好地与理论结果保持一致，以及在几种评估指标下改进的生成结果。此外，我们还进行了一项民意调查，调查结果表明，人类评分者更喜欢我们的方法的结果。



## **6. Efficient Detection and Filtering Systems for Distributed Training**

用于分布式训练的高效检测和过滤系统 cs.LG

18 pages, 14 figures, 6 tables. arXiv admin note: substantial text  overlap with arXiv:2108.02416

**SubmitDate**: 2022-08-18    [paper-pdf](http://arxiv.org/pdf/2208.08085v2)

**Authors**: Konstantinos Konstantinidis, Aditya Ramamoorthy

**Abstracts**: A plethora of modern machine learning tasks require the utilization of large-scale distributed clusters as a critical component of the training pipeline. However, abnormal Byzantine behavior of the worker nodes can derail the training and compromise the quality of the inference. Such behavior can be attributed to unintentional system malfunctions or orchestrated attacks; as a result, some nodes may return arbitrary results to the parameter server (PS) that coordinates the training. Recent work considers a wide range of attack models and has explored robust aggregation and/or computational redundancy to correct the distorted gradients. In this work, we consider attack models ranging from strong ones: $q$ omniscient adversaries with full knowledge of the defense protocol that can change from iteration to iteration to weak ones: $q$ randomly chosen adversaries with limited collusion abilities which only change every few iterations at a time. Our algorithms rely on redundant task assignments coupled with detection of adversarial behavior. For strong attacks, we demonstrate a reduction in the fraction of distorted gradients ranging from 16%-99% as compared to the prior state-of-the-art. Our top-1 classification accuracy results on the CIFAR-10 data set demonstrate 25% advantage in accuracy (averaged over strong and weak scenarios) under the most sophisticated attacks compared to state-of-the-art methods.

摘要: 过多的现代机器学习任务需要利用大规模分布式集群作为培训管道的关键组成部分。然而，工作者节点的异常拜占庭行为会破坏训练，影响推理的质量。此类行为可归因于无意的系统故障或精心策划的攻击；因此，某些节点可能会向协调训练的参数服务器(PS)返回任意结果。最近的工作考虑了广泛的攻击模型，并探索了稳健的聚集和/或计算冗余来纠正扭曲的梯度。在这项工作中，我们考虑了从强到强的攻击模型：$q$全知的对手，完全了解防御协议，可以从一个迭代到另一个迭代变化；$q$随机选择的对手，合谋能力有限，一次只有几个迭代改变。我们的算法依赖于冗余的任务分配以及对敌对行为的检测。对于强攻击，我们展示了与以前最先进的技术相比，扭曲梯度的比例降低了16%-99%。我们在CIFAR-10数据集上的TOP-1分类精度结果显示，在最复杂的攻击下，与最先进的方法相比，准确率(在强和弱场景下平均)提高了25%。



## **7. ObfuNAS: A Neural Architecture Search-based DNN Obfuscation Approach**

ObfuNAS：一种基于神经结构搜索的DNN混淆方法 cs.CR

9 pages

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08569v1)

**Authors**: Tong Zhou, Shaolei Ren, Xiaolin Xu

**Abstracts**: Malicious architecture extraction has been emerging as a crucial concern for deep neural network (DNN) security. As a defense, architecture obfuscation is proposed to remap the victim DNN to a different architecture. Nonetheless, we observe that, with only extracting an obfuscated DNN architecture, the adversary can still retrain a substitute model with high performance (e.g., accuracy), rendering the obfuscation techniques ineffective. To mitigate this under-explored vulnerability, we propose ObfuNAS, which converts the DNN architecture obfuscation into a neural architecture search (NAS) problem. Using a combination of function-preserving obfuscation strategies, ObfuNAS ensures that the obfuscated DNN architecture can only achieve lower accuracy than the victim. We validate the performance of ObfuNAS with open-source architecture datasets like NAS-Bench-101 and NAS-Bench-301. The experimental results demonstrate that ObfuNAS can successfully find the optimal mask for a victim model within a given FLOPs constraint, leading up to 2.6% inference accuracy degradation for attackers with only 0.14x FLOPs overhead. The code is available at: https://github.com/Tongzhou0101/ObfuNAS.

摘要: 恶意体系结构提取已经成为深度神经网络(DNN)安全的一个重要问题。作为防御，体系结构混淆被提出将受害者DNN重新映射到不同的体系结构。尽管如此，我们观察到，只要提取一个混淆的DNN体系结构，攻击者仍然可以高性能(例如，准确性)重新训练替代模型，使得混淆技术无效。为了缓解这一未被充分挖掘的漏洞，我们提出了ObfuNAS，它将DNN体系结构的混淆转化为神经体系结构搜索(NAS)问题。ObfuNAS结合了函数保留混淆策略，确保了混淆后的DNN架构只能达到比受害者更低的准确率。我们使用NAS-BENCH-101和NAS-BENCH-301等开源架构数据集验证了ObfuNAS的性能。实验结果表明，ObfuNAS能够在给定的FLOPS约束下成功地找到受害者模型的最优掩码，使得仅需0.14倍FLOPS开销的攻击者的推理准确率降低2.6%。代码可从以下网址获得：https://github.com/Tongzhou0101/ObfuNAS.



## **8. Learning to Generate Image Source-Agnostic Universal Adversarial Perturbations**

学习生成图像来源不可知的普遍对抗性扰动 cs.LG

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2009.13714v4)

**Authors**: Pu Zhao, Parikshit Ram, Songtao Lu, Yuguang Yao, Djallel Bouneffouf, Xue Lin, Sijia Liu

**Abstracts**: Adversarial perturbations are critical for certifying the robustness of deep learning models. A universal adversarial perturbation (UAP) can simultaneously attack multiple images, and thus offers a more unified threat model, obviating an image-wise attack algorithm. However, the existing UAP generator is underdeveloped when images are drawn from different image sources (e.g., with different image resolutions). Towards an authentic universality across image sources, we take a novel view of UAP generation as a customized instance of few-shot learning, which leverages bilevel optimization and learning-to-optimize (L2O) techniques for UAP generation with improved attack success rate (ASR). We begin by considering the popular model agnostic meta-learning (MAML) framework to meta-learn a UAP generator. However, we see that the MAML framework does not directly offer the universal attack across image sources, requiring us to integrate it with another meta-learning framework of L2O. The resulting scheme for meta-learning a UAP generator (i) has better performance (50% higher ASR) than baselines such as Projected Gradient Descent, (ii) has better performance (37% faster) than the vanilla L2O and MAML frameworks (when applicable), and (iii) is able to simultaneously handle UAP generation for different victim models and image data sources.

摘要: 对抗性扰动对于验证深度学习模型的稳健性至关重要。通用对抗扰动(UAP)可以同时攻击多幅图像，从而提供了更统一的威胁模型，从而避免了图像攻击算法。然而，当从不同的图像来源(例如，不同的图像分辨率)绘制图像时，现有的UAP生成器是不发达的。为了实现真正的跨图像来源的通用性，我们将UAP生成视为一个定制的少镜头学习实例，它利用双层优化和学习优化(L2O)技术来生成UAP，并提高了攻击成功率(ASR)。我们首先考虑流行的模型不可知元学习(MAML)框架来元学习UAP生成器。然而，我们看到MAML框架并没有直接提供跨图像源的通用攻击，需要我们将其与L2O的另一个元学习框架集成。所得到的用于元学习UAP生成器的方案(I)具有比诸如投影梯度下降之类的基线更好的性能(ASR高50%)，(Ii)具有比Vanilla L2O和MAML框架(如果适用)更好的性能(37%)，以及(Iii)能够同时处理不同受害者模型和图像数据源的UAP生成。



## **9. I-GWAS: Privacy-Preserving Interdependent Genome-Wide Association Studies**

I-GWAS：隐私保护、相互依赖的全基因组关联研究 q-bio.GN

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08361v1)

**Authors**: Túlio Pascoal, Jérémie Decouchant, Antoine Boutet, Marcus Völp

**Abstracts**: Genome-wide Association Studies (GWASes) identify genomic variations that are statistically associated with a trait, such as a disease, in a group of individuals. Unfortunately, careless sharing of GWAS statistics might give rise to privacy attacks. Several works attempted to reconcile secure processing with privacy-preserving releases of GWASes. However, we highlight that these approaches remain vulnerable if GWASes utilize overlapping sets of individuals and genomic variations. In such conditions, we show that even when relying on state-of-the-art techniques for protecting releases, an adversary could reconstruct the genomic variations of up to 28.6% of participants, and that the released statistics of up to 92.3% of the genomic variations would enable membership inference attacks. We introduce I-GWAS, a novel framework that securely computes and releases the results of multiple possibly interdependent GWASes. I-GWAScontinuously releases privacy-preserving and noise-free GWAS results as new genomes become available.

摘要: 全基因组关联研究(GWASes)确定在一组个体中与某种特征(如疾病)在统计上相关的基因组变异。不幸的是，粗心大意地分享GWAS统计数据可能会导致隐私攻击。有几部作品试图调和GWAS的安全处理和隐私保护版本之间的关系。然而，我们强调，如果GWAS利用重叠的个体集合和基因组变异，这些方法仍然容易受到攻击。在这种情况下，我们表明，即使依靠最先进的技术来保护释放，对手也可以重建高达28.6%的参与者的基因组变异，并且公布的高达92.3%的基因组变异的统计数据将使成员关系推理攻击成为可能。我们介绍了I-GWAS，这是一个新的框架，可以安全地计算和发布多个可能相互依赖的GWASs的结果。随着新基因组的出现，i-GWAS不断发布隐私保护和无噪音的GWAS结果。



## **10. Attackar: Attack of the Evolutionary Adversary**

攻击性：进化型对手的攻击 cs.CV

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08297v1)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textit{Attackar}, an evolutionary, score-based, black-box attack. Attackar is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate Attackar's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of Attackar, both in terms of accuracy score and query efficiency.

摘要: 深度神经网络(DNN)对各种场景中的敌意数据很敏感，包括黑盒场景，在这种场景中，攻击者只被允许查询训练的模型并接收输出。现有的创建对抗性实例的黑盒方法成本很高，通常使用梯度估计或训练替换网络。本文介绍了一种进化的、基于分数的黑盒攻击--Attackar。ATTACPAR基于一种新的目标函数，可用于无梯度优化问题。攻击只需要访问分类器的输出逻辑，因此不受梯度掩蔽的影响。不需要额外的信息，使我们的方法更适合实际情况。我们使用三个不同的最先进的模型--先启-v3、ResNet-50和VGG-16-BN--针对三个基准数据集：MNIST、CIFAR10和ImageNet测试其性能。此外，我们评估了Attackar在非差分变换防御和最新的稳健模型上的性能。实验结果表明，Attackar在查准率和查询效率方面都具有较好的性能。



## **11. On the Privacy Effect of Data Enhancement via the Lens of Memorization**

论记忆透镜增强数据的隐私效应 cs.LG

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08270v1)

**Authors**: Xiao Li, Qiongxiu Li, Zhanhao Hu, Xiaolin Hu

**Abstracts**: Machine learning poses severe privacy concerns as it is shown that the learned models can reveal sensitive information about their training data. Many works have investigated the effect of widely-adopted data augmentation (DA) and adversarial training (AT) techniques, termed data enhancement in the paper, on the privacy leakage of machine learning models. Such privacy effects are often measured by membership inference attacks (MIAs), which aim to identify whether a particular example belongs to the training set or not. We propose to investigate privacy from a new perspective called memorization. Through the lens of memorization, we find that previously deployed MIAs produce misleading results as they are less likely to identify samples with higher privacy risks as members compared to samples with low privacy risks. To solve this problem, we deploy a recent attack that can capture the memorization degrees of individual samples for evaluation. Through extensive experiments, we unveil non-trivial findings about the connections between three important properties of machine learning models, including privacy, generalization gap, and adversarial robustness. We demonstrate that, unlike existing results, the generalization gap is shown not highly correlated with privacy leakage. Moreover, stronger adversarial robustness does not necessarily imply that the model is more susceptible to privacy attacks.

摘要: 机器学习带来了严重的隐私问题，因为研究表明，学习的模型可能会泄露有关其训练数据的敏感信息。许多工作研究了广泛采用的数据增强(DA)和对抗训练(AT)技术对机器学习模型隐私泄露的影响。这种隐私影响通常通过成员关系推理攻击(MIA)来衡量，其目的是识别特定示例是否属于训练集。我们建议从一种名为记忆的新角度来研究隐私。通过记忆的镜头，我们发现之前部署的MIA产生了误导性的结果，因为与低隐私风险的样本相比，它们不太可能识别出隐私风险较高的样本作为成员。为了解决这个问题，我们部署了一个最近的攻击，可以捕获单个样本的记忆程度进行评估。通过广泛的实验，我们揭示了机器学习模型的三个重要属性之间的联系，包括隐私、泛化差距和对手健壮性之间的关系。我们证明，与已有结果不同，泛化差距与隐私泄露并不高度相关。此外，更强的对抗健壮性并不一定意味着该模型更容易受到隐私攻击。



## **12. Robustness of the Tangle 2.0 Consensus**

Tangle 2.0共识的健壮性 cs.DC

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08254v1)

**Authors**: Bing-Yang Lin, Daria Dziubałtowska, Piotr Macek, Andreas Penzkofer, Sebastian Müller

**Abstracts**: In this paper, we investigate the performance of the Tangle 2.0 consensus protocol in a Byzantine environment. We use an agent-based simulation model that incorporates the main features of the Tangle 2.0 consensus protocol. Our experimental results demonstrate that the Tangle 2.0 protocol is robust to the bait-and-switch attack up to the theoretical upper bound of the adversary's 33% voting weight. We further show that the common coin mechanism in Tangle 2.0 is necessary for robustness against powerful adversaries. Moreover, the experimental results confirm that the protocol can achieve around 1s confirmation time in typical scenarios and that the confirmation times of non-conflicting transactions are not affected by the presence of conflicts.

摘要: 本文研究了Tange2.0一致性协议在拜占庭环境下的性能。我们使用了一个基于代理的仿真模型，该模型结合了Tangel2.0共识协议的主要特征。实验结果表明，Tangel2.0协议对诱饵切换攻击具有较强的鲁棒性，达到了敌手33%投票权重的理论上限。我们进一步证明了Tange2.0中的普通硬币机制对于抵抗强大的对手是必要的。实验结果表明，该协议在典型场景下可以达到1s左右的确认时间，且无冲突事务的确认时间不受冲突的影响。



## **13. Two Heads are Better than One: Robust Learning Meets Multi-branch Models**

两个头总比一个头好：稳健学习遇到多分支模型 cs.CV

10 pages, 5 Figures

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08083v1)

**Authors**: Dong Huang, Qingwen Bu, Yuhao Qing, Haowen Pi, Sen Wang, Heming Cui

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial examples, in which DNNs are misled to false outputs due to inputs containing imperceptible perturbations. Adversarial training, a reliable and effective method of defense, may significantly reduce the vulnerability of neural networks and becomes the de facto standard for robust learning. While many recent works practice the data-centric philosophy, such as how to generate better adversarial examples or use generative models to produce additional training data, we look back to the models themselves and revisit the adversarial robustness from the perspective of deep feature distribution as an insightful complementarity. In this paper, we propose Branch Orthogonality adveRsarial Training (BORT) to obtain state-of-the-art performance with solely the original dataset for adversarial training. To practice our design idea of integrating multiple orthogonal solution spaces, we leverage a simple and straightforward multi-branch neural network that eclipses adversarial attacks with no increase in inference time. We heuristically propose a corresponding loss function, branch-orthogonal loss, to make each solution space of the multi-branch model orthogonal. We evaluate our approach on CIFAR-10, CIFAR-100, and SVHN against \ell_{\infty} norm-bounded perturbations of size \epsilon = 8/255, respectively. Exhaustive experiments are conducted to show that our method goes beyond all state-of-the-art methods without any tricks. Compared to all methods that do not use additional data for training, our models achieve 67.3% and 41.5% robust accuracy on CIFAR-10 and CIFAR-100 (improving upon the state-of-the-art by +7.23% and +9.07%). We also outperform methods using a training set with a far larger scale than ours. All our models and codes are available online at https://github.com/huangd1999/BORT.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响，在这些例子中，DNN的输入含有不可察觉的扰动，容易被误导到错误的输出。对抗性训练是一种可靠而有效的防御方法，可以显著降低神经网络的脆弱性，成为稳健学习的事实标准。虽然最近的许多工作实践了以数据为中心的哲学，例如如何生成更好的对抗性示例或使用生成性模型来生成额外的训练数据，但我们回顾了模型本身，并从深层特征分布的角度重新审视了对抗性健壮性，作为一种有洞察力的补充。在本文中，我们提出了分支正交性对抗训练(BORT)，以获得最先进的性能，仅使用原始数据集进行对抗训练。为了实践我们集成多个正交解空间的设计思想，我们利用了一个简单而直接的多分支神经网络，该网络在不增加推理时间的情况下克服了对手攻击。我们启发式地提出了相应的损失函数--分支正交损失，使多分支模型的每个解空间正交化。我们在CIFAR-10、CIFAR-100和SVHN上分别评估了我们的方法对大小为8/255的范数有界扰动的影响。详尽的实验表明，我们的方法超越了所有最先进的方法，没有任何技巧。与所有不使用额外数据进行训练的方法相比，我们的模型在CIFAR-10和CIFAR-100上分别获得了67.3%和41.5%的稳健准确率(分别比最先进的方法提高了7.23%和9.07%)。我们也比使用比我们规模大得多的训练集的方法表现得更好。我们所有的型号和代码都可以在https://github.com/huangd1999/BORT.上在线获得



## **14. An Efficient Multi-Step Framework for Malware Packing Identification**

一种高效的恶意软件包装识别的多步框架 cs.CR

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08071v1)

**Authors**: Jong-Wouk Kim, Yang-Sae Moon, Mi-Jung Choi

**Abstracts**: Malware developers use combinations of techniques such as compression, encryption, and obfuscation to bypass anti-virus software. Malware with anti-analysis technologies can bypass AI-based anti-virus software and malware analysis tools. Therefore, classifying pack files is one of the big challenges. Problems arise if the malware classifiers learn packers' features, not those of malware. Training the models with unintended erroneous data turn into poisoning attacks, adversarial attacks, and evasion attacks. Therefore, researchers should consider packing to build appropriate malware classifier models. In this paper, we propose a multi-step framework for classifying and identifying packed samples which consists of pseudo-optimal feature selection, machine learning-based classifiers, and packer identification steps. In the first step, we use the CART algorithm and the permutation importance to preselect important 20 features. In the second step, each model learns 20 preselected features for classifying the packed files with the highest performance. As a result, the XGBoost, which learned the features preselected by XGBoost with the permutation importance, showed the highest performance of any other experiment scenarios with an accuracy of 99.67%, an F1-Score of 99.46%, and an area under the curve (AUC) of 99.98%. In the third step, we propose a new approach that can identify packers only for samples classified as Well-Known Packed.

摘要: 恶意软件开发人员使用压缩、加密和混淆等技术组合绕过反病毒软件。带有反分析技术的恶意软件可以绕过基于AI的反病毒软件和恶意软件分析工具。因此，对包文件进行分类是最大的挑战之一。如果恶意软件分类器学习打包程序的功能，而不是恶意软件的功能，就会出现问题。用意想不到的错误数据训练模型会变成中毒攻击、对抗性攻击和逃避攻击。因此，研究人员应该考虑打包构建合适的恶意软件分类器模型。本文提出了一种多步骤分类识别包装样本的框架，该框架由伪最优特征选择、基于机器学习的分类器和包装者识别步骤组成。在第一步中，我们使用CART算法和排列重要度对20个重要特征进行预选。在第二步中，每个模型学习20个预先选择的特征，用于分类性能最高的压缩文件。因此，XGBoost学习了XGBoost预先选择的具有排列重要性的特征，在所有其他实验场景中表现出最高的性能，准确率为99.67%，F1得分为99.46%，曲线下面积(AUC)为99.98%。在第三步中，我们提出了一种新的方法，该方法只对被归类为已知包装的样品进行识别。



## **15. Adversarial Prefetch: New Cross-Core Cache Side Channel Attacks**

对抗性预取：新的跨核心缓存侧通道攻击 cs.CR

camera-ready for IEEE S&P 2022

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2110.12340v3)

**Authors**: Yanan Guo, Andrew Zigerelli, Youtao Zhang, Jun Yang

**Abstracts**: Modern x86 processors have many prefetch instructions that can be used by programmers to boost performance. However, these instructions may also cause security problems. In particular, we found that on Intel processors, there are two security flaws in the implementation of PREFETCHW, an instruction for accelerating future writes. First, this instruction can execute on data with read-only permission. Second, the execution time of this instruction leaks the current coherence state of the target data.   Based on these two design issues, we build two cross-core private cache attacks that work with both inclusive and non-inclusive LLCs, named Prefetch+Reload and Prefetch+Prefetch. We demonstrate the significance of our attacks in different scenarios. First, in the covert channel case, Prefetch+Reload and Prefetch+Prefetch achieve 782 KB/s and 822 KB/s channel capacities, when using only one shared cache line between the sender and receiver, the largest-to-date single-line capacities for CPU cache covert channels. Further, in the side channel case, our attacks can monitor the access pattern of the victim on the same processor, with almost zero error rate. We show that they can be used to leak private information of real-world applications such as cryptographic keys. Finally, our attacks can be used in transient execution attacks in order to leak more secrets within the transient window than prior work. From the experimental results, our attacks allow leaking about 2 times as many secret bytes, compared to Flush+Reload, which is widely used in transient execution attacks.

摘要: 现代x86处理器有许多预取指令，程序员可以使用这些指令来提高性能。然而，这些说明也可能导致安全问题。特别是，我们发现在Intel处理器上，PREFETCHW的实现存在两个安全缺陷，PREFETCHW是一种用于加速未来写入的指令。首先，此指令可以在具有只读权限的数据上执行。其次，此指令的执行时间会泄漏目标数据的当前一致性状态。基于这两个设计问题，我们构建了两种同时适用于包含性和非包含性LLC的跨核私有缓存攻击，分别称为预取+重新加载和预取+预取。我们在不同的情况下展示了我们的攻击的重要性。首先，在隐蔽通道的情况下，当在发送器和接收器之间仅使用一个共享高速缓存线时，预取+重新加载和预取+预取分别达到了782Kb/s和822Kb/s的通道容量，这是迄今为止用于CPU高速缓存隐蔽通道的最大单线容量。此外，在旁信道情况下，我们的攻击可以监测受害者在同一处理器上的访问模式，几乎没有误码率。我们证明了它们可以被用来泄露真实世界应用程序的私人信息，例如密钥。最后，我们的攻击可以用于瞬时执行攻击，以便在瞬时窗口内泄漏比以前的工作更多的秘密。从实验结果来看，与瞬时执行攻击中广泛使用的刷新+重新加载相比，我们的攻击允许泄漏大约2倍的秘密字节。



## **16. A Context-Aware Approach for Textual Adversarial Attack through Probability Difference Guided Beam Search**

一种上下文感知的概率差分导引束搜索文本攻击方法 cs.CL

**SubmitDate**: 2022-08-17    [paper-pdf](http://arxiv.org/pdf/2208.08029v1)

**Authors**: Huijun Liu, Jie Yu, Shasha Li, Jun Ma, Bin Ji

**Abstracts**: Textual adversarial attacks expose the vulnerabilities of text classifiers and can be used to improve their robustness. Existing context-aware methods solely consider the gold label probability and use the greedy search when searching an attack path, often limiting the attack efficiency. To tackle these issues, we propose PDBS, a context-aware textual adversarial attack model using Probability Difference guided Beam Search. The probability difference is an overall consideration of all class label probabilities, and PDBS uses it to guide the selection of attack paths. In addition, PDBS uses the beam search to find a successful attack path, thus avoiding suffering from limited search space. Extensive experiments and human evaluation demonstrate that PDBS outperforms previous best models in a series of evaluation metrics, especially bringing up to a +19.5% attack success rate. Ablation studies and qualitative analyses further confirm the efficiency of PDBS.

摘要: 文本对抗性攻击暴露了文本分类器的弱点，可以用来提高其健壮性。现有的上下文感知方法只考虑黄金标签概率，在搜索攻击路径时采用贪婪搜索，往往限制了攻击效率。为了解决这些问题，我们提出了一种基于概率差导束搜索的上下文感知文本对抗攻击模型PDBS。概率差是对所有类别标签概率的综合考虑，PDBS用它来指导攻击路径的选择。此外，PDBS使用波束搜索来找到一条成功的攻击路径，从而避免了搜索空间有限的问题。广泛的实验和人工评估表明，PDBS在一系列评估指标上优于之前的最佳模型，特别是攻击成功率高达19.5%。烧蚀研究和定性分析进一步证实了PDBS的有效性。



## **17. FRL: Federated Rank Learning**

FRL：联合等级学习 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2110.04350v3)

**Authors**: Hamid Mozaffari, Virat Shejwalkar, Amir Houmansadr

**Abstracts**: Federated learning (FL) allows mutually untrusted clients to collaboratively train a common machine learning model without sharing their private/proprietary training data among each other. FL is unfortunately susceptible to poisoning by malicious clients who aim to hamper the accuracy of the commonly trained model through sending malicious model updates during FL's training process.   We argue that the key factor to the success of poisoning attacks against existing FL systems is the large space of model updates available to the clients, allowing malicious clients to search for the most poisonous model updates, e.g., by solving an optimization problem. To address this, we propose Federated Rank Learning (FRL). FRL reduces the space of client updates from model parameter updates (a continuous space of float numbers) in standard FL to the space of parameter rankings (a discrete space of integer values). To be able to train the global model using parameter ranks (instead of parameter weights), FRL leverage ideas from recent supermasks training mechanisms. Specifically, FRL clients rank the parameters of a randomly initialized neural network (provided by the server) based on their local training data. The FRL server uses a voting mechanism to aggregate the parameter rankings submitted by clients in each training epoch to generate the global ranking of the next training epoch.   Intuitively, our voting-based aggregation mechanism prevents poisoning clients from making significant adversarial modifications to the global model, as each client will have a single vote! We demonstrate the robustness of FRL to poisoning through analytical proofs and experimentation. We also show FRL's high communication efficiency. Our experiments demonstrate the superiority of FRL in real-world FL settings.

摘要: 联合学习(FL)允许相互不信任的客户端协作地训练通用的机器学习模型，而无需彼此共享他们的私有/专有训练数据。不幸的是，FL很容易受到恶意客户的毒害，这些客户的目的是通过在FL的训练过程中发送恶意模型更新来阻碍通常训练的模型的准确性。我们认为，针对现有FL系统的毒化攻击成功的关键因素是客户端可以获得巨大的模型更新空间，允许恶意客户端通过解决优化问题来搜索最有害的模型更新。为了解决这个问题，我们提出了联合等级学习(FRL)。FRL将客户端更新的空间从标准FL中的模型参数更新(浮点数的连续空间)减少到参数排名的空间(整数值的离散空间)。为了能够使用参数等级(而不是参数权重)来训练全局模型，FRL利用了最近超级掩码训练机制的想法。具体地说，FRL客户端基于其本地训练数据对随机初始化的神经网络(由服务器提供)的参数进行排名。FRL服务器使用投票机制来聚合客户端在每个训练时段中提交的参数排名，以生成下一个训练时段的全局排名。直观地说，我们基于投票的聚合机制可以防止中毒客户对全局模型进行重大的对抗性修改，因为每个客户都有一张选票！我们通过分析证明和实验证明了FRL对中毒的稳健性。我们还展示了FRL的高通信效率。我们的实验证明了FRL在真实的外语环境中的优越性。



## **18. Adversarial Image Color Transformations in Explicit Color Filter Space**

显式滤色空间中的对抗性图像颜色变换 cs.CV

Code is available at  https://github.com/ZhengyuZhao/ACE/tree/master/Journal_version. This work has  been submitted to the IEEE for possible publication. Copyright may be  transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2011.06690v2)

**Authors**: Zhengyu Zhao, Zhuoran Liu, Martha Larson

**Abstracts**: Deep Neural Networks have been shown to be vulnerable to adversarial images. Conventional attacks strive for indistinguishable adversarial images with strictly restricted perturbations. Recently, researchers have moved to explore distinguishable yet non-suspicious adversarial images and demonstrated that color transformation attacks are effective. In this work, we propose Adversarial Color Filter (AdvCF), a novel color transformation attack that is optimized with gradient information in the parameter space of a simple color filter. In particular, our color filter space is explicitly specified so that we are able to provide a systematic analysis of model robustness against adversarial color transformations, from both the attack and defense perspectives. In contrast, existing color transformation attacks do not offer the opportunity for systematic analysis due to the lack of such an explicit space. We further conduct extensive comparisons between different color transformation attacks on both the success rate and image acceptability, through a user study. Additional results provide interesting new insights into model robustness against AdvCF in another three visual tasks. We also highlight the human-interpretability of AdvCF, which is promising in practical use scenarios, and show its superiority over the state-of-the-art human-interpretable color transformation attack on both the image acceptability and efficiency.

摘要: 深度神经网络已被证明容易受到敌意图像的影响。传统的攻击努力获得带有严格限制的扰动的难以区分的对抗性图像。最近，研究人员开始探索可区分但不可疑的对抗性图像，并证明了颜色变换攻击是有效的。在这项工作中，我们提出了对抗颜色过滤器(AdvCF)，这是一种新的颜色变换攻击，它利用简单颜色过滤器参数空间中的梯度信息进行优化。特别是，我们的滤色器空间被明确指定，以便我们能够从攻击和防御两个角度提供针对对抗性颜色变换的模型稳健性的系统分析。相比之下，现有的颜色变换攻击由于缺乏这种明确的空间而没有提供系统分析的机会。通过一项用户研究，我们进一步对不同的颜色变换攻击在成功率和图像可接受性方面进行了广泛的比较。其他结果为在另外三个可视化任务中针对AdvCF的模型健壮性提供了有趣的新见解。我们还强调了AdvCF的人类可解释性，这在实际应用场景中是很有前途的，并展示了它在图像可接受性和效率上比最新的人类可解释颜色变换攻击的优越性。



## **19. FedPerm: Private and Robust Federated Learning by Parameter Permutation**

FedPerm：基于参数置换的私有健壮联合学习 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2208.07922v1)

**Authors**: Hamid Mozaffari, Virendra J. Marathe, Dave Dice

**Abstracts**: Federated Learning (FL) is a distributed learning paradigm that enables mutually untrusting clients to collaboratively train a common machine learning model. Client data privacy is paramount in FL. At the same time, the model must be protected from poisoning attacks from adversarial clients. Existing solutions address these two problems in isolation. We present FedPerm, a new FL algorithm that addresses both these problems by combining a novel intra-model parameter shuffling technique that amplifies data privacy, with Private Information Retrieval (PIR) based techniques that permit cryptographic aggregation of clients' model updates. The combination of these techniques further helps the federation server constrain parameter updates from clients so as to curtail effects of model poisoning attacks by adversarial clients. We further present FedPerm's unique hyperparameters that can be used effectively to trade off computation overheads with model utility. Our empirical evaluation on the MNIST dataset demonstrates FedPerm's effectiveness over existing Differential Privacy (DP) enforcement solutions in FL.

摘要: 联合学习(FL)是一种分布式学习范例，它使相互不信任的客户能够协作地训练通用的机器学习模型。在FL中，客户数据隐私是最重要的。与此同时，该模型必须受到保护，不受敌对客户的毒害攻击。现有的解决方案孤立地解决了这两个问题。我们提出了一种新的FL算法FedPerm，它通过结合一种新的模型内参数改组技术来放大数据隐私，以及基于私人信息检索(PIR)的技术，允许对客户的模型更新进行加密聚合，从而解决了这两个问题。这些技术的组合进一步帮助联合服务器限制来自客户端的参数更新，以减少敌对客户端的模型中毒攻击的影响。我们进一步介绍了FedPerm独特的超参数，这些参数可以有效地用来权衡模型实用程序的计算开销。我们在MNIST数据集上的经验评估表明，FedPerm在FL中比现有的差异隐私(DP)实施解决方案更有效。



## **20. Adversarial Relighting Against Face Recognition**

对抗人脸识别的对抗性重发 cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2108.07920v3)

**Authors**: Ruijun Gao, Qing Guo, Qian Zhang, Felix Juefei-Xu, Hongkai Yu, Wei Feng

**Abstracts**: Deep face recognition (FR) has achieved significantly high accuracy on several challenging datasets and fosters successful real-world applications, even showing high robustness to the illumination variation that is usually regarded as a main threat to the FR system. However, in the real world, illumination variation caused by diverse lighting conditions cannot be fully covered by the limited face dataset. In this paper, we study the threat of lighting against FR from a new angle, i.e., adversarial attack, and identify a new task, i.e., adversarial relighting. Given a face image, adversarial relighting aims to produce a naturally relighted counterpart while fooling the state-of-the-art deep FR methods. To this end, we first propose the physical model-based adversarial relighting attack (ARA) denoted as albedo-quotient-based adversarial relighting attack (AQ-ARA). It generates natural adversarial light under the physical lighting model and guidance of FR systems and synthesizes adversarially relighted face images. Moreover, we propose the auto-predictive adversarial relighting attack (AP-ARA) by training an adversarial relighting network (ARNet) to automatically predict the adversarial light in a one-step manner according to different input faces, allowing efficiency-sensitive applications. More importantly, we propose to transfer the above digital attacks to physical ARA (Phy-ARA) through a precise relighting device, making the estimated adversarial lighting condition reproducible in the real world. We validate our methods on three state-of-the-art deep FR methods, i.e., FaceNet, ArcFace, and CosFace, on two public datasets. The extensive and insightful results demonstrate our work can generate realistic adversarial relighted face images fooling FR easily, revealing the threat of specific light directions and strengths.

摘要: 深度人脸识别(FR)已经在几个具有挑战性的数据集上取得了显著的高精度，并促进了现实世界的成功应用，甚至对通常被视为FR系统主要威胁的光照变化表现出高度的稳健性。然而，在现实世界中，有限的人脸数据集不能完全覆盖由于光照条件的变化而引起的光照变化。本文从对抗性攻击这一新的角度研究了闪电对火箭弹的威胁，并提出了一种新的任务，即对抗性重发。给定脸部图像，对抗性重光旨在产生自然重光的对应物，同时愚弄最先进的深度FR方法。为此，我们首先提出了基于物理模型的对抗性重亮攻击(ARA)，称为基于反照率商的对抗性重亮攻击(AQ-ARA)。它在物理照明模型和FR系统的指导下产生自然的对抗性光，并合成对抗性重光的人脸图像。此外，我们提出了自动预测对抗性重光攻击(AP-ARA)，通过训练对抗性重光网络(ARNet)来根据不同的输入人脸一步自动预测对抗性光，从而允许对效率敏感的应用。更重要的是，我们建议通过精确的重新照明装置将上述数字攻击转移到物理ARA(Phy-ARA)，使估计的对抗性照明条件在现实世界中可重现。在两个公开的数据集上，我们在三种最先进的深度FR方法，即FaceNet，ArcFace和CosFace上对我们的方法进行了验证。广泛而有洞察力的结果表明，我们的工作可以生成现实的对抗性重光照人脸图像，轻松愚弄FR，揭示特定光方向和强度的威胁。



## **21. StratDef: a strategic defense against adversarial attacks in malware detection**

StratDef：恶意软件检测中对抗对手攻击的战略防御 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2202.07568v3)

**Authors**: Aqib Rashid, Jose Such

**Abstracts**: Over the years, most research towards defenses against adversarial attacks on machine learning models has been in the image recognition domain. The malware detection domain has received less attention despite its importance. Moreover, most work exploring these defenses has focused on several methods but with no strategy when applying them. In this paper, we introduce StratDef, which is a strategic defense system tailored for the malware detection domain based on a moving target defense approach. We overcome challenges related to the systematic construction, selection and strategic use of models to maximize adversarial robustness. StratDef dynamically and strategically chooses the best models to increase the uncertainty for the attacker, whilst minimizing critical aspects in the adversarial ML domain like attack transferability. We provide the first comprehensive evaluation of defenses against adversarial attacks on machine learning for malware detection, where our threat model explores different levels of threat, attacker knowledge, capabilities, and attack intensities. We show that StratDef performs better than other defenses even when facing the peak adversarial threat. We also show that, from the existing defenses, only a few adversarially-trained models provide substantially better protection than just using vanilla models but are still outperformed by StratDef.

摘要: 多年来，针对机器学习模型的敌意攻击防御的研究大多集中在图像识别领域。恶意软件检测领域尽管很重要，但受到的关注较少。此外，大多数探索这些防御措施的工作都集中在几种方法上，但在应用这些方法时没有策略。本文介绍了StratDef，这是一个针对恶意软件检测领域定制的基于运动目标防御方法的战略防御系统。我们克服了与模型的系统构建、选择和战略使用相关的挑战，以最大限度地提高对手的稳健性。StratDef动态和战略性地选择最佳模型以增加攻击者的不确定性，同时最小化敌对ML领域中的关键方面，如攻击可转移性。我们提供了针对恶意软件检测的机器学习的首次全面防御评估，其中我们的威胁模型探索了不同级别的威胁、攻击者知识、能力和攻击强度。我们表明，即使在面临最大的对手威胁时，StratDef也比其他防御系统表现得更好。我们还表明，从现有的防御措施来看，只有少数经过对抗性训练的模型提供了比只使用普通模型更好的保护，但仍然优于StratDef。



## **22. A Physical-World Adversarial Attack for 3D Face Recognition**

一种面向3D人脸识别的物理世界对抗性攻击 cs.CV

7 pages, 5 figures, Submit to AAAI 2023

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2205.13412v2)

**Authors**: Yanjie Li, Yiquan Li, Xuelong Dai, Songtao Guo, Bin Xiao

**Abstracts**: The 3D face recognition has long been considered secure for its resistance to current physical adversarial attacks, like adversarial patches. However, this paper shows that a 3D face recognition system can be easily attacked, leading to evading and impersonation attacks. We are the first to propose a physically realizable attack for the 3D face recognition system, named structured light imaging attack (SLIA), which exploits the weakness of structured-light-based 3D scanning devices. SLIA utilizes the projector in the structured light imaging system to create adversarial illuminations to contaminate the reconstructed point cloud. Firstly, we propose a 3D transform-invariant loss function (3D-TI) to generate adversarial perturbations that are more robust to head movements. Then we integrate the 3D imaging process into the attack optimization, which minimizes the total pixel shifting of fringe patterns. We realize both dodging and impersonation attacks on a real-world 3D face recognition system. Our methods need fewer modifications on projected patterns compared with Chamfer and Chamfer+kNN-based methods and achieve average attack success rates of 0.47 (impersonation) and 0.89 (dodging). This paper exposes the insecurity of present structured light imaging technology and sheds light on designing secure 3D face recognition authentication systems.

摘要: 长期以来，3D人脸识别一直被认为是安全的，因为它能抵抗当前的物理对抗性攻击，比如对抗性补丁。然而，本文指出，3D人脸识别系统很容易受到攻击，从而导致规避和冒充攻击。我们首次提出了一种针对3D人脸识别系统的物理可实现的攻击，称为结构光成像攻击(SLIA)，它利用了基于结构光的3D扫描设备的弱点。SLIA利用结构光成像系统中的投影仪来产生对抗性照明来污染重建的点云。首先，我们提出了一种3D变换不变损失函数(3D-TI)来产生对抗性扰动，该扰动对头部运动具有更强的鲁棒性。然后，我们将3D成像过程整合到攻击优化中，使条纹图案的总像素漂移最小化。我们在一个真实的3D人脸识别系统上实现了躲避攻击和模仿攻击。与基于Chamfer和Chamfer+KNN的方法相比，我们的方法需要对投影模式进行更少的修改，并且获得了0.47(模仿)和0.89(躲避)的平均攻击成功率。本文揭示了目前结构光成像技术的不足，为设计安全的三维人脸识别认证系统提供了参考。



## **23. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2008.09312v2)

**Authors**: Shiliang Zuo

**Abstracts**: We consider a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. We propose a novel attack strategy that manipulates a UCB principle into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\sqrt{\log T}$, where $T$ is the number of rounds. We also prove the first lower bound on the cumulative attack cost. Our lower bound matches our upper bound up to $\log \log T$ factors, showing our attack to be near optimal.

摘要: 我们考虑了一个随机多臂强盗问题，其中报酬服从对抗性腐败。我们提出了一种新的攻击策略，它利用UCB原理来拉动一些非最优目标臂$T-o(T)$次，累积代价可扩展到$\Sqrt{\log T}$，其中$T$是轮数。我们还证明了累积攻击代价的第一个下界。我们的下界与上界匹配，最高可达$\log\log T$因子，表明我们的攻击接近最优。



## **24. Defending Black-box Skeleton-based Human Activity Classifiers**

基于黑盒骨架防御的人类活动分类器 cs.CV

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2203.04713v3)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstracts**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks.

摘要: 骨骼运动在人类活动识别(HAR)中得到了广泛的应用。最近，基于骨架的HAR在各种分类器和数据中发现了一个普遍的漏洞，需要缓解。为此，我们提出了第一种基于骨架的HAR黑盒防御方法。我们的方法的特点是对干净数据、对手和分类器进行全面的贝叶斯处理，导致(1)新的基于贝叶斯能量的稳健判别分类器的形成，(2)基于自然运动流形的新的对手采样方案，(3)新的训练后贝叶斯策略用于黑盒防御。我们将我们的框架命名为基于贝叶斯能量的对抗性训练或BEAT。BEAT是简单但优雅的，它将脆弱的黑匣子分类器变成了健壮的分类器，而不会牺牲准确性。在各种攻击下，它在广泛的骨架HAR分类器和数据集上展示了令人惊讶的和普遍的有效性。



## **25. CTI4AI: Threat Intelligence Generation and Sharing after Red Teaming AI Models**

CTI4AI：Red Teaming AI模型后威胁情报的生成和共享 cs.CR

**SubmitDate**: 2022-08-16    [paper-pdf](http://arxiv.org/pdf/2208.07476v1)

**Authors**: Chuyen Nguyen, Caleb Morgan, Sudip Mittal

**Abstracts**: As the practicality of Artificial Intelligence (AI) and Machine Learning (ML) based techniques grow, there is an ever increasing threat of adversarial attacks. There is a need to red team this ecosystem to identify system vulnerabilities, potential threats, characterize properties that will enhance system robustness, and encourage the creation of effective defenses. A secondary need is to share this AI security threat intelligence between different stakeholders like, model developers, users, and AI/ML security professionals. In this paper, we create and describe a prototype system CTI4AI, to overcome the need to methodically identify and share AI/ML specific vulnerabilities and threat intelligence.

摘要: 随着人工智能(AI)和机器学习(ML)技术的实用化程度的提高，对手攻击的威胁越来越大。有必要对这一生态系统进行红色团队合作，以识别系统漏洞和潜在威胁，确定将增强系统健壮性的特性，并鼓励创建有效的防御措施。第二个需求是在不同的利益相关者之间共享这种AI安全威胁情报，比如模型开发人员、用户和AI/ML安全专业人员。在本文中，我们创建并描述了一个原型系统CTI4AI，以克服有条不紊地识别和共享AI/ML特定漏洞和威胁情报的需要。



## **26. MENLI: Robust Evaluation Metrics from Natural Language Inference**

MENLI：自然语言推理中的稳健评价指标 cs.CL

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07316v1)

**Authors**: Yanran Chen, Steffen Eger

**Abstracts**: Recently proposed BERT-based evaluation metrics perform well on standard evaluation benchmarks but are vulnerable to adversarial attacks, e.g., relating to factuality errors. We argue that this stems (in part) from the fact that they are models of semantic similarity. In contrast, we develop evaluation metrics based on Natural Language Inference (NLI), which we deem a more appropriate modeling. We design a preference-based adversarial attack framework and show that our NLI based metrics are much more robust to the attacks than the recent BERT-based metrics. On standard benchmarks, our NLI based metrics outperform existing summarization metrics, but perform below SOTA MT metrics. However, when we combine existing metrics with our NLI metrics, we obtain both higher adversarial robustness (+20% to +30%) and higher quality metrics as measured on standard benchmarks (+5% to +25%).

摘要: 最近提出的基于BERT的评估指标在标准评估基准上表现良好，但容易受到敌意攻击，例如与真实性错误有关的攻击。我们认为，这(部分)源于这样一个事实：它们是语义相似性的模型。相比之下，我们基于自然语言推理(NLI)开发评估指标，我们认为这是更合适的建模。我们设计了一个基于偏好的对抗性攻击框架，并表明我们的基于NLI的度量比最近的基于BERT的度量具有更强的抗攻击能力。在标准基准测试中，我们基于NLI的指标优于现有的摘要指标，但低于SOTA MT指标。然而，当我们将现有指标与我们的NLI指标相结合时，我们获得了更高的对抗性健壮性(+20%至+30%)和标准基准测试的更高质量指标(+5%至+25%)。



## **27. Bounding Membership Inference**

边界隶属度推理 cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2202.12232v3)

**Authors**: Anvith Thudi, Ilia Shumailov, Franziska Boenisch, Nicolas Papernot

**Abstracts**: Differential Privacy (DP) is the de facto standard for reasoning about the privacy guarantees of a training algorithm. Despite the empirical observation that DP reduces the vulnerability of models to existing membership inference (MI) attacks, a theoretical underpinning as to why this is the case is largely missing in the literature. In practice, this means that models need to be trained with DP guarantees that greatly decrease their accuracy. In this paper, we provide a tighter bound on the positive accuracy (i.e., attack precision) of any MI adversary when a training algorithm provides $\epsilon$-DP or $(\epsilon, \delta)$-DP. Our bound informs the design of a novel privacy amplification scheme, where an effective training set is sub-sampled from a larger set prior to the beginning of training, to greatly reduce the bound on MI accuracy. As a result, our scheme enables DP users to employ looser DP guarantees when training their model to limit the success of any MI adversary; this ensures that the model's accuracy is less impacted by the privacy guarantee. Finally, we discuss implications of our MI bound on the field of machine unlearning.

摘要: 差分隐私(DP)是对训练算法的隐私保证进行推理的事实标准。尽管经验观察表明DP降低了模型对现有成员推理(MI)攻击的脆弱性，但文献中很大程度上缺乏关于为什么会这样的理论基础。在实践中，这意味着需要用DP保证来训练模型，这会大大降低它们的准确性。在本文中，我们给出了当训练算法提供$\epsilon$-DP或$(\epsilon，\Delta)$-DP时，MI对手的正确率(即攻击精度)的一个更严格的界。我们的界提供了一种新的隐私放大方案的设计，其中有效的训练集在训练开始之前从较大的集合中被亚采样，以极大地降低对MI准确率的界。因此，我们的方案允许DP用户在训练他们的模型时采用更宽松的DP保证来限制任何MI对手的成功；这确保了模型的准确性较少地受到隐私保证的影响。最后，我们讨论了我们的MI界在机器遗忘领域的意义。



## **28. Man-in-the-Middle Attack against Object Detection Systems**

针对目标检测系统的中间人攻击 cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.07174v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Is deep learning secure for robots? As embedded systems have access to more powerful CPUs and GPUs, deep-learning-enabled object detection systems become pervasive in robotic applications. Meanwhile, prior research unveils that deep learning models are vulnerable to adversarial attacks. Does this put real-world robots at threat? Our research borrows the idea of the Main-in-the-Middle attack from Cryptography to attack an object detection system. Our experimental results prove that we can generate a strong Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. Our findings raise a serious concern over the applications of deep learning models in safety-critical systems such as autonomous driving.

摘要: 深度学习对机器人来说安全吗？随着嵌入式系统能够获得更强大的CPU和GPU，支持深度学习的目标检测系统在机器人应用中变得普遍。与此同时，先前的研究表明，深度学习模型很容易受到对手的攻击。这会对现实世界的机器人构成威胁吗？我们的研究借用了密码学中的中间主攻击的思想来攻击目标检测系统。我们的实验结果证明，我们可以在一分钟内产生一个强的通用对抗扰动(UAP)，然后利用该扰动通过中间人攻击来攻击检测系统。我们的发现引发了人们对深度学习模型在自动驾驶等安全关键系统中的应用的严重担忧。



## **29. GUARD: Graph Universal Adversarial Defense**

后卫：GRAPH通用对抗性防御 cs.LG

Preprint. Code is publicly available at  https://github.com/EdisonLeeeee/GUARD

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2204.09803v3)

**Authors**: Jintang Li, Jie Liao, Ruofan Wu, Liang Chen, Jiawang Dan, Changhua Meng, Zibin Zheng, Weiqiang Wang

**Abstracts**: Graph convolutional networks (GCNs) have shown to be vulnerable to small adversarial perturbations, which becomes a severe threat and largely limits their applications in security-critical scenarios. To mitigate such a threat, considerable research efforts have been devoted to increasing the robustness of GCNs against adversarial attacks. However, current approaches for defense are typically designed for the whole graph and consider the global performance, posing challenges in protecting important local nodes from stronger adversarial targeted attacks. In this work, we present a simple yet effective method, named Graph Universal Adversarial Defense (GUARD). Unlike previous works, GUARD protects each individual node from attacks with a universal defensive patch, which is generated once and can be applied to any node (node-agnostic) in a graph. Extensive experiments on four benchmark datasets demonstrate that our method significantly improves robustness for several established GCNs against multiple adversarial attacks and outperforms state-of-the-art defense methods by large margins. Our code is publicly available at https://github.com/EdisonLeeeee/GUARD.

摘要: 图卷积网络(GCNS)容易受到微小的敌意扰动，这是一种严重的威胁，在很大程度上限制了它们在安全关键场景中的应用。为了减轻这种威胁，人们投入了大量的研究努力来提高GCNS对对手攻击的健壮性。然而，当前的防御方法通常是为整个图设计的，并考虑了全局性能，这给保护重要的局部节点免受更强的对抗性目标攻击带来了挑战。在这项工作中，我们提出了一种简单而有效的方法，称为图通用对抗防御(GARD)。与以前的工作不同，Guard使用一个通用的防御补丁来保护每个单独的节点免受攻击，该补丁只生成一次，可以应用于图中的任何节点(与节点无关)。在四个基准数据集上的大量实验表明，我们的方法显着提高了几个已建立的GCN对多个对手攻击的稳健性，并且远远超过了最先进的防御方法。我们的代码在https://github.com/EdisonLeeeee/GUARD.上公开提供



## **30. A Multi-objective Memetic Algorithm for Auto Adversarial Attack Optimization Design**

自动对抗性攻击优化设计的多目标Memtic算法 cs.CV

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06984v1)

**Authors**: Jialiang Sun, Wen Yao, Tingsong Jiang, Xiaoqian Chen

**Abstracts**: The phenomenon of adversarial examples has been revealed in variant scenarios. Recent studies show that well-designed adversarial defense strategies can improve the robustness of deep learning models against adversarial examples. However, with the rapid development of defense technologies, it also tends to be more difficult to evaluate the robustness of the defensed model due to the weak performance of existing manually designed adversarial attacks. To address the challenge, given the defensed model, the efficient adversarial attack with less computational burden and lower robust accuracy is needed to be further exploited. Therefore, we propose a multi-objective memetic algorithm for auto adversarial attack optimization design, which realizes the automatical search for the near-optimal adversarial attack towards defensed models. Firstly, the more general mathematical model of auto adversarial attack optimization design is constructed, where the search space includes not only the attacker operations, magnitude, iteration number, and loss functions but also the connection ways of multiple adversarial attacks. In addition, we develop a multi-objective memetic algorithm combining NSGA-II and local search to solve the optimization problem. Finally, to decrease the evaluation cost during the search, we propose a representative data selection strategy based on the sorting of cross entropy loss values of each images output by models. Experiments on CIFAR10, CIFAR100, and ImageNet datasets show the effectiveness of our proposed method.

摘要: 对抗性例子的现象在不同的情景中被揭示出来。最近的研究表明，设计好的对抗性防御策略可以提高深度学习模型对对抗性例子的稳健性。然而，随着防御技术的快速发展，由于现有人工设计的对抗性攻击性能较弱，评估防御模型的稳健性也变得更加困难。为了应对这一挑战，在防御模型的情况下，需要进一步开发计算负担较小、鲁棒性较低的高效对抗性攻击。为此，本文提出了一种自动对抗性攻击优化设计的多目标迷因算法，实现了对防御模型的近优对抗性攻击的自动搜索。首先，建立了更一般的自动对抗攻击优化设计数学模型，该模型的搜索空间不仅包括攻击操作、规模、迭代次数和损失函数，还包括多个对抗攻击的连接方式。此外，我们还提出了一种结合NSGA-II和局部搜索的多目标模因算法来解决该优化问题。最后，为了降低搜索过程中的评价代价，提出了一种基于模型对每幅图像输出交叉熵损失值排序的代表性数据选择策略。在CIFAR10、CIFAR100和ImageNet数据集上的实验表明了该方法的有效性。



## **31. InvisibiliTee: Angle-agnostic Cloaking from Person-Tracking Systems with a Tee**

隐形Tee：带Te的人跟踪系统的角度不可知隐形 cs.CV

12 pages, 10 figures and the ICANN 2022 accpeted paper

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06962v1)

**Authors**: Yaxian Li, Bingqing Zhang, Guoping Zhao, Mingyu Zhang, Jiajun Liu, Ziwei Wang, Jirong Wen

**Abstracts**: After a survey for person-tracking system-induced privacy concerns, we propose a black-box adversarial attack method on state-of-the-art human detection models called InvisibiliTee. The method learns printable adversarial patterns for T-shirts that cloak wearers in the physical world in front of person-tracking systems. We design an angle-agnostic learning scheme which utilizes segmentation of the fashion dataset and a geometric warping process so the adversarial patterns generated are effective in fooling person detectors from all camera angles and for unseen black-box detection models. Empirical results in both digital and physical environments show that with the InvisibiliTee on, person-tracking systems' ability to detect the wearer drops significantly.

摘要: 在调查了个人跟踪系统引起的隐私问题之后，我们提出了一种针对最新的人类检测模型的黑盒对抗性攻击方法InvisibiliTee。该方法学习了T恤的可打印对抗性图案，这些T恤将穿着者遮盖在现实世界中的人面前。我们设计了一种角度不可知的学习方案，它利用时尚数据集的分割和几何扭曲过程，从而生成的对抗性模式能够有效地愚弄来自所有摄像机角度的人检测器和不可见的黑匣子检测模型。在数字和物理环境中的经验结果表明，随着隐形设备的开启，个人跟踪系统检测佩戴者的能力显著下降。



## **32. ARIEL: Adversarial Graph Contrastive Learning**

Ariel：对抗性图形对比学习 cs.LG

**SubmitDate**: 2022-08-15    [paper-pdf](http://arxiv.org/pdf/2208.06956v1)

**Authors**: Shengyu Feng, Baoyu Jing, Yada Zhu, Hanghang Tong

**Abstracts**: Contrastive learning is an effective unsupervised method in graph representation learning, and the key component of contrastive learning lies in the construction of positive and negative samples. Previous methods usually utilize the proximity of nodes in the graph as the principle. Recently, the data augmentation based contrastive learning method has advanced to show great power in the visual domain, and some works extended this method from images to graphs. However, unlike the data augmentation on images, the data augmentation on graphs is far less intuitive and much harder to provide high-quality contrastive samples, which leaves much space for improvement. In this work, by introducing an adversarial graph view for data augmentation, we propose a simple but effective method, Adversarial Graph Contrastive Learning (ARIEL), to extract informative contrastive samples within reasonable constraints. We develop a new technique called information regularization for stable training and use subgraph sampling for scalability. We generalize our method from node-level contrastive learning to the graph-level by treating each graph instance as a supernode. ARIEL consistently outperforms the current graph contrastive learning methods for both node-level and graph-level classification tasks on real-world datasets. We further demonstrate that ARIEL is more robust in face of adversarial attacks.

摘要: 对比学习是图形表示学习中一种有效的无监督方法，而对比学习的关键在于正负样本的构造。以往的方法通常以图中节点的邻近度为原则。近年来，基于数据增强的对比学习方法在视觉领域显示出强大的生命力，一些工作将该方法从图像扩展到图形。然而，与图像上的数据增强不同，图形上的数据增强的直观性差得多，难以提供高质量的对比样本，这就留下了很大的改进空间。在这项工作中，通过引入一种用于数据增强的对抗性图视图，我们提出了一种简单但有效的方法--对抗性图形对比学习(Ariel)，以在合理的约束下提取信息丰富的对比样本。我们开发了一种称为信息正则化的新技术来稳定训练，并使用子图抽样来实现可伸缩性。通过将每个图实例看作一个超节点，将我们的方法从节点级的对比学习推广到图级。对于实际数据集上的节点级和图级分类任务，Ariel始终优于当前的图对比学习方法。我们进一步证明了Ariel在面对对手攻击时具有更强的鲁棒性。



## **33. GNPassGAN: Improved Generative Adversarial Networks For Trawling Offline Password Guessing**

GNPassGAN：改进的用于拖网离线口令猜测的生成性对抗网络 cs.CR

9 pages, 8 tables, 3 figures

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06943v1)

**Authors**: Fangyi Yu, Miguel Vargas Martin

**Abstracts**: The security of passwords depends on a thorough understanding of the strategies used by attackers. Unfortunately, real-world adversaries use pragmatic guessing tactics like dictionary attacks, which are difficult to simulate in password security research. Dictionary attacks must be carefully configured and modified to represent an actual threat. This approach, however, needs domain-specific knowledge and expertise that are difficult to duplicate. This paper reviews various deep learning-based password guessing approaches that do not require domain knowledge or assumptions about users' password structures and combinations. It also introduces GNPassGAN, a password guessing tool built on generative adversarial networks for trawling offline attacks. In comparison to the state-of-the-art PassGAN model, GNPassGAN is capable of guessing 88.03\% more passwords and generating 31.69\% fewer duplicates.

摘要: 密码的安全性取决于对攻击者使用的策略的透彻理解。不幸的是，现实世界中的对手使用的是实用的猜测策略，如字典攻击，这在密码安全研究中很难模拟。必须仔细配置和修改字典攻击，以表示实际威胁。然而，这种方法需要难以复制的特定领域的知识和专业技能。本文综述了各种基于深度学习的密码猜测方法，这些方法不需要领域知识或关于用户密码结构和组合的假设。它还介绍了GNPassGAN，这是一个建立在生成性对手网络上的密码猜测工具，用于拖网离线攻击。与最新的PassGAN模型相比，GNPassGAN模型能够多猜测88.03个口令，生成的重复项减少31.69个。



## **34. Gradient Mask: Lateral Inhibition Mechanism Improves Performance in Artificial Neural Networks**

梯度掩模：侧抑制机制改善人工神经网络的性能 cs.CV

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06918v1)

**Authors**: Lei Jiang, Yongqing Liu, Shihai Xiao, Yansong Chua

**Abstracts**: Lateral inhibitory connections have been observed in the cortex of the biological brain, and has been extensively studied in terms of its role in cognitive functions. However, in the vanilla version of backpropagation in deep learning, all gradients (which can be understood to comprise of both signal and noise gradients) flow through the network during weight updates. This may lead to overfitting. In this work, inspired by biological lateral inhibition, we propose Gradient Mask, which effectively filters out noise gradients in the process of backpropagation. This allows the learned feature information to be more intensively stored in the network while filtering out noisy or unimportant features. Furthermore, we demonstrate analytically how lateral inhibition in artificial neural networks improves the quality of propagated gradients. A new criterion for gradient quality is proposed which can be used as a measure during training of various convolutional neural networks (CNNs). Finally, we conduct several different experiments to study how Gradient Mask improves the performance of the network both quantitatively and qualitatively. Quantitatively, accuracy in the original CNN architecture, accuracy after pruning, and accuracy after adversarial attacks have shown improvements. Qualitatively, the CNN trained using Gradient Mask has developed saliency maps that focus primarily on the object of interest, which is useful for data augmentation and network interpretability.

摘要: 在生物大脑的皮质中观察到了侧抑制连接，并就其在认知功能中的作用进行了广泛的研究。然而，在深度学习中的反向传播的香草版本中，所有的梯度(可以理解为包括信号和噪声梯度)在权重更新期间流经网络。这可能会导致过度适应。在这项工作中，受生物侧向抑制的启发，我们提出了梯度掩模，它有效地滤除了反向传播过程中的噪声梯度。这允许学习的特征信息更密集地存储在网络中，同时过滤掉噪声或不重要的特征。此外，我们还解析地演示了人工神经网络中的侧抑制如何提高传播梯度的质量。提出了一种新的梯度质量判据，可作为各种卷积神经网络(CNN)训练过程中的一种衡量标准。最后，我们进行了几个不同的实验，从定量和定性两个方面研究了梯度掩码如何提高网络的性能。在数量上，原始CNN架构中的准确性、修剪后的准确性和对抗性攻击后的准确性都显示出改进。在质量上，使用梯度蒙版训练的CNN已经开发出主要集中在感兴趣对象上的显著地图，这对于数据增强和网络可解释性很有用。



## **35. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

IPv6 SeeYou：利用IPv6中泄漏的标识符进行街道级地理定位 cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-08-14    [paper-pdf](http://arxiv.org/pdf/2208.06767v1)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.

摘要: 我们提出了IPv6 SeeYou，这是一种隐私攻击，允许远程和非特权对手以街道级别的精度物理定位许多住宅IPv6主机和网络。我们方法的关键涉及：1)从家庭路由器远程发现广域(WAN)硬件MAC地址；2)将这些MAC地址与已知位置的对应WiFi BSSID关联；以及3)通过关联连接到公共倒数第二个提供商路由器的设备来扩展覆盖范围。我们首先通过高速网络探测获得嵌入在IPv6地址中的大量MAC语料库。这些MAC地址有效地沿协议堆栈向上泄露，主要代表住宅路由器的广域网接口，其中许多是也提供WiFi的一体化设备。我们开发了一种技术来统计推断路由器的广域网和跨制造商和设备的WiFi MAC地址之间的映射，并发动大规模数据融合攻击，将广域网MAC与战争驾驶(地理定位)数据库中提供的WiFi BSSID相关联。利用这些相关性，我们在146个国家和地区对价值超过1200万美元的路由器的IPv6前缀进行了地理定位。选定的验证确认地理位置误差的中位数为39米。然后，我们利用技术和部署限制将攻击扩展到更大的一组IPv6住宅路由器，方法是将设备与常见的倒数第二个提供商路由器进行集群和关联。虽然我们负责任地向几家制造商和供应商披露了我们的结果，但已部署的住宅有线电视和DSL路由器的僵化生态系统表明，在可预见的未来，我们的攻击仍将对隐私构成威胁。



## **36. Adversarial Texture for Fooling Person Detectors in the Physical World**

物理世界中愚人探测器的对抗性纹理 cs.CV

Accepted by CVPR 2022

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2203.03373v4)

**Authors**: Zhanhao Hu, Siyuan Huang, Xiaopei Zhu, Fuchun Sun, Bo Zhang, Xiaolin Hu

**Abstracts**: Nowadays, cameras equipped with AI systems can capture and analyze images to detect people automatically. However, the AI system can make mistakes when receiving deliberately designed patterns in the real world, i.e., physical adversarial examples. Prior works have shown that it is possible to print adversarial patches on clothes to evade DNN-based person detectors. However, these adversarial examples could have catastrophic drops in the attack success rate when the viewing angle (i.e., the camera's angle towards the object) changes. To perform a multi-angle attack, we propose Adversarial Texture (AdvTexture). AdvTexture can cover clothes with arbitrary shapes so that people wearing such clothes can hide from person detectors from different viewing angles. We propose a generative method, named Toroidal-Cropping-based Expandable Generative Attack (TC-EGA), to craft AdvTexture with repetitive structures. We printed several pieces of cloth with AdvTexure and then made T-shirts, skirts, and dresses in the physical world. Experiments showed that these clothes could fool person detectors in the physical world.

摘要: 如今，配备人工智能系统的摄像头可以捕捉和分析图像，自动检测人。然而，人工智能系统在接收到现实世界中故意设计的模式时可能会出错，即物理对抗性例子。以前的工作已经表明，可以在衣服上打印敌意补丁来躲避基于DNN的个人探测器。然而，当视角(即相机指向对象的角度)改变时，这些对抗性的例子可能会使攻击成功率灾难性地下降。为了进行多角度攻击，我们提出了对抗性纹理(AdvTexture)。AdvTexture可以覆盖任意形状的衣服，这样穿着这种衣服的人就可以从不同的视角隐藏起来，躲避人的探测器。提出了一种基于环形裁剪的可扩展产生式攻击方法(TC-EGA)，用于制作具有重复结构的AdvTexture。我们用AdvTexure打印了几块布，然后在现实世界中制作了T恤、裙子和连衣裙。实验表明，这些衣服可以愚弄物理世界中的人体探测器。



## **37. An Analytic Framework for Robust Training of Artificial Neural Networks**

一种神经网络稳健训练的分析框架 cs.LG

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2205.13502v2)

**Authors**: Ramin Barati, Reza Safabakhsh, Mohammad Rahmati

**Abstracts**: The reliability of a learning model is key to the successful deployment of machine learning in various industries. Creating a robust model, particularly one unaffected by adversarial attacks, requires a comprehensive understanding of the adversarial examples phenomenon. However, it is difficult to describe the phenomenon due to the complicated nature of the problems in machine learning. Consequently, many studies investigate the phenomenon by proposing a simplified model of how adversarial examples occur and validate it by predicting some aspect of the phenomenon. While these studies cover many different characteristics of the adversarial examples, they have not reached a holistic approach to the geometric and analytic modeling of the phenomenon. This paper propose a formal framework to study the phenomenon in learning theory and make use of complex analysis and holomorphicity to offer a robust learning rule for artificial neural networks. With the help of complex analysis, we can effortlessly move between geometric and analytic perspectives of the phenomenon and offer further insights on the phenomenon by revealing its connection with harmonic functions. Using our model, we can explain some of the most intriguing characteristics of adversarial examples, including transferability of adversarial examples, and pave the way for novel approaches to mitigate the effects of the phenomenon.

摘要: 学习模型的可靠性是机器学习在各个行业成功部署的关键。创建一个健壮的模型，特别是一个不受对抗性攻击影响的模型，需要对对抗性例子现象有一个全面的了解。然而，由于机器学习中问题的复杂性，这一现象很难描述。因此，许多研究通过提出对抗性例子如何发生的简化模型来研究这一现象，并通过预测该现象的某些方面来验证该模型。虽然这些研究涵盖了对抗性例子的许多不同特征，但它们还没有达成对这一现象的几何和解析建模的整体方法。本文提出了一种学习理论中研究这一现象的形式化框架，并利用复分析和全纯理论为人工神经网络提供了一种稳健的学习规则。在复杂分析的帮助下，我们可以毫不费力地在现象的几何和解析视角之间切换，并通过揭示它与调和函数的联系来提供对该现象的进一步见解。使用我们的模型，我们可以解释对抗性例子的一些最有趣的特征，包括对抗性例子的可转移性，并为缓解这一现象的影响的新方法铺平道路。



## **38. Revisiting Adversarial Attacks on Graph Neural Networks for Graph Classification**

图神经网络图分类中的敌意攻击再探 cs.SI

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06651v1)

**Authors**: Beini Xie, Heng Chang, Xin Wang, Tian Bian, Shiji Zhou, Daixin Wang, Zhiqiang Zhang, Wenwu Zhu

**Abstracts**: Graph neural networks (GNNs) have achieved tremendous success in the task of graph classification and diverse downstream real-world applications. Despite their success, existing approaches are either limited to structure attacks or restricted to local information. This calls for a more general attack framework on graph classification, which faces significant challenges due to the complexity of generating local-node-level adversarial examples using the global-graph-level information. To address this "global-to-local" problem, we present a general framework CAMA to generate adversarial examples by manipulating graph structure and node features in a hierarchical style. Specifically, we make use of Graph Class Activation Mapping and its variant to produce node-level importance corresponding to the graph classification task. Then through a heuristic design of algorithms, we can perform both feature and structure attacks under unnoticeable perturbation budgets with the help of both node-level and subgraph-level importance. Experiments towards attacking four state-of-the-art graph classification models on six real-world benchmarks verify the flexibility and effectiveness of our framework.

摘要: 图形神经网络(GNN)在图形分类和各种下游实际应用中取得了巨大的成功。尽管它们取得了成功，但现有的方法要么限于结构攻击，要么限于局部信息。这就需要一个更通用的图分类攻击框架，由于利用全局图级信息生成局部节点级对抗性实例的复杂性，该框架面临着巨大的挑战。为了解决这个“从全局到局部”的问题，我们提出了一个通用的CAMA框架，通过以层次化的方式操纵图的结构和节点特征来生成对抗性实例。具体地说，我们利用图类激活映射及其变体来产生与图分类任务相对应的节点级重要性。然后通过算法的启发式设计，在节点级和子图级重要性的帮助下，在不可察觉的扰动预算下执行特征攻击和结构攻击。在六个真实世界基准上对四个最先进的图分类模型进行了攻击实验，验证了该框架的灵活性和有效性。



## **39. Poison Ink: Robust and Invisible Backdoor Attack**

毒墨：强大而隐形的后门攻击 cs.CR

IEEE Transactions on Image Processing (TIP)

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2108.02488v3)

**Authors**: Jie Zhang, Dongdong Chen, Qidong Huang, Jing Liao, Weiming Zhang, Huamin Feng, Gang Hua, Nenghai Yu

**Abstracts**: Recent research shows deep neural networks are vulnerable to different types of attacks, such as adversarial attack, data poisoning attack and backdoor attack. Among them, backdoor attack is the most cunning one and can occur in almost every stage of deep learning pipeline. Therefore, backdoor attack has attracted lots of interests from both academia and industry. However, most existing backdoor attack methods are either visible or fragile to some effortless pre-processing such as common data transformations. To address these limitations, we propose a robust and invisible backdoor attack called "Poison Ink". Concretely, we first leverage the image structures as target poisoning areas, and fill them with poison ink (information) to generate the trigger pattern. As the image structure can keep its semantic meaning during the data transformation, such trigger pattern is inherently robust to data transformations. Then we leverage a deep injection network to embed such trigger pattern into the cover image to achieve stealthiness. Compared to existing popular backdoor attack methods, Poison Ink outperforms both in stealthiness and robustness. Through extensive experiments, we demonstrate Poison Ink is not only general to different datasets and network architectures, but also flexible for different attack scenarios. Besides, it also has very strong resistance against many state-of-the-art defense techniques.

摘要: 最近的研究表明，深度神经网络容易受到不同类型的攻击，如对抗性攻击、数据中毒攻击和后门攻击。其中，后门攻击是最狡猾的一种，几乎可以发生在深度学习管道的每个阶段。因此，后门攻击引起了学术界和产业界的广泛关注。然而，大多数现有的后门攻击方法对于一些毫不费力的预处理，如常见的数据转换，要么是可见的，要么是脆弱的。为了解决这些局限性，我们提出了一种强大且不可见的后门攻击，称为“毒墨”。具体地说，我们首先利用图像结构作为目标中毒区域，并在其中填充毒墨(信息)来生成触发图案。由于图像结构在数据转换过程中可以保持其语义，因此这种触发模式对数据转换具有内在的健壮性。然后利用深度注入网络将这种触发模式嵌入到封面图像中，以实现隐身。与现有流行的后门攻击方法相比，毒墨在隐蔽性和健壮性方面都更胜一筹。通过大量的实验，我们证明了毒墨不仅对不同的数据集和网络体系结构具有通用性，而且对不同的攻击场景也具有很强的灵活性。此外，它还对许多最先进的防御技术具有很强的抵抗力。



## **40. MaskBlock: Transferable Adversarial Examples with Bayes Approach**

MaskBlock：贝叶斯方法的可转移对抗性实例 cs.LG

Under Review

**SubmitDate**: 2022-08-13    [paper-pdf](http://arxiv.org/pdf/2208.06538v1)

**Authors**: Mingyuan Fan, Cen Chen, Ximeng Liu, Wenzhong Guo

**Abstracts**: The transferability of adversarial examples (AEs) across diverse models is of critical importance for black-box adversarial attacks, where attackers cannot access the information about black-box models. However, crafted AEs always present poor transferability. In this paper, by regarding the transferability of AEs as generalization ability of the model, we reveal that vanilla black-box attacks craft AEs via solving a maximum likelihood estimation (MLE) problem. For MLE, the results probably are model-specific local optimum when available data is small, i.e., limiting the transferability of AEs. By contrast, we re-formulate crafting transferable AEs as the maximizing a posteriori probability estimation problem, which is an effective approach to boost the generalization of results with limited available data. Because Bayes posterior inference is commonly intractable, a simple yet effective method called MaskBlock is developed to approximately estimate. Moreover, we show that the formulated framework is a generalization version for various attack methods. Extensive experiments illustrate MaskBlock can significantly improve the transferability of crafted adversarial examples by up to about 20%.

摘要: 在黑盒对抗性攻击中，攻击者无法获取有关黑盒模型的信息，因此对抗性示例在不同模型之间的可转移性至关重要。然而，精心制作的AE总是表现出较差的可转移性。通过将攻击事件的可转移性作为模型的泛化能力，揭示了香草黑盒攻击通过求解一个极大似然估计问题来欺骗攻击事件。对于最大似然估计，当可用数据很小时，结果可能是特定于模型的局部最优，即限制了AE的可转移性。相反，我们将构造可转移的AEs重新描述为最大后验概率估计问题，这是在可用数据有限的情况下提高结果普适性的有效方法。由于贝叶斯后验推断通常很难处理，因此发展了一种简单而有效的方法-MaskBlock来近似估计。此外，我们还证明了该框架是各种攻击方法的泛化版本。大量的实验表明，MaskBlock可以显著提高特制的对抗性例子的可转移性，最高可提高约20%。



## **41. Hide and Seek: on the Stealthiness of Attacks against Deep Learning Systems**

捉迷藏：关于深度学习系统攻击的隐蔽性 cs.CR

To appear in European Symposium on Research in Computer Security  (ESORICS) 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2205.15944v2)

**Authors**: Zeyan Liu, Fengjun Li, Jingqiang Lin, Zhu Li, Bo Luo

**Abstracts**: With the growing popularity of artificial intelligence and machine learning, a wide spectrum of attacks against deep learning models have been proposed in the literature. Both the evasion attacks and the poisoning attacks attempt to utilize adversarially altered samples to fool the victim model to misclassify the adversarial sample. While such attacks claim to be or are expected to be stealthy, i.e., imperceptible to human eyes, such claims are rarely evaluated. In this paper, we present the first large-scale study on the stealthiness of adversarial samples used in the attacks against deep learning. We have implemented 20 representative adversarial ML attacks on six popular benchmarking datasets. We evaluate the stealthiness of the attack samples using two complementary approaches: (1) a numerical study that adopts 24 metrics for image similarity or quality assessment; and (2) a user study of 3 sets of questionnaires that has collected 20,000+ annotations from 1,000+ responses. Our results show that the majority of the existing attacks introduce nonnegligible perturbations that are not stealthy to human eyes. We further analyze the factors that contribute to attack stealthiness. We further examine the correlation between the numerical analysis and the user studies, and demonstrate that some image quality metrics may provide useful guidance in attack designs, while there is still a significant gap between assessed image quality and visual stealthiness of attacks.

摘要: 随着人工智能和机器学习的日益普及，文献中提出了针对深度学习模型的广泛攻击。逃避攻击和投毒攻击都试图利用敌意更改的样本来愚弄受害者模型来错误分类敌意样本。虽然这种攻击声称是或预计是隐蔽的，即人眼看不见，但这种说法很少得到评估。本文首次对深度学习攻击中使用的敌意样本的隐蔽性进行了大规模研究。我们已经在六个流行的基准数据集上实施了20个具有代表性的对抗性ML攻击。我们使用两种互补的方法来评估攻击样本的隐蔽性：(1)采用24个度量来评估图像相似性或质量的数值研究；(2)用户研究3组问卷，从1000多个回复中收集了20,000多个注释。我们的结果表明，现有的大多数攻击都引入了不可忽略的扰动，这些扰动对人眼来说是不隐形的。进一步分析了影响攻击隐蔽性的因素。我们进一步检验了数值分析和用户研究之间的相关性，并证明了一些图像质量度量可以为攻击设计提供有用的指导，而评估的图像质量和攻击的视觉隐蔽性之间仍然存在着显著的差距。



## **42. PRIVEE: A Visual Analytic Workflow for Proactive Privacy Risk Inspection of Open Data**

Privee：开放数据主动隐私风险检测的可视化分析工作流 cs.CR

Accepted for IEEE Symposium on Visualization in Cyber Security, 2022

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06481v1)

**Authors**: Kaustav Bhattacharjee, Akm Islam, Jaideep Vaidya, Aritra Dasgupta

**Abstracts**: Open data sets that contain personal information are susceptible to adversarial attacks even when anonymized. By performing low-cost joins on multiple datasets with shared attributes, malicious users of open data portals might get access to information that violates individuals' privacy. However, open data sets are primarily published using a release-and-forget model, whereby data owners and custodians have little to no cognizance of these privacy risks. We address this critical gap by developing a visual analytic solution that enables data defenders to gain awareness about the disclosure risks in local, joinable data neighborhoods. The solution is derived through a design study with data privacy researchers, where we initially play the role of a red team and engage in an ethical data hacking exercise based on privacy attack scenarios. We use this problem and domain characterization to develop a set of visual analytic interventions as a defense mechanism and realize them in PRIVEE, a visual risk inspection workflow that acts as a proactive monitor for data defenders. PRIVEE uses a combination of risk scores and associated interactive visualizations to let data defenders explore vulnerable joins and interpret risks at multiple levels of data granularity. We demonstrate how PRIVEE can help emulate the attack strategies and diagnose disclosure risks through two case studies with data privacy experts.

摘要: 包含个人信息的开放数据集即使在匿名的情况下也容易受到敌意攻击。通过对具有共享属性的多个数据集执行低成本连接，开放数据门户的恶意用户可能会访问侵犯个人隐私的信息。然而，开放数据集主要是使用一种即发布即忘的模式发布的，在这种模式下，数据所有者和托管人很少或根本没有意识到这些隐私风险。我们通过开发可视化分析解决方案来解决这一关键差距，该解决方案使数据防御者能够意识到本地可合并数据社区的披露风险。解决方案是通过与数据隐私研究人员的设计研究得出的，在设计研究中，我们最初扮演红色团队的角色，并根据隐私攻击场景进行道德的数据黑客练习。我们使用这个问题和领域特征来开发一套视觉分析干预作为防御机制，并在Privee中实现它们，Privee是一个视觉风险检测工作流，充当数据防御者的主动监视器。Privee结合使用风险分值和关联的交互式可视化，让数据防御者能够在多个级别的数据粒度上探索易受攻击的连接并解释风险。通过与数据隐私专家的两个案例研究，我们展示了Privee如何帮助模拟攻击策略和诊断泄露风险。



## **43. UniNet: A Unified Scene Understanding Network and Exploring Multi-Task Relationships through the Lens of Adversarial Attacks**

UniNet：一个统一的场景理解网络，通过对抗性攻击的镜头探索多任务关系 cs.CV

Accepted at DeepMTL workshop, ICCV 2021

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2108.04584v2)

**Authors**: Naresh Kumar Gurulingan, Elahe Arani, Bahram Zonooz

**Abstracts**: Scene understanding is crucial for autonomous systems which intend to operate in the real world. Single task vision networks extract information only based on some aspects of the scene. In multi-task learning (MTL), on the other hand, these single tasks are jointly learned, thereby providing an opportunity for tasks to share information and obtain a more comprehensive understanding. To this end, we develop UniNet, a unified scene understanding network that accurately and efficiently infers vital vision tasks including object detection, semantic segmentation, instance segmentation, monocular depth estimation, and monocular instance depth prediction. As these tasks look at different semantic and geometric information, they can either complement or conflict with each other. Therefore, understanding inter-task relationships can provide useful cues to enable complementary information sharing. We evaluate the task relationships in UniNet through the lens of adversarial attacks based on the notion that they can exploit learned biases and task interactions in the neural network. Extensive experiments on the Cityscapes dataset, using untargeted and targeted attacks reveal that semantic tasks strongly interact amongst themselves, and the same holds for geometric tasks. Additionally, we show that the relationship between semantic and geometric tasks is asymmetric and their interaction becomes weaker as we move towards higher-level representations.

摘要: 场景理解对于打算在真实世界中运行的自主系统至关重要。单任务视觉网络仅基于场景的某些方面来提取信息。另一方面，在多任务学习(MTL)中，这些单一任务是共同学习的，从而为任务提供了共享信息和获得更全面理解的机会。为此，我们开发了UniNet，这是一个统一的场景理解网络，可以准确高效地推断重要的视觉任务，包括对象检测、语义分割、实例分割、单目深度估计和单目实例深度预测。当这些任务查看不同的语义和几何信息时，它们可以相互补充，也可以相互冲突。因此，了解任务间的关系可以为实现互补信息共享提供有用的线索。我们通过对抗性攻击的视角来评估UniNet中的任务关系，基于这样的概念，即它们可以利用神经网络中的学习偏差和任务交互。在城市景观数据集上使用无目标和目标攻击进行的广泛实验表明，语义任务之间相互作用很强，几何任务也是如此。此外，我们还证明了语义任务和几何任务之间的关系是不对称的，并且随着我们向更高级别的表示前进，它们之间的交互作用变得更弱。



## **44. Unifying Gradients to Improve Real-world Robustness for Deep Networks**

统一梯度以提高深度网络的真实稳健性 stat.ML

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06228v1)

**Authors**: Yingwen Wu, Sizhe Chen, Kun Fang, Xiaolin Huang

**Abstracts**: The wide application of deep neural networks (DNNs) demands an increasing amount of attention to their real-world robustness, i.e., whether a DNN resists black-box adversarial attacks, among them score-based query attacks (SQAs) are the most threatening ones because of their practicalities and effectiveness: the attackers only need dozens of queries on model outputs to seriously hurt a victim network. Defending against SQAs requires a slight but artful variation of outputs due to the service purpose for users, who share the same output information with attackers. In this paper, we propose a real-world defense, called Unifying Gradients (UniG), to unify gradients of different data so that attackers could only probe a much weaker attack direction that is similar for different samples. Since such universal attack perturbations have been validated as less aggressive than the input-specific perturbations, UniG protects real-world DNNs by indicating attackers a twisted and less informative attack direction. To enhance UniG's practical significance in real-world applications, we implement it as a Hadamard product module that is computationally-efficient and readily plugged into any model. According to extensive experiments on 5 SQAs and 4 defense baselines, UniG significantly improves real-world robustness without hurting clean accuracy on CIFAR10 and ImageNet. For instance, UniG maintains a CIFAR-10 model of 77.80% accuracy under 2500-query Square attack while the state-of-the-art adversarially-trained model only has 67.34% on CIFAR10. Simultaneously, UniG greatly surpasses all compared baselines in clean accuracy and the modification degree of outputs. The code would be released.

摘要: 深度神经网络(DNN)的广泛应用要求人们越来越关注其在现实世界中的健壮性，即DNN是否能抵抗黑箱对抗攻击，其中基于分数的查询攻击(Score-Based Query Attack，SBA)因其实用性和有效性而成为最具威胁性的攻击：攻击者只需对模型输出进行数十次查询即可严重损害受害者网络。由于用户的服务目的，防御SBA需要稍微但巧妙地改变输出，因为用户与攻击者共享相同的输出信息。在本文中，我们提出了一种称为统一梯度的真实防御方法，将不同数据的梯度统一起来，使得攻击者只能探测对不同样本相似的弱得多的攻击方向。由于这种通用的攻击扰动已被验证为不如特定于输入的扰动那么具侵略性，unig通过向攻击者指示一个扭曲的、信息量较少的攻击方向来保护现实世界的DNN。为了增强unig在现实世界应用程序中的实际意义，我们将其实现为Hadamard产品模块，该模块计算效率高，可以方便地插入到任何型号中。根据在5个SQA和4个防御基线上的广泛实验，unig在不损害CIFAR10和ImageNet上的干净准确性的情况下，显著提高了真实世界的健壮性。例如，在2500-Query Square攻击下，unig保持了77.80%的CIFAR-10模型，而最新的对抗性训练模型在CIFAR10上只有67.34%的准确率。同时，unig在清洁精度和输出修改程度上大大超过了所有比较的基线。代码将会被发布。



## **45. Scale-free Photo-realistic Adversarial Pattern Attack**

无尺度照片真实感对抗性模式攻击 cs.CV

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06222v1)

**Authors**: Xiangbo Gao, Weicheng Xie, Minmin Liu, Cheng Luo, Qinliang Lin, Linlin Shen, Keerthy Kusumam, Siyang Song

**Abstracts**: Traditional pixel-wise image attack algorithms suffer from poor robustness to defense algorithms, i.e., the attack strength degrades dramatically when defense algorithms are applied. Although Generative Adversarial Networks (GAN) can partially address this problem by synthesizing a more semantically meaningful texture pattern, the main limitation is that existing generators can only generate images of a specific scale. In this paper, we propose a scale-free generation-based attack algorithm that synthesizes semantically meaningful adversarial patterns globally to images with arbitrary scales. Our generative attack approach consistently outperforms the state-of-the-art methods on a wide range of attack settings, i.e. the proposed approach largely degraded the performance of various image classification, object detection, and instance segmentation algorithms under different advanced defense methods.

摘要: 传统的像素级图像攻击算法对防御算法的健壮性较差，即应用防御算法后攻击强度急剧下降。虽然生成性对抗网络(GAN)可以通过合成更具语义意义的纹理模式来部分解决这个问题，但主要限制是现有生成器只能生成特定规模的图像。在本文中，我们提出了一种基于无尺度生成的攻击算法，该算法将全局具有语义意义的攻击模式合成到任意尺度的图像中。我们的生成性攻击方法在广泛的攻击环境下始终优于最先进的方法，即在不同的高级防御方法下，所提出的方法在很大程度上降低了各种图像分类、目标检测和实例分割算法的性能。



## **46. A Knowledge Distillation-Based Backdoor Attack in Federated Learning**

联邦学习中一种基于知识蒸馏的后门攻击 cs.LG

**SubmitDate**: 2022-08-12    [paper-pdf](http://arxiv.org/pdf/2208.06176v1)

**Authors**: Yifan Wang, Wei Fan, Keke Yang, Naji Alhusaini, Jing Li

**Abstracts**: Federated Learning (FL) is a novel framework of decentralized machine learning. Due to the decentralized feature of FL, it is vulnerable to adversarial attacks in the training procedure, e.g. , backdoor attacks. A backdoor attack aims to inject a backdoor into the machine learning model such that the model will make arbitrarily incorrect behavior on the test sample with some specific backdoor trigger. Even though a range of backdoor attack methods of FL has been introduced, there are also methods defending against them. Many of the defending methods utilize the abnormal characteristics of the models with backdoor or the difference between the models with backdoor and the regular models. To bypass these defenses, we need to reduce the difference and the abnormal characteristics. We find a source of such abnormality is that backdoor attack would directly flip the label of data when poisoning the data. However, current studies of the backdoor attack in FL are not mainly focus on reducing the difference between the models with backdoor and the regular models. In this paper, we propose Adversarial Knowledge Distillation(ADVKD), a method combine knowledge distillation with backdoor attack in FL. With knowledge distillation, we can reduce the abnormal characteristics in model result from the label flipping, thus the model can bypass the defenses. Compared to current methods, we show that ADVKD can not only reach a higher attack success rate, but also successfully bypass the defenses when other methods fails. To further explore the performance of ADVKD, we test how the parameters affect the performance of ADVKD under different scenarios. According to the experiment result, we summarize how to adjust the parameter for better performance under different scenarios. We also use several methods to visualize the effect of different attack and explain the effectiveness of ADVKD.

摘要: 联邦学习(FL)是一种新型的去中心化机器学习框架。由于FL的分散性，它在训练过程中很容易受到对抗性攻击，例如后门攻击。后门攻击的目的是向机器学习模型中注入一个后门，以便该模型将使用某个特定的后门触发器在测试样本上做出任意不正确的行为。尽管已经引入了一系列FL的后门攻击方法，但也有一些方法可以防御它们。许多防御方法利用了后门模型的异常特性，或者利用了后门模型与常规模型的区别。为了绕过这些防御，我们需要减少差异和异常特征。我们发现这种异常的一个来源是后门攻击在毒化数据时会直接翻转数据的标签。然而，目前对FL后门攻击的研究主要集中在缩小后门模型与常规模型之间的差异上。本文提出了一种将知识提取与后门攻击相结合的方法--对抗性知识提取(ADVKD)。通过知识提取，可以减少模型中因标签翻转而导致的异常特征，从而使模型能够绕过防御。与现有方法相比，我们证明了ADVKD不仅可以达到更高的攻击成功率，而且在其他方法失败的情况下可以成功地绕过防御。为了进一步探索ADVKD的性能，我们测试了不同场景下参数对ADVKD性能的影响。根据实验结果，总结了在不同场景下如何调整参数以获得更好的性能。我们还使用几种方法来可视化不同攻击的效果，并解释了ADVKD的有效性。



## **47. A Survey of MulVAL Extensions and Their Attack Scenarios Coverage**

MulVAL扩展及其攻击场景覆盖研究综述 cs.CR

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05750v1)

**Authors**: David Tayouri, Nick Baum, Asaf Shabtai, Rami Puzis

**Abstracts**: Organizations employ various adversary models in order to assess the risk and potential impact of attacks on their networks. Attack graphs represent vulnerabilities and actions an attacker can take to identify and compromise an organization's assets. Attack graphs facilitate both visual presentation and algorithmic analysis of attack scenarios in the form of attack paths. MulVAL is a generic open-source framework for constructing logical attack graphs, which has been widely used by researchers and practitioners and extended by them with additional attack scenarios. This paper surveys all of the existing MulVAL extensions, and maps all MulVAL interaction rules to MITRE ATT&CK Techniques to estimate their attack scenarios coverage. This survey aligns current MulVAL extensions along unified ontological concepts and highlights the existing gaps. It paves the way for methodical improvement of MulVAL and the comprehensive modeling of the entire landscape of adversarial behaviors captured in MITRE ATT&CK.

摘要: 组织使用各种对手模型来评估攻击对其网络的风险和潜在影响。攻击图表示攻击者可以采取的漏洞和行动，以识别和危害组织的资产。攻击图便于以攻击路径的形式对攻击场景进行可视化呈现和算法分析。MulVAL是一个用于构建逻辑攻击图的通用开源框架，已经被研究人员和实践者广泛使用，并被他们用额外的攻击场景进行扩展。本文综述了现有的所有MulVAL扩展，并将所有的MulVAL交互规则映射到MITRE ATT&CK技术，以估计它们的攻击场景覆盖率。这项调查将当前的MulVAL扩展与统一的本体概念保持一致，并强调了存在的差距。它为MulVAL的系统改进和对MITRE ATT&CK捕获的敌对行为的整个场景的全面建模铺平了道路。



## **48. Diverse Generative Adversarial Perturbations on Attention Space for Transferable Adversarial Attacks**

注意空间上可转移对抗性攻击的不同生成性对抗性扰动 cs.CV

ICIP 2022

**SubmitDate**: 2022-08-11    [paper-pdf](http://arxiv.org/pdf/2208.05650v1)

**Authors**: Woo Jae Kim, Seunghoon Hong, Sung-Eui Yoon

**Abstracts**: Adversarial attacks with improved transferability - the ability of an adversarial example crafted on a known model to also fool unknown models - have recently received much attention due to their practicality. Nevertheless, existing transferable attacks craft perturbations in a deterministic manner and often fail to fully explore the loss surface, thus falling into a poor local optimum and suffering from low transferability. To solve this problem, we propose Attentive-Diversity Attack (ADA), which disrupts diverse salient features in a stochastic manner to improve transferability. Primarily, we perturb the image attention to disrupt universal features shared by different models. Then, to effectively avoid poor local optima, we disrupt these features in a stochastic manner and explore the search space of transferable perturbations more exhaustively. More specifically, we use a generator to produce adversarial perturbations that each disturbs features in different ways depending on an input latent code. Extensive experimental evaluations demonstrate the effectiveness of our method, outperforming the transferability of state-of-the-art methods. Codes are available at https://github.com/wkim97/ADA.

摘要: 具有改进的可转移性的对抗性攻击--在已知模型上制作的对抗性例子也能够愚弄未知模型的能力--由于其实用性最近受到了极大的关注。然而，现有的可转移攻击以确定性的方式制造扰动，往往不能充分探索损失曲面，从而陷入较差的局部最优，且可转移性较低。为了解决这一问题，我们提出了注意力多样性攻击(ADA)，它以随机的方式破坏不同的显著特征，以提高可转移性。首先，我们扰乱图像注意力，以扰乱不同模型共享的通用特征。然后，为了有效地避免局部最优，我们以随机的方式破坏了这些特征，并更详尽地探索了可转移扰动的搜索空间。更具体地说，我们使用生成器来产生对抗性扰动，每个扰动都以不同的方式干扰特征，具体取决于输入的潜在代码。广泛的实验评估表明，我们的方法是有效的，超过了最先进的方法的可转移性。有关代码，请访问https://github.com/wkim97/ADA.



## **49. Controlled Quantum Teleportation in the Presence of an Adversary**

对手在场时的受控量子隐形传态 quant-ph

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05554v1)

**Authors**: Sayan Gangopadhyay, Tiejun Wang, Atefeh Mashatan, Shohini Ghose

**Abstracts**: We present a device independent analysis of controlled quantum teleportation where the receiver is not trusted. We show that the notion of genuine tripartite nonlocality allows us to certify control power in such a scenario. By considering a specific adversarial attack strategy on a device characterized by depolarizing noise, we find that control power is a monotonically increasing function of genuine tripartite nonlocality. These results are relevant for building practical quantum communication networks and also shed light on the role of nonlocality in multipartite quantum information processing.

摘要: 在接收者不可信任的情况下，我们提出了受控量子隐形传态的设备无关分析。我们证明了真正的三方非局部性的概念允许我们在这种情况下证明控制权。通过考虑具有去极化噪声特征的设备上的特定对抗攻击策略，我们发现控制功率是真三方非定域性的单调递增函数。这些结果对构建实用的量子通信网络具有重要意义，也有助于揭示非定域性在多体量子信息处理中的作用。



## **50. Pikachu: Securing PoS Blockchains from Long-Range Attacks by Checkpointing into Bitcoin PoW using Taproot**

Pikachu：通过使用Taproot检查点进入比特币PoW来保护PoS区块链免受远程攻击 cs.CR

To appear at ConsensusDay 22 (ACM CCS 2022 Workshop)

**SubmitDate**: 2022-08-10    [paper-pdf](http://arxiv.org/pdf/2208.05408v1)

**Authors**: Sarah Azouvi, Marko Vukolić

**Abstracts**: Blockchain systems based on a reusable resource, such as proof-of-stake (PoS), provide weaker security guarantees than those based on proof-of-work. Specifically, they are vulnerable to long-range attacks, where an adversary can corrupt prior participants in order to rewrite the full history of the chain. To prevent this attack on a PoS chain, we propose a protocol that checkpoints the state of the PoS chain to a proof-of-work blockchain such as Bitcoin. Our checkpointing protocol hence does not rely on any central authority. Our work uses Schnorr signatures and leverages Bitcoin recent Taproot upgrade, allowing us to create a checkpointing transaction of constant size. We argue for the security of our protocol and present an open-source implementation that was tested on the Bitcoin testnet.

摘要: 基于可重用资源的区块链系统，如风险证明(POS)，提供的安全保证比基于工作证明的系统更弱。具体地说，它们容易受到远程攻击，在远程攻击中，对手可以破坏之前的参与者，以便重写链的完整历史。为了防止这种对PoS链的攻击，我们提出了一种协议，将PoS链的状态检查到工作证明区块链，如比特币。因此，我们的检查点协议不依赖于任何中央机构。我们的工作使用Schnorr签名并利用比特币最近的Taproot升级，使我们能够创建恒定大小的检查点交易。我们为协议的安全性进行了论证，并给出了一个在比特币测试网上进行测试的开源实现。



