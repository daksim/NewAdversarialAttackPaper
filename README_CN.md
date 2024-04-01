# Latest Adversarial Attack Papers
**update at 2024-04-01 09:32:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Selective Attention-based Modulation for Continual Learning**

基于选择性注意的持续学习调制 cs.CV

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.20086v1) [paper-pdf](http://arxiv.org/pdf/2403.20086v1)

**Authors**: Giovanni Bellitto, Federica Proietto Salanitri, Matteo Pennisi, Matteo Boschini, Angelo Porrello, Simone Calderara, Simone Palazzo, Concetto Spampinato

**Abstract**: We present SAM, a biologically-plausible selective attention-driven modulation approach to enhance classification models in a continual learning setting. Inspired by neurophysiological evidence that the primary visual cortex does not contribute to object manifold untangling for categorization and that primordial attention biases are still embedded in the modern brain, we propose to employ auxiliary saliency prediction features as a modulation signal to drive and stabilize the learning of a sequence of non-i.i.d. classification tasks. Experimental results confirm that SAM effectively enhances the performance (in some cases up to about twenty percent points) of state-of-the-art continual learning methods, both in class-incremental and task-incremental settings. Moreover, we show that attention-based modulation successfully encourages the learning of features that are more robust to the presence of spurious features and to adversarial attacks than baseline methods. Code is available at: https://github.com/perceivelab/SAM.

摘要: 我们提出了SAM，一种生物学上看似合理的选择性注意驱动的调制方法，以增强持续学习环境中的分类模型。受到神经生理学证据的启发，即初级视觉皮质不有助于物体歧管的分类，以及原始注意偏差仍然嵌入现代大脑，我们建议使用辅助显著预测特征作为调制信号来驱动和稳定对非I.I.D.序列的学习。分类任务。实验结果证实，SAM有效地提高了最先进的持续学习方法的性能(在某些情况下高达约20%)，无论是在班级递增还是任务递增的设置下。此外，我们表明，基于注意力的调制成功地鼓励了对虚假特征的存在和对抗攻击的特征的学习，这些特征比基线方法更健壮。代码可从以下网址获得：https://github.com/perceivelab/SAM.



## **2. Strong Transferable Adversarial Attacks via Ensembled Asymptotically Normal Distribution Learning**

基于集成渐近正态分布学习的强可传递对抗攻击 cs.LG

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2209.11964v2) [paper-pdf](http://arxiv.org/pdf/2209.11964v2)

**Authors**: Zhengwei Fang, Rui Wang, Tao Huang, Liping Jing

**Abstract**: Strong adversarial examples are crucial for evaluating and enhancing the robustness of deep neural networks. However, the performance of popular attacks is usually sensitive, for instance, to minor image transformations, stemming from limited information -- typically only one input example, a handful of white-box source models, and undefined defense strategies. Hence, the crafted adversarial examples are prone to overfit the source model, which hampers their transferability to unknown architectures. In this paper, we propose an approach named Multiple Asymptotically Normal Distribution Attacks (MultiANDA) which explicitly characterize adversarial perturbations from a learned distribution. Specifically, we approximate the posterior distribution over the perturbations by taking advantage of the asymptotic normality property of stochastic gradient ascent (SGA), then employ the deep ensemble strategy as an effective proxy for Bayesian marginalization in this process, aiming to estimate a mixture of Gaussians that facilitates a more thorough exploration of the potential optimization space. The approximated posterior essentially describes the stationary distribution of SGA iterations, which captures the geometric information around the local optimum. Thus, MultiANDA allows drawing an unlimited number of adversarial perturbations for each input and reliably maintains the transferability. Our proposed method outperforms ten state-of-the-art black-box attacks on deep learning models with or without defenses through extensive experiments on seven normally trained and seven defense models.

摘要: 强对抗性的例子对于评估和提高深度神经网络的稳健性至关重要。然而，流行攻击的性能通常对较小的图像变换很敏感，这源于有限的信息--通常只有一个输入示例、几个白盒源模型和未定义的防御策略。因此，精心制作的敌意示例容易过度匹配源模型，这阻碍了它们向未知体系结构的可转移性。在本文中，我们提出了一种名为多重渐近正态分布攻击(Multiple渐近正态分布攻击)的方法，它显式地刻画了来自学习分布的敌对扰动。具体地说，我们利用随机梯度上升(SGA)的渐近正态性质来逼近扰动下的后验分布，然后使用深度集成策略作为贝叶斯边际化的有效代理，目的是估计一个混合的高斯分布，以便更深入地探索潜在的优化空间。近似后验概率本质上描述了SGA迭代的平稳分布，它捕捉了局部最优解附近的几何信息。因此，MultiANDA允许为每个输入绘制无限数量的对抗性扰动，并可靠地保持可转移性。通过在7个正常训练模型和7个防御模型上的大量实验，我们提出的方法在有防御和无防御的深度学习模型上的性能超过了10种最新的黑盒攻击。



## **3. An Anomaly Behavior Analysis Framework for Securing Autonomous Vehicle Perception**

一种保障自主车辆感知的异常行为分析框架 cs.RO

20th ACS/IEEE International Conference on Computer Systems and  Applications (Accepted for publication)

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2310.05041v2) [paper-pdf](http://arxiv.org/pdf/2310.05041v2)

**Authors**: Murad Mehrab Abrar, Salim Hariri

**Abstract**: As a rapidly growing cyber-physical platform, Autonomous Vehicles (AVs) are encountering more security challenges as their capabilities continue to expand. In recent years, adversaries are actively targeting the perception sensors of autonomous vehicles with sophisticated attacks that are not easily detected by the vehicles' control systems. This work proposes an Anomaly Behavior Analysis approach to detect a perception sensor attack against an autonomous vehicle. The framework relies on temporal features extracted from a physics-based autonomous vehicle behavior model to capture the normal behavior of vehicular perception in autonomous driving. By employing a combination of model-based techniques and machine learning algorithms, the proposed framework distinguishes between normal and abnormal vehicular perception behavior. To demonstrate the application of the framework in practice, we performed a depth camera attack experiment on an autonomous vehicle testbed and generated an extensive dataset. We validated the effectiveness of the proposed framework using this real-world data and released the dataset for public access. To our knowledge, this dataset is the first of its kind and will serve as a valuable resource for the research community in evaluating their intrusion detection techniques effectively.

摘要: 作为一个快速发展的网络物理平台，自动驾驶汽车(AVs)随着其能力的不断扩大，面临着更多的安全挑战。近年来，对手积极瞄准自动驾驶车辆的感知传感器，进行复杂的攻击，而这些攻击不容易被车辆的控制系统检测到。本文提出了一种异常行为分析方法来检测感知传感器对自动驾驶车辆的攻击。该框架依赖于从基于物理的自动驾驶车辆行为模型中提取的时间特征来捕捉自动驾驶中车辆感知的正常行为。通过结合基于模型的技术和机器学习算法，该框架区分了正常和异常的车辆感知行为。为了验证该框架在实践中的应用，我们在自主车辆试验台上进行了深度相机攻击实验，并生成了大量的数据集。我们使用这些真实世界的数据验证了提出的框架的有效性，并发布了数据集以供公众访问。据我们所知，该数据集是此类数据的第一个，将为研究界提供宝贵的资源，以有效地评估他们的入侵检测技术。



## **4. MMCert: Provable Defense against Adversarial Attacks to Multi-modal Models**

MMCert：针对多模态模型的对抗攻击的可证明防御 cs.CV

To appear in CVPR'24

**SubmitDate**: 2024-03-29    [abs](http://arxiv.org/abs/2403.19080v2) [paper-pdf](http://arxiv.org/pdf/2403.19080v2)

**Authors**: Yanting Wang, Hongye Fu, Wei Zou, Jinyuan Jia

**Abstract**: Different from a unimodal model whose input is from a single modality, the input (called multi-modal input) of a multi-modal model is from multiple modalities such as image, 3D points, audio, text, etc. Similar to unimodal models, many existing studies show that a multi-modal model is also vulnerable to adversarial perturbation, where an attacker could add small perturbation to all modalities of a multi-modal input such that the multi-modal model makes incorrect predictions for it. Existing certified defenses are mostly designed for unimodal models, which achieve sub-optimal certified robustness guarantees when extended to multi-modal models as shown in our experimental results. In our work, we propose MMCert, the first certified defense against adversarial attacks to a multi-modal model. We derive a lower bound on the performance of our MMCert under arbitrary adversarial attacks with bounded perturbations to both modalities (e.g., in the context of auto-driving, we bound the number of changed pixels in both RGB image and depth image). We evaluate our MMCert using two benchmark datasets: one for the multi-modal road segmentation task and the other for the multi-modal emotion recognition task. Moreover, we compare our MMCert with a state-of-the-art certified defense extended from unimodal models. Our experimental results show that our MMCert outperforms the baseline.

摘要: 与单通道模型的输入来自单一通道不同，多通道模型的输入(称为多通道输入)来自图像、3D点、音频、文本等多个通道。与单通道模型类似，许多现有的研究表明，多通道模型也容易受到对抗性扰动的影响，攻击者可以在多通道输入的所有通道中添加小的扰动，从而使得多通道模型对其做出错误的预测。现有的认证防御大多是针对单模模型设计的，如我们的实验结果所示，当扩展到多模模型时，它们获得了次优的认证稳健性保证。在我们的工作中，我们提出了MMCert，这是第一个认证的多模式对抗攻击防御模型。我们得到了MMCert在两种模式都有界扰动的任意攻击下的性能下界(例如，在自动驾驶的背景下，我们限制了RGB图像和深度图像中变化的像素数量)。我们使用两个基准数据集来评估我们的MMCert：一个用于多模式道路分割任务，另一个用于多模式情感识别任务。此外，我们将我们的MMCert与从单模模型扩展而来的最先进的认证防御进行了比较。我们的实验结果表明，我们的MMCert的性能优于基线。



## **5. Deciphering the Interplay between Local Differential Privacy, Average Bayesian Privacy, and Maximum Bayesian Privacy**

解密局部差分隐私、平均贝叶斯隐私和最大贝叶斯隐私之间的相互作用 cs.LG

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.16591v2) [paper-pdf](http://arxiv.org/pdf/2403.16591v2)

**Authors**: Xiaojin Zhang, Yulin Fei, Wei Chen, Hai Jin

**Abstract**: The swift evolution of machine learning has led to emergence of various definitions of privacy due to the threats it poses to privacy, including the concept of local differential privacy (LDP). Although widely embraced and utilized across numerous domains, this conventional approach to measure privacy still exhibits certain limitations, spanning from failure to prevent inferential disclosure to lack of consideration for the adversary's background knowledge. In this comprehensive study, we introduce Bayesian privacy and delve into the intricate relationship between LDP and its Bayesian counterparts, unveiling novel insights into utility-privacy trade-offs. We introduce a framework that encapsulates both attack and defense strategies, highlighting their interplay and effectiveness. The relationship between LDP and Maximum Bayesian Privacy (MBP) is first revealed, demonstrating that under uniform prior distribution, a mechanism satisfying $\xi$-LDP will satisfy $\xi$-MBP and conversely $\xi$-MBP also confers 2$\xi$-LDP. Our next theoretical contribution are anchored in the rigorous definitions and relationships between Average Bayesian Privacy (ABP) and Maximum Bayesian Privacy (MBP), encapsulated by equations $\epsilon_{p,a} \leq \frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p,m} + \epsilon)\cdot(e^{\epsilon_{p,m} + \epsilon} - 1)}$. These relationships fortify our understanding of the privacy guarantees provided by various mechanisms. Our work not only lays the groundwork for future empirical exploration but also promises to facilitate the design of privacy-preserving algorithms, thereby fostering the development of trustworthy machine learning solutions.

摘要: 机器学习的快速发展导致了各种隐私定义的出现，因为它对隐私构成了威胁，包括局部差异隐私(LDP)的概念。尽管这种衡量隐私的传统方法在许多领域得到了广泛的接受和应用，但它仍然显示出一定的局限性，从未能阻止推论披露到缺乏对对手背景知识的考虑。在这项全面的研究中，我们介绍了贝叶斯隐私，并深入研究了自民党与其贝叶斯同行之间的错综复杂的关系，揭示了对效用-隐私权衡的新见解。我们引入了一个框架，该框架封装了攻击和防御战略，突出了它们的相互作用和有效性。首先揭示了LDP与最大贝叶斯隐私度之间的关系，证明了在均匀先验分布下，满足$xi-LDP的机制将满足$\xi-MBP，反之，$\xi-MBP也赋予2$\xi-LDP。我们的下一个理论贡献是建立在平均贝叶斯隐私度(ABP)和最大贝叶斯隐私度(MBP)之间的严格定义和关系上，用方程$\epsilon_{p，a}\leq\frac{1}{\sqrt{2}}\sqrt{(\epsilon_{p，m}+\epsilon)\cdot(e^{\epsilon_{p，m}+\epsilon}-1)}$来封装。这些关系加强了我们对各种机制提供的隐私保障的理解。我们的工作不仅为未来的经验探索奠定了基础，也承诺促进隐私保护算法的设计，从而促进可信机器学习解决方案的开发。



## **6. Evolving Assembly Code in an Adversarial Environment**

对抗环境下的汇编代码演变 cs.NE

9 pages, 5 figures, 6 listings

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19489v1) [paper-pdf](http://arxiv.org/pdf/2403.19489v1)

**Authors**: Irina Maliukov, Gera Weiss, Oded Margalit, Achiya Elyasaf

**Abstract**: In this work, we evolve assembly code for the CodeGuru competition. The competition's goal is to create a survivor -- an assembly program that runs the longest in shared memory, by resisting attacks from adversary survivors and finding their weaknesses. For evolving top-notch solvers, we specify a Backus Normal Form (BNF) for the assembly language and synthesize the code from scratch using Genetic Programming (GP). We evaluate the survivors by running CodeGuru games against human-written winning survivors. Our evolved programs found weaknesses in the programs they were trained against and utilized them. In addition, we compare our approach with a Large-Language Model, demonstrating that the latter cannot generate a survivor that can win at any competition. This work has important applications for cyber-security, as we utilize evolution to detect weaknesses in survivors. The assembly BNF is domain-independent; thus, by modifying the fitness function, it can detect code weaknesses and help fix them. Finally, the CodeGuru competition offers a novel platform for analyzing GP and code evolution in adversarial environments. To support further research in this direction, we provide a thorough qualitative analysis of the evolved survivors and the weaknesses found.

摘要: 在这项工作中，我们为CodeGuru竞赛演变汇编代码。这项竞赛的目标是创建一个幸存者--一个在共享内存中运行时间最长的汇编程序，通过抵抗对手幸存者的攻击并找到他们的弱点。对于进化的顶级解算器，我们为汇编语言指定了Backus范式(BNF)，并使用遗传编程(GP)从头开始合成代码。我们通过运行CodeGuru游戏来评估幸存者，以对抗人类编写的获胜幸存者。我们的演进计划发现了他们所针对的计划中的弱点，并利用了这些弱点。此外，我们将我们的方法与大语言模型进行比较，表明后者无法产生能够在任何竞争中获胜的幸存者。这项工作在网络安全方面有重要的应用，因为我们利用进化论来检测幸存者的弱点。程序集BNF是独立于域的；因此，通过修改适应度函数，它可以检测代码弱点并帮助修复它们。最后，CodeGuru竞赛为分析对抗性环境中的GP和代码演化提供了一个新的平台。为了支持这方面的进一步研究，我们对进化的幸存者和发现的弱点进行了彻底的定性分析。



## **7. Cloudy with a Chance of Cyberattacks: Dangling Resources Abuse on Cloud Platforms**

云与网络攻击的机会：云平台上的资源滥用危险 cs.NI

17 pages, 29 figures, to be published in NSDI'24: Proceedings of the  21st USENIX Symposium on Networked Systems Design and Implementation

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2403.19368v1) [paper-pdf](http://arxiv.org/pdf/2403.19368v1)

**Authors**: Jens Frieß, Tobias Gattermayer, Nethanel Gelernter, Haya Schulmann, Michael Waidner

**Abstract**: Recent works showed that it is feasible to hijack resources on cloud platforms. In such hijacks, attackers can take over released resources that belong to legitimate organizations. It was proposed that adversaries could abuse these resources to carry out attacks against customers of the hijacked services, e.g., through malware distribution. However, to date, no research has confirmed the existence of these attacks. We identify, for the first time, real-life hijacks of cloud resources. This yields a number of surprising and important insights. First, contrary to previous assumption that attackers primarily target IP addresses, our findings reveal that the type of resource is not the main consideration in a hijack. Attackers focus on hijacking records that allow them to determine the resource by entering freetext. The costs and overhead of hijacking such records are much lower than those of hijacking IP addresses, which are randomly selected from a large pool. Second, identifying hijacks poses a substantial challenge. Monitoring resource changes, e.g., changes in content, is insufficient, since such changes could also be legitimate. Retrospective analysis of digital assets to identify hijacks is also arduous due to the immense volume of data involved and the absence of indicators to search for. To address this challenge, we develop a novel approach that involves analyzing data from diverse sources to effectively differentiate between malicious and legitimate modifications. Our analysis has revealed 20,904 instances of hijacked resources on popular cloud platforms. While some hijacks are short-lived (up to 15 days), 1/3 persist for more than 65 days. We study how attackers abuse the hijacked resources and find that, in contrast to the threats considered in previous work, the majority of the abuse (75%) is blackhat search engine optimization.

摘要: 最近的研究表明，劫持云平台上的资源是可行的。在这种劫持中，攻击者可以接管属于合法组织的已释放资源。有人提出，攻击者可以滥用这些资源，例如通过恶意软件分发，对被劫持服务的客户进行攻击。然而，到目前为止，还没有研究证实这些攻击的存在。我们首次发现了现实生活中的云资源劫持行为。这产生了许多令人惊讶和重要的见解。首先，与之前认为攻击者主要针对IP地址的假设相反，我们的研究结果表明，资源类型不是劫持的主要考虑因素。攻击者专注于劫持记录，这些记录允许他们通过输入freetext来确定资源。劫持此类记录的成本和开销比劫持IP地址的成本和开销要低得多，后者是从一个大的池中随机选择的。其次，识别劫机事件构成了一个巨大的挑战。监视资源改变，例如内容的改变是不够的，因为这样的改变也可能是合法的。对数字资产进行追溯性分析以确定劫持行为也很困难，因为涉及的数据量巨大，而且缺乏可供搜索的指标。为了应对这一挑战，我们开发了一种新的方法，涉及分析来自不同来源的数据，以有效区分恶意修改和合法修改。我们的分析揭示了20904起流行云平台上的资源被劫持事件。虽然一些劫持是短暂的(长达15天)，但三分之一的劫机持续时间超过65天。我们研究了攻击者如何滥用被劫持的资源，发现与以前工作中考虑的威胁相反，大多数滥用(75%)是黑帽搜索引擎优化。



## **8. Data-free Defense of Black Box Models Against Adversarial Attacks**

黑盒模型对抗攻击的无数据防御 cs.LG

CVPR Workshop (Under Review)

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2211.01579v3) [paper-pdf](http://arxiv.org/pdf/2211.01579v3)

**Authors**: Gaurav Kumar Nayak, Inder Khatri, Ruchit Rawal, Anirban Chakraborty

**Abstract**: Several companies often safeguard their trained deep models (i.e., details of architecture, learnt weights, training details etc.) from third-party users by exposing them only as black boxes through APIs. Moreover, they may not even provide access to the training data due to proprietary reasons or sensitivity concerns. In this work, we propose a novel defense mechanism for black box models against adversarial attacks in a data-free set up. We construct synthetic data via generative model and train surrogate network using model stealing techniques. To minimize adversarial contamination on perturbed samples, we propose 'wavelet noise remover' (WNR) that performs discrete wavelet decomposition on input images and carefully select only a few important coefficients determined by our 'wavelet coefficient selection module' (WCSM). To recover the high-frequency content of the image after noise removal via WNR, we further train a 'regenerator' network with an objective to retrieve the coefficients such that the reconstructed image yields similar to original predictions on the surrogate model. At test time, WNR combined with trained regenerator network is prepended to the black box network, resulting in a high boost in adversarial accuracy. Our method improves the adversarial accuracy on CIFAR-10 by 38.98% and 32.01% on state-of-the-art Auto Attack compared to baseline, even when the attacker uses surrogate architecture (Alexnet-half and Alexnet) similar to the black box architecture (Alexnet) with same model stealing strategy as defender. The code is available at https://github.com/vcl-iisc/data-free-black-box-defense

摘要: 几家公司经常保护他们训练有素的深度模型(即架构细节、学习的重量、训练细节等)。通过API仅将第三方用户暴露为黑盒。此外，由于专有原因或敏感性问题，它们甚至可能无法提供对培训数据的访问。在这项工作中，我们提出了一种新的黑盒模型在无数据环境下抵抗敌意攻击的防御机制。我们通过产生式模型构造合成数据，并使用模型窃取技术训练代理网络。为了最大限度地减少扰动样本带来的有害污染，我们提出了小波去噪器(WNR)，它对输入图像进行离散小波分解，并仔细地选择由我们的小波系数选择模块(WCSM)确定的几个重要系数。为了恢复图像经过WNR去噪后的高频内容，我们进一步训练了一个‘再生器’网络，目的是恢复系数，使重建的图像产生与原始预测相似的代理模型。在测试时，将WNR与训练好的再生器网络相结合，加入到黑盒网络中，大大提高了对抗的准确率。与基准相比，我们的方法在CIFAR-10上的攻击准确率分别提高了38.98%和32.01%，即使攻击者使用类似于黑盒体系结构(Alexnet)的代理体系结构(Alexnet-Half和Alexnet)，并且与防御者使用相同的模型窃取策略。代码可在https://github.com/vcl-iisc/data-free-black-box-defense上获得



## **9. Feature Unlearning for Pre-trained GANs and VAEs**

预训练GAN和VAE的特性取消学习 cs.CV

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2303.05699v4) [paper-pdf](http://arxiv.org/pdf/2303.05699v4)

**Authors**: Saemi Moon, Seunghyuk Cho, Dongwoo Kim

**Abstract**: We tackle the problem of feature unlearning from a pre-trained image generative model: GANs and VAEs. Unlike a common unlearning task where an unlearning target is a subset of the training set, we aim to unlearn a specific feature, such as hairstyle from facial images, from the pre-trained generative models. As the target feature is only presented in a local region of an image, unlearning the entire image from the pre-trained model may result in losing other details in the remaining region of the image. To specify which features to unlearn, we collect randomly generated images that contain the target features. We then identify a latent representation corresponding to the target feature and then use the representation to fine-tune the pre-trained model. Through experiments on MNIST, CelebA, and FFHQ datasets, we show that target features are successfully removed while keeping the fidelity of the original models. Further experiments with an adversarial attack show that the unlearned model is more robust under the presence of malicious parties.

摘要: 我们从一个预先训练的图像生成模型GANS和VAE中解决了特征遗忘的问题。与通常的遗忘任务不同，忘记目标是训练集的一个子集，我们的目标是从预先训练的生成模型中忘记特定的特征，如面部图像中的发型。由于目标特征仅呈现在图像的局部区域中，因此从预先训练的模型中不学习整个图像可能导致丢失图像剩余区域中的其他细节。为了指定要取消学习的特征，我们收集包含目标特征的随机生成的图像。然后，我们识别对应于目标特征的潜在表示，然后使用该表示来微调预先训练的模型。通过在MNIST、CelebA和FFHQ数据集上的实验，我们证明了在保持原始模型保真度的情况下，目标特征被成功去除。进一步的对抗性攻击实验表明，未学习模型在恶意方存在的情况下具有更强的鲁棒性。



## **10. Data Poisoning for In-context Learning**

基于上下文学习的数据中毒 cs.CR

**SubmitDate**: 2024-03-28    [abs](http://arxiv.org/abs/2402.02160v2) [paper-pdf](http://arxiv.org/pdf/2402.02160v2)

**Authors**: Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, Jiliang Tang

**Abstract**: In the domain of large language models (LLMs), in-context learning (ICL) has been recognized for its innovative ability to adapt to new tasks, relying on examples rather than retraining or fine-tuning. This paper delves into the critical issue of ICL's susceptibility to data poisoning attacks, an area not yet fully explored. We wonder whether ICL is vulnerable, with adversaries capable of manipulating example data to degrade model performance. To address this, we introduce ICLPoison, a specialized attacking framework conceived to exploit the learning mechanisms of ICL. Our approach uniquely employs discrete text perturbations to strategically influence the hidden states of LLMs during the ICL process. We outline three representative strategies to implement attacks under our framework, each rigorously evaluated across a variety of models and tasks. Our comprehensive tests, including trials on the sophisticated GPT-4 model, demonstrate that ICL's performance is significantly compromised under our framework. These revelations indicate an urgent need for enhanced defense mechanisms to safeguard the integrity and reliability of LLMs in applications relying on in-context learning.

摘要: 在大型语言模型(LLM)领域，情境学习(ICL)因其适应新任务的创新能力而被公认，它依赖于例子而不是再培训或微调。本文深入研究了ICL对数据中毒攻击的易感性这一关键问题，这是一个尚未完全探索的领域。我们想知道ICL是否易受攻击，因为对手能够操纵示例数据来降低模型性能。为了解决这个问题，我们引入了ICLPoison，这是一个专门的攻击框架，旨在利用ICL的学习机制。我们的方法独特地使用离散文本扰动来战略性地影响ICL过程中LLM的隐藏状态。我们概述了在我们的框架下实施攻击的三种具有代表性的战略，每种战略都在各种模型和任务中进行了严格的评估。我们的全面测试，包括对复杂的GPT-4模型的试验，表明ICL的性能在我们的框架下受到了严重影响。这些发现表明，迫切需要增强防御机制，以保障依赖于情景学习的应用程序中LLMS的完整性和可靠性。



## **11. Towards Sustainable SecureML: Quantifying Carbon Footprint of Adversarial Machine Learning**

迈向可持续的SecureML：量化对抗机器学习的碳足迹 cs.LG

Accepted at GreenNet Workshop @ IEEE International Conference on  Communications (IEEE ICC 2024)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.19009v1) [paper-pdf](http://arxiv.org/pdf/2403.19009v1)

**Authors**: Syed Mhamudul Hasan, Abdur R. Shahid, Ahmed Imteaj

**Abstract**: The widespread adoption of machine learning (ML) across various industries has raised sustainability concerns due to its substantial energy usage and carbon emissions. This issue becomes more pressing in adversarial ML, which focuses on enhancing model security against different network-based attacks. Implementing defenses in ML systems often necessitates additional computational resources and network security measures, exacerbating their environmental impacts. In this paper, we pioneer the first investigation into adversarial ML's carbon footprint, providing empirical evidence connecting greater model robustness to higher emissions. Addressing the critical need to quantify this trade-off, we introduce the Robustness Carbon Trade-off Index (RCTI). This novel metric, inspired by economic elasticity principles, captures the sensitivity of carbon emissions to changes in adversarial robustness. We demonstrate the RCTI through an experiment involving evasion attacks, analyzing the interplay between robustness against attacks, performance, and carbon emissions.

摘要: 机器学习(ML)在各个行业的广泛采用引起了人们对其大量能源使用和碳排放的可持续发展的担忧。这一问题在对抗性ML中变得更加紧迫，它的重点是增强模型的安全性，以抵御不同的基于网络的攻击。在ML系统中实施防御通常需要额外的计算资源和网络安全措施，从而加剧了它们对环境的影响。在这篇文章中，我们率先对敌对的ML的碳足迹进行了调查，提供了将更大的模型稳健性与更高的排放量联系起来的经验证据。为了解决量化这种权衡的迫切需要，我们引入了稳健性碳权衡指数(RCTI)。这一新的衡量标准受到经济弹性原则的启发，捕捉了碳排放对对抗性稳健性变化的敏感性。我们通过一个涉及规避攻击的实验来演示RCTI，分析了对攻击的健壮性、性能和碳排放之间的相互影响。



## **12. Robustness and Visual Explanation for Black Box Image, Video, and ECG Signal Classification with Reinforcement Learning**

基于强化学习的黑盒图像、视频和ECG信号分类的鲁棒性和视觉解释 cs.LG

AAAI Proceedings reference:  https://ojs.aaai.org/index.php/AAAI/article/view/30579

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18985v1) [paper-pdf](http://arxiv.org/pdf/2403.18985v1)

**Authors**: Soumyendu Sarkar, Ashwin Ramesh Babu, Sajad Mousavi, Vineet Gundecha, Avisek Naug, Sahand Ghorbanpour

**Abstract**: We present a generic Reinforcement Learning (RL) framework optimized for crafting adversarial attacks on different model types spanning from ECG signal analysis (1D), image classification (2D), and video classification (3D). The framework focuses on identifying sensitive regions and inducing misclassifications with minimal distortions and various distortion types. The novel RL method outperforms state-of-the-art methods for all three applications, proving its efficiency. Our RL approach produces superior localization masks, enhancing interpretability for image classification and ECG analysis models. For applications such as ECG analysis, our platform highlights critical ECG segments for clinicians while ensuring resilience against prevalent distortions. This comprehensive tool aims to bolster both resilience with adversarial training and transparency across varied applications and data types.

摘要: 我们提出了一个通用的强化学习（RL）框架，优化用于针对不同模型类型的对抗攻击，这些模型类型涵盖ECG信号分析（1D），图像分类（2D）和视频分类（3D）。该框架的重点是确定敏感区域，并以最小限度的失真和各种失真类型引起错误分类。新的RL方法在所有三个应用中都优于最先进的方法，证明了其效率。我们的RL方法产生了卓越的定位掩模，增强了图像分类和ECG分析模型的可解释性。对于ECG分析等应用，我们的平台为临床医生突出了关键ECG段，同时确保对普遍失真的恢复能力。这一综合性工具旨在通过对抗性培训和不同应用程序和数据类型的透明度来增强弹性。



## **13. Deep Learning for Robust and Explainable Models in Computer Vision**

用于计算机视觉中鲁棒和可解释模型的深度学习 cs.CV

150 pages, 37 figures, 12 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18674v1) [paper-pdf](http://arxiv.org/pdf/2403.18674v1)

**Authors**: Mohammadreza Amirian

**Abstract**: Recent breakthroughs in machine and deep learning (ML and DL) research have provided excellent tools for leveraging enormous amounts of data and optimizing huge models with millions of parameters to obtain accurate networks for image processing. These developments open up tremendous opportunities for using artificial intelligence (AI) in the automation and human assisted AI industry. However, as more and more models are deployed and used in practice, many challenges have emerged. This thesis presents various approaches that address robustness and explainability challenges for using ML and DL in practice.   Robustness and reliability are the critical components of any model before certification and deployment in practice. Deep convolutional neural networks (CNNs) exhibit vulnerability to transformations of their inputs, such as rotation and scaling, or intentional manipulations as described in the adversarial attack literature. In addition, building trust in AI-based models requires a better understanding of current models and developing methods that are more explainable and interpretable a priori.   This thesis presents developments in computer vision models' robustness and explainability. Furthermore, this thesis offers an example of using vision models' feature response visualization (models' interpretations) to improve robustness despite interpretability and robustness being seemingly unrelated in the related research. Besides methodological developments for robust and explainable vision models, a key message of this thesis is introducing model interpretation techniques as a tool for understanding vision models and improving their design and robustness. In addition to the theoretical developments, this thesis demonstrates several applications of ML and DL in different contexts, such as medical imaging and affective computing.

摘要: 机器和深度学习(ML和DL)研究的最新突破为利用海量数据和优化具有数百万参数的巨大模型提供了极好的工具，以获得用于图像处理的准确网络。这些发展为人工智能(AI)在自动化和人工辅助AI行业中的使用打开了巨大的机会。然而，随着越来越多的模型在实践中部署和使用，出现了许多挑战。这篇论文提出了各种方法来解决在实践中使用ML和DL时的健壮性和可解释性挑战。在实践中认证和部署之前，健壮性和可靠性是任何模型的关键组件。深层卷积神经网络(CNN)表现出对其输入的变换的脆弱性，例如旋转和缩放，或者如对抗性攻击文献中所描述的故意操纵。此外，建立对基于人工智能的模型的信任需要更好地理解当前的模型，并开发更具解释性和先验性的方法。本文介绍了计算机视觉模型的稳健性和可解释性方面的研究进展。此外，本文还给出了一个使用视觉模型的特征响应可视化(模型的解释)来提高稳健性的例子，尽管可解释性和稳健性在相关研究中似乎是无关的。除了稳健和可解释的视觉模型的方法论发展外，本文的一个关键信息是引入模型解释技术作为理解视觉模型的工具，并改进其设计和稳健性。除了理论上的发展，本文还展示了ML和DL在不同环境中的几个应用，例如医学成像和情感计算。



## **14. LCANets++: Robust Audio Classification using Multi-layer Neural Networks with Lateral Competition**

LCANets ++：使用具有横向竞争的多层神经网络的鲁棒音频分类 cs.SD

Accepted at 2024 IEEE International Conference on Acoustics, Speech  and Signal Processing Workshops (ICASSPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2308.12882v2) [paper-pdf](http://arxiv.org/pdf/2308.12882v2)

**Authors**: Sayanton V. Dibbo, Juston S. Moore, Garrett T. Kenyon, Michael A. Teti

**Abstract**: Audio classification aims at recognizing audio signals, including speech commands or sound events. However, current audio classifiers are susceptible to perturbations and adversarial attacks. In addition, real-world audio classification tasks often suffer from limited labeled data. To help bridge these gaps, previous work developed neuro-inspired convolutional neural networks (CNNs) with sparse coding via the Locally Competitive Algorithm (LCA) in the first layer (i.e., LCANets) for computer vision. LCANets learn in a combination of supervised and unsupervised learning, reducing dependency on labeled samples. Motivated by the fact that auditory cortex is also sparse, we extend LCANets to audio recognition tasks and introduce LCANets++, which are CNNs that perform sparse coding in multiple layers via LCA. We demonstrate that LCANets++ are more robust than standard CNNs and LCANets against perturbations, e.g., background noise, as well as black-box and white-box attacks, e.g., evasion and fast gradient sign (FGSM) attacks.

摘要: 音频分类的目的是识别音频信号，包括语音命令或声音事件。然而，当前的音频分类器容易受到扰动和对抗性攻击。此外，现实世界的音频分类任务通常会受到有限的标签数据的影响。为了弥补这些差距，以前的工作发展了神经启发卷积神经网络(CNN)，通过第一层的局部竞争算法(LCA)进行稀疏编码，用于计算机视觉。LCANet在监督和非监督学习的组合中学习，减少了对标记样本的依赖。基于听觉皮层也是稀疏的这一事实，我们将LCANets扩展到音频识别任务，并引入LCANets++，LCANets++是通过LCA在多层进行稀疏编码的CNN。我们证明了LCANet++比标准的CNN和LCANet对扰动(例如背景噪声)以及黑盒和白盒攻击(例如逃避和快速梯度符号(FGSM)攻击)具有更强的鲁棒性。



## **15. The Impact of Uniform Inputs on Activation Sparsity and Energy-Latency Attacks in Computer Vision**

均匀输入对计算机视觉中激活稀疏性和能量延迟攻击的影响 cs.CR

Accepted at the DLSP 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18587v1) [paper-pdf](http://arxiv.org/pdf/2403.18587v1)

**Authors**: Andreas Müller, Erwin Quiring

**Abstract**: Resource efficiency plays an important role for machine learning nowadays. The energy and decision latency are two critical aspects to ensure a sustainable and practical application. Unfortunately, the energy consumption and decision latency are not robust against adversaries. Researchers have recently demonstrated that attackers can compute and submit so-called sponge examples at inference time to increase the energy consumption and decision latency of neural networks. In computer vision, the proposed strategy crafts inputs with less activation sparsity which could otherwise be used to accelerate the computation. In this paper, we analyze the mechanism how these energy-latency attacks reduce activation sparsity. In particular, we find that input uniformity is a key enabler. A uniform image, that is, an image with mostly flat, uniformly colored surfaces, triggers more activations due to a specific interplay of convolution, batch normalization, and ReLU activation. Based on these insights, we propose two new simple, yet effective strategies for crafting sponge examples: sampling images from a probability distribution and identifying dense, yet inconspicuous inputs in natural datasets. We empirically examine our findings in a comprehensive evaluation with multiple image classification models and show that our attack achieves the same sparsity effect as prior sponge-example methods, but at a fraction of computation effort. We also show that our sponge examples transfer between different neural networks. Finally, we discuss applications of our findings for the good by improving efficiency by increasing sparsity.

摘要: 资源效率在当今机器学习中扮演着重要的角色。能量和决策延迟是确保可持续和实际应用的两个关键方面。不幸的是，能量消耗和决策延迟对对手的健壮性不强。研究人员最近证明，攻击者可以在推理时计算并提交所谓的海绵示例，以增加神经网络的能量消耗和决策延迟。在计算机视觉中，所提出的策略以较小的激活稀疏性来制作输入，否则可以用来加速计算。在本文中，我们分析了这些能量延迟攻击降低激活稀疏性的机制。特别是，我们发现输入的一致性是一个关键的推动因素。由于卷积、批处理归一化和REU激活的特定相互作用，统一图像，即具有大部分平坦、统一颜色的表面的图像，会触发更多的激活。基于这些见解，我们提出了两种新的简单而有效的海绵样本制作策略：从概率分布中采样图像，以及在自然数据集中识别密集但不明显的输入。我们在多个图像分类模型的综合评估中对我们的发现进行了实证检验，结果表明，我们的攻击达到了与以前的海绵示例方法相同的稀疏效果，但计算量只有很小一部分。我们还表明，我们的海绵样本在不同的神经网络之间转换。最后，我们讨论了我们的发现的应用，通过增加稀疏性来提高效率。



## **16. MisGUIDE : Defense Against Data-Free Deep Learning Model Extraction**

MisGUIDE：防御无数据深度学习模型提取 cs.CR

Under Review

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18580v1) [paper-pdf](http://arxiv.org/pdf/2403.18580v1)

**Authors**: Mahendra Gurve, Sankar Behera, Satyadev Ahlawat, Yamuna Prasad

**Abstract**: The rise of Machine Learning as a Service (MLaaS) has led to the widespread deployment of machine learning models trained on diverse datasets. These models are employed for predictive services through APIs, raising concerns about the security and confidentiality of the models due to emerging vulnerabilities in prediction APIs. Of particular concern are model cloning attacks, where individuals with limited data and no knowledge of the training dataset manage to replicate a victim model's functionality through black-box query access. This commonly entails generating adversarial queries to query the victim model, thereby creating a labeled dataset.   This paper proposes "MisGUIDE", a two-step defense framework for Deep Learning models that disrupts the adversarial sample generation process by providing a probabilistic response when the query is deemed OOD. The first step employs a Vision Transformer-based framework to identify OOD queries, while the second step perturbs the response for such queries, introducing a probabilistic loss function to MisGUIDE the attackers. The aim of the proposed defense method is to reduce the accuracy of the cloned model while maintaining accuracy on authentic queries. Extensive experiments conducted on two benchmark datasets demonstrate that the proposed framework significantly enhances the resistance against state-of-the-art data-free model extraction in black-box settings.

摘要: 机器学习即服务(MLaaS)的兴起导致了在不同数据集上训练的机器学习模型的广泛部署。这些模型通过API用于预测服务，由于预测API中新出现的漏洞，人们对模型的安全性和保密性表示担忧。特别令人关切的是模型克隆攻击，数据有限且不了解训练数据集的个人设法通过黑盒查询访问复制受害者模型的功能。这通常需要生成敌意查询以查询受害者模型，从而创建标签数据集。本文提出了一种用于深度学习模型的两步防御框架“MisGuide”，该框架通过在查询被认为是面向对象时提供概率响应来中断敌意样本的生成过程。第一步使用基于Vision Transformer的框架来识别OOD查询，而第二步干扰对此类查询的响应，引入概率损失函数来误导攻击者。该防御方法的目的是降低克隆模型的准确性，同时保持对真实查询的准确性。在两个基准数据集上进行的广泛实验表明，该框架显著增强了对黑盒环境下最先进的无数据模型提取的抵抗力。



## **17. CosalPure: Learning Concept from Group Images for Robust Co-Saliency Detection**

CosalPure：基于组图像的学习概念，用于鲁棒共显著性检测 cs.CV

8 pages

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18554v1) [paper-pdf](http://arxiv.org/pdf/2403.18554v1)

**Authors**: Jiayi Zhu, Qing Guo, Felix Juefei-Xu, Yihao Huang, Yang Liu, Geguang Pu

**Abstract**: Co-salient object detection (CoSOD) aims to identify the common and salient (usually in the foreground) regions across a given group of images. Although achieving significant progress, state-of-the-art CoSODs could be easily affected by some adversarial perturbations, leading to substantial accuracy reduction. The adversarial perturbations can mislead CoSODs but do not change the high-level semantic information (e.g., concept) of the co-salient objects. In this paper, we propose a novel robustness enhancement framework by first learning the concept of the co-salient objects based on the input group images and then leveraging this concept to purify adversarial perturbations, which are subsequently fed to CoSODs for robustness enhancement. Specifically, we propose CosalPure containing two modules, i.e., group-image concept learning and concept-guided diffusion purification. For the first module, we adopt a pre-trained text-to-image diffusion model to learn the concept of co-salient objects within group images where the learned concept is robust to adversarial examples. For the second module, we map the adversarial image to the latent space and then perform diffusion generation by embedding the learned concept into the noise prediction function as an extra condition. Our method can effectively alleviate the influence of the SOTA adversarial attack containing different adversarial patterns, including exposure and noise. The extensive results demonstrate that our method could enhance the robustness of CoSODs significantly.

摘要: 共显著目标检测(CoSOD)的目的是识别给定图像组中的共同和显著(通常在前景中)区域。虽然取得了重大进展，但最先进的CoSOD很容易受到一些对抗性扰动的影响，导致精度大幅下降。对抗性扰动会误导CoSOD，但不会改变共显著对象的高级语义信息(例如，概念)。在本文中，我们提出了一种新的稳健性增强框架，该框架首先学习基于输入分组图像的共显著对象的概念，然后利用该概念来净化对抗性扰动，然后将这些扰动馈送到CoSOD以增强稳健性。具体地说，我们提出CosalPure包含两个模块，即组图像概念学习和概念引导的扩散净化。对于第一个模块，我们采用预先训练的文本到图像的扩散模型来学习组图像中共显著对象的概念，其中学习的概念对对抗性例子是健壮的。对于第二个模块，我们将敌意图像映射到潜在空间，然后通过将学习到的概念作为附加条件嵌入到噪声预测函数中来执行扩散生成。我们的方法可以有效地缓解SOTA对抗性攻击的影响，该攻击包含不同的对抗性模式，包括暴露和噪声。实验结果表明，该方法可以显著提高CoSOD的稳健性。



## **18. A Wolf in Sheep's Clothing: Generalized Nested Jailbreak Prompts can Fool Large Language Models Easily**

披着羊皮的狼：广义嵌套越狱陷阱可以轻松愚弄大型语言模型 cs.CL

Acccepted by NAACL 2024, 18 pages, 7 figures, 13 tables

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2311.08268v3) [paper-pdf](http://arxiv.org/pdf/2311.08268v3)

**Authors**: Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs), such as ChatGPT and GPT-4, are designed to provide useful and safe responses. However, adversarial prompts known as 'jailbreaks' can circumvent safeguards, leading LLMs to generate potentially harmful content. Exploring jailbreak prompts can help to better reveal the weaknesses of LLMs and further steer us to secure them. Unfortunately, existing jailbreak methods either suffer from intricate manual design or require optimization on other white-box models, which compromises either generalization or efficiency. In this paper, we generalize jailbreak prompt attacks into two aspects: (1) Prompt Rewriting and (2) Scenario Nesting. Based on this, we propose ReNeLLM, an automatic framework that leverages LLMs themselves to generate effective jailbreak prompts. Extensive experiments demonstrate that ReNeLLM significantly improves the attack success rate while greatly reducing the time cost compared to existing baselines. Our study also reveals the inadequacy of current defense methods in safeguarding LLMs. Finally, we analyze the failure of LLMs defense from the perspective of prompt execution priority, and propose corresponding defense strategies. We hope that our research can catalyze both the academic community and LLMs developers towards the provision of safer and more regulated LLMs. The code is available at https://github.com/NJUNLP/ReNeLLM.

摘要: 大型语言模型(LLM)，如ChatGPT和GPT-4，旨在提供有用和安全的响应。然而，被称为“越狱”的对抗性提示可能会绕过安全措施，导致LLMS生成潜在的有害内容。探索越狱提示可以帮助更好地揭示LLM的弱点，并进一步指导我们确保它们的安全。不幸的是，现有的越狱方法要么需要复杂的人工设计，要么需要对其他白盒模型进行优化，这要么损害了通用性，要么影响了效率。本文将越狱提示攻击概括为两个方面：(1)提示重写和(2)场景嵌套。在此基础上，我们提出了ReNeLLM，这是一个利用LLM自身生成有效越狱提示的自动化框架。广泛的实验表明，与现有的基准相比，ReNeLLM显著提高了攻击成功率，同时大大降低了时间成本。我们的研究也揭示了现有防御方法在保护低密度脂蛋白方面的不足。最后，从即时执行优先级的角度分析了LLMS防御失败的原因，并提出了相应的防御策略。我们希望我们的研究能够促进学术界和低成本管理系统开发商提供更安全和更规范的低成本管理系统。代码可在https://github.com/NJUNLP/ReNeLLM.上获得



## **19. Attack and Defense Analysis of Learned Image Compression**

学习图像压缩的攻防分析 eess.IV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2401.10345v3) [paper-pdf](http://arxiv.org/pdf/2401.10345v3)

**Authors**: Tianyu Zhu, Heming Sun, Xiankui Xiong, Xuanpeng Zhu, Yong Gong, Minge jing, Yibo Fan

**Abstract**: Learned image compression (LIC) is becoming more and more popular these years with its high efficiency and outstanding compression quality. Still, the practicality against modified inputs added with specific noise could not be ignored. White-box attacks such as FGSM and PGD use only gradient to compute adversarial images that mislead LIC models to output unexpected results. Our experiments compare the effects of different dimensions such as attack methods, models, qualities, and targets, concluding that in the worst case, there is a 61.55% decrease in PSNR or a 19.15 times increase in bpp under the PGD attack. To improve their robustness, we conduct adversarial training by adding adversarial images into the training datasets, which obtains a 95.52% decrease in the R-D cost of the most vulnerable LIC model. We further test the robustness of H.266, whose better performance on reconstruction quality extends its possibility to defend one-step or iterative adversarial attacks.

摘要: 近年来，学习图像压缩（LIC）以其高效率和优异的压缩质量得到了越来越广泛的应用。然而，对于添加特定噪声的修改输入的实用性也不容忽视。FGSM和PGD等白盒攻击仅使用梯度来计算对抗图像，从而误导LIC模型以输出意外结果。我们的实验比较了攻击方法、模型、质量和目标等不同维度的影响，得出的结论是，在最坏的情况下，PGD攻击下，PSNR下降了61.55%，bpp增加了19.15倍。为了提高它们的鲁棒性，我们通过将对抗图像添加到训练数据集中来进行对抗训练，这使得最脆弱的LIC模型的R—D成本降低了95.52%。我们进一步测试了H.266的鲁棒性，其在重建质量方面的更好性能扩展了其防御一步或迭代对抗攻击的可能性。



## **20. $\textit{LinkPrompt}$: Natural and Universal Adversarial Attacks on Prompt-based Language Models**

$\textit {LinkPrompt}$：基于XSLT语言模型的自然和普遍对抗攻击 cs.CL

Accepted to the main conference of NAACL2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.16432v2) [paper-pdf](http://arxiv.org/pdf/2403.16432v2)

**Authors**: Yue Xu, Wenjie Wang

**Abstract**: Prompt-based learning is a new language model training paradigm that adapts the Pre-trained Language Models (PLMs) to downstream tasks, which revitalizes the performance benchmarks across various natural language processing (NLP) tasks. Instead of using a fixed prompt template to fine-tune the model, some research demonstrates the effectiveness of searching for the prompt via optimization. Such prompt optimization process of prompt-based learning on PLMs also gives insight into generating adversarial prompts to mislead the model, raising concerns about the adversarial vulnerability of this paradigm. Recent studies have shown that universal adversarial triggers (UATs) can be generated to alter not only the predictions of the target PLMs but also the prediction of corresponding Prompt-based Fine-tuning Models (PFMs) under the prompt-based learning paradigm. However, UATs found in previous works are often unreadable tokens or characters and can be easily distinguished from natural texts with adaptive defenses. In this work, we consider the naturalness of the UATs and develop $\textit{LinkPrompt}$, an adversarial attack algorithm to generate UATs by a gradient-based beam search algorithm that not only effectively attacks the target PLMs and PFMs but also maintains the naturalness among the trigger tokens. Extensive results demonstrate the effectiveness of $\textit{LinkPrompt}$, as well as the transferability of UATs generated by $\textit{LinkPrompt}$ to open-sourced Large Language Model (LLM) Llama2 and API-accessed LLM GPT-3.5-turbo.

摘要: 基于提示的学习是一种新的语言模型训练范式，它使预先训练的语言模型(PLM)适应于下游任务，从而重振各种自然语言处理(NLP)任务的表现基准。一些研究证明了通过优化来搜索提示的有效性，而不是使用固定的提示模板来微调模型。这种基于提示的PLM学习的快速优化过程也为生成对抗性提示以误导模型提供了洞察力，这引发了人们对这种范式的对抗性脆弱性的担忧。最近的研究表明，在基于提示的学习范式下，通用对抗触发器(UAT)不仅可以改变目标PLM的预测，还可以改变相应的基于提示的精调模型(PFM)的预测。然而，在以前的著作中发现的UAT通常是不可读的符号或字符，并且可以很容易地与具有自适应防御的自然文本区分开来。在这项工作中，我们考虑了UAT的自然性，并开发了一种对抗性攻击算法，通过基于梯度的波束搜索算法来生成UAT，该算法不仅有效地攻击了目标PLM和PPM，而且保持了触发令牌之间的自然度。广泛的结果证明了$\textit{LinkPrompt}$的有效性，以及由$\textit{LinkPrompt}$生成的UAT可以移植到开源的大型语言模型(LLM)Llama2和API访问的LLm GPT-3.5-Turbo。



## **21. SemRoDe: Macro Adversarial Training to Learn Representations That are Robust to Word-Level Attacks**

SemRoDe：宏对抗训练，学习对单词级攻击具有鲁棒性的表示 cs.CL

Published in NAACL 2024 (Main Track)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18423v1) [paper-pdf](http://arxiv.org/pdf/2403.18423v1)

**Authors**: Brian Formento, Wenjie Feng, Chuan Sheng Foo, Luu Anh Tuan, See-Kiong Ng

**Abstract**: Language models (LMs) are indispensable tools for natural language processing tasks, but their vulnerability to adversarial attacks remains a concern. While current research has explored adversarial training techniques, their improvements to defend against word-level attacks have been limited. In this work, we propose a novel approach called Semantic Robust Defence (SemRoDe), a Macro Adversarial Training strategy to enhance the robustness of LMs. Drawing inspiration from recent studies in the image domain, we investigate and later confirm that in a discrete data setting such as language, adversarial samples generated via word substitutions do indeed belong to an adversarial domain exhibiting a high Wasserstein distance from the base domain. Our method learns a robust representation that bridges these two domains. We hypothesize that if samples were not projected into an adversarial domain, but instead to a domain with minimal shift, it would improve attack robustness. We align the domains by incorporating a new distance-based objective. With this, our model is able to learn more generalized representations by aligning the model's high-level output features and therefore better handling unseen adversarial samples. This method can be generalized across word embeddings, even when they share minimal overlap at both vocabulary and word-substitution levels. To evaluate the effectiveness of our approach, we conduct experiments on BERT and RoBERTa models on three datasets. The results demonstrate promising state-of-the-art robustness.

摘要: 语言模型(LMS)是自然语言处理任务中不可或缺的工具，但其易受敌意攻击的问题仍是一个令人担忧的问题。虽然目前的研究已经探索了对抗性训练技术，但它们在防御单词级攻击方面的改进有限。在这项工作中，我们提出了一种新的方法，称为语义稳健防御(SemRoDe)，这是一种宏观对抗性训练策略，以增强LMS的健壮性。受图像领域最新研究的启发，我们调查并证实，在语言等离散数据环境中，通过词替换生成的对抗性样本确实属于与基域有较高Wasserstein距离的对抗性领域。我们的方法学习了连接这两个域的健壮表示。我们假设，如果样本不被投影到敌对域，而是投影到位移最小的域，将提高攻击的稳健性。我们通过结合一个新的基于距离的目标来对齐域。有了这一点，我们的模型能够通过对齐模型的高级输出特征来学习更通用的表示，从而更好地处理看不见的敌意样本。这种方法可以在单词嵌入中推广，即使它们在词汇和单词替换级别上共享最小的重叠。为了评估该方法的有效性，我们在三个数据集上对Bert和Roberta模型进行了实验。结果表明，该方法具有良好的稳健性。



## **22. LocalStyleFool: Regional Video Style Transfer Attack Using Segment Anything Model**

LocalStyleFool：基于段任意模型的区域视频风格转移攻击 cs.CV

Accepted to 2024 IEEE Security and Privacy Workshops (SPW)

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.11656v2) [paper-pdf](http://arxiv.org/pdf/2403.11656v2)

**Authors**: Yuxin Cao, Jinghao Li, Xi Xiao, Derui Wang, Minhui Xue, Hao Ge, Wei Liu, Guangwu Hu

**Abstract**: Previous work has shown that well-crafted adversarial perturbations can threaten the security of video recognition systems. Attackers can invade such models with a low query budget when the perturbations are semantic-invariant, such as StyleFool. Despite the query efficiency, the naturalness of the minutia areas still requires amelioration, since StyleFool leverages style transfer to all pixels in each frame. To close the gap, we propose LocalStyleFool, an improved black-box video adversarial attack that superimposes regional style-transfer-based perturbations on videos. Benefiting from the popularity and scalably usability of Segment Anything Model (SAM), we first extract different regions according to semantic information and then track them through the video stream to maintain the temporal consistency. Then, we add style-transfer-based perturbations to several regions selected based on the associative criterion of transfer-based gradient information and regional area. Perturbation fine adjustment is followed to make stylized videos adversarial. We demonstrate that LocalStyleFool can improve both intra-frame and inter-frame naturalness through a human-assessed survey, while maintaining competitive fooling rate and query efficiency. Successful experiments on the high-resolution dataset also showcase that scrupulous segmentation of SAM helps to improve the scalability of adversarial attacks under high-resolution data.

摘要: 以往的工作表明，精心设计的对抗性扰动会威胁到视频识别系统的安全性。当扰动是语义不变的(如StyleFool)时，攻击者可以用较低的查询预算入侵这样的模型。尽管查询效率很高，但细节区域的自然度仍然需要改进，因为StyleFool利用样式传递到每帧中的所有像素。为了缩小这一差距，我们提出了LocalStyleFool，一种改进的黑盒视频对抗性攻击，将基于区域风格转移的扰动叠加到视频上。利用Segment Anything Model(SAM)的普及性和可伸缩性，我们首先根据语义信息提取不同的区域，然后通过视频流对它们进行跟踪，以保持时间一致性。然后，我们根据基于转移的梯度信息和区域面积的关联准则，将基于风格转移的扰动添加到选择的多个区域。随后进行微调，使风格化视频具有对抗性。通过一个人工评估的调查，我们证明了LocalStyleFool可以提高帧内和帧间的自然度，同时保持有竞争力的愚弄率和查询效率。在高分辨率数据集上的成功实验也表明，对SAM进行严格的分割有助于提高高分辨率数据下对抗性攻击的可扩展性。



## **23. Physical 3D Adversarial Attacks against Monocular Depth Estimation in Autonomous Driving**

自主驾驶中单目深度估计的物理3D对抗攻击 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.17301v2) [paper-pdf](http://arxiv.org/pdf/2403.17301v2)

**Authors**: Junhao Zheng, Chenhao Lin, Jiahao Sun, Zhengyu Zhao, Qian Li, Chao Shen

**Abstract**: Deep learning-based monocular depth estimation (MDE), extensively applied in autonomous driving, is known to be vulnerable to adversarial attacks. Previous physical attacks against MDE models rely on 2D adversarial patches, so they only affect a small, localized region in the MDE map but fail under various viewpoints. To address these limitations, we propose 3D Depth Fool (3D$^2$Fool), the first 3D texture-based adversarial attack against MDE models. 3D$^2$Fool is specifically optimized to generate 3D adversarial textures agnostic to model types of vehicles and to have improved robustness in bad weather conditions, such as rain and fog. Experimental results validate the superior performance of our 3D$^2$Fool across various scenarios, including vehicles, MDE models, weather conditions, and viewpoints. Real-world experiments with printed 3D textures on physical vehicle models further demonstrate that our 3D$^2$Fool can cause an MDE error of over 10 meters.

摘要: 基于深度学习的单目深度估计（MDE）广泛应用于自动驾驶中，已知容易受到对抗性攻击。以前针对MDE模型的物理攻击依赖于2D对抗补丁，因此它们只影响MDE地图中的一个小的局部区域，但在不同的视角下都会失败。为了解决这些限制，我们提出了3D Depth Fool（3D$^2$Fool），这是第一个针对MDE模型的基于3D纹理的对抗攻击。3D$^2$Fool经过专门优化，以生成3D对抗纹理，这些纹理不可知，以模型类型的车辆，并在恶劣天气条件下（如雨和雾）具有更高的鲁棒性。实验结果验证了我们的3D$^2$Fool在各种场景中的卓越性能，包括车辆、MDE模型、天气条件和视点。在实体车辆模型上使用打印3D纹理的真实实验进一步表明，我们的3D$^2$Fool可以导致超过10米的MDE误差。



## **24. Uncertainty-Aware SAR ATR: Defending Against Adversarial Attacks via Bayesian Neural Networks**

不确定性感知SAR ATR：利用贝叶斯神经网络防御对抗攻击 cs.CV

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18318v1) [paper-pdf](http://arxiv.org/pdf/2403.18318v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart

**Abstract**: Adversarial attacks have demonstrated the vulnerability of Machine Learning (ML) image classifiers in Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) systems. An adversarial attack can deceive the classifier into making incorrect predictions by perturbing the input SAR images, for example, with a few scatterers attached to the on-ground objects. Therefore, it is critical to develop robust SAR ATR systems that can detect potential adversarial attacks by leveraging the inherent uncertainty in ML classifiers, thereby effectively alerting human decision-makers. In this paper, we propose a novel uncertainty-aware SAR ATR for detecting adversarial attacks. Specifically, we leverage the capability of Bayesian Neural Networks (BNNs) in performing image classification with quantified epistemic uncertainty to measure the confidence for each input SAR image. By evaluating the uncertainty, our method alerts when the input SAR image is likely to be adversarially generated. Simultaneously, we also generate visual explanations that reveal the specific regions in the SAR image where the adversarial scatterers are likely to to be present, thus aiding human decision-making with hints of evidence of adversarial attacks. Experiments on the MSTAR dataset demonstrate that our approach can identify over 80% adversarial SAR images with fewer than 20% false alarms, and our visual explanations can identify up to over 90% of scatterers in an adversarial SAR image.

摘要: 敌意攻击已经证明了合成孔径雷达(SAR)自动目标识别(ATR)系统中机器学习(ML)图像分类器的脆弱性。对抗性攻击可以通过干扰输入的SAR图像来欺骗分类器做出错误的预测，例如，将一些散射体连接到地面对象上。因此，开发稳健的SAR ATR系统至关重要，它可以利用ML分类器中固有的不确定性来检测潜在的对手攻击，从而有效地向人类决策者发出警报。在本文中，我们提出了一种新的不确定性感知的SARATR来检测敌意攻击。具体地说，我们利用贝叶斯神经网络(BNN)在量化认知不确定性的情况下执行图像分类的能力来衡量每一幅输入SAR图像的置信度。通过评估不确定性，当输入的SAR图像可能被恶意生成时，我们的方法会发出警报。同时，我们还生成视觉解释，揭示SAR图像中可能存在对抗性散射体的特定区域，从而为人类决策提供对抗性攻击的证据提示。在MStar数据集上的实验表明，我们的方法可以识别80%以上的对抗性SAR图像，虚警率不到20%，我们的视觉解释可以识别高达90%以上的对抗性SAR图像中的散射体。



## **25. Bayesian Learned Models Can Detect Adversarial Malware For Free**

贝叶斯学习模型可以免费检测对抗性恶意软件 cs.CR

Accepted to the 29th European Symposium on Research in Computer  Security (ESORICS) 2024 Conference

**SubmitDate**: 2024-03-27    [abs](http://arxiv.org/abs/2403.18309v1) [paper-pdf](http://arxiv.org/pdf/2403.18309v1)

**Authors**: Bao Gia Doan, Dang Quang Nguyen, Paul Montague, Tamas Abraham, Olivier De Vel, Seyit Camtepe, Salil S. Kanhere, Ehsan Abbasnejad, Damith C. Ranasinghe

**Abstract**: The vulnerability of machine learning-based malware detectors to adversarial attacks has prompted the need for robust solutions. Adversarial training is an effective method but is computationally expensive to scale up to large datasets and comes at the cost of sacrificing model performance for robustness. We hypothesize that adversarial malware exploits the low-confidence regions of models and can be identified using epistemic uncertainty of ML approaches -- epistemic uncertainty in a machine learning-based malware detector is a result of a lack of similar training samples in regions of the problem space. In particular, a Bayesian formulation can capture the model parameters' distribution and quantify epistemic uncertainty without sacrificing model performance. To verify our hypothesis, we consider Bayesian learning approaches with a mutual information-based formulation to quantify uncertainty and detect adversarial malware in Android, Windows domains and PDF malware. We found, quantifying uncertainty through Bayesian learning methods can defend against adversarial malware. In particular, Bayesian models: (1) are generally capable of identifying adversarial malware in both feature and problem space, (2) can detect concept drift by measuring uncertainty, and (3) with a diversity-promoting approach (or better posterior approximations) lead to parameter instances from the posterior to significantly enhance a detectors' ability.

摘要: 基于机器学习的恶意软件检测器对对手攻击的脆弱性促使人们需要强大的解决方案。对抗性训练是一种有效的方法，但扩大到大型数据集的计算成本很高，而且是以牺牲模型性能来换取健壮性为代价的。我们假设敌意恶意软件利用了模型的低置信度区域，并可以使用ML方法的认知不确定性进行识别--基于机器学习的恶意软件检测器中的认知不确定性是由于问题空间区域中缺乏类似的训练样本造成的。特别是，贝叶斯公式可以捕捉模型参数的分布，并在不牺牲模型性能的情况下量化认知不确定性。为了验证我们的假设，我们考虑了基于互信息的贝叶斯学习方法来量化不确定性，并检测Android、Windows域和PDF恶意软件中的恶意软件。我们发现，通过贝叶斯学习方法量化不确定性可以防御敌意恶意软件。特别是，贝叶斯模型：(1)通常能够在特征和问题空间中识别恶意软件；(2)可以通过测量不确定性来检测概念漂移；(3)通过促进多样性的方法(或更好的后验近似)从后验获得参数实例，从而显著增强检测器的能力。



## **26. Bidirectional Consistency Models**

双向一致性模型 cs.LG

40 pages, 25 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18035v1) [paper-pdf](http://arxiv.org/pdf/2403.18035v1)

**Authors**: Liangchen Li, Jiajun He

**Abstract**: Diffusion models (DMs) are capable of generating remarkably high-quality samples by iteratively denoising a random vector, a process that corresponds to moving along the probability flow ordinary differential equation (PF ODE). Interestingly, DMs can also invert an input image to noise by moving backward along the PF ODE, a key operation for downstream tasks such as interpolation and image editing. However, the iterative nature of this process restricts its speed, hindering its broader application. Recently, Consistency Models (CMs) have emerged to address this challenge by approximating the integral of the PF ODE, thereby bypassing the need to iterate. Yet, the absence of an explicit ODE solver complicates the inversion process. To resolve this, we introduce the Bidirectional Consistency Model (BCM), which learns a single neural network that enables both forward and backward traversal along the PF ODE, efficiently unifying generation and inversion tasks within one framework. Notably, our proposed method enables one-step generation and inversion while also allowing the use of additional steps to enhance generation quality or reduce reconstruction error. Furthermore, by leveraging our model's bidirectional consistency, we introduce a sampling strategy that can enhance FID while preserving the generated image content. We further showcase our model's capabilities in several downstream tasks, such as interpolation and inpainting, and present demonstrations of potential applications, including blind restoration of compressed images and defending black-box adversarial attacks.

摘要: 扩散模型(DM)能够通过迭代地对随机向量去噪来生成非常高质量的样本，该过程对应于沿着概率流常微分方程式(PF ODE)移动。有趣的是，DM还可以通过沿PF ODE向后移动来将输入图像反转为噪声，这是下游任务(如插补和图像编辑)的关键操作。然而，这一过程的迭代性质限制了其速度，阻碍了其更广泛的应用。最近，一致性模型(CM)已经出现，通过近似PF ODE的积分来解决这一挑战，从而绕过了迭代的需要。然而，由于没有显式的常微分方程组解算器，使得反演过程变得更加复杂。为了解决这个问题，我们引入了双向一致性模型(BCM)，它学习一个单一的神经网络，允许沿着PF ODE进行前向和后向遍历，有效地将生成和反转任务统一在一个框架内。值得注意的是，我们提出的方法支持一步生成和反转，同时还允许使用额外的步骤来提高生成质量或减少重建误差。此外，通过利用模型的双向一致性，我们引入了一种采样策略，该策略可以在保留生成的图像内容的同时增强FID。我们进一步展示了我们的模型在几个下游任务中的能力，如插补和修复，并展示了潜在的应用程序，包括压缩图像的盲恢复和防御黑盒攻击。



## **27. Analyzing the Quality Attributes of AI Vision Models in Open Repositories Under Adversarial Attacks**

对抗攻击下开放仓库中人工智能视觉模型的质量属性分析 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2401.12261v2) [paper-pdf](http://arxiv.org/pdf/2401.12261v2)

**Authors**: Zerui Wang, Yan Liu

**Abstract**: As AI models rapidly evolve, they are frequently released to open repositories, such as HuggingFace. It is essential to perform quality assurance validation on these models before integrating them into the production development lifecycle. In addition to evaluating efficiency in terms of balanced accuracy and computing costs, adversarial attacks are potential threats to the robustness and explainability of AI models. Meanwhile, XAI applies algorithms that approximate inputs to outputs post-hoc to identify the contributing features. Adversarial perturbations may also degrade the utility of XAI explanations that require further investigation. In this paper, we present an integrated process designed for downstream evaluation tasks, including validating AI model accuracy, evaluating robustness with benchmark perturbations, comparing explanation utility, and assessing overhead. We demonstrate an evaluation scenario involving six computer vision models, which include CNN-based, Transformer-based, and hybrid architectures, three types of perturbations, and five XAI methods, resulting in ninety unique combinations. The process reveals the explanation utility among the XAI methods in terms of the identified key areas responding to the adversarial perturbation. The process produces aggregated results that illustrate multiple attributes of each AI model.

摘要: 随着AI模型的快速发展，它们经常被发布到开放存储库，如HuggingFace。在将这些模型集成到生产开发生命周期之前，对它们执行质量保证验证是至关重要的。除了在准确性和计算成本方面平衡评估效率外，对抗性攻击还对人工智能模型的健壮性和可解释性构成潜在威胁。同时，XAI将近似输入的算法应用于后期输出，以确定贡献特征。对抗性的干扰也可能降低Xai解释的效用，需要进一步的调查。在本文中，我们提出了一个为下游评估任务设计的完整过程，包括验证人工智能模型的准确性，评估基准扰动下的稳健性，比较解释效用，以及评估开销。我们演示了一个包含六个计算机视觉模型的评估场景，其中包括基于CNN的、基于变形金刚的和混合架构、三种类型的扰动和五种XAI方法，产生了90个独特的组合。这个过程揭示了XAI方法之间的解释效用，就识别的关键区域而言，对对抗性扰动的响应。该过程产生的聚合结果说明了每个AI模型的多个属性。



## **28. Secure Aggregation is Not Private Against Membership Inference Attacks**

安全聚合对成员身份推断攻击不是私有的 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17775v1) [paper-pdf](http://arxiv.org/pdf/2403.17775v1)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.

摘要: 安全聚合(SecAgg)是联合学习中常用的隐私增强机制，仅允许服务器访问模型更新的聚合，同时保护单个更新的机密性。尽管人们普遍声称SecAgg具有保护隐私的能力，但缺乏对其隐私的正式分析，这使得这种假设是不合理的。在本文中，我们深入研究了SecAgg的隐私含义，将其视为针对每个本地更新的本地差异隐私(LDP)机制。我们设计了一个简单的攻击，其中敌对服务器试图在SecAgg下的联合学习的单个训练轮中，从两个可能的更新向量中辨别客户端提交的更新向量。通过进行隐私审计，我们评估了该攻击的成功概率，并量化了SecAgg提供的LDP保证。我们的数值结果表明，与流行的说法相反，SecAgg即使在一轮训练中也提供了针对成员推理攻击的弱隐私。事实上，当更新是高维时，很难通过添加其他独立的本地更新来隐藏本地更新。我们的发现强调了联合学习中额外的隐私增强机制的必要性，例如噪音注入。



## **29. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

基于优化的LLM—as—a—Judge快速注入攻击 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17710v1) [paper-pdf](http://arxiv.org/pdf/2403.17710v1)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge is a novel solution that can assess textual information with large language models (LLMs). Based on existing research studies, LLMs demonstrate remarkable performance in providing a compelling alternative to traditional human assessment. However, the robustness of these systems against prompt injection attacks remains an open question. In this work, we introduce JudgeDeceiver, a novel optimization-based prompt injection attack tailored to LLM-as-a-Judge. Our method formulates a precise optimization objective for attacking the decision-making process of LLM-as-a-Judge and utilizes an optimization algorithm to efficiently automate the generation of adversarial sequences, achieving targeted and effective manipulation of model evaluations. Compared to handcraft prompt injection attacks, our method demonstrates superior efficacy, posing a significant challenge to the current security paradigms of LLM-based judgment systems. Through extensive experiments, we showcase the capability of JudgeDeceiver in altering decision outcomes across various cases, highlighting the vulnerability of LLM-as-a-Judge systems to the optimization-based prompt injection attack.

摘要: LLM-as-a-Court是一种新的解决方案，它可以使用大型语言模型(LLM)来评估文本信息。基于现有的研究，LLMS在提供一种令人信服的替代传统的人类评估方面表现出显著的性能。然而，这些系统对快速注入攻击的健壮性仍然是一个悬而未决的问题。在这项工作中，我们介绍了一种新的基于优化的快速注入攻击，该攻击是针对LLM-as-a-Court定制的。我们的方法为攻击LLM-as-a-Court的决策过程制定了一个精确的优化目标，并利用优化算法高效地自动生成对抗序列，实现了对模型评估的有针对性和有效的操作。与手工即时注入攻击相比，我们的方法表现出更好的有效性，对基于LLM的判断系统的现有安全范例提出了重大挑战。通过大量的实验，我们展示了JudgeDeceiver在改变不同案件的决策结果方面的能力，突出了LLM-as-a-Court系统对基于优化的即时注入攻击的脆弱性。



## **30. Towards more Practical Threat Models in Artificial Intelligence Security**

人工智能安全中更实用的威胁模型 cs.CR

18 pages, 4 figures, 8 tables, accepted to Usenix Security,  incorporated external feedback

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2311.09994v2) [paper-pdf](http://arxiv.org/pdf/2311.09994v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Alexandre Alahi

**Abstract**: Recent works have identified a gap between research and practice in artificial intelligence security: threats studied in academia do not always reflect the practical use and security risks of AI. For example, while models are often studied in isolation, they form part of larger ML pipelines in practice. Recent works also brought forward that adversarial manipulations introduced by academic attacks are impractical. We take a first step towards describing the full extent of this disparity. To this end, we revisit the threat models of the six most studied attacks in AI security research and match them to AI usage in practice via a survey with 271 industrial practitioners. On the one hand, we find that all existing threat models are indeed applicable. On the other hand, there are significant mismatches: research is often too generous with the attacker, assuming access to information not frequently available in real-world settings. Our paper is thus a call for action to study more practical threat models in artificial intelligence security.

摘要: 最近的研究发现了人工智能安全研究和实践之间的差距：学术界研究的威胁并不总是反映人工智能的实际使用和安全风险。例如，虽然模型通常是孤立研究的，但实际上它们构成了更大的ML管道的一部分。最近的研究也提出，学术攻击引入的对抗性操纵是不切实际的。我们朝着描述这种差距的全面程度迈出了第一步。为此，我们重新审视了人工智能安全研究中研究最多的六种攻击的威胁模型，并通过对271名行业从业者的调查，将它们与人工智能的实际使用相匹配。一方面，我们发现所有现有的威胁模型确实都是适用的。另一方面，存在严重的不匹配：研究往往对攻击者过于慷慨，假设他们可以访问现实世界中不常见的信息。因此，我们的论文呼吁采取行动，研究人工智能安全中更实用的威胁模型。



## **31. Targeted Visualization of the Backbone of Encoder LLMs**

编码器LLM骨干的目标可视化 cs.LG

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.18872v1) [paper-pdf](http://arxiv.org/pdf/2403.18872v1)

**Authors**: Isaac Roberts, Alexander Schulz, Luca Hermes, Barbara Hammer

**Abstract**: Attention based Large Language Models (LLMs) are the state-of-the-art in natural language processing (NLP). The two most common architectures are encoders such as BERT, and decoders like the GPT models. Despite the success of encoder models, on which we focus in this work, they also bear several risks, including issues with bias or their susceptibility for adversarial attacks, signifying the necessity for explainable AI to detect such issues. While there does exist various local explainability methods focusing on the prediction of single inputs, global methods based on dimensionality reduction for classification inspection, which have emerged in other domains and that go further than just using t-SNE in the embedding space, are not widely spread in NLP.   To reduce this gap, we investigate the application of DeepView, a method for visualizing a part of the decision function together with a data set in two dimensions, to the NLP domain. While in previous work, DeepView has been used to inspect deep image classification models, we demonstrate how to apply it to BERT-based NLP classifiers and investigate its usability in this domain, including settings with adversarially perturbed input samples and pre-trained, fine-tuned, and multi-task models.

摘要: 基于注意力的大语言模型(LLM)是自然语言处理(NLP)领域的前沿技术。两种最常见的架构是编码器(如BERT)和解码器(如GPT模型)。尽管我们在本工作中重点关注的编码器模型取得了成功，但它们也存在几个风险，包括偏见或它们对对抗性攻击的敏感性问题，这意味着有必要使用可解释的人工智能来检测此类问题。虽然有各种局部可解释方法专注于单输入预测，但在其他领域出现的基于降维的全局分类检测方法并没有在NLP中广泛推广，这些方法比仅仅使用嵌入空间中的t-SNE更深入。为了缩小这一差距，我们研究了DeepView在NLP领域的应用，DeepView是一种将决策函数的一部分与二维数据集一起可视化的方法。在以前的工作中，DeepView已经被用来检查深度图像分类模型，我们演示了如何将其应用于基于BERT的NLP分类器，并研究了它在该领域的可用性，包括设置了相反扰动的输入样本和预先训练的、微调的和多任务模型。



## **32. FaultGuard: A Generative Approach to Resilient Fault Prediction in Smart Electrical Grids**

FaultGuard：智能电网弹性故障预测的生成方法 cs.CR

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.17494v1) [paper-pdf](http://arxiv.org/pdf/2403.17494v1)

**Authors**: Emad Efatinasab, Francesco Marchiori, Alessandro Brighente, Mirco Rampazzo, Mauro Conti

**Abstract**: Predicting and classifying faults in electricity networks is crucial for uninterrupted provision and keeping maintenance costs at a minimum. Thanks to the advancements in the field provided by the smart grid, several data-driven approaches have been proposed in the literature to tackle fault prediction tasks. Implementing these systems brought several improvements, such as optimal energy consumption and quick restoration. Thus, they have become an essential component of the smart grid. However, the robustness and security of these systems against adversarial attacks have not yet been extensively investigated. These attacks can impair the whole grid and cause additional damage to the infrastructure, deceiving fault detection systems and disrupting restoration. In this paper, we present FaultGuard, the first framework for fault type and zone classification resilient to adversarial attacks. To ensure the security of our system, we employ an Anomaly Detection System (ADS) leveraging a novel Generative Adversarial Network training layer to identify attacks. Furthermore, we propose a low-complexity fault prediction model and an online adversarial training technique to enhance robustness. We comprehensively evaluate the framework's performance against various adversarial attacks using the IEEE13-AdvAttack dataset, which constitutes the state-of-the-art for resilient fault prediction benchmarking. Our model outclasses the state-of-the-art even without considering adversaries, with an accuracy of up to 0.958. Furthermore, our ADS shows attack detection capabilities with an accuracy of up to 1.000. Finally, we demonstrate how our novel training layers drastically increase performances across the whole framework, with a mean increase of 154% in ADS accuracy and 118% in model accuracy.

摘要: 预测和分类电网中的故障对于不间断供电和将维护成本保持在最低水平至关重要。由于智能电网在该领域的进步，文献中已经提出了几种数据驱动的方法来处理故障预测任务。实施这些系统带来了一些改进，例如最佳的能源消耗和快速恢复。因此，它们已成为智能电网的重要组成部分。然而，这些系统对敌意攻击的健壮性和安全性还没有得到广泛的研究。这些攻击可能会损害整个电网，并对基础设施造成额外的破坏，欺骗故障检测系统并中断恢复。在本文中，我们提出了FaultGuard，这是第一个对对手攻击具有弹性的故障类型和区域分类框架。为了确保系统的安全性，我们使用了一个异常检测系统(ADS)，该系统利用一个新的生成性对抗性网络训练层来识别攻击。此外，我们还提出了一种低复杂度的故障预测模型和在线对抗性训练技术来增强鲁棒性。我们使用IEEE13-AdvAttack数据集全面评估了该框架对各种敌意攻击的性能，该数据集构成了弹性故障预测基准测试的最新技术。我们的模型甚至在不考虑对手的情况下也超过了最先进的模型，准确率高达0.958。此外，我们的广告显示了攻击检测能力，准确率高达1.000。最后，我们展示了我们的新型训练层如何显著提高了整个框架的性能，ADS准确率平均提高了154%，模型准确率平均提高了118%。



## **33. Rumor Detection with a novel graph neural network approach**

基于图神经网络的谣言检测方法 cs.AI

10 pages, 5 figures

**SubmitDate**: 2024-03-26    [abs](http://arxiv.org/abs/2403.16206v2) [paper-pdf](http://arxiv.org/pdf/2403.16206v2)

**Authors**: Tianrui Liu, Qi Cai, Changxin Xu, Bo Hong, Fanghao Ni, Yuxin Qiao, Tsungwei Yang

**Abstract**: The wide spread of rumors on social media has caused a negative impact on people's daily life, leading to potential panic, fear, and mental health problems for the public. How to debunk rumors as early as possible remains a challenging problem. Existing studies mainly leverage information propagation structure to detect rumors, while very few works focus on correlation among users that they may coordinate to spread rumors in order to gain large popularity. In this paper, we propose a new detection model, that jointly learns both the representations of user correlation and information propagation to detect rumors on social media. Specifically, we leverage graph neural networks to learn the representations of user correlation from a bipartite graph that describes the correlations between users and source tweets, and the representations of information propagation with a tree structure. Then we combine the learned representations from these two modules to classify the rumors. Since malicious users intend to subvert our model after deployment, we further develop a greedy attack scheme to analyze the cost of three adversarial attacks: graph attack, comment attack, and joint attack. Evaluation results on two public datasets illustrate that the proposed MODEL outperforms the state-of-the-art rumor detection models. We also demonstrate our method performs well for early rumor detection. Moreover, the proposed detection method is more robust to adversarial attacks compared to the best existing method. Importantly, we show that it requires a high cost for attackers to subvert user correlation pattern, demonstrating the importance of considering user correlation for rumor detection.

摘要: 谣言在社交媒体上的广泛传播对人们的日常生活造成了负面影响，给公众带来了潜在的恐慌、恐惧和心理健康问题。如何尽早揭穿谣言仍是一个具有挑战性的问题。现有的研究主要是利用信息传播结构来发现谣言，而很少有人关注用户之间的相关性，他们可能会协同传播谣言以获得更大的人气。在本文中，我们提出了一种新的检测模型，该模型同时学习用户相关性和信息传播的表示，以检测社交媒体上的谣言。具体地说，我们利用图神经网络从描述用户和源推文之间的相关性的二部图中学习用户相关性的表示，以及用树结构表示信息传播。然后，我们结合这两个模块的学习表示来对谣言进行分类。由于恶意用户在部署后有意颠覆我们的模型，我们进一步开发了一种贪婪攻击方案，分析了图攻击、评论攻击和联合攻击三种对抗性攻击的代价。在两个公开数据集上的评估结果表明，该模型的性能优于最新的谣言检测模型。我们还证明了我们的方法在早期谣言检测中表现良好。此外，与现有的最佳检测方法相比，本文提出的检测方法对敌意攻击具有更强的鲁棒性。重要的是，我们证明了攻击者要颠覆用户相关性模式需要付出很高的代价，这说明了考虑用户相关性对谣言检测的重要性。



## **34. The Anatomy of Adversarial Attacks: Concept-based XAI Dissection**

对抗性攻击的剖析：基于概念的XAI解剖 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16782v1) [paper-pdf](http://arxiv.org/pdf/2403.16782v1)

**Authors**: Georgii Mikriukov, Gesina Schwalbe, Franz Motzkus, Korinna Bade

**Abstract**: Adversarial attacks (AAs) pose a significant threat to the reliability and robustness of deep neural networks. While the impact of these attacks on model predictions has been extensively studied, their effect on the learned representations and concepts within these models remains largely unexplored. In this work, we perform an in-depth analysis of the influence of AAs on the concepts learned by convolutional neural networks (CNNs) using eXplainable artificial intelligence (XAI) techniques. Through an extensive set of experiments across various network architectures and targeted AA techniques, we unveil several key findings. First, AAs induce substantial alterations in the concept composition within the feature space, introducing new concepts or modifying existing ones. Second, the adversarial perturbation itself can be linearly decomposed into a set of latent vector components, with a subset of these being responsible for the attack's success. Notably, we discover that these components are target-specific, i.e., are similar for a given target class throughout different AA techniques and starting classes. Our findings provide valuable insights into the nature of AAs and their impact on learned representations, paving the way for the development of more robust and interpretable deep learning models, as well as effective defenses against adversarial threats.

摘要: 对抗性攻击(AAs)对深度神经网络的可靠性和健壮性构成了严重威胁。虽然这些攻击对模型预测的影响已经被广泛研究，但它们对这些模型中的学习表示和概念的影响在很大程度上仍未被探索。在这项工作中，我们使用可解释人工智能(XAI)技术深入分析了人工智能对卷积神经网络(CNN)学习的概念的影响。通过跨各种网络架构和有针对性的AA技术进行的一组广泛的实验，我们揭示了几个关键发现。首先，人工智能在特征空间内引起概念构成的实质性变化，引入新的概念或修改现有的概念。其次，敌方扰动本身可以线性分解为一组潜在向量分量，这些分量的子集是攻击成功的原因。值得注意的是，我们发现这些组件是特定于目标的，即在不同的AA技术和起始类中，对于给定的目标类是相似的。我们的发现对人工智能的性质及其对学习陈述的影响提供了有价值的见解，为开发更健壮和可解释的深度学习模型以及有效防御对手威胁铺平了道路。



## **35. DeepKnowledge: Generalisation-Driven Deep Learning Testing**

DeepKnowledge：泛化驱动的深度学习测试 cs.LG

10 pages

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16768v1) [paper-pdf](http://arxiv.org/pdf/2403.16768v1)

**Authors**: Sondess Missaoui, Simos Gerasimou, Nikolaos Matragkas

**Abstract**: Despite their unprecedented success, DNNs are notoriously fragile to small shifts in data distribution, demanding effective testing techniques that can assess their dependability. Despite recent advances in DNN testing, there is a lack of systematic testing approaches that assess the DNN's capability to generalise and operate comparably beyond data in their training distribution. We address this gap with DeepKnowledge, a systematic testing methodology for DNN-based systems founded on the theory of knowledge generalisation, which aims to enhance DNN robustness and reduce the residual risk of 'black box' models. Conforming to this theory, DeepKnowledge posits that core computational DNN units, termed Transfer Knowledge neurons, can generalise under domain shift. DeepKnowledge provides an objective confidence measurement on testing activities of DNN given data distribution shifts and uses this information to instrument a generalisation-informed test adequacy criterion to check the transfer knowledge capacity of a test set. Our empirical evaluation of several DNNs, across multiple datasets and state-of-the-art adversarial generation techniques demonstrates the usefulness and effectiveness of DeepKnowledge and its ability to support the engineering of more dependable DNNs. We report improvements of up to 10 percentage points over state-of-the-art coverage criteria for detecting adversarial attacks on several benchmarks, including MNIST, SVHN, and CIFAR.

摘要: 尽管DNN取得了前所未有的成功，但众所周知，它们对数据分布的微小变化很脆弱，需要有效的测试技术来评估它们的可靠性。尽管最近在DNN测试方面取得了进展，但缺乏系统的测试方法来评估DNN在其训练分布中的数据之外的泛化和相对操作的能力。我们用DeepKnowledge解决了这一差距，DeepKnowledge是一种基于知识泛化理论的DNN系统测试方法，旨在增强DNN的健壮性并降低“黑盒”模型的残余风险。与这一理论相一致的是，DeepKnowledge提出，核心计算DNN单元，称为传递知识神经元，可以在域转移下泛化。DeepKnowledge在给定数据分布漂移的情况下对DNN的测试活动提供了客观的置信度度量，并使用该信息来测试泛化信息的测试充分性标准，以检查测试集的传递知识能力。我们对几个DNN的经验评估，跨越多个数据集和最先进的对手生成技术，证明了DeepKnowledge的有用性和有效性，以及它支持设计更可靠的DNN的能力。我们报告了在包括MNIST、SVHN和CIFAR在内的几个基准上检测对抗性攻击的最新覆盖标准的改进，提高了高达10个百分点。



## **36. Boosting Adversarial Transferability by Block Shuffle and Rotation**

利用块洗牌和旋转增强对抗性传递 cs.CV

Accepted by CVPR 2024

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.10299v3) [paper-pdf](http://arxiv.org/pdf/2308.10299v3)

**Authors**: Kunyu Wang, Xuanran He, Wenxuan Wang, Xiaosen Wang

**Abstract**: Adversarial examples mislead deep neural networks with imperceptible perturbations and have brought significant threats to deep learning. An important aspect is their transferability, which refers to their ability to deceive other models, thus enabling attacks in the black-box setting. Though various methods have been proposed to boost transferability, the performance still falls short compared with white-box attacks. In this work, we observe that existing input transformation based attacks, one of the mainstream transfer-based attacks, result in different attention heatmaps on various models, which might limit the transferability. We also find that breaking the intrinsic relation of the image can disrupt the attention heatmap of the original image. Based on this finding, we propose a novel input transformation based attack called block shuffle and rotation (BSR). Specifically, BSR splits the input image into several blocks, then randomly shuffles and rotates these blocks to construct a set of new images for gradient calculation. Empirical evaluations on the ImageNet dataset demonstrate that BSR could achieve significantly better transferability than the existing input transformation based methods under single-model and ensemble-model settings. Combining BSR with the current input transformation method can further improve the transferability, which significantly outperforms the state-of-the-art methods. Code is available at https://github.com/Trustworthy-AI-Group/BSR

摘要: 对抗性例子用潜移默化的扰动误导了深度神经网络，给深度学习带来了重大威胁。一个重要的方面是它们的可转移性，这指的是它们欺骗其他模型的能力，从而使攻击能够在黑盒环境中进行。虽然已经提出了各种方法来提高可转移性，但与白盒攻击相比，性能仍然不足。在这项工作中，我们观察到现有的基于输入变换的攻击是基于转移的主流攻击之一，在不同的模型上会导致不同的注意力热图，这可能会限制可转移性。我们还发现，打破图像的内在联系会扰乱原始图像的注意热图。基于这一发现，我们提出了一种新的基于输入变换的攻击方法，称为块置乱和旋转攻击(BSR)。具体地说，BSR将输入图像分成几个块，然后随机地对这些块进行洗牌和旋转，以构建一组新的图像用于梯度计算。在ImageNet数据集上的实证评估表明，在单模型和集成模型的设置下，BSR可以获得比现有的基于输入变换的方法更好的可转移性。将BSR与当前的输入变换方法相结合，可以进一步提高可转移性，显著优于最先进的方法。代码可在https://github.com/Trustworthy-AI-Group/BSR上找到



## **37. A Huber Loss Minimization Approach to Byzantine Robust Federated Learning**

拜占庭鲁棒联邦学习的Huber损失最小化方法 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2308.12581v2) [paper-pdf](http://arxiv.org/pdf/2308.12581v2)

**Authors**: Puning Zhao, Fei Yu, Zhiguo Wan

**Abstract**: Federated learning systems are susceptible to adversarial attacks. To combat this, we introduce a novel aggregator based on Huber loss minimization, and provide a comprehensive theoretical analysis. Under independent and identically distributed (i.i.d) assumption, our approach has several advantages compared to existing methods. Firstly, it has optimal dependence on $\epsilon$, which stands for the ratio of attacked clients. Secondly, our approach does not need precise knowledge of $\epsilon$. Thirdly, it allows different clients to have unequal data sizes. We then broaden our analysis to include non-i.i.d data, such that clients have slightly different distributions.

摘要: 联邦学习系统容易受到对抗性攻击。为了解决这一问题，我们提出了一种新的基于Huber损失最小化的聚合器，并提供了全面的理论分析。在独立同分布（i.i. d）假设下，我们的方法与现有方法相比具有几个优点。首先，它对$\rn $具有最优依赖性，它代表受攻击客户端的比率。第二，我们的方法不需要精确的$\rn $知识。第三，它允许不同的客户端具有不相等的数据大小。然后，我们扩大我们的分析范围，包括非i.i. d数据，这样客户的分布略有不同。



## **38. Revealing Vulnerabilities of Neural Networks in Parameter Learning and Defense Against Explanation-Aware Backdoors**

揭示神经网络参数学习中的漏洞及防范知识后门 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16569v1) [paper-pdf](http://arxiv.org/pdf/2403.16569v1)

**Authors**: Md Abdul Kadir, GowthamKrishna Addluri, Daniel Sonntag

**Abstract**: Explainable Artificial Intelligence (XAI) strategies play a crucial part in increasing the understanding and trustworthiness of neural networks. Nonetheless, these techniques could potentially generate misleading explanations. Blinding attacks can drastically alter a machine learning algorithm's prediction and explanation, providing misleading information by adding visually unnoticeable artifacts into the input, while maintaining the model's accuracy. It poses a serious challenge in ensuring the reliability of XAI methods. To ensure the reliability of XAI methods poses a real challenge, we leverage statistical analysis to highlight the changes in CNN weights within a CNN following blinding attacks. We introduce a method specifically designed to limit the effectiveness of such attacks during the evaluation phase, avoiding the need for extra training. The method we suggest defences against most modern explanation-aware adversarial attacks, achieving an approximate decrease of ~99\% in the Attack Success Rate (ASR) and a ~91\% reduction in the Mean Square Error (MSE) between the original explanation and the defended (post-attack) explanation across three unique types of attacks.

摘要: 可解释人工智能(XAI)策略在增加神经网络的理解和可信度方面发挥着至关重要的作用。然而，这些技术可能会产生误导性的解释。盲目攻击可以极大地改变机器学习算法的预测和解释，通过在输入中添加视觉上不可察觉的伪像来提供误导性信息，同时保持模型的准确性。这对确保XAI方法的可靠性提出了严峻的挑战。为了确保XAI方法的可靠性构成了一个真正的挑战，我们利用统计分析来突出CNN在盲人攻击后CNN权重的变化。我们引入了一种专门设计的方法，在评估阶段限制此类攻击的有效性，避免了额外培训的需要。我们提出的方法可以防御大多数现代解释感知的对手攻击，在三种独特的攻击类型中，攻击成功率(ASR)大约降低了99%，原始解释和防御(攻击后)解释之间的均方误差(MSE)降低了约91%。



## **39. Exploring the Adversarial Capabilities of Large Language Models**

探索大型语言模型的对抗能力 cs.AI

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2402.09132v3) [paper-pdf](http://arxiv.org/pdf/2402.09132v3)

**Authors**: Lukas Struppek, Minh Hieu Le, Dominik Hintersdorf, Kristian Kersting

**Abstract**: The proliferation of large language models (LLMs) has sparked widespread and general interest due to their strong language generation capabilities, offering great potential for both industry and research. While previous research delved into the security and privacy issues of LLMs, the extent to which these models can exhibit adversarial behavior remains largely unexplored. Addressing this gap, we investigate whether common publicly available LLMs have inherent capabilities to perturb text samples to fool safety measures, so-called adversarial examples resp.~attacks. More specifically, we investigate whether LLMs are inherently able to craft adversarial examples out of benign samples to fool existing safe rails. Our experiments, which focus on hate speech detection, reveal that LLMs succeed in finding adversarial perturbations, effectively undermining hate speech detection systems. Our findings carry significant implications for (semi-)autonomous systems relying on LLMs, highlighting potential challenges in their interaction with existing systems and safety measures.

摘要: 大型语言模型因其强大的语言生成能力而引起了广泛的关注，为工业和研究提供了巨大的潜力。虽然之前的研究已经深入研究了LLMS的安全和隐私问题，但这些模型在多大程度上可以表现出敌对行为，仍然很大程度上还没有被探索。针对这一差距，我们调查了常见的公开可用的LLM是否具有固有的能力来扰乱文本样本以愚弄安全措施，即所谓的对抗性示例攻击。更具体地说，我们调查LLM是否天生就能够从良性样本中制作敌意示例，以愚弄现有的安全Rail。我们的实验集中在仇恨语音检测上，实验表明，LLMS成功地发现了敌意扰动，有效地破坏了仇恨语音检测系统。我们的发现对依赖LLMS的(半)自治系统具有重大影响，突显了它们与现有系统和安全措施相互作用的潜在挑战。



## **40. On the resilience of Collaborative Learning-based Recommender Systems Against Community Detection Attack**

基于协作学习的推荐系统对社区检测攻击的抵抗能力研究 cs.IR

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2306.08929v2) [paper-pdf](http://arxiv.org/pdf/2306.08929v2)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while maintaining their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at first glance, recent studies have revealed that collaborative learning can be vulnerable to various privacy attacks. In this paper, we study the resilience of collaborative learning-based recommender systems against a novel privacy attack called Community Detection Attack (CDA). This attack enables an adversary to identify community members based on a chosen set of items (eg., identifying users interested in specific points-of-interest). Through experiments on three real recommendation datasets using two state-of-the-art recommendation models, we evaluate the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. The results show that across all models and datasets, the FL setting is more vulnerable to CDA compared to Gossip settings. Furthermore, we assess two off-the-shelf mitigation strategies, namely differential privacy (DP) and a \emph{Share less} policy, which consists of sharing a subset of less sensitive model parameters. The findings indicate a more favorable privacy-utility trade-off for the \emph{Share less} strategy, particularly in FedRecs.

摘要: 基于协作学习的推荐系统是在联邦学习(FL)和八卦学习(GL)等协作学习技术成功之后应运而生的。在这些系统中，用户参与推荐系统的培训，同时在他们的设备上维护他们的消费项目的历史。乍一看，这些解决方案在保护参与者隐私方面似乎很有吸引力，但最近的研究表明，协作学习可能容易受到各种隐私攻击。本文研究了基于协作学习的推荐系统对一种新的隐私攻击--社区检测攻击(CDA)的恢复能力。这种攻击使对手能够根据选定的一组项目识别社区成员(例如，识别对特定兴趣点感兴趣的用户)。通过使用两种最新推荐模型在三个真实推荐数据集上的实验，我们评估了一个基于FL的推荐系统以及两种基于八卦学习的推荐系统对CDA的敏感度。结果表明，在所有模型和数据集中，与八卦设置相比，FL设置更容易受到CDA的影响。此外，我们评估了两种现成的缓解策略，即差异隐私(DP)和共享较少的策略，该策略包括共享不太敏感的模型参数的子集。研究结果表明，EMPH{Share Less}策略的隐私效用权衡更有利，尤其是在FedRecs中。



## **41. Model-less Is the Best Model: Generating Pure Code Implementations to Replace On-Device DL Models**

无模型是最好的模型：生成纯代码实现来替换设备上的DL模型 cs.SE

Accepted by the ACM SIGSOFT International Symposium on Software  Testing and Analysis (ISSTA2024)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16479v1) [paper-pdf](http://arxiv.org/pdf/2403.16479v1)

**Authors**: Mingyi Zhou, Xiang Gao, Pei Liu, John Grundy, Chunyang Chen, Xiao Chen, Li Li

**Abstract**: Recent studies show that deployed deep learning (DL) models such as those of Tensor Flow Lite (TFLite) can be easily extracted from real-world applications and devices by attackers to generate many kinds of attacks like adversarial attacks. Although securing deployed on-device DL models has gained increasing attention, no existing methods can fully prevent the aforementioned threats. Traditional software protection techniques have been widely explored, if on-device models can be implemented using pure code, such as C++, it will open the possibility of reusing existing software protection techniques. However, due to the complexity of DL models, there is no automatic method that can translate the DL models to pure code. To fill this gap, we propose a novel method, CustomDLCoder, to automatically extract the on-device model information and synthesize a customized executable program for a wide range of DL models. CustomDLCoder first parses the DL model, extracts its backend computing units, configures the computing units to a graph, and then generates customized code to implement and deploy the ML solution without explicit model representation. The synthesized program hides model information for DL deployment environments since it does not need to retain explicit model representation, preventing many attacks on the DL model. In addition, it improves ML performance because the customized code removes model parsing and preprocessing steps and only retains the data computing process. Our experimental results show that CustomDLCoder improves model security by disabling on-device model sniffing. Compared with the original on-device platform (i.e., TFLite), our method can accelerate model inference by 21.0% and 24.3% on x86-64 and ARM64 platforms, respectively. Most importantly, it can significantly reduce memory consumption by 68.8% and 36.0% on x86-64 and ARM64 platforms, respectively.

摘要: 最近的研究表明，部署的深度学习(DL)模型，如张量流精简(TFLite)模型，可以很容易地被攻击者从现实世界的应用和设备中提取出来，从而产生多种攻击，如对抗性攻击。尽管保护部署在设备上的DL模型越来越受到关注，但没有一种现有方法可以完全防止上述威胁。传统的软件保护技术已经得到了广泛的探索，如果设备上的模型可以用纯代码实现，如C++，这将打开重用现有软件保护技术的可能性。然而，由于DL模型的复杂性，目前还没有一种自动的方法可以将DL模型转换为纯代码。为了填补这一空白，我们提出了一种新的方法CustomDLCoder，它可以自动提取设备上的模型信息，并为广泛的DL模型合成定制的可执行程序。CustomDLCoder首先解析DL模型，提取其后端计算单元，将计算单元配置为图形，然后生成定制代码来实现和部署ML解决方案，而不需要显式的模型表示。合成的程序隐藏了DL部署环境的模型信息，因为它不需要保留显式的模型表示，从而防止了对DL模型的许多攻击。此外，它还提高了ML的性能，因为定制的代码删除了模型解析和预处理步骤，只保留了数据计算过程。我们的实验结果表明，CustomDLCoder通过禁止设备上的模型嗅探提高了模型的安全性。在x86-64和ARM64平台上，与原有的设备上平台(即TFLite)相比，该方法的模型推理速度分别提高了21.0%和24.3%。最重要的是，它可以在x86-64和ARM64平台上分别显著降低68.8%和36.0%的内存消耗。



## **42. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

使用信任感知鲁棒事件触发控制屏障函数的联网和自动车辆安全控制 eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2401.02306v3) [paper-pdf](http://arxiv.org/pdf/2401.02306v3)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.

摘要: 我们致力于解决互联和自动化车辆(CAV)网络的安全问题，这些车辆通过协作安全地通过冲突区域(例如，交通路口、合并道路、环形交叉路口)。以前的研究表明，这样的网络可以成为导致交通拥堵或以碰撞结束的安全违规行为的对抗性攻击的目标。我们专注于针对用于共享车辆数据的V2X通信网络的攻击，并考虑由于传感器测量和通信通道中的噪声而产生的不确定性。为了应对这些问题，基于最近在CAV安全控制方面的工作，我们提出了一个信任感知的、健壮的、事件触发的分布式控制和协调框架，该框架能够有效地保证安全。我们为网络中的每辆车维护一个基于其行为计算的信任度量，用于平衡保守性(当认为每辆车不值得信任时)与保证的安全和保障之间的权衡。必须强调的是，我们的框架与信任框架的具体选择是不变的。基于该框架，我们提出了一种攻击检测和缓解方案，该方案具有两个优点：(I)信任框架不受误报的影响；(Ii)它可证明地保证了对误报情况的安全性。我们使用大量的仿真(在相扑和CALA中)来验证理论上的保证，并展示了我们所提出的方案在检测和缓解敌意攻击方面的有效性。



## **43. Ensemble Adversarial Defense via Integration of Multiple Dispersed Low Curvature Models**

多离散低曲率模型集成对抗防御 cs.LG

Accepted to The 2024 International Joint Conference on Neural  Networks (IJCNN)

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16405v1) [paper-pdf](http://arxiv.org/pdf/2403.16405v1)

**Authors**: Kaikang Zhao, Xi Chen, Wei Huang, Liuxin Ding, Xianglong Kong, Fan Zhang

**Abstract**: The integration of an ensemble of deep learning models has been extensively explored to enhance defense against adversarial attacks. The diversity among sub-models increases the attack cost required to deceive the majority of the ensemble, thereby improving the adversarial robustness. While existing approaches mainly center on increasing diversity in feature representations or dispersion of first-order gradients with respect to input, the limited correlation between these diversity metrics and adversarial robustness constrains the performance of ensemble adversarial defense. In this work, we aim to enhance ensemble diversity by reducing attack transferability. We identify second-order gradients, which depict the loss curvature, as a key factor in adversarial robustness. Computing the Hessian matrix involved in second-order gradients is computationally expensive. To address this, we approximate the Hessian-vector product using differential approximation. Given that low curvature provides better robustness, our ensemble model was designed to consider the influence of curvature among different sub-models. We introduce a novel regularizer to train multiple more-diverse low-curvature network models. Extensive experiments across various datasets demonstrate that our ensemble model exhibits superior robustness against a range of attacks, underscoring the effectiveness of our approach.

摘要: 集成深度学习模型已被广泛探索，以增强对对手攻击的防御。子模型之间的多样性增加了欺骗大部分集成所需的攻击成本，从而提高了对手的稳健性。虽然现有的方法主要集中在增加特征表示的多样性或关于输入的一阶梯度的离散，但这些多样性度量与对抗稳健性之间的有限相关性限制了集成对抗防御的性能。在这项工作中，我们的目标是通过降低攻击的可转移性来提高集合的多样性。我们认为描述损失曲率的二阶梯度是对抗健壮性的一个关键因素。计算二阶梯度所涉及的海森矩阵在计算上是昂贵的。为了解决这个问题，我们使用微分近似来近似黑森向量积。考虑到低曲率提供了更好的稳健性，我们的集成模型被设计为考虑曲率在不同子模型之间的影响。我们引入了一种新的正则化方法来训练多个更多样化的低曲率网络模型。在不同数据集上的广泛实验表明，我们的集成模型对一系列攻击表现出了卓越的鲁棒性，强调了我们方法的有效性。



## **44. Generating Potent Poisons and Backdoors from Scratch with Guided Diffusion**

利用引导扩散从划痕中产生潜在毒药和后门 cs.LG

**SubmitDate**: 2024-03-25    [abs](http://arxiv.org/abs/2403.16365v1) [paper-pdf](http://arxiv.org/pdf/2403.16365v1)

**Authors**: Hossein Souri, Arpit Bansal, Hamid Kazemi, Liam Fowl, Aniruddha Saha, Jonas Geiping, Andrew Gordon Wilson, Rama Chellappa, Tom Goldstein, Micah Goldblum

**Abstract**: Modern neural networks are often trained on massive datasets that are web scraped with minimal human inspection. As a result of this insecure curation pipeline, an adversary can poison or backdoor the resulting model by uploading malicious data to the internet and waiting for a victim to scrape and train on it. Existing approaches for creating poisons and backdoors start with randomly sampled clean data, called base samples, and then modify those samples to craft poisons. However, some base samples may be significantly more amenable to poisoning than others. As a result, we may be able to craft more potent poisons by carefully choosing the base samples. In this work, we use guided diffusion to synthesize base samples from scratch that lead to significantly more potent poisons and backdoors than previous state-of-the-art attacks. Our Guided Diffusion Poisoning (GDP) base samples can be combined with any downstream poisoning or backdoor attack to boost its effectiveness. Our implementation code is publicly available at: https://github.com/hsouri/GDP .

摘要: 现代神经网络通常是在海量数据集上进行训练的，这些数据集是在最少的人类检查的情况下从网络上刮下来的。由于这种不安全的管理管道，对手可以通过将恶意数据上传到互联网并等待受害者对其进行擦除和训练来毒害或后门生成的模型。现有的制造毒药和后门的方法从随机抽样的清洁数据开始，称为基础样本，然后修改这些样本以制造毒药。然而，一些碱基样品可能比其他样品更容易中毒。因此，通过仔细选择基础样品，我们可能能够制造出更强的毒药。在这项工作中，我们使用引导扩散来从头开始合成基本样本，这些样本导致的毒药和后门比以前最先进的攻击要强得多。我们的引导式扩散中毒(GDP)基础样本可以与任何下游中毒或后门攻击相结合，以提高其有效性。我们的实现代码可在https://github.com/hsouri/GDP上公开获得。



## **45. Subspace Defense: Discarding Adversarial Perturbations by Learning a Subspace for Clean Signals**

子空间防御：通过学习干净信号的子空间来丢弃对抗扰动 cs.LG

Accepted by COLING 2024

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16176v1) [paper-pdf](http://arxiv.org/pdf/2403.16176v1)

**Authors**: Rui Zheng, Yuhao Zhou, Zhiheng Xi, Tao Gui, Qi Zhang, Xuanjing Huang

**Abstract**: Deep neural networks (DNNs) are notoriously vulnerable to adversarial attacks that place carefully crafted perturbations on normal examples to fool DNNs. To better understand such attacks, a characterization of the features carried by adversarial examples is needed. In this paper, we tackle this challenge by inspecting the subspaces of sample features through spectral analysis. We first empirically show that the features of either clean signals or adversarial perturbations are redundant and span in low-dimensional linear subspaces respectively with minimal overlap, and the classical low-dimensional subspace projection can suppress perturbation features out of the subspace of clean signals. This makes it possible for DNNs to learn a subspace where only features of clean signals exist while those of perturbations are discarded, which can facilitate the distinction of adversarial examples. To prevent the residual perturbations that is inevitable in subspace learning, we propose an independence criterion to disentangle clean signals from perturbations. Experimental results show that the proposed strategy enables the model to inherently suppress adversaries, which not only boosts model robustness but also motivates new directions of effective adversarial defense.

摘要: 众所周知，深度神经网络(DNN)容易受到敌意攻击，这些攻击会对正常示例进行精心设计的扰动，以愚弄DNN。为了更好地理解这类攻击，需要对对抗性例子所具有的特征进行描述。在本文中，我们通过谱分析检查样本特征的子空间来应对这一挑战。我们首先从经验上证明了CLEAN信号和对抗性扰动的特征在低维线性子空间中是冗余的且重叠最小，经典的低维子空间投影可以抑制CLEAN信号子空间之外的扰动特征。这使得DNN能够学习一个子空间，其中只有干净信号的特征存在，而扰动的特征被丢弃，这有助于区分对抗性例子。为了防止子空间学习中不可避免的残留扰动，我们提出了一个独立准则来分离干净的信号和扰动。实验结果表明，该策略使模型具有内在的抑制能力，不仅增强了模型的稳健性，而且为有效的对抗防御提供了新的方向。



## **46. Large Language Models for Blockchain Security: A Systematic Literature Review**

区块链安全的大型语言模型：系统文献综述 cs.CR

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.14280v2) [paper-pdf](http://arxiv.org/pdf/2403.14280v2)

**Authors**: Zheyuan He, Zihao Li, Sen Yang

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools in various domains involving blockchain security (BS). Several recent studies are exploring LLMs applied to BS. However, there remains a gap in our understanding regarding the full scope of applications, impacts, and potential constraints of LLMs on blockchain security. To fill this gap, we conduct a literature review on LLM4BS.   As the first review of LLM's application on blockchain security, our study aims to comprehensively analyze existing research and elucidate how LLMs contribute to enhancing the security of blockchain systems. Through a thorough examination of scholarly works, we delve into the integration of LLMs into various aspects of blockchain security. We explore the mechanisms through which LLMs can bolster blockchain security, including their applications in smart contract auditing, identity verification, anomaly detection, vulnerable repair, and so on. Furthermore, we critically assess the challenges and limitations associated with leveraging LLMs for blockchain security, considering factors such as scalability, privacy concerns, and adversarial attacks. Our review sheds light on the opportunities and potential risks inherent in this convergence, providing valuable insights for researchers, practitioners, and policymakers alike.

摘要: 大型语言模型(LLM)在涉及区块链安全(BS)的各个领域中已成为强大的工具。最近的几项研究正在探索将LLMS应用于BS。然而，对于低成本管理的全部应用范围、影响以及对区块链安全的潜在限制，我们的理解仍然存在差距。为了填补这一空白，我们对LLM4BS进行了文献综述。作为LLM在区块链安全方面应用的首次综述，本研究旨在全面分析现有研究，阐明LLM如何为增强区块链系统的安全性做出贡献。通过对学术著作的深入研究，我们深入研究了LLMS在区块链安全的各个方面的整合。我们探讨了LLMS增强区块链安全的机制，包括它们在智能合同审计、身份验证、异常检测、漏洞修复等方面的应用。此外，考虑到可扩展性、隐私问题和敌意攻击等因素，我们严格评估了利用LLM实现区块链安全所面临的挑战和限制。我们的审查揭示了这种融合所固有的机遇和潜在风险，为研究人员、从业者和政策制定者提供了有价值的见解。



## **47. ALI-DPFL: Differentially Private Federated Learning with Adaptive Local Iterations**

ALI—DPFL：具有自适应局部迭代的差分私有联邦学习 cs.LG

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2308.10457v5) [paper-pdf](http://arxiv.org/pdf/2308.10457v5)

**Authors**: Xinpeng Ling, Jie Fu, Kuncan Wang, Haitao Liu, Zhili Chen

**Abstract**: Federated Learning (FL) is a distributed machine learning technique that allows model training among multiple devices or organizations by sharing training parameters instead of raw data. However, adversaries can still infer individual information through inference attacks (e.g. differential attacks) on these training parameters. As a result, Differential Privacy (DP) has been widely used in FL to prevent such attacks.   We consider differentially private federated learning in a resource-constrained scenario, where both privacy budget and communication rounds are constrained. By theoretically analyzing the convergence, we can find the optimal number of local DPSGD iterations for clients between any two sequential global updates. Based on this, we design an algorithm of Differentially Private Federated Learning with Adaptive Local Iterations (ALI-DPFL). We experiment our algorithm on the MNIST, FashionMNIST and Cifar10 datasets, and demonstrate significantly better performances than previous work in the resource-constraint scenario. Code is available at https://github.com/KnightWan/ALI-DPFL.

摘要: 联合学习(FL)是一种分布式机器学习技术，通过共享训练参数而不是原始数据，允许在多个设备或组织之间进行模型训练。然而，攻击者仍然可以通过对这些训练参数的推理攻击(例如差异攻击)来推断个人信息。因此，差分隐私(DP)被广泛应用于FL中以防止此类攻击。我们考虑在资源受限的情况下进行不同的私有联合学习，其中隐私预算和通信回合都受到限制。通过对收敛的理论分析，我们可以找到任意两个连续全局更新之间客户端的最优局部DPSGD迭代次数。在此基础上，设计了一种基于自适应局部迭代的差分私有联邦学习算法(ALI-DPFL)。我们在MNIST、FashionMNIST和Cifar10数据集上测试了我们的算法，并在资源受限的情况下展示了比以前的工作更好的性能。代码可在https://github.com/KnightWan/ALI-DPFL.上找到



## **48. Robust Diffusion Models for Adversarial Purification**

对抗净化的鲁棒扩散模型 cs.CV

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2403.16067v1) [paper-pdf](http://arxiv.org/pdf/2403.16067v1)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.

摘要: 基于扩散模型(DM)的对抗净化(AP)已被证明是对抗训练(AT)最有效的替代方法。然而，这些方法忽略了这样一个事实，即预先训练的扩散模型本身对对手攻击也不是很健壮。此外，扩散过程容易破坏语义信息，生成高质量的图像，但反向处理后的图像与原始输入图像完全不同，导致标准精度下降。为了克服这些问题，一个自然的想法是利用对抗性训练策略来重新训练或微调预先训练的扩散模型，这在计算上是令人望而却步的。我们提出了一种新的具有对抗性指导的稳健逆向过程，它独立于给定的预先训练的DM，并且避免了对DM的重新训练或微调。这种健壮的指导不仅可以确保生成保持更多语义内容的纯化实例，还可以第一次缓解DM的准确性和健壮性之间的权衡，这也为基于DM的AP提供了对新攻击的有效适应能力。大量的实验表明，我们的方法达到了最先进的结果，并对不同的攻击表现出了泛化能力。



## **49. To Generate or Not? Safety-Driven Unlearned Diffusion Models Are Still Easy To Generate Unsafe Images ... For Now**

生成还是不生成？安全驱动的未学习扩散模型仍然容易生成不安全的图像现在 cs.CV

Codes are available at  https://github.com/OPTML-Group/Diffusion-MU-Attack

**SubmitDate**: 2024-03-24    [abs](http://arxiv.org/abs/2310.11868v2) [paper-pdf](http://arxiv.org/pdf/2310.11868v2)

**Authors**: Yimeng Zhang, Jinghan Jia, Xin Chen, Aochuan Chen, Yihua Zhang, Jiancheng Liu, Ke Ding, Sijia Liu

**Abstract**: The recent advances in diffusion models (DMs) have revolutionized the generation of realistic and complex images. However, these models also introduce potential safety hazards, such as producing harmful content and infringing data copyrights. Despite the development of safety-driven unlearning techniques to counteract these challenges, doubts about their efficacy persist. To tackle this issue, we introduce an evaluation framework that leverages adversarial prompts to discern the trustworthiness of these safety-driven DMs after they have undergone the process of unlearning harmful concepts. Specifically, we investigated the adversarial robustness of DMs, assessed by adversarial prompts, when eliminating unwanted concepts, styles, and objects. We develop an effective and efficient adversarial prompt generation approach for DMs, termed UnlearnDiffAtk. This method capitalizes on the intrinsic classification abilities of DMs to simplify the creation of adversarial prompts, thereby eliminating the need for auxiliary classification or diffusion models.Through extensive benchmarking, we evaluate the robustness of five widely-used safety-driven unlearned DMs (i.e., DMs after unlearning undesirable concepts, styles, or objects) across a variety of tasks. Our results demonstrate the effectiveness and efficiency merits of UnlearnDiffAtk over the state-of-the-art adversarial prompt generation method and reveal the lack of robustness of current safety-driven unlearning techniques when applied to DMs. Codes are available at https://github.com/OPTML-Group/Diffusion-MU-Attack. WARNING: This paper contains model outputs that may be offensive in nature.

摘要: 扩散模型的最新进展使逼真和复杂图像的生成发生了革命性的变化。然而，这些模式也带来了潜在的安全隐患，如产生有害内容和侵犯数据著作权。尽管发展了安全驱动的遗忘技术来应对这些挑战，但对其有效性的怀疑依然存在。为了解决这个问题，我们引入了一个评估框架，利用对抗性提示，在这些以安全为导向的DM经历了忘记有害概念的过程后，识别他们的可信度。具体地说，我们研究了DM在消除不需要的概念、风格和对象时，通过对抗性提示评估的对抗性健壮性。本文提出了一种高效的敌意提示生成方法，称为UnlearnDiffAtk。该方法利用DM固有的分类能力来简化敌意提示的生成，从而消除了对辅助分类或扩散模型的需要。通过广泛的基准测试，我们评估了五种广泛使用的安全驱动的未学习DM(即忘记不良概念、风格或对象后的DM)在不同任务中的健壮性。实验结果证明了UnlearnDiffAtk算法相对于最新的对抗性提示生成方法的有效性和高效性，并揭示了当前安全驱动的遗忘技术在应用于决策支持系统时的健壮性不足。有关代码，请访问https://github.com/OPTML-Group/Diffusion-MU-Attack.警告：本文包含可能具有攻击性的模型输出。



## **50. An Embarrassingly Simple Defense Against Backdoor Attacks On SSL**

一个令人尴尬的简单的后门攻击防御SSL cs.CV

10 pages, 5 figures

**SubmitDate**: 2024-03-23    [abs](http://arxiv.org/abs/2403.15918v1) [paper-pdf](http://arxiv.org/pdf/2403.15918v1)

**Authors**: Aryan Satpathy, Nilaksh, Dhruva Rajwade

**Abstract**: Self Supervised Learning (SSL) has emerged as a powerful paradigm to tackle data landscapes with absence of human supervision. The ability to learn meaningful tasks without the use of labeled data makes SSL a popular method to manage large chunks of data in the absence of labels. However, recent work indicates SSL to be vulnerable to backdoor attacks, wherein models can be controlled, possibly maliciously, to suit an adversary's motives. Li et.al (2022) introduce a novel frequency-based backdoor attack: CTRL. They show that CTRL can be used to efficiently and stealthily gain control over a victim's model trained using SSL. In this work, we devise two defense strategies against frequency-based attacks in SSL: One applicable before model training and the second to be applied during model inference. Our first contribution utilizes the invariance property of the downstream task to defend against backdoor attacks in a generalizable fashion. We observe the ASR (Attack Success Rate) to reduce by over 60% across experiments. Our Inference-time defense relies on evasiveness of the attack and uses the luminance channel to defend against attacks. Using object classification as the downstream task for SSL, we demonstrate successful defense strategies that do not require re-training of the model. Code is available at https://github.com/Aryan-Satpathy/Backdoor.

摘要: 自我监督学习(SSL)已经成为一种强大的范式，可以在缺乏人类监督的情况下处理数据环境。无需使用标签数据即可学习有意义的任务的能力使SSL成为在没有标签的情况下管理大量数据的流行方法。然而，最近的研究表明，SSL容易受到后门攻击，在后门攻击中，可以控制模型，可能是恶意的，以适应对手的动机。Li et.al(2022)引入了一种新的基于频率的后门攻击：Ctrl。他们表明，CTRL可以用来有效地、秘密地控制使用SSL训练的受害者模型。在这项工作中，我们针对基于频率的攻击设计了两种防御策略：一种适用于模型训练之前，另一种应用于模型推理中。我们的第一个贡献是利用下游任务的不变性以一种可推广的方式防御后门攻击。我们观察到，在整个实验中，ASR(攻击成功率)降低了60%以上。我们的推理时间防御依赖于攻击的规避，并使用亮度通道来防御攻击。使用对象分类作为SSL的下游任务，我们演示了成功的防御策略，不需要对模型进行重新训练。代码可在https://github.com/Aryan-Satpathy/Backdoor.上找到



