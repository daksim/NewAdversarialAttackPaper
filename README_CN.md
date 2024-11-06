# Latest Adversarial Attack Papers
**update at 2024-11-06 17:29:05**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Introducing Perturb-ability Score (PS) to Enhance Robustness Against Evasion Adversarial Attacks on ML-NIDS**

引入扰动能力评分（PS）增强ML-NIDS对抗规避攻击的鲁棒性 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2409.07448v2) [paper-pdf](http://arxiv.org/pdf/2409.07448v2)

**Authors**: Mohamed elShehaby, Ashraf Matrawy

**Abstract**: As network security threats continue to evolve, safeguarding Machine Learning (ML)-based Network Intrusion Detection Systems (NIDS) from adversarial attacks is crucial. This paper introduces the notion of feature perturb-ability and presents a novel Perturb-ability Score (PS) metric that identifies NIDS features susceptible to manipulation in the problem-space by an attacker. By quantifying a feature's susceptibility to perturbations within the problem-space, the PS facilitates the selection of features that are inherently more robust against evasion adversarial attacks on ML-NIDS during the feature selection phase. These features exhibit natural resilience to perturbations, as they are heavily constrained by the problem-space limitations and correlations of the NIDS domain. Furthermore, manipulating these features may either disrupt the malicious function of evasion adversarial attacks on NIDS or render the network traffic invalid for processing (or both). This proposed novel approach employs a fresh angle by leveraging network domain constraints as a defense mechanism against problem-space evasion adversarial attacks targeting ML-NIDS. We demonstrate the effectiveness of our PS-guided feature selection defense in enhancing NIDS robustness. Experimental results across various ML-based NIDS models and public datasets show that selecting only robust features (low-PS features) can maintain solid detection performance while significantly reducing vulnerability to evasion adversarial attacks. Additionally, our findings verify that the PS effectively identifies NIDS features highly vulnerable to problem-space perturbations.

摘要: 随着网络安全威胁的不断演变，保护基于机器学习(ML)的网络入侵检测系统(NID)免受对手攻击至关重要。引入了特征扰动能力的概念，提出了一种新的扰动能力得分(PS)度量方法，用于识别网络入侵检测系统在问题空间中易受攻击者操纵的特征。通过量化特征对问题空间内扰动的敏感性，PS有助于选择在特征选择阶段对ML-NID的逃避攻击具有更强稳健性的特征。这些功能表现出对扰动的自然弹性，因为它们受到NIDS域的问题空间限制和相关性的严重限制。此外，操纵这些功能可能会破坏躲避对NID的恶意攻击的恶意功能，或者使网络流量无法处理(或两者兼而有之)。该方法从一个全新的角度，利用网络域约束作为针对ML-NID的问题空间规避攻击的防御机制。我们展示了我们的PS引导的特征选择防御在增强网络入侵检测系统健壮性方面的有效性。在各种基于ML的网络入侵检测模型和公共数据集上的实验结果表明，只选择健壮的特征(低PS特征)可以保持可靠的检测性能，同时显著降低对躲避对手攻击的脆弱性。此外，我们的发现验证了PS有效地识别了高度易受问题空间扰动影响的NIDS特征。



## **2. Oblivious Defense in ML Models: Backdoor Removal without Detection**

ML模型中的无意防御：在不检测的情况下删除后门 cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03279v1) [paper-pdf](http://arxiv.org/pdf/2411.03279v1)

**Authors**: Shafi Goldwasser, Jonathan Shafer, Neekon Vafa, Vinod Vaikuntanathan

**Abstract**: As society grows more reliant on machine learning, ensuring the security of machine learning systems against sophisticated attacks becomes a pressing concern. A recent result of Goldwasser, Kim, Vaikuntanathan, and Zamir (2022) shows that an adversary can plant undetectable backdoors in machine learning models, allowing the adversary to covertly control the model's behavior. Backdoors can be planted in such a way that the backdoored machine learning model is computationally indistinguishable from an honest model without backdoors.   In this paper, we present strategies for defending against backdoors in ML models, even if they are undetectable. The key observation is that it is sometimes possible to provably mitigate or even remove backdoors without needing to detect them, using techniques inspired by the notion of random self-reducibility. This depends on properties of the ground-truth labels (chosen by nature), and not of the proposed ML model (which may be chosen by an attacker).   We give formal definitions for secure backdoor mitigation, and proceed to show two types of results. First, we show a "global mitigation" technique, which removes all backdoors from a machine learning model under the assumption that the ground-truth labels are close to a Fourier-heavy function. Second, we consider distributions where the ground-truth labels are close to a linear or polynomial function in $\mathbb{R}^n$. Here, we show "local mitigation" techniques, which remove backdoors with high probability for every inputs of interest, and are computationally cheaper than global mitigation. All of our constructions are black-box, so our techniques work without needing access to the model's representation (i.e., its code or parameters). Along the way we prove a simple result for robust mean estimation.

摘要: 随着社会变得越来越依赖机器学习，确保机器学习系统免受复杂攻击的安全成为一个紧迫的问题。Goldwasser，Kim，Vaikuntanathan和Zamir(2022)最近的一个结果表明，对手可以在机器学习模型中植入不可检测的后门，允许对手秘密控制模型的行为。后门可以被植入这样一种方式，即被后门的机器学习模型在计算上与没有后门的诚实模型没有区别。在本文中，我们提出了在ML模型中防御后门的策略，即使它们是不可检测的。关键的观察结果是，使用受随机自还原概念启发的技术，有时可以在不需要检测到后门的情况下，以可证明的方式减轻甚至删除后门。这取决于基本事实标签的属性(自然选择)，而不是建议的ML模型(可能由攻击者选择)的属性。我们给出了安全后门缓解的正式定义，并继续展示了两种类型的结果。首先，我们展示了一种“全局缓解”技术，该技术在假设基本事实标签接近傅立叶重函数的情况下，从机器学习模型中移除所有后门。其次，我们考虑基本事实标号在$\mathbb{R}^n$中接近于线性或多项式函数的分布。在这里，我们展示了“局部缓解”技术，它为每个感兴趣的输入高概率地移除后门，并且在计算上比全局缓解更便宜。我们的所有构造都是黑盒结构，因此我们的技术无需访问模型的表示形式(即其代码或参数)即可工作。在此过程中，我们证明了稳健均值估计的一个简单结果。



## **3. Formal Logic-guided Robust Federated Learning against Poisoning Attacks**

针对中毒攻击的形式逻辑引导鲁棒联邦学习 cs.CR

arXiv admin note: text overlap with arXiv:2305.00328 by other authors

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03231v1) [paper-pdf](http://arxiv.org/pdf/2411.03231v1)

**Authors**: Dung Thuy Nguyen, Ziyan An, Taylor T. Johnson, Meiyi Ma, Kevin Leach

**Abstract**: Federated Learning (FL) offers a promising solution to the privacy concerns associated with centralized Machine Learning (ML) by enabling decentralized, collaborative learning. However, FL is vulnerable to various security threats, including poisoning attacks, where adversarial clients manipulate the training data or model updates to degrade overall model performance. Recognizing this threat, researchers have focused on developing defense mechanisms to counteract poisoning attacks in FL systems. However, existing robust FL methods predominantly focus on computer vision tasks, leaving a gap in addressing the unique challenges of FL with time series data. In this paper, we present FLORAL, a defense mechanism designed to mitigate poisoning attacks in federated learning for time-series tasks, even in scenarios with heterogeneous client data and a large number of adversarial participants. Unlike traditional model-centric defenses, FLORAL leverages logical reasoning to evaluate client trustworthiness by aligning their predictions with global time-series patterns, rather than relying solely on the similarity of client updates. Our approach extracts logical reasoning properties from clients, then hierarchically infers global properties, and uses these to verify client updates. Through formal logic verification, we assess the robustness of each client contribution, identifying deviations indicative of adversarial behavior. Experimental results on two datasets demonstrate the superior performance of our approach compared to existing baseline methods, highlighting its potential to enhance the robustness of FL to time series applications. Notably, FLORAL reduced the prediction error by 93.27\% in the best-case scenario compared to the second-best baseline. Our code is available at \url{https://anonymous.4open.science/r/FLORAL-Robust-FTS}.

摘要: 联合学习(FL)通过支持分散的、协作的学习，为集中式机器学习(ML)相关的隐私问题提供了一个有前途的解决方案。然而，FL容易受到各种安全威胁，包括中毒攻击，即敌对客户端操纵训练数据或模型更新以降低整体模型性能。认识到这种威胁，研究人员专注于开发防御机制来对抗FL系统中的中毒攻击。然而，现有的稳健的外语学习方法主要集中在计算机视觉任务上，在用时间序列数据解决外语的独特挑战方面留下了空白。在本文中，我们提出了一种防御机制FLOLAR，该机制被设计用于在时间序列任务的联合学习中缓解中毒攻击，即使在具有异质客户端数据和大量对抗性参与者的场景中也是如此。与传统的以模型为中心的防御不同，FLORAL利用逻辑推理来评估客户的可信度，方法是将他们的预测与全球时间序列模式保持一致，而不是仅仅依赖客户更新的相似性。我们的方法从客户端提取逻辑推理属性，然后分层推断全局属性，并使用这些属性来验证客户端更新。通过形式逻辑验证，我们评估每个客户贡献的健壮性，识别指示对抗性行为的偏差。在两个数据集上的实验结果表明，与现有的基线方法相比，我们的方法具有更好的性能，突出了它在增强FL对时间序列应用的稳健性方面的潜力。值得注意的是，与次佳基线相比，FLORAL在最好的情况下将预测误差降低了93.27\%。我们的代码可以在\url{https://anonymous.4open.science/r/FLORAL-Robust-FTS}.上找到



## **4. Gradient-Guided Conditional Diffusion Models for Private Image Reconstruction: Analyzing Adversarial Impacts of Differential Privacy and Denoising**

用于私人图像重建的对象引导条件扩散模型：分析差异隐私和去噪的对抗影响 cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03053v1) [paper-pdf](http://arxiv.org/pdf/2411.03053v1)

**Authors**: Tao Huang, Jiayang Meng, Hong Chen, Guolong Zheng, Xu Yang, Xun Yi, Hua Wang

**Abstract**: We investigate the construction of gradient-guided conditional diffusion models for reconstructing private images, focusing on the adversarial interplay between differential privacy noise and the denoising capabilities of diffusion models. While current gradient-based reconstruction methods struggle with high-resolution images due to computational complexity and prior knowledge requirements, we propose two novel methods that require minimal modifications to the diffusion model's generation process and eliminate the need for prior knowledge. Our approach leverages the strong image generation capabilities of diffusion models to reconstruct private images starting from randomly generated noise, even when a small amount of differentially private noise has been added to the gradients. We also conduct a comprehensive theoretical analysis of the impact of differential privacy noise on the quality of reconstructed images, revealing the relationship among noise magnitude, the architecture of attacked models, and the attacker's reconstruction capability. Additionally, extensive experiments validate the effectiveness of our proposed methods and the accuracy of our theoretical findings, suggesting new directions for privacy risk auditing using conditional diffusion models.

摘要: 我们研究了用于重建私人图像的梯度引导的条件扩散模型的构造，重点研究了差分隐私噪声与扩散模型的去噪能力之间的对抗性相互作用。由于计算复杂性和先验知识的要求，现有的基于梯度的重建方法难以处理高分辨率的图像，我们提出了两种新的方法，它们只需要对扩散模型的生成过程进行最小程度的修改，并且不需要先验知识。我们的方法利用扩散模型强大的图像生成能力，从随机生成的噪声开始重建私人图像，即使在梯度中添加了少量的差分私人噪声。我们还对差分隐私噪声对重建图像质量的影响进行了全面的理论分析，揭示了噪声大小、攻击模型的体系结构和攻击者的重建能力之间的关系。此外，大量的实验验证了我们提出的方法的有效性和我们的理论发现的准确性，为使用条件扩散模型进行隐私风险审计提供了新的方向。



## **5. Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**

对抗性马尔科夫游戏：基于决策的自适应攻击和防御 cs.AI

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2312.13435v2) [paper-pdf](http://arxiv.org/pdf/2312.13435v2)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

摘要: 尽管做出了相当大的努力来使它们健壮，但现实世界中基于ML的系统仍然容易受到基于决策的攻击，因为到目前为止，对其操作健壮性的确凿证据被证明是难以处理的。健壮性评估的规范方法要求自适应攻击，即完全了解防御并量身定做以绕过它。在这项研究中，我们引入了一个更广泛的适应性概念，并展示了攻击和防御如何从它和通过互动相互学习中受益。我们提出并评估了一个框架，用于通过形成的竞争博弈自适应地优化黑盒攻击和防御。要可靠地衡量健壮性，重要的是要针对现实和最坏情况下的攻击进行评估。因此，我们通过自适应控制来增强攻击和可供其使用的躲避武器，并观察到同样可以对防御做同样的事情，然后我们首先分开评估它们，然后在多智能体的角度下进行联合评估。我们演示了控制系统如何响应的主动防御是面对基于决策的攻击时模型强化的必要补充；然后说明如何通过自适应攻击来规避这些防御，最终只会引发主动和自适应防御。我们通过广泛的理论和经验调查验证了我们的观察结果，以确认启用AI的对手对基于黑盒ML的系统构成了相当大的威胁，重新点燃了众所周知的军备竞赛，其中防御也必须启用AI。简而言之，我们解决了适应性对手带来的挑战，并开发了适应性防御，从而制定了有效的策略，以确保部署在现实世界中的基于ML的系统的健壮性。



## **6. Flashy Backdoor: Real-world Environment Backdoor Attack on SNNs with DVS Cameras**

浮华的后门：现实环境对使用DVS摄像机的SNN进行后门攻击 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.03022v1) [paper-pdf](http://arxiv.org/pdf/2411.03022v1)

**Authors**: Roberto Riaño, Gorka Abad, Stjepan Picek, Aitor Urbieta

**Abstract**: While security vulnerabilities in traditional Deep Neural Networks (DNNs) have been extensively studied, the susceptibility of Spiking Neural Networks (SNNs) to adversarial attacks remains mostly underexplored. Until now, the mechanisms to inject backdoors into SNN models have been limited to digital scenarios; thus, we present the first evaluation of backdoor attacks in real-world environments.   We begin by assessing the applicability of existing digital backdoor attacks and identifying their limitations for deployment in physical environments. To address each of the found limitations, we present three novel backdoor attack methods on SNNs, i.e., Framed, Strobing, and Flashy Backdoor. We also assess the effectiveness of traditional backdoor procedures and defenses adapted for SNNs, such as pruning, fine-tuning, and fine-pruning. The results show that while these procedures and defenses can mitigate some attacks, they often fail against stronger methods like Flashy Backdoor or sacrifice too much clean accuracy, rendering the models unusable.   Overall, all our methods can achieve up to a 100% Attack Success Rate while maintaining high clean accuracy in every tested dataset. Additionally, we evaluate the stealthiness of the triggers with commonly used metrics, finding them highly stealthy. Thus, we propose new alternatives more suited for identifying poisoned samples in these scenarios. Our results show that further research is needed to ensure the security of SNN-based systems against backdoor attacks and their safe application in real-world scenarios. The code, experiments, and results are available in our repository.

摘要: 虽然传统深度神经网络(DNN)的安全漏洞已经得到了广泛的研究，但尖峰神经网络(SNN)对敌意攻击的敏感性仍未得到充分的研究。到目前为止，将后门注入SNN模型的机制仅限于数字场景；因此，我们提出了对现实环境中的后门攻击的第一次评估。我们首先评估现有数字后门攻击的适用性，并确定它们在物理环境中部署的局限性。针对这些缺陷，我们提出了三种新的针对SNN的后门攻击方法，即框架式、选通式和闪光式后门。我们还评估了适用于SNN的传统后门程序和防御措施的有效性，例如修剪、微调和精细修剪。结果表明，虽然这些程序和防御措施可以缓解一些攻击，但它们往往无法通过华而不实的后门等更强大的方法，或者牺牲太多干净的准确性，导致模型无法使用。总体而言，我们的所有方法都可以实现高达100%的攻击成功率，同时在每个测试数据集中保持高清洁准确率。此外，我们使用常用的指标评估触发器的隐蔽性，发现它们具有高度的隐蔽性。因此，我们提出了更适合于在这些场景中识别有毒样本的新的替代方案。我们的结果表明，需要进一步的研究来确保基于SNN的系统免受后门攻击的安全性，以及它们在现实场景中的安全应用。代码、实验和结果都可以在我们的存储库中找到。



## **7. Region-Guided Attack on the Segment Anything Model (SAM)**

对分段任意模型（Sam）的区域引导攻击 cs.CV

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02974v1) [paper-pdf](http://arxiv.org/pdf/2411.02974v1)

**Authors**: Xiaoliang Liu, Furao Shen, Jian Zhao

**Abstract**: The Segment Anything Model (SAM) is a cornerstone of image segmentation, demonstrating exceptional performance across various applications, particularly in autonomous driving and medical imaging, where precise segmentation is crucial. However, SAM is vulnerable to adversarial attacks that can significantly impair its functionality through minor input perturbations. Traditional techniques, such as FGSM and PGD, are often ineffective in segmentation tasks due to their reliance on global perturbations that overlook spatial nuances. Recent methods like Attack-SAM-K and UAD have begun to address these challenges, but they frequently depend on external cues and do not fully leverage the structural interdependencies within segmentation processes. This limitation underscores the need for a novel adversarial strategy that exploits the unique characteristics of segmentation tasks. In response, we introduce the Region-Guided Attack (RGA), designed specifically for SAM. RGA utilizes a Region-Guided Map (RGM) to manipulate segmented regions, enabling targeted perturbations that fragment large segments and expand smaller ones, resulting in erroneous outputs from SAM. Our experiments demonstrate that RGA achieves high success rates in both white-box and black-box scenarios, emphasizing the need for robust defenses against such sophisticated attacks. RGA not only reveals SAM's vulnerabilities but also lays the groundwork for developing more resilient defenses against adversarial threats in image segmentation.

摘要: Segment Anything Model(SAM)是图像分割的基石，在各种应用中表现出卓越的性能，特别是在自动驾驶和医学成像中，准确的分割至关重要。然而，SAM很容易受到对抗性攻击，这些攻击可能会通过微小的输入扰动显著损害其功能。传统的分割技术，如FGSM和PGD，在分割任务中往往是无效的，因为它们依赖于忽略空间细微差别的全局扰动。最近的方法，如攻击-SAM-K和UAD已经开始解决这些挑战，但它们经常依赖外部线索，并且没有充分利用分割过程中的结构相互依赖。这一限制强调了需要一种利用分段任务的独特特征的新的对抗性策略。作为回应，我们引入了专门为SAM设计的区域制导攻击(RGA)。RGA利用区域引导地图(RGM)来处理分割的区域，从而实现了将大片段分割并扩展小片段的有针对性的扰动，从而导致SAM的错误输出。我们的实验表明，RGA在白盒和黑盒场景中都取得了很高的成功率，强调了对这种复杂攻击的稳健防御的必要性。RGA不仅揭示了SAM的漏洞，而且为开发更具弹性的防御图像分割中的对抗性威胁奠定了基础。



## **8. Bias in the Mirror: Are LLMs opinions robust to their own adversarial attacks ?**

镜子中的偏见：LLM的观点是否能抵御自己的对抗攻击？ cs.CL

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2410.13517v2) [paper-pdf](http://arxiv.org/pdf/2410.13517v2)

**Authors**: Virgile Rennard, Christos Xypolopoulos, Michalis Vazirgiannis

**Abstract**: Large language models (LLMs) inherit biases from their training data and alignment processes, influencing their responses in subtle ways. While many studies have examined these biases, little work has explored their robustness during interactions. In this paper, we introduce a novel approach where two instances of an LLM engage in self-debate, arguing opposing viewpoints to persuade a neutral version of the model. Through this, we evaluate how firmly biases hold and whether models are susceptible to reinforcing misinformation or shifting to harmful viewpoints. Our experiments span multiple LLMs of varying sizes, origins, and languages, providing deeper insights into bias persistence and flexibility across linguistic and cultural contexts.

摘要: 大型语言模型（LLM）从其训练数据和对齐过程中继承了偏差，以微妙的方式影响其响应。虽然许多研究已经检查了这些偏差，但很少有工作探索它们在互动过程中的稳健性。在本文中，我们引入了一种新颖的方法，其中两个LLM实例进行自我辩论，争论相反的观点以说服模型的中立版本。通过此，我们评估偏见的存在程度，以及模型是否容易强化错误信息或转向有害观点。我们的实验跨越了不同规模、起源和语言的多个LLM，为跨语言和文化背景的偏见持续性和灵活性提供了更深入的见解。



## **9. Towards Efficient Transferable Preemptive Adversarial Defense**

迈向高效的可转让先发制人的对抗防御 cs.CR

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2407.15524v3) [paper-pdf](http://arxiv.org/pdf/2407.15524v3)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning technology has brought convenience and advanced developments but has become untrustworthy because of its sensitivity to inconspicuous perturbations (i.e., adversarial attacks). Attackers may utilize this sensitivity to manipulate predictions. To defend against such attacks, we have devised a proactive strategy for "attacking" the medias before it is attacked by the third party, so that when the protected medias are further attacked, the adversarial perturbations are automatically neutralized. This strategy, dubbed Fast Preemption, provides an efficient transferable preemptive defense by using different models for labeling inputs and learning crucial features. A forward-backward cascade learning algorithm is used to compute protective perturbations, starting with forward propagation optimization to achieve rapid convergence, followed by iterative backward propagation learning to alleviate overfitting. This strategy offers state-of-the-art transferability and protection across various systems. With the running of only three steps, our Fast Preemption framework outperforms benchmark training-time, test-time, and preemptive adversarial defenses. We have also devised the first to our knowledge effective white-box adaptive reversion attack and demonstrate that the protection added by our defense strategy is irreversible unless the backbone model, algorithm, and settings are fully compromised. This work provides a new direction to developing proactive defenses against adversarial attacks. The proposed methodology will be made available on GitHub.

摘要: 深度学习技术带来了便利和先进的发展，但由于其对不起眼的扰动(即对抗性攻击)的敏感性而变得不可信任。攻击者可能会利用这种敏感性来操纵预测。为了防御此类攻击，我们制定了一种主动策略，在媒体受到第三方攻击之前对其进行“攻击”，以便当受保护的媒体进一步受到攻击时，对手的干扰会自动被中和。这一战略被称为快速抢占，通过使用不同的模型来标记输入和学习关键特征，提供了一种高效的可转移的抢占防御。保护摄动的计算采用前向-后向级联学习算法，从前向传播优化开始实现快速收敛，然后迭代后向传播学习以减少过拟合。这一战略提供了最先进的跨各种系统的可转移性和保护。由于只运行了三个步骤，我们的快速抢占框架的性能优于基准训练时间、测试时间和先发制人的对手防御。我们还设计了我们所知的第一个有效的白盒自适应恢复攻击，并证明了我们的防御策略添加的保护是不可逆转的，除非主干模型、算法和设置完全受损。这项工作为主动防御对抗性攻击提供了新的方向。拟议的方法将在GitHub上提供。



## **10. Query-Efficient Adversarial Attack Against Vertical Federated Graph Learning**

针对垂直联邦图学习的查询高效对抗攻击 cs.LG

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02809v1) [paper-pdf](http://arxiv.org/pdf/2411.02809v1)

**Authors**: Jinyin Chen, Wenbo Mu, Luxin Zhang, Guohan Huang, Haibin Zheng, Yao Cheng

**Abstract**: Graph neural network (GNN) has captured wide attention due to its capability of graph representation learning for graph-structured data. However, the distributed data silos limit the performance of GNN. Vertical federated learning (VFL), an emerging technique to process distributed data, successfully makes GNN possible to handle the distributed graph-structured data. Despite the prosperous development of vertical federated graph learning (VFGL), the robustness of VFGL against the adversarial attack has not been explored yet. Although numerous adversarial attacks against centralized GNNs are proposed, their attack performance is challenged in the VFGL scenario. To the best of our knowledge, this is the first work to explore the adversarial attack against VFGL. A query-efficient hybrid adversarial attack framework is proposed to significantly improve the centralized adversarial attacks against VFGL, denoted as NA2, short for Neuron-based Adversarial Attack. Specifically, a malicious client manipulates its local training data to improve its contribution in a stealthy fashion. Then a shadow model is established based on the manipulated data to simulate the behavior of the server model in VFGL. As a result, the shadow model can improve the attack success rate of various centralized attacks with a few queries. Extensive experiments on five real-world benchmarks demonstrate that NA2 improves the performance of the centralized adversarial attacks against VFGL, achieving state-of-the-art performance even under potential adaptive defense where the defender knows the attack method. Additionally, we provide interpretable experiments of the effectiveness of NA2 via sensitive neurons identification and visualization of t-SNE.

摘要: 图神经网络(GNN)因其对图结构数据的图表示学习能力而受到广泛关注。然而，分布式数据孤岛限制了GNN的性能。垂直联合学习(VFL)是一种新兴的分布式数据处理技术，它成功地使GNN处理分布式图结构数据成为可能。尽管垂直联邦图学习(VFGL)得到了蓬勃的发展，但VFGL对敌意攻击的健壮性还没有得到研究。虽然许多针对集中式GNN的对抗性攻击被提出，但它们的攻击性能在VFGL场景中受到挑战。据我们所知，这是第一个探索VFGL对抗性攻击的工作。针对VFGL的集中式对抗攻击，提出了一种查询高效的混合对抗攻击框架，简称NA2，即基于神经元的对抗攻击。具体地说，恶意客户端操纵其本地训练数据，以秘密方式提高其贡献。然后根据处理后的数据建立阴影模型，在VFGL中模拟服务器模型的行为。因此，影子模型可以提高查询次数较少的各种集中式攻击的攻击成功率。在五个真实基准上的广泛实验表明，NA2提高了对VFGL的集中式对抗性攻击的性能，即使在防御者知道攻击方法的潜在自适应防御下也获得了最先进的性能。此外，我们还通过敏感神经元的识别和t-SNE的可视化提供了Na2有效性的可解释性实验。



## **11. TRANSPOSE: Transitional Approaches for Spatially-Aware LFI Resilient FSM Encoding**

TRANSPOSE：空间感知LFI弹性RSM编码的过渡方法 cs.CR

14 pages, 11 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02798v1) [paper-pdf](http://arxiv.org/pdf/2411.02798v1)

**Authors**: Muhtadi Choudhury, Minyan Gao, Avinash Varna, Elad Peer, Domenic Forte

**Abstract**: Finite state machines (FSMs) regulate sequential circuits, including access to sensitive information and privileged CPU states. Courtesy of contemporary research on laser attacks, laser-based fault injection (LFI) is becoming even more precise where an adversary can thwart chip security by altering individual flip-flop (FF) values. Different laser models, e.g., bit flip, bit set, and bit reset, have been developed to appreciate LFI on practical targets. As traditional approaches may incorporate substantial overhead, state-based SPARSE and transition-based TAMED countermeasures were proposed in our prior work to improve FSM resiliency efficiently. TAMED overcame SPARSE's limitation of being too conservative, and generating multiple LFI resilient encodings for contemporary LFI models on demand. SPARSE, however, incorporated design layout information into its vulnerability estimation which makes its vulnerability estimation metric more accurate. In this paper, we extend TAMED by proposing a transition-based encoding CAD framework (TRANSPOSE), that incorporates spatial transitional vulnerability metrics to quantify design susceptibility of FSMs based on both the bit flip model and the set-reset models. TRANSPOSE also incorporates floorplan optimization into its framework to accommodate secure spatial inter-distance of FF-sensitive regions. All TRANSPOSE approaches are demonstrated on 5 multifarious benchmarks and outperform existing FSM encoding schemes/frameworks in terms of security and overhead.

摘要: 有限状态机(FSM)管理时序电路，包括访问敏感信息和特权CPU状态。多亏了当代对激光攻击的研究，基于激光的故障注入(LFI)正变得更加精确，其中对手可以通过改变单个触发器(FF)值来破坏芯片安全。已经开发了不同的激光模型，例如比特翻转、比特设置和比特重置，以在实际目标上评价LFI。由于传统方法可能会带来较大的开销，我们在以前的工作中提出了基于状态的稀疏和基于转移的驯化对策来有效地提高有限状态机的弹性。驯服克服了Sparse过于保守的局限性，并根据需要为当代LFI模型生成多种LFI弹性编码。然而，Sparse将设计布局信息融入到其漏洞估计中，从而使其漏洞估计度量更加准确。在本文中，我们通过提出一个基于转移的编码CAD框架(TRANSPOSE)来扩展TAMD，该框架结合了空间转移脆弱性度量来量化基于位翻转模型和设置-重置模型的FSM的设计敏感度。Transspose还将布局优化整合到其框架中，以适应FF敏感区域的安全空间间距。所有转置方法都在5个不同的基准上进行了演示，并在安全性和开销方面优于现有的FSM编码方案/框架。



## **12. Stochastic Monkeys at Play: Random Augmentations Cheaply Break LLM Safety Alignment**

随机猴子在起作用：随机增强轻松打破LLM安全一致 cs.LG

Under peer review

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2411.02785v1) [paper-pdf](http://arxiv.org/pdf/2411.02785v1)

**Authors**: Jason Vega, Junsheng Huang, Gaokai Zhang, Hangoo Kang, Minjia Zhang, Gagandeep Singh

**Abstract**: Safety alignment of Large Language Models (LLMs) has recently become a critical objective of model developers. In response, a growing body of work has been investigating how safety alignment can be bypassed through various jailbreaking methods, such as adversarial attacks. However, these jailbreak methods can be rather costly or involve a non-trivial amount of creativity and effort, introducing the assumption that malicious users are high-resource or sophisticated. In this paper, we study how simple random augmentations to the input prompt affect safety alignment effectiveness in state-of-the-art LLMs, such as Llama 3 and Qwen 2. We perform an in-depth evaluation of 17 different models and investigate the intersection of safety under random augmentations with multiple dimensions: augmentation type, model size, quantization, fine-tuning-based defenses, and decoding strategies (e.g., sampling temperature). We show that low-resource and unsophisticated attackers, i.e. $\textit{stochastic monkeys}$, can significantly improve their chances of bypassing alignment with just 25 random augmentations per prompt.

摘要: 大型语言模型(LLM)的安全一致性最近已成为模型开发人员的一个重要目标。作为回应，越来越多的工作一直在研究如何通过各种越狱方法绕过安全对准，例如对抗性攻击。然而，这些越狱方法可能相当昂贵，或者涉及大量的创造力和努力，从而引入了恶意用户是高资源或老练的假设。在本文中，我们研究了输入提示的简单随机增强如何影响最新的LLMS的安全对齐有效性，例如Llama 3和Qwen 2。我们对17个不同的模型进行了深入的评估，并研究了在多个维度的随机增强下的安全性交集：增强类型、模型大小、量化、基于微调的防御和解码策略(例如采样温度)。我们表明，低资源和简单的攻击者，即$\textit{随机猴子}$，可以显著提高他们绕过对齐的机会，每个提示只需25个随机扩展。



## **13. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

19 pages, 4 figures

**SubmitDate**: 2024-11-05    [abs](http://arxiv.org/abs/2406.19692v4) [paper-pdf](http://arxiv.org/pdf/2406.19692v4)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **14. Semantic-Aligned Adversarial Evolution Triangle for High-Transferability Vision-Language Attack**

高可移植性视觉语言攻击的语义对齐对抗进化三角 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02669v1) [paper-pdf](http://arxiv.org/pdf/2411.02669v1)

**Authors**: Xiaojun Jia, Sensen Gao, Qing Guo, Ke Ma, Yihao Huang, Simeng Qin, Yang Liu, Ivor Tsang Fellow, Xiaochun Cao

**Abstract**: Vision-language pre-training (VLP) models excel at interpreting both images and text but remain vulnerable to multimodal adversarial examples (AEs). Advancing the generation of transferable AEs, which succeed across unseen models, is key to developing more robust and practical VLP models. Previous approaches augment image-text pairs to enhance diversity within the adversarial example generation process, aiming to improve transferability by expanding the contrast space of image-text features. However, these methods focus solely on diversity around the current AEs, yielding limited gains in transferability. To address this issue, we propose to increase the diversity of AEs by leveraging the intersection regions along the adversarial trajectory during optimization. Specifically, we propose sampling from adversarial evolution triangles composed of clean, historical, and current adversarial examples to enhance adversarial diversity. We provide a theoretical analysis to demonstrate the effectiveness of the proposed adversarial evolution triangle. Moreover, we find that redundant inactive dimensions can dominate similarity calculations, distorting feature matching and making AEs model-dependent with reduced transferability. Hence, we propose to generate AEs in the semantic image-text feature contrast space, which can project the original feature space into a semantic corpus subspace. The proposed semantic-aligned subspace can reduce the image feature redundancy, thereby improving adversarial transferability. Extensive experiments across different datasets and models demonstrate that the proposed method can effectively improve adversarial transferability and outperform state-of-the-art adversarial attack methods. The code is released at https://github.com/jiaxiaojunQAQ/SA-AET.

摘要: 视觉语言预训练(VLP)模型在解释图像和文本方面表现出色，但仍然容易受到多模式对抗性例子(AEs)的影响。推动可转移实体的产生是开发更稳健和实用的VLP模型的关键，这种可转移实体在看不见的模型中取得了成功。以往的方法通过增加图文对来增强对抗性实例生成过程中的多样性，旨在通过扩展图文特征的对比度空间来提高可转移性。然而，这些方法只关注当前企业的多样性，在可转让性方面的收益有限。为了解决这一问题，我们建议在优化过程中利用对抗性轨迹上的交集区域来增加AEs的多样性。具体地说，我们建议从由干净的、历史的和当前的对抗性例子组成的对抗性进化三角形中进行采样，以增强对抗性多样性。我们提供了一个理论分析，以证明所提出的对抗性进化三角的有效性。此外，我们发现冗余的非活动维度会主导相似性计算，扭曲特征匹配，并使AEs依赖于模型，降低了可转移性。因此，我们提出在语义图文特征对比空间中生成AEs，它可以将原始特征空间投影到语义语料库的子空间。提出的语义对齐子空间可以减少图像特征的冗余度，从而提高对抗性转移能力。在不同的数据集和模型上进行的大量实验表明，该方法可以有效地提高对抗攻击的可转移性，并优于最新的对抗攻击方法。该代码在https://github.com/jiaxiaojunQAQ/SA-AET.上发布



## **15. Class-Conditioned Transformation for Enhanced Robust Image Classification**

用于增强鲁棒图像分类的类别条件变换 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.15409v2) [paper-pdf](http://arxiv.org/pdf/2303.15409v2)

**Authors**: Tsachi Blau, Roy Ganz, Chaim Baskin, Michael Elad, Alex M. Bronstein

**Abstract**: Robust classification methods predominantly concentrate on algorithms that address a specific threat model, resulting in ineffective defenses against other threat models. Real-world applications are exposed to this vulnerability, as malicious attackers might exploit alternative threat models. In this work, we propose a novel test-time threat model agnostic algorithm that enhances Adversarial-Trained (AT) models. Our method operates through COnditional image transformation and DIstance-based Prediction (CODIP) and includes two main steps: First, we transform the input image into each dataset class, where the input image might be either clean or attacked. Next, we make a prediction based on the shortest transformed distance. The conditional transformation utilizes the perceptually aligned gradients property possessed by AT models and, as a result, eliminates the need for additional models or additional training. Moreover, it allows users to choose the desired balance between clean and robust accuracy without training. The proposed method achieves state-of-the-art results demonstrated through extensive experiments on various models, AT methods, datasets, and attack types. Notably, applying CODIP leads to substantial robust accuracy improvement of up to $+23\%$, $+20\%$, $+26\%$, and $+22\%$ on CIFAR10, CIFAR100, ImageNet and Flowers datasets, respectively.

摘要: 稳健的分类方法主要集中在处理特定威胁模型的算法上，导致对其他威胁模型的防御无效。现实世界中的应用程序会暴露于此漏洞，因为恶意攻击者可能会利用其他威胁模型。在这项工作中，我们提出了一种新的测试时间威胁模型不可知算法，该算法增强了对手训练(AT)模型。该方法通过条件图像变换和基于距离的预测(CODIP)进行操作，主要包括两个步骤：首先，将输入图像转换为每个数据集类，其中输入图像可能是干净的，也可能是被攻击的。接下来，我们基于最短变换距离进行预测。条件变换利用了AT模型所具有的感知对齐的梯度特性，因此不需要额外的模型或额外的训练。此外，它允许用户在干净和稳健的精确度之间选择所需的平衡，而无需培训。通过在各种模型、AT方法、数据集和攻击类型上的大量实验，该方法获得了最先进的结果。值得注意的是，在CIFAR10、CIFAR100、ImageNet和Flowers数据集上，应用CODIP可以显著提高精度，分别达到+2 3$、$+2 0$、$+2 6$和$+2 2$。



## **16. Attacking Vision-Language Computer Agents via Pop-ups**

通过弹出窗口攻击视觉语言计算机代理 cs.CL

10 pages, preprint

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02391v1) [paper-pdf](http://arxiv.org/pdf/2411.02391v1)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing the tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.

摘要: 由大视觉和语言模型(VLM)驱动的自主代理在完成日常计算机任务方面表现出了巨大的潜力，例如浏览网页预订旅行和操作桌面软件，这需要代理理解这些界面。尽管这样的视觉输入越来越多地集成到代理应用程序中，但围绕它们存在哪些类型的风险和攻击仍不清楚。在这项工作中，我们证明了VLM代理可以很容易地受到一组精心设计的敌意弹出窗口的攻击，人类用户通常会识别并忽略这些弹出窗口。这种干扰会导致工程师单击这些弹出窗口，而不是像往常一样执行任务。将这些弹出窗口集成到OSWorld和VisualWebArena等现有代理测试环境中，攻击成功率(代理单击弹出窗口的频率)平均为86%，任务成功率降低47%。基本的防御技术，如要求代理忽略弹出窗口或包括广告通知，对攻击无效。



## **17. Swiper: a new paradigm for efficient weighted distributed protocols**

Swiper：高效加权分布式协议的新范式 cs.DC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2307.15561v2) [paper-pdf](http://arxiv.org/pdf/2307.15561v2)

**Authors**: Andrei Tonkikh, Luciano Freitas

**Abstract**: The majority of fault-tolerant distributed algorithms are designed assuming a nominal corruption model, in which at most a fraction $f_n$ of parties can be corrupted by the adversary. However, due to the infamous Sybil attack, nominal models are not sufficient to express the trust assumptions in open (i.e., permissionless) settings. Instead, permissionless systems typically operate in a weighted model, where each participant is associated with a weight and the adversary can corrupt a set of parties holding at most a fraction $f_w$ of the total weight.   In this paper, we suggest a simple way to transform a large class of protocols designed for the nominal model into the weighted model. To this end, we formalize and solve three novel optimization problems, which we collectively call the weight reduction problems, that allow us to map large real weights into small integer weights while preserving the properties necessary for the correctness of the protocols. In all cases, we manage to keep the sum of the integer weights to be at most linear in the number of parties, resulting in extremely efficient protocols for the weighted model. Moreover, we demonstrate that, on weight distributions that emerge in practice, the sum of the integer weights tends to be far from the theoretical worst case and, sometimes, even smaller than the number of participants.   While, for some protocols, our transformation requires an arbitrarily small reduction in resilience (i.e., $f_w = f_n - \epsilon$), surprisingly, for many important problems, we manage to obtain weighted solutions with the same resilience ($f_w = f_n$) as nominal ones. Notable examples include erasure-coded distributed storage and broadcast protocols, verifiable secret sharing, and asynchronous consensus.

摘要: 大多数容错分布式算法的设计都假设了一个名义上的腐败模型，在该模型中，至多只有$f_n$的参与者可以被对手破坏。然而，由于臭名昭著的Sybil攻击，名义模型不足以表达开放(即无许可)环境下的信任假设。取而代之的是，未经许可的系统通常在加权模型中运行，其中每个参与者与一个权重相关联，并且对手可以破坏至多持有总权重的一小部分$f_w$的一组当事人。在本文中，我们提出了一种简单的方法，将为标称模型设计的一大类协议转换为加权模型。为此，我们形式化并解决了三个新的优化问题，我们统称为重量减轻问题，它们允许我们将大的实数权重映射为小的整数权重，同时保持协议正确性所必需的性质。在所有情况下，我们设法保持整数权重之和在参与方数量中最多是线性的，从而产生用于加权模型的极其高效的协议。此外，我们证明，在实际出现的权重分布上，整数权重的总和往往远离理论上最坏的情况，有时甚至比参与者的数量更少。虽然对于某些协议，我们的变换需要任意小的弹性降低(即$f_w=f_n-\epsilon$)，但令人惊讶的是，对于许多重要问题，我们设法获得具有与名义问题相同的弹性($f_w=f_n$)的加权解。值得注意的例子包括擦除编码的分布式存储和广播协议、可验证的秘密共享和异步共识。



## **18. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2409.13174v2) [paper-pdf](http://arxiv.org/pdf/2409.13174v2)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompts, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable Analyses of how VLAMs respond to different physical security threats. Our project page is in this link: https://chaducheng.github.io/Manipulat-Facing-Threats/.

摘要: 最近，在多模式大语言模型(MLLM)的推动下，视觉语言动作模型(VLAM)被提出以在机器人操作任务的开放词汇场景中实现更好的性能。由于操作任务涉及与物理世界的直接交互，因此确保该任务执行过程中的健壮性和安全性始终是一个非常关键的问题。本文通过综合当前MLLMS的安全研究现状和物理世界中操纵任务的具体应用场景，对VLAMS在面临潜在物理威胁的情况下进行综合评估。具体地说，我们提出了物理脆弱性评估管道(PVEP)，它可以结合尽可能多的视觉通道物理威胁来评估VLAMS的物理健壮性。PVEP中的物理威胁具体包括分发外、基于排版的视觉提示和对抗性补丁攻击。通过比较VLAM在受到攻击前后的性能波动，我们对VLAM如何应对不同的物理安全威胁提供了一般性的分析。我们的项目页面位于以下链接：https://chaducheng.github.io/Manipulat-Facing-Threats/.



## **19. Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs**

基于对齐的对抗训练（ABAT）用于提高基于脑电的BCI的稳健性和准确性 cs.HC

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.02094v1) [paper-pdf](http://arxiv.org/pdf/2411.02094v1)

**Authors**: Xiaoqing Chen, Ziwei Wang, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI studies focused on improving the decoding accuracy, with only a few considering the adversarial security. Although many adversarial defense approaches have been proposed in other application domains such as computer vision, previous research showed that their direct extensions to BCIs degrade the classification accuracy on benign samples. This phenomenon greatly affects the applicability of adversarial defense approaches to EEG-based BCIs. To mitigate this problem, we propose alignment-based adversarial training (ABAT), which performs EEG data alignment before adversarial training. Data alignment aligns EEG trials from different domains to reduce their distribution discrepancies, and adversarial training further robustifies the classification boundary. The integration of data alignment and adversarial training can make the trained EEG classifiers simultaneously more accurate and more robust. Experiments on five EEG datasets from two different BCI paradigms (motor imagery classification, and event related potential recognition), three convolutional neural network classifiers (EEGNet, ShallowCNN and DeepCNN) and three different experimental settings (offline within-subject cross-block/-session classification, online cross-session classification, and pre-trained classifiers) demonstrated its effectiveness. It is very intriguing that adversarial attacks, which are usually used to damage BCI systems, can be used in ABAT to simultaneously improve the model accuracy and robustness.

摘要: 机器学习在基于脑电(EEG)的脑机接口(BCI)领域取得了巨大的成功。现有的脑机接口研究大多集中在提高译码精度上，很少考虑对抗性安全性。虽然在计算机视觉等其他应用领域已经提出了许多对抗防御方法，但以往的研究表明，这些方法直接扩展到BCI会降低对良性样本的分类精度。这一现象极大地影响了对抗性防御方法对基于脑电信号的脑机接口的适用性。为了缓解这一问题，我们提出了基于对齐的对抗性训练(ABAT)，它在对抗性训练之前进行脑电数据对齐。数据对齐将来自不同领域的脑电试验对齐以减少它们的分布差异，而对抗性训练进一步强化了分类边界。将数据对齐和对抗性训练相结合，可以同时使训练后的脑电分类器更准确、更健壮。在两种不同的BCI范例(运动图像分类和事件相关电位识别)、三种卷积神经网络分类器(EEGNet、ShallowCNN和DeepCNN)以及三种不同的实验设置(离线内跨块/会话分类、在线跨会话分类和预先训练的分类器)上的实验证明了该方法的有效性。非常耐人寻味的是，通常用于破坏BCI系统的对抗性攻击可以同时用于提高模型的精度和鲁棒性。



## **20. Exploiting LLM Quantization**

利用LLM量化 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2405.18137v2) [paper-pdf](http://arxiv.org/pdf/2405.18137v2)

**Authors**: Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, Martin Vechev

**Abstract**: Quantization leverages lower-precision weights to reduce the memory usage of large language models (LLMs) and is a key technique for enabling their deployment on commodity hardware. While LLM quantization's impact on utility has been extensively explored, this work for the first time studies its adverse effects from a security perspective. We reveal that widely used quantization methods can be exploited to produce a harmful quantized LLM, even though the full-precision counterpart appears benign, potentially tricking users into deploying the malicious quantized model. We demonstrate this threat using a three-staged attack framework: (i) first, we obtain a malicious LLM through fine-tuning on an adversarial task; (ii) next, we quantize the malicious model and calculate constraints that characterize all full-precision models that map to the same quantized model; (iii) finally, using projected gradient descent, we tune out the poisoned behavior from the full-precision model while ensuring that its weights satisfy the constraints computed in step (ii). This procedure results in an LLM that exhibits benign behavior in full precision but when quantized, it follows the adversarial behavior injected in step (i). We experimentally demonstrate the feasibility and severity of such an attack across three diverse scenarios: vulnerable code generation, content injection, and over-refusal attack. In practice, the adversary could host the resulting full-precision model on an LLM community hub such as Hugging Face, exposing millions of users to the threat of deploying its malicious quantized version on their devices.

摘要: 量化利用较低精度的权重来减少大型语言模型(LLM)的内存使用，这是在商用硬件上部署LLM的关键技术。虽然LLM量化对效用的影响已经被广泛研究，但这项工作首次从安全的角度研究了它的不利影响。我们发现，广泛使用的量化方法可以被利用来产生有害的量化LLM，即使全精度对应的看起来是良性的，潜在地诱骗用户部署恶意量化模型。我们使用一个三阶段攻击框架演示了这一威胁：(I)首先，我们通过对敌方任务的微调来获得恶意LLM；(Ii)接下来，我们量化恶意模型，并计算映射到相同量化模型的所有全精度模型的约束；(Iii)最后，使用投影梯度下降，我们在确保其权重满足步骤(Ii)中计算的约束的同时，从全精度模型中排除有毒行为。这一过程导致LLM完全精确地表现出良性行为，但当量化时，它遵循在步骤(I)中注入的对抗性行为。我们通过实验演示了这种攻击在三种不同场景中的可行性和严重性：易受攻击的代码生成、内容注入和过度拒绝攻击。在实践中，对手可能会在LLM社区中心(如拥抱脸)上托管产生的全精度模型，使数百万用户面临在他们的设备上部署其恶意量化版本的威胁。



## **21. DeSparsify: Adversarial Attack Against Token Sparsification Mechanisms in Vision Transformers**

DeSparsify：针对Vision Transformers中代币稀疏机制的对抗攻击 cs.CV

18 pages, 6 figures

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2402.02554v2) [paper-pdf](http://arxiv.org/pdf/2402.02554v2)

**Authors**: Oryan Yehezkel, Alon Zolfi, Amit Baras, Yuval Elovici, Asaf Shabtai

**Abstract**: Vision transformers have contributed greatly to advancements in the computer vision domain, demonstrating state-of-the-art performance in diverse tasks (e.g., image classification, object detection). However, their high computational requirements grow quadratically with the number of tokens used. Token sparsification mechanisms have been proposed to address this issue. These mechanisms employ an input-dependent strategy, in which uninformative tokens are discarded from the computation pipeline, improving the model's efficiency. However, their dynamism and average-case assumption makes them vulnerable to a new threat vector - carefully crafted adversarial examples capable of fooling the sparsification mechanism, resulting in worst-case performance. In this paper, we present DeSparsify, an attack targeting the availability of vision transformers that use token sparsification mechanisms. The attack aims to exhaust the operating system's resources, while maintaining its stealthiness. Our evaluation demonstrates the attack's effectiveness on three token sparsification mechanisms and examines the attack's transferability between them and its effect on the GPU resources. To mitigate the impact of the attack, we propose various countermeasures.

摘要: 视觉转换器为计算机视觉领域的进步做出了巨大贡献，在各种任务(例如，图像分类、目标检测)中展示了最先进的性能。然而，它们的高计算需求随着使用的令牌数量的增加而呈二次曲线增长。为了解决这个问题，已经提出了令牌稀疏机制。这些机制采用了一种依赖输入的策略，在该策略中，没有提供信息的令牌被从计算流水线中丢弃，从而提高了模型的效率。然而，它们的动态性和平均情况假设使它们容易受到新的威胁向量的攻击--精心设计的敌意示例能够愚弄稀疏机制，导致最差情况下的性能。在本文中，我们提出了DeSparsify，一种针对使用令牌稀疏机制的视觉转换器的可用性的攻击。这次攻击旨在耗尽操作系统的资源，同时保持其隐蔽性。我们的评估证明了攻击在三种令牌稀疏机制上的有效性，并检查了攻击在它们之间的可传递性及其对GPU资源的影响。为了减轻攻击的影响，我们提出了各种对策。



## **22. LiDAttack: Robust Black-box Attack on LiDAR-based Object Detection**

LiDAttack：对基于LiDART的对象检测的鲁棒黑匣子攻击 cs.CV

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01889v1) [paper-pdf](http://arxiv.org/pdf/2411.01889v1)

**Authors**: Jinyin Chen, Danxin Liao, Sheng Xiang, Haibin Zheng

**Abstract**: Since DNN is vulnerable to carefully crafted adversarial examples, adversarial attack on LiDAR sensors have been extensively studied. We introduce a robust black-box attack dubbed LiDAttack. It utilizes a genetic algorithm with a simulated annealing strategy to strictly limit the location and number of perturbation points, achieving a stealthy and effective attack. And it simulates scanning deviations, allowing it to adapt to dynamic changes in real world scenario variations. Extensive experiments are conducted on 3 datasets (i.e., KITTI, nuScenes, and self-constructed data) with 3 dominant object detection models (i.e., PointRCNN, PointPillar, and PV-RCNN++). The results reveal the efficiency of the LiDAttack when targeting a wide range of object detection models, with an attack success rate (ASR) up to 90%.

摘要: 由于DNN容易受到精心设计的对抗示例的影响，因此对LiDART传感器的对抗攻击已经得到了广泛的研究。我们引入了一种名为LiDAttack的强大黑匣子攻击。它利用遗传算法和模拟退变策略来严格限制扰动点的位置和数量，实现隐蔽有效的攻击。它模拟扫描偏差，使其能够适应现实世界场景变化的动态变化。对3个数据集进行了广泛的实验（即，KITTI、nuScenes和自构建数据），具有3个主要对象检测模型（即PointRCNN、PointPillar和PV-RCNN++）。结果揭示了LiDAttack在针对广泛的对象检测模型时的效率，攻击成功率（ASB）高达90%。



## **23. Learning predictable and robust neural representations by straightening image sequences**

通过拉直图像序列学习可预测且鲁棒的神经表示 cs.CV

Accepted at NeurIPS 2024

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01777v1) [paper-pdf](http://arxiv.org/pdf/2411.01777v1)

**Authors**: Xueyan Niu, Cristina Savin, Eero P. Simoncelli

**Abstract**: Prediction is a fundamental capability of all living organisms, and has been proposed as an objective for learning sensory representations. Recent work demonstrates that in primate visual systems, prediction is facilitated by neural representations that follow straighter temporal trajectories than their initial photoreceptor encoding, which allows for prediction by linear extrapolation. Inspired by these experimental findings, we develop a self-supervised learning (SSL) objective that explicitly quantifies and promotes straightening. We demonstrate the power of this objective in training deep feedforward neural networks on smoothly-rendered synthetic image sequences that mimic commonly-occurring properties of natural videos. The learned model contains neural embeddings that are predictive, but also factorize the geometric, photometric, and semantic attributes of objects. The representations also prove more robust to noise and adversarial attacks compared to previous SSL methods that optimize for invariance to random augmentations. Moreover, these beneficial properties can be transferred to other training procedures by using the straightening objective as a regularizer, suggesting a broader utility for straightening as a principle for robust unsupervised learning.

摘要: 预测是所有生物体的一种基本能力，并被认为是学习感觉表征的目标。最近的工作表明，在灵长类视觉系统中，预测是由神经表征促进的，这些神经表征遵循比其初始光感受器编码更直接的时间轨迹，这允许通过线性外推进行预测。受这些实验结果的启发，我们制定了一个明确量化和促进矫正的自我监督学习(SSL)目标。我们展示了这一目标在平滑渲染的合成图像序列上训练深度前馈神经网络的能力，这些合成图像序列模仿自然视频的常见属性。学习的模型包含神经嵌入，这些神经嵌入是预测性的，但也分解了对象的几何、光度和语义属性。与之前的针对随机增加的不变性进行优化的SSL方法相比，该表示还被证明对噪声和敌意攻击更健壮。此外，通过使用校直目标作为正则化规则，这些有益的属性可以转移到其他训练过程中，这表明校直作为稳健的无监督学习的原则具有更广泛的实用性。



## **24. Survey on Adversarial Attack and Defense for Medical Image Analysis: Methods and Challenges**

医学图像分析的对抗性攻击与防御调查：方法与挑战 eess.IV

Accepted by ACM Computing Surveys (CSUR) (DOI:  https://doi.org/10.1145/3702638)

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2303.14133v2) [paper-pdf](http://arxiv.org/pdf/2303.14133v2)

**Authors**: Junhao Dong, Junxi Chen, Xiaohua Xie, Jianhuang Lai, Hao Chen

**Abstract**: Deep learning techniques have achieved superior performance in computer-aided medical image analysis, yet they are still vulnerable to imperceptible adversarial attacks, resulting in potential misdiagnosis in clinical practice. Oppositely, recent years have also witnessed remarkable progress in defense against these tailored adversarial examples in deep medical diagnosis systems. In this exposition, we present a comprehensive survey on recent advances in adversarial attacks and defenses for medical image analysis with a systematic taxonomy in terms of the application scenario. We also provide a unified framework for different types of adversarial attack and defense methods in the context of medical image analysis. For a fair comparison, we establish a new benchmark for adversarially robust medical diagnosis models obtained by adversarial training under various scenarios. To the best of our knowledge, this is the first survey paper that provides a thorough evaluation of adversarially robust medical diagnosis models. By analyzing qualitative and quantitative results, we conclude this survey with a detailed discussion of current challenges for adversarial attack and defense in medical image analysis systems to shed light on future research directions. Code is available on \href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.

摘要: 深度学习技术在计算机辅助医学图像分析中取得了优异的性能，但仍然容易受到潜移默化的对抗性攻击，导致临床实践中潜在的误诊。相反，近年来在防御深度医疗诊断系统中这些量身定做的对抗性例子方面也取得了显著进展。在这篇论述中，我们对医学图像分析中对抗性攻击和防御的最新进展进行了全面的综述，并根据应用场景对其进行了系统的分类。我们还为医学图像分析中不同类型的对抗性攻击和防御方法提供了一个统一的框架。为了进行公平的比较，我们建立了一个新的基准，用于在不同场景下通过对抗性训练获得对抗性健壮的医疗诊断模型。据我们所知，这是第一份对反面稳健的医疗诊断模型进行彻底评估的调查报告。通过对定性和定量结果的分析，我们对当前医学图像分析系统中对抗性攻击和防御的挑战进行了详细的讨论，以阐明未来的研究方向。代码可在\href{https://github.com/tomvii/Adv_MIA}{\color{red}{GitHub}}.上找到



## **25. A General Recipe for Contractive Graph Neural Networks -- Technical Report**

压缩图神经网络的通用配方--技术报告 cs.LG

**SubmitDate**: 2024-11-04    [abs](http://arxiv.org/abs/2411.01717v1) [paper-pdf](http://arxiv.org/pdf/2411.01717v1)

**Authors**: Maya Bechler-Speicher, Moshe Eliasof

**Abstract**: Graph Neural Networks (GNNs) have gained significant popularity for learning representations of graph-structured data due to their expressive power and scalability. However, despite their success in domains such as social network analysis, recommendation systems, and bioinformatics, GNNs often face challenges related to stability, generalization, and robustness to noise and adversarial attacks. Regularization techniques have shown promise in addressing these challenges by controlling model complexity and improving robustness. Building on recent advancements in contractive GNN architectures, this paper presents a novel method for inducing contractive behavior in any GNN through SVD regularization. By deriving a sufficient condition for contractiveness in the update step and applying constraints on network parameters, we demonstrate the impact of SVD regularization on the Lipschitz constant of GNNs. Our findings highlight the role of SVD regularization in enhancing the stability and generalization of GNNs, contributing to the development of more robust graph-based learning algorithms dynamics.

摘要: 图神经网络(GNN)由于其表达能力和可扩展性，在学习图结构数据的表示方面受到了极大的欢迎。然而，尽管GNN在社会网络分析、推荐系统和生物信息学等领域取得了成功，但它们经常面临与稳定性、泛化以及对噪声和对手攻击的健壮性相关的挑战。正则化技术通过控制模型复杂性和提高稳健性，在解决这些挑战方面表现出了希望。基于压缩GNN结构的最新进展，本文提出了一种通过奇异值分解正则化来诱导任意GNN收缩行为的新方法。通过推导更新过程中收敛的充分条件和对网络参数的约束，我们证明了奇异值分解正则化对GNN的Lipschitz常数的影响。我们的发现突出了奇异值分解正则化在增强GNN的稳定性和泛化方面的作用，有助于开发更健壮的基于图的学习算法动态。



## **26. UniGuard: Towards Universal Safety Guardrails for Jailbreak Attacks on Multimodal Large Language Models**

UniGuard：为多模式大型语言模型越狱攻击建立通用安全护栏 cs.CL

14 pages

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01703v1) [paper-pdf](http://arxiv.org/pdf/2411.01703v1)

**Authors**: Sejoon Oh, Yiqiao Jin, Megha Sharma, Donghyun Kim, Eric Ma, Gaurav Verma, Srijan Kumar

**Abstract**: Multimodal large language models (MLLMs) have revolutionized vision-language understanding but are vulnerable to multimodal jailbreak attacks, where adversaries meticulously craft inputs to elicit harmful or inappropriate responses. We propose UniGuard, a novel multimodal safety guardrail that jointly considers the unimodal and cross-modal harmful signals. UniGuard is trained such that the likelihood of generating harmful responses in a toxic corpus is minimized, and can be seamlessly applied to any input prompt during inference with minimal computational costs. Extensive experiments demonstrate the generalizability of UniGuard across multiple modalities and attack strategies. It demonstrates impressive generalizability across multiple state-of-the-art MLLMs, including LLaVA, Gemini Pro, GPT-4, MiniGPT-4, and InstructBLIP, thereby broadening the scope of our solution.

摘要: 多模式大型语言模型（MLLM）彻底改变了视觉语言理解，但很容易受到多模式越狱攻击，对手精心设计输入以引发有害或不当的反应。我们提出了UniGuard，这是一种新型的多模式安全护栏，它联合考虑了单模式和跨模式有害信号。UniGuard经过训练，可以最大限度地降低在有毒的数据库中生成有害响应的可能性，并且可以以最小的计算成本无缝地应用于推理期间的任何输入提示。大量实验证明了UniGuard在多种模式和攻击策略中的通用性。它在多个最先进的MLLM（包括LLaVA、Gemini Pro、GPT-4、MiniGPT-4和DirecectBLIP）上展示了令人印象深刻的通用性，从而扩大了我们解决方案的范围。



## **27. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

建立自我改进循环：面向目标的语义通信中的错误检测和纠正 cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01544v1) [paper-pdf](http://arxiv.org/pdf/2411.01544v1)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.

摘要: 在现代通信系统中，尤其是在复杂的传输环境中，错误检测和纠错对于确保健壮和可靠的操作是必不可少的。然而，在注重传递意义而不是符号的语义交际(SemCom)中，对这些话题的讨论在很大程度上被忽视了，导致交际效率的显著提高。尽管有这些优点，语义错误--源于发送和接收的含义之间的差异--对系统可靠性构成了重大挑战。本文提出了一个检测和纠正SemCom系统中的语义错误的综合框架，以解决这一差距。我们正式定义了语义错误、检测和纠正机制，并确定了语义错误的关键来源。为了应对这些挑战，我们开发了一种基于高斯过程(GP)的潜在空间监测方法来检测错误，并结合人在环强化学习(HITL-RL)方法来使用用户反馈来优化语义模型配置。实验结果验证了该方法在对抗性攻击、输入特征变化、物理通道变化和用户偏好变化等多种情况下的有效性。这项工作为具有健壮的语义错误管理技术的SemCom系统的可靠性和自适应性奠定了基础。



## **28. Privacy-Preserving Customer Churn Prediction Model in the Context of Telecommunication Industry**

电信行业背景下保护隐私的客户流失预测模型 cs.LG

26 pages, 14 tables, 13 figures

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2411.01447v1) [paper-pdf](http://arxiv.org/pdf/2411.01447v1)

**Authors**: Joydeb Kumar Sana, M Sohel Rahman, M Saifur Rahman

**Abstract**: Data is the main fuel of a successful machine learning model. A dataset may contain sensitive individual records e.g. personal health records, financial data, industrial information, etc. Training a model using this sensitive data has become a new privacy concern when someone uses third-party cloud computing. Trained models also suffer privacy attacks which leads to the leaking of sensitive information of the training data. This study is conducted to preserve the privacy of training data in the context of customer churn prediction modeling for the telecommunications industry (TCI). In this work, we propose a framework for privacy-preserving customer churn prediction (PPCCP) model in the cloud environment. We have proposed a novel approach which is a combination of Generative Adversarial Networks (GANs) and adaptive Weight-of-Evidence (aWOE). Synthetic data is generated from GANs, and aWOE is applied on the synthetic training dataset before feeding the data to the classification algorithms. Our experiments were carried out using eight different machine learning (ML) classifiers on three openly accessible datasets from the telecommunication sector. We then evaluated the performance using six commonly employed evaluation metrics. In addition to presenting a data privacy analysis, we also performed a statistical significance test. The training and prediction processes achieve data privacy and the prediction classifiers achieve high prediction performance (87.1\% in terms of F-Measure for GANs-aWOE based Na\"{\i}ve Bayes model). In contrast to earlier studies, our suggested approach demonstrates a prediction enhancement of up to 28.9\% and 27.9\% in terms of accuracy and F-measure, respectively.

摘要: 数据是成功的机器学习模型的主要燃料。数据集可能包含敏感的个人记录，如个人健康记录、金融数据、行业信息等。当有人使用第三方云计算时，使用这些敏感数据训练模型已成为一个新的隐私问题。训练后的模型还会遭受隐私攻击，导致训练数据的敏感信息泄露。这项研究是为了在电信行业(TCI)的客户流失预测建模的背景下保护训练数据的隐私。在这项工作中，我们提出了一种云环境下隐私保护的客户流失预测(PPCCP)模型框架。我们提出了一种新的方法，它是生成性对抗网络(GANS)和自适应证据权重(AWOE)的结合。从GANS生成合成数据，并在将数据提供给分类算法之前对合成训练数据集应用aWOE。我们的实验是在来自电信部门的三个开放可访问的数据集上使用八个不同的机器学习(ML)分类器进行的。然后，我们使用六个常用的评估指标来评估性能。除了提供数据隐私分析外，我们还执行了统计显著性测试。训练和预测过程实现了数据保密性，预测分类器获得了较高的预测性能(对于基于Gans-aWOE的Na2VeBayes模型的F度量为87.1%)，与以前的研究相比，我们的方法在预测准确率和F度量方面分别提高了28.9%和27.9%。



## **29. Enhancing Cyber-Resilience in Integrated Energy System Scheduling with Demand Response Using Deep Reinforcement Learning**

使用深度强化学习增强具有需求响应的综合能源系统调度中的网络弹性 eess.SY

Accepted by Applied Energy, Manuscript ID: APEN-D-24-03080

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2311.17941v2) [paper-pdf](http://arxiv.org/pdf/2311.17941v2)

**Authors**: Yang Li, Wenjie Ma, Yuanzheng Li, Sen Li, Zhe Chen, Mohammad Shahidehpor

**Abstract**: Optimally scheduling multi-energy flow is an effective method to utilize renewable energy sources (RES) and improve the stability and economy of integrated energy systems (IES). However, the stable demand-supply of IES faces challenges from uncertainties that arise from RES and loads, as well as the increasing impact of cyber-attacks with advanced information and communication technologies adoption. To address these challenges, this paper proposes an innovative model-free resilience scheduling method based on state-adversarial deep reinforcement learning (DRL) for integrated demand response (IDR)-enabled IES. The proposed method designs an IDR program to explore the interaction ability of electricity-gas-heat flexible loads. Additionally, the state-adversarial Markov decision process (SA-MDP) model characterizes the energy scheduling problem of IES under cyber-attack, incorporating cyber-attacks as adversaries directly into the scheduling process. The state-adversarial soft actor-critic (SA-SAC) algorithm is proposed to mitigate the impact of cyber-attacks on the scheduling strategy, integrating adversarial training into the learning process to against cyber-attacks. Simulation results demonstrate that our method is capable of adequately addressing the uncertainties resulting from RES and loads, mitigating the impact of cyber-attacks on the scheduling strategy, and ensuring a stable demand supply for various energy sources. Moreover, the proposed method demonstrates resilience against cyber-attacks. Compared to the original soft actor-critic (SAC) algorithm, it achieves a 10% improvement in economic performance under cyber-attack scenarios.

摘要: 多能流优化调度是利用可再生能源、提高综合能源系统稳定性和经济性的有效方法。然而，工业企业稳定的需求供应面临着挑战，这些挑战来自资源和负载带来的不确定性，以及采用先进信息和通信技术的网络攻击的影响越来越大。针对这些挑战，提出了一种基于状态对抗性深度强化学习(DRL)的集成需求响应(IDR)支持的IES的无模型弹性调度方法。该方法设计了一个IDR程序来研究电-气-热柔性负荷的相互作用能力。此外，状态-对抗性马尔可夫决策过程(SA-MDP)模型刻画了网络攻击下IES的能量调度问题，将网络攻击作为对手直接纳入调度过程。针对网络攻击对调度策略的影响，提出了一种状态对抗性软行为者-批评者(SA-SAC)算法，将对抗性训练融入到学习过程中来对抗网络攻击。仿真结果表明，该方法能够很好地处理资源和负荷带来的不确定性，减轻网络攻击对调度策略的影响，保证各种能源的稳定需求。此外，该方法还表现出了对网络攻击的恢复能力。与原来的软演员-批评者(SAC)算法相比，该算法在网络攻击场景下的经济性能提高了10%。



## **30. High-Frequency Anti-DreamBooth: Robust Defense against Personalized Image Synthesis**

高频反DreamBooth：针对个性化图像合成的强大防御 cs.CV

ECCV 2024 Workshop The Dark Side of Generative AIs and Beyond. Our  code is available at https://github.com/mti-lab/HF-ADB

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2409.08167v3) [paper-pdf](http://arxiv.org/pdf/2409.08167v3)

**Authors**: Takuto Onikubo, Yusuke Matsui

**Abstract**: Recently, text-to-image generative models have been misused to create unauthorized malicious images of individuals, posing a growing social problem. Previous solutions, such as Anti-DreamBooth, add adversarial noise to images to protect them from being used as training data for malicious generation. However, we found that the adversarial noise can be removed by adversarial purification methods such as DiffPure. Therefore, we propose a new adversarial attack method that adds strong perturbation on the high-frequency areas of images to make it more robust to adversarial purification. Our experiment showed that the adversarial images retained noise even after adversarial purification, hindering malicious image generation.

摘要: 最近，文本到图像的生成模型被滥用来创建未经授权的恶意个人图像，造成了日益严重的社会问题。以前的解决方案（例如Anti-DreamBooth）会向图像添加对抗性噪音，以保护它们不被用作恶意生成的训练数据。然而，我们发现对抗性噪音可以通过迪夫Pure等对抗性净化方法去除。因此，我们提出了一种新的对抗性攻击方法，该方法在图像的高频区域添加强扰动，使其对对抗性净化更稳健。我们的实验表明，即使在对抗净化之后，对抗图像也会保留噪音，从而阻碍了恶意图像的生成。



## **31. AED: An black-box NLP classifier model attacker**

AED：黑匣子NLP分类器模型攻击者 cs.LG

**SubmitDate**: 2024-11-03    [abs](http://arxiv.org/abs/2112.11660v4) [paper-pdf](http://arxiv.org/pdf/2112.11660v4)

**Authors**: Yueyang Liu, Yan Huang, Zhipeng Cai

**Abstract**: Deep Neural Networks (DNNs) have been successful in solving real-world tasks in domains such as connected and automated vehicles, disease, and job hiring. However, their implications are far-reaching in critical application areas. Hence, there is a growing concern regarding the potential bias and robustness of these DNN models. A transparency and robust model is always demanded in high-stakes domains where reliability and safety are enforced, such as healthcare and finance. While most studies have focused on adversarial image attack scenarios, fewer studies have investigated the robustness of DNN models in natural language processing (NLP) due to their adversarial samples are difficult to generate. To address this gap, we propose a word-level NLP classifier attack model called "AED," which stands for Attention mechanism enabled post-model Explanation with Density peaks clustering algorithm for synonyms search and substitution. AED aims to test the robustness of NLP DNN models by interpretability their weaknesses and exploring alternative ways to optimize them. By identifying vulnerabilities and providing explanations, AED can help improve the reliability and safety of DNN models in critical application areas such as healthcare and automated transportation. Our experiment results demonstrate that compared with other existing models, AED can effectively generate adversarial examples that can fool the victim model while maintaining the original meaning of the input.

摘要: 深度神经网络(DNN)已经成功地解决了联网和自动化车辆、疾病和就业等领域的现实世界任务。然而，它们对关键应用领域的影响是深远的。因此，人们越来越关注这些DNN模型的潜在偏差和稳健性。在要求可靠性和安全性的高风险领域，如医疗保健和金融，总是需要透明和健壮的模型。虽然大多数研究都集中在对抗性图像攻击场景上，但很少有人研究DNN模型在自然语言处理(NLP)中的稳健性，因为它们的对抗性样本很难生成。为了解决这一问题，我们提出了一种词级NLP分类器攻击模型AED，它代表了注意力机制启用的模型后解释，并使用密度峰值聚类算法进行同义词搜索和替换。AED的目的是测试NLP DNN模型的健壮性，通过解释它们的弱点并探索其他方法来优化它们。通过识别漏洞和提供解释，AED可以帮助提高DNN模型在医疗保健和自动化交通等关键应用领域的可靠性和安全性。实验结果表明，与其他已有的模型相比，AED能够有效地生成能够愚弄受害者模型的对抗性实例，同时保持输入的原始含义。



## **32. Understanding and Improving Adversarial Collaborative Filtering for Robust Recommendation**

了解和改进对抗性协作过滤以实现稳健推荐 cs.IR

To appear in NeurIPS 2024

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2410.22844v2) [paper-pdf](http://arxiv.org/pdf/2410.22844v2)

**Authors**: Kaike Zhang, Qi Cao, Yunfan Wu, Fei Sun, Huawei Shen, Xueqi Cheng

**Abstract**: Adversarial Collaborative Filtering (ACF), which typically applies adversarial perturbations at user and item embeddings through adversarial training, is widely recognized as an effective strategy for enhancing the robustness of Collaborative Filtering (CF) recommender systems against poisoning attacks. Besides, numerous studies have empirically shown that ACF can also improve recommendation performance compared to traditional CF. Despite these empirical successes, the theoretical understanding of ACF's effectiveness in terms of both performance and robustness remains unclear. To bridge this gap, in this paper, we first theoretically show that ACF can achieve a lower recommendation error compared to traditional CF with the same training epochs in both clean and poisoned data contexts. Furthermore, by establishing bounds for reductions in recommendation error during ACF's optimization process, we find that applying personalized magnitudes of perturbation for different users based on their embedding scales can further improve ACF's effectiveness. Building on these theoretical understandings, we propose Personalized Magnitude Adversarial Collaborative Filtering (PamaCF). Extensive experiments demonstrate that PamaCF effectively defends against various types of poisoning attacks while significantly enhancing recommendation performance.

摘要: 对抗性协同过滤(ACF)通过对抗性训练将对抗性扰动应用于用户和项目嵌入，被广泛认为是提高协同过滤推荐系统对中毒攻击的稳健性的有效策略。此外，大量研究表明，与传统的推荐算法相比，自适应过滤算法也能提高推荐性能。尽管取得了这些经验上的成功，但关于ACF在性能和稳健性方面的有效性的理论理解仍然不清楚。为了弥补这一差距，在本文中，我们首先从理论上证明，在干净和有毒的数据环境下，与相同训练周期的传统CF相比，ACF可以获得更低的推荐误差。此外，通过建立ACF优化过程中推荐误差减少的界限，我们发现根据不同用户的嵌入尺度对不同用户应用个性化的扰动幅度可以进一步提高ACF的有效性。在这些理论理解的基础上，我们提出了个性化幅度对抗协同过滤(PamaCF)。大量实验表明，PamaCF在有效防御各种类型的中毒攻击的同时，显著提高了推荐性能。



## **33. $B^4$: A Black-Box Scrubbing Attack on LLM Watermarks**

$B ' 4 $：对LLM水印的黑匣子清除攻击 cs.CL

**SubmitDate**: 2024-11-02    [abs](http://arxiv.org/abs/2411.01222v1) [paper-pdf](http://arxiv.org/pdf/2411.01222v1)

**Authors**: Baizhou Huang, Xiao Pu, Xiaojun Wan

**Abstract**: Watermarking has emerged as a prominent technique for LLM-generated content detection by embedding imperceptible patterns. Despite supreme performance, its robustness against adversarial attacks remains underexplored. Previous work typically considers a grey-box attack setting, where the specific type of watermark is already known. Some even necessitates knowledge about hyperparameters of the watermarking method. Such prerequisites are unattainable in real-world scenarios. Targeting at a more realistic black-box threat model with fewer assumptions, we here propose $\mathcal{B}^4$, a black-box scrubbing attack on watermarks. Specifically, we formulate the watermark scrubbing attack as a constrained optimization problem by capturing its objectives with two distributions, a Watermark Distribution and a Fidelity Distribution. This optimization problem can be approximately solved using two proxy distributions. Experimental results across 12 different settings demonstrate the superior performance of $\mathcal{B}^4$ compared with other baselines.

摘要: 通过嵌入不可察觉的模式，水印已经成为LLM生成的内容检测的一种重要技术。尽管具有最高的性能，但它对对手攻击的健壮性仍未得到充分开发。以前的工作通常考虑灰盒攻击设置，其中特定类型的水印已知。有些甚至需要关于水印方法的超参数的知识。这样的先决条件在现实世界的场景中是无法实现的。针对一个假设更少、更逼真的黑盒威胁模型，本文提出了一种针对水印的黑盒擦除攻击算法$\mathcal{B}^4$。具体地说，我们将水印洗涤攻击描述为一个约束优化问题，通过两个分布来捕获其目标，即水印分布和保真度分布。这个优化问题可以使用两个代理分布近似地解决。在12个不同设置下的实验结果表明，与其他基线相比，$\mathcal{B}^4$具有更好的性能。



## **34. Plentiful Jailbreaks with String Compositions**

弦乐作品丰富越狱 cs.CL

NeurIPS SoLaR Workshop 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.01084v1) [paper-pdf](http://arxiv.org/pdf/2411.01084v1)

**Authors**: Brian R. Y. Huang

**Abstract**: Large language models (LLMs) remain vulnerable to a slew of adversarial attacks and jailbreaking methods. One common approach employed by white-hat attackers, or \textit{red-teamers}, is to process model inputs and outputs using string-level obfuscations, which can include leetspeak, rotary ciphers, Base64, ASCII, and more. Our work extends these encoding-based attacks by unifying them in a framework of invertible string transformations. With invertibility, we can devise arbitrary \textit{string compositions}, defined as sequences of transformations, that we can encode and decode end-to-end programmatically. We devise a automated best-of-n attack that samples from a combinatorially large number of string compositions. Our jailbreaks obtain competitive attack success rates on several leading frontier models when evaluated on HarmBench, highlighting that encoding-based attacks remain a persistent vulnerability even in advanced LLMs.

摘要: 大型语言模型（LLM）仍然容易受到一系列对抗攻击和越狱方法的影响。白帽攻击者（\textit{red-teamers}）采用的一种常见方法是使用字符串级混淆处理模型输入和输出，其中可以包括leetspeak、旋转密码、Base 64、ASC等。我们的工作通过将这些基于编码的攻击统一到可逆字符串转换的框架中来扩展它们。通过可逆性，我们可以设计任意的\textit{字符串合成}，定义为转换序列，我们可以通过编程方式进行端到端编码和解码。我们设计了一种自动化的n中最佳攻击，该攻击从组合大量字符串组合中进行采样。在HarmBench上进行评估时，我们的越狱在几个领先的前沿模型上获得了有竞争力的攻击成功率，这凸显了即使在高级LLM中，基于编码的攻击仍然是一个持久的漏洞。



## **35. AdjointDEIS: Efficient Gradients for Diffusion Models**

AdjointDEIS：扩散模型的有效属性 cs.CV

Accepted in NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15020v2) [paper-pdf](http://arxiv.org/pdf/2405.15020v2)

**Authors**: Zander W. Blasingame, Chen Liu

**Abstract**: The optimization of the latents and parameters of diffusion models with respect to some differentiable metric defined on the output of the model is a challenging and complex problem. The sampling for diffusion models is done by solving either the probability flow ODE or diffusion SDE wherein a neural network approximates the score function allowing a numerical ODE/SDE solver to be used. However, naive backpropagation techniques are memory intensive, requiring the storage of all intermediate states, and face additional complexity in handling the injected noise from the diffusion term of the diffusion SDE. We propose a novel family of bespoke ODE solvers to the continuous adjoint equations for diffusion models, which we call AdjointDEIS. We exploit the unique construction of diffusion SDEs to further simplify the formulation of the continuous adjoint equations using exponential integrators. Moreover, we provide convergence order guarantees for our bespoke solvers. Significantly, we show that continuous adjoint equations for diffusion SDEs actually simplify to a simple ODE. Lastly, we demonstrate the effectiveness of AdjointDEIS for guided generation with an adversarial attack in the form of the face morphing problem. Our code will be released on our project page https://zblasingame.github.io/AdjointDEIS/

摘要: 对于定义在模型输出上的某些可微度量，扩散模型的潜伏期和参数的优化是一个具有挑战性的复杂问题。扩散模型的采样是通过求解概率流常数或扩散SDE来完成的，其中神经网络近似得分函数，从而允许使用数值常数/SDE求解器。然而，朴素的反向传播技术是内存密集型的，需要存储所有中间状态，并且在处理来自扩散SDE的扩散项的注入噪声时面临额外的复杂性。对于扩散模型的连续伴随方程，我们提出了一类新的定制型常微分方程组求解器，我们称之为AdjointDEIS。我们利用扩散SDE的独特结构来进一步简化使用指数积分器的连续伴随方程的建立。此外，我们还为我们定制的求解器提供了收敛阶保证。值得注意的是，我们证明了扩散方程的连续伴随方程实际上简化为一个简单的常微分方程组。最后，我们以人脸变形问题的形式展示了AdjointDEIS在制导生成中的有效性。我们的代码将在我们的项目页面https://zblasingame.github.io/AdjointDEIS/上发布



## **36. AdvI2I: Adversarial Image Attack on Image-to-Image Diffusion models**

AdvI 2I：对图像到图像扩散模型的对抗图像攻击 cs.CV

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.21471v2) [paper-pdf](http://arxiv.org/pdf/2410.21471v2)

**Authors**: Yaopei Zeng, Yuanpu Cao, Bochuan Cao, Yurui Chang, Jinghui Chen, Lu Lin

**Abstract**: Recent advances in diffusion models have significantly enhanced the quality of image synthesis, yet they have also introduced serious safety concerns, particularly the generation of Not Safe for Work (NSFW) content. Previous research has demonstrated that adversarial prompts can be used to generate NSFW content. However, such adversarial text prompts are often easily detectable by text-based filters, limiting their efficacy. In this paper, we expose a previously overlooked vulnerability: adversarial image attacks targeting Image-to-Image (I2I) diffusion models. We propose AdvI2I, a novel framework that manipulates input images to induce diffusion models to generate NSFW content. By optimizing a generator to craft adversarial images, AdvI2I circumvents existing defense mechanisms, such as Safe Latent Diffusion (SLD), without altering the text prompts. Furthermore, we introduce AdvI2I-Adaptive, an enhanced version that adapts to potential countermeasures and minimizes the resemblance between adversarial images and NSFW concept embeddings, making the attack more resilient against defenses. Through extensive experiments, we demonstrate that both AdvI2I and AdvI2I-Adaptive can effectively bypass current safeguards, highlighting the urgent need for stronger security measures to address the misuse of I2I diffusion models.

摘要: 扩散模型的最新进展显著提高了图像合成的质量，但也引入了严重的安全问题，特别是不安全工作(NSFW)内容的产生。以前的研究已经证明，对抗性提示可以用来生成NSFW内容。然而，这种对抗性的文本提示通常很容易被基于文本的过滤器检测到，从而限制了它们的有效性。在本文中，我们暴露了一个以前被忽视的漏洞：针对图像到图像(I2I)扩散模型的对抗性图像攻击。我们提出了AdvI2I，这是一个新的框架，它通过操作输入图像来诱导扩散模型来生成NSFW内容。通过优化生成器来制作敌意图像，AdvI2I在不改变文本提示的情况下绕过了现有的防御机制，如安全潜在扩散(SLD)。此外，我们引入了AdvI2I-自适应，这是一个增强版本，它适应潜在的对策，并将敌对图像和NSFW概念嵌入之间的相似性降至最低，使攻击对防御更具弹性。通过广泛的实验，我们证明了AdvI2I和AdvI2I-自适应都可以有效地绕过当前的保障措施，这突显了迫切需要更强大的安全措施来解决I2I扩散模型的滥用。



## **37. Efficient Adversarial Training in LLMs with Continuous Attacks**

在具有持续攻击的LLC中进行有效的对抗训练 cs.LG

19 pages, 4 figures

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.15589v3) [paper-pdf](http://arxiv.org/pdf/2405.15589v3)

**Authors**: Sophie Xhonneux, Alessandro Sordoni, Stephan Günnemann, Gauthier Gidel, Leo Schwinn

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can bypass their safety guardrails. In many domains, adversarial training has proven to be one of the most promising methods to reliably improve robustness against such attacks. Yet, in the context of LLMs, current methods for adversarial training are hindered by the high computational costs required to perform discrete adversarial attacks at each training iteration. We address this problem by instead calculating adversarial attacks in the continuous embedding space of the LLM, which is orders of magnitudes more efficient. We propose a fast adversarial training algorithm (C-AdvUL) composed of two losses: the first makes the model robust on continuous embedding attacks computed on an adversarial behaviour dataset; the second ensures the usefulness of the final model by fine-tuning on utility data. Moreover, we introduce C-AdvIPO, an adversarial variant of IPO that does not require utility data for adversarially robust alignment. Our empirical evaluation on five models from different families (Gemma, Phi3, Mistral, Zephyr, Llama2) and at different scales (2B, 3.8B, 7B) shows that both algorithms substantially enhance LLM robustness against discrete attacks (GCG, AutoDAN, PAIR), while maintaining utility. Our results demonstrate that robustness to continuous perturbations can extrapolate to discrete threat models. Thereby, we present a path toward scalable adversarial training algorithms for robustly aligning LLMs.

摘要: 大型语言模型(LLM)容易受到敌意攻击，这些攻击可以绕过它们的安全护栏。在许多领域，对抗性训练已被证明是可靠地提高对此类攻击的稳健性的最有前途的方法之一。然而，在LLMS的背景下，当前的对抗性训练方法受到在每次训练迭代中执行离散对抗性攻击所需的高计算成本的阻碍。我们通过在LLM的连续嵌入空间中计算敌意攻击来解决这个问题，该空间的效率要高出几个数量级。我们提出了一种快速对抗性训练算法(C-AdvUL)，该算法由两个损失组成：第一个损失使模型对基于对抗行为数据集的连续嵌入攻击具有健壮性；第二个损失通过对效用数据的微调来确保最终模型的有效性。此外，我们引入了C-AdvIPO，这是IPO的一个对抗性变体，不需要效用数据来进行对抗性强健的比对。我们对来自不同家族(Gema，Phi3，Mistral，Zephy，Llama2)和不同尺度(2B，3.8B，7B)的五个模型的经验评估表明，这两种算法在保持实用性的同时，显著增强了LLM对离散攻击(GCG，AutoDAN，Pair)的稳健性。我们的结果表明，对连续扰动的稳健性可以外推到离散威胁模型。因此，我们提出了一条可扩展的对抗性训练算法，用于鲁棒对齐LLM。



## **38. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

对抗非中心扰动：用$l_1$-模估计精确恢复线性系统 math.OC

Theorem 1 turned out to be incorrect

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2410.03218v2) [paper-pdf](http://arxiv.org/pdf/2410.03218v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are completely arbitrary. We show that when the probability of having an attack at a given time is less than 0.5, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.

摘要: 本文研究一般情况下的线性系统辨识问题，其中扰动是亚高斯的，相关的，可能是对抗性的。首先，我们考虑了具有非中心(非零均值)扰动的情况，对于这种情况，普通的最小二乘(OLS)方法不能正确地辨识系统。我们证明了在每个扰动具有相等的正负概率的条件下，$L_1$-范数估计量能够准确地辨识系统。这一条件限制了每个扰动的符号，但允许其大小任意。其次，在攻击次数偶尔发生但攻击值的分布完全任意的情况下，我们考虑了每次扰动是对抗性的情况。我们证明了当给定时刻发生攻击的概率小于0.5时，$L_1$-范数估计对任何对抗性非中心扰动都是有效的，并且在有限时间内实现了精确的恢复。这些结果为在安全关键系统中有效防御任意规模的非中心攻击铺平了道路。



## **39. Intruding with Words: Towards Understanding Graph Injection Attacks at the Text Level**

文字入侵：在文本层面了解图注入攻击 cs.LG

Accepted by NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16405v2) [paper-pdf](http://arxiv.org/pdf/2405.16405v2)

**Authors**: Runlin Lei, Yuwei Hu, Yuchen Ren, Zhewei Wei

**Abstract**: Graph Neural Networks (GNNs) excel across various applications but remain vulnerable to adversarial attacks, particularly Graph Injection Attacks (GIAs), which inject malicious nodes into the original graph and pose realistic threats. Text-attributed graphs (TAGs), where nodes are associated with textual features, are crucial due to their prevalence in real-world applications and are commonly used to evaluate these vulnerabilities. However, existing research only focuses on embedding-level GIAs, which inject node embeddings rather than actual textual content, limiting their applicability and simplifying detection. In this paper, we pioneer the exploration of GIAs at the text level, presenting three novel attack designs that inject textual content into the graph. Through theoretical and empirical analysis, we demonstrate that text interpretability, a factor previously overlooked at the embedding level, plays a crucial role in attack strength. Among the designs we investigate, the Word-frequency-based Text-level GIA (WTGIA) is particularly notable for its balance between performance and interpretability. Despite the success of WTGIA, we discover that defenders can easily enhance their defenses with customized text embedding methods or large language model (LLM)--based predictors. These insights underscore the necessity for further research into the potential and practical significance of text-level GIAs.

摘要: 图神经网络(GNN)在各种应用中表现出色，但仍然容易受到对手攻击，特别是图注入攻击(GIA)，图注入攻击将恶意节点注入到原始图中，并构成现实威胁。文本属性图(TAG)将节点与文本特征相关联，由于它们在现实应用程序中的普遍存在，因此至关重要，并且通常用于评估这些漏洞。然而，现有的研究只关注嵌入级GIA，这些GIA注入的是节点嵌入而不是实际的文本内容，限制了它们的适用性，简化了检测。在本文中，我们率先在文本层面上探索了GIA，提出了三种向图形中注入文本内容的新颖攻击设计。通过理论和实证分析，我们证明了文本可解释性对攻击强度起着至关重要的作用，而文本可解释性是此前在嵌入层面被忽视的一个因素。在我们研究的设计中，基于词频的文本级别GIA(WTGIA)特别值得注意的是它在性能和可解释性之间的平衡。尽管WTGIA取得了成功，但我们发现，防御者可以很容易地通过定制的文本嵌入方法或基于大型语言模型(LLM)的预测器来增强他们的防御。这些见解突显了进一步研究文本层面全球影响的潜力和现实意义的必要性。



## **40. Improved Generation of Adversarial Examples Against Safety-aligned LLMs**

针对安全一致的LLM改进对抗示例的生成 cs.CR

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.20778v2) [paper-pdf](http://arxiv.org/pdf/2405.20778v2)

**Authors**: Qizhang Li, Yiwen Guo, Wangmeng Zuo, Hao Chen

**Abstract**: Adversarial prompts generated using gradient-based methods exhibit outstanding performance in performing automatic jailbreak attacks against safety-aligned LLMs. Nevertheless, due to the discrete nature of texts, the input gradient of LLMs struggles to precisely reflect the magnitude of loss change that results from token replacements in the prompt, leading to limited attack success rates against safety-aligned LLMs, even in the white-box setting. In this paper, we explore a new perspective on this problem, suggesting that it can be alleviated by leveraging innovations inspired in transfer-based attacks that were originally proposed for attacking black-box image classification models. For the first time, we appropriate the ideologies of effective methods among these transfer-based attacks, i.e., Skip Gradient Method and Intermediate Level Attack, into gradient-based adversarial prompt generation and achieve significant performance gains without introducing obvious computational cost. Meanwhile, by discussing mechanisms behind the gains, new insights are drawn, and proper combinations of these methods are also developed. Our empirical results show that 87% of the query-specific adversarial suffixes generated by the developed combination can induce Llama-2-7B-Chat to produce the output that exactly matches the target string on AdvBench. This match rate is 33% higher than that of a very strong baseline known as GCG, demonstrating advanced discrete optimization for adversarial prompt generation against LLMs. In addition, without introducing obvious cost, the combination achieves >30% absolute increase in attack success rates compared with GCG when generating both query-specific (38% -> 68%) and universal adversarial prompts (26.68% -> 60.32%) for attacking the Llama-2-7B-Chat model on AdvBench. Code at: https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.

摘要: 使用基于梯度的方法生成的对抗性提示在执行针对安全对齐的LLM的自动越狱攻击方面表现出出色的性能。然而，由于文本的离散性，LLMS的输入梯度难以准确反映提示中令牌替换导致的损失变化的大小，导致即使在白盒设置下，对安全对齐的LLM的攻击成功率也是有限的。在这篇文章中，我们探索了一个新的视角来解决这个问题，建议通过利用最初被提出用于攻击黑盒图像分类模型的基于传输的攻击的创新来缓解这个问题。我们首次将这些基于转移的攻击中有效方法的思想，即跳过梯度法和中级攻击，应用到基于梯度的对抗性提示生成中，并且在不引入明显计算代价的情况下获得了显著的性能提升。同时，通过讨论收益背后的机制，得出了新的见解，并开发了这些方法的适当组合。我们的实验结果表明，该组合生成的特定于查询的敌意后缀中，87%可以诱导Llama-2-7B-Chat生成与AdvBch上的目标字符串完全匹配的输出。这一匹配率比非常强的基线GCG的匹配率高出33%，展示了针对LLMS的对抗性提示生成的高级离散优化。此外，在不引入明显成本的情况下，与GCG相比，在生成针对特定查询(38%->68%)和通用对手提示(26.68%->60.32%)的攻击提示时，该组合的攻击成功率绝对值提高了30%以上。代码：https://github.com/qizhangli/Gradient-based-Jailbreak-Attacks.



## **41. Uncertainty-based Offline Variational Bayesian Reinforcement Learning for Robustness under Diverse Data Corruptions**

基于不确定性的离线变分Bayesian强化学习在不同数据损坏下的鲁棒性 cs.LG

Accepted to NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00465v1) [paper-pdf](http://arxiv.org/pdf/2411.00465v1)

**Authors**: Rui Yang, Jie Wang, Guoping Wu, Bin Li

**Abstract**: Real-world offline datasets are often subject to data corruptions (such as noise or adversarial attacks) due to sensor failures or malicious attacks. Despite advances in robust offline reinforcement learning (RL), existing methods struggle to learn robust agents under high uncertainty caused by the diverse corrupted data (i.e., corrupted states, actions, rewards, and dynamics), leading to performance degradation in clean environments. To tackle this problem, we propose a novel robust variational Bayesian inference for offline RL (TRACER). It introduces Bayesian inference for the first time to capture the uncertainty via offline data for robustness against all types of data corruptions. Specifically, TRACER first models all corruptions as the uncertainty in the action-value function. Then, to capture such uncertainty, it uses all offline data as the observations to approximate the posterior distribution of the action-value function under a Bayesian inference framework. An appealing feature of TRACER is that it can distinguish corrupted data from clean data using an entropy-based uncertainty measure, since corrupted data often induces higher uncertainty and entropy. Based on the aforementioned measure, TRACER can regulate the loss associated with corrupted data to reduce its influence, thereby enhancing robustness and performance in clean environments. Experiments demonstrate that TRACER significantly outperforms several state-of-the-art approaches across both individual and simultaneous data corruptions.

摘要: 由于传感器故障或恶意攻击，现实世界中的离线数据集经常受到数据损坏(如噪声或敌意攻击)的影响。尽管在稳健的离线强化学习(RL)方面取得了进展，但现有的方法难以在由不同的被破坏的数据(即，被破坏的状态、动作、奖励和动态)造成的高度不确定性下学习稳健的主体，从而导致在清洁环境中的性能下降。针对这一问题，我们提出了一种新的用于离线跟踪的稳健变分贝叶斯推理方法。它首次引入贝叶斯推理，通过离线数据捕捉不确定性，从而对所有类型的数据损坏具有健壮性。具体地说，Tracer首先将所有腐败建模为动作值函数中的不确定性。然后，为了捕捉这种不确定性，它使用所有离线数据作为观测值，在贝叶斯推理框架下近似动作值函数的后验分布。Tracer的一个吸引人的特点是，它可以使用基于熵的不确定性度量来区分损坏的数据和干净的数据，因为损坏的数据通常会导致更高的不确定性和熵。基于上述措施，Tracer可以控制与损坏数据相关的损失，以减少其影响，从而增强在清洁环境中的健壮性和性能。实验表明，Tracer在单个和同时数据损坏方面的性能明显优于几种最先进的方法。



## **42. Adversarial Purification and Fine-tuning for Robust UDC Image Restoration**

鲁棒UDC图像恢复的对抗净化和微调 eess.IV

Failure to meet expectations

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2402.13629v3) [paper-pdf](http://arxiv.org/pdf/2402.13629v3)

**Authors**: Zhenbo Song, Zhenyuan Zhang, Kaihao Zhang, Zhaoxin Fan, Jianfeng Lu

**Abstract**: This study delves into the enhancement of Under-Display Camera (UDC) image restoration models, focusing on their robustness against adversarial attacks. Despite its innovative approach to seamless display integration, UDC technology faces unique image degradation challenges exacerbated by the susceptibility to adversarial perturbations. Our research initially conducts an in-depth robustness evaluation of deep-learning-based UDC image restoration models by employing several white-box and black-box attacking methods. This evaluation is pivotal in understanding the vulnerabilities of current UDC image restoration techniques. Following the assessment, we introduce a defense framework integrating adversarial purification with subsequent fine-tuning processes. First, our approach employs diffusion-based adversarial purification, effectively neutralizing adversarial perturbations. Then, we apply the fine-tuning methodologies to refine the image restoration models further, ensuring that the quality and fidelity of the restored images are maintained. The effectiveness of our proposed approach is validated through extensive experiments, showing marked improvements in resilience against typical adversarial attacks.

摘要: 该研究深入研究了显示下摄像机(UDC)图像恢复模型的增强，重点研究其对对手攻击的健壮性。尽管UDC技术以创新的方式实现了无缝显示集成，但它面临着独特的图像降级挑战，这一挑战因易受对抗性干扰而加剧。我们的研究首先采用了几种白盒和黑盒攻击方法，对基于深度学习的UDC图像恢复模型进行了深入的稳健性评估。这一评估对于理解当前UDC图像恢复技术的脆弱性至关重要。在评估之后，我们介绍了一个集成了对抗性净化和后续微调过程的防御框架。首先，我们的方法采用了基于扩散的对抗性净化，有效地中和了对抗性扰动。然后，我们应用微调方法进一步改进图像恢复模型，确保恢复图像的质量和保真度得到保持。通过大量的实验验证了该方法的有效性，表明该方法在抵抗典型的对抗性攻击方面有了显著的提高。



## **43. Towards Building Secure UAV Navigation with FHE-aware Knowledge Distillation**

利用FHE感知的知识提炼构建安全的无人机导航 cs.CR

arXiv admin note: text overlap with arXiv:2404.17225

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00403v1) [paper-pdf](http://arxiv.org/pdf/2411.00403v1)

**Authors**: Arjun Ramesh Kaushik, Charanjit Jutla, Nalini Ratha

**Abstract**: In safeguarding mission-critical systems, such as Unmanned Aerial Vehicles (UAVs), preserving the privacy of path trajectories during navigation is paramount. While the combination of Reinforcement Learning (RL) and Fully Homomorphic Encryption (FHE) holds promise, the computational overhead of FHE presents a significant challenge. This paper proposes an innovative approach that leverages Knowledge Distillation to enhance the practicality of secure UAV navigation. By integrating RL and FHE, our framework addresses vulnerabilities to adversarial attacks while enabling real-time processing of encrypted UAV camera feeds, ensuring data security. To mitigate FHE's latency, Knowledge Distillation is employed to compress the network, resulting in an impressive 18x speedup without compromising performance, as evidenced by an R-squared score of 0.9499 compared to the original model's score of 0.9631. Our methodology underscores the feasibility of processing encrypted data for UAV navigation tasks, emphasizing security alongside performance efficiency and timely processing. These findings pave the way for deploying autonomous UAVs in sensitive environments, bolstering their resilience against potential security threats.

摘要: 在保护任务关键系统，如无人机(UAV)中，保护导航过程中路径轨迹的隐私是至关重要的。虽然强化学习(RL)和完全同态加密(FHE)的结合有希望，但FHE的计算开销是一个巨大的挑战。本文提出了一种利用知识蒸馏来增强无人机安全导航实用性的创新方法。通过集成RL和FHE，我们的框架解决了对抗攻击的漏洞，同时允许实时处理加密的无人机摄像头馈送，确保数据安全。为了减少FHE的延迟，使用知识蒸馏来压缩网络，在不影响性能的情况下获得令人印象深刻的18倍加速，R平方分数为0.9499，而原始模型的分数为0.9631。我们的方法强调了为无人机导航任务处理加密数据的可行性，强调安全性以及性能效率和及时处理。这些发现为在敏感环境中部署自动无人机铺平了道路，增强了它们对潜在安全威胁的韧性。



## **44. Replace-then-Perturb: Targeted Adversarial Attacks With Visual Reasoning for Vision-Language Models**

替换然后扰动：视觉语言模型的视觉推理的有针对性的对抗攻击 cs.CV

13 pages, 5 figure

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00898v1) [paper-pdf](http://arxiv.org/pdf/2411.00898v1)

**Authors**: Jonggyu Jang, Hyeonsu Lyu, Jungyeon Koh, Hyun Jong Yang

**Abstract**: The conventional targeted adversarial attacks add a small perturbation to an image to make neural network models estimate the image as a predefined target class, even if it is not the correct target class. Recently, for visual-language models (VLMs), the focus of targeted adversarial attacks is to generate a perturbation that makes VLMs answer intended target text outputs. For example, they aim to make a small perturbation on an image to make VLMs' answers change from "there is an apple" to "there is a baseball." However, answering just intended text outputs is insufficient for tricky questions like "if there is a baseball, tell me what is below it." This is because the target of the adversarial attacks does not consider the overall integrity of the original image, thereby leading to a lack of visual reasoning. In this work, we focus on generating targeted adversarial examples with visual reasoning against VLMs. To this end, we propose 1) a novel adversarial attack procedure -- namely, Replace-then-Perturb and 2) a contrastive learning-based adversarial loss -- namely, Contrastive-Adv. In Replace-then-Perturb, we first leverage a text-guided segmentation model to find the target object in the image. Then, we get rid of the target object and inpaint the empty space with the desired prompt. By doing this, we can generate a target image corresponding to the desired prompt, while maintaining the overall integrity of the original image. Furthermore, in Contrastive-Adv, we design a novel loss function to obtain better adversarial examples. Our extensive benchmark results demonstrate that Replace-then-Perturb and Contrastive-Adv outperform the baseline adversarial attack algorithms. We note that the source code to reproduce the results will be available.

摘要: 传统的有针对性的对抗性攻击在图像上添加一个小的扰动，使神经网络模型将图像估计为预定义的目标类，即使它不是正确的目标类。目前，对于视觉语言模型(VLMS)，目标攻击的重点是产生一种扰动，使VLMS对目标文本输出做出响应。例如，他们的目标是在图像上做一个小扰动，使VLMS的答案从“有一个苹果”变成“有一个棒球”。然而，对于像“如果有棒球，告诉我它下面是什么”这样的棘手问题，仅仅回答预期的文本输出是不够的。这是因为对抗性攻击的目标没有考虑原始图像的整体完整性，从而导致缺乏视觉推理。在这项工作中，我们专注于使用视觉推理来生成针对VLM的对抗性实例。为此，我们提出了1)一种新的对抗性攻击方法--即替换-然后-扰动；2)一种基于对比学习的对抗性损失--即对比性-预见性。在替换-然后-扰动中，我们首先利用文本引导的分割模型来寻找图像中的目标对象。然后，我们删除目标对象，并在空白区域内画上所需的提示符。通过这样做，我们可以生成与所需提示相对应的目标图像，同时保持原始图像的整体完整性。此外，在对比式ADV中，我们设计了一种新的损失函数来获得更好的对抗性实例。我们广泛的基准测试结果表明，替换-然后-扰动和对比-ADV算法的性能优于基线对抗性攻击算法。我们注意到，复制结果的源代码将可用。



## **45. OSLO: One-Shot Label-Only Membership Inference Attacks**

Oslo：一次性标签会员推断攻击 cs.LG

To appear at NeurIPS 2024

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2405.16978v3) [paper-pdf](http://arxiv.org/pdf/2405.16978v3)

**Authors**: Yuefeng Peng, Jaechul Roh, Subhransu Maji, Amir Houmansadr

**Abstract**: We introduce One-Shot Label-Only (OSLO) membership inference attacks (MIAs), which accurately infer a given sample's membership in a target model's training set with high precision using just \emph{a single query}, where the target model only returns the predicted hard label. This is in contrast to state-of-the-art label-only attacks which require $\sim6000$ queries, yet get attack precisions lower than OSLO's. OSLO leverages transfer-based black-box adversarial attacks. The core idea is that a member sample exhibits more resistance to adversarial perturbations than a non-member. We compare OSLO against state-of-the-art label-only attacks and demonstrate that, despite requiring only one query, our method significantly outperforms previous attacks in terms of precision and true positive rate (TPR) under the same false positive rates (FPR). For example, compared to previous label-only MIAs, OSLO achieves a TPR that is at least 7$\times$ higher under a 1\% FPR and at least 22$\times$ higher under a 0.1\% FPR on CIFAR100 for a ResNet18 model. We evaluated multiple defense mechanisms against OSLO.

摘要: 我们引入了一次仅标签(Oslo)成员关系推理攻击(MIA)，该攻击仅使用目标模型返回预测的硬标签，即可高精度地推断给定样本在目标模型训练集中的成员资格。这与最先进的纯标签攻击形成对比，后者需要$\sim6000$查询，但获得的攻击精度低于奥斯陆的攻击精度。奥斯陆利用基于传输的黑盒对抗攻击。其核心思想是成员样本比非成员样本对对抗性扰动表现出更强的抵抗力。我们将Oslo与最先进的纯标签攻击进行了比较，并证明了尽管只需要一次查询，但在相同的误检率(FPR)下，我们的方法在准确率和真阳性率(TPR)方面明显优于以前的攻击。例如，与以前的纯标签MIA相比，对于ResNet18型号，Oslo在CIFAR100上实现的TPR在1 FP R下至少高出7$\x$，在0.1 FP R下至少高出22$\x$。我们评估了针对奥斯陆的多种防御机制。



## **46. Quantum Entanglement Path Selection and Qubit Allocation via Adversarial Group Neural Bandits**

对抗群神经盗贼的量子纠缠路径选择和量子位分配 quant-ph

Accepted by IEEE/ACM Transactions on Networking

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00316v1) [paper-pdf](http://arxiv.org/pdf/2411.00316v1)

**Authors**: Yin Huang, Lei Wang, Jie Xu

**Abstract**: Quantum Data Networks (QDNs) have emerged as a promising framework in the field of information processing and transmission, harnessing the principles of quantum mechanics. QDNs utilize a quantum teleportation technique through long-distance entanglement connections, encoding data information in quantum bits (qubits). Despite being a cornerstone in various quantum applications, quantum entanglement encounters challenges in establishing connections over extended distances due to probabilistic processes influenced by factors like optical fiber losses. The creation of long-distance entanglement connections between quantum computers involves multiple entanglement links and entanglement swapping techniques through successive quantum nodes, including quantum computers and quantum repeaters, necessitating optimal path selection and qubit allocation. Current research predominantly assumes known success rates of entanglement links between neighboring quantum nodes and overlooks potential network attackers. This paper addresses the online challenge of optimal path selection and qubit allocation, aiming to learn the best strategy for achieving the highest success rate of entanglement connections between two chosen quantum computers without prior knowledge of the success rate and in the presence of a QDN attacker. The proposed approach is based on multi-armed bandits, specifically adversarial group neural bandits, which treat each path as a group and view qubit allocation as arm selection. Our contributions encompass formulating an online adversarial optimization problem, introducing the EXPNeuralUCB bandits algorithm with theoretical performance guarantees, and conducting comprehensive simulations to showcase its superiority over established advanced algorithms.

摘要: 利用量子力学的原理，量子数据网络(QDNS)已经成为信息处理和传输领域的一个很有前途的框架。QDNS利用一种通过长距离纠缠连接的量子隐形传态技术，将数据信息编码为量子比特(Qbit)。尽管量子纠缠是各种量子应用的基石，但由于受光纤损耗等因素影响的概率过程，量子纠缠在建立远距离连接方面遇到了挑战。在量子计算机之间建立远距离纠缠连接涉及多个纠缠链路和通过包括量子计算机和量子中继器在内的连续量子节点的纠缠交换技术，这就需要最优路径选择和量子比特分配。目前的研究主要假设相邻量子节点之间纠缠链路的已知成功率，而忽略了潜在的网络攻击者。针对最优路径选择和量子比特分配的在线挑战，研究如何在不知道纠缠成功率的情况下，在量子网络攻击者的存在下，实现两台量子计算机间纠缠连接的最高成功率.该方法基于多臂强盗，特别是对抗性群体神经强盗，将每条路径视为一组，并将量子比特分配视为手臂选择。我们的贡献包括构建一个在线对抗性优化问题，引入具有理论性能保证的EXPNeuralUCB Bandits算法，并进行全面的模拟以展示其相对于已有的高级算法的优势。



## **47. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

10 pages, 6 figures, 2 tables. Version Update: Neurips Camera Ready

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2406.18451v3) [paper-pdf](http://arxiv.org/pdf/2406.18451v3)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate high margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively and confidently use the logit margin to detect brittle decisions with such models. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to assess adversarial vulnerability in deployment scenarios efficiently.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过在CIFAR10和CIFAR100数据集上对各种稳健训练模型的综合实证分析，我们表明它们表明了高边际一致性，并且它们的输入空间边际与Logit边际之间具有很强的相关性。然后，我们证明了我们可以有效和自信地使用Logit边际来检测这样的模型的脆性决策。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **48. Efficient Model Compression for Bayesian Neural Networks**

Bayesian神经网络的高效模型压缩 cs.LG

**SubmitDate**: 2024-11-01    [abs](http://arxiv.org/abs/2411.00273v1) [paper-pdf](http://arxiv.org/pdf/2411.00273v1)

**Authors**: Diptarka Saha, Zihe Liu, Feng Liang

**Abstract**: Model Compression has drawn much attention within the deep learning community recently. Compressing a dense neural network offers many advantages including lower computation cost, deployability to devices of limited storage and memories, and resistance to adversarial attacks. This may be achieved via weight pruning or fully discarding certain input features. Here we demonstrate a novel strategy to emulate principles of Bayesian model selection in a deep learning setup. Given a fully connected Bayesian neural network with spike-and-slab priors trained via a variational algorithm, we obtain the posterior inclusion probability for every node that typically gets lost. We employ these probabilities for pruning and feature selection on a host of simulated and real-world benchmark data and find evidence of better generalizability of the pruned model in all our experiments.

摘要: 模型压缩最近引起了深度学习社区的广泛关注。压缩密集神经网络具有许多优势，包括较低的计算成本、可部署到有限存储和内存的设备以及抵抗对抗性攻击。这可以通过权重修剪或完全丢弃某些输入特征来实现。在这里，我们展示了一种新颖的策略，可以在深度学习设置中模拟Bayesian模型选择的原则。给定一个完全连接的Bayesian神经网络，其具有通过变分算法训练的尖峰和板先验，我们获得每个通常丢失的节点的后验包含概率。我们在大量模拟和现实世界的基准数据上使用这些概率进行修剪和特征选择，并在所有实验中找到修剪模型更好概括性的证据。



## **49. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

The camera-ready version of JailbreakBench v1.0 (accepted at NeurIPS  2024 Datasets and Benchmarks Track): more attack artifacts, more test-time  defenses, a more accurate jailbreak judge (Llama-3-70B with a custom prompt),  a larger dataset of human preferences for selecting a jailbreak judge (300  examples), an over-refusal evaluation dataset, a semantic refusal judge based  on Llama-3-8B

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2404.01318v5) [paper-pdf](http://arxiv.org/pdf/2404.01318v5)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **50. Protecting Feed-Forward Networks from Adversarial Attacks Using Predictive Coding**

使用预测编码保护前向网络免受对抗攻击 cs.CR

**SubmitDate**: 2024-10-31    [abs](http://arxiv.org/abs/2411.00222v1) [paper-pdf](http://arxiv.org/pdf/2411.00222v1)

**Authors**: Ehsan Ganjidoost, Jeff Orchard

**Abstract**: An adversarial example is a modified input image designed to cause a Machine Learning (ML) model to make a mistake; these perturbations are often invisible or subtle to human observers and highlight vulnerabilities in a model's ability to generalize from its training data. Several adversarial attacks can create such examples, each with a different perspective, effectiveness, and perceptibility of changes. Conversely, defending against such adversarial attacks improves the robustness of ML models in image processing and other domains of deep learning. Most defence mechanisms require either a level of model awareness, changes to the model, or access to a comprehensive set of adversarial examples during training, which is impractical. Another option is to use an auxiliary model in a preprocessing manner without changing the primary model. This study presents a practical and effective solution -- using predictive coding networks (PCnets) as an auxiliary step for adversarial defence. By seamlessly integrating PCnets into feed-forward networks as a preprocessing step, we substantially bolster resilience to adversarial perturbations. Our experiments on MNIST and CIFAR10 demonstrate the remarkable effectiveness of PCnets in mitigating adversarial examples with about 82% and 65% improvements in robustness, respectively. The PCnet, trained on a small subset of the dataset, leverages its generative nature to effectively counter adversarial efforts, reverting perturbed images closer to their original forms. This innovative approach holds promise for enhancing the security and reliability of neural network classifiers in the face of the escalating threat of adversarial attacks.

摘要: 一个对抗性的例子是修改的输入图像，旨在导致机器学习(ML)模型出错；这些扰动对于人类观察者来说通常是不可见的或微妙的，并突显了模型从其训练数据进行泛化的能力中的弱点。几个对抗性攻击可以创建这样的例子，每个例子都具有不同的视角、有效性和对变化的感知能力。相反，防御这种敌意攻击提高了ML模型在图像处理和其他深度学习领域的稳健性。大多数防御机制要么需要一定程度的模型意识，要么需要更改模型，或者需要在训练期间获得一套全面的对抗性例子，这是不切实际的。另一种选择是在不改变主模型的情况下以预处理方式使用辅助模型。本研究提出了一种实用有效的解决方案--使用预测编码网络(PCnet)作为对抗防御的辅助步骤。通过将PCnet无缝地整合到前馈网络中作为一个预处理步骤，我们大大增强了对对手扰动的复原力。我们在MNIST和CIFAR10上的实验表明，PCnet在减少敌意例子方面具有显著的效果，健壮性分别提高了82%和65%。PCnet在数据集的一小部分上进行训练，利用其生成性来有效地对抗对手的努力，使受干扰的图像恢复到更接近其原始形式。这种创新的方法有望在面对不断升级的对抗性攻击威胁时增强神经网络分类器的安全性和可靠性。



