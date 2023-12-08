# Latest Adversarial Attack Papers
**update at 2023-12-08 09:48:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Memory Triggers: Unveiling Memorization in Text-To-Image Generative Models through Word-Level Duplication**

记忆触发器：通过词级复制揭示文本到图像生成模型中的记忆 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03692v1) [paper-pdf](http://arxiv.org/pdf/2312.03692v1)

**Authors**: Ali Naseh, Jaechul Roh, Amir Houmansadr

**Abstract**: Diffusion-based models, such as the Stable Diffusion model, have revolutionized text-to-image synthesis with their ability to produce high-quality, high-resolution images. These advancements have prompted significant progress in image generation and editing tasks. However, these models also raise concerns due to their tendency to memorize and potentially replicate exact training samples, posing privacy risks and enabling adversarial attacks. Duplication in training datasets is recognized as a major factor contributing to memorization, and various forms of memorization have been studied so far. This paper focuses on two distinct and underexplored types of duplication that lead to replication during inference in diffusion-based models, particularly in the Stable Diffusion model. We delve into these lesser-studied duplication phenomena and their implications through two case studies, aiming to contribute to the safer and more responsible use of generative models in various applications.

摘要: 基于扩散的模型，如稳定扩散模型，以其生成高质量、高分辨率图像的能力，使文本到图像的合成发生了革命性的变化。这些进步推动了图像生成和编辑任务的重大进步。然而，这些模型也引起了人们的担忧，因为它们倾向于记忆并可能复制准确的训练样本，这会带来隐私风险，并使对抗性攻击成为可能。训练数据集的重复被认为是导致记忆的主要因素，到目前为止，人们已经研究了各种形式的记忆。在基于扩散的模型中，特别是在稳定扩散模型中，本文重点研究了两种不同的、未被充分研究的复制类型，它们在基于扩散的模型中的推理过程中导致复制。我们通过两个案例研究来深入研究这些较少研究的复制现象及其影响，旨在有助于在各种应用中更安全和更负责任地使用生成模型。



## **2. PyraTrans: Attention-Enriched Pyramid Transformer for Malicious URL Detection**

金字塔：用于恶意URL检测的高关注度金字塔转换器 cs.CR

12 pages, 7 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.00508v2) [paper-pdf](http://arxiv.org/pdf/2312.00508v2)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Although advancements in machine learning have driven the development of malicious URL detection technology, current techniques still face significant challenges in their capacity to generalize and their resilience against evolving threats. In this paper, we propose PyraTrans, a novel method that integrates pretrained Transformers with pyramid feature learning to detect malicious URL. PyraTrans utilizes a pretrained CharBERT as its foundation and is augmented with three interconnected feature modules: 1) Encoder Feature Extraction, extracting multi-order feature matrices from each CharBERT encoder layer; 2) Multi-Scale Feature Learning, capturing local contextual insights at various scales and aggregating information across encoder layers; and 3) Spatial Pyramid Attention, focusing on regional-level attention to emphasize areas rich in expressive information. The proposed approach addresses the limitations of the Transformer in local feature learning and regional relational awareness, which are vital for capturing URL-specific word patterns, character combinations, or structural anomalies. In several challenging experimental scenarios, the proposed method has shown significant improvements in accuracy, generalization, and robustness in malicious URL detection. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios, and exceeded the best baseline result by 14.13% in accuracy in adversarial attack scenarios. Additionally, we conduct a case study where our method accurately identifies all 30 active malicious web pages, whereas two pior SOTA methods miss 4 and 7 malicious web pages respectively. Codes and data are available at:https://github.com/Alixyvtte/PyraTrans.

摘要: 尽管机器学习的进步推动了恶意URL检测技术的发展，但当前的技术在泛化能力和对不断变化的威胁的弹性方面仍然面临着巨大的挑战。在本文中，我们提出了一种将预先训练的变换和金字塔特征学习相结合的检测恶意URL的新方法--PyraTrans。金字塔利用预先训练的CharBERT作为其基础，并增加了三个相互关联的特征模块：1)编码器特征提取，从每个CharBERT编码层提取多阶特征矩阵；2)多尺度特征学习，捕获不同尺度上的局部上下文洞察力，并聚合编码层之间的信息；以及3)空间金字塔关注，专注于区域级别的关注，以强调具有丰富表现力信息的区域。提出的方法解决了Transformer在本地特征学习和区域关系感知方面的局限性，这对于捕获特定于URL的单词模式、字符组合或结构异常至关重要。在几个具有挑战性的实验场景中，所提出的方法在恶意URL检测的准确性、泛化和稳健性方面显示出显著的改进。例如，在班级不平衡场景下，它的F1得分峰值提高了40%，在对抗性攻击场景中，它的准确率超过了最佳基线结果14.13%。此外，我们还进行了一个案例研究，其中我们的方法准确地识别了所有30个活跃的恶意网页，而两种高级SOTA方法分别漏掉了4个和7个恶意网页。代码和数据可在以下网址获得：https://github.com/Alixyvtte/PyraTrans.



## **3. Defense Against Adversarial Attacks using Convolutional Auto-Encoders**

利用卷积自动编码器防御对抗性攻击 cs.CV

9 pages, 6 figures, 3 tables

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03520v1) [paper-pdf](http://arxiv.org/pdf/2312.03520v1)

**Authors**: Shreyasi Mandal

**Abstract**: Deep learning models, while achieving state-of-the-art performance on many tasks, are susceptible to adversarial attacks that exploit inherent vulnerabilities in their architectures. Adversarial attacks manipulate the input data with imperceptible perturbations, causing the model to misclassify the data or produce erroneous outputs. This work is based on enhancing the robustness of targeted classifier models against adversarial attacks. To achieve this, an convolutional autoencoder-based approach is employed that effectively counters adversarial perturbations introduced to the input images. By generating images closely resembling the input images, the proposed methodology aims to restore the model's accuracy.

摘要: 深度学习模型虽然在许多任务上实现了最先进的性能，但很容易受到利用其体系结构中固有漏洞的对手攻击。对抗性攻击以不可察觉的扰动操纵输入数据，导致模型对数据进行错误分类或产生错误的输出。这项工作的基础是增强目标分类器模型对对手攻击的稳健性。为了实现这一点，采用了一种基于卷积自动编码器的方法，该方法有效地对抗了引入到输入图像的对抗性扰动。通过生成与输入图像非常相似的图像，所提出的方法旨在恢复模型的准确性。



## **4. Quantum-secured single-pixel imaging under general spoofing attack**

一般欺骗攻击下的量子安全单像素成像 quant-ph

9 pages, 6 figures

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03465v1) [paper-pdf](http://arxiv.org/pdf/2312.03465v1)

**Authors**: Jaesung Heo, Taek Jeong, Nam Hun Park, Yonggi Jo

**Abstract**: In this paper, we introduce a quantum-secured single-pixel imaging (QS-SPI) technique designed to withstand spoofing attacks, wherein adversaries attempt to deceive imaging systems with fake signals. Unlike previous quantum-secured protocols that impose a threshold error rate limiting their operation, even with the existence of true signals, our approach not only identifies spoofing attacks but also facilitates the reconstruction of a true image. Our method involves the analysis of a specific mode correlation of a photon-pair, which is independent of the mode used for image construction, to check security. Through this analysis, we can identify both the targeted image region by the attack and the type of spoofing attack, enabling reconstruction of the true image. A proof-of-principle demonstration employing polarization-correlation of a photon-pair is provided, showcasing successful image reconstruction even under the condition of spoofing signals 2000 times stronger than the true signals. We expect our approach to be applied to quantum-secured signal processing such as quantum target detection or ranging.

摘要: 在这篇文章中，我们介绍了一种量子安全单像素成像(QS-SPI)技术，旨在抵抗欺骗攻击，其中攻击者试图用虚假信号欺骗成像系统。与以前的量子安全协议不同，即使在真实信号存在的情况下，我们的方法也会施加阈值错误率来限制它们的操作，我们的方法不仅可以识别欺骗攻击，还可以帮助重建真实图像。我们的方法包括分析与用于图像构建的模式无关的光子对的特定模式相关性，以检查安全性。通过这种分析，我们可以根据攻击和欺骗攻击的类型来识别目标图像区域，从而能够重建出真实的图像。提供了使用光子对的偏振相关的原理证明演示，展示了即使在欺骗信号比真实信号强2000倍的情况下也成功地重建图像。我们希望将我们的方法应用于量子安全信号处理，如量子目标检测或测距。



## **5. Synthesizing Physical Backdoor Datasets: An Automated Framework Leveraging Deep Generative Models**

综合物理后门数据集：利用深度生成模型的自动化框架 cs.CR

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03419v1) [paper-pdf](http://arxiv.org/pdf/2312.03419v1)

**Authors**: Sze Jue Yang, Chinh D. La, Quang H. Nguyen, Eugene Bagdasaryan, Kok-Seng Wong, Anh Tuan Tran, Chee Seng Chan, Khoa D. Doan

**Abstract**: Backdoor attacks, representing an emerging threat to the integrity of deep neural networks, have garnered significant attention due to their ability to compromise deep learning systems clandestinely. While numerous backdoor attacks occur within the digital realm, their practical implementation in real-world prediction systems remains limited and vulnerable to disturbances in the physical world. Consequently, this limitation has given rise to the development of physical backdoor attacks, where trigger objects manifest as physical entities within the real world. However, creating the requisite dataset to train or evaluate a physical backdoor model is a daunting task, limiting the backdoor researchers and practitioners from studying such physical attack scenarios. This paper unleashes a recipe that empowers backdoor researchers to effortlessly create a malicious, physical backdoor dataset based on advances in generative modeling. Particularly, this recipe involves 3 automatic modules: suggesting the suitable physical triggers, generating the poisoned candidate samples (either by synthesizing new samples or editing existing clean samples), and finally refining for the most plausible ones. As such, it effectively mitigates the perceived complexity associated with creating a physical backdoor dataset, transforming it from a daunting task into an attainable objective. Extensive experiment results show that datasets created by our "recipe" enable adversaries to achieve an impressive attack success rate on real physical world data and exhibit similar properties compared to previous physical backdoor attack studies. This paper offers researchers a valuable toolkit for studies of physical backdoors, all within the confines of their laboratories.

摘要: 后门攻击对深度神经网络的完整性构成了新的威胁，由于它们能够秘密地危害深度学习系统，因此引起了极大的关注。虽然在数字领域内发生了许多后门攻击，但它们在现实世界预测系统中的实际实施仍然有限，容易受到物理世界的干扰。因此，这种限制导致了物理后门攻击的发展，其中触发器对象在真实世界中表现为物理实体。然而，创建必要的数据集来训练或评估物理后门模型是一项艰巨的任务，限制了后门研究人员和实践者研究此类物理攻击场景。这篇文章揭示了一个配方，它使后门研究人员能够基于生成性建模的进步，毫不费力地创建恶意的物理后门数据集。特别是，这个配方涉及三个自动模块：建议合适的物理触发器，生成中毒的候选样本(通过合成新样本或编辑现有的干净样本)，最后提炼出最可信的样本。因此，它有效地减轻了与创建物理后门数据集相关的感知复杂性，将其从令人望而生畏的任务转变为可实现的目标。大量的实验结果表明，由我们的“配方”创建的数据集使攻击者能够在真实的物理世界数据上获得令人印象深刻的攻击成功率，并显示出与之前的物理后门攻击研究类似的特性。这篇论文为研究人员提供了一个宝贵的工具包，用于研究物理后门，所有这些都在他们的实验室范围内。



## **6. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

SAIF：稀疏对抗性和不可察觉攻击框架 cs.CV

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2212.07495v2) [paper-pdf](http://arxiv.org/pdf/2212.07495v2)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.

摘要: 对抗性攻击通过干扰输入信号来阻碍神经网络的决策能力。例如，将计算的小失真添加到图像可以欺骗训练有素的图像分类网络。在这项工作中，我们提出了一种新的攻击技术，称为稀疏对抗性和可解释攻击框架(SAIF)。具体地说，我们设计了在少量像素处包含低幅度扰动的不可察觉攻击，并利用这些稀疏攻击来揭示分类器的脆弱性。我们使用Frank-Wolfe(条件梯度)算法来同时优化有界模和稀疏性的攻击扰动，并且具有$O(1/\Sqrt{T})$收敛。实验结果表明，该算法能够计算高度不可察觉和可解释的敌意实例，并且在ImageNet数据集上的性能优于最新的稀疏攻击方法。



## **7. Privacy-Preserving Task-Oriented Semantic Communications Against Model Inversion Attacks**

基于隐私保护的面向任务的语义通信抗模型反转攻击 cs.IT

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03252v1) [paper-pdf](http://arxiv.org/pdf/2312.03252v1)

**Authors**: Yanhu Wang, Shuaishuai Guo, Yiqin Deng, Haixia Zhang, Yuguang Fang

**Abstract**: Semantic communication has been identified as a core technology for the sixth generation (6G) of wireless networks. Recently, task-oriented semantic communications have been proposed for low-latency inference with limited bandwidth. Although transmitting only task-related information does protect a certain level of user privacy, adversaries could apply model inversion techniques to reconstruct the raw data or extract useful information, thereby infringing on users' privacy. To mitigate privacy infringement, this paper proposes an information bottleneck and adversarial learning (IBAL) approach to protect users' privacy against model inversion attacks. Specifically, we extract task-relevant features from the input based on the information bottleneck (IB) theory. To overcome the difficulty in calculating the mutual information in high-dimensional space, we derive a variational upper bound to estimate the true mutual information. To prevent data reconstruction from task-related features by adversaries, we leverage adversarial learning to train encoder to fool adversaries by maximizing reconstruction distortion. Furthermore, considering the impact of channel variations on privacy-utility trade-off and the difficulty in manually tuning the weights of each loss, we propose an adaptive weight adjustment method. Numerical results demonstrate that the proposed approaches can effectively protect privacy without significantly affecting task performance and achieve better privacy-utility trade-offs than baseline methods.

摘要: 语义通信已被确定为第六代（6G）无线网络的核心技术。最近，面向任务的语义通信已经被提出用于具有有限带宽的低延迟推理。虽然只传输与任务相关的信息确实可以保护一定程度的用户隐私，但攻击者可以应用模型反演技术来重建原始数据或提取有用的信息，从而侵犯用户的隐私。为了减少隐私侵犯，本文提出了一种信息瓶颈和对抗学习（IBAL）的方法来保护用户的隐私免受模型反演攻击。具体来说，我们提取任务相关的功能，从输入的基础上信息瓶颈（IB）理论。为了克服高维空间中互信息计算的困难，我们推导了一个变分上界来估计真实的互信息。为了防止对手从与任务相关的特征中重建数据，我们利用对抗性学习来训练编码器，通过最大化重建失真来欺骗对手。此外，考虑到信道变化对隐私效用权衡的影响以及手动调整每个损失的权重的困难，我们提出了一种自适应权重调整方法。数值结果表明，所提出的方法可以有效地保护隐私，而不会显着影响任务的性能，并实现更好的隐私效用权衡比基线方法。



## **8. A Simple Framework to Enhance the Adversarial Robustness of Deep Learning-based Intrusion Detection System**

一种增强基于深度学习的入侵检测系统抗攻击能力的简单框架 cs.CR

Accepted by Computers & Security

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2312.03245v1) [paper-pdf](http://arxiv.org/pdf/2312.03245v1)

**Authors**: Xinwei Yuan, Shu Han, Wei Huang, Hongliang Ye, Xianglong Kong, Fan Zhang

**Abstract**: Deep learning based intrusion detection systems (DL-based IDS) have emerged as one of the best choices for providing security solutions against various network intrusion attacks. However, due to the emergence and development of adversarial deep learning technologies, it becomes challenging for the adoption of DL models into IDS. In this paper, we propose a novel IDS architecture that can enhance the robustness of IDS against adversarial attacks by combining conventional machine learning (ML) models and Deep Learning models. The proposed DLL-IDS consists of three components: DL-based IDS, adversarial example (AE) detector, and ML-based IDS. We first develop a novel AE detector based on the local intrinsic dimensionality (LID). Then, we exploit the low attack transferability between DL models and ML models to find a robust ML model that can assist us in determining the maliciousness of AEs. If the input traffic is detected as an AE, the ML-based IDS will predict the maliciousness of input traffic, otherwise the DL-based IDS will work for the prediction. The fusion mechanism can leverage the high prediction accuracy of DL models and low attack transferability between DL models and ML models to improve the robustness of the whole system. In our experiments, we observe a significant improvement in the prediction performance of the IDS when subjected to adversarial attack, achieving high accuracy with low resource consumption.

摘要: 基于深度学习的入侵检测系统已经成为针对各种网络入侵攻击提供安全解决方案的最佳选择之一。然而，随着对抗性深度学习技术的出现和发展，将动态链式学习模型应用到入侵检测系统中变得越来越困难。本文将传统的机器学习模型和深度学习模型相结合，提出了一种新的入侵检测体系结构，能够增强入侵检测系统对敌意攻击的健壮性。提出的动态链接库-入侵检测系统由三部分组成：基于动态链接库的入侵检测系统、对抗性实例(AE)检测器和基于ML的入侵检测系统。我们首先提出了一种基于局部内禀维度(LID)的新型声发射检测器。然后，我们利用DL模型和ML模型之间的低攻击传递性来找到一个健壮的ML模型，该模型可以帮助我们确定攻击实体的恶意程度。如果输入流量被检测为AE，则基于ML的入侵检测系统将预测输入流量的恶意程度，否则，基于DL的入侵检测系统将进行预测。该融合机制利用了DL模型的高预测精度和DL模型与ML模型之间较低的攻击传递性，提高了整个系统的鲁棒性。在我们的实验中，我们观察到入侵检测系统在遭受敌意攻击时预测性能有了显著的提高，在低资源消耗的情况下达到了高精度。



## **9. Model-tuning Via Prompts Makes NLP Models Adversarially Robust**

通过自适应调整模型使NLP模型具有逆向鲁棒性 cs.CL

Accepted to the EMNLP 2023 Conference

**SubmitDate**: 2023-12-06    [abs](http://arxiv.org/abs/2303.07320v2) [paper-pdf](http://arxiv.org/pdf/2303.07320v2)

**Authors**: Mrigank Raman, Pratyush Maini, J. Zico Kolter, Zachary C. Lipton, Danish Pruthi

**Abstract**: In recent years, NLP practitioners have converged on the following practice: (i) import an off-the-shelf pretrained (masked) language model; (ii) append a multilayer perceptron atop the CLS token's hidden representation (with randomly initialized weights); and (iii) fine-tune the entire model on a downstream task (MLP-FT). This procedure has produced massive gains on standard NLP benchmarks, but these models remain brittle, even to mild adversarial perturbations. In this work, we demonstrate surprising gains in adversarial robustness enjoyed by Model-tuning Via Prompts (MVP), an alternative method of adapting to downstream tasks. Rather than appending an MLP head to make output prediction, MVP appends a prompt template to the input, and makes prediction via text infilling/completion. Across 5 NLP datasets, 4 adversarial attacks, and 3 different models, MVP improves performance against adversarial substitutions by an average of 8% over standard methods and even outperforms adversarial training-based state-of-art defenses by 3.5%. By combining MVP with adversarial training, we achieve further improvements in adversarial robustness while maintaining performance on unperturbed examples. Finally, we conduct ablations to investigate the mechanism underlying these gains. Notably, we find that the main causes of vulnerability of MLP-FT can be attributed to the misalignment between pre-training and fine-tuning tasks, and the randomly initialized MLP parameters.

摘要: 近年来，NLP实践者在以下实践上趋同：(I)引入现成的预训练(掩蔽)语言模型；(Ii)在CLS令牌的隐藏表示上附加多层感知器(具有随机初始化权重)；以及(Iii)在下游任务(MLP-FT)上微调整个模型。这一过程在标准NLP基准上产生了巨大的收益，但这些模型仍然脆弱，即使是轻微的对抗性扰动。在这项工作中，我们展示了通过提示调整模型(MVP)在对抗健壮性方面的惊人收益，这是一种适应下游任务的替代方法。MVP不是附加MLP头来进行输出预测，而是将提示模板附加到输入，并通过文本填充/完成进行预测。在5个NLP数据集、4个对抗性攻击和3个不同的模型中，MVP在对抗对抗性替换时的性能比标准方法平均提高了8%，甚至比基于对抗性训练的最新防御性能提高了3.5%。通过将MVP与对抗性训练相结合，我们在保持在未受干扰的示例上的性能的同时，进一步提高了对抗性健壮性。最后，我们进行消融来研究这些收益背后的机制。值得注意的是，我们发现MLP-FT脆弱性的主要原因可以归因于预先训练和微调任务之间的不匹配，以及随机初始化的MLP参数。



## **10. Effective Backdoor Mitigation Depends on the Pre-training Objective**

有效的后门缓解取决于培训前的目标 cs.LG

Accepted for oral presentation at BUGS workshop @ NeurIPS 2023  (https://neurips2023-bugs.github.io/)

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.14948v3) [paper-pdf](http://arxiv.org/pdf/2311.14948v3)

**Authors**: Sahil Verma, Gantavya Bhatt, Avi Schwarzschild, Soumye Singhal, Arnav Mohanty Das, Chirag Shah, John P Dickerson, Jeff Bilmes

**Abstract**: Despite the advanced capabilities of contemporary machine learning (ML) models, they remain vulnerable to adversarial and backdoor attacks. This vulnerability is particularly concerning in real-world deployments, where compromised models may exhibit unpredictable behavior in critical scenarios. Such risks are heightened by the prevalent practice of collecting massive, internet-sourced datasets for pre-training multimodal models, as these datasets may harbor backdoors. Various techniques have been proposed to mitigate the effects of backdooring in these models such as CleanCLIP which is the current state-of-the-art approach. In this work, we demonstrate that the efficacy of CleanCLIP in mitigating backdoors is highly dependent on the particular objective used during model pre-training. We observe that stronger pre-training objectives correlate with harder to remove backdoors behaviors. We show this by training multimodal models on two large datasets consisting of 3 million (CC3M) and 6 million (CC6M) datapoints, under various pre-training objectives, followed by poison removal using CleanCLIP. We find that CleanCLIP is ineffective when stronger pre-training objectives are used, even with extensive hyperparameter tuning. Our findings underscore critical considerations for ML practitioners who pre-train models using large-scale web-curated data and are concerned about potential backdoor threats. Notably, our results suggest that simpler pre-training objectives are more amenable to effective backdoor removal. This insight is pivotal for practitioners seeking to balance the trade-offs between using stronger pre-training objectives and security against backdoor attacks.

摘要: 尽管当代机器学习(ML)模型具有先进的能力，但它们仍然容易受到对手和后门攻击。此漏洞在实际部署中尤其令人担忧，在实际部署中，受危害的模型可能会在关键情况下表现出不可预测的行为。为训练前的多模式模型收集来自互联网的海量数据集的普遍做法加剧了这种风险，因为这些数据集可能有后门。已经提出了各种技术来减轻这些模型中回溯的影响，例如CleanCLIP，这是当前最先进的方法。在这项工作中，我们证明了CleanCLIP在缓解后门方面的有效性高度依赖于在模型预培训期间使用的特定目标。我们观察到，较强的培训前目标与较难消除后门行为相关。我们通过在两个由300万(CC3M)和600万(CC6M)数据点组成的大型数据集上训练多模模型，在不同的预训练目标下，然后使用CleanCLIP去除毒物来证明这一点。我们发现，当使用更强的预培训目标时，即使进行了广泛的超参数调整，CleanCLIP也是无效的。我们的发现强调了ML从业者的关键考虑，他们使用大规模的网络管理数据对模型进行预培训，并担心潜在的后门威胁。值得注意的是，我们的结果表明，简单的预培训目标更容易有效地移除后门。对于寻求在使用更强的预培训目标和针对后门攻击的安全性之间进行权衡的从业者来说，这一见解至关重要。



## **11. Beyond Detection: Unveiling Fairness Vulnerabilities in Abusive Language Models**

超越检测：揭开辱骂语言模型中的公平漏洞 cs.CL

Under review

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2311.09428v2) [paper-pdf](http://arxiv.org/pdf/2311.09428v2)

**Authors**: Yueqing Liang, Lu Cheng, Ali Payani, Kai Shu

**Abstract**: This work investigates the potential of undermining both fairness and detection performance in abusive language detection. In a dynamic and complex digital world, it is crucial to investigate the vulnerabilities of these detection models to adversarial fairness attacks to improve their fairness robustness. We propose a simple yet effective framework FABLE that leverages backdoor attacks as they allow targeted control over the fairness and detection performance. FABLE explores three types of trigger designs (i.e., rare, artificial, and natural triggers) and novel sampling strategies. Specifically, the adversary can inject triggers into samples in the minority group with the favored outcome (i.e., "non-abusive") and flip their labels to the unfavored outcome, i.e., "abusive". Experiments on benchmark datasets demonstrate the effectiveness of FABLE attacking fairness and utility in abusive language detection.

摘要: 这项工作调查了在辱骂语言检测中同时破坏公平性和检测性能的可能性。在动态和复杂的数字世界中，研究这些检测模型对敌意公平攻击的脆弱性，以提高其公平性健壮性是至关重要的。我们提出了一个简单而有效的框架寓言，它利用了后门攻击，因为它们允许对公平性和检测性能进行有针对性的控制。Fable探索了三种类型的触发器设计(即罕见的、人工的和自然的触发器)和新颖的抽样策略。具体地说，敌手可以将触发器注入具有有利结果的少数群体中的样本(即，“非虐待”)，并将其标签反转到不利结果，即，“虐待”。在基准数据集上的实验证明了寓言攻击的有效性、公平性和实用性。



## **12. ScAR: Scaling Adversarial Robustness for LiDAR Object Detection**

SCAR：激光雷达目标检测的对抗性缩放算法 cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03085v1) [paper-pdf](http://arxiv.org/pdf/2312.03085v1)

**Authors**: Xiaohu Lu, Hayder Radha

**Abstract**: The adversarial robustness of a model is its ability to resist adversarial attacks in the form of small perturbations to input data. Universal adversarial attack methods such as Fast Sign Gradient Method (FSGM) and Projected Gradient Descend (PGD) are popular for LiDAR object detection, but they are often deficient compared to task-specific adversarial attacks. Additionally, these universal methods typically require unrestricted access to the model's information, which is difficult to obtain in real-world applications. To address these limitations, we present a black-box Scaling Adversarial Robustness (ScAR) method for LiDAR object detection. By analyzing the statistical characteristics of 3D object detection datasets such as KITTI, Waymo, and nuScenes, we have found that the model's prediction is sensitive to scaling of 3D instances. We propose three black-box scaling adversarial attack methods based on the available information: model-aware attack, distribution-aware attack, and blind attack. We also introduce a strategy for generating scaling adversarial examples to improve the model's robustness against these three scaling adversarial attacks. Comparison with other methods on public datasets under different 3D object detection architectures demonstrates the effectiveness of our proposed method.

摘要: 模型的对抗性鲁棒性是指它抵抗以输入数据的小扰动形式的对抗性攻击的能力。通用对抗攻击方法，如快速符号梯度方法（FSGM）和投影梯度分解（PGD），在LiDAR目标检测中很受欢迎，但与特定任务的对抗攻击相比，它们往往存在不足。此外，这些通用方法通常需要不受限制地访问模型的信息，这在现实世界的应用程序中很难获得。为了解决这些限制，我们提出了一种用于LiDAR对象检测的黑盒缩放对抗鲁棒性（ScAR）方法。通过分析KITTI、Waymo和nuScenes等3D对象检测数据集的统计特征，我们发现模型的预测对3D实例的缩放敏感。提出了三种基于可用信息的黑盒缩放对抗攻击方法：模型感知攻击、分布感知攻击和盲攻击。我们还介绍了一种生成缩放对抗性示例的策略，以提高模型对这三种缩放对抗性攻击的鲁棒性。在不同的3D目标检测架构下，与其他方法在公共数据集上的比较证明了我们所提出的方法的有效性。



## **13. Realistic Scatterer Based Adversarial Attacks on SAR Image Classifiers**

基于真实散射体的SAR图像分类器对抗攻击 cs.CV

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02912v1) [paper-pdf](http://arxiv.org/pdf/2312.02912v1)

**Authors**: Tian Ye, Rajgopal Kannan, Viktor Prasanna, Carl Busart, Lance Kaplan

**Abstract**: Adversarial attacks have highlighted the vulnerability of classifiers based on machine learning for Synthetic Aperture Radar (SAR) Automatic Target Recognition (ATR) tasks. An adversarial attack perturbs SAR images of on-ground targets such that the classifiers are misled into making incorrect predictions. However, many existing attacking techniques rely on arbitrary manipulation of SAR images while overlooking the feasibility of executing the attacks on real-world SAR imagery. Instead, adversarial attacks should be able to be implemented by physical actions, for example, placing additional false objects as scatterers around the on-ground target to perturb the SAR image and fool the SAR ATR.   In this paper, we propose the On-Target Scatterer Attack (OTSA), a scatterer-based physical adversarial attack. To ensure the feasibility of its physical execution, we enforce a constraint on the positioning of the scatterers. Specifically, we restrict the scatterers to be placed only on the target instead of in the shadow regions or the background. To achieve this, we introduce a positioning score based on Gaussian kernels and formulate an optimization problem for our OTSA attack. Using a gradient ascent method to solve the optimization problem, the OTSA can generate a vector of parameters describing the positions, shapes, sizes and amplitudes of the scatterers to guide the physical execution of the attack that will mislead SAR image classifiers. The experimental results show that our attack obtains significantly higher success rates under the positioning constraint compared with the existing method.

摘要: 对抗性攻击突出了基于机器学习的分类器在合成孔径雷达(SAR)自动目标识别(ATR)任务中的脆弱性。对抗性攻击会干扰地面目标的合成孔径雷达图像，从而误导分类器做出错误的预测。然而，现有的许多攻击技术依赖于对SAR图像的任意篡改，而忽略了对现实世界的SAR图像执行攻击的可行性。相反，对抗性攻击应该能够通过物理行动来实施，例如，在地面目标周围放置额外的虚假目标作为散射体，以扰乱SAR图像并愚弄SAR ATR。本文提出了一种基于散射体的物理对抗攻击--目标上散射体攻击(OTSA)。为了确保其物理执行的可行性，我们对散射体的位置施加了约束。具体地说，我们将散射体限制为只放置在目标上，而不是放置在阴影区域或背景中。为了实现这一点，我们引入了一个基于高斯核的定位分数，并为我们的OTSA攻击建立了一个优化问题。利用梯度上升法求解优化问题，OTSA可以生成描述散射体位置、形状、大小和幅度的参数向量，以指导攻击的物理执行，从而误导SAR图像分类器。实验结果表明，与现有的攻击方法相比，在位置约束下，我们的攻击获得了更高的成功率。



## **14. Scaling Laws for Adversarial Attacks on Language Model Activations**

语言模型激活对抗攻击的标度律 cs.LG

15 pages, 9 figures

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.02780v1) [paper-pdf](http://arxiv.org/pdf/2312.02780v1)

**Authors**: Stanislav Fort

**Abstract**: We explore a class of adversarial attacks targeting the activations of language models. By manipulating a relatively small subset of model activations, $a$, we demonstrate the ability to control the exact prediction of a significant number (in some cases up to 1000) of subsequent tokens $t$. We empirically verify a scaling law where the maximum number of target tokens $t_\mathrm{max}$ predicted depends linearly on the number of tokens $a$ whose activations the attacker controls as $t_\mathrm{max} = \kappa a$. We find that the number of bits of control in the input space needed to control a single bit in the output space (what we call attack resistance $\chi$) is remarkably constant between $\approx 16$ and $\approx 25$ over 2 orders of magnitude of model sizes for different language models. Compared to attacks on tokens, attacks on activations are predictably much stronger, however, we identify a surprising regularity where one bit of input steered either via activations or via tokens is able to exert control over a similar amount of output bits. This gives support for the hypothesis that adversarial attacks are a consequence of dimensionality mismatch between the input and output spaces. A practical implication of the ease of attacking language model activations instead of tokens is for multi-modal and selected retrieval models, where additional data sources are added as activations directly, sidestepping the tokenized input. This opens up a new, broad attack surface. By using language models as a controllable test-bed to study adversarial attacks, we were able to experiment with input-output dimensions that are inaccessible in computer vision, especially where the output dimension dominates.

摘要: 我们探讨了一类针对语言模型激活的对抗性攻击。通过操纵一个相对较小的模型激活子集，$a$，我们展示了控制大量（在某些情况下高达1000）后续令牌$t$的准确预测的能力。我们根据经验验证了一个缩放定律，其中预测的目标令牌的最大数量$t_\mathrm{max}$线性依赖于攻击者控制其激活的令牌$a$的数量$t_\mathrm{max} = \kappa a$。我们发现，在输入空间中的控制位的数量需要控制一个单一的位在输出空间（我们称之为攻击阻力$\chi$）是显着恒定的$\approximat 16$和$\approximat 25$之间的2个数量级的模型大小为不同的语言模型。与对令牌的攻击相比，对激活的攻击可以预见要强得多，然而，我们发现了一个令人惊讶的规律，即通过激活或令牌引导的一位输入能够控制类似数量的输出位。这支持了对抗性攻击是输入和输出空间之间维度不匹配的结果的假设。攻击语言模型激活而不是标记的容易性的实际含义是多模态和选定的检索模型，其中直接添加额外的数据源作为激活，避开标记化的输入。这开辟了一个新的，广泛的攻击面。通过使用语言模型作为研究对抗性攻击的可控测试平台，我们能够对计算机视觉中无法访问的输入-输出维度进行实验，特别是在输出维度占主导地位的情况下。



## **15. Generating Visually Realistic Adversarial Patch**

生成视觉逼真的对抗性补丁 cs.CV

14 pages

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2312.03030v1) [paper-pdf](http://arxiv.org/pdf/2312.03030v1)

**Authors**: Xiaosen Wang, Kunyu Wang

**Abstract**: Deep neural networks (DNNs) are vulnerable to various types of adversarial examples, bringing huge threats to security-critical applications. Among these, adversarial patches have drawn increasing attention due to their good applicability to fool DNNs in the physical world. However, existing works often generate patches with meaningless noise or patterns, making it conspicuous to humans. To address this issue, we explore how to generate visually realistic adversarial patches to fool DNNs. Firstly, we analyze that a high-quality adversarial patch should be realistic, position irrelevant, and printable to be deployed in the physical world. Based on this analysis, we propose an effective attack called VRAP, to generate visually realistic adversarial patches. Specifically, VRAP constrains the patch in the neighborhood of a real image to ensure the visual reality, optimizes the patch at the poorest position for position irrelevance, and adopts Total Variance loss as well as gamma transformation to make the generated patch printable without losing information. Empirical evaluations on the ImageNet dataset demonstrate that the proposed VRAP exhibits outstanding attack performance in the digital world. Moreover, the generated adversarial patches can be disguised as the scrawl or logo in the physical world to fool the deep models without being detected, bringing significant threats to DNNs-enabled applications.

摘要: 深度神经网络(DNN)容易受到各种类型的敌意例子的攻击，给安全关键应用带来了巨大的威胁。其中，敌意补丁由于在物理世界中很好地适用于愚弄DNN而引起了越来越多的关注。然而，现有的作品经常产生毫无意义的噪声或图案的斑块，使其对人类来说是显眼的。为了解决这个问题，我们探索了如何生成视觉上逼真的敌意补丁来愚弄DNN。首先，我们分析了一个高质量的对抗性补丁应该是逼真的、与位置无关的、可打印的，以便在物理世界中部署。在此基础上，我们提出了一种有效的攻击方法，称为VRAP，用于生成视觉上逼真的对抗性补丁。具体地说，VRAP将面片约束在真实图像的邻域内以确保视觉真实感，对位置不相关的最差位置的面片进行优化，并采用全方差损失和Gamma变换使生成的面片在不丢失信息的情况下可打印。在ImageNet数据集上的实验评估表明，所提出的VRAP在数字世界中具有优异的攻击性能。此外，生成的恶意补丁可以伪装成物理世界中的涂鸦或徽标，从而在不被发现的情况下欺骗深层模型，给支持DNNS的应用程序带来重大威胁。



## **16. Byzantine-Robust Distributed Online Learning: Taming Adversarial Participants in An Adversarial Environment**

拜占庭-稳健的分布式在线学习：在对抗性环境中驯服对抗性参与者 cs.LG

**SubmitDate**: 2023-12-05    [abs](http://arxiv.org/abs/2307.07980v3) [paper-pdf](http://arxiv.org/pdf/2307.07980v3)

**Authors**: Xingrong Dong, Zhaoxian Wu, Qing Ling, Zhi Tian

**Abstract**: This paper studies distributed online learning under Byzantine attacks. The performance of an online learning algorithm is often characterized by (adversarial) regret, which evaluates the quality of one-step-ahead decision-making when an environment provides adversarial losses, and a sublinear bound is preferred. But we prove that, even with a class of state-of-the-art robust aggregation rules, in an adversarial environment and in the presence of Byzantine participants, distributed online gradient descent can only achieve a linear adversarial regret bound, which is tight. This is the inevitable consequence of Byzantine attacks, even though we can control the constant of the linear adversarial regret to a reasonable level. Interestingly, when the environment is not fully adversarial so that the losses of the honest participants are i.i.d. (independent and identically distributed), we show that sublinear stochastic regret, in contrast to the aforementioned adversarial regret, is possible. We develop a Byzantine-robust distributed online momentum algorithm to attain such a sublinear stochastic regret bound. Extensive numerical experiments corroborate our theoretical analysis.

摘要: 本文研究拜占庭攻击下的分布式在线学习。在线学习算法的性能通常以(对抗性)后悔为特征，当环境提供对抗性损失时，该算法评估领先一步的决策的质量，并且次线性界是首选的。但我们证明了，即使在一类最新的稳健聚集规则下，在对抗性环境下，在拜占庭参与者在场的情况下，分布式在线梯度下降也只能达到线性对抗性遗憾界，这是紧的。这是拜占庭攻击的必然结果，即使我们可以将线性对抗性后悔的常量控制在合理的水平。有趣的是，当环境不是完全对抗性的时候，诚实的参与者的损失是I.I.D.(独立同分布)，我们证明了与前面提到的对抗性后悔相比，次线性随机后悔是可能的。我们开发了一种拜占庭稳健的分布式在线动量算法来获得这样的次线性随机后悔界。大量的数值实验证实了我们的理论分析。



## **17. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

InstructTA：针对大型视觉语言模型的指令调整的定向攻击 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01886v1) [paper-pdf](http://arxiv.org/pdf/2312.01886v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。本文提出了一种新颖实用的灰盒攻击方案，即攻击者只能访问受害者LVLM的可视编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大型语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(InstructTA)，对具有高可转移性的LVLMS进行定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高可转移性，我们用转译自LLM的指令扩充了指令$\boldSymbol{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。



## **18. Two-stage optimized unified adversarial patch for attacking visible-infrared cross-modal detectors in the physical world**

物理世界中攻击可见-红外交叉模态探测器的两阶段优化统一对抗补丁 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01789v1) [paper-pdf](http://arxiv.org/pdf/2312.01789v1)

**Authors**: Chengyin Hu, Weiwen Shi

**Abstract**: Currently, many studies have addressed security concerns related to visible and infrared detectors independently. In practical scenarios, utilizing cross-modal detectors for tasks proves more reliable than relying on single-modal detectors. Despite this, there is a lack of comprehensive security evaluations for cross-modal detectors. While existing research has explored the feasibility of attacks against cross-modal detectors, the implementation of a robust attack remains unaddressed. This work introduces the Two-stage Optimized Unified Adversarial Patch (TOUAP) designed for performing attacks against visible-infrared cross-modal detectors in real-world, black-box settings. The TOUAP employs a two-stage optimization process: firstly, PSO optimizes an irregular polygonal infrared patch to attack the infrared detector; secondly, the color QR code is optimized, and the shape information of the infrared patch from the first stage is used as a mask. The resulting irregular polygon visible modal patch executes an attack on the visible detector. Through extensive experiments conducted in both digital and physical environments, we validate the effectiveness and robustness of the proposed method. As the TOUAP surpasses baseline performance, we advocate for its widespread attention.

摘要: 目前，许多研究已经独立地解决了与可见光和红外探测器相关的安全问题。在实际场景中，使用跨模式检测器执行任务被证明比依赖单模式检测器更可靠。尽管如此，目前还缺乏对跨模式探测器的全面安全评估。虽然现有的研究已经探索了针对跨模式检测器的攻击的可行性，但健壮攻击的实现仍然没有得到解决。这项工作介绍了两阶段优化的统一对抗补丁(TOUAP)，设计用于在现实世界的黑匣子环境中执行对可见光-红外交叉模式探测器的攻击。TOUAP算法采用两阶段优化过程：首先，粒子群算法对攻击红外探测器的不规则多边形红外贴片进行优化；其次，对颜色二维码进行优化，并将第一阶段得到的红外贴片的形状信息作为掩码。所得到的不规则多边形可见模式面片对可见检测器执行攻击。通过在数字和物理环境中进行的大量实验，验证了该方法的有效性和稳健性。由于TOUAP超过了基线表现，我们主张广泛关注它。



## **19. Singular Regularization with Information Bottleneck Improves Model's Adversarial Robustness**

带信息瓶颈的奇异正则化提高模型的对抗稳健性 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.02237v1) [paper-pdf](http://arxiv.org/pdf/2312.02237v1)

**Authors**: Guanlin Li, Naishan Zheng, Man Zhou, Jie Zhang, Tianwei Zhang

**Abstract**: Adversarial examples are one of the most severe threats to deep learning models. Numerous works have been proposed to study and defend adversarial examples. However, these works lack analysis of adversarial information or perturbation, which cannot reveal the mystery of adversarial examples and lose proper interpretation. In this paper, we aim to fill this gap by studying adversarial information as unstructured noise, which does not have a clear pattern. Specifically, we provide some empirical studies with singular value decomposition, by decomposing images into several matrices, to analyze adversarial information for different attacks. Based on the analysis, we propose a new module to regularize adversarial information and combine information bottleneck theory, which is proposed to theoretically restrict intermediate representations. Therefore, our method is interpretable. Moreover, the fashion of our design is a novel principle that is general and unified. Equipped with our new module, we evaluate two popular model structures on two mainstream datasets with various adversarial attacks. The results indicate that the improvement in robust accuracy is significant. On the other hand, we prove that our method is efficient with only a few additional parameters and able to be explained under regional faithfulness analysis.

摘要: 对抗性例子是深度学习模式面临的最严重威胁之一。已经提出了大量的工作来研究和辩护对抗性例子。然而，这些作品缺乏对对抗性信息或扰动的分析，不能揭示对抗性例子的奥秘，从而失去了适当的解释。在本文中，我们旨在通过研究非结构化噪声来填补这一空白，这种非结构化噪声没有明确的模式。具体地说，我们提供了一些奇异值分解的实证研究，通过将图像分解成几个矩阵，来分析不同攻击的对抗性信息。在此基础上，我们提出了一种新的对抗性信息正规化模型，并结合信息瓶颈理论，从理论上对中间表征进行了约束。因此，我们的方法是可解释的。此外，我们的设计时尚是一种新的原则，是通用的和统一的。使用我们的新模块，我们在两个主流数据集上对两个流行的模型结构进行了评估，并进行了各种对抗性攻击。结果表明，该方法在稳健性精度方面有显著的提高。另一方面，我们证明了我们的方法是有效的，只需要很少的附加参数，并且能够在区域忠诚度分析下得到解释。



## **20. Warfare:Breaking the Watermark Protection of AI-Generated Content**

战争：打破人工智能生成内容的水印保护 cs.CV

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2310.07726v2) [paper-pdf](http://arxiv.org/pdf/2310.07726v2)

**Authors**: Guanlin Li, Yifei Chen, Jie Zhang, Jiwei Li, Shangwei Guo, Tianwei Zhang

**Abstract**: AI-Generated Content (AIGC) is gaining great popularity, with many emerging commercial services and applications. These services leverage advanced generative models, such as latent diffusion models and large language models, to generate creative content (e.g., realistic images and fluent sentences) for users. The usage of such generated content needs to be highly regulated, as the service providers need to ensure the users do not violate the usage policies (e.g., abuse for commercialization, generating and distributing unsafe content). A promising solution to achieve this goal is watermarking, which adds unique and imperceptible watermarks on the content for service verification and attribution. Numerous watermarking approaches have been proposed recently. However, in this paper, we show that an adversary can easily break these watermarking mechanisms. Specifically, we consider two possible attacks. (1) Watermark removal: the adversary can easily erase the embedded watermark from the generated content and then use it freely bypassing the regulation of the service provider. (2) Watermark forging: the adversary can create illegal content with forged watermarks from another user, causing the service provider to make wrong attributions. We propose Warfare, a unified methodology to achieve both attacks in a holistic way. The key idea is to leverage a pre-trained diffusion model for content processing and a generative adversarial network for watermark removal or forging. We evaluate Warfare on different datasets and embedding setups. The results prove that it can achieve high success rates while maintaining the quality of the generated content. Compared to existing diffusion model-based attacks, Warfare is 5,050~11,000x faster.

摘要: 人工智能生成的内容(AIGC)越来越受欢迎，出现了许多新兴的商业服务和应用程序。这些服务利用高级生成模型，如潜在扩散模型和大型语言模型，为用户生成创造性内容(例如，逼真的图像和流畅的句子)。这种生成的内容的使用需要受到严格的监管，因为服务提供商需要确保用户不违反使用策略(例如，滥用以商业化、生成和分发不安全的内容)。实现这一目标的一个有前途的解决方案是水印，它在内容上添加唯一且不可察觉的水印，用于服务验证和归属。最近，人们提出了许多水印方法。然而，在本文中，我们证明了攻击者可以很容易地破解这些水印机制。具体地说，我们考虑两种可能的攻击。(1)水印去除：攻击者可以很容易地从生成的内容中删除嵌入的水印，然后绕过服务提供商的监管自由使用。(2)水印伪造：对手可以利用来自其他用户的伪造水印创建非法内容，导致服务提供商做出错误的归属。我们提出战争，一种统一的方法论，以整体的方式实现这两种攻击。其关键思想是利用预先训练的扩散模型来进行内容处理，并利用生成性对抗网络来去除或伪造水印。我们对不同的数据集和嵌入设置进行了战争评估。实验结果表明，该算法在保证生成内容质量的同时，具有较高的成功率。与现有的基于扩散模型的攻击相比，战争的速度要快5050~11000倍。



## **21. Malicious Lateral Movement in 5G Core With Network Slicing And Its Detection**

基于网络分片的5G核心恶意侧移及其检测 cs.CR

Accepted for publication in the Proceedings of IEEE ITNAC-2023

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01681v1) [paper-pdf](http://arxiv.org/pdf/2312.01681v1)

**Authors**: Ayush Kumar, Vrizlynn L. L. Thing

**Abstract**: 5G networks are susceptible to cyber attacks due to reasons such as implementation issues and vulnerabilities in 3GPP standard specifications. In this work, we propose lateral movement strategies in a 5G Core (5GC) with network slicing enabled, as part of a larger attack campaign by well-resourced adversaries such as APT groups. Further, we present 5GLatte, a system to detect such malicious lateral movement. 5GLatte operates on a host-container access graph built using host/NF container logs collected from the 5GC. Paths inferred from the access graph are scored based on selected filtering criteria and subsequently presented as input to a threshold-based anomaly detection algorithm to reveal malicious lateral movement paths. We evaluate 5GLatte on a dataset containing attack campaigns (based on MITRE ATT&CK and FiGHT frameworks) launched in a 5G test environment which shows that compared to other lateral movement detectors based on state-of-the-art, it can achieve higher true positive rates with similar false positive rates.

摘要: 由于3GPP标准规范中的实施问题和漏洞等原因，5G网络容易受到网络攻击。在这项工作中，我们提出了启用网络切片的5G核心（5GC）中的横向移动策略，作为资源充足的对手（如APT团体）更大规模攻击活动的一部分。此外，我们提出了5GLatte，一个系统来检测这种恶意的横向移动。5Glatte在使用从5GC收集的主机/NF容器日志构建的主机-容器访问图上运行。从访问图推断的路径基于所选择的过滤标准被评分，并且随后被呈现为基于阈值的异常检测算法的输入，以揭示恶意横向移动路径。我们在包含5G测试环境中启动的攻击活动（基于MITRE ATT&CK和FIGHT框架）的数据集上评估了5GLatte，结果表明，与其他基于最先进技术的横向移动检测器相比，它可以实现更高的真阳性率和类似的假阳性率。



## **22. Adversarial Medical Image with Hierarchical Feature Hiding**

基于分层特征隐藏的对抗性医学图像 eess.IV

Our code is available at  \url{https://github.com/qsyao/Hierarchical_Feature_Constraint}. arXiv admin  note: text overlap with arXiv:2012.09501

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2312.01679v1) [paper-pdf](http://arxiv.org/pdf/2312.01679v1)

**Authors**: Qingsong Yao, Zecheng He, Yuexiang Li, Yi Lin, Kai Ma, Yefeng Zheng, S. Kevin Zhou

**Abstract**: Deep learning based methods for medical images can be easily compromised by adversarial examples (AEs), posing a great security flaw in clinical decision-making. It has been discovered that conventional adversarial attacks like PGD which optimize the classification logits, are easy to distinguish in the feature space, resulting in accurate reactive defenses. To better understand this phenomenon and reassess the reliability of the reactive defenses for medical AEs, we thoroughly investigate the characteristic of conventional medical AEs. Specifically, we first theoretically prove that conventional adversarial attacks change the outputs by continuously optimizing vulnerable features in a fixed direction, thereby leading to outlier representations in the feature space. Then, a stress test is conducted to reveal the vulnerability of medical images, by comparing with natural images. Interestingly, this vulnerability is a double-edged sword, which can be exploited to hide AEs. We then propose a simple-yet-effective hierarchical feature constraint (HFC), a novel add-on to conventional white-box attacks, which assists to hide the adversarial feature in the target feature distribution. The proposed method is evaluated on three medical datasets, both 2D and 3D, with different modalities. The experimental results demonstrate the superiority of HFC, \emph{i.e.,} it bypasses an array of state-of-the-art adversarial medical AE detectors more efficiently than competing adaptive attacks, which reveals the deficiencies of medical reactive defense and allows to develop more robust defenses in future.

摘要: 基于深度学习的医学图像处理方法容易受到对抗性实例的攻击，在临床决策中存在很大的安全缺陷。已经发现，像PGD这样的传统对抗性攻击优化了分类逻辑，在特征空间中很容易区分，从而产生准确的反应性防御。为了更好地理解这一现象，并重新评估医用AEs反应性防御的可靠性，我们深入研究了传统医用AEs的特点。具体地说，我们首先从理论上证明了传统的对抗性攻击通过在固定方向上不断优化易受攻击的特征来改变输出，从而导致特征空间中的孤立点表示。然后，通过与自然图像的比较，进行了压力测试，揭示了医学图像的脆弱性。有趣的是，这个漏洞是一把双刃剑，可以被利用来隐藏AE。然后，我们提出了一种简单有效的层次特征约束(HFC)，这是对传统白盒攻击的一种新的补充，它帮助隐藏目标特征分布中的对抗性特征。该方法在三个医学数据集上进行了评估，包括2D和3D，使用不同的模式。实验结果证明了HFC的优越性，即它比竞争的自适应攻击更有效地绕过了一系列最先进的对抗性医学AE检测器，这揭示了医学反应性防御的不足，并为未来开发更健壮的防御奠定了基础。



## **23. The Queen's Guard: A Secure Enforcement of Fine-grained Access Control In Distributed Data Analytics Platforms**

女王卫队：分布式数据分析平台中细粒度访问控制的安全执行 cs.CR

**SubmitDate**: 2023-12-04    [abs](http://arxiv.org/abs/2106.13123v4) [paper-pdf](http://arxiv.org/pdf/2106.13123v4)

**Authors**: Fahad Shaon, Sazzadur Rahaman, Murat Kantarcioglu

**Abstract**: Distributed data analytics platforms (i.e., Apache Spark, Hadoop) provide high-level APIs to programmatically write analytics tasks that are run distributedly in multiple computing nodes. The design of these frameworks was primarily motivated by performance and usability. Thus, the security takes a back seat. Consequently, they do not inherently support fine-grained access control or offer any plugin mechanism to enable it, making them risky to be used in multi-tier organizational settings.   There have been attempts to build "add-on" solutions to enable fine-grained access control for distributed data analytics platforms. In this paper, first, we show that straightforward enforcement of ``add-on'' access control is insecure under adversarial code execution. Specifically, we show that an attacker can abuse platform-provided APIs to evade access controls without leaving any traces. Second, we designed a two-layered (i.e., proactive and reactive) defense system to protect against API abuses. On submission of a user code, our proactive security layer statically screens it to find potential attack signatures prior to its execution. The reactive security layer employs code instrumentation-based runtime checks and sandboxed execution to throttle any exploits at runtime. Next, we propose a new fine-grained access control framework with an enhanced policy language that supports map and filter primitives. Finally, we build a system named SecureDL with our new access control framework and defense system on top of Apache Spark, which ensures secure access control policy enforcement under adversaries capable of executing code.   To the best of our knowledge, this is the first fine-grained attribute-based access control framework for distributed data analytics platforms that is secure against platform API abuse attacks. Performance evaluation showed that the overhead due to added security is low.

摘要: 分布式数据分析平台(即，ApacheSpark、Hadoop)提供高级API，以编程方式编写在多个计算节点上分布式运行的分析任务。这些框架的设计主要是出于性能和可用性的考虑。因此，安全措施就退居次要地位了。因此，它们本身并不支持细粒度的访问控制，也不提供任何插件机制来启用它，这使得它们在多层组织设置中使用存在风险。已经有人尝试构建“附加”解决方案来实现分布式数据分析平台的细粒度访问控制。在这篇文章中，我们首先证明了直接实施“附加”访问控制在恶意代码执行下是不安全的。具体地说，我们展示了攻击者可以滥用平台提供的API来逃避访问控制，而不会留下任何痕迹。其次，我们设计了一个双层(即主动和被动)防御体系，以防止API滥用。在提交用户代码时，我们的主动安全层会在代码执行之前对其进行静态筛选，以发现潜在的攻击特征。反应式安全层采用基于代码检测的运行时检查和沙箱执行，以在运行时遏制任何利用漏洞。接下来，我们提出了一种新的细粒度访问控制框架，该框架具有增强的策略语言，支持映射和过滤原语。最后，我们使用新的访问控制框架和防御系统在ApacheSpark之上构建了一个名为SecureDL的系统，确保了在能够执行代码的攻击者的情况下安全地执行访问控制策略。据我们所知，这是第一个针对分布式数据分析平台的细粒度基于属性的访问控制框架，可以安全地抵御平台API滥用攻击。性能评估表明，由于增加安全性而产生的开销很低。



## **24. Robust Evaluation of Diffusion-Based Adversarial Purification**

基于扩散的对抗净化算法的稳健性评价 cs.CV

Accepted by ICCV 2023, oral presentation. Code is available at  https://github.com/ml-postech/robust-evaluation-of-diffusion-based-purification

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2303.09051v3) [paper-pdf](http://arxiv.org/pdf/2303.09051v3)

**Authors**: Minjong Lee, Dongwoo Kim

**Abstract**: We question the current evaluation practice on diffusion-based purification methods. Diffusion-based purification methods aim to remove adversarial effects from an input data point at test time. The approach gains increasing attention as an alternative to adversarial training due to the disentangling between training and testing. Well-known white-box attacks are often employed to measure the robustness of the purification. However, it is unknown whether these attacks are the most effective for the diffusion-based purification since the attacks are often tailored for adversarial training. We analyze the current practices and provide a new guideline for measuring the robustness of purification methods against adversarial attacks. Based on our analysis, we further propose a new purification strategy improving robustness compared to the current diffusion-based purification methods.

摘要: 我们对目前基于扩散的纯化方法的评估实践提出了质疑。基于扩散的净化方法旨在消除测试时输入数据点的对抗性影响。由于训练和测试之间的分离，这种方法作为对抗性训练的替代方法受到越来越多的关注。通常使用众所周知的白盒攻击来衡量净化的健壮性。然而，目前尚不清楚这些攻击对于基于扩散的净化是否最有效，因为这些攻击通常是为对抗性训练量身定做的。我们分析了目前的实践，并为衡量净化方法对对手攻击的健壮性提供了新的指导方针。在分析的基础上，进一步提出了一种新的纯化策略，与现有的基于扩散的纯化方法相比，提高了算法的稳健性。



## **25. QuantAttack: Exploiting Dynamic Quantization to Attack Vision Transformers**

QuantAttack：利用动态量化攻击视觉变形金刚 cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02220v1) [paper-pdf](http://arxiv.org/pdf/2312.02220v1)

**Authors**: Amit Baras, Alon Zolfi, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, there has been a significant trend in deep neural networks (DNNs), particularly transformer-based models, of developing ever-larger and more capable models. While they demonstrate state-of-the-art performance, their growing scale requires increased computational resources (e.g., GPUs with greater memory capacity). To address this problem, quantization techniques (i.e., low-bit-precision representation and matrix multiplication) have been proposed. Most quantization techniques employ a static strategy in which the model parameters are quantized, either during training or inference, without considering the test-time sample. In contrast, dynamic quantization techniques, which have become increasingly popular, adapt during inference based on the input provided, while maintaining full-precision performance. However, their dynamic behavior and average-case performance assumption makes them vulnerable to a novel threat vector -- adversarial attacks that target the model's efficiency and availability. In this paper, we present QuantAttack, a novel attack that targets the availability of quantized models, slowing down the inference, and increasing memory usage and energy consumption. We show that carefully crafted adversarial examples, which are designed to exhaust the resources of the operating system, can trigger worst-case performance. In our experiments, we demonstrate the effectiveness of our attack on vision transformers on a wide range of tasks, both uni-modal and multi-modal. We also examine the effect of different attack variants (e.g., a universal perturbation) and the transferability between different models.

摘要: 近年来，深度神经网络(DNN)，特别是基于变压器的模型，有一个显著的趋势，即开发更大、更有能力的模型。虽然它们展示了一流的性能，但其不断增长的规模需要增加计算资源(例如，具有更大内存容量的GPU)。为了解决这个问题，人们提出了量化技术(即低位精度表示和矩阵乘法)。大多数量化技术采用静态策略，其中模型参数在训练或推理期间被量化，而不考虑测试时间样本。相比之下，已经变得越来越流行的动态量化技术在基于所提供的输入进行推理期间进行调整，同时保持全精度性能。然而，它们的动态行为和平均情况性能假设使它们容易受到一种新的威胁矢量--以模型的效率和可用性为目标的对抗性攻击。在本文中，我们提出了一种新的攻击QuantAttack，它的目标是量化模型的可用性，减慢推理速度，增加内存使用和能量消耗。我们展示了精心设计的敌意例子，这些例子旨在耗尽操作系统的资源，可以触发最坏的情况下的性能。在我们的实验中，我们展示了我们对视觉转换器的攻击在各种任务中的有效性，包括单模式和多模式。我们还考察了不同攻击变量(例如，通用扰动)的影响以及不同模型之间的可转移性。



## **26. Exploring Adversarial Robustness of LiDAR-Camera Fusion Model in Autonomous Driving**

自动驾驶中LiDAR-Camera融合模型的对抗稳健性研究 cs.RO

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01468v1) [paper-pdf](http://arxiv.org/pdf/2312.01468v1)

**Authors**: Bo Yang, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji, Xiaoyu Ji

**Abstract**: Our study assesses the adversarial robustness of LiDAR-camera fusion models in 3D object detection. We introduce an attack technique that, by simply adding a limited number of physically constrained adversarial points above a car, can make the car undetectable by the fusion model. Experimental results reveal that even without changes to the image data channel, the fusion model can be deceived solely by manipulating the LiDAR data channel. This finding raises safety concerns in the field of autonomous driving. Further, we explore how the quantity of adversarial points, the distance between the front-near car and the LiDAR-equipped car, and various angular factors affect the attack success rate. We believe our research can contribute to the understanding of multi-sensor robustness, offering insights and guidance to enhance the safety of autonomous driving.

摘要: 我们的研究评估了LiDAR-相机融合模型在3D目标检测中的对抗健壮性。我们介绍了一种攻击技术，只需在汽车上方添加有限数量的物理约束对手点，就可以使汽车无法被融合模型检测到。实验结果表明，即使在不改变图像数据通道的情况下，仅通过操纵LiDAR数据通道也可以欺骗融合模型。这一发现引发了自动驾驶领域的安全担忧。在此基础上，进一步探讨了攻击点的数量、前近车与装备激光雷达的车之间的距离以及各种角度因素对攻击成功率的影响。我们相信，我们的研究将有助于理解多传感器的稳健性，为提高自动驾驶的安全性提供见解和指导。



## **27. Evaluating the Security of Satellite Systems**

评估卫星系统的安全性 cs.CR

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01330v1) [paper-pdf](http://arxiv.org/pdf/2312.01330v1)

**Authors**: Roy Peled, Eran Aizikovich, Edan Habler, Yuval Elovici, Asaf Shabtai

**Abstract**: Satellite systems are facing an ever-increasing amount of cybersecurity threats as their role in communications, navigation, and other services expands. Recent papers have examined attacks targeting satellites and space systems; however, they did not comprehensively analyze the threats to satellites and systematically identify adversarial techniques across the attack lifecycle. This paper presents a comprehensive taxonomy of adversarial tactics, techniques, and procedures explicitly targeting LEO satellites. First, we analyze the space ecosystem including the ground, space, Communication, and user segments, highlighting their architectures, functions, and vulnerabilities. Then, we examine the threat landscape, including adversary types, and capabilities, and survey historical and recent attacks such as jamming, spoofing, and supply chain. Finally, we propose a novel extension of the MITRE ATT&CK framework to categorize satellite attack techniques across the adversary lifecycle from reconnaissance to impact. The taxonomy is demonstrated by modeling high-profile incidents, including the Viasat attack that disrupted Ukraine's communications. The taxonomy provides the foundation for the development of defenses against emerging cyber risks to space assets. The proposed threat model will advance research in the space domain and contribute to the security of the space domain against sophisticated attacks.

摘要: 随着卫星系统在通信、导航和其他服务中的作用不断扩大，它们正面临着越来越多的网络安全威胁。最近的论文审查了针对卫星和空间系统的攻击；然而，它们没有全面分析对卫星的威胁，并在整个攻击生命周期中系统地确定对抗技术。本文对明确针对低轨卫星的对抗战术、技术和程序进行了全面的分类。首先，我们分析了空间生态系统，包括地面、空间、通信和用户部分，重点介绍了它们的架构、功能和漏洞。然后，我们检查威胁环境，包括对手类型和能力，并调查历史和最近的攻击，如干扰、欺骗和供应链。最后，我们提出了一种新的扩展MITRE ATT&CK框架，用于对从侦察到影响的整个敌方生命周期的卫星攻击技术进行分类。通过对备受瞩目的事件进行建模，包括中断乌克兰通信的Viasat袭击，展示了这一分类。该分类为开发针对空间资产新出现的网络风险的防御措施提供了基础。拟议的威胁模型将推动空间领域的研究，并有助于空间领域免受复杂攻击的安全。



## **28. Rethinking PGD Attack: Is Sign Function Necessary?**

对PGD攻击的再思考：是否需要符号功能？ cs.LG

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.01260v1) [paper-pdf](http://arxiv.org/pdf/2312.01260v1)

**Authors**: Junjie Yang, Tianlong Chen, Xuxi Chen, Zhangyang Wang, Yingbin Liang

**Abstract**: Neural networks have demonstrated success in various domains, yet their performance can be significantly degraded by even a small input perturbation. Consequently, the construction of such perturbations, known as adversarial attacks, has gained significant attention, many of which fall within "white-box" scenarios where we have full access to the neural network. Existing attack algorithms, such as the projected gradient descent (PGD), commonly take the sign function on the raw gradient before updating adversarial inputs, thereby neglecting gradient magnitude information. In this paper, we present a theoretical analysis of how such sign-based update algorithm influences step-wise attack performance, as well as its caveat. We also interpret why previous attempts of directly using raw gradients failed. Based on that, we further propose a new raw gradient descent (RGD) algorithm that eliminates the use of sign. Specifically, we convert the constrained optimization problem into an unconstrained one, by introducing a new hidden variable of non-clipped perturbation that can move beyond the constraint. The effectiveness of the proposed RGD algorithm has been demonstrated extensively in experiments, outperforming PGD and other competitors in various settings, without incurring any additional computational overhead. The codes is available in https://github.com/JunjieYang97/RGD.

摘要: 神经网络已经在各个领域取得了成功，但即使是很小的输入扰动也会显著降低其性能。因此，这种被称为对抗性攻击的扰动的构造得到了极大的关注，其中许多都属于我们可以完全访问神经网络的“白盒”情景。现有的攻击算法，如投影梯度下降(PGD)算法，通常在更新敌方输入之前对原始梯度取符号函数，从而忽略了梯度大小信息。本文从理论上分析了这种基于符号的更新算法对分步攻击性能的影响，并给出了相应的警告。我们还解释了为什么以前直接使用原始梯度的尝试失败了。在此基础上，进一步提出了一种新的原始梯度下降(RGD)算法，该算法省去了符号的使用。具体地说，我们通过引入一个可以超越约束的非剪裁扰动的新的隐变量，将约束优化问题转化为无约束优化问题。所提出的RGD算法的有效性已经在实验中得到了广泛的证明，在不引起任何额外计算开销的情况下，在不同环境下的性能优于PGD和其他竞争对手。这些代码可以在https://github.com/JunjieYang97/RGD.中找到



## **29. TranSegPGD: Improving Transferability of Adversarial Examples on Semantic Segmentation**

TranSegPGD：提高语义切分对抗性实例的可转移性 cs.CV

**SubmitDate**: 2023-12-03    [abs](http://arxiv.org/abs/2312.02207v1) [paper-pdf](http://arxiv.org/pdf/2312.02207v1)

**Authors**: Xiaojun Jia, Jindong Gu, Yihao Huang, Simeng Qin, Qing Guo, Yang Liu, Xiaochun Cao

**Abstract**: Transferability of adversarial examples on image classification has been systematically explored, which generates adversarial examples in black-box mode. However, the transferability of adversarial examples on semantic segmentation has been largely overlooked. In this paper, we propose an effective two-stage adversarial attack strategy to improve the transferability of adversarial examples on semantic segmentation, dubbed TranSegPGD. Specifically, at the first stage, every pixel in an input image is divided into different branches based on its adversarial property. Different branches are assigned different weights for optimization to improve the adversarial performance of all pixels.We assign high weights to the loss of the hard-to-attack pixels to misclassify all pixels. At the second stage, the pixels are divided into different branches based on their transferable property which is dependent on Kullback-Leibler divergence. Different branches are assigned different weights for optimization to improve the transferability of the adversarial examples. We assign high weights to the loss of the high-transferability pixels to improve the transferability of adversarial examples. Extensive experiments with various segmentation models are conducted on PASCAL VOC 2012 and Cityscapes datasets to demonstrate the effectiveness of the proposed method. The proposed adversarial attack method can achieve state-of-the-art performance.

摘要: 系统地研究了图像分类中对抗性实例的可转移性，生成了黑盒模式下的对抗性实例。然而，语义切分中对抗性例子的可转移性在很大程度上被忽视了。在本文中，我们提出了一种有效的两阶段对抗性攻击策略，以提高对抗性实例在语义分割上的可转移性，称为TranSegPGD。具体地说，在第一阶段，输入图像中的每个像素根据其对抗性被分成不同的分支。不同的分支被赋予不同的权重进行优化，以提高所有像素的对抗性能；对难以攻击的像素的损失赋予较高的权重，以实现对所有像素的误分类。在第二阶段，根据像素的可移动性将像素划分为不同的分支，该属性依赖于Kullback-Leibler散度。不同的分支被赋予不同的权重进行优化，以提高对抗性实例的可转移性。我们对高可转移性像素的损失赋予较高的权重，以提高对抗性例子的可转移性。在PASCAL VOC 2012和CITYSCAPES数据集上对各种分割模型进行了大量的实验，验证了该方法的有效性。所提出的对抗性攻击方法可以获得最先进的性能。



## **30. Look Closer to Your Enemy: Learning to Attack via Teacher-Student Mimicking**

走近你的敌人：通过师生模仿学习攻击 cs.CV

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2207.13381v4) [paper-pdf](http://arxiv.org/pdf/2207.13381v4)

**Authors**: Mingjie Wang, Jianxiong Guo, Sirui Li, Dingwen Xiao, Zhiqing Tang

**Abstract**: Deep neural networks have significantly advanced person re-identification (ReID) applications in the realm of the industrial internet, yet they remain vulnerable. Thus, it is crucial to study the robustness of ReID systems, as there are risks of adversaries using these vulnerabilities to compromise industrial surveillance systems. Current adversarial methods focus on generating attack samples using misclassification feedback from victim models (VMs), neglecting VM's cognitive processes. We seek to address this by producing authentic ReID attack instances through VM cognition decryption. This approach boasts advantages like better transferability to open-set ReID tests, easier VM misdirection, and enhanced creation of realistic and undetectable assault images. However, the task of deciphering the cognitive mechanism in VM is widely considered to be a formidable challenge. In this paper, we propose a novel inconspicuous and controllable ReID attack baseline, LCYE (Look Closer to Your Enemy), to generate adversarial query images. Specifically, LCYE first distills VM's knowledge via teacher-student memory mimicking the proxy task. This knowledge prior serves as an unambiguous cryptographic token, encapsulating elements deemed indispensable and plausible by the VM, with the intent of facilitating precise adversarial misdirection. Further, benefiting from the multiple opposing task framework of LCYE, we investigate the interpretability and generalization of ReID models from the view of the adversarial attack, including cross-domain adaption, cross-model consensus, and online learning process. Extensive experiments on four ReID benchmarks show that our method outperforms other state-of-the-art attackers with a large margin in white-box, black-box, and target attacks. The source code can be found at https://github.com/MingjieWang0606/LCYE-attack_reid.

摘要: 深度神经网络在工业互联网领域显著推进了个人重新识别(ReID)的应用，但它们仍然很脆弱。因此，研究REID系统的健壮性是至关重要的，因为存在攻击者利用这些漏洞来危害工业监控系统的风险。目前的攻击方法主要是利用受害者模型的错误分类反馈来生成攻击样本，而忽略了受害者模型的认知过程。我们试图通过VM认知解密生成真实的Reid攻击实例来解决这个问题。这种方法拥有更好的可移植到开放设置的Reid测试，更容易的VM误导，以及增强的真实和不可检测的攻击图像的创建等优势。然而，破译VM的认知机制被广泛认为是一项艰巨的挑战。在本文中，我们提出了一种新的隐蔽和可控的Reid攻击基线，LCYE(Look Closer To Your Enemy)，用于生成敌意查询图像。具体地说，LCYE首先通过模仿代理任务的师生记忆提取VM的知识。该知识先验用作明确的加密令牌，封装了被VM认为必不可少且看似可信的元素，目的是促进精确的对抗性误导。此外，借鉴LCYE的多重对立任务框架，我们从对抗性攻击的角度考察了Reid模型的可解释性和泛化，包括跨域适应、跨模型共识和在线学习过程。在四个Reid基准测试上的大量实验表明，我们的方法在白盒、黑盒和目标攻击中的性能远远优于其他最先进的攻击者。源代码可以在https://github.com/MingjieWang0606/LCYE-attack_reid.上找到



## **31. FRAUDability: Estimating Users' Susceptibility to Financial Fraud Using Adversarial Machine Learning**

欺诈性：使用对抗性机器学习估计用户对金融欺诈的敏感性 cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.01200v1) [paper-pdf](http://arxiv.org/pdf/2312.01200v1)

**Authors**: Chen Doytshman, Satoru Momiyama, Inderjeet Singh, Yuval Elovici, Asaf Shabtai

**Abstract**: In recent years, financial fraud detection systems have become very efficient at detecting fraud, which is a major threat faced by e-commerce platforms. Such systems often include machine learning-based algorithms aimed at detecting and reporting fraudulent activity. In this paper, we examine the application of adversarial learning based ranking techniques in the fraud detection domain and propose FRAUDability, a method for the estimation of a financial fraud detection system's performance for every user. We are motivated by the assumption that "not all users are created equal" -- while some users are well protected by fraud detection algorithms, others tend to pose a challenge to such systems. The proposed method produces scores, namely "fraudability scores," which are numerical estimations of a fraud detection system's ability to detect financial fraud for a specific user, given his/her unique activity in the financial system. Our fraudability scores enable those tasked with defending users in a financial platform to focus their attention and resources on users with high fraudability scores to better protect them. We validate our method using a real e-commerce platform's dataset and demonstrate the application of fraudability scores from the attacker's perspective, on the platform, and more specifically, on the fraud detection systems used by the e-commerce enterprise. We show that the scores can also help attackers increase their financial profit by 54%, by engaging solely with users with high fraudability scores, avoiding those users whose spending habits enable more accurate fraud detection.

摘要: 近年来，金融欺诈检测系统在检测欺诈方面变得非常高效，这是电子商务平台面临的一大威胁。这类系统通常包括旨在检测和报告欺诈活动的基于机器学习的算法。本文研究了基于对抗性学习的排序技术在欺诈检测领域中的应用，并提出了一种估计每个用户的金融欺诈检测系统性能的方法FRAUDability。我们的动机是这样一个假设：“并非所有用户都是生而平等的”--虽然一些用户受到欺诈检测算法的良好保护，但其他用户往往会对这样的系统构成挑战。建议的方法产生分数，即“欺诈性分数”，这是对欺诈检测系统在给定特定用户在金融系统中的独特活动的情况下为其检测金融欺诈的能力的数字估计。我们的欺诈性得分使那些在金融平台中负责保护用户的人能够将他们的注意力和资源集中在欺诈性得分较高的用户上，以更好地保护他们。我们使用一个真实的电子商务平台的数据集来验证我们的方法，并从攻击者的角度演示了欺诈性分数在平台上的应用，更具体地说，在电子商务企业使用的欺诈检测系统上的应用。我们发现，这些分数还可以帮助攻击者增加54%的经济利润，只与欺诈性分数较高的用户打交道，避开那些消费习惯能够更准确地检测欺诈的用户。



## **32. Adversarial Training for Graph Neural Networks: Pitfalls, Solutions, and New Directions**

图神经网络的对抗性训练：陷阱、解决方案和新方向 cs.LG

Published as a conference paper at NeurIPS 2023

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2306.15427v2) [paper-pdf](http://arxiv.org/pdf/2306.15427v2)

**Authors**: Lukas Gosch, Simon Geisler, Daniel Sturm, Bertrand Charpentier, Daniel Zügner, Stephan Günnemann

**Abstract**: Despite its success in the image domain, adversarial training did not (yet) stand out as an effective defense for Graph Neural Networks (GNNs) against graph structure perturbations. In the pursuit of fixing adversarial training (1) we show and overcome fundamental theoretical as well as practical limitations of the adopted graph learning setting in prior work; (2) we reveal that more flexible GNNs based on learnable graph diffusion are able to adjust to adversarial perturbations, while the learned message passing scheme is naturally interpretable; (3) we introduce the first attack for structure perturbations that, while targeting multiple nodes at once, is capable of handling global (graph-level) as well as local (node-level) constraints. Including these contributions, we demonstrate that adversarial training is a state-of-the-art defense against adversarial structure perturbations.

摘要: 尽管它在图像领域取得了成功，但对抗性训练(目前还没有)成为图神经网络(GNN)对抗图结构扰动的有效防御。在固定对抗性训练的过程中，(1)我们展示并克服了以前工作中采用的图学习设置的基本理论和实践限制；(2)我们揭示了基于可学习图扩散的更灵活的GNN能够适应对抗性扰动，而学习的消息传递方案自然是可解释的；(3)我们引入了针对结构扰动的第一次攻击，虽然一次针对多个节点，但能够处理全局(图级)和局部(节点级)约束。包括这些贡献，我们证明了对抗性训练是对对抗性结构扰动的一种最先进的防御。



## **33. Scrappy: SeCure Rate Assuring Protocol with PrivacY**

Scarppy：带隐私的安全速率保证协议 cs.CR

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00989v1) [paper-pdf](http://arxiv.org/pdf/2312.00989v1)

**Authors**: Kosei Akama, Yoshimichi Nakatsuka, Masaaki Sato, Keisuke Uehara

**Abstract**: Preventing abusive activities caused by adversaries accessing online services at a rate exceeding that expected by websites has become an ever-increasing problem. CAPTCHAs and SMS authentication are widely used to provide a solution by implementing rate limiting, although they are becoming less effective, and some are considered privacy-invasive. In light of this, many studies have proposed better rate-limiting systems that protect the privacy of legitimate users while blocking malicious actors. However, they suffer from one or more shortcomings: (1) assume trust in the underlying hardware and (2) are vulnerable to side-channel attacks. Motivated by the aforementioned issues, this paper proposes Scrappy: SeCure Rate Assuring Protocol with PrivacY. Scrappy allows clients to generate unforgeable yet unlinkable rate-assuring proofs, which provides the server with cryptographic guarantees that the client is not misbehaving. We design Scrappy using a combination of DAA and hardware security devices. Scrappy is implemented over three types of devices, including one that can immediately be deployed in the real world. Our baseline evaluation shows that the end-to-end latency of Scrappy is minimal, taking only 0.32 seconds, and uses only 679 bytes of bandwidth when transferring necessary data. We also conduct an extensive security evaluation, showing that the rate-limiting capability of Scrappy is unaffected even if the hardware security device is compromised.

摘要: 防止对手以超出网站预期的速度访问在线服务导致的滥用活动已成为一个日益严重的问题。验证码和短信身份验证被广泛用于通过实施速率限制来提供解决方案，尽管它们正在变得不那么有效，而且有些被认为是侵犯隐私的。有鉴于此，许多研究提出了更好的速率限制系统，在保护合法用户隐私的同时阻止恶意行为者。然而，它们有一个或多个缺点：(1)信任底层硬件；(2)容易受到旁路攻击。基于上述问题，本文提出了Scrppy：带隐私的安全速率保证协议。Screppy允许客户端生成不可伪造但不可链接的费率保证证据，这为服务器提供了客户端没有行为不端的加密保证。我们使用DAA和硬件安全设备的组合来设计Screppy。Scrppy在三种类型的设备上实现，其中一种可以立即部署到现实世界中。我们的基线评估表明，Scarppy的端到端延迟最小，仅需0.32秒，在传输必要的数据时仅占用679字节的带宽。我们还进行了广泛的安全评估，表明即使硬件安全设备被攻破，Screppy的限速能力也不受影响。



## **34. Deep Generative Attacks and Countermeasures for Data-Driven Offline Signature Verification**

数据驱动离线签名验证的深度生成攻击及对策 cs.CV

10 pages, 7 figures, 1 table, Signature verification, Deep generative  models, attacks, generative attack explainability, data-driven verification  system

**SubmitDate**: 2023-12-02    [abs](http://arxiv.org/abs/2312.00987v1) [paper-pdf](http://arxiv.org/pdf/2312.00987v1)

**Authors**: An Ngo, MinhPhuong Cao, Rajesh Kumar

**Abstract**: While previous studies have explored attacks via random, simple, and skilled forgeries, generative attacks have received limited attention in the data-driven signature verification (DASV) process. Thus, this paper explores the impact of generative attacks on DASV and proposes practical and interpretable countermeasures. We investigate the power of two prominent Deep Generative Models (DGMs), Variational Auto-encoders (VAE) and Conditional Generative Adversarial Networks (CGAN), on their ability to generate signatures that would successfully deceive DASV. Additionally, we evaluate the quality of generated images using the Structural Similarity Index measure (SSIM) and use the same to explain the attack's success. Finally, we propose countermeasures that effectively reduce the impact of deep generative attacks on DASV.   We first generated six synthetic datasets from three benchmark offline-signature datasets viz. CEDAR, BHSig260- Bengali, and BHSig260-Hindi using VAE and CGAN. Then, we built baseline DASVs using Xception, ResNet152V2, and DenseNet201. These DASVs achieved average (over the three datasets) False Accept Rates (FARs) of 2.55%, 3.17%, and 1.06%, respectively. Then, we attacked these baselines using the synthetic datasets. The VAE-generated signatures increased average FARs to 10.4%, 10.1%, and 7.5%, while CGAN-generated signatures to 32.5%, 30%, and 26.1%. The variation in the effectiveness of attack for VAE and CGAN was investigated further and explained by a strong (rho = -0.86) negative correlation between FARs and SSIMs. We created another set of synthetic datasets and used the same to retrain the DASVs. The retained baseline showed significant robustness to random, skilled, and generative attacks as the FARs shrank to less than 1% on average. The findings underscore the importance of studying generative attacks and potential countermeasures for DASV.

摘要: 虽然以前的研究探索了通过随机、简单和熟练的伪造进行的攻击，但在数据驱动的签名验证(DASV)过程中，生成性攻击受到的关注有限。因此，本文探讨了产生性攻击对DASV的影响，并提出了切实可行的、可解释的对策。我们研究了两个重要的深度生成模型(DGM)--变分自动编码器(VAE)和条件生成对抗网络(CGAN)--对它们生成成功欺骗DASV的签名的能力的影响。此外，我们使用结构相似性指数(SSIM)来评估生成的图像的质量，并用它来解释攻击的成功。最后，提出了有效降低深度生成性攻击对DASV影响的对策。我们首先从三个基准离线签名数据集生成六个合成数据集，即。雪松、BHSig260-孟加拉语和BHSig260-印地语，使用VAE和CGAN。然后，我们使用Xception、ResNet152V2和DenseNet201构建了基准DASV。这些DASV在三个数据集上的平均错误接受率(FAR)分别为2.55%、3.17%和1.06%。然后，我们使用合成数据集攻击这些基线。VAE生成的签名将平均FAR提高到10.4%、10.1%和7.5%，而CGAN生成的签名增加到32.5%、30%和26.1%。对VAE和CGAN攻击效能的变化进行了进一步的研究，并用FARS和SSIM之间的强负相关(Rho=-0.86)来解释。我们创建了另一组合成数据集，并使用相同的数据集重新训练DASV。保留的基线对随机、熟练和生成性攻击显示出显著的稳健性，因为FAR平均收缩到不到1%。这些发现强调了研究DASV的生成性攻击和潜在对策的重要性。



## **35. Stealing the Decoding Algorithms of Language Models**

窃取语言模型的译码算法 cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2303.04729v4) [paper-pdf](http://arxiv.org/pdf/2303.04729v4)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2, GPT-3 and GPT-Neo. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.

摘要: 从现代语言模型(LM)生成文本的一个关键组件是解码算法的选择和调整。这些算法确定如何从LM生成的内部概率分布生成文本。选择解码算法和调整其超参数的过程需要大量的时间、人工和计算，还需要广泛的人工评估。因此，这种译码算法的恒等式和超参数被认为对它们的所有者非常有价值。在这项工作中，我们首次证明，具有典型API访问权限的攻击者可以以非常低的金钱成本窃取其解码算法的类型和超参数。我们的攻击对文本生成API中使用的流行LMS有效，包括GPT-2、GPT-3和GPT-Neo。我们证明了只需几美元即可窃取此类信息的可行性，例如，对于GPT-3的四个版本，仅需$0.8$、$1$、$4$和$40$。



## **36. Adversarial Attacks and Defenses on 3D Point Cloud Classification: A Survey**

三维点云分类的对抗性攻防综述 cs.CV

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2307.00309v2) [paper-pdf](http://arxiv.org/pdf/2307.00309v2)

**Authors**: Hanieh Naderi, Ivan V. Bajić

**Abstract**: Deep learning has successfully solved a wide range of tasks in 2D vision as a dominant AI technique. Recently, deep learning on 3D point clouds is becoming increasingly popular for addressing various tasks in this field. Despite remarkable achievements, deep learning algorithms are vulnerable to adversarial attacks. These attacks are imperceptible to the human eye but can easily fool deep neural networks in the testing and deployment stage. To encourage future research, this survey summarizes the current progress on adversarial attack and defense techniques on point cloud classification.This paper first introduces the principles and characteristics of adversarial attacks and summarizes and analyzes adversarial example generation methods in recent years. Additionally, it provides an overview of defense strategies, organized into data-focused and model-focused methods. Finally, it presents several current challenges and potential future research directions in this domain.

摘要: 深度学习作为一种占主导地位的人工智能技术，已经成功地解决了2D视觉中的一系列任务。近年来，针对三维点云的深度学习成为解决该领域各种问题的热门方法。尽管深度学习算法取得了令人瞩目的成就，但它仍然容易受到对手的攻击。这些攻击是人眼看不见的，但在测试和部署阶段很容易就能愚弄深度神经网络。本文首先介绍了对抗性攻击的原理和特点，并对近年来的对抗性实例生成方法进行了总结和分析。此外，它还概述了防御策略，并将其组织为以数据为中心和以模型为中心的方法。最后提出了该领域目前面临的几个挑战和潜在的研究方向。



## **37. A Unified Approach to Interpreting and Boosting Adversarial Transferability**

一种统一的解释和提高对抗性转移能力的方法 cs.LG

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2010.04055v2) [paper-pdf](http://arxiv.org/pdf/2010.04055v2)

**Authors**: Xin Wang, Jie Ren, Shuyun Lin, Xiangming Zhu, Yisen Wang, Quanshi Zhang

**Abstract**: In this paper, we use the interaction inside adversarial perturbations to explain and boost the adversarial transferability. We discover and prove the negative correlation between the adversarial transferability and the interaction inside adversarial perturbations. The negative correlation is further verified through different DNNs with various inputs. Moreover, this negative correlation can be regarded as a unified perspective to understand current transferability-boosting methods. To this end, we prove that some classic methods of enhancing the transferability essentially decease interactions inside adversarial perturbations. Based on this, we propose to directly penalize interactions during the attacking process, which significantly improves the adversarial transferability.

摘要: 在本文中，我们使用对抗性扰动内部的相互作用来解释和增强对抗性转移。我们发现并证明了对抗性转移与对抗性扰动中的相互作用之间的负相关关系。通过具有不同输入的不同DNN进一步验证了负相关性。此外，这种负相关性可以被视为理解当前可转让性提升方法的统一视角。为此，我们证明了一些增强可转移性的经典方法本质上减少了对抗性扰动中的相互作用。基于此，我们提出了直接惩罚攻击过程中的交互，这显著提高了对手的可转移性。



## **38. Unleashing Cheapfakes through Trojan Plugins of Large Language Models**

通过大型语言模型特洛伊木马插件释放Cheapfake cs.CR

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2312.00374v1) [paper-pdf](http://arxiv.org/pdf/2312.00374v1)

**Authors**: Tian Dong, Guoxing Chen, Shaofeng Li, Minhui Xue, Rayne Holland, Yan Meng, Zhen Liu, Haojin Zhu

**Abstract**: Open-source Large Language Models (LLMs) have recently gained popularity because of their comparable performance to proprietary LLMs. To efficiently fulfill domain-specialized tasks, open-source LLMs can be refined, without expensive accelerators, using low-rank adapters. However, it is still unknown whether low-rank adapters can be exploited to control LLMs. To address this gap, we demonstrate that an infected adapter can induce, on specific triggers, an LLM to output content defined by an adversary and to even maliciously use tools. To train a Trojan adapter, we propose two novel attacks, POLISHED and FUSION, that improve over prior approaches. POLISHED uses LLM-enhanced paraphrasing to polish benchmark poisoned datasets. In contrast, in the absence of a dataset, FUSION leverages an over-poisoning procedure to transform a benign adaptor. Our experiments validate that our attacks provide higher attack effectiveness than the baseline and, for the purpose of attracting downloads, preserves or improves the adapter's utility. Finally, we provide two case studies to demonstrate that the Trojan adapter can lead a LLM-powered autonomous agent to execute unintended scripts or send phishing emails. Our novel attacks represent the first study of supply chain threats for LLMs through the lens of Trojan plugins.

摘要: 开源的大型语言模型(LLM)最近越来越受欢迎，因为它们的性能可以与专有的LLM相媲美。为了高效地完成领域专门化任务，可以使用低级别适配器对开源LLM进行提炼，而无需使用昂贵的加速器。然而，是否可以利用低阶适配器来控制LLM仍然是未知的。为了弥补这一漏洞，我们演示了受感染的适配器可以在特定触发下诱导LLM输出由对手定义的内容，甚至恶意使用工具。为了训练木马适配器，我们提出了两种新的攻击方法，磨光攻击和融合攻击，它们比以前的方法有所改进。波兰德使用LLM增强的释义来抛光基准有毒数据集。相比之下，在没有数据集的情况下，Fusion利用过度中毒的程序来转换良性适配器。我们的实验验证了我们的攻击提供了比基线更高的攻击效率，并且为了吸引下载的目的，保留或提高了适配器的实用性。最后，我们提供了两个案例研究来演示特洛伊木马适配器可以导致LLM驱动的自主代理执行意外脚本或发送钓鱼电子邮件。我们的新型攻击首次通过特洛伊木马插件的镜头研究了LLM的供应链威胁。



## **39. Bayesian Learning with Information Gain Provably Bounds Risk for a Robust Adversarial Defense**

具有信息增益的贝叶斯学习被证明是强健对抗防御的风险界限 cs.LG

Published at ICML 2022. Code is available at  https://github.com/baogiadoan/IG-BNN

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.02003v2) [paper-pdf](http://arxiv.org/pdf/2212.02003v2)

**Authors**: Bao Gia Doan, Ehsan Abbasnejad, Javen Qinfeng Shi, Damith C. Ranasinghe

**Abstract**: We present a new algorithm to learn a deep neural network model robust against adversarial attacks. Previous algorithms demonstrate an adversarially trained Bayesian Neural Network (BNN) provides improved robustness. We recognize the adversarial learning approach for approximating the multi-modal posterior distribution of a Bayesian model can lead to mode collapse; consequently, the model's achievements in robustness and performance are sub-optimal. Instead, we first propose preventing mode collapse to better approximate the multi-modal posterior distribution. Second, based on the intuition that a robust model should ignore perturbations and only consider the informative content of the input, we conceptualize and formulate an information gain objective to measure and force the information learned from both benign and adversarial training instances to be similar. Importantly. we prove and demonstrate that minimizing the information gain objective allows the adversarial risk to approach the conventional empirical risk. We believe our efforts provide a step toward a basis for a principled method of adversarially training BNNs. Our model demonstrate significantly improved robustness--up to 20%--compared with adversarial training and Adv-BNN under PGD attacks with 0.035 distortion on both CIFAR-10 and STL-10 datasets.

摘要: 提出了一种新的学习深度神经网络模型的算法，该模型具有较强的抗攻击能力。以前的算法表明，反向训练的贝叶斯神经网络(BNN)提供了更好的鲁棒性。我们认识到，用对抗性学习方法来逼近贝叶斯模型的多模式后验分布可能会导致模式崩溃，因此，该模型在稳健性和性能方面的成就是次优的。相反，我们首先提出防止模式崩溃，以更好地逼近多模式的后验分布。其次，基于健壮模型应该忽略扰动而只考虑输入的信息内容的直觉，我们概念化和制定了一个信息增益目标来衡量和强制从良性和对抗性训练实例中学习到的信息相似。重要的是。我们证明并证明了最小化信息收益目标使对手风险接近于传统的经验风险。我们相信，我们的努力为对抗性训练BNN的原则性方法奠定了基础。在CIFAR-10和STL-10数据集上，与对抗性训练和ADV-BNN相比，在具有0.035失真的PGD攻击下，我们的模型表现出了高达20%的健壮性。



## **40. Security Defense of Large Scale Networks Under False Data Injection Attacks: An Attack Detection Scheduling Approach**

虚假数据注入攻击下的大规模网络安全防御：一种攻击检测调度方法 eess.SY

14 pages, 11 figures

**SubmitDate**: 2023-12-01    [abs](http://arxiv.org/abs/2212.05500v3) [paper-pdf](http://arxiv.org/pdf/2212.05500v3)

**Authors**: Yuhan Suo, Senchun Chai, Runqi Chai, Zhong-Hua Pang, Yuanqing Xia, Guo-Ping Liu

**Abstract**: In large-scale networks, communication links between nodes are easily injected with false data by adversaries. This paper proposes a novel security defense strategy from the perspective of attack detection scheduling to ensure the security of the network. Based on the proposed strategy, each sensor can directly exclude suspicious sensors from its neighboring set. First, the problem of selecting suspicious sensors is formulated as a combinatorial optimization problem, which is non-deterministic polynomial-time hard (NP-hard). To solve this problem, the original function is transformed into a submodular function. Then, we propose a distributed attack detection scheduling algorithm based on the sequential submodular optimization theory, which incorporates \emph{expert problem} to better utilize historical information to guide the sensor selection task at the current moment. For different attack strategies, theoretical results show that the average optimization rate of the proposed algorithm has a lower bound, and the error expectation for any subset is bounded. In addition, under two kinds of insecurity conditions, the proposed algorithm can guarantee the security of the entire network from the perspective of the augmented estimation error. Finally, the effectiveness of the proposed method is verified by the numerical simulation and practical experiment.

摘要: 在大规模网络中，节点之间的通信链路很容易被对手注入虚假数据。本文从攻击检测调度的角度提出了一种新的安全防御策略，以确保网络的安全。基于该策略，每个传感器可以直接从其相邻集合中排除可疑传感器。首先，将可疑传感器的选择问题描述为一个非确定多项式时间难(NP-Hard)的组合优化问题。为了解决这一问题，将原函数转化为子模函数。在此基础上，提出了一种基于序贯子模优化理论的分布式攻击检测调度算法，该算法结合专家问题，更好地利用历史信息指导当前时刻的传感器选择任务。理论结果表明，对于不同的攻击策略，该算法的平均最优率有一个下界，并且对任何子集的误差期望都是有界的。另外，在两种不安全情况下，从估计误差增大的角度来看，该算法可以保证整个网络的安全性。最后，通过数值仿真和实际实验验证了该方法的有效性。



## **41. SPAM: Secure & Private Aircraft Management**

SPAM：安全和私人飞机管理 cs.CR

6 pages

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00245v1) [paper-pdf](http://arxiv.org/pdf/2312.00245v1)

**Authors**: Yaman Jandali, Nojan Sheybani, Farinaz Koushanfar

**Abstract**: With the rising use of aircrafts for operations ranging from disaster-relief to warfare, there is a growing risk of adversarial attacks. Malicious entities often only require the location of the aircraft for these attacks. Current satellite-aircraft communication and tracking protocols put aircrafts at risk if the satellite is compromised, due to computation being done in plaintext. In this work, we present \texttt{SPAM}, a private, secure, and accurate system that allows satellites to efficiently manage and maintain tracking angles for aircraft fleets without learning aircrafts' locations. \texttt{SPAM} is built upon multi-party computation and zero-knowledge proofs to guarantee privacy and high efficiency. While catered towards aircrafts, \texttt{SPAM}'s zero-knowledge fleet management can be easily extended to the IoT, with very little overhead.

摘要: 随着越来越多的飞机被用于从救灾到战争的各种行动，发生对抗性攻击的风险越来越大。恶意实体通常只需要飞机的位置就可以进行这些攻击。目前的卫星-飞机通信和跟踪协议使飞机面临风险，如果卫星受到威胁，因为计算是以明文进行的。在这项工作中，我们介绍了\exttt{Spam}，这是一个私人、安全和准确的系统，允许卫星在不了解飞机位置的情况下高效地管理和维护飞机机队的跟踪角度。\exttt{Spam}基于多方计算和零知识证明，保证隐私和高效。在迎合飞机的同时，S的零知识机队管理可以很容易地扩展到物联网，而开销很小。



## **42. Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition**

忽略这个标题和HackAPrompt：通过全球规模的即时黑客竞赛揭露LLMS的系统性漏洞 cs.CR

34 pages, 8 figures Codebase:  https://github.com/PromptLabs/hackaprompt Dataset:  https://huggingface.co/datasets/hackaprompt/hackaprompt-dataset/blob/main/README.md  Playground: https://huggingface.co/spaces/hackaprompt/playground

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.16119v2) [paper-pdf](http://arxiv.org/pdf/2311.16119v2)

**Authors**: Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Liu Kost, Christopher Carnahan, Jordan Boyd-Graber

**Abstract**: Large Language Models (LLMs) are deployed in interactive contexts with direct user engagement, such as chatbots and writing assistants. These deployments are vulnerable to prompt injection and jailbreaking (collectively, prompt hacking), in which models are manipulated to ignore their original instructions and follow potentially malicious ones. Although widely acknowledged as a significant security threat, there is a dearth of large-scale resources and quantitative studies on prompt hacking. To address this lacuna, we launch a global prompt hacking competition, which allows for free-form human input attacks. We elicit 600K+ adversarial prompts against three state-of-the-art LLMs. We describe the dataset, which empirically verifies that current LLMs can indeed be manipulated via prompt hacking. We also present a comprehensive taxonomical ontology of the types of adversarial prompts.

摘要: 大型语言模型(LLM)部署在具有直接用户参与的交互上下文中，例如聊天机器人和写作助手。这些部署容易受到即时注入和越狱(统称为即时黑客)的攻击，在这些情况下，模型被操纵以忽略其原始指令并遵循潜在的恶意指令。尽管被广泛认为是一个重大的安全威胁，但缺乏关于即时黑客攻击的大规模资源和量化研究。为了弥补这一漏洞，我们发起了一场全球即时黑客竞赛，允许自由形式的人工输入攻击。我们在三个最先进的LLM上获得了600K+的对抗性提示。我们描述了数据集，这从经验上验证了当前的LLM确实可以通过即时黑客来操纵。我们还提出了对抗性提示类型的全面分类本体。



## **43. Optimal Attack and Defense for Reinforcement Learning**

强化学习的最优攻防 cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00198v1) [paper-pdf](http://arxiv.org/pdf/2312.00198v1)

**Authors**: Jeremy McMahan, Young Wu, Xiaojin Zhu, Qiaomin Xie

**Abstract**: To ensure the usefulness of Reinforcement Learning (RL) in real systems, it is crucial to ensure they are robust to noise and adversarial attacks. In adversarial RL, an external attacker has the power to manipulate the victim agent's interaction with the environment. We study the full class of online manipulation attacks, which include (i) state attacks, (ii) observation attacks (which are a generalization of perceived-state attacks), (iii) action attacks, and (iv) reward attacks. We show the attacker's problem of designing a stealthy attack that maximizes its own expected reward, which often corresponds to minimizing the victim's value, is captured by a Markov Decision Process (MDP) that we call a meta-MDP since it is not the true environment but a higher level environment induced by the attacked interaction. We show that the attacker can derive optimal attacks by planning in polynomial time or learning with polynomial sample complexity using standard RL techniques. We argue that the optimal defense policy for the victim can be computed as the solution to a stochastic Stackelberg game, which can be further simplified into a partially-observable turn-based stochastic game (POTBSG). Neither the attacker nor the victim would benefit from deviating from their respective optimal policies, thus such solutions are truly robust. Although the defense problem is NP-hard, we show that optimal Markovian defenses can be computed (learned) in polynomial time (sample complexity) in many scenarios.

摘要: 为了确保强化学习(RL)在实际系统中的有效性，确保它们对噪声和对手攻击具有健壮性是至关重要的。在对抗性RL中，外部攻击者有权操纵受害者代理与环境的交互。我们研究了所有类型的在线操纵攻击，包括(I)状态攻击，(Ii)观察攻击(它是感知状态攻击的推广)，(Iii)动作攻击，和(Iv)奖励攻击。我们展示了攻击者设计最大化自身期望回报的隐形攻击的问题，这通常对应于最小化受害者的价值，被马尔可夫决策过程(MDP)捕获，我们称之为元MDP，因为它不是真正的环境，而是由攻击交互引起的更高级别的环境。我们证明了攻击者可以通过在多项式时间内进行规划或使用标准RL技术以多项式样本复杂性学习来获得最优攻击。我们认为，受害者的最优防御策略可以归结为一个随机Stackelberg博弈的解，它可以进一步简化为一个部分可观测的基于回合的随机博弈(POTBSG)。攻击者和受害者都不会从偏离各自的最优策略中受益，因此这样的解决方案是真正可靠的。虽然防御问题是NP难的，但我们证明了在许多情况下，最优马尔可夫防御可以在多项式时间(样本复杂性)内计算(学习)。



## **44. Fool the Hydra: Adversarial Attacks against Multi-view Object Detection Systems**

愚弄九头蛇：针对多视点目标检测系统的对抗性攻击 cs.CV

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00173v1) [paper-pdf](http://arxiv.org/pdf/2312.00173v1)

**Authors**: Bilel Tarchoun, Quazi Mishkatul Alam, Nael Abu-Ghazaleh, Ihsen Alouani

**Abstract**: Adversarial patches exemplify the tangible manifestation of the threat posed by adversarial attacks on Machine Learning (ML) models in real-world scenarios. Robustness against these attacks is of the utmost importance when designing computer vision applications, especially for safety-critical domains such as CCTV systems. In most practical situations, monitoring open spaces requires multi-view systems to overcome acquisition challenges such as occlusion handling. Multiview object systems are able to combine data from multiple views, and reach reliable detection results even in difficult environments. Despite its importance in real-world vision applications, the vulnerability of multiview systems to adversarial patches is not sufficiently investigated. In this paper, we raise the following question: Does the increased performance and information sharing across views offer as a by-product robustness to adversarial patches? We first conduct a preliminary analysis showing promising robustness against off-the-shelf adversarial patches, even in an extreme setting where we consider patches applied to all views by all persons in Wildtrack benchmark. However, we challenged this observation by proposing two new attacks: (i) In the first attack, targeting a multiview CNN, we maximize the global loss by proposing gradient projection to the different views and aggregating the obtained local gradients. (ii) In the second attack, we focus on a Transformer-based multiview framework. In addition to the focal loss, we also maximize the transformer-specific loss by dissipating its attention blocks. Our results show a large degradation in the detection performance of victim multiview systems with our first patch attack reaching an attack success rate of 73% , while our second proposed attack reduced the performance of its target detector by 62%

摘要: 对抗性补丁例证了现实世界场景中对抗性攻击对机器学习(ML)模型构成的威胁的有形表现。在设计计算机视觉应用程序时，对这些攻击的健壮性至关重要，特别是对于安全关键领域，如闭路电视系统。在大多数实际情况下，监控开放空间需要多视角系统来克服采集挑战，如遮挡处理。多视点目标系统能够组合来自多个视点的数据，即使在困难的环境中也能达到可靠的检测结果。尽管多视点系统在现实世界的视觉应用中具有重要的意义，但多视点系统对敌意补丁的脆弱性尚未得到充分的研究。在这篇文章中，我们提出了以下问题：增加的性能和跨视图的信息共享是否作为副产品提供了对对手补丁的健壮性？我们首先进行了初步分析，展示了对现成的敌意补丁具有良好的稳健性，即使在我们认为WildTrack基准测试中所有人的所有视图都应用了补丁的极端环境中也是如此。然而，我们通过提出两个新的攻击来挑战这一观察结果：(I)在第一个攻击中，针对多视点CNN，我们通过向不同的视点提出梯度投影并聚合获得的局部梯度来最大化全局损失。(Ii)在第二个攻击中，我们重点介绍了基于Transformer的多视图框架。除了焦点损耗，我们还通过分散其注意力块来最大化特定于变压器的损耗。结果表明，第一次补丁攻击使受害者多视角系统的检测性能大幅下降，攻击成功率达到73%，而第二次补丁攻击使目标检测器的性能下降了62%



## **45. Universal Backdoor Attacks**

通用后门攻击 cs.LG

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00157v1) [paper-pdf](http://arxiv.org/pdf/2312.00157v1)

**Authors**: Benjamin Schneider, Nils Lukas, Florian Kerschbaum

**Abstract**: Web-scraped datasets are vulnerable to data poisoning, which can be used for backdooring deep image classifiers during training. Since training on large datasets is expensive, a model is trained once and re-used many times. Unlike adversarial examples, backdoor attacks often target specific classes rather than any class learned by the model. One might expect that targeting many classes through a naive composition of attacks vastly increases the number of poison samples. We show this is not necessarily true and more efficient, universal data poisoning attacks exist that allow controlling misclassifications from any source class into any target class with a small increase in poison samples. Our idea is to generate triggers with salient characteristics that the model can learn. The triggers we craft exploit a phenomenon we call inter-class poison transferability, where learning a trigger from one class makes the model more vulnerable to learning triggers for other classes. We demonstrate the effectiveness and robustness of our universal backdoor attacks by controlling models with up to 6,000 classes while poisoning only 0.15% of the training dataset.

摘要: 网络抓取的数据集很容易受到数据中毒的影响，在训练过程中，数据中毒可以用于回溯深度图像分类器。由于在大型数据集上进行训练的成本很高，因此一个模型只需训练一次，就可以多次重复使用。与对抗性示例不同，后门攻击通常针对特定类，而不是模型学习到的任何类。人们可能会认为，通过天真的攻击组合以许多类别为目标会极大地增加毒物样本的数量。我们证明这不一定是真的，而且更有效，普遍存在的数据中毒攻击允许在毒物样本少量增加的情况下控制从任何源类到任何目标类的误分类。我们的想法是生成模型可以学习的具有显著特征的触发器。我们制作的触发器利用了一种我们称为类间毒药可转移性的现象，即从一个类学习触发器使模型更容易学习其他类的触发器。我们通过控制多达6,000个类的模型来展示我们的通用后门攻击的有效性和健壮性，而只毒化了0.15%的训练数据集。



## **46. On the Adversarial Robustness of Graph Contrastive Learning Methods**

关于图对比学习方法的对抗稳健性 cs.LG

Accepted at NeurIPS 2023 New Frontiers in Graph Learning Workshop  (NeurIPS GLFrontiers 2023)

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.17853v2) [paper-pdf](http://arxiv.org/pdf/2311.17853v2)

**Authors**: Filippo Guerranti, Zinuo Yi, Anna Starovoit, Rafiq Kamel, Simon Geisler, Stephan Günnemann

**Abstract**: Contrastive learning (CL) has emerged as a powerful framework for learning representations of images and text in a self-supervised manner while enhancing model robustness against adversarial attacks. More recently, researchers have extended the principles of contrastive learning to graph-structured data, giving birth to the field of graph contrastive learning (GCL). However, whether GCL methods can deliver the same advantages in adversarial robustness as their counterparts in the image and text domains remains an open question. In this paper, we introduce a comprehensive robustness evaluation protocol tailored to assess the robustness of GCL models. We subject these models to adaptive adversarial attacks targeting the graph structure, specifically in the evasion scenario. We evaluate node and graph classification tasks using diverse real-world datasets and attack strategies. With our work, we aim to offer insights into the robustness of GCL methods and hope to open avenues for potential future research directions.

摘要: 对比学习(CL)已经成为一种强大的框架，用于以自我监督的方式学习图像和文本的表示，同时增强模型对对手攻击的稳健性。最近，研究人员将对比学习的原理扩展到图结构的数据，从而诞生了图对比学习(GCL)领域。然而，GCL方法在对抗稳健性方面是否能提供与其在图像和文本领域的对应方法相同的优势仍然是一个悬而未决的问题。在本文中，我们介绍了一个全面的健壮性评估协议，以评估GCL模型的健壮性。我们使这些模型受到针对图结构的自适应对抗性攻击，特别是在逃避场景中。我们使用不同的真实数据集和攻击策略来评估节点和图分类任务。通过我们的工作，我们旨在为GCL方法的稳健性提供见解，并希望为未来潜在的研究方向开辟道路。



## **47. Adversarial Attacks and Defenses for Wireless Signal Classifiers using CDI-aware GANs**

基于CDI感知Gans的无线信号分类器的对抗性攻击与防御 cs.IT

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2311.18820v1) [paper-pdf](http://arxiv.org/pdf/2311.18820v1)

**Authors**: Sujata Sinha, Alkan Soysal

**Abstract**: We introduce a Channel Distribution Information (CDI)-aware Generative Adversarial Network (GAN), designed to address the unique challenges of adversarial attacks in wireless communication systems. The generator in this CDI-aware GAN maps random input noise to the feature space, generating perturbations intended to deceive a target modulation classifier. Its discriminators play a dual role: one enforces that the perturbations follow a Gaussian distribution, making them indistinguishable from Gaussian noise, while the other ensures these perturbations account for realistic channel effects and resemble no-channel perturbations.   Our proposed CDI-aware GAN can be used as an attacker and a defender. In attack scenarios, the CDI-aware GAN demonstrates its prowess by generating robust adversarial perturbations that effectively deceive the target classifier, outperforming known methods. Furthermore, CDI-aware GAN as a defender significantly improves the target classifier's resilience against adversarial attacks.

摘要: 我们介绍了一种基于信道分布信息(CDI)的生成性对抗网络(GAN)，旨在解决无线通信系统中对抗攻击的独特挑战。这种CDI感知的GaN中的生成器将随机输入噪声映射到特征空间，生成旨在欺骗目标调制分类器的扰动。它的鉴别器扮演着双重角色：一个强制扰动服从高斯分布，使它们与高斯噪声无法区分，另一个确保这些扰动考虑到真实的信道影响并类似于无信道扰动。我们建议的CDI感知GAN可以用作攻击者和防御者。在攻击场景中，支持CDI的GAN通过生成强健的对抗性扰动来展示其能力，从而有效地欺骗目标分类器，性能优于已知的方法。此外，CDI感知的GAN作为防御者显著提高了目标分类器对对手攻击的弹性。



## **48. Improving the Robustness of Quantized Deep Neural Networks to White-Box Attacks using Stochastic Quantization and Information-Theoretic Ensemble Training**

利用随机量化和信息论集成训练提高量化深度神经网络对白盒攻击的稳健性 cs.CV

9 pages, 9 figures, 4 tables

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2312.00105v1) [paper-pdf](http://arxiv.org/pdf/2312.00105v1)

**Authors**: Saurabh Farkya, Aswin Raghavan, Avi Ziskind

**Abstract**: Most real-world applications that employ deep neural networks (DNNs) quantize them to low precision to reduce the compute needs. We present a method to improve the robustness of quantized DNNs to white-box adversarial attacks. We first tackle the limitation of deterministic quantization to fixed ``bins'' by introducing a differentiable Stochastic Quantizer (SQ). We explore the hypothesis that different quantizations may collectively be more robust than each quantized DNN. We formulate a training objective to encourage different quantized DNNs to learn different representations of the input image. The training objective captures diversity and accuracy via mutual information between ensemble members. Through experimentation, we demonstrate substantial improvement in robustness against $L_\infty$ attacks even if the attacker is allowed to backpropagate through SQ (e.g., > 50\% accuracy to PGD(5/255) on CIFAR10 without adversarial training), compared to vanilla DNNs as well as existing ensembles of quantized DNNs. We extend the method to detect attacks and generate robustness profiles in the adversarial information plane (AIP), towards a unified analysis of different threat models by correlating the MI and accuracy.

摘要: 大多数使用深度神经网络(DNN)的现实世界应用程序将它们量化到低精度，以减少计算需求。提出了一种提高量化DNN对白盒攻击的鲁棒性的方法。我们首先通过引入一种可微随机量化器(SQ)来解决确定性量化对固定‘箱’的限制。我们探索了这样的假设，即不同的量化可能共同比每个量化的DNN更稳健。我们制定了一个训练目标，以鼓励不同的量化DNN学习输入图像的不同表示。训练目标通过集合成员之间的互信息来捕捉多样性和准确性。通过实验表明，与普通的DNN和现有的量化DNN集成相比，即使允许攻击者通过SQ反向传播(例如，在CIFAR10上对PGD(5/255)的准确率>50\%)，我们也表现出对$L_INFTY$攻击的鲁棒性显著提高。我们将该方法扩展到检测攻击并在对抗信息平面(AIP)中生成健壮性配置文件，通过关联MI和准确性来统一分析不同的威胁模型。



## **49. Differentiable JPEG: The Devil is in the Details**

上一篇：JPEG：魔鬼在细节中 cs.CV

Accepted at WACV 2024. Project page:  https://christophreich1996.github.io/differentiable_jpeg/

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2309.06978v3) [paper-pdf](http://arxiv.org/pdf/2309.06978v3)

**Authors**: Christoph Reich, Biplob Debnath, Deep Patel, Srimat Chakradhar

**Abstract**: JPEG remains one of the most widespread lossy image coding methods. However, the non-differentiable nature of JPEG restricts the application in deep learning pipelines. Several differentiable approximations of JPEG have recently been proposed to address this issue. This paper conducts a comprehensive review of existing diff. JPEG approaches and identifies critical details that have been missed by previous methods. To this end, we propose a novel diff. JPEG approach, overcoming previous limitations. Our approach is differentiable w.r.t. the input image, the JPEG quality, the quantization tables, and the color conversion parameters. We evaluate the forward and backward performance of our diff. JPEG approach against existing methods. Additionally, extensive ablations are performed to evaluate crucial design choices. Our proposed diff. JPEG resembles the (non-diff.) reference implementation best, significantly surpassing the recent-best diff. approach by $3.47$dB (PSNR) on average. For strong compression rates, we can even improve PSNR by $9.51$dB. Strong adversarial attack results are yielded by our diff. JPEG, demonstrating the effective gradient approximation. Our code is available at https://github.com/necla-ml/Diff-JPEG.

摘要: JPEG仍然是应用最广泛的有损图像编码方法之一。然而，JPEG的不可微特性限制了其在深度学习管道中的应用。为了解决这个问题，最近已经提出了几种JPEG的可微近似。本文对现有的DIFF进行了全面的回顾。JPEG处理并确定了以前方法遗漏的关键细节。为此，我们提出了一个新颖的Diff。JPEG方法，克服了以前的限制。我们的方法是可微的W.r.t。输入图像、JPEG质量、量化表和颜色转换参数。我们评估了DIFF的向前和向后性能。JPEG方法与现有方法的对比。此外，还进行了广泛的消融，以评估关键的设计选择。我们提议的不同之处。JPEG与(Non-Diff.)参考实现最好，大大超过了最近最好的差异。平均接近3.47美元分贝(PSNR)。对于强压缩率，我们甚至可以将PSNR提高9.51美元分贝。强大的对抗性攻击结果是由我们的差异产生的。JPEG格式，演示了有效的渐变近似。我们的代码可以在https://github.com/necla-ml/Diff-JPEG.上找到



## **50. Diffusion Models for Imperceptible and Transferable Adversarial Attack**

不可察觉和可转移对抗性攻击的扩散模型 cs.CV

Code Page: https://github.com/WindVChen/DiffAttack. In Paper Version  v2, we incorporate more discussions and experiments

**SubmitDate**: 2023-11-30    [abs](http://arxiv.org/abs/2305.08192v2) [paper-pdf](http://arxiv.org/pdf/2305.08192v2)

**Authors**: Jianqi Chen, Hao Chen, Keyan Chen, Yilan Zhang, Zhengxia Zou, Zhenwei Shi

**Abstract**: Many existing adversarial attacks generate $L_p$-norm perturbations on image RGB space. Despite some achievements in transferability and attack success rate, the crafted adversarial examples are easily perceived by human eyes. Towards visual imperceptibility, some recent works explore unrestricted attacks without $L_p$-norm constraints, yet lacking transferability of attacking black-box models. In this work, we propose a novel imperceptible and transferable attack by leveraging both the generative and discriminative power of diffusion models. Specifically, instead of direct manipulation in pixel space, we craft perturbations in the latent space of diffusion models. Combined with well-designed content-preserving structures, we can generate human-insensitive perturbations embedded with semantic clues. For better transferability, we further "deceive" the diffusion model which can be viewed as an implicit recognition surrogate, by distracting its attention away from the target regions. To our knowledge, our proposed method, DiffAttack, is the first that introduces diffusion models into the adversarial attack field. Extensive experiments on various model structures, datasets, and defense methods have demonstrated the superiority of our attack over the existing attack methods.

摘要: 许多现有的对抗性攻击在图像RGB空间上产生$L_p$-范数扰动。尽管在可转移性和攻击成功率方面取得了一些成就，但制作的对抗性例子很容易被人眼察觉。对于视觉不可感知性，最近的一些工作探索了没有$L_p$-范数约束的无限攻击，但缺乏攻击黑盒模型的可转移性。在这项工作中，我们提出了一种新的不可察觉和可转移的攻击，利用扩散模型的生成性和区分性。具体地说，我们不是在像素空间中直接操作，而是在扩散模型的潜在空间中制造扰动。与设计良好的内容保持结构相结合，我们可以生成嵌入语义线索的人类不敏感的扰动。为了获得更好的可转移性，我们通过将扩散模型的注意力从目标区域转移开，进一步欺骗了可以被视为隐式识别代理的扩散模型。据我们所知，我们提出的DiffAttack方法首次将扩散模型引入到对抗性攻击领域。在各种模型结构、数据集和防御方法上的广泛实验证明了该攻击相对于现有攻击方法的优越性。



