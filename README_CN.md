# Latest Adversarial Attack Papers
**update at 2024-05-13 15:28:23**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Certified $\ell_2$ Attribution Robustness via Uniformly Smoothed Attributions**

通过均匀平滑的归因认证$\ell_2$归因稳健性 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06361v1) [paper-pdf](http://arxiv.org/pdf/2405.06361v1)

**Authors**: Fan Wang, Adams Wai-Kin Kong

**Abstract**: Model attribution is a popular tool to explain the rationales behind model predictions. However, recent work suggests that the attributions are vulnerable to minute perturbations, which can be added to input samples to fool the attributions while maintaining the prediction outputs. Although empirical studies have shown positive performance via adversarial training, an effective certified defense method is eminently needed to understand the robustness of attributions. In this work, we propose to use uniform smoothing technique that augments the vanilla attributions by noises uniformly sampled from a certain space. It is proved that, for all perturbations within the attack region, the cosine similarity between uniformly smoothed attribution of perturbed sample and the unperturbed sample is guaranteed to be lower bounded. We also derive alternative formulations of the certification that is equivalent to the original one and provides the maximum size of perturbation or the minimum smoothing radius such that the attribution can not be perturbed. We evaluate the proposed method on three datasets and show that the proposed method can effectively protect the attributions from attacks, regardless of the architecture of networks, training schemes and the size of the datasets.

摘要: 模型归因是解释模型预测背后的理论基础的流行工具。然而，最近的工作表明，属性容易受到微小扰动的影响，可以将这些微小扰动添加到输入样本中，以在保持预测输出的同时愚弄属性。虽然实证研究表明，通过对抗性训练取得了积极的效果，但需要一种有效的认证防御方法来理解归因的稳健性。在这项工作中，我们提出使用均匀平滑技术，通过从特定空间均匀采样的噪声来增强香草属性。证明了对于攻击区域内的所有扰动，扰动样本的一致光滑属性与未扰动样本的余弦相似保证是下界的。我们还推导出了与原始证明等价的证明的替代公式，并且提供了最大扰动大小或最小光滑半径，使得属性不能被扰动。我们在三个数据集上对该方法进行了评估，结果表明，无论网络结构、训练方案和数据集的大小如何，该方法都能有效地保护属性免受攻击。



## **2. Evaluating Adversarial Robustness in the Spatial Frequency Domain**

空间频域中的对抗鲁棒性评估 cs.CV

14 pages

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06345v1) [paper-pdf](http://arxiv.org/pdf/2405.06345v1)

**Authors**: Keng-Hsin Liao, Chin-Yuan Yeh, Hsi-Wen Chen, Ming-Syan Chen

**Abstract**: Convolutional Neural Networks (CNNs) have dominated the majority of computer vision tasks. However, CNNs' vulnerability to adversarial attacks has raised concerns about deploying these models to safety-critical applications. In contrast, the Human Visual System (HVS), which utilizes spatial frequency channels to process visual signals, is immune to adversarial attacks. As such, this paper presents an empirical study exploring the vulnerability of CNN models in the frequency domain. Specifically, we utilize the discrete cosine transform (DCT) to construct the Spatial-Frequency (SF) layer to produce a block-wise frequency spectrum of an input image and formulate Spatial Frequency CNNs (SF-CNNs) by replacing the initial feature extraction layers of widely-used CNN backbones with the SF layer. Through extensive experiments, we observe that SF-CNN models are more robust than their CNN counterparts under both white-box and black-box attacks. To further explain the robustness of SF-CNNs, we compare the SF layer with a trainable convolutional layer with identical kernel sizes using two mixing strategies to show that the lower frequency components contribute the most to the adversarial robustness of SF-CNNs. We believe our observations can guide the future design of robust CNN models.

摘要: 卷积神经网络(CNN)已经主导了计算机视觉的大部分任务。然而，CNN对对手攻击的脆弱性已经引起了人们对将这些模型部署到安全关键应用程序的担忧。相比之下，人类视觉系统(HVS)利用空间频率通道来处理视觉信号，不受对手攻击。因此，本文提出了一项实证研究，探索CNN模型在频域中的脆弱性。具体地说，我们利用离散余弦变换(DCT)来构造空间频率(SF)层来产生输入图像的块状频谱，并通过用SF层替换广泛使用的CNN骨干的初始特征提取层来构造空间频率CNN(SF-CNN)。通过大量的实验，我们观察到SF-CNN模型在白盒和黑盒攻击下都比CNN模型更健壮。为了进一步解释SF-CNN的健壮性，我们使用两种混合策略将SF层与具有相同核大小的可训练卷积层进行了比较，结果表明低频分量对SF-CNN的对抗健壮性贡献最大。我们相信，我们的观察可以指导未来稳健的CNN模型的设计。



## **3. Improving Transferable Targeted Adversarial Attack via Normalized Logit Calibration and Truncated Feature Mixing**

通过规范化Logit校准和截断特征混合改进可转移有针对性的对抗攻击 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06340v1) [paper-pdf](http://arxiv.org/pdf/2405.06340v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: This paper aims to enhance the transferability of adversarial samples in targeted attacks, where attack success rates remain comparatively low. To achieve this objective, we propose two distinct techniques for improving the targeted transferability from the loss and feature aspects. First, in previous approaches, logit calibrations used in targeted attacks primarily focus on the logit margin between the targeted class and the untargeted classes among samples, neglecting the standard deviation of the logit. In contrast, we introduce a new normalized logit calibration method that jointly considers the logit margin and the standard deviation of logits. This approach effectively calibrates the logits, enhancing the targeted transferability. Second, previous studies have demonstrated that mixing the features of clean samples during optimization can significantly increase transferability. Building upon this, we further investigate a truncated feature mixing method to reduce the impact of the source training model, resulting in additional improvements. The truncated feature is determined by removing the Rank-1 feature associated with the largest singular value decomposed from the high-level convolutional layers of the clean sample. Extensive experiments conducted on the ImageNet-Compatible and CIFAR-10 datasets demonstrate the individual and mutual benefits of our proposed two components, which outperform the state-of-the-art methods by a large margin in black-box targeted attacks.

摘要: 本文旨在提高攻击成功率相对较低的定向攻击中对抗性样本的可转移性。为了实现这一目标，我们从损失和特征两个方面提出了两种不同的技术来提高目标可转移性。首先，在以往的方法中，用于目标攻击的Logit校准主要集中在样本中目标类和非目标类之间的Logit差值，而忽略了Logit的标准差。相反，我们引入了一种新的归一化Logit校准方法，该方法同时考虑了Logit裕度和Logit的标准差。这种方法有效地校准了LOGITS，增强了目标可转移性。其次，以往的研究表明，在优化过程中混合清洁样本的特征可以显著提高可转移性。在此基础上，我们进一步研究了一种截断特征混合方法，以减少源训练模型的影响，从而得到进一步的改进。通过去除与从清洁样本的高级卷积层分解的最大奇异值相关联的Rank-1特征来确定截断特征。在ImageNet兼容和CIFAR-10数据集上进行的广泛实验表明，我们提出的两个组件具有单独和共同的好处，在黑盒定向攻击中远远超过最先进的方法。



## **4. PUMA: margin-based data pruning**

SEARCH A：基于利润的数据修剪 cs.LG

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06298v1) [paper-pdf](http://arxiv.org/pdf/2405.06298v1)

**Authors**: Javier Maroto, Pascal Frossard

**Abstract**: Deep learning has been able to outperform humans in terms of classification accuracy in many tasks. However, to achieve robustness to adversarial perturbations, the best methodologies require to perform adversarial training on a much larger training set that has been typically augmented using generative models (e.g., diffusion models). Our main objective in this work, is to reduce these data requirements while achieving the same or better accuracy-robustness trade-offs. We focus on data pruning, where some training samples are removed based on the distance to the model classification boundary (i.e., margin). We find that the existing approaches that prune samples with low margin fails to increase robustness when we add a lot of synthetic data, and explain this situation with a perceptron learning task. Moreover, we find that pruning high margin samples for better accuracy increases the harmful impact of mislabeled perturbed data in adversarial training, hurting both robustness and accuracy. We thus propose PUMA, a new data pruning strategy that computes the margin using DeepFool, and prunes the training samples of highest margin without hurting performance by jointly adjusting the training attack norm on the samples of lowest margin. We show that PUMA can be used on top of the current state-of-the-art methodology in robustness, and it is able to significantly improve the model performance unlike the existing data pruning strategies. Not only PUMA achieves similar robustness with less data, but it also significantly increases the model accuracy, improving the performance trade-off.

摘要: 在许多任务中，深度学习在分类准确率方面已经能够超过人类。然而，为了实现对对抗性扰动的稳健性，最好的方法需要在通常使用生成模型(例如，扩散模型)扩充的大得多的训练集上执行对抗性训练。我们在这项工作中的主要目标是减少这些数据要求，同时实现相同或更好的精度-稳健性权衡。我们的重点是数据剪枝，即根据到模型分类边界的距离(即边界)来删除一些训练样本。我们发现，当我们添加大量的合成数据时，现有的对低边际样本进行剪枝的方法不能提高鲁棒性，并用感知器学习任务来解释这种情况。此外，我们发现，为了更好的准确性而修剪高边缘样本会增加错误标记的扰动数据在对抗性训练中的有害影响，损害稳健性和准确性。因此，我们提出了一种新的数据剪枝策略PUMA，它使用DeepFool计算差值，并在差值最小的样本上联合调整训练攻击范数，在不影响性能的情况下修剪差值最高的训练样本。我们表明，PUMA可以在当前最先进的方法的健壮性上使用，并且它能够显著提高模型的性能，而不是现有的数据剪枝策略。PUMA不仅用更少的数据实现了类似的稳健性，而且还显著提高了模型的精度，改善了性能权衡。



## **5. Exploring the Interplay of Interpretability and Robustness in Deep Neural Networks: A Saliency-guided Approach**

探索深度神经网络中可解释性和鲁棒性的相互作用：显着性引导的方法 cs.CV

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06278v1) [paper-pdf](http://arxiv.org/pdf/2405.06278v1)

**Authors**: Amira Guesmi, Nishant Suresh Aswani, Muhammad Shafique

**Abstract**: Adversarial attacks pose a significant challenge to deploying deep learning models in safety-critical applications. Maintaining model robustness while ensuring interpretability is vital for fostering trust and comprehension in these models. This study investigates the impact of Saliency-guided Training (SGT) on model robustness, a technique aimed at improving the clarity of saliency maps to deepen understanding of the model's decision-making process. Experiments were conducted on standard benchmark datasets using various deep learning architectures trained with and without SGT. Findings demonstrate that SGT enhances both model robustness and interpretability. Additionally, we propose a novel approach combining SGT with standard adversarial training to achieve even greater robustness while preserving saliency map quality. Our strategy is grounded in the assumption that preserving salient features crucial for correctly classifying adversarial examples enhances model robustness, while masking non-relevant features improves interpretability. Our technique yields significant gains, achieving a 35\% and 20\% improvement in robustness against PGD attack with noise magnitudes of $0.2$ and $0.02$ for the MNIST and CIFAR-10 datasets, respectively, while producing high-quality saliency maps.

摘要: 对抗性攻击对在安全关键型应用中部署深度学习模型提出了重大挑战。在确保可解释性的同时保持模型的健壮性对于培养对这些模型的信任和理解至关重要。本研究调查显著引导训练(SGT)对模型稳健性的影响，这是一种旨在提高显著图的清晰度以加深对模型决策过程的理解的技术。实验在标准基准数据集上进行，使用各种深度学习体系结构，在有和没有SGT的情况下进行训练。研究结果表明，SGT既增强了模型的稳健性，又增强了模型的可解释性。此外，我们提出了一种结合SGT和标准对抗性训练的新方法，在保持显著图质量的同时获得更好的稳健性。我们的策略基于这样的假设，即保留对于正确分类对抗性示例至关重要的显著特征可以增强模型的稳健性，而屏蔽不相关的特征可以提高可解释性。我们的技术产生了显著的收益，在MNIST和CIFAR-10数据集的噪声幅度分别为0.2美元和0.02美元的情况下，对PGD攻击的稳健性分别提高了35%和20%，同时生成了高质量的显著图。



## **6. Disttack: Graph Adversarial Attacks Toward Distributed GNN Training**

区别：针对分布式GNN培训的图形对抗攻击 cs.LG

Accepted by 30th International European Conference on Parallel and  Distributed Computing(Euro-Par 2024)

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06247v1) [paper-pdf](http://arxiv.org/pdf/2405.06247v1)

**Authors**: Yuxiang Zhang, Xin Liu, Meng Wu, Wei Yan, Mingyu Yan, Xiaochun Ye, Dongrui Fan

**Abstract**: Graph Neural Networks (GNNs) have emerged as potent models for graph learning. Distributing the training process across multiple computing nodes is the most promising solution to address the challenges of ever-growing real-world graphs. However, current adversarial attack methods on GNNs neglect the characteristics and applications of the distributed scenario, leading to suboptimal performance and inefficiency in attacking distributed GNN training.   In this study, we introduce Disttack, the first framework of adversarial attacks for distributed GNN training that leverages the characteristics of frequent gradient updates in a distributed system. Specifically, Disttack corrupts distributed GNN training by injecting adversarial attacks into one single computing node. The attacked subgraphs are precisely perturbed to induce an abnormal gradient ascent in backpropagation, disrupting gradient synchronization between computing nodes and thus leading to a significant performance decline of the trained GNN. We evaluate Disttack on four large real-world graphs by attacking five widely adopted GNNs. Compared with the state-of-the-art attack method, experimental results demonstrate that Disttack amplifies the model accuracy degradation by 2.75$\times$ and achieves speedup by 17.33$\times$ on average while maintaining unnoticeability.

摘要: 图神经网络(GNN)已经成为图学习的有力模型。将训练过程分布在多个计算节点上是解决不断增长的真实世界图的挑战的最有前途的解决方案。然而，目前针对GNN的对抗性攻击方法忽略了分布式场景的特点和应用，导致在攻击分布式GNN训练时性能不佳且效率低下。在这项研究中，我们介绍了Disttack，这是第一个用于分布式GNN训练的对抗性攻击框架，它利用了分布式系统中频繁梯度更新的特点。具体地说，Disttack通过将敌意攻击注入到单个计算节点来破坏分布式GNN训练。被攻击的子图被精确地扰动，导致反向传播中的异常梯度上升，扰乱了计算节点之间的梯度同步，从而导致训练后的GNN的性能显著下降。我们通过攻击五个广泛使用的GNN来评估四个大型真实世界图上的Disttack。实验结果表明，与最新的攻击方法相比，Disttack在保持不可察觉的情况下，使模型的准确率降低了2.75倍，平均加速比提高了17.33倍。



## **7. Concealing Backdoor Model Updates in Federated Learning by Trigger-Optimized Data Poisoning**

通过触发优化的数据中毒隐藏联邦学习中后门模型更新 cs.CR

**SubmitDate**: 2024-05-10    [abs](http://arxiv.org/abs/2405.06206v1) [paper-pdf](http://arxiv.org/pdf/2405.06206v1)

**Authors**: Yujie Zhang, Neil Gong, Michael K. Reiter

**Abstract**: Federated Learning (FL) is a decentralized machine learning method that enables participants to collaboratively train a model without sharing their private data. Despite its privacy and scalability benefits, FL is susceptible to backdoor attacks, where adversaries poison the local training data of a subset of clients using a backdoor trigger, aiming to make the aggregated model produce malicious results when the same backdoor condition is met by an inference-time input. Existing backdoor attacks in FL suffer from common deficiencies: fixed trigger patterns and reliance on the assistance of model poisoning. State-of-the-art defenses based on Byzantine-robust aggregation exhibit a good defense performance on these attacks because of the significant divergence between malicious and benign model updates. To effectively conceal malicious model updates among benign ones, we propose DPOT, a backdoor attack strategy in FL that dynamically constructs backdoor objectives by optimizing a backdoor trigger, making backdoor data have minimal effect on model updates. We provide theoretical justifications for DPOT's attacking principle and display experimental results showing that DPOT, via only a data-poisoning attack, effectively undermines state-of-the-art defenses and outperforms existing backdoor attack techniques on various datasets.

摘要: 联合学习(FL)是一种去中心化的机器学习方法，允许参与者在不共享私人数据的情况下协作训练模型。尽管FL具有隐私和可扩展性方面的优势，但它很容易受到后门攻击，即攻击者使用后门触发器毒化部分客户端的本地训练数据，目的是在推理时输入满足相同的后门条件时，使聚合模型产生恶意结果。FL中现有的后门攻击存在共同的缺陷：固定的触发模式和依赖模型中毒的辅助。由于恶意模型更新和良性模型更新之间的显著差异，基于拜占庭稳健聚合的最新防御技术在这些攻击中表现出良好的防御性能。为了有效地隐藏良性模型更新中的恶意模型更新，我们提出了一种FL中的后门攻击策略DPOT，它通过优化后门触发器来动态构建后门目标，使后门数据对模型更新的影响最小。我们为DPOT的攻击原理提供了理论依据，并展示了实验结果表明，DPOT仅通过一次数据中毒攻击就可以有效地破坏最先进的防御措施，并在各种数据集上优于现有的后门攻击技术。



## **8. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06134v1) [paper-pdf](http://arxiv.org/pdf/2405.06134v1)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<endoftext>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<endoftext>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在它们的词汇表中加入了“特殊记号”，如$\exttt{<endoftext>}$，以指导它们的语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{<endoftext>}$标记的通用声学实现，当预先添加到任何语音信号时，鼓励模型忽略语音，只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **9. Hard Work Does Not Always Pay Off: Poisoning Attacks on Neural Architecture Search**

努力工作并不总是有回报：对神经架构搜索的毒害攻击 cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06073v1) [paper-pdf](http://arxiv.org/pdf/2405.06073v1)

**Authors**: Zachary Coalson, Huazheng Wang, Qingyun Wu, Sanghyun Hong

**Abstract**: In this paper, we study the robustness of "data-centric" approaches to finding neural network architectures (known as neural architecture search) to data distribution shifts. To audit this robustness, we present a data poisoning attack, when injected to the training data used for architecture search that can prevent the victim algorithm from finding an architecture with optimal accuracy. We first define the attack objective for crafting poisoning samples that can induce the victim to generate sub-optimal architectures. To this end, we weaponize existing search algorithms to generate adversarial architectures that serve as our objectives. We also present techniques that the attacker can use to significantly reduce the computational costs of crafting poisoning samples. In an extensive evaluation of our poisoning attack on a representative architecture search algorithm, we show its surprising robustness. Because our attack employs clean-label poisoning, we also evaluate its robustness against label noise. We find that random label-flipping is more effective in generating sub-optimal architectures than our clean-label attack. Our results suggests that care must be taken for the data this emerging approach uses, and future work is needed to develop robust algorithms.

摘要: 在本文中，我们研究了“以数据为中心”的方法寻找神经网络结构(称为神经结构搜索)对数据分布变化的稳健性。为了检验这种健壮性，我们提出了一种数据中毒攻击，当注入用于体系结构搜索的训练数据时，可以阻止受害者算法以最佳精度找到体系结构。我们首先定义了制作中毒样本的攻击目标，这些样本可以诱导受害者生成次优的体系结构。为此，我们将现有的搜索算法武器化，以生成作为我们目标的对抗性架构。我们还提供了攻击者可以用来显著降低制作中毒样本的计算成本的技术。在对我们对一个典型架构搜索算法的毒化攻击的广泛评估中，我们展示了其惊人的健壮性。因为我们的攻击使用了干净标签中毒，所以我们还评估了它对标签噪声的稳健性。我们发现随机标签翻转在生成次优体系结构方面比我们的干净标签攻击更有效。我们的结果表明，必须注意这种新兴方法使用的数据，并且需要进一步的工作来开发健壮的算法。



## **10. BB-Patch: BlackBox Adversarial Patch-Attack using Zeroth-Order Optimization**

BB-patch：使用零阶优化的黑匣子对抗补丁攻击 cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.06049v1) [paper-pdf](http://arxiv.org/pdf/2405.06049v1)

**Authors**: Satyadwyoom Kumar, Saurabh Gupta, Arun Balaji Buduru

**Abstract**: Deep Learning has become popular due to its vast applications in almost all domains. However, models trained using deep learning are prone to failure for adversarial samples and carry a considerable risk in sensitive applications. Most of these adversarial attack strategies assume that the adversary has access to the training data, the model parameters, and the input during deployment, hence, focus on perturbing the pixel level information present in the input image.   Adversarial Patches were introduced to the community which helped in bringing out the vulnerability of deep learning models in a much more pragmatic manner but here the attacker has a white-box access to the model parameters. Recently, there has been an attempt to develop these adversarial attacks using black-box techniques. However, certain assumptions such as availability large training data is not valid for a real-life scenarios. In a real-life scenario, the attacker can only assume the type of model architecture used from a select list of state-of-the-art architectures while having access to only a subset of input dataset. Hence, we propose an black-box adversarial attack strategy that produces adversarial patches which can be applied anywhere in the input image to perform an adversarial attack.

摘要: 深度学习由于其在几乎所有领域的广泛应用而变得流行起来。然而，使用深度学习训练的模型对于对抗性样本容易失败，并且在敏感应用中具有相当大的风险。这些对抗性攻击策略大多假设对手在部署过程中可以访问训练数据、模型参数和输入，因此，专注于干扰输入图像中存在的像素级信息。社区中引入了对抗性补丁，这有助于以更实用的方式暴露深度学习模型的漏洞，但在这里，攻击者可以通过白盒访问模型参数。最近，有人试图使用黑盒技术来开发这些对抗性攻击。然而，某些假设，如大量训练数据的可用性，对于现实生活场景是不成立的。在现实生活场景中，攻击者只能假定从最先进的体系结构的精选列表中使用的模型体系结构的类型，同时只能访问输入数据集的子集。因此，我们提出了一种黑盒对抗性攻击策略，该策略产生对抗性补丁，可以应用于输入图像中的任何位置来执行对抗性攻击。



## **11. Trustworthy AI-Generative Content in Intelligent 6G Network: Adversarial, Privacy, and Fairness**

智能6G网络中值得信赖的人工智能生成内容：对抗性、隐私性和公平性 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05930v1) [paper-pdf](http://arxiv.org/pdf/2405.05930v1)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have brought revolutionary changes to the content generation fields. The high-speed and extensive 6G technology is an ideal platform for providing powerful AIGC mobile service applications, while future 6G mobile networks also need to support intelligent and personalized mobile generation services. However, the significant ethical and security issues of current AIGC models, such as adversarial attacks, privacy, and fairness, greatly affect the credibility of 6G intelligent networks, especially in ensuring secure, private, and fair AIGC applications. In this paper, we propose TrustGAIN, a novel paradigm for trustworthy AIGC in 6G networks, to ensure trustworthy large-scale AIGC services in future 6G networks. We first discuss the adversarial attacks and privacy threats faced by AIGC systems in 6G networks, as well as the corresponding protection issues. Subsequently, we emphasize the importance of ensuring the unbiasedness and fairness of the mobile generative service in future intelligent networks. In particular, we conduct a use case to demonstrate that TrustGAIN can effectively guide the resistance against malicious or generated false information. We believe that TrustGAIN is a necessary paradigm for intelligent and trustworthy 6G networks to support AIGC services, ensuring the security, privacy, and fairness of AIGC network services.

摘要: 以大语言模型(LLM)为代表的AI-Generated Content(AIGC)模型给内容生成领域带来了革命性的变化。高速和广泛的6G技术是提供强大的AIGC移动业务应用的理想平台，而未来的6G移动网络也需要支持智能化和个性化的移动生成服务。然而，当前AIGC模型存在的重大伦理和安全问题，如对抗性攻击、隐私和公平性，极大地影响了6G智能网络的可信度，特别是在确保安全、私有和公平的AIGC应用方面。本文提出了一种新的6G网络可信AIGC模型TrustGAIN，以保证未来6G网络中可信赖的大规模AIGC服务。我们首先讨论了AIGC系统在6G网络中面临的敌意攻击和隐私威胁，以及相应的保护问题。随后，我们强调了在未来的智能网中确保移动生成业务的无偏性和公平性的重要性。特别是，我们进行了一个用例来证明TrustGAIN可以有效地指导对恶意或生成的虚假信息的抵抗。我们认为，TrustGAIN是智能可信6G网络支持AIGC服务的必备范式，确保AIGC网络服务的安全性、私密性和公平性。



## **12. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

针对合成数据的属性推理攻击的线性重建方法 cs.LG

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2301.10053v3) [paper-pdf](http://arxiv.org/pdf/2301.10053v3)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.

摘要: 合成数据生成(SDG)的最新进展被誉为在保护隐私的同时共享敏感数据这一难题的解决方案。SDG旨在学习真实数据的统计属性，以便生成在结构和统计上与敏感数据相似的“人造”数据。然而，先前的研究表明，对合成数据的推理攻击可能会破坏隐私，但仅限于特定的离群值记录。在这项工作中，我们引入了一种新的针对合成数据的属性推理攻击。该攻击基于聚合统计的线性重建方法，其目标是数据集中的所有记录，而不仅仅是离群值。我们评估了我们对最先进的SDG算法的攻击，包括概率图形模型、生成性对手网络和最近的差异私有SDG机制。通过定义一个正式的隐私游戏，我们证明了我们的攻击即使在任意记录上也可以非常准确，并且这是个人信息泄露的结果(与总体级别的推断相反)。然后，我们系统地评估了保护隐私和保护统计效用之间的权衡。我们的发现表明，现有的SDG方法在保持合理效用的同时，不能始终如一地提供足够的隐私保护来抵御推理攻击。评估的最佳方法是一种不同的私有SDG机制，它可以提供对推理攻击的保护和合理的实用程序，但只能在非常特定的环境中提供。最后，我们表明，发布更多的合成记录可以提高实用性，但代价是使攻击更加有效。



## **13. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

通过注意力细化实现针对基于补丁的攻击的鲁棒语义分割 cs.CV

Accepted by International Journal of Computer Vision (IJCV).34 pages,  5 figures, 16 tables

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2401.01750v2) [paper-pdf](http://arxiv.org/pdf/2401.01750v2)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.

摘要: 近年来，注意机制在各种视觉任务中被证明是有效的。在语义分割任务中，注意力机制被应用于各种方法，包括卷积神经网络(CNN)和视觉转换器(VIT)作为骨干的情况。然而，我们观察到注意机制很容易受到基于补丁的对抗性攻击。通过对有效接受场的分析，我们将其归因于全球注意带来的广泛接受场可能导致对抗性斑块的传播。针对这一问题，本文提出了一种健壮的注意力机制(RAM)来提高语义分割模型的健壮性，该机制可以显著缓解语义分割模型对基于补丁攻击的脆弱性。与Vallina注意机制相比，RAM引入了两个新的模块：最大注意抑制和随机注意丢弃，这两个模块的目的都是为了细化注意矩阵，并限制单个敌意补丁对其他位置语义分割结果的影响。大量实验表明，在不同的攻击环境下，该算法能够有效地提高语义分割模型对各种基于补丁的攻击方法的稳健性。



## **14. TroLLoc: Logic Locking and Layout Hardening for IC Security Closure against Hardware Trojans**

TroLLoc：逻辑锁定和布局硬化，以防止硬件特洛伊木马的IC安全关闭 cs.CR

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05590v1) [paper-pdf](http://arxiv.org/pdf/2405.05590v1)

**Authors**: Fangzhou Wang, Qijing Wang, Lilas Alrahis, Bangqi Fu, Shui Jiang, Xiaopeng Zhang, Ozgur Sinanoglu, Tsung-Yi Ho, Evangeline F. Y. Young, Johann Knechtel

**Abstract**: Due to cost benefits, supply chains of integrated circuits (ICs) are largely outsourced nowadays. However, passing ICs through various third-party providers gives rise to many security threats, like piracy of IC intellectual property or insertion of hardware Trojans, i.e., malicious circuit modifications.   In this work, we proactively and systematically protect the physical layouts of ICs against post-design insertion of Trojans. Toward that end, we propose TroLLoc, a novel scheme for IC security closure that employs, for the first time, logic locking and layout hardening in unison. TroLLoc is fully integrated into a commercial-grade design flow, and TroLLoc is shown to be effective, efficient, and robust. Our work provides in-depth layout and security analysis considering the challenging benchmarks of the ISPD'22/23 contests for security closure. We show that TroLLoc successfully renders layouts resilient, with reasonable overheads, against (i) general prospects for Trojan insertion as in the ISPD'22 contest, (ii) actual Trojan insertion as in the ISPD'23 contest, and (iii) potential second-order attacks where adversaries would first (i.e., before Trojan insertion) try to bypass the locking defense, e.g., using advanced machine learning attacks. Finally, we release all our artifacts for independent verification [2].

摘要: 由于成本效益，如今集成电路(IC)的供应链大多被外包。然而，通过各种第三方提供商传递IC会带来许多安全威胁，如盗版IC知识产权或插入硬件特洛伊木马程序，即恶意电路修改。在这项工作中，我们主动和系统地保护IC的物理布局不受设计后插入特洛伊木马的影响。为此，我们提出了一种新的IC安全闭包方案--TroLLoc，它首次采用了逻辑锁定和版图加固相结合的方法。TroLLoc被完全集成到商业级设计流程中，并被证明是有效、高效和健壮的。考虑到ISPD‘22/23安全关闭竞赛的挑战性基准，我们的工作提供了深入的布局和安全分析。我们表明，TroLLoc成功地以合理的开销使布局具有弹性，以对抗(I)如在ISPD‘22比赛中那样的一般特洛伊木马插入前景，(Ii)如在ISPD’23比赛中那样的实际特洛伊木马插入，以及(Iii)潜在的二阶攻击，其中对手首先(即，在木马插入之前)试图绕过锁定防御，例如，使用高级机器学习攻击。最后，我们发布所有构件以进行独立验证[2]。



## **15. Poisoning-based Backdoor Attacks for Arbitrary Target Label with Positive Triggers**

针对具有阳性触发的任意目标标签的基于中毒的后门攻击 cs.CV

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05573v1) [paper-pdf](http://arxiv.org/pdf/2405.05573v1)

**Authors**: Binxiao Huang, Jason Chun Lok, Chang Liu, Ngai Wong

**Abstract**: Poisoning-based backdoor attacks expose vulnerabilities in the data preparation stage of deep neural network (DNN) training. The DNNs trained on the poisoned dataset will be embedded with a backdoor, making them behave well on clean data while outputting malicious predictions whenever a trigger is applied. To exploit the abundant information contained in the input data to output label mapping, our scheme utilizes the network trained from the clean dataset as a trigger generator to produce poisons that significantly raise the success rate of backdoor attacks versus conventional approaches. Specifically, we provide a new categorization of triggers inspired by the adversarial technique and develop a multi-label and multi-payload Poisoning-based backdoor attack with Positive Triggers (PPT), which effectively moves the input closer to the target label on benign classifiers. After the classifier is trained on the poisoned dataset, we can generate an input-label-aware trigger to make the infected classifier predict any given input to any target label with a high possibility. Under both dirty- and clean-label settings, we show empirically that the proposed attack achieves a high attack success rate without sacrificing accuracy across various datasets, including SVHN, CIFAR10, GTSRB, and Tiny ImageNet. Furthermore, the PPT attack can elude a variety of classical backdoor defenses, proving its effectiveness.

摘要: 基于中毒的后门攻击暴露了深度神经网络(DNN)训练的数据准备阶段的漏洞。在有毒数据集上训练的DNN将嵌入一个后门，使它们在干净的数据上表现良好，同时在应用触发器时输出恶意预测。为了利用输入数据中包含的丰富信息来输出标签映射，我们的方案利用从干净数据集训练的网络作为触发生成器来产生毒药，与传统方法相比，显著提高了后门攻击的成功率。具体地说，我们在对抗性技术的启发下提出了一种新的触发器分类方法，并提出了一种基于多标签和多负载中毒的正触发器后门攻击(PPT)，有效地使输入更接近良性分类器上的目标标签。当分类器在中毒的数据集上训练后，我们可以生成一个输入标签感知触发器，使受感染的分类器预测任何给定的输入到任何目标标签的可能性很高。在脏标签和干净标签两种设置下，我们的经验表明，所提出的攻击在不牺牲包括SVHN、CIFAR10、GTSRB和Tiny ImageNet在内的各种数据集的精度的情况下获得了高的攻击成功率。此外，PPT攻击可以避开各种经典的后门防御，证明了其有效性。



## **16. Universal Adversarial Perturbations for Vision-Language Pre-trained Models**

视觉语言预训练模型的普遍对抗扰动 cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05524v1) [paper-pdf](http://arxiv.org/pdf/2405.05524v1)

**Authors**: Peng-Fei Zhang, Zi Huang, Guangdong Bai

**Abstract**: Vision-language pre-trained (VLP) models have been the foundation of numerous vision-language tasks. Given their prevalence, it becomes imperative to assess their adversarial robustness, especially when deploying them in security-crucial real-world applications. Traditionally, adversarial perturbations generated for this assessment target specific VLP models, datasets, and/or downstream tasks. This practice suffers from low transferability and additional computation costs when transitioning to new scenarios.   In this work, we thoroughly investigate whether VLP models are commonly sensitive to imperceptible perturbations of a specific pattern for the image modality. To this end, we propose a novel black-box method to generate Universal Adversarial Perturbations (UAPs), which is so called the Effective and T ransferable Universal Adversarial Attack (ETU), aiming to mislead a variety of existing VLP models in a range of downstream tasks. The ETU comprehensively takes into account the characteristics of UAPs and the intrinsic cross-modal interactions to generate effective UAPs. Under this regime, the ETU encourages both global and local utilities of UAPs. This benefits the overall utility while reducing interactions between UAP units, improving the transferability. To further enhance the effectiveness and transferability of UAPs, we also design a novel data augmentation method named ScMix. ScMix consists of self-mix and cross-mix data transformations, which can effectively increase the multi-modal data diversity while preserving the semantics of the original data. Through comprehensive experiments on various downstream tasks, VLP models, and datasets, we demonstrate that the proposed method is able to achieve effective and transferrable universal adversarial attacks.

摘要: 视觉语言预训练(VLP)模型是众多视觉语言任务的基础。鉴于它们的普遍存在，评估它们的对手健壮性变得势在必行，特别是在将它们部署在安全关键的现实世界应用程序中时。传统上，为该评估生成的对抗性扰动针对特定的VLP模型、数据集和/或下游任务。这种做法的缺点是可转移性低，在过渡到新方案时需要额外的计算成本。在这项工作中，我们彻底调查了VLP模型是否通常对图像通道的特定模式的不可察觉的扰动敏感。为此，我们提出了一种新的生成通用对抗扰动(UAP)的黑箱方法，即有效且可传递的通用对抗攻击(ETU)，其目的是在一系列下游任务中误导现有的各种VLP模型。ETU综合考虑了UAP的特点和固有的跨模式交互作用，以生成有效的UAP。在这一制度下，ETU鼓励全球和当地的UAP公用事业。这有利于整体效用，同时减少了UAP单元之间的交互，提高了可转移性。为了进一步提高UAP的有效性和可转移性，我们还设计了一种新的数据增强方法ScMix。ScMix包括自混合和交叉混合数据转换，在保持原始数据语义的同时，有效地增加了多模式数据的多样性。通过在各种下游任务、VLP模型和数据集上的综合实验，我们证明了该方法能够实现有效的、可转移的通用对抗性攻击。



## **17. Towards Accurate and Robust Architectures via Neural Architecture Search**

通过神经架构搜索实现准确和稳健的架构 cs.CV

Accepted by CVPR2024. arXiv admin note: substantial text overlap with  arXiv:2212.14049

**SubmitDate**: 2024-05-09    [abs](http://arxiv.org/abs/2405.05502v1) [paper-pdf](http://arxiv.org/pdf/2405.05502v1)

**Authors**: Yuwei Ou, Yuqi Feng, Yanan Sun

**Abstract**: To defend deep neural networks from adversarial attacks, adversarial training has been drawing increasing attention for its effectiveness. However, the accuracy and robustness resulting from the adversarial training are limited by the architecture, because adversarial training improves accuracy and robustness by adjusting the weight connection affiliated to the architecture. In this work, we propose ARNAS to search for accurate and robust architectures for adversarial training. First we design an accurate and robust search space, in which the placement of the cells and the proportional relationship of the filter numbers are carefully determined. With the design, the architectures can obtain both accuracy and robustness by deploying accurate and robust structures to their sensitive positions, respectively. Then we propose a differentiable multi-objective search strategy, performing gradient descent towards directions that are beneficial for both natural loss and adversarial loss, thus the accuracy and robustness can be guaranteed at the same time. We conduct comprehensive experiments in terms of white-box attacks, black-box attacks, and transferability. Experimental results show that the searched architecture has the strongest robustness with the competitive accuracy, and breaks the traditional idea that NAS-based architectures cannot transfer well to complex tasks in robustness scenarios. By analyzing outstanding architectures searched, we also conclude that accurate and robust neural architectures tend to deploy different structures near the input and output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust architectures.

摘要: 为了保护深层神经网络免受对抗性攻击，对抗性训练因其有效性而受到越来越多的关注。然而，对抗训练产生的准确性和稳健性受到体系结构的限制，因为对抗训练通过调整附属于体系结构的权重连接来提高准确性和稳健性。在这项工作中，我们建议ARNAS为对抗训练寻找准确和健壮的体系结构。首先，我们设计了一个精确且稳健的搜索空间，在这个空间中，我们仔细地确定了单元的位置和过滤器数量的比例关系。通过这种设计，结构可以通过将精确和稳健的结构分别部署到其敏感位置来获得精度和稳健性。然后提出了一种可微多目标搜索策略，向有利于自然损失和对手损失的方向进行梯度下降，保证了搜索的准确性和稳健性。我们在白盒攻击、黑盒攻击和可转移性方面进行了全面的实验。实验结果表明，搜索到的体系结构具有最强的稳健性，具有与之相当的准确率，打破了基于NAS的体系结构在健壮性场景下不能很好地迁移到复杂任务的传统思想。通过对搜索到的优秀体系结构的分析，我们还得出结论：准确和健壮的神经体系结构往往在输入和输出附近部署不同的结构，这对于手工制作和自动设计准确健壮的体系结构都具有重要的现实意义。



## **18. Adversary-Guided Motion Retargeting for Skeleton Anonymization**

对抗引导的骨架模拟运动重定向 cs.CV

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05428v1) [paper-pdf](http://arxiv.org/pdf/2405.05428v1)

**Authors**: Thomas Carr, Depeng Xu, Aidong Lu

**Abstract**: Skeleton-based motion visualization is a rising field in computer vision, especially in the case of virtual reality (VR). With further advancements in human-pose estimation and skeleton extracting sensors, more and more applications that utilize skeleton data have come about. These skeletons may appear to be anonymous but they contain embedded personally identifiable information (PII). In this paper we present a new anonymization technique that is based on motion retargeting, utilizing adversary classifiers to further remove PII embedded in the skeleton. Motion retargeting is effective in anonymization as it transfers the movement of the user onto the a dummy skeleton. In doing so, any PII linked to the skeleton will be based on the dummy skeleton instead of the user we are protecting. We propose a Privacy-centric Deep Motion Retargeting model (PMR) which aims to further clear the retargeted skeleton of PII through adversarial learning. In our experiments, PMR achieves motion retargeting utility performance on par with state of the art models while also reducing the performance of privacy attacks.

摘要: 基于骨架的运动可视化是计算机视觉领域的一个新兴领域，尤其是在虚拟现实(VR)领域。随着人体姿态估计和骨骼提取传感器的进一步发展，利用骨骼数据的应用越来越多。这些骨架可能看起来是匿名的，但它们包含嵌入的个人身份信息(PII)。本文提出了一种新的匿名技术，该技术基于运动重定向，利用敌方分类器进一步去除嵌入在骨架中的PII。运动重定目标在匿名化中是有效的，因为它将用户的运动转移到虚拟骨骼上。这样，链接到骨架的任何PII都将基于虚拟骨架，而不是我们正在保护的用户。我们提出了一种以隐私为中心的深度运动重定向模型(PMR)，旨在通过对抗性学习进一步清除PII的重定向骨架。在我们的实验中，PMR实现了与最先进模型相当的运动重定目标实用性能，同时还降低了隐私攻击的性能。



## **19. Air Gap: Protecting Privacy-Conscious Conversational Agents**

空气间隙：保护有隐私意识的对话代理人 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05175v1) [paper-pdf](http://arxiv.org/pdf/2405.05175v1)

**Authors**: Eugene Bagdasaryan, Ren Yi, Sahra Ghalebikesabi, Peter Kairouz, Marco Gruteser, Sewoong Oh, Borja Balle, Daniel Ramage

**Abstract**: The growing use of large language model (LLM)-based conversational agents to manage sensitive user data raises significant privacy concerns. While these agents excel at understanding and acting on context, this capability can be exploited by malicious actors. We introduce a novel threat model where adversarial third-party apps manipulate the context of interaction to trick LLM-based agents into revealing private information not relevant to the task at hand.   Grounded in the framework of contextual integrity, we introduce AirGapAgent, a privacy-conscious agent designed to prevent unintended data leakage by restricting the agent's access to only the data necessary for a specific task. Extensive experiments using Gemini, GPT, and Mistral models as agents validate our approach's effectiveness in mitigating this form of context hijacking while maintaining core agent functionality. For example, we show that a single-query context hijacking attack on a Gemini Ultra agent reduces its ability to protect user data from 94% to 45%, while an AirGapAgent achieves 97% protection, rendering the same attack ineffective.

摘要: 越来越多地使用基于大型语言模型(LLM)的会话代理来管理敏感用户数据，这引发了严重的隐私问题。虽然这些代理擅长理解上下文并根据上下文执行操作，但这种能力可能会被恶意行为者利用。我们引入了一种新的威胁模型，在该模型中，敌意的第三方应用程序操纵交互的上下文，以欺骗基于LLM的代理泄露与手头任务无关的私人信息。基于上下文完整性的框架，我们引入了AirGapAgent，这是一个具有隐私意识的代理，旨在通过限制代理仅访问特定任务所需的数据来防止意外的数据泄露。使用Gemini、GPT和Mistral模型作为代理的大量实验验证了我们的方法在保持核心代理功能的同时缓解这种形式的上下文劫持的有效性。例如，我们表明，对Gemini Ultra代理的单查询上下文劫持攻击将其保护用户数据的能力从94%降低到45%，而AirGapAgent实现了97%的保护，使得相同的攻击无效。



## **20. Filtering and smoothing estimation algorithms from uncertain nonlinear observations with time-correlated additive noise and random deception attacks**

来自具有时间相关添加性噪音和随机欺骗攻击的不确定非线性观测的过滤和平滑估计算法 eess.SP

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05157v1) [paper-pdf](http://arxiv.org/pdf/2405.05157v1)

**Authors**: R. Caballero-Águila, J. Hu, J. Linares-Pérez

**Abstract**: This paper discusses the problem of estimating a stochastic signal from nonlinear uncertain observations with time-correlated additive noise described by a first-order Markov process. Random deception attacks are assumed to be launched by an adversary, and both this phenomenon and the uncertainty in the observations are modelled by two sets of Bernoulli random variables. Under the assumption that the evolution model generating the signal to be estimated is unknown and only the mean and covariance functions of the processes involved in the observation equation are available, recursive algorithms based on linear approximations of the real observations are proposed for the least-squares filtering and fixed-point smoothing problems. Finally, the feasibility and effectiveness of the developed estimation algorithms are verified by a numerical simulation example, where the impact of uncertain observation and deception attack probabilities on estimation accuracy is evaluated.

摘要: 本文讨论了由一阶马尔科夫过程描述的具有时间相关添加性噪音的非线性不确定观测估计随机信号的问题。假设随机欺骗攻击是由对手发起的，这种现象和观察中的不确定性都是由两组伯努里随机变量建模的。在生成待估计信号的进化模型未知且只有观测方程中涉及的过程的均值和协方差函数可用的假设下，提出了基于真实观测值线性逼近的回归算法来解决最小平方过滤和定点平滑问题。最后，通过数值仿真算例验证了所开发的估计算法的可行性和有效性，评估了不确定观测和欺骗攻击概率对估计准确性的影响。



## **21. Towards Efficient Training and Evaluation of Robust Models against $l_0$ Bounded Adversarial Perturbations**

针对1_0美元有界对抗性扰动的稳健模型的有效训练和评估 cs.LG

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05075v1) [paper-pdf](http://arxiv.org/pdf/2405.05075v1)

**Authors**: Xuyang Zhong, Yixiao Huang, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations bounded by $l_0$ norm. We propose a white-box PGD-like attack method named sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against $l_0$ bounded adversarial perturbations. Moreover, the efficiency of sparse-PGD enables us to conduct adversarial training to build robust models against sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.

摘要: 这项工作研究了以$l_0$规范为界的稀疏对抗扰动。我们提出了一种名为sparse-PVD的白盒类PGD攻击方法，以有效且高效地生成此类扰动。此外，我们将稀疏PVD与黑匣子攻击相结合，以全面、更可靠地评估模型对1_0美元有界对抗扰动的鲁棒性。此外，稀疏PVD的效率使我们能够进行对抗训练，以构建针对稀疏扰动的稳健模型。大量实验表明，我们提出的攻击算法在不同场景下表现出强大的性能。更重要的是，与其他稳健模型相比，我们的对抗训练模型表现出了针对各种稀疏攻击的最新稳健性。代码可访问https://github.com/CityU-MLO/sPGD。



## **22. Adversarial Threats to Automatic Modulation Open Set Recognition in Wireless Networks**

无线网络中自动调制开集识别的对抗威胁 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.05022v1) [paper-pdf](http://arxiv.org/pdf/2405.05022v1)

**Authors**: Yandie Yang, Sicheng Zhang, Kuixian Li, Qiao Tian, Yun Lin

**Abstract**: Automatic Modulation Open Set Recognition (AMOSR) is a crucial technological approach for cognitive radio communications, wireless spectrum management, and interference monitoring within wireless networks. Numerous studies have shown that AMR is highly susceptible to minimal perturbations carefully designed by malicious attackers, leading to misclassification of signals. However, the adversarial security issue of AMOSR has not yet been explored. This paper adopts the perspective of attackers and proposes an Open Set Adversarial Attack (OSAttack), aiming at investigating the adversarial vulnerabilities of various AMOSR methods. Initially, an adversarial threat model for AMOSR scenarios is established. Subsequently, by analyzing the decision criteria of both discriminative and generative open set recognition, OSFGSM and OSPGD are proposed to reduce the performance of AMOSR. Finally, the influence of OSAttack on AMOSR is evaluated utilizing a range of qualitative and quantitative indicators. The results indicate that despite the increased resistance of AMOSR models to conventional interference signals, they remain vulnerable to attacks by adversarial examples.

摘要: 自动调制开集识别(AMOSR)是认知无线电通信、无线频谱管理和无线网络干扰监测的重要技术手段。大量研究表明，AMR非常容易受到恶意攻击者精心设计的微小扰动的影响，从而导致信号的错误分类。然而，AMOSR的对抗性安全问题尚未被探讨。本文从攻击者的角度出发，提出了一种开放集对抗性攻击(OSAttack)，旨在研究各种AMOSR方法的对抗性漏洞。首先，建立了AMOSR场景的对抗性威胁模型。随后，通过分析判别性和生成性开集识别的决策准则，提出了OSFGSM和OSPGD来降低AMOSR的性能。最后，利用一系列定性和定量指标对OSAttack对AMOSR的影响进行了评估。结果表明，尽管AMOSR模型对常规干扰信号的抵抗力有所增强，但它们仍然容易受到对手例子的攻击。



## **23. Deep Reinforcement Learning with Spiking Q-learning**

具有峰值Q学习的深度强化学习 cs.NE

15 pages, 7 figures

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2201.09754v3) [paper-pdf](http://arxiv.org/pdf/2201.09754v3)

**Authors**: Ding Chen, Peixi Peng, Tiejun Huang, Yonghong Tian

**Abstract**: With the help of special neuromorphic hardware, spiking neural networks (SNNs) are expected to realize artificial intelligence (AI) with less energy consumption. It provides a promising energy-efficient way for realistic control tasks by combining SNNs with deep reinforcement learning (RL). There are only a few existing SNN-based RL methods at present. Most of them either lack generalization ability or employ Artificial Neural Networks (ANNs) to estimate value function in training. The former needs to tune numerous hyper-parameters for each scenario, and the latter limits the application of different types of RL algorithm and ignores the large energy consumption in training. To develop a robust spike-based RL method, we draw inspiration from non-spiking interneurons found in insects and propose the deep spiking Q-network (DSQN), using the membrane voltage of non-spiking neurons as the representation of Q-value, which can directly learn robust policies from high-dimensional sensory inputs using end-to-end RL. Experiments conducted on 17 Atari games demonstrate the DSQN is effective and even outperforms the ANN-based deep Q-network (DQN) in most games. Moreover, the experiments show superior learning stability and robustness to adversarial attacks of DSQN.

摘要: 在特殊的神经形态硬件的帮助下，脉冲神经网络(SNN)有望以更少的能量消耗实现人工智能(AI)。它将神经网络和深度强化学习相结合，为实际控制任务提供了一种很有前途的节能方法。目前已有的基于SNN的RL方法很少。大多数人要么缺乏泛化能力，要么在训练中使用人工神经网络(ANN)来估计价值函数。前者需要针对每个场景调整大量的超参数，而后者限制了不同类型RL算法的应用，忽略了训练过程中的巨大能量消耗。为了开发一种稳健的基于棘波的RL方法，我们从昆虫中发现的非尖峰中间神经元中吸取灵感，提出了深度尖峰Q-网络(DSQN)，它使用非尖峰神经元的膜电压作为Q值的表示，可以使用端到端RL直接从高维感觉输入中学习鲁棒策略。在17个Atari游戏上的实验表明，DSQN是有效的，甚至在大多数游戏中都优于基于神经网络的深度Q网络(DQN)。实验表明，DSQN具有良好的学习稳定性和对敌意攻击的健壮性。



## **24. Learning-Based Difficulty Calibration for Enhanced Membership Inference Attacks**

基于学习的增强型成员推断攻击难度校准 cs.CR

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2401.04929v2) [paper-pdf](http://arxiv.org/pdf/2401.04929v2)

**Authors**: Haonan Shi, Tu Ouyang, An Wang

**Abstract**: Machine learning models, in particular deep neural networks, are currently an integral part of various applications, from healthcare to finance. However, using sensitive data to train these models raises concerns about privacy and security. One method that has emerged to verify if the trained models are privacy-preserving is Membership Inference Attacks (MIA), which allows adversaries to determine whether a specific data point was part of a model's training dataset. While a series of MIAs have been proposed in the literature, only a few can achieve high True Positive Rates (TPR) in the low False Positive Rate (FPR) region (0.01%~1%). This is a crucial factor to consider for an MIA to be practically useful in real-world settings. In this paper, we present a novel approach to MIA that is aimed at significantly improving TPR at low FPRs. Our method, named learning-based difficulty calibration for MIA(LDC-MIA), characterizes data records by their hardness levels using a neural network classifier to determine membership. The experiment results show that LDC-MIA can improve TPR at low FPR by up to 4x compared to the other difficulty calibration based MIAs. It also has the highest Area Under ROC curve (AUC) across all datasets. Our method's cost is comparable with most of the existing MIAs, but is orders of magnitude more efficient than one of the state-of-the-art methods, LiRA, while achieving similar performance.

摘要: 机器学习模型，特别是深度神经网络，目前是从医疗保健到金融的各种应用程序的组成部分。然而，使用敏感数据来训练这些模型会引发对隐私和安全的担忧。出现的一种验证训练模型是否保护隐私的方法是成员推理攻击(MIA)，它允许对手确定特定数据点是否属于模型训练数据集的一部分。虽然文献中已经提出了一系列的MIA，但只有少数几个MIA能在低假阳性率(FPR)区域(0.01%~1%)获得高的真阳性率(TPR)。要使MIA在实际环境中发挥实际作用，这是需要考虑的关键因素。在本文中，我们提出了一种新的MIA方法，旨在显著改善低FPR下的TPR。我们的方法，称为基于学习的MIA难度校准(LDC-MIA)，使用神经网络分类器来确定成员身份，根据数据记录的硬度来表征数据记录。实验结果表明，与其他基于难度校正的MIA相比，LDC-MIA可以在较低的误码率下将TPR提高4倍。在所有数据集中，它也具有最高的ROC曲线下面积(AUC)。我们的方法的成本与大多数现有的MIA相当，但效率比最先进的方法之一LIRA高出数量级，同时实现了类似的性能。



## **25. BiasKG: Adversarial Knowledge Graphs to Induce Bias in Large Language Models**

BiasKG：对抗性知识图在大型语言模型中诱导偏见 cs.CL

**SubmitDate**: 2024-05-08    [abs](http://arxiv.org/abs/2405.04756v1) [paper-pdf](http://arxiv.org/pdf/2405.04756v1)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Modern large language models (LLMs) have a significant amount of world knowledge, which enables strong performance in commonsense reasoning and knowledge-intensive tasks when harnessed properly. The language model can also learn social biases, which has a significant potential for societal harm. There have been many mitigation strategies proposed for LLM safety, but it is unclear how effective they are for eliminating social biases. In this work, we propose a new methodology for attacking language models with knowledge graph augmented generation. We refactor natural language stereotypes into a knowledge graph, and use adversarial attacking strategies to induce biased responses from several open- and closed-source language models. We find our method increases bias in all models, even those trained with safety guardrails. This demonstrates the need for further research in AI safety, and further work in this new adversarial space.

摘要: 现代大型语言模型（LLM）拥有大量的世界知识，如果利用得当，可以在常识推理和知识密集型任务中取得出色的性能。语言模型还可以学习社会偏见，这具有巨大的社会危害潜力。人们为LLM安全提出了许多缓解策略，但目前尚不清楚它们对于消除社会偏见的有效性如何。在这项工作中，我们提出了一种利用知识图增强生成来攻击语言模型的新方法。我们将自然语言刻板印象重新构建到知识图谱中，并使用对抗性攻击策略来诱导几个开放和封闭源语言模型的偏见反应。我们发现我们的方法增加了所有模型的偏差，甚至是那些接受过安全护栏训练的模型。这表明需要对人工智能安全进行进一步研究，并在这个新的对抗空间中进一步开展工作。



## **26. Demonstration of an Adversarial Attack Against a Multimodal Vision Language Model for Pathology Imaging**

演示针对病理成像多模式视觉语言模型的对抗攻击 eess.IV

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2401.02565v3) [paper-pdf](http://arxiv.org/pdf/2401.02565v3)

**Authors**: Poojitha Thota, Jai Prakash Veerla, Partha Sai Guttikonda, Mohammad S. Nasr, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the context of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted attacks. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial perturbation attacks to induce misclassifications intentionally. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models. The source codes for this experiment can be found at https://github.com/jaiprakash1824/VLM_Adv_Attack.

摘要: 在医学人工智能的背景下，本研究探索了视觉语言基础模型-病理语言-图像预训练(PLIP)模型在有针对性攻击下的脆弱性。利用Kather Colon数据集和9种组织类型的7,180张H&E图像，我们的研究使用了投影梯度下降(PGD)对抗性扰动攻击来故意诱导错误分类。结果显示，PLIP操纵预测的成功率为100%，突显出其易受对手干扰的影响。对抗性例子的定性分析深入到了可解释性的挑战，揭示了对抗性操纵导致的预测的细微变化。这些发现为医学成像中视觉语言模型的可解释性、领域适应性和可信性提供了重要的见解。该研究强调，迫切需要强大的防御措施，以确保人工智能模型的可靠性。这个实验的源代码可以在https://github.com/jaiprakash1824/VLM_Adv_Attack.上找到



## **27. Fully Automated Selfish Mining Analysis in Efficient Proof Systems Blockchains**

高效证明系统区块链中的全自动自私挖掘分析 cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04420v1) [paper-pdf](http://arxiv.org/pdf/2405.04420v1)

**Authors**: Krishnendu Chatterjee, Amirali Ebrahimzadeh, Mehrdad Karrabi, Krzysztof Pietrzak, Michelle Yeo, Đorđe Žikelić

**Abstract**: We study selfish mining attacks in longest-chain blockchains like Bitcoin, but where the proof of work is replaced with efficient proof systems -- like proofs of stake or proofs of space -- and consider the problem of computing an optimal selfish mining attack which maximizes expected relative revenue of the adversary, thus minimizing the chain quality. To this end, we propose a novel selfish mining attack that aims to maximize this objective and formally model the attack as a Markov decision process (MDP). We then present a formal analysis procedure which computes an $\epsilon$-tight lower bound on the optimal expected relative revenue in the MDP and a strategy that achieves this $\epsilon$-tight lower bound, where $\epsilon>0$ may be any specified precision. Our analysis is fully automated and provides formal guarantees on the correctness. We evaluate our selfish mining attack and observe that it achieves superior expected relative revenue compared to two considered baselines.   In concurrent work [Sarenche FC'24] does an automated analysis on selfish mining in predictable longest-chain blockchains based on efficient proof systems. Predictable means the randomness for the challenges is fixed for many blocks (as used e.g., in Ouroboros), while we consider unpredictable (Bitcoin-like) chains where the challenge is derived from the previous block.

摘要: 我们研究了比特币等最长链区块链中的自私挖掘攻击，但工作证明被高效的证明系统取代--如赌注证明或空间证明--并考虑计算最优自私挖掘攻击的问题，该攻击最大化对手的预期相对收益，从而最小化链质量。为此，我们提出了一种新的自私挖掘攻击，旨在最大化这一目标，并将攻击形式化地建模为马尔可夫决策过程(MDP)。然后，我们给出了一个形式的分析程序，它计算了MDP中最优预期相对收益的$\epsilon$-紧下界，并给出了一个实现这个$\epsilon$-紧下界的策略，其中$\epsilon>0$可以是任意指定的精度。我们的分析是完全自动化的，并为正确性提供正式保证。我们评估了我们的自私挖掘攻击，并观察到与两个考虑的基线相比，它实现了更好的预期相对收益。在并发工作[Sarhene FC‘24]中，基于高效的证明系统，对可预测的最长链区块链中的自私挖掘进行了自动化分析。可预测意味着挑战的随机性对于许多区块是固定的(例如，在Ouroboros中使用)，而我们认为挑战来自前一个区块的不可预测(类似比特币的)链。



## **28. NeuroIDBench: An Open-Source Benchmark Framework for the Standardization of Methodology in Brainwave-based Authentication Research**

NeuroIDBench：基于脑电波的认证研究方法标准化的开源基准框架 cs.CR

21 pages, 5 Figures, 3 tables, Submitted to the Journal of  Information Security and Applications

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2402.08656v4) [paper-pdf](http://arxiv.org/pdf/2402.08656v4)

**Authors**: Avinash Kumar Chaurasia, Matin Fallahi, Thorsten Strufe, Philipp Terhörst, Patricia Arias Cabarcos

**Abstract**: Biometric systems based on brain activity have been proposed as an alternative to passwords or to complement current authentication techniques. By leveraging the unique brainwave patterns of individuals, these systems offer the possibility of creating authentication solutions that are resistant to theft, hands-free, accessible, and potentially even revocable. However, despite the growing stream of research in this area, faster advance is hindered by reproducibility problems. Issues such as the lack of standard reporting schemes for performance results and system configuration, or the absence of common evaluation benchmarks, make comparability and proper assessment of different biometric solutions challenging. Further, barriers are erected to future work when, as so often, source code is not published open access. To bridge this gap, we introduce NeuroIDBench, a flexible open source tool to benchmark brainwave-based authentication models. It incorporates nine diverse datasets, implements a comprehensive set of pre-processing parameters and machine learning algorithms, enables testing under two common adversary models (known vs unknown attacker), and allows researchers to generate full performance reports and visualizations. We use NeuroIDBench to investigate the shallow classifiers and deep learning-based approaches proposed in the literature, and to test robustness across multiple sessions. We observe a 37.6% reduction in Equal Error Rate (EER) for unknown attacker scenarios (typically not tested in the literature), and we highlight the importance of session variability to brainwave authentication. All in all, our results demonstrate the viability and relevance of NeuroIDBench in streamlining fair comparisons of algorithms, thereby furthering the advancement of brainwave-based authentication through robust methodological practices.

摘要: 基于大脑活动的生物识别系统已经被提出作为密码的替代方案，或者是对当前身份验证技术的补充。通过利用个人独特的脑电波模式，这些系统提供了创建防盗、免提、可访问甚至可能可撤销的身份验证解决方案的可能性。然而，尽管这一领域的研究越来越多，但重复性问题阻碍了更快的进展。缺乏性能结果和系统配置的标准报告方案，或缺乏通用的评估基准等问题，使不同生物识别解决方案的可比性和适当评估具有挑战性。此外，当源代码不公开、开放获取时，就会为未来的工作设置障碍。为了弥补这一差距，我们引入了NeuroIDBch，这是一个灵活的开源工具，用于对基于脑电波的身份验证模型进行基准测试。它整合了九个不同的数据集，实现了一套全面的预处理参数和机器学习算法，可以在两个常见的对手模型(已知和未知攻击者)下进行测试，并允许研究人员生成完整的性能报告和可视化。我们使用NeuroIDB边来研究文献中提出的浅层分类器和基于深度学习的方法，并测试多个会话的健壮性。我们观察到，对于未知攻击者场景(通常未在文献中进行测试)，等错误率(EER)降低了37.6%，并强调了会话可变性对脑电波身份验证的重要性。总而言之，我们的结果证明了NeuroIDBtch在简化公平的算法比较方面的可行性和相关性，从而通过稳健的方法学实践进一步推进基于脑电波的身份验证。



## **29. Revisiting character-level adversarial attacks**

重新审视角色级对抗攻击 cs.LG

Accepted in ICML 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04346v1) [paper-pdf](http://arxiv.org/pdf/2405.04346v1)

**Authors**: Elias Abad Rocamora, Yongtao Wu, Fanghui Liu, Grigorios G. Chrysos, Volkan Cevher

**Abstract**: Adversarial attacks in Natural Language Processing apply perturbations in the character or token levels. Token-level attacks, gaining prominence for their use of gradient-based methods, are susceptible to altering sentence semantics, leading to invalid adversarial examples. While character-level attacks easily maintain semantics, they have received less attention as they cannot easily adopt popular gradient-based methods, and are thought to be easy to defend. Challenging these beliefs, we introduce Charmer, an efficient query-based adversarial attack capable of achieving high attack success rate (ASR) while generating highly similar adversarial examples. Our method successfully targets both small (BERT) and large (Llama 2) models. Specifically, on BERT with SST-2, Charmer improves the ASR in 4.84% points and the USE similarity in 8% points with respect to the previous art. Our implementation is available in https://github.com/LIONS-EPFL/Charmer.

摘要: 自然语言处理中的对抗攻击对字符或令牌级别施加扰动。令牌级攻击因使用基于梯度的方法而变得越来越重要，很容易改变句子语义，从而导致无效的对抗性示例。虽然字符级攻击很容易维护语义，但它们受到的关注较少，因为它们不能轻易采用流行的基于梯度的方法，并且被认为很容易防御。基于这些信念，我们引入了Charmer，这是一种高效的基于查询的对抗性攻击，能够实现高攻击成功率（ASB），同时生成高度相似的对抗性示例。我们的方法成功地针对小型（BERT）和大型（Llama 2）模型。具体来说，在采用CST-2的BERT上，Charmer将ASB提高了4.84%，与之前的作品相比，USE相似性提高了8%。我们的实现可在https://github.com/LIONS-EPFL/Charmer上获取。



## **30. Who Wrote This? The Key to Zero-Shot LLM-Generated Text Detection Is GECScore**

这是谁写的？零镜头LLM生成文本检测的关键是GECScore cs.CL

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04286v1) [paper-pdf](http://arxiv.org/pdf/2405.04286v1)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xuebo Liu, Lidia S. Chao, Min Zhang

**Abstract**: The efficacy of an large language model (LLM) generated text detector depends substantially on the availability of sizable training data. White-box zero-shot detectors, which require no such data, are nonetheless limited by the accessibility of the source model of the LLM-generated text. In this paper, we propose an simple but effective black-box zero-shot detection approach, predicated on the observation that human-written texts typically contain more grammatical errors than LLM-generated texts. This approach entails computing the Grammar Error Correction Score (GECScore) for the given text to distinguish between human-written and LLM-generated text. Extensive experimental results show that our method outperforms current state-of-the-art (SOTA) zero-shot and supervised methods, achieving an average AUROC of 98.7% and showing strong robustness against paraphrase and adversarial perturbation attacks.

摘要: 大型语言模型（LLM）生成的文本检测器的功效在很大程度上取决于大量训练数据的可用性。白盒零镜头检测器不需要此类数据，但仍受到LLM生成文本源模型可访问性的限制。在本文中，我们提出了一种简单但有效的黑匣子零镜头检测方法，其基础是人类书面文本通常比LLM生成的文本包含更多的语法错误。这种方法需要计算给定文本的语法错误纠正分数（GECScore），以区分人类编写的文本和LLM生成的文本。大量的实验结果表明，我们的方法优于当前最先进的（SOTA）零射击和监督方法，实现了98.7%的平均AUROC，并对重述和对抗性扰动攻击表现出强大的鲁棒性。



## **31. A Stealthy Wrongdoer: Feature-Oriented Reconstruction Attack against Split Learning**

一个偷偷摸摸的犯错者：针对分裂学习的以冲突为导向的重建攻击 cs.CR

Accepted to CVPR 2024

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04115v1) [paper-pdf](http://arxiv.org/pdf/2405.04115v1)

**Authors**: Xiaoyang Xu, Mengda Yang, Wenzhe Yi, Ziang Li, Juan Wang, Hongxin Hu, Yong Zhuang, Yaxin Liu

**Abstract**: Split Learning (SL) is a distributed learning framework renowned for its privacy-preserving features and minimal computational requirements. Previous research consistently highlights the potential privacy breaches in SL systems by server adversaries reconstructing training data. However, these studies often rely on strong assumptions or compromise system utility to enhance attack performance. This paper introduces a new semi-honest Data Reconstruction Attack on SL, named Feature-Oriented Reconstruction Attack (FORA). In contrast to prior works, FORA relies on limited prior knowledge, specifically that the server utilizes auxiliary samples from the public without knowing any client's private information. This allows FORA to conduct the attack stealthily and achieve robust performance. The key vulnerability exploited by FORA is the revelation of the model representation preference in the smashed data output by victim client. FORA constructs a substitute client through feature-level transfer learning, aiming to closely mimic the victim client's representation preference. Leveraging this substitute client, the server trains the attack model to effectively reconstruct private data. Extensive experiments showcase FORA's superior performance compared to state-of-the-art methods. Furthermore, the paper systematically evaluates the proposed method's applicability across diverse settings and advanced defense strategies.

摘要: Split Learning(SL)是一种分布式学习框架，以其隐私保护功能和最小的计算要求而闻名。以前的研究一直强调，通过服务器对手重建训练数据，SL系统中潜在的隐私泄露。然而，这些研究往往依赖强假设或折衷系统效用来提高攻击性能。介绍了一种新的基于SL的半诚实数据重构攻击--面向特征的重构攻击(FORA)。与以前的工作不同，FORA依赖于有限的先验知识，特别是服务器使用来自公共的辅助样本，而不知道任何客户的私人信息。这使得Fora能够悄悄地进行攻击，并实现稳健的性能。Fora利用的关键漏洞是受害者客户端输出的粉碎数据中暴露的模型表示首选项。FORA通过特征级迁移学习构造了一个替代客户，旨在更好地模拟受害客户的表征偏好。利用这个替代客户端，服务器训练攻击模型以有效地重建私有数据。广泛的实验表明，与最先进的方法相比，FORA的性能更优越。此外，本文还系统地评估了该方法在不同环境和先进防御策略下的适用性。



## **32. Explainability-Informed Targeted Malware Misclassification**

有解释性的定向恶意软件错误分类 cs.CR

**SubmitDate**: 2024-05-07    [abs](http://arxiv.org/abs/2405.04010v1) [paper-pdf](http://arxiv.org/pdf/2405.04010v1)

**Authors**: Quincy Card, Kshitiz Aryal, Maanak Gupta

**Abstract**: In recent years, there has been a surge in malware attacks across critical infrastructures, requiring further research and development of appropriate response and remediation strategies in malware detection and classification. Several works have used machine learning models for malware classification into categories, and deep neural networks have shown promising results. However, these models have shown its vulnerabilities against intentionally crafted adversarial attacks, which yields misclassification of a malicious file. Our paper explores such adversarial vulnerabilities of neural network based malware classification system in the dynamic and online analysis environments. To evaluate our approach, we trained Feed Forward Neural Networks (FFNN) to classify malware categories based on features obtained from dynamic and online analysis environments. We use the state-of-the-art method, SHapley Additive exPlanations (SHAP), for the feature attribution for malware classification, to inform the adversarial attackers about the features with significant importance on classification decision. Using the explainability-informed features, we perform targeted misclassification adversarial white-box evasion attacks using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks against the trained classifier. Our results demonstrated high evasion rate for some instances of attacks, showing a clear vulnerability of a malware classifier for such attacks. We offer recommendations for a balanced approach and a benchmark for much-needed future research into evasion attacks against malware classifiers, and develop more robust and trustworthy solutions.

摘要: 近年来，跨关键基础设施的恶意软件攻击激增，需要在恶意软件检测和分类方面进一步研究和开发适当的响应和补救策略。有几项工作使用机器学习模型将恶意软件分类，深度神经网络也显示出了令人振奋的结果。然而，这些模型显示了其针对故意构建的敌意攻击的漏洞，这些攻击会导致对恶意文件的错误分类。本文探讨了基于神经网络的恶意软件分类系统在动态分析和在线分析环境中的攻击漏洞。为了评估我们的方法，我们训练前馈神经网络(FFNN)根据从动态和在线分析环境中获得的特征对恶意软件类别进行分类。对于恶意软件分类的特征属性，我们使用了最新的Shapley Additive Ex释义(Shap)方法，将对分类决策有重要意义的特征告知恶意攻击者。利用可解释性信息特征，我们使用快速梯度符号方法(FGSM)和投影梯度下降(PGD)方法对训练好的分类器进行有针对性的误分类敌意白盒逃避攻击。我们的结果表明，对于一些攻击实例，逃避率很高，这表明恶意软件分类器对此类攻击存在明显的漏洞。我们为针对恶意软件分类器的躲避攻击的未来研究提供了平衡方法的建议和基准，并开发了更强大和值得信赖的解决方案。



## **33. Navigating Quantum Security Risks in Networked Environments: A Comprehensive Study of Quantum-Safe Network Protocols**

应对网络环境中的量子安全风险：量子安全网络协议的全面研究 cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2404.08232v2) [paper-pdf](http://arxiv.org/pdf/2404.08232v2)

**Authors**: Yaser Baseri, Vikas Chouhan, Abdelhakim Hafid

**Abstract**: The emergence of quantum computing poses a formidable security challenge to network protocols traditionally safeguarded by classical cryptographic algorithms. This paper provides an exhaustive analysis of vulnerabilities introduced by quantum computing in a diverse array of widely utilized security protocols across the layers of the TCP/IP model, including TLS, IPsec, SSH, PGP, and more. Our investigation focuses on precisely identifying vulnerabilities susceptible to exploitation by quantum adversaries at various migration stages for each protocol while also assessing the associated risks and consequences for secure communication. We delve deep into the impact of quantum computing on each protocol, emphasizing potential threats posed by quantum attacks and scrutinizing the effectiveness of post-quantum cryptographic solutions. Through carefully evaluating vulnerabilities and risks that network protocols face in the post-quantum era, this study provides invaluable insights to guide the development of appropriate countermeasures. Our findings contribute to a broader comprehension of quantum computing's influence on network security and offer practical guidance for protocol designers, implementers, and policymakers in addressing the challenges stemming from the advancement of quantum computing. This comprehensive study is a crucial step toward fortifying the security of networked environments in the quantum age.

摘要: 量子计算的出现对传统上由经典密码算法保护的网络协议提出了严峻的安全挑战。本文详尽分析了量子计算在各种广泛使用的安全协议(包括TLS、IPSec、SSH、PGP等)的TCP/IP模型各层中引入的漏洞。我们的调查重点是准确地识别在每个协议的不同迁移阶段容易被量子攻击者利用的漏洞，同时还评估了相关的风险和安全通信的后果。我们深入研究了量子计算对每个协议的影响，强调了量子攻击带来的潜在威胁，并仔细审查了后量子密码解决方案的有效性。通过仔细评估后量子时代网络协议面临的漏洞和风险，本研究为指导制定适当的对策提供了宝贵的见解。我们的发现有助于更广泛地理解量子计算对网络安全的影响，并为协议设计者、实施者和政策制定者提供实用指导，以应对量子计算进步带来的挑战。这项全面的研究是在量子时代加强网络环境安全的关键一步。



## **34. Enhancing O-RAN Security: Evasion Attacks and Robust Defenses for Graph Reinforcement Learning-based Connection Management**

增强O-RAN安全性：基于图强化学习的连接管理的规避攻击和稳健防御 cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03891v1) [paper-pdf](http://arxiv.org/pdf/2405.03891v1)

**Authors**: Ravikumar Balakrishnan, Marius Arvinte, Nageen Himayat, Hosein Nikopour, Hassnaa Moustafa

**Abstract**: Adversarial machine learning, focused on studying various attacks and defenses on machine learning (ML) models, is rapidly gaining importance as ML is increasingly being adopted for optimizing wireless systems such as Open Radio Access Networks (O-RAN). A comprehensive modeling of the security threats and the demonstration of adversarial attacks and defenses on practical AI based O-RAN systems is still in its nascent stages. We begin by conducting threat modeling to pinpoint attack surfaces in O-RAN using an ML-based Connection management application (xApp) as an example. The xApp uses a Graph Neural Network trained using Deep Reinforcement Learning and achieves on average 54% improvement in the coverage rate measured as the 5th percentile user data rates. We then formulate and demonstrate evasion attacks that degrade the coverage rates by as much as 50% through injecting bounded noise at different threat surfaces including the open wireless medium itself. Crucially, we also compare and contrast the effectiveness of such attacks on the ML-based xApp and a non-ML based heuristic. We finally develop and demonstrate robust training-based defenses against the challenging physical/jamming-based attacks and show a 15% improvement in the coverage rates when compared to employing no defense over a range of noise budgets

摘要: 对抗性机器学习专注于研究机器学习模型上的各种攻击和防御，随着机器学习模型越来越多地被用于优化开放无线接入网络(O-RAN)等无线系统，机器学习正迅速变得越来越重要。对实用的基于人工智能的O-RAN系统进行安全威胁的全面建模以及对抗性攻击和防御的演示仍处于初级阶段。我们首先以基于ML的连接管理应用程序(XApp)为例进行威胁建模，以确定O-RAN中的攻击面。XApp使用使用深度强化学习训练的图形神经网络，以第5个百分位的用户数据速率衡量，覆盖率平均提高54%。然后，我们制定和演示了规避攻击，通过在不同的威胁表面(包括开放的无线介质本身)注入有界噪声，使覆盖率降低高达50%。重要的是，我们还比较了基于ML的xApp和非基于ML的启发式攻击的有效性。我们最终开发和演示了针对具有挑战性的物理/基于干扰的攻击的基于训练的强大防御，并显示与在一系列噪声预算内不使用防御相比，覆盖率提高了15%



## **35. On Adversarial Examples for Text Classification by Perturbing Latent Representations**

利用扰动潜在表示进行文本分类的对抗示例 cs.LG

7 pages

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03789v1) [paper-pdf](http://arxiv.org/pdf/2405.03789v1)

**Authors**: Korn Sooksatra, Bikram Khanal, Pablo Rivas

**Abstract**: Recently, with the advancement of deep learning, several applications in text classification have advanced significantly. However, this improvement comes with a cost because deep learning is vulnerable to adversarial examples. This weakness indicates that deep learning is not very robust. Fortunately, the input of a text classifier is discrete. Hence, it can prevent the classifier from state-of-the-art attacks. Nonetheless, previous works have generated black-box attacks that successfully manipulate the discrete values of the input to find adversarial examples. Therefore, instead of changing the discrete values, we transform the input into its embedding vector containing real values to perform the state-of-the-art white-box attacks. Then, we convert the perturbed embedding vector back into a text and name it an adversarial example. In summary, we create a framework that measures the robustness of a text classifier by using the gradients of the classifier.

摘要: 最近，随着深度学习的进步，文本分类中的几个应用取得了显着进步。然而，这种改进是有代价的，因为深度学习容易受到对抗性示例的影响。这个弱点表明深度学习不是很强大。幸运的是，文本分类器的输入是离散的。因此，它可以防止分类器受到最先进的攻击。尽管如此，之前的作品已经产生了黑匣子攻击，这些攻击成功地操纵输入的离散值以找到对抗性示例。因此，我们不会更改离散值，而是将输入转换为包含实值的嵌入载体，以执行最先进的白盒攻击。然后，我们将受干扰的嵌入载体转换回文本，并将其命名为对抗性示例。总而言之，我们创建了一个框架，通过使用分类器的梯度来衡量文本分类器的稳健性。



## **36. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

RandOhm：使用随机电路查找器缓解阻抗侧通道攻击 cs.CR

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2401.08925v2) [paper-pdf](http://arxiv.org/pdf/2401.08925v2)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate them. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of the chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits a moving target defense (MTD) strategy based on the partial reconfiguration (PR) feature of mainstream FPGAs and programmable SoCs to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be significantly reduced via runtime reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. Moreover, in contrast to existing PR-based countermeasures, RandOhm deploys open-source bitstream manipulation tools on programmable SoCs to speed up the randomization and provide real-time protection. To validate our claims, we apply RandOhm to AES ciphers realized on 28-nm FPGAs. We analyze the resiliency of our approach by performing non-profiled and profiled impedance analysis attacks and investigate the overhead of our mitigation in terms of delay and performance.

摘要: 物理侧通道攻击可能会危及集成电路的安全性。大多数物理侧通道攻击(例如，电源或电磁)利用芯片的动态行为，通常表现为电流消耗或电压波动的变化，其中算法对策(如掩蔽)可以有效地缓解这些变化。然而，正如最近所证明的那样，这些缓解技术并不能完全有效地对抗诸如阻抗分析之类的反向散射侧信道攻击。在阻抗攻击的情况下，攻击者利用芯片功率传输网络(PDN)的依赖于数据的阻抗变化来提取秘密信息。在这项工作中，我们介绍了RandOhm，它利用了一种基于主流FPGA和可编程SoC的部分重构(PR)特性的移动目标防御(MTD)策略来防御阻抗旁通道攻击。我们证明，通过PDN阻抗的信息泄漏可以通过在运行时重新配置电路的秘密敏感部分来显著减少。因此，通过不断地随机化电路的布局和布线，可以将依赖于数据的计算与阻抗值分离。此外，与现有的基于PR的对策相比，RandOhm在可编程SoC上部署了开源的比特流处理工具，以加快随机化并提供实时保护。为了验证我们的声明，我们将RandOhm应用于在28 nm FPGA上实现的AES密码。我们通过执行非配置文件和配置文件阻抗分析攻击来分析我们方法的弹性，并从延迟和性能方面调查我们的缓解开销。



## **37. Understanding the Vulnerability of Skeleton-based Human Activity Recognition via Black-box Attack**

通过黑匣子攻击了解基于普林斯顿的人类活动识别的漏洞 cs.CV

Accepted in Pattern Recognition. arXiv admin note: substantial text  overlap with arXiv:2103.05266

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2211.11312v2) [paper-pdf](http://arxiv.org/pdf/2211.11312v2)

**Authors**: Yunfeng Diao, He Wang, Tianjia Shao, Yong-Liang Yang, Kun Zhou, David Hogg, Meng Wang

**Abstract**: Human Activity Recognition (HAR) has been employed in a wide range of applications, e.g. self-driving cars, where safety and lives are at stake. Recently, the robustness of skeleton-based HAR methods have been questioned due to their vulnerability to adversarial attacks. However, the proposed attacks require the full-knowledge of the attacked classifier, which is overly restrictive. In this paper, we show such threats indeed exist, even when the attacker only has access to the input/output of the model. To this end, we propose the very first black-box adversarial attack approach in skeleton-based HAR called BASAR. BASAR explores the interplay between the classification boundary and the natural motion manifold. To our best knowledge, this is the first time data manifold is introduced in adversarial attacks on time series. Via BASAR, we find on-manifold adversarial samples are extremely deceitful and rather common in skeletal motions, in contrast to the common belief that adversarial samples only exist off-manifold. Through exhaustive evaluation, we show that BASAR can deliver successful attacks across classifiers, datasets, and attack modes. By attack, BASAR helps identify the potential causes of the model vulnerability and provides insights on possible improvements. Finally, to mitigate the newly identified threat, we propose a new adversarial training approach by leveraging the sophisticated distributions of on/off-manifold adversarial samples, called mixed manifold-based adversarial training (MMAT). MMAT can successfully help defend against adversarial attacks without compromising classification accuracy.

摘要: 人类活动识别(HAR)已被广泛应用于安全和生命受到威胁的自动驾驶汽车等领域。最近，基于骨架的HAR方法的健壮性受到了质疑，因为它们容易受到对手攻击。然而，所提出的攻击需要被攻击分类器的完全知识，这是过度限制的。在这篇文章中，我们证明了这样的威胁确实存在，即使攻击者只有权访问模型的输入/输出。为此，我们在基于骨架的HAR中提出了第一种黑盒对抗攻击方法BASAR。巴萨探索了分类边界和自然运动流形之间的相互作用。据我们所知，这是首次将数据流形引入时间序列的对抗性攻击中。通过BASAR，我们发现流形上的对抗性样本具有极大的欺骗性，并且在骨骼运动中相当常见，而不是通常认为对抗性样本只存在于流形外。通过详尽的评估，我们证明了Basar可以跨分类器、数据集和攻击模式进行成功的攻击。通过攻击，Basar帮助识别模型漏洞的潜在原因，并提供可能改进的见解。最后，为了缓解新识别的威胁，我们提出了一种新的对抗训练方法，即基于混合流形的对抗训练(MMAT)。MMAT可以在不影响分类准确性的情况下成功地帮助防御对手攻击。



## **38. Provably Unlearnable Examples**

可证明难以学习的例子 cs.LG

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03316v1) [paper-pdf](http://arxiv.org/pdf/2405.03316v1)

**Authors**: Derui Wang, Minhui Xue, Bo Li, Seyit Camtepe, Liming Zhu

**Abstract**: The exploitation of publicly accessible data has led to escalating concerns regarding data privacy and intellectual property (IP) breaches in the age of artificial intelligence. As a strategy to safeguard both data privacy and IP-related domain knowledge, efforts have been undertaken to render shared data unlearnable for unauthorized models in the wild. Existing methods apply empirically optimized perturbations to the data in the hope of disrupting the correlation between the inputs and the corresponding labels such that the data samples are converted into Unlearnable Examples (UEs). Nevertheless, the absence of mechanisms that can verify how robust the UEs are against unknown unauthorized models and train-time techniques engenders several problems. First, the empirically optimized perturbations may suffer from the problem of cross-model generalization, which echoes the fact that the unauthorized models are usually unknown to the defender. Second, UEs can be mitigated by train-time techniques such as data augmentation and adversarial training. Furthermore, we find that a simple recovery attack can restore the clean-task performance of the classifiers trained on UEs by slightly perturbing the learned weights. To mitigate the aforementioned problems, in this paper, we propose a mechanism for certifying the so-called $(q, \eta)$-Learnability of an unlearnable dataset via parametric smoothing. A lower certified $(q, \eta)$-Learnability indicates a more robust protection over the dataset. Finally, we try to 1) improve the tightness of certified $(q, \eta)$-Learnability and 2) design Provably Unlearnable Examples (PUEs) which have reduced $(q, \eta)$-Learnability. According to experimental results, PUEs demonstrate both decreased certified $(q, \eta)$-Learnability and enhanced empirical robustness compared to existing UEs.

摘要: 在人工智能时代，对公开可访问数据的利用导致了人们对数据隐私和知识产权(IP)侵犯的担忧不断升级。作为一项保护数据隐私和知识产权相关领域知识的战略，已经做出努力，使未经授权的模型无法在野外学习共享数据。现有方法将经验优化的扰动应用于数据，希望破坏输入和相应标签之间的相关性，从而将数据样本转换为不可学习的示例(UE)。然而，缺乏机制来验证UE对未知的未经授权的模型和训练时间技术的健壮性，会产生几个问题。首先，经验优化的扰动可能会受到跨模型泛化的问题，这呼应了这样一个事实，即未经授权的模型通常对于防御者是未知的。其次，可以通过数据增强和对抗性训练等训练时间技术来缓解UE。此外，我们发现，简单的恢复攻击可以通过对学习的权重进行轻微扰动来恢复在UE上训练的分类器的干净任务性能。为了缓解上述问题，在本文中，我们提出了一种通过参数平滑来证明不可学习数据集的所谓$(Q，\eta)$-可学习性的机制。较低的认证$(Q，\eta)$-可学习性表明对数据集的保护更强大。最后，我们试图1)提高已证明的$(Q，eta)$-可学习性的紧性；2)设计降低了$(q，eta)$-可学习性的可证明不可学习实例(PUE)。实验结果表明，与已有的UE相比，PUE的认证$(Q，ETA)可学习性降低，经验稳健性增强。



## **39. Illusory Attacks: Information-Theoretic Detectability Matters in Adversarial Attacks**

幻象攻击：信息论可检测性在对抗性攻击中很重要 cs.AI

ICLR 2024 Spotlight (top 5%)

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2207.10170v5) [paper-pdf](http://arxiv.org/pdf/2207.10170v5)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of information-theoretic detectability constraints makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce {\epsilon}-illusory, a novel form of adversarial attack on sequential decision-makers that is both effective and of {\epsilon}-bounded statistical detectability. We propose a novel dual ascent algorithm to learn such attacks end-to-end. Compared to existing attacks, we empirically find {\epsilon}-illusory to be significantly harder to detect with automated methods, and a small study with human participants (IRB approval under reference R84123/RE001) suggests they are similarly harder to detect for humans. Our findings suggest the need for better anomaly detectors, as well as effective hardware- and system-level defenses. The project website can be found at https://tinyurl.com/illusory-attacks.

摘要: 部署在现实世界中的自主代理需要强大地抵御对感觉输入的敌意攻击。将代理策略规模化需要预测可能最强的攻击。我们证明了现有的对强化学习代理的观察空间攻击有一个共同的弱点：虽然有效，但它们缺乏信息论的可检测性约束，使得它们可以使用自动手段或人工检查来检测。对于对手来说，可探测性是不可取的，因为它可能会引发安全升级。介绍了一种新的针对序列决策者的对抗性攻击--{epsilon}-幻觉，它既是有效的，又具有{epsilon}-有界的统计可检测性。我们提出了一种新的双重上升算法来端到端地学习此类攻击。与现有的攻击相比，我们根据经验发现，使用自动方法检测{\epsilon}-幻觉要困难得多，一项针对人类参与者的小型研究(参考R84123/RE001下的IRB批准)表明，对于人类来说，它们同样更难检测到。我们的发现表明，需要更好的异常检测器，以及有效的硬件和系统级防御。该项目的网址为：https://tinyurl.com/illusory-attacks.



## **40. Purify Unlearnable Examples via Rate-Constrained Variational Autoencoders**

通过速率约束变分自动编码器净化不可学习的示例 cs.CR

Accepted by ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.01460v2) [paper-pdf](http://arxiv.org/pdf/2405.01460v2)

**Authors**: Yi Yu, Yufei Wang, Song Xia, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Unlearnable examples (UEs) seek to maximize testing error by making subtle modifications to training examples that are correctly labeled. Defenses against these poisoning attacks can be categorized based on whether specific interventions are adopted during training. The first approach is training-time defense, such as adversarial training, which can mitigate poisoning effects but is computationally intensive. The other approach is pre-training purification, e.g., image short squeezing, which consists of several simple compressions but often encounters challenges in dealing with various UEs. Our work provides a novel disentanglement mechanism to build an efficient pre-training purification method. Firstly, we uncover rate-constrained variational autoencoders (VAEs), demonstrating a clear tendency to suppress the perturbations in UEs. We subsequently conduct a theoretical analysis for this phenomenon. Building upon these insights, we introduce a disentangle variational autoencoder (D-VAE), capable of disentangling the perturbations with learnable class-wise embeddings. Based on this network, a two-stage purification approach is naturally developed. The first stage focuses on roughly eliminating perturbations, while the second stage produces refined, poison-free results, ensuring effectiveness and robustness across various scenarios. Extensive experiments demonstrate the remarkable performance of our method across CIFAR-10, CIFAR-100, and a 100-class ImageNet-subset. Code is available at https://github.com/yuyi-sd/D-VAE.

摘要: 不能学习的例子(UE)试图通过对正确标记的训练例子进行微妙的修改来最大化测试误差。针对这些中毒攻击的防御措施可以根据是否在训练期间采取特定干预措施进行分类。第一种方法是训练时间防御，例如对抗性训练，这种方法可以减轻中毒影响，但计算密集。另一种方法是训练前净化，例如图像短压缩，它由几个简单的压缩组成，但在处理各种UE时经常遇到挑战。我们的工作为构建高效的预训练净化方法提供了一种新的解缠机制。首先，我们发现了码率受限的变分自动编码器(VAE)，显示了抑制UE中微扰的明显趋势。我们随后对这一现象进行了理论分析。基于这些见解，我们引入了一种解缠变分自动编码器(D-VAE)，它能够通过可学习的类嵌入来解缠扰动。在这个网络的基础上，自然发展了一种两级提纯方法。第一阶段侧重于粗略地消除干扰，而第二阶段产生精炼的、无毒的结果，确保在各种情况下的有效性和健壮性。广泛的实验表明，我们的方法在CIFAR-10、CIFAR-100和100类ImageNet子集上具有显著的性能。代码可在https://github.com/yuyi-sd/D-VAE.上找到



## **41. Are aligned neural networks adversarially aligned?**

对齐的神经网络是否反向对齐？ cs.CL

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2306.15447v2) [paper-pdf](http://arxiv.org/pdf/2306.15447v2)

**Authors**: Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Anas Awadalla, Pang Wei Koh, Daphne Ippolito, Katherine Lee, Florian Tramer, Ludwig Schmidt

**Abstract**: Large language models are now tuned to align with the goals of their creators, namely to be "helpful and harmless." These models should respond helpfully to user questions, but refuse to answer requests that could cause harm. However, adversarial users can construct inputs which circumvent attempts at alignment. In this work, we study adversarial alignment, and ask to what extent these models remain aligned when interacting with an adversarial user who constructs worst-case inputs (adversarial examples). These inputs are designed to cause the model to emit harmful content that would otherwise be prohibited. We show that existing NLP-based optimization attacks are insufficiently powerful to reliably attack aligned text models: even when current NLP-based attacks fail, we can find adversarial inputs with brute force. As a result, the failure of current attacks should not be seen as proof that aligned text models remain aligned under adversarial inputs.   However the recent trend in large-scale ML models is multimodal models that allow users to provide images that influence the text that is generated. We show these models can be easily attacked, i.e., induced to perform arbitrary un-aligned behavior through adversarial perturbation of the input image. We conjecture that improved NLP attacks may demonstrate this same level of adversarial control over text-only models.

摘要: 大型语言模型现在被调整为与它们的创建者的目标保持一致，即“有益和无害”。这些模型应该对用户的问题做出有益的回应，但拒绝回答可能造成伤害的请求。然而，敌意用户可以构建绕过对齐尝试的输入。在这项工作中，我们研究对抗性对齐，并询问当与构建最坏情况输入(对抗性例子)的对抗性用户交互时，这些模型在多大程度上保持对齐。这些输入旨在导致模型排放本来被禁止的有害内容。我们证明了现有的基于NLP的优化攻击不足以可靠地攻击对齐的文本模型：即使当前基于NLP的攻击失败，我们也可以发现具有暴力的敌意输入。因此，当前攻击的失败不应被视为对齐的文本模型在敌意输入下保持对齐的证据。然而，大规模ML模型的最新趋势是允许用户提供影响所生成文本的图像的多模式模型。我们证明了这些模型可以很容易地被攻击，即通过对输入图像的对抗性扰动来诱导执行任意的非对齐行为。我们推测，改进的NLP攻击可能会展示出对纯文本模型的同样水平的敌意控制。



## **42. Exploring Frequencies via Feature Mixing and Meta-Learning for Improving Adversarial Transferability**

通过特征混合和元学习探索频率以提高对抗性可移植性 cs.CV

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03193v1) [paper-pdf](http://arxiv.org/pdf/2405.03193v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Shaozi Li

**Abstract**: Recent studies have shown that Deep Neural Networks (DNNs) are susceptible to adversarial attacks, with frequency-domain analysis underscoring the significance of high-frequency components in influencing model predictions. Conversely, targeting low-frequency components has been effective in enhancing attack transferability on black-box models. In this study, we introduce a frequency decomposition-based feature mixing method to exploit these frequency characteristics in both clean and adversarial samples. Our findings suggest that incorporating features of clean samples into adversarial features extracted from adversarial examples is more effective in attacking normally-trained models, while combining clean features with the adversarial features extracted from low-frequency parts decomposed from the adversarial samples yields better results in attacking defense models. However, a conflict issue arises when these two mixing approaches are employed simultaneously. To tackle the issue, we propose a cross-frequency meta-optimization approach comprising the meta-train step, meta-test step, and final update. In the meta-train step, we leverage the low-frequency components of adversarial samples to boost the transferability of attacks against defense models. Meanwhile, in the meta-test step, we utilize adversarial samples to stabilize gradients, thereby enhancing the attack's transferability against normally trained models. For the final update, we update the adversarial sample based on the gradients obtained from both meta-train and meta-test steps. Our proposed method is evaluated through extensive experiments on the ImageNet-Compatible dataset, affirming its effectiveness in improving the transferability of attacks on both normally-trained CNNs and defense models.   The source code is available at https://github.com/WJJLL/MetaSSA.

摘要: 最近的研究表明，深度神经网络(DNN)容易受到敌意攻击，频域分析强调了高频分量在影响模型预测中的重要性。相反，瞄准低频分量在增强黑盒模型上的攻击可转移性方面是有效的。在这项研究中，我们引入了一种基于频率分解的特征混合方法来利用干净样本和恶意样本中的这些频率特征。我们的结果表明，在攻击正常训练的模型时，将干净样本的特征与从对抗性样本中提取的对抗性特征相结合是更有效的，而将干净特征与从对抗性样本分解的低频部分提取的对抗性特征相结合，在攻击防御模型中会产生更好的效果。然而，当这两种混合方法同时使用时，就会出现冲突问题。为了解决这一问题，我们提出了一种跨频率元优化方法，包括元训练步骤、元测试步骤和最终更新。在元训练步骤中，我们利用对抗性样本的低频成分来提高攻击对防御模型的可转移性。同时，在元测试步骤中，我们利用对抗性样本来稳定梯度，从而增强了攻击对正常训练模型的可转移性。对于最终的更新，我们基于从元训练和元测试步骤获得的梯度来更新对抗性样本。通过在与ImageNet兼容的数据集上的大量实验对我们提出的方法进行了评估，证实了该方法在提高对正常训练的CNN和防御模型的攻击的可转移性方面的有效性。源代码可在https://github.com/WJJLL/MetaSSA.上找到



## **43. To Each (Textual Sequence) Its Own: Improving Memorized-Data Unlearning in Large Language Models**

每个（文本序列）自有：改进大型语言模型中的简化数据去学习 cs.LG

Published as a conference paper at ICML 2024

**SubmitDate**: 2024-05-06    [abs](http://arxiv.org/abs/2405.03097v1) [paper-pdf](http://arxiv.org/pdf/2405.03097v1)

**Authors**: George-Octavian Barbulescu, Peter Triantafillou

**Abstract**: LLMs have been found to memorize training textual sequences and regurgitate verbatim said sequences during text generation time. This fact is known to be the cause of privacy and related (e.g., copyright) problems. Unlearning in LLMs then takes the form of devising new algorithms that will properly deal with these side-effects of memorized data, while not hurting the model's utility. We offer a fresh perspective towards this goal, namely, that each textual sequence to be forgotten should be treated differently when being unlearned based on its degree of memorization within the LLM. We contribute a new metric for measuring unlearning quality, an adversarial attack showing that SOTA algorithms lacking this perspective fail for privacy, and two new unlearning methods based on Gradient Ascent and Task Arithmetic, respectively. A comprehensive performance evaluation across an extensive suite of NLP tasks then mapped the solution space, identifying the best solutions under different scales in model capacities and forget set sizes and quantified the gains of the new approaches.

摘要: 已经发现LLM在文本生成时间内记忆训练文本序列并逐字地返回所述序列。众所周知，这一事实是隐私和相关(例如，版权)问题的原因。然后，在LLMS中，遗忘的形式是设计新的算法，这些算法将适当地处理记忆数据的这些副作用，同时不会损害模型的实用性。我们为这一目标提供了一个新的视角，即每个被遗忘的文本序列在被遗忘时应该根据它在LLM中的记忆程度而得到不同的对待。我们提出了一种新的遗忘质量度量，一种敌意攻击表明缺乏这种视角的SOTA算法在隐私方面是失败的，以及两种新的遗忘方法，分别基于梯度上升和任务算法。然后，对一系列NLP任务进行了全面的性能评估，绘制了解决方案空间图，确定了模型容量和忘记集合大小不同尺度下的最佳解决方案，并量化了新方法的收益。



## **44. A Characterization of Semi-Supervised Adversarially-Robust PAC Learnability**

半监督对抗鲁棒PAC可学习性的描述 cs.LG

NeurIPS 2022 camera-ready

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2202.05420v3) [paper-pdf](http://arxiv.org/pdf/2202.05420v3)

**Authors**: Idan Attias, Steve Hanneke, Yishay Mansour

**Abstract**: We study the problem of learning an adversarially robust predictor to test time attacks in the semi-supervised PAC model. We address the question of how many labeled and unlabeled examples are required to ensure learning. We show that having enough unlabeled data (the size of a labeled sample that a fully-supervised method would require), the labeled sample complexity can be arbitrarily smaller compared to previous works, and is sharply characterized by a different complexity measure. We prove nearly matching upper and lower bounds on this sample complexity. This shows that there is a significant benefit in semi-supervised robust learning even in the worst-case distribution-free model, and establishes a gap between the supervised and semi-supervised label complexities which is known not to hold in standard non-robust PAC learning.

摘要: 我们研究学习对抗鲁棒预测器以测试半监督PAC模型中的时间攻击的问题。我们解决了需要多少带标签和未带标签的示例来确保学习的问题。我们表明，拥有足够的未标记数据（全监督方法所需的标记样本的大小），标记样本的复杂性与之前的作品相比可以任意小，并且由不同的复杂性衡量标准来鲜明地特征。我们证明了该样本复杂性的上下限几乎匹配。这表明，即使在最坏的无分布模型中，半监督鲁棒学习也有显着的好处，并在监督和半监督标签复杂性之间建立了差距，众所周知，这在标准非鲁棒PAC学习中不存在。



## **45. Adversarially Robust PAC Learnability of Real-Valued Functions**

实值函数的对抗鲁棒PAC可学习性 cs.LG

accepted to ICML2023

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2206.12977v3) [paper-pdf](http://arxiv.org/pdf/2206.12977v3)

**Authors**: Idan Attias, Steve Hanneke

**Abstract**: We study robustness to test-time adversarial attacks in the regression setting with $\ell_p$ losses and arbitrary perturbation sets. We address the question of which function classes are PAC learnable in this setting. We show that classes of finite fat-shattering dimension are learnable in both realizable and agnostic settings. Moreover, for convex function classes, they are even properly learnable. In contrast, some non-convex function classes provably require improper learning algorithms. Our main technique is based on a construction of an adversarially robust sample compression scheme of a size determined by the fat-shattering dimension. Along the way, we introduce a novel agnostic sample compression scheme for real-valued functions, which may be of independent interest.

摘要: 我们研究了在具有$\ell_p$损失和任意扰动集的回归设置中对测试时对抗攻击的鲁棒性。我们解决了在这种环境下哪些函数类可以PAC学习的问题。我们表明，有限的脂肪粉碎维度的类别在可实现和不可知的环境中都是可以学习的。此外，对于凸函数类来说，它们甚至是可以正确学习的。相比之下，一些非凸函数类可以证明需要不当的学习算法。我们的主要技术基于构建一个具有对抗性的稳健样本压缩方案，其大小由脂肪粉碎维度确定。一路上，我们为实值函数引入了一种新颖的不可知样本压缩方案，这可能是一种独立的兴趣。



## **46. Defense against Joint Poison and Evasion Attacks: A Case Study of DERMS**

防御联合毒物和躲避攻击：DEMS案例研究 cs.CR

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02989v1) [paper-pdf](http://arxiv.org/pdf/2405.02989v1)

**Authors**: Zain ul Abdeen, Padmaksha Roy, Ahmad Al-Tawaha, Rouxi Jia, Laura Freeman, Peter Beling, Chen-Ching Liu, Alberto Sangiovanni-Vincentelli, Ming Jin

**Abstract**: There is an upward trend of deploying distributed energy resource management systems (DERMS) to control modern power grids. However, DERMS controller communication lines are vulnerable to cyberattacks that could potentially impact operational reliability. While a data-driven intrusion detection system (IDS) can potentially thwart attacks during deployment, also known as the evasion attack, the training of the detection algorithm may be corrupted by adversarial data injected into the database, also known as the poisoning attack. In this paper, we propose the first framework of IDS that is robust against joint poisoning and evasion attacks. We formulate the defense mechanism as a bilevel optimization, where the inner and outer levels deal with attacks that occur during training time and testing time, respectively. We verify the robustness of our method on the IEEE-13 bus feeder model against a diverse set of poisoning and evasion attack scenarios. The results indicate that our proposed method outperforms the baseline technique in terms of accuracy, precision, and recall for intrusion detection.

摘要: 采用分布式能源管理系统(DERMS)来控制现代电网是一种趋势。然而，DERMS控制器通信线路容易受到网络攻击，这些攻击可能会潜在地影响操作可靠性。虽然数据驱动的入侵检测系统(IDS)可以潜在地阻止部署期间的攻击，也称为逃避攻击，但检测算法的训练可能会被注入数据库的敌意数据破坏，也称为中毒攻击。在本文中，我们提出了第一个入侵检测系统的框架，该框架对联合中毒和逃避攻击具有健壮性。我们将防御机制描述为双层优化，其中内部和外部级别分别处理在训练时间和测试时间发生的攻击。我们在IEEE-13公交支线模型上验证了该方法对不同的中毒和逃避攻击场景的稳健性。实验结果表明，该方法在入侵检测的准确率、精确度和召回率方面均优于Baseline方法。



## **47. You Only Need Half: Boosting Data Augmentation by Using Partial Content**

您只需要一半：通过使用部分内容来增强数据增强 cs.CV

Technical report,16 pages

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02830v1) [paper-pdf](http://arxiv.org/pdf/2405.02830v1)

**Authors**: Juntao Hu, Yuan Wu

**Abstract**: We propose a novel data augmentation method termed You Only Need hAlf (YONA), which simplifies the augmentation process. YONA bisects an image, substitutes one half with noise, and applies data augmentation techniques to the remaining half. This method reduces the redundant information in the original image, encourages neural networks to recognize objects from incomplete views, and significantly enhances neural networks' robustness. YONA is distinguished by its properties of parameter-free, straightforward application, enhancing various existing data augmentation strategies, and thereby bolstering neural networks' robustness without additional computational cost. To demonstrate YONA's efficacy, extensive experiments were carried out. These experiments confirm YONA's compatibility with diverse data augmentation methods and neural network architectures, yielding substantial improvements in CIFAR classification tasks, sometimes outperforming conventional image-level data augmentation methods. Furthermore, YONA markedly increases the resilience of neural networks to adversarial attacks. Additional experiments exploring YONA's variants conclusively show that masking half of an image optimizes performance. The code is available at https://github.com/HansMoe/YONA.

摘要: 我们提出了一种新的数据增强方法，称为你只需要一半(YONA)，它简化了增强过程。Yona将图像一分为二，用噪声替换一半，并对其余一半应用数据增强技术。该方法减少了原始图像中的冗余信息，鼓励神经网络从不完整的图像中识别目标，显著增强了神经网络的鲁棒性。YONA的特点是无参数，应用简单，增强了现有的各种数据增强策略，从而在不增加计算成本的情况下增强了神经网络的健壮性。为了证明Yona的疗效，进行了广泛的实验。这些实验证实了Yona与各种数据增强方法和神经网络体系结构的兼容性，在CIFAR分类任务方面产生了实质性的改进，有时性能优于传统的图像级数据增强方法。此外，Yona显著提高了神经网络对对手攻击的弹性。探索Yona变体的其他实验最终表明，遮盖图像的一半可以优化性能。代码可在https://github.com/HansMoe/YONA.上获得



## **48. Trojans in Large Language Models of Code: A Critical Review through a Trigger-Based Taxonomy**

大型语言代码模型中的特洛伊木马：基于触发器的分类学的批判性评论 cs.SE

arXiv admin note: substantial text overlap with arXiv:2305.03803

**SubmitDate**: 2024-05-05    [abs](http://arxiv.org/abs/2405.02828v1) [paper-pdf](http://arxiv.org/pdf/2405.02828v1)

**Authors**: Aftab Hussain, Md Rafiqul Islam Rabin, Toufique Ahmed, Bowen Xu, Premkumar Devanbu, Mohammad Amin Alipour

**Abstract**: Large language models (LLMs) have provided a lot of exciting new capabilities in software development. However, the opaque nature of these models makes them difficult to reason about and inspect. Their opacity gives rise to potential security risks, as adversaries can train and deploy compromised models to disrupt the software development process in the victims' organization.   This work presents an overview of the current state-of-the-art trojan attacks on large language models of code, with a focus on triggers -- the main design point of trojans -- with the aid of a novel unifying trigger taxonomy framework. We also aim to provide a uniform definition of the fundamental concepts in the area of trojans in Code LLMs. Finally, we draw implications of findings on how code models learn on trigger design.

摘要: 大型语言模型（LLM）在软件开发中提供了许多令人兴奋的新功能。然而，这些模型的不透明性质使得它们难以推理和检查。它们的不透明性会带来潜在的安全风险，因为对手可以训练和部署受影响的模型，以扰乱受害者组织的软件开发流程。   这项工作概述了当前针对大型语言代码模型的最新特洛伊木马攻击，重点关注触发器（特洛伊木马的主要设计点），并在新颖的统一触发器分类框架的帮助下。我们还旨在为LLM代码中特洛伊木马领域的基本概念提供统一的定义。最后，我们得出了有关代码模型如何学习触发器设计的研究结果的影响。



## **49. Assessing Adversarial Robustness of Large Language Models: An Empirical Study**

评估大型语言模型的对抗稳健性：实证研究 cs.CL

16 pages, 9 figures, 10 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02764v1) [paper-pdf](http://arxiv.org/pdf/2405.02764v1)

**Authors**: Zeyu Yang, Zhao Meng, Xiaochen Zheng, Roger Wattenhofer

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing, but their robustness against adversarial attacks remains a critical concern. We presents a novel white-box style attack approach that exposes vulnerabilities in leading open-source LLMs, including Llama, OPT, and T5. We assess the impact of model size, structure, and fine-tuning strategies on their resistance to adversarial perturbations. Our comprehensive evaluation across five diverse text classification tasks establishes a new benchmark for LLM robustness. The findings of this study have far-reaching implications for the reliable deployment of LLMs in real-world applications and contribute to the advancement of trustworthy AI systems.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但其对抗攻击的稳健性仍然是一个关键问题。我们提出了一种新颖的白盒式攻击方法，该方法暴露了领先开源LLM（包括Llama、OPT和T5）中的漏洞。我们评估了模型大小、结构和微调策略对其抵抗对抗性扰动的影响。我们对五种不同文本分类任务的全面评估为LLM稳健性建立了新基准。这项研究的结果对于LLM在现实世界应用程序中的可靠部署具有深远的影响，并有助于发展值得信赖的人工智能系统。



## **50. Updating Windows Malware Detectors: Balancing Robustness and Regression against Adversarial EXEmples**

更新Windows恶意软件检测器：平衡稳健性和回归与对抗性示例 cs.CR

11 pages, 3 figures, 7 tables

**SubmitDate**: 2024-05-04    [abs](http://arxiv.org/abs/2405.02646v1) [paper-pdf](http://arxiv.org/pdf/2405.02646v1)

**Authors**: Matous Kozak, Luca Demetrio, Dmitrijs Trizna, Fabio Roli

**Abstract**: Adversarial EXEmples are carefully-perturbed programs tailored to evade machine learning Windows malware detectors, with an on-going effort in developing robust models able to address detection effectiveness. However, even if robust models can prevent the majority of EXEmples, to maintain predictive power over time, models are fine-tuned to newer threats, leading either to partial updates or time-consuming retraining from scratch. Thus, even if the robustness against attacks is higher, the new models might suffer a regression in performance by misclassifying threats that were previously correctly detected. For these reasons, we study the trade-off between accuracy and regression when updating Windows malware detectors, by proposing EXE-scanner, a plugin that can be chained to existing detectors to promptly stop EXEmples without causing regression. We empirically show that previously-proposed hardening techniques suffer a regression of accuracy when updating non-robust models. On the contrary, we show that EXE-scanner exhibits comparable performance to robust models without regression of accuracy, and we show how to properly chain it after the base classifier to obtain the best performance without the need of costly retraining. To foster reproducibility, we openly release source code, along with the dataset of adversarial EXEmples based on state-of-the-art perturbation algorithms.

摘要: 对抗性的例子是精心设计的程序，旨在逃避机器学习Windows恶意软件检测器，并正在努力开发能够解决检测有效性的健壮模型。然而，即使稳健的模型可以阻止大多数例子，为了随着时间的推移保持预测能力，模型也会针对较新的威胁进行微调，导致要么部分更新，要么从头开始进行耗时的再培训。因此，即使对攻击的稳健性更高，新模型也可能会因为对以前正确检测到的威胁进行错误分类而导致性能退化。出于这些原因，我们研究了更新Windows恶意软件检测器时准确性和回归之间的权衡，提出了EXE-scanner，这是一个可以链接到现有检测器的插件，可以在不导致回归的情况下迅速停止示例。我们的经验表明，以前提出的强化技术在更新非稳健模型时会遭遇精度回归。相反，我们证明了EXE-scanner在没有精度回归的情况下表现出与健壮模型相当的性能，并展示了如何在基本分类器之后适当地将其链接以获得最佳性能，而不需要昂贵的再训练。为了促进重复性，我们公开发布源代码，以及基于最先进的扰动算法的对抗性例子的数据集。



