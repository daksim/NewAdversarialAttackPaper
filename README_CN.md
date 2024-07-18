# Latest Adversarial Attack Papers
**update at 2024-07-18 16:50:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Investigating Adversarial Vulnerability and Implicit Bias through Frequency Analysis**

通过频率分析调查对抗脆弱性和隐性偏见 cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2305.15203v2) [paper-pdf](http://arxiv.org/pdf/2305.15203v2)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification tasks, neural networks are known to be vulnerable to adversarial attacks, subtle perturbations of the input data designed to deceive the model. In this work, we investigate the relation between these perturbations and the implicit bias of neural networks trained with gradient-based algorithms. To this end, we analyse the network's implicit bias through the lens of the Fourier transform. Specifically, we identify the minimal and most critical frequencies necessary for accurate classification or misclassification respectively for each input image and its adversarially perturbed version, and uncover the correlation among those. To this end, among other methods, we use a newly introduced technique capable of detecting non-linear correlations between high-dimensional datasets. Our results provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are highly correlated and suggest new potential strategies for adversarial defence.

摘要: 尽管神经网络在分类任务中的表现令人印象深刻，但众所周知，它很容易受到对抗性攻击，即输入数据的微妙扰动，目的是欺骗模型。在这项工作中，我们研究了这些扰动与用基于梯度的算法训练的神经网络的隐偏差之间的关系。为此，我们通过傅里叶变换的透镜分析了网络的隐含偏差。具体地说，我们分别为每个输入图像及其相反的扰动版本识别准确分类或误分类所需的最小和最关键频率，并揭示这些频率之间的相关性。为此，在其他方法中，我们使用了一种新引入的技术，能够检测高维数据集之间的非线性相关性。我们的结果提供了经验证据，证明了傅立叶空间中的网络偏差与敌方攻击的目标频率高度相关，并为敌方防御提供了新的潜在策略。



## **2. Muting Whisper: A Universal Acoustic Adversarial Attack on Speech Foundation Models**

静音低语：对语音基础模型的通用声学对抗攻击 cs.CL

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.06134v2) [paper-pdf](http://arxiv.org/pdf/2405.06134v2)

**Authors**: Vyas Raina, Rao Ma, Charles McGhee, Kate Knill, Mark Gales

**Abstract**: Recent developments in large speech foundation models like Whisper have led to their widespread use in many automatic speech recognition (ASR) applications. These systems incorporate `special tokens' in their vocabulary, such as $\texttt{<|endoftext|>}$, to guide their language generation process. However, we demonstrate that these tokens can be exploited by adversarial attacks to manipulate the model's behavior. We propose a simple yet effective method to learn a universal acoustic realization of Whisper's $\texttt{<|endoftext|>}$ token, which, when prepended to any speech signal, encourages the model to ignore the speech and only transcribe the special token, effectively `muting' the model. Our experiments demonstrate that the same, universal 0.64-second adversarial audio segment can successfully mute a target Whisper ASR model for over 97\% of speech samples. Moreover, we find that this universal adversarial audio segment often transfers to new datasets and tasks. Overall this work demonstrates the vulnerability of Whisper models to `muting' adversarial attacks, where such attacks can pose both risks and potential benefits in real-world settings: for example the attack can be used to bypass speech moderation systems, or conversely the attack can also be used to protect private speech data.

摘要: 像Whisper这样的大型语音基础模型的最新发展导致它们在许多自动语音识别(ASR)应用中被广泛使用。这些系统在其词汇表中加入了“特殊标记”，如$\exttt{<|endoftext|>}$，以指导其语言生成过程。然而，我们证明了这些令牌可以被敌意攻击利用来操纵模型的行为。我们提出了一种简单而有效的方法来学习Whisper的$\exttt{}$标记的通用声学实现，当该标记被预先添加到任何语音信号时，鼓励模型忽略语音而只转录特殊的标记，从而有效地抑制了模型。我们的实验表明，相同的、通用的0.64秒的对抗性音频片段可以成功地使目标Whisper ASR模型在97%以上的语音样本上静音。此外，我们发现这种普遍的对抗性音频片段经常转移到新的数据集和任务。总体而言，这项工作证明了Whisper模型对“静音”对手攻击的脆弱性，在现实世界中，这种攻击既可以带来风险，也可以带来潜在的好处：例如，攻击可以用来绕过语音调节系统，或者反过来，攻击也可以用来保护私人语音数据。



## **3. Similarity of Neural Architectures using Adversarial Attack Transferability**

使用对抗攻击可转移性的神经架构相似性 cs.LG

ECCV 2024; 35pages, 2.56MB

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2210.11407v4) [paper-pdf](http://arxiv.org/pdf/2210.11407v4)

**Authors**: Jaehui Hwang, Dongyoon Han, Byeongho Heo, Song Park, Sanghyuk Chun, Jong-Seok Lee

**Abstract**: In recent years, many deep neural architectures have been developed for image classification. Whether they are similar or dissimilar and what factors contribute to their (dis)similarities remains curious. To address this question, we aim to design a quantitative and scalable similarity measure between neural architectures. We propose Similarity by Attack Transferability (SAT) from the observation that adversarial attack transferability contains information related to input gradients and decision boundaries widely used to understand model behaviors. We conduct a large-scale analysis on 69 state-of-the-art ImageNet classifiers using our proposed similarity function to answer the question. Moreover, we observe neural architecture-related phenomena using model similarity that model diversity can lead to better performance on model ensembles and knowledge distillation under specific conditions. Our results provide insights into why developing diverse neural architectures with distinct components is necessary.

摘要: 近年来，已经发展了许多用于图像分类的深层神经结构。它们是相似的还是不相似的，以及是什么因素导致了它们(不同)的相似之处，仍然令人好奇。为了解决这个问题，我们的目标是设计一种量化的、可伸缩的神经结构之间的相似性度量。基于对抗性攻击的可移动性包含与输入梯度和决策边界有关的信息，被广泛用于理解模型行为，我们提出了攻击可转移性相似性(SAT)。我们使用我们提出的相似度函数对69个最先进的ImageNet分类器进行了大规模的分析。此外，我们使用模型相似性来观察与神经结构相关的现象，即在特定条件下，模型多样性可以在模型集成和知识提取方面带来更好的性能。我们的结果为为什么开发具有不同组件的不同神经架构提供了洞察力。



## **4. Benchmarking Robust Self-Supervised Learning Across Diverse Downstream Tasks**

在不同的下游任务中对稳健的自我监督学习进行基准测试 cs.CV

Accepted at the ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12588v1) [paper-pdf](http://arxiv.org/pdf/2407.12588v1)

**Authors**: Antoni Kowalczuk, Jan Dubiński, Atiyeh Ashari Ghomi, Yi Sui, George Stein, Jiapeng Wu, Jesse C. Cresswell, Franziska Boenisch, Adam Dziedzic

**Abstract**: Large-scale vision models have become integral in many applications due to their unprecedented performance and versatility across downstream tasks. However, the robustness of these foundation models has primarily been explored for a single task, namely image classification. The vulnerability of other common vision tasks, such as semantic segmentation and depth estimation, remains largely unknown. We present a comprehensive empirical evaluation of the adversarial robustness of self-supervised vision encoders across multiple downstream tasks. Our attacks operate in the encoder embedding space and at the downstream task output level. In both cases, current state-of-the-art adversarial fine-tuning techniques tested only for classification significantly degrade clean and robust performance on other tasks. Since the purpose of a foundation model is to cater to multiple applications at once, our findings reveal the need to enhance encoder robustness more broadly. %We discuss potential strategies for more robust foundation vision models across diverse downstream tasks. Our code is available at $\href{https://github.com/layer6ai-labs/ssl-robustness}{github.com/layer6ai-labs/ssl-robustness}$.

摘要: 大规模视觉模型已经成为许多应用中不可或缺的一部分，因为它们具有前所未有的性能和跨下游任务的多功能性。然而，这些基础模型的稳健性主要是针对单个任务来探索的，即图像分类。其他常见的视觉任务，如语义分割和深度估计，其脆弱性在很大程度上仍然未知。我们提出了一个全面的经验评估的对抗性自监督视觉编码器跨越多个下游任务。我们的攻击在编码器嵌入空间和下游任务输出级别进行。在这两种情况下，当前最先进的对抗性微调技术仅针对分类进行测试，显著降低了其他任务的干净和健壮的性能。由于基础模型的目的是同时迎合多个应用，我们的研究结果揭示了更广泛地增强编码器健壮性的必要性。%我们讨论在不同的下游任务中建立更强大的基础愿景模型的潜在策略。我们的代码可以在$\href{https://github.com/layer6ai-labs/ssl-robustness}{github.com/layer6ai-labs/ssl-robustness}$.上找到



## **5. Open-Vocabulary Object Detectors: Robustness Challenges under Distribution Shifts**

开放词汇对象检测器：分布转移下的鲁棒性挑战 cs.CV

14 + 3 single column pages

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2405.14874v3) [paper-pdf](http://arxiv.org/pdf/2405.14874v3)

**Authors**: Prakash Chandra Chhipa, Kanjar De, Meenakshi Subhash Chippa, Rajkumar Saini, Marcus Liwicki

**Abstract**: The challenge of Out-Of-Distribution (OOD) robustness remains a critical hurdle towards deploying deep vision models. Vision-Language Models (VLMs) have recently achieved groundbreaking results. VLM-based open-vocabulary object detection extends the capabilities of traditional object detection frameworks, enabling the recognition and classification of objects beyond predefined categories. Investigating OOD robustness in recent open-vocabulary object detection is essential to increase the trustworthiness of these models. This study presents a comprehensive robustness evaluation of the zero-shot capabilities of three recent open-vocabulary (OV) foundation object detection models: OWL-ViT, YOLO World, and Grounding DINO. Experiments carried out on the robustness benchmarks COCO-O, COCO-DC, and COCO-C encompassing distribution shifts due to information loss, corruption, adversarial attacks, and geometrical deformation, highlighting the challenges of the model's robustness to foster the research for achieving robustness. Source code shall be made available to the research community on GitHub.

摘要: 分布外(OOD)稳健性的挑战仍然是部署深度视觉模型的关键障碍。视觉语言模型(VLM)最近取得了突破性的成果。基于VLM的开放词汇表目标检测扩展了传统目标检测框架的能力，支持对超出预定义类别的目标进行识别和分类。研究开放词汇对象检测中的面向对象设计的健壮性对于提高这些模型的可信性是至关重要的。这项研究对最近三种开放词汇表(OV)基础目标检测模型的零射能力进行了全面的稳健性评估：OWL-VIT、YOLO World和接地Dino。在稳健性基准COCO-O、COCO-DC和COCO-C上进行的实验涵盖了由于信息丢失、损坏、对抗性攻击和几何变形而导致的分布偏移，突显了模型稳健性的挑战，以促进实现稳健性的研究。应在GitHub上向研究社区提供源代码。



## **6. Preventing Catastrophic Overfitting in Fast Adversarial Training: A Bi-level Optimization Perspective**

防止快速对抗训练中的灾难性过度匹配：双层优化的角度 cs.LG

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12443v1) [paper-pdf](http://arxiv.org/pdf/2407.12443v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Adversarial training (AT) has become an effective defense method against adversarial examples (AEs) and it is typically framed as a bi-level optimization problem. Among various AT methods, fast AT (FAT), which employs a single-step attack strategy to guide the training process, can achieve good robustness against adversarial attacks at a low cost. However, FAT methods suffer from the catastrophic overfitting problem, especially on complex tasks or with large-parameter models. In this work, we propose a FAT method termed FGSM-PCO, which mitigates catastrophic overfitting by averting the collapse of the inner optimization problem in the bi-level optimization process. FGSM-PCO generates current-stage AEs from the historical AEs and incorporates them into the training process using an adaptive mechanism. This mechanism determines an appropriate fusion ratio according to the performance of the AEs on the training model. Coupled with a loss function tailored to the training framework, FGSM-PCO can alleviate catastrophic overfitting and help the recovery of an overfitted model to effective training. We evaluate our algorithm across three models and three datasets to validate its effectiveness. Comparative empirical studies against other FAT algorithms demonstrate that our proposed method effectively addresses unresolved overfitting issues in existing algorithms.

摘要: 对抗性训练(AT)已经成为对抗对抗性范例(AEs)的一种有效的防御方法，它通常被描述为一个双层优化问题。在众多的AT方法中，FAST AT(FAT)采用单步攻击策略来指导训练过程，能够以较低的代价获得对对手攻击的良好健壮性。然而，FAT方法存在灾难性的过拟合问题，特别是在处理复杂任务或大参数模型时。在这项工作中，我们提出了一种称为FGSM-PCO的FAT方法，它通过避免双层优化过程中内部优化问题的崩溃来减轻灾难性的过拟合。FGSM-PCO从历史的AE生成当前阶段的AE，并使用自适应机制将它们合并到训练过程中。该机制根据AEs在训练模型上的表现来确定合适的融合比例。再加上为训练框架量身定做的损失函数，FGSM-PCO可以缓解灾难性的过拟合，并帮助过度拟合的模型恢复到有效的训练。我们在三个模型和三个数据集上对我们的算法进行了评估，以验证其有效性。与其他FAT算法的对比实验表明，我们提出的方法有效地解决了现有算法中尚未解决的过拟合问题。



## **7. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks**

发送者：针对补丁攻击的基于扩散的对抗防御 cs.CV

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2306.09124v4) [paper-pdf](http://arxiv.org/pdf/2306.09124v4)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Yubo Chen, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications. This paper introduces DIFFender, a novel defense framework that harnesses the capabilities of a text-guided diffusion model to combat patch attacks. Central to our approach is the discovery of the Adversarial Anomaly Perception (AAP) phenomenon, which empowers the diffusion model to detect and localize adversarial patches through the analysis of distributional discrepancies. DIFFender integrates dual tasks of patch localization and restoration within a single diffusion model framework, utilizing their close interaction to enhance defense efficacy. Moreover, DIFFender utilizes vision-language pre-training coupled with an efficient few-shot prompt-tuning algorithm, which streamlines the adaptation of the pre-trained diffusion model to defense tasks, thus eliminating the need for extensive retraining. Our comprehensive evaluation spans image classification and face recognition tasks, extending to real-world scenarios, where DIFFender shows good robustness against adversarial attacks. The versatility and generalizability of DIFFender are evident across a variety of settings, classifiers, and attack methodologies, marking an advancement in adversarial patch defense strategies.

摘要: 对抗性攻击，尤其是补丁攻击，对深度学习模型的健壮性和可靠性构成了严重的威胁。开发针对补丁攻击的可靠防御对于现实世界的应用程序至关重要。本文介绍了一种新的防御框架DIFFender，它利用文本引导扩散模型的能力来对抗补丁攻击。我们方法的核心是发现了对抗性异常感知(AAP)现象，这使得扩散模型能够通过分析分布差异来检测和定位对抗性补丁。DIFFender在一个扩散模型框架内集成了补丁定位和恢复的双重任务，利用它们的密切交互来提高防御效率。此外，DIFFender利用视觉语言预训练和高效的少镜头提示调整算法，简化了预先训练的扩散模型对防御任务的适应，从而消除了对广泛再训练的需要。我们的综合评估涵盖图像分类和人脸识别任务，并扩展到真实场景，在这些场景中，DIFFender显示出对对手攻击的良好稳健性。DIFFender的多功能性和通用性在各种设置、分类器和攻击方法中都很明显，标志着对抗性补丁防御策略的进步。



## **8. Bribe & Fork: Cheap Bribing Attacks via Forking Threat**

贿赂与叉子：通过叉子威胁进行廉价贿赂攻击 cs.CR

This is a full version of the paper Bribe & Fork: Cheap Bribing  Attacks via Forking Threat which was accepted to AFT'24

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2402.01363v2) [paper-pdf](http://arxiv.org/pdf/2402.01363v2)

**Authors**: Zeta Avarikioti, Paweł Kędzior, Tomasz Lizurej, Tomasz Michalak

**Abstract**: In this work, we reexamine the vulnerability of Payment Channel Networks (PCNs) to bribing attacks, where an adversary incentivizes blockchain miners to deliberately ignore a specific transaction to undermine the punishment mechanism of PCNs. While previous studies have posited a prohibitive cost for such attacks, we show that this cost may be dramatically reduced (to approximately \$125), thereby increasing the likelihood of these attacks. To this end, we introduce Bribe & Fork, a modified bribing attack that leverages the threat of a so-called feather fork which we analyze with a novel formal model for the mining game with forking. We empirically analyze historical data of some real-world blockchain implementations to evaluate the scale of this cost reduction. Our findings shed more light on the potential vulnerability of PCNs and highlight the need for robust solutions.

摘要: 在这项工作中，我们重新审视了支付渠道网络（PCE）对贿赂攻击的脆弱性，即对手激励区块链矿工故意忽略特定交易，以破坏PCE的惩罚机制。虽然之前的研究假设此类攻击的成本过高，但我们表明，这一成本可能会大幅降低（约为125日元），从而增加这些攻击的可能性。为此，我们引入了贿赂和叉子，这是一种改进的贿赂攻击，利用了所谓的羽毛叉的威胁，我们使用带有分叉的采矿游戏的新颖形式模型来分析该攻击。我们通过经验分析一些现实世界区块链实施的历史数据，以评估成本降低的规模。我们的研究结果进一步揭示了多学科网络的潜在脆弱性，并强调了对强大解决方案的需求。



## **9. Revisiting the Adversarial Robustness of Vision Language Models: a Multimodal Perspective**

重新审视视觉语言模型的对抗鲁棒性：多模式视角 cs.CV

16 pages, 14 figures

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2404.19287v2) [paper-pdf](http://arxiv.org/pdf/2404.19287v2)

**Authors**: Wanqi Zhou, Shuanghao Bai, Qibin Zhao, Badong Chen

**Abstract**: Pretrained vision-language models (VLMs) like CLIP have shown impressive generalization performance across various downstream tasks, yet they remain vulnerable to adversarial attacks. While prior research has primarily concentrated on improving the adversarial robustness of image encoders to guard against attacks on images, the exploration of text-based and multimodal attacks has largely been overlooked. In this work, we initiate the first known and comprehensive effort to study adapting vision-language models for adversarial robustness under the multimodal attack. Firstly, we introduce a multimodal attack strategy and investigate the impact of different attacks. We then propose a multimodal contrastive adversarial training loss, aligning the clean and adversarial text embeddings with the adversarial and clean visual features, to enhance the adversarial robustness of both image and text encoders of CLIP. Extensive experiments on 15 datasets across two tasks demonstrate that our method significantly improves the adversarial robustness of CLIP. Interestingly, we find that the model fine-tuned against multimodal adversarial attacks exhibits greater robustness than its counterpart fine-tuned solely against image-based attacks, even in the context of image attacks, which may open up new possibilities for enhancing the security of VLMs.

摘要: 像CLIP这样的预先训练的视觉语言模型(VLM)在各种下游任务中表现出令人印象深刻的泛化性能，但它们仍然容易受到对手的攻击。虽然以前的研究主要集中在提高图像编码器的对抗健壮性以防止对图像的攻击，但对基于文本的和多模式攻击的探索在很大程度上被忽视了。在这项工作中，我们启动了第一个已知和全面的努力，以研究适应视觉语言模型的对手在多模式攻击下的稳健性。首先，我们介绍了一种多模式攻击策略，并研究了不同攻击的影响。然后，我们提出了一种多模式对抗性训练损失，将干净和对抗性的文本嵌入与对抗性和干净的视觉特征相结合，以增强CLIP图像和文本编码者的对抗性健壮性。在两个任务的15个数据集上的大量实验表明，我们的方法显著地提高了CLIP的对抗健壮性。有趣的是，我们发现，与仅针对基于图像的攻击进行微调的模型相比，针对多模式攻击进行微调的模型表现出更强的稳健性，甚至在图像攻击的背景下也是如此，这可能为增强VLM的安全性开辟新的可能性。



## **10. Augmented Neural Fine-Tuning for Efficient Backdoor Purification**

增强神经微调以实现高效后门净化 cs.CV

Accepted to ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.10052v2) [paper-pdf](http://arxiv.org/pdf/2407.10052v2)

**Authors**: Nazmul Karim, Abdullah Al Arafat, Umar Khalid, Zhishan Guo, Nazanin Rahnavard

**Abstract**: Recent studies have revealed the vulnerability of deep neural networks (DNNs) to various backdoor attacks, where the behavior of DNNs can be compromised by utilizing certain types of triggers or poisoning mechanisms. State-of-the-art (SOTA) defenses employ too-sophisticated mechanisms that require either a computationally expensive adversarial search module for reverse-engineering the trigger distribution or an over-sensitive hyper-parameter selection module. Moreover, they offer sub-par performance in challenging scenarios, e.g., limited validation data and strong attacks. In this paper, we propose Neural mask Fine-Tuning (NFT) with an aim to optimally re-organize the neuron activities in a way that the effect of the backdoor is removed. Utilizing a simple data augmentation like MixUp, NFT relaxes the trigger synthesis process and eliminates the requirement of the adversarial search module. Our study further reveals that direct weight fine-tuning under limited validation data results in poor post-purification clean test accuracy, primarily due to overfitting issue. To overcome this, we propose to fine-tune neural masks instead of model weights. In addition, a mask regularizer has been devised to further mitigate the model drift during the purification process. The distinct characteristics of NFT render it highly efficient in both runtime and sample usage, as it can remove the backdoor even when a single sample is available from each class. We validate the effectiveness of NFT through extensive experiments covering the tasks of image classification, object detection, video action recognition, 3D point cloud, and natural language processing. We evaluate our method against 14 different attacks (LIRA, WaNet, etc.) on 11 benchmark data sets such as ImageNet, UCF101, Pascal VOC, ModelNet, OpenSubtitles2012, etc.

摘要: 最近的研究揭示了深度神经网络(DNN)对各种后门攻击的脆弱性，其中DNN的行为可以通过利用某些类型的触发或中毒机制来危害。最先进的(SOTA)防御使用了过于复杂的机制，需要计算昂贵的对抗性搜索模块来对触发分布进行反向工程，或者需要过于敏感的超参数选择模块。此外，它们在具有挑战性的场景中提供了低于平均水平的性能，例如有限的验证数据和强大的攻击。在本文中，我们提出了神经掩码微调(NFT)，目的是以一种消除后门影响的方式来优化重组神经元的活动。利用简单的数据增强，如混合，NFT放松了触发器合成过程，并消除了对敌方搜索模块的要求。我们的研究进一步表明，在有限的验证数据下直接权重微调会导致净化后清洁测试的准确性较差，这主要是由于过度拟合问题。为了克服这一点，我们建议微调神经掩模而不是模型权重。此外，还设计了一种掩膜正则化算法，以进一步缓解纯化过程中的模型漂移。NFT的独特特性使得它在运行时和样本使用方面都非常高效，因为即使每个类只有一个样本可用，它也可以删除后门。通过在图像分类、目标检测、视频动作识别、三维点云和自然语言处理等方面的大量实验，验证了NFT的有效性。我们针对14种不同的攻击(Lira、WaNet等)对我们的方法进行了评估。基于ImageNet、UCF101、Pascal VOC、ModelNet、OpenSubtitles2012等11个基准数据集。



## **11. Asymmetric Bias in Text-to-Image Generation with Adversarial Attacks**

具有对抗性攻击的文本到图像生成中的不对称偏差 cs.LG

camera-ready version

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2312.14440v3) [paper-pdf](http://arxiv.org/pdf/2312.14440v3)

**Authors**: Haz Sameen Shahgir, Xianghao Kong, Greg Ver Steeg, Yue Dong

**Abstract**: The widespread use of Text-to-Image (T2I) models in content generation requires careful examination of their safety, including their robustness to adversarial attacks. Despite extensive research on adversarial attacks, the reasons for their effectiveness remain underexplored. This paper presents an empirical study on adversarial attacks against T2I models, focusing on analyzing factors associated with attack success rates (ASR). We introduce a new attack objective - entity swapping using adversarial suffixes and two gradient-based attack algorithms. Human and automatic evaluations reveal the asymmetric nature of ASRs on entity swap: for example, it is easier to replace "human" with "robot" in the prompt "a human dancing in the rain." with an adversarial suffix, but the reverse replacement is significantly harder. We further propose probing metrics to establish indicative signals from the model's beliefs to the adversarial ASR. We identify conditions that result in a success probability of 60% for adversarial attacks and others where this likelihood drops below 5%.

摘要: 文本到图像(T2I)模型在内容生成中的广泛使用需要仔细检查它们的安全性，包括它们对对手攻击的健壮性。尽管对对抗性攻击进行了广泛的研究，但其有效性的原因仍未得到充分探讨。本文对T2I模型的对抗性攻击进行了实证研究，重点分析了影响攻击成功率的因素。提出了一种新的攻击目标实体交换算法，利用对抗性后缀和两种基于梯度的攻击算法。人工评估和自动评估揭示了ASR在实体交换上的不对称性质：例如，在提示符“a Human in the雨中跳舞”中，更容易将“Human”替换为“bot”。使用对抗性后缀，但反向替换要困难得多。我们进一步提出了探测度量来建立从模型信念到对抗性ASR的指示性信号。我们确定了对抗性攻击成功概率为60%的条件，以及其他可能性降至5%以下的条件。



## **12. Any Target Can be Offense: Adversarial Example Generation via Generalized Latent Infection**

任何目标都可能是攻击性的：通过普遍潜伏感染生成对抗性示例 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-17    [abs](http://arxiv.org/abs/2407.12292v1) [paper-pdf](http://arxiv.org/pdf/2407.12292v1)

**Authors**: Youheng Sun, Shengming Yuan, Xuanhan Wang, Lianli Gao, Jingkuan Song

**Abstract**: Targeted adversarial attack, which aims to mislead a model to recognize any image as a target object by imperceptible perturbations, has become a mainstream tool for vulnerability assessment of deep neural networks (DNNs). Since existing targeted attackers only learn to attack known target classes, they cannot generalize well to unknown classes. To tackle this issue, we propose $\bf{G}$eneralized $\bf{A}$dversarial attac$\bf{KER}$ ($\bf{GAKer}$), which is able to construct adversarial examples to any target class. The core idea behind GAKer is to craft a latently infected representation during adversarial example generation. To this end, the extracted latent representations of the target object are first injected into intermediate features of an input image in an adversarial generator. Then, the generator is optimized to ensure visual consistency with the input image while being close to the target object in the feature space. Since the GAKer is class-agnostic yet model-agnostic, it can be regarded as a general tool that not only reveals the vulnerability of more DNNs but also identifies deficiencies of DNNs in a wider range of classes. Extensive experiments have demonstrated the effectiveness of our proposed method in generating adversarial examples for both known and unknown classes. Notably, compared with other generative methods, our method achieves an approximately $14.13\%$ higher attack success rate for unknown classes and an approximately $4.23\%$ higher success rate for known classes. Our code is available in https://github.com/VL-Group/GAKer.

摘要: 目标对抗攻击旨在通过不可察觉的扰动来误导模型将任何图像识别为目标对象，已成为深度神经网络脆弱性评估的主流工具。由于现有的目标攻击者只学习攻击已知的目标类，因此他们不能很好地泛化到未知的类。为了解决这个问题，我们提出了推广的$\bf{G}$推广的$\bf{A}$dversarialattac$\bf{Ker}$($\bf{GAKer}$)，它能够构造对任何目标类的对抗性例子。GAKer背后的核心思想是在敌意示例生成期间创建一个潜伏感染的表示。为此，首先在对抗性生成器中将提取的目标对象的潜在表示注入到输入图像的中间特征中。然后，对生成器进行优化，以确保与输入图像的视觉一致性，同时在特征空间中接近目标对象。由于GAKer是类不可知的，也是模型不可知的，它可以被视为一个通用工具，不仅可以揭示更多DNN的脆弱性，还可以在更大范围的类中识别DNN的缺陷。大量的实验表明，该方法在生成已知和未知类别的对抗性样本方面是有效的。值得注意的是，与其他产生式方法相比，该方法对未知类的攻击成功率约高14.13美元，对已知类的攻击成功率约高4.23美元。我们的代码在https://github.com/VL-Group/GAKer.中可用



## **13. Does Refusal Training in LLMs Generalize to the Past Tense?**

LLM中的拒绝培训是否适用于过去时态？ cs.CL

Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11969v1) [paper-pdf](http://arxiv.org/pdf/2407.11969v1)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.

摘要: 拒绝训练被广泛用于防止LLMS产生有害、不受欢迎或非法的输出。我们揭示了当前拒绝训练方法中一个奇怪的概括缺口：简单地用过去时重新表达一个有害的请求(例如，“如何调制燃烧鸡尾酒？”“人们是如何调制燃烧鸡尾酒的？”)通常足以越狱许多最先进的LLM。我们以GPT-3.5 Turbo为改型模型，对Llama-3 8B、GPT-3.5 Turbo、Gema-2 9B、Phi-3-Mini、GPT-40和R2D2模型进行了系统的评估。例如，对GPT-4o的这种简单攻击的成功率从使用直接请求的1%增加到使用20次过去时态重组尝试的88%，这些尝试使用GPT-4作为越狱法官的JailBreakB边的有害请求。有趣的是，我们还发现，未来时的重述没有那么有效，这表明拒绝障碍倾向于考虑过去的历史问题，而不是假设的未来问题。此外，我们在微调GPT-3.5Turbo上的实验表明，当微调数据中明确包含过去时态示例时，防御过去的重新公式是可行的。总体而言，我们的发现强调了广泛使用的对齐技术--如SFT、RLHF和对抗性训练--用于对所研究的模型进行对齐可能是脆弱的，并且并不总是像预期的那样泛化。我们在https://github.com/tml-epfl/llm-past-tense.上提供代码和越狱文物



## **14. JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models**

越狱长凳：越狱大型语言模型的开放鲁棒性基准 cs.CR

JailbreakBench v1.0: more attack artifacts, more test-time defenses,  a more accurate jailbreak judge (Llama-3-70B with a custom prompt), a larger  dataset of human preferences for selecting a jailbreak judge (300 examples),  an over-refusal evaluation dataset (100 benign/borderline behaviors), a  semantic refusal judge based on Llama-3-8B

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2404.01318v4) [paper-pdf](http://arxiv.org/pdf/2404.01318v4)

**Authors**: Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, Eric Wong

**Abstract**: Jailbreak attacks cause large language models (LLMs) to generate harmful, unethical, or otherwise objectionable content. Evaluating these attacks presents a number of challenges, which the current collection of benchmarks and evaluation techniques do not adequately address. First, there is no clear standard of practice regarding jailbreaking evaluation. Second, existing works compute costs and success rates in incomparable ways. And third, numerous works are not reproducible, as they withhold adversarial prompts, involve closed-source code, or rely on evolving proprietary APIs. To address these challenges, we introduce JailbreakBench, an open-sourced benchmark with the following components: (1) an evolving repository of state-of-the-art adversarial prompts, which we refer to as jailbreak artifacts; (2) a jailbreaking dataset comprising 100 behaviors -- both original and sourced from prior work (Zou et al., 2023; Mazeika et al., 2023, 2024) -- which align with OpenAI's usage policies; (3) a standardized evaluation framework at https://github.com/JailbreakBench/jailbreakbench that includes a clearly defined threat model, system prompts, chat templates, and scoring functions; and (4) a leaderboard at https://jailbreakbench.github.io/ that tracks the performance of attacks and defenses for various LLMs. We have carefully considered the potential ethical implications of releasing this benchmark, and believe that it will be a net positive for the community.

摘要: 越狱攻击会导致大型语言模型(LLM)生成有害、不道德或令人反感的内容。评估这些攻击带来了许多挑战，目前收集的基准和评估技术没有充分解决这些挑战。首先，关于越狱评估没有明确的实践标准。其次，现有的工作以无与伦比的方式计算成本和成功率。第三，许多作品是不可复制的，因为它们保留了对抗性提示，涉及封闭源代码，或者依赖于不断发展的专有API。为了应对这些挑战，我们引入了JailBreak，这是一个开源基准，具有以下组件：(1)一个不断发展的最先进对手提示的储存库，我们称之为越狱人工制品；(2)一个包括100种行为的越狱数据集--既有原始的，也有源自先前工作的(邹某等人，2023年；Mazeika等人，2023年，2024年)--与开放人工智能的使用政策保持一致；(3)https://github.com/JailbreakBench/jailbreakbench的标准化评估框架，包括明确定义的威胁模型、系统提示、聊天模板和评分功能；以及(4)https://jailbreakbench.github.io/的排行榜，跟踪各种LLM的攻击和防御性能。我们已仔细考虑发布这一基准的潜在道德影响，并相信它将为社会带来净积极的影响。



## **15. Towards Reliable Evaluation and Fast Training of Robust Semantic Segmentation Models**

迈向稳健的语义分割模型的可靠评估和快速训练 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.12941v2) [paper-pdf](http://arxiv.org/pdf/2306.12941v2)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: Adversarial robustness has been studied extensively in image classification, especially for the $\ell_\infty$-threat model, but significantly less so for related tasks such as object detection and semantic segmentation, where attacks turn out to be a much harder optimization problem than for image classification. We propose several problem-specific novel attacks minimizing different metrics in accuracy and mIoU. The ensemble of our attacks, SEA, shows that existing attacks severely overestimate the robustness of semantic segmentation models. Surprisingly, existing attempts of adversarial training for semantic segmentation models turn out to be weak or even completely non-robust. We investigate why previous adaptations of adversarial training to semantic segmentation failed and show how recently proposed robust ImageNet backbones can be used to obtain adversarially robust semantic segmentation models with up to six times less training time for PASCAL-VOC and the more challenging ADE20k. The associated code and robust models are available at https://github.com/nmndeep/robust-segmentation

摘要: 对抗性稳健性在图像分类中已经得到了广泛的研究，特别是对于威胁模型，但对于目标检测和语义分割等相关任务的研究要少得多，因为在这些任务中，攻击被证明是一个比图像分类更困难的优化问题。我们提出了几种针对特定问题的新攻击，它们在准确率和Miou上最小化不同的度量。我们的SEA攻击集成表明，现有的攻击严重高估了语义分割模型的稳健性。令人惊讶的是，现有的针对语义分割模型的对抗性训练尝试被证明是弱的，甚至是完全不稳健的。我们调查了以前对抗性训练对语义分割的适应失败的原因，并展示了最近提出的健壮ImageNet主干如何用于获得对抗性健壮的语义分割模型，而Pascal-VOC和更具挑战性的ADE20k的训练时间最多减少了六分之一。相关代码和健壮模型可在https://github.com/nmndeep/robust-segmentation上获得



## **16. Variational Randomized Smoothing for Sample-Wise Adversarial Robustness**

样本对抗鲁棒性的变分随机平滑 cs.LG

20 pages, under preparation

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11844v1) [paper-pdf](http://arxiv.org/pdf/2407.11844v1)

**Authors**: Ryo Hase, Ye Wang, Toshiaki Koike-Akino, Jing Liu, Kieran Parsons

**Abstract**: Randomized smoothing is a defensive technique to achieve enhanced robustness against adversarial examples which are small input perturbations that degrade the performance of neural network models. Conventional randomized smoothing adds random noise with a fixed noise level for every input sample to smooth out adversarial perturbations. This paper proposes a new variational framework that uses a per-sample noise level suitable for each input by introducing a noise level selector. Our experimental results demonstrate enhancement of empirical robustness against adversarial attacks. We also provide and analyze the certified robustness for our sample-wise smoothing method.

摘要: 随机平滑是一种防御性技术，旨在针对对抗性示例实现增强的鲁棒性，这些示例是会降低神经网络模型性能的小输入扰动。传统的随机平滑会为每个输入样本添加具有固定噪音水平的随机噪音，以消除对抗性扰动。本文提出了一种新的变分框架，该框架通过引入噪音水平选择器来使用适合每个输入的每样本噪音水平。我们的实验结果表明，针对对抗性攻击的经验鲁棒性增强。我们还提供并分析了我们的样本平滑方法的认证稳健性。



## **17. Exploring the Robustness of Decision-Level Through Adversarial Attacks on LLM-Based Embodied Models**

通过对基于LLM的排队模型的对抗攻击探索决策级的鲁棒性 cs.MM

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2405.19802v3) [paper-pdf](http://arxiv.org/pdf/2405.19802v3)

**Authors**: Shuyuan Liu, Jiawei Chen, Shouwei Ruan, Hang Su, Zhaoxia Yin

**Abstract**: Embodied intelligence empowers agents with a profound sense of perception, enabling them to respond in a manner closely aligned with real-world situations. Large Language Models (LLMs) delve into language instructions with depth, serving a crucial role in generating plans for intricate tasks. Thus, LLM-based embodied models further enhance the agent's capacity to comprehend and process information. However, this amalgamation also ushers in new challenges in the pursuit of heightened intelligence. Specifically, attackers can manipulate LLMs to produce irrelevant or even malicious outputs by altering their prompts. Confronted with this challenge, we observe a notable absence of multi-modal datasets essential for comprehensively evaluating the robustness of LLM-based embodied models. Consequently, we construct the Embodied Intelligent Robot Attack Dataset (EIRAD), tailored specifically for robustness evaluation. Additionally, two attack strategies are devised, including untargeted attacks and targeted attacks, to effectively simulate a range of diverse attack scenarios. At the same time, during the attack process, to more accurately ascertain whether our method is successful in attacking the LLM-based embodied model, we devise a new attack success evaluation method utilizing the BLIP2 model. Recognizing the time and cost-intensive nature of the GCG algorithm in attacks, we devise a scheme for prompt suffix initialization based on various target tasks, thus expediting the convergence process. Experimental results demonstrate that our method exhibits a superior attack success rate when targeting LLM-based embodied models, indicating a lower level of decision-level robustness in these models.

摘要: 具身智能使特工具有深刻的感知力，使他们能够以与现实世界情况密切一致的方式做出反应。大型语言模型(LLM)深入研究语言指令，在为复杂任务制定计划方面发挥着至关重要的作用。因此，基于LLM的具体化模型进一步增强了代理理解和处理信息的能力。然而，这种融合也带来了追求高智商的新挑战。具体地说，攻击者可以通过更改提示来操纵LLMS生成无关甚至恶意的输出。面对这一挑战，我们注意到明显缺乏全面评估基于LLM的体现模型的稳健性所必需的多模式数据集。因此，我们构建了专门为健壮性评估量身定做的具体化智能机器人攻击数据集(Eirad)。此外，设计了两种攻击策略，包括非定向攻击和定向攻击，以有效地模拟一系列不同的攻击场景。同时，在攻击过程中，为了更准确地确定我们的方法在攻击基于LLM的体现模型上是否成功，我们设计了一种新的利用BLIP2模型的攻击成功评估方法。考虑到GCG算法在攻击中的时间和成本密集性，我们设计了一种基于不同目标任务的快速后缀初始化方案，从而加快了收敛过程。实验结果表明，我们的方法在攻击基于LLM的具体模型时表现出了较高的攻击成功率，表明这些模型具有较低的决策级健壮性。



## **18. Relaxing Graph Transformers for Adversarial Attacks**

对抗攻击的放松图变形器 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11764v1) [paper-pdf](http://arxiv.org/pdf/2407.11764v1)

**Authors**: Philipp Foth, Lukas Gosch, Simon Geisler, Leo Schwinn, Stephan Günnemann

**Abstract**: Existing studies have shown that Graph Neural Networks (GNNs) are vulnerable to adversarial attacks. Even though Graph Transformers (GTs) surpassed Message-Passing GNNs on several benchmarks, their adversarial robustness properties are unexplored. However, attacking GTs is challenging due to their Positional Encodings (PEs) and special attention mechanisms which can be difficult to differentiate. We overcome these challenges by targeting three representative architectures based on (1) random-walk PEs, (2) pair-wise-shortest-path PEs, and (3) spectral PEs - and propose the first adaptive attacks for GTs. We leverage our attacks to evaluate robustness to (a) structure perturbations on node classification; and (b) node injection attacks for (fake-news) graph classification. Our evaluation reveals that they can be catastrophically fragile and underlines our work's importance and the necessity for adaptive attacks.

摘要: 现有的研究表明，图形神经网络（GNN）容易受到对抗攻击。尽管图形变形器（GT）在多个基准测试中超过了消息传递GNN，但其对抗鲁棒性属性尚未被探索。然而，攻击GT具有挑战性，因为它们的位置编码（PE）和特殊注意机制很难区分。我们通过针对基于（1）随机游走PE、（2）成对最短路径PE和（3）频谱PE的三种代表性架构来克服这些挑战，并提出了针对GT的第一次自适应攻击。我们利用我们的攻击来评估对（a）节点分类的结构扰动的稳健性;和（b）对（假新闻）图分类的节点注入攻击。我们的评估表明，它们可能是灾难性的脆弱性，并强调了我们工作的重要性和自适应攻击的必要性。



## **19. Enhancing TinyML Security: Study of Adversarial Attack Transferability**

增强TinyML安全性：对抗性攻击可转移性研究 cs.CR

Accepted and presented at tinyML Foundation EMEA Innovation Forum  2024

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11599v1) [paper-pdf](http://arxiv.org/pdf/2407.11599v1)

**Authors**: Parin Shah, Yuvaraj Govindarajulu, Pavan Kulkarni, Manojkumar Parmar

**Abstract**: The recent strides in artificial intelligence (AI) and machine learning (ML) have propelled the rise of TinyML, a paradigm enabling AI computations at the edge without dependence on cloud connections. While TinyML offers real-time data analysis and swift responses critical for diverse applications, its devices' intrinsic resource limitations expose them to security risks. This research delves into the adversarial vulnerabilities of AI models on resource-constrained embedded hardware, with a focus on Model Extraction and Evasion Attacks. Our findings reveal that adversarial attacks from powerful host machines could be transferred to smaller, less secure devices like ESP32 and Raspberry Pi. This illustrates that adversarial attacks could be extended to tiny devices, underscoring vulnerabilities, and emphasizing the necessity for reinforced security measures in TinyML deployments. This exploration enhances the comprehension of security challenges in TinyML and offers insights for safeguarding sensitive data and ensuring device dependability in AI-powered edge computing settings.

摘要: 人工智能(AI)和机器学习(ML)最近的进步推动了TinyML的崛起，TinyML是一种能够在边缘进行AI计算的范式，而不依赖于云连接。虽然TinyML提供对各种应用至关重要的实时数据分析和快速响应，但其设备固有的资源限制使它们面临安全风险。该研究深入研究了资源受限嵌入式硬件上人工智能模型的对抗性漏洞，重点研究了模型提取和规避攻击。我们的发现表明，来自强大主机的对抗性攻击可能会转移到较小、安全性较低的设备上，如ESP32和Raspberry PI。这表明敌意攻击可以扩展到微型设备，这突显了漏洞，并强调了在TinyML部署中加强安全措施的必要性。这一探索增强了对TinyML安全挑战的理解，并为在人工智能支持的边缘计算环境中保护敏感数据和确保设备可靠性提供了见解。



## **20. AEMIM: Adversarial Examples Meet Masked Image Modeling**

AEIM：对抗性示例与掩蔽图像建模相结合 cs.CV

Under review of International Journal of Computer Vision (IJCV)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11537v1) [paper-pdf](http://arxiv.org/pdf/2407.11537v1)

**Authors**: Wenzhao Xiang, Chang Liu, Hang Su, Hongyang Yu

**Abstract**: Masked image modeling (MIM) has gained significant traction for its remarkable prowess in representation learning. As an alternative to the traditional approach, the reconstruction from corrupted images has recently emerged as a promising pretext task. However, the regular corrupted images are generated using generic generators, often lacking relevance to the specific reconstruction task involved in pre-training. Hence, reconstruction from regular corrupted images cannot ensure the difficulty of the pretext task, potentially leading to a performance decline. Moreover, generating corrupted images might introduce an extra generator, resulting in a notable computational burden. To address these issues, we propose to incorporate adversarial examples into masked image modeling, as the new reconstruction targets. Adversarial examples, generated online using only the trained models, can directly aim to disrupt tasks associated with pre-training. Therefore, the incorporation not only elevates the level of challenge in reconstruction but also enhances efficiency, contributing to the acquisition of superior representations by the model. In particular, we introduce a novel auxiliary pretext task that reconstructs the adversarial examples corresponding to the original images. We also devise an innovative adversarial attack to craft more suitable adversarial examples for MIM pre-training. It is noted that our method is not restricted to specific model architectures and MIM strategies, rendering it an adaptable plug-in capable of enhancing all MIM methods. Experimental findings substantiate the remarkable capability of our approach in amplifying the generalization and robustness of existing MIM methods. Notably, our method surpasses the performance of baselines on various tasks, including ImageNet, its variants, and other downstream tasks.

摘要: 蒙面图像建模(MIM)因其在表征学习方面的卓越能力而获得了巨大的吸引力。作为传统方法的替代方法，从损坏的图像中重建图像最近已经成为一项很有前途的借口任务。然而，常规的损坏图像是使用通用生成器生成的，通常与预训练中涉及的特定重建任务缺乏相关性。因此，从常规损坏的图像重建不能确保借口任务的难度，这可能会导致性能下降。此外，生成损坏的图像可能会引入额外的生成器，从而导致显著的计算负担。为了解决这些问题，我们建议将对抗性例子引入到掩蔽图像建模中，作为新的重建目标。仅使用训练过的模型在线生成的对抗性例子可以直接旨在扰乱与预训练相关的任务。因此，合并不仅提高了重建中的挑战水平，而且提高了效率，有助于通过模型获得更好的表示。特别是，我们引入了一种新的辅助借口任务，该任务重建与原始图像相对应的对抗性示例。我们还设计了一种创新的对抗性攻击，为MIM预训练制作更合适的对抗性范例。值得注意的是，我们的方法并不局限于特定的模型体系结构和MIM策略，使其成为一个能够增强所有MIM方法的适应性插件。实验结果证明了该方法在增强现有MIM方法的泛化和健壮性方面具有显著的能力。值得注意的是，我们的方法超过了各种任务的基线性能，包括ImageNet、其变体和其他下游任务。



## **21. Learning on Graphs with Large Language Models(LLMs): A Deep Dive into Model Robustness**

使用大型语言模型（LLM）学习图形：深入研究模型稳健性 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.12068v1) [paper-pdf](http://arxiv.org/pdf/2407.12068v1)

**Authors**: Kai Guo, Zewen Liu, Zhikai Chen, Hongzhi Wen, Wei Jin, Jiliang Tang, Yi Chang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various natural language processing tasks. Recently, several LLMs-based pipelines have been developed to enhance learning on graphs with text attributes, showcasing promising performance. However, graphs are well-known to be susceptible to adversarial attacks and it remains unclear whether LLMs exhibit robustness in learning on graphs. To address this gap, our work aims to explore the potential of LLMs in the context of adversarial attacks on graphs. Specifically, we investigate the robustness against graph structural and textual perturbations in terms of two dimensions: LLMs-as-Enhancers and LLMs-as-Predictors. Through extensive experiments, we find that, compared to shallow models, both LLMs-as-Enhancers and LLMs-as-Predictors offer superior robustness against structural and textual attacks.Based on these findings, we carried out additional analyses to investigate the underlying causes. Furthermore, we have made our benchmark library openly available to facilitate quick and fair evaluations, and to encourage ongoing innovative research in this field.

摘要: 大型语言模型(LLM)在各种自然语言处理任务中表现出了显著的性能。最近，已经开发了几个基于LLMS的管道来增强对具有文本属性的图形的学习，展示了良好的性能。然而，众所周知，图是容易受到敌意攻击的，而且目前还不清楚LLM是否表现出关于图的学习的健壮性。为了解决这一差距，我们的工作旨在探索LLMS在对抗性攻击图的背景下的潜力。具体地说，我们从两个维度考察了对图结构和文本扰动的稳健性：作为增强器的LLMS和作为预测者的LLMS。通过广泛的实验，我们发现，与浅层模型相比，LLMS-as-Enhancer和LLMS-as-Predicator对结构和文本攻击都具有更好的稳健性。基于这些发现，我们进行了额外的分析，以探讨潜在的原因。此外，我们开放了我们的基准图书馆，以促进快速和公平的评估，并鼓励这一领域正在进行的创新研究。



## **22. Boosting the Transferability of Adversarial Attacks with Global Momentum Initialization**

利用全球动量指标提高对抗性攻击的可转移性 cs.CV

Accepted by Expert Systems with Applications (ESWA)

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2211.11236v3) [paper-pdf](http://arxiv.org/pdf/2211.11236v3)

**Authors**: Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Dingkang Yang, Lingyi Hong, Pinxue Guo, Haijing Guo, Wenqiang Zhang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial examples, which are crafted by adding human-imperceptible perturbations to the benign inputs. Simultaneously, adversarial examples exhibit transferability across models, enabling practical black-box attacks. However, existing methods are still incapable of achieving the desired transfer attack performance. In this work, focusing on gradient optimization and consistency, we analyse the gradient elimination phenomenon as well as the local momentum optimum dilemma. To tackle these challenges, we introduce Global Momentum Initialization (GI), providing global momentum knowledge to mitigate gradient elimination. Specifically, we perform gradient pre-convergence before the attack and a global search during this stage. GI seamlessly integrates with existing transfer methods, significantly improving the success rate of transfer attacks by an average of 6.4% under various advanced defense mechanisms compared to the state-of-the-art method. Ultimately, GI demonstrates strong transferability in both image and video attack domains. Particularly, when attacking advanced defense methods in the image domain, it achieves an average attack success rate of 95.4%. The code is available at $\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.

摘要: 深度神经网络(DNN)很容易受到敌意例子的影响，这些例子是通过在良性输入中添加人类无法察觉的扰动来构建的。同时，敌意例子展示了跨模型的可转移性，从而实现了实际的黑盒攻击。然而，现有的方法仍然不能达到期望的传输攻击性能。在这项工作中，我们以梯度优化和一致性为重点，分析了梯度消除现象以及局部动量最优困境。为了应对这些挑战，我们引入了全局动量初始化(GI)，提供了全局动量知识来缓解梯度消除。具体地说，我们在攻击前执行梯度预收敛，并在此阶段进行全局搜索。GI与现有的传输方式无缝集成，在各种先进的防御机制下，相比最先进的方法，传输攻击成功率平均提升了6.4%。最终，GI在图像和视频攻击领域都表现出了很强的可转移性。特别是在攻击图像域的先进防御手段时，平均攻击成功率达到95.4%。代码可在$\href{https://github.com/Omenzychen/Global-Momentum-Initialization}{https://github.com/Omenzychen/Global-Momentum-Initialization}$.上获得



## **23. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

调查表格数据对抗性攻击的不可感知性：实证分析 cs.LG

33 pages

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11463v1) [paper-pdf](http://arxiv.org/pdf/2407.11463v1)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models, as they can cause the model to make incorrect predictions by introducing imperceptible perturbations to the input data. While extensively studied in unstructured data like images, their application to structured data like tabular data presents unique challenges due to the heterogeneity and intricate feature interdependencies of tabular data. Imperceptibility in tabular data involves preserving data integrity while potentially causing misclassification, underscoring the need for tailored imperceptibility criteria for tabular data. However, there is currently a lack of standardised metrics for assessing adversarial attacks specifically targeted at tabular data. To address this gap, we derive a set of properties for evaluating the imperceptibility of adversarial attacks on tabular data. These properties are defined to capture seven perspectives of perturbed data: proximity to original inputs, sparsity of alterations, deviation to datapoints in the original dataset, sensitivity of altering sensitive features, immutability of perturbation, feasibility of perturbed values and intricate feature interdepencies among tabular features. Furthermore, we conduct both quantitative empirical evaluation and case-based qualitative examples analysis for seven properties. The evaluation reveals a trade-off between attack success and imperceptibility, particularly concerning proximity, sensitivity, and deviation. Although no evaluated attacks can achieve optimal effectiveness and imperceptibility simultaneously, unbounded attacks prove to be more promised for tabular data in crafting imperceptible adversarial examples. The study also highlights the limitation of evaluated algorithms in controlling sparsity effectively. We suggest incorporating a sparsity metric in future attack design to regulate the number of perturbed features.

摘要: 对抗性攻击是对机器学习模型的潜在威胁，因为它们可以通过向输入数据引入不可察觉的扰动来导致模型做出不正确的预测。虽然它们在图像等非结构化数据中得到了广泛的研究，但由于表格数据的异构性和复杂的特征相互依赖关系，它们在表格数据等结构化数据中的应用面临着独特的挑战。表格数据的不可察觉涉及在可能造成错误分类的同时保持数据的完整性，强调需要为表格数据制定专门的不可察觉标准。然而，目前缺乏评估专门针对表格数据的对抗性攻击的标准化指标。为了弥补这一差距，我们推导了一组用于评估对抗性攻击对表格数据的不可感知性的性质。这些属性被定义为捕捉扰动数据的七个角度：接近原始输入、改变的稀疏性、与原始数据集中的数据点的偏差、改变敏感特征的敏感度、扰动的不变性、扰动值的可行性以及表格特征之间复杂的特征相互依赖。此外，我们还对七个属性进行了定量的实证评估和基于案例的定性实例分析。评估揭示了攻击成功和不可察觉之间的权衡，特别是在接近、敏感度和偏差方面。虽然没有经过评估的攻击可以同时达到最优的有效性和不可见性，但无界攻击被证明在制作不可察觉的对抗性例子时更有希望获得表格数据。该研究还强调了被评估算法在有效控制稀疏性方面的局限性。我们建议在未来的攻击设计中加入稀疏性度量，以规范受干扰特征的数量。



## **24. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

Entrobust：评估大型语言模型在对抗性预测上的稳健性 cs.CL

Technical report; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2306.04528v5) [paper-pdf](http://arxiv.org/pdf/2306.04528v5)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Zhenqiang Gong, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptRobust, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. The adversarial prompts, crafted to mimic plausible user errors like typos or synonyms, aim to evaluate how slight deviations can affect LLM outcomes while maintaining semantic integrity. These prompts are then employed in diverse tasks including sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,788 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets. Our findings demonstrate that contemporary LLMs are not robust to adversarial prompts. Furthermore, we present a comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptRobust，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些对抗性提示旨在模仿打字或同义词等看似合理的用户错误，旨在评估微小的偏差如何在保持语义完整性的同时影响LLM结果。然后，这些提示被用于各种任务，包括情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4788个对抗性提示，仔细评估了8个任务和13个数据集。我们的研究结果表明，当代的LLM对敌意提示并不健壮。此外，我们给出了一个全面的分析，以理解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。



## **25. Gradients Look Alike: Sensitivity is Often Overestimated in DP-SGD**

学生看起来很相似：DP-Singapore中的敏感性经常被高估 cs.LG

published in 33rd USENIX Security Symposium

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2307.00310v3) [paper-pdf](http://arxiv.org/pdf/2307.00310v3)

**Authors**: Anvith Thudi, Hengrui Jia, Casey Meehan, Ilia Shumailov, Nicolas Papernot

**Abstract**: Differentially private stochastic gradient descent (DP-SGD) is the canonical approach to private deep learning. While the current privacy analysis of DP-SGD is known to be tight in some settings, several empirical results suggest that models trained on common benchmark datasets leak significantly less privacy for many datapoints. Yet, despite past attempts, a rigorous explanation for why this is the case has not been reached. Is it because there exist tighter privacy upper bounds when restricted to these dataset settings, or are our attacks not strong enough for certain datapoints? In this paper, we provide the first per-instance (i.e., ``data-dependent") DP analysis of DP-SGD. Our analysis captures the intuition that points with similar neighbors in the dataset enjoy better data-dependent privacy than outliers. Formally, this is done by modifying the per-step privacy analysis of DP-SGD to introduce a dependence on the distribution of model updates computed from a training dataset. We further develop a new composition theorem to effectively use this new per-step analysis to reason about an entire training run. Put all together, our evaluation shows that this novel DP-SGD analysis allows us to now formally show that DP-SGD leaks significantly less privacy for many datapoints (when trained on common benchmarks) than the current data-independent guarantee. This implies privacy attacks will necessarily fail against many datapoints if the adversary does not have sufficient control over the possible training datasets.

摘要: 差分私人随机梯度下降(DP-SGD)是私人深度学习的典型方法。虽然目前DP-SGD的隐私分析在某些情况下是严格的，但一些经验结果表明，在公共基准数据集上训练的模型对于许多数据点来说泄露的隐私要少得多。然而，尽管过去曾尝试过，但对于为什么会出现这种情况，还没有达成一个严格的解释。是因为限制到这些数据集设置时存在更严格的隐私上限，还是因为我们的攻击对某些数据点不够强大？在这篇文章中，我们提供了DP-SGD的第一个逐实例(即“数据依赖”)DP分析。我们的分析抓住了这样一种直觉，即数据集中具有相似邻居的点比离群值享有更好的数据依赖隐私。形式上，这是通过修改DP-SGD的每一步隐私分析来实现的，以引入对从训练数据集计算的模型更新的分布的依赖。我们进一步开发了一个新的合成定理，以有效地使用这个新的逐步分析来推理整个训练运行。综上所述，我们的评估表明，这种新颖的DP-SGD分析允许我们现在正式地表明，DP-SGD对于许多数据点(当根据公共基准进行训练时)的隐私泄露显著低于当前的数据独立保证。这意味着如果对手对可能的训练数据集没有足够的控制，针对许多数据点的隐私攻击必然会失败。



## **26. Learning to Break Deep Perceptual Hashing: The Use Case NeuralHash**

学习打破深度感知哈希：用例Neural哈希 cs.LG

Accepted by ACM FAccT 2022 as Oral

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2111.06628v5) [paper-pdf](http://arxiv.org/pdf/2111.06628v5)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Daniel Neider, Kristian Kersting

**Abstract**: Apple recently revealed its deep perceptual hashing system NeuralHash to detect child sexual abuse material (CSAM) on user devices before files are uploaded to its iCloud service. Public criticism quickly arose regarding the protection of user privacy and the system's reliability. In this paper, we present the first comprehensive empirical analysis of deep perceptual hashing based on NeuralHash. Specifically, we show that current deep perceptual hashing may not be robust. An adversary can manipulate the hash values by applying slight changes in images, either induced by gradient-based approaches or simply by performing standard image transformations, forcing or preventing hash collisions. Such attacks permit malicious actors easily to exploit the detection system: from hiding abusive material to framing innocent users, everything is possible. Moreover, using the hash values, inferences can still be made about the data stored on user devices. In our view, based on our results, deep perceptual hashing in its current form is generally not ready for robust client-side scanning and should not be used from a privacy perspective.

摘要: 苹果最近公布了其深度感知哈希系统NeuralHash，用于在文件上传到其iCloud服务之前检测用户设备上的儿童性虐待材料(CSAM)。公众很快就对保护用户隐私和系统的可靠性提出了批评。本文首次提出了基于NeuralHash的深度感知哈希算法的综合实证分析。具体地说，我们证明了当前的深度感知散列可能并不健壮。攻击者可以通过在图像中应用微小的更改来操纵散列值，这可以是由基于梯度的方法引起的，也可以只是通过执行标准图像转换来强制或防止散列冲突。这种攻击让恶意行为者很容易利用检测系统：从隐藏滥用材料到陷害无辜用户，一切皆有可能。此外，使用散列值，仍然可以对存储在用户设备上的数据进行推断。在我们看来，根据我们的结果，当前形式的深度感知散列通常还不能用于健壮的客户端扫描，不应该从隐私的角度使用。



## **27. Wicked Oddities: Selectively Poisoning for Effective Clean-Label Backdoor Attacks**

邪恶的怪事：选择性中毒以进行有效的无标签后门攻击 cs.LG

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.10825v2) [paper-pdf](http://arxiv.org/pdf/2407.10825v2)

**Authors**: Quang H. Nguyen, Nguyen Ngoc-Hieu, The-Anh Ta, Thanh Nguyen-Tang, Kok-Seng Wong, Hoang Thanh-Tung, Khoa D. Doan

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, a type of adversarial attack that poisons the training data to manipulate the behavior of models trained on such data. Clean-label attacks are a more stealthy form of backdoor attacks that can perform the attack without changing the labels of poisoned data. Early works on clean-label attacks added triggers to a random subset of the training set, ignoring the fact that samples contribute unequally to the attack's success. This results in high poisoning rates and low attack success rates. To alleviate the problem, several supervised learning-based sample selection strategies have been proposed. However, these methods assume access to the entire labeled training set and require training, which is expensive and may not always be practical. This work studies a new and more practical (but also more challenging) threat model where the attacker only provides data for the target class (e.g., in face recognition systems) and has no knowledge of the victim model or any other classes in the training set. We study different strategies for selectively poisoning a small set of training samples in the target class to boost the attack success rate in this setting. Our threat model poses a serious threat in training machine learning models with third-party datasets, since the attack can be performed effectively with limited information. Experiments on benchmark datasets illustrate the effectiveness of our strategies in improving clean-label backdoor attacks.

摘要: 深度神经网络很容易受到后门攻击，这是一种对抗性攻击，会毒化训练数据，以操纵根据这些数据训练的模型的行为。干净标签攻击是一种更隐蔽的后门攻击形式，可以在不更改有毒数据标签的情况下执行攻击。早期关于干净标签攻击的研究将触发器添加到训练集的随机子集，忽略了样本对攻击成功的贡献不平等的事实。这导致了高中毒率和低攻击成功率。为了缓解这一问题，人们提出了几种基于监督学习的样本选择策略。然而，这些方法假设可以访问整个标记的训练集，并且需要训练，这是昂贵的，并且可能并不总是实用的。这项工作研究了一种新的更实用(但也更具挑战性)的威胁模型，其中攻击者只提供目标类的数据(例如，在人脸识别系统中)，而不知道受害者模型或训练集中的任何其他类。我们研究了不同的策略来选择性地毒化目标类中的一小部分训练样本，以提高在这种情况下的攻击成功率。我们的威胁模型对使用第三方数据集训练机器学习模型构成了严重威胁，因为攻击可以在有限的信息下有效地执行。在基准数据集上的实验表明，我们的策略在改善干净标签后门攻击方面是有效的。



## **28. Feature Inference Attack on Shapley Values**

对Shapley值的特征推理攻击 cs.LG

This work was published in CCS 2022

**SubmitDate**: 2024-07-16    [abs](http://arxiv.org/abs/2407.11359v1) [paper-pdf](http://arxiv.org/pdf/2407.11359v1)

**Authors**: Xinjian Luo, Yangfan Jiang, Xiaokui Xiao

**Abstract**: As a solution concept in cooperative game theory, Shapley value is highly recognized in model interpretability studies and widely adopted by the leading Machine Learning as a Service (MLaaS) providers, such as Google, Microsoft, and IBM. However, as the Shapley value-based model interpretability methods have been thoroughly studied, few researchers consider the privacy risks incurred by Shapley values, despite that interpretability and privacy are two foundations of machine learning (ML) models.   In this paper, we investigate the privacy risks of Shapley value-based model interpretability methods using feature inference attacks: reconstructing the private model inputs based on their Shapley value explanations. Specifically, we present two adversaries. The first adversary can reconstruct the private inputs by training an attack model based on an auxiliary dataset and black-box access to the model interpretability services. The second adversary, even without any background knowledge, can successfully reconstruct most of the private features by exploiting the local linear correlations between the model inputs and outputs. We perform the proposed attacks on the leading MLaaS platforms, i.e., Google Cloud, Microsoft Azure, and IBM aix360. The experimental results demonstrate the vulnerability of the state-of-the-art Shapley value-based model interpretability methods used in the leading MLaaS platforms and highlight the significance and necessity of designing privacy-preserving model interpretability methods in future studies. To our best knowledge, this is also the first work that investigates the privacy risks of Shapley values.

摘要: Shapley值作为合作博弈理论中的一个解决方案概念，在模型可解释性研究中得到高度认可，并被谷歌、微软和IBM等领先的机器学习即服务(MLaaS)提供商广泛采用。然而，随着基于Shapley值的模型可解释性方法的深入研究，很少有人考虑Shapley值带来的隐私风险，尽管可解释性和隐私是机器学习(ML)模型的两个基础。在本文中，我们使用特征推理攻击来研究基于Shapley值的模型可解释性方法的隐私风险：基于其Shapley值的解释来重构私有模型输入。具体地说，我们介绍了两个对手。第一个对手可以通过基于辅助数据集和对模型可解释性服务的黑盒访问来训练攻击模型来重建私有输入。第二个对手，即使在没有任何背景知识的情况下，也可以通过利用模型输入和输出之间的局部线性相关性来成功地重建大多数私有特征。我们对领先的MLaaS平台，即Google Cloud、Microsoft Azure和IBM aix360执行拟议的攻击。实验结果证明了目前主流MLaaS平台所采用的基于Shapley值的模型可解释性方法的脆弱性，并强调了在未来的研究中设计隐私保护模型可解释性方法的重要性和必要性。据我们所知，这也是第一部调查Shapley Value隐私风险的作品。



## **29. Towards Adversarially Robust Vision-Language Models: Insights from Design Choices and Prompt Formatting Techniques**

迈向对抗稳健的视觉语言模型：来自设计选择和提示收件箱技术的见解 cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.11121v1) [paper-pdf](http://arxiv.org/pdf/2407.11121v1)

**Authors**: Rishika Bhagwatkar, Shravan Nayak, Reza Bayat, Alexis Roger, Daniel Z Kaplan, Pouya Bashivan, Irina Rish

**Abstract**: Vision-Language Models (VLMs) have witnessed a surge in both research and real-world applications. However, as they are becoming increasingly prevalent, ensuring their robustness against adversarial attacks is paramount. This work systematically investigates the impact of model design choices on the adversarial robustness of VLMs against image-based attacks. Additionally, we introduce novel, cost-effective approaches to enhance robustness through prompt formatting. By rephrasing questions and suggesting potential adversarial perturbations, we demonstrate substantial improvements in model robustness against strong image-based attacks such as Auto-PGD. Our findings provide important guidelines for developing more robust VLMs, particularly for deployment in safety-critical environments.

摘要: 视觉语言模型（VLM）见证了研究和现实世界应用的激增。然而，随着它们变得越来越普遍，确保它们对对抗攻击的稳健性至关重要。这项工作系统地研究了模型设计选择对VLM对抗基于图像的攻击的对抗鲁棒性的影响。此外，我们还引入了新颖的、具有成本效益的方法，通过即时格式化来增强稳健性。通过重新措辞问题并建议潜在的对抗性扰动，我们证明了模型针对Auto-PVD等强基于图像的攻击的稳健性有了实质性改进。我们的研究结果为开发更强大的VLM提供了重要的指导，特别是对于在安全关键环境中的部署。



## **30. PartImageNet++ Dataset: Scaling up Part-based Models for Robust Recognition**

PartImageNet++数据集：扩展基于零件的模型以实现稳健识别 cs.CV

Accepted by ECCV2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10918v1) [paper-pdf](http://arxiv.org/pdf/2407.10918v1)

**Authors**: Xiao Li, Yining Liu, Na Dong, Sitian Qin, Xiaolin Hu

**Abstract**: Deep learning-based object recognition systems can be easily fooled by various adversarial perturbations. One reason for the weak robustness may be that they do not have part-based inductive bias like the human recognition process. Motivated by this, several part-based recognition models have been proposed to improve the adversarial robustness of recognition. However, due to the lack of part annotations, the effectiveness of these methods is only validated on small-scale nonstandard datasets. In this work, we propose PIN++, short for PartImageNet++, a dataset providing high-quality part segmentation annotations for all categories of ImageNet-1K (IN-1K). With these annotations, we build part-based methods directly on the standard IN-1K dataset for robust recognition. Different from previous two-stage part-based models, we propose a Multi-scale Part-supervised Model (MPM), to learn a robust representation with part annotations. Experiments show that MPM yielded better adversarial robustness on the large-scale IN-1K over strong baselines across various attack settings. Furthermore, MPM achieved improved robustness on common corruptions and several out-of-distribution datasets. The dataset, together with these results, enables and encourages researchers to explore the potential of part-based models in more real applications.

摘要: 基于深度学习的目标识别系统很容易被各种对抗性扰动所愚弄。健壮性较弱的一个原因可能是它们不像人类识别过程那样具有基于部分的归纳偏差。基于此，人们提出了几种基于部分的识别模型，以提高识别的对抗性。然而，由于缺乏部分标注，这些方法的有效性仅在小规模非标准数据集上得到验证。在这项工作中，我们提出了PIN++，即PartImageNet++，一个为ImageNet-1K(IN-1K)的所有类别提供高质量零件分割标注的数据集。有了这些注释，我们直接在标准IN-1K数据集上构建基于部件的方法，以实现稳健识别。与以往基于零件的两阶段模型不同，我们提出了一种多尺度零件监督模型(MPM)，以学习带有零件标注的稳健表示。实验表明，MPM在各种攻击环境下的强基线攻击下，对大规模IN-1K具有更好的攻击健壮性。此外，MPM在常见的损坏和几个非分布的数据集上实现了更好的稳健性。数据集与这些结果一起，使研究人员能够并鼓励他们探索基于零件的模型在更真实的应用中的潜力。



## **31. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

（图）神经网络对抗数据中毒和后门攻击的可证明鲁棒性 cs.LG

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10867v1) [paper-pdf](http://arxiv.org/pdf/2407.10867v1)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data, as well as backdoor attacks that additionally manipulate the test data. These vulnerabilities have led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning and backdoor attacks targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.

摘要: 机器学习模型的泛化可能会受到数据中毒(对训练数据应用对抗性更改)以及另外操纵测试数据的后门攻击的严重影响。这些漏洞导致了人们对证明(即证明)这样的变化不会影响测试预测的兴趣。我们首次证明了图神经网络(GNN)不会受到针对给定图的节点特征的中毒和后门攻击。我们的证书是白盒的，并且基于$(I)$神经正切核，它表征了足够广泛的网络的训练动力学；$(Ii)$是将中毒描述为混合整数线性规划的双层优化问题的新形式。因此，我们利用我们的框架来提供关于图结构及其连通性对基于卷积和基于PageRank的GNN的最坏情况健壮性行为的作用的基本见解。我们注意到，我们的框架更通用，并且构成了第一种为NNS派生白盒中毒证书的方法，这可能是图相关任务之外的独立兴趣。



## **32. Secure Aggregation is Not Private Against Membership Inference Attacks**

安全聚合对于成员推断攻击来说不是私有的 cs.LG

accepted to the European Conference on Machine Learning and  Principles and Practice of Knowledge Discovery in Databases (ECML PKDD) 2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2403.17775v3) [paper-pdf](http://arxiv.org/pdf/2403.17775v3)

**Authors**: Khac-Hoang Ngo, Johan Östman, Giuseppe Durisi, Alexandre Graell i Amat

**Abstract**: Secure aggregation (SecAgg) is a commonly-used privacy-enhancing mechanism in federated learning, affording the server access only to the aggregate of model updates while safeguarding the confidentiality of individual updates. Despite widespread claims regarding SecAgg's privacy-preserving capabilities, a formal analysis of its privacy is lacking, making such presumptions unjustified. In this paper, we delve into the privacy implications of SecAgg by treating it as a local differential privacy (LDP) mechanism for each local update. We design a simple attack wherein an adversarial server seeks to discern which update vector a client submitted, out of two possible ones, in a single training round of federated learning under SecAgg. By conducting privacy auditing, we assess the success probability of this attack and quantify the LDP guarantees provided by SecAgg. Our numerical results unveil that, contrary to prevailing claims, SecAgg offers weak privacy against membership inference attacks even in a single training round. Indeed, it is difficult to hide a local update by adding other independent local updates when the updates are of high dimension. Our findings underscore the imperative for additional privacy-enhancing mechanisms, such as noise injection, in federated learning.

摘要: 安全聚合(SecAgg)是联合学习中常用的隐私增强机制，仅允许服务器访问模型更新的聚合，同时保护单个更新的机密性。尽管人们普遍声称SecAgg具有保护隐私的能力，但缺乏对其隐私的正式分析，这使得这种假设是不合理的。在本文中，我们深入研究了SecAgg的隐私含义，将其视为针对每个本地更新的本地差异隐私(LDP)机制。我们设计了一个简单的攻击，其中敌对服务器试图在SecAgg下的联合学习的单个训练轮中，从两个可能的更新向量中辨别客户端提交的更新向量。通过进行隐私审计，我们评估了该攻击的成功概率，并量化了SecAgg提供的LDP保证。我们的数值结果表明，与流行的说法相反，SecAgg即使在一轮训练中也提供了针对成员推理攻击的弱隐私。事实上，当更新是高维时，很难通过添加其他独立的本地更新来隐藏本地更新。我们的发现强调了联合学习中额外的隐私增强机制的必要性，例如噪音注入。



## **33. The Quantum Imitation Game: Reverse Engineering of Quantum Machine Learning Models**

量子模仿游戏：量子机器学习模型的反向工程 quant-ph

11 pages, 12 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.07237v2) [paper-pdf](http://arxiv.org/pdf/2407.07237v2)

**Authors**: Archisman Ghosh, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) amalgamates quantum computing paradigms with machine learning models, providing significant prospects for solving complex problems. However, with the expansion of numerous third-party vendors in the Noisy Intermediate-Scale Quantum (NISQ) era of quantum computing, the security of QML models is of prime importance, particularly against reverse engineering, which could expose trained parameters and algorithms of the models. We assume the untrusted quantum cloud provider is an adversary having white-box access to the transpiled user-designed trained QML model during inference. Reverse engineering (RE) to extract the pre-transpiled QML circuit will enable re-transpilation and usage of the model for various hardware with completely different native gate sets and even different qubit technology. Such flexibility may not be obtained from the transpiled circuit which is tied to a particular hardware and qubit technology. The information about the number of parameters, and optimized values can allow further training of the QML model to alter the QML model, tamper with the watermark, and/or embed their own watermark or refine the model for other purposes. In this first effort to investigate the RE of QML circuits, we perform RE and compare the training accuracy of original and reverse-engineered Quantum Neural Networks (QNNs) of various sizes. We note that multi-qubit classifiers can be reverse-engineered under specific conditions with a mean error of order 1e-2 in a reasonable time. We also propose adding dummy fixed parametric gates in the QML models to increase the RE overhead for defense. For instance, adding 2 dummy qubits and 2 layers increases the overhead by ~1.76 times for a classifier with 2 qubits and 3 layers with a performance overhead of less than 9%. We note that RE is a very powerful attack model which warrants further efforts on defenses.

摘要: 量子机器学习(QML)融合了量子计算范式和机器学习模型，为解决复杂问题提供了重要的前景。然而，在喧嚣的中间尺度量子计算(NISQ)时代，随着众多第三方供应商的扩张，QML模型的安全性至关重要，特别是在对抗逆向工程时，逆向工程可能会暴露模型的训练参数和算法。我们假设不可信的量子云提供商是一个对手，在推理过程中可以通过白盒访问用户设计的经过训练的QML模型。逆向工程(RE)提取预转换的QML电路将使模型能够重新转置并用于具有完全不同的本机门设置甚至不同的量子比特技术的各种硬件。这种灵活性可能不是从绑定到特定硬件和量子比特技术的分流电路获得的。关于参数数目和最佳值的信息可以允许进一步训练QML模型以改变QML模型、篡改水印、和/或出于其他目的嵌入它们自己的水印或改进模型。在第一次研究QML电路的RE时，我们进行了RE，并比较了不同大小的原始和反向工程量子神经网络(QNN)的训练精度。我们注意到，多量子比特分类器可以在特定条件下进行逆向工程，在合理的时间内，平均误差为1e-2阶。我们还建议在QML模型中增加虚拟固定参数门，以增加防御的RE开销。例如，对于具有2个量子比特和3个层的分类器，添加2个虚拟量子比特和2个层会使开销增加~1.76倍，而性能开销不到9%。我们注意到，RE是一种非常强大的攻击模式，需要在防御上进一步努力。



## **34. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.01295v4) [paper-pdf](http://arxiv.org/pdf/2407.01295v4)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **35. TAPI: Towards Target-Specific and Adversarial Prompt Injection against Code LLMs**

TAPI：针对代码LLM的目标特定和对抗性即时注入 cs.CR

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.09164v2) [paper-pdf](http://arxiv.org/pdf/2407.09164v2)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully used to simplify and facilitate code programming. With these tools, developers can easily generate desired complete functional codes based on incomplete code and natural language prompts. However, a few pioneering works revealed that these Code LLMs are also vulnerable, e.g., against backdoor and adversarial attacks. The former could induce LLMs to respond to triggers to insert malicious code snippets by poisoning the training data or model parameters, while the latter can craft malicious adversarial input codes to reduce the quality of generated codes. However, both attack methods have underlying limitations: backdoor attacks rely on controlling the model training process, while adversarial attacks struggle with fulfilling specific malicious purposes.   To inherit the advantages of both backdoor and adversarial attacks, this paper proposes a new attack paradigm, i.e., target-specific and adversarial prompt injection (TAPI), against Code LLMs. TAPI generates unreadable comments containing information about malicious instructions and hides them as triggers in the external source code. When users exploit Code LLMs to complete codes containing the trigger, the models will generate attacker-specified malicious code snippets at specific locations. We evaluate our TAPI attack on four representative LLMs under three representative malicious objectives and seven cases. The results show that our method is highly threatening (achieving an attack success rate enhancement of up to 89.3%) and stealthy (saving an average of 53.1% of tokens in the trigger design). In particular, we successfully attack some famous deployed code completion integrated applications, including CodeGeex and Github Copilot. This further confirms the realistic threat of our attack.

摘要: 最近，面向代码的大型语言模型(Code LLM)已被广泛并成功地用于简化和促进代码编程。使用这些工具，开发人员可以根据不完整的代码和自然语言提示轻松生成所需的完整功能代码。然而，一些开创性的工作表明，这些代码LLM也容易受到攻击，例如，抵御后门和对手攻击。前者可以通过毒化训练数据或模型参数来诱导LLMS响应插入恶意代码片段的触发器，而后者可以手工创建恶意输入代码来降低生成代码的质量。然而，这两种攻击方法都有潜在的局限性：后门攻击依赖于控制模型训练过程，而对抗性攻击则难以实现特定的恶意目的。为了继承后门攻击和对抗性攻击的优点，提出了一种新的针对Code LLMS的攻击范式，即目标特定和对抗性提示注入(TAPI)。TAPI生成不可读的注释，其中包含有关恶意指令的信息，并将它们作为触发器隐藏在外部源代码中。当用户利用Code LLMS来完成包含触发器的代码时，模型将在特定位置生成攻击者指定的恶意代码片段。我们在三个典型的恶意目标和七个案例下评估了我们的TAPI攻击对四个有代表性的LLM的攻击。结果表明，该方法具有很强的威胁性(攻击成功率提高高达89.3%)和隐蔽性(在触发设计中平均节省53.1%的令牌)。特别是，我们成功地攻击了一些著名的部署代码完成集成应用程序，包括CodeGeex和Github Copilot。这进一步证实了我们攻击的现实威胁。



## **36. Self-Evaluation as a Defense Against Adversarial Attacks on LLMs**

自我评估作为对LLM的对抗攻击的防御 cs.LG

8 pages, 7 figures

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.03234v2) [paper-pdf](http://arxiv.org/pdf/2407.03234v2)

**Authors**: Hannah Brown, Leon Lin, Kenji Kawaguchi, Michael Shieh

**Abstract**: When LLMs are deployed in sensitive, human-facing settings, it is crucial that they do not output unsafe, biased, or privacy-violating outputs. For this reason, models are both trained and instructed to refuse to answer unsafe prompts such as "Tell me how to build a bomb." We find that, despite these safeguards, it is possible to break model defenses simply by appending a space to the end of a model's input. In a study of eight open-source models, we demonstrate that this acts as a strong enough attack to cause the majority of models to generate harmful outputs with very high success rates. We examine the causes of this behavior, finding that the contexts in which single spaces occur in tokenized training data encourage models to generate lists when prompted, overriding training signals to refuse to answer unsafe requests. Our findings underscore the fragile state of current model alignment and promote the importance of developing more robust alignment methods. Code and data will be made available at https://github.com/Linlt-leon/self-eval.

摘要: 当LLM部署在敏感的、面向人的环境中时，至关重要的是它们不输出不安全、有偏见或违反隐私的输出。出于这个原因，模特们既接受了培训，又被指示拒绝回答不安全的提示，比如“告诉我如何制造炸弹。”我们发现，尽管有这些保障措施，但只需在模型输入的末尾添加一个空格，就可以打破模型的防御。在对八个开源模型的研究中，我们证明了这是一种足够强大的攻击，足以导致大多数模型产生非常高的成功率的有害输出。我们研究了这种行为的原因，发现在标记化的训练数据中出现单个空格的上下文鼓励模型在得到提示时生成列表，从而覆盖拒绝回答不安全请求的训练信号。我们的发现强调了当前模型比对的脆弱状态，并促进了开发更稳健的比对方法的重要性。代码和数据将在https://github.com/Linlt-leon/self-eval.上提供



## **37. Backdoor Attacks against Image-to-Image Networks**

针对图像到图像网络的后门攻击 cs.CV

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2407.10445v1) [paper-pdf](http://arxiv.org/pdf/2407.10445v1)

**Authors**: Wenbo Jiang, Hongwei Li, Jiaming He, Rui Zhang, Guowen Xu, Tianwei Zhang, Rongxing Lu

**Abstract**: Recently, deep learning-based Image-to-Image (I2I) networks have become the predominant choice for I2I tasks such as image super-resolution and denoising. Despite their remarkable performance, the backdoor vulnerability of I2I networks has not been explored. To fill this research gap, we conduct a comprehensive investigation on the susceptibility of I2I networks to backdoor attacks. Specifically, we propose a novel backdoor attack technique, where the compromised I2I network behaves normally on clean input images, yet outputs a predefined image of the adversary for malicious input images containing the trigger. To achieve this I2I backdoor attack, we propose a targeted universal adversarial perturbation (UAP) generation algorithm for I2I networks, where the generated UAP is used as the backdoor trigger. Additionally, in the backdoor training process that contains the main task and the backdoor task, multi-task learning (MTL) with dynamic weighting methods is employed to accelerate convergence rates. In addition to attacking I2I tasks, we extend our I2I backdoor to attack downstream tasks, including image classification and object detection. Extensive experiments demonstrate the effectiveness of the I2I backdoor on state-of-the-art I2I network architectures, as well as the robustness against different mainstream backdoor defenses.

摘要: 近年来，基于深度学习的图像到图像(I2I)网络已经成为图像超分辨率和去噪等I2I任务的主要选择。尽管它们表现出色，但I2I网络的后门漏洞尚未被发现。为了填补这一研究空白，我们对I2I网络对后门攻击的敏感度进行了全面的调查。具体地说，我们提出了一种新的后门攻击技术，其中受攻击的I2I网络在干净的输入图像上表现正常，但对于包含触发器的恶意输入图像输出对手的预定义图像。为了实现这种I2I后门攻击，我们提出了一种针对I2I网络的有针对性的通用对抗扰动(UAP)生成算法，生成的UAP被用作后门触发器。此外，在包含主任务和后门任务的后门训练过程中，采用了具有动态加权方法的多任务学习(MTL)来加快收敛速度。除了攻击I2I任务外，我们还将I2I后门扩展到攻击下游任务，包括图像分类和目标检测。广泛的实验证明了I2I后门在最先进的I2I网络架构上的有效性，以及对不同主流后门防御的健壮性。



## **38. Pandora's White-Box: Precise Training Data Detection and Extraction in Large Language Models**

潘多拉的白盒：大型语言模型中的精确训练数据检测和提取 cs.CR

Found software bug in experiments, withdrawing in order to address  and update results

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2402.17012v4) [paper-pdf](http://arxiv.org/pdf/2402.17012v4)

**Authors**: Jeffrey G. Wang, Jason Wang, Marvin Li, Seth Neel

**Abstract**: In this paper we develop state-of-the-art privacy attacks against Large Language Models (LLMs), where an adversary with some access to the model tries to learn something about the underlying training data. Our headline results are new membership inference attacks (MIAs) against pretrained LLMs that perform hundreds of times better than baseline attacks, and a pipeline showing that over 50% (!) of the fine-tuning dataset can be extracted from a fine-tuned LLM in natural settings. We consider varying degrees of access to the underlying model, pretraining and fine-tuning data, and both MIAs and training data extraction. For pretraining data, we propose two new MIAs: a supervised neural network classifier that predicts training data membership on the basis of (dimensionality-reduced) model gradients, as well as a variant of this attack that only requires logit access to the model by leveraging recent model-stealing work on LLMs. To our knowledge this is the first MIA that explicitly incorporates model-stealing information. Both attacks outperform existing black-box baselines, and our supervised attack closes the gap between MIA attack success against LLMs and the strongest known attacks for other machine learning models. In fine-tuning, we find that a simple attack based on the ratio of the loss between the base and fine-tuned models is able to achieve near-perfect MIA performance; we then leverage our MIA to extract a large fraction of the fine-tuning dataset from fine-tuned Pythia and Llama models. Our code is available at github.com/safr-ai-lab/pandora-llm.

摘要: 在本文中，我们开发了针对大型语言模型(LLM)的最先进的隐私攻击，其中对该模型具有一定访问权限的对手试图了解一些关于潜在训练数据的信息。我们的主要结果是针对预先训练的LLM的新成员推理攻击(MIA)，其性能比基线攻击高数百倍，并且管道显示超过50%(！)可以在自然设置下从微调的LLM中提取精调数据集的。我们考虑了不同程度的访问底层模型、预训练和微调数据，以及MIA和训练数据提取。对于预训练数据，我们提出了两个新的MIA：一个有监督的神经网络分类器，它基于(降维)模型梯度预测训练数据的成员资格，以及这种攻击的一个变体，它只需要通过利用LLMS上最近的模型窃取工作来对模型进行Logit访问。据我们所知，这是第一个明确纳入模型窃取信息的MIA。这两种攻击都超过了现有的黑盒基线，我们的监督攻击缩小了针对LLMS的MIA攻击成功与针对其他机器学习模型的已知最强攻击之间的差距。在微调中，我们发现基于基本模型和微调模型之间的损失比率的简单攻击能够获得近乎完美的MIA性能；然后，我们利用我们的MIA从微调的Pythia和Llama模型中提取很大一部分微调数据集。我们的代码可以在githorb.com/Safr-ai-lab/pandora-llm上找到。



## **39. MultiDelete for Multimodal Machine Unlearning**

MultiEdit用于多模式机器取消学习 cs.AI

ECCV 2024

**SubmitDate**: 2024-07-15    [abs](http://arxiv.org/abs/2311.12047v2) [paper-pdf](http://arxiv.org/pdf/2311.12047v2)

**Authors**: Jiali Cheng, Hadi Amiri

**Abstract**: Machine Unlearning removes specific knowledge about training data samples from an already trained model. It has significant practical benefits, such as purging private, inaccurate, or outdated information from trained models without the need for complete re-training. Unlearning within a multimodal setting presents unique challenges due to the complex dependencies between different data modalities and the expensive cost of training on large multimodal datasets and architectures. This paper presents the first machine unlearning approach for multimodal data and models, titled MultiDelete, which is designed to decouple associations between unimodal data points during unlearning without losing the overall representation strength of the trained model. MultiDelete advocates for three key properties for effective multimodal unlearning: (a): modality decoupling, which effectively decouples the association between individual unimodal data points marked for deletion, rendering them as unrelated data points, (b): multimodal knowledge retention, which retains the multimodal representation post-unlearning, and (c): unimodal knowledge retention, which retains the unimodal representation postunlearning. MultiDelete is efficient to train and is not constrained by using a strongly convex loss -- a common restriction among existing baselines. Experiments on two architectures and four datasets, including image-text and graph-text datasets, show that MultiDelete gains an average improvement of 17.6 points over best performing baseline in unlearning multimodal samples, can maintain the multimodal and unimodal knowledge of the original model post unlearning, and can provide better protection to unlearned data against adversarial attacks.

摘要: 机器忘却学习从已训练的模型中移除关于训练数据样本的特定知识。它有显著的实际好处，例如从训练的模型中清除私人的、不准确的或过时的信息，而不需要完全的重新训练。由于不同数据模式之间的复杂依赖关系，以及在大型多模式数据集和体系结构上进行培训的昂贵成本，在多模式环境中遗忘是一项独特的挑战。本文提出了第一种用于多模式数据和模型的机器遗忘方法，称为MultiDelete，它被设计用于在遗忘过程中解耦单峰数据点之间的关联，而不损失训练模型的整体表示能力。MultiDelete倡导有效的多通道去学习的三个关键特性：(A)：通道去耦合，它有效地去耦合标记为删除的单个单通道数据点之间的关联，使它们成为不相关的数据点；(B)：多通道知识保持，它在遗忘后保留多通道表征；(C)：单通道知识保持，它在遗忘后保留单通道表征。多删除训练是有效的，并且不受使用强凸损失的限制--这是现有基线中的常见限制。在两种结构和四个数据集(包括图像-文本和图文数据集)上的实验表明，MultiDelete在遗忘多模式样本时比最佳基线平均提高17.6个点，能够在遗忘后保持原始模型的多模式和单峰知识，并能更好地保护未学习的数据免受敌意攻击。



## **40. SENTINEL: Securing Indoor Localization against Adversarial Attacks with Capsule Neural Networks**

SENTINEL：利用胶囊神经网络确保室内定位免受对抗性攻击 eess.SP

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.11091v1) [paper-pdf](http://arxiv.org/pdf/2407.11091v1)

**Authors**: Danish Gufran, Pooja Anandathirtha, Sudeep Pasricha

**Abstract**: With the increasing demand for edge device powered location-based services in indoor environments, Wi-Fi received signal strength (RSS) fingerprinting has become popular, given the unavailability of GPS indoors. However, achieving robust and efficient indoor localization faces several challenges, due to RSS fluctuations from dynamic changes in indoor environments and heterogeneity of edge devices, leading to diminished localization accuracy. While advances in machine learning (ML) have shown promise in mitigating these phenomena, it remains an open problem. Additionally, emerging threats from adversarial attacks on ML-enhanced indoor localization systems, especially those introduced by malicious or rogue access points (APs), can deceive ML models to further increase localization errors. To address these challenges, we present SENTINEL, a novel embedded ML framework utilizing modified capsule neural networks to bolster the resilience of indoor localization solutions against adversarial attacks, device heterogeneity, and dynamic RSS fluctuations. We also introduce RSSRogueLoc, a novel dataset capturing the effects of rogue APs from several real-world indoor environments. Experimental evaluations demonstrate that SENTINEL achieves significant improvements, with up to 3.5x reduction in mean error and 3.4x reduction in worst-case error compared to state-of-the-art frameworks using simulated adversarial attacks. SENTINEL also achieves improvements of up to 2.8x in mean error and 2.7x in worst-case error compared to state-of-the-art frameworks when evaluated with the real-world RSSRogueLoc dataset.

摘要: 随着室内环境中对边缘设备供电的基于位置的服务的需求不断增加，由于室内无法使用GPS，Wi-Fi接收信号强度(RSS)指纹识别已变得流行起来。然而，由于室内环境的动态变化和边缘设备的异构性导致RSS波动，导致定位精度降低，实现稳健和高效的室内定位面临着一些挑战。虽然机器学习(ML)的进步在缓解这些现象方面显示了希望，但它仍然是一个悬而未决的问题。此外，对ML增强型室内定位系统的敌意攻击，特别是由恶意或流氓接入点(AP)引入的威胁，可能会欺骗ML模型，进一步增加定位错误。为了应对这些挑战，我们提出了一种新颖的嵌入式ML框架Sentinel，它利用改进的胶囊神经网络来增强室内定位解决方案对对手攻击、设备异构性和动态RSS波动的弹性。我们还介绍了RSSRogueLoc，这是一个新的数据集，可以从几个真实的室内环境中捕获恶意AP的影响。实验评估表明，与使用模拟对抗性攻击的最新框架相比，Sentinel取得了显著的改进，平均误差降低了3.5倍，最坏情况误差降低了3.4倍。与最先进的框架相比，Sentinel在使用真实的RSSRogueLoc数据集进行评估时，平均误差和最坏情况误差分别提高了2.8倍和2.7倍。



## **41. Proof-of-Learning with Incentive Security**

具有激励保障的学习证明 cs.CR

17 pages

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2404.09005v6) [paper-pdf](http://arxiv.org/pdf/2404.09005v6)

**Authors**: Zishuo Zhao, Zhixuan Fang, Xuechao Wang, Xi Chen, Yuan Zhou

**Abstract**: Most concurrent blockchain systems rely heavily on the Proof-of-Work (PoW) or Proof-of-Stake (PoS) mechanisms for decentralized consensus and security assurance. However, the substantial energy expenditure stemming from computationally intensive yet meaningless tasks has raised considerable concerns surrounding traditional PoW approaches, The PoS mechanism, while free of energy consumption, is subject to security and economic issues. Addressing these issues, the paradigm of Proof-of-Useful-Work (PoUW) seeks to employ challenges of practical significance as PoW, thereby imbuing energy consumption with tangible value. While previous efforts in Proof of Learning (PoL) explored the utilization of deep learning model training SGD tasks as PoUW challenges, recent research has revealed its vulnerabilities to adversarial attacks and the theoretical hardness in crafting a byzantine-secure PoL mechanism. In this paper, we introduce the concept of incentive-security that incentivizes rational provers to behave honestly for their best interest, bypassing the existing hardness to design a PoL mechanism with computational efficiency, a provable incentive-security guarantee and controllable difficulty. Particularly, our work is secure against two attacks to the recent work of Jia et al. [2021], and also improves the computational overhead from $\Theta(1)$ to $O(\frac{\log E}{E})$. Furthermore, while most recent research assumes trusted problem providers and verifiers, our design also guarantees frontend incentive-security even when problem providers are untrusted, and verifier incentive-security that bypasses the Verifier's Dilemma. By incorporating ML training into blockchain consensus mechanisms with provable guarantees, our research not only proposes an eco-friendly solution to blockchain systems, but also provides a proposal for a completely decentralized computing power market in the new AI age.

摘要: 大多数并发区块链系统严重依赖工作证明(POW)或风险证明(POS)机制来实现去中心化共识和安全保证。然而，计算密集但无意义的任务所产生的大量能源支出引起了人们对传统POW方法的相当大的担忧，POS机制虽然没有能源消耗，但受到安全和经济问题的影响。针对这些问题，有用工作证明(POUW)范式试图将具有实际意义的挑战作为POW来使用，从而使能源消耗具有有形价值。虽然先前在学习证明(Pol)方面的努力探索了利用深度学习模型训练SGD任务作为POW挑战，但最近的研究揭示了它对对手攻击的脆弱性以及在设计拜占庭安全的POL机制方面的理论难度。本文引入激励安全的概念，激励理性的证明者为了他们的最大利益而诚实地行事，绕过现有的困难，设计了一个具有计算效率、可证明的激励安全保证和可控难度的POL机制。特别是，我们的工作是安全的，可以抵抗对Jia等人最近的工作的两次攻击。[2021]并将计算开销从$\theta(1)$提高到$O(\frac{\log E}{E})$。此外，虽然最近的研究假设可信的问题提供者和验证者，但我们的设计也保证了前端激励-安全性，即使问题提供者是不可信的，并且验证者激励-安全绕过了验证者的困境。通过将ML培训融入到具有可证明保证的区块链共识机制中，我们的研究不仅为区块链系统提出了生态友好的解决方案，而且为新AI时代完全去中心化的计算能力市场提供了建议。



## **42. Merging Improves Self-Critique Against Jailbreak Attacks**

合并提高了对越狱袭击的自我批评 cs.CL

Published at ICML 2024 Workshop on Foundation Models in the Wild

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2406.07188v2) [paper-pdf](http://arxiv.org/pdf/2406.07188v2)

**Authors**: Victor Gallego

**Abstract**: The robustness of large language models (LLMs) against adversarial manipulations, such as jailbreak attacks, remains a significant challenge. In this work, we propose an approach that enhances the self-critique capability of the LLM and further fine-tunes it over sanitized synthetic data. This is done with the addition of an external critic model that can be merged with the original, thus bolstering self-critique capabilities and improving the robustness of the LLMs response to adversarial prompts. Our results demonstrate that the combination of merging and self-critique can reduce the attack success rate of adversaries significantly, thus offering a promising defense mechanism against jailbreak attacks. Code, data and models released at https://github.com/vicgalle/merging-self-critique-jailbreaks .

摘要: 大型语言模型（LLM）对越狱攻击等对抗性操纵的稳健性仍然是一个重大挑战。在这项工作中，我们提出了一种增强LLM自我批评能力的方法，并根据净化的合成数据进一步对其进行微调。这是通过添加一个可以与原始模型合并的外部批评者模型来实现的，从而增强自我批评能力并提高LLM对对抗提示反应的稳健性。我们的结果表明，合并和自我批评的结合可以显着降低对手的攻击成功率，从而提供一种有希望的针对越狱攻击的防御机制。代码、数据和模型在https://github.com/vicgalle/merging-self-critique-jailbreaks上发布。



## **43. Boosting Transferability in Vision-Language Attacks via Diversification along the Intersection Region of Adversarial Trajectory**

通过沿着对抗轨迹交叉区域的多样化来提高视觉语言攻击的可移植性 cs.CV

ECCV2024. Code is available at  https://github.com/SensenGao/VLPTransferAttack

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2403.12445v3) [paper-pdf](http://arxiv.org/pdf/2403.12445v3)

**Authors**: Sensen Gao, Xiaojun Jia, Xuhong Ren, Ivor Tsang, Qing Guo

**Abstract**: Vision-language pre-training (VLP) models exhibit remarkable capabilities in comprehending both images and text, yet they remain susceptible to multimodal adversarial examples (AEs). Strengthening attacks and uncovering vulnerabilities, especially common issues in VLP models (e.g., high transferable AEs), can advance reliable and practical VLP models. A recent work (i.e., Set-level guidance attack) indicates that augmenting image-text pairs to increase AE diversity along the optimization path enhances the transferability of adversarial examples significantly. However, this approach predominantly emphasizes diversity around the online adversarial examples (i.e., AEs in the optimization period), leading to the risk of overfitting the victim model and affecting the transferability. In this study, we posit that the diversity of adversarial examples towards the clean input and online AEs are both pivotal for enhancing transferability across VLP models. Consequently, we propose using diversification along the intersection region of adversarial trajectory to expand the diversity of AEs. To fully leverage the interaction between modalities, we introduce text-guided adversarial example selection during optimization. Furthermore, to further mitigate the potential overfitting, we direct the adversarial text deviating from the last intersection region along the optimization path, rather than adversarial images as in existing methods. Extensive experiments affirm the effectiveness of our method in improving transferability across various VLP models and downstream vision-and-language tasks.

摘要: 视觉-语言预训练(VLP)模型在理解图像和文本方面表现出显著的能力，但它们仍然容易受到多通道对抗性例子(AEs)的影响。加强攻击和发现漏洞，特别是VLP模型中的常见问题(例如，高可转移的AE)，可以促进可靠和实用的VLP模型。最近的一项工作(即集合级制导攻击)表明，增加图文对以增加优化路径上的声发射多样性显著地提高了对抗性例子的可转移性。然而，这种方法主要强调围绕在线对抗性例子的多样性(即，处于优化期的AEs)，导致过度匹配受害者模型并影响可转移性的风险。在这项研究中，我们假设，针对干净输入和在线AEs的对抗性例子的多样性对于提高VLP模型之间的可转移性都是关键。因此，我们建议沿着对抗性轨迹的交叉点区域进行多样化，以扩大AEs的多样性。为了充分利用通道之间的交互作用，我们在优化过程中引入了文本引导的对抗性实例选择。此外，为了进一步缓解潜在的过拟合，我们沿着优化路径引导偏离最后一个交集区域的对抗性文本，而不是现有方法中的对抗性图像。广泛的实验证实了我们的方法在提高各种VLP模型和下游视觉和语言任务的可转移性方面的有效性。



## **44. CLIP-Guided Networks for Transferable Targeted Attacks**

CLIP引导的可转移定向攻击网络 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10179v1) [paper-pdf](http://arxiv.org/pdf/2407.10179v1)

**Authors**: Hao Fang, Jiawei Kong, Bin Chen, Tao Dai, Hao Wu, Shu-Tao Xia

**Abstract**: Transferable targeted adversarial attacks aim to mislead models into outputting adversary-specified predictions in black-box scenarios. Recent studies have introduced \textit{single-target} generative attacks that train a generator for each target class to generate highly transferable perturbations, resulting in substantial computational overhead when handling multiple classes. \textit{Multi-target} attacks address this by training only one class-conditional generator for multiple classes. However, the generator simply uses class labels as conditions, failing to leverage the rich semantic information of the target class. To this end, we design a \textbf{C}LIP-guided \textbf{G}enerative \textbf{N}etwork with \textbf{C}ross-attention modules (CGNC) to enhance multi-target attacks by incorporating textual knowledge of CLIP into the generator. Extensive experiments demonstrate that CGNC yields significant improvements over previous multi-target generative attacks, e.g., a 21.46\% improvement in success rate from ResNet-152 to DenseNet-121. Moreover, we propose a masked fine-tuning mechanism to further strengthen our method in attacking a single class, which surpasses existing single-target methods.

摘要: 可转移的目标对抗性攻击旨在误导模型，使其在黑盒场景中输出对手指定的预测。最近的研究引入了生成性攻击，这种攻击为每个目标类训练一个生成器来生成高度可传递的扰动，导致在处理多个类时产生大量的计算开销。\textit{多目标}攻击通过仅训练多个类的一个类条件生成器来解决此问题。然而，生成器简单地使用类标签作为条件，没有利用目标类的丰富语义信息。为此，我们设计了一个唇形引导的生成模块(CGNC)，通过在生成器中加入剪辑文本知识来增强多目标攻击。大量的实验表明，CGNC比以前的多目标生成性攻击有显著的改进，例如，成功率从ResNet-152提高到DenseNet-121，提高了21.46%.此外，我们还提出了一种屏蔽微调机制，进一步加强了我们的攻击单一类的方法，超越了现有的单目标攻击方法。



## **45. Can Adversarial Examples Be Parsed to Reveal Victim Model Information?**

可以解析对抗性例子来揭示受害者模型信息吗？ cs.CV

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2303.07474v3) [paper-pdf](http://arxiv.org/pdf/2303.07474v3)

**Authors**: Yuguang Yao, Jiancheng Liu, Yifan Gong, Xiaoming Liu, Yanzhi Wang, Xue Lin, Sijia Liu

**Abstract**: Numerous adversarial attack methods have been developed to generate imperceptible image perturbations that can cause erroneous predictions of state-of-the-art machine learning (ML) models, in particular, deep neural networks (DNNs). Despite intense research on adversarial attacks, little effort was made to uncover 'arcana' carried in adversarial attacks. In this work, we ask whether it is possible to infer data-agnostic victim model (VM) information (i.e., characteristics of the ML model or DNN used to generate adversarial attacks) from data-specific adversarial instances. We call this 'model parsing of adversarial attacks' - a task to uncover 'arcana' in terms of the concealed VM information in attacks. We approach model parsing via supervised learning, which correctly assigns classes of VM's model attributes (in terms of architecture type, kernel size, activation function, and weight sparsity) to an attack instance generated from this VM. We collect a dataset of adversarial attacks across 7 attack types generated from 135 victim models (configured by 5 architecture types, 3 kernel size setups, 3 activation function types, and 3 weight sparsity ratios). We show that a simple, supervised model parsing network (MPN) is able to infer VM attributes from unseen adversarial attacks if their attack settings are consistent with the training setting (i.e., in-distribution generalization assessment). We also provide extensive experiments to justify the feasibility of VM parsing from adversarial attacks, and the influence of training and evaluation factors in the parsing performance (e.g., generalization challenge raised in out-of-distribution evaluation). We further demonstrate how the proposed MPN can be used to uncover the source VM attributes from transfer attacks, and shed light on a potential connection between model parsing and attack transferability.

摘要: 已经开发了许多对抗性攻击方法来产生不可察觉的图像扰动，这可能导致对最先进的机器学习(ML)模型的错误预测，特别是深度神经网络(DNN)。尽管对对抗性攻击进行了密集的研究，但几乎没有努力去发现对抗性攻击中携带的“奥秘”。在这项工作中，我们问是否有可能从特定于数据的对抗性实例中推断出与数据无关的受害者模型(VM)信息(即，用于生成对抗性攻击的ML模型或DNN的特征)。我们称之为“对抗性攻击的模型解析”--根据攻击中隐藏的VM信息来发现“奥秘”的任务。我们通过有监督的学习来实现模型解析，它正确地将VM的模型属性的类别(根据体系结构类型、核大小、激活函数和权重稀疏性)分配给从该VM生成的攻击实例。我们收集了从135个受害者模型(由5个体系结构类型、3个核大小设置、3个激活函数类型和3个权重稀疏率配置)生成的7种攻击类型的对抗性攻击的数据集。我们证明了一个简单的有监督的模型解析网络(MPN)能够从未知的敌意攻击中推断出VM属性，如果它们的攻击设置与训练设置一致(即分布内泛化评估)。我们还提供了大量的实验，以验证在敌意攻击下进行VM解析的可行性，以及训练和评估因素(例如，在分布外评估中提出的泛化挑战)对解析性能的影响。我们进一步演示了如何使用所提出的MPN来发现来自传输攻击的源VM属性，并阐明了模型解析和攻击可转移性之间的潜在联系。



## **46. Transferable 3D Adversarial Shape Completion using Diffusion Models**

使用扩散模型的可转移3D对抗形状完成 cs.CV

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2407.10077v1) [paper-pdf](http://arxiv.org/pdf/2407.10077v1)

**Authors**: Xuelong Dai, Bin Xiao

**Abstract**: Recent studies that incorporate geometric features and transformers into 3D point cloud feature learning have significantly improved the performance of 3D deep-learning models. However, their robustness against adversarial attacks has not been thoroughly explored. Existing attack methods primarily focus on white-box scenarios and struggle to transfer to recently proposed 3D deep-learning models. Even worse, these attacks introduce perturbations to 3D coordinates, generating unrealistic adversarial examples and resulting in poor performance against 3D adversarial defenses. In this paper, we generate high-quality adversarial point clouds using diffusion models. By using partial points as prior knowledge, we generate realistic adversarial examples through shape completion with adversarial guidance. The proposed adversarial shape completion allows for a more reliable generation of adversarial point clouds. To enhance attack transferability, we delve into the characteristics of 3D point clouds and employ model uncertainty for better inference of model classification through random down-sampling of point clouds. We adopt ensemble adversarial guidance for improved transferability across different network architectures. To maintain the generation quality, we limit our adversarial guidance solely to the critical points of the point clouds by calculating saliency scores. Extensive experiments demonstrate that our proposed attacks outperform state-of-the-art adversarial attack methods against both black-box models and defenses. Our black-box attack establishes a new baseline for evaluating the robustness of various 3D point cloud classification models.

摘要: 最近的研究将几何特征和变换融入到三维点云特征学习中，显著提高了三维深度学习模型的性能。然而，它们对敌意攻击的健壮性还没有得到彻底的研究。现有的攻击方法主要集中在白盒场景，很难转移到最近提出的3D深度学习模型。更糟糕的是，这些攻击对3D坐标引入了扰动，生成了不现实的对抗性示例，并导致对3D对抗性防御的性能不佳。在本文中，我们使用扩散模型来生成高质量的对抗性点云。利用局部点作为先验知识，通过对抗性指导下的形状补全生成真实的对抗性实例。所提出的对抗性形状补全允许更可靠地生成对抗性点云。为了增强攻击的可转移性，我们深入研究了三维点云的特点，并利用模型不确定性通过对点云进行随机下采样来更好地推断模型分类。我们采用集成对抗性指导，以提高跨不同网络架构的可传输性。为了保持生成质量，我们通过计算显著分数，将我们的对抗性指导仅限于点云的关键点。大量的实验表明，我们提出的攻击方法优于最新的对抗性攻击方法，无论是针对黑盒模型还是针对防御。我们的黑盒攻击为评估各种三维点云分类模型的稳健性建立了一个新的基线。



## **47. AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models**

AdvDiff：使用扩散模型生成无限制的对抗示例 cs.LG

ECCV 2024

**SubmitDate**: 2024-07-14    [abs](http://arxiv.org/abs/2307.12499v4) [paper-pdf](http://arxiv.org/pdf/2307.12499v4)

**Authors**: Xuelong Dai, Kaisheng Liang, Bin Xiao

**Abstract**: Unrestricted adversarial attacks present a serious threat to deep learning models and adversarial defense techniques. They pose severe security problems for deep learning applications because they can effectively bypass defense mechanisms. However, previous attack methods often directly inject Projected Gradient Descent (PGD) gradients into the sampling of generative models, which are not theoretically provable and thus generate unrealistic examples by incorporating adversarial objectives, especially for GAN-based methods on large-scale datasets like ImageNet. In this paper, we propose a new method, called AdvDiff, to generate unrestricted adversarial examples with diffusion models. We design two novel adversarial guidance techniques to conduct adversarial sampling in the reverse generation process of diffusion models. These two techniques are effective and stable in generating high-quality, realistic adversarial examples by integrating gradients of the target classifier interpretably. Experimental results on MNIST and ImageNet datasets demonstrate that AdvDiff is effective in generating unrestricted adversarial examples, which outperforms state-of-the-art unrestricted adversarial attack methods in terms of attack performance and generation quality.

摘要: 不受限制的对抗性攻击对深度学习模型和对抗性防御技术构成了严重威胁。它们会给深度学习应用程序带来严重的安全问题，因为它们可以有效地绕过防御机制。然而，以往的攻击方法往往直接将投影梯度下降(PGD)梯度注入生成模型的样本中，这在理论上是不可证明的，因此通过结合对抗性目标来生成不现实的例子，特别是对于基于GAN的方法在像ImageNet这样的大规模数据集上。在这篇文章中，我们提出了一种新的方法，称为AdvDiff，用来生成带有扩散模型的无限制对抗实例。我们设计了两种新的对抗性制导技术，用于在扩散模型的逆向生成过程中进行对抗性采样。这两种技术通过可解释地集成目标分类器的梯度，在生成高质量、真实的对抗性实例方面是有效和稳定的。在MNIST和ImageNet数据集上的实验结果表明，AdvDiff在生成无限制对抗性实例方面是有效的，在攻击性能和生成质量方面都优于现有的无限制对抗性攻击方法。



## **48. Harvesting Private Medical Images in Federated Learning Systems with Crafted Models**

使用精心设计的模型在联邦学习系统中收集私人医疗图像 cs.LG

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2407.09972v1) [paper-pdf](http://arxiv.org/pdf/2407.09972v1)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows a set of clients to collaboratively train a machine-learning model without exposing local training samples. In this context, it is considered to be privacy-preserving and hence has been adopted by medical centers to train machine-learning models over private data. However, in this paper, we propose a novel attack named MediLeak that enables a malicious parameter server to recover high-fidelity patient images from the model updates uploaded by the clients. MediLeak requires the server to generate an adversarial model by adding a crafted module in front of the original model architecture. It is published to the clients in the regular FL training process and each client conducts local training on it to generate corresponding model updates. Then, based on the FL protocol, the model updates are sent back to the server and our proposed analytical method recovers private data from the parameter updates of the crafted module. We provide a comprehensive analysis for MediLeak and show that it can successfully break the state-of-the-art cryptographic secure aggregation protocols, designed to protect the FL systems from privacy inference attacks. We implement MediLeak on the MedMNIST and COVIDx CXR-4 datasets. The results show that MediLeak can nearly perfectly recover private images with high recovery rates and quantitative scores. We further perform downstream tasks such as disease classification with the recovered data, where our results show no significant performance degradation compared to using the original training samples.

摘要: 联合学习(FL)允许一组客户在不暴露本地训练样本的情况下协作训练机器学习模型。在这种情况下，它被认为是隐私保护的，因此被医学中心采用来训练机器学习模型，而不是私人数据。然而，在本文中，我们提出了一种名为MediLeak的新型攻击，该攻击使恶意参数服务器能够从客户端上传的模型更新中恢复高保真的患者图像。MediLeak要求服务器通过在原始模型架构前面添加一个特制的模块来生成对抗性模型。它在定期的FL训练过程中发布给客户端，每个客户端对其进行本地训练，生成相应的模型更新。然后，基于FL协议，模型更新被发送回服务器，并且我们提出的分析方法从定制模块的参数更新中恢复私有数据。我们对MediLeak进行了全面的分析，表明它可以成功地破解最先进的密码安全聚合协议，这些协议旨在保护FL系统免受隐私推理攻击。我们在MedMNIST和COVIDx CXR-4数据集上实现了MediLeak。结果表明，MediLeak能够近乎完美地恢复私密图像，恢复率和量化分数都很高。我们使用恢复的数据进一步执行下游任务，例如疾病分类，其中我们的结果显示与使用原始训练样本相比，性能没有显著下降。



## **49. Black-Box Detection of Language Model Watermarks**

语言模型水印的黑匣子检测 cs.CR

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2405.20777v2) [paper-pdf](http://arxiv.org/pdf/2405.20777v2)

**Authors**: Thibaud Gloaguen, Nikola Jovanović, Robin Staab, Martin Vechev

**Abstract**: Watermarking has emerged as a promising way to detect LLM-generated text. To apply a watermark an LLM provider, given a secret key, augments generations with a signal that is later detectable by any party with the same key. Recent work has proposed three main families of watermarking schemes, two of which focus on the property of preserving the LLM distribution. This is motivated by it being a tractable proxy for maintaining LLM capabilities, but also by the idea that concealing a watermark deployment makes it harder for malicious actors to hide misuse by avoiding a certain LLM or attacking its watermark. Yet, despite much discourse around detectability, no prior work has investigated if any of these scheme families are detectable in a realistic black-box setting. We tackle this for the first time, developing rigorous statistical tests to detect the presence of all three most popular watermarking scheme families using only a limited number of black-box queries. We experimentally confirm the effectiveness of our methods on a range of schemes and a diverse set of open-source models. Our findings indicate that current watermarking schemes are more detectable than previously believed, and that obscuring the fact that a watermark was deployed may not be a viable way for providers to protect against adversaries. We further apply our methods to test for watermark presence behind the most popular public APIs: GPT4, Claude 3, Gemini 1.0 Pro, finding no strong evidence of a watermark at this point in time.

摘要: 水印技术已经成为检测LLM生成文本的一种很有前途的方法。为了应用水印，LLM提供商在给定秘密密钥的情况下，使用稍后可被具有相同密钥的任何一方检测的信号来增加生成。最近的工作已经提出了三类主要的数字水印方案，其中两类侧重于保持LLM分布的性质。这是因为它是维护LLM功能的易于处理的代理，但也是因为隐藏水印部署会使恶意攻击者更难通过避免特定LLM或攻击其水印来隐藏误用。然而，尽管有很多关于可检测性的讨论，但之前的工作还没有调查过这些方案家族中是否有任何一个在现实的黑盒环境中是可检测的。我们首次解决了这一问题，开发了严格的统计测试，仅使用有限数量的黑盒查询来检测所有三个最受欢迎的水印方案家族的存在。我们在一系列方案和一组不同的开源模型上实验证实了我们的方法的有效性。我们的发现表明，目前的水印方案比之前认为的更容易检测到，并且掩盖水印被部署的事实可能不是提供商保护免受对手攻击的可行方法。我们进一步应用我们的方法来测试最流行的公共API：GPT4、Claude 3、Gemini 1.0 Pro背后的水印存在，目前没有找到水印的有力证据。



## **50. SpecFormer: Guarding Vision Transformer Robustness via Maximum Singular Value Penalization**

SpecFormer：通过最大奇异值惩罚保护Vision Transformer的稳健性 cs.CV

Accepted by ECCV 2024; 27 pages; code is at:  https://github.com/microsoft/robustlearn

**SubmitDate**: 2024-07-13    [abs](http://arxiv.org/abs/2402.03317v2) [paper-pdf](http://arxiv.org/pdf/2402.03317v2)

**Authors**: Xixu Hu, Runkai Zheng, Jindong Wang, Cheuk Hang Leung, Qi Wu, Xing Xie

**Abstract**: Vision Transformers (ViTs) are increasingly used in computer vision due to their high performance, but their vulnerability to adversarial attacks is a concern. Existing methods lack a solid theoretical basis, focusing mainly on empirical training adjustments. This study introduces SpecFormer, tailored to fortify ViTs against adversarial attacks, with theoretical underpinnings. We establish local Lipschitz bounds for the self-attention layer and propose the Maximum Singular Value Penalization (MSVP) to precisely manage these bounds By incorporating MSVP into ViTs' attention layers, we enhance the model's robustness without compromising training efficiency. SpecFormer, the resulting model, outperforms other state-of-the-art models in defending against adversarial attacks, as proven by experiments on CIFAR and ImageNet datasets. Code is released at https://github.com/microsoft/robustlearn.

摘要: Vision Transformers（ViT）因其高性能而越来越多地应用于计算机视觉，但其对对抗攻击的脆弱性令人担忧。现有方法缺乏坚实的理论基础，主要侧重于经验性的训练调整。这项研究引入了SpecFormer，旨在增强ViT抵御对抗攻击，并提供了理论基础。我们为自我注意力层建立了局部Lipschitz界限，并提出最大奇异值惩罚（MSVP）来精确管理这些界限。通过将MSVP整合到ViT的注意力层中，我们增强了模型的鲁棒性，而不会影响训练效率。CIFAR和ImageNet数据集的实验证明，由此产生的模型SpecFormer在防御对抗攻击方面优于其他最先进的模型。代码发布于https://github.com/microsoft/robustlearn。



