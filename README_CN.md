# Latest Adversarial Attack Papers
**update at 2024-08-27 18:58:17**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Efficient Model-Stealing Attacks Against Inductive Graph Neural Networks**

针对归纳图神经网络的有效模型窃取攻击 cs.LG

Accepted at ECAI - 27TH EUROPEAN CONFERENCE ON ARTIFICIAL  INTELLIGENCE

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2405.12295v3) [paper-pdf](http://arxiv.org/pdf/2405.12295v3)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pregowska And Tomasz Michalak

**Abstract**: Graph Neural Networks (GNNs) are recognized as potent tools for processing real-world data organized in graph structures. Especially inductive GNNs, which allow for the processing of graph-structured data without relying on predefined graph structures, are becoming increasingly important in a wide range of applications. As such these networks become attractive targets for model-stealing attacks where an adversary seeks to replicate the functionality of the targeted network. Significant efforts have been devoted to developing model-stealing attacks that extract models trained on images and texts. However, little attention has been given to stealing GNNs trained on graph data. This paper identifies a new method of performing unsupervised model-stealing attacks against inductive GNNs, utilizing graph contrastive learning and spectral graph augmentations to efficiently extract information from the targeted model. The new type of attack is thoroughly evaluated on six datasets and the results show that our approach outperforms the current state-of-the-art by Shen et al. (2021). In particular, our attack surpasses the baseline across all benchmarks, attaining superior fidelity and downstream accuracy of the stolen model while necessitating fewer queries directed toward the target model.

摘要: 图神经网络(GNN)被认为是处理以图结构组织的真实世界数据的有力工具。尤其是允许在不依赖预定义的图结构的情况下处理图结构数据的感应式GNN，在广泛的应用中正变得越来越重要。因此，这些网络成为模型窃取攻击的有吸引力的目标，在这种攻击中，对手试图复制目标网络的功能。已经投入了大量的努力来开发窃取模型的攻击，提取针对图像和文本训练的模型。然而，对窃取针对图表数据训练的GNN的关注很少。本文提出了一种新的无监督窃取模型攻击方法，利用图对比学习和谱图扩充来有效地从目标模型中提取信息。在六个数据集上对新类型的攻击进行了全面的评估，结果表明，我们的方法的性能优于沈等人目前的最新技术。(2021年)。特别是，我们的攻击在所有基准测试中都超过了基线，获得了被盗模型的卓越保真度和下游准确性，同时需要更少的针对目标模型的查询。



## **2. Celtibero: Robust Layered Aggregation for Federated Learning**

Celtibero：用于联邦学习的稳健分层聚合 cs.CR

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14240v1) [paper-pdf](http://arxiv.org/pdf/2408.14240v1)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated Learning (FL) is an innovative approach to distributed machine learning. While FL offers significant privacy advantages, it also faces security challenges, particularly from poisoning attacks where adversaries deliberately manipulate local model updates to degrade model performance or introduce hidden backdoors. Existing defenses against these attacks have been shown to be effective when the data on the nodes is identically and independently distributed (i.i.d.), but they often fail under less restrictive, non-i.i.d data conditions. To overcome these limitations, we introduce Celtibero, a novel defense mechanism that integrates layered aggregation to enhance robustness against adversarial manipulation. Through extensive experiments on the MNIST and IMDB datasets, we demonstrate that Celtibero consistently achieves high main task accuracy (MTA) while maintaining minimal attack success rates (ASR) across a range of untargeted and targeted poisoning attacks. Our results highlight the superiority of Celtibero over existing defenses such as FL-Defender, LFighter, and FLAME, establishing it as a highly effective solution for securing federated learning systems against sophisticated poisoning attacks.

摘要: 联合学习(FL)是分布式机器学习的一种创新方法。虽然FL提供了显著的隐私优势，但它也面临着安全挑战，特别是来自毒化攻击的挑战，即攻击者故意操纵本地模型更新以降低模型性能或引入隐藏后门。当节点上的数据相同且独立分布(I.I.D.)时，现有的针对这些攻击的防御已被证明是有效的，但在限制较少的非I.I.D.数据条件下，它们通常会失败。为了克服这些局限性，我们引入了Celtibero，这是一种新的防御机制，它集成了分层聚合来增强对对手操纵的健壮性。通过在MNIST和IMDB数据集上的广泛实验，我们证明了Celtibero在一系列非目标和目标中毒攻击中始终实现了高的主任务准确率(MTA)，同时保持了最低的攻击成功率(ASR)。我们的结果突出了Celtibero相对于FL-Defender、LFighter和FAME等现有防御系统的优势，使其成为保护联邦学习系统免受复杂中毒攻击的高效解决方案。



## **3. SNNGX: Securing Spiking Neural Networks with Genetic XOR Encryption on RRAM-based Neuromorphic Accelerator**

SNNGX：在基于RAM的神经形态加速器上通过遗传异或加密保护尖峰神经网络 cs.CR

International Conference on Computer-Aided Design 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2407.15152v2) [paper-pdf](http://arxiv.org/pdf/2407.15152v2)

**Authors**: Kwunhang Wong, Songqi Wang, Wei Huang, Xinyuan Zhang, Yangu He, Karl M. H. Lai, Yuzhong Jiao, Ning Lin, Xiaojuan Qi, Xiaoming Chen, Zhongrui Wang

**Abstract**: Biologically plausible Spiking Neural Networks (SNNs), characterized by spike sparsity, are growing tremendous attention over intellectual edge devices and critical bio-medical applications as compared to artificial neural networks (ANNs). However, there is a considerable risk from malicious attempts to extract white-box information (i.e., weights) from SNNs, as attackers could exploit well-trained SNNs for profit and white-box adversarial concerns. There is a dire need for intellectual property (IP) protective measures. In this paper, we present a novel secure software-hardware co-designed RRAM-based neuromorphic accelerator for protecting the IP of SNNs. Software-wise, we design a tailored genetic algorithm with classic XOR encryption to target the least number of weights that need encryption. From a hardware perspective, we develop a low-energy decryption module, meticulously designed to provide zero decryption latency. Extensive results from various datasets, including NMNIST, DVSGesture, EEGMMIDB, Braille Letter, and SHD, demonstrate that our proposed method effectively secures SNNs by encrypting a minimal fraction of stealthy weights, only 0.00005% to 0.016% weight bits. Additionally, it achieves a substantial reduction in energy consumption, ranging from x59 to x6780, and significantly lowers decryption latency, ranging from x175 to x4250. Moreover, our method requires as little as one sample per class in dataset for encryption and addresses hessian/gradient-based search insensitive problems. This strategy offers a highly efficient and flexible solution for securing SNNs in diverse applications.

摘要: 与人工神经网络(ANN)相比，生物学上看似合理的尖峰神经网络(SNN)在智能边缘设备和关键的生物医学应用方面受到了极大的关注。然而，恶意尝试从SNN中提取白盒信息(即权重)存在相当大的风险，因为攻击者可以利用训练有素的SNN来获取利润和白盒对手的担忧。迫切需要知识产权(IP)保护措施。在本文中，我们提出了一种基于RRAM的安全软硬件联合设计的神经形态加速器来保护SNN的IP。在软件方面，我们设计了一种定制的遗传算法，采用经典的XOR加密，以达到需要加密的权重最少的目标。从硬件的角度，我们开发了一个低能耗的解密模块，精心设计，以提供零解密延迟。在NMNIST、DVSGesture、EEGMMIDB、盲文字母和SHD等不同的数据集上的广泛结果表明，我们提出的方法通过加密最小部分的隐蔽权重来有效地保护SNN，仅0.00005%到0.016%的权重比特。此外，它实现了从x59到x6780的能耗的大幅降低，并显著降低了从x175到x4250的解密延迟。此外，我们的方法只需要数据集中每个类一个样本进行加密，并解决了基于黑斯/梯度的搜索不敏感问题。该策略为保护不同应用中的SNN提供了一种高效、灵活的解决方案。



## **4. 2D-Malafide: Adversarial Attacks Against Face Deepfake Detection Systems**

2D-Malafide：针对面部Deepfake检测系统的对抗性攻击 cs.CV

Accepted at BIOSIG 2024

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.14143v1) [paper-pdf](http://arxiv.org/pdf/2408.14143v1)

**Authors**: Chiara Galdi, Michele Panariello, Massimiliano Todisco, Nicholas Evans

**Abstract**: We introduce 2D-Malafide, a novel and lightweight adversarial attack designed to deceive face deepfake detection systems. Building upon the concept of 1D convolutional perturbations explored in the speech domain, our method leverages 2D convolutional filters to craft perturbations which significantly degrade the performance of state-of-the-art face deepfake detectors. Unlike traditional additive noise approaches, 2D-Malafide optimises a small number of filter coefficients to generate robust adversarial perturbations which are transferable across different face images. Experiments, conducted using the FaceForensics++ dataset, demonstrate that 2D-Malafide substantially degrades detection performance in both white-box and black-box settings, with larger filter sizes having the greatest impact. Additionally, we report an explainability analysis using GradCAM which illustrates how 2D-Malafide misleads detection systems by altering the image areas used most for classification. Our findings highlight the vulnerability of current deepfake detection systems to convolutional adversarial attacks as well as the need for future work to enhance detection robustness through improved image fidelity constraints.

摘要: 本文介绍了2D-恶意攻击，这是一种新型的、轻量级的对抗性攻击，旨在欺骗人脸深度假冒检测系统。基于语音领域中探索的一维卷积扰动的概念，我们的方法利用二维卷积滤波器来制造扰动，从而显著降低最先进的人脸深度伪检测器的性能。与传统的加性噪声方法不同，2D-恶意优化了少量的滤波系数，以产生可在不同人脸图像之间传输的稳健的对抗性扰动。使用FaceForensics++数据集进行的实验表明，在白盒和黑盒设置下，2D-恶意都会显著降低检测性能，其中较大的过滤器尺寸影响最大。此外，我们报告了一个使用GradCAM的可解释性分析，它说明了2D-恶意如何通过改变最常用于分类的图像区域来误导检测系统。我们的发现突显了当前深度伪检测系统对卷积对抗攻击的脆弱性，以及未来通过改进图像保真度约束来增强检测稳健性的需要。



## **5. A Unified Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

基于部件感知能力的视觉自我监督编码器统一隶属度推理方法 cs.CV

Accepted by ACM CCS2024, Full version

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2404.02462v2) [paper-pdf](http://arxiv.org/pdf/2404.02462v2)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we aim to perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses with the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Our code is available at https://github.com/JiePKU/PartCrop.

摘要: 自我监督学习在利用大量未标记数据方面表现出了希望，但它也面临着重大的隐私问题，特别是在视觉方面。在本文中，我们的目标是在一种更现实的环境下对视觉自我监督模型进行隶属度推理：当对手攻击时，自我监督训练方法和细节是未知的，因为他在实践中通常面临一个黑箱系统。在这种情况下，考虑到自监督模型可以用完全不同的自监督范型来训练，例如蒙版图像建模和对比学习，训练细节复杂，我们提出了一种统一的隶属度推理方法PartCrop。它的动机是模型之间共享的部件感知能力和对训练数据的更强的部件响应。具体地说，PartCrop裁剪图像中对象的一部分，以在表示空间中使用图像查询响应。我们使用三个广泛使用的图像数据集对具有不同训练协议和结构的自监督模型进行了广泛的攻击。实验结果验证了PartCrop的有效性和泛化能力。此外，为了防御PartCrop，我们评估了两种常见的方法，即提前停止和区分隐私，并提出了一种称为缩小作物尺度范围的定制方法。防御实验表明，这些方法都是有效的。我们的代码可以在https://github.com/JiePKU/PartCrop.上找到



## **6. TF-Attack: Transferable and Fast Adversarial Attacks on Large Language Models**

TF攻击：对大型语言模型的可转移且快速对抗攻击 cs.CL

14 pages, 6 figures. arXiv admin note: text overlap with  arXiv:2305.17440 by other authors

**SubmitDate**: 2024-08-26    [abs](http://arxiv.org/abs/2408.13985v1) [paper-pdf](http://arxiv.org/pdf/2408.13985v1)

**Authors**: Zelin Li, Kehai Chen, Xuefeng Bai, Lemao Liu, Mingming Yang, Yang Xiang, Min Zhang

**Abstract**: With the great advancements in large language models (LLMs), adversarial attacks against LLMs have recently attracted increasing attention. We found that pre-existing adversarial attack methodologies exhibit limited transferability and are notably inefficient, particularly when applied to LLMs. In this paper, we analyze the core mechanisms of previous predominant adversarial attack methods, revealing that 1) the distributions of importance score differ markedly among victim models, restricting the transferability; 2) the sequential attack processes induces substantial time overheads. Based on the above two insights, we introduce a new scheme, named TF-Attack, for Transferable and Fast adversarial attacks on LLMs. TF-Attack employs an external LLM as a third-party overseer rather than the victim model to identify critical units within sentences. Moreover, TF-Attack introduces the concept of Importance Level, which allows for parallel substitutions of attacks. We conduct extensive experiments on 6 widely adopted benchmarks, evaluating the proposed method through both automatic and human metrics. Results show that our method consistently surpasses previous methods in transferability and delivers significant speed improvements, up to 20 times faster than earlier attack strategies.

摘要: 近年来，随着大型语言模型的发展，针对大型语言模型的对抗性攻击引起了越来越多的关注。我们发现，现有的对抗性攻击方法表现出有限的可转移性和显著的低效，特别是当应用于LLM时。本文分析了以往主流对抗性攻击方法的核心机制，发现1)不同受害者模型的重要性分数分布明显不同，限制了可转移性；2)顺序攻击过程导致了大量的时间开销。基于以上两点，我们提出了一种新的方案，称为TF-Attack，用于对LLMS进行可转移和快速对抗攻击。TF-Attack使用外部LLM作为第三方监督者，而不是受害者模型来识别判刑内的关键单元。此外，TF-Attack还引入了重要度的概念，允许并行替换攻击。我们在6个广泛采用的基准上进行了广泛的实验，从自动度量和人工度量两个方面对所提出的方法进行了评估。结果表明，我们的方法在可转移性上始终优于以前的方法，并提供了显著的速度改进，比以前的攻击策略快20倍。



## **7. RT-Attack: Jailbreaking Text-to-Image Models via Random Token**

RT攻击：通过随机令牌越狱文本到图像模型 cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13896v1) [paper-pdf](http://arxiv.org/pdf/2408.13896v1)

**Authors**: Sensen Gao, Xiaojun Jia, Yihao Huang, Ranjie Duan, Jindong Gu, Yang Liu, Qing Guo

**Abstract**: Recently, Text-to-Image(T2I) models have achieved remarkable success in image generation and editing, yet these models still have many potential issues, particularly in generating inappropriate or Not-Safe-For-Work(NSFW) content. Strengthening attacks and uncovering such vulnerabilities can advance the development of reliable and practical T2I models. Most of the previous works treat T2I models as white-box systems, using gradient optimization to generate adversarial prompts. However, accessing the model's gradient is often impossible in real-world scenarios. Moreover, existing defense methods, those using gradient masking, are designed to prevent attackers from obtaining accurate gradient information. While some black-box jailbreak attacks have been explored, these typically rely on simply replacing sensitive words, leading to suboptimal attack performance. To address this issue, we introduce a two-stage query-based black-box attack method utilizing random search. In the first stage, we establish a preliminary prompt by maximizing the semantic similarity between the adversarial and target harmful prompts. In the second stage, we use this initial prompt to refine our approach, creating a detailed adversarial prompt aimed at jailbreaking and maximizing the similarity in image features between the images generated from this prompt and those produced by the target harmful prompt. Extensive experiments validate the effectiveness of our method in attacking the latest prompt checkers, post-hoc image checkers, securely trained T2I models, and online commercial models.

摘要: 最近，文本到图像(T2I)模式在图像生成和编辑方面取得了显著的成功，但这些模式仍然存在许多潜在的问题，特别是在生成不适当或不安全的工作内容(NSFW)方面。加强攻击并发现此类漏洞可以促进可靠和实用的T2I模型的发展。以前的工作大多将T2I模型视为白盒系统，使用梯度优化来生成对抗性提示。然而，在现实世界的场景中，访问模型的渐变通常是不可能的。此外，现有的防御方法，即使用梯度掩码的方法，旨在防止攻击者获得准确的梯度信息。虽然已经探索了一些黑盒越狱攻击，但这些攻击通常依赖于简单地替换敏感词，导致攻击性能不佳。为了解决这个问题，我们提出了一种利用随机搜索的两阶段查询黑盒攻击方法。在第一阶段，我们通过最大化敌意提示和目标有害提示之间的语义相似度来建立初步提示。在第二阶段，我们使用这个初始提示来改进我们的方法，创建一个详细的对抗性提示，旨在越狱，并最大化由该提示生成的图像与目标有害提示生成的图像特征之间的相似性。大量的实验验证了该方法在攻击最新的提示检查器、后自组织图像检查器、安全训练的T2I模型和在线商业模型方面的有效性。



## **8. Hiding Backdoors within Event Sequence Data via Poisoning Attacks**

通过中毒攻击在事件序列数据中隐藏后门 cs.LG

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2308.10201v2) [paper-pdf](http://arxiv.org/pdf/2308.10201v2)

**Authors**: Alina Ermilova, Elizaveta Kovtun, Dmitry Berestnev, Alexey Zaytsev

**Abstract**: The financial industry relies on deep learning models for making important decisions. This adoption brings new danger, as deep black-box models are known to be vulnerable to adversarial attacks. In computer vision, one can shape the output during inference by performing an adversarial attack called poisoning via introducing a backdoor into the model during training. For sequences of financial transactions of a customer, insertion of a backdoor is harder to perform, as models operate over a more complex discrete space of sequences, and systematic checks for insecurities occur. We provide a method to introduce concealed backdoors, creating vulnerabilities without altering their functionality for uncontaminated data. To achieve this, we replace a clean model with a poisoned one that is aware of the availability of a backdoor and utilize this knowledge. Our most difficult for uncovering attacks include either additional supervised detection step of poisoned data activated during the test or well-hidden model weight modifications. The experimental study provides insights into how these effects vary across different datasets, architectures, and model components. Alternative methods and baselines, such as distillation-type regularization, are also explored but found to be less efficient. Conducted on three open transaction datasets and architectures, including LSTM, CNN, and Transformer, our findings not only illuminate the vulnerabilities in contemporary models but also can drive the construction of more robust systems.

摘要: 金融业依靠深度学习模型来做出重要决策。这种采用带来了新的危险，因为众所周知，深黑盒模型很容易受到对手的攻击。在计算机视觉中，人们可以在推理过程中通过在训练期间将后门引入模型来执行称为中毒的对抗性攻击来塑造输出。对于客户的金融交易序列，插入后门更难执行，因为模型在更复杂的离散序列空间上操作，并对不安全性进行系统检查。我们提供了一种方法来引入隐藏的后门，在不改变其针对未受污染数据的功能的情况下创建漏洞。为了实现这一点，我们用知道后门可用的有毒模型替换干净的模型，并利用这一知识。我们最难发现的攻击包括在测试期间激活的有毒数据的额外监督检测步骤，或者隐藏良好的模型权重修改。该实验研究深入了解了这些影响在不同数据集、体系结构和模型组件中的差异。还探索了其他方法和基线，如蒸馏型正则化，但发现效率较低。我们在包括LSTM、CNN和Transformer在内的三个开放事务数据集和体系结构上进行了实验，我们的发现不仅揭示了当代模型中的漏洞，而且可以推动构建更健壮的系统。



## **9. Sample-Independent Federated Learning Backdoor Attack**

样本独立联邦学习后门攻击 cs.CR

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13849v1) [paper-pdf](http://arxiv.org/pdf/2408.13849v1)

**Authors**: Weida Xu, Yang Xu, Sicong Zhang

**Abstract**: In federated learning, backdoor attacks embed triggers in the adversarial client's data to inject a backdoor into the model. To evade detection through sample analysis, non-sample-modifying backdoor attack methods based on dropout have been developed. However, these methods struggle to covertly utilize dropout in evaluation mode, thus hindering their deployment in real-world scenarios. To address these, this paper introduces GhostB, a novel approach to federated learning backdoor attacks that neither alters samples nor relies on dropout. This method employs the behavior of neurons producing specific values as triggers. By mapping these neuronal values to categories specified by the adversary, the backdoor is implanted and activated when particular feature values are detected at designated neurons. Our experiments conducted on TIMIT, LibriSpeech, and VoxCeleb2 databases in both Closed Set Identification (CSI) and Open Set Identification (OSI) scenarios demonstrate that GhostB achieves a 100% success rate upon activation, with this rate maintained across experiments involving 1 to 50 ghost neurons. This paper investigates how the dispersion of neurons and their depth within hidden layers affect the success rate, revealing that increased dispersion and positioning of neurons can significantly decrease effectiveness, potentially rendering the attack unsuccessful.

摘要: 在联合学习中，后门攻击在敌对客户端的数据中嵌入触发器，向模型中注入后门。为了通过样本分析逃避检测，提出了基于丢弃的非样本修改后门攻击方法。然而，这些方法很难在评估模式中隐蔽地利用丢弃，从而阻碍了它们在现实世界场景中的部署。为了解决这些问题，本文引入了一种新的联合学习后门攻击方法Ghost B，它既不改变样本，也不依赖于辍学。这种方法使用产生特定值的神经元的行为作为触发器。通过将这些神经元值映射到对手指定的类别，当在指定的神经元处检测到特定特征值时，后门被植入并激活。我们在TIMIT、LibriSpeech和VoxCeleb2数据库上进行的封闭集识别(CSI)和开放集识别(OSI)场景下的实验表明，Ghost B在激活时达到了100%的成功率，并且在涉及1到50个鬼神经元的实验中保持了这一成功率。本文研究了神经元的离散度及其在隐层中的深度对攻击成功率的影响，揭示了神经元离散度的增加和位置的增加会显著降低攻击的有效性，从而潜在地使攻击失败。



## **10. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

论科尔莫戈洛夫-阿诺德网络的鲁棒性：对抗的视角 cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13809v1) [paper-pdf](http://arxiv.org/pdf/2408.13809v1)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper, we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white-box adversarial attacks, comparing their resilience to that of established neural network architectures. Further, we investigate the transferability of adversarial examples between KANs and Multilayer Perceptron (MLPs), deriving critical insights into the unique vulnerabilities of KANs. Our experiments use the MNIST, FashionMNIST, and KMNIST datasets, providing a comprehensive evaluation of KANs in adversarial scenarios. This work offers the first in-depth analysis of security in KANs, laying the groundwork for future research in this emerging field.

摘要: Kolmogorov-Arnold网络(KANS)是最近出现的一种新的函数逼近方法，在各个领域显示出巨大的潜力。尽管它们在理论上有希望，但KANS在对抗条件下的健壮性尚未得到彻底的检验。在这篇文章中，我们探讨了KANS的对抗稳健性，特别关注图像分类任务。我们评估了KANS对抗标准白盒攻击的性能，比较了它们与已建立的神经网络结构的弹性。进一步，我们研究了KANS和多层感知器(MLP)之间的对抗性例子的可转移性，得出了对KANS独特的脆弱性的关键见解。我们的实验使用MNIST、FashionMNIST和KMNIST数据集，提供了对对抗性场景中的KAN的全面评估。这项工作提供了第一次对KANS安全的深入分析，为这一新兴领域的未来研究奠定了基础。



## **11. CAMH: Advancing Model Hijacking Attack in Machine Learning**

CAMH：机器学习中推进模型劫持攻击 cs.CR

9 pages

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2408.13741v1) [paper-pdf](http://arxiv.org/pdf/2408.13741v1)

**Authors**: Xing He, Jiahao Chen, Yuwen Pu, Qingming Li, Chunyi Zhou, Yingcai Wu, Jinbao Li, Shouling Ji

**Abstract**: In the burgeoning domain of machine learning, the reliance on third-party services for model training and the adoption of pre-trained models have surged. However, this reliance introduces vulnerabilities to model hijacking attacks, where adversaries manipulate models to perform unintended tasks, leading to significant security and ethical concerns, like turning an ordinary image classifier into a tool for detecting faces in pornographic content, all without the model owner's knowledge. This paper introduces Category-Agnostic Model Hijacking (CAMH), a novel model hijacking attack method capable of addressing the challenges of class number mismatch, data distribution divergence, and performance balance between the original and hijacking tasks. CAMH incorporates synchronized training layers, random noise optimization, and a dual-loop optimization approach to ensure minimal impact on the original task's performance while effectively executing the hijacking task. We evaluate CAMH across multiple benchmark datasets and network architectures, demonstrating its potent attack effectiveness while ensuring minimal degradation in the performance of the original task.

摘要: 在蓬勃发展的机器学习领域，对第三方服务进行模型培训的依赖和对预先训练模型的采用激增。然而，这种依赖引入了漏洞来模拟劫持攻击，即攻击者操纵模型执行意想不到的任务，导致重大的安全和伦理问题，比如将普通图像分类器变成在色情内容中检测人脸的工具，所有这些都是在模型所有者不知道的情况下进行的。介绍了一种新的模型劫持攻击方法--类别不可知模型劫持(CAMH)，它能够解决类别号不匹配、数据分布差异以及原始任务和劫持任务之间的性能平衡等问题。CAMH结合了同步训练层、随机噪声优化和双环优化方法，以确保在有效执行劫持任务的同时对原始任务的性能影响最小。我们在多个基准数据集和网络体系结构上对CAMH进行了评估，展示了其强大的攻击效率，同时确保原始任务的性能降级最小。



## **12. Attack on Scene Flow using Point Clouds**

使用点云攻击场景流 cs.CV

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2404.13621v4) [paper-pdf](http://arxiv.org/pdf/2404.13621v4)

**Authors**: Haniyeh Ehsani Oskouie, Mohammad-Shahram Moin, Shohreh Kasaei

**Abstract**: Deep neural networks have made significant advancements in accurately estimating scene flow using point clouds, which is vital for many applications like video analysis, action recognition, and navigation. The robustness of these techniques, however, remains a concern, particularly in the face of adversarial attacks that have been proven to deceive state-of-the-art deep neural networks in many domains. Surprisingly, the robustness of scene flow networks against such attacks has not been thoroughly investigated. To address this problem, the proposed approach aims to bridge this gap by introducing adversarial white-box attacks specifically tailored for scene flow networks. Experimental results show that the generated adversarial examples obtain up to 33.7 relative degradation in average end-point error on the KITTI and FlyingThings3D datasets. The study also reveals the significant impact that attacks targeting point clouds in only one dimension or color channel have on average end-point error. Analyzing the success and failure of these attacks on the scene flow networks and their 2D optical flow network variants shows a higher vulnerability for the optical flow networks. Code is available at https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.

摘要: 深度神经网络在利用点云准确估计场景流量方面取得了重大进展，这对于视频分析、动作识别和导航等许多应用都是至关重要的。然而，这些技术的健壮性仍然是一个令人担忧的问题，特别是在面对已被证明在许多领域欺骗最先进的深度神经网络的对抗性攻击时。令人惊讶的是，场景流网络对此类攻击的健壮性还没有得到彻底的研究。为了解决这个问题，提出的方法旨在通过引入专门为场景流网络量身定做的对抗性白盒攻击来弥合这一差距。实验结果表明，生成的对抗性实例在Kitti和FlyingThings3D数据集上的平均端点误差相对下降高达33.7。研究还揭示了仅以一维或颜色通道中的点云为目标的攻击对平均端点误差的显著影响。分析这些攻击对场景流网络及其二维光流网络变体的成功和失败，表明光流网络具有更高的脆弱性。代码可在https://github.com/aheldis/Attack-on-Scene-Flow-using-Point-Clouds.git.上找到



## **13. Interpretable and Robust AI in EEG Systems: A Survey**

脑电系统中可解释且稳健的人工智能：调查 eess.SP

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2304.10755v3) [paper-pdf](http://arxiv.org/pdf/2304.10755v3)

**Authors**: Xinliang Zhou, Chenyu Liu, Zhongruo Wang, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.

摘要: 人工智能(AI)和脑电(EEG)的紧密结合在AI时代极大地推动了人机交互(HCI)技术的进步。与传统的脑电系统不同，基于人工智能的脑电系统的可解释性和稳健性变得尤为关键。这种可解释性阐明了人工智能模型的内部工作机制，从而可以赢得用户的信任。健壮性反映了人工智能对攻击和扰动的可靠性，这对于敏感和脆弱的脑电信号是必不可少的。因此，人工智能在脑电系统中的可解释性和稳健性受到越来越多的关注，近年来其研究取得了很大进展。然而，目前还没有关于这一领域最新进展的调查。在本文中，我们介绍了第一个全面的综述，并总结了可解释的和健壮的脑电系统人工智能技术。具体地说，我们首先提出了一种可解释性的分类，将其描述为三种类型：反向传播方法、扰动方法和内在可解释方法。然后，我们将健壮性机制分为四类：噪声和伪影、人类可变性、数据获取不稳定性和对抗性攻击。最后，我们确定了脑电系统中可解释的和健壮的人工智能面临的几个关键和尚未解决的挑战，并进一步讨论了它们的未来发展方向。



## **14. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

无处不在的捷径：探索多触发后门攻击 cs.LG

**SubmitDate**: 2024-08-25    [abs](http://arxiv.org/abs/2401.15295v2) [paper-pdf](http://arxiv.org/pdf/2401.15295v2)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs.

摘要: 后门攻击已经成为深度神经网络(DNN)预训练和部署的重大威胁。虽然已经提出了许多检测和减轻后门攻击的方法，但大多数都依赖于识别和消除后门创建的将特定源类链接到目标类的“捷径”。然而，通过设计多个后门触发器，可以很容易地绕过这些方法，这些后门触发器在任何地方都创建快捷方式，因此没有特定的捷径。在这项研究中，我们探索了多触发器后门攻击(MTBA)的概念，即多个对手利用不同类型的触发器来毒化同一数据集。通过提出和研究三种类型的多触发器攻击，包括并行式、顺序式和混合式，我们证明了1)多个触发器可以共存、覆盖或交叉激活彼此；2)MTBA很容易打破大多数现有后门检测/删除方法的普遍捷径假设，导致它们无效。考虑到MTBA带来的安全风险，我们创建了一个多触发后门中毒数据集，以便于未来检测和缓解这些攻击的研究，并讨论了针对MTBA的潜在防御策略。



## **15. Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures**

通过稀疏编码架构提高倒置攻击建模的稳健性 cs.CV

ECCV 2024

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2403.14772v2) [paper-pdf](http://arxiv.org/pdf/2403.14772v2)

**Authors**: Sayanton V. Dibbo, Adam Breuer, Juston Moore, Michael Teti

**Abstract**: Recent model inversion attack algorithms permit adversaries to reconstruct a neural network's private and potentially sensitive training data by repeatedly querying the network. In this work, we develop a novel network architecture that leverages sparse-coding layers to obtain superior robustness to this class of attacks. Three decades of computer science research has studied sparse coding in the context of image denoising, object recognition, and adversarial misclassification settings, but to the best of our knowledge, its connection to state-of-the-art privacy vulnerabilities remains unstudied. In this work, we hypothesize that sparse coding architectures suggest an advantageous means to defend against model inversion attacks because they allow us to control the amount of irrelevant private information encoded by a network in a manner that is known to have little effect on classification accuracy. Specifically, compared to networks trained with a variety of state-of-the-art defenses, our sparse-coding architectures maintain comparable or higher classification accuracy while degrading state-of-the-art training data reconstructions by factors of 1.1 to 18.3 across a variety of reconstruction quality metrics (PSNR, SSIM, FID). This performance advantage holds across 5 datasets ranging from CelebA faces to medical images and CIFAR-10, and across various state-of-the-art SGD-based and GAN-based inversion attacks, including Plug-&-Play attacks. We provide a cluster-ready PyTorch codebase to promote research and standardize defense evaluations.

摘要: 最近的模型反转攻击算法允许攻击者通过重复查询网络来重建神经网络的私有和潜在敏感的训练数据。在这项工作中，我们开发了一种新颖的网络体系结构，该体系结构利用稀疏编码层来获得对此类攻击的卓越健壮性。三十年的计算机科学研究已经在图像去噪、目标识别和敌意错误分类环境中研究了稀疏编码，但就我们所知，它与最先进的隐私漏洞的联系仍未被研究。在这项工作中，我们假设稀疏编码体系结构是抵御模型反转攻击的一种有利手段，因为它们允许我们以一种已知对分类精度几乎没有影响的方式来控制网络编码的无关私人信息的数量。具体地说，与使用各种最先进的防御措施训练的网络相比，我们的稀疏编码体系结构保持了相当或更高的分类精度，同时在各种重建质量指标(PSNR、SSIM、FID)上将最先进的训练数据重建降级1.1至18.3倍。这一性能优势涵盖从CelebA Faces到医学图像和CIFAR-10的5个数据集，以及各种最先进的基于SGD和GAN的反转攻击，包括即插即用攻击。我们提供了一个支持集群的PyTorch代码库，以促进研究和标准化防御评估。



## **16. Detecting Adversarial Data via Perturbation Forgery**

通过微扰伪造检测对抗数据 cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.16226v2) [paper-pdf](http://arxiv.org/pdf/2405.16226v2)

**Authors**: Qian Wang, Chen Li, Yuchen Luo, Hefei Ling, Ping Li, Jiazhong Chen, Shijuan Huang, Ning Yu

**Abstract**: As a defense strategy against adversarial attacks, adversarial detection aims to identify and filter out adversarial data from the data flow based on discrepancies in distribution and noise patterns between natural and adversarial data. Although previous detection methods achieve high performance in detecting gradient-based adversarial attacks, new attacks based on generative models with imbalanced and anisotropic noise patterns evade detection. Even worse, existing techniques either necessitate access to attack data before deploying a defense or incur a significant time cost for inference, rendering them impractical for defending against newly emerging attacks that are unseen by defenders. In this paper, we explore the proximity relationship between adversarial noise distributions and demonstrate the existence of an open covering for them. By learning to distinguish this open covering from the distribution of natural data, we can develop a detector with strong generalization capabilities against all types of adversarial attacks. Based on this insight, we heuristically propose Perturbation Forgery, which includes noise distribution perturbation, sparse mask generation, and pseudo-adversarial data production, to train an adversarial detector capable of detecting unseen gradient-based, generative-model-based, and physical adversarial attacks, while remaining agnostic to any specific models. Comprehensive experiments conducted on multiple general and facial datasets, with a wide spectrum of attacks, validate the strong generalization of our method.

摘要: 敌意检测是针对敌意攻击的一种防御策略，其目的是根据自然数据和敌意数据之间的分布和噪声模式的差异，从数据流中识别和过滤敌意数据。虽然以前的检测方法在检测基于梯度的敌意攻击方面取得了较高的性能，但基于非平衡和各向异性噪声模式的生成模型的新攻击可以逃避检测。更糟糕的是，现有技术要么需要在部署防御之前访问攻击数据，要么需要花费大量时间进行推断，这使得它们在防御防御者看不到的新出现的攻击方面不切实际。本文研究了对抗性噪声分布之间的邻近关系，并证明了它们的开覆盖的存在性。通过学习将这种开放覆盖与自然数据的分布区分开来，我们可以开发出一个具有强大的泛化能力的检测器，以抵御所有类型的对抗性攻击。基于这一观点，我们启发式地提出了扰动伪造，它包括噪声分布扰动、稀疏掩码生成和伪对抗数据生成，以训练一个能够检测基于不可见的基于梯度、基于生成模型的和物理的对抗攻击的对抗检测器，同时保持对任何特定模型的不可知性。在多个普通数据集和人脸数据集上进行的综合实验表明，该方法具有较强的泛化能力。



## **17. Continual Adversarial Defense**

持续对抗防御 cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2312.09481v3) [paper-pdf](http://arxiv.org/pdf/2312.09481v3)

**Authors**: Qian Wang, Yaoyao Liu, Hefei Ling, Yingwei Li, Qihao Liu, Ping Li, Jiazhong Chen, Alan Yuille, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. A well-matched approach to the dynamic environment lies in a defense system that continuously collects adversarial data online to quickly improve itself. Therefore, we put forward a practical defense deployment against a challenging threat model and propose, for the first time, the Continual Adversarial Defense (CAD) framework that adapts to attack sequences under four principles: (1) continual adaptation to new attacks without catastrophic forgetting, (2) few-shot adaptation, (3) memory-efficient adaptation, and (4) high accuracy on both clean and adversarial data. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Extensive experiments validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal budget and a low cost of defense failure while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

摘要: 为了应对每月针对视觉分类器的对抗性攻击迅速演变的性质，提出了许多防御措施，以概括尽可能多的已知攻击。然而，设计一种概括所有类型攻击的防御方法是不现实的，因为防御系统运行的环境是动态的，包括随着时间的推移而出现的各种独特的攻击。一种与动态环境相匹配的方法在于一个防御系统，该系统不断在线收集敌对数据，以快速改进自己。因此，我们提出了一种针对具有挑战性的威胁模型的实用防御部署，并首次提出了适应攻击序列的持续对抗防御(CAD)框架，该框架遵循以下四个原则：(1)持续适应新攻击而不发生灾难性遗忘；(2)少射自适应；(3)高效内存自适应；(4)对干净和敌对数据的高准确率。我们探索并集成了尖端的持续学习、少机会学习和集成学习技术来验证这些原则。广泛的实验验证了我们的方法对现代对抗性攻击的多个阶段的有效性，并证明了与许多基线方法相比有显著的改进。特别是，CAD能够以最小的预算和较低的防御失败成本快速适应，同时保持对先前攻击的良好性能。我们的研究揭示了一种针对动态和不断变化的攻击进行持续防御适应的全新范式。



## **18. Safeguarding Vision-Language Models Against Patched Visual Prompt Injectors**

保护视觉语言模型免受修补视觉提示注入器的影响 cs.CV

15 pages

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2405.10529v2) [paper-pdf](http://arxiv.org/pdf/2405.10529v2)

**Authors**: Jiachen Sun, Changsheng Wang, Jiongxiao Wang, Yiwei Zhang, Chaowei Xiao

**Abstract**: Large language models have become increasingly prominent, also signaling a shift towards multimodality as the next frontier in artificial intelligence, where their embeddings are harnessed as prompts to generate textual content. Vision-language models (VLMs) stand at the forefront of this advancement, offering innovative ways to combine visual and textual data for enhanced understanding and interaction. However, this integration also enlarges the attack surface. Patch-based adversarial attack is considered the most realistic threat model in physical vision applications, as demonstrated in many existing literature. In this paper, we propose to address patched visual prompt injection, where adversaries exploit adversarial patches to generate target content in VLMs. Our investigation reveals that patched adversarial prompts exhibit sensitivity to pixel-wise randomization, a trait that remains robust even against adaptive attacks designed to counteract such defenses. Leveraging this insight, we introduce SmoothVLM, a defense mechanism rooted in smoothing techniques, specifically tailored to protect VLMs from the threat of patched visual prompt injectors. Our framework significantly lowers the attack success rate to a range between 0% and 5.0% on two leading VLMs, while achieving around 67.3% to 95.0% context recovery of the benign images, demonstrating a balance between security and usability.

摘要: 大型语言模型已变得越来越突出，这也标志着向多通道的转变，成为人工智能的下一个前沿，在人工智能中，它们的嵌入被用作生成文本内容的提示。视觉语言模型(VLM)站在这一进步的前沿，提供了将视觉和文本数据相结合的创新方法，以增强理解和交互。然而，这种整合也扩大了攻击面。基于补丁的对抗性攻击被认为是物理视觉应用中最现实的威胁模型，许多现有的文献都证明了这一点。在本文中，我们建议解决补丁视觉提示注入，即攻击者利用敌意补丁来生成VLMS中的目标内容。我们的调查显示，打补丁的对抗性提示显示出对像素随机化的敏感性，这一特征即使在旨在对抗此类防御的适应性攻击中也保持健壮。利用这一见解，我们推出了SmoothVLM，这是一种植根于平滑技术的防御机制，专门为保护VLM免受修补的视觉提示注入器的威胁而量身定做。我们的框架将攻击成功率显著降低到了0%到5.0%之间，同时实现了良性映像的67.3%到95.0%的上下文恢复，展示了安全性和可用性之间的平衡。



## **19. Probing the Robustness of Vision-Language Pretrained Models: A Multimodal Adversarial Attack Approach**

探索视觉语言预训练模型的鲁棒性：多模式对抗攻击方法 cs.CV

**SubmitDate**: 2024-08-24    [abs](http://arxiv.org/abs/2408.13461v1) [paper-pdf](http://arxiv.org/pdf/2408.13461v1)

**Authors**: Jiwei Guan, Tianyu Ding, Longbing Cao, Lei Pan, Chen Wang, Xi Zheng

**Abstract**: Vision-language pretraining (VLP) with transformers has demonstrated exceptional performance across numerous multimodal tasks. However, the adversarial robustness of these models has not been thoroughly investigated. Existing multimodal attack methods have largely overlooked cross-modal interactions between visual and textual modalities, particularly in the context of cross-attention mechanisms. In this paper, we study the adversarial vulnerability of recent VLP transformers and design a novel Joint Multimodal Transformer Feature Attack (JMTFA) that concurrently introduces adversarial perturbations in both visual and textual modalities under white-box settings. JMTFA strategically targets attention relevance scores to disrupt important features within each modality, generating adversarial samples by fusing perturbations and leading to erroneous model predictions. Experimental results indicate that the proposed approach achieves high attack success rates on vision-language understanding and reasoning downstream tasks compared to existing baselines. Notably, our findings reveal that the textual modality significantly influences the complex fusion processes within VLP transformers. Moreover, we observe no apparent relationship between model size and adversarial robustness under our proposed attacks. These insights emphasize a new dimension of adversarial robustness and underscore potential risks in the reliable deployment of multimodal AI systems.

摘要: 使用变压器的视觉语言预培训(VLP)在许多多模式任务中表现出了出色的性能。然而，这些模型的对抗稳健性还没有得到彻底的研究。现有的多通道攻击方法在很大程度上忽略了视觉通道和文本通道之间的跨通道交互作用，特别是在交叉注意机制的背景下。本文研究了现有VLP变换的对抗性漏洞，设计了一种在白盒环境下同时引入对抗性扰动的联合多模式变换特征攻击(JMTFA)。JMTFA战略性地将注意力相关性分数作为目标，以扰乱每个通道中的重要特征，通过融合扰动生成对抗性样本，并导致错误的模型预测。实验结果表明，与现有的基线相比，该方法在视觉语言理解和推理的下游任务上获得了更高的攻击成功率。值得注意的是，我们的研究结果显示，语篇情态显著影响VLP转换器内复杂的融合过程。此外，在我们提出的攻击下，我们没有观察到模型大小和对手稳健性之间的明显关系。这些见解强调了对抗性稳健性的一个新维度，并强调了可靠部署多模式人工智能系统的潜在风险。



## **20. Toward Improving Synthetic Audio Spoofing Detection Robustness via Meta-Learning and Disentangled Training With Adversarial Examples**

通过元学习和具有对抗性示例的分解训练来提高合成音频欺骗检测的鲁棒性 cs.SD

IEEE ACCESS 2024

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.13341v1) [paper-pdf](http://arxiv.org/pdf/2408.13341v1)

**Authors**: Zhenyu Wang, John H. L. Hansen

**Abstract**: Advances in automatic speaker verification (ASV) promote research into the formulation of spoofing detection systems for real-world applications. The performance of ASV systems can be degraded severely by multiple types of spoofing attacks, namely, synthetic speech (SS), voice conversion (VC), replay, twins and impersonation, especially in the case of unseen synthetic spoofing attacks. A reliable and robust spoofing detection system can act as a security gate to filter out spoofing attacks instead of having them reach the ASV system. A weighted additive angular margin loss is proposed to address the data imbalance issue, and different margins has been assigned to improve generalization to unseen spoofing attacks in this study. Meanwhile, we incorporate a meta-learning loss function to optimize differences between the embeddings of support versus query set in order to learn a spoofing-category-independent embedding space for utterances. Furthermore, we craft adversarial examples by adding imperceptible perturbations to spoofing speech as a data augmentation strategy, then we use an auxiliary batch normalization (BN) to guarantee that corresponding normalization statistics are performed exclusively on the adversarial examples. Additionally, A simple attention module is integrated into the residual block to refine the feature extraction process. Evaluation results on the Logical Access (LA) track of the ASVspoof 2019 corpus provides confirmation of our proposed approaches' effectiveness in terms of a pooled EER of 0.87%, and a min t-DCF of 0.0277. These advancements offer effective options to reduce the impact of spoofing attacks on voice recognition/authentication systems.

摘要: 自动说话人验证(ASV)的进步促进了针对现实应用的欺骗检测系统的制定的研究。ASV系统的性能会受到多种类型的欺骗攻击，即合成语音(SS)、语音转换(VC)、重放、双胞胎和模仿，特别是在不可见的合成欺骗攻击的情况下。可靠和强大的欺骗检测系统可以充当安全门，过滤掉欺骗攻击，而不是让它们到达ASV系统。为了解决数据不平衡问题，提出了一种加权的加性角度余量损失，并分配了不同的余量来提高对不可见欺骗攻击的泛化能力。同时，我们引入了元学习损失函数来优化支持集和查询集嵌入的差异，从而学习到一个与欺骗类别无关的话语嵌入空间。此外，我们通过在欺骗语音中添加不可察觉的扰动来构造对抗性实例作为数据扩充策略，然后使用辅助批量归一化(BN)来保证相应的归一化统计只针对对抗性实例执行。此外，在残差块中集成了一个简单的注意模块来改进特征提取过程。对ASVspoof2019年语料库的逻辑访问(LA)轨道的评估结果证实了我们提出的方法的有效性，池化EER为0.87%，最小t-DCF为0.0277。这些改进提供了有效的选项来减少欺骗攻击对语音识别/身份验证系统的影响。



## **21. Dynamic Label Adversarial Training for Deep Learning Robustness Against Adversarial Attacks**

动态标签对抗训练，实现深度学习对抗攻击的鲁棒性 cs.LG

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.13102v1) [paper-pdf](http://arxiv.org/pdf/2408.13102v1)

**Authors**: Zhenyu Liu, Haoran Duan, Huizhi Liang, Yang Long, Vaclav Snasel, Guiseppe Nicosia, Rajiv Ranjan, Varun Ojha

**Abstract**: Adversarial training is one of the most effective methods for enhancing model robustness. Recent approaches incorporate adversarial distillation in adversarial training architectures. However, we notice two scenarios of defense methods that limit their performance: (1) Previous methods primarily use static ground truth for adversarial training, but this often causes robust overfitting; (2) The loss functions are either Mean Squared Error or KL-divergence leading to a sub-optimal performance on clean accuracy. To solve those problems, we propose a dynamic label adversarial training (DYNAT) algorithm that enables the target model to gradually and dynamically gain robustness from the guide model's decisions. Additionally, we found that a budgeted dimension of inner optimization for the target model may contribute to the trade-off between clean accuracy and robust accuracy. Therefore, we propose a novel inner optimization method to be incorporated into the adversarial training. This will enable the target model to adaptively search for adversarial examples based on dynamic labels from the guiding model, contributing to the robustness of the target model. Extensive experiments validate the superior performance of our approach.

摘要: 对抗性训练是提高模型稳健性的最有效方法之一。最近的方法将对抗性蒸馏纳入对抗性训练体系中。然而，我们注意到有两种情况限制了防御方法的性能：(1)以前的方法主要使用静态地面真实进行对抗性训练，但这往往会导致稳健的过拟合；(2)损失函数要么是均方误差，要么是KL-发散，导致在干净精度上性能次优。为了解决这些问题，我们提出了一种动态标签对抗训练(DYNAT)算法，使目标模型能够从引导模型的决策中逐步动态地获得健壮性。此外，我们发现，目标模型的内部优化预算维度可能有助于在干净精度和稳健精度之间进行权衡。因此，我们提出了一种新的内点优化方法，并将其引入到对抗性训练中。这将使目标模型能够基于来自引导模型的动态标签自适应地搜索对抗性示例，从而有助于目标模型的健壮性。大量实验验证了该方法的优越性能。



## **22. Robust Diffusion Models for Adversarial Purification**

对抗性净化的鲁棒扩散模型 cs.CV

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2403.16067v3) [paper-pdf](http://arxiv.org/pdf/2403.16067v3)

**Authors**: Guang Lin, Zerui Tao, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Diffusion models (DMs) based adversarial purification (AP) has shown to be the most powerful alternative to adversarial training (AT). However, these methods neglect the fact that pre-trained diffusion models themselves are not robust to adversarial attacks as well. Additionally, the diffusion process can easily destroy semantic information and generate a high quality image but totally different from the original input image after the reverse process, leading to degraded standard accuracy. To overcome these issues, a natural idea is to harness adversarial training strategy to retrain or fine-tune the pre-trained diffusion model, which is computationally prohibitive. We propose a novel robust reverse process with adversarial guidance, which is independent of given pre-trained DMs and avoids retraining or fine-tuning the DMs. This robust guidance can not only ensure to generate purified examples retaining more semantic content but also mitigate the accuracy-robustness trade-off of DMs for the first time, which also provides DM-based AP an efficient adaptive ability to new attacks. Extensive experiments are conducted on CIFAR-10, CIFAR-100 and ImageNet to demonstrate that our method achieves the state-of-the-art results and exhibits generalization against different attacks.

摘要: 基于扩散模型(DM)的对抗净化(AP)已被证明是对抗训练(AT)最有效的替代方法。然而，这些方法忽略了这样一个事实，即预先训练的扩散模型本身对对手攻击也不是很健壮。此外，扩散过程容易破坏语义信息，生成高质量的图像，但反向处理后的图像与原始输入图像完全不同，导致标准精度下降。为了克服这些问题，一个自然的想法是利用对抗性训练策略来重新训练或微调预先训练的扩散模型，这在计算上是令人望而却步的。我们提出了一种新的具有对抗性指导的稳健逆向过程，它独立于给定的预先训练的DM，并且避免了对DM的重新训练或微调。这种健壮的指导不仅可以确保生成保持更多语义内容的纯化实例，还可以第一次缓解DM的准确性和健壮性之间的权衡，这也为基于DM的AP提供了对新攻击的有效适应能力。在CIFAR-10、CIFAR-100和ImageNet上的大量实验表明，该方法达到了最先进的结果，并对不同的攻击表现出了泛化能力。



## **23. Robust Feature Inference: A Test-time Defense Strategy using Spectral Projections**

鲁棒特征推断：使用谱投影的测试时防御策略 cs.LG

Published in TMLR (28 pages, 6 figures, 20 tables)

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2307.11672v2) [paper-pdf](http://arxiv.org/pdf/2307.11672v2)

**Authors**: Anurag Singh, Mahalakshmi Sabanayagam, Krikamol Muandet, Debarghya Ghoshdastidar

**Abstract**: Test-time defenses are used to improve the robustness of deep neural networks to adversarial examples during inference. However, existing methods either require an additional trained classifier to detect and correct the adversarial samples, or perform additional complex optimization on the model parameters or the input to adapt to the adversarial samples at test-time, resulting in a significant increase in the inference time compared to the base model. In this work, we propose a novel test-time defense strategy called Robust Feature Inference (RFI) that is easy to integrate with any existing (robust) training procedure without additional test-time computation. Based on the notion of robustness of features that we present, the key idea is to project the trained models to the most robust feature space, thereby reducing the vulnerability to adversarial attacks in non-robust directions. We theoretically characterize the subspace of the eigenspectrum of the feature covariance that is the most robust for a generalized additive model. Our extensive experiments on CIFAR-10, CIFAR-100, tiny ImageNet and ImageNet datasets for several robustness benchmarks, including the state-of-the-art methods in RobustBench show that RFI improves robustness across adaptive and transfer attacks consistently. We also compare RFI with adaptive test-time defenses to demonstrate the effectiveness of our proposed approach.

摘要: 测试时间防御用于提高深层神经网络在推理过程中对敌意例子的稳健性。然而，现有的方法要么需要额外的训练分类器来检测和校正对抗性样本，要么需要对模型参数或输入进行额外的复杂优化以适应测试时的对抗性样本，导致推理时间与基本模型相比显著增加。在这项工作中，我们提出了一种新的测试时间防御策略，称为稳健特征推理(RFI)，该策略易于与任何现有的(稳健)训练过程集成，而不需要额外的测试时间计算。基于我们提出的特征的稳健性的概念，其关键思想是将训练好的模型投影到最健壮的特征空间，从而降低在非健壮方向上对敌方攻击的脆弱性。我们从理论上刻画了广义加性模型的特征协方差的特征谱子空间，它是最稳健的。我们在CIFAR-10、CIFAR-100、Tiny ImageNet和ImageNet数据集上对几个健壮性基准进行的广泛实验表明，RFI一致地提高了对自适应攻击和传输攻击的健壮性。我们还将RFI与自适应测试时间防御进行了比较，以证明我们所提出的方法的有效性。



## **24. Adversarial Training on Purification (AToP): Advancing Both Robustness and Generalization**

净化对抗训练（AToP）：同时推进稳健性和概括性 cs.CV

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2401.16352v4) [paper-pdf](http://arxiv.org/pdf/2401.16352v4)

**Authors**: Guang Lin, Chao Li, Jianhai Zhang, Toshihisa Tanaka, Qibin Zhao

**Abstract**: The deep neural networks are known to be vulnerable to well-designed adversarial attacks. The most successful defense technique based on adversarial training (AT) can achieve optimal robustness against particular attacks but cannot generalize well to unseen attacks. Another effective defense technique based on adversarial purification (AP) can enhance generalization but cannot achieve optimal robustness. Meanwhile, both methods share one common limitation on the degraded standard accuracy. To mitigate these issues, we propose a novel pipeline to acquire the robust purifier model, named Adversarial Training on Purification (AToP), which comprises two components: perturbation destruction by random transforms (RT) and purifier model fine-tuned (FT) by adversarial loss. RT is essential to avoid overlearning to known attacks, resulting in the robustness generalization to unseen attacks, and FT is essential for the improvement of robustness. To evaluate our method in an efficient and scalable way, we conduct extensive experiments on CIFAR-10, CIFAR-100, and ImageNette to demonstrate that our method achieves optimal robustness and exhibits generalization ability against unseen attacks.

摘要: 众所周知，深度神经网络很容易受到精心设计的对抗性攻击。最成功的基于对抗性训练(AT)的防御技术可以达到对特定攻击的最佳健壮性，但不能很好地推广到看不见的攻击。另一种基于对抗性净化(AP)的有效防御技术可以增强泛化能力，但不能达到最优的健壮性。同时，这两种方法都有一个共同的缺陷，那就是标准精度下降。为了缓解这些问题，我们提出了一种新的获取稳健净化器模型的管道，称为对抗性净化训练(TOOP)，它包括两个组成部分：通过随机变换的扰动破坏(RT)和通过对抗性损失微调(FT)的净化器模型。RT是避免对已知攻击过度学习，导致对未知攻击的健壮性泛化的关键，FT是提高健壮性的关键。为了有效和可扩展地评估我们的方法，我们在CIFAR-10、CIFAR-100和ImageNette上进行了大量的实验，证明了我们的方法具有最好的鲁棒性和对不可见攻击的泛化能力。



## **25. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks on Large Language Models**

BackdoorLLM：大型语言模型后门攻击的综合基准 cs.AI

**SubmitDate**: 2024-08-23    [abs](http://arxiv.org/abs/2408.12798v1) [paper-pdf](http://arxiv.org/pdf/2408.12798v1)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative Large Language Models (LLMs) have made significant strides across various tasks, but they remain vulnerable to backdoor attacks, where specific triggers in the prompt cause the LLM to generate adversary-desired responses. While most backdoor research has focused on vision or text classification tasks, backdoor attacks in text generation have been largely overlooked. In this work, we introduce \textit{BackdoorLLM}, the first comprehensive benchmark for studying backdoor attacks on LLMs. \textit{BackdoorLLM} features: 1) a repository of backdoor benchmarks with a standardized training pipeline, 2) diverse attack strategies, including data poisoning, weight poisoning, hidden state attacks, and chain-of-thought attacks, 3) extensive evaluations with over 200 experiments on 8 attacks across 7 scenarios and 6 model architectures, and 4) key insights into the effectiveness and limitations of backdoors in LLMs. We hope \textit{BackdoorLLM} will raise awareness of backdoor threats and contribute to advancing AI safety. The code is available at \url{https://github.com/bboylyg/BackdoorLLM}.

摘要: 生成性大型语言模型(LLM)已经在各种任务中取得了重大进展，但它们仍然容易受到后门攻击，在后门攻击中，提示中的特定触发器会导致LLM生成对手想要的响应。虽然大多数后门研究都集中在视觉或文本分类任务上，但文本生成中的后门攻击在很大程度上被忽视了。在这项工作中，我们介绍了第一个用于研究对LLM的后门攻击的全面基准测试。\textit{Backdoor LLM}的特点是：1)具有标准化培训管道的后门基准存储库；2)多样化的攻击策略，包括数据中毒、重量中毒、隐藏状态攻击和思想链攻击；3)对7个场景和6个模型架构中的8个攻击进行了200多个实验的广泛评估；4)对LLMS中后门的有效性和局限性的关键洞察。我们希望\textit{Backdoor LLM}将提高人们对后门威胁的认识，并为推进人工智能安全做出贡献。代码可在\url{https://github.com/bboylyg/BackdoorLLM}.



## **26. BankTweak: Adversarial Attack against Multi-Object Trackers by Manipulating Feature Banks**

BankTweak：通过操纵特征库对多对象跟踪器进行对抗攻击 cs.CV

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12727v1) [paper-pdf](http://arxiv.org/pdf/2408.12727v1)

**Authors**: Woojin Shin, Donghwa Kang, Daejin Choi, Brent Kang, Jinkyu Lee, Hyeongboo Baek

**Abstract**: Multi-object tracking (MOT) aims to construct moving trajectories for objects, and modern multi-object trackers mainly utilize the tracking-by-detection methodology. Initial approaches to MOT attacks primarily aimed to degrade the detection quality of the frames under attack, thereby reducing accuracy only in those specific frames, highlighting a lack of \textit{efficiency}. To improve efficiency, recent advancements manipulate object positions to cause persistent identity (ID) switches during the association phase, even after the attack ends within a few frames. However, these position-manipulating attacks have inherent limitations, as they can be easily counteracted by adjusting distance-related parameters in the association phase, revealing a lack of \textit{robustness}. In this paper, we present \textsf{BankTweak}, a novel adversarial attack designed for MOT trackers, which features efficiency and robustness. \textsf{BankTweak} focuses on the feature extractor in the association phase and reveals vulnerability in the Hungarian matching method used by feature-based MOT systems. Exploiting the vulnerability, \textsf{BankTweak} induces persistent ID switches (addressing \textit{efficiency}) even after the attack ends by strategically injecting altered features into the feature banks without modifying object positions (addressing \textit{robustness}). To demonstrate the applicability, we apply \textsf{BankTweak} to three multi-object trackers (DeepSORT, StrongSORT, and MOTDT) with one-stage, two-stage, anchor-free, and transformer detectors. Extensive experiments on the MOT17 and MOT20 datasets show that our method substantially surpasses existing attacks, exposing the vulnerability of the tracking-by-detection framework to \textsf{BankTweak}.

摘要: 多目标跟踪旨在构建目标的运动轨迹，而现代多目标跟踪器主要采用检测跟踪的方法。针对MOT攻击的最初方法主要旨在降低被攻击帧的检测质量，从而降低仅在这些特定帧中的准确性，从而突出缺乏效率。为了提高效率，最近的改进操作对象位置以在关联阶段导致永久身份(ID)切换，即使在攻击在几帧内结束之后也是如此。然而，这些操纵位置的攻击具有固有的局限性，因为它们可以通过在关联阶段调整与距离相关的参数来轻松抵消，从而暴露出缺乏文本{健壮性}。本文提出了一种新的针对MOT跟踪器的对抗性攻击方案-.研究了关联阶段的特征提取方法，揭示了基于特征的MOT系统所使用的匈牙利匹配方法中的漏洞。利用该漏洞，即使在攻击结束后，通过在不修改对象位置的情况下有策略地将更改的特征注入到特征库中(寻址\textit{健壮性})，\extsf{BankTware}也会诱导持久的ID切换(寻址\textit{效率})。为了演示其适用性，我们将\extsf{BankTware}应用于三个多对象跟踪器(DeepSORT、StrongSORT和MOTDT)，这些跟踪器具有单级、双级、无锚点和变压器检测器。在MOT17和MOT20数据集上的大量实验表明，我们的方法大大超过了现有的攻击，暴露了检测跟踪框架的漏洞。



## **27. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

利用GE-AdvGAN+增强对抗性攻击的可转移性：梯度编辑的综合框架 cs.AI

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12673v1) [paper-pdf](http://arxiv.org/pdf/2408.12673v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Yuchen Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP

摘要: 可转移的敌意攻击对深度神经网络构成重大威胁，特别是在内部模型信息不可访问的黑盒场景中。研究对抗性攻击方法有助于提高防御机制的性能，探索模型漏洞。这些方法可以发现和利用模型中的弱点，从而促进更健壮的体系结构的开发。然而，当前的可转移攻击方法往往伴随着巨大的计算成本，限制了它们的部署和应用，特别是在边缘计算场景中。对抗性生成模型，如生成性对抗性网络(GANS)，其特点是能够在初始训练阶段后生成样本，而不需要重新训练。GE-AdvGAN是一种新的可转移的对抗性攻击方法，它基于这一原理。本文提出了一种新颖的基于梯度编辑的可转移攻击通用框架GE-AdvGAN+，该框架集成了几乎所有的主流攻击方法，在提高可转移性的同时显著降低了计算资源消耗。我们的实验证明了该框架的兼容性和有效性。与基准的AdvGAN相比，我们性能最好的方法GE-AdvGAN++实现了平均47.8的ASR改进。此外，它还超过了最新的竞争算法GE-AdvGAN，平均ASR提高了5.9。该框架还表现出更高的计算效率，达到2217.7 FPS，优于传统的BIM和MI-FGSM方法。我们GE-AdvGan+框架的实现代码可在https://github.com/GEAdvGANP上获得



## **28. Leveraging Information Consistency in Frequency and Spatial Domain for Adversarial Attacks**

利用频域和空域的信息一致性进行对抗性攻击 cs.LG

Accepted by PRICAI 2024

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12670v1) [paper-pdf](http://arxiv.org/pdf/2408.12670v1)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Yiyun Huang, Huaming Chen

**Abstract**: Adversarial examples are a key method to exploit deep neural networks. Using gradient information, such examples can be generated in an efficient way without altering the victim model. Recent frequency domain transformation has further enhanced the transferability of such adversarial examples, such as spectrum simulation attack. In this work, we investigate the effectiveness of frequency domain-based attacks, aligning with similar findings in the spatial domain. Furthermore, such consistency between the frequency and spatial domains provides insights into how gradient-based adversarial attacks induce perturbations across different domains, which is yet to be explored. Hence, we propose a simple, effective, and scalable gradient-based adversarial attack algorithm leveraging the information consistency in both frequency and spatial domains. We evaluate the algorithm for its effectiveness against different models. Extensive experiments demonstrate that our algorithm achieves state-of-the-art results compared to other gradient-based algorithms. Our code is available at: https://github.com/LMBTough/FSA.

摘要: 对抗性例子是开发深度神经网络的关键方法。利用梯度信息，可以在不改变受害者模型的情况下以高效的方式生成这样的示例。最近的频域变换进一步增强了这种对抗性例子的可转移性，例如频谱模拟攻击。在这项工作中，我们调查了基于频域的攻击的有效性，与空间域中的类似发现一致。此外，频域和空间域之间的这种一致性为基于梯度的敌意攻击如何在不同的域中引发扰动提供了洞察，这一点还有待探索。因此，我们利用频域和空域的信息一致性，提出了一种简单、有效、可扩展的基于梯度的敌意攻击算法。我们在不同的模型上对算法的有效性进行了评估。大量的实验表明，与其他基于梯度的算法相比，我们的算法取得了最好的结果。我们的代码请访问：https://github.com/LMBTough/FSA.



## **29. Prefix Guidance: A Steering Wheel for Large Language Models to Defend Against Jailbreak Attacks**

前置指导：大型语言模型防御越狱攻击的方向盘 cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.08924v2) [paper-pdf](http://arxiv.org/pdf/2408.08924v2)

**Authors**: Jiawei Zhao, Kejiang Chen, Xiaojian Yuan, Weiming Zhang

**Abstract**: In recent years, the rapid development of large language models (LLMs) has achieved remarkable performance across various tasks. However, research indicates that LLMs are vulnerable to jailbreak attacks, where adversaries can induce the generation of harmful content through meticulously crafted prompts. This vulnerability poses significant challenges to the secure use and promotion of LLMs. Existing defense methods offer protection from different perspectives but often suffer from insufficient effectiveness or a significant impact on the model's capabilities. In this paper, we propose a plug-and-play and easy-to-deploy jailbreak defense framework, namely Prefix Guidance (PG), which guides the model to identify harmful prompts by directly setting the first few tokens of the model's output. This approach combines the model's inherent security capabilities with an external classifier to defend against jailbreak attacks. We demonstrate the effectiveness of PG across three models and five attack methods. Compared to baselines, our approach is generally more effective on average. Additionally, results on the Just-Eval benchmark further confirm PG's superiority to preserve the model's performance. our code is available at https://github.com/weiyezhimeng/Prefix-Guidance.

摘要: 近年来，大型语言模型的快速发展在各种任务中取得了显著的性能。然而，研究表明，LLMS容易受到越狱攻击，在越狱攻击中，攻击者可以通过精心制作的提示来诱导生成有害内容。此漏洞对安全使用和推广LLMS构成重大挑战。现有的防御方法从不同的角度提供保护，但往往存在有效性不足或对模型能力产生重大影响的问题。本文提出了一种即插即用、易于部署的越狱防御框架--前缀引导(PG)，它通过直接设置模型输出的前几个令牌来引导模型识别有害提示。这种方法将模型固有的安全功能与外部分类器相结合，以防御越狱攻击。我们在三个模型和五种攻击方法上演示了PG的有效性。与基线相比，我们的方法总体上更有效。此外，在Just-Eval基准上的结果进一步证实了PG在保持模型性能方面的优越性。我们的代码可以在https://github.com/weiyezhimeng/Prefix-Guidance.上找到



## **30. Talos: A More Effective and Efficient Adversarial Defense for GNN Models Based on the Global Homophily of Graphs**

Talos：基于图的全局同质性的GNN模型更有效和高效的对抗防御 cs.LG

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2406.03833v2) [paper-pdf](http://arxiv.org/pdf/2406.03833v2)

**Authors**: Duanyu Li, Huijun Wu, Min Xie, Xugang Wu, Zhenwei Wu, Wenzhe Zhang

**Abstract**: Graph neural network (GNN) models play a pivotal role in numerous tasks involving graph-related data analysis. Despite their efficacy, similar to other deep learning models, GNNs are susceptible to adversarial attacks. Even minor perturbations in graph data can induce substantial alterations in model predictions. While existing research has explored various adversarial defense techniques for GNNs, the challenge of defending against adversarial attacks on real-world scale graph data remains largely unresolved. On one hand, methods reliant on graph purification and preprocessing tend to excessively emphasize local graph information, leading to sub-optimal defensive outcomes. On the other hand, approaches rooted in graph structure learning entail significant time overheads, rendering them impractical for large-scale graphs. In this paper, we propose a new defense method named Talos, which enhances the global, rather than local, homophily of graphs as a defense. Experiments show that the proposed approach notably outperforms state-of-the-art defense approaches, while imposing little computational overhead.

摘要: 图神经网络(GNN)模型在许多涉及图相关数据分析的任务中发挥着关键作用。尽管GNN像其他深度学习模型一样有效，但它很容易受到敌意攻击。即使是图表数据中的微小扰动，也可能导致模型预测的重大变化。虽然现有的研究已经探索了针对GNN的各种对抗性防御技术，但针对真实世界规模的图数据的对抗性攻击的防御挑战在很大程度上仍然没有解决。一方面，依赖于图提纯和预处理的方法往往过分强调局部图信息，导致次优防御结果。另一方面，扎根于图结构学习的方法需要大量的时间开销，使得它们对于大规模图是不现实的。在本文中，我们提出了一种新的防御方法TALOS，它增强了图的全局同伦而不是局部同伦作为防御。实验表明，该方法在计算开销很小的情况下，显著优于现有的防御方法。



## **31. Regularization for Adversarial Robust Learning**

对抗鲁棒学习的正规化 cs.LG

51 pages, 5 figures

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.09672v2) [paper-pdf](http://arxiv.org/pdf/2408.09672v2)

**Authors**: Jie Wang, Rui Gao, Yao Xie

**Abstract**: Despite the growing prevalence of artificial neural networks in real-world applications, their vulnerability to adversarial attacks remains a significant concern, which motivates us to investigate the robustness of machine learning models. While various heuristics aim to optimize the distributionally robust risk using the $\infty$-Wasserstein metric, such a notion of robustness frequently encounters computation intractability. To tackle the computational challenge, we develop a novel approach to adversarial training that integrates $\phi$-divergence regularization into the distributionally robust risk function. This regularization brings a notable improvement in computation compared with the original formulation. We develop stochastic gradient methods with biased oracles to solve this problem efficiently, achieving the near-optimal sample complexity. Moreover, we establish its regularization effects and demonstrate it is asymptotic equivalence to a regularized empirical risk minimization framework, by considering various scaling regimes of the regularization parameter and robustness level. These regimes yield gradient norm regularization, variance regularization, or a smoothed gradient norm regularization that interpolates between these extremes. We numerically validate our proposed method in supervised learning, reinforcement learning, and contextual learning and showcase its state-of-the-art performance against various adversarial attacks.

摘要: 尽管人工神经网络在现实世界中的应用越来越普遍，但它们对对手攻击的脆弱性仍然是一个重要的问题，这促使我们研究机器学习模型的健壮性。虽然各种启发式方法的目标是使用$\infty$-Wasserstein度量来优化分布健壮性风险，但这样的健壮性概念经常遇到计算困难。为了解决计算上的挑战，我们开发了一种新的对抗性训练方法，将$Phi$-发散正则化整合到分布稳健的风险函数中。与原公式相比，这种正则化方法在计算上有了显著的改进。我们发展了带有有偏预言的随机梯度方法来有效地解决这一问题，获得了接近最优的样本复杂度。此外，通过考虑正则化参数和稳健性水平的不同尺度机制，我们建立了它的正则化效应，并证明了它与正则化经验风险最小化框架的渐近等价。这些区域产生在这些极值之间内插的梯度范数正则化、方差正则化或平滑的梯度范数正则化。我们在监督学习、强化学习和上下文学习中对我们提出的方法进行了数值验证，并展示了它在抵抗各种对手攻击方面的最新表现。



## **32. Adversarial Examples in the Physical World: A Survey**

物理世界中的对抗例子：调查 cs.CV

Adversarial examples, physical-world scenarios, attacks and defenses

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2311.01473v3) [paper-pdf](http://arxiv.org/pdf/2311.01473v3)

**Authors**: Jiakai Wang, Xianglong Liu, Jin Hu, Donghua Wang, Siyang Wu, Tingsong Jiang, Yuanfang Guo, Aishan Liu, Jiantao Zhou

**Abstract**: Deep neural networks (DNNs) have demonstrated high vulnerability to adversarial examples, raising broad security concerns about their applications. Besides the attacks in the digital world, the practical implications of adversarial examples in the physical world present significant challenges and safety concerns. However, current research on physical adversarial examples (PAEs) lacks a comprehensive understanding of their unique characteristics, leading to limited significance and understanding. In this paper, we address this gap by thoroughly examining the characteristics of PAEs within a practical workflow encompassing training, manufacturing, and re-sampling processes. By analyzing the links between physical adversarial attacks, we identify manufacturing and re-sampling as the primary sources of distinct attributes and particularities in PAEs. Leveraging this knowledge, we develop a comprehensive analysis and classification framework for PAEs based on their specific characteristics, covering over 100 studies on physical-world adversarial examples. Furthermore, we investigate defense strategies against PAEs and identify open challenges and opportunities for future research. We aim to provide a fresh, thorough, and systematic understanding of PAEs, thereby promoting the development of robust adversarial learning and its application in open-world scenarios to provide the community with a continuously updated list of physical world adversarial sample resources, including papers, code, \etc, within the proposed framework

摘要: 深度神经网络(DNN)对敌意例子表现出很高的脆弱性，引起了人们对其应用的广泛安全担忧。除了数字世界中的攻击，物理世界中敌意例子的实际影响也带来了重大挑战和安全问题。然而，目前对身体对抗例子(PAE)的研究缺乏对其独特特征的全面了解，导致其意义和理解有限。在本文中，我们通过彻底检查包括培训、制造和重新采样过程在内的实际工作流程中的PAE的特征来解决这一差距。通过分析物理对抗性攻击之间的联系，我们确定制造和重采样是PAE中不同属性和特殊性的主要来源。利用这些知识，我们根据PAE的具体特征开发了一个全面的分析和分类框架，涵盖了100多个物理世界对抗性例子的研究。此外，我们还研究了针对PAE的防御策略，并确定了未来研究的开放挑战和机会。我们的目标是提供一个新的，彻底的和系统的了解，从而促进发展强大的对抗性学习及其在开放世界情景中的应用，以在拟议的框架内为社区提供持续更新的物理世界对抗性样本资源列表，包括论文、代码等



## **33. Query-Efficient Video Adversarial Attack with Stylized Logo**

具有风格化徽标的查询高效视频对抗攻击 cs.CV

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2408.12099v1) [paper-pdf](http://arxiv.org/pdf/2408.12099v1)

**Authors**: Duoxun Tang, Yuxin Cao, Xi Xiao, Derui Wang, Sheng Wen, Tianqing Zhu

**Abstract**: Video classification systems based on Deep Neural Networks (DNNs) have demonstrated excellent performance in accurately verifying video content. However, recent studies have shown that DNNs are highly vulnerable to adversarial examples. Therefore, a deep understanding of adversarial attacks can better respond to emergency situations. In order to improve attack performance, many style-transfer-based attacks and patch-based attacks have been proposed. However, the global perturbation of the former will bring unnatural global color, while the latter is difficult to achieve success in targeted attacks due to the limited perturbation space. Moreover, compared to a plethora of methods targeting image classifiers, video adversarial attacks are still not that popular. Therefore, to generate adversarial examples with a low budget and to provide them with a higher verisimilitude, we propose a novel black-box video attack framework, called Stylized Logo Attack (SLA). SLA is conducted through three steps. The first step involves building a style references set for logos, which can not only make the generated examples more natural, but also carry more target class features in the targeted attacks. Then, reinforcement learning (RL) is employed to determine the style reference and position parameters of the logo within the video, which ensures that the stylized logo is placed in the video with optimal attributes. Finally, perturbation optimization is designed to optimize perturbations to improve the fooling rate in a step-by-step manner. Sufficient experimental results indicate that, SLA can achieve better performance than state-of-the-art methods and still maintain good deception effects when facing various defense methods.

摘要: 基于深度神经网络(DNNS)的视频分类系统在准确验证视频内容方面表现出了优异的性能。然而，最近的研究表明，DNN非常容易受到敌意例子的影响。因此，深入了解对抗性攻击可以更好地应对紧急情况。为了提高攻击性能，人们提出了许多基于样式转移的攻击和基于补丁的攻击。然而，前者的全局扰动会带来不自然的全局色彩，而后者由于扰动空间有限，很难在定向攻击中取得成功。此外，与大量针对图像分类器的方法相比，视频对抗性攻击仍然不太受欢迎。因此，为了以较低的预算生成对抗性实例，并为它们提供更高的逼真度，我们提出了一种新的黑盒视频攻击框架，称为Stylize Logo攻击(SLA)。SLA分三个步骤进行。第一步是为标识建立一个样式引用集，这样不仅可以使生成的示例更加自然，而且可以在有针对性的攻击中携带更多的目标类特征。然后，利用强化学习(RL)来确定标识在视频中的样式参考和位置参数，从而确保风格化的标识以最优的属性放置在视频中。最后，设计了扰动优化算法，对扰动进行了优化，逐步提高了上愚率。充分的实验结果表明，SLA在面对各种防御手段时，能够取得比现有方法更好的性能，同时仍能保持良好的欺骗效果。



## **34. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

通过潜在对抗训练防御不可预见的失败模式 cs.CR

**SubmitDate**: 2024-08-22    [abs](http://arxiv.org/abs/2403.05030v4) [paper-pdf](http://arxiv.org/pdf/2403.05030v4)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove trojans and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.

摘要: 尽管开发人员进行了广泛的诊断和调试，但人工智能系统有时会表现出有害的意外行为。找到并修复这些攻击是具有挑战性的，因为攻击面太大了--要详尽地搜索可能引发有害行为的输入并不容易。红队和对抗性训练(AT)通常用于提高健壮性，然而，根据经验，它们难以修复与训练期间使用的攻击不同的失败模式。在这项工作中，我们利用潜在的对手训练(LAT)来防御漏洞，而不利用对它们是什么的知识或使用引起它们的输入。后者利用网络实际用于预测的概念的压缩、抽象和结构化的潜在表示。在这里，我们使用它来防御没有引出故障模式的示例。具体地说，我们使用LAT来删除特洛伊木马程序并防御抵抗类的对抗性攻击。我们在图像分类、文本分类和文本生成任务中表明，相对于AT，LAT通常可以提高对新攻击的健壮性和对干净数据的性能。这表明，LAT可以成为一种很有前途的工具，用于防御开发人员未明确识别的故障模式。



## **35. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

隐性对抗培训提高了法学硕士对持续有害行为的稳健性 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.15549v2) [paper-pdf](http://arxiv.org/pdf/2407.15549v2)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.

摘要: 大型语言模型(LLM)通常会以不受欢迎的方式运行，因此它们被明确微调为不以这种方式运行。例如，LLM的红队文献已经创造了各种各样的“越狱”技术，从经过微调的无害的模特那里引出有害文本。最近在红团队、模型编辑和可解释性方面的工作表明，这一挑战源于(对抗性的)微调如何在很大程度上抑制而不是消除LLM中不受欢迎的能力。以前的工作已经引入了潜在的对手训练(LAT)，作为一种提高对广泛类别的故障的稳健性的方式。这些先前的工作考虑了无目标的潜在空间攻击，即对手扰乱潜在激活，以最大限度地减少期望行为的示例损失。非定向LAT可以提供一般类型的健壮性，但不利用有关特定故障模式的信息。在这里，我们实验有针对性的LAT，其中对手试图将特定竞争任务的损失降至最低。我们发现，它可以增加各种最先进的方法。首先，我们使用有针对性的LAT来提高对越狱的健壮性，性能优于强大的R2D2基线，计算量少了几个数量级。其次，我们使用它来更有效地删除后门，而不知道触发器。最后，我们使用它来更有效地忘记特定不受欢迎的任务的知识，这种方式也更适合重新学习。总体而言，我们的结果表明，有针对性的LAT可以成为防御LLM有害行为的有效工具。



## **36. Fight Back Against Jailbreaking via Prompt Adversarial Tuning**

通过即时对抗调整反击越狱 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2402.06255v3) [paper-pdf](http://arxiv.org/pdf/2402.06255v3)

**Authors**: Yichuan Mo, Yuji Wang, Zeming Wei, Yisen Wang

**Abstract**: While Large Language Models (LLMs) have achieved tremendous success in various applications, they are also susceptible to jailbreak attacks. Several primary defense strategies have been proposed to protect LLMs from producing harmful information, mostly with a particular focus on harmful content filtering or heuristical defensive prompt designs. However, how to achieve intrinsic robustness through the prompts remains an open problem. In this paper, motivated by adversarial training paradigms for achieving reliable robustness, we propose an approach named Prompt Adversarial Tuning (PAT) that trains a prompt control attached to the user prompt as a guard prefix. To achieve our defense goal whilst maintaining natural performance, we optimize the control prompt with both adversarial and benign prompts. Comprehensive experiments show that our method is effective against both grey-box and black-box attacks, reducing the success rate of advanced attacks to nearly 0 while maintaining the model's utility on the benign task. The proposed defense strategy incurs only negligible computational overhead, charting a new perspective for future explorations in LLM security. Our code is available at https://github.com/rain152/PAT.

摘要: 虽然大型语言模型(LLM)在各种应用中取得了巨大的成功，但它们也容易受到越狱攻击。已经提出了几种主要的防御策略来保护LLMS免受有害信息的影响，主要集中在有害内容过滤或启发式防御提示设计上。然而，如何通过提示实现内在的稳健性仍然是一个悬而未决的问题。受实现可靠健壮性的对抗性训练范例的启发，我们提出了一种称为即时对抗性调整(PAT)的方法，该方法将附加在用户提示上的提示控制训练为保卫前缀。为了在保持自然表现的同时实现我们的防守目标，我们优化了控制提示，包括对抗性提示和良性提示。综合实验表明，该方法对灰盒攻击和黑盒攻击都是有效的，在保持模型对良性任务的实用性的同时，将高级攻击的成功率降低到接近0。所提出的防御策略只需要很少的计算开销，为未来在LLM安全方面的探索开辟了新的前景。我们的代码可以在https://github.com/rain152/PAT.上找到



## **37. Competence-Based Analysis of Language Models**

基于能力的语言模型分析 cs.CL

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2303.00333v4) [paper-pdf](http://arxiv.org/pdf/2303.00333v4)

**Authors**: Adam Davies, Jize Jiang, ChengXiang Zhai

**Abstract**: Despite the recent successes of large, pretrained neural language models (LLMs), comparatively little is known about the representations of linguistic structure they learn during pretraining, which can lead to unexpected behaviors in response to prompt variation or distribution shift. To better understand these models and behaviors, we introduce a general model analysis framework to study LLMs with respect to their representation and use of human-interpretable linguistic properties. Our framework, CALM (Competence-based Analysis of Language Models), is designed to investigate LLM competence in the context of specific tasks by intervening on models' internal representations of different linguistic properties using causal probing, and measuring models' alignment under these interventions with a given ground-truth causal model of the task. We also develop a new approach for performing causal probing interventions using gradient-based adversarial attacks, which can target a broader range of properties and representations than prior techniques. Finally, we carry out a case study of CALM using these interventions to analyze and compare LLM competence across a variety of lexical inference tasks, showing that CALM can be used to explain and predict behaviors across these tasks.

摘要: 尽管最近大型的预训练神经语言模型(LLM)取得了成功，但人们对它们在预训练中学习的语言结构的表征知之甚少，这可能会导致对迅速变化或分布变化的意外行为。为了更好地理解这些模型和行为，我们引入了一个通用的模型分析框架，从它们对人类可解释的语言属性的表示和使用方面来研究LLM。基于能力的语言模型分析框架旨在通过因果探究干预模型对不同语言属性的内部表征，并测量模型在这些干预下与给定任务的基本事实因果模型的一致性，从而考察特定任务背景下的语言学习能力。我们还开发了一种使用基于梯度的对抗性攻击来执行因果探测干预的新方法，该方法可以针对比现有技术更广泛的属性和表示。最后，我们使用这些干预手段对CAMLE进行了个案研究，分析和比较了不同词汇推理任务的LLM能力，结果表明CAMPE可以用来解释和预测这些任务中的行为。



## **38. Against All Odds: Overcoming Typology, Script, and Language Confusion in Multilingual Embedding Inversion Attacks**

克服一切困难：克服多语言嵌入倒置攻击中的类型学、脚本和语言混乱 cs.CL

11 pages, 4 figures, 7 tables

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11749v1) [paper-pdf](http://arxiv.org/pdf/2408.11749v1)

**Authors**: Yiyi Chen, Russa Biswas, Heather Lent, Johannes Bjerva

**Abstract**: Large Language Models (LLMs) are susceptible to malicious influence by cyber attackers through intrusions such as adversarial, backdoor, and embedding inversion attacks. In response, the burgeoning field of LLM Security aims to study and defend against such threats. Thus far, the majority of works in this area have focused on monolingual English models, however, emerging research suggests that multilingual LLMs may be more vulnerable to various attacks than their monolingual counterparts. While previous work has investigated embedding inversion over a small subset of European languages, it is challenging to extrapolate these findings to languages from different linguistic families and with differing scripts. To this end, we explore the security of multilingual LLMs in the context of embedding inversion attacks and investigate cross-lingual and cross-script inversion across 20 languages, spanning over 8 language families and 12 scripts. Our findings indicate that languages written in Arabic script and Cyrillic script are particularly vulnerable to embedding inversion, as are languages within the Indo-Aryan language family. We further observe that inversion models tend to suffer from language confusion, sometimes greatly reducing the efficacy of an attack. Accordingly, we systematically explore this bottleneck for inversion models, uncovering predictable patterns which could be leveraged by attackers. Ultimately, this study aims to further the field's understanding of the outstanding security vulnerabilities facing multilingual LLMs and raise awareness for the languages most at risk of negative impact from these attacks.

摘要: 大型语言模型(LLM)容易受到网络攻击者通过对抗性、后门和嵌入反转攻击等入侵的恶意影响。作为回应，LLM Security这个新兴领域的目标是研究和防御此类威胁。到目前为止，这一领域的研究大多集中在单语英语模型上，然而，新的研究表明，多语种的LLM可能比单语的LLM更容易受到各种攻击。虽然以前的工作已经研究了在一小部分欧洲语言上嵌入倒置，但将这些发现外推到来自不同语系和不同脚本的语言是具有挑战性的。为此，我们在嵌入倒置攻击的情况下探索了多语言LLMS的安全性，并研究了跨语言和跨脚本的跨语言和跨脚本倒置，涉及8个语系和12个脚本。我们的发现表明，用阿拉伯文字和西里尔文字书写的语言特别容易嵌入倒置，印度-雅利安语系的语言也是如此。我们进一步观察到，倒置模型往往受到语言混乱的影响，有时会极大地降低攻击的有效性。因此，我们系统地探索了倒置模型的这一瓶颈，揭示了可被攻击者利用的可预测模式。最终，这项研究旨在加深外地对多语种土地管理系统面临的突出安全漏洞的了解，并提高对最有可能受到这些攻击的负面影响的语言的认识。



## **39. First line of defense: A robust first layer mitigates adversarial attacks**

第一道防线：强大的第一层减轻对抗攻击 cs.LG

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11680v1) [paper-pdf](http://arxiv.org/pdf/2408.11680v1)

**Authors**: Janani Suresh, Nancy Nayak, Sheetal Kalyani

**Abstract**: Adversarial training (AT) incurs significant computational overhead, leading to growing interest in designing inherently robust architectures. We demonstrate that a carefully designed first layer of the neural network can serve as an implicit adversarial noise filter (ANF). This filter is created using a combination of large kernel size, increased convolution filters, and a maxpool operation. We show that integrating this filter as the first layer in architectures such as ResNet, VGG, and EfficientNet results in adversarially robust networks. Our approach achieves higher adversarial accuracies than existing natively robust architectures without AT and is competitive with adversarial-trained architectures across a wide range of datasets. Supporting our findings, we show that (a) the decision regions for our method have better margins, (b) the visualized loss surfaces are smoother, (c) the modified peak signal-to-noise ratio (mPSNR) values at the output of the ANF are higher, (d) high-frequency components are more attenuated, and (e) architectures incorporating ANF exhibit better denoising in Gaussian noise compared to baseline architectures. Code for all our experiments are available at \url{https://github.com/janani-suresh-97/first-line-defence.git}.

摘要: 对抗训练(AT)带来了巨大的计算开销，导致人们对设计具有内在健壮性的体系结构的兴趣与日俱增。我们证明了精心设计的第一层神经网络可以用作隐式对抗性噪声过滤器(ANF)。该过滤器使用较大的内核大小、增加的卷积过滤器和最大池操作的组合来创建。我们表明，在ResNet、VGG和EfficientNet等体系结构中集成该过滤器作为第一层会产生相反的健壮性网络。我们的方法获得了比现有的没有AT的本地健壮体系结构更高的对抗准确率，并且在广泛的数据集上与经过对抗训练的体系结构具有竞争力。支持我们的发现，我们的结果表明：(A)我们的方法的判决区域具有更好的裕度，(B)可视化的损失表面更平滑，(C)ANF输出的修正峰值信噪比(MPSNR)值更高，(D)高频分量更弱，(E)与基线结构相比，结合ANF的结构在高斯噪声中表现出更好的去噪效果。我们所有实验的代码都可以在\url{https://github.com/janani-suresh-97/first-line-defence.git}.上找到



## **40. Latent Feature and Attention Dual Erasure Attack against Multi-View Diffusion Models for 3D Assets Protection**

针对3D资产保护的多视图扩散模型的潜在特征和注意力双重擦除攻击 cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11408v1) [paper-pdf](http://arxiv.org/pdf/2408.11408v1)

**Authors**: Jingwei Sun, Xuchong Zhang, Changfeng Sun, Qicheng Bai, Hongbin Sun

**Abstract**: Multi-View Diffusion Models (MVDMs) enable remarkable improvements in the field of 3D geometric reconstruction, but the issue regarding intellectual property has received increasing attention due to unauthorized imitation. Recently, some works have utilized adversarial attacks to protect copyright. However, all these works focus on single-image generation tasks which only need to consider the inner feature of images. Previous methods are inefficient in attacking MVDMs because they lack the consideration of disrupting the geometric and visual consistency among the generated multi-view images. This paper is the first to address the intellectual property infringement issue arising from MVDMs. Accordingly, we propose a novel latent feature and attention dual erasure attack to disrupt the distribution of latent feature and the consistency across the generated images from multi-view and multi-domain simultaneously. The experiments conducted on SOTA MVDMs indicate that our approach achieves superior performances in terms of attack effectiveness, transferability, and robustness against defense methods. Therefore, this paper provides an efficient solution to protect 3D assets from MVDMs-based 3D geometry reconstruction.

摘要: 多视点扩散模型(MVDM)在三维几何重建领域取得了显著的进步，但由于未经授权的仿制，涉及知识产权的问题也越来越受到关注。最近，一些作品利用对抗性攻击来保护版权。然而，这些工作都集中在单幅图像生成任务上，只需要考虑图像的内部特征。以前的方法在攻击MVDM时效率不高，因为它们没有考虑破坏生成的多视角图像之间的几何和视觉一致性。这是第一篇关于MVDM引起的知识产权侵权问题的论文。因此，我们提出了一种新的潜在特征和注意双重擦除攻击，以同时扰乱潜在特征的分布和多视角、多领域生成图像的一致性。在Sota MVDM上进行的实验表明，我们的方法在攻击有效性、可转移性和对防御方法的健壮性方面取得了优越的性能。因此，本文为保护3D资产免受基于MVDM的3D几何重建提供了一种有效的解决方案。



## **41. AntifakePrompt: Prompt-Tuned Vision-Language Models are Fake Image Detectors**

AntifakePrompt：预算调整的视觉语言模型是假图像检测器 cs.CV

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2310.17419v3) [paper-pdf](http://arxiv.org/pdf/2310.17419v3)

**Authors**: You-Ming Chang, Chen Yeh, Wei-Chen Chiu, Ning Yu

**Abstract**: Deep generative models can create remarkably photorealistic fake images while raising concerns about misinformation and copyright infringement, known as deepfake threats. Deepfake detection technique is developed to distinguish between real and fake images, where the existing methods typically learn classifiers in the image domain or various feature domains. However, the generalizability of deepfake detection against emerging and more advanced generative models remains challenging. In this paper, being inspired by the zero-shot advantages of Vision-Language Models (VLMs), we propose a novel approach called AntifakePrompt, using VLMs (e.g., InstructBLIP) and prompt tuning techniques to improve the deepfake detection accuracy over unseen data. We formulate deepfake detection as a visual question answering problem, and tune soft prompts for InstructBLIP to answer the real/fake information of a query image. We conduct full-spectrum experiments on datasets from a diversity of 3 held-in and 20 held-out generative models, covering modern text-to-image generation, image editing and adversarial image attacks. These testing datasets provide useful benchmarks in the realm of deepfake detection for further research. Moreover, results demonstrate that (1) the deepfake detection accuracy can be significantly and consistently improved (from 71.06% to 92.11%, in average accuracy over unseen domains) using pretrained vision-language models with prompt tuning; (2) our superior performance is at less cost of training data and trainable parameters, resulting in an effective and efficient solution for deepfake detection. Code and models can be found at https://github.com/nctu-eva-lab/AntifakePrompt.

摘要: 深度生成模型可以创建非常逼真的虚假图像，同时引发人们对错误信息和侵犯版权的担忧，即所谓的深度虚假威胁。深伪检测技术是为了区分真实和虚假的图像而发展起来的，现有的方法通常在图像域或各种特征域学习分类器。然而，针对新兴的和更高级的生成模型的深伪检测的泛化能力仍然具有挑战性。受视觉语言模型(VLMS)零射优势的启发，本文提出了一种新的基于视觉语言模型(VLMS，InstructBLIP)和提示调优的方法，以提高对不可见数据的深度伪检测精度。我们将深度伪检测描述为一个视觉问答问题，并对InstructBLIP的软提示进行调整，以回答查询图像的真假信息。我们在来自3个坚持和20个坚持的生成模型的数据集上进行了全谱实验，涵盖了现代文本到图像的生成、图像编辑和对抗性图像攻击。这些测试数据集为深度伪检测领域的进一步研究提供了有用的基准。实验结果表明：(1)通过快速调整预先训练的视觉语言模型，深度伪检测的正确率可以从71.06%提高到92.11%；(2)我们的优越性能是以较少的训练数据和可训练的参数为代价的，从而为深度伪检测提供了一个有效和高效的解决方案。代码和模型可在https://github.com/nctu-eva-lab/AntifakePrompt.上找到



## **42. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

17 pages, 4 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2406.19692v3) [paper-pdf](http://arxiv.org/pdf/2406.19692v3)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **43. Investigating Imperceptibility of Adversarial Attacks on Tabular Data: An Empirical Analysis**

调查表格数据对抗性攻击的不可感知性：实证分析 cs.LG

33 pages

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2407.11463v2) [paper-pdf](http://arxiv.org/pdf/2407.11463v2)

**Authors**: Zhipeng He, Chun Ouyang, Laith Alzubaidi, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks are a potential threat to machine learning models by causing incorrect predictions through imperceptible perturbations to the input data. While these attacks have been extensively studied in unstructured data like images, applying them to tabular data, poses new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ from the image data. To account for this distinction, it is necessary to establish tailored imperceptibility criteria specific to tabular data. However, there is currently a lack of standardised metrics for assessing the imperceptibility of adversarial attacks on tabular data. To address this gap, we propose a set of key properties and corresponding metrics designed to comprehensively characterise imperceptible adversarial attacks on tabular data. These are: proximity to the original input, sparsity of altered features, deviation from the original data distribution, sensitivity in perturbing features with narrow distribution, immutability of certain features that should remain unchanged, feasibility of specific feature values that should not go beyond valid practical ranges, and feature interdependencies capturing complex relationships between data attributes. We evaluate the imperceptibility of five adversarial attacks, including both bounded attacks and unbounded attacks, on tabular data using the proposed imperceptibility metrics. The results reveal a trade-off between the imperceptibility and effectiveness of these attacks. The study also identifies limitations in current attack algorithms, offering insights that can guide future research in the area. The findings gained from this empirical analysis provide valuable direction for enhancing the design of adversarial attack algorithms, thereby advancing adversarial machine learning on tabular data.

摘要: 对抗性攻击通过对输入数据的不可察觉的扰动而导致错误的预测，从而对机器学习模型构成潜在的威胁。虽然这些攻击已经在图像等非结构化数据中得到了广泛研究，但将它们应用于表格数据带来了新的挑战。这些挑战源于表格数据固有的异构性和复杂的特征相互依赖关系，而表格数据不同于图像数据。为了说明这一区别，有必要建立专门针对表格数据的不可察觉标准。然而，目前缺乏用于评估对抗性攻击对表格数据的不可感知性的标准化指标。为了弥补这一差距，我们提出了一组关键属性和相应的度量，旨在全面表征对表格数据的不可察觉的对抗性攻击。它们是：接近原始输入、改变特征的稀疏性、偏离原始数据分布、对具有窄分布的扰动特征的敏感性、某些应保持不变的特征的不变性、不应超出有效实际范围的特定特征值的可行性、以及捕捉数据属性之间的复杂关系的特征相互依赖关系。我们使用所提出的不可感知性度量评估了五种对抗性攻击，包括有界攻击和无界攻击对表格数据的不可感知性。结果揭示了这些攻击的隐蔽性和有效性之间的权衡。该研究还确定了当前攻击算法的局限性，提供了可以指导该领域未来研究的见解。这一实证分析的结果为改进对抗性攻击算法的设计，从而推进对抗性表格数据机器学习提供了有价值的指导。



## **44. Unlocking Adversarial Suffix Optimization Without Affirmative Phrases: Efficient Black-box Jailbreaking via LLM as Optimizer**

在没有肯定短语的情况下解锁敌对后缀优化：通过LLM作为优化器的高效黑匣子越狱 cs.AI

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11313v1) [paper-pdf](http://arxiv.org/pdf/2408.11313v1)

**Authors**: Weipeng Jiang, Zhenting Wang, Juan Zhai, Shiqing Ma, Zhengyu Zhao, Chao Shen

**Abstract**: Despite prior safety alignment efforts, mainstream LLMs can still generate harmful and unethical content when subjected to jailbreaking attacks. Existing jailbreaking methods fall into two main categories: template-based and optimization-based methods. The former requires significant manual effort and domain knowledge, while the latter, exemplified by Greedy Coordinate Gradient (GCG), which seeks to maximize the likelihood of harmful LLM outputs through token-level optimization, also encounters several limitations: requiring white-box access, necessitating pre-constructed affirmative phrase, and suffering from low efficiency. In this paper, we present ECLIPSE, a novel and efficient black-box jailbreaking method utilizing optimizable suffixes. Drawing inspiration from LLMs' powerful generation and optimization capabilities, we employ task prompts to translate jailbreaking goals into natural language instructions. This guides the LLM to generate adversarial suffixes for malicious queries. In particular, a harmfulness scorer provides continuous feedback, enabling LLM self-reflection and iterative optimization to autonomously and efficiently produce effective suffixes. Experimental results demonstrate that ECLIPSE achieves an average attack success rate (ASR) of 0.92 across three open-source LLMs and GPT-3.5-Turbo, significantly surpassing GCG in 2.4 times. Moreover, ECLIPSE is on par with template-based methods in ASR while offering superior attack efficiency, reducing the average attack overhead by 83%.

摘要: 尽管之前做出了安全调整的努力，但主流LLM在受到越狱攻击时仍然会产生有害和不道德的内容。现有的越狱方法主要分为两类：基于模板的方法和基于优化的方法。前者需要大量的人工工作和领域知识，而后者，例如贪婪坐标梯度(GCG)，试图通过令牌级优化最大化有害的LLM输出的可能性，也遇到了几个限制：需要白盒访问，必须预先构建肯定短语，以及效率低下。在本文中，我们提出了一种利用可优化后缀的新颖高效的黑盒越狱方法--ECLIPSE。从LLMS强大的生成和优化能力中获得灵感，我们使用任务提示将越狱目标转换为自然语言指令。这将引导LLM为恶意查询生成敌意后缀。特别是，危害评分器提供持续的反馈，使LLM自我反省和迭代优化能够自主和高效地产生有效的后缀。实验结果表明，ECLIPSE在三个开源LLMS和GPT-3.5-Turbo上的平均攻击成功率(ASR)为0.92，显著超过GCG的2.4倍。此外，在ASR中，eclipse与基于模板的方法不相上下，同时提供了优越的攻击效率，将平均攻击开销降低了83%。



## **45. EEG-Defender: Defending against Jailbreak through Early Exit Generation of Large Language Models**

EEG-Defender：通过早期退出生成大型语言模型来抵御越狱 cs.AI

19 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11308v1) [paper-pdf](http://arxiv.org/pdf/2408.11308v1)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. Built upon this idea, we introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85\% in comparison with 50\% for the present SOTAs, with minimal impact on the utility and effectiveness of LLMs.

摘要: 大语言模型在各种应用中日益引起人们的关注。尽管如此，随着一些用户试图利用这些模型达到恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了减轻这种风险，人们提出了“对准”技术的概念。然而，最近的研究表明，这种对齐可以使用复杂的即时工程或敌对后缀来破坏，这是一种被称为“越狱”的技术。我们的研究从LLMS类似人类的生成过程中获得了线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示。利用这一发现，我们建议利用LLMS的早期变压器输出作为一种手段来检测恶意输入，并立即终止生成。基于这一想法，我们介绍了一种简单但重要的防御方法，称为用于LLMS的EEG-Defender。我们在三个模型上对十种越狱方法进行了全面的实验。我们的结果表明，EEG-Defender能够显著降低攻击成功率(ASR)，与现有SOTAS的50%相比，约为85%，而对LLMS的实用性和有效性的影响最小。



## **46. Correlation Analysis of Adversarial Attack in Time Series Classification**

时间序列分类中对抗性攻击的相关性分析 cs.LG

15 pages, 7 figures

**SubmitDate**: 2024-08-21    [abs](http://arxiv.org/abs/2408.11264v1) [paper-pdf](http://arxiv.org/pdf/2408.11264v1)

**Authors**: Zhengyang Li, Wenhao Liang, Chang Dong, Weitong Chen, Dong Huang

**Abstract**: This study investigates the vulnerability of time series classification models to adversarial attacks, with a focus on how these models process local versus global information under such conditions. By leveraging the Normalized Auto Correlation Function (NACF), an exploration into the inclination of neural networks is conducted. It is demonstrated that regularization techniques, particularly those employing Fast Fourier Transform (FFT) methods and targeting frequency components of perturbations, markedly enhance the effectiveness of attacks. Meanwhile, the defense strategies, like noise introduction and Gaussian filtering, are shown to significantly lower the Attack Success Rate (ASR), with approaches based on noise introducing notably effective in countering high-frequency distortions. Furthermore, models designed to prioritize global information are revealed to possess greater resistance to adversarial manipulations. These results underline the importance of designing attack and defense mechanisms, informed by frequency domain analysis, as a means to considerably reinforce the resilience of neural network models against adversarial threats.

摘要: 本文研究了时间序列分类模型对敌意攻击的脆弱性，重点研究了在这种情况下这些模型是如何处理局部和全局信息的。利用归一化自相关函数(NACF)对神经网络的倾向性进行了探讨。结果表明，正则化技术，特别是利用快速傅立叶变换(FFT)方法和针对扰动的频率分量的正则化技术，显著地提高了攻击的有效性。同时，噪声引入和高斯滤波等防御策略显著降低了攻击成功率，其中基于噪声引入的防御策略在对抗高频失真方面效果显著。此外，旨在对全球信息进行优先排序的模型被揭示出对对手操纵具有更强的抵抗力。这些结果强调了通过频域分析设计攻击和防御机制的重要性，以此作为显著增强神经网络模型对对手威胁的弹性的一种手段。



## **47. Revisiting Min-Max Optimization Problem in Adversarial Training**

重温对抗训练中的最小-最大优化问题 cs.CV

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.11218v1) [paper-pdf](http://arxiv.org/pdf/2408.11218v1)

**Authors**: Sina Hajer Ahmadi, Hassan Bahrami

**Abstract**: The rise of computer vision applications in the real world puts the security of the deep neural networks at risk. Recent works demonstrate that convolutional neural networks are susceptible to adversarial examples - where the input images look similar to the natural images but are classified incorrectly by the model. To provide a rebuttal to this problem, we propose a new method to build robust deep neural networks against adversarial attacks by reformulating the saddle point optimization problem in \cite{madry2017towards}. Our proposed method offers significant resistance and a concrete security guarantee against multiple adversaries. The goal of this paper is to act as a stepping stone for a new variation of deep learning models which would lead towards fully robust deep learning models.

摘要: 现实世界中计算机视觉应用的兴起使深度神经网络的安全面临风险。最近的工作表明，卷积神经网络容易受到对抗性示例的影响--其中输入图像看起来与自然图像相似，但模型分类错误。为了反驳这个问题，我们提出了一种新的方法，通过重新定义\cite{madry 2017 toward}中的鞍点优化问题来构建稳健的深度神经网络来对抗对抗攻击。我们提出的方法提供了针对多个对手的显着的抵抗力和具体的安全保证。本文的目标是成为深度学习模型新变体的垫脚石，这将导致完全稳健的深度学习模型。



## **48. Makeup-Guided Facial Privacy Protection via Untrained Neural Network Priors**

通过未经训练的神经网络先验进行化妆引导的面部隐私保护 cs.CV

Proceedings of ECCV Workshop on Explainable AI for Biometrics, 2024

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.12387v1) [paper-pdf](http://arxiv.org/pdf/2408.12387v1)

**Authors**: Fahad Shamshad, Muzammal Naseer, Karthik Nandakumar

**Abstract**: Deep learning-based face recognition (FR) systems pose significant privacy risks by tracking users without their consent. While adversarial attacks can protect privacy, they often produce visible artifacts compromising user experience. To mitigate this issue, recent facial privacy protection approaches advocate embedding adversarial noise into the natural looking makeup styles. However, these methods require training on large-scale makeup datasets that are not always readily available. In addition, these approaches also suffer from dataset bias. For instance, training on makeup data that predominantly contains female faces could compromise protection efficacy for male faces. To handle these issues, we propose a test-time optimization approach that solely optimizes an untrained neural network to transfer makeup style from a reference to a source image in an adversarial manner. We introduce two key modules: a correspondence module that aligns regions between reference and source images in latent space, and a decoder with conditional makeup layers. The untrained decoder, optimized via carefully designed structural and makeup consistency losses, generates a protected image that resembles the source but incorporates adversarial makeup to deceive FR models. As our approach does not rely on training with makeup face datasets, it avoids potential male/female dataset biases while providing effective protection. We further extend the proposed approach to videos by leveraging on temporal correlations. Experiments on benchmark datasets demonstrate superior performance in face verification and identification tasks and effectiveness against commercial FR systems. Our code and models will be available at https://github.com/fahadshamshad/deep-facial-privacy-prior

摘要: 基于深度学习的人脸识别(FR)系统在未经用户同意的情况下跟踪用户，从而带来了严重的隐私风险。虽然对抗性攻击可以保护隐私，但它们通常会产生影响用户体验的可见人工制品。为了缓解这个问题，最近的面部隐私保护方法主张在自然的化妆风格中嵌入对抗性噪音。然而，这些方法需要在大规模组成数据集上进行培训，而这些数据集并不总是现成的。此外，这些方法还受到数据集偏差的影响。例如，针对主要包含女性面孔的化妆数据进行培训，可能会影响对男性面孔的保护效果。为了解决这些问题，我们提出了一种测试时间优化方法，该方法只优化一个未训练的神经网络，以对抗的方式将化妆风格从参考图像转移到源图像。我们介绍了两个关键模块：一个是在潜在空间中对齐参考图像和源图像区域的对应模块，另一个是带有条件补充层的解码器。未经训练的解码器，通过精心设计的结构和化妆一致性损失进行优化，生成类似于源文件但包含敌意化妆的受保护图像，以欺骗FR模型。由于我们的方法不依赖于化妆人脸数据集的训练，因此在提供有效保护的同时避免了潜在的男性/女性数据集偏见。通过利用时间相关性，我们进一步将所提出的方法扩展到视频。在基准数据集上的实验表明，在人脸验证和识别任务中具有优越的性能，并且与商业FR系统相比具有更高的效率。我们的代码和模型将在https://github.com/fahadshamshad/deep-facial-privacy-prior上提供



## **49. GAIM: Attacking Graph Neural Networks via Adversarial Influence Maximization**

GAIM：通过对抗影响最大化攻击图神经网络 cs.LG

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10948v1) [paper-pdf](http://arxiv.org/pdf/2408.10948v1)

**Authors**: Xiaodong Yang, Xiaoting Li, Huiyuan Chen, Yiwei Cai

**Abstract**: Recent studies show that well-devised perturbations on graph structures or node features can mislead trained Graph Neural Network (GNN) models. However, these methods often overlook practical assumptions, over-rely on heuristics, or separate vital attack components. In response, we present GAIM, an integrated adversarial attack method conducted on a node feature basis while considering the strict black-box setting. Specifically, we define an adversarial influence function to theoretically assess the adversarial impact of node perturbations, thereby reframing the GNN attack problem into the adversarial influence maximization problem. In our approach, we unify the selection of the target node and the construction of feature perturbations into a single optimization problem, ensuring a unique and consistent feature perturbation for each target node. We leverage a surrogate model to transform this problem into a solvable linear programming task, streamlining the optimization process. Moreover, we extend our method to accommodate label-oriented attacks, broadening its applicability. Thorough evaluations on five benchmark datasets across three popular models underscore the effectiveness of our method in both untargeted and label-oriented targeted attacks. Through comprehensive analysis and ablation studies, we demonstrate the practical value and efficacy inherent to our design choices.

摘要: 最近的研究表明，对图结构或节点特征的精心设计的扰动会误导训练好的图神经网络(GNN)模型。然而，这些方法往往忽略了实际的假设，过度依赖启发式方法，或者分离出重要的攻击组件。对此，我们提出了一种基于节点特征的综合对抗性攻击方法GAIM，同时考虑了严格的黑盒设置。具体地说，我们定义了一个对抗性影响函数来从理论上评估节点扰动的对抗性影响，从而将GNN攻击问题重组为对抗性影响最大化问题。在我们的方法中，我们将目标节点的选择和特征扰动的构造统一为一个优化问题，确保每个目标节点具有唯一和一致的特征扰动。我们利用代理模型将这个问题转化为一个可解的线性规划任务，从而简化了优化过程。此外，我们还扩展了我们的方法以适应面向标签的攻击，从而扩大了它的适用性。对三个流行模型上的五个基准数据集进行的全面评估强调了我们的方法在非目标攻击和面向标签的目标攻击中的有效性。通过综合分析和烧蚀研究，我们证明了我们的设计选择所固有的实用价值和功效。



## **50. A Grey-box Attack against Latent Diffusion Model-based Image Editing by Posterior Collapse**

对基于潜在扩散模型的图像编辑的灰箱攻击 cs.CV

21 pages, 7 figures, 10 tables

**SubmitDate**: 2024-08-20    [abs](http://arxiv.org/abs/2408.10901v1) [paper-pdf](http://arxiv.org/pdf/2408.10901v1)

**Authors**: Zhongliang Guo, Lei Fang, Jingyu Lin, Yifei Qian, Shuai Zhao, Zeyu Wang, Junhao Dong, Cunjian Chen, Ognjen Arandjelović, Chun Pong Lau

**Abstract**: Recent advancements in generative AI, particularly Latent Diffusion Models (LDMs), have revolutionized image synthesis and manipulation. However, these generative techniques raises concerns about data misappropriation and intellectual property infringement. Adversarial attacks on machine learning models have been extensively studied, and a well-established body of research has extended these techniques as a benign metric to prevent the underlying misuse of generative AI. Current approaches to safeguarding images from manipulation by LDMs are limited by their reliance on model-specific knowledge and their inability to significantly degrade semantic quality of generated images. In response to these shortcomings, we propose the Posterior Collapse Attack (PCA) based on the observation that VAEs suffer from posterior collapse during training. Our method minimizes dependence on the white-box information of target models to get rid of the implicit reliance on model-specific knowledge. By accessing merely a small amount of LDM parameters, in specific merely the VAE encoder of LDMs, our method causes a substantial semantic collapse in generation quality, particularly in perceptual consistency, and demonstrates strong transferability across various model architectures. Experimental results show that PCA achieves superior perturbation effects on image generation of LDMs with lower runtime and VRAM. Our method outperforms existing techniques, offering a more robust and generalizable solution that is helpful in alleviating the socio-technical challenges posed by the rapidly evolving landscape of generative AI.

摘要: 生成性人工智能的最新进展，特别是潜在扩散模型(LDM)，已经彻底改变了图像合成和处理。然而，这些生成性技术引发了人们对数据挪用和侵犯知识产权的担忧。对机器学习模型的对抗性攻击已经被广泛研究，一系列成熟的研究已经将这些技术扩展为一种良性的衡量标准，以防止潜在的生成性人工智能的滥用。当前保护图像免受LDM操纵的方法受到它们对模型特定知识的依赖以及它们无法显著降低所生成图像的语义质量的限制。针对这些不足，我们提出了后部塌陷攻击(PCA)，基于VAE在训练过程中遭受后部塌陷的观察。我们的方法最大限度地减少了对目标模型白盒信息的依赖，摆脱了对特定模型知识的隐含依赖。通过只访问少量的LDM参数，特别是LDM的VAE编码器，我们的方法导致生成质量的语义崩溃，特别是在感知一致性方面，并表现出强大的跨模型体系结构的可移植性。实验结果表明，主成分分析算法以较低的运行时间和较低的VRAM实现了较好的图像生成扰动效果。我们的方法优于现有的技术，提供了一个更健壮和更具通用性的解决方案，有助于缓解快速发展的生成性人工智能所带来的社会技术挑战。



