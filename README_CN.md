# Latest Adversarial Attack Papers
**update at 2024-01-19 11:40:53**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**

结合Adapters和Mixup有效增强文本分类预训练语言模型的对抗健壮性 cs.CL

10 pages and 2 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10111v1) [paper-pdf](http://arxiv.org/pdf/2401.10111v1)

**Authors**: Tuc Nguyen, Thai Le

**Abstract**: Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.

摘要: 已有的工作表明，使用干净的和对抗性的例子来扩充神经网络的训练数据，可以增强神经网络在对抗性攻击下的泛化能力。然而，这种培训方法往往会导致清洁投入的绩效下降。此外，它需要频繁地重新训练整个模型以考虑新的攻击类型，从而导致大量且昂贵的计算。这些限制使得对抗性训练机制变得不那么实用，特别是对于具有数百万甚至数十亿参数的复杂的预训练语言模型(PLM)。为了克服这些挑战，同时仍然利用对抗性训练的理论优势，本研究结合了两个概念：(1)适配器，它实现了参数高效的微调；(2)MIXUP，它通过对数据对的凸组合来训练NN。直观地，我们建议通过微调适配器的非数据对的凸性组合来微调PLM，其中一个用CLEAN训练，另一个用对抗性例子训练。实验表明，该方法在训练效率和预测性能之间取得了最好的折衷，无论是在有攻击还是没有攻击的情况下，与其他针对各种下游任务的基线相比。



## **2. Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**

数字中的力量：通过每例四个对抗性句子的精细调整来增强阅读理解能力 cs.CL

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.10091v1) [paper-pdf](http://arxiv.org/pdf/2401.10091v1)

**Authors**: Ariel Marcus

**Abstract**: Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.

摘要: 最近的模型已经在斯坦福问答数据集上取得了人类水平的表现，当使用F1分数来评估阅读理解任务时。然而，在一般情况下，教机器理解文本并没有得到解决。过去的研究表明，通过在上下文段落中添加一个对抗性句子，阅读理解模型的F1分数几乎下降了一半。在本文中，我用一个新的模型ELECTRA-Small复制了过去的对抗性研究，并证明了新模型的F1得分从83.9%下降到29.2%。为了提高Electra-Small对这种攻击的抵抗力，我在小队V1.1训练实例上微调了模型，并在上下文段落中附加了一到五个对抗性句子。像过去的研究一样，我发现在一个对抗性句子上的精调模型不能很好地在评估数据集上进行泛化。然而，当对四个或五个对抗性句子进行优化时，该模型在具有多个附加和预先添加的对抗性句子的大多数评估数据集上获得了超过70%的F1分数。结果表明，有了足够的例子，我们可以使模型对对手攻击具有健壮性。



## **3. HGAttack: Transferable Heterogeneous Graph Adversarial Attack**

HGAttack：可转移的异构图对抗攻击 cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09945v1) [paper-pdf](http://arxiv.org/pdf/2401.09945v1)

**Authors**: He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao

**Abstract**: Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.

摘要: 异构图神经网络（HGNN）在网络和电子商务等领域的性能越来越受到认可，在这些领域，对抗性攻击的弹性至关重要。然而，现有的对抗性攻击方法，主要是针对同构图设计的，在应用于HGNN时，由于它们解决HGNN的结构和语义复杂性的能力有限，因此不足。本文介绍了HGAttack，第一个专门针对异构图的灰盒规避攻击方法。我们设计了一个新的代理模型，以密切类似的目标HGNN的行为，并利用基于梯度的方法扰动生成。具体而言，所提出的代理模型通过提取元路径诱导的子图并应用GNN来学习每个子图中具有不同语义的节点嵌入，从而有效地利用异构信息。这种方法提高了目标HGNN上生成的攻击的可转移性，并显着降低了内存成本。对于扰动生成，我们引入了一个语义感知机制，利用子图梯度信息自主识别脆弱的边缘在一个受约束的扰动预算范围广泛的关系。我们验证HGAttack的有效性与全面的实验三个数据集，提供其产生的扰动的实证分析。HGAttack优于基线方法，在降低目标HGNN模型的性能方面表现出显著的效果，肯定了我们的方法在评估HGNN对抗性攻击的鲁棒性方面的有效性。



## **4. Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**

保持邻域相似性的泛健性图神经网络 cs.LG

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09754v1) [paper-pdf](http://arxiv.org/pdf/2401.09754v1)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou

**Abstract**: Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.

摘要: 尽管图神经网络在学习关系数据方面取得了巨大的成功，但人们已经广泛研究了图神经网络容易受到同亲图的结构攻击。在此基础上，设计了一系列健壮模型，以增强图神经网络在同亲图上的对抗健壮性。然而，基于异嗜图的漏洞对我们来说仍然是一个谜。为了弥补这一差距，本文首先探讨了图神经网络在异嗜图上的脆弱性，并从理论上证明了负分类损失的更新与基于加权聚合邻域特征的成对相似性负相关。这一理论证明解释了图攻击者倾向于在同嗜图和异嗜图上基于邻居特征的相似性而不是自我特征的相似性来连接不同的节点对的经验观察。通过这种方式，我们新颖地提出了一种新的健壮模型NSPGNN，该模型结合了双KNN图流水线来监督邻居相似性引导的传播。该传播算法利用低通滤波对正KNN图上的节点对特征进行平滑处理，利用高通滤波来区分负KNN图上的节点对特征。在同亲图和异亲图上的广泛实验验证了NSPGNN相对于最先进的方法的普遍稳健性。



## **5. Hijacking Attacks against Neural Networks by Analyzing Training Data**

基于训练数据分析的神经网络劫持攻击 cs.CR

Accepted by the 33rd USENIX Security Symposium (USENIX Security  2024); Full Version

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09740v1) [paper-pdf](http://arxiv.org/pdf/2401.09740v1)

**Authors**: Yunjie Ge, Qian Wang, Huayang Huang, Qi Li, Cong Wang, Chao Shen, Lingchen Zhao, Peipei Jiang, Zheng Fang, Shenyi Zhang

**Abstract**: Backdoors and adversarial examples are the two primary threats currently faced by deep neural networks (DNNs). Both attacks attempt to hijack the model behaviors with unintended outputs by introducing (small) perturbations to the inputs. Backdoor attacks, despite the high success rates, often require a strong assumption, which is not always easy to achieve in reality. Adversarial example attacks, which put relatively weaker assumptions on attackers, often demand high computational resources, yet do not always yield satisfactory success rates when attacking mainstream black-box models in the real world. These limitations motivate the following research question: can model hijacking be achieved more simply, with a higher attack success rate and more reasonable assumptions? In this paper, we propose CleanSheet, a new model hijacking attack that obtains the high performance of backdoor attacks without requiring the adversary to tamper with the model training process. CleanSheet exploits vulnerabilities in DNNs stemming from the training data. Specifically, our key idea is to treat part of the clean training data of the target model as "poisoned data," and capture the characteristics of these data that are more sensitive to the model (typically called robust features) to construct "triggers." These triggers can be added to any input example to mislead the target model, similar to backdoor attacks. We validate the effectiveness of CleanSheet through extensive experiments on 5 datasets, 79 normally trained models, 68 pruned models, and 39 defensive models. Results show that CleanSheet exhibits performance comparable to state-of-the-art backdoor attacks, achieving an average attack success rate (ASR) of 97.5% on CIFAR-100 and 92.4% on GTSRB, respectively. Furthermore, CleanSheet consistently maintains a high ASR, when confronted with various mainstream backdoor defenses.

摘要: 后门和敌意例子是深度神经网络(DNN)目前面临的两个主要威胁。这两种攻击都试图通过向输入引入(小)扰动来劫持具有非预期输出的模型行为。后门攻击尽管成功率很高，但往往需要强有力的假设，而这在现实中并不总是容易实现的。对抗性例子攻击对攻击者的假设相对较弱，通常需要很高的计算资源，但在攻击现实世界中的主流黑盒模型时，并不总是产生令人满意的成功率。这些局限性引发了以下研究问题：能否以更高的攻击成功率和更合理的假设更简单地实现模型劫持？在本文中，我们提出了一种新的劫持攻击模型CleanSheet，它在不要求对手篡改模型训练过程的情况下获得了后门攻击的高性能。CleanSheet利用源自训练数据的DNN中的漏洞。具体地说，我们的关键思想是将目标模型的部分干净训练数据视为“有毒数据”，并捕获这些数据中对模型更敏感的特征(通常称为稳健特征)来构建“触发器”。这些触发器可以添加到任何输入示例中，以误导目标模型，类似于后门攻击。我们通过在5个数据集、79个正常训练模型、68个剪枝模型和39个防御模型上的大量实验验证了Clear Sheet的有效性。结果表明，CleanSheet的攻击性能与最先进的后门攻击相当，在CIFAR-100和GTSRB上的平均攻击成功率分别达到97.5%和92.4%。此外，当面对各种主流的后门防御时，CleanSheet始终保持着较高的ASR。



## **6. X-CANIDS: Signal-Aware Explainable Intrusion Detection System for Controller Area Network-Based In-Vehicle Network**

X-CANIDS：基于控制器局域网的车载网络信号感知可解释入侵检测系统 cs.CR

This is the Accepted version of an article for publication in IEEE  TVT

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2303.12278v2) [paper-pdf](http://arxiv.org/pdf/2303.12278v2)

**Authors**: Seonghoon Jeong, Sangho Lee, Hwejae Lee, Huy Kang Kim

**Abstract**: Controller Area Network (CAN) is an essential networking protocol that connects multiple electronic control units (ECUs) in a vehicle. However, CAN-based in-vehicle networks (IVNs) face security risks owing to the CAN mechanisms. An adversary can sabotage a vehicle by leveraging the security risks if they can access the CAN bus. Thus, recent actions and cybersecurity regulations (e.g., UNR 155) require carmakers to implement intrusion detection systems (IDSs) in their vehicles. The IDS should detect cyberattacks and provide additional information to analyze conducted attacks. Although many IDSs have been proposed, considerations regarding their feasibility and explainability remain lacking. This study proposes X-CANIDS, which is a novel IDS for CAN-based IVNs. X-CANIDS dissects the payloads in CAN messages into human-understandable signals using a CAN database. The signals improve the intrusion detection performance compared with the use of bit representations of raw payloads. These signals also enable an understanding of which signal or ECU is under attack. X-CANIDS can detect zero-day attacks because it does not require any labeled dataset in the training phase. We confirmed the feasibility of the proposed method through a benchmark test on an automotive-grade embedded device with a GPU. The results of this work will be valuable to carmakers and researchers considering the installation of in-vehicle IDSs for their vehicles.

摘要: 控制器局域网(CAN)是连接车辆中多个电子控制单元(ECU)的基本网络协议。然而，由于CAN机制的存在，基于CAN的车载网络面临着安全隐患。如果对手可以访问CAN总线，则他们可以利用安全风险来破坏车辆。因此，最近的行动和网络安全法规(例如，UNR 155)要求汽车制造商在其车辆中安装入侵检测系统(IDS)。入侵检测系统应该检测网络攻击，并提供其他信息来分析进行的攻击。虽然已经提出了许多入侵检测系统，但仍然缺乏对其可行性和可解释性的考虑。本研究提出了一种新型的基于CAN网络的入侵检测系统--X-CANID。X-Canids使用CAN数据库将CAN消息中的有效载荷分解为人类可理解的信号。与使用原始有效载荷的比特表示相比，该信号提高了入侵检测性能。这些信号还使您能够了解哪个信号或ECU受到攻击。X-CARID可以检测零日攻击，因为它在训练阶段不需要任何标记的数据集。通过在一款搭载GPU的车载嵌入式设备上的基准测试，验证了该方法的可行性。这项工作的结果将对汽车制造商和考虑为其车辆安装车载入侵检测系统的研究人员具有价值。



## **7. Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**

基于局部自适应对抗性色彩攻击的艺术品神经风格转移保护 cs.CV

9 pages, 5 figures

**SubmitDate**: 2024-01-18    [abs](http://arxiv.org/abs/2401.09673v1) [paper-pdf](http://arxiv.org/pdf/2401.09673v1)

**Authors**: Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang

**Abstract**: Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.

摘要: 神经样式转移(NST)是计算机视觉中广泛采用的一种生成任意样式新图像的方法。该过程利用神经网络将样式图像的美学元素与内容图像的结构方面合并为和谐集成的视觉结果。然而，未经授权的NST可以利用艺术品。这种滥用引起了对艺术家权利的社会技术关切，并促使开发技术方法来积极保护原创作品。对抗性攻击是机器学习安全领域主要探讨的一个概念。我们的作品引入了这种技术来保护艺术家的知识产权。在本文中，局部自适应对抗性颜色攻击(LAACA)是一种以人眼无法察觉的方式改变图像的方法，但对NST具有破坏性。具体地说，我们针对高频内容丰富的图像区域设计扰动，这些区域是通过破坏中间特征而产生的。我们的实验和用户研究证实，使用该方法攻击NST会导致视觉上较差的神经风格迁移，从而使其成为视觉艺术品保护的有效解决方案。



## **8. MITS-GAN: Safeguarding Medical Imaging from Tampering with Generative Adversarial Networks**

MITS-GAN：保护医学影像不受生成性对抗网络的篡改 eess.IV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09624v1) [paper-pdf](http://arxiv.org/pdf/2401.09624v1)

**Authors**: Giovanni Pasqualino, Luca Guarnera, Alessandro Ortis, Sebastiano Battiato

**Abstract**: The progress in generative models, particularly Generative Adversarial Networks (GANs), opened new possibilities for image generation but raised concerns about potential malicious uses, especially in sensitive areas like medical imaging. This study introduces MITS-GAN, a novel approach to prevent tampering in medical images, with a specific focus on CT scans. The approach disrupts the output of the attacker's CT-GAN architecture by introducing imperceptible but yet precise perturbations. Specifically, the proposed approach involves the introduction of appropriate Gaussian noise to the input as a protective measure against various attacks. Our method aims to enhance tamper resistance, comparing favorably to existing techniques. Experimental results on a CT scan dataset demonstrate MITS-GAN's superior performance, emphasizing its ability to generate tamper-resistant images with negligible artifacts. As image tampering in medical domains poses life-threatening risks, our proactive approach contributes to the responsible and ethical use of generative models. This work provides a foundation for future research in countering cyber threats in medical imaging. Models and codes are publicly available at the following link \url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.

摘要: 这项研究介绍了一种新的防止医学图像篡改的方法MITS-GaN，重点介绍了CT扫描。该方法通过引入难以察觉但却精确的扰动，破坏了攻击者的CT-GaN体系结构的输出。具体地说，所提出的方法包括在输入中引入适当的高斯噪声作为对各种攻击的保护措施。我们的方法旨在增强抗篡改能力，与现有技术相比是有利的。由于医学领域的图像篡改带来了危及生命的风险，我们积极主动的方法有助于负责任和合乎道德地使用生殖模型。这项工作为未来在医学成像中对抗网络威胁的研究提供了基础。型号和代码可在以下链接\url{https://iplab.dmi.unict.it/MITS-GAN-2024/}.上公开获得



## **9. Towards Scalable and Robust Model Versioning**

走向可扩展和健壮的模型版本控制 cs.LG

Accepted in IEEE SaTML 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09574v1) [paper-pdf](http://arxiv.org/pdf/2401.09574v1)

**Authors**: Wenxin Ding, Arjun Nitin Bhagoji, Ben Y. Zhao, Haitao Zheng

**Abstract**: As the deployment of deep learning models continues to expand across industries, the threat of malicious incursions aimed at gaining access to these deployed models is on the rise. Should an attacker gain access to a deployed model, whether through server breaches, insider attacks, or model inversion techniques, they can then construct white-box adversarial attacks to manipulate the model's classification outcomes, thereby posing significant risks to organizations that rely on these models for critical tasks. Model owners need mechanisms to protect themselves against such losses without the necessity of acquiring fresh training data - a process that typically demands substantial investments in time and capital.   In this paper, we explore the feasibility of generating multiple versions of a model that possess different attack properties, without acquiring new training data or changing model architecture. The model owner can deploy one version at a time and replace a leaked version immediately with a new version. The newly deployed model version can resist adversarial attacks generated leveraging white-box access to one or all previously leaked versions. We show theoretically that this can be accomplished by incorporating parameterized hidden distributions into the model training data, forcing the model to learn task-irrelevant features uniquely defined by the chosen data. Additionally, optimal choices of hidden distributions can produce a sequence of model versions capable of resisting compound transferability attacks over time. Leveraging our analytical insights, we design and implement a practical model versioning method for DNN classifiers, which leads to significant robustness improvements over existing methods. We believe our work presents a promising direction for safeguarding DNN services beyond their initial deployment.

摘要: 随着深度学习模型的部署继续跨行业扩展，旨在访问这些已部署模型的恶意入侵威胁正在上升。如果攻击者获得对已部署模型的访问权限，无论是通过服务器入侵、内部攻击或模型倒置技术，他们都可以构建白盒对抗性攻击来操纵模型的分类结果，从而给依赖这些模型执行关键任务的组织带来重大风险。模型所有者需要机制来保护自己免受此类损失，而不需要获取新的培训数据-这一过程通常需要在时间和资金上进行大量投资。在本文中，我们探索了在不获取新的训练数据或改变模型体系结构的情况下，生成具有不同攻击属性的模型的多个版本的可行性。模型所有者可以一次部署一个版本，并立即用新版本替换泄漏的版本。新部署的模型版本可以抵抗利用白盒访问一个或所有先前泄露的版本而产生的对抗性攻击。我们从理论上证明，这可以通过将参数化的隐藏分布结合到模型训练数据中来实现，迫使模型学习由所选数据唯一定义的与任务无关的特征。此外，隐藏分布的最佳选择可以产生一系列模型版本，能够随着时间的推移抵抗复合可转移性攻击。利用我们的分析洞察力，我们设计并实现了一种实用的DNN分类器模型版本控制方法，与现有方法相比，该方法具有显著的健壮性改进。我们相信，我们的工作为保护DNN服务提供了一个很有前途的方向，而不是最初的部署。



## **10. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

基于扩散的改进隐蔽性和可控性的对抗样本生成 cs.CV

Accepted as a conference paper in NeurIPS'2023. Code repo:  https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2305.16494v3) [paper-pdf](http://arxiv.org/pdf/2305.16494v3)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD

摘要: 众所周知，神经网络容易受到敌意样本的影响，这些样本是自然样本的微小变体，目的是故意误导模型。虽然在数字和物理场景中可以很容易地使用基于梯度的技术来生成它们，但它们往往与自然图像的实际数据分布有很大差异，导致在强度和隐蔽性之间进行权衡。在本文中，我们提出了一种新的框架，称为基于扩散的投影梯度下降(DIFF-PGD)，用于生成真实的对抗性样本。通过利用扩散模型引导的梯度，DIFF-PGD在保持有效性的同时，确保对手样本保持接近原始数据分布。此外，我们的框架可以很容易地针对特定任务进行定制，例如数字攻击、物理世界攻击和基于样式的攻击。与现有的生成自然风格对抗性样本的方法相比，我们的框架能够将优化对抗性损失与其他代理损失(例如，内容/流畅度/风格损失)分离，使其更加稳定和可控。最后，我们证明了DIFF-PGD生成的样本比传统的基于梯度的方法具有更好的可转移性和抗净化能力。代码将在https://github.com/xavihart/Diff-PGD中发布



## **11. Adversarial Examples are Misaligned in Diffusion Model Manifolds**

扩散模型流形中对抗性例子的错位 cs.CV

under review

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.06637v3) [paper-pdf](http://arxiv.org/pdf/2401.06637v3)

**Authors**: Peter Lorenz, Ricard Durall, Janis Keuper

**Abstract**: In recent years, diffusion models (DMs) have drawn significant attention for their success in approximating data distributions, yielding state-of-the-art generative results. Nevertheless, the versatility of these models extends beyond their generative capabilities to encompass various vision applications, such as image inpainting, segmentation, adversarial robustness, among others. This study is dedicated to the investigation of adversarial attacks through the lens of diffusion models. However, our objective does not involve enhancing the adversarial robustness of image classifiers. Instead, our focus lies in utilizing the diffusion model to detect and analyze the anomalies introduced by these attacks on images. To that end, we systematically examine the alignment of the distributions of adversarial examples when subjected to the process of transformation using diffusion models. The efficacy of this approach is assessed across CIFAR-10 and ImageNet datasets, including varying image sizes in the latter. The results demonstrate a notable capacity to discriminate effectively between benign and attacked images, providing compelling evidence that adversarial instances do not align with the learned manifold of the DMs.

摘要: 近年来，扩散模型(DM)因其在近似数据分布方面的成功而引起了人们的极大关注，产生了最先进的生成结果。然而，这些模型的多功能性超出了它们的生成能力，涵盖了各种视觉应用，如图像修复、分割、对抗性鲁棒性等。本研究致力于从扩散模型的角度研究对抗性攻击。然而，我们的目标不涉及增强图像分类器的对抗性稳健性。相反，我们的重点在于利用扩散模型来检测和分析这些攻击对图像带来的异常。为此，我们使用扩散模型系统地考察了对抗性例子在经历转换过程时的分布的一致性。在CIFAR-10和ImageNet数据集上评估了这种方法的有效性，包括在后者中不同的图像大小。实验结果表明，该方法能够有效地区分良性图像和被攻击图像，提供了令人信服的证据，表明敌意实例与学习到的DM流形并不一致。



## **12. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

MIMIR：基于交互信息的对抗性掩蔽图像建模 cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2312.04960v2) [paper-pdf](http://arxiv.org/pdf/2312.04960v2)

**Authors**: Xiaoyun Xu, Shujian Yu, Jingzheng Wu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) achieve superior performance on various tasks compared to convolutional neural networks (CNNs), but ViTs are also vulnerable to adversarial attacks. Adversarial training is one of the most successful methods to build robust CNN models. Thus, recent works explored new methodologies for adversarial training of ViTs based on the differences between ViTs and CNNs, such as better training strategies, preventing attention from focusing on a single block, or discarding low-attention embeddings. However, these methods still follow the design of traditional supervised adversarial training, limiting the potential of adversarial training on ViTs. This paper proposes a novel defense method, MIMIR, which aims to build a different adversarial training methodology by utilizing Masked Image Modeling at pre-training. We create an autoencoder that accepts adversarial examples as input but takes the clean examples as the modeling target. Then, we create a mutual information (MI) penalty following the idea of the Information Bottleneck. Among the two information source inputs and corresponding adversarial perturbation, the perturbation information is eliminated due to the constraint of the modeling target. Next, we provide a theoretical analysis of MIMIR using the bounds of the MI penalty. We also design two adaptive attacks when the adversary is aware of the MIMIR defense and show that MIMIR still performs well. The experimental results show that MIMIR improves (natural and adversarial) accuracy on average by 4.19% on CIFAR-10 and 5.52% on ImageNet-1K, compared to baselines. On Tiny-ImageNet, we obtained improved natural accuracy of 2.99\% on average and comparable adversarial accuracy. Our code and trained models are publicly available https://github.com/xiaoyunxxy/MIMIR.

摘要: 与卷积神经网络（CNN）相比，视觉变换器（ViTs）在各种任务上都具有卓越的性能，但ViTs也容易受到对抗性攻击。对抗训练是构建强大CNN模型的最成功方法之一。因此，最近的工作基于ViTs和CNN之间的差异探索了ViTs对抗性训练的新方法，例如更好的训练策略，防止注意力集中在单个块上，或丢弃低注意力嵌入。然而，这些方法仍然遵循传统监督对抗训练的设计，限制了对抗训练在ViT上的潜力。本文提出了一种新的防御方法MIMIR，其目的是通过在预训练时利用掩蔽图像建模来构建一种不同的对抗训练方法。我们创建了一个自动编码器，它接受对抗性示例作为输入，但将干净的示例作为建模目标。然后，我们创建一个互信息（MI）的惩罚信息瓶颈的想法。在两个信息源输入和相应的对抗扰动中，由于建模目标的约束，扰动信息被消除。接下来，我们提供了一个理论分析MIMIR使用MI惩罚的界限。我们还设计了两个自适应攻击时，对手是知道的MIMIR防御和MIMIR仍然表现良好。实验结果表明，与基线相比，MIMIR在CIFAR-10上平均提高了4.19%，在ImageNet-1 K上平均提高了5.52%。在Tiny-ImageNet上，我们获得了平均2.99%的自然准确率和可比的对抗准确率。我们的代码和训练模型可在https://github.com/xiaoyunxxy/MIMIR上公开获取。



## **13. Username Squatting on Online Social Networks: A Study on X**

基于X的在线社交网络用户名蹲点行为研究 cs.CR

Accepted at ACM ASIA Conference on Computer and Communications  Security (AsiaCCS), 2024

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.09209v1) [paper-pdf](http://arxiv.org/pdf/2401.09209v1)

**Authors**: Anastasios Lepipas, Anastasia Borovykh, Soteris Demetriou

**Abstract**: Adversaries have been targeting unique identifiers to launch typo-squatting, mobile app squatting and even voice squatting attacks. Anecdotal evidence suggest that online social networks (OSNs) are also plagued with accounts that use similar usernames. This can be confusing to users but can also be exploited by adversaries. However, to date no study characterizes this problem on OSNs. In this work, we define the username squatting problem and design the first multi-faceted measurement study to characterize it on X. We develop a username generation tool (UsernameCrazy) to help us analyze hundreds of thousands of username variants derived from celebrity accounts. Our study reveals that thousands of squatted usernames have been suspended by X, while tens of thousands that still exist on the network are likely bots. Out of these, a large number share similar profile pictures and profile names to the original account signalling impersonation attempts. We found that squatted accounts are being mentioned by mistake in tweets hundreds of thousands of times and are even being prioritized in searches by the network's search recommendation algorithm exacerbating the negative impact squatted accounts can have in OSNs. We use our insights and take the first step to address this issue by designing a framework (SQUAD) that combines UsernameCrazy with a new classifier to efficiently detect suspicious squatted accounts. Our evaluation of SQUAD's prototype implementation shows that it can achieve 94% F1-score when trained on a small dataset.

摘要: 对手一直以唯一标识为目标，发动打字蹲守、手机应用蹲守，甚至语音蹲守攻击。坊间证据表明，在线社交网络(OSN)也充斥着使用相似用户名的账户。这可能会让用户感到困惑，但也可能被对手利用。然而，到目前为止，还没有研究描述OSN上的这个问题。在这项工作中，我们定义了用户名下蹲问题，并设计了第一个多方面的测量研究来刻画X上的用户名下蹲问题。我们开发了一个用户名生成工具(UsernameCrazy)来帮助我们分析来自名人账户的数十万个用户名变体。我们的研究显示，数以千计的蹲守用户名已经被X暂停，而网络上仍然存在的数万个用户名很可能是机器人。在这些中，有大量共享与原始帐户信令模拟尝试相似的配置文件图片和配置文件名称。我们发现，在推文中，蹲着的账户被错误地提到了数十万次，甚至在网络的搜索推荐算法的搜索中被优先考虑，加剧了蹲着的账户在OSN中可能产生的负面影响。我们利用我们的见解，通过设计一个框架(Team)来解决这个问题，该框架(Team)将UsernameCrazy与新的分类器相结合，以高效地检测可疑的蹲守帐户。我们对LONG原型实现的评估表明，当在小数据集上训练时，它可以达到94%的F1得分。



## **14. Attack and Reset for Unlearning: Exploiting Adversarial Noise toward Machine Unlearning through Parameter Re-initialization**

遗忘的攻击和重置：通过参数重新初始化利用机器遗忘的对抗性噪声 cs.LG

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08998v1) [paper-pdf](http://arxiv.org/pdf/2401.08998v1)

**Authors**: Yoonhwa Jung, Ikhyun Cho, Shun-Hsiang Hsu, Julia Hockenmaier

**Abstract**: With growing concerns surrounding privacy and regulatory compliance, the concept of machine unlearning has gained prominence, aiming to selectively forget or erase specific learned information from a trained model. In response to this critical need, we introduce a novel approach called Attack-and-Reset for Unlearning (ARU). This algorithm leverages meticulously crafted adversarial noise to generate a parameter mask, effectively resetting certain parameters and rendering them unlearnable. ARU outperforms current state-of-the-art results on two facial machine-unlearning benchmark datasets, MUFAC and MUCAC. In particular, we present the steps involved in attacking and masking that strategically filter and re-initialize network parameters biased towards the forget set. Our work represents a significant advancement in rendering data unexploitable to deep learning models through parameter re-initialization, achieved by harnessing adversarial noise to craft a mask.

摘要: 随着人们对隐私和监管合规性的日益关注，机器遗忘的概念变得突出起来，旨在有选择地忘记或擦除来自训练模型的特定学习信息。针对这一关键需求，我们引入了一种新的方法，称为遗忘攻击和重置(ARU)。该算法利用精心设计的对抗性噪声来生成参数掩码，有效地重置某些参数并使其无法学习。ARU在两个人脸机器遗忘基准数据集MUFAC和MUCAC上的性能优于当前最先进的结果。特别是，我们给出了攻击和掩蔽所涉及的步骤，这些步骤策略性地过滤和重新初始化偏向遗忘集的网络参数。我们的工作代表着在通过参数重新初始化来呈现深度学习模型无法利用的数据方面取得了重大进展，这是通过利用对抗性噪声来制作掩码实现的。



## **15. A GAN-based data poisoning framework against anomaly detection in vertical federated learning**

垂直联邦学习中基于GAN的异常检测数据中毒框架 cs.LG

6 pages, 7 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08984v1) [paper-pdf](http://arxiv.org/pdf/2401.08984v1)

**Authors**: Xiaolin Chen, Daoguang Zan, Wei Li, Bei Guan, Yongji Wang

**Abstract**: In vertical federated learning (VFL), commercial entities collaboratively train a model while preserving data privacy. However, a malicious participant's poisoning attack may degrade the performance of this collaborative model. The main challenge in achieving the poisoning attack is the absence of access to the server-side top model, leaving the malicious participant without a clear target model. To address this challenge, we introduce an innovative end-to-end poisoning framework P-GAN. Specifically, the malicious participant initially employs semi-supervised learning to train a surrogate target model. Subsequently, this participant employs a GAN-based method to produce adversarial perturbations to degrade the surrogate target model's performance. Finally, the generator is obtained and tailored for VFL poisoning. Besides, we develop an anomaly detection algorithm based on a deep auto-encoder (DAE), offering a robust defense mechanism to VFL scenarios. Through extensive experiments, we evaluate the efficacy of P-GAN and DAE, and further analyze the factors that influence their performance.

摘要: 在垂直联合学习(VFL)中，商业实体在保护数据隐私的同时协作训练模型。然而，恶意参与者的中毒攻击可能会降低该协作模型的性能。实现中毒攻击的主要挑战是无法访问服务器端的顶层模型，使恶意参与者没有明确的目标模型。为了应对这一挑战，我们引入了一个创新的端到端中毒框架P-GaN。具体地说，恶意参与者最初使用半监督学习来训练代理目标模型。随后，该参与者使用基于遗传算法的方法来产生对抗性扰动以降低代理目标模型的性能。最后，获得了用于VFL中毒的发生器并进行了定制。此外，我们还开发了一种基于深度自动编码器的异常检测算法，为VFL场景提供了一种健壮的防御机制。通过大量的实验，我们对P-GaN和DAE的性能进行了评估，并进一步分析了影响它们性能的因素。



## **16. RandOhm: Mitigating Impedance Side-channel Attacks using Randomized Circuit Configurations**

RandOhm：使用随机化电路配置缓解阻抗旁通道攻击 cs.CR

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08925v1) [paper-pdf](http://arxiv.org/pdf/2401.08925v1)

**Authors**: Saleh Khalaj Monfared, Domenic Forte, Shahin Tajik

**Abstract**: Physical side-channel attacks can compromise the security of integrated circuits. Most of the physical side-channel attacks (e.g., power or electromagnetic) exploit the dynamic behavior of a chip, typically manifesting as changes in current consumption or voltage fluctuations where algorithmic countermeasures, such as masking, can effectively mitigate the attacks. However, as demonstrated recently, these mitigation techniques are not entirely effective against backscattered side-channel attacks such as impedance analysis. In the case of an impedance attack, an adversary exploits the data-dependent impedance variations of chip power delivery network (PDN) to extract secret information. In this work, we introduce RandOhm, which exploits moving target defense (MTD) strategy based on partial reconfiguration of mainstream FPGAs, to defend against impedance side-channel attacks. We demonstrate that the information leakage through the PDN impedance could be reduced via run-time reconfiguration of the secret-sensitive parts of the circuitry. Hence, by constantly randomizing the placement and routing of the circuit, one can decorrelate the data-dependent computation from the impedance value. To validate our claims, we present a systematic approach equipped with two different partial reconfiguration strategies on implementations of the AES cipher realized on 28-nm FPGAs. We investigate the overhead of our mitigation in terms of delay and performance and provide security analysis by performing non-profiled and profiled impedance analysis attacks against these implementations to demonstrate the resiliency of our approach.

摘要: 物理侧通道攻击可能会危及集成电路的安全性。大多数物理侧通道攻击(例如，功率或电磁)利用芯片的动态行为，通常表现为电流消耗或电压波动的变化，其中算法对策，如掩蔽，可以有效地缓解攻击。然而，正如最近所证明的那样，这些缓解技术并不能完全有效地对抗诸如阻抗分析之类的反向散射侧信道攻击。在阻抗攻击的情况下，敌手利用芯片功率传输网络(PDN)与数据相关的阻抗变化来提取秘密信息。在这项工作中，我们引入了RandOhm，它利用基于主流现场可编程门阵列部分重构的移动目标防御(MTD)策略来防御阻抗旁通道攻击。我们证明，通过PDN阻抗的信息泄漏可以通过运行时重新配置电路的秘密敏感部分来减少。因此，通过不断地随机化电路的布局和布线，可以将依赖于数据的计算与阻抗值分离。为了验证我们的主张，我们提出了一种系统的方法，该方法配备了两种不同的部分重构策略，以实现在28 nm FPGA上实现的AES密码。我们在延迟和性能方面调查了缓解的开销，并通过对这些实施执行非配置文件和配置文件阻抗分析攻击来提供安全分析，以展示我们方法的弹性。



## **17. PPR: Enhancing Dodging Attacks while Maintaining Impersonation Attacks on Face Recognition Systems**

PPR：增强对人脸识别系统的躲避攻击，同时保持模仿攻击 cs.CV

**SubmitDate**: 2024-01-17    [abs](http://arxiv.org/abs/2401.08903v1) [paper-pdf](http://arxiv.org/pdf/2401.08903v1)

**Authors**: Fengfan Zhou, Heifei Ling

**Abstract**: Adversarial Attacks on Face Recognition (FR) encompass two types: impersonation attacks and evasion attacks. We observe that achieving a successful impersonation attack on FR does not necessarily ensure a successful dodging attack on FR in the black-box setting. Introducing a novel attack method named Pre-training Pruning Restoration Attack (PPR), we aim to enhance the performance of dodging attacks whilst avoiding the degradation of impersonation attacks. Our method employs adversarial example pruning, enabling a portion of adversarial perturbations to be set to zero, while tending to maintain the attack performance. By utilizing adversarial example pruning, we can prune the pre-trained adversarial examples and selectively free up certain adversarial perturbations. Thereafter, we embed adversarial perturbations in the pruned area, which enhances the dodging performance of the adversarial face examples. The effectiveness of our proposed attack method is demonstrated through our experimental results, showcasing its superior performance.

摘要: 针对人脸识别(FR)的敌意攻击包括两种类型：模仿攻击和逃避攻击。我们观察到，在黑盒环境下，成功地实现对FR的模仿攻击并不一定确保对FR的成功躲避攻击。引入一种新的攻击方法--预训练剪枝恢复攻击(PPR)，旨在提高躲避攻击的性能，同时避免冒充攻击的降级。该方法采用对抗性样本剪枝，在保持攻击性能的同时，使一部分对抗性扰动被设置为零。通过利用对抗性实例修剪，我们可以修剪预先训练的对抗性实例，并选择性地释放某些对抗性扰动。之后，我们在剪枝区域嵌入对抗性扰动，提高了对抗性人脸样例的躲避性能。实验结果表明，本文提出的攻击方法是有效的，表现出了优越的性能。



## **18. Whispering Pixels: Exploiting Uninitialized Register Accesses in Modern GPUs**

低语像素：利用现代GPU中未初始化的寄存器访问 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08881v1) [paper-pdf](http://arxiv.org/pdf/2401.08881v1)

**Authors**: Frederik Dermot Pustelnik, Xhani Marvin Saß, Jean-Pierre Seifert

**Abstract**: Graphic Processing Units (GPUs) have transcended their traditional use-case of rendering graphics and nowadays also serve as a powerful platform for accelerating ubiquitous, non-graphical rendering tasks. One prominent task is inference of neural networks, which process vast amounts of personal data, such as audio, text or images. Thus, GPUs became integral components for handling vast amounts of potentially confidential data, which has awakened the interest of security researchers. This lead to the discovery of various vulnerabilities in GPUs in recent years. In this paper, we uncover yet another vulnerability class in GPUs: We found that some GPU implementations lack proper register initialization routines before shader execution, leading to unintended register content leakage of previously executed shader kernels. We showcase the existence of the aforementioned vulnerability on products of 3 major vendors - Apple, NVIDIA and Qualcomm. The vulnerability poses unique challenges to an adversary due to opaque scheduling and register remapping algorithms present in the GPU firmware, complicating the reconstruction of leaked data. In order to illustrate the real-world impact of this flaw, we showcase how these challenges can be solved for attacking various workloads on the GPU. First, we showcase how uninitialized registers leak arbitrary pixel data processed by fragment shaders. We further implement information leakage attacks on intermediate data of Convolutional Neural Networks (CNNs) and present the attack's capability to leak and reconstruct the output of Large Language Models (LLMs).

摘要: 图形处理单元(GPU)已经超越了渲染图形的传统用例，如今也成为加速无处不在的非图形渲染任务的强大平台。一项突出的任务是神经网络的推理，它处理大量的个人数据，如音频、文本或图像。因此，GPU成为处理海量潜在机密数据的不可或缺的组件，这唤醒了安全研究人员的兴趣。这导致了近年来GPU中各种漏洞的发现。在本文中，我们发现了GPU中的另一个漏洞类别：我们发现一些GPU实现在着色器执行之前缺乏适当的寄存器初始化例程，导致先前执行的着色器内核的意外寄存器内容泄漏。我们展示了3家主要供应商的产品上存在上述漏洞-苹果、NVIDIA和高通。由于GPU固件中存在不透明的调度和寄存器重新映射算法，该漏洞对对手构成了独特的挑战，使泄漏数据的重建复杂化。为了说明该漏洞的实际影响，我们展示了如何解决这些挑战来攻击GPU上的各种工作负载。首先，我们展示了未初始化的寄存器如何泄漏由片段着色器处理的任意像素数据。在此基础上，对卷积神经网络(CNN)的中间数据进行了信息泄漏攻击，并给出了该攻击对大语言模型(LLM)输出的泄漏和重构能力。



## **19. The Effect of Intrinsic Dataset Properties on Generalization: Unraveling Learning Differences Between Natural and Medical Images**

数据集属性对概化的影响：解开自然图像和医学图像之间的学习差异 cs.CV

ICLR 2024. Code:  https://github.com/mazurowski-lab/intrinsic-properties

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08865v1) [paper-pdf](http://arxiv.org/pdf/2401.08865v1)

**Authors**: Nicholas Konz, Maciej A. Mazurowski

**Abstract**: This paper investigates discrepancies in how neural networks learn from different imaging domains, which are commonly overlooked when adopting computer vision techniques from the domain of natural images to other specialized domains such as medical images. Recent works have found that the generalization error of a trained network typically increases with the intrinsic dimension ($d_{data}$) of its training set. Yet, the steepness of this relationship varies significantly between medical (radiological) and natural imaging domains, with no existing theoretical explanation. We address this gap in knowledge by establishing and empirically validating a generalization scaling law with respect to $d_{data}$, and propose that the substantial scaling discrepancy between the two considered domains may be at least partially attributed to the higher intrinsic "label sharpness" ($K_F$) of medical imaging datasets, a metric which we propose. Next, we demonstrate an additional benefit of measuring the label sharpness of a training set: it is negatively correlated with the trained model's adversarial robustness, which notably leads to models for medical images having a substantially higher vulnerability to adversarial attack. Finally, we extend our $d_{data}$ formalism to the related metric of learned representation intrinsic dimension ($d_{repr}$), derive a generalization scaling law with respect to $d_{repr}$, and show that $d_{data}$ serves as an upper bound for $d_{repr}$. Our theoretical results are supported by thorough experiments with six models and eleven natural and medical imaging datasets over a range of training set sizes. Our findings offer insights into the influence of intrinsic dataset properties on generalization, representation learning, and robustness in deep neural networks.

摘要: 本文研究了神经网络如何从不同的成像领域学习的差异，这些差异是在将计算机视觉技术从自然图像领域应用到其他专业领域(如医学图像)时通常被忽视的。最近的工作发现，训练网络的泛化误差通常随着训练集的固有维度($d_{data}$)的增加而增加。然而，这种关系的陡峭程度在医学(放射)和自然成像领域有很大的不同，没有现有的理论解释。我们通过建立和经验性地验证关于$d{data}$的泛化标度律来解决这一知识缺口，并提出两个被考虑的域之间的显著标度差异至少部分归因于医学成像数据集的更高的固有“标签锐度”($K_F$)，这是我们提出的一种度量。接下来，我们展示了测量训练集的标签锐度的另一个好处：它与训练模型的对抗稳健性负相关，这显著地导致医学图像的模型具有更高的对抗攻击脆弱性。最后，我们将$d_{data}$形式推广到学习表示内在维的相关度量($d_{epr}$)，得到了关于$d_{epr}$的一个推广的标度律，并证明了$d_{data}$是$d_{epr}$的一个上界。我们的理论结果得到了6个模型和11个自然和医学成像数据集的全面实验的支持，这些数据集的训练集大小不同。我们的发现对深入神经网络中内在数据集属性对泛化、表示学习和稳健性的影响提供了深入的见解。



## **20. Game-Theoretic Neyman-Pearson Detection to Combat Strategic Evasion**

对抗战略规避的博弈论Neyman-Pearson检测 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2206.05276v2) [paper-pdf](http://arxiv.org/pdf/2206.05276v2)

**Authors**: Yinan Hu, Quanyan Zhu

**Abstract**: The security in networked systems depends greatly on recognizing and identifying adversarial behaviors. Traditional detection methods focus on specific categories of attacks and have become inadequate for increasingly stealthy and deceptive attacks that are designed to bypass detection strategically. This work aims to develop a holistic theory to countermeasure such evasive attacks. We focus on extending a fundamental class of statistical-based detection methods based on Neyman-Pearson's (NP) hypothesis testing formulation. We propose game-theoretic frameworks to capture the conflicting relationship between a strategic evasive attacker and an evasion-aware NP detector. By analyzing both the equilibrium behaviors of the attacker and the NP detector, we characterize their performance using Equilibrium Receiver-Operational-Characteristic (EROC) curves. We show that the evasion-aware NP detectors outperform the passive ones in the way that the former can act strategically against the attacker's behavior and adaptively modify their decision rules based on the received messages. In addition, we extend our framework to a sequential setting where the user sends out identically distributed messages. We corroborate the analytical results with a case study of anomaly detection.

摘要: 网络系统的安全性在很大程度上取决于对敌方行为的识别和识别。这项工作旨在开发一种整体理论来对抗这种规避攻击。基于Neyman-Pearson(NP)假设检验公式，我们重点扩展了一类基本的基于统计的检测方法。我们提出了博弈论框架来捕捉战略规避攻击者和规避感知NP检测器之间的冲突关系。我们证明了逃避感知NP检测器的性能优于被动NP检测器，前者可以针对攻击者的行为采取策略性行动，并根据收到的消息自适应地修改其决策规则。此外，我们将我们的框架扩展到顺序设置，在该设置中，用户发送相同分布的消息。我们通过一个异常检测的案例验证了分析结果。



## **21. Benchmarking the Robustness of Image Watermarks**

图像水印稳健性的基准测试 cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08573v1) [paper-pdf](http://arxiv.org/pdf/2401.08573v1)

**Authors**: Bang An, Mucong Ding, Tahseen Rabbani, Aakriti Agrawal, Yuancheng Xu, Chenghao Deng, Sicheng Zhu, Abdirisak Mohamed, Yuxin Wen, Tom Goldstein, Furong Huang

**Abstract**: This paper investigates the weaknesses of image watermarking techniques. We present WAVES (Watermark Analysis Via Enhanced Stress-testing), a novel benchmark for assessing watermark robustness, overcoming the limitations of current evaluation methods.WAVES integrates detection and identification tasks, and establishes a standardized evaluation protocol comprised of a diverse range of stress tests. The attacks in WAVES range from traditional image distortions to advanced and novel variations of adversarial, diffusive, and embedding-based attacks. We introduce a normalized score of attack potency which incorporates several widely used image quality metrics and allows us to produce of an ordered ranking of attacks. Our comprehensive evaluation over reveals previously undetected vulnerabilities of several modern watermarking algorithms. WAVES is envisioned as a toolkit for the future development of robust watermarking systems.

摘要: 本文研究了图像水印技术的弱点。本文提出了一种新的水印鲁棒性评估基准WAVES（Watermark Analysis Via Enhanced Stress-testing），它克服了现有评估方法的局限性，将检测和识别任务整合在一起，建立了一个由多种压力测试组成的标准化评估协议。WAVES中的攻击范围从传统的图像失真到对抗性、扩散性和基于嵌入的攻击的高级和新颖变体。我们引入了一个归一化的得分的攻击效力，它结合了几个广泛使用的图像质量指标，并允许我们产生的有序排名的攻击。我们的全面评估揭示了以前未被发现的几个现代水印算法的漏洞。WAVES被设想为鲁棒水印系统的未来发展的工具包。



## **22. Bag of Tricks to Boost Adversarial Transferability**

一袋诡计，提高对手的可转移性 cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08734v1) [paper-pdf](http://arxiv.org/pdf/2401.08734v1)

**Authors**: Zeliang Zhang, Rongyi Zhu, Wei Yao, Xiaosen Wang, Chenliang Xu

**Abstract**: Deep neural networks are widely known to be vulnerable to adversarial examples. However, vanilla adversarial examples generated under the white-box setting often exhibit low transferability across different models. Since adversarial transferability poses more severe threats to practical applications, various approaches have been proposed for better transferability, including gradient-based, input transformation-based, and model-related attacks, \etc. In this work, we find that several tiny changes in the existing adversarial attacks can significantly affect the attack performance, \eg, the number of iterations and step size. Based on careful studies of existing adversarial attacks, we propose a bag of tricks to enhance adversarial transferability, including momentum initialization, scheduled step size, dual example, spectral-based input transformation, and several ensemble strategies. Extensive experiments on the ImageNet dataset validate the high effectiveness of our proposed tricks and show that combining them can further boost adversarial transferability. Our work provides practical insights and techniques to enhance adversarial transferability, and offers guidance to improve the attack performance on the real-world application through simple adjustments.

摘要: 众所周知，深度神经网络很容易受到敌意例子的攻击。然而，在白盒设置下产生的普通对抗性例子往往表现出在不同模型之间的低可转移性。由于对抗性可转移性对实际应用构成了更严重的威胁，人们提出了各种方法来提高可转移性，包括基于梯度的攻击、基于输入变换的攻击和与模型相关的攻击等。在本工作中，我们发现现有对抗性攻击中的几个微小变化会显著影响攻击性能，例如迭代次数和步长。在仔细研究现有对抗性攻击的基础上，提出了一系列增强对抗性可转移性的策略，包括动量初始化、调度步长、对偶例、基于谱的输入变换和几种集成策略。在ImageNet数据集上的大量实验验证了我们提出的技巧的高效性，并表明将它们结合起来可以进一步提高对手的可转移性。我们的工作为增强对手的可转移性提供了实用的见解和技术，并为通过简单的调整提高对现实世界应用的攻击性能提供了指导。



## **23. SecPLF: Secure Protocols for Loanable Funds against Oracle Manipulation Attacks**

SecPLF：可贷资金抵御Oracle操纵攻击的安全协议 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08520v1) [paper-pdf](http://arxiv.org/pdf/2401.08520v1)

**Authors**: Sanidhay Arora, Yingjiu Li, Yebo Feng, Jiahua Xu

**Abstract**: The evolving landscape of Decentralized Finance (DeFi) has raised critical security concerns, especially pertaining to Protocols for Loanable Funds (PLFs) and their dependency on price oracles, which are susceptible to manipulation. The emergence of flash loans has further amplified these risks, enabling increasingly complex oracle manipulation attacks that can lead to significant financial losses. Responding to this threat, we first dissect the attack mechanism by formalizing the standard operational and adversary models for PLFs. Based on our analysis, we propose SecPLF, a robust and practical solution designed to counteract oracle manipulation attacks efficiently. SecPLF operates by tracking a price state for each crypto-asset, including the recent price and the timestamp of its last update. By imposing price constraints on the price oracle usage, SecPLF ensures a PLF only engages a price oracle if the last recorded price falls within a defined threshold, thereby negating the profitability of potential attacks. Our evaluation based on historical market data confirms SecPLF's efficacy in providing high-confidence prevention against arbitrage attacks that arise due to minor price differences. SecPLF delivers proactive protection against oracle manipulation attacks, offering ease of implementation, oracle-agnostic property, and resource and cost efficiency.

摘要: 去中心化金融(DEFI)不断演变的格局引发了严重的安全担忧，特别是关于可贷款资金(PLF)的协议及其对价格先知的依赖，这些先知容易受到操纵。闪电贷款的出现进一步放大了这些风险，使越来越复杂的甲骨文操纵攻击成为可能，可能导致重大经济损失。作为对这种威胁的回应，我们首先通过形式化PLF的标准作战模型和对手模型来剖析攻击机制。在此基础上，我们提出了一种健壮实用的解决方案--SecPLF，该方案能够有效地对抗Oracle操纵攻击。SecPLF通过跟踪每个加密资产的价格状态来运行，包括最近的价格和上次更新的时间戳。通过对价格预言的使用施加价格限制，SecPLF确保只有在最后记录的价格落在定义的阈值内时，PLF才会使用价格预言，从而否定潜在攻击的盈利能力。我们基于历史市场数据的评估证实了SecPLF在提供高置信度防止因微小价格差异而出现的套利攻击方面的有效性。SecPLF提供针对Oracle操纵攻击的主动保护，提供易于实施、与Oracle无关的特性以及资源和成本效益。



## **24. Revealing Vulnerabilities in Stable Diffusion via Targeted Attacks**

通过定向攻击揭示稳定扩散中的漏洞 cs.CV

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08725v1) [paper-pdf](http://arxiv.org/pdf/2401.08725v1)

**Authors**: Chenyu Zhang, Lanjun Wang, Anan Liu

**Abstract**: Recent developments in text-to-image models, particularly Stable Diffusion, have marked significant achievements in various applications. With these advancements, there are growing safety concerns about the vulnerability of the model that malicious entities exploit to generate targeted harmful images. However, the existing methods in the vulnerability of the model mainly evaluate the alignment between the prompt and generated images, but fall short in revealing the vulnerability associated with targeted image generation. In this study, we formulate the problem of targeted adversarial attack on Stable Diffusion and propose a framework to generate adversarial prompts. Specifically, we design a gradient-based embedding optimization method to craft reliable adversarial prompts that guide stable diffusion to generate specific images. Furthermore, after obtaining successful adversarial prompts, we reveal the mechanisms that cause the vulnerability of the model. Extensive experiments on two targeted attack tasks demonstrate the effectiveness of our method in targeted attacks. The code can be obtained in https://github.com/datar001/Revealing-Vulnerabilities-in-Stable-Diffusion-via-Targeted-Attacks.

摘要: 文本到图像模型的最新发展，特别是稳定扩散，在各种应用中取得了显著的成就。随着这些进步，人们越来越担心恶意实体用来生成有针对性的有害图像的模型的漏洞。然而，现有的模型脆弱性方法主要是评估提示图像和生成图像之间的一致性，而不能揭示与目标图像生成相关的脆弱性。在这项研究中，我们描述了稳定扩散上的定向对抗性攻击问题，并提出了一个生成对抗性提示的框架。具体地说，我们设计了一种基于梯度的嵌入优化方法来生成可靠的对抗性提示，以指导稳定扩散生成特定的图像。此外，在获得成功的对抗性提示后，我们揭示了导致该模型脆弱性的机制。在两个目标攻击任务上的大量实验证明了该方法在目标攻击中的有效性。代码可以在https://github.com/datar001/Revealing-Vulnerabilities-in-Stable-Diffusion-via-Targeted-Attacks.中获得



## **25. A Generative Adversarial Attack for Multilingual Text Classifiers**

一种面向多语言文本分类器的产生式对抗性攻击 cs.CL

AAAI-24 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08255v1) [paper-pdf](http://arxiv.org/pdf/2401.08255v1)

**Authors**: Tom Roth, Inigo Jauregi Unanue, Alsharif Abuadbba, Massimo Piccardi

**Abstract**: Current adversarial attack algorithms, where an adversary changes a text to fool a victim model, have been repeatedly shown to be effective against text classifiers. These attacks, however, generally assume that the victim model is monolingual and cannot be used to target multilingual victim models, a significant limitation given the increased use of these models. For this reason, in this work we propose an approach to fine-tune a multilingual paraphrase model with an adversarial objective so that it becomes able to generate effective adversarial examples against multilingual classifiers. The training objective incorporates a set of pre-trained models to ensure text quality and language consistency of the generated text. In addition, all the models are suitably connected to the generator by vocabulary-mapping matrices, allowing for full end-to-end differentiability of the overall training pipeline. The experimental validation over two multilingual datasets and five languages has shown the effectiveness of the proposed approach compared to existing baselines, particularly in terms of query efficiency. We also provide a detailed analysis of the generated attacks and discuss limitations and opportunities for future research.

摘要: 目前的对抗性攻击算法，其中对手更改文本以愚弄受害者模型，已被反复证明对文本分类器有效。然而，这些攻击通常假定受害者模型是单一语言的，不能用于针对多语言受害者模型，这是一个很大的限制，因为这些模型的使用越来越多。为此，在这项工作中，我们提出了一种方法来微调一个具有对抗性目标的多语言释义模型，以便它能够生成针对多语言分类器的有效对抗性实例。培训目标纳入了一套预先训练的模型，以确保生成的文本的文本质量和语言一致性。此外，所有模型都通过词汇映射矩阵适当地连接到生成器，从而允许整个训练管道的完全端到端可区分性。在两个多语言数据集和五种语言上的实验验证表明，与现有的基线相比，该方法的有效性，特别是在查询效率方面。我们还提供了对生成的攻击的详细分析，并讨论了未来研究的局限性和机会。



## **26. FreqFed: A Frequency Analysis-Based Approach for Mitigating Poisoning Attacks in Federated Learning**

FreqFed：一种基于频率分析的联合学习中毒攻击缓解方法 cs.CR

To appear in the Network and Distributed System Security (NDSS)  Symposium 2024. 16 pages, 8 figures, 12 tables, 1 algorithm, 3 equations

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2312.04432v2) [paper-pdf](http://arxiv.org/pdf/2312.04432v2)

**Authors**: Hossein Fereidooni, Alessandro Pegoraro, Phillip Rieger, Alexandra Dmitrienko, Ahmad-Reza Sadeghi

**Abstract**: Federated learning (FL) is a collaborative learning paradigm allowing multiple clients to jointly train a model without sharing their training data. However, FL is susceptible to poisoning attacks, in which the adversary injects manipulated model updates into the federated model aggregation process to corrupt or destroy predictions (untargeted poisoning) or implant hidden functionalities (targeted poisoning or backdoors). Existing defenses against poisoning attacks in FL have several limitations, such as relying on specific assumptions about attack types and strategies or data distributions or not sufficiently robust against advanced injection techniques and strategies and simultaneously maintaining the utility of the aggregated model. To address the deficiencies of existing defenses, we take a generic and completely different approach to detect poisoning (targeted and untargeted) attacks. We present FreqFed, a novel aggregation mechanism that transforms the model updates (i.e., weights) into the frequency domain, where we can identify the core frequency components that inherit sufficient information about weights. This allows us to effectively filter out malicious updates during local training on the clients, regardless of attack types, strategies, and clients' data distributions. We extensively evaluate the efficiency and effectiveness of FreqFed in different application domains, including image classification, word prediction, IoT intrusion detection, and speech recognition. We demonstrate that FreqFed can mitigate poisoning attacks effectively with a negligible impact on the utility of the aggregated model.

摘要: 联合学习（FL）是一种协作学习范式，允许多个客户端在不共享训练数据的情况下联合训练模型。然而，FL容易受到中毒攻击，其中攻击者将操纵的模型更新注入联邦模型聚合过程中，以破坏或破坏预测（无目标中毒）或植入隐藏功能（有目标中毒或后门）。FL中针对中毒攻击的现有防御具有几个限制，例如依赖于关于攻击类型和策略或数据分布的特定假设，或者对高级注入技术和策略不够鲁棒，同时保持聚合模型的实用性。为了解决现有防御的缺陷，我们采取了一种通用的、完全不同的方法来检测中毒（有针对性的和无针对性的）攻击。我们提出了FreqFed，一种新的聚合机制，将模型更新（即，权重）到频域中，在频域中我们可以识别继承关于权重的足够信息的核心频率分量。这使我们能够在客户端的本地训练期间有效地过滤恶意更新，而无论攻击类型，策略和客户端的数据分布如何。我们广泛评估了FreqFed在不同应用领域的效率和有效性，包括图像分类，单词预测，物联网入侵检测和语音识别。我们证明了FreqFed可以有效地减轻中毒攻击，对聚合模型的效用影响可以忽略不计。



## **27. AdvSV: An Over-the-Air Adversarial Attack Dataset for Speaker Verification**

AdvSV：一种用于说话人确认的空中对抗攻击数据集 cs.SD

Accepted by ICASSP2024

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2310.05369v2) [paper-pdf](http://arxiv.org/pdf/2310.05369v2)

**Authors**: Li Wang, Jiaqi Li, Yuhao Luo, Jiahao Zheng, Lei Wang, Hao Li, Ke Xu, Chengfang Fang, Jie Shi, Zhizheng Wu

**Abstract**: It is known that deep neural networks are vulnerable to adversarial attacks. Although Automatic Speaker Verification (ASV) built on top of deep neural networks exhibits robust performance in controlled scenarios, many studies confirm that ASV is vulnerable to adversarial attacks. The lack of a standard dataset is a bottleneck for further research, especially reproducible research. In this study, we developed an open-source adversarial attack dataset for speaker verification research. As an initial step, we focused on the over-the-air attack. An over-the-air adversarial attack involves a perturbation generation algorithm, a loudspeaker, a microphone, and an acoustic environment. The variations in the recording configurations make it very challenging to reproduce previous research. The AdvSV dataset is constructed using the Voxceleb1 Verification test set as its foundation. This dataset employs representative ASV models subjected to adversarial attacks and records adversarial samples to simulate over-the-air attack settings. The scope of the dataset can be easily extended to include more types of adversarial attacks. The dataset will be released to the public under the CC BY-SA 4.0. In addition, we also provide a detection baseline for reproducible research.

摘要: 众所周知，深度神经网络很容易受到敌意攻击。尽管建立在深度神经网络之上的自动说话人确认(ASV)在受控场景下表现出较强的性能，但许多研究证实ASV容易受到对手攻击。缺乏标准数据集是进一步研究的瓶颈，特别是可重复性研究。在这项研究中，我们开发了一个开源的对抗性攻击数据集，用于说话人验证研究。作为第一步，我们把重点放在空中攻击上。空中对抗性攻击涉及扰动生成算法、扬声器、麦克风和声学环境。记录配置的变化使得复制先前的研究非常具有挑战性。AdvSV数据集是使用Voxeleb1验证测试集作为其基础来构建的。该数据集采用了典型的受到对抗性攻击的ASV模型，并记录了对抗性样本以模拟空中攻击设置。数据集的范围可以很容易地扩展到包括更多类型的对抗性攻击。该数据集将根据CC BY-SA 4.0向公众发布。此外，我们还为可重复性研究提供了检测基线。



## **28. IoTWarden: A Deep Reinforcement Learning Based Real-time Defense System to Mitigate Trigger-action IoT Attacks**

IoTWarden：一个基于深度强化学习的实时防御系统，用于减轻触发动作物联网攻击 cs.CR

2024 IEEE Wireless Communications and Networking Conference (WCNC  2024)

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08141v1) [paper-pdf](http://arxiv.org/pdf/2401.08141v1)

**Authors**: Md Morshed Alam, Israt Jahan, Weichao Wang

**Abstract**: In trigger-action IoT platforms, IoT devices report event conditions to IoT hubs notifying their cyber states and let the hubs invoke actions in other IoT devices based on functional dependencies defined as rules in a rule engine. These functional dependencies create a chain of interactions that help automate network tasks. Adversaries exploit this chain to report fake event conditions to IoT hubs and perform remote injection attacks upon a smart environment to indirectly control targeted IoT devices. Existing defense efforts usually depend on static analysis over IoT apps to develop rule-based anomaly detection mechanisms. We also see ML-based defense mechanisms in the literature that harness physical event fingerprints to determine anomalies in an IoT network. However, these methods often demonstrate long response time and lack of adaptability when facing complicated attacks. In this paper, we propose to build a deep reinforcement learning based real-time defense system for injection attacks. We define the reward functions for defenders and implement a deep Q-network based approach to identify the optimal defense policy. Our experiments show that the proposed mechanism can effectively and accurately identify and defend against injection attacks with reasonable computation overhead.

摘要: 在触发式物联网平台中，物联网设备向物联网集线器报告事件条件，通知其网络状态，并让集线器根据规则引擎中定义为规则的功能依赖项调用其他物联网设备中的操作。这些功能依赖项创建了一系列交互，帮助实现网络任务的自动化。攻击者利用这条链向物联网集线器报告虚假事件情况，并对智能环境执行远程注入攻击，以间接控制目标物联网设备。现有的防御工作通常依赖于对物联网应用程序的静态分析来开发基于规则的异常检测机制。我们还在文献中看到了基于ML的防御机制，这些机制利用物理事件指纹来确定物联网网络中的异常。然而，当面对复杂的攻击时，这些方法往往表现出响应时间长和缺乏适应性。本文提出了一种基于深度强化学习的注入式攻击实时防御系统。我们定义了防御者的奖励函数，并实现了一种基于深度Q网络的方法来识别最优的防御策略。实验表明，该机制能够以合理的计算开销有效、准确地识别和防御注入攻击。



## **29. Framework and Classification of Indicator of Compromise for physics-based attacks**

基于物理攻击的危害指示器的框架和分类 cs.CR

Pre-print is submitted to 2024 IEEE World Forum on Public Safety  Technology, and is under review

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2401.08127v1) [paper-pdf](http://arxiv.org/pdf/2401.08127v1)

**Authors**: Vincent Tan

**Abstract**: Quantum communications are based on the law of physics for information security and the implications for this form of future information security enabled by quantum science has to be studied. Physics-based vulnerabilities may exist due to the inherent physics properties and behavior of quantum technologies such as Quantum Key Distribution (QKD), thus resulting in new threats that may emerge with attackers exploiting the physics-based vulnerabilities. There were many studies and experiments done to demonstrate the threat of physics-based attacks on quantum links. However, there is a lack of a framework that provides a common language to communicate about the threats and type of adversaries being dealt with for physics-based attacks. This paper is a review of physics-based attacks that were being investigated and attempt to initialize a framework based on the attack objectives and methodologies, referencing the concept from the well-established MITRE ATT&CK, therefore pioneering the classification of Indicator of Compromises (IoCs) for physics-based attacks. This paper will then pave the way for future work in the development of a forensic tool for the different classification of IoCs, with the methods of evidence collections and possible points of extractions for analysis being further investigated.

摘要: 量子通信是基于信息安全的物理定律，必须研究量子科学对这种形式的未来信息安全的影响。由于量子密钥分发(QKD)等量子技术固有的物理属性和行为，可能存在基于物理的漏洞，从而导致攻击者利用基于物理的漏洞可能出现新的威胁。有许多研究和实验证明了基于物理的攻击对量子链路的威胁。然而，缺乏一个框架来提供一种公共语言来沟通基于物理的攻击所处理的威胁和对手的类型。本文对正在研究的基于物理的攻击进行了回顾，并试图参考著名的MITRE ATT&CK的概念，基于攻击的目标和方法初始化一个框架，从而开创了基于物理的攻击的危害指示器(IOCs)分类。然后，本文件将为今后开发不同分类的海委会取证工具的工作铺平道路，并将进一步调查证据收集方法和可能的分析提出点。



## **30. MGTBench: Benchmarking Machine-Generated Text Detection**

MGTBench：机器生成文本检测的基准测试 cs.CR

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2303.14822v3) [paper-pdf](http://arxiv.org/pdf/2303.14822v3)

**Authors**: Xinlei He, Xinyue Shen, Zeyuan Chen, Michael Backes, Yang Zhang

**Abstract**: Nowadays, powerful large language models (LLMs) such as ChatGPT have demonstrated revolutionary power in a variety of tasks. Consequently, the detection of machine-generated texts (MGTs) is becoming increasingly crucial as LLMs become more advanced and prevalent. These models have the ability to generate human-like language, making it challenging to discern whether a text is authored by a human or a machine. This raises concerns regarding authenticity, accountability, and potential bias. However, existing methods for detecting MGTs are evaluated using different model architectures, datasets, and experimental settings, resulting in a lack of a comprehensive evaluation framework that encompasses various methodologies. Furthermore, it remains unclear how existing detection methods would perform against powerful LLMs. In this paper, we fill this gap by proposing the first benchmark framework for MGT detection against powerful LLMs, named MGTBench. Extensive evaluations on public datasets with curated texts generated by various powerful LLMs such as ChatGPT-turbo and Claude demonstrate the effectiveness of different detection methods. Our ablation study shows that a larger number of words in general leads to better performance and most detection methods can achieve similar performance with much fewer training samples. Moreover, we delve into a more challenging task: text attribution. Our findings indicate that the model-based detection methods still perform well in the text attribution task. To investigate the robustness of different detection methods, we consider three adversarial attacks, namely paraphrasing, random spacing, and adversarial perturbations. We discover that these attacks can significantly diminish detection effectiveness, underscoring the critical need for the development of more robust detection methods.

摘要: 如今，强大的大型语言模型(LLM)，如ChatGPT，已经在各种任务中展示了革命性的力量。因此，随着LLMS变得更加先进和普遍，机器生成文本(MGTS)的检测变得越来越重要。这些模型具有生成类似人类的语言的能力，这使得辨别文本是由人还是由机器创作具有挑战性。这引发了人们对真实性、问责性和潜在偏见的担忧。然而，现有的检测MGTS的方法是使用不同的模型体系结构、数据集和实验设置来评估的，导致缺乏包含各种方法的全面评估框架。此外，目前尚不清楚现有的检测方法将如何对抗强大的LLMS。在本文中，我们通过提出第一个针对强大的LLMS的MGT检测的基准框架来填补这一空白，称为MGTB。在公共数据集上的广泛评估与各种强大的LLMS生成的精选文本，如ChatGPT-Turbo和Claude证明了不同检测方法的有效性。我们的烧蚀研究表明，通常情况下，单词数量越多，性能越好，大多数检测方法都可以在更少的训练样本下获得类似的性能。此外，我们还深入研究了一项更具挑战性的任务：文本归因。我们的研究结果表明，基于模型的检测方法在文本归因任务中仍然表现良好。为了考察不同检测方法的稳健性，我们考虑了三种对抗性攻击，即释义攻击、随机间隔攻击和对抗性扰动攻击。我们发现，这些攻击会显著降低检测效率，这突显了开发更健壮的检测方法的迫切需要。



## **31. Towards Robust Neural Networks via Orthogonal Diversity**

基于正交分集的稳健神经网络研究 cs.CV

accepted by Pattern Recognition

**SubmitDate**: 2024-01-16    [abs](http://arxiv.org/abs/2010.12190v5) [paper-pdf](http://arxiv.org/pdf/2010.12190v5)

**Authors**: Kun Fang, Qinghua Tao, Yingwen Wu, Tao Li, Jia Cai, Feipeng Cai, Xiaolin Huang, Jie Yang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to invisible perturbations on the images generated by adversarial attacks, which raises researches on the adversarial robustness of DNNs. A series of methods represented by the adversarial training and its variants have proven as one of the most effective techniques in enhancing the DNN robustness. Generally, adversarial training focuses on enriching the training data by involving perturbed data. Such data augmentation effect of the involved perturbed data in adversarial training does not contribute to the robustness of DNN itself and usually suffers from clean accuracy drop. Towards the robustness of DNN itself, we in this paper propose a novel defense that aims at augmenting the model in order to learn features that are adaptive to diverse inputs, including adversarial examples. More specifically, to augment the model, multiple paths are embedded into the network, and an orthogonality constraint is imposed on these paths to guarantee the diversity among them. A margin-maximization loss is then designed to further boost such DIversity via Orthogonality (DIO). In this way, the proposed DIO augments the model and enhances the robustness of DNN itself as the learned features can be corrected by these mutually-orthogonal paths. Extensive empirical results on various data sets, structures and attacks verify the stronger adversarial robustness of the proposed DIO utilizing model augmentation. Besides, DIO can also be flexibly combined with different data augmentation techniques (e.g., TRADES and DDPM), further promoting robustness gains.

摘要: 深度神经网络（DNN）容易受到对抗性攻击产生的图像上的不可见扰动的影响，这引发了对DNN对抗性鲁棒性的研究。以对抗训练及其变体为代表的一系列方法已被证明是增强DNN鲁棒性的最有效技术之一。一般来说，对抗训练的重点是通过涉及扰动数据来丰富训练数据。在对抗训练中所涉及的扰动数据的这种数据增强效应对DNN本身的鲁棒性没有贡献，并且通常会遭受干净的准确性下降。对于DNN本身的鲁棒性，我们在本文中提出了一种新的防御方法，旨在增强模型，以学习适应不同输入的特征，包括对抗性示例。更具体地说，为了增强模型，将多条路径嵌入到网络中，并对这些路径施加正交性约束以保证它们之间的多样性。然后设计一个边际最大化损失，以进一步提高这种多样性通过可重复性（DIO）。通过这种方式，所提出的DIO增强了模型并增强了DNN本身的鲁棒性，因为学习的特征可以通过这些相互正交的路径进行校正。在各种数据集、结构和攻击上的大量实验结果验证了所提出的DIO利用模型增强具有更强的对抗鲁棒性。此外，DIO还可以灵活地与不同的数据增强技术（例如，TRADES和DDPM），进一步促进稳健性收益。



## **32. Robustness Against Adversarial Attacks via Learning Confined Adversarial Polytopes**

基于受限对抗性多面体学习的抗敌意攻击能力 cs.LG

The paper has been accepted in ICASSP 2024

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07991v1) [paper-pdf](http://arxiv.org/pdf/2401.07991v1)

**Authors**: Shayan Mohajer Hamidi, Linfeng Ye

**Abstract**: Deep neural networks (DNNs) could be deceived by generating human-imperceptible perturbations of clean samples. Therefore, enhancing the robustness of DNNs against adversarial attacks is a crucial task. In this paper, we aim to train robust DNNs by limiting the set of outputs reachable via a norm-bounded perturbation added to a clean sample. We refer to this set as adversarial polytope, and each clean sample has a respective adversarial polytope. Indeed, if the respective polytopes for all the samples are compact such that they do not intersect the decision boundaries of the DNN, then the DNN is robust against adversarial samples. Hence, the inner-working of our algorithm is based on learning \textbf{c}onfined \textbf{a}dversarial \textbf{p}olytopes (CAP). By conducting a thorough set of experiments, we demonstrate the effectiveness of CAP over existing adversarial robustness methods in improving the robustness of models against state-of-the-art attacks including AutoAttack.

摘要: 深度神经网络(DNN)可以通过产生人类无法察觉的干净样本的扰动来欺骗。因此，提高DNN对敌意攻击的健壮性是一项至关重要的任务。在本文中，我们的目标是通过限制通过添加到干净样本的范数有界扰动可到达的输出集来训练鲁棒的DNN。我们将这个集合称为对抗性多面体，每个干净的样本都有各自的对抗性多面体。事实上，如果所有样本的相应多面体是紧凑的，使得它们不与DNN的决策边界相交，则DNN对对抗性样本是健壮的。因此，我们的算法的内部工作是基于学习文本bf{c}受限的文本bf{a}分叉算法(CAP)。通过一组详细的实验，我们证明了CAP在提高模型对包括AutoAttack在内的最新攻击的稳健性方面优于现有的对抗性稳健性方法。



## **33. Learning to Unlearn: Instance-wise Unlearning for Pre-trained Classifiers**

学习遗忘：基于实例的预先训练分类器的遗忘 cs.LG

AAAI 2024 camera ready version

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2301.11578v3) [paper-pdf](http://arxiv.org/pdf/2301.11578v3)

**Authors**: Sungmin Cha, Sungjun Cho, Dasol Hwang, Honglak Lee, Taesup Moon, Moontae Lee

**Abstract**: Since the recent advent of regulations for data protection (e.g., the General Data Protection Regulation), there has been increasing demand in deleting information learned from sensitive data in pre-trained models without retraining from scratch. The inherent vulnerability of neural networks towards adversarial attacks and unfairness also calls for a robust method to remove or correct information in an instance-wise fashion, while retaining the predictive performance across remaining data. To this end, we consider instance-wise unlearning, of which the goal is to delete information on a set of instances from a pre-trained model, by either misclassifying each instance away from its original prediction or relabeling the instance to a different label. We also propose two methods that reduce forgetting on the remaining data: 1) utilizing adversarial examples to overcome forgetting at the representation-level and 2) leveraging weight importance metrics to pinpoint network parameters guilty of propagating unwanted information. Both methods only require the pre-trained model and data instances to forget, allowing painless application to real-life settings where the entire training set is unavailable. Through extensive experimentation on various image classification benchmarks, we show that our approach effectively preserves knowledge of remaining data while unlearning given instances in both single-task and continual unlearning scenarios.

摘要: 自从最近出现了数据保护条例(例如，《一般数据保护条例》)以来，越来越多的人要求在预先训练的模型中删除从敏感数据中学习的信息，而不需要从头开始进行再培训。神经网络对敌意攻击和不公平的固有脆弱性也需要一种健壮的方法来以实例方式移除或纠正信息，同时保持对剩余数据的预测性能。为此，我们考虑了基于实例的遗忘，其目标是通过将每个实例从其原始预测中错误分类或将实例重新标记到不同的标签来从预先训练的模型中删除关于一组实例的信息。我们还提出了两种减少对剩余数据的遗忘的方法：1)利用对抗性例子在表示级克服遗忘；2)利用权重重要性度量来精确定位传播无用信息的网络参数。这两种方法只需要忘记预先训练的模型和数据实例，从而可以轻松地应用到无法获得整个训练集的现实生活环境中。通过在不同图像分类基准上的广泛实验，我们的方法有效地保留了剩余数据的知识，同时在单任务和连续遗忘场景中都忘记了给定的实例。



## **34. ADMIn: Attacks on Dataset, Model and Input. A Threat Model for AI Based Software**

管理：对数据集、模型和输入的攻击。一种基于人工智能的软件威胁模型 cs.CR

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07960v1) [paper-pdf](http://arxiv.org/pdf/2401.07960v1)

**Authors**: Vimal Kumar, Juliette Mayo, Khadija Bahiss

**Abstract**: Machine learning (ML) and artificial intelligence (AI) techniques have now become commonplace in software products and services. When threat modelling a system, it is therefore important that we consider threats unique to ML and AI techniques, in addition to threats to our software. In this paper, we present a threat model that can be used to systematically uncover threats to AI based software. The threat model consists of two main parts, a model of the software development process for AI based software and an attack taxonomy that has been developed using attacks found in adversarial AI research. We apply the threat model to two real life AI based software and discuss the process and the threats found.

摘要: 机器学习(ML)和人工智能(AI)技术现在已经在软件产品和服务中变得司空见惯。因此，在对系统进行威胁建模时，除了考虑对我们软件的威胁之外，我们还必须考虑ML和AI技术特有的威胁。在本文中，我们提出了一个威胁模型，可以用来系统地发现基于人工智能的软件面临的威胁。威胁模型由两个主要部分组成，一个是基于人工智能的软件开发过程的模型，另一个是利用对抗性人工智能研究中发现的攻击开发的攻击分类。我们将威胁模型应用于两个真实的基于人工智能的软件，并讨论了过程和发现的威胁。



## **35. Authorship Obfuscation in Multilingual Machine-Generated Text Detection**

多语种机器文本检测中的作者身份混淆 cs.CL

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07867v1) [paper-pdf](http://arxiv.org/pdf/2401.07867v1)

**Authors**: Dominik Macko, Robert Moro, Adaku Uchendu, Ivan Srba, Jason Samuel Lucas, Michiharu Yamashita, Nafis Irtiza Tripto, Dongwon Lee, Jakub Simko, Maria Bielikova

**Abstract**: High-quality text generation capability of latest Large Language Models (LLMs) causes concerns about their misuse (e.g., in massive generation/spread of disinformation). Machine-generated text (MGT) detection is important to cope with such threats. However, it is susceptible to authorship obfuscation (AO) methods, such as paraphrasing, which can cause MGTs to evade detection. So far, this was evaluated only in monolingual settings. Thus, the susceptibility of recently proposed multilingual detectors is still unknown. We fill this gap by comprehensively benchmarking the performance of 10 well-known AO methods, attacking 37 MGT detection methods against MGTs in 11 languages (i.e., 10 $\times$ 37 $\times$ 11 = 4,070 combinations). We also evaluate the effect of data augmentation on adversarial robustness using obfuscated texts. The results indicate that all tested AO methods can cause detection evasion in all tested languages, where homoglyph attacks are especially successful.

摘要: 最新的大型语言模型(LLM)的高质量文本生成能力引起了人们对它们的滥用(例如，在大规模生成/传播虚假信息中)的担忧。机器生成文本(MGT)检测对于应对此类威胁非常重要。然而，它容易受到作者身份混淆(AO)方法的影响，例如转译，这可能导致MGTS逃避检测。到目前为止，这只在单一语言环境中进行了评估。因此，最近提出的多语言检测器的敏感性仍然未知。我们通过全面基准测试10种著名的AO方法的性能来填补这一空白，针对11种语言的MGT攻击37种MGT检测方法(即，10$\乘以$37$\乘以$11=4,070个组合)。我们还使用混淆文本来评估数据增强对对手健壮性的影响。结果表明，所有测试的声学方法都能在所有测试的语言中造成检测逃避，其中同形文字攻击尤其成功。



## **36. Predominant Aspects on Security for Quantum Machine Learning: Literature Review**

量子机器学习安全性的主要方面：文献综述 quant-ph

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2401.07774v1) [paper-pdf](http://arxiv.org/pdf/2401.07774v1)

**Authors**: Nicola Franco, Alona Sakhnenko, Leon Stolpmann, Daniel Thuerck, Fabian Petsch, Annika Rüll, Jeanette Miriam Lorenz

**Abstract**: Quantum Machine Learning (QML) has emerged as a promising intersection of quantum computing and classical machine learning, anticipated to drive breakthroughs in computational tasks. This paper discusses the question which security concerns and strengths are connected to QML by means of a systematic literature review. We categorize and review the security of QML models, their vulnerabilities inherent to quantum architectures, and the mitigation strategies proposed. The survey reveals that while QML possesses unique strengths, it also introduces novel attack vectors not seen in classical systems. Techniques like adversarial training, quantum noise exploitation, and quantum differential privacy have shown potential in enhancing QML robustness. Our review discuss the need for continued and rigorous research to ensure the secure deployment of QML in real-world applications. This work serves as a foundational reference for researchers and practitioners aiming to navigate the security aspects of QML.

摘要: 量子机器学习(QML)已经成为量子计算和经典机器学习的一个有前途的交叉点，有望推动计算任务的突破。本文通过系统的文献综述，探讨了QML的安全关注点和优势所在。我们对QML模型的安全性、量子体系结构固有的脆弱性以及提出的缓解策略进行了分类和回顾。调查显示，虽然QML具有独特的优势，但它也引入了经典系统中未曾见过的新攻击载体。对抗性训练、量子噪声利用和量子差分保密等技术在增强QML稳健性方面显示出潜力。我们的综述讨论了持续和严格研究的必要性，以确保QML在现实世界应用程序中的安全部署。这项工作为旨在导航QML安全方面的研究人员和实践者提供了基础性参考。



## **37. Uncertainty-based Detection of Adversarial Attacks in Semantic Segmentation**

语义分割中基于不确定性的对抗性攻击检测 cs.CV

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2305.12825v2) [paper-pdf](http://arxiv.org/pdf/2305.12825v2)

**Authors**: Kira Maag, Asja Fischer

**Abstract**: State-of-the-art deep neural networks have proven to be highly powerful in a broad range of tasks, including semantic image segmentation. However, these networks are vulnerable against adversarial attacks, i.e., non-perceptible perturbations added to the input image causing incorrect predictions, which is hazardous in safety-critical applications like automated driving. Adversarial examples and defense strategies are well studied for the image classification task, while there has been limited research in the context of semantic segmentation. First works however show that the segmentation outcome can be severely distorted by adversarial attacks. In this work, we introduce an uncertainty-based approach for the detection of adversarial attacks in semantic segmentation. We observe that uncertainty as for example captured by the entropy of the output distribution behaves differently on clean and perturbed images and leverage this property to distinguish between the two cases. Our method works in a light-weight and post-processing manner, i.e., we do not modify the model or need knowledge of the process used for generating adversarial examples. In a thorough empirical analysis, we demonstrate the ability of our approach to detect perturbed images across multiple types of adversarial attacks.

摘要: 最先进的深度神经网络已被证明在广泛的任务中非常强大，包括语义图像分割。然而，这些网络容易受到对抗性攻击，即，添加到输入图像的不可感知的扰动导致不正确的预测，这在安全关键应用（如自动驾驶）中是危险的。对抗性的例子和防御策略在图像分类任务中得到了很好的研究，而在语义分割方面的研究却很有限。然而，第一项工作表明，分割结果可能会被对抗性攻击严重扭曲。在这项工作中，我们引入了一种基于不确定性的方法来检测语义分割中的对抗性攻击。我们观察到，例如由输出分布的熵捕获的不确定性在干净和扰动图像上表现不同，并利用该属性来区分这两种情况。我们的方法以轻量级和后处理的方式工作，即，我们不修改模型，也不需要用于生成对抗性示例的过程的知识。在一个全面的实证分析中，我们展示了我们的方法在多种类型的对抗性攻击中检测扰动图像的能力。



## **38. Physics-constrained Attack against Convolution-based Human Motion Prediction**

对基于卷积的人体运动预测的物理约束攻击 cs.CV

**SubmitDate**: 2024-01-15    [abs](http://arxiv.org/abs/2306.11990v3) [paper-pdf](http://arxiv.org/pdf/2306.11990v3)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of convolution-based neural networks. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks. The adversarial attack will encounter problems against human motion prediction in naturalness and data scale. To solve the problems above, we propose a new adversarial attack method that generates the worst-case perturbation by maximizing the human motion predictor's prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the naturalness of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models are vulnerable to the proposed attack. Based on the experimental results, we provide insights on how to enhance the adversarial robustness of the human motion predictor and how to improve the adversarial attack against human motion prediction.

摘要: 在基于卷积的神经网络的帮助下，人体运动预测取得了辉煌的成绩。然而，目前还没有关于人体运动预测在面临敌方攻击时的潜在风险的评估工作。对抗性攻击将在自然度和数据规模上遇到针对人体运动预测的问题。为了解决上述问题，我们提出了一种新的对抗性攻击方法，该方法通过在物理约束下最大化人体运动预测器的预测误差来产生最坏情况的扰动。具体地说，我们引入了一种新的自适应方案，使攻击能够适应目标姿态的规模和两个物理约束，以增强对抗性例子的自然性。在三个数据集上的评估实验表明，所有目标模型的预测误差都明显增大，这意味着现有的基于卷积的人体运动预测模型容易受到所提出的攻击。基于实验结果，我们对如何增强人体运动预测器的对抗性鲁棒性以及如何改善针对人体运动预测的对抗性攻击提供了见解。



## **39. Impartial Games: A Challenge for Reinforcement Learning**

公平博弈：强化学习的挑战 cs.LG

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2205.12787v4) [paper-pdf](http://arxiv.org/pdf/2205.12787v4)

**Authors**: Bei Zhou, Søren Riis

**Abstract**: While AlphaZero-style reinforcement learning (RL) algorithms excel in various board games, in this paper we show that they face challenges on impartial games where players share pieces. We present a concrete example of a game - namely the children's game of Nim - and other impartial games that seem to be a stumbling block for AlphaZero-style and similar self-play reinforcement learning algorithms.   Our work is built on the challenges posed by the intricacies of data distribution on the ability of neural networks to learn parity functions, exacerbated by the noisy labels issue. Our findings are consistent with recent studies showing that AlphaZero-style algorithms are vulnerable to adversarial attacks and adversarial perturbations, showing the difficulty of learning to master the games in all legal states.   We show that Nim can be learned on small boards, but the learning progress of AlphaZero-style algorithms dramatically slows down when the board size increases. Intuitively, the difference between impartial games like Nim and partisan games like Chess and Go can be explained by the fact that if a small part of the board is covered for impartial games it is typically not possible to predict whether the position is won or lost as there is often zero correlation between the visible part of a partly blanked-out position and its correct evaluation. This situation starkly contrasts partisan games where a partly blanked-out board position typically provides abundant or at least non-trifle information about the value of the fully uncovered position.

摘要: 虽然AlphaZero风格的强化学习(RL)算法在各种棋类游戏中表现出色，但在本文中，我们展示了它们在玩家共享棋子的公平游戏中面临的挑战。我们提供了一个具体的游戏示例--即Nim的儿童游戏--以及其他公平的游戏，这些游戏似乎是AlphaZero风格和类似的自我发挥强化学习算法的绊脚石。我们的工作建立在错综复杂的数据分布对神经网络学习奇偶函数能力构成的挑战上，噪声标签问题加剧了这一挑战。我们的发现与最近的研究一致，这些研究表明AlphaZero风格的算法容易受到对手攻击和对手扰动，这表明在所有合法国家学习掌握游戏都是困难的。我们表明，NIM可以在小电路板上学习，但AlphaZero风格的算法的学习进度随着电路板大小的增加而显著减慢。直觉上，像尼姆这样的公正游戏与像国际象棋和围棋这样的党派游戏之间的区别可以用这样一个事实来解释：如果棋盘上的一小部分被公平地覆盖，通常不可能预测位置是赢是输，因为部分空白的位置的可见部分与其正确评估之间往往没有相关性。这种情况与党派游戏形成鲜明对比，在党派游戏中，部分空白的董事会职位通常会提供大量或至少不是无关紧要的信息，以了解完全暴露的职位的价值。



## **40. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

前瞻：通过公布对抗性合同来防止Defi攻击 cs.CR

11 pages, 5 figures

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07261v1) [paper-pdf](http://arxiv.org/pdf/2401.07261v1)

**Authors**: Shoupeng Ren, Tianyu Tu, Jian Liu, Di Wu, Kui Ren

**Abstract**: DeFi incidents stemming from various smart contract vulnerabilities have culminated in financial damages exceeding 3 billion USD. The attacks causing such incidents commonly commence with the deployment of adversarial contracts, subsequently leveraging these contracts to execute adversarial transactions that exploit vulnerabilities in victim contracts. Existing defense mechanisms leverage heuristic or machine learning algorithms to detect adversarial transactions, but they face significant challenges in detecting private adversarial transactions. Namely, attackers can send adversarial transactions directly to miners, evading visibility within the blockchain network and effectively bypassing the detection. In this paper, we propose a new direction for detecting DeFi attacks, i.e., detecting adversarial contracts instead of adversarial transactions, allowing us to proactively identify potential attack intentions, even if they employ private adversarial transactions. Specifically, we observe that most adversarial contracts follow a similar pattern, e.g., anonymous fund source, closed-source, frequent token-related function calls. Based on this observation, we build a machine learning classifier that can effectively distinguish adversarial contracts from benign ones. We build a dataset consists of features extracted from 304 adversarial contracts and 13,000 benign contracts. Based on this dataset, we evaluate different classifiers, the results of which show that our method for identifying DeFi adversarial contracts performs exceptionally well. For example, the F1-Score for LightGBM-based classifier is 0.9434, with a remarkably low false positive rate of only 0.12%.

摘要: 由各种智能合同漏洞引发的Defi事件已导致超过30亿美元的经济损失。造成这类事件的攻击通常从部署对抗性合同开始，然后利用这些合同执行利用受害者合同漏洞的对抗性交易。现有的防御机制利用启发式或机器学习算法来检测对抗性交易，但它们在检测私人对抗性交易方面面临着巨大的挑战。也就是说，攻击者可以直接向挖掘者发送敌意交易，从而逃避区块链网络内的可见性，并有效地绕过检测。在本文中，我们提出了一个检测Defi攻击的新方向，即检测对抗性合同而不是对抗性交易，使我们能够主动识别潜在的攻击意图，即使他们使用私人对抗性交易。具体地说，我们观察到大多数对抗性合同遵循类似的模式，例如，匿名资金来源、封闭来源、频繁的令牌相关函数调用。基于这一观察结果，我们构建了一个机器学习分类器，该分类器能够有效地区分敌意合同和良性合同。我们建立了一个由304份敌意合约和13,000份良性合约中提取的特征组成的数据集。基于这个数据集，我们对不同的分类器进行了评估，结果表明，我们的方法在识别违约对抗性合同方面表现得非常好。例如，基于LightGBM的分类器的F1得分为0.9434，假阳性率非常低，仅为0.12%。



## **41. Crafter: Facial Feature Crafting against Inversion-based Identity Theft on Deep Models**

Crafter：面部特征制作，以防止深度模型上基于反转的身份盗窃 cs.CR

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07205v1) [paper-pdf](http://arxiv.org/pdf/2401.07205v1)

**Authors**: Shiming Wang, Zhe Ji, Liyao Xiang, Hao Zhang, Xinbing Wang, Chenghu Zhou, Bo Li

**Abstract**: With the increased capabilities at the edge (e.g., mobile device) and more stringent privacy requirement, it becomes a recent trend for deep learning-enabled applications to pre-process sensitive raw data at the edge and transmit the features to the backend cloud for further processing. A typical application is to run machine learning (ML) services on facial images collected from different individuals. To prevent identity theft, conventional methods commonly rely on an adversarial game-based approach to shed the identity information from the feature. However, such methods can not defend against adaptive attacks, in which an attacker takes a countermove against a known defence strategy. We propose Crafter, a feature crafting mechanism deployed at the edge, to protect the identity information from adaptive model inversion attacks while ensuring the ML tasks are properly carried out in the cloud. The key defence strategy is to mislead the attacker to a non-private prior from which the attacker gains little about the private identity. In this case, the crafted features act like poison training samples for attackers with adaptive model updates. Experimental results indicate that Crafter successfully defends both basic and possible adaptive attacks, which can not be achieved by state-of-the-art adversarial game-based methods.

摘要: 随着边缘功能的增加(例如移动设备)和更严格的隐私要求，支持深度学习的应用程序在边缘对敏感的原始数据进行预处理并将这些功能传输到后端云进行进一步处理已成为最近的趋势。一个典型的应用是对从不同个人收集的面部图像运行机器学习(ML)服务。为了防止身份被盗，传统方法通常依赖于基于对抗性游戏的方法来从功能中去除身份信息。然而，这种方法不能防御适应性攻击，在适应性攻击中，攻击者对已知的防御策略采取反击。我们提出了一种部署在边缘的特征制作机制Crafter，以保护身份信息不受自适应模型反转攻击，同时确保ML任务在云中正确执行。关键的防御策略是将攻击者误导到非私人身份，这样攻击者几乎不会获得关于私人身份的信息。在这种情况下，特制的功能就像是具有自适应模型更新的攻击者的毒药训练样本。实验结果表明，Crafter成功地防御了基本攻击和可能的自适应攻击，这是最先进的基于对抗性游戏的方法无法实现的。



## **42. Experimental quantum e-commerce**

实验性量子电子商务 quant-ph

19 pages, 5 figures, 5 tables

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2308.08821v2) [paper-pdf](http://arxiv.org/pdf/2308.08821v2)

**Authors**: Xiao-Yu Cao, Bing-Hong Li, Yang Wang, Yao Fu, Hua-Lei Yin, Zeng-Bing Chen

**Abstract**: E-commerce, a type of trading that occurs at a high frequency on the Internet, requires guaranteeing the integrity, authentication and non-repudiation of messages through long distance. As current e-commerce schemes are vulnerable to computational attacks, quantum cryptography, ensuring information-theoretic security against adversary's repudiation and forgery, provides a solution to this problem. However, quantum solutions generally have much lower performance compared to classical ones. Besides, when considering imperfect devices, the performance of quantum schemes exhibits a significant decline. Here, for the first time, we demonstrate the whole e-commerce process of involving the signing of a contract and payment among three parties by proposing a quantum e-commerce scheme, which shows resistance of attacks from imperfect devices. Results show that with a maximum attenuation of 25 dB among participants, our scheme can achieve a signature rate of 0.82 times per second for an agreement size of approximately 0.428 megabit. This proposed scheme presents a promising solution for providing information-theoretic security for e-commerce.

摘要: 电子商务是一种在互联网上频繁进行的交易，需要保证消息的完整性、认证性和远距离不可否认性。由于当前的电子商务方案容易受到计算攻击，量子密码学为解决这一问题提供了一种解决方案，它确保了信息论上的安全性，不受对手的否认和伪造。然而，与经典解决方案相比，量子解决方案的性能通常要低得多。此外，当考虑到不完美的器件时，量子方案的性能表现出显著的下降。在这里，我们首次通过提出一个量子电子商务方案，展示了涉及三方签订合同和支付的整个电子商务过程，显示了对不完美设备攻击的抵抗。结果表明，在参与者之间的最大衰减率为25dB的情况下，对于约0.428兆比特的协议，该方案可以达到每秒0.82次的签名率。该方案为电子商务提供信息论安全提供了一种很有前途的解决方案。



## **43. Left-right Discrepancy for Adversarial Attack on Stereo Networks**

立体声网络对抗性攻击的左右差异 cs.CV

**SubmitDate**: 2024-01-14    [abs](http://arxiv.org/abs/2401.07188v1) [paper-pdf](http://arxiv.org/pdf/2401.07188v1)

**Authors**: Pengfei Wang, Xiaofei Hui, Beijia Lu, Nimrod Lilith, Jun Liu, Sameer Alam

**Abstract**: Stereo matching neural networks often involve a Siamese structure to extract intermediate features from left and right images. The similarity between these intermediate left-right features significantly impacts the accuracy of disparity estimation. In this paper, we introduce a novel adversarial attack approach that generates perturbation noise specifically designed to maximize the discrepancy between left and right image features. Extensive experiments demonstrate the superior capability of our method to induce larger prediction errors in stereo neural networks, e.g. outperforming existing state-of-the-art attack methods by 219% MAE on the KITTI dataset and 85% MAE on the Scene Flow dataset. Additionally, we extend our approach to include a proxy network black-box attack method, eliminating the need for access to stereo neural network. This method leverages an arbitrary network from a different vision task as a proxy to generate adversarial noise, effectively causing the stereo network to produce erroneous predictions. Our findings highlight a notable sensitivity of stereo networks to discrepancies in shallow layer features, offering valuable insights that could guide future research in enhancing the robustness of stereo vision systems.

摘要: 立体匹配神经网络通常采用暹罗结构来从左右图像中提取中间特征。这些中间左右特征之间的相似性对视差估计的准确性有很大影响。在本文中，我们介绍了一种新的对抗性攻击方法，该方法产生的扰动噪声专门设计用于最大化左右图像特征之间的差异。大量实验表明，该方法能够在立体神经网络中引入更大的预测误差，例如在Kitti数据集上的MAE比现有的攻击方法高219%，在场景流数据集上的MAE高出85%。此外，我们扩展了我们的方法，包括代理网络黑盒攻击方法，消除了访问立体神经网络的需要。该方法利用来自不同视觉任务的任意网络作为代理来产生对抗性噪声，从而有效地导致立体网络产生错误的预测。我们的发现突出了立体网络对浅层特征差异的显著敏感性，提供了有价值的见解，可以指导未来在增强立体视觉系统的稳健性方面的研究。



## **44. Exploring Adversarial Attacks against Latent Diffusion Model from the Perspective of Adversarial Transferability**

从对抗性转移角度探讨针对潜在扩散模型的对抗性攻击 cs.CV

24 pages, 13 figures

**SubmitDate**: 2024-01-13    [abs](http://arxiv.org/abs/2401.07087v1) [paper-pdf](http://arxiv.org/pdf/2401.07087v1)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie

**Abstract**: Recently, many studies utilized adversarial examples (AEs) to raise the cost of malicious image editing and copyright violation powered by latent diffusion models (LDMs). Despite their successes, a few have studied the surrogate model they used to generate AEs. In this paper, from the perspective of adversarial transferability, we investigate how the surrogate model's property influences the performance of AEs for LDMs. Specifically, we view the time-step sampling in the Monte-Carlo-based (MC-based) adversarial attack as selecting surrogate models. We find that the smoothness of surrogate models at different time steps differs, and we substantially improve the performance of the MC-based AEs by selecting smoother surrogate models. In the light of the theoretical framework on adversarial transferability in image classification, we also conduct a theoretical analysis to explain why smooth surrogate models can also boost AEs for LDMs.

摘要: 近年来，许多研究利用敌意例子(AES)来提高基于潜在扩散模型(LDM)的恶意图像编辑和侵犯版权的成本。尽管他们取得了成功，但也有一些人研究了他们用来生成企业实体的代理模型。本文从对抗性可转移性的角度，研究代理模型的性质如何影响LDM的AES的性能。具体地说，我们把基于蒙特卡洛(MC)的敌意攻击中的时间步长抽样看作是选择代理模型。我们发现代理模型在不同的时间步长上的光滑度是不同的，并且我们通过选择更平滑的代理模型来显著提高基于MC的AES的性能。根据图像分类中对抗性转移的理论框架，我们还进行了理论分析，解释了为什么光滑代理模型也可以提高LDM的AEs。



## **45. Enhancing targeted transferability via feature space fine-tuning**

通过特征空间微调增强目标可转移性 cs.CV

9 pages, 10 figures, accepted by 2024ICASSP

**SubmitDate**: 2024-01-13    [abs](http://arxiv.org/abs/2401.02727v2) [paper-pdf](http://arxiv.org/pdf/2401.02727v2)

**Authors**: Hui Zeng, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples (AEs) have been extensively studied due to their potential for privacy protection and inspiring robust neural networks. Yet, making a targeted AE transferable across unknown models remains challenging. In this paper, to alleviate the overfitting dilemma common in an AE crafted by existing simple iterative attacks, we propose fine-tuning it in the feature space. Specifically, starting with an AE generated by a baseline attack, we encourage the features conducive to the target class and discourage the features to the original class in a middle layer of the source model. Extensive experiments demonstrate that only a few iterations of fine-tuning can boost existing attacks' targeted transferability nontrivially and universally. Our results also verify that the simple iterative attacks can yield comparable or even better transferability than the resource-intensive methods, which rest on training target-specific classifiers or generators with additional data. The code is available at: github.com/zengh5/TA_feature_FT.

摘要: 对抗性例子(AEs)由于其在隐私保护和激发健壮神经网络方面的潜力而被广泛研究。然而，让一种有针对性的AE可以在未知模型之间转移仍然具有挑战性。在本文中，为了缓解现有简单迭代攻击所产生的AE中普遍存在的过适应困境，我们提出了在特征空间中对其进行微调。具体地说，从基线攻击产生的AE开始，我们鼓励有利于目标类的特征，而不鼓励原始类的特征位于源模型的中间层。广泛的实验表明，只需几次迭代的微调就可以提高现有攻击的目标可转移性，而不是平凡的和普遍的。我们的结果还验证了简单的迭代攻击可以产生与资源密集型方法相当甚至更好的可转移性，后者依赖于用额外的数据来训练特定于目标的分类器或生成器。代码可在以下网址获得：githorb.com/zengh5/TA_Feature_FT。



## **46. How Johnny Can Persuade LLMs to Jailbreak Them: Rethinking Persuasion to Challenge AI Safety by Humanizing LLMs**

约翰尼如何说服LLM越狱他们：重新思考说服通过人性化LLM来挑战AI安全 cs.CL

14 pages of the main text, qualitative examples of jailbreaks may be  harmful in nature

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2401.06373v1) [paper-pdf](http://arxiv.org/pdf/2401.06373v1)

**Authors**: Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, Weiyan Shi

**Abstract**: Most traditional AI safety research has approached AI models as machines and centered on algorithm-focused attacks developed by security experts. As large language models (LLMs) become increasingly common and competent, non-expert users can also impose risks during daily interactions. This paper introduces a new perspective to jailbreak LLMs as human-like communicators, to explore this overlooked intersection between everyday language interaction and AI safety. Specifically, we study how to persuade LLMs to jailbreak them. First, we propose a persuasion taxonomy derived from decades of social science research. Then, we apply the taxonomy to automatically generate interpretable persuasive adversarial prompts (PAP) to jailbreak LLMs. Results show that persuasion significantly increases the jailbreak performance across all risk categories: PAP consistently achieves an attack success rate of over $92\%$ on Llama 2-7b Chat, GPT-3.5, and GPT-4 in $10$ trials, surpassing recent algorithm-focused attacks. On the defense side, we explore various mechanisms against PAP and, found a significant gap in existing defenses, and advocate for more fundamental mitigation for highly interactive LLMs

摘要: 大多数传统的人工智能安全研究都将人工智能模型视为机器，并以安全专家开发的以算法为中心的攻击为中心。随着大型语言模型（LLM）变得越来越普遍和有效，非专家用户也可能在日常交互中带来风险。本文介绍了一种新的视角，将越狱LLM作为类似人类的沟通者，以探索日常语言交互和AI安全之间被忽视的交叉点。具体来说，我们研究如何说服LLM越狱。首先，我们提出了一个来自几十年的社会科学研究的说服分类。然后，我们应用分类法自动生成可解释的说服性对抗提示（PAP）越狱LLM。结果表明，说服显著提高了所有风险类别的越狱性能：PAP在Llama 2- 7 b Chat，GPT-3.5和GPT-4上的攻击成功率在10美元的试验中始终达到92美元以上，超过了最近的算法攻击。在防御方面，我们探讨了各种机制，对PAP和，发现了一个显着的差距，在现有的防御，并主张更根本的缓解高度互动的LLM



## **47. FlashSyn: Flash Loan Attack Synthesis via Counter Example Driven Approximation**

FlashSyn：基于反例驱动近似的闪贷攻击合成 cs.PL

29 pages, 8 figures, conference paper extended version

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2206.10708v3) [paper-pdf](http://arxiv.org/pdf/2206.10708v3)

**Authors**: Zhiyang Chen, Sidi Mohamed Beillahi, Fan Long

**Abstract**: In decentralized finance (DeFi), lenders can offer flash loans to borrowers, i.e., loans that are only valid within a blockchain transaction and must be repaid with fees by the end of that transaction. Unlike normal loans, flash loans allow borrowers to borrow large assets without upfront collaterals deposits. Malicious adversaries use flash loans to gather large assets to exploit vulnerable DeFi protocols. In this paper, we introduce a new framework for automated synthesis of adversarial transactions that exploit DeFi protocols using flash loans. To bypass the complexity of a DeFi protocol, we propose a new technique to approximate the DeFi protocol functional behaviors using numerical methods (polynomial linear regression and nearest-neighbor interpolation). We then construct an optimization query using the approximated functions of the DeFi protocol to find an adversarial attack constituted of a sequence of functions invocations with optimal parameters that gives the maximum profit. To improve the accuracy of the approximation, we propose a novel counterexample driven approximation refinement technique. We implement our framework in a tool named FlashSyn. We evaluate FlashSyn on 16 DeFi protocols that were victims to flash loan attacks and 2 DeFi protocols from Damn Vulnerable DeFi challenges. FlashSyn automatically synthesizes an adversarial attack for 16 of the 18 benchmarks. Among the 16 successful cases, FlashSyn identifies attack vectors yielding higher profits than those employed by historical hackers in 3 cases, and also discovers multiple distinct attack vectors in 10 cases, demonstrating its effectiveness in finding possible flash loan attacks.

摘要: 在去中心化金融(Defi)中，贷款人可以向借款人提供闪存贷款，即仅在区块链交易中有效且必须在该交易结束前支付费用的贷款。与普通贷款不同，闪存贷款允许借款人借入大笔资产，而无需预付抵押金。恶意攻击者使用闪存贷款来收集大量资产，以利用易受攻击的Defi协议。在这篇文章中，我们介绍了一个新的框架，用于自动合成利用闪存的DefI协议的对抗性交易。为了绕过DEFI协议的复杂性，我们提出了一种利用数值方法(多项式线性回归和最近邻内插)来逼近DEFI协议功能行为的新技术。然后，我们使用DEFI协议的近似函数构造一个优化查询，以找到由一系列具有最优参数的函数调用组成的对抗性攻击，从而给出最大利润。为了提高逼近的精度，我们提出了一种新的反例驱动的逼近求精技术。我们在一个名为FlashSyn的工具中实现我们的框架。我们评估了FlashSyn在16个遭受闪电贷款攻击的Defi协议和2个来自Damn Vulnerable Defi Challenges的Defi协议上的性能。FlashSyn自动合成了18个基准中的16个的对抗性攻击。在16个成功案例中，FlashSyn在3个案例中识别出了比历史黑客使用的攻击矢量产生更高利润的攻击矢量，并在10个案例中发现了多个不同的攻击矢量，证明了其在发现可能的闪贷攻击方面的有效性。



## **48. When Fairness Meets Privacy: Exploring Privacy Threats in Fair Binary Classifiers through Membership Inference Attacks**

当公平遇到隐私：通过成员关系推理攻击探索公平二进制分类器中的隐私威胁 cs.LG

Under review

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2311.03865v2) [paper-pdf](http://arxiv.org/pdf/2311.03865v2)

**Authors**: Huan Tian, Guangsheng Zhang, Bo Liu, Tianqing Zhu, Ming Ding, Wanlei Zhou

**Abstract**: Previous studies have developed fairness methods for biased models that exhibit discriminatory behaviors towards specific subgroups. While these models have shown promise in achieving fair predictions, recent research has identified their potential vulnerability to score-based membership inference attacks (MIAs). In these attacks, adversaries can infer whether a particular data sample was used during training by analyzing the model's prediction scores. However, our investigations reveal that these score-based MIAs are ineffective when targeting fairness-enhanced models in binary classifications. The attack models trained to launch the MIAs degrade into simplistic threshold models, resulting in lower attack performance. Meanwhile, we observe that fairness methods often lead to prediction performance degradation for the majority subgroups of the training data. This raises the barrier to successful attacks and widens the prediction gaps between member and non-member data. Building upon these insights, we propose an efficient MIA method against fairness-enhanced models based on fairness discrepancy results (FD-MIA). It leverages the difference in the predictions from both the original and fairness-enhanced models and exploits the observed prediction gaps as attack clues. We also explore potential strategies for mitigating privacy leakages. Extensive experiments validate our findings and demonstrate the efficacy of the proposed method.

摘要: 以前的研究已经为对特定子组表现出歧视性行为的有偏见的模型开发了公平方法。虽然这些模型在实现公平预测方面表现出了希望，但最近的研究发现，它们在基于分数的成员关系推理攻击(MIA)中具有潜在的脆弱性。在这些攻击中，攻击者可以通过分析模型的预测分数来推断训练期间是否使用了特定的数据样本。然而，我们的研究表明，这些基于分数的MIA在针对二进制分类中的公平性增强模型时是无效的。被训练来发起MIA的攻击模型降级为简单的阈值模型，导致较低的攻击性能。同时，我们观察到公平性方法经常导致训练数据的大多数子组的预测性能下降。这提高了成功攻击的障碍，并扩大了成员和非成员数据之间的预测差距。在此基础上，我们提出了一种针对基于公平性差异结果的公平性增强模型的高效MIA方法(FD-MIA)。它利用了原始模型和公平性增强模型中预测的差异，并利用观察到的预测差距作为攻击线索。我们还探索了减轻隐私泄露的潜在策略。大量的实验验证了我们的发现，并证明了该方法的有效性。



## **49. MVPatch: More Vivid Patch for Adversarial Camouflaged Attacks on Object Detectors in the Physical World**

MVPatch：对物理世界中的对象探测器进行对抗性伪装攻击的更生动的补丁 cs.CR

14 pages, 8 figures. This work has been submitted to the IEEE for  possible publication. Copyright may be transferred without notice, after  which this version may no longer be accessible

**SubmitDate**: 2024-01-12    [abs](http://arxiv.org/abs/2312.17431v2) [paper-pdf](http://arxiv.org/pdf/2312.17431v2)

**Authors**: Zheng Zhou, Hongbo Zhao, Ju Liu, Qiaosheng Zhang, Liwei Geng, Shuchang Lyu, Wenquan Feng

**Abstract**: Recent investigations demonstrate that adversarial patches can be utilized to manipulate the result of object detection models. However, the conspicuous patterns on these patches may draw more attention and raise suspicions among humans. Moreover, existing works have primarily focused on enhancing the efficacy of attacks in the physical domain, rather than seeking to optimize their stealth attributes and transferability potential. To address these issues, we introduce a dual-perception-based attack framework that generates an adversarial patch known as the More Vivid Patch (MVPatch). The framework consists of a model-perception degradation method and a human-perception improvement method. To derive the MVPatch, we formulate an iterative process that simultaneously constrains the efficacy of multiple object detectors and refines the visual correlation between the generated adversarial patch and a realistic image. Our method employs a model-perception-based approach that reduces the object confidence scores of several object detectors to boost the transferability of adversarial patches. Further, within the human-perception-based framework, we put forward a lightweight technique for visual similarity measurement that facilitates the development of inconspicuous and natural adversarial patches and eliminates the reliance on additional generative models. Additionally, we introduce the naturalness score and transferability score as metrics for an unbiased assessment of various adversarial patches' natural appearance and transferability capacity. Extensive experiments demonstrate that the proposed MVPatch algorithm achieves superior attack transferability compared to similar algorithms in both digital and physical domains while also exhibiting a more natural appearance. These findings emphasize the remarkable stealthiness and transferability of the proposed MVPatch attack algorithm.

摘要: 最近的研究表明，敌意补丁可以被用来操纵目标检测模型的结果。然而，这些斑块上的明显图案可能会引起更多的关注，并在人类中引起怀疑。此外，现有的工作主要集中在提高攻击在物理领域的效能，而不是寻求优化其隐形属性和可转移性。为了解决这些问题，我们引入了一个基于双重感知的攻击框架，该框架生成一个称为MVPatch(MVPatch)的敌意补丁。该框架由模型-感知退化方法和人-感知改进方法组成。为了得到MVPatch，我们制定了一个迭代过程，该过程同时限制了多个目标检测器的有效性，并细化了生成的对抗性补丁与真实图像之间的视觉相关性。我们的方法采用了基于模型感知的方法，降低了多个目标检测器的目标置信度，从而提高了对抗性补丁的可转移性。此外，在基于人类感知的框架内，我们提出了一种轻量级的视觉相似性度量技术，该技术便于开发不明显的自然对抗性补丁，并消除了对额外生成模型的依赖。此外，我们引入了自然度分数和可转移性分数作为衡量标准，以公正地评估各种对抗性补丁的自然外观和可转移性。大量实验表明，与同类算法相比，提出的MVPatch算法在数字域和物理域都具有更好的攻击可转移性，并且表现出更自然的外观。这些发现强调了所提出的MVPatch攻击算法的显著的隐蔽性和可转移性。



## **50. On the Query Complexity of Training Data Reconstruction in Private Learning**

私学中训练数据重构的查询复杂性研究 cs.LG

Updated proof of Thm 10, fixed typos

**SubmitDate**: 2024-01-11    [abs](http://arxiv.org/abs/2303.16372v6) [paper-pdf](http://arxiv.org/pdf/2303.16372v6)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: We analyze the number of queries that a whitebox adversary needs to make to a private learner in order to reconstruct its training data. For $(\epsilon, \delta)$ DP learners with training data drawn from any arbitrary compact metric space, we provide the \emph{first known lower bounds on the adversary's query complexity} as a function of the learner's privacy parameters. \emph{Our results are minimax optimal for every $\epsilon \geq 0, \delta \in [0, 1]$, covering both $\epsilon$-DP and $(0, \delta)$ DP as corollaries}. Beyond this, we obtain query complexity lower bounds for $(\alpha, \epsilon)$ R\'enyi DP learners that are valid for any $\alpha > 1, \epsilon \geq 0$. Finally, we analyze data reconstruction attacks on locally compact metric spaces via the framework of Metric DP, a generalization of DP that accounts for the underlying metric structure of the data. In this setting, we provide the first known analysis of data reconstruction in unbounded, high dimensional spaces and obtain query complexity lower bounds that are nearly tight modulo logarithmic factors.

摘要: 我们分析了白盒攻击者为了重建其训练数据而需要向私人学习者进行的查询数量。对于具有来自任意紧致度量空间的训练数据的$(\epsilon，\Delta)$DP学习者，我们提供了作为学习者隐私参数的函数的\emph(对手查询复杂性的第一个已知下界)。{我们的结果对[0，1]$中的每个$\epsilon\geq0，\Delta\都是极小极大最优的，推论包括$\epsilon$-dp和$(0，\Delta)$dp}。在此基础上，我们得到了$(\α，\epsilon)$R‘Enyi DP学习者的查询复杂性下界，这些下界对任何$\α>1，\epsion\0$都是有效的。最后，我们通过度量DP框架分析了局部紧度量空间上的数据重构攻击。度量DP是DP的推广，它解释了数据的基本度量结构。在这个背景下，我们首次对无界高维空间中的数据重构进行了分析，得到了几乎紧模对数因子的查询复杂度下界。



