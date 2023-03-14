# Latest Adversarial Attack Papers
**update at 2023-03-14 09:29:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Symmetry Defense Against CNN Adversarial Perturbation Attacks**

CNN对抗扰动攻击的对称性防御 cs.LG

13 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2210.04087v2) [paper-pdf](http://arxiv.org/pdf/2210.04087v2)

**Authors**: Blerta Lindqvist

**Abstract**: Convolutional neural network classifiers (CNNs) are susceptible to adversarial attacks that perturb original samples to fool classifiers such as an autonomous vehicle's road sign image classifier. CNNs also lack invariance in the classification of symmetric samples because CNNs can classify symmetric samples differently. Considered together, the CNN lack of adversarial robustness and the CNN lack of invariance mean that the classification of symmetric adversarial samples can differ from their incorrect classification. Could symmetric adversarial samples revert to their correct classification? This paper answers this question by designing a symmetry defense that inverts or horizontally flips adversarial samples before classification against adversaries unaware of the defense. Against adversaries aware of the defense, the defense devises a Klein four symmetry subgroup that includes the horizontal flip and pixel inversion symmetries. The symmetry defense uses the subgroup symmetries in accuracy evaluation and the subgroup closure property to confine the transformations that an adaptive adversary can apply before or after generating the adversarial sample. Without changing the preprocessing, parameters, or model, the proposed symmetry defense counters the Projected Gradient Descent (PGD) and AutoAttack attacks with near-default accuracies for ImageNet. Without using attack knowledge or adversarial samples, the proposed defense exceeds the current best defense, which trains on adversarial samples. The defense maintains and even improves the classification accuracy of non-adversarial samples.

摘要: 卷积神经网络分类器(CNN)容易受到敌意攻击，这些攻击会将原始样本干扰到愚弄分类器，如自动驾驶汽车的道路标志图像分类器。CNN在对称样本的分类上也缺乏不变性，因为CNN可以对对称样本进行不同的分类。综合考虑，CNN缺乏对抗稳健性和CNN缺乏不变性，这意味着对称对抗样本的分类可能不同于它们的错误分类。对称的对抗性样本能恢复到正确的分类吗？本文通过设计一种对称防御来回答这个问题，该对称防御在分类之前颠倒或水平翻转对手样本，以对抗不知道该防御的对手。针对意识到防御的对手，防御方设计了一个克莱因四对称子群，其中包括水平翻转和像素反转对称。对称性防御利用子群对称性进行精度评估，并利用子群封闭性来限制自适应对手在生成对抗样本之前或之后可以应用的变换。在不更改预处理、参数或模型的情况下，所提出的对称防御以近乎默认的ImageNet精度对抗投影梯度下降(PGD)和AutoAttack攻击。在不使用攻击知识或对抗性样本的情况下，建议的防御超过了目前最好的防御，后者基于对抗性样本进行训练。答辩保持甚至提高了非对抗性样本的分类准确率。



## **2. Review on the Feasibility of Adversarial Evasion Attacks and Defenses for Network Intrusion Detection Systems**

网络入侵检测系统对抗性规避攻击与防御的可行性研究综述 cs.CR

Under review (Submitted to Computer Networks - Elsevier)

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.07003v1) [paper-pdf](http://arxiv.org/pdf/2303.07003v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Nowadays, numerous applications incorporate machine learning (ML) algorithms due to their prominent achievements. However, many studies in the field of computer vision have shown that ML can be fooled by intentionally crafted instances, called adversarial examples. These adversarial examples take advantage of the intrinsic vulnerability of ML models. Recent research raises many concerns in the cybersecurity field. An increasing number of researchers are studying the feasibility of such attacks on security systems based on ML algorithms, such as Intrusion Detection Systems (IDS). The feasibility of such adversarial attacks would be influenced by various domain-specific constraints. This can potentially increase the difficulty of crafting adversarial examples. Despite the considerable amount of research that has been done in this area, much of it focuses on showing that it is possible to fool a model using features extracted from the raw data but does not address the practical side, i.e., the reverse transformation from theory to practice. For this reason, we propose a review browsing through various important papers to provide a comprehensive analysis. Our analysis highlights some challenges that have not been addressed in the reviewed papers.

摘要: 如今，由于机器学习(ML)算法的显著成就，许多应用程序都融入了它们。然而，计算机视觉领域的许多研究表明，ML可以被故意制作的实例所愚弄，这些实例被称为对抗性实例。这些敌对的例子利用了ML模型的内在脆弱性。最近的研究在网络安全领域引发了许多担忧。越来越多的研究人员正在研究对基于ML算法的安全系统(如入侵检测系统)进行此类攻击的可行性。这种对抗性攻击的可行性将受到各种特定领域限制的影响。这可能会增加制作对抗性例子的难度。尽管在这一领域已经做了相当多的研究，但大部分研究的重点是表明，使用从原始数据提取的特征来愚弄模型是可能的，但没有解决实际方面，即从理论到实践的反向转换。为此，我们建议通过浏览各种重要论文进行综述，以提供全面的分析。我们的分析突出了审查文件中没有涉及的一些挑战。



## **3. Towards Making a Trojan-horse Attack on Text-to-Image Retrieval**

对文本到图像检索的木马攻击 cs.MM

Accepted by ICASSP 2023

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2202.03861v4) [paper-pdf](http://arxiv.org/pdf/2202.03861v4)

**Authors**: Fan Hu, Aozhu Chen, Xirong Li

**Abstract**: While deep learning based image retrieval is reported to be vulnerable to adversarial attacks, existing works are mainly on image-to-image retrieval with their attacks performed at the front end via query modification. By contrast, we present in this paper the first study about a threat that occurs at the back end of a text-to-image retrieval (T2IR) system. Our study is motivated by the fact that the image collection indexed by the system will be regularly updated due to the arrival of new images from various sources such as web crawlers and advertisers. With malicious images indexed, it is possible for an attacker to indirectly interfere with the retrieval process, letting users see certain images that are completely irrelevant w.r.t. their queries. We put this thought into practice by proposing a novel Trojan-horse attack (THA). In particular, we construct a set of Trojan-horse images by first embedding word-specific adversarial information into a QR code and then putting the code on benign advertising images. A proof-of-concept evaluation, conducted on two popular T2IR datasets (Flickr30k and MS-COCO), shows the effectiveness of the proposed THA in a white-box mode.

摘要: 虽然基于深度学习的图像检索被报道容易受到敌意攻击，但现有的工作主要是针对图像到图像的检索，他们的攻击是通过修改查询在前端进行的。相比之下，我们在本文中提出了第一个关于发生在文本到图像检索(T2IR)系统后端的威胁的研究。我们研究的动机是，由于来自网络爬虫和广告商等各种来源的新图像的到来，系统索引的图像集合将定期更新。对恶意图像进行索引后，攻击者可能会间接干扰检索过程，让用户看到完全不相关的某些图像。他们的疑问。我们将这一思想付诸实践，提出了一种新颖的特洛伊木马攻击(THA)。具体地说，我们首先在二维码中嵌入特定于单词的敌意信息，然后将该代码放在良性广告图像上，从而构建了一组特洛伊木马图像。在两个流行的T2IR数据集(Flickr30k和MS-COCO)上进行的概念验证评估表明，所提出的THA在白盒模式下是有效的。



## **4. Robust Contrastive Language-Image Pretraining against Adversarial Attacks**

抵抗对抗性攻击的健壮对比语言-图像预训练 cs.CV

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06854v1) [paper-pdf](http://arxiv.org/pdf/2303.06854v1)

**Authors**: Wenhan Yang, Baharan Mirzasoleiman

**Abstract**: Contrastive vision-language representation learning has achieved state-of-the-art performance for zero-shot classification, by learning from millions of image-caption pairs crawled from the internet. However, the massive data that powers large multimodal models such as CLIP, makes them extremely vulnerable to various types of adversarial attacks, including targeted and backdoor data poisoning attacks. Despite this vulnerability, robust contrastive vision-language pretraining against adversarial attacks has remained unaddressed. In this work, we propose RoCLIP, the first effective method for robust pretraining {and fine-tuning} multimodal vision-language models. RoCLIP effectively breaks the association between poisoned image-caption pairs by considering a pool of random examples, and (1) matching every image with the text that is most similar to its caption in the pool, and (2) matching every caption with the image that is most similar to its image in the pool. Our extensive experiments show that our method renders state-of-the-art targeted data poisoning and backdoor attacks ineffective during pre-training or fine-tuning of CLIP. In particular, RoCLIP decreases the poison and backdoor attack success rates down to 0\% during pre-training and 1\%-4\% during fine-tuning, and effectively improves the model's performance.

摘要: 对比视觉-语言表征学习通过从互联网上爬行的数百万个图像-字幕对进行学习，实现了最先进的零镜头分类性能。然而，为CLIP等大型多模式模型提供动力的海量数据使它们极易受到各种类型的对抗性攻击，包括定向和后门数据中毒攻击。尽管存在这一弱点，但针对对抗性攻击的强有力的对比视觉语言预训仍然没有得到解决。在这项工作中，我们提出了RoCLIP，这是第一个稳健的预训练和微调多通道视觉语言模型的有效方法。RoCLIP通过考虑一组随机示例，以及(1)将每个图像与池中与其字幕最相似的文本匹配，以及(2)将每个标题与池中与其图像最相似的图像匹配，有效地打破了有毒图像-字幕对之间的关联。我们的大量实验表明，我们的方法使得最先进的有针对性的数据中毒和后门攻击在预训练或微调CLIP期间无效。特别是，RoCLIP将毒害和后门攻击的成功率在预训练时降至0，在微调时降至1 4，有效地提高了模型的性能。



## **5. Adversarial Attacks to Direct Data-driven Control for Destabilization**

针对不稳定的直接数据驱动控制的对抗性攻击 eess.SY

6 pages

**SubmitDate**: 2023-03-13    [abs](http://arxiv.org/abs/2303.06837v1) [paper-pdf](http://arxiv.org/pdf/2303.06837v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control to adversarial attacks in the form of a small but sophisticated perturbation added to the original data. The directed gradient sign method (DGSM) is developed as a specific attack method, based on the fast gradient sign method (FGSM), which has originally been considered in image classification. DGSM uses the gradient of the eigenvalues of the resulting closed-loop system and crafts a perturbation in the direction where the system becomes less stable. It is demonstrated that the system can be destabilized by the attack, even if the original closed-loop system with the clean data has a large margin of stability. To increase the robustness against the attack, regularization methods that have been developed to deal with random disturbances are considered. Their effectiveness is evaluated by numerical experiments using an inverted pendulum model.

摘要: 这项研究调查了直接数据驱动控制在对抗性攻击中的脆弱性，这种攻击是以添加到原始数据上的微小但复杂的扰动的形式进行的。有向梯度符号方法(DGSM)是在最初被认为用于图像分类的快速梯度符号方法(FGSM)的基础上发展起来的一种特定攻击方法。DGSM使用最终闭环系统的特征值的梯度，并在系统变得不稳定的方向上制造扰动。结果表明，即使原始闭环系统具有较大的稳定裕度，攻击也会破坏系统的稳定性。为了增加对攻击的鲁棒性，考虑了已开发的处理随机干扰的正则化方法。利用倒立摆模型进行了数值实验，对其有效性进行了评价。



## **6. Protecting Quantum Procrastinators with Signature Lifting: A Case Study in Cryptocurrencies**

用签名提升保护量子拖延者：加密货币的案例研究 cs.CR

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06754v1) [paper-pdf](http://arxiv.org/pdf/2303.06754v1)

**Authors**: Or Sattath, Shai Wyborski

**Abstract**: Current solutions to quantum vulnerabilities of widely used cryptographic schemes involve migrating users to post-quantum schemes before quantum attacks become feasible. This work deals with protecting quantum procrastinators: users that failed to migrate to post-quantum cryptography in time.   To address this problem in the context of digital signatures, we introduce a technique called signature lifting, that allows us to lift a deployed pre-quantum signature scheme satisfying a certain property to a post-quantum signature scheme that uses the same keys. Informally, the said property is that a post-quantum one-way function is used "somewhere along the way" to derive the public-key from the secret-key. Our constructions of signature lifting relies heavily on the post-quantum digital signature scheme Picnic (Chase et al., CCS'17).   Our main case-study is cryptocurrencies, where this property holds in two scenarios: when the public-key is generated via a key-derivation function or when the public-key hash is posted instead of the public-key itself. We propose a modification, based on signature lifting, that can be applied in many cryptocurrencies for securely spending pre-quantum coins in presence of quantum adversaries. Our construction improves upon existing constructions in two major ways: it is not limited to pre-quantum coins whose ECDSA public-key has been kept secret (and in particular, it handles all coins that are stored in addresses generated by HD wallets), and it does not require access to post-quantum coins or using side payments to pay for posting the transaction.

摘要: 目前针对广泛使用的密码方案的量子漏洞的解决方案包括在量子攻击变得可行之前将用户迁移到后量子方案。这项工作涉及保护量子拖延者：未能及时迁移到后量子密码学的用户。为了在数字签名的背景下解决这个问题，我们引入了一种称为签名提升的技术，该技术允许我们将满足一定性质的部署的前量子签名方案提升到使用相同密钥的后量子签名方案。非正式地，所述性质是使用后量子单向函数来从秘密密钥导出公钥。我们的签名提升的构造在很大程度上依赖于后量子数字签名方案Picnic(Chase等人，CCS‘17)。我们的主要案例研究是加密货币，其中该属性在两种情况下成立：当公钥是通过密钥派生函数生成时，或者当公钥散列被发布而不是公钥本身时。我们提出了一种基于签名提升的改进方案，该方案可以应用于多种加密货币，以便在存在量子对手的情况下安全地消费前量子币。我们的结构在两个主要方面对现有结构进行了改进：它不仅限于其ECDSA公钥被保密的前量子硬币(尤其是，它处理存储在HD钱包生成的地址中的所有硬币)，并且它不需要访问后量子硬币或使用附带支付来支付发布交易的费用。



## **7. DNN-Alias: Deep Neural Network Protection Against Side-Channel Attacks via Layer Balancing**

DNN-Alias：基于层平衡的深层神经网络抗旁路攻击 cs.CR

10 pages

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06746v1) [paper-pdf](http://arxiv.org/pdf/2303.06746v1)

**Authors**: Mahya Morid Ahmadi, Lilas Alrahis, Ozgur Sinanoglu, Muhammad Shafique

**Abstract**: Extracting the architecture of layers of a given deep neural network (DNN) through hardware-based side channels allows adversaries to steal its intellectual property and even launch powerful adversarial attacks on the target system. In this work, we propose DNN-Alias, an obfuscation method for DNNs that forces all the layers in a given network to have similar execution traces, preventing attack models from differentiating between the layers. Towards this, DNN-Alias performs various layer-obfuscation operations, e.g., layer branching, layer deepening, etc, to alter the run-time traces while maintaining the functionality. DNN-Alias deploys an evolutionary algorithm to find the best combination of obfuscation operations in terms of maximizing the security level while maintaining a user-provided latency overhead budget. We demonstrate the effectiveness of our DNN-Alias technique by obfuscating the architecture of 700 randomly generated and obfuscated DNNs running on multiple Nvidia RTX 2080 TI GPU-based machines. Our experiments show that state-of-the-art side-channel architecture stealing attacks cannot extract the original DNN accurately. Moreover, we obfuscate the architecture of various DNNs, such as the VGG-11, VGG-13, ResNet-20, and ResNet-32 networks. Training the DNNs using the standard CIFAR10 dataset, we show that our DNN-Alias maintains the functionality of the original DNNs by preserving the original inference accuracy. Further, the experiments highlight that adversarial attack on obfuscated DNNs is unsuccessful.

摘要: 通过基于硬件的侧通道提取给定深度神经网络(DNN)的层次结构，使得攻击者能够窃取其知识产权，甚至对目标系统发起强大的对抗性攻击。在这项工作中，我们提出了DNN-Alias，这是一种DNN的混淆方法，它强制给定网络中的所有层具有相似的执行轨迹，防止攻击模型在层之间区分。为此，DNN-Alias执行各种层混淆操作，例如层分支、层加深等，以在保持功能的同时改变运行时跟踪。DNN-Alias部署了一种进化算法，以找到模糊操作的最佳组合，从而最大限度地提高安全级别，同时保持用户提供的延迟开销预算。我们通过对运行在多台基于NVIDIA RTX 2080 TI GPU的机器上的700个随机生成和模糊的DNN的架构进行模糊处理，展示了我们的DNN-Alias技术的有效性。实验表明，现有的旁路结构窃取攻击不能准确提取原始DNN。此外，我们还混淆了各种DNN的体系结构，如VGG-11、VGG-13、ResNet-20和ResNet-32网络。使用标准的CIFAR10数据集对DNN进行训练，我们的DNN-Alias通过保持原始DNN的推理精度来保持原始DNN的功能。此外，实验表明，对模糊DNN的对抗性攻击是不成功的。



## **8. Adv-Bot: Realistic Adversarial Botnet Attacks against Network Intrusion Detection Systems**

ADV-Bot：针对网络入侵检测系统的现实对抗性僵尸网络攻击 cs.CR

This work is published in Computers & Security (an Elsevier journal)  https://www.sciencedirect.com/science/article/pii/S016740482300086X

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06664v1) [paper-pdf](http://arxiv.org/pdf/2303.06664v1)

**Authors**: Islam Debicha, Benjamin Cochez, Tayeb Kenaza, Thibault Debatty, Jean-Michel Dricot, Wim Mees

**Abstract**: Due to the numerous advantages of machine learning (ML) algorithms, many applications now incorporate them. However, many studies in the field of image classification have shown that MLs can be fooled by a variety of adversarial attacks. These attacks take advantage of ML algorithms' inherent vulnerability. This raises many questions in the cybersecurity field, where a growing number of researchers are recently investigating the feasibility of such attacks against machine learning-based security systems, such as intrusion detection systems. The majority of this research demonstrates that it is possible to fool a model using features extracted from a raw data source, but it does not take into account the real implementation of such attacks, i.e., the reverse transformation from theory to practice. The real implementation of these adversarial attacks would be influenced by various constraints that would make their execution more difficult. As a result, the purpose of this study was to investigate the actual feasibility of adversarial attacks, specifically evasion attacks, against network-based intrusion detection systems (NIDS), demonstrating that it is entirely possible to fool these ML-based IDSs using our proposed adversarial algorithm while assuming as many constraints as possible in a black-box setting. In addition, since it is critical to design defense mechanisms to protect ML-based IDSs against such attacks, a defensive scheme is presented. Realistic botnet traffic traces are used to assess this work. Our goal is to create adversarial botnet traffic that can avoid detection while still performing all of its intended malicious functionality.

摘要: 由于机器学习(ML)算法的众多优势，现在许多应用程序都将其纳入其中。然而，图像分类领域的许多研究表明，MLS可以被各种对抗性攻击所愚弄。这些攻击利用了ML算法固有的漏洞。这在网络安全领域引发了许多问题，最近越来越多的研究人员正在调查针对入侵检测系统等基于机器学习的安全系统进行此类攻击的可行性。大多数研究表明，使用从原始数据源提取的特征来愚弄模型是可能的，但它没有考虑到此类攻击的真正实现，即从理论到实践的反向转换。这些对抗性攻击的真正实施将受到各种限制的影响，这些限制将使它们的执行更加困难。因此，本研究的目的是调查针对基于网络的入侵检测系统(NID)的对抗性攻击，特别是逃避攻击的实际可行性，证明使用我们提出的对抗性算法来愚弄这些基于ML的入侵检测系统是完全可能的，同时假设在黑盒设置下尽可能多的约束。此外，由于设计防御机制以保护基于ML的入侵检测系统免受此类攻击是至关重要的，因此提出了一种防御方案。使用真实的僵尸网络流量跟踪来评估这项工作。我们的目标是创建敌意僵尸网络流量，以避免检测，同时仍可执行其所有预期的恶意功能。



## **9. Interpreting Hidden Semantics in the Intermediate Layers of 3D Point Cloud Classification Neural Network**

三维点云分类神经网络中间层隐含语义解释 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06652v1) [paper-pdf](http://arxiv.org/pdf/2303.06652v1)

**Authors**: Weiquan Liu, Minghao Liu, Shijun Zheng, Cheng Wang

**Abstract**: Although 3D point cloud classification neural network models have been widely used, the in-depth interpretation of the activation of the neurons and layers is still a challenge. We propose a novel approach, named Relevance Flow, to interpret the hidden semantics of 3D point cloud classification neural networks. It delivers the class Relevance to the activated neurons in the intermediate layers in a back-propagation manner, and associates the activation of neurons with the input points to visualize the hidden semantics of each layer. Specially, we reveal that the 3D point cloud classification neural network has learned the plane-level and part-level hidden semantics in the intermediate layers, and utilize the normal and IoU to evaluate the consistency of both levels' hidden semantics. Besides, by using the hidden semantics, we generate the adversarial attack samples to attack 3D point cloud classifiers. Experiments show that our proposed method reveals the hidden semantics of the 3D point cloud classification neural network on ModelNet40 and ShapeNet, which can be used for the unsupervised point cloud part segmentation without labels and attacking the 3D point cloud classifiers.

摘要: 虽然三维点云分类神经网络模型已经得到了广泛的应用，但对神经元和层的激活过程的深入解释仍然是一个挑战。提出了一种新的解释三维点云分类神经网络隐含语义的方法--关联流。它以反向传播的方式将类相关性传递给中间层中被激活的神经元，并将神经元的激活与输入点相关联，以可视化每一层的隐藏语义。特别是，我们揭示了三维点云分类神经网络在中间层学习了平面级和零部件级的隐藏语义，并利用Normal和IOU来评估这两个级别的隐藏语义的一致性。此外，利用隐含语义生成了攻击三维点云分类器的对抗性攻击样本。实验表明，该方法在ModelNet40和ShapeNet上揭示了三维点云分类神经网络的隐含语义，可用于无标签的无监督点云部分分割和攻击三维点云分类器。



## **10. Query Attack by Multi-Identity Surrogates**

多身份代理的查询攻击 cs.LG

IEEE TRANSACTIONS ON ARTIFICIAL INTELLIGENCE

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2105.15010v5) [paper-pdf](http://arxiv.org/pdf/2105.15010v5)

**Authors**: Sizhe Chen, Zhehao Huang, Qinghua Tao, Xiaolin Huang

**Abstract**: Deep Neural Networks (DNNs) are acknowledged as vulnerable to adversarial attacks, while the existing black-box attacks require extensive queries on the victim DNN to achieve high success rates. For query-efficiency, surrogate models of the victim are used to generate transferable Adversarial Examples (AEs) because of their Gradient Similarity (GS), i.e., surrogates' attack gradients are similar to the victim's ones. However, it is generally neglected to exploit their similarity on outputs, namely the Prediction Similarity (PS), to filter out inefficient queries by surrogates without querying the victim. To jointly utilize and also optimize surrogates' GS and PS, we develop QueryNet, a unified attack framework that can significantly reduce queries. QueryNet creatively attacks by multi-identity surrogates, i.e., crafts several AEs for one sample by different surrogates, and also uses surrogates to decide on the most promising AE for the query. After that, the victim's query feedback is accumulated to optimize not only surrogates' parameters but also their architectures, enhancing both the GS and the PS. Although QueryNet has no access to pre-trained surrogates' prior, it reduces queries by averagely about an order of magnitude compared to alternatives within an acceptable time, according to our comprehensive experiments: 11 victims (including two commercial models) on MNIST/CIFAR10/ImageNet, allowing only 8-bit image queries, and no access to the victim's training data. The code is available at https://github.com/Sizhe-Chen/QueryNet.

摘要: 深度神经网络(DNN)被认为容易受到对抗性攻击，而现有的黑盒攻击需要对受害者DNN进行广泛的查询才能获得高的成功率。为了提高查询效率，受害者的代理模型被用来生成可转移的对抗实例，因为它们具有梯度相似性，即代理的攻击梯度与受害者的攻击梯度相似。然而，通常忽略了利用它们在输出上的相似性，即预测相似度(PS)来过滤代理在不查询受害者的情况下的低效查询。为了联合利用并优化代理的GS和PS，我们开发了QueryNet，这是一个可以显著减少查询的统一攻击框架。QueryNet创造性地利用多身份代理进行攻击，即通过不同的代理为一个样本构造多个代理实体，并使用代理为查询选择最有希望的代理实体。之后，受害者的查询反馈被累积，不仅优化了代理的参数，还优化了它们的体系结构，提高了GS和PS。虽然QueryNet无法访问预先训练的代理人的先前，但根据我们的综合实验：11名受害者(包括两个商业模型)在MNIST/CIFAR10/ImageNet上仅允许8位图像查询，并且无法访问受害者的训练数据，与替代方案相比，它在可接受的时间内平均减少了一个数量级的查询。代码可在https://github.com/Sizhe-Chen/QueryNet.上获得



## **11. Adaptive Local Adversarial Attacks on 3D Point Clouds for Augmented Reality**

增强现实中3D点云的自适应局部对抗攻击 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06641v1) [paper-pdf](http://arxiv.org/pdf/2303.06641v1)

**Authors**: Weiquan Liu, Shijun Zheng, Cheng Wang

**Abstract**: As the key technology of augmented reality (AR), 3D recognition and tracking are always vulnerable to adversarial examples, which will cause serious security risks to AR systems. Adversarial examples are beneficial to improve the robustness of the 3D neural network model and enhance the stability of the AR system. At present, most 3D adversarial attack methods perturb the entire point cloud to generate adversarial examples, which results in high perturbation costs and difficulty in reconstructing the corresponding real objects in the physical world. In this paper, we propose an adaptive local adversarial attack method (AL-Adv) on 3D point clouds to generate adversarial point clouds. First, we analyze the vulnerability of the 3D network model and extract the salient regions of the input point cloud, namely the vulnerable regions. Second, we propose an adaptive gradient attack algorithm that targets vulnerable regions. The proposed attack algorithm adaptively assigns different disturbances in different directions of the three-dimensional coordinates of the point cloud. Experimental results show that our proposed method AL-Adv achieves a higher attack success rate than the global attack method. Specifically, the adversarial examples generated by the AL-Adv demonstrate good imperceptibility and small generation costs.

摘要: 作为增强现实(AR)的关键技术，3D识别与跟踪往往容易受到敌意攻击，这将给AR系统带来严重的安全隐患。对抗性例子有利于提高3D神经网络模型的鲁棒性，增强AR系统的稳定性。目前，大多数3D对抗性攻击方法都是对整个点云进行扰动来生成对抗性实例，这导致了较高的扰动代价和重建物理世界中对应的真实对象的困难。本文提出了一种基于三维点云的自适应局部对抗攻击方法(AL-ADV)来生成对抗点云。首先，分析三维网络模型的脆弱性，提取输入点云的显著区域，即脆弱区域。其次，提出了一种针对易受攻击区域的自适应梯度攻击算法。该攻击算法在点云三维坐标的不同方向上自适应地分配不同的干扰。实验结果表明，我们提出的方法AL-ADV比全局攻击方法具有更高的攻击成功率。具体地说，AL-ADV生成的对抗性示例具有良好的隐蔽性和较小的生成成本。



## **12. Multi-metrics adaptively identifies backdoors in Federated learning**

多指标自适应地识别联合学习中的后门 cs.CR

13 pages, 8 figures and 6 tables

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2303.06601v1) [paper-pdf](http://arxiv.org/pdf/2303.06601v1)

**Authors**: Siquan Huang, Yijiang Li, Chong Chen, Leyu Shi, Ying Gao

**Abstract**: The decentralized and privacy-preserving nature of federated learning (FL) makes it vulnerable to backdoor attacks aiming to manipulate the behavior of the resulting model on specific adversary-chosen inputs. However, most existing defenses based on statistical differences take effect only against specific attacks, especially when the malicious gradients are similar to benign ones or the data are highly non-independent and identically distributed (non-IID). In this paper, we revisit the distance-based defense methods and discover that i) Euclidean distance becomes meaningless in high dimensions and ii) malicious gradients with diverse characteristics cannot be identified by a single metric. To this end, we present a simple yet effective defense strategy with multi-metrics and dynamic weighting to identify backdoors adaptively. Furthermore, our novel defense has no reliance on predefined assumptions over attack settings or data distributions and little impact on benign performance. To evaluate the effectiveness of our approach, we conduct comprehensive experiments on different datasets under various attack settings, where our method achieves the best defensive performance. For instance, we achieve the lowest backdoor accuracy of 3.06% under the difficult Edge-case PGD, showing significant superiority over previous defenses. The results also demonstrate that our method can be well-adapted to a wide range of non-IID degrees without sacrificing the benign performance.

摘要: 联邦学习(FL)的去中心化和隐私保护特性使其容易受到后门攻击，目的是在特定对手选择的输入上操纵结果模型的行为。然而，现有的大多数基于统计差异的防御措施只对特定的攻击有效，特别是当恶意梯度类似于良性梯度或数据具有高度非独立和同分布(Non-IID)时。在本文中，我们回顾了基于距离的防御方法，发现i)欧氏距离在高维中变得没有意义，ii)具有不同特征的恶意梯度不能用单一的度量来识别。为此，我们提出了一种简单而有效的防御策略，采用多指标和动态加权来自适应地识别后门。此外，我们的新型防御不依赖于对攻击设置或数据分布的预定义假设，并且对良性性能几乎没有影响。为了评估该方法的有效性，我们在不同的攻击环境下对不同的数据集进行了全面的实验，其中我们的方法取得了最好的防御性能。例如，在困难的Edge-Case PGD下，我们实现了3.06%的最低后门精度，显示出明显优于以前的防御。结果还表明，我们的方法可以很好地适应广泛的非IID程度，而不牺牲良好的性能。



## **13. STPrivacy: Spatio-Temporal Privacy-Preserving Action Recognition**

STPrivacy：时空隐私保护动作识别 cs.CV

**SubmitDate**: 2023-03-12    [abs](http://arxiv.org/abs/2301.03046v2) [paper-pdf](http://arxiv.org/pdf/2301.03046v2)

**Authors**: Ming Li, Xiangyu Xu, Hehe Fan, Pan Zhou, Jun Liu, Jia-Wei Liu, Jiahe Li, Jussi Keppo, Mike Zheng Shou, Shuicheng Yan

**Abstract**: Existing methods of privacy-preserving action recognition (PPAR) mainly focus on frame-level (spatial) privacy removal through 2D CNNs. Unfortunately, they have two major drawbacks. First, they may compromise temporal dynamics in input videos, which are critical for accurate action recognition. Second, they are vulnerable to practical attacking scenarios where attackers probe for privacy from an entire video rather than individual frames. To address these issues, we propose a novel framework STPrivacy to perform video-level PPAR. For the first time, we introduce vision Transformers into PPAR by treating a video as a tubelet sequence, and accordingly design two complementary mechanisms, i.e., sparsification and anonymization, to remove privacy from a spatio-temporal perspective. In specific, our privacy sparsification mechanism applies adaptive token selection to abandon action-irrelevant tubelets. Then, our anonymization mechanism implicitly manipulates the remaining action-tubelets to erase privacy in the embedding space through adversarial learning. These mechanisms provide significant advantages in terms of privacy preservation for human eyes and action-privacy trade-off adjustment during deployment. We additionally contribute the first two large-scale PPAR benchmarks, VP-HMDB51 and VP-UCF101, to the community. Extensive evaluations on them, as well as two other tasks, validate the effectiveness and generalization capability of our framework.

摘要: 现有的隐私保护动作识别(PPAR)方法主要集中在通过2D CNN去除帧级(空间)隐私。不幸的是，它们有两个主要缺陷。首先，它们可能会影响输入视频中的时间动态，而时间动态对于准确的动作识别至关重要。其次，它们容易受到实际攻击场景的攻击，即攻击者从整个视频而不是单个帧来探测隐私。为了解决这些问题，我们提出了一种新的框架STPrivacy来执行视频级PPAR。首次将视觉变形器引入到PPAR中，将视频看作一个元组序列，并相应地设计了稀疏化和匿名化两种互补机制，从时空的角度去除隐私。具体地说，我们的隐私稀疏机制采用自适应令牌选择来丢弃与动作无关的tubelet。然后，我们的匿名化机制隐含地操纵剩余的动作元组，通过对抗性学习消除嵌入空间中的隐私。这些机制在人眼隐私保护和部署过程中的动作-隐私权衡调整方面具有显著优势。此外，我们还向社区贡献了头两个大型PPAR基准，VP-HMDB51和VP-UCF101。对它们的广泛评估以及另外两项任务，验证了我们框架的有效性和推广能力。



## **14. Disclosure Risk from Homogeneity Attack in Differentially Private Frequency Distribution**

差分私密频率分布中同质性攻击的泄漏风险 cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2101.00311v5) [paper-pdf](http://arxiv.org/pdf/2101.00311v5)

**Authors**: Fang Liu, Xingyuan Zhao

**Abstract**: Differential privacy (DP) provides a robust model to achieve privacy guarantees for released information. We examine the protection potency of sanitized multi-dimensional frequency distributions via DP randomization mechanisms against homogeneity attack (HA). HA allows adversaries to obtain the exact values on sensitive attributes for their targets without having to identify them from the released data. We propose measures for disclosure risk from HA and derive closed-form relationships between the privacy loss parameters in DP and the disclosure risk from HA. The availability of the closed-form relationships assists understanding the abstract concepts of DP and privacy loss parameters by putting them in the context of a concrete privacy attack and offers a perspective for choosing privacy loss parameters when employing DP mechanisms in information sanitization and release in practice. We apply the closed-form mathematical relationships in real-life datasets to demonstrate the assessment of disclosure risk due to HA on differentially private sanitized frequency distributions at various privacy loss parameters.

摘要: 差异隐私(DP)提供了一种健壮的模型来实现对发布信息的隐私保障。我们通过DP随机化机制来检验经过消毒的多维频率分布对同质性攻击(HA)的保护效力。HA允许攻击者获得其目标的敏感属性的精确值，而不必从发布的数据中识别它们。提出了HA信息泄露风险的度量方法，并推导出DP中的隐私损失参数与HA信息泄露风险之间的闭合关系。封闭关系的可用性通过将DP和隐私丢失参数置于具体的隐私攻击的上下文中来帮助理解DP和隐私丢失参数的抽象概念，并为在实践中使用DP机制进行信息清理和发布时选择隐私丢失参数提供了一个视角。我们将封闭形式的数学关系应用于真实数据集中，以演示在不同隐私损失参数下，HA对不同隐私消毒频率分布的泄露风险的评估。



## **15. Anomaly Detection with Ensemble of Encoder and Decoder**

基于编解码器集成的异常检测 cs.LG

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06431v1) [paper-pdf](http://arxiv.org/pdf/2303.06431v1)

**Authors**: Xijuan Sun, Di Wu, Arnaud Zinflou, Benoit Boulet

**Abstract**: Hacking and false data injection from adversaries can threaten power grids' everyday operations and cause significant economic loss. Anomaly detection in power grids aims to detect and discriminate anomalies caused by cyber attacks against the power system, which is essential for keeping power grids working correctly and efficiently. Different methods have been applied for anomaly detection, such as statistical methods and machine learning-based methods. Usually, machine learning-based methods need to model the normal data distribution. In this work, we propose a novel anomaly detection method by modeling the data distribution of normal samples via multiple encoders and decoders. Specifically, the proposed method maps input samples into a latent space and then reconstructs output samples from latent vectors. The extra encoder finally maps reconstructed samples to latent representations. During the training phase, we optimize parameters by minimizing the reconstruction loss and encoding loss. Training samples are re-weighted to focus more on missed correlations between features of normal data. Furthermore, we employ the long short-term memory model as encoders and decoders to test its effectiveness. We also investigate a meta-learning-based framework for hyper-parameter tuning of our approach. Experiment results on network intrusion and power system datasets demonstrate the effectiveness of our proposed method, where our models consistently outperform all baselines.

摘要: 来自对手的黑客攻击和虚假数据注入可能威胁电网的日常运行，并造成重大经济损失。电网异常检测的目的是检测和识别网络攻击对电力系统造成的异常，这是保证电网正常高效运行的关键。不同的方法被应用于异常检测，如统计方法和基于机器学习的方法。通常，基于机器学习的方法需要对正态数据分布进行建模。在这项工作中，我们提出了一种新的异常检测方法，通过多个编解码器对正常样本的数据分布进行建模。具体地说，该方法将输入样本映射到潜在空间，然后从潜在向量重构输出样本。额外的编码器最终将重构的样本映射到潜在表示。在训练阶段，我们通过最小化重建损失和编码损失来优化参数。训练样本被重新加权，以更多地关注正常数据的特征之间的遗漏相关性。此外，我们使用长短期记忆模型作为编解码器来测试其有效性。我们还研究了一个基于元学习的框架，用于对我们的方法进行超参数调整。在网络入侵和电力系统数据集上的实验结果表明，我们提出的方法是有效的，我们的模型一致地优于所有基线。



## **16. Improving the Robustness of Deep Convolutional Neural Networks Through Feature Learning**

利用特征学习提高深卷积神经网络的稳健性 cs.CV

8 pages, 12 figures, 6 tables. Work in process

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06425v1) [paper-pdf](http://arxiv.org/pdf/2303.06425v1)

**Authors**: Jin Ding, Jie-Chao Zhao, Yong-Zhi Sun, Ping Tan, Ji-En Ma, You-Tong Fang

**Abstract**: Deep convolutional neural network (DCNN for short) models are vulnerable to examples with small perturbations. Adversarial training (AT for short) is a widely used approach to enhance the robustness of DCNN models by data augmentation. In AT, the DCNN models are trained with clean examples and adversarial examples (AE for short) which are generated using a specific attack method, aiming to gain ability to defend themselves when facing the unseen AEs. However, in practice, the trained DCNN models are often fooled by the AEs generated by the novel attack methods. This naturally raises a question: can a DCNN model learn certain features which are insensitive to small perturbations, and further defend itself no matter what attack methods are presented. To answer this question, this paper makes a beginning effort by proposing a shallow binary feature module (SBFM for short), which can be integrated into any popular backbone. The SBFM includes two types of layers, i.e., Sobel layer and threshold layer. In Sobel layer, there are four parallel feature maps which represent horizontal, vertical, and diagonal edge features, respectively. And in threshold layer, it turns the edge features learnt by Sobel layer to the binary features, which then are feeded into the fully connected layers for classification with the features learnt by the backbone. We integrate SBFM into VGG16 and ResNet34, respectively, and conduct experiments on multiple datasets. Experimental results demonstrate, under FGSM attack with $\epsilon=8/255$, the SBFM integrated models can achieve averagely 35\% higher accuracy than the original ones, and in CIFAR-10 and TinyImageNet datasets, the SBFM integrated models can achieve averagely 75\% classification accuracy. The work in this paper shows it is promising to enhance the robustness of DCNN models through feature learning.

摘要: 深层卷积神经网络(DCNN)模型容易受到小扰动样本的影响。对抗性训练是一种广泛使用的通过数据增强来增强DCNN模型稳健性的方法。在AT中，DCNN模型使用特定攻击方法生成的干净实例和敌意实例(简称AE)进行训练，目的是在面对看不见的AE时获得自卫能力。然而，在实际应用中，训练好的DCNN模型往往会被新的攻击方法产生的攻击事件所愚弄。这自然提出了一个问题：DCNN模型是否能够学习对小扰动不敏感的某些特征，并在任何攻击方法提出的情况下进一步自卫。为了回答这个问题，本文首先提出了一种可集成到任何主流主干中的浅二进制特征模块(SBFM)。SBFM包括两种类型的层，即Sobel层和阈值层。在Sobel层中，有四个平行的特征图，分别表示水平、垂直和对角的边缘特征。在阈值层中，将Sobel层学习到的边缘特征转化为二值特征，然后将二值特征送入全连通层，与主干学习的特征进行分类。我们将SBFM分别集成到VGG16和ResNet34中，并在多个数据集上进行了实验。实验结果表明，在$epsilon=8/255$的FGSM攻击下，SBFM集成模型的分类正确率比原始模型平均提高了35%，在CIFAR-10和TinyImageNet数据集中，SBFM集成模型的分类正确率平均达到75%。本文的工作表明，通过特征学习来增强DCNN模型的稳健性是很有前途的。



## **17. MorDIFF: Recognition Vulnerability and Attack Detectability of Face Morphing Attacks Created by Diffusion Autoencoders**

MorDIFF：扩散自动编码器造成的人脸变形攻击的识别漏洞和攻击可检测性 cs.CV

Accepted at the 11th International Workshop on Biometrics and  Forensics 2023 (IWBF 2023)

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2302.01843v2) [paper-pdf](http://arxiv.org/pdf/2302.01843v2)

**Authors**: Naser Damer, Meiling Fang, Patrick Siebke, Jan Niklas Kolf, Marco Huber, Fadi Boutros

**Abstract**: Investigating new methods of creating face morphing attacks is essential to foresee novel attacks and help mitigate them. Creating morphing attacks is commonly either performed on the image-level or on the representation-level. The representation-level morphing has been performed so far based on generative adversarial networks (GAN) where the encoded images are interpolated in the latent space to produce a morphed image based on the interpolated vector. Such a process was constrained by the limited reconstruction fidelity of GAN architectures. Recent advances in the diffusion autoencoder models have overcome the GAN limitations, leading to high reconstruction fidelity. This theoretically makes them a perfect candidate to perform representation-level face morphing. This work investigates using diffusion autoencoders to create face morphing attacks by comparing them to a wide range of image-level and representation-level morphs. Our vulnerability analyses on four state-of-the-art face recognition models have shown that such models are highly vulnerable to the created attacks, the MorDIFF, especially when compared to existing representation-level morphs. Detailed detectability analyses are also performed on the MorDIFF, showing that they are as challenging to detect as other morphing attacks created on the image- or representation-level. Data and morphing script are made public: https://github.com/naserdamer/MorDIFF.

摘要: 研究创建面部变形攻击的新方法对于预见新的攻击并帮助缓解它们是至关重要的。创建变形攻击通常是在图像级或表示级执行的。到目前为止，表示级变形是基于生成对抗网络(GAN)执行的，其中在潜在空间中对编码图像进行内插，以基于内插向量产生变形图像。这一过程受到GaN结构有限重建保真度的限制。扩散式自动编码器模型的最新进展克服了GaN的限制，导致了高重建保真度。从理论上讲，这使它们成为执行表示级面部变形的完美候选者。这项工作使用扩散自动编码器来创建人脸变形攻击，通过将它们与广泛的图像级和表示级变形进行比较。我们对四个最先进的人脸识别模型进行的漏洞分析表明，这些模型对创建的攻击MorDIFF非常脆弱，特别是与现有的表示级变形相比。还对MorDIFF进行了详细的可检测性分析，表明它们与在图像或表示层上创建的其他变形攻击一样具有挑战性。数据和变形脚本公开：https://github.com/naserdamer/MorDIFF.



## **18. Adversarial Attacks and Defenses in Machine Learning-Powered Networks: A Contemporary Survey**

机器学习网络中的对抗性攻击与防御：当代综述 cs.LG

46 pages, 21 figures

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06302v1) [paper-pdf](http://arxiv.org/pdf/2303.06302v1)

**Authors**: Yulong Wang, Tong Sun, Shenghong Li, Xin Yuan, Wei Ni, Ekram Hossain, H. Vincent Poor

**Abstract**: Adversarial attacks and defenses in machine learning and deep neural network have been gaining significant attention due to the rapidly growing applications of deep learning in the Internet and relevant scenarios. This survey provides a comprehensive overview of the recent advancements in the field of adversarial attack and defense techniques, with a focus on deep neural network-based classification models. Specifically, we conduct a comprehensive classification of recent adversarial attack methods and state-of-the-art adversarial defense techniques based on attack principles, and present them in visually appealing tables and tree diagrams. This is based on a rigorous evaluation of the existing works, including an analysis of their strengths and limitations. We also categorize the methods into counter-attack detection and robustness enhancement, with a specific focus on regularization-based methods for enhancing robustness. New avenues of attack are also explored, including search-based, decision-based, drop-based, and physical-world attacks, and a hierarchical classification of the latest defense methods is provided, highlighting the challenges of balancing training costs with performance, maintaining clean accuracy, overcoming the effect of gradient masking, and ensuring method transferability. At last, the lessons learned and open challenges are summarized with future research opportunities recommended.

摘要: 由于深度学习在互联网和相关场景中的应用日益广泛，机器学习和深度神经网络中的对抗性攻击和防御已经得到了广泛的关注。这篇综述全面概述了对抗性攻击和防御技术领域的最新进展，重点介绍了基于深度神经网络的分类模型。具体地说，我们根据攻击原理对目前的对抗性攻击方法和最新的对抗性防御技术进行了全面的分类，并以视觉上吸引人的表格和树形图来呈现它们。这是基于对现有作品的严格评估，包括对它们的优点和局限性的分析。我们还将这些方法分为反攻击检测和稳健性增强两类，重点介绍了基于正则化的增强稳健性的方法。还探索了新的攻击途径，包括基于搜索、基于决策、基于Drop和物理世界的攻击，并提供了最新防御方法的分层分类，突出了在平衡训练成本和性能、保持干净准确性、克服梯度掩蔽的影响和确保方法可转移性方面的挑战。最后，总结了本研究的经验教训和面临的挑战，并对未来的研究方向进行了展望。



## **19. Investigating Stateful Defenses Against Black-Box Adversarial Examples**

黑箱对抗状态防御的研究实例 cs.CR

**SubmitDate**: 2023-03-11    [abs](http://arxiv.org/abs/2303.06280v1) [paper-pdf](http://arxiv.org/pdf/2303.06280v1)

**Authors**: Ryan Feng, Ashish Hooda, Neal Mangaokar, Kassem Fawaz, Somesh Jha, Atul Prakash

**Abstract**: Defending machine-learning (ML) models against white-box adversarial attacks has proven to be extremely difficult. Instead, recent work has proposed stateful defenses in an attempt to defend against a more restricted black-box attacker. These defenses operate by tracking a history of incoming model queries, and rejecting those that are suspiciously similar. The current state-of-the-art stateful defense Blacklight was proposed at USENIX Security '22 and claims to prevent nearly 100% of attacks on both the CIFAR10 and ImageNet datasets. In this paper, we observe that an attacker can significantly reduce the accuracy of a Blacklight-protected classifier (e.g., from 82.2% to 6.4% on CIFAR10) by simply adjusting the parameters of an existing black-box attack. Motivated by this surprising observation, since existing attacks were evaluated by the Blacklight authors, we provide a systematization of stateful defenses to understand why existing stateful defense models fail. Finally, we propose a stronger evaluation strategy for stateful defenses comprised of adaptive score and hard-label based black-box attacks. We use these attacks to successfully reduce even reconfigured versions of Blacklight to as low as 0% robust accuracy.

摘要: 保护机器学习(ML)模型免受白盒对手攻击已被证明是极其困难的。相反，最近的工作提出了状态防御，试图防御更受限制的黑匣子攻击者。这些防御通过跟踪传入模型查询的历史，并拒绝那些可疑的相似查询来运行。目前最先进的状态防御Blacklight是在USENIX Security‘22上提出的，声称可以防止对CIFAR10和ImageNet数据集的近100%攻击。在本文中，我们观察到攻击者可以通过简单地调整现有黑盒攻击的参数来显著降低受Blacklight保护的分类器的准确率(例如，在CIFAR10上从82.2%降低到6.4%)。出于这一令人惊讶的观察，由于现有攻击是由Blacklight作者评估的，我们提供了状态防御的系统化，以了解现有状态防御模型失败的原因。最后，提出了一种更强的状态防御评估策略，该策略由自适应评分和基于硬标签的黑盒攻击组成。我们使用这些攻击成功地将重新配置的Blacklight版本降低到低至0%的稳健准确率。



## **20. Do we need entire training data for adversarial training?**

对抗性训练需要完整的训练数据吗？ cs.CV

6 pages, 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06241v1) [paper-pdf](http://arxiv.org/pdf/2303.06241v1)

**Authors**: Vipul Gupta, Apurva Narayan

**Abstract**: Deep Neural Networks (DNNs) are being used to solve a wide range of problems in many domains including safety-critical domains like self-driving cars and medical imagery. DNNs suffer from vulnerability against adversarial attacks. In the past few years, numerous approaches have been proposed to tackle this problem by training networks using adversarial training. Almost all the approaches generate adversarial examples for the entire training dataset, thus increasing the training time drastically. We show that we can decrease the training time for any adversarial training algorithm by using only a subset of training data for adversarial training. To select the subset, we filter the adversarially-prone samples from the training data. We perform a simple adversarial attack on all training examples to filter this subset. In this attack, we add a small perturbation to each pixel and a few grid lines to the input image.   We perform adversarial training on the adversarially-prone subset and mix it with vanilla training performed on the entire dataset. Our results show that when our method-agnostic approach is plugged into FGSM, we achieve a speedup of 3.52x on MNIST and 1.98x on the CIFAR-10 dataset with comparable robust accuracy. We also test our approach on state-of-the-art Free adversarial training and achieve a speedup of 1.2x in training time with a marginal drop in robust accuracy on the ImageNet dataset.

摘要: 深度神经网络(DNN)正被用来解决许多领域的广泛问题，包括自动驾驶汽车和医学成像等安全关键领域。DNN容易受到敌意攻击。在过去的几年里，已经提出了许多办法来解决这一问题，方法是使用对抗性训练来训练网络。几乎所有的方法都为整个训练数据集生成对抗性的样本，从而大大增加了训练时间。我们证明，只要使用训练数据的一个子集进行对抗性训练，就可以减少任何对抗性训练算法的训练时间。为了选择子集，我们从训练数据中过滤出易受攻击的样本。我们对所有训练样本执行简单的对抗性攻击来过滤这个子集。在这种攻击中，我们为每个像素添加一个小扰动，并在输入图像中添加一些网格线。我们对易发生对抗性的子集进行对抗性训练，并将其与在整个数据集上执行的普通训练混合。我们的结果表明，当我们的方法无关的方法被插入到FGSM中时，我们在MNIST上获得了3.52倍的加速比，在CIFAR-10数据集上获得了1.98倍的加速比，并且具有相当的鲁棒性。我们还在最先进的自由对手训练上测试了我们的方法，在ImageNet数据集上的稳健准确率略有下降的情况下，训练时间加速了1.2倍。



## **21. Turning Strengths into Weaknesses: A Certified Robustness Inspired Attack Framework against Graph Neural Networks**

变优势为劣势：一种经验证的图神经网络健壮性启发攻击框架 cs.CR

Accepted by CVPR 2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.06199v1) [paper-pdf](http://arxiv.org/pdf/2303.06199v1)

**Authors**: Binghui Wang, Meng Pang, Yun Dong

**Abstract**: Graph neural networks (GNNs) have achieved state-of-the-art performance in many graph learning tasks. However, recent studies show that GNNs are vulnerable to both test-time evasion and training-time poisoning attacks that perturb the graph structure. While existing attack methods have shown promising attack performance, we would like to design an attack framework to further enhance the performance. In particular, our attack framework is inspired by certified robustness, which was originally used by defenders to defend against adversarial attacks. We are the first, from the attacker perspective, to leverage its properties to better attack GNNs. Specifically, we first derive nodes' certified perturbation sizes against graph evasion and poisoning attacks based on randomized smoothing, respectively. A larger certified perturbation size of a node indicates this node is theoretically more robust to graph perturbations. Such a property motivates us to focus more on nodes with smaller certified perturbation sizes, as they are easier to be attacked after graph perturbations. Accordingly, we design a certified robustness inspired attack loss, when incorporated into (any) existing attacks, produces our certified robustness inspired attack counterpart. We apply our framework to the existing attacks and results show it can significantly enhance the existing base attacks' performance.

摘要: 图形神经网络(GNN)在许多图形学习任务中取得了最好的性能。然而，最近的研究表明，GNN容易受到测试时间逃避和训练时间中毒攻击，这些攻击扰乱了图的结构。虽然现有的攻击方法已经显示出良好的攻击性能，但我们希望设计一个攻击框架来进一步提高性能。特别是，我们的攻击框架的灵感来自认证的健壮性，这最初是防御者用来防御对手攻击的。从攻击者的角度来看，我们是第一个利用其特性更好地攻击GNN的人。具体地说，我们首先基于随机化平滑分别推导出针对图规避攻击和中毒攻击的节点认证扰动大小。节点的认证扰动大小越大，表明该节点在理论上对图扰动的鲁棒性更强。这样的性质促使我们更多地关注具有较小认证扰动大小的节点，因为它们在图扰动后更容易受到攻击。因此，我们设计了一个认证的健壮性启发攻击损失，当整合到(任何)现有攻击中时，产生我们认证的健壮性启发攻击对手。我们将该框架应用于现有的攻击中，结果表明，该框架可以显著提高现有的基本攻击的性能。



## **22. Harnessing the Speed and Accuracy of Machine Learning to Advance Cybersecurity**

利用机器学习的速度和准确性提高网络安全 cs.CR

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2302.12415v2) [paper-pdf](http://arxiv.org/pdf/2302.12415v2)

**Authors**: Khatoon Mohammed

**Abstract**: As cyber attacks continue to increase in frequency and sophistication, detecting malware has become a critical task for maintaining the security of computer systems. Traditional signature-based methods of malware detection have limitations in detecting complex and evolving threats. In recent years, machine learning (ML) has emerged as a promising solution to detect malware effectively. ML algorithms are capable of analyzing large datasets and identifying patterns that are difficult for humans to identify. This paper presents a comprehensive review of the state-of-the-art ML techniques used in malware detection, including supervised and unsupervised learning, deep learning, and reinforcement learning. We also examine the challenges and limitations of ML-based malware detection, such as the potential for adversarial attacks and the need for large amounts of labeled data. Furthermore, we discuss future directions in ML-based malware detection, including the integration of multiple ML algorithms and the use of explainable AI techniques to enhance the interpret ability of ML-based detection systems. Our research highlights the potential of ML-based techniques to improve the speed and accuracy of malware detection, and contribute to enhancing cybersecurity

摘要: 随着网络攻击的频率和复杂性不断增加，检测恶意软件已成为维护计算机系统安全的关键任务。传统的基于特征码的恶意软件检测方法在检测复杂和不断变化的威胁方面存在局限性。近年来，机器学习作为一种有效检测恶意软件的解决方案应运而生。ML算法能够分析大型数据集，并识别人类难以识别的模式。本文对恶意软件检测中使用的最大似然学习技术进行了全面的综述，包括监督学习和非监督学习、深度学习和强化学习。我们还研究了基于ML的恶意软件检测的挑战和局限性，例如潜在的对抗性攻击和对大量标记数据的需求。此外，我们还讨论了基于ML的恶意软件检测的未来发展方向，包括集成多种ML算法和使用可解释人工智能技术来增强基于ML的检测系统的解释能力。我们的研究突出了基于ML的技术在提高恶意软件检测的速度和准确性方面的潜力，并有助于增强网络安全



## **23. Learning the Legibility of Visual Text Perturbations**

学习视觉文本扰动的易读性 cs.CL

14 pages, 7 figures. Accepted at EACL 2023 (main, long)

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05077v2) [paper-pdf](http://arxiv.org/pdf/2303.05077v2)

**Authors**: Dev Seth, Rickard Stureborg, Danish Pruthi, Bhuwan Dhingra

**Abstract**: Many adversarial attacks in NLP perturb inputs to produce visually similar strings ('ergo' $\rightarrow$ '$\epsilon$rgo') which are legible to humans but degrade model performance. Although preserving legibility is a necessary condition for text perturbation, little work has been done to systematically characterize it; instead, legibility is typically loosely enforced via intuitions around the nature and extent of perturbations. Particularly, it is unclear to what extent can inputs be perturbed while preserving legibility, or how to quantify the legibility of a perturbed string. In this work, we address this gap by learning models that predict the legibility of a perturbed string, and rank candidate perturbations based on their legibility. To do so, we collect and release LEGIT, a human-annotated dataset comprising the legibility of visually perturbed text. Using this dataset, we build both text- and vision-based models which achieve up to $0.91$ F1 score in predicting whether an input is legible, and an accuracy of $0.86$ in predicting which of two given perturbations is more legible. Additionally, we discover that legible perturbations from the LEGIT dataset are more effective at lowering the performance of NLP models than best-known attack strategies, suggesting that current models may be vulnerable to a broad range of perturbations beyond what is captured by existing visual attacks. Data, code, and models are available at https://github.com/dvsth/learning-legibility-2023.

摘要: NLP中的许多对抗性攻击会扰乱输入以产生视觉上相似的字符串(‘ergo’$\right tarrow$‘$\epsilon$rgo’)，这些字符串人类可读，但会降低模型性能。尽管保持易读性是文本扰动的必要条件，但几乎没有做过系统地描述它的工作；相反，易读性通常是通过围绕扰动的性质和程度的直觉来松散地强制执行的。特别是，目前还不清楚在保持易读性的同时，输入可以被干扰到什么程度，也不清楚如何量化被干扰的字符串的可读性。在这项工作中，我们通过学习预测扰动字符串的可读性的模型来解决这一差距，并根据它们的可读性对候选扰动进行排名。为了做到这一点，我们收集并发布Legit，这是一个人类注释的数据集，包括视觉上受到干扰的文本的可读性。使用这个数据集，我们建立了基于文本和基于视觉的模型，在预测输入是否清晰方面达到了高达0.91美元的F1分数，在预测两个给定扰动中的哪一个更易读方面达到了0.86美元的精度。此外，我们发现，与最著名的攻击策略相比，来自合法数据集的可识别的扰动在降低NLP模型的性能方面更有效，这表明当前的模型可能容易受到现有视觉攻击捕获的更大范围的扰动的影响。有关数据、代码和模型，请访问https://github.com/dvsth/learning-legibility-2023.



## **24. Exploring the Relationship between Architecture and Adversarially Robust Generalization**

探索体系结构和相反的健壮性泛化之间的关系 cs.LG

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2209.14105v2) [paper-pdf](http://arxiv.org/pdf/2209.14105v2)

**Authors**: Aishan Liu, Shiyu Tang, Siyuan Liang, Ruihao Gong, Boxi Wu, Xianglong Liu, Dacheng Tao

**Abstract**: Adversarial training has been demonstrated to be one of the most effective remedies for defending adversarial examples, yet it often suffers from the huge robustness generalization gap on unseen testing adversaries, deemed as the adversarially robust generalization problem. Despite the preliminary understandings devoted to adversarially robust generalization, little is known from the architectural perspective. To bridge the gap, this paper for the first time systematically investigated the relationship between adversarially robust generalization and architectural design. Inparticular, we comprehensively evaluated 20 most representative adversarially trained architectures on ImageNette and CIFAR-10 datasets towards multiple `p-norm adversarial attacks. Based on the extensive experiments, we found that, under aligned settings, Vision Transformers (e.g., PVT, CoAtNet) often yield better adversarially robust generalization while CNNs tend to overfit on specific attacks and fail to generalize on multiple adversaries. To better understand the nature behind it, we conduct theoretical analysis via the lens of Rademacher complexity. We revealed the fact that the higher weight sparsity contributes significantly towards the better adversarially robust generalization of Transformers, which can be often achieved by the specially-designed attention blocks. We hope our paper could help to better understand the mechanism for designing robust DNNs. Our model weights can be found at http://robust.art.

摘要: 对抗性训练已经被证明是防御对抗性例子的最有效的补救方法之一，然而它经常在看不见的测试对手上遭受巨大的健壮性泛化差距，被认为是对抗性健壮性泛化问题。尽管对相反的健壮性泛化有了初步的理解，但从体系结构的角度来看却知之甚少。为了弥补这一差距，本文首次系统地研究了逆稳性泛化与建筑设计之间的关系。特别是，我们在ImageNette和CIFAR-10数据集上全面评估了20种最具代表性的经过对手训练的体系结构，以应对多个p范数对手攻击。基于大量的实验，我们发现，在一致的设置下，Vision Transformers(如PVT，CoAtNet)往往能产生更好的对抗健壮性泛化，而CNN往往对特定攻击过于适应，无法对多个对手泛化。为了更好地理解其背后的本质，我们通过Rademacher复杂性的镜头进行了理论分析。我们揭示的事实是，较高的权重稀疏性显著有助于更好的逆境稳健的变形金刚泛化，这通常可以通过特殊设计的注意块来实现。我们希望我们的论文能够帮助我们更好地理解设计健壮DNN的机制。我们的模型重量可以在http://robust.art.上找到



## **25. Machine Learning Security in Industry: A Quantitative Survey**

机器学习在工业中的安全性：一项定量调查 cs.LG

Accepted at TIFS, version with more detailed appendix containing more  detailed statistical results. 17 pages, 6 tables and 4 figures

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2207.05164v2) [paper-pdf](http://arxiv.org/pdf/2207.05164v2)

**Authors**: Kathrin Grosse, Lukas Bieringer, Tarek Richard Besold, Battista Biggio, Katharina Krombholz

**Abstract**: Despite the large body of academic work on machine learning security, little is known about the occurrence of attacks on machine learning systems in the wild. In this paper, we report on a quantitative study with 139 industrial practitioners. We analyze attack occurrence and concern and evaluate statistical hypotheses on factors influencing threat perception and exposure. Our results shed light on real-world attacks on deployed machine learning. On the organizational level, while we find no predictors for threat exposure in our sample, the amount of implement defenses depends on exposure to threats or expected likelihood to become a target. We also provide a detailed analysis of practitioners' replies on the relevance of individual machine learning attacks, unveiling complex concerns like unreliable decision making, business information leakage, and bias introduction into models. Finally, we find that on the individual level, prior knowledge about machine learning security influences threat perception. Our work paves the way for more research about adversarial machine learning in practice, but yields also insights for regulation and auditing.

摘要: 尽管有大量关于机器学习安全的学术工作，但人们对野外发生的针对机器学习系统的攻击知之甚少。在本文中，我们报告了一项对139名工业从业者的定量研究。我们分析了攻击的发生和关注，并对影响威胁感知和暴露的因素进行了统计假设评估。我们的结果揭示了对部署的机器学习的真实世界攻击。在组织层面上，虽然我们在样本中没有发现威胁暴露的预测因素，但实施防御的数量取决于威胁暴露或成为目标的预期可能性。我们还提供了对从业者对单个机器学习攻击相关性的回复的详细分析，揭示了不可靠的决策、商业信息泄露和模型中的偏见引入等复杂问题。最后，我们发现在个体层面上，关于机器学习安全的先验知识会影响威胁感知。我们的工作为在实践中对对抗性机器学习进行更多的研究铺平了道路，但也为监管和审计提供了见解。



## **26. TrojDiff: Trojan Attacks on Diffusion Models with Diverse Targets**

TrojDiff：对具有不同目标的扩散模型的木马攻击 cs.LG

CVPR2023

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05762v1) [paper-pdf](http://arxiv.org/pdf/2303.05762v1)

**Authors**: Weixin Chen, Dawn Song, Bo Li

**Abstract**: Diffusion models have achieved great success in a range of tasks, such as image synthesis and molecule design. As such successes hinge on large-scale training data collected from diverse sources, the trustworthiness of these collected data is hard to control or audit. In this work, we aim to explore the vulnerabilities of diffusion models under potential training data manipulations and try to answer: How hard is it to perform Trojan attacks on well-trained diffusion models? What are the adversarial targets that such Trojan attacks can achieve? To answer these questions, we propose an effective Trojan attack against diffusion models, TrojDiff, which optimizes the Trojan diffusion and generative processes during training. In particular, we design novel transitions during the Trojan diffusion process to diffuse adversarial targets into a biased Gaussian distribution and propose a new parameterization of the Trojan generative process that leads to an effective training objective for the attack. In addition, we consider three types of adversarial targets: the Trojaned diffusion models will always output instances belonging to a certain class from the in-domain distribution (In-D2D attack), out-of-domain distribution (Out-D2D-attack), and one specific instance (D2I attack). We evaluate TrojDiff on CIFAR-10 and CelebA datasets against both DDPM and DDIM diffusion models. We show that TrojDiff always achieves high attack performance under different adversarial targets using different types of triggers, while the performance in benign environments is preserved. The code is available at https://github.com/chenweixin107/TrojDiff.

摘要: 扩散模型在图像合成和分子设计等一系列任务中取得了巨大的成功。由于这种成功取决于从不同来源收集的大规模培训数据，因此这些收集的数据的可信度很难控制或审计。在这项工作中，我们旨在探索扩散模型在潜在的训练数据操纵下的脆弱性，并试图回答：对训练有素的扩散模型执行特洛伊木马攻击的难度有多大？这种特洛伊木马攻击可以达到的敌对目标是什么？为了回答这些问题，我们提出了一种针对扩散模型的有效木马攻击方法TrojDiff，它优化了木马在训练过程中的传播和生成过程。特别是，我们在木马传播过程中设计了新颖的转变，将敌对目标扩散到有偏高斯分布中，并提出了一种新的木马生成过程的参数化，从而为攻击提供了有效的训练目标。此外，我们考虑了三种类型的敌意目标：特洛伊木马扩散模型总是从域内分布(In-D2D攻击)、域外分布(Out-D2D-攻击)和一个特定实例(D2I攻击)输出属于某一类的实例。我们在CIFAR-10和CelebA数据集上根据DDPM和DDIM扩散模型评估了TrojDiff。我们证明了TrojDiff在使用不同类型的触发器时，在不同的敌意目标下都能获得较高的攻击性能，而在良性环境下的性能保持不变。代码可在https://github.com/chenweixin107/TrojDiff.上获得



## **27. MIXPGD: Hybrid Adversarial Training for Speech Recognition Systems**

MIXPGD：语音识别系统的混合对抗性训练 cs.SD

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05758v1) [paper-pdf](http://arxiv.org/pdf/2303.05758v1)

**Authors**: Aminul Huq, Weiyi Zhang, Xiaolin Hu

**Abstract**: Automatic speech recognition (ASR) systems based on deep neural networks are weak against adversarial perturbations. We propose mixPGD adversarial training method to improve the robustness of the model for ASR systems. In standard adversarial training, adversarial samples are generated by leveraging supervised or unsupervised methods. We merge the capabilities of both supervised and unsupervised approaches in our method to generate new adversarial samples which aid in improving model robustness. Extensive experiments and comparison across various state-of-the-art defense methods and adversarial attacks have been performed to show that mixPGD gains 4.1% WER of better performance than previous best performing models under white-box adversarial attack setting. We tested our proposed defense method against both white-box and transfer based black-box attack settings to ensure that our defense strategy is robust against various types of attacks. Empirical results on several adversarial attacks validate the effectiveness of our proposed approach.

摘要: 基于深度神经网络的自动语音识别(ASR)系统对敌意干扰的抵抗能力较弱。为了提高ASR系统模型的健壮性，我们提出了混合PGD对抗训练方法。在标准的对抗性训练中，对抗性样本是通过利用监督或非监督方法生成的。在我们的方法中，我们融合了监督和非监督方法的能力来生成新的对抗性样本，这有助于提高模型的稳健性。实验结果表明，在白盒对抗攻击环境下，MixPGD算法比以往性能最好的模型提高了4.1%的WER。我们针对白盒攻击和基于传输的黑盒攻击设置对我们提出的防御方法进行了测试，以确保我们的防御策略对各种类型的攻击都是健壮的。在几个对抗性攻击上的实验结果验证了该方法的有效性。



## **28. Boosting Adversarial Attacks by Leveraging Decision Boundary Information**

利用决策边界信息增强对抗性攻击 cs.CV

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.05719v1) [paper-pdf](http://arxiv.org/pdf/2303.05719v1)

**Authors**: Boheng Zeng, LianLi Gao, QiLong Zhang, ChaoQun Li, JingKuan Song, ShuaiQi Jing

**Abstract**: Due to the gap between a substitute model and a victim model, the gradient-based noise generated from a substitute model may have low transferability for a victim model since their gradients are different. Inspired by the fact that the decision boundaries of different models do not differ much, we conduct experiments and discover that the gradients of different models are more similar on the decision boundary than in the original position. Moreover, since the decision boundary in the vicinity of an input image is flat along most directions, we conjecture that the boundary gradients can help find an effective direction to cross the decision boundary of the victim models. Based on it, we propose a Boundary Fitting Attack to improve transferability. Specifically, we introduce a method to obtain a set of boundary points and leverage the gradient information of these points to update the adversarial examples. Notably, our method can be combined with existing gradient-based methods. Extensive experiments prove the effectiveness of our method, i.e., improving the success rate by 5.6% against normally trained CNNs and 14.9% against defense CNNs on average compared to state-of-the-art transfer-based attacks. Further we compare transformers with CNNs, the results indicate that transformers are more robust than CNNs. However, our method still outperforms existing methods when attacking transformers. Specifically, when using CNNs as substitute models, our method obtains an average attack success rate of 58.2%, which is 10.8% higher than other state-of-the-art transfer-based attacks.

摘要: 由于替换模型和受害者模型之间的间隙，从替换模型生成的基于梯度的噪声对于受害者模型可能具有低的可转移性，因为它们的梯度不同。受不同模型决策边界差异不大的启发，我们进行了实验，发现不同模型在决策边界上的梯度比原始位置更相似。此外，由于输入图像附近的决策边界在大多数方向上是平坦的，我们推测边界梯度可以帮助找到一个有效的方向来越过受害者模型的决策边界。在此基础上，提出了一种边界拟合攻击来提高可转移性。具体地说，我们介绍了一种获取一组边界点的方法，并利用这些点的梯度信息来更新对抗性实例。值得注意的是，我们的方法可以与现有的基于梯度的方法相结合。大量的实验证明了该方法的有效性，即与最先进的基于传输的攻击相比，对正常训练的CNN的成功率平均提高了5.6%，对防御CNN的成功率平均提高了14.9%。此外，我们将变压器和CNN进行了比较，结果表明，变压器比CNN更稳健。然而，在攻击变压器时，我们的方法仍然优于现有的方法。具体地说，当使用CNN作为替代模型时，我们的方法获得了58.2%的平均攻击成功率，比其他最先进的基于传输的攻击高出10.8%。



## **29. On the Feasibility of Specialized Ability Stealing for Large Language Code Models**

论大型语言代码模型专业能力窃取的可行性 cs.SE

11 pages

**SubmitDate**: 2023-03-10    [abs](http://arxiv.org/abs/2303.03012v2) [paper-pdf](http://arxiv.org/pdf/2303.03012v2)

**Authors**: Zongjie Li, Chaozheng Wang, Pingchuan Ma, Chaowei Liu, Shuai Wang, Daoyuan Wu, Cuiyun Gao

**Abstract**: Recent progress in large language code models (LLCMs) has led to a dramatic surge in the use of software development. Nevertheless, it is widely known that training a well-performed LLCM requires a plethora of workforce for collecting the data and high quality annotation. Additionally, the training dataset may be proprietary (or partially open source to the public), and the training process is often conducted on a large-scale cluster of GPUs with high costs. Inspired by the recent success of imitation attacks in stealing computer vision and natural language models, this work launches the first imitation attack on LLCMs: by querying a target LLCM with carefully-designed queries and collecting the outputs, the adversary can train an imitation model that manifests close behavior with the target LLCM. We systematically investigate the effectiveness of launching imitation attacks under different query schemes and different LLCM tasks. We also design novel methods to polish the LLCM outputs, resulting in an effective imitation training process. We summarize our findings and provide lessons harvested in this study that can help better depict the attack surface of LLCMs. Our research contributes to the growing body of knowledge on imitation attacks and defenses in deep neural models, particularly in the domain of code related tasks.

摘要: 大型语言代码模型(LLCM)的最新进展导致软件开发的使用激增。然而，众所周知，培训一个表现良好的LLCM需要大量的劳动力来收集数据和高质量的注释。此外，训练数据集可能是专有的(或部分向公众开放源代码)，并且训练过程通常在成本较高的大规模GPU集群上进行。受最近成功窃取计算机视觉和自然语言模型的模仿攻击的启发，该工作对LLCM发起了第一次模仿攻击：通过使用精心设计的查询来查询目标LLCM并收集输出，对手可以训练出与目标LLCM表现出密切行为的模仿模型。系统地研究了在不同的查询方案和不同的LLCM任务下发起模仿攻击的有效性。我们还设计了新的方法来完善LLCM的输出，从而产生了一个有效的模拟训练过程。我们总结了我们的发现，并提供了在这项研究中获得的教训，有助于更好地描述LLCM的攻击面。我们的研究有助于在深层神经模型中，特别是在与代码相关的任务领域中，关于模仿攻击和防御的知识不断增长。



## **30. NoiseCAM: Explainable AI for the Boundary Between Noise and Adversarial Attacks**

NoiseCAM：噪声和对抗性攻击之间界限的可解释人工智能 cs.LG

Submitted to IEEE Fuzzy 2023. arXiv admin note: text overlap with  arXiv:2303.06032

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.06151v1) [paper-pdf](http://arxiv.org/pdf/2303.06151v1)

**Authors**: Wenkai Tan, Justus Renkhoff, Alvaro Velasquez, Ziyu Wang, Lusi Li, Jian Wang, Shuteng Niu, Fan Yang, Yongxin Liu, Houbing Song

**Abstract**: Deep Learning (DL) and Deep Neural Networks (DNNs) are widely used in various domains. However, adversarial attacks can easily mislead a neural network and lead to wrong decisions. Defense mechanisms are highly preferred in safety-critical applications. In this paper, firstly, we use the gradient class activation map (GradCAM) to analyze the behavior deviation of the VGG-16 network when its inputs are mixed with adversarial perturbation or Gaussian noise. In particular, our method can locate vulnerable layers that are sensitive to adversarial perturbation and Gaussian noise. We also show that the behavior deviation of vulnerable layers can be used to detect adversarial examples. Secondly, we propose a novel NoiseCAM algorithm that integrates information from globally and pixel-level weighted class activation maps. Our algorithm is susceptible to adversarial perturbations and will not respond to Gaussian random noise mixed in the inputs. Third, we compare detecting adversarial examples using both behavior deviation and NoiseCAM, and we show that NoiseCAM outperforms behavior deviation modeling in its overall performance. Our work could provide a useful tool to defend against certain adversarial attacks on deep neural networks.

摘要: 深度学习和深度神经网络在各个领域有着广泛的应用。然而，敌意攻击很容易误导神经网络，导致错误的决策。防御机制在安全关键型应用中非常受欢迎。首先，我们使用梯度类激活映射(GradCAM)来分析VGG-16网络在输入混合对抗性扰动或高斯噪声时的行为偏差。特别是，我们的方法可以定位对对抗性扰动和高斯噪声敏感的易受攻击的层。我们还表明，易受攻击层的行为偏差可以用来检测敌意示例。其次，我们提出了一种新的NoiseCAM算法，该算法综合了全局和像素级加权类激活图的信息。我们的算法容易受到对抗性扰动的影响，并且不会对输入中混合的高斯随机噪声做出响应。第三，比较了行为偏差和NoiseCAM两种方法检测恶意实例的性能，结果表明，NoiseCAM在整体性能上优于行为偏差建模。我们的工作可以提供一种有用的工具来防御针对深层神经网络的某些敌意攻击。



## **31. Evaluating the Robustness of Conversational Recommender Systems by Adversarial Examples**

用对抗性实例评价会话推荐系统的健壮性 cs.IR

10 pages

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05575v1) [paper-pdf](http://arxiv.org/pdf/2303.05575v1)

**Authors**: Ali Montazeralghaem, James Allan

**Abstract**: Conversational recommender systems (CRSs) are improving rapidly, according to the standard recommendation accuracy metrics. However, it is essential to make sure that these systems are robust in interacting with users including regular and malicious users who want to attack the system by feeding the system modified input data. In this paper, we propose an adversarial evaluation scheme including four scenarios in two categories and automatically generate adversarial examples to evaluate the robustness of these systems in the face of different input data. By executing these adversarial examples we can compare the ability of different conversational recommender systems to satisfy the user's preferences. We evaluate three CRSs by the proposed adversarial examples on two datasets. Our results show that none of these systems are robust and reliable to the adversarial examples.

摘要: 根据标准的推荐准确度指标，会话推荐系统(CRSS)正在迅速改进。但是，必须确保这些系统在与用户(包括想要通过向系统提供修改后的输入数据来攻击系统的普通用户和恶意用户)交互时保持健壮。本文提出了一种包含两类四个场景的对抗性评估方案，并自动生成对抗性实例来评估这些系统在面对不同输入数据时的健壮性。通过执行这些对抗性的例子，我们可以比较不同的会话推荐系统满足用户偏好的能力。我们通过在两个数据集上提出的对抗性实例对三个CRSS进行了评估。我们的结果表明，这些系统对对抗性例子都不是健壮和可靠的。



## **32. Efficient Certified Training and Robustness Verification of Neural ODEs**

神经ODE的高效认证训练和稳健性验证 cs.LG

Accepted at ICLR23

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05246v1) [paper-pdf](http://arxiv.org/pdf/2303.05246v1)

**Authors**: Mustafa Zeqiri, Mark Niklas Müller, Marc Fischer, Martin Vechev

**Abstract**: Neural Ordinary Differential Equations (NODEs) are a novel neural architecture, built around initial value problems with learned dynamics which are solved during inference. Thought to be inherently more robust against adversarial perturbations, they were recently shown to be vulnerable to strong adversarial attacks, highlighting the need for formal guarantees. However, despite significant progress in robustness verification for standard feed-forward architectures, the verification of high dimensional NODEs remains an open problem. In this work, we address this challenge and propose GAINS, an analysis framework for NODEs combining three key ideas: (i) a novel class of ODE solvers, based on variable but discrete time steps, (ii) an efficient graph representation of solver trajectories, and (iii) a novel abstraction algorithm operating on this graph representation. Together, these advances enable the efficient analysis and certified training of high-dimensional NODEs, by reducing the runtime from an intractable $O(\exp(d)+\exp(T))$ to ${O}(d+T^2 \log^2T)$ in the dimensionality $d$ and integration time $T$. In an extensive evaluation on computer vision (MNIST and FMNIST) and time-series forecasting (PHYSIO-NET) problems, we demonstrate the effectiveness of both our certified training and verification methods.

摘要: 神经常微分方程组(节点)是一种新型的神经结构，它建立在具有学习动力学的初值问题周围，并在推理过程中求解。它们被认为在对抗对手干扰方面天生更健壮，但最近被证明容易受到强大的对手攻击，这突显了正式担保的必要性。然而，尽管在标准前馈结构的健壮性验证方面取得了显著进展，但高维节点的验证仍然是一个悬而未决的问题。在这项工作中，我们解决了这一挑战，并提出了一个结合了三个关键思想的节点分析框架：(I)一类新的基于可变但离散时间步长的ODE求解器，(Ii)求解器轨迹的有效图表示，以及(Iii)在该图表示上操作的新抽象算法。总而言之，这些改进通过将运行时间从难以处理的$O(\exp(D)+\exp(T))$减少到维度$d$和积分时间$T$的${O}(d+T^2\log^2T)$，从而实现对高维节点的有效分析和认证训练。在对计算机视觉(MNIST和FMNIST)和时间序列预测(Physio-Net)问题的广泛评估中，我们展示了我们认证的培训和验证方法的有效性。



## **33. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors**

隐形补丁：对物体探测器的自然主义黑箱对抗性攻击 cs.CV

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04238v2) [paper-pdf](http://arxiv.org/pdf/2303.04238v2)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Adversarial attacks on deep-learning models have been receiving increased attention in recent years. Work in this area has mostly focused on gradient-based techniques, so-called white-box attacks, wherein the attacker has access to the targeted model's internal parameters; such an assumption is usually unrealistic in the real world. Some attacks additionally use the entire pixel space to fool a given model, which is neither practical nor physical (i.e., real-world). On the contrary, we propose herein a gradient-free method that uses the learned image manifold of a pretrained generative adversarial network (GAN) to generate naturalistic physical adversarial patches for object detectors. We show that our proposed method works both digitally and physically.

摘要: 近年来，针对深度学习模型的对抗性攻击受到越来越多的关注。这一领域的工作主要集中在基于梯度的技术，即所谓的白盒攻击，即攻击者可以访问目标模型的内部参数；这种假设在现实世界中通常是不现实的。一些攻击还使用整个像素空间来愚弄给定的模型，这既不实用也不物理(即，现实世界)。相反，我们在这里提出了一种无梯度的方法，它使用预先训练的生成性对抗性网络(GAN)的学习图像流形来为目标检测器生成自然的物理对抗性斑块。我们证明了我们提出的方法在数字和物理上都是有效的。



## **34. On Robustness of Prompt-based Semantic Parsing with Large Pre-trained Language Model: An Empirical Study on Codex**

基于大预训练语言模型的基于提示的语义分析的稳健性研究--基于CODEX的实证研究 cs.CL

Accepted at EACL2023 (main)

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2301.12868v3) [paper-pdf](http://arxiv.org/pdf/2301.12868v3)

**Authors**: Terry Yue Zhuo, Zhuang Li, Yujin Huang, Fatemeh Shiri, Weiqing Wang, Gholamreza Haffari, Yuan-Fang Li

**Abstract**: Semantic parsing is a technique aimed at constructing a structured representation of the meaning of a natural-language question. Recent advancements in few-shot language models trained on code have demonstrated superior performance in generating these representations compared to traditional unimodal language models, which are trained on downstream tasks. Despite these advancements, existing fine-tuned neural semantic parsers are susceptible to adversarial attacks on natural-language inputs. While it has been established that the robustness of smaller semantic parsers can be enhanced through adversarial training, this approach is not feasible for large language models in real-world scenarios, as it requires both substantial computational resources and expensive human annotation on in-domain semantic parsing data. This paper presents the first empirical study on the adversarial robustness of a large prompt-based language model of code, \codex. Our results demonstrate that the state-of-the-art (SOTA) code-language models are vulnerable to carefully crafted adversarial examples. To address this challenge, we propose methods for improving robustness without the need for significant amounts of labeled data or heavy computational resources.

摘要: 语义分析是一种旨在构建自然语言问题意义的结构化表示的技术。与在下游任务上训练的传统单峰语言模型相比，在生成这些表示方面，少数几次语言模型的最新进展已经显示出更好的性能。尽管有这些进步，但现有的微调神经语义解析器很容易受到自然语言输入的对抗性攻击。虽然已经确定可以通过对抗性训练来增强较小语义解析器的稳健性，但这种方法对于真实世界场景中的大型语言模型是不可行的，因为它需要大量的计算资源和昂贵的人工对域内语义解析数据的注释。本文首次对基于提示的大型代码语言模型CODEX的对手健壮性进行了实证研究。我们的结果表明，最先进的(SOTA)代码语言模型容易受到精心设计的敌意示例的攻击。为了应对这一挑战，我们提出了在不需要大量标记数据或大量计算资源的情况下提高稳健性的方法。



## **35. Towards Good Practices in Evaluating Transfer Adversarial Attacks**

在评估转会对抗性攻击方面的良好做法 cs.CR

Our code and a list of categorized attacks are publicly available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2211.09565v2) [paper-pdf](http://arxiv.org/pdf/2211.09565v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes

**Abstract**: Transfer adversarial attacks raise critical security concerns in real-world, black-box scenarios. However, the actual progress of this field is difficult to assess due to two common limitations in existing evaluations. First, different methods are often not systematically and fairly evaluated in a one-to-one comparison. Second, only transferability is evaluated but another key attack property, stealthiness, is largely overlooked. In this work, we design good practices to address these limitations, and we present the first comprehensive evaluation of transfer attacks, covering 23 representative attacks against 9 defenses on ImageNet. In particular, we propose to categorize existing attacks into five categories, which enables our systematic category-wise analyses. These analyses lead to new findings that even challenge existing knowledge and also help determine the optimal attack hyperparameters for our attack-wise comprehensive evaluation. We also pay particular attention to stealthiness, by adopting diverse imperceptibility metrics and looking into new, finer-grained characteristics. Overall, our new insights into transferability and stealthiness lead to actionable good practices for future evaluations.

摘要: 在现实世界的黑盒场景中，传输敌意攻击会引发严重的安全问题。然而，由于现有评价中的两个共同限制，这一领域的实际进展很难评估。首先，不同的方法往往不能在一对一的比较中得到系统和公平的评估。其次，只评估了可转移性，但另一个关键攻击属性--隐蔽性在很大程度上被忽视了。在这项工作中，我们设计了良好的实践来解决这些限制，并提出了第一个全面的传输攻击评估，涵盖了23个典型攻击对9个防御ImageNet。特别是，我们建议将现有攻击分为五类，这使得我们能够进行系统的分类分析。这些分析导致了新的发现，甚至挑战了现有的知识，并有助于为我们的攻击智能综合评估确定最佳攻击超参数。我们还特别关注隐蔽性，采用了不同的隐蔽性度量标准，并研究了新的、更细粒度的特征。总体而言，我们对可转移性和隐蔽性的新见解为未来的评估提供了可操作的良好做法。



## **36. On the Robustness of Dataset Inference**

关于数据集推理的稳健性 cs.LG

17 pages, 5 tables, 4 figures

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2210.13631v2) [paper-pdf](http://arxiv.org/pdf/2210.13631v2)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs. Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique introduced at ICLR '21, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods. The authors of DI provided a correctness proof for linear (suspect) models. However, in the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI leads to FPs, with high confidence. Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade. Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.

摘要: 机器学习(ML)模型的训练成本很高，因为它们可能需要大量的数据、计算资源和技术专长。因此，它们构成了宝贵的知识产权，需要保护，不受想要窃取它们的对手的攻击。所有权验证技术允许模型盗窃攻击的受害者证明可疑模型实际上是从他们的模型中被盗的。虽然已经提出了一些基于水印或指纹的所有权验证技术，但它们大多在安全保证(装备良好的攻击者可以逃避验证)或计算代价方面存在不足。在ICLR‘21上引入的一种指纹技术，数据集推理(DI)，已经被证明比以前的方法提供了更好的稳健性和效率。DI的作者为线性(可疑)模型提供了正确性证明。然而，在相同的设置中，我们证明了DI存在高误报(FP)--它可能错误地识别使用来自相同分布的非重叠数据训练的独立模型作为被盗。我们进一步证明，在现实的、非线性的可疑模型中，依赖注入也会触发FP。然后，我们以很高的置信度从经验上证实了DI会导致FP。其次，我们证明了DI也存在假阴性(FN)--对手可以通过使用对抗性训练来调整被盗模型的决策边界来愚弄DI，从而导致FN。为此，我们演示了DI无法识别从窃取的数据集中恶意训练的模型--DI最难逃避的设置。最后，我们讨论了我们的发现的含义，基于指纹的所有权验证总体上的可行性，并对未来的工作提出了方向。



## **37. Identification of Systematic Errors of Image Classifiers on Rare Subgroups**

稀有子群上图像分类器系统误差的辨识 cs.CV

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.05072v1) [paper-pdf](http://arxiv.org/pdf/2303.05072v1)

**Authors**: Jan Hendrik Metzen, Robin Hutmacher, N. Grace Hua, Valentyn Boreiko, Dan Zhang

**Abstract**: Despite excellent average-case performance of many image classifiers, their performance can substantially deteriorate on semantically coherent subgroups of the data that were under-represented in the training data. These systematic errors can impact both fairness for demographic minority groups as well as robustness and safety under domain shift. A major challenge is to identify such subgroups with subpar performance when the subgroups are not annotated and their occurrence is very rare. We leverage recent advances in text-to-image models and search in the space of textual descriptions of subgroups ("prompts") for subgroups where the target model has low performance on the prompt-conditioned synthesized data. To tackle the exponentially growing number of subgroups, we employ combinatorial testing. We denote this procedure as PromptAttack as it can be interpreted as an adversarial attack in a prompt space. We study subgroup coverage and identifiability with PromptAttack in a controlled setting and find that it identifies systematic errors with high accuracy. Thereupon, we apply PromptAttack to ImageNet classifiers and identify novel systematic errors on rare subgroups.

摘要: 尽管许多图像分类器的平均情况性能很好，但在训练数据中表示不足的数据的语义连贯子组上，它们的性能可能会显著恶化。这些系统性错误既会影响人口少数群体的公平性，也会影响领域转移下的稳健性和安全性。一个主要的挑战是在子组没有被注释并且它们的出现非常罕见的情况下，识别这样的子组具有低于标准的性能。我们利用文本到图像模型中的最新进展，并在目标模型对提示条件合成数据的性能较低的子组的子组(提示)的文本描述空间中进行搜索。为了解决指数级增长的子组数量，我们使用了组合测试。我们将这个过程表示为PromptAttack，因为它可以解释为提示空间中的对抗性攻击。我们在受控环境下研究了PromptAttack的子组复盖率和可识别性，发现它识别系统错误的准确率很高。于是，我们将PromptAttack应用于ImageNet分类器，并在稀有子群上识别出新的系统误差。



## **38. BeamAttack: Generating High-quality Textual Adversarial Examples through Beam Search and Mixed Semantic Spaces**

BeamAttack：通过Beam搜索和混合语义空间生成高质量的文本对抗性实例 cs.CL

PAKDD2023

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.07199v1) [paper-pdf](http://arxiv.org/pdf/2303.07199v1)

**Authors**: Hai Zhu, Qingyang Zhao, Yuren Wu

**Abstract**: Natural language processing models based on neural networks are vulnerable to adversarial examples. These adversarial examples are imperceptible to human readers but can mislead models to make the wrong predictions. In a black-box setting, attacker can fool the model without knowing model's parameters and architecture. Previous works on word-level attacks widely use single semantic space and greedy search as a search strategy. However, these methods fail to balance the attack success rate, quality of adversarial examples and time consumption. In this paper, we propose BeamAttack, a textual attack algorithm that makes use of mixed semantic spaces and improved beam search to craft high-quality adversarial examples. Extensive experiments demonstrate that BeamAttack can improve attack success rate while saving numerous queries and time, e.g., improving at most 7\% attack success rate than greedy search when attacking the examples from MR dataset. Compared with heuristic search, BeamAttack can save at most 85\% model queries and achieve a competitive attack success rate. The adversarial examples crafted by BeamAttack are highly transferable and can effectively improve model's robustness during adversarial training. Code is available at https://github.com/zhuhai-ustc/beamattack/tree/master

摘要: 基于神经网络的自然语言处理模型容易受到敌意例子的影响。这些对抗性的例子对人类读者来说是难以察觉的，但可能会误导模型做出错误的预测。在黑盒设置中，攻击者可以在不知道模型参数和体系结构的情况下愚弄模型。以往关于词级攻击的研究大多采用单一语义空间和贪婪搜索作为搜索策略。然而，这些方法没有在攻击成功率、对抗性实例的质量和时间消耗之间取得平衡。在本文中，我们提出了一种文本攻击算法BeamAttack，它利用混合语义空间和改进的BEAM搜索来生成高质量的敌意实例。大量实验表明，BeamAttack在提高攻击成功率的同时，节省了大量的查询和时间，例如，在攻击MR数据集中的示例时，攻击成功率最多比贪婪搜索提高7%。与启发式搜索相比，BeamAttack最多可以节省85个模型查询，并获得与之相当的攻击成功率。BeamAttack生成的对抗性实例具有很强的可移植性，能够有效提高模型在对抗性训练中的健壮性。代码可在https://github.com/zhuhai-ustc/beamattack/tree/master上找到



## **39. On the Risks of Stealing the Decoding Algorithms of Language Models**

论窃取语言模型译码算法的风险 cs.LG

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04729v2) [paper-pdf](http://arxiv.org/pdf/2303.04729v2)

**Authors**: Ali Naseh, Kalpesh Krishna, Mohit Iyyer, Amir Houmansadr

**Abstract**: A key component of generating text from modern language models (LM) is the selection and tuning of decoding algorithms. These algorithms determine how to generate text from the internal probability distribution generated by the LM. The process of choosing a decoding algorithm and tuning its hyperparameters takes significant time, manual effort, and computation, and it also requires extensive human evaluation. Therefore, the identity and hyperparameters of such decoding algorithms are considered to be extremely valuable to their owners. In this work, we show, for the first time, that an adversary with typical API access to an LM can steal the type and hyperparameters of its decoding algorithms at very low monetary costs. Our attack is effective against popular LMs used in text generation APIs, including GPT-2 and GPT-3. We demonstrate the feasibility of stealing such information with only a few dollars, e.g., $\$0.8$, $\$1$, $\$4$, and $\$40$ for the four versions of GPT-3.

摘要: 从现代语言模型(LM)生成文本的一个关键组件是解码算法的选择和调整。这些算法确定如何从LM生成的内部概率分布生成文本。选择解码算法和调整其超参数的过程需要大量的时间、人工和计算，还需要广泛的人工评估。因此，这种译码算法的恒等式和超参数被认为对它们的所有者非常有价值。在这项工作中，我们首次证明，具有典型API访问权限的攻击者可以以非常低的金钱成本窃取其解码算法的类型和超参数。我们的攻击对文本生成API中使用的流行LMS有效，包括GPT-2和GPT-3。我们证明了只需几美元即可窃取此类信息的可行性，例如，对于GPT-3的四个版本，仅需$0.8$、$1$、$4$和$40$。



## **40. Decision-BADGE: Decision-based Adversarial Batch Attack with Directional Gradient Estimation**

Decision-Bigge：方向梯度估计的基于决策的对抗性批量攻击 cs.CV

10 pages (8 pages except for references), 6 figures, 6 tables

**SubmitDate**: 2023-03-09    [abs](http://arxiv.org/abs/2303.04980v1) [paper-pdf](http://arxiv.org/pdf/2303.04980v1)

**Authors**: Geunhyeok Yu, Minwoo Jeon, Hyoseok Hwang

**Abstract**: The vulnerability of deep neural networks to adversarial examples has led to the rise in the use of adversarial attacks. While various decision-based and universal attack methods have been proposed, none have attempted to create a decision-based universal adversarial attack. This research proposes Decision-BADGE, which uses random gradient-free optimization and batch attack to generate universal adversarial perturbations for decision-based attacks. Multiple adversarial examples are combined to optimize a single universal perturbation, and the accuracy metric is reformulated into a continuous Hamming distance form. The effectiveness of accuracy metric as a loss function is demonstrated and mathematically proven. The combination of Decision-BADGE and the accuracy loss function performs better than both score-based image-dependent attack and white-box universal attack methods in terms of attack time efficiency. The research also shows that Decision-BADGE can successfully deceive unseen victims and accurately target specific classes.

摘要: 深层神经网络对对抗性例子的脆弱性导致了对抗性攻击的使用增加。虽然已经提出了各种基于决策的和通用的攻击方法，但还没有人试图创建基于决策的通用对抗性攻击。本研究提出了决策牌，它使用随机无梯度优化和批处理攻击来为基于决策的攻击生成通用的对抗性扰动。通过组合多个对抗性实例来优化单个普遍摄动，并将精度度量重新表示为连续的汉明距离形式。证明了精度度量作为损失函数的有效性，并从数学上进行了证明。决策标记和精度损失函数的结合在攻击时间效率上优于基于分数的图像依赖攻击和白盒通用攻击方法。研究还表明，决策徽章可以成功地欺骗看不见的受害者，并准确地瞄准特定的阶层。



## **41. Local Convolutions Cause an Implicit Bias towards High Frequency Adversarial Examples**

局部卷积导致对高频对抗性例子的隐性偏向 stat.ML

23 pages, 11 figures, 12 Tables

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2006.11440v5) [paper-pdf](http://arxiv.org/pdf/2006.11440v5)

**Authors**: Josue Ortega Caro, Yilong Ju, Ryan Pyle, Sourav Dey, Wieland Brendel, Fabio Anselmi, Ankit Patel

**Abstract**: Adversarial Attacks are still a significant challenge for neural networks. Recent work has shown that adversarial perturbations typically contain high-frequency features, but the root cause of this phenomenon remains unknown. Inspired by theoretical work on linear full-width convolutional models, we hypothesize that the local (i.e. bounded-width) convolutional operations commonly used in current neural networks are implicitly biased to learn high frequency features, and that this is one of the root causes of high frequency adversarial examples. To test this hypothesis, we analyzed the impact of different choices of linear and nonlinear architectures on the implicit bias of the learned features and the adversarial perturbations, in both spatial and frequency domains. We find that the high-frequency adversarial perturbations are critically dependent on the convolution operation because the spatially-limited nature of local convolutions induces an implicit bias towards high frequency features. The explanation for the latter involves the Fourier Uncertainty Principle: a spatially-limited (local in the space domain) filter cannot also be frequency-limited (local in the frequency domain). Furthermore, using larger convolution kernel sizes or avoiding convolutions (e.g. by using Vision Transformers architecture) significantly reduces this high frequency bias, but not the overall susceptibility to attacks. Looking forward, our work strongly suggests that understanding and controlling the implicit bias of architectures will be essential for achieving adversarial robustness.

摘要: 对抗性攻击仍然是神经网络面临的重大挑战。最近的研究表明，对抗性扰动通常包含高频特征，但这种现象的根本原因尚不清楚。受线性全宽度卷积模型理论工作的启发，我们假设当前神经网络中常用的局部(即有界宽度)卷积运算隐含地偏向于学习高频特征，这是高频对抗性例子的根本原因之一。为了验证这一假设，我们在空间域和频域分析了线性和非线性结构的不同选择对学习特征的内隐偏差和对抗性扰动的影响。我们发现，高频对抗性扰动严重依赖于卷积运算，因为局部卷积的空间有限性质导致了对高频特征的隐式偏差。对后者的解释涉及到傅里叶不确定原理：空间受限(空间域中的局部)滤波器不能也是频率受限的(频域中的局部)。此外，使用更大的卷积核大小或避免卷积(例如，通过使用Vision Transformers体系结构)可以显著减少这种高频偏差，但不会显著降低对攻击的总体易感性。展望未来，我们的工作强烈表明，理解和控制体系结构的隐含偏见将是实现对抗性健壮性的关键。



## **42. Immune Defense: A Novel Adversarial Defense Mechanism for Preventing the Generation of Adversarial Examples**

免疫防御：一种防止对抗性事例产生的新型对抗性防御机制 cs.CV

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.04502v1) [paper-pdf](http://arxiv.org/pdf/2303.04502v1)

**Authors**: Jinwei Wang, Hao Wu, Haihua Wang, Jiawei Zhang, Xiangyang Luo, Bin Ma

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial examples has been confirmed. Existing adversarial defenses primarily aim at preventing adversarial examples from attacking DNNs successfully, rather than preventing their generation. If the generation of adversarial examples is unregulated, images within reach are no longer secure and pose a threat to non-robust DNNs. Although gradient obfuscation attempts to address this issue, it has been shown to be circumventable. Therefore, we propose a novel adversarial defense mechanism, which is referred to as immune defense and is the example-based pre-defense. This mechanism applies carefully designed quasi-imperceptible perturbations to the raw images to prevent the generation of adversarial examples for the raw images, and thereby protecting both images and DNNs. These perturbed images are referred to as Immune Examples (IEs). In the white-box immune defense, we provide a gradient-based and an optimization-based approach, respectively. Additionally, the more complex black-box immune defense is taken into consideration. We propose Masked Gradient Sign Descent (MGSD) to reduce approximation error and stabilize the update to improve the transferability of IEs and thereby ensure their effectiveness against black-box adversarial attacks. The experimental results demonstrate that the optimization-based approach has superior performance and better visual quality in white-box immune defense. In contrast, the gradient-based approach has stronger transferability and the proposed MGSD significantly improve the transferability of baselines.

摘要: 深度神经网络(DNN)对敌意例子的脆弱性已被证实。现有的对抗性防御主要是为了防止敌意实例成功攻击DNN，而不是阻止它们的生成。如果敌意示例的生成不受控制，则触手可及的图像不再安全，并对非健壮的DNN构成威胁。虽然渐变模糊试图解决这个问题，但它已被证明是可以规避的。因此，我们提出了一种新颖的对抗性防御机制，称为免疫防御，是基于实例的预防御。该机制将精心设计的准不可感知扰动应用于原始图像，以防止原始图像产生对抗性示例，从而保护图像和DNN。这些受干扰的图像被称为免疫样例(IE)。在白盒免疫防御中，我们分别提出了基于梯度的方法和基于优化的方法。此外，还考虑了更复杂的黑盒免疫防御。我们提出了掩蔽梯度符号下降算法(MGSD)来减少逼近误差，稳定更新，从而提高IES的可转移性，从而确保其对抗黑盒攻击的有效性。实验结果表明，基于优化的方法在白盒免疫防御中具有更好的性能和更好的视觉质量。相比之下，基于梯度的方法具有更强的可转移性，所提出的MGSD显著提高了基线的可转移性。



## **43. Dishing Out DoS: How to Disable and Secure the Starlink User Terminal**

拒绝服务：如何禁用和保护Starlink用户终端 cs.CR

6 pages, 2 figures; the first two authors contributed equally to this  paper

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.00582v2) [paper-pdf](http://arxiv.org/pdf/2303.00582v2)

**Authors**: Joshua Smailes, Edd Salkield, Sebastian Köhler, Simon Birnbach, Ivan Martinovic

**Abstract**: Satellite user terminals are a promising target for adversaries seeking to target satellite communication networks. Despite this, many protections commonly found in terrestrial routers are not present in some user terminals.   As a case study we audit the attack surface presented by the Starlink router's admin interface, using fuzzing to uncover a denial of service attack on the Starlink user terminal. We explore the attack's impact, particularly in the cases of drive-by attackers, and attackers that are able to maintain a continuous presence on the network. Finally, we discuss wider implications, looking at lessons learned in terrestrial router security, and how to properly implement them in this new context.

摘要: 卫星用户终端是寻求以卫星通信网络为目标的敌人的一个有希望的目标。尽管如此，地面路由器中常见的许多保护在一些用户终端中并不存在。作为一个案例研究，我们审计了Starlink路由器的管理界面呈现的攻击面，使用Fuzze发现了对Starlink用户终端的拒绝服务攻击。我们探讨了攻击的影响，特别是在路过攻击者和能够在网络上持续存在的攻击者的情况下。最后，我们将讨论更广泛的影响，总结地面路由器安全方面的经验教训，以及如何在新的环境中正确实施这些经验教训。



## **44. GLOW: Global Layout Aware Attacks on Object Detection**

Glow：针对对象检测的全局布局感知攻击 cs.CV

ICCV

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2302.14166v2) [paper-pdf](http://arxiv.org/pdf/2302.14166v2)

**Authors**: Buyu Liu, BaoJun, Jianping Fan, Xi Peng, Kui Ren, Jun Yu

**Abstract**: Adversarial attacks aim to perturb images such that a predictor outputs incorrect results. Due to the limited research in structured attacks, imposing consistency checks on natural multi-object scenes is a promising yet practical defense against conventional adversarial attacks. More desired attacks, to this end, should be able to fool defenses with such consistency checks. Therefore, we present the first approach GLOW that copes with various attack requests by generating global layout-aware adversarial attacks, in which both categorical and geometric layout constraints are explicitly established. Specifically, we focus on object detection task and given a victim image, GLOW first localizes victim objects according to target labels. And then it generates multiple attack plans, together with their context-consistency scores. Our proposed GLOW, on the one hand, is capable of handling various types of requests, including single or multiple victim objects, with or without specified victim objects. On the other hand, it produces a consistency score for each attack plan, reflecting the overall contextual consistency that both semantic category and global scene layout are considered. In experiment, we design multiple types of attack requests and validate our ideas on MS COCO and Pascal. Extensive experimental results demonstrate that we can achieve about 30$\%$ average relative improvement compared to state-of-the-art methods in conventional single object attack request; Moreover, our method outperforms SOTAs significantly on more generic attack requests by about 20$\%$ in average; Finally, our method produces superior performance under challenging zero-query black-box setting, or 20$\%$ better than SOTAs. Our code, model and attack requests would be made available.

摘要: 敌意攻击的目的是扰乱图像，使预测器输出错误的结果。由于结构化攻击的研究有限，对自然多目标场景进行一致性检查是对抗传统对手攻击的一种很有前途的实用防御方法。为了达到这个目的，更多想要的攻击应该能够通过这样的一致性检查来愚弄防御。因此，我们提出了第一种方法GLOW，它通过生成全局布局感知的对抗性攻击来应对各种攻击请求，其中明确地建立了分类布局约束和几何布局约束。具体地说，我们针对目标检测任务，在给定受害者图像的情况下，GLOW首先根据目标标签定位受害者对象。然后，它生成多个攻击计划，以及它们的上下文一致性分数。一方面，我们提出的Glow能够处理各种类型的请求，包括单个或多个受害者对象，具有或不具有指定的受害者对象。另一方面，它为每个攻击计划生成一个一致性分数，反映了语义类别和全局场景布局都被考虑的整体上下文一致性。在实验中，我们设计了多种类型的攻击请求，并在MS Coco和Pascal上验证了我们的想法。大量的实验结果表明，在传统的单对象攻击请求中，我们的方法可以比现有的方法平均提高约30美元；此外，在更一般的攻击请求上，我们的方法比Sotas显著提高约20美元；最后，在挑战零查询黑盒设置的情况下，我们的方法获得了更好的性能，比Sotas高出20美元。我们的代码、模型和攻击请求将可用。



## **45. Exploring Adversarial Attacks on Neural Networks: An Explainable Approach**

探索对神经网络的敌意攻击：一种可解释的方法 cs.LG

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2303.06032v1) [paper-pdf](http://arxiv.org/pdf/2303.06032v1)

**Authors**: Justus Renkhoff, Wenkai Tan, Alvaro Velasquez, illiam Yichen Wang, Yongxin Liu, Jian Wang, Shuteng Niu, Lejla Begic Fazlic, Guido Dartmann, Houbing Song

**Abstract**: Deep Learning (DL) is being applied in various domains, especially in safety-critical applications such as autonomous driving. Consequently, it is of great significance to ensure the robustness of these methods and thus counteract uncertain behaviors caused by adversarial attacks. In this paper, we use gradient heatmaps to analyze the response characteristics of the VGG-16 model when the input images are mixed with adversarial noise and statistically similar Gaussian random noise. In particular, we compare the network response layer by layer to determine where errors occurred. Several interesting findings are derived. First, compared to Gaussian random noise, intentionally generated adversarial noise causes severe behavior deviation by distracting the area of concentration in the networks. Second, in many cases, adversarial examples only need to compromise a few intermediate blocks to mislead the final decision. Third, our experiments revealed that specific blocks are more vulnerable and easier to exploit by adversarial examples. Finally, we demonstrate that the layers $Block4\_conv1$ and $Block5\_cov1$ of the VGG-16 model are more susceptible to adversarial attacks. Our work could provide valuable insights into developing more reliable Deep Neural Network (DNN) models.

摘要: 深度学习正被应用于各个领域，特别是在自动驾驶等安全关键应用中。因此，保证这些方法的稳健性，从而对抗由对抗性攻击引起的不确定行为具有重要意义。本文利用梯度热图分析了输入图像混合对抗性噪声和统计相似高斯随机噪声时VGG-16模型的响应特性。特别是，我们逐层比较网络响应以确定错误发生的位置。由此得出了几个有趣的发现。首先，与高斯随机噪声相比，故意产生的对抗性噪声通过分散网络中的集中区而导致严重的行为偏差。其次，在许多情况下，对抗性的例子只需要妥协几个中间环节就可以误导最终决定。第三，我们的实验表明，特定的块更容易受到攻击，更容易被对手示例利用。最后，我们证明了VGG-16模型的第$Block4Cov1$层和第$Block5Cov1$层更容易受到敌意攻击。我们的工作可以为开发更可靠的深度神经网络(DNN)模型提供有价值的见解。



## **46. Sampling Attacks on Meta Reinforcement Learning: A Minimax Formulation and Complexity Analysis**

元强化学习中的抽样攻击：一种极小极大公式及其复杂性分析 cs.LG

updates: github repo posted

**SubmitDate**: 2023-03-08    [abs](http://arxiv.org/abs/2208.00081v2) [paper-pdf](http://arxiv.org/pdf/2208.00081v2)

**Authors**: Tao Li, Haozhe Lei, Quanyan Zhu

**Abstract**: Meta reinforcement learning (meta RL), as a combination of meta-learning ideas and reinforcement learning (RL), enables the agent to adapt to different tasks using a few samples. However, this sampling-based adaptation also makes meta RL vulnerable to adversarial attacks. By manipulating the reward feedback from sampling processes in meta RL, an attacker can mislead the agent into building wrong knowledge from training experience, which deteriorates the agent's performance when dealing with different tasks after adaptation. This paper provides a game-theoretical underpinning for understanding this type of security risk. In particular, we formally define the sampling attack model as a Stackelberg game between the attacker and the agent, which yields a minimax formulation. It leads to two online attack schemes: Intermittent Attack and Persistent Attack, which enable the attacker to learn an optimal sampling attack, defined by an $\epsilon$-first-order stationary point, within $\mathcal{O}(\epsilon^{-2})$ iterations. These attack schemes freeride the learning progress concurrently without extra interactions with the environment. By corroborating the convergence results with numerical experiments, we observe that a minor effort of the attacker can significantly deteriorate the learning performance, and the minimax approach can also help robustify the meta RL algorithms.

摘要: 元强化学习作为元学习思想和强化学习的结合，使智能体能够利用少量的样本来适应不同的任务。然而，这种基于采样的自适应也使得Meta RL容易受到对手攻击。通过操纵Meta RL中采样过程的奖励反馈，攻击者可以误导代理从训练经验中建立错误的知识，从而降低代理在适应后处理不同任务的性能。本文为理解这种类型的安全风险提供了博弈论基础。特别地，我们将抽样攻击模型正式定义为攻击者和代理之间的Stackelberg博弈，从而产生极小极大公式。它导致了两种在线攻击方案：间歇攻击和持续攻击，使攻击者能够在$\mathcal{O}(\epsilon^{-2})$迭代内学习由$\epsilon$-一阶固定点定义的最优抽样攻击。这些攻击方案同时加快了学习过程，而无需与环境进行额外的交互。通过数值实验证实了收敛结果，我们观察到攻击者的微小努力会显著降低学习性能，并且极小极大方法也有助于增强Meta RL算法的健壮性。



## **47. Robustness-preserving Lifelong Learning via Dataset Condensation**

基于数据集压缩的保持健壮性的终身学习 cs.LG

Accepted by ICASSP2023 Main Track: Machine Learning for Signal  Processing

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.04183v1) [paper-pdf](http://arxiv.org/pdf/2303.04183v1)

**Authors**: Jinghan Jia, Yihua Zhang, Dogyoon Song, Sijia Liu, Alfred Hero

**Abstract**: Lifelong learning (LL) aims to improve a predictive model as the data source evolves continuously. Most work in this learning paradigm has focused on resolving the problem of 'catastrophic forgetting,' which refers to a notorious dilemma between improving model accuracy over new data and retaining accuracy over previous data. Yet, it is also known that machine learning (ML) models can be vulnerable in the sense that tiny, adversarial input perturbations can deceive the models into producing erroneous predictions. This motivates the research objective of this paper - specification of a new LL framework that can salvage model robustness (against adversarial attacks) from catastrophic forgetting. Specifically, we propose a new memory-replay LL strategy that leverages modern bi-level optimization techniques to determine the 'coreset' of the current data (i.e., a small amount of data to be memorized) for ease of preserving adversarial robustness over time. We term the resulting LL framework 'Data-Efficient Robustness-Preserving LL' (DERPLL). The effectiveness of DERPLL is evaluated for class-incremental image classification using ResNet-18 over the CIFAR-10 dataset. Experimental results show that DERPLL outperforms the conventional coreset-guided LL baseline and achieves a substantial improvement in both standard accuracy and robust accuracy.

摘要: 终身学习的目标是随着数据源的不断发展而改进预测模型。这种学习范式的大部分工作都集中在解决“灾难性遗忘”的问题上，“灾难性遗忘”指的是在提高模型相对于新数据的准确性和保持相对于先前数据的准确性之间的两难境地。然而，众所周知，机器学习(ML)模型可能是脆弱的，因为微小的对抗性输入扰动可能会欺骗模型产生错误的预测。这促使了本文的研究目标-规范一个新的LL框架，该框架可以从灾难性遗忘中挽救模型的健壮性(对抗对手攻击)。具体地说，我们提出了一种新的记忆重放L1策略，该策略利用现代双层优化技术来确定当前数据(即需要记忆的少量数据)的核心重置，从而易于随着时间的推移保持对手的健壮性。我们将所得到的LL框架称为数据高效的保持健壮性的LL(DERPLL)。在CIFAR-10数据集上，使用ResNet-18对DERPLL算法进行了类增量图像分类的有效性评估。实验结果表明，DERPLL的性能优于传统的CoReset制导的LL基线，在标准精度和稳健精度方面都有很大的提高。



## **48. Exploiting Trust for Resilient Hypothesis Testing with Malicious Robots (evolved version)**

利用信任对恶意机器人进行弹性假设测试(进化版) cs.RO

21 pages, 5 figures, 1 table. arXiv admin note: substantial text  overlap with arXiv:2209.12285

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.04075v1) [paper-pdf](http://arxiv.org/pdf/2303.04075v1)

**Authors**: Matthew Cavorsi, Orhan Eren Akgün, Michal Yemini, Andrea Goldsmith, Stephanie Gil

**Abstract**: We develop a resilient binary hypothesis testing framework for decision making in adversarial multi-robot crowdsensing tasks. This framework exploits stochastic trust observations between robots to arrive at tractable, resilient decision making at a centralized Fusion Center (FC) even when i) there exist malicious robots in the network and their number may be larger than the number of legitimate robots, and ii) the FC uses one-shot noisy measurements from all robots. We derive two algorithms to achieve this. The first is the Two Stage Approach (2SA) that estimates the legitimacy of robots based on received trust observations, and provably minimizes the probability of detection error in the worst-case malicious attack. Here, the proportion of malicious robots is known but arbitrary. For the case of an unknown proportion of malicious robots, we develop the Adversarial Generalized Likelihood Ratio Test (A-GLRT) that uses both the reported robot measurements and trust observations to estimate the trustworthiness of robots, their reporting strategy, and the correct hypothesis simultaneously. We exploit special problem structure to show that this approach remains computationally tractable despite several unknown problem parameters. We deploy both algorithms in a hardware experiment where a group of robots conducts crowdsensing of traffic conditions on a mock-up road network similar in spirit to Google Maps, subject to a Sybil attack. We extract the trust observations for each robot from actual communication signals which provide statistical information on the uniqueness of the sender. We show that even when the malicious robots are in the majority, the FC can reduce the probability of detection error to 30.5% and 29% for the 2SA and the A-GLRT respectively.

摘要: 提出了一种用于对抗性多机器人群体感知任务决策的弹性二元假设检验框架。该框架利用机器人之间的随机信任观察，即使在i)网络中存在恶意机器人并且它们的数量可能大于合法机器人的数量，以及ii)FC使用来自所有机器人的一次噪声测量的情况下，也可以在集中式融合中心(FC)获得易于处理的、有弹性的决策。我们推导了两个算法来实现这一点。第一种是两阶段方法(2SA)，它根据接收到的信任观察来估计机器人的合法性，并证明在最坏情况下恶意攻击的检测错误概率最小。在这里，恶意机器人的比例是已知的，但是随意的。对于恶意机器人比例未知的情况，我们提出了对抗性广义似然比检验(A-GLRT)，它同时使用报告的机器人测量值和信任观察来估计机器人的可信性、报告策略和正确的假设。我们利用特殊的问题结构表明，尽管有几个未知的问题参数，该方法在计算上仍然是容易处理的。我们在硬件实验中部署了这两种算法，在硬件实验中，一组机器人在一个类似于谷歌地图的模拟道路网络上对交通状况进行众感，受到Sybil攻击。我们从提供关于发送者唯一性的统计信息的实际通信信号中提取每个机器人的信任观察。实验结果表明，即使恶意机器人占多数，FC算法也能将2SA和A-GLRT的误检率分别降低到30.5%和29%。



## **49. Bounding Information Leakage in Machine Learning**

机器学习中的有界信息泄漏 cs.LG

Published in [Elsevier  Neurocomputing](https://doi.org/10.1016/j.neucom.2023.02.058)

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2105.03875v2) [paper-pdf](http://arxiv.org/pdf/2105.03875v2)

**Authors**: Ganesh Del Grosso, Georg Pichler, Catuscia Palamidessi, Pablo Piantanida

**Abstract**: Recently, it has been shown that Machine Learning models can leak sensitive information about their training data. This information leakage is exposed through membership and attribute inference attacks. Although many attack strategies have been proposed, little effort has been made to formalize these problems. We present a novel formalism, generalizing membership and attribute inference attack setups previously studied in the literature and connecting them to memorization and generalization. First, we derive a universal bound on the success rate of inference attacks and connect it to the generalization gap of the target model. Second, we study the question of how much sensitive information is stored by the algorithm about its training set and we derive bounds on the mutual information between the sensitive attributes and model parameters. Experimentally, we illustrate the potential of our approach by applying it to both synthetic data and classification tasks on natural images. Finally, we apply our formalism to different attribute inference strategies, with which an adversary is able to recover the identity of writers in the PenDigits dataset.

摘要: 最近，有研究表明，机器学习模型会泄露有关其训练数据的敏感信息。这种信息泄露通过成员身份和属性推理攻击暴露出来。虽然已经提出了许多攻击策略，但几乎没有努力将这些问题形式化。我们提出了一种新的形式主义，推广了以前在文献中研究的成员资格和属性推理攻击设置，并将它们与记忆和泛化联系起来。首先，我们推导出推理攻击成功率的一个普适界，并将其与目标模型的泛化差距联系起来。其次，研究了该算法对其训练集存储了多少敏感信息的问题，得到了敏感属性与模型参数之间互信息量的界。在实验上，我们通过将该方法应用于合成数据和自然图像上的分类任务来说明该方法的潜力。最后，我们将我们的形式化应用于不同的属性推理策略，通过这些策略，对手能够恢复PenDigits数据集中作者的身份。



## **50. SCRAMBLE-CFI: Mitigating Fault-Induced Control-Flow Attacks on OpenTitan**

SCRIBLE-CFI：缓解OpenTitan上的错误引起的控制流攻击 cs.CR

**SubmitDate**: 2023-03-07    [abs](http://arxiv.org/abs/2303.03711v1) [paper-pdf](http://arxiv.org/pdf/2303.03711v1)

**Authors**: Pascal Nasahl, Stefan Mangard

**Abstract**: Secure elements physically exposed to adversaries are frequently targeted by fault attacks. These attacks can be utilized to hijack the control-flow of software allowing the attacker to bypass security measures, extract sensitive data, or gain full code execution. In this paper, we systematically analyze the threat vector of fault-induced control-flow manipulations on the open-source OpenTitan secure element. Our thorough analysis reveals that current countermeasures of this chip either induce large area overheads or still cannot prevent the attacker from exploiting the identified threats. In this context, we introduce SCRAMBLE-CFI, an encryption-based control-flow integrity scheme utilizing existing hardware features of OpenTitan. SCRAMBLE-CFI confines, with minimal hardware overhead, the impact of fault-induced control-flow attacks by encrypting each function with a different encryption tweak at load-time. At runtime, code only can be successfully decrypted when the correct decryption tweak is active. We open-source our hardware changes and release our LLVM toolchain automatically protecting programs. Our analysis shows that SCRAMBLE-CFI complementarily enhances security guarantees of OpenTitan with a negligible hardware overhead of less than 3.97 % and a runtime overhead of 7.02 % for the Embench-IoT benchmarks.

摘要: 物理上暴露在对手面前的安全元素经常成为故障攻击的目标。这些攻击可用于劫持软件的控制流，从而允许攻击者绕过安全措施、提取敏感数据或获得完整的代码执行。在本文中，我们系统地分析了开源OpenTitan安全元素上由错误引起的控制流操作的威胁向量。我们的深入分析表明，目前该芯片的应对措施要么导致大面积开销，要么仍然无法阻止攻击者利用已识别的威胁。在此背景下，我们介绍了一种基于加密的控制流完整性方案SCRIBLE-CFI，该方案利用了OpenTitan现有的硬件特性。置乱-CFI通过在加载时使用不同的加密调整对每个函数进行加密，以最小的硬件开销限制了故障引发的控制流攻击的影响。在运行时，只有当正确的解密调整处于活动状态时，才能成功解密代码。我们将我们的硬件更改开源，并发布我们的LLVM工具链自动保护程序。我们的分析表明，在硬件开销小于3.97%、运行时开销为7.02%的情况下，SCRIBLE-CFI互补地增强了OpenTitan的安全保证。



