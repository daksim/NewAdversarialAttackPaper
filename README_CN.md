# Latest Adversarial Attack Papers
**update at 2024-07-02 15:09:58**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On the Abuse and Detection of Polyglot Files**

多语言文件的滥用与检测 cs.CR

18 pages, 11 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01529v1) [paper-pdf](http://arxiv.org/pdf/2407.01529v1)

**Authors**: Luke Koch, Sean Oesch, Amul Chaulagain, Jared Dixon, Matthew Dixon, Mike Huettal, Amir Sadovnik, Cory Watson, Brian Weber, Jacob Hartman, Richard Patulski

**Abstract**: A polyglot is a file that is valid in two or more formats. Polyglot files pose a problem for malware detection systems that route files to format-specific detectors/signatures, as well as file upload and sanitization tools. In this work we found that existing file-format and embedded-file detection tools, even those developed specifically for polyglot files, fail to reliably detect polyglot files used in the wild, leaving organizations vulnerable to attack. To address this issue, we studied the use of polyglot files by malicious actors in the wild, finding $30$ polyglot samples and $15$ attack chains that leveraged polyglot files. In this report, we highlight two well-known APTs whose cyber attack chains relied on polyglot files to bypass detection mechanisms. Using knowledge from our survey of polyglot usage in the wild -- the first of its kind -- we created a novel data set based on adversary techniques. We then trained a machine learning detection solution, PolyConv, using this data set. PolyConv achieves a precision-recall area-under-curve score of $0.999$ with an F1 score of $99.20$% for polyglot detection and $99.47$% for file-format identification, significantly outperforming all other tools tested. We developed a content disarmament and reconstruction tool, ImSan, that successfully sanitized $100$% of the tested image-based polyglots, which were the most common type found via the survey. Our work provides concrete tools and suggestions to enable defenders to better defend themselves against polyglot files, as well as directions for future work to create more robust file specifications and methods of disarmament.

摘要: 多语种是指以两种或多种格式有效的文件。多语言文件给恶意软件检测系统带来了问题，恶意软件检测系统将文件路由到特定格式的检测器/签名，以及文件上传和清理工具。在这项工作中，我们发现现有的文件格式和嵌入式文件检测工具，即使是那些专门为多语言文件开发的工具，也无法可靠地检测在野外使用的多语言文件，从而使组织容易受到攻击。为了解决这个问题，我们研究了恶意攻击者在野外使用多语言文件的情况，发现了$30$多语言样本和$15$利用多语言文件的攻击链。在这份报告中，我们重点介绍了两个著名的APT，它们的网络攻击链依赖于多语言文件来绕过检测机制。利用我们对野外多语种使用情况的调查--这是此类调查中的第一次--我们创建了一个基于对手技术的新数据集。然后，我们使用这个数据集训练了一个机器学习检测解决方案PolyConv。PolyConv获得了0.999美元的曲线下区域精度召回分数，而F1多语言检测的分数为99.20$%，文件格式识别的F1分数为99.47$%，大大超过了所有其他测试工具。我们开发了一个内容解除和重建工具ImSAN，它成功地清理了100美元%的基于图像的测试多语种，这是通过调查发现的最常见的类型。我们的工作提供了具体的工具和建议，使捍卫者能够更好地保护自己免受多国语言文件的伤害，并为今后制定更可靠的文件规格和裁军方法指明了方向。



## **2. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01461v1) [paper-pdf](http://arxiv.org/pdf/2407.01461v1)

**Authors**: Zisu Huang, Xiaohua Wang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型(LLM)生成诚实、无害和有用的响应的能力在很大程度上取决于用户提示的质量。然而，这些提示往往简短而含糊，从而极大地限制了LLM的全部潜力。此外，有害的提示可以被对手精心制作和操纵，以越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLMS的能力，同时保持对有害越狱输入的强大健壮性，本研究提出了一个可移植和可插拔的框架，在将用户提示输入到LLMS之前对其进行提炼。这一策略提高了查询的质量，使LLMS能够生成更真实、良性和有用的响应。具体地说，引入了一种轻量级查询精化模型，并使用专门设计的强化学习方法进行训练，该方法结合了多个目标来增强LLMS的特定能力。大量实验表明，改进模型不仅提高了响应的质量，而且增强了对越狱攻击的健壮性。代码可从以下网址获得：https://github.com/Huangzisu/query-refinement。



## **3. Cutting through buggy adversarial example defenses: fixing 1 line of code breaks Sabre**

突破错误的对抗性示例防御：修复1行代码破解Sabre cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2405.03672v3) [paper-pdf](http://arxiv.org/pdf/2405.03672v3)

**Authors**: Nicholas Carlini

**Abstract**: Sabre is a defense to adversarial examples that was accepted at IEEE S&P 2024. We first reveal significant flaws in the evaluation that point to clear signs of gradient masking. We then show the cause of this gradient masking: a bug in the original evaluation code. By fixing a single line of code in the original repository, we reduce Sabre's robust accuracy to 0%. In response to this, the authors modify the defense and introduce a new defense component not described in the original paper. But this fix contains a second bug; modifying one more line of code reduces robust accuracy to below baseline levels. After we released the first version of our paper online, the authors introduced another change to the defense; by commenting out one line of code during attack we reduce the robust accuracy to 0% again.

摘要: Sabre是对IEEE S & P 2024上接受的敌对例子的辩护。我们首先揭示了评估中的重大缺陷，这些缺陷表明了梯度掩蔽的明显迹象。然后我们展示这种梯度掩蔽的原因：原始评估代码中的一个错误。通过在原始存储库中修复一行代码，我们将Sabre的稳健准确性降低到0%。作为回应，作者修改了辩护并引入了原始论文中未描述的新辩护组件。但此修复包含第二个错误;修改多一行代码会将稳健准确性降低到基线水平以下。在我们在线发布论文的第一版后，作者对防御进行了另一项更改;通过在攻击期间注释掉一行代码，我们将稳健准确性再次降低到0%。



## **4. Jailbreak Vision Language Models via Bi-Modal Adversarial Prompt**

通过双模式对抗提示的越狱视觉语言模型 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.04031v2) [paper-pdf](http://arxiv.org/pdf/2406.04031v2)

**Authors**: Zonghao Ying, Aishan Liu, Tianyuan Zhang, Zhengmin Yu, Siyuan Liang, Xianglong Liu, Dacheng Tao

**Abstract**: In the realm of large vision language models (LVLMs), jailbreak attacks serve as a red-teaming approach to bypass guardrails and uncover safety implications. Existing jailbreaks predominantly focus on the visual modality, perturbing solely visual inputs in the prompt for attacks. However, they fall short when confronted with aligned models that fuse visual and textual features simultaneously for generation. To address this limitation, this paper introduces the Bi-Modal Adversarial Prompt Attack (BAP), which executes jailbreaks by optimizing textual and visual prompts cohesively. Initially, we adversarially embed universally harmful perturbations in an image, guided by a few-shot query-agnostic corpus (e.g., affirmative prefixes and negative inhibitions). This process ensures that image prompt LVLMs to respond positively to any harmful queries. Subsequently, leveraging the adversarial image, we optimize textual prompts with specific harmful intent. In particular, we utilize a large language model to analyze jailbreak failures and employ chain-of-thought reasoning to refine textual prompts through a feedback-iteration manner. To validate the efficacy of our approach, we conducted extensive evaluations on various datasets and LVLMs, demonstrating that our method significantly outperforms other methods by large margins (+29.03% in attack success rate on average). Additionally, we showcase the potential of our attacks on black-box commercial LVLMs, such as Gemini and ChatGLM.

摘要: 在大型视觉语言模型(LVLM)领域，越狱攻击是一种绕过护栏并发现安全隐患的红队方法。现有的越狱主要集中在视觉形式上，只干扰攻击提示中的视觉输入。然而，当面对同时融合视觉和文本特征以生成的对齐模型时，它们不能满足要求。为了解决这一局限性，本文引入了双模式对抗性提示攻击(BAP)，它通过结合优化文本和视觉提示来执行越狱。最初，我们不利地在图像中嵌入普遍有害的扰动，由几个与查询无关的语料库(例如，肯定前缀和否定抑制)引导。此过程确保图像提示LVLMS对任何有害查询做出积极响应。随后，利用敌意图像，我们优化了具有特定有害意图的文本提示。特别是，我们利用一个大的语言模型来分析越狱失败，并使用思想链推理来通过反馈迭代的方式来提炼文本提示。为了验证我们方法的有效性，我们在不同的数据集和LVLM上进行了广泛的评估，结果表明我们的方法在很大程度上优于其他方法(攻击成功率平均为+29.03%)。此外，我们还展示了我们对黑盒商业LVLM的攻击潜力，如Gemini和ChatGLM。



## **5. Formal Verification of Object Detection**

对象检测的形式化验证 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01295v1) [paper-pdf](http://arxiv.org/pdf/2407.01295v1)

**Authors**: Avraham Raviv, Yizhak Y. Elboher, Michelle Aluf-Medina, Yael Leibovich Weiss, Omer Cohen, Roy Assa, Guy Katz, Hillel Kugler

**Abstract**: Deep Neural Networks (DNNs) are ubiquitous in real-world applications, yet they remain vulnerable to errors and adversarial attacks. This work tackles the challenge of applying formal verification to ensure the safety of computer vision models, extending verification beyond image classification to object detection. We propose a general formulation for certifying the robustness of object detection models using formal verification and outline implementation strategies compatible with state-of-the-art verification tools. Our approach enables the application of these tools, originally designed for verifying classification models, to object detection. We define various attacks for object detection, illustrating the diverse ways adversarial inputs can compromise neural network outputs. Our experiments, conducted on several common datasets and networks, reveal potential errors in object detection models, highlighting system vulnerabilities and emphasizing the need for expanding formal verification to these new domains. This work paves the way for further research in integrating formal verification across a broader range of computer vision applications.

摘要: 深度神经网络(DNN)在实际应用中无处不在，但它们仍然容易受到错误和对手攻击。这项工作解决了应用形式化验证来确保计算机视觉模型的安全性的挑战，将验证从图像分类扩展到目标检测。我们提出了使用形式化验证来证明目标检测模型的健壮性的一般公式，并概述了与最先进的验证工具兼容的实现策略。我们的方法使得这些最初设计用于验证分类模型的工具能够应用于目标检测。我们定义了用于目标检测的各种攻击，说明了敌意输入可以损害神经网络输出的不同方式。我们在几个常见的数据集和网络上进行的实验，揭示了对象检测模型中的潜在错误，突出了系统漏洞，并强调了将正式验证扩展到这些新领域的必要性。这项工作为在更广泛的计算机视觉应用中整合形式验证的进一步研究铺平了道路。



## **6. DeepiSign-G: Generic Watermark to Stamp Hidden DNN Parameters for Self-contained Tracking**

DeepiSign-G：通用水印，用于标记隐藏的DNN参数以进行自包含跟踪 cs.CR

13 pages

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01260v1) [paper-pdf](http://arxiv.org/pdf/2407.01260v1)

**Authors**: Alsharif Abuadbba, Nicholas Rhodes, Kristen Moore, Bushra Sabir, Shuo Wang, Yansong Gao

**Abstract**: Deep learning solutions in critical domains like autonomous vehicles, facial recognition, and sentiment analysis require caution due to the severe consequences of errors. Research shows these models are vulnerable to adversarial attacks, such as data poisoning and neural trojaning, which can covertly manipulate model behavior, compromising reliability and safety. Current defense strategies like watermarking have limitations: they fail to detect all model modifications and primarily focus on attacks on CNNs in the image domain, neglecting other critical architectures like RNNs.   To address these gaps, we introduce DeepiSign-G, a versatile watermarking approach designed for comprehensive verification of leading DNN architectures, including CNNs and RNNs. DeepiSign-G enhances model security by embedding an invisible watermark within the Walsh-Hadamard transform coefficients of the model's parameters. This watermark is highly sensitive and fragile, ensuring prompt detection of any modifications. Unlike traditional hashing techniques, DeepiSign-G allows substantial metadata incorporation directly within the model, enabling detailed, self-contained tracking and verification.   We demonstrate DeepiSign-G's applicability across various architectures, including CNN models (VGG, ResNets, DenseNet) and RNNs (Text sentiment classifier). We experiment with four popular datasets: VGG Face, CIFAR10, GTSRB Traffic Sign, and Large Movie Review. We also evaluate DeepiSign-G under five potential attacks. Our comprehensive evaluation confirms that DeepiSign-G effectively detects these attacks without compromising CNN and RNN model performance, highlighting its efficacy as a robust security measure for deep learning applications. Detection of integrity breaches is nearly perfect, while hiding only a bit in approximately 1% of the Walsh-Hadamard coefficients.

摘要: 由于错误的严重后果，自动驾驶汽车、面部识别和情绪分析等关键领域的深度学习解决方案需要谨慎。研究表明，这些模型容易受到数据中毒和神经木马等敌意攻击，这些攻击可能会秘密操纵模型行为，损害可靠性和安全性。当前的防御策略，如水印，都有局限性：它们无法检测到所有模型的修改，主要集中在对图像域中的CNN的攻击，而忽略了其他关键的体系结构，如RNN。为了弥补这些差距，我们引入了DeepSign-G，这是一种多功能水印方法，旨在全面验证领先的DNN架构，包括CNN和RNN。DeepSign-G通过在模型参数的Walsh-Hadamard变换系数中嵌入一个不可见的水印来增强模型的安全性。该水印高度敏感和脆弱，确保了对任何修改的及时检测。与传统的散列技术不同，DeepSign-G允许将大量元数据直接合并到模型中，从而实现详细、独立的跟踪和验证。我们演示了DeepSign-G在各种体系结构上的适用性，包括CNN模型(VGG、ResNet、DenseNet)和RNNS(文本情感分类器)。我们使用了四个流行的数据集：VGG Face、CIFAR10、GTSRB交通标志和大电影评论。我们还评估了DeepSign-G在五种潜在攻击下的性能。我们的全面评估证实，DeepSign-G在不影响CNN和RNN模型性能的情况下有效地检测到这些攻击，突出了其作为深度学习应用程序的强大安全措施的有效性。完整性破坏的检测近乎完美，同时只隐藏了大约1%的沃尔什-哈达玛系数中的一位。



## **7. QUEEN: Query Unlearning against Model Extraction**

QUEEN：针对模型提取的查询取消学习 cs.CR

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01251v1) [paper-pdf](http://arxiv.org/pdf/2407.01251v1)

**Authors**: Huajie Chen, Tianqing Zhu, Lefeng Zhang, Bo Liu, Derui Wang, Wanlei Zhou, Minhui Xue

**Abstract**: Model extraction attacks currently pose a non-negligible threat to the security and privacy of deep learning models. By querying the model with a small dataset and usingthe query results as the ground-truth labels, an adversary can steal a piracy model with performance comparable to the original model. Two key issues that cause the threat are, on the one hand, accurate and unlimited queries can be obtained by the adversary; on the other hand, the adversary can aggregate the query results to train the model step by step. The existing defenses usually employ model watermarking or fingerprinting to protect the ownership. However, these methods cannot proactively prevent the violation from happening. To mitigate the threat, we propose QUEEN (QUEry unlEarNing) that proactively launches counterattacks on potential model extraction attacks from the very beginning. To limit the potential threat, QUEEN has sensitivity measurement and outputs perturbation that prevents the adversary from training a piracy model with high performance. In sensitivity measurement, QUEEN measures the single query sensitivity by its distance from the center of its cluster in the feature space. To reduce the learning accuracy of attacks, for the highly sensitive query batch, QUEEN applies query unlearning, which is implemented by gradient reverse to perturb the softmax output such that the piracy model will generate reverse gradients to worsen its performance unconsciously. Experiments show that QUEEN outperforms the state-of-the-art defenses against various model extraction attacks with a relatively low cost to the model accuracy. The artifact is publicly available at https://anonymous.4open.science/r/queen implementation-5408/.

摘要: 模型提取攻击目前对深度学习模型的安全和隐私构成了不可忽视的威胁。通过使用较小的数据集对模型进行查询，并将查询结果作为地面事实标签，攻击者可以窃取性能与原始模型相当的盗版模型。造成威胁的两个关键问题是，一方面，对手可以获得准确和无限的查询；另一方面，对手可以聚合查询结果，逐步训练模型。现有的防御工事通常采用模型水印或指纹来保护所有权。然而，这些方法不能主动阻止违规行为的发生。为了缓解这种威胁，我们提出了Queue(查询遗忘)算法，它从一开始就主动地对潜在的模型提取攻击发起反击。为了限制潜在威胁，皇后拥有敏感度测量和输出扰动，以防止对手训练高性能的盗版模型。在敏感度度量中，Queue通过距离特征空间中聚类中心的距离来衡量单个查询的敏感度。为了降低攻击的学习精度，对于高度敏感的查询批次，Queue应用查询遗忘，通过梯度反转来实现对Softmax输出的扰动，使得盗版模型会产生反向梯度，从而在不知不觉中恶化其性能。实验表明，Queue在抵抗各种模型提取攻击时的性能优于最先进的防御系统，而模型精确度的代价相对较低。该构件可在https://anonymous.4open.science/r/queen Implementation-5408/上公开获得。



## **8. Multi-View Black-Box Physical Attacks on Infrared Pedestrian Detectors Using Adversarial Infrared Grid**

使用对抗红外网格对红外行人探测器进行多视图黑匣子物理攻击 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01168v1) [paper-pdf](http://arxiv.org/pdf/2407.01168v1)

**Authors**: Kalibinuer Tiliwalidi, Chengyin Hu, Weiwen Shi

**Abstract**: While extensive research exists on physical adversarial attacks within the visible spectrum, studies on such techniques in the infrared spectrum are limited. Infrared object detectors are vital in modern technological applications but are susceptible to adversarial attacks, posing significant security threats. Previous studies using physical perturbations like light bulb arrays and aerogels for white-box attacks, or hot and cold patches for black-box attacks, have proven impractical or limited in multi-view support. To address these issues, we propose the Adversarial Infrared Grid (AdvGrid), which models perturbations in a grid format and uses a genetic algorithm for black-box optimization. These perturbations are cyclically applied to various parts of a pedestrian's clothing to facilitate multi-view black-box physical attacks on infrared pedestrian detectors. Extensive experiments validate AdvGrid's effectiveness, stealthiness, and robustness. The method achieves attack success rates of 80.00\% in digital environments and 91.86\% in physical environments, outperforming baseline methods. Additionally, the average attack success rate exceeds 50\% against mainstream detectors, demonstrating AdvGrid's robustness. Our analyses include ablation studies, transfer attacks, and adversarial defenses, confirming the method's superiority.

摘要: 虽然在可见光光谱内对物理对抗攻击已有广泛的研究，但在红外光谱中对这类技术的研究有限。红外目标探测器在现代技术应用中至关重要，但容易受到对抗性攻击，构成重大安全威胁。以前的研究证明，使用物理扰动，如灯泡阵列和气凝胶进行白盒攻击，或使用冷热补丁进行黑盒攻击，都被证明是不切实际的，或者在多视角支持方面受到限制。为了解决这些问题，我们提出了对抗性红外网格(AdvGrid)，它以网格的形式对扰动进行建模，并使用遗传算法进行黑盒优化。这些扰动被循环应用于行人衣服的不同部分，以促进对红外行人探测器的多视角黑匣子物理攻击。大量实验验证了AdvGrid的有效性、隐蔽性和健壮性。该方法在数字环境下的攻击成功率为80.00%，在物理环境下的攻击成功率为91.86%，优于基准攻击方法。此外，对主流检测器的平均攻击成功率超过50%，显示了AdvGrid的健壮性。我们的分析包括烧蚀研究、转移攻击和对抗性防御，证实了该方法的优越性。



## **9. Unaligning Everything: Or Aligning Any Text to Any Image in Multimodal Models**

将所有内容分开：或将任何文本与多模式模型中的任何图像对齐 cs.CV

14 pages, 14 figures

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01157v1) [paper-pdf](http://arxiv.org/pdf/2407.01157v1)

**Authors**: Shaeke Salman, Md Montasir Bin Shams, Xiuwen Liu

**Abstract**: Utilizing a shared embedding space, emerging multimodal models exhibit unprecedented zero-shot capabilities. However, the shared embedding space could lead to new vulnerabilities if different modalities can be misaligned. In this paper, we extend and utilize a recently developed effective gradient-based procedure that allows us to match the embedding of a given text by minimally modifying an image. Using the procedure, we show that we can align the embeddings of distinguishable texts to any image through unnoticeable adversarial attacks in joint image-text models, revealing that semantically unrelated images can have embeddings of identical texts and at the same time visually indistinguishable images can be matched to the embeddings of very different texts. Our technique achieves 100\% success rate when it is applied to text datasets and images from multiple sources. Without overcoming the vulnerability, multimodal models cannot robustly align inputs from different modalities in a semantically meaningful way. \textbf{Warning: the text data used in this paper are toxic in nature and may be offensive to some readers.}

摘要: 利用共享的嵌入空间，新兴的多模式显示出前所未有的零射击能力。然而，如果不同的模式可能会错位，共享嵌入空间可能会导致新的漏洞。在本文中，我们扩展和利用了最近开发的一种有效的基于梯度的方法，该方法允许我们通过对图像进行最小限度的修改来匹配给定文本的嵌入。利用该过程，我们证明了在联合图文模型中，通过不可察觉的对抗性攻击，可以将可区分文本的嵌入与任何图像对齐，从而揭示了语义无关的图像可以具有相同文本的嵌入，同时视觉上不可区分的图像可以与非常不同的文本的嵌入相匹配。将该方法应用于多个来源的文本数据集和图像，取得了100%的准确率。如果不克服这一弱点，多通道模型就不能以语义有意义的方式稳健地对齐来自不同通道的输入。\textbf{警告：本文中使用的文本数据具有毒性，可能会冒犯某些读者。}



## **10. SecGenAI: Enhancing Security of Cloud-based Generative AI Applications within Australian Critical Technologies of National Interest**

SecGenAI：增强澳大利亚国家利益关键技术中基于云的生成性人工智能应用的安全性 cs.CR

10 pages, 4 figures, 9 tables, submitted to the 2024 11th  International Conference on Soft Computing & Machine Intelligence (ISCMI  2024)

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.01110v1) [paper-pdf](http://arxiv.org/pdf/2407.01110v1)

**Authors**: Christoforus Yoga Haryanto, Minh Hieu Vu, Trung Duc Nguyen, Emily Lomempow, Yulia Nurliana, Sona Taheri

**Abstract**: The rapid advancement of Generative AI (GenAI) technologies offers transformative opportunities within Australia's critical technologies of national interest while introducing unique security challenges. This paper presents SecGenAI, a comprehensive security framework for cloud-based GenAI applications, with a focus on Retrieval-Augmented Generation (RAG) systems. SecGenAI addresses functional, infrastructure, and governance requirements, integrating end-to-end security analysis to generate specifications emphasizing data privacy, secure deployment, and shared responsibility models. Aligned with Australian Privacy Principles, AI Ethics Principles, and guidelines from the Australian Cyber Security Centre and Digital Transformation Agency, SecGenAI mitigates threats such as data leakage, adversarial attacks, and model inversion. The framework's novel approach combines advanced machine learning techniques with robust security measures, ensuring compliance with Australian regulations while enhancing the reliability and trustworthiness of GenAI systems. This research contributes to the field of intelligent systems by providing actionable strategies for secure GenAI implementation in industry, fostering innovation in AI applications, and safeguarding national interests.

摘要: 产生式人工智能(GenAI)技术的快速发展为澳大利亚涉及国家利益的关键技术提供了变革性的机会，同时带来了独特的安全挑战。提出了一种基于云的GenAI应用安全框架SecGenAI，重点研究了检索-增强生成(RAG)系统。SecGenAI解决了功能、基础设施和治理需求，集成了端到端安全分析，以生成强调数据隐私、安全部署和分担责任模型的规范。SecGenAI与澳大利亚隐私原则、人工智能道德原则以及澳大利亚网络安全中心和数字转型机构的指导方针保持一致，可以缓解数据泄露、对抗性攻击和模型反转等威胁。该框架的新方法将先进的机器学习技术与强大的安全措施相结合，在确保符合澳大利亚法规的同时，增强了GenAI系统的可靠性和可信度。这项研究为工业中安全实施GenAI提供了可行的策略，促进了人工智能应用的创新，并维护了国家利益，从而为智能系统领域做出了贡献。



## **11. DifAttack++: Query-Efficient Black-Box Adversarial Attack via Hierarchical Disentangled Feature Space in Cross-Domain**

DifAttack++：跨域中通过分层解纠缠特征空间进行查询高效黑匣子对抗攻击 cs.CV

arXiv admin note: substantial text overlap with arXiv:2309.14585 An  extension of the AAAI24 paper "DifAttack: Query-Efficient Black-Box Attack  via Disentangled Feature Space."

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.03017v3) [paper-pdf](http://arxiv.org/pdf/2406.03017v3)

**Authors**: Jun Liu, Jiantao Zhou, Jiandian Zeng, Jinyu Tian, Zheng Li

**Abstract**: This work investigates efficient score-based black-box adversarial attacks with a high Attack Success Rate (\textbf{ASR}) and good generalizability. We design a novel attack method based on a hierarchical DIsentangled Feature space, called \textbf{DifAttack++}, which differs significantly from the existing ones operating over the entire feature space. Specifically, DifAttack++ firstly disentangles an image's latent feature into an Adversarial Feature (\textbf{AF}) and a Visual Feature (\textbf{VF}) via an autoencoder equipped with our specially designed Hierarchical Decouple-Fusion (\textbf{HDF}) module, where the AF dominates the adversarial capability of an image, while the VF largely determines its visual appearance. We train such two autoencoders for the clean and adversarial image domains (i.e., cross-domain) respectively to achieve image reconstructions and feature disentanglement, by using pairs of clean images and their Adversarial Examples (\textbf{AE}s) generated from available surrogate models via white-box attack methods. Eventually, in the black-box attack stage, DifAttack++ iteratively optimizes the AF according to the query feedback from the victim model until a successful AE is generated, while keeping the VF unaltered. Extensive experimental results demonstrate that our DifAttack++ leads to superior ASR and query efficiency than state-of-the-art methods, meanwhile exhibiting much better visual quality of AEs. The code is available at https://github.com/csjunjun/DifAttack.git.

摘要: 研究了基于分数的高效黑盒对抗攻击，具有较高的攻击成功率(Textbf{asr})和良好的泛化能力。我们设计了一种新的基于分层解缠特征空间的攻击方法，称为Textbf{DifAttack++}，它与现有的操作在整个特征空间上的攻击方法有很大的不同。具体地说，DifAttack++首先通过配备了我们特别设计的分层去耦合融合(\extbf{hdf})模块的自动编码器，将图像的潜在特征分解为对抗特征(\extbf{AF})和视觉特征(\extbf{Vf})，其中，AF主导图像的对抗能力，而VF在很大程度上决定其视觉外观。通过白盒攻击方法，利用已有代理模型生成的干净图像对和对抗性图像对(S)，分别对干净图像和对抗性图像域(即跨域)进行训练，以实现图像重建和特征解缠。最终，在黑盒攻击阶段，DifAttack++根据受害者模型的查询反馈迭代地优化AF，直到生成成功的AE，同时保持VF不变。大量的实验结果表明，我们的DifAttack++比现有的方法具有更高的ASR和查询效率，同时表现出更好的AEs视觉质量。代码可在https://github.com/csjunjun/DifAttack.git.上获得



## **12. Time-Frequency Jointed Imperceptible Adversarial Attack to Brainprint Recognition with Deep Learning Models**

深度学习模型对脑纹识别的时频联合不可感知的对抗攻击 cs.CR

This work is accepted by ICME 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2403.10021v3) [paper-pdf](http://arxiv.org/pdf/2403.10021v3)

**Authors**: Hangjie Yi, Yuhang Ming, Dongjun Liu, Wanzeng Kong

**Abstract**: EEG-based brainprint recognition with deep learning models has garnered much attention in biometric identification. Yet, studies have indicated vulnerability to adversarial attacks in deep learning models with EEG inputs. In this paper, we introduce a novel adversarial attack method that jointly attacks time-domain and frequency-domain EEG signals by employing wavelet transform. Different from most existing methods which only target time-domain EEG signals, our method not only takes advantage of the time-domain attack's potent adversarial strength but also benefits from the imperceptibility inherent in frequency-domain attack, achieving a better balance between attack performance and imperceptibility. Extensive experiments are conducted in both white- and grey-box scenarios and the results demonstrate that our attack method achieves state-of-the-art attack performance on three datasets and three deep-learning models. In the meanwhile, the perturbations in the signals attacked by our method are barely perceptible to the human visual system.

摘要: 基于深度学习模型的脑电脑纹识别在生物特征识别中得到了广泛的关注。然而，研究表明，在有脑电输入的深度学习模型中，容易受到对抗性攻击。本文提出了一种利用小波变换联合攻击时频域脑电信号的对抗性攻击方法。不同于现有的大多数只针对时域脑电信号的方法，我们的方法不仅利用了时域攻击的强大对抗能力，而且得益于频域攻击固有的不可感知性，在攻击性能和不可感知性之间取得了更好的平衡。在白盒和灰盒场景下进行了大量的实验，结果表明，我们的攻击方法在三个数据集和三个深度学习模型上获得了最先进的攻击性能。同时，我们的方法攻击的信号中的扰动几乎不被人类视觉系统察觉到。



## **13. Learning Robust 3D Representation from CLIP via Dual Denoising**

通过双重去噪从CLIP学习稳健的3D表示 cs.CV

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2407.00905v1) [paper-pdf](http://arxiv.org/pdf/2407.00905v1)

**Authors**: Shuqing Luo, Bowen Qu, Wei Gao

**Abstract**: In this paper, we explore a critical yet under-investigated issue: how to learn robust and well-generalized 3D representation from pre-trained vision language models such as CLIP. Previous works have demonstrated that cross-modal distillation can provide rich and useful knowledge for 3D data. However, like most deep learning models, the resultant 3D learning network is still vulnerable to adversarial attacks especially the iterative attack. In this work, we propose Dual Denoising, a novel framework for learning robust and well-generalized 3D representations from CLIP. It combines a denoising-based proxy task with a novel feature denoising network for 3D pre-training. Additionally, we propose utilizing parallel noise inference to enhance the generalization of point cloud features under cross domain settings. Experiments show that our model can effectively improve the representation learning performance and adversarial robustness of the 3D learning network under zero-shot settings without adversarial training. Our code is available at https://github.com/luoshuqing2001/Dual_Denoising.

摘要: 在本文中，我们探索了一个关键但未被研究的问题：如何从预先训练的视觉语言模型(如CLIP)中学习健壮和良好通用的3D表示。前人的工作已经证明，跨峰蒸馏可以为三维数据提供丰富而有用的知识。然而，与大多数深度学习模型一样，生成的3D学习网络仍然容易受到对抗性攻击，特别是迭代攻击。在这项工作中，我们提出了双重去噪，一个新的框架，学习稳健和良好的通用3D表示从CLIP。它将基于去噪的代理任务与一种新颖的特征去噪网络相结合，用于3D预训练。此外，我们还提出利用并行噪声推理来增强跨域环境下点云特征的泛化能力。实验表明，该模型可以有效地提高3D学习网络在零射击环境下的表征学习性能和对抗健壮性。我们的代码可以在https://github.com/luoshuqing2001/Dual_Denoising.上找到



## **14. GRACE: Graph-Regularized Attentive Convolutional Entanglement with Laplacian Smoothing for Robust DeepFake Video Detection**

GRACE：图形正规化注意卷积纠缠与拉普拉斯平滑，用于鲁棒的DeepFake视频检测 cs.CV

Submitted to TPAMI 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2406.19941v2) [paper-pdf](http://arxiv.org/pdf/2406.19941v2)

**Authors**: Chih-Chung Hsu, Shao-Ning Chen, Mei-Hsuan Wu, Yi-Fang Wang, Chia-Ming Lee, Yi-Shiuan Chou

**Abstract**: As DeepFake video manipulation techniques escalate, posing profound threats, the urgent need to develop efficient detection strategies is underscored. However, one particular issue lies with facial images being mis-detected, often originating from degraded videos or adversarial attacks, leading to unexpected temporal artifacts that can undermine the efficacy of DeepFake video detection techniques. This paper introduces a novel method for robust DeepFake video detection, harnessing the power of the proposed Graph-Regularized Attentive Convolutional Entanglement (GRACE) based on the graph convolutional network with graph Laplacian to address the aforementioned challenges. First, conventional Convolution Neural Networks are deployed to perform spatiotemporal features for the entire video. Then, the spatial and temporal features are mutually entangled by constructing a graph with sparse constraint, enforcing essential features of valid face images in the noisy face sequences remaining, thus augmenting stability and performance for DeepFake video detection. Furthermore, the Graph Laplacian prior is proposed in the graph convolutional network to remove the noise pattern in the feature space to further improve the performance. Comprehensive experiments are conducted to illustrate that our proposed method delivers state-of-the-art performance in DeepFake video detection under noisy face sequences. The source code is available at https://github.com/ming053l/GRACE.

摘要: 随着DeepFake视频操纵技术的升级，构成了深刻的威胁，迫切需要开发有效的检测策略。然而，一个特别的问题是面部图像被误检，通常是由于视频降级或对手攻击，导致意外的时间伪影，这可能会破坏DeepFake视频检测技术的效率。提出了一种新的基于图拉普拉斯卷积网络的图正则化注意力卷积纠缠(GRACE)算法，用于检测DeepFake视频。首先，使用传统的卷积神经网络来执行整个视频的时空特征。然后通过构造具有稀疏约束的图将空间特征和时间特征相互纠缠在一起，在剩余的噪声人脸序列中强化有效人脸图像的本质特征，从而增强了DeepFake视频检测的稳定性和性能。此外，在图卷积网络中提出了图拉普拉斯先验，去除了特征空间中的噪声模式，进一步提高了性能。实验结果表明，本文提出的方法在含噪人脸序列下的DeepFake视频检测中具有较好的性能。源代码可在https://github.com/ming053l/GRACE.上找到



## **15. A Two-Layer Blockchain Sharding Protocol Leveraging Safety and Liveness for Enhanced Performance**

利用安全性和活力来增强性能的两层区块链碎片协议 cs.CR

The paper has been accepted to Network and Distributed System  Security (NDSS) Symposium 2024

**SubmitDate**: 2024-07-01    [abs](http://arxiv.org/abs/2310.11373v4) [paper-pdf](http://arxiv.org/pdf/2310.11373v4)

**Authors**: Yibin Xu, Jingyi Zheng, Boris Düdder, Tijs Slaats, Yongluan Zhou

**Abstract**: Sharding is essential for improving blockchain scalability. Existing protocols overlook diverse adversarial attacks, limiting transaction throughput. This paper presents Reticulum, a groundbreaking sharding protocol addressing this issue, boosting blockchain scalability.   Reticulum employs a two-phase approach, adapting transaction throughput based on runtime adversarial attacks. It comprises "control" and "process" shards in two layers. Process shards contain at least one trustworthy node, while control shards have a majority of trusted nodes. In the first phase, transactions are written to blocks and voted on by nodes in process shards. Unanimously accepted blocks are confirmed. In the second phase, blocks without unanimous acceptance are voted on by control shards. Blocks are accepted if the majority votes in favor, eliminating first-phase opponents and silent voters. Reticulum uses unanimous voting in the first phase, involving fewer nodes, enabling more parallel process shards. Control shards finalize decisions and resolve disputes.   Experiments confirm Reticulum's innovative design, providing high transaction throughput and robustness against various network attacks, outperforming existing sharding protocols for blockchain networks.

摘要: 分片对于提高区块链可伸缩性至关重要。现有的协议忽略了不同的对抗性攻击，限制了交易吞吐量。本文提出了一种突破性的分片协议Reetum，解决了这个问题，提高了区块链的可扩展性。RENETUM采用两阶段方法，根据运行时敌意攻击调整事务吞吐量。它包括两层的“控制”和“流程”分片。进程碎片包含至少一个可信节点，而控制碎片包含大多数可信节点。在第一阶段，事务被写入块，并由流程碎片中的节点投票表决。一致接受的障碍得到确认。在第二阶段，未获得一致接受的块由控制碎片投票表决。如果多数人投赞成票，就会接受阻止，从而消除第一阶段的反对者和沉默的选民。第一阶段使用一致投票，涉及的节点更少，支持更多的并行进程碎片。控制碎片最终确定决策并解决纠纷。实验证实了ReNetum的创新设计，提供了高交易吞吐量和对各种网络攻击的稳健性，性能优于现有的区块链网络分片协议。



## **16. Fortify the Guardian, Not the Treasure: Resilient Adversarial Detectors**

强化守护者，而不是宝藏：弹性对抗探测器 cs.CV

**SubmitDate**: 2024-06-30    [abs](http://arxiv.org/abs/2404.12120v2) [paper-pdf](http://arxiv.org/pdf/2404.12120v2)

**Authors**: Raz Lapid, Almog Dubin, Moshe Sipper

**Abstract**: This paper presents RADAR-Robust Adversarial Detection via Adversarial Retraining-an approach designed to enhance the robustness of adversarial detectors against adaptive attacks, while maintaining classifier performance. An adaptive attack is one where the attacker is aware of the defenses and adapts their strategy accordingly. Our proposed method leverages adversarial training to reinforce the ability to detect attacks, without compromising clean accuracy. During the training phase, we integrate into the dataset adversarial examples, which were optimized to fool both the classifier and the adversarial detector, enabling the adversarial detector to learn and adapt to potential attack scenarios. Experimental evaluations on the CIFAR-10 and SVHN datasets demonstrate that our proposed algorithm significantly improves a detector's ability to accurately identify adaptive adversarial attacks -- without sacrificing clean accuracy.

摘要: 本文提出了RADART--通过对抗重训练的鲁棒对抗检测--一种旨在增强对抗检测器对抗自适应攻击的鲁棒性的方法，同时保持分类器性能。自适应攻击是攻击者意识到防御并相应调整策略的攻击。我们提出的方法利用对抗性训练来加强检测攻击的能力，而不会损害准确性。在训练阶段，我们将对抗性示例集成到数据集中，这些示例经过优化以愚弄分类器和对抗性检测器，使对抗性检测器能够学习和适应潜在的攻击场景。对CIFAR-10和SVHN数据集的实验评估表明，我们提出的算法显着提高了检测器准确识别自适应对抗攻击的能力，而不会牺牲清晰的准确性。



## **17. Query-Efficient Hard-Label Black-Box Attack against Vision Transformers**

针对Vision Transformers的查询高效硬标签黑匣子攻击 cs.CV

**SubmitDate**: 2024-06-29    [abs](http://arxiv.org/abs/2407.00389v1) [paper-pdf](http://arxiv.org/pdf/2407.00389v1)

**Authors**: Chao Zhou, Xiaowen Shi, Yuan-Gen Wang

**Abstract**: Recent studies have revealed that vision transformers (ViTs) face similar security risks from adversarial attacks as deep convolutional neural networks (CNNs). However, directly applying attack methodology on CNNs to ViTs has been demonstrated to be ineffective since the ViTs typically work on patch-wise encoding. This article explores the vulnerability of ViTs against adversarial attacks under a black-box scenario, and proposes a novel query-efficient hard-label adversarial attack method called AdvViT. Specifically, considering that ViTs are highly sensitive to patch modification, we propose to optimize the adversarial perturbation on the individual patches. To reduce the dimension of perturbation search space, we modify only a handful of low-frequency components of each patch. Moreover, we design a weight mask matrix for all patches to further optimize the perturbation on different regions of a whole image. We test six mainstream ViT backbones on the ImageNet-1k dataset. Experimental results show that compared with the state-of-the-art attacks on CNNs, our AdvViT achieves much lower $L_2$-norm distortion under the same query budget, sufficiently validating the vulnerability of ViTs against adversarial attacks.

摘要: 最近的研究表明，视觉转换器(VITS)面临着与深层卷积神经网络(CNN)相似的对抗攻击的安全风险。然而，直接将针对CNN的攻击方法应用于VITS已被证明是无效的，因为VITS通常工作在补丁式编码上。本文研究了VITS在黑盒场景下抵抗敌意攻击的脆弱性，提出了一种新的查询高效的硬标签敌意攻击方法AdvViT。具体地说，考虑到VITS对补丁修改高度敏感，我们提出了优化单个补丁上的对抗性扰动。为了降低扰动搜索空间的维度，我们只对每个块的少数低频分量进行了修改。此外，为了进一步优化整个图像不同区域的扰动，我们为所有的块设计了一个加权掩模矩阵。我们在ImageNet-1k数据集上测试了六个主流VIT主干。实验结果表明，与已有的针对CNN的攻击相比，在相同的查询预算下，我们的AdvViT获得了更低的$L_2$范数失真，充分验证了VITS对对手攻击的脆弱性。



## **18. DiffuseDef: Improved Robustness to Adversarial Attacks**

diffuseDef：增强对抗攻击的鲁棒性 cs.CL

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2407.00248v1) [paper-pdf](http://arxiv.org/pdf/2407.00248v1)

**Authors**: Zhenhao Li, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to system built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over different existing adversarial defense methods and achieves state-of-the-art performance against common adversarial attacks.

摘要: 预训练的语言模型在各种自然语言处理任务中显着提高了性能。然而，对抗性攻击继续对使用这些模型构建的系统构成严峻挑战，因为它们可以被精心设计的对抗性文本利用。受到扩散模型预测和减少计算机视觉中噪音的能力的启发，我们提出了一种新颖且灵活的语言分类任务对抗防御方法：DistuseDef，它在编码器和分类器之间引入了扩散层作为降噪器。在推理过程中，对抗性隐藏状态首先与采样噪音相结合，然后迭代去噪，最后集成以产生稳健的文本表示。通过集成对抗性训练、去噪和集成技术，我们证明了DistuseDef比不同的现有对抗性防御方法进行了改进，并在对抗常见对抗性攻击时实现了最先进的性能。



## **19. Deciphering the Definition of Adversarial Robustness for post-hoc OOD Detectors**

破译事后OOD检测器对抗鲁棒性的定义 cs.CR

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.15104v3) [paper-pdf](http://arxiv.org/pdf/2406.15104v3)

**Authors**: Peter Lorenz, Mario Fernandez, Jens Müller, Ullrich Köthe

**Abstract**: Detecting out-of-distribution (OOD) inputs is critical for safely deploying deep learning models in real-world scenarios. In recent years, many OOD detectors have been developed, and even the benchmarking has been standardized, i.e. OpenOOD. The number of post-hoc detectors is growing fast and showing an option to protect a pre-trained classifier against natural distribution shifts, claiming to be ready for real-world scenarios. However, its efficacy in handling adversarial examples has been neglected in the majority of studies. This paper investigates the adversarial robustness of the 16 post-hoc detectors on several evasion attacks and discuss a roadmap towards adversarial defense in OOD detectors.

摘要: 检测非分布（OOD）输入对于在现实世界场景中安全部署深度学习模型至关重要。近年来，开发了很多OOD检测器，甚至基准测试也已经标准化，即OpenOOD。事后检测器的数量正在快速增长，并显示出一种可以保护预训练的分类器免受自然分布变化的影响的选择，声称已经为现实世界的场景做好了准备。然而，它在处理敌对例子方面的功效在大多数研究中被忽视了。本文研究了16个事后检测器对多种规避攻击的对抗鲁棒性，并讨论了OOD检测器对抗防御的路线图。



## **20. Stackelberg Games with $k$-Submodular Function under Distributional Risk-Receptiveness and Robustness**

分布风险接受性和鲁棒性下$k$-次模函数的Stackelberg博弈 math.OC

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.13023v3) [paper-pdf](http://arxiv.org/pdf/2406.13023v3)

**Authors**: Seonghun Park, Manish Bansal

**Abstract**: We study submodular optimization in adversarial context, applicable to machine learning problems such as feature selection using data susceptible to uncertainties and attacks. We focus on Stackelberg games between an attacker (or interdictor) and a defender where the attacker aims to minimize the defender's objective of maximizing a $k$-submodular function. We allow uncertainties arising from the success of attacks and inherent data noise, and address challenges due to incomplete knowledge of the probability distribution of random parameters. Specifically, we introduce Distributionally Risk-Averse $k$-Submodular Interdiction Problem (DRA $k$-SIP) and Distributionally Risk-Receptive $k$-Submodular Interdiction Problem (DRR $k$-SIP) along with finitely convergent exact algorithms for solving them. The DRA $k$-SIP solution allows risk-averse interdictor to develop robust strategies for real-world uncertainties. Conversely, DRR $k$-SIP solution suggests aggressive tactics for attackers, willing to embrace (distributional) risk to inflict maximum damage, identifying critical vulnerable components, which can be used for the defender's defensive strategies. The optimal values derived from both DRA $k$-SIP and DRR $k$-SIP offer a confidence interval-like range for the expected value of the defender's objective function, capturing distributional ambiguity. We conduct computational experiments using instances of feature selection and sensor placement problems, and Wisconsin breast cancer data and synthetic data, respectively.

摘要: 我们研究了对抗性环境下的子模优化，适用于机器学习问题，例如使用对不确定性和攻击敏感的数据进行特征选择。我们主要研究攻击者(或中断者)和防御者之间的Stackelberg博弈，其中攻击者的目标是最小化防御者最大化$k$-子模函数的目标。我们允许攻击成功和固有数据噪声带来的不确定性，并解决由于不完全了解随机参数的概率分布而带来的挑战。具体地，我们引入了分布式风险厌恶$k$-子模阻断问题(DRA$k$-SIP)和分布式风险厌恶$k$-子模阻断问题(DRR$k$-SIP)，并给出了有限收敛的精确算法。DRA$k$-SIP解决方案允许风险厌恶中断者针对现实世界的不确定性制定稳健的策略。相反，DRR$k$-SIP解决方案建议攻击者采用攻击性策略，愿意承担(分布式)风险以造成最大损害，识别关键易受攻击的组件，可用于防御者的防御策略。从DRA$k$-SIP和DRR$k$-SIP导出的最佳值为防御者的目标函数的期望值提供了类似于置信度的范围，从而捕获了分布模糊性。我们分别使用特征选择和传感器放置问题的实例以及威斯康星州的乳腺癌数据和合成数据进行了计算实验。



## **21. Emotion Loss Attacking: Adversarial Attack Perception for Skeleton based on Multi-dimensional Features**

情感损失攻击：基于多维特征的骨架对抗攻击感知 cs.CV

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19815v1) [paper-pdf](http://arxiv.org/pdf/2406.19815v1)

**Authors**: Feng Liu, Qing Xu, Qijian Zheng

**Abstract**: Adversarial attack on skeletal motion is a hot topic. However, existing researches only consider part of dynamic features when measuring distance between skeleton graph sequences, which results in poor imperceptibility. To this end, we propose a novel adversarial attack method to attack action recognizers for skeletal motions. Firstly, our method systematically proposes a dynamic distance function to measure the difference between skeletal motions. Meanwhile, we innovatively introduce emotional features for complementary information. In addition, we use Alternating Direction Method of Multipliers(ADMM) to solve the constrained optimization problem, which generates adversarial samples with better imperceptibility to deceive the classifiers. Experiments show that our method is effective on multiple action classifiers and datasets. When the perturbation magnitude measured by l norms is the same, the dynamic perturbations generated by our method are much lower than that of other methods. What's more, we are the first to prove the effectiveness of emotional features, and provide a new idea for measuring the distance between skeletal motions.

摘要: 骨骼运动的对抗性攻击是一个热门话题。然而，现有的研究在度量骨架图序列之间的距离时只考虑了部分动态特征，导致隐蔽性较差。为此，我们提出了一种新的对抗性攻击方法来攻击骨骼运动的动作识别器。首先，我们的方法系统地提出了一个动态距离函数来衡量骨骼运动之间的差异。同时，我们创新性地引入了情感特征来补充信息。此外，我们使用乘子交替方向法(ADMM)来求解约束优化问题，生成的对抗性样本具有更好的隐蔽性来欺骗分类器。实验表明，该方法在多个动作分类器和数据集上是有效的。在L范数测得的摄动强度相同的情况下，我们的方法产生的动力摄动比其他方法产生的小得多。更重要的是，我们首次证明了情感特征的有效性，并为测量骨骼运动之间的距离提供了新的思路。



## **22. Deceptive Diffusion: Generating Synthetic Adversarial Examples**

欺骗性扩散：生成合成对抗示例 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19807v1) [paper-pdf](http://arxiv.org/pdf/2406.19807v1)

**Authors**: Lucas Beerens, Catherine F. Higham, Desmond J. Higham

**Abstract**: We introduce the concept of deceptive diffusion -- training a generative AI model to produce adversarial images. Whereas a traditional adversarial attack algorithm aims to perturb an existing image to induce a misclassificaton, the deceptive diffusion model can create an arbitrary number of new, misclassified images that are not directly associated with training or test images. Deceptive diffusion offers the possibility of strengthening defence algorithms by providing adversarial training data at scale, including types of misclassification that are otherwise difficult to find. In our experiments, we also investigate the effect of training on a partially attacked data set. This highlights a new type of vulnerability for generative diffusion models: if an attacker is able to stealthily poison a portion of the training data, then the resulting diffusion model will generate a similar proportion of misleading outputs.

摘要: 我们引入了欺骗性扩散的概念--训练生成式人工智能模型来产生对抗性图像。传统的对抗攻击算法旨在扰乱现有图像以引发错误分类，而欺骗性扩散模型可以创建任意数量的新的、错误分类的图像，这些图像与训练或测试图像不直接相关。欺骗性扩散通过大规模提供对抗训练数据（包括其他方式难以发现的错误分类类型）来加强防御算法。在我们的实验中，我们还研究了训练对部分攻击的数据集的影响。这凸显了生成性扩散模型的一种新型漏洞：如果攻击者能够悄悄毒害一部分训练数据，那么产生的扩散模型将生成类似比例的误导性输出。



## **23. Backdoor Attack in Prompt-Based Continual Learning**

基于预算的持续学习中的后门攻击 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19753v1) [paper-pdf](http://arxiv.org/pdf/2406.19753v1)

**Authors**: Trang Nguyen, Anh Tran, Nhat Ho

**Abstract**: Prompt-based approaches offer a cutting-edge solution to data privacy issues in continual learning, particularly in scenarios involving multiple data suppliers where long-term storage of private user data is prohibited. Despite delivering state-of-the-art performance, its impressive remembering capability can become a double-edged sword, raising security concerns as it might inadvertently retain poisoned knowledge injected during learning from private user data. Following this insight, in this paper, we expose continual learning to a potential threat: backdoor attack, which drives the model to follow a desired adversarial target whenever a specific trigger is present while still performing normally on clean samples. We highlight three critical challenges in executing backdoor attacks on incremental learners and propose corresponding solutions: (1) \emph{Transferability}: We employ a surrogate dataset and manipulate prompt selection to transfer backdoor knowledge to data from other suppliers; (2) \emph{Resiliency}: We simulate static and dynamic states of the victim to ensure the backdoor trigger remains robust during intense incremental learning processes; and (3) \emph{Authenticity}: We apply binary cross-entropy loss as an anti-cheating factor to prevent the backdoor trigger from devolving into adversarial noise. Extensive experiments across various benchmark datasets and continual learners validate our continual backdoor framework, achieving up to $100\%$ attack success rate, with further ablation studies confirming our contributions' effectiveness.

摘要: 基于提示的方法为持续学习中的数据隐私问题提供了一种尖端解决方案，特别是在涉及多个数据供应商的场景中，禁止长期存储私人用户数据。尽管提供了最先进的性能，但其令人印象深刻的记忆能力可能会成为一把双刃剑，这引发了安全问题，因为它可能会无意中保留在从私人用户数据学习过程中注入的有毒知识。根据这一见解，在本文中，我们将持续学习暴露于一个潜在的威胁：后门攻击，它驱动模型在出现特定触发时跟踪期望的对手目标，同时仍然在干净的样本上正常运行。我们强调了对增量学习者执行后门攻击的三个关键挑战并提出了相应的解决方案：(1)\emph{可传递性}：我们使用代理数据集并操纵提示选择来将后门知识传输到其他供应商的数据；(2)\emph{弹性}：我们模拟受害者的静态和动态，以确保后门触发在激烈的增量学习过程中保持健壮；以及(3)\emph{真实性}：我们应用二进制交叉熵损失作为反作弊因子，以防止后门触发演变为对抗性噪声。在各种基准数据集和不断学习的人中进行的广泛实验验证了我们的持续后门框架，实现了高达100美元的攻击成功率，进一步的消融研究证实了我们的贡献的有效性。



## **24. Steering cooperation: Adversarial attacks on prisoner's dilemma in complex networks**

指导合作：对复杂网络中囚犯困境的对抗攻击 physics.soc-ph

14 pages, 4 figures

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19692v1) [paper-pdf](http://arxiv.org/pdf/2406.19692v1)

**Authors**: Kazuhiro Takemoto

**Abstract**: This study examines the application of adversarial attack concepts to control the evolution of cooperation in the prisoner's dilemma game in complex networks. Specifically, it proposes a simple adversarial attack method that drives players' strategies towards a target state by adding small perturbations to social networks. The proposed method is evaluated on both model and real-world networks. Numerical simulations demonstrate that the proposed method can effectively promote cooperation with significantly smaller perturbations compared to other techniques. Additionally, this study shows that adversarial attacks can also be useful in inhibiting cooperation (promoting defection). The findings reveal that adversarial attacks on social networks can be potent tools for both promoting and inhibiting cooperation, opening new possibilities for controlling cooperative behavior in social systems while also highlighting potential risks.

摘要: 本研究探讨了对抗攻击概念在复杂网络中囚犯困境游戏中控制合作演变的应用。具体来说，它提出了一种简单的对抗攻击方法，通过向社交网络添加小扰动来推动玩家的策略走向目标状态。在模型和现实世界网络上对所提出的方法进行了评估。数值模拟表明，与其他技术相比，所提出的方法可以有效地促进协作，且扰动要小得多。此外，这项研究表明，对抗性攻击也可能有助于抑制合作（促进叛逃）。研究结果表明，对社交网络的对抗性攻击可以成为促进和抑制合作的有力工具，为控制社会系统中的合作行为开辟了新的可能性，同时也凸显了潜在的风险。



## **25. IDT: Dual-Task Adversarial Attacks for Privacy Protection**

IDT：隐私保护的双重任务对抗攻击 cs.CL

28 pages, 1 figure

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19642v1) [paper-pdf](http://arxiv.org/pdf/2406.19642v1)

**Authors**: Pedro Faustini, Shakila Mahjabin Tonni, Annabelle McIver, Qiongkai Xu, Mark Dras

**Abstract**: Natural language processing (NLP) models may leak private information in different ways, including membership inference, reconstruction or attribute inference attacks. Sensitive information may not be explicit in the text, but hidden in underlying writing characteristics. Methods to protect privacy can involve using representations inside models that are demonstrated not to detect sensitive attributes or -- for instance, in cases where users might not trust a model, the sort of scenario of interest here -- changing the raw text before models can have access to it. The goal is to rewrite text to prevent someone from inferring a sensitive attribute (e.g. the gender of the author, or their location by the writing style) whilst keeping the text useful for its original intention (e.g. the sentiment of a product review). The few works tackling this have focused on generative techniques. However, these often create extensively different texts from the original ones or face problems such as mode collapse. This paper explores a novel adaptation of adversarial attack techniques to manipulate a text to deceive a classifier w.r.t one task (privacy) whilst keeping the predictions of another classifier trained for another task (utility) unchanged. We propose IDT, a method that analyses predictions made by auxiliary and interpretable models to identify which tokens are important to change for the privacy task, and which ones should be kept for the utility task. We evaluate different datasets for NLP suitable for different tasks. Automatic and human evaluations show that IDT retains the utility of text, while also outperforming existing methods when deceiving a classifier w.r.t privacy task.

摘要: 自然语言处理(NLP)模型可能以不同的方式泄露隐私信息，包括成员关系推理、重构或属性推理攻击。敏感信息可能在文本中不是显性的，而是隐藏在潜在的写作特征中。保护隐私的方法可能涉及在模型中使用表示，这些表示被演示为不会检测敏感属性，或者--例如，在用户可能不信任模型的情况下，这里涉及的场景--在模型可以访问它之前更改原始文本。目标是重写文本，以防止有人推断敏感属性(例如，作者的性别或他们的写作风格)，同时保持文本对其原始意图(例如，产品评论的情绪)的有用。解决这一问题的少数作品都集中在生成技术上。然而，这些通常会产生与原始文本截然不同的文本，或者面临模式崩溃等问题。本文探索了一种新颖的对抗性攻击技术，以操纵文本来欺骗一个任务(隐私)的分类器，同时保持为另一个任务(效用)训练的另一个分类器的预测不变。我们提出了IDT，这是一种分析辅助模型和可解释模型所做预测的方法，以确定哪些令牌对于隐私任务是重要的，哪些应该为实用任务保留。我们评估了适合不同任务的自然语言处理的不同数据集。自动和人工评估表明，IDT保留了文本的效用，同时在欺骗分类器w.r.t隐私任务时也优于现有方法。



## **26. Data-Driven Lipschitz Continuity: A Cost-Effective Approach to Improve Adversarial Robustness**

数据驱动的Lipschitz连续性：提高对抗稳健性的经济有效方法 cs.LG

**SubmitDate**: 2024-06-28    [abs](http://arxiv.org/abs/2406.19622v1) [paper-pdf](http://arxiv.org/pdf/2406.19622v1)

**Authors**: Erh-Chung Chen, Pin-Yu Chen, I-Hsin Chung, Che-Rung Lee

**Abstract**: The security and robustness of deep neural networks (DNNs) have become increasingly concerning. This paper aims to provide both a theoretical foundation and a practical solution to ensure the reliability of DNNs. We explore the concept of Lipschitz continuity to certify the robustness of DNNs against adversarial attacks, which aim to mislead the network with adding imperceptible perturbations into inputs. We propose a novel algorithm that remaps the input domain into a constrained range, reducing the Lipschitz constant and potentially enhancing robustness. Unlike existing adversarially trained models, where robustness is enhanced by introducing additional examples from other datasets or generative models, our method is almost cost-free as it can be integrated with existing models without requiring re-training. Experimental results demonstrate the generalizability of our method, as it can be combined with various models and achieve enhancements in robustness. Furthermore, our method achieves the best robust accuracy for CIFAR10, CIFAR100, and ImageNet datasets on the RobustBench leaderboard.

摘要: 深度神经网络(DNN)的安全性和健壮性越来越受到人们的关注。我们利用Lipschitz连续性的概念来证明DNN对敌意攻击的健壮性，目的是通过在输入中添加不可察觉的扰动来误导网络。与现有的对抗性训练模型不同，我们的方法通过引入来自其他数据集或生成性模型的额外样本来增强稳健性，因为它可以与现有模型集成在一起，而不需要重新训练。此外，我们的方法对于罗布斯本奇排行榜上的CIFAR10、CIFAR100和ImageNet数据集获得了最好的稳健精度。



## **27. Zero-Query Adversarial Attack on Black-box Automatic Speech Recognition Systems**

黑匣子自动语音识别系统的零查询对抗攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19311v1) [paper-pdf](http://arxiv.org/pdf/2406.19311v1)

**Authors**: Zheng Fang, Tao Wang, Lingchen Zhao, Shenyi Zhang, Bowen Li, Yunjie Ge, Qi Li, Chao Shen, Qian Wang

**Abstract**: In recent years, extensive research has been conducted on the vulnerability of ASR systems, revealing that black-box adversarial example attacks pose significant threats to real-world ASR systems. However, most existing black-box attacks rely on queries to the target ASRs, which is impractical when queries are not permitted. In this paper, we propose ZQ-Attack, a transfer-based adversarial attack on ASR systems in the zero-query black-box setting. Through a comprehensive review and categorization of modern ASR technologies, we first meticulously select surrogate ASRs of diverse types to generate adversarial examples. Following this, ZQ-Attack initializes the adversarial perturbation with a scaled target command audio, rendering it relatively imperceptible while maintaining effectiveness. Subsequently, to achieve high transferability of adversarial perturbations, we propose a sequential ensemble optimization algorithm, which iteratively optimizes the adversarial perturbation on each surrogate model, leveraging collaborative information from other models. We conduct extensive experiments to evaluate ZQ-Attack. In the over-the-line setting, ZQ-Attack achieves a 100% success rate of attack (SRoA) with an average signal-to-noise ratio (SNR) of 21.91dB on 4 online speech recognition services, and attains an average SRoA of 100% and SNR of 19.67dB on 16 open-source ASRs. For commercial intelligent voice control devices, ZQ-Attack also achieves a 100% SRoA with an average SNR of 15.77dB in the over-the-air setting.

摘要: 近年来，人们对ASR系统的脆弱性进行了广泛的研究，发现黑盒对抗性范例攻击对现实世界的ASR系统构成了巨大的威胁。然而，大多数现有的黑盒攻击依赖于对目标ASR的查询，当查询不被允许时，这是不切实际的。在本文中，我们提出了ZQ-Attack，这是一种在零查询黑箱环境下对ASR系统进行的基于传输的对抗性攻击。通过对现代ASR技术的全面回顾和分类，我们首先精心选择了不同类型的代理ASR来生成对抗性实例。在此之后，ZQ-Attack使用缩放的目标命令音频来初始化对抗性扰动，使其在保持有效性的同时相对难以察觉。随后，为了实现对抗性扰动的高可转移性，我们提出了一种序列集成优化算法，该算法利用来自其他模型的协作信息，迭代地优化每个代理模型上的对抗性扰动。我们进行了大量的实验来评估ZQ攻击。在线上环境下，ZQ-Attack在4种在线语音识别服务上达到100%的攻击成功率(SRoA)，平均信噪比(SNR)为21.91dB；在16种开源ASR上，平均SRoA和SNR分别为100%和19.67dB。对于商业智能语音控制设备，ZQ-Attack还在空中设置下实现了100%的SRoA，平均信噪比为15.77dB。



## **28. GPTFUZZER: Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts**

GPTFUZER：Red将大型语言模型与自动生成的越狱脚本结合起来 cs.AI

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2309.10253v4) [paper-pdf](http://arxiv.org/pdf/2309.10253v4)

**Authors**: Jiahao Yu, Xingwei Lin, Zheng Yu, Xinyu Xing

**Abstract**: Large language models (LLMs) have recently experienced tremendous popularity and are widely used from casual conversations to AI-driven programming. However, despite their considerable success, LLMs are not entirely reliable and can give detailed guidance on how to conduct harmful or illegal activities. While safety measures can reduce the risk of such outputs, adversarial jailbreak attacks can still exploit LLMs to produce harmful content. These jailbreak templates are typically manually crafted, making large-scale testing challenging.   In this paper, we introduce GPTFuzz, a novel black-box jailbreak fuzzing framework inspired by the AFL fuzzing framework. Instead of manual engineering, GPTFuzz automates the generation of jailbreak templates for red-teaming LLMs. At its core, GPTFuzz starts with human-written templates as initial seeds, then mutates them to produce new templates. We detail three key components of GPTFuzz: a seed selection strategy for balancing efficiency and variability, mutate operators for creating semantically equivalent or similar sentences, and a judgment model to assess the success of a jailbreak attack.   We evaluate GPTFuzz against various commercial and open-source LLMs, including ChatGPT, LLaMa-2, and Vicuna, under diverse attack scenarios. Our results indicate that GPTFuzz consistently produces jailbreak templates with a high success rate, surpassing human-crafted templates. Remarkably, GPTFuzz achieves over 90% attack success rates against ChatGPT and Llama-2 models, even with suboptimal initial seed templates. We anticipate that GPTFuzz will be instrumental for researchers and practitioners in examining LLM robustness and will encourage further exploration into enhancing LLM safety.

摘要: 大型语言模型(LLM)最近经历了巨大的流行，并被广泛使用，从随意的对话到人工智能驱动的编程。然而，尽管LLM取得了相当大的成功，但它们并不完全可靠，可以就如何进行有害或非法活动提供详细指导。虽然安全措施可以降低此类输出的风险，但对抗性越狱攻击仍然可以利用LLMS产生有害内容。这些越狱模板通常是手动制作的，这使得大规模测试具有挑战性。在本文中，我们介绍了一种新的黑盒越狱模糊框架GPTFuzz，该框架受到AFL模糊框架的启发。GPTFuzz不是手动设计，而是自动生成用于红队LLM的越狱模板。在其核心，GPTFuzz以人类编写的模板作为初始种子，然后对它们进行突变以产生新的模板。我们详细介绍了GPTFuzz的三个关键组成部分：用于平衡效率和可变性的种子选择策略，用于创建语义等价或相似句子的变异算子，以及用于评估越狱攻击成功的判断模型。我们在不同的攻击场景下，针对各种商业和开源LLM，包括ChatGPT、骆驼2和维库纳，对GPTFuzz进行了评估。我们的结果表明，GPTFuzz一致地生成了成功率较高的越狱模板，超过了人工制作的模板。值得注意的是，GPTFuzz对ChatGPT和Llama-2模型的攻击成功率超过90%，即使在初始种子模板不是最优的情况下也是如此。我们预计，GPTFuzz将有助于研究人员和从业者检查LLM的稳健性，并将鼓励进一步探索增强LLM的安全性。



## **29. Spiking Convolutional Neural Networks for Text Classification**

用于文本分类的尖峰卷积神经网络 cs.NE

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19230v1) [paper-pdf](http://arxiv.org/pdf/2406.19230v1)

**Authors**: Changze Lv, Jianhan Xu, Xiaoqing Zheng

**Abstract**: Spiking neural networks (SNNs) offer a promising pathway to implement deep neural networks (DNNs) in a more energy-efficient manner since their neurons are sparsely activated and inferences are event-driven. However, there have been very few works that have demonstrated the efficacy of SNNs in language tasks partially because it is non-trivial to represent words in the forms of spikes and to deal with variable-length texts by SNNs. This work presents a "conversion + fine-tuning" two-step method for training SNNs for text classification and proposes a simple but effective way to encode pre-trained word embeddings as spike trains. We show empirically that after fine-tuning with surrogate gradients, the converted SNNs achieve comparable results to their DNN counterparts with much less energy consumption across multiple datasets for both English and Chinese. We also show that such SNNs are more robust to adversarial attacks than DNNs.

摘要: 尖峰神经网络（SNN）为以更节能的方式实施深度神经网络（DNN）提供了一种有希望的途径，因为它们的神经元是稀疏激活的，并且推理是事件驱动的。然而，很少有作品证明了SNN在语言任务中的功效，部分原因是以尖峰形式表示单词并通过SNN处理变长文本并不是小事。这项工作提出了一种“转换+微调”两步方法来训练SNN进行文本分类，并提出了一种简单但有效的方法来将预训练的单词嵌入编码为尖峰序列。我们经验表明，在使用替代梯度进行微调后，转换后的SNN可以实现与DNN对应的结果，并且英语和中文的多个数据集的能耗要低得多。我们还表明，此类SNN比DNN对对抗攻击更稳健。



## **30. Understanding the Security Benefits and Overheads of Emerging Industry Solutions to DRAM Read Disturbance**

了解新兴行业解决方案的安全优势和管理费用针对内存读取干扰 cs.CR

To appear in DRAMSec 2024

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.19094v1) [paper-pdf](http://arxiv.org/pdf/2406.19094v1)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Oğuz Ergin, Onur Mutlu

**Abstract**: We present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC), described in JEDEC DDR5 specification's April 2024 update. Unlike prior state-of-the-art that advises the memory controller to periodically issue refresh management (RFM) commands, which provides the DRAM chip with time to perform refreshes, PRAC introduces a new back-off signal. PRAC's back-off signal propagates from the DRAM chip to the memory controller and forces the memory controller to 1) stop serving requests and 2) issue RFM commands. As a result, RFM commands are issued when needed as opposed to periodically, reducing RFM's overheads. We analyze PRAC in four steps. First, we define an adversarial access pattern that represents the worst-case for PRAC's security. Second, we investigate PRAC's configurations and security implications. Our analyses show that PRAC can be configured for secure operation as long as no bitflip occurs before accessing a memory location 10 times. Third, we evaluate the performance impact of PRAC and compare it against prior works using Ramulator 2.0. Our analysis shows that while PRAC incurs less than 13.4% performance overhead for today's DRAM chips, its performance overheads can reach up to 63.2% for future DRAM chips that are more vulnerable to read disturbance bitflips. Fourth, we define an availability adversarial access pattern that exacerbates PRAC's performance overhead to perform a memory performance attack, demonstrating that such an adversarial pattern can hog up to 79% of DRAM throughput and degrade system throughput by up to 65%. We discuss PRAC's implications on future systems and foreshadow future research directions. To aid future research, we open-source our implementations and scripts at https://github.com/CMU-SAFARI/ramulator2.

摘要: 我们首次对JEDEC DDR5规范2024年4月更新中描述的最先进的片上DRAM读取干扰缓解方法-每行激活计数(PRAC)-进行了严格的安全、性能、能量和成本分析。与建议存储器控制器定期发出刷新管理(RFM)命令(为DRAM芯片提供执行刷新的时间)的现有技术不同，PRAC引入了新的退避信号。PRAC的退避信号从DRAM芯片传播到存储器控制器，并迫使存储器控制器1)停止服务请求和2)发出RFM命令。因此，RFM命令在需要时发出，而不是定期发出，从而减少了RFM的管理费用。我们分四个步骤对PRAC进行分析。首先，我们定义了一种对抗性访问模式，它代表了对PRAC安全的最坏情况。其次，我们调查了PRAC的配置和安全影响。我们的分析表明，只要在访问一个存储单元10次之前没有发生位翻转，就可以将PRAC配置为安全操作。第三，我们评估了PRAC对性能的影响，并将其与使用Ramuler2.0的前人工作进行了比较。我们的分析表明，虽然PRAC对今天的DRAM芯片产生的性能开销不到13.4%，但对于更容易受到读取干扰位翻转的未来DRAM芯片，其性能开销可能高达63.2%。第四，我们定义了一种可用性对抗性访问模式，它加剧了PRAC执行内存性能攻击的性能开销，证明了这种对抗性模式可以占用高达79%的DRAM吞吐量，并将系统吞吐量降低高达65%。我们讨论了PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/ramulator2.上开放了我们的实现和脚本



## **31. Intriguing Properties of Adversarial ML Attacks in the Problem Space [Extended Version]**

问题空间中对抗ML攻击的有趣性质[扩展版本] cs.CR

This arXiv version (v3) corresponds to an extended version

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/1911.02142v3) [paper-pdf](http://arxiv.org/pdf/1911.02142v3)

**Authors**: Jacopo Cortellazzi, Feargus Pendlebury, Daniel Arp, Erwin Quiring, Fabio Pierazzi, Lorenzo Cavallaro

**Abstract**: Recent research efforts on adversarial machine learning (ML) have investigated problem-space attacks, focusing on the generation of real evasive objects in domains where, unlike images, there is no clear inverse mapping to the feature space (e.g., software). However, the design, comparison, and real-world implications of problem-space attacks remain underexplored. This article makes three major contributions. Firstly, we propose a general formalization for adversarial ML evasion attacks in the problem-space, which includes the definition of a comprehensive set of constraints on available transformations, preserved semantics, absent artifacts, and plausibility. We shed light on the relationship between feature space and problem space, and we introduce the concept of side-effect features as the by-product of the inverse feature-mapping problem. This enables us to define and prove necessary and sufficient conditions for the existence of problem-space attacks. Secondly, building on our general formalization, we propose a novel problem-space attack on Android malware that overcomes past limitations in terms of semantics and artifacts. We have tested our approach on a dataset with 150K Android apps from 2016 and 2018 which show the practical feasibility of evading a state-of-the-art malware classifier along with its hardened version. Thirdly, we explore the effectiveness of adversarial training as a possible approach to enforce robustness against adversarial samples, evaluating its effectiveness on the considered machine learning models under different scenarios. Our results demonstrate that "adversarial-malware as a service" is a realistic threat, as we automatically generate thousands of realistic and inconspicuous adversarial applications at scale, where on average it takes only a few minutes to generate an adversarial instance.

摘要: 最近关于对抗性机器学习(ML)的研究工作已经研究了问题空间攻击，集中在与图像不同的领域中生成真实的回避对象，其中不存在到特征空间的明确的逆映射(例如软件)。然而，问题空间攻击的设计、比较和现实世界的影响仍然没有得到充分的探索。本文有三个主要贡献。首先，我们提出了问题空间中对抗性ML规避攻击的一般形式化，其中包括对可用变换、保留语义、缺失伪像和似然性的一组综合约束的定义。我们阐明了特征空间和问题空间之间的关系，并引入了副作用特征的概念，作为逆特征映射问题的副产品。这使我们能够定义和证明存在问题空间攻击的充要条件。其次，在一般形式化的基础上，提出了一种新的针对Android恶意软件的问题空间攻击方法，克服了过去在语义和伪装方面的局限性。我们在2016年和2018年使用15万个Android应用程序的数据集上测试了我们的方法，这些程序表明了逃避最先进的恶意软件分类器及其强化版本的实际可行性。第三，我们探讨了对抗性训练作为一种可能的方法来增强对对抗性样本的稳健性的有效性，并在不同场景下对所考虑的机器学习模型的有效性进行了评估。我们的结果表明，“恶意软件即服务”是一种现实的威胁，因为我们会自动生成数以千计的真实且不起眼的敌意应用程序，平均只需要几分钟就能生成一个对抗性实例。



## **32. Dysca: A Dynamic and Scalable Benchmark for Evaluating Perception Ability of LVLMs**

Dysca：评估LVLM感知能力的动态和可扩展基准 cs.CV

**SubmitDate**: 2024-06-27    [abs](http://arxiv.org/abs/2406.18849v1) [paper-pdf](http://arxiv.org/pdf/2406.18849v1)

**Authors**: Jie Zhang, Zhongqi Wang, Mengqi Lei, Zheng Yuan, Bei Yan, Shiguang Shan, Xilin Chen

**Abstract**: Currently many benchmarks have been proposed to evaluate the perception ability of the Large Vision-Language Models (LVLMs). However, most benchmarks conduct questions by selecting images from existing datasets, resulting in the potential data leakage. Besides, these benchmarks merely focus on evaluating LVLMs on the realistic style images and clean scenarios, leaving the multi-stylized images and noisy scenarios unexplored. In response to these challenges, we propose a dynamic and scalable benchmark named Dysca for evaluating LVLMs by leveraging synthesis images. Specifically, we leverage Stable Diffusion and design a rule-based method to dynamically generate novel images, questions and the corresponding answers. We consider 51 kinds of image styles and evaluate the perception capability in 20 subtasks. Moreover, we conduct evaluations under 4 scenarios (i.e., Clean, Corruption, Print Attacking and Adversarial Attacking) and 3 question types (i.e., Multi-choices, True-or-false and Free-form). Thanks to the generative paradigm, Dysca serves as a scalable benchmark for easily adding new subtasks and scenarios. A total of 8 advanced open-source LVLMs with 10 checkpoints are evaluated on Dysca, revealing the drawbacks of current LVLMs. The benchmark is released in \url{https://github.com/Benchmark-Dysca/Dysca}.

摘要: 目前，人们已经提出了许多基准来评估大型视觉语言模型的感知能力。然而，大多数基准测试通过从现有数据集中选择图像来进行问题，从而导致潜在的数据泄漏。此外，这些基准只关注真实感风格的图像和干净的场景来评估LVLMS，而对多风格化的图像和噪声场景没有进行探索。为了应对这些挑战，我们提出了一种动态的、可扩展的基准测试DYSCA，用于利用合成图像来评估LVLMS。具体地说，我们利用稳定扩散并设计了一种基于规则的方法来动态生成新的图像、问题和相应的答案。我们考虑了51种图像风格，并在20个子任务中评估了感知能力。此外，我们还在4个场景(即廉洁、腐败、打印攻击和对抗性攻击)和3个问题类型(即多项选择、对错和自由形式)下进行了评估。多亏了生成性范式，Dysca成为了一个可伸缩的基准，可以轻松添加新的子任务和场景。在Dysca上对8个具有10个检查点的高级开源LVLMS进行了评估，揭示了当前LVLMS的缺陷。该基准测试在\url{https://github.com/Benchmark-Dysca/Dysca}.中发布



## **33. A Zero Auxiliary Knowledge Membership Inference Attack on Aggregate Location Data**

对聚合位置数据的零辅助知识隶属度推理攻击 cs.CR

To be published in PETS 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18671v1) [paper-pdf](http://arxiv.org/pdf/2406.18671v1)

**Authors**: Vincent Guan, Florent Guépin, Ana-Maria Cretu, Yves-Alexandre de Montjoye

**Abstract**: Location data is frequently collected from populations and shared in aggregate form to guide policy and decision making. However, the prevalence of aggregated data also raises the privacy concern of membership inference attacks (MIAs). MIAs infer whether an individual's data contributed to the aggregate release. Although effective MIAs have been developed for aggregate location data, these require access to an extensive auxiliary dataset of individual traces over the same locations, which are collected from a similar population. This assumption is often impractical given common privacy practices surrounding location data. To measure the risk of an MIA performed by a realistic adversary, we develop the first Zero Auxiliary Knowledge (ZK) MIA on aggregate location data, which eliminates the need for an auxiliary dataset of real individual traces. Instead, we develop a novel synthetic approach, such that suitable synthetic traces are generated from the released aggregate. We also develop methods to correct for bias and noise, to show that our synthetic-based attack is still applicable when privacy mechanisms are applied prior to release. Using two large-scale location datasets, we demonstrate that our ZK MIA matches the state-of-the-art Knock-Knock (KK) MIA across a wide range of settings, including popular implementations of differential privacy (DP) and suppression of small counts. Furthermore, we show that ZK MIA remains highly effective even when the adversary only knows a small fraction (10%) of their target's location history. This demonstrates that effective MIAs can be performed by realistic adversaries, highlighting the need for strong DP protection.

摘要: 地点数据经常从人口中收集，并以汇总的形式共享，以指导政策和决策。然而，聚合数据的流行也引发了对成员身份推断攻击(MIA)的隐私问题。MIA推断个人的数据是否对总体发布起到了作用。虽然已经为聚合位置数据制定了有效的MIA，但这些数据需要访问从相似人口收集的相同位置的单个踪迹的广泛辅助数据集。考虑到围绕位置数据的常见隐私做法，这种假设通常是不切实际的。为了衡量现实对手执行MIA的风险，我们开发了第一个关于聚合位置数据的零辅助知识(ZK)MIA，它消除了对真实个人痕迹的辅助数据集的需要。相反，我们开发了一种新的合成方法，以便从释放的聚集体中生成合适的合成痕迹。我们还开发了纠正偏差和噪声的方法，以表明当在发布之前应用隐私机制时，我们基于合成的攻击仍然适用。使用两个大规模位置数据集，我们证明了我们的ZK MIA在广泛的设置范围内与最先进的敲门(KK)MIA相匹配，包括流行的差异隐私(DP)实施和抑制小计数。此外，我们还表明，即使对手只知道目标位置历史的一小部分(10%)，ZK MIA仍然非常有效。这表明，有效的MIA可以由现实的对手执行，突显了强大的DP保护的必要性。



## **34. WildTeaming at Scale: From In-the-Wild Jailbreaks to (Adversarially) Safer Language Models**

大规模的野生合作：从野外越狱到（相反）更安全的语言模型 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18510v1) [paper-pdf](http://arxiv.org/pdf/2406.18510v1)

**Authors**: Liwei Jiang, Kavel Rao, Seungju Han, Allyson Ettinger, Faeze Brahman, Sachin Kumar, Niloofar Mireshghallah, Ximing Lu, Maarten Sap, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildTeaming, an automatic LLM safety red-teaming framework that mines in-the-wild user-chatbot interactions to discover 5.7K unique clusters of novel jailbreak tactics, and then composes multiple tactics for systematic exploration of novel jailbreaks. Compared to prior work that performed red-teaming via recruited human workers, gradient-based optimization, or iterative revision with LLMs, our work investigates jailbreaks from chatbot users who were not specifically instructed to break the system. WildTeaming reveals previously unidentified vulnerabilities of frontier LLMs, resulting in up to 4.6x more diverse and successful adversarial attacks compared to state-of-the-art jailbreak methods.   While many datasets exist for jailbreak evaluation, very few open-source datasets exist for jailbreak training, as safety training data has been closed even when model weights are open. With WildTeaming we create WildJailbreak, a large-scale open-source synthetic safety dataset with 262K vanilla (direct request) and adversarial (complex jailbreak) prompt-response pairs. To mitigate exaggerated safety behaviors, WildJailbreak provides two contrastive types of queries: 1) harmful queries (vanilla & adversarial) and 2) benign queries that resemble harmful queries in form but contain no harm. As WildJailbreak considerably upgrades the quality and scale of existing safety resources, it uniquely enables us to examine the scaling effects of data and the interplay of data properties and model capabilities during safety training. Through extensive experiments, we identify the training properties that enable an ideal balance of safety behaviors: appropriate safeguarding without over-refusal, effective handling of vanilla and adversarial queries, and minimal, if any, decrease in general capabilities. All components of WildJailbeak contribute to achieving balanced safety behaviors of models.

摘要: 我们介绍了WildTeaming，一个自动的LLM安全红色团队框架，它在野外挖掘用户和聊天机器人的交互来发现5.7k独特的新越狱战术簇，然后组成多个策略来系统地探索新的越狱战术。与之前通过招募人类工人进行红团队合作、基于梯度的优化或使用LLMS进行迭代修订相比，我们的工作调查了聊天机器人用户的越狱行为，这些用户没有得到明确的指示来破坏系统。WildTeaming揭示了FronTier LLMS以前未知的漏洞，导致与最先进的越狱方法相比，多样化和成功的对抗性攻击高达4.6倍。虽然存在许多用于越狱评估的数据集，但用于越狱培训的开源数据集很少，因为即使在模型重量打开的情况下，安全培训数据也已关闭。使用WildTeaming，我们创建了WildJailBreak，这是一个大规模的开源合成安全数据集，具有262K的普通(直接请求)和对抗性(复杂的越狱)提示-响应对。为了减少夸张的安全行为，WildJailBreak提供了两种对比类型的查询：1)有害查询(普通查询和对抗性查询)和2)在形式上类似于有害查询但不包含危害的良性查询。由于WildJailBreak极大地提升了现有安全资源的质量和规模，它独特地使我们能够在安全培训期间检查数据的缩放效应以及数据属性和模型功能的相互作用。通过广泛的实验，我们确定了能够实现安全行为的理想平衡的训练属性：适当的保护而不过度拒绝，有效地处理普通和敌意的查询，以及最小程度地降低一般能力(如果有的话)。WildJailbeak的所有组件都有助于实现模型的安全行为平衡。



## **35. WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs**

WildGuard：针对LLC安全风险、越狱和拒绝的开放式一站式审核工具 cs.CL

First two authors contributed equally. Third and fourth authors  contributed equally

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18495v1) [paper-pdf](http://arxiv.org/pdf/2406.18495v1)

**Authors**: Seungju Han, Kavel Rao, Allyson Ettinger, Liwei Jiang, Bill Yuchen Lin, Nathan Lambert, Yejin Choi, Nouha Dziri

**Abstract**: We introduce WildGuard -- an open, light-weight moderation tool for LLM safety that achieves three goals: (1) identifying malicious intent in user prompts, (2) detecting safety risks of model responses, and (3) determining model refusal rate. Together, WildGuard serves the increasing needs for automatic safety moderation and evaluation of LLM interactions, providing a one-stop tool with enhanced accuracy and broad coverage across 13 risk categories. While existing open moderation tools such as Llama-Guard2 score reasonably well in classifying straightforward model interactions, they lag far behind a prompted GPT-4, especially in identifying adversarial jailbreaks and in evaluating models' refusals, a key measure for evaluating safety behaviors in model responses.   To address these challenges, we construct WildGuardMix, a large-scale and carefully balanced multi-task safety moderation dataset with 92K labeled examples that cover vanilla (direct) prompts and adversarial jailbreaks, paired with various refusal and compliance responses. WildGuardMix is a combination of WildGuardTrain, the training data of WildGuard, and WildGuardTest, a high-quality human-annotated moderation test set with 5K labeled items covering broad risk scenarios. Through extensive evaluations on WildGuardTest and ten existing public benchmarks, we show that WildGuard establishes state-of-the-art performance in open-source safety moderation across all the three tasks compared to ten strong existing open-source moderation models (e.g., up to 26.4% improvement on refusal detection). Importantly, WildGuard matches and sometimes exceeds GPT-4 performance (e.g., up to 3.9% improvement on prompt harmfulness identification). WildGuard serves as a highly effective safety moderator in an LLM interface, reducing the success rate of jailbreak attacks from 79.8% to 2.4%.

摘要: 我们介绍了WildGuard--一个开放的、轻量级的LLM安全防御工具，它实现了三个目标：(1)识别用户提示中的恶意意图，(2)检测模型响应的安全风险，(3)确定模型拒绝率。综合起来，WildGuard可满足日益增长的自动安全审核和评估LLM交互作用的需求，提供了一种一站式工具，具有更高的准确性和广泛的覆盖范围，涵盖13个风险类别。虽然现有的开放式审核工具，如Llama-Guard2，在对直接的模型交互进行分类方面得分相当好，但它们远远落后于GPT-4，特别是在识别对抗性越狱和评估模型拒绝方面，这是评估模型响应中安全行为的关键指标。为了应对这些挑战，我们构建了WildGuardMix，这是一个大规模的、仔细平衡的多任务安全缓和数据集，具有92K标记的示例，涵盖普通(直接)提示和对抗性越狱，并与各种拒绝和合规响应配对。WildGuardMix是WildGuard的训练数据WildGuardTrain和WildGuardTest的组合，WildGuardTest是一种高质量的人工注释适度测试集，具有覆盖广泛风险情景的5K标签项目。通过对WildGuardTest和十个现有公共基准的广泛评估，我们表明WildGuard在所有三个任务中建立了开源安全适度的最先进性能，而不是现有的十个强大的开源适度模型(例如，拒绝检测方面高达26.4%的改进)。重要的是，WildGuard的性能与GPT-4相当，有时甚至超过GPT-4(例如，在及时识别危害性方面最高提高3.9%)。WildGuard在LLM界面中充当高效的安全调节器，将越狱攻击的成功率从79.8%降低到2.4%。



## **36. Enhancing Federated Learning with Adaptive Differential Privacy and Priority-Based Aggregation**

利用自适应差异隐私和基于优先级的聚合增强联邦学习 cs.LG

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18491v1) [paper-pdf](http://arxiv.org/pdf/2406.18491v1)

**Authors**: Mahtab Talaei, Iman Izadi

**Abstract**: Federated learning (FL), a novel branch of distributed machine learning (ML), develops global models through a private procedure without direct access to local datasets. However, it is still possible to access the model updates (gradient updates of deep neural networks) transferred between clients and servers, potentially revealing sensitive local information to adversaries using model inversion attacks. Differential privacy (DP) offers a promising approach to addressing this issue by adding noise to the parameters. On the other hand, heterogeneities in data structure, storage, communication, and computational capabilities of devices can cause convergence problems and delays in developing the global model. A personalized weighted averaging of local parameters based on the resources of each device can yield a better aggregated model in each round. In this paper, to efficiently preserve privacy, we propose a personalized DP framework that injects noise based on clients' relative impact factors and aggregates parameters while considering heterogeneities and adjusting properties. To fulfill the DP requirements, we first analyze the convergence boundary of the FL algorithm when impact factors are personalized and fixed throughout the learning process. We then further study the convergence property considering time-varying (adaptive) impact factors.

摘要: 联合学习(FL)是分布式机器学习(ML)的一个新分支，它通过私有过程建立全局模型，而不需要直接访问局部数据集。然而，仍有可能访问在客户端和服务器之间传输的模型更新(深度神经网络的梯度更新)，从而潜在地向使用模型反转攻击的攻击者泄露敏感的本地信息。通过在参数中添加噪声，差分隐私(DP)为解决这一问题提供了一种很有前途的方法。另一方面，设备的数据结构、存储、通信和计算能力的异构性可能会导致全局模型开发的收敛问题和延迟。基于每个设备的资源的本地参数的个性化加权平均可以在每一轮中产生更好的聚合模型。为了有效地保护隐私，我们提出了一种个性化的DP框架，该框架根据客户的相对影响因子和聚集参数来注入噪声，同时考虑异构性和调整属性。为了满足动态规划的要求，我们首先分析了影响因素在学习过程中被个性化和固定时FL算法的收敛边界。然后，我们进一步研究了考虑时变(自适应)影响因子时的收敛性质。



## **37. MultiAgent Collaboration Attack: Investigating Adversarial Attacks in Large Language Model Collaborations via Debate**

多Agent协作攻击：通过辩论调查大型语言模型协作中的对抗性攻击 cs.CL

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.14711v2) [paper-pdf](http://arxiv.org/pdf/2406.14711v2)

**Authors**: Alfonso Amayuelas, Xianjun Yang, Antonis Antoniades, Wenyue Hua, Liangming Pan, William Wang

**Abstract**: Large Language Models (LLMs) have shown exceptional results on current benchmarks when working individually. The advancement in their capabilities, along with a reduction in parameter size and inference times, has facilitated the use of these models as agents, enabling interactions among multiple models to execute complex tasks. Such collaborations offer several advantages, including the use of specialized models (e.g. coding), improved confidence through multiple computations, and enhanced divergent thinking, leading to more diverse outputs. Thus, the collaborative use of language models is expected to grow significantly in the coming years. In this work, we evaluate the behavior of a network of models collaborating through debate under the influence of an adversary. We introduce pertinent metrics to assess the adversary's effectiveness, focusing on system accuracy and model agreement. Our findings highlight the importance of a model's persuasive ability in influencing others. Additionally, we explore inference-time methods to generate more compelling arguments and evaluate the potential of prompt-based mitigation as a defensive strategy.

摘要: 大型语言模型(LLM)在单独工作时，在当前基准上显示了特殊的结果。它们能力的进步，加上参数大小和推理时间的减少，促进了这些模型作为代理的使用，使多个模型之间能够相互作用，以执行复杂的任务。这种协作提供了几个优势，包括使用专门的模型(例如编码)、通过多次计算提高信心以及增强发散思维，从而产生更多样化的产出。因此，语言模型的协作使用预计在未来几年将显著增长。在这项工作中，我们评估了一个模型网络在对手的影响下通过辩论进行合作的行为。我们引入了相关的度量来评估对手的有效性，重点是系统的准确性和模型的一致性。我们的发现突显了模特的说服力在影响他人方面的重要性。此外，我们探索推理时间方法来生成更令人信服的论点，并评估基于即时缓解作为一种防御策略的潜力。



## **38. Detecting Brittle Decisions for Free: Leveraging Margin Consistency in Deep Robust Classifiers**

免费检测脆弱决策：利用深度稳健分类器中的保证金一致性 cs.LG

11 pages, 7 figures, 2 tables, 1 algorithm

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18451v1) [paper-pdf](http://arxiv.org/pdf/2406.18451v1)

**Authors**: Jonas Ngnawé, Sabyasachi Sahoo, Yann Pequignot, Frédéric Precioso, Christian Gagné

**Abstract**: Despite extensive research on adversarial training strategies to improve robustness, the decisions of even the most robust deep learning models can still be quite sensitive to imperceptible perturbations, creating serious risks when deploying them for high-stakes real-world applications. While detecting such cases may be critical, evaluating a model's vulnerability at a per-instance level using adversarial attacks is computationally too intensive and unsuitable for real-time deployment scenarios. The input space margin is the exact score to detect non-robust samples and is intractable for deep neural networks. This paper introduces the concept of margin consistency -- a property that links the input space margins and the logit margins in robust models -- for efficient detection of vulnerable samples. First, we establish that margin consistency is a necessary and sufficient condition to use a model's logit margin as a score for identifying non-robust samples. Next, through comprehensive empirical analysis of various robustly trained models on CIFAR10 and CIFAR100 datasets, we show that they indicate strong margin consistency with a strong correlation between their input space margins and the logit margins. Then, we show that we can effectively use the logit margin to confidently detect brittle decisions with such models and accurately estimate robust accuracy on an arbitrarily large test set by estimating the input margins only on a small subset. Finally, we address cases where the model is not sufficiently margin-consistent by learning a pseudo-margin from the feature representation. Our findings highlight the potential of leveraging deep representations to efficiently assess adversarial vulnerability in deployment scenarios.

摘要: 尽管对对抗性训练策略进行了大量研究以提高稳健性，但即使是最健壮的深度学习模型的决策也可能对不可察觉的扰动非常敏感，当将它们部署到高风险的现实世界应用程序时，会产生严重的风险。虽然检测这类情况可能很关键，但使用对抗性攻击在每个实例级别评估模型的漏洞计算量太大，不适合实时部署场景。输入空间裕度是检测非稳健样本的准确分数，对于深度神经网络来说是很难处理的。为了有效地检测易受攻击的样本，本文引入了边缘一致性的概念--一种将输入空间边缘和健壮模型中的Logit边缘联系起来的属性。首先，我们证明了边际一致性是使用模型的Logit边际作为识别非稳健样本的分数的充要条件。接下来，通过对CIFAR10和CIFAR100数据集上各种稳健训练模型的综合实证分析，我们发现它们表明了很强的边际一致性，并且它们的输入空间边际和Logit边际之间存在很强的相关性。然后，我们证明了我们可以有效地使用Logit裕度来自信地检测此类模型的脆性决策，并通过仅在较小的子集上估计输入裕度来准确地估计任意大测试集上的稳健精度。最后，我们通过从特征表示学习伪边距来处理模型不够边距一致的情况。我们的发现突出了利用深度陈述来有效评估部署场景中的对手脆弱性的潜力。



## **39. Are AI-Generated Text Detectors Robust to Adversarial Perturbations?**

人工智能生成的文本检测器对对抗性扰动是否稳健？ cs.CL

Accepted to ACL 2024 main conference

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.01179v2) [paper-pdf](http://arxiv.org/pdf/2406.01179v2)

**Authors**: Guanhua Huang, Yuchen Zhang, Zhe Li, Yongjian You, Mingze Wang, Zhouwang Yang

**Abstract**: The widespread use of large language models (LLMs) has sparked concerns about the potential misuse of AI-generated text, as these models can produce content that closely resembles human-generated text. Current detectors for AI-generated text (AIGT) lack robustness against adversarial perturbations, with even minor changes in characters or words causing a reversal in distinguishing between human-created and AI-generated text. This paper investigates the robustness of existing AIGT detection methods and introduces a novel detector, the Siamese Calibrated Reconstruction Network (SCRN). The SCRN employs a reconstruction network to add and remove noise from text, extracting a semantic representation that is robust to local perturbations. We also propose a siamese calibration technique to train the model to make equally confidence predictions under different noise, which improves the model's robustness against adversarial perturbations. Experiments on four publicly available datasets show that the SCRN outperforms all baseline methods, achieving 6.5\%-18.25\% absolute accuracy improvement over the best baseline method under adversarial attacks. Moreover, it exhibits superior generalizability in cross-domain, cross-genre, and mixed-source scenarios. The code is available at \url{https://github.com/CarlanLark/Robust-AIGC-Detector}.

摘要: 大型语言模型(LLM)的广泛使用引发了人们对人工智能生成的文本可能被滥用的担忧，因为这些模型可以生成与人类生成的文本非常相似的内容。目前的人工智能生成文本检测器(AIGT)缺乏对对手扰动的稳健性，即使是字符或单词的微小变化也会导致在区分人工生成文本和人工智能生成文本方面出现逆转。本文研究了现有的AIGT检测方法的稳健性，并介绍了一种新的检测器--暹罗校准重建网络(SCRN)。SCRN使用重构网络来添加和去除文本中的噪声，提取对局部扰动具有鲁棒性的语义表示。我们还提出了一种暹罗校正技术来训练模型，使其在不同的噪声下做出相同的置信度预测，从而提高了模型对对抗性扰动的鲁棒性。在四个公开可用的数据集上的实验表明，SCRN的性能优于所有的基线方法，在对抗性攻击下，其绝对准确率比最佳基线方法提高了6.5-18.25。此外，它在跨域、跨流派和混合来源的场景中表现出出色的泛化能力。代码可在\url{https://github.com/CarlanLark/Robust-AIGC-Detector}.上获得



## **40. SUB-PLAY: Adversarial Policies against Partially Observed Multi-Agent Reinforcement Learning Systems**

SUB-SYS：针对部分观察的多智能体强化学习系统的对抗策略 cs.LG

To appear in the ACM Conference on Computer and Communications  Security (CCS'24), October 14-18, 2024, Salt Lake City, UT, USA

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2402.03741v3) [paper-pdf](http://arxiv.org/pdf/2402.03741v3)

**Authors**: Oubo Ma, Yuwen Pu, Linkang Du, Yang Dai, Ruo Wang, Xiaolei Liu, Yingcai Wu, Shouling Ji

**Abstract**: Recent advancements in multi-agent reinforcement learning (MARL) have opened up vast application prospects, such as swarm control of drones, collaborative manipulation by robotic arms, and multi-target encirclement. However, potential security threats during the MARL deployment need more attention and thorough investigation. Recent research reveals that attackers can rapidly exploit the victim's vulnerabilities, generating adversarial policies that result in the failure of specific tasks. For instance, reducing the winning rate of a superhuman-level Go AI to around 20%. Existing studies predominantly focus on two-player competitive environments, assuming attackers possess complete global state observation.   In this study, we unveil, for the first time, the capability of attackers to generate adversarial policies even when restricted to partial observations of the victims in multi-agent competitive environments. Specifically, we propose a novel black-box attack (SUB-PLAY) that incorporates the concept of constructing multiple subgames to mitigate the impact of partial observability and suggests sharing transitions among subpolicies to improve attackers' exploitative ability. Extensive evaluations demonstrate the effectiveness of SUB-PLAY under three typical partial observability limitations. Visualization results indicate that adversarial policies induce significantly different activations of the victims' policy networks. Furthermore, we evaluate three potential defenses aimed at exploring ways to mitigate security threats posed by adversarial policies, providing constructive recommendations for deploying MARL in competitive environments.

摘要: 多智能体强化学习(MAIL)的最新进展为无人机群体控制、机械臂协同操纵、多目标包围等开辟了广阔的应用前景。然而，MAIL部署过程中的潜在安全威胁需要更多的关注和彻底的调查。最近的研究表明，攻击者可以迅速利用受害者的漏洞，生成导致特定任务失败的对抗性策略。例如，将超人级别围棋人工智能的胜率降低到20%左右。现有的研究主要集中在两人竞争环境中，假设攻击者拥有完整的全局状态观测。在这项研究中，我们首次揭示了攻击者即使限于在多智能体竞争环境中对受害者的部分观察也能够生成对抗策略的能力。具体地说，我们提出了一种新的黑盒攻击(子游戏)，它结合了构造多个子博弈的概念来减轻部分可观测性的影响，并建议在子策略之间共享转移以提高攻击者的利用能力。广泛的评估证明了子游戏在三个典型的部分可观测性限制下的有效性。可视化结果表明，对抗性政策导致受害者的政策网络激活显著不同。此外，我们评估了三种潜在的防御措施，旨在探索减轻对抗性政策构成的安全威胁的方法，为在竞争环境中部署Marl提供建设性的建议。



## **41. Artificial Immune System of Secure Face Recognition Against Adversarial Attacks**

对抗攻击的安全人脸识别人工免疫系统 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18144v1) [paper-pdf](http://arxiv.org/pdf/2406.18144v1)

**Authors**: Min Ren, Yunlong Wang, Yuhao Zhu, Yongzhen Huang, Zhenan Sun, Qi Li, Tieniu Tan

**Abstract**: Insect production for food and feed presents a promising supplement to ensure food safety and address the adverse impacts of agriculture on climate and environment in the future. However, optimisation is required for insect production to realise its full potential. This can be by targeted improvement of traits of interest through selective breeding, an approach which has so far been underexplored and underutilised in insect farming. Here we present a comprehensive review of the selective breeding framework in the context of insect production. We systematically evaluate adjustments of selective breeding techniques to the realm of insects and highlight the essential components integral to the breeding process. The discussion covers every step of a conventional breeding scheme, such as formulation of breeding objectives, phenotyping, estimation of genetic parameters and breeding values, selection of appropriate breeding strategies, and mitigation of issues associated with genetic diversity depletion and inbreeding. This review combines knowledge from diverse disciplines, bridging the gap between animal breeding, quantitative genetics, evolutionary biology, and entomology, offering an integrated view of the insect breeding research area and uniting knowledge which has previously remained scattered across diverse fields of expertise.

摘要: 用于食品和饲料的昆虫生产为确保食品安全和解决未来农业对气候和环境的不利影响提供了一种有希望的补充。然而，昆虫生产需要优化才能充分发挥其潜力。这可以通过选择性育种对感兴趣的特征进行有针对性的改进，这种方法到目前为止在昆虫养殖业中还没有得到充分的探索和利用。在此，我们对昆虫生产中的选择性育种框架进行了全面的回顾。我们系统地评估了选择性育种技术对昆虫领域的调整，并强调了育种过程中不可或缺的基本组成部分。讨论涵盖传统育种计划的每一个步骤，如制定育种目标、表型鉴定、估计遗传参数和育种价值、选择适当的育种策略以及缓解与遗传多样性枯竭和近亲繁殖有关的问题。这篇综述结合了不同学科的知识，弥合了动物育种、数量遗传学、进化生物学和昆虫学之间的差距，提供了昆虫育种研究领域的综合视角，并统一了以前分散在不同专业领域的知识。



## **42. Breaking the Barrier: Enhanced Utility and Robustness in Smoothed DRL Agents**

打破障碍：平滑DRL代理的增强实用性和稳健性 cs.LG

Published in ICML 2024

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2406.18062v1) [paper-pdf](http://arxiv.org/pdf/2406.18062v1)

**Authors**: Chung-En Sun, Sicun Gao, Tsui-Wei Weng

**Abstract**: Robustness remains a paramount concern in deep reinforcement learning (DRL), with randomized smoothing emerging as a key technique for enhancing this attribute. However, a notable gap exists in the performance of current smoothed DRL agents, often characterized by significantly low clean rewards and weak robustness. In response to this challenge, our study introduces innovative algorithms aimed at training effective smoothed robust DRL agents. We propose S-DQN and S-PPO, novel approaches that demonstrate remarkable improvements in clean rewards, empirical robustness, and robustness guarantee across standard RL benchmarks. Notably, our S-DQN and S-PPO agents not only significantly outperform existing smoothed agents by an average factor of $2.16\times$ under the strongest attack, but also surpass previous robustly-trained agents by an average factor of $2.13\times$. This represents a significant leap forward in the field. Furthermore, we introduce Smoothed Attack, which is $1.89\times$ more effective in decreasing the rewards of smoothed agents than existing adversarial attacks.

摘要: 稳健性仍然是深度强化学习(DRL)中最重要的问题，随机化平滑成为增强这一属性的关键技术。然而，当前平滑的DRL代理的性能存在着显著的差距，通常具有明显低的清洁回报和较弱的稳健性。为了应对这一挑战，我们的研究引入了创新的算法，旨在训练有效的平滑稳健的DRL代理。我们提出了S-DQN和S-PPO这两种新颖的方法，它们在干净的回报、经验稳健性和跨标准RL基准的稳健性保证方面都有显著的改善。值得注意的是，我们的S-DQN和S-PPO代理不仅在最强攻击下的平均性能比现有的平滑代理高出2.16倍，而且比以前训练有素的代理高出2.13倍。这代表着该领域的一次重大飞跃。此外，我们引入了平滑攻击，它比现有的对抗性攻击在减少平滑代理的报酬方面更有效，其效率为1.89倍。



## **43. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

DirectTA：针对大型视觉语言模型的指令调整有针对性的攻击 cs.CV

**SubmitDate**: 2024-06-26    [abs](http://arxiv.org/abs/2312.01886v3) [paper-pdf](http://arxiv.org/pdf/2312.01886v3)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical targeted attack scenario that the adversary can only know the vision encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed \textsc{InstructTA}) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same vision encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability with instruction tuning, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from GPT-4. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability. The code is available at https://github.com/xunguangwang/InstructTA.

摘要: 大型视觉语言模型在图像理解和响应生成方面表现出了令人难以置信的能力。然而，这种丰富的视觉交互也使LVLM容易受到对抗性例子的攻击。在该文中，我们提出了一种新颖而实用的定向攻击场景，攻击者只能知道受害者LVLM的视觉编码器，而不知道其提示(通常是服务提供商的专有提示，而不是公开可用的)及其底层的大语言模型(LLM)。这一实际设置对目标对抗性攻击的跨提示和跨模型可转移性提出了挑战，其目的是混淆LVLM以输出与攻击者选择的目标文本在语义上相似的响应。为此，我们提出了一种指令调谐的定向攻击(称为\Textsc{InstructTA})，以提供对具有高可转移性的LVLM的定向对抗性攻击。首先，我们利用一个公开的文本到图像的生成模型将目标响应“反转”成目标图像，并使用GPT-4从目标响应中推断出合理的指令符号。然后，我们形成一个局部代理模型(与受害者LVLM共享相同的视觉编码器)来提取对抗性图像示例和目标图像的指令感知特征，并最小化这两个特征之间的距离以优化对抗性示例。为了进一步提高指令调优的可转移性，我们用GPT-4中转译的指令扩充了指令$\boldsign{p}^\Prime$。大量实验证明了该方法在目标攻击性能和可转移性方面的优越性。代码可在https://github.com/xunguangwang/InstructTA.上获得



## **44. Diffusion-based Adversarial Purification for Intrusion Detection**

基于扩散的入侵检测对抗净化 cs.CR

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17606v1) [paper-pdf](http://arxiv.org/pdf/2406.17606v1)

**Authors**: Mohamed Amine Merzouk, Erwan Beurier, Reda Yaich, Nora Boulahia-Cuppens, Frédéric Cuppens

**Abstract**: The escalating sophistication of cyberattacks has encouraged the integration of machine learning techniques in intrusion detection systems, but the rise of adversarial examples presents a significant challenge. These crafted perturbations mislead ML models, enabling attackers to evade detection or trigger false alerts. As a reaction, adversarial purification has emerged as a compelling solution, particularly with diffusion models showing promising results. However, their purification potential remains unexplored in the context of intrusion detection. This paper demonstrates the effectiveness of diffusion models in purifying adversarial examples in network intrusion detection. Through a comprehensive analysis of the diffusion parameters, we identify optimal configurations maximizing adversarial robustness with minimal impact on normal performance. Importantly, this study reveals insights into the relationship between diffusion noise and diffusion steps, representing a novel contribution to the field. Our experiments are carried out on two datasets and against 5 adversarial attacks. The implementation code is publicly available.

摘要: 网络攻击的日益复杂鼓励了将机器学习技术整合到入侵检测系统中，但敌意例子的兴起构成了一个重大挑战。这些精心设计的扰动误导了ML模型，使攻击者能够逃避检测或触发错误警报。作为一种反应，对抗性净化已成为一种引人注目的解决方案，特别是在扩散模型显示了有希望的结果的情况下。然而，在入侵检测的背景下，它们的净化潜力仍然没有被发掘。本文论证了扩散模型在网络入侵检测中净化恶意实例的有效性。通过对扩散参数的综合分析，我们确定了在对正常性能影响最小的情况下最大化对手健壮性的最优配置。重要的是，这项研究揭示了扩散噪声和扩散步骤之间的关系，代表了对该领域的新贡献。我们的实验是在两个数据集上进行的，并针对5个对手攻击进行了测试。实现代码是公开的。



## **45. Treatment of Statistical Estimation Problems in Randomized Smoothing for Adversarial Robustness**

对抗鲁棒性随机平滑中统计估计问题的处理 stat.ML

comments are welcome

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17830v1) [paper-pdf](http://arxiv.org/pdf/2406.17830v1)

**Authors**: Vaclav Voracek

**Abstract**: Randomized smoothing is a popular certified defense against adversarial attacks. In its essence, we need to solve a problem of statistical estimation which is usually very time-consuming since we need to perform numerous (usually $10^5$) forward passes of the classifier for every point to be certified. In this paper, we review the statistical estimation problems for randomized smoothing to find out if the computational burden is necessary. In particular, we consider the (standard) task of adversarial robustness where we need to decide if a point is robust at a certain radius or not using as few samples as possible while maintaining statistical guarantees. We present estimation procedures employing confidence sequences enjoying the same statistical guarantees as the standard methods, with the optimal sample complexities for the estimation task and empirically demonstrate their good performance. Additionally, we provide a randomized version of Clopper-Pearson confidence intervals resulting in strictly stronger certificates.

摘要: 随机平滑是一种流行的对抗对手攻击的认证防御方法。本质上，我们需要解决通常非常耗时的统计估计问题，因为我们需要为要认证的每个点执行许多(通常是$10^5$)分类器的前向传递。在本文中，我们回顾了随机平滑的统计估计问题，以确定是否有必要增加计算负担。特别是，我们考虑了对抗健壮性的(标准)任务，其中我们需要确定一个点在特定半径处是否健壮，或者在保持统计保证的同时不使用尽可能少的样本。我们提出了使用具有与标准方法相同的统计保证的置信度序列的估计方法，对于估计任务具有最优的样本复杂性，并通过经验证明了其良好的性能。此外，我们还提供了Clopper-Pearson可信区间的随机化版本，从而产生严格更强的证书。



## **46. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

合成人脸图像检测：准确性、鲁棒性、概括性 cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17547v1) [paper-pdf](http://arxiv.org/pdf/2406.17547v1)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.

摘要: 进行了合成人脸图像检测的实验研究。我们收集了一个名为FF 5的数据集，包含五个假面部图像生成器，包括最近的扩散模型。我们发现，在特定图像生成器上训练的简单模型可以在分离合成图像和真实图像方面实现近乎完美的准确性。该模型通过使用数据增强来处理常见的图像失真（分辨率降低、压缩）。此外，还可以识别部分操纵（通过修补将合成图像混合到真实图像中），并通过YOLO架构的简单模型来本地化操纵区域。然而，事实证明，该模型很容易受到对抗攻击，并且不能推广到看不见的生成器。最近的最先进方法也会出现无法概括检测由较新生成器产生的图像的情况，我们在Realistic Vision上进行了测试，Realistic Vision是StabilityAI的Stable Dispatch图像生成器的微调版本。



## **47. Practical Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration**

通过自我提示校准对微调大型语言模型的实用成员推断攻击 cs.CL

Repo: https://github.com/wjfu99/MIA-LLMs

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2311.06062v3) [paper-pdf](http://arxiv.org/pdf/2311.06062v3)

**Authors**: Wenjie Fu, Huandong Wang, Chen Gao, Guanghua Liu, Yong Li, Tao Jiang

**Abstract**: Membership Inference Attacks (MIA) aim to infer whether a target data record has been utilized for model training or not. Prior attempts have quantified the privacy risks of language models (LMs) via MIAs, but there is still no consensus on whether existing MIA algorithms can cause remarkable privacy leakage on practical Large Language Models (LLMs). Existing MIAs designed for LMs can be classified into two categories: reference-free and reference-based attacks. They are both based on the hypothesis that training records consistently strike a higher probability of being sampled. Nevertheless, this hypothesis heavily relies on the overfitting of target models, which will be mitigated by multiple regularization methods and the generalization of LLMs. The reference-based attack seems to achieve promising effectiveness in LLMs, which measures a more reliable membership signal by comparing the probability discrepancy between the target model and the reference model. However, the performance of reference-based attack is highly dependent on a reference dataset that closely resembles the training dataset, which is usually inaccessible in the practical scenario. Overall, existing MIAs are unable to effectively unveil privacy leakage over practical fine-tuned LLMs that are overfitting-free and private. We propose a Membership Inference Attack based on Self-calibrated Probabilistic Variation (SPV-MIA). Specifically, since memorization in LLMs is inevitable during the training process and occurs before overfitting, we introduce a more reliable membership signal, probabilistic variation, which is based on memorization rather than overfitting. Furthermore, we introduce a self-prompt approach, which constructs the dataset to fine-tune the reference model by prompting the target LLM itself. In this manner, the adversary can collect a dataset with a similar distribution from public APIs.

摘要: 成员关系推理攻击(MIA)的目的是推断目标数据记录是否已被用于模型训练。以往的研究已经通过MIA量化了语言模型的隐私风险，但对于现有的MIA算法是否会在实际的大型语言模型上造成显著的隐私泄漏，目前还没有达成共识。现有的针对LMS设计的MIA可以分为两类：无引用攻击和基于引用攻击。它们都是基于这样的假设，即培训记录始终具有更高的被抽样概率。然而，这一假设在很大程度上依赖于目标模型的过度拟合，而多种正则化方法和LLMS的推广将缓解这一问题。基于参考的攻击在LLMS中似乎取得了很好的效果，它通过比较目标模型和参考模型之间的概率差异来衡量更可靠的成员信号。然而，基于参考的攻击的性能高度依赖于与训练数据集非常相似的参考数据集，这在实际场景中通常是不可访问的。总体而言，现有的MIA无法有效地揭示实用的微调LLM的隐私泄露，这些LLM是免装修和私密的。提出了一种基于自校准概率变异的成员推理攻击(SPV-MIA)。具体地说，由于LLMS中的记忆在训练过程中是不可避免的，并且发生在过适应之前，因此我们引入了一种更可靠的隶属度信号-概率变异，它基于记忆而不是过适应。此外，我们引入了一种自我提示的方法，该方法构建数据集，通过提示目标LLM本身来微调参考模型。通过这种方式，攻击者可以从公共API收集具有类似分布的数据集。



## **48. TSynD: Targeted Synthetic Data Generation for Enhanced Medical Image Classification**

TSynD：用于增强医学图像分类的有针对性的合成数据生成 cs.CV

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17473v1) [paper-pdf](http://arxiv.org/pdf/2406.17473v1)

**Authors**: Joshua Niemeijer, Jan Ehrhardt, Hristina Uzunova, Heinz Handels

**Abstract**: The usage of medical image data for the training of large-scale machine learning approaches is particularly challenging due to its scarce availability and the costly generation of data annotations, typically requiring the engagement of medical professionals. The rapid development of generative models allows towards tackling this problem by leveraging large amounts of realistic synthetically generated data for the training process. However, randomly choosing synthetic samples, might not be an optimal strategy.   In this work, we investigate the targeted generation of synthetic training data, in order to improve the accuracy and robustness of image classification. Therefore, our approach aims to guide the generative model to synthesize data with high epistemic uncertainty, since large measures of epistemic uncertainty indicate underrepresented data points in the training set. During the image generation we feed images reconstructed by an auto encoder into the classifier and compute the mutual information over the class-probability distribution as a measure for uncertainty.We alter the feature space of the autoencoder through an optimization process with the objective of maximizing the classifier uncertainty on the decoded image. By training on such data we improve the performance and robustness against test time data augmentations and adversarial attacks on several classifications tasks.

摘要: 将医学图像数据用于大规模机器学习方法的培训尤其具有挑战性，因为它的可用性很少，而且生成数据注释的成本很高，通常需要医疗专业人员参与。生成性模型的快速发展允许通过利用大量真实的综合生成的数据来解决这一问题。然而，随机选择合成样本，可能不是最优策略。为了提高图像分类的准确性和稳健性，本文对合成训练数据的定向生成进行了研究。因此，我们的方法旨在指导生成模型合成具有高认知不确定性的数据，因为认知不确定性的大量测量表明训练集中的数据点代表不足。在图像生成过程中，我们将由自动编码器重建的图像送入分类器，计算类别概率分布上的互信息作为不确定性的度量，并通过优化过程改变自动编码器的特征空间，以最大化解码图像上的分类器不确定性为目标。通过对这些数据的训练，我们提高了对测试时间数据增加和对几个分类任务的敌意攻击的性能和稳健性。



## **49. Low-Cost Privacy-Aware Decentralized Learning**

低成本隐私意识的去中心化学习 cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2403.11795v2) [paper-pdf](http://arxiv.org/pdf/2403.11795v2)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: This paper introduces ZIP-DL, a novel privacy-aware decentralized learning (DL) algorithm that exploits correlated noise to provide strong privacy protection against a local adversary while yielding efficient convergence guarantees for a low communication cost. The progressive neutralization of the added noise during the distributed aggregation process results in ZIP-DL fostering a high model accuracy under privacy guarantees. ZIP-DL further uses a single communication round between each gradient descent, thus minimizing communication overhead. We provide theoretical guarantees for both convergence speed and privacy guarantees, thereby making ZIP-DL applicable to practical scenarios. Our extensive experimental study shows that ZIP-DL significantly outperforms the state-of-the-art in terms of vulnerability/accuracy trade-off. In particular, ZIP-DL (i) reduces the efficacy of linkability attacks by up to 52 percentage points compared to baseline DL, (ii) improves accuracy by up to 37 percent w.r.t. the state-of-the-art privacy-preserving mechanism operating under the same threat model as ours, when configured to provide the same protection against membership inference attacks, and (iii) reduces communication by up to 10.5x against the same competitor for the same level of protection.

摘要: 介绍了一种新的隐私感知分散学习算法ZIP-DL，该算法利用相关噪声来提供对本地攻击者的强隐私保护，同时以较低的通信代价产生高效的收敛保证。分布式聚合过程中添加的噪声的渐进式中和导致ZIP-DL在隐私保证下培养高模型精度。Zip-DL还在每个梯度下降之间使用单个通信轮次，从而将通信开销降至最低。我们为收敛速度和隐私保证提供了理论上的保证，从而使ZIP-DL适用于实际场景。我们广泛的实验研究表明，ZIP-DL在脆弱性和准确性之间的权衡方面显著优于最先进的ZIP-DL。特别是，与基准DL相比，ZIP-DL(I)将链接性攻击的有效性降低了高达52个百分点，(Ii)将准确率提高了高达37%。最先进的隐私保护机制在与我们相同的威胁模型下运行，当配置为提供相同的成员身份推理攻击保护时，并且(Iii)在相同保护级别的情况下，针对相同竞争对手的通信最多减少10.5倍。



## **50. CuDA2: An approach for Incorporating Traitor Agents into Cooperative Multi-Agent Systems**

CuDA 2：一种将叛徒代理融入合作多代理系统的方法 cs.LG

**SubmitDate**: 2024-06-25    [abs](http://arxiv.org/abs/2406.17425v1) [paper-pdf](http://arxiv.org/pdf/2406.17425v1)

**Authors**: Zhen Chen, Yong Liao, Youpeng Zhao, Zipeng Dai, Jian Zhao

**Abstract**: Cooperative Multi-Agent Reinforcement Learning (CMARL) strategies are well known to be vulnerable to adversarial perturbations. Previous works on adversarial attacks have primarily focused on white-box attacks that directly perturb the states or actions of victim agents, often in scenarios with a limited number of attacks. However, gaining complete access to victim agents in real-world environments is exceedingly difficult. To create more realistic adversarial attacks, we introduce a novel method that involves injecting traitor agents into the CMARL system. We model this problem as a Traitor Markov Decision Process (TMDP), where traitors cannot directly attack the victim agents but can influence their formation or positioning through collisions. In TMDP, traitors are trained using the same MARL algorithm as the victim agents, with their reward function set as the negative of the victim agents' reward. Despite this, the training efficiency for traitors remains low because it is challenging for them to directly associate their actions with the victim agents' rewards. To address this issue, we propose the Curiosity-Driven Adversarial Attack (CuDA2) framework. CuDA2 enhances the efficiency and aggressiveness of attacks on the specified victim agents' policies while maintaining the optimal policy invariance of the traitors. Specifically, we employ a pre-trained Random Network Distillation (RND) module, where the extra reward generated by the RND module encourages traitors to explore states unencountered by the victim agents. Extensive experiments on various scenarios from SMAC demonstrate that our CuDA2 framework offers comparable or superior adversarial attack capabilities compared to other baselines.

摘要: 众所周知，协作多智能体强化学习(CMARL)策略容易受到对手扰动的影响。以前关于对抗性攻击的研究主要集中在白盒攻击上，白盒攻击直接扰乱受害者代理的状态或行动，通常在攻击次数有限的情况下。然而，在现实环境中完全接触受害者代理是极其困难的。为了创建更真实的对抗性攻击，我们引入了一种新的方法，涉及到向CMARL系统中注入叛徒代理。我们将这个问题建模为叛徒马尔可夫决策过程(TMDP)，其中叛徒不能直接攻击受害者代理，但可以通过碰撞影响他们的队形或定位。在TMDP中，叛逆者使用与受害者代理相同的Marl算法进行训练，其奖励函数设置为受害者代理奖励的负值。尽管如此，叛徒的培训效率仍然很低，因为他们很难将自己的行动与受害者特工的奖励直接联系起来。为了解决这个问题，我们提出了好奇心驱动的对抗性攻击(CuDA2)框架。CuDA2在保持叛徒最优策略不变性的同时，提高了对指定受害代理策略的攻击效率和攻击性。具体地说，我们采用了一个预先训练的随机网络蒸馏(RND)模块，其中RND模块产生的额外奖励鼓励叛徒探索受害者代理未遇到的状态。来自SMAC的各种场景的广泛实验表明，与其他基线相比，我们的CuDA2框架提供了类似或更好的对抗性攻击能力。



