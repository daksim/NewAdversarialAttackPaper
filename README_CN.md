# Latest Adversarial Attack Papers
**update at 2023-06-14 20:40:52**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Class Attribute Inference Attacks: Inferring Sensitive Class Information by Diffusion-Based Attribute Manipulations**

类别属性推断攻击：通过基于扩散的属性操作推断敏感类别信息 cs.LG

46 pages, 37 figures, 5 tables

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2303.09289v2) [paper-pdf](http://arxiv.org/pdf/2303.09289v2)

**Authors**: Lukas Struppek, Dominik Hintersdorf, Felix Friedrich, Manuel Brack, Patrick Schramowski, Kristian Kersting

**Abstract**: Neural network-based image classifiers are powerful tools for computer vision tasks, but they inadvertently reveal sensitive attribute information about their classes, raising concerns about their privacy. To investigate this privacy leakage, we introduce the first Class Attribute Inference Attack (CAIA), which leverages recent advances in text-to-image synthesis to infer sensitive attributes of individual classes in a black-box setting, while remaining competitive with related white-box attacks. Our extensive experiments in the face recognition domain show that CAIA can accurately infer undisclosed sensitive attributes, such as an individual's hair color, gender, and racial appearance, which are not part of the training labels. Interestingly, we demonstrate that adversarial robust models are even more vulnerable to such privacy leakage than standard models, indicating that a trade-off between robustness and privacy exists.

摘要: 基于神经网络的图像分类器是计算机视觉任务的强大工具，但它们无意中泄露了有关其类别的敏感属性信息，引发了对其隐私的担忧。为了调查这种隐私泄露，我们引入了第一类属性推理攻击(CAIA)，它利用文本到图像合成的最新进展来推断黑盒环境中个别类的敏感属性，同时保持与相关白盒攻击的竞争力。我们在人脸识别领域的广泛实验表明，CAIA可以准确地推断出未披露的敏感属性，如个人的头发颜色、性别和种族外观，这些属性不属于训练标签的一部分。有趣的是，我们证明了对抗性稳健模型比标准模型更容易受到这种隐私泄露的影响，这表明存在稳健性和隐私之间的权衡。



## **2. Finite Gaussian Neurons: Defending against adversarial attacks by making neural networks say "I don't know"**

有限高斯神经元：通过让神经网络说出“我不知道”来防御敌意攻击 cs.LG

PhD thesis

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07796v1) [paper-pdf](http://arxiv.org/pdf/2306.07796v1)

**Authors**: Felix Grezes

**Abstract**: Since 2014, artificial neural networks have been known to be vulnerable to adversarial attacks, which can fool the network into producing wrong or nonsensical outputs by making humanly imperceptible alterations to inputs. While defenses against adversarial attacks have been proposed, they usually involve retraining a new neural network from scratch, a costly task. In this work, I introduce the Finite Gaussian Neuron (FGN), a novel neuron architecture for artificial neural networks. My works aims to: - easily convert existing models to Finite Gaussian Neuron architecture, - while preserving the existing model's behavior on real data, - and offering resistance against adversarial attacks. I show that converted and retrained Finite Gaussian Neural Networks (FGNN) always have lower confidence (i.e., are not overconfident) in their predictions over randomized and Fast Gradient Sign Method adversarial images when compared to classical neural networks, while maintaining high accuracy and confidence over real MNIST images. To further validate the capacity of Finite Gaussian Neurons to protect from adversarial attacks, I compare the behavior of FGNs to that of Bayesian Neural Networks against both randomized and adversarial images, and show how the behavior of the two architectures differs. Finally I show some limitations of the FGN models by testing them on the more complex SPEECHCOMMANDS task, against the stronger Carlini-Wagner and Projected Gradient Descent adversarial attacks.

摘要: 自2014年以来，人工神经网络一直被认为容易受到对抗性攻击，这些攻击可以通过对输入进行人类无法察觉的改变来愚弄网络产生错误或毫无意义的输出。虽然有人提出了防御对手攻击的建议，但它们通常涉及从头开始重新训练新的神经网络，这是一项代价高昂的任务。在这项工作中，我介绍了有限高斯神经元(FGN)，一种新的人工神经网络的神经元结构。我的工作目标是：-轻松地将现有模型转换为有限的高斯神经元架构-同时保留现有模型在真实数据上的行为-并提供对对手攻击的抵抗。结果表明，与经典神经网络相比，经过转换和再训练的有限高斯神经网络(FGNN)在对随机和快速梯度符号法对手图像的预测中总是具有较低的置信度(即不过分自信)，而在真实MNIST图像上保持较高的精度和置信度。为了进一步验证有限高斯神经元抵御敌意攻击的能力，我比较了FGNs和贝叶斯神经网络在随机图像和敌意图像下的行为，并展示了这两种体系结构的行为如何不同。最后，我通过在更复杂的SPEECHCOMMANDS任务中测试FGN模型的一些局限性，对抗更强大的Carlini-Wagner和预测的梯度下降对手攻击。



## **3. Area is all you need: repeatable elements make stronger adversarial attacks**

面积就是你所需要的：可重复的元素构成更强的对抗性攻击 cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07768v1) [paper-pdf](http://arxiv.org/pdf/2306.07768v1)

**Authors**: Dillon Niederhut

**Abstract**: Over the last decade, deep neural networks have achieved state of the art in computer vision tasks. These models, however, are susceptible to unusual inputs, known as adversarial examples, that cause them to misclassify or otherwise fail to detect objects. Here, we provide evidence that the increasing success of adversarial attacks is primarily due to increasing their size. We then demonstrate a method for generating the largest possible adversarial patch by building a adversarial pattern out of repeatable elements. This approach achieves a new state of the art in evading detection by YOLOv2 and YOLOv3. Finally, we present an experiment that fails to replicate the prior success of several attacks published in this field, and end with some comments on testing and reproducibility.

摘要: 在过去的十年里，深度神经网络在计算机视觉任务中达到了最先进的水平。然而，这些模型很容易受到异常输入的影响，这些输入被称为对抗性示例，导致它们错误分类或无法检测到对象。在这里，我们提供的证据表明，对抗性攻击的日益成功主要是由于其规模的增加。然后，我们演示了一种通过从可重复元素中构建对抗性模式来生成可能最大的对抗性补丁的方法。该方法在躲避YOLOv2和YOLOv3的检测方面达到了新的技术水平。最后，我们给出了一个未能复制该领域已发表的几种攻击的先前成功的实验，并以对测试和重复性的一些评论结束。



## **4. PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts**

PromptBitch：评估大型语言模型在对抗性提示下的稳健性 cs.CL

Technical report; 23 pages; code is at:  https://github.com/microsoft/promptbench

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04528v2) [paper-pdf](http://arxiv.org/pdf/2306.04528v2)

**Authors**: Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, Xing Xie

**Abstract**: The increasing reliance on Large Language Models (LLMs) across academia and industry necessitates a comprehensive understanding of their robustness to prompts. In response to this vital need, we introduce PromptBench, a robustness benchmark designed to measure LLMs' resilience to adversarial prompts. This study uses a plethora of adversarial textual attacks targeting prompts across multiple levels: character, word, sentence, and semantic. These prompts are then employed in diverse tasks, such as sentiment analysis, natural language inference, reading comprehension, machine translation, and math problem-solving. Our study generates 4,032 adversarial prompts, meticulously evaluated over 8 tasks and 13 datasets, with 567,084 test samples in total. Our findings demonstrate that contemporary LLMs are vulnerable to adversarial prompts. Furthermore, we present comprehensive analysis to understand the mystery behind prompt robustness and its transferability. We then offer insightful robustness analysis and pragmatic recommendations for prompt composition, beneficial to both researchers and everyday users. We make our code, prompts, and methodologies to generate adversarial prompts publicly accessible, thereby enabling and encouraging collaborative exploration in this pivotal field: https://github.com/microsoft/promptbench.

摘要: 学术界和工业界对大型语言模型(LLM)的依赖日益增加，这就要求我们必须全面了解它们对提示的稳健性。为了响应这一关键需求，我们引入了PromptBtch，这是一个健壮性基准，旨在衡量LLMS对敌意提示的弹性。这项研究使用了过多的对抗性文本攻击，目标是多个层面的提示：字符、单词、句子和语义。这些提示随后被用于不同的任务，如情感分析、自然语言推理、阅读理解、机器翻译和数学解题。我们的研究产生了4032个对抗性提示，仔细评估了8个任务和13个数据集，总共有567,084个测试样本。我们的研究结果表明，当代的LLM容易受到对抗性提示的影响。此外，我们还提供了全面的分析，以了解即时健壮性及其可转移性背后的奥秘。然后，我们提供了有洞察力的健壮性分析和实用的即时撰写建议，这对研究人员和日常用户都是有益的。我们将生成对抗性提示的代码、提示和方法公之于众，从而支持并鼓励在这个关键领域进行协作探索：https://github.com/microsoft/promptbench.



## **5. Malafide: a novel adversarial convolutive noise attack against deepfake and spoofing detection systems**

恶意：一种新的对抗深度假冒和欺骗检测系统的对抗性卷积噪声攻击 eess.AS

Accepted at INTERSPEECH 2023

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07655v1) [paper-pdf](http://arxiv.org/pdf/2306.07655v1)

**Authors**: Michele Panariello, Wanying Ge, Hemlata Tak, Massimiliano Todisco, Nicholas Evans

**Abstract**: We present Malafide, a universal adversarial attack against automatic speaker verification (ASV) spoofing countermeasures (CMs). By introducing convolutional noise using an optimised linear time-invariant filter, Malafide attacks can be used to compromise CM reliability while preserving other speech attributes such as quality and the speaker's voice. In contrast to other adversarial attacks proposed recently, Malafide filters are optimised independently of the input utterance and duration, are tuned instead to the underlying spoofing attack, and require the optimisation of only a small number of filter coefficients. Even so, they degrade CM performance estimates by an order of magnitude, even in black-box settings, and can also be configured to overcome integrated CM and ASV subsystems. Integrated solutions that use self-supervised learning CMs, however, are more robust, under both black-box and white-box settings.

摘要: 我们提出了一种针对自动说话人验证(ASV)欺骗对策(CMS)的通用对抗性攻击--恶意攻击。通过使用优化的线性时不变滤波器引入卷积噪声，恶意攻击可以用来损害CM的可靠性，同时保留其他语音属性，如质量和说话人的声音。与最近提出的其他敌意攻击不同，恶意过滤器独立于输入发音和持续时间进行优化，而是根据潜在的欺骗攻击进行调整，并且只需要优化少量的过滤器系数。即便如此，即使在黑盒设置中，它们也会将CM性能估计降低一个数量级，并且还可以配置为克服集成的CM和ASV子系统。然而，使用自我监督学习CMS的集成解决方案在黑盒和白盒设置下都更健壮。



## **6. A Hypergraph-Based Machine Learning Ensemble Network Intrusion Detection System**

一种基于超图的机器学习集成网络入侵检测系统 cs.CR

This work has been submitted to the IEEE for possible publication.  Copyright may be transferred without notice, after which this version may no  longer be accessible

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2211.03933v2) [paper-pdf](http://arxiv.org/pdf/2211.03933v2)

**Authors**: Zong-Zhi Lin, Thomas D. Pike, Mark M. Bailey, Nathaniel D. Bastian

**Abstract**: Network intrusion detection systems (NIDS) to detect malicious attacks continue to meet challenges. NIDS are often developed offline while they face auto-generated port scan infiltration attempts, resulting in a significant time lag from adversarial adaption to NIDS response. To address these challenges, we use hypergraphs focused on internet protocol addresses and destination ports to capture evolving patterns of port scan attacks. The derived set of hypergraph-based metrics are then used to train an ensemble machine learning (ML) based NIDS that allows for real-time adaption in monitoring and detecting port scanning activities, other types of attacks, and adversarial intrusions at high accuracy, precision and recall performances. This ML adapting NIDS was developed through the combination of (1) intrusion examples, (2) NIDS update rules, (3) attack threshold choices to trigger NIDS retraining requests, and (4) a production environment with no prior knowledge of the nature of network traffic. 40 scenarios were auto-generated to evaluate the ML ensemble NIDS comprising three tree-based models. The resulting ML Ensemble NIDS was extended and evaluated with the CIC-IDS2017 dataset. Results show that under the model settings of an Update-ALL-NIDS rule (specifically retrain and update all the three models upon the same NIDS retraining request) the proposed ML ensemble NIDS evolved intelligently and produced the best results with nearly 100% detection performance throughout the simulation.

摘要: 网络入侵检测系统(NID)检测恶意攻击的能力不断受到挑战。当NID面临自动生成的端口扫描渗透尝试时，它们通常是离线开发的，导致从对手适应到NIDS响应有很大的时间延迟。为了应对这些挑战，我们使用聚焦于互联网协议地址和目标端口的超图来捕获端口扫描攻击的演变模式。然后，使用导出的基于超图的度量集合来训练基于机器学习(ML)的集成网络入侵检测系统，该集成机器学习系统允许实时适应监视和检测端口扫描活动、其他类型的攻击以及高准确度、精确度和召回性能的敌对入侵。通过(1)入侵实例、(2)网络入侵检测系统更新规则、(3)用于触发网络入侵检测系统再训练请求的攻击阈值选择和(4)不事先知道网络流量性质的生产环境的组合，开发了该ML适应网络入侵检测系统。自动生成了40个场景来评估包含三个基于树的模型的ML集成网络入侵检测系统。使用CIC-IDS2017数据集对所得到的ML集成网络入侵检测系统进行了扩展和评估。结果表明，在更新-全部-网络入侵检测系统规则的模型设置下(根据同一网络入侵检测系统的重新训练请求，具体地对三个模型进行重新训练和更新)，所提出的最大似然集成网络入侵检测系统以智能方式进化，并在整个仿真过程中产生最好的结果，检测性能接近100%。



## **7. Extracting Cloud-based Model with Prior Knowledge**

基于先验知识的云模型提取 cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.04192v4) [paper-pdf](http://arxiv.org/pdf/2306.04192v4)

**Authors**: Shiqian Zhao, Kangjie Chen, Meng Hao, Jian Zhang, Guowen Xu, Hongwei Li, Tianwei Zhang

**Abstract**: Machine Learning-as-a-Service, a pay-as-you-go business pattern, is widely accepted by third-party users and developers. However, the open inference APIs may be utilized by malicious customers to conduct model extraction attacks, i.e., attackers can replicate a cloud-based black-box model merely via querying malicious examples. Existing model extraction attacks mainly depend on the posterior knowledge (i.e., predictions of query samples) from Oracle. Thus, they either require high query overhead to simulate the decision boundary, or suffer from generalization errors and overfitting problems due to query budget limitations. To mitigate it, this work proposes an efficient model extraction attack based on prior knowledge for the first time. The insight is that prior knowledge of unlabeled proxy datasets is conducive to the search for the decision boundary (e.g., informative samples). Specifically, we leverage self-supervised learning including autoencoder and contrastive learning to pre-compile the prior knowledge of the proxy dataset into the feature extractor of the substitute model. Then we adopt entropy to measure and sample the most informative examples to query the target model. Our design leverages both prior and posterior knowledge to extract the model and thus eliminates generalizability errors and overfitting problems. We conduct extensive experiments on open APIs like Traffic Recognition, Flower Recognition, Moderation Recognition, and NSFW Recognition from real-world platforms, Azure and Clarifai. The experimental results demonstrate the effectiveness and efficiency of our attack. For example, our attack achieves 95.1% fidelity with merely 1.8K queries (cost 2.16$) on the NSFW Recognition API. Also, the adversarial examples generated with our substitute model have better transferability than others, which reveals that our scheme is more conducive to downstream attacks.

摘要: 机器学习即服务是一种现收现付的商业模式，被第三方用户和开发人员广泛接受。然而，开放推理API可能被恶意客户利用来进行模型提取攻击，即攻击者仅通过查询恶意示例就可以复制基于云的黑盒模型。现有的模型提取攻击主要依赖于Oracle的后验知识(即查询样本的预测)。因此，它们要么需要很高的查询开销来模拟决策边界，要么由于查询预算的限制而存在泛化错误和过适应问题。针对这一问题，本文首次提出了一种基于先验知识的高效模型提取攻击方法。结论是，未标记的代理数据集的先验知识有助于搜索决策边界(例如，信息样本)。具体地说，我们利用包括自动编码器和对比学习在内的自监督学习来将代理数据集的先验知识预编译到替代模型的特征提取器中。然后利用信息熵对最具信息量的实例进行度量和采样，以查询目标模型。我们的设计利用先验和后验知识来提取模型，从而消除了泛化误差和过拟合问题。我们对来自真实平台Azure和Clarifai的流量识别、花卉识别、适度识别和NSFW识别等开放API进行了广泛的实验。实验结果证明了该攻击的有效性和高效性。例如，我们的攻击在NSFW识别API上仅用1.8K个查询(成本2.16美元)就达到了95.1%的保真度。此外，我们的替代模型生成的敌意例子具有更好的可移植性，这表明我们的方案更有利于下游攻击。



## **8. I See Dead People: Gray-Box Adversarial Attack on Image-To-Text Models**

我看到死人：对图像到文本模型的灰箱对抗性攻击 cs.CV

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07591v1) [paper-pdf](http://arxiv.org/pdf/2306.07591v1)

**Authors**: Raz Lapid, Moshe Sipper

**Abstract**: Modern image-to-text systems typically adopt the encoder-decoder framework, which comprises two main components: an image encoder, responsible for extracting image features, and a transformer-based decoder, used for generating captions. Taking inspiration from the analysis of neural networks' robustness against adversarial perturbations, we propose a novel gray-box algorithm for creating adversarial examples in image-to-text models. Unlike image classification tasks that have a finite set of class labels, finding visually similar adversarial examples in an image-to-text task poses greater challenges because the captioning system allows for a virtually infinite space of possible captions. In this paper, we present a gray-box adversarial attack on image-to-text, both untargeted and targeted. We formulate the process of discovering adversarial perturbations as an optimization problem that uses only the image-encoder component, meaning the proposed attack is language-model agnostic. Through experiments conducted on the ViT-GPT2 model, which is the most-used image-to-text model in Hugging Face, and the Flickr30k dataset, we demonstrate that our proposed attack successfully generates visually similar adversarial examples, both with untargeted and targeted captions. Notably, our attack operates in a gray-box manner, requiring no knowledge about the decoder module. We also show that our attacks fool the popular open-source platform Hugging Face.

摘要: 现代图像到文本系统通常采用编解码器框架，该框架包括两个主要组件：负责提取图像特征的图像编码器和用于生成字幕的基于转换器的解码器。从神经网络对对抗性扰动的鲁棒性分析中得到启发，我们提出了一种新的灰盒算法，用于在图像到文本模型中创建对抗性示例。与具有有限类别标签集的图像分类任务不同，在图像到文本的任务中找到视觉上相似的对抗性例子带来了更大的挑战，因为字幕系统允许可能的字幕的几乎无限空间。在本文中，我们提出了一种针对图像到文本的灰盒对抗性攻击，包括无目标攻击和目标攻击。我们将发现敌意扰动的过程描述为一个只使用图像编码器组件的优化问题，这意味着所提出的攻击是语言模型不可知的。通过在拥抱脸中最常用的图文转换模型VIT-GPT2模型和Flickr30k数据集上的实验，我们证明了我们的攻击成功地生成了视觉上相似的对抗性例子，无论是无目标字幕还是有目标字幕。值得注意的是，我们的攻击以灰盒方式运行，不需要了解解码器模块。我们还表明，我们的攻击愚弄了流行的开源平台拥抱脸。



## **9. How Secure is Your Website? A Comprehensive Investigation on CAPTCHA Providers and Solving Services**

您的网站有多安全？验证码提供商和解决方案服务的全面调查 cs.CR

**SubmitDate**: 2023-06-13    [abs](http://arxiv.org/abs/2306.07543v1) [paper-pdf](http://arxiv.org/pdf/2306.07543v1)

**Authors**: Rui Jin, Lin Huang, Jikang Duan, Wei Zhao, Yong Liao, Pengyuan Zhou

**Abstract**: Completely Automated Public Turing Test To Tell Computers and Humans Apart (CAPTCHA) has been implemented on many websites to identify between harmful automated bots and legitimate users. However, the revenue generated by the bots has turned circumventing CAPTCHAs into a lucrative business. Although earlier studies provided information about text-based CAPTCHAs and the associated CAPTCHA-solving services, a lot has changed in the past decade regarding content, suppliers, and solvers of CAPTCHA. We have conducted a comprehensive investigation of the latest third-party CAPTCHA providers and CAPTCHA-solving services' attacks. We dug into the details of CAPTCHA-As-a-Service and the latest CAPTCHA-solving services and carried out adversarial experiments on CAPTCHAs and CAPTCHA solvers. The experiment results show a worrying fact: most latest CAPTCHAs are vulnerable to both human solvers and automated solvers. New CAPTCHAs based on hard AI problems and behavior analysis are needed to stop CAPTCHA solvers.

摘要: 区分计算机和人类的全自动公共图灵测试(CAPTCHA)已经在许多网站上实施，以识别有害的自动机器人和合法用户。然而，机器人产生的收入已经把绕过验证码变成了一项有利可图的业务。尽管早期的研究提供了有关基于文本的验证码和相关验证码解析服务的信息，但在过去十年中，验证码的内容、供应商和解算器发生了很大变化。我们对最新的第三方验证码提供商和验证码解决服务的攻击进行了全面调查。我们深入研究了验证码即服务和最新的验证码解算服务的细节，并对验证码和验证码解算器进行了对抗性实验。实验结果显示了一个令人担忧的事实：大多数最新的验证码都容易受到人工求解器和自动求解器的攻击。需要基于硬AI问题和行为分析的新验证码来停止验证码解算器。



## **10. Adversarial Attacks on the Interpretation of Neuron Activation Maximization**

对抗性攻击对神经元激活最大化的解释 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07397v1) [paper-pdf](http://arxiv.org/pdf/2306.07397v1)

**Authors**: Geraldin Nanfack, Alexander Fulleringer, Jonathan Marty, Michael Eickenberg, Eugene Belilovsky

**Abstract**: The internal functional behavior of trained Deep Neural Networks is notoriously difficult to interpret. Activation-maximization approaches are one set of techniques used to interpret and analyze trained deep-learning models. These consist in finding inputs that maximally activate a given neuron or feature map. These inputs can be selected from a data set or obtained by optimization. However, interpretability methods may be subject to being deceived. In this work, we consider the concept of an adversary manipulating a model for the purpose of deceiving the interpretation. We propose an optimization framework for performing this manipulation and demonstrate a number of ways that popular activation-maximization interpretation techniques associated with CNNs can be manipulated to change the interpretations, shedding light on the reliability of these methods.

摘要: 众所周知，经过训练的深度神经网络的内部功能行为很难解释。激活最大化方法是一套用于解释和分析训练有素的深度学习模型的技术。这包括寻找最大限度地激活给定神经元或特征映射的输入。这些输入可以从数据集中选择或通过优化获得。然而，可解释性方法可能会受到欺骗。在这项工作中，我们考虑了对手操纵模型以欺骗解释的概念。我们提出了一个执行这种操作的优化框架，并展示了与CNN相关的流行的激活最大化解释技术可以被操作以改变解释的一些方法，从而揭示了这些方法的可靠性。



## **11. Gaussian Membership Inference Privacy**

高斯隶属度推理隐私性 cs.LG

The first two authors contributed equally

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07273v1) [paper-pdf](http://arxiv.org/pdf/2306.07273v1)

**Authors**: Tobias Leemann, Martin Pawelczyk, Gjergji Kasneci

**Abstract**: We propose a new privacy notion called $f$-Membership Inference Privacy ($f$-MIP), which explicitly considers the capabilities of realistic adversaries under the membership inference attack threat model. By doing so $f$-MIP offers interpretable privacy guarantees and improved utility (e.g., better classification accuracy). Our novel theoretical analysis of likelihood ratio-based membership inference attacks on noisy stochastic gradient descent (SGD) results in a parametric family of $f$-MIP guarantees that we refer to as $\mu$-Gaussian Membership Inference Privacy ($\mu$-GMIP). Our analysis additionally yields an analytical membership inference attack that offers distinct advantages over previous approaches. First, unlike existing methods, our attack does not require training hundreds of shadow models to approximate the likelihood ratio. Second, our analytical attack enables straightforward auditing of our privacy notion $f$-MIP. Finally, our analysis emphasizes the importance of various factors, such as hyperparameters (e.g., batch size, number of model parameters) and data specific characteristics in controlling an attacker's success in reliably inferring a given point's membership to the training set. We demonstrate the effectiveness of our method on models trained across vision and tabular datasets.

摘要: 我们提出了一种新的隐私概念$f$-MIP($f$-MIP)，它显式地考虑了现实对手在成员关系推理攻击威胁模型下的能力。通过这样做，$f$-MIP提供了可解释的隐私保证和改进的实用性(例如，更好的分类准确性)。我们对噪声随机梯度下降(SGD)上基于似然比的成员推理攻击进行了新的理论分析，得到了一个由$f$-MIP保证组成的参数族，我们称之为$\MU$-高斯成员关系推理隐私($\MU$-GMIP)。我们的分析还产生了一种分析性成员关系推理攻击，与以前的方法相比具有明显的优势。首先，与现有方法不同，我们的攻击不需要训练数百个阴影模型来逼近似然比。其次，我们的分析攻击使我们能够直接审计我们的隐私概念$f$-mip。最后，我们的分析强调了各种因素的重要性，例如超参数(例如，批次大小、模型参数的数量)和数据特定特征，以控制攻击者在可靠地推断给定点的训练集的成员资格方面的成功。我们在视觉和表格数据集上训练的模型上展示了我们方法的有效性。



## **12. When Vision Fails: Text Attacks Against ViT and OCR**

当视觉失败：针对VIT和OCR的文本攻击 cs.CR

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.07033v1) [paper-pdf](http://arxiv.org/pdf/2306.07033v1)

**Authors**: Nicholas Boucher, Jenny Blessing, Ilia Shumailov, Ross Anderson, Nicolas Papernot

**Abstract**: While text-based machine learning models that operate on visual inputs of rendered text have become robust against a wide range of existing attacks, we show that they are still vulnerable to visual adversarial examples encoded as text. We use the Unicode functionality of combining diacritical marks to manipulate encoded text so that small visual perturbations appear when the text is rendered. We show how a genetic algorithm can be used to generate visual adversarial examples in a black-box setting, and conduct a user study to establish that the model-fooling adversarial examples do not affect human comprehension. We demonstrate the effectiveness of these attacks in the real world by creating adversarial examples against production models published by Facebook, Microsoft, IBM, and Google.

摘要: 虽然基于文本的机器学习模型对呈现文本的可视输入已经变得对广泛的现有攻击具有健壮性，但我们表明它们仍然容易受到编码为文本的可视对抗性示例的攻击。我们使用组合变音符号的Unicode功能来操作编码文本，以便在呈现文本时出现小的视觉干扰。我们展示了如何使用遗传算法在黑盒环境中生成可视对抗性示例，并进行了用户研究以确定愚弄模型的对抗性示例不会影响人类的理解。我们通过创建针对Facebook、Microsoft、IBM和Google发布的生产模型的对抗性示例，在现实世界中演示了这些攻击的有效性。



## **13. A Linear Reconstruction Approach for Attribute Inference Attacks against Synthetic Data**

一种针对合成数据的属性推理攻击的线性重构方法 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2301.10053v2) [paper-pdf](http://arxiv.org/pdf/2301.10053v2)

**Authors**: Meenatchi Sundaram Muthu Selva Annamalai, Andrea Gadotti, Luc Rocher

**Abstract**: Recent advances in synthetic data generation (SDG) have been hailed as a solution to the difficult problem of sharing sensitive data while protecting privacy. SDG aims to learn statistical properties of real data in order to generate "artificial" data that are structurally and statistically similar to sensitive data. However, prior research suggests that inference attacks on synthetic data can undermine privacy, but only for specific outlier records. In this work, we introduce a new attribute inference attack against synthetic data. The attack is based on linear reconstruction methods for aggregate statistics, which target all records in the dataset, not only outliers. We evaluate our attack on state-of-the-art SDG algorithms, including Probabilistic Graphical Models, Generative Adversarial Networks, and recent differentially private SDG mechanisms. By defining a formal privacy game, we show that our attack can be highly accurate even on arbitrary records, and that this is the result of individual information leakage (as opposed to population-level inference). We then systematically evaluate the tradeoff between protecting privacy and preserving statistical utility. Our findings suggest that current SDG methods cannot consistently provide sufficient privacy protection against inference attacks while retaining reasonable utility. The best method evaluated, a differentially private SDG mechanism, can provide both protection against inference attacks and reasonable utility, but only in very specific settings. Lastly, we show that releasing a larger number of synthetic records can improve utility but at the cost of making attacks far more effective.

摘要: 合成数据生成(SDG)的最新进展被誉为在保护隐私的同时共享敏感数据这一难题的解决方案。SDG旨在学习真实数据的统计属性，以便生成在结构和统计上与敏感数据相似的“人造”数据。然而，先前的研究表明，对合成数据的推理攻击可能会破坏隐私，但仅限于特定的离群值记录。在这项工作中，我们引入了一种新的针对合成数据的属性推理攻击。该攻击基于聚合统计的线性重建方法，其目标是数据集中的所有记录，而不仅仅是离群值。我们评估了我们对最先进的SDG算法的攻击，包括概率图形模型、生成性对手网络和最近的差异私有SDG机制。通过定义一个正式的隐私游戏，我们证明了我们的攻击即使在任意记录上也可以非常准确，并且这是个人信息泄露的结果(与总体级别的推断相反)。然后，我们系统地评估了保护隐私和保护统计效用之间的权衡。我们的发现表明，现有的SDG方法在保持合理效用的同时，不能始终如一地提供足够的隐私保护来抵御推理攻击。评估的最佳方法是一种不同的私有SDG机制，它可以提供对推理攻击的保护和合理的实用程序，但只能在非常特定的环境中提供。最后，我们表明，发布更多的合成记录可以提高实用性，但代价是使攻击更加有效。



## **14. How robust accuracy suffers from certified training with convex relaxations**

带凸松弛的认证训练对稳健精度的影响 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06995v1) [paper-pdf](http://arxiv.org/pdf/2306.06995v1)

**Authors**: Piersilvio De Bartolomeis, Jacob Clarysse, Amartya Sanyal, Fanny Yang

**Abstract**: Adversarial attacks pose significant threats to deploying state-of-the-art classifiers in safety-critical applications. Two classes of methods have emerged to address this issue: empirical defences and certified defences. Although certified defences come with robustness guarantees, empirical defences such as adversarial training enjoy much higher popularity among practitioners. In this paper, we systematically compare the standard and robust error of these two robust training paradigms across multiple computer vision tasks. We show that in most tasks and for both $\mathscr{l}_\infty$-ball and $\mathscr{l}_2$-ball threat models, certified training with convex relaxations suffers from worse standard and robust error than adversarial training. We further explore how the error gap between certified and adversarial training depends on the threat model and the data distribution. In particular, besides the perturbation budget, we identify as important factors the shape of the perturbation set and the implicit margin of the data distribution. We support our arguments with extensive ablations on both synthetic and image datasets.

摘要: 对抗性攻击对在安全关键型应用中部署最先进的分类器构成了重大威胁。解决这一问题的方法有两类：经验性辩护和证明性辩护。虽然认证的防御具有健壮性保证，但经验防御，如对抗性训练，在从业者中享有更高的受欢迎程度。在本文中，我们系统地比较了这两种稳健训练范例在多个计算机视觉任务中的标准误差和稳健误差。结果表明，在大多数任务中，对于$\mathscr{L}_inty$-ball和$\mathscr{L}_2$-ball威胁模型，带凸松弛的认证训练的标准误差和稳健误差都比对抗性训练差。我们进一步探讨认证训练和对抗性训练之间的错误差距如何取决于威胁模型和数据分布。特别是，除了扰动预算之外，我们还将扰动集合的形状和数据分布的隐含裕度确定为重要因素。我们通过在合成数据集和图像数据集上进行广泛的消融来支持我们的论点。



## **15. Backdooring Neural Code Search**

回溯神经编码搜索 cs.SE

Accepted to the 61st Annual Meeting of the Association for  Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2305.17506v2) [paper-pdf](http://arxiv.org/pdf/2305.17506v2)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.

摘要: 重用在线存储库中的现成代码片段是一种常见的做法，这显著提高了软件开发人员的工作效率。为了找到所需的代码片段，开发人员通过自然语言查询求助于代码搜索引擎。因此，神经代码搜索模型是许多此类引擎的幕后推手。这些模型是基于深度学习的，由于其令人印象深刻的性能而获得了大量关注。然而，这些模型的安全性方面的研究很少。特别是，攻击者可以在神经代码搜索模型中注入后门，该模型返回带有安全/隐私问题的错误代码，甚至是易受攻击的代码。这可能会影响下游软件(例如股票交易系统和自动驾驶)，并导致经济损失和/或危及生命的事件。在这篇文章中，我们证明了这种攻击是可行的，并且可以相当隐蔽。只需修改一个变量/函数名称，攻击者就可以使有错误/易受攻击的代码排在前11%。我们的攻击BADCODE具有特殊的触发生成和注入过程，使攻击更有效和隐蔽。在两个神经编码搜索模型上进行了评估，结果表明我们的攻击性能比基线高60%。我们的用户研究表明，基于F1比分，我们的攻击比基线更隐蔽两倍。



## **16. Graph Agent Network: Empowering Nodes with Decentralized Communications Capabilities for Adversarial Resilience**

图代理网络：赋予节点分散的通信能力以提高对抗能力 cs.LG

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.06909v1) [paper-pdf](http://arxiv.org/pdf/2306.06909v1)

**Authors**: Ao Liu, Wenshan Li, Tao Li, Beibei Li, Hanyuan Huang, Guangquan Xu, Pan Zhou

**Abstract**: End-to-end training with global optimization have popularized graph neural networks (GNNs) for node classification, yet inadvertently introduced vulnerabilities to adversarial edge-perturbing attacks. Adversaries can exploit the inherent opened interfaces of GNNs' input and output, perturbing critical edges and thus manipulating the classification results. Current defenses, due to their persistent utilization of global-optimization-based end-to-end training schemes, inherently encapsulate the vulnerabilities of GNNs. This is specifically evidenced in their inability to defend against targeted secondary attacks. In this paper, we propose the Graph Agent Network (GAgN) to address the aforementioned vulnerabilities of GNNs. GAgN is a graph-structured agent network in which each node is designed as an 1-hop-view agent. Through the decentralized interactions between agents, they can learn to infer global perceptions to perform tasks including inferring embeddings, degrees and neighbor relationships for given nodes. This empowers nodes to filtering adversarial edges while carrying out classification tasks. Furthermore, agents' limited view prevents malicious messages from propagating globally in GAgN, thereby resisting global-optimization-based secondary attacks. We prove that single-hidden-layer multilayer perceptrons (MLPs) are theoretically sufficient to achieve these functionalities. Experimental results show that GAgN effectively implements all its intended capabilities and, compared to state-of-the-art defenses, achieves optimal classification accuracy on the perturbed datasets.

摘要: 具有全局优化的端到端训练普及了图神经网络(GNN)用于节点分类，但无意中引入了对敌意边缘扰动攻击的脆弱性。攻击者可以利用GNN输入和输出固有的开放接口，扰乱关键边缘，从而操纵分类结果。目前的防御措施由于持续使用基于全局优化的端到端培训方案，固有地封装了GNN的脆弱性。这一点具体表现在他们无法防御有针对性的二次攻击。在本文中，我们提出了图代理网络(GagN)来解决GNN的上述漏洞。GAGN是一个图结构的代理网络，其中每个节点被设计为一个1跳视图代理。通过代理之间的分散交互，它们可以学习推断全局感知来执行任务，包括推断给定节点的嵌入度、度数和邻居关系。这使节点能够在执行分类任务时过滤敌意边缘。此外，代理的有限视点防止恶意消息在GAGN中全局传播，从而抵抗基于全局优化的二次攻击。我们证明了单隐层多层感知器(MLP)理论上足以实现这些功能。实验结果表明，GAGN有效地实现了其预期的所有功能，并且与现有的防御措施相比，在扰动数据集上获得了最优的分类精度。



## **17. GAN-CAN: A Novel Attack to Behavior-Based Driver Authentication Systems**

GAN-CAN：一种针对基于行为的驾驶员身份认证系统的新型攻击 cs.CR

16 pages, 6 figures

**SubmitDate**: 2023-06-12    [abs](http://arxiv.org/abs/2306.05923v2) [paper-pdf](http://arxiv.org/pdf/2306.05923v2)

**Authors**: Emad Efatinasab, Francesco Marchiori, Denis Donadel, Alessandro Brighente, Mauro Conti

**Abstract**: For many years, car keys have been the sole mean of authentication in vehicles. Whether the access control process is physical or wireless, entrusting the ownership of a vehicle to a single token is prone to stealing attempts. For this reason, many researchers started developing behavior-based authentication systems. By collecting data in a moving vehicle, Deep Learning (DL) models can recognize patterns in the data and identify drivers based on their driving behavior. This can be used as an anti-theft system, as a thief would exhibit a different driving style compared to the vehicle owner's. However, the assumption that an attacker cannot replicate the legitimate driver behavior falls under certain conditions.   In this paper, we propose GAN-CAN, the first attack capable of fooling state-of-the-art behavior-based driver authentication systems in a vehicle. Based on the adversary's knowledge, we propose different GAN-CAN implementations. Our attack leverages the lack of security in the Controller Area Network (CAN) to inject suitably designed time-series data to mimic the legitimate driver. Our design of the malicious time series results from the combination of different Generative Adversarial Networks (GANs) and our study on the safety importance of the injected values during the attack. We tested GAN-CAN in an improved version of the most efficient driver behavior-based authentication model in the literature. We prove that our attack can fool it with an attack success rate of up to 0.99. We show how an attacker, without prior knowledge of the authentication system, can steal a car by deploying GAN-CAN in an off-the-shelf system in under 22 minutes.

摘要: 多年来，汽车钥匙一直是车辆身份验证的唯一手段。无论访问控制过程是物理的还是无线的，将车辆的所有权委托给单个令牌都容易发生窃取尝试。为此，许多研究人员开始开发基于行为的身份认证系统。通过收集移动车辆的数据，深度学习(DL)模型可以识别数据中的模式，并根据司机的驾驶行为识别司机。这可以用作防盗系统，因为与车主相比，小偷会表现出不同的驾驶风格。然而，在某些情况下，攻击者无法复制合法司机行为的假设是成立的。在本文中，我们提出了GAN-CAN，这是第一个能够欺骗车辆中最先进的基于行为的驾驶员身份验证系统的攻击。基于对手的知识，我们提出了不同的GAN-CAN实现方案。我们的攻击利用控制器区域网络(CAN)缺乏安全性来注入适当设计的时间序列数据，以模仿合法的驱动程序。我们的恶意时间序列的设计是不同生成性对抗网络(GANS)组合的结果，也是我们对攻击期间注入的值的安全重要性的研究的结果。我们在文献中最有效的基于司机行为的身份验证模型的改进版本中测试了GAN-CAN。我们证明了我们的攻击可以欺骗它，攻击成功率高达0.99。我们展示了攻击者如何在不了解身份验证系统的情况下，在不到22分钟的时间内通过在现成系统中部署GAN-CAN来窃取一辆汽车。



## **18. Asymptotically Optimal Adversarial Strategies for the Probability Estimation Framework**

概率估计框架的渐近最优对抗策略 quant-ph

54 pages

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06802v1) [paper-pdf](http://arxiv.org/pdf/2306.06802v1)

**Authors**: Soumyadip Patra, Peter Bierhorst

**Abstract**: The Probability Estimation Framework involves direct estimation of the probability of occurrences of outcomes conditioned on measurement settings and side information. It is a powerful tool for certifying randomness in quantum non-locality experiments. In this paper, we present a self-contained proof of the asymptotic optimality of the method. Our approach refines earlier results to allow a better characterisation of optimal adversarial attacks on the protocol. We apply these results to the (2,2,2) Bell scenario, obtaining an analytic characterisation of the optimal adversarial attacks bound by no-signalling principles, while also demonstrating the asymptotic robustness of the PEF method to deviations from expected experimental behaviour. We also study extensions of the analysis to quantum-limited adversaries in the (2,2,2) Bell scenario and no-signalling adversaries in higher $(n,m,k)$ Bell scenarios.

摘要: 概率估计框架涉及根据测量设置和辅助信息直接估计结果发生的概率。它是验证量子非定域性实验中随机性的有力工具。本文给出了该方法渐近最优性的一个完备证明。我们的方法改进了早期的结果，以便更好地描述对协议的最优敌意攻击。我们将这些结果应用于(2，2，2)Bell情形，得到了受无信令原理约束的最优对抗攻击的解析刻画，同时也证明了PEF方法对偏离预期实验行为的渐近鲁棒性。我们还研究了在(2，2，2)Bell情形下的量子受限对手和在更高的$(n，m，k)$Bell情形下的无信号对手的分析的扩展。



## **19. Adversarial Reconnaissance Mitigation and Modeling**

对抗性侦察消解与建模 cs.CR

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06769v1) [paper-pdf](http://arxiv.org/pdf/2306.06769v1)

**Authors**: Shanto Roy, Nazia Sharmin, Mohammad Sujan Miah, Jaime C Acosta, Christopher Kiekintveld, Aron Laszka

**Abstract**: Adversarial reconnaissance is a crucial step in sophisticated cyber-attacks as it enables threat actors to find the weakest points of otherwise well-defended systems. To thwart reconnaissance, defenders can employ cyber deception techniques, such as deploying honeypots. In recent years, researchers have made great strides in developing game-theoretic models to find optimal deception strategies. However, most of these game-theoretic models build on relatively simple models of adversarial reconnaissance -- even though reconnaissance should be a focus point as the very purpose of deception is to thwart reconnaissance. In this paper, we first discuss effective cyber reconnaissance mitigation techniques including deception strategies and beyond. Then we provide a review of the literature on deception games from the perspective of modeling adversarial reconnaissance, highlighting key aspects of reconnaissance that have not been adequately captured in prior work. We then describe a probability-theory based model of the adversaries' belief formation and illustrate using numerical examples that this model can capture key aspects of adversarial reconnaissance. We believe that our review and belief model can serve as a stepping stone for developing more realistic and practical deception games.

摘要: 对抗性侦察是复杂网络攻击的关键一步，因为它使威胁参与者能够找到原本防御良好的系统的最薄弱环节。为了阻止侦察，防御者可以使用网络欺骗技术，例如部署蜜罐。近年来，研究人员在开发博弈论模型以寻找最优欺骗策略方面取得了长足的进步。然而，这些博弈论模型大多建立在相对简单的对抗性侦察模型之上--尽管侦察应该是重点，因为欺骗的目的就是挫败侦察。在本文中，我们首先讨论了有效的网络侦察缓解技术，包括欺骗策略等。然后，我们从建模对抗侦察的角度对欺骗游戏的文献进行了回顾，强调了侦察的关键方面，这些方面在以前的工作中没有被充分捕获。然后，我们描述了一个基于概率论的敌方信念形成模型，并用数值例子说明了该模型能够捕捉敌方侦察的关键方面。我们相信，我们的复习和信念模型可以作为开发更现实和实用的欺骗游戏的垫脚石。



## **20. Neural Architecture Design and Robustness: A Dataset**

神经结构设计与稳健性：一个数据集 cs.LG

ICLR 2023; project page: http://robustness.vision/

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2306.06712v1) [paper-pdf](http://arxiv.org/pdf/2306.06712v1)

**Authors**: Steffen Jung, Jovita Lukasik, Margret Keuper

**Abstract**: Deep learning models have proven to be successful in a wide range of machine learning tasks. Yet, they are often highly sensitive to perturbations on the input data which can lead to incorrect decisions with high confidence, hampering their deployment for practical use-cases. Thus, finding architectures that are (more) robust against perturbations has received much attention in recent years. Just like the search for well-performing architectures in terms of clean accuracy, this usually involves a tedious trial-and-error process with one additional challenge: the evaluation of a network's robustness is significantly more expensive than its evaluation for clean accuracy. Thus, the aim of this paper is to facilitate better streamlined research on architectural design choices with respect to their impact on robustness as well as, for example, the evaluation of surrogate measures for robustness. We therefore borrow one of the most commonly considered search spaces for neural architecture search for image classification, NAS-Bench-201, which contains a manageable size of 6466 non-isomorphic network designs. We evaluate all these networks on a range of common adversarial attacks and corruption types and introduce a database on neural architecture design and robustness evaluations. We further present three exemplary use cases of this dataset, in which we (i) benchmark robustness measurements based on Jacobian and Hessian matrices for their robustness predictability, (ii) perform neural architecture search on robust accuracies, and (iii) provide an initial analysis of how architectural design choices affect robustness. We find that carefully crafting the topology of a network can have substantial impact on its robustness, where networks with the same parameter count range in mean adversarial robust accuracy from 20%-41%. Code and data is available at http://robustness.vision/.

摘要: 深度学习模型已被证明在广泛的机器学习任务中是成功的。然而，它们往往对输入数据的扰动高度敏感，这可能会导致高置信度的错误决策，阻碍它们在实际用例中的部署。因此，寻找对扰动(更)健壮的体系结构在最近几年受到了极大的关注。就像在干净准确性方面寻找表现良好的体系结构一样，这通常涉及一个乏味的反复试验过程，还有一个额外的挑战：评估网络的稳健性比评估其干净准确性的成本要高得多。因此，本文的目的是促进关于建筑设计选择对稳健性的影响的更好的简化研究，以及例如，对稳健性替代措施的评估。因此，我们借用了用于图像分类的神经结构搜索最常用的搜索空间之一NAS-BENCH-201，它包含6466个非同构网络设计的可管理大小。我们对所有这些网络在一系列常见的对抗性攻击和破坏类型上进行了评估，并引入了一个关于神经体系结构设计和健壮性评估的数据库。我们进一步给出了这个数据集的三个典型用例，其中我们(I)基于雅可比矩阵和海森矩阵对健壮性度量进行基准测试，以确定其健壮性可预测性，(Ii)对健壮性精度执行神经体系结构搜索，以及(Iii)提供体系结构设计选择如何影响健壮性的初步分析。我们发现，精心设计网络的拓扑结构可以对其健壮性产生重大影响，其中具有相同参数的网络的平均对抗健壮性准确率在20%-41%之间。代码和数据可在http://robustness.vision/.上获得



## **21. EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection**

EvadeDroid：一种实用的机器学习黑盒Android恶意软件检测规避攻击 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2110.03301v3) [paper-pdf](http://arxiv.org/pdf/2110.03301v3)

**Authors**: Hamid Bostani, Veelasha Moonsamy

**Abstract**: Over the last decade, researchers have extensively explored the vulnerabilities of Android malware detectors to adversarial examples through the development of evasion attacks; however, the practicality of these attacks in real-world scenarios remains arguable. The majority of studies have assumed attackers know the details of the target classifiers used for malware detection, while in reality, malicious actors have limited access to the target classifiers. This paper introduces EvadeDroid, a practical decision-based adversarial attack designed to effectively evade black-box Android malware detectors in real-world scenarios. In addition to generating real-world adversarial malware, the proposed evasion attack can also preserve the functionality of the original malware applications (apps). EvadeDroid constructs a collection of functionality-preserving transformations derived from benign donors that share opcode-level similarity with malware apps by leveraging an n-gram-based approach. These transformations are then used to morph malware instances into benign ones via an iterative and incremental manipulation strategy. The proposed manipulation technique is a novel, query-efficient optimization algorithm that can find and inject optimal sequences of transformations into malware apps. Our empirical evaluation demonstrates the efficacy of EvadeDroid under soft- and hard-label attacks. Furthermore, EvadeDroid exhibits the capability to generate real-world adversarial examples that can effectively evade a wide range of black-box ML-based malware detectors with minimal query requirements. Finally, we show that the proposed problem-space adversarial attack is able to preserve its stealthiness against five popular commercial antiviruses, thus demonstrating its feasibility in the real world.

摘要: 在过去的十年里，研究人员通过规避攻击的开发，广泛探索了Android恶意软件检测器对敌意例子的漏洞；然而，这些攻击在现实世界场景中的实用性仍然存在争议。大多数研究都假设攻击者知道用于恶意软件检测的目标分类器的详细信息，而实际上，恶意行为者对目标分类器的访问权限有限。本文介绍了EvadeDroid，一种实用的基于决策的对抗性攻击，旨在有效地躲避现实场景中的黑盒Android恶意软件检测。除了生成真实世界的敌意恶意软件外，拟议的逃避攻击还可以保留原始恶意软件应用程序(APP)的功能。EvadeDroid构建了一组保留功能的转换，这些转换来自良性捐赠者，通过利用基于n-gram的方法与恶意软件应用程序共享操作码级相似性。然后，这些转换被用于通过迭代和增量操作策略将恶意软件实例变形为良性实例。提出的操纵技术是一种新颖的、查询高效的优化算法，可以找到最优的转换序列并将其注入恶意软件应用程序。我们的经验评估证明了EvadeDroid在软标签和硬标签攻击下的有效性。此外，EvadeDroid展示了生成真实世界敌意示例的能力，这些示例可以有效地躲避各种基于黑盒ML的恶意软件检测器，而查询要求最低。最后，我们证明了所提出的问题空间对抗攻击能够对五种流行的商业反病毒保持其隐蔽性，从而证明了其在现实世界中的可行性。



## **22. Level Up with RealAEs: Leveraging Domain Constraints in Feature Space to Strengthen Robustness of Android Malware Detection**

与RealEs并驾齐驱：利用特征空间中的域约束增强Android恶意软件检测的健壮性 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2205.15128v3) [paper-pdf](http://arxiv.org/pdf/2205.15128v3)

**Authors**: Hamid Bostani, Zhengyu Zhao, Zhuoran Liu, Veelasha Moonsamy

**Abstract**: The vulnerability to adversarial examples remains one major obstacle for Machine Learning (ML)-based Android malware detection. Realistic attacks in the Android malware domain create Realizable Adversarial Examples (RealAEs), i.e., AEs that satisfy the domain constraints of Android malware. Recent studies have shown that using such RealAEs in Adversarial Training (AT) is more effective in defending against realistic attacks than using unrealizable AEs (unRealAEs). This is because RealAEs allow defenders to explore certain pockets in the feature space that are vulnerable to realistic attacks. However, existing defenses commonly generate RealAEs in the problem space, which is known to be time-consuming and impractical for AT. In this paper, we propose to generate RealAEs in the feature space, leading to a simpler and more efficient solution. Our approach is driven by a novel interpretation of Android domain constraints in the feature space. More concretely, our defense first learns feature-space domain constraints by extracting meaningful feature dependencies from data and then applies them to generating feature-space RealAEs during AT. Extensive experiments on DREBIN, a well-known Android malware detector, demonstrate that our new defense outperforms not only unRealAE-based AT but also the state-of-the-art defense that relies on non-uniform perturbations. We further validate the ability of our learned feature-space domain constraints in representing Android malware properties by showing that our feature-space domain constraints can help distinguish RealAEs from unRealAEs.

摘要: 恶意示例的漏洞仍然是基于机器学习(ML)的Android恶意软件检测的主要障碍。Android恶意软件领域中的现实攻击创建了可实现的对抗性实例(RealAE)，即满足Android恶意软件的域约束的AE。最近的研究表明，在对抗训练(AT)中使用这种真实AEs比使用不可实现AEs(UnRealAEs)更有效地防御现实攻击。这是因为RealEs允许防御者探索特征空间中易受现实攻击的某些口袋。然而，现有的防御通常会在问题空间中生成RealEs，这对于AT来说是耗时和不切实际的。在本文中，我们建议在特征空间中生成RealAE，从而得到一个更简单、更有效的解决方案。我们的方法是由特征空间中对Android领域约束的一种新解释驱动的。更具体地说，我们的防御首先通过从数据中提取有意义的特征依赖来学习特征空间域约束，然后将它们应用于在AT过程中生成特征空间RealAE。在著名的Android恶意软件检测器Drebin上的广泛实验表明，我们的新防御不仅优于基于UnRealAE的AT，而且优于依赖非均匀扰动的最先进防御。我们进一步验证了我们学习的特征空间域约束在表示Android恶意软件属性方面的能力，方法是展示我们的特征空间域约束可以帮助区分真正的恶意软件和非真实的恶意软件。



## **23. Attacking Cooperative Multi-Agent Reinforcement Learning by Adversarial Minority Influence**

对抗性少数影响攻击协作式多智能体强化学习 cs.LG

**SubmitDate**: 2023-06-11    [abs](http://arxiv.org/abs/2302.03322v2) [paper-pdf](http://arxiv.org/pdf/2302.03322v2)

**Authors**: Simin Li, Jun Guo, Jingqiao Xiu, Pu Feng, Xin Yu, Aishan Liu, Wenjun Wu, Xianglong Liu

**Abstract**: This study probes the vulnerabilities of cooperative multi-agent reinforcement learning (c-MARL) under adversarial attacks, a critical determinant of c-MARL's worst-case performance prior to real-world implementation. Current observation-based attacks, constrained by white-box assumptions, overlook c-MARL's complex multi-agent interactions and cooperative objectives, resulting in impractical and limited attack capabilities. To address these shortcomes, we propose Adversarial Minority Influence (AMI), a practical and strong for c-MARL. AMI is a practical black-box attack and can be launched without knowing victim parameters. AMI is also strong by considering the complex multi-agent interaction and the cooperative goal of agents, enabling a single adversarial agent to unilaterally misleads majority victims to form targeted worst-case cooperation. This mirrors minority influence phenomena in social psychology. To achieve maximum deviation in victim policies under complex agent-wise interactions, our unilateral attack aims to characterize and maximize the impact of the adversary on the victims. This is achieved by adapting a unilateral agent-wise relation metric derived from mutual information, thereby mitigating the adverse effects of victim influence on the adversary. To lead the victims into a jointly detrimental scenario, our targeted attack deceives victims into a long-term, cooperatively harmful situation by guiding each victim towards a specific target, determined through a trial-and-error process executed by a reinforcement learning agent. Through AMI, we achieve the first successful attack against real-world robot swarms and effectively fool agents in simulated environments into collectively worst-case scenarios, including Starcraft II and Multi-agent Mujoco. The source code and demonstrations can be found at: https://github.com/DIG-Beihang/AMI.

摘要: 该研究探讨了协作多智能体强化学习(c-Marl)在对抗攻击下的脆弱性，这是c-Marl在现实世界实现之前最差情况性能的关键决定因素。目前的基于观测的攻击受白盒假设的约束，忽略了c-Marl复杂的多智能体交互和合作目标，导致攻击能力不切实际和有限。针对这些不足，我们提出了一种实用而强大的c-Marl算法--对抗性少数影响算法。AMI是一种实用的黑盒攻击，可以在不知道受害者参数的情况下启动。通过考虑复杂的多智能体相互作用和智能体的合作目标，使单一对抗智能体能够单方面误导大多数受害者形成有针对性的最坏情况合作，AMI也很强大。这反映了社会心理学中的小众影响现象。为了在复杂的智能体相互作用下实现受害者政策的最大偏差，我们的单边攻击旨在刻画和最大化对手对受害者的影响。这是通过采用来自互信息的单边代理关系度量来实现的，从而减轻了受害者影响对对手的不利影响。为了将受害者引导到共同有害的情景中，我们的有针对性的攻击通过引导每个受害者指向特定的目标，将受害者欺骗到长期的、合作有害的情况中，该特定目标是通过由强化学习代理执行的反复试验过程确定的。通过AMI，我们实现了对真实世界机器人群的第一次成功攻击，并有效地将模拟环境中的代理愚弄到了集体最坏的情况下，包括星际争霸II和多代理Mujoco。源代码和演示可在以下网址找到：https://github.com/DIG-Beihang/AMI.



## **24. Defense Against Adversarial Attacks on Audio DeepFake Detection**

音频DeepFake检测中的敌意攻击防御 cs.SD

Accepted to INTERSPEECH 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2212.14597v2) [paper-pdf](http://arxiv.org/pdf/2212.14597v2)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes (DF) are artificially generated utterances created using deep learning, with the primary aim of fooling the listeners in a highly convincing manner. Their quality is sufficient to pose a severe threat in terms of security and privacy, including the reliability of news or defamation. Multiple neural network-based methods to detect generated speech have been proposed to prevent the threats. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability) and enhancing it later by using adversarial training performed by our novel adaptive training. Moreover, one of the investigated architectures is RawNet3, which, to the best of our knowledge, we adapted for the first time to DeepFake detection.

摘要: Audio DeepFake(DF)是使用深度学习创建的人工生成的话语，其主要目的是以高度令人信服的方式愚弄听众。它们的质量足以在安全和隐私方面构成严重威胁，包括新闻或诽谤的可靠性。为了防止这种威胁，已经提出了多种基于神经网络的生成语音检测方法。在这项工作中，我们讨论了对抗性攻击的主题，它通过向输入数据添加表面(难以被人发现)的更改来降低检测器的性能。我们的贡献包括评估三种检测体系结构在两种场景(白盒和使用可转移性)下对敌意攻击的健壮性，并在以后通过使用我们新的自适应训练执行的对抗性训练来增强它。此外，其中一个被研究的架构是RawNet3，据我们所知，我们第一次将其应用于DeepFake检测。



## **25. The Defense of Networked Targets in General Lotto games**

普通彩票游戏中网络目标的防御 cs.GT

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06485v1) [paper-pdf](http://arxiv.org/pdf/2306.06485v1)

**Authors**: Adel Aghajan, Keith Paarporn, Jason R. Marden

**Abstract**: Ensuring the security of networked systems is a significant problem, considering the susceptibility of modern infrastructures and technologies to adversarial interference. A central component of this problem is how defensive resources should be allocated to mitigate the severity of potential attacks on the system. In this paper, we consider this in the context of a General Lotto game, where a defender and attacker deploys resources on the nodes of a network, and the objective is to secure as many links as possible. The defender secures a link only if it out-competes the attacker on both of its associated nodes. For bipartite networks, we completely characterize equilibrium payoffs and strategies for both the defender and attacker. Surprisingly, the resulting payoffs are the same for any bipartite graph. On arbitrary network structures, we provide lower and upper bounds on the defender's max-min value. Notably, the equilibrium payoff from bipartite networks serves as the lower bound. These results suggest that more connected networks are easier to defend against attacks. We confirm these findings with simulations that compute deterministic allocation strategies on large random networks. This also highlights the importance of randomization in the equilibrium strategies.

摘要: 考虑到现代基础设施和技术对敌方干扰的敏感性，确保联网系统的安全是一个重大问题。这个问题的一个核心部分是应该如何分配防御资源，以减轻对系统的潜在攻击的严重性。在本文中，我们在一般彩票博弈的背景下考虑这一问题，其中防御者和攻击者在网络的节点上部署资源，目标是确保尽可能多的链路。只有当防御者在两个相关节点上都胜过攻击者时，防御者才能确保链路的安全。对于二部网络，我们完全刻画了防御者和攻击者的均衡收益和策略。令人惊讶的是，由此产生的收益对于任何二部图都是相同的。在任意网络结构下，我们给出了防御者的最大最小值的上下界。值得注意的是，二部网络的均衡收益是下限。这些结果表明，连接越紧密的网络越容易抵御攻击。我们通过在大型随机网络上计算确定性分配策略的模拟来证实这些发现。这也突显了随机化在均衡战略中的重要性。



## **26. NeRFool: Uncovering the Vulnerability of Generalizable Neural Radiance Fields against Adversarial Perturbations**

NeRFool：发现可推广的神经辐射场对抗敌方扰动的脆弱性 cs.CV

Accepted by ICML 2023

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06359v1) [paper-pdf](http://arxiv.org/pdf/2306.06359v1)

**Authors**: Yonggan Fu, Ye Yuan, Souvik Kundu, Shang Wu, Shunyao Zhang, Yingyan Lin

**Abstract**: Generalizable Neural Radiance Fields (GNeRF) are one of the most promising real-world solutions for novel view synthesis, thanks to their cross-scene generalization capability and thus the possibility of instant rendering on new scenes. While adversarial robustness is essential for real-world applications, little study has been devoted to understanding its implication on GNeRF. We hypothesize that because GNeRF is implemented by conditioning on the source views from new scenes, which are often acquired from the Internet or third-party providers, there are potential new security concerns regarding its real-world applications. Meanwhile, existing understanding and solutions for neural networks' adversarial robustness may not be applicable to GNeRF, due to its 3D nature and uniquely diverse operations. To this end, we present NeRFool, which to the best of our knowledge is the first work that sets out to understand the adversarial robustness of GNeRF. Specifically, NeRFool unveils the vulnerability patterns and important insights regarding GNeRF's adversarial robustness. Built upon the above insights gained from NeRFool, we further develop NeRFool+, which integrates two techniques capable of effectively attacking GNeRF across a wide range of target views, and provide guidelines for defending against our proposed attacks. We believe that our NeRFool/NeRFool+ lays the initial foundation for future innovations in developing robust real-world GNeRF solutions. Our codes are available at: https://github.com/GATECH-EIC/NeRFool.

摘要: 可概括神经辐射场(GNeRF)是现实世界中最有前途的新型视点合成解决方案之一，这要归功于它们的跨场景泛化能力，从而可以在新场景上进行即时渲染。虽然对抗的稳健性对于现实世界的应用是必不可少的，但很少有研究致力于了解其对GNeRF的影响。我们假设，由于GNeRF是通过对来自新场景的源视图进行条件处理来实现的，这些场景通常是从互联网或第三方提供商获得的，因此其现实世界的应用程序存在潜在的新的安全问题。同时，由于GNeRF的3D性质和独特的多样性操作，现有对神经网络对抗性稳健性的理解和解决方案可能不适用于GNeRF。为此，我们提出了NeRFool，据我们所知，这是第一个开始了解GNeRF的对手健壮性的工作。具体地说，NeRFool揭示了关于GNeRF的对手健壮性的漏洞模式和重要见解。基于以上从NeRFool获得的见解，我们进一步开发了NeRFool+，它集成了两种能够在广泛的目标视图中有效攻击GNeRF的技术，并为防御我们提出的攻击提供了指导方针。我们相信，我们的NeRFool/NeRFool+为未来在开发强大的现实世界GNeRF解决方案方面的创新奠定了初步基础。我们的代码请访问：https://github.com/GATECH-EIC/NeRFool.



## **27. Differentially private sliced inverse regression in the federated paradigm**

联邦范例中的差分私有切片逆回归 stat.ME

**SubmitDate**: 2023-06-10    [abs](http://arxiv.org/abs/2306.06324v1) [paper-pdf](http://arxiv.org/pdf/2306.06324v1)

**Authors**: Shuaida He, Jiarui Zhang, Xin Chen

**Abstract**: We extend the celebrated sliced inverse regression to address the challenges of decentralized data, prioritizing privacy and communication efficiency. Our approach, federated sliced inverse regression (FSIR), facilitates collaborative estimation of the sufficient dimension reduction subspace among multiple clients, solely sharing local estimates to protect sensitive datasets from exposure. To guard against potential adversary attacks, FSIR further employs diverse perturbation strategies, including a novel multivariate Gaussian mechanism that guarantees differential privacy at a low cost of statistical accuracy. Additionally, FSIR naturally incorporates a collaborative variable screening step, enabling effective handling of high-dimensional client data. Theoretical properties of FSIR are established for both low-dimensional and high-dimensional settings, supported by extensive numerical experiments and real data analysis.

摘要: 我们扩展了著名的切片反向回归，以应对分散数据、优先考虑隐私和通信效率的挑战。我们的方法，联合切片逆回归(FSIR)，促进了多个客户之间对足够降维空间的协作估计，只共享局部估计来保护敏感数据集免受暴露。为了防止潜在的对手攻击，FSIR进一步采用了多种扰动策略，包括一种新颖的多变量高斯机制，该机制以较低的统计准确性为代价保证了差分隐私。此外，FSIR自然包含协作可变筛选步骤，从而能够有效处理高维客户数据。在大量的数值实验和实际数据分析的支持下，建立了低维和高维环境下FSIR的理论性质。



## **28. The Certification Paradox: Certifications Admit Better Attacks**

认证悖论：认证允许更好的攻击 cs.LG

16 pages, 6 figures

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2302.04379v2) [paper-pdf](http://arxiv.org/pdf/2302.04379v2)

**Authors**: Andrew C. Cullen, Shijie Liu, Paul Montague, Sarah M. Erfani, Benjamin I. P. Rubinstein

**Abstract**: In guaranteeing that no adversarial examples exist within a bounded region, certification mechanisms play an important role in demonstrating the robustness of neural networks. In this work we ask: Could certifications have any unintended consequences, through exposing additional information about certified models? We answer this question in the affirmative, demonstrating that certifications not only measure model robustness but also present a new attack surface. We propose \emph{Certification Aware Attacks}, that produce smaller adversarial perturbations more than twice as frequently as any prior approach, when launched against certified models. Our attacks achieve an up to $34\%$ reduction in the median perturbation norm (comparing target and attack instances), while requiring $90 \%$ less computational time than approaches like PGD. That our attacks achieve such significant reductions in perturbation size and computational cost highlights an apparent paradox in deploying certification mechanisms. We end the paper with a discussion of how these risks could potentially be mitigated.

摘要: 在保证有界区域内不存在对抗性实例方面，认证机制在证明神经网络的健壮性方面发挥了重要作用。在这项工作中，我们问：通过暴露有关认证模型的更多信息，认证是否会产生任何意想不到的后果？我们对这个问题的回答是肯定的，证明了认证不仅衡量了模型的稳健性，而且还提供了一个新的攻击面。我们提出了认证感知攻击，当针对认证模型发起攻击时，产生较小的对抗性扰动的频率是之前任何方法的两倍以上。我们的攻击使中值扰动范数(比较目标和攻击实例)减少了高达34美元，而与PGD等方法相比，所需计算时间减少了90美元。我们的攻击在扰动大小和计算成本方面实现了如此显著的减少，突显了部署认证机制的一个明显的悖论。我们在文章的最后讨论了如何潜在地减轻这些风险。



## **29. Divide and Repair: Using Options to Improve Performance of Imitation Learning Against Adversarial Demonstrations**

区分和修复：使用选项提高对抗对手演示的模仿学习的性能 cs.LG

33 pages, 4 figures, 3 tables

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.04581v2) [paper-pdf](http://arxiv.org/pdf/2306.04581v2)

**Authors**: Prithviraj Dasgupta

**Abstract**: We consider the problem of learning to perform a task from demonstrations given by teachers or experts, when some of the experts' demonstrations might be adversarial and demonstrate an incorrect way to perform the task. We propose a novel technique that can identify parts of demonstrated trajectories that have not been significantly modified by the adversary and utilize them for learning, using temporally extended policies or options. We first define a trajectory divergence measure based on the spatial and temporal features of demonstrated trajectories to detect and discard parts of the trajectories that have been significantly modified by an adversarial expert, and, could degrade the learner's performance, if used for learning, We then use an options-based algorithm that partitions trajectories and learns only from the parts of trajectories that have been determined as admissible. We provide theoretical results of our technique to show that repairing partial trajectories improves the sample efficiency of the demonstrations without degrading the learner's performance. We then evaluate the proposed algorithm for learning to play an Atari-like, computer-based game called LunarLander in the presence of different types and degrees of adversarial attacks of demonstrated trajectories. Our experimental results show that our technique can identify adversarially modified parts of the demonstrated trajectories and successfully prevent the learning performance from degrading due to adversarial demonstrations.

摘要: 我们认为从教师或专家的演示中学习执行任务的问题是，当一些专家的演示可能是对抗性的，并展示了执行任务的不正确方式时。我们提出了一种新的技术，可以识别所展示的轨迹中没有被对手显著修改的部分，并使用临时扩展的策略或选项来利用它们进行学习。我们首先根据所演示轨迹的时空特征定义一个轨迹离散度来检测和丢弃被对手专家显著修改的轨迹的一部分，并且，如果用于学习，可能会降低学习者的性能，然后我们使用基于选项的算法来划分轨迹，并且只从被确定为可接受的轨迹的部分中学习。我们提供了我们的技术的理论结果，表明修复部分轨迹提高了演示的样本效率，而不会降低学习者的表现。然后，我们评估了在存在不同类型和程度的对抗性攻击所演示的轨迹的情况下，所提出的用于学习玩一款名为LUNARLADER的类似Atari的计算机游戏的算法。我们的实验结果表明，我们的技术能够识别演示轨迹中被敌意修改的部分，并成功地防止了由于对抗性演示而导致的学习性能下降。



## **30. Overcoming Adversarial Attacks for Human-in-the-Loop Applications**

克服针对人在环中应用的敌意攻击 cs.LG

New Frontiers in Adversarial Machine Learning, ICML 2022

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05952v1) [paper-pdf](http://arxiv.org/pdf/2306.05952v1)

**Authors**: Ryan McCoppin, Marla Kennedy, Platon Lukyanenko, Sean Kennedy

**Abstract**: Including human analysis has the potential to positively affect the robustness of Deep Neural Networks and is relatively unexplored in the Adversarial Machine Learning literature. Neural network visual explanation maps have been shown to be prone to adversarial attacks. Further research is needed in order to select robust visualizations of explanations for the image analyst to evaluate a given model. These factors greatly impact Human-In-The-Loop (HITL) evaluation tools due to their reliance on adversarial images, including explanation maps and measurements of robustness. We believe models of human visual attention may improve interpretability and robustness of human-machine imagery analysis systems. Our challenge remains, how can HITL evaluation be robust in this adversarial landscape?

摘要: 包括人类分析有可能对深度神经网络的稳健性产生积极影响，在对抗性机器学习文献中相对未被探索。神经网络视觉解释地图已被证明容易受到敌意攻击。还需要进一步的研究，以便为图像分析员选择稳健的解释可视化来评估给定的模型。这些因素极大地影响了人在环(HITL)评估工具，因为它们依赖于敌方图像，包括解释地图和健壮性测量。我们相信人类视觉注意模型可以提高人机图像分析系统的可解释性和稳健性。我们的挑战仍然是，如何在这种对抗性的环境中进行HITL评估？



## **31. Detecting Adversarial Directions in Deep Reinforcement Learning to Make Robust Decisions**

深度强化学习中检测敌方向以做出稳健决策 cs.LG

Published in ICML 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05873v1) [paper-pdf](http://arxiv.org/pdf/2306.05873v1)

**Authors**: Ezgi Korkmaz, Jonah Brown-Cohen

**Abstract**: Learning in MDPs with highly complex state representations is currently possible due to multiple advancements in reinforcement learning algorithm design. However, this incline in complexity, and furthermore the increase in the dimensions of the observation came at the cost of volatility that can be taken advantage of via adversarial attacks (i.e. moving along worst-case directions in the observation space). To solve this policy instability problem we propose a novel method to detect the presence of these non-robust directions via local quadratic approximation of the deep neural policy loss. Our method provides a theoretical basis for the fundamental cut-off between safe observations and adversarial observations. Furthermore, our technique is computationally efficient, and does not depend on the methods used to produce the worst-case directions. We conduct extensive experiments in the Arcade Learning Environment with several different adversarial attack techniques. Most significantly, we demonstrate the effectiveness of our approach even in the setting where non-robust directions are explicitly optimized to circumvent our proposed method.

摘要: 由于强化学习算法设计的多方面改进，目前在具有高度复杂状态表示的MDP中学习是可能的。然而，这种复杂性的倾向，以及观测维度的增加，是以波动性为代价的，这种波动性可以通过对抗性攻击(即沿着观测空间中最坏情况的方向移动)来利用。为了解决这一策略不稳定问题，我们提出了一种新的方法，通过对深层神经网络策略损失的局部二次逼近来检测这些非稳健方向的存在。我们的方法为安全观测和敌对观测之间的根本界限提供了理论依据。此外，我们的技术在计算上是有效的，并且不依赖于用于产生最坏情况方向的方法。我们在拱廊学习环境中用几种不同的对抗性攻击技术进行了广泛的实验。最重要的是，我们证明了我们的方法的有效性，即使在非稳健方向被明确优化以绕过我们所提出的方法的情况下也是如此。



## **32. Towards a Robust Detection of Language Model Generated Text: Is ChatGPT that Easy to Detect?**

面向语言模型生成文本的稳健检测：ChatGPT真的那么容易检测吗？ cs.CL

Accepted to TALN 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05871v1) [paper-pdf](http://arxiv.org/pdf/2306.05871v1)

**Authors**: Wissam Antoun, Virginie Mouilleron, Benoît Sagot, Djamé Seddah

**Abstract**: Recent advances in natural language processing (NLP) have led to the development of large language models (LLMs) such as ChatGPT. This paper proposes a methodology for developing and evaluating ChatGPT detectors for French text, with a focus on investigating their robustness on out-of-domain data and against common attack schemes. The proposed method involves translating an English dataset into French and training a classifier on the translated data. Results show that the detectors can effectively detect ChatGPT-generated text, with a degree of robustness against basic attack techniques in in-domain settings. However, vulnerabilities are evident in out-of-domain contexts, highlighting the challenge of detecting adversarial text. The study emphasizes caution when applying in-domain testing results to a wider variety of content. We provide our translated datasets and models as open-source resources. https://gitlab.inria.fr/wantoun/robust-chatgpt-detection

摘要: 自然语言处理(NLP)的最新进展导致了诸如ChatGPT的大型语言模型(LLM)的发展。本文提出了一种开发和评估法语文本ChatGPT检测器的方法，重点研究了它们对域外数据和常见攻击方案的健壮性。提出的方法包括将英语数据集翻译成法语，并对翻译后的数据训练分类器。结果表明，该检测器能够有效地检测到ChatGPT生成的文本，对域内环境下的基本攻击技术具有一定的稳健性。然而，漏洞在域外环境中很明显，突显了检测敌意文本的挑战。这项研究强调，在将领域内测试结果应用于更广泛的内容时要谨慎。我们以开源资源的形式提供翻译后的数据集和模型。Https://gitlab.inria.fr/wantoun/robust-chatgpt-detection



## **33. COVER: A Heuristic Greedy Adversarial Attack on Prompt-based Learning in Language Models**

封面：对语言模型中基于提示的学习的启发式贪婪对抗性攻击 cs.CL

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05659v1) [paper-pdf](http://arxiv.org/pdf/2306.05659v1)

**Authors**: Zihao Tan, Qingliang Chen, Wenbin Zhu, Yongjian Huang

**Abstract**: Prompt-based learning has been proved to be an effective way in pre-trained language models (PLMs), especially in low-resource scenarios like few-shot settings. However, the trustworthiness of PLMs is of paramount significance and potential vulnerabilities have been shown in prompt-based templates that could mislead the predictions of language models, causing serious security concerns. In this paper, we will shed light on some vulnerabilities of PLMs, by proposing a prompt-based adversarial attack on manual templates in black box scenarios. First of all, we design character-level and word-level heuristic approaches to break manual templates separately. Then we present a greedy algorithm for the attack based on the above heuristic destructive approaches. Finally, we evaluate our approach with the classification tasks on three variants of BERT series models and eight datasets. And comprehensive experimental results justify the effectiveness of our approach in terms of attack success rate and attack speed. Further experimental studies indicate that our proposed method also displays good capabilities in scenarios with varying shot counts, template lengths and query counts, exhibiting good generalizability.

摘要: 基于提示的学习已被证明是预训练语言模型(PLM)中的一种有效方法，特别是在资源较少的场景中，如少镜头场景。然而，PLM的可信性至关重要，基于提示的模板中已经显示出潜在的漏洞，这些漏洞可能会误导语言模型的预测，导致严重的安全问题。在本文中，我们将通过在黑盒场景中对人工模板提出一种基于提示的对抗性攻击来揭示PLM的一些漏洞。首先，我们分别设计了字字级和词级启发式方法来打破人工模板。在此基础上，提出了一种基于上述启发式破坏性方法的贪婪算法。最后，我们在BERT系列模型的三个变种和八个数据集上对我们的方法进行了评估。综合实验结果从攻击成功率和攻击速度两个方面验证了该方法的有效性。进一步的实验研究表明，该方法在镜头数、模板长度和查询次数不同的场景中也表现出了良好的性能，表现出良好的泛化能力。



## **34. Spike timing reshapes robustness against attacks in spiking neural networks**

尖峰定时重塑尖峰神经网络对攻击的稳健性 q-bio.NC

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05654v1) [paper-pdf](http://arxiv.org/pdf/2306.05654v1)

**Authors**: Jianhao Ding, Zhaofei Yu, Tiejun Huang, Jian K. Liu

**Abstract**: The success of deep learning in the past decade is partially shrouded in the shadow of adversarial attacks. In contrast, the brain is far more robust at complex cognitive tasks. Utilizing the advantage that neurons in the brain communicate via spikes, spiking neural networks (SNNs) are emerging as a new type of neural network model, boosting the frontier of theoretical investigation and empirical application of artificial neural networks and deep learning. Neuroscience research proposes that the precise timing of neural spikes plays an important role in the information coding and sensory processing of the biological brain. However, the role of spike timing in SNNs is less considered and far from understood. Here we systematically explored the timing mechanism of spike coding in SNNs, focusing on the robustness of the system against various types of attacks. We found that SNNs can achieve higher robustness improvement using the coding principle of precise spike timing in neural encoding and decoding, facilitated by different learning rules. Our results suggest that the utility of spike timing coding in SNNs could improve the robustness against attacks, providing a new approach to reliable coding principles for developing next-generation brain-inspired deep learning.

摘要: 过去十年深度学习的成功在一定程度上笼罩在对抗性攻击的阴影之下。相比之下，大脑在复杂的认知任务中要健壮得多。利用大脑中神经元通过棘波进行交流的优势，脉冲神经网络(SNN)正在成为一种新型的神经网络模型，推动了人工神经网络和深度学习的理论研究和实证应用的前沿。神经科学研究表明，神经棘波的精确计时在生物大脑的信息编码和感觉处理中发挥着重要作用。然而，放电时序在SNN中的作用较少被考虑，也远未被理解。本文系统地研究了SNN中脉冲编码的定时机制，重点研究了该系统对各种类型攻击的健壮性。我们发现，在神经编解码中使用精确的尖峰定时的编码原理，并辅之以不同的学习规则，SNN可以获得更高的稳健性改进。我们的结果表明，脉冲定时编码在SNN中的应用可以提高对攻击的稳健性，为开发下一代脑启发深度学习提供了一种新的可靠编码原则。



## **35. McFIL: Model Counting Functionality-Inherent Leakage**

McFIL：模型计数功能-固有泄漏 cs.CR

To appear in USENIX Security 2023

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.05633v1) [paper-pdf](http://arxiv.org/pdf/2306.05633v1)

**Authors**: Maximilian Zinkus, Yinzhi Cao, Matthew Green

**Abstract**: Protecting the confidentiality of private data and using it for useful collaboration have long been at odds. Modern cryptography is bridging this gap through rapid growth in secure protocols such as multi-party computation, fully-homomorphic encryption, and zero-knowledge proofs. However, even with provable indistinguishability or zero-knowledgeness, confidentiality loss from leakage inherent to the functionality may partially or even completely compromise secret values without ever falsifying proofs of security. In this work, we describe McFIL, an algorithmic approach and accompanying software implementation which automatically quantifies intrinsic leakage for a given functionality. Extending and generalizing the Chosen-Ciphertext attack framework of Beck et al. with a practical heuristic, our approach not only quantifies but maximizes functionality-inherent leakage using Maximum Model Counting within a SAT solver. As a result, McFIL automatically derives approximately-optimal adversary inputs that, when used in secure protocols, maximize information leakage of private values.

摘要: 长期以来，保护私人数据的机密性和将其用于有用的合作一直存在分歧。现代密码学正在通过多方计算、完全同态加密和零知识证明等安全协议的快速增长来弥合这一差距。然而，即使在可证明的不可区分或零知识的情况下，由于功能固有的泄漏而造成的机密性损失也可能部分或甚至完全危及保密值，而永远不会伪造安全证明。在这项工作中，我们描述了McFIL，一种算法方法和伴随的软件实现，它自动量化给定功能的固有泄漏。推广和推广了Beck等人的选择密文攻击框架。通过实用的启发式方法，我们的方法不仅量化而且最大化了功能-使用SAT解算器中的最大模型计数来实现固有的泄漏。因此，McFIL自动得出近似最优的对手输入，当在安全协议中使用时，最大限度地泄露私人价值的信息。



## **36. Robustness Testing for Multi-Agent Reinforcement Learning: State Perturbations on Critical Agents**

多智能体强化学习的稳健性测试：关键智能体的状态扰动 cs.LG

**SubmitDate**: 2023-06-09    [abs](http://arxiv.org/abs/2306.06136v1) [paper-pdf](http://arxiv.org/pdf/2306.06136v1)

**Authors**: Ziyuan Zhou, Guanjun Liu

**Abstract**: Multi-Agent Reinforcement Learning (MARL) has been widely applied in many fields such as smart traffic and unmanned aerial vehicles. However, most MARL algorithms are vulnerable to adversarial perturbations on agent states. Robustness testing for a trained model is an essential step for confirming the trustworthiness of the model against unexpected perturbations. This work proposes a novel Robustness Testing framework for MARL that attacks states of Critical Agents (RTCA). The RTCA has two innovations: 1) a Differential Evolution (DE) based method to select critical agents as victims and to advise the worst-case joint actions on them; and 2) a team cooperation policy evaluation method employed as the objective function for the optimization of DE. Then, adversarial state perturbations of the critical agents are generated based on the worst-case joint actions. This is the first robustness testing framework with varying victim agents. RTCA demonstrates outstanding performance in terms of the number of victim agents and destroying cooperation policies.

摘要: 多智能体强化学习在智能交通、无人机等领域有着广泛的应用。然而，大多数MAIL算法容易受到智能体状态的对抗性扰动。对训练好的模型进行稳健性测试是确认模型对意外扰动的可信性的重要步骤。针对攻击关键代理状态的MAIL提出了一种新的健壮性测试框架。RTCA有两个创新之处：1)基于差异进化(DE)的方法，选择关键主体作为受害者，并对最坏情况下的联合行动提出建议；2)将团队合作策略评估方法作为DE优化的目标函数。然后，基于最坏情况下的联合行动产生关键智能体的对抗状态扰动。这是第一个使用不同受害者代理的健壮性测试框架。RTCA在受害者代理的数量和破坏合作政策方面表现出色。



## **37. Adversarial Evasion Attacks Practicality in Networks: Testing the Impact of Dynamic Learning**

对抗性逃避攻击在网络中的实用性：测试动态学习的影响 cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05494v1) [paper-pdf](http://arxiv.org/pdf/2306.05494v1)

**Authors**: Mohamed el Shehaby, Ashraf Matrawy

**Abstract**: Machine Learning (ML) has become ubiquitous, and its deployment in Network Intrusion Detection Systems (NIDS) is inevitable due to its automated nature and high accuracy in processing and classifying large volumes of data. However, ML has been found to have several flaws, on top of them are adversarial attacks, which aim to trick ML models into producing faulty predictions. While most adversarial attack research focuses on computer vision datasets, recent studies have explored the practicality of such attacks against ML-based network security entities, especially NIDS.   This paper presents two distinct contributions: a taxonomy of practicality issues associated with adversarial attacks against ML-based NIDS and an investigation of the impact of continuous training on adversarial attacks against NIDS. Our experiments indicate that continuous re-training, even without adversarial training, can reduce the effect of adversarial attacks. While adversarial attacks can harm ML-based NIDSs, our aim is to highlight that there is a significant gap between research and real-world practicality in this domain which requires attention.

摘要: 机器学习(ML)已经变得无处不在，由于其在处理和分类海量数据方面的自动化和高精度，它在网络入侵检测系统(NIDS)中的应用是不可避免的。然而，ML被发现有几个缺陷，在这些缺陷之上是对抗性攻击，目的是欺骗ML模型产生错误的预测。虽然大多数对抗性攻击研究都集中在计算机视觉数据集上，但最近的研究探索了针对基于ML的网络安全实体，特别是NID的此类攻击的实用性。本文提出了两个不同的贡献：对与基于ML的网络入侵检测系统的对抗性攻击相关的实用性问题的分类，以及对持续训练对针对网络入侵检测系统的对抗性攻击的影响的调查。我们的实验表明，即使在没有对抗性训练的情况下，持续的再训练也可以减少对抗性攻击的效果。虽然敌意攻击可能会损害基于ML的NIDS，但我们的目标是强调，在这一领域的研究和现实世界的实用性之间存在着巨大的差距，需要引起注意。



## **38. Ownership Protection of Generative Adversarial Networks**

生成性对抗网络的所有权保护 cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05233v1) [paper-pdf](http://arxiv.org/pdf/2306.05233v1)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Generative adversarial networks (GANs) have shown remarkable success in image synthesis, making GAN models themselves commercially valuable to legitimate model owners. Therefore, it is critical to technically protect the intellectual property of GANs. Prior works need to tamper with the training set or training process, and they are not robust to emerging model extraction attacks. In this paper, we propose a new ownership protection method based on the common characteristics of a target model and its stolen models. Our method can be directly applicable to all well-trained GANs as it does not require retraining target models. Extensive experimental results show that our new method can achieve the best protection performance, compared to the state-of-the-art methods. Finally, we demonstrate the effectiveness of our method with respect to the number of generations of model extraction attacks, the number of generated samples, different datasets, as well as adaptive attacks.

摘要: 生成性对抗网络(GAN)在图像合成方面取得了显著的成功，使GAN模型本身对合法的模型所有者具有商业价值。因此，从技术上保护甘斯的知识产权至关重要。以往的工作需要篡改训练集或训练过程，并且对新出现的模型提取攻击不是很健壮。本文根据目标模型及其被盗模型的共同特征，提出了一种新的所有权保护方法。我们的方法可以直接适用于所有训练有素的GAN，因为它不需要重新训练目标模型。大量的实验结果表明，与目前最先进的保护方法相比，该方法可以获得最好的保护性能。最后，我们在模型提取攻击的世代数、生成的样本数、不同的数据集以及自适应攻击方面证明了该方法的有效性。



## **39. Boosting Adversarial Transferability by Achieving Flat Local Maxima**

通过实现平坦的局部最大值来提高对手的可转移性 cs.CV

17 pages, 5 figures, 6 tables

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05225v1) [paper-pdf](http://arxiv.org/pdf/2306.05225v1)

**Authors**: Zhijin Ge, Fanhua Shang, Hongying Liu, Yuanyuan Liu, Xiaosen Wang

**Abstract**: Transfer-based attack adopts the adversarial examples generated on the surrogate model to attack various models, making it applicable in the physical world and attracting increasing interest. Recently, various adversarial attacks have emerged to boost adversarial transferability from different perspectives. In this work, inspired by the fact that flat local minima are correlated with good generalization, we assume and empirically validate that adversarial examples at a flat local region tend to have good transferability by introducing a penalized gradient norm to the original loss function. Since directly optimizing the gradient regularization norm is computationally expensive and intractable for generating adversarial examples, we propose an approximation optimization method to simplify the gradient update of the objective function. Specifically, we randomly sample an example and adopt the first-order gradient to approximate the second-order Hessian matrix, which makes computing more efficient by interpolating two Jacobian matrices. Meanwhile, in order to obtain a more stable gradient direction, we randomly sample multiple examples and average the gradients of these examples to reduce the variance due to random sampling during the iterative process. Extensive experimental results on the ImageNet-compatible dataset show that the proposed method can generate adversarial examples at flat local regions, and significantly improve the adversarial transferability on either normally trained models or adversarially trained models than the state-of-the-art attacks.

摘要: 基于转移的攻击采用代理模型上生成的对抗性实例来攻击各种模型，使其适用于物理世界，引起了人们越来越多的兴趣。近年来，各种对抗性攻击层出不穷，从不同的角度提升了对抗性的可转移性。在这项工作中，受平坦局部极小值与良好泛化相关这一事实的启发，我们假设并经验验证了平坦局部区域上的对抗性例子通过在原始损失函数中引入惩罚梯度范数往往具有良好的可转移性。由于直接优化梯度正则化范数的计算量大且难以生成对抗性样本，我们提出了一种近似优化方法来简化目标函数的梯度更新。具体地说，我们随机抽样一个例子，采用一阶梯度逼近二阶Hessian矩阵，通过对两个Jacobian矩阵进行内插来提高计算效率。同时，为了得到一个更稳定的梯度方向，我们对多个样本进行随机采样，并对这些样本的梯度进行平均，以减少迭代过程中随机采样造成的方差。在ImageNet兼容的数据集上的大量实验结果表明，该方法可以在平坦的局部区域生成对抗性实例，并且无论是在正常训练的模型上还是在对抗性训练的模型上，该方法都比最新的攻击方法显著提高了对抗性可转移性。



## **40. PriSampler: Mitigating Property Inference of Diffusion Models**

PriSsamer：减轻扩散模型的性质推断 cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05208v1) [paper-pdf](http://arxiv.org/pdf/2306.05208v1)

**Authors**: Hailong Hu, Jun Pang

**Abstract**: Diffusion models have been remarkably successful in data synthesis. Such successes have also driven diffusion models to apply to sensitive data, such as human face data, but this might bring about severe privacy concerns. In this work, we systematically present the first privacy study about property inference attacks against diffusion models, in which adversaries aim to extract sensitive global properties of the training set from a diffusion model, such as the proportion of the training data for certain sensitive properties. Specifically, we consider the most practical attack scenario: adversaries are only allowed to obtain synthetic data. Under this realistic scenario, we evaluate the property inference attacks on different types of samplers and diffusion models. A broad range of evaluations shows that various diffusion models and their samplers are all vulnerable to property inference attacks. Furthermore, one case study on off-the-shelf pre-trained diffusion models also demonstrates the effectiveness of the attack in practice. Finally, we propose a new model-agnostic plug-in method PriSampler to mitigate the property inference of diffusion models. PriSampler can be directly applied to well-trained diffusion models and support both stochastic and deterministic sampling. Extensive experiments illustrate the effectiveness of our defense and it makes adversaries infer the proportion of properties as close as random guesses. PriSampler also shows its significantly superior performance to diffusion models trained with differential privacy on both model utility and defense performance.

摘要: 扩散模型在数据合成方面取得了显著的成功。这种成功也促使扩散模型应用于敏感数据，如人脸数据，但这可能会带来严重的隐私问题。在这项工作中，我们系统地提出了第一个针对扩散模型的属性推理攻击的隐私研究，其中对手的目标是从扩散模型中提取训练集的敏感全局属性，例如某些敏感属性的训练数据的比例。具体地说，我们考虑了最实际的攻击场景：对手只被允许获取合成数据。在这一现实场景下，我们评估了针对不同类型采样器和扩散模型的属性推理攻击。广泛的评估表明，各种扩散模型及其采样器都容易受到属性推理攻击。此外，一个现成的预训练扩散模型的案例研究也证明了该攻击在实践中的有效性。最后，我们提出了一种新的模型不可知的插件方法PriSsamer来缓解扩散模型的属性推断。PriSsamer可以直接应用于训练良好的扩散模型，并支持随机和确定性抽样。广泛的实验证明了我们防御的有效性，它使对手可以像随机猜测一样推断属性的比例。PriSsamer还显示了其在模型效用和防御性能上显著优于使用差异隐私训练的扩散模型的性能。



## **41. Towards Robust Neural Image Compression: Adversarial Attack and Model Finetuning**

面向稳健的神经图像压缩：对抗性攻击和模型精调 cs.CV

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2112.08691v3) [paper-pdf](http://arxiv.org/pdf/2112.08691v3)

**Authors**: Tong Chen, Zhan Ma

**Abstract**: Deep neural network-based image compression has been extensively studied. However, the model robustness which is crucial to practical application is largely overlooked. We propose to examine the robustness of prevailing learned image compression models by injecting negligible adversarial perturbation into the original source image. Severe distortion in decoded reconstruction reveals the general vulnerability in existing methods regardless of their settings (e.g., network architecture, loss function, quality scale). A variety of defense strategies including geometric self-ensemble based pre-processing, and adversarial training, are investigated against the adversarial attack to improve the model's robustness. Later the defense efficiency is further exemplified in real-life image recompression case studies. Overall, our methodology is simple, effective, and generalizable, making it attractive for developing robust learned image compression solutions. All materials are made publicly accessible at https://njuvision.github.io/RobustNIC for reproducible research.

摘要: 基于深度神经网络的图像压缩已经得到了广泛的研究。然而，对实际应用至关重要的模型稳健性在很大程度上被忽视了。我们建议通过在原始图像中注入可以忽略的对抗性扰动来检验主流学习图像压缩模型的稳健性。解码重建中的严重失真揭示了现有方法中的普遍漏洞，而无论其设置如何(例如，网络体系结构、损失函数、质量尺度)。研究了各种防御策略，包括基于几何自集成的预处理和对抗性训练，以提高模型的鲁棒性。随后，在真实的图像再压缩案例研究中进一步证明了该算法的防御效率。总体而言，我们的方法是简单、有效和可推广的，这使得它对于开发健壮的学习图像压缩解决方案具有吸引力。所有材料都可以在https://njuvision.github.io/RobustNIC上公开获取，以进行可重复的研究。



## **42. Adversarial Sample Detection Through Neural Network Transport Dynamics**

基于神经网络传输动力学的对抗性样本检测 cs.LG

ECML PKDD 2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04252v2) [paper-pdf](http://arxiv.org/pdf/2306.04252v2)

**Authors**: Skander Karkar, Patrick Gallinari, Alain Rakotomamonjy

**Abstract**: We propose a detector of adversarial samples that is based on the view of neural networks as discrete dynamic systems. The detector tells clean inputs from abnormal ones by comparing the discrete vector fields they follow through the layers. We also show that regularizing this vector field during training makes the network more regular on the data distribution's support, thus making the activations of clean inputs more distinguishable from those of abnormal ones. Experimentally, we compare our detector favorably to other detectors on seen and unseen attacks, and show that the regularization of the network's dynamics improves the performance of adversarial detectors that use the internal embeddings as inputs, while also improving test accuracy.

摘要: 基于神经网络作为离散动态系统的观点，我们提出了一种敌意样本检测器。探测器通过比较它们在各层中遵循的离散向量场来区分干净的输入和异常的输入。我们还表明，在训练过程中对该向量场进行正则化，使网络对数据分布的支持更加规则，从而使干净输入的激活与异常输入的激活更容易区分。在实验上，我们将我们的检测器在看得见和看不见的攻击上与其他检测器进行了有利的比较，结果表明，网络动态的正则化改善了使用内部嵌入作为输入的对抗性检测器的性能，同时也提高了测试精度。



## **43. Toward Enhanced Robustness in Unsupervised Graph Representation Learning: A Graph Information Bottleneck Perspective**

基于图信息瓶颈的无监督图表示学习增强稳健性研究 cs.LG

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2201.08557v2) [paper-pdf](http://arxiv.org/pdf/2201.08557v2)

**Authors**: Jihong Wang, Minnan Luo, Jundong Li, Ziqi Liu, Jun Zhou, Qinghua Zheng

**Abstract**: Recent studies have revealed that GNNs are vulnerable to adversarial attacks. Most existing robust graph learning methods measure model robustness based on label information, rendering them infeasible when label information is not available. A straightforward direction is to employ the widely used Infomax technique from typical Unsupervised Graph Representation Learning (UGRL) to learn robust unsupervised representations. Nonetheless, directly transplanting the Infomax technique from typical UGRL to robust UGRL may involve a biased assumption. In light of the limitation of Infomax, we propose a novel unbiased robust UGRL method called Robust Graph Information Bottleneck (RGIB), which is grounded in the Information Bottleneck (IB) principle. Our RGIB attempts to learn robust node representations against adversarial perturbations by preserving the original information in the benign graph while eliminating the adversarial information in the adversarial graph. There are mainly two challenges to optimize RGIB: 1) high complexity of adversarial attack to perturb node features and graph structure jointly in the training procedure; 2) mutual information estimation upon adversarially attacked graphs. To tackle these problems, we further propose an efficient adversarial training strategy with only feature perturbations and an effective mutual information estimator with subgraph-level summary. Moreover, we theoretically establish a connection between our proposed RGIB and the robustness of downstream classifiers, revealing that RGIB can provide a lower bound on the adversarial risk of downstream classifiers. Extensive experiments over several benchmarks and downstream tasks demonstrate the effectiveness and superiority of our proposed method.

摘要: 最近的研究表明，GNN很容易受到对抗性攻击。现有的大多数稳健图学习方法都是基于标签信息来衡量模型的稳健性，当标签信息不可用时，这些方法是不可行的。一个简单的方向是从典型的无监督图表示学习(UGRL)中使用广泛使用的Infomax技术来学习健壮的无监督表示。然而，将Infomax技术从典型的UGRL直接移植到健壮的UGRL可能涉及一个有偏见的假设。针对Infomax的局限性，基于信息瓶颈(IB)原理，提出了一种新的无偏鲁棒图信息瓶颈(RGIB)方法。我们的RGIB试图通过保留良性图中的原始信息，同时消除对抗性图中的对抗性信息，来学习针对对抗性扰动的健壮节点表示。RGIB的优化主要有两个挑战：1)在训练过程中，对抗性攻击对节点特征和图结构的联合扰动的高复杂性；2)对抗性攻击图的互信息估计。为了解决这些问题，我们进一步提出了一种有效的只有特征扰动的对抗性训练策略和一个有效的子图级别摘要的互信息估计器。此外，我们在理论上建立了我们提出的RGIB与下游分类器的稳健性之间的联系，揭示了RGIB可以提供下游分类器对抗风险的下界。在多个基准测试和下游任务上的大量实验证明了该方法的有效性和优越性。



## **44. Generalizable Lightweight Proxy for Robust NAS against Diverse Perturbations**

针对不同扰动的健壮NAS通用型轻量级代理 cs.LG

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.05031v1) [paper-pdf](http://arxiv.org/pdf/2306.05031v1)

**Authors**: Hyeonjeong Ha, Minseon Kim, Sung Ju Hwang

**Abstract**: Recent neural architecture search (NAS) frameworks have been successful in finding optimal architectures for given conditions (e.g., performance or latency). However, they search for optimal architectures in terms of their performance on clean images only, while robustness against various types of perturbations or corruptions is crucial in practice. Although there exist several robust NAS frameworks that tackle this issue by integrating adversarial training into one-shot NAS, however, they are limited in that they only consider robustness against adversarial attacks and require significant computational resources to discover optimal architectures for a single task, which makes them impractical in real-world scenarios. To address these challenges, we propose a novel lightweight robust zero-cost proxy that considers the consistency across features, parameters, and gradients of both clean and perturbed images at the initialization state. Our approach facilitates an efficient and rapid search for neural architectures capable of learning generalizable features that exhibit robustness across diverse perturbations. The experimental results demonstrate that our proxy can rapidly and efficiently search for neural architectures that are consistently robust against various perturbations on multiple benchmark datasets and diverse search spaces, largely outperforming existing clean zero-shot NAS and robust NAS with reduced search cost.

摘要: 最近的神经体系结构搜索(NAS)框架已经成功地找到了针对给定条件(例如，性能或延迟)的最佳体系结构。然而，他们只根据在干净图像上的性能来寻找最佳体系结构，而对各种类型的扰动或损坏的健壮性在实践中是至关重要的。虽然有几个健壮的NAS框架通过将对抗性训练集成到一次性NAS来解决这个问题，但是它们的局限性在于它们只考虑针对对抗性攻击的健壮性，并且需要大量的计算资源来发现单个任务的最佳架构，这使得它们在现实世界的场景中不切实际。为了应对这些挑战，我们提出了一种新的轻量级健壮零代价代理，该代理在初始化状态下考虑了干净图像和扰动图像的特征、参数和梯度的一致性。我们的方法有助于高效和快速地搜索能够学习在不同扰动下表现出健壮性的可概括特征的神经体系结构。实验结果表明，我们的代理能够快速有效地搜索到在多个基准数据集和不同搜索空间上对各种扰动具有一致健壮性的神经体系结构，大大优于现有的干净的零镜头NAS和健壮的NAS，并且降低了搜索成本。



## **45. A Melting Pot of Evolution and Learning**

进化和学习的熔炉 cs.NE

To Appear in Proceedings of Genetic Programming Theory & Practice XX,  2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04971v1) [paper-pdf](http://arxiv.org/pdf/2306.04971v1)

**Authors**: Moshe Sipper, Achiya Elyasaf, Tomer Halperin, Zvika Haramaty, Raz Lapid, Eyal Segal, Itai Tzruia, Snir Vitrack Tamam

**Abstract**: We survey eight recent works by our group, involving the successful blending of evolutionary algorithms with machine learning and deep learning: 1. Binary and Multinomial Classification through Evolutionary Symbolic Regression, 2. Classy Ensemble: A Novel Ensemble Algorithm for Classification, 3. EC-KitY: Evolutionary Computation Tool Kit in Python, 4. Evolution of Activation Functions for Deep Learning-Based Image Classification, 5. Adaptive Combination of a Genetic Algorithm and Novelty Search for Deep Neuroevolution, 6. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks, 7. Foiling Explanations in Deep Neural Networks, 8. Patch of Invisibility: Naturalistic Black-Box Adversarial Attacks on Object Detectors.

摘要: 我们回顾了我们团队最近的八项工作，涉及进化算法与机器学习和深度学习的成功结合：1.基于进化符号回归的二进制和多项式分类；2.有类集成：一种新的分类集成算法；3.进化计算工具包；4.基于深度学习的图像分类激活函数的进化；5.遗传算法和新颖性搜索的自适应组合用于深度神经进化；6.一种进化的、无梯度的、查询高效的黑盒算法，用于在深度网络中生成敌意实例；7.挫败深层神经网络中的解释，8.隐形补丁：对物体探测器的自然主义黑匣子对抗性攻击。



## **46. FedMLSecurity: A Benchmark for Attacks and Defenses in Federated Learning and LLMs**

FedMLSecurity：联合学习和LLMS中攻击和防御的基准 cs.CR

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04959v1) [paper-pdf](http://arxiv.org/pdf/2306.04959v1)

**Authors**: Shanshan Han, Baturalp Buyukates, Zijian Hu, Han Jin, Weizhao Jin, Lichao Sun, Xiaoyang Wang, Chulin Xie, Kai Zhang, Qifan Zhang, Yuhui Zhang, Chaoyang He, Salman Avestimehr

**Abstract**: This paper introduces FedMLSecurity, a benchmark that simulates adversarial attacks and corresponding defense mechanisms in Federated Learning (FL). As an integral module of the open-sourced library FedML that facilitates FL algorithm development and performance comparison, FedMLSecurity enhances the security assessment capacity of FedML. FedMLSecurity comprises two principal components: FedMLAttacker, which simulates attacks injected into FL training, and FedMLDefender, which emulates defensive strategies designed to mitigate the impacts of the attacks. FedMLSecurity is open-sourced 1 and is customizable to a wide range of machine learning models (e.g., Logistic Regression, ResNet, GAN, etc.) and federated optimizers (e.g., FedAVG, FedOPT, FedNOVA, etc.). Experimental evaluations in this paper also demonstrate the ease of application of FedMLSecurity to Large Language Models (LLMs), further reinforcing its versatility and practical utility in various scenarios.

摘要: 本文介绍了联邦学习中模拟对抗性攻击和相应防御机制的基准测试FedMLSecurity。FedMLSecurity作为开源库FedML中一个不可或缺的模块，方便了FL算法的开发和性能比较，增强了FedML的安全评估能力。FedMLSecurity由两个主要组件组成：FedMLAtTacker，它模拟注入FL训练的攻击，以及FedMLDefender，它模拟旨在减轻攻击影响的防御策略。FedMLSecurity是开源的1，可针对多种机器学习模型(如Logistic回归、ResNet、GAN等)进行定制。以及联合优化器(例如，FedAVG、FedOPT、FedNOVA等)。本文的实验评估也证明了FedMLSecurity在大型语言模型(LLM)中的易用性，进一步增强了它在各种场景下的通用性和实用性。



## **47. Degraded Polygons Raise Fundamental Questions of Neural Network Perception**

退化的多边形提出了神经网络感知的基本问题 cs.CV

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04955v1) [paper-pdf](http://arxiv.org/pdf/2306.04955v1)

**Authors**: Leonard Tang, Dan Ley

**Abstract**: It is well-known that modern computer vision systems often exhibit behaviors misaligned with those of humans: from adversarial attacks to image corruptions, deep learning vision models suffer in a variety of settings that humans capably handle. In light of these phenomena, here we introduce another, orthogonal perspective studying the human-machine vision gap. We revisit the task of recovering images under degradation, first introduced over 30 years ago in the Recognition-by-Components theory of human vision. Specifically, we study the performance and behavior of neural networks on the seemingly simple task of classifying regular polygons at varying orders of degradation along their perimeters. To this end, we implement the Automated Shape Recoverability Test for rapidly generating large-scale datasets of perimeter-degraded regular polygons, modernizing the historically manual creation of image recoverability experiments. We then investigate the capacity of neural networks to recognize and recover such degraded shapes when initialized with different priors. Ultimately, we find that neural networks' behavior on this simple task conflicts with human behavior, raising a fundamental question of the robustness and learning capabilities of modern computer vision models.

摘要: 众所周知，现代计算机视觉系统经常表现出与人类不一致的行为：从敌意攻击到图像损坏，深度学习视觉模型在人类能够处理的各种环境中受到影响。鉴于这些现象，我们在这里介绍了另一个研究人机视觉鸿沟的正交视角。我们回顾了30多年前在人类视觉的成分识别理论中首次引入的恢复退化图像的任务。具体地说，我们研究了神经网络在一项看似简单的任务中的性能和行为，该任务是对沿其周长以不同降级顺序的规则多边形进行分类。为此，我们实现了自动形状可恢复性测试，用于快速生成周长退化的规则多边形的大规模数据集，使历史上手动创建图像可恢复性实验的过程现代化。然后，我们研究了当用不同的先验进行初始化时，神经网络识别和恢复这些退化形状的能力。最终，我们发现神经网络在这一简单任务中的行为与人类行为相冲突，这引发了现代计算机视觉模型的稳健性和学习能力的根本问题。



## **48. Open Set Relation Extraction via Unknown-Aware Training**

基于未知感知训练的开集关系提取 cs.CL

Accepted by ACL2023

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04950v1) [paper-pdf](http://arxiv.org/pdf/2306.04950v1)

**Authors**: Jun Zhao, Xin Zhao, Wenyu Zhan, Qi Zhang, Tao Gui, Zhongyu Wei, Yunwen Chen, Xiang Gao, Xuanjing Huang

**Abstract**: The existing supervised relation extraction methods have achieved impressive performance in a closed-set setting, where the relations during both training and testing remain the same. In a more realistic open-set setting, unknown relations may appear in the test set. Due to the lack of supervision signals from unknown relations, a well-performing closed-set relation extractor can still confidently misclassify them into known relations. In this paper, we propose an unknown-aware training method, regularizing the model by dynamically synthesizing negative instances. To facilitate a compact decision boundary, ``difficult'' negative instances are necessary. Inspired by text adversarial attacks, we adaptively apply small but critical perturbations to original training instances and thus synthesizing negative instances that are more likely to be mistaken by the model as known relations. Experimental results show that this method achieves SOTA unknown relation detection without compromising the classification of known relations.

摘要: 现有的有监督关系提取方法在闭集环境下取得了令人印象深刻的性能，在闭集环境下，训练和测试期间的关系保持不变。在更真实的开放集设置中，未知关系可能出现在测试集中。由于缺乏来自未知关系的监督信号，性能良好的闭集关系抽取器仍然可以自信地将它们错误分类为已知关系。在本文中，我们提出了一种未知感知训练方法，通过动态合成否定实例来正则化模型。为了便于紧凑的决策边界，“困难”的否定实例是必要的。受文本对抗攻击的启发，我们自适应地对原始训练实例应用小但关键的扰动，从而合成更容易被模型误认为已知关系的否定实例。实验结果表明，该方法在不影响已知关系分类的前提下，实现了SOTA未知关系的检测。



## **49. Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework**

弥合简历和NLP之间的差距！一种基于梯度的文本对抗攻击框架 cs.CL

Accepted to Findings of ACL 2023. Codes are available at:  https://github.com/Phantivia/T-PGD

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2110.15317v4) [paper-pdf](http://arxiv.org/pdf/2110.15317v4)

**Authors**: Lifan Yuan, Yichi Zhang, Yangyi Chen, Wei Wei

**Abstract**: Despite recent success on various tasks, deep learning techniques still perform poorly on adversarial examples with small perturbations. While optimization-based methods for adversarial attacks are well-explored in the field of computer vision, it is impractical to directly apply them in natural language processing due to the discrete nature of the text. To address the problem, we propose a unified framework to extend the existing optimization-based adversarial attack methods in the vision domain to craft textual adversarial samples. In this framework, continuously optimized perturbations are added to the embedding layer and amplified in the forward propagation process. Then the final perturbed latent representations are decoded with a masked language model head to obtain potential adversarial samples. In this paper, we instantiate our framework with an attack algorithm named Textual Projected Gradient Descent (T-PGD). We find our algorithm effective even using proxy gradient information. Therefore, we perform the more challenging transfer black-box attack and conduct comprehensive experiments to evaluate our attack algorithm with several models on three benchmark datasets. Experimental results demonstrate that our method achieves overall better performance and produces more fluent and grammatical adversarial samples compared to strong baseline methods. The code and data are available at \url{https://github.com/Phantivia/T-PGD}.

摘要: 尽管最近在各种任务上取得了成功，但深度学习技术在具有小扰动的对抗性例子中仍然表现不佳。虽然基于优化的对抗性攻击方法在计算机视觉领域得到了很好的探索，但由于文本的离散性质，将其直接应用于自然语言处理是不切实际的。为了解决这个问题，我们提出了一个统一的框架来扩展现有的视觉领域基于优化的对抗性攻击方法，以制作文本对抗性样本。在该框架中，不断优化的扰动被添加到嵌入层，并在前向传播过程中被放大。然后，使用掩蔽语言模型头部对最终扰动的潜在表示进行解码，以获得潜在的对抗性样本。在本文中，我们使用一种名为文本投影梯度下降(T-PGD)的攻击算法来实例化我们的框架。我们发现我们的算法即使使用代理梯度信息也是有效的。因此，我们执行了更具挑战性的转移黑盒攻击，并在三个基准数据集上用几个模型进行了全面的实验来评估我们的攻击算法。实验结果表明，与强基线方法相比，我们的方法在总体上取得了更好的性能，并产生了更多流畅和语法上的对抗性样本。代码和数据可在\url{https://github.com/Phantivia/T-PGD}.



## **50. Expanding Scope: Adapting English Adversarial Attacks to Chinese**

扩展范围：将英语对抗性攻击改编为中文 cs.CL

11 pages; in ACL23 TrustNLP 2023: TrustNLP: Third Workshop on  Trustworthy Natural Language Processing Colocated with the Annual Conference  of the Association for Computational Linguistics (ACL 2023)

**SubmitDate**: 2023-06-08    [abs](http://arxiv.org/abs/2306.04874v1) [paper-pdf](http://arxiv.org/pdf/2306.04874v1)

**Authors**: Hanyu Liu, Chengyuan Cai, Yanjun Qi

**Abstract**: Recent studies have revealed that NLP predictive models are vulnerable to adversarial attacks. Most existing studies focused on designing attacks to evaluate the robustness of NLP models in the English language alone. Literature has seen an increasing need for NLP solutions for other languages. We, therefore, ask one natural question: whether state-of-the-art (SOTA) attack methods generalize to other languages. This paper investigates how to adapt SOTA adversarial attack algorithms in English to the Chinese language. Our experiments show that attack methods previously applied to English NLP can generate high-quality adversarial examples in Chinese when combined with proper text segmentation and linguistic constraints. In addition, we demonstrate that the generated adversarial examples can achieve high fluency and semantic consistency by focusing on the Chinese language's morphology and phonology, which in turn can be used to improve the adversarial robustness of Chinese NLP models.

摘要: 最近的研究表明，NLP预测模型容易受到对手攻击。现有的大多数研究都集中在设计攻击来评估仅在英语中的NLP模型的稳健性。文献表明，对其他语言的NLP解决方案的需求越来越大。因此，我们自然会问一个问题：最先进的(SOTA)攻击方法是否适用于其他语言。本文研究了如何将英文的SOTA对抗性攻击算法移植到中文中。我们的实验表明，以前应用于英语自然语言处理的攻击方法，如果结合适当的文本切分和语言约束，可以生成高质量的中文对抗性实例。此外，我们还证明了生成的对抗性实例可以通过关注汉语的词法和音位来达到较高的流畅度和语义一致性，从而可以用于提高中文自然语言处理模型的对抗性健壮性。



