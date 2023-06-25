# Latest Adversarial Attack Papers
**update at 2023-06-25 16:52:04**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Evading Forensic Classifiers with Attribute-Conditioned Adversarial Faces**

基于属性条件的对抗性面孔规避法医分类器 cs.CV

Accepted in CVPR 2023. Project page:  https://koushiksrivats.github.io/face_attribute_attack/

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13091v1) [paper-pdf](http://arxiv.org/pdf/2306.13091v1)

**Authors**: Fahad Shamshad, Koushik Srivatsan, Karthik Nandakumar

**Abstract**: The ability of generative models to produce highly realistic synthetic face images has raised security and ethical concerns. As a first line of defense against such fake faces, deep learning based forensic classifiers have been developed. While these forensic models can detect whether a face image is synthetic or real with high accuracy, they are also vulnerable to adversarial attacks. Although such attacks can be highly successful in evading detection by forensic classifiers, they introduce visible noise patterns that are detectable through careful human scrutiny. Additionally, these attacks assume access to the target model(s) which may not always be true. Attempts have been made to directly perturb the latent space of GANs to produce adversarial fake faces that can circumvent forensic classifiers. In this work, we go one step further and show that it is possible to successfully generate adversarial fake faces with a specified set of attributes (e.g., hair color, eye size, race, gender, etc.). To achieve this goal, we leverage the state-of-the-art generative model StyleGAN with disentangled representations, which enables a range of modifications without leaving the manifold of natural images. We propose a framework to search for adversarial latent codes within the feature space of StyleGAN, where the search can be guided either by a text prompt or a reference image. We also propose a meta-learning based optimization strategy to achieve transferable performance on unknown target models. Extensive experiments demonstrate that the proposed approach can produce semantically manipulated adversarial fake faces, which are true to the specified attribute set and can successfully fool forensic face classifiers, while remaining undetectable by humans. Code: https://github.com/koushiksrivats/face_attribute_attack.

摘要: 生成模型生成高度逼真的合成人脸图像的能力引发了安全和伦理方面的担忧。作为对抗这种假脸的第一道防线，基于深度学习的法医分类器已经被开发出来。虽然这些取证模型可以高精度地检测人脸图像是合成的还是真实的，但它们也容易受到对手的攻击。虽然这类攻击可以非常成功地躲避法医分类器的检测，但它们引入了可通过仔细的人类检查检测到的可见噪声模式。此外，这些攻击假定可以访问目标模型(S)，但这可能并不总是正确的。有人试图直接扰乱Gans的潜在空间，以产生可以绕过法医分类器的敌对假面。在这项工作中，我们更进一步，证明了可以成功地生成具有特定属性集(例如，头发颜色、眼睛大小、种族、性别等)的敌意伪脸。为了实现这一目标，我们利用最先进的生成模型StyleGAN和分离的表示法，可以在不离开各种自然图像的情况下进行一系列修改。提出了一种在StyleGAN的特征空间内搜索敌意潜在码的框架，该框架可以通过文本提示或参考图像来指导搜索。我们还提出了一种基于元学习的优化策略，以实现在未知目标模型上的可转移性能。大量的实验表明，该方法可以生成符合指定属性集的语义操纵的敌意伪脸，并且能够成功地欺骗取证人脸分类器，而又不被人类发现。代码：https://github.com/koushiksrivats/face_attribute_attack.



## **2. Impacts and Risk of Generative AI Technology on Cyber Defense**

产生式人工智能技术对网络防御的影响和风险 cs.CR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.13033v1) [paper-pdf](http://arxiv.org/pdf/2306.13033v1)

**Authors**: Subash Neupane, Ivan A. Fernandez, Sudip Mittal, Shahram Rahimi

**Abstract**: Generative Artificial Intelligence (GenAI) has emerged as a powerful technology capable of autonomously producing highly realistic content in various domains, such as text, images, audio, and videos. With its potential for positive applications in creative arts, content generation, virtual assistants, and data synthesis, GenAI has garnered significant attention and adoption. However, the increasing adoption of GenAI raises concerns about its potential misuse for crafting convincing phishing emails, generating disinformation through deepfake videos, and spreading misinformation via authentic-looking social media posts, posing a new set of challenges and risks in the realm of cybersecurity. To combat the threats posed by GenAI, we propose leveraging the Cyber Kill Chain (CKC) to understand the lifecycle of cyberattacks, as a foundational model for cyber defense. This paper aims to provide a comprehensive analysis of the risk areas introduced by the offensive use of GenAI techniques in each phase of the CKC framework. We also analyze the strategies employed by threat actors and examine their utilization throughout different phases of the CKC, highlighting the implications for cyber defense. Additionally, we propose GenAI-enabled defense strategies that are both attack-aware and adaptive. These strategies encompass various techniques such as detection, deception, and adversarial training, among others, aiming to effectively mitigate the risks posed by GenAI-induced cyber threats.

摘要: 生成性人工智能(GenAI)已经成为一项强大的技术，能够在文本、图像、音频和视频等各个领域自主产生高度逼真的内容。凭借其在创意艺术、内容生成、虚拟助手和数据合成方面的积极应用潜力，GenAI获得了极大的关注和采用。然而，越来越多的人使用GenAI引发了人们的担忧，即它可能被滥用来制作令人信服的钓鱼电子邮件，通过深度虚假视频产生虚假信息，以及通过看起来真实的社交媒体帖子传播错误信息，给网络安全领域带来了一系列新的挑战和风险。为了应对GenAI构成的威胁，我们建议利用网络杀伤链(CKC)来了解网络攻击的生命周期，作为网络防御的基础模型。本文旨在全面分析在CKC框架的每个阶段中攻击性使用GenAI技术所带来的风险领域。我们还分析了威胁参与者使用的策略，并检查了它们在CKC不同阶段的使用情况，强调了对网络防御的影响。此外，我们还提出了支持GenAI的攻击感知和自适应防御策略。这些战略包括各种技术，如检测、欺骗和对抗性训练等，旨在有效地缓解GenAI引发的网络威胁带来的风险。



## **3. AI Security for Geoscience and Remote Sensing: Challenges and Future Trends**

地球科学和遥感的人工智能安全：挑战和未来趋势 cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2212.09360v2) [paper-pdf](http://arxiv.org/pdf/2212.09360v2)

**Authors**: Yonghao Xu, Tao Bai, Weikang Yu, Shizhen Chang, Peter M. Atkinson, Pedram Ghamisi

**Abstract**: Recent advances in artificial intelligence (AI) have significantly intensified research in the geoscience and remote sensing (RS) field. AI algorithms, especially deep learning-based ones, have been developed and applied widely to RS data analysis. The successful application of AI covers almost all aspects of Earth observation (EO) missions, from low-level vision tasks like super-resolution, denoising and inpainting, to high-level vision tasks like scene classification, object detection and semantic segmentation. While AI techniques enable researchers to observe and understand the Earth more accurately, the vulnerability and uncertainty of AI models deserve further attention, considering that many geoscience and RS tasks are highly safety-critical. This paper reviews the current development of AI security in the geoscience and RS field, covering the following five important aspects: adversarial attack, backdoor attack, federated learning, uncertainty and explainability. Moreover, the potential opportunities and trends are discussed to provide insights for future research. To the best of the authors' knowledge, this paper is the first attempt to provide a systematic review of AI security-related research in the geoscience and RS community. Available code and datasets are also listed in the paper to move this vibrant field of research forward.

摘要: 人工智能(AI)的最新进展极大地加强了地学和遥感(RS)领域的研究。人工智能算法，特别是基于深度学习的人工智能算法在遥感数据分析中得到了广泛的应用。人工智能的成功应用几乎涵盖了地球观测(EO)任务的方方面面，从超分辨率、去噪和修复等低层视觉任务，到场景分类、目标检测和语义分割等高级视觉任务。虽然人工智能技术使研究人员能够更准确地观察和了解地球，但考虑到许多地学和遥感任务是高度安全关键的，人工智能模型的脆弱性和不确定性值得进一步关注。本文回顾了人工智能安全在地学和遥感领域的发展现状，包括对抗性攻击、后门攻击、联邦学习、不确定性和可解释性五个重要方面。此外，还讨论了潜在的机会和趋势，为未来的研究提供了见解。据作者所知，本文是第一次对地学和遥感社区中与人工智能安全相关的研究进行系统回顾。文中还列出了可用的代码和数据集，以推动这一充满活力的研究领域向前发展。



## **4. Robust Semantic Segmentation: Strong Adversarial Attacks and Fast Training of Robust Models**

稳健语义分割：强对抗性攻击和稳健模型的快速训练 cs.CV

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12941v1) [paper-pdf](http://arxiv.org/pdf/2306.12941v1)

**Authors**: Francesco Croce, Naman D Singh, Matthias Hein

**Abstract**: While a large amount of work has focused on designing adversarial attacks against image classifiers, only a few methods exist to attack semantic segmentation models. We show that attacking segmentation models presents task-specific challenges, for which we propose novel solutions. Our final evaluation protocol outperforms existing methods, and shows that those can overestimate the robustness of the models. Additionally, so far adversarial training, the most successful way for obtaining robust image classifiers, could not be successfully applied to semantic segmentation. We argue that this is because the task to be learned is more challenging, and requires significantly higher computational effort than for image classification. As a remedy, we show that by taking advantage of recent advances in robust ImageNet classifiers, one can train adversarially robust segmentation models at limited computational cost by fine-tuning robust backbones.

摘要: 虽然大量的工作集中在设计针对图像分类器的对抗性攻击，但只有少数方法存在攻击语义分割模型。我们发现攻击分段模型带来了特定于任务的挑战，为此我们提出了新的解决方案。我们的最终评估协议的性能优于现有的方法，并表明这些方法可能高估了模型的稳健性。此外，对抗性训练是获得稳健的图像分类器的最成功的方法，到目前为止还不能成功地应用于语义分割。我们认为这是因为要学习的任务更具挑战性，并且需要比图像分类更高的计算工作量。作为补救措施，我们表明，通过利用健壮ImageNet分类器的最新进展，可以通过微调健壮的主干来以有限的计算成本训练相反的健壮分割模型。



## **5. Cross-lingual Cross-temporal Summarization: Dataset, Models, Evaluation**

跨语言跨期摘要：数据集、模型、评价 cs.CL

Work in progress

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12916v1) [paper-pdf](http://arxiv.org/pdf/2306.12916v1)

**Authors**: Ran Zhang, Jihed Ouni, Steffen Eger

**Abstract**: While summarization has been extensively researched in natural language processing (NLP), cross-lingual cross-temporal summarization (CLCTS) is a largely unexplored area that has the potential to improve cross-cultural accessibility, information sharing, and understanding. This paper comprehensively addresses the CLCTS task, including dataset creation, modeling, and evaluation. We build the first CLCTS corpus, leveraging historical fictive texts and Wikipedia summaries in English and German, and examine the effectiveness of popular transformer end-to-end models with different intermediate task finetuning tasks. Additionally, we explore the potential of ChatGPT for CLCTS as a summarizer and an evaluator. Overall, we report evaluations from humans, ChatGPT, and several recent automatic evaluation metrics where we find our intermediate task finetuned end-to-end models generate bad to moderate quality summaries; ChatGPT as a summarizer (without any finetuning) provides moderate to good quality outputs and as an evaluator correlates moderately with human evaluations though it is prone to giving lower scores. ChatGPT also seems to be very adept at normalizing historical text. We finally test ChatGPT in a scenario with adversarially attacked and unseen source documents and find that ChatGPT is better at omission and entity swap than negating against its prior knowledge.

摘要: 摘要在自然语言处理(NLP)领域得到了广泛的研究，而跨语言的跨时序摘要(CLCTS)在很大程度上是一个未被开发的领域，它有可能改善跨文化的可获得性、信息共享和理解。本文全面介绍了CLCTS的任务，包括数据集的创建、建模和评估。我们构建了第一个CLCTS语料库，利用英语和德语的历史虚构文本和维基百科摘要，并检查了具有不同中间任务精调任务的流行变压器端到端模型的有效性。此外，我们还探讨了ChatGPT作为CLCTS的摘要和评价器的潜力。总体而言，我们报告了来自人工、ChatGPT和最近几个自动评估指标的评估，在这些评估指标中，我们发现我们的中间任务微调的端到端模型生成了较差到中等质量的摘要；ChatGPT作为汇总器(没有任何微调)提供了中等到良好的质量输出，并且作为评估者与人工评估适度相关，尽管它倾向于给出较低的分数。ChatGPT似乎也非常擅长将历史文本正常化。最后，我们在恶意攻击和不可见源文档的情况下对ChatGPT进行了测试，发现ChatGPT在遗漏和实体交换方面比否认其先验知识更好。



## **6. On the explainable properties of 1-Lipschitz Neural Networks: An Optimal Transport Perspective**

1-Lipschitz神经网络的可解释性：最优传输观点 cs.AI

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2206.06854v2) [paper-pdf](http://arxiv.org/pdf/2206.06854v2)

**Authors**: Mathieu Serrurier, Franck Mamalet, Thomas Fel, Louis Béthune, Thibaut Boissin

**Abstract**: Input gradients have a pivotal role in a variety of applications, including adversarial attack algorithms for evaluating model robustness, explainable AI techniques for generating Saliency Maps, and counterfactual explanations. However, Saliency Maps generated by traditional neural networks are often noisy and provide limited insights. In this paper, we demonstrate that, on the contrary, the Saliency Maps of 1-Lipschitz neural networks, learnt with the dual loss of an optimal transportation problem, exhibit desirable XAI properties: They are highly concentrated on the essential parts of the image with low noise, significantly outperforming state-of-the-art explanation approaches across various models and metrics. We also prove that these maps align unprecedentedly well with human explanations on ImageNet. To explain the particularly beneficial properties of the Saliency Map for such models, we prove this gradient encodes both the direction of the transportation plan and the direction towards the nearest adversarial attack. Following the gradient down to the decision boundary is no longer considered an adversarial attack, but rather a counterfactual explanation that explicitly transports the input from one class to another. Thus, Learning with such a loss jointly optimizes the classification objective and the alignment of the gradient , i.e. the Saliency Map, to the transportation plan direction. These networks were previously known to be certifiably robust by design, and we demonstrate that they scale well for large problems and models, and are tailored for explainability using a fast and straightforward method.

摘要: 输入梯度在各种应用中具有举足轻重的作用，包括用于评估模型稳健性的对抗性攻击算法、用于生成显著图的可解释人工智能技术以及反事实解释。然而，传统神经网络生成的显著图往往噪声较大，提供的洞察力有限。相反，我们证明了在最优传输问题的对偶损失下学习的1-Lipschitz神经网络的显著图表现出理想的XAI性质：它们高度集中在低噪声的图像的基本部分，在各种模型和度量上显著优于最新的解释方法。我们还证明，这些地图与人们在ImageNet上的解释前所未有地吻合。为了解释显著图对这类模型特别有益的特性，我们证明了这种梯度既编码了运输计划的方向，也编码了指向最近的敌方攻击的方向。沿着梯度向下到决策边界不再被认为是对抗性攻击，而是一种反事实的解释，明确地将输入从一个类别传输到另一个类别。因此，具有这样的损失的学习联合优化了分类目标和梯度(即显著图)与运输计划方向的对准。众所周知，这些网络在设计上是可靠的，我们展示了它们对于大型问题和模型的良好伸缩性，并使用快速而直接的方法针对可解释性进行了定制。



## **7. Conditional Generators for Limit Order Book Environments: Explainability, Challenges, and Robustness**

极限订单簿环境的条件生成器：可解释性、挑战性和稳健性 q-fin.TR

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12806v1) [paper-pdf](http://arxiv.org/pdf/2306.12806v1)

**Authors**: Andrea Coletta, Joseph Jerome, Rahul Savani, Svitlana Vyetrenko

**Abstract**: Limit order books are a fundamental and widespread market mechanism. This paper investigates the use of conditional generative models for order book simulation. For developing a trading agent, this approach has drawn recent attention as an alternative to traditional backtesting due to its ability to react to the presence of the trading agent. Using a state-of-the-art CGAN (from Coletta et al. (2022)), we explore its dependence upon input features, which highlights both strengths and weaknesses. To do this, we use "adversarial attacks" on the model's features and its mechanism. We then show how these insights can be used to improve the CGAN, both in terms of its realism and robustness. We finish by laying out a roadmap for future work.

摘要: 限价令是一种基本的、广泛存在的市场机制。本文研究了条件生成模型在订单模拟中的应用。对于开发交易代理，这种方法最近引起了人们的注意，作为传统回溯测试的替代方法，因为它能够对交易代理的存在做出反应。使用最先进的CGAN(来自Coletta等人(2022))，我们探索了它对输入特征的依赖，这既强调了优点，也强调了缺点。为此，我们对模型的特点及其机制进行了“对抗性攻击”。然后，我们展示了如何使用这些见解来改进CGAN，无论是在其真实性和健壮性方面。最后，我们为未来的工作制定了路线图。



## **8. Towards quantum enhanced adversarial robustness in machine learning**

机器学习中的量子增强对抗鲁棒性 quant-ph

10 Pages, 4 Figures

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12688v1) [paper-pdf](http://arxiv.org/pdf/2306.12688v1)

**Authors**: Maxwell T. West, Shu-Lok Tsang, Jia S. Low, Charles D. Hill, Christopher Leckie, Lloyd C. L. Hollenberg, Sarah M. Erfani, Muhammad Usman

**Abstract**: Machine learning algorithms are powerful tools for data driven tasks such as image classification and feature detection, however their vulnerability to adversarial examples - input samples manipulated to fool the algorithm - remains a serious challenge. The integration of machine learning with quantum computing has the potential to yield tools offering not only better accuracy and computational efficiency, but also superior robustness against adversarial attacks. Indeed, recent work has employed quantum mechanical phenomena to defend against adversarial attacks, spurring the rapid development of the field of quantum adversarial machine learning (QAML) and potentially yielding a new source of quantum advantage. Despite promising early results, there remain challenges towards building robust real-world QAML tools. In this review we discuss recent progress in QAML and identify key challenges. We also suggest future research directions which could determine the route to practicality for QAML approaches as quantum computing hardware scales up and noise levels are reduced.

摘要: 机器学习算法是用于图像分类和特征检测等数据驱动任务的强大工具，但它们在敌意示例(输入样本被操纵以愚弄算法)面前的脆弱性仍然是一个严峻的挑战。机器学习与量子计算的结合有可能产生不仅具有更好的准确性和计算效率，而且具有针对对手攻击的卓越稳健性的工具。事实上，最近的工作利用量子力学现象来防御对抗性攻击，刺激了量子对抗性机器学习(QAML)领域的快速发展，并可能产生新的量子优势来源。尽管早期结果很有希望，但在构建强大的现实世界QAML工具方面仍然存在挑战。在这篇综述中，我们讨论了QAML的最新进展，并确定了关键挑战。我们还提出了未来的研究方向，这些方向可以确定随着量子计算硬件规模的扩大和噪声水平的降低而实现QAML方法实用化的途径。



## **9. On the Security Risks of Knowledge Graph Reasoning**

论知识图推理的安全风险 cs.CR

In proceedings of USENIX Security'23. Codes:  https://github.com/HarrialX/security-risk-KG-reasoning

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2305.02383v2) [paper-pdf](http://arxiv.org/pdf/2305.02383v2)

**Authors**: Zhaohan Xi, Tianyu Du, Changjiang Li, Ren Pang, Shouling Ji, Xiapu Luo, Xusheng Xiao, Fenglong Ma, Ting Wang

**Abstract**: Knowledge graph reasoning (KGR) -- answering complex logical queries over large knowledge graphs -- represents an important artificial intelligence task, entailing a range of applications (e.g., cyber threat hunting). However, despite its surging popularity, the potential security risks of KGR are largely unexplored, which is concerning, given the increasing use of such capability in security-critical domains.   This work represents a solid initial step towards bridging the striking gap. We systematize the security threats to KGR according to the adversary's objectives, knowledge, and attack vectors. Further, we present ROAR, a new class of attacks that instantiate a variety of such threats. Through empirical evaluation in representative use cases (e.g., medical decision support, cyber threat hunting, and commonsense reasoning), we demonstrate that ROAR is highly effective to mislead KGR to suggest pre-defined answers for target queries, yet with negligible impact on non-target ones. Finally, we explore potential countermeasures against ROAR, including filtering of potentially poisoning knowledge and training with adversarially augmented queries, which leads to several promising research directions.

摘要: 知识图推理(KGR)--在大型知识图上回答复杂的逻辑查询--代表着一项重要的人工智能任务，需要一系列应用程序(例如，网络威胁搜索)。然而，尽管KGR越来越受欢迎，但其潜在的安全风险在很大程度上还没有被探索出来，这是令人担忧的，因为这种能力在安全关键领域中的使用越来越多。这项工作是朝着弥合这一显著差距迈出的坚实的第一步。我们根据对手的目标、知识和攻击载体，对KGR面临的安全威胁进行系统化。此外，我们还介绍了咆哮，这是一种新的攻击类型，它实例化了各种此类威胁。通过对典型用例(如医疗决策支持、网络威胁搜索和常识推理)的实证评估，我们证明了Roar对于误导KGR为目标查询建议预定义答案是非常有效的，而对非目标查询的影响可以忽略不计。最后，我们探索了针对Roar的潜在对策，包括过滤潜在的中毒知识和使用恶意增强的查询进行训练，这导致了几个有前途的研究方向。



## **10. Rethinking the Backward Propagation for Adversarial Transferability**

关于对抗性转移的后向传播的再思考 cs.CV

14 pages

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.12685v1) [paper-pdf](http://arxiv.org/pdf/2306.12685v1)

**Authors**: Xiaosen Wang, Kangheng Tong, Kun He

**Abstract**: Transfer-based attacks generate adversarial examples on the surrogate model, which can mislead other black-box models without any access, making it promising to attack real-world applications. Recently, several works have been proposed to boost adversarial transferability, in which the surrogate model is usually overlooked. In this work, we identify that non-linear layers (e.g., ReLU, max-pooling, etc.) truncate the gradient during backward propagation, making the gradient w.r.t.input image imprecise to the loss function. We hypothesize and empirically validate that such truncation undermines the transferability of adversarial examples. Based on these findings, we propose a novel method called Backward Propagation Attack (BPA) to increase the relevance between the gradient w.r.t. input image and loss function so as to generate adversarial examples with higher transferability. Specifically, BPA adopts a non-monotonic function as the derivative of ReLU and incorporates softmax with temperature to smooth the derivative of max-pooling, thereby mitigating the information loss during the backward propagation of gradients. Empirical results on the ImageNet dataset demonstrate that not only does our method substantially boost the adversarial transferability, but it also is general to existing transfer-based attacks.

摘要: 基于传输的攻击在代理模型上生成敌意示例，这可能会在没有任何访问权限的情况下误导其他黑盒模型，使其有可能攻击现实世界的应用程序。最近，已有一些关于提高对抗性转移能力的工作被提出，但其中的代理模型往往被忽视。在这项工作中，我们确定了非线性层(例如，RELU、最大池等)。在反向传播过程中截断梯度，使得梯度w.r.t.输入图像对于损失函数不精确。我们假设和经验验证，这种截断破坏了对抗性例子的可转移性。基于这些发现，我们提出了一种新的方法，称为反向传播攻击(BPA)，以提高梯度之间的相关性。输入图像和损失函数，生成具有较高可转移性的对抗性实例。具体地说，BPA采用非单调函数作为RELU的导数，并将Softmax与温度相结合以平滑max-Pooling的导数，从而减少了梯度反向传播过程中的信息损失。在ImageNet数据集上的实验结果表明，我们的方法不仅大大提高了攻击的对抗性可转移性，而且对现有的基于传输的攻击也是通用的。



## **11. FDINet: Protecting against DNN Model Extraction via Feature Distortion Index**

FDINET：通过特征失真指数防止DNN模型提取 cs.CR

13 pages, 7 figures

**SubmitDate**: 2023-06-22    [abs](http://arxiv.org/abs/2306.11338v2) [paper-pdf](http://arxiv.org/pdf/2306.11338v2)

**Authors**: Hongwei Yao, Zheng Li, Haiqin Weng, Feng Xue, Kui Ren, Zhan Qin

**Abstract**: Machine Learning as a Service (MLaaS) platforms have gained popularity due to their accessibility, cost-efficiency, scalability, and rapid development capabilities. However, recent research has highlighted the vulnerability of cloud-based models in MLaaS to model extraction attacks. In this paper, we introduce FDINET, a novel defense mechanism that leverages the feature distribution of deep neural network (DNN) models. Concretely, by analyzing the feature distribution from the adversary's queries, we reveal that the feature distribution of these queries deviates from that of the model's training set. Based on this key observation, we propose Feature Distortion Index (FDI), a metric designed to quantitatively measure the feature distribution deviation of received queries. The proposed FDINET utilizes FDI to train a binary detector and exploits FDI similarity to identify colluding adversaries from distributed extraction attacks. We conduct extensive experiments to evaluate FDINET against six state-of-the-art extraction attacks on four benchmark datasets and four popular model architectures. Empirical results demonstrate the following findings FDINET proves to be highly effective in detecting model extraction, achieving a 100% detection accuracy on DFME and DaST. FDINET is highly efficient, using just 50 queries to raise an extraction alarm with an average confidence of 96.08% for GTSRB. FDINET exhibits the capability to identify colluding adversaries with an accuracy exceeding 91%. Additionally, it demonstrates the ability to detect two types of adaptive attacks.

摘要: 机器学习即服务(MLaaS)平台因其可访问性、成本效益、可扩展性和快速开发能力而广受欢迎。然而，最近的研究突显了MLaaS中基于云的模型对提取攻击的脆弱性。在本文中，我们介绍了FDINET，一种利用深度神经网络(DNN)模型特征分布的新型防御机制。具体地说，通过分析对手查询的特征分布，我们发现这些查询的特征分布偏离了模型训练集的特征分布。基于这一关键观察，我们提出了特征失真指数(FDI)，这是一种用来定量衡量所接收查询的特征分布偏差的度量。FDINET利用FDI来训练二进制检测器，并利用FDI的相似性从分布式抽取攻击中识别合谋对手。我们在四个基准数据集和四个流行的模型体系结构上进行了广泛的实验，以评估FDINET对六种最先进的提取攻击的攻击。实验结果表明，FDINET在检测模型提取方面具有很高的效率，在DFME和DAST上的检测准确率达到100%。FDINET的效率很高，仅使用50个查询就可以发出提取警报，GTSRB的平均置信度为96.08%。FDINET显示出识别串通对手的能力，准确率超过91%。此外，它还演示了检测两种类型的自适应攻击的能力。



## **12. SNAP: Efficient Extraction of Private Properties with Poisoning**

Snap：高效提取带有毒物的私有财产 cs.LG

28 pages, 16 figures

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2208.12348v2) [paper-pdf](http://arxiv.org/pdf/2208.12348v2)

**Authors**: Harsh Chaudhari, John Abascal, Alina Oprea, Matthew Jagielski, Florian Tramèr, Jonathan Ullman

**Abstract**: Property inference attacks allow an adversary to extract global properties of the training dataset from a machine learning model. Such attacks have privacy implications for data owners sharing their datasets to train machine learning models. Several existing approaches for property inference attacks against deep neural networks have been proposed, but they all rely on the attacker training a large number of shadow models, which induces a large computational overhead.   In this paper, we consider the setting of property inference attacks in which the attacker can poison a subset of the training dataset and query the trained target model. Motivated by our theoretical analysis of model confidences under poisoning, we design an efficient property inference attack, SNAP, which obtains higher attack success and requires lower amounts of poisoning than the state-of-the-art poisoning-based property inference attack by Mahloujifar et al. For example, on the Census dataset, SNAP achieves 34% higher success rate than Mahloujifar et al. while being 56.5x faster. We also extend our attack to infer whether a certain property was present at all during training and estimate the exact proportion of a property of interest efficiently. We evaluate our attack on several properties of varying proportions from four datasets and demonstrate SNAP's generality and effectiveness. An open-source implementation of SNAP can be found at https://github.com/johnmath/snap-sp23.

摘要: 属性推理攻击允许对手从机器学习模型中提取训练数据集的全局属性。此类攻击对共享数据集以训练机器学习模型的数据所有者具有隐私影响。已有的几种针对深度神经网络的属性推理攻击方法都依赖于攻击者训练大量的影子模型，这导致了较大的计算开销。在本文中，我们考虑了属性推理攻击的设置，在该攻击中，攻击者可以毒化训练数据集的子集并查询训练的目标模型。在对中毒下的模型可信度进行理论分析的基础上，设计了一种高效的属性推理攻击SNAP，它比MahLoujifar等人提出的基于中毒的属性推理攻击具有更高的攻击成功率和更低的投毒量。例如，在人口普查数据集上，SNAP的成功率比MahLoujifar等人高34%。同时速度提高了56.5倍。我们还扩展了我们的攻击，以推断在训练过程中是否存在某个属性，并有效地估计感兴趣的属性的确切比例。我们对来自四个数据集的几个不同比例的属性进行了评估，并展示了SNAP的通用性和有效性。有关SNAP的开源实现，请访问https://github.com/johnmath/snap-sp23.



## **13. Adversarial Attacks Neutralization via Data Set Randomization**

基于数据集随机化的对抗性攻击中和 cs.LG

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.12161v1) [paper-pdf](http://arxiv.org/pdf/2306.12161v1)

**Authors**: Mouna Rabhi, Roberto Di Pietro

**Abstract**: Adversarial attacks on deep-learning models pose a serious threat to their reliability and security. Existing defense mechanisms are narrow addressing a specific type of attack or being vulnerable to sophisticated attacks. We propose a new defense mechanism that, while being focused on image-based classifiers, is general with respect to the cited category. It is rooted on hyperspace projection. In particular, our solution provides a pseudo-random projection of the original dataset into a new dataset. The proposed defense mechanism creates a set of diverse projected datasets, where each projected dataset is used to train a specific classifier, resulting in different trained classifiers with different decision boundaries. During testing, it randomly selects a classifier to test the input. Our approach does not sacrifice accuracy over legitimate input. Other than detailing and providing a thorough characterization of our defense mechanism, we also provide a proof of concept of using four optimization-based adversarial attacks (PGD, FGSM, IGSM, and C\&W) and a generative adversarial attack testing them on the MNIST dataset. Our experimental results show that our solution increases the robustness of deep learning models against adversarial attacks and significantly reduces the attack success rate by at least 89% for optimization attacks and 78% for generative attacks. We also analyze the relationship between the number of used hyperspaces and the efficacy of the defense mechanism. As expected, the two are positively correlated, offering an easy-to-tune parameter to enforce the desired level of security. The generality and scalability of our solution and adaptability to different attack scenarios, combined with the excellent achieved results, other than providing a robust defense against adversarial attacks on deep learning networks, also lay the groundwork for future research in the field.

摘要: 针对深度学习模型的对抗性攻击对其可靠性和安全性构成了严重威胁。现有的防御机制只针对特定类型的攻击，或者容易受到复杂的攻击。我们提出了一种新的防御机制，该机制虽然专注于基于图像的分类器，但对于所引用的类别是通用的。它植根于超空间投影。特别是，我们的解决方案提供了原始数据集到新数据集的伪随机投影。该防御机制创建了一组不同的投影数据集，每个投影数据集用于训练特定的分类器，从而产生具有不同决策边界的不同训练分类器。在测试期间，它随机选择一个分类器来测试输入。我们的方法不会牺牲准确性而不是合法输入。除了详细描述和详细描述我们的防御机制外，我们还提供了使用四种基于优化的对抗性攻击(PGD、FGSM、IGSM和C\&W)的概念证明，以及在MNIST数据集上测试它们的生成性对抗性攻击。实验结果表明，该方法提高了深度学习模型对敌意攻击的稳健性，使优化攻击的成功率至少降低了89%，生成性攻击的成功率降低了78%。我们还分析了使用超空间的数量与防御机制有效性的关系。正如预期的那样，这两者是正相关的，提供了一个易于调整的参数来实施所需的安全级别。我们的解决方案的通用性和可扩展性以及对不同攻击场景的适应性，加上所取得的良好结果，除了对深度学习网络上的对手攻击提供了稳健的防御外，还为该领域未来的研究奠定了基础。



## **14. Sample Attackability in Natural Language Adversarial Attacks**

自然语言对抗性攻击中的样本可攻击性 cs.CL

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.12043v1) [paper-pdf](http://arxiv.org/pdf/2306.12043v1)

**Authors**: Vyas Raina, Mark Gales

**Abstract**: Adversarial attack research in natural language processing (NLP) has made significant progress in designing powerful attack methods and defence approaches. However, few efforts have sought to identify which source samples are the most attackable or robust, i.e. can we determine for an unseen target model, which samples are the most vulnerable to an adversarial attack. This work formally extends the definition of sample attackability/robustness for NLP attacks. Experiments on two popular NLP datasets, four state of the art models and four different NLP adversarial attack methods, demonstrate that sample uncertainty is insufficient for describing characteristics of attackable/robust samples and hence a deep learning based detector can perform much better at identifying the most attackable and robust samples for an unseen target model. Nevertheless, further analysis finds that there is little agreement in which samples are considered the most attackable/robust across different NLP attack methods, explaining a lack of portability of attackability detection methods across attack methods.

摘要: 自然语言处理中的对抗性攻击研究在设计强大的攻击方法和防御手段方面取得了重大进展。然而，很少有人试图确定哪些源样本是最具攻击性或健壮性的，即，对于看不见的目标模型，我们能否确定哪些样本最容易受到对手攻击。该工作正式扩展了NLP攻击的样本可攻击性/健壮性的定义。在两个流行的NLP数据集、四个最新模型和四种不同的NLP对抗攻击方法上的实验表明，样本不确定性不足以描述可攻击/健壮样本的特征，因此基于深度学习的检测器可以更好地识别不可见目标模型中最易攻击和健壮的样本。然而，进一步的分析发现，对于样本被认为是不同NLP攻击方法中最具可攻击性/健壮性的样本，几乎没有达成一致，这解释了为什么攻击方法之间缺乏可攻击性检测方法的可移植性。



## **15. Evaluating Adversarial Robustness of Convolution-based Human Motion Prediction**

基于卷积的人体运动预测的对抗稳健性评价 cs.CV

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.11990v1) [paper-pdf](http://arxiv.org/pdf/2306.11990v1)

**Authors**: Chengxu Duan, Zhicheng Zhang, Xiaoli Liu, Yonghao Dang, Jianqin Yin

**Abstract**: Human motion prediction has achieved a brilliant performance with the help of CNNs, which facilitates human-machine cooperation. However, currently, there is no work evaluating the potential risk in human motion prediction when facing adversarial attacks, which may cause danger in real applications. The adversarial attack will face two problems against human motion prediction: 1. For naturalness, pose data is highly related to the physical dynamics of human skeletons where Lp norm constraints cannot constrain the adversarial example well; 2. Unlike the pixel value in images, pose data is diverse at scale because of the different acquisition equipment and the data processing, which makes it hard to set fixed parameters to perform attacks. To solve the problems above, we propose a new adversarial attack method that perturbs the input human motion sequence by maximizing the prediction error with physical constraints. Specifically, we introduce a novel adaptable scheme that facilitates the attack to suit the scale of the target pose and two physical constraints to enhance the imperceptibility of the adversarial example. The evaluating experiments on three datasets show that the prediction errors of all target models are enlarged significantly, which means current convolution-based human motion prediction models can be easily disturbed under the proposed attack. The quantitative analysis shows that prior knowledge and semantic information modeling can be the key to the adversarial robustness of human motion predictors. The qualitative results indicate that the adversarial sample is hard to be noticed when compared frame by frame but is relatively easy to be detected when the sample is animated.

摘要: 在人工神经网络的帮助下，人体运动预测取得了很好的效果，有利于人机协作。然而，目前还没有对人体运动预测中的潜在风险进行评估的工作，这在实际应用中可能会造成危险。对抗性攻击将面临两个针对人体运动预测的问题：1.对于自然度，姿势数据与人体骨骼的物理动力学高度相关，其中Lp范数约束不能很好地约束对抗性示例；2.与图像中的像素值不同，由于采集设备和数据处理的不同，姿势数据在尺度上是多样的，这使得很难设置固定的参数来执行攻击。为了解决上述问题，我们提出了一种新的对抗性攻击方法，该方法通过在物理约束下最大化预测误差来扰动输入的人体运动序列。具体地说，我们引入了一种新的自适应方案来促进攻击以适应目标姿态的规模和两个物理约束来增强对抗性例子的不可见性。在三个数据集上的评估实验表明，所有目标模型的预测误差都显著增大，这意味着现有的基于卷积的人体运动预测模型在所提出的攻击下很容易受到干扰。定量分析表明，先验知识和语义信息建模是人体运动预测器对抗性健壮性的关键。定性结果表明，对抗性样本在逐帧比较时很难被注意到，但在样本被动画时相对容易被检测到。



## **16. Universal adversarial perturbations for multiple classification tasks with quantum classifiers**

量子分类器多分类任务的普遍对抗性扰动 quant-ph

**SubmitDate**: 2023-06-21    [abs](http://arxiv.org/abs/2306.11974v1) [paper-pdf](http://arxiv.org/pdf/2306.11974v1)

**Authors**: Yun-Zhong Qiu

**Abstract**: Quantum adversarial machine learning is an emerging field that studies the vulnerability of quantum learning systems against adversarial perturbations and develops possible defense strategies. Quantum universal adversarial perturbations are small perturbations, which can make different input samples into adversarial examples that may deceive a given quantum classifier. This is a field that was rarely looked into but worthwhile investigating because universal perturbations might simplify malicious attacks to a large extent, causing unexpected devastation to quantum machine learning models. In this paper, we take a step forward and explore the quantum universal perturbations in the context of heterogeneous classification tasks. In particular, we find that quantum classifiers that achieve almost state-of-the-art accuracy on two different classification tasks can be both conclusively deceived by one carefully-crafted universal perturbation. This result is explicitly demonstrated with well-designed quantum continual learning models with elastic weight consolidation method to avoid catastrophic forgetting, as well as real-life heterogeneous datasets from hand-written digits and medical MRI images. Our results provide a simple and efficient way to generate universal perturbations on heterogeneous classification tasks and thus would provide valuable guidance for future quantum learning technologies.

摘要: 量子对抗机器学习是一个新兴的研究领域，它研究量子学习系统对对抗扰动的脆弱性，并开发可能的防御策略。量子通用对抗性扰动是一种微小的扰动，它可以使不同的输入样本变成可能欺骗给定量子分类器的对抗性例子。这是一个很少被研究但值得研究的领域，因为普遍的扰动可能会在很大程度上简化恶意攻击，给量子机器学习模型造成意想不到的破坏。在这篇文章中，我们向前迈进了一步，探索了异质分类任务背景下的量子普适微扰。特别是，我们发现，在两个不同的分类任务上获得几乎最先进的精度的量子分类器都可能最终被一个精心设计的普遍扰动所欺骗。这一结果通过设计良好的弹性权重巩固方法的量子连续学习模型来避免灾难性遗忘，以及来自手写数字和医学MRI图像的现实生活中的异质数据集得到了明确的证明。我们的结果提供了一种简单而有效的方法来产生对异类分类任务的普遍扰动，从而为未来的量子学习技术提供了有价值的指导。



## **17. Spectral Augmentation for Self-Supervised Learning on Graphs**

图的自监督学习的谱增强算法 cs.LG

ICLR 2023

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2210.00643v2) [paper-pdf](http://arxiv.org/pdf/2210.00643v2)

**Authors**: Lu Lin, Jinghui Chen, Hongning Wang

**Abstract**: Graph contrastive learning (GCL), as an emerging self-supervised learning technique on graphs, aims to learn representations via instance discrimination. Its performance heavily relies on graph augmentation to reflect invariant patterns that are robust to small perturbations; yet it still remains unclear about what graph invariance GCL should capture. Recent studies mainly perform topology augmentations in a uniformly random manner in the spatial domain, ignoring its influence on the intrinsic structural properties embedded in the spectral domain. In this work, we aim to find a principled way for topology augmentations by exploring the invariance of graphs from the spectral perspective. We develop spectral augmentation which guides topology augmentations by maximizing the spectral change. Extensive experiments on both graph and node classification tasks demonstrate the effectiveness of our method in self-supervised representation learning. The proposed method also brings promising generalization capability in transfer learning, and is equipped with intriguing robustness property under adversarial attacks. Our study sheds light on a general principle for graph topology augmentation.

摘要: 图对比学习(GCL)是一种新兴的关于图的自监督学习技术，旨在通过实例区分来学习表示。它的性能在很大程度上依赖于图的增强来反映对小扰动具有健壮性的不变模式；然而，GCL应该捕获什么图不变性仍然是不清楚的。目前的研究主要是在空间域以均匀随机的方式进行拓扑增强，而忽略了其对谱域固有结构性质的影响。在这项工作中，我们的目的是通过从谱的角度探索图的不变性来寻找一种原则性的拓扑增强方法。我们开发了光谱增强，它通过最大化光谱变化来指导拓扑增强。在图和节点分类任务上的大量实验证明了该方法在自监督表示学习中的有效性。该方法在转移学习中具有良好的泛化能力，并且在对抗攻击下具有良好的鲁棒性。我们的研究揭示了图拓扑增强的一般原理。



## **18. Towards a robust and reliable deep learning approach for detection of compact binary mergers in gravitational wave data**

一种稳健可靠的深度学习方法检测引力波数据中的紧密二元合并 gr-qc

22 pages, 21 figures

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11797v1) [paper-pdf](http://arxiv.org/pdf/2306.11797v1)

**Authors**: Shreejit Jadhav, Mihir Shrivastava, Sanjit Mitra

**Abstract**: The ability of deep learning (DL) approaches to learn generalised signal and noise models, coupled with their fast inference on GPUs, holds great promise for enhancing gravitational-wave (GW) searches in terms of speed, parameter space coverage, and search sensitivity. However, the opaque nature of DL models severely harms their reliability. In this work, we meticulously develop a DL model stage-wise and work towards improving its robustness and reliability. First, we address the problems in maintaining the purity of training data by deriving a new metric that better reflects the visual strength of the "chirp" signal features in the data. Using a reduced, smooth representation obtained through a variational auto-encoder (VAE), we build a classifier to search for compact binary coalescence (CBC) signals. Our tests on real LIGO data show an impressive performance of the model. However, upon probing the robustness of the model through adversarial attacks, its simple failure modes were identified, underlining how such models can still be highly fragile. As a first step towards bringing robustness, we retrain the model in a novel framework involving a generative adversarial network (GAN). Over the course of training, the model learns to eliminate the primary modes of failure identified by the adversaries. Although absolute robustness is practically impossible to achieve, we demonstrate some fundamental improvements earned through such training, like sparseness and reduced degeneracy in the extracted features at different layers inside the model. Through comparative inference on real LIGO data, we show that the prescribed robustness is achieved at practically zero cost in terms of performance. Through a direct search on ~8.8 days of LIGO data, we recover two significant CBC events from GWTC-2.1, GW190519_153544 and GW190521_074359, and report the search sensitivity.

摘要: 深度学习(DL)方法学习广义信号和噪声模型的能力，加上它们在GPU上的快速推理，在速度、参数空间覆盖和搜索灵敏度方面都有望提高引力波(GW)搜索。然而，DL模型的不透明性质严重损害了它们的可靠性。在这项工作中，我们精心开发了一个阶段性的DL模型，并致力于提高其稳健性和可靠性。首先，我们通过推导一种新的度量来解决保持训练数据的纯度方面的问题，该度量更好地反映了数据中“chirp”信号特征的视觉强度。利用变分自动编码器(VAE)得到的简化的平滑表示，我们构建了一个分类器来搜索紧凑的二进制合并(CBC)信号。我们对真实LIGO数据的测试表明，该模型具有令人印象深刻的性能。然而，在通过对抗性攻击探测该模型的稳健性之后，它的简单故障模式被识别出来，这突显了这种模型如何仍然非常脆弱。作为带来稳健性的第一步，我们在一个新的框架中重新训练该模型，该框架涉及生成性对抗网络(GAN)。在训练过程中，模型学习消除对手确定的主要失败模式。虽然绝对的稳健性实际上是不可能实现的，但我们展示了通过这样的训练获得的一些基本的改进，例如在模型内部不同层提取的特征的稀疏性和减少的简并度。通过对实际LIGO数据的比较推理，我们表明，在性能方面几乎没有代价就达到了所规定的鲁棒性。通过直接搜索8.8d的LIGO资料，我们从GWTC2.1、GW190519_153544和GW190521_074359中恢复了两个有意义的CBC事件，并报告了搜索敏感性。



## **19. Illusory Attacks: Detectability Matters in Adversarial Attacks on Sequential Decision-Makers**

虚幻攻击：对顺序决策者的对抗性攻击中的可探测性问题 cs.AI

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2207.10170v3) [paper-pdf](http://arxiv.org/pdf/2207.10170v3)

**Authors**: Tim Franzmeyer, Stephen McAleer, João F. Henriques, Jakob N. Foerster, Philip H. S. Torr, Adel Bibi, Christian Schroeder de Witt

**Abstract**: Autonomous agents deployed in the real world need to be robust against adversarial attacks on sensory inputs. Robustifying agent policies requires anticipating the strongest attacks possible. We demonstrate that existing observation-space attacks on reinforcement learning agents have a common weakness: while effective, their lack of temporal consistency makes them detectable using automated means or human inspection. Detectability is undesirable to adversaries as it may trigger security escalations. We introduce perfect illusory attacks, a novel form of adversarial attack on sequential decision-makers that is both effective and provably statistically undetectable. We then propose the more versatile R-attacks, which result in observation transitions that are consistent with the state-transition function of the adversary-free environment and can be learned end-to-end. Compared to existing attacks, we empirically find R-attacks to be significantly harder to detect with automated methods, and a small study with human subjects suggests they are similarly harder to detect for humans. We propose that undetectability should be a central concern in the study of adversarial attacks on mixed-autonomy settings.

摘要: 部署在现实世界中的自主代理需要强大地抵御对感觉输入的敌意攻击。将代理策略规模化需要预测可能最强的攻击。我们证明了现有的对强化学习代理的观察空间攻击有一个共同的弱点：虽然有效，但它们缺乏时间一致性，使得它们可以使用自动手段或人工检查来检测。对于对手来说，可探测性是不可取的，因为它可能会引发安全升级。我们引入了完全虚幻攻击，这是一种针对序列决策者的新型对抗性攻击，既有效，又在统计上可证明是不可检测的。然后，我们提出了更通用的R-攻击，它产生的观察转移与无对手环境的状态转移函数一致，并且可以端到端地学习。与现有的攻击相比，我们经验上发现，使用自动化方法检测R攻击要困难得多，一项针对人类受试者的小型研究表明，人类同样更难检测到R攻击。我们建议在混合自主环境下的对抗性攻击研究中，不可检测性应该是一个核心问题。



## **20. Robust Adversarial Attacks Detection based on Explainable Deep Reinforcement Learning For UAV Guidance and Planning**

基于可解释深度强化学习的无人机制导规划鲁棒对抗攻击检测 cs.LG

13 pages, 16 figures

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2206.02670v4) [paper-pdf](http://arxiv.org/pdf/2206.02670v4)

**Authors**: Thomas Hickling, Nabil Aouf, Phillippa Spencer

**Abstract**: The dangers of adversarial attacks on Uncrewed Aerial Vehicle (UAV) agents operating in public are increasing. Adopting AI-based techniques and, more specifically, Deep Learning (DL) approaches to control and guide these UAVs can be beneficial in terms of performance but can add concerns regarding the safety of those techniques and their vulnerability against adversarial attacks. Confusion in the agent's decision-making process caused by these attacks can seriously affect the safety of the UAV. This paper proposes an innovative approach based on the explainability of DL methods to build an efficient detector that will protect these DL schemes and the UAVs adopting them from attacks. The agent adopts a Deep Reinforcement Learning (DRL) scheme for guidance and planning. The agent is trained with a Deep Deterministic Policy Gradient (DDPG) with Prioritised Experience Replay (PER) DRL scheme that utilises Artificial Potential Field (APF) to improve training times and obstacle avoidance performance. A simulated environment for UAV explainable DRL-based planning and guidance, including obstacles and adversarial attacks, is built. The adversarial attacks are generated by the Basic Iterative Method (BIM) algorithm and reduced obstacle course completion rates from 97\% to 35\%. Two adversarial attack detectors are proposed to counter this reduction. The first one is a Convolutional Neural Network Adversarial Detector (CNN-AD), which achieves accuracy in the detection of 80\%. The second detector utilises a Long Short Term Memory (LSTM) network. It achieves an accuracy of 91\% with faster computing times compared to the CNN-AD, allowing for real-time adversarial detection.

摘要: 对在公共场合工作的无人驾驶飞行器(UAV)特工进行对抗性攻击的危险正在增加。采用基于人工智能的技术，更具体地说，深度学习(DL)方法来控制和引导这些无人机在性能方面可能是有益的，但可能会增加对这些技术的安全性及其抵御对手攻击的脆弱性的担忧。这些攻击导致的代理决策过程中的混乱会严重影响无人机的安全。本文提出了一种基于DL方法的可解释性的创新方法，以构建一个有效的检测器来保护这些DL方案以及采用这些方案的无人机免受攻击。代理采用深度强化学习(DRL)方案进行指导和规划。代理使用深度确定性策略梯度(DDPG)和优先体验重播(PER)DRL方案进行训练，该方案利用人工势场(APF)来改进训练时间和避障性能。建立了无人机基于DRL的可解释规划和制导的仿真环境，包括障碍物和对抗性攻击。对抗性攻击由基本迭代法(BIM)算法生成，障碍路径完成率由97降至35。提出了两个对抗性攻击检测器来对抗这种减少。第一种是卷积神经网络敌手检测器(CNN-AD)，它可以达到80%的检测精度。第二检测器利用长短期记忆(LSTM)网络。与CNN-AD相比，它具有91%的准确率和更快的计算时间，允许实时检测对手。



## **21. Reversible Adversarial Examples with Beam Search Attack and Grayscale Invariance**

波束搜索攻击和灰度不变性的可逆对抗实例 cs.CR

Submitted to ICICS2023

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11322v1) [paper-pdf](http://arxiv.org/pdf/2306.11322v1)

**Authors**: Haodong Zhang, Chi Man Pun, Xia Du

**Abstract**: Reversible adversarial examples (RAE) combine adversarial attacks and reversible data-hiding technology on a single image to prevent illegal access. Most RAE studies focus on achieving white-box attacks. In this paper, we propose a novel framework to generate reversible adversarial examples, which combines a novel beam search based black-box attack and reversible data hiding with grayscale invariance (RDH-GI). This RAE uses beam search to evaluate the adversarial gain of historical perturbations and guide adversarial perturbations. After the adversarial examples are generated, the framework RDH-GI embeds the secret data that can be recovered losslessly. Experimental results show that our method can achieve an average Peak Signal-to-Noise Ratio (PSNR) of at least 40dB compared to source images with limited query budgets. Our method can also achieve a targeted black-box reversible adversarial attack for the first time.

摘要: 可逆对抗性实例(RAE)在单个图像上结合了对抗性攻击和可逆数据隐藏技术，以防止非法访问。大多数RAE研究都集中在实现白盒攻击上。本文提出了一种新的生成可逆攻击实例的框架，该框架结合了一种新的基于波束搜索的黑盒攻击和具有灰度不变性的可逆数据隐藏(RDH-GI)。该RAE使用波束搜索来评估历史扰动的对抗性增益，并指导对抗性扰动。在生成对抗性样本后，框架RDH-GI嵌入了可以无损恢复的秘密数据。实验结果表明，与查询预算有限的源图像相比，我们的方法可以获得平均峰值信噪比(PSNR)至少40dB。我们的方法还可以首次实现有针对性的黑盒可逆对抗攻击。



## **22. Comparative Evaluation of Recent Universal Adversarial Perturbations in Image Classification**

图像分类中近期普遍存在的对抗性扰动的比较评价 cs.CV

18 pages,8 figures, 7 tables

**SubmitDate**: 2023-06-20    [abs](http://arxiv.org/abs/2306.11261v1) [paper-pdf](http://arxiv.org/pdf/2306.11261v1)

**Authors**: Juanjuan Weng, Zhiming Luo, Dazhen Lin, Shaozi Li

**Abstract**: The vulnerability of Convolutional Neural Networks (CNNs) to adversarial samples has recently garnered significant attention in the machine learning community. Furthermore, recent studies have unveiled the existence of universal adversarial perturbations (UAPs) that are image-agnostic and highly transferable across different CNN models. In this survey, our primary focus revolves around the recent advancements in UAPs specifically within the image classification task. We categorize UAPs into two distinct categories, i.e., noise-based attacks and generator-based attacks, thereby providing a comprehensive overview of representative methods within each category. By presenting the computational details of these methods, we summarize various loss functions employed for learning UAPs. Furthermore, we conduct a comprehensive evaluation of different loss functions within consistent training frameworks, including noise-based and generator-based. The evaluation covers a wide range of attack settings, including black-box and white-box attacks, targeted and untargeted attacks, as well as the examination of defense mechanisms.   Our quantitative evaluation results yield several important findings pertaining to the effectiveness of different loss functions, the selection of surrogate CNN models, the impact of training data and data size, and the training frameworks involved in crafting universal attackers. Finally, to further promote future research on universal adversarial attacks, we provide some visualizations of the perturbations and discuss the potential research directions.

摘要: 卷积神经网络(CNN)对敌意样本的脆弱性最近在机器学习界引起了极大的关注。此外，最近的研究揭示了普遍的对抗性扰动(UAP)的存在，这些UAP是与图像无关的，并且可以在不同的CNN模型中高度转移。在这项调查中，我们主要关注UAP的最新进展，特别是在图像分类任务中。我们将UAP分为两个不同的类别，即基于噪声的攻击和基于生成器的攻击，从而全面概述了每一类中具有代表性的方法。通过给出这些方法的计算细节，我们总结了用于学习UAP的各种损失函数。此外，我们在一致的训练框架内对不同的损失函数进行了综合评估，包括基于噪声的和基于生成器的。评估涵盖了广泛的攻击设置，包括黑盒和白盒攻击、定向攻击和非定向攻击，以及对防御机制的检查。我们的定量评估结果产生了几个重要的发现，涉及不同损失函数的有效性、代理CNN模型的选择、训练数据和数据大小的影响，以及制作通用攻击者所涉及的训练框架。最后，为了进一步促进通用对抗攻击的研究，我们提供了一些扰动的可视化，并讨论了潜在的研究方向。



## **23. Adversarial Robustness of Learning-based Static Malware Classifiers**

基于学习的静态恶意软件分类器的对抗健壮性 cs.CR

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2303.13372v2) [paper-pdf](http://arxiv.org/pdf/2303.13372v2)

**Authors**: Shoumik Saha, Wenxiao Wang, Yigitcan Kaya, Soheil Feizi, Tudor Dumitras

**Abstract**: Malware detection has long been a stage for an ongoing arms race between malware authors and anti-virus systems. Solutions that utilize machine learning (ML) gain traction as the scale of this arms race increases. This trend, however, makes performing attacks directly on ML an attractive prospect for adversaries. We study this arms race from both perspectives in the context of MalConv, a popular convolutional neural network-based malware classifier that operates on raw bytes of files. First, we show that MalConv is vulnerable to adversarial patch attacks: appending a byte-level patch to malware files bypasses detection 94.3% of the time. Moreover, we develop a universal adversarial patch (UAP) attack where a single patch can drop the detection rate in constant time of any malware file that contains it by 80%. These patches are effective even being relatively small with respect to the original file size -- between 2%-8%. As a countermeasure, we then perform window ablation that allows us to apply de-randomized smoothing, a modern certified defense to patch attacks in vision tasks, to raw files. The resulting `smoothed-MalConv' can detect over 80% of malware that contains the universal patch and provides certified robustness up to 66%, outlining a promising step towards robust malware detection. To our knowledge, we are the first to apply universal adversarial patch attack and certified defense using ablations on byte level in the malware field.

摘要: 恶意软件检测长期以来一直是恶意软件作者和反病毒系统之间持续军备竞赛的一个阶段。随着军备竞赛规模的扩大，利用机器学习(ML)的解决方案获得了吸引力。然而，这种趋势使得直接对ML进行攻击对对手来说是一个有吸引力的前景。我们在MalConv的背景下从两个角度研究了这场军备竞赛，MalConv是一种流行的基于卷积神经网络的恶意软件分类器，它对文件的原始字节进行操作。首先，我们发现MalConv容易受到恶意补丁的攻击：向恶意软件文件添加字节级补丁可以在94.3%的时间内绕过检测。此外，我们还开发了一种通用对手补丁(UAP)攻击，其中单个补丁可以在固定时间内使包含它的任何恶意软件文件的检测率下降80%。即使相对于原始文件大小相对较小--介于2%-8%之间，这些补丁也是有效的。作为对策，我们然后执行窗口消融，允许我们对RAW文件应用去随机化平滑，这是一种针对视觉任务中的补丁攻击的现代认证防御。由此产生的“平滑MalConv”可以检测包含通用补丁的80%以上的恶意软件，并提供高达66%的经验证的健壮性，概述了朝着健壮的恶意软件检测迈出的有希望的一步。据我们所知，我们是第一个在恶意软件领域应用通用对抗性补丁攻击和使用字节级烧蚀的认证防御的公司。



## **24. CosPGD: a unified white-box adversarial attack for pixel-wise prediction tasks**

CosPGD：一种针对像素预测任务的统一白盒对抗攻击 cs.CV

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2302.02213v2) [paper-pdf](http://arxiv.org/pdf/2302.02213v2)

**Authors**: Shashank Agnihotri, Steffen Jung, Margret Keuper

**Abstract**: While neural networks allow highly accurate predictions in many tasks, their lack of robustness towards even slight input perturbations hampers their deployment in many real-world applications. Recent research towards evaluating the robustness of neural networks such as the seminal projected gradient descent(PGD) attack and subsequent works have drawn significant attention, as they provide an effective insight into the quality of representations learned by the network. However, these methods predominantly focus on image classification tasks, while only a few approaches specifically address the analysis of pixel-wise prediction tasks such as semantic segmentation, optical flow, disparity estimation, and others, respectively. Thus, there is a lack of a unified adversarial robustness benchmarking tool(algorithm) that is applicable to all such pixel-wise prediction tasks. In this work, we close this gap and propose CosPGD, a novel white-box adversarial attack that allows optimizing dedicated attacks for any pixel-wise prediction task in a unified setting. It leverages the cosine similarity between the distributions over the predictions and ground truth (or target) to extend directly from classification tasks to regression settings. We outperform the SotA on semantic segmentation attacks in our experiments on PASCAL VOC2012 and CityScapes. Further, we set a new benchmark for adversarial attacks on optical flow, and image restoration displaying the ability to extend to any pixel-wise prediction task.

摘要: 虽然神经网络允许在许多任务中进行高精度的预测，但它们对即使是轻微的输入扰动缺乏稳健性，阻碍了它们在许多现实世界应用中的部署。最近关于评估神经网络的稳健性的研究，如种子投影梯度下降(PGD)攻击和后续工作，引起了人们的极大关注，因为它们提供了对网络学习的表示的质量的有效洞察。然而，这些方法主要集中在图像分类任务上，而只有少数方法分别具体地处理像素级预测任务的分析，例如语义分割、光流、视差估计等。因此，缺乏适用于所有这种逐像素预测任务的统一对抗性健壮性基准测试工具(算法)。在这项工作中，我们缩小了这一差距，并提出了CosPGD，一种新的白盒对抗攻击，允许在统一的环境下为任何像素级预测任务优化专用攻击。它利用预测上的分布和基本事实(或目标)之间的余弦相似性，直接从分类任务扩展到回归设置。在PASCAL VOC2012和CITYSCAPES上的实验中，我们在语义分割攻击上优于SOTA。此外，我们为光流的对抗性攻击和图像恢复设置了一个新的基准，显示了扩展到任何像素预测任务的能力。



## **25. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

基于重放的自主机器人对传感器欺骗攻击的恢复 cs.RO

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2209.04554v4) [paper-pdf](http://arxiv.org/pdf/2209.04554v4)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstract**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Unfortunately, RV sensors can be compromised by physical attacks such as tampering or spoofing, leading to a crash. In this paper, we present DeLorean, a modelfree recovery framework for recovering autonomous RVs from sensor deception attacks (SDA). DeLorean is designed to recover RVs even from a strong SDA in which the adversary targets multiple heterogeneous sensors simultaneously (even all the sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used to derive actuator signals. DeLorean then replays historic state information in the RV's feedback control loop for a temporary mitigation and recovers the RV from SDA. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from SDAs, and ensure mission success in 90.7% of the cases on average.

摘要: 传感器对于机器人车辆(RV)的自主操作至关重要。不幸的是，房车传感器可能会受到物理攻击，如篡改或欺骗，导致崩溃。本文提出了一种无模型恢复框架DeLorean，用于从传感器欺骗攻击(SDA)中恢复自主房车。DeLorean被设计成即使在强大的SDA中也能恢复RV，在SDA中，对手同时瞄准多个不同的传感器(甚至所有传感器)。在SDAS下，DeLorean检查攻击引起的错误，识别目标传感器，并防止错误的传感器输入被用于推导执行器信号。然后，DeLorean在房车的反馈控制环路中重放历史状态信息以暂时缓解，并从SDA恢复房车。我们对4辆真实房车和2辆模拟房车的评估表明，DeLorean可以从SDA中恢复房车，并平均确保90.7%的任务成功。



## **26. Adversarial Training Should Be Cast as a Non-Zero-Sum Game**

对抗性训练应视为一种非零和博弈 cs.LG

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.11035v1) [paper-pdf](http://arxiv.org/pdf/2306.11035v1)

**Authors**: Alexander Robey, Fabian Latorre, George J. Pappas, Hamed Hassani, Volkan Cevher

**Abstract**: One prominent approach toward resolving the adversarial vulnerability of deep neural networks is the two-player zero-sum paradigm of adversarial training, in which predictors are trained against adversarially-chosen perturbations of data. Despite the promise of this approach, algorithms based on this paradigm have not engendered sufficient levels of robustness, and suffer from pathological behavior like robust overfitting. To understand this shortcoming, we first show that the commonly used surrogate-based relaxation used in adversarial training algorithms voids all guarantees on the robustness of trained classifiers. The identification of this pitfall informs a novel non-zero-sum bilevel formulation of adversarial training, wherein each player optimizes a different objective function. Our formulation naturally yields a simple algorithmic framework that matches and in some cases outperforms state-of-the-art attacks, attains comparable levels of robustness to standard adversarial training algorithms, and does not suffer from robust overfitting.

摘要: 解决深层神经网络的对抗性脆弱性的一个重要方法是对抗性训练的两人零和范例，在这种范例中，预测者被训练来对抗对抗性选择的数据扰动。尽管这种方法前景看好，但基于这种范例的算法并没有产生足够的健壮性，并且存在健壮性过适应等病态行为。为了理解这一缺陷，我们首先证明了对抗性训练算法中常用的基于代理的松弛算法无效了对训练后的分类器的健壮性的所有保证。对这一陷阱的识别提供了一种新的对抗性训练的非零和双层公式，其中每个参与者优化一个不同的目标函数。我们的公式自然产生了一个简单的算法框架，该框架匹配并在某些情况下优于最先进的攻击，达到了与标准对抗性训练算法相当的健壮性水平，并且不会受到健壮过度匹配的影响。



## **27. Eigenpatches -- Adversarial Patches from Principal Components**

特征斑块--来自主成分的对抗性斑块 cs.CV

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10963v1) [paper-pdf](http://arxiv.org/pdf/2306.10963v1)

**Authors**: Jens Bayer, Stefan Becker, David Münch, Michael Arens

**Abstract**: Adversarial patches are still a simple yet powerful white box attack that can be used to fool object detectors by suppressing possible detections. The patches of these so-called evasion attacks are computational expensive to produce and require full access to the attacked detector. This paper addresses the problem of computational expensiveness by analyzing 375 generated patches, calculating the principal components of these and show, that linear combinations of the resulting "eigenpatches" can be used to fool object detections successfully.

摘要: 对抗性补丁仍然是一种简单但强大的白盒攻击，可以通过抑制可能的检测来愚弄对象检测器。这些所谓的逃避攻击的补丁是计算昂贵的，并且需要完全访问被攻击的检测器。本文通过分析375个生成的面片，计算这些面片的主成分，从而解决了计算量过大的问题，并证明了所得到的“特征面片”的线性组合可以成功地愚弄目标检测。



## **28. Attack-Resilient Design for Connected and Automated Vehicles**

联网和自动化车辆的抗攻击设计 eess.SY

arXiv admin note: text overlap with arXiv:2109.01553

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10925v1) [paper-pdf](http://arxiv.org/pdf/2306.10925v1)

**Authors**: Tianci Yang, Carlos Murguia, Dragan Nesic, Chau Yuen

**Abstract**: By sharing local sensor information via Vehicle-to-Vehicle (V2V) wireless communication networks, Cooperative Adaptive Cruise Control (CACC) is a technology that enables Connected and Automated Vehicles (CAVs) to drive autonomously on the highway in closely-coupled platoons. The use of CACC technologies increases safety and the traffic throughput, and decreases fuel consumption and CO2 emissions. However, CAVs heavily rely on embedded software, hardware, and communication networks that make them vulnerable to a range of cyberattacks. Cyberattacks to a particular CAV compromise the entire platoon as CACC schemes propagate corrupted data to neighboring vehicles potentially leading to traffic delays and collisions. Physics-based monitors can be used to detect the presence of False Data Injection (FDI) attacks to CAV sensors; however, unavoidable system disturbances and modelling uncertainty often translates to conservative detection results. Given enough system knowledge, adversaries are still able to launch a range of attacks that can surpass the detection scheme by hiding within the system disturbances and uncertainty -- we refer to this class of attacks as \textit{stealthy FDI attacks}. Stealthy attacks are hard to deal with as they affect the platoon dynamics without being noticed. In this manuscript, we propose a co-design methodology (built around a series convex programs) to synthesize distributed attack monitors and $H_{\infty}$ CACC controllers that minimize the joint effect of stealthy FDI attacks and system disturbances on the platoon dynamics while guaranteeing a prescribed platooning performance (in terms of tracking and string stability). Computer simulations are provided to illustrate the performance of out tools.

摘要: 通过车对车(V2V)无线通信网络共享本地传感器信息，协作自适应巡航控制(CACC)是一种使互联和自动车辆(CAV)能够在紧密耦合的排中自动在高速公路上行驶的技术。CACC技术的使用提高了安全性和交通吞吐量，降低了油耗和二氧化碳排放。然而，骑士队严重依赖嵌入式软件、硬件和通信网络，这使得他们容易受到一系列网络攻击。针对特定CAV的网络攻击危及整个排，因为CACC方案将损坏的数据传播到邻近的车辆，可能导致交通延误和碰撞。基于物理的监测器可以用来检测对CAV传感器的虚假数据注入(FDI)攻击的存在；然而，不可避免的系统干扰和建模不确定性通常会转化为保守的检测结果。在有足够的系统知识的情况下，攻击者仍然能够通过隐藏在系统中的干扰和不确定性来发起一系列可以超越检测方案的攻击--我们将这类攻击称为\Texttit{隐蔽的FDI攻击}。隐形攻击很难处理，因为它们在没有被注意到的情况下影响了排的动态。在这篇手稿中，我们提出了一种协同设计方法(建立在一系列凸规划的基础上)来综合分布式攻击监视器和$HINFTY$CACC控制器，以最小化隐身FDI攻击和系统干扰对排动力学的联合影响，同时保证指定的排性能(在跟踪和字符串稳定性方面)。为了说明OUT工具的性能，提供了计算机模拟。



## **29. On the Robustness of Dataset Inference**

关于数据集推理的稳健性 cs.LG

19 pages; Accepted to Transactions on Machine Learning Research  06/2023

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2210.13631v3) [paper-pdf](http://arxiv.org/pdf/2210.13631v3)

**Authors**: Sebastian Szyller, Rui Zhang, Jian Liu, N. Asokan

**Abstract**: Machine learning (ML) models are costly to train as they can require a significant amount of data, computational resources and technical expertise. Thus, they constitute valuable intellectual property that needs protection from adversaries wanting to steal them. Ownership verification techniques allow the victims of model stealing attacks to demonstrate that a suspect model was in fact stolen from theirs.   Although a number of ownership verification techniques based on watermarking or fingerprinting have been proposed, most of them fall short either in terms of security guarantees (well-equipped adversaries can evade verification) or computational cost. A fingerprinting technique, Dataset Inference (DI), has been shown to offer better robustness and efficiency than prior methods.   The authors of DI provided a correctness proof for linear (suspect) models. However, in a subspace of the same setting, we prove that DI suffers from high false positives (FPs) -- it can incorrectly identify an independent model trained with non-overlapping data from the same distribution as stolen. We further prove that DI also triggers FPs in realistic, non-linear suspect models. We then confirm empirically that DI in the black-box setting leads to FPs, with high confidence.   Second, we show that DI also suffers from false negatives (FNs) -- an adversary can fool DI (at the cost of incurring some accuracy loss) by regularising a stolen model's decision boundaries using adversarial training, thereby leading to an FN. To this end, we demonstrate that black-box DI fails to identify a model adversarially trained from a stolen dataset -- the setting where DI is the hardest to evade.   Finally, we discuss the implications of our findings, the viability of fingerprinting-based ownership verification in general, and suggest directions for future work.

摘要: 机器学习(ML)模型的训练成本很高，因为它们可能需要大量的数据、计算资源和技术专长。因此，它们构成了宝贵的知识产权，需要保护，不受想要窃取它们的对手的攻击。所有权验证技术允许模型盗窃攻击的受害者证明可疑模型实际上是从他们的模型中被盗的。虽然已经提出了一些基于水印或指纹的所有权验证技术，但它们大多在安全保证(装备良好的攻击者可以逃避验证)或计算代价方面存在不足。一种指纹技术，数据集推理(DI)，已经被证明比以前的方法提供了更好的稳健性和效率。DI的作者为线性(可疑)模型提供了正确性证明。然而，在相同设置的子空间中，我们证明DI存在高误报(FP)--它可能错误地识别使用来自相同分布的非重叠数据训练的独立模型。我们进一步证明，在现实的、非线性的可疑模型中，依赖注入也会触发FP。然后，我们以很高的置信度从经验上证实了黑盒设置中的DI会导致FP。其次，我们证明了DI也存在假阴性(FN)--对手可以通过使用对抗性训练来规则被盗模型的决策边界来愚弄DI(以招致一些准确性损失为代价)，从而导致FN。为此，我们演示了黑盒DI无法识别从窃取的数据集中恶意训练的模型--DI最难逃避的设置。最后，我们讨论了我们的发现的含义，基于指纹的所有权验证总体上的可行性，并对未来的工作提出了方向。



## **30. Hidden Backdoor Attack against Deep Learning-Based Wireless Signal Modulation Classifiers**

针对基于深度学习的无线信号调制分类器的隐藏后门攻击 eess.SP

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2306.10753v1) [paper-pdf](http://arxiv.org/pdf/2306.10753v1)

**Authors**: Yunsong Huang, Weicheng Liu, Hui-Ming Wang

**Abstract**: Recently, DL has been exploited in wireless communications such as modulation classification. However, due to the openness of wireless channel and unexplainability of DL, it is also vulnerable to adversarial attacks. In this correspondence, we investigate a so called hidden backdoor attack to modulation classification, where the adversary puts elaborately designed poisoned samples on the basis of IQ sequences into training dataset. These poisoned samples are hidden because it could not be found by traditional classification methods. And poisoned samples are same to samples with triggers which are patched samples in feature space. We show that the hidden backdoor attack can reduce the accuracy of modulation classification significantly with patched samples. At last, we propose activation cluster to detect abnormal samples in training dataset.

摘要: 最近，下行链路已被用于无线通信中，例如调制分类。然而，由于无线信道的开放性和下行链路的不可解释性，它也容易受到对手的攻击。在这种通信中，我们调查了所谓的隐藏后门攻击调制分类，其中对手将精心设计的基于IQ序列的有毒样本放入训练数据集中。这些有毒样本被隐藏起来，因为它不能被传统的分类方法发现。有毒样本与带有触发器的样本相同，后者是特征空间中的修补样本。结果表明，隐藏后门攻击会显著降低修补样本的调制分类准确率。最后，我们提出了激活聚类来检测训练数据集中的异常样本。



## **31. Adversarial Camouflage for Node Injection Attack on Graphs**

图上节点注入攻击的对抗性伪装 cs.LG

Submitted to Information Sciences. Code:  https://github.com/TaoShuchang/CANA

**SubmitDate**: 2023-06-19    [abs](http://arxiv.org/abs/2208.01819v3) [paper-pdf](http://arxiv.org/pdf/2208.01819v3)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Fei Sun, Xueqi Cheng

**Abstract**: Node injection attacks on Graph Neural Networks (GNNs) have received emerging attention due to their potential to significantly degrade GNN performance with high attack success rates. However, our study indicates these attacks often fail in practical scenarios, since defense/detection methods can easily identify and remove the injected nodes. To address this, we devote to camouflage node injection attack, making injected nodes appear normal and imperceptible to defense/detection methods. Unfortunately, the non-Euclidean nature of graph data and lack of intuitive prior present great challenges to the formalization, implementation, and evaluation of camouflage. In this paper, we first propose and define camouflage as distribution similarity between ego networks of injected nodes and normal nodes. Then for implementation, we propose an adversarial CAmouflage framework for Node injection Attack, namely CANA, to improve attack performance under defense/detection methods in practical scenarios. A novel camouflage metric is further designed under the guide of distribution similarity. Extensive experiments demonstrate that CANA can significantly improve the attack performance under defense/detection methods with higher camouflage or imperceptibility. This work urges us to raise awareness of the security vulnerabilities of GNNs in practical applications. The implementation of CANA is available at https://github.com/TaoShuchang/CANA.

摘要: 针对图神经网络(GNN)的节点注入攻击因其攻击成功率高而显著降低GNN性能而受到越来越多的关注。然而，我们的研究表明，这些攻击在实际场景中经常失败，因为防御/检测方法可以很容易地识别和删除注入的节点。为了解决这个问题，我们致力于伪装节点注入攻击，使注入的节点看起来正常，对防御/检测方法是不可察觉的。不幸的是，图形数据的非欧几里德性质和缺乏直观的先验知识给伪装的形式化、实现和评估带来了巨大的挑战。在本文中，我们首先提出并定义伪装为注入节点的EGO网络与正常节点之间的分布相似性。在实现上，我们提出了一种节点注入攻击的对抗性伪装框架CANA，以提高实际场景中防御/检测方法下的攻击性能。在分布相似性的指导下，进一步设计了一种新的伪装度量。大量实验表明，CANA能够显著提高伪装或隐蔽性较高的防御/检测方法下的攻击性能。这项工作促使我们提高对GNN在实际应用中的安全漏洞的认识。CANA的实施可在https://github.com/TaoShuchang/CANA.上获得



## **32. Intriguing Properties of Text-guided Diffusion Models**

文本引导扩散模型的有趣性质 cs.CV

Project page: https://sage-diffusion.github.io/

**SubmitDate**: 2023-06-18    [abs](http://arxiv.org/abs/2306.00974v3) [paper-pdf](http://arxiv.org/pdf/2306.00974v3)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns.

摘要: 文本引导扩散模型(TDM)被广泛应用，但可能会意外失败。常见的故障包括：(I)看起来自然的文本提示生成具有错误内容的图像，或(Ii)潜在变量的不同随机样本，尽管以相同的文本提示为条件，但生成的输出却截然不同，甚至是无关的。在这项工作中，我们旨在更详细地研究和理解TDM的故障模式。为此，我们提出了一种针对TDMS的对抗性攻击方法SAGE，它使用图像分类器作为代理损失函数，在TDMS的离散提示空间和高维潜在空间中进行搜索，自动发现图像生成中的意外行为和失败案例。我们做出了几项技术贡献，以确保SAGE找到扩散模型的故障案例，而不是分类器，并在人体研究中验证这一点。我们的研究揭示了TDM的四个以前没有被系统研究的有趣的性质：(1)我们发现各种自然的文本提示产生的图像无法捕捉到输入文本的语义。根据根本原因，我们将这些故障分为十种不同的类型。(2)我们在潜在空间(不是离群点)中发现了与文本提示无关的导致失真图像的样本，这表明潜在空间的部分结构不是良好的。(3)我们还发现潜在样本导致了与文本提示无关的看起来自然的图像，这意味着潜在空间和提示空间之间存在潜在的错位。(4)通过在输入提示中添加单个对抗性令牌，我们可以生成各种指定的目标对象，而对片段得分的影响最小。这表明了语言表达的脆弱性，并引发了潜在的安全问题。



## **33. Towards A Proactive ML Approach for Detecting Backdoor Poison Samples**

一种主动检测后门毒物样本的最大似然方法 cs.LG

USENIX Security 2023

**SubmitDate**: 2023-06-18    [abs](http://arxiv.org/abs/2205.13616v3) [paper-pdf](http://arxiv.org/pdf/2205.13616v3)

**Authors**: Xiangyu Qi, Tinghao Xie, Jiachen T. Wang, Tong Wu, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Adversaries can embed backdoors in deep learning models by introducing backdoor poison samples into training datasets. In this work, we investigate how to detect such poison samples to mitigate the threat of backdoor attacks. First, we uncover a post-hoc workflow underlying most prior work, where defenders passively allow the attack to proceed and then leverage the characteristics of the post-attacked model to uncover poison samples. We reveal that this workflow does not fully exploit defenders' capabilities, and defense pipelines built on it are prone to failure or performance degradation in many scenarios. Second, we suggest a paradigm shift by promoting a proactive mindset in which defenders engage proactively with the entire model training and poison detection pipeline, directly enforcing and magnifying distinctive characteristics of the post-attacked model to facilitate poison detection. Based on this, we formulate a unified framework and provide practical insights on designing detection pipelines that are more robust and generalizable. Third, we introduce the technique of Confusion Training (CT) as a concrete instantiation of our framework. CT applies an additional poisoning attack to the already poisoned dataset, actively decoupling benign correlation while exposing backdoor patterns to detection. Empirical evaluations on 4 datasets and 14 types of attacks validate the superiority of CT over 14 baseline defenses.

摘要: 攻击者可以通过将后门毒药样本引入训练数据集中，在深度学习模型中嵌入后门。在这项工作中，我们研究如何检测此类毒物样本以减轻后门攻击的威胁。首先，我们揭示了大多数先前工作背后的事后工作流，在这种工作中，防御者被动地允许攻击继续进行，然后利用攻击后模型的特征来发现毒物样本。我们发现，这种工作流没有充分利用防御者的能力，在其上构建的防御管道在许多场景下容易出现故障或性能下降。其次，我们建议通过促进一种积极主动的心态来实现范式转变，在这种心态中，防御者主动参与整个模型培训和毒物检测管道，直接实施和放大攻击后模型的独特特征，以促进毒物检测。在此基础上，我们制定了一个统一的框架，并为设计更健壮和更具通用性的检测管道提供了实用的见解。第三，我们引入混淆训练(CT)技术作为我们的框架的具体实例。CT对已经中毒的数据集进行额外的中毒攻击，主动去耦合良性关联，同时暴露后门模式以供检测。在4个数据集和14种攻击类型上的经验评估验证了CT相对于14种基线防御的优越性。



## **34. Adversaries with Limited Information in the Friedkin--Johnsen Model**

Friedkin-Johnsen模型中信息有限的对手 cs.SI

To appear at KDD'23

**SubmitDate**: 2023-06-17    [abs](http://arxiv.org/abs/2306.10313v1) [paper-pdf](http://arxiv.org/pdf/2306.10313v1)

**Authors**: Sijing Tu, Stefan Neumann, Aristides Gionis

**Abstract**: In recent years, online social networks have been the target of adversaries who seek to introduce discord into societies, to undermine democracies and to destabilize communities. Often the goal is not to favor a certain side of a conflict but to increase disagreement and polarization. To get a mathematical understanding of such attacks, researchers use opinion-formation models from sociology, such as the Friedkin--Johnsen model, and formally study how much discord the adversary can produce when altering the opinions for only a small set of users. In this line of work, it is commonly assumed that the adversary has full knowledge about the network topology and the opinions of all users. However, the latter assumption is often unrealistic in practice, where user opinions are not available or simply difficult to estimate accurately.   To address this concern, we raise the following question: Can an attacker sow discord in a social network, even when only the network topology is known? We answer this question affirmatively. We present approximation algorithms for detecting a small set of users who are highly influential for the disagreement and polarization in the network. We show that when the adversary radicalizes these users and if the initial disagreement/polarization in the network is not very high, then our method gives a constant-factor approximation on the setting when the user opinions are known. To find the set of influential users, we provide a novel approximation algorithm for a variant of MaxCut in graphs with positive and negative edge weights. We experimentally evaluate our methods, which have access only to the network topology, and we find that they have similar performance as methods that have access to the network topology and all user opinions. We further present an NP-hardness proof, which was an open question by Chen and Racz [IEEE Trans. Netw. Sci. Eng., 2021].

摘要: 近年来，在线社交网络一直是试图在社会中制造不和谐、破坏民主和破坏社区稳定的敌人的目标。通常，目标不是偏袒冲突的某一方，而是增加分歧和两极分化。为了从数学上理解这类攻击，研究人员使用了社会学中的观点形成模型，如弗里德金-约翰森模型，并正式研究了当对手只为一小部分用户改变观点时，可以产生多大的不和谐。在这方面的工作中，通常假设对手完全了解网络拓扑和所有用户的意见。然而，后一种假设在实践中往往是不现实的，因为用户的意见是不可用的，或者只是很难准确估计。为了解决这一问题，我们提出了以下问题：即使只知道网络拓扑，攻击者也能在社交网络中挑拨离间吗？我们肯定地回答了这个问题。我们提出了一种近似算法，用于检测对网络中的不一致和极化有很大影响的一小部分用户。我们证明，当对手激化这些用户时，如果网络中的初始分歧/极化不是很高，那么当用户意见已知时，我们的方法在设置上给出一个恒定因子近似。为了寻找有影响力的用户集，我们给出了一种新的算法，用于计算边权为正负的图中MaxCut的一种变种。我们对我们的方法进行了实验评估，这些方法只能访问网络拓扑，我们发现它们具有与可以访问网络拓扑和所有用户意见的方法类似的性能。我们进一步给出了NP-硬度证明，这是Chen和racz[IEEE译文]提出的一个未决问题。奈特。SCI。Eng.，2021]。



## **35. Edge Learning for 6G-enabled Internet of Things: A Comprehensive Survey of Vulnerabilities, Datasets, and Defenses**

支持6G的物联网的边缘学习：漏洞、数据集和防御的全面调查 cs.CR

**SubmitDate**: 2023-06-17    [abs](http://arxiv.org/abs/2306.10309v1) [paper-pdf](http://arxiv.org/pdf/2306.10309v1)

**Authors**: Mohamed Amine Ferrag, Othmane Friha, Burak Kantarci, Norbert Tihanyi, Lucas Cordeiro, Merouane Debbah, Djallel Hamouda, Muna Al-Hawawreh, Kim-Kwang Raymond Choo

**Abstract**: The ongoing deployment of the fifth generation (5G) wireless networks constantly reveals limitations concerning its original concept as a key driver of Internet of Everything (IoE) applications. These 5G challenges are behind worldwide efforts to enable future networks, such as sixth generation (6G) networks, to efficiently support sophisticated applications ranging from autonomous driving capabilities to the Metaverse. Edge learning is a new and powerful approach to training models across distributed clients while protecting the privacy of their data. This approach is expected to be embedded within future network infrastructures, including 6G, to solve challenging problems such as resource management and behavior prediction. This survey article provides a holistic review of the most recent research focused on edge learning vulnerabilities and defenses for 6G-enabled IoT. We summarize the existing surveys on machine learning for 6G IoT security and machine learning-associated threats in three different learning modes: centralized, federated, and distributed. Then, we provide an overview of enabling emerging technologies for 6G IoT intelligence. Moreover, we provide a holistic survey of existing research on attacks against machine learning and classify threat models into eight categories, including backdoor attacks, adversarial examples, combined attacks, poisoning attacks, Sybil attacks, byzantine attacks, inference attacks, and dropping attacks. In addition, we provide a comprehensive and detailed taxonomy and a side-by-side comparison of the state-of-the-art defense methods against edge learning vulnerabilities. Finally, as new attacks and defense technologies are realized, new research and future overall prospects for 6G-enabled IoT are discussed.

摘要: 作为万物互联(IoE)应用的关键驱动力，第五代(5G)无线网络的持续部署不断暴露出其原始概念的局限性。这些5G挑战是全球努力的背后，目的是使未来的网络(如第六代(6G)网络)能够有效支持从自动驾驶功能到Metverse的各种复杂应用。边缘学习是一种新的、功能强大的方法，用于跨分布式客户端训练模型，同时保护其数据的隐私。这种方法有望嵌入包括6G在内的未来网络基础设施，以解决资源管理和行为预测等具有挑战性的问题。本调查文章全面回顾了有关支持6G的物联网的边缘学习漏洞和防御的最新研究。我们总结了机器学习在集中式、联合式和分布式三种不同的学习模式下对6G物联网安全和机器学习相关威胁的现有研究。然后，我们将概述支持6G物联网智能的新兴技术。此外，我们对机器学习攻击的现有研究进行了全面的综述，并将威胁模型分为八类，包括后门攻击、对抗性例子、组合攻击、中毒攻击、Sybil攻击、拜占庭攻击、推理攻击和丢弃攻击。此外，我们还提供了全面而详细的分类以及针对边缘学习漏洞的最先进防御方法的并列比较。最后，随着新的攻击和防御技术的实现，讨论了6G物联网的新研究和未来的整体前景。



## **36. You Don't Need Robust Machine Learning to Manage Adversarial Attack Risks**

您不需要健壮的机器学习来管理对手攻击风险 cs.LG

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09951v1) [paper-pdf](http://arxiv.org/pdf/2306.09951v1)

**Authors**: Edward Raff, Michel Benaroch, Andrew L. Farris

**Abstract**: The robustness of modern machine learning (ML) models has become an increasing concern within the community. The ability to subvert a model into making errant predictions using seemingly inconsequential changes to input is startling, as is our lack of success in building models robust to this concern. Existing research shows progress, but current mitigations come with a high cost and simultaneously reduce the model's accuracy. However, such trade-offs may not be necessary when other design choices could subvert the risk. In this survey we review the current literature on attacks and their real-world occurrences, or limited evidence thereof, to critically evaluate the real-world risks of adversarial machine learning (AML) for the average entity. This is done with an eye toward how one would then mitigate these attacks in practice, the risks for production deployment, and how those risks could be managed. In doing so we elucidate that many AML threats do not warrant the cost and trade-offs of robustness due to a low likelihood of attack or availability of superior non-ML mitigations. Our analysis also recommends cases where an actor should be concerned about AML to the degree where robust ML models are necessary for a complete deployment.

摘要: 现代机器学习(ML)模型的稳健性已经成为社区内越来越关注的问题。通过对输入进行看似无关紧要的改变来颠覆模型做出错误预测的能力令人震惊，就像我们在构建针对这种担忧的强大模型方面缺乏成功一样。现有研究表明取得了进展，但目前的缓解措施代价高昂，同时降低了模型的准确性。然而，当其他设计选择可能会颠覆风险时，这样的权衡可能没有必要。在这项调查中，我们回顾了当前关于攻击及其真实世界发生的文献，或其有限的证据，以批判性地评估对抗性机器学习(AML)对于普通实体的真实世界风险。这样做的目的是考虑到如何在实践中减轻这些攻击、生产部署的风险以及如何管理这些风险。在此过程中，我们阐明了许多AML威胁并不保证健壮性的成本和权衡，因为攻击的可能性很低，或者可以获得更好的非ML缓解措施。我们的分析还推荐了参与者应该关注AML的情况，在这种程度上，健壮的ML模型是完整部署所必需的。



## **37. Adversarial Cheap Talk**

对抗性的低级谈资 cs.LG

To be published at ICML 2023. Project video and code are available at  https://sites.google.com/view/adversarial-cheap-talk

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2211.11030v3) [paper-pdf](http://arxiv.org/pdf/2211.11030v3)

**Authors**: Chris Lu, Timon Willi, Alistair Letcher, Jakob Foerster

**Abstract**: Adversarial attacks in reinforcement learning (RL) often assume highly-privileged access to the victim's parameters, environment, or data. Instead, this paper proposes a novel adversarial setting called a Cheap Talk MDP in which an Adversary can merely append deterministic messages to the Victim's observation, resulting in a minimal range of influence. The Adversary cannot occlude ground truth, influence underlying environment dynamics or reward signals, introduce non-stationarity, add stochasticity, see the Victim's actions, or access their parameters. Additionally, we present a simple meta-learning algorithm called Adversarial Cheap Talk (ACT) to train Adversaries in this setting. We demonstrate that an Adversary trained with ACT still significantly influences the Victim's training and testing performance, despite the highly constrained setting. Affecting train-time performance reveals a new attack vector and provides insight into the success and failure modes of existing RL algorithms. More specifically, we show that an ACT Adversary is capable of harming performance by interfering with the learner's function approximation, or instead helping the Victim's performance by outputting useful features. Finally, we show that an ACT Adversary can manipulate messages during train-time to directly and arbitrarily control the Victim at test-time. Project video and code are available at https://sites.google.com/view/adversarial-cheap-talk

摘要: 强化学习(RL)中的对抗性攻击通常假定具有访问受害者参数、环境或数据的高度特权。相反，本文提出了一种新的对抗性环境，称为廉价谈话MDP，在该环境中，对手只需将确定性消息附加到受害者的观察中，从而产生最小的影响范围。敌手不能掩盖基本事实、影响潜在环境动态或奖励信号、引入非平稳性、增加随机性、看到受害者的行为或获取他们的参数。此外，我们还提出了一个简单的元学习算法，称为对抗性廉价谈话(ACT)，以在这种情况下训练对手。我们证明，尽管在高度受限的环境下，接受过ACT训练的对手仍然会显著影响受害者的训练和测试表现。影响训练时间性能揭示了新的攻击向量，并提供了对现有RL算法的成功和失败模式的洞察。更具体地说，我们证明了ACT对手能够通过干扰学习者的函数逼近来损害性能，或者相反地通过输出有用的特征来帮助受害者的性能。最后，我们证明了ACT攻击者可以在训练时间内操纵消息，从而在测试时间直接任意控制受害者。项目视频和代码可在https://sites.google.com/view/adversarial-cheap-talk上获得



## **38. Query-Free Evasion Attacks Against Machine Learning-Based Malware Detectors with Generative Adversarial Networks**

基于产生式对抗网络的机器学习恶意软件检测器的无查询逃避攻击 cs.CR

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09925v1) [paper-pdf](http://arxiv.org/pdf/2306.09925v1)

**Authors**: Daniel Gibert, Jordi Planes, Quan Le, Giulio Zizzo

**Abstract**: Malware detectors based on machine learning (ML) have been shown to be susceptible to adversarial malware examples. However, current methods to generate adversarial malware examples still have their limits. They either rely on detailed model information (gradient-based attacks), or on detailed outputs of the model - such as class probabilities (score-based attacks), neither of which are available in real-world scenarios. Alternatively, adversarial examples might be crafted using only the label assigned by the detector (label-based attack) to train a substitute network or an agent using reinforcement learning. Nonetheless, label-based attacks might require querying a black-box system from a small number to thousands of times, depending on the approach, which might not be feasible against malware detectors. This work presents a novel query-free approach to craft adversarial malware examples to evade ML-based malware detectors. To this end, we have devised a GAN-based framework to generate adversarial malware examples that look similar to benign executables in the feature space. To demonstrate the suitability of our approach we have applied the GAN-based attack to three common types of features usually employed by static ML-based malware detectors: (1) Byte histogram features, (2) API-based features, and (3) String-based features. Results show that our model-agnostic approach performs on par with MalGAN, while generating more realistic adversarial malware examples without requiring any query to the malware detectors. Furthermore, we have tested the generated adversarial examples against state-of-the-art multimodal and deep learning malware detectors, showing a decrease in detection performance, as well as a decrease in the average number of detections by the anti-malware engines in VirusTotal.

摘要: 基于机器学习(ML)的恶意软件检测器已被证明容易受到敌意恶意软件示例的影响。然而，当前生成敌意恶意软件示例的方法仍然有其局限性。它们要么依赖于详细的模型信息(基于梯度的攻击)，要么依赖于模型的详细输出--例如类别概率(基于分数的攻击)，这两者在现实世界的场景中都不可用。或者，可以仅使用检测器分配的标签(基于标签的攻击)来制作对抗性示例，以使用强化学习来训练替代网络或代理。尽管如此，基于标签的攻击可能需要查询黑匣子系统从少量到数千次，具体取决于方法，这在恶意软件检测器面前可能是不可行的。这项工作提出了一种新的无查询方法来构建恶意软件示例，以躲避基于ML的恶意软件检测器。为此，我们设计了一个基于GAN的框架来生成在特征空间中看起来类似于良性可执行文件的敌意恶意软件示例。为了证明我们的方法的适用性，我们将基于GAN的攻击应用于基于静态ML的恶意软件检测器通常使用的三种常见特征：(1)字节直方图特征，(2)基于API的特征，和(3)基于字符串的特征。结果表明，我们的模型无关方法的性能与MalGan相当，同时生成更真实的敌意恶意软件示例，而不需要向恶意软件检测器进行任何查询。此外，我们已经针对最先进的多模式和深度学习恶意软件检测器测试了生成的恶意示例，显示出检测性能的下降，以及VirusTotal中反恶意软件引擎的平均检测次数的下降。



## **39. Wasserstein distributional robustness of neural networks**

神经网络的Wasserstein分布稳健性 cs.LG

23 pages, 6 figures, 8 tables

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2306.09844v1) [paper-pdf](http://arxiv.org/pdf/2306.09844v1)

**Authors**: Xingjian Bai, Guangyi He, Yifan Jiang, Jan Obloj

**Abstract**: Deep neural networks are known to be vulnerable to adversarial attacks (AA). For an image recognition task, this means that a small perturbation of the original can result in the image being misclassified. Design of such attacks as well as methods of adversarial training against them are subject of intense research. We re-cast the problem using techniques of Wasserstein distributionally robust optimization (DRO) and obtain novel contributions leveraging recent insights from DRO sensitivity analysis. We consider a set of distributional threat models. Unlike the traditional pointwise attacks, which assume a uniform bound on perturbation of each input data point, distributional threat models allow attackers to perturb inputs in a non-uniform way. We link these more general attacks with questions of out-of-sample performance and Knightian uncertainty. To evaluate the distributional robustness of neural networks, we propose a first-order AA algorithm and its multi-step version. Our attack algorithms include Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) as special cases. Furthermore, we provide a new asymptotic estimate of the adversarial accuracy against distributional threat models. The bound is fast to compute and first-order accurate, offering new insights even for the pointwise AA. It also naturally yields out-of-sample performance guarantees. We conduct numerical experiments on the CIFAR-10 dataset using DNNs on RobustBench to illustrate our theoretical results. Our code is available at https://github.com/JanObloj/W-DRO-Adversarial-Methods.

摘要: 众所周知，深度神经网络容易受到对手攻击(AA)。对于图像识别任务，这意味着原始图像的微小扰动可能会导致图像被错误分类。这种攻击的设计以及对抗它们的对抗性训练方法都是深入研究的主题。我们使用沃瑟斯坦分布稳健优化(DRO)技术重塑了这个问题，并利用DRO敏感性分析的最新见解获得了新的贡献。我们考虑一组分布式威胁模型。与传统的逐点攻击不同，分布式威胁模型允许攻击者以非一致的方式干扰输入。我们将这些更普遍的攻击与超出样本的性能和奈特的不确定性联系在一起。为了评估神经网络的分布稳健性，我们提出了一种一阶AA算法及其多步算法。我们的攻击算法包括快速梯度符号法(FGSM)和投影梯度下降法(PGD)作为特例。此外，我们还对分布式威胁模型的对抗精度给出了一个新的渐近估计。边界计算速度快，一阶精度高，即使对于逐点的AA也提供了新的见解。它还自然而然地产生了超出样本的性能保证。为了验证我们的理论结果，我们在CIFAR-10数据集上进行了数值实验。我们的代码可以在https://github.com/JanObloj/W-DRO-Adversarial-Methods.上找到



## **40. TransFool: An Adversarial Attack against Neural Machine Translation Models**

TransFool：对神经机器翻译模型的敌意攻击 cs.CL

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2302.00944v2) [paper-pdf](http://arxiv.org/pdf/2302.00944v2)

**Authors**: Sahar Sadrizadeh, Ljiljana Dolamic, Pascal Frossard

**Abstract**: Deep neural networks have been shown to be vulnerable to small perturbations of their inputs, known as adversarial attacks. In this paper, we investigate the vulnerability of Neural Machine Translation (NMT) models to adversarial attacks and propose a new attack algorithm called TransFool. To fool NMT models, TransFool builds on a multi-term optimization problem and a gradient projection step. By integrating the embedding representation of a language model, we generate fluent adversarial examples in the source language that maintain a high level of semantic similarity with the clean samples. Experimental results demonstrate that, for different translation tasks and NMT architectures, our white-box attack can severely degrade the translation quality while the semantic similarity between the original and the adversarial sentences stays high. Moreover, we show that TransFool is transferable to unknown target models. Finally, based on automatic and human evaluations, TransFool leads to improvement in terms of success rate, semantic similarity, and fluency compared to the existing attacks both in white-box and black-box settings. Thus, TransFool permits us to better characterize the vulnerability of NMT models and outlines the necessity to design strong defense mechanisms and more robust NMT systems for real-life applications.

摘要: 深度神经网络已被证明容易受到其输入的微小扰动，即所谓的对抗性攻击。本文研究了神经机器翻译(NMT)模型对敌意攻击的脆弱性，提出了一种新的攻击算法TransFool。为了愚弄NMT模型，TransFool建立在多项优化问题和梯度投影步骤的基础上。通过集成语言模型的嵌入表示，我们在源语言中生成流畅的对抗性实例，这些实例与干净的样本保持较高的语义相似度。实验结果表明，对于不同的翻译任务和自然机器翻译体系结构，我们的白盒攻击可以在原句和对抗性句子之间保持较高语义相似度的情况下，严重降低翻译质量。此外，我们还证明了TransFool可以转移到未知目标模型。最后，基于自动和人工评估，TransFool在成功率、语义相似度和流畅度方面都比现有的白盒和黑盒攻击都有所提高。因此，TransFool使我们能够更好地刻画NMT模型的脆弱性，并概述为现实应用设计强大的防御机制和更健壮的NMT系统的必要性。



## **41. Fact-Saboteurs: A Taxonomy of Evidence Manipulation Attacks against Fact-Verification Systems**

事实破坏者：针对事实核查系统的证据操纵攻击的分类 cs.CR

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2209.03755v4) [paper-pdf](http://arxiv.org/pdf/2209.03755v4)

**Authors**: Sahar Abdelnabi, Mario Fritz

**Abstract**: Mis- and disinformation are a substantial global threat to our security and safety. To cope with the scale of online misinformation, researchers have been working on automating fact-checking by retrieving and verifying against relevant evidence. However, despite many advances, a comprehensive evaluation of the possible attack vectors against such systems is still lacking. Particularly, the automated fact-verification process might be vulnerable to the exact disinformation campaigns it is trying to combat. In this work, we assume an adversary that automatically tampers with the online evidence in order to disrupt the fact-checking model via camouflaging the relevant evidence or planting a misleading one. We first propose an exploratory taxonomy that spans these two targets and the different threat model dimensions. Guided by this, we design and propose several potential attack methods. We show that it is possible to subtly modify claim-salient snippets in the evidence and generate diverse and claim-aligned evidence. Thus, we highly degrade the fact-checking performance under many different permutations of the taxonomy's dimensions. The attacks are also robust against post-hoc modifications of the claim. Our analysis further hints at potential limitations in models' inference when faced with contradicting evidence. We emphasize that these attacks can have harmful implications on the inspectable and human-in-the-loop usage scenarios of such models, and we conclude by discussing challenges and directions for future defenses.

摘要: 错误和虚假信息是对我们的安全和安全的重大全球威胁。为了应对网上虚假信息的规模，研究人员一直在致力于通过检索和验证相关证据来实现事实核查的自动化。然而，尽管取得了许多进展，但仍然缺乏对针对此类系统的可能攻击媒介的全面评估。特别是，自动化的事实核查过程可能容易受到它试图打击的虚假信息运动的影响。在这项工作中，我们假设一个对手自动篡改在线证据，以便通过伪装相关证据或植入误导性证据来扰乱事实核查模型。我们首先提出了一种探索性分类，该分类跨越这两个目标和不同的威胁模型维度。在此指导下，我们设计并提出了几种潜在的攻击方法。我们表明，可以巧妙地修改证据中突出声明的片段，并生成多样化的与声明一致的证据。因此，在分类维度的许多不同排列下，我们极大地降低了事实检查的性能。这些攻击也对索赔的事后修改具有很强的抵御能力。我们的分析进一步暗示，在面对相互矛盾的证据时，模型的推理可能存在局限性。我们强调，这些攻击可能会对此类模型的可检查和人在环中使用场景产生有害影响，我们最后讨论了未来防御的挑战和方向。



## **42. Adversarial Image Color Transformations in Explicit Color Filter Space**

显式滤色空间中的对抗性图像颜色变换 cs.CV

Published at IEEE Transactions on Information Forensics and Security  2023. Code is available at  https://github.com/ZhengyuZhao/ACE/tree/master/Journal_version

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2011.06690v3) [paper-pdf](http://arxiv.org/pdf/2011.06690v3)

**Authors**: Zhengyu Zhao, Zhuoran Liu, Martha Larson

**Abstract**: Deep Neural Networks have been shown to be vulnerable to adversarial images. Conventional attacks strive for indistinguishable adversarial images with strictly restricted perturbations. Recently, researchers have moved to explore distinguishable yet non-suspicious adversarial images and demonstrated that color transformation attacks are effective. In this work, we propose Adversarial Color Filter (AdvCF), a novel color transformation attack that is optimized with gradient information in the parameter space of a simple color filter. In particular, our color filter space is explicitly specified so that we are able to provide a systematic analysis of model robustness against adversarial color transformations, from both the attack and defense perspectives. In contrast, existing color transformation attacks do not offer the opportunity for systematic analysis due to the lack of such an explicit space. We further demonstrate the effectiveness of our AdvCF in fooling image classifiers and also compare it with other color transformation attacks regarding their robustness to defenses and image acceptability through an extensive user study. We also highlight the human-interpretability of AdvCF and show its superiority over the state-of-the-art human-interpretable color transformation attack on both image acceptability and efficiency. Additional results provide interesting new insights into model robustness against AdvCF in another three visual tasks.

摘要: 深度神经网络已被证明容易受到敌意图像的影响。传统的攻击努力获得带有严格限制的扰动的难以区分的对抗性图像。最近，研究人员已经开始探索可区分但不可疑的对抗性图像，并证明了颜色变换攻击是有效的。在这项工作中，我们提出了对抗颜色过滤器(AdvCF)，这是一种新的颜色变换攻击，它利用简单颜色过滤器参数空间中的梯度信息进行优化。特别是，我们的滤色器空间被明确指定，以便我们能够从攻击和防御两个角度提供针对对抗性颜色变换的模型稳健性的系统分析。相比之下，现有的颜色变换攻击由于缺乏这种明确的空间而没有提供系统分析的机会。我们进一步证明了我们的AdvCF在愚弄图像分类器方面的有效性，并通过广泛的用户研究将其与其他颜色变换攻击在防御和图像可接受性方面的鲁棒性进行了比较。我们还强调了AdvCF的人类可解释性，并展示了它在图像可接受性和效率方面相对于最先进的人类可解释颜色变换攻击的优越性。其他结果为在另外三个可视化任务中针对AdvCF的模型健壮性提供了有趣的新见解。



## **43. Distributed Energy Resources Cybersecurity Outlook: Vulnerabilities, Attacks, Impacts, and Mitigations**

分布式能源网络安全展望：漏洞、攻击、影响和缓解 cs.CR

IEEE Systems Journal

**SubmitDate**: 2023-06-16    [abs](http://arxiv.org/abs/2205.11171v3) [paper-pdf](http://arxiv.org/pdf/2205.11171v3)

**Authors**: Ioannis Zografopoulos, Nikos D. Hatziargyriou, Charalambos Konstantinou

**Abstract**: The digitization and decentralization of the electric power grid are key thrusts for an economically and environmentally sustainable future. Towards this goal, distributed energy resources (DER), including rooftop solar panels, battery storage, electric vehicles, etc., are becoming ubiquitous in power systems. Power utilities benefit from DERs as they minimize operational costs; at the same time, DERs grant users and aggregators control over the power they produce and consume. DERs are interconnected, interoperable, and support remotely controllable features, thus, their cybersecurity is of cardinal importance. DER communication dependencies and the diversity of DER architectures widen the threat surface and aggravate the cybersecurity posture of power systems. In this work, we focus on security oversights that reside in the cyber and physical layers of DERs and can jeopardize grid operations. Existing works have underlined the impact of cyberattacks targeting DER assets, however, they either focus on specific system components (e.g., communication protocols), do not consider the mission-critical objectives of DERs, or neglect the adversarial perspective (e.g., adversary/attack models) altogether. To address these omissions, we comprehensively analyze adversarial capabilities and objectives when manipulating DER assets, and then present how protocol and device-level vulnerabilities can materialize into cyberattacks impacting power system operations. Finally, we provide mitigation strategies to thwart adversaries and directions for future DER cybersecurity research.

摘要: 电网的数字化和分散化是实现经济和环境可持续未来的关键推动力。为了实现这一目标，分布式能源(DER)在电力系统中变得无处不在，包括屋顶太阳能电池板、电池储存、电动汽车等。电力公用事业受益于DER，因为它们最大限度地降低了运营成本；同时，DER使用户和聚合器能够控制他们生产和消耗的电力。DER是互联的、可互操作的，并支持远程控制的功能，因此，其网络安全至关重要。DER通信的依赖性和DER体系结构的多样性扩大了威胁面，加剧了电力系统的网络安全态势。在这项工作中，我们重点关注驻留在DER的网络层和物理层并可能危及电网运营的安全疏忽。现有的工作强调了针对DER资产的网络攻击的影响，然而，它们要么专注于特定的系统组件(例如，通信协议)，没有考虑DER的关键任务目标，要么完全忽视了对抗性的观点(例如，对手/攻击模型)。为了解决这些疏漏，我们全面分析了操纵DER资产时的对抗能力和目标，然后介绍了协议和设备级漏洞如何转化为影响电力系统运行的网络攻击。最后，我们提供了挫败对手的缓解策略和未来网络安全研究的方向。



## **44. Inroads into Autonomous Network Defence using Explained Reinforcement Learning**

基于解释强化学习的自主网络防御研究 cs.CR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09318v1) [paper-pdf](http://arxiv.org/pdf/2306.09318v1)

**Authors**: Myles Foley, Mia Wang, Zoe M, Chris Hicks, Vasilios Mavroudis

**Abstract**: Computer network defence is a complicated task that has necessitated a high degree of human involvement. However, with recent advancements in machine learning, fully autonomous network defence is becoming increasingly plausible. This paper introduces an end-to-end methodology for studying attack strategies, designing defence agents and explaining their operation. First, using state diagrams, we visualise adversarial behaviour to gain insight about potential points of intervention and inform the design of our defensive models. We opt to use a set of deep reinforcement learning agents trained on different parts of the task and organised in a shallow hierarchy. Our evaluation shows that the resulting design achieves a substantial performance improvement compared to prior work. Finally, to better investigate the decision-making process of our agents, we complete our analysis with a feature ablation and importance study.

摘要: 计算机网络防御是一项复杂的任务，需要高度的人工参与。然而，随着最近机器学习的进步，完全自主的网络防御正变得越来越有可能。本文介绍了一种端到端的方法，用于研究攻击策略、设计防御代理并解释它们的操作。首先，我们使用状态图将敌对行为形象化，以洞察潜在的干预点，并为我们的防御模型的设计提供信息。我们选择使用一组针对任务不同部分进行培训的深度强化学习代理，并以浅层次进行组织。我们的评估结果表明，与以前的工作相比，所得到的设计实现了显著的性能改进。最后，为了更好地研究我们的代理的决策过程，我们用特征消融和重要性研究来完成我们的分析。



## **45. DIFFender: Diffusion-Based Adversarial Defense against Patch Attacks in the Physical World**

DIFFender：物理世界中基于扩散的对抗性防御补丁攻击 cs.CV

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.09124v1) [paper-pdf](http://arxiv.org/pdf/2306.09124v1)

**Authors**: Caixin Kang, Yinpeng Dong, Zhengyi Wang, Shouwei Ruan, Hang Su, Xingxing Wei

**Abstract**: Adversarial attacks in the physical world, particularly patch attacks, pose significant threats to the robustness and reliability of deep learning models. Developing reliable defenses against patch attacks is crucial for real-world applications, yet current research in this area is severely lacking. In this paper, we propose DIFFender, a novel defense method that leverages the pre-trained diffusion model to perform both localization and defense against potential adversarial patch attacks. DIFFender is designed as a pipeline consisting of two main stages: patch localization and restoration. In the localization stage, we exploit the intriguing properties of a diffusion model to effectively identify the locations of adversarial patches. In the restoration stage, we employ a text-guided diffusion model to eliminate adversarial regions in the image while preserving the integrity of the visual content. Additionally, we design a few-shot prompt-tuning algorithm to facilitate simple and efficient tuning, enabling the learned representations to easily transfer to downstream tasks, which optimize two stages jointly. We conduct extensive experiments on image classification and face recognition to demonstrate that DIFFender exhibits superior robustness under strong adaptive attacks and generalizes well across various scenarios, diverse classifiers, and multiple attack methods.

摘要: 物理世界中的对抗性攻击，特别是补丁攻击，对深度学习模型的健壮性和可靠性构成了严重威胁。开发针对补丁攻击的可靠防御对于现实世界的应用至关重要，但目前这一领域的研究严重缺乏。在本文中，我们提出了一种新的防御方法DIFFender，它利用预先训练的扩散模型来定位和防御潜在的敌意补丁攻击。DIFFender被设计为一个由两个主要阶段组成的管道：补丁定位和恢复。在本地化阶段，我们利用扩散模型的有趣性质来有效地识别敌方补丁的位置。在恢复阶段，我们使用文本引导的扩散模型来消除图像中的对抗性区域，同时保持视觉内容的完整性。此外，我们设计了几个镜头的提示调整算法，以便于简单有效的调整，使学习到的表示可以很容易地转移到下游任务，共同优化两个阶段。我们在图像分类和人脸识别上进行了大量的实验，证明了DIFFender在强自适应攻击下表现出了良好的鲁棒性，并且能够很好地适用于各种场景、不同的分类器和多种攻击方法。



## **46. The Effect of Length on Key Fingerprint Verification Security and Usability**

长度对密钥指纹验证安全性和可用性的影响 cs.CR

Accepted to International Conference on Availability, Reliability and  Security (ARES 2023)

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.04574v2) [paper-pdf](http://arxiv.org/pdf/2306.04574v2)

**Authors**: Dan Turner, Siamak F. Shahandashti, Helen Petrie

**Abstract**: In applications such as end-to-end encrypted instant messaging, secure email, and device pairing, users need to compare key fingerprints to detect impersonation and adversary-in-the-middle attacks. Key fingerprints are usually computed as truncated hashes of each party's view of the channel keys, encoded as an alphanumeric or numeric string, and compared out-of-band, e.g. manually, to detect any inconsistencies. Previous work has extensively studied the usability of various verification strategies and encoding formats, however, the exact effect of key fingerprint length on the security and usability of key fingerprint verification has not been rigorously investigated. We present a 162-participant study on the effect of numeric key fingerprint length on comparison time and error rate. While the results confirm some widely-held intuitions such as general comparison times and errors increasing significantly with length, a closer look reveals interesting nuances. The significant rise in comparison time only occurs when highly similar fingerprints are compared, and comparison time remains relatively constant otherwise. On errors, our results clearly distinguish between security non-critical errors that remain low irrespective of length and security critical errors that significantly rise, especially at higher fingerprint lengths. A noteworthy implication of this latter result is that Signal/WhatsApp key fingerprints provide a considerably lower level of security than usually assumed.

摘要: 在端到端加密即时消息、安全电子邮件和设备配对等应用中，用户需要比较密钥指纹来检测模仿和中间人攻击。密钥指纹通常被计算为每一方的频道密钥视图的截断散列，被编码为字母数字或数字字符串，并例如手动地进行带外比较以检测任何不一致。以往的工作已经广泛地研究了各种验证策略和编码格式的可用性，但还没有严格地研究密钥指纹长度对密钥指纹验证的安全性和可用性的确切影响。我们对162名参与者进行了一项关于数字密钥指纹长度对比较时间和错误率的影响的研究。虽然结果证实了一些普遍存在的直觉，如一般的比较时间和误差随着长度的增加而显著增加，但仔细观察会发现有趣的细微差别。只有当比较高度相似的指纹时，比较时间才会显著增加，否则比较时间保持相对恒定。在错误方面，我们的结果清楚地区分了无论长度如何都保持较低的安全非关键错误和显著上升的安全关键错误，特别是在较长的指纹长度时。后一种结果的一个值得注意的含义是，Signal/WhatsApp密钥指纹提供的安全级别比通常假设的要低得多。



## **47. Community Detection Attack against Collaborative Learning-based Recommender Systems**

基于协作学习的推荐系统的社区检测攻击 cs.IR

**SubmitDate**: 2023-06-15    [abs](http://arxiv.org/abs/2306.08929v1) [paper-pdf](http://arxiv.org/pdf/2306.08929v1)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning based recommender systems emerged following the success of collaborative learning techniques such as Federated Learning (FL) and Gossip Learning (GL). In these systems, users participate in the training of a recommender system while keeping their history of consumed items on their devices. While these solutions seemed appealing for preserving the privacy of the participants at a first glance, recent studies have shown that collaborative learning can be vulnerable to a variety of privacy attacks. In this paper we propose a novel privacy attack called Community Detection Attack (CDA), which allows an adversary to discover the members of a community based on a set of items of her choice (e.g., discovering users interested in LGBT content). Through experiments on three real recommendation datasets and by using two state-of-the-art recommendation models, we assess the sensitivity of an FL-based recommender system as well as two flavors of Gossip Learning-based recommender systems to CDA. Results show that on all models and all datasets, the FL setting is more vulnerable to CDA than Gossip settings. We further evaluated two off-the-shelf mitigation strategies, namely differential privacy (DP) and a share less policy, which consists in sharing a subset of model parameters. Results show a better privacy-utility trade-off for the share less policy compared to DP especially in the Gossip setting.

摘要: 基于协作学习的推荐系统是在联邦学习(FL)和八卦学习(GL)等协作学习技术成功之后应运而生的。在这些系统中，用户参与推荐系统的培训，同时在他们的设备上保存他们的消费项目的历史。虽然这些解决方案乍一看似乎在保护参与者的隐私方面很有吸引力，但最近的研究表明，协作学习可能容易受到各种隐私攻击。在本文中，我们提出了一种新的隐私攻击，称为社区检测攻击(CDA)，它允许攻击者根据她选择的一组项目来发现社区成员(例如，发现对LGBT内容感兴趣的用户)。通过在三个真实推荐数据集上的实验，使用两种最新的推荐模型，我们评估了一个基于FL的推荐系统以及两种基于八卦学习的推荐系统对CDA的敏感度。结果表明，在所有模型和所有数据集上，FL设置比八卦设置更容易受到CDA的影响。我们进一步评估了两种现成的缓解策略，即差异隐私(DP)策略和共享较少策略，该策略包括共享模型参数的子集。结果表明，与DP相比，共享更少的策略具有更好的隐私效用权衡，尤其是在八卦环境下。



## **48. MalProtect: Stateful Defense Against Adversarial Query Attacks in ML-based Malware Detection**

MalProtect：基于ML的恶意软件检测中对抗恶意查询攻击的状态防御 cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2302.10739v2) [paper-pdf](http://arxiv.org/pdf/2302.10739v2)

**Authors**: Aqib Rashid, Jose Such

**Abstract**: ML models are known to be vulnerable to adversarial query attacks. In these attacks, queries are iteratively perturbed towards a particular class without any knowledge of the target model besides its output. The prevalence of remotely-hosted ML classification models and Machine-Learning-as-a-Service platforms means that query attacks pose a real threat to the security of these systems. To deal with this, stateful defenses have been proposed to detect query attacks and prevent the generation of adversarial examples by monitoring and analyzing the sequence of queries received by the system. Several stateful defenses have been proposed in recent years. However, these defenses rely solely on similarity or out-of-distribution detection methods that may be effective in other domains. In the malware detection domain, the methods to generate adversarial examples are inherently different, and therefore we find that such detection mechanisms are significantly less effective. Hence, in this paper, we present MalProtect, which is a stateful defense against query attacks in the malware detection domain. MalProtect uses several threat indicators to detect attacks. Our results show that it reduces the evasion rate of adversarial query attacks by 80+\% in Android and Windows malware, across a range of attacker scenarios. In the first evaluation of its kind, we show that MalProtect outperforms prior stateful defenses, especially under the peak adversarial threat.

摘要: 众所周知，ML模型容易受到敌意查询攻击。在这些攻击中，查询被迭代地扰动到特定的类，除了其输出之外，不知道目标模型。远程托管的ML分类模型和机器学习即服务平台的流行意味着查询攻击对这些系统的安全构成了真正的威胁。为了解决这个问题，已经提出了状态防御来检测查询攻击，并通过监控和分析系统接收到的查询序列来防止敌对实例的生成。近年来，有人提出了几项有状态的辩护。然而，这些防御完全依赖于可能在其他领域有效的相似性或分布外检测方法。在恶意软件检测领域，生成恶意示例的方法本质上是不同的，因此我们发现这种检测机制的有效性显著降低。因此，在本文中，我们提出了MalProtect，它是恶意软件检测领域中针对查询攻击的一种状态防御。MalProtect使用多个威胁指示器来检测攻击。我们的结果表明，在各种攻击场景下，该算法将Android和Windows恶意软件中恶意查询攻击的逃避率降低了80%+\%。在该类型的第一次评估中，我们表明MalProtect的性能优于先前的状态防御，特别是在峰值敌意威胁下。



## **49. Augment then Smooth: Reconciling Differential Privacy with Certified Robustness**

先增强后平滑：使差异隐私与认证的健壮性相协调 cs.LG

25 pages, 19 figures

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08656v1) [paper-pdf](http://arxiv.org/pdf/2306.08656v1)

**Authors**: Jiapeng Wu, Atiyeh Ashari Ghomi, David Glukhov, Jesse C. Cresswell, Franziska Boenisch, Nicolas Papernot

**Abstract**: Machine learning models are susceptible to a variety of attacks that can erode trust in their deployment. These threats include attacks against the privacy of training data and adversarial examples that jeopardize model accuracy. Differential privacy and randomized smoothing are effective defenses that provide certifiable guarantees for each of these threats, however, it is not well understood how implementing either defense impacts the other. In this work, we argue that it is possible to achieve both privacy guarantees and certified robustness simultaneously. We provide a framework called DP-CERT for integrating certified robustness through randomized smoothing into differentially private model training. For instance, compared to differentially private stochastic gradient descent on CIFAR10, DP-CERT leads to a 12-fold increase in certified accuracy and a 10-fold increase in the average certified radius at the expense of a drop in accuracy of 1.2%. Through in-depth per-sample metric analysis, we show that the certified radius correlates with the local Lipschitz constant and smoothness of the loss surface. This provides a new way to diagnose when private models will fail to be robust.

摘要: 机器学习模型容易受到各种攻击，这些攻击可能会侵蚀对其部署的信任。这些威胁包括对训练数据隐私的攻击，以及危及模型准确性的敌意例子。差异隐私和随机平滑是为这些威胁中的每一种提供可证明的保证的有效防御措施，然而，实施这两种防御措施对另一种威胁的影响还不是很清楚。在这项工作中，我们认为可以同时实现隐私保证和认证的健壮性。我们提供了一个称为DP-CERT的框架，用于将通过随机平滑验证的稳健性集成到不同的私有模型训练中。例如，与CIFAR10上的差分私有随机梯度下降相比，DP-CERT的认证精度提高了12倍，平均认证半径增加了10倍，但精度下降了1.2%。通过深入的逐样本度量分析，我们发现认证半径与损失曲面的局部Lipschitz常数和光滑度相关。这提供了一种新的方法来诊断何时私人车型将不再健壮。



## **50. A Unified Framework of Graph Information Bottleneck for Robustness and Membership Privacy**

面向健壮性和成员隐私的图信息瓶颈统一框架 cs.LG

**SubmitDate**: 2023-06-14    [abs](http://arxiv.org/abs/2306.08604v1) [paper-pdf](http://arxiv.org/pdf/2306.08604v1)

**Authors**: Enyan Dai, Limeng Cui, Zhengyang Wang, Xianfeng Tang, Yinghan Wang, Monica Cheng, Bing Yin, Suhang Wang

**Abstract**: Graph Neural Networks (GNNs) have achieved great success in modeling graph-structured data. However, recent works show that GNNs are vulnerable to adversarial attacks which can fool the GNN model to make desired predictions of the attacker. In addition, training data of GNNs can be leaked under membership inference attacks. This largely hinders the adoption of GNNs in high-stake domains such as e-commerce, finance and bioinformatics. Though investigations have been made in conducting robust predictions and protecting membership privacy, they generally fail to simultaneously consider the robustness and membership privacy. Therefore, in this work, we study a novel problem of developing robust and membership privacy-preserving GNNs. Our analysis shows that Information Bottleneck (IB) can help filter out noisy information and regularize the predictions on labeled samples, which can benefit robustness and membership privacy. However, structural noises and lack of labels in node classification challenge the deployment of IB on graph-structured data. To mitigate these issues, we propose a novel graph information bottleneck framework that can alleviate structural noises with neighbor bottleneck. Pseudo labels are also incorporated in the optimization to minimize the gap between the predictions on the labeled set and unlabeled set for membership privacy. Extensive experiments on real-world datasets demonstrate that our method can give robust predictions and simultaneously preserve membership privacy.

摘要: 图神经网络(GNN)在图结构数据建模方面取得了巨大的成功。然而，最近的研究表明，GNN很容易受到敌意攻击，这些攻击可以欺骗GNN模型做出所需的攻击者预测。此外，在成员关系推理攻击下，GNN的训练数据可能会被泄露。这在很大程度上阻碍了在电子商务、金融和生物信息学等高风险领域采用GNN。虽然已经在稳健预测和保护成员隐私方面进行了研究，但他们通常没有同时考虑稳健性和成员隐私。因此，在这项工作中，我们研究了一个新的问题，即开发健壮的、保护成员隐私的GNN。我们的分析表明，信息瓶颈(IB)可以帮助过滤噪声信息，并使对标记样本的预测正规化，这有利于稳健性和成员隐私。然而，结构噪声和节点分类中标签的缺乏对图结构数据上的IB的部署提出了挑战。为了缓解这些问题，我们提出了一种新的图信息瓶颈框架，该框架可以缓解带有邻居瓶颈的结构噪声。在优化过程中还加入了伪标签，以最大限度地减少对已标记集合和未标记集合的预测之间的差距，从而保证成员隐私。在真实数据集上的大量实验表明，我们的方法可以给出稳健的预测，同时保护成员隐私。



