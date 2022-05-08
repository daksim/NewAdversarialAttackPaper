# Latest Adversarial Attack Papers
**update at 2022-05-09 06:31:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can collaborative learning be private, robust and scalable?**

协作学习能否做到私密性、健壮性和可扩展性？ cs.LG

Submitted to TPDP 2022

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02652v1)

**Authors**: Dmitrii Usynin, Helena Klause, Daniel Rueckert, Georgios Kaissis

**Abstracts**: We investigate the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples in train- and inference-time attacks. We explore the applications of these techniques as well as their combinations to determine which method performs best, without a significant utility trade-off. Our investigation provides a practical overview of various methods that allow one to achieve a competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation.

摘要: 我们研究了将差分隐私、模型压缩和对抗性训练相结合来提高模型在训练和推理时间攻击中对对抗性样本的稳健性的有效性。我们探索这些技术的应用以及它们的组合，以确定哪种方法执行得最好，而不需要进行重大的实用权衡。我们的研究提供了各种方法的实用概述，这些方法允许人们在不严重性能下降的情况下实现竞争性模型性能、显著减小模型规模和改善经验对抗性稳健性。



## **2. Holistic Approach to Measure Sample-level Adversarial Vulnerability and its Utility in Building Trustworthy Systems**

样本级别敌方脆弱性的整体度量方法及其在构建可信系统中的应用 cs.CV

Accepted in CVPR Workshop 2022 on Human-centered Intelligent  Services: Safe and Trustworthy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02604v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Rohit Lal, Himanshu Patil, Anirban Chakraborty

**Abstracts**: Adversarial attack perturbs an image with an imperceptible noise, leading to incorrect model prediction. Recently, a few works showed inherent bias associated with such attack (robustness bias), where certain subgroups in a dataset (e.g. based on class, gender, etc.) are less robust than others. This bias not only persists even after adversarial training, but often results in severe performance discrepancies across these subgroups. Existing works characterize the subgroup's robustness bias by only checking individual sample's proximity to the decision boundary. In this work, we argue that this measure alone is not sufficient and validate our argument via extensive experimental analysis. It has been observed that adversarial attacks often corrupt the high-frequency components of the input image. We, therefore, propose a holistic approach for quantifying adversarial vulnerability of a sample by combining these different perspectives, i.e., degree of model's reliance on high-frequency features and the (conventional) sample-distance to the decision boundary. We demonstrate that by reliably estimating adversarial vulnerability at the sample level using the proposed holistic metric, it is possible to develop a trustworthy system where humans can be alerted about the incoming samples that are highly likely to be misclassified at test time. This is achieved with better precision when our holistic metric is used over individual measures. To further corroborate the utility of the proposed holistic approach, we perform knowledge distillation in a limited-sample setting. We observe that the student network trained with the subset of samples selected using our combined metric performs better than both the competing baselines, viz., where samples are selected randomly or based on their distances to the decision boundary.

摘要: 对抗性攻击使图像受到难以察觉的噪声干扰，从而导致错误的模型预测。最近，一些研究表明与这种攻击相关的固有偏见(健壮性偏差)，其中数据集中的某些子组(例如，基于类别、性别等)。都不如其他人那么健壮。这种偏见不仅在对抗性训练后仍然存在，而且经常导致这些小组之间的严重表现差异。已有的工作仅通过检查单个样本与决策边界的接近程度来表征子组的稳健性偏差。在这项工作中，我们认为单靠这一措施是不够的，并通过广泛的实验分析验证了我们的论点。已经观察到，对抗性攻击经常破坏输入图像的高频分量。因此，我们通过结合不同的角度，即模型对高频特征的依赖程度和(常规)样本到决策边界的距离，提出了一种量化样本对抗脆弱性的整体方法。我们证明，通过使用所提出的整体度量在样本级别可靠地估计对手漏洞，有可能开发一个可信的系统，在该系统中，可以向人类发出关于在测试时极有可能被错误分类的传入样本的警报。当我们的整体度量用于单个度量时，这是以更高的精度实现的。为了进一步证实所提出的整体方法的实用性，我们在有限样本环境下进行了知识提炼。我们观察到，用使用我们的组合度量选择的样本子集训练的学生网络比两个竞争基线(即，随机选择样本或基于样本到决策边界的距离)的性能都要好。



## **3. Resilience of Bayesian Layer-Wise Explanations under Adversarial Attacks**

贝叶斯层次型解释在对抗性攻击下的弹性 cs.LG

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2102.11010v3)

**Authors**: Ginevra Carbone, Guido Sanguinetti, Luca Bortolussi

**Abstracts**: We consider the problem of the stability of saliency-based explanations of Neural Network predictions under adversarial attacks in a classification task. Saliency interpretations of deterministic Neural Networks are remarkably brittle even when the attacks fail, i.e. for attacks that do not change the classification label. We empirically show that interpretations provided by Bayesian Neural Networks are considerably more stable under adversarial perturbations of the inputs and even under direct attacks to the explanations. By leveraging recent results, we also provide a theoretical explanation of this result in terms of the geometry of the data manifold. Additionally, we discuss the stability of the interpretations of high level representations of the inputs in the internal layers of a Network. Our results demonstrate that Bayesian methods, in addition to being more robust to adversarial attacks, have the potential to provide more stable and interpretable assessments of Neural Network predictions.

摘要: 在一个分类任务中，我们考虑了在对抗性攻击下神经网络预测的基于显著的解释的稳定性问题。即使攻击失败，即对于没有改变分类标签的攻击，确定性神经网络的显著解释也是非常脆弱的。我们的经验表明，贝叶斯神经网络提供的解释在输入的对抗性扰动下甚至在对解释的直接攻击下都更稳定。通过利用最近的结果，我们还从数据流形的几何角度对这一结果进行了理论解释。此外，我们还讨论了网络内部层中输入的高级表示的解释的稳定性。我们的结果表明，贝叶斯方法除了对对手攻击具有更强的鲁棒性外，还有可能为神经网络预测提供更稳定和更可解释的评估。



## **4. Subverting Fair Image Search with Generative Adversarial Perturbations**

利用生成性对抗性扰动颠覆公平图像搜索 cs.LG

Accepted as a full paper at the 2022 ACM Conference on Fairness,  Accountability, and Transparency (FAccT 22)

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02414v1)

**Authors**: Avijit Ghosh, Matthew Jagielski, Christo Wilson

**Abstracts**: In this work we explore the intersection fairness and robustness in the context of ranking: \textit{when a ranking model has been calibrated to achieve some definition of fairness, is it possible for an external adversary to make the ranking model behave unfairly without having access to the model or training data?} To investigate this question, we present a case study in which we develop and then attack a state-of-the-art, fairness-aware image search engine using images that have been maliciously modified using a Generative Adversarial Perturbation (GAP) model. These perturbations attempt to cause the fair re-ranking algorithm to unfairly boost the rank of images containing people from an adversary-selected subpopulation.   We present results from extensive experiments demonstrating that our attacks can successfully confer significant unfair advantage to people from the majority class relative to fairly-ranked baseline search results. We demonstrate that our attacks are robust across a number of variables, that they have close to zero impact on the relevance of search results, and that they succeed under a strict threat model. Our findings highlight the danger of deploying fair machine learning algorithms in-the-wild when (1) the data necessary to achieve fairness may be adversarially manipulated, and (2) the models themselves are not robust against attacks.

摘要: 在这项工作中，我们探讨了排序上下文中的交叉点公平性和稳健性：{当排序模型被校准以实现某种公平性定义时，外部对手有可能在没有访问该模型或训练数据的情况下使排序模型表现不公平吗？}为了研究这个问题，我们提供了一个案例研究，在该案例中，我们开发了一个最先进的、公平感知的图像搜索引擎，然后使用使用生成性对抗扰动(GAP)模型进行恶意修改的图像来攻击该搜索引擎。这些扰动试图导致公平重新排序算法不公平地提升包含来自对手选择的子群体的人的图像的排名。我们给出了大量实验的结果，表明我们的攻击可以成功地向大多数类别的人提供相对于排名公平的基线搜索结果的显著不公平优势。我们证明，我们的攻击在许多变量中都是健壮的，它们对搜索结果的相关性几乎没有影响，并且它们在严格的威胁模型下成功。我们的发现突显了在以下情况下部署公平机器学习算法的危险：(1)实现公平所需的数据可能被恶意操纵，以及(2)模型本身对攻击没有健壮性。



## **5. Robust Conversational Agents against Imperceptible Toxicity Triggers**

强大的对话代理，可抵御潜伏的毒性触发 cs.CL

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02392v1)

**Authors**: Ninareh Mehrabi, Ahmad Beirami, Fred Morstatter, Aram Galstyan

**Abstracts**: Warning: this paper contains content that maybe offensive or upsetting. Recent research in Natural Language Processing (NLP) has advanced the development of various toxicity detection models with the intention of identifying and mitigating toxic language from existing systems. Despite the abundance of research in this area, less attention has been given to adversarial attacks that force the system to generate toxic language and the defense against them. Existing work to generate such attacks is either based on human-generated attacks which is costly and not scalable or, in case of automatic attacks, the attack vector does not conform to human-like language, which can be detected using a language model loss. In this work, we propose attacks against conversational agents that are imperceptible, i.e., they fit the conversation in terms of coherency, relevancy, and fluency, while they are effective and scalable, i.e., they can automatically trigger the system into generating toxic language. We then propose a defense mechanism against such attacks which not only mitigates the attack but also attempts to maintain the conversational flow. Through automatic and human evaluations, we show that our defense is effective at avoiding toxic language generation even against imperceptible toxicity triggers while the generated language fits the conversation in terms of coherency and relevancy. Lastly, we establish the generalizability of such a defense mechanism on language generation models beyond conversational agents.

摘要: 警告：本文包含可能冒犯或令人反感的内容。自然语言处理(NLP)的最新研究推动了各种毒性检测模型的发展，目的是从现有系统中识别和缓解有毒语言。尽管在这一领域进行了大量的研究，但对迫使系统生成有毒语言的对抗性攻击以及对它们的防御的关注较少。生成此类攻击的现有工作要么基于代价高昂且不可扩展的人为生成的攻击，要么在自动攻击的情况下，攻击向量不符合可使用语言模型丢失来检测的类人类语言。在这项工作中，我们提出了针对不可察觉的会话代理的攻击，即它们在连贯性、关联性和流畅性方面符合会话，而它们是有效和可扩展的，即它们可以自动触发系统生成有毒语言。然后，我们提出了一种针对此类攻击的防御机制，该机制不仅可以缓解攻击，还可以尝试保持会话流。通过自动和人工评估，我们的防御措施有效地避免了有毒语言的生成，即使是针对潜在的有毒触发，而生成的语言在连贯性和关联性方面符合会话。最后，我们建立了这种防御机制在会话主体之外的语言生成模型上的泛化能力。



## **6. Zero Day Threat Detection Using Graph and Flow Based Security Telemetry**

基于图和流的安全遥测零日威胁检测 cs.CR

11 pages, 6 figures, submitting to NeurIPS 2022

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02298v1)

**Authors**: Christopher Redino, Dhruv Nandakumar, Robert Schiller, Kevin Choi, Abdul Rahman, Edward Bowen, Matthew Weeks, Aaron Shaha, Joe Nehila

**Abstracts**: Zero Day Threats (ZDT) are novel methods used by malicious actors to attack and exploit information technology (IT) networks or infrastructure. In the past few years, the number of these threats has been increasing at an alarming rate and have been costing organizations millions of dollars to remediate. The increasing expansion of network attack surfaces and the exponentially growing number of assets on these networks necessitate the need for a robust AI-based Zero Day Threat detection model that can quickly analyze petabyte-scale data for potentially malicious and novel activity. In this paper, the authors introduce a deep learning based approach to Zero Day Threat detection that can generalize, scale, and effectively identify threats in near real-time. The methodology utilizes network flow telemetry augmented with asset-level graph features, which are passed through a dual-autoencoder structure for anomaly and novelty detection respectively. The models have been trained and tested on four large scale datasets that are representative of real-world organizational networks and they produce strong results with high precision and recall values. The models provide a novel methodology to detect complex threats with low false-positive rates that allow security operators to avoid alert fatigue while drastically reducing their mean time to response with near-real-time detection. Furthermore, the authors also provide a novel, labelled, cyber attack dataset generated from adversarial activity that can be used for validation or training of other models. With this paper, the authors' overarching goal is to provide a novel architecture and training methodology for cyber anomaly detectors that can generalize to multiple IT networks with minimal to no retraining while still maintaining strong performance.

摘要: 零日威胁(ZDT)是恶意行为者用来攻击和利用信息技术(IT)网络或基础设施的新方法。在过去的几年里，这些威胁的数量一直在以惊人的速度增长，并花费了组织数百万美元来补救。网络攻击面日益扩大，这些网络上的资产数量呈指数级增长，这就需要一个强大的基于人工智能的零日威胁检测模型，该模型可以快速分析PB级数据，以发现潜在的恶意和新活动。在本文中，作者介绍了一种基于深度学习的零日威胁检测方法，该方法可以近乎实时地概括、扩展和有效地识别威胁。该方法利用网络流量遥测和资产级别图形特征，这些特征通过一个双自动编码器结构分别用于异常和新奇检测。这些模型已经在代表真实世界组织网络的四个大规模数据集上进行了训练和测试，它们产生了具有高精确度和召回值的强大结果。这些模型提供了一种新的方法来检测错误率低的复杂威胁，使安全操作员能够避免警报疲劳，同时通过近实时检测大幅缩短他们的平均响应时间。此外，作者还提供了一个新的、标记的、从敌对活动中生成的网络攻击数据集，可以用于验证或训练其他模型。在这篇论文中，作者的总体目标是为网络异常检测器提供一种新的体系结构和训练方法，该结构和训练方法可以推广到多个IT网络，而不需要重新训练，同时仍然保持强大的性能。



## **7. Adversarial Training for High-Stakes Reliability**

高风险可靠性的对抗性训练 cs.LG

31 pages, 6 figures, small tweak

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.01663v2)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.

摘要: 未来，强大的人工智能系统可能会部署在高风险的环境中，在那里，单一的故障可能是灾难性的。在高风险环境中提高人工智能安全性的一种技术是对抗性训练，它使用对手生成样本进行训练，以实现更好的最坏情况下的性能。在这项工作中，我们使用一个语言生成任务作为测试平台，通过对抗性训练来实现高可靠性。我们创建了一系列对抗性训练技术--包括一个帮助人类对手的工具--来发现并消除分类器中的故障，该分类器过滤生成器建议的文本完成。在我们简单的“避免受伤”任务中，我们确定可以设置非常保守的分类器阈值，而不会显著影响过滤输出的质量。在我们选择的阈值下，使用我们的基准分类器进行过滤可以将分发内数据的不安全完成率从大约2.4%降低到0.003%，这接近我们的测量能力极限。我们发现，对抗性训练显著提高了对我们训练的对抗性攻击的健壮性，而不影响分发内性能。我们希望在高风险的可靠性环境中看到进一步的工作，包括更强大的工具来增强人类对手，以及更好的方法来衡量高水平的可靠性，直到我们可以自信地排除强大模型部署时灾难性故障的可能性。



## **8. Rethinking Classifier And Adversarial Attack**

对量词与对抗性攻击的再思考 cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02743v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e. not approaching the lower bound of robustness). To solve this problem, this paper first uses the Decouple Space method to divide the classifier into two parts: non-linear and linear. On this basis, this paper defines the representation vector of original example (and its space, i.e., the representation space) and uses Absolute Classification Boundaries Initialization (ACBI) iterative optimization to obtain a better attack starting point (i.e. attacking from this point can approach the lower bound of robustness faster). Particularly, this paper apply ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.

摘要: 人们提出了各种防御模型来抵抗对抗攻击算法，但现有的对抗稳健性评估方法往往高估了这些模型的对抗稳健性(即没有接近稳健性的下界)。为了解决这一问题，本文首先利用解耦空间方法将分类器分为两部分：非线性部分和线性部分。在此基础上，定义了原始样本的表示向量(及其空间，即表示空间)，并通过绝对分类边界初始化(ACBI)迭代优化，获得了更好的攻击起点(即从这里攻击可以更快地逼近鲁棒性下界)。特别是，本文将ACBI应用于近50个广泛使用的防御模型(包括8个体系结构)。实验结果表明，ACBI在所有情况下都表现出较低的稳健性。



## **9. Based-CE white-box adversarial attack will not work using super-fitting**

基于CE的白盒对抗性攻击将不会使用超级拟合 cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02741v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep Neural Networks (DNN) are widely used in various fields due to their powerful performance, but recent studies have shown that deep learning models are vulnerable to adversarial attacks-by adding a slight perturbation to the input, the model will get wrong results. It is especially dangerous for some systems with high security requirements, so this paper proposes a new defense method by using the model super-fitting status. Model's adversarial robustness (i.e., the accuracry under adversarial attack) has been greatly improved in this status. This paper mathematically proves the effectiveness of super-fitting, and proposes a method to make the model reach this status quickly-minimaze unrelated categories scores (MUCS). Theoretically, super-fitting can resist any existing (even future) Based on CE white-box adversarial attack. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting and other nearly 50 defense models from recent conferences. The experimental results show that super-fitting method in this paper can make the trained model obtain the highest adversarial performance robustness.

摘要: 深度神经网络(DNN)以其强大的性能在各个领域得到了广泛的应用，但最近的研究表明，深度学习模型容易受到对手的攻击--只要在输入中加入一点扰动，模型就会得到错误的结果。对于一些安全性要求较高的系统尤其危险，因此本文提出了一种利用模型的超拟合态进行防御的新方法。在这种情况下，模型的对抗稳健性(即在对抗攻击下的准确性)得到了很大的提高。本文从数学上证明了超拟合的有效性，并提出了一种使模型快速达到这一状态的方法--最小化无关类别得分(MUC)。理论上，超拟合可以抵抗任何现有的(甚至是未来的)基于CE白盒的对抗性攻击。此外，本文使用多种强大的攻击算法对最近几次会议上的超拟合等近50种防御模型的对抗健壮性进行了评估。实验结果表明，本文提出的超拟合方法可以使训练后的模型获得最高的对抗性能稳健性。



## **10. Few-Shot Backdoor Attacks on Visual Object Tracking**

视觉目标跟踪中的几次后门攻击 cs.CV

This work is accepted by the ICLR 2022. The first two authors  contributed equally to this work. In this version, we fix some typos and  errors contained in the last one. 21 pages

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2201.13178v2)

**Authors**: Yiming Li, Haoxiang Zhong, Xingjun Ma, Yong Jiang, Shu-Tao Xia

**Abstracts**: Visual object tracking (VOT) has been widely adopted in mission-critical applications, such as autonomous driving and intelligent surveillance systems. In current practice, third-party resources such as datasets, backbone networks, and training platforms are frequently used to train high-performance VOT models. Whilst these resources bring certain convenience, they also introduce new security threats into VOT models. In this paper, we reveal such a threat where an adversary can easily implant hidden backdoors into VOT models by tempering with the training process. Specifically, we propose a simple yet effective few-shot backdoor attack (FSBA) that optimizes two losses alternately: 1) a \emph{feature loss} defined in the hidden feature space, and 2) the standard \emph{tracking loss}. We show that, once the backdoor is embedded into the target model by our FSBA, it can trick the model to lose track of specific objects even when the \emph{trigger} only appears in one or a few frames. We examine our attack in both digital and physical-world settings and show that it can significantly degrade the performance of state-of-the-art VOT trackers. We also show that our attack is resistant to potential defenses, highlighting the vulnerability of VOT models to potential backdoor attacks.

摘要: 视觉对象跟踪(VOT)已被广泛应用于任务关键型应用，如自动驾驶和智能监控系统。在目前的实践中，经常使用数据集、骨干网、培训平台等第三方资源来培训高性能的VOT模型。这些资源在带来一定便利的同时，也给VOT模型带来了新的安全威胁。在本文中，我们揭示了这样一种威胁，其中对手可以通过调整训练过程来轻松地在VOT模型中植入隐藏的后门。具体地说，我们提出了一种简单而有效的少射击后门攻击(FSBA)，它交替优化了两种损失：1)定义在隐藏特征空间中的a\emph{特征损失}，2)标准\emph{跟踪损失}。我们证明，一旦FSBA将后门嵌入到目标模型中，它就可以欺骗模型，使其失去对特定对象的跟踪，即使\emph{触发器}只出现在一个或几个帧中。我们在数字和物理环境中检查了我们的攻击，并表明它可以显著降低最先进的VoT跟踪器的性能。我们还表明，我们的攻击是抵抗潜在防御的，这突显了VOT模型对潜在后门攻击的脆弱性。



## **11. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

对基于投影的可取消生物识别方案的身份验证攻击(长版) cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2110.15163v3)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.

摘要: 可取消生物识别方案旨在通过将用户特定的令牌(例如密码、存储的秘密或盐)与生物识别数据相结合来生成安全的生物识别模板。这种类型的变换被构造为生物测定变换与特征提取算法的组合。可撤销生物特征识别方案的安全性要求涉及模板的不可逆性、不可链接性和可撤销性，而不损失比较的准确性。虽然最近有几个方案在这些要求方面受到了攻击，但据我们所知，这种组合物的完全可逆性以产生碰撞的生物测定特征，特别是呈现攻击，从未得到证明。在这篇文章中，我们借助整数线性规划(ILP)和二次约束二次规划(QCQP)对传统的可取消方案进行了形式化描述。解决这些优化问题允许对手稍微更改其指纹图像，以冒充任何个人。此外，在更严重的情况下，可以同时模拟几个人。



## **12. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

AdaptOver：蜂窝网络中的自适应遮蔽攻击 cs.CR

This version introduces uplink overshadowing

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2106.05039v2)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Röschlin, Srdjan Čapkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver -- a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.

摘要: 在蜂窝网络中，对移动设备和核心网络之间的通信链路的攻击会严重影响隐私和可用性。到目前为止，伪基站已经被要求执行这样的攻击。由于它们需要持续高的输出功率来吸引受害者，因此它们的射程有限，运营商和用户智能手机上的专用应用程序都很容易检测到它们。本文介绍了一种专为LTE和5G-NSA蜂窝网络设计的MITM攻击系统AdaptOver。AdaptOver允许对手在网络和移动设备之间的任一方向上通过空中解码、掩盖(替换)和注入任意消息。使用遮蔽，AdaptOver可以触发UE以纯文本形式传输其永久标识符(IMSI)，从而导致持续($\geq$12h)DoS或隐私泄露。这些攻击可以针对一个小区内的所有用户，也可以根据受害者的电话号码专门针对受害者。我们使用软件定义的无线电和低成本的放大设置来实现AdaptOver。我们使用各种智能手机演示了这些攻击对实时运行的LTE和5G-NSA网络的影响和实用性。我们的实验表明，AdaptOver可以对距离攻击者3.8公里以上的受害者发动攻击。考虑到其实用性和效率，AdaptOver表明，专注于伪基站的现有对策不再足够，标志着蜂窝网络安全机制设计的范式转变。



## **13. Can Rationalization Improve Robustness?**

合理化能提高健壮性吗？ cs.CL

Accepted to NAACL 2022; The code is available at  https://github.com/princeton-nlp/rationale-robustness

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2204.11790v2)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.

摘要: 越来越多的工作研究了神经NLP模型的发展，这种模型可以产生原理--输入的子集可以解释他们的模型预测。在本文中，我们询问这些基本模型除了具有可解释的性质外，是否还可以提供对对手攻击的稳健性。由于这些模型在做出预测(“预测者”)之前需要首先生成理由(“理性器”)，因此它们有可能忽略噪声或相反添加的文本，只需将其从生成的理由中掩盖出来。为此，我们系统地为标记和句子级合理化任务生成了各种类型的AddText攻击，并在五个不同的任务中对最先进的理性模型进行了广泛的经验评估。我们的实验表明，当理性器对位置偏差或攻击文本的词汇选择敏感时，基本模型显示出提高稳健性的前景，而它们在某些场景中却举步维艰。此外，利用人的理性作为监督并不总是能转化为更好的业绩。我们的研究是探索在合理化-然后预测框架中可解释性和稳健性之间的相互作用的第一步。



## **14. Don't sweat the small stuff, classify the rest: Sample Shielding to protect text classifiers against adversarial attacks**

不要为小事操心，对其余的事情进行分类：样本屏蔽保护文本分类器免受对手攻击 cs.CL

9 pages, 8 figures, Accepted to NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01714v1)

**Authors**: Jonathan Rusert, Padmini Srinivasan

**Abstracts**: Deep learning (DL) is being used extensively for text classification. However, researchers have demonstrated the vulnerability of such classifiers to adversarial attacks. Attackers modify the text in a way which misleads the classifier while keeping the original meaning close to intact. State-of-the-art (SOTA) attack algorithms follow the general principle of making minimal changes to the text so as to not jeopardize semantics. Taking advantage of this we propose a novel and intuitive defense strategy called Sample Shielding. It is attacker and classifier agnostic, does not require any reconfiguration of the classifier or external resources and is simple to implement. Essentially, we sample subsets of the input text, classify them and summarize these into a final decision. We shield three popular DL text classifiers with Sample Shielding, test their resilience against four SOTA attackers across three datasets in a realistic threat setting. Even when given the advantage of knowing about our shielding strategy the adversary's attack success rate is <=10% with only one exception and often < 5%. Additionally, Sample Shielding maintains near original accuracy when applied to original texts. Crucially, we show that the `make minimal changes' approach of SOTA attackers leads to critical vulnerabilities that can be defended against with an intuitive sampling strategy.

摘要: 深度学习正被广泛地用于文本分类。然而，研究人员已经证明了这种分类器在对抗攻击时的脆弱性。攻击者以一种误导量词的方式修改文本，同时几乎保持原始含义不变。最新的(SOTA)攻击算法遵循对文本进行最小程度的更改以不危及语义的一般原则。利用这一点，我们提出了一种新颖而直观的防御策略，称为样本屏蔽。它与攻击者和分类器无关，不需要重新配置分类器或外部资源，并且易于实现。基本上，我们对输入文本的子集进行采样，对它们进行分类，并将其总结为最终决策。我们用样本屏蔽了三个流行的DL文本分类器，在真实的威胁设置中测试了它们在三个数据集上对抗四个SOTA攻击者的弹性。即使在知道我们的屏蔽策略的优势下，对手的攻击成功率也是<=10%，只有一个例外，而且通常<5%。此外，当应用于原始文本时，样本屏蔽保持了接近原始的准确性。至关重要的是，我们展示了SOTA攻击者的“最小改动”方法导致了可以通过直观的抽样策略防御的关键漏洞。



## **15. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

受限特征空间中的对抗性攻防统一框架 cs.AI

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2112.01156v2)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.

摘要: 为了正确评估在受限特征空间中工作的模型，需要生成可行的对抗性示例。然而，对为计算机视觉设计的攻击实施约束仍然是一项具有挑战性的任务。我们提出了一个统一的框架来生成满足给定领域约束的可行对抗性实例。该框架既可以处理线性约束，也可以处理非线性约束。我们将我们的框架实例化为两种算法：一种是在损失函数中引入约束以最大化的基于梯度的攻击，另一种是以误分类、扰动最小化和约束满足为目标的多目标搜索算法。我们表明，我们的方法在四个不同的领域是有效的，成功率高达100%，在这些领域，最先进的攻击无法生成一个可行的例子。除了对抗性再训练，我们还建议引入工程非凸约束来提高模型对抗性的稳健性。我们证明了这种新的防御与对抗性的再训练一样有效。我们的框架构成了受限对抗攻击研究的起点，并为未来的研究提供了相关的基线和数据集。



## **16. On the uncertainty principle of neural networks**

论神经网络的不确定性原理 cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01493v1)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang

**Abstracts**: Despite the successes in many fields, it is found that neural networks are vulnerability and difficult to be both accurate and robust (robust means that the prediction of the trained network stays unchanged for inputs with non-random perturbations introduced by adversarial attacks). Various empirical and analytic studies have suggested that there is more or less a trade-off between the accuracy and robustness of neural networks. If the trade-off is inherent, applications based on the neural networks are vulnerable with untrustworthy predictions. It is then essential to ask whether the trade-off is an inherent property or not. Here, we show that the accuracy-robustness trade-off is an intrinsic property whose underlying mechanism is deeply related to the uncertainty principle in quantum mechanics. We find that for a neural network to be both accurate and robust, it needs to resolve the features of the two conjugated parts $x$ (the inputs) and $\Delta$ (the derivatives of the normalized loss function $J$ with respect to $x$), respectively. Analogous to the position-momentum conjugation in quantum mechanics, we show that the inputs and their conjugates cannot be resolved by a neural network simultaneously.

摘要: 尽管在许多领域取得了成功，但人们发现神经网络是脆弱的，很难既准确又稳健(稳健是指对于受到对抗性攻击引入的非随机扰动的输入，训练网络的预测保持不变)。各种经验和分析研究表明，神经网络的准确性和稳健性之间或多或少存在权衡。如果这种权衡是与生俱来的，那么基于神经网络的应用程序很容易受到不可信预测的影响。因此，至关重要的是要问一问，这种权衡是否是一种固有属性。在这里，我们证明了精度-稳健性权衡是一种内在的性质，其潜在的机制与量子力学中的测不准原理密切相关。我们发现，为了使神经网络既准确又稳健，它需要分别解析两个共轭部分$x$(输入)和$\Delta$(归一化损失函数$J$关于$x$的导数)的特征。类似于量子力学中的位置-动量共轭，我们证明了输入及其共轭不能由神经网络同时求解。



## **17. Self-Ensemble Adversarial Training for Improved Robustness**

提高健壮性的自我集成对抗性训练 cs.LG

18 pages, 3 figures, ICLR 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2203.09678v2)

**Authors**: Hongjun Wang, Yisen Wang

**Abstracts**: Due to numerous breakthroughs in real-world applications brought by machine intelligence, deep neural networks (DNNs) are widely employed in critical applications. However, predictions of DNNs are easily manipulated with imperceptible adversarial perturbations, which impedes the further deployment of DNNs and may result in profound security and privacy implications. By incorporating adversarial samples into the training data pool, adversarial training is the strongest principled strategy against various adversarial attacks among all sorts of defense methods. Recent works mainly focus on developing new loss functions or regularizers, attempting to find the unique optimal point in the weight space. But none of them taps the potentials of classifiers obtained from standard adversarial training, especially states on the searching trajectory of training. In this work, we are dedicated to the weight states of models through the training process and devise a simple but powerful \emph{Self-Ensemble Adversarial Training} (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise. We also discuss the relationship between the ensemble of predictions from different adversarially trained models and the prediction of weight-ensembled models, as well as provide theoretical and empirical evidence that the proposed self-ensemble method provides a smoother loss landscape and better robustness than both individual models and the ensemble of predictions from different classifiers. We further analyze a subtle but fatal issue in the general settings for the self-ensemble model, which causes the deterioration of the weight-ensembled method in the late phases.

摘要: 由于机器智能在实际应用中取得了许多突破，深度神经网络(DNN)被广泛应用于关键应用中。然而，DNN的预测很容易受到潜移默化的敌意干扰，这阻碍了DNN的进一步部署，并可能导致深刻的安全和隐私影响。通过将对抗性样本纳入训练数据库，对抗性训练是各种防御方法中对抗各种对抗性攻击的最强原则性策略。最近的工作主要集中在开发新的损失函数或正则化函数，试图在权空间中找到唯一的最优点。但它们都没有挖掘从标准对抗性训练中获得的分类器的潜力，特别是在训练的搜索轨迹上。在这项工作中，我们致力于通过训练过程来研究模型的权重状态，并设计了一种简单但强大的自集成对抗性训练(SEAT)方法，通过平均历史模型的权重来产生稳健的分类器。这在很大程度上提高了目标模型对几种众所周知的敌意攻击的稳健性，甚至仅仅利用天真的交叉熵损失来监督。我们还讨论了不同对手训练模型的预测集成与权重集成模型预测之间的关系，并提供了理论和经验证据，表明所提出的自集成方法提供了比单个模型和来自不同分类器的预测集成更平滑的损失情况和更好的稳健性。我们进一步分析了自集成模型一般设置中的一个微妙但致命的问题，该问题导致了权重集成方法在后期的恶化。



## **18. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

SemAttack：基于不同语义空间的自然文本攻击 cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01287v1)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.

摘要: 最近的研究表明，预先训练的语言模型(LMS)容易受到文本攻击。然而，现有的攻击方法要么攻击成功率低，要么不能在指数级的大扰动空间中进行有效的搜索。通过构造不同的语义扰动函数，提出了一种高效的自然对抗性文本生成框架SemAttack。具体地，SemAttack优化约束在通用语义空间上的所生成的扰动，所述通用语义空间包括打字错误空间、知识空间(例如，WordNet)、上下文化的语义空间(例如，BERT聚类的嵌入空间)或这些空间的组合。因此，生成的对抗性文本在语义上更接近原始输入。大量实验表明，最先进的大规模LMS(如DeBERTa-v2)和防御策略(如FreeLB)仍然容易受到SemAttack的攻击。我们进一步证明了SemAttack是通用的，能够生成不同语言(如英语和汉语)的自然对抗性文本，具有很高的攻击成功率。人类评估还证实，我们生成的对抗性文本是自然的，几乎不会影响人类的表现。我们的代码在https://github.com/AI-secure/SemAttack.上公开提供



## **19. MIRST-DM: Multi-Instance RST with Drop-Max Layer for Robust Classification of Breast Cancer**

MIRST-DM：用于乳腺癌稳健分类的Drop-Max层多实例RST eess.IV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.01674v1)

**Authors**: Shoukun Sun, Min Xian, Aleksandar Vakanski, Hossny Ghanem

**Abstracts**: Robust self-training (RST) can augment the adversarial robustness of image classification models without significantly sacrificing models' generalizability. However, RST and other state-of-the-art defense approaches failed to preserve the generalizability and reproduce their good adversarial robustness on small medical image sets. In this work, we propose the Multi-instance RST with a drop-max layer, namely MIRST-DM, which involves a sequence of iteratively generated adversarial instances during training to learn smoother decision boundaries on small datasets. The proposed drop-max layer eliminates unstable features and helps learn representations that are robust to image perturbations. The proposed approach was validated using a small breast ultrasound dataset with 1,190 images. The results demonstrate that the proposed approach achieves state-of-the-art adversarial robustness against three prevalent attacks.

摘要: 稳健自训练(RST)可以在不显著牺牲模型泛化能力的情况下，增强图像分类模型的对抗性。然而，RST和其他最先进的防御方法未能保持其泛化能力，并在小的医学图像集上重现其良好的对抗性鲁棒性。在这项工作中，我们提出了具有最大丢弃层的多实例RST，即MIRST-DM，它在训练过程中包含一系列迭代生成的对抗性实例，以在小数据集上学习更平滑的决策边界。提出的Drop-max层消除了不稳定的特征，并帮助学习对图像扰动具有鲁棒性的表示。使用1,190幅图像的小型乳腺超声数据集对所提出的方法进行了验证。实验结果表明，该方法对三种流行的攻击具有最好的抗攻击能力。



## **20. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

分段和完全：利用稳健的补丁检测保护对象检测器免受敌意补丁攻击 cs.CV

CVPR 2022 camera ready

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2112.04532v2)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.

摘要: 目标检测在许多安全关键系统中起着关键作用。对抗性补丁攻击很容易在物理世界中实现，对最先进的对象检测器构成了严重威胁。为目标探测器开发可靠的防御补丁攻击是至关重要的，但研究严重不足。在本文中，我们提出了分段和完全防御(SAC)，这是一个通用的框架，通过检测和删除敌意补丁来防御对象检测器的补丁攻击。我们首先训练一个补丁分割器，该分割器输出提供对抗性补丁像素级定位的补丁掩码。然后，我们提出了一种自对抗训练算法来增强补丁分割器的鲁棒性。此外，我们还设计了一种稳健的形状补全算法，如果斑块分割器的输出与地面真实斑块掩模的汉明距离在一定范围内，该算法就能保证从图像中去除整个斑块。我们在CoCo和xView数据集上的实验表明，SAC在不降低对干净图像的性能的情况下，即使在强自适应攻击下也具有优异的鲁棒性，并且对看不见的补丁形状、攻击预算和看不见的攻击方法具有很好的泛化能力。此外，我们还给出了APRICOT-MASK数据集，它用对抗性斑块的像素级标注来扩充APRICOT数据集。结果表明，SAC能够显著降低物理补丁攻击的定向攻击成功率。我们的代码可以在https://github.com/joellliu/SegmentAndComplete.上找到



## **21. Defending Against Advanced Persistent Threats using Game-Theory**

利用博弈论防御高级持续性威胁 cs.CR

preprint of a correction to the article with the same name, published  with PLOS ONE, and currently under review

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00956v1)

**Authors**: Stefan Rass, Sandra König, Stefan Schauer

**Abstracts**: Advanced persistent threats (APT) combine a variety of different attack forms ranging from social engineering to technical exploits. The diversity and usual stealthiness of APT turns them into a central problem of contemporary practical system security, since information on attacks, the current system status or the attacker's incentives is often vague, uncertain and in many cases even unavailable. Game theory is a natural approach to model the conflict between the attacker and the defender, and this work investigates a generalized class of matrix games as a risk mitigation tool for an APT defense. Unlike standard game and decision theory, our model is tailored to capture and handle the full uncertainty that is immanent to APT, such as disagreement among qualitative expert risk assessments, unknown adversarial incentives and uncertainty about the current system state (in terms of how deeply the attacker may have penetrated into the system's protective shells already). Practically, game-theoretic APT models can be derived straightforwardly from topological vulnerability analysis, together with risk assessments as they are done in common risk management standards like the ISO 31000 family. Theoretically, these models come with different properties than classical game theoretic models, whose technical solution presented in this work may be of independent interest.

摘要: 高级持续威胁(APT)结合了各种不同的攻击形式，从社会工程到技术利用。APT的多样性和通常的隐蔽性使其成为当代实用系统安全的核心问题，因为关于攻击、当前系统状态或攻击者的动机的信息通常是模糊的、不确定的，在许多情况下甚至是不可用的。博弈论是对攻击者和防御者之间的冲突进行建模的一种自然方法，本工作研究了一类广义的矩阵博弈，作为APT防御的风险缓解工具。与标准的博弈和决策理论不同，我们的模型是为捕捉和处理APT固有的全部不确定性而量身定做的，例如定性专家风险评估之间的分歧、未知的对抗性激励以及关于当前系统状态的不确定性(就攻击者可能已经渗透到系统保护壳的深度而言)。实际上，博弈论的APT模型可以直接从拓扑脆弱性分析和风险评估中推导出来，就像它们在国际标准化组织31000系列等常见风险管理标准中所做的那样。理论上，这些模型具有不同于经典博弈论模型的性质，其在本工作中提出的技术解决方案可能具有独立的兴趣。



## **22. BERTops: Studying BERT Representations under a Topological Lens**

BERTOPS：研究拓扑透镜下的BERT表示 cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00953v1)

**Authors**: Jatin Chauhan, Manohar Kaul

**Abstracts**: Proposing scoring functions to effectively understand, analyze and learn various properties of high dimensional hidden representations of large-scale transformer models like BERT can be a challenging task. In this work, we explore a new direction by studying the topological features of BERT hidden representations using persistent homology (PH). We propose a novel scoring function named "persistence scoring function (PSF)" which: (i) accurately captures the homology of the high-dimensional hidden representations and correlates well with the test set accuracy of a wide range of datasets and outperforms existing scoring metrics, (ii) captures interesting post fine-tuning "per-class" level properties from both qualitative and quantitative viewpoints, (iii) is more stable to perturbations as compared to the baseline functions, which makes it a very robust proxy, and (iv) finally, also serves as a predictor of the attack success rates for a wide category of black-box and white-box adversarial attack methods. Our extensive correlation experiments demonstrate the practical utility of PSF on various NLP tasks relevant to BERT.

摘要: 提出评分函数来有效地理解、分析和学习大型变压器模型的高维隐藏表示的各种性质可能是一项具有挑战性的任务。在这项工作中，我们探索了一个新的方向，通过研究BERT隐藏表示的拓扑特征，使用持久同调(PH)。我们提出了一种新的评分函数“持久性评分函数(PSF)”，它(I)准确地捕捉高维隐藏表示的同源性，并与大范围数据集的测试集精度很好地关联，并优于现有的评分度量；(Ii)从定性和定量的角度捕捉有趣的微调后“每类”级别的属性；(Iii)与基线函数相比，对扰动更稳定，这使得它成为一个非常健壮的代理；(Iv)还可以预测大范围的黑盒和白盒对抗攻击方法的攻击成功率。我们广泛的相关实验证明了PSF在与BERT相关的各种NLP任务中的实用价值。



## **23. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

未知聚类个数在线聚类的重访高斯神经元 cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00920v1)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. A novel local learning rule is presented that performs online clustering with a maximum limit of the number of cluster to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.

摘要: 尽管人工神经网络最近取得了成功，但可能需要更多生物学上可信的学习方法来解决反向传播训练模型的弱点，如灾难性遗忘和对抗性攻击。提出了一种新的局部学习规则，该规则在线聚类的最大限制是要找到的簇的数目，而不是固定的簇数目。不使用正交权重或输出激活约束，而是通过侧向高斯神经元的相互排斥来获得激活稀疏性，以确保多个神经元中心不会占据输入域中的相同位置。在数据样本可以用均值和方差表示的情况下，提出了一种调整高斯神经元宽度的更新方法。这些算法被应用于MNIST和CIFAR-10数据集，以创建捕捉不同大小的像素斑块的输入模式的过滤器。实验结果表明，在大量的训练样本中，学习的参数是稳定的。



## **24. Deep-Attack over the Deep Reinforcement Learning**

深度强化学习中的深度攻击 cs.LG

Accepted to Knowledge-Based Systems

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00807v1)

**Authors**: Yang Li, Quan Pan, Erik Cambria

**Abstracts**: Recent adversarial attack developments have made reinforcement learning more vulnerable, and different approaches exist to deploy attacks against it, where the key is how to choose the right timing of the attack. Some work tries to design an attack evaluation function to select critical points that will be attacked if the value is greater than a certain threshold. This approach makes it difficult to find the right place to deploy an attack without considering the long-term impact. In addition, there is a lack of appropriate indicators of assessment during attacks. To make the attacks more intelligent as well as to remedy the existing problems, we propose the reinforcement learning-based attacking framework by considering the effectiveness and stealthy spontaneously, while we also propose a new metric to evaluate the performance of the attack model in these two aspects. Experimental results show the effectiveness of our proposed model and the goodness of our proposed evaluation metric. Furthermore, we validate the transferability of the model, and also its robustness under the adversarial training.

摘要: 近年来对抗性攻击的发展使得强化学习变得更加脆弱，存在不同的方法来部署针对它的攻击，其中关键是如何选择正确的攻击时机。一些工作试图设计一个攻击评估函数来选择临界点，如果该值大于一定的阈值就会受到攻击。这种方法使得在不考虑长期影响的情况下很难找到部署攻击的正确位置。此外，在袭击期间缺乏适当的评估指标。为了使攻击更加智能化，同时也为了弥补存在的问题，我们提出了基于强化学习的攻击框架，该框架自发地考虑了攻击的有效性和隐蔽性，同时还提出了一种新的度量来评估攻击模型在这两个方面的性能。实验结果表明了所提模型的有效性和所提评价指标的有效性。此外，我们还验证了该模型的可转移性，以及在对抗性训练下的鲁棒性。



## **25. Enhancing Adversarial Training with Feature Separability**

利用特征可分性增强对抗性训练 cs.CV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00637v1)

**Authors**: Yaxin Li, Xiaorui Liu, Han Xu, Wentao Wang, Jiliang Tang

**Abstracts**: Deep Neural Network (DNN) are vulnerable to adversarial attacks. As a countermeasure, adversarial training aims to achieve robustness based on the min-max optimization problem and it has shown to be one of the most effective defense strategies. However, in this work, we found that compared with natural training, adversarial training fails to learn better feature representations for either clean or adversarial samples, which can be one reason why adversarial training tends to have severe overfitting issues and less satisfied generalize performance. Specifically, we observe two major shortcomings of the features learned by existing adversarial training methods:(1) low intra-class feature similarity; and (2) conservative inter-classes feature variance. To overcome these shortcomings, we introduce a new concept of adversarial training graph (ATG) with which the proposed adversarial training with feature separability (ATFS) enables to coherently boost the intra-class feature similarity and increase inter-class feature variance. Through comprehensive experiments, we demonstrate that the proposed ATFS framework significantly improves both clean and robust performance.

摘要: 深度神经网络(DNN)容易受到敌意攻击。作为一种对策，对抗性训练旨在实现基于最小-最大优化问题的稳健性，它已被证明是最有效的防御策略之一。然而，在这项工作中，我们发现与自然训练相比，对抗性训练无论是对于干净的样本还是对抗性样本都无法学习到更好的特征表示，这可能是对抗性训练往往存在严重的过拟合问题和较不满意的泛化性能的原因之一。具体地说，我们观察到现有对抗性训练方法学习的特征有两个主要缺陷：(1)类内特征相似度低；(2)类间特征方差保守。为了克服这些不足，我们引入了对抗性训练图(ATG)的概念，在此基础上提出了基于特征可分性的对抗性训练(ATFS)，从而能够一致地提高类内特征相似度和增加类间特征方差。通过全面的实验，我们证明了所提出的ATFS框架在清洁和健壮性方面都有显著的提高。



## **26. Robust Fine-tuning via Perturbation and Interpolation from In-batch Instances**

从批内实例通过扰动和内插实现稳健的精调 cs.CL

IJCAI-ECAI 2022 (the 31st International Joint Conference on  Artificial Intelligence and the 25th European Conference on Artificial  Intelligence)

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00633v1)

**Authors**: Shoujie Tong, Qingxiu Dong, Damai Dai, Yifan song, Tianyu Liu, Baobao Chang, Zhifang Sui

**Abstracts**: Fine-tuning pretrained language models (PLMs) on downstream tasks has become common practice in natural language processing. However, most of the PLMs are vulnerable, e.g., they are brittle under adversarial attacks or imbalanced data, which hinders the application of the PLMs on some downstream tasks, especially in safe-critical scenarios. In this paper, we propose a simple yet effective fine-tuning method called Match-Tuning to force the PLMs to be more robust. For each instance in a batch, we involve other instances in the same batch to interact with it. To be specific, regarding the instances with other labels as a perturbation, Match-Tuning makes the model more robust to noise at the beginning of training. While nearing the end, Match-Tuning focuses more on performing an interpolation among the instances with the same label for better generalization. Extensive experiments on various tasks in GLUE benchmark show that Match-Tuning consistently outperforms the vanilla fine-tuning by $1.64$ scores. Moreover, Match-Tuning exhibits remarkable robustness to adversarial attacks and data imbalance.

摘要: 对下游任务的预训练语言模型(PLM)进行微调已成为自然语言处理中的常见做法。然而，大多数PLM都是脆弱的，例如，它们在敌意攻击或数据不平衡的情况下很脆弱，这阻碍了PLM在一些下游任务上的应用，特别是在安全关键的场景中。在本文中，我们提出了一种简单而有效的微调方法，称为匹配调谐，以迫使PLM变得更健壮。对于批次中的每个实例，我们都会让同一批次中的其他实例与其交互。具体地说，将具有其他标签的实例视为扰动，匹配调整使模型在训练开始时对噪声具有更强的鲁棒性。在接近尾声时，匹配调优更侧重于在具有相同标签的实例之间执行内插，以实现更好的泛化。在GLUE基准测试中的各种任务上的广泛实验表明，匹配调整的性能始终比普通微调高出1.64美元分数。此外，匹配调整对敌意攻击和数据失衡表现出显著的健壮性。



## **27. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

一句话抵得上一千美元：敌意攻击推特傻瓜股预测 cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-01    [paper-pdf](http://arxiv.org/pdf/2205.01094v1)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.

摘要: 越来越多的投资者和机器学习模型依赖社交媒体(如Twitter和Reddit)来收集实时信息和情绪，以预测股价走势。尽管众所周知，基于文本的模型容易受到对手攻击，但股票预测模型是否也有类似的脆弱性，还没有得到充分的探讨。在本文中，我们实验了各种对抗性攻击配置，以愚弄三个股票预测受害者模型。我们通过求解具有语义和预算约束的组合优化问题来解决对抗性生成问题。我们的结果表明，该攻击方法可以获得一致的成功率，并在交易模拟中通过简单地连接一条扰动但语义相似的推文来造成巨大的金钱损失。



## **28. Analysis of a blockchain protocol based on LDPC codes**

一种基于LDPC码的区块链协议分析 cs.CR

**SubmitDate**: 2022-04-30    [paper-pdf](http://arxiv.org/pdf/2202.07265v3)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check codes to counter DAAs. In this paper, we show that the protocol is less secure than claimed, owing to a redefinition of the adversarial success probability. As a consequence we show that, for some realistic choices of the parameters, the total amount of data downloaded by light nodes is larger than that obtainable with competitor solutions.

摘要: 在区块链数据可用性攻击(DAA)中，恶意节点发布块标头，但保留包含无效事务的部分块。可以下载并存储完整区块链的诚实全节点，知道有些数据不可用，但没有正式的方法向轻节点证明，即资源有限、无法访问整个区块链数据的节点。对抗这些攻击的常见解决方案使用线性纠错码来编码块内容。最近的一种称为SPAR的协议使用编码Merkle树和低密度奇偶校验码来对抗DAA。在这篇文章中，我们证明了该协议并不像所声称的那样安全，这是因为重新定义了对抗性成功概率。因此，我们表明，对于一些现实的参数选择，光节点下载的总数据量比竞争对手的解决方案所能获得的数据量要大。



## **29. Optimizing One-pixel Black-box Adversarial Attacks**

优化单像素黑盒对抗性攻击 cs.CR

9 pasges, 4 figures

**SubmitDate**: 2022-04-30    [paper-pdf](http://arxiv.org/pdf/2205.02116v1)

**Authors**: Tianxun Zhou, Shubhankar Agrawal, Prateek Manocha

**Abstracts**: The output of Deep Neural Networks (DNN) can be altered by a small perturbation of the input in a black box setting by making multiple calls to the DNN. However, the high computation and time required makes the existing approaches unusable. This work seeks to improve the One-pixel (few-pixel) black-box adversarial attacks to reduce the number of calls to the network under attack. The One-pixel attack uses a non-gradient optimization algorithm to find pixel-level perturbations under the constraint of a fixed number of pixels, which causes the network to predict the wrong label for a given image. We show through experimental results how the choice of the optimization algorithm and initial positions to search can reduce function calls and increase attack success significantly, making the attack more practical in real-world settings.

摘要: 深度神经网络(DNN)的输出可以通过对DNN进行多次调用来改变黑盒设置中的输入的微小扰动。然而，所需的高计算量和高时间使得现有的方法无法使用。这项工作旨在改进单像素(几个像素)的黑盒对抗性攻击，以减少受到攻击的网络呼叫数量。单像素攻击使用非梯度优化算法在固定像素数的约束下发现像素级扰动，从而导致网络预测给定图像的错误标签。实验结果表明，优化算法和初始搜索位置的选择可以显著减少函数调用，提高攻击成功率，使攻击在现实世界中更具实用性。



## **30. Logically Consistent Adversarial Attacks for Soft Theorem Provers**

软定理证明者的逻辑相容敌意攻击 cs.LG

IJCAI-ECAI 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.00047v1)

**Authors**: Alexander Gaskell, Yishu Miao, Lucia Specia, Francesca Toni

**Abstracts**: Recent efforts within the AI community have yielded impressive results towards "soft theorem proving" over natural language sentences using language models. We propose a novel, generative adversarial framework for probing and improving these models' reasoning capabilities. Adversarial attacks in this domain suffer from the logical inconsistency problem, whereby perturbations to the input may alter the label. Our Logically consistent AdVersarial Attacker, LAVA, addresses this by combining a structured generative process with a symbolic solver, guaranteeing logical consistency. Our framework successfully generates adversarial attacks and identifies global weaknesses common across multiple target models. Our analyses reveal naive heuristics and vulnerabilities in these models' reasoning capabilities, exposing an incomplete grasp of logical deduction under logic programs. Finally, in addition to effective probing of these models, we show that training on the generated samples improves the target model's performance.

摘要: 最近人工智能社区内的努力取得了令人印象深刻的结果，即使用语言模型对自然语言句子进行“软定理证明”。我们提出了一个新颖的、生成性的对抗性框架来探索和改进这些模型的推理能力。该领域中的对抗性攻击存在逻辑不一致问题，因此对输入的扰动可能会改变标签。我们的逻辑一致的敌意攻击者LAVA通过将结构化的生成过程与符号求解器相结合来解决这个问题，从而保证了逻辑一致性。我们的框架成功地生成了对抗性攻击，并识别了跨多个目标模型的全球共同弱点。我们的分析揭示了这些模型推理能力的天真启发式和漏洞，暴露了对逻辑程序下逻辑演绎的不完全掌握。最后，除了对这些模型进行有效的探测外，我们还表明，对生成的样本进行训练可以提高目标模型的性能。



## **31. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

信任还是不信任成员关系推断攻击的预测分数 cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2111.09076v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.

摘要: 成员关系推理攻击(MIA)的目的是确定特定样本是否被用来训练预测模型。知道这一点确实可能会导致隐私被侵犯。然而，大多数MIA都利用模型的预测分数--每个输出给定一些输入的概率--遵循这样一种直觉，即训练后的模型在其训练数据上往往表现不同。我们认为，对于许多现代深度网络体系结构来说，这是一种谬误。因此，MIA将悲惨地失败，因为过度自信不仅会导致已知域上的高假阳性率，而且还会导致分布外数据的高假阳性率，并隐含地充当对MIA的防御。具体地说，使用生成性对抗性网络，我们能够产生潜在无限数量的样本，这些样本被错误地归类为训练数据的一部分。换句话说，MIA的威胁被高估了，泄露的信息比之前假设的要少。此外，在模型的过度自信和他们对MIA的敏感性之间实际上存在着权衡：分类器知道的越多，他们不知道的时候，做出低置信度预测的人就越多，他们透露的训练数据就越多。



## **32. Adversarial attacks on an optical neural network**

对光学神经网络的敌意攻击 cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.01226v1)

**Authors**: Shuming Jiao, Ziwei Song, Shuiying Xiang

**Abstracts**: Adversarial attacks have been extensively investigated for machine learning systems including deep learning in the digital domain. However, the adversarial attacks on optical neural networks (ONN) have been seldom considered previously. In this work, we first construct an accurate image classifier with an ONN using a mesh of interconnected Mach-Zehnder interferometers (MZI). Then a corresponding adversarial attack scheme is proposed for the first time. The attacked images are visually very similar to the original ones but the ONN system becomes malfunctioned and generates wrong classification results in most time. The results indicate that adversarial attack is also a significant issue for optical machine learning systems.

摘要: 对抗性攻击已经被广泛地研究用于机器学习系统，包括数字领域的深度学习。然而，以前很少考虑对抗性攻击光学神经网络(ONN)。在这项工作中，我们首先使用互连的Mach-Zehnder干涉仪(MZI)构建了一个具有ONN的精确图像分类器。然后，首次提出了相应的对抗性攻击方案。被攻击的图像在视觉上与原始图像非常相似，但ONN系统在大多数情况下会出现故障并产生错误的分类结果。结果表明，对抗性攻击也是光学机器学习系统的一个重要问题。



## **33. Finding MNEMON: Reviving Memories of Node Embeddings**

寻找Mnemon：唤醒节点嵌入的记忆 cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.06963v2)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.

摘要: 以前围绕图的安全研究一直专注于图的(去)匿名化或理解图神经网络的安全和隐私问题。很少有人注意到将图嵌入模型(例如，节点嵌入)的输出与复杂的下游机器学习管道集成的隐私风险。在本文中，我们填补了这一空白，并提出了一种新的模型不可知图恢复攻击，该攻击利用了图节点嵌入中保留的隐含的图结构信息。我们证明了敌手只需访问原始图的节点嵌入矩阵，而不需要与节点嵌入模型交互，就能以相当高的精度恢复边。我们通过大量的实验证明了我们的图恢复攻击的有效性和适用性。



## **34. Exploration and Exploitation in Federated Learning to Exclude Clients with Poisoned Data**

联合学习中排除有毒数据客户端的探索与利用 cs.DC

Accepted at 2022 IWCMC

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14020v1)

**Authors**: Shadha Tabatabai, Ihab Mohammed, Basheer Qolomany, Abdullatif Albasser, Kashif Ahmad, Mohamed Abdallah, Ala Al-Fuqaha

**Abstracts**: Federated Learning (FL) is one of the hot research topics, and it utilizes Machine Learning (ML) in a distributed manner without directly accessing private data on clients. However, FL faces many challenges, including the difficulty to obtain high accuracy, high communication cost between clients and the server, and security attacks related to adversarial ML. To tackle these three challenges, we propose an FL algorithm inspired by evolutionary techniques. The proposed algorithm groups clients randomly in many clusters, each with a model selected randomly to explore the performance of different models. The clusters are then trained in a repetitive process where the worst performing cluster is removed in each iteration until one cluster remains. In each iteration, some clients are expelled from clusters either due to using poisoned data or low performance. The surviving clients are exploited in the next iteration. The remaining cluster with surviving clients is then used for training the best FL model (i.e., remaining FL model). Communication cost is reduced since fewer clients are used in the final training of the FL model. To evaluate the performance of the proposed algorithm, we conduct a number of experiments using FEMNIST dataset and compare the result against the random FL algorithm. The experimental results show that the proposed algorithm outperforms the baseline algorithm in terms of accuracy, communication cost, and security.

摘要: 联合学习(FL)是当前研究的热点之一，它以分布式的方式利用机器学习(ML)，不需要直接访问客户端的私有数据。然而，FL面临着许多挑战，包括难以获得高准确率、客户端与服务器之间的通信成本较高以及与敌意ML相关的安全攻击。为了应对这三个挑战，我们提出了一种受进化技术启发的FL算法。该算法将客户端随机分组到多个簇中，每个簇随机选择一个模型来考察不同模型的性能。然后，在重复过程中训练集群，其中在每次迭代中移除表现最差的集群，直到保留一个集群。在每次迭代中，一些客户端会因使用有毒数据或性能低下而被逐出群集。幸存的客户端将在下一次迭代中被利用。然后，具有幸存客户端的剩余簇用于训练最佳FL模型(即，剩余FL模型)。由于在FL模型的最终训练中使用更少的客户，因此降低了通信成本。为了评估算法的性能，我们使用FEMNIST数据集进行了大量的实验，并将结果与随机FL算法进行了比较。实验结果表明，该算法在准确率、通信开销和安全性方面均优于基线算法。



## **35. Backdoor Attacks in Federated Learning by Rare Embeddings and Gradient Ensembling**

基于稀有嵌入和梯度集成的联合学习后门攻击 cs.LG

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14017v1)

**Authors**: KiYoon Yoo, Nojun Kwak

**Abstracts**: Recent advances in federated learning have demonstrated its promising capability to learn on decentralized datasets. However, a considerable amount of work has raised concerns due to the potential risks of adversaries participating in the framework to poison the global model for an adversarial purpose. This paper investigates the feasibility of model poisoning for backdoor attacks through \textit{rare word embeddings of NLP models} in text classification and sequence-to-sequence tasks. In text classification, less than 1\% of adversary clients suffices to manipulate the model output without any drop in the performance of clean sentences. For a less complex dataset, a mere 0.1\% of adversary clients is enough to poison the global model effectively. We also propose a technique specialized in the federated learning scheme called gradient ensemble, which enhances the backdoor performance in all experimental settings.

摘要: 联邦学习的最新进展已经证明了它在分散的数据集上学习的前景。然而，由于参与该框架的对手出于对抗目的而破坏全球模式的潜在风险，大量工作引起了关注。通过文本分类和序列到序列任务中的NLP模型的稀有单词嵌入，研究了模型中毒用于后门攻击的可行性。在文本分类中，只有不到1%的敌意客户端足以在不降低干净句子性能的情况下操纵模型输出。对于不太复杂的数据集，仅0.1%的恶意客户端就足以有效地毒化全局模型。我们还提出了一种专门用于联邦学习方案的技术，称为梯度集成，它在所有实验设置中都提高了后门性能。



## **36. Using 3D Shadows to Detect Object Hiding Attacks on Autonomous Vehicle Perception**

利用3D阴影检测自主车辆感知中的目标隐藏攻击 cs.CV

To appear in the Proceedings of the 2022 IEEE Security and Privacy  Workshop on the Internet of Safe Things (SafeThings 2022)

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13973v1)

**Authors**: Zhongyuan Hau, Soteris Demetriou, Emil C. Lupu

**Abstracts**: Autonomous Vehicles (AVs) are mostly reliant on LiDAR sensors which enable spatial perception of their surroundings and help make driving decisions. Recent works demonstrated attacks that aim to hide objects from AV perception, which can result in severe consequences. 3D shadows, are regions void of measurements in 3D point clouds which arise from occlusions of objects in a scene. 3D shadows were proposed as a physical invariant valuable for detecting spoofed or fake objects. In this work, we leverage 3D shadows to locate obstacles that are hidden from object detectors. We achieve this by searching for void regions and locating the obstacles that cause these shadows. Our proposed methodology can be used to detect an object that has been hidden by an adversary as these objects, while hidden from 3D object detectors, still induce shadow artifacts in 3D point clouds, which we use for obstacle detection. We show that using 3D shadows for obstacle detection can achieve high accuracy in matching shadows to their object and provide precise prediction of an obstacle's distance from the ego-vehicle.

摘要: 自动驾驶汽车(AVs)大多依赖于LiDAR传感器，该传感器能够对周围环境进行空间感知，并帮助做出驾驶决策。最近的研究表明，攻击的目的是隐藏对象，使其不被反病毒感知，这可能会导致严重的后果。3D阴影是由于场景中对象的遮挡而导致的3D点云中没有测量结果的区域。3D阴影被认为是一种物理不变量，对于检测欺骗或虚假对象很有价值。在这项工作中，我们利用3D阴影来定位物体探测器隐藏的障碍物。我们通过搜索空洞区域和定位导致这些阴影的障碍物来实现这一点。我们提出的方法可以用于检测被对手隐藏的对象，因为这些对象虽然隐藏在3D对象检测器之外，但仍然会在3D点云中产生阴影伪影，我们将其用于障碍物检测。结果表明，使用3D阴影进行障碍物检测可以达到较高的匹配精度，并能准确预测障碍物与自行车者之间的距离。



## **37. Detecting Textual Adversarial Examples Based on Distributional Characteristics of Data Representations**

基于数据表示分布特征的文本对抗性实例检测 cs.CL

13 pages, RepL4NLP 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.13853v1)

**Authors**: Na Liu, Mark Dras, Wei Emma Zhang

**Abstracts**: Although deep neural networks have achieved state-of-the-art performance in various machine learning tasks, adversarial examples, constructed by adding small non-random perturbations to correctly classified inputs, successfully fool highly expressive deep classifiers into incorrect predictions. Approaches to adversarial attacks in natural language tasks have boomed in the last five years using character-level, word-level, phrase-level, or sentence-level textual perturbations. While there is some work in NLP on defending against such attacks through proactive methods, like adversarial training, there is to our knowledge no effective general reactive approaches to defence via detection of textual adversarial examples such as is found in the image processing literature. In this paper, we propose two new reactive methods for NLP to fill this gap, which unlike the few limited application baselines from NLP are based entirely on distribution characteristics of learned representations: we adapt one from the image processing literature (Local Intrinsic Dimensionality (LID)), and propose a novel one (MultiDistance Representation Ensemble Method (MDRE)). Adapted LID and MDRE obtain state-of-the-art results on character-level, word-level, and phrase-level attacks on the IMDB dataset as well as on the later two with respect to the MultiNLI dataset. For future research, we publish our code.

摘要: 尽管深度神经网络在各种机器学习任务中取得了最先进的性能，但通过在正确分类的输入中添加微小的非随机扰动而构建的对抗性例子，成功地愚弄了高表达能力的深度分类器，导致了错误的预测。在过去的五年中，自然语言任务中使用字符级别、单词级别、短语级别或句子级别的文本扰动进行对抗性攻击的方法得到了蓬勃发展。虽然NLP中有一些关于通过主动方法来防御这种攻击的工作，如对抗性训练，但据我们所知，没有有效的一般反应性方法来通过检测文本对抗性实例来防御，例如在图像处理文献中找到的。在本文中，我们提出了两种新的反应式方法来填补这一空白，这两种方法不同于NLP的少数有限的应用基线完全基于学习表示的分布特征：我们借鉴了图像处理文献中的一种方法(局部本征维度(LID))，并提出了一种新的方法(多距离表示集成方法(MDRE))。适配的LID和MDRE获得了关于IMDB数据集的字符级、词级和短语级攻击以及关于MultiNLI数据集的后两种攻击的最新结果。为了将来的研究，我们发布了我们的代码。



## **38. DeepAdversaries: Examining the Robustness of Deep Learning Models for Galaxy Morphology Classification**

深度学习：检验深度学习模型对星系形态分类的稳健性 cs.LG

19 pages, 7 figures, 5 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2112.14299v2)

**Authors**: Aleksandra Ćiprijanović, Diana Kafkes, Gregory Snyder, F. Javier Sánchez, Gabriel Nathan Perdue, Kevin Pedro, Brian Nord, Sandeep Madireddy, Stefan M. Wild

**Abstracts**: Data processing and analysis pipelines in cosmological survey experiments introduce data perturbations that can significantly degrade the performance of deep learning-based models. Given the increased adoption of supervised deep learning methods for processing and analysis of cosmological survey data, the assessment of data perturbation effects and the development of methods that increase model robustness are increasingly important. In the context of morphological classification of galaxies, we study the effects of perturbations in imaging data. In particular, we examine the consequences of using neural networks when training on baseline data and testing on perturbed data. We consider perturbations associated with two primary sources: 1) increased observational noise as represented by higher levels of Poisson noise and 2) data processing noise incurred by steps such as image compression or telescope errors as represented by one-pixel adversarial attacks. We also test the efficacy of domain adaptation techniques in mitigating the perturbation-driven errors. We use classification accuracy, latent space visualizations, and latent space distance to assess model robustness. Without domain adaptation, we find that processing pixel-level errors easily flip the classification into an incorrect class and that higher observational noise makes the model trained on low-noise data unable to classify galaxy morphologies. On the other hand, we show that training with domain adaptation improves model robustness and mitigates the effects of these perturbations, improving the classification accuracy by 23% on data with higher observational noise. Domain adaptation also increases by a factor of ~2.3 the latent space distance between the baseline and the incorrectly classified one-pixel perturbed image, making the model more robust to inadvertent perturbations.

摘要: 宇宙学测量实验中的数据处理和分析管道引入了数据扰动，这可能会显著降低基于深度学习的模型的性能。鉴于越来越多的人采用有监督的深度学习方法来处理和分析宇宙学观测数据，评估数据扰动效应和开发提高模型稳健性的方法变得越来越重要。在星系形态分类的背景下，我们研究了成像数据中微扰的影响。特别是，我们检查了在对基线数据进行训练和对扰动数据进行测试时使用神经网络的后果。我们考虑与两个主要来源相关的扰动：1)以更高水平的泊松噪声为代表的观测噪声的增加；2)以单像素对抗性攻击为代表的图像压缩或望远镜误差等步骤所引起的数据处理噪声。我们还测试了领域自适应技术在缓解扰动驱动的错误方面的有效性。我们使用分类精度、潜在空间可视化和潜在空间距离来评估模型的稳健性。在没有域自适应的情况下，我们发现处理像素级误差很容易将分类反转到不正确的类别，并且更高的观测噪声使得基于低噪声数据训练的模型无法对星系形态进行分类。另一方面，我们表明，域自适应训练提高了模型的稳健性，缓解了这些扰动的影响，在观测噪声较高的数据上将分类精度提高了23%。域自适应还将基线和错误分类的单像素扰动图像之间的潜在空间距离增加了约2.3倍，使模型对无意扰动更具鲁棒性。



## **39. Survey and Taxonomy of Adversarial Reconnaissance Techniques**

对抗性侦察技术综述及分类 cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2105.04749v2)

**Authors**: Shanto Roy, Nazia Sharmin, Jaime C. Acosta, Christopher Kiekintveld, Aron Laszka

**Abstracts**: Adversaries are often able to penetrate networks and compromise systems by exploiting vulnerabilities in people and systems. The key to the success of these attacks is information that adversaries collect throughout the phases of the cyber kill chain. We summarize and analyze the methods, tactics, and tools that adversaries use to conduct reconnaissance activities throughout the attack process. First, we discuss what types of information adversaries seek, and how and when they can obtain this information. Then, we provide a taxonomy and detailed overview of adversarial reconnaissance techniques. The taxonomy introduces a categorization of reconnaissance techniques based on the source as third-party, human-, and system-based information gathering. This paper provides a comprehensive view of adversarial reconnaissance that can help in understanding and modeling this complex but vital aspect of cyber attacks as well as insights that can improve defensive strategies, such as cyber deception.

摘要: 攻击者通常能够通过利用人和系统中的漏洞来渗透网络并危害系统。这些攻击成功的关键是对手在网络杀伤链的各个阶段收集的信息。我们总结和分析了对手在整个攻击过程中进行侦察活动所使用的方法、战术和工具。首先，我们讨论对手寻求什么类型的信息，以及他们如何以及何时可以获得这些信息。然后，我们对对抗性侦察技术进行了分类和详细的概述。该分类引入了基于来源的侦察技术分类，即第三方、基于人员和基于系统的信息收集。本文提供了对抗性侦察的全面观点，有助于理解和建模网络攻击的这一复杂但至关重要的方面，以及可以改进防御策略的见解，例如网络欺骗。



## **40. AGIC: Approximate Gradient Inversion Attack on Federated Learning**

AGIC：联邦学习中的近似梯度反转攻击 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13784v1)

**Authors**: Jin Xu, Chi Hong, Jiyue Huang, Lydia Y. Chen, Jérémie Decouchant

**Abstracts**: Federated learning is a private-by-design distributed learning paradigm where clients train local models on their own data before a central server aggregates their local updates to compute a global model. Depending on the aggregation method used, the local updates are either the gradients or the weights of local learning models. Recent reconstruction attacks apply a gradient inversion optimization on the gradient update of a single minibatch to reconstruct the private data used by clients during training. As the state-of-the-art reconstruction attacks solely focus on single update, realistic adversarial scenarios are overlooked, such as observation across multiple updates and updates trained from multiple mini-batches. A few studies consider a more challenging adversarial scenario where only model updates based on multiple mini-batches are observable, and resort to computationally expensive simulation to untangle the underlying samples for each local step. In this paper, we propose AGIC, a novel Approximate Gradient Inversion Attack that efficiently and effectively reconstructs images from both model or gradient updates, and across multiple epochs. In a nutshell, AGIC (i) approximates gradient updates of used training samples from model updates to avoid costly simulation procedures, (ii) leverages gradient/model updates collected from multiple epochs, and (iii) assigns increasing weights to layers with respect to the neural network structure for reconstruction quality. We extensively evaluate AGIC on three datasets, CIFAR-10, CIFAR-100 and ImageNet. Our results show that AGIC increases the peak signal-to-noise ratio (PSNR) by up to 50% compared to two representative state-of-the-art gradient inversion attacks. Furthermore, AGIC is faster than the state-of-the-art simulation based attack, e.g., it is 5x faster when attacking FedAvg with 8 local steps in between model updates.

摘要: 联合学习是一种私人设计的分布式学习范例，其中客户端在中央服务器聚合其本地更新以计算全局模型之前，根据自己的数据训练本地模型。根据所使用的聚合方法，局部更新要么是局部学习模型的梯度，要么是局部学习模型的权重。最近的重建攻击将梯度倒置优化应用于单个小批量的梯度更新，以重建客户在训练期间使用的私有数据。由于最新的重建攻击只关注单个更新，因此忽略了现实的对抗性场景，例如跨多个更新的观察和从多个小批次训练的更新。一些研究考虑了一种更具挑战性的对抗性场景，其中只能观察到基于多个小批次的模型更新，并求助于计算代价高昂的模拟来解开每个局部步骤的潜在样本。在本文中，我们提出了AGIC，一种新的近似梯度反转攻击，它可以高效地从模型或梯度更新中重建图像，并跨越多个历元。简而言之，AGIC(I)根据模型更新近似使用的训练样本的梯度更新以避免昂贵的模拟过程，(Ii)利用从多个历元收集的梯度/模型更新，以及(Iii)为重建质量向层分配相对于神经网络结构的不断增加的权重。我们在三个数据集CIFAR-10、CIFAR-100和ImageNet上对AGIC进行了广泛的评估。实验结果表明，与两种典型的梯度反转攻击相比，AGIC的峰值信噪比(PSNR)提高了50%。此外，AGIC比最先进的基于模拟的攻击速度更快，例如，在模型更新之间有8个本地步骤的情况下，攻击FedAvg的速度要快5倍。



## **41. Formulating Robustness Against Unforeseen Attacks**

针对不可预见的攻击形成健壮性 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13779v1)

**Authors**: Sihui Dai, Saeed Mahloujifar, Prateek Mittal

**Abstracts**: Existing defenses against adversarial examples such as adversarial training typically assume that the adversary will conform to a specific or known threat model, such as $\ell_p$ perturbations within a fixed budget. In this paper, we focus on the scenario where there is a mismatch in the threat model assumed by the defense during training, and the actual capabilities of the adversary at test time. We ask the question: if the learner trains against a specific "source" threat model, when can we expect robustness to generalize to a stronger unknown "target" threat model during test-time? Our key contribution is to formally define the problem of learning and generalization with an unforeseen adversary, which helps us reason about the increase in adversarial risk from the conventional perspective of a known adversary. Applying our framework, we derive a generalization bound which relates the generalization gap between source and target threat models to variation of the feature extractor, which measures the expected maximum difference between extracted features across a given threat model. Based on our generalization bound, we propose adversarial training with variation regularization (AT-VR) which reduces variation of the feature extractor across the source threat model during training. We empirically demonstrate that AT-VR can lead to improved generalization to unforeseen attacks during test-time compared to standard adversarial training on Gaussian and image datasets.

摘要: 现有的针对对抗性示例的防御，例如对抗性训练，通常假设对手将符合特定或已知的威胁模型，例如固定预算内的$\ell_p$扰动。在本文中，我们重点讨论在训练过程中防御方假设的威胁模型与测试时对手的实际能力存在不匹配的情况。我们问这样一个问题：如果学习者针对特定的“源”威胁模型进行训练，我们何时才能期望健壮性在测试期间推广到更强的未知“目标”威胁模型？我们的主要贡献是正式定义了与不可预见的对手的学习和泛化问题，这有助于我们从已知对手的传统角度来推理对手风险的增加。应用我们的框架，我们得到了一个泛化界限，它将源威胁模型和目标威胁模型之间的泛化差距与特征抽取器的变化联系起来，它度量了在给定威胁模型中提取的特征之间的期望最大差异。基于我们的泛化界，我们提出了带变异正则化的对抗性训练(AT-VR)，它减少了训练过程中特征提取子在源威胁模型上的变异。我们的实验证明，与基于高斯和图像数据集的标准对抗性训练相比，AT-VR能够提高对测试时间内不可预见攻击的泛化能力。



## **42. UNBUS: Uncertainty-aware Deep Botnet Detection System in Presence of Perturbed Samples**

UNBUS：存在扰动样本的不确定性感知深度僵尸网络检测系统 cs.CR

8 pages, 5 figures, 5 Tables

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.09502v2)

**Authors**: Rahim Taheri

**Abstracts**: A rising number of botnet families have been successfully detected using deep learning architectures. While the variety of attacks increases, these architectures should become more robust against attacks. They have been proven to be very sensitive to small but well constructed perturbations in the input. Botnet detection requires extremely low false-positive rates (FPR), which are not commonly attainable in contemporary deep learning. Attackers try to increase the FPRs by making poisoned samples. The majority of recent research has focused on the use of model loss functions to build adversarial examples and robust models. In this paper, two LSTM-based classification algorithms for botnet classification with an accuracy higher than 98% are presented. Then, the adversarial attack is proposed, which reduces the accuracy to about 30%. Then, by examining the methods for computing the uncertainty, the defense method is proposed to increase the accuracy to about 70%. By using the deep ensemble and stochastic weight averaging quantification methods it has been investigated the uncertainty of the accuracy in the proposed methods.

摘要: 使用深度学习体系结构已成功检测到越来越多的僵尸网络家族。随着攻击种类的增加，这些体系结构应该变得更强大，以抵御攻击。事实证明，它们对输入中的微小但构造良好的扰动非常敏感。僵尸网络检测需要极低的假阳性率(FPR)，这在当代深度学习中是不常见的。攻击者试图通过制作有毒样本来增加FPR。最近的大多数研究都集中在使用模型损失函数来构建对抗性例子和稳健模型。本文提出了两种基于LSTM的僵尸网络分类算法，分类正确率高于98%。然后，提出了对抗性攻击，使准确率降低到30%左右。然后，通过研究不确定度的计算方法，提出了将准确度提高到70%左右的防御方法。通过使用深度集成和随机加权平均量化方法，对所提出方法的精度的不确定度进行了研究。



## **43. Deepfake Forensics via An Adversarial Game**

通过对抗性游戏进行深度假冒取证 cs.CV

Accepted by IEEE Transactions on Image Processing; 13 pages, 4  figures

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2103.13567v2)

**Authors**: Zhi Wang, Yiwen Guo, Wangmeng Zuo

**Abstracts**: With the progress in AI-based facial forgery (i.e., deepfake), people are increasingly concerned about its abuse. Albeit effort has been made for training classification (also known as deepfake detection) models to recognize such forgeries, existing models suffer from poor generalization to unseen forgery technologies and high sensitivity to changes in image/video quality. In this paper, we advocate adversarial training for improving the generalization ability to both unseen facial forgeries and unseen image/video qualities. We believe training with samples that are adversarially crafted to attack the classification models improves the generalization ability considerably. Considering that AI-based face manipulation often leads to high-frequency artifacts that can be easily spotted by models yet difficult to generalize, we further propose a new adversarial training method that attempts to blur out these specific artifacts, by introducing pixel-wise Gaussian blurring models. With adversarial training, the classification models are forced to learn more discriminative and generalizable features, and the effectiveness of our method can be verified by plenty of empirical evidence. Our code will be made publicly available.

摘要: 随着基于人工智能的人脸伪造(即深度假)的发展，人们越来越关注它的滥用。尽管已经努力训练分类(也称为深度伪检测)模型来识别此类伪造物，但现有模型对不可见的伪造物技术的泛化能力差，并且对图像/视频质量的变化高度敏感。在本文中，我们提倡对抗性训练，以提高对看不见的人脸伪造和看不见的图像/视频质量的泛化能力。我们相信，用恶意设计的样本来攻击分类模型的训练大大提高了泛化能力。考虑到基于人工智能的人脸操作往往会导致高频伪影，这些伪影很容易被模型发现，但很难推广，我们进一步提出了一种新的对抗性训练方法，试图通过引入像素级的高斯模糊模型来模糊这些特定的伪影。通过对抗性训练，迫使分类模型学习更具区分性和泛化能力的特征，并通过大量的经验证据验证了该方法的有效性。我们的代码将公开可用。



## **44. Randomized Smoothing under Attack: How Good is it in Pratice?**

攻击下的随机平滑：它在实践中有多好？ cs.CR

ICASSP 2022

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.14187v1)

**Authors**: Thibault Maho, Teddy Furon, Erwan Le Merrer

**Abstracts**: Randomized smoothing is a recent and celebrated solution to certify the robustness of any classifier. While it indeed provides a theoretical robustness against adversarial attacks, the dimensionality of current classifiers necessarily imposes Monte Carlo approaches for its application in practice. This paper questions the effectiveness of randomized smoothing as a defense, against state of the art black-box attacks. This is a novel perspective, as previous research works considered the certification as an unquestionable guarantee. We first formally highlight the mismatch between a theoretical certification and the practice of attacks on classifiers. We then perform attacks on randomized smoothing as a defense. Our main observation is that there is a major mismatch in the settings of the RS for obtaining high certified robustness or when defeating black box attacks while preserving the classifier accuracy.

摘要: 随机平滑是最近一个著名的解决方案，用来证明任何分类器的稳健性。虽然它确实在理论上提供了对对手攻击的稳健性，但当前分类器的维度必然要求它在实践中应用蒙特卡罗方法。本文对随机平滑作为防御最先进的黑盒攻击的有效性提出了质疑。这是一个新的观点，因为以前的研究工作认为认证是毋庸置疑的保证。我们首先正式强调理论证明和对分类器的攻击实践之间的不匹配。然后我们对随机平滑进行攻击，作为一种防御。我们的主要观察是，在RS的设置中存在严重的不匹配，以获得高度认证的稳健性，或者当击败黑盒攻击时，同时保持分类器的准确性。



## **45. Adversarial Fine-tune with Dynamically Regulated Adversary**

动态调整对手的对抗性微调 cs.LG

**SubmitDate**: 2022-04-28    [paper-pdf](http://arxiv.org/pdf/2204.13232v1)

**Authors**: Pengyue Hou, Ming Zhou, Jie Han, Petr Musilek, Xingyu Li

**Abstracts**: Adversarial training is an effective method to boost model robustness to malicious, adversarial attacks. However, such improvement in model robustness often leads to a significant sacrifice of standard performance on clean images. In many real-world applications such as health diagnosis and autonomous surgical robotics, the standard performance is more valued over model robustness against such extremely malicious attacks. This leads to the question: To what extent we can boost model robustness without sacrificing standard performance? This work tackles this problem and proposes a simple yet effective transfer learning-based adversarial training strategy that disentangles the negative effects of adversarial samples on model's standard performance. In addition, we introduce a training-friendly adversarial attack algorithm, which facilitates the boost of adversarial robustness without introducing significant training complexity. Extensive experimentation indicates that the proposed method outperforms previous adversarial training algorithms towards the target: to improve model robustness while preserving model's standard performance on clean data.

摘要: 对抗性训练是提高模型对恶意、对抗性攻击稳健性的有效方法。然而，这种模型稳健性的改进经常导致在干净图像上的标准性能的显著牺牲。在许多真实世界的应用中，例如健康诊断和自主手术机器人，对于这种极端恶意的攻击，标准性能比模型健壮性更受重视。这就引出了一个问题：在不牺牲标准性能的情况下，我们可以在多大程度上提高模型的健壮性？针对这一问题，提出了一种简单而有效的基于迁移学习的对抗性训练策略，消除了对抗性样本对模型标准性能的负面影响。此外，我们还引入了一种训练友好的对抗性攻击算法，该算法在不引入显著训练复杂度的情况下，有助于提高对抗性攻击的健壮性。大量实验表明，该方法优于以往对抗性训练算法的目标：在保持模型在干净数据上的标准性能的同时，提高模型的稳健性。



## **46. An Adversarial Attack Analysis on Malicious Advertisement URL Detection Framework**

恶意广告URL检测框架的对抗性攻击分析 cs.LG

13

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13172v1)

**Authors**: Ehsan Nowroozi, Abhishek, Mohammadreza Mohammadi, Mauro Conti

**Abstracts**: Malicious advertisement URLs pose a security risk since they are the source of cyber-attacks, and the need to address this issue is growing in both industry and academia. Generally, the attacker delivers an attack vector to the user by means of an email, an advertisement link or any other means of communication and directs them to a malicious website to steal sensitive information and to defraud them. Existing malicious URL detection techniques are limited and to handle unseen features as well as generalize to test data. In this study, we extract a novel set of lexical and web-scrapped features and employ machine learning technique to set up system for fraudulent advertisement URLs detection. The combination set of six different kinds of features precisely overcome the obfuscation in fraudulent URL classification. Based on different statistical properties, we use twelve different formatted datasets for detection, prediction and classification task. We extend our prediction analysis for mismatched and unlabelled datasets. For this framework, we analyze the performance of four machine learning techniques: Random Forest, Gradient Boost, XGBoost and AdaBoost in the detection part. With our proposed method, we can achieve a false negative rate as low as 0.0037 while maintaining high accuracy of 99.63%. Moreover, we devise a novel unsupervised technique for data clustering using K- Means algorithm for the visual analysis. This paper analyses the vulnerability of decision tree-based models using the limited knowledge attack scenario. We considered the exploratory attack and implemented Zeroth Order Optimization adversarial attack on the detection models.

摘要: 恶意广告URL构成了安全风险，因为它们是网络攻击的来源，而且在工业界和学术界，解决这一问题的需求都在不断增长。通常，攻击者通过电子邮件、广告链接或任何其他通信方式向用户发送攻击矢量，并将他们定向到恶意网站，以窃取敏感信息并诈骗他们。现有的恶意URL检测技术在处理看不见的功能以及泛化测试数据方面都是有限的。在这项研究中，我们提取了一组新颖的词汇和网页废弃特征，并利用机器学习技术建立了欺诈性广告URL检测系统。六种不同特征的组合集合恰好克服了欺诈性URL分类中的混淆。基于不同的统计特性，我们使用了12个不同格式的数据集进行检测、预测和分类任务。我们将我们的预测分析扩展到不匹配和未标记的数据集。在检测部分，分析了四种机器学习技术：随机森林、梯度增强、XGBoost和AdaBoost的性能。该方法在保持99.63%的准确率的同时，假阴性率可低至0.0037。此外，我们设计了一种新的无监督数据聚类技术，使用K-Means算法进行可视化分析。分析了基于决策树的模型在有限知识攻击场景下的脆弱性。考虑了探索性攻击，在检测模型上实现了零阶优化对抗性攻击。



## **47. SSR-GNNs: Stroke-based Sketch Representation with Graph Neural Networks**

SSR-GNNS：基于图形神经网络的笔画表示 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13153v1)

**Authors**: Sheng Cheng, Yi Ren, Yezhou Yang

**Abstracts**: This paper follows cognitive studies to investigate a graph representation for sketches, where the information of strokes, i.e., parts of a sketch, are encoded on vertices and information of inter-stroke on edges. The resultant graph representation facilitates the training of a Graph Neural Networks for classification tasks, and achieves accuracy and robustness comparable to the state-of-the-art against translation and rotation attacks, as well as stronger attacks on graph vertices and topologies, i.e., modifications and addition of strokes, all without resorting to adversarial training. Prior studies on sketches, e.g., graph transformers, encode control points of stroke on vertices, which are not invariant to spatial transformations. In contrary, we encode vertices and edges using pairwise distances among control points to achieve invariance. Compared with existing generative sketch model for one-shot classification, our method does not rely on run-time statistical inference. Lastly, the proposed representation enables generation of novel sketches that are structurally similar to while separable from the existing dataset.

摘要: 在认知研究的基础上，对素描的图形表示进行了研究，其中笔画的信息，即草图的部分，在顶点上编码，边上的笔画间的信息编码。所得到的图表示促进了图神经网络的分类任务的训练，并且获得了与最新技术相媲美的针对平移和旋转攻击的准确性和稳健性，以及对图顶点和拓扑的更强攻击，即修改和添加笔划，所有这些都不求助于对抗性训练。以往对草图的研究，例如图形转换器，对顶点上的笔划控制点进行编码，而这些控制点并不是空间变换的不变性。相反，我们使用控制点之间的成对距离对顶点和边进行编码，以实现不变性。与现有的一次分类生成式草图模型相比，该方法不依赖于运行时的统计推理。最后，所提出的表示法能够生成在结构上与现有数据集相似但可与现有数据集分开的新草图。



## **48. Defending Against Person Hiding Adversarial Patch Attack with a Universal White Frame**

用通用白框防御隐藏敌方补丁攻击的人 cs.CV

Submitted by NeurIPS 2021 with response letter to the anonymous  reviewers' comments

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.13004v1)

**Authors**: Youngjoon Yu, Hong Joo Lee, Hakmin Lee, Yong Man Ro

**Abstracts**: Object detection has attracted great attention in the computer vision area and has emerged as an indispensable component in many vision systems. In the era of deep learning, many high-performance object detection networks have been proposed. Although these detection networks show high performance, they are vulnerable to adversarial patch attacks. Changing the pixels in a restricted region can easily fool the detection network in the physical world. In particular, person-hiding attacks are emerging as a serious problem in many safety-critical applications such as autonomous driving and surveillance systems. Although it is necessary to defend against an adversarial patch attack, very few efforts have been dedicated to defending against person-hiding attacks. To tackle the problem, in this paper, we propose a novel defense strategy that mitigates a person-hiding attack by optimizing defense patterns, while previous methods optimize the model. In the proposed method, a frame-shaped pattern called a 'universal white frame' (UWF) is optimized and placed on the outside of the image. To defend against adversarial patch attacks, UWF should have three properties (i) suppressing the effect of the adversarial patch, (ii) maintaining its original prediction, and (iii) applicable regardless of images. To satisfy the aforementioned properties, we propose a novel pattern optimization algorithm that can defend against the adversarial patch. Through comprehensive experiments, we demonstrate that the proposed method effectively defends against the adversarial patch attack.

摘要: 目标检测在计算机视觉领域引起了极大的关注，已经成为许多视觉系统中不可或缺的组成部分。在深度学习时代，已经提出了许多高性能的目标检测网络。虽然这些检测网络表现出高性能，但它们很容易受到对抗性补丁攻击。更改受限区域中的像素可以很容易地欺骗物理世界中的检测网络。特别是，在自动驾驶和监控系统等许多安全关键应用中，藏人攻击正在成为一个严重的问题。尽管防御对抗性补丁攻击是必要的，但很少有人致力于防御人员隐藏攻击。针对这一问题，本文提出了一种新的防御策略，该策略通过优化防御模式来缓解人员躲藏攻击，而以往的方法则对该模型进行了优化。在所提出的方法中，一个被称为“通用白框”(UWF)的框形图案被优化并放置在图像的外部。为了防御对抗性补丁攻击，UWF应该具有三个性质(I)抑制对抗性补丁的影响，(Ii)保持其原始预测，以及(Iii)适用于任何图像。为了满足上述性质，我们提出了一种新的模式优化算法，该算法能够防御恶意补丁。通过综合实验，我们证明了该方法能够有效地防御敌意补丁攻击。



## **49. The MeVer DeepFake Detection Service: Lessons Learnt from Developing and Deploying in the Wild**

Mever DeepFake检测服务：从野外开发和部署中吸取的教训 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12816v1)

**Authors**: Spyridon Baxevanakis, Giorgos Kordopatis-Zilos, Panagiotis Galopoulos, Lazaros Apostolidis, Killian Levacher, Ipek B. Schlicht, Denis Teyssou, Ioannis Kompatsiaris, Symeon Papadopoulos

**Abstracts**: Enabled by recent improvements in generation methodologies, DeepFakes have become mainstream due to their increasingly better visual quality, the increase in easy-to-use generation tools and the rapid dissemination through social media. This fact poses a severe threat to our societies with the potential to erode social cohesion and influence our democracies. To mitigate the threat, numerous DeepFake detection schemes have been introduced in the literature but very few provide a web service that can be used in the wild. In this paper, we introduce the MeVer DeepFake detection service, a web service detecting deep learning manipulations in images and video. We present the design and implementation of the proposed processing pipeline that involves a model ensemble scheme, and we endow the service with a model card for transparency. Experimental results show that our service performs robustly on the three benchmark datasets while being vulnerable to Adversarial Attacks. Finally, we outline our experience and lessons learned when deploying a research system into production in the hopes that it will be useful to other academic and industry teams.

摘要: 由于最近在生成方法上的改进，DeepFake已经成为主流，因为它们的视觉质量越来越好，易于使用的生成工具的增加，以及通过社交媒体的快速传播。这一事实对我们的社会构成严重威胁，有可能侵蚀社会凝聚力并影响我们的民主国家。为了减轻威胁，文献中已经引入了许多DeepFake检测方案，但很少提供可以在野外使用的Web服务。在本文中，我们介绍了Mever DeepFake检测服务，这是一个检测图像和视频中的深度学习操作的Web服务。我们给出了所提出的处理流水线的设计和实现，该流水线涉及模型集成方案，并且我们为服务赋予模型卡以实现透明性。实验结果表明，我们的服务在三个基准数据集上表现出很好的性能，但很容易受到对手攻击。最后，我们概述了我们在将研究系统部署到生产中时的经验和教训，希望它对其他学术和行业团队有用。



## **50. Improving the Transferability of Adversarial Examples with Restructure Embedded Patches**

利用重构嵌入补丁提高对抗性实例的可转移性 cs.CV

**SubmitDate**: 2022-04-27    [paper-pdf](http://arxiv.org/pdf/2204.12680v1)

**Authors**: Huipeng Zhou, Yu-an Tan, Yajie Wang, Haoran Lyu, Shangbo Wu, Yuanzhang Li

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance in various computer vision tasks. However, the adversarial examples generated by ViTs are challenging to transfer to other networks with different structures. Recent attack methods do not consider the specificity of ViTs architecture and self-attention mechanism, which leads to poor transferability of the generated adversarial samples by ViTs. We attack the unique self-attention mechanism in ViTs by restructuring the embedded patches of the input. The restructured embedded patches enable the self-attention mechanism to obtain more diverse patches connections and help ViTs keep regions of interest on the object. Therefore, we propose an attack method against the unique self-attention mechanism in ViTs, called Self-Attention Patches Restructure (SAPR). Our method is simple to implement yet efficient and applicable to any self-attention based network and gradient transferability-based attack methods. We evaluate attack transferability on black-box models with different structures. The result show that our method generates adversarial examples on white-box ViTs with higher transferability and higher image quality. Our research advances the development of black-box transfer attacks on ViTs and demonstrates the feasibility of using white-box ViTs to attack other black-box models.

摘要: 视觉转换器(VITS)在各种计算机视觉任务中表现出令人印象深刻的性能。然而，VITS生成的对抗性例子很难转移到其他具有不同结构的网络上。现有的攻击方法没有考虑VITS体系结构和自我注意机制的特殊性，导致VITS生成的攻击样本可移植性较差。我们通过重组输入的嵌入补丁来攻击VITS中独特的自我注意机制。重构后的嵌入贴片使自我注意机制能够获得更多样化的贴片连接，并帮助VITS保持对象上的感兴趣区域。因此，我们提出了一种针对VITS中独特的自我注意机制的攻击方法，称为自我注意补丁重构(SAPR)。该方法实现简单，效率高，适用于任何基于自我注意的网络攻击方法和基于梯度转移的攻击方法。我们在不同结构的黑盒模型上评估了攻击的可转移性。实验结果表明，该方法在白盒VITS上生成的对抗性样本具有较高的可移植性和较高的图像质量。我们的研究推动了针对VITS的黑盒传输攻击的发展，并论证了利用白盒VITS攻击其他黑盒模型的可行性。



