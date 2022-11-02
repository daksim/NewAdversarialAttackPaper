# Latest Adversarial Attack Papers
**update at 2022-11-03 06:31:34**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Hyperspectral Adversarial Robustness using Ensemble Networks in the Presences of Multiple Attacks**

利用集成网络提高多重攻击下的高谱对抗鲁棒性 cs.LG

6 pages, 2 figures, 1 table, 1 algorithm

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2210.16346v2) [paper-pdf](http://arxiv.org/pdf/2210.16346v2)

**Authors**: Nicholas Soucy, Salimeh Yasaei Sekeh

**Abstract**: Semantic segmentation of hyperspectral images (HSI) has seen great strides in recent years by incorporating knowledge from deep learning RGB classification models. Similar to their classification counterparts, semantic segmentation models are vulnerable to adversarial examples and need adversarial training to counteract them. Traditional approaches to adversarial robustness focus on training or retraining a single network on attacked data, however, in the presence of multiple attacks these approaches decrease the performance compared to networks trained individually on each attack. To combat this issue we propose an Adversarial Discriminator Ensemble Network (ADE-Net) which focuses on attack type detection and adversarial robustness under a unified model to preserve per data-type weight optimally while robustifiying the overall network. In the proposed method, a discriminator network is used to separate data by attack type into their specific attack-expert ensemble network. Our approach allows for the presence of multiple attacks mixed together while also labeling attack types during testing. We experimentally show that ADE-Net outperforms the baseline, which is a single network adversarially trained under a mix of multiple attacks, for HSI Indian Pines, Kennedy Space, and Houston datasets.

摘要: 近年来，通过融合深度学习RGB分类模型的知识，高光谱图像的语义分割(HSI)取得了长足的进步。与分类模型类似，语义分割模型容易受到对抗性实例的影响，需要对抗性训练来抵消它们。传统的对抗稳健性方法侧重于针对受攻击的数据训练或重新训练单个网络，然而，在存在多个攻击的情况下，与针对每个攻击单独训练的网络相比，这些方法降低了性能。为了解决这个问题，我们提出了一种对抗性鉴别集成网络(ADE-Net)，它在统一的模型下关注攻击类型的检测和对抗性的健壮性，以便在使整个网络稳健的同时最优地保持每种数据类型的权重。在该方法中，利用鉴别器网络根据攻击类型将数据分离到其特定的攻击专家集成网络中。我们的方法允许混合存在多个攻击，同时还在测试期间标记攻击类型。我们的实验表明，ADE-Net在HSI Indian Pines、Kennedy Space和Houston数据集上的性能优于基线，后者是在多种攻击的混合情况下进行对抗性训练的单一网络。



## **2. A Unified Evaluation of Textual Backdoor Learning: Frameworks and Benchmarks**

文本后门学习的统一评价：框架和基准 cs.LG

NeurIPS 2022 Datasets & Benchmarks; Toolkits avaliable at  https://github.com/thunlp/OpenBackdoor

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2206.08514v2) [paper-pdf](http://arxiv.org/pdf/2206.08514v2)

**Authors**: Ganqu Cui, Lifan Yuan, Bingxiang He, Yangyi Chen, Zhiyuan Liu, Maosong Sun

**Abstract**: Textual backdoor attacks are a kind of practical threat to NLP systems. By injecting a backdoor in the training phase, the adversary could control model predictions via predefined triggers. As various attack and defense models have been proposed, it is of great significance to perform rigorous evaluations. However, we highlight two issues in previous backdoor learning evaluations: (1) The differences between real-world scenarios (e.g. releasing poisoned datasets or models) are neglected, and we argue that each scenario has its own constraints and concerns, thus requires specific evaluation protocols; (2) The evaluation metrics only consider whether the attacks could flip the models' predictions on poisoned samples and retain performances on benign samples, but ignore that poisoned samples should also be stealthy and semantic-preserving. To address these issues, we categorize existing works into three practical scenarios in which attackers release datasets, pre-trained models, and fine-tuned models respectively, then discuss their unique evaluation methodologies. On metrics, to completely evaluate poisoned samples, we use grammar error increase and perplexity difference for stealthiness, along with text similarity for validity. After formalizing the frameworks, we develop an open-source toolkit OpenBackdoor to foster the implementations and evaluations of textual backdoor learning. With this toolkit, we perform extensive experiments to benchmark attack and defense models under the suggested paradigm. To facilitate the underexplored defenses against poisoned datasets, we further propose CUBE, a simple yet strong clustering-based defense baseline. We hope that our frameworks and benchmarks could serve as the cornerstones for future model development and evaluations.

摘要: 文本后门攻击是对NLP系统的一种实际威胁。通过在训练阶段插入后门，对手可以通过预定义的触发器控制模型预测。随着各种攻防模型的提出，进行严格的评估具有重要意义。然而，我们强调了以往的后门学习评估中的两个问题：(1)忽略了现实世界场景(如发布有毒数据集或模型)之间的差异，我们认为每个场景都有自己的约束和关注点，因此需要特定的评估协议；(2)评估指标只考虑攻击是否会颠覆模型对有毒样本的预测，并保持对良性样本的性能，而忽略了有毒样本也应该是隐蔽的和保持语义的。为了解决这些问题，我们将现有的工作分为三个实际场景，攻击者分别发布数据集、预先训练的模型和微调的模型，然后讨论他们独特的评估方法。在度量上，为了全面评估中毒样本，我们使用语法错误增加和困惑差异来隐蔽性，并使用文本相似性来验证有效性。在形式化框架之后，我们开发了一个开源工具包OpenBackdoor来促进文本后门学习的实现和评估。利用这个工具包，我们进行了大量的实验，在建议的范式下对攻击和防御模型进行基准测试。为了方便对有毒数据集的未充分挖掘的防御，我们进一步提出了CUBE，一种简单但强大的基于聚类的防御基线。我们希望我们的框架和基准能够成为未来模式发展和评价的基石。



## **3. A Comprehensive Evaluation Framework for Deep Model Robustness**

一种深度模型稳健性的综合评价框架 cs.CV

Submitted to Pattern Recognition

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2101.09617v2) [paper-pdf](http://arxiv.org/pdf/2101.09617v2)

**Authors**: Jun Guo, Wei Bao, Jiakai Wang, Yuqing Ma, Xinghai Gao, Gang Xiao, Aishan Liu, Jian Dong, Xianglong Liu, Wenjun Wu

**Abstract**: Deep neural networks (DNNs) have achieved remarkable performance across a wide range of applications, while they are vulnerable to adversarial examples, which motivates the evaluation and benchmark of model robustness. However, current evaluations usually use simple metrics to study the performance of defenses, which are far from understanding the limitation and weaknesses of these defense methods. Thus, most proposed defenses are quickly shown to be attacked successfully, which results in the ``arm race'' phenomenon between attack and defense. To mitigate this problem, we establish a model robustness evaluation framework containing 23 comprehensive and rigorous metrics, which consider two key perspectives of adversarial learning (i.e., data and model). Through neuron coverage and data imperceptibility, we use data-oriented metrics to measure the integrity of test examples; by delving into model structure and behavior, we exploit model-oriented metrics to further evaluate robustness in the adversarial setting. To fully demonstrate the effectiveness of our framework, we conduct large-scale experiments on multiple datasets including CIFAR-10, SVHN, and ImageNet using different models and defenses with our open-source platform. Overall, our paper provides a comprehensive evaluation framework, where researchers could conduct comprehensive and fast evaluations using the open-source toolkit, and the analytical results could inspire deeper understanding and further improvement to the model robustness.

摘要: 深度神经网络(DNN)在广泛的应用领域取得了显著的性能，但它们容易受到敌意例子的影响，这促使了对模型稳健性的评估和基准。然而，目前的评估通常采用简单的度量来研究防御的性能，这远远不能理解这些防御方法的局限性和弱点。因此，大多数提议的防御很快就被证明是成功的，这导致了进攻和防守之间的“军备竞赛”现象。为了缓解这一问题，我们建立了一个包含23个全面和严格的度量的模型健壮性评估框架，该框架考虑了对抗性学习的两个关键角度(即数据和模型)。通过神经元覆盖和数据不可感知性，我们使用面向数据的度量来度量测试用例的完整性；通过深入研究模型结构和行为，我们利用面向模型的度量来进一步评估对抗环境下的健壮性。为了充分展示该框架的有效性，我们在CIFAR-10、SVHN和ImageNet等多个数据集上使用不同的模型和防御机制在我们的开源平台上进行了大规模的实验。总体而言，本文提供了一个全面的评估框架，研究人员可以使用开源工具包进行全面而快速的评估，分析结果可以启发对模型稳健性的更深入理解和进一步改进。



## **4. The Perils of Learning From Unlabeled Data: Backdoor Attacks on Semi-supervised Learning**

从未标记数据中学习的危险：对半监督学习的后门攻击 cs.CR

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00453v1) [paper-pdf](http://arxiv.org/pdf/2211.00453v1)

**Authors**: Virat Shejwalkar, Lingjuan Lyu, Amir Houmansadr

**Abstract**: Semi-supervised machine learning (SSL) is gaining popularity as it reduces the cost of training ML models. It does so by using very small amounts of (expensive, well-inspected) labeled data and large amounts of (cheap, non-inspected) unlabeled data. SSL has shown comparable or even superior performances compared to conventional fully-supervised ML techniques.   In this paper, we show that the key feature of SSL that it can learn from (non-inspected) unlabeled data exposes SSL to strong poisoning attacks. In fact, we argue that, due to its reliance on non-inspected unlabeled data, poisoning is a much more severe problem in SSL than in conventional fully-supervised ML.   Specifically, we design a backdoor poisoning attack on SSL that can be conducted by a weak adversary with no knowledge of target SSL pipeline. This is unlike prior poisoning attacks in fully-supervised settings that assume strong adversaries with practically-unrealistic capabilities. We show that by poisoning only 0.2% of the unlabeled training data, our attack can cause misclassification of more than 80% of test inputs (when they contain the adversary's backdoor trigger). Our attacks remain effective across twenty combinations of benchmark datasets and SSL algorithms, and even circumvent the state-of-the-art defenses against backdoor attacks. Our work raises significant concerns about the practical utility of existing SSL algorithms.

摘要: 半监督机器学习(SSL)由于降低了训练ML模型的成本而越来越受欢迎。它通过使用少量(昂贵的、经过良好检查的)标记数据和大量(廉价的、未检查的)未标记数据来做到这一点。与传统的全监督ML技术相比，SSL表现出了相当甚至更好的性能。在本文中，我们证明了SSL的关键特征是它可以从(未检查的)未标记数据中学习，从而使SSL暴露在强大的中毒攻击之下。事实上，我们认为，由于它依赖于未检查的未标记数据，所以中毒在SSL中比在传统的完全监督的ML中要严重得多。具体地说，我们设计了一种针对SSL的后门中毒攻击，该攻击可以由对目标SSL管道一无所知的弱攻击者进行。这不同于之前在完全监督的环境中进行的投毒攻击，后者假设强大的对手具有几乎不切实际的能力。我们表明，通过只毒化0.2%的未标记训练数据，我们的攻击可以导致超过80%的测试输入的错误分类(当它们包含对手的后门触发器时)。我们的攻击在基准数据集和SSL算法的20种组合中仍然有效，甚至绕过了针对后门攻击的最先进防御。我们的工作引起了人们对现有SSL算法的实用价值的极大关注。



## **5. Relative Attention-based One-Class Adversarial Autoencoder for Continuous Authentication of Smartphone Users**

基于相对关注度的智能手机用户连续认证一类对抗性自动编码器 cs.HC

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2210.16819v2) [paper-pdf](http://arxiv.org/pdf/2210.16819v2)

**Authors**: Mingming Hu, Kun Zhang, Ruibang You, Bibo Tu

**Abstract**: Behavioral biometrics-based continuous authentication is a promising authentication scheme, which uses behavioral biometrics recorded by built-in sensors to authenticate smartphone users throughout the session. However, current continuous authentication methods suffer some limitations: 1) behavioral biometrics from impostors are needed to train continuous authentication models. Since the distribution of negative samples from diverse attackers are unknown, it is a difficult problem to solve in real-world scenarios; 2) most deep learning-based continuous authentication methods need to train two models to improve authentication performance. A deep learning model for deep feature extraction, and a machine learning-based classifier for classification; 3) weak capability of capturing users' behavioral patterns leads to poor authentication performance. To solve these issues, we propose a relative attention-based one-class adversarial autoencoder for continuous authentication of smartphone users. First, we propose a one-class adversarial autoencoder to learn latent representations of legitimate users' behavioral patterns, which is trained only with legitimate smartphone users' behavioral biometrics. Second, we present the relative attention layer to capture richer contextual semantic representation of users' behavioral patterns, which modifies the standard self-attention mechanism using convolution projection instead of linear projection to perform the attention maps. Experimental results demonstrate that we can achieve superior performance of 1.05% EER, 1.09% EER, and 1.08% EER with a high authentication frequency (0.7s) on three public datasets.

摘要: 基于行为生物识别的连续认证是一种很有前途的认证方案，它使用内置传感器记录的行为生物识别在整个会话过程中对智能手机用户进行认证。然而，当前的连续认证方法存在一些局限性：1)需要来自冒名顶替者的行为生物特征来训练连续认证模型。由于来自不同攻击者的负样本分布未知，这在现实场景中是一个难以解决的问题；2)大多数基于深度学习的连续认证方法需要训练两个模型来提高认证性能。深度学习模型用于深度特征提取，基于机器学习的分类器用于分类；3)对用户行为模式的捕获能力较弱，导致认证性能较差。为了解决这些问题，我们提出了一种基于相对关注度的一类对抗性自动编码器，用于智能手机用户的持续认证。首先，我们提出了一种一类对抗性自动编码器来学习合法用户行为模式的潜在表征，该编码器只使用合法智能手机用户的行为生物特征进行训练。其次，我们提出了相对关注层，以获取更丰富的用户行为模式的上下文语义表示，改进了标准的自我注意机制，使用卷积投影而不是线性投影来执行注意图。实验结果表明，在较高的认证频率(0.7s)下，该算法在三个公共数据集上分别获得了1.05%、1.09%和1.08%的性能。



## **6. Universal Perturbation Attack on Differentiable No-Reference Image- and Video-Quality Metrics**

对可区分的无参考图像和视频质量度量的通用扰动攻击 cs.CV

BMVC 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00366v1) [paper-pdf](http://arxiv.org/pdf/2211.00366v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Universal adversarial perturbation attacks are widely used to analyze image classifiers that employ convolutional neural networks. Nowadays, some attacks can deceive image- and video-quality metrics. So sustainability analysis of these metrics is important. Indeed, if an attack can confuse the metric, an attacker can easily increase quality scores. When developers of image- and video-algorithms can boost their scores through detached processing, algorithm comparisons are no longer fair. Inspired by the idea of universal adversarial perturbation for classifiers, we suggest a new method to attack differentiable no-reference quality metrics through universal perturbation. We applied this method to seven no-reference image- and video-quality metrics (PaQ-2-PiQ, Linearity, VSFA, MDTVSFA, KonCept512, Nima and SPAQ). For each one, we trained a universal perturbation that increases the respective scores. We also propose a method for assessing metric stability and identify the metrics that are the most vulnerable and the most resistant to our attack. The existence of successful universal perturbations appears to diminish the metric's ability to provide reliable scores. We therefore recommend our proposed method as an additional verification of metric reliability to complement traditional subjective tests and benchmarks.

摘要: 通用对抗性扰动攻击被广泛用于分析使用卷积神经网络的图像分类器。如今，一些攻击可以欺骗图像和视频质量指标。因此，对这些指标的可持续性分析非常重要。事实上，如果攻击可以混淆度量，攻击者可以很容易地提高质量分数。当图像和视频算法的开发者可以通过独立的处理来提高他们的分数时，算法比较不再公平。受分类器泛化对抗性扰动思想的启发，提出了一种利用泛化扰动攻击可微无参考质量度量的新方法。我们将该方法应用于七个无参考图像和视频质量指标(PaQ-2-PiQ、线性度、VSFA、MDTVSFA、KonCept512、NIMA和SpaQ)。对于每一个，我们都训练了一个普遍的扰动，以增加各自的分数。我们还提出了一种评估指标稳定性的方法，并确定了最易受攻击和最能抵抗攻击的指标。成功的普遍扰动的存在似乎削弱了该指标提供可靠分数的能力。因此，我们建议将我们提出的方法作为度量可靠性的额外验证，以补充传统的主观测试和基准。



## **7. FRSUM: Towards Faithful Abstractive Summarization via Enhancing Factual Robustness**

FRSUM：通过增强事实稳健性实现真实的摘要摘要 cs.CL

Findings of EMNLP 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00294v1) [paper-pdf](http://arxiv.org/pdf/2211.00294v1)

**Authors**: Wenhao Wu, Wei Li, Jiachen Liu, Xinyan Xiao, Ziqiang Cao, Sujian Li, Hua Wu

**Abstract**: Despite being able to generate fluent and grammatical text, current Seq2Seq summarization models still suffering from the unfaithful generation problem. In this paper, we study the faithfulness of existing systems from a new perspective of factual robustness which is the ability to correctly generate factual information over adversarial unfaithful information. We first measure a model's factual robustness by its success rate to defend against adversarial attacks when generating factual information. The factual robustness analysis on a wide range of current systems shows its good consistency with human judgments on faithfulness. Inspired by these findings, we propose to improve the faithfulness of a model by enhancing its factual robustness. Specifically, we propose a novel training strategy, namely FRSUM, which teaches the model to defend against both explicit adversarial samples and implicit factual adversarial perturbations. Extensive automatic and human evaluation results show that FRSUM consistently improves the faithfulness of various Seq2Seq models, such as T5, BART.

摘要: 尽管能够生成流畅且符合语法的文本，但当前的Seq2Seq摘要模型仍然存在不忠实生成问题。本文从事实稳健性这一新的角度来研究现有系统的忠实性，即在敌意失信信息的基础上正确生成事实信息的能力。我们首先通过模型的成功率来衡量模型的事实稳健性，以便在生成事实信息时防御对手攻击。对大量现有系统的事实稳健性分析表明，该算法与人类对信任度的判断具有良好的一致性。受这些发现的启发，我们建议通过增强模型的事实稳健性来提高模型的忠实性。具体地说，我们提出了一种新的训练策略，即FRSUM，它教导该模型同时防御显式敌意样本和隐式事实对抗性扰动。广泛的自动和人工评估结果表明，FRSUM一致地提高了各种Seq2Seq模型的忠实性，如T5，BART。



## **8. Adversarial Training with Complementary Labels: On the Benefit of Gradually Informative Attacks**

标签互补的对抗性训练：关于渐进式信息攻击的好处 cs.LG

NeurIPS 2022

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00269v1) [paper-pdf](http://arxiv.org/pdf/2211.00269v1)

**Authors**: Jianan Zhou, Jianing Zhu, Jingfeng Zhang, Tongliang Liu, Gang Niu, Bo Han, Masashi Sugiyama

**Abstract**: Adversarial training (AT) with imperfect supervision is significant but receives limited attention. To push AT towards more practical scenarios, we explore a brand new yet challenging setting, i.e., AT with complementary labels (CLs), which specify a class that a data sample does not belong to. However, the direct combination of AT with existing methods for CLs results in consistent failure, but not on a simple baseline of two-stage training. In this paper, we further explore the phenomenon and identify the underlying challenges of AT with CLs as intractable adversarial optimization and low-quality adversarial examples. To address the above problems, we propose a new learning strategy using gradually informative attacks, which consists of two critical components: 1) Warm-up Attack (Warm-up) gently raises the adversarial perturbation budgets to ease the adversarial optimization with CLs; 2) Pseudo-Label Attack (PLA) incorporates the progressively informative model predictions into a corrected complementary loss. Extensive experiments are conducted to demonstrate the effectiveness of our method on a range of benchmarked datasets. The code is publicly available at: https://github.com/RoyalSkye/ATCL.

摘要: 监督不完善的对抗性训练(AT)意义重大，但受到的关注有限。为了将AT推向更实际的场景，我们探索了一个全新但具有挑战性的设置，即具有互补标签(CLS)的AT，它指定了数据样本不属于的类。然而，AT与现有CLS方法的直接结合导致了一致的失败，但不是在两阶段训练的简单基线上。在这篇文章中，我们进一步探讨了这一现象，并确定了具有CLS的AT的潜在挑战，作为棘手的对抗性优化和低质量的对抗性例子。针对上述问题，我们提出了一种基于渐进式信息攻击的学习策略，该策略由两个关键部分组成：1)预热攻击(预热)温和地提高对抗性扰动预算，以简化CLS的对抗性优化；2)伪标签攻击将渐进式信息模型预测合并到校正的互补损失中。在一系列基准数据集上进行了广泛的实验，以证明我们的方法的有效性。该代码可在以下网址公开获得：https://github.com/RoyalSkye/ATCL.



## **9. Adversarial Policies Beat Professional-Level Go AIs**

对抗性政策击败专业级围棋人工智能 cs.LG

21 pages, 11 figures

**SubmitDate**: 2022-11-01    [abs](http://arxiv.org/abs/2211.00241v1) [paper-pdf](http://arxiv.org/pdf/2211.00241v1)

**Authors**: Tony Tong Wang, Adam Gleave, Nora Belrose, Tom Tseng, Joseph Miller, Michael D Dennis, Yawen Duan, Viktor Pogrebniak, Sergey Levine, Stuart Russell

**Abstract**: We attack the state-of-the-art Go-playing AI system, KataGo, by training an adversarial policy that plays against a frozen KataGo victim. Our attack achieves a >99% win-rate against KataGo without search, and a >50% win-rate when KataGo uses enough search to be near-superhuman. To the best of our knowledge, this is the first successful end-to-end attack against a Go AI playing at the level of a top human professional. Notably, the adversary does not win by learning to play Go better than KataGo -- in fact, the adversary is easily beaten by human amateurs. Instead, the adversary wins by tricking KataGo into ending the game prematurely at a point that is favorable to the adversary. Our results demonstrate that even professional-level AI systems may harbor surprising failure modes. See https://goattack.alignmentfund.org/ for example games.

摘要: 我们通过训练对抗冻结的KataGo受害者的对抗性策略来攻击最先进的围棋人工智能系统KataGo。我们的攻击在没有搜索的情况下对KataGo的胜率超过99%，当KataGo使用足够的搜索近乎超人时，胜率>50%。据我们所知，这是第一次成功地对围棋人工智能进行顶尖人类职业水平的端到端攻击。值得注意的是，对手并不是通过学习比KataGo下得更好来取胜的--事实上，对手很容易被人类业余爱好者击败。取而代之的是，对手通过欺骗KataGo在对对手有利的点上提前结束游戏而获胜。我们的结果表明，即使是专业级的人工智能系统也可能存在令人惊讶的故障模式。有关游戏示例，请参阅https://goattack.alignmentfund.org/。



## **10. Synthetic ID Card Image Generation for Improving Presentation Attack Detection**

用于改进呈现攻击检测的合成身份证图像生成 cs.CV

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2211.00098v1) [paper-pdf](http://arxiv.org/pdf/2211.00098v1)

**Authors**: Daniel Benalcazar, Juan E. Tapia, Sebastian Gonzalez, Christoph Busch

**Abstract**: Currently, it is ever more common to access online services for activities which formerly required physical attendance. From banking operations to visa applications, a significant number of processes have been digitised, especially since the advent of the COVID-19 pandemic, requiring remote biometric authentication of the user. On the downside, some subjects intend to interfere with the normal operation of remote systems for personal profit by using fake identity documents, such as passports and ID cards. Deep learning solutions to detect such frauds have been presented in the literature. However, due to privacy concerns and the sensitive nature of personal identity documents, developing a dataset with the necessary number of examples for training deep neural networks is challenging. This work explores three methods for synthetically generating ID card images to increase the amount of data while training fraud-detection networks. These methods include computer vision algorithms and Generative Adversarial Networks. Our results indicate that databases can be supplemented with synthetic images without any loss in performance for the print/scan Presentation Attack Instrument Species (PAIS) and a loss in performance of 1% for the screen capture PAIS.

摘要: 目前，为以前需要亲自参加的活动访问在线服务比以往任何时候都更常见。从银行业务到签证申请，大量流程都已实现数字化，尤其是在新冠肺炎疫情出现以来，这要求对用户进行远程生物识别认证。不利的一面是，一些主体意图通过使用护照、身份证等假身份证件，干扰远程系统的正常运行，以谋取个人利益。在文献中已经提出了用于检测此类欺诈的深度学习解决方案。然而，由于隐私问题和个人身份文件的敏感性质，开发一个具有必要数量的样本来训练深度神经网络的数据集是具有挑战性的。这项工作探索了三种综合生成身份证图像的方法，以在训练欺诈检测网络的同时增加数据量。这些方法包括计算机视觉算法和生成性对手网络。我们的结果表明，可以用合成图像来补充数据库，而打印/扫描呈现攻击工具种类(PAI)的性能不会有任何损失，屏幕捕获PAI的性能不会损失1%。



## **11. Lessons Learned: How (Not) to Defend Against Property Inference Attacks**

经验教训：如何(不)防御属性推断攻击 cs.CR

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2205.08821v3) [paper-pdf](http://arxiv.org/pdf/2205.08821v3)

**Authors**: Joshua Stock, Jens Wettlaufer, Daniel Demmler, Hannes Federrath

**Abstract**: This work investigates and evaluates multiple defense strategies against property inference attacks (PIAs), a privacy attack against machine learning models. Given a trained machine learning model, PIAs aim to extract statistical properties of its underlying training data, e.g., reveal the ratio of men and women in a medical training data set. While for other privacy attacks like membership inference, a lot of research on defense mechanisms has been published, this is the first work focusing on defending against PIAs. With the primary goal of developing a generic mitigation strategy against white-box PIAs, we propose the novel approach property unlearning. Extensive experiments with property unlearning show that while it is very effective when defending target models against specific adversaries, property unlearning is not able to generalize, i.e., protect against a whole class of PIAs. To investigate the reasons behind this limitation, we present the results of experiments with the explainable AI tool LIME. They show how state-of-the-art property inference adversaries with the same objective focus on different parts of the target model. We further elaborate on this with a follow-up experiment, in which we use the visualization technique t-SNE to exhibit how severely statistical training data properties are manifested in machine learning models. Based on this, we develop the conjecture that post-training techniques like property unlearning might not suffice to provide the desirable generic protection against PIAs. As an alternative, we investigate the effects of simpler training data preprocessing methods like adding Gaussian noise to images of a training data set on the success rate of PIAs. We conclude with a discussion of the different defense approaches, summarize the lessons learned and provide directions for future work.

摘要: 本文研究和评估了针对机器学习模型的隐私攻击--属性推理攻击(PIA)的多种防御策略。给定一个经过训练的机器学习模型，PIA的目标是提取其基本训练数据的统计属性，例如，揭示医学训练数据集中的男性和女性的比例。虽然对于其他隐私攻击，如成员身份推断，已经发表了大量关于防御机制的研究，但这是第一个专注于防御PIA的工作。以开发一种针对白盒PIA的通用缓解策略为主要目标，我们提出了一种新的方法-属性遗忘。大量的属性遗忘实验表明，虽然它在防御特定对手的目标模型时非常有效，但属性遗忘不能泛化，即保护一整类PIA。为了调查这种限制背后的原因，我们给出了使用可解释的人工智能工具LIME的实验结果。它们展示了具有相同目标的最先进的属性推理对手如何专注于目标模型的不同部分。我们通过后续实验进一步阐述了这一点，在该实验中，我们使用可视化技术t-SNE来展示统计训练数据属性在机器学习模型中的表现是多么严重。在此基础上，我们提出了这样的猜测，即训练后的技术，如属性遗忘，可能不足以提供理想的通用保护，以防止PIA。作为另一种选择，我们研究了更简单的训练数据预处理方法，如向训练数据集的图像添加高斯噪声对PIA成功率的影响。最后，我们讨论了不同的防御方法，总结了经验教训，并为未来的工作提供了方向。



## **12. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

Sardino：移动边缘安全视觉感知的超快动态合奏 cs.CV

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2204.08189v4) [paper-pdf](http://arxiv.org/pdf/2204.08189v4)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstract**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.

摘要: 对抗性示例攻击危及采用深度神经网络进行视觉传感的移动边缘系统，如车辆和无人机。提出了一种主动的、动态的防御方法{em Sardino}，该方法在运行时更新推理集成，以提高安全性，防止自适应对手试图渗透集成并构造相应的有效对抗实例。通过对合奏的预测应用一致性检查和数据融合，萨迪诺可以检测和挫败敌方的输入。与基于训练的集成更新相比，我们使用HyperNet实现了加速和每帧集成更新，这对先决条件渗透攻击呈现出最高的难度。我们设计了一个运行时规划器，在保持处理帧速率的同时最大化集成大小以利于安全性。除了敌对的例子，萨迪诺还可以有效地解决分配外投入的问题。本文对Sardino在对抗敌意例子方面的表现进行了广泛的评估，并将其应用于构建一个实时车载交通标志识别系统。现场道路测试表明，所建立的系统在保持帧速率和检测由于先前基于YOLO的交通标志检测器的错误阳性而导致的不分布输入方面是有效的。



## **13. Symmetries, flat minima, and the conserved quantities of gradient flow**

对称性、平坦极小值和梯度流的守恒量 cs.LG

Preliminary version; comments welcome

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17216v1) [paper-pdf](http://arxiv.org/pdf/2210.17216v1)

**Authors**: Bo Zhao, Iordan Ganev, Robin Walters, Rose Yu, Nima Dehmamy

**Abstract**: Empirical studies of the loss landscape of deep networks have revealed that many local minima are connected through low-loss valleys. Ensemble models sampling different parts of a low-loss valley have reached SOTA performance. Yet, little is known about the theoretical origin of such valleys. We present a general framework for finding continuous symmetries in the parameter space, which carve out low-loss valleys. Importantly, we introduce a novel set of nonlinear, data-dependent symmetries for neural networks. These symmetries can transform a trained model such that it performs similarly on new samples. We then show that conserved quantities associated with linear symmetries can be used to define coordinates along low-loss valleys. The conserved quantities help reveal that using common initialization methods, gradient flow only explores a small part of the global minimum. By relating conserved quantities to convergence rate and sharpness of the minimum, we provide insights on how initialization impacts convergence and generalizability. We also find the nonlinear action to be viable for ensemble building to improve robustness under certain adversarial attacks.

摘要: 对深层网络损失格局的实证研究表明，许多局部极小值通过低损失谷连接在一起。在一个低损失山谷的不同部分采样的整体模型已经达到了SOTA的表现。然而，人们对这种山谷的理论起源知之甚少。我们给出了一个在参数空间中寻找连续对称性的一般框架，它划出了低损失谷。重要的是，我们为神经网络引入了一组新的非线性、依赖于数据的对称性。这些对称性可以变换训练过的模型，使其在新样本上执行类似的操作。然后，我们证明了与线性对称有关的守恒量可以用来定义沿低损耗山谷的坐标。守恒量有助于揭示，使用常见的初始化方法，梯度流只探索全局极小值的一小部分。通过将守恒量与最小值的收敛速度和锐度联系起来，我们提供了关于初始化如何影响收敛和泛化的见解。我们还发现，非线性行为对于集成构建是可行的，以提高在某些对抗性攻击下的稳健性。



## **14. Defending Against Adversarial Attacks by Energy Storage Facility**

利用储能设施防御敌意攻击 cs.CR

5 pages, 5 main figures. Published in PESGM 2022

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2205.09522v2) [paper-pdf](http://arxiv.org/pdf/2205.09522v2)

**Authors**: Jiawei Li, Jianxiao Wang, Lin Chen, Yang Yu

**Abstract**: Adversarial attacks on data-driven algorithms applied in the power system will be a new type of threat to grid security. Literature has demonstrated that the adversarial attack on the deep-neural network can significantly mislead the load fore-cast of a power system. However, it is unclear how the new type of attack impacts the operation of the grid system. In this research, we manifest that the adversarial algorithm attack induces a significant cost-increase risk which will be exacerbated by the growing penetration of intermittent renewable energy. In Texas, a 5% adversarial attack can increase the total generation cost by 17% in a quarter, which accounts for around $20 million. When wind-energy penetration increases to over 40%, the 5% adversarial attack will inflate the genera-tion cost by 23%. Our research discovers a novel approach to defending against the adversarial attack: investing in the energy-storage system. All current literature focuses on developing algorithms to defend against adversarial attacks. We are the first research revealing the capability of using the facility in a physical system to defend against the adversarial algorithm attack in a system of the Internet of Things, such as a smart grid system.

摘要: 针对电力系统中应用的数据驱动算法的对抗性攻击将是一种新型的电网安全威胁。已有文献表明，对深度神经网络的敌意攻击会严重误导电力系统的负荷预测。然而，目前尚不清楚这种新型攻击对电网系统的运行有何影响。在这项研究中，我们证明了对抗性算法攻击导致了显著的成本增加风险，这种风险将随着间歇性可再生能源的日益普及而加剧。在德克萨斯州，5%的对抗性攻击可以在一个季度内使发电总成本增加17%，约为2000万美元。当风能渗透率增加到40%以上时，5%的对抗性攻击将使发电成本膨胀23%。我们的研究发现了一种防御对手攻击的新方法：投资于储能系统。目前的所有文献都集中在开发算法来防御对手攻击。我们是第一个揭示在物理系统中使用该设施来防御物联网系统(如智能电网系统)中的对抗性算法攻击的能力的研究。



## **15. Scoring Black-Box Models for Adversarial Robustness**

对抗性稳健性的评分黑箱模型 cs.LG

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17140v1) [paper-pdf](http://arxiv.org/pdf/2210.17140v1)

**Authors**: Jian Vora, Pranay Reddy Samala

**Abstract**: Deep neural networks are susceptible to adversarial inputs and various methods have been proposed to defend these models against adversarial attacks under different perturbation models. The robustness of models to adversarial attacks has been analyzed by first constructing adversarial inputs for the model, and then testing the model performance on the constructed adversarial inputs. Most of these attacks require the model to be white-box, need access to data labels, and finding adversarial inputs can be computationally expensive. We propose a simple scoring method for black-box models which indicates their robustness to adversarial input. We show that adversarially more robust models have a smaller $l_1$-norm of LIME weights and sharper explanations.

摘要: 深度神经网络对敌意输入很敏感，在不同的扰动模型下，人们已经提出了各种方法来防御这些模型的敌意攻击。通过首先为模型构建对抗性输入，然后在构建的对抗性输入上测试模型的性能，分析了模型对对抗性攻击的稳健性。这些攻击中的大多数都要求模型是白盒的，需要访问数据标签，并且寻找敌对输入的计算成本可能很高。我们对黑盒模型提出了一种简单的评分方法，表明了其对敌意输入的稳健性。相反，我们证明了更稳健的模型具有更小的LIME权重的$l_1$-范数和更清晰的解释。



## **16. Character-level White-Box Adversarial Attacks against Transformers via Attachable Subwords Substitution**

基于可附加子词替换的字符级白盒对抗变形金刚攻击 cs.CL

13 pages, 3 figures. EMNLP 2022

**SubmitDate**: 2022-10-31    [abs](http://arxiv.org/abs/2210.17004v1) [paper-pdf](http://arxiv.org/pdf/2210.17004v1)

**Authors**: Aiwei Liu, Honghai Yu, Xuming Hu, Shu'ang Li, Li Lin, Fukun Ma, Yawen Yang, Lijie Wen

**Abstract**: We propose the first character-level white-box adversarial attack method against transformer models. The intuition of our method comes from the observation that words are split into subtokens before being fed into the transformer models and the substitution between two close subtokens has a similar effect to the character modification. Our method mainly contains three steps. First, a gradient-based method is adopted to find the most vulnerable words in the sentence. Then we split the selected words into subtokens to replace the origin tokenization result from the transformer tokenizer. Finally, we utilize an adversarial loss to guide the substitution of attachable subtokens in which the Gumbel-softmax trick is introduced to ensure gradient propagation. Meanwhile, we introduce the visual and length constraint in the optimization process to achieve minimum character modifications. Extensive experiments on both sentence-level and token-level tasks demonstrate that our method could outperform the previous attack methods in terms of success rate and edit distance. Furthermore, human evaluation verifies our adversarial examples could preserve their origin labels.

摘要: 提出了第一种针对变压器模型的特征级白盒对抗攻击方法。我们方法的直觉来自于观察到单词在被馈入转换器模型之前被拆分成子标记，并且两个接近的子标记之间的替换具有类似于字符修改的效果。我们的方法主要包括三个步骤。首先，采用基于梯度的方法找出句子中最脆弱的词。然后，我们将所选择的词拆分为子标记词，以替换来自变换标记器的原始标记化结果。最后，我们利用对抗性损失来指导可附加子令牌的替换，其中引入了Gumbel-Softmax技巧来确保梯度传播。同时，我们在优化过程中引入了视觉和长度约束，以达到最小的字符修改。在句子级和令牌级任务上的大量实验表明，我们的方法在成功率和编辑距离方面都优于以往的攻击方法。此外，人类评估验证了我们的对抗性例子可以保留它们的来源标签。



## **17. A Comparative Study of Adversarial Attacks against Point Cloud Semantic Segmentation**

点云语义分割对抗性攻击的比较研究 cs.CV

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2112.05871v3) [paper-pdf](http://arxiv.org/pdf/2112.05871v3)

**Authors**: Jiacen Xu, Zhe Zhou, Boyuan Feng, Yufei Ding, Zhou Li

**Abstract**: Recent research efforts on 3D point cloud semantic segmentation (PCSS) have achieved outstanding performance by adopting deep CNN (convolutional neural networks) and GCN (graph convolutional networks). However, the robustness of these complex models has not been systematically analyzed. Given that PCSS has been applied in many safety-critical applications (e.g., autonomous driving, geological sensing), it is important to fill this knowledge gap, in particular, how these models are affected under adversarial samples. While adversarial attacks against point clouds have been studied, we found many questions remain about the robustness of PCSS. For instance, all the prior attacks perturb the point coordinates of a point cloud, but the features associated with a point are also leveraged by some PCSS models, and whether they are good targets to attack is unknown yet.   We present a comparative study of PCSS robustness in this work. In particular, we formally define the attacker's objective under targeted attack and non-targeted attack and develop new attacks considering a variety of options, including feature-based and coordinate-based, norm-bounded and norm-unbounded, etc. We conduct evaluations with different combinations of attack options on two datasets (S3DIS and Semantic3D) and three PCSS models (PointNet++, DeepGCNs, and RandLA-Net). We found all of the PCSS models are vulnerable under both targeted and non-targeted attacks, and attacks against point features like color are more effective. With this study, we call the attention of the research community to develop new approaches to harden PCSS models against adversarial attacks.

摘要: 近年来，基于深度卷积神经网络和图形卷积网络的三维点云语义分割方法取得了较好的效果。然而，这些复杂模型的稳健性还没有得到系统的分析。鉴于PCSS已经应用于许多安全关键应用(例如，自动驾驶、地质传感)，填补这一知识空白是很重要的，特别是这些模型在敌方样本下是如何受到影响的。虽然针对点云的敌意攻击已经被研究，但我们发现关于PCSS的健壮性仍然存在许多问题。例如，所有先前的攻击都会扰乱点云的点坐标，但与点相关的特征也被一些PCSS模型利用，它们是否是好的攻击目标尚不清楚。在这项工作中，我们对PCSS的稳健性进行了比较研究。特别是，我们形式化地定义了定向攻击和非定向攻击下攻击者的目标，并考虑了基于特征和基于坐标、范数有界和范数无界等多种选项来开发新的攻击。我们在两个数据集(S3DIS和Semanc3D)和三个PCSS模型(PointNet++、DeepGCNs和RandLA-Net)上进行了不同组合的攻击评估。我们发现所有的PCSS模型在目标攻击和非目标攻击下都是脆弱的，而针对颜色等点特征的攻击更有效。通过这项研究，我们呼吁研究界关注开发新的方法来加强PCSS模型对对手攻击的攻击。



## **18. Symmetric Saliency-based Adversarial Attack To Speaker Identification**

基于对称显著性的说话人识别对抗性攻击 cs.SD

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2210.16777v1) [paper-pdf](http://arxiv.org/pdf/2210.16777v1)

**Authors**: Jiadi Yao, Xing Chen, Xiao-Lei Zhang, Wei-Qiang Zhang, Kunde Yang

**Abstract**: Adversarial attack approaches to speaker identification either need high computational cost or are not very effective, to our knowledge. To address this issue, in this paper, we propose a novel generation-network-based approach, called symmetric saliency-based encoder-decoder (SSED), to generate adversarial voice examples to speaker identification. It contains two novel components. First, it uses a novel saliency map decoder to learn the importance of speech samples to the decision of a targeted speaker identification system, so as to make the attacker focus on generating artificial noise to the important samples. It also proposes an angular loss function to push the speaker embedding far away from the source speaker. Our experimental results demonstrate that the proposed SSED yields the state-of-the-art performance, i.e. over 97% targeted attack success rate and a signal-to-noise level of over 39 dB on both the open-set and close-set speaker identification tasks, with a low computational cost.

摘要: 据我们所知，对抗性攻击的说话人识别方法要么需要很高的计算代价，要么不是很有效。针对这一问题，本文提出了一种新的基于生成网络的方法，称为基于对称显著度的编解码器(SSED)，用于生成用于说话人识别的对抗性语音样本。它包含两个新颖的组成部分。首先，使用一种新颖的显著图解码器来学习语音样本对目标说话人识别系统决策的重要性，从而使攻击者专注于对重要样本产生人工噪声。提出了一种角度损失函数，将嵌入的说话人推到远离源说话人的位置。实验结果表明，该算法具有最好的性能，在开集和闭集说话人辨认任务中的目标攻击成功率均在97%以上，信噪比均在39dB以上，且计算量较小。



## **19. Benchmarking Adversarial Patch Against Aerial Detection**

针对空中探测的对抗性补丁基准测试 cs.CV

14 pages, 14 figures

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2210.16765v1) [paper-pdf](http://arxiv.org/pdf/2210.16765v1)

**Authors**: Jiawei Lian, Shaohui Mei, Shun Zhang, Mingyang Ma

**Abstract**: DNNs are vulnerable to adversarial examples, which poses great security concerns for security-critical systems. In this paper, a novel adaptive-patch-based physical attack (AP-PA) framework is proposed, which aims to generate adversarial patches that are adaptive in both physical dynamics and varying scales, and by which the particular targets can be hidden from being detected. Furthermore, the adversarial patch is also gifted with attack effectiveness against all targets of the same class with a patch outside the target (No need to smear targeted objects) and robust enough in the physical world. In addition, a new loss is devised to consider more available information of detected objects to optimize the adversarial patch, which can significantly improve the patch's attack efficacy (Average precision drop up to 87.86% and 85.48% in white-box and black-box settings, respectively) and optimizing efficiency. We also establish one of the first comprehensive, coherent, and rigorous benchmarks to evaluate the attack efficacy of adversarial patches on aerial detection tasks. Finally, several proportionally scaled experiments are performed physically to demonstrate that the elaborated adversarial patches can successfully deceive aerial detection algorithms in dynamic physical circumstances. The code is available at https://github.com/JiaweiLian/AP-PA.

摘要: DNN很容易受到敌意例子的攻击，这给安全关键系统带来了极大的安全隐患。提出了一种新的基于自适应补丁的物理攻击框架(AP-PA)，该框架旨在生成在物理动态和不同尺度上都是自适应的敌意补丁，并通过该补丁隐藏特定目标而不被检测到。此外，敌方补丁也被赋予了对同一类别的所有目标的攻击效果，在目标之外有一个补丁(不需要抹黑目标对象)，并且在物理世界中足够强大。此外，设计了一种新的损失来考虑更多的检测对象的可用信息来优化对抗性补丁，可以显著提高补丁的攻击效率(白盒和黑盒环境下的平均精度分别下降87.86%和85.48%)和优化效率。我们还建立了第一个全面、连贯和严格的基准之一，以评估空中探测任务中对抗性补丁的攻击效能。最后，进行了几个比例比例的物理实验，证明了所设计的敌方补丁能够在动态物理环境中成功地欺骗空中探测算法。代码可在https://github.com/JiaweiLian/AP-PA.上获得



## **20. RUSH: Robust Contrastive Learning via Randomized Smoothing**

RASH：随机平滑的稳健对比学习 cs.LG

incomplete validation, the defense strategy will fail when  considering Expectation Over Test (EOT)

**SubmitDate**: 2022-10-30    [abs](http://arxiv.org/abs/2207.05127v2) [paper-pdf](http://arxiv.org/pdf/2207.05127v2)

**Authors**: Yijiang Pang, Boyang Liu, Jiayu Zhou

**Abstract**: Recently, adversarial training has been incorporated in self-supervised contrastive pre-training to augment label efficiency with exciting adversarial robustness. However, the robustness came at a cost of expensive adversarial training. In this paper, we show a surprising fact that contrastive pre-training has an interesting yet implicit connection with robustness, and such natural robustness in the pre trained representation enables us to design a powerful robust algorithm against adversarial attacks, RUSH, that combines the standard contrastive pre-training and randomized smoothing. It boosts both standard accuracy and robust accuracy, and significantly reduces training costs as compared with adversarial training. We use extensive empirical studies to show that the proposed RUSH outperforms robust classifiers from adversarial training, by a significant margin on common benchmarks (CIFAR-10, CIFAR-100, and STL-10) under first-order attacks. In particular, under $\ell_{\infty}$-norm perturbations of size 8/255 PGD attack on CIFAR-10, our model using ResNet-18 as backbone reached 77.8% robust accuracy and 87.9% standard accuracy. Our work has an improvement of over 15% in robust accuracy and a slight improvement in standard accuracy, compared to the state-of-the-arts.

摘要: 最近，对抗性训练被结合到自我监督的对比预训练中，以增强标记的效率和令人兴奋的对抗性健壮性。然而，这种健壮性是以昂贵的对抗性训练为代价的。在这篇文章中，我们展示了一个令人惊讶的事实，即对比预训练与稳健性有着有趣而隐含的联系，而预训练表示中的这种自然的稳健性使我们能够设计出一种结合了标准的对比预训练和随机平滑的强大的抗对手攻击的鲁棒算法RASH。与对抗性训练相比，它同时提高了标准准确率和稳健准确率，并显著降低了训练成本。我们使用广泛的实证研究表明，在一阶攻击下，所提出的冲刺算法在常见基准(CIFAR-10、CIFAR-100和STL-10)上的性能明显优于来自对手训练的稳健分类器。特别是，在CIFAR-10遭受8/255 PGD攻击时，以ResNet-18为主干的模型达到了77.8%的稳健准确率和87.9%的标准准确率。与最新水平相比，我们的工作在稳健精度上提高了15%以上，在标准精度上略有提高。



## **21. BERTops: Studying BERT Representations under a Topological Lens**

BERTOPS：研究拓扑透镜下的BERT表示 cs.LG

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2205.00953v2) [paper-pdf](http://arxiv.org/pdf/2205.00953v2)

**Authors**: Jatin Chauhan, Manohar Kaul

**Abstract**: Proposing scoring functions to effectively understand, analyze and learn various properties of high dimensional hidden representations of large-scale transformer models like BERT can be a challenging task. In this work, we explore a new direction by studying the topological features of BERT hidden representations using persistent homology (PH). We propose a novel scoring function named "persistence scoring function (PSF)" which: (i) accurately captures the homology of the high-dimensional hidden representations and correlates well with the test set accuracy of a wide range of datasets and outperforms existing scoring metrics, (ii) captures interesting post fine-tuning "per-class" level properties from both qualitative and quantitative viewpoints, (iii) is more stable to perturbations as compared to the baseline functions, which makes it a very robust proxy, and (iv) finally, also serves as a predictor of the attack success rates for a wide category of black-box and white-box adversarial attack methods. Our extensive correlation experiments demonstrate the practical utility of PSF on various NLP tasks relevant to BERT.

摘要: 提出评分函数来有效地理解、分析和学习大型变压器模型的高维隐藏表示的各种性质可能是一项具有挑战性的任务。在这项工作中，我们探索了一个新的方向，通过研究BERT隐藏表示的拓扑特征，使用持久同调(PH)。我们提出了一种新的评分函数“持久性评分函数(PSF)”，它(I)准确地捕捉高维隐藏表示的同源性，并与大范围数据集的测试集精度很好地关联，并优于现有的评分度量；(Ii)从定性和定量的角度捕捉有趣的微调后“每类”级别的属性；(Iii)与基线函数相比，对扰动更稳定，这使得它成为一个非常健壮的代理；(Iv)还可以预测大范围的黑盒和白盒对抗攻击方法的攻击成功率。我们广泛的相关实验证明了PSF在与BERT相关的各种NLP任务中的实用价值。



## **22. On the Need of Neuromorphic Twins to Detect Denial-of-Service Attacks on Communication Networks**

论神经形态双胞胎检测通信网络拒绝服务攻击的必要性 cs.IT

submitted for publication

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16690v1) [paper-pdf](http://arxiv.org/pdf/2210.16690v1)

**Authors**: Holger Boche, Rafael F. Schaefer, H. Vincent Poor, Frank H. P. Fitzek

**Abstract**: As we are more and more dependent on the communication technologies, resilience against any attacks on communication networks is important to guarantee the digital sovereignty of our society. New developments of communication networks tackle the problem of resilience by in-network computing approaches for higher protocol layers, while the physical layer remains an open problem. This is particularly true for wireless communication systems which are inherently vulnerable to adversarial attacks due to the open nature of the wireless medium. In denial-of-service (DoS) attacks, an active adversary is able to completely disrupt the communication and it has been shown that Turing machines are incapable of detecting such attacks. As Turing machines provide the fundamental limits of digital information processing and therewith of digital twins, this implies that even the most powerful digital twins that preserve all information of the physical network error-free are not capable of detecting such attacks. This stimulates the question of how powerful the information processing hardware must be to enable the detection of DoS attacks. Therefore, in the paper the need of neuromorphic twins is advocated and by the use of Blum-Shub-Smale machines a first implementation that enables the detection of DoS attacks is shown. This result holds for both cases of with and without constraints on the input and jamming sequences of the adversary.

摘要: 随着我们对通信技术的依赖程度越来越高，抵御任何针对通信网络的攻击对于保障我们社会的数字主权至关重要。通信网络的新发展通过更高协议层的网络内计算方法解决了弹性问题，而物理层仍然是一个开放的问题。这对于无线通信系统尤其如此，由于无线介质的开放性质，无线通信系统天生就容易受到敌意攻击。在拒绝服务(DoS)攻击中，活跃的对手能够完全中断通信，并且已表明图灵机无法检测到此类攻击。由于图灵机提供了数字信息处理的基本限制以及数字双胞胎的基本限制，这意味着即使是最强大的数字双胞胎也无法检测到这种攻击，这些数字双胞胎正确地保存了物理网络的所有信息。这引发了一个问题，即信息处理硬件必须有多强大才能检测到DoS攻击。因此，本文提出了神经形态双胞胎的需要，并通过Blum-Shub-Smer机器的使用，展示了能够检测DoS攻击的第一个实现。这一结果对于对对手的输入和干扰序列有约束和无约束的两种情况都成立。



## **23. Security-Preserving Federated Learning via Byzantine-Sensitive Triplet Distance**

基于拜占庭敏感三元组距离的安全保护联合学习 cs.LG

5 pages

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16519v1) [paper-pdf](http://arxiv.org/pdf/2210.16519v1)

**Authors**: Youngjoon Lee, Sangwoo Park, Joonhyuk Kang

**Abstract**: While being an effective framework of learning a shared model across multiple edge devices, federated learning (FL) is generally vulnerable to Byzantine attacks from adversarial edge devices. While existing works on FL mitigate such compromised devices by only aggregating a subset of the local models at the server side, they still cannot successfully ignore the outliers due to imprecise scoring rule. In this paper, we propose an effective Byzantine-robust FL framework, namely dummy contrastive aggregation, by defining a novel scoring function that sensitively discriminates whether the model has been poisoned or not. Key idea is to extract essential information from every local models along with the previous global model to define a distance measure in a manner similar to triplet loss. Numerical results validate the advantage of the proposed approach by showing improved performance as compared to the state-of-the-art Byzantine-resilient aggregation methods, e.g., Krum, Trimmed-mean, and Fang.

摘要: 虽然联合学习(FL)是在多个边缘设备上学习共享模型的有效框架，但它通常容易受到来自敌对边缘设备的拜占庭攻击。虽然现有的FL工作仅通过在服务器端聚合本地模型的子集来缓解这种受危害的设备，但由于不精确的评分规则，它们仍然无法成功地忽略离群值。在本文中，我们通过定义一个新的评分函数来敏感地判断模型是否有毒，从而提出了一个有效的拜占庭-稳健FL框架，即虚拟对比聚集。关键思想是从每个局部模型和先前的全局模型中提取基本信息，以类似于三元组损失的方式定义距离度量。数值结果表明，与最先进的拜占庭弹性聚集方法(如Krum、Trimmed-Mean和Fang)相比，该方法的性能有所改善，从而验证了该方法的优势。



## **24. Robust Boosting Forests with Richer Deep Feature Hierarchy**

具有更丰富的深层特征层次结构的健壮增强型森林 cs.CV

**SubmitDate**: 2022-10-29    [abs](http://arxiv.org/abs/2210.16451v1) [paper-pdf](http://arxiv.org/pdf/2210.16451v1)

**Authors**: Jianqiao Wangni

**Abstract**: We propose a robust variant of boosting forest to the various adversarial defense methods, and apply it to enhance the robustness of the deep neural network. We retain the deep network architecture, weights, and middle layer features, then install gradient boosting forest to select the features from each layer of the deep network, and predict the target. For training each decision tree, we propose a novel conservative and greedy trade-off, with consideration for less misprediction instead of pure gain functions, therefore being suboptimal and conservative. We actively increase tree depth to remedy the accuracy with splits in more features, being more greedy in growing tree depth. We propose a new task on 3D face model, whose robustness has not been carefully studied, despite the great security and privacy concerns related to face analytics. We tried a simple attack method on a pure convolutional neural network (CNN) face shape estimator, making it degenerate to only output average face shape with invisible perturbation. Our conservative-greedy boosting forest (CGBF) on face landmark datasets showed a great improvement over original pure deep learning methods under the adversarial attacks.

摘要: 针对不同的对抗性防御方法，提出了一种稳健的Boost森林算法，并将其应用于增强深度神经网络的鲁棒性。我们保留了深层网络的结构、权值和中间层特征，然后设置梯度增强森林从深层网络的每一层中选择特征，并对目标进行预测。对于每一棵决策树的训练，我们提出了一种新的保守和贪婪的权衡，考虑了较少的误预测而不是纯增益函数，因此是次优的和保守的。我们积极增加树深，通过分裂更多的特征来弥补准确性，在生长树深时更加贪婪。我们提出了一项关于3D人脸模型的新任务，尽管人脸分析存在很大的安全和隐私问题，但其稳健性尚未被仔细研究。我们尝试了一种简单的攻击方法，对一个纯粹的卷积神经网络(CNN)人脸形状估计器进行攻击，使其退化为只输出带有不可见扰动的平均人脸形状。在人脸标志性数据集上，我们的保守贪婪增强森林(CGBF)方法在敌意攻击下比原有的纯深度学习方法有了很大的改进。



## **25. MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation**

迷宫：基于零阶梯度估计的无数据模型窃取攻击 stat.ML

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2005.03161v2) [paper-pdf](http://arxiv.org/pdf/2005.03161v2)

**Authors**: Sanjay Kariyappa, Atul Prakash, Moinuddin Qureshi

**Abstract**: Model Stealing (MS) attacks allow an adversary with black-box access to a Machine Learning model to replicate its functionality, compromising the confidentiality of the model. Such attacks train a clone model by using the predictions of the target model for different inputs. The effectiveness of such attacks relies heavily on the availability of data necessary to query the target model. Existing attacks either assume partial access to the dataset of the target model or availability of an alternate dataset with semantic similarities. This paper proposes MAZE -- a data-free model stealing attack using zeroth-order gradient estimation. In contrast to prior works, MAZE does not require any data and instead creates synthetic data using a generative model. Inspired by recent works in data-free Knowledge Distillation (KD), we train the generative model using a disagreement objective to produce inputs that maximize disagreement between the clone and the target model. However, unlike the white-box setting of KD, where the gradient information is available, training a generator for model stealing requires performing black-box optimization, as it involves accessing the target model under attack. MAZE relies on zeroth-order gradient estimation to perform this optimization and enables a highly accurate MS attack. Our evaluation with four datasets shows that MAZE provides a normalized clone accuracy in the range of 0.91x to 0.99x, and outperforms even the recent attacks that rely on partial data (JBDA, clone accuracy 0.13x to 0.69x) and surrogate data (KnockoffNets, clone accuracy 0.52x to 0.97x). We also study an extension of MAZE in the partial-data setting and develop MAZE-PD, which generates synthetic data closer to the target distribution. MAZE-PD further improves the clone accuracy (0.97x to 1.0x) and reduces the query required for the attack by 2x-24x.

摘要: 模型窃取(MS)攻击允许对机器学习模型具有黑盒访问权限的对手复制其功能，从而危及模型的机密性。这种攻击通过使用目标模型对不同输入的预测来训练克隆模型。此类攻击的有效性在很大程度上取决于查询目标模型所需的数据的可用性。现有攻击要么假设部分访问目标模型的数据集，要么假设具有语义相似性的备用数据集的可用性。提出了一种基于零阶梯度估计的无数据模型窃取攻击方法--迷宫攻击。与以前的工作不同，迷宫不需要任何数据，而是使用生成性模型创建合成数据。受最近无数据知识蒸馏(KD)方面工作的启发，我们使用不一致的目标来训练生成模型，以产生最大限度地提高克隆模型和目标模型之间的不一致性的输入。然而，与KD的白盒设置不同，在白盒设置中，梯度信息是可用的，训练生成器进行模型窃取需要执行黑盒优化，因为它涉及访问受到攻击的目标模型。迷宫依靠零阶梯度估计来执行这种优化，并实现高精度的MS攻击。我们对四个数据集的评估表明，MAZE在0.91x到0.99x的范围内提供了归一化克隆准确率，并且甚至优于最近依赖部分数据(JBDA，克隆准确率0.13x到0.69x)和代理数据(KnoockoffNets，克隆准确率0.52x到0.97x)的攻击。我们还研究了迷宫在部分数据环境下的扩展，提出了迷宫-PD算法，该算法生成更接近目标分布的合成数据。Maze-PD进一步提高了克隆准确率(0.97x到1.0x)，并将攻击所需的查询减少了2x-24x。



## **26. Distributed Black-box Attack against Image Classification Cloud Services**

针对图像分类云服务的分布式黑盒攻击 cs.LG

10 pages, 11 figures

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16371v1) [paper-pdf](http://arxiv.org/pdf/2210.16371v1)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: Black-box adversarial attacks can fool image classifiers into misclassifying images without requiring access to model structure and weights. Recently proposed black-box attacks can achieve a success rate of more than 95\% after less than 1,000 queries. The question then arises of whether black-box attacks have become a real threat against IoT devices that rely on cloud APIs to achieve image classification. To shed some light on this, note that prior research has primarily focused on increasing the success rate and reducing the number of required queries. However, another crucial factor for black-box attacks against cloud APIs is the time required to perform the attack. This paper applies black-box attacks directly to cloud APIs rather than to local models, thereby avoiding multiple mistakes made in prior research. Further, we exploit load balancing to enable distributed black-box attacks that can reduce the attack time by a factor of about five for both local search and gradient estimation methods.

摘要: 黑盒对抗性攻击可以欺骗图像分类器对图像进行错误分类，而不需要访问模型结构和权重。最近提出的黑盒攻击在不到1000次查询的情况下可以达到95%以上的成功率。随之而来的问题是，黑盒攻击是否已经成为对依赖云API实现图像分类的物联网设备的真正威胁。为了阐明这一点，请注意，以前的研究主要集中在提高成功率和减少所需的查询数量上。然而，针对云API的黑盒攻击的另一个关键因素是执行攻击所需的时间。本文将黑盒攻击直接应用于云API，而不是本地模型，从而避免了以往研究中的多个错误。此外，我们利用负载平衡来实现分布式黑盒攻击，对于局部搜索和梯度估计方法，可以将攻击时间减少约5倍。



## **27. Universalization of any adversarial attack using very few test examples**

使用极少的测试示例实现任何对抗性攻击的通用化 cs.LG

Appeared in ACM CODS-COMAD 2022 (Research Track)

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2005.08632v2) [paper-pdf](http://arxiv.org/pdf/2005.08632v2)

**Authors**: Sandesh Kamath, Amit Deshpande, K V Subrahmanyam, Vineeth N Balasubramanian

**Abstract**: Deep learning models are known to be vulnerable not only to input-dependent adversarial attacks but also to input-agnostic or universal adversarial attacks. Dezfooli et al. \cite{Dezfooli17,Dezfooli17anal} construct universal adversarial attack on a given model by looking at a large number of training data points and the geometry of the decision boundary near them. Subsequent work \cite{Khrulkov18} constructs universal attack by looking only at test examples and intermediate layers of the given model. In this paper, we propose a simple universalization technique to take any input-dependent adversarial attack and construct a universal attack by only looking at very few adversarial test examples. We do not require details of the given model and have negligible computational overhead for universalization. We theoretically justify our universalization technique by a spectral property common to many input-dependent adversarial perturbations, e.g., gradients, Fast Gradient Sign Method (FGSM) and DeepFool. Using matrix concentration inequalities and spectral perturbation bounds, we show that the top singular vector of input-dependent adversarial directions on a small test sample gives an effective and simple universal adversarial attack. For VGG16 and VGG19 models trained on ImageNet, our simple universalization of Gradient, FGSM, and DeepFool perturbations using a test sample of 64 images gives fooling rates comparable to state-of-the-art universal attacks \cite{Dezfooli17,Khrulkov18} for reasonable norms of perturbation. Code available at https://github.com/ksandeshk/svd-uap .

摘要: 众所周知，深度学习模型不仅容易受到依赖输入的对抗性攻击，而且还容易受到输入不可知的或普遍的对抗性攻击。德兹戈尼等人。{Dezfooli17，Dezfooli17anal}通过查看大量的训练数据点和它们附近的决策边界的几何形状来构造对给定模型的通用对抗性攻击。后续工作{Khrulkov18}通过只关注给定模型的测试用例和中间层来构造通用攻击。在本文中，我们提出了一种简单的普适化技术，可以接受任何依赖于输入的对抗性攻击，并且只需查看极少的对抗性测试实例就可以构建通用攻击。我们不需要给定模型的细节，并且通用性的计算开销可以忽略不计。我们从理论上证明了我们的普适化技术是由许多依赖于输入的对抗性扰动所共有的谱性质来证明的，例如梯度、快速梯度符号方法(FGSM)和DeepFool。利用矩阵集中不等式和谱摄动界，我们证明了在小样本上依赖于输入的对抗方向的顶部奇异向量给出了一种有效且简单的通用对抗攻击。对于在ImageNet上训练的VGG16和VGG19模型，我们使用64幅图像的测试样本对梯度、FGSM和DeepFool扰动的简单通用化提供了与最先进的通用攻击相当的愚弄率\引用{Dezfooli17，Khrulkov18}的合理扰动规范。代码可在https://github.com/ksandeshk/svd-uap上找到。



## **28. Local Model Reconstruction Attacks in Federated Learning and their Uses**

联合学习中的局部模型重构攻击及其应用 cs.LG

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16205v1) [paper-pdf](http://arxiv.org/pdf/2210.16205v1)

**Authors**: Ilias Driouich, Chuan Xu, Giovanni Neglia, Frederic Giroire, Eoin Thomas

**Abstract**: In this paper, we initiate the study of local model reconstruction attacks for federated learning, where a honest-but-curious adversary eavesdrops the messages exchanged between a targeted client and the server, and then reconstructs the local/personalized model of the victim. The local model reconstruction attack allows the adversary to trigger other classical attacks in a more effective way, since the local model only depends on the client's data and can leak more private information than the global model learned by the server. Additionally, we propose a novel model-based attribute inference attack in federated learning leveraging the local model reconstruction attack. We provide an analytical lower-bound for this attribute inference attack. Empirical results using real world datasets confirm that our local reconstruction attack works well for both regression and classification tasks. Moreover, we benchmark our novel attribute inference attack against the state-of-the-art attacks in federated learning. Our attack results in higher reconstruction accuracy especially when the clients' datasets are heterogeneous. Our work provides a new angle for designing powerful and explainable attacks to effectively quantify the privacy risk in FL.

摘要: 在本文中，我们发起了联合学习的本地模型重建攻击的研究，其中诚实但好奇的攻击者窃听目标客户端和服务器之间交换的消息，然后重建受害者的本地/个性化模型。本地模型重构攻击允许攻击者以更有效的方式触发其他经典攻击，因为本地模型仅依赖于客户端的数据，并且可以比服务器学习的全局模型泄露更多的私人信息。此外，利用局部模型重构攻击，提出了一种新的联邦学习中基于模型的属性推理攻击。我们给出了这种属性推理攻击的一个分析下界。使用真实世界数据集的实验结果证实，我们的局部重建攻击对于回归和分类任务都很好地工作。此外，我们还对联邦学习中最新的属性推理攻击进行了基准测试。我们的攻击导致了更高的重建精度，特别是当客户的数据集是异质的时候。我们的工作为设计强大的、可解释的攻击以有效量化FL中的隐私风险提供了一个新的角度。



## **29. Improving Transferability of Adversarial Examples on Face Recognition with Beneficial Perturbation Feature Augmentation**

利用有益扰动特征增强提高人脸识别中对抗性样本的可转移性 cs.CV

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.16117v1) [paper-pdf](http://arxiv.org/pdf/2210.16117v1)

**Authors**: Fengfan Zhou, Hefei Ling, Yuxuan Shi, Jiazhong Chen, Zongyi Li, Qian Wang

**Abstract**: Face recognition (FR) models can be easily fooled by adversarial examples, which are crafted by adding imperceptible perturbations on benign face images. To improve the transferability of adversarial examples on FR models, we propose a novel attack method called Beneficial Perturbation Feature Augmentation Attack (BPFA), which reduces the overfitting of the adversarial examples to surrogate FR models by the adversarial strategy. Specifically, in the backpropagation step, BPFA records the gradients on pre-selected features and uses the gradient on the input image to craft adversarial perturbation to be added on the input image. In the next forward propagation step, BPFA leverages the recorded gradients to add perturbations(i.e., beneficial perturbations) that can be pitted against the adversarial perturbation added on the input image on their corresponding features. The above two steps are repeated until the last backpropagation step before the maximum number of iterations is reached. The optimization process of the adversarial perturbation added on the input image and the optimization process of the beneficial perturbations added on the features correspond to a minimax two-player game. Extensive experiments demonstrate that BPFA outperforms the state-of-the-art gradient-based adversarial attacks on FR.

摘要: 人脸识别(FR)模型很容易被敌意的例子所愚弄，这些例子是通过在良性的人脸图像上添加难以察觉的扰动来构建的。为了提高对抗实例在FR模型上的可转移性，我们提出了一种新的攻击方法，称为有益扰动特征增强攻击(BPFA)，它通过对抗策略减少了对抗实例对替代FR模型的过度拟合。具体地说，在反向传播步骤中，BPFA记录预先选择的特征的梯度，并使用输入图像上的梯度来构造要添加到输入图像上的对抗性扰动。在下一前向传播步骤中，BPFA利用记录的梯度来添加扰动(即，有益扰动)，该扰动可以相对于添加在输入图像上的对抗性扰动添加到其相应的特征上。重复上述两个步骤，直到达到最大迭代次数之前的最后一个反向传播步骤。添加在输入图像上的对抗性扰动的优化过程和添加在特征上的有益扰动的优化过程对应于极小极大两人博弈。大量实验表明，BPFA的性能优于目前最先进的基于梯度的对抗性FR攻击。



## **30. Watermarking Graph Neural Networks based on Backdoor Attacks**

基于后门攻击的数字水印图神经网络 cs.LG

18 pages, 9 figures

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2110.11024v4) [paper-pdf](http://arxiv.org/pdf/2110.11024v4)

**Authors**: Jing Xu, Stefanos Koffas, Oguzhan Ersoy, Stjepan Picek

**Abstract**: Graph Neural Networks (GNNs) have achieved promising performance in various real-world applications. Building a powerful GNN model is not a trivial task, as it requires a large amount of training data, powerful computing resources, and human expertise in fine-tuning the model. Moreover, with the development of adversarial attacks, e.g., model stealing attacks, GNNs raise challenges to model authentication. To avoid copyright infringement on GNNs, verifying the ownership of the GNN models is necessary.   This paper presents a watermarking framework for GNNs for both graph and node classification tasks. We 1) design two strategies to generate watermarked data for the graph classification task and one for the node classification task, 2) embed the watermark into the host model through training to obtain the watermarked GNN model, and 3) verify the ownership of the suspicious model in a black-box setting. The experiments show that our framework can verify the ownership of GNN models with a very high probability (up to $99\%$) for both tasks. Finally, we experimentally show that our watermarking approach is robust against a state-of-the-art model extraction technique and four state-of-the-art defenses against backdoor attacks.

摘要: 图神经网络(GNN)在各种实际应用中取得了良好的性能。构建一个强大的GNN模型不是一项简单的任务，因为它需要大量的训练数据、强大的计算资源和微调模型的人力专业知识。此外，随着敌意攻击的发展，例如模型窃取攻击，GNN对模型认证提出了挑战。为了避免对GNN的版权侵权，有必要核实GNN模型的所有权。本文提出了一种适用于图和节点分类任务的GNN水印框架。我们设计了两种策略来为图分类任务和节点分类任务生成水印数据，2)通过训练将水印嵌入到宿主模型中，得到带水印的GNN模型，3)在黑盒环境下验证可疑模型的所有权。实验表明，我们的框架能够以很高的概率(高达99美元)验证这两个任务的GNN模型的所有权。最后，我们的实验表明，我们的水印方法对于一种最先进的模型提取技术和四种最先进的后门攻击防御方法是健壮的。



## **31. RoChBert: Towards Robust BERT Fine-tuning for Chinese**

RoChBert：为中文走向稳健的BERT微调 cs.CL

Accepted by Findings of EMNLP 2022

**SubmitDate**: 2022-10-28    [abs](http://arxiv.org/abs/2210.15944v1) [paper-pdf](http://arxiv.org/pdf/2210.15944v1)

**Authors**: Zihan Zhang, Jinfeng Li, Ning Shi, Bo Yuan, Xiangyu Liu, Rong Zhang, Hui Xue, Donghong Sun, Chao Zhang

**Abstract**: Despite of the superb performance on a wide range of tasks, pre-trained language models (e.g., BERT) have been proved vulnerable to adversarial texts. In this paper, we present RoChBERT, a framework to build more Robust BERT-based models by utilizing a more comprehensive adversarial graph to fuse Chinese phonetic and glyph features into pre-trained representations during fine-tuning. Inspired by curriculum learning, we further propose to augment the training dataset with adversarial texts in combination with intermediate samples. Extensive experiments demonstrate that RoChBERT outperforms previous methods in significant ways: (i) robust -- RoChBERT greatly improves the model robustness without sacrificing accuracy on benign texts. Specifically, the defense lowers the success rates of unlimited and limited attacks by 59.43% and 39.33% respectively, while remaining accuracy of 93.30%; (ii) flexible -- RoChBERT can easily extend to various language models to solve different downstream tasks with excellent performance; and (iii) efficient -- RoChBERT can be directly applied to the fine-tuning stage without pre-training language model from scratch, and the proposed data augmentation method is also low-cost.

摘要: 尽管在广泛的任务中表现出色，但预先训练的语言模型(如BERT)已被证明容易受到敌意文本的攻击。在本文中，我们提出了RoChBERT，这是一个框架，通过在微调过程中利用更全面的对抗性图将汉语语音和字形特征融合到预先训练的表示中来建立更健壮的基于ERT的模型。受课程学习的启发，我们进一步提出用对抗性文本结合中间样本来扩充训练数据集。大量的实验表明，RoChBERT在以下几个方面明显优于以往的方法：(I)健壮性--RoChBERT在不牺牲对良性文本的准确性的情况下，大大提高了模型的健壮性。具体来说，防御使无限和有限攻击的成功率分别降低了59.43%和39.33%，同时保持了93.30%的准确率；(Ii)灵活的-RoChBERT可以很容易地扩展到各种语言模型，以优异的性能解决不同的下游任务；(Iii)高效-RoChBERT可以直接应用到微调阶段，而不需要从零开始预先训练语言模型，并且所提出的数据增强方法也是低成本的。



## **32. DICTION: DynamIC robusT whIte bOx watermarkiNg scheme**

动态稳健白盒水印方案 cs.CR

18 pages, 5 figures, PrePrint

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15745v1) [paper-pdf](http://arxiv.org/pdf/2210.15745v1)

**Authors**: Reda Bellafqira, Gouenou Coatrieux

**Abstract**: Deep neural network (DNN) watermarking is a suitable method for protecting the ownership of deep learning (DL) models derived from computationally intensive processes and painstakingly compiled and annotated datasets. It secretly embeds an identifier (watermark) within the model, which can be retrieved by the owner to prove ownership. In this paper, we first provide a unified framework for white box DNN watermarking schemes. It includes current state-of-the art methods outlining their theoretical inter-connections. In second, we introduce DICTION, a new white-box Dynamic Robust watermarking scheme, we derived from this framework. Its main originality stands on a generative adversarial network (GAN) strategy where the watermark extraction function is a DNN trained as a GAN discriminator, and the target model to watermark as a GAN generator taking a GAN latent space as trigger set input. DICTION can be seen as a generalization of DeepSigns which, to the best of knowledge, is the only other Dynamic white-box watermarking scheme from the literature. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves much better performance. Especially, and contrarily to DeepSigns, with DICTION one can increase the watermark capacity while preserving at best the model accuracy and ensuring simultaneously a strong robustness against a wide range of watermark removal and detection attacks.

摘要: 深度神经网络(DNN)水印是一种适合于保护深度学习(DL)模型所有权的方法，该模型源于计算密集型过程和精心编译和注释的数据集。它在模型中秘密嵌入一个标识符(水印)，所有者可以检索该标识符以证明所有权。本文首先给出了白盒DNN数字水印方案的统一框架。它包括当前最先进的方法，概述了它们理论上的相互联系。其次，介绍了一种新的白盒动态鲁棒水印方案--WICH，它是由该框架衍生出来的。它的主要创新之处在于一种生成性对抗网络(GAN)策略，其中水印提取函数是训练成GAN鉴别器的DNN，目标模型是以GAN潜在空间作为触发集输入的GAN生成器。就目前所知，DeepSigns是文献中唯一的动态白盒水印方案，它可以被视为DeepSigns的推广。在与DeepDesign相同的模型测试集上进行的实验表明，我们的方案取得了更好的性能。特别是，与DeepSigns相反，使用该算法可以在最好地保持模型精度的同时增加水印容量，并同时确保对广泛的水印去除和检测攻击具有很强的鲁棒性。



## **33. TAD: Transfer Learning-based Multi-Adversarial Detection of Evasion Attacks against Network Intrusion Detection Systems**

TAD：基于转移学习的网络入侵检测系统逃避攻击的多对手检测 cs.CR

This is a preprint of an already published journal paper

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15700v1) [paper-pdf](http://arxiv.org/pdf/2210.15700v1)

**Authors**: Islam Debicha, Richard Bauwens, Thibault Debatty, Jean-Michel Dricot, Tayeb Kenaza, Wim Mees

**Abstract**: Nowadays, intrusion detection systems based on deep learning deliver state-of-the-art performance. However, recent research has shown that specially crafted perturbations, called adversarial examples, are capable of significantly reducing the performance of these intrusion detection systems. The objective of this paper is to design an efficient transfer learning-based adversarial detector and then to assess the effectiveness of using multiple strategically placed adversarial detectors compared to a single adversarial detector for intrusion detection systems. In our experiments, we implement existing state-of-the-art models for intrusion detection. We then attack those models with a set of chosen evasion attacks. In an attempt to detect those adversarial attacks, we design and implement multiple transfer learning-based adversarial detectors, each receiving a subset of the information passed through the IDS. By combining their respective decisions, we illustrate that combining multiple detectors can further improve the detectability of adversarial traffic compared to a single detector in the case of a parallel IDS design.

摘要: 如今，基于深度学习的入侵检测系统提供了最先进的性能。然而，最近的研究表明，精心设计的扰动，称为对抗性示例，能够显著降低这些入侵检测系统的性能。本文的目的是设计一种高效的基于转移学习的敌意检测器，并在此基础上评估在入侵检测系统中使用多个策略放置的敌意检测器与使用单个敌意检测器的有效性。在我们的实验中，我们实现了现有的最先进的入侵检测模型。然后，我们用一系列有选择的规避攻击来攻击这些模型。为了检测这些敌意攻击，我们设计并实现了多个基于转移学习的敌意检测器，每个检测器接收通过入侵检测系统传递的信息的一个子集。通过结合它们各自的决策，我们说明了在并行入侵检测系统设计的情况下，与单一检测器相比，组合多个检测器可以进一步提高敌意流量的可检测性。



## **34. Learning Location from Shared Elevation Profiles in Fitness Apps: A Privacy Perspective**

从隐私的角度从健身应用程序中的共享高程配置文件学习位置 cs.CR

16 pages, 12 figures, 10 tables; accepted for publication in IEEE  Transactions on Mobile Computing (October 2022). arXiv admin note:  substantial text overlap with arXiv:1910.09041

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15529v1) [paper-pdf](http://arxiv.org/pdf/2210.15529v1)

**Authors**: Ulku Meteriz-Yildiran, Necip Fazil Yildiran, Joongheon Kim, David Mohaisen

**Abstract**: The extensive use of smartphones and wearable devices has facilitated many useful applications. For example, with Global Positioning System (GPS)-equipped smart and wearable devices, many applications can gather, process, and share rich metadata, such as geolocation, trajectories, elevation, and time. For example, fitness applications, such as Runkeeper and Strava, utilize the information for activity tracking and have recently witnessed a boom in popularity. Those fitness tracker applications have their own web platforms and allow users to share activities on such platforms or even with other social network platforms. To preserve the privacy of users while allowing sharing, several of those platforms may allow users to disclose partial information, such as the elevation profile for an activity, which supposedly would not leak the location of the users. In this work, and as a cautionary tale, we create a proof of concept where we examine the extent to which elevation profiles can be used to predict the location of users. To tackle this problem, we devise three plausible threat settings under which the city or borough of the targets can be predicted. Those threat settings define the amount of information available to the adversary to launch the prediction attacks. Establishing that simple features of elevation profiles, e.g., spectral features, are insufficient, we devise both natural language processing (NLP)-inspired text-like representation and computer vision-inspired image-like representation of elevation profiles, and we convert the problem at hand into text and image classification problem. We use both traditional machine learning- and deep learning-based techniques and achieve a prediction success rate ranging from 59.59\% to 99.80\%. The findings are alarming, highlighting that sharing elevation information may have significant location privacy risks.

摘要: 智能手机和可穿戴设备的广泛使用促进了许多有用的应用。例如，使用配备全球定位系统(GPS)的智能可穿戴设备，许多应用程序可以收集、处理和共享丰富的元数据，如地理位置、轨迹、高程和时间。例如，RunKeeper和Strava等健身应用程序利用这些信息进行活动跟踪，最近见证了这种应用程序的流行。这些健身跟踪应用程序有自己的网络平台，允许用户在这些平台上甚至与其他社交网络平台分享活动。为了在允许分享的同时保护用户的隐私，其中几个平台可能会允许用户披露部分信息，比如活动的海拔概况，这应该不会泄露用户的位置。在这项工作中，作为一个警示故事，我们创建了一个概念证明，其中我们检查了高程分布可以在多大程度上用于预测用户的位置。为了解决这个问题，我们设计了三个可信的威胁设置，在这些设置下可以预测目标的城市或行政区。这些威胁设置定义了对手可用来发动预测攻击的信息量。建立了高程剖面的简单特征，例如光谱特征是不够的，我们设计了受自然语言处理(NLP)启发的类似文本的高程剖面表示和受计算机视觉启发的类似图像的高程剖面表示，并将手头的问题转化为文本和图像分类问题。我们同时使用了传统的机器学习和基于深度学习的技术，预测成功率从59.59到99.80%不等。这些发现令人担忧，突显出共享高程信息可能会带来重大的位置隐私风险。



## **35. An Analysis of Robustness of Non-Lipschitz Networks**

非Lipschitz网络的稳健性分析 cs.LG

42 pages, 9 figures

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2010.06154v3) [paper-pdf](http://arxiv.org/pdf/2010.06154v3)

**Authors**: Maria-Florina Balcan, Avrim Blum, Dravyansh Sharma, Hongyang Zhang

**Abstract**: Despite significant advances, deep networks remain highly susceptible to adversarial attack. One fundamental challenge is that small input perturbations can often produce large movements in the network's final-layer feature space. In this paper, we define an attack model that abstracts this challenge, to help understand its intrinsic properties. In our model, the adversary may move data an arbitrary distance in feature space but only in random low-dimensional subspaces. We prove such adversaries can be quite powerful: defeating any algorithm that must classify any input it is given. However, by allowing the algorithm to abstain on unusual inputs, we show such adversaries can be overcome when classes are reasonably well-separated in feature space. We further provide strong theoretical guarantees for setting algorithm parameters to optimize over accuracy-abstention trade-offs using data-driven methods. Our results provide new robustness guarantees for nearest-neighbor style algorithms, and also have application to contrastive learning, where we empirically demonstrate the ability of such algorithms to obtain high robust accuracy with low abstention rates. Our model is also motivated by strategic classification, where entities being classified aim to manipulate their observable features to produce a preferred classification, and we provide new insights into that area as well.

摘要: 尽管取得了重大进展，深度网络仍然极易受到对手的攻击。一个基本的挑战是，小的输入扰动通常会在网络的最后一层特征空间中产生大的运动。在本文中，我们定义了一个抽象这一挑战的攻击模型，以帮助理解其内在属性。在我们的模型中，敌手可以将数据在特征空间中移动任意距离，但只能在随机低维子空间中移动。我们证明了这样的对手可以是相当强大的：击败任何必须对给定的任何输入进行分类的算法。然而，通过允许算法在不寻常的输入上弃权，我们证明了当类在特征空间中合理地分离时，这样的对手可以被克服。我们进一步为使用数据驱动方法设置算法参数以优化过度精确度权衡提供了有力的理论保证。我们的结果为最近邻式算法提供了新的稳健性保证，并在对比学习中也得到了应用，我们的经验证明了这种算法能够在较低的弃权率下获得较高的鲁棒性精度。我们的模型也受到战略分类的推动，在战略分类中，被分类的实体旨在操纵它们的可观察特征来产生首选的分类，我们也提供了对该领域的新见解。



## **36. LeNo: Adversarial Robust Salient Object Detection Networks with Learnable Noise**

Leno：具有可学习噪声的对抗性鲁棒显著目标检测网络 cs.CV

8 pages, 5 figures, submitted to AAAI

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15392v1) [paper-pdf](http://arxiv.org/pdf/2210.15392v1)

**Authors**: He Tang, He Wang

**Abstract**: Pixel-wise predction with deep neural network has become an effective paradigm for salient object detection (SOD) and achieved remakable performance. However, very few SOD models are robust against adversarial attacks which are visually imperceptible for human visual attention. The previous work robust salient object detection against adversarial attacks (ROSA) shuffles the pre-segmented superpixels and then refines the coarse saliency map by the densely connected CRF. Different from ROSA that rely on various pre- and post-processings, this paper proposes a light-weight Learnble Noise (LeNo) to against adversarial attacks for SOD models. LeNo preserves accuracy of SOD models on both adversarial and clean images, as well as inference speed. In general, LeNo consists of a simple shallow noise and noise estimation that embedded in the encoder and decoder of arbitrary SOD networks respectively. Inspired by the center prior of human visual attention mechanism, we initialize the shallow noise with a cross-shaped gaussian distribution for better defense against adversarial attacks. Instead of adding additional network components for post-processing, the proposed noise estimation modifies only one channel of the decoder. With the deeply-supervised noise-decoupled training on state-of-the-art RGB and RGB-D SOD networks, LeNo outperforms previous works not only on adversarial images but also clean images, which contributes stronger robustness for SOD.

摘要: 基于深度神经网络的像素预测已成为显著目标检测的一种有效范例，并取得了较好的性能。然而，很少有SOD模型对人类视觉上不可察觉的对抗性攻击具有健壮性。针对敌意攻击的稳健显著目标检测(ROSA)算法首先对预分割的超像素进行置乱处理，然后利用稠密连接的CRF函数对粗略显著图进行细化。不同于ROSA依赖于各种前后处理，本文提出了一种轻量级可学习噪声(Leno)来抵抗对SOD模型的敌意攻击。Leno保持了SOD模型在对抗性图像和干净图像上的准确性，以及推理速度。一般来说，LENO由简单的浅层噪声和噪声估计组成，分别嵌入到任意SOD网络的编码器和译码中。受人类视觉注意机制中心先验的启发，我们用十字形高斯分布对浅层噪声进行初始化，以更好地防御对手的攻击。所提出的噪声估计只需修改解码器的一个通道，而不是为后处理增加额外的网络组件。通过在最新的RGB和RGB-D SOD网络上进行深度监督的噪声解耦训练，Leno不仅在对抗性图像上而且在干净的图像上都优于以往的工作，这为SOD提供了更强的稳健性。



## **37. Isometric 3D Adversarial Examples in the Physical World**

物理世界中的等距3D对抗性例子 cs.CV

NeurIPS 2022

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15291v1) [paper-pdf](http://arxiv.org/pdf/2210.15291v1)

**Authors**: Yibo Miao, Yinpeng Dong, Jun Zhu, Xiao-Shan Gao

**Abstract**: 3D deep learning models are shown to be as vulnerable to adversarial examples as 2D models. However, existing attack methods are still far from stealthy and suffer from severe performance degradation in the physical world. Although 3D data is highly structured, it is difficult to bound the perturbations with simple metrics in the Euclidean space. In this paper, we propose a novel $\epsilon$-isometric ($\epsilon$-ISO) attack to generate natural and robust 3D adversarial examples in the physical world by considering the geometric properties of 3D objects and the invariance to physical transformations. For naturalness, we constrain the adversarial example to be $\epsilon$-isometric to the original one by adopting the Gaussian curvature as a surrogate metric guaranteed by a theoretical analysis. For invariance to physical transformations, we propose a maxima over transformation (MaxOT) method that actively searches for the most harmful transformations rather than random ones to make the generated adversarial example more robust in the physical world. Experiments on typical point cloud recognition models validate that our approach can significantly improve the attack success rate and naturalness of the generated 3D adversarial examples than the state-of-the-art attack methods.

摘要: 研究表明，3D深度学习模型与2D模型一样容易受到敌意例子的影响。然而，现有的攻击方法还远远不是隐身的，在物理世界中还存在严重的性能下降。虽然三维数据是高度结构化的，但在欧氏空间中很难用简单的度量来约束扰动。考虑到三维物体的几何特性和对物理变换的不变性，提出了一种新的三维等距($-ISO)攻击，用于在物理世界中生成自然的和健壮的3D对抗实例。对于自然度，我们通过采用高斯曲率作为理论分析所保证的替代度量来约束对抗性实例与原始实例的-等距。对于物理变换的不变性，我们提出了一种最大值过变换(MaxOT)方法，该方法主动地搜索最有害的变换而不是随机的变换，以使生成的对抗性实例在物理世界中更健壮。在典型点云识别模型上的实验证明，与现有的攻击方法相比，该方法可以显著提高生成的3D对抗性实例的攻击成功率和自然度。



## **38. TASA: Deceiving Question Answering Models by Twin Answer Sentences Attack**

TASA：利用双答句攻击欺骗问答模型 cs.CL

Accepted by EMNLP 2022 (long), 9 pages main + 2 pages references + 7  pages appendix

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15221v1) [paper-pdf](http://arxiv.org/pdf/2210.15221v1)

**Authors**: Yu Cao, Dianqi Li, Meng Fang, Tianyi Zhou, Jun Gao, Yibing Zhan, Dacheng Tao

**Abstract**: We present Twin Answer Sentences Attack (TASA), an adversarial attack method for question answering (QA) models that produces fluent and grammatical adversarial contexts while maintaining gold answers. Despite phenomenal progress on general adversarial attacks, few works have investigated the vulnerability and attack specifically for QA models. In this work, we first explore the biases in the existing models and discover that they mainly rely on keyword matching between the question and context, and ignore the relevant contextual relations for answer prediction. Based on two biases above, TASA attacks the target model in two folds: (1) lowering the model's confidence on the gold answer with a perturbed answer sentence; (2) misguiding the model towards a wrong answer with a distracting answer sentence. Equipped with designed beam search and filtering methods, TASA can generate more effective attacks than existing textual attack methods while sustaining the quality of contexts, in extensive experiments on five QA datasets and human evaluations.

摘要: 我们提出了一种针对问答(QA)模型的对抗性攻击方法Twin Answer语句攻击(TASA)，该方法在保持黄金答案的同时产生流畅的语法对抗性上下文。尽管在一般对抗性攻击方面取得了显著的进展，但很少有研究专门针对QA模型的脆弱性和攻击。在这项工作中，我们首先探讨了现有模型中的偏差，发现它们主要依赖于问题和上下文之间的关键字匹配，而忽略了相关的上下文关系来进行答案预测。基于以上两个偏差，TASA从两个方面对目标模型进行攻击：(1)用扰动答案句降低模型对黄金答案的置信度；(2)用令人分心的答案句将模型误导向错误答案。在五个QA数据集和人工评估的广泛实验中，TASA配备了设计的波束搜索和过滤方法，在保持上下文质量的情况下，可以产生比现有文本攻击方法更有效的攻击。



## **39. V-Cloak: Intelligibility-, Naturalness- & Timbre-Preserving Real-Time Voice Anonymization**

V-Cloak：可理解性、自然性和保留音色的实时语音匿名化 cs.SD

Accepted by USENIX Security Symposium 2023

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.15140v1) [paper-pdf](http://arxiv.org/pdf/2210.15140v1)

**Authors**: Jiangyi Deng, Fei Teng, Yanjiao Chen, Xiaofu Chen, Zhaohui Wang, Wenyuan Xu

**Abstract**: Voice data generated on instant messaging or social media applications contains unique user voiceprints that may be abused by malicious adversaries for identity inference or identity theft. Existing voice anonymization techniques, e.g., signal processing and voice conversion/synthesis, suffer from degradation of perceptual quality. In this paper, we develop a voice anonymization system, named V-Cloak, which attains real-time voice anonymization while preserving the intelligibility, naturalness and timbre of the audio. Our designed anonymizer features a one-shot generative model that modulates the features of the original audio at different frequency levels. We train the anonymizer with a carefully-designed loss function. Apart from the anonymity loss, we further incorporate the intelligibility loss and the psychoacoustics-based naturalness loss. The anonymizer can realize untargeted and targeted anonymization to achieve the anonymity goals of unidentifiability and unlinkability.   We have conducted extensive experiments on four datasets, i.e., LibriSpeech (English), AISHELL (Chinese), CommonVoice (French) and CommonVoice (Italian), five Automatic Speaker Verification (ASV) systems (including two DNN-based, two statistical and one commercial ASV), and eleven Automatic Speech Recognition (ASR) systems (for different languages). Experiment results confirm that V-Cloak outperforms five baselines in terms of anonymity performance. We also demonstrate that V-Cloak trained only on the VoxCeleb1 dataset against ECAPA-TDNN ASV and DeepSpeech2 ASR has transferable anonymity against other ASVs and cross-language intelligibility for other ASRs. Furthermore, we verify the robustness of V-Cloak against various de-noising techniques and adaptive attacks. Hopefully, V-Cloak may provide a cloak for us in a prism world.

摘要: 即时消息或社交媒体应用程序上生成的语音数据包含独特的用户声纹，恶意攻击者可能会利用这些声纹进行身份推断或身份窃取。现有的语音匿名化技术，例如信号处理和语音转换/合成，存在感知质量下降的问题。在本文中，我们开发了一个语音匿名系统V-Cloak，它在保持音频的可理解性、自然度和音色的同时，实现了实时的语音匿名。我们设计的匿名器具有一次性生成模型，可以在不同的频率水平上调制原始音频的特征。我们用精心设计的损失函数来训练匿名者。除了匿名性损失外，我们还进一步引入了可理解性损失和基于心理声学的自然度损失。匿名者可以实现无定向和定向匿名化，达到不可识别和不可链接的匿名性目标。我们在LibriSpeech(英语)、AISHELL(中文)、CommonVoice(法语)和CommonVoice(意大利语)四个数据集上进行了广泛的实验，五个自动说话人确认(ASV)系统(包括两个基于DNN的、两个统计的和一个商业的ASV)和11个自动语音识别(ASR)系统(针对不同的语言)。实验结果表明，V-Cloak在匿名性方面优于5条Baseline。我们还证明了仅在VoxCeleb1数据集上针对ECAPA-TDNN ASV和DeepSpeech2 ASR进行训练的V-Cloak具有针对其他ASV的可传递匿名性和针对其他ASR的跨语言可理解性。此外，我们还验证了V-Cloak对各种去噪技术和自适应攻击的稳健性。希望V-Cloak可以为我们在棱镜世界中提供一件斗篷。



## **40. Adaptive Test-Time Defense with the Manifold Hypothesis**

流形假设下的自适应测试时间防御 cs.LG

**SubmitDate**: 2022-10-27    [abs](http://arxiv.org/abs/2210.14404v2) [paper-pdf](http://arxiv.org/pdf/2210.14404v2)

**Authors**: Zhaoyuan Yang, Zhiwei Xu, Jing Zhang, Richard Hartley, Peter Tu

**Abstract**: In this work, we formulate a novel framework of adversarial robustness using the manifold hypothesis. Our framework provides sufficient conditions for defending against adversarial examples. We develop a test-time defense method with our formulation and variational inference. The developed approach combines manifold learning with the Bayesian framework to provide adversarial robustness without the need for adversarial training. We show that our proposed approach can provide adversarial robustness even if attackers are aware of existence of test-time defense. In additions, our approach can also serve as a test-time defense mechanism for variational autoencoders.

摘要: 在这项工作中，我们使用流形假设建立了一个新的对抗健壮性框架。我们的框架为防御对抗性例子提供了充分的条件。利用我们的公式和变分推理，我们开发了一种测试时间防御方法。该方法将流形学习与贝叶斯框架相结合，在不需要对抗性训练的情况下提供对抗性健壮性。我们证明，即使攻击者知道测试时间防御的存在，我们所提出的方法也可以提供对抗健壮性。此外，我们的方法还可以作为可变自动编码器的测试时间防御机制。



## **41. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

自定步长硬类对重加权提高对手健壮性 cs.CV

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15068v1) [paper-pdf](http://arxiv.org/pdf/2210.15068v1)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most recognized methods. Theoretically, the predicted labels of untargeted attacks should be unpredictable and uniformly-distributed overall false classes. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs to become the virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair loss in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boost model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.

摘要: 深度神经网络很容易受到敌意攻击。在众多的防御策略中，非靶向攻击的对抗性训练是最受认可的方法之一。从理论上讲，非目标攻击的预测标签应该是不可预测的，且总体上均匀分布的伪类。然而，我们发现，自然不平衡的类间语义相似度使得这些硬类对成为彼此的虚拟目标。本研究调查了这种紧密耦合的课程对对抗性攻击的影响，并相应地在对抗性训练中开发了一种自定步调重权重策略。具体地说，我们提出了在模型优化中增加硬类对损失的权重，从而促进了从硬类中学习区分特征。在对抗性训练中，我们进一步引入了一个术语来量化硬类对一致性，这大大提高了模型的稳健性。大量的实验表明，所提出的对抗性训练方法在对抗广泛的对抗性攻击时获得了比最先进的防御方法更好的健壮性性能。



## **42. Using Deception in Markov Game to Understand Adversarial Behaviors through a Capture-The-Flag Environment**

利用马尔可夫博弈中的欺骗来理解捕获旗帜环境中的敌方行为 cs.GT

Accepted at GameSec 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15011v1) [paper-pdf](http://arxiv.org/pdf/2210.15011v1)

**Authors**: Siddhant Bhambri, Purv Chauhan, Frederico Araujo, Adam Doupé, Subbarao Kambhampati

**Abstract**: Identifying the actual adversarial threat against a system vulnerability has been a long-standing challenge for cybersecurity research. To determine an optimal strategy for the defender, game-theoretic based decision models have been widely used to simulate the real-world attacker-defender scenarios while taking the defender's constraints into consideration. In this work, we focus on understanding human attacker behaviors in order to optimize the defender's strategy. To achieve this goal, we model attacker-defender engagements as Markov Games and search for their Bayesian Stackelberg Equilibrium. We validate our modeling approach and report our empirical findings using a Capture-The-Flag (CTF) setup, and we conduct user studies on adversaries with varying skill-levels. Our studies show that application-level deceptions are an optimal mitigation strategy against targeted attacks -- outperforming classic cyber-defensive maneuvers, such as patching or blocking network requests. We use this result to further hypothesize over the attacker's behaviors when trapped in an embedded honeypot environment and present a detailed analysis of the same.

摘要: 识别针对系统漏洞的实际对手威胁一直是网络安全研究的长期挑战。为了确定防御者的最优策略，基于博弈论的决策模型被广泛用于模拟现实世界中的攻防场景，同时考虑了防御者的约束。在这项工作中，我们重点了解人类攻击者的行为，以便优化防御者的策略。为了实现这一目标，我们将攻防双方的交战建模为马尔可夫博弈，并寻找他们的贝叶斯Stackelberg均衡。我们验证了我们的建模方法，并使用捕获旗帜(CTF)设置报告了我们的经验结果，并对具有不同技能水平的对手进行了用户研究。我们的研究表明，应用程序级别的欺骗是针对目标攻击的最佳缓解策略--性能优于修补或阻止网络请求等传统的网络防御策略。我们利用这一结果进一步假设攻击者在被困在嵌入式蜜罐环境中时的行为，并对此进行了详细的分析。



## **43. Model-Free Prediction of Adversarial Drop Points in 3D Point Clouds**

三维点云中对抗性滴点的无模式预报 cs.CV

10 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14164v2) [paper-pdf](http://arxiv.org/pdf/2210.14164v2)

**Authors**: Hanieh Naderi, Chinthaka Dinesh, Ivan V. Bajic, Shohreh Kasaei

**Abstract**: Adversarial attacks pose serious challenges for deep neural network (DNN)-based analysis of various input signals. In the case of 3D point clouds, methods have been developed to identify points that play a key role in the network decision, and these become crucial in generating existing adversarial attacks. For example, a saliency map approach is a popular method for identifying adversarial drop points, whose removal would significantly impact the network decision. Generally, methods for identifying adversarial points rely on the deep model itself in order to determine which points are critically important for the model's decision. This paper aims to provide a novel viewpoint on this problem, in which adversarial points can be predicted independently of the model. To this end, we define 14 point cloud features and use multiple linear regression to examine whether these features can be used for model-free adversarial point prediction, and which combination of features is best suited for this purpose. Experiments show that a suitable combination of features is able to predict adversarial points of three different networks -- PointNet, PointNet++, and DGCNN -- significantly better than a random guess. The results also provide further insight into DNNs for point cloud analysis, by showing which features play key roles in their decision-making process.

摘要: 对抗性攻击对基于深度神经网络(DNN)的各种输入信号分析提出了严峻的挑战。在3D点云的情况下，已经开发出方法来识别在网络决策中起关键作用的点，并且这些点在生成现有的对抗性攻击时变得至关重要。例如，显著图方法是一种流行的识别对抗性丢弃点的方法，其移除将显著影响网络决策。通常，识别敌对点的方法依赖于深度模型本身，以确定哪些点对模型的决策至关重要。本文旨在为这一问题提供一种新的观点，即可以独立于模型来预测敌对点。为此，我们定义了14个点云特征，并使用多元线性回归来检验这些特征是否可以用于无模型对抗点预测，以及哪种特征组合最适合于此目的。实验表明，适当的特征组合能够预测三种不同网络--PointNet、PointNet++和DGCNN的敌对点--明显好于随机猜测。通过显示哪些特征在其决策过程中起关键作用，结果还提供了对用于点云分析的DNN的进一步洞察。



## **44. Disentangled Text Representation Learning with Information-Theoretic Perspective for Adversarial Robustness**

基于信息论视角的解缠文本表征学习的对抗性 cs.CL

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14957v1) [paper-pdf](http://arxiv.org/pdf/2210.14957v1)

**Authors**: Jiahao Zhao, Wenji Mao

**Abstract**: Adversarial vulnerability remains a major obstacle to constructing reliable NLP systems. When imperceptible perturbations are added to raw input text, the performance of a deep learning model may drop dramatically under attacks. Recent work argues the adversarial vulnerability of the model is caused by the non-robust features in supervised training. Thus in this paper, we tackle the adversarial robustness challenge from the view of disentangled representation learning, which is able to explicitly disentangle robust and non-robust features in text. Specifically, inspired by the variation of information (VI) in information theory, we derive a disentangled learning objective composed of mutual information to represent both the semantic representativeness of latent embeddings and differentiation of robust and non-robust features. On the basis of this, we design a disentangled learning network to estimate these mutual information. Experiments on text classification and entailment tasks show that our method significantly outperforms the representative methods under adversarial attacks, indicating that discarding non-robust features is critical for improving adversarial robustness.

摘要: 对抗性漏洞仍然是构建可靠的自然语言处理系统的主要障碍。当向原始输入文本添加不可察觉的扰动时，深度学习模型的性能可能会在攻击下显著下降。最近的工作认为，该模型的对抗性漏洞是由监督训练中的非稳健特征造成的。因此，在本文中，我们从解缠表示学习的角度来解决对抗性的健壮性挑战，它能够显式地解开文本中的健壮和非健壮特征。具体地说，受信息论中信息变化(VI)的启发，我们提出了一个由互信息组成的解缠学习目标，以表示潜在嵌入的语义代表性以及稳健和非稳健特征的区分。在此基础上，我们设计了一个解缠学习网络来估计这些互信息。在文本分类和蕴涵任务上的实验表明，我们的方法在对抗攻击下的性能明显优于典型的方法，这表明丢弃非健壮性特征对于提高对抗攻击的稳健性至关重要。



## **45. On the Versatile Uses of Partial Distance Correlation in Deep Learning**

偏距离相关在深度学习中的广泛应用 cs.CV

This paper has been selected as best paper award for ECCV 2022!

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2207.09684v2) [paper-pdf](http://arxiv.org/pdf/2207.09684v2)

**Authors**: Xingjian Zhen, Zihang Meng, Rudrasis Chakraborty, Vikas Singh

**Abstract**: Comparing the functional behavior of neural network models, whether it is a single network over time or two (or more networks) during or post-training, is an essential step in understanding what they are learning (and what they are not), and for identifying strategies for regularization or efficiency improvements. Despite recent progress, e.g., comparing vision transformers to CNNs, systematic comparison of function, especially across different networks, remains difficult and is often carried out layer by layer. Approaches such as canonical correlation analysis (CCA) are applicable in principle, but have been sparingly used so far. In this paper, we revisit a (less widely known) from statistics, called distance correlation (and its partial variant), designed to evaluate correlation between feature spaces of different dimensions. We describe the steps necessary to carry out its deployment for large scale models -- this opens the door to a surprising array of applications ranging from conditioning one deep model w.r.t. another, learning disentangled representations as well as optimizing diverse models that would directly be more robust to adversarial attacks. Our experiments suggest a versatile regularizer (or constraint) with many advantages, which avoids some of the common difficulties one faces in such analyses. Code is at https://github.com/zhenxingjian/Partial_Distance_Correlation.

摘要: 比较神经网络模型的功能行为，无论是随着时间的推移是单个网络还是在训练期间或训练后的两个(或更多)网络，对于了解它们正在学习什么(以及它们不是什么)以及确定正规化或效率改进的策略是至关重要的一步。尽管最近取得了进展，例如，将视觉转换器与CNN进行了比较，但系统地比较功能，特别是跨不同网络的功能，仍然很困难，而且往往是逐层进行的。典型相关分析(CCA)等方法在原则上是适用的，但到目前为止一直很少使用。在本文中，我们回顾了统计学中的一个(不太广为人知的)方法，称为距离相关(及其部分变量)，旨在评估不同维度的特征空间之间的相关性。我们描述了在大规模模型中实施其部署所需的步骤--这为一系列令人惊讶的应用打开了大门，从调节一个深度模型到W.r.t。另一种是，学习分离的表示以及优化多样化的模型，这些模型将直接对对手攻击更健壮。我们的实验提出了一种具有许多优点的通用正则化(或约束)方法，它避免了人们在此类分析中面临的一些常见困难。代码在https://github.com/zhenxingjian/Partial_Distance_Correlation.上



## **46. Identifying Threats, Cybercrime and Digital Forensic Opportunities in Smart City Infrastructure via Threat Modeling**

通过威胁建模识别智能城市基础设施中的威胁、网络犯罪和数字取证机会 cs.CR

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14692v1) [paper-pdf](http://arxiv.org/pdf/2210.14692v1)

**Authors**: Yee Ching Tok, Sudipta Chattopadhyay

**Abstract**: Technological advances have enabled multiple countries to consider implementing Smart City Infrastructure to provide in-depth insights into different data points and enhance the lives of citizens. Unfortunately, these new technological implementations also entice adversaries and cybercriminals to execute cyber-attacks and commit criminal acts on these modern infrastructures. Given the borderless nature of cyber attacks, varying levels of understanding of smart city infrastructure and ongoing investigation workloads, law enforcement agencies and investigators would be hard-pressed to respond to these kinds of cybercrime. Without an investigative capability by investigators, these smart infrastructures could become new targets favored by cybercriminals.   To address the challenges faced by investigators, we propose a common definition of smart city infrastructure. Based on the definition, we utilize the STRIDE threat modeling methodology and the Microsoft Threat Modeling Tool to identify threats present in the infrastructure and create a threat model which can be further customized or extended by interested parties. Next, we map offences, possible evidence sources and types of threats identified to help investigators understand what crimes could have been committed and what evidence would be required in their investigation work. Finally, noting that Smart City Infrastructure investigations would be a global multi-faceted challenge, we discuss technical and legal opportunities in digital forensics on Smart City Infrastructure.

摘要: 技术进步使多个国家能够考虑实施智慧城市基础设施，以深入了解不同的数据点，并改善公民的生活。不幸的是，这些新的技术实施也引诱对手和网络罪犯对这些现代基础设施进行网络攻击和犯罪行为。鉴于网络攻击的无边界性质、对智能城市基础设施的不同程度的了解以及正在进行的调查工作量，执法机构和调查人员将很难对此类网络犯罪做出回应。如果调查人员没有调查能力，这些智能基础设施可能会成为网络犯罪分子青睐的新目标。为了应对调查人员面临的挑战，我们提出了智能城市基础设施的共同定义。在定义的基础上，我们利用STRIDE威胁建模方法和Microsoft威胁建模工具来识别基础设施中存在的威胁，并创建可由感兴趣的各方进一步定制或扩展的威胁模型。接下来，我们将绘制罪行、可能的证据来源和已确定的威胁类型的地图，以帮助调查人员了解哪些罪行可能发生，以及在调查工作中需要哪些证据。最后，注意到智能城市基础设施调查将是一项全球多方面的挑战，我们讨论了智能城市基础设施数字取证的技术和法律机会。



## **47. Certified Robustness in Federated Learning**

联合学习中的认证稳健性 cs.LG

Accepted at Workshop on Federated Learning: Recent Advances and New  Challenges, NeurIPS 2022

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2206.02535v2) [paper-pdf](http://arxiv.org/pdf/2206.02535v2)

**Authors**: Motasem Alfarra, Juan C. Pérez, Egor Shulgin, Peter Richtárik, Bernard Ghanem

**Abstract**: Federated learning has recently gained significant attention and popularity due to its effectiveness in training machine learning models on distributed data privately. However, as in the single-node supervised learning setup, models trained in federated learning suffer from vulnerability to imperceptible input transformations known as adversarial attacks, questioning their deployment in security-related applications. In this work, we study the interplay between federated training, personalization, and certified robustness. In particular, we deploy randomized smoothing, a widely-used and scalable certification method, to certify deep networks trained on a federated setup against input perturbations and transformations. We find that the simple federated averaging technique is effective in building not only more accurate, but also more certifiably-robust models, compared to training solely on local data. We further analyze personalization, a popular technique in federated training that increases the model's bias towards local data, on robustness. We show several advantages of personalization over both~(that is, only training on local data and federated training) in building more robust models with faster training. Finally, we explore the robustness of mixtures of global and local~(i.e. personalized) models, and find that the robustness of local models degrades as they diverge from the global model

摘要: 由于联邦学习在训练分布式数据上的机器学习模型方面的有效性，它最近获得了极大的关注和普及。然而，与单节点监督学习设置中一样，在联合学习中训练的模型容易受到称为对抗性攻击的不可察觉的输入转换的影响，从而质疑其在安全相关应用中的部署。在这项工作中，我们研究了联合训练、个性化和经过认证的健壮性之间的相互作用。特别是，我们采用了随机化平滑，这是一种广泛使用和可扩展的认证方法，用于认证在联合设置上训练的深层网络不受输入扰动和转换的影响。我们发现，与仅基于本地数据进行训练相比，简单的联合平均技术不仅在建立更准确的模型方面是有效的，而且在可证明的健壮性方面也更有效。我们进一步分析了个性化，这是联合训练中的一种流行技术，它增加了模型对本地数据的偏差，并对稳健性进行了分析。我们展示了个性化比这两者(即只在本地数据上训练和联合训练)在建立更健壮的模型和更快的训练方面的几个优势。最后，我们研究了全局模型和局部模型(即个性化模型)的混合模型的稳健性，发现局部模型的稳健性随着偏离全局模型而降低



## **48. Short Paper: Static and Microarchitectural ML-Based Approaches For Detecting Spectre Vulnerabilities and Attacks**

短文：基于静态和微体系结构ML的检测Spectre漏洞和攻击的方法 cs.CR

5 pages, 2 figures. Accepted to the Hardware and Architectural  Support for Security and Privacy (HASP'22), in conjunction with the 55th  IEEE/ACM International Symposium on Microarchitecture (MICRO'22)

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14452v1) [paper-pdf](http://arxiv.org/pdf/2210.14452v1)

**Authors**: Chidera Biringa, Gaspard Baye, Gökhan Kul

**Abstract**: Spectre intrusions exploit speculative execution design vulnerabilities in modern processors. The attacks violate the principles of isolation in programs to gain unauthorized private user information. Current state-of-the-art detection techniques utilize micro-architectural features or vulnerable speculative code to detect these threats. However, these techniques are insufficient as Spectre attacks have proven to be more stealthy with recently discovered variants that bypass current mitigation mechanisms. Side-channels generate distinct patterns in processor cache, and sensitive information leakage is dependent on source code vulnerable to Spectre attacks, where an adversary uses these vulnerabilities, such as branch prediction, which causes a data breach. Previous studies predominantly approach the detection of Spectre attacks using the microarchitectural analysis, a reactive approach. Hence, in this paper, we present the first comprehensive evaluation of static and microarchitectural analysis-assisted machine learning approaches to detect Spectre vulnerable code snippets (preventive) and Spectre attacks (reactive). We evaluate the performance trade-offs in employing classifiers for detecting Spectre vulnerabilities and attacks.

摘要: 幽灵入侵利用现代处理器中的推测性执行设计漏洞。这些攻击违反了程序中的隔离原则，以获取未经授权的私人用户信息。当前最先进的检测技术利用微体系结构特征或易受攻击的推测代码来检测这些威胁。然而，这些技术是不够的，因为Spectre攻击已被证明是更隐蔽的，最近发现的变体绕过了当前的缓解机制。侧通道在处理器缓存中生成不同的模式，敏感信息泄漏依赖于易受Spectre攻击的源代码，其中对手使用这些漏洞，如分支预测，从而导致数据泄露。以前的研究主要是使用微体系结构分析来检测Spectre攻击，这是一种反应性方法。因此，在本文中，我们首次对静态和微体系结构分析辅助的机器学习方法进行了全面评估，以检测Spectre易受攻击的代码片段(预防性的)和Spectre攻击(反应性的)。我们评估了使用分类器来检测Spectre漏洞和攻击时的性能权衡。



## **49. LP-BFGS attack: An adversarial attack based on the Hessian with limited pixels**

LP-BFGS攻击：一种基于有限像素黑森的对抗性攻击 cs.CR

5 pages, 4 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.15446v1) [paper-pdf](http://arxiv.org/pdf/2210.15446v1)

**Authors**: Jiebao Zhang, Wenhua Qian, Rencan Nie, Jinde Cao, Dan Xu

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Most white-box attacks are based on the gradient of models to the input. Since the computation and memory budget, adversarial attacks based on the Hessian information are not paid enough attention. In this work, we study the attack performance and computation cost of the attack method based on the Hessian with a limited perturbation pixel number. Specifically, we propose the Limited Pixel BFGS (LP-BFGS) attack method by incorporating the BFGS algorithm. Some pixels are selected as perturbation pixels by the Integrated Gradient algorithm, which are regarded as optimization variables of the LP-BFGS attack. Experimental results across different networks and datasets with various perturbation pixel numbers demonstrate our approach has a comparable attack with an acceptable computation compared with existing solutions.

摘要: 深度神经网络很容易受到敌意攻击。大多数白盒攻击都是基于模型对输入的梯度。由于计算和内存预算的限制，基于黑森信息的对抗性攻击没有引起足够的重视。在这项工作中，我们研究了有限扰动像素数的基于Hessian的攻击方法的攻击性能和计算代价。具体地说，我们结合有限像素BFGS算法，提出了LP-BFGS攻击方法。综合梯度算法选取部分像素点作为扰动像素点，作为LP-BFGS攻击的优化变量。在具有不同扰动像素数的不同网络和数据集上的实验结果表明，该方法具有与现有解决方案相当的攻击能力和可接受的计算量。



## **50. Improving Adversarial Robustness via Joint Classification and Multiple Explicit Detection Classes**

联合分类和多个显式检测类提高敌方鲁棒性 cs.CV

21 pages, 6 figures

**SubmitDate**: 2022-10-26    [abs](http://arxiv.org/abs/2210.14410v1) [paper-pdf](http://arxiv.org/pdf/2210.14410v1)

**Authors**: Sina Baharlouei, Fatemeh Sheikholeslami, Meisam Razaviyayn, Zico Kolter

**Abstract**: This work concerns the development of deep networks that are certifiably robust to adversarial attacks. Joint robust classification-detection was recently introduced as a certified defense mechanism, where adversarial examples are either correctly classified or assigned to the "abstain" class. In this work, we show that such a provable framework can benefit by extension to networks with multiple explicit abstain classes, where the adversarial examples are adaptively assigned to those. We show that naively adding multiple abstain classes can lead to "model degeneracy", then we propose a regularization approach and a training method to counter this degeneracy by promoting full use of the multiple abstain classes. Our experiments demonstrate that the proposed approach consistently achieves favorable standard vs. robust verified accuracy tradeoffs, outperforming state-of-the-art algorithms for various choices of number of abstain classes.

摘要: 这项工作涉及到深度网络的发展，这些网络对对手攻击具有可证明的健壮性。联合稳健分类-检测是最近引入的一种认证防御机制，在这种机制中，对抗性例子要么被正确分类，要么被分配到“弃权”类别。在这项工作中，我们表明这样一个可证明的框架可以通过扩展到具有多个显式弃权类的网络而受益，其中对抗性示例被自适应地分配给那些显式弃权类。我们证明了简单地添加多个弃权类会导致“模型退化”，然后我们提出了一种正则化方法和一种训练方法，通过促进多个弃权类的充分利用来克服这种退化。我们的实验表明，该方法一致地达到了良好的标准和健壮的验证精度折衷，在不同数量的弃权类的选择上优于最新的算法。



