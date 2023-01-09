# Latest Adversarial Attack Papers
**update at 2023-01-09 10:13:54**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Stealthy Backdoor Attack for Code Models**

针对代码模型的隐蔽后门攻击 cs.CR

Under review of IEEE Transactions on Software Engineering

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02496v1) [paper-pdf](http://arxiv.org/pdf/2301.02496v1)

**Authors**: Zhou Yang, Bowen Xu, Jie M. Zhang, Hong Jin Kang, Jieke Shi, Junda He, David Lo

**Abstract**: Code models, such as CodeBERT and CodeT5, offer general-purpose representations of code and play a vital role in supporting downstream automated software engineering tasks. Most recently, code models were revealed to be vulnerable to backdoor attacks. A code model that is backdoor-attacked can behave normally on clean examples but will produce pre-defined malicious outputs on examples injected with triggers that activate the backdoors. Existing backdoor attacks on code models use unstealthy and easy-to-detect triggers. This paper aims to investigate the vulnerability of code models with stealthy backdoor attacks. To this end, we propose AFRAIDOOR (Adversarial Feature as Adaptive Backdoor). AFRAIDOOR achieves stealthiness by leveraging adversarial perturbations to inject adaptive triggers into different inputs. We evaluate AFRAIDOOR on three widely adopted code models (CodeBERT, PLBART and CodeT5) and two downstream tasks (code summarization and method name prediction). We find that around 85% of adaptive triggers in AFRAIDOOR bypass the detection in the defense process. By contrast, only less than 12% of the triggers from previous work bypass the defense. When the defense method is not applied, both AFRAIDOOR and baselines have almost perfect attack success rates. However, once a defense is applied, the success rates of baselines decrease dramatically to 10.47% and 12.06%, while the success rate of AFRAIDOOR are 77.05% and 92.98% on the two tasks. Our finding exposes security weaknesses in code models under stealthy backdoor attacks and shows that the state-of-the-art defense method cannot provide sufficient protection. We call for more research efforts in understanding security threats to code models and developing more effective countermeasures.

摘要: 代码模型，如CodeBERT和CodeT5，提供了代码的通用表示，并在支持下游自动化软件工程任务方面发挥了至关重要的作用。最近，代码模型被发现容易受到后门攻击。被后门攻击的代码模型可以在干净的示例上正常运行，但会在注入了激活后门的触发器的示例上生成预定义的恶意输出。现有对代码模型的后门攻击使用隐蔽且易于检测的触发器。本文旨在研究具有隐蔽后门攻击的代码模型的脆弱性。为此，我们提出了AFRAIDOOR(对抗性特征作为自适应后门)。AFRAIDOOR通过利用对抗性扰动将自适应触发器注入不同的输入来实现隐蔽性。我们在三个广泛采用的代码模型(CodeBERT、PLBART和CodeT5)和两个下游任务(代码摘要和方法名称预测)上对AFRAIDOOR进行了评估。我们发现，AFRAIDOOR中约85%的自适应触发器在防御过程中绕过了检测。相比之下，只有不到12%的以前工作中的触发因素绕过了防御。当不应用防御方法时，AFRAIDOOR和基线都具有几乎完美的攻击成功率。然而，一旦实施防御，基线的成功率急剧下降到10.47%和12.06%，而AFRAIDOOR在两个任务上的成功率分别为77.05%和92.98%。我们的发现暴露了代码模型在秘密后门攻击下的安全漏洞，并表明最先进的防御方法不能提供足够的保护。我们呼吁在了解代码模型的安全威胁和开发更有效的对策方面做出更多研究努力。



## **2. Watching your call: Breaking VoLTE Privacy in LTE/5G Networks**

关注您的电话：打破LTE/5G网络中的VoLTE隐私 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02487v1) [paper-pdf](http://arxiv.org/pdf/2301.02487v1)

**Authors**: Zishuai Cheng, Mihai Ordean, Flavio D. Garcia, Baojiang Cui, Dominik Rys

**Abstract**: Voice over LTE (VoLTE) and Voice over NR (VoNR) are two similar technologies that have been widely deployed by operators to provide a better calling experience in LTE and 5G networks, respectively. The VoLTE/NR protocols rely on the security features of the underlying LTE/5G network to protect users' privacy such that nobody can monitor calls and learn details about call times, duration, and direction. In this paper, we introduce a new privacy attack which enables adversaries to analyse encrypted LTE/5G traffic and recover any VoLTE/NR call details. We achieve this by implementing a novel mobile-relay adversary which is able to remain undetected by using an improved physical layer parameter guessing procedure. This adversary facilitates the recovery of encrypted configuration messages exchanged between victim devices and the mobile network. We further propose an identity mapping method which enables our mobile-relay adversary to link a victim's network identifiers to the phone number efficiently, requiring a single VoLTE protocol message. We evaluate the real-world performance of our attacks using four modern commercial off-the-shelf phones and two representative, commercial network carriers. We collect over 60 hours of traffic between the phones and the mobile networks and execute 160 VoLTE calls, which we use to successfully identify patterns in the physical layer parameter allocation and in VoLTE traffic, respectively. Our real-world experiments show that our mobile-relay works as expected in all test cases, and the VoLTE activity logs recovered describe the actual communication with 100% accuracy. Finally, we show that we can link network identifiers such as International Mobile Subscriber Identities (IMSI), Subscriber Concealed Identifiers (SUCI) and/or Globally Unique Temporary Identifiers (GUTI) to phone numbers while remaining undetected by the victim.

摘要: LTE语音(VoLTE)和NR语音(VoNR)是运营商广泛部署的两项类似技术，分别在LTE和5G网络中提供更好的呼叫体验。VoLTE/NR协议依靠底层LTE/5G网络的安全功能来保护用户隐私，因此没有人可以监控通话并了解有关通话时间、时长和方向的详细信息。在本文中，我们介绍了一种新的隐私攻击，使攻击者能够分析加密的LTE/5G流量，并恢复任何VoLTE/NR呼叫细节。我们通过使用改进的物理层参数猜测过程实现了一种新的移动中继对手，该对手能够保持不被检测到。该敌手便于恢复在受害者设备和移动网络之间交换的加密配置消息。我们进一步提出了一种身份映射方法，使我们的移动中继攻击者能够有效地将受害者的网络标识符链接到电话号码，只需要一条VoLTE协议消息。我们使用四部现代商用现成手机和两家具有代表性的商用网络运营商来评估我们的攻击的真实性能。我们收集电话和移动网络之间超过60小时的流量，并执行160个VoLTE呼叫，我们使用这些呼叫分别成功识别物理层参数分配和VoLTE流量中的模式。实际测试表明，我们的移动中继在所有测试用例中都能正常工作，恢复的VoLTE活动日志对实际通信的描述准确率达到100%。最后，我们展示了我们可以将诸如国际移动用户标识(IMSI)、用户隐藏标识(SUCI)和/或全球唯一临时标识(GUTI)之类的网络标识符链接到电话号码，而不被受害者检测到。



## **3. Adversarial Attacks on Neural Models of Code via Code Difference Reduction**

码差缩减对码神经模型的敌意攻击 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02412v1) [paper-pdf](http://arxiv.org/pdf/2301.02412v1)

**Authors**: Zhao Tian, Junjie Chen, Zhi Jin

**Abstract**: Deep learning has been widely used to solve various code-based tasks by building deep code models based on a large number of code snippets. However, deep code models are still vulnerable to adversarial attacks. As source code is discrete and has to strictly stick to the grammar and semantics constraints, the adversarial attack techniques in other domains are not applicable. Moreover, the attack techniques specific to deep code models suffer from the effectiveness issue due to the enormous attack space. In this work, we propose a novel adversarial attack technique (i.e., CODA). Its key idea is to use the code differences between the target input and reference inputs (that have small code differences but different prediction results with the target one) to guide the generation of adversarial examples. It considers both structure differences and identifier differences to preserve the original semantics. Hence, the attack space can be largely reduced as the one constituted by the two kinds of code differences, and thus the attack process can be largely improved by designing corresponding equivalent structure transformations and identifier renaming transformations. Our experiments on 10 deep code models (i.e., two pre trained models with five code-based tasks) demonstrate the effectiveness and efficiency of CODA, the naturalness of its generated examples, and its capability of defending against attacks after adversarial fine-tuning. For example, CODA improves the state-of-the-art techniques (i.e., CARROT and ALERT) by 79.25% and 72.20% on average in terms of the attack success rate, respectively.

摘要: 深度学习通过基于大量代码片段构建深度代码模型，被广泛用于解决各种基于代码的任务。然而，深层代码模型仍然容易受到敌意攻击。由于源代码是离散的，并且必须严格遵守语法和语义的约束，因此其他领域的对抗性攻击技术不适用。此外，针对深层代码模型的攻击技术由于攻击空间巨大而存在有效性问题。在这项工作中，我们提出了一种新的对抗性攻击技术(即CODA)。它的核心思想是利用目标输入和参考输入之间的编码差异(编码差异很小，但预测结果与目标输入不同)来指导对抗性实例的生成。它同时考虑了结构差异和标识差异，以保持原有的语义。因此，通过设计相应的等价结构变换和标识符重命名变换，可以将攻击空间大大缩减为由两种代码差异构成的攻击空间，从而大大改善了攻击过程。我们在10个深层代码模型(即两个具有5个基于代码的任务的预训练模型)上的实验证明了CODA的有效性和高效性、生成的示例的自然性以及经过对抗性微调后的抵御攻击的能力。例如，在攻击成功率方面，CODA将最先进的技术(即胡萝卜和警报)平均分别提高了79.25%和72.20%。



## **4. TrojanPuzzle: Covertly Poisoning Code-Suggestion Models**

特洛伊木马之谜：秘密中毒代码-建议模型 cs.CR

**SubmitDate**: 2023-01-06    [abs](http://arxiv.org/abs/2301.02344v1) [paper-pdf](http://arxiv.org/pdf/2301.02344v1)

**Authors**: Hojjat Aghakhani, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, Robert Sim

**Abstract**: With tools like GitHub Copilot, automatic code suggestion is no longer a dream in software engineering. These tools, based on large language models, are typically trained on massive corpora of code mined from unvetted public sources. As a result, these models are susceptible to data poisoning attacks where an adversary manipulates the model's training or fine-tuning phases by injecting malicious data. Poisoning attacks could be designed to influence the model's suggestions at run time for chosen contexts, such as inducing the model into suggesting insecure code payloads. To achieve this, prior poisoning attacks explicitly inject the insecure code payload into the training data, making the poisoning data detectable by static analysis tools that can remove such malicious data from the training set. In this work, we demonstrate two novel data poisoning attacks, COVERT and TROJANPUZZLE, that can bypass static analysis by planting malicious poisoning data in out-of-context regions such as docstrings. Our most novel attack, TROJANPUZZLE, goes one step further in generating less suspicious poisoning data by never including certain (suspicious) parts of the payload in the poisoned data, while still inducing a model that suggests the entire payload when completing code (i.e., outside docstrings). This makes TROJANPUZZLE robust against signature-based dataset-cleansing methods that identify and filter out suspicious sequences from the training data. Our evaluation against two model sizes demonstrates that both COVERT and TROJANPUZZLE have significant implications for how practitioners should select code used to train or tune code-suggestion models.

摘要: 有了GitHub Copilot这样的工具，自动代码建议不再是软件工程中的梦想。这些工具基于大型语言模型，通常针对从未经审查的公共来源挖掘的大量代码语料库进行培训。因此，这些模型容易受到数据中毒攻击，即对手通过注入恶意数据来操纵模型的训练或微调阶段。毒化攻击可以被设计成影响模型在运行时对所选上下文的建议，例如诱导模型建议不安全的代码有效负载。为了实现这一点，先前的中毒攻击显式地将不安全的代码有效载荷注入到训练数据中，使得可以从训练集中移除此类恶意数据的静态分析工具可以检测到中毒数据。在这项工作中，我们展示了两种新型的数据中毒攻击：ASTIFT和TROJANPUZLE，它们可以通过在文档字符串等脱离上下文的区域植入恶意中毒数据来绕过静态分析。我们最新颖的攻击TROJANPUZLE在生成不那么可疑的中毒数据方面又向前迈进了一步，它从未在有毒数据中包括有效负载的某些(可疑)部分，同时仍诱导出一个模型，该模型在完成代码时(即在文档字符串外部)建议整个有效负载。这使得TROJANPUZLE对于基于签名的数据集清理方法具有健壮性，这些方法从训练数据中识别和过滤可疑序列。我们对两个模型大小的评估表明，COVERT和TROJANPUZLE对于实践者应该如何选择用于训练或调优代码建议模型的代码具有重要影响。



## **5. Silent Killer: Optimizing Backdoor Trigger Yields a Stealthy and Powerful Data Poisoning Attack**

无声杀手：优化后门触发器可产生隐形且强大的数据中毒攻击 cs.CR

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02615v1) [paper-pdf](http://arxiv.org/pdf/2301.02615v1)

**Authors**: Tzvi Lederer, Gallil Maimon, Lior Rokach

**Abstract**: We propose a stealthy and powerful backdoor attack on neural networks based on data poisoning (DP). In contrast to previous attacks, both the poison and the trigger in our method are stealthy. We are able to change the model's classification of samples from a source class to a target class chosen by the attacker. We do so by using a small number of poisoned training samples with nearly imperceptible perturbations, without changing their labels. At inference time, we use a stealthy perturbation added to the attacked samples as a trigger. This perturbation is crafted as a universal adversarial perturbation (UAP), and the poison is crafted using gradient alignment coupled to this trigger. Our method is highly efficient in crafting time compared to previous methods and requires only a trained surrogate model without additional retraining. Our attack achieves state-of-the-art results in terms of attack success rate while maintaining high accuracy on clean samples.

摘要: 提出了一种基于数据毒化(DP)的隐蔽而强大的神经网络后门攻击方法。与以前的攻击不同，我们方法中的毒药和触发器都是隐蔽的。我们能够将模型的样本分类从源类更改为攻击者选择的目标类。我们通过使用少量具有几乎不可察觉的扰动的有毒训练样本来做到这一点，而不改变它们的标签。在推断时，我们使用添加到被攻击样本的隐形扰动作为触发器。该扰动被精心设计为通用对抗性扰动(UAP)，并且毒药是使用与该触发器耦合的梯度对齐来定制的。与以前的方法相比，我们的方法在计算时间上具有很高的效率，并且只需要一个经过训练的代理模型，而不需要额外的重新训练。我们的攻击在攻击成功率方面实现了最先进的结果，同时保持了对干净样本的高精度。



## **6. Holistic Adversarial Robustness of Deep Learning Models**

深度学习模型的整体对抗稳健性 cs.LG

survey paper on holistic adversarial robustness for deep learning;  published at AAAI 2023 Senior Member Presentation Track

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2202.07201v3) [paper-pdf](http://arxiv.org/pdf/2202.07201v3)

**Authors**: Pin-Yu Chen, Sijia Liu

**Abstract**: Adversarial robustness studies the worst-case performance of a machine learning model to ensure safety and reliability. With the proliferation of deep-learning-based technology, the potential risks associated with model development and deployment can be amplified and become dreadful vulnerabilities. This paper provides a comprehensive overview of research topics and foundational principles of research methods for adversarial robustness of deep learning models, including attacks, defenses, verification, and novel applications.

摘要: 对抗健壮性研究机器学习模型的最坏情况下的性能，以确保安全性和可靠性。随着基于深度学习的技术的激增，与模型开发和部署相关的潜在风险可能会放大，并成为可怕的漏洞。本文全面综述了深度学习模型对抗性稳健性的研究主题和基本原理，包括攻击、防御、验证和新的应用。



## **7. Randomized Message-Interception Smoothing: Gray-box Certificates for Graph Neural Networks**

随机消息拦截平滑：图神经网络的灰箱证书 cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.02039v1) [paper-pdf](http://arxiv.org/pdf/2301.02039v1)

**Authors**: Yan Scholten, Jan Schuchardt, Simon Geisler, Aleksandar Bojchevski, Stephan Günnemann

**Abstract**: Randomized smoothing is one of the most promising frameworks for certifying the adversarial robustness of machine learning models, including Graph Neural Networks (GNNs). Yet, existing randomized smoothing certificates for GNNs are overly pessimistic since they treat the model as a black box, ignoring the underlying architecture. To remedy this, we propose novel gray-box certificates that exploit the message-passing principle of GNNs: We randomly intercept messages and carefully analyze the probability that messages from adversarially controlled nodes reach their target nodes. Compared to existing certificates, we certify robustness to much stronger adversaries that control entire nodes in the graph and can arbitrarily manipulate node features. Our certificates provide stronger guarantees for attacks at larger distances, as messages from farther-away nodes are more likely to get intercepted. We demonstrate the effectiveness of our method on various models and datasets. Since our gray-box certificates consider the underlying graph structure, we can significantly improve certifiable robustness by applying graph sparsification.

摘要: 随机化平滑是证明机器学习模型(包括图神经网络)对抗稳健性的最有前途的框架之一。然而，现有的用于GNN的随机化平滑证书过于悲观，因为它们将模型视为黑匣子，忽略了底层架构。为了解决这个问题，我们提出了一种新的灰盒证书，它利用了GNN的消息传递原理：我们随机截获消息，并仔细分析来自恶意控制节点的消息到达目标节点的概率。与现有的证书相比，我们证明了对控制图中的整个节点并可以任意操纵节点特征的更强大的攻击者的健壮性。我们的证书为更远距离的攻击提供了更强有力的保证，因为来自较远节点的消息更有可能被拦截。我们在不同的模型和数据集上演示了我们的方法的有效性。由于我们的灰盒证书考虑了底层的图结构，所以我们可以通过应用图稀疏来显著提高可证明的健壮性。



## **8. Beckman Defense**

贝克曼辩护 cs.LG

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01495v2) [paper-pdf](http://arxiv.org/pdf/2301.01495v2)

**Authors**: A. V. Subramanyam

**Abstract**: Optimal transport (OT) based distributional robust optimisation (DRO) has received some traction in the recent past. However, it is at a nascent stage but has a sound potential in robustifying the deep learning models. Interestingly, OT barycenters demonstrate a good robustness against adversarial attacks. Owing to the computationally expensive nature of OT barycenters, they have not been investigated under DRO framework. In this work, we propose a new barycenter, namely Beckman barycenter, which can be computed efficiently and used for training the network to defend against adversarial attacks in conjunction with adversarial training. We propose a novel formulation of Beckman barycenter and analytically obtain the barycenter using the marginals of the input image. We show that the Beckman barycenter can be used to train adversarially trained networks to improve the robustness. Our training is extremely efficient as it requires only a single epoch of training. Elaborate experiments on CIFAR-10, CIFAR-100 and Tiny ImageNet demonstrate that training an adversarially robust network with Beckman barycenter can significantly increase the performance. Under auto attack, we get a a maximum boost of 10\% in CIFAR-10, 8.34\% in CIFAR-100 and 11.51\% in Tiny ImageNet. Our code is available at https://github.com/Visual-Conception-Group/test-barycentric-defense.

摘要: 最近，基于最优传输(OT)的分布式稳健优化(DRO)受到了一些关注。然而，它还处于初级阶段，但在推动深度学习模式方面具有良好的潜力。有趣的是，OT重心对敌方攻击表现出良好的健壮性。由于OT重心的计算代价很高，因此尚未在DRO框架下对其进行研究。在这项工作中，我们提出了一种新的重心，即Beckman重心，它可以被有效地计算出来，并用于训练网络在对抗训练的同时防御对手攻击。我们提出了一种新的Beckman重心公式，并利用输入图像的边缘来解析地获得重心。我们证明了Beckman重心可以用于训练对抗性训练的网络，以提高网络的健壮性。我们的训练非常有效，因为它只需要一个时期的训练。在CIFAR-10、CIFAR-100和Tiny ImageNet上的详细实验表明，使用Beckman重心训练一个对抗健壮的网络可以显著提高性能。在AUTO攻击下，CIFAR-10、CIFAR-100和TING ImageNet的最大性能提升分别为10%、8.34%和11.51%。我们的代码可以在https://github.com/Visual-Conception-Group/test-barycentric-defense.上找到



## **9. Enhancement attacks in biomedical machine learning**

生物医学机器学习中的增强攻击 stat.ML

13 pages, 3 figures

**SubmitDate**: 2023-01-05    [abs](http://arxiv.org/abs/2301.01885v1) [paper-pdf](http://arxiv.org/pdf/2301.01885v1)

**Authors**: Matthew Rosenblatt, Javid Dadashkarimi, Dustin Scheinost

**Abstract**: The prevalence of machine learning in biomedical research is rapidly growing, yet the trustworthiness of such research is often overlooked. While some previous works have investigated the ability of adversarial attacks to degrade model performance in medical imaging, the ability to falsely improve performance via recently-developed "enhancement attacks" may be a greater threat to biomedical machine learning. In the spirit of developing attacks to better understand trustworthiness, we developed three techniques to drastically enhance prediction performance of classifiers with minimal changes to features, including the enhancement of 1) within-dataset predictions, 2) a particular method over another, and 3) cross-dataset generalization. Our within-dataset enhancement framework falsely improved classifiers' accuracy from 50% to almost 100% while maintaining high feature similarities between original and enhanced data (Pearson's r's>0.99). Similarly, the method-specific enhancement framework was effective in falsely improving the performance of one method over another. For example, a simple neural network outperformed LR by 50% on our enhanced dataset, although no performance differences were present in the original dataset. Crucially, the original and enhanced data were still similar (r=0.95). Finally, we demonstrated that enhancement is not specific to within-dataset predictions but can also be adapted to enhance the generalization accuracy of one dataset to another by up to 38%. Overall, our results suggest that more robust data sharing and provenance tracking pipelines are necessary to maintain data integrity in biomedical machine learning research.

摘要: 机器学习在生物医学研究中的盛行正在迅速增长，但此类研究的可信度往往被忽视。虽然以前的一些工作已经研究了对抗性攻击降低医学成像中模型性能的能力，但通过最近开发的增强攻击来错误地提高性能的能力可能会对生物医学机器学习构成更大的威胁。本着开发攻击以更好地理解可信度的精神，我们开发了三种技术来在对特征进行最小更改的情况下显著提高分类器的预测性能，包括1)数据集内预测的增强，2)一种特定方法优于另一种方法，以及3)跨数据集泛化。我们的数据集内增强框架错误地将分类器的准确率从50%提高到几乎100%，同时保持了原始数据和增强数据之间的高度特征相似性(Pearson‘s r’s>0.99)。类似地，特定于方法的增强框架在错误地改进一种方法的性能方面是有效的。例如，一个简单的神经网络在我们的增强数据集上的性能比LR高50%，尽管在原始数据集中没有表现出性能差异。重要的是，原始数据和增强数据仍然相似(r=0.95)。最后，我们证明了增强并不特定于数据集内的预测，但也可以用于将一个数据集到另一个数据集的泛化精度提高高达38%。总体而言，我们的结果表明，在生物医学机器学习研究中，为了保持数据的完整性，需要更健壮的数据共享和来源跟踪管道。



## **10. Availability Adversarial Attack and Countermeasures for Deep Learning-based Load Forecasting**

基于深度学习的负荷预测可用性攻击与对策 cs.LG

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01832v1) [paper-pdf](http://arxiv.org/pdf/2301.01832v1)

**Authors**: Wangkun Xu, Fei Teng

**Abstract**: The forecast of electrical loads is essential for the planning and operation of the power system. Recently, advances in deep learning have enabled more accurate forecasts. However, deep neural networks are prone to adversarial attacks. Although most of the literature focuses on integrity-based attacks, this paper proposes availability-based adversarial attacks, which can be more easily implemented by attackers. For each forecast instance, the availability attack position is optimally solved by mixed-integer reformulation of the artificial neural network. To tackle this attack, an adversarial training algorithm is proposed. In simulation, a realistic load forecasting dataset is considered and the attack performance is compared to the integrity-based attack. Meanwhile, the adversarial training algorithm is shown to significantly improve robustness against availability attacks. All codes are available at https://github.com/xuwkk/AAA_Load_Forecast.

摘要: 电力负荷预测对于电力系统的规划和运行是至关重要的。最近，深度学习的进步使预测更加准确。然而，深度神经网络容易受到对抗性攻击。虽然大多数文献关注的是基于完整性的攻击，但本文提出的基于可用性的对抗性攻击更容易被攻击者实现。对于每个预测实例，通过人工神经网络的混合整数重构来最优地求解可用攻击位置。针对这种攻击，提出了一种对抗性训练算法。在仿真中，考虑了真实的负载预测数据集，并将其攻击性能与基于完整性的攻击进行了比较。同时，对抗性训练算法显著提高了对可用性攻击的健壮性。所有代码均可在https://github.com/xuwkk/AAA_Load_Forecast.上获得。



## **11. GUAP: Graph Universal Attack Through Adversarial Patching**

GUAP：通过对抗性补丁实现通用攻击 cs.LG

8 pages

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01731v1) [paper-pdf](http://arxiv.org/pdf/2301.01731v1)

**Authors**: Xiao Zang, Jie Chen, Bo Yuan

**Abstract**: Graph neural networks (GNNs) are a class of effective deep learning models for node classification tasks; yet their predictive capability may be severely compromised under adversarially designed unnoticeable perturbations to the graph structure and/or node data. Most of the current work on graph adversarial attacks aims at lowering the overall prediction accuracy, but we argue that the resulting abnormal model performance may catch attention easily and invite quick counterattack. Moreover, attacks through modification of existing graph data may be hard to conduct if good security protocols are implemented. In this work, we consider an easier attack harder to be noticed, through adversarially patching the graph with new nodes and edges. The attack is universal: it targets a single node each time and flips its connection to the same set of patch nodes. The attack is unnoticeable: it does not modify the predictions of nodes other than the target. We develop an algorithm, named GUAP, that achieves high attack success rate but meanwhile preserves the prediction accuracy. GUAP is fast to train by employing a sampling strategy. We demonstrate that a 5% sampling in each epoch yields 20x speedup in training, with only a slight degradation in attack performance. Additionally, we show that the adversarial patch trained with the graph convolutional network transfers well to other GNNs, such as the graph attention network.

摘要: 图神经网络(GNN)是一类用于节点分类任务的有效深度学习模型，但在图结构和/或节点数据受到恶意设计的不可察觉扰动时，其预测能力可能会受到严重影响。目前关于图对抗攻击的大部分工作都是为了降低整体预测精度，但我们认为，由此产生的异常模型性能可能容易引起注意并引发快速反击。此外，如果实施了良好的安全协议，通过修改现有图形数据进行的攻击可能很难进行。在这项工作中，我们认为更容易的攻击更难被注意，通过用新的节点和边恶意修补图。这种攻击是通用的：它每次只针对一个节点，并将其连接反转到同一组补丁节点。攻击是不可察觉的：它不会修改除目标之外的其他节点的预测。提出了一种在保持预测精度的同时获得较高攻击成功率的GUAP算法。通过采用抽样策略，GAP的训练速度很快。我们证明，在每个时期进行5%的采样可以在训练中获得20倍的加速，而攻击性能只有轻微的下降。此外，我们还证明了用图卷积网络训练的敌意补丁可以很好地移植到其他GNN上，例如图注意网络。



## **12. A Survey on Physical Adversarial Attack in Computer Vision**

计算机视觉中的身体对抗攻击研究综述 cs.CV

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2209.14262v2) [paper-pdf](http://arxiv.org/pdf/2209.14262v2)

**Authors**: Donghua Wang, Wen Yao, Tingsong Jiang, Guijian Tang, Xiaoqian Chen

**Abstract**: In the past decade, deep learning has dramatically changed the traditional hand-craft feature manner with strong feature learning capability, resulting in tremendous improvement of conventional tasks. However, deep neural networks have recently been demonstrated vulnerable to adversarial examples, a kind of malicious samples crafted by small elaborately designed noise, which mislead the DNNs to make the wrong decisions while remaining imperceptible to humans. Adversarial examples can be divided into digital adversarial attacks and physical adversarial attacks. The digital adversarial attack is mostly performed in lab environments, focusing on improving the performance of adversarial attack algorithms. In contrast, the physical adversarial attack focus on attacking the physical world deployed DNN systems, which is a more challenging task due to the complex physical environment (i.e., brightness, occlusion, and so on). Although the discrepancy between digital adversarial and physical adversarial examples is small, the physical adversarial examples have a specific design to overcome the effect of the complex physical environment. In this paper, we review the development of physical adversarial attacks in DNN-based computer vision tasks, including image recognition tasks, object detection tasks, and semantic segmentation. For the sake of completeness of the algorithm evolution, we will briefly introduce the works that do not involve the physical adversarial attack. We first present a categorization scheme to summarize the current physical adversarial attacks. Then discuss the advantages and disadvantages of the existing physical adversarial attacks and focus on the technique used to maintain the adversarial when applied into physical environment. Finally, we point out the issues of the current physical adversarial attacks to be solved and provide promising research directions.

摘要: 在过去的十年里，深度学习以其强大的特征学习能力，极大地改变了传统的手工特征学习方式，使常规任务得到了极大的改善。然而，深度神经网络最近被证明容易受到敌意例子的攻击，这是一种由精心设计的小噪声制作的恶意样本，它误导DNN做出错误的决定，同时保持对人类的不可察觉。对抗性攻击可分为数字对抗性攻击和物理对抗性攻击。数字对抗攻击大多在实验室环境中进行，致力于提高对抗攻击算法的性能。相比之下，物理对抗性攻击侧重于攻击物理世界中部署的DNN系统，由于物理环境复杂(即亮度、遮挡等)，这是一项更具挑战性的任务。虽然数字对抗例子和物理对抗例子之间的差异很小，但物理对抗例子有一个特定的设计来克服复杂物理环境的影响。本文回顾了基于DNN的计算机视觉任务中物理对抗攻击的发展，包括图像识别任务、目标检测任务和语义分割任务。为了算法演化的完备性，我们将简要介绍不涉及物理对抗攻击的工作。我们首先提出了一种分类方案来总结当前的物理对抗性攻击。然后讨论了现有物理对抗性攻击的优缺点，并重点介绍了应用于物理环境中维护对抗性的技术。最后，指出了当前物理对抗性攻击需要解决的问题，并提出了有前景的研究方向。



## **13. Validity in Music Information Research Experiments**

音乐信息研究实验中的效度 cs.SD

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01578v1) [paper-pdf](http://arxiv.org/pdf/2301.01578v1)

**Authors**: Bob L. T. Sturm, Arthur Flexer

**Abstract**: Validity is the truth of an inference made from evidence, such as data collected in an experiment, and is central to working scientifically. Given the maturity of the domain of music information research (MIR), validity in our opinion should be discussed and considered much more than it has been so far. Considering validity in one's work can improve its scientific and engineering value. Puzzling MIR phenomena like adversarial attacks and performance glass ceilings become less mysterious through the lens of validity. In this article, we review the subject of validity in general, considering the four major types of validity from a key reference: Shadish et al. 2002. We ground our discussion of these types with a prototypical MIR experiment: music classification using machine learning. Through this MIR experimentalists can be guided to make valid inferences from data collected from their experiments.

摘要: 有效性是从证据中做出的推论的真实性，例如在实验中收集的数据，它是科学工作的核心。鉴于音乐信息研究(MIR)领域的成熟，我们认为应该比目前更多地讨论和考虑有效性。在工作中考虑有效性可以提高工作的科学价值和工程价值。令人费解的MIR现象，如对抗性攻击和性能玻璃天花板，通过有效性的镜头变得不那么神秘。在这篇文章中，我们大体回顾了有效性的主题，从一个关键的参考文献考虑了四种主要的有效性类型：Shaish等人。2002年。我们用一个典型的MIR实验来讨论这些类型：使用机器学习的音乐分类。通过这一实验，可以指导实验者从他们的实验中收集的数据中做出有效的推断。



## **14. Passive Triangulation Attack on ORide**

ORIDE上的被动三角剖分攻击 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2208.12216v3) [paper-pdf](http://arxiv.org/pdf/2208.12216v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.

摘要: 网约车服务中的隐私保护旨在保护司机和乘客的隐私。ORIDE是USENIX安全研讨会2017上发布的早期RHS提案之一。在ORIDE协议中，在区域中操作的乘客和司机使用某种同态加密方案(SHE)加密他们的位置，并将其转发给服务提供商(SP)。SP同态计算乘客和可用司机之间的平方欧几里得距离。骑手收到加密的距离，解密后选择最优的骑手。为了防止三角测量攻击，SP在将距离发送给骑手之前随机排列距离。在这项工作中，我们使用了一种被动攻击，该攻击使用三角测量来确定所有参与的司机的坐标，这些司机的置换距离是从多个诚实但好奇的对手车手的角度出发的。对ORide的攻击在SAC 2021上发表。同时提出了一种利用噪声欧几里德距离来阻止他们攻击的对策。当给定司机与多个参考点的置换和噪声欧几里德距离时，我们将我们的攻击扩展到确定司机的位置，其中噪声扰动来自均匀分布。我们对不同数量的驱动器和不同的摄动值进行了实验。我们的实验表明，我们可以确定所有参与ORIDE协议的司机的位置。对于受干扰的距离版本的ORide协议，我们的算法显示了大约25%到50%的参与司机的位置。我们的算法以时间多项式的形式运行在驱动器的数量上。



## **15. The Feasibility and Inevitability of Stealth Attacks**

隐形攻击的可行性和必然性 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2106.13997v4) [paper-pdf](http://arxiv.org/pdf/2106.13997v4)

**Authors**: Ivan Y. Tyukin, Desmond J. Higham, Alexander Bastounis, Eliyas Woldegeorgis, Alexander N. Gorban

**Abstract**: We develop and study new adversarial perturbations that enable an attacker to gain control over decisions in generic Artificial Intelligence (AI) systems including deep learning neural networks. In contrast to adversarial data modification, the attack mechanism we consider here involves alterations to the AI system itself. Such a stealth attack could be conducted by a mischievous, corrupt or disgruntled member of a software development team. It could also be made by those wishing to exploit a ``democratization of AI'' agenda, where network architectures and trained parameter sets are shared publicly. We develop a range of new implementable attack strategies with accompanying analysis, showing that with high probability a stealth attack can be made transparent, in the sense that system performance is unchanged on a fixed validation set which is unknown to the attacker, while evoking any desired output on a trigger input of interest. The attacker only needs to have estimates of the size of the validation set and the spread of the AI's relevant latent space. In the case of deep learning neural networks, we show that a one neuron attack is possible - a modification to the weights and bias associated with a single neuron - revealing a vulnerability arising from over-parameterization. We illustrate these concepts using state of the art architectures on two standard image data sets. Guided by the theory and computational results, we also propose strategies to guard against stealth attacks.

摘要: 我们开发和研究了新的对抗性扰动，使攻击者能够控制通用人工智能(AI)系统中的决策，包括深度学习神经网络。与对抗性数据修改不同，我们在这里考虑的攻击机制涉及对AI系统本身的更改。这种隐形攻击可能是由软件开发团队中调皮的、腐败的或心怀不满的成员实施的。它也可以由那些希望利用“人工智能民主化”议程的人提出，在这种议程中，网络架构和经过训练的参数集是公开共享的。我们开发了一系列新的可实现的攻击策略，并进行了分析，表明在高概率情况下，隐形攻击可以变得透明，也就是说，在攻击者未知的固定验证集上，系统性能不变，同时在感兴趣的触发器输入上唤起任何期望的输出。攻击者只需要对验证集的大小和人工智能相关潜在空间的传播进行估计。在深度学习神经网络的情况下，我们证明了一个神经元攻击是可能的--对与单个神经元相关的权重和偏差的修改--揭示了由于过度参数化而产生的漏洞。我们在两个标准图像数据集上使用最先进的体系结构来说明这些概念。在理论和计算结果的指导下，我们还提出了防范隐身攻击的策略。



## **16. Driver Locations Harvesting Attack on pRide**

司机位置收割对Pride的攻击 cs.CR

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2210.13263v3) [paper-pdf](http://arxiv.org/pdf/2210.13263v3)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstract**: Privacy preservation in Ride-Hailing Services (RHS) is intended to protect privacy of drivers and riders. pRide, published in IEEE Trans. Vehicular Technology 2021, is a prediction based privacy-preserving RHS protocol to match riders with an optimum driver. In the protocol, the Service Provider (SP) homomorphically computes Euclidean distances between encrypted locations of drivers and rider. Rider selects an optimum driver using decrypted distances augmented by a new-ride-emergence prediction. To improve the effectiveness of driver selection, the paper proposes an enhanced version where each driver gives encrypted distances to each corner of her grid. To thwart a rider from using these distances to launch an inference attack, the SP blinds these distances before sharing them with the rider. In this work, we propose a passive attack where an honest-but-curious adversary rider who makes a single ride request and receives the blinded distances from SP can recover the constants used to blind the distances. Using the unblinded distances, rider to driver distance and Google Nearest Road API, the adversary can obtain the precise locations of responding drivers. We conduct experiments with random on-road driver locations for four different cities. Our experiments show that we can determine the precise locations of at least 80% of the drivers participating in the enhanced pRide protocol.

摘要: 网约车服务(RHS)中的隐私保护旨在保护司机和乘客的隐私。Pride，发表在IEEE Trans上。Vehicular Technology 2021是一种基于预测的隐私保护RHS协议，用于将乘客与最佳司机进行匹配。在该协议中，服务提供商(SP)同态地计算司机和乘客的加密位置之间的欧几里德距离。骑手使用解密的距离选择最优的司机，并增加了一个新的乘车出现预测。为了提高驾驶员选择的有效性，本文提出了一种增强版本，每个驾驶员给出了到其网格每个角落的加密距离。为了阻止骑手使用这些距离来发动推理攻击，SP在与骑手共享这些距离之前会先隐藏这些距离。在这项工作中，我们提出了一种被动攻击，在这种攻击中，诚实但好奇的敌方骑手发出一个骑行请求，并从SP接收到盲距离，就可以恢复用于盲距离的常量。使用非盲目距离、骑手到司机的距离和谷歌最近道路API，对手可以获得回应司机的准确位置。我们对四个不同城市的随机道路司机位置进行了实验。我们的实验表明，我们可以确定至少80%参与增强PROID协议的司机的准确位置。



## **17. Organised Firestorm as strategy for business cyber-attacks**

有组织的火暴作为商业网络攻击的战略 cs.CY

9 pages, 3 figures, 2 table

**SubmitDate**: 2023-01-04    [abs](http://arxiv.org/abs/2301.01518v1) [paper-pdf](http://arxiv.org/pdf/2301.01518v1)

**Authors**: Andrea Russo

**Abstract**: Having a good reputation is paramount for most organisations and companies. In fact, having an optimal corporate image allows them to have better transaction relationships with various customers and partners. However, such reputation is hard to build and easy to destroy for all kind of business commercial activities (B2C, B2B, B2B2C, B2G). A misunderstanding during the communication process to the customers, or just a bad communication strategy, can lead to a disaster for the entire company. This is emphasised by the reaction of millions of people on social networks, which can be very detrimental for the corporate image if they react negatively to a certain event. This is called a firestorm.   In this paper, I propose a well-organised strategy for firestorm attacks on organisations, also showing how an adversary can leverage them to obtain private information on the attacked firm. Standard business security procedures are not designed to operate against multi-domain attacks; therefore, I will show how it is possible to bypass the classic and advised security procedures by operating different kinds of attack. I also propose a different firestorm attack, targeting a specific business company network in an efficient way. Finally, I present defensive procedures to reduce the negative effect of firestorms on a company.

摘要: 对大多数组织和公司来说，拥有良好的声誉是最重要的。事实上，拥有最佳的公司形象可以让他们与各种客户和合作伙伴建立更好的交易关系。然而，对于所有类型的商业商业活动(B2C、B2B、B2B2C、B2G)来说，这样的声誉很难建立，也很容易被摧毁。在与客户沟通的过程中产生误解，或者只是沟通策略不当，都可能给整个公司带来灾难。数以百万计的人在社交网络上的反应突显了这一点，如果他们对某个事件做出负面反应，这可能会对公司形象造成非常不利的影响。这被称为大火风暴。在这篇文章中，我提出了一种组织严密的战略，以应对对组织的火暴攻击，并展示了对手如何利用它们来获取被攻击公司的私人信息。标准的业务安全程序不是为抵御多域攻击而设计的；因此，我将展示如何通过操作不同类型的攻击来绕过经典的和建议的安全程序。我还提出了一种不同的FireStorm攻击，以一种高效的方式针对特定的商业公司网络。最后，我提出了一些防御措施，以减少风暴对公司的负面影响。



## **18. Universal adversarial perturbation for remote sensing images**

遥感图像的普遍对抗性摄动 cs.CV

Published in the Twenty-Fourth International Workshop on Multimedia  Signal Processing, MMSP 2022

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2202.10693v2) [paper-pdf](http://arxiv.org/pdf/2202.10693v2)

**Authors**: Qingyu Wang, Guorui Feng, Zhaoxia Yin, Bin Luo

**Abstract**: Recently, with the application of deep learning in the remote sensing image (RSI) field, the classification accuracy of the RSI has been dramatically improved compared with traditional technology. However, even the state-of-the-art object recognition convolutional neural networks are fooled by the universal adversarial perturbation (UAP). The research on UAP is mostly limited to ordinary images, and RSIs have not been studied. To explore the basic characteristics of UAPs of RSIs, this paper proposes a novel method combining an encoder-decoder network with an attention mechanism to generate the UAP of RSIs. Firstly, the former is used to generate the UAP, which can learn the distribution of perturbations better, and then the latter is used to find the sensitive regions concerned by the RSI classification model. Finally, the generated regions are used to fine-tune the perturbation making the model misclassified with fewer perturbations. The experimental results show that the UAP can make the classification model misclassify, and the attack success rate of our proposed method on the RSI data set is as high as 97.09%.

摘要: 近年来，随着深度学习在遥感图像领域的应用，遥感图像的分类精度与传统技术相比有了很大的提高。然而，即使是最先进的目标识别卷积神经网络也被通用对抗性摄动(UAP)愚弄了。对UAP的研究大多局限于普通图像，对RIS的研究尚未见报道。为了探索RSIS的UAP的基本特征，提出了一种将编解码器网络和注意力机制相结合的RSIS UAP生成方法。首先利用前者生成能更好地学习扰动分布的UAP，然后利用后者寻找RSI分类模型所关注的敏感区域。最后，生成的区域被用来微调扰动，使得模型在较少扰动的情况下被误分类。实验结果表明，UAP能够使分类模型发生误分类，本文提出的方法在RSI数据集上的攻击成功率高达97.09%。



## **19. Surveillance Face Anti-spoofing**

监控面反欺骗 cs.CV

15 pages, 9 figures

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00975v1) [paper-pdf](http://arxiv.org/pdf/2301.00975v1)

**Authors**: Hao Fang, Ajian Liu, Jun Wan, Sergio Escalera, Chenxu Zhao, Xu Zhang, Stan Z. Li, Zhen Lei

**Abstract**: Face Anti-spoofing (FAS) is essential to secure face recognition systems from various physical attacks. However, recent research generally focuses on short-distance applications (i.e., phone unlocking) while lacking consideration of long-distance scenes (i.e., surveillance security checks). In order to promote relevant research and fill this gap in the community, we collect a large-scale Surveillance High-Fidelity Mask (SuHiFiMask) dataset captured under 40 surveillance scenes, which has 101 subjects from different age groups with 232 3D attacks (high-fidelity masks), 200 2D attacks (posters, portraits, and screens), and 2 adversarial attacks. In this scene, low image resolution and noise interference are new challenges faced in surveillance FAS. Together with the SuHiFiMask dataset, we propose a Contrastive Quality-Invariance Learning (CQIL) network to alleviate the performance degradation caused by image quality from three aspects: (1) An Image Quality Variable module (IQV) is introduced to recover image information associated with discrimination by combining the super-resolution network. (2) Using generated sample pairs to simulate quality variance distributions to help contrastive learning strategies obtain robust feature representation under quality variation. (3) A Separate Quality Network (SQN) is designed to learn discriminative features independent of image quality. Finally, a large number of experiments verify the quality of the SuHiFiMask dataset and the superiority of the proposed CQIL.

摘要: 人脸反欺骗技术是保护人脸识别系统免受各种物理攻击的重要手段。然而，目前的研究大多集中在短距离应用(如手机解锁)上，而缺乏对远程场景(如监控安检)的考虑。为了推动相关研究，填补社区这一空白，我们收集了40个监控场景下的大规模监控高保真面具(SuHiFiMASK)数据集，其中包含来自不同年龄段的101名受试者，分别进行了232次3D攻击(高保真面具)、200次2D攻击(海报、肖像和屏幕)和2次对抗性攻击。在这种情况下，图像分辨率低和噪声干扰是自动监控系统面临的新挑战。结合SuHiFiMASK数据集，我们从三个方面提出了一种对比质量-不变性学习(CQIL)网络来缓解图像质量引起的性能下降：(1)引入图像质量变量模块(IQV)，通过结合超分辨率网络来恢复与区分相关的图像信息。(2)使用生成的样本对来模拟质量方差分布，以帮助对比学习策略在质量变化下获得稳健的特征表示。(3)设计了一个独立的质量网络(SQN)来学习与图像质量无关的区分特征。最后，通过大量实验验证了SuHiFiMASK数据集的质量和CQIL算法的优越性。



## **20. Efficient Robustness Assessment via Adversarial Spatial-Temporal Focus on Videos**

对抗性时空聚焦视频的高效稳健性评估 cs.CV

**SubmitDate**: 2023-01-03    [abs](http://arxiv.org/abs/2301.00896v1) [paper-pdf](http://arxiv.org/pdf/2301.00896v1)

**Authors**: Wei Xingxing, Wang Songping, Yan Huanqian

**Abstract**: Adversarial robustness assessment for video recognition models has raised concerns owing to their wide applications on safety-critical tasks. Compared with images, videos have much high dimension, which brings huge computational costs when generating adversarial videos. This is especially serious for the query-based black-box attacks where gradient estimation for the threat models is usually utilized, and high dimensions will lead to a large number of queries. To mitigate this issue, we propose to simultaneously eliminate the temporal and spatial redundancy within the video to achieve an effective and efficient gradient estimation on the reduced searching space, and thus query number could decrease. To implement this idea, we design the novel Adversarial spatial-temporal Focus (AstFocus) attack on videos, which performs attacks on the simultaneously focused key frames and key regions from the inter-frames and intra-frames in the video. AstFocus attack is based on the cooperative Multi-Agent Reinforcement Learning (MARL) framework. One agent is responsible for selecting key frames, and another agent is responsible for selecting key regions. These two agents are jointly trained by the common rewards received from the black-box threat models to perform a cooperative prediction. By continuously querying, the reduced searching space composed of key frames and key regions is becoming precise, and the whole query number becomes less than that on the original video. Extensive experiments on four mainstream video recognition models and three widely used action recognition datasets demonstrate that the proposed AstFocus attack outperforms the SOTA methods, which is prevenient in fooling rate, query number, time, and perturbation magnitude at the same.

摘要: 视频识别模型的对抗性健壮性评估由于其在安全关键任务中的广泛应用而引起了人们的关注。与图像相比，视频的维度要高得多，这在生成对抗性视频时带来了巨大的计算代价。这对于基于查询的黑盒攻击尤为严重，这种攻击通常使用威胁模型的梯度估计，高维将导致大量的查询。为了缓解这一问题，我们提出同时消除视频中的时间和空间冗余，在缩减的搜索空间上实现有效和高效的梯度估计，从而减少查询数量。为了实现这一思想，我们设计了一种新颖的对抗性时空聚焦(AstFocus)攻击，它从视频的帧间和帧内对同时聚焦的关键帧和关键区域进行攻击。AstFocus攻击基于协作多智能体强化学习(MAIL)框架。一个代理负责选择关键帧，另一个代理负责选择关键区域。这两个代理通过从黑盒威胁模型获得的共同奖励来联合训练，以执行合作预测。通过连续查询，缩小了由关键帧和关键区域组成的搜索空间，变得更加精确，整个查询次数比原始视频上的少。在四个主流视频识别模型和三个广泛使用的动作识别数据集上的大量实验表明，AstFocus攻击的性能优于SOTA方法，后者在愚弄率、查询次数、时间和扰动幅度方面都优于SOTA方法。



## **21. Adaptive Perturbation for Adversarial Attack**

对抗性攻击的自适应摄动 cs.CV

13 pages, 5 figures, 9 tables

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2111.13841v2) [paper-pdf](http://arxiv.org/pdf/2111.13841v2)

**Authors**: Zheng Yuan, Jie Zhang, Zhaoyan Jiang, Liangliang Li, Shiguang Shan

**Abstract**: In recent years, the security of deep learning models achieves more and more attentions with the rapid development of neural networks, which are vulnerable to adversarial examples. Almost all existing gradient-based attack methods use the sign function in the generation to meet the requirement of perturbation budget on $L_\infty$ norm. However, we find that the sign function may be improper for generating adversarial examples since it modifies the exact gradient direction. Instead of using the sign function, we propose to directly utilize the exact gradient direction with a scaling factor for generating adversarial perturbations, which improves the attack success rates of adversarial examples even with fewer perturbations. At the same time, we also theoretically prove that this method can achieve better black-box transferability. Moreover, considering that the best scaling factor varies across different images, we propose an adaptive scaling factor generator to seek an appropriate scaling factor for each image, which avoids the computational cost for manually searching the scaling factor. Our method can be integrated with almost all existing gradient-based attack methods to further improve their attack success rates. Extensive experiments on the CIFAR10 and ImageNet datasets show that our method exhibits higher transferability and outperforms the state-of-the-art methods.

摘要: 近年来，随着神经网络的快速发展，深度学习模型的安全性越来越受到人们的关注，因为神经网络容易受到敌意例子的攻击。几乎所有现有的基于梯度的攻击方法都在生成时使用符号函数，以满足$L_INFTY$范数上的扰动预算要求。然而，我们发现符号函数可能不适合于生成对抗性示例，因为它修改了精确的梯度方向。我们不使用符号函数，而是直接利用带有比例因子的精确梯度方向来产生对抗性扰动，从而在扰动较少的情况下提高了对抗性实例的攻击成功率。同时，我们还从理论上证明了该方法可以达到更好的黑盒可转移性。此外，考虑到不同图像的最佳比例因子不同，我们提出了一种自适应比例因子生成器来为每幅图像寻找合适的比例因子，从而避免了手动搜索比例因子的计算代价。我们的方法可以与几乎所有现有的基于梯度的攻击方法相集成，进一步提高它们的攻击成功率。在CIFAR10和ImageNet数据集上的大量实验表明，我们的方法表现出更高的可转移性，并且性能优于最先进的方法。



## **22. Differentiable Search of Accurate and Robust Architectures**

精确且健壮的体系结构的可微搜索 cs.LG

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2212.14049v2) [paper-pdf](http://arxiv.org/pdf/2212.14049v2)

**Authors**: Yuwei Ou, Xiangning Xie, Shangce Gao, Yanan Sun, Kay Chen Tan, Jiancheng Lv

**Abstract**: Deep neural networks (DNNs) are found to be vulnerable to adversarial attacks, and various methods have been proposed for the defense. Among these methods, adversarial training has been drawing increasing attention because of its simplicity and effectiveness. However, the performance of the adversarial training is greatly limited by the architectures of target DNNs, which often makes the resulting DNNs with poor accuracy and unsatisfactory robustness. To address this problem, we propose DSARA to automatically search for the neural architectures that are accurate and robust after adversarial training. In particular, we design a novel cell-based search space specially for adversarial training, which improves the accuracy and the robustness upper bound of the searched architectures by carefully designing the placement of the cells and the proportional relationship of the filter numbers. Then we propose a two-stage search strategy to search for both accurate and robust neural architectures. At the first stage, the architecture parameters are optimized to minimize the adversarial loss, which makes full use of the effectiveness of the adversarial training in enhancing the robustness. At the second stage, the architecture parameters are optimized to minimize both the natural loss and the adversarial loss utilizing the proposed multi-objective adversarial training method, so that the searched neural architectures are both accurate and robust. We evaluate the proposed algorithm under natural data and various adversarial attacks, which reveals the superiority of the proposed method in terms of both accurate and robust architectures. We also conclude that accurate and robust neural architectures tend to deploy very different structures near the input and the output, which has great practical significance on both hand-crafting and automatically designing of accurate and robust neural architectures.

摘要: 深度神经网络(DNN)被发现容易受到敌意攻击，并且已经提出了各种防御方法。在这些方法中，对抗性训练因其简单性和有效性而受到越来越多的关注。然而，对抗性训练的性能很大程度上受到目标DNN结构的限制，这往往使所得到的DNN具有较差的准确性和较差的稳健性。为了解决这一问题，我们提出了DSARA在对抗性训练后自动搜索准确和健壮的神经结构。特别是，我们设计了一种专门用于对抗性训练的基于单元的搜索空间，通过仔细设计单元的位置和过滤器数量的比例关系，提高了搜索结构的准确性和鲁棒性上界。然后，我们提出了一种两阶段搜索策略来搜索准确和健壮的神经结构。在第一阶段，对结构参数进行优化，使对抗性损失最小，充分利用对抗性训练在增强鲁棒性方面的有效性。在第二阶段，利用所提出的多目标对抗性训练方法对结构参数进行优化，使自然损失和对抗性损失最小化，从而使搜索到的神经结构既准确又健壮。我们在自然数据和各种敌意攻击下对该算法进行了测试，结果表明该算法在准确性和健壮性方面都具有一定的优势。我们还得出结论，精确和健壮的神经结构往往在输入和输出附近部署截然不同的结构，这对于手工制作和自动设计准确和健壮的神经结构都具有重要的现实意义。



## **23. Reversible Attack based on Local Visual Adversarial Perturbation**

基于局部视觉对抗扰动的可逆攻击 cs.CV

**SubmitDate**: 2023-01-02    [abs](http://arxiv.org/abs/2110.02700v3) [paper-pdf](http://arxiv.org/pdf/2110.02700v3)

**Authors**: Li Chen, Shaowei Zhu, Zhaoxia Yin

**Abstract**: Adding perturbations to images can mislead classification models to produce incorrect results. Recently, researchers exploited adversarial perturbations to protect image privacy from retrieval by intelligent models. However, adding adversarial perturbations to images destroys the original data, making images useless in digital forensics and other fields. To prevent illegal or unauthorized access to sensitive image data such as human faces without impeding legitimate users, the use of reversible adversarial attack techniques is increasing. The original image can be recovered from its reversible adversarial examples. However, existing reversible adversarial attack methods are designed for traditional imperceptible adversarial perturbations and ignore the local visible adversarial perturbation. In this paper, we propose a new method for generating reversible adversarial examples based on local visible adversarial perturbation. The information needed for image recovery is embedded into the area beyond the adversarial patch by the reversible data hiding technique. To reduce image distortion, lossless compression and the B-R-G (bluered-green) embedding principle are adopted. Experiments on CIFAR-10 and ImageNet datasets show that the proposed method can restore the original images error-free while ensuring good attack performance.

摘要: 向图像添加扰动可能会误导分类模型产生不正确的结果。最近，研究人员利用对抗性扰动来保护图像隐私，使其不受智能模型的检索。然而，向图像添加对抗性扰动会破坏原始数据，使图像在数字取证等领域毫无用处。为了防止在不妨碍合法用户的情况下非法或未经授权地访问人脸等敏感图像数据，可逆对抗性攻击技术的使用正在增加。原始图像可以从其可逆的对抗性例子中恢复。然而，现有的可逆对抗性攻击方法是针对传统的不可察觉的对抗性扰动而设计的，忽略了局部可见的对抗性扰动。本文提出了一种基于局部可视对抗性扰动的可逆对抗性实例生成方法。利用可逆数据隐藏技术将图像恢复所需的信息嵌入到敌方补丁之外的区域。为了减少图像失真，采用了无损压缩和B-R-G(蓝绿色)嵌入原理。在CIFAR-10和ImageNet数据集上的实验表明，该方法能够在保证良好攻击性能的前提下无差错地恢复原始图像。



## **24. Trojaning semi-supervised learning model via poisoning wild images on the web**

中毒网络野图的特洛伊木马半监督学习模型 cs.CY

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00435v1) [paper-pdf](http://arxiv.org/pdf/2301.00435v1)

**Authors**: Le Feng, Zhenxing Qian, Sheng Li, Xinpeng Zhang

**Abstract**: Wild images on the web are vulnerable to backdoor (also called trojan) poisoning, causing machine learning models learned on these images to be injected with backdoors. Most previous attacks assumed that the wild images are labeled. In reality, however, most images on the web are unlabeled. Specifically, we study the effects of unlabeled backdoor images under semi-supervised learning (SSL) on widely studied deep neural networks. To be realistic, we assume that the adversary is zero-knowledge and that the semi-supervised learning model is trained from scratch. Firstly, we find the fact that backdoor poisoning always fails when poisoned unlabeled images come from different classes, which is different from poisoning the labeled images. The reason is that the SSL algorithms always strive to correct them during training. Therefore, for unlabeled images, we implement backdoor poisoning on images from the target class. Then, we propose a gradient matching strategy to craft poisoned images such that their gradients match the gradients of target images on the SSL model, which can fit poisoned images to the target class and realize backdoor injection. To the best of our knowledge, this may be the first approach to backdoor poisoning on unlabeled images of trained-from-scratch SSL models. Experiments show that our poisoning achieves state-of-the-art attack success rates on most SSL algorithms while bypassing modern backdoor defenses.

摘要: 网络上的狂野图像很容易受到后门(也称为特洛伊木马)的毒害，导致从这些图像上学习的机器学习模型被注入后门。以前的大多数攻击都假设野生图像被标记了。然而，在现实中，网络上的大多数图片都是没有标签的。具体地说，我们研究了半监督学习(半监督学习)下的未标记后门图像对广泛研究的深度神经网络的影响。为了现实，我们假设对手是零知识，半监督学习模型是从头开始训练的。首先，我们发现，当中毒的未标记图像来自不同类别时，后门攻击总是失败的，这与对标记图像进行中毒是不同的。这是因为，在训练过程中，SSL算法总是努力纠正它们。因此，对于未标记的图像，我们对来自目标类的图像进行后门毒化。然后，我们提出了一种梯度匹配策略来构造中毒图像，使其梯度与目标图像在SSL模型上的梯度相匹配，从而将中毒图像匹配到目标类，实现后门注入。据我们所知，这可能是第一种对从头开始训练的SSL模型的未标记图像进行后门中毒的方法。实验表明，我们的毒剂在绕过现代后门防御的同时，在大多数SSL算法上实现了最先进的攻击成功率。



## **25. Differential Evolution based Dual Adversarial Camouflage: Fooling Human Eyes and Object Detectors**

基于差异进化的双重对抗性伪装：愚弄人眼和目标探测器 cs.CV

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2210.08870v3) [paper-pdf](http://arxiv.org/pdf/2210.08870v3)

**Authors**: Jialiang Sun, Tingsong Jiang, Wen Yao, Donghua Wang, Xiaoqian Chen

**Abstract**: Recent studies reveal that deep neural network (DNN) based object detectors are vulnerable to adversarial attacks in the form of adding the perturbation to the images, leading to the wrong output of object detectors. Most current existing works focus on generating perturbed images, also called adversarial examples, to fool object detectors. Though the generated adversarial examples themselves can remain a certain naturalness, most of them can still be easily observed by human eyes, which limits their further application in the real world. To alleviate this problem, we propose a differential evolution based dual adversarial camouflage (DE_DAC) method, composed of two stages to fool human eyes and object detectors simultaneously. Specifically, we try to obtain the camouflage texture, which can be rendered over the surface of the object. In the first stage, we optimize the global texture to minimize the discrepancy between the rendered object and the scene images, making human eyes difficult to distinguish. In the second stage, we design three loss functions to optimize the local texture, making object detectors ineffective. In addition, we introduce the differential evolution algorithm to search for the near-optimal areas of the object to attack, improving the adversarial performance under certain attack area limitations. Besides, we also study the performance of adaptive DE_DAC, which can be adapted to the environment. Experiments show that our proposed method could obtain a good trade-off between the fooling human eyes and object detectors under multiple specific scenes and objects.

摘要: 最近的研究表明，基于深度神经网络(DNN)的目标检测器容易受到敌意攻击，其形式是向图像添加扰动，导致目标检测器的输出错误。目前大多数现有的工作都集中在生成扰动图像，也称为对抗性示例，以愚弄对象检测器。尽管生成的对抗性例子本身可以保持一定的自然度，但其中大部分仍然很容易被人眼观察到，这限制了它们在现实世界中的进一步应用。为了缓解这一问题，我们提出了一种基于差异进化的双重对抗伪装(DE_DAC)方法，该方法由两个阶段组成，同时欺骗人眼和目标检测器。具体地说，我们试图获得伪装纹理，它可以在对象的表面上渲染。在第一阶段，我们对全局纹理进行优化，最小化绘制对象和场景图像之间的差异，使人眼难以辨别。在第二阶段，我们设计了三个损失函数来优化局部纹理，使得目标检测失效。此外，我们还引入了差分进化算法来搜索攻击对象的近最优区域，提高了在一定攻击区域限制下的对抗性能。此外，我们还研究了适应环境的自适应DE_DAC的性能。实验表明，在多个特定场景和目标的情况下，我们提出的方法可以在愚弄人眼和目标检测器之间取得良好的折衷。



## **26. Generalizable Black-Box Adversarial Attack with Meta Learning**

基于元学习的泛化黑箱对抗攻击 cs.LG

T-PAMI 2022. Project Page is at https://github.com/SCLBD/MCG-Blackbox

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.00364v1) [paper-pdf](http://arxiv.org/pdf/2301.00364v1)

**Authors**: Fei Yin, Yong Zhang, Baoyuan Wu, Yan Feng, Jingyi Zhang, Yanbo Fan, Yujiu Yang

**Abstract**: In the scenario of black-box adversarial attack, the target model's parameters are unknown, and the attacker aims to find a successful adversarial perturbation based on query feedback under a query budget. Due to the limited feedback information, existing query-based black-box attack methods often require many queries for attacking each benign example. To reduce query cost, we propose to utilize the feedback information across historical attacks, dubbed example-level adversarial transferability. Specifically, by treating the attack on each benign example as one task, we develop a meta-learning framework by training a meta-generator to produce perturbations conditioned on benign examples. When attacking a new benign example, the meta generator can be quickly fine-tuned based on the feedback information of the new task as well as a few historical attacks to produce effective perturbations. Moreover, since the meta-train procedure consumes many queries to learn a generalizable generator, we utilize model-level adversarial transferability to train the meta-generator on a white-box surrogate model, then transfer it to help the attack against the target model. The proposed framework with the two types of adversarial transferability can be naturally combined with any off-the-shelf query-based attack methods to boost their performance, which is verified by extensive experiments.

摘要: 在黑盒对抗性攻击场景中，目标模型的参数未知，攻击者的目标是在查询预算内根据查询反馈找到一个成功的对抗性扰动。由于反馈信息有限，现有的基于查询的黑盒攻击方法往往需要多次查询才能攻击每个良性实例。为了降低查询代价，我们提出了利用历史攻击中的反馈信息，称为实例级对抗性转移。具体地说，通过将对每个良性样本的攻击视为一个任务，我们通过训练元生成器来产生以良性样本为条件的扰动，从而开发了一个元学习框架。当攻击新的良性示例时，元生成器可以根据新任务的反馈信息以及一些历史攻击来快速微调，以产生有效的扰动。此外，由于元训练过程需要消耗大量的查询来学习可泛化的生成器，我们利用模型级的对抗性转移来训练白盒代理模型上的元生成器，然后将其转移以帮助对目标模型的攻击。该框架具有两种对抗性可转移性，可以自然地与任何现有的基于查询的攻击方法相结合，从而提高其性能，这一点得到了广泛的实验验证。



## **27. ExploreADV: Towards exploratory attack for Neural Networks**

ExplreADV：对神经网络的探索性攻击 cs.CR

**SubmitDate**: 2023-01-01    [abs](http://arxiv.org/abs/2301.01223v1) [paper-pdf](http://arxiv.org/pdf/2301.01223v1)

**Authors**: Tianzuo Luo, Yuyi Zhong, Siaucheng Khoo

**Abstract**: Although deep learning has made remarkable progress in processing various types of data such as images, text and speech, they are known to be susceptible to adversarial perturbations: perturbations specifically designed and added to the input to make the target model produce erroneous output. Most of the existing studies on generating adversarial perturbations attempt to perturb the entire input indiscriminately. In this paper, we propose ExploreADV, a general and flexible adversarial attack system that is capable of modeling regional and imperceptible attacks, allowing users to explore various kinds of adversarial examples as needed. We adapt and combine two existing boundary attack methods, DeepFool and Brendel\&Bethge Attack, and propose a mask-constrained adversarial attack system, which generates minimal adversarial perturbations under the pixel-level constraints, namely ``mask-constraints''. We study different ways of generating such mask-constraints considering the variance and importance of the input features, and show that our adversarial attack system offers users good flexibility to focus on sub-regions of inputs, explore imperceptible perturbations and understand the vulnerability of pixels/regions to adversarial attacks. We demonstrate our system to be effective based on extensive experiments and user study.

摘要: 虽然深度学习在处理图像、文本和语音等各种类型的数据方面取得了显著的进展，但众所周知，它们容易受到对抗性扰动的影响：这些扰动是专门设计并添加到输入中的，以使目标模型产生错误的输出。现有的关于产生对抗性扰动的研究大多试图不加区别地扰乱整个输入。本文提出了一种通用的、灵活的对抗性攻击系统DevelopreADV，它能够对局部的、不可察觉的攻击进行建模，允许用户根据需要探索各种对抗性的例子。本文对现有的两种边界攻击方法DeepFool和Brendel-Bethge进行了改进和结合，提出了一种基于掩码约束的对抗性攻击系统，该系统在像素级约束下产生最小的对抗性扰动，即“掩码约束”。考虑到输入特征的方差和重要性，我们研究了不同的生成掩码约束的方法，并表明我们的对抗性攻击系统为用户提供了良好的灵活性，使用户能够专注于输入子区域，探索不可察觉的扰动，并了解像素/区域对对抗性攻击的脆弱性。通过大量的实验和用户研究，我们证明了我们的系统是有效的。



## **28. WiFi Physical Layer Stays Awake and Responds When it Should Not**

WiFi物理层保持唤醒，并在不应唤醒时进行响应 cs.NI

12 pages

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.00269v1) [paper-pdf](http://arxiv.org/pdf/2301.00269v1)

**Authors**: Ali Abedi, Haofan Lu, Alex Chen, Charlie Liu, Omid Abari

**Abstract**: WiFi communication should be possible only between devices inside the same network. However, we find that all existing WiFi devices send back acknowledgments (ACK) to even fake packets received from unauthorized WiFi devices outside of their network. Moreover, we find that an unauthorized device can manipulate the power-saving mechanism of WiFi radios and keep them continuously awake by sending specific fake beacon frames to them. Our evaluation of over 5,000 devices from 186 vendors confirms that these are widespread issues. We believe these loopholes cannot be prevented, and hence they create privacy and security concerns. Finally, to show the importance of these issues and their consequences, we implement and demonstrate two attacks where an adversary performs battery drain and WiFi sensing attacks just using a tiny WiFi module which costs less than ten dollars.

摘要: WiFi通信应该只能在同一网络内的设备之间进行。然而，我们发现，所有现有的WiFi设备都会向从其网络外部的未经授权的WiFi设备接收的虚假数据包发送回确认(ACK)。此外，我们发现未经授权的设备可以操纵WiFi无线电的节电机制，并通过向其发送特定的虚假信标帧来保持其持续唤醒。我们对186家供应商的5,000多台设备进行的评估证实，这些问题普遍存在。我们认为这些漏洞是无法阻止的，因此它们会造成隐私和安全方面的问题。最后，为了说明这些问题的重要性及其后果，我们实现并演示了两个攻击，其中对手仅使用一个成本不到10美元的微小WiFi模块就可以执行电池耗尽攻击和WiFi传感攻击。



## **29. A Comparative Study of Image Disguising Methods for Confidential Outsourced Learning**

用于保密外包学习的图像伪装方法比较研究 cs.CR

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.00252v1) [paper-pdf](http://arxiv.org/pdf/2301.00252v1)

**Authors**: Sagar Sharma, Yuechun Gu, Keke Chen

**Abstract**: Large training data and expensive model tweaking are standard features of deep learning for images. As a result, data owners often utilize cloud resources to develop large-scale complex models, which raises privacy concerns. Existing solutions are either too expensive to be practical or do not sufficiently protect the confidentiality of data and models. In this paper, we study and compare novel \emph{image disguising} mechanisms, DisguisedNets and InstaHide, aiming to achieve a better trade-off among the level of protection for outsourced DNN model training, the expenses, and the utility of data. DisguisedNets are novel combinations of image blocktization, block-level random permutation, and two block-level secure transformations: random multidimensional projection (RMT) and AES pixel-level encryption (AES). InstaHide is an image mixup and random pixel flipping technique \cite{huang20}. We have analyzed and evaluated them under a multi-level threat model. RMT provides a better security guarantee than InstaHide, under the Level-1 adversarial knowledge with well-preserved model quality. In contrast, AES provides a security guarantee under the Level-2 adversarial knowledge, but it may affect model quality more. The unique features of image disguising also help us to protect models from model-targeted attacks. We have done an extensive experimental evaluation to understand how these methods work in different settings for different datasets.

摘要: 大量的训练数据和昂贵的模型调整是图像深度学习的标准特征。因此，数据所有者经常利用云资源开发大规模的复杂模型，这引发了隐私问题。现有的解决方案要么过于昂贵，不切实际，要么不能充分保护数据和模型的机密性。本文研究和比较了伪装网和InstaHide这两种新的图像伪装机制，旨在实现对外包DNN模型训练的保护水平、费用和数据利用率之间的更好的权衡。伪装网是图像分块、块级随机置换和两种块级安全变换的新组合：随机多维投影(RMT)和AES像素级加密(AES)。InstaHide是一种图像混合和随机像素翻转技术。我们已经在多级别威胁模型下对它们进行了分析和评估。RMT提供了比InstaHide更好的安全保障，在模型质量保持良好的情况下，提供了级别1的对抗知识。与之相比，高级加密标准提供了2级对抗知识下的安全保证，但对模型质量的影响可能更大。图像伪装的独特功能也有助于保护模型免受针对模型的攻击。我们已经做了广泛的实验评估，以了解这些方法如何在不同的环境下针对不同的数据集工作。



## **30. FLAME: Taming Backdoors in Federated Learning**

火焰：联合学习中的后门驯服 cs.CR

To appear in the 31st USENIX Security Symposium, August 2022, Boston,  MA, USA

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2101.02281v4) [paper-pdf](http://arxiv.org/pdf/2101.02281v4)

**Authors**: Thien Duc Nguyen, Phillip Rieger, Huili Chen, Hossein Yalame, Helen Möllering, Hossein Fereidooni, Samuel Marchal, Markus Miettinen, Azalia Mirhoseini, Shaza Zeitouni, Farinaz Koushanfar, Ahmad-Reza Sadeghi, Thomas Schneider

**Abstract**: Federated Learning (FL) is a collaborative machine learning approach allowing participants to jointly train a model without having to share their private, potentially sensitive local datasets with others. Despite its benefits, FL is vulnerable to so-called backdoor attacks, in which an adversary injects manipulated model updates into the federated model aggregation process so that the resulting model will provide targeted false predictions for specific adversary-chosen inputs. Proposed defenses against backdoor attacks based on detecting and filtering out malicious model updates consider only very specific and limited attacker models, whereas defenses based on differential privacy-inspired noise injection significantly deteriorate the benign performance of the aggregated model. To address these deficiencies, we introduce FLAME, a defense framework that estimates the sufficient amount of noise to be injected to ensure the elimination of backdoors. To minimize the required amount of noise, FLAME uses a model clustering and weight clipping approach. This ensures that FLAME can maintain the benign performance of the aggregated model while effectively eliminating adversarial backdoors. Our evaluation of FLAME on several datasets stemming from application areas including image classification, word prediction, and IoT intrusion detection demonstrates that FLAME removes backdoors effectively with a negligible impact on the benign performance of the models.

摘要: 联合学习(FL)是一种协作式机器学习方法，允许参与者联合训练模型，而不必与其他人共享他们私有的、潜在敏感的本地数据集。尽管FL有好处，但它很容易受到所谓的后门攻击，即对手将操纵的模型更新注入联邦模型聚合过程，以便结果模型将为对手选择的特定输入提供有针对性的错误预测。提出的基于检测和过滤恶意模型更新的后门攻击防御方案只考虑非常具体和有限的攻击者模型，而基于差异隐私激发噪声注入的防御方案会显著降低聚合模型的良性性能。为了解决这些不足，我们引入了FLAME，这是一个防御框架，它可以估计要注入的足够数量的噪音，以确保消除后门。为了最大限度地减少所需的噪声量，FLAME使用了模型聚类和权重裁剪方法。这确保了火焰可以在有效消除对手后门的同时，保持聚合模型的良性性能。我们在几个来自图像分类、词语预测和物联网入侵检测等应用领域的数据集上的评估表明，FLAME有效地移除了后门，而对模型的良性性能的影响可以忽略不计。



## **31. Tracing the Origin of Adversarial Attack for Forensic Investigation and Deterrence**

追踪对抗性攻击的来源以进行法医调查和威慑 cs.CR

**SubmitDate**: 2022-12-31    [abs](http://arxiv.org/abs/2301.01218v1) [paper-pdf](http://arxiv.org/pdf/2301.01218v1)

**Authors**: Han Fang, Jiyi Zhang, Yupeng Qiu, Ke Xu, Chengfang Fang, Ee-Chien Chang

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. In this paper, we take the role of investigators who want to trace the attack and identify the source, that is, the particular model which the adversarial examples are generated from. Techniques derived would aid forensic investigation of attack incidents and serve as deterrence to potential attacks. We consider the buyers-seller setting where a machine learning model is to be distributed to various buyers and each buyer receives a slightly different copy with same functionality. A malicious buyer generates adversarial examples from a particular copy $\mathcal{M}_i$ and uses them to attack other copies. From these adversarial examples, the investigator wants to identify the source $\mathcal{M}_i$. To address this problem, we propose a two-stage separate-and-trace framework. The model separation stage generates multiple copies of a model for a same classification task. This process injects unique characteristics into each copy so that adversarial examples generated have distinct and traceable features. We give a parallel structure which embeds a ``tracer'' in each copy, and a noise-sensitive training loss to achieve this goal. The tracing stage takes in adversarial examples and a few candidate models, and identifies the likely source. Based on the unique features induced by the noise-sensitive loss function, we could effectively trace the potential adversarial copy by considering the output logits from each tracer. Empirical results show that it is possible to trace the origin of the adversarial example and the mechanism can be applied to a wide range of architectures and datasets.

摘要: 深度神经网络很容易受到敌意攻击。在本文中，我们扮演的是调查者的角色，他们想要追踪攻击并识别来源，即生成对抗性例子的特定模型。得出的技术将有助于对袭击事件进行法医调查，并对潜在的袭击起到威慑作用。我们考虑买家-卖家的设置，其中机器学习模型将分发给不同的买家，每个买家收到的副本略有不同，具有相同的功能。恶意买家从特定副本$\Mathcal{M}_I$生成敌意示例，并使用它们攻击其他副本。从这些敌对的例子中，调查者想要确定来源$\mathcal{M}_I$。为了解决这个问题，我们提出了一个两阶段分离和跟踪的框架。模型分离阶段为同一分类任务生成模型的多个副本。这一过程为每个副本注入了独特的特征，从而使生成的对抗性例子具有明显和可追踪的特征。为了实现这一目标，我们给出了一种并行结构，即在每个副本中嵌入一个“跟踪器”，以及一个对噪声敏感的训练损失。跟踪阶段采用对抗性例子和一些候选模型，并确定可能的来源。基于噪声敏感损失函数的独特特征，通过考虑每个跟踪器的输出日志，可以有效地跟踪潜在的敌意副本。实验结果表明，该机制可以追溯对抗性例子的来源，并且可以应用于广泛的体系结构和数据集。



## **32. Guidance Through Surrogate: Towards a Generic Diagnostic Attack**

通过代理提供指导：通向通用诊断攻击 cs.LG

IEEE Transactions on Neural Networks and Learning Systems (TNNLS)

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14875v1) [paper-pdf](http://arxiv.org/pdf/2212.14875v1)

**Authors**: Muzammal Naseer, Salman Khan, Fatih Porikli, Fahad Shahbaz Khan

**Abstract**: Adversarial training is an effective approach to make deep neural networks robust against adversarial attacks. Recently, different adversarial training defenses are proposed that not only maintain a high clean accuracy but also show significant robustness against popular and well studied adversarial attacks such as PGD. High adversarial robustness can also arise if an attack fails to find adversarial gradient directions, a phenomenon known as `gradient masking'. In this work, we analyse the effect of label smoothing on adversarial training as one of the potential causes of gradient masking. We then develop a guided mechanism to avoid local minima during attack optimization, leading to a novel attack dubbed Guided Projected Gradient Attack (G-PGA). Our attack approach is based on a `match and deceive' loss that finds optimal adversarial directions through guidance from a surrogate model. Our modified attack does not require random restarts, large number of attack iterations or search for an optimal step-size. Furthermore, our proposed G-PGA is generic, thus it can be combined with an ensemble attack strategy as we demonstrate for the case of Auto-Attack, leading to efficiency and convergence speed improvements. More than an effective attack, G-PGA can be used as a diagnostic tool to reveal elusive robustness due to gradient masking in adversarial defenses.

摘要: 对抗性训练是使深度神经网络对对抗性攻击具有较强鲁棒性的一种有效方法。最近，不同的对抗性训练防御方法被提出，它们不仅保持了很高的清洁准确率，而且对流行的和研究得很好的对抗性攻击，如PGD，表现出了显著的鲁棒性。如果攻击未能找到对抗性梯度方向，也可能出现较高的对抗性稳健性，这一现象称为“梯度掩蔽”。在这项工作中，我们分析了标签平滑对对抗性训练的影响，作为梯度掩蔽的潜在原因之一。在此基础上，提出了一种在攻击优化过程中避免陷入局部极小的引导机制，提出了一种称为引导投影梯度攻击(G-PGA)的攻击方法。我们的攻击方法是基于“匹配和欺骗”的损失，通过代理模型的指导找到最优的对手方向。改进后的攻击不需要随机重新启动，不需要大量的攻击迭代，也不需要搜索最优步长。此外，我们提出的G-PGA是通用的，因此它可以与集成攻击策略相结合，就像我们在自动攻击的情况下所演示的那样，从而提高了效率和收敛速度。除了有效的攻击，G-PGA还可以作为一种诊断工具来揭示由于梯度掩蔽而在对抗性防御中难以捉摸的稳健性。



## **33. Secure Fusion Estimation Against FDI Sensor Attacks in Cyber-Physical Systems**

网络物理系统中抗FDI传感器攻击的安全融合估计 eess.SY

10 pages, 5 figures; the first version of this manuscript was  completed on 2020

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14755v1) [paper-pdf](http://arxiv.org/pdf/2212.14755v1)

**Authors**: Bo Chen, Pindi Weng, Daniel W. C. Ho, Li Yu

**Abstract**: This paper is concerned with the problem of secure multi-sensors fusion estimation for cyber-physical systems, where sensor measurements may be tampered with by false data injection (FDI) attacks. In this work, it is considered that the adversary may not be able to attack all sensors. That is, several sensors remain not being attacked. In this case, new local reorganized subsystems including the FDI attack signals and un-attacked sensor measurements are constructed by the augmentation method. Then, a joint Kalman fusion estimator is designed under linear minimum variance sense to estimate the system state and FDI attack signals simultaneously. Finally, illustrative examples are employed to show the effectiveness and advantages of the proposed methods.

摘要: 研究了网络物理系统中传感器测量可能被虚假数据注入(FDI)攻击篡改的安全多传感器融合估计问题。在这项工作中，考虑到对手可能不能攻击所有的传感器。也就是说，几个传感器仍未受到攻击。在这种情况下，利用增强方法构造新的局部重组子系统，包括FDI攻击信号和未受攻击的传感器测量。然后，在线性最小方差意义下设计了联合卡尔曼融合估值器来同时估计系统状态和FDI攻击信号。最后，通过算例说明了该方法的有效性和优越性。



## **34. Adversarial attacks and defenses on ML- and hardware-based IoT device fingerprinting and identification**

基于ML和硬件的物联网设备指纹识别和识别的对抗性攻击和防御 cs.CR

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14677v1) [paper-pdf](http://arxiv.org/pdf/2212.14677v1)

**Authors**: Pedro Miguel Sánchez Sánchez, Alberto Huertas Celdrán, Gérôme Bovet, Gregorio Martínez Pérez

**Abstract**: In the last years, the number of IoT devices deployed has suffered an undoubted explosion, reaching the scale of billions. However, some new cybersecurity issues have appeared together with this development. Some of these issues are the deployment of unauthorized devices, malicious code modification, malware deployment, or vulnerability exploitation. This fact has motivated the requirement for new device identification mechanisms based on behavior monitoring. Besides, these solutions have recently leveraged Machine and Deep Learning techniques due to the advances in this field and the increase in processing capabilities. In contrast, attackers do not stay stalled and have developed adversarial attacks focused on context modification and ML/DL evaluation evasion applied to IoT device identification solutions. This work explores the performance of hardware behavior-based individual device identification, how it is affected by possible context- and ML/DL-focused attacks, and how its resilience can be improved using defense techniques. In this sense, it proposes an LSTM-CNN architecture based on hardware performance behavior for individual device identification. Then, previous techniques have been compared with the proposed architecture using a hardware performance dataset collected from 45 Raspberry Pi devices running identical software. The LSTM-CNN improves previous solutions achieving a +0.96 average F1-Score and 0.8 minimum TPR for all devices. Afterward, context- and ML/DL-focused adversarial attacks were applied against the previous model to test its robustness. A temperature-based context attack was not able to disrupt the identification. However, some ML/DL state-of-the-art evasion attacks were successful. Finally, adversarial training and model distillation defense techniques are selected to improve the model resilience to evasion attacks, without degrading its performance.

摘要: 在过去的几年里，部署的物联网设备数量无疑经历了爆炸性的增长，达到了数十亿的规模。然而，伴随着这一发展，一些新的网络安全问题也随之出现。其中一些问题是部署未经授权的设备、恶意代码修改、恶意软件部署或漏洞利用。这一事实促使了对基于行为监控的新设备识别机制的需求。此外，由于这一领域的进步和处理能力的增加，这些解决方案最近利用了机器和深度学习技术。相比之下，攻击者不会停滞不前，并已开发出专注于上下文修改和应用于物联网设备识别解决方案的ML/DL评估规避的对抗性攻击。这项工作探讨了基于硬件行为的单个设备识别的性能，它如何受到可能的上下文和ML/DL重点攻击的影响，以及如何使用防御技术提高其弹性。在此意义上，提出了一种基于硬件性能行为的LSTM-CNN设备识别体系结构。然后，使用从45个运行相同软件的Raspberry PI设备收集的硬件性能数据集，将以前的技术与所提出的体系结构进行了比较。LSTM-CNN改进了以前的解决方案，实现了所有设备的+0.96平均F1得分和0.8最低TPR。然后，对先前的模型进行了上下文和ML/DL重点对抗性攻击，以测试其稳健性。基于温度的上下文攻击不能中断识别。然而，一些最先进的ML/DL逃避攻击是成功的。最后，采用对抗性训练和模型升华防御技术，在不降低模型性能的前提下，提高了模型对逃避攻击的恢复能力。



## **35. Defense Against Adversarial Attacks on Audio DeepFake Detection**

音频DeepFake检测中的敌意攻击防御 cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2212.14597v1) [paper-pdf](http://arxiv.org/pdf/2212.14597v1)

**Authors**: Piotr Kawa, Marcin Plata, Piotr Syga

**Abstract**: Audio DeepFakes are artificially generated utterances created using deep learning methods with the main aim to fool the listeners, most of such audio is highly convincing. Their quality is sufficient to pose a serious threat in terms of security and privacy, such as the reliability of news or defamation. To prevent the threats, multiple neural networks-based methods to detect generated speech have been proposed. In this work, we cover the topic of adversarial attacks, which decrease the performance of detectors by adding superficial (difficult to spot by a human) changes to input data. Our contribution contains evaluating the robustness of 3 detection architectures against adversarial attacks in two scenarios (white-box and using transferability mechanism) and enhancing it later by the use of adversarial training performed by our novel adaptive training method.

摘要: Audio DeepFake是人工生成的话语，使用深度学习方法创建，主要目的是愚弄听众，此类音频大多具有很强的说服力。它们的质量足以在安全和隐私方面构成严重威胁，例如新闻的可靠性或诽谤。为了防止这种威胁，已经提出了多种基于神经网络的语音生成检测方法。在这项工作中，我们讨论了对抗性攻击的主题，这种攻击通过向输入数据添加表面(难以被人发现)的更改来降低检测器的性能。我们的贡献包括评估三种检测体系结构在两种场景(白盒和使用可转移机制)下对敌意攻击的健壮性，并在以后通过使用新的自适应训练方法进行对抗性训练来增强检测体系结构的健壮性。



## **36. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

注意：通过变分推理进行推理的可证明稳健学习 cs.LG

**SubmitDate**: 2022-12-30    [abs](http://arxiv.org/abs/2209.05055v3) [paper-pdf](http://arxiv.org/pdf/2209.05055v3)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstract**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.

摘要: 尽管深度神经网络(DNN)最近取得了很大的进展，但它们往往容易受到对手的攻击。人们已经进行了大量的研究来提高DNN的稳健性，然而，大多数经验防御都可以再次自适应攻击，理论上证明的健壮性是有限的，特别是在大规模数据集上。DNN这种漏洞的一个潜在根本原因是，尽管它们表现出强大的表现力，但它们缺乏做出稳健和可靠预测的推理能力。在本文中，我们的目标是将领域知识集成到推理范式中，以实现稳健的学习。特别地，我们提出了一种带推理的可证明稳健学习流水线(CARE)，该流水线由学习组件和推理组件组成。具体地说，我们使用一组标准的DNN作为学习组件进行语义预测，并利用马尔可夫逻辑网络(MLN)等概率图形模型作为推理组件来实现知识/逻辑推理。然而，众所周知，MLN(推理)的精确推理是#P-完全的，这限制了流水线的可扩展性。为此，我们提出了基于一种有效的期望最大化算法的变分推理来逼近最大似然推理。特别是，我们利用图卷积网络(GCNS)对变分推理过程中的后验分布进行编码，并迭代地更新GCNS的参数(E步)和MLN中知识规则的权值(M步)。我们在不同的数据集上进行了广泛的实验，并表明与最先进的基线相比，CARE实现了显著更高的认证稳健性。此外，我们还进行了不同的消融研究，以证明CARE的经验稳健性和不同知识整合的有效性。



## **37. "Real Attackers Don't Compute Gradients": Bridging the Gap Between Adversarial ML Research and Practice**

“真正的攻击者不计算梯度”：弥合对抗性ML研究和实践之间的差距 cs.CR

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2212.14315v1) [paper-pdf](http://arxiv.org/pdf/2212.14315v1)

**Authors**: Giovanni Apruzzese, Hyrum S. Anderson, Savino Dambra, David Freeman, Fabio Pierazzi, Kevin A. Roundy

**Abstract**: Recent years have seen a proliferation of research on adversarial machine learning. Numerous papers demonstrate powerful algorithmic attacks against a wide variety of machine learning (ML) models, and numerous other papers propose defenses that can withstand most attacks. However, abundant real-world evidence suggests that actual attackers use simple tactics to subvert ML-driven systems, and as a result security practitioners have not prioritized adversarial ML defenses.   Motivated by the apparent gap between researchers and practitioners, this position paper aims to bridge the two domains. We first present three real-world case studies from which we can glean practical insights unknown or neglected in research. Next we analyze all adversarial ML papers recently published in top security conferences, highlighting positive trends and blind spots. Finally, we state positions on precise and cost-driven threat modeling, collaboration between industry and academia, and reproducible research. We believe that our positions, if adopted, will increase the real-world impact of future endeavours in adversarial ML, bringing both researchers and practitioners closer to their shared goal of improving the security of ML systems.

摘要: 近年来，对抗性机器学习的研究激增。许多论文展示了针对各种机器学习(ML)模型的强大算法攻击，以及许多其他论文提出了可以抵御大多数攻击的防御措施。然而，大量的现实世界证据表明，实际攻击者使用简单的策略来颠覆ML驱动的系统，因此安全从业者并没有优先考虑对抗性的ML防御。出于研究人员和实践者之间的明显差距，本立场文件旨在弥合这两个领域。我们首先介绍了三个真实世界的案例研究，从中我们可以收集研究中未知或被忽视的实用见解。接下来，我们分析最近在顶级安全会议上发表的所有对抗性ML论文，强调积极的趋势和盲区。最后，我们阐述了对精确和成本驱动的威胁建模、产业界和学术界之间的合作以及可重复研究的立场。我们相信，如果我们的立场被采纳，将增加对抗性ML未来努力的现实世界影响，使研究人员和实践者更接近他们提高ML系统安全性的共同目标。



## **38. Towards Comprehensively Understanding the Run-time Security of Programmable Logic Controllers: A 3-year Empirical Study**

全面理解可编程逻辑控制器的运行时安全性：为期3年的经验研究 cs.CR

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2212.14296v1) [paper-pdf](http://arxiv.org/pdf/2212.14296v1)

**Authors**: Rongkuan Ma, Qiang Wei, Jingyi Wang, Shunkai Zhu, Shouling Ji, Peng Cheng, Yan Jia, Qingxian Wang

**Abstract**: Programmable Logic Controllers (PLCs) are the core control devices in Industrial Control Systems (ICSs), which control and monitor the underlying physical plants such as power grids. PLCs were initially designed to work in a trusted industrial network, which however can be brittle once deployed in an Internet-facing (or penetrated) network. Yet, there is a lack of systematic empirical analysis of the run-time security of modern real-world PLCs. To close this gap, we present the first large-scale measurement on 23 off-the-shelf PLCs across 13 leading vendors. We find many common security issues and unexplored implications that should be more carefully addressed in the design and implementation. To sum up, the unsupervised logic applications can cause system resource/privilege abuse, which gives adversaries new means to hijack the control flow of a runtime system remotely (without exploiting memory vulnerabilities); 2) the improper access control mechanisms bring many unauthorized access implications; 3) the proprietary or semi-proprietary protocols are fragile regarding confidentiality and integrity protection of run-time data. We empirically evaluated the corresponding attack vectors on multiple PLCs, which demonstrates that the security implications are severe and broad. Our findings were reported to the related parties responsibly, and 20 bugs have been confirmed with 7 assigned CVEs.

摘要: 可编程逻辑控制器(PLC)是工业控制系统(ICSS)中的核心控制设备，对电网等底层物理对象进行控制和监控。PLC最初设计为在受信任的工业网络中工作，但一旦部署在面向互联网(或渗透)的网络中，该网络可能会变得脆弱。然而，缺乏对现代现实世界PLC运行时安全性的系统实证分析。为了缩小这一差距，我们首次对13家领先供应商的23个现成PLC进行了大规模测量。我们发现许多共同的安全问题和未探索的影响，应该在设计和实施时更仔细地加以处理。综上所述，非监督逻辑应用程序可能导致系统资源/权限滥用，这为攻击者远程劫持运行时系统的控制流提供了新的手段(而不利用内存漏洞)；2)不正确的访问控制机制带来了许多未经授权的访问影响；3)专有或半专有协议在运行时数据的机密性和完整性保护方面是脆弱的。我们在多个PLC上对相应的攻击向量进行了经验性评估，表明其安全影响是严重和广泛的。我们的发现被负责任地报告给了相关方，并已确认了20个错误，并指定了7个CVE。



## **39. Attention, Please! Adversarial Defense via Activation Rectification and Preservation**

请注意！激活、整改、保存的对抗性防御 cs.CV

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/1811.09831v5) [paper-pdf](http://arxiv.org/pdf/1811.09831v5)

**Authors**: Shangxi Wu, Jitao Sang, Kaiyuan Xu, Jiaming Zhang, Jian Yu

**Abstract**: This study provides a new understanding of the adversarial attack problem by examining the correlation between adversarial attack and visual attention change. In particular, we observed that: (1) images with incomplete attention regions are more vulnerable to adversarial attacks; and (2) successful adversarial attacks lead to deviated and scattered attention map. Accordingly, an attention-based adversarial defense framework is designed to simultaneously rectify the attention map for prediction and preserve the attention area between adversarial and original images. The problem of adding iteratively attacked samples is also discussed in the context of visual attention change. We hope the attention-related data analysis and defense solution in this study will shed some light on the mechanism behind the adversarial attack and also facilitate future adversarial defense/attack model design.

摘要: 本研究通过考察对抗性攻击与视觉注意变化之间的关系，对对抗性攻击问题有了新的认识。特别是，我们观察到：(1)注意区域不完整的图像更容易受到对抗性攻击；(2)对抗性攻击成功会导致注意图偏离和分散。在此基础上，设计了一种基于注意力的对抗性防御框架，该框架能够同时校正用于预测的注意图，并保留对抗性图像和原始图像之间的注意区域。在视觉注意变化的背景下，还讨论了添加迭代攻击样本的问题。我们希望本研究中与注意力相关的数据分析和防御解决方案将有助于揭示对抗性攻击背后的机制，并为未来对抗性防御/攻击模型的设计提供便利。



## **40. Reducing Certified Regression to Certified Classification for General Poisoning Attacks**

将一般中毒发作的认证回归归结为认证分类 cs.LG

Accepted at the 1st IEEE conference on Secure and Trustworthy Machine  Learning (SaTML'23)

**SubmitDate**: 2022-12-29    [abs](http://arxiv.org/abs/2208.13904v2) [paper-pdf](http://arxiv.org/pdf/2208.13904v2)

**Authors**: Zayd Hammoudeh, Daniel Lowd

**Abstract**: Adversarial training instances can severely distort a model's behavior. This work investigates certified regression defenses, which provide guaranteed limits on how much a regressor's prediction may change under a poisoning attack. Our key insight is that certified regression reduces to voting-based certified classification when using median as a model's primary decision function. Coupling our reduction with existing certified classifiers, we propose six new regressors provably-robust to poisoning attacks. To the extent of our knowledge, this is the first work that certifies the robustness of individual regression predictions without any assumptions about the data distribution and model architecture. We also show that the assumptions made by existing state-of-the-art certified classifiers are often overly pessimistic. We introduce a tighter analysis of model robustness, which in many cases results in significantly improved certified guarantees. Lastly, we empirically demonstrate our approaches' effectiveness on both regression and classification data, where the accuracy of up to 50% of test predictions can be guaranteed under 1% training set corruption and up to 30% of predictions under 4% corruption. Our source code is available at https://github.com/ZaydH/certified-regression.

摘要: 对抗性训练实例会严重扭曲模型的行为。这项工作调查了经过认证的回归防御，它为回归者的预测在中毒攻击下可能改变多少提供了保证限制。我们的主要见解是，当使用中值作为模型的主要决策函数时，认证回归简化为基于投票的认证分类。结合我们的约简和现有的认证分类器，我们提出了六个新的回归函数-对中毒攻击具有健壮性。就我们所知，这是第一个在没有任何关于数据分布和模型体系结构的假设的情况下证明个体回归预测的稳健性的工作。我们还表明，现有最先进的认证分类器所做的假设往往过于悲观。我们引入了对模型稳健性的更严格的分析，这在许多情况下导致了显著改进的认证保证。最后，我们在回归和分类数据上验证了我们的方法的有效性，其中在1%的训练集损坏情况下可以保证高达50%的测试预测的准确性，在4%的损坏情况下可以保证高达30%的预测准确率。我们的源代码可以在https://github.com/ZaydH/certified-regression.上找到



## **41. Certifying Safety in Reinforcement Learning under Adversarial Perturbation Attacks**

对抗性扰动攻击下强化学习的安全性证明 cs.LG

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.14115v1) [paper-pdf](http://arxiv.org/pdf/2212.14115v1)

**Authors**: Junlin Wu, Hussein Sibai, Yevgeniy Vorobeychik

**Abstract**: Function approximation has enabled remarkable advances in applying reinforcement learning (RL) techniques in environments with high-dimensional inputs, such as images, in an end-to-end fashion, mapping such inputs directly to low-level control. Nevertheless, these have proved vulnerable to small adversarial input perturbations. A number of approaches for improving or certifying robustness of end-to-end RL to adversarial perturbations have emerged as a result, focusing on cumulative reward. However, what is often at stake in adversarial scenarios is the violation of fundamental properties, such as safety, rather than the overall reward that combines safety with efficiency. Moreover, properties such as safety can only be defined with respect to true state, rather than the high-dimensional raw inputs to end-to-end policies. To disentangle nominal efficiency and adversarial safety, we situate RL in deterministic partially-observable Markov decision processes (POMDPs) with the goal of maximizing cumulative reward subject to safety constraints. We then propose a partially-supervised reinforcement learning (PSRL) framework that takes advantage of an additional assumption that the true state of the POMDP is known at training time. We present the first approach for certifying safety of PSRL policies under adversarial input perturbations, and two adversarial training approaches that make direct use of PSRL. Our experiments demonstrate both the efficacy of the proposed approach for certifying safety in adversarial environments, and the value of the PSRL framework coupled with adversarial training in improving certified safety while preserving high nominal reward and high-quality predictions of true state.

摘要: 函数逼近使强化学习(RL)技术能够以端到端的方式在具有高维输入(如图像)的环境中应用，将此类输入直接映射到低级控制。然而，事实证明，它们容易受到小的对抗性输入扰动的影响。因此，出现了一些改进或验证端到端RL对对抗性扰动的稳健性的方法，重点关注累积奖励。然而，在对抗性情景中，往往事关重大的是对安全等基本属性的违反，而不是将安全与效率结合在一起的整体回报。此外，诸如安全性之类的属性只能根据真实状态来定义，而不是端到端策略的高维原始输入。为了分离名义效率和对抗安全性，我们将RL置于确定性部分可观测马尔可夫决策过程(POMDP)中，目标是在安全约束下最大化累积报酬。然后，我们提出了一个部分监督强化学习(PSRL)框架，它利用了一个额外的假设，即POMDP的真实状态在训练时是已知的。我们提出了第一种证明PSRL策略在对抗性输入扰动下的安全性的方法，以及两种直接使用PSRL的对抗性训练方法。我们的实验证明了所提出的方法在对抗性环境中认证安全性的有效性，以及PSRL框架与对抗性训练相结合在提高认证安全性的同时保持高名义回报和对真实状态的高质量预测的价值。



## **42. Robust Ranking Explanations**

稳健的排名解释 cs.LG

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.14106v1) [paper-pdf](http://arxiv.org/pdf/2212.14106v1)

**Authors**: Chao Chen, Chenghua Guo, Guixiang Ma, Xi Zhang, Sihong Xie

**Abstract**: Gradient-based explanation is the cornerstone of explainable deep networks, but it has been shown to be vulnerable to adversarial attacks. However, existing works measure the explanation robustness based on $\ell_p$-norm, which can be counter-intuitive to humans, who only pay attention to the top few salient features. We propose explanation ranking thickness as a more suitable explanation robustness metric. We then present a new practical adversarial attacking goal for manipulating explanation rankings. To mitigate the ranking-based attacks while maintaining computational feasibility, we derive surrogate bounds of the thickness that involve expensive sampling and integration. We use a multi-objective approach to analyze the convergence of a gradient-based attack to confirm that the explanation robustness can be measured by the thickness metric. We conduct experiments on various network architectures and diverse datasets to prove the superiority of the proposed methods, while the widely accepted Hessian-based curvature smoothing approaches are not as robust as our method.

摘要: 基于梯度的解释是可解释深度网络的基石，但已被证明容易受到对手攻击。然而，现有的工作基于$\ell_p$-范数来衡量解释的稳健性，这对于只关注最显著的几个特征的人类来说可能是违反直觉的。我们提出解释排序厚度作为一种更合适的解释稳健性度量。然后，我们提出了一种新的实用的对抗性攻击目标，用于操纵解释排名。为了在保持计算可行性的同时减轻基于排名的攻击，我们推导了包含昂贵的采样和积分的厚度的代理界限。我们使用多目标方法对基于梯度的攻击的收敛进行了分析，以确认解释的稳健性可以通过厚度度量来度量。我们在不同的网络结构和不同的数据集上进行了实验，以证明所提出的方法的优越性，而被广泛接受的基于Hessian的曲率平滑方法不如我们的方法健壮。



## **43. ObjectSeeker: Certifiably Robust Object Detection against Patch Hiding Attacks via Patch-agnostic Masking**

ObjectSeeker：基于补丁无关掩蔽的抗补丁隐藏攻击的可证明鲁棒目标检测 cs.CV

IEEE Symposium on Security and Privacy 2023; extended version

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2202.01811v2) [paper-pdf](http://arxiv.org/pdf/2202.01811v2)

**Authors**: Chong Xiang, Alexander Valtchanov, Saeed Mahloujifar, Prateek Mittal

**Abstract**: Object detectors, which are widely deployed in security-critical systems such as autonomous vehicles, have been found vulnerable to patch hiding attacks. An attacker can use a single physically-realizable adversarial patch to make the object detector miss the detection of victim objects and undermine the functionality of object detection applications. In this paper, we propose ObjectSeeker for certifiably robust object detection against patch hiding attacks. The key insight in ObjectSeeker is patch-agnostic masking: we aim to mask out the entire adversarial patch without knowing the shape, size, and location of the patch. This masking operation neutralizes the adversarial effect and allows any vanilla object detector to safely detect objects on the masked images. Remarkably, we can evaluate ObjectSeeker's robustness in a certifiable manner: we develop a certification procedure to formally determine if ObjectSeeker can detect certain objects against any white-box adaptive attack within the threat model, achieving certifiable robustness. Our experiments demonstrate a significant (~10%-40% absolute and ~2-6x relative) improvement in certifiable robustness over the prior work, as well as high clean performance (~1% drop compared with undefended models).

摘要: 被广泛部署在自动驾驶车辆等安全关键系统中的对象探测器被发现容易受到补丁隐藏攻击。攻击者可以使用单个物理上可实现的对抗性补丁来使对象检测器错过对受害者对象的检测，并破坏对象检测应用程序的功能。在这篇文章中，我们提出了一种针对补丁隐藏攻击的可证明稳健的目标检测方法。ObjectSeeker的关键洞察是与补丁无关的掩蔽：我们的目标是在不知道补丁的形状、大小和位置的情况下掩盖整个敌对补丁。这种掩蔽操作消除了对抗性效应，并允许任何香草对象检测器安全地检测掩蔽图像上的对象。值得注意的是，我们可以以可认证的方式评估ObjectSeeker的健壮性：我们开发了一个认证过程来正式确定ObjectSeeker是否可以检测到某些对象，以抵御威胁模型中的任何白盒自适应攻击，从而实现可认证的健壮性。我们的实验表明，与以前的工作相比，可证明的健壮性有了显著的提高(~10%-40%的绝对和~2-6倍的相对)，以及高的清洁性能(与未防御的模型相比下降了~1%)。



## **44. Internal Wasserstein Distance for Adversarial Attack and Defense**

对抗性攻防的瓦瑟斯坦内部距离 cs.LG

Due to the need for major revisions to the submitted manuscript, we  request to withdraw this manuscript from arXiv

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2103.07598v3) [paper-pdf](http://arxiv.org/pdf/2103.07598v3)

**Authors**: Mingkui Tan, Shuhai Zhang, Jiezhang Cao, Jincheng Li, Yanwu Xu

**Abstract**: Deep neural networks (DNNs) are known to be vulnerable to adversarial attacks that would trigger misclassification of DNNs but may be imperceptible to human perception. Adversarial defense has been important ways to improve the robustness of DNNs. Existing attack methods often construct adversarial examples relying on some metrics like the $\ell_p$ distance to perturb samples. However, these metrics can be insufficient to conduct adversarial attacks due to their limited perturbations. In this paper, we propose a new internal Wasserstein distance (IWD) to capture the semantic similarity of two samples, and thus it helps to obtain larger perturbations than currently used metrics such as the $\ell_p$ distance We then apply the internal Wasserstein distance to perform adversarial attack and defense. In particular, we develop a novel attack method relying on IWD to calculate the similarities between an image and its adversarial examples. In this way, we can generate diverse and semantically similar adversarial examples that are more difficult to defend by existing defense methods. Moreover, we devise a new defense method relying on IWD to learn robust models against unseen adversarial examples. We provide both thorough theoretical and empirical evidence to support our methods.

摘要: 深度神经网络(DNN)容易受到敌意攻击，这种攻击可能会导致DNN的错误分类，但可能无法被人类感知到。对抗性防御已成为提高DNN健壮性的重要途径。现有的攻击方法通常依赖于诸如$\ell_p$距离之类的度量来构建敌意示例来扰动样本。然而，由于其有限的扰动，这些指标可能不足以进行对抗性攻击。本文提出了一种新的内部Wasserstein距离(IWD)来刻画两个样本之间的语义相似性，从而有助于获得比现有度量(如$\ell_p$距离)更大的扰动。特别是，我们开发了一种新的攻击方法，利用IWD来计算图像与其对手示例之间的相似度。通过这种方式，我们可以生成不同的和语义相似的对抗性例子，这些例子用现有的防御方法更难防御。此外，我们设计了一种新的防御方法，依赖于IWD来学习针对未知对手实例的稳健模型。我们提供了充分的理论和经验证据来支持我们的方法。



## **45. Thermal Heating in ReRAM Crossbar Arrays: Challenges and Solutions**

ReRAM交叉开关阵列中的热加热：挑战和解决方案 cs.AR

18 pages

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13707v1) [paper-pdf](http://arxiv.org/pdf/2212.13707v1)

**Authors**: Kamilya Smagulova, Mohammed E. Fouda, Ahmed Eltawil

**Abstract**: Increasing popularity of deep-learning-powered applications raises the issue of vulnerability of neural networks to adversarial attacks. In other words, hardly perceptible changes in input data lead to the output error in neural network hindering their utilization in applications that involve decisions with security risks. A number of previous works have already thoroughly evaluated the most commonly used configuration - Convolutional Neural Networks (CNNs) against different types of adversarial attacks. Moreover, recent works demonstrated transferability of the some adversarial examples across different neural network models. This paper studied robustness of the new emerging models such as SpinalNet-based neural networks and Compact Convolutional Transformers (CCT) on image classification problem of CIFAR-10 dataset. Each architecture was tested against four White-box attacks and three Black-box attacks. Unlike VGG and SpinalNet models, attention-based CCT configuration demonstrated large span between strong robustness and vulnerability to adversarial examples. Eventually, the study of transferability between VGG, VGG-inspired SpinalNet and pretrained CCT 7/3x1 models was conducted. It was shown that despite high effectiveness of the attack on the certain individual model, this does not guarantee the transferability to other models.

摘要: 深度学习驱动的应用程序越来越受欢迎，这引发了神经网络在敌意攻击中的脆弱性问题。换言之，输入数据几乎不可察觉的变化导致神经网络的输出误差，阻碍了它们在涉及安全风险的决策应用中的应用。许多以前的工作已经彻底评估了最常用的结构-卷积神经网络(CNN)对抗不同类型的对手攻击。此外，最近的工作证明了一些对抗性例子在不同的神经网络模型之间的可转移性。研究了基于SpinalNet神经网络和紧凑型卷积变换(CCT)等新兴模型对CIFAR-10数据集图像分类问题的稳健性。每个架构都针对四个白盒攻击和三个黑盒攻击进行了测试。与VGG和SpinalNet模型不同，基于注意力的CCT配置在强鲁棒性和对敌意例子的脆弱性之间表现出很大的跨度。最后，对VGG、VGG启发的SpinalNet和预先训练的CCT7/3x1模型之间的可转移性进行了研究。结果表明，尽管对特定个体模型的攻击很有效，但这并不能保证可移植到其他模型上。



## **46. Publishing Efficient On-device Models Increases Adversarial Vulnerability**

发布高效的设备上模型会增加对手的脆弱性 cs.CR

Accepted to IEEE SaTML 2023

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13700v1) [paper-pdf](http://arxiv.org/pdf/2212.13700v1)

**Authors**: Sanghyun Hong, Nicholas Carlini, Alexey Kurakin

**Abstract**: Recent increases in the computational demands of deep neural networks (DNNs) have sparked interest in efficient deep learning mechanisms, e.g., quantization or pruning. These mechanisms enable the construction of a small, efficient version of commercial-scale models with comparable accuracy, accelerating their deployment to resource-constrained devices.   In this paper, we study the security considerations of publishing on-device variants of large-scale models. We first show that an adversary can exploit on-device models to make attacking the large models easier. In evaluations across 19 DNNs, by exploiting the published on-device models as a transfer prior, the adversarial vulnerability of the original commercial-scale models increases by up to 100x. We then show that the vulnerability increases as the similarity between a full-scale and its efficient model increase. Based on the insights, we propose a defense, $similarity$-$unpairing$, that fine-tunes on-device models with the objective of reducing the similarity. We evaluated our defense on all the 19 DNNs and found that it reduces the transferability up to 90% and the number of queries required by a factor of 10-100x. Our results suggest that further research is needed on the security (or even privacy) threats caused by publishing those efficient siblings.

摘要: 最近，深度神经网络(DNN)计算需求的增加引起了人们对有效的深度学习机制的兴趣，例如量化或剪枝。这些机制使构建小型、高效的商业规模模型具有相当的精确度，加快了它们在资源受限设备上的部署。在本文中，我们研究了在设备上发布大规模模型变体的安全考虑因素。我们首先展示对手可以利用设备上的模型来使攻击大型模型变得更容易。在对19个DNN的评估中，通过利用已公布的设备上模型作为传输之前，原始商业规模模型的对手漏洞增加了高达100倍。然后，我们证明了脆弱性随着全尺度模型与其有效模型之间的相似性的增加而增加。基于这些见解，我们提出了一种辩护，即$相似性$-$不配对$，对设备上的模型进行微调，目标是减少相似性。我们在所有19个DNN上对我们的防御进行了评估，发现它将可转移性降低了高达90%，所需的查询数量减少了10-100倍。我们的结果表明，需要对发布这些高效兄弟姐妹所造成的安全(甚至隐私)威胁进行进一步的研究。



## **47. Learning When to Use Adaptive Adversarial Image Perturbations against Autonomous Vehicles**

学习何时对自主车辆使用自适应对抗性图像扰动 cs.RO

**SubmitDate**: 2022-12-28    [abs](http://arxiv.org/abs/2212.13667v1) [paper-pdf](http://arxiv.org/pdf/2212.13667v1)

**Authors**: Hyung-Jin Yoon, Hamidreza Jafarnejadsani, Petros Voulgaris

**Abstract**: The deep neural network (DNN) models for object detection using camera images are widely adopted in autonomous vehicles. However, DNN models are shown to be susceptible to adversarial image perturbations. In the existing methods of generating the adversarial image perturbations, optimizations take each incoming image frame as the decision variable to generate an image perturbation. Therefore, given a new image, the typically computationally-expensive optimization needs to start over as there is no learning between the independent optimizations. Very few approaches have been developed for attacking online image streams while considering the underlying physical dynamics of autonomous vehicles, their mission, and the environment. We propose a multi-level stochastic optimization framework that monitors an attacker's capability of generating the adversarial perturbations. Based on this capability level, a binary decision attack/not attack is introduced to enhance the effectiveness of the attacker. We evaluate our proposed multi-level image attack framework using simulations for vision-guided autonomous vehicles and actual tests with a small indoor drone in an office environment. The results show our method's capability to generate the image attack in real-time while monitoring when the attacker is proficient given state estimates.

摘要: 利用摄像机图像进行目标检测的深度神经网络(DNN)模型在自动驾驶车辆中得到了广泛的应用。然而，DNN模型被证明容易受到对抗性图像扰动的影响。在现有的产生对抗性图像扰动的方法中，优化将每一进入的图像帧作为决策变量来产生图像扰动。因此，给定一个新的图像，通常计算代价高昂的优化需要重新开始，因为在独立的优化之间没有学习。在考虑自动驾驶车辆的基本物理动力学、它们的任务和环境的同时，很少有人开发出攻击在线图像流的方法。我们提出了一个多层次随机优化框架，用于监控攻击者产生敌意扰动的能力。基于这一能力水平，引入了二元判决攻击/非攻击，以增强攻击者的有效性。我们通过对视觉制导自动驾驶车辆的模拟和办公室环境中小型室内无人机的实际测试来评估我们提出的多级图像攻击框架。结果表明，我们的方法能够实时生成图像攻击，同时监控攻击者何时熟练掌握给定的状态估计。



## **48. A Comprehensive Test Pattern Generation Approach Exploiting SAT Attack for Logic Locking**

一种利用SAT攻击进行逻辑锁定的综合测试码生成方法 cs.CR

12 pages, 7 figures, 5 tables

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2204.11307v3) [paper-pdf](http://arxiv.org/pdf/2204.11307v3)

**Authors**: Yadi Zhong, Ujjwal Guin

**Abstract**: The need for reducing manufacturing defect escape in today's safety-critical applications requires increased fault coverage. However, generating a test set using commercial automatic test pattern generation (ATPG) tools that lead to zero-defect escape is still an open problem. It is challenging to detect all stuck-at faults to reach 100% fault coverage. In parallel, the hardware security community has been actively involved in developing solutions for logic locking to prevent IP piracy. Locks (e.g., XOR gates) are inserted in different locations of the netlist so that an adversary cannot determine the secret key. Unfortunately, the Boolean satisfiability (SAT) based attack, introduced in [1], can break different logic locking schemes in minutes. In this paper, we propose a novel test pattern generation approach using the powerful SAT attack on logic locking. A stuck-at fault is modeled as a locked gate with a secret key. Our modeling of stuck-at faults preserves the property of fault activation and propagation. We show that the input pattern that determines the key is a test for the stuck-at fault. We propose two different approaches for test pattern generation. First, a single stuck-at fault is targeted, and a corresponding locked circuit with one key bit is created. This approach generates one test pattern per fault. Second, we consider a group of faults and convert the circuit to its locked version with multiple key bits. The inputs obtained from the SAT tool are the test set for detecting this group of faults. Our approach is able to find test patterns for hard-to-detect faults that were previously failed in commercial ATPG tools. The proposed test pattern generation approach can efficiently detect redundant faults present in a circuit. We demonstrate the effectiveness of the approach on ITC'99 benchmarks. The results show that we can achieve a perfect fault coverage reaching 100%.

摘要: 在当今的安全关键应用中，减少制造缺陷逃逸的需要需要增加故障覆盖率。然而，使用商业自动测试模式生成(ATPG)工具生成测试集以实现零缺陷逃逸仍然是一个未解决的问题。要检测所有固定故障以达到100%的故障覆盖率是具有挑战性的。与此同时，硬件安全界一直积极参与开发逻辑锁定解决方案，以防止知识产权盗版。锁(例如，异或门)被插入网表的不同位置，使得对手不能确定密钥。不幸的是，在[1]中引入的基于布尔可满足性(SAT)的攻击可以在几分钟内破解不同的逻辑锁定方案。在本文中，我们提出了一种新的测试模式生成方法，该方法利用了对逻辑锁的强大SAT攻击。一个顽固的错误被建模为一扇锁着的门和一把密钥。我们对固定故障的建模保留了故障激活和传播的性质。我们证明了决定关键字的输入模式是对固定错误的测试。我们提出了两种不同的测试模式生成方法。首先，针对单个固定故障，创建具有一个密钥位的相应锁定电路。该方法为每个故障生成一个测试模式。其次，我们考虑一组故障，并将电路转换为具有多个密钥位的锁定版本。从SAT工具获得的输入是用于检测这组故障的测试集。我们的方法能够为以前在商业ATPG工具中失败的难以检测的故障找到测试模式。提出的测试码生成方法可以有效地检测电路中存在的冗余故障。我们在ITC‘99基准上证明了该方法的有效性。结果表明，我们可以达到100%的完美故障覆盖率。



## **49. EDoG: Adversarial Edge Detection For Graph Neural Networks**

EDoG：图神经网络的对抗性边缘检测 cs.LG

Accepted by IEEE Conference on Secure and Trustworthy Machine  Learning 2023

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2212.13607v1) [paper-pdf](http://arxiv.org/pdf/2212.13607v1)

**Authors**: Xiaojun Xu, Yue Yu, Hanzhang Wang, Alok Lal, Carl A. Gunter, Bo Li

**Abstract**: Graph Neural Networks (GNNs) have been widely applied to different tasks such as bioinformatics, drug design, and social networks. However, recent studies have shown that GNNs are vulnerable to adversarial attacks which aim to mislead the node or subgraph classification prediction by adding subtle perturbations. Detecting these attacks is challenging due to the small magnitude of perturbation and the discrete nature of graph data. In this paper, we propose a general adversarial edge detection pipeline EDoG without requiring knowledge of the attack strategies based on graph generation. Specifically, we propose a novel graph generation approach combined with link prediction to detect suspicious adversarial edges. To effectively train the graph generative model, we sample several sub-graphs from the given graph data. We show that since the number of adversarial edges is usually low in practice, with low probability the sampled sub-graphs will contain adversarial edges based on the union bound. In addition, considering the strong attacks which perturb a large number of edges, we propose a set of novel features to perform outlier detection as the preprocessing for our detection. Extensive experimental results on three real-world graph datasets including a private transaction rule dataset from a major company and two types of synthetic graphs with controlled properties show that EDoG can achieve above 0.8 AUC against four state-of-the-art unseen attack strategies without requiring any knowledge about the attack type; and around 0.85 with knowledge of the attack type. EDoG significantly outperforms traditional malicious edge detection baselines. We also show that an adaptive attack with full knowledge of our detection pipeline is difficult to bypass it.

摘要: 图形神经网络被广泛应用于生物信息学、药物设计、社会网络等领域。然而，最近的研究表明，GNN很容易受到敌意攻击，这些攻击的目的是通过添加微妙的扰动来误导节点或子图分类预测。由于扰动的小幅度和图形数据的离散性质，检测这些攻击是具有挑战性的。在本文中，我们提出了一种通用的对抗性边缘检测流水线EDoG，它不需要了解基于图生成的攻击策略。具体地说，我们提出了一种新的结合链接预测的图生成方法来检测可疑的敌对边。为了有效地训练图形生成模型，我们从给定的图形数据中采样了几个子图。我们证明，由于实际中对抗性边的数量通常很少，因此采样子图中包含基于并界值的对抗性边的概率很低。此外，考虑到干扰了大量边缘的强攻击，我们提出了一组新的特征来进行孤立点检测，作为检测的预处理。在三个真实世界图数据集上的大量实验结果表明，对于四种最先进的不可见攻击策略，EDoG在不需要任何攻击类型知识的情况下可以达到0.8AUC以上；在知道攻击类型的情况下，EDoG可以达到0.85左右的AUC。EDoG的性能明显优于传统的恶意边缘检测基线。我们还表明，在完全了解我们的检测管道的情况下，自适应攻击很难绕过它。



## **50. Towards Transferable Unrestricted Adversarial Examples with Minimum Changes**

以最小的变化走向可转让的不受限制的对抗性例子 cs.CV

Accepted at SaTML 2023

**SubmitDate**: 2022-12-27    [abs](http://arxiv.org/abs/2201.01102v2) [paper-pdf](http://arxiv.org/pdf/2201.01102v2)

**Authors**: Fangcheng Liu, Chao Zhang, Hongyang Zhang

**Abstract**: Transfer-based adversarial example is one of the most important classes of black-box attacks. However, there is a trade-off between transferability and imperceptibility of the adversarial perturbation. Prior work in this direction often requires a fixed but large $\ell_p$-norm perturbation budget to reach a good transfer success rate, leading to perceptible adversarial perturbations. On the other hand, most of the current unrestricted adversarial attacks that aim to generate semantic-preserving perturbations suffer from weaker transferability to the target model. In this work, we propose a geometry-aware framework to generate transferable adversarial examples with minimum changes. Analogous to model selection in statistical machine learning, we leverage a validation model to select the best perturbation budget for each image under both the $\ell_{\infty}$-norm and unrestricted threat models. We propose a principled method for the partition of training and validation models by encouraging intra-group diversity while penalizing extra-group similarity. Extensive experiments verify the effectiveness of our framework on balancing imperceptibility and transferability of the crafted adversarial examples. The methodology is the foundation of our entry to the CVPR'21 Security AI Challenger: Unrestricted Adversarial Attacks on ImageNet, in which we ranked 1st place out of 1,559 teams and surpassed the runner-up submissions by 4.59% and 23.91% in terms of final score and average image quality level, respectively. Code is available at https://github.com/Equationliu/GA-Attack.

摘要: 基于转移的对抗性例子是黑盒攻击中最重要的一类。然而，在对抗性扰动的可转移性和不可感知性之间存在权衡。以前在这个方向上的工作通常需要固定但很大的$\ell_p$-范数扰动预算才能达到良好的转移成功率，从而导致可感知的对抗性扰动。另一方面，目前大多数旨在产生语义保持扰动的无限制对抗性攻击都存在对目标模型的可转移性较弱的问题。在这项工作中，我们提出了一个几何感知框架，以最小的变化生成可转移的对抗性实例。与统计机器学习中的模型选择类似，我们利用验证模型为每个图像在$-范数和无限制威胁模型下选择最佳扰动预算。我们提出了一种原则性的划分训练和验证模型的方法，鼓励组内多样性，同时惩罚组外相似性。大量实验验证了该框架在平衡恶意例子的不可感知性和可转移性方面的有效性。该方法是我们进入CVPR‘21 Security AI Challenger：对ImageNet的无限制对手攻击的基础，我们在1,559个团队中排名第一，在最终得分和平均图像质量水平方面分别以4.59%和23.91%的优势超过亚军提交的项目。代码可在https://github.com/Equationliu/GA-Attack.上找到



