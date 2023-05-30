# Latest Adversarial Attack Papers
**update at 2023-05-30 09:51:22**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. BITE: Textual Backdoor Attacks with Iterative Trigger Injection**

BITE：使用迭代触发器注入的文本后门攻击 cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2205.12700v3) [paper-pdf](http://arxiv.org/pdf/2205.12700v3)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstract**: Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a "backdoor" into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it is possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and a set of "trigger words". These trigger words are iteratively identified and injected into the target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four text classification datasets show that our proposed attack is significantly more effective than baseline methods while maintaining decent stealthiness, raising alarm on the usage of untrusted training data. We further propose a defense method named DeBITE based on potential trigger word removal, which outperforms existing methods in defending against BITE and generalizes well to handling other backdoor attacks.

摘要: 后门攻击已成为对NLP系统的新威胁。通过提供有毒的训练数据，敌手可以在受害者模型中嵌入“后门”，这允许满足某些文本模式(例如，包含关键字)的输入实例被预测为敌手选择的目标标签。在这篇文章中，我们证明了设计一种既隐蔽(即难以察觉)又有效(即具有高攻击成功率)的后门攻击是可能的。我们提出了BITE，一种毒化训练数据的后门攻击，以建立目标标签和一组“触发词”之间的强关联。这些触发词被迭代地识别并通过自然的词级扰动注入到目标标签实例中。有毒的训练数据指示受害者模型预测包含触发词的输入上的目标标签，从而形成后门。在四个文本分类数据集上的实验表明，我们提出的攻击方法比基准方法更有效，同时保持了良好的隐蔽性，并对不可信训练数据的使用发出了警报。在此基础上，提出了一种基于潜在触发字移除的防御方法DeBITE，该方法在防御BITE攻击方面优于已有方法，并能很好地推广到应对其他后门攻击。



## **2. Fundamental Limitations of Alignment in Large Language Models**

大型语言模型中对齐的基本限制 cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2304.11082v2) [paper-pdf](http://arxiv.org/pdf/2304.11082v2)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.

摘要: 开发与人类交互的语言模型的一个重要方面是使他们的行为对人类用户有用而无害。这通常是通过调整模型来实现的，这种方式增强了期望的行为，抑制了不期望的行为，这一过程称为对齐。在本文中，我们提出了一种名为行为期望界限(BEB)的理论方法，它允许我们正式地研究大型语言模型中对齐的几个固有特征和限制。重要的是，我们证明了对于模型表现出的有限概率的任何行为，都存在可以触发模型输出该行为的提示，其概率随着提示的长度增加而增加。这意味着，任何减弱不受欢迎的行为但不能完全消除它的对准过程，在对抗提示攻击时都是不安全的。此外，我们的框架暗示了一种机制，通过这种机制，领先的对齐方法，如来自人类反馈的强化学习，增加了LLM被提示进入不希望看到的行为的倾向。此外，我们在我们的BEB框架中包括了人物角色的概念，并发现通过促使模型表现为特定的人物角色，通常不太可能在模型中表现的行为可以被带到前面。这一理论结果正在由所谓的当代“聊天GPT越狱”大规模实验证明，在这种情况下，敌对用户通过触发LLM充当恶意角色来欺骗LLM打破其对齐护栏。我们的结果暴露了LLM对齐方面的根本限制，并将设计可靠的机制以确保人工智能安全的必要性放在了首位。



## **3. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

基于骨架的图卷积神经网络鲁棒性的傅立叶分析 cs.CV

17 pages, 13 figures

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17939v1) [paper-pdf](http://arxiv.org/pdf/2305.17939v1)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.

摘要: 利用傅立叶分析，我们研究了基于骨架的动作识别的图卷积神经网络(GCNS)的稳健性和脆弱性。我们采用联合傅里叶变换(JFT)，即图傅里叶变换(GFT)和离散傅立叶变换(DFT)的组合，来检验经过对抗性训练的GCNS对敌意攻击和常见腐败的健壮性。在NTU RGB+D数据集上的实验结果表明，对抗性训练不会在对抗性攻击和低频扰动之间引入稳健性权衡，而这通常发生在基于卷积神经网络的图像分类中。这一发现表明，在基于骨架的动作识别中，对抗性训练是一种增强对对抗性攻击和常见腐败的稳健性的实用方法。此外，我们发现傅立叶方法不能解释对骨骼部分遮挡破坏的脆弱性，这突出了它的局限性。这些发现扩展了我们对GCNS健壮性的理解，潜在地指导了基于骨骼的动作识别的更健壮的学习方法的发展。



## **4. NaturalFinger: Generating Natural Fingerprint with Generative Adversarial Networks**

NaturalFinger：利用产生式对抗网络生成自然指纹 cs.CV

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17868v1) [paper-pdf](http://arxiv.org/pdf/2305.17868v1)

**Authors**: Kang Yang, Kunhao Lai

**Abstract**: Deep neural network (DNN) models have become a critical asset of the model owner as training them requires a large amount of resource (i.e. labeled data). Therefore, many fingerprinting schemes have been proposed to safeguard the intellectual property (IP) of the model owner against model extraction and illegal redistribution. However, previous schemes adopt unnatural images as the fingerprint, such as adversarial examples and noisy images, which can be easily perceived and rejected by the adversary. In this paper, we propose NaturalFinger which generates natural fingerprint with generative adversarial networks (GANs). Besides, our proposed NaturalFinger fingerprints the decision difference areas rather than the decision boundary, which is more robust. The application of GAN not only allows us to generate more imperceptible samples, but also enables us to generate unrestricted samples to explore the decision boundary.To demonstrate the effectiveness of our fingerprint approach, we evaluate our approach against four model modification attacks including adversarial training and two model extraction attacks. Experiments show that our approach achieves 0.91 ARUC value on the FingerBench dataset (154 models), exceeding the optimal baseline (MetaV) over 17\%.

摘要: 深度神经网络(DNN)模型已经成为模型所有者的重要资产，因为训练它们需要大量的资源(即标记数据)。因此，许多指纹方案被提出来保护模型所有者的知识产权(IP)，以防止模型提取和非法再分发。然而，以往的方案采用非自然的图像作为指纹，如敌意图像和噪声图像，这些图像很容易被攻击者感知和拒绝。在本文中，我们提出了利用生成性对抗网络(GANS)生成自然指纹的NaturalFinger。此外，我们提出的NaturalFinger指纹提取决策差异区而不是决策边界，从而更健壮。该方法不仅可以生成更多的隐蔽样本，还可以生成不受限制的样本来探索决策边界。为了验证指纹方法的有效性，我们对包括对抗性训练和两个模型提取攻击在内的四种模型修改攻击进行了评估。实验表明，该方法在FingerB边数据集(154个模型)上达到了0.91ARUC值，超过了最优基线(MetaV)17。



## **5. NOTABLE: Transferable Backdoor Attacks Against Prompt-based NLP Models**

值得注意：针对基于提示的NLP模型的可转移后门攻击 cs.CL

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17826v1) [paper-pdf](http://arxiv.org/pdf/2305.17826v1)

**Authors**: Kai Mei, Zheng Li, Zhenting Wang, Yang Zhang, Shiqing Ma

**Abstract**: Prompt-based learning is vulnerable to backdoor attacks. Existing backdoor attacks against prompt-based models consider injecting backdoors into the entire embedding layers or word embedding vectors. Such attacks can be easily affected by retraining on downstream tasks and with different prompting strategies, limiting the transferability of backdoor attacks. In this work, we propose transferable backdoor attacks against prompt-based models, called NOTABLE, which is independent of downstream tasks and prompting strategies. Specifically, NOTABLE injects backdoors into the encoders of PLMs by utilizing an adaptive verbalizer to bind triggers to specific words (i.e., anchors). It activates the backdoor by pasting input with triggers to reach adversary-desired anchors, achieving independence from downstream tasks and prompting strategies. We conduct experiments on six NLP tasks, three popular models, and three prompting strategies. Empirical results show that NOTABLE achieves superior attack performance (i.e., attack success rate over 90% on all the datasets), and outperforms two state-of-the-art baselines. Evaluations on three defenses show the robustness of NOTABLE. Our code can be found at https://github.com/RU-System-Software-and-Security/Notable.

摘要: 基于提示的学习很容易受到后门攻击。现有针对基于提示的模型的后门攻击考虑向整个嵌入层或单词嵌入向量中注入后门。这类攻击很容易受到下游任务再培训和不同提示策略的影响，限制了后门攻击的可转移性。在这项工作中，我们提出了独立于下游任务和提示策略的针对提示模型的可转移后门攻击，称为显著模型。具体地说，值得注意的是，通过利用自适应动词器将触发器绑定到特定单词(即锚)，将后门注入PLM的编码器。它通过粘贴带有触发器的输入来激活后门，以到达对手想要的锚，实现对下游任务和提示策略的独立。我们在六个自然语言处理任务、三个流行模式和三个提示策略上进行了实验。实验结果表明，该算法取得了较好的攻击性能(即在所有数据集上的攻击成功率都在90%以上)，并超过了两个最先进的基线。对三种防御措施的评估表明，该算法具有较好的稳健性。我们的代码可以在https://github.com/RU-System-Software-and-Security/Notable.上找到



## **6. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

DiffProtect：使用扩散模型生成用于面部隐私保护的敌意示例 cs.CV

Code will be available at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.13625v2) [paper-pdf](http://arxiv.org/pdf/2305.13625v2)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.

摘要: 日益普及的面部识别(FR)系统引发了对个人隐私的严重担忧，特别是对数十亿在社交媒体上公开分享照片的用户来说。已经进行了几次尝试，以保护个人不被未经授权的FR系统利用敌意攻击来生成加密的面部图像来识别。然而，现有的方法存在视觉质量差或攻击成功率低的问题，这限制了它们的实用性。近年来，扩散模型在图像生成方面取得了巨大的成功。在这项工作中，我们问：扩散模型能否被用来生成对抗性例子，以提高视觉质量和攻击性能？我们提出了DiffProtect，它利用扩散自动编码器在FR系统上产生语义上有意义的扰动。大量实验表明，与最先进的方法相比，DiffProtect生成的加密图像看起来更自然，同时实现了显著更高的攻击成功率，例如，在CelebA-HQ和FFHQ数据集上的绝对改进了24.5%和25.1%。



## **7. Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness**

放大特洛伊木马网络：通过放大深层神经网络的固有弱点来攻击它们 cs.CR

Published Sep 2022 in Neurocomputing

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17688v1) [paper-pdf](http://arxiv.org/pdf/2305.17688v1)

**Authors**: Zhanhao Hu, Jun Zhu, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works found that deep neural networks (DNNs) can be fooled by adversarial examples, which are crafted by adding adversarial noise on clean inputs. The accuracy of DNNs on adversarial examples will decrease as the magnitude of the adversarial noise increase. In this study, we show that DNNs can be also fooled when the noise is very small under certain circumstances. This new type of attack is called Amplification Trojan Attack (ATAttack). Specifically, we use a trojan network to transform the inputs before sending them to the target DNN. This trojan network serves as an amplifier to amplify the inherent weakness of the target DNN. The target DNN, which is infected by the trojan network, performs normally on clean data while being more vulnerable to adversarial examples. Since it only transforms the inputs, the trojan network can hide in DNN-based pipelines, e.g. by infecting the pre-processing procedure of the inputs before sending them to the DNNs. This new type of threat should be considered in developing safe DNNs.

摘要: 最近的工作发现，深度神经网络(DNN)可以被敌意例子愚弄，这些例子是通过在干净的输入上添加对抗性噪声来构建的。DNN对对抗性样本的准确率会随着对抗性噪声的增加而降低。在这项研究中，我们证明了在某些情况下，当噪声非常小时，DNN也可以被愚弄。这种新型攻击被称为放大特洛伊木马攻击(ATAttack)。具体地说，我们使用特洛伊木马网络在将输入发送到目标DNN之前对其进行转换。此特洛伊木马网络充当放大器，放大目标DNN的固有弱点。被特洛伊木马网络感染的目标DNN在干净的数据上正常执行，但更容易受到敌意示例的攻击。由于它只转换输入，特洛伊木马网络可以隐藏在基于DNN的管道中，例如通过在将输入发送到DNN之前感染输入的预处理过程。在开发安全的DNN时，应该考虑这种新型的威胁。



## **8. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

空间和时间上的威胁模型：E2EE消息传递应用程序的案例研究 cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2301.05653v2) [paper-pdf](http://arxiv.org/pdf/2301.05653v2)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.

摘要: 威胁建模是安全系统工程的基础，应该考虑到系统运行的环境。另一方面，威胁的技术复杂性和系统攻击面的不断演变是一个不可避免的现实。在这项工作中，我们探索现实世界系统工程反映不断变化的威胁背景的程度。为此，我们研究了六个广泛使用的端到端加密移动消息传递应用程序的桌面客户端，以了解它们在空间(当在新平台上启用客户端时)和时间(当出现新威胁时)对其威胁模型进行调整的程度。我们对这些桌面客户端进行了短暂的恶意访问试验，并分析了两个流行的威胁诱导框架STRIDE和LINDDUN的结果。结果表明，系统设计者需要在系统运行的不断变化的环境中认识到威胁，更重要的是，通过以行政边界内的信任边界不能违反安全和隐私属性的方式重新应对信任边界来缓解这些威胁。对信任边界作用域及其与管理边界的关系的这种细致入微的理解允许更好地管理共享组件，包括使用安全缺省值保护它们。



## **9. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

SneakyPrompt：评估文本到图像生成模型的安全过滤器的健壮性 cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.12082v2) [paper-pdf](http://arxiv.org/pdf/2305.12082v2)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.

摘要: 从文本到图像的生成模型，如稳定扩散模型和Dall$\CDOT$E2模型自问世以来，由于其在现实世界中的广泛应用而引起了人们的广泛关注。文本到图像生成模型的一个具有挑战性的问题是生成非安全工作(NSFW)内容，例如与暴力和成人有关的内容。因此，一种常见的做法是部署所谓的安全过滤器，即根据文本或图像特征阻止NSFW内容。以前的工作已经研究了这种安全过滤器的可能旁路。然而，现有的工作主要是手动的，专门针对稳定扩散的官方安全过滤器。此外，根据我们的评估，稳定扩散安全过滤器的旁路比低至23.51%。在本文中，我们提出了第一个自动攻击框架，称为SneakyPrompt，用于评估最新的文本到图像生成模型中现实世界安全过滤器的稳健性。我们的主要见解是在生成NSFW图像的提示中搜索替代令牌，以便生成的提示(称为对抗性提示)绕过现有的安全过滤器。具体地说，SneakyPrompt利用强化学习(RL)来指导代理在语义相似性方面获得积极回报，并绕过成功。我们的评估表明，SneakyPrompt成功地使用在线模型DALL$\CDOT$E 2生成了NSFW内容，并启用了默认的闭箱安全过滤器。同时，我们还在一个稳定的扩散模型上部署了几个开源的最先进的安全过滤器，并表明SneakyPrompt不仅成功地生成了NSFW内容，而且在查询数量和图像质量方面都优于现有的对抗性攻击。



## **10. Tubes Among Us: Analog Attack on Automatic Speaker Identification**

我们之间的管道：对自动说话人识别的模拟攻击 cs.LG

Published at USENIX Security 2023  https://www.usenix.org/conference/usenixsecurity23/presentation/ahmed

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2202.02751v2) [paper-pdf](http://arxiv.org/pdf/2202.02751v2)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstract**: Recent years have seen a surge in the popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. A large number of modern systems protect themselves against such attacks by targeting artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness detection, bringing into question their use in security-critical settings in real life, such as phone banking.

摘要: 近年来，由机器学习驱动的声学个人设备的受欢迎程度激增。然而，事实证明，机器学习很容易受到对抗性例子的影响。大量现代系统通过瞄准人为攻击来保护自己免受此类攻击，即，它们部署机制来检测在生成对抗性例子时缺乏人的参与。然而，这些防御隐含地假设人类没有能力制造有意义和有针对性的对抗性例子。在本文中，我们证明了这个基本假设是错误的。特别是，我们证明了对于像说话人识别这样的任务，人类能够在几乎不需要成本和监督的情况下直接产生模拟的对抗性例子：通过简单地通过管道说话，对手可靠地在ML模型的眼中模仿其他说话人进行说话人识别。我们的发现延伸到了其他一系列声学-生物识别任务，如活体检测，这让人质疑它们在现实生活中对安全至关重要的环境中的使用，比如电话银行。



## **11. PowerGAN: A Machine Learning Approach for Power Side-Channel Attack on Compute-in-Memory Accelerators**

PowerGAN：一种针对内存计算加速器电源侧通道攻击的机器学习方法 cs.CR

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2304.11056v2) [paper-pdf](http://arxiv.org/pdf/2304.11056v2)

**Authors**: Ziyu Wang, Yuting Wu, Yongmo Park, Sangmin Yoo, Xinxin Wang, Jason K. Eshraghian, Wei D. Lu

**Abstract**: Analog compute-in-memory (CIM) systems are promising for deep neural network (DNN) inference acceleration due to their energy efficiency and high throughput. However, as the use of DNNs expands, protecting user input privacy has become increasingly important. In this paper, we identify a potential security vulnerability wherein an adversary can reconstruct the user's private input data from a power side-channel attack, under proper data acquisition and pre-processing, even without knowledge of the DNN model. We further demonstrate a machine learning-based attack approach using a generative adversarial network (GAN) to enhance the data reconstruction. Our results show that the attack methodology is effective in reconstructing user inputs from analog CIM accelerator power leakage, even at large noise levels and after countermeasures are applied. Specifically, we demonstrate the efficacy of our approach on an example of U-Net inference chip for brain tumor detection, and show the original magnetic resonance imaging (MRI) medical images can be successfully reconstructed even at a noise-level of 20% standard deviation of the maximum power signal value. Our study highlights a potential security vulnerability in analog CIM accelerators and raises awareness of using GAN to breach user privacy in such systems.

摘要: 模拟内存计算(CIM)系统由于其高能量效率和高吞吐量，在深度神经网络(DNN)推理加速中具有广阔的应用前景。然而，随着DNN的使用范围扩大，保护用户输入隐私变得越来越重要。在本文中，我们发现了一个潜在的安全漏洞，在该漏洞中，攻击者即使在不了解DNN模型的情况下，也可以在适当的数据获取和预处理下，从功率侧通道攻击中重构用户的私人输入数据。我们进一步展示了一种基于机器学习的攻击方法，使用生成性对抗网络(GAN)来增强数据重构。我们的结果表明，即使在大噪声水平和采取对策后，该攻击方法也能有效地从模拟CIM加速器功率泄漏中恢复用户输入。具体地说，我们在U-Net推理芯片用于脑肿瘤检测的例子上验证了该方法的有效性，并表明即使在最大功率信号值的20%标准差的噪声水平下，也可以成功地重建原始磁共振成像(MRI)医学图像。我们的研究突出了模拟CIM加速器中的潜在安全漏洞，并提高了人们对在此类系统中使用GAN来侵犯用户隐私的认识。



## **12. Two Heads are Better than One: Towards Better Adversarial Robustness by Combining Transduction and Rejection**

两个头比一个头好：通过结合转导和拒绝来实现更好的对手稳健性 cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17528v1) [paper-pdf](http://arxiv.org/pdf/2305.17528v1)

**Authors**: Nils Palumbo, Yang Guo, Xi Wu, Jiefeng Chen, Yingyu Liang, Somesh Jha

**Abstract**: Both transduction and rejection have emerged as important techniques for defending against adversarial perturbations. A recent work by Tram\`er showed that, in the rejection-only case (no transduction), a strong rejection-solution can be turned into a strong (but computationally inefficient) non-rejection solution. This detector-to-classifier reduction has been mostly applied to give evidence that certain claims of strong selective-model solutions are susceptible, leaving the benefits of rejection unclear. On the other hand, a recent work by Goldwasser et al. showed that rejection combined with transduction can give provable guarantees (for certain problems) that cannot be achieved otherwise. Nevertheless, under recent strong adversarial attacks (GMSA, which has been shown to be much more effective than AutoAttack against transduction), Goldwasser et al.'s work was shown to have low performance in a practical deep-learning setting. In this paper, we take a step towards realizing the promise of transduction+rejection in more realistic scenarios. Theoretically, we show that a novel application of Tram\`er's classifier-to-detector technique in the transductive setting can give significantly improved sample-complexity for robust generalization. While our theoretical construction is computationally inefficient, it guides us to identify an efficient transductive algorithm to learn a selective model. Extensive experiments using state of the art attacks (AutoAttack, GMSA) show that our solutions provide significantly better robust accuracy.

摘要: 转导和拒绝都已成为防御敌意干扰的重要技术。Tramer最近的一项工作表明，在只有拒绝的情况下(没有转导)，强拒绝解可以变成强(但计算效率低)的非拒绝解。这种探测器到分类器的减少主要是为了提供证据，证明某些声称的强选择性模型解决方案是敏感的，留下了拒绝的好处不清楚。另一方面，Goldwasser等人最近的一项工作。表明拒绝和转导相结合可以提供可证明的保证(对于某些问题)，而不是通过其他方式无法实现的。然而，在最近强大的对手攻击下(GMSA，已被证明比AutoAttack对抗转导要有效得多)，Goldwasser等人的工作被证明在实际的深度学习环境中表现不佳。在这篇论文中，我们朝着在更现实的情景中实现转导+拒绝的承诺迈出了一步。理论上，我们证明了Tram‘er的分类器到检测器技术在换能式环境中的一种新的应用可以显著改善样本复杂度以实现稳健的泛化。虽然我们的理论构建在计算上效率低下，但它指导我们识别一个有效的换能式算法来学习一个选择的模型。使用最先进的攻击(AutoAttack，GMSA)进行的广泛实验表明，我们的解决方案提供了显著更好的健壮性准确性。



## **13. Backdooring Neural Code Search**

回溯神经编码搜索 cs.SE

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17506v1) [paper-pdf](http://arxiv.org/pdf/2305.17506v1)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.

摘要: 重用在线存储库中的现成代码片段是一种常见的做法，这显著提高了软件开发人员的工作效率。为了找到所需的代码片段，开发人员通过自然语言查询求助于代码搜索引擎。因此，神经代码搜索模型是许多此类引擎的幕后推手。这些模型是基于深度学习的，由于其令人印象深刻的性能而获得了大量关注。然而，这些模型的安全性方面的研究很少。特别是，攻击者可以在神经代码搜索模型中注入后门，该模型返回带有安全/隐私问题的错误代码，甚至是易受攻击的代码。这可能会影响下游软件(例如股票交易系统和自动驾驶)，并导致经济损失和/或危及生命的事件。在这篇文章中，我们证明了这种攻击是可行的，并且可以相当隐蔽。只需修改一个变量/函数名称，攻击者就可以使有错误/易受攻击的代码排在前11%。我们的攻击BADCODE具有特殊的触发生成和注入过程，使攻击更有效和隐蔽。在两个神经编码搜索模型上进行了评估，结果表明我们的攻击性能比基线高60%。我们的用户研究表明，基于F1比分，我们的攻击比基线更隐蔽两倍。



## **14. Modeling Adversarial Attack on Pre-trained Language Models as Sequential Decision Making**

基于预训练语言模型的对抗性攻击序列决策建模 cs.CL

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17440v1) [paper-pdf](http://arxiv.org/pdf/2305.17440v1)

**Authors**: Xuanjie Fang, Sijie Cheng, Yang Liu, Wei Wang

**Abstract**: Pre-trained language models (PLMs) have been widely used to underpin various downstream tasks. However, the adversarial attack task has found that PLMs are vulnerable to small perturbations. Mainstream methods adopt a detached two-stage framework to attack without considering the subsequent influence of substitution at each step. In this paper, we formally model the adversarial attack task on PLMs as a sequential decision-making problem, where the whole attack process is sequential with two decision-making problems, i.e., word finder and word substitution. Considering the attack process can only receive the final state without any direct intermediate signals, we propose to use reinforcement learning to find an appropriate sequential attack path to generate adversaries, named SDM-Attack. Extensive experimental results show that SDM-Attack achieves the highest attack success rate with a comparable modification rate and semantic similarity to attack fine-tuned BERT. Furthermore, our analyses demonstrate the generalization and transferability of SDM-Attack. The code is available at https://github.com/fduxuan/SDM-Attack.

摘要: 预训练语言模型(PLM)已被广泛用于支持各种下游任务。然而，对抗性攻击任务发现，PLM很容易受到小扰动的影响。主流方法采用分离的两阶段框架进行攻击，没有考虑每一步替换的后续影响。在本文中，我们将PLM上的对抗性攻击任务形式化地建模为一个序列决策问题，其中整个攻击过程是由两个决策问题(即单词查找和单词替换)组成的序列。考虑到攻击过程只能得到最终状态，没有任何直接的中间信号，我们提出使用强化学习来寻找一条合适的顺序攻击路径来生成对手，称为SDM-攻击。大量的实验结果表明，SDM-Attack具有最高的攻击成功率，其修改率和语义相似度与改进后的BERT算法相当。此外，我们的分析还证明了SDM攻击的泛化和可转移性。代码可在https://github.com/fduxuan/SDM-Attack.上获得



## **15. On the Importance of Backbone to the Adversarial Robustness of Object Detectors**

论脊椎对目标检测器对抗稳健性的重要性 cs.CV

12 pages

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17438v1) [paper-pdf](http://arxiv.org/pdf/2305.17438v1)

**Authors**: Xiao Li, Hang Chen, Xiaolin Hu

**Abstract**: Object detection is a critical component of various security-sensitive applications, such as autonomous driving and video surveillance. However, existing deep learning-based object detectors are vulnerable to adversarial attacks, which poses a significant challenge to their reliability and safety. Through experiments, we found that existing works on improving the adversarial robustness of object detectors have given a false sense of security. We argue that using adversarially pre-trained backbone networks is essential for enhancing the adversarial robustness of object detectors. We propose a simple yet effective recipe for fast adversarial fine-tuning on object detectors with adversarially pre-trained backbones. Without any modifications to the structure of object detectors, our recipe achieved significantly better adversarial robustness than previous works. Moreover, we explore the potential of different modern object detectors to improve adversarial robustness using our recipe and demonstrate several interesting findings. Our empirical results set a new milestone and deepen the understanding of adversarially robust object detection. Code and trained checkpoints will be publicly available.

摘要: 目标检测是各种安全敏感应用的关键组件，例如自动驾驶和视频监控。然而，现有的基于深度学习的目标检测器容易受到敌意攻击，这对其可靠性和安全性构成了巨大的挑战。通过实验我们发现，现有的提高目标检测器的对抗性健壮性的工作给人一种错误的安全感。我们认为，使用对抗性预训练的骨干网络对于增强目标检测器的对抗性健壮性是必不可少的。我们提出了一个简单但有效的配方，用于在具有对抗性预训练主干的对象检测器上进行快速对抗性微调。在不对对象检测器的结构进行任何修改的情况下，我们的配方比以前的工作获得了更好的对抗健壮性。此外，我们探索了不同的现代对象检测器的潜力，以提高对手健壮性使用我们的食谱，并展示了几个有趣的发现。我们的实验结果建立了一个新的里程碑，加深了对敌方鲁棒目标检测的理解。代码和训练有素的检查站将公开提供。



## **16. Rethinking Adversarial Policies: A Generalized Attack Formulation and Provable Defense in Multi-Agent RL**

对抗性策略的再思考：多智能体RL中的广义攻击公式和可证明防御 cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17342v1) [paper-pdf](http://arxiv.org/pdf/2305.17342v1)

**Authors**: Xiangyu Liu, Souradip Chakraborty, Yanchao Sun, Furong Huang

**Abstract**: Most existing works consider direct perturbations of victim's state/action or the underlying transition dynamics to show vulnerability of reinforcement learning agents under adversarial attacks. However, such direct manipulation may not always be feasible in practice. In this paper, we consider another common and realistic attack setup: in a multi-agent RL setting with well-trained agents, during deployment time, the victim agent $\nu$ is exploited by an attacker who controls another agent $\alpha$ to act adversarially against the victim using an \textit{adversarial policy}. Prior attack models under such setup do not consider that the attacker can confront resistance and thus can only take partial control of the agent $\alpha$, as well as introducing perceivable ``abnormal'' behaviors that are easily detectable. A provable defense against these adversarial policies is also lacking. To resolve these issues, we introduce a more general attack formulation that models to what extent the adversary is able to control the agent to produce the adversarial policy. Based on such a generalized attack framework, the attacker can also regulate the state distribution shift caused by the attack through an attack budget, and thus produce stealthy adversarial policies that can exploit the victim agent. Furthermore, we provide the first provably robust defenses with convergence guarantee to the most robust victim policy via adversarial training with timescale separation, in sharp contrast to adversarial training in supervised learning which may only provide {\it empirical} defenses.

摘要: 现有的大多数工作都考虑了受害者状态/动作的直接扰动或潜在的转移动力学，以显示强化学习代理在对抗性攻击下的脆弱性。然而，这种直接操纵在实践中并不总是可行的。在本文中，我们考虑了另一种常见且现实的攻击设置：在具有训练有素的代理的多代理RL环境中，在部署期间，受害者代理$\nu$被攻击者利用，该攻击者控制另一代理$\α$以使用对抗策略对受害者进行敌意操作。在这种情况下，以前的攻击模型不认为攻击者可以抵抗，因此只能部分控制代理$\α$，并引入容易检测的可感知的“异常”行为。针对这些对抗性政策，也缺乏可证明的防御措施。为了解决这些问题，我们引入了一种更一般的攻击公式，该公式对对手能够在多大程度上控制代理以产生对抗策略进行建模。基于这种广义攻击框架，攻击者还可以通过攻击预算来调整攻击引起的状态分布变化，从而产生可以利用受害者代理的隐蔽对抗性策略。此外，我们通过时间尺度分离的对抗性训练，提供了第一个具有收敛保证的可证明的稳健防御，以保证最稳健的受害者策略，这与监督学习中的对抗性训练形成鲜明对比，后者可能只提供经验上的防御。



## **17. Certifiably Robust Reinforcement Learning through Model-Based Abstract Interpretation**

基于模型抽象解释的可证明稳健强化学习 cs.LG

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2301.11374v2) [paper-pdf](http://arxiv.org/pdf/2301.11374v2)

**Authors**: Chenxi Yang, Greg Anderson, Swarat Chaudhuri

**Abstract**: We present a reinforcement learning (RL) framework in which the learned policy comes with a machine-checkable certificate of provable adversarial robustness. Our approach, called CAROL, learns a model of the environment. In each learning iteration, it uses the current version of this model and an external abstract interpreter to construct a differentiable signal for provable robustness. This signal is used to guide learning, and the abstract interpretation used to construct it directly leads to the robustness certificate returned at convergence. We give a theoretical analysis that bounds the worst-case accumulative reward of CAROL. We also experimentally evaluate CAROL on four MuJoCo environments with continuous state and action spaces. On these tasks, CAROL learns policies that, when contrasted with policies from the state-of-the-art robust RL algorithms, exhibit: (i) markedly enhanced certified performance lower bounds; and (ii) comparable performance under empirical adversarial attacks.

摘要: 我们提出了一种强化学习(RL)框架，其中学习的策略带有机器可检查的可证明对抗健壮性的证书。我们的方法被称为卡罗尔，它学习了一个环境模型。在每次学习迭代中，它使用该模型的当前版本和外部抽象解释器来构造可区分信号以实现可证明的稳健性。该信号用于指导学习，用于构造该信号的抽象解释直接导致在收敛时返回的稳健性证书。我们对Carol的最坏情况下的累积奖励进行了理论分析。我们还在具有连续状态和动作空间的四个MuJoCo环境中对Carol进行了实验评估。在这些任务中，Carol学习的策略与来自最先进的健壮RL算法的策略相比，表现出：(I)显著增强的认证性能下限；以及(Ii)在经验对手攻击下的可比性能。



## **18. Adversarial Attacks on Online Learning to Rank with Click Feedback**

在线学习点击反馈排名的对抗性攻击 cs.LG

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.17071v1) [paper-pdf](http://arxiv.org/pdf/2305.17071v1)

**Authors**: Jinhang Zuo, Zhiyao Zhang, Zhiyong Wang, Shuai Li, Mohammad Hajiesmaili, Adam Wierman

**Abstract**: Online learning to rank (OLTR) is a sequential decision-making problem where a learning agent selects an ordered list of items and receives feedback through user clicks. Although potential attacks against OLTR algorithms may cause serious losses in real-world applications, little is known about adversarial attacks on OLTR. This paper studies attack strategies against multiple variants of OLTR. Our first result provides an attack strategy against the UCB algorithm on classical stochastic bandits with binary feedback, which solves the key issues caused by bounded and discrete feedback that previous works can not handle. Building on this result, we design attack algorithms against UCB-based OLTR algorithms in position-based and cascade models. Finally, we propose a general attack strategy against any algorithm under the general click model. Each attack algorithm manipulates the learning agent into choosing the target attack item $T-o(T)$ times, incurring a cumulative cost of $o(T)$. Experiments on synthetic and real data further validate the effectiveness of our proposed attack algorithms.

摘要: 在线学习排序是一个顺序决策问题，学习代理选择一个有序的项目列表，并通过用户点击接收反馈。虽然针对OLTR算法的潜在攻击可能会在现实世界的应用中造成严重的损失，但对OLTR的对抗性攻击知之甚少。本文研究了针对多个OLTR变种的攻击策略。我们的第一个结果给出了一种针对经典的二进制反馈随机盗贼的UCB算法的攻击策略，解决了以往工作不能处理的有界和离散反馈引起的关键问题。在此基础上，设计了针对基于位置模型和级联模型的基于UCB的OLTR算法的攻击算法。最后，在一般点击模型下，针对任意算法提出了一种通用攻击策略。每个攻击算法操纵学习代理选择目标攻击项$T-o(T)$次，产生$o(T)$的累积成本。在人工数据和真实数据上的实验进一步验证了本文提出的攻击算法的有效性。



## **19. Leveraging characteristics of the output probability distribution for identifying adversarial audio examples**

利用输出概率分布的特征来识别对抗性音频示例 cs.SD

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.17000v1) [paper-pdf](http://arxiv.org/pdf/2305.17000v1)

**Authors**: Matías P. Pizarro B., Dorothea Kolossa, Asja Fischer

**Abstract**: Adversarial attacks represent a security threat to machine learning based automatic speech recognition (ASR) systems. To prevent such attacks we propose an adversarial example detection strategy applicable to any ASR system that predicts a probability distribution over output tokens in each time step. We measure a set of characteristics of this distribution: the median, maximum, and minimum over the output probabilities, the entropy, and the Jensen-Shannon divergence of the distributions of subsequent time steps. Then, we fit a Gaussian distribution to the characteristics observed for benign data. By computing the likelihood of incoming new audio we can distinguish malicious inputs from samples from clean data with an area under the receiving operator characteristic (AUROC) higher than 0.99, which drops to 0.98 for less-quality audio. To assess the robustness of our method we build adaptive attacks. This reduces the AUROC to 0.96 but results in more noisy adversarial clips.

摘要: 敌意攻击是对基于机器学习的自动语音识别(ASR)系统的安全威胁。为了防止此类攻击，我们提出了一种敌意示例检测策略，该策略适用于任何ASR系统，预测每个时间步输出令牌上的概率分布。我们测量了这一分布的一组特征：输出概率的中值、最大值和最小值、熵和后续时间步分布的Jensen-Shannon散度。然后，我们对观察到的良性数据的特征进行了高斯分布的拟合。通过计算传入新音频的可能性，我们可以区分恶意输入和来自干净数据的样本，其中接收操作员特征(AUROC)下的区域高于0.99，对于质量较差的音频，AUROC下降到0.98。为了评估该方法的健壮性，我们构建了自适应攻击。这将AUROC降低到0.96，但会产生更多嘈杂的对抗性片段。



## **20. Training Socially Aligned Language Models in Simulated Human Society**

在模拟人类社会中训练社会一致的语言模型 cs.CL

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16960v1) [paper-pdf](http://arxiv.org/pdf/2305.16960v1)

**Authors**: Ruibo Liu, Ruixin Yang, Chenyan Jia, Ge Zhang, Denny Zhou, Andrew M. Dai, Diyi Yang, Soroush Vosoughi

**Abstract**: Social alignment in AI systems aims to ensure that these models behave according to established societal values. However, unlike humans, who derive consensus on value judgments through social interaction, current language models (LMs) are trained to rigidly replicate their training corpus in isolation, leading to subpar generalization in unfamiliar scenarios and vulnerability to adversarial attacks. This work presents a novel training paradigm that permits LMs to learn from simulated social interactions. In comparison to existing methodologies, our approach is considerably more scalable and efficient, demonstrating superior performance in alignment benchmarks and human evaluations. This paradigm shift in the training of LMs brings us a step closer to developing AI systems that can robustly and accurately reflect societal norms and values.

摘要: 人工智能系统中的社会一致性旨在确保这些模型的行为符合既定的社会价值观。然而，与通过社交互动就价值判断达成共识的人类不同，当前的语言模型(LMS)被训练成孤立地僵硬地复制他们的训练语料库，导致在不熟悉的场景中的泛化能力不佳，并且容易受到对手攻击。这项工作提出了一种新的训练范式，允许LMS从模拟的社会互动中学习。与现有方法相比，我们的方法可伸缩性更强，效率更高，在比对基准和人工评估方面表现出卓越的性能。LMS培训的这种范式转变使我们离开发能够有力而准确地反映社会规范和价值观的人工智能系统又近了一步。



## **21. Trust-Aware Resilient Control and Coordination of Connected and Automated Vehicles**

互联和自动化车辆的信任感知弹性控制和协调 cs.MA

Keywords: Resilient control and coordination, Cybersecurity, Safety  guaranteed coordination, Connected And Autonomous Vehicles

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16818v1) [paper-pdf](http://arxiv.org/pdf/2305.16818v1)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: Security is crucial for cyber-physical systems, such as a network of Connected and Automated Vehicles (CAVs) cooperating to navigate through a road network safely. In this paper, we tackle the security of a cooperating network of CAVs in conflict areas by identifying the critical adversarial objectives from the point of view of uncooperative/malicious agents from our preliminary study, which are (i) safety violations resulting in collisions, and (ii) traffic jams. We utilize a trust framework (and our work doesn't depend on the specific choice of trust/reputation framework) to propose a resilient control and coordination framework that mitigates the effects of such agents and guarantees safe coordination. A class of attacks that can be used to achieve the adversarial objectives is Sybil attacks, which we use to validate our proposed framework through simulation studies. Besides that, we propose an attack detection and mitigation scheme using the trust framework. The simulation results demonstrate that our proposed scheme can detect fake CAVs during a Sybil attack, guarantee safe coordination, and mitigate their effects.

摘要: 安全对于网络物理系统至关重要，例如互联和自动化车辆(CAV)网络，它们相互协作，在道路网络中安全导航。在本文中，我们通过从不合作/恶意代理的角度来确定冲突区域中协作的CAV网络的安全问题，即(I)导致碰撞的安全违规行为和(Ii)交通拥堵。我们利用信任框架(我们的工作不依赖于信任/声誉框架的具体选择)来提出一个弹性控制和协调框架，以减轻此类代理的影响，并保证安全的协调。Sybil攻击是一类可以用来达到对抗目标的攻击，我们用它来通过仿真研究来验证我们提出的框架。此外，我们还提出了一种使用信任框架的攻击检测和缓解方案。仿真结果表明，该方案能够在Sybil攻击过程中检测到伪CAV，保证安全协作，并减轻它们的影响。



## **22. Robust Nonparametric Regression under Poisoning Attack**

中毒攻击下的稳健非参数回归 math.ST

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2305.16771v1) [paper-pdf](http://arxiv.org/pdf/2305.16771v1)

**Authors**: Puning Zhao, Zhiguo Wan

**Abstract**: This paper studies robust nonparametric regression, in which an adversarial attacker can modify the values of up to $q$ samples from a training dataset of size $N$. Our initial solution is an M-estimator based on Huber loss minimization. Compared with simple kernel regression, i.e. the Nadaraya-Watson estimator, this method can significantly weaken the impact of malicious samples on the regression performance. We provide the convergence rate as well as the corresponding minimax lower bound. The result shows that, with proper bandwidth selection, $\ell_\infty$ error is minimax optimal. The $\ell_2$ error is optimal if $q\lesssim \sqrt{N/\ln^2 N}$, but is suboptimal with larger $q$. The reason is that this estimator is vulnerable if there are many attacked samples concentrating in a small region. To address this issue, we propose a correction method by projecting the initial estimate to the space of Lipschitz functions. The final estimate is nearly minimax optimal for arbitrary $q$, up to a $\ln N$ factor.

摘要: 本文研究了稳健非参数回归，其中敌方攻击者可以从大小为$N$的训练数据集中修改多达$Q$个样本值。我们的初始解是基于Huber损失最小化的M-估计量。与简单的核回归，即Nadaraya-Watson估计相比，该方法可以显著减弱恶意样本对回归性能的影响。我们给出了收敛速度以及相应的极小极大下界。结果表明，在适当选择带宽的情况下，误差值是最小极大最优的。如果$q\less sim\sqrt{N/\ln^2 N}$，则$\ell_2$错误是最优的，但如果$q$较大，则是次优的。原因是，如果有许多攻击样本集中在一个小区域，则该估计器是脆弱的。为了解决这个问题，我们提出了一种将初始估计投影到Lipschitz函数空间的修正方法。对于任意的$Q$，最终估计几乎是极小极大最优的，直到$ln N$因子。



## **23. Mitigating Adversarial Attacks by Distributing Different Copies to Different Users**

通过将不同的副本分发给不同的用户来缓解敌意攻击 cs.CR

**SubmitDate**: 2023-05-26    [abs](http://arxiv.org/abs/2111.15160v3) [paper-pdf](http://arxiv.org/pdf/2111.15160v3)

**Authors**: Jiyi Zhang, Han Fang, Wesley Joon-Wie Tann, Ke Xu, Chengfang Fang, Ee-Chien Chang

**Abstract**: Machine learning models are vulnerable to adversarial attacks. In this paper, we consider the scenario where a model is distributed to multiple buyers, among which a malicious buyer attempts to attack another buyer. The malicious buyer probes its copy of the model to search for adversarial samples and then presents the found samples to the victim's copy of the model in order to replicate the attack. We point out that by distributing different copies of the model to different buyers, we can mitigate the attack such that adversarial samples found on one copy would not work on another copy. We observed that training a model with different randomness indeed mitigates such replication to a certain degree. However, there is no guarantee and retraining is computationally expensive. A number of works extended the retraining method to enhance the differences among models. However, a very limited number of models can be produced using such methods and the computational cost becomes even higher. Therefore, we propose a flexible parameter rewriting method that directly modifies the model's parameters. This method does not require additional training and is able to generate a large number of copies in a more controllable manner, where each copy induces different adversarial regions. Experimentation studies show that rewriting can significantly mitigate the attacks while retaining high classification accuracy. For instance, on GTSRB dataset with respect to Hop Skip Jump attack, using attractor-based rewriter can reduce the success rate of replicating the attack to 0.5% while independently training copies with different randomness can reduce the success rate to 6.5%. From this study, we believe that there are many further directions worth exploring.

摘要: 机器学习模型容易受到敌意攻击。在本文中，我们考虑了这样的场景：一个模型被分发给多个买家，其中一个恶意买家试图攻击另一个买家。恶意买家探测其模型副本以搜索对抗性样本，然后将找到的样本呈现给受害者的模型副本以复制攻击。我们指出，通过将模型的不同副本分发给不同的买家，我们可以缓解攻击，使得在一个副本上发现的敌对样本在另一个副本上不起作用。我们观察到，训练一个具有不同随机性的模型确实在一定程度上减轻了这种复制。然而，这并不能保证，再培训在计算上是昂贵的。一些工作扩展了再训练方法，以增强模型之间的差异。然而，使用这种方法可以产生的模型数量非常有限，计算成本变得更高。因此，我们提出了一种灵活的参数重写方法，可以直接修改模型的参数。这种方法不需要额外的训练，并且能够以更可控的方式生成大量副本，其中每个副本诱导不同的对抗性区域。实验研究表明，重写可以在保持较高分类准确率的同时显著缓解攻击。例如，在针对跳跃攻击的GTSRB数据集上，使用基于吸引子的重写器可以将复制攻击的成功率降低到0.5%，而独立训练不同随机性的副本可以将复制成功率降低到6.5%。从本次研究来看，我们认为还有很多值得进一步探索的方向。



## **24. IMBERT: Making BERT Immune to Insertion-based Backdoor Attacks**

Imbert：使Bert对基于插入的后门攻击免疫 cs.CL

accepted to Third Workshop on Trustworthy Natural Language Processing

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16503v1) [paper-pdf](http://arxiv.org/pdf/2305.16503v1)

**Authors**: Xuanli He, Jun Wang, Benjamin Rubinstein, Trevor Cohn

**Abstract**: Backdoor attacks are an insidious security threat against machine learning models. Adversaries can manipulate the predictions of compromised models by inserting triggers into the training phase. Various backdoor attacks have been devised which can achieve nearly perfect attack success without affecting model predictions for clean inputs. Means of mitigating such vulnerabilities are underdeveloped, especially in natural language processing. To fill this gap, we introduce IMBERT, which uses either gradients or self-attention scores derived from victim models to self-defend against backdoor attacks at inference time. Our empirical studies demonstrate that IMBERT can effectively identify up to 98.5% of inserted triggers. Thus, it significantly reduces the attack success rate while attaining competitive accuracy on the clean dataset across widespread insertion-based attacks compared to two baselines. Finally, we show that our approach is model-agnostic, and can be easily ported to several pre-trained transformer models.

摘要: 后门攻击是对机器学习模型的潜在安全威胁。攻击者可以通过在训练阶段插入触发器来操纵受损模型的预测。已经设计了各种后门攻击，可以在不影响对干净输入的模型预测的情况下实现近乎完美的攻击成功。缓解此类漏洞的手段还不够完善，特别是在自然语言处理方面。为了填补这一空白，我们引入了Imbert，它使用来自受害者模型的梯度或自我注意分数来在推理时防御后门攻击。我们的实验研究表明，Imbert可以有效地识别高达98.5%的插入触发器。因此，与两个基线相比，它显著降低了攻击成功率，同时在广泛的基于插入的攻击中获得了与干净数据集相当的准确性。最后，我们证明了我们的方法是模型不可知的，并且可以很容易地移植到几个预先训练的变压器模型上。



## **25. Diffusion-Based Adversarial Sample Generation for Improved Stealthiness and Controllability**

基于扩散的改进隐蔽性和可控性的对抗样本生成 cs.CV

code repo: https://github.com/xavihart/Diff-PGD

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16494v1) [paper-pdf](http://arxiv.org/pdf/2305.16494v1)

**Authors**: Haotian Xue, Alexandre Araujo, Bin Hu, Yongxin Chen

**Abstract**: Neural networks are known to be susceptible to adversarial samples: small variations of natural examples crafted to deliberately mislead the models. While they can be easily generated using gradient-based techniques in digital and physical scenarios, they often differ greatly from the actual data distribution of natural images, resulting in a trade-off between strength and stealthiness. In this paper, we propose a novel framework dubbed Diffusion-Based Projected Gradient Descent (Diff-PGD) for generating realistic adversarial samples. By exploiting a gradient guided by a diffusion model, Diff-PGD ensures that adversarial samples remain close to the original data distribution while maintaining their effectiveness. Moreover, our framework can be easily customized for specific tasks such as digital attacks, physical-world attacks, and style-based attacks. Compared with existing methods for generating natural-style adversarial samples, our framework enables the separation of optimizing adversarial loss from other surrogate losses (e.g., content/smoothness/style loss), making it more stable and controllable. Finally, we demonstrate that the samples generated using Diff-PGD have better transferability and anti-purification power than traditional gradient-based methods. Code will be released in https://github.com/xavihart/Diff-PGD

摘要: 众所周知，神经网络容易受到敌意样本的影响，这些样本是自然样本的微小变体，目的是故意误导模型。虽然在数字和物理场景中可以很容易地使用基于梯度的技术来生成它们，但它们往往与自然图像的实际数据分布有很大差异，导致在强度和隐蔽性之间进行权衡。在本文中，我们提出了一种新的框架，称为基于扩散的投影梯度下降(DIFF-PGD)，用于生成真实的对抗性样本。通过利用扩散模型引导的梯度，DIFF-PGD在保持有效性的同时，确保对手样本保持接近原始数据分布。此外，我们的框架可以很容易地针对特定任务进行定制，例如数字攻击、物理世界攻击和基于样式的攻击。与现有的生成自然风格对抗性样本的方法相比，我们的框架能够将优化对抗性损失与其他代理损失(例如，内容/流畅度/风格损失)分离，使其更加稳定和可控。最后，我们证明了DIFF-PGD生成的样本比传统的基于梯度的方法具有更好的可转移性和抗净化能力。代码将在https://github.com/xavihart/Diff-PGD中发布



## **26. Path Defense in Dynamic Defender-Attacker Blotto Games (dDAB) with Limited Information**

有限信息动态防御者-攻击者Blotto博弈(DDAB)中的路径防御 cs.GT

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2204.04176v2) [paper-pdf](http://arxiv.org/pdf/2204.04176v2)

**Authors**: Austin K. Chen, Bryce L. Ferguson, Daigo Shishika, Michael Dorothy, Jason R. Marden, George J. Pappas, Vijay Kumar

**Abstract**: We consider a path guarding problem in dynamic Defender-Attacker Blotto games (dDAB), where a team of robots must defend a path in a graph against adversarial agents. Multi-robot systems are particularly well suited to this application, as recent work has shown the effectiveness of these systems in related areas such as perimeter defense and surveillance. When designing a defender policy that guarantees the defense of a path, information about the adversary and the environment can be helpful and may reduce the number of resources required by the defender to achieve a sufficient level of security. In this work, we characterize the necessary and sufficient number of assets needed to guarantee the defense of a shortest path between two nodes in dDAB games when the defender can only detect assets within $k$-hops of a shortest path. By characterizing the relationship between sensing horizon and required resources, we show that increasing the sensing capability of the defender greatly reduces the number of defender assets needed to defend the path.

摘要: 我们考虑了动态防御者-攻击者Blotto博弈(DDAB)中的路径保护问题，其中一组机器人必须防御图中的一条路径以对抗对手代理。多机器人系统特别适合这一应用，因为最近的研究表明，这些系统在周边防御和监视等相关领域是有效的。在设计保证路径防御的防御方策略时，有关对手和环境的信息可能会有所帮助，并且可以减少防御方实现足够安全级别所需的资源数量。在这项工作中，我们刻画了当防御者只能检测到最短路径$k$-跳内的资产时，保证dDAB博弈中两个节点之间最短路径的防御所需的必要且足够数量的资产。通过描述感知范围和所需资源之间的关系，我们表明，增加防御者的感知能力可以极大地减少防御路径所需的防御者资产的数量。



## **27. Don't Retrain, Just Rewrite: Countering Adversarial Perturbations by Rewriting Text**

不要重新训练，只需重写：通过重写文本来对抗对抗性的干扰 cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16444v1) [paper-pdf](http://arxiv.org/pdf/2305.16444v1)

**Authors**: Ashim Gupta, Carter Wood Blum, Temma Choji, Yingjie Fei, Shalin Shah, Alakananda Vempala, Vivek Srikumar

**Abstract**: Can language models transform inputs to protect text classifiers against adversarial attacks? In this work, we present ATINTER, a model that intercepts and learns to rewrite adversarial inputs to make them non-adversarial for a downstream text classifier. Our experiments on four datasets and five attack mechanisms reveal that ATINTER is effective at providing better adversarial robustness than existing defense approaches, without compromising task accuracy. For example, on sentiment classification using the SST-2 dataset, our method improves the adversarial accuracy over the best existing defense approach by more than 4% with a smaller decrease in task accuracy (0.5% vs 2.5%). Moreover, we show that ATINTER generalizes across multiple downstream tasks and classifiers without having to explicitly retrain it for those settings. Specifically, we find that when ATINTER is trained to remove adversarial perturbations for the sentiment classification task on the SST-2 dataset, it even transfers to a semantically different task of news classification (on AGNews) and improves the adversarial robustness by more than 10%.

摘要: 语言模型能否转换输入以保护文本分类器免受敌意攻击？在这项工作中，我们提出了ATINTER，一个模型，拦截并学习重写对抗性输入，使它们对于下游文本分类器来说是非对抗性的。我们在四个数据集和五个攻击机制上的实验表明，ATINTER在提供比现有防御方法更好的对抗健壮性方面是有效的，而不会影响任务的准确性。例如，在使用SST-2数据集进行情感分类时，我们的方法比现有的最佳防御方法提高了4%以上的对手准确率，而任务准确率的下降较小(0.5%比2.5%)。此外，我们还展示了ATINTER在多个下游任务和分类器上的泛化，而不必显式地针对这些设置重新训练它。具体地说，我们发现，当ATINTER被训练来消除SST-2数据集上情感分类任务的敌意扰动时，它甚至转移到了语义不同的新闻分类任务(在AgNews上)，并将对手的健壮性提高了10%以上。



## **28. On the Robustness of Segment Anything**

关于Segment Anything的健壮性 cs.CV

22 pages

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.16220v1) [paper-pdf](http://arxiv.org/pdf/2305.16220v1)

**Authors**: Yihao Huang, Yue Cao, Tianlin Li, Felix Juefei-Xu, Di Lin, Ivor W. Tsang, Yang Liu, Qing Guo

**Abstract**: Segment anything model (SAM) has presented impressive objectness identification capability with the idea of prompt learning and a new collected large-scale dataset. Given a prompt (e.g., points, bounding boxes, or masks) and an input image, SAM is able to generate valid segment masks for all objects indicated by the prompts, presenting high generalization across diverse scenarios and being a general method for zero-shot transfer to downstream vision tasks. Nevertheless, it remains unclear whether SAM may introduce errors in certain threatening scenarios. Clarifying this is of significant importance for applications that require robustness, such as autonomous vehicles. In this paper, we aim to study the testing-time robustness of SAM under adversarial scenarios and common corruptions. To this end, we first build a testing-time robustness evaluation benchmark for SAM by integrating existing public datasets. Second, we extend representative adversarial attacks against SAM and study the influence of different prompts on robustness. Third, we study the robustness of SAM under diverse corruption types by evaluating SAM on corrupted datasets with different prompts. With experiments conducted on SA-1B and KITTI datasets, we find that SAM exhibits remarkable robustness against various corruptions, except for blur-related corruption. Furthermore, SAM remains susceptible to adversarial attacks, particularly when subjected to PGD and BIM attacks. We think such a comprehensive study could highlight the importance of the robustness issues of SAM and trigger a series of new tasks for SAM as well as downstream vision tasks.

摘要: 分段任意模型(SAM)以快速学习的思想和新收集的大规模数据集显示了令人印象深刻的客观性识别能力。在给定提示(例如，点、边界框或遮罩)和输入图像的情况下，SAM能够为提示所指示的所有对象生成有效的分段遮罩，呈现跨不同场景的高度概括性，并且是向下游视觉任务进行零镜头转移的通用方法。然而，目前尚不清楚SAM是否会在某些威胁场景中引入错误。澄清这一点对于需要健壮性的应用(如自动驾驶汽车)具有重要意义。在本文中，我们旨在研究SAM在对抗场景和常见腐败下的测试时间稳健性。为此，我们首先通过整合现有的公共数据集，为SAM构建了一个测试时健壮性评估基准。其次，扩展了针对SAM的典型对抗性攻击，并研究了不同提示对健壮性的影响。第三，通过对具有不同提示的受损数据集上的SAM进行评估，研究了SAM在不同破坏类型下的健壮性。通过在SA-1B和KITTI数据集上进行的实验，我们发现SAM对除模糊相关的腐败之外的各种腐败表现出了显著的健壮性。此外，SAM仍然容易受到对抗性攻击，特别是在受到PGD和BIM攻击时。我们认为，这样的综合研究可以突出SAM稳健性问题的重要性，并引发SAM以及下游视觉任务的一系列新任务。



## **29. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

ByzSecAgg：一种基于编码计算和向量承诺的联合学习抗拜占庭安全聚合方案 cs.CR

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2302.09913v2) [paper-pdf](http://arxiv.org/pdf/2302.09913v2)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose an efficient secure aggregation scheme for federated learning that is protected against Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving privacy of data against colluding nodes, requires some sort of secure secret sharing. However, communication load for secret sharing of long vectors of updates can be very high. To resolve this issue, in the proposed scheme, local updates are partitioned into smaller sub-vectors and shared using ramp secret sharing. However, this sharing method does not admit bi-linear computations, such as pairwise distance calculations, needed by outlier-detection algorithms. To overcome this issue, each user runs another round of ramp sharing, with different embedding of data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, the proposed scheme also uses a vector commitment method, in which the commitment size remains constant (i.e. does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process.

摘要: 在本文中，我们提出了一种有效的联合学习安全聚合方案，该方案可以防止拜占庭攻击和隐私泄露。处理个人更新以管理敌对行为，同时针对串通节点保护数据隐私，需要某种类型的安全秘密共享。然而，更新的长矢量的秘密共享的通信负荷可能非常高。为了解决这一问题，在所提出的方案中，局部更新被划分为更小的子向量，并使用斜坡秘密共享来共享。然而，这种共享方法不允许离群点检测算法所需的双线性计算，例如成对距离计算。为了解决这个问题，每个用户运行另一轮坡道共享，在共享多项式中嵌入不同的数据。这项技术受编码计算思想的启发，实现了两两距离的安全计算。此外，为了保持本地更新的完整性和私密性，该方案还使用了向量承诺方法，承诺大小保持不变(即不随本地更新的长度增加)，同时允许对秘密共享过程进行验证。



## **30. Impact of Adversarial Training on Robustness and Generalizability of Language Models**

对抗性训练对语言模型稳健性和泛化能力的影响 cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2211.05523v2) [paper-pdf](http://arxiv.org/pdf/2211.05523v2)

**Authors**: Enes Altinisik, Hassan Sajjad, Husrev Taha Sencar, Safa Messaoud, Sanjay Chawla

**Abstract**: Adversarial training is widely acknowledged as the most effective defense against adversarial attacks. However, it is also well established that achieving both robustness and generalization in adversarially trained models involves a trade-off. The goal of this work is to provide an in depth comparison of different approaches for adversarial training in language models. Specifically, we study the effect of pre-training data augmentation as well as training time input perturbations vs. embedding space perturbations on the robustness and generalization of transformer-based language models. Our findings suggest that better robustness can be achieved by pre-training data augmentation or by training with input space perturbation. However, training with embedding space perturbation significantly improves generalization. A linguistic correlation analysis of neurons of the learned models reveals that the improved generalization is due to 'more specialized' neurons. To the best of our knowledge, this is the first work to carry out a deep qualitative analysis of different methods of generating adversarial examples in adversarial training of language models.

摘要: 对抗性训练被广泛认为是对抗对抗性攻击的最有效的防御方法。然而，众所周知，在对抗性训练的模型中实现稳健性和泛化都需要权衡。这项工作的目标是深入比较语言模型中对抗性训练的不同方法。具体地说，我们研究了训练前数据扩充以及训练时间输入扰动与嵌入空间扰动对基于变压器的语言模型的稳健性和泛化的影响。我们的发现表明，通过预训练数据增强或通过输入空间扰动训练可以获得更好的稳健性。然而，嵌入空间扰动的训练显著提高了泛化能力。对学习模型的神经元进行的语言相关性分析表明，改进的泛化是由于更专业的神经元。据我们所知，这是第一次对语言模型对抗性训练中生成对抗性实例的不同方法进行深入的定性分析。



## **31. IDEA: Invariant Causal Defense for Graph Adversarial Robustness**

IDEA：图对抗健壮性的不变因果防御 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15792v1) [paper-pdf](http://arxiv.org/pdf/2305.15792v1)

**Authors**: Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Bingbing Xu, Xueqi Cheng

**Abstract**: Graph neural networks (GNNs) have achieved remarkable success in various tasks, however, their vulnerability to adversarial attacks raises concerns for the real-world applications. Existing defense methods can resist some attacks, but suffer unbearable performance degradation under other unknown attacks. This is due to their reliance on either limited observed adversarial examples to optimize (adversarial training) or specific heuristics to alter graph or model structures (graph purification or robust aggregation). In this paper, we propose an Invariant causal DEfense method against adversarial Attacks (IDEA), providing a new perspective to address this issue. The method aims to learn causal features that possess strong predictability for labels and invariant predictability across attacks, to achieve graph adversarial robustness. Through modeling and analyzing the causal relationships in graph adversarial attacks, we design two invariance objectives to learn the causal features. Extensive experiments demonstrate that our IDEA significantly outperforms all the baselines under both poisoning and evasion attacks on five benchmark datasets, highlighting the strong and invariant predictability of IDEA. The implementation of IDEA is available at https://anonymous.4open.science/r/IDEA_repo-666B.

摘要: 图神经网络(GNN)在各种任务中取得了显著的成功，但其对敌意攻击的脆弱性引起了人们对现实世界应用的担忧。现有的防御方法可以抵抗一些攻击，但在其他未知攻击下，性能会出现无法承受的下降。这是因为他们依赖于有限的观察到的对抗性例子来优化(对抗性训练)，或者依赖特定的启发式方法来改变图形或模型结构(图形净化或健壮聚合)。本文提出了一种对抗攻击的不变因果防御方法(IDEA)，为解决这一问题提供了一个新的视角。该方法旨在学习对标签具有很强可预测性和对攻击具有不变可预测性的因果特征，以实现图对抗的健壮性。通过对图对抗攻击中因果关系的建模和分析，设计了两个不变目标来学习因果关系的特征。大量实验表明，在五个基准数据集上，无论是中毒攻击还是逃避攻击，我们的算法都显著优于所有的基线算法，突出了IDEA算法强大且不变的可预测性。IDEA的实施可在https://anonymous.4open.science/r/IDEA_repo-666B.上获得



## **32. Healing Unsafe Dialogue Responses with Weak Supervision Signals**

用微弱的监督信号修复不安全的对话反应 cs.CL

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15757v1) [paper-pdf](http://arxiv.org/pdf/2305.15757v1)

**Authors**: Zi Liang, Pinghui Wang, Ruofei Zhang, Shuo Zhang, Xiaofan Ye Yi Huang, Junlan Feng

**Abstract**: Recent years have seen increasing concerns about the unsafe response generation of large-scale dialogue systems, where agents will learn offensive or biased behaviors from the real-world corpus. Some methods are proposed to address the above issue by detecting and replacing unsafe training examples in a pipeline style. Though effective, they suffer from a high annotation cost and adapt poorly to unseen scenarios as well as adversarial attacks. Besides, the neglect of providing safe responses (e.g. simply replacing with templates) will cause the information-missing problem of dialogues. To address these issues, we propose an unsupervised pseudo-label sampling method, TEMP, that can automatically assign potential safe responses. Specifically, our TEMP method groups responses into several clusters and samples multiple labels with an adaptively sharpened sampling strategy, inspired by the observation that unsafe samples in the clusters are usually few and distribute in the tail. Extensive experiments in chitchat and task-oriented dialogues show that our TEMP outperforms state-of-the-art models with weak supervision signals and obtains comparable results under unsupervised learning settings.

摘要: 近年来，人们越来越担心大规模对话系统的不安全响应生成，在这种系统中，代理将从现实世界的语料库中学习攻击性或偏见行为。提出了通过检测和替换流水线形式的不安全训练实例来解决上述问题的一些方法。虽然它们很有效，但它们存在着较高的注释成本，并且对未知场景以及对抗性攻击的适应性很差。此外，忽视提供安全的响应(例如，简单地用模板替换)将导致对话的信息缺失问题。为了解决这些问题，我们提出了一种无监督的伪标签抽样方法TEMP，该方法可以自动分配潜在的安全响应。具体地说，我们的TEMP方法将响应分组到几个簇中，并使用自适应锐化的采样策略对多个标签进行采样，灵感来自于观察到簇中不安全的样本通常很少，并且分布在尾部。在聊天和面向任务的对话中的大量实验表明，我们的TEMP优于具有弱监督信号的最新模型，并且在无监督学习环境下获得了类似的结果。



## **33. PEARL: Preprocessing Enhanced Adversarial Robust Learning of Image Deraining for Semantic Segmentation**

PEAR：用于语义分割的图像降维的预处理增强的对抗性稳健学习 cs.CV

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15709v1) [paper-pdf](http://arxiv.org/pdf/2305.15709v1)

**Authors**: Xianghao Jiao, Yaohua Liu, Jiaxin Gao, Xinyuan Chu, Risheng Liu, Xin Fan

**Abstract**: In light of the significant progress made in the development and application of semantic segmentation tasks, there has been increasing attention towards improving the robustness of segmentation models against natural degradation factors (e.g., rain streaks) or artificially attack factors (e.g., adversarial attack). Whereas, most existing methods are designed to address a single degradation factor and are tailored to specific application scenarios. In this work, we present the first attempt to improve the robustness of semantic segmentation tasks by simultaneously handling different types of degradation factors. Specifically, we introduce the Preprocessing Enhanced Adversarial Robust Learning (PEARL) framework based on the analysis of our proposed Naive Adversarial Training (NAT) framework. Our approach effectively handles both rain streaks and adversarial perturbation by transferring the robustness of the segmentation model to the image derain model. Furthermore, as opposed to the commonly used Negative Adversarial Attack (NAA), we design the Auxiliary Mirror Attack (AMA) to introduce positive information prior to the training of the PEARL framework, which improves defense capability and segmentation performance. Our extensive experiments and ablation studies based on different derain methods and segmentation models have demonstrated the significant performance improvement of PEARL with AMA in defense against various adversarial attacks and rain streaks while maintaining high generalization performance across different datasets.

摘要: 鉴于在语义分割任务的开发和应用方面取得的重大进展，人们越来越关注提高分割模型对自然退化因素(例如，雨带)或人为攻击因素(例如，对抗性攻击)的稳健性。然而，大多数现有的方法都是为解决单一退化因素而设计的，并且是针对特定的应用场景量身定做的。在这项工作中，我们首次尝试通过同时处理不同类型的退化因素来提高语义分割任务的稳健性。具体地说，我们在分析了我们提出的朴素对抗性训练(NAT)框架的基础上，引入了预处理增强的对抗性稳健学习(PEAR)框架。通过将分割模型的稳健性转移到图像的DERAIN模型，我们的方法有效地处理了雨滴和对抗性扰动。此外，与常用的消极对抗攻击(NAA)不同，我们设计了辅助镜像攻击(AMA)，在PEARL框架的训练之前引入积极信息，从而提高了防御能力和分割性能。我们基于不同DERAIN方法和分割模型的大量实验和烧蚀研究表明，在保持对不同数据集的高泛化性能的同时，使用AMA的PEARE算法在防御各种对手攻击和雨带方面的性能有了显著的提高。



## **34. Rethink Diversity in Deep Learning Testing**

深度学习测试中的多样性再思考 cs.SE

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.15698v1) [paper-pdf](http://arxiv.org/pdf/2305.15698v1)

**Authors**: Zi Wang, Jihye Choi, Somesh Jha

**Abstract**: Deep neural networks (DNNs) have demonstrated extraordinary capabilities and are an integral part of modern software systems. However, they also suffer from various vulnerabilities such as adversarial attacks and unfairness. Testing deep learning (DL) systems is therefore an important task, to detect and mitigate those vulnerabilities. Motivated by the success of traditional software testing, which often employs diversity heuristics, various diversity measures on DNNs have been proposed to help efficiently expose the buggy behavior of DNNs. In this work, we argue that many DNN testing tasks should be treated as directed testing problems rather than general-purpose testing tasks, because these tasks are specific and well-defined. Hence, the diversity-based approach is less effective.   Following our argument based on the semantics of DNNs and the testing goal, we derive $6$ metrics that can be used for DNN testing and carefully analyze their application scopes. We empirically show their efficacy in exposing bugs in DNNs compared to recent diversity-based metrics. Moreover, we also notice discrepancies between the practices of the software engineering (SE) community and the DL community. We point out some of these gaps, and hopefully, this can lead to bridging the SE practice and DL findings.

摘要: 深度神经网络(DNN)已经显示出非凡的能力，是现代软件系统不可或缺的一部分。然而，它们也存在各种脆弱性，如对抗性攻击和不公平。因此，测试深度学习(DL)系统是检测和缓解这些漏洞的一项重要任务。传统的软件测试通常采用多样性启发式方法，受此启发，人们提出了各种针对DNN的多样性措施，以帮助有效地暴露DNN的错误行为。在这项工作中，我们认为许多DNN测试任务应该被视为有指导的测试问题，而不是通用的测试任务，因为这些任务是特定的和定义良好的。因此，基于多样性的方法不太有效。根据DNN的语义和测试目标，我们推导出可用于DNN测试的$6$度量，并仔细分析了它们的应用范围。与最近的基于多样性的度量相比，我们经验地展示了它们在暴露DNN中的错误方面的有效性。此外，我们还注意到软件工程(SE)社区和DL社区的实践之间的差异。我们指出了其中的一些差距，希望这能导致SE实践和DL发现之间的桥梁。



## **35. AdvFunMatch: When Consistent Teaching Meets Adversarial Robustness**

AdvFunMatch：当一致的教学遇到对手的健壮性 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2305.14700v2) [paper-pdf](http://arxiv.org/pdf/2305.14700v2)

**Authors**: Zihui Wu, Haichang Gao, Bingqian Zhou, Ping Wang

**Abstract**: \emph{Consistent teaching} is an effective paradigm for implementing knowledge distillation (KD), where both student and teacher models receive identical inputs, and KD is treated as a function matching task (FunMatch). However, one limitation of FunMatch is that it does not account for the transfer of adversarial robustness, a model's resistance to adversarial attacks. To tackle this problem, we propose a simple but effective strategy called Adversarial Function Matching (AdvFunMatch), which aims to match distributions for all data points within the $\ell_p$-norm ball of the training data, in accordance with consistent teaching. Formulated as a min-max optimization problem, AdvFunMatch identifies the worst-case instances that maximizes the KL-divergence between teacher and student model outputs, which we refer to as "mismatched examples," and then matches the outputs on these mismatched examples. Our experimental results show that AdvFunMatch effectively produces student models with both high clean accuracy and robustness. Furthermore, we reveal that strong data augmentations (\emph{e.g.}, AutoAugment) are beneficial in AdvFunMatch, whereas prior works have found them less effective in adversarial training. Code is available at \url{https://gitee.com/zihui998/adv-fun-match}.

摘要: 一致性教学是实现知识提炼的一种有效范式，其中学生模型和教师模型接受相同的输入，而一致性教学被视为一项功能匹配任务。然而，FunMatch的一个局限性是它没有考虑到对抗健壮性的转移，即模型对对抗攻击的抵抗力。为了解决这一问题，我们提出了一种简单而有效的策略，称为对抗函数匹配(AdvFunMatch)，该策略旨在根据一致的教学匹配训练数据的$\ell_p$-范数球内所有数据点的分布。AdvFunMatch被描述为一个最小-最大优化问题，它识别最大化教师和学生模型输出之间的KL-分歧的最坏情况实例，我们将其称为“不匹配示例”，然后将输出与这些不匹配示例进行匹配。我们的实验结果表明，AdvFunMatch有效地生成了具有较高清洁准确率和鲁棒性的学生模型。此外，我们发现强数据扩充(例如，AutoAugment)在AdvFunMatch中是有益的，而先前的研究发现它们在对抗性训练中效果较差。代码可在\url{https://gitee.com/zihui998/adv-fun-match}.上找到



## **36. Near Optimal Adversarial Attack on UCB Bandits**

对UCB土匪的近最优敌意攻击 cs.LG

**SubmitDate**: 2023-05-25    [abs](http://arxiv.org/abs/2008.09312v4) [paper-pdf](http://arxiv.org/pdf/2008.09312v4)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.

摘要: 我研究了一个随机多臂强盗问题，其中报酬受到对抗性腐败的影响。提出了一种新的攻击策略，该策略利用UCB算法操纵学习者拉出一些非最优目标臂$T-o(T)$次，累积代价为$\widehat{O}(\Sqrt{\log T})$，其中$T$是轮数。我还证明了累积攻击成本的第一个下限。下界与最高可达$O(\LOG\LOG T)$因子的上界匹配，表明所提出的攻击策略接近最优。



## **37. How do humans perceive adversarial text? A reality check on the validity and naturalness of word-based adversarial attacks**

人类是如何感知敌意文本的？基于词语的对抗性攻击有效性和自然性的真实性检验 cs.CL

ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15587v1) [paper-pdf](http://arxiv.org/pdf/2305.15587v1)

**Authors**: Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy

**Abstract**: Natural Language Processing (NLP) models based on Machine Learning (ML) are susceptible to adversarial attacks -- malicious algorithms that imperceptibly modify input text to force models into making incorrect predictions. However, evaluations of these attacks ignore the property of imperceptibility or study it under limited settings. This entails that adversarial perturbations would not pass any human quality gate and do not represent real threats to human-checked NLP systems. To bypass this limitation and enable proper assessment (and later, improvement) of NLP model robustness, we have surveyed 378 human participants about the perceptibility of text adversarial examples produced by state-of-the-art methods. Our results underline that existing text attacks are impractical in real-world scenarios where humans are involved. This contrasts with previous smaller-scale human studies, which reported overly optimistic conclusions regarding attack success. Through our work, we hope to position human perceptibility as a first-class success criterion for text attacks, and provide guidance for research to build effective attack algorithms and, in turn, design appropriate defence mechanisms.

摘要: 基于机器学习(ML)的自然语言处理(NLP)模型容易受到敌意攻击--恶意算法潜移默化地修改输入文本，迫使模型做出错误的预测。然而，对这些攻击的评估忽略了不可感知性的属性，或者在有限的设置下研究它。这意味着对抗性扰动不会通过任何人类素质的关口，也不会对人类检查的NLP系统构成真正的威胁。为了绕过这一限制，并使适当的评估(以及后来的改进)自然语言处理模型的稳健性，我们调查了378名人类参与者关于由最先进的方法产生的文本对抗性例子的感知能力。我们的结果强调，现有的文本攻击在涉及人类的真实世界场景中是不切实际的。这与之前规模较小的人体研究形成了鲜明对比，后者报告了关于攻击成功的过于乐观的结论。通过我们的工作，我们希望将人类感知能力定位为文本攻击的一流成功标准，并为构建有效的攻击算法，进而设计适当的防御机制的研究提供指导。



## **38. Non-Asymptotic Lower Bounds For Training Data Reconstruction**

训练数据重构的非渐近下界 cs.LG

Additional experiments and minor bug fixes

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2303.16372v4) [paper-pdf](http://arxiv.org/pdf/2303.16372v4)

**Authors**: Prateeti Mukherjee, Satya Lokam

**Abstract**: Mathematical notions of privacy, such as differential privacy, are often stated as probabilistic guarantees that are difficult to interpret. It is imperative, however, that the implications of data sharing be effectively communicated to the data principal to ensure informed decision-making and offer full transparency with regards to the associated privacy risks. To this end, our work presents a rigorous quantitative evaluation of the protection conferred by private learners by investigating their resilience to training data reconstruction attacks. We accomplish this by deriving non-asymptotic lower bounds on the reconstruction error incurred by any adversary against $(\epsilon, \delta)$ differentially private learners for target samples that belong to any compact metric space. Working with a generalization of differential privacy, termed metric privacy, we remove boundedness assumptions on the input space prevalent in prior work, and prove that our results hold for general locally compact metric spaces. We extend the analysis to cover the high dimensional regime, wherein, the input data dimensionality may be larger than the adversary's query budget, and demonstrate that our bounds are minimax optimal under certain regimes.

摘要: 隐私的数学概念，如差异隐私，通常被声明为难以解释的概率保证。然而，必须将数据共享的影响有效地传达给数据负责人，以确保做出明智的决策，并就相关的隐私风险提供充分的透明度。为此，我们的工作通过调查私人学习者对训练数据重建攻击的弹性来对他们提供的保护进行严格的定量评估。我们通过对属于任何紧致度量空间的目标样本的任何对手对$(\epsilon，\Delta)$差分私人学习者所引起的重构误差的非渐近下界来实现这一点。利用差分度量隐私的推广，我们去掉了以前工作中普遍存在的输入空间的有界性假设，并证明了我们的结果对一般的局部紧度量空间成立。我们将分析扩展到高维体制，其中输入数据的维度可能大于对手的查询预算，并证明了在某些体制下我们的界是极小极大最优的。



## **39. Fast Adversarial CNN-based Perturbation Attack on No-Reference Image- and Video-Quality Metrics**

基于CNN的无参考图像和视频质量指标的快速对抗性扰动攻击 cs.CV

ICLR 2023 TinyPapers

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15544v1) [paper-pdf](http://arxiv.org/pdf/2305.15544v1)

**Authors**: Ekaterina Shumitskaya, Anastasia Antsiferova, Dmitriy Vatolin

**Abstract**: Modern neural-network-based no-reference image- and video-quality metrics exhibit performance as high as full-reference metrics. These metrics are widely used to improve visual quality in computer vision methods and compare video processing methods. However, these metrics are not stable to traditional adversarial attacks, which can cause incorrect results. Our goal is to investigate the boundaries of no-reference metrics applicability, and in this paper, we propose a fast adversarial perturbation attack on no-reference quality metrics. The proposed attack (FACPA) can be exploited as a preprocessing step in real-time video processing and compression algorithms. This research can yield insights to further aid in designing of stable neural-network-based no-reference quality metrics.

摘要: 现代基于神经网络的无参考图像和视频质量指标表现出与全参考指标一样高的性能。这些度量被广泛用于改善计算机视觉方法中的视觉质量和比较视频处理方法。然而，这些指标对传统的对抗性攻击并不稳定，这可能会导致错误的结果。我们的目标是研究无参考度量的适用范围，在本文中，我们提出了一种针对无参考质量度量的快速对抗性扰动攻击。所提出的攻击(FACPA)可以作为实时视频处理和压缩算法的预处理步骤。这项研究可以为设计稳定的基于神经网络的无参考质量度量提供进一步的帮助。



## **40. Robust Classification via a Single Diffusion Model**

基于单扩散模型的稳健分类 cs.CV

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15241v1) [paper-pdf](http://arxiv.org/pdf/2305.15241v1)

**Authors**: Huanran Chen, Yinpeng Dong, Zhengyi Wang, Xiao Yang, Chengqi Duan, Hang Su, Jun Zhu

**Abstract**: Recently, diffusion models have been successfully applied to improving adversarial robustness of image classifiers by purifying the adversarial noises or generating realistic data for adversarial training. However, the diffusion-based purification can be evaded by stronger adaptive attacks while adversarial training does not perform well under unseen threats, exhibiting inevitable limitations of these methods. To better harness the expressive power of diffusion models, in this paper we propose Robust Diffusion Classifier (RDC), a generative classifier that is constructed from a pre-trained diffusion model to be adversarially robust. Our method first maximizes the data likelihood of a given input and then predicts the class probabilities of the optimized input using the conditional likelihood of the diffusion model through Bayes' theorem. Since our method does not require training on particular adversarial attacks, we demonstrate that it is more generalizable to defend against multiple unseen threats. In particular, RDC achieves $73.24\%$ robust accuracy against $\ell_\infty$ norm-bounded perturbations with $\epsilon_\infty=8/255$ on CIFAR-10, surpassing the previous state-of-the-art adversarial training models by $+2.34\%$. The findings highlight the potential of generative classifiers by employing diffusion models for adversarial robustness compared with the commonly studied discriminative classifiers.

摘要: 近年来，扩散模型已被成功地应用于提高图像分类器的对抗性鲁棒性，方法是净化对抗性噪声或生成用于对抗性训练的真实数据。然而，基于扩散的净化方法可以通过更强的自适应攻击来规避，而对抗性训练在看不见的威胁下表现不佳，显示出这些方法不可避免的局限性。为了更好地利用扩散模型的表达能力，本文提出了稳健扩散分类器(RDC)，这是一种由预先训练的扩散模型构造的反之稳健的生成式分类器。我们的方法首先最大化给定输入的数据似然，然后通过贝叶斯定理利用扩散模型的条件似然来预测优化输入的类别概率。由于我们的方法不需要在特定的对抗性攻击上进行培训，因此我们证明了它更具一般性，可以防御多个看不见的威胁。特别是，RDC在CIFAR-10上对$epsilon_INFTY=8/255$的范数有界摄动获得了$73.24$的稳健精度，比以前最先进的对抗性训练模型高出$+2.34$。这些发现突出了生成性分类器的潜力，与通常研究的判别性分类器相比，它使用扩散模型来实现对抗稳健性。



## **41. Adaptive Data Analysis in a Balanced Adversarial Model**

均衡对抗性模型中的自适应数据分析 cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15452v1) [paper-pdf](http://arxiv.org/pdf/2305.15452v1)

**Authors**: Kobbi Nissim, Uri Stemmer, Eliad Tsfadia

**Abstract**: In adaptive data analysis, a mechanism gets $n$ i.i.d. samples from an unknown distribution $D$, and is required to provide accurate estimations to a sequence of adaptively chosen statistical queries with respect to $D$. Hardt and Ullman (FOCS 2014) and Steinke and Ullman (COLT 2015) showed that in general, it is computationally hard to answer more than $\Theta(n^2)$ adaptive queries, assuming the existence of one-way functions.   However, these negative results strongly rely on an adversarial model that significantly advantages the adversarial analyst over the mechanism, as the analyst, who chooses the adaptive queries, also chooses the underlying distribution $D$. This imbalance raises questions with respect to the applicability of the obtained hardness results -- an analyst who has complete knowledge of the underlying distribution $D$ would have little need, if at all, to issue statistical queries to a mechanism which only holds a finite number of samples from $D$.   We consider more restricted adversaries, called \emph{balanced}, where each such adversary consists of two separated algorithms: The \emph{sampler} who is the entity that chooses the distribution and provides the samples to the mechanism, and the \emph{analyst} who chooses the adaptive queries, but does not have a prior knowledge of the underlying distribution. We improve the quality of previous lower bounds by revisiting them using an efficient \emph{balanced} adversary, under standard public-key cryptography assumptions. We show that these stronger hardness assumptions are unavoidable in the sense that any computationally bounded \emph{balanced} adversary that has the structure of all known attacks, implies the existence of public-key cryptography.

摘要: 在适应性数据分析中，一个机制得到$n$I.I.D.来自未知分布$D$的样本，并且需要对关于$D$的适应性选择的统计查询序列提供准确的估计。Hardt和Ullman(FOCS 2014)和Steinke和Ullman(COLT 2015)表明，假设存在单向函数，通常很难回答超过$\theta(n^2)$自适应查询。然而，这些负面结果强烈依赖于对抗性模型，该模型显著地使对抗性分析师相对于该机制具有优势，因为选择自适应查询的分析师也选择基础分布$D$。这种不平衡对所获得的硬度结果的适用性提出了问题--完全了解基本分布$D$的分析员将几乎不需要向一个仅保存来自$D$的有限数量样本的机制发出统计查询。我们考虑更受限制的对手，称为\emph{平衡}，其中每个这样的对手由两个独立的算法组成：\emph{Sampler}是选择分布并将样本提供给机制的实体，以及\emph{Analyst}选择自适应查询，但不事先知道底层分布。在标准的公钥密码学假设下，我们通过使用一个有效的、平衡的对手来重新访问以前的下界，从而提高了它们的质量。我们证明了这些更强的难度假设是不可避免的，因为任何具有所有已知攻击结构的计算有界的对手都意味着公钥密码学的存在。



## **42. Relating Implicit Bias and Adversarial Attacks through Intrinsic Dimension**

内隐偏见与对抗性攻击的内在维度关联 cs.LG

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15203v1) [paper-pdf](http://arxiv.org/pdf/2305.15203v1)

**Authors**: Lorenzo Basile, Nikos Karantzas, Alberto D'Onofrio, Luca Bortolussi, Alex Rodriguez, Fabio Anselmi

**Abstract**: Despite their impressive performance in classification, neural networks are known to be vulnerable to adversarial attacks. These attacks are small perturbations of the input data designed to fool the model. Naturally, a question arises regarding the potential connection between the architecture, settings, or properties of the model and the nature of the attack. In this work, we aim to shed light on this problem by focusing on the implicit bias of the neural network, which refers to its inherent inclination to favor specific patterns or outcomes. Specifically, we investigate one aspect of the implicit bias, which involves the essential Fourier frequencies required for accurate image classification. We conduct tests to assess the statistical relationship between these frequencies and those necessary for a successful attack. To delve into this relationship, we propose a new method that can uncover non-linear correlations between sets of coordinates, which, in our case, are the aforementioned frequencies. By exploiting the entanglement between intrinsic dimension and correlation, we provide empirical evidence that the network bias in Fourier space and the target frequencies of adversarial attacks are closely tied.

摘要: 尽管神经网络在分类方面的表现令人印象深刻，但众所周知，它很容易受到对手的攻击。这些攻击是对输入数据的微小干扰，旨在愚弄模型。自然，模型的体系结构、设置或属性与攻击性质之间的潜在联系就会出现问题。在这项工作中，我们的目标是通过关注神经网络的隐含偏差来阐明这个问题，隐含偏差指的是它固有的偏爱特定模式或结果的倾向。具体地说，我们研究了隐式偏差的一个方面，它涉及准确图像分类所需的基本傅立叶频率。我们进行测试，以评估这些频率与成功攻击所必需的频率之间的统计关系。为了深入研究这种关系，我们提出了一种新的方法，可以发现坐标集合之间的非线性关联，在我们的例子中，这些集合就是前面提到的频率。通过利用内在维度和相关性之间的纠缠，我们提供了经验证据，证明了傅立叶空间中的网络偏差与对抗性攻击的目标频率密切相关。



## **43. IoT Threat Detection Testbed Using Generative Adversarial Networks**

基于产生式对抗网络的物联网威胁检测测试平台 cs.CR

8 pages, 5 figures

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15191v1) [paper-pdf](http://arxiv.org/pdf/2305.15191v1)

**Authors**: Farooq Shaikh, Elias Bou-Harb, Aldin Vehabovic, Jorge Crichigno, Aysegul Yayimli, Nasir Ghani

**Abstract**: The Internet of Things(IoT) paradigm provides persistent sensing and data collection capabilities and is becoming increasingly prevalent across many market sectors. However, most IoT devices emphasize usability and function over security, making them very vulnerable to malicious exploits. This concern is evidenced by the increased use of compromised IoT devices in large scale bot networks (botnets) to launch distributed denial of service(DDoS) attacks against high value targets. Unsecured IoT systems can also provide entry points to private networks, allowing adversaries relatively easy access to valuable resources and services. Indeed, these evolving IoT threat vectors (ranging from brute force attacks to remote code execution exploits) are posing key challenges. Moreover, many traditional security mechanisms are not amenable for deployment on smaller resource-constrained IoT platforms. As a result, researchers have been developing a range of methods for IoT security, with many strategies using advanced machine learning(ML) techniques. Along these lines, this paper presents a novel generative adversarial network(GAN) solution to detect threats from malicious IoT devices both inside and outside a network. This model is trained using both benign IoT traffic and global darknet data and further evaluated in a testbed with real IoT devices and malware threats.

摘要: 物联网(IoT)模式提供持久的传感和数据收集能力，并在许多市场领域变得越来越普遍。然而，大多数物联网设备强调可用性和功能，而不是安全性，这使得它们非常容易受到恶意攻击。大规模僵尸网络(僵尸网络)中越来越多地使用受攻击的物联网设备来对高价值目标发动分布式拒绝服务(DDoS)攻击，这证明了这一担忧。不安全的物联网系统还可以提供专用网络的入口点，使对手能够相对轻松地访问有价值的资源和服务。事实上，这些不断演变的物联网威胁向量(从暴力攻击到远程代码执行漏洞)构成了关键挑战。此外，许多传统的安全机制不适合在较小的资源受限的物联网平台上部署。因此，研究人员一直在开发一系列物联网安全方法，其中许多策略使用高级机器学习(ML)技术。为此，本文提出了一种新的生成性对抗网络(GAN)解决方案，用于检测来自网络内外恶意物联网设备的威胁。该模型使用良性物联网流量和全球暗网数据进行训练，并在试验台上使用真实物联网设备和恶意软件威胁进行进一步评估。



## **44. Another Dead End for Morphological Tags? Perturbed Inputs and Parsing**

形态标签的又一个死胡同？受干扰的输入和解析 cs.CL

Accepted at Findings of ACL 2023

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.15119v1) [paper-pdf](http://arxiv.org/pdf/2305.15119v1)

**Authors**: Alberto Muñoz-Ortiz, David Vilares

**Abstract**: The usefulness of part-of-speech tags for parsing has been heavily questioned due to the success of word-contextualized parsers. Yet, most studies are limited to coarse-grained tags and high quality written content; while we know little about their influence when it comes to models in production that face lexical errors. We expand these setups and design an adversarial attack to verify if the use of morphological information by parsers: (i) contributes to error propagation or (ii) if on the other hand it can play a role to correct mistakes that word-only neural parsers make. The results on 14 diverse UD treebanks show that under such attacks, for transition- and graph-based models their use contributes to degrade the performance even faster, while for the (lower-performing) sequence labeling parsers they are helpful. We also show that if morphological tags were utopically robust against lexical perturbations, they would be able to correct parsing mistakes.

摘要: 由于单词上下文解析器的成功，词性标签对语法分析的有用性受到了严重质疑。然而，大多数研究仅限于粗粒度标签和高质量的书面内容；而当涉及到生产中面临词汇错误的模型时，我们对它们的影响知之甚少。我们扩展了这些设置，并设计了一个对抗性攻击来验证解析器对形态信息的使用：(I)有助于错误传播，或者(Ii)另一方面，它可以起到纠正只使用单词的神经解析器所犯错误的作用。在14个不同的UD树库上的结果表明，在这种攻击下，对于基于转换和基于图的模型，它们的使用有助于更快地降低性能，而对于(性能较差的)序列标记解析器，它们是有帮助的。我们还表明，如果形态标签对词汇扰动具有超乎寻常的健壮性，它们将能够纠正语法分析错误。



## **45. Adversarial Demonstration Attacks on Large Language Models**

针对大型语言模型的对抗性演示攻击 cs.CL

Work in Progress

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14950v1) [paper-pdf](http://arxiv.org/pdf/2305.14950v1)

**Authors**: Jiongxiao Wang, Zichen Liu, Keun Hee Park, Muhao Chen, Chaowei Xiao

**Abstract**: With the emergence of more powerful large language models (LLMs), such as ChatGPT and GPT-4, in-context learning (ICL) has gained significant prominence in leveraging these models for specific tasks by utilizing data-label pairs as precondition prompts. While incorporating demonstrations can greatly enhance the performance of LLMs across various tasks, it may introduce a new security concern: attackers can manipulate only the demonstrations without changing the input to perform an attack. In this paper, we investigate the security concern of ICL from an adversarial perspective, focusing on the impact of demonstrations. We propose an ICL attack based on TextAttack, which aims to only manipulate the demonstration without changing the input to mislead the models. Our results demonstrate that as the number of demonstrations increases, the robustness of in-context learning would decreases. Furthermore, we also observe that adversarially attacked demonstrations exhibit transferability to diverse input examples. These findings emphasize the critical security risks associated with ICL and underscore the necessity for extensive research on the robustness of ICL, particularly given its increasing significance in the advancement of LLMs.

摘要: 随着更强大的大型语言模型(LLM)的出现，如ChatGPT和GPT-4，情境学习(ICL)通过将数据-标签对作为前提提示来利用这些模型来执行特定任务，从而获得了显著的突出地位。虽然合并演示可以极大地提高LLMS在各种任务中的性能，但它可能会引入一个新的安全问题：攻击者只能操作演示，而不会更改输入来执行攻击。在本文中，我们从对抗的角度研究了ICL的安全问题，重点关注了示威活动的影响。我们提出了一种基于TextAttack的ICL攻击，其目的是只操纵演示，而不改变输入来误导模型。我们的结果表明，随着演示数量的增加，情境学习的稳健性会降低。此外，我们还观察到，被敌意攻击的演示表现出对不同输入示例的迁移。这些研究结果强调了与大规模杀伤性武器相关的重大安全风险，并强调了对大规模杀伤性武器的稳健性进行广泛研究的必要性，特别是考虑到其在推进LLMS方面日益重要。



## **46. Madvex: Instrumentation-based Adversarial Attacks on Machine Learning Malware Detection**

MAdvex：机器学习恶意软件检测中基于工具的敌意攻击 cs.CR

20 pages. To be published in The 20th Conference on Detection of  Intrusions and Malware & Vulnerability Assessment (DIMVA 2023)

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.02559v2) [paper-pdf](http://arxiv.org/pdf/2305.02559v2)

**Authors**: Nils Loose, Felix Mächtle, Claudius Pott, Volodymyr Bezsmertnyi, Thomas Eisenbarth

**Abstract**: WebAssembly (Wasm) is a low-level binary format for web applications, which has found widespread adoption due to its improved performance and compatibility with existing software. However, the popularity of Wasm has also led to its exploitation for malicious purposes, such as cryptojacking, where malicious actors use a victim's computing resources to mine cryptocurrencies without their consent. To counteract this threat, machine learning-based detection methods aiming to identify cryptojacking activities within Wasm code have emerged. It is well-known that neural networks are susceptible to adversarial attacks, where inputs to a classifier are perturbed with minimal changes that result in a crass misclassification. While applying changes in image classification is easy, manipulating binaries in an automated fashion to evade malware classification without changing functionality is non-trivial. In this work, we propose a new approach to include adversarial examples in the code section of binaries via instrumentation. The introduced gadgets allow for the inclusion of arbitrary bytes, enabling efficient adversarial attacks that reliably bypass state-of-the-art machine learning classifiers such as the CNN-based Minos recently proposed at NDSS 2021. We analyze the cost and reliability of instrumentation-based adversarial example generation and show that the approach works reliably at minimal size and performance overheads.

摘要: WebAssembly(WASM)是一种用于Web应用程序的低级二进制格式，由于其改进的性能和与现有软件的兼容性而被广泛采用。然而，WASM的流行也导致了对其进行恶意攻击，例如加密劫持，即恶意行为者在未经受害者同意的情况下使用受害者的计算资源来挖掘加密货币。为了应对这种威胁，出现了基于机器学习的检测方法，旨在识别WASM代码中的加密劫持活动。众所周知，神经网络容易受到敌意攻击，在这种攻击中，分类器的输入会受到干扰，只需进行极小的更改，就会导致粗略的错误分类。虽然在图像分类中应用更改很容易，但在不更改功能的情况下以自动方式操作二进制文件来规避恶意软件分类并不是一件容易的事情。在这项工作中，我们提出了一种新的方法，通过插装将对抗性示例包括在二进制文件的代码部分中。引入的小工具允许包含任意字节，从而实现了高效的对抗性攻击，可靠地绕过了最先进的机器学习分类器，例如最近在NDSS 2021上提出的基于CNN的Minos。我们分析了基于插桩的对抗性实例生成的代价和可靠性，结果表明该方法在最小规模和最小性能开销的情况下能够可靠地工作。



## **47. Introducing Competition to Boost the Transferability of Targeted Adversarial Examples through Clean Feature Mixup**

引入竞争，通过干净的特征混合提高目标对抗性例子的可转移性 cs.CV

CVPR 2023 camera-ready

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14846v1) [paper-pdf](http://arxiv.org/pdf/2305.14846v1)

**Authors**: Junyoung Byun, Myung-Joon Kwon, Seungju Cho, Yoonji Kim, Changick Kim

**Abstract**: Deep neural networks are widely known to be susceptible to adversarial examples, which can cause incorrect predictions through subtle input modifications. These adversarial examples tend to be transferable between models, but targeted attacks still have lower attack success rates due to significant variations in decision boundaries. To enhance the transferability of targeted adversarial examples, we propose introducing competition into the optimization process. Our idea is to craft adversarial perturbations in the presence of two new types of competitor noises: adversarial perturbations towards different target classes and friendly perturbations towards the correct class. With these competitors, even if an adversarial example deceives a network to extract specific features leading to the target class, this disturbance can be suppressed by other competitors. Therefore, within this competition, adversarial examples should take different attack strategies by leveraging more diverse features to overwhelm their interference, leading to improving their transferability to different models. Considering the computational complexity, we efficiently simulate various interference from these two types of competitors in feature space by randomly mixing up stored clean features in the model inference and named this method Clean Feature Mixup (CFM). Our extensive experimental results on the ImageNet-Compatible and CIFAR-10 datasets show that the proposed method outperforms the existing baselines with a clear margin. Our code is available at https://github.com/dreamflake/CFM.

摘要: 众所周知，深度神经网络容易受到对抗性例子的影响，这可能会通过微妙的输入修改导致错误的预测。这些对抗性的例子往往可以在模型之间转换，但由于决策边界的显著差异，定向攻击的攻击成功率仍然较低。为了提高目标对抗性实例的可转移性，我们建议在优化过程中引入竞争。我们的想法是在两种新的竞争对手噪声存在的情况下制造对抗性扰动：针对不同目标类别的对抗性扰动和针对正确类别的友好扰动。对于这些竞争对手，即使敌对的例子欺骗网络以提取导致目标类的特定特征，这种干扰也可以被其他竞争对手抑制。因此，在这场比赛中，对抗性榜样应该采取不同的攻击策略，利用更多样化的特征来压倒他们的干扰，从而提高他们对不同模型的可转移性。考虑到计算的复杂性，我们通过在模型推理中随机混合存储的干净特征来有效地模拟这两种竞争对手在特征空间中的各种干扰，将该方法命名为清洁特征混合(CFM)方法。我们在ImageNet兼容数据集和CIFAR-10数据集上的大量实验结果表明，该方法的性能明显优于现有的基线方法。我们的代码可以在https://github.com/dreamflake/CFM.上找到



## **48. Block Coordinate Descent on Smooth Manifolds**

光滑流形上的块坐标下降 math.OC

**SubmitDate**: 2023-05-24    [abs](http://arxiv.org/abs/2305.14744v1) [paper-pdf](http://arxiv.org/pdf/2305.14744v1)

**Authors**: Liangzu Peng, René Vidal

**Abstract**: Block coordinate descent is an optimization paradigm that iteratively updates one block of variables at a time, making it quite amenable to big data applications due to its scalability and performance. Its convergence behavior has been extensively studied in the (block-wise) convex case, but it is much less explored in the non-convex case. In this paper we analyze the convergence of block coordinate methods on non-convex sets and derive convergence rates on smooth manifolds under natural or weaker assumptions than prior work. Our analysis applies to many non-convex problems (e.g., generalized PCA, optimal transport, matrix factorization, Burer-Monteiro factorization, outlier-robust estimation, alternating projection, maximal coding rate reduction, neural collapse, adversarial attacks, homomorphic sensing), either yielding novel corollaries or recovering previously known results.

摘要: 块坐标下降是一种每次迭代更新一个变量块的优化范例，由于其可扩展性和性能，使其非常适合大数据应用。它的收敛行为在(分块)凸的情况下已经被广泛地研究，但在非凸的情况下的研究要少得多。本文分析了块坐标方法在非凸集上的收敛，并在自然或弱于已有工作的假设下，得到了光滑流形上的收敛速度。我们的分析适用于许多非凸问题(例如，广义主成分分析、最优传输、矩阵因式分解、布里-蒙泰罗因式分解、离群点稳健估计、交替投影、最大码率降低、神经崩溃、敌意攻击、同态检测)，要么产生新的推论，要么恢复已有的已知结果。



## **49. Adversarial Machine Learning and Cybersecurity: Risks, Challenges, and Legal Implications**

对抗性机器学习和网络安全：风险、挑战和法律含义 cs.CR

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2305.14553v1) [paper-pdf](http://arxiv.org/pdf/2305.14553v1)

**Authors**: Micah Musser, Andrew Lohn, James X. Dempsey, Jonathan Spring, Ram Shankar Siva Kumar, Brenda Leong, Christina Liaghati, Cindy Martinez, Crystal D. Grant, Daniel Rohrer, Heather Frase, Jonathan Elliott, John Bansemer, Mikel Rodriguez, Mitt Regan, Rumman Chowdhury, Stefan Hermanek

**Abstract**: In July 2022, the Center for Security and Emerging Technology (CSET) at Georgetown University and the Program on Geopolitics, Technology, and Governance at the Stanford Cyber Policy Center convened a workshop of experts to examine the relationship between vulnerabilities in artificial intelligence systems and more traditional types of software vulnerabilities. Topics discussed included the extent to which AI vulnerabilities can be handled under standard cybersecurity processes, the barriers currently preventing the accurate sharing of information about AI vulnerabilities, legal issues associated with adversarial attacks on AI systems, and potential areas where government support could improve AI vulnerability management and mitigation.   This report is meant to accomplish two things. First, it provides a high-level discussion of AI vulnerabilities, including the ways in which they are disanalogous to other types of vulnerabilities, and the current state of affairs regarding information sharing and legal oversight of AI vulnerabilities. Second, it attempts to articulate broad recommendations as endorsed by the majority of participants at the workshop.

摘要: 2022年7月，乔治城大学安全与新兴技术中心(CSET)和斯坦福网络政策中心的地缘政治、技术和治理项目召开了一次专家研讨会，研究人工智能系统中的漏洞与更传统类型的软件漏洞之间的关系。讨论的主题包括在标准网络安全流程下可以在多大程度上处理人工智能漏洞，目前阻碍准确共享人工智能漏洞信息的障碍，与对人工智能系统的对抗性攻击相关的法律问题，以及政府支持可以改善人工智能漏洞管理和缓解的潜在领域。这份报告意在完成两件事。首先，它提供了对人工智能漏洞的高级别讨论，包括它们与其他类型的漏洞的不同之处，以及有关信息共享和对人工智能漏洞的法律监督的现状。第二，它试图阐明研讨会上大多数与会者赞同的广泛建议。



## **50. Translate your gibberish: black-box adversarial attack on machine translation systems**

翻译你的胡言乱语：对机器翻译系统的黑箱对抗性攻击 cs.CL

**SubmitDate**: 2023-05-23    [abs](http://arxiv.org/abs/2303.10974v2) [paper-pdf](http://arxiv.org/pdf/2303.10974v2)

**Authors**: Andrei Chertkov, Olga Tsymboi, Mikhail Pautov, Ivan Oseledets

**Abstract**: Neural networks are deployed widely in natural language processing tasks on the industrial scale, and perhaps the most often they are used as compounds of automatic machine translation systems. In this work, we present a simple approach to fool state-of-the-art machine translation tools in the task of translation from Russian to English and vice versa. Using a novel black-box gradient-free tensor-based optimizer, we show that many online translation tools, such as Google, DeepL, and Yandex, may both produce wrong or offensive translations for nonsensical adversarial input queries and refuse to translate seemingly benign input phrases. This vulnerability may interfere with understanding a new language and simply worsen the user's experience while using machine translation systems, and, hence, additional improvements of these tools are required to establish better translation.

摘要: 神经网络在工业规模的自然语言处理任务中被广泛部署，也许最常被用作自动机器翻译系统的复合体。在这项工作中，我们提出了一种简单的方法，在从俄语到英语的翻译任务中愚弄最先进的机器翻译工具，反之亦然。使用一种新的黑盒无梯度张量优化器，我们证明了许多在线翻译工具，如Google，DeepL和Yandex，都可能对无意义的对抗性输入查询产生错误或攻击性的翻译，并拒绝翻译看似良性的输入短语。此漏洞可能会干扰对新语言的理解，并只会恶化用户在使用机器翻译系统时的体验，因此，需要对这些工具进行额外的改进才能建立更好的翻译。



